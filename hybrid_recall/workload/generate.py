"""Deterministic, seeded generator for the agentic-memory hybrid-recall workload.

Design goals:
  * Reproducible across machines (seeded NumPy; synthetic embeddings by default).
  * The three recall signals (vector, BM25, graph) are *correlated but distinct* — the
    regime where fusion actually changes the answer. We achieve this by organizing
    entities, chunks, edges, and query topics around shared latent "clusters":
      - embeddings: cluster center + noise        (vector signal ~ cluster)
      - chunk text: sampled from cluster topic words (BM25 signal ~ cluster)
      - edges: mostly intra-cluster, some cross    (graph proximity ~ cluster, loosely)
    Because graph reachability is not the same set as "same cluster", and ANN/BM25 are
    approximate, no single signal dominates — fusion wins, as in real agentic memory.
  * A synthetic vocabulary of pseudo-words (no English morphology) makes tokenization
    and stemming near-no-ops, so recall differences reflect index fidelity (ANN recall,
    BM25 scoring, graph traversal) rather than tokenizer disagreements.
"""

from __future__ import annotations

import numpy as np

from . import embeddings as emb
from .schema import (
    EDGE_TYPES,
    ENTITY_TYPES,
    Chunk,
    Edge,
    Entity,
    Query,
    Workload,
    WorkloadMeta,
)

_SYLL = [
    "ba", "ke", "mi", "to", "lu", "ra", "ne", "si", "do", "va", "po", "ze",
    "fi", "gu", "ha", "jo", "ki", "le", "mo", "nu", "qa", "ri", "su", "ti",
    "wo", "xe", "yu", "za", "bo", "ce", "di", "fa", "go", "hi", "ja", "ko",
]


def _make_vocab(rng: np.random.Generator, size: int) -> list[str]:
    """`size` unique pseudo-words built from 2-3 syllables (deterministic)."""
    words: set[str] = set()
    out: list[str] = []
    while len(out) < size:
        n_syll = 2 + int(rng.integers(0, 2))
        w = "".join(_SYLL[i] for i in rng.integers(0, len(_SYLL), n_syll))
        if w not in words:
            words.add(w)
            out.append(w)
    return out


def generate(
    name: str,
    *,
    entities: int,
    edges: int,
    chunks: int,
    dim: int,
    queries: int,
    graph_hops: int,
    k: int,
    clusters: int,
    seed: int = 1234,
    pool_n: int | None = None,
    rrf_k: float = 60.0,
    real_embeddings: bool = False,
    topic_size: int = 200,
    chunk_words: int = 10,
    query_words: int = 4,
    summary_words: int = 6,
) -> Workload:
    rng = np.random.default_rng(seed)
    pool_n = pool_n if pool_n is not None else max(100, 10 * k)

    # --- vocabulary & per-cluster topics --------------------------------
    # A large vocabulary with fairly distinct per-cluster topic sets makes term incidence
    # sparse, so a query's few terms select a clear subset of chunks. Any competent BM25
    # then agrees closely with the oracle — the recall differences come from *capability*
    # (graph vs no graph) and ANN recall, not from BM25 scoring-formula trivia.
    vocab_size = max(2000, clusters * topic_size)
    vocab = _make_vocab(rng, vocab_size)
    # Each cluster owns `topic_size` topic words (overlap allowed across clusters).
    cluster_topics = [
        rng.choice(vocab_size, size=topic_size, replace=False) for _ in range(clusters)
    ]
    filler = rng.choice(vocab_size, size=20, replace=False)  # shared connective words

    def topic_text(cluster: int, n_words: int) -> tuple[str, np.ndarray]:
        """Return (text, token-id array). Embeddings are derived from these same tokens,
        so vector similarity and BM25 agree on lexically-related items."""
        topics = cluster_topics[cluster]
        idx = rng.choice(topics, size=min(n_words, topic_size), replace=False)
        # sprinkle a couple of filler words so corpora aren't perfectly separable
        ids = np.concatenate([idx, rng.choice(filler, size=2, replace=False)])
        rng.shuffle(ids)
        return " ".join(vocab[i] for i in ids), ids

    centers = emb.make_cluster_centers(rng, clusters, dim)
    word_vectors = emb.make_word_vectors(rng, vocab_size, dim)
    ent_tokens: list[np.ndarray] = []
    chunk_tokens: list[np.ndarray] = []
    query_tokens: list[np.ndarray] = []

    # --- entities --------------------------------------------------------
    ent_cluster = rng.integers(0, clusters, entities).astype(np.int64)
    entity_list: list[Entity] = []
    for i in range(entities):
        c = int(ent_cluster[i])
        etype = ENTITY_TYPES[i % len(ENTITY_TYPES)]
        label = f"{vocab[cluster_topics[c][i % topic_size]].capitalize()}-{i}"
        summary, toks = topic_text(c, summary_words)
        ent_tokens.append(toks)
        entity_list.append(
            Entity(id=f"e{i}", etype=etype, label=label, summary=summary, cluster=c, vec_idx=i)
        )

    # --- edges (mostly intra-cluster, some cross-cluster) ---------------
    # Group entity indices by cluster for fast intra-cluster sampling.
    by_cluster: list[np.ndarray] = [np.where(ent_cluster == c)[0] for c in range(clusters)]
    p_intra = 0.7
    edge_list: list[Edge] = []
    src_arr = rng.integers(0, entities, edges)
    intra_mask = rng.random(edges) < p_intra
    rel_arr = rng.integers(0, len(EDGE_TYPES), edges)
    for j in range(edges):
        s = int(src_arr[j])
        if intra_mask[j]:
            pool = by_cluster[int(ent_cluster[s])]
            d = int(pool[rng.integers(0, len(pool))]) if len(pool) else int(rng.integers(0, entities))
        else:
            d = int(rng.integers(0, entities))
        if d == s:
            d = (d + 1) % entities
        edge_list.append(Edge(src=f"e{s}", dst=f"e{d}", rel=EDGE_TYPES[int(rel_arr[j])]))

    # --- chunks (the recall corpus) -------------------------------------
    chunk_cluster = rng.integers(0, clusters, chunks).astype(np.int64)
    chunk_list: list[Chunk] = []
    for i in range(chunks):
        c = int(chunk_cluster[i])
        # link to 1-3 entities, preferentially same-cluster
        n_links = 1 + int(rng.integers(0, 3))
        links: list[str] = []
        for _ in range(n_links):
            if rng.random() < 0.8 and len(by_cluster[c]):
                e = int(by_cluster[c][rng.integers(0, len(by_cluster[c]))])
            else:
                e = int(rng.integers(0, entities))
            links.append(f"e{e}")
        text, toks = topic_text(c, chunk_words)
        chunk_tokens.append(toks)
        chunk_list.append(
            Chunk(id=f"c{i}", text=text, entity_ids=sorted(set(links)), cluster=c, vec_idx=i)
        )

    # --- queries ---------------------------------------------------------
    query_list: list[Query] = []
    q_seed_idx = rng.integers(0, entities, queries)
    q_cluster = ent_cluster[q_seed_idx]
    for i in range(queries):
        c = int(q_cluster[i])
        text, toks = topic_text(c, query_words)
        query_tokens.append(toks)
        query_list.append(
            Query(id=f"q{i}", text=text, seed_entity=f"e{int(q_seed_idx[i])}", cluster=c, vec_idx=i)
        )

    # --- embeddings ------------------------------------------------------
    if real_embeddings:
        entity_emb = emb.real_embeddings([e.summary for e in entity_list])
        chunk_emb = emb.real_embeddings([c.text for c in chunk_list])
        query_emb = emb.real_embeddings([q.text for q in query_list])
        dim = chunk_emb.shape[1]
    else:
        entity_emb = emb.text_derived_embeddings(ent_tokens, ent_cluster, word_vectors, centers)
        chunk_emb = emb.text_derived_embeddings(chunk_tokens, chunk_cluster, word_vectors, centers)
        query_emb = emb.text_derived_embeddings(
            query_tokens, np.asarray(q_cluster), word_vectors, centers
        )

    meta = WorkloadMeta(
        name=name,
        seed=seed,
        dim=dim,
        k=k,
        pool_n=pool_n,
        graph_hops=graph_hops,
        rrf_k=rrf_k,
        clusters=clusters,
        n_entities=entities,
        n_edges=edges,
        n_chunks=chunks,
        n_queries=queries,
        real_embeddings=real_embeddings,
    )
    return Workload(
        meta=meta,
        entities=entity_list,
        edges=edge_list,
        chunks=chunk_list,
        queries=query_list,
        entity_emb=entity_emb,
        chunk_emb=chunk_emb,
        query_emb=query_emb,
    )
