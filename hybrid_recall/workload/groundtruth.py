"""Exact ground truth for the hybrid-recall task.

For each query we compute three *exact* component rankings over the chunk corpus and fuse
them with the canonical RRF (see hybrid_recall.fusion):

  1. vector  : exact cosine similarity (full scan, no ANN approximation)
  2. bm25    : exact Okapi BM25 (k1=1.5, b=0.75) over whitespace-tokenized chunk text
  3. graph   : k-hop BFS proximity from the query's seed entity, chunks ranked by the
               minimum hop distance of any linking entity (ties broken by chunk id)

Each component is truncated to `pool_n` candidates *before* fusion — the same candidate
depth engines are asked to fetch — so recall@k is achievable and fair. The fused top-k is
the ground-truth answer set; recall@k = |engine_topk ∩ truth| / k.

This module is pure NumPy/Python and is the single source of truth the engines are scored
against. It is intentionally simple and slow-but-correct.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np

from ..fusion import rrf_fuse
from .schema import Workload

BM25_K1 = 1.5
BM25_B = 0.75


class _BM25Index:
    """Minimal exact Okapi BM25 over the chunk corpus."""

    def __init__(self, docs: list[str]):
        self.n = len(docs)
        self.doc_tokens: list[list[str]] = [d.split() for d in docs]
        self.doc_len = np.array([len(t) for t in self.doc_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        # postings: term -> list[(doc_idx, tf)]
        postings: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for i, toks in enumerate(self.doc_tokens):
            for t in toks:
                postings[t][i] += 1
        self.postings = {t: list(d.items()) for t, d in postings.items()}
        self.idf = {
            t: math.log(1.0 + (self.n - len(pl) + 0.5) / (len(pl) + 0.5))
            for t, pl in self.postings.items()
        }

    def top(self, query_terms: list[str], pool_n: int) -> list[int]:
        scores: dict[int, float] = defaultdict(float)
        for t in set(query_terms):
            pl = self.postings.get(t)
            if not pl:
                continue
            idf = self.idf[t]
            for doc, tf in pl:
                denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * self.doc_len[doc] / self.avgdl)
                scores[doc] += idf * (tf * (BM25_K1 + 1.0)) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [doc for doc, _ in ranked[:pool_n]]


def exact_vector_topn(query_vec: np.ndarray, chunk_emb: np.ndarray, pool_n: int) -> list[int]:
    """Exact cosine top-n. Embeddings are L2-normalized so dot == cosine."""
    sims = chunk_emb @ query_vec
    n = min(pool_n, sims.shape[0])
    part = np.argpartition(-sims, n - 1)[:n]
    return list(part[np.argsort(-sims[part])])


def khop_entities(adjacency: dict[str, list[str]], seed: str, hops: int) -> dict[str, int]:
    """BFS: entity id -> minimum hop distance from seed (0..hops)."""
    dist: dict[str, int] = {seed: 0}
    q: deque[str] = deque([seed])
    while q:
        cur = q.popleft()
        d = dist[cur]
        if d >= hops:
            continue
        for nb in adjacency.get(cur, ()):  # noqa: SIM118
            if nb not in dist:
                dist[nb] = d + 1
                q.append(nb)
    return dist


def graph_proximity_topn(
    wl: Workload, seed: str, hops: int, pool_n: int
) -> list[str]:
    """Chunks within `hops` of seed, ranked by min hop distance (then chunk id)."""
    dist = khop_entities(wl.adjacency, seed, hops)
    cbe = wl.chunks_by_entity
    chunk_hop: dict[str, int] = {}
    for ent, d in dist.items():
        for cid in cbe.get(ent, ()):  # noqa: SIM118
            if d < chunk_hop.get(cid, math.inf):
                chunk_hop[cid] = d
    ranked = sorted(chunk_hop.items(), key=lambda kv: (kv[1], kv[0]))
    return [cid for cid, _ in ranked[:pool_n]]


def compute_ground_truth(wl: Workload) -> dict[str, list[str]]:
    """query id -> ordered top-k chunk ids (the exact fused answer)."""
    wl.ensure_index()
    m = wl.meta
    bm25 = _BM25Index([c.text for c in wl.chunks])
    chunk_id_by_idx = [c.id for c in wl.chunks]
    truth: dict[str, list[str]] = {}
    for q in wl.queries:
        qv = wl.query_emb[q.vec_idx]
        vec_ids = [chunk_id_by_idx[i] for i in exact_vector_topn(qv, wl.chunk_emb, m.pool_n)]
        bm25_ids = [chunk_id_by_idx[i] for i in bm25.top(q.text.split(), m.pool_n)]
        graph_ids = graph_proximity_topn(wl, q.seed_entity, m.graph_hops, m.pool_n)
        truth[q.id] = rrf_fuse([vec_ids, bm25_ids, graph_ids], k=m.k, rrf_k=m.rrf_k)
    return truth
