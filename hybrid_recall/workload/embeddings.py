"""Embedding generation.

Default mode is **synthetic but text-derived**: deterministic, seeded, no model download.
Real embeddings are a *function of the text*, so vector similarity and BM25 should agree on
lexically-related documents. We reproduce that here:

    embedding(item) = L2_normalize( cluster_weight * center[cluster(item)]
                                    + mean( word_vector[token] for token in item.text ) )

where `word_vector` is a fixed random unit vector per vocabulary word and `center` is a
random unit vector per latent cluster. This yields the regime hybrid recall is designed
for: vector ~ BM25 (shared words) are correlated, while graph proximity (shared cluster /
reachability) is correlated-but-distinct — so no single signal dominates and fusion wins.

Optional `real` mode embeds text with sentence-transformers (extra: real-embeddings).
"""

from __future__ import annotations

import numpy as np


def make_cluster_centers(rng: np.random.Generator, clusters: int, dim: int) -> np.ndarray:
    return _l2_normalize(rng.standard_normal((clusters, dim)).astype(np.float32))


def make_word_vectors(rng: np.random.Generator, vocab_size: int, dim: int) -> np.ndarray:
    return _l2_normalize(rng.standard_normal((vocab_size, dim)).astype(np.float32))


def text_derived_embeddings(
    token_ids: list[np.ndarray],
    cluster_ids: np.ndarray,
    word_vectors: np.ndarray,
    centers: np.ndarray,
    cluster_weight: float = 0.6,
) -> np.ndarray:
    """One embedding per item from its token ids + cluster (see module docstring)."""
    dim = word_vectors.shape[1]
    out = np.empty((len(token_ids), dim), dtype=np.float32)
    cw = np.float32(cluster_weight)
    for i, toks in enumerate(token_ids):
        bag = word_vectors[toks].mean(axis=0) if len(toks) else np.zeros(dim, np.float32)
        out[i] = cw * centers[cluster_ids[i]] + bag
    return _l2_normalize(out)


def real_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    model = SentenceTransformer(model_name)
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (x / norms).astype(np.float32)
