"""Canonical Reciprocal Rank Fusion (RRF).

This is THE fusion formula for the benchmark. Ground truth and every adapter that fuses
app-side use this exact function, so recall@k measures each engine's *retrieval fidelity*
(ANN recall, BM25 scoring, graph traversal) rather than differences in fusion math.

RRF score for a document d over ranked lists R_1..R_m:
    score(d) = sum_i  1 / (rrf_k + rank_i(d))      (rank is 1-based; absent => no term)

Ties are broken by document id (ascending) for fully deterministic output.
"""

from __future__ import annotations

from collections.abc import Sequence


def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    k: int,
    rrf_k: float = 60.0,
) -> list[str]:
    """Fuse `rankings` (each a rank-ordered list of ids) into a top-k id list."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (rrf_k + rank)
    # sort by score desc, then id asc for determinism
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [doc for doc, _ in ordered[:k]]
