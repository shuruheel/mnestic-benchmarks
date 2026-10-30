"""Filtered-vector axis: the kind assignment and its exact ground truth."""

from __future__ import annotations

import numpy as np

from hybrid_recall.runner import _specs, filtered_ground_truth
from hybrid_recall.workload import generate as g
from hybrid_recall.workload.schema import FILTER_EVERY, FILTER_KIND, chunk_kind

TINY = dict(
    entities=300, edges=1500, chunks=1200, dim=48, queries=60, graph_hops=2, k=10, clusters=12
)


def test_kind_is_deterministic_and_rare():
    wl = g.generate("tiny", seed=7, **TINY)
    kinds = [chunk_kind(c.id) for c in wl.chunks]
    assert kinds == [chunk_kind(c.id) for c in wl.chunks]
    share = kinds.count(FILTER_KIND) / len(kinds)
    assert 0.5 / FILTER_EVERY < share < 2.0 / FILTER_EVERY


def test_filtered_truth_is_the_exact_top_k_within_the_kind():
    wl = g.generate("tiny", seed=7, **TINY)
    specs = _specs(wl)[:5]
    truth = filtered_ground_truth(wl, specs, FILTER_KIND, wl.meta.k)
    rows = [c for c in wl.chunks if chunk_kind(c.id) == FILTER_KIND]
    by_id = {c.id: wl.chunk_emb[c.vec_idx] for c in rows}
    for s, gt in zip(specs, truth):
        assert len(gt) == min(wl.meta.k, len(rows))
        assert all(chunk_kind(cid) == FILTER_KIND for cid in gt)

        def cos(cid):
            v = by_id[cid]
            return float(np.dot(v, s.vector) / (np.linalg.norm(v) * np.linalg.norm(s.vector)))

        worst_kept = min(cos(cid) for cid in gt)
        best_left = max((cos(c.id) for c in rows if c.id not in gt), default=-1.0)
        assert worst_kept >= best_left - 1e-9
