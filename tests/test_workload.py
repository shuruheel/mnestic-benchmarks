"""Workload, ground-truth, and fusion correctness — no engine dependencies."""

from __future__ import annotations

import tempfile
from pathlib import Path

from hybrid_recall.fusion import rrf_fuse
from hybrid_recall.metrics import recall_at_k
from hybrid_recall.workload import generate as g
from hybrid_recall.workload.groundtruth import compute_ground_truth
from hybrid_recall.workload.schema import Workload

TINY = dict(
    entities=300, edges=1500, chunks=1200, dim=48, queries=60, graph_hops=2, k=10, clusters=12
)


def _gen(seed=7):
    return g.generate("tiny", seed=seed, **TINY)


def test_generation_deterministic():
    wl1, wl2 = _gen(), _gen()
    gt1, gt2 = compute_ground_truth(wl1), compute_ground_truth(wl2)
    assert gt1 == gt2
    assert all(len(v) == TINY["k"] for v in gt1.values())


def test_save_load_roundtrip():
    wl = _gen()
    gt = compute_ground_truth(wl)
    with tempfile.TemporaryDirectory() as d:
        wl.save(d)
        wl2 = Workload.load(Path(d))
    assert compute_ground_truth(wl2) == gt


def test_signals_correlated_but_distinct():
    """Vector~FTS should agree more than either agrees with graph (the regime fusion needs)."""
    from hybrid_recall.workload.groundtruth import (
        _BM25Index,
        exact_vector_topn,
        graph_proximity_topn,
    )

    wl = _gen()
    wl.ensure_index()
    bm = _BM25Index([c.text for c in wl.chunks])
    cid = [c.id for c in wl.chunks]

    def jac(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b) if a | b else 1.0

    vb = vg = 0.0
    n = 40
    for q in wl.queries[:n]:
        v = [cid[i] for i in exact_vector_topn(wl.query_emb[q.vec_idx], wl.chunk_emb, 20)]
        b = [cid[i] for i in bm.top(q.text.split(), 20)]
        gr = graph_proximity_topn(wl, q.seed_entity, 2, 20)
        vb += jac(v, b)
        vg += jac(v, gr)
    assert vb / n > vg / n  # content signals reinforce; graph is distinct


def test_rrf_fuse_basic():
    # an item ranked highly in two lists should beat one ranked once
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=3)
    assert fused[0] in {"a", "b"}
    assert set(fused) <= {"a", "b", "c", "d"}


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["a", "b", "x"], 3) == 2 / 3
    assert recall_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    assert recall_at_k(["x"], ["a"], 1) == 0.0
