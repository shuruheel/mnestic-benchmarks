"""Lifecycle axis: oracle helpers, churn determinism, and a tiny end-to-end run per
installed engine (load + churn, reopen in a new process, one SIGKILL crash trial)."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict

import numpy as np
import pytest

from hybrid_recall import lifecycle
from hybrid_recall.workload import generate as g

TINY = dict(
    entities=200, edges=800, chunks=600, dim=16, queries=40, graph_hops=2, k=10, clusters=8,
    incremental=10,
)


def test_exact_topk_matches_bruteforce():
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((50, 8)).astype(np.float32)
    q = rng.standard_normal((3, 8)).astype(np.float32)
    ids = [f"r{i}" for i in range(50)]
    got = lifecycle.exact_topk(ids, mat, q, 5)
    m = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    for row, qq in zip(got, q):
        sims = m @ (qq / np.linalg.norm(qq))
        assert row == [ids[i] for i in np.argsort(-sims)[:5]]


def test_churn_plan_is_deterministic_and_disjoint():
    wl = g.generate("t", seed=3, **TINY)
    u1, d1 = lifecycle.churn_plan(wl, 7, 0.1, 0.05)
    u2, d2 = lifecycle.churn_plan(wl, 7, 0.1, 0.05)
    assert [c for c, _t, _v in u1] == [c for c, _t, _v in u2] and d1 == d2
    assert all(np.array_equal(a[2], b[2]) for a, b in zip(u1, u2))
    assert not {c for c, _t, _v in u1} & set(d1)
    assert len(u1) == 60 and len(d1) == 30


def test_error_code_prefers_engine_code():
    assert lifecycle._error_code(Exception({"code": "hnsw::repair_limit"})) == "hnsw::repair_limit"
    assert lifecycle._error_code(ValueError("x")) == "ValueError"


@pytest.mark.parametrize("engine", ["mnestic", "relate"])
def test_lifecycle_end_to_end(engine, tmp_path):
    if importlib.util.find_spec(engine) is None:
        pytest.skip(f"{engine} not installed")
    wl = g.generate("t", seed=3, **TINY)
    wdir = tmp_path / "wl"
    wl.save(wdir)
    res = lifecycle.run_lifecycle(engine, wdir, tmp_path / "work", trials=1,
                                  kill_window=(0.2, 0.4))
    assert res["ok"], res.get("error")
    ld, ro, cr = res["load"], res["reopen"], res["crash"]
    assert ld["pre_churn"]["recall"] > 0.8
    ch = ld["churn"]
    assert ch["updates"] == 60 and ch["deletes"] == 30
    # a rejected write must roll back, never lose the row
    assert "rejected_update_lost_row" not in ld
    # reopen in a fresh process answers like the store it closed
    assert abs(ro["steady"]["recall"] - ld["post_churn"]["recall"]) < 0.05
    assert ro["deleted_resurrected"] == 0
    assert cr[0]["open_ok"] and cr[0]["acked"] > 0 and cr[0]["lost"] == 0
    md = lifecycle.render_markdown(
        {"meta": asdict(wl.meta), "generated_at": "t", "environment": {"platform": "p"}, "results": [res]}
    )
    assert engine in md
