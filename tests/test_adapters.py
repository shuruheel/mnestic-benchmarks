"""Adapter smoke tests — each runs only if its engine package is importable."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import pytest

from hybrid_recall.adapters import EMBEDDED, load_adapter
from hybrid_recall.runner import run_engine
from hybrid_recall.workload import generate as g
from hybrid_recall.workload.groundtruth import compute_ground_truth

TINY = dict(
    entities=300, edges=1500, chunks=1200, dim=48, queries=60, graph_hops=2, k=10, clusters=12
)

# engine -> (import name, expected-good wiring? recall floor for an all-signal engine)
_IMPORT = {
    "mnestic": "mnestic",
    "sqlite": "apsw",
    "duckdb": "duckdb",
    "lancedb": "lancedb",
    "kuzu": "kuzu",
}


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


@pytest.fixture(scope="module")
def workload_and_gt():
    wl = g.generate("tiny", seed=7, **TINY)
    return wl, compute_ground_truth(wl)


@pytest.mark.parametrize("engine", EMBEDDED)
def test_adapter_runs(engine, workload_and_gt):
    mod = _IMPORT[engine]
    if not _have(mod):
        pytest.skip(f"{engine}: {mod} not installed")
    wl, gt = workload_and_gt
    with tempfile.TemporaryDirectory() as d:
        res = run_engine(engine, wl, gt, Path(d), concurrency=2)
    if engine == "kuzu" and not res.ok:
        pytest.skip(f"kuzu unavailable (expected — archived): {res.error[:80]}")
    assert res.ok, f"{engine} failed: {res.error}"
    assert 0.0 <= res.recall_at_k <= 1.0
    caps = load_adapter(engine).capabilities
    if caps.vector and caps.fts and caps.graph:
        # an engine with all three signals should reconstruct most of the fused top-k
        assert res.recall_at_k > 0.6, f"{engine} recall too low: {res.recall_at_k}"
    if not caps.graph:
        # missing the graph signal must cost recall vs the 3-signal oracle
        assert res.recall_at_k < 0.8
