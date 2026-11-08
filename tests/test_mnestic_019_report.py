"""Report coverage must not hide non-E1 regressions or failed observations."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    "campaign_report", Path(__file__).parents[1] / "scripts/mnestic_019_report.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_non_e1_slowdowns_are_reported_in_pooled_and_seed_cells(tmp_path, monkeypatch):
    for build, ms in [("baseline", 10.0), ("candidate", 14.0)]:
        rows = [
            dict(phase="manifest", args=dict(backend="rocksdb", build=build, seed="1234")),
            dict(phase="E1", path="native", ms=10.0, recall=1.0, error=None),
            dict(phase="mutation", write=0, ms=ms, error=None, repair_limit=False),
            dict(
                phase="E3",
                dim=384,
                shape=1,
                changing=False,
                query_kind="retrieval",
                mode="parameter",
                serialization_ms=0.0,
                construction_ms=0.0,
                engine_ms=ms,
                total_ms=ms,
            ),
        ]
        for phase in ("E2", "E2_1536"):
            rows.append(
                dict(
                    phase=phase,
                    query=0,
                    assignment="random",
                    selectivity=0.01,
                    strategy="filter_ef100",
                    ids=["a"],
                    ms=ms,
                    recall=1.0,
                    error=None,
                )
            )
        (tmp_path / f"rocksdb-1234-{build}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
    output = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", ["report", str(tmp_path), "--output", str(output)])
    report.main()
    result = json.loads(output.read_text())
    assert {(t["phase"], t["seed"]) for t in result["review_triggers"]} == {
        (phase, seed) for phase in ("mutation", "E2", "E2_1536", "E3") for seed in (None, 1234)
    }
    assert all(t["reasons"] == ["p95_latency"] for t in result["review_triggers"])
    assert not result["performance_accepted"]
    assert not result["all_planned_runs_completed"]


def test_failure_latency_and_na_recall_have_distinct_denominators():
    rows = [
        dict(ms=1000.0, error="timeout", recall=0.0, empty=True),
        dict(ms=2.0, error=None, recall=None, empty=True),
    ]
    value = report.stats(rows)
    assert value["n"] == 2 and value["errors"] == 1
    assert value["recall_n"] == 1 and value["recall"] == 0.0
    assert value["latency_ms"]["p95"] == 2.0
    assert value["successful_qps"] < value["attempt_qps"]


def test_missing_or_unequal_pairs_are_not_clean_comparisons():
    point = dict(ms=10.0, error=None, recall=1.0)
    cells = {
        ("mutation", "sqlite", (), None): {"candidate": [point]},
        ("mutation", "rocksdb", (), None): {"baseline": [point], "candidate": [point, point]},
    }
    triggers, issues = report.screen_comparisons(cells)
    assert not triggers
    assert {x["reason"] for x in issues} == {
        "missing baseline/candidate pair",
        "unequal attempt counts",
    }


def test_candidate_errors_remain_visible_without_a_latency_increase():
    cells = {
        ("mutation", "sqlite", (), None): {
            "baseline": [dict(ms=10.0, error=None)],
            "candidate": [dict(ms=1.0, error="hnsw::repair_limit")],
        }
    }
    triggers, issues = report.screen_comparisons(cells)
    assert not issues
    assert triggers[0]["candidate_errors"] == 1
    assert triggers[0]["reasons"] == ["candidate_errors_or_invalid_results"]


@pytest.mark.parametrize("lost_neighbors,expected", [(2, False), (3, True), (10, True)])
def test_strict_one_percentage_point_recall_boundary(lost_neighbors, expected):
    baseline = [dict(ms=10.0, recall=1.0) for _ in range(20)]
    candidate = [dict(ms=10.0, recall=1.0) for _ in range(20)]
    candidate[18]["recall"] = (10 - lost_neighbors) / 10
    cells = {
        ("E2_1536", "sqlite", ("random", "1.0", "filter_ef100"), 1236): {
            "baseline": baseline,
            "candidate": candidate,
        }
    }
    triggers, issues = report.screen_comparisons(cells)
    assert not issues
    assert bool(triggers) is expected
    if expected:
        assert triggers[0]["recall_loss"] == lost_neighbors / 200
        assert triggers[0]["reasons"] == ["recall"]


def test_recall_boundary_does_not_hide_latency_or_na_denominators():
    baseline = [dict(ms=10.0, recall=1.0) for _ in range(20)] + [dict(ms=10.0, recall=None)]
    candidate = [dict(ms=14.0, recall=1.0) for _ in range(20)] + [dict(ms=14.0, recall=None)]
    candidate[0]["recall"] = 0.8
    cells = {("E2", "sqlite", (), None): {"baseline": baseline, "candidate": candidate}}
    triggers, issues = report.screen_comparisons(cells)
    assert not issues
    assert triggers[0]["reasons"] == ["p95_latency"]
    assert triggers[0]["recall_loss"] == 0.01


def test_exact_recall_rejects_a_different_query_contract():
    with pytest.raises(ValueError, match="frozen k=10"):
        report.exact_mean_recall([dict(recall=0.12345)])
