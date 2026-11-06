"""Oracle and failure accounting checks for the release campaign."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("campaign", Path(__file__).parents[1] / "scripts/mnestic_019_campaign.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def test_eligible_fts_uses_global_statistics_and_key_ties():
    bm = c._BM25Index(["alpha", "alpha", "other other", "alpha alpha"])
    ids = ["c2", "c10", "c0", "c3"]
    assert c.eligible_fts_truth(bm, ["alpha"], ids, {"c2", "c10"}, 1) == ["c10"]
    assert c.eligible_fts_truth(bm, ["alpha"], ids, {"c0"}, 10) == []
    assert c.eligible_fts_truth(bm, ["alpha"], ids, set(), 10) == []


def test_errors_stay_in_recall_denominator_and_empty_truth_is_na():
    def fail(): raise RuntimeError("fixture timeout")
    out = c.observe(fail, ["a"], {"a"})
    assert out["error"] and out["recall"] == 0 and out["empty"]
    assert c.observe(lambda: [], [], set())["recall"] is None
    bad = c.observe(lambda: ["b", "b"], ["a"], {"a"})
    assert bad["duplicates"] == 1 and bad["ineligible"] == ["b"]


@pytest.mark.parametrize("message", ["hnsw::repair_limit: degree cap", "storage unavailable"])
def test_write_error_is_retained_and_stops_before_visibility_probes(monkeypatch, message):
    chunk = SimpleNamespace(id="new", text="text", entity_ids=["entity"])
    workload = SimpleNamespace(inc_chunks=[chunk, chunk], inc_chunk_emb=[None, None])
    calls = []
    records = []

    def fail(*args):
        calls.append(args)
        raise RuntimeError(message)

    monkeypatch.setattr(c, "write_atomic", fail)
    with pytest.raises(RuntimeError, match=message):
        c.freshness(object(), workload, lambda **record: records.append(record))
    assert len(calls) == 1
    assert len(records) == 1
    assert records[0]["phase"] == "mutation"
    assert records[0]["error"] == message
    assert records[0]["repair_limit"] == ("hnsw::repair_limit" in message)
    assert records[0]["ms"] >= 0


def test_report_retains_rejected_write_and_refuses_completion(tmp_path, monkeypatch):
    report_spec = importlib.util.spec_from_file_location(
        "campaign_report", Path(__file__).parents[1] / "scripts/mnestic_019_report.py")
    report = importlib.util.module_from_spec(report_spec)
    report_spec.loader.exec_module(report)
    rows = [
        {"phase": "manifest", "args": {"backend": "sqlite", "build": "candidate"}},
        {"phase": "mutation", "write": 0, "ms": 1.5,
         "error": "hnsw::repair_limit", "repair_limit": True},
    ]
    (tmp_path / "sqlite-1234-candidate.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n")
    output = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", ["report", str(tmp_path), "--output", str(output)])
    report.main()
    result = json.loads(output.read_text())
    run = result["runs"]["sqlite-1234-candidate"]
    assert not run["complete"] and not result["all_planned_runs_completed"]
    assert run["freshness"] == {}
    assert run["mutations"]["attempted"] == run["mutations"]["errors"] == 1
    assert run["mutations"]["repair_limit_rejections"] == 1
    assert run["mutations"]["records"] == rows[1:]


def test_success_mutation_logging_does_not_delay_visibility_probes(monkeypatch):
    chunk = SimpleNamespace(id="new", text="text", entity_ids=["entity"])
    workload = SimpleNamespace(inc_chunks=[chunk], inc_chunk_emb=[None])
    events = []
    monkeypatch.setattr(c, "write_atomic", lambda *args: events.append("write"))

    def observe(_probe):
        events.append("probe")
        return {"ids": ["new"], "ms": 0, "error": None}

    def emit(**record):
        if record["phase"] == "mutation":
            assert events == ["write"] + ["probe"] * 6
            assert record["error"] is None
            raise StopIteration("probes verified; skip unrelated rollback fixture")

    monkeypatch.setattr(c, "observe", observe)
    with pytest.raises(StopIteration, match="probes verified"):
        c.freshness(object(), workload, emit)
