"""Oracle and failure accounting checks for the release campaign."""
import importlib.util
from pathlib import Path

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
