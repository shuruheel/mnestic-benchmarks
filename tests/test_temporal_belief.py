"""The temporal-belief workload IS its own test: every phase asserts against
the pure-Python oracle. This wrapper runs it at two scales on mem and once on
sqlite, and additionally pins the oracle's own invariants."""

import pytest

from temporal_belief.oracle import (
    build_scenario,
    contested_readings,
    live_after_contradiction,
    live_after_retraction,
)
from temporal_belief.workload import Workload

mnestic = pytest.importorskip("mnestic")


def _run(engine, tmp_path, subjects):
    if engine == "sqlite":
        db = mnestic.CozoDbPy("sqlite", str(tmp_path / "b.db"), "{}")
    else:
        db = mnestic.CozoDbPy("mem", "", "{}")
    wl = Workload(db, build_scenario(n_subjects=subjects))
    report = wl.run()
    assert report["ok"], "\n".join(report["failures"])
    return report


def test_mem_small(tmp_path):
    _run("mem", tmp_path, 60)


def test_mem_default_scale(tmp_path):
    _run("mem", tmp_path, 200)


def test_sqlite(tmp_path):
    _run("sqlite", tmp_path, 60)


def test_oracle_contest_appears_and_settles():
    s = build_scenario(n_subjects=100)
    contested = contested_readings(s, live_after_contradiction(s))
    settled = contested_readings(s, live_after_retraction(s))
    # The contradictor creates multi-reading subjects...
    multi = [subj for subj, vals in contested.items() if len(vals) > 1]
    assert multi, "scenario must actually produce a contest"
    # ...and retraction settles every one of them back to a single reading.
    assert all(len(vals) == 1 for vals in settled.values())
