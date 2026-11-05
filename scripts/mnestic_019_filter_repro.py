# SPDX-License-Identifier: MPL-2.0
"""Isolated PR #65 selective-vector case; this is a reproducer, not an E2 gate run.

Adapted from kmaragon's Mnestic PR #65, runtime/tests.rs::line_db and
hnsw_filter_does_not_truncate_the_result at e7fba2abf4b88ce153e880ca2e8b022af391826d.
No layered storage or new validity option is required. ANN completeness at small
ef is measured, not asserted as an existing engine contract.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import mnestic

SCHEMA = ":create pt {k: Int => v: <F32; 2>}"
INSERT = "?[k, v] := k in int_range(25), v = vec([k, k]) :put pt {k => v}"
INDEX = "::hnsw create pt:i {fields: [v], dim: 2, dtype: F32, distance: L2, m: 16, ef_construction: 20}"
CASES = {"rare": [20, 22, 24], "empty": [], "fewer_than_k": [24]}
EFS = [3, 5, 50, 100, 200, 400, 800, 1600]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(backend, repetition):
    with tempfile.TemporaryDirectory(prefix="mnestic-019-filter-") as directory:
        db = mnestic.CozoDbPy(backend, directory + "/db", "{}")
        try:
            for script in (SCHEMA, INSERT, INDEX):
                db.run_script(script, {}, False)
            records = []
            for name, eligible in CASES.items():
                # Squared L2 to [0,0] for point [k,k] is 2*k*k.
                truth = sorted(eligible, key=lambda key: (2 * key * key, key))[:3]
                for ef in EFS:
                    query = (
                        "?[k] := ~pt:i{k | query: q, k: 3, ef: $ef, "
                        "filter: " + (" || ".join(f"k == {key}" for key in eligible) or "false")
                        + "}, q = vec([0.0, 0.0])"
                    )
                    started = time.perf_counter_ns()
                    result = db.run_script(query, {"ef": ef}, True)
                    elapsed = time.perf_counter_ns() - started
                    keys = [row[0] for row in result["rows"]]
                    if not set(keys).issubset(eligible) or len(keys) != len(set(keys)):
                        raise AssertionError(f"invalid result: {backend} {name} {ef}: {keys}")
                    records.append({
                        "backend": backend, "repetition": repetition, "case": name,
                        "ef": ef, "query": query, "eligible": eligible, "truth": truth,
                        "returned": keys, "returned_count": len(keys),
                        "recall": len(set(keys) & set(truth)) / len(truth) if truth else None,
                        "elapsed_ns": elapsed,
                    })
            return records
        finally:
            db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="sqlite,rocksdb")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    root = Path(__file__).resolve().parents[1]
    package = Path(mnestic.__file__).resolve().parent
    tracked_diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
    manifest = {
        "experiment": "019-pr65-isolated-filter-reproducer-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": importlib.metadata.version("mnestic"),
        "installed_extension_sha256": {p.name: digest(p) for p in package.glob("*.so")},
        "python": platform.python_version(), "platform": platform.platform(),
        "harness_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "harness_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "script_sha256": digest(Path(__file__).resolve()),
        "source_pr": "https://github.com/shuruheel/mnestic/pull/65",
        "source_sha": "e7fba2abf4b88ce153e880ca2e8b022af391826d",
        "fixture": {"schema": SCHEMA, "insert": INSERT, "index": INDEX, "cases": CASES},
        "limitations": [
            "Tiny fixture only; not the E2 selectivity sweep or release-quality gate.",
            "No visit counters, shared deadline, or statement memory bound measured.",
            "Elapsed times are smoke observations, not benchmark claims; no warmup.",
            "HNSW topology RNG seed is not controlled; each repetition builds a fresh database.",
            "Installed extension hash is recorded; wheel hash and engine source provenance need separate qualification.",
        ],
    }
    records = []
    for backend in args.backends.split(","):
        for repetition in range(args.repetitions):
            records.extend(run_case(backend, repetition))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"manifest": manifest, "records": records}, indent=2) + "\n")
    print(f"Wrote {len(records)} observations to {args.output}")
    for backend in args.backends.split(","):
        print(backend, [(r["ef"], r["returned"]) for r in records
                        if r["backend"] == backend and r["repetition"] == 0 and r["case"] == "rare"])


if __name__ == "__main__":
    main()
