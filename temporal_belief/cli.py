"""Run the temporal-belief workload and print the assertion + timing report.

    python -m temporal_belief.cli [--engine mem|sqlite] [--subjects N]

Exit code 0 iff every engine answer matched the oracle row-for-row.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile

from temporal_belief.oracle import build_scenario
from temporal_belief.workload import Workload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="mem", choices=["mem", "sqlite"])
    ap.add_argument("--subjects", type=int, default=200)
    ap.add_argument("--sources", type=int, default=8)
    ap.add_argument("--json", action="store_true", help="machine-readable report")
    args = ap.parse_args()

    from mnestic import CozoDbPy

    if args.engine == "sqlite":
        tmp = tempfile.mkdtemp(prefix="temporal_belief_")
        db = CozoDbPy("sqlite", f"{tmp}/belief.db", "{}")
    else:
        db = CozoDbPy("mem", "", "{}")

    wl = Workload(db, build_scenario(n_subjects=args.subjects, n_sources=args.sources))
    report = wl.run()
    report["engine"] = args.engine

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"temporal-belief workload — engine={args.engine}, "
              f"{report['scale']['subjects']} subjects, "
              f"{report['scale']['claims']} claims")
        print(f"  oracle verdict: {'ALL MATCHED' if report['ok'] else 'MISMATCH'}")
        for f in report["failures"]:
            print(f"  FAIL {f}")
        print("  timings:")
        for k, v in report["timings_s"].items():
            print(f"    {k:>22}: {v*1000:9.1f} ms")
        rec = report["timings_s"].get("reconcile_contested", 0) + report[
            "timings_s"
        ].get("reconcile_settled", 0)
        total = sum(report["timings_s"].values())
        if total > 0:
            print(f"  :reconcile share of total: {rec/total:.0%} "
                  "(if this dominates at scale, that is the evidence a SCOPED "
                  ":reconcile has been waiting for — see ROADMAP)")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
