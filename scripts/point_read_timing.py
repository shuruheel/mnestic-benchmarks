"""Point-read latency microbench for the snapshot read path (RocksDB).

The retrieval-scale harness queries (40-150 ms) can't see per-script
transaction overhead; this measures where it actually lives — cheap,
high-frequency reads: single-row keyed lookups and small prefix scans, run as
immutable scripts. Reports per-call µs (p50/p99) over N iterations.

Usage: .venv/bin/python scripts/point_read_timing.py [n_rows] [n_iters]
"""

from __future__ import annotations

import statistics
import sys
import tempfile
import time
from pathlib import Path

import mnestic

N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
N_ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000

tmp = tempfile.mkdtemp(prefix="mnestic-point-read-")
db = mnestic.CozoDbPy("rocksdb", str(Path(tmp) / "store.rocksdb"), "{}")

db.run_script(":create kv { k: Int => v: String }", {}, False)
batch = [[i, f"value-{i:08d}"] for i in range(N_ROWS)]
for i in range(0, N_ROWS, 2000):
    db.run_script("?[k, v] <- $r :put kv {k => v}", {"r": batch[i : i + 2000]}, False)

def bench(script: str, params_for, label: str) -> None:
    # warmup
    for i in range(500):
        db.run_script(script, params_for(i), True)
    times = []
    for i in range(N_ITERS):
        t0 = time.perf_counter_ns()
        db.run_script(script, params_for(i), True)
        times.append((time.perf_counter_ns() - t0) / 1000.0)
    times.sort()
    print(
        f"{label}: p50 {statistics.median(times):.1f}us  "
        f"p99 {times[int(len(times) * 0.99)]:.1f}us  mean {statistics.mean(times):.1f}us"
    )

bench("?[v] := *kv{k: $k, v}", lambda i: {"k": (i * 7919) % N_ROWS}, "point read   ")
bench(
    "?[k, v] := *kv{k, v}, k >= $k, k < $k2 :limit 20",
    lambda i: {"k": (i * 104729) % (N_ROWS - 30), "k2": (i * 104729) % (N_ROWS - 30) + 30},
    "small scan   ",
)
