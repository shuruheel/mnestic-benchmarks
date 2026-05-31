# hybrid-recall-bench

A **public, reproducible benchmark for hybrid recall** — the task at the heart of agentic
memory: retrieving the right memories by fusing **vector similarity**, **full-text (BM25)**,
and **graph proximity** into a single ranked result.

It compares [**mnestic**](https://github.com/shuruheel/mnestic) (a maintained fork of CozoDB,
tuned as a substrate for agentic memory) against other embedded and server databases:
**SQLite** (sqlite-vec + FTS5), **DuckDB** (VSS + FTS), **Kuzu**, **LanceDB**, and — in
phase 2 — **Neo4j**, **Qdrant**, and **ArcadeDB**.

> **Why this benchmark?** Most stores do *some* of vector / full-text / graph. Few do all
> three, and fewer still fuse them in one query. This suite measures both **retrieval
> quality** (recall@k vs an exact ground truth) and **cost** (latency, throughput, build
> time, memory), and is explicit about *how much application glue* each engine needs to
> express the fused query. See [`docs/COMPARISON.md`](docs/COMPARISON.md).

## Honesty first

This benchmark is authored by the maintainers of mnestic. A vendor benchmark is only worth
anything if you can re-run it and check it. So:

- **Open methodology** — every workload parameter, the exact RRF fusion, and the
  ground-truth definition are specified in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).
- **Pinned versions & fixed seeds** — same inputs on every machine.
- **Exact ground truth** — recall is measured against a brute-force NumPy oracle, not
  against any engine's own output.
- **Fair fusion** — all app-side engines fuse with the *same* canonical RRF; engines that
  lack a signal (e.g. graph) get a clearly-flagged assist instead of a silent zero.
- **Stated caveats** — e.g. DuckDB's HNSW persistence is experimental; sqlite-vec is
  pre-1.0; **Kuzu was archived in October 2025** (last release v0.10.0).
- **PRs welcome** — if any engine is configured unfairly, open a PR.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[embedded,plots]"      # mnestic, sqlite, duckdb, kuzu, lancedb
# (mnestic: see docs/METHODOLOGY.md to build the wheel from the mnestic repo)

hybrid-recall gen  --scale small                              # generate workload + ground truth
hybrid-recall run  --scale small --engines mnestic,sqlite,duckdb,kuzu,lancedb
hybrid-recall report                                          # -> docs/RESULTS.md + plots
```

CI smoke-runs the embedded engines at the `tiny` scale on every push.

## What gets measured

| Metric | Meaning |
|---|---|
| **recall@k** | overlap of the engine's fused top-k with the exact ground-truth top-k |
| **latency p50/p95/p99** | per-query wall time (single-threaded) |
| **QPS** | throughput under concurrent load |
| **build time** | index construction time after ingest |
| **ingest rate** | rows/sec during load |
| **peak RSS** | max resident memory during build + query |
| **disk size** | on-disk footprint of the populated store |
| **fusion locus** | does the engine fuse natively, or in application code? |
| **round-trips/query** | how many separate engine calls the fused query needs |
| **graph-assisted** | flagged when a signal the engine can't do is supplied externally |

## Layout

```
hybrid_recall/        the harness (workload, ground truth, metrics, runner, report, CLI)
hybrid_recall/adapters/   one module per engine
configs/scales.yaml   tiny / small / medium / large workload definitions
docker/               docker-compose for the phase-2 server engines
docs/                 METHODOLOGY.md, COMPARISON.md, RESULTS.md
```

## License

Apache-2.0 (the harness). mnestic itself is MPL-2.0. CozoDB © Ziyang Hu and the Cozo
Project Authors; mnestic is an independent fork.
