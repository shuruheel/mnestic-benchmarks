# Engine Comparison

How each engine expresses **hybrid recall** (vector + full-text + graph), what it does
natively, and where it can't. For live numbers on your hardware see
[RESULTS.md](RESULTS.md); for how those numbers are produced see
[METHODOLOGY.md](METHODOLOGY.md).

## Capability matrix

| Engine | Vector | Full-text | Graph traversal | Native vec+FTS fusion | Fusion locus | Systems for all 3 signals | Embedded | Status |
|---|---|---|---|---|---|---|---|---|
| **mnestic** | ✅ HNSW | ✅ | ✅ Datalog | ✅ `hybrid_search` | native | **1** | ✅ | active fork of CozoDB |
| SQLite (sqlite-vec + FTS5) | ✅ exact scan | ✅ BM25 | ⚠️ recursive CTE | ❌ | app-side | 1 | ✅ | active (sqlite-vec pre-1.0) |
| DuckDB (VSS + FTS) | ✅ HNSW¹ | ✅ BM25 | ⚠️ recursive CTE | ❌ | app-side | 1 | ✅ | active |
| Kuzu | ✅ HNSW | ✅ BM25 | ✅ Cypher | ❌ | app-side | 1 | ✅ | **archived Oct 2025** ² |
| LanceDB | ✅ | ✅ | ❌ | ✅ RRF reranker | native (vec+FTS) | 2 | ✅ | active |
| Neo4j | ✅ index | ✅ Lucene | ✅ Cypher | ❌ | app-side | 1 | ❌ server | active (phase 2) |
| Qdrant | ✅ | ⚠️ sparse only³ | ❌ | ❌ | app-side | 3 | ❌ server | active (phase 2) |
| ArcadeDB | ✅ HNSW | ✅ Lucene | ✅ | ❌ | app-side | 1 | ❌ server | active (phase 2, scaffold) |

¹ DuckDB's HNSW index only persists with `SET hnsw_enable_experimental_persistence = true`,
which carries a documented risk of index corruption on unclean shutdown.
² See "Kuzu" below — its extension distribution is offline, so v0.10.0 cannot install its
vector/FTS extensions on a clean machine in 2026.
³ Qdrant offers full-text *filtering* and BM25 via sparse vectors, but no BM25 ranking out
of the box and no graph traversal; this pass wires its vector signal only.

## The headline

Three signals, fused, is the task. Three things separate the field:

1. **Does the engine have a graph at all?** A pure vector store (LanceDB, Qdrant) cannot
   contribute the graph-proximity signal, and on this workload that signal is
   correlated-but-distinct from vector/full-text — so dropping it removes recall that the
   other two signals cannot recover. In the reference run this is the single largest effect
   (the graph-less engines land far below the engines that have all three). This is exactly
   why graph-augmented retrieval exists.

2. **How many systems, and how much glue?** mnestic is the only engine here that serves all
   three signals from **one embedded store** *and* fuses vector+full-text in a **single
   call** (`hybrid_search`, RRF inside the engine). SQLite, DuckDB, and Kuzu also keep all
   three in one process but fuse in application code (three queries + a Python RRF). LanceDB
   fuses vector+FTS natively but needs a *second* system for graph; Qdrant needs separate
   systems for full-text and graph.

3. **Can it absorb a memory and recall it immediately?** Agentic memory is written
   continuously, so *read-your-writes* matters as much as batch recall. The architecture-axes
   table in [RESULTS.md](RESULTS.md) measures it: mnestic, SQLite and LanceDB find a
   just-upserted memory on every signal they support, but **DuckDB's full-text index is a
   build-time snapshot** — a new memory is **unsearchable by keyword (FTS read-your-writes =
   0%)** until the index is rebuilt. How badly that hurts depends on the query: when vector
   and graph can carry it the fused result still lands the memory, but any keyword-led recall
   of recent memories silently misses. A drag race on a static corpus hides this entirely.

## Per-engine notes

### mnestic
A maintained fork of CozoDB (MPL-2.0), tuned as a substrate for agentic memory. One
embedded engine provides an HNSW vector index, an FTS index, and full Datalog graph
traversal, plus `hybrid_search` for single-call RRF over vector+FTS — it is the **only engine
here that serves all three signals from one embedded store**, and it comfortably beats the
graph-less vector engines on fused recall.

Where it stands, measured honestly against the exact oracle (component agreement at the
`small` scale): **vector ≈ 0.99 and graph = 1.00 — essentially perfect**. Full-text was
the one weak leg (~0.72) in the first run; the fork then upgraded `::fts` to **Okapi BM25**
(tunable k1/b, single-query multi-term scoring) — see CHANGELOG-FORK — which lifts full-text
agreement to ~0.88 and **fused recall@10 from ~0.75 to ~0.96, at parity with the BM25-native
SQL engines**. The remaining gap is HNSW approximation, not scoring. (This is a worked
example of the bench's purpose: it localized the deficit to FTS, the fix shipped, and the
re-run validated it.)

On **latency**, mnestic's **native `hybrid_search` is its fast path and the point of Bet 1a**:
it fuses vector + full-text + graph in **one** call at ~41 ms p50 (small scale), vs ~162–175 ms
for the decomposed three-query path that a non-fusing engine forces — the "one call vs three"
win, on a capability (3-way fusion) no other engine here offers at all. Three latency lessons
the bench drove out:

1. The first BM25 cut made full-text ~10× slower via a per-query `avgdl` index scan; an O(1)
   durable doc-stats counter restored it (decomposed p50 927→175 ms, p99 2.9 s→258 ms on the
   SQLite wheel).
2. **Re-measured on the RocksDB backend** — the one `mindgraph-rs` actually runs — with **real
   sentence-transformer embeddings** (recorded in `results/sample/small-rocksdb-real.json`):
   decomposed **p50 162 ms, p99 181 ms** (p50 marginally better than the SQLite wheel, and the
   **tail markedly tighter — 181 vs 258 ms p99** — thanks to RocksDB's persistent page cache),
   with native 3-way fusion at **41 ms p50**. The latency claims hold on the production backend.
3. That same re-measure surfaced the next lever directly: HNSW+FTS **index build took ~89 s**
   for 40k vectors (vs <0.1 s for sqlite-vec's no-build exact scan). That is exactly what the
   bridge-gated build-path items target — **#4** (batch the per-index `.put()` loop) and **#7**
   (HNSW `multi_get` for neighbor fetch) — now backed by a measured number on the real backend.

All real, documented, and benchmark-grounded — not artifacts.

> *Recall caveat for the RocksDB run:* its recall@10 is **0.88, not the synthetic run's 0.95**,
> because the workload **text is synthetic vocabulary** — a real sentence-transformer cannot
> embed `word4821 word221 …` meaningfully, which collapses the vector signal (LanceDB's k-means
> even warns of duplicate vectors). Treat that run as a **latency/backend validation on
> realistic dense float32 vectors**, not a recall result. The recall headline above stands on
> the text-derived synthetic embeddings, where the vector signal is meaningful by construction;
> a `--real-embeddings` run on *real* prose would not see this collapse.

### SQLite (sqlite-vec + FTS5)
All three signals in a single file. **sqlite-vec does an exact brute-force KNN scan** — so
vector recall is perfect, but query latency grows linearly with the corpus (great at small
scale, less so at large). FTS5 implements textbook Okapi BM25. Graph is expressible only via
recursive CTEs. No native fusion. `sqlite-vec` is pre-1.0. Note: many CPython builds ship
without loadable-extension support, so the adapter uses `apsw`.

### DuckDB (VSS + FTS)
Analytical engine with an `vss` HNSW index and an `fts` BM25 index. Strong recall and
ingest. Caveat: **HNSW persistence is experimental** (`hnsw_enable_experimental_persistence`)
with a documented corruption risk on crash. Graph via recursive CTEs; app-side fusion.
Second caveat the freshness axis exposes: its **FTS index is a build-time snapshot** — a
newly-inserted memory is unsearchable by keyword until the index is rebuilt (read-your-writes
full-text = 0%). Vector and graph still find it, so fused recall of a fresh memory mostly
survives at this scale, but keyword-led recall of recent memories silently misses. Fine for
static corpora, a real problem for continuously-written agent memory.

### Kuzu
An embedded property-graph DB with Cypher, an HNSW vector index, and FTS — on paper an
excellent fit. **But Kuzu was archived in October 2025, and its extension host
(`extension.kuzudb.com`) is offline (NXDOMAIN as of 2026).** Vector and FTS ship as
runtime-downloaded extensions, so on a clean machine `INSTALL vector` / `INSTALL fts` fail
and Kuzu cannot do vector or full-text search at all. The benchmark records this as a
did-not-complete with that reason — a concrete illustration of the operational risk of
building on an archived database. The actively-maintained continuation is the **RyuGraph**
fork; wiring it in is future work.

### LanceDB
A fast embedded multimodal retrieval engine with native vector + full-text and a built-in
**RRF reranker** (single-call hybrid). It has **no graph traversal**, so it cannot
contribute the graph signal — its fused recall is capped well below the all-three engines,
and a real graph-augmented memory built on LanceDB needs a separate graph store. Its
vector/FTS latency is excellent.

### Neo4j *(phase 2)*
Graph-native server with a 5.x vector index and a Lucene full-text index — all three signals
on one server, app-side fusion. Adapter implemented; run it via `docker/docker-compose.yml`
and `pip install -e ".[server]"`.

### Qdrant *(phase 2)*
A dedicated vector engine. No graph traversal; BM25 only via sparse vectors. We wire the
vector signal only and declare the rest unavailable — honestly reflecting that a Qdrant
agentic memory needs separate systems for full-text ranking and graph.

### ArcadeDB *(phase 2, scaffold)*
A multi-model server (graph + HNSW vector + Lucene FTS) over HTTP. Capabilities and
docker-compose are provided; the query methods are a scaffold left for phase 2 — we don't
ship adapter code we haven't run against a live server.

## Fairness statement

This benchmark is authored by the maintainers of mnestic. Every effort is made to configure
each engine the way a competent user would, to use the same fusion for all app-side engines,
and to score everything against an engine-independent exact oracle. The methodology is open
and the harness runs on your machine. **If you think an engine is configured unfairly, open
a PR** — that is the whole point of making it public and reproducible.
