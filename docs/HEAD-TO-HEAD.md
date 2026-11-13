# mnestic vs Relate: head-to-head

Relate is Neo4j Labs' experimental embedded graph engine (Cypher, vector,
full-text, preview fusion). This page compares it with mnestic on the axes an
agent-memory store lives on: retrieval quality, read latency, writes under
churn, reopening, crash durability and memory. It also records what the first
comparison exposed in mnestic and what 0.19.2 changed.

Engines: Relate 1.0.0 (Python wheel); mnestic 0.19.0 (PyPI) and 0.19.2
(release build). One run each on one 8 GB Apple-silicon laptop; numbers are
indicative, not a lab benchmark. Raw results are in [`results/sample/head-to-head-2026-11/`](../results/sample/head-to-head-2026-11/).
Reproduce with `hybrid-recall run|lifecycle|quality` (see
[METHODOLOGY.md](METHODOLOGY.md)); set `MNESTIC_BACKEND=sqlite|rocksdb`.

## Retrieval quality (BEIR SciFact, real embeddings)

5,183 documents, 300 labelled queries, `BAAI/bge-small-en-v1.5`.

| system | vector nDCG@10 | BM25 nDCG@10 | fused nDCG@10 | vector R@100 | vector p50 | BM25 p50 |
|---|---|---|---|---|---|---|
| exact reference | 0.721 | 0.632 | 0.713 | 0.955 | — | — |
| mnestic 0.19.2 | 0.717 | 0.662 | 0.715 | 0.943 | 12.7 ms | 36.6 ms |
| Relate 1.0.0 | 0.721 | 0.662 | 0.719 | 0.955 | 5.9 ms | 137.3 ms |

Quality is at parity. Relate's vector search is an exact scan, so it matches the
exact reference; mnestic's HNSW gives up 0.004 nDCG. At this small corpus the
scan is faster; mnestic's maintained BM25 index is 3.7× faster than Relate's
per-query full-text evaluation.

## Hybrid recall (40,000 memories, synthetic)

`small` scale: 40,000 chunks, 10,000 entities, dim 384, clustered synthetic
vectors (`real_embeddings: false`). SQLite backend for mnestic.

| | mnestic 0.19.0 | mnestic 0.19.2 | Relate 1.0.0 |
|---|---|---|---|
| fused recall@10 (vector + FTS + graph, RRF) | 0.959 | 0.956 | 1.000 |
| fused p50 / p99 | 189 / 806 ms | 37 / 58 ms | 440 / 990 ms |
| native vec+FTS fusion p50 / p99 | 131 / 4,454 ms | 41 / 65 ms | 186 / 737 ms |
| incremental upsert p50 | 438 ms | 17 ms | 4.5 ms |
| filtered vector search, 1.9% filter, recall@10 | 0.18 (`ef: 100`) | 1.0 (`ef: 100`, 100 ms) | 1.0 (45 ms) |
| filtered via partial HNSW index | 0.997 | 0.997 (8.9 ms) | — |

## Lifecycle (40,000 memories)

Churn = 10% re-embedded + 5% deleted (6,000 writes); reopen is process-cold,
OS-cache-warm; crash = SIGKILL of a writer mid-stream, process-crash durability
only (not power loss).

| | mnestic 0.19.0 (SQLite) | mnestic 0.19.2 (SQLite) | mnestic 0.19.2 (RocksDB) | Relate 1.0.0 |
|---|---|---|---|---|
| writes rejected | 517 of 6,000 | 0 | 0 | 0 |
| update / delete p50 | 300 / 193 ms | 20 / 7.2 ms | 37 / 78 ms | 0.8 / 0.2 ms |
| recall@10 after churn | 0.9985 | 0.999 | 0.994 | 1.0 |
| reopen | 38 ms | 43 ms | 170 ms | 4,988 ms |
| first query after reopen | 285 ms | 19.5 ms | 70 ms | 247 ms |
| first write after reopen | 7.8 s | 0.03 s | 0.07 s | 0.01 s |
| acknowledged writes lost after SIGKILL | 0 | 0 | 0 | 0 |
| RSS after reopen + queries, over baseline | under 20 MB | 54 MB | 87 MB | 992 MB |

The 0.19.2 RocksDB run used the release candidate before the SQLite-only
checkpoint fix; that fix does not touch RocksDB.

## What the comparison exposed in mnestic, and what 0.19.2 did

- **Rejected writes under churn** (0.19.0's HNSW repair limit): fixed in 0.19.1;
  0.19.2 adds guards so the fallback never leaves a vector with no in-edge.
- **Slow single-row writes and a multi-second first write after reopening:**
  SQLite ran with a rollback journal, a 2 MB cache and a rowid table; writes
  rescanned the whole full-text corpus when its in-memory statistics were cold;
  unchanged fields still had their indexes rewritten; and the write-ahead log
  was never checkpointed while open. All fixed in 0.19.2.
- **Selective filtered search returned 18% of the right answers:** 0.19.2 falls
  back to an exact scan of the eligible rows when the graph window holds fewer
  than `k` matches.

## Where Relate is still ahead, and where it is not

Relate holds the whole graph in RAM and scans vectors exactly, so single-row
writes are sub-millisecond and every search is exact. mnestic's writes maintain
disk-resident HNSW and full-text indexes inside a durable transaction: tens of
milliseconds, not one. Its filtered fallback costs a full scan (100 ms at 40,000
rows) where Relate scans in memory (45 ms); a partial index is the fast path.

mnestic reopens in tens of milliseconds and uses tens of megabytes; Relate
reloads its graph into memory (5 s and about 1 GB at 40,000 memories), and its
full-text search re-derives statistics per query. mnestic also keeps valid-time
and transaction-time history, which Relate does not offer.
