# Methodology

This benchmark measures **hybrid recall** — retrieving the right memories by fusing vector
similarity, full-text relevance, and graph proximity into one ranked result — and reports
both its **quality** (recall vs an exact oracle) and its **cost** (latency, throughput,
build time, memory, disk). It is deliberately simple and fully reproducible.

## The task

A *workload* is a synthetic agentic-memory knowledge graph:

- **Entities** — typed nodes (Person / Concept / Event / Org), each with a label, a short
  text summary, and an embedding.
- **Edges** — typed directed relations between entities (the graph structure). ~70% are
  intra-cluster, ~30% cross-cluster, giving navigable community structure.
- **Chunks** — the recall corpus: short "memory" documents, each with text, an embedding,
  and links to ≥1 entity (provenance).
- **Queries** — each is `(text, embedding, seed_entity)`.

A single **recall query** asks for the top-`k` chunks under **Reciprocal Rank Fusion (RRF)**
of three component rankings:

1. **vector** — cosine kNN of the query embedding against chunk embeddings,
2. **full-text** — BM25 of the query terms against chunk text,
3. **graph** — chunks within `graph_hops` of `seed_entity`, ranked by minimum hop distance.

Each component is truncated to `pool_n` candidates before fusion (the same depth engines
are asked to fetch). RRF uses `score(d) = Σ 1/(rrf_k + rank_i(d))`, `rrf_k = 60`.

## Why synthetic, and why it's realistic

Embeddings are **text-derived**: an item's vector is the (L2-normalized) sum of a fixed
per-word random vector over its tokens plus a per-cluster bias. Because real embeddings are
a function of text, this reproduces the regime hybrid recall is built for —

- **vector and full-text are correlated** (they agree on lexically-related chunks), while
- **graph proximity is correlated-but-distinct** (it surfaces memories the content signals
  miss).

No single signal dominates, so fusion changes the answer — and an engine *missing* a signal
loses real recall. A synthetic vocabulary of pseudo-words (no English morphology) keeps
tokenization/stemming near-no-ops, so recall differences reflect **index fidelity**, not
tokenizer disagreements. Everything is seeded, so the same inputs are produced on every
machine. A `--real-embeddings` mode (sentence-transformers) is available for sensitivity
checks; the headline numbers use synthetic embeddings for reproducibility.

## Ground truth and recall

The oracle is pure NumPy/Python (`hybrid_recall/workload/groundtruth.py`):

- exact full-scan cosine kNN (no ANN approximation),
- exact Okapi BM25 (`k1 = 1.5`, `b = 0.75`),
- exact k-hop BFS proximity,

fused with the **same** canonical RRF the engines use. `recall@k = |engine_topk ∩
truth_topk| / k`. Recall is measured against this oracle, never against another engine.

## Fair fusion

Every engine that fuses in application code uses the **identical** `rrf_fuse()` function, so
recall reflects each engine's *retrieval fidelity* (ANN recall, BM25 scoring, graph
traversal) — not differences in fusion math. Engines fetch `pool_n` candidates per signal
they support; an engine that lacks a signal (e.g. LanceDB has no graph) contributes no
ranking for it and loses the recall that signal would have provided. That is the point, and
it is recorded via the capability matrix (`graph`, `fusion_locus`, `engines_needed`).

`native_hybrid` — a single-call vector+FTS fusion offered by some engines (mnestic's
`hybrid_search`, LanceDB's RRF reranker) — is timed and reported **separately** for
latency/ergonomics. It is *not* used in the scored recall comparison, to keep that
comparison apples-to-apples.

## What is measured

recall@k · latency p50/p95/p99 · QPS (single-thread; concurrent where the client is
thread-safe) · index build time · ingest rows/s · peak RSS (build + query) · on-disk size ·
fusion locus (native vs app-side) · round-trips per fused query · graph-assisted flag.

## Architecture axes (where integration matters)

Raw recall/latency is a drag race specialized single-purpose engines tend to win. But an
agentic memory is written *continuously*, queried immediately, and must keep all three
signals consistent. So we also measure the axes a vector-only or multi-system stack
struggles with:

- **Read-your-writes / freshness.** After the base build, we hold out a set of new memories
  (each a chunk + embedding linked to an existing entity, with a probe query equal to its
  own text and embedding, seeded at that entity). For each, the adapter `upsert`s the memory
  the way an agent would at runtime, then *immediately* issues the probe. We record the
  fraction found **per signal** (vector / full-text / graph) and in the **fused** top-k. A
  signal whose index is a build-time snapshot (rather than maintained on write) cannot find
  the new memory — and the missing signal can drag it out of the fused result entirely.
- **Incremental upsert latency** — the per-memory write + index-maintenance cost.
- **Capability flags** — `transactional` (chunk + edges + indexes commit atomically),
  `time-travel` (query memory as-of a past point), `incremental index`, and
  `systems needed` (how many separate stores a real deployment of that engine requires for
  all three signals — 1 for the integrated engines, more for vector-only ones).

These reframe the comparison from "fastest single-signal lookup" to "can this be one
consistent, live agentic-memory substrate" — the question mnestic is actually built to win.

## Reproduce it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[embedded,plots]"
# mnestic wheel: build from the mnestic repo's cozo-lib-python with maturin, e.g.
#   (cd ../mnestic/cozo-lib-python && maturin develop --release)

hybrid-recall gen  --scale small
hybrid-recall run  --scale small --engines mnestic,sqlite,duckdb,lancedb --report --plots
```

Scales (`configs/scales.yaml`): `tiny` (CI smoke), `small` (reference), `medium`, `large`.
Numbers are **hardware-specific**; the reference run's environment is recorded in
`RESULTS.md`. Re-run on your own hardware to compare.

## Honest limitations

- **Synthetic data** approximates, but is not, a real corpus. The `--real-embeddings` mode
  and larger scales are provided to probe sensitivity.
- **Full-text scoring varies by engine.** The oracle uses textbook Okapi BM25; engines that
  implement standard BM25 (SQLite FTS5, DuckDB) align most closely. We make the lexical
  signal discriminative so that the *set* of relevant chunks is robust across reasonable
  BM25 implementations, but small ranking differences remain.
- **ANN vs exact.** sqlite-vec does an exact brute-force scan (perfect vector recall, linear
  cost); HNSW-based engines trade a little recall for speed. Both behaviors are visible in
  the results.
- We **maintain mnestic.** The methodology is open and the harness runs on your machine; if
  any engine is configured unfairly, please open a PR.
