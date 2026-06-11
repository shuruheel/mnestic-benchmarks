"""Split-time ::hnsw create vs ::fts create on the small-scale shape.

Reproduces the `latest-small.json` mnestic build (40k chunks, 384-dim, RocksDB)
but times the two index builds separately, since the harness reports them as one
`build_seconds`. Synthetic vectors/text match the workload's shape (10-word
chunks from a clustered vocabulary), not its exact content — fine for a
where-does-the-time-go split, not for recall numbers.

Usage: .venv/bin/python scripts/build_split_timing.py [n_chunks]
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import mnestic

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40_000
MODE = sys.argv[2] if len(sys.argv) > 2 else "clustered"  # or "uniform"
DIM = 384
CLUSTERS = 64
VOCAB = 4096
TOPIC = 64
CHUNK_WORDS = 10
BATCH = 1024

rng = np.random.default_rng(1234)
vocab = [f"w{i:04d}" for i in range(VOCAB)]
topics = [rng.choice(VOCAB, size=TOPIC, replace=False) for _ in range(CLUSTERS)]
# clustered unit vectors: cluster centroid + noise, normalized (HNSW build cost
# depends on graph structure, so keep the clustered geometry of the workload)
centroids = rng.normal(size=(CLUSTERS, DIM)).astype(np.float32)
centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

tmp = tempfile.mkdtemp(prefix="mnestic-build-split-")
db = mnestic.CozoDbPy("rocksdb", str(Path(tmp) / "store.rocksdb"), "{}")
print(f"db: {tmp} | rows={N} dim={DIM}")

db.run_script(
    ":create chunk {cid: String => text: String, emb: <F32; %d>}" % DIM,
    {},
    False,
)

t0 = time.perf_counter()
batch = []
for i in range(N):
    c = int(rng.integers(0, CLUSTERS))
    words = rng.choice(topics[c], size=CHUNK_WORDS, replace=True)
    text = " ".join(vocab[w] for w in words)
    if MODE == "uniform":
        v = rng.normal(size=DIM).astype(np.float32)
    else:
        v = centroids[c] + 0.3 * rng.normal(size=DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    batch.append([f"c{i:06d}", text, [float(x) for x in v]])
    if len(batch) >= BATCH:
        db.run_script("?[cid,text,emb] <- $r :put chunk {cid=>text,emb}", {"r": batch}, False)
        batch = []
if batch:
    db.run_script("?[cid,text,emb] <- $r :put chunk {cid=>text,emb}", {"r": batch}, False)
print(f"ingest: {time.perf_counter() - t0:.1f}s")

t0 = time.perf_counter()
db.run_script(
    "::hnsw create chunk:vec { dim: %d, m: 16, dtype: F32, fields: [emb], "
    "distance: Cosine, ef_construction: 64 }" % DIM,
    {},
    False,
)
hnsw_s = time.perf_counter() - t0
print(f"::hnsw create: {hnsw_s:.1f}s")

t0 = time.perf_counter()
db.run_script(
    "::fts create chunk:fts { extractor: text, tokenizer: Simple, filters: [Lowercase] }",
    {},
    False,
)
fts_s = time.perf_counter() - t0
print(f"::fts create:  {fts_s:.1f}s")
print(f"total build:   {hnsw_s + fts_s:.1f}s")
