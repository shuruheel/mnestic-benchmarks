"""LanceDB adapter — native vector + full-text + single-call hybrid (RRF reranker).

LanceDB fuses vector and full-text natively in one call (default `RRFReranker`), which we
report under the native-fusion highlight. But it has **no graph traversal** — so it cannot
contribute the graph signal at all. That gap shows up honestly as lost recall here, and in
practice means a separate graph store is needed for graph-augmented memory.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("LANCE_LOG", "error")  # quiet the Rust-side autoprojection warnings

import lancedb  # noqa: E402
import pyarrow as pa  # noqa: E402

from ..workload.schema import Workload, WorkloadMeta
from .base import Adapter, BuildStats, Capabilities, IngestStats, QuerySpec, dir_size_bytes

_BATCH = 10000


class LanceDBAdapter(Adapter):
    name = "lancedb"
    thread_safe = False
    capabilities = Capabilities(
        vector=True,
        fts=True,
        graph=False,
        native_fusion=True,
        fusion_locus="native",
        engines_needed=2,  # needs a separate graph store for the graph signal
        embedded=True,
        notes="Native vector + full-text with a built-in RRF reranker (single-call hybrid). "
        "No graph traversal — the graph signal is simply unavailable.",
    )

    def setup(self, meta: WorkloadMeta, workdir: Path) -> None:
        self.meta = meta
        self.dim = meta.dim
        self._dir = workdir
        self.version = lancedb.__version__
        self._db = lancedb.connect(str(workdir))
        self._schema = pa.schema(
            [
                pa.field("cid", pa.string()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.dim)),
            ]
        )
        self._tbl = self._db.create_table("chunks", schema=self._schema)

    def ingest(self, wl: Workload) -> IngestStats:
        t0 = time.perf_counter()
        emb = wl.chunk_emb
        rows = 0
        batch = []
        for c in wl.chunks:
            batch.append({"cid": c.id, "text": c.text, "vector": [float(x) for x in emb[c.vec_idx]]})
            if len(batch) >= _BATCH:
                self._tbl.add(batch)
                rows += len(batch)
                batch = []
        if batch:
            self._tbl.add(batch)
            rows += len(batch)
        return IngestStats(rows=rows, seconds=time.perf_counter() - t0)

    def build_indexes(self) -> BuildStats:
        t0 = time.perf_counter()
        self._tbl.create_fts_index("text", use_tantivy=False, replace=True)
        # ANN index needs enough rows to train; fall back to exact (flat) search otherwise.
        try:
            self._tbl.create_index(metric="cosine", vector_column_name="vector")
        except Exception:  # noqa: BLE001
            pass
        return BuildStats(seconds=time.perf_counter() - t0, disk_bytes=dir_size_bytes(self._dir))

    # --- component retrievals -------------------------------------------
    def search_vector(self, spec: QuerySpec) -> list[str]:
        rows = (
            self._tbl.search([float(x) for x in spec.vector], vector_column_name="vector")
            .metric("cosine")
            .limit(spec.pool_n)
            .select(["cid"])
            .to_list()
        )
        return [r["cid"] for r in rows]

    def search_fts(self, spec: QuerySpec) -> list[str]:
        rows = (
            self._tbl.search(spec.text, query_type="fts")
            .limit(spec.pool_n)
            .select(["cid"])
            .to_list()
        )
        return [r["cid"] for r in rows]

    def native_hybrid(self, spec: QuerySpec) -> list[str] | None:
        rows = (
            self._tbl.search(query_type="hybrid", vector_column_name="vector")
            .vector([float(x) for x in spec.vector])
            .text(spec.text)
            .limit(spec.k)
            .select(["cid"])
            .to_list()
        )
        return [r["cid"] for r in rows][: spec.k]
