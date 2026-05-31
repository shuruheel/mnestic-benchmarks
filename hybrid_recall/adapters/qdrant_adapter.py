"""Qdrant adapter — PHASE 2 (server engine).

Qdrant is a dedicated vector search engine. It supports full-text *filtering* and BM25 via
sparse vectors, and has no graph traversal at all. In this pass we wire the **vector**
signal only and declare the others unavailable — honestly reflecting that a Qdrant-based
agentic memory needs separate systems for full-text ranking and graph. (Sparse-vector BM25
and a companion graph store are future work; see docs/COMPARISON.md.)

Requires a running Qdrant (docker/docker-compose.yml). Config via env QDRANT_URL.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from ..workload.schema import Workload, WorkloadMeta
from .base import Adapter, BuildStats, Capabilities, IngestStats, QuerySpec

_BATCH = 1000


class QdrantAdapter(Adapter):
    name = "qdrant"
    thread_safe = True
    capabilities = Capabilities(
        vector=True,
        fts=False,   # full-text/BM25 only via sparse vectors — not wired in this pass
        graph=False,  # no graph traversal
        native_fusion=False,
        fusion_locus="app-side",
        engines_needed=3,  # vector here; needs separate full-text + graph systems
        embedded=False,
        notes="Dedicated vector DB. No graph traversal; BM25 only via sparse vectors "
        "(not wired here). PHASE 2 — vector signal only.",
    )

    def setup(self, meta: WorkloadMeta, workdir: Path) -> None:
        from qdrant_client import QdrantClient, models  # noqa: PLC0415

        self.meta = meta
        self.dim = meta.dim
        self._models = models
        self._client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
        import qdrant_client  # noqa: PLC0415

        self.version = qdrant_client.__version__
        self._coll = "chunks"
        self._client.recreate_collection(
            collection_name=self._coll,
            vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
        )

    def ingest(self, wl: Workload) -> IngestStats:
        t0 = time.perf_counter()
        emb = wl.chunk_emb
        rows = 0
        pts = []
        for i, c in enumerate(wl.chunks):
            pts.append(
                self._models.PointStruct(
                    id=i, vector=[float(x) for x in emb[c.vec_idx]], payload={"cid": c.id}
                )
            )
            if len(pts) >= _BATCH:
                self._client.upsert(collection_name=self._coll, points=pts)
                rows += len(pts)
                pts = []
        if pts:
            self._client.upsert(collection_name=self._coll, points=pts)
            rows += len(pts)
        return IngestStats(rows=rows, seconds=time.perf_counter() - t0)

    def build_indexes(self) -> BuildStats:
        # Qdrant builds the HNSW index incrementally during upsert.
        return BuildStats(seconds=0.0, disk_bytes=0)

    def search_vector(self, spec: QuerySpec) -> list[str]:
        res = self._client.search(
            collection_name=self._coll,
            query_vector=[float(x) for x in spec.vector],
            limit=spec.pool_n,
            with_payload=True,
        )
        return [p.payload["cid"] for p in res]

    def teardown(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass
