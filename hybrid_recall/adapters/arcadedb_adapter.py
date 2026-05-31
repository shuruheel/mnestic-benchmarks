"""ArcadeDB adapter — PHASE 2 (server engine), scaffold only.

ArcadeDB is a multi-model engine (graph + document + key-value) with an HNSW vector index
and a Lucene full-text index, queried over HTTP (SQL/Cypher/Gremlin). It can express all
three signals on one server, with app-side fusion.

This is a SCAFFOLD: capabilities and connection settings are declared, and docker-compose
provisions a server, but the query implementations are intentionally left for phase 2 —
ArcadeDB's exact vector/FTS SQL surface is version-sensitive and we will not ship adapter
code we have not run against a live server. `setup` raises a clear, actionable error.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..workload.schema import Workload, WorkloadMeta
from .base import Adapter, BuildStats, Capabilities, IngestStats, QuerySpec


class ArcadeDBAdapter(Adapter):
    name = "arcadedb"
    thread_safe = True
    capabilities = Capabilities(
        vector=True,
        fts=True,
        graph=True,
        native_fusion=False,
        fusion_locus="app-side",
        engines_needed=1,
        embedded=False,
        notes="Multi-model server (graph + HNSW vector + Lucene full-text), HTTP API, "
        "app-side fusion. PHASE 2 — scaffold not yet implemented against a live server.",
    )

    def setup(self, meta: WorkloadMeta, workdir: Path) -> None:
        url = os.environ.get("ARCADEDB_URL", "http://localhost:2480")
        raise NotImplementedError(
            "ArcadeDB adapter is a phase-2 scaffold and not yet implemented. Start the server "
            f"via docker/docker-compose.yml ({url}) and implement the HTTP query methods. "
            "See docs/COMPARISON.md."
        )

    def ingest(self, wl: Workload) -> IngestStats:  # pragma: no cover - phase 2
        raise NotImplementedError

    def build_indexes(self) -> BuildStats:  # pragma: no cover - phase 2
        raise NotImplementedError

    def search_vector(self, spec: QuerySpec) -> list[str]:  # pragma: no cover - phase 2
        raise NotImplementedError
