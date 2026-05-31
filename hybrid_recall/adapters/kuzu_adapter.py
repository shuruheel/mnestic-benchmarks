"""Kuzu adapter — embedded property graph DB with Cypher + native vector & FTS indexes.

Kuzu serves all three signals natively (HNSW vector index, FTS/BM25, and Cypher graph
traversal — the graph signal is expressed idiomatically as a variable-length path query).
Fusion is still app-side (no single fused call).

NOTE: Kuzu was archived in October 2025; v0.10.0 is its final release and is what we pin.
The actively-maintained continuation is the RyuGraph fork. See docs/COMPARISON.md.
"""

from __future__ import annotations

import time
from pathlib import Path

import kuzu
import pyarrow as pa
import pyarrow.parquet as pq

from ..workload.schema import Workload, WorkloadMeta
from .base import Adapter, BuildStats, Capabilities, IngestStats, QuerySpec, dir_size_bytes


class KuzuAdapter(Adapter):
    name = "kuzu"
    thread_safe = False
    capabilities = Capabilities(
        vector=True,
        fts=True,
        graph=True,
        native_fusion=False,
        fusion_locus="app-side",
        engines_needed=1,
        embedded=True,
        notes="Embedded graph DB: HNSW vector index + FTS + Cypher variable-length traversal; "
        "app-side fusion. Archived Oct 2025 (v0.10.0 final); RyuGraph is the active fork.",
    )

    def setup(self, meta: WorkloadMeta, workdir: Path) -> None:
        self.meta = meta
        self.dim = meta.dim
        self._dir = workdir
        self.version = kuzu.__version__
        self._db = kuzu.Database(str(workdir / "store.kuzu"))
        self._conn = kuzu.Connection(self._db)
        c = self._conn
        # Kuzu's vector/FTS are downloadable extensions fetched at runtime from
        # extension.kuzudb.com. Following the Oct-2025 archival that host is offline
        # (NXDOMAIN as of 2026), so these installs fail on a clean machine. We surface a
        # clear, documented error rather than a raw network traceback.
        try:
            c.execute("INSTALL vector; LOAD vector;")
            c.execute("INSTALL fts; LOAD fts;")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Kuzu vector/FTS extensions could not be installed — the extension host "
                "(extension.kuzudb.com) is offline following Kuzu's October 2025 archival. "
                "Kuzu v0.10.0 cannot do vector or full-text search without them. The active "
                f"continuation is the RyuGraph fork. Underlying error: {exc}"
            ) from exc
        c.execute(
            f"CREATE NODE TABLE Chunk(cid STRING, text STRING, emb FLOAT[{self.dim}], PRIMARY KEY(cid))"
        )
        c.execute("CREATE NODE TABLE Entity(eid STRING, PRIMARY KEY(eid))")
        c.execute("CREATE REL TABLE Rel(FROM Entity TO Entity)")
        c.execute("CREATE REL TABLE Mentions(FROM Chunk TO Entity)")

    def ingest(self, wl: Workload) -> IngestStats:
        t0 = time.perf_counter()
        tmp = self._dir / "_load"
        tmp.mkdir(exist_ok=True)
        emb = wl.chunk_emb

        ent_p = tmp / "entity.parquet"
        pq.write_table(pa.table({"eid": [e.id for e in wl.entities]}), ent_p)

        chunk_p = tmp / "chunk.parquet"
        pq.write_table(
            pa.table(
                {
                    "cid": [c.id for c in wl.chunks],
                    "text": [c.text for c in wl.chunks],
                    "emb": pa.array(
                        [[float(x) for x in emb[c.vec_idx]] for c in wl.chunks],
                        type=pa.list_(pa.float32(), self.dim),
                    ),
                }
            ),
            chunk_p,
        )

        rel_p = tmp / "rel.parquet"
        pq.write_table(
            pa.table({"src": [e.src for e in wl.edges], "dst": [e.dst for e in wl.edges]}), rel_p
        )

        ment_p = tmp / "mention.parquet"
        m_cid = [c.id for c in wl.chunks for _ in c.entity_ids]
        m_eid = [eid for c in wl.chunks for eid in c.entity_ids]
        pq.write_table(pa.table({"cid": m_cid, "eid": m_eid}), ment_p)

        c = self._conn
        c.execute(f"COPY Entity FROM '{ent_p}'")
        c.execute(f"COPY Chunk FROM '{chunk_p}'")
        c.execute(f"COPY Rel FROM '{rel_p}'")
        c.execute(f"COPY Mentions FROM '{ment_p}'")
        rows = len(wl.entities) + len(wl.chunks) + len(wl.edges) + len(m_cid)
        return IngestStats(rows=rows, seconds=time.perf_counter() - t0)

    def build_indexes(self) -> BuildStats:
        t0 = time.perf_counter()
        self._conn.execute("CALL CREATE_VECTOR_INDEX('Chunk', 'vidx', 'emb', metric := 'cosine')")
        self._conn.execute("CALL CREATE_FTS_INDEX('Chunk', 'fidx', ['text'])")
        return BuildStats(seconds=time.perf_counter() - t0, disk_bytes=dir_size_bytes(self._dir))

    # --- component retrievals -------------------------------------------
    def search_vector(self, spec: QuerySpec) -> list[str]:
        res = self._conn.execute(
            "CALL QUERY_VECTOR_INDEX('Chunk', 'vidx', $q, $k) "
            "RETURN node.cid AS cid ORDER BY distance",
            {"q": [float(x) for x in spec.vector], "k": spec.pool_n},
        )
        return [row[0] for row in _iter(res)]

    def search_fts(self, spec: QuerySpec) -> list[str]:
        res = self._conn.execute(
            "CALL QUERY_FTS_INDEX('Chunk', 'fidx', $q, conjunctive := false) "
            "RETURN node.cid AS cid, score ORDER BY score DESC LIMIT $k",
            {"q": spec.text, "k": spec.pool_n},
        )
        return [row[0] for row in _iter(res)]

    def search_graph(self, spec: QuerySpec) -> list[str]:
        chunk_hop: dict[str, int] = {}
        # hop 0: the seed's own mentioned chunks
        res = self._conn.execute(
            "MATCH (s:Entity {eid: $seed})<-[:Mentions]-(c:Chunk) RETURN c.cid AS cid",
            {"seed": spec.seed_entity},
        )
        for row in _iter(res):
            chunk_hop.setdefault(row[0], 0)
        # hops 1..H: chunks mentioning entities reachable within H rel-hops
        res = self._conn.execute(
            f"MATCH p = (s:Entity {{eid: $seed}})-[:Rel*1..{spec.hops}]-(n:Entity)"
            "<-[:Mentions]-(c:Chunk) "
            "RETURN c.cid AS cid, min(length(p)) - 1 AS hop",
            {"seed": spec.seed_entity},
        )
        for row in _iter(res):
            cid, hop = row[0], int(row[1])
            if cid not in chunk_hop or hop < chunk_hop[cid]:
                chunk_hop[cid] = hop
        ranked = sorted(chunk_hop.items(), key=lambda kv: (kv[1], kv[0]))
        return [cid for cid, _ in ranked[: spec.pool_n]]

    def teardown(self) -> None:
        self._conn = None
        self._db = None


def _iter(result):
    while result.has_next():
        yield result.get_next()
