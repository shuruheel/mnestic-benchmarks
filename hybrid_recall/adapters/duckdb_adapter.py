"""DuckDB adapter — VSS (HNSW) + FTS (BM25) + recursive-CTE graph.

DuckDB serves all three signals in-process, but:
  * the HNSW index (vss extension) only persists with the experimental
    `hnsw_enable_experimental_persistence` flag (documented risk of corruption on crash).
  * there is no native fusion: three queries fused in application code.
  * graph traversal is expressible via recursive CTEs only.
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb

from ..graphwalk import bfs_chunk_ranking
from ..workload.schema import Workload, WorkloadMeta
from .base import Adapter, BuildStats, Capabilities, IngestStats, QuerySpec, dir_size_bytes

_BATCH = 5000


class DuckDBAdapter(Adapter):
    name = "duckdb"
    thread_safe = False  # use one connection; duckdb needs a cursor per thread
    capabilities = Capabilities(
        vector=True,
        fts=True,
        graph=True,
        native_fusion=False,
        fusion_locus="app-side",
        engines_needed=1,
        embedded=True,
        notes="VSS HNSW persistence is experimental (hnsw_enable_experimental_persistence); "
        "FTS via PRAGMA create_fts_index/match_bm25; graph via recursive CTEs; app-side fusion.",
    )

    def setup(self, meta: WorkloadMeta, workdir: Path) -> None:
        self.meta = meta
        self.dim = meta.dim
        self._dir = workdir
        self.version = duckdb.__version__
        con = duckdb.connect(str(workdir / "store.db"))
        con.execute("INSTALL vss; LOAD vss;")
        con.execute("INSTALL fts; LOAD fts;")
        con.execute("SET hnsw_enable_experimental_persistence = true;")
        con.execute(f"CREATE TABLE chunks (cid VARCHAR, text VARCHAR, emb FLOAT[{self.dim}])")
        con.execute("CREATE TABLE edges (src VARCHAR, dst VARCHAR)")
        con.execute("CREATE TABLE mentions (cid VARCHAR, eid VARCHAR)")
        self._con = con

    def ingest(self, wl: Workload) -> IngestStats:
        t0 = time.perf_counter()
        con = self._con
        emb = wl.chunk_emb
        rows = 0
        batch = []
        for c in wl.chunks:
            batch.append((c.id, c.text, [float(x) for x in emb[c.vec_idx]]))
            if len(batch) >= _BATCH:
                con.executemany("INSERT INTO chunks VALUES (?, ?, ?)", batch)
                rows += len(batch)
                batch = []
        if batch:
            con.executemany("INSERT INTO chunks VALUES (?, ?, ?)", batch)
            rows += len(batch)
        con.executemany("INSERT INTO edges VALUES (?, ?)", [(e.src, e.dst) for e in wl.edges])
        rows += len(wl.edges)
        ments = [(c.id, eid) for c in wl.chunks for eid in c.entity_ids]
        con.executemany("INSERT INTO mentions VALUES (?, ?)", ments)
        rows += len(ments)
        return IngestStats(rows=rows, seconds=time.perf_counter() - t0)

    def build_indexes(self) -> BuildStats:
        t0 = time.perf_counter()
        self._con.execute("CREATE INDEX hnsw_idx ON chunks USING HNSW (emb) WITH (metric = 'cosine')")
        self._con.execute("PRAGMA create_fts_index('chunks', 'cid', 'text', overwrite = 1)")
        self._con.execute("CREATE INDEX idx_edges_src ON edges (src)")
        self._con.execute("CREATE INDEX idx_edges_dst ON edges (dst)")
        self._con.execute("CREATE INDEX idx_ment_eid ON mentions (eid)")
        return BuildStats(seconds=time.perf_counter() - t0, disk_bytes=dir_size_bytes(self._dir))

    # --- component retrievals -------------------------------------------
    def search_vector(self, spec: QuerySpec) -> list[str]:
        v = [float(x) for x in spec.vector]
        rows = self._con.execute(
            f"SELECT cid FROM chunks ORDER BY array_cosine_distance(emb, ?::FLOAT[{self.dim}]) LIMIT ?",
            (v, spec.pool_n),
        ).fetchall()
        return [r[0] for r in rows]

    def search_fts(self, spec: QuerySpec) -> list[str]:
        # match_bm25 is disjunctive by default (conjunctive := false).
        rows = self._con.execute(
            "SELECT cid FROM (SELECT cid, fts_main_chunks.match_bm25(cid, ?) AS score FROM chunks) "
            "WHERE score IS NOT NULL ORDER BY score DESC LIMIT ?",
            (spec.text, spec.pool_n),
        ).fetchall()
        return [r[0] for r in rows]

    def search_graph(self, spec: QuerySpec) -> list[str]:
        con = self._con

        def neighbors(frontier: list[str]) -> list[str]:
            ph = ",".join("?" * len(frontier))
            rows = con.execute(
                f"SELECT dst FROM edges WHERE src IN ({ph}) "
                f"UNION SELECT src FROM edges WHERE dst IN ({ph})",
                [*frontier, *frontier],
            ).fetchall()
            return [r[0] for r in rows]

        def chunks_of(entities: list[str]) -> list[str]:
            ph = ",".join("?" * len(entities))
            rows = con.execute(
                f"SELECT DISTINCT cid FROM mentions WHERE eid IN ({ph})", entities
            ).fetchall()
            return [r[0] for r in rows]

        return bfs_chunk_ranking(neighbors, chunks_of, spec.seed_entity, spec.hops, spec.pool_n)

    def teardown(self) -> None:
        try:
            self._con.close()
        except Exception:  # noqa: BLE001
            pass
