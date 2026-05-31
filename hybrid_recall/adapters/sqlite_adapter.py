"""SQLite adapter — sqlite-vec (vector) + FTS5 (BM25) + recursive-CTE graph.

SQLite serves all three signals *in one file*, but:
  * sqlite-vec does a brute-force exact KNN scan (no ANN index) — perfect recall, but
    query cost grows linearly with the corpus.
  * there is no native fusion: the application runs three queries and fuses with RRF.
  * "graph" is expressible only via recursive CTEs over edge tables.
"""

from __future__ import annotations

import time
from pathlib import Path

import apsw
import sqlite_vec

from ..graphwalk import bfs_chunk_ranking
from ..workload.schema import Workload, WorkloadMeta
from .base import Adapter, BuildStats, Capabilities, IngestStats, QuerySpec, dir_size_bytes

_BATCH = 5000

# We use apsw rather than the stdlib sqlite3 because many CPython builds (incl. the
# python.org macOS builds) are compiled without loadable-extension support, which
# sqlite-vec requires. apsw always supports it.


class SQLiteAdapter(Adapter):
    name = "sqlite"
    thread_safe = False  # one sqlite3 connection, single-threaded
    capabilities = Capabilities(
        vector=True,
        fts=True,
        graph=True,
        native_fusion=False,
        fusion_locus="app-side",
        engines_needed=1,
        embedded=True,
        transactional=True,
        time_travel=False,
        incremental_index=True,  # vec0 + FTS5 are updated by the INSERT itself
        notes="sqlite-vec is a brute-force exact KNN scan (no ANN index); FTS5 for BM25; "
        "graph via recursive CTEs; fusion done in application code.",
    )

    def setup(self, meta: WorkloadMeta, workdir: Path) -> None:
        self.meta = meta
        self.dim = meta.dim
        self._dir = workdir
        self._path = workdir / "store.db"
        self.version = sqlite_vec.__version__ if hasattr(sqlite_vec, "__version__") else ""
        con = apsw.Connection(str(self._path))
        con.enableloadextension(True)
        con.loadextension(sqlite_vec.loadable_path())
        con.enableloadextension(False)
        con.execute(f"CREATE VIRTUAL TABLE vchunks USING vec0(embedding float[{self.dim}])")
        con.execute("CREATE TABLE chunks (rid INTEGER PRIMARY KEY, cid TEXT, text TEXT)")
        con.execute("CREATE VIRTUAL TABLE fchunks USING fts5(text, content='')")
        con.execute("CREATE TABLE edges (src TEXT, dst TEXT)")
        con.execute("CREATE TABLE mentions (cid TEXT, eid TEXT)")
        self._con = con

    def ingest(self, wl: Workload) -> IngestStats:
        t0 = time.perf_counter()
        con = self._con
        rows = 0
        emb = wl.chunk_emb
        cur = con.cursor()
        vbatch, cbatch, fbatch = [], [], []
        with con:  # single transaction (apsw autocommits per-statement otherwise — slow)
            for c in wl.chunks:
                rid = c.vec_idx
                vbatch.append((rid, sqlite_vec.serialize_float32([float(x) for x in emb[c.vec_idx]])))
                cbatch.append((rid, c.id, c.text))
                fbatch.append((rid, c.text))
                if len(vbatch) >= _BATCH:
                    self._flush(cur, vbatch, cbatch, fbatch)
                    rows += len(vbatch)
                    vbatch, cbatch, fbatch = [], [], []
            if vbatch:
                self._flush(cur, vbatch, cbatch, fbatch)
                rows += len(vbatch)
            cur.executemany(
                "INSERT INTO edges (src, dst) VALUES (?, ?)", [(e.src, e.dst) for e in wl.edges]
            )
            rows += len(wl.edges)
            ments = [(c.id, eid) for c in wl.chunks for eid in c.entity_ids]
            cur.executemany("INSERT INTO mentions (cid, eid) VALUES (?, ?)", ments)
            rows += len(ments)
        self._next_rid = len(wl.chunks)  # next free rowid for incremental upserts
        return IngestStats(rows=rows, seconds=time.perf_counter() - t0)

    @staticmethod
    def _flush(cur, vbatch, cbatch, fbatch) -> None:
        cur.executemany("INSERT INTO vchunks (rowid, embedding) VALUES (?, ?)", vbatch)
        cur.executemany("INSERT INTO chunks (rid, cid, text) VALUES (?, ?, ?)", cbatch)
        cur.executemany("INSERT INTO fchunks (rowid, text) VALUES (?, ?)", fbatch)

    def build_indexes(self) -> BuildStats:
        t0 = time.perf_counter()
        # vec0 has no separate index build (brute force); finalize FTS + b-tree indexes.
        self._con.execute("INSERT INTO fchunks (fchunks) VALUES ('optimize')")
        self._con.execute("CREATE INDEX idx_edges_src ON edges (src)")
        self._con.execute("CREATE INDEX idx_edges_dst ON edges (dst)")
        self._con.execute("CREATE INDEX idx_ment_eid ON mentions (eid)")
        return BuildStats(seconds=time.perf_counter() - t0, disk_bytes=dir_size_bytes(self._dir))

    # --- component retrievals -------------------------------------------
    def search_vector(self, spec: QuerySpec) -> list[str]:
        q = sqlite_vec.serialize_float32([float(x) for x in spec.vector])
        rows = self._con.execute(
            "SELECT c.cid FROM vchunks v JOIN chunks c ON c.rid = v.rowid "
            "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
            (q, spec.pool_n),
        )
        return [r[0] for r in rows]

    def search_fts(self, spec: QuerySpec) -> list[str]:
        match = " OR ".join(f'"{t}"' for t in spec.text.split())
        rows = self._con.execute(
            "SELECT c.cid FROM fchunks f JOIN chunks c ON c.rid = f.rowid "
            "WHERE fchunks MATCH ? ORDER BY bm25(fchunks) LIMIT ?",
            (match, spec.pool_n),
        )
        return [r[0] for r in rows]

    def search_graph(self, spec: QuerySpec) -> list[str]:
        con = self._con

        def neighbors(frontier: list[str]) -> list[str]:
            ph = ",".join("?" * len(frontier))
            rows = con.execute(
                f"SELECT dst FROM edges WHERE src IN ({ph}) "
                f"UNION SELECT src FROM edges WHERE dst IN ({ph})",
                (*frontier, *frontier),
            )
            return [r[0] for r in rows]

        def chunks_of(entities: list[str]) -> list[str]:
            ph = ",".join("?" * len(entities))
            rows = con.execute(f"SELECT DISTINCT cid FROM mentions WHERE eid IN ({ph})", entities)
            return [r[0] for r in rows]

        return bfs_chunk_ranking(neighbors, chunks_of, spec.seed_entity, spec.hops, spec.pool_n)

    def upsert_memory(self, cid: str, text: str, vector, entity_ids: list[str]) -> None:
        con = self._con
        rid = self._next_rid
        self._next_rid += 1
        with con:
            con.execute(
                "INSERT INTO vchunks (rowid, embedding) VALUES (?, ?)",
                (rid, sqlite_vec.serialize_float32([float(x) for x in vector])),
            )
            con.execute("INSERT INTO chunks (rid, cid, text) VALUES (?, ?, ?)", (rid, cid, text))
            con.execute("INSERT INTO fchunks (rowid, text) VALUES (?, ?)", (rid, text))
            con.executemany(
                "INSERT INTO mentions (cid, eid) VALUES (?, ?)", [(cid, e) for e in entity_ids]
            )

    def teardown(self) -> None:
        try:
            self._con.close()
        except Exception:  # noqa: BLE001
            pass
