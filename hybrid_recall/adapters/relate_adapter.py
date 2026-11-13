"""Relate adapter — Neo4j Labs' embedded, in-memory property-graph engine (Rust).

Relate (`pip install relate-db`, import `relate`; Apache-2.0; an *experimental* Neo4j Labs
project) serves all three signals in one embedded store through Cypher: a vector index,
a full-text index, and native graph traversal, with a WAL-backed directory store.

Architecture facts that matter when reading its numbers (relatedb/relate@ce8abb6):

- The live graph is held in RAM ("the working set must fit in memory"; no paging/spill).
  Durability is a fsync'd WAL plus a snapshot codec (default durability mode `full`).
- `SEARCH ... VECTOR INDEX` is an **exact scan** of the index population with a bounded
  top-k window, not an ANN graph — so its vector recall should match the oracle exactly,
  and its latency should grow linearly with the corpus. Filters are applied per scored
  candidate, so a filtered search is complete (no ANN candidate-window loss).
- `SEARCH ... FULLTEXT INDEX` builds documents and corpus statistics per search.
- `COMBINE (...) USING WRRF` (preview, `RELATE_PREVIEW=hybrid-search`) fuses vector+FTS
  in one call; measured separately as `native_hybrid`, like every engine's fast path.

For the scored recall comparison we use the three component queries and the canonical
RRF, exactly as for every other engine.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from ..workload.schema import Workload, WorkloadMeta, chunk_kind
from .base import Adapter, BuildStats, Capabilities, IngestStats, QuerySpec, dir_size_bytes

_BATCH = 5000
# Relate caps one WAL record at 64 MB per statement; real-text chunks with 384-d vectors
# exceed that at 5,000 rows, so chunk rows are written in smaller batches.
_CHUNK_BATCH = 500


def _vec(v) -> list[float]:
    return [float(x) for x in v]


class RelateAdapter(Adapter):
    name = "relate"
    # relate.threadsafety == 1 (PEP 249: threads may share the module, not connections),
    # so the concurrent-QPS phase is skipped rather than measured unfairly.
    thread_safe = False
    capabilities = Capabilities(
        vector=True,
        fts=True,
        graph=True,
        native_fusion=True,     # COMBINE ... USING WRRF (preview gate), vector+FTS only
        fusion_locus="native",
        engines_needed=1,
        embedded=True,
        transactional=True,     # MVCC transactions over a WAL
        time_travel=False,      # snapshot isolation only; no as-of / historical query surface
        incremental_index=True,  # indexes are scanned at query time, so writes are visible
        notes="Experimental Neo4j Labs engine: whole graph in RAM, WAL + snapshot durability; "
        "vector search is an exact scan; native WRRF fusion of vector+FTS is a preview.",
    )

    def setup(self, meta: WorkloadMeta, workdir: Path) -> None:
        # The preview gate is read from the environment; set it before opening.
        os.environ.setdefault("RELATE_PREVIEW", "hybrid-search")
        import relate  # noqa: PLC0415

        self.meta = meta
        self.dim = meta.dim
        self._dir = workdir / "store"
        self._db = relate.connect(str(self._dir))
        try:
            from importlib.metadata import version  # noqa: PLC0415

            self.version = version("relate-db")
        except Exception:  # noqa: BLE001
            self.version = "unknown"
        # Key lookups used by the edge/mention loads and upserts. These play the role of the
        # primary keys the relational engines get for free, so they are created up front and
        # not counted as index build time (same treatment as the other adapters' PKs).
        self._run("CREATE INDEX entity_eid IF NOT EXISTS FOR (e:Entity) ON (e.eid)")
        self._run("CREATE INDEX chunk_cid IF NOT EXISTS FOR (c:Chunk) ON (c.cid)")

    def _run(self, q: str, params: dict | None = None) -> list[dict]:
        return list(self._db.execute(q, params or {}))

    def _batched(self, q: str, rows: list[dict], size: int = _BATCH) -> None:
        for i in range(0, len(rows), size):
            self._run(q, {"rows": rows[i : i + size]})

    # --- lifecycle -------------------------------------------------------
    def ingest(self, wl: Workload) -> IngestStats:
        t0 = time.perf_counter()
        emb = wl.chunk_emb
        self._batched(
            "UNWIND $rows AS r CREATE (:Entity {eid: r.eid})", [{"eid": e.id} for e in wl.entities]
        )
        chunks = [
            {"cid": c.id, "text": c.text, "emb": _vec(emb[c.vec_idx]), "kind": chunk_kind(c.id)}
            for c in wl.chunks
        ]
        self._batched(
            f"UNWIND $rows AS r CREATE (:Chunk {{cid: r.cid, text: r.text, kind: r.kind, "
            f"emb: vector(r.emb, {self.dim}, FLOAT32)}})",
            chunks,
            _CHUNK_BATCH,
        )
        self._batched(
            "UNWIND $rows AS r MATCH (a:Entity {eid: r.src}), (b:Entity {eid: r.dst}) "
            "CREATE (a)-[:REL]->(b)",
            [{"src": e.src, "dst": e.dst} for e in wl.edges],
        )
        mentions = [{"cid": c.id, "eid": eid} for c in wl.chunks for eid in c.entity_ids]
        self._batched(
            "UNWIND $rows AS r MATCH (c:Chunk {cid: r.cid}), (e:Entity {eid: r.eid}) "
            "CREATE (c)-[:MENTIONS]->(e)",
            mentions,
        )
        rows = len(wl.entities) + len(chunks) + len(wl.edges) + len(mentions)
        return IngestStats(rows=rows, seconds=time.perf_counter() - t0)

    def build_indexes(self) -> BuildStats:
        t0 = time.perf_counter()
        self._run(
            "CREATE VECTOR INDEX chunk_vec FOR (c:Chunk) ON (c.emb) WITH [c.kind] "
            f"OPTIONS {{indexConfig: {{'vector.dimensions': {self.dim}, "
            "'vector.similarity_function': 'cosine'}}"
        )
        self._run("CREATE FULLTEXT INDEX chunk_fts FOR (c:Chunk) ON EACH [c.text]")
        self._db.flush()
        secs = time.perf_counter() - t0
        return BuildStats(seconds=secs, disk_bytes=dir_size_bytes(self._dir))

    # --- component retrievals -------------------------------------------
    def search_vector(self, spec: QuerySpec) -> list[str]:
        rows = self._run(
            f"MATCH (c:Chunk) SEARCH c IN (VECTOR INDEX chunk_vec FOR "
            f"vector($q, {self.dim}, FLOAT32) LIMIT $k) RETURN c.cid AS cid",
            {"q": _vec(spec.vector), "k": spec.pool_n},
        )
        return [r["cid"] for r in rows]

    def search_fts(self, spec: QuerySpec) -> list[str]:
        q = " OR ".join(dict.fromkeys(spec.text.split()))
        rows = self._run(
            "MATCH (c:Chunk) SEARCH c IN (FULLTEXT INDEX chunk_fts FOR $q LIMIT $k) "
            "RETURN c.cid AS cid",
            {"q": q, "k": spec.pool_n},
        )
        return [r["cid"] for r in rows]

    def search_graph(self, spec: QuerySpec) -> list[str]:
        # Same two-query shape as the Kuzu adapter: hop-0 chunks of the seed, then chunks
        # of entities within `hops` undirected REL hops, ranked by (min hop, cid) to match
        # the oracle's graph component exactly.
        chunk_hop: dict[str, int] = {}
        for r in self._run(
            "MATCH (s:Entity {eid: $s})<-[:MENTIONS]-(c:Chunk) RETURN c.cid AS cid",
            {"s": spec.seed_entity},
        ):
            chunk_hop.setdefault(r["cid"], 0)
        for r in self._run(
            f"MATCH p = (s:Entity {{eid: $s}})-[:REL*1..{spec.hops}]-(n:Entity)"
            "<-[:MENTIONS]-(c:Chunk) RETURN c.cid AS cid, min(length(p)) - 1 AS hop",
            {"s": spec.seed_entity},
        ):
            cid, hop = r["cid"], int(r["hop"])
            if cid not in chunk_hop or hop < chunk_hop[cid]:
                chunk_hop[cid] = hop
        ranked = sorted(chunk_hop.items(), key=lambda kv: (kv[1], kv[0]))
        return [cid for cid, _ in ranked[: spec.pool_n]]

    def native_hybrid(self, spec: QuerySpec) -> list[str] | None:
        q = " OR ".join(dict.fromkeys(spec.text.split()))
        rows = self._run(
            "MATCH (c:Chunk) SEARCH c IN (COMBINE ("
            f"VECTOR INDEX chunk_vec FOR vector($qv, {self.dim}, FLOAT32) LIMIT $pool WEIGHT 1, "
            "FULLTEXT INDEX chunk_fts FOR $qt LIMIT $pool WEIGHT 1"
            f") USING WRRF WITH CONSTANT {float(spec.rrf_k)} LIMIT $k) RETURN c.cid AS cid",
            {"qv": _vec(spec.vector), "qt": q, "pool": spec.pool_n, "k": spec.k},
        )
        return [r["cid"] for r in rows]

    # --- filtered vector axis -------------------------------------------
    # search_filter is the in-index WHERE on a `WITH [c.kind]` property (applied per scored
    # candidate inside the exact scan); exact_scan is the plain Cypher equivalent.
    filter_strategies = ("search_filter", "exact_scan")

    def search_vector_filtered(self, spec: QuerySpec, kind: str, strategy: str) -> list[str]:
        params = {"q": _vec(spec.vector), "kind": kind, "k": spec.k}
        if strategy == "search_filter":
            q = (
                f"MATCH (c:Chunk) SEARCH c IN (VECTOR INDEX chunk_vec FOR "
                f"vector($q, {self.dim}, FLOAT32) WHERE c.kind = $kind LIMIT $k) "
                "RETURN c.cid AS cid"
            )
        elif strategy == "exact_scan":
            q = (
                "MATCH (c:Chunk) WHERE c.kind = $kind "
                f"WITH c, vector.similarity.cosine(c.emb, vector($q, {self.dim}, FLOAT32)) AS s "
                "RETURN c.cid AS cid ORDER BY s DESC LIMIT $k"
            )
        else:
            raise ValueError(strategy)
        return [r["cid"] for r in self._run(q, params)]

    # --- writes ----------------------------------------------------------
    def upsert_memory(self, cid: str, text: str, vector, entity_ids: list[str]) -> None:
        self._run(
            f"MERGE (c:Chunk {{cid: $cid}}) SET c.text = $text, c.kind = $kind, "
            f"c.emb = vector($emb, {self.dim}, FLOAT32) "
            "WITH c UNWIND $eids AS eid MATCH (e:Entity {eid: eid}) MERGE (c)-[:MENTIONS]->(e)",
            {"cid": cid, "text": text, "kind": chunk_kind(cid), "emb": _vec(vector),
             "eids": list(entity_ids)},
        )

    # --- lifecycle axis --------------------------------------------------
    def reopen(self, meta: WorkloadMeta, workdir: Path) -> None:
        os.environ.setdefault("RELATE_PREVIEW", "hybrid-search")
        import relate  # noqa: PLC0415

        self.meta = meta
        self.dim = meta.dim
        self._dir = workdir / "store"
        if not self._dir.exists():
            raise FileNotFoundError(self._dir)
        self._db = relate.connect(str(self._dir))
        from importlib.metadata import version  # noqa: PLC0415

        self.version = version("relate-db")

    def update_memory(self, cid: str, text: str, vector) -> None:
        self._run(
            f"MATCH (c:Chunk {{cid: $cid}}) SET c.text = $text, "
            f"c.emb = vector($emb, {self.dim}, FLOAT32)",
            {"cid": cid, "text": text, "emb": _vec(vector)},
        )

    def delete_memory(self, cid: str) -> None:
        self._run("MATCH (c:Chunk {cid: $cid}) DETACH DELETE c", {"cid": cid})

    def has_memory(self, cid: str) -> bool:
        return bool(self._run("MATCH (c:Chunk {cid: $cid}) RETURN c.cid AS cid", {"cid": cid}))

    def teardown(self) -> None:
        try:
            self._db.close()
        except Exception:  # noqa: BLE001
            pass
