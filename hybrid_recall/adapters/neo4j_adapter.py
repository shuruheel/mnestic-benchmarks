"""Neo4j adapter — PHASE 2 (server engine).

Neo4j 5.x serves all three signals on a single server: native vector index, a Lucene
full-text index, and Cypher graph traversal. Fusion is app-side.

Requires a running Neo4j (see docker/docker-compose.yml). Configured via env:
    NEO4J_URI (default bolt://localhost:7687), NEO4J_USER, NEO4J_PASSWORD.

This adapter is implemented but is exercised in phase 2 (it is not part of the embedded
reference run). Reported caveats and methodology live in docs/COMPARISON.md.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from ..graphwalk import bfs_chunk_ranking
from ..workload.schema import Workload, WorkloadMeta
from .base import Adapter, BuildStats, Capabilities, IngestStats, QuerySpec

_BATCH = 5000


class Neo4jAdapter(Adapter):
    name = "neo4j"
    thread_safe = True
    capabilities = Capabilities(
        vector=True,
        fts=True,
        graph=True,
        native_fusion=False,
        fusion_locus="app-side",
        engines_needed=1,
        embedded=False,
        notes="Server engine: native vector index + Lucene full-text + Cypher traversal; "
        "app-side fusion. PHASE 2 — run via docker/docker-compose.yml.",
    )

    def setup(self, meta: WorkloadMeta, workdir: Path) -> None:
        from neo4j import GraphDatabase  # noqa: PLC0415

        self.meta = meta
        self.dim = meta.dim
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        pw = os.environ.get("NEO4J_PASSWORD", "password")
        self._driver = GraphDatabase.driver(uri, auth=(user, pw))
        self._driver.verify_connectivity()
        import neo4j  # noqa: PLC0415

        self.version = neo4j.__version__
        with self._driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
            s.run("CREATE CONSTRAINT chunk_cid IF NOT EXISTS FOR (c:Chunk) REQUIRE c.cid IS UNIQUE")
            s.run("CREATE CONSTRAINT ent_eid IF NOT EXISTS FOR (e:Entity) REQUIRE e.eid IS UNIQUE")

    def ingest(self, wl: Workload) -> IngestStats:
        t0 = time.perf_counter()
        emb = wl.chunk_emb
        rows = 0
        with self._driver.session() as s:
            ents = [{"eid": e.id} for e in wl.entities]
            for i in range(0, len(ents), _BATCH):
                s.run("UNWIND $rows AS r CREATE (:Entity {eid: r.eid})", rows=ents[i : i + _BATCH])
            rows += len(ents)
            chunks = [
                {"cid": c.id, "text": c.text, "emb": [float(x) for x in emb[c.vec_idx]]}
                for c in wl.chunks
            ]
            for i in range(0, len(chunks), _BATCH):
                s.run(
                    "UNWIND $rows AS r CREATE (:Chunk {cid: r.cid, text: r.text, emb: r.emb})",
                    rows=chunks[i : i + _BATCH],
                )
            rows += len(chunks)
            edges = [{"src": e.src, "dst": e.dst} for e in wl.edges]
            for i in range(0, len(edges), _BATCH):
                s.run(
                    "UNWIND $rows AS r MATCH (a:Entity {eid: r.src}), (b:Entity {eid: r.dst}) "
                    "CREATE (a)-[:REL]->(b)",
                    rows=edges[i : i + _BATCH],
                )
            rows += len(edges)
            ments = [{"cid": c.id, "eid": eid} for c in wl.chunks for eid in c.entity_ids]
            for i in range(0, len(ments), _BATCH):
                s.run(
                    "UNWIND $rows AS r MATCH (c:Chunk {cid: r.cid}), (e:Entity {eid: r.eid}) "
                    "CREATE (c)-[:MENTIONS]->(e)",
                    rows=ments[i : i + _BATCH],
                )
            rows += len(ments)
        return IngestStats(rows=rows, seconds=time.perf_counter() - t0)

    def build_indexes(self) -> BuildStats:
        t0 = time.perf_counter()
        with self._driver.session() as s:
            s.run(
                "CREATE VECTOR INDEX chunk_vec IF NOT EXISTS FOR (c:Chunk) ON (c.emb) "
                "OPTIONS {indexConfig: {`vector.dimensions`: $d, `vector.similarity_function`: 'cosine'}}",
                d=self.dim,
            )
            s.run("CREATE FULLTEXT INDEX chunk_fts IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]")
            s.run("CALL db.awaitIndexes(300)")
        return BuildStats(seconds=time.perf_counter() - t0, disk_bytes=0)

    def search_vector(self, spec: QuerySpec) -> list[str]:
        with self._driver.session() as s:
            res = s.run(
                "CALL db.index.vector.queryNodes('chunk_vec', $k, $q) YIELD node RETURN node.cid AS cid",
                k=spec.pool_n,
                q=[float(x) for x in spec.vector],
            )
            return [r["cid"] for r in res]

    def search_fts(self, spec: QuerySpec) -> list[str]:
        # Lucene query: OR the terms (disjunctive, ranked by Lucene's BM25-like score).
        q = " OR ".join(spec.text.split())
        with self._driver.session() as s:
            res = s.run(
                "CALL db.index.fulltext.queryNodes('chunk_fts', $q) YIELD node RETURN node.cid AS cid "
                "LIMIT $k",
                q=q,
                k=spec.pool_n,
            )
            return [r["cid"] for r in res]

    def search_graph(self, spec: QuerySpec) -> list[str]:
        sess = self._driver.session()

        def neighbors(frontier: list[str]) -> list[str]:
            res = sess.run(
                "UNWIND $f AS x MATCH (a:Entity {eid: x})-[:REL]-(b:Entity) RETURN DISTINCT b.eid AS eid",
                f=frontier,
            )
            return [r["eid"] for r in res]

        def chunks_of(entities: list[str]) -> list[str]:
            res = sess.run(
                "UNWIND $e AS x MATCH (c:Chunk)-[:MENTIONS]->(:Entity {eid: x}) RETURN DISTINCT c.cid AS cid",
                e=entities,
            )
            return [r["cid"] for r in res]

        try:
            return bfs_chunk_ranking(neighbors, chunks_of, spec.seed_entity, spec.hops, spec.pool_n)
        finally:
            sess.close()

    def teardown(self) -> None:
        try:
            self._driver.close()
        except Exception:  # noqa: BLE001
            pass
