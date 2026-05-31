"""mnestic adapter — a maintained fork of CozoDB (relational-graph-vector + Datalog).

mnestic is the only engine here that natively serves all three signals *in one embedded
store*: HNSW vector search, FTS full-text, and Datalog graph traversal — plus a single
`hybrid_search` call that fuses vector+FTS with RRF inside the engine.

For the scored recall comparison we use the three component queries and the canonical RRF
(like every other engine, for fairness). `native_hybrid` additionally exercises the
single-call `hybrid_search` so its latency/ergonomics can be reported separately.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from ..workload.schema import Workload, WorkloadMeta
from .base import Adapter, BuildStats, Capabilities, IngestStats, QuerySpec, dir_size_bytes

_BATCH = 2000


class MnesticAdapter(Adapter):
    name = "mnestic"
    thread_safe = True  # CozoDbPy releases the GIL on run_script
    capabilities = Capabilities(
        vector=True,
        fts=True,
        graph=True,
        native_fusion=True,
        fusion_locus="native",
        engines_needed=1,
        embedded=True,
        transactional=True,
        time_travel=True,   # `@` validity / as-of queries
        incremental_index=True,  # per-row :put maintains HNSW + FTS live
        notes="Single embedded engine for vector + full-text + Datalog graph; "
        "hybrid_search() fuses vector+FTS in one call; time-travel via Validity.",
    )

    def setup(self, meta: WorkloadMeta, workdir: Path) -> None:
        import mnestic  # noqa: PLC0415

        self.meta = meta
        self.dim = meta.dim
        self._dir = workdir
        # Backend is selectable via MNESTIC_BACKEND (default: sqlite). The RocksDB backend is
        # the one mindgraph-rs actually runs on, so latency claims that gate the bridge-level
        # perf items (#4 batched index put, #7 HNSW multi_get) must be re-measured on it.
        # RocksDB wants a *directory* path; sqlite wants a *file*.
        backend = os.environ.get("MNESTIC_BACKEND", "sqlite").lower()
        if backend == "rocksdb":
            path = str(workdir / "store.rocksdb")
        else:
            path = str(workdir / "store.db")
        self._backend = backend
        self._db = mnestic.CozoDbPy(backend, path, "{}")
        try:
            base = getattr(mnestic, "__version__", "") or ""
        except Exception:  # noqa: BLE001
            base = ""
        self.version = f"{base}({backend})" if base else f"({backend})"
        self._run(f":create chunk {{ cid: String => text: String, emb: <F32; {self.dim}> }}")
        self._run(":create edge { src: String, dst: String }")
        self._run(":create mention { cid: String, eid: String }")
        # Unified traversal relation for the native 3-way hybrid_search graph leg: entity↔entity
        # edges (both directions) + entity→chunk mentions (one-way, so chunks are leaves). A
        # chunk's hop from a seed entity is then (entity-hop + 1) — monotonic with the oracle's
        # entity-hop ordering — so max_hops = graph_hops + 1 reaches all in-range chunks.
        self._run(":create link { src: String, dst: String }")

    def _run(self, q: str, params: dict | None = None, immutable: bool = False):
        return self._db.run_script(q, params or {}, immutable)

    def ingest(self, wl: Workload) -> IngestStats:
        t0 = time.perf_counter()
        rows = 0
        # chunks
        emb = wl.chunk_emb
        batch = []
        for c in wl.chunks:
            batch.append([c.id, c.text, [float(x) for x in emb[c.vec_idx]]])
            if len(batch) >= _BATCH:
                self._run("?[cid,text,emb] <- $r :put chunk {cid=>text,emb}", {"r": batch})
                rows += len(batch)
                batch = []
        if batch:
            self._run("?[cid,text,emb] <- $r :put chunk {cid=>text,emb}", {"r": batch})
            rows += len(batch)
        # edges
        self._put_pairs("?[src,dst] <- $r :put edge {src,dst}", [[e.src, e.dst] for e in wl.edges])
        rows += len(wl.edges)
        # mentions (chunk -> entity)
        mentions = [[c.id, eid] for c in wl.chunks for eid in c.entity_ids]
        self._put_pairs("?[cid,eid] <- $r :put mention {cid,eid}", mentions)
        rows += len(mentions)
        # link relation for the native graph leg: entity edges both directions + entity->chunk
        link = [[e.src, e.dst] for e in wl.edges] + [[e.dst, e.src] for e in wl.edges]
        link += [[eid, cid] for cid, eid in mentions]
        self._put_pairs("?[src,dst] <- $r :put link {src,dst}", link)
        rows += len(link)
        return IngestStats(rows=rows, seconds=time.perf_counter() - t0)

    def _put_pairs(self, script: str, pairs: list[list[str]]) -> None:
        for i in range(0, len(pairs), _BATCH):
            self._run(script, {"r": pairs[i : i + _BATCH]})

    def build_indexes(self) -> BuildStats:
        t0 = time.perf_counter()
        self._run(
            f"::hnsw create chunk:vec {{ dim: {self.dim}, m: 16, dtype: F32, "
            f"fields: [emb], distance: Cosine, ef_construction: 64 }}"
        )
        # Lowercase only (no stemming/stopwords): the corpus is a synthetic vocabulary, so
        # we keep tokenization aligned with the oracle's plain whitespace tokenization.
        self._run("::fts create chunk:fts { extractor: text, tokenizer: Simple, filters: [Lowercase] }")
        secs = time.perf_counter() - t0
        return BuildStats(seconds=secs, disk_bytes=dir_size_bytes(self._dir))

    # --- component retrievals -------------------------------------------
    def search_vector(self, spec: QuerySpec) -> list[str]:
        ef = max(64, spec.pool_n)
        r = self._run(
            "?[cid, dist] := ~chunk:vec{cid | query: q, k: $k, ef: $ef, bind_distance: dist}, "
            "q = vec($qv) :order dist :limit $k",
            {"qv": [float(x) for x in spec.vector], "k": spec.pool_n, "ef": ef},
            True,
        )
        return [row[0] for row in r["rows"]]

    def search_fts(self, spec: QuerySpec) -> list[str]:
        # Single multi-term disjunctive BM25 query. mnestic's FTS now sums per-term BM25
        # contributions for an `a OR b` query (BM25 is the default scorer, fork 0.8.x), so
        # one call returns the proper full-BM25 top-k — no per-term decomposition or
        # over-fetch (the earlier workaround was needed only when `OR` took the max score).
        q = " OR ".join(dict.fromkeys(spec.text.split()))
        r = self._run(
            "?[cid, score] := ~chunk:fts{cid | query: $q, k: $k, bind_score: score} "
            ":order -score :limit $k",
            {"q": q, "k": spec.pool_n},
            True,
        )
        return [row[0] for row in r["rows"]]

    def search_graph(self, spec: QuerySpec) -> list[str]:
        # k-hop entities with min hop distance in ONE recursive Datalog query (undirected),
        # then one query for the chunks of those entities. Two engine round-trips total —
        # far cheaper than level-by-level BFS, and reproduces the oracle's (hop, cid) order.
        rules = ["h1[eid] := *edge{src: $s, dst: eid}", "h1[eid] := *edge{src: eid, dst: $s}"]
        for d in range(2, spec.hops + 1):
            rules.append(f"h{d}[eid] := h{d - 1}[m], *edge{{src: m, dst: eid}}")
            rules.append(f"h{d}[eid] := h{d - 1}[m], *edge{{src: eid, dst: m}}")
        for d in range(1, spec.hops + 1):
            rules.append(f"?[eid, hop] := h{d}[eid], hop = {d}")
        r = self._run("\n".join(rules), {"s": spec.seed_entity}, True)
        emap: dict[str, int] = {spec.seed_entity: 0}
        for eid, hop in r["rows"]:
            if eid not in emap or hop < emap[eid]:
                emap[eid] = int(hop)
        c = self._run(
            "ent[eid, hop] <- $rows\n?[cid, min(hop)] := ent[eid, hop], *mention{cid, eid}",
            {"rows": [[e, h] for e, h in emap.items()]},
            True,
        )
        ranked = sorted(((row[0], int(row[1])) for row in c["rows"]), key=lambda kv: (kv[1], kv[0]))
        return [cid for cid, _ in ranked[: spec.pool_n]]

    def native_hybrid(self, spec: QuerySpec) -> list[str] | None:
        # Native 3-way fusion in ONE call (Bet 1a): vector + FTS + a graph-proximity leg
        # over the unified `link` relation, all RRF-fused inside the engine.
        res = self._db.hybrid_search(
            {
                "relation": "chunk",
                "id_col": "cid",
                "vector_index": "vec",
                "query_vector": [float(x) for x in spec.vector],
                "vector_k": spec.pool_n,
                "ef": max(64, spec.pool_n),
                "fts_index": "fts",
                "query_text": " OR ".join(dict.fromkeys(spec.text.split())),
                "fts_k": spec.pool_n,
                "graph_legs": [
                    {
                        "label": "graph",
                        "edge_relation": "link",
                        "from_col": "src",
                        "to_col": "dst",
                        "seeds": [spec.seed_entity],
                        "max_hops": spec.hops + 1,  # +1 for the entity->chunk leaf hop
                        "undirected": False,  # edges already stored both ways; chunks stay leaves
                    }
                ],
                "rrf_k": spec.rrf_k,
                "limit": spec.k,
            }
        )
        return [row[0] for row in res["rows"]][: spec.k]

    def upsert_memory(self, cid: str, text: str, vector, entity_ids: list[str]) -> None:
        self._run(
            "?[cid, text, emb] <- $r :put chunk {cid => text, emb}",
            {"r": [[cid, text, [float(x) for x in vector]]]},
        )
        if entity_ids:
            self._run(
                "?[cid, eid] <- $r :put mention {cid, eid}",
                {"r": [[cid, eid] for eid in entity_ids]},
            )
            self._run(
                "?[src, dst] <- $r :put link {src, dst}",
                {"r": [[eid, cid] for eid in entity_ids]},
            )

    def teardown(self) -> None:
        try:
            self._db.close()
        except Exception:  # noqa: BLE001
            pass
