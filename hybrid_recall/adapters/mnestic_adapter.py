"""mnestic adapter — a maintained fork of CozoDB (relational-graph-vector + Datalog).

mnestic is the only engine here that natively serves all three signals *in one embedded
store*: HNSW vector search, FTS full-text, and Datalog graph traversal — plus a single
`hybrid_search` call that fuses vector+FTS with RRF inside the engine.

For the scored recall comparison we use the three component queries and the canonical RRF
(like every other engine, for fairness). `native_hybrid` additionally exercises the
single-call `hybrid_search` so its latency/ergonomics can be reported separately.
"""

from __future__ import annotations

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
        notes="Single embedded engine for vector + full-text + Datalog graph; "
        "hybrid_search() fuses vector+FTS in one call.",
    )

    def setup(self, meta: WorkloadMeta, workdir: Path) -> None:
        import mnestic  # noqa: PLC0415

        self.meta = meta
        self.dim = meta.dim
        self._dir = workdir
        self._db = mnestic.CozoDbPy("sqlite", str(workdir / "store.db"), "{}")
        try:
            self.version = getattr(mnestic, "__version__", "")
        except Exception:  # noqa: BLE001
            self.version = ""
        self._run(f":create chunk {{ cid: String => text: String, emb: <F32; {self.dim}> }}")
        self._run(":create edge { src: String, dst: String }")
        self._run(":create mention { cid: String, eid: String }")

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
        # Proper disjunctive ranked FTS: cozo's `a OR b` query does not sum per-term
        # relevance (a 2-term match can tie a 1-term match), so we score each query term
        # against the FTS index and sum per chunk — the idiomatic cozo formulation, in one
        # run_script call. This matches the oracle's disjunctive BM25 ordering.
        terms = list(dict.fromkeys(spec.text.split()))
        # Per-term k a few × pool_n so multi-term docs (whose per-term rank can fall just
        # outside the top pool_n) survive the sum and reproduce the disjunctive BM25 ranking.
        k = max(spec.pool_n, 200)
        r = self._run(
            "term[t] <- $terms\n"
            "m[cid, sum(score)] := term[t], ~chunk:fts{cid | query: t, k: $k, bind_score: score}\n"
            "?[cid, s] := m[cid, s] :order -s :limit $lim",
            {"terms": [[t] for t in terms], "k": k, "lim": spec.pool_n},
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
        res = self._db.hybrid_search(
            {
                "relation": "chunk",
                "id_col": "cid",
                "vector_index": "vec",
                "query_vector": [float(x) for x in spec.vector],
                "vector_k": spec.pool_n,
                "ef": max(64, spec.pool_n),
                "fts_index": "fts",
                "query_text": spec.text,
                "fts_k": spec.pool_n,
                "rrf_k": spec.rrf_k,
                "limit": spec.k,
            }
        )
        return [row[0] for row in res["rows"]][: spec.k]

    def teardown(self) -> None:
        try:
            self._db.close()
        except Exception:  # noqa: BLE001
            pass
