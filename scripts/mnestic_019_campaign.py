"""Frozen 0.19 paired campaign. One process / database / build / backend.

Raw JSONL is flushed per observation, including errors. A missing completion record
means an interrupted run, never a successful shorter run. See docs/0.19-CAMPAIGN.md.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import yaml
from hybrid_recall.adapters.base import QuerySpec, dir_size_bytes
from hybrid_recall.adapters.mnestic_adapter import MnesticAdapter
from hybrid_recall.fusion import rrf_fuse
from hybrid_recall.runner import _specs
from hybrid_recall.workload.generate import generate
from hybrid_recall.workload.groundtruth import _BM25Index, compute_ground_truth


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def observe(fn, truth=None, eligible=None):
    start = time.perf_counter()
    try:
        ids = fn()
        error = None
    except Exception as exc:
        ids, error = [], str(exc)
    result = {"ms": (time.perf_counter() - start) * 1000, "ids": ids, "error": error,
              "count": len(ids), "empty": not ids, "duplicates": len(ids) - len(set(ids))}
    if truth is not None:
        result.update(truth=truth, recall=len(set(ids) & set(truth)) / len(truth) if truth else None)
    if eligible is not None:
        result["ineligible"] = sorted(set(ids) - eligible)
    return result


def write_atomic(a, chunk, vector, text):
    return a._run("{ ?[cid,text,emb] <- [[$id,$text,$v]] :put chunk {cid=>text,emb} } "
                  "{ ?[cid,eid] <- $mentions :put mention {cid,eid} } "
                  "{ ?[src,dst] <- $links :put link {src,dst} }",
                  {"id": chunk.id, "text": text, "v": vector.tolist(),
                   "mentions": [[chunk.id, eid] for eid in chunk.entity_ids],
                   "links": [[eid, chunk.id] for eid in chunk.entity_ids]})


def freshness(a, wl, emit):
    for i, chunk in enumerate(wl.inc_chunks):
        # A unique token lets exact FTS visibility be distinguished from ANN omission.
        token = "freshprobe" + str(i)
        spec = QuerySpec(token, wl.inc_chunk_emb[i], chunk.entity_ids[0], 10, 100, 2, 60)
        start = time.perf_counter()
        write_atomic(a, chunk, wl.inc_chunk_emb[i], chunk.text + " " + token)
        ack = time.perf_counter()
        for name, probe in [
            ("exact", lambda: [r[0] for r in a._run("?[cid] := *chunk{cid}, cid==$id", {"id": chunk.id})["rows"]]),
            ("fts", lambda: a.search_fts(spec)),
            ("graph", lambda: [r[0] for r in a._run("?[cid] := *mention:by_eid{eid:$eid,cid}, cid==$id", {"eid": chunk.entity_ids[0], "id": chunk.id})["rows"]]),
            ("vector", lambda: a.search_vector(spec)),
            ("decomposed", lambda: a.query(spec).ids),
            ("native", lambda: a.native_hybrid(spec)),
        ]:
            out = observe(probe)
            end = time.perf_counter()
            emit(phase="freshness", write=i, signal=name, write_ms=(ack-start)*1000,
                 ack_to_probe_ms=(end-ack)*1000, end_to_end_ms=(end-start)*1000,
                 hit=chunk.id in out["ids"], **out)
    # Exact rows and both indexes must exclude a failed transaction's unique row.
    try:
        a._run("{ ?[cid,text,emb] <- [['rollback019','rollbacktoken019',$v]] :put chunk {cid=>text,emb} } { ?[x] := *missing019[x] }", {"v": wl.inc_chunk_emb[0].tolist()})
    except Exception as exc:
        error = str(exc)
    else:
        raise AssertionError("rollback fixture unexpectedly committed")
    exact = a._run("?[cid] := *chunk{cid}, cid=='rollback019'")["rows"]
    fts = a._run("?[cid] := ~chunk:fts{cid | query:'rollbacktoken019',k:10}")["rows"]
    vector = a.search_vector(replace(_specs(wl)[0], vector=wl.inc_chunk_emb[0]))
    emit(phase="rollback", exact=exact, fts=fts, vector_leak="rollback019" in vector, expected_error=error)
    assert not exact and not fts and "rollback019" not in vector


def eligible_fts_truth(bm, terms, ids, eligible, k):
    # Match engine defaults, then break equal-score ties by primary key, not
    # numeric generation position (c100 sorts before c20).
    scores = {}
    for term in dict.fromkeys(terms):
        for index, tf in bm.postings.get(term, []):
            cid = ids[index]
            if cid not in eligible:
                continue
            score = bm.idf[term] * tf * 2.2 / (tf + 1.2 * (.25 + .75 * bm.doc_len[index] / bm.avgdl))
            scores[cid] = scores.get(cid, 0.) + score
    return sorted(scores, key=lambda cid: (-scores[cid], cid))[:k]


def filtered(a, wl, specs, emit, nqueries):
    ids = [c.id for c in wl.chunks]
    idset = set(ids)
    bm = _BM25Index([c.text for c in wl.chunks])
    mat = wl.chunk_emb.astype(np.float64)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    for qi, spec in enumerate(specs[:nqueries]):
        q = spec.vector.astype(np.float64); q /= np.linalg.norm(q)
        order = np.argsort(-(mat @ q), kind="stable")
        rng = np.random.default_rng(wl.meta.seed + qi)
        random_order = rng.permutation(len(ids))
        for assignment in ("random", "adversarial"):
            for fraction in (1.0, .1, .01, .001, 0.0, "fewer"):
                n = 3 if fraction == "fewer" else int(len(ids) * fraction)
                selected = (random_order[:n] if assignment == "random" else order[len(ids)-n:]) if n else []
                eligible = {ids[int(i)] for i in selected}
                truth = [ids[int(i)] for i in order if ids[int(i)] in eligible][:spec.k]
                ft = eligible_fts_truth(bm, spec.text.split(), ids, eligible, spec.k)
                common = dict(phase="E2", query=qi, assignment=assignment, selectivity=fraction,
                              eligible_count=n, eligible_sha256=digest(sorted(eligible)))
                # Store the actual fixture keys, so adversarial eligibility can be replayed.
                emit(**common, strategy="fixture", eligible=sorted(eligible), truth=truth, fts_truth=ft)
                params = {"qv": spec.vector.tolist(), "eligible": sorted(eligible), "eligible_rows": [[cid] for cid in sorted(eligible)], "k": spec.k}
                scripts = {
                    "post_filter": "?[cid,d] := ~chunk:vec{cid | query:vec($qv), k:100, ef:100, bind_distance:d}, is_in(cid,$eligible) :order d,cid :limit $k",
                    "exact": "eligible[cid] <- $eligible_rows\n?[cid,d] := eligible[cid], *chunk{cid,emb}, d=cos_dist(emb,vec($qv)) :order d,cid :limit $k",
                }
                for ef in (100, 200, 400, 800, 1600):
                    scripts[f"filter_ef{ef}"] = f"?[cid,d] := ~chunk:vec{{cid | query:vec($qv), k:$k, ef:{ef}, filter:is_in(cid,$eligible), bind_distance:d}} :order d,cid :limit $k"
                # Candidate-aware FTS keeps corpus-global BM25 while restricting before top-k.
                fp = dict(params, ftsk=max(1,len(eligible)), text=" OR ".join(dict.fromkeys(spec.text.split())))
                fs = "?[cid,s] := ~chunk:fts{cid | query:$text,k:$ftsk,candidates:$eligible,bind_score:s} :order -s,cid :limit $k"
                fp["eligible_rows"] = [[cid] for cid in sorted(eligible)]
                fresult = observe(lambda: [r[0] for r in a._db.run_script(fs, fp, True, timeout=5)["rows"]], ft, eligible)
                emit(**common, strategy="fts", **fresult)
                for strategy, script in scripts.items():
                    out = observe(lambda: [r[0] for r in a._db.run_script(script, params, True, timeout=5)["rows"]], truth, eligible)
                    fusion_truth = rrf_fuse([truth, ft], spec.k)
                    fusion = rrf_fuse([out["ids"], fresult["ids"]], spec.k)
                    emit(**common, strategy=strategy, **out, fused_ids=fusion, fused_truth=fusion_truth,
                         fused_recall=len(set(fusion)&set(fusion_truth))/len(fusion_truth) if fusion_truth else None)
                emit(**common, strategy="bounded_widening", outcome="unsupported",
                     reason="Python binding has per-call timeout but no governed snapshot handle or statement memory option; five independent calls cannot qualify stable-snapshot/64-MiB gate.")


def literal_cost(a, seed, emit, pairs):
    rng = np.random.default_rng(seed)
    for dim in (384, 1536):
        relation = f"e3_{dim}"
        vectors = rng.normal(size=(2000, dim)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        a._run(f":create {relation} {{id:Int => v:<F32;{dim}>}}")
        for offset in range(0, len(vectors), 100):
            a._run(f"?[id,v] <- $r :put {relation} {{id=>v}}", {"r": [[i,vectors[i].tolist()] for i in range(offset,offset+100)]})
        a._run(f"::hnsw create {relation}:vec {{dim:{dim},fields:[v],dtype:F32,distance:Cosine,m:16,ef_construction:64}}")
        emit(phase="E3_fixture", dim=dim, rows=len(vectors), vectors_sha256=hashlib.sha256(vectors.tobytes()).hexdigest())
        for shape in (1, 3):
            for changing in (False, True):
                for query_kind in ("minimal", "retrieval"):
                    for i in range(pairs + 10):
                        value = vectors[i % len(vectors) if changing else 0].tolist()
                        outputs, records = {}, []
                        for mode in (("literal", "parameter") if i % 2 == 0 else ("parameter", "literal")):
                            serialization = construction = engine = 0.0
                            rows = None
                            for _ in range(shape):
                                start = time.perf_counter()
                                encoded = json.dumps(value, separators=(",", ":")) if mode == "literal" else None
                                serialization += (time.perf_counter()-start)*1000
                                start = time.perf_counter()
                                params = {"v":value} if mode == "parameter" else {}
                                expr = "vec($v)" if mode == "parameter" else f"vec({encoded})"
                                script = (f"?[n] := n=length({expr})" if query_kind == "minimal" else
                                          f"?[id,d] := ~{relation}:vec{{id | query:{expr},k:10,ef:100,bind_distance:d}} :order d,id")
                                construction += (time.perf_counter()-start)*1000
                                start = time.perf_counter()
                                rows = a._db.run_script(script, params, True)["rows"]
                                engine += (time.perf_counter()-start)*1000
                            outputs[mode] = rows
                            records.append(dict(phase="E3", dim=dim, shape=shape, changing=changing,
                                query_kind=query_kind, pair=i-10, mode=mode, serialization_ms=serialization,
                                construction_ms=construction, engine_ms=engine,
                                total_ms=serialization+construction+engine, result_sha256=digest(rows)))
                        assert outputs["literal"] == outputs["parameter"]
                        if i >= 10:
                            for record in records: emit(**record)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["sqlite","rocksdb"], required=True)
    p.add_argument("--build", choices=["baseline","candidate"], required=True)
    p.add_argument("--engine-sha", required=True)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--scale", choices=["tiny","small"], default="small")
    p.add_argument("--phases", default="E1,E2,E3,E2_1536")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--database", type=Path, required=True)
    args=p.parse_args()
    phases=set(args.phases.split(","))
    if not phases <= {"E1","E2","E3","E2_1536"}:raise ValueError("unknown phase")
    if args.output.exists() or args.database.exists(): raise ValueError("fresh output and database paths required")
    args.output.parent.mkdir(parents=True,exist_ok=True); args.database.mkdir(parents=True)
    fh=args.output.open("x")
    def emit(**record):
        fh.write(json.dumps(record,separators=(",",":"))+"\n"); fh.flush()
    import mnestic
    extension = next(Path(mnestic.__file__).parent.glob("*.so"))
    emit(phase="manifest", args={k:str(v) for k,v in vars(args).items()},
         harness_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
         harness_status=subprocess.check_output(["git","status","--porcelain"],text=True),
         artifact_sha256=hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
         extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
         package=importlib.metadata.version("mnestic"), python=sys.version, platform=platform.platform(),
         dependency_freeze=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True),
         power=subprocess.check_output(["pmset","-g"],text=True),
         cpu=subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"],text=True).strip(),
         ram=subprocess.check_output(["sysctl","-n","hw.memsize"],text=True).strip(),
         concurrency=1, phases=args.phases, warmup=100, e2_queries=20, e3_pairs=100,
         limits={"e1_call_timeout":"not set", "e2_sweep_call_seconds":5,
                 "widening_attempts":5,"widening_total_ms":250,"estimated_statement_mib":64,
                 "widening_enforcement":"unsupported in this binding", "process_cap":"external 7200-second supervisor"},
         phase_instrumentation="unavailable; literal delta is not parse time", hnsw_rng="not seedable")
    cfg=yaml.safe_load(Path("configs/scales.yaml").read_text())[args.scale]
    start=time.perf_counter(); wl=generate(args.scale,seed=args.seed,**cfg)
    specs=_specs(wl)
    emit(phase="dataset", meta=asdict(wl.meta), chunk_hash=hashlib.sha256(wl.chunk_emb.tobytes()).hexdigest(),
         query_hash=hashlib.sha256(wl.query_emb.tobytes()).hexdigest(),
         rows_hash=digest([asdict(c) for c in wl.chunks]), edges_hash=digest([asdict(e) for e in wl.edges]),
         query_text_hash=digest([asdict(q) for q in wl.queries]), generate_seconds=time.perf_counter()-start)
    os.environ["MNESTIC_BACKEND"]=args.backend
    a=MnesticAdapter(); a.setup(wl.meta,args.database)
    try:
        print(args.build,args.backend,args.seed,"ingest",flush=True)
        ing=a.ingest(wl); build=a.build_indexes()
        emit(phase="build", ingest=asdict(ing), build=asdict(build))
        if "E1" in phases:
            print("E1 oracle",flush=True); truth=compute_ground_truth(wl)
            emit(phase="oracle", sha256=digest(truth))
            for name, fn in [("decomposed",lambda s:a.query(s).ids),("native",a.native_hybrid)]:
                print("E1",name,flush=True)
                for s in specs[:100]: fn(s)
                for i,s in enumerate(specs):
                    emit(phase="E1",path=name,query=i,**observe(lambda:fn(s),truth[wl.queries[i].id]))
        if "E2" in phases:
            print("E2",flush=True); filtered(a,wl,specs,emit,20 if args.scale=="small" else 2)
        if "E1" in phases:
            print("freshness",flush=True); freshness(a,wl,emit)
        if "E3" in phases:
            print("E3",flush=True); literal_cost(a,args.seed,emit,100 if args.scale=="small" else 2)
        if "E2_1536" in phases:
            print("E2 dimension 1536 sensitivity",flush=True)
            a.teardown()
            cfg1536=dict(cfg,dim=1536)
            sensitivity=generate(args.scale+"-1536",seed=args.seed,**cfg1536)
            emit(phase="dataset1536",meta=asdict(sensitivity.meta),
                 chunk_hash=hashlib.sha256(sensitivity.chunk_emb.tobytes()).hexdigest(),
                 query_hash=hashlib.sha256(sensitivity.query_emb.tobytes()).hexdigest(),
                 rows_hash=digest([asdict(c) for c in sensitivity.chunks]))
            directory=args.database/'dim1536';directory.mkdir()
            a=MnesticAdapter();a.setup(sensitivity.meta,directory)
            ing=a.ingest(sensitivity);build=a.build_indexes()
            emit(phase="build1536",ingest=asdict(ing),build=asdict(build))
            def emit1536(**record):
                record['phase']='E2_1536'
                emit(**record)
            filtered(a,sensitivity,_specs(sensitivity),emit1536,20 if args.scale=="small" else 2)
        emit(phase="complete", peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             disk_bytes=dir_size_bytes(args.database), elapsed_seconds=time.perf_counter()-start)
    finally:
        a.teardown(); fh.close()


if __name__=="__main__": main()
