"""Labelled retrieval quality on a real corpus (BEIR), through the same adapters.

The hybrid-recall workload's text is a synthetic vocabulary: recall@k there measures an
engine's *fidelity* to an exact oracle, not whether it retrieves relevant documents.
This module measures relevance against human judgments:

  dataset   BEIR SciFact (5,183 abstracts, 300 test queries with qrels); pinned by
            SHA-256. Other BEIR sets with the same layout work via --dataset.
  vectors   fastembed `BAAI/bge-small-en-v1.5` (384-d), computed once and cached.
  engines   each adapter ingests the corpus as chunks (no graph), builds its vector and
            full-text indexes, then answers every test query with its vector leg, its
            BM25 leg, the canonical app-side RRF of the two, and its native fusion if any.
  metrics   nDCG@10 and Recall@100 per leg against the qrels, plus the same metrics
            for exact references (NumPy exact cosine; the harness's exact BM25) so
            an engine's approximation loss is separated from the method's ceiling.

Analyzer note: engines are configured as in the main suite (mnestic: Simple tokenizer
+ Lowercase; Relate: lowercase alphanumeric terms, not configurable). Query text is
reduced to lowercase alphanumeric terms joined with OR for every engine.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .adapters import load_adapter
from .adapters.base import QuerySpec
from .fusion import rrf_fuse
from .workload.groundtruth import _BM25Index
from .workload.schema import Chunk, Workload, WorkloadMeta

DATASETS = {
    "scifact": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "sha256": "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
    },
}
MODEL = "BAAI/bge-small-en-v1.5"
K = 10
POOL = 100
RRF_K = 60.0
_OPERATORS = {"and", "or", "not", "near"}


def _terms(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in dict.fromkeys(toks) if t not in _OPERATORS]


def prepare(name: str, root: Path) -> Path:
    """Download (checksum-verified), unpack and embed a BEIR dataset; idempotent."""
    spec = DATASETS[name]
    d = root / f"beir-{name}"
    d.mkdir(parents=True, exist_ok=True)
    z = d / f"{name}.zip"
    if not z.exists():
        print(f"downloading {spec['url']}", flush=True)
        urllib.request.urlretrieve(spec["url"], z)  # noqa: S310 (pinned https URL)
    digest = hashlib.sha256(z.read_bytes()).hexdigest()
    if digest != spec["sha256"]:
        raise RuntimeError(f"{z} sha256 {digest} != pinned {spec['sha256']}")
    if not (d / name / "corpus.jsonl").exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(d)
    if not (d / "corpus_emb.npy").exists():
        from fastembed import TextEmbedding  # noqa: PLC0415

        corpus, queries, _ = _load(d / name)
        model = TextEmbedding(MODEL)
        t0 = time.perf_counter()
        print(f"embedding {len(corpus):,} docs with {MODEL} ...", flush=True)
        # Small batches: fastembed's default (256 long abstracts) drives an 8 GB
        # machine into swap; 16 runs at a steady ~7 docs/s on an M-series laptop.
        docs = np.array(list(model.embed([c["text"] for c in corpus], batch_size=16)),
                        dtype=np.float32)
        qs = np.array(list(model.query_embed([q["text"] for q in queries])), dtype=np.float32)
        np.save(d / "corpus_emb.npy", docs)
        np.save(d / "query_emb.npy", qs)
        (d / "embedding.json").write_text(json.dumps(
            {"model": MODEL, "dim": int(docs.shape[1]), "seconds": time.perf_counter() - t0}))
    return d


def _load(d: Path) -> tuple[list[dict], list[dict], dict[str, dict[str, int]]]:
    corpus = []
    with (d / "corpus.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            text = f"{r.get('title', '')}. {r.get('text', '')}".strip()
            corpus.append({"id": str(r["_id"]), "text": text})
    qrels: dict[str, dict[str, int]] = {}
    with (d / "qrels" / "test.tsv").open() as fh:
        next(fh)
        for line in fh:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    queries = []
    with (d / "queries.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if str(r["_id"]) in qrels:
                queries.append({"id": str(r["_id"]), "text": r["text"]})
    return corpus, queries, qrels


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int = K) -> float:
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at(ranked: list[str], rel: dict[str, int], n: int = POOL) -> float:
    return len(set(ranked[:n]) & set(rel)) / len(rel) if rel else 0.0


def _score(runs: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict:
    nd = [ndcg_at_k(r, qrels[q]) for q, r in runs.items()]
    rc = [recall_at(r, qrels[q]) for q, r in runs.items()]
    return {"ndcg@10": float(np.mean(nd)), "recall@100": float(np.mean(rc))}


def _references(corpus, queries, qrels, demb, qemb) -> dict:
    ids = [c["id"] for c in corpus]
    m = demb / (np.linalg.norm(demb, axis=1, keepdims=True) + 1e-12)
    vec_runs, bm_runs, fused = {}, {}, {}
    bm = _BM25Index([" ".join(_terms(c["text"])) for c in corpus])
    for i, q in enumerate(queries):
        qv = qemb[i] / (np.linalg.norm(qemb[i]) + 1e-12)
        order = np.argsort(-(m @ qv), kind="stable")[:POOL]
        vec_runs[q["id"]] = [ids[j] for j in order]
        bm_runs[q["id"]] = [ids[j] for j in bm.top(_terms(q["text"]), POOL)]
        fused[q["id"]] = rrf_fuse([vec_runs[q["id"]], bm_runs[q["id"]]], k=POOL, rrf_k=RRF_K)
    return {"vector_exact": _score(vec_runs, qrels), "bm25_exact": _score(bm_runs, qrels),
            "rrf_exact": _score(fused, qrels)}


def run_engine(engine: str, corpus, queries, qrels, demb, qemb, workdir: Path) -> dict:
    import shutil  # noqa: PLC0415

    dim = int(demb.shape[1])
    meta = WorkloadMeta(name="beir", seed=0, dim=dim, k=K, pool_n=POOL, graph_hops=0,
                        rrf_k=RRF_K, clusters=0, n_entities=0, n_edges=0,
                        n_chunks=len(corpus), n_queries=len(queries))
    chunks = [Chunk(id=c["id"], text=c["text"], entity_ids=[], cluster=0, vec_idx=i)
              for i, c in enumerate(corpus)]
    wl = Workload(meta=meta, entities=[], edges=[], chunks=chunks, queries=[],
                  entity_emb=np.zeros((0, dim), np.float32), chunk_emb=demb,
                  query_emb=qemb)
    edir = workdir / engine
    if edir.exists():
        shutil.rmtree(edir)
    edir.mkdir(parents=True)
    adapter = load_adapter(engine)()
    out: dict = {"engine": engine, "ok": False}
    try:
        adapter.setup(meta, edir)
        out["version"] = adapter.version
        t0 = time.perf_counter()
        adapter.ingest(wl)
        adapter.build_indexes()
        out["load_seconds"] = time.perf_counter() - t0
        runs: dict[str, dict[str, list[str]]] = {"vector": {}, "bm25": {}, "rrf": {}}
        native: dict[str, list[str]] = {}
        lat: dict[str, list[float]] = {"vector": [], "bm25": []}
        for i, q in enumerate(queries):
            spec = QuerySpec(text=" ".join(_terms(q["text"])), vector=qemb[i], seed_entity="",
                             k=K, pool_n=POOL, hops=0, rrf_k=RRF_K)
            t0 = time.perf_counter()
            v = adapter.search_vector(spec)
            lat["vector"].append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            b = adapter.search_fts(spec) if spec.text else []
            lat["bm25"].append((time.perf_counter() - t0) * 1000)
            runs["vector"][q["id"]] = v
            runs["bm25"][q["id"]] = b
            runs["rrf"][q["id"]] = rrf_fuse([v, b], k=POOL, rrf_k=RRF_K)
            try:
                nh = adapter.native_hybrid(_native_spec(spec))
            except Exception as exc:  # noqa: BLE001
                out["native_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                nh = None
            if nh is not None:
                native[q["id"]] = nh
        out["legs"] = {leg: _score(r, qrels) for leg, r in runs.items()}
        if native and len(native) == len(queries):
            out["legs"]["native_fusion"] = _score(native, qrels)
        out["latency_p50_ms"] = {k: float(np.percentile(v, 50)) for k, v in lat.items()}
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        adapter.teardown()
    return out


def _native_spec(spec: QuerySpec) -> QuerySpec:
    # Native fusion returns its fused top-k only; ask for the full pool so Recall@100 is
    # comparable with the app-side RRF row. mnestic's native call has no graph here
    # (the workdir has no `link` relation), so it is called without graph legs.
    return QuerySpec(text=spec.text, vector=spec.vector, seed_entity=spec.seed_entity,
                     k=POOL, pool_n=spec.pool_n, hops=spec.hops, rrf_k=spec.rrf_k)


def run(dataset: str, engines: list[str], root: Path, workdir: Path) -> dict:
    d = prepare(dataset, root)
    corpus, queries, qrels = _load(d / dataset)
    demb, qemb_all = np.load(d / "corpus_emb.npy"), np.load(d / "query_emb.npy")
    qemb = qemb_all[: len(queries)]
    payload = {
        "dataset": dataset,
        "n_docs": len(corpus),
        "n_queries": len(queries),
        "embedding": json.loads((d / "embedding.json").read_text()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "references": _references(corpus, queries, qrels, demb, qemb),
        "results": [],
    }
    for e in engines:
        print(f"  [{e}] ...", flush=True)
        payload["results"].append(run_engine(e, corpus, queries, qrels, demb, qemb, workdir))
    return payload


def render_markdown(p: dict) -> str:
    lines = [
        f"# Labelled retrieval quality — BEIR {p['dataset']}",
        "",
        f"{p['n_docs']:,} documents, {p['n_queries']} test queries with human relevance "
        f"labels; embeddings {p['embedding']['model']} ({p['embedding']['dim']}-d). "
        f"Generated {p['generated_at']}.",
        "",
        "| system | vector nDCG@10 | BM25 nDCG@10 | RRF nDCG@10 | native fusion nDCG@10 "
        "| vector R@100 | BM25 R@100 | RRF R@100 | vector p50 | BM25 p50 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    ref = p["references"]
    lines.append(
        f"| exact reference (NumPy cosine; harness BM25) | {ref['vector_exact']['ndcg@10']:.3f} "
        f"| {ref['bm25_exact']['ndcg@10']:.3f} | {ref['rrf_exact']['ndcg@10']:.3f} | — "
        f"| {ref['vector_exact']['recall@100']:.3f} | {ref['bm25_exact']['recall@100']:.3f} "
        f"| {ref['rrf_exact']['recall@100']:.3f} | — | — |"
    )
    for r in p["results"]:
        if not r.get("ok"):
            lines.append(f"| {r['engine']} | FAILED: {r.get('error', '')[:150]} |" + " |" * 8)
            continue
        lg = r["legs"]
        nat = lg.get("native_fusion")
        nat_txt = f"{nat['ndcg@10']:.3f}" if nat else "—"
        lines.append(
            f"| {r['engine']} {r['version']} | {lg['vector']['ndcg@10']:.3f} "
            f"| {lg['bm25']['ndcg@10']:.3f} | {lg['rrf']['ndcg@10']:.3f} | {nat_txt} "
            f"| {lg['vector']['recall@100']:.3f} | {lg['bm25']['recall@100']:.3f} "
            f"| {lg['rrf']['recall@100']:.3f} | {r['latency_p50_ms']['vector']:.2f} ms "
            f"| {r['latency_p50_ms']['bm25']:.2f} ms |"
        )
    lines += ["", "Single-threaded, warm, after one load; latency includes the Python call.", ""]
    return "\n".join(lines)
