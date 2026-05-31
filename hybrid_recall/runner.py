"""Benchmark orchestration: run each engine through ingest -> build -> query and collect
quality + cost metrics against the shared exact ground truth."""

from __future__ import annotations

import json
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .adapters import load_adapter
from .adapters.base import Adapter, QuerySpec
from .metrics import PeakRSS, percentiles, recall_at_k, time_call
from .workload.schema import Workload

WARMUP = 50            # warm disk-resident caches (e.g. mnestic's HNSW) before timing
DEFAULT_CONCURRENCY = 8
SAMPLE = 300           # query sample for the concurrent-QPS and native-fusion phases


@dataclass
class EngineResult:
    engine: str
    version: str
    scale: str
    capabilities: dict
    ok: bool
    error: str = ""
    # quality
    recall_at_k: float = 0.0
    recall_k: int = 0
    # ingest / build
    ingest_rows: int = 0
    ingest_seconds: float = 0.0
    ingest_rows_per_sec: float = 0.0
    build_seconds: float = 0.0
    disk_mb: float = 0.0
    build_peak_rss_mb: float = 0.0
    # query
    latency_ms: dict = field(default_factory=dict)
    mean_latency_ms: float = 0.0
    qps_serial: float = 0.0
    qps_concurrent: float = 0.0
    concurrency: int = 0
    query_peak_rss_mb: float = 0.0
    avg_round_trips: float = 0.0
    n_queries: int = 0
    # native fusion highlight (vector+FTS single call), if the engine offers one
    native_fusion_latency_ms: dict = field(default_factory=dict)


def _specs(wl: Workload) -> list[QuerySpec]:
    m = wl.meta
    return [
        QuerySpec(
            text=q.text,
            vector=wl.query_emb[q.vec_idx],
            seed_entity=q.seed_entity,
            k=m.k,
            pool_n=m.pool_n,
            hops=m.graph_hops,
            rrf_k=m.rrf_k,
        )
        for q in wl.queries
    ]


def run_engine(
    name: str,
    wl: Workload,
    ground_truth: dict[str, list[str]],
    workdir: Path,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> EngineResult:
    m = wl.meta
    cls = load_adapter(name)
    adapter: Adapter = cls()
    res = EngineResult(
        engine=name,
        version="",
        scale=m.name,
        capabilities=asdict(adapter.capabilities),
        ok=False,
        recall_k=m.k,
        n_queries=m.n_queries,
    )
    edir = workdir / name
    edir.mkdir(parents=True, exist_ok=True)
    try:
        adapter.setup(m, edir)
        res.version = adapter.version
        res.capabilities = asdict(adapter.capabilities)

        ing, _ = time_call(adapter.ingest, wl)
        res.ingest_rows = ing.rows
        res.ingest_seconds = ing.seconds
        res.ingest_rows_per_sec = ing.rows_per_sec

        with PeakRSS() as rss:
            build = adapter.build_indexes()
        res.build_seconds = build.seconds
        res.disk_mb = build.disk_bytes / (1024 * 1024)
        res.build_peak_rss_mb = rss.peak_mb

        specs = _specs(wl)
        qids = [q.id for q in wl.queries]

        # warmup (also surfaces errors before timing)
        for s in specs[:WARMUP]:
            adapter.query(s)

        # serial latency + recall
        latencies: list[float] = []
        recalls: list[float] = []
        round_trips: list[int] = []
        with PeakRSS() as rss:
            for qid, s in zip(qids, specs):
                out, dt = time_call(adapter.query, s)
                latencies.append(dt * 1000.0)
                round_trips.append(out.round_trips)
                recalls.append(recall_at_k(out.ids, ground_truth[qid], m.k))
        res.query_peak_rss_mb = rss.peak_mb
        res.recall_at_k = statistics.mean(recalls) if recalls else 0.0
        res.latency_ms = percentiles(latencies)
        res.mean_latency_ms = statistics.mean(latencies) if latencies else 0.0
        res.avg_round_trips = statistics.mean(round_trips) if round_trips else 0.0
        res.qps_serial = (len(latencies) / (sum(latencies) / 1000.0)) if latencies else 0.0

        # concurrent throughput on a sample (only where the adapter declares thread-safety)
        if getattr(adapter, "thread_safe", False) and concurrency > 1:
            sample = specs[:SAMPLE]
            res.concurrency = concurrency
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                list(ex.map(adapter.query, sample))
            elapsed = time.perf_counter() - t0
            res.qps_concurrent = len(sample) / elapsed if elapsed > 0 else 0.0

        # native single-call vector+FTS fusion highlight (not part of recall scoring).
        # Probe support once, then time a single call per query over a sample.
        if adapter.native_hybrid(specs[0]) is not None:
            nat_latencies = []
            for s in specs[:SAMPLE]:
                _, dt = time_call(adapter.native_hybrid, s)
                nat_latencies.append(dt * 1000.0)
            res.native_fusion_latency_ms = percentiles(nat_latencies)

        res.ok = True
    except Exception as exc:  # noqa: BLE001  (record, don't crash the suite)
        res.error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            adapter.teardown()
        except Exception:  # noqa: BLE001
            pass
    return res


def run_suite(
    wl: Workload,
    ground_truth: dict[str, list[str]],
    engines: list[str],
    workdir: Path,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict:
    results = []
    for name in engines:
        print(f"  [{name}] running ...", flush=True)
        r = run_engine(name, wl, ground_truth, workdir, concurrency)
        status = "ok" if r.ok else f"FAILED ({r.error})"
        extra = f"recall@{r.recall_k}={r.recall_at_k:.3f} p50={r.latency_ms.get('p50', 0):.2f}ms" if r.ok else ""
        print(f"  [{name}] {status} {extra}", flush=True)
        results.append(asdict(r))
    return {
        "meta": asdict(wl.meta),
        "environment": _environment(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }


def write_results(payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = payload["meta"]["name"]
    stamp = payload["generated_at"].replace(":", "").replace("-", "")
    path = out_dir / f"{scale}-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    # also update a stable "latest-<scale>.json" pointer
    (out_dir / f"latest-{scale}.json").write_text(json.dumps(payload, indent=2))
    return path
