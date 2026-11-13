"""Lifecycle axis: what happens to a memory store *after* the warm-read benchmark.

The recall/latency suite measures a freshly built store answering warm reads. A
long-lived agent memory also has to survive churn, restarts and crashes, and its
memory footprint decides which machines it fits on. This axis measures those:

  load     ingest + build, then vector recall@k vs the exact oracle and latency on a
           query sample (the pre-churn baseline)
  churn    re-embed a fraction of memories into a different neighbourhood and delete
           another fraction, one write call each (timed); then recall@k vs the exact
           oracle *recomputed over the post-churn corpus*, plus the stale-hit rate
           (queries whose top-k still returns a deleted memory)
  reopen   a NEW process opens the closed store: open time, first-query latency,
           steady latency, recall vs the post-churn oracle (must not regress), RSS
  crash    a writer process acknowledges single-memory writes (each ack is appended
           to a file and fsync'd *after* the engine call returns), and is SIGKILLed at
           a random moment; a fresh process then reopens the store and checks that
           every acknowledged write is present. Repeated `trials` times.

Every phase runs in its own subprocess, so RSS is attributable to one engine (the
suite runner keeps every engine in one process) and reopen is a genuinely new process.

Scope, stated so results are not over-read:
  - "cold" reopen is process-cold, not OS-page-cache-cold (that needs `sudo purge`).
  - SIGKILL tests process-crash durability. It does not test power loss: data in the
    OS page cache survives SIGKILL. Durability modes are reported per engine.
  - RSS is the engine process's resident set (baseline = after the harness loads its
    inputs, before the engine opens). Allocator retention makes it an upper bound.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import resource
import signal
import statistics
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import psutil

from .adapters import load_adapter
from .adapters.base import QuerySpec
from .metrics import PeakRSS, percentiles, recall_at_k
from .workload.schema import Workload, WorkloadMeta

QUERIES = 200          # query sample per measurement
WARMUP = 20
UPDATE_FRAC = 0.10     # fraction of memories re-embedded during churn
DELETE_FRAC = 0.05     # fraction deleted during churn
TRUTH_FILE = "lifecycle_truth.json"


# --------------------------------------------------------------------------- helpers
def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _peak_rss_mb() -> float:
    # ru_maxrss: bytes on macOS, KiB on Linux
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def _normalize(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32, copy=True)
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
    return m


def exact_topk(ids: list[str], mat: np.ndarray, queries: np.ndarray, k: int) -> list[list[str]]:
    """Exact cosine top-k of each query over the rows of `mat` (NumPy oracle)."""
    m = _normalize(mat)
    q = _normalize(queries)
    sims = q @ m.T
    out = []
    for row in sims:
        top = np.argpartition(-row, k)[:k] if len(row) > k else np.arange(len(row))
        top = top[np.argsort(-row[top], kind="stable")]
        out.append([ids[i] for i in top])
    return out


def churn_plan(wl: Workload, seed: int, update_frac: float, delete_frac: float):
    """Deterministic churn: which memories are re-embedded (and to what) or deleted.
    A re-embedded memory takes the vector of a random memory from a different cluster
    plus small noise, so it moves to a new neighbourhood (the hard case for a graph
    index, which must unlink the old neighbours and link new ones)."""
    rng = np.random.default_rng(seed)
    n = len(wl.chunks)
    perm = rng.permutation(n)
    n_upd, n_del = int(n * update_frac), int(n * delete_frac)
    upd_idx, del_idx = perm[:n_upd], perm[n_upd : n_upd + n_del]
    dim = wl.chunk_emb.shape[1]
    updates = []
    for i in upd_idx:
        c = wl.chunks[i]
        for _ in range(8):
            j = int(rng.integers(n))
            if wl.chunks[j].cluster != c.cluster:
                break
        vec = wl.chunk_emb[wl.chunks[j].vec_idx] + rng.normal(0, 0.05, dim).astype(np.float32)
        updates.append((c.id, c.text, vec.astype(np.float32)))
    deletes = [wl.chunks[i].id for i in del_idx]
    return updates, deletes


def _specs(meta: WorkloadMeta, qemb: np.ndarray, n: int) -> list[QuerySpec]:
    return [
        QuerySpec(text="", vector=qemb[i], seed_entity="", k=meta.k, pool_n=meta.pool_n,
                  hops=meta.graph_hops, rrf_k=meta.rrf_k)
        for i in range(min(n, len(qemb)))
    ]


def _measure(adapter, specs, truth, k) -> dict:
    for s in specs[:WARMUP]:
        adapter.search_vector(s)
    lat, rec, results = [], [], []
    for s, gt in zip(specs, truth):
        t0 = time.perf_counter()
        out = adapter.search_vector(s)[:k]
        lat.append((time.perf_counter() - t0) * 1000.0)
        rec.append(recall_at_k(out, gt, k))
        results.append(out)
    return {"recall": statistics.mean(rec), "latency_ms": percentiles(lat), "_results": results}


def _load_meta(wdir: Path) -> tuple[WorkloadMeta, np.ndarray]:
    meta = WorkloadMeta(**json.loads((wdir / "meta.json").read_text()))
    return meta, np.load(wdir / "query_emb.npy")


# --------------------------------------------------------------------------- workers
def worker_load(engine: str, wdir: Path, edir: Path, seed: int) -> dict:
    wl = Workload.load(wdir)
    m = wl.meta
    specs = _specs(m, wl.query_emb, QUERIES)
    qvecs = np.stack([s.vector for s in specs])
    ids = [c.id for c in wl.chunks]
    truth_pre = exact_topk(ids, wl.chunk_emb[[c.vec_idx for c in wl.chunks]], qvecs, m.k)
    updates, deletes = churn_plan(wl, seed, UPDATE_FRAC, DELETE_FRAC)
    base_rss = _rss_mb()

    adapter = load_adapter(engine)()
    adapter.setup(m, edir)
    t0 = time.perf_counter()
    adapter.ingest(wl)
    ingest_s = time.perf_counter() - t0
    with PeakRSS() as build_rss:
        t0 = time.perf_counter()
        build = adapter.build_indexes()
        build_s = time.perf_counter() - t0
    out: dict = {
        "version": adapter.version,
        "n_chunks": len(ids),
        "baseline_rss_mb": base_rss,
        "ingest_seconds": ingest_s,
        "build_seconds": build_s,
        "disk_mb_after_build": build.disk_bytes / 2**20,
        "rss_after_build_mb": _rss_mb(),
        "build_peak_rss_mb": build_rss.peak_mb,
    }
    pre = _measure(adapter, specs, truth_pre, m.k)
    pre.pop("_results")
    out["pre_churn"] = pre

    # churn. A rejected write is counted by error code and the run continues; the
    # engine is expected to roll it back, so the oracle keeps the old state for it.
    upd_ms, del_ms = [], []
    rejected: dict[str, dict[str, int]] = {"update": {}, "delete": {}}
    applied_upd, applied_del = [], []

    def _attempt(kind: str, fn, *args) -> bool:
        t0 = time.perf_counter()
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001
            code = _error_code(exc)
            rejected[kind][code] = rejected[kind].get(code, 0) + 1
            return False
        (upd_ms if kind == "update" else del_ms).append((time.perf_counter() - t0) * 1000.0)
        return True

    for cid, text, vec in updates:
        if _attempt("update", adapter.update_memory, cid, text, vec):
            applied_upd.append((cid, vec))
    for cid in deletes:
        if _attempt("delete", adapter.delete_memory, cid):
            applied_del.append(cid)
    out["churn"] = {
        "updates": len(updates),
        "deletes": len(deletes),
        "updates_rejected": sum(rejected["update"].values()),
        "deletes_rejected": sum(rejected["delete"].values()),
        "rejections_by_code": rejected,
        "update_ms": percentiles(upd_ms),
        "delete_ms": percentiles(del_ms),
    }

    # exact oracle over the post-churn corpus (applied operations only)
    vecs = {c.id: wl.chunk_emb[c.vec_idx] for c in wl.chunks}
    for cid, vec in applied_upd:
        vecs[cid] = vec
    dead = set(applied_del)
    # consistency: a rejected operation must leave the old state readable
    for cid in [c for c, _t, _v in updates if c not in dict(applied_upd)][:50]:
        if not adapter.has_memory(cid):
            out.setdefault("rejected_update_lost_row", []).append(cid)
    live = [cid for cid in ids if cid not in dead]
    truth_post = exact_topk(live, np.stack([vecs[c] for c in live]), qvecs, m.k)
    post = _measure(adapter, specs, truth_post, m.k)
    results = post.pop("_results")
    post["stale_hit_queries"] = sum(1 for r in results if dead.intersection(r))
    post["stale_hit_rate"] = post["stale_hit_queries"] / len(results)
    out["post_churn"] = post
    out["rss_after_churn_mb"] = _rss_mb()

    t0 = time.perf_counter()
    adapter.teardown()
    out["close_seconds"] = time.perf_counter() - t0
    out["disk_mb_after_close"] = _dir_mb(edir)
    out["peak_rss_mb"] = _peak_rss_mb()
    (edir / TRUTH_FILE).write_text(json.dumps({"truth": truth_post, "deleted": applied_del[:50]}))
    return out


def _error_code(exc: Exception) -> str:
    """A stable label for a rejected write: mnestic raises a dict-shaped diagnostic with
    a `code`; Relate raises typed exceptions; anything else falls back to the type."""
    arg = exc.args[0] if exc.args else None
    if isinstance(arg, dict) and arg.get("code"):
        return str(arg["code"])
    return type(exc).__name__


def worker_reopen(engine: str, wdir: Path, edir: Path) -> dict:
    meta, qemb = _load_meta(wdir)
    saved = json.loads((edir / TRUTH_FILE).read_text())
    specs = _specs(meta, qemb, QUERIES)
    base_rss = _rss_mb()
    adapter = load_adapter(engine)()
    t0 = time.perf_counter()
    adapter.reopen(meta, edir)
    open_s = time.perf_counter() - t0
    rss_open = _rss_mb()
    t0 = time.perf_counter()
    adapter.search_vector(specs[0])
    first_ms = (time.perf_counter() - t0) * 1000.0
    rss_first = _rss_mb()
    meas = _measure(adapter, specs, saved["truth"], meta.k)
    meas.pop("_results")
    deleted_present = sum(1 for cid in saved["deleted"] if adapter.has_memory(cid))
    out = {
        "baseline_rss_mb": base_rss,
        "open_seconds": open_s,
        "rss_after_open_mb": rss_open,
        "first_query_ms": first_ms,
        "rss_after_first_query_mb": rss_first,
        "steady": meas,
        "rss_after_queries_mb": _rss_mb(),
        "deleted_resurrected": deleted_present,
    }
    adapter.teardown()
    out["peak_rss_mb"] = _peak_rss_mb()
    return out


def worker_write(engine: str, wdir: Path, edir: Path, prefix: str, acked: Path, limit: int) -> dict:
    meta, _ = _load_meta(wdir)
    adapter = load_adapter(engine)()
    adapter.reopen(meta, edir)
    rng = np.random.default_rng(zlib.crc32(prefix.encode()))
    fd = os.open(acked, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    print("READY", flush=True)
    t0 = time.perf_counter()
    n = 0
    for i in range(limit):
        cid = f"{prefix}{i:06d}"
        vec = rng.standard_normal(meta.dim).astype(np.float32)
        adapter.upsert_memory(cid, f"durability probe {cid}", vec, [])
        os.write(fd, (cid + "\n").encode())
        os.fsync(fd)
        n += 1
    os.close(fd)
    adapter.teardown()
    return {"written": n, "seconds": time.perf_counter() - t0}


def worker_verify(engine: str, wdir: Path, edir: Path, acked: Path) -> dict:
    meta, _ = _load_meta(wdir)
    ids = [line for line in acked.read_text().splitlines() if line] if acked.exists() else []
    adapter = load_adapter(engine)()
    t0 = time.perf_counter()
    try:
        adapter.reopen(meta, edir)
    except Exception as exc:  # noqa: BLE001
        return {"open_ok": False, "error": f"{type(exc).__name__}: {exc}", "acked": len(ids)}
    open_s = time.perf_counter() - t0
    lost = [cid for cid in ids if not adapter.has_memory(cid)]
    adapter.teardown()
    return {"open_ok": True, "open_seconds": open_s, "acked": len(ids), "lost": len(lost),
            "lost_sample": lost[:5]}


def _dir_mb(p: Path) -> float:
    from .adapters.base import dir_size_bytes  # noqa: PLC0415

    return dir_size_bytes(p) / 2**20


# --------------------------------------------------------------------------- parent
def _count_lines(p: Path) -> int:
    return sum(1 for line in p.read_text().splitlines() if line) if p.exists() else 0


def _spawn(args: list[str], out: Path, timeout: float | None = None) -> dict:
    cmd = [sys.executable, "-m", "hybrid_recall.lifecycle", "_worker", *args, "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not out.exists():
        tail = (proc.stderr or proc.stdout or "")[-800:]
        raise RuntimeError(f"worker {args[:2]} failed (rc={proc.returncode}): {tail}")
    return json.loads(out.read_text())


def run_lifecycle(engine: str, wdir: Path, workdir: Path, seed: int = 7, trials: int = 3,
                  write_limit: int = 20_000, kill_window: tuple[float, float] = (0.5, 2.0)) -> dict:
    import shutil  # noqa: PLC0415

    edir = workdir / engine
    if edir.exists():
        shutil.rmtree(edir)
    edir.mkdir(parents=True)
    common = ["--engine", engine, "--wdir", str(wdir), "--edir", str(edir)]
    res: dict = {"engine": engine, "ok": False}
    try:
        print(f"  [{engine}] load + churn ...", flush=True)
        res["load"] = _spawn(["--mode", "load", *common, "--seed", str(seed)], edir / "_load.json")
        print(f"  [{engine}] reopen ...", flush=True)
        res["reopen"] = _spawn(["--mode", "reopen", *common], edir / "_reopen.json")
        rnd = random.Random(seed)
        crash = []
        for t in range(trials):
            acked = edir / f"_acked_{t}.txt"
            cmd = [sys.executable, "-m", "hybrid_recall.lifecycle", "_worker", "--mode", "write",
                   *common, "--prefix", f"dur{t}_", "--acked", str(acked),
                   "--limit", str(write_limit), "--out", str(edir / f"_write_{t}.json")]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            line = proc.stdout.readline().strip()
            if line != "READY":
                proc.kill()
                raise RuntimeError(f"writer did not start: {line} {proc.stderr.read()[-500:]}")
            # The kill window starts at the first acknowledged write, so a slow first
            # write cannot turn a trial into a no-op; its latency is reported instead.
            t_ready = time.perf_counter()
            first_ack_s = None
            while proc.poll() is None and time.perf_counter() - t_ready < 120:
                if acked.exists() and acked.stat().st_size > 0:
                    first_ack_s = time.perf_counter() - t_ready
                    break
                time.sleep(0.005)
            if first_ack_s is None:
                proc.kill()
                proc.wait()
                raise RuntimeError(f"writer acknowledged nothing in 120 s: {proc.stderr.read()[-500:]}")
            n_at_first = _count_lines(acked)
            delay = rnd.uniform(*kill_window)
            time.sleep(delay)
            killed = proc.poll() is None
            if killed:
                proc.send_signal(signal.SIGKILL)
            proc.wait()
            print(f"  [{engine}] crash trial {t}: first ack {first_ack_s:.2f}s, "
                  f"killed {delay:.2f}s later", flush=True)
            v = _spawn(["--mode", "verify", *common, "--acked", str(acked)], edir / f"_verify_{t}.json")
            v.update(first_ack_s=first_ack_s, kill_after_first_ack_s=delay, killed_mid_write=killed)
            v["acked_writes_per_s"] = (v["acked"] - n_at_first) / delay if killed else None
            crash.append(v)
        res["crash"] = crash
        res["ok"] = True
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res


def render_markdown(payload: dict) -> str:
    meta = payload["meta"]
    lines = [
        f"# Lifecycle results — scale `{meta['name']}`",
        "",
        f"{meta['n_chunks']:,} memories, dim {meta['dim']}, k={meta['k']}; "
        f"churn = {UPDATE_FRAC:.0%} re-embedded + {DELETE_FRAC:.0%} deleted; "
        f"{QUERIES} queries per measurement; vector leg only, recall vs an exact NumPy oracle.",
        f"Generated {payload['generated_at']} on {payload['environment']['platform']}.",
        "",
        "| engine | recall pre | churn writes rejected | recall post-churn | stale hits "
        "| update p50 | delete p50 | reopen | first query | recall after reopen | RSS loaded "
        "| RSS reopened | first write after reopen / acked writes | crash: acked lost |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in payload["results"]:
        if not r.get("ok"):
            err = r.get("error", "").replace("|", "/").replace("\n", " ")[:160]
            lines.append(f"| {r['engine']} | FAILED: {err} |" + " |" * 12)
            continue
        ld, ro, cr = r["load"], r["reopen"], r["crash"]
        lost = sum(c.get("lost", 0) for c in cr if c.get("open_ok"))
        acked = sum(c.get("acked", 0) for c in cr)
        bad_open = sum(1 for c in cr if not c.get("open_ok"))
        crash_txt = f"{lost} of {acked}" + (f" ({bad_open} failed reopen)" if bad_open else "")
        rates = [c["acked_writes_per_s"] for c in cr if c.get("acked_writes_per_s")]
        firsts = [c["first_ack_s"] for c in cr if c.get("first_ack_s") is not None]
        write_txt = (f"{statistics.median(firsts):.2f} s / "
                     f"{statistics.median(rates):.0f} per s" if rates and firsts else "n/a")
        ch = ld["churn"]
        rej = (f"{ch['updates_rejected']}/{ch['updates']} upd, "
               f"{ch['deletes_rejected']}/{ch['deletes']} del")
        lines.append(
            f"| {r['engine']} {ld['version']} | {ld['pre_churn']['recall']:.3f} | {rej} "
            f"| {ld['post_churn']['recall']:.3f} | {ld['post_churn']['stale_hit_rate']:.1%} "
            f"| {ld['churn']['update_ms']['p50']:.2f} ms | {ld['churn']['delete_ms']['p50']:.2f} ms "
            f"| {ro['open_seconds'] * 1000:.0f} ms | {ro['first_query_ms']:.1f} ms "
            f"| {ro['steady']['recall']:.3f} "
            f"| {ld['rss_after_build_mb'] - ld['baseline_rss_mb']:.0f} MB "
            f"| {ro['rss_after_queries_mb'] - ro['baseline_rss_mb']:.0f} MB | {write_txt} "
            f"| {crash_txt} |"
        )
    lines += [
        "",
        "RSS columns are deltas over the worker's baseline before the engine opened "
        "(loaded = after ingest + build; reopened = after reopen + the query sample). "
        "Reopen is process-cold, OS-cache-warm. Crash = SIGKILL of a single-memory writer; "
        "process-crash durability only, not power loss.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hybrid_recall.lifecycle")
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("_worker")
    w.add_argument("--mode", required=True, choices=["load", "reopen", "write", "verify"])
    w.add_argument("--engine", required=True)
    w.add_argument("--wdir", required=True)
    w.add_argument("--edir", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--seed", type=int, default=7)
    w.add_argument("--prefix", default="dur_")
    w.add_argument("--acked", default="")
    w.add_argument("--limit", type=int, default=20_000)
    a = p.parse_args(argv)
    wdir, edir = Path(a.wdir), Path(a.edir)
    edir.mkdir(parents=True, exist_ok=True)
    if a.mode == "load":
        out = worker_load(a.engine, wdir, edir, a.seed)
    elif a.mode == "reopen":
        out = worker_reopen(a.engine, wdir, edir)
    elif a.mode == "write":
        out = worker_write(a.engine, wdir, edir, a.prefix, Path(a.acked), a.limit)
    else:
        out = worker_verify(a.engine, wdir, edir, Path(a.acked))
    Path(a.out).write_text(json.dumps(out, indent=2))
    return 0


def run_suite(engines: list[str], wdir: Path, workdir: Path, trials: int = 3) -> dict:
    meta = json.loads((wdir / "meta.json").read_text())
    results = [run_lifecycle(e, wdir, workdir, trials=trials) for e in engines]
    return {
        "meta": meta,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "backend_env": {"MNESTIC_BACKEND": os.environ.get("MNESTIC_BACKEND", "sqlite")}},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": {"queries": QUERIES, "update_frac": UPDATE_FRAC, "delete_frac": DELETE_FRAC,
                   "trials": trials},
        "results": results,
    }


if __name__ == "__main__":
    raise SystemExit(main())
