"""`hybrid-recall` command-line entrypoint: gen | run | report."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from .adapters import EMBEDDED
from .report import load_results, render_markdown, render_plots
from .runner import run_suite, write_results
from .workload import generate as gen_mod
from .workload.groundtruth import compute_ground_truth
from .workload.schema import Workload

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "scales.yaml"
WORKLOADS = ROOT / "workloads"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"


def _load_scales() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def _workload_dir(scale: str) -> Path:
    return WORKLOADS / scale


def cmd_gen(args: argparse.Namespace) -> int:
    scales = _load_scales()
    if args.scale not in scales:
        print(f"unknown scale '{args.scale}'. choices: {sorted(scales)}", file=sys.stderr)
        return 2
    params = dict(scales[args.scale])
    wdir = _workload_dir(args.scale)
    print(f"generating workload '{args.scale}' -> {wdir}", flush=True)
    t0 = time.perf_counter()
    wl = gen_mod.generate(
        args.scale, seed=args.seed, real_embeddings=args.real_embeddings, **params
    )
    print(
        f"  generated in {time.perf_counter()-t0:.1f}s: "
        f"{len(wl.chunks):,} chunks, {len(wl.entities):,} entities, "
        f"{len(wl.edges):,} edges, {len(wl.queries):,} queries",
        flush=True,
    )
    wl.save(wdir)
    print("  computing exact ground truth ...", flush=True)
    t0 = time.perf_counter()
    gt = compute_ground_truth(wl)
    (wdir / "ground_truth.json").write_text(json.dumps(gt))
    print(f"  ground truth in {time.perf_counter()-t0:.1f}s -> {wdir/'ground_truth.json'}", flush=True)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    wdir = _workload_dir(args.scale)
    if not (wdir / "meta.json").exists():
        print(f"no workload at {wdir}. run `hybrid-recall gen --scale {args.scale}` first.", file=sys.stderr)
        return 2
    wl = Workload.load(wdir)
    gt = json.loads((wdir / "ground_truth.json").read_text())
    engines = args.engines.split(",") if args.engines else list(EMBEDDED)
    workdir = Path(args.workdir) if args.workdir else (ROOT / ".bench_work" / args.scale)
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"running engines {engines} on scale '{args.scale}'", flush=True)
    payload = run_suite(
        wl, gt, engines, workdir, concurrency=args.concurrency, max_queries=args.max_queries or None
    )
    path = write_results(payload, RESULTS)
    print(f"wrote {path}", flush=True)
    if args.report:
        _do_report(path, plots=args.plots)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    if args.results:
        path = Path(args.results)
    else:
        # pick the most recent latest-*.json, else newest json in results/
        latest = sorted(RESULTS.glob("latest-*.json"))
        if not latest:
            latest = sorted(RESULTS.glob("*.json"))
        if not latest:
            print("no results found. run `hybrid-recall run ...` first.", file=sys.stderr)
            return 2
        path = max(latest, key=lambda p: p.stat().st_mtime)
    _do_report(path, plots=args.plots)
    return 0


def _do_report(path: Path, plots: bool) -> None:
    payload = load_results(path)
    DOCS.mkdir(parents=True, exist_ok=True)
    md = render_markdown(payload)
    out = DOCS / "RESULTS.md"
    if plots:
        imgs = render_plots(payload, DOCS / "assets")
        if imgs:
            md += "\n## Plots\n\n" + "\n".join(f"![{p.stem}](assets/{p.name})" for p in imgs)
    out.write_text(md)
    print(f"wrote {out} (from {path.name})", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hybrid-recall", description="Hybrid-recall benchmark suite")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate a workload + exact ground truth")
    g.add_argument("--scale", default="small")
    g.add_argument("--seed", type=int, default=1234)
    g.add_argument("--real-embeddings", action="store_true", help="use sentence-transformers")
    g.set_defaults(func=cmd_gen)

    r = sub.add_parser("run", help="run engines against a generated workload")
    r.add_argument("--scale", default="small")
    r.add_argument("--engines", default="", help="comma list; default = embedded engines")
    r.add_argument("--workdir", default="", help="engine working dir (default .bench_work/<scale>)")
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="cap the measured query set (0 = full bank); percentiles/recall stay solid at ~400",
    )
    r.add_argument("--report", action="store_true", help="render RESULTS.md after running")
    r.add_argument("--plots", action="store_true", help="also render plots (needs [plots])")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="render docs/RESULTS.md from a results JSON")
    rp.add_argument("--results", default="", help="path to a results JSON (default: latest)")
    rp.add_argument("--plots", action="store_true")
    rp.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
