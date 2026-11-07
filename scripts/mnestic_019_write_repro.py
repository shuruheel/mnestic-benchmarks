"""Full-small-workload write diagnostic; retains the database on failure.

This skips retrieval timing and cannot replace the paired E1–E3 campaign.
Run with the exact candidate's Python environment, a fresh root and a time cap.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml
from mnestic_019_campaign import write_atomic
from hybrid_recall.adapters.mnestic_adapter import MnesticAdapter
from hybrid_recall.workload.generate import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--backend', choices=['sqlite', 'rocksdb'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    database = args.root / 'db'
    database.mkdir()
    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / 'configs/scales.yaml').read_text())['small']
    workload = generate('small', seed=args.seed, **cfg)
    assert len(workload.chunks) == 40000 and len(workload.inc_chunks) == 200
    os.environ['MNESTIC_BACKEND'] = args.backend
    import mnestic
    extension = next(Path(mnestic.__file__).parent.glob('*.so'))
    result = dict(seed=args.seed, backend=args.backend, complete=False,
                  package=importlib.metadata.version('mnestic'), python=sys.version,
                  extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
                  harness_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                  chunk_hash=hashlib.sha256(workload.chunk_emb.tobytes()).hexdigest(), records=[])
    output = args.root / 'outcome.json'
    def save():
        output.write_text(json.dumps(result, indent=2) + '\n')
    save()
    adapter = MnesticAdapter()
    start = time.monotonic()
    adapter.setup(workload.meta, database)
    try:
        print('ingest', flush=True)
        result['ingest'] = asdict(adapter.ingest(workload))
        print('build', flush=True)
        result['build'] = asdict(adapter.build_indexes())
        print('writes', flush=True)
        for i, chunk in enumerate(workload.inc_chunks):
            try:
                write_atomic(adapter, chunk, workload.inc_chunk_emb[i], chunk.text + ' freshprobe' + str(i))
            except Exception as error:
                result['records'].append(dict(write=i, error=str(error)))
                raise
            else:
                result['records'].append(dict(write=i, error=None))
        result['complete'] = True
    finally:
        result['elapsed_seconds'] = time.monotonic() - start
        save()
        adapter.teardown()
        print('retained database', database, flush=True)


if __name__ == '__main__':
    main()
