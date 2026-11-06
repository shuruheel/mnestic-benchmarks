"""Wait for a quiet host, then supervise the frozen campaign and retain interference.

Operational gate only: no workload reduction, retries, or successful resumption
of an interrupted run. Run from the benchmark repository with its frozen manifest.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid_recall.host import power_status, require_disk_storage, supported_system, swap_counts

BUILD_NAMES = {'cargo', 'rustc', 'clang', 'clang++', 'gcc', 'g++', 'cc1', 'cc1plus', 'ld', 'ld.lld', 'ld64.lld', 'rust-lld'}
BACKGROUND_NAMES = {
    'fileproviderd', 'ANECompilerService', 'backupd', 'mds', 'mds_stores',
    'mdworker', 'mdworker_shared', 'corespotlightd', 'spotlightknowledged',
    'spotlightknowledged.updater', 'mediaanalysisd',
}
GIB = 1024 ** 3


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(manifest):
    if manifest.get('qualification_ready') is not True:
        raise ValueError('Candidate qualification is not cleared for timing')
    for path, expected in manifest['harness_files'].items():
        if sha256(path) != expected:
            raise ValueError(f'Frozen harness input changed: {path}')
    if sha256('../mnestic/docs/plans/0.19-measurement-protocol.md') != manifest['engine_protocol_sha256']:
        raise ValueError('Frozen engine protocol changed')
    for build in ('baseline', 'candidate'):
        item = manifest[build]
        if sha256(item['wheel']) != item['sha256']:
            raise ValueError(f'{build} wheel changed')
        if sha256(item['installed']['extension']) != item['installed']['extension_sha256']:
            raise ValueError(f'{build} installed extension changed')
        python = manifest['command'][manifest['command'].index(f'--{build}-python') + 1]
        freeze = subprocess.check_output([python, '-m', 'pip', 'freeze'], text=True)
        if freeze != item['installed']['freeze']:
            raise ValueError(f'{build} dependencies changed')
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('Benchmark harness must be committed and clean')


def sample(previous, storage_path='/', campaign_pgid=None):
    system = supported_system()
    times = psutil.cpu_times_percent(interval=10)
    now = swap_counts()
    background, builds = [], []
    for process in psutil.process_iter(['pid', 'name']):
        try:
            name = process.info['name']
            cpu = process.cpu_percent()
            if name in BUILD_NAMES:
                builds.append({'pid': process.pid, 'name': name})
            own_process = process.pid == os.getpid() or (
                campaign_pgid is not None and os.getpgid(process.pid) == campaign_pgid)
            watched = name in BACKGROUND_NAMES or (system == 'Linux' and not own_process)
            if watched and cpu >= 25:
                background.append({'pid': process.pid, 'name': name, 'cpu': cpu})
        except (ProcessLookupError, psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    power = power_status()
    return {
        'epoch': time.time(), 'cpu_idle': times.idle,
        'cpu_steal': getattr(times, 'steal', 0),
        'available_memory': psutil.virtual_memory().available,
        'free_disk': psutil.disk_usage(storage_path).free,
        'swap_in': max(0, now['Swapins'] - previous['Swapins']),
        'swap_out': max(0, now['Swapouts'] - previous['Swapouts']),
        'ac_power': power['ac_power'], 'power_check': power['check'],
        'builds': builds, 'busy_background': background,
    }, now


def quiet(s):
    return (s['cpu_idle'] >= 85 and s['free_disk'] >= 3 * GIB and s['ac_power']
            and not s['builds'] and not s['busy_background']
            and s['swap_in'] <= 16 * 1024 ** 2 and s['swap_out'] == 0
            and s['cpu_steal'] <= 1 and s['available_memory'] >= 2 * GIB)


def resource_pressure(s):
    return (s['cpu_steal'] > 1 or s['available_memory'] < GIB
            or s['swap_in'] > 16 * 1024 ** 2 or s['swap_out'] > 0)


def live_group_members(pgid):
    """Include orphaned descendants, but not exited/zombie group members."""
    members = []
    for member in psutil.process_iter(['pid']):
        try:
            if os.getpgid(member.pid) == pgid and member.status() not in (
                    psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                members.append(member)
        except (ProcessLookupError, psutil.NoSuchProcess):
            pass
    return members


def signal_group(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        # A denied group signal is harmless only if no live member remains.
        # Otherwise use PID-reuse-aware Process objects to signal those members;
        # a real denial must propagate instead of claiming cleanup succeeded.
        for member in live_group_members(pgid):
            try:
                member.send_signal(sig)
            except psutil.NoSuchProcess:
                pass


def stop(process):
    if process.pid == os.getpgrp():
        raise RuntimeError('Refusing to stop the guard process group')
    if live_group_members(process.pid):
        signal_group(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while live_group_members(process.pid) and time.monotonic() < deadline:
        process.poll()  # Reap the leader without overlooking its descendants.
        time.sleep(.05)
    if live_group_members(process.pid):
        signal_group(process.pid, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while live_group_members(process.pid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(.05)
    remaining = live_group_members(process.pid)
    if remaining:
        raise RuntimeError(f'Cleanup left live processes: {[p.pid for p in remaining]}')
    process.wait(timeout=1)


def termination_requested(signum, _frame):
    # Raising lets the protected subprocess lifetime clean up the whole group.
    raise SystemExit(128 + signum)


def main():
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, termination_requested)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--wait-seconds', type=int, default=7200)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    command = manifest['command']
    root = Path(command[command.index('--root') + 1])
    if root.exists():
        raise ValueError('Fresh campaign root required')
    verify_manifest(manifest)
    storage = require_disk_storage(root)
    log_path = Path(str(root) + '-guard.jsonl')
    status_path = Path(str(root) + '-guard-status.json')
    process = None
    with log_path.open('x') as log:
        last_status = None
        def record(event):
            log.write(json.dumps(event) + '\n')
            log.flush()

        def status(state, **fields):
            nonlocal last_status
            result = dict(state=state, epoch=time.time(), guard_pid=os.getpid(), **fields)
            status_path.write_text(json.dumps(result, indent=2) + '\n')
            record(result)
            last_status = result

        record({'manifest': str(args.manifest), 'manifest_sha256': sha256(args.manifest),
                'harness_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'guard_sha256': sha256(__file__), 'wait_seconds': args.wait_seconds,
                'storage': storage})
        status('waiting_for_quiet')
        deadline = time.monotonic() + args.wait_seconds
        previous = swap_counts()
        # Prime per-process CPU counters before the first ten-second sample.
        for p in psutil.process_iter():
            try:
                p.cpu_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        quiet_samples = 0
        while time.monotonic() < deadline:
            s, previous = sample(previous, storage_path=storage['probe_path'])
            record(dict(phase='preflight', **s))
            quiet_samples = quiet_samples + 1 if quiet(s) else 0
            if quiet_samples >= 6:
                break
        else:
            status('quiet_window_unavailable', timing_started=False)
            return 2
        verify_manifest(manifest)
        with Path(str(root) + '-supervisor.log').open('x') as output:
            try:
                process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                status('running', supervisor_pid=process.pid, supervisor_started=True)
                busy_samples = 0
                while process.poll() is None:
                    s, previous = sample(previous, storage_path=storage['probe_path'],
                                         campaign_pgid=process.pid)
                    record(dict(phase='campaign', **s))
                    busy_samples = busy_samples + 1 if s['busy_background'] or resource_pressure(s) else 0
                    if s['builds'] or not s['ac_power'] or s['free_disk'] < GIB or busy_samples >= 3:
                        status('interrupted', timing_valid=False, reason=s,
                               note='Retain raw rows; do not accept this campaign or silently retry.')
                        return 3
                    if process.poll() is not None:
                        break
            except BaseException:
                status('guard_failed', timing_valid=False)
                raise
            finally:
                if process is not None:
                    try:
                        stop(process)
                        record({'phase': 'cleanup', 'epoch': time.time(), 'success': True})
                    except Exception as error:
                        status('cleanup_failed', timing_valid=False,
                               error=repr(error), previous_status=last_status)
                        raise
        results = [json.loads(p.read_text()) for p in root.glob('*-status.json')]
        complete = (process.returncode == 0 and len(results) == manifest['run_count']
                    and all(r['exit_code'] == 0 and r['timing_valid'] and not r['timed_out'] for r in results))
        status('completed' if complete else 'incomplete', supervisor_exit=process.returncode,
               run_count=len(results), accepted=False,
               note='Raw results still require protocol analysis and independent review.')
        return 0 if complete else 4


if __name__ == '__main__':
    raise SystemExit(main())
