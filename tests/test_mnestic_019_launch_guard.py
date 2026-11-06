"""The operational guard must fail closed and stop its entire process group."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

import pytest


spec = importlib.util.spec_from_file_location(
    'guard', Path(__file__).parents[1] / 'scripts/mnestic_019_launch_guard.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


@pytest.fixture(autouse=True)
def restore_signal_handlers():
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    yield
    for s, handler in saved.items():
        signal.signal(s, handler)


def clean_sample():
    return dict(cpu_idle=95, free_disk=5 * g.GIB, ac_power=True,
                builds=[], busy_background=[], swap_in=0, swap_out=0)


@pytest.mark.parametrize('change', [
    {'cpu_idle': 70}, {'free_disk': g.GIB}, {'ac_power': False},
    {'builds': [{'name': 'cargo'}]}, {'busy_background': [{'name': 'backupd'}]},
    {'swap_in': 17 * 1024 ** 2}, {'swap_out': 16384},
])
def test_busy_host_cannot_pass_preflight(change):
    s = clean_sample()
    assert g.quiet(s)
    s.update(change)
    assert not g.quiet(s)


def test_mutated_input_prevents_launch(tmp_path):
    source = tmp_path / 'input.py'
    source.write_text('frozen')
    manifest = {'harness_files': {str(source): g.sha256(source)}}
    source.write_text('changed')
    with pytest.raises(ValueError, match='Frozen harness input changed'):
        g.verify_manifest(manifest)


def test_no_quiet_window_never_starts_supervisor(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'
    root = tmp_path / 'campaign'
    manifest.write_text(json.dumps({'command': ['unused', '--root', str(root)]}))
    monkeypatch.setattr(sys, 'argv', ['guard', '--manifest', str(manifest), '--wait-seconds', '0'])
    monkeypatch.setattr(g, 'verify_manifest', lambda _: None)
    monkeypatch.setattr(g, 'swap_counts', lambda: {'Swapins': 0, 'Swapouts': 0})
    monkeypatch.setattr(g.subprocess, 'Popen', lambda *a, **kw: pytest.fail('must not launch'))
    monkeypatch.setattr(g.subprocess, 'check_output', lambda *a, **kw: 'test-sha\n')
    assert g.main() == 2
    assert not root.exists()
    assert json.loads(Path(str(root) + '-guard-status.json').read_text())['timing_started'] is False


def test_stop_reaches_supervisor_child(tmp_path):
    child_pid = tmp_path / 'child.pid'
    program = """import subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
open(sys.argv[1], 'w').write(str(child.pid))
time.sleep(120)
"""
    p = subprocess.Popen([sys.executable, '-c', program, str(child_pid)], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not child_pid.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert child_pid.exists()
        child = int(child_pid.read_text())
        g.stop(p)
        assert p.poll() is not None
        deadline = time.monotonic() + 5
        while g.psutil.pid_exists(child) and time.monotonic() < deadline:
            if g.psutil.Process(child).status() == g.psutil.STATUS_ZOMBIE:
                break
            time.sleep(.01)
        assert not g.psutil.pid_exists(child) or g.psutil.Process(child).status() == g.psutil.STATUS_ZOMBIE
    finally:
        if p.poll() is None:
            os.killpg(p.pid, 9)
            p.wait()


def test_stop_reaches_child_after_leader_exits(tmp_path):
    ready = tmp_path / 'ready'
    child_pid = tmp_path / 'pid'
    child_code = "import signal,time,sys; signal.signal(signal.SIGTERM, signal.SIG_IGN); open(sys.argv[1],'w').write('ready'); time.sleep(120)"
    parent_code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); open(sys.argv[3],'w').write(str(p.pid))"
    p = subprocess.Popen([sys.executable, '-c', parent_code, child_code, str(ready), str(child_pid)], start_new_session=True)
    try:
        p.wait(timeout=5)
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        child = g.psutil.Process(int(child_pid.read_text()))
        g.stop(p)
        deadline = time.monotonic() + 5
        while child.is_running() and child.status() != g.psutil.STATUS_ZOMBIE and time.monotonic() < deadline:
            time.sleep(.01)
        assert not child.is_running() or child.status() == g.psutil.STATUS_ZOMBIE
    finally:
        g.stop(p)


def test_final_sample_interference_cannot_be_completed(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'
    root = tmp_path / 'campaign'
    manifest.write_text(json.dumps({'command': ['unused', '--root', str(root)], 'run_count': 0}))
    monkeypatch.setattr(sys, 'argv', ['guard', '--manifest', str(manifest)])
    monkeypatch.setattr(g, 'verify_manifest', lambda _: None)
    monkeypatch.setattr(g, 'swap_counts', lambda: {'Swapins': 0, 'Swapouts': 0})
    monkeypatch.setattr(g.subprocess, 'check_output', lambda *a, **kw: 'test-sha\n')
    class Process:
        pid = 123
        returncode = 0
        done = False
        def poll(self):
            return 0 if self.done else None
    process = Process()
    monkeypatch.setattr(g.subprocess, 'Popen', lambda *a, **kw: process)
    stopped = []
    monkeypatch.setattr(g, 'stop', stopped.append)
    count = 0
    def sample(previous):
        nonlocal count
        count += 1
        s = clean_sample()
        if count > 6:
            process.done = True
            s['ac_power'] = False
        return s, previous
    monkeypatch.setattr(g, 'sample', sample)
    assert g.main() == 3
    assert stopped == [process]
    assert json.loads(Path(str(root) + '-guard-status.json').read_text())['state'] == 'interrupted'


@pytest.mark.parametrize('signum', [signal.SIGTERM, signal.SIGHUP])
def test_guard_signal_stops_launched_group(tmp_path, signum):
    root = tmp_path / 'campaign'
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'command': [sys.executable, '-c', 'import time; time.sleep(120)', '--root', str(root)]}))
    launcher = """import importlib.util, sys, time
spec=importlib.util.spec_from_file_location('guard',sys.argv.pop(1))
g=importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
g.verify_manifest=lambda _:None
g.swap_counts=lambda:{'Swapins':0,'Swapouts':0}
def sample(previous):
 time.sleep(.05)
 return dict(cpu_idle=99,free_disk=5*g.GIB,ac_power=True,builds=[],busy_background=[],swap_in=0,swap_out=0),previous
g.sample=sample
g.main()
"""
    p = subprocess.Popen([sys.executable, '-c', launcher, g.__file__, '--manifest', str(manifest)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    supervisor = None
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status_file = Path(str(root) + '-guard-status.json')
            try:
                status = json.loads(status_file.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                status = {}
            if status.get('state') == 'running':
                supervisor = status['supervisor_pid']
                break
            time.sleep(.02)
        assert supervisor is not None
        p.send_signal(signum)
        p.wait(timeout=5)
        assert not g.psutil.pid_exists(supervisor)
        assert json.loads(status_file.read_text())['timing_valid'] is False
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
        if supervisor and g.psutil.pid_exists(supervisor):
            os.killpg(supervisor, signal.SIGKILL)
