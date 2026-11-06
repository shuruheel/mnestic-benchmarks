"""Host evidence shared by the macOS and Linux release campaign runners."""
from pathlib import Path
import platform
import re
import resource
import subprocess

import psutil


def supported_system():
    system = platform.system()
    if system not in {'Darwin', 'Linux'}:
        raise RuntimeError(f'Unsupported measurement platform: {system}')
    return system


def power_status():
    if supported_system() == 'Darwin':
        raw = subprocess.check_output(['pmset', '-g', 'batt'], text=True)
        return {'check': 'pmset', 'ac_power': "drawing from 'AC Power'" in raw, 'raw': raw}
    battery = psutil.sensors_battery()
    if battery is None:
        return {'check': 'not_applicable_no_battery', 'ac_power': True}
    return {'check': 'battery', 'ac_power': battery.power_plugged}


def swap_counts():
    if supported_system() == 'Linux':
        swap = psutil.swap_memory()
        return {'Swapins': swap.sin, 'Swapouts': swap.sout}
    raw = subprocess.check_output(['vm_stat'], text=True)
    page_size = int(re.search(r'page size of (\d+) bytes', raw).group(1))
    return {key: int(re.search(rf'^{key}:\s+(\d+)', raw, re.M).group(1)) * page_size
            for key in ('Swapins', 'Swapouts')}


def storage_info(path):
    """Resolve the actual volume even for a not-yet-created campaign directory."""
    resolved = Path(path).resolve()
    existing = resolved
    while not existing.exists():
        existing = existing.parent
    mounts = [p for p in psutil.disk_partitions(all=True)
              if resolved.is_relative_to(Path(p.mountpoint))]
    if not mounts:
        raise RuntimeError(f'Cannot identify benchmark filesystem: {resolved}')
    mount = max(mounts, key=lambda p: len(Path(p.mountpoint).parts))
    return {'path': str(resolved), 'probe_path': str(existing),
            'mountpoint': mount.mountpoint, 'fstype': mount.fstype, 'device': mount.device}


def require_disk_storage(path):
    info = storage_info(path)
    if info['fstype'] in {'tmpfs', 'ramfs'}:
        raise ValueError(f"Benchmark storage must be disk-backed: {info}")
    return info


def peak_rss_bytes():
    # getrusage reports bytes on macOS, KiB on Linux.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value * (1024 if supported_system() == 'Linux' else 1)


def machine_metadata(path):
    if supported_system() == 'Darwin':
        cpu = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    else:
        cpu = subprocess.check_output(['lscpu'], text=True)
    return {'cpu': cpu, 'ram': psutil.virtual_memory().total,
            'power': power_status(), 'logical_cpus': psutil.cpu_count(),
            'storage': storage_info(path)}
