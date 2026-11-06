"""Portability must preserve resource units and reject RAM-backed timing paths."""
from types import SimpleNamespace as NS

import pytest

from hybrid_recall import host


@pytest.mark.parametrize('name', ['cc', 'c++', 'gcc-15', 'clang-20',
                                 '/usr/bin/x86_64-linux-gnu-gcc-15',
                                 'aarch64-linux-gnu-g++', 'ld.lld-20', 'cargo', 'rustc'])
def test_native_compiler_aliases_and_supervisor_detection(monkeypatch, name):
    assert host.is_native_build(name)
    monkeypatch.setattr(host.subprocess, 'check_output',
                        lambda *a, **kw: f'123 {name}\n124 clangd\n125 python3\n')
    assert host.competing_builds() == [f'123 {name}']


@pytest.mark.parametrize('system,multiplier', [('Linux', 1024), ('Darwin', 1)])
def test_peak_rss_units(monkeypatch, system, multiplier):
    monkeypatch.setattr(host.platform, 'system', lambda: system)
    monkeypatch.setattr(host.resource, 'getrusage', lambda _: NS(ru_maxrss=1024))
    assert host.peak_rss_bytes() == 1024 * multiplier


def test_linux_swap_is_already_bytes(monkeypatch):
    monkeypatch.setattr(host.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(host.psutil, 'swap_memory', lambda: NS(sin=8192, sout=4096))
    assert host.swap_counts() == {'Swapins': 8192, 'Swapouts': 4096}


@pytest.mark.parametrize('battery,allowed', [(None, True), (NS(power_plugged=False), False)])
def test_linux_power_distinguishes_server_and_battery(monkeypatch, battery, allowed):
    monkeypatch.setattr(host.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(host.psutil, 'sensors_battery', lambda: battery)
    assert host.power_status()['ac_power'] is allowed


def test_nested_ram_mount_is_rejected_even_before_database_exists(tmp_path, monkeypatch):
    mounts = [NS(mountpoint='/', fstype='ext4', device='/dev/root'),
              NS(mountpoint=str(tmp_path), fstype='tmpfs', device='tmpfs')]
    monkeypatch.setattr(host.psutil, 'disk_partitions', lambda all: mounts)
    with pytest.raises(ValueError, match='disk-backed'):
        host.require_disk_storage(tmp_path / 'new' / 'database')


def test_storage_resolves_symlink(tmp_path, monkeypatch):
    disk = tmp_path / 'disk'
    disk.mkdir()
    link = tmp_path / 'link'
    link.symlink_to(disk, target_is_directory=True)
    monkeypatch.setattr(host.psutil, 'disk_partitions', lambda all: [
        NS(mountpoint=str(disk), fstype='ext4', device='/dev/disk')])
    info = host.require_disk_storage(link / 'new')
    assert info['probe_path'] == str(disk)
    assert info['device'] == '/dev/disk'
