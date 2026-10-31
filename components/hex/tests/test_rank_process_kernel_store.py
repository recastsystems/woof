"""Rank processes serve their NVRTC requests from the forecast process's warm kernel store (woof.hex.rank_processes).

THE BREAKAGE THIS PREVENTS (woof hex HB18, RTX 5090, lam7k hrrrv5, 2026-10-29): the forecast process
installed the warm store (kernel_plan.install -> kernel_warm.install) but each rank process, a fresh ``python -m``
interpreter, never did, so every direct/raw NVRTC unit (RUC LSM, MYNN surface layer, transport, acoustic, regional)
compiled again on one core in every rank process of every run, warm cache or not.  Step 1 of every multi-rank run
waited for those compiles.  These check, on the CPU, that the parent's store reaches each child's environment and
that both rank-process workers install it before their request arrives.
"""
from __future__ import annotations

import inspect
import json
import socket

import pytest

from woof.hex import kernel_warm
from woof.hex import rank_processes as rp

CONFIG = {"root": "/cache/kernel-packs", "tag": '{"argv": ["forecast"]}',
          "identity": {"cupy": "14.2.0", "nvrtc": [13, 0], "nvrtc_library_sha256": "ab" * 32}}


def test_parent_store_text_is_the_worker_config(monkeypatch):
    monkeypatch.setattr(kernel_warm, "worker_config", lambda: dict(CONFIG))
    assert json.loads(rp.parent_kernel_store()) == CONFIG
    monkeypatch.setattr(kernel_warm, "worker_config", lambda: None)
    assert rp.parent_kernel_store() is None


def test_child_installs_the_parent_store(monkeypatch):
    seen = []
    monkeypatch.setattr(kernel_warm, "install", lambda **kwargs: seen.append(kwargs))
    monkeypatch.setenv(rp.KERNEL_STORE_ENVIRONMENT, json.dumps(CONFIG))
    rp.install_parent_kernel_store()
    assert len(seen) == 1
    assert str(seen[0]["root"]) == CONFIG["root"]
    assert seen[0]["tag"] == CONFIG["tag"] and seen[0]["identity"] == CONFIG["identity"]


def test_no_parent_store_installs_nothing(monkeypatch):
    seen = []
    monkeypatch.setattr(kernel_warm, "install", lambda **kwargs: seen.append(kwargs))
    monkeypatch.delenv(rp.KERNEL_STORE_ENVIRONMENT, raising=False)
    rp.install_parent_kernel_store()
    assert seen == []


def test_a_store_that_cannot_install_is_reported_and_the_rank_runs_on(monkeypatch, capsys):
    def broken(**_):
        raise OSError("read-only cache")
    monkeypatch.setattr(kernel_warm, "install", broken)
    monkeypatch.setenv(rp.KERNEL_STORE_ENVIRONMENT, json.dumps(CONFIG))
    monkeypatch.setenv("WOOF_HEX_RANK_PROCESS", "3")
    rp.install_parent_kernel_store()
    assert "rank process 3: warm kernel store not installed (OSError: read-only cache)" in capsys.readouterr().err


@pytest.mark.skipif(not rp.supported()[0], reason="needs Linux memfd and fd passing")
def test_every_child_environment_carries_the_parent_store(monkeypatch):
    monkeypatch.setattr(kernel_warm, "worker_config", lambda: dict(CONFIG))
    started = {}

    class Child:
        pid = 0
        returncode = 0

        def __init__(self, command, env, **_):
            started[int(command[4])] = env

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(rp.subprocess, "Popen", Child)
    group = rp.RankProcesses([0, 0, 0], {0: [1], 1: [0, 2], 2: [1]}, module="hexcore.nothing", timeout=5.0, job="j")
    group.start()
    try:
        assert sorted(started) == [1, 2]
        for env in started.values():
            assert json.loads(env[rp.KERNEL_STORE_ENVIRONMENT]) == CONFIG
    finally:
        group.join(1.0)
        group.close()


@pytest.mark.parametrize("module", ["woof.hex.mpasphys.process_ranks_v842", "woof.hex.drivers.samehost_rank_group_v841"])
def test_both_rank_workers_install_the_store_before_their_request(module):
    import importlib
    source = inspect.getsource(importlib.import_module(module).worker)
    assert "install_parent_kernel_store()" in source
    assert source.index("install_parent_kernel_store()") < source.index(".recv(")
