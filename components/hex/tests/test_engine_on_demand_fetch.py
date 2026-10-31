"""An engine woof ships on demand is staged on first use, not refused (woof 2.9.10, PyPI per-file limit).

THE BREAKAGE: woof 2.9.10 leaves rw_mpas_init_v831 and rw_mpas_gsl_oro to its release bundle so its platform
wheels stay under PyPI's 100,000,000-byte limit. woof.hex.engines.resolve read only the directories, so a plain
install's `woof hex init --physics-suite hrrrv5` refused with "rw_mpas_init_v831 not found" until the user ran
`woof fetch-bridges` by hand.
"""
import sys
import types
from pathlib import Path

import pytest

from woof.hex import engines


def _fake_gpuwm(monkeypatch, tmp_path, *, on_demand, stages):
    staged = tmp_path / "bridges" / engines.executable_name(engines.INIT_RRFS.name)
    calls = []

    def fetch(name):
        calls.append(name)
        if stages:
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(b"#!/bin/sh\n")
            staged.chmod(0o755)
        return stages

    bridges = types.SimpleNamespace(fetched_on_demand=lambda name: on_demand, fetch_on_demand=fetch)
    package = types.ModuleType("woof")
    package.bridges = bridges
    monkeypatch.setitem(sys.modules, "woof", package)
    monkeypatch.setitem(sys.modules, "woof.bridges", bridges)
    monkeypatch.setattr(engines, "bundled_candidate", lambda spec, **_: tmp_path / "nothing" / spec.name)
    monkeypatch.setattr(engines, "gpuwm_candidates", lambda spec: (staged,))
    monkeypatch.setattr(engines.shutil, "which", lambda name: None)
    for name in (*engines.INIT_RRFS.env_names, engines.INIT_RRFS.gpuwm_env):
        monkeypatch.delenv(name, raising=False)
    return staged, calls


def test_an_on_demand_engine_is_staged_on_first_use(monkeypatch, tmp_path):
    staged, calls = _fake_gpuwm(monkeypatch, tmp_path, on_demand=True, stages=True)
    assert engines.resolve(engines.INIT_RRFS) == staged.resolve()
    assert calls == [engines.INIT_RRFS.name]


def test_an_engine_not_on_demand_is_refused_without_a_download(monkeypatch, tmp_path):
    _, calls = _fake_gpuwm(monkeypatch, tmp_path, on_demand=False, stages=True)
    with pytest.raises(engines.EngineRefusal, match="rw_mpas_init_v831 not found"):
        engines.resolve(engines.INIT_RRFS)
    assert calls == []


def test_a_failed_stage_keeps_the_refusal_and_its_remedy(monkeypatch, tmp_path):
    _, calls = _fake_gpuwm(monkeypatch, tmp_path, on_demand=True, stages=False)
    with pytest.raises(engines.EngineRefusal, match="rw_mpas_init_v831 not found"):
        engines.resolve(engines.INIT_RRFS)
    assert calls == [engines.INIT_RRFS.name]


def test_locate_never_downloads(monkeypatch, tmp_path):
    _, calls = _fake_gpuwm(monkeypatch, tmp_path, on_demand=True, stages=True)
    assert engines.locate(engines.INIT_RRFS)[0] is None
    assert calls == []
