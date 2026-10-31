"""The WRF-input door's checkpoint identity leaves out what a restart may change.

V2W WoFS, 2026-10-29: resuming an nwp_diagnostics = 0 checkpoint with
nwp_diagnostics = 1 written into namelist.input refused "tree restart d01 was
written for a different run: experiment_identity, preparation_receipt_sha256
differ(s)", because the import receipt binds the namelist bytes.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from woof.wrfinput_forecast import WrfInitialization

_NAMELIST = """&time_control
 run_hours = 6,
 history_interval = 60,
 restart = .false.,
 debug_level = 0,
/
&physics
 mp_physics = 8,
/
"""


def _identity(tmp_path, name, namelist_text):
    namelist = tmp_path / f"{name}.input"
    namelist.write_text(namelist_text, encoding="utf-8")
    keys = [{"key": key, "section": section} for section, key in (
        ("time_control", "run_hours"), ("time_control", "history_interval"),
        ("physics", "mp_physics"))]
    if "nwp_diagnostics" in namelist_text:
        keys.append({"key": "nwp_diagnostics", "section": "time_control"})
    receipt = tmp_path / f"{name}.json"
    receipt.write_bytes(json.dumps({
        "schema": "gpuwm-wrf-input-import-v1",
        "source_files": {
            "namelist_input": hashlib.sha256(namelist.read_bytes()).hexdigest(),
            "wrfbdy": "0" * 64},
        "namelist_translation": {"translated": keys, "dropped": []},
        "namelist_translation_text": f"{len(keys)} key(s)"}).encode("utf-8"))
    initialization = object.__new__(WrfInitialization)
    initialization.inputs = SimpleNamespace(
        artifact_paths={"preparation_receipt": receipt, "namelist_input": namelist},
        authority_sha256={"preparation_receipt": hashlib.sha256(receipt.read_bytes()).hexdigest()})
    return initialization.preparation_receipt_sha256()


def test_restart_adjustable_namelist_entries_leave_the_identity(tmp_path):
    base = _identity(tmp_path, "base", _NAMELIST)
    flipped = _NAMELIST.replace(" debug_level = 0,", " debug_level = 0,\n nwp_diagnostics = 1,")
    assert _identity(tmp_path, "nwp", flipped) == base
    cadence = _NAMELIST.replace("history_interval = 60", "history_interval = 15") \
                       .replace(" restart = .false.,", " restart = .false.,\n restart_interval = 30,")
    assert _identity(tmp_path, "cadence", cadence) == base


def test_every_other_namelist_entry_still_binds(tmp_path):
    base = _identity(tmp_path, "base", _NAMELIST)
    assert _identity(tmp_path, "mp", _NAMELIST.replace("mp_physics = 8", "mp_physics = 28")) != base


def test_a_namelist_changed_after_its_receipt_refuses(tmp_path):
    _identity(tmp_path, "base", _NAMELIST)
    initialization = object.__new__(WrfInitialization)
    receipt = tmp_path / "base.json"
    namelist = tmp_path / "base.input"
    namelist.write_text(_NAMELIST.replace("mp_physics = 8", "mp_physics = 28"), encoding="utf-8")
    initialization.inputs = SimpleNamespace(
        artifact_paths={"preparation_receipt": receipt, "namelist_input": namelist},
        authority_sha256={"preparation_receipt": hashlib.sha256(receipt.read_bytes()).hexdigest()})
    with pytest.raises(RuntimeError, match="changed after its import receipt"):
        initialization.preparation_receipt_sha256()
