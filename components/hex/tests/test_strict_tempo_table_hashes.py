"""Strict numerics refuses a TEMPO lookup file that is not the strict packet's, before the run starts.

THE BREAKAGE (hrrrv5 x1.40962, strict, a test host, 2026-11-06): the named table directories did not hold the strict
reference's lookup files, so TEMPO silently read the fork source's NOAA copies (freeze_water_data_tempo_v3 sha256
4624...) where native and a test host read the strict-rebuilt 9cf5..., and the bitwise comparison ran on other tables.
"""
from __future__ import annotations

import hashlib
import inspect

import pytest

from woof.hex import numerics
from woof.hex.mpasphys import runtime_v842
from woof.hex.mpasphys.rrfs import framework_init_rrfs as fi

FILES = [row[0] for row in fi.FrameworkInitializationRRFS.TEMPO_LOOKUP_FILES]
CONFIGS = {"config_microp_scheme": "mp_tempo"}
SOURCE = {"resource": "mpas-source-rrfs-44c0849", "pin_set": "rrfs-44c0849"}


def _tables(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(numerics, "_mode", mode)
    directory = tmp_path / "tables"
    directory.mkdir()
    digests = {}
    for name in FILES:
        data = f"{name} strict words".encode()
        (directory / name).write_bytes(data)
        digests[name] = hashlib.sha256(data).hexdigest()
    return directory, digests


def _packet(digests):
    return {"mp_tempo": {"session_tables": True, "lookup_sha256": dict(digests)}}


def test_strict_refuses_a_lookup_file_that_is_not_the_strict_packets(monkeypatch, tmp_path):
    directory, digests = _tables(monkeypatch, tmp_path, "strict")
    (directory / "freeze_water_data_tempo_v3").write_bytes(b"the fork source's NOAA copy")
    actual = hashlib.sha256(b"the fork source's NOAA copy").hexdigest()
    with pytest.raises(ValueError) as refused:
        fi.FrameworkInitializationRRFS.verify_strict_tables(CONFIGS, [directory], _packet(digests), **SOURCE)
    message = str(refused.value)
    assert "\n" not in message
    assert "freeze_water_data_tempo_v3" in message
    assert actual in message and digests["freeze_water_data_tempo_v3"] in message
    assert "qr_acr_qs_data_tempo_v3" not in message


def test_strict_runs_when_every_lookup_file_is_the_strict_packets(monkeypatch, tmp_path):
    directory, digests = _tables(monkeypatch, tmp_path, "strict")
    seen = fi.FrameworkInitializationRRFS.verify_strict_tables(CONFIGS, [directory], _packet(digests), **SOURCE)
    assert seen == digests


def test_strict_without_packet_digests_checks_the_pinned_fork_files(monkeypatch, tmp_path):
    directory, _ = _tables(monkeypatch, tmp_path, "strict")
    with pytest.raises(ValueError, match="pinned rrfs-44c0849 fork file"):
        fi.FrameworkInitializationRRFS.verify_strict_tables(CONFIGS, [directory], None, **SOURCE)


def test_fast_runs_are_not_checked(monkeypatch, tmp_path):
    directory, digests = _tables(monkeypatch, tmp_path, "fast")
    (directory / "freeze_water_data_tempo_v3").write_bytes(b"other words")
    assert fi.FrameworkInitializationRRFS.verify_strict_tables(
        CONFIGS, [directory], _packet(digests), **SOURCE) is None


def test_the_forecast_checks_before_the_dycore_inputs():
    source = inspect.getsource(runtime_v842._run_native_dynamics)
    assert source.index("verify_strict_tables") < source.index("'dycore-inputs'")
