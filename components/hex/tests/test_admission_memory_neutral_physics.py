"""--physics overrides that change no memory leave the admission price unchanged at all three gates.

THE BREAKAGE THIS PREVENTS (global16 prove, 2026-11-06, woof hex 0.4.13 and 0.4.14): any ``--physics KEY=VALUE``
disqualified the measured compact row at the request gate, the MPAS-route native gate and the MPAS rank gate, so
``--physics config_sst_update=.true.`` (MPAS SST/sea-ice updating, which the s2s door passes on the MPAS route) moved
the 60 km global (x1.163842) on a 16 GB RTX 5070 Ti from its 15,623,782,400-byte measured row to the legacy row and
refused it (0.4.13: 27,659.5 MiB and 29,930,042,163 bytes; 0.4.14: 20,918,671,770 bytes).  The override only switches
alarms and reads per-cell surface planes.  A key that does change the footprint still takes the legacy row.
"""
import contextlib
import sys
import types
from types import SimpleNamespace

import netCDF4
import pytest

from woof.hex import device_admission as da
from woof.hex import forecast_door as fd
from woof.hex.mpasphys import memory_admission as ma

CARD16 = da.KNOWN_CARDS["16gib-70sm"]
CELLS60, LEVELS = 163_842, 55
FREE16 = 16_368_795_648
SST = ["config_sst_update=.true."]
MICROP = ["config_microp_scheme=mp_wsm6"]


@pytest.fixture(autouse=True)
def _seams(monkeypatch, tmp_path):
    from woof.hex import numerics
    monkeypatch.setattr(numerics, "_mode", "fast")
    for key in ("WOOF_HEX_MEMBER_MEMORY", "WOOF_HEX_THOMPSON_WORKSPACE", "WOOF_HEX_THOMPSON_TILE_COLUMNS",
                "WOOF_HEX_THOMPSON_WORKSPACE_MIB", "WOOF_HEX_TRANSPORT_WORKSPACE_BYTES",
                "WOOF_HEX_LIMITER_SPECIES_PER_PASS", "GPUWM_HEX_POOL_RELEASE_MIN_BYTES",
                "GPUWM_HEX_RELEASE_HEADROOM", "WOOF_HEX_POOL_RELEASE", "GPUWM_HEX_LMEM_RELEASE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace(cuda=types.SimpleNamespace(
        Device=lambda device=0: contextlib.nullcontext(), runtime=types.SimpleNamespace(is_hip=False))))
    monkeypatch.setattr(fd, "read_card_profile", lambda device=None: CARD16)
    monkeypatch.setattr(fd, "measure_device_memory", lambda: (FREE16, 16_303 * da.MIB))


def _init(tmp_path):
    path = tmp_path / "init.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("nCells", CELLS60)
        ds.createDimension("nVertLevels", LEVELS)
        ds.createDimension("nVertLevelsP1", LEVELS + 1)
    return path


def _arguments(tmp_path, physics):
    return SimpleNamespace(init=str(_init(tmp_path)), out=str(tmp_path / "out"), mesh="hex-x1.163842-native",
                           samehost_devices=None, lbc_dir=None, physics=physics, physics_profile=None,
                           physics_namelist=None, physics_default_backend=None, local_timestep=False,
                           fullstep_local_timestep=False, stop_on_refusal=False, headroom_mib=None, stochastic=None,
                           physics_suite=None)


def _native_gate(tmp_path, physics):
    arguments = _arguments(tmp_path, physics)
    try:
        price = fd.admit_native_device(arguments, SimpleNamespace(suite="convection_permitting")).required_bytes
    except fd.ForecastDoorRefusal:
        price = None
    return price, arguments


def _rank_gate(arguments):
    try:
        return ma.admit(arguments)["required_bytes"][0]
    except ValueError:
        return None


def _request_gate(tmp_path, overrides):
    request = SimpleNamespace(init=_init(tmp_path), model=None, lbc_dir=None, stop_on_refusal=False,
                              physics_overrides=overrides, physics_backend="wsm6_column", local_timestep=False,
                              fullstep_local_timestep=False, physics_suite=None, cells=CELLS60)
    model = fd.resolve_admission_model(request)
    return type(model).__name__, model.required_bytes(CELLS60)


def test_neutral_keys_are_named_and_others_move_the_footprint():
    assert not da.physics_overrides_move_footprint(None)
    assert not da.physics_overrides_move_footprint(SST + ["config_sstdiurn_update=.true.",
                                                          "config_deepsoiltemp_update=.true."])
    assert not da.physics_overrides_move_footprint((("config_sst_update", True),))
    assert da.physics_overrides_move_footprint(MICROP)
    assert da.physics_overrides_move_footprint(SST + MICROP)
    assert da.physics_overrides_move_footprint((("config_pbl_scheme", "bl_ysu"),))


def test_sst_update_leaves_the_price_unchanged_at_all_three_gates(tmp_path):
    bare, bare_args = _native_gate(tmp_path, None)
    sst, sst_args = _native_gate(tmp_path, SST)
    assert bare is not None and bare < 16_000_000_000
    assert sst == bare
    assert _rank_gate(sst_args) == _rank_gate(bare_args) == bare
    assert _request_gate(tmp_path, (("config_sst_update", True),)) == _request_gate(tmp_path, ())


def test_a_footprint_key_still_takes_the_legacy_row(tmp_path):
    bare, _ = _native_gate(tmp_path, None)
    microp, microp_args = _native_gate(tmp_path, MICROP)
    assert microp is None                       # legacy row, refused on the 16 GB card as before
    assert _rank_gate(microp_args) is None
    assert ma.qualified_defaults(_arguments(tmp_path, SST)) and not ma.qualified_defaults(_arguments(tmp_path, MICROP))
    assert _request_gate(tmp_path, (("config_microp_scheme", "mp_wsm6"),))[0] != "CompactFootprintModel"
