"""The refusals the second combo sweep left standing, each fixed or named.

The 2026-10-07 round-2 sweep refused 110 runs and crashed 3; its check
found that the restored RUC soil-moisture refusal named no breakage and that
four groups and one crash were deferred as "not cheap".  Each test pins one:

* RUC and the state health gate refused real.exe's small negative deep-soil
  moisture (24 runs), which stock WRF integrates.
* The NSSL door required QNCCN, and for mp 22 the hail fields, which the
  deprecated variants 17 and 22 do not carry (26 runs).
* Noah-mosaic cold-start carriers and urban morphology records were
  "unmapped" (30 runs); WRF rebuilds the first and reads the second only
  where BUILD_HEIGHT > 0 on a city column, which is now the refusal.
* The GSL drag's large-scale set was unmapped (16 runs) and topo_wind /
  gwd_opt had no static field set on this door (14 runs).
* New Tiedtke handed up an ice tendency on a Kessler run, which the driver
  refused mid-run (A028, 3 runs); WRF's driver drops it (F_QI gate).
"""
from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from woof.ingest import wrfinput as wi
from test_wrfinput_physics_fields import _cfg, _dimensions, _input, _surface_values


# -- RUC soil moisture ------------------------------------------------------

def test_ruc_cold_start_takes_real_exe_negative_deep_soil_moisture():
    from woof.core.ruc import ruc_initialize_cold_start
    tslb = np.full((9, 2, 3), 285.0, np.float32)
    smois = np.full((9, 2, 3), 0.25, np.float32)
    smois[7:, 0, 0] = -0.032          # the gulf-coast case's deepest words
    init = ruc_initialize_cold_start(
        tslb, smois, np.full((2, 3), 6), np.full((2, 3), 10),
        np.zeros((2, 3), np.float32))
    # ruclsminit: an unfrozen land layer gets sh2o = smois, word for word.
    np.testing.assert_array_equal(init.sh2o[7:, 0, 0], smois[7:, 0, 0])
    with pytest.raises(ValueError, match="finite"):
        bad = smois.copy()
        bad[0, 0, 0] = np.nan
        ruc_initialize_cold_start(tslb, bad, np.full((2, 3), 6),
                                  np.full((2, 3), 10), np.zeros((2, 3), np.float32))


def test_health_gate_soil_moisture_has_no_floor_and_keeps_its_ceiling():
    from woof.core.health import rule_for_field
    for name in ("surface.smois", "surface.sh2o"):
        rule = rule_for_field(name)
        assert rule.status_class == "soil_moisture"
        assert rule.lower is None and rule.upper == 1.0


def test_health_gate_mirror_passes_negative_input_words_and_fails_runaways():
    from woof.core import health
    rule = health.rule_for_field("surface.smois")
    words = np.full((9, 2, 3), 0.25, np.float32)
    words[8] = -0.032
    assert not health._bad_mask(words, rule).any()
    words[0, 0, 0] = 1.5
    words[1, 0, 0] = np.nan
    assert int(health._bad_mask(words, rule).sum()) == 2


# -- NSSL variants ----------------------------------------------------------

_MODE17 = dict(nssl_2moment_on=1, nssl_hail_on=1, nssl_ccn_on=0, nssl_density_on=2,
               nssl_3moment=0)
_MODE22 = dict(nssl_2moment_on=1, nssl_hail_on=0, nssl_ccn_on=0, nssl_density_on=1,
               nssl_3moment=0)


def test_nssl_inventory_follows_the_resolved_variant():
    full = set(wi.active_moisture_inventory(SimpleNamespace(moist=True, mp_physics=18))[0])
    assert {"QNCCN", "QHAIL", "QNHAIL", "QVHAIL"} <= full
    mode17 = set(wi.active_moisture_inventory(
        SimpleNamespace(moist=True, mp_physics=18, **_MODE17))[0])
    assert full - mode17 == {"QNCCN"}
    mode22 = set(wi.active_moisture_inventory(
        SimpleNamespace(moist=True, mp_physics=18, **_MODE22))[0])
    assert full - mode22 == {"QNCCN", "QHAIL", "QNHAIL", "QVHAIL"}
    # A file of the variant then misses nothing a start needs.
    raw = {name: None for name in mode22 | set(wi.REQUIRED_WRFINPUT)}
    for aliases in wi.ALIASES.values():
        raw[aliases[0]] = None
    cfg22 = SimpleNamespace(moist=True, mp_physics=18, **_MODE22)
    assert wi.missing_required_wrfinput(raw, cfg22) == []
    assert "QNCCN" in wi.missing_required_wrfinput(
        raw, SimpleNamespace(moist=True, mp_physics=18))


def test_runconfig_variant_defaults_keep_the_full_option18_inventory():
    cfg = _cfg(mp_physics=18)
    assert set(wi.nssl_moisture_wrfinput(cfg)) == set(wi.NSSL_MOISTURE_WRFINPUT)


# -- Noah mosaic and urban records ------------------------------------------

def _mosaic_urban_input(tmp_path, cfg, *, height=0.0, at=None):
    values = _surface_values(cfg)
    path = _input(tmp_path / "input", cfg, values)
    ny, nx = cfg.ny, cfg.nx
    with netCDF4.Dataset(path, "a") as ds:
        ds.createDimension("mosaic", 3)
        ds.createDimension("land_cat_stag", 21)
        for name in ("TSK_MOSAIC", "QSFC_MOSAIC", "CANWAT_MOSAIC", "SNOW_MOSAIC",
                     "SNOWH_MOSAIC", "SNOWC_MOSAIC"):
            ds.createVariable(name, "f4", ("Time", "mosaic", "south_north", "west_east"))[:] = 0.
        for name in ("LANDUSEF2", "MOSAIC_CAT_INDEX"):
            ds.createVariable(name, "f4" if name == "LANDUSEF2" else "i4",
                              ("Time", "land_cat_stag", "south_north", "west_east"))[:] = 0
        for name in ("BUILD_AREA_FRACTION", "BUILD_HEIGHT", "BUILD_SURF_RATIO",
                     "MH_URB2D", "STDH_URB2D", "Z0_URB2D", "LF_URB2D_S"):
            ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))[:] = 0.
        if at is not None:
            ds.variables["BUILD_HEIGHT"][0, at[0], at[1]] = height
    return path


def test_reader_takes_mosaic_and_urban_records_by_name(tmp_path):
    cfg = _cfg(sf_surface_physics=2, num_soil_layers=4)
    mosaic_urban = SimpleNamespace(**{k: getattr(cfg, k) for k in (
        "moist", "mp_physics", "sf_surface_physics", "num_soil_layers", "nx", "ny",
        "nz", "bl_pbl_physics", "sf_sfclay_physics")}, sf_surface_mosaic=1,
        sf_urban_physics=1)
    path = _mosaic_urban_input(tmp_path, cfg)
    with pytest.raises(ValueError, match="unmapped WRF variable"):
        wi.read_wrfinput(path, cfg=cfg, expected_dimensions=_dimensions(cfg),
                         require_complete=False)
    from woof.ingest.wrfinput_surface_statics import surface_static_dispositions
    dispositions = surface_static_dispositions(mosaic_urban)
    assert "TSK_MOSAIC" in dispositions and "BUILD_HEIGHT" in dispositions
    assert "AHE" in dispositions
    assert surface_static_dispositions(replace(cfg)) == {}


def test_reader_accepts_them_with_the_options_set(tmp_path, monkeypatch):
    cfg = _cfg(sf_surface_physics=2, num_soil_layers=4)
    path = _mosaic_urban_input(tmp_path, cfg)
    from woof.ingest import wrfinput_surface_statics as statics
    real = statics.surface_static_dispositions
    monkeypatch.setattr(statics, "surface_static_dispositions",
                        lambda c: real(SimpleNamespace(sf_surface_mosaic=1,
                                                       sf_urban_physics=1)))
    restored = wi.read_wrfinput(path, cfg=cfg, expected_dimensions=_dimensions(cfg),
                                require_complete=False)
    for name in ("TSK_MOSAIC", "LANDUSEF2", "BUILD_HEIGHT", "Z0_URB2D"):
        assert name not in restored.raw
        assert name in restored.surface_input_dispositions


@pytest.mark.parametrize("height, city, refused", [
    (0.0, True, False),     # real.exe without gridded urban data: WRF's table branch
    (12.0, False, False),   # positive off the city: WRF zeroes it too
    (12.0, True, True),     # positive on a city column: WRF keeps it, the port would not
])
def test_gridded_urban_morphology_on_a_city_column_is_refused_by_its_breakage(
        tmp_path, height, city, refused):
    cfg = _cfg(sf_surface_physics=2, num_soil_layers=4)
    path = _mosaic_urban_input(tmp_path, cfg, height=height, at=(1, 2))
    from woof.ingest.wrfinput_surface_statics import require_table_urban_morphology
    restored = SimpleNamespace(path=path, dimensions={"south_north": cfg.ny,
                                                      "west_east": cfg.nx})
    utype = np.zeros((cfg.ny, cfg.nx), np.int32)
    if city:
        utype[1, 2] = 1
    if refused:
        with pytest.raises(ValueError, match="different building geometry"):
            require_table_urban_morphology(restored, utype)
    else:
        require_table_urban_morphology(restored, utype)


# -- terrain drag statics ---------------------------------------------------

_LS = ("VARLS", "CONLS", "OA1LS", "OA2LS", "OA3LS", "OA4LS",
       "OL1LS", "OL2LS", "OL3LS", "OL4LS")


def _orographic_input(tmp_path, cfg, names):
    values = _surface_values(cfg)
    values["HGT"] = np.full((cfg.ny, cfg.nx), 250.0, np.float32)
    path = _input(tmp_path / "input", cfg, values)
    with netCDF4.Dataset(path, "a") as ds:
        for index, name in enumerate(names):
            ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))[:] = (
                np.arange(cfg.ny * cfg.nx, dtype=np.float32).reshape(cfg.ny, cfg.nx)
                + np.float32(index))
    return path


def test_gsl_large_scale_set_is_no_longer_unmapped(tmp_path):
    cfg = _cfg(sf_surface_physics=2, num_soil_layers=4)
    path = _orographic_input(tmp_path, cfg, _LS)
    restored = wi.read_wrfinput(path, cfg=cfg, expected_dimensions=_dimensions(cfg),
                                require_complete=False)
    # The 2.8.9 door already carries these real producers in raw state.
    # Keeping them is part of the default-mode restoration contract.
    assert set(_LS) <= set(restored.raw)
    with netCDF4.Dataset(path) as dataset:
        for name in _LS:
            np.testing.assert_array_equal(
                np.asarray(restored.raw[name], np.float32).view(np.uint32),
                np.asarray(dataset.variables[name][0], np.float32).view(np.uint32))


@pytest.mark.parametrize("topo_wind, gwd_opt", [(1, 0), (2, 0), (0, 1), (2, 1), (0, 3)])
def test_the_door_hands_the_drag_the_files_own_statistics(tmp_path, topo_wind, gwd_opt):
    from woof.static.orographic import (GSL_SS_FIELDS, GWDO_FIELDS,
                                         required_static_fields)
    from woof.ingest.wrfinput_surface_statics import terrain_drag_static
    cfg = _cfg(sf_surface_physics=2, num_soil_layers=4)
    names = ("VAR_SSO",) + GWDO_FIELDS + _LS + GSL_SS_FIELDS
    path = _orographic_input(tmp_path, cfg, names)
    restored = wi.read_wrfinput(path, cfg=cfg, expected_dimensions=_dimensions(cfg),
                                require_complete=False)
    static = terrain_drag_static(restored, SimpleNamespace(topo_wind=topo_wind,
                                                           gwd_opt=gwd_opt))
    need = required_static_fields(topo_wind, gwd_opt)
    assert set(static) == set(need) | {"HGT_M"}
    with netCDF4.Dataset(path) as ds:
        for name in need:
            np.testing.assert_array_equal(static[name], ds.variables[name][0])
    np.testing.assert_array_equal(static["HGT_M"], restored.raw["HGT"])
    assert terrain_drag_static(restored, SimpleNamespace(topo_wind=0, gwd_opt=0)) is None


def test_a_statistic_the_file_lacks_is_left_for_the_drag_to_refuse_by_name(tmp_path):
    from woof.ingest.wrfinput_surface_statics import terrain_drag_static
    cfg = _cfg(sf_surface_physics=2, num_soil_layers=4)
    path = _orographic_input(tmp_path, cfg, ("VAR",))
    restored = wi.read_wrfinput(path, cfg=cfg, expected_dimensions=_dimensions(cfg),
                                require_complete=False)
    static = terrain_drag_static(restored, SimpleNamespace(topo_wind=1, gwd_opt=0))
    assert "VAR_SSO" not in static      # build_terrain_drag names it


# -- New Tiedtke on a run with no ice ---------------------------------------

class _FakeWorkspace:
    def __init__(self, nz, width):
        self.arrays = {}
        self.nz, self.width = nz, width

    def bind(self, name, kind):
        shape = (self.width,) if kind == 1 else (self.nz + 1, self.width)
        if name not in self.arrays:
            self.arrays[name] = np.full(shape, 0.5, np.float32)
        return self.arrays[name]


class _FakePipeline:
    def __init__(self, nz, width):
        self.w = _FakeWorkspace(nz, width)
        self.stages = SimpleNamespace(_llo3_mask=np.zeros(1))
        self.scalars = {}

    def begin_llo3_batch(self):
        pass

    def run_chunk(self):
        pass

    def end_llo3_batch(self):
        pass


@pytest.fixture
def forget_modules_imported_under_fake_cupy():
    """Drop first-party modules first imported while ``cupy`` was faked.

    A module imported while ``sys.modules["cupy"]`` is numpy binds ``cp``
    to numpy for the rest of the process; monkeypatch restores the
    ``cupy`` entry but not those bindings.  On a card host that turned
    ``tests/test_health.py``'s real-GPU inventory test into "GPU health
    field 'held.pbl.ru' is not a CUDA array" whenever it ran after this
    file (W1, 2026-10-07, development branch).
    """
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.split(".")[0] in ("woof", "tilestream"):
            del sys.modules[name]


class _NumpyAsCupy:
    """numpy standing in for cupy on this CPU test.

    Every attribute is numpy's; ``get_array_module``, which the donor
    depletion bound asks of its arrays (woof/core/ntiedtke.py,
    bound_donor_depletion), answers numpy as it would for host arrays.
    """

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def get_array_module(*arrays, **kwargs):
        return np


@pytest.mark.parametrize("carries_ice", [False, True])
@pytest.mark.usefixtures("forget_modules_imported_under_fake_cupy")
def test_new_tiedtke_follows_wrfs_f_qi_gate(monkeypatch, carries_ice):
    from woof.core import ntiedtke
    monkeypatch.setitem(sys.modules, "cupy", _NumpyAsCupy())
    nz, ny, nx = 3, 2, 2
    monkeypatch.setattr(ntiedtke, "_nt_multiprocessor_count", lambda: 1)
    # nt_tile_columns(ncol, multiprocessor_count, columns_per_sm): the
    # per-instance tile cap is the caller's third argument (a96209b59d3).
    monkeypatch.setattr(ntiedtke, "nt_tile_columns",
                        lambda ncol, sms, per_sm=ntiedtke.NT_TILE_COLUMNS_PER_SM: ncol)
    scheme = ntiedtke.NewTiedtke()
    monkeypatch.setattr(scheme, "_for", lambda width, nz_, cfg: _FakePipeline(nz_, width))
    zeros = np.zeros((nz, ny, nx), np.float32)
    state = SimpleNamespace(p=zeros, w=np.zeros((nz + 1, ny, nx), np.float32),
                            qc=zeros, qi=zeros if carries_ice else None,
                            elapsed_seconds=0.0)
    atmosphere = {name: zeros for name in ("exner", "temperature", "qv", "qc", "qi",
                                           "u", "v", "pressure", "dz", "rho")}
    atmosphere["p_interface"] = np.zeros((nz + 1, ny, nx), np.float32)
    fields = {name: np.zeros((ny, nx), np.float32) for name in ("xland", "hfx", "qfx")}
    result = scheme(atmosphere=atmosphere, fields=fields, state=state,
                    cfg=SimpleNamespace(dx=3000.0, dt=1.0))
    assert result.rqccuten is not None
    assert (result.rqicuten is not None) == carries_ice


# -- the BEP / BEM crash the urban refusal hid ------------------------------

@pytest.mark.usefixtures("forget_modules_imported_under_fake_cupy")
def test_a_signed_32_bit_status_word_records_into_the_deferred_ledger(monkeypatch):
    """BEP's error_flags is int32 (a Fortran line number); recording it into
    the uint32 ledger slot promoted to int64 and raised, which crashed every
    BEP and BEM replay at its first urban call."""
    from woof.core import health_ledger
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        zeros=np.zeros, uint32=np.uint32, bitwise_or=np.bitwise_or,
        asnumpy=np.array))   # a copy, as a device-to-host read is
    ledger = health_ledger.HealthLedger(4)
    seen = []

    def refuse(flags):
        seen.append(flags)
        raise RuntimeError(f"BEP stopped at line {flags}")

    ledger.record("urban_bep", np.array([0], np.int32), refuse)
    ledger.drain()                      # a zero word is no fault
    ledger.record("urban_bep", np.array([3240], np.int32), refuse)
    with pytest.raises(RuntimeError, match="3240"):
        ledger.drain()
    assert seen == [3240]
    with pytest.raises(TypeError, match="32-bit integer"):
        ledger.record("urban_bep", np.array([1.0], np.float32), refuse)
