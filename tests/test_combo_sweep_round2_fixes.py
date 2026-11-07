"""The WOOF-side causes the 2026-10-07 combo sweep found, each pinned.

* Strict mode ran with its four controls off unless each was set by name, so
  a ``GPUWM_WRF_EXACT=1`` sweep left WRF at the first pressure diagnosis.
* The WRF-file door refused every ``hybrid_opt = 0`` pair (WRF writes
  ``ETAC = 0`` there) and every no-land-model pair (real.exe writes five soil
  layers, the importer resolved four).
* RUC refused real.exe's small negative deep-soil moisture, which WRF runs.
* The door could not run with every physics scheme off.
"""
from __future__ import annotations

from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from _run_config_fake import run_config_fake

ROOT = Path(__file__).resolve().parents[1]


def _controls(env_extra: dict[str, str]) -> dict:
    """Load woof/wrf_exact.py alone in a fresh process (no package import,
    so no compiler is installed) and report what it selected."""
    code = (
        "import importlib.util, json, sys\n"
        f"spec = importlib.util.spec_from_file_location('wx', {str(ROOT / 'woof' / 'wrf_exact.py')!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "print(json.dumps({'enabled': m.ENABLED, 'controls': m.effective_controls(),\n"
        "                  'options': list(m.STRICT_OPTIONS)}))\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("GPUWM_WRF_EXACT")}
    env.update(env_extra)
    out = subprocess.run([sys.executable, "-c", code], env=env, check=True,
                         capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_strict_mode_turns_every_control_on_by_default():
    got = _controls({"GPUWM_WRF_EXACT": "1"})
    assert got["enabled"] is True
    assert got["controls"] == {"diagnostics": True, "advection": True,
                               "diffusion": True, "bigstep": True}
    for macro in ("GPUWM_WRF_EXACT_D_DIAGNOSTICS", "GPUWM_WRF_EXACT_C_ADVECTION",
                  "GPUWM_WRF_EXACT_C_DIFFUSION", "GPUWM_WRF_EXACT_C_BIGSTEP"):
        assert f"-D{macro}=1" in got["options"]


def test_control_zero_stays_off_and_default_wrf_diffusion_is_selected():
    got = _controls({"GPUWM_WRF_EXACT": "1", "GPUWM_WRF_EXACT_BIGSTEP": "0"})
    assert got["controls"]["bigstep"] is False
    assert got["controls"]["diagnostics"] is True
    assert "-DGPUWM_WRF_EXACT_C_BIGSTEP=1" not in got["options"]
    default = _controls({})
    assert default["enabled"] is False
    # The selected diffusion lane is default-on; the other exact controls
    # still require the explicit strict process switch.
    assert default["controls"] == {"diagnostics": False, "advection": False,
                                   "diffusion": True, "bigstep": False}
    alone = _controls({"GPUWM_WRF_EXACT_BIGSTEP": "1"})
    assert alone["enabled"] is False
    assert alone["controls"] == default["controls"]


def _identity(hybrid_opt: int, etac: float, soil: int = 5):
    from woof.ingest.wrfinput_identity import WrfinputIdentity
    eta = (1.0, 0.5, 0.0)
    return WrfinputIdentity(
        Path("wrfinput_d01"),
        MappingProxyType({"west_east": 4, "south_north": 3, "bottom_top": 2,
                          "soil_layers_stag": soil}),
        MappingProxyType({"USE_THETA_M": 0, "GRID_ID": 1, "MP_PHYSICS": 8,
                          "SF_SURFACE_PHYSICS": 0, "HYBRID_OPT": hybrid_opt,
                          "ETAC": etac, "DX": 3000.0, "DY": 3000.0}),
        eta, 2000.0, datetime(2024, 5, 21, 21))


def _check(identity, *, hybrid_opt, etac, soil_layers=5):
    from woof.ingest.wrfinput_identity import check_wrfinput_identity
    run = SimpleNamespace(nx=4, ny=3, nz=2, mp_physics=8, sf_surface_physics=0,
                          dx=3000.0, dy=3000.0)
    vertical = SimpleNamespace(p_top=2000.0, etac=etac, hybrid_opt=hybrid_opt,
                               eta_levels=(1.0, 0.5, 0.0))
    check_wrfinput_identity(identity, domain=SimpleNamespace(run=run, grid_id=1),
                            vertical=vertical, start_time=datetime(2024, 5, 21, 21),
                            soil_layers=soil_layers)


def test_hybrid_opt_zero_file_carries_etac_zero_and_is_accepted():
    from woof.ingest.wrfinput_identity import written_etac
    assert written_etac(SimpleNamespace(hybrid_opt=0, etac=0.2)) == 0.0
    assert written_etac(SimpleNamespace(hybrid_opt=2, etac=0.2)) == pytest.approx(0.2)
    # WRF's own header for hybrid_opt = 0 (share/output_wrf.F:214-217).
    _check(_identity(0, 0.0), hybrid_opt=0, etac=0.2)
    # A hybrid file still has to carry the namelist's etac.
    with pytest.raises(ValueError, match="ETAC"):
        _check(_identity(2, 0.0), hybrid_opt=2, etac=0.2)


def test_no_land_model_namelist_resolves_wrf_five_soil_layers(tmp_path):
    from test_namelist_import import INPUT_TEXT, _pair
    from woof.namelist_import import import_namelists
    from woof.config import soil_layer_count
    from woof.experiment import load_experiment
    text = INPUT_TEXT.replace(" sf_surface_physics = 2, 2,", " sf_surface_physics = 0, 0,")
    assert text != INPUT_TEXT
    translated, _report = import_namelists(*_pair(tmp_path, inp=text), name="no_lsm")
    output = tmp_path / "no_lsm.toml"
    output.write_text(translated, encoding="utf-8")
    for dc in load_experiment(output).domains:
        assert dc.run.sf_surface_physics == 0
        assert soil_layer_count(dc.run) == 5     # module_check_a_mundo.F:3548-3549
    # The door's identity check against real.exe's five-layer file then passes.
    _check(_identity(2, 0.2, soil=5), hybrid_opt=2, etac=0.2,
           soil_layers=soil_layer_count(dc.run))


def test_soil_columns_without_a_land_model_are_checked_for_finiteness_only():
    """real.exe writes TSLB = 0 everywhere when sf_surface_physics = 0 and no
    scheme reads it; the state health gate's 150..400 K rule refused those
    runs.  With a land model the range rules stand."""
    from woof.core.health import collect_state_fields
    shape = (5, 3, 4)
    fields = {"tslb": np.zeros(shape, np.float32), "smois": np.zeros(shape, np.float32),
              "sh2o": np.zeros(shape, np.float32), "smcrel": np.zeros(shape, np.float32),
              "tsk": np.full(shape[1:], 300.0, np.float32)}

    def rules(lsm_entry):
        driver = SimpleNamespace(fields=fields, scheme_dispatch={"sf_surface_physics": lsm_entry})
        state = SimpleNamespace(physics=driver, p=np.zeros((2, 3, 4), np.float32))
        return {f.name: f.rule for f in collect_state_fields(state)}

    off = rules(None)
    for name in ("surface.tslb", "surface.smois", "surface.sh2o", "surface.smcrel"):
        assert off[name].lower is None and off[name].upper is None, name
    assert off["surface.tsk"].lower == 150.0             # TSK is still read
    on = rules("_run_noah")
    assert on["surface.tslb"].lower == 150.0
    assert on["surface.smois"].upper == 1.0


def test_wrf_file_door_runs_with_every_physics_scheme_off(monkeypatch):
    from woof.ingest import wrfinput
    import woof.core.physics as physics
    called = []
    monkeypatch.setattr(wrfinput, "_validate_supplied_physics_fields",
                        lambda *a, **k: called.append("validated"))
    monkeypatch.setattr(physics, "initialize_physics",
                        lambda *a, **k: pytest.fail("no driver is attached when every scheme is off"))
    restored = SimpleNamespace(raw={}, global_attributes={})
    every_scheme_off = run_config_fake(mp_physics=0, sf_sfclay_physics=0, sf_surface_physics=0,
                                       bl_pbl_physics=0, cu_physics=0, ra_physics=0)
    assert wrfinput.initialize_wrfinput_physics(object(), restored, every_scheme_off) is None
    assert called == ["validated"]


def test_wrf_file_door_history_writes_the_files_grid_words():
    from woof.io.history_layout import (WRF_FILE_GRID_HISTORY_FIELDS,
                                         wrf_file_grid_history_fields)
    rng = np.random.default_rng(1)
    raw = {name: rng.random((3, 4)).astype(np.float64) for name in WRF_FILE_GRID_HISTORY_FIELDS}
    raw["T"] = np.zeros((2, 3, 4))
    got = wrf_file_grid_history_fields(SimpleNamespace(restored=SimpleNamespace(raw=raw)))
    assert sorted(got) == sorted(WRF_FILE_GRID_HISTORY_FIELDS)
    for name, value in got.items():
        assert value.dtype == np.float32
        np.testing.assert_array_equal(value, raw[name].astype(np.float32))
    # Every other door keeps the projection's fields.
    assert wrf_file_grid_history_fields(SimpleNamespace(static_fields={})) == {}


def test_wrf_file_door_history_carries_wrfs_static_words(tmp_path):
    """wrf.exe writes its coordinate constants, directional map factors and
    static surface words to every history frame exactly as its input holds
    them; 2.9.0 wrote none (129 of WRF's 203 history fields were missing on
    NCAR's CONUS benchmark).  They are read from the file itself, float32
    words unchanged, scalars kept 0-D; integer Registry flags stay out."""
    netCDF4 = pytest.importorskip("netCDF4")
    from woof.io.history_layout import wrf_file_grid_history_fields
    from woof.netcdf_bridge import find_netcdf_bin

    # The words are read through the rw_netcdf bridge, as the door reads them.
    if find_netcdf_bin() is None:
        pytest.skip("rw_netcdf is not built in this tree")

    path = tmp_path / "wrfinput_d01"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("bottom_top", 3)
        ds.createDimension("soil_layers_stag", 4)
        ds.createDimension("south_north", 2)
        ds.createDimension("west_east", 2)
        ds.createVariable("C1H", "f4", ("Time", "bottom_top"))[0] = [0.9, 0.5, 0.1]
        ds.createVariable("ZS", "f4", ("Time", "soil_layers_stag"))[0] = [.05, .25, .7, 1.5]
        ds.createVariable("T00", "f4", ("Time",))[0] = 290.0
        ds.createVariable("MAPFAC_MX", "f4", ("Time", "south_north", "west_east"))[0] = 1.01
        ds.createVariable("GOT_VAR_SSO", "i4", ("Time",))[0] = 1
    got = wrf_file_grid_history_fields(SimpleNamespace(
        restored=SimpleNamespace(raw={"XLAT": np.zeros((2, 2))}, path=path)))
    assert {"C1H", "ZS", "T00", "MAPFAC_MX"} <= set(got)
    assert "GOT_VAR_SSO" not in got
    assert got["T00"].shape == () and got["T00"].dtype == np.float32
    np.testing.assert_array_equal(got["ZS"], np.float32([.05, .25, .7, 1.5]))
