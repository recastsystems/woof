"""TEMF selectors, coupling and carried restart state."""
import json
from pathlib import Path
import numpy as np
import pytest
from woof.config import (
    RunConfig, selector_coupling_refusal, validate_selector_couplings)
from woof.core.physics_inventory import (
    TEMF_PBL_STATE_3D, TEMF_PBL_STATE_2D, TEMF_SFCLAY_STATE_2D,
    PHYSICS_SLOT_DISPATCH)

def test_pairing_is_bidirectional_and_names_the_missing_writer():
    for a, b in ((10, 1), (1, 10), (10, 0), (0, 10)):
        text = selector_coupling_refusal({
            "bl_pbl_physics": a, "sf_sfclay_physics": b})
        assert "requires" in text and "required writer" in text
        assert "module_check_a_mundo.F:2294-2299" in text
    assert selector_coupling_refusal({
        "bl_pbl_physics": 10, "sf_sfclay_physics": 10}) is None
    assert selector_coupling_refusal({
        "bl_pbl_physics": 1, "sf_sfclay_physics": 91}) is None

def test_registry_and_dispatch_cover_both_halves():
    from woof.physics_registry import component_options
    from woof.checkpoint_identity import (
        PBL_ALGORITHM_IDENTITIES, SURFACE_LAYER_ALGORITHM_IDENTITIES)
    for component, key, identity in (
        ("pbl", "bl_pbl_physics", PBL_ALGORITHM_IDENTITIES),
        ("surface_layer", "sf_sfclay_physics", SURFACE_LAYER_ALGORITHM_IDENTITIES)):
        row = component_options(component)["temf"]
        assert row["implemented"] and row["selectors"][key] == 10
        assert row["consumers"]["restart_algorithm_identity"] == identity[10]
        assert row["reachability"]["state"] == "component-override"
        assert PHYSICS_SLOT_DISPATCH[key][10].startswith("_run_temf_")
    carried = component_options("pbl")["temf"]["extensions"]["carried_restart_fields"]
    assert carried == list(TEMF_PBL_STATE_3D + TEMF_PBL_STATE_2D)
    assert len(carried) == 17

def test_a_wrf_namelist_import_preserves_both_selectors(tmp_path):
    from test_namelist_import import WPS_TEXT, INPUT_TEXT, _pair
    from woof.namelist_import import import_namelists
    text = INPUT_TEXT.replace("sf_sfclay_physics = 91, 91",
                              "sf_sfclay_physics = 10, 10").replace(
        "bl_pbl_physics = 11, 11", "bl_pbl_physics = 10, 10")
    out, report = import_namelists(*_pair(tmp_path, inp=text))
    assert "sf_sfclay_physics = 10" in out
    assert "bl_pbl_physics = 10" in out

@pytest.mark.gpu
def test_real_driver_restart_roundtrip_and_continuation(tmp_path):
    import cupy as cp
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_at_rest
    from woof.core.physics import initialize_physics
    from woof.core.dycore import run_steps
    from woof.io.restart import write_restart, restore_restart
    from woof.core.temf_constants import TEMF_CONSTANTS_SHA256
    from woof.io.restart import read_restart_header
    cfg = RunConfig(nx=6, ny=5, nz=24, dx=3000., dy=3000., dt=6.,
                    ztop=15000., run_seconds=60., moist=True,
                    sf_sfclay_physics=10, bl_pbl_physics=10,
                    sf_surface_physics=0, mp_physics=0, cu_physics=0,
                    ra_physics=0, bldt=0., km_opt=4)
    def fresh():
        coord = make_vertical_coord(cfg.nz)
        base = make_base_state(coord,
            lambda z: 300. + .003 * np.asarray(z, np.float64),
            p_surf=cfg.p_surf, ztop=cfg.ztop)
        state = init_at_rest(cfg, coord, base)
        state.u[...] = cp.float32(6.)
        state.qv[...] = cp.float32(.005)
        driver = initialize_physics(state, cfg, landmask=1., tsk=305.,
                                    glw=300., swdown=0.)
        return state, driver
    state, driver = fresh()
    carried = TEMF_PBL_STATE_3D + TEMF_PBL_STATE_2D + TEMF_SFCLAY_STATE_2D
    assert np.all(cp.asnumpy(driver.fields["te_temf"]) == np.float32(1e-3))
    run_steps(state, cfg, 3)
    assert driver.call_counts["sfclay"] == 3
    assert driver.call_counts["ysu"] == 3
    archive = write_restart(tmp_path / "temf.npz", state, cfg)
    assert read_restart_header(archive)["physics_setup"]["algorithms"][
        "temf_constants"] == TEMF_CONSTANTS_SHA256
    resumed, other = fresh()
    restore_restart(archive, resumed, cfg)
    assert set(driver.fields) == set(other.fields)
    for name in sorted(driver.fields):
        assert cp.asnumpy(driver.fields[name]).tobytes() == cp.asnumpy(
            other.fields[name]).tobytes(), name
    run_steps(state, cfg, 2)
    run_steps(resumed, cfg, 2)
    for name in sorted(driver.fields):
        assert cp.asnumpy(driver.fields[name]).tobytes() == cp.asnumpy(
            other.fields[name]).tobytes(), name
    for name in ("u", "v", "thp", "qv", "qc", "mup"):
        assert cp.asnumpy(getattr(state, name)).tobytes() == cp.asnumpy(
            getattr(resumed, name)).tobytes(), name


def test_history_schema_covers_all_carried_fields():
    from woof.io.wrf_output_schema import TEMF_OUTPUT_FIELDS, HISTORY_FIELDS_BY_NETCDF_NAME
    carried = TEMF_PBL_STATE_3D + TEMF_PBL_STATE_2D + TEMF_SFCLAY_STATE_2D
    assert set(TEMF_OUTPUT_FIELDS) == set(carried)
    for name in carried:
        row = TEMF_OUTPUT_FIELDS[name]
        assert row.netcdf_name == name.upper()
        assert HISTORY_FIELDS_BY_NETCDF_NAME[row.netcdf_name] == row
        assert row.dtype == "f4" and row.stagger == "" and row.wrf_history


def test_preflight_prices_every_temf_array():
    from woof.core.preflight import (
        physics_array_shapes, temf_output_transient_shapes)
    cfg = RunConfig(nx=6, ny=5, nz=24, dx=3000., dy=3000., ztop=15000.,
                    dt=6., run_seconds=60., sf_sfclay_physics=10,
                    bl_pbl_physics=10, moist=True)
    shapes = physics_array_shapes(cfg)
    for name in TEMF_PBL_STATE_3D:
        assert shapes["fields/" + name] == (24, 5, 6)
    for name in TEMF_PBL_STATE_2D + TEMF_SFCLAY_STATE_2D:
        assert shapes["fields/" + name] == (5, 6)
    assert temf_output_transient_shapes(cfg)["temf/tendencies"] == (5, 24, 5, 6)
