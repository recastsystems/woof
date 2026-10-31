"""Admission, independent carrier and MYNN feedback gates for scalar QKE."""
from dataclasses import replace

import numpy as np
import pytest
import os
from pathlib import Path
import subprocess
import sys

from woof.config import RunConfig, validate_run_config
import _mynn_families as F


def config(on=True):
    return RunConfig(nx=12, ny=12, nz=30, dx=3000., dy=3000., ztop=12000.,
                     dt=1., run_seconds=3600., moist=True, mp_physics=6,
                     bl_pbl_physics=5, sf_sfclay_physics=5,
                     sf_surface_physics=2, bl_mynn_tkeadvect=on)


def test_selected_transport_is_admitted():
    validate_run_config(config())


def test_nested_scalar_inventory_and_boundary_cache_admit_qke():
    from woof.core.nest_fields import nest_field_kinds
    from woof.boundary_fields import admissible_boundary_inventory
    assert "qke_adv" in nest_field_kinds(config())
    assert "qke_adv" not in nest_field_kinds(config(False))
    assert admissible_boundary_inventory(config(), ("qv", "qke_adv"))
    assert not admissible_boundary_inventory(config(False), ("qv", "qke_adv"))


def test_every_nest_field_kind_has_a_scratch_lifetime_row():
    # A nested run with TKE advection on died at preflight with
    # KeyError: scratch slot 'nest_qke_adv_bxs' has no lifetime audit row.
    from woof.core.nest_fields import nest_field_kinds
    from woof.core.preflight import scratch_slot_uses_arena
    for cfg in (config(), replace(config(), mp_physics=8), replace(config(), mp_physics=28)):
        for kind in nest_field_kinds(cfg):
            for prefix in ("b", "bt"):
                for side in ("xs", "xe", "ys", "ye"):
                    scratch_slot_uses_arena(f"nest_{kind}_{prefix}{side}")


def test_dry_scalar_prices_the_existing_pd_and_mass_flux_slots():
    from woof.core.preflight import scratch_slot_registry, scratch_slot_lifetime
    cfg = replace(config(), moist=False, mp_physics=0)
    slots = scratch_slot_registry(cfg)
    for name in ("rk_ru_m", "rk_rv_m", "rk_ww_m", "moist_rq_t", "pd_fxl", "pd_fxc", "pd_fzl"):
        assert name in slots
        assert scratch_slot_lifetime(name) is not None


@pytest.mark.parametrize("prefix", ("qke_adv", "QKE_ADV"))
def test_native_lateral_tables_transport_supplied_qke(tmp_path, prefix):
    import netCDF4
    from test_n5s_toolchain import _real_shaped_d01_wrfinput, _small_wrfbdy
    from woof.ingest.wrfinput import read_wrfinput, read_wrfbdy
    from dataclasses import replace as change
    input_path, boundary_path = tmp_path / "input", tmp_path / "boundary"
    _real_shaped_d01_wrfinput(input_path)
    _small_wrfbdy(boundary_path)
    identity = dict(GRID_ID=1, DX=3000., DY=3000., MAP_PROJ=1,
                    TRUELAT1=38.5, TRUELAT2=38.5, STAND_LON=-97.5,
                    CEN_LAT=35., CEN_LON=-97.5, HYBRID_OPT=0, ETAC=.2,
                    USE_THETA_M=0, START_DATE="1974-04-03_12:00:00")
    with netCDF4.Dataset(input_path, "a") as dataset:
        dataset.setncatts(dict(identity, MP_PHYSICS=6, SF_SURFACE_PHYSICS=2, BL_PBL_PHYSICS=5))
    restored = read_wrfinput(input_path, require_complete=False,
                             expected_dimensions=dict(bottom_top=49, bottom_top_stag=50,
                                 south_north=200, south_north_stag=201,
                                 west_east=250, west_east_stag=251, soil_layers_stag=4))
    # Use the fixture's independently coupled vapour values for a scalar
    # with identical primitive words. The reader still checks pairing.
    restored = change(restored, raw=dict(restored.raw, qke_adv=restored.raw["QVAPOR"]))
    with netCDF4.Dataset(boundary_path, "a") as dataset:
        dataset.setncatts(identity)
        for suffix in ("BXS", "BXE", "BYS", "BYE", "BTXS", "BTXE", "BTYS", "BTYE"):
            original = dataset["QVAPOR_" + suffix]
            dataset.createVariable(prefix + "_" + suffix, "f4", original.dimensions)[:] = original[:]
    cfg = change(config(), nx=250, ny=200, nz=49, specified=True)
    boundaries = read_wrfbdy(boundary_path, run_seconds=3600, restored=restored,
                             forcing_interval_seconds=21600, cfg=cfg)
    assert "qke_adv" in boundaries.intervals[0].fields


def test_prepared_and_checkpoint_identities_distinguish_transport():
    from dataclasses import asdict
    from woof.ingest.prepared_cache import compare_prepared_identity
    from woof.io.restart import configuration_echo
    _, different = compare_prepared_identity(
        {"domain_config": {"run": asdict(config(False))}},
        {"domain_config": {"run": asdict(config(True))}})
    assert "run.bl_mynn_tkeadvect" in different
    assert configuration_echo(config(True)) != configuration_echo(config(False))


def test_tree_restart_keeps_the_historical_scalar_default(monkeypatch):
    from copy import deepcopy
    import woof.experiment as experiment_module
    from woof.core.model import restart_identity_payload
    exp = experiment_module.load_experiment(
        Path(__file__).resolve().parents[1] / "configs/hrrr_native_quick_demo.toml")
    document = experiment_module.experiment_config_document(exp)
    historical = deepcopy(document)
    for domain in historical["domains"]:
        # The public document already omits the key at its historical
        # default (golden/wrf-fixes 39d0bcf5d), so it may be absent here.
        domain["run"].pop("scalar_adv_opt", None)
    public_document = experiment_module.experiment_config_document
    monkeypatch.setattr(experiment_module, "experiment_config_document", lambda _: historical)
    expected = restart_identity_payload(exp)
    monkeypatch.setattr(experiment_module, "experiment_config_document", public_document)
    assert restart_identity_payload(exp) == expected
    active = deepcopy(document)
    for domain in active["domains"]:
        domain["run"]["bl_mynn_tkeadvect"] = True
    monkeypatch.setattr(experiment_module, "experiment_config_document", lambda _: active)
    assert restart_identity_payload(exp) != expected


@pytest.mark.parametrize("on", (False, True))
def test_prepared_registry_preserves_only_the_inactive_transport(on):
    from copy import deepcopy
    from woof.physics_compat import single_domain_physics_selection, physics_selection_differences
    from woof.physics_registry import canonical_sha256, strip_registry_documentation
    cfg = config(on)
    expected = single_domain_physics_selection(cfg)
    recorded = deepcopy(expected)
    # Exact prior declaration, before the transport gained an active scope.
    old_spec = dict(type="boolean", default=False, per_domain=False,
                    consuming_read="woof/core/physics.py")
    recorded["registry_physics"]["parts"]["parameters.bl_mynn_tkeadvect"] = (
        canonical_sha256(strip_registry_documentation(old_spec)))
    differences = physics_selection_differences(recorded, expected, settings=cfg)
    assert bool(differences) is on, differences


def test_namelist_import_accepts_true(tmp_path):
    from test_namelist_import import _pair, INPUT_TEXT
    from woof.namelist_import import import_namelists
    from woof.experiment import load_experiment
    source = INPUT_TEXT.replace("&physics", "&physics\n bl_mynn_tkeadvect = .true., .true.,")
    source = source.replace("bl_pbl_physics = 11, 11,", "bl_pbl_physics = 5, 5,")
    source = source.replace("sf_sfclay_physics = 91, 91,", "sf_sfclay_physics = 5, 5,")
    text, _ = import_namelists(*_pair(tmp_path, inp=source), name="scalar-tke")
    path = tmp_path / "resolved.toml"
    path.write_text(text)
    experiment = load_experiment(path)
    assert all(domain.run.bl_mynn_tkeadvect is True for domain in experiment.domains)


@pytest.mark.parametrize("initflag", (0, 1))
def test_cpu_driver_copies_the_scalar_at_the_wrf_call_boundary(initflag):
    from woof.core.mynn_pbl import mynn_bl_driver
    blocks = F.load(1)[2]
    values = F.step_inputs(blocks)
    values["qke_adv"] = values["qke"].copy() * np.float32(1.75)
    expected_values = dict(values, qke=values["qke_adv"])
    expected = mynn_bl_driver(expected_values, initflag=initflag, delt=20., flag_qs=True)
    actual = mynn_bl_driver(values, initflag=initflag, delt=20., flag_qs=True,
                            bl_mynn_tkeadvect=True)
    for name in expected:
        np.testing.assert_array_equal(actual[name].view(np.uint32),
                                      expected[name].view(np.uint32))
    np.testing.assert_array_equal(actual["qke_adv"].view(np.uint32),
                                  actual["qke"].view(np.uint32))


@pytest.mark.gpu
@pytest.mark.parametrize("initflag", (0, 1))
def test_gpu_driver_copies_the_scalar_at_the_wrf_call_boundary(initflag):
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_bl_driver_cuda
    values = {k: cp.asarray(v) for k, v in F.step_inputs(F.load(1)[2]).items()}
    values["qke_adv"] = values["qke"].copy() * np.float32(1.75)
    expected = mynn_bl_driver_cuda(dict(values, qke=values["qke_adv"]),
                                   initflag=initflag, delt=20., flag_qs=True)
    actual = mynn_bl_driver_cuda(values, initflag=initflag, delt=20., flag_qs=True,
                                 bl_mynn_tkeadvect=True)
    for name in expected:
        np.testing.assert_array_equal(cp.asnumpy(actual[name]).view(np.uint32),
                                      cp.asnumpy(expected[name]).view(np.uint32))
    np.testing.assert_array_equal(cp.asnumpy(actual["qke_adv"]).view(np.uint32),
                                  cp.asnumpy(actual["qke"]).view(np.uint32))


@pytest.mark.gpu
def test_qke_uses_the_scalar_inventory_without_water_loading():
    from woof.core.state import DomainState
    from woof.core.moist import moist_species, WRF_MOIST_ARRAY_SPECIES
    from woof.core.device_inventory import state_array_shapes
    on, off = config(), config(False)
    state = DomainState(on)
    assert "qke_adv" in moist_species(state)
    assert "qke_adv" not in WRF_MOIST_ARRAY_SPECIES
    assert state.qke_adv.shape == (30, 12, 12)
    assert state.qke_adv0.shape == state.qke_adv.shape
    assert "qke_adv" not in moist_species(DomainState(off))
    assert "qke_adv" in state_array_shapes(on)
    assert "qke_adv" not in state_array_shapes(off)


@pytest.mark.gpu
def test_qke_only_scalar_takes_wrfs_first_scalar_zero_out_rule():
    import cupy as cp
    from woof.core.state import DomainState
    from woof.core.microphysics import microphysics_zero_out
    cfg = replace(config(), mp_zero_out=1, mp_zero_out_all=1, mp_zero_out_thresh=1e-12)
    state = DomainState(cfg)
    state.qke_adv.fill(np.float32(1e-13))
    microphysics_zero_out(state, cfg)
    np.testing.assert_array_equal(cp.asnumpy(state.qke_adv), np.full(state.qke_adv.shape, 1e-13, np.float32))


@pytest.mark.gpu
@pytest.mark.parametrize("supplied", (True, False))
def test_native_initializer_preserves_scalar_or_seeds_it_from_qke(tmp_path, monkeypatch, supplied):
    import cupy as cp
    import test_mynn_pbl_runtime as fixture
    from test_wrfinput_physics_fields import _input, _read, _from_driver
    from woof.ingest.wrfinput import initialize_wrfinput_physics
    constructor = fixture.RunConfig
    monkeypatch.setattr(fixture, "RunConfig", lambda **kw: constructor(**kw, bl_mynn_tkeadvect=True))
    state, cfg, template = fixture._build()
    raw = _from_driver(template, cp)
    raw["qke"] = np.full((cfg.nz, cfg.ny, cfg.nx), .75, np.float32)
    if supplied:
        raw["qke_adv"] = np.full_like(raw["qke"], 1.25)
    else:
        raw.pop("qke_adv", None)
    restored = _read(_input(tmp_path / "input", cfg, raw), cfg)
    driver = initialize_wrfinput_physics(state, restored, cfg)
    want = raw["qke_adv"] if supplied else raw["qke"]
    np.testing.assert_array_equal(cp.asnumpy(state.qke_adv).view(np.uint32), want.view(np.uint32))
    assert driver.fields["qke_adv"] is state.qke_adv


@pytest.mark.gpu
@pytest.mark.parametrize("fork,length", ((False, 1), (False, 2), (True, 1), (True, 2)))
def test_active_driver_is_word_exact_to_the_independent_oracle(fork, length):
    root = os.environ.get("MYNN_TKEADVECT_ORACLES")
    if not root:
        pytest.skip("build the stock and fork oracles and set MYNN_TKEADVECT_ORACLES")
    from tools.mynn_tkeadvect_check import oracle
    csv = Path(root) / ("oracle-fork" if fork else "oracle-stock") / f"active-{length}.csv"
    result = oracle(csv, fork, length)
    assert result["words"] > 30000
    assert result["different_words"] == result["max_ulp"] == 0


@pytest.mark.gpu
def test_one_rk_stage_is_strict_wrf_and_detects_a_one_ulp_change(tmp_path):
    root = os.environ.get("MYNN_TKEADVECT_ORACLES")
    if not root:
        pytest.skip("build the scalar oracle and set MYNN_TKEADVECT_ORACLES")
    from tools.mynn_tkeadvect_check import transport
    result = transport(Path(root) / "oracle-advect/run", tmp_path, full_rk=True)
    assert len(result["rk_stages"]) == 3
    assert all(stage["different_words"] == stage["max_ulp"] == 0 for stage in result["rk_stages"])
    assert result["strict"]["different_words"] == result["strict"]["max_ulp"] == 0
    assert result["planted_one_ulp"]["different_words"] == 1
    assert result["planted_one_ulp"]["max_ulp"] == 1


@pytest.mark.gpu
@pytest.mark.parametrize("version,length", (("wrf_461", 1), ("gsd_41", 2)))
def test_active_scalar_restart_matches_straight_run(tmp_path, version, length):
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "tools/mynn_review_restart.py"),
               "--folder", str(tmp_path), "--version", version, "--mixlength", str(length),
               "--tkeadvect", "--out", str(tmp_path / "receipt.json")]
    for phase in ("reference", "split", "resume", "check"):
        run = subprocess.run(command + ["--phase", phase], capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
