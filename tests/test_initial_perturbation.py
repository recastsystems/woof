"""[perturbation] initial-state theta bubbles: schema, OFF path, application.

One focused node per behavior (the project's iterating discipline):
schema refusals, the absence no-op (byte-identical state), the applied
bubble's location/magnitude, RH preservation, and the two geometry
refusals (center outside the coarse domain; enabled bubble touching
zero cells).  The application tests run ``initialize_real`` end to end
on the CPU backend so the seam is exercised where it lives, not in a
mock.
"""

from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from woof.config import RunConfig
from woof.core.diagnostics import update_diagnostics
from woof.core.grid import make_vertical_coord
from woof.experiment import (
    BUBBLE_AMPLITUDE_WARNING_K,
    DEFERRED_PERTURBATION_SCHEMA,
    BubbleConfig,
    PerturbationConfig,
    build_experiment,
    deferred_initial_perturbation,
    deferred_perturbation_config,
    refuse_unrouted_perturbation,
)
from woof.ingest.horiz import HorizontalSnapshot
from woof.ingest.init_perturbation import (
    RH_PRESERVE_QV_LIMIT_KG_KG,
    build_initial_state_perturbation,
    radiation_temperature_ceiling,
)
from woof.ingest.real import (
    _mixing_ratio_to_relative_humidity,
    _temperature_from_potential_temperature,
    hydrostatic_residual,
    initialize_real,
)
from woof.static.lambert import LambertGrid


# ---------------------------------------------------------------------------
# Schema fixtures
# ---------------------------------------------------------------------------

def _experiment_raw(perturbation=None):
    raw = {
        "experiment": {
            "name": "schema_probe",
            "start_time": datetime(2026, 8, 1, 12),
            "run_seconds": 3600.0,
            "restart_interval_s": 0.0,
        },
        "projection": {
            "map_proj": "lambert", "ref_lat": 38.5, "ref_lon": -99.5,
            "truelat1": 30.0, "truelat2": 50.0, "stand_lon": -99.5,
        },
        "shared": {
            "nz": 6, "ztop": 16000.0, "p_top": 10000.0,
            "eta_levels": [1.0, 0.9, 0.74, 0.56, 0.38, 0.19, 0.0],
            "hybrid_opt": 2, "etac": 0.2, "moist": True,
            "terrain_opt": 1, "base_temp": 290.0,
        },
        "domain": [{
            "grid_id": 1, "parent_id": 0, "i_parent_start": 1,
            "j_parent_start": 1, "parent_grid_ratio": 1,
            "parent_time_step_ratio": 1, "nx": 24, "ny": 20,
            "dx": 3000.0, "time_step": 15, "specified": True,
            "nested": False, "history_interval_s": 3600.0,
        }],
    }
    if perturbation is not None:
        raw["perturbation"] = perturbation
    return raw


def _bubble_entry(**overrides):
    entry = {
        "center_lat": 38.5, "center_lon": -99.5,
        "center_height_m": 1500.0, "radius_km": 10.0,
        "depth_m": 1500.0, "amplitude_k": 2.5,
    }
    entry.update(overrides)
    return entry


# ---------------------------------------------------------------------------
# Schema: refusals and acceptance
# ---------------------------------------------------------------------------

def test_unknown_key_in_perturbation_block_is_refused():
    raw = _experiment_raw({"bubbles": [_bubble_entry()], "bubles": 1})
    with pytest.raises(ValueError, match=r"does not have a key"):
        build_experiment(raw, source="probe.toml")


def test_unknown_key_in_bubble_is_refused_with_suggestion():
    raw = _experiment_raw({"bubbles": [_bubble_entry(radius_kms=5.0)]})
    with pytest.raises(ValueError, match=r"radius_kms.*radius_km"):
        build_experiment(raw, source="probe.toml")


def test_missing_required_bubble_key_is_refused():
    entry = _bubble_entry()
    del entry["amplitude_k"]
    raw = _experiment_raw({"bubbles": [entry]})
    with pytest.raises(ValueError, match=r"missing required key.*amplitude_k"):
        build_experiment(raw, source="probe.toml")


@pytest.mark.parametrize("key", ["radius_km", "depth_m", "amplitude_k"])
def test_nonpositive_bubble_geometry_is_refused(key):
    raw = _experiment_raw({"bubbles": [_bubble_entry(**{key: 0.0})]})
    with pytest.raises(ValueError, match=rf"{key} = 0\.0 must be positive"):
        build_experiment(raw, source="probe.toml")


def test_a_12_5_k_bubble_builds_and_warns_with_the_value_named(
        capsys, monkeypatch):
    from woof import explain

    monkeypatch.setattr(explain, "_PRINTED_ONCE", set())
    raw = _experiment_raw({"bubbles": [_bubble_entry(amplitude_k=12.5)]})
    exp = build_experiment(raw, source="probe.toml")
    (bubble,) = exp.perturbation.bubbles
    assert bubble.amplitude_k == 12.5
    err = capsys.readouterr().err
    assert "warning: perturbation bubble amplitude_k = 12.5 K" in err
    assert f"above {BUBBLE_AMPLITUDE_WARNING_K:g} K" in err
    assert "WRF's idealized warm bubble (3 K, em_quarter_ss)" in err


def test_the_amplitude_warning_prints_once_and_reaches_every_observer(
        capsys, monkeypatch):
    """One command loads its config several times; the line prints once.

    `woof run` printed the same bubble sentence three times and `woof
    go` five.  A machine consumer attached after the first load (run-plan
    collects warnings around its own load) must still receive it.
    """
    from woof import explain

    monkeypatch.setattr(explain, "_PRINTED_ONCE", set())
    raw = _experiment_raw({"bubbles": [_bubble_entry(amplitude_k=12.5)]})
    build_experiment(raw, source="probe.toml")
    records = []
    explain.add_warning_observer(records.append)
    try:
        build_experiment(raw, source="probe.toml")
        build_experiment(raw, source="probe.toml")
    finally:
        explain.remove_warning_observer(records.append)
    err = capsys.readouterr().err
    assert err.count("warning: perturbation bubble amplitude_k = 12.5 K") == 1
    bubble_records = [record for record in records
                      if "amplitude_k = 12.5 K" in record["action"]]
    assert len(bubble_records) == 2


def test_a_bubble_at_the_warning_level_builds_without_a_warning(capsys):
    raw = _experiment_raw({"bubbles": [
        _bubble_entry(amplitude_k=BUBBLE_AMPLITUDE_WARNING_K)]})
    exp = build_experiment(raw, source="probe.toml")
    assert "amplitude_k" not in capsys.readouterr().err
    (row,) = exp.perturbation.receipt()["bubbles"]
    assert "warning" not in row


def test_the_amplitude_warning_is_recorded_in_the_perturbation_receipt(
        tmp_path):
    import json

    from woof.runtime import _write_initial_perturbation_receipt

    exp = build_experiment(_experiment_raw({"bubbles": [
        _bubble_entry(), _bubble_entry(amplitude_k=30.0)]}),
        source="probe.toml")
    small, large = exp.perturbation.receipt()["bubbles"]
    assert "warning" not in small
    assert large["amplitude_k"] == 30.0
    assert large["warning"].startswith(
        "perturbation bubble amplitude_k = 30 K is above 10 K and 10.0 "
        "times WRF's idealized warm bubble (3 K, em_quarter_ss)")
    path = _write_initial_perturbation_receipt(tmp_path, exp, ())
    written = json.loads(path.read_text(encoding="utf-8"))
    assert "warning" not in written["config"]["bubbles"][0]
    assert written["config"]["bubbles"][1]["warning"] == large["warning"]


def test_empty_bubbles_array_is_refused():
    raw = _experiment_raw({"bubbles": []})
    with pytest.raises(ValueError, match=r"array of tables"):
        build_experiment(raw, source="probe.toml")


def test_accepted_block_echoes_every_value_into_the_receipt():
    raw = _experiment_raw({"bubbles": [_bubble_entry(rh_preserve=True)]})
    exp = build_experiment(raw, source="probe.toml")
    assert exp.perturbation is not None
    receipt = exp.perturbation.receipt()
    assert receipt["schema"] == "gpuwm-initial-perturbation-v1"
    assert receipt["bubbles"] == [{
        "center_lat": 38.5, "center_lon": -99.5,
        "center_height_m": 1500.0, "radius_km": 10.0,
        "depth_m": 1500.0, "amplitude_k": 2.5, "rh_preserve": True,
    }]


def test_unrouted_route_refuses_a_configured_block_by_name():
    exp = build_experiment(
        _experiment_raw({"bubbles": [_bubble_entry()]}),
        source="probe.toml")
    with pytest.raises(ValueError, match=r"probe-route.*does not apply"):
        refuse_unrouted_perturbation(exp, "probe-route")
    refuse_unrouted_perturbation(  # absent block: no-op
        build_experiment(_experiment_raw(), source="probe.toml"),
        "probe-route")


def _tree_raw(perturbation=None):
    """``_experiment_raw`` with a ratio-3 child well inside a wider root."""
    raw = _experiment_raw(perturbation)
    raw["domain"][0].update(nx=60, ny=60)
    raw["domain"].append({
        "grid_id": 2, "parent_id": 1, "i_parent_start": 20,
        "j_parent_start": 20, "parent_grid_ratio": 3,
        "parent_time_step_ratio": 3, "e_we": 25, "e_sn": 25,
        "history_interval_s": 3600.0,
    })
    return raw


def test_a_tree_preparation_defers_the_block_in_one_source_neutral_receipt(
        capsys):
    """Every source's tree preparation writes this one receipt.

    The domain-tree runner applies the bubbles to the restored states
    whatever source prepared the tree, so the deferral names no source:
    two routes asking about the same experiment get the same document.
    """
    exp = build_experiment(
        _tree_raw({"bubbles": [_bubble_entry(amplitude_k=0.01)]}),
        source="probe.toml")
    receipt = deferred_initial_perturbation(exp, "probe-route")
    assert receipt == {
        "schema": DEFERRED_PERTURBATION_SCHEMA,
        "status": "DEFERRED_TO_FORECAST_INITIALIZATION",
        "prepared_arrays": "unperturbed source initial and boundary states",
        "application_route": "woof.prepared_domain_tree_forecast",
        "application_point": "restored states at experiment start time",
        "applied_on_restart": False,
        "applied_to_delayed_domains": False,
        "config": exp.perturbation.receipt(),
    }
    assert "deferred to prepared forecast initialization" in (
        capsys.readouterr().err)
    assert deferred_initial_perturbation(
        exp, "another-route", announce=False) == receipt
    assert capsys.readouterr().err == ""


def test_a_single_domain_preparation_defers_the_block_to_its_runner():
    """The prepared single-domain runner applies the bubbles to its restored
    state (fix/single-domain-perturbation), so a single-domain preparation
    defers the block exactly as a tree does, naming that runner."""
    exp = build_experiment(
        _experiment_raw({"bubbles": [_bubble_entry()]}), source="probe.toml")
    receipt = deferred_initial_perturbation(exp, "probe-route", announce=False)
    assert receipt["application_route"] == (
        "woof.prepared_single_domain_forecast")
    assert receipt["status"] == "DEFERRED_TO_FORECAST_INITIALIZATION"
    assert receipt["config"] == exp.perturbation.receipt()


def test_a_deferral_receipt_gives_back_the_block_it_recorded():
    """A stage that builds its experiment without the configuration
    (the native HRRR hierarchy) takes the block back from the sealed
    receipt, through the one validator."""
    exp = build_experiment(
        _tree_raw({"bubbles": [_bubble_entry(amplitude_k=12.0,
                                             rh_preserve=True)]}),
        source="probe.toml")
    receipt = deferred_initial_perturbation(exp, "probe-route", announce=False)
    assert "warning" in receipt["config"]["bubbles"][0]
    assert deferred_perturbation_config(receipt) == exp.perturbation
    with pytest.raises(ValueError, match="deferral must be"):
        deferred_perturbation_config(exp.perturbation.receipt())


def test_an_absent_block_defers_nothing_on_either_shape(capsys):
    for raw in (_experiment_raw(), _tree_raw()):
        exp = build_experiment(raw, source="probe.toml")
        assert deferred_initial_perturbation(exp, "probe-route") is None
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# Absence: zero behavior change, byte-identical prepared state
# ---------------------------------------------------------------------------

def _application_fixture(ny=20, nx=24, nz=6):
    """A CPU-only initialize_real setup with a real Lambert grid."""
    eta = np.array([1.0, 0.9, 0.74, 0.56, 0.38, 0.19, 0.0])
    assert eta.size == nz + 1
    coord = make_vertical_coord(nz, hybrid_opt=2, etac=0.2, eta_levels=eta)
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=3000.0, dy=3000.0,
                    ztop=16000.0, dt=15.0, run_seconds=900.0,
                    hybrid_opt=2, etac=0.2, moist=True, terrain_opt=1,
                    base_temp=290.0)
    grid = LambertGrid(ref_lat=38.5, ref_lon=-99.5, truelat1=30.0,
                       truelat2=50.0, stand_lon=-99.5, dx=3000.0,
                       dy=3000.0, e_we=nx + 1, e_sn=ny + 1)
    levels = np.array([100.0, 200.0, 300.0, 500.0, 700.0, 850.0, 1000.0])
    pressure = levels[:, None, None] * 100.0
    shape = (levels.size, ny, nx)
    temperature = np.broadcast_to(
        215.0 + 72.0 * (pressure / 100000.0) ** 0.20, shape).copy()
    height = np.broadcast_to(
        -7800.0 * np.log(pressure / 100000.0), shape).copy()
    rh = np.broadcast_to(35.0 + 45.0 * (pressure / 100000.0), shape).copy()
    fields = {
        "TT": temperature.astype(np.float32),
        "GHT": height.astype(np.float32),
        "RH": rh.astype(np.float32),
        "UU": np.full((levels.size, ny, nx + 1), 8.0, np.float32),
        "VV": np.full((levels.size, ny + 1, nx), -2.0, np.float32),
        "PSFC": np.full((ny, nx), 96500.0, np.float32),
        "T2": np.full((ny, nx), 288.0, np.float32),
        "D2": np.full((ny, nx), 281.0, np.float32),
        "U10": np.full((ny, nx + 1), 7.0, np.float32),
        "V10": np.full((ny + 1, nx), -1.5, np.float32),
    }
    snapshot = HorizontalSnapshot(
        valid_time=datetime(2026, 8, 1, 12), levels_hpa=levels,
        fields=fields)
    terrain = np.full((ny, nx), 350.0)
    source_orography = np.full((ny, nx), 340.0)
    return cfg, coord, grid, snapshot, terrain, source_orography


def _rrtmgp(cfg):
    """The fixture's configuration with RTE+RRTMGP on both streams."""
    return replace(cfg, ra_lw_physics=4, ra_sw_physics=4,
                   ra_rrtmg_variant="rte-rrtmgp")


def _initialize(perturbation_cfg=None, require_containment=True,
                radiation=None, **kwargs):
    cfg, coord, grid, snapshot, terrain, source_orography = (
        _application_fixture())
    if radiation is not None:
        cfg = radiation(cfg)
    applier = build_initial_state_perturbation(
        perturbation_cfg, grid, grid_id=1,
        require_containment=require_containment, cfg=cfg)
    return initialize_real(
        snapshot, cfg, coord, terrain,
        source_orography=source_orography, p_top=10000.0,
        sfcp_to_sfcp=True, preprocess_backend="cpu",
        state_backend="preprocess", initial_perturbation=applier,
        **kwargs), grid


def test_absent_block_leaves_the_prepared_state_byte_identical():
    """OFF contract: no applier, no branch, byte-identical fields."""
    baseline, _ = _initialize()          # kwarg carries applier=None
    off_result, _ = _initialize(perturbation_cfg=None)
    for name in ("thp", "php", "mup", "qv", "u", "v"):
        np.testing.assert_array_equal(
            np.asarray(getattr(baseline.state, name)),
            np.asarray(getattr(off_result.state, name)))
    assert off_result.initial_perturbation == {}


def test_applied_bubble_peaks_at_the_declared_center():
    bubble = BubbleConfig(center_lat=38.5, center_lon=-99.5,
                          center_height_m=1500.0, radius_km=10.0,
                          depth_m=1500.0, amplitude_k=2.5)
    spec = PerturbationConfig(bubbles=(bubble,))
    baseline, _ = _initialize()
    perturbed, grid = _initialize(perturbation_cfg=spec)
    receipt = perturbed.initial_perturbation
    assert receipt["schema"] == "gpuwm-initial-perturbation-apply-v1"
    assert receipt["grid_id"] == 1
    (row,) = receipt["bubbles"]
    assert row["applied"] is True
    assert row["cells_touched"] > 0
    assert 0.0 < row["max_theta_added_k"] <= 2.5
    # The written delta is where the receipt says it is: the theta
    # difference between the runs peaks at the cell nearest the center
    # and is exactly zero outside the ellipse.
    delta = (np.asarray(perturbed.state.thp, dtype=np.float64)
             - np.asarray(baseline.state.thp, dtype=np.float64))
    k, j, i = np.unravel_index(np.abs(delta).argmax(), delta.shape)
    x, y = grid.latlon_to_ij(38.5, -99.5)
    assert abs((i + 1) - float(x)) <= 0.51
    assert abs((j + 1) - float(y)) <= 0.51
    assert delta.max() == pytest.approx(row["max_theta_added_k"],
                                        abs=2.0e-3)
    assert delta.max() > 2.0        # near-center cell close to the peak
    far = np.ones_like(delta, dtype=bool)
    j0, i0 = int(round(float(y))) - 1, int(round(float(x))) - 1
    lo_j, hi_j = max(0, j0 - 5), min(delta.shape[1], j0 + 6)
    lo_i, hi_i = max(0, i0 - 5), min(delta.shape[2], i0 + 6)
    far[:, lo_j:hi_j, lo_i:hi_i] = False
    assert np.all(delta[far] == 0.0)
    # qv untouched without rh_preserve
    np.testing.assert_array_equal(np.asarray(perturbed.state.qv),
                                  np.asarray(baseline.state.qv))
    # and the perturbed state still passes the hydrostatic gate.
    residual = float(np.max(hydrostatic_residual(perturbed)))
    baseline_residual = float(np.max(hydrostatic_residual(baseline)))
    assert residual <= 4.0 * max(baseline_residual, 1.0e-3)


def test_a_60_k_bubble_leaves_the_initial_state_hydrostatic():
    """The bubble is written before alpha and geopotential are formed.

    So the amplitude is not what keeps the initial state consistent: a
    60 K bubble passes the same hydrostatic gate as the unperturbed
    state, and every field stays finite.
    """
    bubble = BubbleConfig(center_lat=38.5, center_lon=-99.5,
                          center_height_m=1500.0, radius_km=10.0,
                          depth_m=1500.0, amplitude_k=60.0)
    baseline, _ = _initialize()
    perturbed, _ = _initialize(
        perturbation_cfg=PerturbationConfig(bubbles=(bubble,)))
    (row,) = perturbed.initial_perturbation["bubbles"]
    assert row["max_theta_added_k"] > 40.0
    for name in ("thp", "php", "mup", "qv"):
        assert np.isfinite(np.asarray(getattr(perturbed.state, name))).all()
    residual = float(np.max(hydrostatic_residual(perturbed)))
    baseline_residual = float(np.max(hydrostatic_residual(baseline)))
    assert residual <= 4.0 * max(baseline_residual, 1.0e-3)


def test_rh_preserve_holds_relative_humidity_through_the_theta_change():
    bubble = BubbleConfig(center_lat=38.5, center_lon=-99.5,
                          center_height_m=1500.0, radius_km=10.0,
                          depth_m=1500.0, amplitude_k=2.5,
                          rh_preserve=True)
    spec = PerturbationConfig(bubbles=(bubble,))
    baseline, _ = _initialize()
    perturbed, _ = _initialize(perturbation_cfg=spec)
    (row,) = perturbed.initial_perturbation["bubbles"]
    assert row["rh_preserve"] is True
    assert row["max_qv_delta_kg_kg"] > 0.0
    mask = (np.asarray(perturbed.state.thp, dtype=np.float64)
            != np.asarray(baseline.state.thp, dtype=np.float64))
    assert mask.any()

    def relative_humidity(result):
        theta = (np.asarray(result.state.thp, dtype=np.float64)
                 + np.asarray(result.base.thb, dtype=np.float64))
        pressure = np.asarray(result.total_pressure, dtype=np.float64)
        temperature = _temperature_from_potential_temperature(
            theta, pressure)
        return _mixing_ratio_to_relative_humidity(
            temperature, pressure,
            np.asarray(result.state.qv, dtype=np.float64))

    rh_before = relative_humidity(baseline)[mask]
    rh_after = relative_humidity(perturbed)[mask]
    # FP32 state storage bounds the round trip; RH must survive to well
    # under one percent where a 2.5 K bubble would otherwise move it by
    # tens of percent.
    assert float(np.max(np.abs(rh_after - rh_before))) < 0.5
    unchanged_rh = float(np.max(np.abs(
        relative_humidity(perturbed)[~mask]
        - relative_humidity(baseline)[~mask])))
    assert unchanged_rh == 0.0


def test_center_outside_the_coarse_domain_is_refused():
    bubble = BubbleConfig(center_lat=45.0, center_lon=-80.0,
                          center_height_m=1500.0, radius_km=10.0,
                          depth_m=1500.0, amplitude_k=2.5)
    spec = PerturbationConfig(bubbles=(bubble,))
    with pytest.raises(ValueError, match=r"outside domain d01"):
        _initialize(perturbation_cfg=spec)


def test_center_outside_a_child_domain_is_recorded_not_refused():
    bubble = BubbleConfig(center_lat=45.0, center_lon=-80.0,
                          center_height_m=1500.0, radius_km=10.0,
                          depth_m=1500.0, amplitude_k=2.5)
    spec = PerturbationConfig(bubbles=(bubble,))
    result, _ = _initialize(perturbation_cfg=spec,
                            require_containment=False)
    (row,) = result.initial_perturbation["bubbles"]
    assert row["applied"] is False
    assert row["reason"] == "center outside this domain"
    assert row["cells_touched"] == 0


def test_enabled_bubble_touching_zero_cells_is_refused():
    bubble = BubbleConfig(center_lat=38.5, center_lon=-99.5,
                          center_height_m=15000.0, radius_km=0.4,
                          depth_m=10.0, amplitude_k=2.5)
    spec = PerturbationConfig(bubbles=(bubble,))
    with pytest.raises(ValueError, match=r"touches zero cells"):
        _initialize(perturbation_cfg=spec)


def test_apply_to_state_matches_the_initialize_real_seam():
    """The prepared-tree application point writes the same bubble.

    Route A applies inside ``initialize_real`` (FP64 columns, rebalanced
    geopotential); route B applies to the already-initialized state (the
    restored-prepared-cache seam).  Same cells, same receipt stats, and
    the written theta agrees to FP32 storage precision.
    """
    bubble = BubbleConfig(center_lat=38.5, center_lon=-99.5,
                          center_height_m=1500.0, radius_km=10.0,
                          depth_m=1500.0, amplitude_k=2.5)
    spec = PerturbationConfig(bubbles=(bubble,))
    seam, grid = _initialize(perturbation_cfg=spec)
    baseline, _ = _initialize()
    applier = build_initial_state_perturbation(
        spec, grid, grid_id=1, require_containment=True,
        cfg=_application_fixture()[0])
    state_receipt = applier.apply_to_state(baseline.state)
    assert state_receipt["application_point"] == "restored-prepared-state"
    (seam_row,) = seam.initial_perturbation["bubbles"]
    (state_row,) = state_receipt["bubbles"]
    assert state_row["cells_touched"] == seam_row["cells_touched"]
    # The state route reads FP32-stored thp/php, so the peak agrees to
    # storage precision, not to FP64 identity.
    assert state_row["max_theta_added_k"] == pytest.approx(
        seam_row["max_theta_added_k"], abs=1.0e-5)
    thp_seam = np.asarray(seam.state.thp, dtype=np.float64)
    thp_state = np.asarray(baseline.state.thp, dtype=np.float64)
    assert float(np.max(np.abs(thp_seam - thp_state))) < 5.0e-3
    # qv untouched on both routes without rh_preserve
    np.testing.assert_array_equal(np.asarray(baseline.state.qv),
                                  np.asarray(seam.state.qv))


def _bubble(amplitude_k, **overrides):
    return PerturbationConfig(bubbles=(BubbleConfig(
        center_lat=38.5, center_lon=-99.5, center_height_m=1500.0,
        radius_km=10.0, depth_m=1500.0, amplitude_k=amplitude_k,
        **overrides),))


def _restored(**kwargs):
    """An unperturbed state with its EOS diagnosed, as the tree restores it."""
    result, grid = _initialize(**kwargs)
    update_diagnostics(result.state, _application_fixture()[0].hypsometric_opt)
    return result, grid


@pytest.mark.parametrize("use_theta_m", [0, 1])
def test_apply_to_state_rebalances_the_geopotential_at_the_held_pressure(use_theta_m):
    """The prepared-tree route keeps p and re-integrates phi, as WRF does.

    Without the rebalance the diagnostic held phi and raised p inside the
    bubble instead: +25 % at the core of a 60 K bubble on a GFS 12/3 km
    tree, which RRTMGP then refused at step 1 as a 363 K layer.  Here p
    after the bubble equals p before it to FP32 precision, and the
    geopotential equals what initialize_real forms from the same bubble.
    """
    spec = _bubble(60.0)
    seam, grid = _initialize(perturbation_cfg=spec)
    restored, _ = _restored()
    state = restored.state
    if use_theta_m:
        from woof.core import constants as c
        base = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
        state.thp[...] = ((base + state.thp)
                          * (np.float32(1.0) + np.float32(c.RVOVRD) * state.qv)
                          - base)
        state.use_theta_m = 1
    p_before = np.array(state.p, dtype=np.float64)
    php_before = np.array(state.php, dtype=np.float64)
    thp_before = np.array(state.thp, dtype=np.float64)
    applier = build_initial_state_perturbation(
        spec, grid, grid_id=1, require_containment=True,
        cfg=_application_fixture()[0])
    receipt = applier.apply_to_state(state)
    assert receipt["geopotential"] == "rebalanced at the held pressure"
    update_diagnostics(state, _application_fixture()[0].hypsometric_opt)
    p_after = np.array(state.p, dtype=np.float64)
    assert float(np.max(np.abs(p_after / p_before - 1.0))) < 2.0e-6
    rise = np.array(state.php, dtype=np.float64) - php_before
    assert float(rise.max()) > 1000.0          # the warm column is taller
    assert np.all(rise[0] == 0.0)              # the ground stays put
    bubble_columns = (np.array(state.thp, dtype=np.float64)
                      != thp_before).any(axis=0)
    assert bubble_columns.any()
    assert np.all(rise[:, ~bubble_columns] == 0.0)
    php_seam = np.asarray(seam.state.php, dtype=np.float64)
    assert float(np.max(np.abs(np.asarray(state.php, dtype=np.float64)
                               - php_seam))) < 0.05


def test_the_rrtmgp_ceiling_is_the_top_of_its_gas_tables():
    cfg = _application_fixture()[0]
    ceiling = radiation_temperature_ceiling(_rrtmgp(cfg))
    assert ceiling.kelvin == 355.0
    assert "RTE+RRTMGP" in ceiling.tables
    assert radiation_temperature_ceiling(cfg) is None      # radiation off
    assert radiation_temperature_ceiling(
        replace(_rrtmgp(cfg), ra_rrtmg_variant="rrtmg_legacy")) is None


def test_a_layer_past_the_radiation_tables_is_refused_before_integration():
    spec = _bubble(120.0)
    with pytest.raises(ValueError, match=(
            r"perturbation\.bubbles #1 heats a layer on domain d01 to "
            r"3\d\d\.\d K at \d+ hPa, above 355 K, the top of the "
            r"RTE\+RRTMGP gas-optics tables .* stop at step 1")):
        _initialize(perturbation_cfg=spec, radiation=_rrtmgp)
    # The same bubble under radiation with no such table runs.
    result, _ = _initialize(perturbation_cfg=spec)
    (row,) = result.initial_perturbation["bubbles"]
    assert row["max_theta_added_k"] > 100.0


def test_the_prepared_route_refuses_the_hot_layer_and_writes_nothing():
    restored, grid = _restored()
    state = restored.state
    before = {name: np.array(getattr(state, name))
              for name in ("thp", "php", "qv")}
    applier = build_initial_state_perturbation(
        _bubble(120.0), grid, grid_id=1, require_containment=True,
        cfg=_rrtmgp(_application_fixture()[0]))
    with pytest.raises(ValueError, match=r"above 355 K"):
        applier.apply_to_state(state)
    for name, value in before.items():
        np.testing.assert_array_equal(np.asarray(getattr(state, name)), value)


def test_rh_preserve_past_the_measured_vapour_limit_is_refused():
    with pytest.raises(ValueError, match=(
            r"perturbation\.bubbles #1 \(amplitude_k = 60 K, rh_preserve = "
            r"true\) builds water vapour up to 0\.\d+ kg/kg on domain d01, "
            rf"above the {RH_PRESERVE_QV_LIMIT_KG_KG:g} kg/kg")):
        _initialize(perturbation_cfg=_bubble(60.0, rh_preserve=True))
    restored, grid = _restored()
    applier = build_initial_state_perturbation(
        _bubble(60.0, rh_preserve=True), grid, grid_id=1,
        require_containment=True, cfg=_application_fixture()[0])
    qv = np.array(restored.state.qv)
    with pytest.raises(ValueError, match=r"builds water vapour"):
        applier.apply_to_state(restored.state)
    np.testing.assert_array_equal(np.asarray(restored.state.qv), qv)


def test_rh_preserve_under_the_vapour_limit_builds():
    result, _ = _initialize(perturbation_cfg=_bubble(20.0, rh_preserve=True))
    (row,) = result.initial_perturbation["bubbles"]
    assert row["max_qv_delta_kg_kg"] > 0.0
    assert float(np.max(np.asarray(result.state.qv))) <= (
        RH_PRESERVE_QV_LIMIT_KG_KG)


def test_restart_identity_omits_an_absent_block_and_binds_a_present_one():
    from woof.core.model import restart_identity_payload

    absent = build_experiment(_experiment_raw(), source="probe.toml")
    assert "perturbation" not in restart_identity_payload(absent)
    present = build_experiment(
        _experiment_raw({"bubbles": [_bubble_entry()]}),
        source="probe.toml")
    payload = restart_identity_payload(present)
    assert payload["perturbation"]["bubbles"][0]["amplitude_k"] == 2.5


def test_prepared_row_windows_preserve_whole_domain_bubble_bytes():
    from copy import copy
    from dataclasses import replace
    from types import SimpleNamespace
    bubble = BubbleConfig(center_lat=38.5, center_lon=-99.5,
                          center_height_m=1500.0, radius_km=10.0,
                          depth_m=1500.0, amplitude_k=2.5, rh_preserve=True)
    spec = PerturbationConfig(bubbles=(bubble,))
    full, grid = _initialize()
    windowed, _ = _initialize()
    full.state.p[...] = 80000.0
    windowed.state.p[...] = 80000.0
    applier = build_initial_state_perturbation(
        spec, grid, grid_id=1, require_containment=True,
        cfg=_application_fixture()[0])
    expected = applier.apply_to_state(full.state)
    touched = 0
    for j in range(windowed.state.thp.shape[-2]):
        rows = slice(j, j+1)
        slab = SimpleNamespace(**{name: (getattr(windowed.state, name)[:, rows, :]
            if getattr(windowed.state, name).ndim == 3 else getattr(windowed.state, name))
            for name in ('thb', 'thp', 'qv', 'p', 'phb', 'php',
                         'dphb_resid')})
        local = copy(applier)
        local._placed = tuple(replace(p, horizontal_km=p.horizontal_km[rows])
                              for p in applier._placed)
        receipt = local.apply_to_state(slab, allow_empty=True)
        touched += receipt['bubbles'][0]['cells_touched']
    assert touched == expected['bubbles'][0]['cells_touched']
    np.testing.assert_array_equal(windowed.state.thp, full.state.thp)
    np.testing.assert_array_equal(windowed.state.qv, full.state.qv)
    # The rebalance is per column, so row slabs rebuild the same php.
    assert (np.asarray(full.state.php) != np.asarray(
        _initialize()[0].state.php)).any()
    np.testing.assert_array_equal(windowed.state.php, full.state.php)
