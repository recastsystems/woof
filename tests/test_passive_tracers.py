"""Native non-Chem tracer state and rate-bank lifecycle contracts."""
from dataclasses import asdict

import numpy as np
import pytest

from woof.core.passive_tracer_contract import (
    TR17_NAMES as TRACER_NAMES, active_tracers, drop_inactive_tracer_controls,
    initialize_real_tracers, validate_passive_tracers,
)
from woof.io import restart
from test_restart import _cfg, _shim_state, _shim_driver_state, _NumpyCupyShim, _fill_setup


def test_shared_pbl_scratch_lifecycle():
    # Device addresses of this state's own carriers: rebuilt per state, never
    # carried (a streamed copy would hand one tile buffer another's addresses).
    assert restart.classify_scratch_slot("tracer_pblmix_pointers") == "rebuild"
    assert restart.classify_state_attr("_tracer_pblmix_pointer_identity") == "infra"
    for slot in ("tracer_pblmix_rates", "tracer_pblmix_work"):
        assert restart.classify_scratch_slot(slot) == "rebuild"


def test_off_preserves_allocation_and_config_identity(monkeypatch):
    cfg = _cfg()
    state = _shim_state(cfg, monkeypatch)
    assert not active_tracers(cfg)
    assert all(not hasattr(state, name) for name in TRACER_NAMES)
    identity = asdict(cfg)
    drop_inactive_tracer_controls(identity)
    assert not any(key.startswith("tracer_") for key in identity)
    assert not any("tr17_" in key for key in restart.state_manifest(state))


@pytest.mark.parametrize("nx,ny", [(12, 12), (13, 15), (6, 4)])
def test_real_seed_matches_native_integer_indices(monkeypatch, nx, ny):
    cfg = _cfg(nx=nx, ny=ny, tracer_opt=2)
    state = _shim_state(cfg, monkeypatch)
    initialize_real_tracers(state, cfg)
    expected = np.zeros((cfg.nz, ny, nx), np.float32)
    # Native ids=jds=1 and ide=nx+1,jde=ny+1, inclusive native loops.
    for j in range((ny + 2) // 2 - 4, (ny + 2) // 2 + 5):
        for i in range((nx + 2) // 2 - 4, (nx + 2) // 2 + 5):
            if 1 <= i <= nx and 1 <= j <= ny:
                expected[0, j - 1, i - 1] = 1
    for name in TRACER_NAMES[:4]:
        np.testing.assert_array_equal(getattr(state, name), expected)
    for name in TRACER_NAMES[4:]:
        assert not np.any(getattr(state, name))


def test_state_history_nests_and_checkpoint_share_eight_species(monkeypatch, tmp_path):
    from woof.core.device_inventory import state_array_shapes
    from woof.core.nest_fields import nest_field_kinds
    from woof.io.history_layout import live_state_history_fields, produced_history_shapes
    cfg = _cfg(tracer_opt=2)
    state = _shim_state(cfg, monkeypatch)
    for index, name in enumerate(TRACER_NAMES):
        getattr(state, name).fill(np.float32(index + 0.125))
    shapes = state_array_shapes(cfg)
    history = live_state_history_fields(state)
    produced = produced_history_shapes(cfg)
    assert set(TRACER_NAMES) <= set(nest_field_kinds(cfg))
    for name in TRACER_NAMES:
        assert shapes[name] == shapes[name + "0"] == (cfg.nz, cfg.ny, cfg.nx)
        assert history[name] is getattr(state, name)
        assert produced[name] == shapes[name]
    path = restart.write_restart(tmp_path / "tracers.npz", state, cfg)
    with np.load(path) as saved:
        for name in TRACER_NAMES:
            np.testing.assert_array_equal(saved["state/" + name], getattr(state, name))
            assert "state/" + name + "0" not in saved


def test_held_pbl_rates_are_checkpointed_and_missing_rate_fails(monkeypatch):
    monkeypatch.setattr(_NumpyCupyShim, "zeros_like", staticmethod(np.zeros_like), raising=False)
    cfg = _cfg(tracer_opt=2, moist=True, bl_pbl_physics=1, bldt=2.0)
    state, driver = _shim_driver_state(cfg, monkeypatch)
    for index, name in enumerate(TRACER_NAMES):
        driver.pbl_tendencies.extra_scalars[name].fill(index + 0.25)
    manifest = restart._tracer_pbl_manifest(driver)
    assert len(manifest) == 8
    for name in TRACER_NAMES:
        assert manifest["driver/pbl_tendencies/extra_scalars/" + name] is driver.pbl_tendencies.extra_scalars[name]
    driver.pbl_tendencies.extra_scalars.pop(TRACER_NAMES[0])
    with pytest.raises(restart.RestartManifestError, match="tracer PBL rate bank"):
        restart._tracer_pbl_manifest(driver)


def test_selectors_refuse_other_native_packages_and_operators():
    for changes, message in ((dict(tracer_opt=1), "non-Chem"),
                             (dict(tracer_opt=2, tracer_adv_opt=2), "monotonic"),
                             (dict(tracer_opt=2, open_x=True), "open-boundary PD")):
        with pytest.raises(ValueError, match=message):
            validate_passive_tracers(_cfg(**changes))
    validate_passive_tracers(_cfg(tracer_opt=2))  # no PBL: native no-op


def test_stock_namelist_import_keeps_domain_tracer_controls(tmp_path):
    from test_namelist_import import WPS_TEXT, INPUT_TEXT
    from woof.namelist_import import import_namelists
    from woof.experiment import load_experiment
    wps = tmp_path / "namelist.wps"
    inp = tmp_path / "namelist.input"
    wps.write_text(WPS_TEXT)
    inp.write_text(INPUT_TEXT.replace("&dynamics", "&dynamics\n tracer_opt=2,2,\n tracer_adv_opt=0,1,\n tracer_mix2_off=.true.,.false.,"))
    output = tmp_path / "tracers.toml"
    text, _ = import_namelists(wps, inp, name="tracers")
    output.write_text(text)
    experiment = load_experiment(output)
    assert experiment.root.run.tracer_opt == 2
    assert experiment.root.run.tracer_adv_opt == 0
    assert experiment.root.run.tracer_mix2_off is True


def test_pbl_checkpoint_roundtrip_preserves_held_tracer_rates(monkeypatch, tmp_path):
    monkeypatch.setattr(_NumpyCupyShim, "zeros_like", staticmethod(np.zeros_like), raising=False)
    cfg = _cfg(tracer_opt=2, moist=True, bl_pbl_physics=1, bldt=2.0)
    state, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    for index, name in enumerate(TRACER_NAMES):
        getattr(state, name).fill(index + 0.125)
        driver.pbl_tendencies.extra_scalars[name].fill(index + 0.25)
    path = restart.write_restart(tmp_path / "held.npz", state, cfg)
    target, resumed = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(target)
    restart.restore_restart(path, target, cfg)
    for name in TRACER_NAMES:
        np.testing.assert_array_equal(getattr(target, name), getattr(state, name))
        np.testing.assert_array_equal(resumed.pbl_tendencies.extra_scalars[name], driver.pbl_tendencies.extra_scalars[name])
    # Native WRF initializes tracer_tend, unlike its held R*BLTEN fields,
    # before deciding whether this step calls PBL. A restored bank must
    # disappear on a skipped call and preserve unrelated scalar tendencies.
    resumed.pbl_tendencies.extra_scalars["unrelated"] = np.ones(1, np.float32)
    resumed._reset_step_tracer_tendencies(cfg)
    for name in TRACER_NAMES:
        assert not np.any(resumed.pbl_tendencies.extra_scalars[name])
    assert resumed.pbl_tendencies.extra_scalars["unrelated"][0] == 1


def _native_go_args(*values):
    import argparse
    from woof.go_cli import _add_go_arguments
    parser = argparse.ArgumentParser()
    _add_go_arguments(parser)
    return parser.parse_args(values)


def test_go_native_input_uses_shared_runner(monkeypatch):
    from pathlib import Path
    from woof.native_go import run_native_go
    import woof.wrfinput_forecast as runner
    calls = []
    monkeypatch.setattr(runner, "run_wrf_forecast", lambda *a, **kw: calls.append((a, kw)) or 0)
    assert run_native_go(_native_go_args("--wrfinput", "native", "--outdir", "forecast",
                                         "--run-seconds", "3600")) == 0
    args, kw = calls[0]
    assert args == (Path("native"), Path("forecast"))
    assert kw["run_seconds"] == 3600
    assert kw["exclusive_gpu"] is True


def test_go_native_shared_gpu_and_memory_override_reach_runner(monkeypatch):
    from woof.native_go import run_native_go
    from woof.core.resident_admission import memory_gate_overridden
    import woof.wrfinput_forecast as runner
    calls = []
    def run(*args, **kw):
        calls.append(kw)
        assert memory_gate_overridden()
        return 0
    monkeypatch.setattr(runner, "run_wrf_forecast", run)
    assert run_native_go(_native_go_args("--wrfinput", "native", "--allow-shared-gpu",
                                         "--no-memory-gate")) == 0
    assert calls[0]["exclusive_gpu"] is False
    assert calls[0]["allow_shared_gpu"] is True


def test_tracer_members_decline_packing_before_allocation():
    from dataclasses import replace
    from test_ensemble_prepared_execution import source, collector
    from woof.ensemble.prepared_execution import plan_initialized_member_execution
    inputs, node = source()
    node.cfg.run = replace(node.cfg.run, tracer_opt=2)
    plan = plan_initialized_member_execution(inputs, node, 2, collector(2))
    assert not plan.native
    assert any("passive tracers keep the ordinary member runner" in reason
               for reason in plan.fallback_reasons)
    assert node.clock.ticks == node.clock.step_count == 0


@pytest.mark.parametrize("flags", [("--prepared-root", "bundle"), ("--storm-nests", "2")])
def test_go_native_input_refuses_unused_requests(flags):
    from woof.native_go import run_native_go
    from woof.go_cli import GoRefusal
    with pytest.raises(GoRefusal, match="does not consume"):
        run_native_go(_native_go_args("--wrfinput", "native", *flags))
