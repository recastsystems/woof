"""MAD-WRF's native package must survive all configuration and state doors."""
from dataclasses import replace
import numpy as np
import pytest
from woof.core.madwrf_contract import validate_madwrf
from woof.core.passive_tracer_contract import active_tracers, input_fields
from woof.io import restart
from test_restart import _cfg, _shim_driver_state, _NumpyCupyShim, _fill_setup


@pytest.mark.parametrize("mode,mp,re,accepted", [
    (1, 96, 0, True), (2, 96, 0, True), (2, 8, 1, True),
    (1, 8, 0, False), (1, 96, 1, False), (2, 1, 0, False)])
def test_planner_and_run_share_native_package_requirements(mode, mp, re, accepted):
    from woof.physics_registry import (PLAN_SCHEMA, physics_registry, registry_sha256,
                                         validate_physics_plan)
    registry = physics_registry()
    option = next(k for k,v in registry["components"]["microphysics"]["options"].items()
                  if v.get("selectors", {}).get("mp_physics") == mp)
    plan = {"schema": PLAN_SCHEMA, "plan_id": "madwrf-doors",
            "registry_sha256": registry_sha256(),
            "context": {"source_id": "hrrr",
                        "runner_id": "tools.prepared_domain_tree_forecast",
                        "topology_id": "one-way-nested-v1"},
            "domains": [{"domain_id": domain,
                "template_id": "wsm6-ysu-mm5-noah-no-radiation-v1",
                "components": {"microphysics": option},
                "parameters": {"madwrf_opt": mode,
                               **({"use_mp_re": re} if not accepted else {})}}
                for domain in ("d01", "d02")],
            "edges": [{"parent_domain_id": "d01", "child_domain_id": "d02"}]}
    report = validate_physics_plan(plan)
    assert (not any(e["code"] == "madwrf-dependency" for e in report["errors"])) is accepted
    cfg = _cfg(mp_physics=mp, madwrf_opt=mode, use_mp_re=re)
    if accepted:
        validate_madwrf(cfg)
        assert report["launchable"], report["errors"]
    else:
        with pytest.raises(ValueError):
            validate_madwrf(cfg)


@pytest.mark.parametrize("mode", [1, 2])
def test_mad_package_inventory_and_checkpoint(monkeypatch, tmp_path, mode):
    from woof.core.device_inventory import state_array_shapes
    from woof.core.nest_fields import nest_field_kinds
    from woof.core.moist import moist_species, WRF_MOIST_ARRAY_SPECIES
    monkeypatch.setattr(_NumpyCupyShim, "zeros_like", staticmethod(np.zeros_like), raising=False)
    cfg = _cfg(mp_physics=96, madwrf_opt=mode, use_mp_re=0, moist=True,
               bl_pbl_physics=1, bldt=2.)
    validate_madwrf(cfg)
    state, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    assert tuple(n for n in moist_species(state) if n in WRF_MOIST_ARRAY_SPECIES) == ("qv", "qc", "qi", "qs")
    names = active_tracers(cfg)
    assert len(names) == (3 if mode == 2 else 0)
    assert [f.netcdf_name for f in input_fields(cfg)] == (
        ["tr_qcloud", "tr_qice", "tr_qsnow"] if mode == 2 else [])
    for i, name in enumerate(names):
        getattr(state, name).fill(i + .125)
        driver.pbl_tendencies.extra_scalars[name].fill(i + .25)
        assert state_array_shapes(cfg)[name] == (cfg.nz, cfg.ny, cfg.nx)
        assert name in nest_field_kinds(cfg)
    path = restart.write_restart(tmp_path / "mad.npz", state, cfg)
    target, resumed = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(target)
    restart.restore_restart(path, target, cfg)
    for name in names:
        np.testing.assert_array_equal(getattr(target, name), getattr(state, name))
        np.testing.assert_array_equal(resumed.pbl_tendencies.extra_scalars[name],
                                      driver.pbl_tendencies.extra_scalars[name])
    assert not any(k.startswith("madwrf_") for k in restart.configuration_echo(
        replace(cfg, mp_physics=8, madwrf_opt=0)))


@pytest.mark.parametrize("changes", [dict(mp_physics=8), dict(use_mp_re=1)])
def test_mode_one_keeps_native_required_settings(changes):
    cfg = _cfg(mp_physics=96, madwrf_opt=1, use_mp_re=0)
    with pytest.raises(ValueError, match="requires mp_physics=96 and use_mp_re=0"):
        validate_madwrf(replace(cfg, **changes))


@pytest.mark.parametrize("names", [("tr_qc", "tr_qi", "tr_qs"),
                                  tuple(f"tr17_{i}" for i in range(1, 9))])
def test_variable_tracer_bank_matches_native_column_rates(names):
    import cupy as cp
    from types import SimpleNamespace
    from pathlib import Path
    from woof.core.tracer_pblmix import tracer_pblmix_rates
    with np.load(Path(__file__).parent / "data/scalar_pblmix_wrf461.npz") as data:
        values = {k: np.ascontiguousarray(data[k][:, :4]).reshape(32, 50).T.copy()[:, None, :]
                  for k in ("qn", "dz", "rho", "exch_h", "dt", "tendency")}
    # The native reference has a different dt per column. Use one column
    # for each call, retaining the original Fortran output words.
    for col in (0, 15, 31):
        arrays = {k: cp.asarray(v[:, :, col:col+1].copy()) for k, v in values.items()}
        state = SimpleNamespace(**{n: arrays["qn"].copy() for n in names})
        scratch = {}
        state.scratch = lambda shape, slot: scratch.setdefault(slot, cp.zeros(shape, cp.float32))
        rates = tracer_pblmix_rates(state, arrays["dz"], arrays["rho"], arrays["exch_h"],
                                    float(values["dt"][0, 0, col]))
        assert set(rates) == set(names)
        for rate in rates.values():
            np.testing.assert_array_equal(cp.asnumpy(rate).view(np.uint32),
                values["tendency"][:, :, col:col+1].copy().view(np.uint32))
