"""Current-clock grouping and unchanged-code leaf binding contracts."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import threading
from types import SimpleNamespace

import pytest

from woof.ensemble.packed_production_physics import (
    PackedProductionPhysics, _MemberDriver, _rebind_function, _mynn_pbl_selectors,
    validate_production_physics_group)


def _group():
    cfg = SimpleNamespace(dt=6.0, time_step_sound=4, use_adaptive_time_step=True,
        sf_sfclay_physics=5, sf_surface_physics=3, bl_pbl_physics=5,
        cu_physics=0, spp_pbl=0, spp_lsm=0, mosaic_lu=0, mosaic_soil=0,
        bl_mynn_version="wrf_461", bl_mynn_gsd41_unsquared_qtke=False,
        bl_mynn_cloud_tendency_form="wrf_461", icloud_bl=1,
        swint_opt=0, aer_opt=0, alb_sol=0)
    driver = SimpleNamespace(state=SimpleNamespace(elapsed_seconds=0.0,
        domain_start_offset=0.0, p_top=10000.0), radt_minutes=12.0,
        bldt_seconds=6.0, stepbl=1, radiation_due_override=True,
        surface_pbl_due_override=True, carriers_need_producer_refresh=False,
        carriers=SimpleNamespace(unsourced_consumed=lambda scheme: True))
    return [deepcopy(driver), deepcopy(driver)], [deepcopy(cfg), deepcopy(cfg)]


def test_group_uses_current_adaptive_dt_and_due_overrides_without_forcing_a_step():
    drivers, configs = _group()
    first = validate_production_physics_group(drivers, configs)
    for cfg in configs:
        cfg.dt, cfg.time_step_sound = 7.5, 8
    for driver in drivers:
        driver.state.elapsed_seconds = 7.5
        driver.radiation_due_override = False
        driver.bldt_seconds = 7.5
    current = validate_production_physics_group(drivers, configs)
    assert current["config"] != first["config"]
    assert current["cadence"] != first["cadence"]
    assert all(cfg.dt == 7.5 for cfg in configs)


@pytest.mark.parametrize("target,name,value,reason", [
    ("state", "elapsed_seconds", 0.01, "live clock"),
    ("state", "domain_start_offset", 1.0, "activation"),
    ("state", "p_top", 9000.0, "top"),
    ("driver", "radiation_due_override", False, "due schedule"),
    ("driver", "surface_pbl_due_override", False, "due schedule"),
    ("cfg", "dt", 7.0, "current configuration"),
    ("cfg", "time_step_sound", 8, "current configuration"),
])
def test_group_refuses_divergence_before_driver_writes(target, name, value, reason):
    drivers, configs = _group()
    owner = {"cfg": configs[1], "driver": drivers[1], "state": drivers[1].state}[target]
    setattr(owner, name, value)
    with pytest.raises(ValueError, match=reason):
        validate_production_physics_group(drivers, configs)


def test_group_refuses_different_producer_refresh_and_shared_driver():
    drivers, configs = _group()
    drivers[1].carriers.unsourced_consumed = lambda scheme: False
    with pytest.raises(ValueError, match="producer refreshes"):
        validate_production_physics_group(drivers, configs)
    with pytest.raises(ValueError, match="mutable driver owner"):
        validate_production_physics_group([drivers[0], drivers[0]], configs)


@pytest.mark.parametrize("name,value", [("ruc_irrigation", "wrf_45"), ("ruc_qvg_cold_start", "air"),
    ("ruc_2m_diagnostic", "log_profile"), ("ruc_snow", "wrf_45")])
def test_ruc_science_selector_divergence_is_refused_before_composition(name, value):
    drivers, configs = _group()
    for cfg in configs:
        cfg.ruc_irrigation, cfg.ruc_qvg_cold_start = "wrf_461", "wrf"
        cfg.ruc_2m_diagnostic, cfg.ruc_snow = "flux", "wrf_461"
    setattr(configs[1], name, value)
    with pytest.raises(ValueError, match="current configuration"):
        validate_production_physics_group(drivers, configs)


@pytest.mark.parametrize("name,value", [("bl_mynn_version", "gsd_41"),
    ("bl_mynn_gsd41_unsquared_qtke", True), ("bl_mynn_cloud_tendency_form", "gsd_41")])
def test_mynn_generation_divergence_is_refused_before_member_writes(name, value):
    drivers, configs = _group()
    setattr(configs[1], name, value)
    with pytest.raises(ValueError, match="current configuration"):
        validate_production_physics_group(drivers, configs)


@pytest.mark.parametrize("name,value,reason", [("bl_mynn_version", "unknown", "version"),
    ("bl_mynn_version", 1, "version"), ("bl_mynn_gsd41_unsquared_qtke", 0, "unsquared_qtke"),
    ("bl_mynn_cloud_tendency_form", "unknown", "cloud tendency")])
def test_invalid_mynn_generation_metadata_is_refused_before_entry(name, value, reason):
    drivers, configs = _group()
    for cfg in configs:
        setattr(cfg, name, value)
    with pytest.raises((ValueError, TypeError), match=reason):
        validate_production_physics_group(drivers, configs)


def test_gsd_clouds_keep_the_original_legacy_radiation_guard():
    drivers, configs = _group()
    for cfg in configs:
        cfg.bl_mynn_version = "gsd_41"
    with pytest.raises(ValueError, match="legacy RRTMG in-cloud"):
        validate_production_physics_group(drivers, configs)
    for cfg in configs:
        cfg.icloud_bl = 0
    with pytest.raises(ValueError, match="original numerical driver requirement icloud_bl=1"):
        validate_production_physics_group(drivers, configs)


@pytest.mark.parametrize("name,value,reason", [
    ("swint_opt", 1, "between-call fit state"),
    ("aer_opt", 3, "aerosol-band optics"),
    ("alb_sol", 1, "ALBSOL/ALBBCKSOL aliases"),
])
def test_unbound_radiation_carriers_refuse_before_original_driver_mutation(name, value, reason):
    drivers, configs = _group()
    for cfg in configs:
        setattr(cfg, name, value)
    before = [deepcopy(vars(driver.state)) for driver in drivers]
    with pytest.raises(ValueError, match=reason):
        validate_production_physics_group(drivers, configs)
    assert [vars(driver.state) for driver in drivers] == before


def test_inactive_unsquared_flag_retains_the_original_default_omitted_arguments():
    drivers, configs = _group()
    for cfg in configs:
        cfg.bl_mynn_gsd41_unsquared_qtke = True
    assert validate_production_physics_group(drivers, configs)["members"] == 2
    assert _mynn_pbl_selectors(configs[0]) == {"bl_mynn_version": "wrf_461",
        "bl_mynn_gsd41_unsquared_qtke": False, "bl_mynn_cloud_tendency_form": "wrf_461",
        "mynn_fixes_v61": False, "mynn_boulac_down_sign": False}


@pytest.mark.parametrize("version,unsquared,cloud", [("wrf_461", False, "wrf_461"),
    ("gsd_41", False, "wrf_461"), ("gsd_41", True, "gsd_41")])
def test_pbl_leaf_forwards_actual_generation_options_without_losing_default_omission(
        version, unsquared, cloud, monkeypatch):
    import woof.ensemble.batch_mynn as mynn
    _, configs = _group()
    for cfg in configs:
        cfg.bl_mynn_version, cfg.bl_mynn_gsd41_unsquared_qtke = version, unsquared
        cfg.bl_mynn_cloud_tendency_form, cfg.icloud_bl = cloud, 1
        cfg.ny, cfg.nx, cfg.nz = 3, 5, 8
    seen, gathers = [], []
    owner = object.__new__(PackedProductionPhysics)
    owner.configs, owner.members, owner._components = configs, 2, {}
    owner.available_bytes, owner._priced, owner.xp = 1000000, 0, object()
    owner._gather_maps = lambda *args, **kwargs: gathers.append(args[0]) or {}
    owner._gather = lambda *args, **kwargs: object()
    owner._reserve = lambda *args: None
    owner._scatter = lambda *args: None
    component = SimpleNamespace(storage=SimpleNamespace(plan=SimpleNamespace(required_bytes=lambda n: 0)))
    class Bound:
        storage = component.storage
        def __call__(self, **kwargs):
            return {}
    def prepare(*args, **kwargs):
        seen.append(kwargs["options"])
        return Bound()
    monkeypatch.setattr(mynn, "prepare_mynn_pbl_column_batch", prepare)
    kwargs = dict(dx=1000., delt=6., itimestep=1, mp_physics=8, scalar_pblmix=0,
        spp_pbl=0, column_chunk=4, w=object(), closure=2.6, bl_mynn_cloudpdf=2,
        bl_mynn_mixlength=2, bl_mynn_edmf=1, bl_mynn_edmf_mom=1, bl_mynn_edmf_tke=0,
        bl_mynn_mixscalars=0, bl_mynn_cloudmix=1, bl_mynn_mixqt=0, bl_mynn_output=0,
        bl_mynn_tkeadvect=False, icloud_bl=1)
    if version != "wrf_461":
        kwargs.update(bl_mynn_version=version, bl_mynn_gsd41_unsquared_qtke=unsquared,
                      bl_mynn_cloud_tendency_form=cloud)
    requests = [(({}, {}), dict(kwargs), None) for _ in configs]
    assert owner._pbl(("pbl", 0), requests) == ({}, {})
    assert {name: seen[0][name] for name in _mynn_pbl_selectors(configs[0])} == {
        "bl_mynn_version": version, "bl_mynn_gsd41_unsquared_qtke": unsquared,
        "bl_mynn_cloud_tendency_form": cloud, "mynn_fixes_v61": False,
        "mynn_boulac_down_sign": False}
    assert len(gathers) == 2
    requests[1][1]["bl_mynn_gsd41_unsquared_qtke"] = not unsquared
    with pytest.raises(ValueError, match="actual member configuration"):
        owner._pbl(("pbl", 0), requests)
    assert len(gathers) == 2


@pytest.mark.parametrize("different", ["presence", "shape", "dtype"])
def test_legacy_irrigation_fraction_authority_cannot_broadcast_another_members_layout(different):
    import numpy as np
    drivers, configs = _group()
    for cfg in configs:
        cfg.ruc_irrigation = "wrf_45"
    for driver in drivers:
        driver.fields = {"landusef": np.zeros((24, 3, 5), np.float32)}
    if different == "presence":
        del drivers[1].fields["landusef"]
    elif different == "shape":
        drivers[1].fields["landusef"] = np.zeros((20, 3, 5), np.float32)
    else:
        drivers[1].fields["landusef"] = np.zeros((24, 3, 5), np.float64)
    with pytest.raises(ValueError, match="landusef"):
        validate_production_physics_group(drivers, configs)


def _ordinary_leaf(value):
    return "ordinary", value


class _OriginalDriver:
    def __init__(self):
        self.calls, self.state = [], object()
        self.radiation_callable = _ordinary_leaf

    def _run_sfclay(self, value):
        self.calls.append("seed")
        first = launch_mynn_surface_layer(value)
        self.calls.append("sea staging")
        second = launch_mynn_surface_layer(value)
        self.calls.append("blend")
        return first, second


launch_mynn_surface_layer = _ordinary_leaf


def test_proxy_preserves_seed_sea_staging_blend_order_and_original_owner_identity():
    driver = _OriginalDriver()
    calls = []
    owner = SimpleNamespace(drivers=(driver,), _rendezvous=lambda *args: calls.append(args) or "packed")
    proxy = _MemberDriver(owner, 0)
    original_code = driver._run_sfclay.__func__.__code__
    assert proxy._run_sfclay("input") == ("packed", "packed")
    assert driver.calls == ["seed", "sea staging", "blend"]
    assert [row[3] for row in calls] == [("surface", 0), ("surface", 1)]
    assert driver._run_sfclay.__func__.__code__ is original_code
    assert driver.radiation_callable is _ordinary_leaf
    assert launch_mynn_surface_layer is _ordinary_leaf
    assert proxy.state is driver.state
    assert "compute" not in vars(driver) and "_run_sfclay" not in vars(driver)


def test_function_binding_preserves_original_code_and_other_module_globals():
    bound = _rebind_function(_OriginalDriver._run_sfclay,
        {"launch_mynn_surface_layer": lambda value: ("packed", value)})
    assert bound.__code__ is _OriginalDriver._run_sfclay.__code__
    assert bound.__globals__["_ordinary_leaf"] is _ordinary_leaf
    assert _OriginalDriver._run_sfclay.__globals__["launch_mynn_surface_layer"] is _ordinary_leaf


class _Queue:
    def __enter__(self):
        return self

    def __exit__(self, *unused):
        return False

    def wait_event(self, unused):
        pass

    def record(self, unused):
        pass


def _rendezvous_owner():
    owner = object.__new__(PackedProductionPhysics)
    owner.xp = SimpleNamespace(cuda=SimpleNamespace(Event=_Queue,
        Device=lambda device: _Queue(), get_current_stream=_Queue))
    owner.device, owner.members, owner.stream = 0, 2, _Queue()
    owner._condition, owner._rounds = threading.Condition(), {}
    owner._finished, owner._error = set(), None
    owner.workspace_reuse = None
    owner.receipt = {"leaf_calls": {}}
    return owner


def test_two_native_requests_join_one_leaf_without_reordering_members():
    owner = _rendezvous_owner()
    calls = []
    def surface(binding, requests):
        calls.append((binding, requests))
        return tuple(args[0] for args, _, _ in requests)
    owner._surface = surface
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(owner._rendezvous, member, 0, "surface", ("surface", 0),
                    (f"member {member}",), {}) for member in (1, 0)]
        assert [future.result(timeout=2) for future in futures] == ["member 1", "member 0"]
    assert len(calls) == 1
    assert [request[0][0] for request in calls[0][1]] == ["member 0", "member 1"]
    assert owner.receipt["leaf_calls"] == {"surface": 1}


def test_divergent_leaf_order_wakes_both_workers_instead_of_deadlocking():
    owner = _rendezvous_owner()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(owner._rendezvous, member, 0, kind, (kind, 0), (), {})
                   for member, kind in ((0, "surface"), (1, "land"))]
        for future in futures:
            with pytest.raises(ValueError, match="leaf order differs"):
                future.result(timeout=2)


def test_finished_member_cannot_leave_a_due_leaf_waiting_forever():
    owner = _rendezvous_owner()
    owner._finished.add(1)
    with pytest.raises(ValueError, match="completed before another member's leaf"):
        owner._rendezvous(0, 0, "surface", ("surface", 0), (), {})


def test_original_workspace_reuse_exists_only_inside_complete_leaf_rendezvous():
    from woof.ensemble.packed_production_physics import OriginalWorkspaceReuseScope
    owner = _rendezvous_owner()
    owner._active, owner._closed = True, False
    owner.drivers = tuple(SimpleNamespace(state=object()) for _ in range(2))
    owner.adapters = (object(), object())
    owner.workspace_reuse = OriginalWorkspaceReuseScope(owner)
    scope = owner.workspace_reuse
    with pytest.raises(RuntimeError, match="complete active member leaf"):
        scope.require("pbl", scratch_owner=owner.drivers[0].state)
    trace = []
    def pbl(binding, requests):
        scope.require("pbl", scratch_owner=owner.drivers[0].state)
        assert len(requests) == 2
        trace.append("all original owners parked")
        with pytest.raises(ValueError, match="another original member scratch"):
            scope.require("pbl", scratch_owner=owner.drivers[1].state)
        return (0, 1)
    owner._pbl = pbl
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(owner._rendezvous, member, 0, "pbl", ("pbl", 0), (), {})
                   for member in (0, 1)]
        assert [future.result(timeout=2) for future in futures] == [0, 1]
    assert trace == ["all original owners parked"] and scope.kind is None
    with pytest.raises(RuntimeError, match="complete active member leaf"):
        scope.require("pbl", scratch_owner=owner.drivers[0].state)


def test_failed_leaf_clears_original_workspace_reuse_scope_before_original_recovery():
    from woof.ensemble.packed_production_physics import OriginalWorkspaceReuseScope
    owner = _rendezvous_owner()
    owner._active, owner._closed = True, False
    owner.drivers = tuple(SimpleNamespace(state=object()) for _ in range(2))
    owner.adapters = (object(), object())
    owner.workspace_reuse = OriginalWorkspaceReuseScope(owner)
    def radiation(binding, requests):
        owner.workspace_reuse.require("radiation", adapters=owner.adapters)
        with pytest.raises(ValueError, match="another original member adapter"):
            owner.workspace_reuse.require("radiation", adapters=(object(), object()))
        raise ValueError("native leaf failed")
    owner._radiation = radiation
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(owner._rendezvous, member, 0, "radiation", ("radiation", 0), (), {})
                   for member in (0, 1)]
        for future in futures:
            with pytest.raises(ValueError, match="native leaf failed"):
                future.result(timeout=2)
    assert owner.workspace_reuse.kind is None


@pytest.mark.parametrize("version,on", [("wrf_461", True), ("gsd_41", False)])
def test_pbl_leaf_carries_the_default_mynn_v61_fixes(version, on):
    """Correct-mode MY1 (mynn_fixes_v61, on by default since 3.0.3) rides the
    packed MYNN leaf where the member wrapper runs it (wrf_461), so a
    default MYNN ensemble keeps the packed path with the dew fix instead of
    being declined or silently packed without it.  Breakage prevented: a
    packed member running the WRF 4.6.1 dew limiter its own config fixed."""
    from woof.ensemble.batch_mynn import _pbl_options
    _, configs = _group()
    cfg = configs[0]
    cfg.bl_mynn_version, cfg.mynn_fixes_v61, cfg.bl_pbl_physics = version, True, 5
    if version == "gsd_41":
        cfg.bl_mynn_cloud_tendency_form = "wrf_461"
    assert _mynn_pbl_selectors(cfg)["mynn_fixes_v61"] is on
    assert _pbl_options({"mynn_fixes_v61": True})["mynn_fixes_v61"] is True
    with pytest.raises(ValueError, match="wrf_461 unit"):
        _pbl_options({"mynn_fixes_v61": True, "bl_mynn_version": "gsd_41"})
    with pytest.raises(ValueError, match="must be bool"):
        _pbl_options({"mynn_fixes_v61": 1})
