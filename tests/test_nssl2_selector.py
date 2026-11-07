"""CPU contracts for the production MP18 selector and persistent binding."""

from __future__ import annotations

from dataclasses import replace
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import RunConfig
from woof.core import microphysics
from woof.core import nssl2_driver_support as driver_support
from woof.core import nssl2_runtime as runtime
from woof.core import physics
from woof.core import preflight
from woof.core.microphysics import MicrophysicsDiagnostics
from woof.core.nssl2_default_hooks import (
    NSSL2_NUCOND_SCRATCH,
    NSSL2ProductionBinding,
    make_nssl2_production_binding,
)
from woof.core.nssl2_contract import DEFAULT_RESTART_FIELDS
from woof.core.nssl2_driver_support import NSSL2DriverWorkspace
from woof.core.nssl2_fused_gs import NSSL2FusedGS
from woof.core.nssl2_production_coordinator import (
    NSSL2ProductionConfigurationError,
)
from woof.io import restart


_SHAPE = (3, 1, 2)


def _cfg(**changes):
    values = dict(
        nx=2, ny=1, nz=3, dx=1000.0, dy=1000.0, ztop=3000.0,
        dt=1.0, run_seconds=2.0, moist=True, mp_physics=18,
    )
    values.update(changes)
    return RunConfig(**values)


def test_mp18_state_and_precipitation_authorities_are_exactly_cross_pinned():
    assert driver_support.NSSL2_DRIVER_FIELDS == DEFAULT_RESTART_FIELDS
    assert runtime._REGISTRY_NAMES == DEFAULT_RESTART_FIELDS
    assert restart.NSSL2_RESTART_PROGNOSTICS == DEFAULT_RESTART_FIELDS

    runtime_slots = tuple(slot for _component, slot in
                          runtime._PRECIPITATION_SLOTS)
    physics_slots = tuple(slot for _component, slot in
                          physics.microphysics_scratch_slots(18))
    assert runtime_slots == restart.NSSL2_RESTART_PRECIPITATION_SLOTS
    assert physics_slots == restart.NSSL2_RESTART_PRECIPITATION_SLOTS


class _ScratchState:
    def __init__(self):
        self.p = np.full(_SHAPE, 85000.0, dtype=np.float32)
        self.w = np.zeros((_SHAPE[0] + 1, *_SHAPE[1:]), dtype=np.float32)
        self._scratch = {}

    def scratch(self, shape, name):
        shape = tuple(shape)
        value = self._scratch.get(name)
        if value is None:
            value = np.empty(shape, dtype=np.float32)
            self._scratch[name] = value
        elif value.shape != shape:
            raise ValueError(f"scratch shape mismatch for {name}")
        return value


def _manual_binding(state):
    workspace = NSSL2DriverWorkspace(
        state=np.empty((16, *_SHAPE), dtype=np.float32),
        category_surface_export=np.empty((5, *_SHAPE[1:]), dtype=np.float32),
        shape=_SHAPE,
        ignored_accumulator=np.empty(_SHAPE[1:], dtype=np.float32),
    )
    temperature = np.empty(_SHAPE, dtype=np.float32)
    target = np.empty(_SHAPE, dtype=np.float32)
    fused = NSSL2FusedGS(temperature, target, np.float32(1.0))
    hooks = runtime.NSSL2RuntimeHooks(
        fused_gs=fused,
        nucond_qvexcess=lambda workspace, fields: None,
    )
    return NSSL2ProductionBinding(
        state=state,
        shape=_SHAPE,
        dt_s=np.float32(1.0),
        workspace=workspace,
        fused_gs=fused,
        hooks=hooks,
        nucond_scratch=np.empty(_SHAPE, dtype=np.float32),
    )


def test_selector_lazily_forwards_binding_and_exact_due_policy(monkeypatch):
    state = SimpleNamespace(qv=np.ones((1,), dtype=np.float32))
    binding = _manual_binding(state)
    state.physics = SimpleNamespace(state=state, nssl2_binding=binding)
    sentinel = object()
    calls = []

    def apply(actual_state, cfg, dt_s, hooks, **kwargs):
        calls.append((actual_state, cfg, dt_s, hooks, kwargs))
        return sentinel

    monkeypatch.setattr(runtime, "apply_nssl2_production", apply)
    result = microphysics.apply(
        state, _cfg(), 1.0, refl_10cm_due=True)

    assert result is sentinel
    assert len(calls) == 1
    actual_state, cfg, step, hooks, kwargs = calls[0]
    assert actual_state is state
    assert cfg.mp_physics == 18
    assert step == 1.0
    assert hooks is binding.hooks
    assert kwargs == {
        "output_due": True,
        "radiation_due": True,
        "validate_values": False,
        "binding": binding,
    }


@pytest.mark.parametrize("failure", ["missing-driver", "wrong-state", "binding"])
def test_selector_rejects_missing_or_wrong_binding(failure):
    state = SimpleNamespace(qv=np.ones((1,), dtype=np.float32))
    if failure == "missing-driver":
        match = "PhysicsDriver"
    elif failure == "wrong-state":
        state.physics = SimpleNamespace(
            state=object(), nssl2_binding=_manual_binding(state))
        match = "PhysicsDriver"
    else:
        state.physics = SimpleNamespace(state=state, nssl2_binding=object())
        match = "persistent production binding"
    with pytest.raises(NSSL2ProductionConfigurationError, match=match):
        microphysics.apply(state, _cfg(), 1.0)


def test_binding_factory_owns_exact_named_scratch_and_float32_step():
    state = _ScratchState()
    binding = make_nssl2_production_binding(state, 1.0)

    assert isinstance(binding.dt_s, np.float32)
    assert binding.workspace.state is state._scratch["nssl2_driver_state"]
    assert binding.workspace.category_surface_export is \
        state._scratch["nssl2_driver_surface_export"]
    assert binding.workspace.ignored_accumulator is \
        state._scratch["nssl2_driver_ignored_accumulator"]
    assert binding.fused_gs.temperature_k is \
        state._scratch["nssl2_fused_temperature"]
    assert binding.fused_gs.primary_ice_target_m3 is \
        state._scratch["nssl2_primary_ice_target"]
    assert binding.nucond_scratch is state._scratch[NSSL2_NUCOND_SCRATCH]
    assert binding.hooks.fused_gs is binding.fused_gs
    binding.validate(state, np.nextafter(1.0, 2.0, dtype=np.float64))

    with pytest.raises(
            NSSL2ProductionConfigurationError, match="timestep differs"):
        binding.validate(state, 1.25)
    with pytest.raises(
            NSSL2ProductionConfigurationError, match="different DomainState"):
        binding.validate(_ScratchState(), 1.0)


def test_binding_detects_pointer_replacement_before_runtime_mutation():
    state = _ScratchState()
    binding = make_nssl2_production_binding(state, 1.0)
    state._scratch["nssl2_driver_state"] = np.empty_like(
        binding.workspace.state)
    with pytest.raises(
            NSSL2ProductionConfigurationError, match="canonical DomainState"):
        binding.validate(state, 1.0)


def test_selector_follows_a_changed_step_on_the_same_buffers(monkeypatch):
    """An adaptive clock hands microphysics a new dt every root step."""
    state = _ScratchState()
    state.qv = np.ones(_SHAPE, dtype=np.float32)
    binding = make_nssl2_production_binding(state, 1.0)
    state.physics = SimpleNamespace(
        state=state, nssl2_binding=binding, mp_physics=18)
    before = {name: value.tobytes()
              for name, value in state._scratch.items()}
    identities = {name: id(value) for name, value in state._scratch.items()}
    calls = []

    def apply(actual_state, cfg, dt_s, hooks, **kwargs):
        calls.append((dt_s, hooks, kwargs["binding"]))
        return None

    monkeypatch.setattr(runtime, "apply_nssl2_production", apply)
    for step in (1.25, 1.25, 0.5, 1.0):
        microphysics.apply(state, _cfg(), step)

    assert [dt_s for dt_s, _hooks, _binding in calls] == [
        1.25, 1.25, 0.5, 1.0]
    first, second, third, fourth = (bound for _dt, _h, bound in calls)
    assert first is not binding
    assert second is first
    assert third is not first
    for (dt_s, hooks, bound) in calls:
        assert bound.dt_s == np.float32(dt_s)
        assert isinstance(bound.dt_s, np.float32)
        assert bound.fused_gs.dt_s == bound.dt_s
        assert hooks is bound.hooks
        assert bound.hooks.fused_gs is bound.fused_gs
        assert bound.workspace is binding.workspace
        assert bound.nucond_scratch is binding.nucond_scratch
        assert bound.fused_gs.temperature_k is binding.fused_gs.temperature_k
        assert bound.fused_gs.primary_ice_target_m3 is \
            binding.fused_gs.primary_ice_target_m3
        assert bound.mode == binding.mode
        bound.validate(state, dt_s)
    assert state.physics.nssl2_binding is fourth
    # Rebinding allocates nothing and writes nothing.
    assert {name: id(value) for name, value in state._scratch.items()} == \
        identities
    assert {name: value.tobytes()
            for name, value in state._scratch.items()} == before
    assert binding.with_step(1.0) is binding


def test_adapter_still_refuses_a_binding_on_another_step():
    state = _ScratchState()
    state.qv = np.ones(_SHAPE, dtype=np.float32)
    binding = make_nssl2_production_binding(state, 1.0)
    state.physics = SimpleNamespace(
        state=state, nssl2_binding=binding, mp_physics=18)
    before = {name: value.tobytes()
              for name, value in state._scratch.items()}

    with pytest.raises(
            NSSL2ProductionConfigurationError, match="timestep differs"):
        runtime.apply_nssl2_production(
            state, _cfg(), 1.25, binding.hooks, binding=binding)

    assert {name: value.tobytes()
            for name, value in state._scratch.items()} == before


def test_supplied_workspace_reuses_buffers_and_allocates_no_zero_rate(
        monkeypatch):
    shape = (2, 1, 2)
    fields = [np.zeros(shape, dtype=np.float32) for _ in range(16)]
    density = np.ones(shape, dtype=np.float32)
    dz = np.full(shape, 100.0, dtype=np.float32)
    workspace = NSSL2DriverWorkspace(
        state=np.full((16, *shape), np.nan, dtype=np.float32),
        category_surface_export=np.full((5, 1, 2), np.nan, dtype=np.float32),
        shape=shape,
        ignored_accumulator=np.full((1, 2), 17.0, dtype=np.float32),
    )
    launches = []

    def forbidden(*args, **kwargs):
        raise AssertionError("selector-reachable CuPy allocation")

    fake_cp = SimpleNamespace(
        empty=forbidden, zeros=forbidden, zeros_like=forbidden)
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)

    def get_kernel(module, name):
        def launch(grid, block, args):
            launches.append((module, name, args))
        return launch

    monkeypatch.setattr(driver_support, "shape_kernel",
                        lambda module, name, shapes: get_kernel(module, name))
    result = driver_support.gather_initialize_and_sediment(
        density, dz, *fields, 1.0, temperature_k=density,
        workspace=workspace, cu_used=False)

    assert result.state is workspace.state
    assert result.category_surface_export is workspace.category_surface_export
    assert result.ignored_accumulator is workspace.ignored_accumulator
    np.testing.assert_array_equal(workspace.ignored_accumulator, 0.0)
    gather_args = launches[0][2]
    assert all(value is fields[0] for value in gather_args[17:21])


def test_reusable_workspace_cu_path_requires_and_uses_canonical_rates(
        monkeypatch):
    shape = (2, 1, 1)
    fields = [np.zeros(shape, dtype=np.float32) for _ in range(16)]
    density = np.ones(shape, dtype=np.float32)
    dz = np.ones(shape, dtype=np.float32)
    workspace = NSSL2DriverWorkspace(
        state=np.empty((16, *shape), dtype=np.float32),
        category_surface_export=np.empty((5, 1, 1), dtype=np.float32),
        shape=shape,
        ignored_accumulator=np.empty((1, 1), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="all four KF rate arrays"):
        driver_support.gather_initialize_and_sediment(
            density, dz, *fields, 1.0, temperature_k=density,
            workspace=workspace, cu_used=True)

    rates = [np.full(shape, index, dtype=np.float32)
             for index in range(1, 5)]
    launches = []
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    monkeypatch.setattr(
        driver_support, "shape_kernel",
        lambda module, name, shapes: (
            lambda grid, block, args: launches.append((name, args))))
    driver_support.gather_initialize_and_sediment(
        density, dz, *fields, 1.0, temperature_k=density,
        workspace=workspace, cu_used=True,
        qrcuten=rates[0], qscuten=rates[1],
        qicuten=rates[2], qccuten=rates[3])
    assert all(actual is expected for actual, expected in zip(
        launches[0][1][17:21], rates, strict=True))


def test_trusted_mp18_acceptance_uses_alias_checks_without_reductions(
        monkeypatch):
    surface = (1, 2)
    values = {
        name: np.full(surface, index, dtype=np.float32)
        for index, name in enumerate((
            "rainnc", "rainncv", "sr", "snownc", "snowncv",
            "graupelnc", "graupelncv", "hailnc", "hailncv"), start=1)
    }
    result = MicrophysicsDiagnostics(**values)
    driver = object.__new__(physics.PhysicsDriver)
    driver.state = SimpleNamespace(mup=np.zeros(surface, dtype=np.float32))
    driver.mp_physics = 18
    driver.microphysics = result
    driver._pending_rainbl = np.zeros(surface, dtype=np.float32)
    driver.microphysics_updates = 0
    driver.ruc_params = None
    driver.noahmp_params = None

    def forbidden(*args, **kwargs):
        raise AssertionError("trusted MP18 acceptance performed a reduction")

    monkeypatch.setattr(
        physics, "cp", SimpleNamespace(any=forbidden, maximum=np.maximum))
    driver.accept_microphysics(result)
    assert driver.microphysics_updates == 1
    np.testing.assert_array_equal(
        driver._pending_rainbl, np.maximum(result.rainncv, 0.0))

    with pytest.raises(ValueError, match="canonical PhysicsDriver array"):
        driver.accept_microphysics(replace(result, rainnc=result.rainnc.copy()))
    assert driver.microphysics_updates == 1


def test_mp18_persistent_scratch_preflight_and_restart_ownership():
    slots = preflight.scratch_slot_registry(_cfg())
    expected = {
        "nssl2_driver_state": (16, *_SHAPE),
        "nssl2_driver_surface_export": (5, *_SHAPE[1:]),
        "nssl2_driver_ignored_accumulator": _SHAPE[1:],
        "nssl2_fused_temperature": _SHAPE,
        "nssl2_primary_ice_target": _SHAPE,
        "nssl2_nucond_ss": _SHAPE,
    }
    for name, shape in expected.items():
        assert slots[name] == shape
        assert preflight.scratch_slot_uses_arena(name)
        assert name in restart.REBUILT_SCRATCH_SLOTS
    assert "nssl2_binding" in restart.DRIVER_REBUILT_ATTRS


def _initialized_gpu_mp18_state(cfg):
    import cupy as cp

    from woof.core.physics import initialize_physics
    from woof.core.state import DomainState

    state = DomainState(cfg)
    state.thb[...] = cp.asarray([299.0, 300.0, 301.0], dtype=cp.float32)
    state.phb[...] = cp.asarray(
        [0.0, 125.0, 250.0, 375.0], dtype=cp.float32) * cp.float32(9.81)
    state.thp[...] = cp.float32(0.0)
    state.php[...] = cp.float32(0.0)
    state.p[...] = cp.float32(85000.0)
    state.alt[...] = cp.float32(0.8)
    state.w[...] = cp.float32(0.0)
    values = {
        "qv": 0.005, "qc": 2.0e-5, "qr": 2.0e-6,
        "qi": 1.0e-6, "qs": 2.0e-6, "qg": 1.0e-6,
        "qh": 5.0e-7, "qndrop": 1.0e8, "qnr": 1.0e5,
        "qni": 1.0e5, "qns": 1.0e4, "qng": 1.0e4,
        "qnh": 1.0e3, "qnn": 4.0e8,
        "qvolg": 1.0e-6 / 500.0, "qvolh": 5.0e-7 / 900.0,
    }
    for name, value in values.items():
        getattr(state, name)[...] = cp.float32(value)
    return state, initialize_physics(state, cfg)


@pytest.mark.gpu
@requires_gpu
def test_gpu_real_selector_reuses_and_overwrites_all_binding_buffers():
    import cupy as cp

    from woof.core.refl import consume_refl_10cm

    cfg = _cfg(nx=2, ny=2, dt=0.25)
    state, driver = _initialized_gpu_mp18_state(cfg)
    shape = state.p.shape
    binding = driver.nssl2_binding
    owned = (
        binding.workspace.state,
        binding.workspace.category_surface_export,
        binding.workspace.ignored_accumulator,
        binding.fused_gs.temperature_k,
        binding.fused_gs.primary_ice_target_m3,
        binding.nucond_scratch,
    )
    identities = tuple(id(value) for value in owned)
    for value in owned:
        value.fill(cp.nan)

    diagnostics = microphysics.apply(
        state, cfg, cfg.dt, refl_10cm_due=False)
    driver.accept_microphysics(diagnostics)
    cp.cuda.Stream.null.synchronize()
    assert driver.refl_10cm is None
    for value in owned:
        assert bool(cp.isfinite(value).all())
    assert tuple(id(value) for value in owned) == identities

    diagnostics = microphysics.apply(
        state, cfg, cfg.dt, refl_10cm_due=True)
    driver.accept_microphysics(diagnostics)
    reflectivity = consume_refl_10cm(state)
    cp.cuda.Stream.null.synchronize()
    assert reflectivity.shape == shape
    assert bool(cp.isfinite(reflectivity).all())
    pool = cp.get_default_memory_pool()
    warmed_total = pool.total_bytes()

    diagnostics = microphysics.apply(
        state, cfg, cfg.dt, refl_10cm_due=False)
    driver.accept_microphysics(diagnostics)
    diagnostics = microphysics.apply(
        state, cfg, cfg.dt, refl_10cm_due=True)
    driver.accept_microphysics(diagnostics)
    consume_refl_10cm(state)
    cp.cuda.Stream.null.synchronize()
    assert pool.total_bytes() == warmed_total
    assert tuple(id(value) for value in owned) == identities
    assert driver.microphysics_updates == 4


@pytest.mark.gpu
@requires_gpu
def test_gpu_real_selector_step_change_matches_a_binding_built_on_it():
    """A mid-run dt change runs the scheme exactly as a fresh binding would.

    Both domains take one step at 0.25 s.  The reference then gets a
    binding built from scratch on 0.5 s; the selector domain is left to
    move its own.  Every prognostic and precipitation slot must agree to
    the bit after the 0.5 s step and after a return to 0.25 s.
    """
    import cupy as cp

    from woof.core.nssl2_contract import resolve_nssl2_mode_for_config

    cfg = _cfg(nx=2, ny=2, dt=0.25)
    mode = resolve_nssl2_mode_for_config(cfg)

    def advance(state, driver, step):
        diagnostics = microphysics.apply(state, cfg, step)
        driver.accept_microphysics(diagnostics)

    def snapshot(state):
        cp.cuda.Stream.null.synchronize()
        values = {name: cp.asnumpy(getattr(state, name)).tobytes()
                  for name in DEFAULT_RESTART_FIELDS}
        for slot in restart.NSSL2_RESTART_PRECIPITATION_SLOTS:
            values[slot] = cp.asnumpy(state._scratch[slot]).tobytes()
        values["thp"] = cp.asnumpy(state.thp).tobytes()
        return values

    moving, moving_driver = _initialized_gpu_mp18_state(cfg)
    reference, reference_driver = _initialized_gpu_mp18_state(cfg)
    first_binding = moving_driver.nssl2_binding
    for state, driver in ((moving, moving_driver),
                          (reference, reference_driver)):
        advance(state, driver, 0.25)
    assert snapshot(moving) == snapshot(reference)

    reference_driver.nssl2_binding = make_nssl2_production_binding(
        reference, 0.5, mode=mode)
    advance(moving, moving_driver, 0.5)
    advance(reference, reference_driver, 0.5)
    assert moving_driver.nssl2_binding.dt_s == np.float32(0.5)
    assert moving_driver.nssl2_binding.workspace is first_binding.workspace
    assert snapshot(moving) == snapshot(reference)

    reference_driver.nssl2_binding = make_nssl2_production_binding(
        reference, 0.25, mode=mode)
    advance(moving, moving_driver, 0.25)
    advance(reference, reference_driver, 0.25)
    assert moving_driver.nssl2_binding.dt_s == np.float32(0.25)
    assert snapshot(moving) == snapshot(reference)
    assert moving_driver.microphysics_updates == 3


@pytest.mark.gpu
@requires_gpu
def test_gpu_real_selector_restart_continuation_is_bitwise(tmp_path):
    import cupy as cp

    from woof.core.refl import consume_refl_10cm

    cfg = _cfg(nx=2, ny=2, dt=0.25)

    def advance(state, driver, steps):
        for _ in range(steps):
            diagnostics = microphysics.apply(
                state, cfg, cfg.dt, refl_10cm_due=True)
            driver.accept_microphysics(diagnostics)
            reflectivity = consume_refl_10cm(state)
            assert reflectivity is not None
            state.elapsed_seconds += float(cfg.dt)
        cp.cuda.Stream.null.synchronize()

    def serialized_bytes(state):
        manifest = {}
        manifest.update(restart.state_manifest(state))
        manifest.update(restart._scratch_manifest(state))
        manifest.update(restart._driver_manifest(state.physics))
        return {
            key: restart._host(value).tobytes()
            for key, value in sorted(manifest.items())
        }

    straight, straight_driver = _initialized_gpu_mp18_state(cfg)
    advance(straight, straight_driver, 4)

    split, split_driver = _initialized_gpu_mp18_state(cfg)
    advance(split, split_driver, 2)
    boundary = serialized_bytes(split)
    path = restart.write_restart(tmp_path / "mp18-selector.npz", split, cfg)
    header = restart.read_restart_header(path)
    assert not any(
        name.startswith("scratch/nssl2_")
        for name in header["array_manifest"])

    resumed, resumed_driver = _initialized_gpu_mp18_state(cfg)
    rebuilt_binding = resumed_driver.nssl2_binding
    rebuilt_ids = tuple(id(value) for value in (
        rebuilt_binding.workspace.state,
        rebuilt_binding.workspace.category_surface_export,
        rebuilt_binding.workspace.ignored_accumulator,
        rebuilt_binding.fused_gs.temperature_k,
        rebuilt_binding.fused_gs.primary_ice_target_m3,
        rebuilt_binding.nucond_scratch,
    ))
    assert rebuilt_binding is not split_driver.nssl2_binding
    restart.restore_restart(path, resumed, cfg)
    assert resumed_driver.nssl2_binding is rebuilt_binding
    assert tuple(id(value) for value in (
        rebuilt_binding.workspace.state,
        rebuilt_binding.workspace.category_surface_export,
        rebuilt_binding.workspace.ignored_accumulator,
        rebuilt_binding.fused_gs.temperature_k,
        rebuilt_binding.fused_gs.primary_ice_target_m3,
        rebuilt_binding.nucond_scratch,
    )) == rebuilt_ids
    assert serialized_bytes(resumed) == boundary

    advance(resumed, resumed_driver, 2)
    assert serialized_bytes(resumed) == serialized_bytes(straight)
    assert resumed_driver.microphysics_updates == \
        straight_driver.microphysics_updates == 4
    assert resumed.elapsed_seconds == straight.elapsed_seconds == 1.0
