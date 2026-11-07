"""Every integer field a routed scheme collects must carry a device policy.

``StateHealthValidator._refresh`` classifies each collected descriptor by
dtype.  float32 is checked; int32 is checked only if the field name carries an
explicit range policy, or skipped only if it carries a documented exclusion;
anything else raises.  That refusal is deliberate -- an integer field silently
admitted to a float32 gate would be compared against reinterpreted bit
patterns -- but it is also a *launch* failure, not a compile failure, and it
fires at ``require_healthy(phase="initialized-or-restored")``, i.e. after
static, after ingest, after the device state is built and before step one.

MYNN found that edge.  ``surface.ktop_plume`` is int32 and was in neither set,
so a MYNN 5/5 forecast died at initialization on *every* domain, single or
nested -- observed on the real d01 geometry through ``python -m woof.cli run``
before this file existed.

So the tests here are about the *policy*, not about MYNN.  They rebuild the
real production inventory on the host for every routed PBL / surface-layer /
LSM combination and classify it through the shipped
``woof.core.health.gpu_integer_policy``.  A future scheme that adds an int32
field with no policy fails here, at ``pytest`` time, instead of at minute one
of somebody's forecast.

NO DEVICE IS OPENED
-------------------
``woof.core.state`` and ``woof.core.physics`` are bound to NumPy for the
inventory tests, and the arrays they produce are asserted to be
``numpy.ndarray``.  The device tests are marked ``gpu``.

Noah-MP had the identical defect in two fields, ``surface.isnowxy`` and
``surface.pgsxy``, found by this sweep rather than by a forecast.  Both now
carry range policies read off the pinned WSL WRF tree, and the second half of
this file is their evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.core.health import (
    GPU_INTEGER_EXCLUSIONS,
    FieldRule,
    collect_state_fields,
    gpu_integer_policy,
    validate_fields_cpu,
)

#: The message ``_refresh`` has always raised.  Pinned as a literal because it
#: is what a user reads when a scheme is admitted without a policy, and
#: because a test that rebuilt it from the implementation could not detect the
#: implementation dropping it.
_REFUSAL = ("has unsupported dtype int32; add an explicit integer range "
            "policy or a documented exclusion before this inventory can run")

#: Small enough to sweep the whole routed cross-product on the host, deep
#: enough that ``nz - 1`` and ``nz`` are distinguishable bounds.
_NX, _NY, _NZ = 3, 2, 8

#: Measured, not projected: the number of (sf_sfclay_physics,
#: sf_surface_physics, bl_pbl_physics) triples that are BOTH routed by
#: PHYSICS_SLOT_DISPATCH and accepted by validate_run_config at this
#: checkout.  Pinned so a sweep that silently shrank cannot pass by
#: measuring less.  Re-measured 2026-07-30: the WRF-owned MYNN/RUC and
#: MYNN/Noah-MP pairings increase the admitted matrix from 19 to 21.
#:
#: Re-measured again 2026-07-30 for the v1.3.1 wave: 21 -> 33.  The
#: combination lane made WRF v4.6.1's own PBL/surface-layer table the
#: admission authority and it admits three pairings ArWen used to refuse
#: under the (too broad) "MYNN half suite" story:
#:
#:   (bl_pbl=0, sf_sfclay=5)   PBL off with MYNN surface
#:   (bl_pbl=5, sf_sfclay=1)   MYNN PBL with revised MM5 surface
#:   (bl_pbl=5, sf_sfclay=91)  MYNN PBL with classic MM5 surface
#:
#: Each is legal for all four routed land-surface values {0, 2, 3, 4},
#: so the admitted set grows by exactly 3 x 4 = 12.  The complement is
#: unchanged and still accounts for the whole 4 x 4 x 3 = 48 product:
#: 11 refusals with the surface layer off (its one admitted member is
#: the all-off (0, 0, 0)), plus the 4 remaining WRF-fatal
#: (bl_pbl=1, sf_sfclay=5) rows -- 15 refused, 33 admitted.
#:
#: Re-measured 2026-08-03: 33 -> 41, because the gray-zone lane made
#: Shin-Hong (``bl_pbl_physics=11``) routable and never re-measured this
#: pin.  It is lane bookkeeping and not a release-line movement: 33 is
#: IDENTICAL at ``cf159eb2`` (the shipped 1.5.1), ``cfcfa9a9`` (the
#: pre-merge lane tip that added the route) and ``61488333`` (the merge).
#: The whole +8 is scheme 11's own admitted slice, and nothing else moved:
#:
#:   sfclay {1, 91} x lsm {0, 2, 3, 4} x pbl 11  ->  8 admitted
#:   sfclay {0, 5}  x lsm {0, 2, 3, 4} x pbl 11  ->  8 refused
#:
#: Shin-Hong inherits YSU's surface-layer pairing exactly, which is the
#: point: WRF v4.6.1's own table admits it with the revised and classic
#: MM5 surface layers and refuses it with the surface layer off and with
#: MYNN's, and the refusal names that authority verbatim ("WRF v4.6.1
#: PBL/surface-layer compatibility ... WRF v4.6.1 refuses this pairing").
#: pbl=1 measures the same 8 admitted / 8 refused split at this checkout.
#:
#: Measured whole rather than as a delta, the sweep is now the
#: 4 x 4 x 5 = 80 product of the routed selectors:
#:
#:   admitted 41 = 13 (pbl 0) + 8 (pbl 1) + 12 (pbl 5) + 8 (pbl 11)
#:                  + 0 (SASE)
#:   refused  39 =  3        + 8        + 4        + 8        + 16
#:
#: The "4 x 4 x 3 = 48" above was already one axis behind before this
#: lane: SASE's routed ``pbl900`` selector had made the product 64.  The
#: 33 it explained stayed correct anyway, because SASE is admitted with
#: no surface-layer value on this probe's reference config (it wants
#: km_opt=0, which ``_config`` does not set), so it contributed 16
#: refusals and zero admissions.  Scheme 11 is the first PBL addition
#: since that actually lands in the admitted set.
#:
#: MYJ RE-MEASURE, 2026-08-09 (development branch).  This port routes TWO new
#: selectors at once -- sf_sfclay_physics=2 and bl_pbl_physics=2 -- so the
#: product goes 4 x 4 x 5 = 80 to 5 x 4 x 6 = 120, and the admitted count
#: is re-measured, not projected:
#:
#:   admitted 45 = 13 (pbl 0) + 8 (pbl 1) + 4 (pbl 2) + 12 (pbl 5)
#:                  + 8 (pbl 11) + 0 (SASE)
#:
#: MYJ's own 4 is one per land-surface value at the single surface layer
#: it admits, which is what a PAIRED scheme looks like on this axis:
#: every other PBL here spreads over two or three surface layers, and MYJ
#: over exactly one (module_physics_init.F:3770-3772).  The 4 new
#: sfclay=2 cells under the other PBLs are all refused by
#: validate_myj_pairing, which is the reverse half of the same law.
#:
#: UW RE-MEASURE, 2026-09-30 (development branch).  bl_pbl_physics=9 is
#: routed, so the product goes 5 x 4 x 6 = 120 to 5 x 4 x 7 = 140, and
#: the admitted count is re-measured: 57 = 45 + 12 (pbl 9), one per
#: land-surface value at each of the three surface layers it admits (1,
#: 5, 91).  Its sfclay=0 cells are refused by name (no UST/HFX/QFX) and
#: its sfclay=2 cells by validate_myj_pairing (the TKE_MYJ scan).
# Re-measured 2026-10-09: CCPP adds eight admitted inventories; int32 policy is unchanged.
# Batch004 measured all admitted surface/PBL routes, including slab and new PBLs.
# Batch007 full routed surface/PBL sweep: 247 accepted inventories.
# Measured in the batch007 estimator-inventory receipt; seven integer names unchanged.
_ROUTED_COMBINATIONS = 247

#: Every int32 descriptor the routed cross-product actually produces.
_INT32_DESCRIPTORS = frozenset({
    "surface.ebal",       # Noah energy-balance case
    "surface.isltyp",     # Noah soil category (excluded)
    "surface.isnowxy",    # Noah-MP snow-layer count
    "surface.ivgtyp",     # Noah vegetation category (excluded)
    "surface.kpbl",       # YSU/MYNN PBL top level
    "surface.ktop_plume",  # MYNN EDMF plume top
    "surface.pgsxy",      # Noah-MP crop growth stage
})

#: Integer descriptors that still have NO device policy, and therefore still
#: refuse at ``require_healthy`` before step one.  Both belong to Noah-MP
#: (``sf_surface_physics=4``), so **Noah-MP cannot launch through the
#: supervised run path on any domain** -- the same defect MYNN had, found by
#: this sweep rather than by a forecast.
#:
#: **Empty, and it has to stay a measurement rather than a waiver.**
#:
#: It previously held ``surface.isnowxy`` and ``surface.pgsxy``, on the
#: grounds that the reference bundle's ``phys/noahmp/`` directory is empty --
#: the submodule was never vendored -- so there was no pinned source in that
#: checkout to bound them against.  The premise was too narrow: the bundle is
#: broken, but the pinned stock-WRF v4.6.1 gate tree (the one every
#: ``tools/noahmp_wrf461_oracle/build_*.sh`` defaults to) is at the same commit
#: ``d66e442`` with ``phys/noahmp`` initialized at ``848f54ad``, and it is the
#: tree every ``tools/noahmp_wrf461_oracle/build_*.sh`` already defaults to.
#: Both bounds are now read off that source; see
#: ``woof.core.health._GPU_RANGE_CHECKED_INT32``.
_UNCLASSIFIED_AT_THIS_CHECKOUT: frozenset[str] = frozenset()


@pytest.mark.parametrize("name", ["surface.cupflag", "surface.cu_act_flag"])
def test_cup_logical_integer_policy(name):
    from woof.core.health import field_from_array
    assert gpu_integer_policy(name, np.int32) is not None
    assert validate_fields_cpu([field_from_array(name, np.array([0, 1], np.int32))]).ok
    assert not validate_fields_cpu([field_from_array(name, np.array([2], np.int32))]).ok


def _config(**overrides):
    """One host-sized RunConfig, validated by the production validator."""
    from woof.config import RunConfig, validate_run_config, \
        validated_soil_layer_count

    base = dict(
        nx=_NX, ny=_NY, nz=_NZ, dx=2000.0, dy=2000.0, ztop=8000.0,
        dt=10.0, run_seconds=0.0, time_step_sound=4, moist=True,
        mp_physics=10, ra_physics=4, cu_physics=1,
        sf_sfclay_physics=1, sf_surface_physics=2, bl_pbl_physics=1)
    base.update(overrides)
    base.setdefault(
        "num_soil_layers",
        validated_soil_layer_count(base["sf_surface_physics"]))
    return validate_run_config(RunConfig(**base))


def _host_cupy_stub():
    """A NumPy-backed stand-in for ``cupy``, for constructors only.

    Same shape as ``tools/health_field_census.install_host_array_backend``'s
    stub: attribute access delegates to NumPy, ``asnumpy`` is a no-op, and
    every device entry point refuses.  A CuPy-only symbol reaching it means a
    constructor started computing rather than allocating, which is a fact
    worth a loud failure.
    """
    import types

    stub = types.ModuleType("cupy")
    stub.__gpuwm_host_backend__ = True

    def _delegate(name):
        try:
            return getattr(np, name)
        except AttributeError as exc:
            raise AttributeError(
                f"host array backend has no {name!r}; this test allocates "
                "state, it does not execute kernels") from exc

    def _no_device(*args, **kwargs):
        raise RuntimeError("host array backend: no CUDA device is opened")

    stub.__getattr__ = _delegate
    stub.asnumpy = lambda value, **kwargs: np.asarray(value)
    runtime = types.ModuleType("cupy.cuda.runtime")
    runtime.getDeviceCount = _no_device
    runtime.memGetInfo = _no_device
    runtime.deviceSynchronize = _no_device
    cuda = types.ModuleType("cupy.cuda")
    cuda.runtime = runtime
    stub.cuda = cuda
    return stub, cuda, runtime


def _bind_host_arrays(monkeypatch):
    """Bind every route into CuPy to NumPy for the duration of one test.

    Two bindings are needed, and neither alone is sufficient.  Modules that
    did ``import cupy as cp`` at import time hold their own reference, so
    each woof module's ``cp`` attribute is rebound -- swept out of
    ``sys.modules`` rather than listed, so a scheme added later is bound too.
    But ``woof.core.ruc_runtime.ruc_cold_start`` imports CuPy *inside the
    function*, which no attribute patch can reach, so ``sys.modules['cupy']``
    is rebound as well.

    A third route appeared when 431843507 made ``cold_start_on_device`` the
    default Noah-MP cold-start arm: it compiles the real CUDA driver kernel
    through ``cp.RawModule``, which no array rebinding can host.  The
    production escape hatch ``WOOF_NOAHMP_HOST_COLD_START=1`` selects the
    paired numpy authority instead -- byte-for-byte equal to the device arm
    per ``tests/test_noahmp_cold_start_device.py`` -- so the inventory here
    is still the one a forecast builds.

    ``monkeypatch`` undoes all of it, so the ``gpu``-marked test later in the
    same session still gets the real CuPy.

    The sweep picks woof modules by their ``sys.modules`` KEY and touches
    no other module.  ``cupy.testing`` is an ``importlib.util.LazyLoader``
    module, and any attribute read on one (``__name__`` included) runs its
    import; run while the stub stood in for ``cupy``, that bound
    ``cupy.testing._array`` to the stub for the rest of the process, and
    the next file in the same worker to call ``cp.testing`` failed inside
    ``asnumpy`` (tests/test_ruc_runtime.py, whenever xdist scheduled it
    after this file).
    """
    import sys

    import woof.core.physics  # noqa: F401  (imports every runtime module)
    from woof.core.noahmp_runtime import HOST_COLD_START_ENV

    monkeypatch.setenv(HOST_COLD_START_ENV, "1")
    stub, cuda, runtime = _host_cupy_stub()
    monkeypatch.setitem(sys.modules, "cupy", stub)
    monkeypatch.setitem(sys.modules, "cupy.cuda", cuda)
    monkeypatch.setitem(sys.modules, "cupy.cuda.runtime", runtime)
    for name, module in list(sys.modules.items()):
        if not name.startswith("woof.") or module is None:
            continue
        if getattr(module, "cp", None) is not None:
            monkeypatch.setattr(module, "cp", np, raising=False)
        if getattr(module, "DTYPE", None) is not None:
            monkeypatch.setattr(module, "DTYPE", np.float32, raising=False)


def _host_inventory(monkeypatch, cfg):
    """The real ``collect_state_fields`` inventory, host-backed.

    Production constructors throughout: ``DomainState`` and
    ``initialize_physics`` are the same objects a forecast builds, with their
    array module bound to NumPy so no device is touched.
    """
    import woof.core.physics as physics_mod
    import woof.core.state as state_mod

    _bind_host_arrays(monkeypatch)

    from datetime import datetime, timezone

    grid = np.zeros((cfg.ny, cfg.nx), dtype=np.float64)
    start = datetime(1974, 4, 3, 12, tzinfo=timezone.utc)
    state = state_mod.DomainState(cfg)
    px_inputs = {}
    if int(cfg.sf_surface_physics) == 7:
        from woof.core import pxlsm_runtime, kernels
        # This instrument counts fields, so compiling a step kernel is outside
        # its scope. The production PX initializer still allocates every field.
        monkeypatch.setattr(pxlsm_runtime, "prepare_pxlsm", lambda nlcat: None)
        original_kernel = kernels.get_kernel
        def compile_only(module, symbol, *args, **kwargs):
            if (module, symbol) != ("pxlsm_driver", "pxlsm_post"):
                return original_kernel(module, symbol, *args, **kwargs)
            def refuse_launch(*args, **kwargs):
                raise AssertionError("host inventory cannot launch a PX kernel")
            return refuse_launch
        monkeypatch.setattr(kernels, "get_kernel", compile_only)
        landusef = np.zeros((24, cfg.ny, cfg.nx), np.float32)
        soilcbot = np.zeros((16, cfg.ny, cfg.nx), np.float32)
        landusef[9] = 1.0
        soilcbot[5] = 1.0
        px_inputs = dict(landusef=landusef, soilcbot=soilcbot)
    physics_mod.initialize_physics(
        state, cfg,
        # The bundle belongs to sf_surface_physics=2 only; initialize_physics
        # refuses it for any other LSM and loads its own for Noah.
        noah_params=object() if int(cfg.sf_surface_physics) == 2 else None,
        landmask=1.0, tsk=290.0, soil_temperature=285.0,
        thermal_inertia=0.04 if int(cfg.sf_surface_physics) == 1 else None,
        soil_moisture=0.30, ivgtyp=10, isltyp=6, **px_inputs,
        # PX cold start only writes existing arrays; its device arithmetic
        # is checked by the PX owner tests rather than this field census.
        lsm_cold_start=int(cfg.sf_surface_physics) != 7,
        radiation=lambda **kwargs: None,
        cumulus=lambda **kwargs: None,
        radiation_start_time=start,
        radiation_latitude=grid + 40.0,
        radiation_longitude=grid - 84.0,
        # Noah-MP refuses to construct without these; a silent zero latitude
        # at day zero would run every column on the equator at New Year.
        noahmp_start_time=start,
        noahmp_latitude=grid + 40.0,
        noahmp_longitude=grid - 84.0)
    state._scratch["lbc_forcing_tables"] = np.zeros((5,), np.float32)
    state._scratch["lbc_weights_0"] = np.zeros((2,), np.float32)
    fields = collect_state_fields(state, backend="gpu")
    # The safety argument of this module is that nothing here allocates on a
    # device; assert the positive rather than trusting the monkeypatch.
    assert type(state.p) is np.ndarray
    return fields


def _mynn_config():
    """The MYNN 5/5 selectors, everything else at the reference values."""
    return _config(sf_sfclay_physics=5, bl_pbl_physics=5)


def _noahmp_config():
    """The Noah-MP selector under its original MM5/YSU profile."""
    return _config(sf_surface_physics=4, sf_sfclay_physics=1,
                   bl_pbl_physics=1)


def _routed_selections():
    """Every (sfclay, lsm, pbl) triple the production validator accepts.

    Probed rather than transcribed: ``PHYSICS_SLOT_DISPATCH`` is what the
    driver dispatches on, so a scheme joins this sweep by being routed, and
    ``validate_run_config`` then rejects the combinations a user cannot pick.
    """
    from woof.core.physics import PHYSICS_SLOT_DISPATCH

    selections = []
    for sfclay in sorted(PHYSICS_SLOT_DISPATCH["sf_sfclay_physics"]):
        for lsm in sorted(PHYSICS_SLOT_DISPATCH["sf_surface_physics"]):
            for pbl in sorted(PHYSICS_SLOT_DISPATCH["bl_pbl_physics"]):
                try:
                    cfg = _config(sf_sfclay_physics=sfclay,
                                  sf_surface_physics=lsm,
                                  bl_pbl_physics=pbl)
                except Exception:
                    continue
                selections.append(cfg)
    assert selections, "no routed physics combination validated"
    return selections


# ---------------------------------------------------------------------------
# The gate itself: it must be able to refuse, or it is evidence of nothing.
# ---------------------------------------------------------------------------

def test_an_int32_field_with_no_policy_is_refused():
    """Negative control on the classifier, independent of any scheme."""
    with pytest.raises(TypeError) as excinfo:
        gpu_integer_policy("surface.some_future_counter", np.int32)
    assert "'surface.some_future_counter'" in str(excinfo.value)
    assert _REFUSAL in str(excinfo.value)


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.int16, np.uint32])
def test_a_dtype_outside_the_two_supported_storages_is_refused(dtype):
    with pytest.raises(TypeError):
        gpu_integer_policy("surface.kpbl", dtype)


def test_the_two_policy_kinds_classify_differently():
    from woof.core.health import _INT32_STORAGE

    assert gpu_integer_policy("surface.tsk", np.float32) == 0
    assert gpu_integer_policy("surface.kpbl", np.int32) == _INT32_STORAGE
    assert gpu_integer_policy(
        "surface.ktop_plume", np.int32) == _INT32_STORAGE
    # An exclusion is skipped, not checked; None is not 0.
    for name in sorted(GPU_INTEGER_EXCLUSIONS):
        assert gpu_integer_policy(name, np.int32) is None


def test_binding_host_arrays_loads_no_lazily_imported_module(
        monkeypatch, tmp_path):
    """The host binding reaches woof modules only; a lazy module stays unloaded.

    A LazyLoader module (``cupy.testing`` is one) runs its import on the first
    attribute read.  If the binding's sweep reads one while the stub stands in
    for ``cupy``, the module binds the stub for good and a later
    ``cp.testing`` call in the same process fails.  A lazy stand-in is
    registered here and must come out of the binding still unloaded.
    """
    import importlib.util
    import sys
    import types

    probe = tmp_path / "gpuwm_lazy_probe_module.py"
    probe.write_text("LOADED = True\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "gpuwm_lazy_probe_module", probe)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "gpuwm_lazy_probe_module", module)
    lazy = type(module)
    assert lazy is not types.ModuleType
    with monkeypatch.context() as patch:
        _bind_host_arrays(patch)
    # type() reads the object's class without an attribute lookup, so it
    # does not trigger the load it is checking for.
    assert type(module) is lazy, "the binding loaded a lazy module"


def test_removing_the_plume_top_policy_restores_the_launch_failure(
        monkeypatch):
    """The falsification: this is the failure the MYNN runs actually hit.

    Withdrawing exactly one name from the policy table must put the real MYNN
    inventory back into the refusal, with the same message a user saw at
    ``require_healthy``.  A gate that has never been observed to fail proves
    nothing, so this keeps the observation live on every run.
    """
    import woof.core.health as health_mod

    fields = _host_inventory(monkeypatch, _mynn_config())
    monkeypatch.setattr(
        health_mod, "_GPU_RANGE_CHECKED_INT32",
        health_mod._GPU_RANGE_CHECKED_INT32 - {"surface.ktop_plume"})
    with pytest.raises(TypeError) as excinfo:
        for field in fields:
            health_mod.gpu_integer_policy(field.name, field.values.dtype)
    assert "'surface.ktop_plume'" in str(excinfo.value)
    assert _REFUSAL in str(excinfo.value)


# ---------------------------------------------------------------------------
# The inventory, for every scheme a user can route.
# ---------------------------------------------------------------------------

def test_the_mynn_inventory_really_does_publish_an_int32_plume_top(
        monkeypatch):
    """Pin the fact the policy exists for; do not infer it from the policy."""
    fields = _host_inventory(monkeypatch, _mynn_config())
    by_name = {field.name: field for field in fields}
    assert "surface.ktop_plume" in by_name
    assert np.dtype(by_name["surface.ktop_plume"].values.dtype) == np.dtype(
        np.int32)
    assert by_name["surface.ktop_plume"].values.shape == (_NY, _NX)


def _sweep_routed_inventories(monkeypatch):
    """Classify every routed combination; return what was seen and refused.

    Collects instead of stopping at the first refusal, because "which schemes
    cannot launch" is the measurement worth having, and a sweep that aborted
    on the first would have reported one blocker where there are two.
    """
    seen_int32: set[str] = set()
    refused: dict[str, set[tuple[int, int, int]]] = {}
    combinations = 0
    for cfg in _routed_selections():
        with monkeypatch.context() as patch:
            fields = _host_inventory(patch, cfg)
        combinations += 1
        selector = (int(cfg.sf_sfclay_physics), int(cfg.sf_surface_physics),
                    int(cfg.bl_pbl_physics))
        for field in fields:
            if np.dtype(field.values.dtype) == np.dtype(np.int32):
                seen_int32.add(field.name)
            try:
                gpu_integer_policy(field.name, field.values.dtype)
            except TypeError:
                refused.setdefault(field.name, set()).add(selector)
    return combinations, seen_int32, refused


def test_the_routed_cross_product_is_the_size_it_was_measured_at(monkeypatch):
    """Guard the sweep itself: a shrunken matrix proves less than it looks."""
    combinations, seen_int32, _ = _sweep_routed_inventories(monkeypatch)
    assert combinations == _ROUTED_COMBINATIONS
    assert seen_int32 == _INT32_DESCRIPTORS


def test_every_routed_combination_classifies_or_is_a_recorded_blocker(
        monkeypatch):
    """The forward-looking gate: a NEW int32 field with no policy fails here.

    This is the test that would have caught ``ktop_plume`` when MYNN was
    admitted, months before a forecast tried to start.  The recorded set is a
    measurement of what is still broken at this checkout, not a waiver: a
    field added to it needs the same argument ``ktop_plume`` got.
    """
    _, _, refused = _sweep_routed_inventories(monkeypatch)
    assert set(refused) == _UNCLASSIFIED_AT_THIS_CHECKOUT
    for name, selectors in refused.items():
        assert {lsm for _, lsm, _ in selectors} == {4}, (name, selectors)


def test_the_mynn_five_five_suite_is_no_longer_a_blocker(monkeypatch):
    """The specific claim this change makes, stated on its own."""
    _, _, refused = _sweep_routed_inventories(monkeypatch)
    assert "surface.ktop_plume" not in refused
    for cfg in _routed_selections():
        if int(cfg.bl_pbl_physics) != 5:
            continue
        with monkeypatch.context() as patch:
            fields = _host_inventory(patch, cfg)
        for field in fields:
            gpu_integer_policy(field.name, field.values.dtype)


# ---------------------------------------------------------------------------
# The bound, against WRF.
# ---------------------------------------------------------------------------

def test_the_plume_top_cap_is_the_wrf_clamp_not_the_kpbl_cap(monkeypatch):
    """``module_bl_mynn.F:6392`` is ``MIN(ktop,KTE-1)``, one below kpbl."""
    fields = _host_inventory(monkeypatch, _mynn_config())
    rules = {field.name: field.rule for field in fields}
    assert rules["surface.ktop_plume"] == FieldRule(
        "surface", 0.0, float(_NZ - 1))
    # kpbl is the adjacent precedent and must NOT have moved.
    assert rules["surface.kpbl"] == FieldRule("surface", 0.0, float(_NZ))


def test_the_cap_admits_the_top_level_the_port_can_write(monkeypatch):
    """nz-1 must pass and nz must fail, on the real descriptor.

    A cap that is one too tight fires spuriously on a real domain the first
    time a plume reaches the model top; a cap that is one too loose is not
    the clamp WRF applies.  Both directions are checked against the real
    collected field rather than a hand-built one.
    """
    fields = _host_inventory(monkeypatch, _mynn_config())
    field = next(f for f in fields if f.name == "surface.ktop_plume")

    for legal in (0, 1, _NZ - 1):
        field.values[...] = np.int32(legal)
        assert validate_fields_cpu([field]).ok, legal

    for illegal in (_NZ, _NZ + 1):
        field.values[...] = np.int32(illegal)
        report = validate_fields_cpu([field])
        assert not report.ok
        assert report.first_bad_field == "surface.ktop_plume"

    field.values[...] = np.int32(0)
    field.values.reshape(-1)[0] = np.int32(-1)
    report = validate_fields_cpu([field])
    assert not report.ok
    assert report.first_bad_flat_index == 0


def test_a_state_with_no_pressure_array_admits_only_the_unwritten_zero():
    """nz == 0 degenerate: the cap must not invert into "nothing is legal"."""
    from types import SimpleNamespace

    driver = SimpleNamespace(
        pbl_tendencies=None, radiation_tendencies=None,
        cumulus_tendencies=None,
        fields={"ktop_plume": np.zeros((2, 3), dtype=np.int32)},
        microphysics=None, rthratenlw=None, rthratensw=None,
        cu_nca=None, cu_pratec=None, cu_raincv=None, cu_rates=None,
        _pending_rainbl=None)
    state = SimpleNamespace(
        physics=driver, lateral_boundaries=None,
        _lateral_boundary_device=None, _scratch={})
    fields = collect_state_fields(state, backend="cpu")
    field = next(f for f in fields if f.name == "surface.ktop_plume")
    assert field.rule == FieldRule("surface", 0.0, 0.0)
    assert validate_fields_cpu([field]).ok


# ---------------------------------------------------------------------------
# The device twin.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_the_real_mynn_descriptor_build_reaches_the_device():
    """The controller twin: ``_refresh`` on a real MYNN ``DomainState``.

    This is the call that raised.  It is asserted on its own, without a
    ``validate()`` verdict, because a freshly constructed ``DomainState`` is
    all zeros and a zero theta legitimately fails the gate -- descriptor
    CONSTRUCTION is what MYNN broke, and conflating the two would let a
    descriptor regression hide behind an unphysical state.
    """
    import cupy as cp
    import woof.core.physics as physics_mod
    from woof.core.health import StateHealthValidator
    from woof.core.state import DomainState

    cfg = _mynn_config()
    state = DomainState(cfg)
    physics_mod.initialize_physics(
        state, cfg, noah_params=object(), radiation=lambda **kwargs: None,
        cumulus=lambda **kwargs: None)
    state._scratch["lbc_forcing_tables"] = cp.zeros((5,), cp.float32)
    state._scratch["lbc_weights_0"] = cp.zeros((2,), cp.float32)

    validator = StateHealthValidator(state)
    validator._refresh()
    checked = {field.name: field for field in validator.fields}
    assert "surface.ktop_plume" in checked
    assert checked["surface.ktop_plume"].values.dtype == cp.int32
    assert checked["surface.ktop_plume"].rule == FieldRule(
        "surface", 0.0, float(cfg.nz - 1))
    assert {field.name for field in validator.excluded_integer_fields} == {
        "surface.ivgtyp", "surface.isltyp"}


class _PlumeTopOnlyState:
    """A device state carrying only ``p`` and the plume top.

    Isolating the descriptor is the point: on a full state the first failing
    field is whichever comes first in the inventory, so a bound that never
    fired would be indistinguishable from one that did.
    """

    def __init__(self, cp, nz, plume_top):
        from types import SimpleNamespace

        self._cp = cp
        self._scratch = {}
        self.p = cp.full((nz, 2, 3), cp.float32(1.0e5), dtype=cp.float32)
        self.physics = SimpleNamespace(
            pbl_tendencies=None, radiation_tendencies=None,
            cumulus_tendencies=None,
            fields={"ktop_plume": cp.full((2, 3), plume_top, dtype=cp.int32)},
            microphysics=None, rthratenlw=None, rthratensw=None,
            cu_nca=None, cu_pratec=None, cu_raincv=None, cu_rates=None,
            _pending_rainbl=None)
        self.lateral_boundaries = None
        self._lateral_boundary_device = None

    def scratch(self, shape, slot):
        if slot not in self._scratch:
            self._scratch[slot] = self._cp.zeros(shape, dtype=self._cp.float32)
        return self._scratch[slot]


@pytest.mark.gpu
@pytest.mark.parametrize("plume_top,expected_ok", [
    (0, True),            # no plume / pre-first-call (module_bl_mynn.F:646)
    (1, True),
    (_NZ - 1, True),      # the clamp itself (module_bl_mynn.F:6392)
    (_NZ, False),         # one above what any writer can produce
    (-1, False),
])
def test_the_device_gate_reads_int32_storage_at_the_clamp(plume_top,
                                                          expected_ok):
    """The CUDA kernel, not the NumPy mirror, must honour the bound.

    ``_INT32_STORAGE`` makes the kernel reinterpret the payload as an integer
    rather than as float32 bits; without it, ``nz`` and ``nz - 1`` would both
    read as denormal garbage and the gate would be decorative.
    """
    import cupy as cp
    from woof.core.health import StateHealthValidator

    state = _PlumeTopOnlyState(cp, _NZ, plume_top)
    validator = StateHealthValidator(state)
    report = validator.validate(phase=f"plume-top-{plume_top}")
    assert report.ok is expected_ok, report
    if not expected_ok:
        assert report.first_bad_field == "surface.ktop_plume"
        assert report.first_bad_value == float(plume_top)


# ---------------------------------------------------------------------------
# Noah-MP: the other two blockers this sweep found, and their bounds.
# ---------------------------------------------------------------------------

def test_the_noahmp_suite_is_no_longer_a_blocker(monkeypatch):
    """The specific claim the Noah-MP half of this change makes.

    ``sf_surface_physics=4`` published two unclassified int32 descriptors and
    therefore refused at ``require_healthy(phase="initialized-or-restored")``
    on every domain, exactly as MYNN did.  Every field of every Noah-MP
    inventory must now classify.
    """
    _, _, refused = _sweep_routed_inventories(monkeypatch)
    assert "surface.isnowxy" not in refused
    assert "surface.pgsxy" not in refused
    for cfg in _routed_selections():
        if int(cfg.sf_surface_physics) != 4:
            continue
        with monkeypatch.context() as patch:
            fields = _host_inventory(patch, cfg)
        for field in fields:
            gpu_integer_policy(field.name, field.values.dtype)


@pytest.mark.parametrize("name", ["surface.isnowxy", "surface.pgsxy"])
def test_removing_a_noahmp_policy_restores_the_launch_failure(name,
                                                              monkeypatch):
    """The falsification, one name at a time.

    This is the failure a Noah-MP run actually hit.  Withdrawing exactly one
    name must put the real Noah-MP inventory back into the refusal with the
    production message; a gate never observed failing proves nothing.
    """
    import woof.core.health as health_mod

    fields = _host_inventory(monkeypatch, _noahmp_config())
    monkeypatch.setattr(
        health_mod, "_GPU_RANGE_CHECKED_INT32",
        health_mod._GPU_RANGE_CHECKED_INT32 - {name})
    with pytest.raises(TypeError) as excinfo:
        for field in fields:
            health_mod.gpu_integer_policy(field.name, field.values.dtype)
    assert f"'{name}'" in str(excinfo.value)
    assert _REFUSAL in str(excinfo.value)


def test_the_noahmp_inventory_really_does_publish_both_int32_fields(
        monkeypatch):
    """Pin the facts the policies exist for; do not infer them from them."""
    fields = _host_inventory(monkeypatch, _noahmp_config())
    by_name = {field.name: field for field in fields}
    for name in ("surface.isnowxy", "surface.pgsxy"):
        assert name in by_name, name
        assert np.dtype(by_name[name].values.dtype) == np.dtype(np.int32)
        assert by_name[name].values.shape == (_NY, _NX)


def test_the_snow_index_bound_is_negative_and_matches_the_port(monkeypatch):
    """``ISNOWXY`` counts ACTIVE snow layers downward, so it is <= 0.

    A ``>= 0`` bound would have been the obvious guess and would fire on the
    first snowpack column of a real forecast.  ``-NSNOW`` is not guessed
    either: it is the same constant the column runner pins, and the two are
    asserted equal here so they cannot drift.
    """
    from woof.core.health import NOAHMP_SNOW_LAYERS
    from woof.core.noahmp_runtime import NSNOW

    assert NOAHMP_SNOW_LAYERS == NSNOW

    fields = _host_inventory(monkeypatch, _noahmp_config())
    rules = {field.name: field.rule for field in fields}
    assert rules["surface.isnowxy"] == FieldRule(
        "surface", float(-NSNOW), 0.0)
    assert rules["surface.pgsxy"] == FieldRule("surface", 0.0, 8.0)


def test_the_snow_index_bound_admits_every_value_a_writer_can_produce(
        monkeypatch):
    """0, -1, -2 and -3 must pass; -4 and +1 must fail.

    Checked on the real collected descriptor, in both directions.  A bound one
    too tight fires spuriously the first time a column builds a full three-layer
    pack; a bound one too loose is not what any writer can produce.
    """
    from woof.core.noahmp_runtime import NSNOW

    fields = _host_inventory(monkeypatch, _noahmp_config())
    field = next(f for f in fields if f.name == "surface.isnowxy")

    for legal in range(-NSNOW, 1):
        field.values[...] = np.int32(legal)
        assert validate_fields_cpu([field]).ok, legal

    for illegal in (-NSNOW - 1, 1):
        field.values[...] = np.int32(illegal)
        report = validate_fields_cpu([field])
        assert not report.ok, illegal
        assert report.first_bad_field == "surface.isnowxy"
        assert report.first_bad_value == float(illegal)

    field.values[...] = np.int32(0)
    field.values.reshape(-1)[0] = np.int32(-NSNOW - 1)
    report = validate_fields_cpu([field])
    assert not report.ok
    assert report.first_bad_flat_index == 0


def test_the_growth_stage_bound_admits_every_stage_growing_gdd_assigns(
        monkeypatch):
    """0 (the cold value, and the only one under opt_crop=0) through 8."""
    fields = _host_inventory(monkeypatch, _noahmp_config())
    field = next(f for f in fields if f.name == "surface.pgsxy")

    for legal in range(0, 9):
        field.values[...] = np.int32(legal)
        assert validate_fields_cpu([field]).ok, legal

    for illegal in (-1, 9):
        field.values[...] = np.int32(illegal)
        report = validate_fields_cpu([field])
        assert not report.ok, illegal
        assert report.first_bad_field == "surface.pgsxy"


def test_the_snow_index_writers_stay_inside_the_bound_in_this_port(
        monkeypatch):
    """Every ISNOW a Noah-MP forecast actually writes, over six steps.

    The bound is read off WRF; this is the independent check that woof's own
    port stays inside it on a domain that builds and melts a pack, rather than
    a bound that is right about WRF and wrong about the code that runs.
    """
    import os
    import sys

    pytest.importorskip("cupy")
    if os.environ.get("GPUWM_NO_LOCAL_GPU") == "1":
        pytest.skip("GPU disabled for this process")

    import cupy as cp

    tests_dir = os.path.dirname(os.path.abspath(__file__))
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from test_noahmp_runtime import _build
    from woof.core.dycore import step
    from woof.core.noahmp_runtime import NSNOW

    seen = set()
    for kwargs in ({"nx": 6, "ny": 4, "water_columns": 1},
                   {"nx": 6, "ny": 4, "water_columns": 1,
                    "snow_mm": 45.0, "snow_depth_m": 0.16}):
        state, cfg, driver = _build(**kwargs)
        for _ in range(6):
            step(state, cfg)
            seen.update(
                int(v) for v in np.unique(cp.asnumpy(driver.fields["isnowxy"])))
    assert seen, "no ISNOWXY values were observed"
    assert min(seen) >= -NSNOW and max(seen) <= 0, sorted(seen)
    # The snowpack domain has to have produced a real snow layer, or this
    # observed nothing the bound is about.
    assert min(seen) < 0, sorted(seen)


@pytest.mark.gpu
def test_the_real_noahmp_descriptor_build_reaches_the_device():
    """The controller twin for Noah-MP: ``_refresh`` on a real device state.

    This is the call that raised.  As with the MYNN twin, the assertion is on
    descriptor CONSTRUCTION rather than on a ``validate()`` verdict, so a
    descriptor regression cannot hide behind an unphysical state -- but this
    state is a real cold-started Noah-MP domain rather than a zeroed one, so
    the verdict is asserted too.
    """
    import os
    import sys

    if os.environ.get("GPUWM_NO_LOCAL_GPU") == "1":
        pytest.skip("GPU disabled for this process")

    import cupy as cp

    tests_dir = os.path.dirname(os.path.abspath(__file__))
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from test_noahmp_runtime import _build

    from woof.core.health import StateHealthValidator
    from woof.core.noahmp_runtime import NSNOW

    state, cfg, driver = _build(nx=6, ny=4, water_columns=1,
                                snow_mm=45.0, snow_depth_m=0.16)
    validator = StateHealthValidator(state)
    validator._refresh()
    checked = {field.name: field for field in validator.fields}

    for name, rule in (
        ("surface.isnowxy", FieldRule("surface", float(-NSNOW), 0.0)),
        ("surface.pgsxy", FieldRule("surface", 0.0, 8.0)),
    ):
        assert name in checked, name
        assert checked[name].values.dtype == cp.int32, name
        assert checked[name].rule == rule, name

    assert {field.name for field in validator.excluded_integer_fields} == {
        "surface.ivgtyp", "surface.isltyp"}

    report = validator.validate(phase="noahmp-initialized-or-restored")
    assert report.ok, report


def test_the_ebal_range_admits_every_case_the_noah_kernel_writes():
    """Current SNOPAC codes and historical glacial code 4 remain admitted.

    The full glacial subtree now reports its SNOPAC branch (0 through 3).
    Older checkpoints can still carry the earlier glacial marker 4.
    """
    import re
    from pathlib import Path

    from woof.core.health import rule_for_field

    source = (Path(__file__).resolve().parents[1]
              / "woof" / "core" / "kernels" / "noah.cu").read_text()
    codes = {int(v) for v in re.findall(
        r"\bebal_(?:case|a\[idx\])\s*=\s*(\d+)\s*;", source)}
    codes |= {int(v) for pair in re.findall(
        r"ebal_case\s*=\s*\w+\s*\?\s*(\d+)\s*:\s*(\d+)", source) for v in pair}
    # The current glacial subtree reports its SNOPAC branch (1, 2 or 3).
    # Retain health admission of historical glacial code 4 for old checkpoints.
    assert codes == {0, 1, 2, 3}, codes
    rule = rule_for_field("surface.ebal")
    assert (rule.lower, rule.upper) == (min(codes), max(codes | {4}))
@pytest.mark.parametrize("name",["surface.ncfrst","surface.ncfrcv"])
def test_gfdl_cloud_sample_counts_are_checked_and_cannot_be_negative(name):
    from woof.core.health import _INT32_STORAGE,field_from_array
    assert gpu_integer_policy(name,np.int32)==_INT32_STORAGE
    assert validate_fields_cpu([field_from_array(name,np.array([0,1,2147483647],np.int32))]).ok
    bad=validate_fields_cpu([field_from_array(name,np.array([0,-1],np.int32))])
    assert not bad.ok and bad.first_bad_field==name and bad.first_bad_index==(1,)
@pytest.mark.parametrize("name", ("surface.irnumsi", "surface.irnummi", "surface.irnumfi"))
def test_noahmp_irrigation_counter_health_policy(name):
    from woof.core.health import gpu_integer_policy, rule_for_field, FieldRule
    assert gpu_integer_policy(name, np.int32) is not None
    assert rule_for_field(name) == FieldRule("surface", 0.0)
