"""Process body for tests/test_physics_inventory_census.py.

``python tests/physics_allocation_census.py`` builds the real
:class:`woof.core.physics.PhysicsDriver`, through ``initialize_physics``,
for every admitted cell of :func:`admitted_cells` on a NumPy stand-in for
cupy, and prints one JSON object: per cell, every array the construction
left on the driver (path, shape, bytes) beside
``preflight.physics_array_shapes`` for the same configuration.

Why a child interpreter: ``woof.core.physics`` binds ``cupy`` when it is
first imported, so the stand-in has to be installed before any woof
import, which a pytest process that may already hold the real module
cannot promise.  The child also never imports the real cupy, so no device
is opened on a machine that has one.

What the stand-in counts.  It is NumPy with every array a ``cp.*`` call
returns viewed as :class:`_OnCard`, and NumPy keeps that subclass through
methods and arithmetic, so an array is "on the card" exactly when a cupy
call, or an operation on an array a cupy call made, produced it.  Arrays
the code builds with NumPy stay plain ndarrays (host tables, lat/lon
inputs) and are not counted.  ``cp.asarray`` of a host array copies, as the
real upload does.  Kernel launches are no-ops: a launch writes into arrays
that already exist, so skipping it changes no allocation, and it is what
lets Noah-MP's cold start run here.  Arrays reachable from the
``DomainState`` (its fields and its scratch slots) are left out: the
state inventory and the scratch registry price those, not this one.
"""

from __future__ import annotations

import json
import math
import sys
import types

import numpy as np

#: Small grid: the census is about names and shapes, and every inventory
#: entry is a product of these three numbers.
GRID = {"nx": 6, "ny": 5, "nz": 12}


def admitted_cells() -> list[tuple[int, int, int, float]]:
    """Every (surface layer, PBL, land surface, bldt) the config door admits.

    Both PBL cadences are swept because the inventory branches on them
    (``bldt == 0`` reuses the PBL stack as the composed target, a positive
    cadence holds raw rates).  Radiation, cumulus and microphysics stay off:
    their inventories are not what this census is about, and a radiation
    adapter needs packaged tables the grid is too small to exercise.
    """
    from woof.config import (LAND_SURFACE_SCHEMES, PBL_SCHEMES,
                              SURFACE_LAYER_SCHEMES)

    cells = []
    for sfclay in SURFACE_LAYER_SCHEMES:
        for pbl in PBL_SCHEMES:
            for lsm in LAND_SURFACE_SCHEMES:
                for bldt in (0.0, 5.0):
                    cell = (int(sfclay), int(pbl), int(lsm), bldt)
                    if cell_config(cell) is not None:
                        cells.append(cell)
    return cells


def cell_id(cell: tuple[int, int, int, float]) -> str:
    sfclay, pbl, lsm, bldt = cell
    return f"sfclay{sfclay}-pbl{pbl}-lsm{lsm}-bldt{bldt:g}"


def cell_config(cell: tuple[int, int, int, float]):
    """The validated RunConfig for one cell, or None when it is refused."""
    from woof.config import (SASE_PBL_SCHEME, RunConfig,
                              validate_run_config,
                              validated_soil_layer_count)
    from woof.core.physics_inventory import physics_driver_required

    sfclay, pbl, lsm, bldt = cell
    cfg = RunConfig(
        **GRID, dx=3000.0, dy=3000.0, ztop=12000.0, dt=12.0,
        run_seconds=60.0, time_step_sound=4, moist=True, mp_physics=6 if pbl == 12 else 0,
        sf_sfclay_physics=sfclay, bl_pbl_physics=pbl,
        sf_surface_physics=lsm,
        num_soil_layers=validated_soil_layer_count(lsm), bldt=bldt,
        # SASE supplies the horizontal mixing km_opt would apply and is
        # refused beside it; every other PBL runs the default operator.
        km_opt=0 if pbl == SASE_PBL_SCHEME else 1)
    from woof.physics_registry import column_surface_contract
    contract = column_surface_contract(lsm)
    if contract is not None:
        # Price a valid column coupling, preserving the swept PBL/layer.
        from dataclasses import replace
        cfg = replace(cfg, ra_lw_physics=1, ra_sw_physics=1,
                      ra_rrtmg_variant="rrtmg_legacy", fractional_seaice=1)
    try:
        cfg = validate_run_config(cfg)
    except ValueError:
        return None
    return cfg if physics_driver_required(cfg) else None


class _OnCard(np.ndarray):
    """An array the stand-in put on the card."""

    def get(self, *args, **kwargs):
        """cupy's device-to-host copy."""
        return np.array(self).view(np.ndarray)


class _OnCardRow(_OnCard):
    """A 1-D card array whose elements index as 0-d card arrays, as cupy's do
    (``float(ds[1].get())`` reads the solar constant that way)."""

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(value, np.generic):
            return np.asarray(value).view(_OnCard)
        return value


def _on_card(value):
    if isinstance(value, np.ndarray) and not isinstance(value, _OnCard):
        return value.view(_OnCard)
    return value


def _returns_on_card(fn):
    def call(*args, **kwargs):
        return _on_card(fn(*args, **kwargs))
    call.__name__ = getattr(fn, "__name__", "call")
    return call


def _uploads(fn):
    def call(value, *args, **kwargs):
        if isinstance(value, _OnCard):
            return _on_card(fn(value, *args, **kwargs))
        return _on_card(np.array(fn(value, *args, **kwargs), copy=True))
    return call


class _Launch:
    def __call__(self, *args, **kwargs):
        return None


class _RawModule:
    def __init__(self, *args, **kwargs):
        pass

    def compile(self, *args, **kwargs):
        return None

    def get_function(self, name):
        return _Launch()


def install_numpy_cupy() -> None:
    """Make ``import cupy`` resolve to the counting NumPy stand-in."""
    shim = types.ModuleType("cupy")
    for name in dir(np):
        if name.startswith("__"):
            continue
        value = getattr(np, name)
        if callable(value) and not isinstance(value, type):
            setattr(shim, name, _returns_on_card(value))
        else:
            setattr(shim, name, value)
    for name in ("asarray", "ascontiguousarray", "array"):
        setattr(shim, name, _uploads(getattr(np, name)))
    shim.ndarray = np.ndarray
    shim.asnumpy = lambda value, *args, **kwargs: np.array(value).view(
        np.ndarray)
    class CensusDevice:
        id = 0
        compute_capability = "120"
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
    shim.cuda = types.SimpleNamespace(
        Device=CensusDevice, runtime=types.SimpleNamespace(getDevice=lambda: 0))
    shim.RawModule = _RawModule
    shim.RawKernel = lambda *args, **kwargs: _Launch()
    sys.modules["cupy"] = shim
    # The real package's companions resolve as absent, exactly as on an
    # install without cupy, so nothing reaches the real runtime through them.
    for name in ("cupyx", "cupy_backends"):
        sys.modules[name] = None


def _owner(array: np.ndarray) -> np.ndarray:
    while isinstance(getattr(array, "base", None), np.ndarray):
        array = array.base
    return array


_LEAVES = (str, bytes, int, float, complex, bool, type(None), type,
           types.ModuleType, types.FunctionType, types.BuiltinFunctionType,
           types.MethodType)


def _children(obj):
    if isinstance(obj, dict):
        return list(obj.items())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return list(enumerate(obj))
    attrs = dict(vars(obj)) if hasattr(obj, "__dict__") else {}
    for klass in type(obj).__mro__:
        for slot in getattr(klass, "__slots__", ()):
            if hasattr(obj, slot):
                attrs.setdefault(slot, getattr(obj, slot))
    return list(attrs.items())


def _state_owned(state, driver) -> set[int]:
    """ids of every array and container reachable from the state."""
    owned: set[int] = set()
    stack = [state]
    while stack:
        obj = stack.pop()
        if obj is driver or id(obj) in owned or isinstance(obj, _LEAVES):
            continue
        owned.add(id(obj))
        if isinstance(obj, np.ndarray):
            owned.add(id(_owner(obj)))
            continue
        stack.extend(value for _, value in _children(obj))
    return owned


def driver_arrays(driver, state) -> dict[str, tuple[tuple[int, ...], int]]:
    """Every distinct on-card buffer the driver holds and the state does not.

    Keyed by the largest view that reaches it (``fields/tke_myj``,
    ``pbl_tendencies/ru``, ``rthratenlw``), which is the spelling the
    inventory uses; equal-sized aliases keep the first path. A prefix view
    must not hide the full backing shape of a shared zero buffer.
    """
    skip = _state_owned(state, driver)
    found: dict[int, tuple[str, tuple[int, ...], int]] = {}
    seen: set[int] = set()

    def visit(obj, path: str) -> None:
        if id(obj) in seen or id(obj) in skip or isinstance(obj, _LEAVES):
            return
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            owner = _owner(obj)
            if isinstance(obj, _OnCard) and id(owner) not in skip:
                prior = found.get(id(owner))
                if prior is None or obj.size > math.prod(prior[1]):
                    found[id(owner)] = (path, tuple(obj.shape),
                                        int(owner.nbytes))
            return
        for key, value in _children(obj):
            visit(value, f"{path}/{key}" if path else str(key))

    visit(driver, "")
    return {path: (shape, nbytes) for path, shape, nbytes in found.values()}


def census() -> dict[str, dict]:
    import datetime

    from woof.core import solar
    from woof.core.physics import initialize_physics
    from woof.core.preflight import physics_array_shapes
    from woof.core.state import DomainState

    # The WRF solar geometry (radconst + calc_coszen, both generations) is a
    # compiled unit reached through the device-keyed module cache, which the
    # stand-in has no device for; Noah-MP's cold start asks it for COSZEN.
    # Its launches are no-ops here like every other, so the stand-in returns
    # the same three on-card buffers, zero-filled, with the same shapes.
    def solar_geometry_device(julian, xtime, gmt, latitude, longitude, *,
                              generation=solar.DEFAULT_SOLAR_GENERATION,
                              declination=None):
        import cupy as cp
        solar.solar_generation(generation)
        lat = cp.ascontiguousarray(cp.asarray(latitude, dtype=cp.float32))
        return (cp.zeros(2, cp.float32).view(_OnCardRow),
                cp.zeros_like(lat), cp.zeros_like(lat))

    solar.solar_geometry_device = solar_geometry_device

    # Noah-MP compiles at the direct-NVRTC boundary (woof.core.ieee_module)
    # rather than through cp.RawModule; the stand-in compiles nothing there
    # either, and its module launches nothing, exactly as _RawModule does.
    from woof.core import ieee_module

    def compile_module(source, options, module_key, log_stream=None):
        module = _RawModule()
        module.code, module.options = source, tuple(options)
        return module

    ieee_module.compile_module = compile_module

    report: dict[str, dict] = {}
    for cell in admitted_cells():
        cfg = cell_config(cell)
        latitude = np.full((cfg.ny, cfg.nx), 40.0)
        try:
            state = DomainState(cfg)
            driver = initialize_physics(
                state, cfg, glw=300.0,
                thermal_inertia=0.04 if cfg.sf_surface_physics == 1 else None,
                radiation_start_time=datetime.datetime(2020, 6, 1, 12),
                radiation_latitude=latitude, radiation_longitude=latitude - 135.0,
                noahmp_start_time=datetime.datetime(2020, 6, 1, 12),
                noahmp_latitude=latitude,
                noahmp_longitude=latitude - 135.0)
        except Exception as exc:  # reported, and failed, by the test
            report[cell_id(cell)] = {
                "error": f"{type(exc).__name__}: {exc}"}
            continue
        allocated = driver_arrays(driver, state)
        report[cell_id(cell)] = {
            "allocated": {path: [list(shape), nbytes]
                          for path, (shape, nbytes) in allocated.items()},
            "inventory": {key: list(shape) for key, shape in
                          physics_array_shapes(cfg).items()},
        }
    return report


def scalar_pblmix_census() -> dict[str, list]:
    """Record the actual local scalar launcher's allocation shapes.

    Construction alone cannot see this opt-in step workspace. Exercise the
    real launcher on the same allocation-counting CuPy stand-in, once with
    caller-owned work and once without, so unpriced output or work growth
    fails before it can overrun a card. Kernel arithmetic is checked by the
    separate Fortran and GPU oracles.
    """
    import cupy as cp
    from woof.core import mynn_scalar_mix_gpu

    ncol, nz = 7, 50
    column = cp.ones((ncol, nz), dtype=cp.float32, order="F")
    scratch = cp.empty((ncol, 5 * nz + 1), dtype=cp.float32, order="F")
    original = cp.empty
    allocated = []

    def record(shape, *args, **kwargs):
        result = original(shape, *args, **kwargs)
        allocated.append([list(result.shape), int(result.nbytes)])
        return result

    cp.empty = record
    kernel_lookup = mynn_scalar_mix_gpu.get_kernel
    # The context-owned kernel cache needs a real CUDA context. As in the
    # constructor census, launch no arithmetic; only allocation is counted.
    mynn_scalar_mix_gpu.get_kernel = lambda *args, **kwargs: _Launch()
    try:
        mynn_scalar_mix_gpu.scalar_pblmix_columns_cuda(
            column, column, column, column, 15.0, scratch=scratch)
        runtime = list(allocated)
        allocated.clear()
        mynn_scalar_mix_gpu.scalar_pblmix_columns_cuda(
            column, column, column, column, 15.0)
        standalone = list(allocated)
    finally:
        cp.empty = original
        mynn_scalar_mix_gpu.get_kernel = kernel_lookup
    return {"runtime": runtime, "standalone": standalone}


if __name__ == "__main__":
    install_numpy_cupy()
    json.dump(scalar_pblmix_census() if "--scalar-pblmix" in sys.argv
              else census(), sys.stdout)
