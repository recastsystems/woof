"""Grenier-Bretherton-McCaa moist TKE PBL, ``bl_pbl_physics = 12``.

CUDA launcher for :mod:`woof.core.kernels.gbmpbl`, the port of WRF v4.6.1
``phys/module_bl_gbmpbl.F`` (GBMPBL over GBM2D, with tridag, n2, my and
pblhgt), graded word for word against WRF's own compiled object by
``tools/gbmpbl_wrf461_oracle`` and ``tests/test_gbmpbl_wrf461_parity.py``.

WHAT THE SCHEME OWNS.  GBM predicts TKE (``TKE_PBL = e = q**2/2``) with a
Mellor-Yamada / Galperin stability closure, finds convectively unstable
layers and extends them by a buoyancy-integral rule that includes
cloud-top radiative cooling (``RTHRATEN``) and an inversion entrainment
closure, and then does its OWN implicit vertical diffusion of liquid-water
potential temperature, total water, momentum and cloud ice.  The launcher
therefore returns finished tendencies, like MYJ and UW.  WRF runs GBM2D
twice per call: the first pass's TKE becomes the new ``TKE_PBL`` and the
momentum tendencies are the mean of both passes (module_bl_gbmpbl.F
:251-280).

INPUTS.  Mass-level ``(nz, ny, nx)``: u and v at mass points, temperature,
qv, qc, qi (the WRF driver requires cloud water and cloud ice), pressure, the Exner function, layer thickness DZ8W, and the
radiation step's held RTHRATEN.  Surface ``(ny, nx)``: PSFC, GZ1OZ0, PSIM,
XLAND, HFX, QFX and TSK.  The scheme computes its own friction velocity
from GZ1OZ0 and PSIM and writes it back to UST (:539-541, :835).

COMPILATION. The declared strict options preserve scalar WRF arithmetic.
The existing woof.numerics boundary removes no-contraction in the fast
default before the image cache key and NVRTC compile. Fast mode also uses
CUDA float32 exp/log/pow instead of the replay helpers' double intermediates.
A warp owns each fast column: independent vertical levels run concurrently,
serial recurrences retain WRF order in shared memory, and paired right hand
sides reuse one factorization. Very deep columns retain global workspace.
Strict numerics is an opt-in; its source and options remain unchanged, and
fast and strict images have separate cache keys.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from woof.core.physics_inventory import (
    GBMPBL_BLOCK, GBMPBL_MASS_OUTPUTS, GBMPBL_STATE_3D,
    GBMPBL_SURFACE_OUTPUTS, GBMPBL_TILE_BLOCKS_PER_SM,
    gbmpbl_workspace_floats)
from woof.physics_vertical_contract import (
    GBMPBL_VERTICAL_LEVEL_BOUNDS, outside_vertical_bounds,
    refuse_vertical_levels)

MODULE_OPTIONS = ("-std=c++17", "-fmad=false", "--ftz=false")
MODULE_KEY = "woof.core.kernels:gbmpbl"

#: The level bounds woof.physics_vertical_contract publishes for scheme 12.
VERTICAL_LEVEL_BOUNDS = GBMPBL_VERTICAL_LEVEL_BOUNDS

#: Bottom-up ``(nz, ny, nx)`` inputs, in kernel argument order.
GBMPBL_COLUMN_INPUTS = ("u", "v", "t", "qv", "qc", "qi", "p", "exner",
                        "dz8w", "rthraten")
#: Surface ``(ny, nx)`` inputs, in kernel argument order.
GBMPBL_SURFACE_INPUTS = ("psfc", "gz1oz0", "psim", "xland", "hfx", "qfx",
                         "tsk")
#: Mass-level outputs, in kernel argument order (the inventory's roster;
#: TKE_PBL, the carried state, is updated in place and is not among them).
GBMPBL_COLUMN_OUTPUTS = GBMPBL_MASS_OUTPUTS
#: WRF's gbmpblinit cold start (module_bl_gbmpbl.F:1252).
GBMPBL_TKE_COLD_START = GBMPBL_STATE_3D["tke_pbl"]


def module_source() -> str:
    """The canonical strict GBM composition, preserving its replay identity."""
    from woof.core.kernels import _KDIR, _preamble, _unit_text
    from woof.core.thompson_contract import TableAsset
    from woof.table_assets import read_pinned_text_asset
    constants = read_pinned_text_asset(
        _KDIR.parents[1] / "data" / "gbmpbl",
        TableAsset("constants.cuh", 1629,
                   "ae3379d57ef2311b659ce811a583b24c6b4f5f495904b5fef2024b4bcd248c82"))
    return (_preamble() + (_KDIR / "glibc_flt32.cuh").read_text(encoding="utf-8")
            + constants + _unit_text("gbmpbl"))


def compile_source(*, cooperative: bool = True) -> str:
    """Select fast arithmetic and scheduling without changing the strict image.

    Define the aliases after the replay helpers have been parsed, so they
    affect only GBM's call sites. CUDA's float32 functions retain the same
    equations and iterations; no whole-program fast-math or FTZ switch is
    introduced. The cooperative source distributes independent levels over a
    warp, retaining serial recurrences and sharing identical factorizations.
    This source is what the cache and manifest record.
    """
    from woof.numerics import mode

    source = module_source()
    if mode() == "strict":
        return source
    marker = "// Grenier-Bretherton-McCaa moist TKE PBL"
    if source.count(marker) != 1:
        raise RuntimeError("GBM source boundary missing; cannot select fast arithmetic")
    source = source.replace(
        marker,
        "// Fast GBM uses CUDA float32 transcendental functions.\n"
        "#define gfk_exp expf\n#define gfk_log logf\n#define gfk_pow powf\n"
        + marker, 1)
    from woof.core.gbmpbl_fast import paired_source, warp_source
    return warp_source(source) if cooperative else paired_source(source)


@lru_cache(maxsize=None)
def _module(device: int, cooperative: bool = True):
    """Compile and record GBM without extending the sealed shared loader."""
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler
    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile
    from woof.numerics import effective_options
    source = compile_source(cooperative=cooperative)
    options = effective_options(MODULE_OPTIONS)
    with cp.cuda.Device(device), observe_module_compile(MODULE_KEY):
        ptx, _ = compiler.compile_using_nvrtc(
            source, options, None, "gbmpbl.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(MODULE_KEY, source=source, options=options, module=None)
    return module


def _kernel():
    import cupy as cp

    return _module(cp.cuda.Device().id).get_function("gbmpbl_column")


def tile_columns(ncol: int) -> int:
    """Columns per launch, bounding the workspace to what can be resident."""
    import cupy as cp

    sms = int(cp.cuda.Device().attributes["MultiProcessorCount"])
    return int(max(GBMPBL_BLOCK, min(int(ncol),
               sms * GBMPBL_TILE_BLOCKS_PER_SM * GBMPBL_BLOCK)))


def launch_gbmpbl(columns, surface, tke, outputs, *, dt: float) -> None:
    """Run one GBM PBL call in place.

    ``tke`` is TKE_PBL ``(nz, ny, nx)``, read on entry and rewritten with
    the first pass's TKE (module_bl_gbmpbl.F:270, :276).  ``outputs`` holds
    every :data:`GBMPBL_COLUMN_OUTPUTS` and :data:`GBMPBL_SURFACE_OUTPUTS`
    array.
    """
    # Validate the required bindings before touching geometry or a device.
    for name in GBMPBL_COLUMN_INPUTS:
        if columns.get(name) is None:
            raise KeyError(
                f"GBM PBL requires {name!r}; missing thermodynamic, wind, "
                "condensate or radiation input would corrupt its moist TKE "
                "and implicit diffusion budget")
    for name in GBMPBL_SURFACE_INPUTS:
        if surface.get(name) is None:
            raise KeyError(
                f"GBM PBL requires surface input {name!r}; missing surface "
                "forcing or MM5 stability input would corrupt its stress "
                "and heat/moisture flux budget")
    if tke is None:
        raise KeyError(
            "GBM PBL requires prognostic TKE_PBL; a missing carried state "
            "would reset turbulence between steps and invalidate its TKE budget")
    import cupy as cp

    shape = tuple(surface["psfc"].shape)
    nz = int(columns["t"].shape[0])
    if outside_vertical_bounds(nz, VERTICAL_LEVEL_BOUNDS):
        raise refuse_vertical_levels(
            "GBM PBL", VERTICAL_LEVEL_BOUNDS, nz,
            breakage=("the TKE equation is solved on the kte-1 interior "
                      "full levels (module_bl_gbmpbl.F:770-809); below two "
                      "mass levels that system is empty and the surface "
                      "TKE flux has no level to enter."))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"GBM PBL requires a positive finite dt, got {dt!r}")
    f32 = np.dtype(np.float32)
    arrays3 = [columns[name] for name in GBMPBL_COLUMN_INPUTS]
    arrays3 += [tke] + [outputs[name] for name in GBMPBL_COLUMN_OUTPUTS]
    for array in arrays3:
        if (array.shape != (nz, *shape) or array.dtype != f32
                or not array.flags.c_contiguous):
            raise ValueError(
                "launch_gbmpbl requires contiguous float32 "
                f"{(nz, *shape)} columns, got {array.shape} {array.dtype}")
    arrays2 = [surface[name] for name in GBMPBL_SURFACE_INPUTS]
    arrays2 += [outputs[name] for name in GBMPBL_SURFACE_OUTPUTS]
    for array in arrays2:
        expected = np.dtype(np.int32) if array is outputs["kpbl"] else f32
        if (array.shape != shape or array.dtype != expected
                or not array.flags.c_contiguous):
            raise ValueError("launch_gbmpbl requires same-shape contiguous "
                             "surface arrays (float32, kpbl int32)")
    ncol = int(np.prod(shape))
    from woof.numerics import mode
    # A warp owns one column in the fast image. Its independent vertical
    # levels run in parallel and its serial recurrences stay in shared RAM.
    # WRF has no defined upper-level limit here in our extended workspace;
    # retain the global-workspace image when a very deep column cannot fit
    # one block's shared memory, rather than introducing a new refusal.
    from woof.core.gbmpbl_fast import COOPERATIVE_SLOTS
    shared_bytes = COOPERATIVE_SLOTS * (nz + 3) * f32.itemsize
    cooperative = (mode() == "fast" and shared_bytes <=
                   int(cp.cuda.Device().attributes["MaxSharedMemoryPerBlock"]))
    kernel = (_module(cp.cuda.Device().id, cooperative=False).get_function(
              "gbmpbl_column") if mode() == "fast" and not cooperative else _kernel())
    tile = ncol if cooperative else tile_columns(ncol)
    ws = (tke if cooperative else
          cp.empty(gbmpbl_workspace_floats(nz, tile), dtype=cp.float32))
    for lo in range(0, ncol, tile):
        count = min(tile, ncol - lo)
        blocks = (count if cooperative else
                  (count + GBMPBL_BLOCK - 1) // GBMPBL_BLOCK)
        kernel((blocks,), (GBMPBL_BLOCK,),
               tuple(columns[name] for name in GBMPBL_COLUMN_INPUTS)
               + (tke,)
               + tuple(surface[name] for name in GBMPBL_SURFACE_INPUTS)
               + tuple(outputs[name] for name in GBMPBL_COLUMN_OUTPUTS)
               + (tke,)
               + tuple(outputs[name] for name in GBMPBL_SURFACE_OUTPUTS)
               + (np.float32(dt), np.int32(nz), np.int64(ncol), ws,
                  np.int32(nz + 3), np.int64(lo)),
               shared_mem=shared_bytes if cooperative else 0)


def gbmpbl_step(columns, surface, tke, *, dt: float, flqi: bool,
                out=None) -> dict:
    """Run one call, writing supplied output carriers in place.

    The returned dict carries the WRF names plus the ``du/dv/dtheta/dqv/
    dqc/dqi`` aliases :func:`woof.core.physics.couple_ysu_tendencies`
    consumes (the same device arrays, not copies).  ``dqi`` is withheld
    when the moist set has no ice: WRF's phy_bl_ten adds RQIBLTEN only under
    F_QI, and the scheme then diffused the driver's zero plane.
    """
    import cupy as cp

    shape = tuple(surface["psfc"].shape)
    nz = int(columns["t"].shape[0])
    outputs = {} if out is None else dict(out)
    for name in GBMPBL_COLUMN_OUTPUTS:
        if name not in outputs:
            outputs[name] = cp.empty((nz, *shape), dtype=cp.float32)
    for name, dtype in (("pblh", cp.float32), ("ust", cp.float32),
                        ("kpbl", cp.int32)):
        if name not in outputs:
            outputs[name] = cp.empty(shape, dtype=dtype)
    launch_gbmpbl(columns, surface, tke, outputs, dt=dt)
    outputs.update(du=outputs["rublten"], dv=outputs["rvblten"],
                   dtheta=outputs["rthblten"], dqv=outputs["rqvblten"],
                   dqc=outputs["rqcblten"])
    if flqi:
        outputs["dqi"] = outputs["rqiblten"]
    return outputs


@lru_cache(maxsize=1)
def _output_validation_kernel():
    """Check every published float in one pass, preserving field order.

    This is the same finite predicate as the individual reductions. No
    physics value is written. A bit per field keeps the first-invalid
    diagnosis while avoiding twelve separate reduction/ledger sequences.
    """
    import cupy as cp

    names = (*GBMPBL_COLUMN_OUTPUTS, "pblh", "ust")
    n3 = len(GBMPBL_COLUMN_OUTPUTS)
    inputs = [f"float32 a{i}" for i in range(n3)]
    inputs += ["raw float32 pblh", "raw float32 ust", "int64 n2"]
    operation = "unsigned int flags = 0;\n"
    for bit in range(n3):
        operation += f"if (!isfinite(a{bit})) flags |= {1 << bit}u;\n"
    for bit, name in enumerate(names[n3:], n3):
        operation += (f"if (i < n2 && !isfinite({name}[i])) "
                      f"flags |= {1 << bit}u;\n")
    operation += "if (flags) atomicOr(&status[0], flags);"
    return cp.ElementwiseKernel(
        ", ".join(inputs), "raw uint32 status", operation,
        "gpuwm_gbmpbl_finite_outputs")


def invalid_gbmpbl_output(outputs) -> str | None:
    """Name the first non-finite published GBM field, or ``None``.

    GBM has no WRF 0/0 of its own on any column the oracle exercises
    (calm, stable, saturated and dry columns included), so a NaN here is a
    defect and the driver refuses to advance on it.
    """
    import cupy as cp
    from woof.core import health_ledger

    names = (*GBMPBL_COLUMN_OUTPUTS, "pblh", "ust")
    shape = outputs[GBMPBL_COLUMN_OUTPUTS[0]].shape
    if len(shape) != 3:
        raise ValueError("GBM validation requires level, y, x output arrays")
    for name in names:
        array = outputs[name]
        expected = shape if name in GBMPBL_COLUMN_OUTPUTS else shape[1:]
        if (array.shape != expected or array.dtype != cp.float32
                or not array.flags.c_contiguous):
            raise ValueError(f"GBM validation requires contiguous float32 {name} "
                             f"with shape {expected}")
    status = cp.zeros(1, dtype=cp.uint32)
    _output_validation_kernel()(
        *(outputs[name] for name in names), np.int64(outputs["pblh"].size), status)

    def first_name(flags):
        return next((name for bit, name in enumerate(names)
                     if flags & (1 << bit)), None)

    def refuse(flags):
        raise FloatingPointError(
            f"the GBM PBL returned non-finite {first_name(flags)}"
            + health_ledger.deferred_note())

    invalid = health_ledger.read_status(
        status, site="gbmpbl:outputs", describe=refuse)
    return first_name(invalid)
