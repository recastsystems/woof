"""Batched CUDA execution of the NOAHMP_GLACIER column.

The arithmetic lives in ``woof/core/kernels/noahmp_glacier.cu`` (one
thread per glacier column, compiled after ``noahmp_leaves.cu`` for the
single glibc 2.39 device transcription -- see
``woof/core/noahmp_kernel_sources.py``).  This module is the
runtime-facing half: it packs the per-column input dictionary the
runtime gathers, launches one kernel over all glacier columns, and
unpacks the result columns.

The paired host authority is :func:`evaluate_glacier_columns_on_host`,
which answers the identical input dictionary through
:func:`woof.core.noahmp_glacier.noahmp_glacier`, the CPython
transcription -- the same two-implementation discipline every other
Noah-MP leaf carries, and what the GPU parity test compares bit for bit.

The Fortran aborts (``wrf_error_fatal``) in four places; the kernel
cannot, so each column reports an error code and this wrapper raises
:class:`woof.core.noahmp_glacier.GlacierBalanceError` naming the first
offending column and the per-code counts.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from woof.core.noahmp_driver_gpu import _current_device
from woof.core.noahmp_kernel_sources import DEFAULT_OPTIONS, compile_runtime_unit
from woof.core.noahmp_glacier import GlacierBalanceError, noahmp_glacier

__all__ = [
    "GLACIER_INPUT_SCALARS",
    "GLACIER_OUTPUT_SCALARS",
    "evaluate_glacier_columns",
    "evaluate_glacier_columns_on_host",
    "evaluate_simple_glacier_slab",
    "pack_glacier_columns",
]

NSNOW = 3
NSOIL = 4
NLAYER = NSNOW + NSOIL
THREADS = 64

#: Per-column float input slots 0..20, in kernel order.
GLACIER_INPUT_SCALARS = (
    "cosz", "sfctmp", "sfcprs", "uu", "vv", "q2", "soldn", "prcp", "lwdn",
    "tbot", "zlvl", "qsnow", "sneqvo", "albold", "cm", "ch", "sneqv",
    "snowh", "tg", "tauss", "qsfc",
)

#: (name, width) of the packed float array slots following the scalars.
GLACIER_INPUT_ARRAYS = (
    ("ficeold", NSNOW), ("smc", NSOIL), ("sh2o", NSOIL),
    ("zsnso", NLAYER), ("stc", NLAYER), ("snice", NSNOW),
    ("snliq", NSNOW),
)

N_INPUT = len(GLACIER_INPUT_SCALARS) + sum(w for _, w in
                                           GLACIER_INPUT_ARRAYS)

#: Per-column float output slots 0..32, in kernel order.
GLACIER_OUTPUT_SCALARS = (
    "tg", "tauss", "qsfc", "qsnow", "sneqvo", "albold", "cm", "ch",
    "sneqv", "snowh", "fsa", "fsr", "fira", "fsh", "fgev", "ssoil",
    "trad", "edir", "runsrf", "runsub", "sag", "albedo", "qsnbot",
    "ponding", "ponding1", "ponding2", "t2m", "q2e", "emissi", "fpice",
    "ch2b", "qmelt", "eflxb",
)

GLACIER_OUTPUT_ARRAYS = (
    ("smc", NSOIL), ("sh2o", NSOIL), ("stc", NLAYER), ("zsnso", NLAYER),
    ("hcpct", NLAYER), ("snice", NSNOW), ("snliq", NSNOW),
)

N_OUTPUT = len(GLACIER_OUTPUT_SCALARS) + sum(w for _, w in
                                             GLACIER_OUTPUT_ARRAYS)

_ERROR_NAMES = {
    1: "emitted longwave <= 0 (module_sf_noahmp_glacier.F:500)",
    2: "ERRSW radiation budget (:3009-3016)",
    3: "ERRENG energy budget (:3018-3025)",
    4: "ERRWAT water budget (:3031-3045)",
}


def _require(inputs: dict, count: int) -> None:
    for name in GLACIER_INPUT_SCALARS:
        value = np.asarray(inputs[name])
        if value.shape != (count,):
            raise ValueError(
                f"glacier input {name!r} must have shape ({count},), got "
                f"{value.shape}")
    for name, width in GLACIER_INPUT_ARRAYS:
        value = np.asarray(inputs[name])
        if value.shape != (count, width):
            raise ValueError(
                f"glacier input {name!r} must have shape ({count}, {width}),"
                f" got {value.shape}")
    isnow = np.asarray(inputs["isnow"])
    if isnow.shape != (count,):
        raise ValueError(
            f"glacier input 'isnow' must have shape ({count},), got "
            f"{isnow.shape}")
    zsoil = np.asarray(inputs["zsoil"], dtype=np.float32)
    if zsoil.shape != (NSOIL,):
        raise ValueError(
            f"glacier input 'zsoil' must have shape ({NSOIL},), got "
            f"{zsoil.shape}")


def pack_glacier_columns(inputs: dict, count: int) -> np.ndarray:
    """The ``float32[count, 52]`` row block the kernel reads."""
    _require(inputs, count)
    packed = np.empty((count, N_INPUT), dtype=np.float32)
    base = 0
    for name in GLACIER_INPUT_SCALARS:
        packed[:, base] = np.asarray(inputs[name], dtype=np.float32)
        base += 1
    for name, width in GLACIER_INPUT_ARRAYS:
        packed[:, base:base + width] = np.asarray(inputs[name],
                                                  dtype=np.float32)
        base += width
    assert base == N_INPUT
    return packed


def _raise_on_errors(errors: np.ndarray) -> None:
    if not errors.any():
        return
    bad = np.nonzero(errors)[0]
    first = int(bad[0])
    code = int(errors[first])
    counts = {int(c): int((errors == c).sum())
              for c in np.unique(errors[bad])}
    raise GlacierBalanceError(
        f"NOAHMP_GLACIER batch: {bad.size} of {errors.size} columns "
        f"failed a WRF balance gate; first is batch column {first} with "
        f"{_ERROR_NAMES.get(code, f'code {code}')}; per-code counts "
        f"{counts}")


@lru_cache(maxsize=None)
def _module(device: int, options: tuple[str, ...], selected_options: bool = False):
    """Compile (once per option set) the COMPOSED glacier translation
    unit -- noahmp_leaves.cu (the single glibc device transcription)
    followed by noahmp_glacier.cu, exactly as
    woof/core/noahmp_kernel_sources.py declares it.  ``get_kernel``
    loads single files and must not be handed a fragment."""
    return compile_runtime_unit(
        "noahmp_glacier", module_key="woof.core.noahmp_glacier_gpu:glacier"
        + ("(selected-options)" if selected_options else ""),
        options=options, selected_options=selected_options)


def glacier_module(options: tuple[str, ...] = DEFAULT_OPTIONS, *,
                   noahmp_snow_age="wrf_470", selected_options: bool = False):
    """The compiled composed glacier unit (cached per option set)."""
    from woof.noahmp_generation import snow_age_generation
    generation = snow_age_generation(noahmp_snow_age)
    # noahmp_glacier.cu defaults GPUWM_NOAHMP_NIGHT_AGE to 1, so the default
    # generation compiles runtime_unit("noahmp_glacier") exactly -- the unit
    # identity the frame recordings and the kernel manifest are keyed on --
    # and only the explicit wrf_461 reproduction adds a define.
    if generation == "wrf_470":
        return _module(_current_device(), tuple(options), selected_options)
    return _module(_current_device(), tuple(options) + ("-DGPUWM_NOAHMP_NIGHT_AGE=0",),
                   selected_options)


def evaluate_simple_glacier_slab(fields, count: int, dt: float, *,
                                  opt_alb=2, opt_stc=1, opt_tbot=2,
                                  noahmp_snow_age="wrf_470"):
    """Resident GLA=2 leaf candidate; returns device fields and error codes.

    This implements the selected native branch, independently of full parent
    option admission. Inputs follow GLACIER_INPUT_SCALARS/ARRAYS, all resident
    f32 C(count[,width]); isnow is i32[count], shared zsoil f32[4]. Native
    BATS glacier uses fixed constants rather than land's MPTABLE parameters.
    All packing, output splitting and numerical work stay on the current
    card/stream. Error codes remain resident for output/checkpoint handling.
    """
    import cupy as cp
    import math
    from woof.core.noahmp_slab_libm import copy_slab_slots, split_slab_slots

    if type(count) is not int or count < 0:
        raise ValueError("GLA=2 slab count must be a non-negative int")
    if type(opt_alb) is not int or opt_alb not in (1, 2):
        raise ValueError("GLA=2 opt_alb must be scalar int 1 or 2")
    if type(opt_stc) is not int or opt_stc not in (1, 2, 3):
        raise ValueError("GLA=2 opt_stc must be scalar int 1, 2 or 3")
    if type(opt_tbot) is not int or opt_tbot not in (1, 2):
        raise ValueError("GLA=2 opt_tbot must be scalar int 1 or 2")
    if (not isinstance(dt, (int, float, np.floating)) or isinstance(dt, bool)
            or not math.isfinite(dt) or dt <= 0):
        raise ValueError("GLA=2 dt must be finite positive host cadence metadata")

    device = cp.cuda.runtime.getDevice()

    def resident(name, shape, dtype):
        a = fields[name]
        if (not isinstance(a, cp.ndarray) or a.shape != shape
                or a.dtype != dtype or a.device.id != device):
            raise ValueError(f"GLA=2 {name} needs resident {dtype} {shape} "
                             "on the current CUDA device")
        return cp.ascontiguousarray(a)

    packed = cp.empty((count, N_INPUT), dtype=cp.float32)
    blocks = []
    slot = 0
    for name, width in (tuple((name, 1) for name in GLACIER_INPUT_SCALARS)
                        + GLACIER_INPUT_ARRAYS):
        shape = (count,) if width == 1 else (count, width)
        blocks.append((slot, resident(name, shape, cp.float32)))
        slot += width
    isnow = resident("isnow", (count,), cp.int32)
    zsoil = resident("zsoil", (NSOIL,), cp.float32)
    copy_slab_slots(packed, blocks)
    out = cp.empty((count, N_OUTPUT), dtype=cp.float32)
    io = cp.empty((count, 2), dtype=cp.int32)
    if count:
        name = ("noahmp_glacier_column_gla2_bats" if opt_alb == 1
                else "noahmp_glacier_column_gla2")
        selected_thermal = opt_stc == 2 or opt_tbot == 1
        if selected_thermal:
            name = "noahmp_glacier_column_options"
        kernel = glacier_module(noahmp_snow_age=noahmp_snow_age,
                                 selected_options=True).get_function(name)
        kernel(((count + THREADS - 1) // THREADS,), (THREADS,),
               (packed, isnow, zsoil, np.float32(dt), out, io, np.int32(count),
                *((np.int32(opt_alb), np.int32(2), np.int32(opt_stc),
                   np.int32(opt_tbot)) if selected_thermal else ())))
    specs = {}
    slot = 0
    for name, width in (tuple((name, 1) for name in GLACIER_OUTPUT_SCALARS)
                        + GLACIER_OUTPUT_ARRAYS):
        specs[name] = (slot, width)
        slot += width
    result = split_slab_slots(out, specs)
    result.update(split_slab_slots(io, {"isnow": (0, 1), "error_code": (1, 1)}))
    return result


def evaluate_glacier_columns(inputs: dict, count: int,
                             dt: float) -> dict[str, np.ndarray]:
    """Advance ``count`` glacier columns in one CUDA launch."""
    import cupy as cp

    packed = pack_glacier_columns(inputs, count)
    isnow = np.ascontiguousarray(np.asarray(inputs["isnow"],
                                            dtype=np.int32))
    zsoil = np.ascontiguousarray(np.asarray(inputs["zsoil"],
                                            dtype=np.float32))
    device_out = cp.empty(count * N_OUTPUT, dtype=cp.float32)
    device_io = cp.empty(count * 2, dtype=cp.int32)
    kernel = glacier_module(noahmp_snow_age=inputs.get("noahmp_snow_age", "wrf_470")).get_function("noahmp_glacier_column")
    blocks = (count + THREADS - 1) // THREADS
    kernel(
        (blocks,), (THREADS,),
        (cp.asarray(packed.ravel()), cp.asarray(isnow), cp.asarray(zsoil),
         np.float32(dt), device_out, device_io, np.int32(count)),
    )
    host = cp.asnumpy(device_out).reshape(count, N_OUTPUT)
    host_io = cp.asnumpy(device_io).reshape(count, 2)
    _raise_on_errors(host_io[:, 1])
    out: dict[str, np.ndarray] = {}
    base = 0
    for name in GLACIER_OUTPUT_SCALARS:
        out[name] = host[:, base].copy()
        base += 1
    for name, width in GLACIER_OUTPUT_ARRAYS:
        out[name] = host[:, base:base + width].copy()
        base += width
    assert base == N_OUTPUT
    out["isnow"] = host_io[:, 0].copy()
    return out


def evaluate_glacier_columns_on_host(inputs: dict, count: int,
                                     dt: float) -> dict[str, np.ndarray]:
    """The paired host authority: the identical batch through the CPython
    transcription, one column at a time."""
    _require(inputs, count)
    zsoil = np.asarray(inputs["zsoil"], dtype=np.float32)
    out: dict[str, np.ndarray] = {
        name: np.empty(count, dtype=np.float32)
        for name in GLACIER_OUTPUT_SCALARS}
    for name, width in GLACIER_OUTPUT_ARRAYS:
        out[name] = np.empty((count, width), dtype=np.float32)
    out["isnow"] = np.empty(count, dtype=np.int32)
    for i in range(count):
        result = noahmp_glacier(
            cosz=inputs["cosz"][i], nsnow=NSNOW, nsoil=NSOIL, dt=dt,
            sfctmp=inputs["sfctmp"][i], sfcprs=inputs["sfcprs"][i],
            uu=inputs["uu"][i], vv=inputs["vv"][i], q2=inputs["q2"][i],
            soldn=inputs["soldn"][i], prcp=inputs["prcp"][i],
            lwdn=inputs["lwdn"][i], tbot=inputs["tbot"][i],
            zlvl=inputs["zlvl"][i], ficeold=inputs["ficeold"][i],
            zsoil=zsoil, qsnow=inputs["qsnow"][i],
            sneqvo=inputs["sneqvo"][i], albold=inputs["albold"][i],
            cm=inputs["cm"][i], ch=inputs["ch"][i],
            isnow=int(inputs["isnow"][i]), sneqv=inputs["sneqv"][i],
            smc=inputs["smc"][i], zsnso=inputs["zsnso"][i],
            snowh=inputs["snowh"][i], snice=inputs["snice"][i],
            snliq=inputs["snliq"][i], tg=inputs["tg"][i],
            stc=inputs["stc"][i], sh2o=inputs["sh2o"][i],
            tauss=inputs["tauss"][i], qsfc=inputs["qsfc"][i],
            noahmp_snow_age=inputs.get("noahmp_snow_age", "wrf_470"))
        for name in GLACIER_OUTPUT_SCALARS:
            out[name][i] = getattr(result, name)
        for name, _w in GLACIER_OUTPUT_ARRAYS:
            out[name][i] = getattr(result, name)
        out["isnow"][i] = result.isnow
    return out
