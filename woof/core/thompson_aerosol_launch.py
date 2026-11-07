"""Shared launcher utilities for aerosol-aware Thompson (``mp_physics=28``).

Every ``woof.core.thompson_aerosol_*`` launcher module imports its argument
validation, its grid arithmetic and its CUDA module names from here, so the
six aerosol translation units cannot drift apart on conventions that are
invisible in a diff.

Two things this module deliberately does *not* do:

* It does not import anything private from :mod:`woof.core.thompson`.  That
  module is read-only for this port and its ``_validate_fields`` is a private
  name; :func:`validate_fields` below is a transcription of
  ``thompson.py:50-64``, not an alias, so a future change to the classic
  launcher cannot silently retarget mp=28.
* It does not make ``mp_physics=28`` selectable.  Importing it compiles
  nothing and registers nothing.

It also carries the pointwise probe launchers for
``woof/core/kernels/thompson_aerosol_probe.cu``.  Those exist so the five
aerosol kernel packages can unit-test one ``__device__`` helper at a time
against ``woof/data/thompson/oracle-aero/probe-*.csv`` without standing up a
network kernel first.
"""

from __future__ import annotations

import contextlib
import contextvars
from functools import lru_cache

import numpy as np

from woof.core.kernels import get_kernel, get_kernel_int_defines
from woof.core.state import DTYPE


# ---------------------------------------------------------------------------
# Which Thompson generation the aerosol kernels compile (RunConfig.
# thompson_version).
# ---------------------------------------------------------------------------
#
# ``wrf_461`` compiles every unit exactly as before, through get_kernel, so
# its assembled source and binary are the ones every earlier gate measured.
# ``wrf_39_noaa`` compiles the SAME files with one integer define,
# THOMPSON_AA_WRF39, whose #if arms carry the operational WRF 3.9 fork's
# statements (NOAA-EMC/HRRR v4.1.21 module_mp_thompson.F, cited per arm).
# The define is inserted after thompson_aerosol_common.cuh (kernels/
# __init__.py module_source_int_defines), so the header carries no #if:
# its fork variants are separate functions the .cu arms call.
#
# The generation is a property of one microphysics call, set by the
# adapter around it (thompson_version_scope) and read by every launcher
# through aerosol_kernel, so no launcher signature changes.
THOMPSON_VERSIONS = ("wrf_461", "wrf_39_noaa")
WRF39_DEFINES = (("THOMPSON_AA_WRF39", 1),)

_ACTIVE_VERSION: contextvars.ContextVar[str] = contextvars.ContextVar(
    "gpuwm_thompson_version", default="wrf_461")


def active_thompson_version() -> str:
    """The generation the current microphysics call runs."""
    return _ACTIVE_VERSION.get()


@contextlib.contextmanager
def thompson_version_scope(version: str):
    """Run the enclosed launches as ``version`` (one of THOMPSON_VERSIONS)."""
    if version not in THOMPSON_VERSIONS:
        raise ValueError(
            f"thompson_version={version!r} is not one of {THOMPSON_VERSIONS}")
    token = _ACTIVE_VERSION.set(version)
    try:
        yield version
    finally:
        _ACTIVE_VERSION.reset(token)


# ---------------------------------------------------------------------------
# Hail awareness (mp_physics = 38).
# ---------------------------------------------------------------------------
#
# WRF v4.6.1 runs ONE module for mp=28 and mp=38; the hail-aware member is
# the aerosol-aware scheme plus a prognostic graupel number and volume, nine
# graupel density bins and their nine-slab rain-graupel tables
# (module_mp_thompson.F:459-466, :1260-1264, :3266-3282, :3750-3753).  The
# same translation units therefore compile it, with one integer define,
# THOMPSON_AA_HAIL, whose #if arms carry exactly those statements.  Like the
# generation, it is a property of one microphysics call, set by the adapter
# around it (hail_aware_scope) and read by every launcher through
# aerosol_kernel, so the launch sites do not fork.
HAIL_DEFINES = (("THOMPSON_AA_HAIL", 1),)

_HAIL_AWARE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "gpuwm_thompson_hail_aware", default=False)


def active_hail_aware() -> bool:
    """Whether the current microphysics call is the hail-aware scheme."""
    return _HAIL_AWARE.get()


@contextlib.contextmanager
def hail_aware_scope(enabled: bool):
    """Run the enclosed launches as the hail-aware (mp=38) scheme or not."""
    if enabled and _ACTIVE_VERSION.get() != "wrf_461":
        # The operational WRF 3.9 fork has no hail-aware branch at all: its
        # graupel carries no number, no volume and no density axis (fork
        # :2031-2054, tables (28,28,37,37)), so there is nothing to compile
        # mp=38 against.  Refused rather than silently run as mp=28.
        raise ValueError(
            "hail-aware Thompson (mp_physics=38) exists only in the WRF "
            "v4.6.1 generation; thompson_version="
            f"{_ACTIVE_VERSION.get()!r} carries no graupel number, volume "
            "or density bins")
    token = _HAIL_AWARE.set(bool(enabled))
    try:
        yield bool(enabled)
    finally:
        _HAIL_AWARE.reset(token)


@lru_cache(maxsize=None)
def _hail_module(name: str):
    """Compile the hail image with its measured noncontracting arithmetic."""
    import cupy as cp
    from woof import nvrtc_ptx_cache
    from woof.core.kernels import module_source_int_defines, MODULE_KEY_ROOT
    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile
    from woof.physics_params import note_compiled
    source = module_source_int_defines(name, HAIL_DEFINES)
    options = ("-std=c++17", "--ftz=false", "--fmad=false")
    key = f"{MODULE_KEY_ROOT}:{name}[THOMPSON_AA_HAIL=1]"
    with observe_module_compile(key):
        image, _ = nvrtc_ptx_cache.compile_using_nvrtc(source, options, None, name + ".cu")
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    note_compiled(name)
    record_module(key,
                  source=source, options=options, module=None)
    return module


def aerosol_kernel(module: str, func: str):
    """``get_kernel(module, func)`` compiled for the active generation."""
    if _HAIL_AWARE.get():
        return _hail_module(module).get_function(func)
    if _ACTIVE_VERSION.get() == "wrf_39_noaa":
        return get_kernel_int_defines(module, func, WRF39_DEFINES)
    return get_kernel(module, func)


def hail_graupel_arguments(fields: dict, *, graupel_volume, qgten, ngten,
                           qbten) -> tuple:
    """The four hail-aware kernel arguments, validated, or four ``None``.

    Under :func:`hail_aware_scope` all four are required: the hail-aware
    graupel exists only as WRF's qb1d and its qgten/ngten/qbten.  Outside it
    all four must be absent, because the mp=28 kernels have no use for them
    and a caller passing them believes it is running a scheme it is not.
    The validated arrays join ``fields`` for the shape/dtype check.
    """
    given = {"graupel_volume": graupel_volume, "qgten": qgten,
             "ngten": ngten, "qbten": qbten}
    present = [name for name, value in given.items() if value is not None]
    if _HAIL_AWARE.get():
        if len(present) != 4:
            missing = sorted(set(given) - set(present))
            raise ValueError(
                "the hail-aware (mp_physics=38) source network needs "
                "graupel_volume, qgten, ngten and qbten; missing "
                + ", ".join(missing))
        fields.update(given)
        return (graupel_volume, qgten, ngten, qbten)
    if graupel_volume is not None or qbten is not None:
        raise ValueError(
            "graupel_volume/qgten/ngten/qbten are the hail-aware "
            "(mp_physics=38) arguments; this call is not hail aware (got "
            + ", ".join(present) + ")")
    return (None, None, None, None)


# ---------------------------------------------------------------------------
# CUDA module names.  These are the six .cu files that receive
# thompson_aerosol_common.cuh from woof/core/kernels/__init__.py's
# _EXTRA_HEADERS allow-list; the tuple below and that dict must agree, which
# tests/test_kernel_loader_inert.py asserts.
# ---------------------------------------------------------------------------

AEROSOL_COMMON_HEADER = "thompson_aerosol_common.cuh"

STATE_MODULE = "thompson_aerosol_state"
SAT_MODULE = "thompson_aerosol_sat"
COLD_MODULE = "thompson_aerosol_cold"
WARM_MODULE = "thompson_aerosol_warm"
SED_MODULE = "thompson_aerosol_sed"
PROBE_MODULE = "thompson_aerosol_probe"

AEROSOL_KERNEL_MODULES = (
    STATE_MODULE, SAT_MODULE, COLD_MODULE, WARM_MODULE, SED_MODULE,
    PROBE_MODULE,
)

#: The mp=8 translation unit.  It is byte-frozen and is never given a header.
CLASSIC_MODULE = "thompson"

#: woof's default 1-D launch width, matching woof/core/thompson.py.
DEFAULT_THREADS = 256

#: Fortran shape of ``tnccn_act`` (module_mp_thompson.F:247-251, :393).
CCN_ACTIVATION_SHAPE = (7, 9, 7, 5, 4)

#: Row order of ``thompson_aa_probe_constant_tables``'s output.  Part of the
#: published probe API; do not reorder.
PROBE_TABLE_ROWS = (
    "cce1", "cce2", "cce3", "cce4", "cce5",
    "ccg1", "ccg2", "ccg3", "ccg4", "ccg5",
    "ocg1", "ocg2", "g_ratio",
)
PROBE_TABLE_COLS = 16

#: Eff_aero collector species selectors; must match
#: THOMPSON_AA_SPECIES_* in thompson_aerosol_common.cuh.
SPECIES_RAIN = 0
SPECIES_SNOW = 1
SPECIES_GRAUPEL = 2
SPECIES_CODES = {"r": SPECIES_RAIN, "s": SPECIES_SNOW, "g": SPECIES_GRAUPEL}


# ---------------------------------------------------------------------------
# Argument validation.
# ---------------------------------------------------------------------------

def validate_fields(fields: dict[str, object]) -> tuple[tuple[int, ...], int]:
    """Require every named field to be one float32 C-contiguous shape.

    Transcribed from ``woof/core/thompson.py:50-64``.  Returns the common
    shape and its element count.
    """
    first = next(iter(fields.values()))
    shape = first.shape
    if not shape:
        raise ValueError("Thompson aerosol fields must be arrays")
    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    return shape, int(np.prod(shape, dtype=np.int64))


def add_optional_fields(fields, **group):
    """Add an all-or-none accumulator group, preserving argument order."""
    present = [name for name, value in group.items() if value is not None]
    if present and len(present) != len(group):
        names = list(group)
        label = ", ".join(names[:-1]) + " and " + names[-1]
        detail = f" (got {present})" if len(group) > 2 else ""
        raise ValueError(label + " are given together or not at all" + detail)
    if present:
        fields.update(group)


def validate_fp64_fortran_table(name, value, shape) -> None:
    """Require a float64 Fortran-ordered device table of the given shape.

    Transcribed from ``woof/core/thompson.py:64-71``.  mp=28's aerosol table
    contract (:mod:`woof.core.thompson_aerosol_runtime`) uploads float64
    Fortran arrays exactly like the classic one, so the same check applies.
    """
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must be float64, got {value.dtype}")
    if not value.flags.f_contiguous:
        raise ValueError(f"{name} must be Fortran-contiguous")


def validate_int_fields(fields: dict[str, object],
                        shape: tuple[int, ...]) -> None:
    """Require every named field to be one int32 C-contiguous ``shape``."""
    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != np.dtype(np.int32):
            raise TypeError(f"{name} must be int32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")


def launch_grid(size: int, threads: int = DEFAULT_THREADS
                ) -> tuple[tuple[int], tuple[int]]:
    """Return ``(grid, block)`` for a flat ``size``-element 1-D kernel."""
    if size <= 0:
        raise ValueError("launch size must be positive")
    blocks = (int(size) + threads - 1) // threads
    return (blocks,), (threads,)


# ---------------------------------------------------------------------------
# Pointwise probes of thompson_aerosol_common.cuh.
# ---------------------------------------------------------------------------

def _empty_like_int(reference):
    import cupy as cp
    return cp.empty(reference.shape, dtype=cp.int32)


def _empty_like_float(reference):
    import cupy as cp
    return cp.empty(reference.shape, dtype=DTYPE)


def _probe(func: str, size: int, args: tuple) -> None:
    grid, block = launch_grid(size)
    get_kernel(PROBE_MODULE, func)(grid, block, args)


def probe_nint(x):
    """Fortran ``NINT`` (round half away from zero) on device."""
    _, size = validate_fields({"x": x})
    out = _empty_like_int(x)
    _probe("thompson_aa_probe_nint", size, (x, out, np.int32(size)))
    return out


#: thompson_aa_probe_libm's operation codes.
LIBM_PROBE_OPS = {"expf": 0, "logf": 1, "log10f": 2, "powf": 3,
                  "exp": 4, "log": 5, "log10": 6, "pow": 7,
                  "hypot": 8, "log1p": 9, "atan2": 10}


def probe_libm(name, x, y=None):
    """One of WOOF's own libm words (thompson_aerosol_libm.cuh) on device.

    ``expf``, ``logf``, ``log10f`` and ``powf`` take and return float32;
    ``exp``, ``log``, ``log10``, ``pow``, ``hypot``, ``log1p`` and
    ``atan2`` binary64.  ``y`` is the exponent of the two powers, the second
    leg of ``hypot`` and the abscissa of ``atan2`` (``x`` its ordinate).
    """
    import cupy as cp
    op = LIBM_PROBE_OPS[name]
    double = op >= 4
    x = cp.ascontiguousarray(x, dtype=cp.float64 if double else DTYPE)
    y = cp.ascontiguousarray(x if y is None else y, dtype=x.dtype)
    size = int(x.size)
    out = cp.empty(x.shape, dtype=x.dtype)
    dummy_f = cp.empty((1,), dtype=DTYPE)
    dummy_d = cp.empty((1,), dtype=cp.float64)
    args = ((dummy_f, dummy_f, x, y, dummy_f, out) if double
            else (x, y, dummy_d, dummy_d, out, dummy_d))
    _probe("thompson_aa_probe_libm", size,
           (np.int32(op), *args, np.int32(size)))
    return out


def probe_reenforce_pair(a, b):
    """Blossey's re-enforcement of a paired collision transfer on device,
    module_mp_thompson.F:2945-2954: returns ``(a', b')`` with
    ``a' = REAL(MIN(ABS(a), ABS(b))) * SIGN(1.0, SNGL(a))`` and ``b' = -a'``.
    """
    import cupy as cp
    a = cp.ascontiguousarray(a, dtype=cp.float64)
    b = cp.ascontiguousarray(b, dtype=cp.float64)
    if a.shape != b.shape:
        raise ValueError("a and b must have one shape")
    size = int(a.size)
    a_out = cp.empty(a.shape, dtype=cp.float64)
    b_out = cp.empty(a.shape, dtype=cp.float64)
    _probe("thompson_aa_probe_reenforce_pair", size,
           (a, b, a_out, b_out, np.int32(size)))
    return a_out, b_out


def probe_nu_c(nc_m3):
    """``nu_c = MIN(15, NINT(1000.E6/nc) + 2)``, module_mp_thompson.F:2171."""
    _, size = validate_fields({"nc_m3": nc_m3})
    out = _empty_like_int(nc_m3)
    _probe("thompson_aa_probe_nu_c", size, (nc_m3, out, np.int32(size)))
    return out


def probe_droplet_bin(nc_m3):
    """Zero-based ``idx_n`` into ``tnc_wev``, module_mp_thompson.F:3447."""
    _, size = validate_fields({"nc_m3": nc_m3})
    out = _empty_like_int(nc_m3)
    _probe("thompson_aa_probe_droplet_bin", size,
           (nc_m3, out, np.int32(size)))
    return out


def probe_decade_index(value, first_exponent: int, table_size: int):
    """Zero-based base-ten decade/mantissa bin."""
    _, size = validate_fields({"value": value})
    out = _empty_like_int(value)
    _probe("thompson_aa_probe_decade_index", size,
           (value, np.int32(first_exponent), np.int32(table_size), out,
            np.int32(size)))
    return out


def probe_in_bin(xni):
    """Zero-based ``idx_IN``, module_mp_thompson.F:2579-2591."""
    _, size = validate_fields({"xni": xni})
    out = _empty_like_int(xni)
    _probe("thompson_aa_probe_in_bin", size, (xni, out, np.int32(size)))
    return out


def probe_inu_c_effrad(nc_m3):
    """``calc_effectRad``'s three-branch shape selector, :5637-5643."""
    _, size = validate_fields({"nc_m3": nc_m3})
    out = _empty_like_int(nc_m3)
    _probe("thompson_aa_probe_inu_c_effrad", size,
           (nc_m3, out, np.int32(size)))
    return out


def probe_clamps(nc_m3, nwfa_m3, nifa_m3):
    """WRF's terminal aerosol clamps, :1805-1806 and :3979-3981."""
    _, size = validate_fields(
        {"nc_m3": nc_m3, "nwfa_m3": nwfa_m3, "nifa_m3": nifa_m3})
    nc_out = _empty_like_float(nc_m3)
    nwfa_out = _empty_like_float(nwfa_m3)
    nifa_out = _empty_like_float(nifa_m3)
    _probe("thompson_aa_probe_clamps", size,
           (nc_m3, nwfa_m3, nifa_m3, nc_out, nwfa_out, nifa_out,
            np.int32(size)))
    return nc_out, nwfa_out, nifa_out


def probe_saturation(pressure, temperature):
    """``RSLF`` and ``RSIF``, module_mp_thompson.F:5378-5446."""
    _, size = validate_fields(
        {"pressure": pressure, "temperature": temperature})
    rslf = _empty_like_float(pressure)
    rsif = _empty_like_float(pressure)
    _probe("thompson_aa_probe_saturation", size,
           (pressure, temperature, rslf, rsif, np.int32(size)))
    return rslf, rsif


def probe_field_ab(tc, moment):
    """Field et al snow-moment power-law coefficients."""
    _, size = validate_fields({"tc": tc, "moment": moment})
    a_out = _empty_like_float(tc)
    b_out = _empty_like_float(tc)
    _probe("thompson_aa_probe_field_ab", size,
           (tc, moment, a_out, b_out, np.int32(size)))
    return a_out, b_out


def probe_activ_ncloud(temperature, w, nccn, tnccn_act):
    """``activ_ncloud``, module_mp_thompson.F:5178-5253.

    ``tnccn_act`` is the float64 Fortran-ordered ``(7,9,7,5,4)`` device array
    published by :mod:`woof.core.thompson_aerosol_runtime`.
    """
    _, size = validate_fields(
        {"temperature": temperature, "w": w, "nccn": nccn})
    validate_fp64_fortran_table(
        "tnccn_act", tnccn_act, CCN_ACTIVATION_SHAPE)
    out = _empty_like_float(temperature)
    _probe("thompson_aa_probe_activ_ncloud", size,
           (temperature, w, nccn, tnccn_act, out, np.int32(size)))
    return out


def probe_ice_demott(tempc, rho, nifa_m3):
    """``iceDeMott``, module_mp_thompson.F:5448-5518 (pure in tempc/rho/nifa)."""
    _, size = validate_fields(
        {"tempc": tempc, "rho": rho, "nifa_m3": nifa_m3})
    out = _empty_like_float(tempc)
    _probe("thompson_aa_probe_ice_demott", size,
           (tempc, rho, nifa_m3, out, np.int32(size)))
    return out


def probe_ice_koop(temperature, qv, qvs, naero, dt):
    """``iceKoop``, module_mp_thompson.F:5521-5546."""
    _, size = validate_fields({
        "temperature": temperature, "qv": qv, "qvs": qvs, "naero": naero,
        "dt": dt,
    })
    out = _empty_like_float(temperature)
    _probe("thompson_aa_probe_ice_koop", size,
           (temperature, qv, qvs, naero, dt, out, np.int32(size)))
    return out


def probe_eff_aero(d_collector, d_aerosol, visc, rhoa, temperature, species):
    """``Eff_aero``, module_mp_thompson.F:4965-5001.

    ``species`` is an int32 array of :data:`SPECIES_RAIN`,
    :data:`SPECIES_SNOW` or :data:`SPECIES_GRAUPEL`.
    """
    shape, size = validate_fields({
        "d_collector": d_collector, "d_aerosol": d_aerosol, "visc": visc,
        "rhoa": rhoa, "temperature": temperature,
    })
    validate_int_fields({"species": species}, shape)
    out = _empty_like_float(d_collector)
    _probe("thompson_aa_probe_eff_aero", size,
           (d_collector, d_aerosol, visc, rhoa, temperature, species, out,
            np.int32(size)))
    return out


def probe_snow_number(smob, smoc):
    """WRF's explicit two-gamma ``ns(k)``, module_mp_thompson.F:2081-2088."""
    _, size = validate_fields({"smob": smob, "smoc": smoc})
    out = _empty_like_float(smob)
    _probe("thompson_aa_probe_snow_number", size,
           (smob, smoc, out, np.int32(size)))
    return out


def probe_cloud_dist(rc, nc_per_kg, rho):
    """Entry droplet-distribution diagnosis, module_mp_thompson.F:1826-1842.

    Returns ``(nc_m3, nu_c, lamc)``; ``lamc`` is float64 because WRF declares
    it DOUBLE PRECISION.
    """
    import cupy as cp
    _, size = validate_fields(
        {"rc": rc, "nc_per_kg": nc_per_kg, "rho": rho})
    nc_out = _empty_like_float(rc)
    nu_c_out = _empty_like_int(rc)
    lamc_out = cp.empty(rc.shape, dtype=cp.float64)
    _probe("thompson_aa_probe_cloud_dist", size,
           (rc, nc_per_kg, rho, nc_out, nu_c_out, lamc_out, np.int32(size)))
    return nc_out, nu_c_out, lamc_out


def probe_effect_rad(temperature, pressure, qv, qc, nc_per_kg, qi, ni_per_kg,
                     qs):
    """``calc_effectRad``, module_mp_thompson.F:5594-5699.  Returns METRES."""
    _, size = validate_fields({
        "temperature": temperature, "pressure": pressure, "qv": qv, "qc": qc,
        "nc_per_kg": nc_per_kg, "qi": qi, "ni_per_kg": ni_per_kg, "qs": qs,
    })
    effc = _empty_like_float(temperature)
    effi = _empty_like_float(temperature)
    effs = _empty_like_float(temperature)
    _probe("thompson_aa_probe_effect_rad", size,
           (temperature, pressure, qv, qc, nc_per_kg, qi, ni_per_kg, qs,
            effc, effi, effs, np.int32(size)))
    return effc, effi, effs


def probe_constant_tables():
    """Read the ``__constant__`` gamma tables back as a ``(13, 16)`` array."""
    import cupy as cp
    out = cp.empty((len(PROBE_TABLE_ROWS), PROBE_TABLE_COLS), dtype=DTYPE)
    get_kernel(PROBE_MODULE, "thompson_aa_probe_constant_tables")(
        (1,), (PROBE_TABLE_COLS,), (out,))
    return out


__all__ = [
    "HAIL_DEFINES",
    "active_hail_aware",
    "hail_aware_scope",
    "hail_graupel_arguments",
    "AEROSOL_COMMON_HEADER",
    "AEROSOL_KERNEL_MODULES",
    "CCN_ACTIVATION_SHAPE",
    "CLASSIC_MODULE",
    "COLD_MODULE",
    "DEFAULT_THREADS",
    "LIBM_PROBE_OPS",
    "PROBE_MODULE",
    "PROBE_TABLE_COLS",
    "PROBE_TABLE_ROWS",
    "SAT_MODULE",
    "SED_MODULE",
    "SPECIES_CODES",
    "SPECIES_GRAUPEL",
    "SPECIES_RAIN",
    "SPECIES_SNOW",
    "STATE_MODULE",
    "WARM_MODULE",
    "launch_grid",
    "probe_activ_ncloud",
    "probe_clamps",
    "probe_cloud_dist",
    "probe_constant_tables",
    "probe_decade_index",
    "probe_droplet_bin",
    "probe_eff_aero",
    "probe_effect_rad",
    "probe_field_ab",
    "probe_ice_demott",
    "probe_ice_koop",
    "probe_in_bin",
    "probe_inu_c_effrad",
    "probe_libm",
    "probe_nint",
    "probe_nu_c",
    "probe_saturation",
    "probe_snow_number",
    "validate_fields",
    "validate_fp64_fortran_table",
    "validate_int_fields",
]
