"""WRF-real-like moist initialization on an explicit hybrid eta grid.

Setup and hydrostatic recurrences are float64.  The pressure-column vertical
interpolations follow WRF real's ratified defaults (interp_theta=F,
lagrange_order=2, use_surface=T with force_sfc_in_vinterp=1 and
zap_close_levels=500) through the common CUDA/parallel-CPU preprocessing
contract, and the completed prognostic state is FP32 on device.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import json
import sys
from time import perf_counter
import zlib

import numpy as np

from woof.config import (RunConfig, validate_aerosol_source_options,
                          validate_run_preparation)
from woof.core import constants as c
from woof.core import portable_math as pm
from woof.core.grid import (BaseState, VerticalCoord,
                             finalize_vertical_coord,
                             hybrid_column_ordering_refusal)
from woof.core.noahmp_libm import log1pf_array
from woof.core.state import DomainState
from woof.ingest.horiz import (
    HorizontalSnapshot,
    source_orography_from_catalog as _source_orography_from_catalog,
)
from woof.ingest.preprocess_backend import preprocess_math_call, resolve_preprocess_backend


def surface_fields_to_device(met, array_module, *, preserve_dtype=False):
    """Upload restored or freshly mapped near-surface fields to a device."""

    return {
        name: array_module.asarray(
            met.fields[name], dtype=(met.fields[name].dtype if preserve_dtype
                                     else array_module.float32))
        for name in ("T2", "U10", "V10")
    }


def _default_column_workers(preprocess_backend, preprocess_workers) -> int:
    """Setup column threads when the caller named none.

    On the CPU preparation the columns take the preparation's own worker
    count: the one it was given, else the CPU and memory budget. The same
    explicit count reaches host columns under either preparation backend.
    """
    if isinstance(preprocess_backend, str):
        name = preprocess_backend.strip().lower()
        workers = preprocess_workers
    else:
        name = str(getattr(preprocess_backend, "name", "")).strip().lower()
        workers = (preprocess_workers if preprocess_workers is not None
                   else getattr(preprocess_backend, "workers", None))
    if name != "cpu":
        from woof.ingest.cpu_backend import automatic_workers
        from woof.ingest.preparation_workers import effective_workers
        host_workers = (preprocess_workers if preprocess_workers is not None
                        else getattr(preprocess_backend, "host_step_workers", None))
        return automatic_workers() if host_workers is None else effective_workers(host_workers)
    if workers is not None:
        from woof.ingest.preparation_workers import effective_workers
        return effective_workers(workers)
    from woof.ingest.cpu_backend import automatic_workers
    return automatic_workers()


_MALLOC_TRIM = None


def _return_freed_host_memory() -> None:
    """Hand the C allocator's free pages back to the operating system.

    glibc gives each thread its own allocator arena and keeps what a thread
    freed there resident, so the slab temporaries of a threaded setup column
    stay in the process after its pool closes. ``malloc_trim(0)`` returns
    the whole free pages every arena holds in its free lists (not the free
    space at the top of a thread arena, which glibc keeps) and changes no
    value. Where the C library has no ``malloc_trim`` this does nothing.
    """
    global _MALLOC_TRIM
    if _MALLOC_TRIM is None:
        trim = False
        if sys.platform.startswith("linux"):
            try:
                import ctypes
                trim = ctypes.CDLL(None).malloc_trim
                trim.argtypes = (ctypes.c_size_t,)
                trim.restype = ctypes.c_int
            except (AttributeError, OSError):
                trim = False
        _MALLOC_TRIM = trim
    if _MALLOC_TRIM:
        _MALLOC_TRIM(0)


#: Whether a column helper's pool hands its freed pages back when it
#: closes.  A column-blocked initialization turns it off and trims once per
#: block instead.  THE BREAKAGE: ``malloc_trim`` walks every arena of a
#: process holding 150 GiB, and a blocked 1 km forcing time closes some
#: 25 helper pools per block over 39 blocks, about a thousand trims where
#: the whole-domain build made 25.
_POOL_TRIM: ContextVar[bool] = ContextVar("initialization_pool_trim", default=True)
#: Measurement only: True restores a trim at every pool close in blocks.
_BLOCK_POOL_TRIM = False


@contextmanager
def _column_pool(max_workers: int):
    """Threads for one setup column helper; their freed memory is returned
    to the operating system when the pool closes (once per block instead,
    in a column-blocked initialization)."""
    # The process's reused host threads (woof.ingest.host_parallel), not a
    # pool started and joined per helper call: a GFS preparation made ~25
    # helper calls per forcing time, each starting a thread per chunk, and
    # on a loaded host the thread starts alone cost ~10 % of its main
    # thread (measured 2026-10-27).  Same tasks, same chunks, same results;
    # leaving the block still waits for every task it submitted.
    from woof.ingest.host_parallel import shared_executor
    try:
        with shared_executor(max_workers) as executor:
            binding = pm.current_cpu_bridge_binding()
            if binding is None:
                yield executor
            else:
                class BoundExecutor:
                    def submit(self, function, *args, **kwargs):
                        return executor.submit(pm.run_with_cpu_bridge_binding,
                                               binding, function, *args, **kwargs)
                    def map(self, function, *iterables, **kwargs):
                        def bound(*values):
                            return pm.run_with_cpu_bridge_binding(binding, function, *values)
                        return executor.map(bound, *iterables, **kwargs)
                    def __getattr__(self, name):
                        return getattr(executor, name)
                yield BoundExecutor()
    finally:
        if _POOL_TRIM.get():
            _return_freed_host_memory()


def _column_worker_count(value) -> int:
    """Validate an explicit setup-only CPU column-worker count."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("column_workers must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError("column_workers must be positive")
    from woof.ingest.preparation_workers import effective_workers
    return effective_workers(value)


def _axis0_chunks(length: int, workers: int):
    """Return non-empty, deterministic contiguous chunks of axis zero."""
    workers = min(int(workers), int(length))
    q, r = divmod(int(length), workers)
    start = 0
    chunks = []
    for index in range(workers):
        stop = start + q + (index < r)
        chunks.append((start, stop))
        start = stop
    return tuple(chunks)


def _ordered_levels(array, order):
    """Apply a vertical order without copying monotonic source inventories."""
    array = np.asarray(array)
    order = np.asarray(order)
    identity = np.arange(array.shape[0], dtype=order.dtype)
    if np.array_equal(order, identity):
        return array
    if np.array_equal(order, identity[::-1]):
        return array[::-1]
    return array[order]


def _fill_axis0_chunk(output, start, stop, operation, args, kwargs=None):
    """Evaluate and immediately store one disjoint threaded output slab."""
    output[start:stop] = operation(*args, **({} if kwargs is None else kwargs))


def _host(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=np.float64)


def _host_owned_float64(value) -> np.ndarray:
    """Widen once into independent storage, retaining the existing layout."""
    if hasattr(value, "get"):
        value = value.get()
    return np.array(value, dtype=np.float64, copy=True, order="K")


HRRR_ANALYZED_HYDROMETEORS = ("QC", "QR", "QI", "QS", "QG")
DECLARED_ANALYZED_HYDROMETEORS = (*HRRR_ANALYZED_HYDROMETEORS, "QH")

#: Vertical velocity at a real-data cold start when the source carries
#: none: exact zero.  WRF real never interpolates a source W: v4.6.1
#: ``dyn_em/module_initialize_real.F`` (sha256
#: b4199371567369fa93471f66eef60cb63a55407718b828d724e43587bc4dc82e)
#: contains no assignment to ``grid%w_1`` or ``grid%w_2`` at all, so the W
#: real.exe writes is the allocation zero, and the first thing the model
#: does with it is diagnose the terrain-following lower boundary
#: (``dyn_em/start_em.F:1519-1530``, ``set_w_surface``).  That is the
#: right start for a source whose analysis holds no vertical motion.  It
#: was the WRONG start for an analysis that does: HRRR's own forecast
#: launches from a cycled state that keeps every updraft, while a start
#: from its published wrfnat analysis under this zero policy dropped them
#: all, so every storm the analysis held collapsed and had to re-form in
#: the first hour (2024-05-21 18Z Iowa crop: f01 FSS35 at 27 km 0.30
#: against HRRR's own 0.42; hour-one spin-up defect, fix program
#: 2026-10-06).  A source that publishes a vertical velocity on its model
#: levels (HRRR wrfnat and RAP awp130bgrb publish omega, 0/2/8, on all
#: fifty) now reaches ``initialize_real`` as ``OMEGA`` (Pa s-1) or ``WW``
#: (m s-1) through the regular-source join
#: (woof.mapped_source.VERTICAL_VELOCITY_LEGACY_NAMES) and is carried:
#: see :func:`vertical_velocity_from_omega` and
#: :func:`interface_vertical_velocity`.  The receipt names which of the
#: two happened rather than leaving it to be inferred from an absence.
WRF_REAL_VERTICAL_VELOCITY_POLICY = {
    "state_field": "w",
    "policy": "exact-fp32-zero",
    "wrf_version": "v4.6.1",
    "wrf_commit": "d66e442fccc04111067e29274c9f9eaccc3cef28",
    "real_citation": (
        "dyn_em/module_initialize_real.F assigns neither grid%w_1 nor "
        "grid%w_2; dyn_em/start_em.F:1519-1530 diagnoses the lower boundary"),
    "source_vertical_velocity": "the source carried none (no OMEGA or WW field)",
}

#: The constants UPP uses to publish omega from WRF's W
#: (``INITPOST.F``: ``omga = -w * pmid * g / (rd * t * (1 + d608 * q))``),
#: inverted here exactly so a round trip through the published product
#: returns the model's own W.
UPP_OMEGA_RD = 287.04
UPP_OMEGA_G = 9.81
UPP_OMEGA_D608 = 0.608

#: What the carried-W receipt says, before its measured numbers are added.
CARRIED_VERTICAL_VELOCITY_POLICY = {
    "state_field": "w",
    "policy": "carried-from-source",
    "interface_rule": (
        "W is interpolated on the mass levels (linear in pressure, constant "
        "below the source's lowest level), then each interior interface "
        "takes the mean of its two bounding mass levels; the surface "
        "interface is left to set_w_surface (WRF start_em.F:1519-1530) and "
        "the model top is zero"),
}


def vertical_velocity_from_omega(omega, temperature, qv, pressure, *, xp=np):
    """Geometric W (m s-1) from omega (Pa s-1), inverting UPP's publication.

    ``w = -omega * Rd * T * (1 + 0.608 qv) / (p * g)`` with UPP's own
    constants, on whatever array module ``xp`` the four operands share.
    """

    virtual = temperature * (1.0 + UPP_OMEGA_D608 * qv)
    return -(omega * (UPP_OMEGA_RD / UPP_OMEGA_G)) * virtual / pressure


def interface_vertical_velocity(w_mass, *, xp=np):
    """W on the ``nz + 1`` interfaces from W on the ``nz`` mass levels.

    Interior interfaces average their bounding mass levels; the surface
    interface is zero here (set_w_surface diagnoses it from the terrain
    slope on the first step, as it does for a zero start) and the model
    top is zero, WRF's upper boundary condition for W.
    """

    if w_mass.ndim != 3 or w_mass.shape[0] < 2:
        raise ValueError("mass-level W must be (nz, ny, nx) with nz >= 2")
    nz = w_mass.shape[0]
    out = xp.zeros((nz + 1,) + tuple(w_mass.shape[1:]), dtype=w_mass.dtype)
    out[1:nz] = 0.5 * (w_mass[:-1] + w_mass[1:])
    return out


def vertical_velocity_receipt(source_name, w_mass, w_interfaces, *, xp=np):
    """The carried-W receipt: the source field, the formula and the numbers."""

    receipt = dict(CARRIED_VERTICAL_VELOCITY_POLICY)
    receipt["source_field"] = source_name
    if source_name == "OMEGA":
        receipt["conversion"] = (
            "w = -omega * Rd * T * (1 + 0.608 qv) / (p * g) on the target "
            f"mass levels, Rd={UPP_OMEGA_RD}, g={UPP_OMEGA_G} (UPP INITPOST.F "
            "inverted)")
    else:
        receipt["conversion"] = "source geometric vertical velocity in m s-1, no conversion"
    receipt["mass_level_w_m_s"] = {
        "minimum": float(xp.min(w_mass)), "maximum": float(xp.max(w_mass)),
        "cells_above_1_m_s": int(xp.count_nonzero(w_mass > 1.0)),
        "cells_below_minus_1_m_s": int(xp.count_nonzero(w_mass < -1.0)),
    }
    receipt["interface_w_m_s"] = {
        "minimum": float(xp.min(w_interfaces)),
        "maximum": float(xp.max(w_interfaces)),
        "nonzero_count": int(xp.count_nonzero(w_interfaces)),
    }
    return receipt

HRRR_HYDROMETEOR_CORRESPONDENCE_SCHEMA_V1 = (
    "gpuwm-real-hydrometeor-correspondence-v1")
HRRR_HYDROMETEOR_CORRESPONDENCE_SCHEMA_V2 = (
    "gpuwm-real-hydrometeor-correspondence-v2")
HRRR_HYDROMETEOR_VERTICAL_DISPOSITION_SCHEMA = (
    "gpuwm-wrf-real-hydrometeor-vertical-disposition-v1")

_HRRR_DISPOSITION_CLASSES = {
    "TARGET_INFLUENCING": 1,
    "WRF_FORCE_SURFACE_EXCLUDED": 2,
    "WRF_ZAP_CLOSE_EXCLUDED": 3,
    "WRF_BELOW_GROUND_OUTSIDE_TARGET_SUPPORT": 4,
    "WRF_ABOVE_TARGET_TOP_OUTSIDE_TARGET_SUPPORT": 5,
    "WRF_NO_TARGET_STENCIL": 6,
}
_HRRR_DISPOSITION_CLASS_BY_CODE = {
    value: name for name, value in _HRRR_DISPOSITION_CLASSES.items()}
_HRRR_DISPOSITION_ENCODING = "base64-zlib-uint8-c-order-v1"
_HRRR_PRODUCTION_SUPPORT_SCHEMA = (
    "gpuwm-wrf-q-source-level-production-support-v1")
_HRRR_PRODUCTION_SUPPORT_ENCODING = (
    "base64-zlib-packbits-little-c-order-v1")
_HRRR_DISPOSITION_EXAMPLE_LIMIT = 16
_HRRR_DISPOSITION_OPERATOR = {
    "schema": "wrf-v4.6.1-real-linear-q-vertical-operator-v1",
    "wrf_version": "v4.6.1",
    "wrf_commit": "d66e442fccc04111067e29274c9f9eaccc3cef28",
    "source": "dyn_em/module_initialize_real.F:5664-6574",
    "use_surface": True,
    "surface_value": "exact-fp32-zero",
    "use_levels_below_ground": True,
    "force_sfc_in_vinterp": 1,
    "zap_close_levels_pa": 500.0,
    "zap_predicate": "pressure_separation < 500 Pa",
    "interp_in_logp": True,
    "lagrange_order": 2,
    "target_operator": "linear-at-every-target-vboundb=ntarget+1",
    "below_ground_extrapolation": "constant-deepest-assembled-point",
    "above_source_top": "fatal",
}

#: What each admitted id's WRF Registry ``moist:`` package can RECEIVE, keyed
#: by the id, WRF v4.6.1 (commit d66e442fccc04111067e29274c9f9eaccc3cef28).
#:
#: This mapping is the tuple's membership rule made executable rather than
#: written down.  Retention used to be an ``if cfg.mp_physics == 1`` ladder
#: below, so Kessler's narrow package was the only one anybody had to
#: remember, and the next admitted scheme whose package is narrower than
#: WSM6's would have been handed five species -- two of them into state
#: fields that do not exist for it.
#:
#:   1  kesslerscheme   Registry.EM_COMMON:3015  moist:qv,qc,qr
#:   3  wsm3scheme      Registry.EM_COMMON:3017  moist:qv,qc,qr
#:   6  wsm6scheme      Registry.EM_COMMON:3021  moist:qv,qc,qr,qi,qs,qg
#:   8  thompson        Registry.EM_COMMON:3024  moist:qv,qc,qr,qi,qs,qg
#:   9  milbrandt2mom   Registry.EM_COMMON:3025  moist:qv,qc,qr,qi,qs,qg,qh
#:  10  morr_two_moment Registry.EM_COMMON:3026  moist:qv,qc,qr,qi,qs,qg
#:  16  wdm6scheme      Registry.EM_COMMON:3031  moist:qv,qc,qr,qi,qs,qg
#:  18  nssl_2mom       Registry.EM_COMMON:3033  moist:qv,qc,qr,qi,qs,qg
#:  28  thompsonaero    Registry.EM_COMMON:3036  moist:qv,qc,qr,qi,qs,qg
#:  50  p3_1category    Registry.EM_COMMON:3038  moist:qv,qc,qr,qi
#:  51  p3_1category_nc Registry.EM_COMMON:3039  moist:qv,qc,qr,qi
#:
#: mp=28's moist list is character for character mp=8's -- the aerosol-aware
#: scheme adds only ``scalar:`` members -- so the mass-species handling is
#: the same at both, and that identity was verified against :3024 and :3036
#: directly rather than assumed from the mp=8 arm.
#:
#: mp=50 is the second narrow package, and narrow in the other direction
#: from Kessler's: P3 one-category HAS qi and has no qs and no qg at all,
#: because it carries ONE ice category and predicts rime mass fraction and
#: rime density (``scalar:qir,qib``, :3038) instead of splitting the frozen
#: mass into snow and graupel.  ``woof/core/state.py:464-497`` allocates no
#: qs and no qg on an mp=50 state, so retaining five species here would not
#: merely be wrong about the scheme, it would write into fields that do not
#: exist.  Its number moments (qni/qnr) and its rime pair are Registry
#: ``scalar:`` members that no real-data source read here supplies, so they
#: stay at the allocator's exact FP32 zero and P3 owns their first update --
#: the same source-absent policy mp=8/10/16/18/28 already take for their own
#: scalars, and the scheme is defined at that zero and says so: it floors
#: nitot at ``nsmall`` before any mean size is taken
#: (phys/module_mp_p3.F:2572-2573) and zeroes an unsupported rime pair in
#: ``calc_bulkRhoRime`` (:6799-6813).
#:
#: mp=9 is the first package WIDER than the decoded inventory: Milbrandt-Yau
#: carries hail mass beside graupel (``moist:qv,qc,qr,qi,qs,qg,qh``, :3025),
#: so it retains all five decoded species and discards nothing.  The seventh
#: member has no decoded source -- the native HRRR decoder reads exactly
#: QC/QI/QR/QS/QG (woof/ingest/hrrr.py:47), hail among them nowhere --
#: and real.exe's own QH arm agrees about what that means: the
#: interpolation at dyn_em/module_initialize_real.F:1979-1997 runs only
#: under ``flag_qh``, which a source that supplies no QH never raises, so
#: stock real.exe leaves QH at its allocated zero for this source too.
#: ``qh`` therefore stays at the allocator's exact FP32 zero
#: (woof/core/state.py mp=9 arm) exactly like the scheme's six number
#: moments (``scalar:qnc,qnr,qni,qns,qng,qnh``, :3025), which no real-data
#: source read here supplies either -- the same source-absent policy
#: mp=8/10/16/18/28 take.  The scheme is defined at that zero and says so:
#: its moment-consistency pass prescribes each number from its own mass
#: wherever Qx>epsQ and Nx<epsN, before any process rate is formed
#: (phys/module_mp_milbrandt2mom.F:1444-1530 -- N_c_SM for cloud at
#: :1459-1460, the No_r_SM/No_g_SM/No_h_SM intercept inversions for rain,
#: graupel and hail, Cooper for ice, Nos_Thompson for snow), so analyzed
#: mass entering with zero moments gets the scheme's own prescribed size
#: distributions on the first call rather than undefined arithmetic.
HRRR_ANALYZED_HYDROMETEOR_MOIST_PACKAGE = {
    2: {"registry_citation":"Registry/Registry.EM_COMMON:3016",
        "retained":HRRR_ANALYZED_HYDROMETEORS},
    1: {"registry_citation": "Registry/Registry.EM_COMMON:3015",
        "retained": ("QC", "QR")},
    # WSM3: Kessler's narrow package; real.exe keeps QC and QR and drops
    # the analysed QI/QS/QG (module_initialize_real.F:1859-1977).
    3: {"registry_citation": "Registry/Registry.EM_COMMON:3017",
        "retained": ("QC", "QR")},
    4: {"registry_citation": "Registry/Registry.EM_COMMON:3018",
        "retained": ("QC", "QR", "QI", "QS")},
    6: {"registry_citation": "Registry/Registry.EM_COMMON:3021",
        "retained": HRRR_ANALYZED_HYDROMETEORS},
    106: {"registry_citation": "Registry/Registry.EM_COMMON:3022",
          "retained": HRRR_ANALYZED_HYDROMETEORS},
    7: {"registry_citation": "Registry/Registry.EM_COMMON:3023",
        "retained": HRRR_ANALYZED_HYDROMETEORS},
    8: {"registry_citation": "Registry/Registry.EM_COMMON:3024",
        "retained": HRRR_ANALYZED_HYDROMETEORS},
    9: {"registry_citation": "Registry/Registry.EM_COMMON:3025",
        "retained": HRRR_ANALYZED_HYDROMETEORS},
    10: {"registry_citation": "Registry/Registry.EM_COMMON:3026",
         "retained": HRRR_ANALYZED_HYDROMETEORS},
    14: {"registry_citation": "Registry/Registry.EM_COMMON:3030",
         "retained": ("QC", "QR", "QI", "QS")},
    16: {"registry_citation": "Registry/Registry.EM_COMMON:3031",
         "retained": HRRR_ANALYZED_HYDROMETEORS},
    18: {"registry_citation": "Registry/Registry.EM_COMMON:3033",
         "retained": HRRR_ANALYZED_HYDROMETEORS},
    24: {"registry_citation": "Registry/Registry.EM_COMMON:3034",
         "retained": HRRR_ANALYZED_HYDROMETEORS},
    26: {"registry_citation": "Registry/Registry.EM_COMMON:3035",
         "retained": HRRR_ANALYZED_HYDROMETEORS},
    27: {"registry_citation": "WRF 4.7.1 Registry/Registry.EM_COMMON:3040",
         "retained": HRRR_ANALYZED_HYDROMETEORS},
    28: {"registry_citation": "Registry/Registry.EM_COMMON:3036",
         "retained": HRRR_ANALYZED_HYDROMETEORS},
    29: {"registry_citation": "WRF 4.7.1 Registry/Registry.EM_COMMON:3053",
         "retained": HRRR_ANALYZED_HYDROMETEORS},
    38: {"registry_citation": "Registry/Registry.EM_COMMON:3037",
         "retained": HRRR_ANALYZED_HYDROMETEORS},
    30: {"registry_citation": "Registry/registry.sbm:318",
         "retained": HRRR_ANALYZED_HYDROMETEORS},
    52: {"registry_citation": "Registry/Registry.EM_COMMON:3040",
         "retained": ("QC", "QR", "QI")},
    50: {"registry_citation": "Registry/Registry.EM_COMMON:3038",
         "retained": ("QC", "QR", "QI")},
    # Ferrier-Aligo: moist:qv,qc,qr,qi, so real.exe interpolates QC, QR and
    # QI and drops QS/QG (dyn_em/module_initialize_real.F:1859-1977 keys
    # every species on its P_Q* membership).  QI here is the analysis's
    # cloud ice alone, which is exactly what real.exe hands the scheme.
    15: {"registry_citation": "Registry/Registry.EM_COMMON:3020",
         "retained": ("QC", "QR", "QI")},
    5: {"registry_citation": "Registry/Registry.EM_COMMON:3019",
        "retained": ("QC", "QR", "QI")},
    97: {"registry_citation": "Registry/Registry.EM_COMMON:3046",
         "retained": HRRR_ANALYZED_HYDROMETEORS},
    # mp=51 carries mp=50's moist list (:3039); its extra member, the
    # droplet number qnc, is a ``scalar:`` one no source read here supplies,
    # so it stays at the allocator's FP32 zero exactly like qni/qnr -- what
    # real.exe does too, whose QNC interpolation runs only under flag_qnc
    # (dyn_em/module_initialize_real.F:2019-2024).  P3 is defined at that
    # zero: get_cloud_dsd2 floors nc at nsmall and its lambda limiter
    # rebuilds a droplet number from the cloud water (module_mp_p3.F
    # :6655-6685) before any process rate reads it.
    51: {"registry_citation": "Registry/Registry.EM_COMMON:3039",
         "retained": ("QC", "QR", "QI")},
    53: {"registry_citation": "Registry/Registry.EM_COMMON:3041",
         "retained": ("QC", "QR", "QI")},
    56: {"registry_citation": "Registry/Registry.EM_COMMON:3044",
         "retained": DECLARED_ANALYZED_HYDROMETEORS},
}

#: The microphysics ids whose WRF Registry ``moist:`` package can receive the
#: decoded HRRR analyzed hydrometeor inventory, so the decoder must supply it.
#:
#: This is ONE tuple on purpose.  It was three separate literals, and mp=28
#: was added to none of them: an mp=28 real-data run reached
#: :func:`initialize_real` with QC/QR/QI/QS/QG present in the snapshot, never
#: declared them required, never shape-checked them, never interpolated them,
#: and produced a condensate-free initial state from a cloudy analysis with
#: no error anywhere.  A single name makes that class of omission impossible.
#: mp=50 then reproduced that failure verbatim: the tuple was already a
#: single name, and P3 was simply never added to it, so a decoded native
#: HRRR analysis initialized an mp=50 forecast with qc/qr/qi identically
#: zero and nothing raised.  mp=9 was the third recurrence: Milbrandt-Yau's
#: package holds every decoded species and two more Registry members
#: besides, yet with 9 absent from the tuple an mp=9 real-data run started
#: with all five mass species identically zero,
#: ``hydrometeor_initialization`` empty, and nothing raised -- the same
#: condensate-free start from a cloudy analysis.  The ids are now exactly
#: the keys of HRRR_ANALYZED_HYDROMETEOR_MOIST_PACKAGE above, and the
#: retention each one gets is read from that same mapping instead of from
#: a ladder here.
#:
#: Kessler and P3 are members even though their packages are narrower than
#: the decoded inventory, because real.exe still requires the decoded
#: inventory and then discards what the ACTIVE package lacks
#: (WRF_REAL_PACKAGE_ABSENT_SPECIES_POLICY below); mp_physics=0 is refused
#: outright above.
#:
#: The literal is spelled out rather than derived from the mapping's keys so
#: the repo-wide admission-tuple census in
#: ``tests/test_mp28_runtime_reachability.py`` -- which walks for literal
#: integer sets and would go blind on a comprehension -- can still see this
#: site.  ``test_the_analyzed_inventory_tuple_and_its_moist_packages_agree``
#: pins the two equal.
HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 18, 24, 26, 27, 28, 29, 30, 38, 50, 51, 52, 53, 56, 97, 106)
# Named schemes with the six-mass moist package retain every analysed
# hydrometeor (capability row, woof.microphysics_schemes).
from woof.microphysics_schemes import NAMED_SCHEMES as _NAMED_MP_SCHEMES
for _s in _NAMED_MP_SCHEMES.values():
    if _s.ice_mass_species == ("qi", "qs", "qg"):
        HRRR_ANALYZED_HYDROMETEOR_MOIST_PACKAGE[_s.mp_id] = {
            "registry_citation": f"woof.microphysics_schemes:{_s.name}",
            "retained": HRRR_ANALYZED_HYDROMETEORS}
        HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS += (_s.mp_id,)

# WRF v4.6.1, commit d66e442fccc04111067e29274c9f9eaccc3cef28:
# dyn_em/module_initialize_real.F:1859-1977 interpolates QR/QC/QI/QS/QG one
# species at a time, and each interpolation runs only when the source flag
# is set AND that species' P_Q* index is a member of the ACTIVE num_moist
# package.  So real.exe retains the analyzed species the scheme's package
# carries and drops the rest, with no error and no diagnostic: QI/QS/QG for
# Kessler (Registry.EM_COMMON:3015, moist:qv,qc,qr) and QS/QG for P3
# one-category (:3038, moist:qv,qc,qr,qi).  Dropping them SILENTLY is what
# real.exe does; this receipt is what woof does instead, so that a species
# deliberately discarded and a species accidentally lost stay
# distinguishable after the fact.  ``registry_citation`` is filled per id
# from HRRR_ANALYZED_HYDROMETEOR_MOIST_PACKAGE, because WHICH package did
# the discarding is the content of the claim.
WRF_REAL_PACKAGE_ABSENT_SPECIES_POLICY = {
    "policy": "discard-source-species-absent-from-active-moist-package",
    "wrf_version": "v4.6.1",
    "wrf_commit": "d66e442fccc04111067e29274c9f9eaccc3cef28",
    "real_citation": "dyn_em/module_initialize_real.F:1859-1977",
}

#: What a real-data mp_physics=28 initial state does and does NOT contain,
#: as a receipt rather than as a comment: an mp=28 aerosol field that is
#: silently zero and an mp=28 aerosol field that is deliberately zero look
#: identical in the state, and only one of them is correct.
#:
#: ``deferred_to`` is the essential entry.  Nothing in this module fills
#: nwfa/nifa/nwfa2d; the fill is a one-time per-domain step that belongs to
#: the physics init path, exactly as WRF calls ``thompson_init`` from
#: ``phys/module_physics_init.F`` and not from ``module_initialize_real.F``.
WRF_REAL_MP28_AEROSOL_SOURCE_POLICY = {
    "policy": "aer-init-opt-0-zero-then-thompson-init-synthetic-profile",
    "wrf_version": "v4.6.1",
    "wrf_commit": "d66e442fccc04111067e29274c9f9eaccc3cef28",
    "registry_citation": "Registry/Registry.EM_COMMON:3036",
    "real_citation": "dyn_em/module_initialize_real.F:2332-2345",
    "microphysics_citation": (
        "phys/module_mp_thompson.F:493 (CCN MAXVAL test), :498-515 (CCN "
        "fill), :509-510 (nwfa2d), :531 (IN MAXVAL test), :536-551 (IN fill)"
    ),
    "wrf_real_refuses_this_configuration": (
        "dyn_em/module_initialize_real.F:2735-2736"),
    "deferred_to": (
        "woof.core.physics.initialize_physics -> "
        "woof.core.microphysics.microphysics_init"),
    "not_initialized_here": ("nwfa", "nifa", "nwfa2d", "nifa2d"),
    "number_moments": (
        "nc, nr, ni: where the mass is present and the number is at or "
        "below zero, real.exe's make_DropletNumber (from the cloud water "
        "and the aerosol), make_RainNumber and make_IceNumber (from the "
        "mass and the temperature) seed them "
        "(dyn_em/module_initialize_real.F:4829-4852, WRF v4.7.1), and the "
        "scheme's entry block then closes them over the analyzed mass "
        "(hydrometeor_initialization.cold_start_moment_closure)"),
}


def _warn_synthetic_aerosol_fallback(source_choice, resolution) -> None:
    """Announce the synthetic aerosol fallback on the way past it.

    The receipt is the durable record, but a receipt is read afterwards and
    this is a fact about what the run IS.  A user who expected a data-
    initialized forecast and got an analytic profile should learn it while
    the run is starting, not while reconciling numbers a week later, so the
    same named sentence also goes to the logger at WARNING.

    A DELIBERATE ``mp28_aerosol_source='synthetic'`` is not warned about at
    that level: nothing went wrong, the user asked for it.  It is still
    recorded in the receipt, because the receipt has to say what the run
    was regardless of who chose it.
    """
    from woof.config import MP28_AEROSOL_SYNTHETIC_FALLBACK

    if source_choice == "synthetic":
        print("mp_physics=28 aerosol: synthetic profile selected by name "
              "(mp28_aerosol_source='synthetic'); no aerosol dataset was "
              "searched for.", file=sys.stderr)
        return
    detail = ""
    if resolution is not None and resolution.fallback_reason:
        detail = " " + str(resolution.fallback_reason)
    print(MP28_AEROSOL_SYNTHETIC_FALLBACK + detail, file=sys.stderr)


def _resolved_mp28_aerosol_source(cfg) -> str:
    """The mp=28 aerosol source this configuration actually selects.

    ONE SPELLING, because two sites decide on it and a disagreement
    between them is a refusal for a dataset the run would never open: the
    run-door floor in :func:`initialize_real` asks it before deciding
    whether the monthly climatology is a precondition, and the aerosol
    resolver asks it again when it picks the source.

    ``aer_init_opt = 1`` with ``wif_input_opt = 1`` is real.exe's own
    spelling of "use the climatology", so it resolves 'auto' to the
    strict form exactly as the resolver does; every other value of
    ``mp28_aerosol_source`` is the operator's own word and is returned
    unchanged.  The pair test itself is
    :func:`woof.ingest.analyzed_numbers.wif_climatology_named_by_namelist`
    and not a second copy of the tuple, because the messages that tell an
    operator how to UNDO this resolution ask the same question and a
    disagreement between them is a way out that does nothing.
    """
    from woof.ingest.analyzed_numbers import wif_climatology_named_by_namelist

    if bool(getattr(cfg, "use_rap_aero_icbc", False)):
        return "analysis"
    if wif_climatology_named_by_namelist(cfg):
        return "climatology"
    return str(getattr(cfg, "mp28_aerosol_source", "auto") or "auto")


COLD_START_MOMENT_CLOSURE_SCHEMA = "gpuwm-real-cold-start-moment-closure-v1"


def _cold_start_droplet_number(qc, nc, inverse_density, aerosol_number,
                               landmask, *, build_receipt=True):
    """real.exe's cold-start droplet number, WRF v4.7.1
    ``dyn_em/module_initialize_real.F:4829-4838``.

    Where a cell has cloud water (``qc > 0``) and no droplet number
    (``nc <= 0``), real.exe sets ``nc = make_DropletNumber(qc*rho,
    nwfa*rho, xland) / rho`` with ``rho = 1./alt``, reading the aerosol
    number the state holds at that point: the climatology or the analyzed
    field when one was installed, zero when thompson_init's synthetic
    profile is still to come.  With zero aerosol the droplet size is
    set by the surface, so XLAND is taken from the target grid's
    LANDMASK (1 land, 2 water, the assignment of :3249-3260).

    Returns ``(seeded_nc, seed_mask, receipt)``; ``seeded_nc`` equals
    ``nc`` outside ``seed_mask``.
    """
    from woof.core.thompson_entry import (
        MAKE_DROPLET_NUMBER_SOURCE, cold_start_aerosol_row_refusal,
        droplet_mean_diameter_m, make_droplet_number,
    )

    alt = np.asarray(_host(inverse_density), dtype=np.float32)
    # temp_rho = 1./grid%alt(i,k,j) is REAL (:4825).
    nwfa = np.broadcast_to(
        np.asarray(_host(aerosol_number), dtype=np.float32), qc.shape)
    seed = (qc > np.float32(0.0)) & (nc <= np.float32(0.0))
    rho_seed = (np.float32(1.0) / alt[seed]).astype(np.float32)
    aerosol_seed = (nwfa[seed] * rho_seed).astype(np.float32)
    surface_needed = np.zeros(qc.shape, dtype=np.bool_)
    surface_needed[seed] = aerosol_seed <= np.float32(0.0)
    if landmask is None:
        if bool(surface_needed.any()):
            # A cloudy cell with no aerosol takes its droplet size from
            # the surface: 11 um continental drops over land, 17 um
            # maritime drops over water, about 3.7 times fewer.  Without
            # the target LANDMASK the choice would be a guess that sets
            # every such cell's droplet number wrong over one of the two.
            raise ValueError(
                "mp_physics=28 cold start: "
                f"{int(np.count_nonzero(surface_needed))} cloudy cell(s) "
                "carry no aerosol, and real.exe's make_DropletNumber then "
                "sizes the droplets by land or water (XLAND), but this "
                "initialization was given no target LANDMASK; pass "
                "landmask=<static LANDMASK> to initialize_real")
        xland = np.full(qc.shape[-2:], 1.0, dtype=np.float32)
    else:
        xland = np.where(np.asarray(_host(landmask), dtype=np.float64)
                         >= 0.5, 1.0, 2.0).astype(np.float32)
    xland3 = np.broadcast_to(xland, qc.shape)
    seeded = np.array(nc, dtype=np.float32)
    count = int(np.count_nonzero(seed))
    receipt = {
        "authority": MAKE_DROPLET_NUMBER_SOURCE,
        "rule": ("qc > 0 and nc <= 0: nc = make_DropletNumber(qc*rho, "
                 "nwfa*rho, xland) / rho, rho = 1/alt"),
        "seeded_cells": count,
    }
    if count:
        # A NaN per-volume aerosol number has no gamma-table row; named
        # here, with its cells, before the mirror indexes with it.
        not_a_number = int(np.count_nonzero(np.isnan(aerosol_seed)))
        if not_a_number:
            raise ValueError(cold_start_aerosol_row_refusal(not_a_number))
        rc = (qc[seed] * rho_seed).astype(np.float32)
        per_volume = make_droplet_number(
            rc, aerosol_seed, xland3[seed])
        seeded[seed] = (per_volume / rho_seed).astype(np.float32)
        if build_receipt:
            aerosol_cells = ~surface_needed[seed]
            ocean_cells = surface_needed[seed] & (xland3[seed] > 1.5)
            diameter = droplet_mean_diameter_m(rc, per_volume)
            receipt.update({
                "aerosol_branch_cells": int(np.count_nonzero(aerosol_cells)),
                "land_branch_cells": int(np.count_nonzero(
                    surface_needed[seed] & ~(xland3[seed] > 1.5))),
                "water_branch_cells": int(np.count_nonzero(ocean_cells)),
                "number_per_cm3_min": float(per_volume.min() * 1.0e-6),
                "number_per_cm3_max": float(per_volume.max() * 1.0e-6),
                "mean_diameter_um_min": float(diameter.min() * 1.0e6),
                "mean_diameter_um_max": float(diameter.max() * 1.0e6),
            })
    return seeded, seed, receipt


def _cold_start_rain_ice_number(species, mass, number, inverse_density,
                                temperature, *, build_receipt=True):
    """real.exe's cold-start rain or ice number, WRF v4.7.1
    ``dyn_em/module_initialize_real.F:4840-4852``.

    Where a cell has rain (ice) mass and no number (``<= 0``), real.exe
    sets ``nr = make_RainNumber(qr*rho, T) / rho`` (``ni =
    make_IceNumber(qi*rho, T) / rho``) with ``rho = 1./alt`` and ``T``
    the temperature it holds in ``grid%v_1``, the dry potential
    temperature over ``(p00/p_hyd)**(Rd/Cp)`` (:4113-4118).  Both
    arguments of the product and the quotient are REAL.

    ``temperature`` is the array, or a callable returning it, so the
    caller's temperature is built only when a cell is seeded.

    Returns ``(seeded, seed_mask, receipt)``; ``seeded`` equals
    ``number`` outside ``seed_mask``.
    """
    from woof.core.thompson_entry import (
        MAKE_RAIN_ICE_NUMBER_SOURCE, ice_mean_diameter_m, make_ice_number,
        make_rain_number, rain_median_volume_diameter_m,
    )

    function, diameter_of, rule = {
        "rain": (make_rain_number, rain_median_volume_diameter_m,
                 "qr > 0 and nr <= 0: nr = make_RainNumber(qr*rho, T) / rho"),
        "ice": (make_ice_number, ice_mean_diameter_m,
                "qi > 0 and ni <= 0: ni = make_IceNumber(qi*rho, T) / rho"),
    }[species]
    alt = np.asarray(_host(inverse_density), dtype=np.float32)
    # temp_rho = 1./grid%alt(i,k,j) is REAL (:4825).
    seed = (mass > np.float32(0.0)) & (number <= np.float32(0.0))
    seeded = np.array(number, dtype=np.float32)
    count = int(np.count_nonzero(seed))
    receipt = {"authority": MAKE_RAIN_ICE_NUMBER_SOURCE, "rule": rule,
               "seeded_cells": count}
    if count:
        if temperature is None:
            raise ValueError(
                f"Thompson cold start: {count} cell(s) carry {species} mass "
                "and no number, and real.exe's make_"
                f"{species.capitalize()}Number sizes them by temperature, "
                "but the closure was given none")
        if callable(temperature):
            temperature = temperature()
        rho_seed = (np.float32(1.0) / alt[seed]).astype(np.float32)
        per_mass = (mass[seed] * rho_seed).astype(np.float32)
        temp_seed = np.asarray(temperature, dtype=np.float32)[seed]
        per_volume = function(per_mass, temp_seed)
        seeded[seed] = (per_volume / rho_seed).astype(np.float32)
        if build_receipt:
            diameter = diameter_of(per_mass, per_volume)
            receipt.update({
                "number_per_m3_min": float(per_volume.min()),
                "number_per_m3_max": float(per_volume.max()),
                "diameter_um_min": float(diameter.min() * 1.0e6),
                "diameter_um_max": float(diameter.max() * 1.0e6),
                "diameter": ("median volume diameter" if species == "rain"
                             else "mean diameter 3/lambda"),
            })
            if species == "rain":
                receipt["supercooled_cells"] = int(np.count_nonzero(
                    temp_seed <= np.float32(271.15)))
    return seeded, seed, receipt


def _thompson_cold_start_moment_closure(state, state_xp, cfg: RunConfig,
                                        inverse_density, *,
                                        aerosol_number=None, landmask=None,
                                        temperature=None, temperature_at=None,
                                        temperature_fields=None, receipt=True):
    """Use the native cell path when host fields expose its exact inputs."""
    if state_xp is np and (temperature_fields is not None or temperature_at is None):
        from woof.ingest.cold_start_cpu import close_numbers
        native = close_numbers(
            state, cfg, inverse_density, aerosol_number=aerosol_number,
            landmask=landmask, temperature=temperature,
            temperature_fields=temperature_fields, receipt=receipt)
        if native is not None:
            return native
    return _thompson_cold_start_moment_closure_reference(
        state, state_xp, cfg, inverse_density, aerosol_number=aerosol_number,
        landmask=landmask, temperature=temperature,
        temperature_at=temperature_at, receipt=receipt)


def _thompson_cold_start_moment_closure_reference(state, state_xp, cfg: RunConfig,
                                        inverse_density, *,
                                        aerosol_number=None,
                                        landmask=None,
                                        temperature=None, temperature_at=None, receipt=True) -> dict[str, object]:
    """Close the scheme's number moments over the analyzed mass, once.

    Every number moment the scheme carries is first set as real.exe sets
    it where the mass is present and the number is not (WRF v4.7.1
    ``module_initialize_real.F:4829-4852``): cloud droplets from
    ``make_DropletNumber`` (:func:`_cold_start_droplet_number`, mp=28
    only, the one package with a droplet number), rain from
    ``make_RainNumber`` and ice from ``make_IceNumber``
    (:func:`_cold_start_rain_ice_number`, mp=8 and mp=28).  The entry
    block below then rediagnoses them.  A number the state already holds
    above zero, an installed analysed NC/NR/NI, is kept as real.exe keeps
    it.  Before the seed the entry block was handed zero numbers and
    returned its limits: 0.02 to 0.7 drops per cm3 at a mean diameter
    near 89 um for 0.01 to 0.3 g/kg of cloud water, 1 mm rain drops
    whatever the temperature, and 5 um ice capped at 999e3 per m3, where
    real.exe starts supercooled rain near 0.3 mm and sizes ice by
    temperature.

    The scheme's own entry block (:mod:`woof.core.thompson_entry`, the
    authority :func:`woof.da.moments.repair_moments` applies after an
    analysis) evaluated on the initial state at the density the
    initializer formed (``rho = 1/alt``), with the receipt the repair
    writes.  Only cells carrying mass above the scheme's activity gate
    with a number at or below zero are written; every other cell is left
    exactly as real.exe leaves it.  A written array is validated finite
    and non-negative before it reaches the state, and a state that fails
    that is a refusal, not a clip.  The block is called here rather than
    through the assimilation repair because this module ships in the
    preparation-only distribution and that repair's module does not.

    ``temperature`` is the array or a callable returning it; a callable
    is called at most once, and only when a rain or ice seed mask is
    non-empty, so a start with no seeded cell never builds it.
    """
    from woof.core.thompson_entry import (
        R1, THOMPSON_ENTRY_AUTHORITY, THOMPSON_ENTRY_SOURCE,
        cold_start_droplet_row_refusal, np_thompson_entry_numbers,
    )

    alt = np.asarray(_host(inverse_density), dtype=np.float32)
    # Widening a nonzero float32 to float64 makes even its smallest
    # subnormal reciprocal finite. Infinity reciprocates to finite zero.
    invalid_density = np.isnan(alt) | (alt == 0.0)
    if bool(invalid_density.any()):
        raise ValueError(
            "the initializer's inverse density is zero or not finite in "
            f"{int(np.count_nonzero(invalid_density))} cell(s), so the "
            "density Thompson's entry block works in cannot be formed there")
    species: list[dict[str, object]] = []
    written: list[str] = []
    total = 0
    seeds: dict[str, object] = {}
    built_temperature: list[object] = []

    def seed_temperature():
        if not built_temperature:
            built_temperature.append(
                temperature() if callable(temperature) else temperature)
        return built_temperature[0]

    # mp=8 carries no droplet number (its cloud number is the constant
    # Nt_c), so real.exe's P_QNC test is false there and only rain and ice
    # are seeded and closed.  The pairs are the table the lateral boundary
    # reads too (woof.boundary_fields.COLD_START_SEEDED_NUMBERS), so the
    # ring carries exactly the numbers this closure seeds.
    from woof.boundary_fields import COLD_START_SEEDED_NUMBERS
    kinds = {"qc": "cloud", "qr": "rain", "qi": "ice"}
    pairs = tuple((kinds[mass], mass, number) for mass, number
                  in COLD_START_SEEDED_NUMBERS[int(cfg.mp_physics)])
    for name, mass_field, number_field in pairs:
        mass_device = getattr(state, mass_field)
        number_device = getattr(state, number_field)
        # Gather the seed set once in flat order on the state's owner.
        if state_xp is np:
            seed = (mass_device > np.float32(0.0)) & (number_device <= np.float32(0.0))
        else:
            # Integer predicates retain FP32 subnormals on both cards.
            # Positive finite/infinite mass excludes both zeros and NaNs;
            # nonpositive numbers include both zeros and exclude NaNs.
            mbits = mass_device.view(state_xp.uint32)
            nbits = number_device.view(state_xp.uint32)
            positive = (mbits > 0) & (mbits <= state_xp.uint32(0x7f800000))
            nonpositive = ((nbits == 0)
                           | ((nbits >= state_xp.uint32(0x80000000))
                              & (nbits <= state_xp.uint32(0xff800000))))
            seed = positive & nonpositive
        seed_indices = state_xp.flatnonzero(seed.ravel())
        del seed
        indices = np.asarray(seed_indices.get() if hasattr(seed_indices, "get")
                             else seed_indices, dtype=np.int64)
        mass = _host_float32(mass_device.ravel()[seed_indices])
        number = _host_float32(number_device.ravel()[seed_indices])
        alt_seed = alt.ravel()[indices]
        offenders = mass > R1
        count = int(np.count_nonzero(offenders))
        entry = {
            "species": name, "mass_field": mass_field,
            "number_field": number_field, "offending_cells": count,
            "repaired_cells": 0,
        }
        if name == "cloud":
            if hasattr(aerosol_number, "get"):
                aerosol = _host_float32(state_xp.broadcast_to(
                    aerosol_number, mass_device.shape)[state_xp.unravel_index(
                        seed_indices, mass_device.shape)])
            else:
                aerosol = np.broadcast_to(
                    0.0 if aerosol_number is None else aerosol_number,
                    mass_device.shape).flat[indices]
            surface = (None if landmask is None else np.broadcast_to(
                np.asarray(_host(landmask)), mass_device.shape).flat[indices])
            seeded, seed_mask, seeds[name] = _cold_start_droplet_number(
                mass, number, alt_seed, aerosol, surface,
                build_receipt=receipt)
        else:
            selected_temperature = None
            if indices.size and temperature_at is not None:
                selected_temperature = temperature_at(indices)
            elif temperature is not None:
                selected_temperature = lambda: np.asarray(
                    seed_temperature(), dtype=np.float32).ravel()[indices]
            seeded, seed_mask, seeds[name] = _cold_start_rain_ice_number(
                name, mass, number, alt_seed, selected_temperature,
                build_receipt=receipt)
        if count or bool(seed_mask.any()):
            # Only the offender set reaches the entry arithmetic. Each
            # division and library call keeps its original operands/order.
            if count:
                density = 1.0 / alt_seed[offenders].astype(np.float64)
                if name == "cloud":
                    # The entry block's own gamma-table row comes from the
                    # droplet number per volume; a NaN there has no row.
                    with np.errstate(invalid="ignore", over="ignore"):
                        not_a_number = int(np.count_nonzero(np.isnan(
                            seeded[offenders].astype(np.float64) * density)))
                    if not_a_number:
                        raise ValueError(
                            cold_start_droplet_row_refusal(not_a_number))
                closed = np_thompson_entry_numbers(
                    name, mass[offenders], seeded[offenders], density)
                seeded[offenders] = np.asarray(closed, dtype=np.float32)
            fixed = number_device.copy()
            fixed.ravel()[seed_indices] = state_xp.asarray(
                seeded, dtype=state_xp.float32)
            if state_xp is np:
                negative = fixed < 0.0
            else:
                bits = fixed.view(state_xp.uint32)
                negative = ((bits > state_xp.uint32(0x80000000))
                            & (bits <= state_xp.uint32(0xff800000)))
            if not bool(state_xp.isfinite(fixed).all()) or bool(negative.any()):
                raise ValueError(
                    f"the cold-start moment closure produced a non-finite or "
                    f"negative {number_field}; the analyzed {mass_field} it "
                    "was closed over is not a state the scheme can start from")
            number_device[...] = fixed
            entry["repaired_cells"] = count
            if count and receipt:
                entry["repaired_number_min"] = float(seeded[offenders].min())
                entry["repaired_number_max"] = float(seeded[offenders].max())
            total += count
            written.append(number_field)
        species.append(entry)
    return {
        "schema": COLD_START_MOMENT_CLOSURE_SCHEMA,
        "droplet_number_seed": seeds.get("cloud"),
        "rain_number_seed": seeds["rain"],
        "ice_number_seed": seeds["ice"],
        "repaired": True,
        "repaired_cells_total": total,
        "authority": THOMPSON_ENTRY_AUTHORITY,
        "mp_physics": int(cfg.mp_physics),
        "q_threshold_kg_kg": float(R1),
        "species": species,
        "note": ("cells with mass and a number at or below zero first take "
                 "real.exe's make_DropletNumber, make_RainNumber and "
                 "make_IceNumber (droplet_number_seed, rain_number_seed, "
                 "ice_number_seed); cells with mass above the scheme's "
                 "activity threshold and a number moment at or below zero "
                 "were then written by the entry block; every other cell, "
                 "an analysed number above zero included, keeps the value "
                 "real.exe writes"),
        "written_state_fields": written,
        "density": "initializer moist specific volume, rho = 1/alt",
        "entry_block": THOMPSON_ENTRY_SOURCE
                       + " through woof.core.thompson_entry",
        "cells_without_mass": "exact FP32 zero, as real.exe writes them",
    }


def _mp28_aerosol_source_policy(cfg: RunConfig, state, *, receipt=True) -> dict[str, object]:
    """Receipt for the mp=28 half of the source-absent initialization.

    Emitted as ``RealInitResult.aerosol_initialization`` so that a
    consumer can PROVE, from the returned object alone, three things a
    later reader would otherwise have to take on trust: which aerosol
    selectors this ingest ran under, that the two aerosol fields left this
    function at exact FP32 zero, and which named function is expected to
    fill them next.  The fingerprints are the same
    :func:`array_correspondence_fingerprint` identity used for the mass
    species, so an "already filled" state and an "awaiting fill" state are
    distinguishable after the fact and not only in the moment.
    """
    if not receipt:
        return {}
    receipt = {
        **WRF_REAL_MP28_AEROSOL_SOURCE_POLICY,
        "aer_init_opt": int(cfg.aer_init_opt),
        "wif_input_opt": int(cfg.wif_input_opt),
        "awaiting_profile_fill": True,
    }
    # The four aerosol fields only: the three number moments are no longer
    # source-absent zeros once the cold-start closure has run over the
    # analyzed mass, and their receipt is
    # hydrometeor_initialization["cold_start_moment_closure"].
    absent_names = ("nwfa", "nifa", "nwfa2d", "nifa2d")
    receipt["source_absent_state_fields"] = dict(zip(
        absent_names,
        _fingerprints_side_by_side(getattr(state, name) for name in absent_names)))
    return receipt


def array_correspondence_fingerprint(value) -> dict[str, object]:
    """Return the byte/mask/extrema identity used by init receipts.

    The checksum covers exactly the contiguous array bytes.  Shape and dtype
    are separate fields, and the nonzero mask is independently packed and
    hashed so an all-zero replacement cannot masquerade as analyzed mass.
    """

    if hasattr(value, "get"):
        value = value.get()
    array = np.ascontiguousarray(np.asarray(value))
    if array.size == 0:
        raise ValueError("cannot fingerprint an empty correspondence array")
    from woof.ingest.preparation_fingerprints import array_fingerprint
    native = array_fingerprint(array)
    if native is not None:
        return native
    mask = np.packbits(
        np.ravel(array != 0.0), bitorder="little")
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "nonzero_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "nonzero_count": int(np.count_nonzero(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _canonical_receipt_sha256(value: Mapping[str, object]) -> str:
    def plain(item):
        if isinstance(item, Mapping):
            return {str(key): plain(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [plain(child) for child in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    encoded = json.dumps(
        plain(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _packed_mask_sha256(mask) -> str:
    packed = np.packbits(
        np.ravel(np.asarray(mask, dtype=np.bool_)), bitorder="little")
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _zero_byte_sha256(length: int) -> str:
    digest = hashlib.sha256()
    block = b"\0" * min(length, 1024 * 1024)
    complete, remainder = divmod(length, len(block)) if block else (0, 0)
    for _ in range(complete):
        digest.update(block)
    if remainder:
        digest.update(block[:remainder])
    return digest.hexdigest()


def _host_float32(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.ascontiguousarray(value, dtype=np.float32)


def _hrrr_operator_geometry(
        source_pressure, surface_pressure, target_pressure, active_mask):
    """Classify WRF's assembled Q-column support in exact FP32 geometry.

    ``active_mask`` is the union of positive samples across retained source
    species.  Assembly is field-independent; restricting target-stencil
    searches to that union avoids turning sparse HRRR cloud evidence into a
    dense ``nsource * ntarget * ny * nx`` temporary.  ``log_tie`` marks the
    active assembled samples whose support hinges on how a kernel rounds a
    logarithm.  The result is a prediction: the caller reads which samples
    the operator used from the operator itself, and uses this geometry to
    name why the others were left out and to count where the two differ.
    """

    source = _host_float32(source_pressure)
    surface = _host_float32(surface_pressure)
    target = _host_float32(target_pressure)
    active = np.ascontiguousarray(active_mask, dtype=np.bool_)
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError(
            "hydrometeor pressure geometry must be (level, y, x)")
    if (surface.shape != source.shape[1:]
            or target.shape[1:] != source.shape[1:]
            or active.shape != source.shape):
        raise ValueError("hydrometeor disposition geometry shapes differ")
    if (not np.isfinite(source).all()
            or not np.isfinite(surface).all()
            or not np.isfinite(target).all()
            or np.any(source <= 0.0)
            or np.any(surface <= 0.0)
            or np.any(target <= 0.0)):
        raise ValueError("hydrometeor disposition pressure is invalid")
    if np.any(np.diff(source, axis=0) >= 0.0):
        raise ValueError(
            "hydrometeor disposition source pressure is not descending")
    if np.any(np.diff(target, axis=0) >= 0.0):
        raise ValueError(
            "hydrometeor disposition target pressure is not descending")

    nsource, ny, nx = source.shape
    ncolumn = ny * nx
    pd = source.reshape(nsource, ncolumn)
    psfc = surface.reshape(ncolumn)
    target_column = target.reshape(target.shape[0], ncolumn)
    active_column = active.reshape(nsource, ncolumn)
    above = pd < psfc[None, :]
    if np.any(~np.any(above, axis=0)):
        raise ValueError(
            "hydrometeor disposition has no source level above surface")
    first_above = np.argmax(above, axis=0).astype(np.int32)
    level = np.arange(nsource, dtype=np.int32)[:, None]
    force_candidate = (
        (level >= first_above[None, :])
        & (pd <= target_column[0][None, :]))
    has_force_candidate = np.any(force_candidate, axis=0)
    first_force_candidate = np.argmax(force_candidate, axis=0)
    knext = np.where(
        has_force_candidate, first_force_candidate, first_above).astype(
            np.int32)

    force_code = _HRRR_DISPOSITION_CLASSES[
        "WRF_FORCE_SURFACE_EXCLUDED"]
    zap_code = _HRRR_DISPOSITION_CLASSES["WRF_ZAP_CLOSE_EXCLUDED"]
    # 255 is deliberately invalid.  Assembly must replace every cell with
    # included/no-target-yet (0), force exclusion (2), or zap exclusion (3).
    operator_class = np.full((nsource, ncolumn), 255, dtype=np.uint8)
    has_below_ground = first_above > 0
    surface_lowest = ~has_below_ground
    last_surface_branch_pressure = psfc.copy()
    for k in range(nsource):
        pressure = pd[k]

        below = has_below_ground & (k < first_above)
        close_below = (
            below & (k == first_above - 1)
            & ((pressure - psfc) < np.float32(500.0)))
        operator_class[k, below & ~close_below] = 0
        operator_class[k, close_below] = zap_code

        above_surface = has_below_ground & (k >= first_above)
        forced = above_surface & (k < knext)
        close_above = (
            above_surface & (k == knext)
            & ((psfc - pressure) < np.float32(500.0)))
        operator_class[k, above_surface & ~forced & ~close_above] = 0
        operator_class[k, forced] = force_code
        operator_class[k, close_above] = zap_code

        forced_lowest = surface_lowest & (k < knext)
        candidate_lowest = surface_lowest & (k >= knext)
        close_lowest = (
            candidate_lowest
            & ((last_surface_branch_pressure - pressure)
               < np.float32(500.0))
            & (k < nsource - 1))
        accepted_lowest = candidate_lowest & ~close_lowest
        operator_class[k, accepted_lowest] = 0
        operator_class[k, forced_lowest] = force_code
        operator_class[k, close_lowest] = zap_code
        last_surface_branch_pressure = np.where(
            accepted_lowest, pressure, last_surface_branch_pressure)

    if np.any(operator_class == 255):
        raise AssertionError(
            "WRF hydrometeor column assembly left an unclassified level")

    # All hydrometeor targets are in WRF's linear branch (vboundb=ntarget+1),
    # so an assembled source point carries weight to exactly the targets
    # strictly between its two assembled neighbours, and the deepest point
    # also to every target below it by constant extrapolation.  In exact
    # arithmetic that is a pressure comparison, made here on the FP32
    # pressures themselves.  The production kernels compare FP32 logarithms
    # instead, and each kernel's logarithm is its own (the C library's logf
    # on the CPU bridge, CUDA's logf on the GPU), so where a target sits
    # within an ulp or two of a neighbour's logarithm the kernel's rounding,
    # not the pressure, decides whether the sample reaches it.  Those samples
    # are returned in ``log_tie``; only the kernel can say which way they go.
    # Below the deepest assembled point the lower neighbour is +inf, above
    # the top one the upper neighbour is 0: every target is on their side.
    below_pressure = np.full((nsource, ncolumn), np.inf, dtype=np.float32)
    neighbour = np.full(ncolumn, np.inf, dtype=np.float32)
    for k in range(nsource):
        neighbour = np.where(first_above == k, psfc, neighbour)
        below_pressure[k] = neighbour
        neighbour = np.where(operator_class[k] == 0, pd[k], neighbour)
    tie_band = _logarithm_tie_band(source, surface, target)
    log_tie = np.zeros((nsource, ncolumn), dtype=np.bool_)
    neighbour = np.zeros(ncolumn, dtype=np.float32)
    for k in range(nsource - 1, -1, -1):
        columns = np.flatnonzero(
            active_column[k] & (operator_class[k] == 0))
        for start in range(0, columns.size, _Q_STENCIL_COLUMN_CHUNK):
            chunk = columns[start:start + _Q_STENCIL_COLUMN_CHUNK]
            decided, reachable = _q_stencil_support(
                target_column[:, chunk], below_pressure[k, chunk],
                neighbour[chunk], tie_band)
            operator_class[k, chunk[decided]] = 1
            log_tie[k, chunk[reachable & ~decided]] = True
        neighbour = np.where(operator_class[k] <= 1, pd[k], neighbour)
        neighbour = np.where(first_above == k, psfc, neighbour)

    return {
        "source_pressure": source,
        "surface_pressure": surface,
        "target_pressure": target,
        "first_above": first_above.reshape(ny, nx),
        "knext": knext.reshape(ny, nx),
        "operator_class": operator_class.reshape(source.shape),
        "log_tie": log_tie.reshape(source.shape),
        "logarithm_tie_band": tie_band,
    }


#: Columns per block when a source level's targets are compared with its two
#: assembled neighbours; bounds the comparison temporaries.
_Q_STENCIL_COLUMN_CHUNK = 65536


def _logarithm_tie_band(*pressures) -> float:
    """Relative pressure gap inside which two FP32 logarithms can tie or cross.

    Each production logarithm is within one FP32 ulp of the exact value
    (CUDA documents logf at 1 ulp; the C library's logf is tighter), counted
    in the ulp of the result's binade, which is at most twice the ulp of the
    largest logarithm magnitude present.  Two logarithms can therefore only
    tie or cross when the exact ones are within four of those ulps.  Since
    ``|ln(a) - ln(b)| >= |a - b| / max(a, b)``, a pressure pair whose gap
    exceeds that many ulps times the larger pressure is ordered the same way
    by every such kernel as by the pressures themselves.
    """

    lowest = min(float(np.min(value)) for value in pressures)
    highest = max(float(np.max(value)) for value in pressures)
    extent = max(abs(pm.log(lowest)), abs(pm.log(highest)))
    return 4.0 * float(np.spacing(np.float32(extent)))


def _q_stencil_support(target, below, above, tie_band):
    """Which assembled samples reach a target, and which might.

    ``target`` is (ntarget, n) FP32; ``below`` and ``above`` are the
    pressures of each sample's assembled neighbours (higher and lower
    pressure), +inf under the deepest point and 0 over the top one.  A
    sample reaches the targets strictly between its neighbours.
    ``decided`` is support that holds under every production logarithm: a
    target inside the interval and farther than the tie band from both
    ends.  ``reachable`` adds the samples whose only candidate targets lie
    within the band of an end, other than exactly on it, where every
    kernel takes the same logarithm and the sample's weight is exactly 0.
    The band is a few thousandths of a percent of the pressure, far wider
    than the half-ulp rounding of the FP32 thresholds below.
    """

    shrink = 1.0 - tie_band
    lower = np.asarray(below, dtype=np.float64)
    upper = np.asarray(above, dtype=np.float64)
    t = np.asarray(target, dtype=np.float32)
    decided = np.any(
        (t > (upper / shrink).astype(np.float32))
        & (t < (lower * shrink).astype(np.float32)), axis=0)
    reachable = decided.copy()
    undecided = np.flatnonzero(~decided)
    if undecided.size:
        candidate = t[:, undecided]
        lower_end = lower[undecided]
        upper_end = upper[undecided]
        reachable[undecided] = np.any(
            (candidate >= (upper_end * shrink).astype(np.float32))
            & (candidate <= (lower_end / shrink).astype(np.float32))
            & (candidate != upper_end.astype(np.float32))
            & (candidate != lower_end.astype(np.float32)), axis=0)
    return decided, reachable


_HRRR_PRESSURE_PARTITION_SCHEMA = (
    "gpuwm-wrf-q-pressure-partition-crosscheck-v1")
_HRRR_PRESSURE_PARTITION_FIELDS = frozenset({
    "schema", "support_authority", "logarithm_tie_band",
    "active_sample_count", "logarithm_tie_sample_count",
    "logarithm_tie_supported_count", "disagreeing_sample_count",
    "geometry_only_count", "production_only_count",
    "disagreement_mask_sha256", "disagreements"})


def _pressure_partition_crosscheck(
        geometry, active, production_support, order) -> dict[str, object]:
    """Count where the pressure geometry and the operator disagree.

    ``active`` and ``production_support`` are in WRF's ordered source-level
    order.  A sample in the logarithm tie band is not a disagreement: the
    geometry says only that the kernel's rounding decides it.  Every other
    active sample whose predicted support differs from the operator's is
    counted, and the first few are listed in decoded-source order so a
    reader can find the column.  Nothing here can refuse.
    """

    predicted = geometry["operator_class"] == 1
    log_tie = active & geometry["log_tie"]
    disagreement = active & ~log_tie & (predicted != production_support)
    raw_disagreement = np.zeros_like(disagreement)
    raw_disagreement[order] = disagreement
    records = []
    for flat_index in np.flatnonzero(raw_disagreement.ravel())[
            :_HRRR_DISPOSITION_EXAMPLE_LIMIT]:
        raw_level, y, x = (int(value) for value in np.unravel_index(
            int(flat_index), raw_disagreement.shape))
        ordered_level = int(np.flatnonzero(order == raw_level)[0])
        column = geometry["target_pressure"][:, y, x]
        pressure = float(geometry["source_pressure"][ordered_level, y, x])
        nearest = int(np.argmin(np.abs(
            column.astype(np.float64) - pressure)))
        assembly = int(geometry["operator_class"][ordered_level, y, x])
        records.append({
            "raw_source_index": [raw_level, y, x],
            "wrf_ordered_source_level": ordered_level,
            "geometry_support": bool(predicted[ordered_level, y, x]),
            "production_support": bool(
                production_support[ordered_level, y, x]),
            "column_assembly": (
                _HRRR_DISPOSITION_CLASS_BY_CODE[assembly]
                if assembly >= 2 else "ASSEMBLED"),
            "source_pressure_pa": pressure,
            "surface_pressure_pa": float(
                geometry["surface_pressure"][y, x]),
            "nearest_target_index": nearest,
            "nearest_target_pressure_pa": float(column[nearest]),
        })
    return {
        "schema": _HRRR_PRESSURE_PARTITION_SCHEMA,
        "support_authority": "production-operator-replay",
        "logarithm_tie_band": float(geometry["logarithm_tie_band"]),
        "active_sample_count": int(np.count_nonzero(active)),
        "logarithm_tie_sample_count": int(np.count_nonzero(log_tie)),
        "logarithm_tie_supported_count": int(np.count_nonzero(
            log_tie & production_support)),
        "disagreeing_sample_count": int(np.count_nonzero(disagreement)),
        "geometry_only_count": int(np.count_nonzero(
            disagreement & predicted)),
        "production_only_count": int(np.count_nonzero(
            disagreement & production_support)),
        "disagreement_mask_sha256": _packed_mask_sha256(raw_disagreement),
        "disagreements": records,
    }


def _hrrr_disposition_example(
        *, raw_field, raw_labels, order, ordered_level, y, x,
        geometry, influencing_targets) -> dict[str, object]:
    """Return inspectable pressure/stencil evidence for one source sample.

    ``influencing_targets`` are the target indices the production replay of
    this sample's level wrote in its column, so a record states what the
    kernel did rather than a re-derivation that could round differently.
    """

    raw_level = int(order[ordered_level])
    source = geometry["source_pressure"][:, y, x]
    surface = float(geometry["surface_pressure"][y, x])
    target = geometry["target_pressure"][:, y, x]
    included = geometry["operator_class"][:, y, x] <= 1
    first_above = int(geometry["first_above"][y, x])
    assembled: list[tuple[str, int | None, float]] = []
    for level in range(source.size):
        if level == first_above:
            assembled.append(("surface", None, surface))
        if included[level]:
            assembled.append(("source", level, float(source[level])))
    position = next(
        (index for index, item in enumerate(assembled)
         if item[0] == "source" and item[1] == ordered_level), None)

    def point(index):
        if index is None or index < 0 or index >= len(assembled):
            return None
        kind, level, pressure = assembled[index]
        return {
            "kind": kind,
            "wrf_ordered_source_level": level,
            "pressure_pa": pressure,
        }

    code = int(raw_labels[raw_level, y, x])
    return {
        "class": _HRRR_DISPOSITION_CLASS_BY_CODE[code],
        "raw_source_index": [raw_level, int(y), int(x)],
        "raw_flat_index": int(np.ravel_multi_index(
            (raw_level, y, x), raw_labels.shape)),
        "wrf_ordered_source_level": int(ordered_level),
        "source_value": float(raw_field[raw_level, y, x]),
        "source_pressure_pa": float(source[ordered_level]),
        "surface_pressure_pa": surface,
        "force_target_pressure_pa": float(target[0]),
        "target_pressure_minimum_pa": float(np.min(target)),
        "target_pressure_maximum_pa": float(np.max(target)),
        "previous_assembled_point": (
            point(position - 1) if position is not None else None),
        "next_assembled_point": (
            point(position + 1) if position is not None else None),
        "influencing_target_indices": influencing_targets,
    }


_RECEIPT_HASH_MIN_CELLS = 1_000_000


class _ReceiptHostArray(np.ndarray):
    """Pinned bytes with reductions made by the array's device owner."""


def _receipt_host_float32(value):
    reductions = None
    validation = column_support = None
    if hasattr(value, "get"):
        if not type(value).__module__.startswith("cupy"):
            return _host_float32(value)
        if value.size == 0:
            return np.empty(value.shape, dtype=np.float32)
        # Pinned storage lets the next replay copy overlap a worker's hash.
        import cupy as cp
        if value.dtype == np.dtype(np.float32) and value.size:
            magnitude_bits = value.view(cp.uint32) & cp.uint32(0x7fffffff)
            nonzero = magnitude_bits != 0
            subnormal = bool(cp.any(nonzero & (magnitude_bits < cp.uint32(0x00800000))))
            negative_zero = bool(cp.any(value.view(cp.uint32) == cp.uint32(0x80000000)))
            finite = not bool(cp.any(magnitude_bits >= cp.uint32(0x7f800000)))
            bits = value.view(cp.uint32)
            nonnegative = not bool(cp.any((bits > cp.uint32(0x80000000))
                                          & (bits <= cp.uint32(0xff800000))))
            validation = finite and nonnegative
            if value.ndim == 3:
                column_support = cp.any(nonzero, axis=0).get()
            packed = cp.packbits(nonzero.ravel(), bitorder="little")
            reductions = {
                "nonzero_mask_sha256": hashlib.sha256(packed.get().tobytes()).hexdigest(),
                "nonzero_count": int(cp.count_nonzero(nonzero)),
                "minimum": float(value.min()),
                "maximum": float(value.max()),
                "extrema_on_host": subnormal,
                "zero_extrema_on_host": negative_zero,
            }
            del magnitude_bits, nonzero, packed
        storage = cp.cuda.alloc_pinned_memory(value.nbytes)
        host = np.frombuffer(storage, dtype=value.dtype, count=value.size).reshape(value.shape)
        value.get(out=host)
        value = host
    result = np.ascontiguousarray(value, dtype=np.float32)
    if reductions is not None:
        result = result.view(_ReceiptHostArray)
        result.receipt_reductions = reductions
        result.receipt_valid = validation
        result.receipt_column_support = column_support
    return result


def _receipt_array_fingerprint(value):
    reductions = getattr(value, "receipt_reductions", None)
    if reductions is None:
        return array_correspondence_fingerprint(value)
    array = np.ascontiguousarray(value)
    reductions = dict(reductions)
    extrema_on_host = reductions.pop("extrema_on_host")
    zero_extrema_on_host = reductions.pop("zero_extrema_on_host")
    # Signed-zero ties and non-finite extrema retain NumPy's exact reduction
    # order. A finite nonzero extremum has only one floating representation.
    for name, operation in (("minimum", np.min), ("maximum", np.max)):
        if (extrema_on_host or not np.isfinite(reductions[name])
                or (zero_extrema_on_host and reductions[name] == 0.0)):
            reductions[name] = float(operation(array))
    from woof.ingest.preparation_fingerprints import array_sha256
    digest = array_sha256(array)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": (digest if digest is not None else
                       hashlib.sha256(array.tobytes(order="C")).hexdigest()),
            **reductions}


def _receipt_mask_float32(mask):
    """``np.asarray(mask, dtype=np.float32)`` for a boolean replay mask,
    widened in slabs side by side (the same 0/1 floats).

    Breakage it removes: a CPU-prepared 1 km child (678 x 534 x 50) widened
    its 18 M-cell mask once per source level on the receipt thread, 1.4 s of
    the receipt the start joins.
    """
    mask = np.asarray(mask)
    if mask.size < _RECEIPT_HASH_MIN_CELLS or not mask.flags.c_contiguous:
        return np.asarray(mask, dtype=np.float32)
    out = np.empty(mask.shape, dtype=np.float32)
    flat_in, flat_out = mask.reshape(-1), out.reshape(-1)
    step = 1 << 21
    from os import cpu_count

    def widen(start):
        flat_out[start:start + step] = flat_in[start:start + step]
    with ThreadPoolExecutor(max_workers=max(1, min(16, cpu_count() or 1)),
                            thread_name_prefix="receipt-mask") as pool:
        for _ in pool.map(widen, range(0, flat_in.size, step)):
            pass
    return out


def _receipt_nonnegative_finite(value):
    valid = getattr(value, "receipt_valid", None)
    if valid is not None:
        return bool(valid)
    array = np.asarray(value)
    if array.size < _RECEIPT_HASH_MIN_CELLS or not array.flags.c_contiguous:
        return bool(np.isfinite(array).all()) and not bool(np.any(array < 0.0))
    # A host replay output is one CONUS volume per source level: the two
    # whole-array passes took 6.3 s of the receipt thread on one core
    # (bounded CUDA road, a 5090 host, 2026-10-21).  Chunks answer the same
    # question side by side (NumPy releases the GIL); the verdict is an AND.
    flat = array.reshape(-1)
    step = 1 << 22
    chunks = [flat[start:start + step] for start in range(0, flat.size, step)]

    def ok(chunk):
        return bool(np.isfinite(chunk).all()) and not bool(np.any(chunk < 0.0))
    from os import cpu_count
    with ThreadPoolExecutor(max_workers=max(1, min(16, cpu_count() or 1)),
                            thread_name_prefix="receipt-check") as pool:
        return all(pool.map(ok, chunks))


def _fingerprints_side_by_side(values) -> list:
    """``[array_correspondence_fingerprint(v) for v in values]``, hashed side by side.

    Each array is still one ordered SHA-256 stream (so every receipt is the
    one the serial call writes); independent arrays run on separate native
    workers instead of one after another on the calling thread.
    """
    arrays = []
    for value in values:
        if hasattr(value, "get"):
            value = value.get()
        arrays.append(np.ascontiguousarray(np.asarray(value)))
    if len(arrays) < 2 or any(array.size == 0 for array in arrays):
        return [array_correspondence_fingerprint(array) for array in arrays]
    from os import cpu_count
    from woof.ingest.preparation_fingerprints import array_fingerprints
    receipts = array_fingerprints(arrays, workers=max(1, min(len(arrays), cpu_count() or 1)))
    if receipts is None:
        return [array_correspondence_fingerprint(array) for array in arrays]
    return receipts


class _ReceiptHasher:
    def __init__(self, pool, depth=2):
        self.pool = pool
        self.pending = []
        self.depth = max(1, int(depth))

    def submit(self, value):
        from concurrent.futures import Future
        # Bound live host arrays to two pending hashes, rather than retaining
        # all source-level outputs until the receipt has been assembled.
        if len(self.pending) >= self.depth:
            self.pending.pop(0).result()
        if hasattr(value, "get"):
            value = _receipt_host_float32(value)
        if self.pool is None:
            job = Future()
            job.set_result(_receipt_array_fingerprint(value))
        else:
            job = self.pool.submit(_receipt_array_fingerprint, value)
        self.pending.append(job)
        return job


#: CUDA runtime statuses that mean "no card here", not a device fault.
#: 100 cudaErrorNoDevice: the card is masked (CUDA_VISIBLE_DEVICES empty, the
#: CPU preprocess backend).  35 cudaErrorInsufficientDriver: a host with cupy
#: installed and no NVIDIA driver at all, where the backend choice already fell
#: back to the CPU ("cupy is installed but CUDA could not start").  Breakage
#: the 35 prevents: every HRRR prepare on a CPU-only host (the WOOF bake's
#: data stage, 2026-10-22 dry bake prepare_la-basin-750m) died in this receipt
#: job after the whole CPU preparation had run.
_NO_CARD_STATUSES = frozenset({100, 35})


def _thread_holds_cuda_context(cupy) -> bool:
    """Whether the calling thread already has a CUDA context current.

    ``cuCtxGetCurrent`` reads the thread's context without creating one, so
    asking never touches the card.  A driver that cannot answer (not
    initialised, absent) means there is no context here.
    """
    driver = getattr(cupy.cuda, "driver", None)
    if driver is None:
        return False
    try:
        return bool(driver.ctxGetCurrent())
    except driver.CUDADriverError:
        return False


def _start_receipt_job(function, *args, **kwargs):
    """Run one receipt builder on its own thread, on this thread's card.

    The HRRR hydrometeor disposition replays the vertical operator once per
    source level and hashes every replay: 59-65 s of a 120 s full HRRR CONUS
    start (1797x1057x50 from RAP native, 2026-10-16) at about two cores with
    the card idle, all of it before the rest of the start could proceed.
    Its inputs are read-only from here on, so it runs beside the remaining
    stages and the caller joins it with ``result()``.
    """
    import threading
    from concurrent.futures import Future
    device = None
    cupy = sys.modules.get("cupy")
    if cupy is not None:
        try:
            device = cupy.cuda.runtime.getDevice()
        except cupy.cuda.runtime.CUDARuntimeError as exc:
            if exc.status not in _NO_CARD_STATUSES:
                raise
            # No card on this thread (the CPU preprocess backend masks it;
            # a host without one): the receipt runs on the host, as the
            # whole preparation does there.  Asking unconditionally refused
            # every CPU-backend start with cudaErrorNoDevice.
            device = None
        if device is not None and not _thread_holds_cuda_context(cupy):
            # cudaGetDevice answers 0 on any box with a visible card, context
            # or not, so it cannot tell a CUDA-backend caller from a CPU one.
            # Binding that answer on the job thread (Device.use, i.e.
            # cudaSetDevice) is what stood a ~386 MiB primary context up on
            # the card in every CPU-backend HRRR start (U-E7: CPU prep held a
            # context on each card unless CUDA_VISIBLE_DEVICES was emptied).
            # A caller with no current context has no card to hand on.
            device = None
    binding = pm.current_cpu_bridge_binding()
    job = Future()

    def run():
        try:
            if device is not None:
                sys.modules["cupy"].cuda.Device(device).use()
            if binding is None:
                value = function(*args, **kwargs)
            else:
                # The caller's worker cap, not a private worker's one thread:
                # this is one job beside its caller, and its CPU replays are
                # the start's critical path (run_with_shared_cpu_bridge_binding).
                value = pm.run_with_shared_cpu_bridge_binding(
                    binding, function, *args, **kwargs)
        except BaseException as error:  # re-raised by the caller's result()
            job.set_exception(error)
        else:
            job.set_result(value)

    threading.Thread(target=run, name="gpuwm-receipt", daemon=True).start()
    return job


def build_hrrr_hydrometeor_vertical_disposition(
        source_fields: Mapping[str, object], source_level_order,
        source_pressure, surface_pressure, target_pressure,
        initialized_fields: Mapping[str, object], *,
        operator_replay) -> dict[str, object]:
    args = (source_fields, source_level_order, source_pressure,
            surface_pressure, target_pressure, initialized_fields)
    count = max((int(value.size) for value in source_fields.values()), default=0)
    if count < _RECEIPT_HASH_MIN_CELLS:
        return _build_hrrr_hydrometeor_vertical_disposition(
            *args, operator_replay=operator_replay, hasher=_ReceiptHasher(None))
    # One SHA-256 stream per replay, up to eight replays hashing side by
    # side.  Two workers left the receipt waiting on its hashes (11.8 s of
    # hashing at HRRR CONUS, 1797x1057x50 from RAP native, which held the
    # start in its join); the pending host copies stay bounded by the pool.
    from os import cpu_count
    workers = max(2, min(8, cpu_count() or 2))
    with _column_pool(workers) as pool:
        return _build_hrrr_hydrometeor_vertical_disposition(
            *args, operator_replay=operator_replay,
            hasher=_ReceiptHasher(pool, depth=workers))


def _build_hrrr_hydrometeor_vertical_disposition(
        source_fields: Mapping[str, object], source_level_order,
        source_pressure, surface_pressure, target_pressure,
        initialized_fields: Mapping[str, object], *,
        operator_replay, hasher) -> dict[str, object]:
    """Build exhaustive, source-ordered WRF Q-support evidence.

    Every exact nonzero decoded FP32 source sample receives exactly one class:
    target-influencing, one of WRF's two column-assembly exclusions, or one
    of three explicit reasons an assembled sample has no target stencil.
    Labels are stored in decoded-source order and compressed losslessly; their
    nonzero mask must therefore reproduce the decoded field's independently
    published mask checksum.

    Which samples influence a target is read from the production operator,
    one binary replay per source level, never from the pressure geometry.
    This receipt is written after the initial state exists and describes it;
    it does not decide it.  The geometry names why an unused sample was left
    out, and its own prediction of the support is kept as a cross-check
    whose disagreements with the operator are counted in the receipt
    (``geometry.pressure_partition``).  A cloudy column whose target sits
    within a logarithm ulp of a source level or of the surface is decided by
    the kernel's own ``logf``, which no host-side prediction can reproduce
    on every backend, and refusing a valid state over that would make
    preparation fail by weather.
    """

    if (not source_fields or set(source_fields) != set(initialized_fields)
            or not callable(operator_replay)):
        raise ValueError(
            "hydrometeor disposition source/state/replay inventory differs")
    order = np.ascontiguousarray(source_level_order, dtype=np.int32)
    first_shape = tuple(next(iter(source_fields.values())).shape)
    if (len(first_shape) != 3 or order.shape != (first_shape[0],)
            or not np.array_equal(np.sort(order), np.arange(first_shape[0]))
            or any(tuple(value.shape) != first_shape
                   for value in source_fields.values())):
        raise ValueError("hydrometeor disposition source inventory is invalid")
    active = np.zeros(first_shape, dtype=np.bool_)
    raw_positive = {}
    for name in sorted(source_fields):
        raw = _receipt_host_float32(source_fields[name])
        if not _receipt_nonnegative_finite(raw):
            raise ValueError(
                f"hydrometeor disposition source {name} is invalid")
        raw_positive[name] = raw != 0.0
        active |= _ordered_levels(raw, order) != 0.0
    geometry = _hrrr_operator_geometry(
        source_pressure, surface_pressure, target_pressure, active)
    if geometry["source_pressure"].shape != first_shape:
        raise ValueError(
            "hydrometeor disposition pressure/source shapes differ")
    if np.any(~np.isin(
            geometry["operator_class"], np.asarray([0, 1, 2, 3],
                                                   dtype=np.uint8))):
        raise AssertionError(
            "WRF hydrometeor operator geometry is unclassified")

    # Establish the exact production support of each source sample from the
    # operator that wrote the state.  Vertical interpolation is
    # column-local, linear, and nonnegative for this Q path, so replaying one
    # ordered source level of binary ones across every column identifies that
    # level's support exactly: a source cell participates iff any target in
    # its column is exact nonzero.  The nsource replays are shared by every
    # retained hydrometeor species.  Each replay also records, per species,
    # the targets it wrote for the first supported samples of that level in
    # raw column order: the TARGET_INFLUENCING example records are the first
    # of those in raw flat order, and they cite what the kernel wrote.
    production_support = np.zeros(first_shape, dtype=np.bool_)
    level_replays = []
    influencing_targets: dict[tuple[str, int, int], list[int]] = {}
    level_input = np.zeros(first_shape, dtype=np.bool_)
    for ordered_level in range(first_shape[0]):
        level_input[ordered_level] = True
        level_output = _receipt_host_float32(operator_replay(level_input))
        level_input[ordered_level] = False
        if (level_output.shape != geometry["target_pressure"].shape
                or not _receipt_nonnegative_finite(level_output)):
            raise ValueError(
                "hydrometeor disposition source-level production replay "
                f"failed at ordered level {ordered_level}")
        level_support = getattr(level_output, "receipt_column_support", None)
        if level_support is None:
            level_support = np.any(level_output != 0.0, axis=0)
        production_support[ordered_level] = level_support
        raw_level = int(order[ordered_level])
        written = level_output.reshape(level_output.shape[0], -1)
        for name, positive_mask in raw_positive.items():
            for column in np.flatnonzero(
                    (positive_mask[raw_level] & level_support).ravel())[
                        :_HRRR_DISPOSITION_EXAMPLE_LIMIT]:
                influencing_targets[(name, raw_level, int(column))] = [
                    int(index) for index in
                    np.flatnonzero(written[:, column] != 0.0)]
        level_replays.append({
            "ordered_source_level": ordered_level,
            "raw_source_level": raw_level,
            "supported_column_count": int(np.count_nonzero(level_support)),
            "support_mask_sha256": _packed_mask_sha256(level_support),
            "output": hasher.submit(level_output),
        })
        del level_output, level_support, written
    for replay in level_replays:
        replay["output"] = replay["output"].result()
    del level_input, raw_positive
    pressure_partition_receipt = _pressure_partition_crosscheck(
        geometry, active, production_support, order)
    del active

    raw_production_support = np.zeros_like(production_support)
    raw_production_support[order] = production_support
    packed_production_support = np.packbits(
        raw_production_support.ravel(), bitorder="little")
    production_support_receipt = {
        "schema": _HRRR_PRODUCTION_SUPPORT_SCHEMA,
        "shape": list(first_shape),
        "encoding": _HRRR_PRODUCTION_SUPPORT_ENCODING,
        "mask_base64": base64.b64encode(zlib.compress(
            packed_production_support.tobytes(), level=9)).decode("ascii"),
        "mask_sha256": hashlib.sha256(
            packed_production_support.tobytes()).hexdigest(),
        "support_count": int(np.count_nonzero(raw_production_support)),
        "level_replays": level_replays,
    }
    del raw_production_support, packed_production_support

    pressure_fingerprints = _fingerprints_side_by_side(
        (geometry["source_pressure"], geometry["surface_pressure"],
         geometry["target_pressure"], order))
    geometry_receipt = {
        "source_pressure": pressure_fingerprints[0],
        "surface_pressure": pressure_fingerprints[1],
        "target_pressure": pressure_fingerprints[2],
        "source_level_order": pressure_fingerprints[3],
        "source_level_order_values": order.tolist(),
        "production_target_support": production_support_receipt,
        "pressure_partition": pressure_partition_receipt,
    }
    species_receipts = {}
    for name in sorted(source_fields):
        raw = _receipt_host_float32(source_fields[name])
        ordered = _ordered_levels(raw, order)
        positive = ordered != 0.0
        labels = np.zeros(first_shape, dtype=np.uint8)
        operator_class = geometry["operator_class"]
        # The operator's own support decides TARGET_INFLUENCING.  A sample
        # it did not use keeps the geometry's reason: a column-assembly
        # exclusion, or one of the three no-stencil reasons below.
        target_influencing = positive & production_support
        unused = positive & ~production_support
        excluded = unused & (operator_class >= 2)
        labels[excluded] = operator_class[excluded]
        labels[target_influencing] = _HRRR_DISPOSITION_CLASSES[
            "TARGET_INFLUENCING"]
        unsupported = unused & (operator_class <= 1)
        labels[
            unsupported
            & (geometry["source_pressure"]
               >= geometry["surface_pressure"][None, :, :])
        ] = _HRRR_DISPOSITION_CLASSES[
            "WRF_BELOW_GROUND_OUTSIDE_TARGET_SUPPORT"]
        labels[
            unsupported
            & (geometry["source_pressure"]
               < np.min(geometry["target_pressure"], axis=0)[None, :, :])
        ] = _HRRR_DISPOSITION_CLASSES[
            "WRF_ABOVE_TARGET_TOP_OUTSIDE_TARGET_SUPPORT"]
        labels[unsupported & (labels == 0)] = _HRRR_DISPOSITION_CLASSES[
            "WRF_NO_TARGET_STENCIL"]
        if np.any(positive & (labels == 0)) or np.any(~positive & (labels != 0)):
            raise AssertionError(
                f"hydrometeor disposition partition failed for {name}")

        raw_labels = np.zeros_like(labels)
        raw_labels[order] = labels
        encoded = zlib.compress(raw_labels.tobytes(order="C"), level=9)
        counts = {}
        mask_hashes = {}
        examples = {}
        for class_name, code in _HRRR_DISPOSITION_CLASSES.items():
            class_mask = raw_labels == code
            count = int(np.count_nonzero(class_mask))
            counts[class_name] = count
            mask_hashes[class_name] = _packed_mask_sha256(class_mask)
            records = []
            flat_indices = np.flatnonzero(class_mask.ravel())[
                :_HRRR_DISPOSITION_EXAMPLE_LIMIT]
            for flat_index in flat_indices:
                raw_level, y, x = np.unravel_index(
                    int(flat_index), first_shape)
                ordered_level = int(np.flatnonzero(order == raw_level)[0])
                # A sample outside production support wrote no target.
                written_targets = []
                if class_name == "TARGET_INFLUENCING":
                    written_targets = influencing_targets[(
                        name, int(raw_level),
                        int(y) * first_shape[2] + int(x))]
                records.append(_hrrr_disposition_example(
                    raw_field=raw, raw_labels=raw_labels, order=order,
                    ordered_level=ordered_level, y=int(y), x=int(x),
                    geometry=geometry,
                    influencing_targets=written_targets))
            examples[class_name] = {
                "complete": count <= _HRRR_DISPOSITION_EXAMPLE_LIMIT,
                "total_count": count,
                "records": records,
            }
        source_fingerprint_job = hasher.submit(raw)
        live_fingerprint_job = hasher.submit(
            initialized_fields[name])
        # The per-level support composes: Q interpolation is linear with
        # nonnegative weights and an exact-zero surface pseudo-level, so a
        # binary replay of every excluded sample is identically zero and
        # replaying only TARGET_INFLUENCING samples is byte-equal to
        # replaying the complete decoded nonzero mask.  The callback uses
        # the very same prepared backend plan/options as the analyzed field.
        # Both facts are recorded as observed.  Neither is a property of
        # the initial state, which the operator has already written, so an
        # operator that ever broke them would be reported by this receipt
        # rather than cost a valid preparation.
        source_replay = _receipt_host_float32(operator_replay(labels != 0))
        source_replay_fingerprint_job = hasher.submit(source_replay)
        live_fingerprint = live_fingerprint_job.result()
        if (list(source_replay.shape) != live_fingerprint["shape"]
                or not _receipt_nonnegative_finite(source_replay)):
            raise ValueError(
                f"hydrometeor disposition operator replay failed for {name}")
        del source_replay
        influencing_replay = _receipt_host_float32(
            operator_replay(
                labels == _HRRR_DISPOSITION_CLASSES[
                    "TARGET_INFLUENCING"]))
        if (list(influencing_replay.shape) != live_fingerprint["shape"]
                or not _receipt_nonnegative_finite(influencing_replay)):
            raise ValueError(
                f"hydrometeor disposition operator replay failed for {name}")
        influencing_replay_fingerprint_job = hasher.submit(
            influencing_replay)
        del influencing_replay
        excluded_replay = _receipt_host_float32(operator_replay(
            (labels != 0)
            & (labels != _HRRR_DISPOSITION_CLASSES[
                "TARGET_INFLUENCING"])))
        if (list(excluded_replay.shape) != live_fingerprint["shape"]
                or not _receipt_nonnegative_finite(excluded_replay)):
            raise ValueError(
                f"hydrometeor disposition operator replay failed for {name}")
        source_fingerprint = source_fingerprint_job.result()
        source_replay_fingerprint = source_replay_fingerprint_job.result()
        influencing_replay_fingerprint = influencing_replay_fingerprint_job.result()
        operator_replay_receipt = {
            "schema": "gpuwm-wrf-q-binary-mask-production-replay-v1",
            "source_mask_sha256": _packed_mask_sha256(raw != 0.0),
            "target_influencing_mask_sha256": _packed_mask_sha256(
                raw_labels == _HRRR_DISPOSITION_CLASSES[
                    "TARGET_INFLUENCING"]),
            "excluded_mask_sha256": _packed_mask_sha256(
                (raw_labels != 0)
                & (raw_labels != _HRRR_DISPOSITION_CLASSES[
                    "TARGET_INFLUENCING"])),
            "source_output": source_replay_fingerprint,
            "target_influencing_output": influencing_replay_fingerprint,
            "excluded_output": array_correspondence_fingerprint(
                excluded_replay),
            "source_target_outputs_byte_equal": (
                source_replay_fingerprint == influencing_replay_fingerprint),
            "excluded_output_all_exact_zero": not bool(
                np.any(excluded_replay != 0.0)),
        }
        del excluded_replay
        influencing_count = counts["TARGET_INFLUENCING"]
        if influencing_count > 0 and int(live_fingerprint["nonzero_count"]) == 0:
            raise ValueError(
                "WRF target-influencing analyzed hydrometeor source mass "
                f"was lost for {name}")
        if influencing_count == 0 and int(live_fingerprint["nonzero_count"]) != 0:
            raise ValueError(
                "WRF target-independent hydrometeor source produced "
                f"unexpected initialized mass for {name}")
        species_receipts[name] = {
            "shape": list(first_shape),
            "encoding": _HRRR_DISPOSITION_ENCODING,
            "labels_base64": base64.b64encode(encoded).decode("ascii"),
            "labels_sha256": hashlib.sha256(
                raw_labels.tobytes(order="C")).hexdigest(),
            "decoded_source_fingerprint_sha256": (
                _canonical_receipt_sha256(source_fingerprint)),
            "initialized_state_fingerprint_sha256": (
                _canonical_receipt_sha256(live_fingerprint)),
            "partition_complete": True,
            "source_nonzero_count": int(source_fingerprint["nonzero_count"]),
            "target_influencing_source_count": influencing_count,
            "wrf_excluded_source_count": (
                int(source_fingerprint["nonzero_count"])
                - influencing_count),
            "class_counts": counts,
            "class_mask_sha256": mask_hashes,
            "examples": examples,
            "operator_replay": operator_replay_receipt,
        }

    receipt = {
        "schema": HRRR_HYDROMETEOR_VERTICAL_DISPOSITION_SCHEMA,
        "operator": dict(_HRRR_DISPOSITION_OPERATOR),
        "geometry": geometry_receipt,
        "species": species_receipts,
    }
    receipt["evidence_sha256"] = _canonical_receipt_sha256(receipt)
    return receipt


def _decode_hrrr_disposition_labels(
        encoded: object, *, expected_bytes: int) -> np.ndarray:
    if not isinstance(encoded, str) or expected_bytes < 1:
        raise ValueError("hydrometeor disposition label encoding is invalid")
    try:
        compressed = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(
            "hydrometeor disposition label encoding is invalid") from exc
    decompressor = zlib.decompressobj()
    try:
        decoded = decompressor.decompress(compressed, expected_bytes + 1)
        decoded += decompressor.flush(max(1, expected_bytes + 1 - len(decoded)))
    except zlib.error as exc:
        raise ValueError(
            "hydrometeor disposition labels are not valid zlib data") from exc
    if (len(decoded) != expected_bytes or not decompressor.eof
            or decompressor.unused_data or decompressor.unconsumed_tail):
        raise ValueError(
            "hydrometeor disposition label payload has the wrong length")
    return np.frombuffer(decoded, dtype=np.uint8).copy()


def _decode_hrrr_production_support(
        encoded: object, *, expected_bits: int) -> np.ndarray:
    """Decode one canonical packed production-support certificate."""

    expected_bytes = (expected_bits + 7) // 8
    try:
        packed = _decode_hrrr_disposition_labels(
            encoded, expected_bytes=expected_bytes)
    except ValueError as exc:
        raise ValueError(
            "hydrometeor production support encoding is invalid") from exc
    support = np.unpackbits(
        packed, bitorder="little", count=expected_bits).astype(
            np.bool_, copy=False)
    if not np.array_equal(
            np.packbits(support, bitorder="little"), packed):
        raise ValueError(
            "hydrometeor production support padding is noncanonical")
    return support


def _validate_pressure_partition(partition: object) -> None:
    """Check the cross-check block is well formed; its counts are facts.

    A disagreement count of any size is a statement about the geometry,
    not about the initial state, so only the shape of the block is
    checked: the fields are present, the counts are non-negative integers
    that add up, and the listed disagreements are bounded by the count.
    """

    def count(value):
        return (not isinstance(value, bool) and isinstance(value, int)
                and value >= 0)

    if (not isinstance(partition, Mapping)
            or set(partition) != _HRRR_PRESSURE_PARTITION_FIELDS
            or partition.get("schema") != _HRRR_PRESSURE_PARTITION_SCHEMA
            or partition.get("support_authority")
            != "production-operator-replay"
            or isinstance(partition.get("logarithm_tie_band"), bool)
            or not isinstance(partition.get("logarithm_tie_band"),
                              (int, float))
            or not np.isfinite(float(partition["logarithm_tie_band"]))
            or not all(count(partition.get(name)) for name in (
                "active_sample_count", "logarithm_tie_sample_count",
                "logarithm_tie_supported_count",
                "disagreeing_sample_count", "geometry_only_count",
                "production_only_count"))
            or not isinstance(partition.get("disagreement_mask_sha256"), str)
            or not isinstance(partition.get("disagreements"), list)):
        raise ValueError(
            "hydrometeor disposition pressure partition cross-check is "
            "malformed")
    disagreeing = partition["disagreeing_sample_count"]
    if (partition["geometry_only_count"]
            + partition["production_only_count"] != disagreeing
            or partition["logarithm_tie_supported_count"]
            > partition["logarithm_tie_sample_count"]
            or disagreeing + partition["logarithm_tie_sample_count"]
            > partition["active_sample_count"]
            or len(partition["disagreements"])
            != min(disagreeing, _HRRR_DISPOSITION_EXAMPLE_LIMIT)
            or not all(isinstance(record, Mapping)
                       for record in partition["disagreements"])):
        raise ValueError(
            "hydrometeor disposition pressure partition cross-check counts "
            "do not add up")


def validate_hrrr_hydrometeor_vertical_disposition(
        source_fingerprints: Mapping[str, object],
        initialized_fingerprints: Mapping[str, object],
        disposition: object) -> dict[str, object]:
    """Fail closed over a producer's exhaustive WRF-support partition.

    Closed over the receipt's integrity: its identity, its partition of the
    decoded source mask, and its agreement with the operator's own replays.
    Where the receipt records an observation about the geometry or about
    how the replays composed, the observation is checked against the
    fingerprints it summarises, never required to come out one way.
    """

    if not isinstance(disposition, Mapping):
        raise ValueError("hydrometeor vertical disposition evidence is missing")
    if set(disposition) != {
            "schema", "operator", "geometry", "species",
            "evidence_sha256"}:
        raise ValueError("hydrometeor vertical disposition fields are invalid")
    if (disposition.get("schema")
            != HRRR_HYDROMETEOR_VERTICAL_DISPOSITION_SCHEMA
            or disposition.get("operator") != _HRRR_DISPOSITION_OPERATOR
            or not isinstance(disposition.get("geometry"), Mapping)
            or not isinstance(disposition.get("species"), Mapping)
            or not isinstance(disposition.get("evidence_sha256"), str)):
        raise ValueError("hydrometeor vertical disposition schema is invalid")
    unsigned = dict(disposition)
    observed_hash = unsigned.pop("evidence_sha256")
    if observed_hash != _canonical_receipt_sha256(unsigned):
        raise ValueError("hydrometeor vertical disposition evidence was changed")
    geometry = disposition["geometry"]
    # ``pressure_partition`` is the geometry's cross-check against the
    # operator.  Receipts written before it existed carry none and are
    # the same evidence otherwise, so its absence is accepted.
    if set(geometry) - {"pressure_partition"} != {
            "source_pressure", "surface_pressure", "target_pressure",
            "source_level_order", "source_level_order_values",
            "production_target_support"}:
        raise ValueError("hydrometeor disposition geometry is incomplete")
    if "pressure_partition" in geometry:
        _validate_pressure_partition(geometry["pressure_partition"])
    for name in (
            "source_pressure", "surface_pressure", "target_pressure",
            "source_level_order"):
        if (not isinstance(geometry[name], Mapping)
                or not {"shape", "dtype", "sha256",
                        "nonzero_mask_sha256", "nonzero_count",
                        "minimum", "maximum"} <= set(geometry[name])):
            raise ValueError(
                "hydrometeor disposition geometry fingerprint is invalid")
    if (set(source_fingerprints) != set(initialized_fingerprints)
            or set(disposition["species"]) != set(source_fingerprints)
            or not source_fingerprints):
        raise ValueError(
            "hydrometeor disposition species inventory is incomplete")
    first_source = next(iter(source_fingerprints.values()))
    first_initialized = next(iter(initialized_fingerprints.values()))
    source_shape = first_source.get("shape") \
        if isinstance(first_source, Mapping) else None
    state_shape = first_initialized.get("shape") \
        if isinstance(first_initialized, Mapping) else None
    order_values = geometry["source_level_order_values"]
    if (not isinstance(source_shape, list) or len(source_shape) != 3
            or not isinstance(state_shape, list) or len(state_shape) != 3
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 1 for value in source_shape + state_shape)
            or source_shape[0] * source_shape[1] * source_shape[2]
            > 250_000_000
            or state_shape[0] * state_shape[1] * state_shape[2]
            > 250_000_000
            or not isinstance(order_values, list)
            or len(order_values) != source_shape[0]
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in order_values)
            or sorted(order_values) != list(range(source_shape[0]))
            or geometry["source_pressure"].get("shape") != source_shape
            or geometry["source_pressure"].get("dtype") != "float32"
            or geometry["surface_pressure"].get("shape") != source_shape[1:]
            or geometry["surface_pressure"].get("dtype") != "float32"
            or geometry["target_pressure"].get("shape") != state_shape
            or geometry["target_pressure"].get("dtype") != "float32"
            or any(float(geometry[key].get("minimum", np.nan)) <= 0.0
                   for key in ("source_pressure", "surface_pressure",
                               "target_pressure"))
            or geometry["source_level_order"]
            != array_correspondence_fingerprint(
                np.asarray(order_values, dtype=np.int32))):
        raise ValueError("hydrometeor disposition geometry is inconsistent")

    required_fingerprint = {
        "shape", "dtype", "sha256", "nonzero_mask_sha256",
        "nonzero_count", "minimum", "maximum"}
    support_receipt = geometry["production_target_support"]
    if (not isinstance(support_receipt, Mapping)
            or set(support_receipt) != {
                "schema", "shape", "encoding", "mask_base64",
                "mask_sha256", "support_count", "level_replays"}
            or support_receipt.get("schema")
            != _HRRR_PRODUCTION_SUPPORT_SCHEMA
            or support_receipt.get("shape") != source_shape
            or support_receipt.get("encoding")
            != _HRRR_PRODUCTION_SUPPORT_ENCODING
            or not isinstance(support_receipt.get("mask_sha256"), str)
            or isinstance(support_receipt.get("support_count"), bool)
            or not isinstance(support_receipt.get("support_count"), int)
            or not 0 <= support_receipt["support_count"] <= int(
                np.prod(source_shape, dtype=np.int64))
            or not isinstance(support_receipt.get("level_replays"), list)
            or len(support_receipt["level_replays"]) != source_shape[0]):
        raise ValueError(
            "hydrometeor production target support evidence is invalid")
    production_support = _decode_hrrr_production_support(
        support_receipt.get("mask_base64"),
        expected_bits=int(np.prod(source_shape, dtype=np.int64))).reshape(
            source_shape)
    if (support_receipt["mask_sha256"]
            != _packed_mask_sha256(production_support)
            or support_receipt["support_count"]
            != int(np.count_nonzero(production_support))):
        raise ValueError(
            "hydrometeor production target support mask is invalid")
    ncolumn = source_shape[1] * source_shape[2]
    ntarget = state_shape[0]
    for ordered_level, replay in enumerate(
            support_receipt["level_replays"]):
        raw_level = order_values[ordered_level]
        level_support = production_support[raw_level]
        support_count = int(np.count_nonzero(level_support))
        if (not isinstance(replay, Mapping)
                or set(replay) != {
                    "ordered_source_level", "raw_source_level",
                    "supported_column_count", "support_mask_sha256",
                    "output"}
                or replay.get("ordered_source_level") != ordered_level
                or replay.get("raw_source_level") != raw_level
                or isinstance(replay.get("supported_column_count"), bool)
                or replay.get("supported_column_count") != support_count
                or support_count > ncolumn
                or replay.get("support_mask_sha256")
                != _packed_mask_sha256(level_support)
                or not isinstance(replay.get("output"), Mapping)
                or set(replay["output"]) != required_fingerprint
                or replay["output"].get("shape") != state_shape
                or replay["output"].get("dtype") != "float32"
                or isinstance(
                    replay["output"].get("nonzero_count"), bool)
                or not isinstance(
                    replay["output"].get("nonzero_count"), int)
                or not support_count
                <= replay["output"]["nonzero_count"]
                <= support_count * ntarget
                or not 0.0 <= float(
                    replay["output"].get("minimum", np.nan))
                <= float(replay["output"].get("maximum", np.nan))
                <= 1.0):
            raise ValueError(
                "hydrometeor source-level production replay is invalid")

    validated = {}
    for name in sorted(source_fingerprints):
        source = source_fingerprints[name]
        initialized = initialized_fingerprints[name]
        item = disposition["species"][name]
        if (not isinstance(source, Mapping)
                or not isinstance(initialized, Mapping)
                or not isinstance(item, Mapping)):
            raise ValueError(
                f"hydrometeor disposition {name} fingerprints are invalid")
        if (not required_fingerprint <= set(source)
                or not required_fingerprint <= set(initialized)):
            raise ValueError(
                f"hydrometeor disposition {name} fingerprint is incomplete")
        if set(item) != {
                "shape", "encoding", "labels_base64", "labels_sha256",
                "decoded_source_fingerprint_sha256",
                "initialized_state_fingerprint_sha256",
                "partition_complete", "source_nonzero_count",
                "target_influencing_source_count",
                "wrf_excluded_source_count", "class_counts",
                "class_mask_sha256", "examples", "operator_replay"}:
            raise ValueError(
                f"hydrometeor disposition {name} fields are incomplete")
        count_fields = (
            source.get("nonzero_count"), initialized.get("nonzero_count"),
            item.get("source_nonzero_count"),
            item.get("target_influencing_source_count"),
            item.get("wrf_excluded_source_count"),
        )
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value < 0 for value in count_fields):
            raise ValueError(
                f"hydrometeor disposition {name} count is invalid")
        shape = item.get("shape")
        if (not isinstance(shape, list) or len(shape) != 3
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value < 1 for value in shape)
                or shape != source.get("shape")
                or shape != source_shape
                or initialized.get("shape") != state_shape
                or np.prod(shape, dtype=np.int64) > 250_000_000):
            raise ValueError(
                f"hydrometeor disposition {name} shape is invalid")
        expected_bytes = int(np.prod(shape, dtype=np.int64))
        if item.get("encoding") != _HRRR_DISPOSITION_ENCODING:
            raise ValueError(
                f"hydrometeor disposition {name} encoding is invalid")
        labels = _decode_hrrr_disposition_labels(
            item.get("labels_base64"), expected_bytes=expected_bytes)
        labels = labels.reshape(shape)
        if (item.get("labels_sha256") != hashlib.sha256(
                labels.tobytes(order="C")).hexdigest()
                or item.get("decoded_source_fingerprint_sha256")
                != _canonical_receipt_sha256(source)
                or item.get("initialized_state_fingerprint_sha256")
                != _canonical_receipt_sha256(initialized)
                or item.get("partition_complete") is not True):
            raise ValueError(
                f"hydrometeor disposition {name} identity is invalid")
        allowed_codes = np.asarray(
            [0, *_HRRR_DISPOSITION_CLASS_BY_CODE], dtype=np.uint8)
        if np.any(~np.isin(labels, allowed_codes)):
            raise ValueError(
                f"hydrometeor disposition {name} has an unknown class")
        source_count = int(source.get("nonzero_count", -1))
        if (source_count < 0
                or item.get("source_nonzero_count") != source_count
                or int(np.count_nonzero(labels)) != source_count
                or _packed_mask_sha256(labels != 0)
                != source.get("nonzero_mask_sha256")):
            raise ValueError(
                f"hydrometeor disposition {name} does not partition the "
                "decoded source mask")
        counts = item.get("class_counts")
        mask_hashes = item.get("class_mask_sha256")
        examples = item.get("examples")
        if (not isinstance(counts, Mapping)
                or set(counts) != set(_HRRR_DISPOSITION_CLASSES)
                or not isinstance(mask_hashes, Mapping)
                or set(mask_hashes) != set(_HRRR_DISPOSITION_CLASSES)
                or not isinstance(examples, Mapping)
                or set(examples) != set(_HRRR_DISPOSITION_CLASSES)):
            raise ValueError(
                f"hydrometeor disposition {name} class evidence is invalid")
        observed_counts = {}
        for class_name, code in _HRRR_DISPOSITION_CLASSES.items():
            class_mask = labels == code
            count = int(np.count_nonzero(class_mask))
            observed_counts[class_name] = count
            example = examples[class_name]
            if (isinstance(counts[class_name], bool)
                    or not isinstance(counts[class_name], int)
                    or counts[class_name] != count
                    or mask_hashes[class_name]
                    != _packed_mask_sha256(class_mask)
                    or not isinstance(example, Mapping)
                    or set(example) != {"complete", "total_count", "records"}
                    or isinstance(example.get("total_count"), bool)
                    or not isinstance(example.get("total_count"), int)
                    or example.get("total_count") != count
                    or example.get("complete")
                    is not (count <= _HRRR_DISPOSITION_EXAMPLE_LIMIT)
                    or not isinstance(example.get("records"), list)
                    or len(example["records"])
                    != min(count, _HRRR_DISPOSITION_EXAMPLE_LIMIT)):
                raise ValueError(
                    f"hydrometeor disposition {name}/{class_name} "
                    "evidence is inconsistent")
            seen = set()
            for record in example["records"]:
                if (not isinstance(record, Mapping)
                        or record.get("class") != class_name
                        or not isinstance(record.get("raw_flat_index"), int)
                        or record["raw_flat_index"] in seen
                        or not 0 <= record["raw_flat_index"] < expected_bytes
                        or int(labels.ravel()[record["raw_flat_index"]])
                        != code
                        or not isinstance(
                            record.get("influencing_target_indices"), list)
                        or (class_name == "TARGET_INFLUENCING")
                        != bool(record["influencing_target_indices"])):
                    raise ValueError(
                        f"hydrometeor disposition {name}/{class_name} "
                        "example is invalid")
                seen.add(record["raw_flat_index"])
        influencing_count = observed_counts["TARGET_INFLUENCING"]
        excluded_count = source_count - influencing_count
        expected_target = (labels != 0) & production_support
        if not np.array_equal(
                labels == _HRRR_DISPOSITION_CLASSES["TARGET_INFLUENCING"],
                expected_target):
            raise ValueError(
                f"hydrometeor disposition {name} contradicts exact "
                "production target support")
        if (item.get("target_influencing_source_count") != influencing_count
                or item.get("wrf_excluded_source_count") != excluded_count):
            raise ValueError(
                f"hydrometeor disposition {name} partition totals differ")
        replay = item.get("operator_replay")
        # The two composition facts are recorded as observed; each must
        # match the fingerprints it summarises, whichever way it came out.
        outputs_byte_equal = (
            isinstance(replay, Mapping)
            and replay.get("source_output")
            == replay.get("target_influencing_output"))
        if (not isinstance(replay, Mapping)
                or set(replay) != {
                    "schema", "source_mask_sha256",
                    "target_influencing_mask_sha256",
                    "excluded_mask_sha256", "source_output",
                    "target_influencing_output", "excluded_output",
                    "source_target_outputs_byte_equal",
                    "excluded_output_all_exact_zero"}
                or replay.get("schema")
                != "gpuwm-wrf-q-binary-mask-production-replay-v1"
                or replay.get("source_mask_sha256")
                != _packed_mask_sha256(labels != 0)
                or replay.get("target_influencing_mask_sha256")
                != _packed_mask_sha256(
                    labels == _HRRR_DISPOSITION_CLASSES[
                        "TARGET_INFLUENCING"])
                or replay.get("excluded_mask_sha256")
                != _packed_mask_sha256(
                    (labels != 0)
                    & (labels != _HRRR_DISPOSITION_CLASSES[
                        "TARGET_INFLUENCING"]))
                or replay.get("source_target_outputs_byte_equal")
                is not outputs_byte_equal
                or not isinstance(
                    replay.get("excluded_output_all_exact_zero"), bool)):
            raise ValueError(
                f"hydrometeor disposition {name} production replay is invalid")
        excluded_output = replay.get("excluded_output")
        replay_output = replay.get("source_output")
        influencing_output = replay.get("target_influencing_output")
        state_shape = initialized.get("shape")
        if (not isinstance(state_shape, list) or len(state_shape) != 3
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value < 1 for value in state_shape)
                or np.prod(state_shape, dtype=np.int64) > 250_000_000):
            raise ValueError(
                f"hydrometeor disposition {name} state shape is invalid")
        state_size = int(np.prod(state_shape, dtype=np.int64))
        exact_zero_output = {
            "shape": state_shape,
            "dtype": "float32",
            "sha256": _zero_byte_sha256(4 * state_size),
            "nonzero_mask_sha256": _zero_byte_sha256((state_size + 7) // 8),
            "nonzero_count": 0,
            "minimum": 0.0,
            "maximum": 0.0,
        }
        def well_formed(output):
            return (
                isinstance(output, Mapping)
                and required_fingerprint <= set(output)
                and output.get("shape") == initialized.get("shape")
                and output.get("dtype") == "float32"
                and not any(isinstance(output.get(key), bool)
                            for key in ("nonzero_count", "minimum",
                                        "maximum"))
                and isinstance(output.get("nonzero_count"), int)
                and int(output["nonzero_count"]) >= 0
                and bool(np.isfinite(float(output.get("minimum", np.nan))))
                and bool(np.isfinite(float(output.get("maximum", np.nan)))))

        if (not well_formed(replay_output)
                or not well_formed(influencing_output)
                or not well_formed(excluded_output)
                or replay.get("excluded_output_all_exact_zero")
                is not (dict(excluded_output) == exact_zero_output)):
            raise ValueError(
                f"hydrometeor disposition {name} replay record is invalid")
        live_count = int(initialized.get("nonzero_count", -1))
        if live_count < 0:
            raise ValueError(
                f"hydrometeor disposition {name} state count is invalid")
        if influencing_count > 0 and live_count == 0:
            raise ValueError(
                "WRF target-influencing analyzed hydrometeor source mass "
                f"was lost for {name}")
        if influencing_count == 0 and live_count != 0:
            raise ValueError(
                "WRF target-independent hydrometeor source produced "
                f"unexpected initialized mass for {name}")
        if source_count == 0:
            strength = "VACUOUS"
        elif influencing_count == 0:
            strength = "WRF_EXCLUDED"
        elif excluded_count:
            strength = "PARTIALLY_WRF_EXCLUDED"
        else:
            strength = "PROVEN"
        validated[name] = {
            "strength": strength,
            "source_nonzero_count": source_count,
            "target_influencing_source_count": influencing_count,
            "wrf_excluded_source_count": excluded_count,
            "class_counts": observed_counts,
            "labels_sha256": item["labels_sha256"],
        }
    return {
        "schema": HRRR_HYDROMETEOR_VERTICAL_DISPOSITION_SCHEMA,
        "evidence_sha256": disposition["evidence_sha256"],
        "species": validated,
    }


def _saturation_mixing_ratio_serial(temperature, pressure,
                                    relative_humidity):
    """Evaluate one contiguous WRF/Bolton humidity chunk."""
    temperature, pressure, rh = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
        np.clip(np.asarray(relative_humidity, dtype=np.float64), 0.0, 100.0),
    )
    # SVP1 is kPa in module_model_constants; convert to hPa to pair with p/100.
    # WRF v4.6.1 rh_to_mxrat1 uses its own local EPS = 0.622
    # (module_initialize_real.F:7366), NOT module ep_2 = 0.62175.
    # Its unconditional floor is at
    # (module_initialize_real.F:7402, q = MAX(eps*es/(p/100.-es), 1.E-6)).
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        es_hpa = (rh * 0.01) * (10.0 * c.SVP1) * pm.exp(
            c.SVP2 * (temperature - c.SVPT0) / (temperature - c.SVP3))
        candidate = 0.622 * es_hpa / (pressure / 100.0 - es_hpa)
    valid = ((temperature != 0.0) & np.isfinite(es_hpa)
             & (es_hpa < pressure / 100.0))
    return np.where(valid, np.maximum(candidate, 1.0e-6), 1.0e-6)


def _saturation_mixing_ratio(temperature, pressure, relative_humidity=100.0,
                             *, column_workers=1):
    """WRF/Bolton liquid-water saturation mixing ratio (kg kg-1).

    Large setup arrays are divided into deterministic contiguous chunks.
    NumPy performs the unchanged float64 expression in independent threads,
    preserving every element's arithmetic while using otherwise-idle host
    cores during native initial-condition preparation.
    """
    workers = _column_worker_count(column_workers)
    temperature, pressure, rh = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
        np.clip(np.asarray(relative_humidity, dtype=np.float64), 0.0, 100.0),
    )
    if workers == 1 or temperature.ndim == 0 or temperature.shape[0] < 2:
        with pm.worker_limit(workers):
            return _saturation_mixing_ratio_serial(temperature, pressure, rh)

    chunks = _axis0_chunks(temperature.shape[0], workers)
    output = np.empty(temperature.shape, dtype=np.float64)
    with _column_pool(len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _saturation_mixing_ratio_serial,
                (temperature[start:stop], pressure[start:stop],
                 rh[start:stop]))
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


def _cap_stratospheric_qv_serial(qv, pressure):
    """Evaluate one contiguous stratospheric-cap chunk."""
    qv = np.asarray(qv, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    return np.where((pressure < 10000.0) & (qv > 1.0e-5), 3.0e-6, qv)


def _cap_stratospheric_qv(qv, pressure, *, column_workers=1):
    """WRF ``rh_to_mxrat1`` stratospheric qv sanity cap.

    module_initialize_real.F:7490-7498 with the Registry defaults
    (Registry.EM_COMMON:2306-2308): where ``p < qv_max_p_safe`` (10000 Pa,
    strict) and ``qv > qv_max_flag`` (1e-5, strict), force ``qv_max_value``
    (3e-6).  The companion qv_min cap (:7499-7503, p < 110000 Pa, qv < 1e-6
    -> 1e-6) is already realized by the unconditional 1e-6 floor in
    :func:`_saturation_mixing_ratio` for this domain's pressure range.
    """
    workers = _column_worker_count(column_workers)
    qv, pressure = np.broadcast_arrays(
        np.asarray(qv, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
    )
    if workers == 1 or qv.ndim == 0 or qv.shape[0] < 2:
        return _cap_stratospheric_qv_serial(qv, pressure)

    chunks = _axis0_chunks(qv.shape[0], workers)
    output = np.empty(qv.shape, dtype=np.float64)
    with _column_pool(len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _cap_stratospheric_qv_serial,
                (qv[start:stop], pressure[start:stop]))
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


#: The overlapping-parabolic operator's undershoot envelope for a source
#: this process did not map itself: WPS's own SPECHUMD, gated to 0..0.1
#: upstream, mapped by metgrid into a met_em file.  It is
#: :data:`woof.ingest.horiz.WPS_PARABOLIC_NEGATIVE_WEIGHT` evaluated at
#: that gate -- ``-9/32 * 0.1 = -0.028125`` -- plus one micro-unit for
#: FP32 evaluation rounding.  A snapshot that mapped SPFH ITSELF carries
#: the envelope of the source it actually saw
#: (``HorizontalSnapshot.specific_humidity_undershoot_floor``), which is
#: tighter for every source drier than that gate, and this constant is
#: then not used.
_WPS_SPFH_UNDERSHOOT_LOWER_BOUND = -0.028126


def _specific_humidity_undershoot_bound(
        allow_wps_undershoot, undershoot_floor):
    """The lower bound this call admits, and the text that names it."""
    if not allow_wps_undershoot:
        return 0.0, "[0, 1)"
    if undershoot_floor is None:
        bound = _WPS_SPFH_UNDERSHOOT_LOWER_BOUND
    else:
        bound = float(undershoot_floor)
        if not np.isfinite(bound) or bound > 0.0:
            raise ValueError(
                "undershoot_floor must be a finite, non-positive value")
    return bound, f"[{bound:g}, 1)"


def _specific_humidity_to_mixing_ratio_serial(
        specific_humidity, *, allow_wps_undershoot=False,
        undershoot_floor=None, axis0_span=None):
    """Convert and validate one contiguous specific-humidity chunk.

    ``axis0_span`` is the half-open axis-0 range this array occupies in
    the field it came from, present exactly when the caller split that
    field across workers.  The counts and the extremes a refusal quotes
    are this array's, so the text says which array that is: a reader
    told "48 of 2246320" and given a minimum to go and find has to be
    able to find it, and on a split field those numbers were one chunk's
    while the sentence called them the field's.
    """
    specific = np.asarray(specific_humidity, dtype=np.float64)
    lower, interval = _specific_humidity_undershoot_bound(
        allow_wps_undershoot, undershoot_floor)
    finite = np.isfinite(specific)
    outside = ~finite | (specific < lower) | (specific >= 1.0)
    if outside.any():
        scope = ("field" if axis0_span is None else
                 f"levels {int(axis0_span[0])}..{int(axis0_span[1]) - 1}")
        extremes = ("none finite" if not finite.any() else
                    f"{scope} minimum {float(specific[finite].min())!r}, "
                    f"{scope} maximum {float(specific[finite].max())!r}")
        raise ValueError(
            f"specific humidity must be finite in {interval} | observed: "
            f"outside_cells={int(np.count_nonzero(outside))} of "
            f"{specific.size} in the {scope}, of which "
            f"non_finite={int(np.count_nonzero(~finite))}, "
            f"below={int(np.count_nonzero(finite & (specific < lower)))}, "
            f"at_or_above_one="
            f"{int(np.count_nonzero(finite & (specific >= 1.0)))}; "
            f"{extremes}; the bound is the overlapping-parabolic "
            "operator's own undershoot envelope for the source it was "
            "handed, so a value past it is not one that operator made")
    return specific / (1.0 - specific)


def _specific_humidity_to_mixing_ratio(
        specific_humidity, *, allow_wps_undershoot=False,
        undershoot_floor=None, column_workers=1):
    """Convert HRRR/WPS specific humidity to dry-air mixing ratio.

    This is WRF ``module_initialize_real.F``'s ``flag_sh`` branch:
    ``qv_gc = sh_gc / (1 - sh_gc)``.  Unlike the RH path, WRF does not
    apply the ``rh_to_mxrat1`` floor or stratospheric cap when
    ``use_sh_qv`` is active.  WPS's overlapping-parabolic horizontal
    interpolation can create negative undershoots from an everywhere
    non-negative source field.  ``real.exe`` retains those values in
    ``qv_gc`` for ``integ_moist`` even when ``use_sh_qv = .false.``.  HRRR
    SPFH is gated to 0..0.1 upstream; the 2-D sixteen-point operator's most
    negative coefficient sum is -9/32, so -0.028125 is its exact lower
    envelope for that gate.  ``undershoot_floor`` supplies that envelope
    for the source THIS process mapped instead of assuming WPS's gate:
    the mapper records what that operator could have made from the array
    it was about to map
    (:func:`woof.ingest.horiz.parabolic_undershoot_floor`), which is
    tighter than the constant for every source drier than 0.1 and correct
    for one that is wetter.

    This IS the physical-range check on specific humidity, and every
    initialization passes through it.  It is made here, on the mapped
    field, rather than on the source: a mapped value below zero is made
    by the horizontal operator, and the only thing that tells that
    undershoot from bad forcing is the envelope of the operator that made
    it.  The bound applies on BOTH lanes, because real.exe's FLAG_SH
    branch converts what metgrid handed it with no range test at all;
    refusing it on the direct-qv lane alone refused a window for the
    shape of the ground it covered.
    """
    if not isinstance(allow_wps_undershoot, (bool, np.bool_)):
        raise TypeError("allow_wps_undershoot must be boolean")
    workers = _column_worker_count(column_workers)
    specific = np.asarray(specific_humidity, dtype=np.float64)
    if (workers == 1 or specific.ndim == 0 or specific.shape[0] < 2):
        return _specific_humidity_to_mixing_ratio_serial(
            specific, allow_wps_undershoot=allow_wps_undershoot,
            undershoot_floor=undershoot_floor)

    chunks = _axis0_chunks(specific.shape[0], workers)
    output = np.empty(specific.shape, dtype=np.float64)
    with _column_pool(len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _specific_humidity_to_mixing_ratio_serial,
                (specific[start:stop],),
                {"allow_wps_undershoot": allow_wps_undershoot,
                 "undershoot_floor": undershoot_floor,
                 "axis0_span": (start, stop)})
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


#: WRF v4.6.1's ``qv_min`` sanity floor and the pressure below which it applies
#: (Registry.EM_COMMON:2306-2308 defaults ``qv_min_p_safe = 110000`` Pa,
#: ``qv_min_flag = qv_min_value = 1e-6``; applied at
#: ``module_initialize_real.F:7499-7503``).  This is the SAME 1e-6 that
#: :func:`_saturation_mixing_ratio` already applies unconditionally on the
#: RH lane (:7402, ``q = MAX(eps*es/(p/100.-es), 1.E-6)``), named here so
#: the FLAG_SH surface lane can share one floor with it instead of
#: carrying none.
_WRF_QV_MIN_P_SAFE = 110_000.0
_WRF_QV_MIN_VALUE = 1.0e-6

_SURFACE_QV_FLOOR_WRF_REFERENCE = {
    "wrf_version": "v4.6.1",
    "wrf_citation": (
        "dyn_em/module_initialize_real.F:1138-1167 (FLAG_SH branch: "
        "qv_gc = sh_gc/(1-sh_gc) at :1157, no floor), :1253-1259 "
        "(grid%q2 = qv_gc(i,1,j) verbatim); the qv_min sanity floor "
        "real.exe applies to every other moisture value lives in "
        "rh_to_mxrat1 (:7402 unconditional 1e-6; :7499-7503 "
        "qv_min_p_safe/qv_min_flag/qv_min_value) and is called on the "
        "prognostic field only (:1837-1860), never on the surface "
        "pseudo-level of this branch"),
    "wrf_behavior": (
        "real.exe publishes an unfloored Q2: WPS routes SPECHUMD through "
        "the overshooting sixteen_pt operator (METGRID.TBL "
        "interp_option=sixteen_pt+four_pt+average_4pt), so a 2 m specific "
        "humidity that is exactly zero over high dry terrain interpolates "
        "NEGATIVE, and both qv_gc(:,1,:) and grid%q2 keep the sign"),
    "gpuwm_behavior": (
        "the PUBLISHED surface mixing ratio (RealInitResult.surface_qv, "
        "which becomes Q2) is floored at WRF's own qv_min_value where the "
        "WPS undershoot took it below that floor, which is exactly what "
        "the RH lane's _saturation_mixing_ratio already does to the same "
        "quantity; sfcprs2, integ_moist and the vertical-interpolation "
        "pseudo-level keep the raw WRF value, so no prognostic field and "
        "no existing fingerprint moves"),
    "gpuwm_divergence_reason": (
        "a mixing ratio cannot be negative and every consumer of Q2 "
        "(relative humidity, dewpoint, vapour-pressure deficit) needs the "
        "log of a positive number -- the same reason the runtime's own "
        "SFCDIAGS transcription refuses to publish WRF's negative Q2 "
        "(woof/core/physics.py _refresh_surface_diagnostics), applied at "
        "initialization so the value the forecast STARTS from is physical "
        "too"),
}


def _floor_flag_sh_surface_mixing_ratio(surface_qv, surface_pressure):
    """Apply WRF's ``qv_min`` floor to the PUBLISHED 2 m mixing ratio.

    WPS's ``sixteen_pt`` overlapping parabolic is an overshooting
    operator and METGRID.TBL routes ``SPECHUMD`` through it, so a source
    2 m specific humidity that decodes to EXACTLY zero -- which HRRR's
    GRIB2 packing does over high, dry terrain -- interpolates to a small
    negative.  ``real.exe`` keeps that sign all the way into ``grid%q2``
    (see :data:`_SURFACE_QV_FLOOR_WRF_REFERENCE`), and reproducing it
    bit-for-bit would publish a negative 2 m mixing ratio.

    The floor is WRF's own ``qv_min_value``, not an invented number, and
    it is the identical constant the RH lane already applies to the
    identical quantity inside :func:`_saturation_mixing_ratio` -- so this
    makes the two surface lanes agree rather than inventing a third
    behaviour.

    It is applied at the PUBLICATION point and nowhere else.  ``sfcprs2``,
    ``integ_moist`` and the vertical-interpolation surface pseudo-level
    are uses real.exe is DEFINED for, and they keep the raw value, so the
    prognostic state stays bit-for-bit what it was on every lane --
    including the explicit ``use_sh_qv=True`` lane, whose prognostic qv IS
    the interpolated surface value and which WRF likewise never floors.
    Only cells the undershoot pushed below the floor move, and only in the
    published array, which is why no artifact whose surface moisture stays
    above 1e-6 changes at all.

    Returns the floored array and a receipt; the receipt is empty when
    nothing was floored, and that is the overwhelmingly common case.
    """
    surface_qv = np.asarray(surface_qv, dtype=np.float64)
    pressure = np.asarray(surface_pressure, dtype=np.float64)
    if pressure.shape != surface_qv.shape:
        raise ValueError(
            "surface pressure and surface mixing ratio shapes differ")
    below = ((pressure < _WRF_QV_MIN_P_SAFE)
             & (surface_qv < _WRF_QV_MIN_VALUE))
    if not below.any():
        return surface_qv, {}
    floored = np.where(below, _WRF_QV_MIN_VALUE, surface_qv)
    count = int(np.count_nonzero(below))
    negative = int(np.count_nonzero(below & (surface_qv < 0.0)))
    minimum = float(np.min(surface_qv[below]))
    receipt = {
        "policy": "flag-sh-surface-qv-floored-to-wrf-qv-min-value",
        "wrf_reference": dict(_SURFACE_QV_FLOOR_WRF_REFERENCE),
        "qv_min_value": _WRF_QV_MIN_VALUE,
        "qv_min_p_safe": _WRF_QV_MIN_P_SAFE,
        "floored_cells": count,
        "negative_cells": negative,
        "min_pre_floor": minimum,
    }
    print(
        f"surface moisture floor: {count} FLAG_SH 2 m mixing-ratio "
        f"value(s) below WRF's qv_min_value {_WRF_QV_MIN_VALUE:g} floored "
        f"to it ({negative} of them negative; min pre-floor "
        f"{minimum:.6g}); WPS's overshooting sixteen_pt operator "
        "undershoots SPECHUMD that is exactly zero over high dry terrain, "
        "and real.exe carries that sign into grid%q2 unfloored "
        "(module_initialize_real.F:1157,1257)",
        file=sys.stderr)
    return floored, receipt


_PROGNOSTIC_QV_FLOOR_WRF_REFERENCE = {
    "wrf_version": "v4.7.1",
    "wrf_citation": (
        "dyn_em/module_initialize_real.F:1744-1758 (use_sh_qv: vert_interp "
        "of qv_gc as var_type 'Q' at the namelist lagrange_order, "
        "Registry.EM_COMMON:2290 default 2, lagrange_setup's averaged pair "
        "of second-order Lagrange polynomials), :1830-1851 (rh_to_mxrat1 "
        "and rh_to_mxrat2, and with them the qv_min floor, are skipped when "
        "use_sh_qv is true), :7318-7320 and :7505-7507 (the floor those "
        "routines apply: p < qv_min_p_safe and q < qv_min_flag -> "
        "qv_min_value), :1862-2075 (every hydrometeor and number field is "
        "vertically interpolated with linear_interp, Registry:2291, never "
        "with lagrange_order)"),
    "wrf_behavior": (
        "the second-order polynomial overshoots at a sharp minimum: a "
        "vapour column that is non-negative on every source level yields a "
        "NEGATIVE mixing ratio on a target level that falls inside a dry "
        "slot bracketed by moister levels, and real.exe writes it to "
        "wrfinput as it is, because the use_sh_qv lane is the one lane that "
        "never passes through rh_to_mxrat"),
    "gpuwm_behavior": (
        "a target level the operator took below zero is floored to WRF's "
        "own qv_min_value where p < qv_min_p_safe -- the identical floor "
        "rh_to_mxrat1 applies to the identical quantity on the RH lane and "
        "_floor_flag_sh_surface_mixing_ratio applies to this lane's "
        "surface value -- and the count, the levels and the minimum are "
        "receipted on RealInitResult.prognostic_moisture_floor.  A negative "
        "at or above qv_min_p_safe, where WRF's conditional rule does not "
        "reach, is floored the same way and counted apart in the receipt, "
        "as rh_to_mxrat1's unconditional 1e-6 floor (:7421) does on the RH "
        "lane; only a non-finite value is still refused, naming the level, "
        "the column and the value"),
    "gpuwm_divergence_reason": (
        "a negative vapour mixing ratio is not a state any scheme "
        "integrates, and what reaches this operator was admitted against "
        "the horizontal operator's own undershoot envelope for the source "
        "it mapped (horiz.parabolic_undershoot_floor, applied to the "
        "MAPPED field at _specific_humidity_to_mixing_ratio), so a "
        "negative here is an interpolating polynomial's own undershoot -- "
        "this vertical one, or the horizontal one carried into it within "
        "that envelope -- and not evidence about the forcing; refusing the "
        "whole domain for it halted real-data initializations at the first "
        "sharp mid-tropospheric dry slot the forcing carried"),
}


def _refuse_non_finite_prognostic_qv(qv, *, row_offset=0):
    """The garbage guard, kept: a non-finite interpolated vapour value is
    refused naming where it is, because no floor is a remedy for it.
    ``row_offset`` is the first domain row of a row block."""
    finite = np.isfinite(qv)
    if finite.all():
        return
    bad = np.argwhere(~finite)
    k, j, i = (int(v) for v in bad[0])
    raise ValueError(
        "interpolated specific-humidity qv is invalid | observed: "
        f"non_finite_cells={int(bad.shape[0])}, first at level {k} row "
        f"{j + int(row_offset)} "
        f"column {i} value {float(qv[k, j, i])!r}, levels "
        f"{sorted({int(v) for v in bad[:, 0]})}; the vertical operator was "
        "handed a finite column, admitted against the horizontal "
        "operator's own undershoot envelope, so this is a defect in the "
        "pressures the column was interpolated on, not a value to floor")


def _floor_sh_vertical_undershoot(qv, pressure):
    """Apply WRF's ``qv_min`` floor to the use_sh_qv lane's interpolated vapour.

    WRF real's default vertical operator is the second-order Lagrange
    polynomial (``lagrange_order=2``), and a second-order polynomial
    through three non-negative samples undershoots below zero when the
    middle one is a sharp minimum: a dry slot a few levels deep between
    moister layers -- the ordinary shape of a mid-tropospheric dry
    intrusion in a model's own column -- puts a NEGATIVE mixing ratio on
    the target level that lands inside it.  On the RH lane that value
    never surfaces, because ``rh_to_mxrat1`` floors everything it returns
    at ``qv_min_value``; on the use_sh_qv lane real.exe skips
    ``rh_to_mxrat`` and writes the negative to wrfinput
    (:data:`_PROGNOSTIC_QV_FLOOR_WRF_REFERENCE`).  This function makes the
    two lanes agree: cells the operator took below zero are floored at the
    identical constant, under WRF's own ``qv_min_p_safe`` condition, and
    receipted.  Cells the operator left at or above zero are untouched, so
    no artifact that passed before this floor existed moves.

    What reaches this function has passed the horizontal operator's own
    bounded undershoot envelope for the source it mapped
    (:func:`woof.ingest.horiz.parabolic_undershoot_floor`, applied to
    the MAPPED field at :func:`_specific_humidity_to_mixing_ratio`), so a
    negative here is an interpolating polynomial's own -- this vertical
    operator's undershoot, the horizontal one's carried into it, or the
    surface undershoot carried up one layer -- and not evidence about the
    forcing.  A negative at or above ``qv_min_p_safe``, where WRF's
    conditional rule does not reach, is floored the same way and counted
    apart (``floored_cells_at_or_above_qv_min_p_safe``): the RH lane's
    ``rh_to_mxrat1`` floors at 1e-6 there too, unconditionally, and the
    refusal this replaced named no breakage a floored value would cause,
    only that the rule stopped at that pressure.
    """
    qv = np.asarray(qv, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    if pressure.shape != qv.shape:
        raise ValueError("pressure and interpolated qv shapes differ")
    negative = qv < 0.0
    if not negative.any():
        return qv, {}
    floored = np.where(negative, _WRF_QV_MIN_VALUE, qv)
    return floored, _sh_vertical_undershoot_receipt(
        [_sh_vertical_undershoot_stats(qv, pressure)])


def _sh_vertical_undershoot_stats(qv, pressure, *, row_offset=0):
    """What :func:`_floor_sh_vertical_undershoot` floors in one row block.

    ``None`` when nothing in it is negative.  ``row_offset`` is the block's
    first domain row, so blocks of one domain merge into the receipt the
    whole domain gives (:func:`_sh_vertical_undershoot_receipt`).
    """
    negative = qv < 0.0
    if not negative.any():
        return None
    cells = np.argwhere(negative)
    at = np.unravel_index(
        int(np.argmin(np.where(negative, qv, np.inf))), qv.shape)
    return {
        "floored": int(cells.shape[0]),
        "unreached": int(np.count_nonzero(
            negative & ~(pressure < _WRF_QV_MIN_P_SAFE))),
        "levels": {int(v) for v in cells[:, 0]},
        "columns": int(np.unique(cells[:, 1:], axis=0).shape[0]),
        "minimum": float(np.min(qv[negative])),
        "at": (int(at[0]), int(at[1]) + int(row_offset), int(at[2])),
        "pressure_pa": float(pressure[at]),
    }


def _sh_vertical_undershoot_receipt(parts):
    """The floor receipt (and its one stderr line) for disjoint row blocks.

    Counts add, levels unite, and the minimum is the first in the domain's
    C order among equal values, which is what ``argmin`` over the whole
    domain picks.  Empty when no block floored anything.
    """
    parts = [part for part in parts if part is not None]
    if not parts:
        return {}
    first = min(parts, key=lambda part: (part["minimum"], part["at"]))
    count = sum(part["floored"] for part in parts)
    unreached = sum(part["unreached"] for part in parts)
    columns = sum(part["columns"] for part in parts)
    levels = sorted(set().union(*(part["levels"] for part in parts)))
    minimum = first["minimum"]
    at = first["at"]
    pressure_at = first["pressure_pa"]
    receipt = {
        "policy": "use-sh-qv-vertical-undershoot-floored-to-wrf-qv-min-value",
        "wrf_reference": dict(_PROGNOSTIC_QV_FLOOR_WRF_REFERENCE),
        "qv_min_value": _WRF_QV_MIN_VALUE,
        "qv_min_p_safe": _WRF_QV_MIN_P_SAFE,
        "floored_cells": count,
        "floored_cells_at_or_above_qv_min_p_safe": unreached,
        "columns": columns,
        "levels": levels,
        "min_pre_floor": minimum,
        "min_pre_floor_at": {
            "level": at[0], "row": at[1], "column": at[2],
            "pressure_pa": pressure_at},
    }
    print(
        f"prognostic moisture floor: {count} interpolated vapour "
        f"value(s) in {columns} column(s) on level(s) {levels} were taken "
        "below zero by WRF's second-order vertical operator at a sharp dry "
        f"slot (min {minimum:.6g} at level {at[0]} row {at[1]} column "
        f"{at[2]}, {pressure_at:.0f} Pa) and are floored to WRF's "
        f"qv_min_value {_WRF_QV_MIN_VALUE:g}, the floor rh_to_mxrat1 applies "
        "on the RH lane and real.exe never applies on the use_sh_qv lane "
        "(module_initialize_real.F:1744-1758, :1831)"
        + (f"; {unreached} of them at or above qv_min_p_safe "
           f"{_WRF_QV_MIN_P_SAFE:g} Pa, where only the RH lane's "
           "unconditional floor reaches" if unreached else ""),
        file=sys.stderr)
    return receipt


def _mixing_ratio_to_relative_humidity_serial(
        temperature, pressure, mixing_ratio, *, allow_wps_undershoot=False):
    """Diagnose and validate one contiguous relative-humidity chunk."""
    temperature, pressure, qv = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
        np.asarray(mixing_ratio, dtype=np.float64),
    )
    minimum_qv = (_WPS_SPFH_UNDERSHOOT_LOWER_BOUND
                  / (1.0 - _WPS_SPFH_UNDERSHOOT_LOWER_BOUND)
                  if allow_wps_undershoot else 0.0)
    if (not np.isfinite(temperature).all()
            or not np.isfinite(pressure).all()
            or not np.isfinite(qv).all()
            or np.any(pressure <= 0.0) or np.any(qv < minimum_qv)):
        qv_requirement = ("inside the bounded WPS undershoot envelope"
                          if allow_wps_undershoot else "non-negative")
        raise ValueError(
            "temperature, pressure, and mixing ratio must be finite; "
            f"pressure must be positive and mixing ratio {qv_requirement}")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        es_hpa = (10.0 * c.SVP1) * pm.exp(
            c.SVP2 * (temperature - c.SVPT0) / (temperature - c.SVP3))
        vapor_hpa = qv * (pressure / 100.0) / (qv + 0.622)
        rh = 100.0 * vapor_hpa / es_hpa
    if not np.isfinite(rh).all():
        raise ValueError("diagnosed relative humidity is non-finite")
    return rh


def _mixing_ratio_to_relative_humidity(
        temperature, pressure, mixing_ratio, *, allow_wps_undershoot=False,
        column_workers=1):
    """Diagnose WPS/WRF relative humidity (%) from dry-air mixing ratio.

    HRRR supplies specific humidity.  WPS horizontally maps SPECHUMD into its
    intermediate output; real.exe's FLAG_SH branch converts it to qv_gc
    and overwrites rh_gc before vertical interpolation.  This is the algebraic
    inverse of its later ``rh_to_mxrat1`` Bolton saturation relation, before
    that routine's 0--100 percent RH clipping.
    """
    if not isinstance(allow_wps_undershoot, (bool, np.bool_)):
        raise TypeError("allow_wps_undershoot must be boolean")
    workers = _column_worker_count(column_workers)
    temperature, pressure, qv = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
        np.asarray(mixing_ratio, dtype=np.float64),
    )
    if (workers == 1 or temperature.ndim == 0
            or temperature.shape[0] < 2):
        with pm.worker_limit(workers):
            return _mixing_ratio_to_relative_humidity_serial(
                temperature, pressure, qv,
                allow_wps_undershoot=allow_wps_undershoot)

    chunks = _axis0_chunks(temperature.shape[0], workers)
    output = np.empty(temperature.shape, dtype=np.float64)
    with _column_pool(len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _mixing_ratio_to_relative_humidity_serial,
                (temperature[start:stop], pressure[start:stop],
                 qv[start:stop]),
                {"allow_wps_undershoot": allow_wps_undershoot})
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


def _potential_temperature_from_temperature_serial(temperature, pressure):
    """Evaluate WRF's T-to-theta relation over one contiguous chunk."""
    return temperature * pm.power(c.P0 / pressure, c.RCP)


def _potential_temperature_from_temperature(
        temperature, pressure, *, column_workers=1):
    """Column-parallel, byte-stable WRF T-to-theta conversion."""
    workers = _column_worker_count(column_workers)
    temperature, pressure = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
    )
    if (workers == 1 or temperature.ndim == 0
            or temperature.shape[0] < 2):
        with pm.worker_limit(workers):
            return _potential_temperature_from_temperature_serial(
                temperature, pressure)
    chunks = _axis0_chunks(temperature.shape[0], workers)
    output = np.empty(temperature.shape, dtype=np.float64)
    with _column_pool(len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _potential_temperature_from_temperature_serial,
                (temperature[start:stop], pressure[start:stop]))
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


def _temperature_from_potential_temperature_serial(theta, pressure):
    """Evaluate WRF's theta-to-T relation over one contiguous chunk."""
    return theta * pm.power(pressure / c.P0, c.RCP)


def _temperature_from_potential_temperature(
        theta, pressure, *, column_workers=1):
    """Column-parallel, byte-stable WRF theta-to-T conversion."""
    workers = _column_worker_count(column_workers)
    theta, pressure = np.broadcast_arrays(
        np.asarray(theta, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
    )
    if workers == 1 or theta.ndim == 0 or theta.shape[0] < 2:
        with pm.worker_limit(workers):
            return _temperature_from_potential_temperature_serial(
                theta, pressure)
    chunks = _axis0_chunks(theta.shape[0], workers)
    output = np.empty(theta.shape, dtype=np.float64)
    with _column_pool(len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _temperature_from_potential_temperature_serial,
                (theta[start:stop], pressure[start:stop]))
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


def _moist_specific_volume_serial(theta, qv, pressure):
    """Evaluate one contiguous chunk of WRF moist specific volume."""
    theta_m = theta * (1.0 + c.RVOVRD * qv)
    return c.RD * theta_m * pm.power(pressure / c.P0, c.RCP) / pressure


def _moist_specific_volume(theta, qv, pressure, *, column_workers=1):
    """Column-parallel, byte-stable moist specific-volume diagnostic."""
    workers = _column_worker_count(column_workers)
    theta, qv, pressure = np.broadcast_arrays(
        np.asarray(theta, dtype=np.float64),
        np.asarray(qv, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
    )
    if workers == 1 or theta.ndim == 0 or theta.shape[0] < 2:
        with pm.worker_limit(workers):
            return _moist_specific_volume_serial(theta, qv, pressure)
    chunks = _axis0_chunks(theta.shape[0], workers)
    output = np.empty(theta.shape, dtype=np.float64)
    with _column_pool(len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _moist_specific_volume_serial,
                (theta[start:stop], qv[start:stop], pressure[start:stop]))
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


def _wrf_flag_sh_surface_specific_humidity(
        q2, spfh, pressure, *, force_fallback=None):
    """Apply real.exe's whole-domain FLAG_SH surface fallback.

    WPS's surface SPECHUMD occupies metgrid level one.  WRF checks its first
    valid horizontal point; when that value is below ``1e-6``, it replaces the
    complete surface field with the nearest atmospheric SPECHUMD level before
    converting specific humidity to mixing ratio.  Native HRRR stores that
    surface value separately as Q2, so reproduce the same decision explicitly.
    """
    q2 = np.asarray(q2, dtype=np.float64)
    spfh = np.asarray(spfh, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    if spfh.ndim != 3 or pressure.shape != spfh.shape:
        raise ValueError("SPFH and pressure must share shape (level, y, x)")
    if q2.shape != spfh.shape[1:]:
        raise ValueError("Q2 must match the SPFH horizontal grid")
    nearest = _flag_sh_surface_level(q2, pressure, force_fallback)
    if nearest is None:
        return q2
    return spfh[nearest].copy()


def _flag_sh_surface_level(q2, pressure, force_fallback=None):
    """The SPFH level real.exe's FLAG_SH fallback substitutes for Q2, or None.

    One decision for the whole domain, read at its first horizontal point
    (:func:`_wrf_flag_sh_surface_specific_humidity`).  A column-blocked
    initialization makes it here once, on the whole domain's first point,
    and hands every block the same surface, so no block reads its own
    first column as the domain's.
    """
    if force_fallback is not None and not isinstance(
            force_fallback, (bool, np.bool_)):
        raise TypeError("force_fallback must be boolean or None")
    fallback = (np.float64(q2[0, 0]) < 1.0e-6
                if force_fallback is None else bool(force_fallback))
    if not fallback:
        return None
    return (0 if np.float64(pressure[-1, 0, 0]) < np.float64(pressure[0, 0, 0])
            else -1)


def _surface_relative_humidity(dewpoint, temperature):
    """ungrib's 2 m relative humidity from D2/T2 (surface RH level 1).

    WPS v4.6 ``ungrib/src/rrpr.F:compute_rh_dewpt`` (:1168-1185):
    ``RH2m = 100 * exp((Xlv/Rv) * (1/T2 - 1/D2))`` with ``Xlv = 2.5e6`` and
    ``Rv = 461.5`` -- a constant-latent-heat Clausius-Clapeyron ratio, not
    the Bolton/Magnus curve used elsewhere in this module.
    The value is deliberately NOT clipped to 100: WRF interpolates ``rh_gc``
    as delivered and clips only inside ``rh_to_mxrat1``
    (module_initialize_real.F:7392-7399), which
    :func:`_saturation_mixing_ratio` mirrors.
    """
    dewpoint = _host(dewpoint)
    temperature = _host(temperature)
    xlv_over_rv = 2.5e6 / 461.5
    return 100.0 * pm.exp(
        xlv_over_rv * (1.0 / temperature - 1.0 / dewpoint))


def surface_pressure_from_surface(psfc_in, source_orography, terrain,
                                  surface_temperature, surface_qv):
    """WRF ``sfcprs2`` surface-to-surface pressure adjustment.

    ``source_orography`` is the invariant geopotential-height surface that
    belongs to ``psfc_in`` (the retained profile's declared ``SOILHGT``
    artifact); ``terrain`` is the target WRF terrain.
    """
    psfc = _host(psfc_in)
    source_z = _host(source_orography)
    target_z = _host(terrain)
    temperature = _host(surface_temperature)
    qv = _host(surface_qv)
    if len({a.shape for a in (psfc, source_z, target_z, temperature, qv)}) != 1:
        raise ValueError("surface-pressure input shapes differ")
    # Equal source and target terrain is WRF's DEFINED behavior: sfcprs2
    # returns psfc_in unchanged (exp(0) = 1, module_initialize_real.F:8496).
    # Generic code performs that no-op; protecting a specific case against
    # a silently regressed source-orography artifact (e.g. a SOILHGT
    # regenerated byte-equal to HGT_M) is the job of that case's pinned
    # wrfinput gates, which fail by orders of magnitude on such data.
    virtual_temperature = temperature * (1.0 + 0.608 * qv)
    if (not np.isfinite(virtual_temperature).all()
            or np.any(virtual_temperature <= 0.0)):
        raise ValueError("surface virtual temperature must be finite and positive")
    result = psfc * pm.exp(
        c.G * (source_z - target_z) / (c.RD * virtual_temperature))
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError("adjusted surface pressure is invalid")
    return result


def _integrate_moisture_scalar_reference(
        qv, pressure, temperature, height, psfc, tsfc, qsfc,
        surface_height):
    """Pre-vectorization row/column oracle retained for byte-parity tests."""
    """Float64 transcription of ``module_initialize_real.F:integ_moist``.

    Inputs are pressure levels only, normalized here to bottom-to-top order;
    the separate surface values play the WPS surface pseudo-level role.
    Returns bottom-to-top dry pressure and the column-integrated vapor
    pressure removed from surface pressure.
    """
    order = np.argsort(-pressure[:, 0, 0])
    p = pressure[order].copy()
    q = qv[order].copy()
    t = temperature[order].copy()
    z = height[order].copy()
    nlev, ny, nx = p.shape
    pd = np.empty_like(p)
    intq = np.zeros((ny, nx), dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            above = np.flatnonzero(p[:, j, i] < psfc[j, i])
            if above.size == 0:
                raise ValueError(
                    f"no pressure level above the surface at column ({j}, {i})")
            ka = int(above[0])
            cumulative = np.zeros(nlev, dtype=np.float64)
            pd[-1, j, i] = p[-1, j, i]
            for k in range(nlev - 2, ka - 1, -1):
                rhobar = 0.5 * (p[k, j, i] / (c.RD * t[k, j, i])
                                + p[k + 1, j, i] / (c.RD * t[k + 1, j, i]))
                qbar = 0.5 * (q[k, j, i] + q[k + 1, j, i])
                dz = z[k + 1, j, i] - z[k, j, i]
                if dz > 0.0:
                    cumulative[k] = (cumulative[k + 1]
                                     + c.G * qbar * rhobar / (1.0 + qbar) * dz)
                else:
                    cumulative[k] = cumulative[k + 1]
                pd[k, j, i] = p[k, j, i] - cumulative[k]
            rhobar = 0.5 * (psfc[j, i] / (c.RD * tsfc[j, i])
                            + p[ka, j, i] / (c.RD * t[ka, j, i]))
            qbar = 0.5 * (qsfc[j, i] + q[ka, j, i])
            dz = z[ka, j, i] - surface_height[j, i]
            surface_intq = cumulative[ka]
            if dz > 0.1:
                surface_intq += c.G * qbar * rhobar / (1.0 + qbar) * dz
            intq[j, i] = surface_intq
            pd[:ka, j, i] = p[:ka, j, i] - surface_intq
            # ka was assigned in the loop unless it is the topmost level.
            pd[ka:, j, i] = p[ka:, j, i] - cumulative[ka:]
    return pd, intq, order


def _integrate_moisture_vectorized_slab(
        qv, pressure, temperature, height, psfc, tsfc, qsfc,
        surface_height, *, order, out_pd, out_intq, row_offset=0):
    """Evaluate one contiguous row slab of WRF ``integ_moist``.

    Vertical recurrence order remains top-down within every column, while
    NumPy evaluates all independent horizontal columns in one native loop.
    ``order`` is resolved once from the complete domain so worker partitioning
    cannot change the source-level ordering contract.
    """
    nlev, ny, nx = pressure.shape
    ka = np.full((ny, nx), nlev, dtype=np.intp)
    for k in range(nlev):
        source_k = int(order[k])
        take = (ka == nlev) & (pressure[source_k] < psfc)
        ka[take] = k
    missing = ka == nlev
    if bool(np.any(missing)):
        j, i = np.argwhere(missing)[0]
        raise ValueError(
            "no pressure level above the surface at column "
            f"({int(j) + int(row_offset)}, {i})")

    running = np.zeros_like(psfc, dtype=pressure.dtype)
    surface_intq = np.zeros_like(psfc, dtype=pressure.dtype)
    out_pd[-1] = pressure[int(order[-1])]
    for k in range(nlev - 2, -1, -1):
        source_k = int(order[k])
        source_kp1 = int(order[k + 1])
        pk = pressure[source_k]
        pkp1 = pressure[source_kp1]
        qk = qv[source_k]
        qkp1 = qv[source_kp1]
        tk = temperature[source_k]
        tkp1 = temperature[source_kp1]
        zk = height[source_k]
        zkp1 = height[source_kp1]
        active = ka <= k
        rhobar = 0.5 * (
            pk / (c.RD * tk) + pkp1 / (c.RD * tkp1))
        qbar = 0.5 * (qk + qkp1)
        dz = zkp1 - zk
        increment = c.G * qbar * rhobar / (1.0 + qbar) * dz
        running = np.where(
            active,
            running + np.where(dz > 0.0, increment, 0.0),
            0.0)
        out_pd[k] = pk - running
        surface_intq = np.where(ka == k, running, surface_intq)

    p_ka = np.empty_like(psfc, dtype=pressure.dtype)
    q_ka = np.empty_like(psfc, dtype=qv.dtype)
    t_ka = np.empty_like(psfc, dtype=temperature.dtype)
    z_ka = np.empty_like(psfc, dtype=height.dtype)
    for k in range(nlev):
        selected = ka == k
        source_k = int(order[k])
        p_ka[selected] = pressure[source_k][selected]
        q_ka[selected] = qv[source_k][selected]
        t_ka[selected] = temperature[source_k][selected]
        z_ka[selected] = height[source_k][selected]
    rhobar = 0.5 * (
        psfc / (c.RD * tsfc) + p_ka / (c.RD * t_ka))
    qbar = 0.5 * (qsfc + q_ka)
    dz = z_ka - surface_height
    surface_increment = c.G * qbar * rhobar / (1.0 + qbar) * dz
    surface_intq = np.where(
        dz > 0.1, surface_intq + surface_increment, surface_intq)
    for k in range(nlev):
        below = k < ka
        if bool(np.any(below)):
            source_k = int(order[k])
            out_pd[k, below] = (
                pressure[source_k, below] - surface_intq[below])
    out_intq[...] = surface_intq


def _integrate_moisture(qv, pressure, temperature, height, psfc, tsfc, qsfc,
                        surface_height, *, column_workers=1,
                        level_order=None, row_offset=0):
    """Column-parallel transcription of WRF ``integ_moist``.

    The original Python row/column loop is vectorized within each row slab.
    Large domains may additionally divide those independent slabs among an
    explicit number of setup-only host threads.  Every column keeps the same
    vertical operation order, and source-level ordering is resolved once from
    the complete domain, so the threaded result is byte-identical to the
    single-thread vectorized implementation.

    ``level_order`` is that domain-wide ordering, for a caller handing in
    one row block of the domain (``row_offset`` is the block's first row):
    the block's own first column is not the domain's.
    """
    workers = _column_worker_count(column_workers)
    pressure = np.asarray(pressure)
    if pressure.ndim != 3:
        raise ValueError("pressure must have shape (level, y, x)")
    order = (np.argsort(-pressure[:, 0, 0]) if level_order is None
             else np.asarray(level_order))
    pd = np.empty(pressure.shape, dtype=pressure.dtype)
    intq = np.empty(pressure.shape[1:], dtype=pressure.dtype)
    if workers == 1 or pressure.shape[1] < 2:
        _integrate_moisture_vectorized_slab(
            qv, pressure, temperature, height, psfc, tsfc, qsfc,
            surface_height, order=order, out_pd=pd, out_intq=intq,
            row_offset=row_offset)
        return pd, intq, order

    chunks = _axis0_chunks(pressure.shape[1], workers)
    with _column_pool(len(chunks)) as executor:
        futures = [
            executor.submit(
                _integrate_moisture_vectorized_slab,
                qv[:, start:stop], pressure[:, start:stop],
                temperature[:, start:stop], height[:, start:stop],
                psfc[start:stop], tsfc[start:stop], qsfc[start:stop],
                surface_height[start:stop], order=order,
                out_pd=pd[:, start:stop], out_intq=intq[start:stop],
                row_offset=row_offset + start)
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return pd, intq, order


def _make_real_base_serial(coord: VerticalCoord, terrain: np.ndarray,
                           p_top: float, base_temp: float,
                           hypsometric_opt: int = 1, *,
                           row_offset: int = 0) -> BaseState:
    """WRF ``module_initialize_real.F`` analytic hydrostatic base state.

    The base geopotential integration is keyed on ``hypsometric_opt``
    exactly as WRF (module_initialize_real.F:3811-3825): opt 1 is the
    discrete ``phb(k+1) = phb(k) - dnw(k)*(c1h*mub+c2h)*alb(k)``
    recurrence; opt 2 the log-pressure form ``phb(k+1) = phb(k) +
    alb(k)*phm*LOG(pfd/pfu)`` on the base-state reference dry pressures
    ``pf/ph = c3*MUB + c4 + p_top`` (F:3816-3822).
    """
    finalize_vertical_coord(coord, p_top)
    terrain = np.array(terrain, dtype=np.float64, copy=True)
    lapse = 50.0
    iso_temperature = 200.0
    root = (base_temp / lapse) ** 2 - (
        2.0 * c.G * terrain / (lapse * c.RD))
    if np.any(root <= 0.0):
        raise ValueError("terrain is outside the analytic base-state range")
    ps_base = c.P0 * pm.exp(-base_temp / lapse + np.sqrt(root))
    mub = ps_base - p_top
    pb = (coord.c3h[:, None, None] * mub[None]
          + coord.c4h[:, None, None] + p_top)
    # WRF checks the FULL-level reference column (nest_init_utils.F:1166-1167)
    # and this base state divides that same column at :732 -- an inverted
    # full-level pair silently produces a negative-thickness base layer, so
    # the half-level column alone is not the constraint.
    pb_full = (coord.c3f[:, None, None] * mub[None]
               + coord.c4f[:, None, None] + p_top)
    if (np.any(pb <= 0.0) or not np.all(np.diff(pb, axis=0) < 0.0)
            or not np.all(np.diff(pb_full, axis=0) < 0.0)):
        raise ValueError(hybrid_column_ordering_refusal(
            coord, p_top, ps_base, terrain=terrain,
            quantity="hybrid base pressure", row_offset=row_offset)
            or "hybrid base pressure is not monotonic")
    temperature = np.maximum(
        iso_temperature, base_temp + lapse * pm.log(pb / c.P0))
    thb = temperature * pm.power(c.P0 / pb, c.RCP)
    alb = c.RD * thb * pm.power(pb / c.P0, c.RCP) / pb
    phb = np.empty((coord.znw.size,) + terrain.shape, dtype=np.float64)
    phb[0] = c.G * terrain
    if hypsometric_opt == 1:
        for k in range(coord.dnw.size):
            phb[k + 1] = (phb[k] - coord.dnw[k]
                          * (coord.c1h[k] * mub + coord.c2h[k]) * alb[k])
    elif hypsometric_opt == 2:
        # module_initialize_real.F:3816-3822 (indices shifted to 0-based:
        # WRF's k/k-1 full levels and k-1 half level become k+1/k and k).
        for k in range(coord.dnw.size):
            pfu = coord.c3f[k + 1] * mub + coord.c4f[k + 1] + p_top
            pfd = coord.c3f[k] * mub + coord.c4f[k] + p_top
            phm = coord.c3h[k] * mub + coord.c4h[k] + p_top
            phb[k + 1] = phb[k] + alb[k] * phm * pm.log(pfd / pfu)
    else:
        raise ValueError(
            f"hypsometric_opt must be 1 or 2, got {hypsometric_opt}")
    return BaseState(mub=mub, p_top=float(p_top), pb=pb, alb=alb, thb=thb,
                     phb=phb, terrain_z=terrain)


def _fill_real_base_rows(output, coord, terrain, start, stop, p_top,
                         base_temp, hypsometric_opt):
    """Build then immediately copy one independent analytic-base row tile."""
    part = _make_real_base_serial(
        coord, terrain[start:stop], p_top, base_temp, hypsometric_opt,
        row_offset=start)
    output.mub[start:stop] = part.mub
    output.pb[:, start:stop] = part.pb
    output.alb[:, start:stop] = part.alb
    output.thb[:, start:stop] = part.thb
    output.phb[:, start:stop] = part.phb
    output.terrain_z[start:stop] = part.terrain_z


def _make_real_base(coord: VerticalCoord, terrain: np.ndarray, p_top: float,
                    base_temp: float, hypsometric_opt: int = 1, *,
                    column_workers=1) -> BaseState:
    from woof.ingest.preparation_setup import array_key, current_setup
    import struct

    owner = current_setup()
    if owner is None:
        return _make_real_base_uncached(
            coord, terrain, p_top, base_temp, hypsometric_opt,
            column_workers=column_workers)
    # Finalization is still run for every caller. Cache the finalized inputs,
    # including all coordinate arrays and scalar representations.
    finalize_vertical_coord(coord, p_top)
    key = (array_key(terrain), struct.pack('ddq', p_top, base_temp, hypsometric_opt),
           tuple((name, array_key(value)) for name, value in sorted(vars(coord).items())
                 if isinstance(value, np.ndarray)),
           tuple((name, repr(value)) for name, value in sorted(vars(coord).items())
                 if not isinstance(value, np.ndarray)))
    with owner.lock:
        if owner.closed:
            return _make_real_base_uncached(
                coord, terrain, p_top, base_temp, hypsometric_opt,
                column_workers=column_workers)
        if key not in owner.base:
            owner.base.clear()
            owner.base[key] = _make_real_base_uncached(
                coord, terrain, p_top, base_temp, hypsometric_opt,
                column_workers=column_workers)
        cached = owner.base[key]
        # Result callers keep independently writable arrays, as before.
        # Only the private analytic state is reused between forcing times.
        return BaseState(**{
            name: np.array(value, copy=True, order="K")
            if isinstance(value, np.ndarray) else value
            for name, value in vars(cached).items()})


def _make_real_base_uncached(coord: VerticalCoord, terrain: np.ndarray, p_top: float,
                    base_temp: float, hypsometric_opt: int = 1, *,
                    column_workers=1) -> BaseState:
    """Build the exact analytic base over independent horizontal chunks."""
    workers = _column_worker_count(column_workers)
    finalize_vertical_coord(coord, p_top)
    terrain = np.asarray(terrain, dtype=np.float64)
    if workers == 1 or terrain.shape[0] < 2:
        with pm.worker_limit(workers):
            return _make_real_base_serial(
                coord, terrain, p_top, base_temp, hypsometric_opt)

    ny, nx = terrain.shape
    nz = coord.dnw.size
    output = BaseState(
        mub=np.empty((ny, nx), dtype=np.float64),
        p_top=float(p_top),
        pb=np.empty((nz, ny, nx), dtype=np.float64),
        alb=np.empty((nz, ny, nx), dtype=np.float64),
        thb=np.empty((nz, ny, nx), dtype=np.float64),
        phb=np.empty((nz + 1, ny, nx), dtype=np.float64),
        terrain_z=np.empty((ny, nx), dtype=np.float64),
    )
    # More tiles than workers bounds live temporary BaseStates and avoids a
    # serial multi-gigabyte concatenate on large domains.
    chunks = _axis0_chunks(ny, min(ny, workers * 4))
    with _column_pool(workers) as executor:
        futures = [
            executor.submit(
                _fill_real_base_rows, output, coord, terrain, start, stop,
                p_top, base_temp, hypsometric_opt)
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


def _pressure_at_u(pressure):
    out = np.empty(pressure.shape[:2] + (pressure.shape[2] + 1,),
                   dtype=np.float64)
    out[..., 0] = pressure[..., 0]
    out[..., -1] = pressure[..., -1]
    out[..., 1:-1] = 0.5 * (pressure[..., :-1] + pressure[..., 1:])
    return out


def _pressure_at_v(pressure):
    out = np.empty((pressure.shape[0], pressure.shape[1] + 1,
                    pressure.shape[2]), dtype=np.float64)
    out[:, 0, :] = pressure[:, 0, :]
    out[:, -1, :] = pressure[:, -1, :]
    out[:, 1:-1, :] = 0.5 * (pressure[:, :-1, :] + pressure[:, 1:, :])
    return out


def _slice_base_rows(base: BaseState, start: int, stop: int) -> BaseState:
    """View one contiguous mass-grid row slab of an analytic base state."""
    return BaseState(
        mub=base.mub[start:stop], p_top=base.p_top,
        pb=base.pb[:, start:stop], alb=base.alb[:, start:stop],
        thb=base.thb[:, start:stop], phb=base.phb[:, start:stop],
        terrain_z=base.terrain_z[start:stop])


def _rebalance_moist_pressure_serial(pressure_guess, qtot, dry_mass, base,
                                     coord, *, out=None):
    """Integrate the discrete WRF moist w-balance pressure recurrence.

    This is the initialization counterpart of ``pg_buoy_w``: it chooses
    perturbation-pressure differences so the large-step vertical pressure
    gradient, dry-mass perturbation, and moisture loading cancel row by row.

    ``qtot`` is TOTAL water, not vapour: WRF accumulates
    ``qtot = sum(moist(i,kk,j,im), im = PARAM_FIRST_SCALAR, num_3d_m)``
    over the whole active moist package before forming ``qvf2 = 1./(1.+qtot)``
    and ``qvf1 = qtot*qvf2`` (module_initialize_real.F:3913-3916 for the
    top row, :3931-3935 for the downward leg), and the runtime it hands the
    state to agrees -- ``calc_cq``/``cq_pair`` build ``cqw`` from the same
    species list.  Vapour alone would leave the analyzed condensate's
    weight out of the column.
    """
    if out is None:
        out = np.empty_like(pressure_guess)
    elif out.shape != pressure_guess.shape or out.dtype != pressure_guess.dtype:
        raise ValueError("rebalance output shape/dtype differs from pressure")
    perturbation = out
    mup = dry_mass - base.mub
    nz = pressure_guess.shape[0]
    # WRF initializes the top half-level pressure from the rigid-lid row,
    # then integrates downward (module_initialize_real/ideal qvf1/qvf2
    # recurrence).  Anchoring at the top avoids importing horizontally
    # varying surface interpolation error into every pressure level.
    cq = 1.0 / (1.0 + qtot[-1])
    load = qtot[-1] * cq
    perturbation[-1] = (
        -0.5 * (coord.c1f[nz] * mup
                + load * (coord.c1f[nz] * base.mub + coord.c2f[nz]))
        / coord.rdnw[nz - 1] / cq)
    for k in range(nz - 2, -1, -1):
        kw = k + 1
        qbar = 0.5 * (qtot[k] + qtot[k + 1])
        cq = 1.0 / (1.0 + qbar)
        load = qbar * cq
        perturbation[k] = (
            perturbation[k + 1]
            - (coord.c1f[kw] * mup
               + load * (coord.c1f[kw] * base.mub + coord.c2f[kw]))
            / cq / coord.rdn[kw])
    np.add(base.pb, perturbation, out=out)
    if not np.isfinite(out).all() or np.any(out <= 0.0):
        raise ValueError("moist hydrostatic pressure recurrence failed")
    return out


def _rebalance_moist_pressure(pressure_guess, qtot, dry_mass, base, coord, *,
                              column_workers=1):
    """Use native columns, retaining the exact parallel row-slab reference."""
    workers = _column_worker_count(column_workers)
    from woof.ingest import host_arrays
    native = host_arrays.moist_pressure(
        pressure_guess, qtot, dry_mass, base, coord, workers=workers)
    if native is not None:
        return native
    pressure = np.empty_like(pressure_guess)
    if workers == 1 or pressure_guess.shape[1] < 2:
        return _rebalance_moist_pressure_serial(
            pressure_guess, qtot, dry_mass, base, coord, out=pressure)

    chunks = _axis0_chunks(pressure_guess.shape[1], workers)
    with _column_pool(len(chunks)) as executor:
        futures = [
            executor.submit(
                _rebalance_moist_pressure_serial,
                pressure_guess[:, start:stop], qtot[:, start:stop],
                dry_mass[start:stop], _slice_base_rows(base, start, stop),
                coord, out=pressure[:, start:stop])
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return pressure


def _half_level_height_agl(base, coord, dry_mass, alpha, *,
                           hypsometric_opt: int = 1) -> np.ndarray:
    """FP64 half-level (mass point) heights AGL of the hydrostatic column.

    The same discrete operator :func:`_fp32_geopotential_split` quantizes
    -- opt 1 ``d(phi) = -dnw*(c1h*mu+c2h)*alpha``, opt 2 ``d(phi) =
    alpha*phm*log(pfd/pfu)`` on the total dry mass -- evaluated in plain
    float64 and referenced to the surface, so the caller gets geometric
    metres above local terrain without any FP32 quantization.  Used to
    place configured initial-state perturbations on the UNPERTURBED
    column before the perturbed geopotential is formed.
    """
    if hypsometric_opt not in (1, 2):
        raise ValueError(
            f"hypsometric_opt must be 1 or 2, got {hypsometric_opt}")
    mu = np.asarray(dry_mass, dtype=np.float64)
    target = np.asarray(alpha, dtype=np.float64)
    nz = coord.dnw.size
    dphi = np.empty_like(target)
    if hypsometric_opt == 2:
        c3f = np.asarray(coord.c3f, dtype=np.float64)[:, None, None]
        c4f = np.asarray(coord.c4f, dtype=np.float64)[:, None, None]
        c3h = np.asarray(coord.c3h, dtype=np.float64)[:, None, None]
        c4h = np.asarray(coord.c4h, dtype=np.float64)[:, None, None]
        p_top = float(base.p_top)
        pfu = c3f[1:] * mu[None] + c4f[1:] + p_top
        pfd = c3f[:-1] * mu[None] + c4f[:-1] + p_top
        phm = c3h * mu[None] + c4h + p_top
        dphi[...] = target * phm * pm.log(pfd / pfu)
    else:
        dnw = np.asarray(coord.dnw, dtype=np.float64)[:, None, None]
        increment = (np.asarray(coord.c1h, dtype=np.float64)[:, None, None]
                     * mu[None]
                     + np.asarray(coord.c2h, dtype=np.float64)[:, None, None])
        dphi[...] = -dnw * increment * target
    phi_agl = np.zeros((nz + 1,) + mu.shape, dtype=np.float64)
    np.cumsum(dphi, axis=0, out=phi_agl[1:])
    return (0.5 * (phi_agl[:-1] + phi_agl[1:])) / c.G


def _fp32_geopotential_split_serial(base, coord, dry_mass, alpha,
                                    hypsometric_opt: int = 1):
    """Choose phi' ulps that best preserve the float64 hydrostatic layers.

    Over high terrain the lowest explicit eta layers are only a few metres
    thick.  Forming ``phi=phb+phi'`` in FP32 can otherwise lose enough of a
    layer geopotential difference to create an artificial vertical impulse.
    This setup-only quantizer tests the nearest phi' ulp and its two
    neighbors against the exact diagnostic-alpha equation at every level.

    The target layer thickness and the diagnostic operator are keyed on
    ``hypsometric_opt`` to match the runtime EOS (calc_p_alpha) and WRF's
    real-init geopotential integration: opt 1 inverts ``alpha =
    -d(phi)*rdnw/(c1h*mu+c2h)``; opt 2 integrates ``d(phi) =
    alt*phm*LOG(pfd/pfu)`` on the TOTAL dry-mass reference pressures
    (module_initialize_real.F:3970-3981) and diagnoses ``alt =
    d(phi)/phm/LOG(pfd/pfu)`` (F:4002-4010), all in the FP32 arithmetic
    the device kernel uses.

    "Matches the runtime EOS" is the whole contract, so the two
    cancellations calc_p_alpha stopped spelling are gone from here too:
    the layer thickness is the exact FP32 base difference plus the
    float64 residual the state carries as ``dphb_resid``, and opt 2's log
    ratio is ``log1p((pfd - pfu)/pfu)`` off the float64-differenced
    coefficient drops.  Measured on the 12-level hybrid opt-2 probe in
    tests/test_hypsometric.py (``_real_base_inputs``, terrain 200-350 m),
    re-diagnosing with the operator calc_p_alpha actually runs: quantizing
    against THIS operator leaves 1.46 ulps of the target alpha, quantizing
    against the pre-change one leaves 15.17 -- a tenth of that test's
    16-ulp bound versus within 5 % of it.  The search would otherwise be
    optimizing a rounding tree nobody runs.
    """
    if hypsometric_opt not in (1, 2):
        raise ValueError(
            f"hypsometric_opt must be 1 or 2, got {hypsometric_opt}")
    phb64 = np.asarray(base.phb, dtype=np.float64)
    phb = np.asarray(base.phb, dtype=np.float32)
    # The state's dphb_resid, built here by the same two lines
    # DomainState.set_base_geopotential uses, so the search optimizes the
    # exact thickness the kernel will reconstruct.
    resid = np.asarray(np.diff(phb64, axis=0)
                       - np.diff(phb, axis=0).astype(np.float64),
                       dtype=np.float32)
    dnw = np.asarray(coord.dnw, dtype=np.float32)
    rdnw = np.asarray(coord.rdnw, dtype=np.float32)
    increment = np.asarray(
        coord.c1h[:, None, None] * dry_mass[None]
        + coord.c2h[:, None, None], dtype=np.float32)
    target = np.asarray(alpha, dtype=np.float32)
    if hypsometric_opt == 2:
        # Per-layer reference dry pressures on the TOTAL dry mass (WRF
        # MU0 = mub + mu'), in the kernel's FP32 arithmetic.
        c3f64 = np.asarray(coord.c3f, dtype=np.float64)
        c4f64 = np.asarray(coord.c4f, dtype=np.float64)
        c3f = np.asarray(coord.c3f, dtype=np.float32)
        c4f = np.asarray(coord.c4f, dtype=np.float32)
        c3h = np.asarray(coord.c3h, dtype=np.float32)
        c4h = np.asarray(coord.c4h, dtype=np.float32)
        dc3f = np.asarray(c3f64[:-1] - c3f64[1:], dtype=np.float32)
        dc4f = np.asarray(c4f64[:-1] - c4f64[1:], dtype=np.float32)
        mu32 = np.asarray(dry_mass, dtype=np.float32)
        pt32 = np.float32(base.p_top)
    php = np.zeros_like(phb, dtype=np.float32)
    for k in range(coord.dnw.size):
        # The kernel's own base thickness: an exact FP32 difference plus
        # the float64 residual.
        dphb = np.asarray(
            np.asarray(phb[k + 1] - phb[k], dtype=np.float32) + resid[k],
            dtype=np.float32)
        if hypsometric_opt == 2:
            pfu = c3f[k + 1] * mu32 + c4f[k + 1] + pt32
            phm = c3h[k] * mu32 + c4h[k] + pt32
            dpf = np.asarray(dc3f[k] * mu32 + dc4f[k],   # = pfd - pfu
                             dtype=np.float32)
            # The CPU equation of state's own log1pf, so the split is
            # searched against the operator that diagnoses it on every host.
            log_ratio = log1pf_array(np.asarray(dpf / pfu, dtype=np.float32))
            desired_dphi = np.asarray(target[k] * phm * log_ratio,
                                      dtype=np.float32)
        else:
            desired_dphi = np.asarray(-dnw[k] * increment[k] * target[k],
                                      dtype=np.float32)
        centre = np.asarray(
            php[k] + np.asarray(desired_dphi - dphb, dtype=np.float32),
            dtype=np.float32)
        candidates = (
            np.nextafter(centre, np.float32(-np.inf)), centre,
            np.nextafter(centre, np.float32(np.inf)),
        )
        best = None
        best_error = None
        for candidate in candidates:
            dphi = np.asarray(
                dphb + np.asarray(candidate - php[k], dtype=np.float32),
                dtype=np.float32)
            if hypsometric_opt == 2:
                diagnosed = np.asarray(
                    np.asarray(dphi / phm, dtype=np.float32) / log_ratio,
                    dtype=np.float32)
            else:
                diagnosed = np.asarray(
                    np.asarray(-dphi * rdnw[k], dtype=np.float32)
                    / increment[k], dtype=np.float32)
            error = np.abs(diagnosed.astype(np.float64)
                           - target[k].astype(np.float64))
            if best is None:
                best, best_error = candidate, error
            else:
                choose = error < best_error
                best = np.where(choose, candidate, best)
                best_error = np.where(choose, error, best_error)
        php[k + 1] = best
    return php


def _fp32_geopotential_split(base, coord, dry_mass, alpha,
                             hypsometric_opt: int = 1, *,
                             column_workers=1):
    """Quantize geopotential independently over parallel mass-grid slabs."""
    workers = _column_worker_count(column_workers)
    if workers == 1 or dry_mass.shape[0] < 2:
        return _fp32_geopotential_split_serial(
            base, coord, dry_mass, alpha, hypsometric_opt)

    chunks = _axis0_chunks(dry_mass.shape[0], workers)
    with _column_pool(len(chunks)) as executor:
        futures = [
            executor.submit(
                _fp32_geopotential_split_serial,
                _slice_base_rows(base, start, stop), coord,
                dry_mass[start:stop], alpha[:, start:stop],
                hypsometric_opt)
            for start, stop in chunks
        ]
        parts = [future.result() for future in futures]
    return np.concatenate(parts, axis=1)


@dataclass(frozen=True)
class RealInitResult:
    """Initialization outputs.

    Boundary-only results leave fingerprint payloads and disposition
    empty, aerosol fingerprints empty, and total_geopotential None.
    State and its external scalar boundary inventory retain full behavior.
    """
    state: DomainState
    coord: VerticalCoord
    base: BaseState
    surface_pressure: np.ndarray
    surface_qv: np.ndarray
    dry_mass: np.ndarray
    dry_pressure: np.ndarray
    total_pressure: np.ndarray
    total_geopotential: np.ndarray | None
    total_specific_volume: np.ndarray
    integrated_moisture_pressure: np.ndarray
    #: The hypsometric_opt the geopotential/base construction was keyed on;
    #: hydrostatic_residual grades the state against the same operator.
    hypsometric_opt: int = 1
    #: Immutable source/interpolated/state correspondence for native HRRR
    #: analyzed hydrometeors.  Other real-source routes carry an empty map.
    hydrometeor_initialization: dict[str, object] = field(default_factory=dict)
    #: mp_physics=28 only: what this initialization did and did not put in
    #: the aerosol fields, and which named function is expected to fill them
    #: next.  Empty for every other scheme.  Deliberately NOT folded into
    #: ``hydrometeor_initialization``: the aerosol policy applies to both the
    #: native-HRRR lane and the pressure-level/RH lane, and that map only
    #: exists on the former.
    aerosol_initialization: dict[str, object] = field(default_factory=dict)
    #: What the start state's W is: the zero policy
    #: (WRF_REAL_VERTICAL_VELOCITY_POLICY) or the carried source field with
    #: its measured range (vertical_velocity_receipt).  Kept beside, not
    #: inside, ``hydrometeor_initialization`` for the same reason as the
    #: aerosol receipt: W is carried on any route whose source publishes
    #: one, and that map only exists on the analyzed-hydrometeor lane.
    vertical_velocity_initialization: dict[str, object] = field(default_factory=dict)
    #: How many cells of ``surface_qv`` ABOVE were floored at WRF's
    #: ``qv_min_value`` on the way out, and how far below it they were
    #: (:func:`_floor_flag_sh_surface_mixing_ratio`).  Only the FLAG_SH
    #: lane can populate this: it is the one lane whose surface value WPS's
    #: overshooting sixteen_pt operator can push below zero and real.exe
    #: never floors.  Empty when nothing was floored -- the usual case --
    #: and always empty on the RH lane, whose
    #: :func:`_saturation_mixing_ratio` applies the same floor inline and
    #: so has no separate divergence to report.
    surface_moisture_floor: dict[str, object] = field(default_factory=dict)
    #: What the configured [perturbation] bubbles actually wrote into THIS
    #: domain's initial theta/qv: per-bubble cells touched and max deltas
    #: (:mod:`woof.ingest.init_perturbation`).  Empty when no perturbation
    #: was requested -- the OFF path stores nothing and runs nothing.
    initial_perturbation: dict[str, object] = field(default_factory=dict)
    #: The use_sh_qv lane's interpolated vapour cells that WRF's
    #: second-order vertical operator took below zero and this
    #: initialization floored at WRF's ``qv_min_value``
    #: (:func:`_floor_sh_vertical_undershoot`): the count, the levels, the
    #: minimum and where it was.  Empty when nothing was floored, and
    #: always empty on the RH lane, whose :func:`_saturation_mixing_ratio`
    #: floors inline.
    prognostic_moisture_floor: dict[str, object] = field(default_factory=dict)
    #: Row-unit unstaggered eta fields for IC and boundary table callers.
    boundary_tracers: dict[str, object] = field(default_factory=dict)
    #: One receipt per chem boundary source that filled this frame's chem
    #: rows (:func:`woof.chem_source_init.fill_boundary_sources`): the
    #: frames, weights, files and operators.  Empty on a chem-off run.
    chem_initialization: tuple = ()


def _wif_grid_latlon_from(grid, state):
    """``(lat2d, lon2d)`` for the WIF ingest from whatever the caller holds.

    FOUR CARRIERS, one meaning.  The tree does not have a single geodesy
    object every real front door passes around: ``woof run`` and the
    tilestream cases hold a projection ``Grid``; the direct GFS/ERA5/
    mapped runners hold a geogrid ``static`` mapping; the prepared road
    restores a state that already carries the mass-point mirror
    ``initialize_prepared_physics`` stamps on it.  Accepting all of them
    is what makes this derivation reach every door instead of the one
    that happened to be wired first, and every one of them is the SAME
    mass-point lat/lon -- there is no per-source variant here, only a
    per-caller container.

    ``None`` when the caller holds none of them, which the caller above
    turns into a named refusal rather than a guess.
    """

    for candidate in (grid, state):
        if candidate is None:
            continue
        latlon_mass = getattr(candidate, "latlon_mass", None)
        if callable(latlon_mass):
            lat, lon = latlon_mass()
            return (lat, lon)
        latitude = getattr(candidate, "latitude_deg", None)
        longitude = getattr(candidate, "longitude_deg", None)
        if latitude is not None and longitude is not None:
            return (latitude, longitude)
        try:
            names = ("XLAT_M", "XLONG_M") if "XLAT_M" in candidate                 else ("XLAT", "XLONG")
            return (candidate[names[0]], candidate[names[1]])
        except (TypeError, KeyError, IndexError):
            pass
        if isinstance(candidate, (tuple, list)) and len(candidate) == 2:
            return (candidate[0], candidate[1])
    return None


#: The donor fields besides its aerosol numbers that
#: :func:`initialize_real` reads from an ``aerosol_snapshot``: its own
#: dry-pressure coordinate.
AEROSOL_DONOR_COORDINATE_FIELDS = ("PRES", "SPFH", "Q2", "PSFC", "T2", "TT",
                                   "GHT", "SOURCE_OROGRAPHY")


def aerosol_donor_snapshot(snapshot):
    """``snapshot`` cut to the fields an aerosol donor is read for.

    The aerosol numbers with their surface pseudo-levels, and the donor's
    own dry-pressure coordinate (:data:`AEROSOL_DONOR_COORDINATE_FIELDS`);
    :func:`initialize_real` reads nothing else from a donor.  THE BREAKAGE:
    a preparation that takes its aerosol from the boundary analysis kept
    that whole mapped analysis alive through the initial analysis' decode
    and build, about 20 GiB of fields nothing would read again at 4160 x
    2450 (pi-a4, box G).
    """
    import copy
    from dataclasses import is_dataclass, replace
    from woof.ingest.analyzed_numbers import AEROSOL_NUMBER_FIELDS

    keep = (*AEROSOL_DONOR_COORDINATE_FIELDS, *AEROSOL_NUMBER_FIELDS,
            *(name + "_SFC" for name in AEROSOL_NUMBER_FIELDS))
    fields = {name: value for name, value in snapshot.fields.items() if name in keep}
    if is_dataclass(snapshot):
        return replace(snapshot, fields=fields)
    donor = copy.copy(snapshot)
    donor.fields = fields
    return donor


#: How many columns one block of a CPU initialization builds at least
#: (:func:`initialize_real`).  At 50 levels a block's float64 work arrays
#: come to about 3.5 GiB, against about 174 GiB for a 4160 x 2450 root
#: built whole.
INITIALIZATION_BLOCK_COLUMNS = 1 << 18
_INITIALIZATION_MIN_BLOCK_ROWS = 16
#: The rows each column worker takes of one block where host memory
#: allows it.  THE BREAKAGE: blocks of INITIALIZATION_BLOCK_COLUMNS alone
#: gave each of 48 workers one or two rows of a 4160-wide block, so the
#: setup columns' per-level NumPy steps ran on a few thousand elements
#: each, the threads queued on the interpreter lock, and a boundary-only
#: 1 km forcing time took 352 s where the whole-domain build took 145 s
#: (pi-a4b, box J); at 16 rows per worker it took 126-130 s.
_INITIALIZATION_MIN_ROWS_PER_WORKER = 16
#: The share of available host memory one block's work arrays may take
#: before the block is narrowed toward INITIALIZATION_BLOCK_COLUMNS.
_INITIALIZATION_BLOCK_MEMORY_SHARE = 0.125


def initialization_column_bytes(cfg, source_levels: int) -> int:
    """Host bytes one column's float64 work arrays take while its block is built.

    Measured on the 4160 x 2450 x 50 root with 50 source levels: about 174
    GiB of work arrays over 10,192,000 columns built whole, 18 KB a column,
    which is 23 float64 values per model or source level and per column.
    """
    return 8 * 23 * (int(cfg.nz) + int(source_levels) + 2)


def _initialization_host_available():
    """Host bytes this preparation may still take (cgroup-aware), or None."""
    try:
        from woof.ingest.preparation_workers import host_available_bytes
        return host_available_bytes()
    except Exception:  # an unreadable probe sizes for speed, as before
        return None


def _initialization_block_rows(cfg, requested, *, eligible, workers=1,
                               source_levels=None, available=None) -> int:
    """Rows per column block of one initialization; ``cfg.ny`` builds whole.

    ``requested`` is the caller's explicit count.  ``None`` sizes it: the
    fastest block, :data:`_INITIALIZATION_MIN_ROWS_PER_WORKER` rows for
    each column worker (``workers``), when its work arrays
    (:func:`initialization_column_bytes`) fit
    :data:`_INITIALIZATION_BLOCK_MEMORY_SHARE` of ``available`` host bytes;
    narrower where they do not, down to :data:`INITIALIZATION_BLOCK_COLUMNS`,
    the least host memory the blocks take.  Every count gives the same bytes.
    Only the CPU preparation's host state is built in blocks
    (``eligible``); asking for blocks anywhere else is refused, because the
    card routes hold their state on the device and a perturbed start places
    its bubbles on the whole domain.
    """
    ny, nx = int(cfg.ny), int(cfg.nx)
    if requested is not None:
        if isinstance(requested, (bool, np.bool_)) or not isinstance(
                requested, (int, np.integer)):
            raise TypeError("column_block_rows must be an integer")
        if int(requested) < 1:
            raise ValueError("column_block_rows must be positive")
        if int(requested) < ny and not eligible:
            raise ValueError(
                "column_block_rows builds the CPU preparation's host state in "
                "row blocks; this initialization holds its state on the card "
                "or perturbs the whole domain, so it is built whole")
        return min(int(requested), ny)
    if not eligible:
        return ny
    least = max(_INITIALIZATION_MIN_BLOCK_ROWS,
                -(-INITIALIZATION_BLOCK_COLUMNS // max(1, nx)))
    rows = max(least, _INITIALIZATION_MIN_ROWS_PER_WORKER * max(1, int(workers)))
    if available is not None and source_levels is not None:
        affordable = int(_INITIALIZATION_BLOCK_MEMORY_SHARE * max(0, int(available))) // (
            initialization_column_bytes(cfg, source_levels) * max(1, nx))
        rows = max(least, min(rows, affordable))
    return ny if 2 * rows > ny else rows


@preprocess_math_call
def initialize_real(snapshot: HorizontalSnapshot, cfg: RunConfig,
                    coord: VerticalCoord, terrain, *, source_orography=None,
                    p_top=5000.0, sfcp_to_sfcp=True,
                    use_sh_qv=None,
                    analyzed_species=None,
                    analyzed_number_fields=None,
                    analyzed_surface_fields=(),
                    column_workers=None,
                    preprocess_backend="cuda",
                    preprocess_workers=None,
                    cpu_bridge=None,
                    state_backend="cuda",
                    flag_sh_surface_fallback=None,
                    timing_report=None,
                    scratch_arena=None,
                    dycore_state_workspace=None,
                    initial_perturbation=None,
                    grid=None,
                    wif_grid_latlon=None,
                    wif_valid_date=None,
                    aerosol_snapshot=None,
                    boundary_only=False,
                    landmask=None,
                    boundary_species=(),
                    column_block_rows=None,
                    receipt_joins=None) -> RealInitResult:
    """Construct a moist, discretely hydrostatic :class:`DomainState`.

    ``receipt_joins``, a list, defers the hydrometeor disposition receipt's
    join to the caller: its slot is reserved in ``hydrometeor_initialization``
    and a zero-argument join is appended, which the caller runs before it
    hands the result on (a child joins it after its blend, so the receipt
    thread runs beside the blend instead of the blend waiting for it).

    ``analyzed_species`` optionally declares the actual analyzed mass inventory
    from source metadata. An empty tuple declares no analyzed mass fields;
    absent species retain WRF's allocated zero. None uses the snapshot's
    declaration when present, otherwise preserving the established native
    caller contract, including its required five-species inventory.

    ``analyzed_number_fields`` declares flagged metgrid QNI/QNC/QNR/QNS/QNG/QNH
    inputs. They use WRF's same linear Q vertical operator and selected scalar
    package, with the supplied NAME_SFC surface pseudo-level. They do not contribute condensate to
    the pressure recurrence. Omission discovers the fields carried by the snapshot.

    ``aerosol_snapshot`` supplies a separate same-time aerosol donor on the
    target horizontal grid. Its own pressure and moisture columns remain
    attached to its QNWFA/QNIFA through vertical interpolation, before the
    cold-start droplet closure reads the initialized aerosol.

    ``analyzed_surface_fields`` selects supplied NAME_SFC mass pseudo-levels.
    Native HRRR callers retain their established exact-zero surface policy.

    ``boundary_species`` names the hydrometeor masses (state names) the
    SOURCE publishes on every forcing frame, read from its table row
    (:func:`woof.boundary_fields.mapping_boundary_species`,
    :func:`woof.boundary_fields.source_boundary_species`).  Those this
    initializer installs from the analysis, with the number moments the
    cold start seeds from them, join the state's lateral-boundary
    inventory, so the root's specified boundary carries the analysed
    cloud, rain, ice, snow and graupel instead of draining them.  The
    default, none, keeps water vapour only (WRF's have_bcs_moist off).

    ``landmask`` is the target grid's static LANDMASK (1 land, 0 water).
    An mp=28 cold start reads it as WRF's XLAND where a cloudy cell holds
    no aerosol, because real.exe's make_DropletNumber then sizes the
    droplets by land or water; such a start without it is refused.

    The pressure-level/RH lane requires TT, RH, GHT, UU, VV, PSFC, T2,
    exactly one of D2 or RH2, U10, and V10.  Native HRRR requires
    per-column PRES, SPFH, and Q2, and for
    WSM6/Thompson/Morrison/NSSL-2 requires analyzed QC/QR/QI/QS/QG.
    Kessler requires the same decoded inventory, retains QC/QR, and records
    WRF-real's explicit active-moist-package discard of QI/QS/QG.  MP off is
    refused because it cannot faithfully retain the five analyzed species.
    Classic
    Thompson retains the five analyzed HRRR mass categories, and its
    source-absent QNICE/QNRAIN moments are seeded as real.exe seeds them:
    make_RainNumber and make_IceNumber where rain or ice mass is present
    and the number is at or below zero, exact zero where there is no mass.
    Aerosol-aware Thompson (mp_physics=28) retains the same five
    categories -- its Registry moist package is character for character
    mp=8's (Registry.EM_COMMON:3024 vs :3036) -- adds nc, seeded from
    make_DropletNumber the same way, and deliberately leaves
    nwfa/nifa/nwfa2d/nifa2d at exact zero for ``microphysics_init`` to
    fill from thompson_init's
    synthetic profile; ``RealInitResult.aerosol_initialization`` is the
    receipt for that hand-off.  WRF's default
    ``use_sh_qv = .false.`` vertically interpolates HRRR RH and then diagnoses
    qv. Omitted ``use_sh_qv`` follows the snapshot's explicit
    ``specific_humidity_authority`` metadata; unmarked sources keep WRF's
    False default. An explicit boolean overrides that declaration. With
    ``use_sh_qv=False``, RH is diagnosed from the already horizontally mapped
    SPFH, temperature, and pressure exactly where real.exe handles FLAG_SH.
    Surface fields build WRF's ``use_surface`` pseudo-level that anchors
    every vertical-interpolation column.
    ``source_orography`` is mandatory for ``sfcp_to_sfcp`` unless the
    horizontal snapshot carries catalog-resolved ``SOURCE_OROGRAPHY``.
    ``preprocess_backend`` selects the interpolation engine independently of
    ``state_backend``.  The latter defaults to ``'cuda'`` so a GPU forecast
    may use CPU transforms and still receive the historical device state.
    Native export callers select ``state_backend='preprocess'``: CPU
    preprocessing then retains the completed setup/export state in NumPy
    host memory, while CUDA preprocessing keeps it on device.
    ``initial_perturbation`` is an optional
    :class:`woof.ingest.init_perturbation.InitialStatePerturbation`:
    when present its theta bubbles are applied once to the final
    theta/qv columns before alpha/geopotential are formed (constant
    analyzed pressure and column mass, geopotential rebalanced -- the
    em_quarter_ss convention) and the per-bubble application stats are
    returned as ``RealInitResult.initial_perturbation``.  ``None`` (the
    default) runs not one instruction of that path.
    ``column_workers`` is how many host threads the float64 setup columns
    use; ``None`` (the default) is every CPU this process may run on for a
    preparation, capped by its CPU and memory budget. An explicit worker
    count reaches both backends. Every count gives the same bytes.
    ``boundary_only`` marks a forcing time whose state only feeds the
    lateral boundaries: its caller keeps the state and drops the result's
    receipts, so the hydrometeor vertical disposition receipt, one operator
    replay per source level over the whole domain, is not built for it.
    The state is the same either way.
    ``column_block_rows`` is how many rows of columns the CPU preparation
    builds at once (:func:`_initialization_block_rows`); ``None`` sizes the
    blocks from the domain.  Every count gives the same bytes; only the
    host RAM the work arrays take changes.  The card and bounded routes,
    and a perturbed start, build the whole domain at once.
    """
    # The CPU implementation remains the reference. Device twins share only
    # this orchestration; no host helper is substituted on the CPU route.
    np = globals()["np"]
    # ``np`` is CuPy on the card route below; a host-state test needs NumPy.
    host_numpy = np
    from woof.ingest import host_arrays as _host_arrays
    _ops = sys.modules[__name__]
    _setup = _host
    route_name = (preprocess_backend.strip().lower() if isinstance(preprocess_backend, str)
                  else getattr(preprocess_backend, "name", None))
    bounded_cuda = bool(getattr(preprocess_backend, "bounded_cuda", False))
    state_route = state_backend.strip().lower() if isinstance(state_backend, str) else None
    device_route = (
        route_name == "cuda" and state_route in ("cuda", "preprocess")
        and {"PRES", "SPFH", "Q2"}.issubset(snapshot.fields)
        and initial_perturbation is None and cfg.mp_physics not in THOMPSON_AEROSOL_FAMILY
        and bool(sfcp_to_sfcp))
    release_completed_host = bounded_cuda or route_name == "cpu"
    if device_route:
        if bounded_cuda:
            _ops = preprocess_backend.real_ops
        else:
            from woof.ingest import real_device as _ops
        np = _ops._cp()
        _setup = _ops.widen
    if not isinstance(boundary_only, (bool, np.bool_)):
        raise TypeError("boundary_only must be boolean")
    def receipt_fingerprint(value):
        # Boundary callers retain only state. Checks stay at their owners.
        return {} if boundary_only else array_correspondence_fingerprint(value)

    def receipt_fingerprints_of(values):
        # receipt_fingerprint over several independent arrays, side by side.
        values = tuple(values)
        return ([{} for _ in values] if boundary_only
                else _fingerprints_side_by_side(values))

    def receipt_fingerprints(names, value_for):
        names = tuple(names)
        if boundary_only:
            return {name: {} for name in names}
        from woof.ingest.preparation_fingerprints import array_fingerprints
        if route_name != "cpu" or state_xp is not host_numpy:
            # The device route hashed one field at a time on the calling
            # thread (7 s of the HRRR CONUS start on one core).  The host
            # copies of up to eight fields now hash side by side; each is
            # still one ordered SHA-256 stream, so every receipt is the one
            # array_correspondence_fingerprint wrote.
            result = {}
            for start in range(0, len(names), 8):
                batch = names[start:start + 8]
                arrays = []
                for name in batch:
                    value = value_for(name)
                    if hasattr(value, "get"):
                        value = value.get()
                    arrays.append(host_numpy.ascontiguousarray(host_numpy.asarray(value)))
                receipts = (None if any(array.size == 0 for array in arrays) else
                            array_fingerprints(arrays, workers=max(
                                1, min(int(column_workers or 1), len(arrays)))))
                if receipts is None:
                    receipts = [receipt_fingerprint(array) for array in arrays]
                result.update(zip(batch, receipts))
                del arrays
            return result
        result = {}
        # Normalize at most two fields at a time. A five-species source may
        # hold float64 fields whose FP32 receipt copies are each substantial.
        for start in range(0, len(names), 2):
            batch = names[start:start + 2]
            arrays = [host_numpy.ascontiguousarray(host_numpy.asarray(value_for(name)))
                      for name in batch]
            receipts = array_fingerprints(arrays, workers=min(column_workers, 2))
            if receipts is None:
                receipts = [receipt_fingerprint(array) for array in arrays]
            result.update(zip(batch, receipts))
            del arrays
        return result

    from woof.boundary_fields import BOUNDARY_HYDROMETEOR_MASSES
    if (not isinstance(boundary_species, (tuple, list))
            or any(name not in BOUNDARY_HYDROMETEOR_MASSES
                   for name in boundary_species)):
        raise ValueError(
            "boundary_species must list state hydrometeor masses from "
            f"{BOUNDARY_HYDROMETEOR_MASSES}, got {boundary_species!r}")
    if timing_report is not None:
        if not isinstance(timing_report, MutableMapping):
            raise TypeError("timing_report must be a mutable mapping")
        if len(timing_report):
            raise ValueError("timing_report must be empty")
    timing_start = timing_last = perf_counter()

    def mark_timing(name):
        # A column-blocked initialization passes each stage once per block;
        # the report keeps one total per stage.
        nonlocal timing_last
        now = perf_counter()
        if timing_report is not None:
            timing_report[name] = timing_report.get(name, 0.0) + (now - timing_last)
        timing_last = now

    if not isinstance(sfcp_to_sfcp, (bool, np.bool_)):
        raise TypeError("sfcp_to_sfcp must be a boolean")
    if not cfg.moist:
        raise ValueError("real initialization requires cfg.moist=True")
    # THE ANALYZED NUMBER INVENTORY IS RESOLVED HERE, before the mp=28
    # floor below, because the floor's question depends on the answer: an
    # analysis carrying its own QNWFA/QNIFA pair opens no climatology, so
    # "is the climatology installed on this machine" is not a
    # precondition of that run at all.  Resolving it after the floor is
    # what made the analyzed route unreachable on the one configuration
    # it was written for.
    from woof.ingest.analyzed_numbers import (
        AEROSOL_NUMBER_FIELDS, METGRID_NUMBER_FIELDS, analyzed_aerosol_fields,
        analyzed_aerosol_way_out, metgrid_number_targets,
        wif_climatology_named_by_namelist,
        WIF_CLIMATOLOGY_NAMELIST_PAIR, WIF_CLIMATOLOGY_NAMELIST_PHRASE)
    if analyzed_number_fields is None:
        analyzed_number_fields = tuple(name for name in METGRID_NUMBER_FIELDS
                                       if name in snapshot.fields)
    if (not isinstance(analyzed_number_fields, (tuple, list))
            or any(name not in METGRID_NUMBER_FIELDS for name in analyzed_number_fields)
            or len(set(analyzed_number_fields)) != len(analyzed_number_fields)):
        raise ValueError("analyzed_number_fields must list distinct supported metgrid number fields")
    donor_numbers = ()
    if aerosol_snapshot is not None:
        if aerosol_snapshot.valid_time != snapshot.valid_time:
            raise ValueError("aerosol donor and initial analysis must have the same valid time")
        donor_numbers = tuple(name for name in AEROSOL_NUMBER_FIELDS
                              if name in aerosol_snapshot.fields)
        if len(donor_numbers) != len(AEROSOL_NUMBER_FIELDS):
            raise ValueError("aerosol donor must carry both QNWFA and QNIFA; a half pair would mix different aerosol sources")
    decoded_numbers = tuple(name for name in METGRID_NUMBER_FIELDS
                            if name in analyzed_number_fields or name in donor_numbers)
    main_numbers = tuple(name for name in decoded_numbers if name not in donor_numbers)
    number_targets = metgrid_number_targets(cfg) if decoded_numbers else {}
    if cfg.mp_physics in THOMPSON_AEROSOL_FAMILY:
        # The aerosol-source selectors decide what this function is even
        # allowed to leave in nwfa/nifa, so they are checked HERE and not
        # only in validate_run_config: initialize_real is reachable with a
        # RunConfig that never went through it, and an mp=28 ingest that
        # silently accepted wif_input_opt=1 would promise a WIF metgrid
        # stream nothing in this module can read and then hand the
        # microphysics an all-zero aerosol field as if that were the
        # requested climatology.  Refusing by name is the same posture the
        # namelist importer already prints, which since development branch
        # prints a RESOLUTION (MP28_AEROSOL_SOURCE_DEFAULT) or a named
        # fallback (MP28_AEROSOL_SYNTHETIC_FALLBACK) rather than a
        # deviation notice.
        validate_aerosol_source_options(cfg)
        # AND THE RUN DOOR'S half: the machine-dependent preconditions.
        # ``validate_run_config`` deliberately does not ask them -- it is
        # also the namelist importer's battery, and that door emits a
        # TOML, reads no dataset and cannot be handed the way out -- so
        # the question "is WRF's monthly aerosol climatology installed on
        # this machine" is asked where nwfa/nifa would otherwise be filled
        # with the synthetic profile and a specified-BC domain handed
        # zero-inflow aerosol boundaries.
        #
        # THIS IS THE FLOOR, NOT THE FRONT DOOR.  It catches a RunConfig
        # that reached initialization without passing any door -- but it
        # is late on every route a user types: in the prepare stage of
        # `woof go` (after the cycle has been fetched and its manifest
        # verified) and inside runtime's time loop on `woof run`.  The
        # same sentence is raised from the same inventory at the doors
        # that commit to building a forecast, before they spend anything
        # (``woof.config.validate_experiment_preparation``).
        #
        # AND IT IS NOT ASKED OF A RUN THAT READS NO CLIMATOLOGY.  When
        # this analysis carries the whole QNWFA/QNIFA pair and nobody
        # named a source, the aerosol comes from the analysis
        # (``_metgrid_analyzed_aerosol`` below), the resolver is never
        # called and no dataset is searched for, so a precondition about
        # an uninstalled dataset would refuse a run that does not want
        # it.  The condition is the SAME pair of facts the resolver
        # decides on, read through the same two helpers, so the floor and
        # the resolver cannot disagree about one analysis.
        _carried_aerosol = analyzed_aerosol_fields(
            decoded_numbers, number_targets)
        if (_resolved_mp28_aerosol_source(cfg) == "analysis"
                and len(_carried_aerosol) != len(AEROSOL_NUMBER_FIELDS)):
            missing = sorted(set(AEROSOL_NUMBER_FIELDS) - set(_carried_aerosol))
            # A source that PUBLISHED the pair but whose decoder withheld
            # it (a GRIB2 bitmap masking points, with no fill policy) says
            # so: "missing" alone sends the operator looking for fields
            # the file in hand carries.
            withheld = getattr(snapshot, "withheld_fields", None) or {}
            reasons = sorted({withheld[name] for name in missing if name in withheld})
            published = ("; the source published " + ", ".join(name for name in missing if name in withheld)
                         + " with a GRIB2 bitmap (masked points) on this cycle and the decoder withheld them: "
                         + "; ".join(reasons)) if reasons else ""
            raise ValueError("analyzed aerosol IC/BC requires QNWFA and QNIFA on every initial and boundary frame; missing "
                             + ", ".join(missing) + "; substituting climatology would change the requested forcing"
                             + published)
        if cfg.use_rap_aero_icbc and not boundary_only:
            validate_run_preparation(cfg)
        if not (len(_carried_aerosol) == len(AEROSOL_NUMBER_FIELDS)
                and _resolved_mp28_aerosol_source(cfg) in ("auto", "analysis")):
            try:
                validate_run_preparation(cfg)
            except ValueError as refused:
                # The precondition sentence lives in woof/config.py and
                # predates the analyzed route, so it names three ways out
                # and not the fourth.  Naming it HERE, where the analysis
                # in hand says which half is missing, is what keeps the
                # refusal actionable for a met_em operator.
                #
                # AND IT NAMES THE SELECTOR THAT ACTUALLY ASKED.  A
                # namelist carrying (aer_init_opt, wif_input_opt) = (1, 1)
                # reaches this floor with mp28_aerosol_source at its
                # default 'auto', so "leave mp28_aerosol_source at 'auto'"
                # would be an instruction the operator has already
                # followed; the pair is what resolved the source and the
                # pair is what the sentence has to name.
                _way_out = analyzed_aerosol_way_out(
                    _carried_aerosol,
                    named_by_namelist=wif_climatology_named_by_namelist(cfg))
                raise ValueError(f"{refused} {_way_out}") from None
    if use_sh_qv is None:
        use_sh_qv = getattr(snapshot, "specific_humidity_authority", False)
    if not isinstance(use_sh_qv, (bool, np.bool_)):
        raise TypeError("use_sh_qv must be boolean")
    use_sh_qv = bool(use_sh_qv)
    if analyzed_species is None:
        analyzed_species = getattr(snapshot, "analyzed_species", None)
    # Every host column helper below splits its arrays into contiguous
    # slabs whose results are byte-identical to the serial evaluation
    # (tests/test_real_init.py), so the count changes only the wall time
    # and, on the CPU preparation, the host RAM it peaks at.
    # Serial by default left a 1792 x 1024 x 55 domain's float64 setup
    # columns on one core while the rest of the machine, and the GPU,
    # waited.
    column_workers = _column_worker_count(
        _default_column_workers(preprocess_backend, preprocess_workers)
        if column_workers is None else column_workers)
    if cfg.nz != coord.dnw.size:
        raise ValueError("RunConfig.nz and vertical coordinate differ")
    finalize_vertical_coord(coord, float(p_top))
    terrain = _setup(terrain)
    if terrain.shape != (cfg.ny, cfg.nx):
        raise ValueError("terrain must have shape (ny, nx)")
    specific_markers = ("PRES", "SPFH", "Q2")
    marker_count = sum(name in snapshot.fields for name in specific_markers)
    if marker_count not in (0, len(specific_markers)):
        present = [name for name in specific_markers if name in snapshot.fields]
        missing_specific = [name for name in specific_markers
                            if name not in snapshot.fields]
        raise KeyError(
            "partial specific-humidity forcing inventory: "
            f"present={present}, missing={missing_specific}")
    has_specific_humidity = marker_count == len(specific_markers)
    if use_sh_qv and not has_specific_humidity:
        raise ValueError(
            "use_sh_qv=True requires PRES, SPFH, and Q2 forcing")
    if analyzed_species is None:
        decoded_species = (HRRR_ANALYZED_HYDROMETEORS
                           if has_specific_humidity and cfg.mp_physics in HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS
                           else ())
    else:
        if not isinstance(analyzed_species, (tuple, list)) or any(
                name not in DECLARED_ANALYZED_HYDROMETEORS for name in analyzed_species):
            raise ValueError("analyzed_species must explicitly list supported analyzed mass fields")
        if len(set(analyzed_species)) != len(analyzed_species):
            raise ValueError("analyzed_species repeats a field")
        decoded_species = tuple(name for name in DECLARED_ANALYZED_HYDROMETEORS if name in analyzed_species)
        if decoded_species and cfg.mp_physics != 0 and cfg.mp_physics not in HRRR_ANALYZED_HYDROMETEOR_MOIST_PACKAGE:
            raise ValueError(f"mp_physics={cfg.mp_physics} has no analyzed-mass disposition contract for {decoded_species}")
    if (not isinstance(analyzed_surface_fields, (tuple, list))
            or any(name not in decoded_species for name in analyzed_surface_fields)
            or len(set(analyzed_surface_fields)) != len(analyzed_surface_fields)):
        raise ValueError("analyzed_surface_fields must list distinct declared analyzed mass fields")
    supplied_mass_surfaces = tuple(name for name in decoded_species if name in analyzed_surface_fields)
    if has_specific_humidity:
        if cfg.mp_physics == 0 and analyzed_species is None:
            raise ValueError(
                "native HRRR preparation with mp_physics=0 is refused: "
                "the MP-off state cannot faithfully retain analyzed "
                "QC/QR/QI/QS/QG, and a radiation-only analyzed-cloud "
                "carrier is out of scope")
        required = ("TT", "PRES", "SPFH", "GHT", "UU", "VV", "PSFC",
                    "T2", "Q2", "U10", "V10")
    else:
        surface_rh_markers = tuple(
            name for name in ("D2", "RH2") if name in snapshot.fields)
        if len(surface_rh_markers) != 1:
            raise KeyError(
                "pressure-level RH forcing requires exactly one of D2 or RH2")
        surface_rh_name = surface_rh_markers[0]
        required = ("TT", "RH", "GHT", "UU", "VV", "PSFC", "T2",
                    surface_rh_name, "U10", "V10")
    required += decoded_species + main_numbers + tuple(name+"_SFC" for name in (*main_numbers, *supplied_mass_surfaces))
    if not sfcp_to_sfcp:
        required += ("PMSL",)
    missing = [name for name in required if name not in snapshot.fields]
    if missing:
        raise KeyError(f"missing real-data field(s): {missing}")
    # Only fields consumed by float64 WRF-real setup are materialized on the
    # host.  Winds and analyzed hydrometeors remain in their mapped FP32
    # device representation until the FP32 vertical interpolation.  The old
    # FP32 -> host-FP64 -> device-FP32 round trip changed no bits but cost
    # gigabytes of transfer and host residency on large domains.
    host_required = {"TT", "GHT", "PSFC", "T2"}
    if not sfcp_to_sfcp:
        host_required.add("PMSL")
    if has_specific_humidity:
        host_required.update({"PRES", "SPFH", "Q2"})
    else:
        host_required.update({"RH", surface_rh_name})
    # COLUMN BLOCKS.  Every step from the source moisture to the specific
    # volume is column-local (the winds' v staggering reads one row below,
    # which each block carries as a halo row), so the CPU preparation builds
    # them one block of rows at a time and keeps only the finished fields
    # whole.  THE BREAKAGE: built whole, a 4160 x 2450 x 50 root held about
    # 174 GiB of float64 work arrays in this function at once (the 1 km
    # CONUS preparation, pi-a4 on box G), so every 1 km preparation needed
    # 115-245 GB of host RAM and two of them were SIGKILLed together by a
    # container's memory limit.  Blocks give the same bytes: the
    # domain-wide decisions (source level order, the FLAG_SH surface
    # fallback, the analytic base state) are made once on the whole domain,
    # and every receipt is taken on the assembled whole arrays.
    block_rows = _initialization_block_rows(
        cfg, column_block_rows, eligible=(
            route_name == "cpu" and not bounded_cuda and not device_route
            and state_route in ("cpu", "preprocess")
            and initial_perturbation is None and scratch_arena is None
            and dycore_state_workspace is None), workers=column_workers,
        source_levels=snapshot.levels_hpa.size,
        available=(None if column_block_rows is not None or route_name != "cpu"
                   else _initialization_host_available()))
    blocked = block_rows < cfg.ny
    fields = {
        name: (_setup(snapshot.fields[name])
               if name in host_required
               and not (blocked and np.ndim(snapshot.fields[name]) == 3)
               else snapshot.fields[name])
        for name in required
    }
    # The source's vertical velocity, when the join packed one: geometric
    # (WW, m s-1) is taken ahead of omega (OMEGA, Pa s-1) because it needs
    # no conversion.  Neither is required; a source without one keeps the
    # zero start (WRF_REAL_VERTICAL_VELOCITY_POLICY).
    vertical_velocity_source = next(
        (name for name in ("WW", "OMEGA") if name in snapshot.fields), None)
    if vertical_velocity_source is not None:
        fields[vertical_velocity_source] = snapshot.fields[vertical_velocity_source]
    else:
        vertical_velocity_source = None
    # Source precedence is explicit: a case may declare an artifact OR use
    # the forcing catalog's validated era5_z_invariant provider.  Silently
    # replacing a declaration would make provenance depend on GRIB inventory.
    if source_orography is not None and "SOURCE_OROGRAPHY" in snapshot.fields:
        raise ValueError(
            "source-orography conflict: both declared source_orography "
            "argument and forcing catalog SOURCE_OROGRAPHY "
            "(era5_z_invariant/SOILGEO) are present; declare exactly one")
    if "SOURCE_OROGRAPHY" in snapshot.fields:
        source_orography = _setup(snapshot.fields["SOURCE_OROGRAPHY"])
    mark_timing("validate_and_materialize_host_fields")
    nsource = snapshot.levels_hpa.size
    mass_shape = (nsource, cfg.ny, cfg.nx)
    mass_names = ["TT", "GHT"]
    mass_names += (["PRES", "SPFH"] if has_specific_humidity else ["RH"])
    mass_names += list(decoded_species) + list(main_numbers)
    if vertical_velocity_source is not None:
        mass_names.append(vertical_velocity_source)
    if any(fields[name].shape != mass_shape for name in mass_names):
        raise ValueError(
            f"mass-field shapes do not match levels and mass grid: {mass_names}")
    if any(fields[name+"_SFC"].shape != (cfg.ny, cfg.nx) for name in main_numbers):
        raise ValueError("number-field surface pseudo-levels must match the mass grid")
    if any(fields[name+"_SFC"].shape != (cfg.ny, cfg.nx) for name in supplied_mass_surfaces):
        raise ValueError("hydrometeor surface pseudo-levels must match the mass grid")
    if fields["UU"].shape != (nsource, cfg.ny, cfg.nx + 1):
        raise ValueError("UU does not have WRF u staggering")
    if fields["VV"].shape != (nsource, cfg.ny + 1, cfg.nx):
        raise ValueError("VV does not have WRF v staggering")
    if fields["U10"].shape != (cfg.ny, cfg.nx + 1):
        raise ValueError("U10 does not have WRF u staggering")
    if fields["V10"].shape != (cfg.ny + 1, cfg.nx):
        raise ValueError("V10 does not have WRF v staggering")
    block_chunks = (_axis0_chunks(cfg.ny, -(-cfg.ny // block_rows))
                    if blocked else ((0, cfg.ny),))
    if has_specific_humidity and blocked:
        # The whole-domain check each block build below makes on its rows.
        for row_start, row_stop in block_chunks:
            part = fields["PRES"][:, row_start:row_stop]
            if not np.isfinite(part).all() or np.any(part <= 0.0):
                raise ValueError("PRES must be finite and positive")
    if source_orography is None:
        raise ValueError("source_orography is required for the analyzed surface and column moisture")

    # Backend-selected FP32 vertical interpolation, after the float64
    # pressure/mass setup.
    # WRF (interp_theta=F defaults) interpolates TEMPERATURE in LOG(p) with
    # the t_extrap_type=2 below-ground branch (module_initialize_real.F:
    # 1784-1802), full pressure linearly in p through the same machinery
    # (:1805-1820, var type 'T' so it shares the temperature extrapolation
    # branch), RH in the default LOG(p) with constant extrapolation
    # (:1736-1748), and converts T -> theta only afterwards with the
    # interpolated pressure (t_to_theta, :1862-1867).
    preprocess = resolve_preprocess_backend(
        preprocess_backend, workers=preprocess_workers,
        cpu_bridge=cpu_bridge)
    backend_xp = preprocess.array_module
    if not isinstance(state_backend, str):
        raise TypeError("state_backend must be 'cuda', 'cpu', or 'preprocess'")
    normalized_state_backend = state_backend.strip().lower()
    if normalized_state_backend == "preprocess":
        state_xp = backend_xp
    elif normalized_state_backend == "cpu":
        state_xp = np
    elif normalized_state_backend == "cuda":
        import cupy as cp
        state_xp = cp
    else:
        raise ValueError(
            "state_backend must be 'cuda', 'cpu', or 'preprocess'")
    plan_float32 = _ops.float32 if device_route else preprocess.float32

    def every_block_valid(value, check):
        """``check`` holds on ``value`` one row block at a time."""
        if not blocked:
            return check(value)
        return all(check(value[..., row_start:row_stop, :])
                   for row_start, row_stop in block_chunks)

    # The analyzed inventory's whole-domain refusals, made once before any
    # column is built (a block sees only its own rows).
    retained_names = ()
    moist_package = None
    if decoded_species:
        invalid_source = []
        for name in decoded_species:
            if not every_block_valid(fields[name], lambda part: (
                    bool(backend_xp.isfinite(plan_float32(part)).all())
                    and not bool((plan_float32(part) < 0.0).any()))):
                invalid_source.append(name)
            if name in supplied_mass_surfaces:
                surface = plan_float32(fields[name+"_SFC"])
                if not bool(backend_xp.isfinite(surface).all()) or bool((surface < 0).any()):
                    invalid_source.append(name+"_SFC")
        if invalid_source:
            raise ValueError(
                "mapped HRRR hydrometeor forcing is non-finite or negative: "
                f"{invalid_source}")
        # Retention is READ from the Registry-package mapping, never decided
        # here: the membership test above and the species this loop writes
        # have to be answers to the same question, and an if-ladder is how
        # they stop being.  The lookup cannot miss -- the mapping's keys ARE
        # HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS, pinned by
        # test_the_analyzed_inventory_tuple_and_its_moist_packages_agree.
        moist_package = ({"registry_citation": "Registry/Registry.EM_COMMON:3014",
                          "retained": ()}
                         if analyzed_species is not None and cfg.mp_physics == 0
                         else HRRR_ANALYZED_HYDROMETEOR_MOIST_PACKAGE[int(cfg.mp_physics)])
        retained_names = tuple(name for name in moist_package["retained"] if name in decoded_species)
        if "QH" in decoded_species:
            from woof.core.nest_fields import nest_field_kinds
            if "qh" in nest_field_kinds(cfg):
                retained_names += ("QH",)
    required_donor = ()
    if decoded_numbers and donor_numbers:
        # Aerosols retain their donor's own dry-pressure coordinate.
        # Pairing donor numbers with the initial meteorology's PRES
        # silently shifts every layer when their hybrid ladders differ.
        required_donor = (*AEROSOL_DONOR_COORDINATE_FIELDS, *donor_numbers,
                          *(name + "_SFC" for name in donor_numbers))
        missing = sorted(set(required_donor) - set(aerosol_snapshot.fields))
        if missing:
            raise ValueError(f"aerosol donor dry-pressure coordinate is missing {missing}")
        donor_shape = (len(aerosol_snapshot.levels_hpa), cfg.ny, cfg.nx)
        for name in ("PRES", "SPFH", "TT", "GHT", *donor_numbers):
            if aerosol_snapshot.fields[name].shape != donor_shape:
                raise ValueError(f"aerosol donor {name} must match its own levels and the target mass grid")
        for name in ("PSFC", "Q2", "T2", "SOURCE_OROGRAPHY",
                     *(name + "_SFC" for name in donor_numbers)):
            if aerosol_snapshot.fields[name].shape != (cfg.ny, cfg.nx):
                raise ValueError(f"aerosol donor {name} must match the target mass grid")
    if decoded_numbers:
        for name in decoded_numbers:
            number_fields = (aerosol_snapshot.fields if name in donor_numbers
                             else fields)
            if any(not every_block_valid(value, lambda part: (
                    bool(backend_xp.isfinite(plan_float32(part)).all())
                    and not bool((plan_float32(part) < 0).any())))
                   for value in (number_fields[name], number_fields[name+"_SFC"])):
                raise ValueError(f"analyzed number field {name} is non-finite or negative")

    # THE DOMAIN-WIDE DECISIONS a row block cannot make from its own rows:
    # the source level order and the FLAG_SH surface fallback are read at
    # the domain's first column (integ_moist, sfcprs3, real.exe's FLAG_SH
    # test), and the analytic base state is built once and sliced.
    level_order = sea_level_order = donor_level_order = None
    surface_specific_whole = donor_surface_specific_whole = None
    host_base_whole = None
    if blocked:
        if has_specific_humidity:
            first_column = _host(fields["PRES"][:, 0:1, 0:1])
            surface_level = _flag_sh_surface_level(
                fields["Q2"], first_column, flag_sh_surface_fallback)
            surface_specific_whole = (
                np.asarray(fields["Q2"], dtype=np.float64) if surface_level is None
                else _host(fields["SPFH"][surface_level]).copy())
        else:
            first_column = np.broadcast_to(
                np.asarray(snapshot.levels_hpa)[:, None, None] * 100.0,
                (nsource, 1, 1)).copy()
        level_order = np.argsort(-np.asarray(first_column)[:, 0, 0])
        sea_level_order = np.argsort(
            -np.asarray(first_column, dtype=np.float64)[:, 0, 0])
        if required_donor:
            donor_first = _host(aerosol_snapshot.fields["PRES"][:, 0:1, 0:1])
            donor_level = _flag_sh_surface_level(
                _host(aerosol_snapshot.fields["Q2"][0:1, 0:1]), donor_first)
            donor_surface_specific_whole = (
                _host(aerosol_snapshot.fields["Q2"]) if donor_level is None
                else _host(aerosol_snapshot.fields["SPFH"][donor_level]).copy())
            donor_level_order = np.argsort(-donor_first[:, 0, 0])
        host_base_whole = _make_real_base(
            coord, _host(terrain), float(p_top), cfg.base_temp,
            hypsometric_opt=cfg.hypsometric_opt,
            column_workers=column_workers)

    def initialize_columns(start, stop, owned_start=0, owned_stop=None):
        """Every column-local field of rows ``start:stop`` (all rows, whole).

        Unblocked, this is the initialization's own column phase on the
        whole domain.  Blocked, the block owns rows ``owned_start:stop`` and
        ``start`` may sit one row above them: the halo row the v-point
        pressures average with.
        """
        def rows(value, staggered_v=False):
            if not blocked or value is None:
                return value
            return value[..., start:stop + (1 if staggered_v else 0), :]

        fields_b = {}
        for name, value in fields.items():
            part = rows(value, staggered_v=name in ("VV", "V10"))
            if blocked and name in host_required and np.ndim(value) == 3:
                part = _setup(part)
            fields_b[name] = part
        terrain_b = rows(terrain)
        orography_b = rows(source_orography)
        block_shape = (stop - start, cfg.nx) if blocked else (cfg.ny, cfg.nx)
        offsets = {"level_order": level_order, "row_offset": start} if blocked else {}
        out = {}

        if has_specific_humidity:
            pressure = fields_b["PRES"]
            if not blocked and (
                    not np.isfinite(pressure).all() or np.any(pressure <= 0.0)):
                raise ValueError("PRES must be finite and positive")
            surface_specific = (rows(surface_specific_whole) if blocked else
                                _ops._wrf_flag_sh_surface_specific_humidity(
                                    fields_b["Q2"], fields_b["SPFH"], pressure,
                                    force_fallback=flag_sh_surface_fallback))
            # The WPS undershoot envelope is ADMITTED here, exactly as WRF
            # admits it, and it stays admitted through sfcprs2, integ_moist
            # and the vertical-interpolation pseudo-level, because those are
            # the uses real.exe is DEFINED for and it hands them the raw
            # value.  Only the value this function PUBLISHES as Q2 is floored,
            # and that happens once at the result, below.
            surface_qv = _ops._specific_humidity_to_mixing_ratio(
                surface_specific, allow_wps_undershoot=True,
                column_workers=column_workers)
            # The same envelope on both use_sh_qv lanes, because the SAME
            # operator makes the undershoot whether or not the source claims
            # specific-humidity authority.  The surface conversion above keeps
            # the WPS gate constant instead: its operand is bilinear-mapped
            # Q2, which carries no negative weights and cannot undershoot, so
            # there is no envelope of its own to record.  This used to be
            # `not use_sh_qv`, which refused the direct-qv lane any horizontal
            # undershoot at all -- and the undershoot is ours:
            # `fields["SPFH"]` is already mapped onto the model grid by WPS's
            # overshooting sixteen_pt.  Whether a window met one was a
            # question of how much ground it covered, so an ERA5 window sized
            # for a 16 GiB card refused in initialization while the same
            # centre sized for an 8 GiB card ran.  Everything below zero is
            # floored at WRF's own qv_min before it is published, by
            # _floor_sh_vertical_undershoot and
            # _floor_flag_sh_surface_mixing_ratio.
            source_qv = _ops._specific_humidity_to_mixing_ratio(
                fields_b["SPFH"], allow_wps_undershoot=True,
                undershoot_floor=getattr(
                    snapshot, "specific_humidity_undershoot_floor", None),
                column_workers=column_workers)
        else:
            pressure = np.broadcast_to(
                np.asarray(snapshot.levels_hpa)[:, None, None] * 100.0,
                (nsource,) + block_shape).copy()
            if surface_rh_name == "D2":
                surface_qv = _ops._saturation_mixing_ratio(
                    fields_b["D2"], fields_b["PSFC"], 100.0,
                    column_workers=column_workers)
            else:
                surface_qv = _ops._saturation_mixing_ratio(
                    fields_b["T2"], fields_b["PSFC"], fields_b["RH2"],
                    column_workers=column_workers)
            source_qv = _ops._cap_stratospheric_qv(
                _ops._saturation_mixing_ratio(
                    fields_b["TT"], pressure, fields_b["RH"],
                    column_workers=column_workers),
                pressure, column_workers=column_workers)
        if sfcp_to_sfcp:
            with pm.worker_limit(column_workers):
                surface_pressure = _ops.surface_pressure_from_surface(
                    fields_b["PSFC"], orography_b, terrain_b, fields_b["T2"],
                    surface_qv)
        else:
            from woof.ingest.cpu_backend import CpuPreprocessBackend
            surface_pressure = _setup(CpuPreprocessBackend().surface_pressure_from_sea_level(
                pressure, fields_b["GHT"], terrain_b, fields_b["PMSL"], workers=column_workers,
                **({"level_order": sea_level_order, "row_offset": start} if blocked else {})))
        # WRF integrates moisture on the ORIGINAL met surface (integ_moist is
        # called with p_gc whose level 1 is the met PSFC on SOILHGT,
        # module_initialize_real.F:1457/7022); only p_dts (:1482) pairs the
        # sfcprs2-adjusted psfc with the resulting intq for the dry mass.
        source_pd, intq, order = _ops._integrate_moisture(
            source_qv, pressure, fields_b["TT"], fields_b["GHT"], fields_b["PSFC"],
            fields_b["T2"], surface_qv,
            _setup(orography_b) if orography_b is not None else terrain_b,
            column_workers=column_workers, **offsets)
        pressure = _ops._ordered_levels(pressure, order)
        source_temperature = _ops._ordered_levels(fields_b["TT"], order)
        source_qv = _ops._ordered_levels(source_qv, order)
        if use_sh_qv:
            source_rh = None
        elif has_specific_humidity:
            # module_initialize_real.F:1138-1167: FLAG_SH first converts the
            # horizontally mapped SPECHUMD to qv_gc, then overwrites rh_gc at the
            # same target points.  Deriving RH on the HRRR source grid and mapping
            # it separately is not equivalent because both transforms are
            # nonlinear.  Negative WPS SPFH undershoots produce negative RH here;
            # rh_to_mxrat1 clips that RH to zero only after vertical interpolation.
            source_rh = _ops._mixing_ratio_to_relative_humidity(
                source_temperature, pressure, source_qv,
                allow_wps_undershoot=True,
                column_workers=column_workers)
        else:
            source_rh = _ops._ordered_levels(fields_b["RH"], order)
        # WRF's vert-interp source column carries the surface pseudo-level at
        # the met-source dry surface pressure pd_gc(:,1,:) = p_gc(:,1,:) - intq
        # (integ_moist:7130), i.e. the ORIGINAL met PSFC minus the vapor column.
        surface_pd = fields_b["PSFC"] - intq
        if use_sh_qv:
            surface_rh = None
        elif has_specific_humidity:
            surface_rh = _ops._mixing_ratio_to_relative_humidity(
                fields_b["T2"], fields_b["PSFC"], surface_qv,
                allow_wps_undershoot=True,
                column_workers=column_workers)
        elif surface_rh_name == "D2":
            with pm.worker_limit(column_workers):
                surface_rh = _ops._surface_relative_humidity(fields_b["D2"], fields_b["T2"])
        else:
            surface_rh = fields_b["RH2"].copy()
        dry_mass = surface_pressure - intq - float(p_top)
        if np.any(dry_mass <= 0.0):
            raise ValueError("non-positive dry column mass")
        if device_route:
            dry_pressure = _ops.dry_pressure_ladder(dry_mass, coord.c3h, coord.c4h, p_top)
            dry_pressure_full = _ops.dry_pressure_ladder(dry_mass, coord.c3f, coord.c4f, p_top)
        else:
            dry_pressure = (coord.c3h[:, None, None] * dry_mass[None]
                            + coord.c4h[:, None, None] + float(p_top))
            dry_pressure_full = (coord.c3f[:, None, None] * dry_mass[None]
                                 + coord.c4f[:, None, None] + float(p_top))
        if (not np.all(np.diff(dry_pressure, axis=0) < 0.0)
                or not np.all(np.diff(dry_pressure_full, axis=0) < 0.0)):
            raise ValueError(hybrid_column_ordering_refusal(
                coord, float(p_top), _host(dry_mass + float(p_top)),
                quantity="target dry pressure", row_offset=start)
                or "target dry pressure is not monotonic")
        if release_completed_host:
            del dry_pressure_full
        mark_timing("source_moisture_and_dry_mass")

        order_backend = backend_xp.asarray(order, dtype=backend_xp.int32)

        def backend_ordered_levels(value):
            value = plan_float32(value)
            if backend_xp is host_numpy and value.ndim == 3 and value.size >= _RECEIPT_HASH_MIN_CELLS:
                # The same take, one source level per copy, side by side: on the
                # host roads a CONUS volume per field was one 4.6 s single-core
                # np.take (bounded CUDA road, a 5090 host, 2026-10-21).
                out = host_numpy.empty((len(order), *value.shape[1:]), dtype=value.dtype)
                def copy(level):
                    out[level] = value[int(order[level])]
                from os import cpu_count
                with ThreadPoolExecutor(max_workers=max(1, min(16, cpu_count() or 1)),
                                        thread_name_prefix="ordered-levels") as pool:
                    list(pool.map(copy, range(len(order))))
                return out
            return backend_xp.take(value, order_backend, axis=0)

        mass_source_pd_f32 = plan_float32(source_pd)
        mass_surface_pd_f32 = plan_float32(surface_pd)
        mass_target_pd_f32 = plan_float32(dry_pressure)
        mass_vertical_plan = preprocess.prepare_wrf_vertical(
            mass_source_pd_f32, mass_surface_pd_f32, mass_target_pd_f32)
        temperature = mass_vertical_plan.apply(
            plan_float32(source_temperature),
            plan_float32(fields_b["T2"]),
            interp_in_logp=True, extrap="temperature")
        if use_sh_qv:
            qv = mass_vertical_plan.apply(
                plan_float32(source_qv),
                plan_float32(surface_qv),
                interp_in_logp=True, extrap="constant")
            rh = None
        else:
            rh = mass_vertical_plan.apply(
                plan_float32(source_rh),
                plan_float32(surface_rh),
                interp_in_logp=True, extrap="constant")
            qv = None
        total_pressure = mass_vertical_plan.apply(
            plan_float32(pressure),
            plan_float32(fields_b["PSFC"]),
            interp_in_logp=False, extrap="temperature")
        # The source's vertical velocity rides the same mass-level plan as
        # pressure (linear in p); below the source's lowest level it holds
        # that level's value, and the surface pseudo-level is zero, the
        # kinematic ground value, so nothing is invented beneath the source.
        source_w_target = None
        if vertical_velocity_source is not None:
            source_w_target = mass_vertical_plan.apply(
                backend_ordered_levels(fields_b[vertical_velocity_source]),
                plan_float32(np.zeros(block_shape, dtype=np.float32)),
                interp_in_logp=False, extrap="constant")
        temperature_h = (_setup(temperature).astype(np.float64) if device_route
                         else _host_owned_float64(temperature))
        rh_h = (None if rh is None else
                _setup(rh).astype(np.float64) if device_route else
                _host_owned_float64(rh))
        total_pressure_h = (_setup(total_pressure).astype(np.float64) if device_route
                            else _host_owned_float64(total_pressure))
        # NOT clamped to the target dry pressure.  This line used to read
        # ``np.maximum(total_pressure_h, dry_pressure)``; real.exe has no such
        # step.  WRF interpolates p_gc onto pd_gc with var_type 'T' and
        # t_extrap_type=2 (module_initialize_real.F:1795-1807) and hands the
        # RESULT straight to rh_to_mxrat1 (:1827) and t_to_theta (:1850-1853)
        # -- there is no MAX, no floor and no comparison against grid%pb
        # anywhere between the vert_interp call and those two uses.  The clamp
        # mattered because theta is formed HERE, once, and never recomputed:
        # the two-pass moist rebalance below replaces total_pressure_h but
        # leaves theta_h standing, so a pressure raised at this line is baked
        # into the initial state permanently.  It fired exactly where the
        # target column's lowest eta levels fall BELOW the source column's
        # surface pseudo-level, which is the below-surface extrapolation
        # branch, and in that branch WRF's own CRC formula deliberately
        # returns a pressure lower than the target dry pressure.  Measured
        # against the real.exe that read the same forcing, on the 2021-12-30
        # 3 km case, the clamp displaced theta by up to 3.56 K over 3229
        # columns; the oracle that measured it is
        # woof/verify/metem_differential.py, graded by
        # tests/test_metem_differential.py.  (The upstream file token for
        # that forcing is deliberately not spelled here: tests/test_runtime.py
        # ::test_generic_ingest_and_runtime_have_no_metgrid-output_reader asserts
        # nothing under woof/ingest names it, and a comment that did would
        # retire that instrument.)
        if (not np.isfinite(total_pressure_h).all()
                or np.any(total_pressure_h <= 0.0)):
            raise ValueError(
                "interpolated total pressure is non-finite or non-positive: "
                "the vertical interpolation of p_gc produced a value no "
                "thermodynamic conversion can use.  This is a defect in the "
                "forcing column, not a value to be clamped away")
        theta_h = _ops._potential_temperature_from_temperature(
            temperature_h, total_pressure_h,
            column_workers=column_workers)
        prognostic_qv_floor: dict[str, object] = {}
        if use_sh_qv:
            qv_h = _setup(qv).astype(np.float64)
            if blocked:
                _refuse_non_finite_prognostic_qv(qv_h, row_offset=start)
                # Floored here on the block's every row; counted only on the
                # rows the block owns, and receipted once for the domain.
                owned = slice(owned_start - start, stop - start)
                out["prognostic_qv_floor_stats"] = _sh_vertical_undershoot_stats(
                    qv_h[:, owned], total_pressure_h[:, owned],
                    row_offset=owned_start)
                qv_h = np.where(qv_h < 0.0, _WRF_QV_MIN_VALUE, qv_h)
            else:
                _ops._refuse_non_finite_prognostic_qv(qv_h)
                # What reaches the plan was admitted against the horizontal
                # operator's own undershoot envelope for the source it mapped, so
                # a negative here is a polynomial's own: this vertical one's at a
                # sharp dry slot, or the horizontal one's carried into it, within
                # that envelope.  WRF's qv_min floor, the one the RH lane gets
                # from rh_to_mxrat1, is applied here and receipted
                # (_PROGNOSTIC_QV_FLOOR_WRF_REFERENCE).
                qv_h, prognostic_qv_floor = _ops._floor_sh_vertical_undershoot(
                    qv_h, total_pressure_h)
        else:
            qv_h = _ops._cap_stratospheric_qv(
                _ops._saturation_mixing_ratio(
                    temperature_h, total_pressure_h, rh_h,
                    column_workers=column_workers),
                total_pressure_h, column_workers=column_workers)
        mark_timing("thermodynamic_vertical_interpolation")
        # The carried vertical velocity on the target mass levels, in m s-1,
        # converted from omega with this column's own interpolated T, qv and
        # p (vertical_velocity_from_omega) or taken as published (WW).  Held
        # as one FP32 field until the state's W interfaces are formed below;
        # the receipt it fills is the one hydrometeor_initialization carries.
        w_mass_h = None
        if source_w_target is not None:
            carried_w = (_setup(source_w_target).astype(np.float64) if device_route
                         else _host_owned_float64(source_w_target))
            if vertical_velocity_source == "OMEGA":
                carried_w = vertical_velocity_from_omega(
                    carried_w, temperature_h, qv_h, total_pressure_h, xp=np)
            if not bool(np.isfinite(carried_w).all()):
                raise ValueError(
                    f"the source vertical velocity ({vertical_velocity_source}) "
                    "interpolated to a non-finite value: a defect in the forcing "
                    "column, not a value to be clamped away")
            w_mass_h = carried_w.astype(np.float32)
            del carried_w, source_w_target
            if release_completed_host:
                fields_b[vertical_velocity_source] = rows(snapshot.fields[vertical_velocity_source])
        mark_timing("vertical_velocity_conversion")
        if release_completed_host:
            # Their final consumers have finished. Retaining widened source
            # fields beside the finished host state needlessly doubled host RAM.
            del source_temperature, source_qv, source_rh, pressure
            del temperature, rh, total_pressure, qv
            for name in ("TT", "GHT", "PRES", "SPFH", "RH"):
                if name in fields_b:
                    fields_b[name] = rows(snapshot.fields[name])

        # WRF interpolates the analyzed hydrometeors BEFORE the moist
        # pressure recurrence -- the vert_interp calls for QR/QC/QI/QS/QG/QH
        # are at module_initialize_real.F:1862-1982 and the recurrence at
        # :3908 -- because that recurrence loads TOTAL water (:3913-3916),
        # not vapour.  The order is essential, not cosmetic.
        hydrometeors = {}
        if decoded_species:
            # WRF's Q interpolation uses the metgrid surface pseudo-level.
            # Native HRRR has no supplied surface analysis and retains zero.
            # vboundb above the target keeps the shared kernel linear throughout.
            zero_surface = backend_xp.zeros(block_shape, dtype=backend_xp.float32)
            for name in retained_names:
                value = mass_vertical_plan.apply(
                    backend_ordered_levels(fields_b[name]),
                    (plan_float32(fields_b[name+"_SFC"])
                     if name in supplied_mass_surfaces else zero_surface),
                    interp_in_logp=True, extrap="constant",
                    vboundb=cfg.nz + 1, values_are_finite=True)
                if (not bool(backend_xp.isfinite(value).all())
                        or bool((value < 0.0).any())):
                    raise ValueError(
                        f"interpolated HRRR hydrometeor {name} is invalid")
                hydrometeors[name] = value
            if not blocked:
                out["hydrometeor_initialization"] = hydrometeor_receipts(
                    order, mass_source_pd_f32, mass_surface_pd_f32,
                    mass_target_pd_f32, mass_vertical_plan, hydrometeors,
                    zero_surface)
        from woof.ingest.native_extras import interpolate_boundary_tracers
        # A snapshot-shaped namespace (the HRRR runner's boundary strips, a
        # detached mapped hour) may carry no tracer mapping at all: no tracers.
        source_tracers = {name: rows(value) for name, value in
                          (getattr(snapshot, "boundary_tracers", None) or {}).items()}
        boundary_tracers = interpolate_boundary_tracers(
            source_tracers, mass_vertical_plan, backend_ordered_levels,
            backend_xp.zeros(block_shape, dtype=backend_xp.float32), cfg.nz
        ) if source_tracers else {}
        mark_timing("hydrometeor_vertical_interpolation")

        number_moments = {}
        if decoded_numbers:
            if required_donor:
                donor_fields = {
                    name: _setup(rows(aerosol_snapshot.fields[name]))
                    for name in required_donor}
                donor_surface_specific = (
                    rows(donor_surface_specific_whole) if blocked else
                    _ops._wrf_flag_sh_surface_specific_humidity(
                        donor_fields["Q2"], donor_fields["SPFH"], donor_fields["PRES"]))
                donor_surface_qv = _ops._specific_humidity_to_mixing_ratio(
                    donor_surface_specific, allow_wps_undershoot=True,
                    column_workers=column_workers)
                donor_qv = _ops._specific_humidity_to_mixing_ratio(
                    donor_fields["SPFH"], allow_wps_undershoot=True,
                    undershoot_floor=getattr(aerosol_snapshot, "specific_humidity_undershoot_floor", None),
                    column_workers=column_workers)
                donor_pd, donor_intq, donor_order = _ops._integrate_moisture(
                    donor_qv, donor_fields["PRES"], donor_fields["TT"],
                    donor_fields["GHT"], donor_fields["PSFC"], donor_fields["T2"],
                    donor_surface_qv, donor_fields["SOURCE_OROGRAPHY"],
                    column_workers=column_workers,
                    **({"level_order": donor_level_order, "row_offset": start}
                       if blocked else {}))
                donor_plan = preprocess.prepare_wrf_vertical(
                    plan_float32(donor_pd),
                    plan_float32(donor_fields["PSFC"] - donor_intq),
                    mass_target_pd_f32)
                donor_order_backend = backend_xp.asarray(donor_order, dtype=backend_xp.int32)
            for name in decoded_numbers:
                if name not in number_targets:
                    continue
                number_fields = donor_fields if name in donor_numbers else fields_b
                surface = plan_float32(number_fields[name+"_SFC"])
                number_plan = donor_plan if name in donor_numbers else mass_vertical_plan
                ordered_number = (backend_xp.take(plan_float32(number_fields[name]),
                                                  donor_order_backend, axis=0)
                                  if name in donor_numbers
                                  else backend_ordered_levels(fields_b[name]))
                value = number_plan.apply(
                    ordered_number, surface,
                    interp_in_logp=True, extrap="constant",
                    vboundb=cfg.nz + 1, values_are_finite=True)
                if not bool(backend_xp.isfinite(value).all()) or bool((value < 0).any()):
                    raise ValueError(f"interpolated number field {name} is invalid")
                number_moments[number_targets[name]] = value
            if required_donor:
                del donor_fields, donor_plan
            mark_timing("number_moment_vertical_interpolation")
        if blocked:
            out["mass_source_pd_f32"] = mass_source_pd_f32
        if release_completed_host:
            del mass_vertical_plan, mass_source_pd_f32
            del mass_surface_pd_f32, mass_target_pd_f32

        # WRF's qtot for the pressure recurrence is a sum over the ACTIVE moist
        # package (module_initialize_real.F:3913-3916, PARAM_FIRST_SCALAR ..
        # num_3d_m = num_moist at :1856), so a species the package does not
        # carry contributes nothing there either -- ``hydrometeors`` holds
        # exactly the retained set, in the package's own qc/qr/qi/qs/qg order,
        # and the accumulation below is WRF's own left-to-right one.
        condensate_h = (None if device_route else _host_arrays.sum_fields(
            hydrometeors.values(), workers=column_workers))
        if condensate_h is None:
            for analyzed in hydrometeors.values():
                contribution = _setup(analyzed).astype(np.float64)
                condensate_h = (contribution if condensate_h is None
                                else condensate_h + contribution)
            if hydrometeors:
                del contribution, analyzed

        def moist_total(vapour):
            """WRF ``qtot``: vapour plus every analyzed condensate species."""
            if condensate_h is None:
                return vapour
            total = (None if device_route else _host_arrays.sum_fields(
                (vapour, condensate_h), workers=column_workers))
            return vapour + condensate_h if total is None else total

        host_base = (_slice_base_rows(host_base_whole, start, stop) if blocked else
                     _make_real_base(coord, _host(terrain), float(p_top), cfg.base_temp,
                                     hypsometric_opt=cfg.hypsometric_opt,
                                     column_workers=column_workers))
        base = _ops.upload_base(host_base) if device_route else host_base
        if use_sh_qv:
            # WRF use_sh_qv retains the directly interpolated dry-air mixing
            # ratio while diagnosing the final moist-hydrostatic pressure.
            total_pressure_h = _ops._rebalance_moist_pressure(
                total_pressure_h, moist_total(qv_h), dry_mass, base, coord,
                column_workers=column_workers)
        else:
            # WRF diagnoses qv from interpolated RH, then recomputes a
            # hydrostatic pressure and diagnoses qv once more from that pressure.
            # Two passes make the q/pressure coupling converge below FP32 setup
            # precision.
            for _ in range(2):
                total_pressure_h = _ops._rebalance_moist_pressure(
                    total_pressure_h, moist_total(qv_h), dry_mass, base, coord,
                    column_workers=column_workers)
                temperature_h = _ops._temperature_from_potential_temperature(
                    theta_h, total_pressure_h,
                    column_workers=column_workers)
                qv_h = _ops._cap_stratospheric_qv(
                    _ops._saturation_mixing_ratio(
                        temperature_h, total_pressure_h, rh_h,
                        column_workers=column_workers),
                    total_pressure_h, column_workers=column_workers)
            total_pressure_h = _ops._rebalance_moist_pressure(
                total_pressure_h, moist_total(qv_h), dry_mass, base, coord,
                column_workers=column_workers)
            # WRF invokes rh_to_mxrat1 again against its final hydrostatic
            # pressure; apply the strict pressure-side cap once more in case a
            # target level crossed 10 kPa during the last rebalance.
            qv_h = _ops._cap_stratospheric_qv(
                qv_h, total_pressure_h, column_workers=column_workers)
        mark_timing("base_state_and_moist_rebalance")

        # U/V columns include the 10 m surface pseudo-level with pd averaged to
        # the staggered points exactly like the interior levels
        # (module_initialize_real.F:2785-2811 with vert_interp's 'U'/'V'
        # pressure averaging at :5664-5713; extrap_type=2 constant).
        def stagger_pressure(value, axis):
            result = (None if device_route else _host_arrays.stagger_pressure(
                value, axis, workers=column_workers))
            if result is not None:
                return result
            return (_ops._pressure_at_u(value) if axis == 2
                    else _ops._pressure_at_v(value))

        source_pd_u = stagger_pressure(source_pd, 2)
        source_pd_v = stagger_pressure(source_pd, 1)
        surface_pd_u = stagger_pressure(surface_pd[None], 2)[0]
        surface_pd_v = stagger_pressure(surface_pd[None], 1)[0]
        target_pd_u = stagger_pressure(dry_pressure, 2)
        target_pd_v = stagger_pressure(dry_pressure, 1)
        u_plan = preprocess.prepare_wrf_vertical(
            plan_float32(source_pd_u),
            plan_float32(surface_pd_u),
            plan_float32(target_pd_u))
        v_plan = preprocess.prepare_wrf_vertical(
            plan_float32(source_pd_v),
            plan_float32(surface_pd_v),
            plan_float32(target_pd_v))
        u = u_plan.apply(
            backend_ordered_levels(fields_b["UU"]),
            plan_float32(fields_b["U10"]),
            interp_in_logp=True, extrap="constant")
        v = v_plan.apply(
            backend_ordered_levels(fields_b["VV"]),
            plan_float32(fields_b["V10"]),
            interp_in_logp=True, extrap="constant")
        if release_completed_host:
            del u_plan, v_plan, source_pd_u, source_pd_v
            del surface_pd_u, surface_pd_v, target_pd_u, target_pd_v, source_pd
            del condensate_h, temperature_h, rh_h
            _return_freed_host_memory()

        # -- Configured initial-state perturbation (theta bubbles) ----------
        # Applied ONCE, here, after the base real-data state is final (the
        # WRF two-pass moist rebalance above) and BEFORE the specific volume
        # and geopotential are formed, so alpha/thp/php below are computed
        # FROM the perturbed theta/qv and the state stays discretely
        # hydrostatic at the analyzed pressure -- the em_quarter_ss
        # convention (bubble at constant column mass, geopotential
        # rebalanced), which hydrostatic_residual then grades unchanged.
        # ``initial_perturbation is None`` is the whole OFF contract: not one
        # instruction of this block runs and the state is byte-identical to a
        # build without the feature.  (A perturbed initialization is never
        # row-blocked: its bubbles are placed on the whole domain.)
        perturbation_receipt: dict[str, object] = {}
        if initial_perturbation is not None:
            alpha_unperturbed = _ops._moist_specific_volume(
                theta_h, qv_h, total_pressure_h,
                column_workers=column_workers)
            with pm.worker_limit(column_workers):
                z_half_agl = _half_level_height_agl(
                    base, coord, dry_mass, alpha_unperturbed,
                    hypsometric_opt=cfg.hypsometric_opt)
                perturbation_receipt = initial_perturbation.apply(
                    theta=theta_h, qv=qv_h, pressure=total_pressure_h,
                    z_half_agl=z_half_agl)
            del alpha_unperturbed, z_half_agl
            mark_timing("initial_perturbation")

        alpha = _ops._moist_specific_volume(
            theta_h, qv_h, total_pressure_h,
            column_workers=column_workers)
        mark_timing("wind_interpolation_and_alpha")
        out.update(
            surface_qv=surface_qv, surface_pressure=surface_pressure,
            intq=intq, order=order, dry_mass=dry_mass, dry_pressure=dry_pressure,
            total_pressure_h=total_pressure_h, theta_h=theta_h, qv_h=qv_h,
            alpha=alpha, host_base=host_base, base=base,
            hydrometeors=hydrometeors, number_moments=number_moments,
            u=u, v=v, w_mass_h=w_mass_h, boundary_tracers=boundary_tracers,
            prognostic_qv_floor=prognostic_qv_floor,
            perturbation_receipt=perturbation_receipt)
        return out

    def hydrometeor_receipts(order, mass_source_pd_f32, mass_surface_pd_f32,
                             mass_target_pd_f32, mass_vertical_plan,
                             initialized, zero_surface):
        """The analyzed-hydrometeor correspondence receipt, on whole arrays."""
        nonlocal vertical_disposition_job
        # The plan and surface are bound at definition: the replay runs on
        # the receipt thread after this function has released its own names.
        def replay_hydrometeor_support(ordered_mask, _plan=mass_vertical_plan,
                                       _surface=zero_surface, _xp=backend_xp,
                                       _vboundb=cfg.nz + 1):
            if (_xp is host_numpy and getattr(_plan, "accepts_bool_mask", False)
                    and isinstance(ordered_mask, host_numpy.ndarray)
                    and ordered_mask.dtype == host_numpy.bool_):
                # The bounded CUDA plan widens each batch's bytes on the card
                # (bounded_cuda._VerticalPlan): the host never builds the
                # 380 MB float copy per replay (6.7 s of the receipt thread).
                operand = ordered_mask
            elif (_xp is not host_numpy and isinstance(ordered_mask, host_numpy.ndarray)
                    and ordered_mask.dtype == host_numpy.bool_):
                # Upload the mask as bytes and widen it on the card: the
                # same 0/1 floats, a quarter of the host conversion and
                # transfer per replay (5.6 s of the receipt thread at
                # HRRR CONUS, one replay per source level).
                operand = _xp.asarray(ordered_mask).astype(_xp.float32)
            elif (_xp is host_numpy and isinstance(ordered_mask, host_numpy.ndarray)
                    and ordered_mask.dtype == host_numpy.bool_):
                # The host plan takes floats: widen the mask on every core.
                operand = _receipt_mask_float32(ordered_mask)
            else:
                operand = _xp.asarray(ordered_mask, dtype=_xp.float32)
            return _plan.apply(
                operand,
                _surface,
                interp_in_logp=True, extrap="constant",
                vboundb=_vboundb, values_are_finite=True)

        source_fingerprints = receipt_fingerprints(
            decoded_species, lambda name: plan_float32(fields[name]))
        # Everything decoded and not retained is discarded BY THE PACKAGE,
        # and is named as such with that package's own Registry line.  The
        # partition is derived from one source (retained + discarded is the
        # decoded inventory, always, for every id) rather than listed twice.
        discarded = {
            name: {
                "source": source_fingerprints[name],
                **WRF_REAL_PACKAGE_ABSENT_SPECIES_POLICY,
                "registry_citation": moist_package["registry_citation"],
            }
            for name in decoded_species
            if name not in retained_names
        }
        # The disposition receipt describes the state; it does not decide
        # it.  It runs beside the rest of the initialization and is joined
        # before the result is returned, so its bytes and its failures are
        # unchanged and it no longer sits on the start's critical path.
        vertical_disposition_job = _start_receipt_job(
            build_hrrr_hydrometeor_vertical_disposition,
            {name: fields[name] for name in retained_names},
            order,
            mass_source_pd_f32,
            mass_surface_pd_f32,
            mass_target_pd_f32,
            initialized,
            operator_replay=replay_hydrometeor_support,
        ) if (retained_names and not supplied_mass_surfaces
              and not boundary_only) else None
        vertical_disposition = {}
        # Which regular-grid operator carried each retained species to the
        # mass grid, read off the snapshot that mapped it (the mapped
        # profiles, ERA5 and GFS publish it; the native HRRR decoder and a
        # met_em file do not).  It is a receipt for the four_pt owner: a
        # parabolic entry here is the overshoot the initializer's
        # non-negativity check above would have refused.
        horizontal_operators = getattr(
            snapshot, "horizontal_operators", None) or {}
        receipt = {
            "schema": HRRR_HYDROMETEOR_CORRESPONDENCE_SCHEMA_V2,
            "source": "native-hrrr-horizontal-decoder-output",
            "mp_physics": int(cfg.mp_physics),
            "decoded_source_species": source_fingerprints,
            "retained_correspondence": {
                name: name.lower() for name in retained_names
            },
            "horizontal_operator": {
                name: horizontal_operators.get(name, "unrecorded")
                for name in retained_names
            },
            "discarded_source_species": discarded,
            "vertical_disposition": vertical_disposition,
            "vertical_velocity": vertical_velocity_initialization,
        }
        if supplied_mass_surfaces:
            receipt["schema"] = "gpuwm-metgrid-hydrometeor-initialization-v1"
            receipt.pop("vertical_disposition")
            receipt["surface_pseudo_levels"] = receipt_fingerprints(
                supplied_mass_surfaces, lambda name: plan_float32(fields[name+"_SFC"]))
            receipt["vertical_operator"] = {
                "source": "WRF module_initialize_real.F:1862-1997",
                "operation": "linear Q interpolation with supplied surface before moist-pressure recurrence",
                "source_level_order": order.tolist(),
                **dict(zip(("source_dry_pressure", "surface_dry_pressure",
                            "target_dry_pressure"),
                           receipt_fingerprints_of((mass_source_pd_f32,
                                                    mass_surface_pd_f32,
                                                    mass_target_pd_f32)))),
            }
        return receipt

    def number_moment_receipt():
        """The analyzed-number receipt's source half, on whole arrays."""
        source_numbers, surface_numbers = {}, {}
        for name in decoded_numbers:
            number_fields = (aerosol_snapshot.fields if name in donor_numbers
                             else fields)
            source_numbers[name], surface_numbers[name] = receipt_fingerprints_of(
                (plan_float32(number_fields[name]), plan_float32(number_fields[name+"_SFC"])))
        return {
            "schema": "gpuwm-metgrid-number-initialization-v1",
            "source": source_numbers,
            "surface_pseudo_level": surface_numbers,
            "separate_aerosol_donor": bool(donor_numbers),
            "aerosol_donor_valid_time": (aerosol_snapshot.valid_time.isoformat()
                                         if donor_numbers else None),
            "retained_correspondence": {name: number_targets[name] for name in decoded_numbers if name in number_targets},
            "discarded_inactive_package_fields": [name for name in decoded_numbers if name not in number_targets],
            "operator": "WRF module_initialize_real.F:1999-2120; linear Q vertical interpolation with supplied surface pseudo-level",
            "boundary_policy": "initial analysis; default flow-dependent scalar boundaries are unchanged",
        }

    vertical_disposition_job = None
    vertical_velocity_initialization = dict(WRF_REAL_VERTICAL_VELOCITY_POLICY)
    hydrometeor_initialization: dict[str, object] = {}
    number_receipt = {}
    state_kwargs = {}
    if scratch_arena is not None:
        state_kwargs["scratch_arena"] = scratch_arena
    if dycore_state_workspace is not None:
        state_kwargs["dycore_state_workspace"] = dycore_state_workspace
    if not blocked:
        columns = initialize_columns(0, cfg.ny)
        hydrometeor_initialization = columns.pop("hydrometeor_initialization", {})
        if decoded_numbers:
            number_receipt = number_moment_receipt()
        (surface_qv, surface_pressure, intq, order, dry_mass, dry_pressure,
         total_pressure_h, theta_h, qv_h, alpha, host_base, base, hydrometeors,
         number_moments, u, v, prognostic_qv_floor, perturbation_receipt) = (
            columns[name] for name in (
                "surface_qv", "surface_pressure", "intq", "order", "dry_mass",
                "dry_pressure", "total_pressure_h", "theta_h", "qv_h", "alpha",
                "host_base", "base", "hydrometeors", "number_moments", "u", "v",
                "prognostic_qv_floor", "perturbation_receipt"))
        w_mass_h = columns.pop("w_mass_h")
        boundary_tracers = columns.pop("boundary_tracers")
        del columns

        state = DomainState(cfg, array_module=state_xp, **state_kwargs)
        mark_timing("domain_state_allocation")
        if device_route:
            _ops.load_base(state, coord, host_base, base)
        else:
            with pm.worker_limit(column_workers):
                state.load_base(coord, host_base, native_host=True)
        mark_timing("base_state_upload")
        if not (state_xp is host_numpy and _host_arrays.difference_float32(
                state.mup, dry_mass, base.mub, workers=column_workers)):
            state.mup[...] = (plan_float32(dry_mass - base.mub) if device_route else
                              state_xp.asarray(dry_mass - base.mub, dtype=state_xp.float32))
        if not (state_xp is host_numpy and _host_arrays.difference_float32(
                state.thp, theta_h, base.thb, workers=column_workers)):
            state.thp[...] = (plan_float32(theta_h - base.thb) if device_route else
                              state_xp.asarray(theta_h - base.thb, dtype=state_xp.float32))
        mark_timing("mass_theta_upload")
        state.php[...] = state_xp.asarray(
            _ops._fp32_geopotential_split(base, coord, dry_mass, alpha,
                                     hypsometric_opt=cfg.hypsometric_opt,
                                     column_workers=column_workers),
            dtype=state_xp.float32)
        mark_timing("fp32_geopotential_split_and_upload")
        if not (state_xp is host_numpy and _host_arrays.copy_float32(
                state.qv, qv_h, workers=column_workers)):
            state.qv[...] = (plan_float32(qv_h) if device_route else
                             state_xp.asarray(qv_h, dtype=state_xp.float32))
        if hydrometeors:
            for source_name, value in hydrometeors.items():
                if state_xp is host_numpy:
                    value = _host_float32(value)
                getattr(state, source_name.lower())[...] = state_xp.asarray(
                    value, dtype=state_xp.float32)
    else:
        # One block of rows at a time: its fields are finished into the
        # whole state and the whole result arrays, then released.
        state = DomainState(cfg, array_module=state_xp, **state_kwargs)
        mark_timing("domain_state_allocation")
        host_base = base = host_base_whole
        keep_theta = bool(retained_names) and cfg.mp_physics in (8, 28)
        keep_source_pd = bool(decoded_species) and not boundary_only and bool(
            retained_names or supplied_mass_surfaces)
        whole: dict[str, np.ndarray] = {}
        number_moments = {}
        boundary_tracers = {}
        floor_parts = []
        order = level_order
        def block_float32_difference(first, second):
            """``float32(first - second)`` as the whole build forms it."""
            out = np.empty(first.shape, dtype=np.float32)
            second = np.ascontiguousarray(second)
            if not _host_arrays.difference_float32(out, first, second,
                                                   workers=column_workers):
                out[...] = np.asarray(first - second, dtype=np.float32)
            return out

        pool_trim = _POOL_TRIM.set(_BLOCK_POOL_TRIM)
        try:
            for owned_start, owned_stop in block_chunks:
                start = max(0, owned_start - 1)
                columns = initialize_columns(start, owned_stop, owned_start)
                owned = slice(owned_start - start, owned_stop - start)
                last = owned_stop == cfg.ny

                def keep(name, value, dtype=None):
                    target = whole.get(name)
                    if target is None:
                        shape = value.shape[:-2] + (cfg.ny, value.shape[-1])
                        target = whole[name] = np.empty(shape, dtype=dtype or value.dtype)
                    target[..., owned_start:owned_stop, :] = value[..., owned, :]

                for name in ("surface_qv", "surface_pressure", "intq", "dry_mass",
                             "dry_pressure", "total_pressure_h", "alpha"):
                    keep(name, columns[name])
                if columns["w_mass_h"] is not None:
                    keep("w_mass_h", columns["w_mass_h"])
                for name, value in columns["boundary_tracers"].items():
                    part = boundary_tracers.get(name)
                    if part is None:
                        part = boundary_tracers[name] = np.empty(
                            value.shape[:-2] + (cfg.ny, cfg.nx), dtype=value.dtype)
                    part[..., owned_start:owned_stop, :] = value[..., owned, :]
                if keep_theta:
                    keep("theta_h", columns["theta_h"])
                if keep_source_pd:
                    keep("mass_source_pd_f32", columns["mass_source_pd_f32"])
                for name, value in columns["number_moments"].items():
                    part = number_moments.get(name)
                    if part is None:
                        part = number_moments[name] = np.empty(
                            value.shape[:-2] + (cfg.ny, cfg.nx), dtype=value.dtype)
                    part[..., owned_start:owned_stop, :] = value[..., owned, :]
                floor_parts.append(columns.get("prognostic_qv_floor_stats"))
                # The whole-domain build's own float32 kernels, on the
                # block's contiguous arrays; only the owned rows are kept.
                block_base = _slice_base_rows(host_base_whole, start, owned_stop)
                state.mup[owned_start:owned_stop] = block_float32_difference(
                    columns["dry_mass"], block_base.mub)[owned]
                state.thp[:, owned_start:owned_stop] = block_float32_difference(
                    columns["theta_h"], block_base.thb)[:, owned]
                mark_timing("mass_theta_upload")
                # On the block's contiguous arrays (halo row included), as
                # the whole-domain build does; only the owned rows are kept.
                state.php[:, owned_start:owned_stop] = np.asarray(
                    _ops._fp32_geopotential_split(
                        _slice_base_rows(host_base_whole, start, owned_stop), coord,
                        columns["dry_mass"], columns["alpha"],
                        hypsometric_opt=cfg.hypsometric_opt,
                        column_workers=column_workers)[:, owned],
                    dtype=np.float32)
                mark_timing("fp32_geopotential_split_and_upload")
                qv_b = np.empty(columns["qv_h"].shape, dtype=np.float32)
                if not _host_arrays.copy_float32(qv_b, columns["qv_h"], workers=column_workers):
                    qv_b[...] = np.asarray(columns["qv_h"], dtype=np.float32)
                state.qv[:, owned_start:owned_stop] = qv_b[:, owned]
                del qv_b
                for source_name, value in columns["hydrometeors"].items():
                    getattr(state, source_name.lower())[:, owned_start:owned_stop] = (
                        _host_float32(value[:, owned]))
                state.u[:, owned_start:owned_stop] = _host_float32(columns["u"][:, owned])
                v_rows = slice(owned.start, owned.stop + (1 if last else 0))
                state.v[:, owned_start:owned_stop + (1 if last else 0)] = (
                    _host_float32(columns["v"][:, v_rows]))
                del columns
                _return_freed_host_memory()
        finally:
            _POOL_TRIM.reset(pool_trim)
        (surface_qv, surface_pressure, intq, dry_mass, dry_pressure,
         total_pressure_h, alpha) = (whole.pop(name) for name in (
             "surface_qv", "surface_pressure", "intq", "dry_mass",
             "dry_pressure", "total_pressure_h", "alpha"))
        w_mass_h = whole.pop("w_mass_h", None)
        theta_h = whole.pop("theta_h", None)
        perturbation_receipt = {}
        prognostic_qv_floor = (_sh_vertical_undershoot_receipt(floor_parts)
                               if use_sh_qv else {})
        hydrometeors = {name: getattr(state, name.lower()) for name in retained_names}
        if decoded_species:
            # The receipts the column blocks cannot give, on the assembled
            # whole arrays: the replay reads the whole domain's operator.
            source_pd_whole = whole.pop("mass_source_pd_f32", None)
            target_pd_whole = plan_float32(dry_pressure)
            surface_pd_whole = plan_float32(fields["PSFC"] - intq)
            receipt_plan = (preprocess.prepare_wrf_vertical(
                source_pd_whole, surface_pd_whole, target_pd_whole)
                if source_pd_whole is not None else None)
            hydrometeor_initialization = hydrometeor_receipts(
                order, source_pd_whole, surface_pd_whole, target_pd_whole,
                receipt_plan, hydrometeors,
                backend_xp.zeros((cfg.ny, cfg.nx), dtype=backend_xp.float32))
            del receipt_plan, source_pd_whole, target_pd_whole, surface_pd_whole
            _return_freed_host_memory()
        if decoded_numbers:
            number_receipt = number_moment_receipt()
        del whole
        # The base state's float32 copies and its float64 geopotential cache
        # (about 13 GiB at 4160 x 2450 x 50) are installed after the
        # receipts above, which read none of them, so they are not resident
        # under the receipts' whole-domain replay.
        with pm.worker_limit(column_workers):
            state.load_base(coord, host_base, native_host=True)
        mark_timing("base_state_upload")
    if hydrometeors:
        hydrometeor_initialization["initialized_state_species"] = receipt_fingerprints(
            hydrometeor_initialization["retained_correspondence"].values(),
            lambda name: getattr(state, name))
        if cfg.mp_physics == 8:
            # HRRR provides the shared five WRF mass species but not classic
            # Thompson's Registry scalar QNICE/QNRAIN fields.  Both number
            # moments start from real.exe's source-absent zero; the
            # cold-start closure below (after any analysed numbers are
            # installed) then gives rain and ice mass real.exe's
            # make_RainNumber / make_IceNumber where the number is still
            # at or below zero, as module_initialize_real.F:4840-4852 does.
            state.ni[...] = state_xp.float32(0.0)
            state.nr[...] = state_xp.float32(0.0)
    else:
        state.qc[...] = 0.0
        state.qr[...] = 0.0
    if analyzed_species is not None:
        # WRF only interpolates species whose metgrid FLAG_* is set. Missing
        # species retain the allocated zero state, never a fabricated analysis.
        if not hydrometeor_initialization:
            hydrometeor_initialization = {
                "schema": "gpuwm-declared-initial-species-v1",
                "mp_physics": int(cfg.mp_physics),
            }
        hydrometeor_initialization["source"] = ("supplied-metgrid-mass-and-surface-analysis"
            if supplied_mass_surfaces else "declared-analyzed-field-inventory")
        hydrometeor_initialization["declared_analyzed_species"] = list(decoded_species)
        hydrometeor_initialization["source_absent_state_fields"] = receipt_fingerprints(
            (name.lower() for name in DECLARED_ANALYZED_HYDROMETEORS
             if name not in decoded_species and getattr(state, name.lower(), None) is not None),
            lambda name: getattr(state, name))
        hydrometeor_initialization["source_absent_policy"] = (
            "WRF module_initialize_real.F:1862-1997 only interpolates fields with FLAG_*=1; others retain allocated zero")
    aerosol_initialization: dict[str, object] = {}
    # The mp=28 branch resolves whether aerosols were READ; None keeps the
    # configuration's own default for every other scheme.
    boundary_aerosol_from_input = None
    if cfg.mp_physics in THOMPSON_AEROSOL_FAMILY:
        # Aerosol-aware Thompson.  Its Registry package
        # (Registry.EM_COMMON:3036) carries the same six moist mass members
        # as mp=8 -- so every mass decision above is already correct for it
        # -- plus SIX scalar members: qni, qnr, qnc, qnwfa, qnifa, qnbca.
        # No real-data source this module reads supplies any of the six, so
        # this block is the mp=28 statement of real.exe's ``i0``
        # source-absent policy.  It runs on BOTH lanes (native HRRR and
        # pressure-level/RH) because the policy is about the scheme's
        # scalars, not about analyzed condensate.  It splits into two
        # halves that must not be confused.
        #
        # (a) THE NUMBER MOMENTS -- zero, then SEEDED and CLOSED over the
        # analyzed mass.  mp=28 promotes cloud droplet number from the
        # constant Nt_c to a prognostic scalar, so nc joins ni and nr as a
        # transported moment the analysis does not carry.  real.exe seeds
        # all three where the mass is present and the number is not
        # (make_DropletNumber, make_RainNumber, make_IceNumber,
        # module_initialize_real.F:4829-4852, WRF v4.7.1), and Thompson's
        # entry block (module_mp_thompson.F:1827-1899) rediagnoses them on
        # the first call.  Before the seed the state carried mass with no
        # number in every cloudy cell, and anything that read it there -- the
        # between-step reflectivity operator, an analysis at the start time,
        # a picture of the initial frame -- read a rain number at the
        # scheme's R2 floor and diagnosed a reflectivity burst.  So the same
        # entry block runs ONCE here, through the repair authority every DA
        # door already carries (woof.core.thompson_entry, the block
        # woof.da.moments.repair_moments applies), on the density this
        # initializer just formed; cells without mass keep the exact zero
        # real.exe writes.  The closure is the scheme's own answer, not a
        # clip: nothing here bounds a number the scheme would not, and the
        # health limits are untouched.  Where WRF leaves the initial state
        # undefined between real and the first step, the tree defines it
        # and the receipt says so.  qnbca is not a woof state field. WRF's
        # mp=28 Registry package writes it, but Thompson consumes it only
        # at wif_input_opt=2; that black-carbon operation remains
        # unimplemented here.
        state.ni[...] = state_xp.float32(0.0)
        state.nr[...] = state_xp.float32(0.0)
        state.nc[...] = state_xp.float32(0.0)
        # The closure itself runs after the supplied moments are
        # installed, once the aerosol source is decided: real.exe sets the
        # numbers (module_initialize_real.F:4829-4852) after its aerosol
        # initialization (:2327-2740) and its analysed-number input, from
        # the aerosol that left, and only where a number is still <= 0.
        # (b) THE AEROSOLS -- deliberately NOT written here, and the
        # deliberateness is the whole point.  WRF's own initializer, with
        # aer_init_opt=0, sets QNWFA and QNIFA to 0.0 and nothing else
        # (dyn_em/module_initialize_real.F:2332-2345).  ``thompson_init``
        # then tests MAXVAL(nwfa) < eps (phys/module_mp_thompson.F:493) and
        # MAXVAL(nifa) < eps (:531) and, finding them empty, installs the
        # synthetic boundary-layer-following CCN profile at :498-515 and
        # the IN profile at :536-551, deriving the surface emission nwfa2d
        # at :509-510.  Exact zero is therefore not "unset": it is the
        # SIGNAL that selects that branch.  Assigning any nonzero
        # placeholder here -- a floor, a background, a climatology guess --
        # would flip WRF's own has_CCN/has_IN test and permanently suppress
        # the profile fill, which is the failure mode that costs 5.6x in
        # droplet number and +74% in domain-total RAINNC while raising no
        # error anywhere.
        #
        # WHICH CODE ACTUALLY FILLS IT: woof.core.physics.
        # initialize_physics, via woof.core.microphysics.
        # microphysics_init -- WRF's own split, where thompson_init is
        # called from phys/module_physics_init.F and not from
        # module_initialize_real.F.  Both production real-data front doors
        # reach it (woof/ingest/hrrr_physics.py::initialize_prepared_physics
        # and ::initialize_hrrr_physics both call initialize_physics on the
        # state this function returns), so a fill here would be the FIRST of
        # two calls, never the only one.
        #
        # MEASURED, so the reason given is the real one: a duplicate call is
        # IDEMPOTENT, not destructive.  thompson_aerosol_init_fill re-runs
        # WRF's own MAXVAL presence tests every time, so a second
        # initialize_physics on the same state reports
        # {'ccn': False, 'in': False} and changes not one bit
        # (tests/test_real_init.py::
        # test_mp28_real_ingest_through_initialize_physics_fills_exactly_once
        # asserts precisely that, on device).  The reason this module still
        # must not do the fill is structural, not a race: WRF puts
        # thompson_init in module_physics_init.F and not in
        # module_initialize_real.F, the domain construction the fill belongs
        # to has not happened yet here, and filling would make the
        # "awaiting_profile_fill" receipt this function publishes false.
        # THE DEFAULT (development branch).  This used to read
        # ``(aer_init_opt, wif_input_opt) == (1, 1)`` -- an opt-in.  An
        # opt-in flag on a correctness remedy is a workaround, so it is
        # gone: a real-data mp=28 run now RESOLVES WRF's monthly WIF
        # climatology and uses it, and reaches thompson_init's synthetic
        # profile only when there is genuinely no dataset to read.  The two
        # namelist selectors keep WRF's Registry defaults (the
        # prepared-forecast runner compares those rows for exact equality);
        # the decision lives in cfg.mp28_aerosol_source, whose default is
        # "auto".
        #
        # ``climatology`` refuses instead of falling back, because a
        # request that silently degrades is worse than one that fails.
        # ``synthetic`` skips the resolver entirely -- it does not "fail to
        # find" anything, so it is not a fallback and must not be reported
        # as one.
        from woof.ingest.wif_climatology import (
            describe_wif_source)
        # aer_init_opt=1/wif_input_opt=1 is real.exe's own spelling of
        # "use the climatology"; the shared resolver honours it as the
        # strict form, and the run-door floor above reads the same one.
        _source_choice = _resolved_mp28_aerosol_source(cfg)
        # AND WHICH SPELLING ASKED FOR IT, read once here so that the
        # discard warning below, the missing-dataset refusal and the
        # underivable-grid refusal all name the selector the operator
        # actually set.  `mp28_aerosol_source` is the field spelling;
        # (aer_init_opt, wif_input_opt) = (1, 1) is the namelist spelling
        # that resolves to the same strict request while the field stays
        # at its default, and a message naming the field to that operator
        # offers a change that is already made.
        _source_named_by_namelist = wif_climatology_named_by_namelist(cfg)
        _source_request = (
            WIF_CLIMATOLOGY_NAMELIST_PHRASE if _source_named_by_namelist
            else f"mp28_aerosol_source={_source_choice!r}")
        # THE SNAPSHOT'S OWN AEROSOL.  Whether the analysis carried an
        # QNWFA/QNIFA pair is a property of the FILES, not of the
        # configuration, so it is read here from ``decoded_numbers`` and
        # needs no new ``mp28_aerosol_source`` value, no RunConfig field
        # and no new row in the prepared-cache identity table.  "auto"
        # keeps meaning "ArWen chooses", and what it chooses when the
        # analysis carries the fields is the analysis: a climatology is a
        # monthly mean standing in for exactly this.
        #
        # PRECEDENCE, and why it is resolved HERE rather than at the
        # install below.  The install runs after this whole block, so a
        # climatology write would land first and the analysis would
        # silently overwrite it while the receipt still said
        # "wif-climatology".  Choosing the source once, before the
        # resolver runs, is what keeps the receipt and the bytes the same
        # decision.
        _analyzed_aerosol = analyzed_aerosol_fields(
            decoded_numbers, number_targets)
        _analyzed_aerosol_missing = tuple(
            name for name in AEROSOL_NUMBER_FIELDS
            if name not in _analyzed_aerosol)
        # THE PAIR IS THE SOURCE, NOT EITHER HALF OF IT.  With FLAG_QNWFA
        # and FLAG_QNIFA independent metgrid flags, an analysis can carry
        # one and not the other; taking that as "the analysis supplies
        # the aerosol" left the absent half at exact zero while the
        # receipt said no synthetic profile was used and the lateral
        # boundaries still declared it carried from input -- an
        # aerosol-free inflow with nothing saying so, which is the one
        # breakage the mp=28 precondition exists to name.  A half pair is
        # therefore not this source: it is warned about by name and the
        # resolved climatology/synthetic source initializes both fields,
        # which is the one basis that covers them both.
        _metgrid_analyzed_aerosol = (
            not _analyzed_aerosol_missing and _source_choice in ("auto", "analysis"))
        if _analyzed_aerosol and not _metgrid_analyzed_aerosol:
            # RULE: warn, never refuse.  The run proceeds on a resolved
            # source; what it must not do is proceed and stay quiet about
            # the analyzed fields that went in the bin.
            _discarded = "/".join(_analyzed_aerosol)
            if _source_choice != "auto":
                # WHO ASKED, not merely what was resolved.  The pair
                # (aer_init_opt, wif_input_opt) = (1, 1) resolves to
                # 'climatology' with mp28_aerosol_source left at its
                # default, so attributing the request to that field named
                # a setting the operator never touched and offered a way
                # out ("leave it at 'auto'") that was already taken.  One
                # question, one answer, the same one the floor above and
                # the resolver read.
                _by_namelist = _source_named_by_namelist
                if _by_namelist:
                    _why = (
                        "and " + WIF_CLIMATOLOGY_NAMELIST_PHRASE + " asks "
                        "for the WIF climatology (mp28_aerosol_source is at "
                        "its default 'auto'), so the analyzed field(s) are "
                        "discarded and the climatology is used. Set either "
                        "selector to 0 to initialize the aerosol from the "
                        "analysis.")
                else:
                    _why = (
                        f"and mp28_aerosol_source={_source_choice!r} was "
                        "requested explicitly, so the analyzed field(s) are "
                        "discarded and the requested source is used. Leave "
                        "mp28_aerosol_source at 'auto' to initialize the "
                        "aerosol from the analysis.")
                _discard_key = "discarded_by_requested_aerosol_source"
                _discard_detail = {
                    "fields": list(_analyzed_aerosol),
                    "mp28_aerosol_source": _source_choice,
                    "requested_by": ("namelist" if _by_namelist
                                     else "mp28_aerosol_source"),
                }
                if _by_namelist:
                    _discard_detail.update(
                        {name: int(getattr(cfg, name, 0) or 0)
                         for name, _ in WIF_CLIMATOLOGY_NAMELIST_PAIR})
            else:
                _why = (
                    "but not " + "/".join(_analyzed_aerosol_missing) +
                    ", and an aerosol-aware run needs the whole pair, so the "
                    "analyzed field(s) are discarded and both fields are "
                    "initialized from the resolved aerosol source. Regenerate "
                    "met_em with " + " and ".join(
                        f"FLAG_{name}=1" for name in AEROSOL_NUMBER_FIELDS) +
                    " to initialize the pair from the analysis.")
                _discard_key = "discarded_by_incomplete_analyzed_aerosol_pair"
                _discard_detail = {
                    "fields": list(_analyzed_aerosol),
                    "missing": list(_analyzed_aerosol_missing),
                }
            print(
                f"mp_physics=28 aerosol: this analysis carries {_discarded}, "
                f"{_why}", file=sys.stderr)
            for name in _analyzed_aerosol:
                number_moments.pop(number_targets[name], None)
            number_receipt["retained_correspondence"] = {
                name: target for name, target
                in number_receipt["retained_correspondence"].items()
                if name not in _analyzed_aerosol}
            number_receipt[_discard_key] = _discard_detail
        if _metgrid_analyzed_aerosol or _source_choice == "synthetic":
            wif_resolution = None
        else:
            # 'auto' and 'climatology' both COMMIT to the dataset: found,
            # else downloaded under its pin, else refused by name.  'auto'
            # used to fall through to the synthetic profile with a warning
            # (a real-data mp=28 run, 2026-10-23 12:58Z).
            from woof.ingest.wif_climatology import acquire_wif_climatology
            wif_resolution = acquire_wif_climatology(
                cfg.wif_climatology_path or None,
                requested_by=_source_request)
        wif_climatology_selected = (
            wif_resolution is not None and wif_resolution.resolved)
        wif_receipt = None
        if wif_climatology_selected:
            # (b') THE PORTED CLIMATOLOGY -- real.exe's aer_init_opt=1
            # path (dyn_em/module_initialize_real.F:2357-2536 3-D,
            # :4530-4547 2-D), fed from the SAME global monthly dataset
            # WRF routes through metgrid's constants_name.  This is the
            # one aerosol source that fills the fields HERE rather than
            # leaving thompson_init's synthetic profile to do it: the
            # nonzero nwfa/nifa this writes is exactly the signal WRF's
            # own MAXVAL presence tests (module_mp_thompson.F:493/:531)
            # read to SKIP the synthetic fill, so microphysics_init needs
            # no new teaching -- WRF's mechanism already carries it.  The
            # vertical target is ``dry_pressure``: the ``grid%pb`` the
            # WIF vert_interp call receives is, at that point in
            # init_domain_rk, the p_dry(mu0, znw, p_top) scratch
            # (:1625-1629, :1701), not the final base state.
            from woof.ingest.wif_climatology import (
                load_wif_climatology, wif_fields_for_grid)
            # BOTH INPUTS ARE DERIVED, and neither is guessed.  Making a
            # caller hand these in was the whole reason the capability had
            # no front door: every runner already holds both, so asking
            # for them again meant a configuration that selects the
            # ingest still could not run it.
            #
            # The valid date is ``snapshot.valid_time`` -- the case's own
            # valid time, the same one every runner would have passed and
            # the one metgrid stamps on this forcing frame.  There is no
            # second candidate; a date that disagreed with the snapshot
            # would be interpolating the climatology to a month this
            # state is not.
            #
            # The grid latitudes/longitudes are the model's mass-point
            # geodesy, taken from whatever the caller already carries: a
            # projection Grid (``latlon_mass()``), the state's own
            # ``latitude_deg``/``longitude_deg`` mirror, a geogrid mapping
            # (``XLAT_M``/``XLONG_M``), or a plain ``(lat2d, lon2d)``
            # pair.  This is METADATA, not source data -- which is the
            # arbitrary-acceptance argument the module docstring already
            # makes: the derivation is identical for HRRR, GFS, ERA5,
            # 20CRv3, RUC and any mapped source, because none of them is
            # consulted.
            if wif_valid_date is None:
                wif_valid_date = getattr(snapshot, "valid_time", None)
            if wif_grid_latlon is None:
                wif_grid_latlon = _wif_grid_latlon_from(grid, state)
            if wif_grid_latlon is None or wif_valid_date is None:
                # The dataset resolved but this CALLER still cannot use
                # it.  The derivation above (development branch)
                # takes the valid date from snapshot.valid_time and the
                # mass-point lat/lon from whichever carrier the caller
                # holds, so reaching here means the caller supplied NONE
                # of them -- not that a front door forgot to pass grid=.
                #
                # WHICH BEHAVIOUR is decided by who asked, not by what is
                # missing (development branch).  An EXPLICIT climatology
                # request refuses: honouring it is impossible, and
                # silently substituting a different initial condition for
                # a named one is the failure that lane exists to remove.
                # The DEFAULT falls back, because a call site that carries
                # no geodesy at all is a property of the call site, not a
                # configuration error the user made, and a default that
                # hard-fails there is a default nobody can ship.  Either
                # way it is NAMED, and it names WHICH input could not be
                # derived rather than only that one could not.
                missing = []
                if wif_grid_latlon is None:
                    missing.append(
                        "the model mass-point latitudes/longitudes (pass "
                        "grid=<projection Grid>, grid={'XLAT_M':..., "
                        "'XLONG_M':...} or wif_grid_latlon=(lat2d, lon2d))")
                if wif_valid_date is None:
                    missing.append(
                        "the case valid time (the snapshot carries it as "
                        ".valid_time; pass wif_valid_date='YYYY-MM-DD...' "
                        "when it does not)")
                unusable = (
                    "the WIF aerosol climatology resolved at "
                    f"{wif_resolution.path} but this initialization could "
                    "not derive " + " and ".join(missing)
                    + ", which a GLOBAL monthly dataset needs and which "
                    "cannot be guessed: a wrong grid or a wrong month is a "
                    "silently different aerosol field, not an error")
                if _source_choice == "climatology":
                    raise ValueError(
                        _source_request + " cannot be honoured: " + unusable)
                from woof.ingest.wif_climatology import WifSourceResolution
                wif_resolution = WifSourceResolution(
                    None, "resolved-but-caller-supplied-no-grid",
                    wif_resolution.candidates, unusable)
                wif_climatology_selected = False
        if wif_climatology_selected:
            wif_lat2d, wif_lon2d = wif_grid_latlon
            # ONE RESOLVER, not two.  development branch reached the
            # dataset a second time here through
            # wif_dataset.resolve_wif_climatology_path.  It cannot stay:
            # resolve_wif_climatology (development branch) has ALREADY chosen
            # the file above -- that choice is what made
            # wif_climatology_selected true -- and a second resolver over
            # the same config field with a different precedence is how a
            # run reads one dataset and reports another.  The staged root
            # that call existed to reach (`woof fetch-tables --wif` ->
            # ~/.woof/wif) is now a rung INSIDE resolve_wif_climatology,
            # so nothing was lost by deleting it.
            wif_fields, wif_receipt = wif_fields_for_grid(
                load_wif_climatology(wif_resolution.path),
                _setup(wif_lat2d), _setup(wif_lon2d), str(wif_valid_date),
                _host_float32(dry_pressure), _host_float32(base.phb))
            for name in ("nwfa", "nifa", "nwfa2d", "nifa2d"):
                target_field = getattr(state, name)
                target_field[...] = state_xp.asarray(
                    wif_fields[name], dtype=state_xp.float32)
        else:
            for name in ("nwfa", "nifa", "nwfa2d", "nifa2d"):
                value = getattr(state, name)
                if bool((value != 0).any()):
                    raise ValueError(
                        f"mp_physics=28 real initialization left "
                        f"state.{name} "
                        "nonzero; exact zero is WRF's aer_init_opt=0 value "
                        "(dyn_em/module_initialize_real.F:2332-2345) AND "
                        "the signal thompson_init's MAXVAL test reads "
                        "(phys/module_mp_thompson.F:493/:531) to install "
                        "the synthetic profile. The two ingests that MAY "
                        "populate them are the wif-climatology branch "
                        "above and the analyzed QNWFA/QNIFA install "
                        "below, and this run selected neither.")
        surface_emission_receipt = None
        if _metgrid_analyzed_aerosol and cfg.use_rap_aero_icbc and not boundary_only:
            # Operational WRF keeps monthly surface emissions while taking
            # its three-dimensional initial and boundary numbers from RAP.
            from woof.ingest.wif_climatology import (
                load_wif_climatology, wif_lowest_number_for_grid)
            from woof.ingest.cpu_backend import CpuPreprocessBackend
            from woof.ingest.wif_climatology import acquire_wif_climatology
            emission_resolution = acquire_wif_climatology(
                cfg.wif_climatology_path or None,
                requested_by="use_rap_aero_icbc surface emissions")
            emission_latlon = wif_grid_latlon or _wif_grid_latlon_from(grid, state)
            if emission_latlon is None:
                raise ValueError("monthly aerosol surface emissions require the target mass-point latitude and longitude")
            monthly_number = wif_lowest_number_for_grid(
                load_wif_climatology(emission_resolution.path),
                _host_float32(emission_latlon[0]), _host_float32(emission_latlon[1]),
                str(wif_valid_date or snapshot.valid_time))
            surface_emission = CpuPreprocessBackend(cpu_bridge).aerosol_surface_mass(
                monthly_number, _host_float32(base.phb), _host_float32(alpha[0]),
                cfg.dx, cfg.dy)
            state.nwfa2d[...] = state_xp.asarray(surface_emission, dtype=state_xp.float32)
            surface_emission_receipt = {
                "source": "monthly-climatology", "dataset": describe_wif_source(emission_resolution),
                "operator": "qnwfa2d=monthly_number*0.000196*(airmass*2e-10); airmass=(1/alt)*z1*dx*dy",
                "authority": "NOAA-EMC/HRRR WRFV3.9 module_initialize_real.F:4424-4430",
            }
        elif _metgrid_analyzed_aerosol and "QNWFA" in _analyzed_aerosol and not cfg.use_rap_aero_icbc:
            # WRF's own surface-emission formula
            # (dyn_em/module_initialize_real.F:4530-4547), reached through
            # the SAME wif_surface_emission the climatology branch uses,
            # fed from the lowest ANALYZED level rather than the lowest
            # climatology level.  One formula, two sources: a second
            # transcription here is how the two would drift.  ``nifa2d``
            # stays at zero exactly as WRF leaves it
            # (woof/core/state.py:682-689): WRF derives no ice-friendly
            # surface emission from an aerosol field at all.
            from woof.ingest.wif_climatology import wif_surface_emission
            _analyzed_surface_emission, _ = wif_surface_emission(
                _host_float32(number_moments[number_targets["QNWFA"]][0]),
                _host_float32(base.phb))
            state.nwfa2d[...] = state_xp.asarray(
                _analyzed_surface_emission, dtype=state_xp.float32)
        # Record the final resolution, including a resolved dataset which
        # could not be used without grid metadata. Zero supplied aerosol
        # remains input. Immutable LBC field names/bytes carry the decision
        # through preparation/cache/streaming/restart; tiles need no flag.
        from woof.boundary_fields import external_scalar_fields
        boundary_aerosol_from_input = bool(
            wif_climatology_selected or _metgrid_analyzed_aerosol)
        state._external_scalar_boundary_fields = external_scalar_fields(
            cfg, aerosol_from_input=boundary_aerosol_from_input)
        aerosol_initialization = _mp28_aerosol_source_policy(cfg, state, receipt=not boundary_only)
        aerosol_initialization["mp28_aerosol_source"] = _source_choice
        if _metgrid_analyzed_aerosol:
            # THE ANALYSIS ITSELF.  nwfa/nifa are written by the analyzed
            # number install below, from the same vertical operator every
            # other metgrid number field uses, so thompson_init's MAXVAL
            # presence tests find them filled and skip the synthetic
            # profile.  Nothing is awaited and no dataset was searched
            # for, so neither the climatology's ``dataset`` key nor the
            # fallback's ``synthetic_fallback_in_use`` belongs here.
            aerosol_initialization["awaiting_profile_fill"] = False
            aerosol_initialization["aerosol_source"] = "metgrid-analyzed"
            aerosol_initialization["aerosol_source_statement"] = (
                "mp_physics=28 aerosol initialized from the analyzed "
                "aerosol number fields carried by this analysis "
                f"({', '.join(_analyzed_aerosol)}), interpolated onto the "
                "eta grid by the shared metgrid number operator "
                "(WRF module_initialize_real.F:1999-2120). The surface "
                "emission nwfa2d is WRF's own :4530-4547 formula applied "
                "to the lowest analyzed level; nifa2d stays zero, as it "
                "does in WRF. No aerosol climatology was searched for "
                "and thompson_init's synthetic profile is not used.")
            aerosol_initialization["analyzed_aerosol_fields"] = list(
                _analyzed_aerosol)
            aerosol_initialization["policy"] = (
                "metgrid-analyzed-number-vertical-interp")
            if cfg.use_rap_aero_icbc:
                aerosol_initialization["aerosol_source_statement"] = (
                    "Three-dimensional QNWFA/QNIFA initial and boundary values come from the analyzed driving fields. "
                    "The two-dimensional surface source uses operational WRF's monthly climatology and cell air mass formula.")
                aerosol_initialization["surface_emission"] = surface_emission_receipt
        elif wif_receipt is not None:
            # The fields are ALREADY populated: thompson_init's MAXVAL
            # tests will find them nonzero and skip the synthetic fill,
            # so nothing is awaited.  The stage receipt binds which
            # dataset, which weights and which operators produced them.
            from woof.config import MP28_AEROSOL_SOURCE_DEFAULT
            aerosol_initialization["awaiting_profile_fill"] = False
            aerosol_initialization["wif_climatology"] = wif_receipt
            aerosol_initialization["aerosol_source"] = "wif-climatology"
            aerosol_initialization["aerosol_source_statement"] = (
                MP28_AEROSOL_SOURCE_DEFAULT)
            aerosol_initialization["dataset"] = describe_wif_source(
                wif_resolution)
            aerosol_initialization["policy"] = (
                "wif-climatology-monthly-interp-then-vert-interp")
        else:
            # THE FALLBACK, SAID OUT LOUD.  This is the branch that changes
            # what the forecast IS, so it is the branch that must be
            # impossible to miss in a receipt: a named source, the WRF
            # sentence explaining what a synthetic aerosol profile is, and
            # -- when the fallback was reached rather than requested -- the
            # concrete search that came up empty.
            from woof.config import MP28_AEROSOL_SYNTHETIC_FALLBACK
            aerosol_initialization["aerosol_source"] = (
                "thompson_init-synthetic-profile")
            aerosol_initialization["aerosol_source_statement"] = (
                MP28_AEROSOL_SYNTHETIC_FALLBACK)
            aerosol_initialization["synthetic_fallback_in_use"] = True
            aerosol_initialization["synthetic_fallback_requested"] = (
                _source_choice == "synthetic")
            if wif_resolution is not None:
                aerosol_initialization["dataset"] = describe_wif_source(
                    wif_resolution)
            from woof.core.chem_mp_coupling import effective_mode
            if effective_mode(cfg) != "diagnose":
                _warn_synthetic_aerosol_fallback(
                    _source_choice, wif_resolution)
    # Install supplied moments after the explicit source-absent MP8/28
    # branches. The analysis must never be overwritten by their zero policy.
    if decoded_numbers:
        for name, value in number_moments.items():
            if state_xp is host_numpy:
                value = _host_float32(value)
            getattr(state, name)[...] = state_xp.asarray(value, dtype=state_xp.float32)
        number_receipt["initialized_state_fields"] = receipt_fingerprints(
            number_moments, lambda name: getattr(state, name))
        hydrometeor_initialization["number_moments"] = number_receipt
    # The analysed hydrometeors the source publishes on every frame ride
    # the root's specified boundary (woof.boundary_fields): the masses
    # this initializer installed, and the numbers the closure below seeds
    # from them on this same frame.  None published, water vapour only.
    boundary_masses = tuple(
        name for name in BOUNDARY_HYDROMETEOR_MASSES
        if name in boundary_species and name.upper() in hydrometeors)
    if boundary_masses:
        from woof.boundary_fields import (
            boundary_hydrometeor_fields, external_scalar_fields)
        state._external_scalar_boundary_fields = external_scalar_fields(
            cfg, aerosol_from_input=boundary_aerosol_from_input,
            boundary_species=boundary_masses)
        hydrometeor_initialization["lateral_boundary_species"] = list(
            boundary_hydrometeor_fields(cfg, boundary_masses))
    # CHEM BOUNDARY SOURCES (woof/chem_source_init.py).  Rows whose first
    # enabled boundary source is a data-store composition source take their
    # field for THIS frame on its final total hydrostatic pressure (WRF's
    # p+pb, the target WRF-Chem's mozbc uses), so source and target are both
    # total pressure.  A chem-off state has no ``chem`` attribute and nothing
    # here runs; a chem state with no such source returns at once.
    chem_initialization = ()
    if getattr(state, "chem", None) is not None:
        from woof.chem_source_init import fill_boundary_sources
        chem_initialization = tuple(fill_boundary_sources(
            state, cfg, valid_time=getattr(snapshot, "valid_time", None),
            latlon=_wif_grid_latlon_from(grid, state),
            pressure=_host_float32(total_pressure_h),
            frame="boundary" if boundary_only else "initial"))
        for receipt in chem_initialization:
            if "thompson_initialization" in receipt:
                number_source = receipt["thompson_initialization"]
                aerosol_initialization.update(number_source)
                boundary_aerosol_from_input = True
                carried = getattr(state, "_external_scalar_boundary_fields", ())
                state._external_scalar_boundary_fields = tuple(dict.fromkeys(
                    (*carried, *number_source["initialized_fields"])))
                initialized = aerosol_initialization.setdefault("initialized_state_fields", {})
                absent = aerosol_initialization.get("source_absent_state_fields", {})
                names = tuple(number_source["initialized_fields"])
                initialized.update(zip(names, _fingerprints_side_by_side(
                    getattr(state, name) for name in names)))
                for name in names:
                    absent.pop(name, None)
                if not absent:
                    aerosol_initialization.pop("source_absent_state_fields", None)
    # Thompson's cold-start numbers (module_initialize_real.F:4819-4855),
    # which real.exe runs after every analysed field is in place and which
    # fill only where the number is at or below zero: an installed
    # analysed NC/NR/NI above zero is kept, a zero one is seeded.  Only a
    # lane that installed analyzed mass has anything to close; the RH lane
    # carries no condensate and keeps real.exe's zeros and its empty
    # hydrometeor receipt.  The droplet number reads the aerosol real.exe
    # would hold here: the analyzed QNWFA just installed, the climatology,
    # or zero ahead of thompson_init's synthetic profile.
    mark_timing("surface_and_aerosol_initialization")
    if hydrometeors and cfg.mp_physics in (8, 28):
        if state_xp is not host_numpy or getattr(preprocess, "bounded_cuda", False):
            # The state is on the card: the closure runs there, gathering
            # only the seeded cells (woof/ingest/closure_device.py), with
            # the seed temperature from the libm crate's device twin.
            from woof.ingest.closure_device import (
                make_temperature_provider, thompson_cold_start_moment_closure)
            closure_receipt = thompson_cold_start_moment_closure(
                state, state_xp, cfg, alpha, receipt=not boundary_only,
                aerosol_number=(state.nwfa
                                if cfg.mp_physics in THOMPSON_AEROSOL_FAMILY else None),
                landmask=landmask,
                temperature=make_temperature_provider(theta_h, total_pressure_h),
                **({"chunk_cells": preprocess.chunk_cells} if bounded_cuda else {}))
        else:
            with pm.worker_limit(column_workers):
                closure_receipt = _thompson_cold_start_moment_closure(
                    state, state_xp, cfg, alpha, receipt=not boundary_only,
                    aerosol_number=(state.nwfa
                                    if cfg.mp_physics in THOMPSON_AEROSOL_FAMILY else None),
                    landmask=landmask,
                    temperature_fields=(theta_h, total_pressure_h),
                    # Built only if a rain or ice cell is seeded; only the
                    # seeded cells' columns are evaluated.
                    temperature_at=lambda indices: _temperature_from_potential_temperature(
                        theta_h.ravel()[indices], total_pressure_h.ravel()[indices],
                        column_workers=column_workers).astype(np.float32),
                    temperature=lambda: _temperature_from_potential_temperature(
                        theta_h, total_pressure_h,
                        column_workers=column_workers).astype(np.float32))
        hydrometeor_initialization["cold_start_moment_closure"] = closure_receipt
    elif int(cfg.mp_physics) == 10:
        # Morrison: real.exe seeds no number, and a mass at N = 0 is sized
        # at the slope limit (woof/ingest/handover_numbers.py).  The
        # droplet number is the scheme's own diagnostic NDCNST everywhere;
        # rain, ice, snow and graupel are diagnosed where an analysed mass
        # has no number, on every forcing frame, so the ring that carries
        # the mass carries its number (woof.boundary_fields).
        from woof.ingest import handover_numbers as _handover
        if _handover.closure_active(cfg):
            if getattr(state, "nc", None) is not None:
                state.nc[...] = state_xp.asarray(
                    state_xp.float32(_handover.MORRISON_NDCNST_PER_CM3 * 1.0e6)
                    * state_xp.asarray(alpha, dtype=state_xp.float32),
                    dtype=state_xp.float32)
            if hydrometeors:
                masses = {m: getattr(state, m) for m, _ in
                          _handover.MORRISON_HANDOVER_NUMBERS
                          if getattr(state, m, None) is not None}
                numbers = {n: getattr(state, n) for _, n in
                           _handover.MORRISON_HANDOVER_NUMBERS
                           if getattr(state, n, None) is not None}
                closure_receipt = _handover.close_numbers(
                    10, masses, numbers,
                    rho=lambda idx: 1.0 / _handover.take(
                        alpha, idx, masses["qr"].shape).astype("float64"),
                    temperature=lambda idx: _handover.temperature_from_theta(
                        _handover.take(theta_h, idx, masses["qr"].shape),
                        _handover.take(total_pressure_h, idx, masses["qr"].shape)),
                    rimed=int(getattr(cfg, "morr_rimed_ice", 1)))
                if closure_receipt is not None and not boundary_only:
                    closure_receipt["droplet_number"] = "NDCNST/rho everywhere"
                    hydrometeor_initialization["cold_start_moment_closure"] = closure_receipt
    mark_timing("cold_start_moment_closure")
    # CUDA transforms may feed a host setup state. Transfer their FP32
    # wind arrays explicitly; NumPy cannot implicitly consume a CuPy array.
    # Column blocks already installed their horizontal winds.
    if not blocked:
        if state_xp is host_numpy:
            u, v = _host_float32(u), _host_float32(v)
        state.u[...] = state_xp.asarray(u, dtype=state_xp.float32)
        state.v[...] = state_xp.asarray(v, dtype=state_xp.float32)
    if w_mass_h is None:
        # WRF_REAL_VERTICAL_VELOCITY_POLICY: the source carried no vertical
        # velocity, so W starts at exact zero and the receipt says so.
        state.w[...] = 0.0
    else:
        # The source's vertical motion, on the W interfaces: the start keeps
        # the updrafts its analysis holds.  The receipt replaces the zero
        # policy with the measured field.
        w_interfaces = interface_vertical_velocity(w_mass_h, xp=np)
        vertical_velocity_initialization.clear()
        vertical_velocity_initialization.update(vertical_velocity_receipt(
            vertical_velocity_source, w_mass_h, w_interfaces, xp=np))
        if state_xp is host_numpy:
            w_interfaces = _host_float32(w_interfaces)
        state.w[...] = state_xp.asarray(w_interfaces, dtype=state_xp.float32)
        del w_interfaces, w_mass_h
    mark_timing("vertical_velocity_upload")
    total_phi = None if boundary_only else _setup(state.phb + state.php)
    mark_timing("remaining_state_upload_and_geopotential_readback")
    # The ONE place a surface mixing ratio is published rather than
    # consumed: everything above has had WRF's raw value, and this is what
    # becomes Q2 in the physics driver, in wrfinput, and in the prepared
    # cache.  A no-op on the RH lane, whose _saturation_mixing_ratio
    # already returned nothing below this floor.
    published_surface_qv, surface_qv_floor = (
        _ops._floor_flag_sh_surface_mixing_ratio(surface_qv, fields["PSFC"]))
    # The native and preprocessing workers above free on their own threads
    # too; the next forcing time and the companion files start without it.
    _return_freed_host_memory()
    if vertical_disposition_job is not None and receipt_joins is not None:
        # The slot is taken here, where the join writes it, so the receipt's
        # key order is the joined call's.
        hydrometeor_initialization["vertical_disposition"] = None

        def join_vertical_disposition(job=vertical_disposition_job,
                                      into=hydrometeor_initialization):
            into["vertical_disposition"] = job.result()
        receipt_joins.append(join_vertical_disposition)
    elif vertical_disposition_job is not None:
        hydrometeor_initialization["vertical_disposition"] = (
            vertical_disposition_job.result())
        mark_timing("hydrometeor_disposition_receipt_join")
    if timing_report is not None:
        timing_report["total_seconds"] = perf_counter() - timing_start
    if cfg.use_theta_m:
        # Native real converts after all dry initialization (initialize_real:4904).
        # These are setup arrays only; forecast time never takes a host path.
        thermal_base = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
        state.thp[...] = ((thermal_base + state.thp)
                          * (state_xp.float32(1.0) + state_xp.float32(c.RVOVRD) * state.qv)
                          - thermal_base)
        state.thp0[...] = state.thp
    from woof.core.passive_tracer_contract import initialize_real_tracers
    initialize_real_tracers(state, cfg)
    result = RealInitResult(
        state=state, coord=coord, base=host_base,
        surface_pressure=surface_pressure, surface_qv=published_surface_qv,
        dry_mass=dry_mass, dry_pressure=dry_pressure,
        total_pressure=total_pressure_h, total_geopotential=total_phi,
        total_specific_volume=alpha,
        integrated_moisture_pressure=intq,
        hypsometric_opt=cfg.hypsometric_opt,
        boundary_tracers=boundary_tracers,
        hydrometeor_initialization=hydrometeor_initialization,
        aerosol_initialization=aerosol_initialization,
        vertical_velocity_initialization=vertical_velocity_initialization,
        surface_moisture_floor=surface_qv_floor,
        initial_perturbation=perturbation_receipt,
        prognostic_moisture_floor=prognostic_qv_floor,
        chem_initialization=chem_initialization)
    if device_route and not boundary_only:
        result = _ops.export_result(result, base=host_base)
    return result


def source_orography_from_catalog(catalog, grid, *,
                                  provider="era5_z_invariant",
                                  valid_time=None) -> np.ndarray:
    """Public real-ingest resolver for catalog-declared source terrain."""
    return _source_orography_from_catalog(
        catalog, grid, provider=provider, valid_time=valid_time)


def hydrostatic_residual(result: RealInitResult) -> np.ndarray:
    """Maximum discrete moist-hydrostatic residual of the live FP32 state.

    Pressure remains the setup-time thermodynamic target, while geopotential,
    dry mass, potential temperature, vapor, coefficients, and arithmetic inputs
    are read back from the initialized :class:`DomainState`.  This makes the
    gate sensitive to the actual ``state.php`` quantization loaded on device.

    The discrete operator is the one ``calc_p_alpha`` runs, which is what
    makes this a measurement of the state rather than of a rounding tree
    nobody executes: the layer geopotential difference is the exact
    float32 base difference plus ``state.dphb_resid``, and opt 2's log
    ratio is ``log1p((pfd - pfu)/pfu)`` off the float64-differenced
    coefficient drops.  Measured on the seven-level probe in
    tests/test_real_init.py
    (test_real_init_builds_nonnegative_balanced_fp32_domain_state), max
    residual 2.4934e-2 grading that state with the pre-2.6.6 operator and
    6.9648e-3 grading the same state with this one.
    """
    state = result.state
    phb = _host(state.phb)
    php = _host(state.php)
    dphb = np.asarray(
        np.asarray(phb[1:] - phb[:-1], dtype=np.float32)
        + _host(state.dphb_resid), dtype=np.float32)
    dphi = np.asarray(dphb + np.asarray(php[1:] - php[:-1],
                                        dtype=np.float32),
                      dtype=np.float32).astype(np.float64)
    dry_mass = _host(state.mub2d + state.mup)
    theta = _host(state.total_theta()) if getattr(state, "use_theta_m", 0) else _host(state.thb + state.thp)
    qv = _host(state.qv)
    pressure = np.asarray(result.total_pressure, dtype=np.float64)
    alpha = (c.RD * theta * (1.0 + c.RVOVRD * qv)
             * pm.power(pressure / c.P0, c.RCP) / pressure)
    if result.hypsometric_opt == 2:
        # Log-pressure hydrostatic operator on the total dry mass, the
        # opt-2 counterpart of the discrete d(phi)/d(eta) relation
        # (module_initialize_real.F:3970-3981 / :4002-4010).
        c3h = _host(state.c3h)[:, None, None]
        c4h = _host(state.c4h)[:, None, None]
        c3f = _host(state.c3f)[:, None, None]
        c4f = _host(state.c4f)[:, None, None]
        dc3f = _host(state.dc3f)[:, None, None]
        dc4f = _host(state.dc4f)[:, None, None]
        p_top = float(state.p_top)
        pfu = c3f[1:] * dry_mass[None] + c4f[1:] + p_top
        dpf = dc3f * dry_mass[None] + dc4f          # = pfd - pfu
        phm = c3h * dry_mass[None] + c4h + p_top
        ratio = dpf / pfu
        # The state's float32 coefficients give a float32 ratio, and
        # calc_p_alpha's CPU log1p for that is log1pf_array.
        log_ratio = (log1pf_array(ratio) if ratio.dtype == np.float32
                     else pm.log1p(ratio))
        residual = dphi - alpha * phm * log_ratio
    else:
        c1h = _host(state.c1h)[:, None, None]
        c2h = _host(state.c2h)[:, None, None]
        dnw = _host(state.dnw)[:, None, None]
        increment = c1h * dry_mass[None] + c2h
        residual = dphi + dnw * increment * alpha
    return np.max(np.abs(residual), axis=0)


__all__ = ["HRRR_ANALYZED_HYDROMETEORS",
           "HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS", "RealInitResult",
           "WRF_REAL_MP28_AEROSOL_SOURCE_POLICY", "hydrostatic_residual",
           "initialize_real", "source_orography_from_catalog",
           "surface_pressure_from_surface"]

from woof.config import THOMPSON_AEROSOL_FAMILY
