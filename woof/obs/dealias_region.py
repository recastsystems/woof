# ======================================================================
# THIRD-PARTY NOTICE.  This module drives a library ported from
# third-party work.  ArWen distributes it under the Apache License 2.0;
# the notice below belongs to that upstream.  Full texts are in the
# repository NOTICE and in the licenses/ directory.
#
#   region-global-dealias, the Rust library this module drives, is a port
#   of Py-ART's dealias_region_based.
#
#       Copyright (c) 2013, UChicago Argonne, LLC.  All rights reserved.
#
#   Py-ART is BSD 3-Clause with an added U.S. Department of Energy
#   government-rights paragraph (contract DE-AC02-06CH11357) requiring
#   that a modified derivative be clearly marked so as not to be confused
#   with the version available from ANL.  This port is such a derivative
#   and is named region_global_dealias, not Py-ART.  Full text in
#   licenses/LICENSE-PyART-Argonne-BSD-3-Clause.txt and in the port's own
#   tree at tools/region_global_dealias/PYART-LICENSE.txt.
# ======================================================================
"""The region-global dealiasing engine: WOOF-ArWen's Rust crate behind the seam.

:mod:`woof.obs.dealias` said it in its own docstring -- ``dealias_sweep``
was written against exactly what crosses the ingest seam "so that when the
plumbing exists it is replaced by a call, not by a rewrite."  This module is
that call.

What it drives
--------------
``region-global-dealias`` (`FahrenheitResearch/region-global-dealias
<https://github.com/FahrenheitResearch/region-global-dealias>`_) is WOOF-ArWen's
Rust port of Py-ART's ``dealias_region_based`` (Helmus & Collis 2016), which
is itself the Jing & Wiener (1993) region method with a dynamic network
reduction over the region graph.  The crate is vendored verbatim under
:data:`CRATE_RELATIVE` at the upstream commit recorded in
:data:`UPSTREAM_COMMIT`; nothing in this tree modifies the solver, and the
vendor directory carries the BSD-3-Clause notice that travels with the
Py-ART derivation.

Why a shared library and not a binary
-------------------------------------
The house has both shapes.  ``tools/grib1_bridge`` builds executables that
speak a tabular contract over stdout (:mod:`woof.bridges`), and it builds
one **cdylib** driven through :mod:`ctypes` (:mod:`woof.ingest.cpu_backend`).
The rule that picks between them is what crosses the seam: a bridge that
moves *files* is a process, and a bridge that moves *arrays* is a library.

This one moves arrays.  A velocity sweep is ``rows * gates`` floats -- 3.4 MB
for a NEXRAD super-resolution cut -- and there are fourteen of them per
volume, on a path that runs per assimilation cycle.  Routing that through a
process would mean writing 48 MB to disk and reading it back per volume to
avoid a data copy the library route does not make at all.  The crate already
exports the C ABI this needs (``region_global_dealias.h``, ``bw_dealias`` and
``bw_dealias_rift_v1``), so the library route is also the one that requires
no wrapper crate around WOOF-ArWen's code: woof calls his exported symbols
directly.

The ABI handshake is :func:`bw_abi_version`, checked against
:data:`REGION_DEALIAS_ABI` before any solve, exactly as
:class:`woof.ingest.cpu_backend.CpuPreprocessBackend` checks its own.  The
refinement path is versioned separately upstream and is checked separately
here: a library that predates it fails the RIFT probe instead of failing
inside a solve.

What this engine decides, and what it does not
----------------------------------------------
The two engines answer different questions and their gate decisions are
*expected* to differ:

* :mod:`woof.obs.dealias` (``vad-region``) anchors regions to an
  environmental VAD reference and **abstains** -- a region it cannot justify
  is rejected, and a gate that departs too far from the reference is
  rejected.  It is built for a filter that believes what it is handed.
* This engine (``region-global``) unfolds the whole region network jointly
  and **assigns a fold to every region it resolved**.  It carries no
  environmental reference, so it has nothing to abstain against: the
  solver itself refuses nothing, and every finite gate comes back from it
  either unchanged or unfolded.

That difference is the reason ``engine`` is a recorded parameter rather
than an implementation detail.  The three-state contract is preserved
exactly -- :data:`~woof.obs.dealias.STATE_REJECTED` is reachable for a
non-finite gate, for a sweep whose Nyquist cannot be believed, and for a
gate the physical bound below refuses -- so a consumer that reads
``state`` reads the same field whichever engine ran.

The physical bound
------------------
A solver that never abstains needs one check that does, or a region
resolved onto the wrong branch reaches the filter as a confident wind of
80-115 m/s.  :func:`~woof.obs.dealias.speed_bounded` is that check, and
it is the same one, by the same function and the same reason name, that
the VAD engine has always applied: a gate whose unfolded speed exceeds
:attr:`~woof.obs.dealias.DealiasParams.max_speed_ms` (75 m/s by default)
is REJECTED and counted under ``speed_out_of_range``.  Never clamped --
a clamped gate is a fabricated observation the filter cannot tell from a
measurement -- and never passed.

References
----------
Helmus, J. J. and S. M. Collis, 2016: The Python ARM Radar Toolkit
(Py-ART). *J. Open Research Software*, **4**(1), e25.

Jing, Z. and G. Wiener, 1993: Two-dimensional dealiasing of Doppler
velocities. *J. Atmos. Oceanic Technol.*, **10**, 798-808.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import threading
from typing import Final

import numpy as np

#: The engine name this module answers to.  :mod:`woof.obs.dealias`
#: imports it rather than repeating the string, so the selector and the
#: implementation cannot drift apart into two spellings of one engine.
ENGINE_REGION_GLOBAL: Final[str] = "region-global"

#: Upstream repository the vendored crate is a verbatim copy of.
UPSTREAM_URL: Final[str] = (
    "https://github.com/FahrenheitResearch/region-global-dealias")

#: The upstream commit ``tools/region_global_dealias`` was taken at.  This
#: is the provenance of the *algorithm*, so it is recorded here beside the
#: code that calls it and again in the vendor directory's own note -- a
#: reader asking "which version of ArWen's solver produced these winds"
#: must not have to diff two trees to find out.
UPSTREAM_COMMIT: Final[str] = "a7d4baf6b8a11ca5602fe44a533efd8200ef6cea"

#: Upstream crate version at :data:`UPSTREAM_COMMIT`.
UPSTREAM_VERSION: Final[str] = "0.2.0"

#: The vendored crate's path inside a checkout.
CRATE_RELATIVE: Final[str] = "tools/region_global_dealias"

#: Environment variable naming a prebuilt shared library.
REGION_DEALIAS_ENV: Final[str] = "WOOF_DEALIAS_REGION_BRIDGE"

#: ``bw_abi_version()`` this wrapper was written against.
REGION_DEALIAS_ABI: Final[int] = 1

#: ``bw_rift_api_version()`` the refinement path was written against.  Kept
#: separate because upstream versions it separately: the legacy solve
#: surface is frozen at 1 and refinement work is additive on top of it.
REGION_DEALIAS_RIFT_API: Final[int] = 1

#: ``BW_ERR_*`` from ``region_global_dealias.h``, in the words the header
#: uses.  A negative return is a contract violation by this wrapper, never
#: a property of the radar data, so each one names what the caller got
#: wrong rather than what the sweep looked like.
_ERRORS: Final[dict[int, str]] = {
    -1: "a required pointer was null",
    -2: "rows/gates were zero, overflowed, or exceeded the gate ceiling "
        "(rows * gates must not exceed 4194304)",
    -3: "an array was not 4-byte aligned",
    -4: "the options structure or a flag is not understood by this library",
    -5: "an advanced input length is inconsistent with the sweep",
    -6: "a refinement safety budget is outside its supported range",
    -7: "a reference field or reference kind is invalid",
    -8: "input and output memory ranges overlap",
    -9: "the polar range geometry is missing or non-physical",
}

#: ``BW_RIFT_REASON_*``, for reading the per-sweep reason mask back out.
RIFT_REASON_NAMES: Final[dict[int, str]] = {
    1 << 0: "residue_trigger",
    1 << 1: "branch_unstable",
    1 << 2: "temporal_anchor",
    1 << 3: "vertical_anchor",
    1 << 4: "environmental_anchor",
    1 << 5: "caller_anchor",
    1 << 6: "vortex_proposal",
    1 << 7: "fusion_accepted",
    1 << 8: "conflicting_references",
    1 << 9: "low_coverage",
    1 << 10: "abstained",
    1 << 11: "budget_exceeded",
    1 << 12: "nyquist_transition",
}


class RegionDealiasError(RuntimeError):
    """The native region-global solver refused a call, with its reason."""


class BwStats(ctypes.Structure):
    """``BwStats`` -- five ``u32``, 20 bytes, no padding."""

    _fields_ = [
        ("gates_total", ctypes.c_uint32),
        ("gates_finite", ctypes.c_uint32),
        ("gates_modified", ctypes.c_uint32),
        ("max_abs_fold", ctypes.c_uint32),
        ("wraps", ctypes.c_uint32),
    ]


class BwRiftOptionsV1(ctypes.Structure):
    """``BwRiftOptionsV1`` -- 48 bytes, ``struct_size`` first."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("max_abs_fold", ctypes.c_uint32),
        ("max_rois", ctypes.c_uint32),
        ("max_roi_gates", ctypes.c_uint32),
        ("max_total_roi_gates", ctypes.c_uint32),
        ("min_confidence", ctypes.c_uint32),
        ("first_gate_m", ctypes.c_float),
        ("gate_spacing_m", ctypes.c_float),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class BwRiftStatsV1(ctypes.Structure):
    """``BwRiftStatsV1`` -- 64 bytes, ``struct_size`` first."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("gates_total", ctypes.c_uint32),
        ("gates_finite", ctypes.c_uint32),
        ("gates_modified", ctypes.c_uint32),
        ("max_abs_fold", ctypes.c_uint32),
        ("wraps", ctypes.c_uint32),
        ("rois_detected", ctypes.c_uint32),
        ("rois_solved", ctypes.c_uint32),
        ("rois_accepted", ctypes.c_uint32),
        ("gates_refined", ctypes.c_uint32),
        ("gates_ambiguous", ctypes.c_uint32),
        ("budget_aborts", ctypes.c_uint32),
        ("reason_flags", ctypes.c_uint32),
        ("abstain_flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 2),
    ]


#: The upstream defaults for every refinement budget this wrapper does not
#: expose.  Copied from ``BwRiftOptionsV1::default()`` rather than left as
#: zeros: a zeroed ``max_rois`` is not "the default", it is "refine
#: nothing", and a silent no-op is the one outcome an opt-in refinement
#: pass must never produce.  ``max_total_roi_gates`` is genuinely 0
#: upstream, which means "no aggregate cap"; it is listed so that reading
#: this table answers the question rather than raising it.
_RIFT_DEFAULTS: Final[dict[str, int]] = {
    "max_abs_fold": 4,
    "max_rois": 4,
    "max_roi_gates": 65_536,
    "max_total_roi_gates": 0,
    "min_confidence": 160,
}


def library_name() -> str:
    """The cdylib filename cargo produces for this crate on this platform."""

    if os.name == "nt":
        return "region_global_dealias.dll"
    if os.uname().sysname == "Darwin":     # pragma: no cover - platform route
        return "libregion_global_dealias.dylib"
    return "libregion_global_dealias.so"


def crate_dir() -> Path:
    """The vendored crate inside a source checkout (may not exist)."""

    return Path(__file__).resolve().parent.parent.parent / "tools" \
        / "region_global_dealias"


def region_bridge_candidates() -> tuple[Path, ...]:
    """Deterministic library candidates, best first, without loading them.

    The same four-rung ladder every other built artifact uses -- see
    :func:`woof.bridges.artifact_candidates` and
    :func:`woof.obs.nexrad.nexrad_candidates`.  It is spelled out here
    rather than delegated because ``artifact_candidates`` searches
    ``tools/grib1_bridge``, and this crate is its own tree; the ladder is
    shared, the crate is not.
    """

    from woof.bridges import (default_bridge_dir, legacy_bridge_candidates,
                               packaged_bridge_dir)

    filename = library_name()
    candidates: list[Path] = []
    override = os.environ.get(REGION_DEALIAS_ENV)
    if override:
        candidates.append(Path(override))
    root = Path(__file__).resolve().parent.parent.parent
    candidates.extend((
        crate_dir() / "target" / "release" / filename,
        crate_dir() / "target" / "debug" / filename,
        root / "libexec" / "bridges" / filename,
        packaged_bridge_dir() / filename,
        default_bridge_dir() / filename,
        *legacy_bridge_candidates(filename),
    ))
    return tuple(candidates)


def find_region_bridge() -> Path | None:
    """First existing candidate, or None.

    An environment override naming a missing file is a hard error, for the
    reason :func:`woof.bridges.find_artifact` gives: explicit
    configuration must fail loudly rather than fall through to whatever
    other library happens to be on the ladder.
    """

    from woof.bridges import accept_resolved

    override = os.environ.get(REGION_DEALIAS_ENV)
    for candidate in region_bridge_candidates():
        if candidate.is_file():
            return accept_resolved(candidate.resolve())
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{REGION_DEALIAS_ENV} names a missing file: {candidate}")
    return None


def region_bridge_remedy() -> str:
    """The remedy for a missing library, true for THIS install."""

    from woof.bridges import artifact_remedy, cargo_build_one_liner

    return artifact_remedy(
        env_var=REGION_DEALIAS_ENV, filename=library_name(),
        subject="the region-global dealiasing engine",
        crate_relative=CRATE_RELATIVE,
        one_liner=cargo_build_one_liner(CRATE_RELATIVE))


def resolve_region_bridge(path: Path | str | None = None) -> Path:
    """Resolve the shared library, failing with every location searched."""

    if path is not None:
        explicit = Path(path)
        if explicit.is_file():
            return explicit.resolve()
        raise FileNotFoundError(
            "the region-global dealiasing engine was not found; searched:\n  "
            f"{explicit}\n\n{region_bridge_remedy()}")
    found = find_region_bridge()
    if found is not None:
        return found
    rendered = "\n  ".join(str(c) for c in region_bridge_candidates())
    raise FileNotFoundError(
        "the region-global dealiasing engine was not found; searched:\n  "
        + rendered + "\n\n" + region_bridge_remedy())


def _c_f32(array) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32)


def _ptr(array, kind=ctypes.c_float):
    return array.ctypes.data_as(ctypes.POINTER(kind))


class RegionDealiaser:
    """A loaded, ABI-checked region-global solver.

    One instance holds one library handle.  The native side allocates its
    own scratch per call and keeps no state between calls -- unlike the
    WebAssembly wrapper upstream ships, which owns persistent scratch
    buffers and is therefore not concurrency-safe.  Calls here are
    nonetheless serialised by :attr:`_lock`, because "the C entry points
    look reentrant" is an inference and a lock is a guarantee, and a solve
    is milliseconds.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = resolve_region_bridge(path)
        self._lock = threading.Lock()
        self.library = ctypes.CDLL(str(self.path))
        self._configure()

    def _configure(self) -> None:
        library = self.library
        library.bw_abi_version.argtypes = []
        library.bw_abi_version.restype = ctypes.c_uint32
        observed = int(library.bw_abi_version())
        if observed != REGION_DEALIAS_ABI:
            raise RegionDealiasError(
                f"{self.path} reports region-global dealias ABI {observed}, "
                f"this build speaks {REGION_DEALIAS_ABI}; rebuild the "
                f"vendored crate ({CRATE_RELATIVE}) from this checkout")
        self.abi_version = observed

        library.bw_dealias.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(BwStats)]
        library.bw_dealias.restype = ctypes.c_int32

        # The refinement surface is versioned separately upstream, so a
        # library that has the frozen legacy ABI and not this one is a
        # legitimate build rather than a broken one: record the gap and let
        # a caller who asks for refinement hear about it, instead of
        # refusing every solve.
        try:
            library.bw_rift_api_version.argtypes = []
            library.bw_rift_api_version.restype = ctypes.c_uint32
            self.rift_api_version = int(library.bw_rift_api_version())
        except AttributeError:
            self.rift_api_version = 0
        if self.rift_api_version == REGION_DEALIAS_RIFT_API:
            library.bw_dealias_rift_v1.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(BwRiftOptionsV1),
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int8),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.POINTER(BwRiftStatsV1)]
            library.bw_dealias_rift_v1.restype = ctypes.c_int32

    @property
    def refinement_available(self) -> bool:
        """Whether this library carries the refinement entry point."""

        return self.rift_api_version == REGION_DEALIAS_RIFT_API

    def _check(self, code: int, call: str) -> None:
        if code == 0:
            return
        reason = _ERRORS.get(int(code), "an unknown error code")
        raise RegionDealiasError(
            f"{call} returned {int(code)}: {reason} ({self.path})")

    def dealias(self, velocity: np.ndarray, azimuth_deg: np.ndarray,
                nyquist_ms: np.ndarray) -> tuple[np.ndarray, dict]:
        """``bw_dealias``: one sweep in, unfolded velocities and stats out."""

        velocity = _c_f32(velocity)
        rows, gates = velocity.shape
        azimuth = _c_f32(np.asarray(azimuth_deg).ravel())
        nyquist = _c_f32(np.asarray(nyquist_ms).ravel())
        if azimuth.size != rows or nyquist.size != rows:
            raise ValueError(
                f"azimuth ({azimuth.size}) and nyquist ({nyquist.size}) carry "
                f"one value per ray; this sweep has {rows} rays")
        out = np.empty((rows, gates), dtype=np.float32)
        stats = BwStats()
        with self._lock:
            code = self.library.bw_dealias(
                _ptr(velocity), _ptr(azimuth), _ptr(nyquist),
                ctypes.c_size_t(rows), ctypes.c_size_t(gates), _ptr(out),
                ctypes.byref(stats))
        self._check(code, "bw_dealias")
        return out, _stats_payload(stats)

    def dealias_rift(self, velocity: np.ndarray, azimuth_deg: np.ndarray,
                     nyquist_ms: np.ndarray, *, first_gate_m: float,
                     gate_spacing_m: float) -> tuple[np.ndarray, dict]:
        """``bw_dealias_rift_v1`` with automatic single-sweep refinement.

        The region-global result is computed first and returned unchanged
        wherever the refinement does not accept a proposal, so this is
        additive by construction: the only gates that can differ from
        :meth:`dealias` are ones an accepted refinement moved.
        """

        if not self.refinement_available:
            raise RegionDealiasError(
                f"{self.path} reports refinement API version "
                f"{self.rift_api_version}, this build speaks "
                f"{REGION_DEALIAS_RIFT_API}; rebuild the vendored crate "
                f"({CRATE_RELATIVE}) to use the refinement pass")
        velocity = _c_f32(velocity)
        rows, gates = velocity.shape
        azimuth = _c_f32(np.asarray(azimuth_deg).ravel())
        nyquist = _c_f32(np.asarray(nyquist_ms).ravel())
        if azimuth.size != rows or nyquist.size != rows:
            raise ValueError(
                f"azimuth ({azimuth.size}) and nyquist ({nyquist.size}) carry "
                f"one value per ray; this sweep has {rows} rays")
        options = BwRiftOptionsV1(
            struct_size=ctypes.sizeof(BwRiftOptionsV1),
            flags=0,                       # automatic single-sweep enabled
            first_gate_m=ctypes.c_float(float(first_gate_m)),
            gate_spacing_m=ctypes.c_float(float(gate_spacing_m)),
            **_RIFT_DEFAULTS)
        out = np.empty((rows, gates), dtype=np.float32)
        folds = np.zeros((rows, gates), dtype=np.int8)
        confidence = np.zeros((rows, gates), dtype=np.uint8)
        reasons = np.zeros((rows, gates), dtype=np.uint16)
        stats = BwRiftStatsV1()
        null_f32 = ctypes.POINTER(ctypes.c_float)()
        null_u8 = ctypes.POINTER(ctypes.c_uint8)()
        # THE REFERENCE ARGUMENTS STAY NULL.  ``bw_dealias_rift_v1`` takes
        # (reference_velocity, reference_quality, reference_kind,
        # reference_count) -- the ABI hook through which woof's own VAD
        # field, :func:`woof.obs.dealias.vad_reference`, could anchor the
        # refinement pass.  Wiring it is REFUSED for now: owner ruling,
        # 2026-08-12.  The two engines are selected against each other on
        # what they decide, and a region-global run whose refinement was
        # steered by the other engine's environmental fit is neither arm --
        # it would make every comparison in the lane's receipts describe a
        # configuration nobody chose.  If it is ever wired it arrives as
        # its own engine name with its own measurement, not as a quiet
        # argument change here.  The polarimetric inputs below
        # (reflectivity, spectrum width, rho_hv) are null for the plainer
        # reason that the sweep pack does not carry them to this seam.
        with self._lock:
            code = self.library.bw_dealias_rift_v1(
                _ptr(velocity), _ptr(azimuth), _ptr(nyquist),
                ctypes.c_size_t(rows), ctypes.c_size_t(gates),
                null_f32, null_u8, null_u8, ctypes.c_uint32(0),
                null_f32, null_f32, null_f32,
                ctypes.byref(options), _ptr(out),
                _ptr(folds, ctypes.c_int8), _ptr(confidence, ctypes.c_uint8),
                _ptr(reasons, ctypes.c_uint16), ctypes.byref(stats))
        self._check(code, "bw_dealias_rift_v1")
        payload = _rift_stats_payload(stats)
        payload["confidence_max"] = int(confidence.max()) if confidence.size \
            else 0
        return out, payload


def _stats_payload(stats: BwStats) -> dict:
    return {
        "gates_total": int(stats.gates_total),
        "gates_finite": int(stats.gates_finite),
        "gates_modified": int(stats.gates_modified),
        "max_abs_fold": int(stats.max_abs_fold),
        "wraps": bool(stats.wraps),
    }


def _reason_names(mask: int) -> list[str]:
    return [name for bit, name in sorted(RIFT_REASON_NAMES.items())
            if int(mask) & bit]


def _rift_stats_payload(stats: BwRiftStatsV1) -> dict:
    return {
        "gates_total": int(stats.gates_total),
        "gates_finite": int(stats.gates_finite),
        "gates_modified": int(stats.gates_modified),
        "max_abs_fold": int(stats.max_abs_fold),
        "wraps": bool(stats.wraps),
        "rois_detected": int(stats.rois_detected),
        "rois_solved": int(stats.rois_solved),
        "rois_accepted": int(stats.rois_accepted),
        "gates_refined": int(stats.gates_refined),
        "gates_ambiguous": int(stats.gates_ambiguous),
        "budget_aborts": int(stats.budget_aborts),
        "reasons": _reason_names(stats.reason_flags),
        "abstained": _reason_names(stats.abstain_flags),
    }


_CACHE: dict[str, RegionDealiaser] = {}
_CACHE_LOCK = threading.Lock()


def load_region_dealiaser(path: Path | str | None = None) -> RegionDealiaser:
    """The loaded solver for ``path``, opened at most once per process.

    A volume is fourteen sweeps and a cycle is many volumes; reopening the
    library and re-running the ABI handshake per sweep would make the
    handshake the measurement.  Keyed on the resolved path so an explicit
    override and the ladder's answer are never confused for each other.
    """

    resolved = str(resolve_region_bridge(path))
    with _CACHE_LOCK:
        engine = _CACHE.get(resolved)
        if engine is None:
            engine = RegionDealiaser(resolved)
            _CACHE[resolved] = engine
    return engine


def region_engine_available() -> bool:
    """Whether this install can run the region-global engine at all.

    Answered at the front door for the reason
    :func:`woof.obs.dealias.scipy_available` is: an operator asking for an
    engine this box cannot run should hear it in the second before the run
    starts, not from a traceback an hour into a cycle.
    """

    try:
        return find_region_bridge() is not None
    except FileNotFoundError:
        return False


def _uniform_nyquist_or_refuse(rows: int, nyquist_by_radial, disagree: bool):
    """The believable per-radial mask, once the sweep is known uniform.

    Returns ``(believable_row, distinct)`` -- a boolean per radial and the
    distinct believable values -- or raises when this engine must not run.

    **Why this engine refuses a nonuniform cut instead of serving it.**
    The native solver takes a per-ray Nyquist pointer, which reads as
    though it solves per ray.  It does not.  ``region_folds``
    (``solver.rs``) reduces the array with ``sweep_nyquist`` -- *the first
    usable element*, not the median and not the minimum -- and every
    cross-ray fold decision divides by that one interval, while
    ``apply_folds`` then applies the chosen integer at each ray's *own*
    interval.  On a cut split between 25.51 m/s and 32.0 m/s that decide/
    apply mismatch puts every corrected gate on the 32.0 m/s half 12.98 m/s
    out per fold, and multi-fold seams pick the wrong integer outright
    (a true 2-fold step reads as 2.509 and rounds to 3).  The error is
    finite, smooth, inside ``max_speed_ms``, and reported as a clean
    integer fold, so nothing downstream can see it.  Worse, because the
    scalar is the *first* usable element, rotating the same sweep changes
    the answer.

    So: this engine runs a uniform sweep, and a nonuniform sweep is the
    VAD-referenced engine's, which carries the array into its own
    arithmetic.  A radial with no believable value is a narrower case and
    is refused radial by radial rather than costing the sweep -- the
    remaining radials still share one interval, which is the only thing
    this solver needed.
    """

    from woof.obs.dealias import DealiasParamsError  # noqa: PLC0415

    if nyquist_by_radial is None:
        if disagree:
            raise DealiasParamsError(
                "this sweep reports nyquist_radials_disagree=True and carries "
                "no nyquist_velocity_ms_by_radial, so the only Nyquist "
                "available is the sweep summary -- and unfolding a nonuniform "
                "cut in one interval mis-corrects every radial above it by a "
                "whole fold difference, finite and plausible and undetectable "
                "downstream.  Re-decode the volume with a decoder that emits "
                "the per-radial Nyquist array, or run this volume without "
                "dealiasing")
        return np.ones(rows, dtype=bool), None

    row_nyquist = np.asarray(nyquist_by_radial, dtype=np.float64).ravel()
    if row_nyquist.size != rows:
        raise ValueError(
            f"nyquist_by_radial has {row_nyquist.size} entries, the sweep "
            f"has {rows} radials")
    believable = np.isfinite(row_nyquist) & (row_nyquist > 0.0)
    distinct = np.unique(row_nyquist[believable])
    if disagree or distinct.size > 1:
        raise DealiasParamsError(
            f"engine={ENGINE_REGION_GLOBAL!r} decides every fold in ONE "
            f"Nyquist interval for the whole sweep -- the native solver "
            f"reduces the per-ray array to its first usable element -- and "
            f"this sweep is nonuniform: {[float(v) for v in distinct]} m/s "
            f"across its radials.  Unfolding it here mis-corrects the radials "
            f"whose interval differs from that one by a whole fold, an error "
            f"that is finite, inside every speed bound, reported as a clean "
            f"integer fold, and dependent on which radial happens to be "
            f"first.  Select engine='vad-region', which unfolds each radial "
            f"in its own interval")
    return believable, distinct


def dealias_sweep_region(velocity: np.ndarray, azimuth_deg: np.ndarray,
                         nyquist: float | None, params,
                         *, first_gate_m: float | None = None,
                         gate_spacing_m: float | None = None,
                         nyquist_by_radial: np.ndarray | None = None,
                         nyquist_radials_disagree: bool = False,
                         library: Path | str | None = None):
    """Unfold one sweep with the region-global engine.

    ``nyquist_by_radial`` is the sweep's per-radial Nyquist velocity, NaN
    where a radial reports none.  This engine uses it for exactly two
    things -- refusing a nonuniform sweep, and refusing the individual
    radials that carry no believable value -- because that is the whole of
    what it can do with it correctly.  See
    :func:`_uniform_nyquist_or_refuse` for the measurement behind that
    sentence.

    Returns the same :class:`~woof.obs.dealias.SweepDealiasResult` the
    VAD-referenced engine returns, so :mod:`woof.obs.superob` consumes
    either without knowing which ran.  The planes carry what this engine
    actually knows:

    ``state``
        :data:`~woof.obs.dealias.STATE_UNFOLDED` where a whole number of
        Nyquist intervals was applied, ``STATE_UNCHANGED`` where none was,
        and :data:`~woof.obs.dealias.STATE_REJECTED` where the gate was
        not a number to begin with, where the sweep had no believable
        Nyquist, or where the unfolded speed left the physical bound.  The
        solver abstains at none of those -- see the module docstring.
    ``reference``
        All NaN.  There is no environmental reference in this algorithm,
        and filling the plane with the raw velocities (or with zeros) would
        state one that was never fitted.
    """

    from woof.obs.dealias import (REASON_NO_NYQUIST, REASON_NONE,
                                   REASON_NONFINITE, REASON_SPEED,
                                   STATE_REJECTED, STATE_UNCHANGED,
                                   STATE_UNFOLDED, REASON_NAMES,
                                   SweepDealiasResult, speed_bounded)

    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 2:
        raise ValueError(
            f"dealias_sweep_region needs a (radial, gate) plane, got shape "
            f"{velocity.shape}")

    # The default route: the whole sweep -- including the per-sector solve
    # and reconciliation of a mixed-Nyquist sweep below -- runs natively in
    # rw-superob, which links this same solver.  What follows in this
    # function is the parity reference (WOOF_SUPEROB_PYTHON=1).
    from woof.obs import superob_bridge                  # noqa: PLC0415

    lib = superob_bridge.route()
    if lib is not None:
        output, state, reason, fold_plane, stats = (
            superob_bridge.dealias_region(
                lib, velocity, azimuth_deg, nyquist, params,
                first_gate_m=first_gate_m, gate_spacing_m=gate_spacing_m,
                nyquist_by_radial=nyquist_by_radial,
                nyquist_radials_disagree=nyquist_radials_disagree))
        for native in [stats.get("native"), *stats.get("native_sectors", ())]:
            if isinstance(native, dict) and "skipped" not in native:
                native["library"] = str(lib.path)
                native["upstream_commit"] = UPSTREAM_COMMIT
        return SweepDealiasResult(
            output, state, reason, fold_plane,
            np.full(velocity.shape, np.nan, dtype=np.float64), stats)
    rows, gates = velocity.shape
    state = np.zeros(velocity.shape, dtype=np.int8)
    reason = np.full(velocity.shape, REASON_NONFINITE, dtype=np.int8)
    fold_plane = np.zeros(velocity.shape, dtype=np.int16)
    output = np.full(velocity.shape, np.nan, dtype=np.float64)
    reference_plane = np.full(velocity.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(velocity)

    stats = {
        "engine": ENGINE_REGION_GLOBAL,
        "gates_finite": int(finite.sum()),
        "gates_unchanged": 0,
        "gates_unfolded": 0,
        "gates_rejected": 0,
        "rejected": {name: 0 for name in REASON_NAMES.values()
                     if name not in ("none", "nonfinite", "disabled")},
        "fold_histogram": {},
        "reference": {"external_supplied": False, "bands": 0,
                      "bands_valid": 0},
        "nyquist_ms": None if nyquist is None else float(nyquist),
    }

    # ---- a nonuniform sweep: one solve per constant-Nyquist sector --------
    # The breakage this replaces, named: a nonuniform sweep was refused
    # outright, and WSR-88D VCPs mix Nyquist within a cut often enough that
    # a whole case (the da-tune 2026-09-06 case) failed to build its
    # observations.  The native solver decides folds in one interval, so it
    # is run once per sector of radials sharing one Nyquist value, with
    # every other radial masked to no-data: no region, and so no fold
    # decision, ever spans two intervals, and every correction is an exact
    # multiple of that radial's own 2*Vn.  The VAD-referenced engine already
    # fits one sector at a time for the same reason.
    if nyquist_by_radial is not None:
        row_nyquist = np.asarray(nyquist_by_radial, dtype=np.float64).ravel()
        if row_nyquist.size == rows:
            believable_all = np.isfinite(row_nyquist) & (row_nyquist > 0.0)
            sectors = np.unique(row_nyquist[believable_all])
            if sectors.size > 1:
                return _dealias_by_sector(
                    velocity, azimuth_deg, row_nyquist, believable_all,
                    sectors, params, first_gate_m=first_gate_m,
                    gate_spacing_m=gate_spacing_m, library=library,
                    stats=stats, state=state, reason=reason,
                    fold_plane=fold_plane, output=output,
                    reference_plane=reference_plane)

    # Raises before any work when this sweep is not one this engine can
    # solve; otherwise tells us which radials carry a believable value.
    believable_row, distinct_row = _uniform_nyquist_or_refuse(
        rows, nyquist_by_radial, bool(nyquist_radials_disagree))
    stats["nyquist_by_radial"] = nyquist_by_radial is not None
    stats["nyquist_radials_no_value"] = int((~believable_row).sum())
    if distinct_row is not None and distinct_row.size:
        # The array is uniform here by construction, so this is the same
        # number the scalar carries -- recorded because a receipt that
        # says which radials were dropped has to say what the kept ones
        # agreed on.
        stats["nyquist_distinct"] = [float(v) for v in distinct_row]

    # A radial with no believable Nyquist is not a radial with the sweep's
    # Nyquist.  The native solver would substitute the sweep median for it
    # silently, with no reason bit and no stat; a gate whose Nyquist is
    # unknown has an unknown fold state, so it is refused here instead --
    # the same refusal this function already makes for a whole sweep, and
    # the same one the VAD-referenced engine makes radial by radial.
    if not believable_row.all():
        refused = finite & ~believable_row[:, None]
        reason[refused] = REASON_NO_NYQUIST
        stats["rejected"]["no_nyquist"] = int(refused.sum())
        # ``gates_finite`` is left alone: it is what this sweep OFFERED,
        # and the volume account balances offered against unchanged +
        # unfolded + rejected.  The VAD-referenced engine leaves it alone
        # across the same refusal for the same reason.
        finite = finite & believable_row[:, None]
        if not finite.any():
            stats["gates_rejected"] = int(refused.sum())
            return SweepDealiasResult(output, state, reason, fold_plane,
                                      reference_plane, stats)

    if nyquist is None or not np.isfinite(nyquist) or float(nyquist) <= 0.0:
        # The same refusal woof.obs.superob and the VAD engine already
        # make.  The native solver would instead fall back to the sweep
        # median Nyquist and pass the velocities through, which is a
        # reasonable answer for a display and the wrong one for a filter:
        # a gate whose Nyquist is unknown has an unknown fold state, and
        # this pipeline says so rather than guessing.
        reason[finite] = REASON_NO_NYQUIST
        stats["gates_rejected"] = int(finite.sum())
        stats["rejected"]["no_nyquist"] = int(finite.sum())
        return SweepDealiasResult(output, state, reason, fold_plane,
                                  reference_plane, stats)
    if not finite.any():
        stats["native"] = {"gates_total": int(velocity.size),
                           "gates_finite": 0, "gates_modified": 0,
                           "max_abs_fold": 0, "wraps": None,
                           "skipped": "no finite gate in this sweep"}
        return SweepDealiasResult(output, state, reason, fold_plane,
                                  reference_plane, stats)

    nyquist = float(nyquist)
    interval = 2.0 * nyquist
    engine = load_region_dealiaser(library)
    nyquist_rays = np.full(rows, nyquist, dtype=np.float32)
    refine = bool(getattr(params, "refinement", False))
    if refine:
        if first_gate_m is None or gate_spacing_m is None:
            raise ValueError(
                "the region-global refinement pass needs the sweep's physical "
                "gate geometry (first_gate_m, gate_spacing_m); it fits a "
                "wrapped vortex in real space and cannot do that on gate "
                "indices")
        unfolded, native = engine.dealias_rift(
            velocity, azimuth_deg, nyquist_rays,
            first_gate_m=float(first_gate_m),
            gate_spacing_m=float(gate_spacing_m))
    else:
        unfolded, native = engine.dealias(velocity, azimuth_deg, nyquist_rays)
    native["refinement"] = refine
    native["library"] = str(engine.path)
    native["upstream_commit"] = UPSTREAM_COMMIT
    stats["native"] = native

    unfolded = np.asarray(unfolded, dtype=np.float64)
    # The engine moves a gate by a whole number of Nyquist intervals and
    # nothing else, and returns an untouched gate bit-identically.  Rounding
    # the ratio therefore recovers the integer it applied rather than
    # estimating it; the ratio is checked against that integer below so a
    # library that ever stopped honouring the contract is caught here rather
    # than silently reported as a fold of zero.
    ratio = np.zeros(velocity.shape, dtype=np.float64)
    ratio[finite] = (unfolded[finite] - velocity[finite]) / interval
    fold = np.rint(ratio).astype(np.int16)
    drift = np.abs(ratio - fold)
    worst = float(drift[finite].max()) if finite.any() else 0.0
    if worst > 1e-3:
        raise RegionDealiasError(
            f"{engine.path} moved a gate by {worst:.6f} of a Nyquist interval "
            "away from a whole number; the engine's contract is that it "
            "applies whole intervals only, so this library does not speak it")

    state[finite] = np.where(fold[finite] != 0, STATE_UNFOLDED,
                             STATE_UNCHANGED)
    reason[finite] = REASON_NONE
    fold_plane[finite] = fold[finite]
    output[finite] = unfolded[finite]

    # ---- the physical bound ---------------------------------------------
    # The only observation-QC this engine has.  It assigns a fold to every
    # region it resolved and reports no refusal of its own, so a region
    # placed on the wrong branch arrives here as a finite, confident wind
    # of 80-115 m/s and reaches the filter as a measurement.  The same
    # bound the VAD engine applies, by the same function and under the
    # same reason name, so a consumer reading ``rejected`` finds one
    # spelling of one refusal whichever engine ran.
    beyond = speed_bounded(output, state, reason,
                           max_speed_ms=params.max_speed_ms)
    output[beyond] = np.nan
    fold_plane[beyond] = 0
    stats["max_speed_ms"] = float(params.max_speed_ms)

    # Counted over the FINITE gates only, which is what ``gates_offered``
    # means and what the volume account balances against.  ``state`` is
    # zero-initialised and ``STATE_REJECTED`` is zero, so counting over the
    # whole plane would report every no-data gate as a refusal -- the
    # totals stop balancing and the engine looks like it threw away
    # three million observations it was never handed.
    finite_state = state[finite]
    stats["gates_unchanged"] = int((finite_state == STATE_UNCHANGED).sum())
    stats["gates_unfolded"] = int((finite_state == STATE_UNFOLDED).sum())
    # Plus the gates already refused above for carrying no believable
    # Nyquist: they were taken out of ``finite`` so the solver never saw
    # them, and counting only over ``finite`` would report them as
    # observations nobody was ever handed rather than as refusals.
    stats["gates_rejected"] = (int((finite_state == STATE_REJECTED).sum())
                               + int(stats["rejected"]["no_nyquist"]))
    stats["rejected"][REASON_NAMES[REASON_SPEED]] = int(beyond.sum())
    applied = fold_plane[finite & ~beyond]
    if applied.size:
        values, counts = np.unique(applied, return_counts=True)
        stats["fold_histogram"] = {int(v): int(c)
                                   for v, c in zip(values, counts)}
    return SweepDealiasResult(output, state, reason, fold_plane,
                              reference_plane, stats)


def _dealias_by_sector(velocity, azimuth_deg, row_nyquist, believable,
                       sectors, params, *, first_gate_m, gate_spacing_m,
                       library, stats, state, reason, fold_plane, output,
                       reference_plane):
    """A nonuniform sweep unfolded sector by sector (see the caller)."""
    from woof.obs.dealias import (REASON_NO_NYQUIST,  # noqa: PLC0415
                                   SweepDealiasResult)

    rows = velocity.shape[0]
    finite = np.isfinite(velocity)
    stats["nyquist_by_radial"] = True
    stats["nyquist_distinct"] = [float(v) for v in sectors]
    stats["nyquist_radials_no_value"] = int((~believable).sum())
    stats["nyquist_sectors"] = []
    stats["nyquist_ms"] = float(sectors.min())
    unknown = finite & ~believable[:, None]
    reason[unknown] = REASON_NO_NYQUIST
    stats["rejected"]["no_nyquist"] = int(unknown.sum())
    totals = {"gates_unchanged": 0, "gates_unfolded": 0, "gates_rejected":
              int(unknown.sum())}
    histogram: dict[int, int] = {}
    for value in sectors:
        sel = believable & (row_nyquist == value)
        masked = np.where(sel[:, None], velocity, np.nan)
        part = dealias_sweep_region(
            masked, azimuth_deg, float(value), params,
            first_gate_m=first_gate_m, gate_spacing_m=gate_spacing_m,
            nyquist_by_radial=np.where(sel, float(value), np.nan),
            nyquist_radials_disagree=False, library=library)
        rows_sel = np.flatnonzero(sel)
        output[rows_sel] = part.velocity[rows_sel]
        state[rows_sel] = part.state[rows_sel]
        reason[rows_sel] = part.reason[rows_sel]
        fold_plane[rows_sel] = part.fold[rows_sel]
        pst = part.stats
        for key in totals:
            if key == "gates_rejected":
                # the masked radials are not this sector's refusals
                totals[key] += int(pst["gates_rejected"]) - int(
                    pst["rejected"].get("no_nyquist", 0))
            else:
                totals[key] += int(pst[key])
        for name, count in pst["rejected"].items():
            if name != "no_nyquist":
                stats["rejected"][name] = stats["rejected"].get(name, 0) + int(count)
        for fold, count in pst.get("fold_histogram", {}).items():
            histogram[int(fold)] = histogram.get(int(fold), 0) + int(count)
        stats["nyquist_sectors"].append({
            "nyquist_ms": float(value), "radials": int(rows_sel.size),
            "gates_unfolded": int(pst["gates_unfolded"])})
        stats["max_speed_ms"] = pst.get("max_speed_ms")
        stats.setdefault("native_sectors", []).append(pst.get("native"))
    # ---- one sweep, not N independent ones --------------------------------
    # A sector solved alone has no anchor for its absolute fold: measured on
    # a 25.51/32.0 m/s half-and-half cut, one half came back a whole
    # 51.02 m/s fold off while every correction still sat on its own
    # lattice.  Sectors are therefore shifted by whole multiples of their
    # own interval to agree with their azimuthal neighbours across each
    # sector boundary, and the reference sector's own shift is chosen so the
    # whole sweep's mean radial velocity is closest to zero (the VAD
    # condition: a horizontally uniform wind averages to zero round the
    # circle).  Shifted gates pass the same physical bound again.
    shifts = _reconcile_sectors(output, azimuth_deg, row_nyquist, believable,
                                sectors)
    from woof.obs.dealias import (REASON_SPEED, STATE_REJECTED,  # noqa: PLC0415
                                   STATE_UNCHANGED, STATE_UNFOLDED)
    for value, k in shifts.items():
        if k == 0:
            continue
        rows_sel = believable & (row_nyquist == value)
        live = rows_sel[:, None] & (state != STATE_REJECTED)
        output[live] += k * 2.0 * value
        fold_plane[live] += k
        state[live] = np.where(fold_plane[live] != 0, STATE_UNFOLDED,
                               STATE_UNCHANGED)
    max_speed = float(getattr(params, "max_speed_ms", np.inf))
    beyond = (state != STATE_REJECTED) & (np.abs(output) > max_speed)
    if beyond.any():
        output[beyond] = np.nan
        fold_plane[beyond] = 0
        state[beyond] = STATE_REJECTED
        reason[beyond] = REASON_SPEED
    stats["sector_shifts"] = {str(float(v)): int(k) for v, k in shifts.items()}
    finite_offered = finite & believable[:, None]
    fs = state[finite_offered]
    totals["gates_unchanged"] = int((fs == STATE_UNCHANGED).sum())
    totals["gates_unfolded"] = int((fs == STATE_UNFOLDED).sum())
    totals["gates_rejected"] = (int((fs == STATE_REJECTED).sum())
                                + int(unknown.sum()))
    from woof.obs.dealias import REASON_NAMES  # noqa: PLC0415
    stats["rejected"][REASON_NAMES[REASON_SPEED]] = int(
        ((reason == REASON_SPEED) & finite).sum())
    applied = fold_plane[finite_offered & (state != STATE_REJECTED)]
    histogram = {}
    if applied.size:
        values, counts = np.unique(applied, return_counts=True)
        histogram = {int(v): int(c) for v, c in zip(values, counts)}
    stats.update(totals)
    stats["fold_histogram"] = dict(sorted(histogram.items()))
    return SweepDealiasResult(output, state, reason, fold_plane,
                              reference_plane, stats)


def _reconcile_sectors(output, azimuth_deg, row_nyquist, believable, sectors):
    """``{nyquist: integer shift}`` making the sectors one consistent sweep."""
    azimuth = np.asarray(azimuth_deg, dtype=np.float64).ravel()
    order = np.argsort(azimuth, kind="stable")
    order = order[believable[order]]
    sector_of = {}
    for row in order:
        sector_of[int(row)] = float(row_nyquist[row])
    # boundary radial pairs, azimuthal neighbours in different sectors
    pairs = []
    n = order.size
    for i in range(n):
        a, b = int(order[i]), int(order[(i + 1) % n])
        if sector_of[a] != sector_of[b]:
            pairs.append((a, b))
    keys = [float(v) for v in sectors]
    sizes = {v: int(np.count_nonzero(believable & (row_nyquist == v)))
             for v in keys}
    reference = max(keys, key=lambda v: (sizes[v], -v))

    def propagate(ref_shift):
        shift = {reference: ref_shift}
        for _ in range(len(keys)):
            for a, b in pairs + [(q, p) for p, q in pairs]:
                sa, sb = sector_of[a], sector_of[b]
                if sa in shift and sb not in shift:
                    va = output[a] + shift[sa] * 2.0 * sa
                    vb = output[b]
                    ok = np.isfinite(va) & np.isfinite(vb)
                    if ok.sum() < 3:
                        continue
                    k = int(np.rint(np.median(va[ok] - vb[ok]) / (2.0 * sb)))
                    shift[sb] = k
        for v in keys:
            shift.setdefault(v, 0)
        return shift

    def sweep_mean(shift):
        total, count = 0.0, 0
        for v, k in shift.items():
            block = output[believable & (row_nyquist == v)]
            ok = np.isfinite(block)
            total += float((block[ok] + k * 2.0 * v).sum())
            count += int(ok.sum())
        return abs(total / count) if count else 0.0

    covered = np.unique(np.floor(azimuth[believable] / 10.0)).size
    candidates = (-2, -1, 0, 1, 2) if covered >= 27 else (0,)
    best = min((propagate(k) for k in candidates), key=sweep_mean)
    return best


__all__ = [
    "CRATE_RELATIVE",
    "ENGINE_REGION_GLOBAL",
    "REGION_DEALIAS_ABI",
    "REGION_DEALIAS_ENV",
    "REGION_DEALIAS_RIFT_API",
    "RIFT_REASON_NAMES",
    "RegionDealiasError",
    "RegionDealiaser",
    "UPSTREAM_COMMIT",
    "UPSTREAM_URL",
    "UPSTREAM_VERSION",
    "crate_dir",
    "dealias_sweep_region",
    "find_region_bridge",
    "library_name",
    "load_region_dealiaser",
    "region_bridge_candidates",
    "region_bridge_remedy",
    "region_engine_available",
    "resolve_region_bridge",
]
