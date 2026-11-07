"""Pinned Arwen two-phase physics backend for resident MPAS-A v8.4.1.

This is the production adapter between the audited MPAS CUDA preparation
carriers and woof's persistent ``run_mpas_column_batch`` seam.  It owns no
parameterization arithmetic: phase one and WSM6 remain the exact frozen-v2 Arwen
objects, while MPAS preparation, optional native YSU-GWDO, conservative
coupling, and post-RK recovery remain the separately audited MPAS objects.

The adapter is deliberately strict.  Construction requires a sealed mapping
containing every real surface, soil, land-use, solar, and cadence input; no
Arwen constructor default is admitted.  The imported Arwen tree is accepted
only when all production source bytes match
:data:`ARWEN_BUILD_COMMIT` below, which is the ONE place that commit is
spelled -- this sentence deliberately names no digest of its own -- or when
they are, in full and under its own version string, the measured row of an
engine :mod:`woof.hex.engine_pin` admits (``ADMITTED_ENGINES``); a run on
such an engine is provenanced by THAT row's commit and digests, and the
constants below stay the public pin's.  A prose
copy of the pin is how this module came to contradict itself: it carried the
0.1.x seam-converge merge commit as present-tense fact for the whole of the
2.5.8 re-pin, while the constant below already named the 2.5.8 cut, so the
refusal a user hit and the documentation they read named different engines.
Every step is transactional: the boundary export stays on the device before
phase one. An abort or adapter failure materializes that export on the host,
rebuilds a fresh seam and restores the boundary snapshot.

The export is the ROLLBACK, and a caller that will never use a rollback may
decline it (``rollback_snapshot=False``). The armed export copies persisted
device arrays to detached device buffers. Host-owned arrays remain on the
host, and an allocation failure uses the original host exporter. What reads
the backup is a refused
step: the seam is rebuilt from it so the run can keep the last committed
state.  The forecast keeps that state only under ``--stop-on-refusal``,
where it writes the last committed frame; otherwise a refused step ends the
run and the restored seam is never read again.  A checkpoint does not read
it either: :meth:`restart_state` exports the seam at the committed boundary
itself.  An unarmed transaction that fails retires the seam instead of
restoring it (phase ``rollback_not_armed``): nothing is published, the
original error propagates, and every later step refuses.  The default is
armed, so every caller that does not say otherwise, the proof harness
among them, is unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
import sys
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable

import numpy as np

from . import engine_pin
from .diagnostic_export import diagnostic_export
from .cuda_backend.containers import require_resident_array
from .cuda_backend.runtime import KernelCache
from .cuda_gwdo_v841 import (
    CUDA_GWDO_V841_CONTRACT_SHA256,
    CUDA_GWDO_V841_KERNEL_SHA256,
    CudaYsuGwdoColumnViewV841,
    CudaYsuGwdoResultV841,
    CudaYsuGwdoStaticV841,
    run_bl_ysu_gwdo_cuda_v841,
)
from .cuda_physics_prep_v841 import (
    CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256,
    CUDA_PHYSICS_PREP_V841_KERNEL_SHA256,
    CudaMpasToPhysGeometryV841,
    prepare_mpas_to_phys_cuda_v841,
)
from .species_row import registered_species_rows, species_row_for_names
from .cuda_seam_snapshot_v841 import DeviceBoundarySnapshotV841
from .cuda_physics_v841 import (
    CUDA_PHYSICS_V841_CONTRACT_SHA256,
    CUDA_PHYSICS_V841_KERNEL_SHA256,
    CudaPhaseOneExecutionProvenanceV841,
    CudaPostRkWsm6UpdateV841,
    CudaRawColumnPhysicsV841,
    RV_OVER_RD_F32,
)


#: v3 (WOOF 1.0.1): the authority document's texts and keys name WOOF and
#: the engine where v2 named the engine's earlier name; no field, check or
#: arithmetic moved.  v2 is read back as WOOF_1_0_0_CONTRACT_IDS below.
CUDA_ARWEN_PHYSICS_V841_SCHEMA = "mpas-port.cuda-woof-physics-v841/v3"
ARWEN_BUILD_COMMIT = "4726b689c6db142577eca3968a1f74256e3c057f"
MPAS_SEAM_CONTRACT_SHA256 = (
    "5c629e23be2af20c0b1660d262443c415256126b812493f6681590bf07aff92a"
)
MPAS_SEAM_CONTRACT_SURFACE_SHA256 = (
    "649e5483357fa2748ab9b111dd3ecc974015a4bcbf4b5c0126f093798877108a"
)
ARWEN_GLACIER_COMPOSED_TU_SHA256 = (
    "8b0589436845fbec04711b14b8f2a865062f018ad3108df969fe38749e4c78f1"
)

# Every byte below participates in the phase-one/phase-two execution path.
# Hashing the imported files at construction makes the private effective-
# radius access no broader than this exact, reviewed implementation.
ARWEN_SOURCE_MANIFEST: Mapping[str, str] = MappingProxyType(
    {
        "woof/core/mpas_column_batch.py": (
            "f17c08bde1023fea39c3452fa88e7a19ada3ee9a56b47028c2103e3adab86125"
        ),
        "woof/core/physics.py": (
            "1550a56a13b31172e43874586b84194e994359bbb3ea72fe034d94cd0b03b9fe"
        ),
        "woof/core/microphysics.py": (
            "388f2018fa52e3b0d00523d55264221d239ad1251185fcb1f0d03f27cc2abb29"
        ),
        "woof/core/gf.py": (
            "f4dd01070dfb8cf60760bc6190d3382e7bb47e721f00f12db60aa7192de014c4"
        ),
        "woof/core/kernels/gf.cu": (
            "120dc7e606dae18cece64e1d1af899ec8930de339e82943dcab3457c9115532d"
        ),
        "woof/core/rrtmg_legacy.py": (
            "8026dfe218e405c36f9ada1bc4713d97d0c19ee4966e87a9d7d7b1b9d9a4b1d3"
        ),
        "woof/core/noahmp_runtime.py": (
            "a25f24c8acc070908a83c1fbc75b4e61727b676962b86c90ed8816af6365cc97"
        ),
        "woof/core/noahmp_glacier.py": (
            "5b94145bc7122a2f6dd59ada85260cde9ae5fe2c038b16d65fe75328b795ae88"
        ),
        "woof/core/noahmp_glacier_gpu.py": (
            "81fe20b1fda560bc220c4a880639ee7fa1d5129a2fef1869cbb7c65e6a4057d0"
        ),
        "woof/core/noahmp_kernel_sources.py": (
            "6aff082a02f4ed5883e6c7086c712e30324d79fe8b2dbe9a16665051a564b708"
        ),
        "woof/core/kernels/__init__.py": (
            "cd5533e48c6b7c898aa49da76f480561c0983b1e616e5709d4f036de4a734d50"
        ),
        "woof/core/kernels/noahmp_leaves.cu": (
            "4a3b6ce94993ab1c7055de39880e1209b54527751dcda07f26aa4b4a612f3497"
        ),
        "woof/core/kernels/noahmp_glacier.cu": (
            "14c6e57f2a2df7926a05101c89f2a6778b150ce68af58c78996e288bda8cd954"
        ),
        "woof/config.py": (
            "66bd34317bb62c0a9336a148a3d5819f666d779b291288ddccbb65e7b061fccc"
        ),
        "woof/io/restart.py": (
            "90c820a8cde4450d88371b50cc05791f265b3a6028c4de90f790b13283024ef4"
        ),
        "docs/mpas-seam.md": (
            "4b734d8ebdf0686fe6bb40dd699c56dce6ae3460d32969c9056a1846b48f739e"
        ),
    }
)


_GLACIER_CUDA_PROVENANCE = (
    "noahmp-glacier/cuda (woof/core/kernels/noahmp_glacier.cu)"
)
_ISICE_TABLE = 15

_LIMITATIONS = (
    "fa35 exposes one phase-one pressure family: MPAS supplies moist-hydrostatic "
    "pres_hyd_p/pres2_hyd_p to all WOOF phase-one consumers; cloud fraction "
    "cannot independently receive EOS pressure through the published seam",
    "fa35 legacy RRTMG rebuilds t8w from the constructor's nominal one-dimensional "
    "vertical weights; frozen MPAS t2_p is not a published phase-one argument",
    "legacy RRTMG stages columns through the host on radiation-due calls; this "
    "occurs at the WOOF-owned radiation cadence rather than every model step",
    "fa35 Noah-MP consumes the common published phase-one atmosphere; the separate "
    "raw-qv MPAS Noah-MP sounding is prepared and validated but is not accepted by "
    "the published WOOF call signature",
    "fa35 GF convective momentum tendencies are not coupled; native MPAS v8.4.1 "
    "does not couple them either (its cu_grell_freitas call carries no "
    "rucuten/rvcuten), so this is native parity rather than a gap",
    "fa35 accepts one p_top_pa scalar derived by the runner as an area-weighted "
    "value, whereas native MPAS plrad is per-column; this adapter therefore makes "
    "no source-matched/native-parity claim for that pressure boundary",
)

_REQUIRED_ARWEN_EXPORT_FIELDS = ("tsk", "smois", "tslb", "hfx", "qfx", "lh")
_OPTIONAL_ARWEN_EXPORT_FIELDS = (
    "t2", "q2", "pblh", "u10", "v10", "psfc",
    "swdown", "glw", "olr",
    # The land surface's own sea ice and snow state, so a history frame can
    # be a regional model's initial and boundary surface (the ArWen parent
    # route, gap 9 item 3), not only a picture.
    "xice", "snow", "snowh",
)
_SOIL_DIAGNOSTIC_FIELDS = frozenset(("smois", "tslb"))
_GWDO_SURFACE_FIELDS = ("dusfcg", "dvsfcg")
_GWDO_LEVEL_FIELDS = (
    "dtaux3d",
    "dtauy3d",
    "rubldiff",
    "rvbldiff",
)
_GWDO_DIAGNOSTIC_FIELDS = (*_GWDO_SURFACE_FIELDS, *_GWDO_LEVEL_FIELDS)

_CONSTRUCTOR_ARRAY_FIELDS = (
    "latitude_deg",
    "longitude_deg",
    "terrain_height_m",
    "z_interface_nominal_m",
    "landmask",
    "xland",
    "ivgtyp",
    "isltyp",
    "vegfra",
    "tsk",
    "tmn",
    "xice",
    "snow",
    "snow_depth",
    "soil_temperature",
    "soil_moisture",
    # Native MPAS v8.4.1 hands GF a per-cell dx built from the mesh
    # (mpas_atmphys_driver_convection.F:718, len_disp/meshDensity**0.25);
    # one scalar dx is a lie on a variable-resolution mesh, so this is a
    # required sealed static, not an option.
    "dx_column_m",
)
_CONSTRUCTOR_KEYS = frozenset(
    (
        "n_levels",
        "n_columns",
        "dt",
        "radiation_seconds",
        "surface_pbl_seconds",
        "cumulus_seconds",
        "cumulus_scheme",
        # The seam's ``microphysics_scheme`` row, named from the species
        # row's ``engine_scheme`` and never defaulted: the engine constructs
        # a WSM6 seam when the key is absent, so a P3 or mp=28 door that
        # left it out would seal a seam that refuses its own phase-one call.
        "microphysics_scheme",
        "start_time",
        "p_top_pa",
        "dx_m",
        "gf_ishallow",
        "wsm6_hail_opt",
        "xice_threshold",
        *_CONSTRUCTOR_ARRAY_FIELDS,
    )
)
_SURFACE_FLOAT_FIELDS = (
    "latitude_deg",
    "longitude_deg",
    "terrain_height_m",
    "landmask",
    "xland",
    "vegfra",
    "tsk",
    "tmn",
    "xice",
    "snow",
    "snow_depth",
)
_SURFACE_INT_FIELDS = ("ivgtyp", "isltyp")
# Noah-MP statics the frozen engine constructor has no keyword for.
#
# THE DEFECT THIS CARRIES (obs scorecard, 2026-10-12).  The row's land surface
# is Noah-MP at dveg = 4, whose vegetation fraction is FVEG = SHDMAX / 100
# (module_sf_noahmplsm.F:863; engine woof/core/noahmp_sflx.py:741 and
# noahmp_runtime.py:2060).  ``MpasColumnBatchPhysics.__init__`` hands
# ``initialize_physics`` no ``shdmax``, so the driver's launch-field fill
# (woof/core/physics.py, ``"shdmax": 90.0``) became the vegetation fraction
# of EVERY vegetated land column: 0.90 over shrubland, grassland, tundra and
# cropland alike, where the init's own ``shdmax`` has a land median near 49 %.
# That is an Arwen constructor default reaching a real forecast, which this
# module's docstring says cannot happen.  Native MPAS hands its Noah-MP the
# static ``shdmax``.  The sealed mapping therefore carries the init's
# ``shdmax`` (percent) and ``_bind_noahmp_statics`` on the backend writes it
# into the driver's own carrier right after the seam is constructed and
# before its first step.
#
# OPTIONAL in the sealed mapping only so the anchored x4 proof driver, whose
# receipts are keyed to the bytes it already measured, keeps sealing the
# mapping it always sealed; the forecast door always supplies it, and a
# mapping without it says so in its receipt instead of claiming no default.
_NOAHMP_STATIC_FIELDS = ("shdmax",)
NOAHMP_ENGINE_DEFAULT_SHDMAX_PERCENT = 90.0
_SOIL_FIELDS = ("soil_temperature", "soil_moisture")
_CONSTRUCTOR_SEAL = object()


def _digest_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json_digest(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _array_identity(array: np.ndarray) -> Mapping[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "sha256": sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _positive_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _unit_interval(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be a finite fraction in [0, 1]")
    return result


def _cadence_steps(name: str, seconds: float, dt: float) -> int:
    """Mirror frozen Arwen v2's pure-host exact integer cadence refusal."""

    ratio = seconds / dt
    rounded = int(round(ratio))
    if rounded < 1 or abs(ratio - rounded) > 1.0e-9 * max(ratio, 1.0):
        raise ValueError(
            f"{name}={seconds} s is not a positive integer multiple of "
            f"dt={dt} s"
        )
    return rounded


def _host_exact_array(
    mapping: Mapping[str, Any], name: str, *, dtype: np.dtype, shape: tuple[int, ...]
) -> np.ndarray:
    value = mapping[name]
    if hasattr(value, "__cuda_array_interface__"):
        raise TypeError(f"constructor {name} must be an official host array")
    array = np.asarray(value)
    if array.dtype != np.dtype(dtype):
        raise TypeError(f"constructor {name} must be {np.dtype(dtype)}, got {array.dtype}")
    if array.shape != shape:
        raise ValueError(f"constructor {name} must have shape {shape}, got {array.shape}")
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ValueError(f"constructor {name} contains non-finite values")
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


def _locate_degenerate_columns(prepared: Any) -> str:
    """Name the columns a column scheme cannot integrate, and why.

    Every input the seam is handed is finite -- the preparation refuses
    otherwise -- so when a scheme's own arithmetic produces a non-finite
    tendency, the cause is a column that is finite and DEGENERATE.  The two
    that matter to a boundary-layer scheme are a column with no wind shear
    anywhere, because the bulk Richardson number divides by it, and a column
    with no stratification.  Both are reported with column indices, so the
    reader is told which cells rather than which array.
    """

    import numpy as _np

    cp = __import__("cupy")
    findings: list[str] = []
    u = cp.asnumpy(prepared.u_p)
    v = cp.asnumpy(prepared.v_p)
    theta = cp.asnumpy(prepared.th_p)
    speed = _np.hypot(u, v)
    still = _np.argwhere(speed.max(axis=0) == _np.float32(0.0)).ravel()
    if still.size:
        findings.append(
            f"{still.size} column(s) carry zero wind at every level "
            f"(first {int(still[0])} of {speed.shape[1]})"
        )
    shear = _np.abs(_np.diff(u, axis=0)) + _np.abs(_np.diff(v, axis=0))
    flat = _np.setdiff1d(
        _np.argwhere(shear.max(axis=0) == _np.float32(0.0)).ravel(), still
    )
    if flat.size:
        findings.append(
            f"{flat.size} further column(s) carry a wind that does not change "
            f"with height (first {int(flat[0])})"
        )
    isothermal = _np.argwhere(
        _np.abs(_np.diff(theta, axis=0)).max(axis=0) == _np.float32(0.0)
    ).ravel()
    if isothermal.size:
        findings.append(
            f"{isothermal.size} column(s) carry a constant potential "
            f"temperature (first {int(isothermal[0])})"
        )
    if not findings:
        return (
            "No column is degenerate in wind shear or stratification, so the "
            "cause is inside the scheme's own carried state rather than this "
            "step's sounding."
        )
    return "Degenerate columns handed to it: " + "; ".join(findings) + "."


@dataclass(frozen=True, slots=True)
class SealedArwenConstructorV841:
    """Complete, immutable exact-real constructor mapping for frozen Arwen v2."""

    _values: Mapping[str, Any] = field(repr=False, compare=False)
    identity_sha256: str
    host_array_bytes: int
    _seal: object = field(repr=False, compare=False)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SealedArwenConstructorV841":
        if not isinstance(values, Mapping):
            raise TypeError("WOOF constructor values must be a mapping")
        from . import correct_mode

        missing = sorted(_CONSTRUCTOR_KEYS - set(values))
        extra = sorted(
            set(values) - _CONSTRUCTOR_KEYS - set(_NOAHMP_STATIC_FIELDS)
            - correct_mode.OPTIONAL_CONSTRUCTOR_KEYS - {"pbl_scheme"}
        )
        if missing or extra:
            raise ValueError(
                "exact-real WOOF constructor mapping is not exhaustive: "
                f"missing={missing}, extra={extra}"
            )
        nlev = values["n_levels"]
        ncol = values["n_columns"]
        if isinstance(nlev, bool) or not isinstance(nlev, (int, np.integer)) or int(nlev) < 3:
            raise ValueError("n_levels must be an integer >= 3")
        if isinstance(ncol, bool) or not isinstance(ncol, (int, np.integer)) or int(ncol) < 1:
            raise ValueError("n_columns must be a positive integer")
        nlev, ncol = int(nlev), int(ncol)
        if not isinstance(values["start_time"], datetime):
            raise TypeError("start_time must be a UTC datetime")
        scheme = values["cumulus_scheme"]
        if scheme not in ("gf", "kf", "ntiedtke", None):
            raise ValueError(
                "cumulus_scheme must be 'gf', 'kf', 'ntiedtke', or None"
            )
        if scheme is None and values["cumulus_seconds"] is not None:
            raise ValueError("cumulus_seconds requires a cumulus_scheme")
        if scheme is not None:
            _positive_real("cumulus_seconds", values["cumulus_seconds"])
        dt = _positive_real("dt", values["dt"])
        radiation_seconds = _positive_real(
            "radiation_seconds", values["radiation_seconds"]
        )
        surface_pbl_seconds = _positive_real(
            "surface_pbl_seconds", values["surface_pbl_seconds"]
        )
        _cadence_steps("radiation_seconds", radiation_seconds, dt)
        _cadence_steps("surface_pbl_seconds", surface_pbl_seconds, dt)
        cumulus_seconds = None
        if values["cumulus_seconds"] is not None:
            cumulus_seconds = _positive_real(
                "cumulus_seconds", values["cumulus_seconds"]
            )
            cumulus_steps = _cadence_steps(
                "cumulus_seconds", cumulus_seconds, dt
            )
            if scheme == "gf" and cumulus_steps != 1:
                raise ValueError(
                    "cumulus_scheme='gf' requires cumulus_seconds == dt"
                )
            # New Tiedtke is an every-step scheme for the same reason: WRF
            # pins cudt=0 for cu_physics=16 and its RAINCV is a per-call
            # rate with no hold (woof/config.py validate_run_config).
            if scheme == "ntiedtke" and cumulus_steps != 1:
                raise ValueError(
                    "cumulus_scheme='ntiedtke' requires cumulus_seconds == dt"
                )
        hail_opt = values["wsm6_hail_opt"]
        if (
            isinstance(hail_opt, bool)
            or not isinstance(hail_opt, (int, np.integer))
            or int(hail_opt) not in (0, 1)
        ):
            raise ValueError("wsm6_hail_opt must be the integer 0 or 1")
        microphysics_scheme = values["microphysics_scheme"]
        engine_schemes = tuple(
            row.engine_scheme
            for row in registered_species_rows().values()
            if row.engine_scheme is not None
        )
        if (
            not isinstance(microphysics_scheme, str)
            or microphysics_scheme not in engine_schemes
        ):
            raise ValueError(
                "microphysics_scheme must be one of the engine rows the "
                f"species table declares {sorted(engine_schemes)}, got "
                f"{microphysics_scheme!r}"
            )
        if microphysics_scheme != "wsm6" and int(hail_opt) != 0:
            # The seam refuses this pair itself; refusing it here keeps the
            # refusal ahead of every device allocation, like the rest.
            raise ValueError(
                "wsm6_hail_opt is a WSM6 hail-mode knob; on a "
                f"microphysics_scheme={microphysics_scheme!r} seam it must "
                "be 0"
            )
        # GF shallow convection.  Native MPAS v8.4.1 hardwires ishallow = 1
        # (mpas_atmphys_vars.F:340); shallow OFF is reachable only as an
        # explicit A/B arm and is meaningless without GF.
        ishallow = values["gf_ishallow"]
        if (
            isinstance(ishallow, bool)
            or not isinstance(ishallow, (int, np.integer))
            or int(ishallow) not in (0, 1)
        ):
            raise ValueError("gf_ishallow must be the integer 0 or 1")
        if int(ishallow) == 1 and scheme != "gf":
            raise ValueError("gf_ishallow=1 requires cumulus_scheme='gf'")
        scalars = {
            "n_levels": nlev,
            "n_columns": ncol,
            "dt": dt,
            "radiation_seconds": radiation_seconds,
            "surface_pbl_seconds": surface_pbl_seconds,
            "cumulus_seconds": cumulus_seconds,
            "cumulus_scheme": scheme,
            "microphysics_scheme": microphysics_scheme,
            "start_time": values["start_time"],
            "p_top_pa": _positive_real("p_top_pa", values["p_top_pa"]),
            "dx_m": _positive_real("dx_m", values["dx_m"]),
            "gf_ishallow": int(ishallow),
            "wsm6_hail_opt": int(hail_opt),
            "xice_threshold": _unit_interval(
                "xice_threshold", values["xice_threshold"]
            ),
            # physics.mode (woof.hex.correct_mode): stored only in correct
            # mode, so an exact mapping seals the identity it always did.
            **correct_mode.sealed_scalars(values),
        }
        if "pbl_scheme" in values:
            if values["pbl_scheme"] not in (None, "ysu"):
                raise ValueError("pbl_scheme must be ysu or None")
            scalars["pbl_scheme"] = values["pbl_scheme"]
        arrays: dict[str, np.ndarray] = {}
        for name in _SURFACE_FLOAT_FIELDS:
            arrays[name] = _host_exact_array(
                values, name, dtype=np.dtype(np.float32), shape=(ncol,)
            )
        for name in _SURFACE_INT_FIELDS:
            arrays[name] = _host_exact_array(
                values, name, dtype=np.dtype(np.int32), shape=(ncol,)
            )
        for name in _SOIL_FIELDS:
            arrays[name] = _host_exact_array(
                values, name, dtype=np.dtype(np.float32), shape=(4, ncol)
            )
        # Per-column GF dx.  Native builds it per cell; the sealed mapping
        # therefore carries the whole vector, positive and finite.
        dx_column = _host_exact_array(
            values, "dx_column_m", dtype=np.dtype(np.float32), shape=(ncol,)
        )
        if not np.all(np.isfinite(dx_column)) or np.any(dx_column <= np.float32(0.0)):
            raise ValueError("dx_column_m must be finite and positive")
        arrays["dx_column_m"] = dx_column
        if "shdmax" in values:
            # Annual-maximum green vegetation fraction, PERCENT, exactly as
            # the init carries it.  A 0..1 fraction handed here would leave
            # every column under 1 % vegetated without an error, so the range
            # is checked against the unit it must be in.
            shdmax = _host_exact_array(
                values, "shdmax", dtype=np.dtype(np.float32), shape=(ncol,)
            )
            if (
                not np.all(np.isfinite(shdmax))
                or np.any(shdmax < np.float32(0.0))
                or np.any(shdmax > np.float32(100.0))
            ):
                raise ValueError(
                    "shdmax must be finite and in percent [0, 100]: Noah-MP "
                    "reads it as FVEG = SHDMAX / 100"
                )
            if np.float32(0.0) < shdmax.max() <= np.float32(1.0) and ncol > 64:
                raise ValueError(
                    "shdmax never exceeds 1.0: it looks like a 0..1 fraction, "
                    "and Noah-MP divides it by 100, which would leave every "
                    "land column under 1 % vegetated"
                )
            arrays["shdmax"] = shdmax
        z = np.asarray(values["z_interface_nominal_m"])
        if z.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("z_interface_nominal_m must be host float32 or float64")
        if z.shape != (nlev + 1,) or not np.all(np.isfinite(z)) or np.any(np.diff(z) <= 0):
            raise ValueError(
                "z_interface_nominal_m must be finite, strictly increasing, and "
                f"have shape {(nlev + 1,)}"
            )
        arrays["z_interface_nominal_m"] = np.array(z, copy=True, order="C")
        arrays["z_interface_nominal_m"].setflags(write=False)
        sealed: dict[str, Any] = {**scalars, **arrays}
        identity = {
            name: (
                _array_identity(value)
                if isinstance(value, np.ndarray)
                else value.isoformat()
                if isinstance(value, datetime)
                else value
            )
            for name, value in sorted(sealed.items())
        }
        return cls(
            _values=MappingProxyType(sealed),
            identity_sha256=_json_digest(identity),
            host_array_bytes=sum(int(value.nbytes) for value in arrays.values()),
            _seal=_CONSTRUCTOR_SEAL,
        )

    @property
    def n_levels(self) -> int:
        return int(self._values["n_levels"])

    @property
    def n_columns(self) -> int:
        return int(self._values["n_columns"])

    @property
    def dt(self) -> float:
        return float(self._values["dt"])

    @property
    def xice_threshold(self) -> float:
        return float(self._values["xice_threshold"])

    @property
    def microphysics_scheme(self) -> str:
        return str(self._values["microphysics_scheme"])

    @property
    def cumulus_scheme(self) -> str | None:
        scheme = self._values["cumulus_scheme"]
        return None if scheme is None else str(scheme)

    @property
    def physics_mode(self) -> str:
        """``physics.mode`` as sealed: ``"exact"`` unless correct was asked."""
        return str(self._values.get("physics_mode", "exact"))

    @property
    def physics_correct_skip(self) -> tuple[str, ...]:
        return tuple(self._values.get("physics_correct_skip", ()))

    def expected_surface_classification(self) -> Mapping[str, Any]:
        """Authority-only host classification for the sealed constructor."""

        xland = np.asarray(self._values["xland"], dtype=np.float32)
        xice = np.asarray(self._values["xice"], dtype=np.float32)
        ivgtyp = np.asarray(self._values["ivgtyp"], dtype=np.int32)
        threshold = np.float32(self.xice_threshold)
        sea_ice = xice >= threshold
        open_water = (xland >= np.float32(1.5)) & ~sea_ice
        land = ~(sea_ice | open_water)
        glacier = land & (ivgtyp == np.int32(_ISICE_TABLE))
        return MappingProxyType(
            {
                "xland_source": "native",
                "xland_land_columns": int(np.count_nonzero(xland < np.float32(1.5))),
                "xland_water_columns": int(np.count_nonzero(xland >= np.float32(1.5))),
                "xice_threshold": self.xice_threshold,
                "sea_ice_columns": int(np.count_nonzero(sea_ice)),
                "open_water_columns": int(np.count_nonzero(open_water)),
                "sflx_land_columns": int(np.count_nonzero(land & ~glacier)),
                "glacier_columns": int(np.count_nonzero(glacier)),
            }
        )

    def arwen_kwargs(self) -> dict[str, Any]:
        if self._seal is not _CONSTRUCTOR_SEAL:
            raise TypeError("WOOF constructor mapping is not sealed")
        # Arrays remain the sealed read-only objects.  Arwen copies/uploads
        # them during construction and never receives an adapter-owned mutable
        # static carrier.  The Noah-MP statics are not constructor keywords
        # of the frozen engine class; they are bound after construction.
        return {
            name: value
            for name, value in self._values.items()
            if name not in _NOAHMP_STATIC_FIELDS
        }

    def noahmp_statics(self) -> Mapping[str, np.ndarray]:
        """The sealed Noah-MP statics to bind after construction (maybe empty)."""

        if self._seal is not _CONSTRUCTOR_SEAL:
            raise TypeError("WOOF constructor mapping is not sealed")
        return MappingProxyType(
            {
                name: self._values[name]
                for name in _NOAHMP_STATIC_FIELDS
                if name in self._values
            }
        )

    def receipt(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "identity_sha256": self.identity_sha256,
                "n_levels": self.n_levels,
                "n_columns": self.n_columns,
                "dt": self.dt,
                "microphysics_scheme": self.microphysics_scheme,
                "host_array_bytes": self.host_array_bytes,
                "defaults_used": False,
                "noahmp_vegetation_fraction": (
                    "init shdmax (percent), bound to the driver carrier"
                    if "shdmax" in self._values
                    else "engine launch-field fill, "
                    f"{NOAHMP_ENGINE_DEFAULT_SHDMAX_PERCENT:g} % on every "
                    "column: this mapping carries no shdmax"
                ),
                "surface_soil_statics": "official-exhaustive-sealed-host-mapping",
                "xland_source": "native",
                "xice_threshold": self.xice_threshold,
                "expected_surface_classification": dict(
                    self.expected_surface_classification()
                ),
                # Present only in correct mode (MODE-KEY.md rule 3).
                **(
                    {"physics_mode": self.physics_mode,
                     "physics_correct_skip": list(self.physics_correct_skip)}
                    if self.physics_mode != "exact" else {}
                ),
            }
        )


_ADAPTER_AUTHORITY = {
    "schema": CUDA_ARWEN_PHYSICS_V841_SCHEMA,
    "engine_commit": ARWEN_BUILD_COMMIT,
    "engine_sources": dict(ARWEN_SOURCE_MANIFEST),
    "contract_document_sha256": MPAS_SEAM_CONTRACT_SHA256,
    "contract_surface_sha256": MPAS_SEAM_CONTRACT_SURFACE_SHA256,
    "glacier_composed_tu_sha256": ARWEN_GLACIER_COMPOSED_TU_SHA256,
    "prep_contract_sha256": CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256,
    "gwdo_contract_sha256": CUDA_GWDO_V841_CONTRACT_SHA256,
    "coupling_contract_sha256": CUDA_PHYSICS_V841_CONTRACT_SHA256,
    "theta": "frozen prep th_p dry theta",
    "phase1_pressure": "pres_hyd_p/pres2_hyd_p with explicit pi_p",
    "phase2_pressure": "pres_p EOS with rho_dry and z_p",
    "h_diabatic": "explicitly declined; never replayed or folded",
    "lagged_heating": (
        "SG2: MPAS rt_diabatic_tend produced here (modified-theta change across "
        "phase two / step dt, woof.hex.lagged_heating_v841), carried to the next "
        "step's dynamics and carried in restart state"
    ),
    "constructor": "exhaustive sealed exact-real mapping including native xland, explicit xice_threshold and the row's microphysics_scheme; no defaults",
    "pin_order": "pin exact WOOF v2 before MPAS KernelCache construction",
    "execution_provenance": (
        "typed aggregate-executed carrier; actual GWDO result identity and 4B gate"
    ),
    "publication": (
        "begin -> finished_unpublished -> explicit commit; abort restores seam/scalars"
    ),
    "diagnostics": (
        "committed-boundary public export plus retained six-field resident GWDO copy"
    ),
    "restart": "v2 threshold/xland-source identity plus adapter-owned committed gwdo_calls counter",
    "limitations": list(_LIMITATIONS),
}
CUDA_ARWEN_PHYSICS_V841_CONTRACT_SHA256 = _json_digest(_ADAPTER_AUTHORITY)

#: The contract ids WOOF 1.0.0 wrote into every backend receipt and restart
#: payload: the v2 adapter schema and digest, and the coupling contract's
#: digest before its texts said WOOF.  The adapter contract itself moved
#: with the hex model's 0.4.0 engine seam, so these ids only NAME a 1.0.0
#: receipt; a 1.0.0 restart payload is not this contract and is refused.
#: Written by the assembly from the digests 1.0.0 published, not typed.
WOOF_1_0_0_CONTRACT_IDS: Mapping[str, str] = MappingProxyType({
    "schema": "mpas-port.cuda-arwen-physics-v841/v2",
    "adapter_contract_sha256": (
        "f82124c1dea5b46d9cfb497f3bbe004c5561081685176ce81f6458e3d6d67fff"
    ),
    "coupling_contract_sha256": (
        "332899f64a24b45fc93a686ce056a674a4bd07a26d635447d0aff884ce13d001"
    ),
})


def contract_id_status(receipt: Mapping[str, Any]) -> str | None:
    """Whose contract ids a backend receipt or restart payload carries.

    ``"current"`` for this build's, ``"woof-1.0.0"`` for the ids WOOF
    1.0.0 wrote for the same contracts, None for anything else.  Reads
    ``schema`` and ``adapter_contract_sha256`` (a restart payload keeps the
    digest in its ``identity``) and, when present,
    ``coupling_contract_sha256``."""

    if not isinstance(receipt, Mapping):
        return None
    identity = receipt.get("identity")
    adapter = receipt.get("adapter_contract_sha256")
    if adapter is None and isinstance(identity, Mapping):
        adapter = identity.get("adapter_contract_sha256")
    coupling = receipt.get("coupling_contract_sha256")
    for status, ids in (
        ("current", {
            "schema": CUDA_ARWEN_PHYSICS_V841_SCHEMA,
            "adapter_contract_sha256": CUDA_ARWEN_PHYSICS_V841_CONTRACT_SHA256,
            "coupling_contract_sha256": CUDA_PHYSICS_V841_CONTRACT_SHA256,
        }),
        ("woof-1.0.0", WOOF_1_0_0_CONTRACT_IDS),
    ):
        if (receipt.get("schema") == ids["schema"]
                and adapter == ids["adapter_contract_sha256"]
                and coupling in (None, ids["coupling_contract_sha256"])):
            return status
    return None


def restart_identity_matches(
    schema: object, identity: object, expected: Mapping[str, Any]
) -> bool:
    """A restart payload's identity against the running adapter's.

    A payload WOOF 1.0.0 wrote carries the v2 schema and adapter digest for
    the same contract; every other field must still be equal, the engine's
    seam bytes among them."""

    if not isinstance(identity, Mapping):
        return False
    identity = dict(identity)
    if schema == WOOF_1_0_0_CONTRACT_IDS["schema"]:
        # 1.0.0 wrote it under the adapter contract before the 0.4.0 seam
        return False
    if schema != CUDA_ARWEN_PHYSICS_V841_SCHEMA:
        return False
    return identity == dict(expected)


def _verify_checkout_root(root: Path) -> engine_pin.EngineIdentity:
    """Say which engine ``root`` is: a measured row's, or the tree as found.

    The comparison is :func:`engine_pin.inspect_seam`, the same one the
    forecast door and ``doctor`` run, so the three surfaces cannot disagree
    about a tree.

    RETIRED 2026-10-10 as a gate (owner ruling; Refusals law): a moved seam file used to raise "pinned Arwen
    source hash mismatch", and a moved ``docs/mpas-seam.md`` "pinned Arwen
    contract surface hash mismatch".  Neither names a breakage -- a digest
    that differs says the engine is another build, not that it cannot run --
    and together they refused every forecast on an engine newer than the pin
    (8-card RTX PRO 6000 host, gpuwm 2.8.9 against the 2.8.7 pin).  The match is recorded
    (:func:`engine_pin.identity_of`; ``matches_measured_row`` in the pin
    mapping and every receipt) and nothing refuses on it.

    What IS still refused is what cannot run: the column-batch module the
    seam constructs is not in the tree at all.
    """

    inspection = engine_pin.inspect_seam(root)
    if _SEAM_MODULE_PATH in inspection.absent:
        raise FileNotFoundError(
            f"{root / _SEAM_MODULE_PATH} does not exist: the physics seam "
            "constructs woof's MPAS column batch from that module, so there "
            "is no physics to run in this tree.  Point --gpuwm-checkout at a "
            "woof source tree or leave it unset to use the installed engine."
        )
    return engine_pin.identity_of(inspection)


#: The one engine module the seam cannot run without.
_SEAM_MODULE_PATH = "woof/core/mpas_column_batch.py"


def _engine_correct_mode(constructor: Any) -> tuple[Mapping[str, Any] | None, bool]:
    """``(receipt, engine_takes_mode)``: the engine's correct-mode receipt
    (``None`` in exact mode) and whether the engine call may carry the mode
    keywords.

    An engine whose column batch takes no ``physics_mode`` cannot apply the
    corrections: the run goes ahead with the matched arithmetic and the
    receipt names each missing finding as a gap (woof.hex.correct_mode: a
    named gap, never a refusal).  Exact mode imports nothing here.
    """

    from . import correct_mode

    if getattr(constructor, "physics_mode", "exact") == "exact":
        return None, True
    seam = None
    try:
        import woof.core.mpas_column_batch as column_batch
    except ImportError:
        column_batch = None
    if column_batch is not None:
        seam = column_batch.MpasColumnBatchPhysics.__init__
    schemes = {str(getattr(constructor, "microphysics_scheme", "") or ""),
               str(getattr(constructor, "cumulus_scheme", "") or "")}
    resolved = correct_mode.engine_settings(
        constructor.physics_mode, constructor.physics_correct_skip,
        seam=seam, seam_label="MpasColumnBatchPhysics",
        active_schemes=schemes - {""})
    return MappingProxyType(dict(resolved["receipt"])), bool(resolved["engine_takes_mode"])


def _contract_surface_sha256(root: Path) -> str | None:
    """Digest of the seam class file and its contract document as found."""

    surface = sha256()
    for relative in (_SEAM_MODULE_PATH, "docs/mpas-seam.md"):
        try:
            surface.update((root / relative).read_bytes())
        except OSError:
            return None
    return surface.hexdigest()


def _load_pinned_arwen_factory(
    checkout: str | Path | None,
) -> tuple[Callable[..., Any], Path, engine_pin.EngineIdentity]:
    requested: Path | None = None
    if checkout is not None:
        requested = Path(checkout).resolve()
        _verify_checkout_root(requested)
        already = sys.modules.get("woof")
        if already is not None:
            loaded = Path(inspect.getfile(already)).resolve().parent.parent
            if loaded != requested:
                raise RuntimeError(
                    f"woof is already imported from {loaded} and this run "
                    f"names {requested}: a live package cannot be replaced, "
                    "so the run would execute the first tree's physics while "
                    "its receipt named the second"
                )
        elif not any(isinstance(entry, str) and Path(entry or ".").resolve() == requested.resolve()
                     for entry in sys.path):
            sys.path.insert(0, str(requested))

    import woof.core.microphysics as microphysics
    import woof.core.mpas_column_batch as column_batch
    import woof.core.noahmp_kernel_sources as noahmp_kernel_sources
    import woof.core.physics as physics

    root = Path(inspect.getfile(column_batch)).resolve().parents[2]
    if requested is not None and root != requested:
        raise RuntimeError(
            f"this run names the engine tree {requested} and Python imported "
            f"woof from {root}: the run would execute the second tree's "
            "physics while its receipt named the first"
        )
    identity = _verify_checkout_root(root)
    factory = physics.run_mpas_column_batch
    if factory is not column_batch.run_mpas_column_batch:
        raise ValueError(
            "woof.core.physics.run_mpas_column_batch is not "
            "woof.core.mpas_column_batch.run_mpas_column_batch in this "
            "engine: the seam would construct a column batch other than the "
            "class whose state layout this adapter reads and writes"
        )
    if column_batch.MpasColumnBatchPhysics._PHASE1_ORCHESTRATION is not physics.PhysicsDriver.compute:
        raise ValueError(
            "this engine's column batch no longer runs PhysicsDriver.compute "
            "as phase one: the adapter couples phase-one tendencies in that "
            "call's order and would couple another routine's output as if "
            "it were"
        )
    if column_batch.MpasColumnBatchPhysics._PHASE2_MICROPHYSICS is not microphysics.apply:
        raise ValueError(
            "this engine's column batch no longer runs microphysics.apply as "
            "phase two: the adapter hands phase two the post-dynamics state "
            "that routine expects and would hand it to another routine"
        )
    composed = noahmp_kernel_sources.translation_unit_source("noahmp_glacier")
    composed_sha256 = sha256(composed.encode("ascii")).hexdigest()
    if not identity.matches_measured_row:
        # Recorded as found: the glacier translation unit of an engine no
        # measured row names is whatever that engine composes.
        identity = replace(identity, glacier_composed_tu_sha256=composed_sha256)
    elif composed_sha256 != identity.glacier_composed_tu_sha256:
        # The sixteen bytes are a measured row's and the composed unit is
        # not: a file outside the sixteen moved.  Recorded as found, never
        # refused (same retirement as _verify_checkout_root).
        identity = replace(
            identity,
            glacier_composed_tu_sha256=composed_sha256,
            matches_measured_row=False,
        )
    return factory, root, identity


def pin_arwen_physics_v841(checkout: str | Path) -> Mapping[str, Any]:
    """Pin exact Arwen v2 before KernelCache imports any other woof tree.

    The mapping names the engine the tree WAS MEASURED TO BE: the public
    pin's commit, manifest and glacier digest when the bytes are the pin's,
    an admitted row's when they are that row's.  ``engine_version`` and
    ``admitted`` say which, so a receipt can be read without re-deriving it.
    """

    factory, root, identity = _load_pinned_arwen_factory(checkout)
    return MappingProxyType(
        {
            "arwen_commit": identity.build_commit,
            "root": str(root),
            "source_manifest": dict(identity.manifest),
            "contract_surface_sha256": MPAS_SEAM_CONTRACT_SURFACE_SHA256,
            "contract_surface_found_sha256": _contract_surface_sha256(root),
            "glacier_composed_tu_sha256": identity.glacier_composed_tu_sha256,
            "engine_version": identity.version,
            "admitted": identity.admitted,
            # Evidence, never a permission (engine_pin.identity_of).
            "matches_measured_row": identity.matches_measured_row,
            "pinned_engine_version": engine_pin.wanted_version(),
            "moved_from_pin": list(identity.moved_from_pin),
            "absent_from_pin": list(identity.absent_from_pin),
            "factory_module": factory.__module__,
            "factory_name": factory.__name__,
            "must_precede": "MPAS KernelCache construction",
        }
    )


#: What an open transaction holds in place of the boundary export when the
#: backend was built with ``rollback_snapshot=False``.  Not ``None``: ``None``
#: means "no transaction is open", and every phase guard below reads it so.
_ROLLBACK_NOT_ARMED: Mapping[str, Any] = MappingProxyType({"rollback": "not armed"})

#: The receipt text an unarmed failure leaves, so the refusal a user reads
#: says why no boundary came back.
_ROLLBACK_NOT_ARMED_TEXT = (
    "no boundary export was taken for this step (rollback_snapshot=False), so "
    "the seam holds the refused step's partial state and is retired; nothing "
    "was published and every later step refuses"
)


def _snapshot_nbytes(value: Any) -> int:
    if isinstance(value, np.ndarray) or hasattr(value, "__cuda_array_interface__"):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(_snapshot_nbytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_snapshot_nbytes(item) for item in value)
    return 0


def _scalar_names(names: Sequence[str]) -> tuple[str, ...]:
    """The block must be a DECLARED row, in that row's order.

    This asserted WSM6's exact six at four call sites.  It now asserts the
    ROW's order -- still an exact order, still refusing a block nobody
    declared, but the row it checks against is the one the run selected.
    """

    result = tuple(str(name).strip().lower() for name in names)
    species_row_for_names(result)
    return result


@dataclass(frozen=True, slots=True)
class CudaArwenDiagnosticSnapshotV841:
    """Detached resident diagnostics from one committed frozen Arwen v2 boundary."""

    surface: Mapping[str, Any]
    soil: Mapping[str, Any]
    precipitation: Mapping[str, Any]
    gwdo: Mapping[str, Any]
    metadata: Mapping[str, Any]
    receipt: Mapping[str, Any]


def history_cumulus_rain_rate(owner: Any, seam: Any, rainc: Any, cp: Any) -> dict[str, Any]:
    """``{"cuprec": mm/s}`` when the run has a cumulus scheme, the name MPAS-A's history gives it; else nothing.

    The scheme's mean rate since the last snapshot, from its accumulated rain: GF and New Tiedtke add an increment to
    ``rainc`` and leave no rate on the driver, so this one rule serves every scheme. ``rainc`` is the snapshot's own
    detached copy, held for the next one; the card does one subtraction per history frame and nothing per step.

    THE BREAKAGE THIS PREVENTS (WOOF physics, GFS 2026-10-31 06Z, f006, 120 km global and 96 -> 3 km zoom globe): the
    history carried only ``rainc``, so rw_mpas_convert had no rate to add to REFC and the reflectivity map showed none
    of the tropical rain bands the run's own rain map showed. A run with no cumulus scheme has no accumulator on its
    driver: it writes no field and its frame keeps drawing its volume.
    """

    if getattr(getattr(seam, "_driver", None), "rainc", None) is None:
        return {}
    seconds = float(seam.elapsed_seconds)
    held = getattr(owner, "_history_cuprec", None)
    if held is None or held[0] != seconds:
        rate = cp.zeros_like(rainc)
        if held is not None and seconds > held[0]:
            rate = cp.maximum(rainc - held[1], 0.0) / (seconds - held[0])
        held = owner._history_cuprec = (seconds, rainc, rate)
    return {"cuprec": cp.array(held[2], copy=True, order="C")}


#: A species or moment name of the engine's state (qv, qr, nr, nwfa, qir, bir, ...).
_SPECIES_NAME = __import__("re").compile(r"[qnb][a-z0-9]{1,4}")


class InitialReflectivityViewV841:
    """A between-steps column view of the INITIAL state for the engine's 10 cm reflectivity operator.

    The engine's operator (``woof.da.obsop.simulated_reflectivity``) reads a
    state shaped ``(nz, 1, ncol)``: the species by their engine names (hex's
    scalar names are those names), full theta as ``thb + thp`` and pressure
    ``p``, plus a persistent ``scratch`` slot.  This view holds exactly those,
    as read-only reshapes of the hex arrays, and its own scratch, so the
    seam's state is read for nothing and written by nothing.  The melting
    term is copied explicitly from the seam's own answer
    (``refl10cm_from_melting``), which ``woof.core.refl.refl_melting_for``
    honours ahead of the state's class.
    """

    def __init__(self, *, theta: Any, pressure: Any, species: Mapping[str, Any], melting: bool, cp: Any) -> None:
        nz, ncol = (int(n) for n in theta.shape)
        shape = (nz, 1, ncol)

        def column(name: str, value: Any) -> Any:
            if tuple(value.shape) != (nz, ncol):
                raise ValueError(f"initial reflectivity input {name} is {tuple(value.shape)}, not {(nz, ncol)}")
            return cp.ascontiguousarray(value, dtype=cp.float32).reshape(shape)

        self.thp = column("theta", theta)
        self.p = column("pressure", pressure)
        self.thb = cp.zeros((nz,), dtype=cp.float32)
        for name, value in species.items():
            setattr(self, name, column(name, value))
        self.refl10cm_from_melting = bool(melting)
        self._cp = cp
        self._scratch: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        # Only reached for names not set above.  A species the selection does
        # not carry is absent (the operator checks its own required inputs by
        # name), never a silent zero array.
        if _SPECIES_NAME.fullmatch(name):
            return None
        raise AttributeError(name)

    def scratch(self, shape: Any, slot: str, dtype: Any = None) -> Any:
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        buffer = self._scratch.get(slot)
        if buffer is None or tuple(buffer.shape) != shape:
            buffer = self._scratch[slot] = self._cp.zeros(shape, dtype=self._cp.float32 if dtype is None else dtype)
        return buffer


class PersistentTwoPhaseCudaPhysicsBackendV841:
    """Transactional frozen-v2 Arwen aggregate implementing the MPAS protocol."""

    def __init__(
        self,
        *,
        constructor: SealedArwenConstructorV841,
        prep_geometry: CudaMpasToPhysGeometryV841,
        kernel_cache: KernelCache,
        gwdo_static: CudaYsuGwdoStaticV841 | None = None,
        gwdo_kernel_cache: KernelCache | None = None,
        arwen_checkout: str | Path | None = None,
        rollback_snapshot: bool = True,
    ) -> None:
        if not isinstance(constructor, SealedArwenConstructorV841) or constructor._seal is not _CONSTRUCTOR_SEAL:
            raise TypeError("constructor must be SealedArwenConstructorV841.from_mapping")
        if not isinstance(rollback_snapshot, bool):
            raise TypeError("rollback_snapshot must be a bool")
        if not isinstance(prep_geometry, CudaMpasToPhysGeometryV841):
            raise TypeError("prep_geometry must be sealed v8.4.1 MPAS preparation geometry")
        prep_geometry.validate()
        if prep_geometry.n_cells != constructor.n_columns:
            raise ValueError("preparation geometry and WOOF constructor column counts differ")
        if gwdo_static is not None:
            if not isinstance(gwdo_static, CudaYsuGwdoStaticV841):
                raise TypeError("gwdo_static must be a sealed CudaYsuGwdoStaticV841")
            gwdo_static.validate()
            if gwdo_static.n_cells != constructor.n_columns:
                raise ValueError("GWDO statics and WOOF constructor column counts differ")
            if gwdo_kernel_cache is None:
                raise ValueError("GWDO activity requires gwdo_kernel_cache")
        elif gwdo_kernel_cache is not None:
            raise ValueError("gwdo_kernel_cache requires gwdo_static")
        factory, root, identity = _load_pinned_arwen_factory(arwen_checkout)
        self._correct_mode_receipt, self._engine_takes_mode = _engine_correct_mode(constructor)
        self._constructor = constructor
        self._engine = identity
        self._prep_geometry = prep_geometry
        self._kernel_cache = kernel_cache
        self._gwdo_static = gwdo_static
        self._gwdo_kernel_cache = gwdo_kernel_cache
        self._factory = factory
        self._arwen_root = root
        from .cuda_arwen_helper_scope_v2 import bind_arwen_helpers_v2
        self._helper_scope = bind_arwen_helpers_v2(root)
        self._rollback_snapshot = rollback_snapshot
        self._seam = self._new_seam()
        self._phase = "boundary"
        self._boundary_snapshot: Mapping[str, Any] | None = None
        self._step_start: float | None = None
        self._candidate_scalar_target: Any | None = None
        self._candidate_scalar_backup: Any | None = None
        self._pending_gwdo_result: CudaYsuGwdoResultV841 | None = None
        self._last_gwdo_result: CudaYsuGwdoResultV841 | None = None
        # The one-frame refl10cm handoff (WRF diagflag semantics): staged by
        # a due finish_step, published by commit_step, consumed exactly once
        # by the history capture.  Never restart state -- the field is
        # recomputed by the next due microphysics call.
        self._pending_refl10cm: Any | None = None
        self._committed_refl10cm: Any | None = None
        self._gwdo_calls = 0
        # SG2 -- MPAS-A's rt_diabatic_tend (woof.hex.lagged_heating_v841):
        # staged by finish_step, published by commit_step, read by the next
        # step's held tendencies.  None before the first microphysics call,
        # as native's zeroed tend pool.
        self._pending_lagged_diabatic_theta: Any | None = None
        self._lagged_diabatic_theta: Any | None = None
        self._lagged_heating_restored = None
        # True once a GF advective-forcing carrier has been consumed; a later
        # None then means the runner regressed to the zero lanes.
        self._gf_forcing_seen = False
        self._last_receipt: dict[str, Any] = self._base_receipt()
        self._private_binding_guard()
        from . import kernel_prewarm
        kernel_prewarm.start_static(self._seam._cfg)

    @property
    def lagged_diabatic_theta(self) -> Any | None:
        """SG2: the committed MPAS ``rt_diabatic_tend`` the next step's dynamics carries."""

        return self._lagged_diabatic_theta

    @property
    def contract_sha256(self) -> str:
        # This is the contract consumed by CudaRawColumnPhysicsV841 and the
        # existing driver protocol.  The adapter-specific digest is carried in
        # step_receipt so the two independently frozen contracts remain named.
        return CUDA_PHYSICS_V841_CONTRACT_SHA256

    def _engine_kwargs(self) -> dict[str, Any]:
        """The sealed keywords for the engine seam.  A correct-mode run on an
        engine whose seam takes no mode keywords keeps them out of the call
        (the receipt already names the corrections as gaps)."""

        kwargs = self._constructor.arwen_kwargs()
        if not getattr(self, "_engine_takes_mode", True):
            from . import correct_mode

            kwargs = correct_mode.without_mode_keywords(kwargs)
        return kwargs

    def _new_seam(self) -> Any:
        if self._constructor.cumulus_scheme == "ntiedtke":
            # The New Tiedtke row: the frozen batch class, subclassed, with
            # its driver built at cu_physics=16 (woof.hex.cuda_ntiedtke_column_v841).
            # The two engine files the scheme executes beyond the frozen
            # sixteen are verified against the same engine root first.
            from . import cuda_ntiedtke_column_v841 as ntiedtke_row

            self._ntiedtke_sources = dict(
                ntiedtke_row.verify_ntiedtke_sources(self._arwen_root)
            )
            seam = ntiedtke_row.run_ntiedtke_column_batch(
                **self._engine_kwargs()
            )
        else:
            seam = self._factory(**self._engine_kwargs())
        expected = {
            "run_phase1",
            "run_phase2",
            "export_state",
            "restore_state",
            "accumulated_precipitation",
        }
        missing = sorted(name for name in expected if not hasattr(seam, name))
        if missing:
            raise TypeError(f"pinned WOOF seam lacks required methods {missing}")
        public_receipts = ("surface_classification", "last_noahmp_census")
        missing_receipts = sorted(
            name for name in public_receipts if not hasattr(seam, name)
        )
        if missing_receipts:
            raise TypeError(
                f"pinned WOOF seam lacks v2 public receipts {missing_receipts}"
            )
        self._validate_surface_classification(seam)
        # Radiation chunk width only: per-column results are bitwise identical
        # at any width (woof.hex.radiation_chunk_v841 names the measured cost).
        import cupy as _cp
        from .radiation_chunk_v841 import widen_radiation_chunk

        from .radiation_ozone_memo_v841 import install_ozone_memo

        self.radiation_chunk_receipt = widen_radiation_chunk(seam, cp=_cp)
        self.radiation_chunk_receipt["ozone_table_memo"] = install_ozone_memo(seam)
        print(json.dumps({"radiation_chunk": self.radiation_chunk_receipt}), flush=True)
        self._bind_noahmp_statics(seam)
        from .arwen_seaice import install_noahmp_seaice
        install_noahmp_seaice(self, seam)
        return seam

    def _bind_noahmp_statics(self, seam: Any) -> None:
        """Write the sealed Noah-MP statics into the driver's own carriers.

        Runs on every seam this backend builds (the first one and every
        rebuild after a refused step), before the seam's first phase-one
        call, so no Noah-MP step reads the launch-field fill.  A seam that
        exposes no such carrier is refused: silently leaving the fill in
        place is the defect this exists to end.
        """

        statics = self._constructor.noahmp_statics()
        if not statics:
            self._noahmp_statics_receipt = {
                "bound": False,
                "reason": "the sealed mapping carries no shdmax",
                "vegetation_fraction_percent": NOAHMP_ENGINE_DEFAULT_SHDMAX_PERCENT,
            }
            return
        driver = getattr(seam, "_driver", None)
        fields = getattr(driver, "fields", None)
        receipt: dict[str, Any] = {"bound": True, "fields": {}}
        for name, host_value in statics.items():
            if fields is None or name not in fields:
                raise TypeError(
                    f"the pinned WOOF seam exposes no Noah-MP {name!r} "
                    "carrier (seam._driver.fields): without it the land "
                    "surface would keep the engine's launch-field fill "
                    f"({NOAHMP_ENGINE_DEFAULT_SHDMAX_PERCENT:g} % vegetation "
                    "on every column) while the receipt claims the init's "
                    "values"
                )
            target = fields[name]
            if int(target.size) != int(host_value.size):
                raise ValueError(
                    f"Noah-MP {name} carrier holds {int(target.size)} columns; "
                    f"the sealed mapping carries {int(host_value.size)}"
                )
            before = [float(target.min()), float(target.max())]
            if hasattr(target, "__cuda_array_interface__"):
                import cupy as cp

                target[...] = cp.asarray(host_value, dtype=target.dtype).reshape(
                    target.shape
                )
            else:
                target[...] = np.asarray(host_value, dtype=target.dtype).reshape(
                    target.shape
                )
            receipt["fields"][name] = {
                "engine_fill_min_max": before,
                "bound_min": float(host_value.min()),
                "bound_max": float(host_value.max()),
                "bound_mean": float(host_value.mean(dtype=np.float64)),
                "sha256": sha256(
                    np.ascontiguousarray(host_value).tobytes()
                ).hexdigest(),
            }
        self._noahmp_statics_receipt = receipt
        print(
            "[noahmp-statics] shdmax bound from the init: engine fill "
            f"{before[0]:g}..{before[1]:g} % -> "
            f"{receipt['fields']['shdmax']['bound_min']:g}.."
            f"{receipt['fields']['shdmax']['bound_max']:g} % "
            f"(mean {receipt['fields']['shdmax']['bound_mean']:.2f} %) on "
            f"{int(host_value.size)} columns",
            flush=True,
        )

    def _validate_surface_classification(
        self, seam: Any | None = None
    ) -> Mapping[str, Any]:
        selected = self._seam if seam is None else seam
        actual = getattr(selected, "surface_classification", None)
        if not isinstance(actual, Mapping):
            raise TypeError("WOOF v2 surface_classification is not a mapping")
        service = getattr(self, "_surface_updates", None)
        expected = (dict(self._constructor.expected_surface_classification())
                    if service is None or seam is not None else dict(service.classification))
        normalized = dict(actual)
        if normalized != expected:
            raise ValueError(
                "WOOF v2 surface classification differs from sealed native "
                f"constructor authority: {normalized} != {expected}"
            )
        return MappingProxyType(normalized)

    def noahmp_routing_census(self) -> dict[str, Any]:
        """Independent pre-LSM layout counts, including evolving ice routing.

        Extends the executed-layout fix from ec819d5 to uncoupled sea ice.
        The surface service retains the preceding call through held steps and
        analysis alarms. Without that service, Noah-MP's byte-checked layout
        cache is the authority. Neither path reads device arrays.
        """
        service = getattr(self, "_surface_updates", None)
        driver = getattr(self._seam, "_driver", None)
        cached = getattr(getattr(driver, "noahmp_params", None), "_surface_layout_cache", None)
        classification = (dict(self._constructor.expected_surface_classification())
                          if service is None else dict(service.physics_classification or service.classification))
        expected = (dict(cached[2]["census"]) if cached is not None and service is None else {
            "land": classification["sflx_land_columns"],
            "water": classification["open_water_columns"],
            "sea_ice": classification["sea_ice_columns"],
            "glacier": classification["glacier_columns"],
        })
        if expected["glacier"]:
            expected["glacier_path"] = _GLACIER_CUDA_PROVENANCE
        return expected

    def _validate_noahmp_census(
        self, *, require: bool
    ) -> Mapping[str, Any] | None:
        raw = getattr(self._seam, "last_noahmp_census", None)
        if raw is None:
            if require:
                raise ValueError("WOOF v2 did not publish a NoahMP execution census")
            return None
        if not isinstance(raw, Mapping):
            raise TypeError("WOOF v2 last_noahmp_census is not a mapping")
        expected = PersistentTwoPhaseCudaPhysicsBackendV841.noahmp_routing_census(self)
        normalized = dict(raw)
        if normalized != expected:
            raise ValueError(
                "WOOF NoahMP census/provenance differs from executed layout "
                f"authority: {normalized} != {expected}"
            )
        return MappingProxyType(normalized)

    def _private_binding_guard(self) -> None:
        # Effective radii are persisted in the engine's column-batch state but
        # not public.  This guard scopes the one private read to the class
        # that carries them.  (It no longer re-hashes the engine's sixteen
        # files on every call: that hash stopped being a gate on 2026-10-10,
        # see _verify_checkout_root, and the identity is recorded once.)
        seam_type = type(self._seam)
        if seam_type.__module__ != "woof.core.mpas_column_batch":
            # The New Tiedtke row's seam is the frozen class subclassed in
            # woof.hex.cuda_ntiedtke_column_v841; its radius carriers are the
            # frozen class's own, so it is admitted by exact identity of the
            # row's class and of its one base, and nothing broader.
            from . import cuda_ntiedtke_column_v841 as ntiedtke_row

            row_class = ntiedtke_row.seam_class()
            if not (
                self._constructor.cumulus_scheme == "ntiedtke"
                and seam_type is row_class
                and seam_type.__bases__[0].__module__ == "woof.core.mpas_column_batch"
            ):
                raise TypeError(
                    f"the seam is a {seam_type.__module__}.{seam_type.__qualname__}, "
                    "not woof's MPAS column batch (or the New Tiedtke row's "
                    "one subclass of it): radiation reads the effective radii "
                    "from that class's private state and would read attributes "
                    "another class does not define"
                )
        state = getattr(self._seam, "_state", None)
        for name in ("effc", "effi", "effs"):
            if state is None or not hasattr(state, name):
                raise TypeError(
                    f"the engine's column-batch state carries no {name!r}: "
                    "radiation takes the cloud effective radii from it, so "
                    "the radiation call would have no radius to use"
                )

    def _base_receipt(self) -> dict[str, Any]:
        return {
            "schema": CUDA_ARWEN_PHYSICS_V841_SCHEMA,
            "adapter_contract_sha256": CUDA_ARWEN_PHYSICS_V841_CONTRACT_SHA256,
            "coupling_contract_sha256": CUDA_PHYSICS_V841_CONTRACT_SHA256,
            "contract_document_sha256": MPAS_SEAM_CONTRACT_SHA256,
            "contract_surface_sha256": MPAS_SEAM_CONTRACT_SURFACE_SHA256,
            "glacier_composed_tu_sha256": self._engine.glacier_composed_tu_sha256,
            "arwen_commit": self._engine.build_commit,
            "arwen_source_manifest": dict(self._engine.manifest),
            "arwen_engine_version": self._engine.version,
            "arwen_engine_admitted": self._engine.admitted,
            "helper_execution": self._helper_scope.receipt(),
            "dependencies": {
                "prep_contract_sha256": CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256,
                "prep_kernel_sha256": CUDA_PHYSICS_PREP_V841_KERNEL_SHA256,
                "gwdo_contract_sha256": CUDA_GWDO_V841_CONTRACT_SHA256,
                "gwdo_kernel_sha256": CUDA_GWDO_V841_KERNEL_SHA256,
                "coupling_kernel_sha256": CUDA_PHYSICS_V841_KERNEL_SHA256,
            },
            "constructor": dict(self._constructor.receipt()),
            "sea_ice_energy": {"implementation": "MPAS-A v8.4.2 Noah seaice GPU",
                "enabled": self._seam._driver.noahmp_params is not None,
                "cadence_seconds": self._seam._driver.bldt_seconds,
                "fractional_threshold": (self._seam._driver.noahmp_params.xice_threshold
                    if self._seam._driver.noahmp_params is not None else None),
                "before_pbl": True},
            "surface_update": (self._surface_updates.receipt()
                if getattr(self, "_surface_updates", None) is not None else
                getattr(self, "_surface_updates_receipt",
                        {"enabled": False, "reason": "no surface series attached"})),
            # physics.mode = "correct" only: the engine's own resolution, so
            # the receipt lists the findings the switch changed.
            **(
                {"correct_mode": dict(self._correct_mode_receipt)}
                if getattr(self, "_correct_mode_receipt", None) is not None
                else {}
            ),
            "surface_classification": dict(
                self._validate_surface_classification()
            ),
            "noahmp_routing_census": self.noahmp_routing_census(),
            "last_noahmp_census": (
                None
                if self._validate_noahmp_census(require=False) is None
                else dict(self._validate_noahmp_census(require=False))
            ),
            "phase": self._phase if hasattr(self, "_phase") else "constructing",
            "h_diabatic": {
                "reported_by_arwen": True,
                "applied": False,
                "replayed": False,
                "policy": (
                    "the engine's dry-theta ARW rate is not replayed; the MPAS "
                    "modified-theta carry is produced instead (lagged_heating)"
                ),
            },
            "lagged_heating": {
                "finding": "SG2",
                "authority": (
                    "mpas_atmphys_interface.F:909-919 producer; "
                    "mpas_atm_time_integration.F:6937 consumer, :4436-4437 recovery"
                ),
                "carried": getattr(self, "_lagged_diabatic_theta", None) is not None,
                "restored_from_restart": getattr(self, "_lagged_heating_restored", None),
            },
            "gwdo": {"enabled": self._gwdo_static is not None},
            **(
                {
                    "ntiedtke_row": {
                        **dict(getattr(self._seam, "ntiedtke_receipt", {})),
                        "engine_sources_verified": dict(
                            getattr(self, "_ntiedtke_sources", {})
                        ),
                    }
                }
                if self._constructor.cumulus_scheme == "ntiedtke"
                else {}
            ),
            "transaction_rollback": {
                "armed": self._rollback_snapshot,
                "boundary_export": (
                    "every step, before phase one"
                    if self._rollback_snapshot
                    else "not taken; a refused step retires the seam"
                ),
            },
            "limitations": list(_LIMITATIONS),
            "nonclaims": [
                "no claim of separate EOS cloud-pressure routing inside frozen WOOF v2",
                "no claim of exact MPAS t2_p routing inside frozen WOOF v2 RRTMG",
                "no claim that legacy RRTMG is device-resident",
            ],
        }

    def _validate_dt(self, dt: float) -> None:
        if float(dt) != self._constructor.dt:
            raise ValueError(
                f"backend dt={dt} does not equal sealed constructor dt="
                f"{self._constructor.dt}"
            )

    def _validate_phase1_output(self, output: Any, *, time_seconds: float) -> None:
        nlev, ncol = self._constructor.n_levels, self._constructor.n_columns
        shape = (nlev, ncol)
        for name in (
            "du",
            "dv",
            "dtheta",
            "dqv",
            "dqc",
            "dqr",
            "dqi",
            "dqs",
            "dqg",
            "h_diabatic",
        ):
            require_resident_array(
                f"arwen_phase1.{name}", getattr(output, name), dtype=np.float32, shape=shape
            )
        if float(output.elapsed_seconds) != float(time_seconds):
            raise ValueError("WOOF held-output time does not equal MPAS candidate start")
        if int(output.step_index) != int(self._seam.step_index):
            raise ValueError("WOOF held-output step index changed during phase one")

    def _rollback_receipt(self, prior: Mapping[str, Any], armed_text: str) -> dict[str, Any]:
        """The receipt a failed transaction leaves, armed or not."""

        if self._phase == "rollback_not_armed":
            return {
                **prior,
                "phase": "rollback_not_armed",
                "rollback": _ROLLBACK_NOT_ARMED_TEXT,
            }
        return {**prior, "phase": "automatic_rollback", "rollback": armed_text}

    def _restore_boundary(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot is _ROLLBACK_NOT_ARMED:
            # Nothing to restore from.  The seam has run part of the refused
            # step, so it is retired rather than reused: the phase below is
            # not "boundary", and begin_step refuses anything but "boundary".
            self._phase = "rollback_not_armed"
            self._boundary_snapshot = None
            self._step_start = None
            self._candidate_scalar_target = None
            self._candidate_scalar_backup = None
            self._pending_gwdo_result = None
            self._pending_refl10cm = None
            return
        try:
            payload = DeviceBoundarySnapshotV841(self._seam).to_host(snapshot)
            # This internal export is consumed by rollback. Drop its device
            # copies before allocating the replacement seam, so recovery
            # retains only the old seam and the host restore payload.
            snapshot["arrays"].clear()
            fresh = self._new_seam()
            fresh.restore_state(payload)
            self._seam = fresh
            if getattr(self, "_surface_updates", None) is not None:
                self._surface_updates.rollback()
            self._private_binding_guard()
        except Exception as error:
            self._phase = "broken"
            raise RuntimeError("failed to reconstruct the pinned WOOF transaction") from error
        self._phase = "boundary"
        self._boundary_snapshot = None
        self._step_start = None
        self._candidate_scalar_target = None
        self._candidate_scalar_backup = None
        self._pending_gwdo_result = None
        self._pending_refl10cm = None

    def _gf_dynamics_lanes(
        self, carrier: Any, *, start: float, dt: float
    ) -> tuple[Any, Any]:
        """Validate and unpack the previous step's GF advective forcing.

        Naming the breakage this refuses: handing GF the CURRENT step's
        rthdynten (or any other step's) silently feeds the scheme forcing it
        never saw in native, which moves the closure family that decides
        convective mass flux.  A stale or mislabelled carrier is a wrong
        answer that still runs, so the clock is checked, not assumed.
        """

        if carrier is None:
            # Native's tend_physics is zero before the first dynamics step
            # forms it, so zero lanes ARE native at a cold start, and a
            # restart resume has no retained carrier either.  What is NOT
            # allowed is a runner that fed the lane and then stopped: that
            # is a mid-run regression to the pre-parity zeros, which is the
            # exact defect this seam closes.
            if self._gf_forcing_seen:
                raise ValueError(
                    "GF dynamics forcing vanished mid-run: this adapter "
                    "already consumed a rthdynten/rqvdynten carrier, so "
                    "omitting it now would silently restore the pre-parity "
                    "zero-forcing lanes"
                )
            return None, None
        self._gf_forcing_seen = True
        # The driver stamps the carrier with the ENDPOINT time of the step
        # that produced it, which is exactly this step's start.
        held = float(getattr(carrier, "time_seconds", float("nan")))
        if not math.isfinite(held) or held != start:
            raise ValueError(
                "GF dynamics forcing must come from the PREVIOUS dynamics "
                f"step, whose endpoint is this step's start t={start} s; "
                f"got a carrier stamped t={held} s"
            )
        rthdynten = getattr(carrier, "rthdynten", None)
        rqvdynten = getattr(carrier, "rqvdynten", None)
        if rthdynten is None or rqvdynten is None:
            raise TypeError(
                "GF dynamics forcing carrier must publish rthdynten/rqvdynten"
            )
        nlev = self._constructor.n_levels
        ncol = self._constructor.n_columns
        for name, value in (("rthdynten", rthdynten), ("rqvdynten", rqvdynten)):
            if tuple(value.shape) != (nlev, ncol):
                raise ValueError(
                    f"GF {name} must be [level, cell]={(nlev, ncol)}, "
                    f"got {tuple(value.shape)}"
                )
        return rthdynten, rqvdynten

    def begin_step(
        self,
        *,
        atmosphere: Any,
        scalar_names: Sequence[str],
        dt: float,
        dynamics_tendencies: Any = None,
    ) -> CudaRawColumnPhysicsV841:
        """Prepare MPAS columns and invoke frozen Arwen v2 phase one exactly once.

        ``dynamics_tendencies`` is the PREVIOUS step's driver-owned
        :class:`CudaV841GfDynamicsTendencies` carrier -- GF's RTHFTEN/RQVFTEN
        advective forcing.  Native MPAS v8.4.1 forms rthdynten/rqvdynten at
        the end of the dynamics step and the next physics call consumes them
        (mpas_atm_time_integration.F:6936 + :2789), so the carrier handed here
        must be the one produced by the step before this one.  ``None`` is
        the first step only: native's own tend_physics starts at zero.
        """

        if self._phase == "rollback_not_armed":
            raise RuntimeError(f"begin_step refused: {_ROLLBACK_NOT_ARMED_TEXT}")
        if self._phase != "boundary":
            raise RuntimeError("begin_step requires a clean step boundary")
        _scalar_names(scalar_names)
        self._validate_dt(dt)
        start = float(atmosphere.state.time_seconds)
        if not math.isfinite(start) or start < 0.0:
            raise ValueError("candidate start time must be finite and non-negative")
        if float(self._seam.elapsed_seconds) != start:
            raise ValueError(
                "WOOF and MPAS clocks differ at begin_step: "
                f"{self._seam.elapsed_seconds} != {start}"
            )
        snapshot = (
            DeviceBoundarySnapshotV841(self._seam).export_state()
            if self._rollback_snapshot
            else _ROLLBACK_NOT_ARMED
        )
        self._boundary_snapshot = snapshot
        self._step_start = start
        snapshot_bytes = _snapshot_nbytes(snapshot)
        snapshot_d2d_bytes = sum(
            int(value.nbytes) for value in snapshot.get("arrays", {}).values()
            if hasattr(value, "__cuda_array_interface__")
        )
        # Host-owned manifest arrays do not cross the device bus. A memory
        # fallback uses the original exporter and records its actual D2H.
        snapshot_d2h_bytes = (
            0 if (snapshot_d2d_bytes or snapshot is _ROLLBACK_NOT_ARMED) else sum(
                int(value.nbytes)
                for value in self._seam._restart_manifest().values()
                if hasattr(value, "__cuda_array_interface__")
            )
        )
        self._pending_gwdo_result = None
        try:
            if getattr(self, "_surface_updates", None) is not None:
                self._surface_updates.advance(start, dt)
            prepared = prepare_mpas_to_phys_cuda_v841(
                atmosphere,
                scalar_names=scalar_names,
                geometry=self._prep_geometry,
                kernel_cache=self._kernel_cache,
                post_rk_wsm6=False,
            )
            if float(prepared.time_seconds) != start:
                raise ValueError("phase-one prep time changed from the exact MPAS start")
            rthdynten, rqvdynten = self._gf_dynamics_lanes(
                dynamics_tendencies, start=start, dt=dt
            )
            try:
                with self._helper_scope.phase1(
                        n_columns=self._constructor.n_columns,
                        n_levels=self._constructor.n_levels):
                    output = self._seam.run_phase1(
                        dt=dt,
                        u=prepared.u_p,
                        v=prepared.v_p,
                        theta=prepared.th_p,
                        pressure=prepared.pres_hyd_p,
                        pressure_interface=prepared.pres2_hyd_p,
                        z_interface=prepared.z_p,
                        w=prepared.w_p,
                        rho_dry=prepared.rho_dry,
                        # The row's species, by name, in the row's order: the
                        # seam takes them as keyword arguments and this is the
                        # one place they are spelled.
                        **{
                            name: prepared.species_p[name]
                            for name in self._species_row(scalar_names).names()
                        },
                        exner=prepared.pi_p,
                        rthdynten=rthdynten,
                        rqvdynten=rqvdynten,
                    )
            except FloatingPointError as error:
                # The sealed seam refuses by scheme and by field, but it has
                # no way to say WHICH column produced it: it is handed the
                # whole aggregate and it validates the aggregate.  Locating
                # the column is the difference between "the physics blew up"
                # and a sentence a reader can act on, and it costs a passing
                # step nothing because it runs only on this path.
                raise FloatingPointError(
                    f"{error}.  " + _locate_degenerate_columns(prepared)
                ) from error
            # LES reads only these two winds. Retaining the whole preparation
            # kept pressure, species, interfaces and scratch through the
            # dycore and alongside the next step's new preparation.
            # The wave coupling reads one more row: the lowest level's moist air density (waves._arwen_forcing).
            # THE BREAKAGE THIS PREVENTS: trimming this to the LES winds (26ebf4d3) left the wave forcing reading
            # prepared.rho_dry off it, so every ArWen-route run with --waves on stopped at its first commit.
            wave_rho_air = None
            if getattr(self, "wave_binding", None) is not None:
                wave_rho_air = __import__("cupy").ascontiguousarray(
                    prepared.rho_dry[0] * (np.float32(1) + prepared.species_p["qv"][0]))
            self._les_columns = SimpleNamespace(u_p=prepared.u_p, v_p=prepared.v_p, wave_rho_air=wave_rho_air)
            self._validate_phase1_output(output, time_seconds=start)
            surface_classification = self._validate_surface_classification()
            noahmp_census = self._validate_noahmp_census(require=True)
            du, dv = output.du, output.dv
            gwdo_receipt: Mapping[str, Any] | None = None
            gwdo_validation_bytes = 0
            gwdo = None
            if self._gwdo_static is not None:
                view = CudaYsuGwdoColumnViewV841.from_prepared(prepared)
                gwdo_input_du, gwdo_input_dv = output.du, output.dv
                gwdo = run_bl_ysu_gwdo_cuda_v841(
                    view,
                    rublten=gwdo_input_du,
                    rvblten=gwdo_input_dv,
                    static=self._gwdo_static,
                    dt_seconds=dt,
                    kernel_cache=self._gwdo_kernel_cache,
                )
                # Native GWDO returns the already-composed YSU+GWD tendency.
                # Adding dtau again here would double count the operator.
                du, dv = gwdo.rublten, gwdo.rvblten
                gwdo_receipt = gwdo.receipt()
                gwdo_validation_bytes = int(gwdo.validation_d2h.bytes)
                self._pending_gwdo_result = gwdo
            if gwdo is None:
                execution_provenance = (
                    CudaPhaseOneExecutionProvenanceV841.arwen_gwd_off(
                        aggregate_executed=True
                    )
                )
            else:
                execution_provenance = (
                    CudaPhaseOneExecutionProvenanceV841.arwen_with_external_gwdo(
                        aggregate_executed=True,
                        gwdo_executed=True,
                        gwdo_composed_once=True,
                        gwdo_result_module=type(gwdo).__module__,
                        gwdo_result_class=type(gwdo).__name__,
                        gwdo_contract_sha256=gwdo.contract_sha256,
                        gwdo_kernel_sha256=CUDA_GWDO_V841_KERNEL_SHA256,
                        gwdo_validation_d2h=gwdo.validation_d2h,
                        gwdo_input_du_is_arwen_output=(
                            gwdo_input_du is output.du
                        ),
                        gwdo_input_dv_is_arwen_output=(
                            gwdo_input_dv is output.dv
                        ),
                        raw_du_is_gwdo_output=(du is gwdo.rublten),
                        raw_dv_is_gwdo_output=(dv is gwdo.rvblten),
                    )
                )
            raw = CudaRawColumnPhysicsV841(
                du=du,
                dv=dv,
                dtheta=output.dtheta,
                dscalars={
                    "qv": output.dqv,
                    "qc": output.dqc,
                    "qr": output.dqr,
                    "qi": output.dqi,
                    "qs": output.dqs,
                    "qg": output.dqg,
                },
                time_seconds=start,
                execution_provenance=execution_provenance,
            )
            raw.validate(
                n_vert_levels=self._constructor.n_levels,
                n_cells=self._constructor.n_columns,
            )
            self._phase = "begun"
            self._last_receipt = {
                **self._base_receipt(),
                "phase": "begun",
                "start_time_seconds": start,
                "end_time_seconds": start + self._constructor.dt,
                "arwen_step_index": int(output.step_index),
                "surface_classification": dict(surface_classification),
                "noahmp_census": dict(noahmp_census),
                "cadence": {
                    "radiation_ran": bool(output.radiation_ran),
                    "surface_pbl_ran": bool(output.surface_pbl_ran),
                    "cumulus_ran": bool(output.cumulus_ran),
                    "call_counts": dict(self._seam.call_counts),
                },
                "validation_d2h_bytes": {
                    "prep": int(prepared.validation_d2h.bytes),
                    "gwdo": gwdo_validation_bytes,
                    "transaction_boundary_snapshot": snapshot_d2h_bytes,
                },
                "copies": {
                    "arwen_phase1": "published frozen WOOF v2 persistent input/output copies",
                    "transaction_boundary_snapshot_d2h_bytes": snapshot_d2h_bytes,
                    "transaction_boundary_snapshot_d2d_bytes": snapshot_d2d_bytes,
                    "transaction_boundary_snapshot_bytes": snapshot_bytes,
                    "gwdo_candidate_outputs": self._gwdo_static is not None,
                },
                "gwdo": {
                    "enabled": self._gwdo_static is not None,
                    "composed_once": self._gwdo_static is not None,
                    "receipt": None if gwdo_receipt is None else dict(gwdo_receipt),
                },
                "execution_provenance": execution_provenance.receipt(),
            }
            return raw
        except Exception:
            prior = self._last_receipt
            self._restore_boundary(snapshot)
            if self._phase == "rollback_not_armed":
                self._last_receipt = self._rollback_receipt(prior, "")
            raise

    def refresh_les(self, les) -> None:
        """Bind phase-one GPU winds and our surface-layer fluxes, no pools."""
        if self._phase != "begun":
            raise RuntimeError("LES binding requires completed phase one")
        les.winds = self._les_columns
        fluxes = self._seam.surface_fluxes_device()
        for target, source in (("ustm", "ust"), ("hfx", "hfx"), ("qfx", "qfx")):
            les.set(target, fluxes[source])
        les.counts["arwen_surface_flux_bindings"] += 1

    def computes_reflectivity(self) -> bool:
        """Whether this backend's microphysics forms refl10cm.

        Every frozen seam's scheme does.  A subclass whose selection runs no
        microphysics answers False, and the history frame then carries the
        allocated default, as at the initial time.
        """

        return True

    def runs_microphysics(self) -> bool:
        """Whether phase two runs a microphysics scheme.

        Every frozen seam's scheme does.  SG2: native forms rt_diabatic_tend
        only inside microphysics_to_MPAS, which runs only when
        config_microp_scheme /= 'off' (mpas_atm_time_integration.F:2806), so a
        selection without microphysics carries no heating.
        """

        return True

    def initial_refl10cm(self, *, atmosphere: Any, scalar_names: Sequence[str]) -> tuple[Any | None, str]:
        """Hour-0 ``refl10cm`` (dBZ, ``(nlev, ncol)``) of the initial state, and its provenance.

        Native MPAS-A and WRF write the allocated default, zero, at the
        initial time because their first microphysics call has not happened.
        Hex goes its own way here (a declared divergence, documented in
        docs/declared-divergences.md): the initial frame carries the active
        scheme's own 10 cm reflectivity of the initial hydrometeors, through
        the engine's between-steps operator
        ``woof.da.obsop.simulated_reflectivity`` -- the same per-scheme
        routines the due step's microphysics calls, with temperature
        diagnosed from the initial theta and pressure.  THE BREAKAGE the
        zeros caused: an init that carries storms showed none at f000 in
        maps and scores (the ensemble's hour-0 CSI 0 and -60 dBZ max bias).

        Returns ``(None, reason)`` for a selection with no microphysics, and
        for a scheme whose reflectivity the engine cannot separate from its
        step (it refuses by name); the caller then writes the allocated
        default and the reason travels in the frame's provenance.  Nothing
        in the seam or the atmosphere is written.
        """

        if not self.computes_reflectivity():
            return None, "no microphysics in this selection: no reflectivity at any time"
        cp = __import__("cupy")
        from woof.core.refl import refl_melting_for
        from woof.da.obsop import simulated_reflectivity

        names = tuple(scalar_names)
        prepared = prepare_mpas_to_phys_cuda_v841(
            atmosphere,
            scalar_names=names,
            geometry=self._prep_geometry,
            kernel_cache=self._kernel_cache,
            post_rk_wsm6=False,
        )
        scalars = atmosphere.state.scalars
        species = {name: scalars[index] for index, name in enumerate(names)}
        seam = self._seam
        view = InitialReflectivityViewV841(
            theta=prepared.th_p, pressure=prepared.pres_p, species=species,
            melting=refl_melting_for(seam._state), cp=cp)
        try:
            refl = simulated_reflectivity(view, seam._cfg)
        except NotImplementedError as refusal:
            return None, f"engine refuses a between-steps reflectivity for this scheme: {refusal}"
        except Exception as refusal:
            if type(refusal).__name__ != "NativeReflectivityScheme":
                raise
            return None, f"engine refuses a between-steps reflectivity for this scheme: {refusal}"
        nz, ncol = (int(n) for n in prepared.th_p.shape)
        result = cp.array(refl.reshape(nz, ncol), dtype=cp.float32, copy=True, order="C")
        return result, ("hour 0: the scheme's own 10 cm reflectivity of the initial hydrometeors "
                        "(woof.da.obsop.simulated_reflectivity, mp_physics="
                        f"{int(seam._cfg.mp_physics)}, temperature from initial theta and pressure)")

    def _species_row(self, scalar_names: Sequence[str]) -> Any:
        """The species row this backend's scalar block runs under.

        The frozen seams are told apart by their names alone.  A subclass
        whose selection names its row (the engine's generic seam) answers
        from the selection, because two schemes may carry the same block.
        """

        return species_row_for_names(scalar_names)

    def _private_radii(self, radius_names: Sequence[str]) -> dict[str, Any]:
        """The scheme's effective radii, named by its row.

        WSM6 publishes three; P3 publishes two, because its single ice
        category has no separate snow radius.  Reading a fixed three would
        raise AttributeError on a P3 seam rather than refuse by name.
        """

        self._private_binding_guard()
        cp = __import__("cupy")
        state = self._seam._state
        shape = (self._constructor.n_levels, self._constructor.n_columns)
        result = {}
        for name in radius_names:
            value = getattr(state, name).reshape(shape)
            require_resident_array(
                f"fa35_private.{name}", value, dtype=np.float32, shape=shape
            )
            result[name] = cp.array(value, copy=True, order="C")
        return result

    def finish_step(
        self,
        *,
        atmosphere: Any,
        scalar_names: Sequence[str],
        dt: float,
        refl_10cm_due: bool = False,
    ) -> CudaPostRkWsm6UpdateV841:
        """Invoke frozen Arwen v2 WSM6 once on the clamped endpoint and seal its outputs.

        ``refl_10cm_due`` is WRF's history-step ``diagflag`` carried to the
        seam: the due step's microphysics computes ``refl10cm`` from its
        post-call temperature and unchanged prepared pressure (native MPAS-A
        v8.4.1 computes the history field at exactly this point), and the
        staged copy is published by ``commit_step`` for exactly one
        ``take_history_refl10cm`` consumer.
        """

        if self._phase != "begun" or self._boundary_snapshot is None:
            raise RuntimeError("finish_step requires one successful begin_step")
        snapshot = self._boundary_snapshot
        try:
            _scalar_names(scalar_names)
            self._validate_dt(dt)
            start = float(self._step_start)
            endpoint = float(atmosphere.state.time_seconds)
            if endpoint != start + self._constructor.dt:
                raise ValueError(
                    "post-RK candidate time must equal the exact step endpoint: "
                    f"{endpoint} != {start + self._constructor.dt}"
                )
        except Exception:
            prior = self._last_receipt
            self._restore_boundary(snapshot)
            self._last_receipt = self._rollback_receipt(
                prior, "pre-phase-two validation refused; boundary restored"
            )
            raise
        cp = __import__("cupy")
        # WSM6 receives zero-copy scalar aliases.  This device backup is the
        # transaction guard that restores the unpublished MPAS candidate if
        # adaptation/diagnostic validation fails after the in-place call.
        scalar_backup = cp.array(atmosphere.state.scalars, copy=True, order="C")
        self._pending_lagged_diabatic_theta = None
        try:
            # SG2: native mpas_atmphys_interface.F:909 saves theta_m -- the
            # value the preparation divides into th_p -- before the call.
            theta_m_before = cp.array(atmosphere.saved.theta_m, copy=True, order="C")
            prepared = prepare_mpas_to_phys_cuda_v841(
                atmosphere,
                scalar_names=scalar_names,
                geometry=self._prep_geometry,
                kernel_cache=self._kernel_cache,
                post_rk_wsm6=True,
                phase2_only=True,
                phase2_winds=bool(getattr(self._seam, "phase2_takes_winds", False)),
            )
            if float(prepared.time_seconds) != endpoint:
                raise ValueError("phase-two prep time changed from the exact endpoint")
            wsm6 = prepared.wsm6_input_view()
            phase2_row = self._species_row(scalar_names)
            # No scheme, no reflectivity: the engine refuses refl_10cm_due
            # with mp_physics=0 ("no scheme computes a reflectivity").
            refl_10cm_due = bool(refl_10cm_due) and self.computes_reflectivity()
            phase2_kwargs = {
                "theta": prepared.th_p,
                "pressure": prepared.pres_p,
                "z_interface": prepared.z_p,
                "refl_10cm_due": bool(refl_10cm_due),
                **{name: wsm6.species[name] for name in phase2_row.names()},
            }
            # rho_dry is a WSM6 argument.  The pinned seam REFUSES it by name
            # on a P3 batch ("does not consume rho_dry"), because P3's
            # sedimentation does not take the dry-density alternative, so
            # passing it unconditionally would refuse every P3 step.
            # A seam that says it takes both (the engine's generic entry,
            # which dropped the per-scheme input rules) is handed both.
            takes_both = bool(getattr(self._seam, "phase2_takes_density_and_w", False))
            if takes_both or phase2_row.engine_scheme == "wsm6":
                phase2_kwargs["rho_dry"] = prepared.rho_dry
            if takes_both or phase2_row.engine_scheme == "tempo":
                phase2_kwargs["w"] = prepared.w_p
            # A seam whose phase two takes the post-RK winds is handed them.
            # Without them a microphysics that reads winds in phase two (the
            # engine's bin scheme) steps on the phase-one winds, one RK step
            # stale (engine SEAM-CONTRACT v3, run_phase2 u/v).  The pinned
            # batch takes no winds in phase two and is handed none.
            if getattr(self._seam, "phase2_takes_winds", False):
                phase2_kwargs["u"] = prepared.u_p
                phase2_kwargs["v"] = prepared.v_p
            receipt = self._seam.run_phase2(**phase2_kwargs)
            if float(self._seam.elapsed_seconds) != endpoint:
                raise ValueError("WOOF phase two did not advance to the MPAS endpoint")
            staged_refl = None
            if refl_10cm_due:
                staged_refl = receipt.get("refl_10cm")
                require_resident_array(
                    "history_refl10cm",
                    staged_refl,
                    dtype=np.float32,
                    shape=(
                        self._constructor.n_levels,
                        self._constructor.n_columns,
                    ),
                )
            row = phase2_row
            cumulative = self._seam.accumulated_precipitation()
            # The buckets the ROW declares, plus RAINC which every scheme
            # reports.  WSM6 declares three; P3 declares two and the seam
            # binds no graupel accumulator for it, so demanding one here
            # would refuse a correct P3 seam.
            required = {
                item.history_name for item in row.surface_accumulators
            } | {"RAINC"}
            if set(cumulative) != required:
                raise ValueError(
                    "frozen WOOF v2 cumulative precipitation keys changed: "
                    f"{sorted(cumulative)}, expected {sorted(required)}"
                )
            radii = self._private_radii(row.radius_names)
            # SR comes from a scheme that precipitates; a row with no
            # bucket (microphysics off) has no SR and its seam publishes
            # none.  Every other row must hand it over.
            surface = {}
            if row.surface_accumulators or "sr" in receipt:
                surface["sr"] = receipt["sr"]
            for item in row.surface_accumulators:
                surface[item.name] = cumulative[item.history_name]
                surface[f"{item.name}v"] = receipt[f"{item.name}v"]
            update = CudaPostRkWsm6UpdateV841(
                theta=prepared.th_p,
                species={
                    name: wsm6.species[name] for name in row.names()
                },
                surface=surface,
                radii=radii,
                time_seconds=endpoint,
                row=row,
            )
            update.validate(
                n_vert_levels=self._constructor.n_levels,
                n_cells=self._constructor.n_columns,
            )
            # Include endpoint analyses in committed history, while record I/O
            # and conversion can still roll back with the unfinished step.
            if getattr(self, "_surface_updates", None) is not None:
                self._surface_updates.records(
                    self._surface_updates.start + timedelta(seconds=endpoint))
            # SG2: mpas_atmphys_interface.F:918-919 -- the modified-theta
            # change across this call over the step it ran at, staged with the
            # rest of the transaction and published only by commit_step.
            # Microphysics off: native never calls microphysics_to_MPAS, so
            # nothing is carried (the round trip through th_p would otherwise
            # leave a last-place rate that native does not have).
            pending_lagged = None
            if self.runs_microphysics():
                from .lagged_heating_v841 import lagged_theta_m_rate
                pending_lagged = lagged_theta_m_rate(
                    theta_m_before=theta_m_before,
                    theta_after=prepared.th_p,
                    qv_after=wsm6.species[scalar_names[0]],
                    dt=float(dt),
                    rvord=RV_OVER_RD_F32,
                    kernel_cache=self._kernel_cache,
                )
            self._phase = "finished"
            self._pending_lagged_diabatic_theta = pending_lagged
            self._pending_refl10cm = staged_refl
            self._candidate_scalar_target = atmosphere.state.scalars
            self._candidate_scalar_backup = scalar_backup
            prior = self._last_receipt
            self._last_receipt = {
                **self._base_receipt(),
                "phase": "finished_unpublished",
                "start_time_seconds": start,
                "end_time_seconds": endpoint,
                "publication": {
                    "state": "finished_unpublished",
                    "requires": "commit_step after MPAS recovery/driver commit",
                },
                "arwen_step_index": int(self._seam.step_index),
                "cadence": prior.get("cadence", {}),
                "gwdo": prior.get("gwdo", {"enabled": False}),
                "validation_d2h_bytes": {
                    **prior.get("validation_d2h_bytes", {}),
                    "post_rk_prep": int(prepared.validation_d2h.bytes),
                },
                "copies": {
                    **prior.get("copies", {}),
                    "candidate_scalar_transaction_backup_d2d_bytes": int(
                        scalar_backup.nbytes
                    ),
                    "effective_radius_snapshot_d2d_bytes": int(
                        sum(value.nbytes for value in radii.values())
                    ),
                    "cumulative_precipitation": "frozen WOOF v2 public device copies",
                },
                "post_rk": {
                    "in_place_species": list(scalar_names),
                    "refl_10cm_due": bool(refl_10cm_due),
                    "theta": "prepared th_p dry theta",
                    "pressure": "prepared pres_p EOS",
                    "rho": "prepared rho_dry",
                    "z_interface": "prepared z_p",
                    "cumulative_fields": [
                        item.name for item in row.surface_accumulators],
                    "increment_fields": [
                        *(f"{item.name}v"
                          for item in row.surface_accumulators),
                        "sr",
                    ],
                    "private_exact_hash_binding": list(row.radius_names),
                },
            }
            return update
        except Exception:
            self._pending_lagged_diabatic_theta = None
            scalar_restore_error = None
            try:
                atmosphere.state.scalars[...] = scalar_backup
            except Exception as error:
                scalar_restore_error = error
            prior = self._last_receipt
            self._restore_boundary(snapshot)
            self._last_receipt = self._rollback_receipt(
                prior, "phase-two execution refused; candidate and boundary restored"
            )
            if scalar_restore_error is not None:
                raise RuntimeError(
                    "failed to restore unpublished MPAS candidate scalars"
                ) from scalar_restore_error
            raise

    def abort_step(self) -> None:
        """Rollback a begun or finished-unpublished cross-component step."""

        if (
            self._phase not in ("begun", "finished")
            or self._boundary_snapshot is None
        ):
            raise RuntimeError("abort_step requires a begun or finished transaction")
        snapshot = self._boundary_snapshot
        scalar_restore_error = None
        # SG2: an aborted step publishes no heating; the carry the step began
        # with stays the carry of the retried step.
        self._pending_lagged_diabatic_theta = None
        if self._phase == "finished":
            try:
                self._candidate_scalar_target[...] = self._candidate_scalar_backup
            except Exception as error:
                scalar_restore_error = error
        prior = self._last_receipt
        self._restore_boundary(snapshot)
        if self._phase == "rollback_not_armed":
            self._last_receipt = self._rollback_receipt(prior, "")
        else:
            self._last_receipt = {
                **prior,
                "phase": "rolled_back",
                "rollback": "fresh frozen WOOF v2 seam reconstructed from device boundary export",
            }
        if scalar_restore_error is not None:
            raise RuntimeError(
                "WOOF rolled back but unpublished MPAS scalar restoration failed"
            ) from scalar_restore_error

    def commit_step(self) -> None:
        """Publish a finished seam only after MPAS recovery/driver commit succeeds."""

        if self._phase != "finished" or self._boundary_snapshot is None:
            raise RuntimeError("commit_step requires a finished-unpublished transaction")
        if (
            self._candidate_scalar_target is None
            or self._candidate_scalar_backup is None
        ):
            snapshot = self._boundary_snapshot
            self._restore_boundary(snapshot)
            raise RuntimeError("finished transaction lost its MPAS scalar rollback guard")
        if self._gwdo_static is not None and self._pending_gwdo_result is None:
            snapshot = self._boundary_snapshot
            self._restore_boundary(snapshot)
            raise RuntimeError("finished transaction lost its validated GWDO result")
        if self._pending_gwdo_result is not None:
            self._last_gwdo_result = self._pending_gwdo_result
            self._gwdo_calls += 1
        self._pending_gwdo_result = None
        if self._pending_refl10cm is not None:
            self._committed_refl10cm = self._pending_refl10cm
        self._pending_refl10cm = None
        # SG2: the committed step's heating becomes the next step's carry.
        if self._pending_lagged_diabatic_theta is not None:
            self._lagged_diabatic_theta = self._pending_lagged_diabatic_theta
        self._pending_lagged_diabatic_theta = None
        if getattr(self, "_surface_updates", None) is not None:
            self._surface_updates.commit()
        if getattr(self, "wave_binding", None) is not None:
            self.wave_binding.after_arwen_step(self)
        prior = self._last_receipt
        self._phase = "boundary"
        self._boundary_snapshot = None
        self._step_start = None
        self._candidate_scalar_target = None
        self._candidate_scalar_backup = None
        self._last_receipt = {
            **prior,
            "phase": "complete",
            "publication": {
                "state": "committed",
                "committed_after": "MPAS recovery/driver candidate commit",
            },
            "gwdo": {
                **prior.get("gwdo", {"enabled": False}),
                "committed_calls": self._gwdo_calls,
            },
        }

    def take_history_refl10cm(self) -> Any | None:
        """Consume the committed one-frame ``refl10cm`` exactly once.

        Legal only at a committed boundary, mirroring the D2 handoff rule the
        engine applies to its own output frames: the capture that writes the
        history file is the single consumer, and a second read without a new
        due step gets ``None`` rather than a stale frame.
        """

        if self._phase != "boundary":
            raise RuntimeError(
                "take_history_refl10cm is legal only at a committed boundary"
            )
        refl = self._committed_refl10cm
        self._committed_refl10cm = None
        return refl

    def peek_history_refl10cm(self) -> Any | None:
        """The committed one-frame ``refl10cm`` without consuming it.

        A nest-trigger look (:mod:`woof.hex.nest_triggers`) reads the same
        resident array the history capture consumes, on the card, before the
        capture takes it; on a look step that is not a history step the loop
        then clears it with ``take_history_refl10cm`` so no stale frame is
        left for a later capture.
        """

        if self._phase != "boundary":
            raise RuntimeError(
                "peek_history_refl10cm is legal only at a committed boundary"
            )
        return self._committed_refl10cm

    def history_diagnostic_snapshot(self):
        """Resident history fields; caller queues their copies before the next step.

        The frame hashes these fields after its transfer event, so this path
        omits duplicate per-field host hashes and the host/device round trip.
        """
        return self.diagnostic_snapshot(selective=True, resident=True)

    def diagnostic_snapshot(self, *, selective=False, resident=False) -> CudaArwenDiagnosticSnapshotV841:
        """Snapshot public frozen-v2 diagnostics at a committed boundary only."""

        if self._phase != "boundary":
            raise RuntimeError(
                "diagnostic_snapshot is legal only at a committed boundary"
            )
        self._private_binding_guard()
        names = (*_REQUIRED_ARWEN_EXPORT_FIELDS, *_OPTIONAL_ARWEN_EXPORT_FIELDS)
        exported, export_inventory = diagnostic_export(self._seam, names, selective=selective, resident=resident)
        if (
            not isinstance(exported, Mapping)
            or set(exported) != {"identity", "arrays", "scalars"}
            or not isinstance(exported["arrays"], Mapping)
        ):
            raise ValueError("frozen WOOF v2 public export_state schema changed")
        cp = __import__("cupy")
        nlev = self._constructor.n_levels
        ncol = self._constructor.n_columns
        export_arrays = exported["arrays"]
        selected: dict[str, Any] = {}
        selected_hashes: dict[str, Any] = {}
        selected_h2d_bytes = 0
        for name in names:
            key = f"fields/{name}"
            if key not in export_arrays and f"diag/{name}" in export_arrays:
                key = f"diag/{name}"
            if key not in export_arrays:
                if name in _REQUIRED_ARWEN_EXPORT_FIELDS:
                    raise ValueError(f"frozen WOOF v2 public export lacks required {key!r}")
                continue
            host = export_arrays[key] if resident else np.asarray(export_arrays[key])
            if host.dtype != np.dtype(np.float32):
                raise TypeError(f"frozen WOOF v2 public export {key!r} is not FP32")
            if name in _SOIL_DIAGNOSTIC_FIELDS:
                if host.shape == (4, 1, ncol):
                    normalized = host[:, 0, :]
                elif host.shape == (4, ncol):
                    normalized = host
                else:
                    raise ValueError(
                        f"frozen WOOF v2 public export {key!r} has shape {host.shape}; "
                        f"expected {(4, 1, ncol)} or {(4, ncol)}"
                    )
                expected_shape = (4, ncol)
            else:
                if host.shape == (1, 1, ncol):
                    normalized = host[0, 0, :]
                elif host.shape == (1, ncol):
                    normalized = host[0, :]
                elif host.shape == (ncol,):
                    normalized = host
                else:
                    raise ValueError(
                        f"frozen WOOF v2 public export {key!r} has shape {host.shape}; "
                        f"expected a singleton-ny {(ncol,)} field"
                    )
                expected_shape = (ncol,)
            if resident:
                device_value = cp.ascontiguousarray(normalized)
            else:
                normalized = np.array(normalized, copy=True, order="C")
                device_value = cp.asarray(normalized)
            require_resident_array(
                f"arwen_diagnostic.{name}",
                device_value,
                dtype=np.float32,
                shape=expected_shape,
            )
            selected[name] = device_value
            if not resident:
                selected_hashes[key] = _array_identity(normalized)
                selected_h2d_bytes += int(device_value.nbytes)

        public_precip = self._seam.accumulated_precipitation()
        # The seam's PUBLISHED inventory, not a welded four: every row's
        # buckets are the row's own (WSM6 three, P3 two, a provider's
        # declared extras), and finish_step checks the exact row set on
        # every step.  What is fixed here is the shape of the contract --
        # upper-case public keys, RAINC/RAINNC/SNOWNC always present, one
        # lower-case leaf per key.
        precip_keys = {key.lower(): key for key in public_precip}
        if not {"RAINC", "RAINNC", "SNOWNC"} <= set(public_precip) or any(
            key != key.upper() or not key for key in public_precip
        ):
            raise ValueError(
                "frozen WOOF v2 public precipitation inventory changed: "
                f"{sorted(public_precip)}"
            )
        precipitation: dict[str, Any] = {}
        for public_name, fa35_name in precip_keys.items():
            source = public_precip[fa35_name]
            require_resident_array(
                f"arwen_diagnostic.{public_name}",
                source,
                dtype=np.float32,
                shape=(ncol,),
            )
            precipitation[public_name] = cp.array(
                source, copy=True, order="C"
            )

        gwdo: dict[str, Any] = {}
        if self._last_gwdo_result is None:
            for name in _GWDO_SURFACE_FIELDS:
                gwdo[name] = cp.zeros((ncol,), dtype=cp.float32)
            for name in _GWDO_LEVEL_FIELDS:
                gwdo[name] = cp.zeros((nlev, ncol), dtype=cp.float32)
        else:
            self._last_gwdo_result.validate(
                n_vert_levels=nlev, n_cells=ncol
            )
            for name in _GWDO_DIAGNOSTIC_FIELDS:
                gwdo[name] = cp.array(
                    getattr(self._last_gwdo_result, name),
                    copy=True,
                    order="C",
                )

        surface = {
            name: selected[name]
            for name in names
            if name in selected and name not in _SOIL_DIAGNOSTIC_FIELDS
        }
        surface.update(history_cumulus_rain_rate(self, self._seam, precipitation["rainc"], cp))
        if getattr(self, "_surface_updates", None) is not None:
            surface.update({name: self._surface_updates.owned[name].copy()
                            for name in ("sst", "sstsk")})
            surface["tmn"] = cp.array(self._seam._driver.fields["tmn"].reshape(-1), copy=True)
            ocean = getattr(self._surface_updates, "ocean", None)
            waves = getattr(self._surface_updates, "waves", None)
            if waves is not None:
                surface.update(waves.history())
                # The enclosing snapshot reads step_receipt after this history
                # export. Refresh its cached wave counters here so the final
                # interval is represented without a new step synchronization.
                self._last_receipt = {
                    **self._last_receipt,
                    "surface_update": self._surface_updates.receipt(),
                }
            if ocean is not None:
                surface.update({"ocean_"+key: ocean.array(key).copy() for key in
                    ("depth", "volume", "area", "ice_temp", "snow_water", "melt", "basal",
                     "net_flux", "energy", "entrainment", "residual", "initial_sst", "initial_ice")})
                if "ocean_pwp_diagnostics" in self._surface_updates.owned:
                    surface["ocean_pwp_mixing_failures"] = ocean.array("pwp_diagnostics")[:,2].copy()
                    import os
                    if os.environ.get('WOOF_TC_OCEAN_DIAGNOSTICS') == '1':
                        from .ocean_tc import history_fields
                        surface.update(history_fields(ocean))
                    columns = ocean.array("pwp_columns")
                    surface["ocean_pwp_active"] = ocean.array("pwp_diagnostics")[:,3].copy()
                    surface["ocean_temperature"] = cp.ascontiguousarray(columns[:,0,:].T)
                    surface["ocean_salinity"] = cp.ascontiguousarray(columns[:,1,:].T)
                    bottoms = (cp.arange(columns.shape[2], dtype=cp.float32)+1)*ocean.options["pwp"]["layer_m"]
                    surface["ocean_layer_bottom"] = cp.broadcast_to(bottoms[:,None], (columns.shape[2], columns.shape[0])).copy()
        soil = {
            name: selected[name]
            for name in names
            if name in selected and name in _SOIL_DIAGNOSTIC_FIELDS
        }
        full_export_bytes = 0 if resident else _snapshot_nbytes(exported)
        surface_classification = self._validate_surface_classification()
        noahmp_census = self._validate_noahmp_census(require=False)
        receipt = {
            "schema": CUDA_ARWEN_PHYSICS_V841_SCHEMA,
            "boundary": "committed",
            "full_export_d2h_bytes": full_export_bytes,
            "full_export_array_inventory_sha256": _json_digest(
                sorted(export_inventory)
            ),
            "selected_export_inventory": [
                f"fields/{name}" for name in names if name in selected
            ],
            "exported_array_count": len(export_arrays),
            "available_array_count": len(export_inventory),
            "selected_export_hashes": selected_hashes,
            "selected_export_h2d_bytes": selected_h2d_bytes,
            "precipitation_d2d_bytes": sum(
                int(value.nbytes) for value in precipitation.values()
            ),
            "gwdo_d2d_or_zero_fill_bytes": sum(
                int(value.nbytes) for value in gwdo.values()
            ),
            "surface_classification": dict(surface_classification),
            **({"surface_update": self._surface_updates.receipt()}
               if getattr(self, "_surface_updates", None) is not None else {}),
            "noahmp_routing_census": self.noahmp_routing_census(),
            "last_noahmp_census": (
                None if noahmp_census is None else dict(noahmp_census)
            ),
            "q2_policy": "preserved bitwise; negative values are audit data",
        }
        return CudaArwenDiagnosticSnapshotV841(
            surface=MappingProxyType(surface),
            soil=MappingProxyType(soil),
            precipitation=MappingProxyType(precipitation),
            gwdo=MappingProxyType(gwdo),
            metadata=MappingProxyType(
                {
                    "step_index": int(self._seam.step_index),
                    "time_seconds": float(self._seam.elapsed_seconds),
                    "call_counts": MappingProxyType(dict(self._seam.call_counts)),
                    "surface_classification": surface_classification,
                    **({"surface_update": self._surface_updates.receipt()}
                       if getattr(self, "_surface_updates", None) is not None else {}),
                    "noahmp_routing_census": self.noahmp_routing_census(),
                    "last_noahmp_census": noahmp_census,
                    "gwdo_enabled": self._gwdo_static is not None,
                    "gwdo_calls": self._gwdo_calls,
                    "gwdo_has_last_result": self._last_gwdo_result is not None,
                }
            ),
            receipt=MappingProxyType(receipt),
        )

    def _restart_identity(self) -> Mapping[str, Any]:
        return {
            "adapter_contract_sha256": CUDA_ARWEN_PHYSICS_V841_CONTRACT_SHA256,
            "arwen_commit": self._engine.build_commit,
            "arwen_source_manifest": dict(self._engine.manifest),
            "contract_surface_sha256": MPAS_SEAM_CONTRACT_SURFACE_SHA256,
            "glacier_composed_tu_sha256": self._engine.glacier_composed_tu_sha256,
            "constructor_identity_sha256": self._constructor.identity_sha256,
            "prep_contract_sha256": CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256,
            "gwdo_contract_sha256": (
                CUDA_GWDO_V841_CONTRACT_SHA256
                if self._gwdo_static is not None
                else None
            ),
        }

    def restart_state(self) -> Mapping[str, Any]:
        if self._phase != "boundary":
            raise RuntimeError("restart_state is legal only at a step boundary")
        carry = self._lagged_diabatic_theta
        return {
            "schema": CUDA_ARWEN_PHYSICS_V841_SCHEMA,
            **({"surface_updates": self._surface_updates.state()}
               if getattr(self, "_surface_updates", None) is not None else {}),
            **({"waves": self.wave_binding.service.snapshot()}
               if getattr(self, "wave_binding", None) is not None else {}),
            "identity": self._restart_identity(),
            "seam": self._seam.export_state(),
            "adapter": {
                "gwdo_calls": self._gwdo_calls,
                "last_gwdo_result_persisted": False,
            },
            # SG2: native restarts carry rt_diabatic_tend (Registry.xml:704).
            "lagged_heating": {
                "rt_diabatic_tend": (
                    None if carry is None
                    else np.ascontiguousarray(__import__("cupy").asnumpy(carry))
                ),
            },
        }

    def restore_restart_state(self, payload: Mapping[str, Any]) -> None:
        if self._phase != "boundary":
            raise RuntimeError("restore_restart_state is legal only at a step boundary")
        expected_keys = {"schema", "identity", "seam", "adapter"}
        # SG2: a payload written before the lagged heating existed has no
        # "lagged_heating" key and resumes with no carry for its first step
        # (recorded as restored_from_restart False); one that has it must
        # carry a plane of this backend's shape.
        if not isinstance(payload, Mapping) or set(payload) - {"lagged_heating", "surface_updates", "waves"} != expected_keys:
            raise ValueError(
                "backend restart payload must contain "
                "schema/identity/seam/adapter (and optionally lagged_heating) exactly"
            )
        lagged = payload.get("lagged_heating")
        lagged_plane = None
        if lagged is not None:
            if not isinstance(lagged, Mapping) or set(lagged) != {"rt_diabatic_tend"}:
                raise ValueError("backend restart lagged_heating must hold rt_diabatic_tend only")
            if lagged["rt_diabatic_tend"] is not None:
                lagged_plane = np.asarray(lagged["rt_diabatic_tend"])
                shape = (self._constructor.n_levels, self._constructor.n_columns)
                if lagged_plane.dtype != np.float32 or lagged_plane.shape != shape:
                    raise ValueError(
                        f"restart rt_diabatic_tend must be float32 {shape}; a different "
                        "plane would heat the wrong columns in the first resumed step")
        expected = self._restart_identity()
        # this build's schema, or the v2 id WOOF 1.0.0 wrote for the same contract
        if payload["schema"] not in (
            CUDA_ARWEN_PHYSICS_V841_SCHEMA, WOOF_1_0_0_CONTRACT_IDS["schema"]
        ):
            raise ValueError("backend restart schema mismatch")
        if not restart_identity_matches(payload["schema"], payload["identity"], expected):
            raise ValueError("backend restart identity mismatch")
        adapter = payload["adapter"]
        if not isinstance(adapter, Mapping) or set(adapter) != {
            "gwdo_calls",
            "last_gwdo_result_persisted",
        }:
            raise ValueError("backend restart adapter metadata mismatch")
        calls = adapter["gwdo_calls"]
        if isinstance(calls, bool) or not isinstance(calls, (int, np.integer)):
            raise TypeError("restart gwdo_calls must be a non-negative integer")
        calls = int(calls)
        if calls < 0 or (self._gwdo_static is None and calls != 0):
            raise ValueError("restart gwdo_calls conflicts with GWDO identity")
        if adapter["last_gwdo_result_persisted"] is not False:
            raise ValueError("frozen WOOF v2 adapter restart does not persist trajectory-inert GWDO arrays")
        fresh = self._new_seam()
        from .moving_seaice import prepare_restart_ice
        # A seam payload without arrays (one with no prognostic sea ice) has
        # nothing to allocate; restore_state below still checks the payload.
        prepare_restart_ice(fresh, (payload["seam"] or {}).get("arrays") or {})
        fresh.restore_state(payload["seam"])
        self._seam = fresh
        self._wave_restart = payload.get("waves")
        if getattr(self, "wave_binding", None) is not None:
            if self._wave_restart is None:
                raise ValueError("wave-enabled backend requires its spectral checkpoint")
            self.wave_binding.service.restore(self._wave_restart)
            if self.wave_binding.service.options["roughness_feedback"]:
                from .waves import ArwenFeedback
                fresh._driver.ocean = ArwenFeedback(self.wave_binding, fresh._driver.fields["psfc"].shape,
                                                     getattr(fresh._driver, "ocean", None))
        self._surface_restart = payload.get("surface_updates")
        if getattr(self, "_surface_updates", None) is not None:
            if self._surface_restart is None:
                raise ValueError("surface-updated backend requires surface state in its restart")
            self._surface_updates.restore(self._surface_restart)
        self._gwdo_calls = calls
        self._pending_lagged_diabatic_theta = None
        self._lagged_diabatic_theta = (
            None if lagged_plane is None
            else __import__("cupy").asarray(np.ascontiguousarray(lagged_plane))
        )
        self._lagged_heating_restored = lagged is not None
        self._pending_gwdo_result = None
        self._last_gwdo_result = None
        self._private_binding_guard()
        self._last_receipt = {
            **self._base_receipt(),
            "phase": "restored",
            "arwen_step_index": int(self._seam.step_index),
            "time_seconds": float(self._seam.elapsed_seconds),
            "gwdo": {
                "enabled": self._gwdo_static is not None,
                "committed_calls": self._gwdo_calls,
                "last_result_restored": False,
                "nonclaim": "trajectory-inert last diagnostics omitted; next phase1 replaces",
            },
        }

    def step_receipt(self) -> Mapping[str, Any]:
        # Detached JSON data: callers cannot mutate backend state through a
        # nested receipt mapping.
        return MappingProxyType(json.loads(json.dumps(self._last_receipt, sort_keys=True)))


__all__ = [
    "ARWEN_BUILD_COMMIT",
    "ARWEN_GLACIER_COMPOSED_TU_SHA256",
    "ARWEN_SOURCE_MANIFEST",
    "CUDA_ARWEN_PHYSICS_V841_CONTRACT_SHA256",
    "CUDA_ARWEN_PHYSICS_V841_SCHEMA",
    "MPAS_SEAM_CONTRACT_SHA256",
    "MPAS_SEAM_CONTRACT_SURFACE_SHA256",
    "WOOF_1_0_0_CONTRACT_IDS",
    "CudaArwenDiagnosticSnapshotV841",
    "PersistentTwoPhaseCudaPhysicsBackendV841",
    "SealedArwenConstructorV841",
    "contract_id_status",
    "pin_arwen_physics_v841",
    "restart_identity_matches",
]
