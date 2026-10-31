# ======================================================================
# THIRD-PARTY NOTICE.  Parts of this file are hand transcriptions of
# third-party work.  ArWen distributes the file under the Apache License
# 2.0; the notices below belong to the transcribed parts and are kept here
# because their own licences require it.  Full texts are in the repository
# NOTICE and in the licenses/ directory.
#
#   RRTM longwave, transcribed from WRF v4.6.1 phys/module_ra_rrtm.F.
#   That file carries AER's authorship -- Eli J. Mlawer, Steven J.
#   Taubman, Shepard A. Clough, Atmospheric & Environmental Research, Inc.
#   -- but no copyright line and no licence clause, so it falls under
#   WRF's public-domain declaration.  AER is credited here as a matter of
#   accuracy rather than of condition.
# ======================================================================
"""Coefficient ingestion and pinned policies for WRF legacy RRTM LW.

This module is intentionally not an approximate longwave solver.  It loads
the exact WRF v4.6.1 big-endian sequential-unformatted ``RRTM_DATA`` records
into named native-FP32 arrays and exposes setup policies needed by the future
CUDA kernels.  Until ``SETCOEF``, ``GASABS`` (TAUGB1--16), and ``RTRN`` are
ported and validated, :mod:`woof.core.physics` rejects
``ra_lw_physics=1`` at setup instead of silently substituting RRTMGP or a
gray-gas proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import struct

import numpy as np


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "wrf_radiation"
RRTM_DATA_SHA256 = \
    "45dc91514cf018e133301a1b667deeb17f5b65c49ccfaa9140761a8028bc8ae0"
RRTM_SOURCE_SHA256 = \
    "07ebd4e83b2a571837a27748a29ca678440c363f96a3694fc9549ae0911335da"
RRTM_NBANDS = 16
RRTM_NGPT = 140
RRTM_DELTAP_HPA = 4.0


# READ record order and Fortran array bounds from module_ra_rrtm.F:24-46,
# 6903-7474.  Shape order is the declared Fortran order; reshaping uses
# order="F" so array indices match the oracle source directly.
_RECORD_LAYOUT = (
    (("abscoefL1", (5, 13, 16)), ("abscoefH1", (5, 47, 16)),
     ("selfref1", (10, 16))),
    (("abscoefL2", (5, 13, 16)), ("abscoefH2", (5, 47, 16)),
     ("selfref2", (10, 16))),
    (("abscoefL3", (10, 5, 13, 16)),
     ("abscoefH3", (5, 5, 47, 16)), ("selfref3", (10, 16))),
    (("abscoefL4", (9, 5, 13, 16)),
     ("abscoefH4", (6, 5, 47, 16)), ("selfref4", (10, 16))),
    (("abscoefL5", (9, 5, 13, 16)),
     ("abscoefH5", (5, 5, 47, 16)), ("selfref5", (10, 16))),
    (("abscoefL6", (5, 13, 16)), ("selfref6", (10, 16))),
    (("abscoefL7", (9, 5, 13, 16)), ("abscoefH7", (5, 47, 16)),
     ("selfref7", (10, 16))),
    (("abscoefL8", (5, 7, 16)), ("abscoefH8", (5, 53, 16)),
     ("selfref8", (10, 16))),
    (("abscoefL9", (11, 5, 13, 16)), ("abscoefH9", (5, 47, 16)),
     ("selfref9", (10, 16))),
    (("abscoefL10", (5, 13, 16)), ("abscoefH10", (5, 47, 16))),
    (("abscoefL11", (5, 13, 16)), ("abscoefH11", (5, 47, 16)),
     ("selfref11", (10, 16))),
    (("abscoefL12", (9, 5, 13, 16)), ("selfref12", (10, 16))),
    (("abscoefL13", (9, 5, 13, 16)), ("selfref13", (10, 16))),
    (("abscoefL14", (5, 13, 16)), ("abscoefH14", (5, 47, 16)),
     ("selfref14", (10, 16))),
    (("abscoefL15", (9, 5, 13, 16)), ("selfref15", (10, 16))),
    (("abscoefL16", (9, 5, 13, 16)), ("selfref16", (10, 16))),
)


@dataclass(frozen=True)
class RRTMRawTables:
    """Exact native-FP32 images of the 16 WRF ``RRTM_DATA`` records."""

    arrays: dict[str, np.ndarray]
    source_path: Path
    sha256: str
    record_bytes: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for array in self.arrays.values())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_fortran_record(stream, record_number: int) -> bytes:
    marker = stream.read(4)
    if len(marker) != 4:
        raise ValueError(
            f"RRTM_DATA ended before record {record_number} marker")
    (size,) = struct.unpack(">i", marker)
    if size <= 0 or size % 4:
        raise ValueError(
            f"RRTM_DATA record {record_number} has invalid byte count {size}")
    payload = stream.read(size)
    trailer = stream.read(4)
    if len(payload) != size or len(trailer) != 4:
        raise ValueError(f"RRTM_DATA record {record_number} is truncated")
    (trailing_size,) = struct.unpack(">i", trailer)
    if trailing_size != size:
        raise ValueError(
            f"RRTM_DATA record {record_number} marker mismatch: "
            f"{size} != {trailing_size}")
    return payload


@lru_cache(maxsize=1)
def load_rrtm_raw_tables(path: str | Path | None = None) -> RRTMRawTables:
    """Read and structurally validate WRF's 16 raw coefficient records."""
    source = DATA_DIR / "RRTM_DATA" if path is None else Path(path)
    from woof.tables import table_path
    source = table_path("rrtm", source)
    digest = _sha256(source)
    if path is None and digest != RRTM_DATA_SHA256:
        raise ValueError(
            "packaged RRTM_DATA checksum mismatch: "
            f"{digest} != {RRTM_DATA_SHA256}")
    arrays: dict[str, np.ndarray] = {}
    sizes = []
    with source.open("rb") as stream:
        for number, layout in enumerate(_RECORD_LAYOUT, start=1):
            payload = _read_fortran_record(stream, number)
            expected_values = sum(int(np.prod(shape)) for _, shape in layout)
            expected_bytes = 4 * expected_values
            if len(payload) != expected_bytes:
                raise ValueError(
                    f"RRTM_DATA record {number} has {len(payload)} bytes; "
                    f"layout requires {expected_bytes}")
            flat = np.frombuffer(payload, dtype=">f4")
            offset = 0
            for name, shape in layout:
                count = int(np.prod(shape))
                value = np.asarray(
                    flat[offset:offset + count], dtype=np.float32).reshape(
                        shape, order="F")
                if not np.isfinite(value).all() or np.any(value < 0.0):
                    raise ValueError(
                        f"RRTM_DATA array {name} contains invalid coefficients")
                value.setflags(write=False)
                arrays[name] = value
                offset += count
            sizes.append(len(payload))
        if stream.read(1):
            raise ValueError("RRTM_DATA has trailing bytes after record 16")
    return RRTMRawTables(
        arrays=arrays, source_path=source, sha256=digest,
        record_bytes=tuple(sizes))


def rrtm_upper_layer_count(p_top_pa: float) -> int:
    """Transcribe ``NINT(p_top*0.01/deltap)`` from ``rrtminit``.

    ``p_top`` is REAL and ``0.01``/``deltap`` are REAL literals, so the
    quotient is formed in float32 before NINT rounds it half away from
    zero; the float32 value plus 0.5 is exact in binary64, so the floor
    below is NINT for the positive operand.
    """
    if not np.isfinite(p_top_pa) or p_top_pa <= 0.0:
        raise ValueError("p_top_pa must be finite and positive")
    f32 = np.float32
    quotient = f32(f32(f32(p_top_pa) * f32(0.01)) / f32(RRTM_DELTAP_HPA))
    return int(np.floor(float(quotient) + 0.5))


def rrtm_default_trace_gases(year: int, overrides=None) -> dict[str, float]:
    """WRF v4.6.1 pre-CAM-gas RRTM trace-gas policy (RRTMLWRAD F:1854-1863).

    The three variables are REAL(8), but every operand on the right is a
    REAL literal: ``co2vmr = (280. + 90.*exp(0.02*(yr-2000)))*1.e-6`` is
    evaluated in float32 (EXP is the float32 libm expf) and only then
    widened, and ``319.e-9``/``1774.e-9`` are float32 literals widened.  The
    values returned are those binary64 numbers exactly.  A binary64
    evaluation of the same formula lands up to one float32 ULP away once
    MM5ATM stores it into the REAL WKL; the WRF column oracle
    (tools/rrtm_lw_wrf461_oracle) measured exactly that on WKL(2).
    """
    if isinstance(year, bool) or not isinstance(year, (int, np.integer)):
        raise TypeError("year must be an integer")
    from woof.core.noahmp_libm import expf

    f32 = np.float32
    growth = expf(f32(f32(0.02) * f32(int(year) - 2000)))
    co2 = f32(f32(f32(280.0) + f32(f32(90.0) * growth)) * f32(1.0e-6))
    gases = {
        "co2": float(co2),
        "n2o": float(f32(319.0e-9)),
        "ch4": float(f32(1774.0e-9)),
    }
    if overrides is not None:
        from woof.core.trace_gases import CLASSIC_GASES, validate_trace_gas_overrides
        gases.update(validate_trace_gas_overrides(
            overrides, supported=CLASSIC_GASES, consumer="classic RRTM longwave"))
    return gases


__all__ = ["DATA_DIR", "RRTM_DATA_SHA256", "RRTM_DELTAP_HPA",
           "RRTM_NBANDS", "RRTM_NGPT", "RRTMRawTables",
           "load_rrtm_raw_tables", "rrtm_default_trace_gases",
           "rrtm_upper_layer_count"]
