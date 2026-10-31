"""Exact-bit ingest of WRF v4.6.1 legacy RRTMG LW/SW coefficient data.

This module parses the big-endian sequential-unformatted ``RRTMG_LW_DATA``
and ``RRTMG_SW_DATA`` files (the non-DBL variants, which WRF's FP32
``RWORDSIZE=4`` build reads through ``rrtmg_lwlookuptable`` /
``rrtmg_swlookuptable``) and then reproduces, bit for bit, the g-point
reduction that ``rrtmg_lw_ini`` / ``rrtmg_sw_ini`` perform at model start
(``lwcmbdat``/``swcmbdat`` relative weights plus ``cmbgb1``..``cmbgb16`` and
``cmbgb16s``..``cmbgb29``), in native FP32 with WRF's accumulation order.

FROZEN CONTRACT (consumed by the RRTMG LW/SW compute lanes)
-----------------------------------------------------------
``load_rrtmg_lw_coefficients(path)`` / ``load_rrtmg_sw_coefficients(path)``
return ``dict[band_module_name -> dict[var_name -> np.ndarray]]`` where:

* band module names are exactly the WRF Fortran module names
  (``rrlw_kg01``..``rrlw_kg16``, ``rrsw_kg16``..``rrsw_kg29``);
* variable names are exactly the Fortran variable names, both the raw
  file image (``kao``, ``kbo``, ``selfrefo``, ``forrefo``, ``fracrefao``,
  ``sfluxrefo``, ...) and every array the WRF init reduction produces
  (``ka``/``absa``, ``kb``/``absb``, ``selfref``, ``forref``, ``fracrefa``,
  ``fracrefb``, ``sfluxref``, ``ka_mn2``, ... plus per-band extras such as
  ``ccl4``, ``cfc11adj``, ``absch4``, ``rayl``/``rayla``/``raylb``,
  ``abso3a``/``abso3b``, ``absh2o``, ``absco2``);
* array extents match the Fortran declarations and data is loaded in
  Fortran storage order, so ``array[i-1, j-1, ...]`` equals the Fortran
  ``array(i, j, ...)`` element for axes whose declared lower bound is 1.
  The only non-unit lower bounds are the ``13:59`` pressure axes of
  ``kbo``/``kb`` (use :func:`fortran_lower_bounds`); ``absa``/``absb`` are
  the exact WRF ``EQUIVALENCE`` flattenings of ``ka``/``kb``;
* dtype is ``float32`` everywhere except the integer scalar ``layreffr``
  (``int32``); Fortran scalars are 0-d arrays.

Every returned value is validated bit-for-bit against a dump written by a
Fortran program that USEs the unmodified WRF modules after running the real
init path (``tools/rrtmg_wrf461_oracle/coeffs_dump_lw.F90`` / ``_sw.F90``);
see ``tests/test_rrtmg_coeffs.py`` and the pinned per-array SHA-256
manifest ``woof/data/wrf_radiation/rrtmg_coeffs_oracle_manifest.json``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import struct

import numpy as np

from woof.file_digest import sha256_file as _sha256


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "wrf_radiation"

#: SHA-256 of run/RRTMG_LW_DATA from the WRF v4.6.1 source authority.
RRTMG_LW_DATA_SHA256 = \
    "bcfdee24b63a4c909522a329b8e16c539f0173c7e5aea2caf933ab4fe28c5c97"
#: SHA-256 of run/RRTMG_SW_DATA from the WRF v4.6.1 source authority.
RRTMG_SW_DATA_SHA256 = \
    "a7d25f5b4d33be8629cbef7ecacc1ff413bf398a021297793e843ba1cc627baf"

#: SHA-256 of the oracle coefficient dumps this loader is proven against
#: (tools/rrtmg_wrf461_oracle/build.sh, gfortran 13.3.0 FP32 build).
RRTMG_COEFFS_LW_DUMP_SHA256 = \
    "07e7b1f6f71c371187bf7607ac1873a572fe13146a2e47d779c855ecfa5c7b64"
RRTMG_COEFFS_SW_DUMP_SHA256 = \
    "913059dc76a784d29c0be8b5a970f4f5c509a0ab27d2a15bbfeb2bb9fca9f9f7"

_MG = 16

# ---------------------------------------------------------------------------
# g-point reduction data, transcribed from WRF v4.6.1
# phys/module_ra_rrtmg_lw.F subroutine lwcmbdat and
# phys/module_ra_rrtmg_sw.F subroutine swcmbdat.  ``wt`` is identical for
# both spectral halves.  All are 1-based Fortran values kept verbatim;
# indexing converts where used.
# ---------------------------------------------------------------------------

_WT_16 = (
    "0.1527534276", "0.1491729617", "0.1420961469", "0.1316886544",
    "0.1181945205", "0.1019300893", "0.0832767040", "0.0626720116",
    "0.0424925000", "0.0046269894", "0.0038279891", "0.0030260086",
    "0.0022199750", "0.0014140010", "0.0005330000", "0.0000750000",
)

_LW_NGC = (10, 12, 16, 14, 16, 8, 12, 8, 12, 6, 8, 8, 4, 2, 2, 2)
_LW_NGN = (
    1, 1, 2, 2, 2, 2, 2, 2, 1, 1,                    # band 1
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,              # band 2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # band 3
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3,        # band 4
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # band 5
    2, 2, 2, 2, 2, 2, 2, 2,                          # band 6
    2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,              # band 7
    2, 2, 2, 2, 2, 2, 2, 2,                          # band 8
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,              # band 9
    2, 2, 2, 2, 4, 4,                                # band 10
    1, 1, 2, 2, 2, 2, 3, 3,                          # band 11
    1, 1, 1, 1, 2, 2, 4, 4,                          # band 12
    3, 3, 4, 6,                                      # band 13
    8, 8,                                            # band 14
    8, 8,                                            # band 15
    4, 12,                                           # band 16
)
_LW_NGM = (
    1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10,          # band 1
    1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 11, 11, 12, 12,     # band 2
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,    # band 3
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14,    # band 4
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,    # band 5
    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,           # band 6
    1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 12,      # band 7
    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,           # band 8
    1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 11, 11, 12, 12,     # band 9
    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6,           # band 10
    1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8,           # band 11
    1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,           # band 12
    1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,           # band 13
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,           # band 14
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,           # band 15
    1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,           # band 16
)

_SW_NGC = (6, 12, 8, 8, 10, 10, 2, 10, 8, 6, 6, 8, 6, 12)
_SW_NGN = (
    2, 2, 2, 2, 4, 4,                                # band 16
    1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2,              # band 17
    1, 1, 1, 1, 2, 2, 4, 4,                          # band 18
    1, 1, 1, 1, 2, 2, 4, 4,                          # band 19
    1, 1, 1, 1, 1, 1, 1, 1, 2, 6,                    # band 20
    1, 1, 1, 1, 1, 1, 1, 1, 2, 6,                    # band 21
    8, 8,                                            # band 22
    2, 2, 1, 1, 1, 1, 1, 1, 2, 4,                    # band 23
    2, 2, 2, 2, 2, 2, 2, 2,                          # band 24
    1, 1, 2, 2, 4, 6,                                # band 25
    1, 1, 2, 2, 4, 6,                                # band 26
    1, 1, 1, 1, 1, 1, 4, 6,                          # band 27
    1, 1, 2, 2, 4, 6,                                # band 28
    1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,              # band 29
)
_SW_NGM = (
    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6,            # band 16
    1, 2, 3, 4, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 12, 12,       # band 17
    1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,            # band 18
    1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,            # band 19
    1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 10, 10, 10, 10,      # band 20
    1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 10, 10, 10, 10,      # band 21
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,            # band 22
    1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 10, 10,        # band 23
    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,            # band 24
    1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,            # band 25
    1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,            # band 26
    1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8,            # band 27
    1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,            # band 28
    1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10, 11, 12,         # band 29
)

# Raw variables reduced WITHOUT the relative weights (Planck fractions and
# solar source data; WRF combines these by plain summation).
_UNWEIGHTED = frozenset({"fracrefao", "fracrefbo", "sfluxrefo"})


def _g_axis(name: str, ndim: int) -> int:
    """Original-g-point axis of a raw array, from the cmbgb index patterns."""
    if name.startswith(("kao", "kbo", "selfrefo", "forrefo")):
        return ndim - 1
    return 0


def _reduced_name(raw: str) -> str:
    if raw.startswith("kao_"):
        return "ka_" + raw[4:]
    if raw.startswith("kbo_"):
        return "kb_" + raw[4:]
    if raw == "kao":
        return "ka"
    if raw == "kbo":
        return "kb"
    if not raw.endswith("o"):
        raise ValueError(f"no reduction rule for raw variable {raw!r}")
    return raw[:-1]


def _relative_weights(ngc: tuple[int, ...], ngn: tuple[int, ...],
                      ngm: tuple[int, ...]) -> np.ndarray:
    """FP32 ``rwgt`` exactly as rrtmg_lw_ini/rrtmg_sw_ini compute it."""
    wt = np.array([np.float32(v) for v in _WT_16], dtype=np.float32)
    nbnd = len(ngc)
    rwgt = np.zeros(nbnd * _MG, dtype=np.float32)
    igcsm = 0
    for ibnd in range(nbnd):
        wtsm = np.zeros(_MG, dtype=np.float32)
        iprsm = 0
        if ngc[ibnd] < _MG:
            for igc in range(ngc[ibnd]):
                wtsum = np.float32(0.0)
                for _ in range(ngn[igcsm]):
                    wtsum = np.float32(wtsum + wt[iprsm])
                    iprsm += 1
                igcsm += 1
                wtsm[igc] = wtsum
            for ig in range(_MG):
                ind = ibnd * _MG + ig
                rwgt[ind] = np.float32(wt[ig] / wtsm[ngm[ind] - 1])
        else:
            for ig in range(_MG):
                igcsm += 1
                rwgt[ibnd * _MG + ig] = np.float32(1.0)
    return rwgt


def _reduce(raw: np.ndarray, name: str, band_index: int,
            ngn_segment: tuple[int, ...], rwgt: np.ndarray) -> np.ndarray:
    """One cmbgb reduction: FP32 ordered accumulation over grouped g-points.

    Vectorized over the non-g axes; the accumulation order over the
    original g-points inside each group matches the Fortran ipr loop, so
    every element reproduces WRF's float32 rounding sequence.
    """
    axis = _g_axis(name, raw.ndim)
    moved = np.moveaxis(raw, axis, -1)
    weighted = name not in _UNWEIGHTED
    base = band_index * _MG
    groups = []
    iprsm = 0
    for count in ngn_segment:
        acc = np.zeros(moved.shape[:-1], dtype=np.float32)
        for _ in range(count):
            term = moved[..., iprsm]
            if weighted:
                term = np.float32(rwgt[base + iprsm]) * term
                term = term.astype(np.float32)
            acc = (acc + term).astype(np.float32)
            iprsm += 1
        groups.append(acc)
    out = np.stack(groups, axis=-1)
    return np.moveaxis(out, -1, axis)


def _read_fortran_record(stream, label: str, record_number: int) -> bytes:
    marker = stream.read(4)
    if len(marker) != 4:
        raise ValueError(
            f"{label} ended before record {record_number} marker")
    (size,) = struct.unpack(">i", marker)
    if size <= 0 or size % 4:
        raise ValueError(
            f"{label} record {record_number} has invalid byte count {size}")
    payload = stream.read(size)
    trailer = stream.read(4)
    if len(payload) != size or len(trailer) != 4:
        raise ValueError(f"{label} record {record_number} is truncated")
    (trailing_size,) = struct.unpack(">i", trailer)
    if trailing_size != size:
        raise ValueError(
            f"{label} record {record_number} marker mismatch: "
            f"{size} != {trailing_size}")
    return payload


def _load_side(source: Path, label: str, spec: dict, ngc, ngn, ngm,
               expected_sha: str, path_was_default: bool):
    digest = _sha256(source)
    if path_was_default and digest != expected_sha:
        raise ValueError(
            f"packaged {label} checksum mismatch: {digest} != {expected_sha}")
    rwgt = _relative_weights(ngc, ngn, ngm)
    ngs_offsets = np.concatenate(([0], np.cumsum(ngc))).tolist()
    result: dict[str, dict[str, np.ndarray]] = {}
    with source.open("rb") as stream:
        for band_index, (module, (record_spec, reduced_spec)) in enumerate(
                spec.items()):
            payload = _read_fortran_record(stream, label, band_index + 1)
            expected_bytes = 4 * sum(
                int(np.prod(shape)) if shape else 1
                for _, _, shape, _ in record_spec)
            if len(payload) != expected_bytes:
                raise ValueError(
                    f"{label} record {band_index + 1} ({module}) has "
                    f"{len(payload)} bytes; declarations require "
                    f"{expected_bytes}")
            arrays: dict[str, np.ndarray] = {}
            offset = 0
            for name, kind, shape, _ in record_spec:
                count = int(np.prod(shape)) if shape else 1
                flat = np.frombuffer(
                    payload, dtype=">" + kind, count=count, offset=offset)
                offset += 4 * count
                value = flat.astype(kind)
                value = value.reshape(shape, order="F") if shape else \
                    value.reshape(())
                if kind == "f4" and not np.isfinite(value).all():
                    raise ValueError(
                        f"{label} {module}/{name} contains non-finite "
                        "coefficients")
                arrays[name] = value
            # WRF init reduction (cmbgb*) for this band.
            seg = tuple(
                ngn[ngs_offsets[band_index]:ngs_offsets[band_index + 1]])
            reduced_names = {name for name, _, _, _ in reduced_spec}
            for name, kind, shape, _ in record_spec:
                if shape is None:
                    continue
                target = _reduced_name(name) if name.endswith("o") or \
                    name.startswith(("kao", "kbo")) else None
                if target is None or target not in reduced_names:
                    continue
                arrays[target] = _reduce(
                    arrays[name], name, band_index, seg, rwgt)
            if "absa" in reduced_names:
                ka = arrays["ka"]
                arrays["absa"] = ka.reshape(
                    (-1, ka.shape[-1]), order="F")
            if "absb" in reduced_names:
                kb = arrays["kb"]
                arrays["absb"] = kb.reshape(
                    (-1, kb.shape[-1]), order="F")
            # Structural check: exactly the contracted names, shapes, dtypes.
            for name, kind, shape, _ in reduced_spec:
                if name not in arrays:
                    raise ValueError(
                        f"{label} {module}/{name} was not produced by the "
                        "reduction")
                got = arrays[name]
                want_shape = shape if shape is not None else ()
                if tuple(got.shape) != tuple(want_shape) or \
                        got.dtype != np.dtype(kind):
                    raise ValueError(
                        f"{label} {module}/{name} has shape {got.shape} "
                        f"dtype {got.dtype}; declarations require "
                        f"{want_shape} {kind}")
            for value in arrays.values():
                value.setflags(write=False)
            result[module] = arrays
        if stream.read(1):
            raise ValueError(f"{label} has trailing bytes after the last "
                             "band record")
    return result


@lru_cache(maxsize=1)
def load_rrtmg_lw_coefficients(path: str | Path | None = None):
    """Load the 16 LW band-coefficient modules (raw + reduced), exact-bit.

    Returns the frozen-contract nested dict described in the module
    docstring.  With ``path=None`` the packaged
    ``woof/data/wrf_radiation/RRTMG_LW_DATA`` is used and its SHA-256 is
    enforced.
    """
    source = DATA_DIR / "RRTMG_LW_DATA" if path is None else Path(path)
    from woof.tables import table_path
    source = table_path("rrtmg-lw", source)
    return _load_side(source, "RRTMG_LW_DATA", _LW_SPEC,
                      _LW_NGC, _LW_NGN, _LW_NGM,
                      RRTMG_LW_DATA_SHA256, path is None)


@lru_cache(maxsize=1)
def load_rrtmg_sw_coefficients(path: str | Path | None = None):
    """Load the 14 SW band-coefficient modules (raw + reduced), exact-bit.

    Returns the frozen-contract nested dict described in the module
    docstring.  With ``path=None`` the packaged
    ``woof/data/wrf_radiation/RRTMG_SW_DATA`` is used and its SHA-256 is
    enforced.
    """
    source = DATA_DIR / "RRTMG_SW_DATA" if path is None else Path(path)
    from woof.tables import table_path
    source = table_path("rrtmg-sw", source)
    return _load_side(source, "RRTMG_SW_DATA", _SW_SPEC,
                      _SW_NGC, _SW_NGN, _SW_NGM,
                      RRTMG_SW_DATA_SHA256, path is None)


def fortran_lower_bounds(module: str, var: str) -> tuple[int, ...]:
    """Declared Fortran lower bound per axis (1 except kbo/kb 13:59 axes)."""
    spec = _LW_SPEC.get(module) or _SW_SPEC.get(module)
    if spec is None:
        raise KeyError(module)
    for name, _, shape, lbounds in spec[0] + spec[1]:
        if name == var:
            if shape is None:
                return ()
            return tuple(lbounds)
    raise KeyError(f"{module}/{var}")


# ---------------------------------------------------------------------------
# Machine-transcribed declaration/read-statement spec.
#
# Per band module: (record_spec, reduced_spec); each entry is
# (fortran_name, dtype, extents, declared_lower_bounds), with None extents
# for Fortran scalars.  record_spec lists the exact READ statement order of
# lw_kgb01..16 / sw_kgb16..29; reduced_spec lists every additional module
# variable the WRF init path fills (cmbgb reduction targets and the
# absa/absb EQUIVALENCE views).  Transcribed from the rrlw_kg01..16 /
# rrsw_kg16..29 declarations in the WRF v4.6.1 authority sources (SHA-256
# in tools/rrtmg_wrf461_oracle/) and verified bit-for-bit against the
# coeffs_dump_lw/_sw oracle dumps by tests/test_rrtmg_coeffs.py.
# ---------------------------------------------------------------------------

_LW_SPEC = {
    "rrlw_kg01": (
        (
            ("fracrefao", "f4", (16,), (1,)),
            ("fracrefbo", "f4", (16,), (1,)),
            ("kao", "f4", (5, 13, 16), (1, 1, 1)),
            ("kbo", "f4", (5, 47, 16), (1, 13, 1)),
            ("kao_mn2", "f4", (19, 16), (1, 1)),
            ("kbo_mn2", "f4", (19, 16), (1, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (4, 16), (1, 1)),
        ),
        (
            ("fracrefa", "f4", (10,), (1,)),
            ("fracrefb", "f4", (10,), (1,)),
            ("ka", "f4", (5, 13, 10), (1, 1, 1)),
            ("absa", "f4", (65, 10), (1, 1)),
            ("kb", "f4", (5, 47, 10), (1, 13, 1)),
            ("absb", "f4", (235, 10), (1, 1)),
            ("ka_mn2", "f4", (19, 10), (1, 1)),
            ("kb_mn2", "f4", (19, 10), (1, 1)),
            ("selfref", "f4", (10, 10), (1, 1)),
            ("forref", "f4", (4, 10), (1, 1)),
        ),
    ),
    "rrlw_kg02": (
        (
            ("fracrefao", "f4", (16,), (1,)),
            ("fracrefbo", "f4", (16,), (1,)),
            ("kao", "f4", (5, 13, 16), (1, 1, 1)),
            ("kbo", "f4", (5, 47, 16), (1, 13, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (4, 16), (1, 1)),
        ),
        (
            ("fracrefa", "f4", (12,), (1,)),
            ("fracrefb", "f4", (12,), (1,)),
            ("ka", "f4", (5, 13, 12), (1, 1, 1)),
            ("absa", "f4", (65, 12), (1, 1)),
            ("kb", "f4", (5, 47, 12), (1, 13, 1)),
            ("absb", "f4", (235, 12), (1, 1)),
            ("selfref", "f4", (10, 12), (1, 1)),
            ("forref", "f4", (4, 12), (1, 1)),
        ),
    ),
    "rrlw_kg03": (
        (
            ("fracrefao", "f4", (16, 9), (1, 1)),
            ("fracrefbo", "f4", (16, 5), (1, 1)),
            ("kao", "f4", (9, 5, 13, 16), (1, 1, 1, 1)),
            ("kbo", "f4", (5, 5, 47, 16), (1, 1, 13, 1)),
            ("kao_mn2o", "f4", (9, 19, 16), (1, 1, 1)),
            ("kbo_mn2o", "f4", (5, 19, 16), (1, 1, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (4, 16), (1, 1)),
        ),
        (
            ("fracrefa", "f4", (16, 9), (1, 1)),
            ("fracrefb", "f4", (16, 5), (1, 1)),
            ("ka", "f4", (9, 5, 13, 16), (1, 1, 1, 1)),
            ("absa", "f4", (585, 16), (1, 1)),
            ("kb", "f4", (5, 5, 47, 16), (1, 1, 13, 1)),
            ("absb", "f4", (1175, 16), (1, 1)),
            ("ka_mn2o", "f4", (9, 19, 16), (1, 1, 1)),
            ("kb_mn2o", "f4", (5, 19, 16), (1, 1, 1)),
            ("selfref", "f4", (10, 16), (1, 1)),
            ("forref", "f4", (4, 16), (1, 1)),
        ),
    ),
    "rrlw_kg04": (
        (
            ("fracrefao", "f4", (16, 9), (1, 1)),
            ("fracrefbo", "f4", (16, 5), (1, 1)),
            ("kao", "f4", (9, 5, 13, 16), (1, 1, 1, 1)),
            ("kbo", "f4", (5, 5, 47, 16), (1, 1, 13, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (4, 16), (1, 1)),
        ),
        (
            ("fracrefa", "f4", (14, 9), (1, 1)),
            ("fracrefb", "f4", (14, 5), (1, 1)),
            ("ka", "f4", (9, 5, 13, 14), (1, 1, 1, 1)),
            ("absa", "f4", (585, 14), (1, 1)),
            ("kb", "f4", (5, 5, 47, 14), (1, 1, 13, 1)),
            ("absb", "f4", (1175, 14), (1, 1)),
            ("selfref", "f4", (10, 14), (1, 1)),
            ("forref", "f4", (4, 14), (1, 1)),
        ),
    ),
    "rrlw_kg05": (
        (
            ("fracrefao", "f4", (16, 9), (1, 1)),
            ("fracrefbo", "f4", (16, 5), (1, 1)),
            ("kao", "f4", (9, 5, 13, 16), (1, 1, 1, 1)),
            ("kbo", "f4", (5, 5, 47, 16), (1, 1, 13, 1)),
            ("kao_mo3", "f4", (9, 19, 16), (1, 1, 1)),
            ("ccl4o", "f4", (16,), (1,)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (4, 16), (1, 1)),
        ),
        (
            ("fracrefa", "f4", (16, 9), (1, 1)),
            ("fracrefb", "f4", (16, 5), (1, 1)),
            ("ka", "f4", (9, 5, 13, 16), (1, 1, 1, 1)),
            ("absa", "f4", (585, 16), (1, 1)),
            ("kb", "f4", (5, 5, 47, 16), (1, 1, 13, 1)),
            ("absb", "f4", (1175, 16), (1, 1)),
            ("ka_mo3", "f4", (9, 19, 16), (1, 1, 1)),
            ("selfref", "f4", (10, 16), (1, 1)),
            ("forref", "f4", (4, 16), (1, 1)),
            ("ccl4", "f4", (16,), (1,)),
        ),
    ),
    "rrlw_kg06": (
        (
            ("fracrefao", "f4", (16,), (1,)),
            ("kao", "f4", (5, 13, 16), (1, 1, 1)),
            ("kao_mco2", "f4", (19, 16), (1, 1)),
            ("cfc11adjo", "f4", (16,), (1,)),
            ("cfc12o", "f4", (16,), (1,)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (4, 16), (1, 1)),
        ),
        (
            ("fracrefa", "f4", (8,), (1,)),
            ("ka", "f4", (5, 13, 8), (1, 1, 1)),
            ("absa", "f4", (65, 8), (1, 1)),
            ("ka_mco2", "f4", (19, 8), (1, 1)),
            ("selfref", "f4", (10, 8), (1, 1)),
            ("forref", "f4", (4, 8), (1, 1)),
            ("cfc11adj", "f4", (8,), (1,)),
            ("cfc12", "f4", (8,), (1,)),
        ),
    ),
    "rrlw_kg07": (
        (
            ("fracrefao", "f4", (16, 9), (1, 1)),
            ("fracrefbo", "f4", (16,), (1,)),
            ("kao", "f4", (9, 5, 13, 16), (1, 1, 1, 1)),
            ("kbo", "f4", (5, 47, 16), (1, 13, 1)),
            ("kao_mco2", "f4", (9, 19, 16), (1, 1, 1)),
            ("kbo_mco2", "f4", (19, 16), (1, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (4, 16), (1, 1)),
        ),
        (
            ("fracrefb", "f4", (12,), (1,)),
            ("fracrefa", "f4", (12, 9), (1, 1)),
            ("ka", "f4", (9, 5, 13, 12), (1, 1, 1, 1)),
            ("absa", "f4", (585, 12), (1, 1)),
            ("kb", "f4", (5, 47, 12), (1, 13, 1)),
            ("absb", "f4", (235, 12), (1, 1)),
            ("ka_mco2", "f4", (9, 19, 12), (1, 1, 1)),
            ("kb_mco2", "f4", (19, 12), (1, 1)),
            ("selfref", "f4", (10, 12), (1, 1)),
            ("forref", "f4", (4, 12), (1, 1)),
        ),
    ),
    "rrlw_kg08": (
        (
            ("fracrefao", "f4", (16,), (1,)),
            ("fracrefbo", "f4", (16,), (1,)),
            ("kao", "f4", (5, 13, 16), (1, 1, 1)),
            ("kbo", "f4", (5, 47, 16), (1, 13, 1)),
            ("kao_mco2", "f4", (19, 16), (1, 1)),
            ("kbo_mco2", "f4", (19, 16), (1, 1)),
            ("kao_mn2o", "f4", (19, 16), (1, 1)),
            ("kbo_mn2o", "f4", (19, 16), (1, 1)),
            ("kao_mo3", "f4", (19, 16), (1, 1)),
            ("cfc12o", "f4", (16,), (1,)),
            ("cfc22adjo", "f4", (16,), (1,)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (4, 16), (1, 1)),
        ),
        (
            ("fracrefa", "f4", (8,), (1,)),
            ("fracrefb", "f4", (8,), (1,)),
            ("cfc12", "f4", (8,), (1,)),
            ("cfc22adj", "f4", (8,), (1,)),
            ("ka", "f4", (5, 13, 8), (1, 1, 1)),
            ("absa", "f4", (65, 8), (1, 1)),
            ("kb", "f4", (5, 47, 8), (1, 13, 1)),
            ("absb", "f4", (235, 8), (1, 1)),
            ("ka_mco2", "f4", (19, 8), (1, 1)),
            ("ka_mn2o", "f4", (19, 8), (1, 1)),
            ("ka_mo3", "f4", (19, 8), (1, 1)),
            ("kb_mco2", "f4", (19, 8), (1, 1)),
            ("kb_mn2o", "f4", (19, 8), (1, 1)),
            ("selfref", "f4", (10, 8), (1, 1)),
            ("forref", "f4", (4, 8), (1, 1)),
        ),
    ),
    "rrlw_kg09": (
        (
            ("fracrefao", "f4", (16, 9), (1, 1)),
            ("fracrefbo", "f4", (16,), (1,)),
            ("kao", "f4", (9, 5, 13, 16), (1, 1, 1, 1)),
            ("kbo", "f4", (5, 47, 16), (1, 13, 1)),
            ("kao_mn2o", "f4", (9, 19, 16), (1, 1, 1)),
            ("kbo_mn2o", "f4", (19, 16), (1, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (4, 16), (1, 1)),
        ),
        (
            ("fracrefb", "f4", (12,), (1,)),
            ("fracrefa", "f4", (12, 9), (1, 1)),
            ("ka", "f4", (9, 5, 13, 12), (1, 1, 1, 1)),
            ("absa", "f4", (585, 12), (1, 1)),
            ("kb", "f4", (5, 47, 12), (1, 13, 1)),
            ("absb", "f4", (235, 12), (1, 1)),
            ("ka_mn2o", "f4", (9, 19, 12), (1, 1, 1)),
            ("kb_mn2o", "f4", (19, 12), (1, 1)),
            ("selfref", "f4", (10, 12), (1, 1)),
            ("forref", "f4", (4, 12), (1, 1)),
        ),
    ),
    "rrlw_kg10": (
        (
            ("fracrefao", "f4", (16,), (1,)),
            ("fracrefbo", "f4", (16,), (1,)),
            ("kao", "f4", (5, 13, 16), (1, 1, 1)),
            ("kbo", "f4", (5, 47, 16), (1, 13, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (4, 16), (1, 1)),
        ),
        (
            ("fracrefa", "f4", (6,), (1,)),
            ("fracrefb", "f4", (6,), (1,)),
            ("ka", "f4", (5, 13, 6), (1, 1, 1)),
            ("absa", "f4", (65, 6), (1, 1)),
            ("kb", "f4", (5, 47, 6), (1, 13, 1)),
            ("absb", "f4", (235, 6), (1, 1)),
            ("selfref", "f4", (10, 6), (1, 1)),
            ("forref", "f4", (4, 6), (1, 1)),
        ),
    ),
    "rrlw_kg11": (
        (
            ("fracrefao", "f4", (16,), (1,)),
            ("fracrefbo", "f4", (16,), (1,)),
            ("kao", "f4", (5, 13, 16), (1, 1, 1)),
            ("kbo", "f4", (5, 47, 16), (1, 13, 1)),
            ("kao_mo2", "f4", (19, 16), (1, 1)),
            ("kbo_mo2", "f4", (19, 16), (1, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (4, 16), (1, 1)),
        ),
        (
            ("fracrefa", "f4", (8,), (1,)),
            ("fracrefb", "f4", (8,), (1,)),
            ("ka", "f4", (5, 13, 8), (1, 1, 1)),
            ("absa", "f4", (65, 8), (1, 1)),
            ("kb", "f4", (5, 47, 8), (1, 13, 1)),
            ("absb", "f4", (235, 8), (1, 1)),
            ("ka_mo2", "f4", (19, 8), (1, 1)),
            ("kb_mo2", "f4", (19, 8), (1, 1)),
            ("selfref", "f4", (10, 8), (1, 1)),
            ("forref", "f4", (4, 8), (1, 1)),
        ),
    ),
    "rrlw_kg12": (
        (
            ("fracrefao", "f4", (16, 9), (1, 1)),
            ("kao", "f4", (9, 5, 13, 16), (1, 1, 1, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (4, 16), (1, 1)),
        ),
        (
            ("fracrefa", "f4", (8, 9), (1, 1)),
            ("ka", "f4", (9, 5, 13, 8), (1, 1, 1, 1)),
            ("absa", "f4", (585, 8), (1, 1)),
            ("selfref", "f4", (10, 8), (1, 1)),
            ("forref", "f4", (4, 8), (1, 1)),
        ),
    ),
    "rrlw_kg13": (
        (
            ("fracrefao", "f4", (16, 9), (1, 1)),
            ("fracrefbo", "f4", (16,), (1,)),
            ("kao", "f4", (9, 5, 13, 16), (1, 1, 1, 1)),
            ("kao_mco2", "f4", (9, 19, 16), (1, 1, 1)),
            ("kao_mco", "f4", (9, 19, 16), (1, 1, 1)),
            ("kbo_mo3", "f4", (19, 16), (1, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (4, 16), (1, 1)),
        ),
        (
            ("fracrefb", "f4", (4,), (1,)),
            ("fracrefa", "f4", (4, 9), (1, 1)),
            ("ka", "f4", (9, 5, 13, 4), (1, 1, 1, 1)),
            ("absa", "f4", (585, 4), (1, 1)),
            ("ka_mco2", "f4", (9, 19, 4), (1, 1, 1)),
            ("ka_mco", "f4", (9, 19, 4), (1, 1, 1)),
            ("kb_mo3", "f4", (19, 4), (1, 1)),
            ("selfref", "f4", (10, 4), (1, 1)),
            ("forref", "f4", (4, 4), (1, 1)),
        ),
    ),
    "rrlw_kg14": (
        (
            ("fracrefao", "f4", (16,), (1,)),
            ("fracrefbo", "f4", (16,), (1,)),
            ("kao", "f4", (5, 13, 16), (1, 1, 1)),
            ("kbo", "f4", (5, 47, 16), (1, 13, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (4, 16), (1, 1)),
        ),
        (
            ("fracrefa", "f4", (2,), (1,)),
            ("fracrefb", "f4", (2,), (1,)),
            ("ka", "f4", (5, 13, 2), (1, 1, 1)),
            ("absa", "f4", (65, 2), (1, 1)),
            ("kb", "f4", (5, 47, 2), (1, 13, 1)),
            ("absb", "f4", (235, 2), (1, 1)),
            ("selfref", "f4", (10, 2), (1, 1)),
            ("forref", "f4", (4, 2), (1, 1)),
        ),
    ),
    "rrlw_kg15": (
        (
            ("fracrefao", "f4", (16, 9), (1, 1)),
            ("kao", "f4", (9, 5, 13, 16), (1, 1, 1, 1)),
            ("kao_mn2", "f4", (9, 19, 16), (1, 1, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (4, 16), (1, 1)),
        ),
        (
            ("fracrefa", "f4", (2, 9), (1, 1)),
            ("ka", "f4", (9, 5, 13, 2), (1, 1, 1, 1)),
            ("absa", "f4", (585, 2), (1, 1)),
            ("ka_mn2", "f4", (9, 19, 2), (1, 1, 1)),
            ("selfref", "f4", (10, 2), (1, 1)),
            ("forref", "f4", (4, 2), (1, 1)),
        ),
    ),
    "rrlw_kg16": (
        (
            ("fracrefao", "f4", (16, 9), (1, 1)),
            ("fracrefbo", "f4", (16,), (1,)),
            ("kao", "f4", (9, 5, 13, 16), (1, 1, 1, 1)),
            ("kbo", "f4", (5, 47, 16), (1, 13, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (4, 16), (1, 1)),
        ),
        (
            ("fracrefb", "f4", (2,), (1,)),
            ("fracrefa", "f4", (2, 9), (1, 1)),
            ("ka", "f4", (9, 5, 13, 2), (1, 1, 1, 1)),
            ("absa", "f4", (585, 2), (1, 1)),
            ("kb", "f4", (5, 47, 2), (1, 13, 1)),
            ("absb", "f4", (235, 2), (1, 1)),
            ("selfref", "f4", (10, 2), (1, 1)),
            ("forref", "f4", (4, 2), (1, 1)),
        ),
    ),
}

_SW_SPEC = {
    "rrsw_kg16": (
        (
            ("rayl", "f4", None, None),
            ("strrat1", "f4", None, None),
            ("layreffr", "i4", None, None),
            ("kao", "f4", (9, 5, 13, 16), (1, 1, 1, 1)),
            ("kbo", "f4", (5, 47, 16), (1, 13, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (3, 16), (1, 1)),
            ("sfluxrefo", "f4", (16,), (1,)),
        ),
        (
            ("ka", "f4", (9, 5, 13, 6), (1, 1, 1, 1)),
            ("absa", "f4", (585, 6), (1, 1)),
            ("kb", "f4", (5, 47, 6), (1, 13, 1)),
            ("absb", "f4", (235, 6), (1, 1)),
            ("selfref", "f4", (10, 6), (1, 1)),
            ("forref", "f4", (3, 6), (1, 1)),
            ("sfluxref", "f4", (6,), (1,)),
        ),
    ),
    "rrsw_kg17": (
        (
            ("rayl", "f4", None, None),
            ("strrat", "f4", None, None),
            ("layreffr", "i4", None, None),
            ("kao", "f4", (9, 5, 13, 16), (1, 1, 1, 1)),
            ("kbo", "f4", (5, 5, 47, 16), (1, 1, 13, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (4, 16), (1, 1)),
            ("sfluxrefo", "f4", (16, 5), (1, 1)),
        ),
        (
            ("ka", "f4", (9, 5, 13, 12), (1, 1, 1, 1)),
            ("absa", "f4", (585, 12), (1, 1)),
            ("kb", "f4", (5, 5, 47, 12), (1, 1, 13, 1)),
            ("absb", "f4", (1175, 12), (1, 1)),
            ("selfref", "f4", (10, 12), (1, 1)),
            ("forref", "f4", (4, 12), (1, 1)),
            ("sfluxref", "f4", (12, 5), (1, 1)),
        ),
    ),
    "rrsw_kg18": (
        (
            ("rayl", "f4", None, None),
            ("strrat", "f4", None, None),
            ("layreffr", "i4", None, None),
            ("kao", "f4", (9, 5, 13, 16), (1, 1, 1, 1)),
            ("kbo", "f4", (5, 47, 16), (1, 13, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (3, 16), (1, 1)),
            ("sfluxrefo", "f4", (16, 9), (1, 1)),
        ),
        (
            ("ka", "f4", (9, 5, 13, 8), (1, 1, 1, 1)),
            ("absa", "f4", (585, 8), (1, 1)),
            ("kb", "f4", (5, 47, 8), (1, 13, 1)),
            ("absb", "f4", (235, 8), (1, 1)),
            ("selfref", "f4", (10, 8), (1, 1)),
            ("forref", "f4", (3, 8), (1, 1)),
            ("sfluxref", "f4", (8, 9), (1, 1)),
        ),
    ),
    "rrsw_kg19": (
        (
            ("rayl", "f4", None, None),
            ("strrat", "f4", None, None),
            ("layreffr", "i4", None, None),
            ("kao", "f4", (9, 5, 13, 16), (1, 1, 1, 1)),
            ("kbo", "f4", (5, 47, 16), (1, 13, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (3, 16), (1, 1)),
            ("sfluxrefo", "f4", (16, 9), (1, 1)),
        ),
        (
            ("ka", "f4", (9, 5, 13, 8), (1, 1, 1, 1)),
            ("absa", "f4", (585, 8), (1, 1)),
            ("kb", "f4", (5, 47, 8), (1, 13, 1)),
            ("absb", "f4", (235, 8), (1, 1)),
            ("selfref", "f4", (10, 8), (1, 1)),
            ("forref", "f4", (3, 8), (1, 1)),
            ("sfluxref", "f4", (8, 9), (1, 1)),
        ),
    ),
    "rrsw_kg20": (
        (
            ("rayl", "f4", None, None),
            ("layreffr", "i4", None, None),
            ("absch4o", "f4", (16,), (1,)),
            ("kao", "f4", (5, 13, 16), (1, 1, 1)),
            ("kbo", "f4", (5, 47, 16), (1, 13, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (4, 16), (1, 1)),
            ("sfluxrefo", "f4", (16,), (1,)),
        ),
        (
            ("ka", "f4", (5, 13, 10), (1, 1, 1)),
            ("absa", "f4", (65, 10), (1, 1)),
            ("kb", "f4", (5, 47, 10), (1, 13, 1)),
            ("absb", "f4", (235, 10), (1, 1)),
            ("selfref", "f4", (10, 10), (1, 1)),
            ("forref", "f4", (4, 10), (1, 1)),
            ("sfluxref", "f4", (10,), (1,)),
            ("absch4", "f4", (10,), (1,)),
        ),
    ),
    "rrsw_kg21": (
        (
            ("rayl", "f4", None, None),
            ("strrat", "f4", None, None),
            ("layreffr", "i4", None, None),
            ("kao", "f4", (9, 5, 13, 16), (1, 1, 1, 1)),
            ("kbo", "f4", (5, 5, 47, 16), (1, 1, 13, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (4, 16), (1, 1)),
            ("sfluxrefo", "f4", (16, 9), (1, 1)),
        ),
        (
            ("ka", "f4", (9, 5, 13, 10), (1, 1, 1, 1)),
            ("absa", "f4", (585, 10), (1, 1)),
            ("kb", "f4", (5, 5, 47, 10), (1, 1, 13, 1)),
            ("absb", "f4", (1175, 10), (1, 1)),
            ("selfref", "f4", (10, 10), (1, 1)),
            ("forref", "f4", (4, 10), (1, 1)),
            ("sfluxref", "f4", (10, 9), (1, 1)),
        ),
    ),
    "rrsw_kg22": (
        (
            ("rayl", "f4", None, None),
            ("strrat", "f4", None, None),
            ("layreffr", "i4", None, None),
            ("kao", "f4", (9, 5, 13, 16), (1, 1, 1, 1)),
            ("kbo", "f4", (5, 47, 16), (1, 13, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (3, 16), (1, 1)),
            ("sfluxrefo", "f4", (16, 9), (1, 1)),
        ),
        (
            ("ka", "f4", (9, 5, 13, 2), (1, 1, 1, 1)),
            ("absa", "f4", (585, 2), (1, 1)),
            ("kb", "f4", (5, 47, 2), (1, 13, 1)),
            ("absb", "f4", (235, 2), (1, 1)),
            ("selfref", "f4", (10, 2), (1, 1)),
            ("forref", "f4", (3, 2), (1, 1)),
            ("sfluxref", "f4", (2, 9), (1, 1)),
        ),
    ),
    "rrsw_kg23": (
        (
            ("raylo", "f4", (16,), (1,)),
            ("givfac", "f4", None, None),
            ("layreffr", "i4", None, None),
            ("kao", "f4", (5, 13, 16), (1, 1, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (3, 16), (1, 1)),
            ("sfluxrefo", "f4", (16,), (1,)),
        ),
        (
            ("ka", "f4", (5, 13, 10), (1, 1, 1)),
            ("absa", "f4", (65, 10), (1, 1)),
            ("selfref", "f4", (10, 10), (1, 1)),
            ("forref", "f4", (3, 10), (1, 1)),
            ("sfluxref", "f4", (10,), (1,)),
            ("rayl", "f4", (10,), (1,)),
        ),
    ),
    "rrsw_kg24": (
        (
            ("raylao", "f4", (16, 9), (1, 1)),
            ("raylbo", "f4", (16,), (1,)),
            ("strrat", "f4", None, None),
            ("layreffr", "i4", None, None),
            ("abso3ao", "f4", (16,), (1,)),
            ("abso3bo", "f4", (16,), (1,)),
            ("kao", "f4", (9, 5, 13, 16), (1, 1, 1, 1)),
            ("kbo", "f4", (5, 47, 16), (1, 13, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (3, 16), (1, 1)),
            ("sfluxrefo", "f4", (16, 9), (1, 1)),
        ),
        (
            ("ka", "f4", (9, 5, 13, 8), (1, 1, 1, 1)),
            ("absa", "f4", (585, 8), (1, 1)),
            ("kb", "f4", (5, 47, 8), (1, 13, 1)),
            ("absb", "f4", (235, 8), (1, 1)),
            ("selfref", "f4", (10, 8), (1, 1)),
            ("forref", "f4", (3, 8), (1, 1)),
            ("sfluxref", "f4", (8, 9), (1, 1)),
            ("abso3a", "f4", (8,), (1,)),
            ("abso3b", "f4", (8,), (1,)),
            ("rayla", "f4", (8, 9), (1, 1)),
            ("raylb", "f4", (8,), (1,)),
        ),
    ),
    "rrsw_kg25": (
        (
            ("raylo", "f4", (16,), (1,)),
            ("layreffr", "i4", None, None),
            ("abso3ao", "f4", (16,), (1,)),
            ("abso3bo", "f4", (16,), (1,)),
            ("kao", "f4", (5, 13, 16), (1, 1, 1)),
            ("sfluxrefo", "f4", (16,), (1,)),
        ),
        (
            ("ka", "f4", (5, 13, 6), (1, 1, 1)),
            ("absa", "f4", (65, 6), (1, 1)),
            ("sfluxref", "f4", (6,), (1,)),
            ("abso3a", "f4", (6,), (1,)),
            ("abso3b", "f4", (6,), (1,)),
            ("rayl", "f4", (6,), (1,)),
        ),
    ),
    "rrsw_kg26": (
        (
            ("raylo", "f4", (16,), (1,)),
            ("sfluxrefo", "f4", (16,), (1,)),
        ),
        (
            ("sfluxref", "f4", (6,), (1,)),
            ("rayl", "f4", (6,), (1,)),
        ),
    ),
    "rrsw_kg27": (
        (
            ("raylo", "f4", (16,), (1,)),
            ("scalekur", "f4", None, None),
            ("layreffr", "i4", None, None),
            ("kao", "f4", (5, 13, 16), (1, 1, 1)),
            ("kbo", "f4", (5, 47, 16), (1, 13, 1)),
            ("sfluxrefo", "f4", (16,), (1,)),
        ),
        (
            ("ka", "f4", (5, 13, 8), (1, 1, 1)),
            ("absa", "f4", (65, 8), (1, 1)),
            ("kb", "f4", (5, 47, 8), (1, 13, 1)),
            ("absb", "f4", (235, 8), (1, 1)),
            ("sfluxref", "f4", (8,), (1,)),
            ("rayl", "f4", (8,), (1,)),
        ),
    ),
    "rrsw_kg28": (
        (
            ("rayl", "f4", None, None),
            ("strrat", "f4", None, None),
            ("layreffr", "i4", None, None),
            ("kao", "f4", (9, 5, 13, 16), (1, 1, 1, 1)),
            ("kbo", "f4", (5, 5, 47, 16), (1, 1, 13, 1)),
            ("sfluxrefo", "f4", (16, 5), (1, 1)),
        ),
        (
            ("ka", "f4", (9, 5, 13, 6), (1, 1, 1, 1)),
            ("absa", "f4", (585, 6), (1, 1)),
            ("kb", "f4", (5, 5, 47, 6), (1, 1, 13, 1)),
            ("absb", "f4", (1175, 6), (1, 1)),
            ("sfluxref", "f4", (6, 5), (1, 1)),
        ),
    ),
    "rrsw_kg29": (
        (
            ("rayl", "f4", None, None),
            ("layreffr", "i4", None, None),
            ("absh2oo", "f4", (16,), (1,)),
            ("absco2o", "f4", (16,), (1,)),
            ("kao", "f4", (5, 13, 16), (1, 1, 1)),
            ("kbo", "f4", (5, 47, 16), (1, 13, 1)),
            ("selfrefo", "f4", (10, 16), (1, 1)),
            ("forrefo", "f4", (4, 16), (1, 1)),
            ("sfluxrefo", "f4", (16,), (1,)),
        ),
        (
            ("ka", "f4", (5, 13, 12), (1, 1, 1)),
            ("absa", "f4", (65, 12), (1, 1)),
            ("kb", "f4", (5, 47, 12), (1, 13, 1)),
            ("absb", "f4", (235, 12), (1, 1)),
            ("selfref", "f4", (10, 12), (1, 1)),
            ("forref", "f4", (4, 12), (1, 1)),
            ("sfluxref", "f4", (12,), (1,)),
            ("absh2o", "f4", (12,), (1,)),
            ("absco2", "f4", (12,), (1,)),
        ),
    ),
}
