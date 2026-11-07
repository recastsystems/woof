"""Multiscale Kain-Fritsch cumulus (WRF ``cu_physics = 11``).

The column scheme is ``woof/core/kernels/mskf.cu``, a WRF-order
transcription of WRF v4.6.1 ``phys/module_cu_mskf.F`` checked word for word
against the column oracle in ``tools/mskf_wrf461_oracle``.  This module owns
the device launch (workspace, tiles, the KF lookup table) and the
low-level ``cu_physics = 11`` column launcher. Forecast admission uses the
registry column contract and shared cumulus consumer.

Array layout is ``(nz, ny, nx)`` with x fastest, the same as ``kf.py``.

What is reused from ``cu_physics = 1`` and what is not.  The six thermodynamic
helpers MSKF calls (TPMIX2, DTFRZNEW, CONDLOAD, PROF5, TPMIX2DD, ENVIRTHT) and
the lookup table share arithmetic with the corresponding routines in ``module_cu_kfeta.F``;
``tools/mskf_wrf461_oracle/reuse_map.py`` records the exact source differences
(including the extra QSu output on TPMIX2); the table is therefore read from the same
``woof/data/kf_lutab`` path through :func:`woof.core.kf.load_kf_table`, and
the phase flags follow the selected package's declared moisture state.  The column kernel is
not shared: both kernels now preserve native WRF operation order, but
``MSKF_eta_PARA`` diverges from ``KF_eta_PARA`` on 66 hunks anyway (the
scale factor, the trigger, the LCL radius, the time scale, the closure floor,
the cloud fraction, the momentum transport).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from woof.core.device_cache import cuda_cache

import numpy as np

DTYPE = np.float32
_TPB = 128
_LANES = 32
#: Workspace slots per column; must equal MSKF_FSLOTS / MSKF_DSLOTS /
#: MSKF_ASLOTS in kernels/mskf.cu (tests/test_mskf_kernel.py re-reads them).
MSKF_FSLOTS = 101
MSKF_DSLOTS = 62
MSKF_ASLOTS = 512
#: MSKF_KMAX in kernels/mskf.cu: a refusal ceiling, nothing is sized by it.
VERTICAL_LEVEL_BOUNDS = (8, 160)
#: Columns in flight per streaming multiprocessor.
MSKF_TILE_COLUMNS_PER_SM = 64

# Status bits the kernel raises (kernels/mskf.cu).  Each is a place where WRF
# stops the model with wrf_error_fatal; the port refuses at the same place,
# with WRF's own words, instead of returning a column WRF never produces.
MSKF_STATUS_MESSAGES = {
    1: "MSKF: QG < 0 at the lowest model level "
       "(WRF module_cu_mskf.F:6053 'QG, QG(NK).LT.0' stops the model)",
    2: "MSKF: the compensating-subsidence mass budget does not close "
       "(WRF module_cu_mskf.F:6545 'KAIN-FRITSCH, istop=1, diags' stops "
       "the model)",
    4: "MSKF: this microphysics phase combination is not allowed "
       "(WRF module_cu_mskf.F:6684)",
    8: "MSKF: the aerosol path found cloud bottom on the ground "
       "(WRF module_cu_mskf.F:5088 stops the model)",
    16: "MSKF: the aerosol path found cloud top above the model top "
        "(WRF module_cu_mskf.F:5093 stops the model)",
    32: "MSKF: incorrect ice nucleation number on the aerosol path "
        "(WRF module_cu_mskf.F:3228 stops the model)",
    64: "MSKF requires 8 <= nz <= 160 (the kernel's refusal ceiling)",
}

OUTPUT_3D = ("rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten",
             "rqscuten", "rucuten", "rvcuten", "cldfra_dp", "cldfra_sh",
             "qc_cu", "qi_cu", "qr_cu", "qs_cu", "nc_cu", "ni_cu", "nr_cu",
             "ns_cu", "ccn_cu", "efcs", "efis", "efss", "udr_kf", "ddr_kf",
             "uer_kf", "der_kf")
OUTPUT_2D = ("raincv", "pratec", "nca", "cutop", "cubot", "ainc_frac",
             "timec_kf")
# WRF Registry.EM_COMMON:3245 and :3209 allocate these only for the
# corresponding package. GPUWM retains zero extensions when a package is off.
AEROSOL_OUTPUTS = ("qr_cu", "qs_cu", "nc_cu", "ni_cu", "nr_cu", "ns_cu",
                   "ccn_cu", "efcs", "efis", "efss", "ainc_frac")
EDRATE_OUTPUTS = ("udr_kf", "ddr_kf", "uer_kf", "der_kf", "timec_kf")
INPUT_3D = ("u", "v", "t", "qv", "p", "rho", "pi", "dz8w", "w0avg")
INPUT_2D = ("hfx", "ust", "pblh", "zol")


@dataclass(frozen=True)
class MskfPhase:
    """The WRF feedback switches MSKF_CPS reads (module_cu_mskf.F:3956-3981
    and :6646-6685): ``warm_rain`` from mp_init, F_QR/F_QI/F_QS from the
    microphysics package's moist inventory."""

    warm_rain: bool
    f_qr: bool
    f_qi: bool
    f_qs: bool


def workspace_floats(nz: int, columns: int) -> int:
    blocks = (int(columns) + _TPB - 1) // _TPB
    return blocks * MSKF_FSLOTS * (int(nz) + 2) * _TPB


def workspace_doubles(nz: int, columns: int) -> int:
    blocks = (int(columns) + _TPB - 1) // _TPB
    return blocks * MSKF_DSLOTS * (int(nz) + 2) * _TPB


def workspace_aerosol_doubles(nz: int, columns: int) -> int:
    blocks = (int(columns) + _TPB - 1) // _TPB
    return blocks * MSKF_ASLOTS * (int(nz) + 2) * _TPB


def tile_columns(ncol: int, multiprocessors: int, *, aerosol: bool = False) -> int:
    """Columns kept in flight: the workspace is charged per column in flight."""
    from woof.numerics import is_strict

    # Positive activation spills heavily: expose one 128-thread block per SM.
    # Keep native replay and the option-off launch geometry unchanged.
    columns_per_sm = (2 if aerosol and not is_strict() else 1) * MSKF_TILE_COLUMNS_PER_SM
    return int(min(int(ncol), max(1, int(multiprocessors)) * columns_per_sm))


def workspace_bytes(nz: int, columns: int, *, aerosol: bool) -> int:
    """Device bytes the workspace of one launch tile holds."""
    total = (workspace_floats(nz, columns) * 4
             + workspace_doubles(nz, columns) * 8)
    if aerosol:
        total += workspace_aerosol_doubles(nz, columns) * 8
    return int(total)


@lru_cache(maxsize=None)
def _device_table_on(_device: int):
    import cupy as cp

    from woof.core.kf import load_kf_table

    table = load_kf_table()
    return tuple(cp.asarray(value) for value in
                 (table.temperature, table.qsat, table.thetae_base,
                  table.log_ratio))


def _device_table():
    import cupy as cp

    return _device_table_on(cp.cuda.runtime.getDevice())


@cuda_cache(maxsize=None, ready=True)
def _aerosol_constants():
    """The 224 invariant module-array cells, computed on this device once."""
    import cupy as cp
    from woof.core.kernels.no_ftz import get_kernel_no_ftz
    table = cp.empty(224 * _LANES, dtype=cp.float64)
    get_kernel_no_ftz("mskf", "mskf_aerosol_init")((1,), (1,), (table,))
    return table


def _multiprocessors() -> int:
    import cupy as cp

    return int(cp.cuda.Device().attributes["MultiProcessorCount"])


def update_w0avg(w0avg, w, *, dt: float, cudt_minutes: float, stepcu: int,
                 adaptive: bool) -> None:
    """MSKF_CPS's W0AVG running mean (module_cu_mskf.F:3649-3681), in place.

    Fixed step: ``(W0AVG*(TST-1) + W0*1)/TST`` with ``TST = 2*STEPCU``.
    Adaptive step: ``(W0AVG*(2*MAX(CUDT*60,DT)-DT) + W0*DT)/(2*MAX(CUDT*60,DT))``.
    Every factor is formed in float32 exactly as the Fortran forms it.
    """
    import cupy as cp

    from woof.core.kernels.no_ftz import get_kernel_no_ftz as get_kernel

    nz = int(w0avg.shape[0])
    ncol = int(np.prod(w0avg.shape[1:]))
    f32 = np.float32
    if adaptive:
        span = max(f32(f32(cudt_minutes) * f32(60.0)), f32(dt))
        fctr = f32(f32(f32(2.0) * span) - f32(dt))
        w0f = f32(dt)
        den = f32(f32(2.0) * span)
    else:
        tst = f32(int(stepcu) * 2)
        fctr = f32(tst - f32(1.0))
        w0f = f32(1.0)
        den = tst
    blocks = (nz * ncol + 255) // 256
    get_kernel("mskf", "mskf_w0avg")(
        (blocks,), (256,),
        (w0avg, cp.ascontiguousarray(w), fctr, w0f, den, np.int32(nz),
         np.int32(ncol)))


def validate_layout(inputs: dict, outputs: dict) -> tuple[int, int]:
    """Refuse layouts that would let CUDA read or write outside a field."""
    shape = inputs["t"].shape
    if len(shape) != 3 or any(int(n) <= 0 for n in shape):
        raise ValueError("MSKF t must have nonempty shape (nz, ny, nx)")
    nz = int(shape[0])
    lo, hi = VERTICAL_LEVEL_BOUNDS
    if not lo <= nz <= hi:
        raise ValueError(MSKF_STATUS_MESSAGES[64] + f", got nz={nz}")
    for fields, names, expected, dtype in (
        (inputs, INPUT_3D, shape, DTYPE),
        (inputs, INPUT_2D, shape[1:], DTYPE),
        (outputs, OUTPUT_3D, shape, DTYPE),
        (outputs, OUTPUT_2D, shape[1:], DTYPE),
        (outputs, ("cu_act",), shape[1:], np.int32),
    ):
        for name in names:
            a = fields[name]
            if a.shape != expected or a.dtype != dtype or not a.flags.c_contiguous:
                raise ValueError(f"MSKF {name} must be C-contiguous {np.dtype(dtype)} "
                                 f"with shape {expected}; incompatible storage "
                                 "would let the CUDA column index outside the field")
    return nz, int(np.prod(shape[1:]))


def launch_mskf(inputs: dict, outputs: dict, *, dx: float, dt: float,
                cudt_minutes: float, phase: MskfPhase, aercu_opt: int = 0,
                aercu_fct: float = 1.0, kf_edrates: int = 1,
                aerocu=None) -> int:
    """One MSKF_CPS call over every column, writing ``outputs`` in place.

    ``inputs``: INPUT_3D (nz, ny, nx) and INPUT_2D (ny, nx) float32 device
    arrays; ``w0avg`` must already carry this call's running-mean update.
    ``outputs``: every OUTPUT_3D/OUTPUT_2D name, float32, plus ``cu_act``
    (ny, nx) int32.  WRF's contract is reproduced exactly: a column held by
    ``nca >= 0.5*dt`` is not written at all, an active column has its
    cumulus fields reset and recomputed, and ``nca`` is read and updated in
    place.  Returns the OR of the per-column status bits (0 on success); the
    caller refuses with :data:`MSKF_STATUS_MESSAGES`.
    """
    import cupy as cp

    from woof.core.kernels.no_ftz import get_kernel_no_ftz as get_kernel
    from woof.core.kf import load_kf_table

    nz, ncol = validate_layout(inputs, outputs)
    for name, value, allow_zero in (("dx", dx, False), ("dt", dt, False),
                                    ("cudt_minutes", cudt_minutes, True)):
        if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
            raise ValueError(f"MSKF {name} must be finite and "
                             + ("nonnegative" if allow_zero else "positive")
                             + "; invalid values break the scale or time denominator")
    if int(aercu_opt) > 0:
        if aerocu is None:
            raise ValueError("MSKF aercu_opt > 0 needs the aerocu aerosol "
                             "state (10 species on model levels)")
    if int(aercu_opt) > 0 and (aerocu.shape != (10, nz) + tuple(inputs["t"].shape[1:]) or aerocu.dtype != DTYPE):
        raise ValueError("MSKF aerocu must have float32 shape (10,nz,ny,nx); wrong species or grid storage would index outside the aerosol state")
    table = load_kf_table()
    ttab, qstab, the0k, alu = _device_table()
    defines = (("GPUWM_MSKF_NO_AEROSOL", 1),) if int(aercu_opt) == 0 else ()
    kernel = get_kernel("mskf", "mskf_column", defines)
    from woof.numerics import is_strict
    constants = ((_aerosol_constants(),)
                 if int(aercu_opt) > 0 and not is_strict() else ())
    tile = tile_columns(ncol, _multiprocessors(), aerosol=int(aercu_opt) > 0)
    wsf = cp.empty(workspace_floats(nz, tile), dtype=cp.float32)
    wsd = cp.empty(workspace_doubles(nz, tile), dtype=cp.float64)
    wsa = (cp.empty(max(1, workspace_aerosol_doubles(nz, tile)),
                    dtype=cp.float64) if int(aercu_opt) > 0
           else cp.empty(1, dtype=cp.float64))
    status = cp.zeros(1, dtype=cp.int32)
    aer = (cp.ascontiguousarray(aerocu, dtype=cp.float32)
           if aerocu is not None else cp.zeros(1, dtype=cp.float32))
    args_in = tuple(inputs[n] for n in INPUT_3D) + tuple(
        cp.ascontiguousarray(inputs[n], dtype=cp.float32) for n in INPUT_2D)
    args_out = tuple(outputs[n] for n in OUTPUT_3D[:26]) + (
        outputs["raincv"], outputs["pratec"], outputs["nca"],
        outputs["cutop"], outputs["cubot"], outputs["ainc_frac"],
        outputs["timec_kf"], outputs["cu_act"], status)
    for col0 in range(0, ncol, tile):
        span = min(tile, ncol - col0)
        blocks = (span + _TPB - 1) // _TPB
        kernel((blocks,), (_TPB,), args_in + (aer,) + args_out + (
            ttab, qstab, the0k, alu, DTYPE(table.pressure_top),
            DTYPE(table.pressure_reciprocal), DTYPE(table.thetae_reciprocal),
            wsf, wsd, wsa, DTYPE(dt), DTYPE(dx), DTYPE(cudt_minutes),
            DTYPE(aercu_fct), np.int32(phase.warm_rain),
            np.int32(phase.f_qr), np.int32(phase.f_qi), np.int32(phase.f_qs),
            np.int32(aercu_opt), np.int32(kf_edrates), np.int32(nz),
            np.int32(ncol), np.int32(col0), np.int32(span)) + constants)
    del wsf, wsd, wsa
    from woof.core.health_ledger import read_status
    return read_status(status, site="mskf", describe=_raise_deferred_status)


def raise_on_status(status: int) -> None:
    if status:
        raise FloatingPointError("; ".join(
            msg for bit, msg in MSKF_STATUS_MESSAGES.items() if status & bit))


def _raise_deferred_status(status: int) -> None:
    from woof.core.health_ledger import deferred_note
    if status:
        message = "; ".join(msg for bit, msg in MSKF_STATUS_MESSAGES.items() if status & bit)
        raise FloatingPointError(message + deferred_note())


__all__ = ["DTYPE", "INPUT_2D", "INPUT_3D", "MSKF_DSLOTS", "MSKF_FSLOTS",
           "MSKF_ASLOTS", "MSKF_STATUS_MESSAGES", "MskfPhase", "OUTPUT_2D",
           "OUTPUT_3D", "VERTICAL_LEVEL_BOUNDS", "launch_mskf",
           "raise_on_status", "tile_columns", "update_w0avg",
           "workspace_bytes", "workspace_doubles", "workspace_floats", "validate_layout"]
