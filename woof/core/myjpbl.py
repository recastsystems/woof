"""MYJ (Mellor-Yamada-Janjic level 2.5) PBL, ``bl_pbl_physics = 2``.

CUDA launcher around :mod:`woof.core.kernels.myjpbl`, the device port of
the byte-frozen WRF v4.6.1 ``phys/module_bl_myjpbl.F`` (the float32 CPU
authority ``woof.verify.myj_ref.np_myjpbl_column`` is its tolerance
twin).

WHAT THIS SCHEME OWNS.  MYJ predicts TKE (``q**2 = 2*TKE``) with Janjic's
extension of Mellor-Yamada level 2.5, builds the mixing length from it,
forms the exchange coefficients, and then does its OWN implicit vertical
diffusion of heat, moisture, cloud water/ice and momentum -- so the scheme
returns finished TENDENCIES, not diffusivities for someone else to apply.
That is why its surface pairing is not negotiable: the lower boundary of
every one of those solves is ``AKHS``/``AKMS``/``THZ0``/``QZ0``/``UZ0``/
``VZ0``, which only the Eta surface layer produces.

COLUMN LIMIT.  The kernel keeps the whole column in per-thread local
arrays, as YSU's does, so it carries the same compile-time ceiling.  A
deeper column is refused here rather than silently truncated.

Conformance status: BIT-IDENTICAL to WRF v4.6.1's MYJPBL on the
176-column oracle (tools/myjpbl_wrf461_oracle,
tests/test_myjpbl_wrf461_parity.py) under the strict build; registry
maturity stays implemented-unverified because no matched forecast
trajectory exists.  ``surface["ht"]`` is WRF's HT: MYJPBL's interface
heights start at the terrain height (ZINT(KTE+1)=HT, :312), and the float32
words depend on it.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from woof.core.kernels import get_kernel
from woof.core.state import DTYPE
from woof.physics_vertical_contract import (
    outside_vertical_bounds, refuse_vertical_levels)

_TPB = 128

#: Deepest column ``kernels/myjpbl.cu``'s per-thread arrays hold
#: (``MYJ_KMAX``).  Kept in step with the kernel by
#: ``tests/test_myj_port.py``.
MYJ_MAX_COLUMN_LEVELS = 128

#: Shallowest column the tridiagonal chain is defined on.  VDIFQ solves
#: over ``LMH-2`` interior rows (module_bl_myjpbl.F:1376-1403) and MIXLEN's
#: profile smoothing reads ``REL(K-1)``/``REL(K+1)``; four levels is the
#: floor at which every one of those slices is non-empty.
MYJ_MIN_COLUMN_LEVELS = 4

#: Full bottom-up ``(nz, ny, nx)`` inputs, in kernel argument order.
_COLUMN_INPUTS = ("dz", "u", "v", "t", "th", "exner", "qv", "qc", "qi", "p")
#: Read-only surface inputs, in kernel argument order.
_SURFACE_INPUTS = ("psfc", "ust", "tsk", "chklowq", "xland", "sice", "snow",
                   "akhs", "akms", "elflx", "uz0", "vz0", "ht")
#: WRF INOUT surface state MYJPBL rewrites (:542, :556-566, :852).
MYJ_PBL_INOUT = ("thz0", "qz0", "qsfc", "ct")
#: Bottom-up ``(nz, ny, nx)`` outputs, in kernel argument order.
MYJ_PBL_COLUMN_OUTPUTS = ("rublten", "rvblten", "rthblten", "rqvblten",
                          "rqcblten", "rqiblten", "el_myj", "exch_h")
#: Surface outputs, in kernel argument order.  ``kpbl`` is int32.
MYJ_PBL_SURFACE_OUTPUTS = ("pblh", "kpbl", "mixht")


def launch_myj_pbl(columns, surface, state, tke, outputs, *,
                   dtturbl: float, flqi: bool) -> None:
    """Run one MYJ PBL step in place.

    ``tke`` is the scheme's carried TKE_MYJ column, read as ``2*TKE`` on
    entry and rewritten as ``0.5*q2`` on exit (module_bl_myjpbl.F:377,
    :462).  ``flqi`` selects WRF's ``PRESENT(QCI)`` arm: with ice the
    species stack is four rows and RQIBLTEN is produced, without it three
    and RQIBLTEN stays zero (:264-273).
    """
    shape = surface["psfc"].shape
    nz = int(columns["dz"].shape[0])
    bounds = (MYJ_MIN_COLUMN_LEVELS, MYJ_MAX_COLUMN_LEVELS)
    if outside_vertical_bounds(nz, bounds):
        raise refuse_vertical_levels(
            "MYJ PBL", bounds, nz,
            breakage=(
                "the kernel holds at most "
                f"{MYJ_MAX_COLUMN_LEVELS} levels per column (MYJ_KMAX in "
                "kernels/myjpbl.cu) -- raise the define and re-measure "
                "rather than truncating -- and below the floor its TKE "
                "tridiagonal solve has no interior rows."))
    column_arrays = []
    for name in _COLUMN_INPUTS:
        array = columns.get(name)
        if array is None:
            if name != "qi" or flqi:
                raise KeyError(f"the MYJ PBL requires a {name!r} column")
            # WRF's OPTIONAL QCI is absent: the kernel never reads the
            # pointer when flqi is false, but a null argument is not
            # something CuPy will marshal, so pass the qc column and let
            # the flag hold it out.
            array = columns["qc"]
        column_arrays.append(array)
    for array in (*column_arrays, tke,
                  *(outputs[name] for name in MYJ_PBL_COLUMN_OUTPUTS)):
        if (array.shape != (nz, *shape) or array.dtype != DTYPE
                or not array.flags.c_contiguous):
            raise ValueError(
                "launch_myj_pbl requires contiguous float32 "
                f"({nz}, {shape[0]}, {shape[1]}) columns")
    surface_arrays = ([surface[name] for name in _SURFACE_INPUTS]
                      + [state[name] for name in MYJ_PBL_INOUT]
                      + [outputs[name] for name in MYJ_PBL_SURFACE_OUTPUTS])
    for array in surface_arrays:
        expected = cp.int32 if array is outputs["kpbl"] else DTYPE
        if (array.shape != shape or array.dtype != expected
                or not array.flags.c_contiguous):
            raise ValueError("launch_myj_pbl requires same-shape contiguous "
                             "surface arrays (float32, kpbl int32)")
    n = int(np.prod(shape))
    blocks = (n + _TPB - 1) // _TPB
    kernel = get_kernel("myjpbl", "myjpbl_column")
    kernel((blocks,), (_TPB,),
           tuple(column_arrays) + (tke,)
           + tuple(surface[name] for name in _SURFACE_INPUTS)
           + tuple(state[name] for name in MYJ_PBL_INOUT)
           + tuple(outputs[name] for name in MYJ_PBL_COLUMN_OUTPUTS)
           + tuple(outputs[name] for name in MYJ_PBL_SURFACE_OUTPUTS)
           + (DTYPE(dtturbl), np.int32(bool(flqi)), np.int32(nz),
              np.int32(n)))


def myj_pbl_step(columns, surface, state, tke, *,
                 dtturbl: float, flqi: bool) -> dict:
    """Allocate MYJ's output set, run one step, and return the outputs.

    The returned dict carries the WRF names the kernel writes plus the
    ``du/dv/dtheta/dqv/dqc/dqi`` aliases :func:`couple_ysu_tendencies`
    consumes -- the SAME device arrays under both keys, not copies, so a
    caller cannot mass-couple one spelling while reading the other.
    """
    shape = surface["psfc"].shape
    nz = int(columns["dz"].shape[0])
    outputs = {name: cp.empty((nz, *shape), dtype=DTYPE)
               for name in MYJ_PBL_COLUMN_OUTPUTS}
    outputs["pblh"] = cp.empty(shape, dtype=DTYPE)
    outputs["mixht"] = cp.empty(shape, dtype=DTYPE)
    outputs["kpbl"] = cp.empty(shape, dtype=cp.int32)
    launch_myj_pbl(columns, surface, state, tke, outputs,
                   dtturbl=dtturbl, flqi=flqi)
    outputs.update(
        du=outputs["rublten"], dv=outputs["rvblten"],
        dtheta=outputs["rthblten"], dqv=outputs["rqvblten"],
        dqc=outputs["rqcblten"],
        # WRF's phy_bl_ten adds RQIBLTEN exactly when the moist set
        # carries ice (module_physics_addtendc.F, IF(F_QI)); without ice
        # MYJ mixed three species and the row is a published zero, so it
        # is withheld rather than coupled as a real ice tendency.
        dqi=outputs["rqiblten"] if flqi else None)
    if outputs["dqi"] is None:
        del outputs["dqi"]
    return outputs


def validate_myj_pbl_outputs(outputs) -> str | None:
    """Name the first non-finite MYJ tendency field, or ``None``.

    Host-side and deliberately cheap: the four tendency planes plus the
    published TKE-derived diagnostics are what a downstream integrator
    consumes, and a NaN in any of them must name its field rather than
    silently advance the state.  Unlike Shin-Hong, MYJ has no known WRF
    0/0 of its own, so a NaN here is a defect until an oracle campaign
    proves otherwise.
    """
    for name in ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
                 "rqiblten", "exch_h", "el_myj", "pblh", "mixht"):
        array = outputs[name]
        if not bool(cp.all(cp.isfinite(array))):
            return name
    return None


# ===========================================================================
# QNSE (bl_pbl_physics = 4) and its EDMF shallow mass flux (mfshconv = 1).
#
# QNSE is a member of this family, not a separate launcher: WRF's
# module_bl_qnsepbl.F is MYJ's column with three subroutines rewritten
# (MIXLEN's stable length, DIFCOF's QNSE stability functions, PRODQ2's
# explicit production/dissipation) and the rest -- the flip, VDIFQ, VDIFH,
# VDIFV, the THZ0/QZ0/QSFC surface rewrite -- MYJ's, with the differences
# recorded in kernels/qnsepbl.cu.  The device code is that kernel; this is
# its marshalling, the same as launch_myj_pbl above.
# ===========================================================================

#: Full bottom-up QNSEPBL column inputs, in kernel argument order.
QNSE_COLUMN_INPUTS = ("dz", "u", "v", "t", "th", "exner", "qv", "qc", "p")
#: QNSEPBL surface inputs, in kernel argument order.  ``ht`` seeds the
#: interface heights as WRF seeds them (module_bl_qnsepbl.F:273); ``hfx``
#: builds THSK (:442-451); ``corf`` is the stable length's Coriolis
#: parameter (:869).
QNSE_SURFACE_INPUTS = ("ht", "psfc", "ust", "chklowq", "xland", "sice",
                       "snow", "akhs", "akms", "elflx", "hfx", "uz0", "vz0",
                       "corf")
#: WRF INOUT surface state QNSEPBL rewrites (:489, :503-513, :750).
QNSE_PBL_INOUT = ("thz0", "qz0", "qsfc", "ct")
#: Bottom-up outputs.  The five tendencies are ACCUMULATED onto the slot
#: (:543-545, :649-650); EXCH_H/EXCH_M keep their lowest level (:393-400).
QNSE_PBL_COLUMN_OUTPUTS = ("rublten", "rvblten", "rthblten", "rqvblten",
                           "rqcblten", "el_myj", "exch_h", "exch_m")
QNSE_PBL_SURFACE_OUTPUTS = ("pblh", "kpbl")

#: MFSHCONVPBL column inputs, in kernel argument order (QC and PINT are
#: arguments of the Fortran but never read, module_bl_mfshconvpbl.F:170-227).
MFSHCONV_COLUMN_INPUTS = ("dz", "rho", "p", "th", "exner", "qv", "u", "v")
MFSHCONV_SURFACE_INPUTS = ("hfx", "qfx")
#: The mass flux's carried RC_MF and published EDKF set: one source of
#: truth, the runtime-free inventory the VRAM estimate prices from.
from woof.core.physics_inventory import (  # noqa: E402
    MFSHCONV_OUTPUTS_3D, MFSHCONV_STATE_3D)
#: What WRF hands QNSEPBL through u_phytmp/v_phytmp and QNSEPBL never
#: reads (:337-343 load them; nothing else touches them).
MFSHCONV_DIAGNOSTICS_3D = ("wthv_mf", "lm_bl89")

_QNSE_TENDENCIES = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten")


def _require_planes(arrays, nz, shape, who):
    for array in arrays:
        if (array.shape != (nz, *shape) or array.dtype != DTYPE
                or not array.flags.c_contiguous):
            raise ValueError(
                f"{who} requires contiguous float32 ({nz}, {shape[0]}, "
                f"{shape[1]}) columns")


def launch_qnse_pbl(columns, surface, state, tke, outputs, *,
                    dtturbl: float, mfshconv: bool = False,
                    mf_state=None, mf_outputs=None,
                    diagnostics=None, native_tile: bool = True) -> None:
    """Run one QNSE PBL step in place, the mass flux first when asked.

    WRF's order (module_pbl_driver.F:1515-1557): the five BL tendencies
    are the driver's zeroed rows (:1028-1043); MFSHCONVPBL, when
    ``mfshconv``, ASSIGNS its rates to them; QNSEPBL then ADDS its own.
    ``tke`` is TKE_PBL, read by both and rewritten by QNSEPBL (:333, :429).
    ``mf_state`` carries ``rc_mf``; ``mf_outputs`` receives the EDKF set;
    ``diagnostics`` (optional) receives ``wthv_mf``/``lm_bl89``.
    """
    shape = surface["psfc"].shape
    nz = int(columns["dz"].shape[0])
    bounds = (MYJ_MIN_COLUMN_LEVELS, MYJ_MAX_COLUMN_LEVELS)
    if outside_vertical_bounds(nz, bounds):
        raise refuse_vertical_levels(
            "QNSE PBL", bounds, nz,
            breakage=(
                "the kernel holds at most "
                f"{MYJ_MAX_COLUMN_LEVELS} levels per column (QNSE_KMAX in "
                "kernels/qnsepbl.cu) -- raise the define and re-measure "
                "rather than truncating -- and below the floor its TKE "
                "tridiagonal solve has no interior rows."))
    _require_planes([columns[name] for name in QNSE_COLUMN_INPUTS]
                    + [tke] + [outputs[name] for name in QNSE_PBL_COLUMN_OUTPUTS],
                    nz, shape, "launch_qnse_pbl")
    for array in ([surface[name] for name in QNSE_SURFACE_INPUTS]
                  + [state[name] for name in QNSE_PBL_INOUT]
                  + [outputs[name] for name in QNSE_PBL_SURFACE_OUTPUTS]):
        expected = cp.int32 if array is outputs["kpbl"] else DTYPE
        if (array.shape != shape or array.dtype != expected
                or not array.flags.c_contiguous):
            raise ValueError("launch_qnse_pbl requires same-shape contiguous "
                             "surface arrays (float32, kpbl int32)")
    n = int(np.prod(shape))
    blocks = (n + _TPB - 1) // _TPB
    if mfshconv:
        if mf_state is None or mf_outputs is None:
            raise ValueError("mfshconv = 1 needs the rc_mf state and the "
                             "EDKF output set")
        _require_planes([columns[name] for name in MFSHCONV_COLUMN_INPUTS]
                        + [mf_state[name] for name in MFSHCONV_STATE_3D]
                        + [mf_outputs[name] for name in MFSHCONV_OUTPUTS_3D],
                        nz, shape, "the EDMF mass flux")
        diag = diagnostics is not None
        if diag:
            _require_planes([diagnostics[name]
                             for name in MFSHCONV_DIAGNOSTICS_3D],
                            nz, shape, "the EDMF diagnostics")
            planes = tuple(diagnostics[name]
                           for name in MFSHCONV_DIAGNOSTICS_3D)
        else:
            # Never written when diag == 0; CuPy cannot marshal a null.
            planes = (outputs["el_myj"], outputs["el_myj"])
        kernel = get_kernel("qnsepbl", "mfshconv_column")
        args = (tuple(columns[name] for name in MFSHCONV_COLUMN_INPUTS) + (tke,)
            + tuple(surface[name] for name in MFSHCONV_SURFACE_INPUTS)
            + (mf_state["rc_mf"],)
            + tuple(outputs[name] for name in _QNSE_TENDENCIES)
            + tuple(mf_outputs[name] for name in MFSHCONV_OUTPUTS_3D)
            + planes
            + (DTYPE(dtturbl), np.int32(diag), np.int32(nz), np.int32(n)))
        tile_top = mf_state.get("_tile_top")
        if tile_top is None:
            tile_top = mf_state["_tile_top"] = cp.empty(1, dtype=cp.int32)
        if native_tile:
            tile_top.fill(0)
            kernel((blocks,), (_TPB,), args + (tile_top, np.int32(1)))
            kernel((blocks,), (_TPB,), args + (tile_top, np.int32(2)))
        else:
            # The historical fixture invokes native WRF once per column.
            kernel((blocks,), (_TPB,), args + (tile_top, np.int32(0)))
    else:
        for name in _QNSE_TENDENCIES:
            outputs[name].fill(0)
    get_kernel("qnsepbl", "qnsepbl_column")(
        (blocks,), (_TPB,),
        tuple(columns[name] for name in QNSE_COLUMN_INPUTS) + (tke,)
        + tuple(surface[name] for name in QNSE_SURFACE_INPUTS)
        + tuple(state[name] for name in QNSE_PBL_INOUT)
        + tuple(outputs[name] for name in QNSE_PBL_COLUMN_OUTPUTS)
        + tuple(outputs[name] for name in QNSE_PBL_SURFACE_OUTPUTS)
        + (DTYPE(dtturbl), np.int32(nz), np.int32(n)))


def qnse_pbl_step(columns, surface, state, tke, *, dtturbl: float,
                  exch_h, exch_m, mfshconv: bool = False, mf_state=None,
                  mf_outputs=None, diagnostics=None, native_tile: bool = True) -> dict:
    """Allocate QNSE's outputs, run one step, return them with the aliases
    :func:`couple_ysu_tendencies` consumes (the same arrays, not copies).

    ``exch_h``/``exch_m`` are the carried planes: QNSEPBL rewrites levels
    1..nz-1 and leaves level 0 as it found it (module_bl_qnsepbl.F:393-400).
    """
    shape = surface["psfc"].shape
    nz = int(columns["dz"].shape[0])
    outputs = {name: cp.empty((nz, *shape), dtype=DTYPE)
               for name in _QNSE_TENDENCIES + ("el_myj",)}
    outputs["exch_h"] = exch_h
    outputs["exch_m"] = exch_m
    outputs["pblh"] = cp.empty(shape, dtype=DTYPE)
    outputs["kpbl"] = cp.empty(shape, dtype=cp.int32)
    launch_qnse_pbl(columns, surface, state, tke, outputs, dtturbl=dtturbl,
                    mfshconv=mfshconv, mf_state=mf_state,
                    mf_outputs=mf_outputs, diagnostics=diagnostics,
                    native_tile=native_tile)
    outputs.update(du=outputs["rublten"], dv=outputs["rvblten"],
                   dtheta=outputs["rthblten"], dqv=outputs["rqvblten"],
                   dqc=outputs["rqcblten"])
    return outputs


def validate_qnse_pbl_outputs(outputs) -> str | None:
    """Check QNSE outputs through the driver's deferred device ledger."""
    from woof.core import health_ledger
    for name in ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
                 "exch_h", "exch_m", "el_myj", "pblh"):
        if not health_ledger.check_finite(
                outputs[name], site=f"qnse:{name}",
                message=f"QNSE returned non-finite {name}"):
            return name
    return None
