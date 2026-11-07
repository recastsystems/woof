"""NOAHMP_SFLX's second half, evaluated for every land column at once.

``module_sf_noahmplsm.F:979-1076``.  :func:`woof.core.noahmp_sflx
.sflx_post_steps` holds the transcription the unmodified-WRF whole-column
fixture pins: one CPython frame per column, Python floats, an explicit ``f32``
rounding call after every operation, and ordinary ``if`` statements.  Measured
with ``cProfile``, that postfix is **13.7% of a Noah-MP land-surface call**, and
like ENERGY's composition it amortises over nothing -- there is no work in it
that two columns share.

This module is the same statements over the column axis.  It is not a second
port of the physics: the scalar module remains the authority and
``tests/test_noahmp_sflx_post_slab.py`` compares every field it owns against it,
bitwise.

Two segments, because WATER interrupts the postfix
==================================================

======== ==================================================================
seg1     :979-984.  SICE, SNEQVO and the QVAP/QDEW/EDIR block -- everything
         before the WATER call, plus the ENERGY exit column that WATER is
         handed.
seg2     :1014-1076.  The dead CARBON zeros, the dead irrigation identities,
         ERROR, the urban QSFC override, the snow clamp and the albedo
         sentinel.
======== ==================================================================

ERROR is not transcribed here
-----------------------------
``k_sflx_error`` in ``woof/core/kernels/noahmp_sflx.cu`` already **is** the
slab form of ERROR: it is one thread per column over ``[ncase][NSC]`` inputs,
it is held to ``noahmp-sflx-error.csv`` at ``max_ulp 0`` on all sixteen cases
by ``tests/test_noahmp_sflx_cuda.py``, and its argument shape is exactly the
one a slab has.  So :func:`sflx_post_seg2` launches it rather than writing a
fourth copy of ERROR's nine statements.

Its scalar packing is *read off the ``.cu``* -- the ``SC_``/``OU_`` blocks are
parsed by :func:`_slot_table` at call time -- rather than restated here.  A
34-slot layout that exists in three places (the kernel, its CUDA test and this
module) is a layout that will eventually disagree with itself, and the failure
mode of disagreeing is every column after the moved slot being read as the
wrong quantity.  Parsed, a slot added to the kernel without a value here is a
``KeyError`` that names it.

The one thing a kernel cannot do is abort.  ``k_sflx_error`` writes a status
word instead -- 1 shortwave, 2 energy, 3 water -- and this module raises the
same three exceptions :mod:`woof.core.noahmp_sflx` raises, naming the first
offending column, exactly as :mod:`woof.core.noahmp_energy_slab` does at the
emitted-longwave check.  A slab cannot raise for one column and continue for
the rest.

Where a branch became a mask, and where the denominator was substituted
----------------------------------------------------------------------
Every ``if`` on a per-column value is a :func:`cupy.where`.  Two of them guard
a division, and both substitute the denominator on the columns the mask
discards rather than relying on a NaN being thrown away:

* :1063's ``QFX / (RHOAIR * CH)`` runs only on urban columns.  The substitution
  is ``where(urban, rhoair * ch, 1.0)``, so an urban column keeps whatever the
  scalar path would have produced -- including a genuine division by zero, if
  one were ever reachable -- and a non-urban column stays finite.
* :1073's ``FSR / SWDOWN`` is guarded by ``SWDOWN /= 0`` at :1072, which is the
  night sentinel; the substituted denominator is ``1.0`` on exactly the columns
  the sentinel claims.

``min`` and ``max`` are :func:`woof.core.noahmp_slab_libm.fmn`/``fmx``, not
``cupy.minimum``/``maximum``: the scalar spelling returns the second argument on
a tie and the CuPy builtins do not.  That is not decorative here.  :983 is
``QDEW = ABS(MIN(RATIO, 0.0))`` and :982 is ``QVAP = MAX(RATIO, 0.0)``, so a
column whose FGEV is exactly zero lands on the tie in both, and ``ABS`` is a
sign-bit clear (:func:`woof.core.noahmp_slab_libm.slab_fabsf`), which is the
one operation that can still tell ``-0.0`` from ``+0.0`` after the clamp.

The layer accumulations
-----------------------
The postfix has exactly one, END_WB's ``SUM(SMC*DZSNSO)*1000`` at :1698-1700,
and it is inside ``k_sflx_error``, where it is already a layer loop with only
the column axis parallel.  FP32 addition is not associative, so that is the
only shape it can have.  SICE at :979 is elementwise across layers with no
accumulation, so the whole ``(n, NSOIL)`` slab is one expression.

The ENERGY exit column
----------------------
:func:`woof.core.noahmp_sflx.sflx_post_steps` opens by landing ENERGY's exit
arrays on the caller's ``SnowColumn`` (``col.stc[:] = en.stc`` and so on) before
handing that column to WATER.  That landing is the Python port's spelling of
WRF's own aliasing -- NOAHMP_SFLX passes one set of arrays to both ENERGY and
WATER -- and it is arithmetic-free, but it is also exactly where a slab can
transpose two same-shaped arrays and be wrong in a way no bitwise arithmetic
check can see.  :func:`sflx_post_seg1` therefore performs it and returns the
result, as copies, under the ``SnowColumn`` attribute names.
"""

from __future__ import annotations

import re
from functools import lru_cache

from woof.core.noahmp_slab_libm import copy_slab_slots, split_slab_slots

import numpy as np

from woof.core.kernels import get_kernel
from woof.core.noahmp_kernel_sources import translation_unit_source
from woof.core.noahmp_sflx import (EnergyBalanceError, NIGHT_ALBEDO,
                                    NSNOW_DEFAULT, NSOIL_DEFAULT,
                                    ShortwaveBalanceError, WaterBalanceError,
                                    _SNOW_EPS)
from woof.core.noahmp_slab_libm import fmn, fmx, slab_fabsf

#: WRF's ``-NSNOW+1 : NSOIL`` extent, as a 0-based width.
_NFULL = NSNOW_DEFAULT + NSOIL_DEFAULT

#: One CUDA thread per column; ``k_sflx_error`` is elementwise over columns.
_THREADS = 128

_ZERO = np.float32(0.0)
_ONE = np.float32(1.0)

#: The translation unit ``k_sflx_error`` lives in.
_ERROR_UNIT = "noahmp_sflx"
_ERROR_KERNEL = "k_sflx_error"


def _blocks(n: int) -> int:
    return (n + _THREADS - 1) // _THREADS


# ---------------------------------------------------------------------------
# ERROR's argument layout, read off the kernel that defines it
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _slot_table(prefix: str, count_macro: str) -> dict:
    """``{lower_case_name: slot}`` for one ``#define`` block of the kernel.

    The kernel is the definition of its own packing; a copy of it here would be
    a third spelling of a 34-slot layout, and the way that fails is silent.
    Parsing also makes the contract enforceable in the other direction: the
    caller below fills *every* name this returns, so a slot added to the ``.cu``
    without a value here raises rather than being launched as a zero.
    """
    text = translation_unit_source(_ERROR_UNIT, preamble=False)
    found = re.findall(rf"^#define {prefix}(\w+)\s+(\d+)$", text, re.M)
    if not found:
        raise AssertionError(
            f"{_ERROR_UNIT}.cu carries no {prefix}* block; the ERROR packing "
            "cannot be derived and must not be guessed")
    table = {name.lower(): int(slot) for name, slot in found}
    if sorted(table.values()) != list(range(len(table))):
        raise AssertionError(
            f"{_ERROR_UNIT}.cu's {prefix}* slots are not 0..{len(table) - 1}: "
            f"{sorted(table.values())}")
    declared = re.search(rf"^#define {count_macro}\s+(\d+)$", text, re.M)
    if declared is None or int(declared.group(1)) != len(table):
        raise AssertionError(
            f"{_ERROR_UNIT}.cu's {count_macro} disagrees with its {prefix}* "
            f"block ({declared and declared.group(1)} vs {len(table)})")
    return table


def error_input_slots() -> dict:
    """``SC_*``: which scalar ERROR reads from which column of its input row."""
    return dict(_slot_table("SC_", "NSC"))


def error_output_slots() -> dict:
    """``OU_*``: which ERROR output lands in which column of its output row."""
    return dict(_slot_table("OU_", "NOUT"))


# ---------------------------------------------------------------------------
# the input contract
# ---------------------------------------------------------------------------

def _missing(name: str, where: str) -> "KeyError":
    return KeyError(
        f"{where}: no {name!r} in the state mapping; the slab carries the "
        "value names sflx_post_steps uses, one array over the column axis "
        "each, with the layer axis last")


def _col(cp, s, name, n, where, dtype=None):
    """One ``(n,)`` input, contiguous and in the dtype the consumer declares."""
    if name not in s:
        raise _missing(name, where)
    dtype = cp.float32 if dtype is None else dtype
    value = cp.asarray(s[name])
    if value.ndim == 0:
        value = cp.broadcast_to(value, (n,))
    if value.shape != (n,):
        raise ValueError(
            f"{where}: {name!r} must have shape ({n},), got {value.shape}")
    return cp.ascontiguousarray(value, dtype=dtype)


def _lay(cp, s, name, n, width, where):
    """One ``(n, width)`` input, contiguous, layer axis last."""
    if name not in s:
        raise _missing(name, where)
    value = cp.asarray(s[name])
    if value.shape != (n, width):
        raise ValueError(
            f"{where}: {name!r} must have shape ({n}, {width}) with the layer "
            f"axis last, got {value.shape}")
    return cp.ascontiguousarray(value, dtype=cp.float32)


def _width(s, name, where) -> int:
    if name not in s:
        raise _missing(name, where)
    shape = np.shape(s[name])
    if not shape:
        raise ValueError(f"{where}: {name!r} carries no column axis")
    return int(shape[0])


# ---------------------------------------------------------------------------
# seg1: ENERGY's return (:979) to the WATER call (:988)
# ---------------------------------------------------------------------------

def sflx_post_seg1(s) -> dict:
    """SICE, SNEQVO, QVAP/QDEW/EDIR, and the column WATER is handed.

    ``s`` maps the scalar routine's value names to CuPy arrays over the column
    axis.  What it must carry:

    * ENERGY's exit column -- ``stc`` and ``dzsnso`` as ``(n, NSNOW+NSOIL)``,
      ``snice``/``snliq`` as ``(n, NSNOW)``, ``sh2o`` and ``smc`` as
      ``(n, NSOIL)``, ``snowh`` and ``sneqv`` as ``(n,)``.  ``dzsnso`` is
      NOAHMP_SFLX's own, from the prefix (:827-833), not ENERGY's;
    * ``fgev`` and ``latheag``, ENERGY's ground latent heat flux and its
      latent heat of vaporisation, which are the whole of :982-983.

    The returned dict carries the five values this segment computes and the
    eight arrays the ENERGY exit lands on, under the ``SnowColumn`` attribute
    names, as copies -- so the WATER batch mutates them exactly as WRF's WATER
    mutates NOAHMP_SFLX's own arrays, and the caller's ENERGY slab does not
    move underneath it.
    """
    import cupy as cp

    where = "sflx_post_seg1"
    n = _width(s, "sneqv", where)

    stc = _lay(cp, s, "stc", n, _NFULL, where)
    dzsnso = _lay(cp, s, "dzsnso", n, _NFULL, where)
    snice = _lay(cp, s, "snice", n, NSNOW_DEFAULT, where)
    snliq = _lay(cp, s, "snliq", n, NSNOW_DEFAULT, where)
    sh2o = _lay(cp, s, "sh2o", n, NSOIL_DEFAULT, where)
    smc = _lay(cp, s, "smc", n, NSOIL_DEFAULT, where)
    snowh = _col(cp, s, "snowh", n, where)
    sneqv = _col(cp, s, "sneqv", n, where)
    fgev = _col(cp, s, "fgev", n, where)
    latheag = _col(cp, s, "latheag", n, where)

    # -- :979  SICE.  Elementwise across layers: no accumulation, so the whole
    # slab is one expression and no layer loop is owed.
    sice = fmx(_ZERO, smc - sh2o)

    # -- :980
    sneqvo = sneqv.copy()

    # -- :982-984.  LATHEAG is HSUB or HVAP and never zero, so no substituted
    # denominator is owed here; MAX/MIN are the tie-correct forms and ABS is a
    # sign-bit clear, which is what tells -0.0 from +0.0 after the clamp.
    # RATIO is the scalar port's own factoring of FGEV/LATHEAG, which :982 and
    # :983 each spell out; a binary32 divide is deterministic, so evaluating it
    # once is the same number twice, and this keeps the two ports one shape.
    ratio = fgev / latheag
    qvap = fmx(ratio, _ZERO)                                        # :982
    qdew = slab_fabsf(fmn(ratio, _ZERO))                            # :983
    edir = qvap - qdew                                              # :984

    return {
        # the ENERGY exit column, as WATER receives it
        "stc": stc.copy(), "snice": snice.copy(), "snliq": snliq.copy(),
        "sh2o": sh2o.copy(), "smc": smc.copy(), "dzsnso": dzsnso.copy(),
        "snowh": snowh.copy(), "sneqv": sneqv.copy(),
        # what :979-984 computes
        "sice": cp.ascontiguousarray(sice),
        "sneqvo": cp.ascontiguousarray(sneqvo),
        "qvap": cp.ascontiguousarray(qvap),
        "qdew": cp.ascontiguousarray(qdew),
        "edir": cp.ascontiguousarray(edir),
    }


#: Every field :func:`sflx_post_seg1` returns, split by what it is.  Named
#: exhaustively so a field the segment stops writing fails a gate instead of
#: quietly leaving the bundle.
SEG1_LANDED = ("stc", "snice", "snliq", "sh2o", "smc", "dzsnso",
               "snowh", "sneqv")
SEG1_COMPUTED = ("sice", "sneqvo", "qvap", "qdew", "edir")


# ---------------------------------------------------------------------------
# seg2: WATER's return (:1014) to the end of the routine (:1076)
# ---------------------------------------------------------------------------

#: The ERROR arguments that are identically zero under the pinned identity.
#: FIRR is the irrigation heat flux (opt_irr = 0); IRMIRATE and IRFIRATE are
#: micro and flood irrigation rates, dead with it.  They are passed rather than
#: dropped because ``k_sflx_error`` reads all three and a slot left unwritten
#: is whatever the allocation held.
_ERROR_ZEROS = ("firr", "irmirate", "irfirate")


def sflx_post_seg2(s, *, carbon=None) -> dict:
    """CARBON's zeros, ERROR, the urban override, the snow clamp, the albedo.

    ``s`` is the state at WATER's return.  It must carry:

    * seg1's ``edir``;
    * ENERGY's ``fsa fsr fira fsh fcev fgev fctr ssoil sav sag pah canhs
      qsfc q2b ch``;
    * the prefix's ``swdown beg_wb prcp rhoair qair``;
    * WATER's ``canliq canice ecan etran runsrf runsub qtldrn`` and its exit
      ``smc`` as ``(n, NSOIL)``;
    * the column at the ERROR call -- ``sneqv``, ``snowh`` and ``dzsnso`` as
      ``(n, NSNOW+NSOIL)``, all *after* WATER and *before* the snow clamp,
      which is the order :1049-1069 has them in;
    * ``wa``, ``dt``, ``acc_dwater acc_prcp acc_ecan acc_etran acc_edir``, the
      integer ``ist``, and the boolean ``urban_flag``;
    * optionally ``calculate_soil``, per column or as one value, defaulting to
      true -- which is what ``soiltstep = 0.0`` pins it at.

    Raises :class:`~woof.core.noahmp_sflx.ShortwaveBalanceError`,
    :class:`~woof.core.noahmp_sflx.EnergyBalanceError` or
    :class:`~woof.core.noahmp_sflx.WaterBalanceError` for the slab, naming the
    first offending column, where WRF calls ``wrf_error_fatal`` for the column.
    """
    import cupy as cp

    where = "sflx_post_seg2"
    n = _width(s, "swdown", where)

    # ---- 1014-1018.  CARBON and the irrigation block, both dead ------------
    # dveg_active is .false. under DVEG = 4 and crop_active is .false. under
    # OPT_CROP = 0, so NEE/GPP/NPP keep the zeros written at 806-808.  PRCP and
    # FSH are :1042-1045's ``+= 0.0`` and ``-= 0.0``, exact identities on
    # binary32 for the values that reach them; they are carried through under
    # their own names so a caller reads the post-irrigation quantity.
    carbon = ({name:carbon[name] for name in ("nee","gpp","npp")} if carbon else
              {name: cp.zeros(n,dtype=cp.float32) for name in ("nee","gpp","npp")})
    prcp = _col(cp, s, "prcp", n, where)
    fsh = _col(cp, s, "fsh", n, where)

    # ---- ERROR, 1049-1058 --------------------------------------------------
    err = _error_slab(cp, s, n, where, prcp=prcp, fsh=fsh)

    # ---- 1061-1064  the urban QSFC override --------------------------------
    urban = _col(cp, s, "urban_flag", n, where, dtype=cp.bool_)
    qsfc = _col(cp, s, "qsfc", n, where)
    q2b = _col(cp, s, "q2b", n, where)
    edir = _col(cp, s, "edir", n, where)
    rhoair = _col(cp, s, "rhoair", n, where)
    ch = _col(cp, s, "ch", n, where)
    qair = _col(cp, s, "qair", n, where)

    qfx = (_col(cp, s, "etran", n, where)
           + _col(cp, s, "ecan", n, where)) + edir                  # :1061
    # Substituted on the columns the mask discards, so a non-urban column with
    # CH == 0 cannot produce a NaN that only happens to be masked away.  An
    # urban column keeps its own denominator, whatever it is.
    denom = rhoair * ch
    urban_qsfc = (qfx / cp.where(urban, denom, _ONE)) + qair        # :1063
    qsfc = cp.where(urban, urban_qsfc, qsfc)
    q2b = cp.where(urban, urban_qsfc, q2b)                          # :1064

    # ---- 1067-1069  the snow clamp -----------------------------------------
    snowh = _col(cp, s, "snowh", n, where)
    sneqv = _col(cp, s, "sneqv", n, where)
    clamp = (snowh <= _SNOW_EPS) | (sneqv <= _SNOW_EPS)             # :1067
    snowh = cp.where(clamp, _ZERO, snowh)                           # :1068
    sneqv = cp.where(clamp, _ZERO, sneqv)                           # :1069

    # ---- 1072-1076  the albedo, and the night sentinel ---------------------
    swdown = _col(cp, s, "swdown", n, where)
    lit = swdown != _ZERO                                           # :1072
    albedo = cp.where(lit,
                      _col(cp, s, "fsr", n, where)
                      / cp.where(lit, swdown, _ONE),                # :1073
                      NIGHT_ALBEDO)                                 # :1075

    out = {
        **carbon,
        "prcp": prcp, "fsh": fsh,
        "qsfc": cp.ascontiguousarray(qsfc),
        "q2b": cp.ascontiguousarray(q2b),
        "snowh": cp.ascontiguousarray(snowh),
        "sneqv": cp.ascontiguousarray(sneqv),
        "albedo": cp.ascontiguousarray(albedo),
    }
    out.update(err)
    return out


#: Every field :func:`sflx_post_seg2` returns.  ``errsw`` and ``erreng`` are
#: ERROR's two residuals: NOAHMP_SFLX keeps neither, but they are what the two
#: abort gates test, and a port that computed them wrongly and never showed
#: them would pass every whole-column check there is.
SEG2_FIELDS = ("nee", "gpp", "npp", "prcp", "fsh",
               "errwat", "end_wb", "errsw", "erreng",
               "acc_dwater", "acc_prcp", "acc_ecan", "acc_etran", "acc_edir",
               "qsfc", "q2b", "snowh", "sneqv", "albedo")


def _error_slab(cp, s, n, where, *, prcp, fsh) -> dict:
    """ERROR for the whole slab, through the kernel that already is ERROR.

    ``k_sflx_error`` takes one row of ``NSC`` scalars per column plus SMC and
    the ``1..NSOIL`` slice of DZSNSO, which is the argument shape a slab has;
    it is held to ``noahmp-sflx-error.csv`` at ``max_ulp 0`` on all sixteen
    oracle cases.  Nothing about ERROR is re-expressed here.
    """
    _slots_sc = []
    slots = _slot_table("SC_", "NSC")
    outs = _slot_table("OU_", "NOUT")

    # Every SC_* name, by name.  A slot the kernel declares and this does not
    # fill raises below rather than being launched as an uninitialised zero.
    values = {name: _col(cp, s, name, n, where)
              for name in ("swdown", "fsa", "fsr", "fira", "fcev", "fgev",
                           "fctr", "ssoil", "sav", "sag", "beg_wb", "canliq",
                           "canice", "sneqv", "wa", "ecan", "etran", "edir",
                           "runsrf", "runsub", "dt", "qtldrn", "pah", "canhs",
                           "acc_dwater", "acc_prcp", "acc_ecan", "acc_etran",
                           "acc_edir")}
    values["prcp"] = prcp
    values["fsh"] = fsh
    zeros = cp.zeros(n, dtype=cp.float32)
    values.update({name: _col(cp,s,name,n,where) if name in s else zeros for name in _ERROR_ZEROS})

    unknown = sorted(set(values) - set(slots))
    if unknown:
        raise AssertionError(
            f"{where}: {unknown} is not an SC_* slot of {_ERROR_UNIT}.cu; the "
            "ERROR packing moved and this launch would be filling the wrong "
            "columns")
    sc = cp.empty((n, len(slots)), dtype=cp.float32)
    for name, slot in slots.items():
        try:
            _slots_sc.append((slot, values[name]))
        except KeyError:
            raise AssertionError(
                f"{where}: {_ERROR_UNIT}.cu declares SC_{name.upper()} and "
                "this launch has no value for it") from None

    smc = _lay(cp, s, "smc", n, NSOIL_DEFAULT, where)
    # The 1..NSOIL slice ERROR's only loop indexes.  ``[:, NSNOW:]`` is a
    # strided view and a raw-kernel argument carries a pointer, not a stride.
    dzsnso = cp.ascontiguousarray(
        _lay(cp, s, "dzsnso", n, _NFULL, where)[:, NSNOW_DEFAULT:])
    ist = _col(cp, s, "ist", n, where, dtype=cp.int32)
    calc = (cp.ones(n, dtype=cp.int32) if "calculate_soil" not in s
            else _col(cp, s, "calculate_soil", n, where, dtype=cp.int32))

    out = cp.zeros((n, len(outs)), dtype=cp.float32)
    status = cp.zeros(n, dtype=cp.int32)
    copy_slab_slots(sc, _slots_sc)
    if n:
        # A grid of zero blocks is a CUDA configuration error, so an empty
        # tile skips the launch and still answers with well-formed arrays.
        get_kernel(_ERROR_UNIT, _ERROR_KERNEL)(
            (_blocks(n),), (_THREADS,),
            (cp.ascontiguousarray(sc), smc, dzsnso, ist, calc, out, status,
             np.int32(n)))

    _raise_for_status(cp, status, out, outs)
    return split_slab_slots(out, {name: (slot, 1) for name, slot in outs.items()})


def _raise_for_status(cp, status, out, outs) -> None:
    """WRF's three ``wrf_error_fatal`` calls, as one exception for the slab.

    A kernel cannot abort and a slab cannot abort for one column and keep
    going for the rest, so the kernel reports and this raises -- the same three
    exception types :mod:`woof.core.noahmp_sflx` raises, with the first
    offending column named.  Reporting instead of aborting is a change in
    mechanism, not in behaviour.
    """
    from woof.core import health_ledger
    from woof.core.noahmp_surface_device import active_probe
    ledger = health_ledger.active()
    if ledger is not None:
        # OR categories across columns and steps. A callback retaining the
        # last slab would lose an earlier failure when that slab was healthy.
        def refuse(flags):
            errors = ((1, ShortwaveBalanceError, "Shortwave budget"),
                      (2, EnergyBalanceError, "Energy budget"),
                      (4, WaterBalanceError, "Water budget"))
            for bit, kind, name in errors:
                if flags & bit:
                    raise kind(f"{name} problem in NOAHMP LSM"
                               + health_ledger.deferred_note())
        flags = cp.zeros(1, dtype=cp.uint32)
        for code in (1, 2, 3):
            flags |= active_probe(status == code).any().astype(cp.uint32) << (code - 1)
        ledger.record("noahmp.post.native_budgets", flags, refuse)
        return
    n = int(status.size)
    if status.shape != (n,):
        raise ValueError("Noah-MP slab status must contain one value per column")
    codes = cp.asnumpy(cp.where(active_probe(cp.ones(n, dtype=cp.bool_)), status, 0))
    tripped = np.flatnonzero(codes)
    if not tripped.size:
        return
    column = int(tripped[0])
    code = int(codes[column])
    row = cp.asnumpy(out)[column]
    if code == 1:                                                   # :1641
        raise ShortwaveBalanceError(
            f"land column {column}: Stop in Noah-MP: ERRSW = "
            f"{float(row[outs['errsw']])!r} W/m2")
    if code == 2:                                                   # :1665
        raise EnergyBalanceError(
            f"land column {column}: Energy budget problem in NOAHMP LSM: "
            f"ERRENG = {float(row[outs['erreng']])!r} W/m2")
    errwat = float(row[outs["errwat"]])                             # :1713
    direction = "gaining" if errwat > 0.0 else "losing"
    raise WaterBalanceError(
        f"land column {column}: Water budget problem in NOAHMP LSM: the model "
        f"is {direction} water, ERRWAT = {errwat!r} kg m{{-2}} timestep{{-1}}")


__all__ = [
    "SEG1_COMPUTED",
    "SEG1_LANDED",
    "SEG2_FIELDS",
    "error_input_slots",
    "error_output_slots",
    "sflx_post_seg1",
    "sflx_post_seg2",
]
