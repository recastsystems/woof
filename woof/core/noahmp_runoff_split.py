"""Separate surface and subsurface runoff options for Noah-MP.

Re-implemented from published descriptions, not bitwise to upstream.

Noah-MP v5.0 split the single ``opt_run`` into a surface runoff option and a
subsurface runoff option (He et al. 2023, GMD 16, 5131, Sect. 2.6: "we have
separated the runoff options for surface and subsurface runoff processes"),
and WRF 4.8.0 carries them as ``opt_runsrf`` / ``opt_runsub``.  The public
technical description (He et al. 2023, NCAR/TN-575+STR, Sect. 3.7.8) numbers
both lists 1-8 with the same meaning as the old ``opt_run`` and says the
tested choice is the same number for both, so a stock ``opt_run=N`` is the
pair ``(N, N)`` and keeps its meaning exactly.

What the split adds is the mixed pairs, for example VIC surface runoff with
the Niu et al. (2007) groundwater baseflow.  The physics of each half is our
own WRF-v4-lineage port; only the gates move (see
``woof/core/kernels/noahmp_water_split.cuh`` and its generator).  No
NCAR/noahmp or other UCAR-licensed source was read for any of this.

The engine represents a pair as ``opt_run`` = the surface option plus
``opt_runsub`` = the subsurface option when it differs, and 0 when the pair is
stock.  Pairs whose subsurface halves are the same drainage scheme as the
stock pair (subsurface 3, 6, 7 and 8 all use Eq. 3.7.338 with the Eq. 3.7.255
free-drainage bottom flux) collapse to the stock pair, so nothing changes for
them.
"""
from __future__ import annotations

LABEL = "re-implemented from published descriptions, not bitwise to upstream"
RUNOFF_OPTIONS = (1, 2, 3, 4, 5, 6, 7, 8)
#: Subsurface options that are one scheme: the drainage-dependent subsurface
#: runoff of tech note Sect. 3.7.8.5 item 3 (Eq. 3.7.338) with the
#: ``SLOPE * WCND`` bottom flux of Eq. 3.7.255.
DRAINAGE = (3, 6, 7, 8)
#: Subsurface options whose state is the TOPMODEL water table (ZWT positive
#: depth, WA, WT) and the MMF one (ZWT negative, WA, SMCWTD, DEEPRECH, RECH).
TOPMODEL = (1, 2)
MMF = (5,)


class RunoffSplitRefused(ValueError):
    """A pair whose combination has a named breakage."""


def resolve(opt_run: int = 3, opt_runsrf: int = 0, opt_runsub: int = 0) -> tuple[int, int]:
    """``(surface, subsurface)`` from the stock knob and the WRF 4.8.0 pair.

    0 means "not given": a missing half inherits ``opt_run``, which is what
    keeps a stock namelist's meaning.
    """
    for name, value in (("opt_run", opt_run), ("opt_runsrf", opt_runsrf), ("opt_runsub", opt_runsub)):
        if type(value) is not int:
            raise RunoffSplitRefused(f"Noah-MP {name} must be an int, got {value!r}")
    if opt_run not in RUNOFF_OPTIONS:
        raise RunoffSplitRefused(f"Noah-MP opt_run must be in {RUNOFF_OPTIONS}, got {opt_run}")
    srf = opt_runsrf or opt_run
    sub = opt_runsub or opt_run
    if srf not in RUNOFF_OPTIONS or sub not in RUNOFF_OPTIONS:
        raise RunoffSplitRefused(
            f"Noah-MP opt_runsrf/opt_runsub must be 0 (inherit opt_run) or in {RUNOFF_OPTIONS}, "
            f"got {opt_runsrf}/{opt_runsub}")
    return srf, sub


def canonical(srf: int, sub: int) -> tuple[int, int]:
    """Engine form: ``(opt_run, opt_runsub)`` with ``opt_runsub=0`` for stock."""
    if sub == srf or (srf in DRAINAGE and sub in DRAINAGE):
        return srf, 0
    return srf, sub


def kernel_subsurface(sub: int) -> int:
    """The subsurface instantiation: the four drainage numbers are one scheme."""
    return 3 if sub in DRAINAGE else sub


def validate(srf: int, sub: int, *, opt_tdrn: int = 0, opt_infdv: int = 0) -> None:
    """Refuse the pairs that would silently compute the wrong thing.

    Each refusal names its breakage (gate law).  Stock pairs never refuse here.
    """
    if srf == sub or (srf in DRAINAGE and sub in DRAINAGE):
        return
    if srf == 1 and sub not in TOPMODEL:
        raise RunoffSplitRefused(
            f"opt_runsrf=1 with opt_runsub={sub}: the Niu 2007 saturated fraction reads the "
            "groundwater water table (tech note Eq. 3.7.200-3.7.201), and only opt_runsub=1 "
            "(groundwater) or 2 (equilibrium) updates it; the surface would read the cold-start "
            "ZWT forever")
    if srf == 5:
        raise RunoffSplitRefused(
            f"opt_runsrf=5 with opt_runsub={sub}: the MMF saturated fraction (Eq. 3.7.227) reads "
            "the MMF water table, which only opt_runsub=5 and its lateral-flow step update")
    if sub in MMF and srf in TOPMODEL:
        raise RunoffSplitRefused(
            f"opt_runsrf={srf} with opt_runsub=5: TOPMODEL reads ZWT as a positive depth and MMF "
            "writes it as a negative one; the pair would compute a saturated fraction from the "
            "wrong sign")
    if opt_tdrn:
        raise RunoffSplitRefused(
            f"opt_tdrn={opt_tdrn} with opt_runsrf={srf}/opt_runsub={sub}: tile drainage is "
            "described and calibrated only for runoff option 3 in both halves (tech note "
            "Sect. 3.7.8.4); on a split pair it would silently not drain")
    if srf == 8 and opt_infdv == 0:
        raise RunoffSplitRefused("opt_runsrf=8 requires opt_infdv=1,2,3: native FSUR is unassigned at zero")


def carrier_run(opt_run: int = 3, opt_runsub: int = 0) -> int:
    """The ``opt_run`` value whose groundwater carriers the pair needs.

    Host code that sizes state (ZWT/WA/WT or the MMF set), reads geography or
    runs the MMF lateral step asks this instead of ``opt_run``.  For a stock
    pair it returns ``opt_run`` itself, so every stock decision is unchanged.
    """
    if not opt_runsub or opt_runsub == opt_run:
        return opt_run
    if opt_run in MMF or opt_runsub in MMF:
        return 5
    if opt_run in TOPMODEL:
        return opt_run
    if opt_runsub in TOPMODEL:
        return opt_runsub
    return opt_run


def carrier_of(cfg) -> int:
    """:func:`carrier_run` for a RunConfig-like object (absent fields are stock)."""
    return carrier_run(int(getattr(cfg, "opt_run", 3)), int(getattr(cfg, "opt_runsub", 0) or 0))


def transferred_names(opt_run: int = 3, opt_runsub: int = 0) -> tuple[str, ...]:
    """MPTABLE scalars each half reads (upper case, as the table spells them)."""
    def one(run):
        return (("FSATMX", "TIMEAN") if run in TOPMODEL else ("FSATMX",) if run in MMF
                else ("BVIC",) if run == 6 else ("BBVIC", "BDVIC", "GDVIC") if run == 8
                else ("AXAJ", "BXAJ", "XXAJ") if run == 7 else ())
    names = one(opt_run)
    if opt_runsub and opt_runsub != opt_run:
        names += tuple(n for n in one(kernel_subsurface(opt_runsub)) if n not in names)
    return names


__all__ = ["DRAINAGE", "LABEL", "MMF", "RUNOFF_OPTIONS", "RunoffSplitRefused", "TOPMODEL",
           "canonical", "carrier_of", "carrier_run", "kernel_subsurface", "resolve", "transferred_names",
           "validate"]
