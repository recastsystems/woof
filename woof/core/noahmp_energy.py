"""Noah-MP ``ENERGY`` -- WRF v4.6.1 ``phys/module_sf_noahmplsm.F:1741-2396``.

ENERGY is not a leaf.  It is the composition that owns the surface energy
balance: it derives the roughness/displacement geometry, the snow-cover
fraction, the soil-water stress factors and the two psychrometric constants,
hands them to six already-pinned subsystems, and then does the tile-average
bookkeeping that turns a vegetated answer and a bare answer into one column.

Everything it calls is imported, never re-transcribed:

============================  ==========================================
``THERMOPROP``                :mod:`woof.core.noahmp_leaves`
``ALBEDO`` / ``SURRAD``       :mod:`woof.core.noahmp_radiation`
``VEGE_FLUX``                 :mod:`woof.core.noahmp_vegeflux`
``BARE_FLUX``                 :mod:`woof.core.noahmp_bareflux`
``TSNOSOI`` / ``PHASECHANGE`` :mod:`woof.core.noahmp_thermal`
``expf`` / ``powf`` / ...     :mod:`woof.core.noahmp_libm`
============================  ==========================================

``RADIATION`` (:2691-2806) itself has no module of its own -- the radiation
lane pinned its leaves, not the wrapper -- so :func:`_radiation` below is the
wrapper, and it is composition, not physics: an ``ALBEDO`` call, five
assignments, a ``SURRAD`` call.

Pinned option identity (WRF Registry defaults)
----------------------------------------------
``dveg=4 opt_crs=1 opt_btr=1 opt_run=3 opt_sfc=1 opt_frz=1 opt_inf=1
opt_rad=3 opt_alb=2 opt_snf=1 opt_tbot=2 opt_stc=1 opt_rsf=1 opt_soil=1
opt_pedo=1 opt_crop=0 opt_irr=0 opt_irrm=0 opt_infdv=0 opt_tdrn=0``, with
``soil_update_steps=1`` and ``calculate_soil=.true.``

Under that identity whole routines are unreachable, and this module **asserts
them off rather than porting them** -- every one of these raises
``NotImplementedError`` if the option moves:

======================================================  =============
``SFCDIF2``                                             ``opt_sfc=1``
``CANRES`` / ``CALHUM``                                 ``opt_crs=1``
``SNOWALB_BATS``                                        ``opt_alb=2``
ENERGY's ``OPT_STC==2`` snow-surface reset (:2384-2396) ``opt_stc=1``
ENERGY's CLM and SSiB ``BTRAN`` legs (:2153-2163)       ``opt_btr=1``
ENERGY's Sellers ``RSURF`` legs (:2188-2202)            ``opt_rsf=1``
gecros / crop chain                                     ``opt_crop=0``
irrigation                                              ``opt_irr=0``
======================================================  =============

Outside the admitted slice, refused rather than approximated: ``ICE == 1``
(the glacier ground-emissivity leg at :2145-2147) and ``IST == 2`` (the lake
legs at :2076-2082 and :2178-2181).  ``noahmp-sflx.csv`` does not admit them
either, so there is no fixture that could hold a transcription of them to
``max_ulp 0``.

Two libm questions, one of which is a trap
------------------------------------------
* ``TANH`` at :2072 **is** a trap.  It is glibc's ``tanhf``, still the 1993
  SunPro ``expm1f``-based routine, and over the 204,064,836 FP32 inputs in
  [1e-6, 22] it disagrees with ``(float)tanh((double)x)`` on 23.8% of them.
  The snow-cover fraction cannot be reached through an FP64 shim.  The
  transcription lives in :mod:`woof.core.noahmp_libm` with the rest of the
  glibc FP32 surface, verified against the live glibc 2.39 over 1,106,247,680
  inputs with zero mismatches.
* ``UU**2.0`` at :2057 **is not**, though it looks like one.  The exponent is
  a REAL literal, so gfortran emits ``powf`` rather than a multiply (``nm -u``
  on a two-line probe shows the undefined symbol), but swept over the
  167,177,618 FP32 values in [1e-3, 1e3] glibc's ``powf(x, 2.0f)`` is
  bit-identical to ``x*x`` on every one.  The ``powf`` spelling is kept
  because it is what the Fortran compiles to, not because it is observable;
  the measurement is recorded so the question is not re-derived.

What ``max_ulp 0`` did not prove
--------------------------------
Nine columns matched the fixture bitwise on the first run, including the
TANH.  They also matched it with ``math.tanh`` substituted for ``tanhf``: all
four snow columns happened to land on arguments where the two agree.  The
fixture's case 9 now carries ``SNEQV = 8.75`` mm -- a snow density of exactly
250 kg/m3, putting the FSNO argument just below 1.75 -- specifically so that it
can tell the difference.  ``tests/test_noahmp_energy.py`` keeps that as a live
negative control.

Where WRF is undefined
----------------------
Four groups of ``INTENT(OUT)`` slots are left unassigned on some paths.  This
module produces the **defined** behaviour, ``0.0``, and
``woof/data/noahmp/oracle/PROVENANCE-energy.md`` records the divergence; the
fixture marks the same slots with role ``undef`` and never pins the residue.

* ``IMELT(k)``, ``HCPCT(k)``, ``SNICEV(k)``, ``SNLIQV(k)``, ``EPORE(k)`` for
  ``k <= ISNOW`` -- PHASECHANGE, THERMOPROP and CSNOW all loop from ``ISNOW+1``.
* ``BTRANI(k)`` for ``k > NROOT`` -- :2164-2172 covers the root zone only.
  (``NOAHMP_SFLX`` then hands all ``NSOIL`` of them to ``WATER``.  That is a
  live defect in the column, not in ENERGY.)
* ``FSRV``, ``FSRG``, ``BGAP``, ``WGAP`` when ``COSZ <= 0`` -- ALBEDO's
  initialisation loop (:2908-2921) misses ``FREVD``/``FREVI``/``FREGD``/
  ``FREGI``/``BGAP``/``WGAP`` before ``IF(COSZ <= 0) GOTO 100``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from woof.core.noahmp_bareflux import bare_flux
from woof.core.noahmp_leaves import thermoprop
from woof.core.noahmp_libm import expf, f32, powf, sqrtf, tanhf
from woof.core.noahmp_radiation import RadiationOptions, albedo, surrad
from woof.core.noahmp_thermal import phasechange, tsnosoi
from woof.core.noahmp_vegeflux import VegeFluxParameters, vege_flux

class LeafRequest:
    """One paused physical leaf call, as :func:`energy_steps` yields it.

    ``leaf`` is ``"vege_flux"`` or ``"bare_flux"``; ``args`` and ``kwargs`` are
    that routine's own argument list, unaltered.  The object exists so a
    scheduler can collect the same call from many columns and answer them
    together; it carries no physics and no column identity.
    """

    __slots__ = ("leaf", "args", "kwargs")

    def __init__(self, leaf: str, args: tuple, kwargs: dict):
        self.leaf = leaf
        self.args = args
        self.kwargs = kwargs

    def __repr__(self) -> str:                      # pragma: no cover - debug
        return f"LeafRequest({self.leaf!r}, {len(self.args)} args)"


#: The host answer for a paused leaf.
#: ENERGY owns the two flux leaves; anything paused outside ENERGY registers
#: itself, which is what keeps this module from importing its own callers.
#:
#: ``thermoprop``, ``tsnosoi`` and ``phasechange`` are the unmodified CPython
#: routines the WRF fixtures pin, bound here by keyword only.  They are paused
#: for the same reason the flux leaves are -- so a scheduler can answer one
#: leaf for many columns in one launch -- and for no other; nothing about the
#: arithmetic changes when they are.
_HOST_LEAF = {
    "vege_flux": lambda *args, **kw: vege_flux(*args, **kw),
    "bare_flux": lambda *args, **kw: bare_flux(*args, **kw),
    "thermoprop": lambda **kw: thermoprop(**kw),
    "tsnosoi": lambda **kw: tsnosoi(**kw),
    "phasechange": lambda **kw: phasechange(**kw),
}


def register_host_leaf(name: str, routine) -> None:
    """Bind the CPython answer for a leaf paused outside this module."""
    existing = _HOST_LEAF.get(name)
    if existing is not None and existing is not routine:
        raise ValueError(
            f"Noah-MP leaf {name!r} already has a different host answer")
    _HOST_LEAF[name] = routine


def answer_on_host(request: LeafRequest):
    """Evaluate one :class:`LeafRequest` with the unmodified CPython leaf."""
    try:
        leaf = _HOST_LEAF[request.leaf]
    except KeyError:
        raise ValueError(f"unknown Noah-MP leaf request {request.leaf!r}")
    return leaf(*request.args, **request.kwargs)


def drive_on_host(steps):
    """Run a leaf-yielding generator to completion against the host leaves.

    This is the whole difference between :func:`energy` and
    :func:`energy_steps`: identical arithmetic, one of them driven here.
    """
    reply = None
    while True:
        try:
            request = steps.send(reply)
        except StopIteration as stop:
            return stop.value
        reply = answer_on_host(request)

NSNOW = 3
NSOIL = 4

# module_sf_noahmplsm.F:204-220
GRAV = f32(9.80616)
SB = f32(5.67e-08)
TFRZ = f32(273.16)
HSUB = f32(2.8440e06)
HVAP = f32(2.5104e06)
CPAIR = f32(1004.64)
RW = f32(461.269)

# ENERGY's own PARAMETERs, :2028-2030
MPE = f32(1.0e-6)
Z0 = f32(0.002)          # bare-soil roughness length under the canopy
#: RADIATION's own MPE, :2767 -- a different literal from ENERGY's.
_RADIATION_MPE = f32(1.0e-6)


def _mn(a, b):
    a = f32(a); b = f32(b)
    return a if a < b else b


def _mx(a, b):
    a = f32(a); b = f32(b)
    return a if a > b else b


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnergyOptions:
    """The Noah-MP options ENERGY and its subtree branch on."""

    dveg: int = 4
    opt_crs: int = 1
    opt_btr: int = 1
    opt_sfc: int = 1
    opt_frz: int = 1
    opt_rad: int = 3
    opt_alb: int = 2
    opt_stc: int = 1
    opt_rsf: int = 1
    opt_crop: int = 0
    opt_irr: int = 0
    soil_update_steps: int = 1
    calculate_soil: bool = True
    noahmp_snow_age: str = "wrf_470"

    def check(self) -> None:
        from woof.noahmp_generation import snow_age_generation
        snow_age_generation(self.noahmp_snow_age)
        if self.opt_sfc != 1:
            raise NotImplementedError(
                "opt_sfc != 1 selects SFCDIF2, dead under the Registry "
                "default opt_sfc=1; it is deliberately not transcribed")
        if self.opt_crs != 1:
            raise NotImplementedError(
                "opt_crs != 1 selects CANRES/CALHUM, dead under the Registry "
                "default opt_crs=1; they are deliberately not transcribed")
        if self.opt_alb != 2:
            raise NotImplementedError(
                "opt_alb must be 2 (CLASS); opt_alb=1 needs SNOWALB_BATS's "
                "snow-age-dependent albedo calculation, which is not implemented")
        if self.opt_btr != 1:
            raise NotImplementedError(
                "opt_btr != 1 selects the CLM (2) or SSiB (3) BTRAN leg at "
                "module_sf_noahmplsm.F:2153-2163; the Registry default is 1 "
                "and the other two are deliberately not transcribed")
        if self.opt_rsf != 1:
            raise NotImplementedError(
                "opt_rsf != 1 selects the Sellers RSURF legs at :2188-2202 or "
                "the FSNO weighting at :2204-2206; the Registry default is 1 "
                "and the others are deliberately not transcribed")
        if self.opt_stc != 1:
            raise NotImplementedError(
                "opt_stc != 1 activates ENERGY's snow-surface temperature "
                "reset at :2384-2396; the Registry default is 1 and that "
                "block is deliberately not transcribed")
        if self.opt_crop != 0:
            raise NotImplementedError(
                "opt_crop != 0 activates the gecros chain, dead under the "
                "Registry default opt_crop=0")
        if self.opt_irr != 0:
            raise NotImplementedError(
                "opt_irr != 0 activates irrigation, dead under the Registry "
                "default opt_irr=0")
        if self.dveg != 4:
            raise NotImplementedError(
                "only dveg=4 is admitted; DVEG selects how NOAHMP_SFLX derives "
                "FVEG and LAI before ENERGY is reached")

    def radiation_options(self) -> RadiationOptions:
        return RadiationOptions(opt_rad=self.opt_rad, opt_alb=self.opt_alb,
                                noahmp_snow_age=self.noahmp_snow_age)


PINNED_OPTIONS = EnergyOptions(noahmp_snow_age="wrf_461")


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnergyParameters:
    """The ``noahmp_parameters`` components the ENERGY subtree reads.

    Build one with :meth:`from_transferred` from
    :func:`woof.core.noahmp.transfer_mp_parameters`, whose output is itself
    pinned in ``woof/data/noahmp/oracle/noahmp-parameters.csv``.
    """

    # ENERGY's own body
    urban_flag: bool
    nroot: int
    mfsno: float
    scffac: float
    z0sno: float
    z0mvt: float
    hvt: float
    cwpvt: float
    snow_emis: float
    rsurf_exp: float
    rsurf_snow: float
    eg: Tuple[float, float]
    smcmax: Tuple[float, ...]
    smcref: Tuple[float, ...]
    smcwlt: Tuple[float, ...]
    psisat: Tuple[float, ...]
    bexp: Tuple[float, ...]
    # THERMOPROP
    quartz: Tuple[float, ...]
    csoil: float
    # TSNOSOI
    zbot: float
    # RADIATION
    tau0: float
    grain_growth: float
    extra_growth: float
    dirt_soot: float
    swemx: float
    albsat: Tuple[float, float]
    albdry: Tuple[float, float]
    alblak: Tuple[float, float]
    rhol: Tuple[float, float]
    rhos: Tuple[float, float]
    taul: Tuple[float, float]
    taus: Tuple[float, float]
    xl: float
    omegas: Tuple[float, float]
    betads: float
    betais: float
    # VEGE_FLUX
    vege: VegeFluxParameters
    # Native SNOWALB_BATS's seven transferred scalars in
    # noahmp_radiation_gpu.BATS_PARAMETER_NAMES order. Older manually built
    # CLASS-only handles may omit them; ALB=1 preparation must require them.
    bats: Tuple[float, ...] | None = None

    @classmethod
    def from_transferred(cls, p, nsoil: int = NSOIL) -> "EnergyParameters":
        from woof.core.noahmp_radiation_gpu import BATS_PARAMETER_NAMES

        def sc(name):
            return f32(float(p.scalar(name)))

        def vec(name, n):
            return tuple(f32(float(p.array(name)[i])) for i in range(n))

        return cls(
            urban_flag=bool(p.urban_flag),
            nroot=int(p.scalar("NROOT")),
            mfsno=sc("MFSNO"), scffac=sc("SCFFAC"), z0sno=sc("Z0SNO"),
            z0mvt=sc("Z0MVT"), hvt=sc("HVT"), cwpvt=sc("CWPVT"),
            snow_emis=sc("SNOW_EMIS"),
            rsurf_exp=sc("RSURF_EXP"), rsurf_snow=sc("RSURF_SNOW"),
            eg=vec("EG", 2),
            smcmax=vec("SMCMAX", nsoil), smcref=vec("SMCREF", nsoil),
            smcwlt=vec("SMCWLT", nsoil), psisat=vec("PSISAT", nsoil),
            bexp=vec("BEXP", nsoil), quartz=vec("QUARTZ", nsoil),
            csoil=sc("CSOIL"), zbot=sc("ZBOT"),
            tau0=sc("TAU0"), grain_growth=sc("GRAIN_GROWTH"),
            extra_growth=sc("EXTRA_GROWTH"), dirt_soot=sc("DIRT_SOOT"),
            swemx=sc("SWEMX"),
            albsat=vec("ALBSAT", 2), albdry=vec("ALBDRY", 2),
            alblak=vec("ALBLAK", 2), rhol=vec("RHOL", 2), rhos=vec("RHOS", 2),
            taul=vec("TAUL", 2), taus=vec("TAUS", 2), xl=sc("XL"),
            omegas=vec("OMEGAS", 2), betads=sc("BETADS"), betais=sc("BETAIS"),
            vege=VegeFluxParameters(
                DLEAF=sc("DLEAF"), HVT=sc("HVT"), CBIOM=sc("CBIOM"),
                C3PSN=sc("C3PSN"), KC25=sc("KC25"), AKC=sc("AKC"),
                KO25=sc("KO25"), AKO=sc("AKO"), AVCMX=sc("AVCMX"),
                VCMX25=sc("VCMX25"), BP=sc("BP"), MP=sc("MP"),
                QE25=sc("QE25"), FOLNMX=sc("FOLNMX"),
            ),
            bats=tuple(sc(name.upper()) for name in BATS_PARAMETER_NAMES),
        )


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------
@dataclass
class EnergyState:
    """Every ENERGY ``INTENT(OUT)`` and ``INTENT(INOUT)`` argument.

    Layer arrays are 0-based sequences spanning Fortran ``-NSNOW+1 .. NSOIL``
    (or ``-NSNOW+1 .. 0``), i.e. element 0 is the ``-NSNOW+1`` layer.  Slots
    WRF leaves undefined carry ``0.0``; see the module docstring.
    """

    z0wrf: float = 0.0
    imelt: Tuple[int, ...] = ()
    snicev: Tuple[float, ...] = ()
    snliqv: Tuple[float, ...] = ()
    epore: Tuple[float, ...] = ()
    hcpct: Tuple[float, ...] = ()
    t2m: float = 0.0
    fsno: float = 0.0
    sav: float = 0.0
    sag: float = 0.0
    qmelt: float = 0.0
    fsa: float = 0.0
    fsr: float = 0.0
    taux: float = 0.0
    tauy: float = 0.0
    fira: float = 0.0
    fsh: float = 0.0
    fcev: float = 0.0
    fgev: float = 0.0
    fctr: float = 0.0
    trad: float = 0.0
    psn: float = 0.0
    apar: float = 0.0
    ssoil: float = 0.0
    btrani: Tuple[float, ...] = ()
    btran: float = 0.0
    ponding: float = 0.0
    ts: float = 0.0
    latheav: float = 0.0
    latheag: float = 0.0
    frozen_canopy: bool = False
    frozen_ground: bool = False
    tv: float = 0.0
    tg: float = 0.0
    stc: Tuple[float, ...] = ()
    snowh: float = 0.0
    eah: float = 0.0
    tah: float = 0.0
    sneqvo: float = 0.0
    sneqv: float = 0.0
    sh2o: Tuple[float, ...] = ()
    smc: Tuple[float, ...] = ()
    snice: Tuple[float, ...] = ()
    snliq: Tuple[float, ...] = ()
    albold: float = 0.0
    cm: float = 0.0
    ch: float = 0.0
    tauss: float = 0.0
    laisun: float = 0.0
    laisha: float = 0.0
    rb: float = 0.0
    qsfc: float = 0.0
    t2mv: float = 0.0
    t2mb: float = 0.0
    fsrv: float = 0.0
    fsrg: float = 0.0
    rssun: float = 0.0
    rssha: float = 0.0
    albsnd: Tuple[float, float] = (0.0, 0.0)
    albsni: Tuple[float, float] = (0.0, 0.0)
    bgap: float = 0.0
    wgap: float = 0.0
    tgv: float = 0.0
    tgb: float = 0.0
    q1: float = 0.0
    q2v: float = 0.0
    q2b: float = 0.0
    q2e: float = 0.0
    chv: float = 0.0
    chb: float = 0.0
    emissi: float = 0.0
    pah: float = 0.0
    canhs: float = 0.0
    shg: float = 0.0
    shc: float = 0.0
    shb: float = 0.0
    evg: float = 0.0
    evb: float = 0.0
    ghv: float = 0.0
    ghb: float = 0.0
    irg: float = 0.0
    irc: float = 0.0
    irb: float = 0.0
    tr: float = 0.0
    evc: float = 0.0
    chleaf: float = 0.0
    chuc: float = 0.0
    chv2: float = 0.0
    chb2: float = 0.0
    eflxb: float = 0.0
    acc_ssoil: float = 0.0
    #: Which side of each live branch this column took, keyed for the tests.
    branches: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RADIATION -- module_sf_noahmplsm.F:2691-2806
# ---------------------------------------------------------------------------
def _radiation(p: EnergyParameters, options: EnergyOptions,
               vegtyp, ist, ice, nsoil, sneqvo, sneqv, dt, cosz, snowh,
               tg, tv, fsno, qsnow, fwet, elai, esai, smc, solad, solai,
               fveg, albold, tauss):
    """ALBEDO, five assignments, SURRAD.  Nothing else is in this routine.

    ``FAGE`` (:2758) is an uninitialised local that WRF hands to ALBEDO with no
    INTENT.  ``SNOW_AGE`` writes it on the daytime path and ``SNOWALB_BATS``
    -- dead under ``opt_alb=2`` -- is the only reader, so the entry value is
    unobservable and 0.0 is passed.  ``FREVD``/``FREVI``/``FREGD``/``FREGI``
    are uninitialised locals too, and at ``COSZ <= 0`` ALBEDO returns them
    untouched; 0.0 is the defined behaviour this port produces.
    """
    alb = albedo(
        tau0=p.tau0, grain_growth=p.grain_growth,
        extra_growth=p.extra_growth, dirt_soot=p.dirt_soot, swemx=p.swemx,
        albsat=p.albsat, albdry=p.albdry, alblak=p.alblak,
        rhol=p.rhol, rhos=p.rhos, taul=p.taul, taus=p.taus, xl=p.xl,
        omegas=p.omegas, betads=p.betads, betais=p.betais,
        vegtyp=vegtyp, ist=ist, ice=ice, nsoil=nsoil,
        dt=dt, cosz=cosz, fage=f32(0.0), elai=elai, esai=esai,
        tg=tg, tv=tv, snowh=snowh, fsno=fsno, fwet=fwet, smc=smc,
        sneqvo=sneqvo, sneqv=sneqv, qsnow=qsnow, fveg=fveg,
        albold=albold, tauss=tauss,
        frevd_in=(0.0, 0.0), frevi_in=(0.0, 0.0),
        fregd_in=(0.0, 0.0), fregi_in=(0.0, 0.0),
        options=options.radiation_options(),
    )

    fsun = f32(alb["fsun"])
    fsha = f32(f32(1.0) - fsun)                                 # :2787
    laisun = f32(f32(elai) * fsun)                              # :2788
    laisha = f32(f32(elai) * fsha)                              # :2789
    vai = f32(f32(elai) + f32(esai))                            # :2790

    parsun, parsha, sav, sag, fsa, fsr, fsrv, fsrg = surrad(
        _RADIATION_MPE, fsun, fsha, elai, vai, laisun, laisha,
        solad, solai, alb["fabd"], alb["fabi"],
        alb["ftdd"], alb["ftid"], alb["ftii"],
        alb["albgrd"], alb["albgri"], alb["albd"], alb["albi"],
        alb["frevd"], alb["frevi"], alb["fregd"], alb["fregi"])

    return {
        "fsun": fsun, "laisun": laisun, "laisha": laisha,
        "parsun": f32(parsun), "parsha": f32(parsha),
        "sav": f32(sav), "sag": f32(sag), "fsa": f32(fsa), "fsr": f32(fsr),
        "fsrv": f32(fsrv), "fsrg": f32(fsrg),
        "albsnd": tuple(f32(v) for v in alb["albsnd"]),
        "albsni": tuple(f32(v) for v in alb["albsni"]),
        "bgap": f32(alb["bgap"]), "wgap": f32(alb["wgap"]),
        "albold": f32(alb["albold"]), "tauss": f32(alb["tauss"]),
    }


#: RADIATION is paused like the two flux leaves.  It is registered here rather
#: than listed in :data:`_HOST_LEAF` above because that dictionary is built
#: before ``_radiation`` exists.
register_host_leaf("radiation", _radiation)


# ---------------------------------------------------------------------------
# ENERGY -- module_sf_noahmplsm.F:1741-2396
# ---------------------------------------------------------------------------
def energy(*args, **kwargs) -> EnergyState:
    """One column of the Noah-MP surface energy balance, host-driven.

    Identical arithmetic to :func:`energy_steps`; the only difference is that
    the two paused leaf calls are answered here, immediately, by the CPython
    routines.  Every existing caller and every fixture test uses this.
    """
    return drive_on_host(energy_steps(*args, **kwargs))


def energy_steps(p: EnergyParameters, *,
                 ice: int, vegtyp: int, ist: int, isnow: int,
                 dt, rhoair, sfcprs, qair, sfctmp, thair, lwdn, uu, vv, zref,
                 co2air, o2air, solad, solai, cosz, igs, eair, tbot,
                 zsnso, zsoil, dzsnso, elai, esai, fwet, foln, fveg,
                 pahv, pahg, pahb, qsnow, lat, canliq, canice,
                 tv, tg, stc, snowh, eah, tah, sneqvo, sneqv, sh2o, smc,
                 snice, snliq, albold, cm, ch, dx, dz8w, q2, tauss,
                 qc, qsfc, psfc, acc_ssoil,
                 iloc: int = 1, jloc: int = 1,
                 nsnow: int = NSNOW, nsoil: int = NSOIL,
                 options: EnergyOptions = PINNED_OPTIONS):
    """One column of the Noah-MP surface energy balance, as a generator.

    Layer arrays are 0-based sequences spanning Fortran ``-nsnow+1 .. nsoil``
    (``zsnso``, ``dzsnso``, ``stc``) or ``-nsnow+1 .. 0`` (``snice``,
    ``snliq``); ``zsoil``, ``sh2o`` and ``smc`` span ``1 .. nsoil``.  Inputs
    are not mutated.

    The generator yields a :class:`LeafRequest` at VEGE_FLUX (vegetated tiles
    only) and again at BARE_FLUX (every column), and expects that routine's own
    return value back through ``send``.  It returns the :class:`EnergyState`.

    This is a control-flow seam and nothing else: the suspension points sit
    exactly where the physical call sites are, the frame is *not* restarted,
    and no value is recomputed.  A scheduler can therefore collect the same
    leaf from many columns and evaluate them in one device batch without any
    column repeating work or observing a partially mutated state.  Noah-MP has
    no horizontal coupling, so the order in which the collected calls are
    answered cannot change any column's answer.
    """
    options.check()
    if ice != 0:
        raise NotImplementedError(
            "ICE != 0 is the glacier column: it selects the ground-emissivity "
            "leg at module_sf_noahmplsm.F:2145-2147 and, in NOAHMP_SFLX, "
            "NOAHMP_GLACIER.  No fixture in this tree admits it")
    if ist != 1:
        raise NotImplementedError(
            "IST != 1 is a lake column: it selects the Z0MG leg at :2076-2082 "
            "and the RSURF/RHSUR leg at :2178-2181.  No fixture in this tree "
            "admits it")

    zsnso = [f32(v) for v in zsnso]
    dzsnso = [f32(v) for v in dzsnso]
    stc = [f32(v) for v in stc]
    snice = [f32(v) for v in snice]
    snliq = [f32(v) for v in snliq]
    sh2o = [f32(v) for v in sh2o]
    smc = [f32(v) for v in smc]
    zsoil = [f32(v) for v in zsoil]
    solad = tuple(f32(v) for v in solad)
    solai = tuple(f32(v) for v in solai)

    def L(seq, k):
        """Fortran ``seq(k)`` for an array declared ``(-nsnow+1:nsoil)``."""
        return seq[k + nsnow - 1]

    dt = f32(dt); rhoair = f32(rhoair); sfcprs = f32(sfcprs); qair = f32(qair)
    sfctmp = f32(sfctmp); thair = f32(thair); lwdn = f32(lwdn)
    uu = f32(uu); vv = f32(vv); zref = f32(zref)
    co2air = f32(co2air); o2air = f32(o2air); cosz = f32(cosz); igs = f32(igs)
    eair = f32(eair); tbot = f32(tbot)
    elai = f32(elai); esai = f32(esai); fwet = f32(fwet); foln = f32(foln)
    fveg = f32(fveg); pahv = f32(pahv); pahg = f32(pahg); pahb = f32(pahb)
    qsnow = f32(qsnow); lat = f32(lat)
    canliq = f32(canliq); canice = f32(canice)
    tv = f32(tv); tg = f32(tg); snowh = f32(snowh); eah = f32(eah)
    tah = f32(tah); sneqvo = f32(sneqvo); sneqv = f32(sneqv)
    albold = f32(albold); cm = f32(cm); ch = f32(ch)
    dx = f32(dx); dz8w = f32(dz8w); q2 = f32(q2); tauss = f32(tauss)
    qc = f32(qc); qsfc = f32(qsfc); psfc = f32(psfc)
    acc_ssoil = f32(acc_ssoil)

    st = EnergyState()
    br = st.branches

    # -- :2033-2053 initialise the vegetated-fraction fluxes -----------------
    tauxv = tauyv = f32(0.0)
    irc = shc = irg = shg = evg = evc = tr = ghv = f32(0.0)
    psnsun = psnsha = f32(0.0)
    t2mv = q2v = chv = chleaf = chuc = chv2 = f32(0.0)
    rb = f32(0.0)

    # -- :2057  UR.  The exponent is a REAL literal, so this is powf. --------
    ur = _mx(sqrtf(f32(f32(powf(uu, f32(2.0))) + f32(powf(vv, f32(2.0))))),
             f32(1.0))
    br["ur_clamped"] = ur == f32(1.0) and \
        f32(sqrtf(f32(f32(powf(uu, f32(2.0))) + f32(powf(vv, f32(2.0)))))) \
        < f32(1.0)

    # -- :2061-2063  vegetated or not ---------------------------------------
    vai = f32(elai + esai)
    veg = vai > f32(0.0)
    br["veg"] = veg

    # -- :2067-2073  ground snow-cover fraction [Niu and Yang, 2007] ---------
    fsno = f32(0.0)
    if snowh > f32(0.0):
        bdsno = f32(sneqv / snowh)
        fmelt = powf(f32(bdsno / f32(100.0)), p.mfsno)
        fsno = tanhf(f32(snowh / f32(p.scffac * fmelt)))
    br["fsno_snow"] = snowh > f32(0.0)

    # -- :2077-2085  ground roughness length (IST == 1 leg) ------------------
    z0mg = f32(f32(Z0 * f32(f32(1.0) - fsno)) + f32(fsno * p.z0sno))

    # -- :2089-2097  roughness length and displacement height ----------------
    zpdg = snowh
    if veg:
        z0m = p.z0mvt
        zpd = f32(f32(0.65) * p.hvt)
        if snowh > zpd:
            zpd = snowh
            br["zpd_snow"] = True
        else:
            br["zpd_snow"] = False
    else:
        z0m = z0mg
        zpd = zpdg

    # -- :2101-2106  urban override -----------------------------------------
    if p.urban_flag:
        z0mg = p.z0mvt
        zpdg = f32(f32(0.65) * p.hvt)
        z0m = z0mg
        zpd = zpdg
    br["urban"] = bool(p.urban_flag)

    # -- :2109-2110 ----------------------------------------------------------
    zlvl = f32(_mx(zpd, p.hvt) + zref)
    if zpdg >= zlvl:                                            # unreachable
        zlvl = f32(zpdg + zref)                 # for ZREF > 0; see PROVENANCE
    cwp = p.cwpvt                                               # :2115

    # -- :2119-2124  THERMOPROP ---------------------------------------------
    df_a, hcpct_a, snicev_a, snliqv_a, epore_a, fact_a = yield LeafRequest(
        "thermoprop", (), dict(
            isnow=isnow, ist=ist, dzsnso=np.asarray(dzsnso, np.float32), dt=dt,
            snowh=snowh, snice=np.asarray(snice, np.float32),
            snliq=np.asarray(snliq, np.float32),
            smc=np.asarray(smc, np.float32), sh2o=np.asarray(sh2o, np.float32),
            stc=np.asarray(stc, np.float32),
            smcmax=np.asarray(p.smcmax, np.float32), csoil=p.csoil,
            quartz=np.asarray(p.quartz, np.float32), urban_flag=p.urban_flag,
            nsnow=nsnow, nsoil=nsoil))
    df = [f32(float(v)) for v in df_a]
    hcpct = [f32(float(v)) for v in hcpct_a]
    fact = [f32(float(v)) for v in fact_a]

    # -- :2128-2135  RADIATION ----------------------------------------------
    rad = yield LeafRequest("radiation", (p, options), dict(
        vegtyp=vegtyp, ist=ist, ice=ice, nsoil=nsoil,
        sneqvo=sneqvo, sneqv=sneqv, dt=dt, cosz=cosz, snowh=snowh,
        tg=tg, tv=tv, fsno=fsno, qsnow=qsnow, fwet=fwet,
        elai=elai, esai=esai, smc=smc, solad=solad, solai=solai,
        fveg=fveg, albold=albold, tauss=tauss))
    laisun = rad["laisun"]
    laisha = rad["laisha"]
    parsun = rad["parsun"]
    parsha = rad["parsha"]
    sav = rad["sav"]
    sag = rad["sag"]
    fsa = rad["fsa"]
    fsr = rad["fsr"]
    albold = rad["albold"]
    tauss = rad["tauss"]

    # -- :2139-2147  emissivities -------------------------------------------
    emv = f32(f32(1.0) - expf(f32(f32(-f32(elai + esai)) / f32(1.0))))
    emg = f32(f32(p.eg[ist - 1] * f32(f32(1.0) - fsno))
              + f32(p.snow_emis * fsno))

    # -- :2151-2173  BTRAN (OPT_BTR == 1, Noah) ------------------------------
    btran = f32(0.0)
    btrani = [f32(0.0)] * nsoil
    for iz in range(1, p.nroot + 1):
        gx = f32(f32(sh2o[iz - 1] - p.smcwlt[iz - 1])
                 / f32(p.smcref[iz - 1] - p.smcwlt[iz - 1]))
        if gx < f32(0.0):
            br["btran_gx_low_clamp"] = True
        if gx > f32(1.0):
            br["btran_gx_high_clamp"] = True
        gx = _mn(f32(1.0), _mx(f32(0.0), gx))
        btrani[iz - 1] = _mx(
            MPE, f32(f32(L(dzsnso, iz) / f32(-zsoil[p.nroot - 1])) * gx))
        btran = f32(btran + btrani[iz - 1])
    btran = _mx(MPE, btran)
    for iz in range(1, p.nroot + 1):
        btrani[iz - 1] = f32(btrani[iz - 1] / btran)

    # -- :2177-2206  ground surface resistance (IST == 1, OPT_RSF == 1) ------
    _bevap = _mx(f32(0.0), f32(sh2o[0] / p.smcmax[0]))          # :2177, unread
    sat = _mn(f32(1.0), f32(sh2o[0] / p.smcmax[0]))
    br["rsurf_saturated"] = sat == f32(1.0)
    l_rsurf = f32(f32(f32(-zsoil[0])
                      * f32(expf(powf(f32(f32(1.0) - sat), p.rsurf_exp))
                            - f32(1.0)))
                  / f32(f32(2.71828) - f32(1.0)))
    d_rsurf = f32(f32(f32(f32(f32(2.2e-5) * p.smcmax[0]) * p.smcmax[0]))
                  * powf(f32(f32(1.0) - f32(p.smcwlt[0] / p.smcmax[0])),
                         f32(f32(2.0) + f32(f32(3.0) / p.bexp[0]))))
    rsurf = f32(l_rsurf / d_rsurf)
    if sh2o[0] < f32(0.01) and snowh == f32(0.0):               # :2199
        rsurf = f32(1.0e6)
        br["rsurf_dry_cap"] = True
    psi = f32(f32(-p.psisat[0])
              * powf(f32(_mx(f32(0.01), sh2o[0]) / p.smcmax[0]),
                     f32(-p.bexp[0])))
    rhsur = f32(fsno + f32(f32(f32(1.0) - fsno)
                           * expf(f32(f32(psi * GRAV) / f32(RW * tg)))))

    if p.urban_flag and snowh == f32(0.0):                      # :2204-2206
        rsurf = f32(1.0e6)
        br["rsurf_urban_cap"] = True

    # -- :2211-2227  psychrometric constants ---------------------------------
    if tv > TFRZ:
        latheav = HVAP
        frozen_canopy = False
    else:
        latheav = HSUB
        frozen_canopy = True
    gammav = f32(f32(CPAIR * sfcprs) / f32(f32(0.622) * latheav))

    if tg > TFRZ:
        latheag = HVAP
        frozen_ground = False
    else:
        latheag = HSUB
        frozen_ground = True
    gammag = f32(f32(CPAIR * sfcprs) / f32(f32(0.622) * latheag))
    br["frozen_canopy"] = frozen_canopy
    br["frozen_ground"] = frozen_ground

    # -- :2238-2262  VEGE_FLUX over the vegetated fraction -------------------
    tile_veg = veg and fveg > f32(0.0)
    br["tile_veg"] = tile_veg
    rssun = rssha = f32(0.0)
    canhs = f32(0.0)
    tgv = tg
    cmv = cm
    if tile_veg:
        tgv = tg
        cmv = cm
        chv = ch
        vf = yield LeafRequest("vege_flux", (
            p.vege, nsnow, nsoil, isnow,
            dt, sav, sag, lwdn, ur, uu, vv, sfctmp, thair, qair, eair,
            rhoair, snowh, vai, gammav, gammag, fwet, laisun, laisha, cwp,
            {k: L(dzsnso, k) for k in range(-nsnow + 1, nsoil + 1)},
            zlvl, zpd, z0m, fveg, z0mg, emv, emg, canliq, fsno, canice,
            {k: L(stc, k) for k in range(-nsnow + 1, nsoil + 1)},
            {k: L(df, k) for k in range(-nsnow + 1, nsoil + 1)},
            rsurf, latheav, latheag, parsun, parsha, igs, foln, co2air,
            o2air, btran, sfcprs, rhsur, q2, pahv, pahg, eah, tah, tv, tgv,
            cmv, chv, qc, qsfc, psfc, fsr), dict(
            opt_sfc=options.opt_sfc, opt_crs=options.opt_crs,
            opt_crop=options.opt_crop, opt_stc=options.opt_stc))
        eah = f32(vf.EAH); tah = f32(vf.TAH); tv = f32(vf.TV)
        tgv = f32(vf.TG); cmv = f32(vf.CM); chv = f32(vf.CH)
        tauxv = f32(vf.TAUXV); tauyv = f32(vf.TAUYV)
        irg = f32(vf.IRG); irc = f32(vf.IRC)
        shg = f32(vf.SHG); shc = f32(vf.SHC)
        evg = f32(vf.EVG); evc = f32(vf.EVC)
        tr = f32(vf.TR); ghv = f32(vf.GH)
        t2mv = f32(vf.T2MV)
        psnsun = f32(vf.PSNSUN); psnsha = f32(vf.PSNSHA)
        canhs = f32(vf.CANHS)
        qsfc = f32(vf.QSFC); q2v = f32(vf.Q2V); chv2 = f32(vf.CAH2)
        chleaf = f32(vf.CHLEAF); chuc = f32(vf.CHUC)
        rssun = f32(vf.RSSUN); rssha = f32(vf.RSSHA)

    # -- :2265-2276  BARE_FLUX over the bare fraction ------------------------
    tgb = tg
    cmb = cm
    chb = ch
    bf = yield LeafRequest("bare_flux", (), dict(
        nsnow=nsnow, nsoil=nsoil, isnow=isnow, dt=dt, sag=sag, lwdn=lwdn,
        ur=ur, uu=uu, vv=vv, sfctmp=sfctmp, thair=thair, qair=qair,
        eair=eair, rhoair=rhoair, snowh=snowh, dzsnso=dzsnso, zlvl=zlvl,
        zpd=zpdg, z0m=z0mg, fsno=fsno, emg=emg, stc=stc, df=df,
        rsurf=rsurf, lathea=latheag, gamma=gammag, rhsur=rhsur,
        iloc=iloc, jloc=jloc, q2=q2, pahb=pahb, tgb=tgb, cm=cmb, ch=chb,
        dx=dx, dz8w=dz8w, ivgtyp=vegtyp, qc=qc, qsfc=qsfc, psfc=psfc,
        sfcprs=sfcprs, urban_flag=p.urban_flag,
        opt_sfc=options.opt_sfc, opt_stc=options.opt_stc))
    tgb = f32(bf.tgb); cmb = f32(bf.cm); chb = f32(bf.ch); qsfc = f32(bf.qsfc)
    tauxb = f32(bf.tauxb); tauyb = f32(bf.tauyb)
    irb = f32(bf.irb); shb = f32(bf.shb); evb = f32(bf.evb); ghb = f32(bf.ghb)
    t2mb = f32(bf.t2mb); q2b = f32(bf.q2b); chb2 = f32(bf.ehb2)

    # -- :2282-2326  tile average --------------------------------------------
    if tile_veg:
        one_m = f32(f32(1.0) - fveg)
        taux = f32(f32(fveg * tauxv) + f32(one_m * tauxb))
        tauy = f32(f32(fveg * tauyv) + f32(one_m * tauyb))
        fira = f32(f32(f32(fveg * irg) + f32(one_m * irb)) + irc)
        fsh = f32(f32(f32(fveg * shg) + f32(one_m * shb)) + shc)
        fgev = f32(f32(fveg * evg) + f32(one_m * evb))
        ssoil = f32(f32(fveg * ghv) + f32(one_m * ghb))
        fcev = evc
        fctr = tr
        pah = f32(f32(f32(fveg * pahg) + f32(one_m * pahb)) + pahv)
        tg = f32(f32(fveg * tgv) + f32(one_m * tgb))
        t2m = f32(f32(fveg * t2mv) + f32(one_m * t2mb))
        ts = f32(f32(fveg * tv) + f32(one_m * tgb))
        cm = f32(f32(fveg * cmv) + f32(one_m * cmb))
        ch = f32(f32(fveg * chv) + f32(one_m * chb))
        q1 = f32(f32(fveg * f32(f32(eah * f32(0.622))
                                / f32(sfcprs - f32(f32(0.378) * eah))))
                 + f32(one_m * qsfc))
        q2e = f32(f32(fveg * q2v) + f32(one_m * q2b))
        z0wrf = z0m
    else:
        taux = tauxb
        tauy = tauyb
        fira = irb
        fsh = shb
        fgev = evb
        ssoil = ghb
        tg = tgb
        t2m = t2mb
        fcev = f32(0.0)
        fctr = f32(0.0)
        pah = pahb
        ts = tg
        cm = cmb
        ch = chb
        q1 = qsfc
        q2e = q2b
        rssun = f32(0.0)
        rssha = f32(0.0)
        tgv = tgb
        chv = chb
        z0wrf = z0mg

    # -- :2321-2329  the emitted-longwave sanity check -----------------------
    fire = f32(lwdn + fira)
    if fire <= f32(0.0):
        raise ValueError(
            "emitted longwave <= 0: SHDFAC is inconsistent with LAI "
            "(module_sf_noahmplsm.F:2323-2329 calls wrf_error_fatal here)")

    # -- :2332-2340 ----------------------------------------------------------
    emissi = f32(f32(fveg * f32(f32(f32(emg * f32(f32(1) - emv)) + emv)
                                + f32(f32(emv * f32(f32(1) - emv))
                                      * f32(f32(1) - emg))))
                 + f32(f32(f32(1) - fveg) * emg))
    trad = powf(f32(f32(fire - f32(f32(f32(1) - emissi) * lwdn))
                    / f32(emissi * SB)), f32(0.25))

    apar = f32(f32(parsun * laisun) + f32(parsha * laisha))     # :2344
    psn = f32(f32(psnsun * laisun) + f32(psnsha * laisha))      # :2345

    # -- :2349-2371  the soil column ----------------------------------------
    eflxb = f32(0.0)
    acc_ssoil = f32(acc_ssoil + ssoil)
    if options.calculate_soil:
        ssoil_avg = f32(acc_ssoil / f32(options.soil_update_steps))
        dt_soil = f32(dt * f32(options.soil_update_steps))
        stc_a, eflxb = yield LeafRequest("tsnosoi", (), dict(
            isnow=isnow, zsnso=np.asarray(zsnso, np.float32),
            stc=np.asarray(stc, np.float32), tbot=tbot, ssoil=ssoil_avg,
            dt=dt_soil, snowh=snowh, df=np.asarray(df, np.float32),
            hcpct=np.asarray(hcpct, np.float32), zbot=p.zbot,
            nsnow=nsnow, nsoil=nsoil))
        stc = [f32(float(v)) for v in stc_a]
        eflxb = f32(f32(eflxb) * dt_soil)

    # :2384-2396 is the OPT_STC == 2 snow-surface reset; opt_stc = 1 here.

    # -- :2400-2404  PHASECHANGE --------------------------------------------
    (stc_a, snice_a, snliq_a, sneqv, snowh, smc_a, sh2o_a, qmelt, ponding,
     imelt_a) = yield LeafRequest("phasechange", (), dict(
        isnow=isnow, ist=ist, dt=dt, fact=np.asarray(fact, np.float32),
        dzsnso=np.asarray(dzsnso, np.float32),
        stc=np.asarray(stc, np.float32), snice=np.asarray(snice, np.float32),
        snliq=np.asarray(snliq, np.float32), sneqv=sneqv, snowh=snowh,
        smc=np.asarray(smc, np.float32), sh2o=np.asarray(sh2o, np.float32),
        smcmax=np.asarray(p.smcmax, np.float32),
        psisat=np.asarray(p.psisat, np.float32),
        bexp=np.asarray(p.bexp, np.float32), nsnow=nsnow, nsoil=nsoil))

    # ---- pack --------------------------------------------------------------
    st.z0wrf = z0wrf
    st.imelt = tuple(int(v) for v in imelt_a)
    st.snicev = tuple(f32(float(v)) for v in snicev_a)
    st.snliqv = tuple(f32(float(v)) for v in snliqv_a)
    st.epore = tuple(f32(float(v)) for v in epore_a)
    st.hcpct = tuple(hcpct)
    st.t2m = t2m; st.fsno = fsno; st.sav = sav; st.sag = sag
    st.qmelt = f32(qmelt); st.fsa = fsa; st.fsr = fsr
    st.taux = taux; st.tauy = tauy; st.fira = fira; st.fsh = fsh
    st.fcev = fcev; st.fgev = fgev; st.fctr = fctr; st.trad = f32(trad)
    st.psn = psn; st.apar = apar; st.ssoil = ssoil
    st.btrani = tuple(btrani); st.btran = btran
    st.ponding = f32(ponding); st.ts = ts
    st.latheav = latheav; st.latheag = latheag
    st.frozen_canopy = frozen_canopy; st.frozen_ground = frozen_ground
    st.tv = tv; st.tg = tg
    st.stc = tuple(f32(float(v)) for v in stc_a)
    st.snowh = f32(snowh); st.eah = eah; st.tah = tah
    st.sneqvo = sneqvo; st.sneqv = f32(sneqv)
    st.sh2o = tuple(f32(float(v)) for v in sh2o_a)
    st.smc = tuple(f32(float(v)) for v in smc_a)
    st.snice = tuple(f32(float(v)) for v in snice_a)
    st.snliq = tuple(f32(float(v)) for v in snliq_a)
    st.albold = albold; st.cm = cm; st.ch = ch; st.tauss = tauss
    st.laisun = laisun; st.laisha = laisha; st.rb = rb
    st.qsfc = qsfc
    st.t2mv = t2mv; st.t2mb = t2mb
    st.fsrv = rad["fsrv"]; st.fsrg = rad["fsrg"]
    st.rssun = rssun; st.rssha = rssha
    st.albsnd = rad["albsnd"]; st.albsni = rad["albsni"]
    st.bgap = rad["bgap"]; st.wgap = rad["wgap"]
    st.tgv = tgv; st.tgb = tgb
    st.q1 = q1; st.q2v = q2v; st.q2b = q2b; st.q2e = q2e
    st.chv = chv; st.chb = chb; st.emissi = emissi
    st.pah = pah; st.canhs = canhs
    st.shg = shg; st.shc = shc; st.shb = shb
    st.evg = evg; st.evb = evb
    st.ghv = ghv; st.ghb = ghb
    st.irg = irg; st.irc = irc; st.irb = irb
    st.tr = tr; st.evc = evc
    st.chleaf = chleaf; st.chuc = chuc; st.chv2 = chv2; st.chb2 = chb2
    st.eflxb = f32(eflxb); st.acc_ssoil = acc_ssoil

    # ---- WRF leaves these undefined; produce the defined 0.0 ---------------
    st.hcpct = tuple(f32(0.0) if k <= isnow else st.hcpct[k + nsnow - 1]
                     for k in range(-nsnow + 1, nsoil + 1))
    st.imelt = tuple(0 if k <= isnow else st.imelt[k + nsnow - 1]
                     for k in range(-nsnow + 1, nsoil + 1))
    st.snicev = tuple(f32(0.0) if k <= isnow else st.snicev[k + nsnow - 1]
                      for k in range(-nsnow + 1, 1))
    st.snliqv = tuple(f32(0.0) if k <= isnow else st.snliqv[k + nsnow - 1]
                      for k in range(-nsnow + 1, 1))
    st.epore = tuple(f32(0.0) if k <= isnow else st.epore[k + nsnow - 1]
                     for k in range(-nsnow + 1, 1))
    st.btrani = tuple(v if k <= p.nroot else f32(0.0)
                      for k, v in enumerate(st.btrani, start=1))
    if cosz <= f32(0.0):
        st.fsrv = f32(0.0); st.fsrg = f32(0.0)
        st.bgap = f32(0.0); st.wgap = f32(0.0)

    br["isnow"] = int(isnow)
    return st


#: :func:`energy` forwards its whole argument list to :func:`energy_steps`, so
#: make introspection say so.  Without this, ``inspect.signature(energy)``
#: reports ``(*args, **kwargs)`` and every test that asserts a particular
#: argument is *absent* passes for free -- a gate that cannot fail.
energy.__wrapped__ = energy_steps
