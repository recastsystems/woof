"""WRF v4.6.1 WDM6 coefficient initialization, and the WDM family table.

The values and operation order follow ``wdm6init`` in
``phys/module_mp_wdm6.F`` (:2085-2211 of the byte-frozen reference bundle).
WDM6's cold half shares WSM6's rimed-ice identity exactly -- ``wdm6init``'s
``hail_opt`` branch (:2096-2108) sets the same five constants
``mp_wsm6_init`` does -- so :func:`woof.core.wsm6_constants.
rimed_ice_constants` is imported rather than copied, and the Weierstrass
Gamma is the same shared ``_rgmma``.

Differences from WSM6's block that are the double-moment warm rain:

- rain PSD is gamma with shape mu = 1 and a PROGNOSTIC intercept:
  ``lamdar(q, rho, nr) = (pidnr*nr/(q*rho))**(1/3)`` with
  ``pidnr = 4*pi*denr`` (:2143), against WSM6's fixed-N0r fourth root.
- ``pvtr = avtr*Gamma(5+bvtr)/24`` (:2136) is the gamma-PSD mass-weighted
  rain fall speed; ``pvtrn = avtr*Gamma(2+bvtr)`` (:2137) is the
  number-weighted one that moves ``nr``.
- ``precr1 = 2*pi*1.56`` / ``precr2 = 2*pi*0.31*sqrt(avtr)*Gamma(3.5+
  0.5*bvtr)`` (:2140-2141) carry NO ``n0r`` factor: rain evaporation is
  scaled by the prognostic ``nr`` at the call site (:1240).
- ``qck1 = 0.104*9.8*peaut/denr**(1/3)/xmyu*den0**(4/3)`` (:2115) has no
  droplet-number factor; autoconversion applies ``nc**(-1/3)`` explicitly
  (:1172).
- cloud droplets carry their own slope bounds (``lamdacmin/max``,
  :69-70) and ``pidnc = pi*denr/6`` (:2116).
- the land/sea autoconversion thresholds ``qc0``/``qc1`` derive from
  ``xncr0 = 5e7`` / ``xncr1 = 5e8`` (:2113-2114); the WSM6-era ``xncr =
  3e8`` parameter (:77) is DECLARED but read nowhere in the v4.6.1 module,
  so it deliberately has no counterpart here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

from woof.core.wsm6_constants import (WSM6RimedIceConstants, _rgmma,
                                       rimed_ice_constants)

#: The prognostic moments WDM6 adds to WSM6's mass set, in the order
#: ``module_mp_wdm6.F`` packs them into ``ncr`` (:236-238): CCN, cloud
#: droplet number, rain number -- WRF's ``scalar:qnn,qnc,qnr``
#: (Registry.EM_COMMON:3031).  It lives in this leaf module, which imports
#: nothing from ``woof.core.state``, so the state allocator and the WDM6
#: adapter can both name the set without an import cycle.
WDM6_NUMBER_SPECIES = ("nn", "nc", "nr")

#: ``kernels/wdm6.cu``'s ``WDM6_KMAX`` ladder.  One CUDA thread owns a whole
#: column, so the bound sizes ~30 per-thread arrays and the driver reserves
#: a local-memory backing store for the device's entire resident-thread
#: capacity at first launch -- the law at the head of
#: ``woof/core/preflight.py``.  The launcher therefore compiles the
#: SHALLOWEST tier that holds the configuration's ``nz``, and it compiles a
#: TIER rather than ``nz`` itself (kf/refl's shape) so that a nested tree
#: with several level counts loads at most two WDM6 modules.
#:
#: 64 is the ``#ifndef`` default baked into the source, which is why it is
#: also the value the driver-regeneration gate
#: (``tests/test_preflight.py::test_the_recorded_local_frames_match_the_
#: driver``) reads back for the unspecialized module, and therefore the
#: value recorded in ``KERNEL_MAX_LOCAL_SIZE_BYTES``.  The 80 tier prices
#: ABOVE that row; ``preflight.WDM6_TIER_FRAME`` is what carries it.
#:
#: This ladder lives in the constants module, not in ``woof/core/wdm6.py``,
#: for one reason: the adapter imports CuPy at module scope and the memory
#: estimator must price a configuration on a host with no device at all.
WDM6_SHALLOW_KMAX = 64
WDM6_DEEP_KMAX = 80
WDM6_KERNEL_LEVEL_TIERS = (WDM6_SHALLOW_KMAX, WDM6_DEEP_KMAX)
#: ``(minimum, maximum)`` levels WDM6 admits: 2 (the scheme needs a layer
#: above and below to sediment between) up to the deepest compiled tier.
WDM6_VERTICAL_LEVEL_BOUNDS = (2, WDM6_DEEP_KMAX)


def wdm6_level_tier(nz: int) -> int:
    """Return the ``WDM6_KMAX`` tier ``wdm6.cu`` compiles for ``nz``."""
    nz = int(nz)
    minimum, maximum = WDM6_VERTICAL_LEVEL_BOUNDS
    if not minimum <= nz <= maximum:
        raise ValueError(
            f"WDM6 requires {minimum} <= nz <= {maximum}, got {nz}")
    return next(tier for tier in WDM6_KERNEL_LEVEL_TIERS if nz <= tier)


#: ``kernels/wdm5.cu``'s ``WDM5_KMAX`` ladder, on WDM6's shape for WDM6's
#: reason (one thread per column, ~25 per-thread level arrays, the driver's
#: local-memory backing store sized by the deepest frame).  64 is the
#: source's ``#ifndef`` default.
WDM5_SHALLOW_KMAX = 64
WDM5_DEEP_KMAX = 80
WDM5_KERNEL_LEVEL_TIERS = (WDM5_SHALLOW_KMAX, WDM5_DEEP_KMAX)
#: ``(minimum, maximum)`` levels WDM5 admits: 2 (a layer above and below to
#: sediment between) up to the deepest compiled tier.
WDM5_VERTICAL_LEVEL_BOUNDS = (2, WDM5_DEEP_KMAX)


def wdm5_level_tier(nz: int) -> int:
    """Return the ``WDM5_KMAX`` tier ``wdm5.cu`` compiles for ``nz``."""
    nz = int(nz)
    minimum, maximum = WDM5_VERTICAL_LEVEL_BOUNDS
    if not minimum <= nz <= maximum:
        raise ValueError(
            f"WDM5 requires {minimum} <= nz <= {maximum}, got {nz}")
    return next(tier for tier in WDM5_KERNEL_LEVEL_TIERS if nz <= tier)


@dataclass(frozen=True)
class WdmScheme:
    """One WRF WDM option, as the WDM family adapter runs it.

    The family is WRF's double-moment warm rain (cloud droplet number, rain
    number and a prognostic CCN reservoir) beside a WSM ice half.  A sibling
    is admitted by a row here and its physics-registry row
    (tools/build_registry.py), never by an adapter of its own:
    :func:`woof.core.wdm6.apply` reads everything scheme-specific from the
    row.
    """
    mp_physics: int
    name: str
    #: CUDA translation unit under woof/core/kernels.
    kernel_unit: str
    #: WRF's ``moist`` package for the option (Registry.EM_COMMON).
    mass_species: tuple
    #: WRF's ``scalar`` package: CCN, cloud and rain number (qnn,qnc,qnr).
    number_species: tuple
    #: True when the ice half carries a graupel category (and fallout).
    graupel: bool
    #: True when process rates read the XLAND land/sea mask.
    land_sea_mask: bool
    vertical_level_bounds: tuple
    hail: bool = False


#: Registry.EM_COMMON:3030-3031 (wdm5scheme, wdm6scheme).
WDM_SCHEMES = {
    14: WdmScheme(14, "WDM5", "wdm5", ("qv", "qc", "qr", "qi", "qs"),
                  WDM6_NUMBER_SPECIES, graupel=False, land_sea_mask=False,
                  vertical_level_bounds=WDM5_VERTICAL_LEVEL_BOUNDS),
    16: WdmScheme(16, "WDM6", "wdm6", ("qv", "qc", "qr", "qi", "qs", "qg"),
                  WDM6_NUMBER_SPECIES, graupel=True, land_sea_mask=True,
                  vertical_level_bounds=WDM6_VERTICAL_LEVEL_BOUNDS),
    26: WdmScheme(26, "WDM7", "wdm7", ("qv", "qc", "qr", "qi", "qs", "qg", "qh"),
                  WDM6_NUMBER_SPECIES, graupel=True, land_sea_mask=True,
                  vertical_level_bounds=WDM6_VERTICAL_LEVEL_BOUNDS, hail=True),

}

WDM_SCHEME_NAMES = {selector: row.name for selector, row in WDM_SCHEMES.items()}
WDM_HAIL_SCHEMES = frozenset(selector for selector, row in WDM_SCHEMES.items() if row.hail)
WDM7_HAIL_SPECIES = ("qh",)


def wdm_scheme(mp_physics: int) -> WdmScheme | None:
    """The WDM family row for ``mp_physics``, or None for any other scheme."""
    return WDM_SCHEMES.get(int(mp_physics))


@dataclass(frozen=True)
class WDM6Coefficients:
    rimed: WSM6RimedIceConstants
    xlv1: float
    qc0: float          # maritime autoconversion threshold (xncr0 = 5e7)
    qc1: float          # continental autoconversion threshold (xncr1 = 5e8)
    qck1: float
    pidnc: float
    pidnr: float
    pidn0s: float
    pidn0g: float
    pvtr: float
    pvtrn: float
    pvts: float
    pvtg: float
    pacrc: float
    pacrg: float
    precr1: float
    precr2: float
    precs1: float
    precs2: float
    precg1: float
    precg2: float
    g4pbr: float
    g7pbr: float
    xmmax: float
    roqimax: float
    rslopecmax: float
    rslopec2max: float
    rslopec3max: float
    rslopermax: float
    rslopesmax: float
    rslopegmax: float
    rsloperbmax: float
    rslopesbmax: float
    rslopegbmax: float
    rsloper2max: float
    rslopes2max: float
    rslopeg2max: float
    rsloper3max: float
    rslopes3max: float
    rslopeg3max: float
    cdm2: float         # effectRad_wdm6's rgmma(5/3) (:3189)


@lru_cache(maxsize=2)
def coefficients(hail_opt: int = 0) -> WDM6Coefficients:
    """Reproduce ``wdm6init`` (module_mp_wdm6.F:2085-2211) in float64.

    Inputs are the driver's values (phys/module_microphysics_driver.F:
    2476-2500 via phy_init): den0 = rhoair0 = 1.28, denr = rhowater = 1000,
    dens = rhosnow = 100, cl = cliq = 4190, cpv = 1846.4.
    """
    r = rimed_ice_constants(hail_opt)
    pi = 4.0 * math.atan(1.0)                          # :2110
    den0, denr, dens = 1.28, 1000.0, 100.0
    cliq, cpv = 4190.0, 1846.4
    n0r, n0s = 8.0e6, 2.0e6
    avtr, bvtr, avts, bvts = 841.9, 0.8, 11.72, 0.41
    r0, peaut, xmyu = 0.8e-5, 0.55, 1.718e-5
    xncr0, xncr1 = 5.0e7, 5.0e8
    dicon, dimax = 11.9, 500.0e-6
    lamdacmax, lamdarmax, lamdasmax = 5.0e5, 5.0e4, 1.0e5

    g2pbr = _rgmma(2.0 + bvtr)
    g3pbr = _rgmma(3.0 + bvtr)
    g4pbr = _rgmma(4.0 + bvtr)
    g5pbr = _rgmma(5.0 + bvtr)
    g7pbr = _rgmma(7.0 + bvtr)
    g7pbro2 = _rgmma(3.5 + 0.5 * bvtr)
    g3pbs, g4pbs = _rgmma(3.0 + bvts), _rgmma(4.0 + bvts)
    g5pbso2 = _rgmma(2.5 + 0.5 * bvts)
    g3pbg, g4pbg = _rgmma(3.0 + r.bvtg), _rgmma(4.0 + r.bvtg)
    g5pbgo2 = _rgmma(2.5 + 0.5 * r.bvtg)

    rc, rr = 1.0 / lamdacmax, 1.0 / lamdarmax
    rs, rg = 1.0 / lamdasmax, 1.0 / r.lamdagmax
    return WDM6Coefficients(
        rimed=r,
        xlv1=cliq - cpv,                               # :2111
        qc0=4.0 / 3.0 * pi * denr * r0 ** 3 * xncr0 / den0,   # :2113
        qc1=4.0 / 3.0 * pi * denr * r0 ** 3 * xncr1 / den0,   # :2114
        qck1=(0.104 * 9.8 * peaut / denr ** (1.0 / 3.0)
              / xmyu * den0 ** (4.0 / 3.0)),           # :2115
        pidnc=pi * denr / 6.0,                         # :2116
        pidnr=4.0 * pi * denr,                         # :2143
        pidn0s=pi * dens * n0s,                        # :2160
        pidn0g=pi * r.deng * r.n0g,                    # :2176
        pvtr=avtr * g5pbr / 24.0,                      # :2136
        pvtrn=avtr * g2pbr,                            # :2137
        pvts=avts * g4pbs / 6.0,                       # :2156
        pvtg=r.avtg * g4pbg / 6.0,                     # :2173
        pacrc=pi * n0s * avts * g3pbs * 0.25,          # :2162 (eacrc = 1)
        pacrg=pi * r.n0g * r.avtg * g3pbg * 0.25,      # :2172
        precr1=2.0 * pi * 1.56,                        # :2140
        precr2=2.0 * pi * 0.31 * math.sqrt(avtr) * g7pbro2,   # :2141
        precs1=4.0 * n0s * 0.65,                       # :2158
        precs2=4.0 * n0s * 0.44 * math.sqrt(avts) * g5pbso2,  # :2159
        precg1=2.0 * pi * r.n0g * 0.78,                # :2174
        precg2=2.0 * pi * r.n0g * 0.31 * math.sqrt(r.avtg) * g5pbgo2,
        g4pbr=g4pbr,                                   # :2130 (niacr)
        g7pbr=g7pbr,                                   # :2133 (piacr)
        xmmax=(dimax / dicon) ** 2,                    # :2145
        roqimax=2.08e22 * dimax ** 8,                  # :2146
        rslopecmax=rc, rslopec2max=rc * rc, rslopec3max=rc ** 3,
        rslopermax=rr, rslopesmax=rs, rslopegmax=rg,
        rsloperbmax=rr ** bvtr, rslopesbmax=rs ** bvts,
        rslopegbmax=rg ** r.bvtg,
        rsloper2max=rr * rr, rslopes2max=rs * rs, rslopeg2max=rg * rg,
        rsloper3max=rr ** 3, rslopes3max=rs ** 3, rslopeg3max=rg ** 3,
        cdm2=_rgmma(5.0 / 3.0),
    )


#: The float32 words WRF v4.6.1's own ``wdm6init`` leaves in its saved
#: variables, as bit patterns, read back from the Fortran by
#: tools/wdm6_wrf461_oracle (gfortran 13.3 -O0, glibc 2.39; the oracle's
#: ``constants-*.txt`` dumps).  ``kernels/wdm6.cu`` bakes exactly these words
#: (tests/test_wdm6.py pins the correspondence) because they are WRF's
#: coefficients: ``wdm6init`` declares them default REAL, so ``rgmma``'s
#: 10,000-term product and every derived product are float32 arithmetic.
#: :func:`coefficients` evaluates the same formulas in float64; its values
#: differ from these words by 1 to about 70 ULP (rgmma's float32
#: accumulation for the fall speeds, single roundings such as pidnr's
#: 4*pi*denr for the rest) and stay as the readable derivation, checked
#: against the words to 1e-5 relative.  The column oracle grades the kernel
#: bitwise against WRF with these words in place
#: (tests/test_wdm6_wrf461_oracle.py).
WDM6_WRF_WORDS: dict[str, int] = {
    "pi": 0x40490FDB, "qc0": 0x38AFB0CB, "qc1": 0x3A5B9CFE,
    "qck1": 0x458DB55A, "pidnc": 0x4402E653, "g4pbr": 0x418E89D3,
    "g7pbr": 0x45526AA0, "pvtr": 0x453B679F, "pvtrn": 0x44B05C29,
    "precr1": 0x411CD40C, "precr2": 0x4395A266, "pidnr": 0x4644597C,
    "roqimax": 0x38AA64C8, "pvts": 0x41A069C1, "precs1": 0x4A9EB100,
    "precs2": 0x4B8E882C, "pidn0s": 0x4E15CD85, "pacrc": 0x4C537EB0,
    "rslopecmax": 0x360637BD, "rslopec2max": 0x2C8CBCCC,
    "rslopec3max": 0x231392EE, "rslopermax": 0x37A7C5AC,
    "rsloperbmax": 0x39369152, "rsloper2max": 0x2FDBE6FE,
    "rsloper3max": 0x28101D7C, "rslopesmax": 0x3727C5AC,
    "rslopesbmax": 0x3C1205C6, "rslopes2max": 0x2EDBE6FE,
    "rslopes3max": 0x26901D7C, "cdm2": 0x3F6711B5,
}

#: The hail_opt-dependent words (wdm6init :2096-2108 and the products of
#: n0g/deng/avtg/lamdagmax), same provenance.
WDM6_WRF_RIMED_WORDS: dict[int, dict[str, int]] = {
    0: {"pidn0g": 0x4FBB40E7, "pvtg": 0x4474FCE3, "pacrg": 0x4F90EE38,
        "precg1": 0x4B959029, "precg2": 0x4D768C0C,
        "rslopegmax": 0x378BCF65, "rslopegbmax": 0x391DCA2F,
        "rslopeg2max": 0x2F98B5BF, "rslopeg3max": 0x27A6CCCC},
    1: {"pidn0g": 0x4CA7C777, "pvtg": 0x44539495, "pacrg": 0x4C2036D0,
        "precg1": 0x483F70D8, "precg2": 0x4A12A338,
        "rslopegmax": 0x3851B717, "rslopegbmax": 0x39BDFF22,
        "rslopeg2max": 0x312BCC76, "rslopeg3max": 0x2A0CBCCB},
}


#: The hail words WRF v4.6.1's own ``wdm7init`` leaves in its saved
#: variables (module_mp_wdm7.F:2457-2488; n0h = 4e4, avth = 285, bvth = 0.8,
#: denh = 912, lamdahmax = 2e4), read back from the Fortran by
#: tools/wdm7_wrf461_oracle (gfortran 13.3 -O0, glibc 2.39).  Every other
#: wdm7init word equals a WDM6 word in :data:`WDM6_WRF_WORDS` or the
#: ``hail_opt = 0`` row of :data:`WDM6_WRF_RIMED_WORDS` (the oracle's
#: constants dump shows it); all but ``pidn0h`` here also equal the
#: ``hail_opt = 1`` graupel words, which share n0/avt/bvt/lambda-max with
#: WDM7's hail and differ only in density. ``kernels/wdm7.cu`` bakes these
#: words; tests/test_wdm7_wrf461_oracle.py pins them.
WDM7_WRF_HAIL_WORDS: dict[str, int] = {
    "pvth": 0x44539495, "pacrh": 0x4C2036D0, "prech1": 0x483F70D8,
    "prech2": 0x4A12A338, "pidn0h": 0x4CDA979F,
    "rslopehmax": 0x3851B717, "rslopehbmax": 0x39BDFF22,
    "rslopeh2max": 0x312BCC76, "rslopeh3max": 0x2A0CBCCB,
}


def wrf_word(bits: int) -> float:
    """The float32 value of one WRF word (returned as a Python float)."""
    import struct
    return struct.unpack("<f", struct.pack("<I", bits))[0]


__all__ = ["WDM5_DEEP_KMAX", "WDM5_KERNEL_LEVEL_TIERS", "WDM5_SHALLOW_KMAX",
           "WDM5_VERTICAL_LEVEL_BOUNDS", "WDM6Coefficients", "WDM6_DEEP_KMAX",
           "WDM6_KERNEL_LEVEL_TIERS", "WDM6_NUMBER_SPECIES",
           "WDM6_SHALLOW_KMAX", "WDM6_VERTICAL_LEVEL_BOUNDS", "WDM_SCHEMES",
           "WdmScheme", "coefficients", "rimed_ice_constants",
           "wdm5_level_tier", "wdm6_level_tier", "wdm_scheme", "wrf_word",
           "WDM6_WRF_WORDS", "WDM6_WRF_RIMED_WORDS", "WDM7_WRF_HAIL_WORDS",
           "WDM7_HAIL_SPECIES", "WDM_HAIL_SCHEMES", "WDM_SCHEME_NAMES"]
