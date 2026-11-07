"""Data-driven neutral exchange curves, evaluated with NumPy or CuPy.

Equations implemented independently from Green and Zhang (2013),
doi:10.1175/MWR-D-12-00274.1, equations 7--13. The observational rows
explicitly distinguish measurements from extensions of their wind range.
No observational momentum study is treated as a measurement of enthalpy.
"""
from __future__ import annotations

import math
from types import MappingProxyType

GZ = "10.1175/MWR-D-12-00274.1"
CURVES = MappingProxyType({
    "gz-option0": dict(momentum="charnock", scalar="mm5", alpha=.0185,
                       cap=None, continuation=1., doi=GZ),
    "gz-option1": dict(momentum="blend", scalar="constant_roughness", alpha=.011,
                       cap=.00285, continuation=0., doi=GZ),
    "gz-option2": dict(momentum="blend", scalar="reynolds", alpha=.011,
                       cap=.00285, continuation=0., doi=GZ),
    "gz-psu": dict(momentum="blend", scalar="reynolds", alpha=.011,
                   cap=.00285, continuation=.5, doi=GZ),
    "gz-k2d0": dict(momentum="charnock", scalar="reynolds", alpha=.0185,
                    cap=None, continuation=1., doi=GZ),
    "powell2003": dict(momentum="anchors", scalar="equal_exchange", ck=.001,
        wind=(27.3, 33.0, 40.8, 50.8), cd=(.00198, .00215, .00185, .00148),
        doi="10.1038/nature01481",
        note="Figure 3c, 10--150 m square symbols, manually digitized; approximate "
             "uncertainty 0.1e-3. Linear interpolation. GZ option2 below 20 m/s, "
             "blend to first point, hold last point above 50.8 m/s. "
             "Ck=1e-3 is a separately selected CBLAST mean, not this paper's result."),
    "donelan2004": dict(momentum="anchors", scalar="equal_exchange", ck=.001,
        wind=(33., 120.), cd=(.0025, .0025), doi="10.1029/2004GL019460",
        note="Limiting drag 2.5e-3 above 33 m/s; GZ option2 below 20 m/s, "
             "linear transition to 33 m/s. Ck uses the separately selected CBLAST mean."),
    "bell2012": dict(momentum="anchors", scalar="equal_exchange", ck=.001,
        wind=(52., 72.), cd=(.0024, .0024), doi="10.1175/JAS-D-11-0276.1",
        note="CBLAST mean, observed range 52--72 m/s, Cd SD 46%, Ck SD 40%. "
             "GZ option2 below 20 m/s and linear transition to 52 m/s. "
             "Constant extension above 72 m/s is an assumption."),
    "emanuel2003-adapted": dict(momentum="linear_cap", scalar="equal_exchange",
        intercept=.0008, slope=.00004, cap_wind=30., ck=.002,
        doi="10.1175/1520-0469(2003)060<1420:ASHFAE>2.0.CO;2",
        note="Capped equal-coefficient experiment in section 4. "
             "The paper uses gradient wind; using U10N here is an explicit "
             "experimental adaptation, not a measured U10 curve."),
})


def _roughness(ustar, row, xp):
    z1 = row["alpha"] * ustar**2 / 9.81 + 1.59e-5
    if row["momentum"] == "charnock":
        z0 = z1
    else:
        w = xp.minimum(1., (ustar / 1.06)**.3)
        z2 = 10. * xp.exp(-9.5 * xp.maximum(ustar, 1e-8)**(-1/3))
        z2 += 1.65e-6 / xp.maximum(ustar, .01)
        z0 = w*z2 + (1.-w)*z1
    if row["cap"] is not None:
        z0 = xp.minimum(z0, row["cap"]) + row["continuation"] * xp.maximum(0., z0-row["cap"])
    z0 = xp.maximum(1.27e-7, z0)
    if row["scalar"] == "mm5":
        zt = z0
        zq = 1. / (1./z0 + .4*ustar/2.4e-5)
    elif row["scalar"] == "constant_roughness":
        zt = xp.full_like(z0, 1e-4)
        zq = zt
    else:
        re = ustar*z0/1.5e-5
        zt = z0*xp.exp(-.4*(7.3*re**.25*math.sqrt(.71)-5.))
        zq = z0*xp.exp(-.4*(7.3*re**.25*math.sqrt(.60)-5.))
    return z0, zt, zq


def coefficients(wind, name, *, xp):
    """Return neutral Cd, Ch, Cq; Ck needs temperature/humidity weighting.

    Independent points are vectorized. Production passes CuPy, so neither
    table construction nor the neutral root solve runs on a single CPU core.
    The bisection solves U10N=ustar/k*log(10/z0(ustar)).
    """
    row = CURVES[name]
    wind = xp.asarray(wind, dtype=xp.float64)
    if row["momentum"] in ("charnock", "blend"):
        low = xp.full_like(wind, 1e-10)
        high = xp.full_like(wind, 20.)
        for _ in range(48):
            u = (low+high)*.5
            z0, _, _ = _roughness(u, row, xp)
            below = u/.4*xp.log(10./z0) < wind
            low = xp.where(below, u, low)
            high = xp.where(below, high, u)
        z0, zt, zq = _roughness((low+high)*.5, row, xp)
        lm = xp.log(10./z0)
        return .16/lm**2, .16/(lm*xp.log(10./zt)), .16/(lm*xp.log(10./zq))
    if row["momentum"] == "linear_cap":
        cd = row["intercept"] + row["slope"]*xp.minimum(wind, row["cap_wind"])
        # The capped equal-exchange experiment also has equal coefficients
        # below its cap; the tabulated ck records the high-wind limit.
        return cd, cd, cd
    base = coefficients(wind, "gz-option2", xp=xp)
    anchors = xp.asarray(row["wind"], dtype=xp.float64)
    cd = xp.interp(wind, anchors, xp.asarray(row["cd"], dtype=xp.float64))
    a = xp.clip((wind-20.)/(row["wind"][0]-20.), 0., 1.)
    return tuple((1.-a)*b+a*c for b, c in zip(base, (cd, row["ck"], row["ck"])))


def make_table(name, *, xp):
    """[Cd, Ch, Cq] at U10N=0..120 m/s in 0.1 m/s increments."""
    u = xp.arange(1201, dtype=xp.float64)*.1
    return xp.ascontiguousarray(xp.stack(coefficients(u, name, xp=xp), axis=1), dtype=xp.float32)
