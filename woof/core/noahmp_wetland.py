"""Noah-MP wetland option: re-implemented from published descriptions, not bitwise to upstream.

Source: Zhang, Z., Chen, F., Barlage, M., Bortolotti, L. E., Famiglietti, J.,
Li, Z., Ma, X. and Li, Y. (2022), "Cooling effects revealed by modeling of
wetlands and land-atmosphere interactions", Water Resources Research 58,
e2021WR030573, doi:10.1029/2021WR030573.  The equations were read from the
authors' preprint of that paper (ESS Open Archive, doi:10.1002/essoar.10507347,
version 3, section 2.3-2.5 and 3.2-3.3); equation numbers below are that
text's.  No Noah-MP v5 source file was opened, read or quoted.  Provenance per
routine: ``docs/noahmp/PROVENANCE-clean-wetland.md``.

The scheme has two parts.

1. **Saturated fraction (Eq. 7).**  ``FSAT = FSATMX * (SH2O_1 - SMCWLT_1) /
   (SMCREF_1 - SMCWLT_1)`` replaces the water-table FSAT of the TOPMODEL-type
   saturation-fraction runoffs (Eqs. 4-6).  ``FSATMX`` becomes a per-cell map
   (the paper uses the GIEMS-2 maximum inundation fraction; we use the GLWD v2
   wetland percentage, CC BY 4.0).  This lives in the WATER kernel
   (``k_water_wetland`` in ``noahmp_water_options.cuh``).
2. **Wetland store (Eqs. 8-13).**  The FSAT share of rain + snowmelt (Eq. 9)
   fills a bucket of capacity ``WCAP`` (Eq. 14, from a 90 m DEM); open water
   over FSAT evaporates at the Penman rate (Eq. 10); the excess spills as
   surface runoff (Eq. 12); latent heat is the area-weighted sum of the land
   tiles and the open water (Eq. 11) and sensible heat is the energy-balance
   residual (Sec. 2.5).  This module's :func:`advance_store` does it on the
   GPU in one elementwise launch.

Runoff pairing, read from the paper: the single-point and offline tests
replace the TOPMODEL saturated fraction (Eqs. 4-6, Niu and Yang 2005 /
Niu et al. 2011) and the coupled WRF runs put the wetland scheme "upon the
shallow groundwater scheme" of Miguez-Macho et al. (2007) (Sec. 3.3 and the
Discussion: "on top of the MMF groundwater scheme").  In Noah-MP v4 those are
``opt_run`` 1 and 2 (TOPMODEL with groundwater / equilibrium water table) and
5 (MMF).  ``opt_run`` 3, the Noah-MP default (Schaake 1996 infiltration
excess), has no water-table FSAT, but Eq. 7 needs none: FSAT comes from
first-layer wetness alone.  Following the paper's Eqs. 4-5 partition, the
FSAT share of the surface input fills the store (Eq. 9) and the (1 - FSAT)
land tile meets Schaake's scheme with the same input per unit area, its
runoff and infiltration area-weighted by (1 - FSAT), as Eq. 11 weights the
land tiles' fluxes.  That pairing is our derivation from the paper's
structure (the paper itself ran TOPMODEL and MMF); FSAT = 0 is the v4 code.
The other runoff options (4 BATS, 6 VIC, 7 XAJ, 8 dynamic VIC) carry their
own saturated-area or storage formulations that Eq. 7 would double count, so
the option refuses them by name.

Choices that are ours, not the paper's (each named in the provenance note):
Eq. 7 is clamped to ``[0, FSATMX]``; Eq. 12 is read as the store spilling
what exceeds ``WCAP`` after inflow and evaporation (the printed form compares
a rate with a depth); the Eq. 14 sum is taken as the cell-mean depth below
the cell-mean elevation (the printed sum has no 1/n and a sign that is never
positive, while the paper's capacities are tens to hundreds of mm); open
water exists only while the store holds water, the ground is unfrozen and
snow-free; canopy interception evaporation is left unweighted because the
canopy store is grid-mean; Penman uses Tetens saturation vapour pressure and
the bare-ground exchange coefficient as the aerodynamic conductance; the
store's cold start is the paper's own relative-saturation index times
``WCAP`` (the paper spins up one month instead).
"""
from __future__ import annotations

import numpy as np

#: The option's label wherever it is listed.
LABEL = "re-implemented from published descriptions, not bitwise to upstream"
CITATION = "Zhang et al. (2022), Water Resources Research 58, e2021WR030573"

#: Option values: 0 off (default; the v4 port unchanged), 1 Zhang (2022).
VALUES = (0, 1)
#: Runoffs the scheme pairs with: the saturation-fraction runoffs of the
#: paper (Secs. 2.3, 3.3) and the default Schaake runoff (module docstring).
RUNOFF_PAIRS = (1, 2, 3, 5)

#: Carried state (mm, grid-mean wetland storage) and the two static maps.
STATE = ("wetstorexy",)
STATIC = ("wetfsatmx", "wetcap")
#: Instantaneous diagnostics written to history.
DIAGNOSTICS = ("wetfsatxy", "wetlhxy")

# Noah-MP v4 constants (WRF public-domain source our port already carries):
# HVAP latent heat of vaporisation, CPAIR heat capacity of dry air.
HVAP = 2.5104e6
CPAIR = 1004.64
TFRZ = 273.16


def validate(*, wetland, sf_surface_physics, opt_run, soiltstep):
    """Refuse a combination the scheme is not defined for, naming why."""
    if wetland not in VALUES:
        raise ValueError(f"noahmp_wetland must be one of {VALUES}, got {wetland!r}")
    if not wetland:
        return
    if int(sf_surface_physics) != 4:
        raise ValueError("noahmp_wetland=1 is a Noah-MP option (sf_surface_physics=4)")
    if int(opt_run) not in RUNOFF_PAIRS:
        raise ValueError(
            "noahmp_wetland=1 partitions the surface input between the "
            "wetland store and the land runoff scheme (Zhang et al. 2022 "
            f"Eqs. 4-9), defined here for opt_run 1, 2, 3 or 5; opt_run={opt_run} "
            "has its own saturated-area or storage formulation that Eq. 7 "
            "would double count (set noahmp_wetland=0 with it)")
    if float(soiltstep) > 0.0:
        raise ValueError(
            "noahmp_wetland=1 advances the wetland store once per LSM step "
            "from that step's inflow; soiltstep>0 averages the inflow over "
            "several steps and would fill the store at the wrong time")


def validate_water(*, opt_run, soil_update_steps, calculate_soil):
    if int(opt_run) not in RUNOFF_PAIRS:
        raise ValueError(f"wetland WATER needs opt_run 1, 2, 3 or 5, got {opt_run}")
    if soil_update_steps != 1 or not calculate_soil:
        raise ValueError("wetland WATER needs the native single-step soil update")


_STORE_KERNEL = {}


def _store_kernel():
    import cupy as cp
    card = cp.cuda.runtime.getDevice()
    if card not in _STORE_KERNEL:
        _STORE_KERNEL[card] = cp.ElementwiseKernel(
            "float32 fsat, float32 inflow, float32 open_, float32 store, "
            "float32 wcap, float32 dt, float32 fsa, float32 fira, "
            "float32 tair, float32 qair, float32 psfc, float32 rhoair, "
            "float32 chb, float32 qseva, float32 fctr, float32 edir, "
            "float32 etran, float32 fsh, float32 runsrf",
            "float32 store_new, float32 runsrf_new, float32 edir_new, "
            "float32 etran_new, float32 fsh_new, float32 dqfx, float32 dlh, "
            "float32 wet_lh, float32 fsat_open",
            f"""
            const float hvap = {HVAP:.6e}f, cpair = {CPAIR:.6e}f;
            float fe = open_ != 0.0f ? fsat : 0.0f;
            // Eq. 10, Penman open-water evaporation (kg m-2 s-1 = mm/s).
            float tc = tair - 273.15f;
            float es = 610.78f * expf(17.27f * tc / (tc + 237.3f));
            float delta = 4098.171f * es / ((tc + 237.3f) * (tc + 237.3f));
            float ea = qair * psfc / (0.622f + 0.378f * qair);
            float vpd = fmaxf(es - ea, 0.0f);
            float gamma = cpair * psfc / (0.622f * hvap);
            float rn = fsa - fira;
            float epot = (delta * rn + rhoair * cpair * vpd * chb)
                         / (hvap * (delta + gamma));
            epot = fmaxf(epot, 0.0f);
            // Eqs. 9, 12, 13: fill, evaporate (water-limited), spill.
            float avail = store + inflow * dt;
            float eact = (fe > 0.0f) ? fminf(epot, avail / (fe * dt)) : 0.0f;
            float s1 = fmaxf(avail - fe * eact * dt, 0.0f);
            float spill = fmaxf(s1 - wcap, 0.0f);
            store_new = s1 - spill;
            runsrf_new = runsrf - inflow * dt + spill;
            // Eq. 11: land tiles weighted by (1 - FSAT) (soil evaporation and
            // transpiration; the kernel weighted the same soil extraction),
            // open water by FSAT.  Sensible heat is the residual (Sec. 2.5).
            float qseva_mm = qseva * 1000.0f;
            edir_new = edir - fe * qseva_mm;
            etran_new = etran * (1.0f - fe);
            wet_lh = fe * hvap * eact;
            dlh = wet_lh - fe * qseva_mm * hvap - fe * fctr;
            dqfx = fe * eact;
            fsh_new = fsh - dlh;
            fsat_open = fe;
            """,
            "gpuwm_noahmp_wetland_store")
    return _STORE_KERNEL[card]


def open_water(store, sneqv, isnow, frozen_ground, urban, ist):
    """1 where the FSAT area is open water this step, else 0 (float32)."""
    import cupy as cp
    ok = ((cp.asarray(store) > np.float32(0.0)) & (cp.asarray(sneqv) == np.float32(0.0))
          & (cp.asarray(isnow) == 0) & (cp.asarray(frozen_ground) == 0)
          & (cp.asarray(urban) == 0) & (cp.asarray(ist) == 1))
    return cp.ascontiguousarray(ok.astype(cp.float32))


def advance_store(*, fsat, inflow, open_, store, wcap, dt, fsa, fira, tair,
                  qair, psfc, rhoair, chb, qseva, fctr, edir, etran, fsh,
                  runsrf):
    """One wetland-store step for every land column, on the device."""
    import cupy as cp
    f32 = lambda a: cp.ascontiguousarray(cp.asarray(a, dtype=cp.float32))
    out = _store_kernel()(
        f32(fsat), f32(inflow), f32(open_), f32(store), f32(wcap), f32(dt),
        f32(fsa), f32(fira), f32(tair), f32(qair), f32(psfc), f32(rhoair),
        f32(chb), f32(qseva), f32(fctr), f32(edir), f32(etran), f32(fsh),
        f32(runsrf))
    names = ("store", "runsrf", "edir", "etran", "fsh", "dqfx", "dlh",
             "wet_lh", "fsat_open")
    return dict(zip(names, out))


#: Ceiling of the cold-start store, mm.  Zhang et al. (2022) report summer
#: storage of "more than 200 mm" only in the wettest part of the Prairie
#: Pothole Region (Sec. 3.2, Fig. 6a).  Eq. 14 capacities on a 3 km cell reach
#: metres where the cell has relief; as a starting depth that would keep the
#: FSAT area open water for months.  The store may still fill to WCAP.
COLD_START_MAX_MM = 200.0


def cold_start_store(wcap, fsatmx, sh2o1, smcwlt1, smcref1):
    """Initial store: min(WCAP, 200 mm) times the Eq. 7 relative saturation.

    The paper spins the store up for one month (Sec. 3.3).  A forecast has no
    month, so the store starts at the same relative wetness the scheme reads
    from the first soil layer; cells with no wetland (FSATMX = 0) start empty.
    """
    xp = _xp(wcap)
    span = smcref1 - smcwlt1
    rel = xp.where(span > 0, (sh2o1 - smcwlt1) / xp.where(span > 0, span, 1), 0)
    rel = xp.clip(rel, 0, 1)
    depth = xp.minimum(wcap, COLD_START_MAX_MM)
    return xp.where(fsatmx > 0, depth * rel, 0).astype(np.float32)


def initialize_store(fields, params=None):
    """Cold start: mark every store "not started" (-1).

    The first land step replaces the marker with :func:`cold_start_store`,
    where the soil parameters it needs are already gathered.  A restart
    carries the real store and never passes through here.
    """
    fields["wetstorexy"].fill(np.float32(-1.0))


def _xp(a):
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp
    except ImportError:
        pass
    return np


# ---- NumPy reference of the same equations, float64, for tests ------------

def reference_fsat(fsatmx, sh2o1, smcwlt1, smcref1):
    span = smcref1 - smcwlt1
    if not span > 0:
        return 0.0
    return fsatmx * min(1.0, max(0.0, (sh2o1 - smcwlt1) / span))


def reference_store_step(*, fsat, inflow, open_, store, wcap, dt, fsa, fira,
                         tair, qair, psfc, rhoair, chb, qseva, fctr, edir,
                         etran, fsh, runsrf):
    fe = fsat if open_ else 0.0
    tc = tair - 273.15
    es = 610.78 * np.exp(17.27 * tc / (tc + 237.3))
    delta = 4098.171 * es / (tc + 237.3) ** 2
    ea = qair * psfc / (0.622 + 0.378 * qair)
    vpd = max(es - ea, 0.0)
    gamma = CPAIR * psfc / (0.622 * HVAP)
    epot = max((delta * (fsa - fira) + rhoair * CPAIR * vpd * chb)
               / (HVAP * (delta + gamma)), 0.0)
    avail = store + inflow * dt
    eact = min(epot, avail / (fe * dt)) if fe > 0 else 0.0
    s1 = max(avail - fe * eact * dt, 0.0)
    spill = max(s1 - wcap, 0.0)
    wet_lh = fe * HVAP * eact
    dlh = wet_lh - fe * qseva * 1000.0 * HVAP - fe * fctr
    return {"store": s1 - spill, "runsrf": runsrf - inflow * dt + spill,
            "edir": edir - fe * qseva * 1000.0, "etran": etran * (1 - fe),
            "fsh": fsh - dlh, "dqfx": fe * eact, "dlh": dlh,
            "wet_lh": wet_lh, "epot": epot, "eact": eact, "spill": spill}


__all__ = ["CITATION", "DIAGNOSTICS", "LABEL", "RUNOFF_PAIRS", "STATE",
           "STATIC", "VALUES", "advance_store", "cold_start_store",
           "initialize_store", "open_water", "reference_fsat", "reference_store_step", "validate",
           "validate_water"]
