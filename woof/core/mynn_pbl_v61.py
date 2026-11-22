"""Upstream MYNN v6.1 bug fixes on WOOF-ArWen's WRF 4.6.1 MYNN (``mynn_fixes_v61``).

The WRF 4.6.1 MYNN route is the frozen WRF v4.6.1 unit ``kernels/mynn_pbl.cu``
and the WRF v4.6.1 ``SFCLAY_mynn`` first-step block.  NCAR's official MYNN
homes (NCAR/MYNN-EDMF v6.1.1 2b2da61b05a8, NCAR/MYNN-SFC v6.1.0 a038a9ea35a1,
both Apache-2.0, J. Olson, NOAA GSL) have since fixed defects that the WRF
4.6.1 lineage still carries.  ``RunConfig.mynn_fixes_v61 = True`` runs the
ones that apply to that lineage.  It is correct-mode finding MY1
(woof/correct_mode.py), so it is the default; ``False`` (physics.mode =
'exact', a strict WRF process, or the key set false) never reads this
module and compiles and runs the frozen unit's bytes, which every WRF 4.6.1
parity proof names.  The key is read only by the MYNN PBL at the wrf_461
generation (:func:`pbl_fixes_active`); the surface-layer part S1 only beside
it, on the wrf_461 surface layer (:func:`sfclay_fixes_active`).

Labelled: upstream MYNN v6.1 fixes, re-applied to the WRF 4.6.1 lineage;
not bitwise to WRF and not bitwise to MYNN v6.1 (the rest of the scheme is
WRF 4.6.1's).

Applied, each a marked edit ``MYNN-V61 <id>`` of the frozen unit text
(``MYNN_V61_EDITS``; every anchor must occur exactly once, so an edit of
``mynn_pbl.cu`` that moves one refuses to build the variant instead of
silently dropping the fix):

* **A1 dew and frost** (MYNN-EDMF f3c9d9e; v6.1.1 module_bl_mynnedmf.F90:4543).
  WRF 4.6.1 limits a downward surface vapour flux with
  ``max(qvflux, min(0.9*sqv - 1e-8, 0)/dtz)``; for any ``sqv > 1.1e-8`` the
  MIN is 0, so dew and frost are deleted, not limited.  v6.1.1 limits it so
  the step cannot take ``sqv`` below 1e-8:
  ``max(qvflux, min(1e-8 - 0.9*sqv, 0)/(dtz*rhosfc*rhoinv))``.
* **A2 surface density** (MYNN-EDMF eb5029c; v6.1.1 :4051).
  ``psfc/(r_d*(tk + p608*qv))`` adds a mixing ratio to a temperature;
  v6.1.1 has the virtual-temperature factor ``psfc/(r_d*tk*(1 + p608*qv))``
  (about 0.6 % in the surface density).  A1 divides by this ``rhosfc``.
* **A5 surface theta** (MYNN-EDMF 27da0f0; v6.1.1 :689-700).
  ``th_sfc = ts/ex1(kts)`` takes the ground temperature to theta with the
  lowest model level's Exner function; v6.1.1 uses the ground's
  ``exnerg = (ps/p1000mb)**rcp``, which WRF 4.6.1 computes and never uses.
  The surface-flux kernel gains ``psfc`` as one extra argument.
* **A6 PBL-height search start** (MYNN-EDMF 27da0f0; v6.1.1 get_pblh
  :6060-6083).  WRF 4.6.1 finds the minimum theta-v level ``kthv`` and then
  searches for ``theta-v >= min + delta`` from ``kts+1``, so a superadiabatic
  layer below the minimum ends the search there; v6.1.1 starts it at
  ``max(kthv+1, kts+1)``.
* **S1 surface-layer cold start under cycling** (MYNN-SFC f0c1839; v6.1.0
  module_sf_mynnsfc_driver.F90:430).  The cold-start reset runs only when
  the start is neither a restart nor a cycled start.  WRF 4.6.1's
  ``SFCLAY_mynn`` (module_sf_mynn.F:330-337) gates its reset of UST, MOL and
  QSFC on ``itimestep == 1`` alone; a restart never sees step 1 again, so
  only the cycled half applies: a ``cycling`` start keeps the input's
  surface-layer state.  ``qstar`` is still zeroed on step 1, as v6.1.0 does.

Not applicable to the WRF 4.6.1 lineage, so not carried:

* MYNN-SFC fca16e4 (first-step land blend ``0.99*tskin + 0.01*t_1``
  removed): WRF 4.6.1 ``SFCLAY1D_mynn`` never had the blend; land QSFC is the
  LSM's, or saturation at TSK where it is not positive (module_sf_mynn.F:
  522-534).
* MYNN-SFC winds relative to ocean currents: no current input on this route.
* MYNN-EDMF stale cloud water in the theta tendency (sqc/sqi vs sqc2/sqi2):
  WRF 4.6.1 already uses the post-mixing sqc2/sqi2.
* MYNN-EDMF plume sigma exponent sign + retuned plume excess (1935d2b): a
  valid upstream fix, not in this port's scope (it is coupled to a retune of
  the excess constants); PORT-PLAN.md item A3.
* MYNN-EDMF BouLac downward length sign (399f50f, PR 94; PORT-PLAN.md A4):
  carried by correct mode, not by this unit, as finding ML1
  (``RunConfig.mynn_boulac_down_sign``, the ``MYNN_BOULAC_DOWN_SIGN`` define
  of the two mixing-length kernels), so it is on by default and reaches
  gsd_41 as well; those kernels are not in :data:`MYNN_V61_FUNCS`.
"""

from __future__ import annotations

#: The kernels of ``mynn_pbl.cu`` the fixes edit.  Every other kernel of a
#: ``mynn_fixes_v61`` run is the frozen unit's own.
MYNN_V61_FUNCS = frozenset({
    "mynn_pblh_scale_columns",       # A6
    "mynn_tendencies_columns",       # A1, A2
    "mynn_driver_surface_columns",   # A5 (+ psfc argument)
})

MYNN_V61_NOTICE = (
    "// mynn_pbl_v61: kernels/mynn_pbl.cu with the marked MYNN-V61 edits\n"
    "// (woof/core/mynn_pbl_v61.py).  Changes from NCAR/MYNN-EDMF v6.1.1\n"
    "// 2b2da61b05a8 (Apache-2.0, Copyright NOAA GSL / J. Olson), re-applied\n"
    "// to the WRF 4.6.1 lineage; upstream MYNN v6.1 fixes, not bitwise to\n"
    "// WRF.  Licensed under the Apache License, Version 2.0.\n"
)

#: (id, frozen text, replacement).  Each frozen text occurs exactly once in
#: ``kernels/mynn_pbl.cu``; each replacement carries ``MYNN-V61 <id>``.
MYNN_V61_EDITS: tuple[tuple[str, str, str], ...] = (
    ("A6a",
     "    real minthv = 9.0e9f;\n"
     "    int k = 1;  // zero-based WRF kts+1.\n"
     "    while (zw[zbase + k] <= 200.0f && k < nz) {\n"
     "        minthv = fminf(minthv, thetav[base + k]);\n"
     "        ++k;\n"
     "    }\n",
     "    real minthv = 9.0e9f;\n"
     "    int k = 1;  // zero-based WRF kts+1.\n"
     "    int kthv = 0;  // MYNN-V61 A6: one-based kthv = 1 (kts).\n"
     "    while (zw[zbase + k] <= 200.0f && k < nz) {\n"
     "        if (minthv > thetav[base + k]) {  // MYNN-V61 A6\n"
     "            minthv = thetav[base + k];\n"
     "            kthv = k;\n"
     "        }\n"
     "        ++k;\n"
     "    }\n"),
    ("A6b",
     "    real zi = 0.0f;\n"
     "    for (k = 1; k < nz - 1; ++k) {\n"
     "        real current = thetav[base + k];",
     "    real zi = 0.0f;\n"
     "    // MYNN-V61 A6: do k = max(kthv+1, kts+1), kte-1 (v6.1.1 :6083).\n"
     "    for (k = max(kthv + 1, 1); k < nz - 1; ++k) {\n"
     "        real current = thetav[base + k];"),
    ("A2",
     "    real rhosfc = MYNN_DIV(\n"
     "        psfc,\n"
     "        MYNN_MUL(RD, MYNN_ADD(tk[base], MYNN_MUL(p608, qv[base]))));",
     "    // MYNN-V61 A2: rhosfc = psfc/(r_d*tk(kts)*(one+p608*qv(kts))).\n"
     "    real rhosfc = MYNN_DIV(\n"
     "        psfc,\n"
     "        MYNN_MUL(MYNN_MUL(RD, tk[base]),\n"
     "                 MYNN_ADD(1.0f, MYNN_MUL(p608, qv[base]))));"),
    ("A1",
     "        qvflux = mynn_max2(\n"
     "            qvflux,\n"
     "            MYNN_DIV(\n"
     "                mynn_min2(MYNN_SUB(MYNN_MUL(0.9f, sqv[base]), 1.0e-8f), 0.0f),\n"
     "                dtz[base]));",
     "        // MYNN-V61 A1: max(qvflux, min(1e-8 - p9*sqv(kts), zero)\n"
     "        //              /(dtz(kts)*rhosfc*rhoinv(kts))) (v6.1.1 :4543).\n"
     "        qvflux = mynn_max2(\n"
     "            qvflux,\n"
     "            MYNN_DIV(\n"
     "                mynn_min2(MYNN_SUB(1.0e-8f, MYNN_MUL(0.9f, sqv[base])), 0.0f),\n"
     "                MYNN_MUL(MYNN_MUL(dtz[base], rhosfc), rhoinv[base])));"),
    ("A5a",
     "    real* __restrict__ pmz, real* __restrict__ phh,\n"
     "    int nz, int ncol\n"
     "#if defined(MYNN_GSD41)\n"
     "    , const real* __restrict__ rmol_in",
     "    real* __restrict__ pmz, real* __restrict__ phh,\n"
     "    int nz, int ncol,\n"
     "    const real* __restrict__ psfc_v61  // MYNN-V61 A5: ground pressure\n"
     "#if defined(MYNN_GSD41)\n"
     "    , const real* __restrict__ rmol_in"),
    ("A5b",
     "    real surface_theta = MYNN_DIV(ts[column], exner0);",
     "    // MYNN-V61 A5: th_sfc = ts/exnerg, exnerg = (ps/p1000mb)**rcp.\n"
     "    real surface_theta = MYNN_DIV(ts[column], mynn_powf(\n"
     "        MYNN_DIV(psfc_v61[column], 100000.0f), RD / CP));"),
)


class MynnV61SourceError(RuntimeError):
    """The frozen unit no longer holds an anchor a v6.1 fix edits."""


def mynn_v61_source(frozen: str) -> str:
    """The ``mynn_pbl_v61`` unit text from the frozen ``mynn_pbl.cu`` text."""

    text = frozen
    for ident, old, new in MYNN_V61_EDITS:
        found = text.count(old)
        if found != 1:
            raise MynnV61SourceError(
                f"MYNN v6.1 fix {ident}: kernels/mynn_pbl.cu holds its anchor "
                f"{found}x (need exactly 1); the variant would run without "
                "the fix while claiming it.")
        text = text.replace(old, new, 1)
    return MYNN_V61_NOTICE + text


def pbl_fixes_active(cfg) -> bool:
    """Whether a run's MYNN PBL runs the v6.1 fixes: the key is on and the
    PBL is MYNN at the wrf_461 generation (correct_mode.KEY_SCOPE and
    KEY_GENERATIONS say the same, for the echo and the receipt)."""

    return (getattr(cfg, "mynn_fixes_v61", False) is True
            and int(getattr(cfg, "bl_pbl_physics", 0) or 0) == 5
            and getattr(cfg, "bl_mynn_version", "wrf_461") == "wrf_461")


def sfclay_fixes_active(cfg) -> bool:
    """Whether the WRF 4.6.1 MYNN surface layer runs fix S1: beside an
    active PBL fix set, on the wrf_461 surface-layer variant."""

    return (pbl_fixes_active(cfg)
            and int(getattr(cfg, "sf_sfclay_physics", 0) or 0) == 5
            and getattr(cfg, "mynn_sfclay_variant", "wrf_461") == "wrf_461")


def mynn_v61_kernel_name(func: str, fixes_v61: bool) -> str:
    """The module a MYNN kernel loads from: the variant only where it edits."""

    return "mynn_pbl_v61" if fixes_v61 and func in MYNN_V61_FUNCS else "mynn_pbl"


def mynn_sfclay_cold_start(itimestep: int, *, cycling: bool,
                           fixes_v61: bool) -> bool:
    """Whether ``SFCLAY_mynn``'s first-step reset of UST/MOL/QSFC runs.

    WRF 4.6.1 (module_sf_mynn.F:330): ``itimestep == 1``.  With the v6.1
    fix S1 (MYNN-SFC f0c1839): ``itimestep == 1 .and. .not.(restart .or.
    cycling)``; a restart is never at step 1 on this route, so it reduces to
    ``itimestep == 1 and not cycling``.
    """

    if int(itimestep) != 1:
        return False
    return not (fixes_v61 and cycling)


__all__ = [
    "MYNN_V61_EDITS",
    "MYNN_V61_FUNCS",
    "MYNN_V61_NOTICE",
    "MynnV61SourceError",
    "mynn_sfclay_cold_start",
    "mynn_v61_kernel_name",
    "mynn_v61_source",
    "pbl_fixes_active",
    "sfclay_fixes_active",
]
