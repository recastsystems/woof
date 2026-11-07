"""The field order the Pleim-Xiu LSM oracle's case and output files carry.

ONE list, read by the case generator, by the parity tests and (as the
literal read/write order in run_pxlsm.F90) by the Fortran driver.  The
Fortran side cannot import this module, so run_pxlsm.F90 spells the same
order out by hand and ``tests/test_pxlsm_wrf461_parity.py`` checks the two
agree by reading back a file the driver wrote: every array is stored in the
driver's memory order (i fastest, then the level/category index, then j),
which is numpy C order over (ny, levels, nx) and (ny, nx).

Case file (stream, little endian):

  int32[16]  NX NY NLCAT NSCAT NSTEPS PXLSM_SMOIS_INIT PXLSM_SOIL_NUDGE
             PXLSM_MODIS_VEG ANAL_INTERVAL ITIMESTEP0 0 0 0 0 0 0
  float32[8] DT DTBL CURR_SECS0 DZS1 DZS2 ZS1 ZS2 0
  then every array of IN_2D, INOUT_2D, OUT_2D (float32 (NY,NX)), ISLTYP
  (int32 (NY,NX)), ATM_3D (float32 (NY,2,NX)), SOIL_3D (float32
  (NY,2,NX)), LANDUSEF (float32 (NY,NLCAT,NX)), SOILCTOP and SOILCBOT
  (float32 (NY,NSCAT,NX)), in that order.

Output file: for each step 1..NSTEPS, every array of INOUT_2D, OUT_2D,
ISLTYP and SOIL_3D in that order, as the driver holds them after the call.
"""

#: Inputs pxlsm reads and never writes.
IN_2D = (
    "psfc", "gsw", "glw", "rainbl", "emiss", "albbck", "shdmin", "shdmax",
    "pblh", "rmol", "snowncv", "ust", "mavail", "sst",
    "t2_ndg_old", "t2_ndg_new", "q2_ndg_old", "q2_ndg_new",
    "sn_ndg_old", "sn_ndg_new",
)

#: INTENT(INOUT) arrays (module_sf_pxlsm.F:276-309), including ISLTYP's
#: REAL siblings.  Their entry value matters wherever pxlsm leaves them.
INOUT_2D = (
    "ra", "rs", "lai", "znt", "qsfc", "capg", "canwat", "qfx", "hfx", "lh",
    "psih", "vegfra", "vegf_px", "snow", "snoalb", "snowh", "snowc",
    "albedo", "xland", "xice", "imperv", "canfra",
)

#: INTENT(OUT) arrays (:286, :303, :314-315).  Several are written only on
#: some columns (GRDFLX is not written over water by SURFPX, T2OBS/Q2OBS
#: only under soil nudging), so the generator seeds each with a sentinel
#: and the port must leave an unwritten word as it found it.
OUT_2D = (
    "grdflx", "tsk", "ta2", "qa2", "t2obs", "q2obs",
    "lai_px", "wwlt_px", "wfc_px", "wsat_px", "clay_px", "csand_px",
    "fmsand_px",
)

#: The 3-D atmosphere pxlsm reads, two levels (it reads k=1 and k=2 only).
ATM_3D = ("u3d", "v3d", "dz8w", "qv3d", "t3d", "th3d", "rho")

#: The two soil layers, INOUT.
SOIL_3D = ("smois", "tslb")

#: Sentinel the OUT_2D arrays start at.  A finite, exactly representable
#: number no physical output takes, so "left unwritten" is visible.
OUT_SENTINEL = -7777.0
