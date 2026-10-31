# MYJ wrapper capture and replay, WRF v4.6.1

Grades woof's MYJ surface layer (`sf_sfclay_physics = 2`) and MYJ PBL
(`bl_pbl_physics = 2`) word for word on a real forecast, with WRF's own
inputs.

## What it does

1. `build_cap.sh WRF_BUILD OUT_DIR` relinks a built WRF 4.6.1 tree's
   `wrf.exe` with capture blocks around `CALL MYJSFC` (module_surface_driver.F)
   and `CALL myjpbl` (module_pbl_driver.F). The scheme objects are the tree's
   own and are not touched. `patch_drivers.py` writes the blocks;
   `module_cap_io.f90` writes them out.
2. Run that `wrf.exe` on the case with `WRF_CAP_STEPS=1,2,3,50,100` and
   `WRF_CAP_DIR=<dir>`. Every tile writes, per requested step, each wrapper
   argument just before the call (`*_in_*`) and every INOUT/OUT argument just
   after it (`*_out_*`). The files are flushed when wrf.exe exits.
3. `myj_replay.py CAPDIR STEP... --json OUT` assembles the tiles, binds the
   `in` words to `launch_myj_sfclay` and `myj_pbl_step` exactly as
   `woof/core/physics.py` binds them, and counts differing float32 words
   against the `out` words. Run it with `GPUWM_WRF_EXACT=1` and, for a referee
   built on glibc 2.41 or later, `GPUWM_WRF_GLIBC=<version>`.
4. `driver_check.py CAPDIR STEPS OUT.json` checks the two driver seams the
   replay found: Noah's `myj` CHKLOWQ arm and MYJSFC's Z0BASE binding.
5. `sfc_case.py` turns a captured step into a `run_myjsfc` case
   (tools/myjsfc_wrf461_oracle/run_myjsfc.F90) so WRF's MYJSFC can be rebuilt
   standalone on the referee host and its MYJSFCINIT tables read back.

## Result on the WRF-JAX paper case (2026-10-29)

500 x 500 x 44 Mercator, 3 km, 18 s, Thompson / RRTMG / MYJ / Eta / Noah /
Tiedtke, pinned WRF 4.6.1 (gfortran 15.2, -O2, glibc 2.43), 3 x 4 ranks,
steps 1, 2, 3, 50 and 100:

- MYJPBL: 0 words different at every step.
- MYJSFC: 0 words different at every step once the PSIM table takes the
  referee's atanf generation (`GPUWM_WRF_GLIBC=2.43`). With the glibc 2.39
  generation 21,063 to 45,811 words differed per step, all traced to 816 of
  10,001 PSIM table words.
- The standalone MYJSFC at -O0 and at WRF's -O2 flags reproduced WRF's
  captured outputs on all 249,001 columns (0 words different).
