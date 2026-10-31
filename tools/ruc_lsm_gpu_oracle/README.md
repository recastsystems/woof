# RUC LSM GPU column oracle (WRF v4.6.1)

Proves, word for word, that WOOF's production RUC land-surface step on the GPU
(`woof.core.ruc_runtime.ruc_lsm_step`, the fused CUDA kernels) computes what
WRF v4.6.1's Fortran computes for the same columns.

## What is compared

Both sides read one `inputs.bin`:

* WOOF: `run_woof.py` builds a one-row physics driver from `columns.py`'s 102
  columns, lets WOOF's own cold start run, sets every LSMRUC argument and every
  surface-driver seam field to the case's words, writes `inputs.bin`, then runs
  `ruc_lsm_step` for every step and saves every compared field.
* WRF: `layout.py` generates `run_columns.F90`, a transcription of
  `module_surface_driver.F:3438-3593` (the `CASE (RUCLSMSCHEME)` arm: sea-ice
  albedo override, the fractional sea-ice deblend and reblend, the CQS/CHS
  rebuild) around the unmodified `LSMRUC`, `RUCLSMINIT` and `SFCDIAGS_RUCLSM`.
  `build_wrf.sh` compiles the pristine WRF source three ways (`defined`: WRF's
  GNU flags plus `-finit-local-zero`; `stock`: WRF's GNU flags; `o0`) and runs
  each.

`compare.py` grades RUCLSMINIT's four outputs and, after every step, 49 column
fields plus the five soil profiles, by ULP on the ordered float32 line.
Free-running (each side carries its own state) and replay (each WOOF step
starts from WRF's state after the previous step) are both reported.

Two WRF defects are graded, not copied:

* SFCEVP: WRF adds `qfx*dt` twice per land step (`module_sf_ruclsm.F:1095`
  and `:1116`).  WOOF counts it once.  The driver writes WRF's single count
  (`sfcevp_once_acc` free running, `sfcevp_once_step` replayed) and WOOF's
  SFCEVP is graded against it.
* Uninitialised locals: SFCTMP's `ilnb` is read (`:4410`) under a pack
  thinner than 10 mm of water before anything sets it.  The `defined` arm
  zeroes locals so the read has one value (WOOF's, the single-layer branch);
  the `stock` arm shows what the uninitialised read does (only TSNAV moves).

## Rerun (box with a card behind a card lease, CPU for Fortran)

    export T=<tree> WRF_SRC=<pristine WRF v4.6.1>
    source /work/pverify/base/env.sh
    bash $T/tools/ruc_lsm_gpu_oracle/run_all.sh <runs-dir>      # 6 configs x strict/default
    bash $T/tools/ruc_lsm_gpu_oracle/run_case.sh <dir> strict --nzs 9   # one case

`run_case.sh` runs WOOF under the mutex (strict = `GPUWM_WRF_EXACT=1`), the
three Fortran arms, a replay, and both comparisons; `compare.txt` holds the
verdict lines. The shell exits nonzero on either comparison failure. Its
free run grades the defined-local and O0 references; stock output is retained
for a separate diagnostic comparison because its uninitialized TSNAV can
differ. Replay copies WRF's TSK_SAVE without substituting TSK. Set
`RUC_ORACLE_LANE` to the assigned queue tag when running a private lane.

`libm_sweep.py OUTDIR` sweeps RUC's float32 libm words (device and host)
against the C library on the box; `libm_fma_probe.py` is the diagnostic for
the one expf argument the shared `gfk_exp` misses.  `pack_fixture.py` packs a
case's WRF side into `tests/data/oracles/ruc_lsm/`, which
`tests/test_ruc_gpu_column_oracle.py` grades on any card without Fortran.

## The operational branch (`--lineage fork`)

`run_woof.py --lineage fork` runs the RUC forms the HRRR door selects
(`ruc_soilprop`, `ruc_irrigation`, `ruc_snow` = `wrf_45`,
`ruc_qvg_cold_start = air`, `ruc_2m_diagnostic = log_profile`) and writes
the driver for the operational RAP/HRRR branch's surface-driver arm
(NOAA-EMC/HRRR v4.1.21 WRFV3.9 `module_surface_driver.F:3279-3420`): the same
seam, and the branch's `SFCDIAGS_RUCLSM`, which also returns
`T2_alt/TH2_alt/Q2_alt` and reads `DZ8W`.  Point `WRF_SRC` at the branch tree
(tools/fork_forms_oracle/fetch_fork.py) and `RUC_TABLES` at the WRF v4.6.1
`run/` tables WOOF loads, so the arm grades code forms and not a table
difference.  The branch also accumulates SNOWFALLAC in metres under a
Registry label of mm; WOOF keeps mm, and that one field is graded by the
unit rule in tests/test_noaa_wrf39_fork_forms_oracle.py.
