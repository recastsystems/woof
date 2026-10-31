# Noah-MP column oracle: WOOF's device path against WRF v4.6.1's driver

This harness runs the unmodified WRF v4.6.1 Noah-MP driver (`NOAHMP_INIT`,
then `noahmplsm` once per step) and WOOF's shipped forecast path
(`noahmp_cold_start`, then `noahmp_lsm_step` once per step) on the same
inputs, and compares every value WOOF writes, bit for bit.

| file | what it does |
|---|---|
| `gen_columns.py` | Writes the input file: 128 columns over the land model's regimes, and forcing that changes every step. Seeded, so the file is reproducible. Two configurations: `summer60` (July, dt 60 s, 36 steps, dx 3 km) and `winter300` (January, dt 300 s, 24 steps, dx 12 km). |
| `run_column.F90` | WRF's side. Calls `noahmp_init` and `noahmplsm` from the pinned sources with WRF's own actual arguments (keyword form, including the six optional MP_* precipitation amounts), on an NX by 1 grid. |
| `build_column.sh` | Checks the three pinned source hashes, compiles them as they ship, links `run_column`, stages WRF's own tables. `noopt` is the fixture build (-O0); `wrf` is WRF's stock gfortran FCOPTIM, kept as a recorded comparison. |
| `run_woof_column.py` | WOOF's side, on a GPU. Refuses to run if any host path is selected. `GPUWM_WRF_EXACT=1` selects the strict build. |
| `compare_column.py` | Bitwise comparison per field and snapshot: differing values, max ULP, max absolute difference, first miss. |

## Option identity

WRF Registry defaults for `sf_surface_physics = 4`: dveg 4, opt_crs 1,
opt_btr 1, opt_run 3, opt_sfc 1, opt_frz 1, opt_inf 1, opt_rad 3, opt_alb 2,
opt_snf 1, opt_tbot 2, opt_stc 1, opt_gla 1, opt_rsf 1, opt_soil 1,
opt_pedo 1, opt_crop 0, opt_irr 0, opt_irrm 0, opt_infdv 1, opt_tdrn 0,
soiltstep 0, four soil layers (0.1, 0.3, 0.6, 1.0 m), sf_urban_physics 0,
xice_threshold 0.5, land use MODIFIED_IGBP_MODIS_NOAH.

## What is compared

73 fields, 94 values per column, at the cold-started state and after every
step: the generic LSM fields (TSK, HFX, QFX, LH, GRDFLX, SMSTAV, SMSTOT,
SFCRUNOFF, UDRUNOFF, ALBEDO, SNOWC, SNOW, SNOWH, CANWAT, ACSNOM, ACSNOW,
EMISS, QSFC, Z0, ZNT, LAI), the 17 carried Noah-MP scalars, the 26 published
Noah-MP diagnostics, ISNOWXY and PGSXY, the soil column (TSLB, SMOIS, SH2O)
and the snow stack (TSNOXY, SNICEXY, SNLIQXY, ZSNSOXY).

One documented exclusion: on glacier columns WRF writes EFLXBXY, SOILENERGY
and SNOWENERGY from EFLXB and HCPCT, two locals that only NOAHMP_SFLX
assigns. The glacier arm never sets them, so WRF publishes the preceding
land column's values (a read of an undefined variable). WOOF publishes the
glacier column's own values. The comparator counts these values separately.

## How to rerun (box W1, CPU for WRF, one GPU for WOOF)

```
T=/work/pverify/<lane>/src; source /work/pverify/base/env.sh
C=$T/tools/noahmp_wrf461_oracle/column
W=/work/pverify/wrf-build/src/WRF-4.6.1
sh $C/build_column.sh $W build-noopt noopt
sh $C/build_column.sh $W build-wrf wrf          # optional control
for cfg in summer60 winter300; do
  python $C/gen_columns.py in-$cfg.bin --config $cfg
  (cd build-noopt && ./run_column ../in-$cfg.bin ../wrf-$cfg.bin)
done
# GPU only through the mutex:
card-lease <lane> --cards 1 --est 900 --wait 1200 bash -c '
  source /work/pverify/base/env.sh; cd <dir>
  for cfg in summer60 winter300; do
    GPUWM_WRF_EXACT=1 python $C/run_woof_column.py in-$cfg.bin woof-strict-$cfg.bin
    python $C/run_woof_column.py in-$cfg.bin woof-default-$cfg.bin
  done'
python $C/compare_column.py wrf-summer60.bin woof-strict-summer60.bin in-summer60.bin
```

The committed fixtures (`tests/data/oracles/noahmp/column`) and
`tests/test_noahmp_wrf461_column_oracle.py` hold the result in the GPU shard.
