# New Tiedtke column oracle (cu_physics = 16)

WRF v4.6.1's own `cu_ntiedtke_driver` and WOOF's `NewTiedtke` adapter, run
over the same columns, compared word for word.

The stage oracle in `tools/ntiedtke_wrf461_oracle` grades each of the 21
kernels on 18 analytic soundings.  This harness grades the whole scheme, the
way a forecast calls it, on real columns: full model domains taken from
wrfout files (a WRF v4.4 CONUS 12 km benchmark state, a WOOF night state on
the same grid, and two WOOF convection-permitting states at 1 km and
2.25 km), plus stratified subsets of them.

## Files

| file | what it does |
| --- | --- |
| `make_columns.py` | wrfout -> a case: every driver-boundary field as raw float32, WRF memory order, plus the six forcing lanes WRF folds |
| `make_subset.py` | a stratified, a stable-only, or an explicit column subset of a case |
| `run_nt_oracle.F90` | the WRF side: the cumulus driver's fold, then one `cu_ntiedtke_driver` call over the whole patch |
| `build.sh` | builds three WRF executables: `-O0` reference, WRF's stock `-O2` flags, and the `-O0` reference with cutypen's `zcbase` declared real |
| `woof_side.py` | the WOOF side: the production adapter on the GPU, with replication to reach the fused and multi-chunk walks |
| `compare.py` | bitwise comparison, ULP distances, regime tags |
| `summarize.py` | the matrix table and the attribution check |
| `run_matrix.sh` | every run: `cases`, `wrf`, `woof` (GPU), `compare` |

## The WRF bug the third build exists for

`cu_ntiedtke.F90:1300` declares

```fortran
integer,dimension(klon):: zcbase,itoppacel
```

so `zcbase(jl) = paph(jl,ik) + zdp`, the cloud-base pressure in Pa, is
truncated toward zero before cutypen picks the nearer half level as the
cloud base.  The name follows the scheme's real-variable convention, its
neighbour `itoppacel` is a genuine level index, and nothing in the routine
wants a whole-Pa pressure.  WOOF keeps `zcbase` real and does not copy the
truncation.  The third build moves only that declaration to real, so the
comparison can say which columns the bug decides (`summarize.py`'s
attribution check: every column where WOOF differs from the pinned build
must be one where the pinned and corrected builds differ).

## Rerun

On a box with gfortran, a WRF v4.6.1 source tree and the GPU mutex:

```
export T=<tree> OUT=<work dir> WRF_SRC=<WRF-4.6.1>
export NCAR00Z=<wrfout> NCAR13Z=<wrfout> TOR20Z=<wrfout> BOS20Z=<wrfout>
export DIFF18_COLS=<18 comma-separated column indices of the mix subset>
bash $T/tools/ntiedtke_column_oracle/run_matrix.sh cases
bash $T/tools/ntiedtke_column_oracle/run_matrix.sh wrf
card-lease <lane> --cards 1 --est 1800 --wait 3600 \
    bash $T/tools/ntiedtke_column_oracle/run_matrix.sh woof
bash $T/tools/ntiedtke_column_oracle/run_matrix.sh compare
```

`woof` runs every case twice: default arithmetic, then `GPUWM_WRF_EXACT=1`.
