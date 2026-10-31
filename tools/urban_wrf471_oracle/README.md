# Urban column oracles, WRF v4.7.1

One harness for all three urban options (single-layer UCM, BEP, BEP+BEM) and
the Noah / Noah-MP hand-over they depend on.  It compiles the pinned WRF
sources byte-unmodified at `-O0` and runs Fortran drivers that call WRF's own
routines and write every input and output they touch.

```
bash tools/urban_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
```

## The WRF tree

`WRF_SOURCE_ROOT` = a clone of wrf-model/WRF tag **v4.7.1** (commit
`f52c197ed39d12e087d02c50f412d90d418f6186`) with `phys/noahmp` at NCAR/noahmp
`e5c0859874407859936739e8be8741f9aed369ee` and `phys/physics_mmm` at
NCAR/MMM-physics tag `20240626-MPASv8.2` (WRF's `arch/Externals.cfg`).  On
a development machine and a development machine it is `~/work/urban-wrf471/WRF`, read only.

The tree is not checked by commit: `build.sh` checks every file it compiles or
copies against `SOURCES.sha256` (and any lane's `SOURCES-<lane>.sha256`) and
stops on the first mismatch, before compiling anything.

## What it builds

Core, in order: `module_model_constants`, `module_wrf_error`,
`module_sf_urban`, `module_bep_bem_helper`, `module_sf_bep`, `module_sf_bem`,
`module_sf_bep_bem`, `module_sf_noahlsm`, `module_sf_noahlsm_glacial_only`,
WRF's own `CAL_MON_DAY` (extracted byte for byte from `module_ra_gfdleta.F`
into a one-routine module, because the rest of that radiation module needs
`MODULE_CONFIGURE`), `module_sf_noahdrv`, `module_bl_myjurb`.  Then every
path listed in a `sources-<lane>.list` file here, then `oracle_io.F90`,
`stub_wrf.F90` (service only: `wrf_abort`, `wrf_debug`, single-rank
broadcast shims) and every `run_*.F90`.

A lane adds a driver (`run_<name>.F90`) or a WRF source (`sources-<lane>.list`
+ `SOURCES-<lane>.sha256`) by adding files, never by editing `build.sh`.

## What it runs

Each `run_<name>` with `BUILD_DIR/fixtures/<name>` as its argument.
`oracle_io.F90` writes one directory per case: `<name>.bin` (raw
little-endian float32/int32, Fortran order) and `MANIFEST.txt`
(`name kind rank extents...`).  `woof.verify.urban_oracle.load` reads them;
`ulp_table` measures a port against them.  Publish a lane's fixtures under
`tests/data/oracles/urban/<lane>/` with a `PROVENANCE.md`.

## libmvec

WRF builds phys/ at `-O2 -ftree-vectorize`, where gfortran may call glibc's
4-ULP vector `expf`/`powf`.  The reference is `-O0`; `build.sh` fails on any
`_ZGV*` symbol in an `-O0` object, and a positive control
(`libmvec_positive_control.F90` at `-Ofast`) proves the grep can see one.
`libmvec-report.txt` lists the scalar libm symbols each object calls.

## Toolchain recorded

a development machine, 2026-09-30: GNU Fortran 15.2.0, glibc 2.43.  Record the compiler and
C library in each fixture's PROVENANCE: a port that evaluates a libm call on
the host through `woof.core.noahmp_libm` is a glibc 2.39 transcription.

## Drivers

* `run_init.F90` (infra): `urban_param_init` + `urban_var_init` for all six
  `(sf_urban_physics, use_wudapt_lcz)` pairs, one process each
  (`urban_param_init` allocates its tables once per process).  Gates:
  `tests/test_urban_tables.py`, `tests/test_urban_init_wrf471_parity.py`.
* `run_noah_hook.F90` (infra): WRF's `urban_param_init` + `urban_var_init`
  and then WRF's own `lsm` (module_sf_noahdrv) with the urban arguments,
  eight cases (`off_reference`, the FRC = 0 UCM cases where the blend returns
  the rural words exactly, FRC in (0, 0.99) with TS_URB2D != TSK for the
  T1 recovery, BEP with TSK_RURAL_BEP != TSK, both monthly-albedo arms, the
  eleven LCZ categories).  Gate: `tests/test_urban_noah_hook_wrf471_parity.py`
  (measured ULP table per architecture; TSK_RURAL_BEP 0 ULP).
