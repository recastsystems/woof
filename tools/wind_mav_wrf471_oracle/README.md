# Native MAV, induction and turbine-file oracle

This oracle uses real inputs captured from a whole native WRF forecast. It
replaces the historical synthetic MAV tolerance claim. `native-pin.json`
records immutable public WRF URLs, SHA-256 values and licences. WRF's public
domain statement covers MAV, its coordinate routines and distributed turbine
table. The later induction additions carry the included Fraunhofer MIT notice.
Read both notices before downloading or copying the sources. No competitor
source is used.

The base is WRF v4.7.1; its MAV and original Fitch modules are byte-identical
to v4.6.1. Induction is pinned to public WRF commit
`c9d69bf7f10536275989826117782da543a4fba7`. Its added THRUST formal is adapted
to a local diagnostic for the older native driver; the arithmetic is unchanged.
The capture patch records the first real module call and the MAV ensemble arm
speeds, without replacing the meteorological inputs or altering calculations.
It assumes one MPI rank/one tile for the captured domain. Production CUDA
operates on the resident domain; this oracle does not qualify distributed
partial-domain turbine ownership.

## Reproduction

Start with an owned copy of a configured strict WRF v4.7.1 source/build.
Place reviewed public source files in `CACHE/<ref>/<path>` as listed by the
pin manifest. This checks every hash before adapting the two modules:

```sh
python tools/wind_mav_wrf471_oracle/prepare_native.py OWNED_WRF CACHE
cd OWNED_WRF
# Configure GNU strict flags: -O2 -fno-fast-math -ffp-contract=off
# -fno-tree-vectorize; keep the build's big-endian Fortran stream conversion.
# Rebuild the registry and whole em_real executable, at most eight make jobs.
./compile -j 2 em_real > native-build.log 2>&1
```

The evidence case uses NCAR's public `wps_30km.tar.gz` forcing from
`https://www2.mmm.ucar.edu/wrf/src/data/wps_30km.tar.gz`, 2000-01-24
12:00–15:00 UTC. The delivered evidence includes exact forcing checksums,
namelists, turbine files, commands, build/run logs and binary captures.
Run `real.exe`, then the whole `wrf.exe` in each case directory. The MAV
case has eight turbines in four separated pairs, wake ensemble 7, overlap
4 and rotation 70.10736067 degrees. The induction case uses Fitch 1,
`windfarm_induction=1`, and eight turbines in four occupied columns.
Both use MP6, MYNN5/surface-layer5, Noah2, RRTMG4/4, `bldt=0`, 60-second
steps and advected TKE. Copy `wind-turbine-1.tbl` beside the location files.

After building the Rust `tools/wind-table` executable, run with CuPy and
an available GPU. `GPUWM_WRF_EXACT` controls standalone module numerics;
`GPUWM_NUMERICS` alone is a CLI selector and does not enable this proof.
The native build used GNU 15.2 and glibc 2.43.

```sh
export PYTHONPATH="$PWD" GPUWM_WRF_EXACT=1 GPUWM_WRF_GLIBC=2.43
export WOOF_RW_WIND_TABLE=/absolute/path/to/rw-wind-table
python tools/wind_mav_wrf471_oracle/replay_mav.py MAV_CASE --debug mav.npz
python tools/wind_mav_wrf471_oracle/replay_induction.py INDUCTION_CASE
python tools/wind_mav_wrf471_oracle/replay_input_path.py MAV_CASE INDUCTION_CASE \
  --rotation 70.10736067
```

The replayers report compiler settings, capture hashes and bitwise float32
word counts, and fail if any output word differs. The production-input replay
also includes the shipped Rust reader, native power-curve conversion, farm
rotation and turbine grouping. Expected counts are 451,140 MAV words
(du, dv, TKE increment, POWER) and 455,520 induction words (du, dv, qke,
POWER, THRUST), each zero differences. MAV's 224 undisturbed and 224 wake
arm speeds also match exactly in the delivered arm comparison.

Unset `GPUWM_WRF_EXACT` for normal fast forecasts. The whole forecasts use
`woof go CASE --outdir OUTPUT --benchmark`; maps and observation comparisons
use `woof.rustwx` with the Rust `rw_wrfbatch` and `rw_verify` executables.
Delivered paper evidence records 13, 14 and 15Z map review, station scores,
per-stage CUDA-event timings, disabled-option identity, restart/checkpoint,
ensemble and both nest feedback tests.

## Native undefined sort

For some dense multiple-upstream layouts native MAV calls `sort_gm` with a
strided `INTENT(OUT)` temporary, then reads it before defining it. A native
whole-run crash and an exact-source, bounds-checked minimal witness are in
`native-mav-undefined/` in the lane evidence. The witness changes incoming
indices `[1,2]` into out-of-range indices and fails the native bounds check.
The CUDA implementation keeps the defined upstream list before sorting.
The qualified four-pair real capture does not exercise undefined behavior;
its replay is exactly equal without a tolerance or residue exception.

## Timing interpretation

Compare stages with the same real captured weather and turbine cells.
MAV ensemble 7 executes 28 turbine/direction arms and GM's 2,500-point
rotor quadratures, while original Fitch has one drag pass. Induction runs a
second CT calculation and corrected rotor/power normalization. A diagnostic-only
ablation keeps the same CUDA layout and outputs while removing induction's
extra physics: it falls to within 3% of original Fitch. It is never installed
under a WRF selector. Exact timings and source hashes are in the lane report.

The speed lane keeps strict arithmetic in its original preprocessor branch.
Fast numerics use a warp for XA's 24 independent nodes and JS upstream
contributions, cache GM's single-upstream geometry outside its disk loop,
and reduce arm diagnostics cooperatively. Fast hub interpolation reads the
needed interface heights directly; it does not prepare a full-domain mass
height field. Fitch's fast ATAN uses REAL32, as the native expression does.
The strict real-input replay remains the authority for exact output words;
`tests/test_wind_fast_quadrature_gpu.py` checks the cooperative integration
against the original serial quadrature in fast numerics.
