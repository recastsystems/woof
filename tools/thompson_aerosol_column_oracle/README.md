# mp=28 column oracle: WOOF's GPU kernels against WRF v4.6.1, bit for bit

This harness runs WOOF's production `mp_physics=28` adapter
(`woof.core.microphysics_aerosol._apply_thompson_aerosol`, generation
`wrf_461`, `thompson_fork_snow_fall=blend`) on the GPU and unmodified WRF
v4.6.1 `phys/module_mp_thompson.F` on the CPU, on identical float32 columns,
and compares every output word: the eleven moments, theta, 10 cm
reflectivity, the three effective radii and the seven surface accumulations
(23 fields).

A match claim is 0 ULP on every field of every column, under the strict
build (`GPUWM_WRF_EXACT=1`: no FMA, no flush-to-zero, IEEE divide and square
root, WOOF's own libm words).  The same columns also run under default
arithmetic, and both are reported.

## Pieces

| file | what it does |
|---|---|
| `make_columns.py` | the input set: the 42 real columns of `tests/data/thompson_real_columns_wrf461.npz`, 8 synthetic regimes x 4 variants (deep convection, tropical anvil, stable night with fog, snow/ice, marine stratocumulus over the ocean, high terrain near 640 hPa, freezing rain under a warm nose, polluted cumulus), each raw and after 10 and 45 steps of 20 s through WRF's own Fortran, and 19 edge cases (four of them walk the cloud through every path that empties it in one step: whole and droplet evaporation, the cloud-water limiter under rain and under riming, the freeze below HGFR and the melt above 0 C); 157 columns of 49 levels. The first 153 are the original set, unchanged |
| `gpu_run.py` | one adapter call over all columns on the GPU; saves the outputs, the Exner function and layer depths the adapter formed, and the NVRTC options actually used |
| `compare.py` | runs the pristine WRF driver on the inputs the GPU run used and compares all 23 fields bitwise; per field, per regime and per column, plus the "share of WRF's one-step tendency" measure |
| `run_oracle.sh` | the whole thing for a list of time steps, strict and default |
| `host_bitwise.py` | bitwise view of a `tools/thompson_real_column_parity/real_column_parity.py --dump` run: all 64 process rates and the outputs of the kernels compiled for the host, to locate a difference without a GPU |
| `libm_check.cpp` | WOOF's own libm words (`woof/core/kernels/thompson_aerosol_libm.cuh`) against the host C library: expf, logf, log10f over all 2^32 inputs; powf, exp, log, pow, log10, hypot, log1p sampled; atan2's agreement and worst error reported (link with `-lquadmath`) |
| `refl_check.py`, `refl_driver.F90` | the 10 cm echo alone: WRF's own `calc_refl10cm` (a PRIVATE-dropped copy of the pristine module, the oracle's flags) and WOOF's `launch_aa_refl10cm` on the same states, every word compared; writes `tests/data/thompson_aerosol_refl_wrf461.npz` |
| `constdump.F90` | `thompson_init` and `radar_init` module state bit for bit (`refl_check.py constants`), the source of `tests/data/thompson_wrf461_init_constants.txt` |
| `make_stress_columns.py` | stress inputs beyond the 157: every synthetic regime at 2, 17, 49 and 73 levels, isothermal columns at the phase thresholds (190 to 310 K, RH 0.001 to 1.08), thin air, and condensate from the smallest float32 to 0.02 kg/kg; `--bounded` keeps the warm isothermal columns below 700 hPa |
| `make_racg_read_copy.sh` | a measurement copy of a tree that reproduces WRF's out-of-bounds rain-graupel read (see below); never shipped |
| `oracle_io.py` | the WRF driver's stream format and the bitwise metrics |

The WRF side is `tools/thompson_real_column_parity/run_columns_aero.F90`,
built by that folder's `build_wrf.sh` (gfortran 13.3, `-O2
-fno-tree-vectorize`, scalar libm, no FMA; the source SHA-256s are pinned).

## Rerun (W1 or any Linux box with a GPU, gfortran and the WRF source)

```sh
T=<woof checkout>; source /work/pverify/base/env.sh      # PYTHONPATH=$T
bash $T/tools/thompson_real_column_parity/build_wrf.sh \
     <WRF-4.6.1>/phys wrfbuild $T/recast-woof-data/woof_data/data/thompson/tables
# GPU work only through the box's mutex:
card-lease <lane> --cards 1 --est 900 --wait 3600 \
  bash $T/tools/thompson_aerosol_column_oracle/run_oracle.sh $T wrfbuild run 20 5
```

`run/summary-{strict,default}-dt{20,5}.json` hold every number;
`run/summary-*.wrf.npz` WRF's answers; `run/gpu-*.npz` WOOF's.

The same run on the measurement copy, and on the stress sets:

```sh
bash $T/tools/thompson_aerosol_column_oracle/make_racg_read_copy.sh $T tree-wrfread
python $T/tools/thompson_aerosol_column_oracle/make_stress_columns.py stress
python $T/tools/thompson_aerosol_column_oracle/make_stress_columns.py stress-bounded --bounded
for tree in $T tree-wrfread; do                     # each through the mutex
  bash $tree/tools/thompson_aerosol_column_oracle/run_oracle.sh $tree wrfbuild run-$(basename $tree) 20 5
  for nz in 2 17 49 73; do
    mkdir -p s-$(basename $tree)-nz$nz && cp stress/columns-nz$nz.npz s-$(basename $tree)-nz$nz/columns.npz
    bash $tree/tools/thompson_aerosol_column_oracle/run_oracle.sh $tree wrfbuild s-$(basename $tree)-nz$nz 0.001 1 60 300
  done
done
```

Run the measurement copy with its own `CUPY_CACHE_DIR`: the two trees
compile different sources under the same module names.

## Conventions

* The adapter forms the Exner function and the layer depths itself; WRF is
  handed exactly those device arrays, so both codes see identical inputs.
* WOOF stores effective radii in microns, formed as WRF's metre value times
  `1.0e6f`; WRF's answer is converted the same way (the conversion WRF's own
  radiation drivers apply) before the bitwise comparison.
