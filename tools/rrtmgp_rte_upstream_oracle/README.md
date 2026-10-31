# RTE+RRTMGP upstream column oracle

WOOF's RTE+RRTMGP (ra_lw_physics = 4, ra_sw_physics = 4) has no counterpart in
stock WRF 4.6.1: WRF option 4 is legacy RRTMG.  The referee for this scheme is
therefore the upstream RTE+RRTMGP Fortran, at the commit
`woof/core/rrtmgp.py` transcribes:
`earth-system-radiation/rte-rrtmgp` fa107a16120051c4124305c6b3d4c87059119f58,
coefficient data rrtmgp-data v1.9 (the tables WOOF ships).

## What is compared

`rte_column_oracle.F90` reads columns, calls the upstream frontends in the
order WOOF's production radiation call composes them, and writes every
intermediate:

| stage | upstream routine | WOOF |
|---|---|---|
| LW gas optics, Planck sources | `gas_optics` (1scl + `ty_source_func_lw`, tlev given) | `rrtmgp_gas_optics`, `rrtmgp_planck_sources`, in-solver Planck |
| band cloud optics, ice roughness 2 | `cloud_optics` (2str on bands) | `rrtmgp_cloud_optics` |
| McICA | g-point clouds = band clouds with tau times the mask | the same mask as input |
| LW clouds into gas | `increment` (increment_1scalar_by_2stream) | `rrtmgp_finalize_cloud_lw`, fused solver |
| LW fluxes | `rte_lw`, one angle, broadband | `rrtmgp_lw_noscat` + fold |
| SW gas optics, TOA source | `gas_optics` (2str + toa_src) | `rrtmgp_gas_optics`, `load_gas_tables` |
| SW clouds, delta scaling | `increment` (2stream by 2stream), `delta_scale` | `rrtmgp_finalize_cloud_sw`, fused solver |
| SW fluxes | `rte_sw`, mu0 by column, broadband | `rrtmgp_sw_2stream` + fold |

The McICA masks and the SW incident flux are inputs, identical on both
sides: the mask generator is WOOF's WRF-coupling choice (WRF's kissvec
maximum-random overlap), not part of RTE+RRTMGP, and the TOA normalisation
to the solar constant is likewise coupling.

## Build and run (W1, CPU for the Fortran, one card for WOOF)

    git clone https://github.com/earth-system-radiation/rte-rrtmgp upstream
    git -C upstream checkout fa107a16120051c4124305c6b3d4c87059119f58
    tools/rrtmgp_rte_upstream_oracle/build.sh upstream build
    tools/rrtmgp_rte_upstream_oracle/build.sh upstream build-fixed --boundary-fixes
    card-lease rte-rrtmgp --cards 1 --est 900 --wait 3600 \
      bash -c "source /work/pverify/base/env.sh; \
        bash tools/rrtmgp_rte_upstream_oracle/run.sh $T build build-fixed runs/a"

`run.sh` makes the column set once (`oracle.py make`: 72 columns x 60 layers
by default; `--per-regime 64 --nlay 91 --seed 7` gives the 512-column set),
runs both Fortran builds, runs WOOF under default arithmetic and under
`GPUWM_WRF_EXACT=1`, and writes `compare-*.json` and `refdiff.json`.
`oracle.py bench` times the production stages on the columns tiled to a
given count.

Build flags: gfortran 13.3.0, `-O0 -ffp-contract=off -cpp -DRTE_USE_SP`,
glibc 2.39 (so a real EXP/LOG is that host's expf/logf).

## The second referee: upstream with three boundary fixes

`upstream-boundary-fixes.patch` changes three lines of
`mo_gas_optics_rrtmgp_kernels.F90`.  In each, the Fortran clamps a table
index but keeps the unclamped interpolation fraction, so at the top of a
table it reads the wrong node:

1. `interpolation`: `ftemp` uses `temp_ref(jtemp_)` (unclamped).  At a
   layer temperature of 355 K (the table top) the absorption is read at
   340 K.
2. `interpolation`: `feta = loceta - aint(loceta)`.  At eta = 1 (second key
   gas absent, e.g. zero ozone) the coefficients are read at eta = 7/8.
3. `interpolate1D`: `frac = val0 - aint(val0)`.  At 355 K the Planck
   function is read at 354 K.

WOOF measures each fraction from the node it reads (the fixed behaviour)
and documents the upstream defect rather than copying it.  The
`--boundary-fixes` build is the referee that proves every word WOOF differs
from stock upstream by is one of these three.

## Committed fixture

`oracle.py fixture` writes `tests/data/oracles/rrtmgp/upstream-fa107a1-columns.npz`:
the 72-column inputs, bit-packed masks and a SHA-256 of every column of every
output of both builds.  `tests/test_rrtmgp_upstream_oracle_gpu.py` grades the
three WOOF compositions against it.
