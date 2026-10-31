# km_opt=2 (1.5-order TKE) column oracle against WRF v4.6.1

Runs the unmodified WRF v4.6.1 Fortran for the diff_opt=2, km_opt=2 package
and WOOF's production GPU launcher (`woof.core.dycore.launch_wrf_tke_km`)
on the same float32 words, and compares every output word bit for bit.

km_opt=2 is not a pure column scheme: the deformation tensors read the
neighbouring columns.  Each case is therefore an 8 x 8 patch of 64 columns
that share one regime, 40 levels deep.

## What is compared

WRF side (`wrf_side.py`, library from `build.py`), in the order
module_first_rk_step_part1/part2 call it: `phy_prep`, `compute_diff_metrics`
with WRF's physical boundary extensions, `cal_deform_and_div`, `phy_bc` on
the tensors, `calculate_km_kh` (`calculate_N2` + `tke_km`), `phy_bc` on the
coefficients, then `tke_rhs`.  WOOF side (`woof_side.py`): the forecast's
launcher, plus the existing diagnostic probe
(`tools/wrf_diffusion_oracle/deformation_probe.cu`) that exposes the inline
defor13/defor23/defor33 that `tke_rhs` reads.

Twelve fields: `kmh kmv khh khv` (the four exchange coefficients), `bn2`,
`rtke` (the coupled TKE tendency after the positivity limiter), and the six
deformation tensors `d11 d22 d12 d33 d13 d23`.

Cases (`make_cases.py`; WOOF's own hydrostatic base state plus a float64
perturbation state rounded once to float32): a dry LES convective boundary
layer (dx 50 m, no moisture species on the WOOF side, so the engine's dry
branches run; zeros in WRF's moist array, as mp_physics = 0 carries it),
convective land (superadiabatic
surface layer, shallow cumulus), stable night (inversion, low-level jet,
TKE 0 aloft), snow/ice (cold, saturated mixed-phase layers), ocean
stratocumulus, high terrain (2.5 km ridge, open boundaries, map factor
1.02), tropical deep (saturated with cloud water and ice), map-factor
extremes 0.25 to 4 with southern winds, and an edge-case patch (calm air,
zero, negative, vanishing and very large TKE, an isentropic dry column, qc
exactly at the 1e-5 saturation threshold, strong shear, strong surface
cooling, strong drag; open boundaries).  Each case runs four namelist arms
(`arms.py`): isotropic 0 and 1, isfflx 0, 1 and 2, c_k 0.10 and 0.15, and
the tke_seed arm.  576 columns, 36 runs, 1,158,624 words.

## Builds

* WRF referee: `build.py ... --variant noopt`, gfortran `-O0
  -ffp-contract=off -fno-tree-vectorize`.  Every `**`, `EXP` and `LOG` is one
  scalar libm call; no contraction.  `-O0` lowers `x**0.5` to a power call,
  not SQRT.
* WRF stock flags (`--variant stock`): reported only.  At -O2 gfortran
  vectorises into libmvec (`_ZGVbN4vv_powf`, `_ZGVbN4v_expf`), so it differs
  from its own -O0 build.
* WOOF strict: `GPUWM_WRF_EXACT=1 GPUWM_WRF_EXACT_DIFFUSION=1` (no FMA, no
  FTZ, IEEE division and square root, WRF's diffusion metric order).
* WOOF default: no selector.

## Rerun (box W1, from the lane's tree T)

```sh
source /work/pverify/base/env.sh
cd $T/tools/tke_km2_wrf461_oracle
python build.py <WRF-4.6.1 source root> BUILD           # noopt + stock
python make_cases.py RUNS/cases
python wrf_side.py RUNS/cases BUILD/noopt/oracle.so RUNS/wrf-noopt
python wrf_side.py RUNS/cases BUILD/stock/oracle.so RUNS/wrf-stock
# GPU, through the mutex:
card-lease <lane> --cards 1 --est 600 --wait 1800 bash -c '
  source /work/pverify/base/env.sh
  GPUWM_WRF_EXACT=1 GPUWM_WRF_EXACT_DIFFUSION=1 python woof_side.py RUNS/cases RUNS/woof-strict
  python woof_side.py RUNS/cases RUNS/woof-default'
python compare.py RUNS/cases RUNS/wrf-noopt RUNS/woof-strict OUT.json --label strict
python compare.py RUNS/cases RUNS/wrf-noopt RUNS/wrf-stock OUT.json --got-prefix wrf
python make_fixture.py RUNS BUILD/noopt/build-receipt.json $T/tests/data/oracles/tke_km2
```

`check_fixture.py` replays the committed fixture; the GPU test
`tests/test_tke_km2_wrf461_column_oracle.py` runs it in a strict-build
subprocess.  `receipt_words.py` hashes the words of the existing
smag2d-pinned receipt families for one tree, so two trees measured on one
card can be diffed (`receipts/receipt-family-words-moved.json`).

## Receipts

`receipts/` holds the measured summaries: the lane under each build, the
unmodified verify base under each build, WRF's stock build against its own
-O0 build, and the receipt-family words the lane moved.
