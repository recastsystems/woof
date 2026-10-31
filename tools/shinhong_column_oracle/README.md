# Shin-Hong wide column oracle, WRF v4.6.1

A column oracle for `bl_pbl_physics = 11` built from realistic columns
rather than branch probes. The branch fixture next door
(`tools/shinhong_wrf461_oracle`, 30 cases x 6 dx) stays; this one adds what
it lacks: columns shaped like the regimes the scheme actually runs in, a
binary dump that keeps NaN payloads and signed zeros, the model-side input
contract, and the reference built the way WRF builds itself.

## What it runs

`make_columns.py` writes 903 columns: 129 base columns times six grid
spacings (100 m, 333 m, 1 km, 3 km, 9 km, 27 km, with dt 1, 2, 6, 18, 54 and
150 s), plus each base column once more with the TKE diagnostic off.
The base columns are 18 families:

| family | columns | what it is |
| --- | --- | --- |
| convective-land-dry | 8 | afternoon mixed layer 1 to 2.8 km, hfx 150 to 450 W/m2 |
| convective-land-moist-cumulus | 8 | humid mixed layer with cumulus cloud water at its top |
| convective-desert | 6 | 3 to 5 km mixed layer, hfx 400 to 650 W/m2, very dry |
| morning-transition | 6 | shallow growing mixed layer under a residual jet |
| stable-night-land-weak | 8 | windy night, low-level jet |
| stable-night-land-radiative | 7 | calm radiative inversion, 5 to 15 K |
| snow-arctic-mixed-phase | 7 | 245 to 262 K, supercooled liquid and ice cloud |
| snow-ice-fog | 5 | 235 to 252 K, ice fog at the surface |
| snow-spring-daytime | 6 | heated snowpack, cirrus aloft |
| ocean-tropical-trades | 6 | trade cumulus under a trade inversion |
| ocean-stratocumulus | 7 | cloud-topped mixed layer under a 7 to 13 K inversion |
| ocean-cold-air-outbreak | 6 | 262 to 272 K air over open water, hfx up to 450 W/m2 |
| ocean-warm-advection-fog | 5 | stable fog over cold water |
| high-terrain-day | 7 | surface pressure 560 to 750 hPa, deep heated mixed layer |
| high-terrain-night | 6 | surface pressure 580 to 800 hPa, stable |
| high-terrain-snow | 6 | surface pressure 550 to 720 hPa, snow, ice cloud |
| windy-neutral-overcast | 6 | 12 to 25 m/s, overcast, land and sea |
| edge-probe | 19 | see below |

Columns use five WRF-like stretched grids (35 to 60 levels, first layer 4 to
60 m). Pressure is hydrostatic from the virtual temperature, moisture comes
from relative humidity over water or ice, and a Monin-Obukhov surface layer
makes ust, br, psim (WRF's fm), psih (fhh) and u10/v10 consistent with each
column's wind, roughness and fluxes. Latitudes include the southern
hemisphere, the equator and the pole. The generator is seeded and its
output is the same bytes on Windows and Linux (the gate checks this).

Edge probes: `br` at +0, -0, plus and minus the smallest subnormal and the
smallest normal; calm columns with zero fluxes; the two prfac2 0/0 probes
(below); `qc` exactly at the 0.01e-3 in-cloud test and one float32 step
above it; a PBL that fills the column; uniform wind at the equator;
subnormal, signed-zero and slightly negative moisture; 900 W/m2 heating; a
4 m first layer; `dy = 4 dx`; a very stable column at the `br = 2` cap; a
strong-shear southern-hemisphere column; an ice-fog column at the pole.

## The input contract

The model owns theta and Exner. WRF's `phy_prep` forms
`t_phy = th_phy*pi_phy` (one float32 multiply) and `module_pbl_driver.F`
hands that to the scheme as `t3d`, with `psfc = p8w(1)` and the tendencies
zeroed. The record stores theta, pi and that product: the Fortran driver
gets `t3d` exactly as WRF would, the WOOF kernel gets theta and exner as
the engine hands them.

## The reference build

`build.sh` compiles the byte-pinned `module_bl_shinhong.F` three ways:

* `wrf-out.bin`, THE REFERENCE: WRF's own flags (`-O2 -funroll-loops`)
  with the vectoriser off, so every libm call is scalar. No FMA (checked).
* `wrf-out-O0.bin`: gfortran 13.3 at -O0 keeps the 17 `x**2.` sites as
  calls to the C library's `powf(x, 2.0)`, and on this host that call is not
  `x*x` for 769,747 normal float32 inputs (`pow2-receipt.txt`). Every
  optimised WRF build folds `x**2.` to `x*x`. This arm is kept as evidence.
* `wrf-out-O2vec.bin`: WRF's flags exactly, vectoriser on (libmvec
  `_ZGVbN4vv_powf`). On these 903 columns it is word for word the reference
  outside the out-of-bounds lanes below.

## Two lane classes reported apart, never hidden

* The prfac2 0/0 probes (columns 812 to 825). WRF divides `15.9*wstar3/ust3`
  with both cubes zero (module_bl_shinhong.F:1010) when `br <= 0` and
  `ust` is zero or subnormal. WRF's surface layers never produce that input.
  WRF writes NaN in 700 `dtheta` lanes and 259 `exch_h` lanes; the kernel
  writes NaN in the same lanes (payload is the hardware's default NaN, x86
  0xFFC00000, CUDA 0x7FFFFFFF). In 588 `tke` lanes the scheme's
  `amax1(q2xk, epsq2l)` meets the NaN: optimised WRF builds return the floor
  (tke 0.005), -O0 and the kernel return NaN. The engine treats a NaN
  `dtheta` as fatal and repairs a NaN `tke` to 0.005.
* `kpbl == kte` with `tke_diag = 1` (columns 840 to 845). WRF reads
  `q2xk(kte+1)`, one past the end (:1557). Their `tke` and `el` are excluded.

## Rerun

On W1 (`/work/pverify/shinhong-pbl11`):

```
T=/work/pverify/shinhong-pbl11/src; source /work/pverify/base/env.sh
bash $T/tools/shinhong_column_oracle/build.sh \
    /work/pverify/wrf-build/src/WRF-4.6.1 /work/pverify/shinhong-pbl11/oracle python
card-lease shinhong-pbl11 --cards 1 --est 600 --wait 1800 \
    /work/pverify/shinhong-pbl11/run_gpu.sh TAG
```

`run_gpu.sh` runs `compare.py` once with `GPUWM_WRF_EXACT=1` and once
without, writing `report-{strict,default}-TAG.json`. The gate is
`tests/test_shinhong_column_oracle.py` (CPU checks plus three GPU tests,
run it under both arithmetic settings).

## Refute sets (`refute_columns.py`)

Two more column sets in the same record layout, read by the same Fortran
driver and the same comparison module.

* `tanh`: csfac = 0.5*(tanh(uwstx)+3) (:847). WRF calls the C library's
  tanhf; on the reference host it is not the once-rounded double tanh on
  118,674,314 float32 arguments in [-10, 14], and csfac moves on 4,098,664
  of them (`tanh_receipt.c`). For each wide-oracle column with a non-zero
  wstar, ust is re-chosen twice so uwstx lands on such an argument; wstar
  does not depend on ust, so every other input word is unchanged. 1,134
  columns, committed as `tests/data/oracles/shinhong_columns_tanh` and gated
  by `tests/test_shinhong_tanh_column_oracle.py`. The kernel spelled tanhf as
  a once-rounded double tanh until lane/verify-shinhong-refute; the strict
  build then differed on du, dv, tke and el of these columns. It now
  evaluates WOOF's own float32 tanhf, `sh_tanhf`.
* `fuzz`: 4,000 base columns from the family builders with a new seed, on
  grids of 4 to 128 levels (first layers 2 to 100 m), with per-level noise,
  random cloud layers, surface scalars drawn apart from one another, the
  whole sphere, and dx from 25 m to 100 km (dt 0.15 s to 3,000 s), three
  runs each: 12,000 columns. Not committed (31 MB of input); the generator
  is seeded and reproduces the bytes.

```
G=$T/tools/shinhong_column_oracle/refute_columns.py; B=<oracle>/build
python $G fuzz fuzz/columns-in.bin fuzz/columns.json 4000
python $G tanh <oracle>/columns-in.bin <oracle>/wrf-out.bin tanh/columns-in.bin tanh/columns.json
$B/run_shinhong_columns_O2s columns-in.bin wrf-out.bin       # in each set's folder
python $T/tools/shinhong_column_oracle/compare.py <set> report.json
```
