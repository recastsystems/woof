# P3 (mp_physics = 50) column oracle against WRF 4.6.1

WOOF's P3 device kernels against WRF's own Fortran, compared word for word:
every one of the 22 output fields, every column, every step.

## What runs on each side

**WRF.** `phys/module_mp_p3.F` from WRF v4.6.1 (tag `v4.6.1`, commit
`d66e442f`), sha256 `716950a3...f6a3c6`, compiled unmodified. The driver is
`evidence/p3-fortran-oracle-20260828/harness/p3_ref_driver.f90`, which calls
`p3_init` and then `mp_p3_wrapper_wrf` with the mp_physics = 50 call shape
(one ice category, two-moment ice, specified droplet number). The lookup
table is WRF's `run/p3_lookupTable_1.dat-v5.4_2momI`; `build.sh` checks that
it is the same text as WOOF's packaged copy (WOOF's has CRLF line endings).

Three Fortran builds (`build.sh`):

| arm | flags and C library | role |
|---|---|---|
| `CR` | `-O0 -ffp-contract=off`, with `crlibm_shim.c` linked ahead of libm so `tgammaf` and `log10f` are correctly rounded | **the oracle for the claim** |
| `A` | `-O0 -ffp-contract=off`, this box's glibc 2.39 | reported |
| `B` | WRF's own GNU flags, `-O2 -ftree-vectorize -funroll-loops`, glibc 2.39 | reported |

Why `CR` is the oracle: the Fortran calls six C-library functions (`expf`,
`logf`, `powf`, `exp`, `log10f`, `tgammaf`). glibc 2.39 rounds `tgammaf`
incorrectly on 39.4 % of [0.25, 36] (worst 6 ULP,
`docs/gf_gamma_known_delta.md`) and `log10f` incorrectly on part of its
domain. So "what WRF computes" depends on the C library it was
linked against. `CR` is WRF on a C library that rounds correctly, and it is
byte-identical, on all thirteen p3-fortref fixtures, to the same driver
built with gfortran 15.2 on glibc 2.43 (the committed dumps in
`evidence/p3-fortran-oracle-20260828/dumps/A/`). Matching glibc 2.39's
incorrectly rounded answers would mean copying a C library's error, which
this project does not do (see also `docs/gf_gamma_known_delta.md`).

**WOOF.** `run_woof.py` calls `woof.core.p3_device.run_p3_device`, the
function the forecast step calls, with the wrapper's own locals and the
tables `device_tables()` uploads in a run, for every arm in `ARM_KERNELS`
(`unfused`, `fused`, `levels`, `sedlevels`; the forecast uses `sedlevels`).
Run once with `GPUWM_WRF_EXACT=1` (strict build) and once without (the
shipped default).

## Fixtures (`make_columns.py`)

Analytic soundings on a stretched 50-level grid, built without calling any
P3 or WOOF code. 78 columns:

| regime | columns |
|---|---|
| deep convection | 12 |
| stratiform with a melting layer | 8 |
| stable night, surface inversion, fog and drizzle | 8 |
| snow, ice supersaturation, mixed phase, freezing drizzle | 10 |
| ocean, marine stratocumulus under a capping inversion | 8 |
| high terrain, surface pressure 58 to 76 kPa | 8 |
| anvil, cirrus and homogeneous freezing near 205 K | 6 |
| hail, rime fraction near 1 at 900 kg/m3 | 4 |
| edge cases | 14 |

Edge cases: no condensate; values straddling `qsmall`; rain with zero number;
ice with zero number; ice number at 1e9 /kg; rime mass above total ice and
rime mass with zero volume; small negative values; temperatures at 273.15,
269.15, 258.15 and 233.15 K; 150 % supersaturation; 5 m layers; drops at
10 /kg and at 1e9 /kg; tiny cloud water at the droplet-size clamp; a 180 K
top; melting graupel.

| fixture | columns | dt | steps |
|---|---|---|---|
| `R01-regimes-1step` | 64 regime columns | 20 s | 1 |
| `R30-regimes-30step` | 64 | 20 s | 30 |
| `R60-regimes-dt60` | 64 | 60 s | 10 |
| `E05-edges-5step` | 14 edge columns | 20 s | 5 |
| `S30-subset-30step` | 10 of R30 (committed) | 20 s | 30 |
| `X01-coldstart` | 6, with `th_old = qv_old = 0` | 20 s | 2 |
| `X02-overflow-column` | 1 real-forecast column, 49 levels | 60 s | 2 |

`X01` and `X02` are outside the claim; each documents one divergence that
is a WRF defect, which WOOF does not copy:

* `X01`: on WRF's first call `th_old` and `qv_old` are zero. WRF guards
  the saturation call with `max(t_old, 1.)` (module_mp_p3.F:2328-2330),
  but the saturation mixing ratio at 1 K is zero in float32, so WRF still
  evaluates `0/0` for `sup`/`supi`. WOOF floors `qvs`/`qvi` at `1e-20`
  (woof/core/p3.py).
* `X02`: the one column taken from a real forecast
  (`tests/data/p3_cold_cloud_overflow_column.npz`, the cell that took a
  6 h forecast non-finite). WRF's immersion-freezing product
  `cons6*gamma(7.+mu_c)*tmp1*dum**2` (:2994-2996) overflows float32 on its
  partial product although the answer is an ordinary float, and the
  conservation limiter turns the `+Inf` into NaN. WOOF recomputes only that
  product when it is not finite (woof/core/kernels/p3.cu, "OVERFLOW
  RESCUE"), so every finite value is still WRF's.

The thirteen p3-fortref fixtures (F01-F12, F13) run through the same
oracle as a second set.

## Rerun

On a Linux box with gfortran and a card, from a tree at this commit
(`$T`), with WRF 4.6.1's source at `$WRF`:

```
O=<work dir>
python $T/tools/p3_wrf461_oracle/make_columns.py $O/fixtures
bash   $T/tools/p3_wrf461_oracle/build.sh $WRF $O
bash   $T/tools/p3_wrf461_oracle/run_fortran.sh $O $O/fixtures A B CR
bash   $T/tools/p3_wrf461_oracle/rain_table_dump.sh $WRF $O/rain
T=$T bash $T/tools/p3_wrf461_oracle/run_gpu.sh $O <tag>   # through the GPU mutex
python $T/tools/p3_wrf461_oracle/compare.py $O/fixtures $O/out/CR $O/woof-<tag>/strict/fixtures/sedlevels
```

`compare.py` prints, per fixture, the number of differing words, the
largest ULP distance, the first step that differs and the columns, then
every differing field with its worst element. `SUMMARY` covers the
fixtures inside the claim.

For the legacy set, copy `evidence/p3-fortran-oracle-20260828/fixtures/*`
and `evidence/p3-cuda-verify-20260829/fixtures2/F13*` into one folder,
merge the two `index.json` case lists, and run the same three steps.

The device test `tests/test_p3_wrf461_oracle_gpu.py` replays three of the
fixtures (`R01`, `E05`, `S30`, with their `CR` dumps in
`tests/data/oracles/p3/`) and the thirteen legacy ones on every GPU run.
`tests/test_p3_wrf461_rain_tables.py` holds the start-up rain tables to
WRF's own on any CPU.

## Results (2026-10-07, box W1, RTX PRO 6000 sm_120, tree 051eda99)

**0 ULP against arm CR**: 2,872,468 words (the five claim fixtures plus
the thirteen older ones), all 22 fields, every step, all four device arms,
strict build and default build alike.  Before the two fixes the same
comparison had 162,330 differing words (up to 2,053,916,876 ULP).

Against arm A (same objects, glibc 2.39's `tgammaf`/`log10f`): 230,604 of
2,559,870 claim words differ, step-1 maximum 4,516 ULP (R01, `ni`,
4.1e-4 relative), growing to threshold flips over 30 steps; the older
set differs in 34,656 of 312,598.  Against arm B (WRF's -O2 flags):
295,401 and 34,924.  That gap is the C library's rounding: A and CR link
the same object files.

Defects fixed in WOOF: the rain tables' NumPy libm (now WOOF's own
`expf`/`powf` and a correctly rounded `log10f`;
tests/test_p3_wrf461_rain_tables.py) and the subnormal flush CuPy's
RawModule imposed (now `--ftz=false` through the direct NVRTC route;
tests/test_p3_cuda_gpu.py).  WRF defects documented, not copied: X01
(first-call 0/0) and X02 (immersion-freezing overflow to NaN, 20 NaN
words in WRF).  Open: the CPU reference arm (`p3_backend = "reference"`)
is not bit-identical to WRF (run_reference_arm.py: 5,138 of 54,720 R01
words in one step).


## Two-category port (mp_physics=52)

The shared kernels are instantiated at n_cat=1 and n_cat=2. The second
instance carries prognostic droplet number and qi2/ni2/qir2/qib2, reads
table 2 for inter-category collection, selects destination categories,
enables rime splintering, merges close populations, and sediments and
diagnoses both categories. The CPU reference remains one-category and
refuses mp=52 because it lacks those operations.

The check is `tests/test_p3_2cat_gpu.py`. It generates the column deck,
runs default and GPUWM_WRF_EXACT=1 through all four launch arms, and uses
`check_hashes.py` against the pinned WRF 4.6.1 CR referee. The fingerprints
cover every post-wrapper state and diagnostic and each successive-state
net tendency. Internal transient process rates are not wrapper outputs.
The negative control flips one bit in qicol[cc][c] after its limiter and
must fail the same state/tendency gate. Missing cases and empty manifests
also fail the comparison command.

The resumed lane's historical receipts report per arm: 5,880,930 MP52 state words and their net tendencies;
2,559,870 MP50 state words and their net tendencies. Both modes match
exactly. The inherited CR referee uses the original Fortran object with
its established rounding shim. The stock-linkage arm A is separately
reported and is not claimed bit-identical. The cold-start fixture remains
outside the column claim, following the MP50 oracle's established first-
call saturation guard distinction.

Full-state hashes from the 600-second public-input CLI forecasts are
retained by `forecast_receipt.py`. The input extension helper reads and
writes through Rust and checks every original field in its wire dtype.
Weather planes are derived by `rw_p3_planes` and rendered by rw_wrfbatch.
The plane helper is supplied as Rust source and built with Cargo's locked,
offline recipe. The older prebuilt executable is not part of this port.

No matched WRF full-forecast trajectory or observation skill is claimed.
Mixed-scheme nest edges and offline parent ingestion of MP52 have explicit
category-partition refusals. Same-scheme transport includes all categories.

### Candidate qualification (2026-11-01)

The MP52 lane uses the WRF v4.6.1 public-domain source and tables. The P3
source in the native v4.7.1 build is byte-identical to that pin. Source and
asset hashes, commands, private instrumentation, configs, logs and Rust
figures are retained in the lane evidence under
`<reports>/WRF-PORTS-2026-11-01/evidence/port-mp52-p3-2cat/`.
`qualification_20261101.json` records the portable receipt totals.

The real case is 2024-05-25 18–21 UTC, Oklahoma/north Texas, 150×150 mass
columns, 49 levels, 3 km, 12-second steps. Native WRF completed all 900
steps using four MPI ranks. The instrumentation captures real P3 wrapper
inputs and native outputs at iterations 2, 300, 600 and 900, four stripes
per iteration. CUDA strict replay compares all 16 calls: **1,456,320 words,
zero differences**, with both ice populations and prognostic cloud number
active in the later captures. Native initialization of table 2 compares
all **500,000 words exactly**. Rimed mass is zero in the selected real
stripes; the historical synthetic fixtures exercise that branch but do
not establish real rimed-cloud agreement.

A complete default-fast `woof go` forecast completed 900 steps in 64.4
model wall seconds on a shared RTX 5090. Reflectivity, 2 m temperature and
accumulated precipitation were inspected using `woof.rustwx` and the Rust
renderer at 19, 20 and 21 UTC beside native WRF and public MRMS composite
reflectivity. Both models produce weak convection and miss the stronger
observed north-Texas storms. Similar temperature gradients and small
precipitation totals are visible; no observation skill is claimed. This
comparison is visual, not a whole-model bitwise WRF trajectory comparison,
and the registry maturity remains conservative pending ratification.

The real run exposed pointer-table uploads on every MP52 call. Persistent
workspace bindings now retain those device tables. Before/after forecasts
are byte-identical across all four hourly files. CUDA-event medians for
22,500 columns × 49 levels on RTX 5090 are:

| stage | MP50 (ms) | MP52 (ms) |
|---|---:|---:|
| prepare | 0.215808 | 0.256672 |
| process | 0.222880 | 0.378528 |
| sedimentation | 0.122528 | 0.171680 |
| finish | 0.075424 | 0.130720 |
| total | 0.647520 | 0.937312 |

The historical **1.448×** result missed the 1.2× target. It did not establish
a physics-cost exception: ordinary native REAL expressions still used
replay math with binary64 internals in fast mode. The 2026-11-02 speed
change selects CUDA float transcendental leaves for default-fast MP52.
Both strict selectors retain the replay leaves and operation order; the
native source's explicit double exponentials remain double.

One paired, interleaved run on a shared RTX PRO 6000 Blackwell measured
MP50 **0.743696 ms**, MP52 before **1.034368 ms**, and MP52 after
**0.735168 ms**, or **0.989× MP50**. These are medians of per-sample sums
of four kernel times, using the same real iteration-300 stripe and grid
size as above. Separate counter profiles on RTX 5070 Ti showed FP64
pipeline use falling from 77.82% to zero in prepare and from 50.86% to
2.64% in process, while memory traffic stayed about 439 MB per step.
There are still four launches and no scheme host sync. The strict real
replay remains **0/1,456,320 differing words**. Commands, raw samples,
profiles and the forecast receipt are in the speed lane's report and
`qualification_20261102.json`. The earlier JSON receipt remains a record
of the earlier code. Shared-device launch gaps are not device stage times.

A 19→21 UTC GPU checkpoint continuation matches **41,012,238 history words
exactly**. Same-scheme parent/child runs with feedback 0 and 1 each match
**47,578,866 words exactly** after checkpoint continuation, across both
domains. All P3 number, second-ice and old-thermodynamic carriers are in
the saved members. These runs exposed a missing serialized shortwave
provenance flag, now retained in resident and streamed checkpoints.
Independent seeded ensemble members execute both categories without state
aliasing. Streamed diagnostics retain canonical state-array identities.

A native `wrfrst` warm start completes one strict step. It maps `QICE2`,
`QNICE2`, `QIR2`, `QIB2`, `TH_OLD` and `QV_OLD` without inventing carriers.
This actual native file contains finite negative moisture/number residues;
strict mode preserves their words and records the minima/counts. Default
fast mode retains its existing negative-input refusal. The strict solar
ASIN gate now accepts the native scalar profile witnessed by the bundled
Fortran probe as well as its previously qualified profile; unknown profiles
still refuse. No default-fast numerical path is changed by that gate.

Stock WRF namelist admission/import, histories, restart inventories and
both same-scheme nest feedback modes are wired. Mixed-scheme nest edges
and offline-parent ingestion still refuse because no verified category
partition contract exists. MP53 is not advertised or substituted. Merge
the MP51 lane before this lane and reconcile only the P3 registry entries.
