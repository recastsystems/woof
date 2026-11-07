# GBM PBL oracle, WRF v4.6.1

Column oracle for `bl_pbl_physics=12` (`GBMPBLSCHEME`, `phys/module_bl_gbmpbl.F`,
sha256 `92d318297a54b3e89a2735da1bf6cfcef27b053c02339c3fb82dbbc4ca59a4cf`).

```
bash tools/gbmpbl_wrf461_oracle/build.sh <WRF_SOURCE_ROOT> <WRF_SCALAR_BUILD> <BUILD_DIR>
```

## What it runs

`run_gbmpbl.F90` calls WRF's own `gbmpbl` entry, the routine
`module_pbl_driver.F` calls under `CASE (GBMPBLSCHEME)`, with one column set
per call as one tile (`its..ite` the columns, `kte = nz`, `kme = kde = nz+1`).
Each set runs 8 steps; between steps the harness advances u, v, theta and the
three moisture species by `dt` times the tendency (species clipped at zero),
recomputes T as theta times pi, and carries TKE_PBL exactly as the scheme left
it.  The port's test repeats that float32 update word for word.

Two links of the same driver:

* `run_wrf` against `phys/module_bl_gbmpbl.o` from the scalar referee build
  (`FCOPTIM = -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`),
  i.e. the words `wrf.exe` computes;
* `run_o0` against a -O0 compile of the pristine source.

`build.sh` fails unless both outputs are byte-identical, and fails if either
object pulls a libmvec vector symbol.  Both objects call only `expf`, `logf`
and `powf` from libm (`oracle-report.txt`).

## Columns

`make_cases.py` builds 86 columns from a fixed seed, all float32:

| set | levels, top | dt | columns |
| --- | --- | --- | --- |
| 0 | 40, ~13 km | 60 s | 10 stratocumulus-topped marine, 12 convective daytime land, 10 stable night, 8 neutral windy, 8 decoupled cloud layers, 8 very shallow PBLs, 6 ice and mixed-phase, 10 randomly layered (several unstable layers, drives pblhgt's layer merging), 2 calm |
| 1 | 16, ~2.6 km | 20 s | 3 convective and 2 stratocumulus PBLs reaching the model top, 2 neutral, 5 layered |

Half the columns start from gbmpblinit's cold TKE (0.001), half from a
spun-up profile.  Stratocumulus, decoupled and ice columns carry RTHRATEN
cooling at the cloud top so the radiative term of the buoyancy budget and
pblhgt's radiative/entrainment extension are both reached.

## Published fixture

`woof/data/gbmpbl/oracle/`: `gbm-cases.bin` (inputs), `gbm-oracle.bin`
(`run_wrf` output), `gbm-cases.json` (layout and column tags),
`oracle-report.txt`, `oracle-sha256sums.txt` and `FIXTURES.sha256`.
`woof/verify/gbmpbl_oracle.py` reads them;
`tests/test_gbmpbl_wrf461_parity.py` grades the port against them.

REGIME is passed through unchanged by WRF GBMPBL. The oracle records it and the replay checks its words. The engine publishes the preserved surface-layer flag as a GBM history diagnostic; WRF has no Registry history row for it.

## Real native replay

`source-pin.json` records the public-domain statement and checked source
hashes. The WRF 4.7.1 GBM source is byte-identical to the 4.6.1 module above.
Instrument an owned strict WRF build, rebuild its GBM object, and relink:

```
python tools/gbmpbl_wrf461_oracle/instrument_native.py OWN_WRF/phys/module_bl_gbmpbl.F --receipt instrumentation.json
```

Run a real WRF forecast with option 12 and the build's big-endian stream
convention. The instrument writes the first two rank-zero calls, at most
16 by 4 columns per call, before and after GBMPBL. It captures all column
and surface bindings, the carried TKE, and all 14 output fields.

```
GPUWM_NUMERICS=strict python tools/gbmpbl_wrf461_oracle/replay_native.py CASE/gbm-capture-0001.bin CASE/gbm-capture-0002.bin --receipt native-replay.json
```

The 2026-11-01 real January case replay matched 45,440 words with zero
differences on an RTX 4090. This is module conformance evidence. It does
not assert whole-model equivalence or observation skill.

## Stock namelist forecast

With `bl_pbl_physics=12`, MM5 surface layer 1 or 91, and microphysics
carrying cloud water and cloud ice in `CASE/namelist.input`:

```
python -m woof go --wrfinput CASE --outdir OUTPUT --products 2m_temperature,mslp_10m_winds
```

This uses the existing native-input forecast, nest, restart, ensemble and
Rust history/render paths. Optional `scalar_pblmix=1` uses GBM's EXCH_H
through the shared CUDA diff4d path, including the aerosol-aware species.
The GBM state remains local to each domain and ensemble member.
