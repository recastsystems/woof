# Native cloud, FARMS, aerosol and eclipse proof

Authority is public WRF v4.6.1, with the byte-identical v4.7.1 radiation
driver and FARMS source retained from the initial proof. `sources.json`
records original URLs, SHA256 pins and the source/table licence checks.
The WRF distribution statement places this code in the public domain.
The isolated strict log10 helper retains the CORE-MATH MIT notice.
No competitor code is used. The external CESM aerosol file is neither
licensed nor shipped; `aercu_fct/opt` remain held with their MSKF parent.

`instrument_native.py`, `instrument_cu_native.py`,
`instrument_aer3_native.py`, and `instrument_eclipse_native.py` each take
an **owned WRF source tree** and verify original source pins before adding
bounded captures to real native calls. They preserve the physics statements;
use a separate pristine owned copy for each instrumenter. Never instrument
another lane's tree or instrument a tree twice. Capture
environment variables and record layouts are documented in each instrumenter.
Use strict native flags `-O2 -fno-fast-math -ffp-contract=off
-fno-tree-vectorize`, at most eight build jobs. The initial cloud/FARMS build
used gfortran 15.2; resumed CU/aerosol/eclipse used gfortran 12 and glibc 2.43.
The eclipse track explicitly matches the latter compiler's optimized
`SIN(ATAN(x))` rewrite. This compiler dependence is part of the referee.

Replay the actual captured inputs through production CUDA launchers:

```sh
export GPUWM_WRF_EXACT=1 GPUWM_WRF_GLIBC=2.43
python tools/cloud_aerosol_wrf471_oracle/replay.py CAPTURE_DIR replay.json
python tools/cloud_aerosol_wrf471_oracle/replay_cu.py OWNED_CASE_ROOT
python tools/cloud_aerosol_wrf471_oracle/replay_aer3.py OWNED_CASE_ROOT
python tools/cloud_aerosol_wrf471_oracle/replay_eclipse.py OWNED_CASE_ROOT
python tools/cloud_aerosol_wrf471_oracle/read_frames.py
```

The resumed replay tools use `capture-cu1`, `capture-cu3`, `capture-aer3`,
and `capture-eclipse` beneath the case root and write small receipts beneath
`evidence/resume`. Strict NVRTC options include `--fmad=false --ftz=false
--prec-div=true --prec-sqrt=true`; ordinary execution uses fast numerics.
Record effective options and assembled source hashes with the receipts.

| Native real-input replay | Output words | Differing words |
| --- | ---: | ---: |
| cal_cldfra3, 128 columns x 39 levels | 19,968 | 0 |
| FARMS, 128 columns, 22 daylight | 768 | 0 |
| ICLOUD_CU=1, CU3 active capture | 29,952 | 0 |
| ICLOUD_CU=2, CU1 active capture | 29,952 | 0 |
| Stock aerosol, 256 columns x 39 levels | 430,336 | 0 |
| Active April 2024 eclipse, geometry/mask/track | 368,896 | 0 |

Cloud replay also preserves all 34,944 input words. FARMS integrates 38 of
39 levels, as native `Integrate_1var/2var` does. Stock aerosol captures the
producer-unfilled upper pressure halo: all 256 captured halo words were zero;
production defines that same zero. This is not a claim that arbitrary native
uninitialized storage has portable values. AOD5503D remains the user-aerosol
carrier and stays at its native allocation words under option 3; the separate
TAOD5503D layer optics are checked by the module replay.

The completed integration proof used stock namelists and the real CLI:

```sh
woof go --wrfinput INPUT_DIRECTORY --rrtmg-variant rrtmg_legacy \
  --outdir OUTPUT_DIRECTORY --benchmark --allow-shared-gpu --no-supervise
# Continue the same imported setup using its complete committed NPZ set:
woof go --wrfinput INPUT_DIRECTORY --rrtmg-variant rrtmg_legacy \
  --outdir RESUMED_DIRECTORY --restart CHECKPOINT.npz --allow-shared-gpu
```

The January 24, 2000 12-15 UTC case is 73x60x39, 30 km, dt 180 seconds,
legacy RRTMG, YSU/Noah, cloud3/FARMS. Separate real CU1/CU3 native and GPU
forecasts exercise both feedback modes; MP28 uses the stock analytic aerosol
initialization (`wif_input_opt=1`, `use_aero_icbc=.false.`). Its 45x45 10 km
child was run with feedback 0 and 1, starting from genuine native restart
files. Both trees continue from GPU checkpoints with zero differing output
bytes. Native warm-start import retains the cloud averages, FARMS/aerosol
outputs and O3RAD needed before a child's first subsequent radiation call.

The April 8, 2024 15-21 UTC case is 60x48x39, 30 km, dt 150 seconds,
WSM6, cloud3/FARMS and eclipse enabled. Forcing is real NOAA GFS exported
through the Rust fetch/preparation path and used by both models. The active
39x39 10 km child was run with both feedback modes from native 17 UTC
restarts; both trees continue from 18 UTC GPU checkpoints byte-exactly.
Root eclipse checkpoint continuation is also byte-exact across all six
remaining half-hour frames. Native eclipse geometry uses xtime+radt and
publishes at radiation cadence. Legacy RRTMG dims the incoming solar
constant; native FARMS diagnostics intentionally remain undimmed.

Two production ensembles completed with member-owned CUDA streams and Rust
reduction: January CU1/CU3 with stock aerosol and FARMS coupling false/true,
and April cloud1/cloud3 with eclipse active. Their durable restart rosters
validate all sealed files and bind distinct member identities. These new
fields use the ordinary full-GPU member route; this proof does not claim
packed-executor qualification. All new carriers use the existing allocation,
nest transfer, checkpoint and history paths. The disabled January run's
entire NetCDF files are byte-identical to the candidate at all four hours.

Rust maps were inspected beside native and NOAA GML SURFRAD at three January
hours and four eclipse hours. Large cloud structures and eclipse passage
agree closely. January has a positive surface-shortwave station bias in
both models; at the April 19 UTC eclipse minimum, observed/native/GPU
shortwave is 40.9/37.22/37.20 W m-2. Other eclipse hours retain model bias;
these few frames are not a forecast-skill qualification. January native
moist-theta defaults and the importer's dry-theta conversion also mean
whole-model trajectories are not word-for-word native replays.

Fast CUDA-event stage profiles use real captured columns and production
launchers. At 4380x39: cloud3 0.14133 ms vs cloud1 0.46085 ms; stock aerosol
0.02323 ms vs existing optics 0.02161 ms; CU modes 1/2 0.00168/0.00213 ms
vs two condensate additions 0.00261 ms. At 2880 columns eclipse is 0.00267 ms
vs solar geometry 0.00948 ms. FARMS uses a measured 64-thread launch and
costs 0.02402 ms vs interpolation 0.00173 ms. Vertical integration alone is
0.01362 ms; native FARMS additionally performs eleven Bird optical solves
per daylight column (each thirteen powers and three exponentials). This
measured physics cost is the exception to the 1.2x target. Benchmark packing
scopes are recorded in each receipt; do not mix them or quote these as whole
model step times. The profile tools beside this file take the owned case root.

The dated lane report and small receipts, exact commands, stock namelists,
Rust figures, source licence checks and source hashes are delivered locally
under `<reports>/WRF-PORTS-2026-11-01/evidence/port-cloud-aerosol/`.
Regenerable forecast output is removed after retaining that paper evidence.
