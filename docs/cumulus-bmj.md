# Betts-Miller-Janjic, cu_physics=2

BMJ runs through the registry-driven CUDA column route. A stock WRF
`namelist.input` selects it with `cu_physics=2`; `cudt` retains WRF's
minute-based call cadence. A prepared WRF directory can be run with:

```sh
python -m woof go /path/to/wrf-run --outdir out/bmj --products t2,refl,precip
```

`woof import-namelist` also imports the selector, domain-specific `cudt`,
and `bmj_rad_feedback`. The option needs moisture but does not impose a
PBL pairing of its own. Existing PBL and surface-layer pairing rules
continue to apply.

The port includes BMJDRV, BMJ, TTBLEX, BMJINIT and SPLINE from public WRF
v4.6.1. WRF v4.7.1 distributes the same module byte for byte. Its SHA-256
is `62d2805aae152b58093d07100000b581458b536164d90f9cecc27a157e2bd92f`.
WRF's public-domain statement is retained in
`licenses/LICENSE-WRF-public-domain.txt` and `NOTICE`. There is no
external BMJ coefficient file. BMJINIT generates 87,972 float32 words on
the GPU at initialization and checks their hash against the registry pin.

One CUDA thread advances each column, with level loops inside the
thread. Workspace is coalesced and bounded by columns in flight rather
than the horizontal grid size. Tables stay on the device after the
initial hash check. The forecast column path has no host copy or
synchronization. Production uses the existing fast numerics setting;
strict numerics remains an explicit proof setting.

CLDEFI starts at WRF's float32 value of 0.6 and persists across calls.
The model accumulates the held PRATEC on every model step, including
steps between BMJ calls. CLDEFI, convective cloud diagnostics and rain
diagnostics are checkpoint-covered fields. BMJ history includes CLDEFI,
QCCONV, QICONV, CONVCLD, CCLDFRA, HTOP, HBOT, PRATEC and RAINCV; some
are extensions to native WRF's default history inventory. Their units
and descriptions come from the native registry.

Native WRF restart import reads CLDEFI, convective clouds, cloud bounds
and held PRATEC/RAINCV. Switching to or from BMJ at a native restart is
refused because that would change the owner of the carried scheme state.

`bmj_rad_feedback=true` merges the carried convective condensate and
cloud fraction into temporary radiation inputs using WRF's radiation
driver rule. It supports RRTM/Dudhia, legacy RRTMG, RTE+RRTMGP and
New Goddard, including mixed or disabled streams. Prognostic condensate
is preserved. Analytic clear-sky radiation refuses this feedback setting
because it has no cloud optical inputs.

The final feedback check runs the production New Goddard CUDA solver over
48 columns and 12 layers under strict numerics. The merged cloud fraction
matches the native overlap rule exactly, surface shortwave flux changes
by more than 1 W/m2, and model cloud water and ice remain byte-identical.
Together with admission and registry checks, 20 targeted tests pass.
This coupling fixture is separate from the 40-layer restart checks and
the 49-layer real forecast. See `evidence/port-cu2-bmj/feedback-completion.json`.

## Verification and its limits

The registry retains `implemented-unverified`. These checks establish
specific code behavior, rather than general forecast skill:

* A strict replay of real inputs captured from a private full native WRF
  v4.6.1 run compared **251,244 float32 words with zero differences**:
  two 150-column, 49-level rows, all eleven output fields and both
  copies of the BMJINIT tables. DT was 15 seconds, STEPCU was 20 and
  radiation feedback was off. All 300 first-call columns were quiet.
  This is real-input evidence for the quiet branch, not an active
  native-convection replay.
* The retained independent Fortran sounding oracle covers 96 columns,
  45 levels, twelve families, four calls and two feedback/cadence arms.
  It exercises deep and shallow paths as well as rejected columns.
  The declared reference, tables and output fields compare bitwise on
  CPU and GPU. See the [oracle instructions](../tools/bmj_wrf461_oracle/README.md)
  for the reference arms and their exact scope.
* GPU tests run the real two-domain executor with both feedback modes,
  one-member and four-member ensembles, and continuous versus split
  checkpoint runs. The checkpoint check compares every serialized
  state, driver and scratch array, including BMJ feedback with legacy
  RRTMG and RTE+RRTMGP.
* With BMJ off, a fast eight-step forecast compared byte for byte with
  the candidate: 158 serialized arrays, 427,392 bytes, plus public
  configuration and restart configuration were identical.

Small replay, timing, default-off and allocation receipts are retained in
[`evidence/port-cu2-bmj`](../evidence/port-cu2-bmj/). The allocation check
measured BMJ at 315 descriptors on the root and 687 on each child of the
four-domain reference configuration, below the 1,024 descriptor limit.
This is a targeted measurement at the recorded inventory-heavy
configuration. The full constructor cross-product was not rerun.

On a shared RTX 4090, CUDA event timing of the production column
launchers gave the following medians. Both include bounded workspace
allocation and exclude table initialization, host uploads and outer
physics-driver preparation. New SAS is the existing column launcher
used as the nearest timing reference; it receives the same
thermodynamics and the documented neutral values for its extra inputs.

| input batch | BMJ | New SAS | BMJ / New SAS |
|---|---:|---:|---:|
| real quiet rows repeated to 90,000 columns, 49 levels | 2.978 ms | 2.959 ms | 1.006 |
| twelve sounding families repeated to 86,400 columns, 45 levels | 6.330 ms | 7.762 ms | 0.816 |

The first batch costs 0.149 ms per model step when amortized over its
20-step cadence. These measurements compare column stages, not whole
forecasts with the two schemes.

A fast three-hour, 300 by 300 by 49 forecast for 2026-10-16 09Z to 12Z
completed all 720 steps through `woof go`, using BMJ, WSM6, YSU,
classic MM5 similarity, Noah and legacy RRTMG. The corrected command
exited 0 after forecast execution and all twelve Rust-rendered maps,
using `--products t2,refl,precip`. Forecast execution took 89.00 seconds,
with 92.14 seconds recorded for the run. All 1,928 tracked runtime
files matched commit `83d90d6dcc1f100144b3854464d620b339bcd05d` before
and after execution. Its four output hashes match the earlier completed
forecast, whose initial render request used an invalid product name.
BMJ produced nonzero convective rain, reaching 3.59 mm. Rendering uses
`woof.rustwx` and `rw_wrfbatch`.

After completing New Goddard and mixed-stream feedback, the same full
forecast passed again from runtime export `6c89cc170282b80fb7916d4c7089fe0b83dfc6d5`.
All 1,928 runtime source checks matched before and after execution.
Execution took 52.42 seconds, with 53.65 seconds recorded for the run.
All four output files are byte-identical to the reviewed forecast, so
its native comparisons, maps and observation metrics apply unchanged.
See `evidence/port-cu2-bmj/feedback-complete-forecast.json`.

Hourly station comparisons covered 233 to 238 stations. Temperature
RMSE ranged from 1.28 K to 1.44 K; the 12Z bias was -0.12 K. The
temperature maps retained the broad observed gradient. Reflectivity
maps showed an overly concentrated southwest core and missed observed
northeast extensions. These are findings from one case. The first-call
module replay does not explain or validate the later forecast rain
placement. MRMS composite reflectivity was available; the optional
one-hour precipitation observation failed to decode and was not graded.

The matching strict native forecast also completed all 720 steps and
four hourly frames. Native and GPU maps were opened beside observations
at 10Z, 11Z and 12Z. The two forecasts retain visually matching
temperature, southwest reflectivity and accumulated-rain patterns.
Native convective rain reaches 3.588678 mm. At 12Z, GPU minus native
RMS differences are 0.025198 K for T2, 0.010576 mm for RAINC and
0.051371 mm for RAINNC. Their largest cell differences are 0.892273 K,
0.486840 mm and 2.444743 mm. These compare fast whole-model GPU
arithmetic with strict native arithmetic. No whole-forecast bitwise
identity is claimed. The forecast and map-review receipts are retained
with the other small evidence. Source pins, commands, receipt hashes,
timing samples and the exact checked scope must accompany any stronger
maturity claim.
