# MSKF (`cu_physics=11`)

The native multiscale KF scheme runs as CUDA column physics through the stock
WRF namelist door. The source is WRF v4.7.1 `phys/module_cu_mskf.F`, byte
identical to WRF v4.6.1. Source and licence hashes are in
`tools/mskf_wrf461_oracle/source-pin.json`. The WRF public-domain statement and
the embedded permissive aerosol notice are retained. No competitor source was
used.

Use the original `namelist.input`, `wrfinput_d0*` and `wrfbdy_d01` together:

```sh
woof go --wrfinput PRIVATE_WRF_INPUT --outdir forecast --checkpoint-at end
```

The physics namelist selects `cu_physics=11`, with WRF's `cudt`,
`kf_edrates`, `aercu_opt` and `aercu_fct`. MSKF publishes cloud
feedback as WRF requires. The native incompatibility with `km_opt=5` is enforced.
The declared native `kfeta_trigger=2` inputs never reach the executed MSKF
trigger, so both native selector values run the same trigger physics.

Fast arithmetic is the default. Binary64 momentum transport and aerosol
elementary functions use CUDA libm; strict module replay is opt-in through
`GPUWM_WRF_EXACT=1 GPUWM_NUMERICS=strict`. The fast aerosol-off specialization uses CUDA float elementary functions
and permits FMA. Strict replay and aerosol-coupled operation order remain pinned.
Fast and strict aerosol-off checkpoints carry separate algorithm identities. This does not change other schemes' arithmetic.

All 35 module carriers belong to the domain and are retained through GPUWM
checkpoints, continuation and history. Held thermal, moisture and momentum
tendencies feed subsequent steps. The history extension exposes native
Registry names, plus `MSKF_CU_ACT` for native WRF's local activity flag. Native
warm starts read the allocated module packages, rebuild absent inactive
packages as zero and regenerate the local activity flag on the next call.
They do not invent missing active state. Fixed nests run with feedback off
and on; ensemble members own independent carriers.

Positive `aercu_opt` requires MP40. The default provider builds a checked
NASA MERRA-2 GMI climatology from public US government data. It is **written
from papers, not bitwise to WRF**. Rust performs conversion and horizontal
interpolation; CUDA retains time/pressure interpolation and activation.
See [public forcing preparation and approximations](public-cumulus-aerosol.md).
The uncleared CESM file is never selected by the public runtime.

The retained aerosol fixtures use the documented native defined-read reference
and carry an uninitialized-memory witness in
`tools/mskf_wrf461_oracle/build.sh`. They are separate from real native replay.
The real-run proof uses `aercu_opt=0`, so inactive native aerosol dummy arrays
are never indexed. Its receipt counts zero history extensions separately.

The measured active-input stage cost is higher than KF. MSKF adds ten
binary64 harmonic pressure solves and convective momentum transport, absent
from the nearest KF column implementation. Exact measurements and their
nonprecipitating input scope belong in the final verification receipt.

The 2026-11-01 receipt in `docs/evidence/mskf-20261101/verification.json`
records a fresh strict native replay of 107,520 native output words and
62,848 inactive zero-extension words at zero differences. Default arithmetic
and CUDA graph replay repeat these same 128 columns at zero differences.
The real captured columns are nonprecipitating, with 192 nonzero thermal
output words. Deep-precipitating and positive-aerosol fixtures remain separate
from that real-input proof.

The earlier 2026-11-01 uninterrupted stock-namelist `woof go` forecast covers 2026-10-16
09 to 12 UTC on a 300 by 300 by 49, 3 km grid with `aercu_opt=0`, MP6 and
radiation 4/4. All 720 steps completed in 144.1 seconds of integration,
with 37.89852 seconds of separate kernel compilation. Four hourly native/GPU
map pairs use the production Rust renderer, with 23 collocated ASOS
temperature stations per hour. The northwestern cooling, warmer Gulf-side
temperatures and developing southwestern reflectivity band align with the
native reference. This establishes one case comparison. No radar observations
were scored, and no general forecast-skill claim follows from this case.

The 2026-11-02 column-kernel speed receipt at
`docs/evidence/mskf-20261102-speed/verification.json` records one paired,
interleaved fast kernel measurement on the RTX 4090: 7.312384 ms before,
6.797312 ms after, and 1.018880 ms for CU1 (8,192 columns, 49 levels).
The change removes software float libm and its binary64 polynomial work
from the aerosol-off fast specialization and permits contraction. It cuts
local-memory spill requests from 113,920 to 5,120; local storage falls from
640 to 416 bytes. Strict real-input replay retains zero differences in
170,368 words, including 107,520 native output words.

The matched real-forecast stage now costs **3.030382x CU1**: 9.008338 versus
2.972674 ms per call, 59 warm calls per arm, on the same RTX PRO 6000 and
commit. Independent three-hour trajectories alternate AB/BA at every step.
The whole warm-step ratio is 1.158206x. The first actual step is kept
separately. The older column-kernel deck costs 6.6714x CU1.
The [forecast receipt](evidence/retime-20261102/cu11-summary.json) names the
logs and source commit. Strict replay on the pinned GNU 13.3 WRF 4.7.1
referee adds 67,904 zero-difference words (42,880 native words plus 25,024
inactive extensions), whose inputs and references are retained under
`~/weather/data/datasets/paper/wrf-ports/spd-retime-rad-cu/cu11/`.
 Nsight Compute attributes
55.9 percent of sampled activity to the native momentum/pressure routine,
including its ten harmonic solves. WRF explicitly declares this routine's
fields and solve arrays `real(r8)` (`module_cu_mskf.F:7566-7608`);
`module_cam_shr_kind_mod.F:20` defines that kind with
`selected_real_kind(12)`. The nearest CU1 scheme has no such pressure solve.
The profile also records 255 registers, 8.29 percent achieved occupancy and
uncoalesced accesses; the measurement does not claim an irreducible lower
bound. Full commands, profiles and scope are retained with the speed receipt.
MSKF is a named, explicit component override with its measured cost stated.
It must never become a default or a recommended preset.

The speed candidate completed the same three-hour fast forecast on 2026-11-02:
720 steps from 09 to 12 UTC, four hourly frames, and no stability retry.
The run reports 459.9 seconds, including a 305.67-second cold first step on
the shared RTX PRO 6000. That receipt predates the startup fix below.
Warmup selects the runtime aerosol tier instead of compiling the unused
full aerosol unit. Rust temperature and
reflectivity maps at 10 and 12 UTC were opened beside the original maps;
the temperature gradient and southwest Texas reflectivity band agree by eye.
This is not a pixel-identity or general forecast-skill claim.

The startup fix ships source-bound MSKF images for both aerosol tiers,
fast/strict arithmetic and sm_89/sm_90/sm_100/sm_120. The two exact compiler
libraries currently supplied are NVRTC 13.4.59 and 13.4.92 with CuPy 14.2.0.
The loader checks the compiler identity, assembled source and options before
using an image. An explicit pack path or disabled pack remains authoritative.
`python tools/build_mskf_pack.py --work BUILD_DIR --workers 4` rebuilds the
images with independent compiler processes. Its direct NVRTC input contains
the engine headers and does not traverse CuPy's unrelated `nvToolsExt.h`
include. This is the ordinary loader path; no startup flag is required.

The whole installed-wheel cold forecast at commit `b7462a62a42` completed
60 steps on the RTX PRO 6000 with empty kernel, CuPy and CUDA driver caches.
Its first step took 19.1382 seconds. Both loaded MSKF images match the shipped
pack. Four compiler workers were observed, and the longest sampled interval
with zero or one active process was eight seconds across the complete run.
These CPU samples do not by themselves establish when the GPU waited.
The command, exact compiler hashes, cache paths and image hashes are in
`docs/evidence/mskf-20261102-retime/cold-start.json`. This startup check is
separate from the matched forecast-stage cost measurement.

A whole one-minute CU1 control keeps the MSKF option off. With the same
runtime, both history files and all 216 checkpoint array members are byte
identical before and after the MSKF changes; all 47 compiled module images
also match. Checkpoint header differences are limited to run and source
provenance. An earlier comparison with six different compiled images had
tiny surface-field differences and is retained as a failed check; its cause
is not claimed. The corrected control and exact scope are in the speed receipt.

The resident executor records health and stability flags on the device and
polls one bounded pinned snapshot asynchronously. Step polls query completed
CUDA events without waiting for them. Every domain ledger drains before
history, checkpoint publication and final return, including exception paths.
A transient failure can advance through internal nest work before detection;
it cannot publish a successful output or return. Debug mode retains immediate
attribution. Periodic whole-state validation and streamed store health checks
retain their existing cadence and semantics. Tests inject actual GPU failures
and check these publication boundaries and nondefault stream ordering.

The `aercu_opt=0` row requires the measured native-physics opt-in exception;
it does not meet the 1.2x ordinary admission target.
The registry remains `experimental-runtime`, with scientific evidence `none`,
because a single short case does not establish general skill. The original
CESM climatology remains uncleared; the public replacement has separate evidence. Production GPU lifecycle checks exercise
both fixed-nest feedback modes and independent ensemble members; complete
real nested or ensemble forecasts were not added to this one-case proof.

`kfeta_trigger=2/3` is refused with MSKF. WRF 4.7.1 executes these alternative triggers only in KF-eta (`cu_physics=1`); the MSKF path does not consume them.
