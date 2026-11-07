# Previous Kain-Fritsch, cu_physics=99

This option runs WRF's previous KF scheme on CUDA. It is opt-in and remains
`implemented-unverified` pending broader observation-skill qualification.
CU99 cadence, compatibility-clock advancement and bounded CUDA tile
submission run in Rust; column arithmetic and carrier updates run on CUDA.
Python marshals launch metadata under the repository orchestration boundary.

Use a stock WRF run directory through `woof go --wrfinput WRF_RUN --outdir OUT`, with
`cu_physics=99` in `namelist.input`. `kf_edrates=1` additionally writes
`UDR_KF`, `DDR_KF`, `UER_KF`, `DER_KF` and `TIMEC_KF`. The existing namelist
import and authored experiment paths accept the same selectors. With
`kf_edrates=0`, these diagnostic planes are not allocated or published.

The public WRF source is pinned in
[`PIN.json`](../tools/kf_old_wrf461_oracle/PIN.json). The column source and
model constants are byte-identical between v4.6.1 and the native v4.7.1 pin.
WRF distributes these files under its public-domain statement; the complete
UCAR notice is retained in `WRF-LICENSE.txt` beside the pin. CU99 uses no
external table. No competitor implementation is used.

`woof/core/kernels/kf_old.cu` transcribes the column arithmetic and its
three native fatal checks. The shared KF adapter selects this kernel only
for 99. WRF's old-KF hold condition is `NCA > 0.5*DT`, and expiry clears
tendencies when `NINT(NCA/DT) <= 0`. Its internal W0AVG update runs once.
These differ from CU1's clock and trigger-history contract. The CU1 path
keeps its own kernel, lookup table and workspace.

Fast numerics uses CUDA's standard `expf`, `logf` and `powf` in the WRF
equations. Strict numerics selects the native-matching scalar helpers with
`GPUWM_KF_OLD_STRICT`; this define separates the compiled-image cache keys.
Both modes retain the complete source-layer search, plume iterations,
species folding and native fatal checks. Strict proof does not imply
bitwise identity for fast numerics.

Species folding follows WRF's `F_QR`, `F_QI` and `F_QS` flags. With
`mp_physics=0`, the raw cloud-water tendency is retained but not applied:
WRF's passive-vapour package has no cloud-water prognostic. This scalar
gate also applies when held tendencies are recoupled after a restore or
nest relocation.

The optional diagnostics, W0AVG, held tendencies, NCA and rain rate are
domain-owned carriers. The implementation connects allocation and memory
admission, history metadata, restart identity, checkpoint storage and both
nest feedback modes. The lane's GPU checks compare uninterrupted stepping
against checkpoint continuation and compare ordinary concurrent members
against their separate executions. Distinct held diagnostic values make
missing or shared planes observable. Native WRF restart import also restores
CU99 NCA/PRATEC/RAINCV, W0AVG and the optional diagnostics, binds the
trigger history to the restored state/time, and refuses missing required
planes. The targeted native-overlay regression exercises a real GPU driver
and advances its held rain clock; it does not represent a real native
restart forecast. These lifecycle checks use small
synthetic states; active column arithmetic has its own native oracle.

The retained generated fixture grades 141258 output words over 256 columns
at zero differing words with `GPUWM_WRF_EXACT=0` and `1`. Both pytest
processes use the battery's `GPUWM_NUMERICS=strict` compiler policy. This
fixture does not establish bitwise identity under the production fast
compiler policy. It is a module harness fixture, not a real-run capture.
The native instrumenter and replay
commands in the [oracle README](../tools/kf_old_wrf461_oracle/README.md)
provide the separate real-run proof. Instrumentation is confined to the
private native proof build and does not enter the forecast path. The strict
real-run replay grades 104448 captured native output words at zero
differing words, plus 192 inferred zero-fatal witnesses. These 192 real
49-level captures span steps 1, 240, 480 and 720. The sampled captures
have zero convective rain. They establish
the real input path but do not prove active real-run convection; the
separate generated fixture covers active columns.

The lane ran a 300 by 300 by 49, 3 km forecast for 2026-10-16 09Z to 12Z
through `woof go`, with CU99, `kf_edrates=1`, MP6, RRTMG, YSU, MM5 and
Noah. Default numerics completed 720 steps and wrote four hourly frames.
The inputs are a shared WRF-compatible direct export; provenance does not
establish that `real.exe` generated them. Production Rust bridges perform
NetCDF reads, writes and weather-map rendering. Airport temperatures are
point observations within 20 minutes of each valid hour. This short case
does not establish observation skill, and no radar precipitation comparison
or whole-model bitwise parity is claimed. A four-rank strict native run
provides all four matched hourly frames; both models are compared by eye
on Rust T2 and accumulated-rain maps with airport observations. Earlier
incomplete native runs are superseded; their signal-9 cause remains unknown.
The final lane report records exact terminal success and replay counts.

The initial native-math implementation measured 2.205 ms against 1.224 ms
for CU1 on a shared RTX 4090. A subsequent RTX 5090 profile of the same
90000 by 49 quiet columns counted zero TPMIX calls and 15.57 million
power evaluations in the source-layer search. Iterative plume physics
therefore did not explain that timing. Entrainment prelude writes added
0.073 ms in the diagnostic ablation. Replacing the native-matching scalar
functions with CUDA's standard functions reduced the isolated default
stage from 1.752 to 0.828 ms, without changing the WRF algorithm. These
diagnostic measurements are distinct from the production paired
measurements. The Rust launcher on the RTX 5090 measures CU99 0.886816 ms
and CU1 1.215424 ms, ratio 0.729635, retaining all seven interleaved
samples. The earlier RTX 4090 measurement was 1.049632 ms against
1.147904 ms, ratio 0.914390. The full forecast and
per-stage receipts are in the lane report. The repeated initial
columns do not establish a convective-workload speed bound.

CU99 uses the MIT-licensed `rw-cu99-launch` Rust crate (ABI 2) for both
per-step scheduling and bounded CUDA tile submission. The native planner
computes WRF NINT intervals, first-call/modulo due events, adaptive event
overrides and compatibility-clock advancement from each domain's restored
elapsed time and epoch. It has no mutable timer, Python callback, field
read or CUDA wait. This keeps restart, late-starting nest and concurrent
member clocks independent. Python marshals scalar metadata and submits the
CUDA calls selected by that plan; it performs no CU99 cadence calculation.
The shared model retains Python launch orchestration, as in the other
cumulus ports. This change is a production CU99 scheduler, not a launch
recording or a claim that the entire shared model was rewritten in Rust.

The bridge passes device addresses and scalar metadata on the original
stream. The normal Rust workspace and bridge packaging include it. The
CU1 cadence and submission path are unchanged. Domain-clock tests cover
720 steps, checkpoint timestamps, shifted nest epochs, adaptive overrides
and internal/outer step ratios. Actual GPU checkpoint continuation checks
also refuse the old Python cumulus predicate, demonstrating that the
production CU99 path uses the native scheduler.

The original-member MYNN/RUC/RRTMGP coordinated path now joins CU99
columns. Each original adapter retains its trigger clock, W0AVG, held
rates, rainfall, diagnostics, native result provenance and fatal handling.
Joined workspace and return banks are admitted before allocation. Its
configuration-only plan reserves a conservative full-column workspace
ceiling without creating a CUDA context; runtime uses the target's bounded
SM count. Equal-clock member tests compare all composed forcing and
checkpoint-manifest words, including optional diagnostics and adaptive
steps. A joined raining/quiet pair also matches the native oracle words.
The scheduled two-step test retains real member dycores.

The default Thompson/Noah/YSU/MM5 packed graph also accepts CU99 with
legacy RRTMG. Each member owns the six adapter carriers, raw held rates,
clock, rain and diagnostic output; a third persistent coupler keeps its
held forcing independent of PBL and radiation. Configuration-only
preflight prices the carriers and due-call workspace without initializing
CUDA. Restart, nests, adaptive clocks and unqualified suites retain their
ordinary member runner with the original selections.

Two-member checks compare all persistent scientific arrays, forcing,
outputs and microphysics after six calls, including held-rate expiry, both
entrainment settings and boundary rings. Four complete moist RK steps
match separate member state words and Rust history file bytes with both
profile and shared 3D base fields. These are synthetic lifecycle checks,
separate from the real forecast above. The full RK check exposed a missing
stock cumulus rain-number seed in the packed Thompson entry. The binding
now retains that original seed inside each member's ring guard, broadcasting
only the shared immutable base; source structure changes refuse a stale
binding. No physical formula or floating-point order was changed.
During this check the joined radiation call was found to drop SWDDIR,
which controls production of diffuse flux. Retaining and pricing that
original field restores the ordinary member outputs.

The native-input `go` door carries `--restart-from` and `--run-seconds`
through to the same WRF runner. A real WRF-written restart at step 4
continued through steps 5 and 6 with CU99, including W0AVG, held rain and
the optional diagnostic planes. SHDAVG is classified as setup climatology:
WRF uses it to initialize FRC_URB2D, which is separately restored. Native
namelist imports retain the candidate's omitted `hailcast_opt=-1` behavior.
Automatic hail diagnostics remain conditional on grid spacing at most 4 km
and nonzero microphysics; the 30 km real.exe case below does not activate
them in either the baseline or the candidate. The earlier
native-default change was forward-reverted after review because it changed
every stock run independently of CU99. A proposal to change that shared
default is outside this port.

Before that reversion, one whole fast `woof go` forecast completed all 720
steps and four hourly frames. All 825 retained field hashes match the
previous Rust-launch forecast; the only removals are 13 hail diagnostic
variables at each hour. The final maps were rendered again through
`woof.rustwx` and inspected beside native WRF and airport observations.
That older field inventory does not describe the restored hail default.

No claim that the implementation is optimal follows from the scalar-math
ablation. The paired timing uses quiet real columns and establishes no
active-convection workload bound.

The lane report and evidence directory carry the exact commands, native
build configuration, source and executable hashes, capture word counts,
forecast receipts, map review, timing samples and deletion inventory.

The shared resident executor now polls health through nonblocking pinned
snapshots and CUDA event queries. Every history, checkpoint and successful
return boundary drains pending failures. This forward change comes from
`d8fd86898736223a993c1ffd91605fb59763c7ab`; CU99-specific actual-device
checks retain transient failures and validate private stream order, output
refusal and immediate debug behavior. Internal clock progress can advance
before an asynchronous fault is reported; no failed state can be published.
The former per-step blocking resident health read is removed. CU99 now
uses the native per-step planner described above; shared model launch
orchestration remains Python.

The forecast after this health change also completes all 720 steps.
All 825 field instances, including 494144276 numeric words, have the same
typed-byte hashes as the earlier mapped forecast. Median complete step
wall time is 16.318 ms, mean 22.360 ms, with a 56.503 s process wall time.
These shared-card observations are separate runs, not a controlled speedup
estimate. The retained Rust maps and native/airport comparisons therefore
still describe the final fields. Actual-device checks cover asynchronous
fault retention, nondefault streams and refusal before history, checkpoint
or successful return. The lane's fourth-pass evidence retains exact commands,
configuration, complete output hashes and the build-artifact inspection.

The [joined SWDDIR transport fix](defect-joined-radiation-swddir.md) has an
independent commit and a native-evidence note and remains default-on.
The separate HAILCAST default change is reverted, including its obsolete
defect note. Neither that old change nor its historical reduced field
inventory is part of the current CU99 behavior.

After the native cadence change, the final built ABI 2 replays all 104448
captured native output words with zero differences (plus 192 derived fatal
witnesses). Twelve native-clock/FFI checks and four Rust planner/launcher
tests pass. The whole fast stock-namelist forecast completes 720 steps and
four hourly frames in 44.184 s process time; median full-step launch wall
is 12.904 ms and mean is 20.919 ms. All 825 field instances and 494144276
numeric words retain the mapped forecast's bytes. All sixteen Rust/native
T2 and rain maps with airport observations were inspected again. These are
shared-card observations, not a controlled whole-model speed comparison.
The final paired quiet-column stage measures 0.848576 ms for CU99 versus
1.179552 ms for CU1, ratio 0.719405. Exact commands, final built-library
hash, field manifests, map review and separated defect patches are in the
lane report's fifth-pass evidence. The current report supersedes those
historical readiness and inventory statements after the port/all merge;
registry maturity continues to withhold a formal observation-skill claim.

The skeptic-2 completion forward-merges port/all `c36c3b16` and reverts the
unrelated HAILCAST-default change. Independent stock real.exe initializations
provide CU1 and CU99 inputs for 24 January 2000, 12–15 UTC, 73 by 60 by 34
cells at 30 km, with MP6, RRTMG, YSU, MM5 and Noah. Both strict native WRF
and fast `woof go --wrfinput` complete 180 steps and four hourly frames.
All sixteen paired Rust T2/rain maps were inspected with airport temperature
observations. At hour three, GPU/native T2 mean absolute difference is
0.029616 K and accumulated-rain difference is 0.018865 mm. Both models show
a cold bias at some airports; this comparison is not a skill certification.
The native capture replay remains zero differences across 104448 original
output words. Its 192 captured input/reference packets are permanently kept
under `datasets/paper/wrf-ports/port-cu99-kfold/cu_physics-99/`.

The same stock CU1 case matches next/2.9.12 across 805 field instances and
15095212 original storage words, with zero differences. Automatic hail is
inactive at 30 km in both arms; the restored import default and existing
3 km hail allocation/selection tests cover that conditional behavior.
The complete production cumulus stage measures median CU99 1.665536 ms
versus CU1 2.290656 ms after the first call, ratio 0.727100, on a shared
RTX PRO 6000. The complete warmed model step median is 10.712 ms; its first
step takes 6.739 s, including cold compilation. The merged packed Thompson
binding matches the current thermal conversion without changing arithmetic.
The whole Rust workspace and Python wheel build, both complete packed
RK/history cases, checkpoint continuation and both nest feedback checks
pass. Exact commands, configurations, logs, scope limits and hashes are in
the current lane report and its skeptic-2 evidence.
