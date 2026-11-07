# BouLac qualification, 2026-11-01

The native module and licence are pinned in PIN.json. WRF distributes this
module under its public-domain statement. There are no external BouLac tables.
The source is byte-identical between WRF v4.6.1 and v4.7.1.

A strict WRF v4.7.1 run supplied 128 real columns at 49 levels from four MPI
tiles. The untouched module expressions were taped at their first invocation.
Under GPUWM_WRF_EXACT=1 the replay compared 81,664 float32 words: zero
changed words and zero native-undefined words. It grades five tendencies,
TKE, length, both diffusivities, four turbulent fluxes and PBLH. Rural DLG_BEP
is not read or written and is excluded. Native compiler flags were -O2,
-fno-fast-math, -ffp-contract=off and -fno-tree-vectorize, gfortran 13.3.
The separate generated-column deck passed seven tests, including a native
calm-wind NaN witness and a planted one-bit mutation that must fail.

Fast arithmetic remains the default. The production runtime before the speed pass was
4faa468c0967c372a9218b7c3cca28f34c3847b6. All 2,178 exported production files
matched the local manifest before execution. A complete run used:

    python -m woof go --wrfinput DIR --outdir OUT --run-stamp off --run-seconds 10800 --benchmark

The stock namelist selected bl_pbl_physics=8, sf_sfclay_physics=91,
sf_surface_physics=2, mp_physics=6, ra_lw_physics=4, ra_sw_physics=4,
hybrid_opt=2, diff_opt=1, km_opt=4. The grid was 300 x 300 x 49 mass levels,
3 km, dt=15 s, from 2026-10-16 09 through 12 UTC. All 720 steps completed
on the shared RTX 4090. The benchmark receipt gives a 108.712 ms median step,
147.653 ms mean, 131.096 s forecast phase and 137.573 s total. Shared-card
whole-run timings are not a controlled comparison against a different run.

All 13 audited history fields were finite in four frames. TKE, EL and
exchange coefficients have WRF's top face; native WU_TUR, WV_TUR, WT_TUR
and WQ_TUR have mass levels. The fluxes are native restart diagnostics and
are also exposed as additional history products. Checkpoint identity v3
retains them. The final eight-test GPU driver deck passed with nonzero fluxes,
spaced PBL calls, byte-identical restart continuation, BEP, both nest feedback
modes, packed columns and the original member executor. YSU default-off
state arrays matched the base: 185 arrays, 106,720 bytes, zero changed bytes;
NPZ metadata creation timestamps are excluded.

The speed pass profiles one real first-step atmosphere from that case,
evenly sampled across the domain: 4,096 columns at 49 levels. A separate
Nsight Systems 2026.3.2 run records one kernel launch per column-stage call,
zero device copies, zero allocations and zero stream synchronizations inside
the 25-call timed range. The sole event synchronization closes the timer.
Diagnostic clock64 instrumentation attributes 49.72% of thread cycles to
the five tendency solves, 19.53% to exchange/TKE and 27.28% to parcel searches.
These shares describe the instrumented kernel, not elapsed GPU stage time.

Default fast arithmetic now uses __fdividef in FDIV. Correctly rounded
float division had consumed time and registers in the serial recurrences.
The source equations, loops and physical branches are retained. On the same
RTX 5090, in one process with five warmups and 25 timed CUDA-event calls,
the old kernel takes 0.568680 ms and the corrected kernel 0.398158 ms;
MYJ takes 0.352052 ms. The ratio improves from 1.61533 to 1.13096, passing
the 1.2x bar. The earlier 1.57233 result used tiled native columns and another
compiler; it is superseded by this controlled same-input comparison. MYJ's
pressure is derived from the same atmosphere with neutral Eta coefficients,
as in the earlier timing; this is not a MYJ conformance replay.

Registers fall from 78 to 48. The occupancy API reports a theoretical
capacity of six versus ten 128-thread blocks per SM. The local frame is
7,696 bytes with NVRTC 13.4.59, below the retained 8,208-byte preflight
ceiling. Static PTX contains no float64 instructions. No achieved-occupancy,
cache-miss or dynamic-divergence counters are claimed. The strict branch's
compiled image is byte-identical before and after this change, SHA-256
0e75d5e16cdf4dc323e4db31eb8448a48a0d2c92de9dacfa45b09c2f3c6f2f7a.
The completed real-native replay remains 81,664 words with zero differences;
no fresh native capture is claimed. Seven generated strict replay tests and
all eight GPU driver tests pass after the correction. Evidence is retained
under speed-pass/ in the lane delivery, including the raw small Nsight report,
paired.json, profile-analysis.json, strict image receipts and test logs.
There is no longer a physics cost exception.

Native WRF uses the same hashed WRF-compatible shared input and boundary
files and physics selectors. Their exporter title is recorded; upstream
forcing provenance and real.exe provenance were unavailable. The final native
completion, Rust-rendered hourly map inspection and observation metrics are
recorded below.

Native WRF completed 720 steps through 2026-10-16_12:00:00, with SUCCESS COMPLETE WRF. Its executable hash matches the real-input replay build.

Rust maps were inspected beside native WRF and station markers at 10, 11 and 12 UTC. Broad temperature and PBL-depth patterns match, with no visible seams or isolated spikes. No PBLH observations were used. The following pre-speed-correction table gives station RMSE for WOOF / native, 30 paired stations per time.


| UTC | T2 K | 10 m wind m/s |
|---|---|---|
| 10 | 1.506467 / 1.507265 | 2.142513 / 2.142573 |
| 11 | 1.229785 / 1.228937 | 1.920349 / 1.924254 |
| 12 | 1.189292 / 1.188091 | 1.580207 / 1.594722 |

These observations are archived IEM airport reports with limited archive quality control. Rust verification used nearest reports within 15 minutes, bilinear grid samples and no elevation adjustment. Map markers use the most recent report in the preceding 15 minutes. This is one case, without a general skill claim.

The corrected production runtime 09877dc6efa9538083805dce17f8d5f5aefb898e
completed a fresh whole 720-step forecast through the same literal go command,
stock namelist and hashed input/boundary pair, with fast numerics. Its launch
receipt records clean source. The shared RTX 5090 run used six host-backed tiles
under memory pressure, so its whole-model time is not comparable to the earlier
resident 4090 cases: median step 308.017 ms, mean 337.243 ms, forecast phase
318.869 s, total 323.386 s. Initial and final health pass, recovery is NOT_NEEDED,
the four sampled CFL values are below 0.824, and 52 field audits cover
144,360,000 finite decoded values. Exact receipts are in speed-pass/operational/;
forecast-closure.json, collection.json and history.json retain the audit.

All six corrected T2/PBLH Rust maps at 10, 11 and 12 UTC were inspected at
original resolution beside retained native and previous WOOF maps. The cool
northern/western land, warm southeastern band and evolving PBL texture remain
aligned. No visible new seams or isolated spikes were found. Station markers
support the broad T2 pattern; there are no PBLH observations. Joined images,
source hashes and visual-inspection.json are in speed-pass/maps/.

Fresh explicit Rust requests use the same frozen observations and 30 in-grid
stations. These receipts are unpaired because native raw outputs were already
cleaned; the native scores in the earlier table remain the native reference.
Corrected T2 RMSE is 1.507472, 1.230773 and 1.188020 K at 10, 11 and 12 UTC;
wind RMSE is 2.142500, 1.919463 and 1.585715 m/s. The largest absolute RMSE
change from the previous WOOF run is 0.001272 K and 0.005508 m/s, respectively.
No skill advantage is inferred. Requests, logs, station-provenance.json and
forecast-closure.json retain the numbers and common-observation checks.

The corrected source also passes the seven strict generated tests and all
eight GPU driver tests (2.18 s and 217.94 s including cold compilation).
The driver-test process forces global strict numerics but non-exact BouLac;
the complete forecast above uses fast numerics. The background automatic
verification adapter reports an invalid verification time (2) and stays pending.
The explicit Rust requests completed; automatic certification is not claimed.

Two interface decisions remain explicit: finite zero drag replaces native
0/0 for calm zero-stress winds, and the driver diagnoses KPBL from BouLac
PBLH because native BouLac does not write KPBL. The real replay encountered
neither native-undefined word. Numerical conformance does not imply whole-model
bitwise agreement under fast arithmetic.

Not checked: a whole real ensemble forecast, a real multi-hour nested case,
a native WRF restart-file warm start or the entire acceptance battery. GPU
integration covers ensembles, nesting and serialized checkpoint continuation.
The registry maturity remains implemented-unverified. A third-pass repeat
closed the earlier loss of nested operational JSON during cleanup. The
unchanged production code completed another 720 steps through `woof go`.
All 2,178 production files matched the earlier execution manifest, and all
four entire history files were byte-identical to the files used for the
native map comparisons and station scores. The 52 repeated field audits
cover 144,360,000 decoded values, all finite. Stored-type inspection through
the Rust bridge confirmed float32 physics fields and int32 KPBL.

The collected run receipt reports PASS, initial and final health pass,
and stability recovery NOT_NEEDED. The four recorded CFL values are below
0.824. Nested import, progress, run and stability receipts are retained under
`evidence/port-bl8-boulac/third-pass/operational/` in the delivery folder.
This shared-card repeat took 70.879 s overall, with a 50.284 ms median step;
the different card load prevents a speedup claim. The stage-cost comparison
above remains the measurement of record. At 49 levels the native parcel
searches traverse 2,352 intervals per column; no early exit or approximate
replacement was introduced.

Automatic verification.json is pending because the private execution export
does not carry rw_verify. It is not claimed as a passed certification. The
separate Rust observation decoder, paired verifier and Rust maps above are
the completed observation evidence.

The full small receipt set and exact commands are delivered as
WRF-PORTS-2026-11-01/evidence/port-bl8-boulac, beside the lane report.
Raw native and GPU simulation outputs and captures were deleted after
recording their hashes, numbers and figures; per-file deletion inventories
record path and size. No shared prepared input was deleted.
