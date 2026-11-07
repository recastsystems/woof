# WRF 4.7.1 diagnostic sources and replay tools

The resident NWP driver lives in `woof/core/nwp_diagnostics.py`. It produces
all nine native NWP history fields, samples helicity in the first RK stage,
holds native phy_prep pressure/density, and applies the native reset clock.
The pressure- and height-level resident drivers emit native auxhist23/22
fields from stock `&diags` and stream controls. AFWA and misc diagnostic drivers are implemented below; cloud and turbulence
remain held because valid native inputs trap unassigned native values. Traditional-fields belong to the
adjacent diagnostic lane. Similar output names do not substitute
for those native packages.

## Authority and licence

The authority is public WRF v4.7.1, commit
`f52c197ed39d12e087d02c50f412d90d418f6186`. `sources.json` records official
URLs, byte counts and SHA-256 hashes. `LICENSE-WRF.txt` is the release's
public-domain statement and stays with copied sources. `verify_sources.py`
checks the immutable licence first, then each source and local build input.
No competitor source or external diagnostic table is used. The hail grid is
generated from native source constants, not a separately licensed data table.

## Census

`census.py --output DIR` inventories 64 Registry controls and 44 generated
auxhist22/23 controls, with exact nondefault namelist probes, importer results,
native source lines and candidate writer shapes. Generated stream controls
come from the pinned `tools/gen_streams.c`. An imported selector and a matching
field name establish neither production nor native numeric equivalence.
The lane evidence preserves the original candidate census and final runtime
census separately.

## Strict native references

Build `build_levels.py --output DIR` and `build_nwp_windows.py --output DIR`
with gfortran. Build receipts record compiler arguments, original module and
adapter hashes and library hashes. The NWP adapter derives scheme constants
from native registries. Its Thompson dependency supplies `idx_bg1`; compared
PSD cases either supply GH volume or do not consume that index. Original
modules retain their bytes. Analytic PSD coverage is separate from real-input
replay and must never be described as an active-hail native forecast.

Instrument an owned full native WRF checkout with `instrument_levels.py`,
`instrument_nwp.py` and `instrument_prepare.py`, then compile `em_real` with
strict flags (`-O0 -ffp-contract=off -fno-fast-math`, at most eight make jobs).
The instrumenters add bounded binary IO without arithmetic changes.
`WOOF_DIAG_CAPTURE_DIR` selects an empty owned capture directory.
`instrument_input.py` adds an initial-time `write_input` dump to native
mediation IO for native-initialized child inputs, without changing physics.
Each instrumentation command writes original/after source hashes.

A successful full native run is the input authority. Decode bounded level
captures with `decode_levels.py`. Run these offline GPU replay tools under
`GPUWM_WRF_EXACT=1 GPUWM_NUMERICS=strict`:

- `replay_levels.py --library DIR/levels_ref.so --capture CAPTURE.npz --output RECEIPT.json`
- `replay_level_pressure.py --capture CAPTURE.npz --wrfinput REAL_WRFINPUT --output RECEIPT.json`
- `replay_native_nwp.py CAPTURES --run-receipt RUN.json --output RECEIPT.json`
- `replay_prepare.py CAPTURES --output RECEIPT.json`
- `replay_uh.py REAL_WRFRST --build OWN_BUILD_DIR --output RECEIPT.json`

The helicity reference extracts native `compute_diff_metrics` and
`cal_helicity` verbatim from the pinned 4.7.1 source. The latter has the same
source bytes as the existing 4.6.1 device port; this is independently checked.
Only the ABI wrapper and module dependencies are adapted. Padding outside
valid native bounds is zeroed, with regions and hashes recorded. This replay
has nonzero native helicity; the early bounded NWP calls do not.

The lane's real-input results contain 41,332 compared words and zero differing
words: PLD 3,584; ZLD 1,008; initial face pressure 800; NWP windows/PSD 768; phy_prep 12,672; helicity
22,500. The early NWP capture has zero graupel/hail/reflectivity/helicity but
nonzero wind and vertical velocity. Separate analytic PSD cases cover active
branches; the three-hour forecast has no graupel or hail. These limits matter.

## Operational evidence

The lane report `WRF-PORTS-2026-11-01/port-diag-nwp.md` records exact commands,
configs, receipts and code lines. One real stock-input `woof go --wrfinput DIR`
forecast and native WRF run span 2024-05-25 18–21 UTC, 150 by 150 by 49 levels,
3 km, MP8, YSU/Noah and legacy RRTMG. Rust maps were inspected at several hours
beside native output and genuine ASOS observations. The case tests weak ascent
and limited echo; it does not establish active-hail forecast skill.

Steady CUDA event means on the shared RTX 4090 are 0.0292 ms preparation,
0.0996 ms helicity, 0.3211 ms radar preparation/computation and 0.1953 ms
windows/PSD: 0.6452 ms total versus 1.3675 ms for the existing GSD package on
the same live columns (ratio 0.472). Event collection synchronizes only after
the whole forecast. First compilation calls are reported separately.

Two nest feedback modes, both domains' checkpoint resume, ordinary two-member
ensemble carrier/history lifecycles and a stock native restart are exercised.
The native restart proof uses `hailcast_opt=0`; the preexisting automatic
HAILCAST warm-start carrier gap and native negative-condensate admission limit
are named separately in the report. No packed NWP ensemble claim is made.

Pressure/height output has a separate complete native and GPU 18–21 UTC
forecast with seven pressure targets and seven height targets, including AGL
and MSL heights. Native output dimensions, units and stream clocks are retained;
calculation clocks and auxhist alarms remain separate. Both nest feedback modes,
checkpoint continuation, native restart, ordinary ensemble execution and forced
2x2 streaming are exercised. The private held pressures/bounds are rebuilt;
level output arrays are checkpointed but are recomputed on native restart,
as start_em does. DFI companion flags parse, but the existing complete-DFI
refusal still applies. No full DFI or packed ensemble qualification is claimed.

Thirty Rust level maps at 19, 20 and 21 UTC show matching thermal gradients,
moist/dry regions and 500-hPa height structure. Surface ASOS verification anchors
the underlying forecast; it does not establish independent aloft skill. Latest
whole-run variable-byte comparisons bind the inspected outputs to the retained
implementation. Whole-model equality to native WRF is not claimed; module replay
is bitwise. Initial PW comes from the actual input reference constants and the
native start_em recurrence, with 800 exact real-capture words. Held mass/face
pressure copies add 1,584 word checks; these repeat captured inputs and are
reported separately from the 41,332 comparisons above.

On the shared RTX 5090, the final live-forecast event means are 0.062816 ms
pressure interpolation, 0.018512 ms required pressure capture, and 0.042656 ms
height interpolation. The existing GSD diagnostic package, with additional
helicity bands and independent outputs, takes 0.660397 ms on the same live
columns. Ratios are 0.12315 for pressure plus capture and 0.06459 for height.
Pressure/height alone is 1.4726: both are new ports, not the preexisting scheme
comparator. Pressure must separately search hydrostatic mass and face pressure
and interpolate geopotential in log-pressure coordinates; height uses geometric
coordinates. The original pressure scan took 0.135221 ms; safe endpoint rejection
and fusing exact maxima into the required pressure capture remove repeated
column scans without assuming monotone inputs or targets. Strict mode retains
the native search. Timing collection reads events only after the whole forecast.
Failed graph-capture and memory-contention profiling attempts are not proof.

Default-off checks retain 205 existing CUDA units' source/options in fast and
strict modes, and compare actual prognostics and common history bytes.

## AFWA capture and replay

`instrument_afwa.py` and `instrument_afwa_extended.py` add bounded input/result
IO to unchanged pinned native expressions. Set `WOOF_AFWA_CAPTURE_DIR` to an
empty owned directory. The extended format holds twelve input volumes, eleven
surface inputs, five previous precipitation-type amounts and twenty result
planes. `--select-condensate-max` selects a 4x4 neighborhood of the greatest
absolute QR+QS+QG column within each native patch and writes its coordinates.
This changes capture selection only. It does not change a native physics value.

Run a strict complete native build and actual forecast first, then:

```
GPUWM_WRF_EXACT=1 GPUWM_NUMERICS=strict \
python tools/diag_wrf471_oracle/replay_afwa_extended.py \
  --run-receipt RUN.json --output PROOF.json
```

The replay verifies capture hashes. If a launcher exit code is unavailable,
an explicit completed-native receipt must instead carry a hash-verified log
and its `SUCCESS COMPLETE WRF` witness; the process exit code stays unknown.
No numerical tolerance or undefined-result exemption is used by this proof.
The full native 18–21 UTC run contributes 8,704 output words. A real 20Z native
restart with `override_restart_timers=.true.` and one-minute history contributes
3,584 more: 76 nonzero rain and 657 nonzero snow input words on the output
calls, 92 nonzero VIL, 88 RADARVIL and 41 composite-reflectivity result words.
Every compared word matches. Earlier base/thermal-only proof has a separate
1,536-word receipt and is not added to this latest total.

Per-package latest counts are base 2,816, thermal 1,536, precipitation type
3,840, VIL 1,024, radar 1,536 and visibility 1,536. The base count includes its
private step precipitation. The warm case has no frozen surface precipitation,
graupel, 18-dBZ echo tops or active dust. Native undefined cold-precipitation
cloud-top and model-top echo operands are identified in the CUDA source;
this real proof does not exercise those branches or claim an exemption.

The lane report carries the completed whole GPU forecast, both nest feedback
modes, checkpoint continuation, native restart, ordinary ensemble, forced tile
execution, default-off byte comparisons, Rust maps, real ASOS scores and stage
measurements. AFWA chemistry dust coupling and the severe/icing/cloud/turbulence/
buoyancy packages are not qualified.

`afwa_bad_data_check` is qualified separately against the pinned native
output-call predicate, including the literal simulation-first-step exception.
`instrument_afwa_check.py` and `replay_afwa_check.py` compare 32 real native
verdict words from 75,264 captured input words: zero differences and no
undefined exemption. U/V use held phy_prep values; theta is raw post-MP full
theta before heating limits. The device atomic status uses the production
HealthLedger. A complete native/GPU forecast, 18 current Rust maps reviewed
across three hours beside native/ASOS, both feedback modes, checkpoint/native
restart, ordinary ensemble and forced tiles pass. Same-card healthy-check on/off
histories are byte-identical. Live RTX 4090 means are 0.021867 ms for the check
and 0.025579 ms for AFWA reflectivity (three actual output calls). All real
captures are healthy; GPU threshold, fatal-path and raw-theta tests supplement
the real acceptance replay. See the lane report's exact commands and limits.

## Native precipitation accumulation

`instrument_precipitation.py` surrounds the unchanged pinned
`module_diag_misc.F:415-448` arithmetic with bounded IO. It selects the wettest
4x4 neighborhood in each MPI patch, on the first step and the calls immediately
before/on an hourly boundary. No precipitation, temperature or clock value is
modified. Build the complete strict native executable and run it with
`WOOF_PRECIP_CAPTURE_DIR` set to an empty owned directory, then run:

```
GPUWM_WRF_EXACT=1 GPUWM_NUMERICS=strict \
python tools/diag_wrf471_oracle/replay_precipitation.py \
  --run-receipt RUN.json --output PROOF.json
```

The actual 18–21 UTC forecast finishes 900 steps with exit code zero. Its 48
captures compare 2,304 result words, all bitwise equal, with no undefined-result
exemption. Each of `PREC_ACC_C`, `PREC_ACC_NC` and `SNOW_ACC_NC` contributes 768
words; 49 grid-scale rain results are nonzero, maximum 0.00974329 mm. The real
case has neither convective nor frozen surface precipitation. Separate device
edge checks exercise the exact FP32 reset, fractional restart origin and
conditional snow clamp; they are not counted as real-input words.

Stock `prec_acc_dt` uses this native entry. The separately pinned fork extension
`prec_acc_dt1` retains its previous implementation. A two-word split clock origin
belongs to the domain, survives GPU checkpoints, and is rebuilt from a native
restart's start instant. The additional fork graupel amount has no WRF 4.7.1
restart counterpart and starts at zero on native warm start.

The lane's receipts cover a complete fast GPU forecast and non-boundary
checkpoint continuation, both nest feedback modes, native warm start, ordinary
ensemble execution, forced host-store tiles, and unchanged common default-off
history words. Same-column deferred CUDA event means are 0.065725 ms for the
native entry and 0.069964 ms for the existing fork entry (ratio 0.9394); events
are read after the forecast. Eighteen Rust maps, three paired three-hour sheets
and actual ASOS verification accompany the report. Surface ASOS observations
contextualize the forecast; they do not establish precipitation skill.

Native water and energy buckets

`instrument_buckets.py` verifies the pinned public-domain source before adding
IO around the unchanged native bucket block. It records all eighteen actual
amounts and int32 counters before/after the first and hourly calls. A complete
strict 900-step native forecast has exit code zero and 48 bounded captures.

```
GPUWM_WRF_EXACT=1 GPUWM_NUMERICS=strict PYTHONPATH=. python \
  tools/diag_wrf471_oracle/replay_buckets.py \
  --run-receipt RUN.json --output PROOF.json
```

The production CUDA entry compares 27,648 real result words, zero differences
and no undefined-result exemptions: water 3,072, energy 24,576. Separate device
edge tests establish strict greater-than, one subtraction, NINT ties, fractional
restart origin and integer state without counting them as native input proof.
The native source is `upstream/module_diag_misc.F:296-431`; its radiation block
is inside `bucket_mm>0`. No floor/remainder algorithm is substituted.

The lane report records the complete GPU forecast, non-boundary checkpoint,
both nest feedback modes, actual native warm start, ordinary ensemble, host-store
tiles, default-off byte identity, six paired three-hour Rust map sheets and ASOS
context. RRTMG supplies all sixteen native radiation amounts; other active
radiation schemes remain refused for those missing carriers. The hourly stage
cost is measured and attributed to eighteen amounts plus eighteen counters;
raw live timing means, graph profile and the removed repeated-grid cost are all
reported rather than selecting only the faster measurement.


Native accumulated physics tendencies

`instrument_tendencies.py` verifies the public-domain pin before IO-only capture
around `module_diag_misc.F:229-291`. Fifty-six bounded captures from a complete
strict 900-step CU3 forecast record sixteen rates and amounts before/after each
selected call. Native inactive shallow rate arguments are ignored by the native
and CUDA IF guards; none is presented as a valid consumed input.

```
GPUWM_WRF_EXACT=1 GPUWM_NUMERICS=strict PYTHONPATH=. python \
  tools/diag_wrf471_oracle/replay_tendencies.py \
  --run-receipt RUN.json --output PROOF.json
```

The replay compares 716,800 real output words with zero differences and no
undefined-result exemption. The complete fast GPU forecast, native warm start,
both nest feedback modes, full checkpoint continuation, ordinary ensemble and
host-store streaming complete. All common history bytes are unchanged with the
option off. Nine paired three-hour Rust map sheets and actual ASOS were viewed;
the report quantifies whole-forecast CU3 trigger/rate differences and tiny MP
rounding differences without claiming whole-model bit identity. Deferred live
stage events and required-byte accounting explain the measured physics-cost
exception to the 1.2x target. Source/licence hashes, raw samples and exact commands
are retained beside the lane report.


AFWA icing (native QNC path)
-----------------------------

The real MP16/WDM6 strict native forecast is captured and replayed by
``instrument_icing.py`` and ``replay_icing.py``: 20,544 output words, zero
differences, no exemptions. Native implicit INTEGER cloud number conversion
overflow is witnessed on actual inputs and reproduced explicitly. Production
uses a voxel partition kernel followed by ordered column integration; all seven
history and two raw restart fields remain device resident. The lane report
records the three-hour native/GPU Rust-map review, default-off identity,
continuation, native warm start, both nest feedback modes, ensemble, streamed
tiles and measured physics-cost exception. This does not qualify icing for a
microphysics package without native F_QNC, or establish observed icing skill.

Noah-MP output amounts
---------------------

``instrument_noahmp.py`` surrounds the unchanged WRF 4.7.1 amount update.
``replay_noahmp.py`` replays all 58 amounts on CUDA from actual entries. The
source manifest also pins WRF's Noah-MP gitlink and its driver/Registry for
producer and restart semantics. Output level 3 retains actual solver fluxes;
other levels leave the existing producer path unchanged. Qualification and any
remaining foundation limits are stated in the lane report.


AFWA buoyancy
-------------

`instrument_buoy.py` records actual native surface and most-unstable parcel
entries. `replay_buoy.py` uses the production four-stage CUDA path and compares
2,688 output words from a complete strict forecast with zero differences and
no exemptions. The native allocated ghost level is excluded using a separately
captured actual active-depth witness. The native Kelvin lifted-index offset,
500 hPa source search and ordered parcel integration are retained. The lane
report records native/GPU/observation map review, off identity, checkpoint,
native warm start, both nest feedback modes, ensembles, streamed tiles and
the measured Newton-iteration physics-cost exception.

## Severe diagnostics and diagnostic printing

`replay_severe.py --captures CAPTURE_DIR --output PROOF.json` verifies the
interval maxima, native parcel results, neighbor tornado state, actual relative
humidity and instantaneous shear from a complete native WRF run.
`replay_diag_print.py` takes the same arguments and verifies all thirteen
actual local native reductions before MPI. With `--native-rsl RSL`, it also
checks each global mass-change maximum from all captured patch maxima. Both
require strict numerics.
The capture layout is `severe-print-layout.json`; the lane evidence contains
the IO-only instrumenter and full native build/run hashes. Default forecasts
use parallel reduction trees and defer formatting to existing output boundaries.
Native all-zero precipitation maximum coordinates are unassigned; the native
zero maximum value remains defined, and formatted coordinates are null.

The complete MP9 native forecast supplies 164,864 compared severe-output words,
zero differences. The fused two-level humidity path also matches all 164,864
words in strict replay. Diagnostic printing matches 420 words (416 native local
reductions and four global mass maxima), with zero differences. A separate
`replay_prepare.py CAPTURES --print-surface --output PROOF.json` matches 128
surface-pressure words. The pressure sample is held after RK and before
microphysics, matching `moist_physics_prep_em`; the pre-RK NWP sample is distinct.

One complete stock-input 900-step fast `woof go` forecast and its 150-step
continuation preserve all 279 history variables (35,769,738 continuation words)
and all continued time-series records. Both nest feedback modes, native warm
restart, ordinary ensembles and host-store streaming are exercised. The lane
report records 108 Rust weather maps, eighteen paired sheets, three station
scorecards, three station-dot maps and all 899 scalar diagnostic records.
The remaining fast/full-model pressure-tendency difference is explicit; default
parallel trees fed actual native pressures agree within 0.0000461 mb/3h.

Same-process alternating CUDA graphs reduce severe column cost 8.318% by
calculating relative humidity only at the two consumed levels. Full-forecast
4090 event medians total 0.061440 ms for severe versus 0.023552 ms for VIL/radar,
plus 0.096320 ms at each parcel-output alarm. The extra ordered column work,
height/pressure searches, humidity exponentials and neighbor-duration reads are
quantified in the lane report as a physics-cost exception. Printing totals
0.018272 ms, ratio 0.776 to VIL. All new fast stages are parallel CUDA kernels;
strict native FP32 left-fold reductions are restricted to proof mode.
