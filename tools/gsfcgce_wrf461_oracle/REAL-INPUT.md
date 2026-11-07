# MP97 real-input evidence, 2026-11-01

The forward port onto 2.9.12 runs WRF's GCE column and radar diagnostic
on CUDA. Normal forecasts keep the engine's fast arithmetic policy.
`GPUWM_WRF_EXACT=1` selects the strict proof arm. The real-input proof
includes the original `gsfcgce_hail=0`, `gsfcgce_2ice=0`,
`do_radar_ref=1` capture and the supplemental switch checks below.

Source pins, embedded tables and public-domain terms are in
[SOURCE.md](SOURCE.md). No competitor implementation was used.

## Native capture and replay

The native source is WRF 4.7.1 at
`f52c197ed39d12e087d02c50f412d90d418f6186`. Its GCE file is byte-identical
to the repository's 4.6.1 pin. `capture_real.py` instruments an owned copy
of that file. It records one midpoint row from each of four MPI tiles
at step 60, before and after the actual `gsfcgce` call. It changes no
physics operands. The instrumented source SHA256 is
`5dad2c9dbd952b0c7acbd6a7a568798610c35d9121c608a0fc811c4335ad96d5`.

The native executable was built with GCC Fortran 15.2.0, dmpar,
`-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`, eight build
jobs and four MPI ranks for the run. Executable SHA256:
`9fb7e3a57f73c413f53a9f041e435922d0435e2e76ac021046f226d5daa84b12`.

`replay_real.py` compares raw uint32 words for theta, all six moisture
fields, three precipitation accumulators, their increments, snow ratio
and reflectivity. Four calls contain 194 columns, 44 levels and dt=180 s.
All **69,646 output words match**, with zero differing words and zero
nonfinite outputs. [real-input-receipt.json](real-input-receipt.json)
records every input/output hash, field count and strict compile contract.

Typical reproduction, with paths to owned native source and captured files:

```sh
python tools/gsfcgce_wrf461_oracle/capture_real.py "$WRF_SOURCE"
# Build and run this source with the strict flags above and the recorded case.
GPUWM_WRF_EXACT=1 python tools/gsfcgce_wrf461_oracle/replay_real.py \
  "$CAPTURES" real-input-receipt.json
```

The radar correction keeps the native REAL exponent, float32 complex
water-index square root and REAL material constants. GCE owns this
backscatter entry; the shared `kernels/refl.cu` is unchanged. The
sedimentation optimization saves each scanned terminal velocity into one
reused device array, as native WRF does, and uses it for the flux pass.
No per-step host synchronization was added.

## Whole forecast and visual inspection

The initial and boundary files come from the public NCAR tutorial:
<https://www2.mmm.ucar.edu/wrf/TUTORIAL_DATA/severe_outbreak_2023_output.tar.gz>.
The single 30 km domain has 99 by 99 by 44 mass cells. The run spans
2023-03-31 00 through 03 UTC, with dt=180 s and hourly history. Physics:
MP97, PBL2, surface layer2, land surface2, cumulus6 and legacy radiation4/4.
All six analyzed masses and their lateral fields were present. No active
carrier was synthesized. Inactive QNICE/QNRAIN fields inherited from the
source scheme were removed from this owned GCE input copy.

`prepare_cli.copy_inputs` copies fields through the Rust NetCDF bridge.
Its metadata helper preserves each native attribute type, including the
integer attributes read by the native driver. Build it before preparing
a reproduction case:

```sh
rustc -O tools/gsfcgce_wrf461_oracle/restore_attributes.rs \
  -o restore_attributes -l netcdf
export GPUWM_GSFCGCE_ATTRIBUTE_COPIER="$PWD/restore_attributes"
```

The imported route receipts two existing adaptations: fractional sea ice
is disabled for this Noah route, and the moist-theta flag resolves to
dry theta. The public tutorial supplies the native eta coordinates through
the real.exe directory route. These adaptations limit whole-model
equivalence. The module replay above uses actual native operands and has
none of these integration differences.

The forecast command was:

```sh
GPUWM_WRF_EXACT=0 python -m woof go prepared-real-case \
  --outdir gpu-real-forecast --products none --no-supervise \
  --no-verify-visuals --run-stamp off
```

The lane's exact pool command and bridge paths are in
`gpu-real-forecast-final-manifest.json`. The successful log reports
`SUCCESS COMPLETE SIMULATION, 60 steps, 14.7 wall seconds`. Forecast timing
includes initial compilation and history IO; it is not an isolated speed
comparison. The native run completed the same three forecast hours.

All weather panels were drawn through `woof.rustwx` using `rw_wrfbatch`.
The 01, 02 and 03 UTC sheets compare native WRF, GPU GCE and archived
observed radar. Both model forecasts place a broad northern precipitation
shield over the Upper Midwest and a southern convective area in similar
locations, with closely matching movement and shape. The GPU southern
core becomes slightly stronger by hour three. Both forecasts are smoother
and broader than the observed southern radar line. The inspection found
no missing map tiles or visible field corruption.

Observations are the IEM archived `composite_n0q` lowest-tilt mosaic, from
its raster-to-NetCDF service. The request's valid time was added as CF
metadata in Rust. For the renderer's 25-million-cell limit, a Rust reader
selected every second pixel in both axes without averaging. The sheets
retain each source's grid. Model vertical-maximum reflectivity and
observed lowest-tilt reflectivity are different quantities. This inspection
does not establish an observation skill score or bitwise forecast equality.

## Runtime, lifecycle and defaults

On a shared RTX 5090, one captured 49-column row repeated over 200 rows
gives the same 9,800-column workload to each scheme. Five warmed CUDA-event
samples restore the original inputs outside each interval. Fast-mode
medians are 1.1232 ms for the GCE column and 2.8193 ms for WSM6, a ratio
of 0.3984. Radar medians are 9.9973 and 9.6235 ms, a ratio of 1.0388.
WSM6's temperature conversion is outside the interval. Shared GPU load
affects absolute times. [stage-times.json](stage-times.json) retains all
samples and scope; `benchmark_real.py` reproduces the stage comparison.

GPU lifecycle checks in `tests/test_gsfcgce_lifecycle_gpu.py` exercise:

- A 12-step full-physics continuation versus a 5-step checkpoint and
  7-step resume, with nonzero six-species state and retained heating.
  Every serialized array matches exactly.
- Parent-to-child six-species forcing and child-to-parent feedback with
  both feedback modes. Mixed-scheme GCE edges retain a named refusal.
- Two independent ensemble members through the member physics executor,
  compared word for word with individual calls, plus analysis reflectivity
  that leaves all prognostic carriers unchanged.

These are component lifecycle checks. A separate full nested real forecast,
a production ensemble forecast and a packed ensemble execution have not
been run in this lane.

With MP97 off, a fast 12-step full-physics MP6 check against candidate
`4123f1916e3` matches all 95 stored history fields (9,597 values) and all
179 serialized arrays (24,376 values) byte for byte. Checkpoint metadata
changes for the newly declared configuration and setup identity; this is
not a claim that the checkpoint archive itself is byte-identical.
All 205 pre-existing CUDA translation units retain identical assembled
source and compile options in both strict and fast modes.

The local development wheel builds with `WOOF_ALLOW_UNPINNED_WHEEL=1`.
The ordinary wheel build refuses this unpinned release tree as designed.
Nothing was published. Current Rust static-fields and NetCDF writer
libraries were built from the repository because the installed shared
bridges predated this candidate's ABI.

The option stays `implemented-unverified`: no quantitative matched
whole-model trajectory or observation skill gate is claimed. All eight switches also have the real-input module checks below.
Two witnessed native undefined reads are repaired by default.
The mixed-scheme nest closure remains explicitly unqualified.


## Supplemental switch and integration closure, 2026-11-01

An owned copy of an existing strict WRF 4.6.1 dmpar build was instrumented
at step 1 and rebuilt with at most eight jobs. The native run used two
MPI ranks on a real 2024-05-25 18 UTC, 3 km, 150 by 150 by 49 case, dt=12 s,
with hail=1 and ice2=0. Two actual GCE calls supply 148 columns each.
All 118,104 output words match strict CUDA, with no nonfinite output.
The first-step input has no condensate; the output forms cloud water and ice.

To cover nonzero frozen carriers, the same case was advanced to step 60
(12 simulated minutes). Capture selects the row with the greatest summed
rain plus graupel mixing ratio in each tile, without changing model operands.
The native source was rebuilt with two jobs and the native run used two ranks.
These two calls also match all 118,104 words. All six species are nonzero
in the first captured row; rain and graupel are traces below the fallout
threshold. The original three-hour default-switch replay remains the
precipitating-case proof. The diagnostic flag is false in both supplemental
actual-run captures; their radar output is unchanged native state.

The wet captured operands, unchanged byte for byte, were passed to the
standalone native module with radar enabled for each hail/ice2 pair.
This is a module control check on real inputs, not eight separate forecasts.
[switch-real-input-receipt.json](switch-real-input-receipt.json) retains
field-by-field counts, input/output hashes and native source identities.
All 944,832 output words match strict CUDA against the native arm with only
the two declared undefined-read repairs. The stock arm matches every word
for ice2=0 and ice2=2. The signaling-NaN native arm traps in ice2=1 radar
(uninitialized qg1d) and ice2=3 evaporation (uninitialized transport arrays),
for both hail values and both captured rows. The stock wet arm differs in
26,788 radar words for ice2=1 and 96 rain words for ice2=3; CUDA has no
nonfinite result. The deterministic native ice2=2 omission of graupel
fallout is preserved; the older port changed it without an undefined-read
witness. The first-step module controls separately match 944,832 words
against the same two-repair native arm.

The supplemental receipts and exact commands are in the delivered lane
evidence under `third-pass-*`. Fast paired CUDA-event timings cover all
eight controls on restored captured inputs. On the wet 9,916-column batch,
the maximum column ratio to WSM6 is 0.590 and maximum radar ratio is 1.055.
The original three-hour real forecast and its three inspected Rust map
sheets remain the whole-forecast proof; they use unchanged default switches.

The GCE adapter now obtains dry theta through `state.total_theta()` when
the integrated state carries moist theta. A CPU protocol regression checks
actual column-launch operands with nonzero water. This branch predates
the global moist-theta state implementation; it does not independently
claim that global feature. The change matches the integration branch fix.
Full atmospheric stepping additionally passes a parent/child schedule
with feedback=0 and feedback=1 (three parent and nine child steps each),
including a specified parent, nonzero boundary tendency and bound root clock.
Checkpoint continuation and independent ensemble-member GPU checks pass.
The stale forecast-dispatch source inspection and the new keyword-only
configuration-field expectations are corrected and pass.
