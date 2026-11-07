# Local rapid data assimilation

`woof local-da` authors one externally forced regional domain, reviews
its resource costs, and composes the existing preparation, ensemble cycle,
checkpoint analysis and short-forecast paths. There is no second forecast
integrator or observation decoder.

Each forecast member writes its actual terminal state, even when the
forecast ends between history times. The frame uses the same diagnostics,
inventory and renderer as ordinary history output. A completed older
forecast with an empty frame inventory is preserved; restarting that case
reuses its saved analysis and produces the short forecast in an output
recovery directory. Missing frames or failed rendering leave execution
status `FAILED`, with the product failure recorded for retry.

## Review, publish, launch

```
woof local-da --point 40,-100 --epoch 2026-09-10T12:00:00Z \
  --vram-gib 10 --host-gib 32 --budget-seconds 3600 --scale 1 --dry-run
```

The timestamp is an explicit example, not a latest-cycle resolver. It must
fall on the selected forcing source's initialization lattice. Review emits
one JSON document on stdout; diagnostics go to stderr. `--json` is accepted
and says so explicitly, since this door has no other output form. Every
field of that document, the ladder rows with their prices and verdicts, the
observation streams present and missing, the requested rung and its
advisories are written down once, in
[the companion protocol](local-da-companion-protocol.md), which
`woof local-da --capabilities` names and whose field roster it publishes. Review does not
fetch observations, repair native libraries, write a run directory or open
a device. The installed dependencies and source/physics tables still have
to be present for the canonical forecast estimator to operate.

Replace `--dry-run` with `--out local-cycle` to publish the reviewed
configuration. Inspect the returned review before executing:

```
woof local-da --launch local-cycle/local-da.json
```

Repeating that command uses the existing cycle and ensemble recovery
contracts. `--launch local-cycle/local-da.json --dry-run` validates and
reads the saved review without starting it. A generated directory contains
`experiment.toml`, `ensemble.toml`, `experiment.namelist.wps`,
`local-da.json`, and every other file the configuration's input route
reads beside it: on the native regional route that is the two
namelists and the target-domain document the run reads, rendered from
the published configuration and listed in `local-da.json` under
`files` with the rest.
Their hashes, and any explicit observation input hashes, are checked again
before launch. `--run` combines publication and execution for a caller
that has explicitly requested both actions.

`--region west,south,east,north` replaces `--point`. East less than west
means the short arc across the dateline. The perimeter must fit the
projected domain with its boundary margin. The author never silently moves
the requested point or crops the requested region.

## Continuous cycling

`--continuous N` reviews and publishes the same configuration and runs it
as N consecutive analysis windows instead of one finite cycle:

```
woof local-da --point 35.2,-97.4 --epoch 2026-09-10T12:00:00Z \
  --vram-gib 16 --forecast-seconds 1800 --continuous 3 --out local-continuous --run
```

Window `i` analyses at `epoch + (i + 1) * cadence`. Its forecast leg
restarts from the previous window's analysis, its observations are the
ones assigned to its own analysis time, and after the analysis the
reviewed short forecast runs and is rendered through the ordinary
`woof render` products. A window whose analysis time has not arrived is
waited for; a window that is late runs late and its receipt records the
lag. Nothing is skipped and no scientific setting is reduced to catch up.

The reviewed background covers the finite cycle. When a window's short
forecast reaches past that forcing, the window renews it first: the same
source cycle is prepared again over a longer window, keeping the initial
condition and every frame already run under, and the renewal is recorded
in `continuous/window_NNNNNN/forcing/renewal.json`. A review that binds a
prepared bundle or a local input root is not renewable and says so when a
window would need it; give such a review inputs that cover the whole
continuous window.

Three more commands address a saved continuous plan:

```
woof local-da --status local-continuous/local-da.json
woof local-da --stop local-continuous/local-da.json
woof local-da --launch local-continuous/local-da.json
```

`--status` prints `continuous/status.json` with the controller's liveness
decided by its held lock; `--stop` writes a durable stop request that the
controller honours between operations, or the next launch honours first;
`--launch` resumes an unfinished window through its committed decisions
and, once every window is committed, reports `COMPLETE`. The status
vocabulary, the window directory layout, the renewal receipt and the
`CONTINUOUS_WINDOW_FAILED` refusal are written down in
[the companion protocol](local-da-companion-protocol.md).

## The nowcast score

Every completed forecast is graded against the radar, with no flag. When a
window's short forecast finishes, the run scores each of the registered
leads -- 15, 30, 45 and 60 minutes after that window's analysis -- that the
forecast actually reached, against the MRMS composite scan nearest that
instant, and writes `nowcast-score.json` beside the window's own receipts.
`complete.json` names that receipt and carries the score as it stood at
completion; the plain single-window run puts the same summary in
`execution.json`; `--status` shows it for every completed window.

The instrument is the one the observation battery already uses: masked
neighbourhood fractions skill score, Roberts and Lean (2008), on the column
maximum of the forecast's own stored `REFL_10CM` against the composite, at
20, 30 and 40 dBZ in 9 km and 27 km neighbourhoods. The primary scalar is
FSS at 30 dBZ in the 9 km box. The scored region is the domain minus its
specified and relaxation rows minus a three-cell rim, intersected with the
validity of both scans and the model's finite cells, and the neighbourhood
boxcar treats outside the domain as unobserved.

**Every model number is published beside radar persistence.** The MRMS scan
nearest the analysis time is carried forward unchanged to each lead's valid
time and scored against that lead's scan, under the same thresholds, boxes
and masks, and the receipt carries the difference model minus persistence.
Inside the first hour persistence is the forecast to beat, and a model FSS
with nothing beside it cannot be read: 0.6 is a good number against a
persistence of 0.4 and a poor one against a persistence of 0.7. The carried
field is an observation, not a model run, so this is a baseline and not a
reference-model control.

**An empty box scores 1 and says so.** When no cell of the scored
interior reaches the primary threshold in the lead scan, the fractions
the score compares are zero everywhere and the convention gives FSS 1 to
the model and 1 to persistence, with a difference of 0. That is a clear
sky, not a skilful forecast, and every lead row carries
`primary_observed_base_rate` and `primary_model_base_rate` beside the
three numbers so a reader can tell the two apart at a glance: an
observed base rate of 0.0 means there was nothing to forecast.

Latency is ordinary, not an error. At real time the 60 minute lead cannot be
scored until an hour after the analysis, so a lead whose valid time has not
arrived, or whose scan the archive has not published yet, is recorded
`pending` with its reason and is never a zero; a lead whose matching window
holds no frame at or above the coverage floor is `missing-obs`; a run with no
network records `unavailable` with the error and is otherwise untouched.
The continuous controller scores earlier windows' pending leads as later
windows complete, and

```
woof local-da --score local-continuous/local-da.json
```

scores every still-unscored lead of a saved case on demand, rewriting each
window receipt. The composites live under `nowcast-observations/` inside the
case, named by the archive object each came from, so a rescore refetches
nothing. A scoring problem never fails a run: the forecast ran, and this is a
number about it rather than a gate on it.

## The scale policy

Rung one has one forecast trajectory, one update, a 3 km domain with a
nominal 192 km span, and a 30 minute forecast. It uses eight independently
perturbed analysis states to prescribe a static covariance. Those states
are evaluated at the analysis instant only; they are not eight forecast
members. An observation innovation is evaluated against the actual
background, including for nonlinear operators. A zero innovation produces
an exact zero deterministic increment.

Above rung one, forecast membership grows as `2**(rung-1)`, bounded by the
ensemble owner's member limit. Spacing refines every third rung, the domain
span grows, and cadence derives from the cadence owner's reference
interval. The finite cycle count comes from the requested rung and is never
reduced by estimated cost. Review prices lower alternatives but preserves
the requested rung's region, resolution, members, cadence and duration.

The hierarchy is a stated policy, not a proof of optimal forecast skill.
A user may explicitly request any supported positive whole-second cadence.
Fractional seconds cannot be preserved by the analysis and forecast-output
timestamps and receive a named error instead of being floored. A cycle cost
above cadence is an advisory queue-lag projection, with fewer members,
coarser spacing and longer cadence listed as optional choices. Source horizon, geometry, missing inputs and inconsistent
clocks remain real refusal conditions.

Forecast members execute sequentially on one card. The analysis holds
multiple states simultaneously. Pricing separates those two memory peaks,
accounts for the full restart-field inventory and observation batches, and
bounds the localized solve workspace through its existing owner. The card's
compute multiplier is independent of its memory capacity. Timing uses the
existing forecast pace estimate plus a conservative extrapolation of the
recorded LETKF solve basis. Preparation has a stated allowance; network
latency is not bounded by it. Ten GiB fit tests inject forecast costs and
exercise the real analysis storage formula. They are not a measured device
admission certificate for the complete runtime. Estimated peaks above
declared or currently free memory are warnings, and actual allocation
failures remain operation failures with saved-state recovery guidance.

`--budget-seconds` is an advisory wall-time comparison target. It does not
change the finite cycle count or enforce a deadline. Execution records
measured per-cycle wall time and lag during this launch, excluding
preparation, and projects remaining finite work from the measured mean.
The chosen domain stays unchanged even when the run falls behind cadence.

## Observation paths and their limits

The regional registries supply radar coverage, native decoder identities
and surface networks. Missing streams are reported and skipped rather than
converted into synthetic observations. Available types enter the same
localized analysis, not independent increments added without accounting
for their combined covariance.

* Radar: the registered grid builder, per-radar radial velocity, the
  selected physics scheme's reflectivity operator, and clear-air products.
  Existing thinning, error, beam, positivity and paired-moment contracts
  remain in force. The reference geometry is a frozen initial field written
  and read through the existing native geometry paths.
* Surface: the existing nominal-time surface record supplies 2 m
  temperature and 10 m wind speed, evaluated against end-of-leg diagnostics.
  Vector components from neutral tables use earth-relative winds. Terrain
  elevation mismatch and absent diagnostics are counted rejections.
* Shared neutral tables: pressure-level temperature, wind and dewpoint,
  assigned-pressure satellite winds, and tangent-point refractivity use the
  engine's own table decoder and operators (`woof.obs`). Each member supplies
  its own full pressure column. There is no vertical extrapolation. The
  default supported surface measurement labels are explicit; station
  pressure, sea-level pressure and unsupported platform quantities are not
  silently mapped to another measurement.
* Satellite cloud water: existing phase-aware cloud-water-path grids can
  enter through their registered adapter. Automatic regional satellite and
  sector selection is not implemented. Infrared/microwave radiances are not
  enabled: they need a regional column interface, coefficient identities,
  cloud treatment, bias/error configuration and numerical validation.
  Brightness temperature is never treated as a point air temperature.

The shared stream registry can be reused when its package and native doors
are installed. Account-gated streams, undecoded payloads and unbounded
subscriptions are reported by name. It does not create another ingest
stack. Explicit inputs use repeatable `--obs-table`, `--radar-grid` and
`--satellite-grid` arguments.

Observation windows are immutable after their first successful publication.
Their receipt binds the review, time, grid and exact input bytes. Resume
reads those same bytes. Quality control records future measurements,
arrival/publication cutoffs, invalid errors, gross limits, background checks,
duplicates, revision selection, thinning and out-of-domain observations.

**Current timing limitation:** neutral rows must have arrived and been
published by the analysis valid time. A fresh download made after that time
is excluded when its receipt time is recorded, including historical replay.
No receipt timestamp is rewritten or removed. A separate explicit
acquisition cutoff/replay policy is still needed for routine delayed live
cycling and retrospective fetched tables. Missing arrival metadata is
counted as latency unverified, not evidence of real-time availability.
The legacy surface record carries nominal times, not complete original
report/arrival timing. The current window is one cadence wide; delayed
upper-air and satellite reports may therefore be absent.

## Scientific and operational qualification

The deterministic covariance is prescribed, not an evolved ensemble
covariance. Covariance samples use the existing perturbation generator,
paired hydrometeor treatment and CPU equation-of-state refresh. Surface
covariance uses a frozen-transfer tangent approximation around the actual
end-of-leg diagnostics. The prior boundary conditions are shared.

Multiplicative condensate perturbations cannot create a storm where all
samples are cloud-free. Cloud initialization, displacement-aware updates,
additive inflation, balanced insertion, bias correction, correlated-error
operators and 4D trajectory sampling are not implemented by this patch.
Forecast-skill evaluation is: every completed forecast carries the nowcast
score above, neighbourhood FSS against the MRMS composite at 15 to 60
minutes with the radar-persistence baseline beside it. That score grades
the forecast reflectivity field and nothing else; the analysis increments,
the surface fields and the accumulated precipitation are not scored by it.
Refractivity uses fixed reference heights, and
radar gridding/localization also uses initial geometry rather than updated
member geopotential. These approximations need sensitivity tests.

The default member count, samples, localization, perturbation amplitudes
and error choices are a starting policy, not settings tuned against the
nowcast score the runs now publish.
Cadence-derived inflation is applied to every enabled observation family.
No-observation cycles publish explicit zero increments and a forecast-only
receipt. There is no new persistent no-observation staleness budget in this
regional launcher.

Products reuse the existing Rust renderer and hash-verified per-member
output inventories. No ensemble probability or mean-product aggregation is
added. Prepared-cache startup, native radar acquisition, reference-file
creation, device execution, desktop execution and rendered products still
need end-to-end qualification. A partial preparation without its final proof
needs manual recovery; only completed preparation and published cycles have
the tested recovery path. The lock is per run directory, not a cross-run
physical-card reservation. Online ingest size and timing are not a hard
upper bound. None of these limits is a forecast-skill result; the nowcast
score is, and it is a score of one case's reflectivity at one to four leads,
not a claim about the configuration in general.
