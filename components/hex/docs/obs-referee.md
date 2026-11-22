# Observational referee

This subsystem answers a narrow question: **does a woof hex/WOOF-ArWen arm produce
better observable weather than its reference arm?** It does not revive
native-MPAS parity as a product goal, and it does not use a historical model as
an oracle. Observations are the primary referees: Stage-IV for precipitation,
MRMS for reflectivity, ASOS for the surface; historical native output and an
operational model may appear only as provenance-complete context arms.

The pinned integration base is:

```text
c14bc1045b0d4516f18004f13265d900ec12563b
```

## Scientific boundary

The implementation scores directly observable consequences:

- hourly precipitation and radar reflectivity;
- ASOS temperature, dewpoint, wind speed, and surface pressure;
- continuous error/bias, threshold contingency statistics, fractions skill
  score, and deterministic connected-object displacement;
- paired case-block confidence intervals, resampling whole meteorological cases.

**The precipitation instrument is Stage-IV, not MRMS.** The declared referee
names MRMS one-hour precipitation, and the shipped MRMS door decodes composite
reflectivity only: `rw_mrms decode` selects one canonical field at one altitude
and refuses anything whose units are not dBZ, so no QPE product reaches a pack
through it. NCEP/EMC Stage-IV hourly multi-sensor QPE is behind its own shipped
door, `rw_stage4`, on the 4.7625 km HRAP grid, and it is the substitution of
record: the manifest's precipitation metrics name `stage4` as their source and
carry `stage4-` ids, so no metric id claims an instrument that did not take the
measurement. Reflectivity remains MRMS, exactly as declared.

It deliberately does **not** claim that MRMS/ASOS can verify model theta above
level 45, cloud-water/rain-water partition, or the full GF causal chain. Those
claims remain `UNRESOLVED` until a secondary profile/process referee is supplied
(for example radiosondes or a provenance-pinned analysis profile).

## No second observation parser

Raw MRMS and Stage-IV GRIB2 and raw METAR/ASOS parsing remain a rustwx
boundary, and so does putting the unstructured model history on a structured
grid. The referee accepts only two normalized, checksummed contracts:

- `gpuwm-hex.canonical-grid/v1` (`.npz`);
- `gpuwm-hex.canonical-stations/v1` (canonical JSON Lines).

Every bundle requires a
`gpuwm-hex.normalized-artifact-receipt/v1` receipt that records its producer,
producer version, artifact name, and SHA-256. A production manifest cannot
allow a producer whose name begins with `synthetic`.

A source can optionally declare `producer_command`. The command is an argv
array, never a shell string, and may use only these placeholders:

```text
{output} {receipt} {case_id} {arm_id}
```

The producers that actually write these bundles are
[`../verification/producers/`](../verification/producers/README.md): they drive
the Rust doors through woof's own resolver and repack what the doors wrote.
Until 2026-08-25 the only thing that had ever written a canonical bundle was the
synthetic fixture, which a production manifest is forbidden to accept, so this
contract named a producer nobody could run.

This bridge lets the existing rustwx machinery materialize the canonical
artifact without coupling the referee to a guessed rustwx CLI. The producer
must create both the artifact and receipt; success without both is a refusal.

## Case and arm ownership

Dates and case identities live only in JSON manifests. The production manifest
carries four selected cases:

| case | role | cycle | window |
| --- | --- | --- | --- |
| `gfs-20260812-divergence` | known divergence | 2026-08-12 06Z | 24 h |
| `era5-20240521-divergence` | known divergence | 2024-05-21 12Z | 24 h |
| `gfs-20250714-independent-control` | independent control | 2025-07-14 12Z | 24 h |
| `gfs-20250114-weak-convection-control` | weak convection control | 2025-01-14 12Z | 24 h |

The two divergence cycles read 06Z and 12Z rather than the 00Z the first
edition of this manifest carried. 00Z was wrong: the initial conditions those
cases were measured from exist, and they carry those stamps.

The two controls were `selection_status: pending` with null dates until
2026-08-24, on the rule that code does not invent weather cases. They are
selected now, and not by taste either: the dates come out of a mechanical
screen of the observation archive, recorded in
`verification/manifests/` metadata and reproducible from
`screen_stage4.py`:

- the pool is the 15th of every month of 2025, Stage-IV 24 h accumulation
  ending 12Z, which is the only hour the archive publishes a 24 h object at;
- a coverage guard drops any day whose object covers fewer than 0.9 of the
  pool's median valid cells. It fired once, on 2025-05-15 at 72,825 cells
  against a 590,556 median, a River Forecast Centre outage, not a dry day,
  and its wet fractions are taken over a different denominator;
- the independent convective control is argmax heavy-rain coverage
  (fraction of valid cells at or above 25 mm): **2025-07-15 at 3.92 %**, with
  a 347.8 mm maximum;
- the weak-convection control is argmin any-rain coverage (fraction at or
  above 1 mm): **2025-01-15 at 4.68 %**, mean 0.29 mm.

Each case is then the 24 h forecast *ending* at the screened window, so the
cycles are the day before at 12Z.

A pending case still makes the scorecard `NOT_MEASURED`; there simply are none
left.

The production arm set is:

- `arwen-current`: pinned current behavior;
- `gf-subsidence-experiment`: optional isolated treatment;
- `native-history-context`: historical context, not a parity target;
- `operational-baseline`: optional external context.

## Optional GF treatment

woof hex does not own the upstream engine's GF tendency arithmetic, so this
patch does not guess a sibling file or function. Instead it enforces an external
receipt contract. An enabled experiment is scoreable only when its receipt
proves all of the following:

- treatment name and mode exactly match the manifest;
- scope is exactly `gf_subsidence_only`;
- the hook ran at least once and touched at least one column;
- pre/post tendency SHA-256 values differ;
- a full producer commit is recorded.

Disabled/default neutrality has two independent gates:

1. a disabled receipt must report zero calls, zero touched columns, and identical
   pre/post tendency hashes;
2. `compare-identity` hashes selected output trees and refuses any byte
   difference.

This establishes a safe integration seam without making the treatment
default-on. The scorecard always emits `DO_NOT_ENABLE`; explicit owner
acceptance is outside the automatic runner.

## What this build cannot be scored on, and why

One of the twelve registered metrics has no number at any case, and the cause
is named rather than reported as an absence:

| metric | side | cause |
| --- | --- | --- |
| `asos-pressure-rmse` | observation | `rw_asos` reports MSLP, a sea-level reduction, not station pressure. The two are not the same quantity, and mapping one onto the other would fabricate the reduction. |

The first run (2026-08-25) had four more unscorable metrics on the model side,
and closing them was its highest-value finding. Both gaps are closed in the
default history stream:

- `refl10cm` is computed inside the due step's own WSM6 call (post-call
  temperature, unchanged prepared pressure -- WRF's `diagflag` arrangement and
  the point where native MPAS-A computes the field) and published per history
  frame. `rw_mpas_convert` carries it as `REFL_10CM`; the model bundle's
  `reflectivity_dbz` is its column maximum, the MRMS-comparable composite.
  All three MRMS reflectivity metrics score against the model's own field,
  never a renderer diagnostic.
- `q2` is published bitwise (`q2_products_allowed = "true"`; native q2 itself
  carries occasional small negatives, and dewpoint consumers clamp at their
  own boundary). The model bundle's `dewpoint_k` is the engine's own
  dewpoint-from-mixing-ratio on `Q2`/`PSFC`, transcribed exactly from
  `rustwx-calc/src/derived.rs:653-658`, which scores `asos-dewpoint-rmse`.

A history stream produced before this publication is refused by the model
bundle producer by field name; re-run the forecast rather than repacking
around the hole.

## Deterministic workflow

From the root of this distribution. `verification/producers/` materializes the
bundles; see its README for the per-case invocations.

```bash
PYTHONPATH=src python tools/run_obs_referee.py validate \
  verification/manifests/obs-referee-283.production.json

OBS_REFEREE_DATA_ROOT=/absolute/case-root \
PYTHONPATH=src python tools/run_obs_referee.py run \
  verification/manifests/obs-referee-283.production.json \
  --output /absolute/out/obs-referee-real
```

`not-measured` writes an explicit unrun record instead, with no fabricated
values, for the case where the artifacts do not exist:

```bash
PYTHONPATH=src python tools/run_obs_referee.py not-measured \
  verification/manifests/obs-referee-283.production.json \
  --output out/obs-referee \
  --reason "why nothing could be measured"
```

No result file contains wall-clock time, random UUIDs, temporary paths, or
platform-specific JSON ordering. The synthetic end-to-end test runs the same
suite twice and requires every output file to have the same SHA-256; the
2026-08-25 production run was made twice for the same reason, against the same
bundles, and the two evidence directories agree file for file.

## Synthetic proof

The fixture script creates four tiny cases and all four arms, including an
enabled treatment receipt:

```bash
PYTHONPATH=src python verification/fixtures/build_synthetic_suite.py \
  --root /tmp/hex-obs-referee-synthetic --run
```

Its verdict is always `SYNTHETIC_ONLY`. It is a software test, not a skill
result and not evidence for changing model defaults.

## Refusal behavior

The runner refuses rather than silently degrading when:

- a required artifact or receipt is missing;
- a checksum, producer, schema, field shape, coordinate range, or monotonic time
  check fails;
- a production manifest admits synthetic producers;
- an enabled treatment receipt is absent or inconsistent;
- a raw/unknown adapter is requested;
- automatic default promotion is enabled;
- confidence claims have fewer paired cases than the manifest minimum.

An optional input can become an explicit per-metric `NOT_MEASURED` record.
Integrity errors are never downgraded to missing evidence.
