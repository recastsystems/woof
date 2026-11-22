# Hourly global ensembles

`woof hex ensemble` is the default ensemble path. It defaults to 20 members,
60 km and one hour. `--resolution-km 30`, `--members 80` (or larger) and
`--cycle-hours H` select the other mesh, member count and whole-hour cycle
length. Every hour is written, including the initial state and hour one.
The deterministic `forecast` door remains the numerical worker.

The input is a `gpuwm-hex.hourly-ensemble/v1` JSON manifest. It names one
registered global mesh, its grid/static/row files, a native MPAS template,
the cycle time, and distinct prepared member files with source descriptions.
All inputs must have the forecast mesh's cell/edge order, identical vertical
and reference states and the same valid time. Output is restored to the
native template's cell and edge order by persistent MPAS IDs. It is refused
when the IDs differ. A valid-time label is written into `xtime`.

Prepared GDAS/GFS EnKF members can supply 80 or more distinct inputs. Build
their manifest with `tools/ensemble_roster.py`. Public GEFS is also supported:
`ensemble-prepare` downloads the atmosphere's a/b products, calls the existing
Rust `met_intermediate` and `woof hex init`, and writes the manifest. GEFS has
30 perturbed members. That door refuses larger requests instead of repeating
members. The public GDAS EnKF listing checked for 2026-10-10 06Z contained
surface-only files, which cannot initialize atmospheric perturbations.

Prepare once under a shared folder (`prepared/` below), then reuse the manifest in
both timing arms and in cycling. `tools/ensemble_stage_mesh.py` stages a stock
60/30 km global mesh under a locked shared directory. Existing 30 km meshes
can be used directly. Run it on the machine that runs the ensemble.

```sh
# After preparing the manifest (on a shared host, launch it through the host's card queue).
woof hex ensemble \
  --manifest prepared/ensemble/ensemble-input.json \
  --out runs/ensemble/cycle-20261010T06
```

One independent process runs per slot. Each sees only its assigned card.
The existing hex footprint model provides the initial floor. On a new
configuration, the first real ensemble member runs alone and its sampled
peak prices the remaining residents with the existing model's margin. This
prevents the unmeasured 188-SM New Tiedtke configuration from overpacking:
the two-member smoke peaked at 65,783 MiB, above the old derived fit.
A calibration is reused only with matching mesh, levels, physics flags,
engine config hash, CuPy version, card hardware and cycle length, linked to
the original member receipt. Slots are spread over every granted card
before adding a second resident. The receipt
labels derived footprint models, records the grant, commands, elapsed time,
source/input hashes and one-second whole-card memory/utilization samples.
These are independent resident processes, not a new batched numerical
kernel. `--members-per-card N` can lower concurrency, but cannot exceed
admitted capacity. `--gpuwm-checkout PATH` selects the physics engine and
retains the normal version/hash checks.

Native export uses up to eight isolated CPU processes by default. NetCDF/HDF5
calls remain serial within each process. `--io-workers N` changes this count.

`--analysis PATH` recenters each prognostic field as analysis plus member
minus sample mean. The analysis must share order, reference state, vertical
grid and valid time. Negative moisture or nonpositive density/temperature/
surface pressure refuses rather than clipping anomalies. The unchanged hex
cold start rebuilds the coupled fields. This operation preserves anomalies
to the output precision; it does not demonstrate balance or DA skill.
An analysis from the 15 km control must be regridded onto this ensemble mesh
before this call. Cross-resolution analysis interpolation is not supplied
by this door. Carried ice fields are recentered when the analysis supplies
them, or inherited member by member when it does not. Available surface and
soil fields are also preserved instead of becoming the template member's.

`spread.f000.json`, `spread.f001.json` and later hours report the area-weighted
RMS of pointwise sample standard deviation (`ddof=1`) for each variable and
native model level. Cell fields use `areaCell`; edge wind uses `dcEdge*dvEdge`.
`spread-growth.json` gives the one-hour ratio for each level and counts levels
with growth. A zero initial spread has an undefined ratio. Growth is reported
as measured; not every variable/level is expected to grow.

Output files carry physical forecast variables in native MPAS order, with
the original template geometry and static/reference fields. They include
pressure, pressure_base, pressure_p and both reconstructed winds. Filenames
use MPAS's dot-separated clock; colons are interpreted by MPI-IO as filesystem
URI separators and caused an actual open failure in validation.

Member files follow the stock MPAS-JEDI state contract (`woof.hex.jedi_contract`).
`jedi_open_checked` stays false in a run of this distribution, which does not open
the states with MPAS-JEDI.

`tools/ensemble_recover_export.py RUN` recovers export from completed forecasts
after checking input hashes. It retains the prior error and reports forecast
and recovery times separately. It does not claim a continuous cycle time.

The JSON files, forecast logs and hash receipts are retained. Source GRIB,
intermediate and constructed vertical files are deleted after preparation
has recorded each init. Forecast histories and exported simulation data must
be disposed after their spread/JEDI consumption receipts are recorded.
Use `tools/ensemble_dispose.py RUN` only after the run's consumers have finished;
it records every removed file and size and retains the measurement receipts.

## Long ensembles: streamed members and frame products (hex/ensemble, 2026-10-12)

A 72 h ensemble at 30 km does not fit the hourly design: every member init is
6.2 GB, every hourly frame 1.2 GB, and the full-template JEDI member 8.0 GB.
Frame mode keeps the disk bounded and the cards busy:

- `prepare` (argv) on a manifest member builds its init just before its
  forecast; `--prepare-ahead K` overlaps the next K preparations with running
  members; each member is validated on arrival against `reference_init`
  (ids, zgrid, base state, clock, duplicate physical state).
- `--export-hours 6` turns each durable frame into products while the member
  runs: an MPAS-JEDI ensemble-stream member (exactly the fields of
  `stream_list.atmosphere.ensemble`, native order, CDF-5; 1.60 GB at 30 km),
  `--extract-levels` pressure-level/MSLP fields (`--extract-grid-deg 0.5`
  regrids them, 17.5 MB per frame), then `--discard-history` deletes the frame.
- `--discard-inputs` drops a streamed init after its member passes (the driver re-verifies the init after the last step).
- `--members-per-card 1` runs no solo calibration pilot.
- `--deadline-seconds` launches no member that cannot finish in time, so a run
  fits a mutex backfill grant; unlaunched members are listed in the receipt.

Verified: MPAS-JEDI 4.0.1 `mpasjedi_convertstate.x` reads a slim member with
`stream name: ensemble` and returns theta, pressure_p, qv..qg,
surface_pressure and uReconstruct* byte-identical (geometry namelist at the
template's own valid time, state date at the member's).

Member sources: public GEFS members, prepared by `woof hex ensemble-prepare`.
Spread-skill against GFS analyses and IGRA2 radiosondes, beside the GEFS
members with the same identities: `tools/ensemble_spread_skill.py`.

## Perturbed-physics members

A manifest member may carry its own `forecast_flags` (an argv list, appended
after the manifest's). Any forecast-door physics flag is allowed
(`--convection kain-fritsch`, `--physics config_microp_scheme=...`,
`--physics-namelist FILE`); flags that would replace the member, its clock,
its output or its card are refused by name. Members with their own flags run
one per card (`--members-per-card 1`): the packing pilot prices a single
configuration. The receipt records each member's flags.

Why this exists (measured 2026-10-12, 7 members, 30 km, GEFS 2026-10-03 12Z
initial states, door-default physics for every member): the error of the
ensemble mean matches GEFS in the northern extratropics (z500 16.4 m against
15.8 m at 72 h) but the spread is 64 % of GEFS's there and 42 % in the
tropics, where it hardly grows (z500 1.6 m at 6 h, 2.0 m at 72 h). Initial
perturbations alone under-disperse; GEFS adds stochastic physics. Members
that differ in physics are the door's first source of model spread.

## Patch forecasts with one-card LTS

`--forecast-only` runs prepared native members without the JEDI covariance
export. It accepts a one-member pilot and any positive manifest resolution,
retains native histories after command closure, and refuses `--export-hours`,
`--discard-history` and analysis recentering. It does not arm rollback.

Active full-step LTS is supplied as ordinary manifest `forecast_flags`:
`--dt 18.75 --fullstep-local-timestep --fullstep-lts-mode active
--fullstep-lts-max-rate 8`. LTS requires `--members-per-card 1`. Each child
sees one granted CUDA token even when the parent has eight. Forecast flags
are parsed by the real forecast parser before launch. LTS event intervals
must satisfy the forecast driver's coarsest-class clock, so hourly output
is a case setting rather than a fixed-step frame interpolation.

`patch` is manifest data. A member may override `mesh`, `grid`, `static`,
`mesh_rows`, `native_template` and `reference_init`; paths are relative to
the manifest. Each member is validated against its own geometry and
reference state. Receipts record segment-zero asset hashes, device token,
whole-card memory sampled during that member, and subprocess wall time.
Passed LTS jobs additionally record the sum of this run's synchronized macro
walls, including the first macro. That integration measure excludes stack
construction, health checks and history IO. The one-card v1 executor does
not report a memory-pool peak, so that field stays null rather than being
inferred from card usage.

Moving segments still need a scheduler, conservative remap and restart
authority, segment clocks, and Rust remapping onto a common product grid.
The ensemble does not implement those operations. Frame-mode history
disposal waits for forecast command closure so the forecast door can count
its durable outputs; init disposal still waits for the driver's final
re-verification.
