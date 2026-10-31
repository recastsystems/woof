# Configuration knobs (WRF namelist parity)

**Two ways in, both first-class.** Write the experiment TOML yourself --
`[shared]` and `[[domain]]` take the WRF namelist keys verbatim, same
spelling, same meaning, per-domain where WRF is per-domain -- or hand
`woof import-namelist WPS INPUT` a WRF `namelist.wps` +
`namelist.input` pair and it writes one for you. A shipped physics
profile is a third and shorter road to the same loader: a name for
switches you could have typed, never a gate on what you may write
([PHYSICS.md](PHYSICS.md)).

WOOF's configuration surface is the experiment TOML
(`[experiment]` / `[projection]` / `[shared]` / `[[domain]]` /
`[case_data]` / `[perturbation]` / `[tiles]` / `[output]`), and the
import contract is *never silent*: every namelist key lands in exactly
one of three report sections --

- **translated** -- a TOML value came out of it (including the one
  ratified physics substitution, RRTMG to RTE+RRTMGP);
- **fixed by WOOF** -- WRF has options there, WOOF implements exactly
  one value; the key is validated against that value (anything else is
  a hard error) and the report records the pin and why;
- **not implemented** -- consumed without a counterpart, each with a
  reason.

A setting that changes the forecast maps onto an implemented equivalent or
is refused with the breakage named ("What a WRF namelist setting becomes",
at the end of this page). Machinery WOOF does not run (`&dfi_control`,
`&logging`, the operational fork's `gsd_diagnostics` and history-cadence
switch, `prec_acc_dt`, real.exe vertical-interpolation choices)
is inert while its selector is off. Requested, the digital filter
(`dfi_opt`) is refused by name; the
others are a declared divergence printed under "Physics substitutions",
limited to the initial placement or the output files.

This page is the complete knob table. "Default" is the value a key
takes when omitted from the TOML (the frozen `RunConfig` default,
`woof/config.py`); where WRF's Registry default differs, the importer
emits the WRF value explicitly so an imported experiment resolves the
way WRF would. Physics *selection* values and their maturity labels
live in [PHYSICS.md](PHYSICS.md); this page covers the knobs around
them.

**Matching names does not match the whole WRF configuration.** The defaults for
`top_lid`, `emdiv`, `hypsometric_opt` and `h_sca_adv_order` differ from WRF's
Registry defaults, as the rows below state. `moist_cq` now defaults to `true`;
explicit `false` omits the moist correction and is a comparison counterfactual.
A TOML file that relies on differing defaults is outside a WRF comparison unless
that exact configuration was measured. The published WRF validation record does
not transfer to an untested configuration. Importing a namelist makes the mapped
values and declared substitutions explicit; it does not itself establish
statistical equivalence or forecast accuracy.

## `[output]` -- which variables the wrfout files carry

WRF's `iofields_filename`, as a selection over the inventory the run
already produces. `preset = "full"` (the default) writes everything;
`"minimal"` and `"severe"` are named sets; `history_vars` /
`history_drop` are explicit include and exclude lists, mutually
exclusive. The same table sits inline on a `[[domain]]` as
`output = { ... }` and overrides the tree-wide one for that domain, as
`[tiles]` does. It changes no number the model computes and does not
enter the restart identity, so a trimmed run resumes a full run's
checkpoints. Full page:
[OUTPUT-VARIABLES.md](OUTPUT-VARIABLES.md).

## GRIB2 output

`woof export-grib2 RUN_DIR --out EXPORT_DIR --zip` turns saved history
into `wrfsfc_dNN_<valid>.grib2` and `wrfprs_dNN_<valid>.grib2` per domain
and time, plus `manifest.json`. Every product comes from the WOOF
post-processor (catalog `woof-post/v1`), computed on a GPU when one has room
and on the CPU otherwise; both give the same bytes. `--post-device gpu|cpu`
forces a device. `--domains`, `--times`, `--start`, `--end`, `--fields`
and `--levels` choose what is written, `--packing simple` replaces the
default complex packing, and `--winds earth` rotates wind components to
east/north. [GRIB2 export](../grib2-export.md) lists the fields.

In a run-plan document `run_options.grib2 = true` exports each history
frame while the forecast runs (off by default); `woof go CONFIG.toml
--grib2` sets the same option. Files land in `<run>/grib2/` and
`<run>/grib2-grib2.zip` follows when the forecast ends. A restarted
forecast exports into its own attempt folder. `woof render HISTORY...
--grib2-out EXPORT_DIR` exports the frames a render draws.

## `[case_data]` -- the inputs a config-driven run declares

`woof run CONFIG.toml` is the config-driven route, and `[case_data]` is
where that config names the data it runs on. Every input path and every
policy is **declared, never implicit**: the loader
(`woof/case_data.py`) refuses an unknown key and refuses a missing
required one, listing all of them at once rather than the first it
reached.

`woof domain` and `woof import-namelist` write this table for you. The
table below is for hand-authoring it, and for reading what the wizard
wrote.

**Required -- the run is refused without them.**

| key | what it names |
|---|---|
| `forcing` | the meteorological input: one path, a list of paths, or a glob. A glob that matches nothing is refused, except under `woof static`, which never opens it. |
| `vtable` | the WPS Vtable selector authority for `forcing`. Also unread by `woof static`. |
| `wps_namelist` | the `namelist.wps` whose geometry the domains are built on. |
| `geog_root` | the WPS_GEOG static geography tree ([DATA.md](DATA.md)). |
| `sfcp_to_sfcp` | real-init surface-pressure policy: a knob, and `false` is a legitimate answer, so it is declared rather than defaulted. |
| `output_title` | the `TITLE` global attribute written into every wrfout this config produces. It is required because it is provenance: a history file that cannot say which experiment wrote it is not reproducible, and there is no sensible default for the name of your own run. |

**Optional.**

| key | what it names |
|---|---|
| `forcing_interval_s` | forcing cadence, when it should not be discovered from the files. |
| `output_domain` | which domain's history the run publishes. |
| `source_orography` | a NetCDF file supplying source orography; without it the forcing catalog's invariant geopotential is used. |
| `source_orography_variable` | the variable to read out of `source_orography`. |
| `co2_vmr` | Positive COâ‚‚ mole fraction consumed by the selected classic RRTM, legacy RRTMG or RRTMGP absorption; e.g. `0.000420` for 420 ppm. Off, analytic and Dudhia-only radiation retain it as inactive input authority. |
| `water_temperature_overlay` | a water-temperature overlay file ([water-temperature-overlay](../water-temperature-overlay.md)). |
| `water_temperature_policy` | how that overlay is applied. |
| `preprocess_backend` | where the root domain's preparation runs: `"cuda"`, `"cpu"` or `"auto"`. Absent is `auto`, which prepares on the CPU when the card reads busy or cannot hold the preparation, so two runs you compare could start from different preparations; naming it pins both. `woof run --preprocess-backend` overrides it. Nests prepare on the card either way. |

Per-domain source orography is declared as `d01`, `d02`, ... keys inside `[case_data.source_orography]`.

`woof run` on a config with **no** `[case_data]` table refuses by name
and lists the required inputs; that refusal and this table are the same
list.

## Tweakable knobs

### `[experiment]`

| TOML key | WRF equivalent | default | allowed | note |
|---|---|---|---|---|
| `diff_opt` | `diff_opt` | 2 | 1, 2, per domain | 1 selects model-coordinate horizontal diffusion with `km_opt = 2` or 4; 2 selects terrain-aware metric stress and scalar diffusion. |
| `mix_full_fields` | `mix_full_fields` | true | bool, per domain | WRF logical, either value. Under `diff_opt = 2` WRF's `false` branch subtracts the 1-D base-state profiles that `real.exe` leaves at zero, so a real-data run mixes identically at either value; `false` is the WRF Registry default and what operational HRRR runs (its namelist omits the key). An ideal.exe wrfinput carrying a nonzero base-state profile that WRF would subtract is refused by name: under `diff_opt = 2` with `false`, `U_BASE`/`V_BASE` (and, with no PBL scheme and `km_opt` other than 4, `T_BASE`/`QV_BASE`, which `vertical_diffusion_2` subtracts); under `diff_opt = 1`, `U_BASE`/`V_BASE` only with `km_opt = 3` and `false` (WRF forms the shear for `diff_opt` 1 or 2, `module_first_rk_step_part2.F:448`, but `tke_rhs` runs only under `diff_opt = 2`, `:888`), plus `U_BASE`/`V_BASE`/`QV_BASE` at either value with no PBL scheme and `kvdif` above 0. Measured on compiled WRF `cal_deform_and_div` (byte-identical in operational HRRR's WRFV3.9 fork and v4.7.1): with zero profiles the two values differ in 7 of 966,854 tensor words, all signed zeros. Under `diff_opt = 1` the coordinate operator mixes theta relative to its initial field for either value. |
| `name` | -- | required | non-empty string | run identity |
| `start_time` | `&time_control start_*` | required | TOML datetime, offset-free | |
| `run_seconds` | `run_days/hours/minutes/seconds` or `end_*` | required | > 0 | |
| `restart_interval_s` | `restart_interval` (minutes in WRF) | required | >= 0; 0 disables | whole multiple of d01 dt |
| `feedback` | `feedback` | 0 | **0 or 1** | 0 is one-way and supported; 1 is the EXPERIMENTAL two-way path (stamped as such in run provenance; one-way consumers refuse a feedback-modified parent). Any other value is refused, and a two-way namelist is never quietly imported one-way |
| `smooth_option` | `smooth_option` | 0 | **0 only** | the parent smoother acts only under two-way feedback |
| `blend_width` | `blend_width` | 5 | >= 0 | terrain blend zone; enters the parent-row clearance rule |
| `spec_bdy_width` | `&bdy_control spec_bdy_width` | 5 | >= spec_zone + relax_zone | |
| `smooth_cg_topo` | `&domains smooth_cg_topo` | false | bool | WRF v4.7.1's d01 boundary terrain blend: the outer `spec_bdy_width + blend_width` rows of domain 1's terrain are blended toward the source model's own terrain, bit for bit as WRF's `blend_terrain` does, once at the first time. Needs the source's terrain (its SOILHGT); refused without it |
| `column_chunk` | -- | 3125 | >= 1 | WOOF-only radiation throughput knob; byte-identical across values |
| `physics_mode` | -- | absent (see note) | `"wrf-faithful"` or `"arwen-patched"` | WOOF-only axis selecting WRF-faithful code paths or registered patches. This describes faithfulness to WRF code, not accuracy against observations. Present, it becomes the author of every divergence-ledger key and writes the faithful or patched side of each edge onto every domain; an explicit occurrence of one of those keys in `[shared]` or `[[domain]]` is then refused rather than merged, because a key with two authors runs a value neither of them chose. ABSENT it authors nothing, which is what every configuration written before the axis means -- and the reported mode is still `wrf-faithful`, because no registered patch is applied. The register is PROVENANCE.md, "Divergence ledger v1"; the resolved vector lands in the run receipt |
| `patchset` | -- | `"v1"` | a registered patch-set version | Which frozen ledger set the axis resolves. A version is frozen when it is registered, so a receipt naming `v1` keeps meaning the vector it meant; later entries get a new version beside it |
| `patches` | -- | the whole set | array of ledger entry ids, e.g. `["L4"]` | The single-patch ablation arms. Only under `physics_mode = "arwen-patched"` -- a subset of the patches APPLIED is meaningless when none is. An entry the ledger holds back (SASE's entry gate; the dormant class-C rows) is refused with the gate as the reason |

### `[projection]` and WPS geometry

`map_proj` is one of `"lambert"` (Lambert conformal, either
hemisphere), `"mercator"`, or `"polar"` (polar stereographic, either
pole), with `ref_lat`, `ref_lon`, `truelat1`, `truelat2`, `stand_lon`
-- the WPS `&geogrid` set, all required, all inside the experiment
fingerprint (Mercator and polar consume `truelat1`; `truelat2`
mirrors it). All three projections are transcription-gated at
binary64 against the pinned WRF v4.6.1 `share/module_llxy.F` oracle
(`tests/test_projection_oracle.py`), but their maturity differs:
northern-hemisphere Lambert carries the historical matched WOOF-versus-WRF
comparison, a code-verification check. Mercator, polar stereographic and
southern-hemisphere Lambert have the binary64 projection oracle and finite-state
smoke runs, with no matched-run comparison. A smoke run checks execution and
finiteness, not numerical accuracy. See the worldwide section of
[VERIFICATION.md](VERIFICATION.md) and the projection
maturity rows in [PHYSICS.md](PHYSICS.md). Latitude-longitude
(cylindrical) and rotated grids are refused, as are domains
containing or touching a pole and forcing footprints wider than 180
degrees of longitude. Domain layout (`e_we`/`e_sn`, `parent_id`,
`i/j_parent_start`, `parent_grid_ratio`, `parent_time_step_ratio`)
translates 1:1 with WRF's staggered-to-mass conversion (one fewer
point); child `dx`/`dt` are never hand-typed -- they derive exactly
from the parent chain and hand-typed values are cross-checked, not
copied.

### Vertical grid (`[shared]`)

| TOML key | WRF equivalent | default | allowed | note |
|---|---|---|---|---|
| `e_vert` / `nz` | `e_vert` | required (one of) | nz >= 4 | `nz = e_vert - 1` mass levels |
| `eta_levels` | `eta_levels` | () | 1.0 -> 0.0 strictly decreasing | explicit coordinates are required by config-driven native preparation; the `run --met-em` door generates omitted levels with WRF `auto_levels_opt` 1/2 and records the resolved controls; explicit `eta_levels` bypass generation |
| `p_top` | `p_top_requested` | 0.0 | >= 0, Pa | Registry default 5000 Pa applies on import when omitted |
| `hybrid_opt` | `hybrid_opt` | 0 (legacy) | 0/1 (sigma), 2 (WRF cubic-B) | importer default 2 (Registry) |
| `etac` | `etac` | 0.2 | [0, 1] | the value asked for; preparation lowers it when this run's terrain needs it (below) |
| `ztop` | -- | required | > 0 | WOOF-only scaffold height; real runs derive heights from p_top/eta_levels |

#### The coordinate a run's terrain can order

At `hybrid_opt = 2` the WRF cubic `B(eta)` can only order a column while
its surface pressure stays above a floor set by `etac` and `p_top`; WRF
itself calls a column below that floor fatal
(`dyn_em/nest_init_utils.F:1158-1182`, "tends to be caused by very high
topography", remedy "reduce etac"). At the shipped `etac = 0.2` with
`p_top = 10000` Pa that floor is 46408 Pa, about 6082 m of terrain.

Preparation does not stop there. It surveys every terrain field the run
can touch -- each declared domain's static terrain at its own resolution
(a nest carries higher peaks than its parent), and for a following nest
the whole statics corridor it may traverse -- and sets `etac` to the
largest value that orders the lowest surface pressure it found. `p_top`
is never changed: `etac` is WRF's own named remedy, and it keeps the
model top where you put it.

When the derived value differs from the configured one, preparation says
so in one line naming the column, its height and the value chosen, and
`proof.json` carries a `vertical_coordinate` block with the same numbers
plus every terrain field surveyed. The forecast then runs on the
coordinate the prepared inputs carry, not on the configured one: the
prepared caches hold the coefficient arrays themselves, as WRF's
`wrfinput` holds `C3H`/`C4H`, and the runner adopts the value beside them
after checking that it is at or below the one you configured and that it
orders every prepared column.

Every domain of the run takes that one coordinate, not the root alone: a
nest on the streamed road rebuilds its tile buffer from it, a nest
spawned mid-run is priced on it, and an offline child reads it off the
history's `ETAC` attribute. A domain tree that somehow held two is
refused by name rather than integrated.

Terrain that no positive `etac` orders is still refused, with the
constraint, the column and the remaining remedy (a lower model top).

A preparation with nothing to derive writes the block anyway, at status
`NOT_APPLICABLE`, and its `why` names which of the three reasons applied:
the configuration carries no eta ladder, so no terrain was surveyed;
`hybrid_opt` is 0 or 1, where `B(eta) = eta` leaves no terrain ceiling to
derive; or `hybrid_opt` is 2 over an eta ladder with no segment where
`dB/deta` rises above 1, which the ladder and `etac` set between them,
leaving the surface-pressure floor at zero so no ground can reach it.

### Clock (`[[domain]]`, root)

`time_step` (integer seconds) plus optional `time_step_fract_num` /
`time_step_fract_den` -- WRF's exact rational clock keys. Children
carry no clock: `dt_child = dt_parent / parent_time_step_ratio`
exactly, chained in float32 exactly as WRF chains it.
`history_interval_s` is per-domain and must divide into whole domain
steps. `history_begin_s` / `history_end_s` (per-domain, WRF's
`history_begin_*` / `history_end_*`) window the frames: the first frame is
written at `history_begin_s` after the start, rounded up to the domain's
step as WRF's alarm rings, and none after `history_end_s`; absent, every
interval from the start is written. With the adaptive time step on (`use_adaptive_time_step` in
`[shared]`, `docs/ADAPTIVE-TIMESTEP.md`) each domain's step instead follows
its own measured Courant number between `min_time_step` and
`max_time_step`, landing on every output time.

Steep ground under a strong wind at crest height takes a shorter step.
Before a forecast starts, each domain is read for its steepest slope, its
highest ground and the strongest wind its start state and boundary data
carry over the forecast window from the ground up to that height. Where
the engine's measured stability map (`woof/terrain_clock.py`,
`woof/terrain_clock_map.json`) says the configured step and substep count
do not hold there, the domain runs the smallest whole division of its step
the map holds, with the fewest substeps (the configured count or 6) that
hold it, prints one line saying so and records it under `terrain_clock` in
the run report. A nest's step ratio grows by whatever division its parent
does not already give it, so every cadence stays a whole number of steps.
Under the adaptive clock the held step caps `max_time_step`, and a count
the rule raises becomes `min_time_step_sound`, the fewest substeps that
clock takes whatever step it adapts to. A domain the map holds runs
exactly as configured. The map was measured with fixed steps up to 5 s
per km of grid spacing (15 s at 3 km), and at 3 km up to 40 s: rows of
crests up to 4.5 km and slopes up to 0.4 were tried from 40 s down to 20 s
under 20 to 90 m/s on four and six substeps, and every step that held was
run again for three hours. Where a longer step stopped, the step that held
caps an adaptive `max_time_step` and divides a fixed step above it. A
domain is read against every mapped row as gentle as its ground or
gentler, at every wind up to its own, and a stop seen on any of them
counts: 3 km ground steeper than 0.4, itself measured only to 15 s, is
capped by the stops its gentler rows saw at 20 s, winds above 90 m/s by
those seen at 90 m/s, and a spacing between 2 and 3 km by its 3 km rows.
3 km crests above 4.5 km were measured only to 15 s, so a longer step
seen to stop under the 4.5 km and lower crests, at the same slope and
wind, caps them too: a 5.5 km crest of slope 0.36 under 30 m/s is capped
at 15 s as a 3.9 km one is. Only where nothing read saw a longer step
stop is a longer `max_time_step` left to the adaptive clock's own
vertical and horizontal CFL limits. A cap only ever shortens the step:
one at or above the adaptive clock's own longest step (your
`max_time_step`, or 8 s per km of grid spacing when it is -1, 24 s at
3 km) is not written. Those stops were seen on fixed steps, so the
four-substep rows at 2 and 3 km of crests up to 4.5 km and slopes up to
0.4 (under a 4.5 km crest, to 0.47 at 3 km and 0.48 at 2 km) were also run on the adaptive clock,
for three hours at `max_time_step` from 15 down to 5 s per km, 20 to 60 m/s.
Where every row a domain reads held the longest step tried there, and its
ground is no steeper than the steepest row at its crest, its adaptive clock
is capped at that step (45 s at 3 km, 30 s at 2 km) instead of the fixed-step
stop, unless your `target_cfl`, `target_hcfl` or `max_step_increase_pct` is above what was measured (1.4, 0.98 and 5).
Where a row saw a longer adaptive step stop, the step it held caps the
clock whatever the fixed rows say, and where a row held none the line
says the domain may still stop. A fixed step reads only the fixed-step
rows. On a 3 km CONUS domain (highest ground about 3.9 km, steepest
slope 0.36), the 24 s `max_time_step` the adaptive clock takes by default
stands under a crest-level wind of 20 and 30 m/s, since every row read
held 45 s on the adaptive clock there; at 40 m/s it is capped at 15 s,
because 20 s steps stopped there. At 50 and 60 m/s the cap
depends on the configured `time_step`: with 12 s it is 13.5 s, as before;
with the 15 s the domain wizard writes, the clock now takes at least six
substeps with the step capped at 15 s, where before it kept the default
24 s on six. From 70 m/s the cap is 12 s or less with either, as before.
A fixed 3 km step of 15 s or less (the domain wizard
writes 15 s) reads as before. Over that CONUS ground a fixed 18 s or 20 s
runs on six substeps at 30 m/s, and from 40 m/s is halved to 9 s or 10 s.

From 2.8.8 the clock reads every terrain face, not just one number per
domain (`terrain_clock = "local_face"`, the default;
`woof/terrain_clock_local.py`). The reading above takes the steepest
slope, the highest ground and the strongest crest-level wind anywhere in
the domain, even when they are hundreds of km apart. On NCAR's CONUS
benchmarks that put a Plains jet on a mountain face far from it and halved
both published steps, 72 s to 36 s at 12 km and 15 s to 7.5 s at 2.5 km,
although both grids ran clean at NCAR's own steps. The local reading pairs
each face with the crest and wind around it (the wind read at 2.3 times
what the inputs carry there, never under 1.5 times, and never past the
domain-wide wind), reads each pair on the map, and holds the domain to the
least of them. It is never worse than the domain-wide reading: where the
face-by-face reading finds no measured step, or would run a shorter first
or longest step, the domain takes the domain-wide decision, and the run
line and the `terrain_clock` receipt (`never_worse`) say so. On the six
long cases it was checked on, NCAR 12 km now runs 72 s on four substeps,
NCAR 2.5 km 15 s on six, SF Diablo's parent grid has no cap (13.5 s
longest), and LA Santa Ana, Toronto and Boston run what they ran before;
each of those settings held a 48 h or full-window run. `terrain_clock =
"measured"` keeps the domain-wide reading, and a checkpoint written under
it (any checkpoint from before 2.8.8 that did not name the clock) resumes
only under it.

Each `[[domain]]` may also carry an offset-free `start_time`. It defaults to
`[experiment].start_time`; d01 must equal that root start. A delayed child is
dormant until its timestamp, then follows the ordinary parent-state nest
initialization path -- it initializes from the analysis valid at its
activation instant, so that instant must be one the declared inputs decode.
The timestamp must be an exact boundary of its parent's step clock and an
exact external-forcing seam. Its first history frame is the domain's own
analysis frame and carries no `REFL_10CM`, exactly as d01's frame at t = 0
does; every frame after it carries the field. Restart headers persist
`STARTED`/`NOT_STARTED` lifecycle state, so resuming before the timestamp does
not initialize the child early.

Delayed activation is a `woof run` capability. The prepared-tree runner
(`woof-prepared-tree-forecast`, which `woof stream` and the wizard's nested
route drive) restores every domain from a prepared cache and has no way to
bring one to life mid-run, so it refuses a delayed `start_time` by name and
tells you which door runs it.

External boundary cadence has no whole-hour rule. It must be a positive,
uniform whole-second interval and an exact integer number of d01 steps because
the Davies boundary clock resets at a top-of-step seam. Five-minute forcing is
therefore valid for a 60-second d01 step; 310-second forcing is not
(`310/60 = 31/6` steps). History cadence is independent.

### Constant moving frame

Set `umove` and `vmove` (m/s, default `0.0`) in `[shared]`, or pass them
to `RunConfig`, to translate a flat idealized Cartesian domain at constant
speed. Initial and boundary winds are supplied ground-relative; the model
subtracts the frame once for transport, and surface physics and history
use ground-relative winds. Both zero skips the new path entirely. History
records the speed and ground offset at each output time.

Terrain, real-data land fields, nests, moving nests and Coriolis are
refused: ground geometry would need translation each step, and Coriolis
would need a consistently shifted geostrophic balance. Use a uniform
prescribed surface; evolving soil, urban, lake, ocean and fire fields also
need ground translation. The real-data WRF namelist importer accepts
explicit zero `&dynamics umove/vmove` and refuses nonzero values for the
same geography reason. See [moving-frame.md](../moving-frame.md) for the
direct Python API, boundary conversion and output attributes.

### Dynamics

Shared across domains unless marked per-domain. Every key below is a
consumed `RunConfig` field -- the knob-parity battery
(`tests/test_namelist_import.py`) proves each one lands on the
consuming kernel/module rather than being decorative -- and every one
is importable from a WRF namelist.

**Which keys a `[[domain]]` table may override.** Exactly these 267,
and no others (`woof/experiment.py`'s `_DOMAIN_RUN_OVERRIDES`):

    lightning_option  lightning_dt  lightning_start_seconds  flashrate_factor
    iccg_method  iccg_prescribed_num  iccg_prescribed_den  cellcount_method
    cldtop_adjustment  do_radar_ref  cu_physics  cudt_minutes  clos_choice  ishallow
    cugd_avedx  maxiens  maxens  maxens2  maxens3  ensdim  cu_diag
    convtrans_avglen_m  sas_pgcon  nsas_dx_factor  radt  radt_minutes  bldt
    ra_physics  ra_lw_physics  ra_sw_physics  ra_rrtmg_variant
    wrf_rrtmg_compatibility  o3input  use_mp_re  swrad_scat  diff_6th_factor  epssm
    spec_exp  mp_physics  moist  moist_cq  nest_microphysics_transition  spp_conv
    spp_pbl  km_opt  bl_pbl_physics  sf_sfclay_physics  c_s  c_k  moist_mix6_off
    diff_6th_factor2  diff_6th_opt  mix_isotropic  mix_upper_bound  isfflx
    tke_heat_flux  tke_drag_coefficient  tke_upper_bound  diff_6th_slopeopt
    diff_6th_thresh  dampcoef  zdamp  emdiv  smdiv  khdif  kvdif  diff_opt
    mix_full_fields  h_sca_adv_order  moist_adv_opt  v_sca_adv_order
    v_mom_adv_order  h_mom_adv_order  tke_budget  sase_flux_diag  hmix_k_diag
    surface_energy_diag  inflow_perturbation  inflow_perturbation_seed
    inflow_perturbation_amplitude_scale  inflow_perturbation_faces  target_cfl
    target_hcfl  max_step_increase_pct  starting_time_step  starting_time_step_den
    max_time_step  max_time_step_den  min_time_step  min_time_step_den
    min_time_step_sound  slope_rad  topo_shading  mosaic_urban_canopy
    sf_lake_physics  use_lakedepth  lakedepth_default  lake_min_elev  topo_wind
    history_interval2_s  history_interval_change_s  prec_acc_dt  prec_acc_dt1
    gsd_diagnostics  reset_interval1
    gwd_opt  fmoist_run  fmoist_interp  fmoist_only  fmoist_freq  fmoist_dt  ifire
    fire_boundary_guard  fire_num_ignitions  fire_ignition_ros1
    fire_ignition_start_lon1  fire_ignition_start_lat1  fire_ignition_end_lon1
    fire_ignition_end_lat1  fire_ignition_radius1  fire_ignition_start_time1
    fire_ignition_end_time1  fire_ignition_ros2  fire_ignition_start_lon2
    fire_ignition_start_lat2  fire_ignition_end_lon2  fire_ignition_end_lat2
    fire_ignition_radius2  fire_ignition_start_time2  fire_ignition_end_time2
    fire_ignition_ros3  fire_ignition_start_lon3  fire_ignition_start_lat3
    fire_ignition_end_lon3  fire_ignition_end_lat3  fire_ignition_radius3
    fire_ignition_start_time3  fire_ignition_end_time3  fire_ignition_ros4
    fire_ignition_start_lon4  fire_ignition_start_lat4  fire_ignition_end_lon4
    fire_ignition_end_lat4  fire_ignition_radius4  fire_ignition_start_time4
    fire_ignition_end_time4  fire_ignition_ros5  fire_ignition_start_lon5
    fire_ignition_start_lat5  fire_ignition_end_lon5  fire_ignition_end_lat5
    fire_ignition_radius5  fire_ignition_start_time5  fire_ignition_end_time5
    fire_ignition_start_x1  fire_ignition_start_y1  fire_ignition_end_x1
    fire_ignition_end_y1  fire_ignition_start_x2  fire_ignition_start_y2
    fire_ignition_end_x2  fire_ignition_end_y2  fire_ignition_start_x3
    fire_ignition_start_y3  fire_ignition_end_x3  fire_ignition_end_y3
    fire_ignition_start_x4  fire_ignition_start_y4  fire_ignition_end_x4
    fire_ignition_end_y4  fire_ignition_start_x5  fire_ignition_start_y5
    fire_ignition_end_x5  fire_ignition_end_y5  fire_lat_init  fire_lon_init
    fire_ign_time  fire_shape  fire_sprd_mdl  fire_crwn_hgt  fire_ext_grnd
    fire_ext_crwn  fire_sfc_flx  fire_heat_peak  fire_tg_ub  fire_smk_scheme
    fire_smk_peak  fire_smk_ext  fire_wind_height  fire_fuel_read  fire_fuel_cat
    fire_fmc_read  fire_print_msg  fire_print_file  fire_fuel_left_method
    fire_fuel_left_irl  fire_fuel_left_jrl  fire_grows_only  fire_upwinding
    fire_upwind_split  fire_viscosity  fire_lfn_ext_up  fire_topo_from_atm
    fire_advection  fire_test_steps  fire_const_time  fire_const_grnhfx
    fire_const_grnqfx  fire_atm_feedback  fire_mountain_type  fire_mountain_height
    fire_mountain_start_x  fire_mountain_start_y  fire_mountain_end_x
    fire_mountain_end_y  delt_perturbation  xrad_perturbation  yrad_perturbation
    zrad_perturbation  hght_perturbation  stretch_grd  stretch_hyp  z_grd_scale
    sfc_full_init  sfc_lu_index  sfc_tsk  sfc_tmn  fire_read_lu  fire_read_tsk
    fire_read_tmn  fire_read_atm_ht  fire_read_fire_ht  fire_read_atm_grad
    fire_read_fire_grad  sfc_vegfra  sfc_canwat  sfc_ivgtyp  sfc_isltyp
    fire_lsm_reinit  fire_lsm_reinit_iter  fire_upwinding_reinit  fire_is_real_perim
    fire_lsm_band_ngp  fire_lsm_zcoupling  fire_lsm_zcoupling_ref  fire_tracer_smoke
    fire_viscosity_bg  fire_viscosity_band  fire_viscosity_ngp  fire_slope_factor
    fuel_crosswalk  sr_x  sr_y  fire_static  fire_fuel_namelist  fire_smoke
    hailcast_opt  haildt  hailcast_version  lake_state_source  lake_state_path
    lake_state_accept_licence
    wind_farm_scheme  wind_farms  wind_farm_table  windfarm_tke_factor  wind_farm_max_dx_m

`hailcast_opt` and `haildt` select the GPU WRF HAILCAST hail diameter per
domain (see the physics table below), and `hailcast_version` its stone
model: `wrf_39_noaa` (default, the HAILCAST in NOAA's operational HRRR v4)
or `wrf_461` (WRF v4.6.1's). `lake_state_source` picks the lake
initial-state row from `woof/data/surface_obs/lake-state-sources.v1.toml`
(`auto` and `forcing` are the two built-in choices); a research-only row
also needs `lake_state_path`, its acquired CF NetCDF file, and
`lake_state_accept_licence` set to that row's licence string.

`clos_choice` and `ishallow` configure the Grell-Freitas cumulus scheme
(`cu_physics = 3`): which closure the deep scheme uses (0, the default,
is the mean of all sixteen members; 1 to 16 run one member alone, with
a warning that only 0 has been compared against WRF), and whether the
shallow scheme runs. They are per domain because the scheme they
configure is, and they are validated inert on any domain that does not
select `cu_physics = 3`. On a tree where some domains run Grell-Freitas
and others do not, a value in `[shared]` reaches the Grell-Freitas
domains and leaves the others at 0; a value written into a non-Grell
domain's own table is refused. The HRRR route writes both keys into its
namelists, where WRF reads each once for the whole run, so on that
route every Grell-Freitas domain of a tree takes the same values.

The eleven numerics from `diff_6th_slopeopt` through `tke_budget` are
per domain because WRF declares every one of them `max_domains` and the
split here had drifted from that: `diff_6th_opt` and `diff_6th_factor`
were per domain while `diff_6th_slopeopt` and `diff_6th_thresh` -- the
same filter -- were not, and `epssm` was while `emdiv` and `smdiv` were
not. A tree can now damp or filter the nest that needs it without
moving its parent, which a refinement tree needs: the relaxation sponge
is 40 km wide on a 10 km root and 2.7 km on a 667 m nest at the same
cell count. Geometry (`dx`, `dy`, `ztop`, `grid_id`, `nested`,
`specified`) stays tree-wide because the domain tree authors it, and so
do two of the scheme selectors WRF also scopes `max_domains` --
`sf_surface_physics` and the `bl_mynn_*` block: a tree whose domains ran
different land-surface or MYNN closures cannot be compared across its
own boundary, and two-way feedback already requires one microphysics
tree-wide. The radiation selectors are not among them; the radiation
row below says why.

The nine adaptive-time-step keys from `target_cfl` to
`min_time_step_den` are per domain because each domain runs its own
controller (see `docs/ADAPTIVE-TIMESTEP.md`). Note that a clamp is an
absolute number of seconds while a nest's step is a fraction of the
root's, so a `min_time_step` written once in `[shared]` reaches every
domain unchanged and can sit ABOVE the step it was meant to protect on
an inner nest. Set the clamps per domain on a refinement tree.
`min_time_step_sound` is per domain because the ground that needs it is:
the steep-terrain rules set it on the domains whose substep count they
raise and leave every other domain at WRF's derived count.

`sase_flux_diag` and `hmix_k_diag` are output-only diagnostics, and
they are per domain for the same reason: their cost scales with the
grid, so a tree can carry them on the domain whose mixing or subgrid
flux is being read and leave them off the rest.  The turbulence row
(`km_opt` through `tke_upper_bound`) and the PBL/surface selectors are
per domain because that is what makes a PBL parent able to carry a
PBL-off LES child (see `docs/public/LES.md`).  The four
`inflow_perturbation*` keys seed an LES nest child's inflow turbulence
transition (default off; deterministic, seeded, and gated
byte-identical to a build without the mechanism when off); they are
per domain because the mechanism is per nest edge â€” it perturbs one
child's parent-forced boundary tables â€” and, like per-domain
`isfflx`, they have no WRF namelist spelling, so a config using them
cannot round-trip to a namelist.  `inflow_perturbation = true` needs a
parent that runs a PBL scheme: the perturbation's depth is the parent's
diagnosed PBLH, so a child under a `bl_pbl_physics = 0` parent is
refused at load, by name, with the domain to change.  That is the
mesoscale-to-LES edge and only that edge â€” a PBL-off parent is itself
LES, and its resolved eddies already are the child's inflow turbulence.

Only `woof domain`'s own emission and hand-written TOML reach some of
them, so the list is stated here rather than left to be discovered. A
`[[domain]]` table carrying any other key is **refused** naming the
key, not accepted and not silently dropped: a config that appeared to
ask for it while the `[shared]` value ran on every nest would be a
wrong answer reported as a success. Put them in `[shared]`.
`bl_pbl_physics` takes every scheme per domain, SASE (900) included.

| TOML key | WRF equivalent | default | allowed | note |
|---|---|---|---|---|
| `time_step_sound` | `time_step_sound` | 4 | even, > 0 | WRF 0 = auto imports as 4, recorded. A domain whose terrain is steeper than four substeps were measured stable on (a slope of 0.70 in any direction at `epssm` 0.5, lower at smaller `epssm`) runs 6 and the run says so; a larger value is never lowered (`woof/acoustic_adaptation.py`). Under the adaptive clock the count follows the step, and the 6 is held as `min_time_step_sound` |
| `min_time_step_sound` | -- (WOOF) | 0 | even, >= 0, per domain | under the adaptive clock, the fewest acoustic substeps per step: the count the clock derives from its step (WRF's `time_step_sound = 0` rule, 4 at any step under about 3.3 s at 1 km) is raised to this. 0 keeps WRF's count. The steep-terrain rules set it on each adaptive domain whose count they raise. Nothing reads it under a fixed clock |
| `terrain_clock` | -- (WOOF) | `"local_face"` | `"local_face"`, `"measured"`, `"pinned"` | whether the measured terrain rules may rewrite the clock at launch. `"local_face"` (the default from 2.8.8; `"measured"` before): the terrain clock reads every terrain face with the crest and crest-level wind around it (`woof/terrain_clock_local.py`) and holds the domain to the least of them, never worse than `"measured"`: where that reading finds no measured step or would run a shorter first or longest step, the domain takes the `"measured"` decision and the receipt's `never_worse` entry says so. A checkpoint written under `"measured"` resumes only under `"measured"`. `"measured"`: the terrain clock (`woof/terrain_clock.py`) divides the step, raises the substep count or caps an adaptive step where its map saw a longer step stop under the domain's slope, crest and crest-level wind, and the steep-ground rule (`woof/acoustic_adaptation.py`) raises four substeps to six where its map says four fail. `"pinned"`: these launch rules preserve the configured clock; `time_step` and `time_step_sound` integrate exactly as written with `use_adaptive_time_step = false`, while a selected adaptive controller still updates its live clock; both rules still read the domain and write what they would have done into the run's `terrain_clock` and `acoustic_substeps` receipts (`clock = "pinned"`, an `advice` entry with `applied = false`) and print it, but apply nothing. The off-centering floor (a chosen `epssm` the map holds no count at) is refused either way. Pinned is what an operational WRF namelist means by its clock; the maps were measured on generated ridges, never on an operational grid, so there their verdict is advice and the run's own stability evidence is the referee. From 2.8.9 a WRF folder or namelist pair on a fixed step imports as `"pinned"`: it runs the `time_step_sound` the namelist names, or the count WRF's `solve_em.F` derives for 0, as wrf.exe does; an adaptive namelist, or one whose `! gpuwm-physics-selectors-v1:` comment states `terrain_clock`, keeps the terrain clock |
| `epssm` | `epssm` | 0.1 | per-domain | acoustic off-centering; scalar namelist assignment changes d01 only (Registry tail keeps 0.1), preserved per-domain |
| `smdiv` | `smdiv` | 0.1 | finite | 3-D divergence damping |
| `emdiv` | `emdiv` | 0.0 (WOOF legacy) | finite | WRF Registry default 0.01 is emitted explicitly on import |
| `km_opt` | `km_opt` | 1 | 1 (constant K), 2 (1.5-order prognostic TKE), 3 (3-D Smagorinsky), 4 (2-D Smagorinsky), 0 (no operator -- with `bl_pbl_physics = 900`, or with `km_opt_zero_acknowledgement`) | `diff_opt=2` form implied; WRF -1 must-set honored: omission refuses. **2 and 3 are the LES closures and carry extra conditions:** `km_opt=2` is admitted only with `bl_pbl_physics=0`; on a nest child it cold-starts its own TKE whatever the parent runs, and under a `km_opt=2` parent the load warns that the tree is not yet verified (see `woof/experiment.py`). `km_opt=3` has no nest restriction. See `docs/public/LES.md` |
| `km_opt_zero_acknowledgement` | (none) | `""` | the exact id `no-horizontal-mixing-operator-v1` | admits `km_opt = 0` with a PBL scheme that produces no horizontal mixing of its own -- i.e. a run with NO horizontal mixing operator, WRF's `diff_opt = 0`. Refused by default because that is what a mis-set switch looks like; the acknowledged path is the single-variable research control that varies the closure while holding the mixing at none. A literal id, not a boolean, so no stray `= true` reaches it. Refused where it would acknowledge nothing (`km_opt != 0`, or SASE, which supplies the producer). Not needed with `bl_pbl_physics = 900` |
| `hmix_k_diag` | (none) | false | bool, per domain | publishes the horizontal eddy viscosities the run's own producer used, under that producer's name: `XKMH`/`XKHH` for `km_opt = 4`, `SASE_KMH`/`SASE_KHH` for the SASE closure. Same units (m2 s-1), same mass grid, so the two are directly comparable. A run with no producer publishes neither pair -- an absent variable cannot be misread as a measured zero. Two extra (nz, ny, nx) planes per frame |
| `c_s` | `c_s` | 0.25 | > 0 | Smagorinsky constant (smag2d kernel); per-domain |
| `c_k` | `c_k` | 0.15 | > 0 | km_opt=2 TKE-closure constant, K = c_k sqrt(e) l; the WRF `em_les` reference namelist sets 0.10; per-domain |
| `mix_isotropic` | `mix_isotropic` | auto | 0, 1, `"auto"` | 0 = anisotropic mixing lengths, 1 = isotropic (dx dy dz)^(1/3); per-domain. Unset (or the string `"auto"`) lets the model choose: isotropic where `mix_upper_bound*(dz_max/dx)^2` exceeds 0.25 (announced at load and by `woof check`), the WRF-default anisotropic form otherwise. A written 0 is honoured everywhere, with a warning naming the instability and the measured ratio in the danger zone |
| `mix_upper_bound` | `mix_upper_bound` | 0.1 | > 0 | non-dimensional cap K <= mix_upper_bound len^2 / dt, applied per direction; per-domain |
| `tke_upper_bound` | `tke_upper_bound` | 1000.0 | > 0 | km_opt=2 TKE ceiling in m2 s-2, `bound_tke` clamp; per-domain |
| `tke_heat_flux` | `tke_heat_flux` | 0.0 | finite | prescribed kinematic surface heat flux, K m s-1; consumed under `isfflx` 0 and 2 with the PBL off; per-domain |
| `tke_drag_coefficient` | `tke_drag_coefficient` | 0.0 | >= 0 | prescribed surface drag coefficient; consumed under `isfflx=0` with the PBL off; per-domain |
| `khdif`, `kvdif` | `khdif`, `kvdif` | 0.0 | >= 0 | km_opt=1 only; refused with open/specified boundaries |
| `diff_6th_opt` | `diff_6th_opt` | 0 | 0, 1, 2 | option 1 refused when moist (PD bypass) |
| `diff_6th_factor` | `diff_6th_factor` | 0.12 | per-domain | |
| `diff_6th_factor2` | `diff_6th_factor2` | unset | per-domain, NOAA WRFV3.9 only | declaring it selects `diff_6th_form = "noaa_wrf39"`; fork default 0.04 for an unassigned tail |
| `diff_6th_slopeopt` | `diff_6th_slopeopt` | 0 | 0, 1 | terrain-slope taper |
| `diff_6th_thresh` | `diff_6th_thresh` | 0.10 | > 0 | slope threshold, m/m |
| `damp_opt` | `damp_opt` | 0 | 0, 3 | Rayleigh implicit w-damping |
| `zdamp` | `zdamp` | 5000.0 | m | |
| `dampcoef` | `dampcoef` | 0.2 | | |
| `w_damping`, `w_crit_cfl` | `w_damping`, `w_crit_cfl` | 0, 1.0 | 0, 1; > 0 | `w_crit_cfl` is where w-damping measures the excess vertical Courant number from, and with `zadvect_implicit = 1` where it starts (WRF suggests 2.0 there); without it damping starts at 1, so a value above 1 is refused there (it would push `w` along its own direction) |
| `zadvect_implicit` | `zadvect_implicit` | 0 | 0, 1 (a positive WRF value imports as 1) | WRF's implicit-explicit vertical advection on the last RK substep; refused with open boundaries. The default numerical generation is WRF v4.7.1, with two declared `w` boundary corrections: uncouple the lower boundary's momentum tendencies, and divide the upper boundary's geopotential change over dt by g. The lower correction prevents a column-mass-sized acceleration error |
| `zadvect_implicit_variant` | no namelist spelling | `"wrf_471"` | `"wrf_471"`, `"wrf_legacy"` | `[shared]` selects the WRF numerical generation. `wrf_legacy` ports the operational HRRR v4 `module_advect_em` current-mass solve and `WW_SPLIT` one-sided horizontal Courant allowance (alpha_max 1.0); `wrf_471` retains the newer `module_ieva_em` old/new-mass solve and mean-flow allowance (alpha_max 1.1). Both retain the declared lower-boundary unit correction. A namelist does not identify its source revision, so import preserves `wrf_471`; select `wrf_legacy` explicitly when matching the older source. Changing this value changes forecast answers and is refused on restart |
| `base_temp` | `base_temp` | 290.0 | K | base state; init-time only (see fixed table for `iso_temp`/lapse) |
| `hypsometric_opt` | `hypsometric_opt` | 1 (WOOF legacy) | 1, 2 | WRF Registry default 2 emitted explicitly on import; WRF declares this key in **`&domains`**, as one scalar for the whole run (`Registry.EM_COMMON:2283`) -- a namelist that puts it in `&dynamics` is one `wrf.exe` cannot read, and the importer refuses it there by name |
| `h_sca_adv_order` | `h_sca_adv_order` | 2 (WOOF legacy) | 2, 5 | **feeds the geopotential equation only**; transported-scalar horizontal stencils are fixed 5th order (the vertical order is `v_sca_adv_order`), so the importer accepts only the Registry default 5 |
| `v_sca_adv_order` | `v_sca_adv_order` | 3 | 3, 5 | the vertical face-flux ladder of every scalar (theta, moisture, scalars, TKE) AND of w, which WRF's advect_w keys on the scalar order: 3 is WRF's vert_order 3 (2nd order one face in from the eta boundaries, flux3 between), 5 is WRF's vert_order 5 (2nd order one face in, flux3 two in, flux5 between), the operational HRRR value, which runs it with `zadvect_implicit = 1`; applies to the positive-definite limiter's high-order flux and to the explicit share under `zadvect_implicit`. Per domain (WRF max_domains). Checkpoints bind it; prepared bundles do not |
| `v_mom_adv_order` | `v_mom_adv_order` | 3 | 3, 5 | the same ladder for u and v. Per domain |
| `h_mom_adv_order` | `h_mom_adv_order` | 5 | 5 | declaration only: the u, v and w horizontal stencils are WRF's flux5; any other value is refused by name |
| `moist_adv_opt` | `moist_adv_opt` | 1 | 0, 1 in TOML; import pins 1 | PD limiter; `scalar_adv_opt` must match (WRF option 1 on both) |
| `top_lid` | `top_lid` | **true** (WOOF) | bool | WRF Registry default is false (open top); a WOOF TOML that omits it gets the rigid lid (the 2026-07-18 open-top NaN probes; flip only with a stability receipt). A **WRF namelist** that omits it imports as WRF's open top (`top_lid = false` in the emitted TOML), because that is the run the namelist describes; on NCAR's CONUS 12 km restart hour the rigid lid put PH RMS at 28.2 m2/s2 against 2.55 with the open top. Write `top_lid = .true.` in the namelist to keep the lid |
| `moist_cq` | -- (WRF derives cq from its moist state) | **true** | bool | applies whenever water vapor exists, including passive vapor with microphysics off; dry states bypass it. Explicit `false` is a verification counterfactual |
| `spec_zone`, `relax_zone`, `spec_exp` | `&bdy_control` | 1, 4, 0.0 | | `spec_exp` acts on the root (specified) branch only, exactly as in WRF's `lbc_fcx_gcx`; nonzero on a nested child is refused. `woof downscale --point` sets `relax_zone` to two parent cells and `spec_exp` to 0 |
| `relax_timescale_s` | -- (WOOF) | 0.0 | >= 0 | Davies relaxation time scale in seconds on the first relaxed row: `fcx = ramp / relax_timescale_s`, `gcx = ramp / (5 relax_timescale_s)`. 0 is WRF's recipe (`0.1/dt`, `1/(50 dt)`, a time scale of 10 of the domain's own steps). A nest reads it in WRF's nested operation order. `woof downscale --point` sets it to the time a 20 m/s flow takes to cross one child cell, never shorter than 10 child steps |
| `relax_w` | -- (WOOF) | false | bool | a specified domain relaxes `w` toward its boundary table and takes the table's `w` on the specified rows, as a WRF nest does. False is WRF's root rule: `w` is not relaxed and the specified rows copy the first interior row. Needs a `w` table (an offline child's parent history carries `W`); without one the run stops at its first step saying so. `woof downscale --point` sets it |

### Physics cadences and scheme knobs

| TOML key | WRF equivalent | default | allowed | note |
|---|---|---|---|---|
| `ra_lw_physics` | `ra_lw_physics` | -1 | 0 (off), 1 (WRF RRTM), 4 (RTE+RRTMGP), 90 (analytic proxy) | per-domain. The split spelling of the radiation selection, and the only spelling that can ask for a MIXED pair: `ra_physics = N` means N on both streams, so longwave off under Dudhia shortwave (0/1) and WRF RRTM under Dudhia (1/1) can be written no other way. `-1` means this table does not state it, and the stream takes `ra_physics` instead. Stating one half and leaving the other at `-1` is refused -- `ra_lw_physics and ra_sw_physics must both be explicit or both be -1` -- because a half-stated pair reads as a selection and resolves as the aggregate |
| `ra_sw_physics` | `ra_sw_physics` | -1 | 0 (off), 1 (WRF Dudhia), 4 (RTE+RRTMGP), 90 (analytic proxy) | per-domain, resolved by the same rule. Restating one engine on all three keys (`ra_physics = 4` beside `4`/`4`) is one selection written twice and resolves to that pair. A nonzero `ra_physics` beside a split pair naming a DIFFERENT engine is refused as a contradiction rather than resolved: the run and its receipts would name different radiation and nothing in the configuration says which was meant. Keep the pair and set `ra_physics = 0`, or drop both to `-1` and keep `ra_physics` |
| `radt` / `radt_minutes` | `radt` | 0.0 / 12.0 | minutes; 0 = every step | per-domain; WRF `radt = 0` imports as `radt_minutes = 0.0` |
| `bldt` | `bldt` | 0.0 | minutes; 0 = every step | surface layer + LSM + PBL interval |
| `cudt_minutes` | `cudt` | 5.0 | minutes | consumed where `cu_physics = 1` |
| `icloud` | `icloud` | 1 | 0, 1 (Dudhia); fixed 1 with any RRTMG spectrum | |
| `swrad_scat` | `swrad_scat` | 1.0 | >= 0 | Dudhia scattering |
| `no_mp_heating` | `no_mp_heating` | 0 | 0, 1 | disables microphysics latent heating |
| `mp_tend_lim` | `mp_tend_lim` | 10.0 | > 0, K/s | microphysics theta-tendency clamp |
| `morr_rimed_ice` | `morr_rimed_ice` | 1 (hail) | 0, 1 | Morrison dense ice identity |
| `wsm6_hail_opt` | `hail_opt` | 0 (graupel) | 0, 1 | WSM6 rimed-ice identity |
| `ysu_topdown_pblmix` | `ysu_topdown_pblmix` | 1 | 0, 1 | YSU top-down radiation-driven mixing |
| `ysu_version` | no stock namelist spelling | `"wrf_480"` | `"wrf_480"`, `"wrf_461"` | Corrected YSU default. WRF 4.8 MMM `550b5b4` removes the old efficiency offset and caps `0.2*8*tmp1` at 0.4. `[shared]` selects the generation; namelist import uses the corrected default and preserves an explicit physics-selector comment (`! gpuwm-physics-selectors-v1: {...}`). `wrf_461` under default arithmetic reproduces 2.8.9's YSU kernel, under its restart identity. Under `GPUWM_WRF_EXACT=1` it matches WRF 4.6.1 `bl_ysu_run` word for word on the 1,188-column test set, with the same declared exceptions as `wrf_480` (YSU surface drag removed on mixed BEP urban columns; the bounded PBL-height search on four-level columns), under its own restart identity, so a strict-mode `wrf_461` checkpoint from 2.9.0 or earlier is refused rather than continued with different arithmetic. Four-level shallow columns use the declared bounded PBL-height search and zero final TKE word. A generation change is refused on restart; old headers without a selector mean `wrf_461`. The corrected path uses the shared no-FTZ loader and WOOF's own math routines |
| `nwp_diagnostics` | `nwp_diagnostics` (&time_control) | 0 | 0, 1 | per-step UP_HELI_MAX running max (2-5 km updraft helicity, WRF cal_helicity), reset each history frame, restart-carried, trajectory-inert; the other WRF nwp_output maxima are not carried; wizard configs set 1 |
| `isftcflx` | `isftcflx` | 0 | 0, 1, 2 | MM5 sfclay water-point roughness (Garratt/Donelan) |
| `iz0tlnd` | `iz0tlnd` | 0 | 0, 1, 2 | MM5 sfclay land thermal roughness |
| `usemonalb` | `usemonalb` | false | bool | Monthly background albedo for Noah and RUC; the HRRR configuration recipe selects true |
| `rdlai2d` | `rdlai2d` | false | bool | Prescribed LAI for Noah and RUC; the HRRR configuration recipe selects true |
| `opt_thcnd` | `opt_thcnd` | 1 | 1, 2 | Noah soil thermal conductivity (Johansen/McCumber-Pielke) |
| `slope_rad` | `slope_rad` | 0 | 0, 1 | per-domain. WRF v4.7.1's slope-dependent surface shortwave: the land surface receives the flux on the local slope (direct beam by slope and aspect, diffuse part unchanged), and SWNORM is written. Needs a longwave, a shortwave and a land-surface scheme, as in WRF. Refused on moving nests and streamed tiles |
| `topo_shading` | `topo_shading` | 0 | 0, 1 | per-domain, with `slope_rad = 1`. WRF's terrain shadowing: a column in a neighbour's shadow gets the diffuse part only |
| `shadlen` | `shadlen` | 25000.0 | > 0, metres | how far the shadow search looks (`[shared]` only) |
| `swint_opt` | `swint_opt` | 0 | 0, 1 | `[shared]` only (one value for the run, as in WRF). 1 is WRF's shortwave interpolation between radiation calls, as operational HRRR runs it: each radiation call fits the column's surface direct and global shortwave as a power of the solar zenith cosine, and every step rewrites SWDOWN, SWDDIR, SWDDIF, SWDDNI and GSW at the current sun, night columns zero. Needs the RRTMG shortwave (`ra_sw_physics = 4`). 0 holds the radiation call's fluxes for the whole interval. Checkpoints bind it; prepared bundles do not |
| `aer_opt` | `aer_opt` | 0 | 0, 3 | `[shared]` only. 3 is the aerosol-aware radiation operational HRRR runs: on each radiation call the legacy RRTMG shortwave gets per-band aerosol optical depth, single-scattering albedo and asymmetry built from the Thompson water- and ice-friendly aerosol numbers; needs `mp_physics = 28` and the legacy RRTMG shortwave (`ra_sw_physics = 4`, `ra_rrtmg_variant = "rrtmg_legacy"`). The longwave takes no aerosol, as in that WRF. 1 and 2 (WRF's ECMWF climatology and its aod550 namelist path) are refused by name. Checkpoints bind it; prepared bundles do not |
| `alb_sol` | `alb_sol` | 0 | 0, 1 | `[shared]` only. 1 updates sun-angle-dependent land albedo `ALBSOL` and background albedo `ALBBCKSOL` on radiation steps. Shortwave uses `ALBSOL`; RUC uses both fields and retains the fractional sea-ice blend. Needs active shortwave and the MODIS21 land-use categories. HRRR namelist imports honor the supplied value; shipped HRRR demos and the new solar-albedo monthly RUC legacy-RRTMG template explicitly select 1. Other configurations remain at 0. Checkpoints bind it when enabled; prepared bundles do not. Stock WRF 4.6.1 export omits this fork-only key, and its comparison receipt records that the stock arm does not apply the correction |
| `num_soil_layers` | `num_soil_layers` | 4 | scheme-defined | WOOF *refuses* a count the scheme does not define where WRF silently overwrites it |
| `nest_microphysics_transition` | -- | `same-scheme-only` | + `mp8-to-mp18-mass-diagnosed-v1`, `mp-edge-mass-diagnosed-v1` | WOOF-only, one-way nest MP edges. Left at the default, a mixed edge between two ported schemes resolves to the closure that pair takes (`mp8-to-mp18-mass-diagnosed-v1` for Thompson over NSSL-2, the matrix id for every other pair) and the coupler receipt records the requested and the effective policy; naming the pair's own id pins it, and naming the other mixed id is refused. An `mp_physics = 28` child entering from another scheme is seeded with WRF's own non-aerosol-aware droplet number and aerosol floors, named in the receipt |

The spelling is not part of the selection anywhere, including where a
configuration is checked against a named physics profile. A profile
pins the split pair; a configuration that reached the same two engines
through `ra_physics` matches it, which is what makes `woof
import-namelist` output runnable under the profile its namelist named
(the importer emits the aggregate for a coupled pair, and for radiation
off).

Both radiation selectors are per domain, and so are the six keys that
travel with them (`ra_physics`, `ra_rrtmg_variant`,
`wrf_rrtmg_compatibility`, `o3input`, `use_mp_re`, `swrad_scat`). Every
domain builds its own radiation driver, and spectrum composition, CAM
ozone parent transport and shared workspace sizing all resolve per
domain, so a parent running RTE+RRTMGP on both streams can carry a child
running the legacy RRTMG longwave against Dudhia shortwave -- that exact
tree is loaded, resolved and round-tripped through the rendered
experiment document in `tests/test_cam_ozone.py`. Requiring the streams
to match tree-wide would refuse a configuration the engine runs. A
domain that states neither key still takes the `[shared]` value, so no
experiment written before the split moves. Which VALUES each stream
implements, and how far each is verified, is the radiation section of
[PHYSICS.md](PHYSICS.md).

Scheme selectors (`mp_physics`, `bl_pbl_physics`, `ra_lw_physics`,
`ra_sw_physics`, `sf_sfclay_physics`, `sf_surface_physics`,
`cu_physics`) and their
allowed values are the subject of [PHYSICS.md](PHYSICS.md). Selectable
in the TOML schema is deliberately wider than runnable: readiness is
owned by `woof/physics_compat.py`, which fails closed with a complete
port receipt, and the importer's runnable sets are narrower still.

### Correct mode (`[shared]`)

Correct mode is the default. It fixes water-budget defects that WRF itself
carries, one key per scheme, each `true` by default and read only by the
schemes named in its row. Setting a key to `false` in `[shared]` runs that
scheme's WRF-matched arithmetic. Strict WRF mode (`GPUWM_WRF_EXACT=1`)
forces the WRF-matched arithmetic for every scheme. The run receipt's `correct_mode` block lists the
findings each domain's schemes carry: `applied_findings` (corrected in this
run) and `uncorrected` (carried with no correction). An engine
configuration has no `[physics]` table; `physics.mode` is not an engine key.

| key | schemes | finding | what `true` changes |
|---|---|---|---|
| `wsm6_conservative_surface` | WSM6 (`mp_physics = 6`) | WB5 | the below-ground part of the lowest falling cell is booked as surface rain with the remap's sloped profile, not the flat cell mean, so column water plus surface precipitation closes |
| `wdm6_conservative_surface` | WDM6 (16) | WB5 | the same, in WDM6 |
| `wsm3_conservative_surface` | WSM3 (3) | WB5 | the same, in WSM3 |
| `wsm5_conservative_surface` | WSM5 (4) | WB5 | the same, in WSM5 |
| `wsm7_conservative_surface` | WSM7 (24) | WB5 | the same, in WSM7 |
| `ntiedtke_dry_mass_coupling` | New Tiedtke (`cu_physics = 16`) | WB4 | moist-mass tendencies are scaled onto dry-mass mixing ratios |
| `thompson_cloud_surface_booking` | Thompson (`mp_physics = 8`) and aerosol-aware Thompson (28) | WB1 | cloud water that settles out of the lowest level is kept in `CLOUDNC` (mm, `CLOUDNCV` per call) instead of being lost; no other output changes |

A checkpoint resumes only under the arithmetic it was written with; a
mismatch is refused, and the refusal names the key to set.

### `[perturbation]` -- initial-state theta bubbles (WOOF-only)

Adjusted initial conditions for real-data experiments: one or more
warm bubbles added ONCE to the initial potential temperature after the
base real-data state is final, WRF's `em_quarter_ss` cosine-squared
shape evaluated in geographic coordinates. No WRF namelist can express
this block (stock WRF has warm bubbles only in its idealized
initializers), so `import-namelist` never emits it. Applied per domain
inside each domain's own `initialize_real` -- real-data nest init
re-ingests the source analysis per domain, so a bubble inside a nest's
footprint reaches the nest through the nest's own init, not through
the parent's state. Domains with a delayed start take no fresh bubble
(they initialize from the analysis at activation time). Absent block =
byte-identical prepared state and an unchanged experiment fingerprint.

```toml
[[perturbation.bubbles]]
center_lat = 38.5        # degrees
center_lon = -99.5
center_height_m = 1500.0 # bubble center, metres AGL
radius_km = 10.0         # horizontal radius
depth_m = 1500.0         # vertical HALF-depth
amplitude_k = 2.5        # peak theta perturbation, K (above 10 K warns)
rh_preserve = false      # optional: adjust qv so RH survives the theta change
```

The block is honored on two routes: `woof run` / `woof ingest`
(applied inside each domain's `initialize_real`, geopotential
rebalanced) and the prepared domain-tree forecast runner (`woof sim`
or `woof go` on a tree; applied to the restored sealed states, then
the geopotential is rebalanced at the held pressure as WRF's
`em_quarter_ss` does -- the preparation stays the pure analysis, so a
bubble-on arm and its control can share one preparation).
Preparing a domain tree takes the block on every source: GFS, the
mapped sources (HRRR pressure levels among them), ERA5 and native HRRR
all leave their arrays unperturbed and record the bubbles in the
preparation receipt (`proof.json`, or `receipt.json` for a native HRRR
tree) as an `initial_perturbation` entry with status
`DEFERRED_TO_FORECAST_INITIALIZATION`, the same document whichever
source wrote it. The companion stock-WRF file set cannot carry a
deferred bubble: a tree preparation that asks for it as optional (the
default) records that export as refused, and one that requires it is
refused before any source is decoded.
Refusals (never silent): unknown keys; nonpositive
`radius_km`/`depth_m`/`amplitude_k`; a center outside the
coarse domain; an enabled bubble that touches zero cells on a domain
that contains its center; a bubble that heats a layer above the top of
the radiation's temperature table (355 K under RTE+RRTMGP, which stops
the forecast at step 1 on such a layer); `rh_preserve` building more
than 0.09 kg/kg of water vapour (3 km forecasts that built 0.092 kg/kg
and more went non-finite within six minutes; 0.084 kg/kg ran). Routes
that do not thread the block refuse it by name rather than dropping
it: the single-domain prepared runner applies no bubble, so a
single-domain preparation on any source refuses the block before it
decodes a source file, as do `woof go`, `woof check` and `run-plan`
on a single-domain prepared configuration. An `amplitude_k` above
10 K runs with a warning that names it beside WRF's 3 K idealized
bubble, and the warning is recorded in the
receipt. What was actually written
-- per-domain cells touched, max theta delta, qv adjustment under
`rh_preserve` -- lands in `initial-perturbation.json` in the run
directory (the tree runner's `evidence/`), written before integration
starts.

### `[spectral_numerics]` -- Level-2 regional spectral operators (WOOF-only)

Optional, default off, and off is bitwise inert.  Scale-selective
spectral hyperdiffusion of chosen scalars and divergent-mode wind
damping, fired once per completed slow large step in `off` / `shadow`
(receipts only, state-bitwise inert) / `apply` (opt-in) modes, with
hash-bound step receipts feeding the run capsule.  Full schema, the
boundary and streamed-domain refusals, and the CLI evidence door
(`woof spectral-op`) are documented in
[LEVEL2_SPECTRAL_NUMERICS.md](LEVEL2_SPECTRAL_NUMERICS.md).  A present
table binds the restart identity; an absent table leaves every existing
fingerprint untouched.

### `[ingest]` -- soil-state ingest policy (WOOF-only)

This switch disables a deliberate divergence from WRF that removes the
forcing grid's imprint from the initial soil state. Its measured scope is
soil-state structure, not an observation-based forecast-skill improvement.

```toml
[ingest]
soil_texture_downscale = false   # default: true
```

A forcing model delivers its soil state on its own mesh -- 0.25 degrees
for GFS and ERA5 -- and stock WRF uses the interpolated result as-is, so
`SMOIS` holds no information below the source spacing and retains that
imprint in the soil state. Boxes in analysis-time 2 m dewpoint instead come
from the interpolated near-surface fields, which this operation does not
touch. In the measured run, later dewpoint frames had no source-mesh
signature with or without the change, and the block-scale amplitude moved
by 0.9 percent ([soil-state measurements](../soil-texture-downscaling.md)).
WOOF carries soil moisture across the resolution change as Noah's own
degree-of-saturation ratio and reconstitutes it against the target
grid's own 30 arc-second soil texture, and anchors the deep `TSLB`
layers on the sub-source-cell part of `TMN` with WRF's own
linear-in-depth weight. Both are ON by default and apply on every
route, nests included.

`soil_texture_downscale = false` restores the previous WOOF behaviour byte
for byte: the interpolated source soil state is used as-is, as stock WRF does,
without texture reconstitution. This is not byte identity with WPS/real.exe:
WOOF's masked surface and soil interpolation differs from METGRID's
([WRF-INTEROP.md](WRF-INTEROP.md)). Every run records the soil-state source
resolution -- and whether the reconstitution ran -- under
`soil_texture_downscale` in `proof.json`, and preparation prints an
advisory when the model resolves more than five cells across one source
cell. Refusals (never silent): an unknown key in the table, or a
non-boolean value. Full rationale, the WRF references, and the
measurements in `docs/soil-texture-downscaling.md`.

### `[radar_heating]` -- radar latent heating for the first part of a forecast (WOOF-only)

For the first `active_minutes` of a deterministic forecast, the model's
microphysics heating is replaced wherever a reflectivity window says what
is there, by NOAA's HRRR rule (`woof/da/radar_tten.py`). After that the
model runs free. Off when the table is absent.

```toml
[radar_heating]
windows = "windows/stormscope"   # required: <root>/<YYYYmmddTHHMMZ>/ref.f32 per window end
window_minutes = 10              # default 15; must divide active_minutes
active_minutes = 120             # default 60; forced period from the run start
latent_heat_period_min = 20      # NOAA dt_cond
strict_suppression = false       # observed clear means exactly zero heating
pbl_extension = false            # heat down to the PBL top where 200 hPa is observed
mp_tend_lim = 0.07               # HRRR's microphysics clamp, K/s, only while forced
```

| key | type | default | meaning |
|---|---|---|---|
| `windows` | path | none (table absent = off) | window root; a relative path is anchored at the config file |
| `window_minutes` | number | 15 | window length; must divide `active_minutes` |
| `active_minutes` | number | 60 | forced period from the run start, then pass-through |
| `latent_heat_period_min` | number | 20 | NOAA dt_cond |
| `strict_suppression` | bool | false | WOOF option, as in DA lane 6 |
| `pbl_extension` | bool | false | WOOF option, as in DA lane 6 |
| `mp_tend_lim` | number | 0.07 | HRRR's microphysics clamp, applied only while forced |

The forcing attaches at the run start and reads the window ending at start
plus n times `window_minutes`, for n = 1 to `active_minutes /
window_minutes`. There is no arrangement key: an observed pre-forecast hour
is "start an hour early with observed windows", a nowcast is "start at the
analysis with nowcast windows". Windows come from `tools/radar_tten_windows.py
--grid-prepared` (observed Level II) or `rw_nexrad grid-composite` (a 2D
nowcast or the MRMS composite, see `docs/nowcast-frames.md`).

The prepared door also takes the table on its command line
(`--radar-heating-table JSON`), so one preparation serves the unheated
control and every heated arm. A table inside the door's
`--experiment-config` works the same way: the door cuts it out and binds
the rest to the preparation, writing that copy beside the config as
`<name>.bound-<digest>.toml` so the config's relative paths still resolve.
When the preparation was made from the config with the table in it, the
door keeps the config whole instead. `woof go` on the HRRR route forwards
the table as the flag and prepares without it.

Every window is read by the strict reader: its schema, status, digest and
grid identity must match, the identity is the run's own
(`woof.da.forecast_heating.grid_for_prepared`), and its receipt must state
the window end its directory names. The run report carries a
`radar_heating` block with every window's `lead_class`: `forecast` for a
causal nowcast, `oracle` for frames issued or valid after the start (the
MRMS composite, for one), and `observed` for a Level II window. An
`observed` forced period is an analysis period: the forecast proper starts
when the forced period ends.

A present table binds the restart identity (without the windows path); an
absent table leaves every fingerprint untouched. Refusals (never silent):
a nested or tree run; a tile-streamed or batched-ensemble state;
`mp_physics = 0`; a window gridded onto another grid; a missing window or a
`window_minutes` that does not divide `active_minutes`; resuming from a
checkpoint inside the forced period; any route other than the prepared door
(the table would otherwise be dropped). A multi-card (`[devices]`) run
builds every slot on the first card and hands each card its slab, so the
heating is the one-card heating by construction; the box test that proves
it byte for byte is `tests/test_forecast_heating_gpu.py`.

## Identity-pinned option families

These are real WRF namelist keys that WOOF carries as configuration
fields but admits at exactly one value each. The admitted values and their
implementation evidence are listed below. Comparisons with unmodified WRF
Fortran are code verification, not validation against observations.
`validate_run_config` checks configuration admission and refuses other values
before a run starts, and the importer records
each supplied key as *fixed by WOOF* (or refuses a non-identity
value). Three Noah-MP keys are the exception, because they reach no
transcribed code at all: `opt_pedo`, `noahmp_output` and
`noahmp_acc_dt` run at any value of their own type, warn once, and are
still recorded as fixed at the pin the run used:

- **MYNN** (`&physics`): `bl_mynn_closure 2.6`, `bl_mynn_cloudpdf 2`,
  `bl_mynn_edmf 1`, `bl_mynn_edmf_mom 1`,
  `bl_mynn_edmf_tke 0`, `bl_mynn_cloudmix 1`,
  `bl_mynn_mixqt 0`, `bl_mynn_output 0`,
  `icloud_bl 1` (`MYNN_PBL_OPTION_IDENTITY`, `woof/config.py`).
  `bl_mynn_mixlength` instead accepts 1 (default) or 2, and
  `bl_mynn_tkeadvect` accepts false (default) or true (QKE advected as
  the `qke_adv` scalar).
  `scalar_pblmix = 1` runs WRF post-PBL local diffusion of
  `nc/ni/nwfa/nifa`; `bl_mynn_mixscalars = 1` runs MYNN plume transport.
  Both default to 0 (the WRF v4.6.1 Registry default; WRF 4.7.x defaults
  `bl_mynn_mixscalars` to 1 and the importer records that divergence when
  a MYNN namelist omits the key). `scalar_pblmix = 1` requires MYNN with
  `mp_physics = 28` and `bldt = 0`. `bl_mynn_mixscalars = 1` mixes the qn
  family under MYNN with `mp_physics = 28` (and then requires `bldt = 0`),
  is admitted as WRF's no-op under a microphysics with no number species
  (`mp_physics` 0, 1, 6), and is refused by name under a scheme WRF would
  mix only in part (8, 9, 10, 16, 18, 50: Thompson's `qni`, WDM6's `qnc`
  and the like; `MYNN_QN_FLAG_SPECIES`, `woof/config.py`).
  Selecting both is refused because WRF disables the former in that pair.
- **Noah-MP** (`&noah_mp`): `dveg 4`, `opt_crs 1`, `opt_btr 1`,
  `opt_run 3`, `opt_sfc 1`, `opt_frz 1`, `opt_inf 1`, `opt_rad 3`,
  `opt_alb 2`, `opt_snf 1`, `opt_tbot 2`, `opt_stc 1`, `opt_gla 1`,
  `opt_rsf 1`, `opt_soil 1`, `opt_pedo 1`, `opt_crop 0`, `opt_irr 0`,
  `opt_irrm 0`, `opt_infdv 0`, `opt_tdrn 0`, `soiltstep 0`,
  `noahmp_output 1`, `noahmp_acc_dt 0` -- each with its evidence line
  in `NOAHMP_OPTION_IDENTITY_EVIDENCE`.
- **RUC** (`&physics`/`&stoch`): `mosaic_lu` and `mosaic_soil` accept 0 or 1,
  default 0. `flag_sm_adj 0` remains pinned. `spp_lsm 1` is recognised
  and binds its RUC parameter-pattern consumer through the ensemble door,
  with the uncalibrated-spread warning in logs and receipts.
- **CLM lake** (`&physics`): `sf_lake_physics` accepts 0 or 1, default 0.
  `use_lakedepth` defaults to 1 and requires input bathymetry;
  `lakedepth_default` defaults to 50 m. `lake_min_elev` defaults to 5 m
  when lake cells must be derived without an input mask. These lake
  controls are per domain. See `docs/ruc-mosaic-and-clm-lake.md`.
- **NSSL 2-moment parameters** (`&physics`, `mp_physics = 18`): the
  port runs at the WRF v4.6.1 Registry defaults pinned by
  `woof/core/nssl2_contract.py` (`nssl_cccn 0.5e9`, `nssl_alphah 0`,
  `nssl_alphahl 1`, `nssl_cnoh 4e5`, ... `nssl_3moment 0`); tunable
  NSSL parameters are not yet plumbed.
- **Thompson aerosol-aware** (`&physics`/`&domains`,
  `mp_physics = 28`): native `met_em` preparation accepts an analyzed
  `QNWFA`/`QNIFA` pair. The monthly WIF dataset is selected by imported
  `use_aero_icbc .true.`, `wif_input_opt 1`, `num_wif_levels 30`;
  `auto` uses the complete analyzed pair when no source was explicitly
  requested. Receipts identify the source and any ignored analyzed pair.
  Fire emissions and black carbon remain unsupported. The synthetic
  fallback uses `use_aero_icbc .false.`, `use_rap_aero_icbc .false.`,
  `wif_input_opt 0`,
  `num_wif_levels` unused, `qna_update 0`, `wif_fire_emit .false.`,
  `wif_fire_inj` unused, `dust_emis 0`, `grav_settling 0`,
  `scalar_pblmix 0`. WRF *derives* `aer_init_opt` and
  `aer_fire_emit_opt` from the first four of those
  (`Registry.EM_COMMON:2656`, `:2658` are declared `derived`, not
  `namelist`), so they are not WOOF settings and are not exposed. The
  activation table `CCN_ACTIVATE.BIN` is a launch prerequisite WOOF
  ships, byte-validated on every load -- see [PHYSICS.md](PHYSICS.md).

  "No aerosol IC/BC" means no *ingest lane*, not an empty aerosol field:
  a cold-start mp=28 domain is initialised with WRF's own fallback, the
  synthetic CCN/IN profile `thompson_init` fills, installed once per
  domain by `woof/core/physics.py::initialize_physics`. What is missing
  is any way to *supply* an aerosol field of your own, and any aerosol at
  a specified lateral boundary -- which matters, because that boundary
  policy sweeps the initial field out of the domain in `L/U`. Both are
  quantified in [PHYSICS.md](PHYSICS.md).

### The one way to supply your own mp=28 aerosol: the WIF climatology

The paragraph above describes the **default**. There is now exactly one
implemented alternative, and it is the same one WRF has: the GLOBAL
monthly water/ice-friendly aerosol climatology
(`QNWFA_QNIFA_SIGMA_MONTHLY.dat`) that WPS routes through metgrid's
`constants_name`. Two keys in `[run]` select it, and nothing else in a
config changes:

```toml
[run]
aer_init_opt  = 1     # WRF's use_aero_icbc = .true. (real.exe derives it)
wif_input_opt = 1     # the use_wif_input package
```

With both set, `initialize_real` reads the dataset, interpolates it to
your grid and your case's valid date exactly as `real.exe` does â€”
metgrid `four_pt` horizontally, `monthly_interp_to_date` temporally
(integer julian-day weighting between month middles), `vert_interp`
linear in `log(p)` onto the dry eta pressure â€” and writes `QNWFA`,
`QNIFA`, `QNWFA2D` and `QNIFA2D`. The nonzero fields are themselves the
signal WRF's `MAXVAL` presence tests
(`phys/module_mp_thompson.F:493/:531`) read to **skip** the synthetic
profile, so nothing downstream needs a new flag.

Nothing else has to be supplied. The grid latitudes/longitudes and the
valid date are derived from the runner's own geometry and the snapshot's
`valid_time`; every real front door (`woof run`, the prepared
single-domain and two-domain runners, the direct HRRR/GFS/ERA5/mapped
routes, nests) passes them for you. There is no per-source branch: the
climatology does not come from the driving model, so the derivation is
identical for every input source.

**Staging the dataset.** It is a fixed 225,443,520-byte external file
that never changes. WOOF does **not** redistribute it â€” that is over
PyPI's 100 MB per-file cap and GitHub's 100 MiB blob limit, the same
reason `freezeH2O.dat` is externalized â€” so it joins the same command:

```
woof fetch-tables --wif --wif-only --from /path/to/WRF/run
```

stages it into `~/.woof/wif` after verifying exact size and SHA-256
`2f828eabd96a45f3872390f901240ea2259a1e9a629247010f42ce7a31cc46be`;
`woof fetch-tables --wif` alone downloads the fixed UCAR archive,
decompresses it and verifies the raw bytes before installation. Named
mirrors and offline `--from` copies remain supported. Every forecast chain that
fetches its forcing (GFS, HRRR and the staged sources) acquires this dependency
automatically during fetch when its selected physics needs monthly aerosol. Plan review and dry runs
report the pending dependency without downloading it. `[fetch] wif = false`
disables automatic acquisition and retains the missing-dataset refusal;
`woof fetch --wif` explicitly requests it. Any WRF tree that has ever run
`mp_physics = 28` from climatology already has the file. Precedence,
highest first: `wif_climatology_path` in the config,
`WOOF_WIF_CLIMATOLOGY` (full path), `WOOF_WIF_DATA_ROOT` (directory),
then `~/.woof/wif`. Leave `wif_climatology_path` unset and the staged
copy is found. A path you *name* and that does not exist is an error,
never a silent fall-through to the staged copy.

**If the dataset is absent the run refuses at config load**, by name,
with the acquisition route in the message. It never falls back to the
synthetic profile: that is a different, valid configuration
(`aer_init_opt = 0`) and a silent demotion would leave a run whose
receipt says "monthly climatology" and whose aerosol came from an
analytic curve. `wif_input_opt = 2` stays refused by name â€” it
additionally allocates the black-carbon scalar `qnbca`
(`Registry/registry.new3d_wif:82`), which has no consumer here.

**Importing a WRF namelist that uses it.** The key triple
`&physics use_aero_icbc = .true.` with `&domains wif_input_opt = 1` and
`num_wif_levels = 30` imports; `woof import-namelist` emits the two
`[run]` keys
above and prints which dataset the run will read. Half the triple is
refused, naming which half.

**Demo.** `configs/demos/mp28_wif_climatology.toml` is the two-line
delta against an ordinary 3 km real-data configuration, with the staging
command and the receipt in its header.

**The receipt.** `RealInitResult.aerosol_initialization` carries a
`wif_climatology` entry â€” dataset path, the two month indices and their
weights, the operators used â€” and sets `awaiting_profile_fill` to
`false`.

## Fixed by WOOF (WRF has a knob; WOOF has one implemented value)

Each of these keys has exactly one implemented value, and it is already
what your run gets: in the TOML the key does not exist at all, so there
is nothing for you to set. Import a namelist that names the pinned
value and it passes; name anything else and the importer says which key
and which value, rather than flipping it behind you. The row to read
before importing is `use_theta_m`, whose pin differs from what a WRF V4
namelist assumes for an omitted key (`mix_full_fields` is an ordinary knob).

| WRF key | fixed at | where it is pinned |
|---|---|---|
| `rk_ord` | 3 | RK3 stage table, `woof/core/dycore.py` |
| `h_mom_adv_order` | 5 | WRF flux5 stencil hardcoded, `woof/core/kernels/advection.cu` |
| `momentum_adv_opt` | 1 | standard (non-PD) momentum advection |
| `non_hydrostatic` | .true. | nonhydrostatic-only |
| `use_theta_m` | 0 | the engine evolves dry theta and has no moist-theta branch; every import door (`import-namelist`, `run --wrfinput` and `run --met-em`) admits a namelist's `use_theta_m = 1` (the WRF V4 omitted default) as a DECLARED DIVERGENCE announced at the terminal and recorded under "Physics substitutions" in the import receipt: the initial and boundary state is recovered exactly (moist wrfbdy THM/QV/MU converted at each forcing time; metgrid TT is physical temperature; native initialization builds dry theta from physical temperature) but the integration is dry theta, so it differs from a `use_theta_m = 1` WRF run. The V3.9 line that operational HRRR v4 runs defaults an omitted key to 0 (NOAA-EMC/HRRR v4.1.21 `Registry.EM_COMMON:2633`; dry theta, matched, no substitution): `run --wrfinput` reads the line from the files' own `TITLE` ("OUTPUT FROM REAL_EM V3.9...") and needs no flag; `import-namelist`, which sees only the namelist, takes `--wrf-version 3` (the default stays 4). The import report and the run receipt name the line that was read, what chose it (the flag, the files' `TITLE` or the default) and the Registry row the default came from |
| `scalar_adv_opt` | 1 | must match `moist_adv_opt` |
| `isfflx` | 1 | surface fluxes on |
| `sf_lake_physics`, `mosaic_lu/soil` | 0 (default), 1 | CLM lake columns and RUC weighted land-use/soil parameters; lake bathymetry and mosaic source fractions must be supplied |
| `solar_generation` | `"wrf_480"` (default), `"wrf_461"` | woof key, `[shared]`: every WRF radiation, terrain, solar albedo, urban, Noah-MP and solar interpolation consumer takes this generation. `wrf_480` uses WRF 4.8 equation of time with julian minus 0.5 and sine coefficient 0.040849. `wrf_461` retains the old julian minus 1 and 0.04089 calculations for reproduction. A plain WRF namelist takes the corrected default; an explicit selector comment carries either generation. Restart refuses a generation change; older headers without the key mean `wrf_461`. The analytic flux proxy retains its separately declared NOAA formula. |
| `noahmp_snow_age` | `"wrf_470"` (default), `"wrf_461"` | `[shared]` selects snow-aging call placement for Noah-MP (`sf_surface_physics = 4`). The default calls SNOW_AGE and SNOW_AGE_GLACIER before testing COSZ, as WRF v4.7.0 and v4.7.1 do. Solar calculations remain inside the daylight branch. `wrf_461` explicitly reproduces daylight-only aging. This does not select the complete Noah-MP 5.1.2 model. Ordinary namelist imports retain the corrected default; an explicit physics-selector comment preserves reproduction. Restart refuses a flip, including an old header without the key resumed under the corrected default. BATS albedo remains refused because SNOWALB_BATS and SNOWALB_BATS_GLACIER are not implemented. |
| `mynn_sfclay_variant` | `"wrf_461"` (global default), `"gsl_wrf39"` | woof key, `[shared]`: which generation of the MYNN surface layer (`sf_sfclay_physics = 5`) runs. `wrf_461` is WRF v4.6.1 `module_sf_mynn.F`. `gsl_wrf39` uses the GSL WRF 3.9 fork's 5-pass secant z/L search, 5 Ri / 8 Ri fallback, cap and Richardson clamp at 50, thermal log numerators and psih lower limit. Imports named `hrrr_wrf.nl` or `hrrr_wrf.nl.*`, newly authored HRRR recipes, and shipped HRRR templates select `gsl_wrf39` explicitly. Other requests retain the global default. Explicit TOML selections are preserved, and a flip is refused on restart. This does not claim exact HRRR forecast parity |
| `ruc_soilprop` | `"wrf_45"` (default), `"wrf_461"` | woof key, `[shared]`: which WRF lineage's LSMRUC SOILPROP sets soil-water diffusivity and hydraulic conductivity. `wrf_45` (WRF v4.0-4.5, also the operational RAP/HRRR branch) normalises both by the moisture above the residual, (theta - qmin)/(theta_sat - qmin), with mineral conductivity 2.0 at every quartz fraction. `wrf_461` (WRF v4.6.1) uses total moisture over porosity and 3.0 below 20 percent quartz; in dry soil its water diffusivity is 2.5 to 8 times larger, measured to raise a 3 km afternoon top soil level from 0.161 to 0.187 m3/m3 in one hour from the levels below. Select it by name for WRF v4.6.1 parity. Every RUC configuration changes answers with this key; a flip is refused on restart |
| `thompson_version` | `"wrf_461"` (default), `"wrf_39_noaa"` | woof key, `[shared]`: which generation of the aerosol-aware Thompson microphysics (`mp_physics = 28`) runs. `wrf_461` is WRF v4.6.1's. `wrf_39_noaa` is the operational WRF 3.9 fork's (NOAA-EMC/HRRR v4.1.21): graupel intercept from graupel content and supercooled rain, non-increasing downward; ice-to-snow size 200 microns; graupel density 500 kg/m3; the fork's rain-number, ice-number, ice fall speed, nucleation, sublimation and melting rules, surface CCN emission recomputed from the analyzed lowest-level number at each domain start, and its own lookup tables, acquired before first use into `WOOF_THOMPSON_FORK_TABLE_ROOT` or `~/.woof/tables/thompson-wrf39-noaa`. A fresh cache downloads the four `thompson-wrf39-noaa-*` release assets (330 MiB, from this version's release, then from v2.8.7, the first to carry them); when no release answers it builds the unmodified, hash-pinned public Fortran source using GNU Fortran and a pinned portable libc on Linux x86-64, or an installed WSL Ubuntu distribution on Windows, for which `gfortran` and `dpkg-deb` are required. `woof fetch-tables --thompson-fork --thompson-fork-only` acquires the same set explicitly. A run that does not select `wrf_39_noaa` never reads or acquires it. Offline, add `--from DIR` or set `WOOF_THOMPSON_FORK_TABLE_SOURCE_ROOT`; a pinned mirror can use `WOOF_THOMPSON_FORK_TABLE_ASSET_URL_BASE`. Every file must match the existing size and SHA-256 pins. Configuration preview declares those pins and defers acquisition and validation to execution. Refused with `mp_physics = 8`; a flip is refused on restart |
| `thompson_fork_snow_fall` | `"blend"` (default), `"wrf_39_noaa"` | woof key, `[shared]`, read only with `thompson_version = "wrf_39_noaa"`: how melting snow falls. `blend` uses the rain-share blend (WRF v4.6.1, and the fix the fork carries commented out). `wrf_39_noaa` is the fork's live form, a 1.5 boost above 0 C and a speed divided by (T - 273.15) just above +0.1 C, singular there. It is what the operational model integrates: the HRRR namelist importer, the shipped mp28 HRRR recipes and a configuration whose `[fetch] source` names an operational-fork model (hrrr*, rap*) under the fork generation on every domain select it at load when the key is omitted; a written value is kept, and the generic default stays `blend` |
| `bl_mynn_version` | `"wrf_461"` (default), `"gsd_41"` | woof key, `[shared]`: which generation of the MYNN boundary layer runs. `wrf_461` is WRF v4.6.1 `module_bl_mynn.F`. `gsd_41` ports the GSD MYNN v4.1 surface vapour flux, mixing length option 2, cloud block and radiation merge, mass-flux block, TKE predictor and water tendency conversion. Mixing length option 1, cycled initialization and closure 2.5 remain unported (`docs/dev/mynn-gsd41.md`). The named `hrrr_wrf.nl` importer, explicit HRRR recipes and the fork budget spelling `bl_mynn_tkebudget` select `gsd_41`. Requires the legacy RRTMG pair for radiation and `bl_mynn_mixscalars = 0`; refused with `spp_pbl = 1`. A flip is refused on restart |
| `bl_mynn_gsd41_unsquared_qtke` | false (default), true | woof key, `[shared]`: true takes the `gsd_41` option-2 mixing length's TKE conversion as written, 0.5*q without the square (v4.1.21 `module_bl_mynn.F:995`); false takes 0.5*q**2, as that file's option 1 and every later generation do. Read only under `bl_mynn_version = "gsd_41"`, `bl_mynn_mixlength = 2` |

| `ruc_irrigation` | `"wrf_461"` (generic default), `"wrf_45"` | woof key, `[shared]`: `wrf_461` preserves WRF v4.6.1's root-layer relaxation under `mosaic_lu = 1`. `wrf_45` uses the operational WRF v4.0-4.5 crop-fraction floor, gated on leaf area, with or without mosaic land use. The operational HRRR namelist importer and the shipped HRRR configuration recipes explicitly select `wrf_45`. An HRRR data source alone does not select it. A named change is refused on restart |
| `ruc_qvg_cold_start` | `"wrf"` (default), `"air"` | woof key, `[shared]`: how LSMRUC starts the ground vapour and condensate when the run starts without them. `wrf` (public WRF) starts QCG from the lowest-level condensate and QVG from saturation at the skin times moisture availability. `air` (the operational RAP/HRRR branch's fallback; that branch cycles QVG) starts an invalid QVG from the lowest-level vapour with no ground condensate; because SOILTEMP carries the old QVG as vapour storage it pulls the skin toward the air's dewpoint on the first steps, measured 2.7 K colder after 20 steps on a moist test column and -0.013 to +0.018 K of 2 m dewpoint on a 3 km cut of an operational-HRRR start. Read only on a cold start; a flip is refused on restart. `woof domain`, `woof physics-catalog --into` and a `woof downscale` child moved onto RUC write `air` for a RUC suite whose file states no form (no source carries the ground vapour, so every RUC forecast cold-starts it, and the public-WRF form dropped the land skin 3.5-4 K on the first step over dry soil); an imported WRF namelist, or a file already on RUC that states no form, keeps `wrf` |
| `ruc_2m_diagnostic` | `"flux"` (default), `"log_profile"` | woof key, `[shared]`: how RUC's SFCDIAGS_RUCLSM writes T2, TH2 and Q2. `flux` is public WRF's flux form. `log_profile` adds the block the operational RAP/HRRR branch carries and no public WRF has: where the air is warmer or moister than the surface, T2 and Q2 follow a logarithmic profile between the surface and half the lowest layer, with no saturation cap. Its final surface-driver bound limits Q2 to 1.05 times the lowest-level vapour mixing ratio over both land and water; `flux` applies that bound to land only. Measured at night on a 3 km cut of an operational-HRRR start: T2 0.33 to 0.36 K lower and 2 m dewpoint 0.01 K lower than the flux form, where the operational model's own files match the flux form's T2 within 0.04 K. A flip is refused on restart |
| `ruc_snow` | `"wrf_461"` (generic default), `"wrf_45"` | woof key, `[shared]`: `wrf_461` preserves WRF v4.6.1 snow conductivity, cover, albedo and melt. `wrf_45` selects the operational WRF v4.0-4.5 set: conductivity 0.265 W/m/K, depth-based cover without the final rebuild, fresh-snow albedo from depth on the ground, a melt cap independent of the step, capped bottom melt and cover-weighted bookkeeping. SNOWFALLAC stays in millimetres. The operational HRRR namelist importer and shipped HRRR configuration recipes explicitly select `wrf_45`; a data source alone does not. The full raw HRRR namelist still refuses unported controls; the automatic selections apply to supported resolved imports. A named change is refused on restart |
| `sf_urban_physics` | 0 (default) | 1 single-layer urban canopy, 2 BEP, 3 BEP+BEM; requires Noah or Noah-MP; mosaic admits only option 1 |
| `urban_water_balance` | `"wrf_480"` (default), `"wrf_471"` | woof key in `[shared]`, read with `sf_urban_physics = 1`, including Noah mosaic. `wrf_480` removes the roof, wall and ground rain-positive storage clamps, so weak rain can accompany net evaporation. `wrf_471` explicitly reproduces the older clamps. Both retain the `rho_air/1000` conversion and zero/cap bounds. This selects the WRF 4.8 water-balance change only; it does not select vegetation or tree physics. A public WRF namelist keeps the corrected default because it carries no source-version selector; the woof extension `&physics urban_water_balance` can select a generation explicitly. A generation change refuses an SLUCM restart, including an older header that omitted the key. |
| `urban_ucm_generation` | `"wrf_471"` (default), `"wrf_480"` | `[shared]` numerical generation for `sf_urban_physics = 1`. The 4.8 arm adds vegetated canyon ground, moisture/root uptake, ground temperature and canopy flux feedback. It carries eight additional restart fields. A generation change refuses an incompatible restart. Qualification limits are stated in the port report |
| `urban_treeoption`, `urban_ttscheme` | -1 (table), 0, 1 | `[shared]` overrides for WRF 4.8 URBPARM TREEOPTION and TTscheme. TREEOPTION=1 activates the vegetation/tree solve. TTscheme=0 diagnoses tree temperature; 1 carries prognostic temperature. TS_SCHEME=2 with TREEOPTION=1 is refused because the source omits vegetated-ground heat/water calculations and reads undefined exchange coefficients |
| `urban_fvg` | -1 (table), 0.0 through 1.0 | `[shared]` vegetated ground fraction for WRF 4.8. FVG alone does not activate the TREEOPTION=1 solve. Changing it refuses an incompatible restart |
| `urban_table_dir` | empty (bundled pinned tables) | `[shared]` directory of explicit URBPARM tables for the selected generation. Table hashes and packed coefficients bind restart identity; custom tables use the same parser and bounds checks. The first-level height admission reads the selected tables |
| `sf_surface_mosaic`, `mosaic_cat` | 0, 3 | Noah land-use tiles; enabled only at 1, positive tile count; requires LANDUSEF; urban option 1 runs per tile; urban options 2 and 3 are refused as in WRF |
| `mosaic_urban_canopy` | "dominant" | woof key, per domain: where mosaic runs urban option 1. "dominant" is WRF's rule (only cells whose dominant category is urban); "every_tile" also runs the town tiles of mostly rural cells at their own land-use weights, sharing the URBPARM urban fraction of the largest urban tile's type; needs `sf_surface_mosaic = 1` and `sf_urban_physics = 1` |
| `use_mp_re` | 1 | microphysics effective radii reach radiation per WRF's scheme table |
| `o3input` | 2 | CAM climatological ozone (RRTMG spectra) |
| `ghg_input` | 0 | analytic year-formula trace gases (no CAMtr reader) |
| `cldovrlp` / `idcor` | 2 / 0 | McICA maximum-random overlap, constant decorrelation |
| `gwd_opt` | 0 | no gravity-wave drag |
| `shcu_physics` | 0 | no shallow cumulus |
| `cu_rad_feedback` | .false. | KF cloud fraction does not feed radiation |
| `kf_edrates` | 0 | no KF rate diagnostics |
| `sst_update`, `sst_skin`, `tmn_update` | 0 | single-analysis case runs |
| `use_aero_icbc` | .false. | imported `.true.` with `wif_input_opt 1` selects the monthly WIF dataset |
| `use_rap_aero_icbc` | .false. | `.true.` selects analyzed QNWFA/QNIFA initial and lateral values, with operational monthly surface emissions; see [analyzed aerosol inputs](ANALYZED-AEROSOL-INPUT.md) |
| `wif_input_opt` | 0 | synthetic fallback identity; the imported monthly WIF route accepts value 1 with `num_wif_levels = 30`. Value 2 requires unimplemented black carbon. At 0, `num_wif_levels` is inert. **WRF's `real.exe` FATALs `mp_physics = 28` at this value** (`dyn_em/module_initialize_real.F:2734-2736`) while WOOF runs it, taking WRF's own internal fallback â€” the synthetic CCN/IN profile `thompson_init` installs â€” as the aerosol initial condition. So a WOOF mp=28 run and a WIF-initialised WRF mp=28 run are **not** directly comparable; see D9a/D9b in [PROVENANCE.md](../../PROVENANCE.md) |
| `qna_update` | 0 | no auxiliary `wrfqnainp` input stream |
| `wif_fire_emit`, `wif_fire_inj` | .false. / unused | no biomass-burning aerosol emission inventory |
| `dust_emis` | 0 | no non-chem dust source; `nifa2d` stays exactly zero, matching `thompson_init` |
| `grav_settling` | 0 | fog gravitational settling not ported. WRF *silently* forces 0 on every `mp_physics = 28` domain (`share/module_check_a_mundo.F:2459-2474`); WOOF refuses a nonzero value instead |
| `interp_method_type` | 2 | SINT nest interpolation only |
| `input_from_file` | .true. | per-domain real init is the T branch |
| `&stoch` selectors (`sppt`, `skebs`, `spp`, `spp_conv`, `spp_pbl`, `spp_lsm`) | 0 | connected consumers run with `uncalibrated spread, WRF default amplitudes` in logs and receipts; SPP requires GF, WRF MYNN or RUC as appropriate. All-off controls preserve ordinary forecast bytes. `rand_perturb`, `multi_perturb`, boundary and `pert_*` switches refuse because their missing runtime consumers would leave requested fields unperturbed. The ensemble also accepts `perturbation = {kind = "uniform-wind", amplitude = 0.5}` in m/s, labelled as an illustrative engine IC amplitude. See [stochastic import](../ensemble-wrf-stochastic-import.md) |

Init-side constants frozen at the WRF reference behavior (no namelist
counterpart is honored): base-state `iso_temp = 200 K` and
`base_lapse = 50 K` (`woof/ingest/real.py`), `p00 = 1e5 Pa`,
turbulent Prandtl number 1/3, Smagorinsky K cap `10*sqrt(dx*dy)`, and
the whole real.exe vertical-interpolation policy set
(`lagrange_order 2`, `extrap_type 2`, `t_extrap_type 2`,
`zap_close_levels 500 Pa`, `force_sfc_in_vinterp 1`,
`use_levels_below_ground/use_surface .true.`), pinned by the
preprocessing provenance contract. `sfcp_to_sfcp` is the one
real-init policy that is a knob (`[case_data]`, and `false` is
fail-loud unimplemented).

## Sub-grid terrain drag

`topo_wind` accepts 0, 1 and 2; `gwd_opt` accepts 0, 1 and 3. Both default
to 0. They are selected in `[shared]` or on a domain and imported from
WRF namelists. See [Terrain drag](TERRAIN-DRAG.md) for the schemes, required
WPS geography and preparation command.

## Terrain smoothing (`[[domain]] static`)

Preparation automatically adds Rust 1-2-1 passes when a domain exceeds the
measured slope envelope. It checks the final terrain after any high-resolution
overlay, before building the vertical coordinate. Only that domain changes.
Terrain already below the limit keeps its arrays and existing preparation keys.
The limit comes from the local-face clock's combined measured map, capped by
the acoustic stability map, not a location or source-specific threshold.
At 500 m the acoustic cap is 0.85. The clock still checks crest, wind and step.

The static receipt and forecast identity record the domain, additional pass
count, slope before and after, bound, and full terrain hashes. Each changed
domain prints a `terrain: dNN auto-smoothed ...` line. `woof domain` also
surveys an explicitly supplied, locally staged WPS_GEOG terrain tree and writes
`*.terrain-autosmooth.json` when it needs smoothing. Preparation checks the
final overlays again, so a domain survey is not an attestation of a later
high-resolution overlay.

After at most 32 additional passes, terrain still outside the limit is refused
before GPU initialization. Use a coarser grid or smoother terrain input and
prepare again. A previously prepared over-limit forecast is also refused and
names re-preparation as the remedy. A moving domain prepared with
`--statics-corridor` is smoothed once over its whole corridor and starts on
the corridor's crop, so every move shares its edge heights; over steep
ground that footprint can get one more smoothing pass than a fixed domain
would. A moving domain without a corridor that requires smoothing is refused
before initialization: its shared edge heights would differ at relocation.
Use a fixed domain or prepare a statics corridor.

WPS smooths `HGT_M` with whatever its `GEOGRID.TBL` names; the stock table
runs one `smth-desmth_special` pass, and so does WOOF unless a domain asks
otherwise:

```toml
[[domain]]
grid_id = 2
# ...
static = { smooth_option = "none" }                       # sampled terrain, unsmoothed
# static = { smooth_option = "1-2-1", smooth_passes = 3 }  # or any GEOGRID.TBL smoother
```

`smooth_option` is `smth-desmth_special` (the default), `smth-desmth`,
`1-2-1` or `none`; `smooth_passes` is a whole number of passes, 1 or more,
default 1. Each domain chooses its own. The default keeps WOOF's own
double-precision smoother, within 1.5 mm of WPS on the Alpine domains measured; every
other setting is WPS's single-precision arithmetic, identical bit for bit
to `geogrid.exe` given the same unsmoothed terrain.

`smooth_precision = "wps-float32"` is the option that runs the default
smoother in WPS's single-precision arithmetic as well, so the domain's
terrain is `geogrid.exe`'s bit for bit; `"float64"`, WOOF's own smoother,
stays the default. On its own the key keeps the default smoother:

```toml
static = { smooth_precision = "wps-float32" }   # WPS's default, WPS's arithmetic
```

Only the default smoother has the choice. Every other smoother has WPS's
arithmetic alone, so `"float64"` beside one is refused, as is any precision
beside `none`.

Doors: `woof domain --terrain-smoothing none,1-2-1:3` (one item per
domain, the last repeating), and `woof import-namelist`, which reads HGT_M's
setting from `--geogrid-tbl PATH`, else the `opt_geogrid_tbl_path` in
`namelist.wps`, else `./geogrid/GEOGRID.TBL` beside it, and writes it on
every domain as WPS applies one table to all.

The WPS-exact default through the doors: `--terrain-smoothing-precision
wps-float32` on `woof domain` or `woof import-namelist` sets it on every
domain whose smoother is the default. On import, a `GEOGRID.TBL` whose
HGT_M entry carries `smooth_precision = wps-float32` selects it too:

```
name = HGT_M
        ...
        smooth_option = smth-desmth_special; smooth_passes=1
        smooth_precision = wps-float32
```

That key is WOOF's, not WPS's: `geogrid.exe` logs it as an unrecognized
option and smooths in single precision as it always does, so one table
serves both. The flag, when given, wins over the table.

With nests:

* A nest's outer rows are still its parent's terrain. As in WRF's
  `blend_terrain`, the first `spec_bdy_width` rows (default 5) take the
  parent's interpolated terrain and the next `blend_width` rows (default 5)
  ramp to the nest's own, so an unsmoothed nest shows its own valleys from
  the 11th row in. Keep the valleys that matter at least that far inside.
* The parent keeps its own terrain under the nest. WOOF's two-way feedback
  carries the prognostic fields only (WRF also feeds the nest's terrain
  back), so each domain's setting governs that domain alone.
* `[experiment] smooth_option` is WRF's `&domains smooth_option`, the
  two-way feedback smoother, and has nothing to do with this key.
* WRF's `smooth_cg_topo` (d01's outer rows blended toward the driving
  model's terrain) is not implemented; `import-namelist` refuses it.
* A moving domain (a `follow` table or `[relocation]`, and every domain
  under it) takes `none`, `1-2-1` up to 3 passes, or a smoother-desmoother
  with 1 pass. More passes carry the smoother past the 3 cells sampled
  beyond the domain edge, the edge rows then depend on where the domain
  sits, and the first move would be refused; the configuration is refused
  at load instead.

Unsmoothed terrain is steeper (a 1 km Alpine domain went from a steepest
slope of 0.85 to 1.07), and the terrain clock (Clock, above) reads it before
the forecast starts. Past the slopes its map was measured on, the clock runs
the configured step and warns that the run may stop: under a 44 m/s crest
wind that domain's smoothed terrain was held to 3.5 s steps, while the
unsmoothed terrain ran 8 s steps (and finished its 3 hours). Watch for that
warning, and set `max_time_step` yourself if the run stops.

Roots built from `WPS_GEOG` on the ERA5, GFS, native HRRR and mapped-source (ICON, ECMWF and the rest) routes carry the setting, and so does every nest.
A root loaded from a prebuilt static cache that does not record its smoothing refuses a non-default setting before integrating default terrain.

## Runtime choices on an existing prepared state

The Python prepared-forecast API accepts `runtime_run_overrides` for run
fields listed in `woof.ingest.prepared_cache.PREPARATION_INERT_RUN_FIELDS`.
Keep the original experiment file, WPS file and preparation digests in
the same preflight keyword bindings. For a prepared RUC configuration:

```python
from pathlib import Path
from woof.prepared_single_domain_forecast import (
    preflight_prepared_forecast, run_prepared_forecast,
)

inputs = preflight_prepared_forecast(
    **original_preflight_bindings,
    runtime_run_overrides={"rdlai2d": True, "usemonalb": True},
)
run_prepared_forecast(inputs, output_directory=Path("forecast-prescribed-surface"))
```

Each value receives the normal run-config validation. The execution plan
records the original and executed values, and the prepared bytes and
their digests remain unchanged. Grid, soil geometry, input sources and
other preparation inputs require a new preparation. The CLI continues
to require the original run-control bytes.

## Not implemented (refused or dropped with a reason)

WRF's namelist moving nests (the vortex-following keys; WOOF's own moving
nests are configured in its experiment file),
vertical nest refinement, FDDA nudging
(active `grid_fdda`/`grid_sfdda`/`obs_nudge_opt` refuse; inert keys
drop), implemented active `&stoch` selectors (admitted with a spread warning; see the
`&stoch` row above), unimplemented stochastic field and boundary consumers,
the physics options `docs/public/WRF-OPTIONS.md` lists as refused,
auxiliary I/O streams (`auxhist*`/`auxinput*`, `iofields_filename`;
WOOF writes one fixed wrfout frame per file per domain -- fields are
not namelist-selectable), quilt servers, and WRF process/tile
decomposition (`numtiles`, `nproc_x/y` -- dropped; GPU decomposition
is internal). A namelist key outside every table above is a hard
`unmapped key(s)` error: the importer never drops a setting silently.

## Where the values come from

`diff_6th_form = "wrf_461"` keeps the single filter factor, scalar step
`dt/3`, and three-point specified-boundary exclusion. `"noaa_wrf39"`
uses `diff_6th_factor2` on the full step for moisture and number scalars
and filters to the specified or nested domain edge. Its factor defaults
to 0.04 when unset. `upper_wind_limiter_form` takes the same source names;
the fork form applies its 110 m/s saved-wind limiter in the damping layer
when `damp_opt = 3`. Both source selectors default to `"wrf_461"`.
An imported namelist declaring `diff_6th_factor2` selects both fork forms.

`mp_zero_out` defaults to 0 (off). Mode 1 zeroes non-vapour fields below
`mp_zero_out_thresh` (default 1e-8), and mode 2 also floors vapour at zero.
Either mode floors the outer ring at zero. `mp_zero_out_all = 1` also
applies the pass to number scalars, with the first Registry scalar taking
vapour's rule. Its default is 0. Fork namelists that enable `mp_zero_out`
import with this switch set to 1. No chemistry or tracer array is bound
to this pass. The native member batch declines active fork filter and
upper-wind forms and active zero-out; those configurations use the
ordinary forecast door.

- Schema + invariants: `woof/config.py` (`RunConfig`,
  `validate_run_config`), `woof/experiment.py` (experiment tables).
- Importer: `woof/namelist_import.py`; every decision lands in the
  three-section substitution report.
- Reach-tests: `tests/test_namelist_import.py` (knob-parity battery)
  proves each translated knob lands on the consuming `RunConfig`
  field by driving a distinctive value through the import and
  asserting it reaches the consuming kernel/module -- the tests
  themselves name the per-kernel consumption sites.

## Smoke, dust and air quality (`[shared]`, opt-in)

Off by default: with `chem_sets` absent or empty a run, its wrfout files and
its checkpoints are byte for byte what they were before the option existed.
Turning it on is a few `[shared]` keys. A worked example for the editable
starter (`configs/starters/regional-gfs-rrtmgp.toml`, 12 km with Kain-Fritsch
cumulus) that carries wildfire smoke from NOAA's RAVE fire product plus
GOCART dust, sea salt and black and organic carbon:

```toml
[shared]
chem_sets = "gocart_primary,smoke"
chem_sources = "rave-3km"
dust_opt = 1
seas_opt = 1
wetscav_onoff = -1
chem_conv_tr = 0
```

| key | values | what it does |
|---|---|---|
| `chem_sets` | `smoke`, `gocart_primary`, `dust`, `gocart_lite`, `gocart_simple`, `cams_aq`, comma separated | the transported species; empty is off |
| `chem_sources` | `rave-3km`, `cams-global`, `cams-oxidants`, `hrrr-native-smoke` | where emissions and start and edge values come from; a smoke run whose `[fetch] source` is HRRR gets `hrrr-native-smoke` (HRRR-Smoke's own smoke as start and edge values) without naming it |
| `fire_emission_mode` | `trailing_24h_dcycle` (default), `observed_hourly`, `daily_mean_dcycle` | how RAVE feeds the fires: by default each fire's mean over the 24 hours before the start, newest hour included, on a diurnal cycle (HRRR-Smoke's method); observed only for hindcasts |
| `dust_opt`, `seas_opt` | 1 GOCART dust (3 AFWA), 1 GOCART sea salt | dust and sea-salt emission |
| `wetscav_onoff` | -1 (the default when `smoke` is on) | large-scale rain washes species out; 0 turns it off |
| `chem_conv_tr` | 0 | required with a cumulus scheme on: convective transport is not transcribed and the default 1 is refused |

The run then writes `smoke`, `PM2_5_DRY`, `PM10`, `SMOKE_SFC`,
`SMOKE_COLUMN`, `DUST_SFC`, `AOD5502D` and the species rows to every wrfout,
and `woof render --products smoke_near_surface,pm25_near_surface,aod_550`
draws them. Dust needs `woof fetch-geog --datasets chem-dust` once; CAMS
sources need an Atmosphere Data Store key. Every option, refusal and output is
in [SMOKE-AND-AIR-QUALITY.md](SMOKE-AND-AIR-QUALITY.md).

## What a WRF namelist setting becomes

One rule: a namelist setting that changes the forecast either maps onto an
implemented equivalent or is **refused with the breakage named**; it never
imports to run as something else. `tools/namelist_door_audit.py` checks
every WRF v4.7.1 key against it (each key written off its Registry default
into a valid pair, one at a time), and `tests/test_namelist_door_audit.py`
keeps the count of keys consumed with no record at zero.

| Namelist asks for | What the importer does |
|---|---|
| `mp_physics = 55` (Jensen-ISHMAEL) | refused by name (it ran Morrison until 2026-10-12); ask for 10 to run Morrison |
| `mp_tend_radar = 1` (operational fork radar latent heating) | applied on `woof run --wrfinput` when the WRF input carries the cloud analysis's `RAD_TTEN_DFI_1..4` and `TTEN_TIMES`, as the operational fork applies them (the tendency replaces the clamped microphysics heating where the radar covers, the last slot holding after its time); refused, with the breakage named, on a door whose input has no tendency (GRIB, met_em). WOOF's radar heating from observed reflectivity is the `[radar_heating]` table |
| `ra_sw_eclipse = 1` | exact when WRF's eclipse table has no eclipse inside the run; refused, with the eclipse named, when it has one |
| `&dynamics top_lid` omitted | WRF's open top (`top_lid = false`); write `.true.` to keep the rigid lid |
| `&chem` with `chem_opt` on | mapped onto WOOF's chemistry (`chem_sets`, `chem_sources`, the carried `&chem` keys) or refused key by key; never run with no chemistry |
| a `&stoch` selector with no consumer | refused, naming the WRF operation that would be lost |
| `fractional_seaice = 1` under a non-RUC land surface | declared divergence: whole-cell ice at XICE >= 0.5; differs only where the sea-ice fraction is strictly between 0.02 and 1 |
| `hailcast_opt = 1` | GPU WRF HAILCAST and PSD hail diameter; automatic at spacing up to 4 km by default. `haildt` is seconds (60 by default, 0 means every step); `hailcast_version` picks operational HRRR v4's stone model (`wrf_39_noaa`, default) or WRF v4.6.1's (`wrf_461`); [hail diameter products](HAIL-DIAMETER.md) |
| `dfi_opt` other than 0 | refused: WOOF has no digital filter, so every hour would integrate from a different initial state (a declared divergence until 2026-10-12) |

The unmodified operational 3 km namelist (NOAA-EMC/HRRR v4.1.21
`parm/conus/hrrr_wrf.nl`) is refused with its four uncarried smoke
settings listed in one message (smoke dry deposition, `enh_vermix`,
`flam_part`, smoke direct radiative feedback); with those changed it
imports with `chem_sets = "smoke"` and HAILCAST enabled.
Its WPS partner (`parm/conus/hrrr_namelist.wps`) names
`geog_data_res = 'modis_15s+modis_fpar+modis_lai+30s'`, which builds: the
15 arc-second MODIS land use, the MODIS FPAR greenness fraction (the same
dataset and interpolation as `default`), the 30 arc-second MODIS LAI and
the 30 arc-second soil types, as WPS 4.6 `GEOGRID.TBL.ARW` selects them.
The 15 arc-second land use has no lake class, so geogrid writes
`NUM_LAND_CAT = 20` for it and WRF's real.exe stops on the pair's
`num_land_cat = 21`; the importer refuses that value by name with the same
reason (set `num_land_cat = 20`). Stage the two datasets the default fetch
does not carry with
`woof fetch-geog --datasets modis_landuse_20class_15s,lai_modis_30s`.


## GFS humidity, snow and model-top preparation

GFS RH is converted with WPS `fix_gfs_rh` before horizontal interpolation,
including the 2 m field. Below freezing this uses the existing mixed-phase
saturation conversion; ERA5 and explicitly water-relative routes retain their
own conventions. Lakes use the water mask for ice and skin temperature and
exclude land snow.

A lake GFS does not resolve takes the skin temperature of the nearest GFS
water, as WPS metgrid and real.exe give it. This is the default,
`gfs_lake_skin = "wps"`. To give such lakes the 2 m air temperature instead,
as the inland-water reach does for other routes, add
`gfs_lake_skin = "air_temperature"` to the complete `[case_data]` table. Other
inland water (enclosed water that is not a land-use lake) keeps the inland-water
reach either way, and the preparation log counts the cells it fills.

GFS carries no ice for a lake it does not resolve, so the GFS route starts its
lakes from an observed snow and ice analysis: the NOAA/NESDIS IMS daily
Northern Hemisphere analysis (NSIDC data set G02156, 1 km, with 4 km as the
fallback), valid at 00 UTC on the start day or up to three days earlier. A
lake cell is frozen where the analysis shows lake ice, or, for a lake too small
for its water mask, snow-covered ground. The cell then starts with
`SEAICE = XICE = 1` and a skin temperature of at most 271.4 K, and the soil
column is built as for sea ice. Open cells keep the rule above, and ice the
forcing already has is never removed. Every domain of a nested tree is frozen
from the same analysis on its own grid. This is the default,
`gfs_lake_ice = "observed"`. The preparation log says which analysis it read
and how many cells it froze, and the source identity records the file and its
SHA-256. The analysis is fetched without a key into
`~/.woof/cache/surface-obs` (`WOOF_SURFACE_OBS_CACHE` moves it). On a machine
without network access, put the file there under its own name. When no
analysis can be read, the run prints a warning that names each file it tried,
and keeps the forcing's ice for that run only. `gfs_lake_ice = "wps"` in the
complete `[case_data]` table keeps the forcing's ice outright, which is what
WPS and real.exe give.

The GFS decoder defaults to the WPS snow convention: `SNOW = 2 * WEASD` and
`SNOWH = 0.005 * SNOW`. To retain the raw GFS WEASD and separately supplied SNOD,
add `gfs_snow_convention = "raw"` to the experiment's complete `[case_data]`
table. The table's other required input declarations still apply. The decoder
records the convention and refuses a cached decoder result using another
convention. Direct bridge callers can select the same behavior with
`GPUWM_GFS_SNOW_CONVENTION=raw`; normal preparation uses the experiment policy.
This default establishes WPS parity. On one winter start (2025-02-20 12Z,
71,467 snow-covered US land cells around the Great Lakes) it was also closer
to the SNODAS snow analysis than the raw value: snow water bias +3.5 mm
(RMSE 23.2 mm) against -17.8 mm (35.7 mm) for raw WEASD. That is one case.

Fetched and decoded pressure ladders include at least one level strictly above
`p_top`. A 100 hPa top therefore includes 70 hPa. An old full-file download can
be decoded again; an old subset missing that extra level must be fetched again.
