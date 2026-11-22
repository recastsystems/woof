# RW-WPS namelist compatibility

RW-WPS accepts the user's standard `namelist.wps` and `namelist.input` as
authorities. It does not silently replace a scheme, shorten a domain array, or
force the historical 49-mass-level profile.

The machine-readable preflight is:

```text
woof-wrf-init --namelist-support-report \
  --wps-namelist namelist.wps \
  --namelist-input namelist.input \
  --source-top-pressure-pa 5000
```

Exit status is zero only when the stock-WRF initialized-state contract passes;
an unsupported or unclassified setting prints the same JSON schema with
`"verdict": "FAIL"` and exits with configuration status 78. The report schema
is `rw-wps.namelist-support.v1`.

## What the report separates

- `preprocessing_relevant`: time, projection, ordered domain topology,
  horizontal geometry, explicit eta/p-top, external boundary, and similar
  settings that change native preparation.
- `physics_state_relevant`: scheme choices that alter the variables or lower
  boundary state which must exist in `wrfinput_dNN`.
- `runtime_output_only`: settings such as history cadence and physics call
  cadence which CPU WRF consumes after initialization.
- `legacy_stage_only`: WPS `ungrib`/`metgrid` controls retained in the report
  for visibility but replaced by native source decoding.

An unknown key is not guessed to be harmless. It receives
`UNCLASSIFIED_NAMELIST_SETTING` with an action describing the missing rule.

Every issue carries a `severity`. `blocking` (the default) decides the
verdict: the export cannot be written, or the pair contradicts itself.
`advisory` is stated and does not fail the report, which is how a prepared
route says what differs about it. `"verdict": "FAIL"` means at least one
blocking issue, never merely that the report had something to say.

## Current initialized-state contract

The stock-WRF and gpuwm-runtime verdicts are intentionally distinct. RW-WPS
can inventory and prepare state for a CPU-WRF physics package even when the
woof forecast runtime does not implement that package.

The following microphysics package inventories come directly from WRF v4.6.1
`Registry/Registry.EM_COMMON` package declarations and field I/O flags:

| `mp_physics` | Package | `wrfinput` package members |
|---:|---|---|
| 6 | WSM6 | `QVAPOR QCLOUD QRAIN QICE QSNOW QGRAUP` |
| 8 | Thompson | WSM6 mass species plus `QNICE QNRAIN` |
| 10 | Morrison two-moment | WSM6 mass species plus `QNICE QNSNOW QNRAIN QNGRAUPEL` |
| 18 | NSSL two-moment | WSM6 mass species plus `QHAIL QNDROP QNRAIN QNICE QNSNOW QNGRAUPEL QNHAIL QNCCN QVGRAUPEL QVHAIL` |
| 28 | Thompson aerosol-aware | Thompson's members plus `QNCLOUD QNWFA QNIFA QNBCA QNWFA2D QNIFA2D` |
| 50 | P3 one-category two-moment ice | `QVAPOR QCLOUD QRAIN QICE QNICE QNRAIN QIR QIB` |

This table is the whole of `woof/wrf_physics_inventory.py`'s
`_INVENTORIES`, which is what the export refusal names when a scheme has no
row. A selector absent from it is an EXPORT limit and says nothing about
what WOOF-ArWen runs: the native door (`woof import-namelist` then `woof run`)
runs every scheme WOOF-ArWen implements, and the report's `gpuwm_runtime` verdict
is computed from the engine's own implemented set, never from this table.

All are WRF `real` fields: NetCDF float32 on
`Time,bottom_top,south_north,west_east`. Water vapor comes from the source;
source-absent hydrometeors and number moments use real.exe's zero
initialization policy, except that a Thompson (`mp_physics = 8` or `28`) cold
start over analyzed condensate closes the number moments the scheme carries
through its own entry block once, and the prepared cache's
`hydrometeor_initialization.cold_start_moment_closure` receipt says how many
cells were written. Before that closure, mass that arrives with a number at or
below zero takes real.exe's own starting number: `make_DropletNumber` for cloud
droplets (`mp_physics = 28`, recorded under the receipt's `droplet_number_seed`
key), `make_RainNumber` and `make_IceNumber` for rain and ice (`rain_number_seed`
and `ice_number_seed`), while an analysed number above zero is kept.
Restart/runtime-only effective radii and Morrison
convective tendencies are listed separately and are not falsely claimed as
required `wrfinput` variables.

The direct NetCDF writer resolves this inventory per domain. For Thompson and
Morrison it adds the corresponding number-moment variables to each
`wrfinput_dNN` and their value/tendency arrays on all four sides of root
`wrfbdy_d01`; the resolved contract digest and per-domain scheme are sealed in
the export manifest. Structural writer tests exercise 35, 49, and 80 mass
levels. An unchanged-stock-WRF launch remains a separate live evidence gate.

The root's specified lateral boundary is a declared divergence from stock
WRF's default. `real.exe` writes water vapour alone among the moist species
to `wrfbdy_d01` (WRF v4.7.1 `main/real_em.F:956`, `:1156`, `:1374`), and
`have_bcs_moist` and `have_bcs_scalar` default to `.false.`
(`Registry.EM_COMMON:2979-2980`), so every other moist species of a WRF run
takes a flow-dependent boundary with zero inflow, and an analysed cloud or
snow field drains out through the edges. WOOF-ArWen's forecast carries every
hydrometeor mass the source publishes on each forcing time: the canonical
hydrometeor fields its mapping declares, or a native row's
`boundary_species` column (`hrrr`, `hrrr-prs` and `icon-d2` today), with the
number moments the Thompson cold start seeds from them on the same forcing
time, and treats them as WRF does with both switches on: relaxed and
specified on the ring, the masses forced back at the end of each step
(`solve_em.F:2265-2267`, `:2346`, `:4701-4703`), the numbers moving by
their boundary tendency. A source that publishes none keeps water vapour
only. The exported `wrfbdy_d01` keeps `real.exe`'s water-vapour-only moist
boundary, so a CPU WRF run from the export starts as `real.exe` would.

The accepted companion state is presently YSU (`bl_pbl_physics=1`), classic
MM5 surface layer (`sf_sfclay_physics=91`), four-layer Noah
(`sf_surface_physics=2`, `num_soil_layers=4`), no urban state, nests whose
footprints are fixed for the run, and Lambert conformal, Mercator, or polar
stereographic geometry. Other choices fail with the exact missing
initialized-state adapter rather than being changed.

`feedback` and `smooth_option` are read from the engine's own validator
rather than from a table in this door. `feedback=0` is the certified one-way
path and `feedback=1` (WRF's Registry default, so also what an omitted key
selects) is reported as the experimental two-way path without failing the
report. `smooth_option` 0, 1 (`sm121`) and 2 (`smdsm`, WRF's Registry
default) are all implemented and are reported, not refused; WRF reads the key
only while `feedback = 1` and so does this product. A value the validator
rejects still fails, and the message names the validator and the set it
admits. WRF's moving-nest keys are still refused, in the loader's own
words: the specified-move keys (`num_moves`, `move_id`, `move_interval`,
`move_cd_x`, `move_cd_y`, `time_to_move`) are answered with the exact
`[relocation]` rows that reproduce the same itinerary at cycle boundaries,
and the vortex-following keys (`vortex_interval`, `max_vortex_speed`,
`corral_dist`, `track_level`) are refused as having no counterpart.
`tile_sz_x` and `tile_sz_y` are not moving-nest keys and are not refused:
they size the CPU build's shared-memory tiles, reach neither the prepared
state nor the integration, and are reported as a note. The importer
records them as dropped keys beside `numtiles`, `nproc_x` and `nproc_y`,
so both doors take the same namelist.

## Domain and vertical behavior

Domain columns remain in d01...dNN order. WRF's short-array convention is
honored by repeating the final declared value; extra values beyond `max_dom`
are rejected rather than truncated. The stock-export verdict is bounded by
WRF's own compiled `max_domains` (21 in the stock build), because an
unchanged WRF executable cannot read a namelist that declares more domains;
the message names that artefact and both ways out. The geometry, physics and
timing analysis is not bounded by it, so a larger tree is still examined and
still answers the woof runtime verdict. There is an explicit six-domain
regression gate.

`fine_input_stream` has WRF's two defined values and both have a prepared
route: 0 takes every field from the nest's own input, and 2 takes only the
static and masked land-surface fields from it (WRF's delayed-nest-start
pattern), which the stock export satisfies by writing `wrfinput_d0N` at each
domain's configured start time and the runtime satisfies by initializing a
delayed child from its own analysis at activation. The report states that
substitution and the one difference it carries: the masked surface state
comes from the child's own-grid analysis rather than from a real.exe
wrfinput. An index WRF does not define still fails.

The report and `woof.namelist_import` read that answer from one function,
`woof.namelist_import.fine_input_stream_decision`, so a pair this report
passes is a pair the importer imports. The delayed-nest route is booked
there as a declared divergence with the same sentence, which
`announce_wrf_substitutions` prints at the terminal and which the met_em
and wrfinput doors admit by name
(`woof.wrfinput_door.ADMITTED_DECLARED_DIVERGENCES`).
`map_proj='lambert'`, `'mercator'`, and `'polar'` pass. Following WPS
`module_llxy` semantics, Mercator may omit `truelat2` and `stand_lon`, and
polar stereographic may omit `truelat2`; those parameters do not enter the
respective projection math. The Mercator, polar stereographic, and
southern-hemisphere Lambert acceptances carry oracle-verified plus
smoke-run-verified maturity, not matched-run verification. The report
verifies finite projection parameters and positive spacing, exact WPS/WRF
dimension and topology agreement, per-domain spacing implied by each parent
ratio, WRF's `(e_we-1)`/`(e_sn-1)` ratio divisibility, child containment
within its parent, one specified root plus nested children, and
`spec_bdy_width=spec_zone+relax_zone`. Latitude/longitude geometry, invalid
placement, and inconsistent boundary declarations fail with dedicated issue
codes before source data is touched.

Vertical acceptance is structural. `e_vert` must equal
`len(eta_levels)`, eta must decrease exactly from 1 to 0, all domains must
share the coordinate for the current one-way initializer, and p-top must lie
within source coverage. Tests cover 35, 49, and 80 mass levels. A source whose
atmosphere stops below the requested model top is rejected; RW-WPS does not
extrapolate it.

The support report is a preflight/state-inventory gate. A source/domain/scheme
combination is not called end-to-end certified until its native export is
opened by unchanged stock WRF and the resulting run has its separate evidence
receipt.
