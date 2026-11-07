# B4. The observation battery's run routes, qualified

Written 2026-08-03 against `fc15d9ae` (spec `spec/obs-battery`, whose base is
the shipped `release/1.5.2` tip `56768262`). Every claim below was produced by
running the route's own code over the battery's own configs; the machine-readable
answers are the JSON files beside this one, regenerable with:

```
python tools/battery_route_preflight.py \
  --experiment-config configs/battery/shape_3km_thompson_rrtmg_legacy.toml \
  --receipt docs/public/receipts/obsbattery/B4-ARWEN-HRRR-ROUTE.json
python tools/battery_route_preflight.py \
  --experiment-config configs/battery/shape_smoke_3km_thompson_dudhia.toml \
  --receipt docs/public/receipts/obsbattery/B4-ARWEN-HRRR-SIZING-SMOKE.json
```

No forecast was run for this document, on either venue, and it makes no claim
about skill, of any model, against anything.

---

## 1. The WOOF HRRR route: VERIFY-FIRST answer is NO

The battery spec (section 7.2) proposed running each case as a one-domain
experiment tree through `tools/prepared_domain_tree_forecast.py`, and marked
"that route's single-domain + Thompson-suite + `rrtmg_legacy` composition"
VERIFY-FIRST. **The composition is refused today, at three independent gates,
and all three refuse for one reason: the HRRR route pins radiation to
`ra_lw_physics = 0`, `ra_sw_physics = 1`.**

| gate | authority | when it fires | refusal |
|---|---|---|---|
| route-input authoring | `gpuwm/hrrr_route_inputs.py:50-58` `REQUIRED_PHYSICS`, enforced at `:93` | wizard emission, free | `d01 ra_lw_physics=4 (the route requires 0); d01 ra_sw_physics=4 (the route requires 1)` |
| root preparation | `gpuwm/physics_compat.py` `validate_single_domain_physics_profile`, over `SINGLE_DOMAIN_PHYSICS_PROFILES` | **after the HRRR fetch** | no shipped profile matches; the nearest, `thompson-mp8-ysu-mm5-noah-validation-v1`, pins `ra_lw_physics 0 / ra_sw_physics 1 / radt 1.0` |
| tree assembly | `gpuwm/hrrr_hierarchy_direct.py:50-57` `_SUPPORTED_PHYSICS` plus the separate `:554` check | **after the root preparation** | `d01 is outside the certified native HRRR physics slice: {'ra_sw_physics': (4, 1)}`, then `native hierarchy preparation requires ra_lw_physics=0` |

Everything that is not radiation is admitted: geometry, the 49-level ladder,
`dt` 15 s, `mp_physics 8`, `bl_pbl_physics 1`, `sf_sfclay_physics 91`,
`sf_surface_physics 2`, `cu_physics 0`, HRRR grid coverage with its donor
margin, and `validate_run_config`. The finding is one mechanism, not three
problems.

**The fallback the spec named is closed for the same reason.** Section 7.2 calls
widening the benchmark runner's contract "the fallback, not the plan", but the
HRRR *root preparation for the tree route* IS the benchmark runner:
`tools/prepare_hrrr_wrf.py:1388-1394` shells out to
`tools/hrrr_single_domain_benchmark.py --prepare-only --physics-profile ...`.
There is no HRRR root without a shipped profile, and no shipped profile pairs
Thompson with RRTMG.

**A shipped-registry contradiction, independent of the battery.**
`gpuwm/physics_registry_v2.json` lists
`thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1` (radiation component
`rte-rrtmgp`, i.e. the resolved 4/4 pair) under
`runner_routes["tools.prepared_domain_tree_forecast"].source_template_ids.hrrr`,
and `docs/public/PHYSICS.md:876` advertises the prepared-tree row as carrying
"the normal profile family" for HRRR. Both are refused by
`hrrr_hierarchy_direct`. So the tree is inconsistent with itself here at
`fc15d9ae`, and a fix restores the shipped registry's own claim rather than
inventing a capability.

### What the route needs, costed

Landing `Thompson + legacy RRTMG + cumulus off` on the HRRR route is four
edits, one of which is the coordination-sensitive one:

1. `gpuwm/hrrr_route_inputs.py`: move the three radiation keys out of the
   pinned `REQUIRED_PHYSICS` into an admitted-pairs check `{(0, 1), (4, 4)}`,
   and render `ra_lw_physics`/`ra_sw_physics` from the config in
   `render_namelist_input` instead of the current literals.
2. `gpuwm/hrrr_hierarchy_direct.py`: same admitted-pairs treatment for
   `_SUPPORTED_PHYSICS` and the `:554` longwave check. The evidence for this
   being safe is the same evidence the file already published for
   `bl_pbl_physics` at `:102-112`: preparation writes static fields and an
   interpolated initial state, and a sweep of `gpuwm/ingest/`,
   `gpuwm/native_hierarchy.py`, `gpuwm/native_domain_artifacts.py` and
   `tools/prepare_hrrr_wrf.py` for `ra_physics|ra_lw_physics|ra_sw_physics|ra_rrtmg_variant`
   returns **zero** hits. Radiation selects tendencies only the forecast
   computes.
3. `gpuwm/hrrr_hierarchy_direct.py`: `_require_raw_stock_delta` and
   `_compare_stock_experiment` currently define the native/stock pair as
   "`ra_lw_physics` 0 → 1". Under a 4/4 native suite the stock twin is also
   4/4 and that delta collapses; `ghg_input = 0` stays stock-only, and is
   still correct, because `gpuwm/core/rrtmg_legacy.py:21,:122,:721` pins
   `ghg_input=0` on the native side.
4. **A registered profile/template** so the root preparer will prepare it,
   `physics_registry_v2.json`, `SINGLE_DOMAIN_PHYSICS_PROFILES`, the benchmark
   runner's switch table, and `_HRRR_COLD_START_CONTRACT` (which can reuse the
   Thompson entry verbatim: the cold-start species contract is a microphysics
   property, not a radiation one).

Item 4 is a registry edit. The battery spec's own section 9.4 binds registry
edits from the battery and the LES programs to **separate integration waves**,
and a new template carries a maturity label, which is an owner ruling and not
an agent's to invent. **B4 therefore stops here and hands items 1-4 to the lead
as one costed motion** rather than landing a certified-slice widening plus a
registry row in a lane that cannot smoke either tonight.

### What CAN run at battery shape today

`configs/battery/shape_smoke_3km_thompson_dudhia.toml` is the registered
geometry (480 x 400 x 49 = 9.408 M cells, dx 3 km, dt 15 s, 49-level ladder,
Thompson / YSU / classic MM5 / Noah, cumulus off) on the one HRRR suite the
route admits, and it is **ADMITTED by every static gate** including the root
preparer's profile contract. It measures the shape: device peak, seconds per
simulated minute, bytes per wrfout frame. It does not measure the battery's
radiation, and says so in its own header.

---

## 2. The WRF node route (spec section 6.2): exporter parity is viable, with three caveats

The in-tree stock-WRF exporter is `gpuwm/source_cli.py:main`
(`rw-wps` / `gpuwm-wrf-init`, `pyproject.toml:56-57`), with the export itself
in `gpuwm/wrf_direct.py`.

**HRRR source support: yes.** `hrrr` is a `CERTIFIED` adapter
(`gpuwm/source_adapters.py:155`) routed to the native HRRR ingest
(`gpuwm/hrrr_hierarchy_direct.py:28` → `gpuwm.ingest.hrrr`), not through any
mapped/WPS path.

**Physics: the v1.0.x pin is still in the code**, at
`gpuwm/wrf_direct.py:1913-1936` (the profile-free branch) and unconditionally
for every domain of a hierarchy at `:1529-1557`: `bl_pbl_physics = 1`,
`sf_sfclay_physics = 91`, `sf_surface_physics = 2`, plus `hybrid_opt = 2`,
`hypsometric_opt = 2`, `spec_bdy_width = 5`. `docs/public/PHYSICS.md:880-885`
describes v1.1.0 as having "removed that coupling"; what v1.1.0 removed is the
export's power to *kill* a prepared forecast (`StockWrfExportUnsupported` is
catchable, `stock_wrf_export="optional"`), and
`gpuwm/native_hierarchy.py:43-44` says so in the code's own words: "Nothing
about the gate's CONTENT changes here." **For the battery this is harmless:
the battery suite IS the pinned slice.** Radiation is invisible to the
exporter (`ra_*` appears in no `required` dict) so legacy RRTMG is not an
obstacle on this side.

**Field completeness for Thompson + Noah + RRTMG: complete.** `mp_physics = 8`
is declared in `gpuwm/wrf_physics_inventory.py:201-210`, and `QNICE`/`QNRAIN`
are written to `wrfinput` and given their `_B*`/`_BT*` boundary arrays
(`gpuwm/wrf_direct.py:244-245,:292-308,:1212-1220`), zero-filled when the
analysis carries no number concentrations, which is stock `real.exe`'s own
Registry `i0` policy. Soil (`TSLB`, `SMOIS`, `SH2O`, `TMN`, `ZS`, `DZS`),
masks (`SEAICE`, `XLAND`, `LANDMASK`, `IVGTYP`, `ISLTYP`, `LU_INDEX`, `SST`,
`SNOW`, `SNOWH`, `SNOWC`) and the radiation/vegetation statics (`ALBBCK`,
`SNOALB`, `VEGFRA`, `SHDMAX/MIN/AVG`, `LAI`, `LANDUSEF`, `SOILCTOP/BOT`) are
all present. Eighteen contract variables land as zeros; for this suite none is
fatal, and the two worth naming are `CANWAT` (Noah canopy water) and
`O3_GFS_DU` (immaterial unless the WRF namelist sets `o3input` to read it,
which this exporter neither reads nor writes).

### Caveat 1. Frozen soil is a hard refusal

`gpuwm/wrf_direct.py:1007-1009`:

```
if np.any(soilt < 273.15):
    raise StockWrfExportUnsupported(
        "direct WRF export does not yet support frozen-soil SH2O setup")
```

This fires on the branch that synthesizes `SH2O` from `SMOIS` when the
prepared cache carries no `surface/*` group. **Any cold-season case is exposed**
(the spec's own proposed B-01 is a December event) and the exposure must be
settled per case at its entry receipt, not discovered on the node. The HRRR
route restores a real surface group (`gpuwm/hrrr_hierarchy_direct.py:726-746`),
which is the path that avoids the synthesis, but that is a property of the
prepared root and belongs in the case receipt as a checked fact.

### Caveat 2. The boundary cadence is hourly, not 3-hourly

Spec section 2.4 plans "LBC donor files at 3-h interval ... ~1.0-1.5 GB/slot x
9 slots ≈ 10-15 GB per case". The HRRR route does not offer that cadence:
`gpuwm/hrrr_route_inputs.py:40` fixes `FORCING_INTERVAL_SECONDS = 3600`, the
raw-WPS and raw-runtime contract gates pin the integer `3600`, and
`gpuwm/hrrr_hierarchy_direct.py:949` passes `boundary_interval_seconds=3600`
to the export. A 24 h battery case therefore seals **f00..f24, 25 analyses**,
not 9. That is better forcing and worse arithmetic: **~25-37 GB of staged HRRR
per case instead of 10-15**, and the section 8.3 storage table needs the
correction. The preflight prints the sealed inventory for any config
(`hrrr.forcing_inventory`).

### Caveat 3. No stock-WRF receipt exists for Thompson

Every artifact in the tree that binds exporter output to an actual `wrf.exe`
run is WSM6: `gpuwm/source_adapters.py:555-636` (the GFS d01-d04 hierarchy
evidence, with `wrfinput_d0N`/`wrfbdy_d01` digests against a pinned
`wrf_exe_sha256` and the resulting `wrfout` digests) and `PROVENANCE.md:53-63`
(the GFS d01 direct adapter). The HRRR and ERA5 `stock_wrf_gate` strings in
`source_adapters.py:159-161,:346-348` are free text that occurs nowhere else in
the repository, they are **not** hash-bound receipts. So section 6.2's
"exporter-parity as default" is a reasonable default with WSM6 evidence and
**no mp=8 evidence**; the first battery case's t0 receipt is where that gets
paid for, and it should be treated as a first-use risk rather than a proven
path.

---

## 3. Verified seeds

**Seed 1: CONFIRMED, not on the battery's path.**
`gpuwm/hrrr_route_inputs.py:100-113` requires `bl_pbl_physics == 1` on **every**
domain, while `gpuwm/hrrr_hierarchy_direct.py:102-112`
(`_CHILD_PHYSICS_SLICE_OVERRIDES`) exempts exactly `bl_pbl_physics` on a child,
with a published reason ("pinning it here refused the PBL-off LES child that
the experiment schema, the forecast runner and the namelist importer all
admit"). The emission-time validator is therefore stricter than the gate it
exists to anticipate. Battery v1 is single-domain, so this blocks nothing here;
the fix is to give `validate_route_physics` the same child exemption, one
frozenset and one `if`.

**Seed 2: CONFIRMED, and it is a run-siting constraint.**
`gpuwm/hrrr_hierarchy_direct.py:715-722` refuses when
`identity["git_status_short"]` is non-empty, and that value comes from
`gpuwm/runtime_manifest.py:414` `git status --short`, **without**
`--untracked-files=no`, so a single untracked file anywhere in the checkout
refuses the preparation. `gpuwm/gfs_direct.py:250-251` asks the same question
with `--untracked-files=no`. Consequences for the campaign, both real: every
battery config must be **committed** before its root is prepared, and all run
scratch must live **outside** the worktree. The ready-to-smoke commands below
assume both. Whether to relax it is a lead ruling; note that the strict form is
also what currently prevents seed 3 on this route.

**Seed 3: CONFIRMED as a provenance gap, mitigated here.**
`tools/prepared_domain_tree_forecast.py` takes `--experiment-config` and
`--experiment-config-sha256` and binds the **content** digest of whatever file
the operator passed (`gpuwm/prepared_domain_tree_forecast.py:827-836`). Nothing
checks that the file is a committed blob, so a node-side copy that never existed
in git produces a receipt that looks fully bound. B4's mitigation is procedural
and cheap: the battery's configs live in `configs/battery/` and are committed
before use, and seed 2's clean-tree refusal enforces that on the preparation
side. A mechanical fix (recording the config's git blob sha and its
reachability from a commit in the run receipt) is a small addition to the tree
runner's receipt and is filed, not done.

---

## 4. Ready to smoke

Both venues were occupied when this was written (the 5090 by the LES P3 lane,
test host 1 by LES P1), so nothing below has been executed. Run in order.

### Venue A. The RTX 5090, WOOF sizing smoke at battery shape

> **Corrected 2026-08-04 after the rented-node smoke** (see the section 6
> addendum): step 1's original command passed `--cycle`, `--wps-namelist` and
> `--stock-wrf-namelist-input` to the root preparation, which refuses all
> three (they belong to step 2's hierarchy export; the cycle is spelled
> `--valid-time` here), and it omitted the required
> `--source-manifest`/`--source-manifest-sha256` pair a prior `gpuwm fetch`
> produces.  Step 3 originally named the tree runner, which carries a
> designed two-domain floor and refuses every single-domain battery config;
> the single-domain run step is the benchmark runner's forecast mode.  The
> preflight now names both facts (`run.runner_selection`).

Prerequisites: this branch is **committed and the worktree is clean** (seed
2), `$SCRATCH` is a directory **outside** any git worktree, and the mp8
launch guards are set for every step that integrates (the profile's runtime
guard and table root; preparation does not need them, the forecast refuses
without them):

```
export GPUWM_EXPERIMENTAL_THOMPSON_MP8=1
export GPUWM_THOMPSON_TABLE_ROOT=<thompson lookup-table root>
```

```
# 0a. free-of-charge, seconds: confirm every static gate still admits it,
#     and read run.runner_selection -- it names the run-step runner for
#     this config's domain count.
python tools/battery_route_preflight.py \
  --experiment-config configs/battery/shape_smoke_3km_thompson_dudhia.toml

# 0b. author the four route input files from that one config, so the
#     preparation, the tree and the WRF arm all read one geometry authority
python tools/battery_wrf_node_plan.py \
  --experiment-config configs/battery/shape_smoke_3km_thompson_dudhia.toml \
  --outdir $SCRATCH/smoke --ranks 24
#     -> $SCRATCH/smoke/{namelist.native.input, namelist.input,
#        namelist.wps, d01-target.json}
#     namelist.native.input is the gpuwm side; namelist.input is the stock
#     twin; they differ only by the route's three documented deltas.

# 1a. fetch the HRRR window (~2-3 GB for f00..f01); writes SHA256SUMS and
#     fetch-manifest.json into --out, which are the manifest pair every
#     later stage binds.
gpuwm fetch --source hrrr --cycle 2026-07-28T12 --hours 1 \
  --out $SCRATCH/hrrr

# 1b. prepare the HRRR root.  Rehearse with --dry-run first (free; prints
#     the route the adapter resolves to).  The cycle is --valid-time on
#     this command; the wps and stock namelists are NOT accepted here --
#     they belong to step 2.
python -m gpuwm.source_cli --source hrrr --dry-run \
  --valid-time 2026-07-28_12:00:00 --forecast-start-hour 0 \
  --forecast-end-hour 1 \
  --physics-profile thompson-mp8-ysu-mm5-noah-validation-v1 \
  --namelist-input $SCRATCH/smoke/namelist.native.input \
  --domain-spec    $SCRATCH/smoke/d01-target.json \
  --geog-root <geog> --source-root $SCRATCH/hrrr \
  --source-manifest $SCRATCH/hrrr/SHA256SUMS \
  --source-manifest-sha256 <sha256 of that file> \
  --run-seconds 3600 --history-interval-seconds 3600 \
  --output-root $SCRATCH/smoke/rootprep
#    then re-run without --dry-run

# 2. assemble the one-domain tree.  For a single-domain battery case this
#    step exists for the WRF arm -- it publishes wrfinput/wrfbdy and the
#    per-domain artifact with the canonical surface/* group -- and its
#    product is NOT the WOOF run input.  The manifest pair is the same
#    one step 1b was given.
python -m gpuwm.hrrr_hierarchy_direct \
  --root-preparation $SCRATCH/smoke/rootprep \
  --root-domain-spec $SCRATCH/smoke/d01-target.json \
  --wps-namelist     $SCRATCH/smoke/namelist.wps \
  --namelist-input   $SCRATCH/smoke/namelist.native.input \
  --stock-wrf-namelist-input $SCRATCH/smoke/namelist.input \
  --geog-root <geog> \
  --source-manifest $SCRATCH/hrrr/SHA256SUMS \
  --source-manifest-sha256 <same sha as step 1b> \
  --cycle 2026-07-28_12:00:00 --forecast-start-hour 0 \
  --output-root $SCRATCH/smoke/tree

# 3. run it, twice (the standing dual-run byte screen; no ECC on this
#    card).  Single-domain configs run through the benchmark runner's
#    forecast mode, restoring the root preparation's own prepared cache;
#    the tree runner refuses them by design.  --bridge and the two
#    manifest digests are the preparation's own artifacts:
#    bridge_manifest_sha256 sits in
#    $SCRATCH/smoke/rootprep/native/prepared-cache/header.json (identity).
python tools/hrrr_single_domain_benchmark.py \
  --bridge $SCRATCH/smoke/rootprep/native/native-bridge \
  --manifest-sha256 <bridge_manifest_sha256 from the prepared-cache header> \
  --cycle 2026-07-28_12:00:00 --forecast-start-hour 0 \
  --forecast-end-hour 1 \
  --physics-profile thompson-mp8-ysu-mm5-noah-validation-v1 \
  --source-manifest-sha256 <same sha as step 1b> \
  --static-cache $SCRATCH/smoke/rootprep/native-static.npz \
  --static-receipt $SCRATCH/smoke/rootprep/native-static-receipt.json \
  --namelist-input $SCRATCH/smoke/namelist.native.input \
  --domain-spec    $SCRATCH/smoke/d01-target.json \
  --prepared-cache $SCRATCH/smoke/rootprep/native/prepared-cache \
  --run-seconds 3600 --io-mode history --history-interval-seconds 3600 \
  --outdir $SCRATCH/smoke/runA
#    repeat with --outdir $SCRATCH/smoke/runB and byte-compare the wrfouts
```

Expected, from `B4-SPEED-ANCHORS.json` (all superseded by what this measures):

| quantity | expectation | source of the expectation |
|---|---|---|
| wall clock, 1 h forecast | **1.5-3.6 min** | 0.55-1.48 s/sim-min at 9.408 M cells |
| device pool request | **7.24 GiB** | `gpuwm.core.preflight` for this exact config |
| device peak, observed | **9.6-10.9 GiB** | the committed 0.75-0.84 KiB/cell + 2.9-3.4 GiB fit |
| wrfout | **2 frames, ~1.3 GiB** | 0.69 GB/frame scaled from the 3 km anchor |
| HRRR staged | **~2-3 GB** (f00, f01) | `hrrr.forcing_inventory` |

Add **1.85 GiB of pool** to the memory line for the battery's real radiation:
that is legacy RRTMG's cost over Dudhia at this shape, itemized by the same
estimator.

### Venue B. Node 1 or 2, the WRF arm

```
# 3. the node bundle is venue A step 0b -- the same command, the same files.
#    plan.json and launch_arm.sh beside them are the WRF arm's.

# 4. export wrfinput/wrfbdy for WRF from the same prepared root
#    (rw-wps / gpuwm-wrf-init; hierarchy export publishes wrfinput_d01 +
#    wrfbdy_d01 atomically)

# 5. the twin, rung 1: exactly one documented FP ULP on wrfinput theta
python tools/n5s/perturb_ulp.py \
  $SCRATCH/node/control/wrfinput_d01 $SCRATCH/node/twin/wrfinput_d01 \
  --field T --member-ordinal 0 --record $SCRATCH/node/twin/perturbation.json

# 6. launch each arm detached (tmux), 24 ranks
bash $SCRATCH/smoke/launch_arm.sh control $SCRATCH/node/control $WRF_BUILD
bash $SCRATCH/smoke/launch_arm.sh twin    $SCRATCH/node/twin    $WRF_BUILD
```

Expected, at 24 ranks:

| quantity | 1 h smoke | 24 h battery case |
|---|---|---|
| projected rate | 8.78 s/sim-min | 8.78 s/sim-min |
| wall clock per arm | **~8.8 min** | **~3.5 h** |
| control + twin | ~18 min | **~7.0 h** |
| detached launch | not required | **required** (past the 55 min window) |

The rate is scaled linearly in cells from an **unreceipted controller
measurement** (14.0 s/sim-min at 24 ranks and 15 M cells, 2026-08-03). Parsing
the smoke's own `rsl.error.0000` timing lines into a receipt beside
`B4-SPEED-ANCHORS.json` is what upgrades it, and is the cheapest thing on this
list.

**Do not rebuild WRF.** The pinned campaign build is the arm; the reference
bundle is not campaign source state (patched `real` source, regenerated
`SOILHGT == HGT_M`), so an arm that cannot find the pinned build stops rather
than compiling one.

---

## 5. Open to the lead

1. **The four-item route motion in section 1.** Items 1-3 are contained; item 4
   is a registry row with a maturity label and its own integration wave. Ruling
   needed before any battery case can run its faithful arm.
2. **Section 2.4's boundary arithmetic** needs the hourly correction: 25 HRRR
   analyses per 24 h case, ~25-37 GB staged, not 9 slots and 10-15 GB.
3. **Frozen-soil exposure** belongs in the per-case entry receipt as a checked
   fact, and it interacts with the cold-season cases the menu proposes.
4. **Seed 2's clean-tree rule**: relax to `--untracked-files=no` (matching
   `gfs_direct`), or keep it strict and site all run scratch outside worktrees?
   The commands above assume the strict form.

---

## 6. Addendum: integration wave, 2026-08-04

Sections 1-5 above are the dated qualification against `fc15d9ae` and are left
as written. The obs-battery integration wave then landed **route-motion items
1-3 exactly as costed in section 1**:

* `gpuwm/hrrr_route_inputs.py`: the three radiation keys left
  `REQUIRED_PHYSICS` for the admitted-pairs check
  `ADMITTED_RADIATION_PAIRS = {(0, 1), (4, 4)}`, and
  `render_namelist_input` renders `ra_lw_physics`/`ra_sw_physics` from the
  config (stock substitutes longwave 1 exactly where the native entry is 0).
* `gpuwm/hrrr_hierarchy_direct.py`: the same pair treatment for the
  certified slice (`_ADMITTED_RADIATION_PAIRS`, with `ra_physics = 0` still
  pinned beside the explicit pair), replacing the flat `ra_lw_physics = 0`
  check.
* `gpuwm/hrrr_hierarchy_direct.py`: `_require_raw_stock_delta` and
  `_compare_stock_experiment` treat the longwave delta as the per-domain
  0 -> 1 substitution, so under a (4, 4) native suite the delta collapses;
  `ghg_input = 0` stays stock-only.

`B4-ARWEN-HRRR-ROUTE.json` beside this file is regenerated: the emission and
slice gates now ADMIT the section 1.3 composition, and the **only** refusing
gate is `hrrr.root_preparation.profile`, item 4, the registered
profile/template with its maturity label, which remains the lead's ruling and
is NOT landed by the wave. The sizing-smoke receipt is byte-unchanged.

One seam found while landing, for the item-4 ruling to note: the tree route
reads its physics from the profile named at root preparation and from
namelists imported with the importer's default `rrtmg_variant`, so a faithful
arm that must run **legacy** RRTMG (not RTE+RRTMGP) needs the item-4
profile/template to carry that selection, items 1-3 admit the 4/4 pair
through the static gates but do not, and cannot, choose the variant.

**Item 4, landed by ruling (2026-08-04, same wave).** The lead ruled the
registry step in: `thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1`
(wrf-matched-run-candidate; the composition's first stock-WRF-paired t0/case
receipt is the named upgrade payer; per-domain row transcribed from the
battery config, ladder divergence noted in the row), HRRR-only route
registration on the Kessler rule, the physics_compat profile row, and the
cold-start contract reused from Thompson verbatim. The variant seam above was
resolved verify-first: the route now imports its namelists under the sealed
root's recorded `ra_rrtmg_variant`, and the verification also surfaced a
radiation SPELLING seam (profile-explicit `(0, 4, 4)` vs importer-aggregate
`(4, -1, -1)`), canonicalized comparison-only in
`effective_prepared_domain_config` and pinned with controls.
`B4-ARWEN-HRRR-ROUTE.json` is regenerated once more: **ADMITTED, zero
refusing gates**, the profile gate naming the new profile.

**The rented-node smoke falsified section 4's untested commands
(2026-08-04, later the same day).** A full-route validation smoke on a
provisioned node ran section 4 as written: the preflight ADMITTED, the fetch
and preparation and tree build all PASSED, and the run step was refused --
`gpuwm.prepared_domain_tree_forecast` carries a designed two-domain floor
("it does not flatten a nest tree into the single-domain benchmark runner",
its own docstring), so every single-domain battery config is admitted by the
static gates and refused by that runner by construction.  Three corrections,
all applied in section 4 above (the original commands were marked "nothing
below has been executed", and this is the price):

1. **The run step for a single-domain config is
   `tools/hrrr_single_domain_benchmark.py` forecast mode**, restoring the
   root preparation's own prepared cache; the profile's every switch --
   `ra_rrtmg_variant` included -- reaches that runner's RunConfig through
   `_forward_profile_switches`, pinned per profile by
   `tests/test_hrrr_single_domain_benchmark.py`.  The tree runner's floor is
   NOT widened; the preflight's new `run.runner_selection` gate names the
   refusal and the right runner per domain count, before a node is booked.
2. **Step 1's flags**: the root preparation takes `--valid-time` (the cycle),
   refuses `--cycle`/`--wps-namelist`/`--stock-wrf-namelist-input` (hierarchy
   flags), and requires the `--source-manifest`/`--source-manifest-sha256`
   pair a prior `gpuwm fetch` writes.
3. **The mp8 launch guards** (`GPUWM_EXPERIMENTAL_THOMPSON_MP8=1`,
   `GPUWM_THOMPSON_TABLE_ROOT`) are documented in the prerequisites; the
   forecast refuses without them and no earlier draft said so.
