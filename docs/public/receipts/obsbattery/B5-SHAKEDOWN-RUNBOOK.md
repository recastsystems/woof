# B5. The shakedown run book

2026-08-04, against `0279f0f2`. Both venues were occupied when this was written
(the 5090 and node 1 by the LES program), so **nothing below has been executed**
except where a step says it was. Everything that could be done without either
venue has been: the observations are pulled and hash-pinned, the case config is
committed and ADMITTED by every static gate, the qualification controls that
need no forecast are measured, and the scoring command now exists and its
reading half has been exercised against a real WRF v4.6.1 `wrfout`.

Run in order. Steps 1-6 are the WOOF arm on the card; steps 7-10 are the WRF
arm on a node and are deferred, not blocked.

---

## 0. Preconditions

* **The branch is committed and the worktree is clean.** B4 seed 2 is still
  live: `gpuwm/hrrr_hierarchy_direct.py` refuses preparation when
  `git status --short` is non-empty, and it asks **without**
  `--untracked-files=no`, so one stray untracked file anywhere in the checkout
  refuses the run.
* **`$SCRATCH` is outside every git worktree.** Same reason. The observation
  cache already lives outside, at
  `~/gpuwm/cache/obsbattery/battery`.
* **The GPU is authorized and free.** No step here shares the card with a
  timing run; the dual-run pair is bursty and may split across authorized
  cards, a long timing run may not.
* Environment: `set PYTHONDONTWRITEBYTECODE=1` so no `__pycache__` lands in the
  clean tree the preparer is about to check.
* **The mp8 launch guards are set.**  The profile's runtime guard and lookup
  tables gate the FORECAST step (preparation runs without them; the forecast
  refuses, by name, without them):

  ```
  export GPUWM_EXPERIMENTAL_THOMPSON_MP8=1
  export GPUWM_THOMPSON_TABLE_ROOT=<thompson lookup-table root>
  ```
* **The file-descriptor limit is raised** in the shell that launches the run.
  The nodes default to a soft limit of 1024 and the 24 h window's stock-WRF
  export dies there on EMFILE (measured 2026-08-04; the export step now
  refuses loudly instead of warning past it, and the preflight's
  `run.fd_budget` gate prices it):

  ```
  ulimit -n 4096
  ```

```
CFG=configs/battery/case_20240521.toml
CYCLE=2024-05-21_12:00:00
OBS=~/gpuwm/cache/obsbattery/battery
```

## 1. Free, seconds: re-ask every static gate

```
python tools/battery_route_preflight.py --experiment-config $CFG
```

**Measured 2026-08-04: `status: ADMITTED`, zero refusing gates**, including
`hrrr.coverage` for this case's own Iowa centre, and
`hrrr.root_preparation.profile` naming
`thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1`. Re-run it anyway: it is free and it
quotes each gate verbatim from the route's own code.

## 2. Author the route inputs from the one config

```
python tools/battery_wrf_node_plan.py --experiment-config $CFG \
  --outdir $SCRATCH/case --ranks 24
```

**Measured 2026-08-04** (into `$OBS/nodeplan`, outside the worktree): writes
`namelist.native.input`, `namelist.input`, `namelist.wps`, `d01-target.json`,
`plan.json`, `launch_arm.sh`. The native and stock namelists come from one
renderer, so a physics switch cannot differ between the arms by hand.

## 3. Fetch and prepare the HRRR root

> **Corrected 2026-08-04 after the rented-node smoke** (B4 section 6): the
> original command passed `--cycle`, `--wps-namelist` and
> `--stock-wrf-namelist-input`, all refused by the root preparation (the
> cycle is spelled `--valid-time` here; the two namelists belong to step 4),
> and it omitted the required manifest pair a prior `gpuwm fetch` writes.

```
# 3a. fetch the window; writes SHA256SUMS + fetch-manifest.json into --out
gpuwm fetch --source hrrr --cycle 2024-05-21T12 --hours 24 \
  --out $SCRATCH/hrrr

# 3b. prepare the root (rehearse with --dry-run first, it is free)
python -m gpuwm.source_cli --source hrrr --dry-run \
  --valid-time $CYCLE --forecast-start-hour 0 --forecast-end-hour 24 \
  --physics-profile thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1 \
  --namelist-input $SCRATCH/case/namelist.native.input \
  --domain-spec    $SCRATCH/case/d01-target.json \
  --geog-root <geog> --source-root $SCRATCH/hrrr \
  --source-manifest $SCRATCH/hrrr/SHA256SUMS \
  --source-manifest-sha256 <sha256 of that file> \
  --run-seconds 86400 --history-interval-seconds 3600 \
  --output-root $SCRATCH/case/rootprep
#   then re-run without --dry-run
```

Sizes, from the preflight's own `hrrr.forcing_inventory` and the errata's
corrected arithmetic (`WAVE-ERRATA-20260804.md` section 3, which supersedes
spec section 2.4's 3-hourly plan):

| quantity | expect |
|---|---|
| HRRR analyses sealed | **25** (f00..f24, hourly: the route pins `FORCING_INTERVAL_SECONDS = 3600`) |
| staged HRRR | **~25-37 GB** |
| free disk before starting | ≥ 80 GB |

## 4. Assemble the one-domain tree, for the WRF arm, not the WOOF run

For a single-domain case this step exists for steps 7-10: it publishes
`wrfinput`/`wrfbdy` and the per-domain artifact with the canonical
`surface/*` group.  Its product is **not** the WOOF run input (step 5 runs
from the root preparation's own prepared cache).  The manifest pair is the
same one step 3b was given.

```
python -m gpuwm.hrrr_hierarchy_direct \
  --root-preparation $SCRATCH/case/rootprep \
  --root-domain-spec $SCRATCH/case/d01-target.json \
  --wps-namelist     $SCRATCH/case/namelist.wps \
  --namelist-input   $SCRATCH/case/namelist.native.input \
  --stock-wrf-namelist-input $SCRATCH/case/namelist.input \
  --geog-root <geog> \
  --source-manifest $SCRATCH/hrrr/SHA256SUMS \
  --source-manifest-sha256 <same sha as step 3b> \
  --cycle $CYCLE --forecast-start-hour 0 \
  --output-root $SCRATCH/case/tree
```

**Then run the frozen-soil check against the artifact, not against the
calendar** (B5-CASE-ENTRY-20240521.md section 4):

```
python -c "import numpy as np; from gpuwm.ingest.prepared_cache import PreparedCacheReader; \
  c=PreparedCacheReader(r'$SCRATCH/case/tree/<domain>/prepared'); \
  print(sorted(n for n in c._arrays if n.startswith('surface/'))); \
  print('min soil T', float(np.min(c.read_array('surface/TSLB'))))"
```

Expect the ten canonical `surface/*` names present. Their presence is what
makes the exporter read `SH2O` instead of synthesizing it, which is the branch
the frozen-soil refusal lives on; record both the name list and the minimum
soil temperature in the run receipt either way.

## 5. Run it twice: the dual-run byte screen

> **Corrected 2026-08-04 after the rented-node smoke**: the original step
> named `tools/prepared_domain_tree_forecast.py`, which refused it,
> `prepared domain-tree runner requires at least two domains`, a designed
> division of labor ("it does not flatten a nest tree into the
> single-domain benchmark runner", its own docstring) that is not widened.
> A single-domain config runs through the benchmark runner's forecast mode,
> restoring the root preparation's own prepared cache; the preflight's
> `run.runner_selection` gate now says so per config.  The profile reaches
> this runner on TWO separate paths, both pinned per profile in
> `tests/test_hrrr_single_domain_benchmark.py`: the switch forward
> (`_forward_profile_switches`, `ra_rrtmg_variant = rrtmg_legacy` included)
> and the per-profile contract tables (`_native_hrrr_profile_contract` and
> the species/cold-start tables) -- the first case run was refused on the
> second path alone, which the switch pin does not reach, and the
> completeness walk now resolves every shipped profile through every
> per-profile table.

```
# bridge_manifest_sha256 sits in the prepared cache's own header:
#   $SCRATCH/case/rootprep/native/prepared-cache/header.json -> identity
python tools/hrrr_single_domain_benchmark.py \
  --bridge $SCRATCH/case/rootprep/native/native-bridge \
  --manifest-sha256 <bridge_manifest_sha256 from that header> \
  --cycle $CYCLE --forecast-start-hour 0 --forecast-end-hour 24 \
  --physics-profile thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1 \
  --source-manifest-sha256 <same sha as step 3b> \
  --static-cache $SCRATCH/case/rootprep/native-static.npz \
  --static-receipt $SCRATCH/case/rootprep/native-static-receipt.json \
  --namelist-input $SCRATCH/case/namelist.native.input \
  --domain-spec    $SCRATCH/case/d01-target.json \
  --prepared-cache $SCRATCH/case/rootprep/native/prepared-cache \
  --run-seconds 86400 --io-mode history --history-interval-seconds 3600 \
  --outdir $SCRATCH/case/runA
#   repeat with --outdir $SCRATCH/case/runB, then byte-compare the wrfouts
```

> **Amended 2026-08-04 after the first off-node scoring attempt**: the
> `gpuwm dual-run --capsule-a/--capsule-b` step this section used to end on
> is unreachable from this route. **The benchmark runner writes no
> certification capsule**: its `--outdir` publishes `report.json`,
> `progress.json` and the `wrfout_d*` frames, and nothing else
> (`tools/hrrr_single_domain_benchmark.py`, `_PUBLISHED_OUTPUT_NAMES` and
> `_PUBLISHED_OUTPUT_GLOB`); capsules come from the `gpuwm.certify` path that
> `gpuwm/prepared_single_domain_forecast.py` and the supervisor run on.

**The sanctioned screen for a capsule-less runner is the per-frame digest
comparison**, and it is what was performed and accepted here. The runner
already takes a SHA-256 of every published frame, with the atomic writer's
own readback verified, into `report.json` at
`gridded_output.files[*].sha256`, so the screen compares two receipts and
re-reads no wrfout:

```
python - <<'PY'
import json, pathlib
def frames(run):
    report = json.loads((pathlib.Path(run) / "report.json").read_text())
    return {pathlib.Path(row["path"]).name: row["sha256"]
            for row in report["gridded_output"]["files"]}
a, b = frames("$SCRATCH/case/runA"), frames("$SCRATCH/case/runB")
assert a and set(a) == set(b), (sorted(a), sorted(b))
differing = sorted(name for name in a if a[name] != b[name])
print(f"{len(a) - len(differing)} of {len(a)} frames byte-identical")
print("differing:", differing or "none")
PY
```

The verdict is the same one the capsule comparison would have returned; what
changed is where the digests are read from. **For a runner that does publish
a capsule** (the `gpuwm.certify` route) the capsule comparison stays the
screen, and it carries the run identity as well as the bytes:

```
gpuwm dual-run --capsule-a $SCRATCH/case/runA/<capsule> \
               --capsule-b $SCRATCH/case/runB/<capsule> \
               --out-report $SCRATCH/case/dualrun.json
```

Expect, from the preflight's own projection for this exact config (every one
superseded by what the run measures):

| quantity | expectation | source |
|---|---|---|
| wall clock, 24 h forecast | **0.59 h** at 1.4838 s/sim-min | preflight `arwen.wall_clock_projection` |
| device pool request | **9.144 GiB** | `gpuwm.core.preflight` for this config |
| device peak envelope | **18.43 GiB** | same |
| `wrfout` | **25 frames, 16.1 GiB** | 0.69 GB/frame scaled from the 3 km anchor |
| both runs + prep, resident | ~70 GB with the staged HRRR | errata section 3 |

The card carries no ECC. The byte pair is the corruption detector, and a
non-identical pair is a hardware finding, not a model finding.

## 6. Score it

The scoring command did not exist before this lane; `gpuwm/verify/obs/` shipped
the engine, the registration discipline and the readers as libraries with
nothing wiring them together. `tools/obs_battery_score.py` is that wiring.

**Its reading half is already exercised, on a real WRF v4.6.1 `wrfout`**
(a Thompson 500 m all-domains run of 2026-07-22, 13 frames,
200 x 250 at dx 12 km):

```
python tools/obs_battery_score.py --run-directory <run> --domain d01 --read-only
```

```
frames 13, 1974-04-03T12:00:00 .. 1974-04-04T00:00:00
grid 200x250, dx 12000.0 m, terrain -0.45 .. 1511.53 m, land fraction 0.7132
column-max REFL_10CM at the middle frame: -35.0 .. 50.13 dBZ over 50000 cells
ll_to_xy of the grid centre -> x 125.0, y 100.0   (the centre index, exactly)
```

That last line is a projection round trip through the mandated science core,
and it is the check that would catch an off-by-one before it silently moved
every station by one cell.

**The scored inputs are already decoded and waiting**, so step 6 moves no bytes
over a network:

| input | where | what is there |
|---|---|---|
| MRMS packs, F00-F18 | `$OBS/packs/2024-05-21` | 19 frames + geometry, `observed_fraction` 1.0000 on every one |
| Stage-IV packs | `$OBS/packs/stage4-2024-05-21` | 25 hourly + 5 six-hourly + geometry, decoded 2026-08-04 |
| ASOS | `$OBS/asos/2024-05-21` | 642-station frozen table, 13 816 reports over 25 valid times |

The scored command, once the run exists:

```
# --run-directory points at the runner's wrfout SUBDIRECTORY: the
# benchmark forecast writes its frames one level down, <outdir>/wrfout
# (measured on the node; the runner publishes them there by construction).
python tools/obs_battery_score.py \
  --run-directory $SCRATCH/case/runA/wrfout --domain d01 \
  --case-id case-20240521 --arm-id F --init-time 2024-05-21T12:00:00 \
  --reflectivity-packs $OBS/packs/2024-05-21 \
  --precipitation-packs $OBS/packs/stage4-2024-05-21 \
  --asos-stations $OBS/asos/2024-05-21/stations.json \
  --asos-surface  $OBS/asos/2024-05-21/surface.json \
  --boundary-width-cells 5 \
  --registration docs/public/receipts/obsbattery/registration/REGISTRATION-v2-ratified-under-delegation.json \
  --score-out    $SCRATCH/case/score-F.json
```

> **Amended 2026-08-04 after the first off-node scoring attempt.** Three of
> that command's defaults moved out of the tool and into the registration the
> score is bound to, and one flag is new:
>
> * `--precipitation-window-hours` takes a list and defaults to **every
>   window the registration scores** (`1,6` here). The single `6` this step
>   used to pass left the ratified 1 h accumulation without a source, and the
>   engine refused (correctly) with *"the registration scores a 1 h
>   accumulation and no source was supplied for it"*. Both windows are
>   decoded in the kit already (25 hourly + 5 six-hourly).
> * `--reported-lead-hours` defaults to the document's own spin-up leads:
>   with F02-F18 scored, that is **F01 alone**. The old hardcoded
>   `19,20,21,22,23,24` asked for leads past the end of the MRMS objects
>   (which stop at F18) and the reader refused on a frame 3560 s from the
>   nearest one it had. **Lead 0 is refused outright**: the t=0 history frame
>   is the initial condition written back out and carries no `REFL_10CM`.
> * **Frame selection now carries the registered coverage floor**
>   (amendment v2.1, `registration/REGISTRATION-v2.1-amendment-frame-coverage.md`):
>   inside the unchanged ±240 s tolerance the scored frame is the nearest one
>   observing at least 0.9 of the packed subdomain, and the command prints the
>   floor it read from the document. A lead whose whole window is below it is
>   recorded missing-obs in `reflectivity.excluded_leads` and left out of the
>   lead mean: check that list on every score, it is empty for a clean kit.
>   For a case day with an upstream outage, **the kit must decode every frame
>   inside the tolerance**, not only the one nearest the hour, or the rule has
>   nothing to select.
> * `--obs-archive-root` is new, and is needed **only when the archive is not
>   where it was fetched to**. Every observation's provenance records an
>   absolute path on the box that pulled it, and the integrity clause
>   re-hashes all of them; without a root, scoring only runs on that one box.
>   Pass it once per directory holding relocated objects: the lookup is by
>   the object's own name under each root, and the archive keeps radar,
>   precipitation and surface objects apart:
>
>   ```
>   --obs-archive-root $OBS/objects/mrms/noaa-mrms-pds/CONUS/MergedReflectivityQCComposite_00.50/20240521 \
>   --obs-archive-root $OBS/objects/stage4 \
>   --obs-archive-root $OBS/asos/2024-05-21
>   ```
>
>   A named directory holding a *different* file of the same name fails the
>   digest, as it must; the root relocates the lookup and widens nothing.

`--boundary-width-cells 5` is this config's `spec_zone 1 + relax_zone 4`; with
the registered 45 km rim it excludes 15 cells on every side, leaving **158 400
of 192 000** cells scored: the same interior the controls were measured on.

Expect `S_refl` (mean over F02-F18 of FSS at 30 dBZ and the 27 km box) as the
one number the promotion rule consumes, and these already-measured context rows
beside it (`B5-CASE-ENTRY-20240521.md` section 5):

| row | `S_refl` |
|---|---|
| persistence (the zero-skill floor) | **0.1598** |
| useful-skill line `0.5 + f_obs/2` | 0.5249 |
| a perfect forecast, right day | 1.0000 |
| a perfect forecast, wrong day | 0.0281 |

`--registration` binds the score to the committed campaign document
(`registration_sha256`
**`1d8d0e8a5e76cf47e115fe0a5c2f6be5b1aa8ca3c1f272baa7984c2d8e9b82b6`**,
`rule_status` `ratified-under-delegation`), and refuses a case or an arm that
document never registered.  **That digest is the canonical-JSON hash of the
document's `parameters` block**, the `registration_sha256` field inside the
file, recomputed and checked by `gpuwm.verify.obs.registration`
(`canonical_hash(parameters)`), **not** the SHA-256 of the file's bytes,
which is `3f5cae88…` and will move with any whitespace or prose edit while
the registered parameters, and therefore the digest above, stay fixed.
Verify it as the tool does:

```
python -c "import json; from gpuwm.verify.chaos_envelope import canonical_hash; \
  doc=json.load(open('docs/public/receipts/obsbattery/registration/REGISTRATION-v2-ratified-under-delegation.json')); \
  print(canonical_hash(doc['parameters']) == doc['registration_sha256'] == \
  '1d8d0e8a5e76cf47e115fe0a5c2f6be5b1aa8ca3c1f272baa7984c2d8e9b82b6')"
``` The score file carries that digest, so a score
published beside a different registration stops matching its own rule. Without
`--registration` the tool builds a single-arm `proposed-unratified` document
instead, which is the one-off path and cannot support a promotion.

The score file's `observation_rehash` re-proves every archive object; it was
**34 of 34 clean** when the controls ran, and a mismatch at scoring time is a
disk finding, not a model finding.

## 7-10. The WRF arm: deferred, node 1 busy

Per B4 section 4 venue B. The bundle is already written (step 2): `plan.json`
and `launch_arm.sh` beside the namelists are the WRF arm's.

7. Export `wrfinput_d01`/`wrfbdy_d01` from the **assembled tree**, not from the
   root preparation. This is not a style preference: the tree's per-domain
   artifact carries the canonical `surface/*` group
   (`gpuwm/native_domain_artifacts.py:252-256`), so the exporter reads `SH2O`;
   the root preparation cache does not
   (`tools/hrrr_single_domain_benchmark.py:2954`), so an export from the root
   takes the synthesis branch and is exposed to the frozen-soil refusal at
   `gpuwm/wrf_direct.py:1007-1009`. For this case the season also points the
   same way, but the artifact is the stronger argument and the one to rely on.
8. The twin, rung 1:
   ```
   python tools/n5s/perturb_ulp.py \
     $SCRATCH/node/control/wrfinput_d01 $SCRATCH/node/twin/wrfinput_d01 \
     --field T --member-ordinal 0 --record $SCRATCH/node/twin/perturbation.json
   ```
9. Launch each arm detached, 24 ranks:
   ```
   bash $SCRATCH/case/launch_arm.sh control $SCRATCH/node/control $WRF_BUILD
   bash $SCRATCH/case/launch_arm.sh twin    $SCRATCH/node/twin    $WRF_BUILD
   ```
   Projected **3.512 h per arm, 7.025 h for the pair** at 24 ranks
   (`plan.json`, scaled from the still-unreceipted 14.0 s/sim-min controller
   measurement). Both are past the 55 minute supervision window, so both are
   detached launches, per standing law. **Do not rebuild WRF**: the pinned
   campaign build is the arm, and the reference bundle is not campaign source
   state.
10. Score each WRF arm with the same step-6 command (one reader, one scorer,
    both models) and then the twin band on the primary score is
    `|S_refl(twin) - S_refl(control)|`. If that is exactly zero, the registered
    escalation ladder fires as **one** re-registered motion before any campaign
    case is scored, never per case.

## What the shakedown must answer before B6 can start

1. Does the WOOF arm beat the measured persistence floor of **0.1598** beyond
   F03? A no indicts the case or the scoring first, not the model.
2. Is the twin band non-degenerate? The one twin envelope this project ever
   built came out identically zero, and the convective 24 h window is expected
   to differ, but it is checked, not assumed.
3. Is the regrid delta of **0.00731** below the twin band? If not, the remap
   operator is re-registered once, before any campaign run.
4. Does the exporter produce a usable `wrfinput` for **mp8**? Every artifact in
   the tree binding exporter output to an actual `wrf.exe` run is WSM6 (B4
   caveat 3). This is a first-use risk, and the case's t0 receipt is where it
   gets paid for.
5. Does the frozen station set survive the terrain and land screens? 479
   stations sit in the interior on unique cells; the land mask and the 100 m
   terrain tolerance have not run yet, and they only subtract.

## Open items this staging could not close

* **The RFC seam mask** (spec section 2.2) is not in the case entry receipt: no
  RFC boundary geometry exists in the tree, and inventing one would be an
  agent-invented registration parameter. It touches the QPF scores only.
* **The section 9.2 registration freeze** is B5's exit and still waits for
  the scored shakedown. Q1 and Q2 are answered -- the seven-case menu stands
  as registered with B-04 as shakedown, and the promotion numbers are ratified
  under the owner's standing delegation with an overrule window that closes at
  campaign launch. The version series and everything the freeze still waits
  for are in `B5-REGISTRATION-FREEZE.md`.
* **The archive-of-record mirror** to node 2's 8 TB has not been attempted and
  no host was contacted. `OBS-ARCHIVE-MANIFEST.json` and
  `manifests/obs-*.json` are what the mirror is verified against, per object.
