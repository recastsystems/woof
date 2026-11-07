# 9. Reference

## 9.1 The doors

| command | what it does | needs |
| --- | --- | --- |
| `woof hex version` | report the installed distribution, version, package path | nothing |
| `woof hex doctor [--explain] [--json]` | report every estate this install can reach; exit 1 while a required one is missing | nothing |
| `woof hex mesh-check --grid G --static S` | validate a mesh pair; print dimensions and SHA-256 digests; a regional cull gains a `regional` receipt block; `--grid-only --grid G` validates a grid before its static exists | the pair (or the grid alone) |
| `woof hex oracle-gate --grid G --static S --fixtures DIR` | replay the source-extracted Fortran M1 fixtures against a mesh | a source checkout's oracle fixtures |
| `woof hex cull --parent-grid G --parent-static S --parent-init I --region R` | cut a limited-area grid, static and init out of a global case; `init` refuses a regional grid by name, so this is how a limited-area case gets an init at all | `rw_mpas_mesh`, the global triple, one region row |
| `woof hex init ...` | build initial conditions (chapter 5) | `rw_mpas_init`, met file, mesh pair, capsule |
| `woof hex forecast ...` | run the model on a registered mesh (chapter 6); `--preflight` answers "will it fit?" without integrating | a CUDA device with room for the mesh, the pinned `woof` installed, mesh pair, init |
| `woof hex swath {plan,metrics,explain}` | place fine grids from a coarse forecast's own fields, print the armed threat rows, explain why each candidate was taken or declined; `plan` prices every admitted swath through a real `rw_mpas_mesh --dry-run` unless `--no-size` | CPU only; a coarse forecast or its run receipt, `rw_mpas_mesh` to price |
| `woof hex cycle {plan,run}` | follow weather across cycles ([`docs/cycle-door.md`](../cycle-door.md)): `plan` says what each cycle would place and opens no device, `run` does cull → mid-window init → boundaries → forecast → render | `plan`: the parent case and `rw_mpas_mesh`. `run`: everything `forecast` needs |
| `woof hex render ...` | history → product PNGs (chapter 7) | `rw_mpas_convert`, `rw_wrfbatch` |
| `woof mesh ...` | generate a grid + static pair (chapter 4; engine door) | `rw_mpas_mesh`, `rw_mpas_static`, WPS_GEOG |
| `woof fetch-bridges` | stage the engine's published binary bundle into `~/.woof/bridges` | network |
| `woof fetch-geog --root DIR [--list]` | stage / inventory the WPS_GEOG archive | network, ~28 GiB unpacked |

Under the forecast door (chapter 6):
`tools/run_cuda_v841_forecast_mesh.py` (registered-mesh runner, with
`--verify-only` and `--selftest`), `src/hexcore/drivers/run_cuda_v841_forecast.py` (the
arbitrary-case driver whose `execute_forecast` the door drives),
`src/hexcore/drivers/run_cuda_v841_full_physics_x4.py` (the sealed proof harness: native
comparison, checkpoint/restart proof),
`tools/device_memory_ledger/hex_ledger_probe.py` (the per-allocation
device-memory ledger a footprint row is fitted from).
Obs-referee: `tools/run_obs_referee.py` with manifests under
`verification/manifests/` ([`docs/obs-referee.md`](../obs-referee.md)).

**Device memory is not `fixed + slope × cells`.** The footprint model is
`core(card, configuration)` plus the Grell-Freitas workspace at
`min(cells, SMs × 4 × 64)` columns, plus the YSU workspace at
`min(cells, SMs × 16 × 32)`, plus `bytes_per_cell × cells`; the margin held
back is the card's own RRTMG shortwave workspace plus 11.2 MiB of
instrument convention, both named and measured. Ask it with
`woof.hex.device_admission.model_for_card(card, configuration)`: the door
reads your card's multiprocessor count at the moment of the decision and
selects or derives its row, so the answer is your card's. Chapter 4.6 has
the shape, the knees and the inversion. `--device-fixed-mib` /
`--device-bytes-per-cell` survive as an escape hatch for a card whose own
ledger you have run; they are no longer the remedy of first resort, and
they are one row that must be given together.

`woof hex --help` and `<door> --help` are the authoritative flag lists;
chapter 5.3 tabulates the init switches, chapter 7.5 the render selection.

## 9.2 Engine resolution ladder

Every door resolves every engine the same way, best rung first; an explicit
flag or variable naming a missing file is a hard error, never a
fall-through.

| binary | order |
| --- | --- |
| `rw_mpas_init` | `--engine`, `$WOOF_HEX_RW_MPAS_INIT`, `$RW_MPAS_INIT`, `$WOOF_RW_MPAS_INIT`, woof bridge directories, `PATH` |
| `rw_mpas_convert` | `--convert-exe`, `$WOOF_HEX_RW_MPAS_CONVERT`, `$MPAS_PORT_RW_MPAS_CONVERT`, `$WOOF_RW_MPAS_CONVERT`, woof bridge directories, `PATH` |
| `rw_mpas_mesh` | `--mesh-exe` (`--engine` on `cull`), `$WOOF_HEX_RW_MPAS_MESH`, `$RW_MPAS_MESH`, `$WOOF_RW_MPAS_MESH`, woof bridge directories, `PATH` |
| `rw_wrfbatch` | `--renderer-exe`, `$WOOF_HEX_RW_WRFBATCH`, `$MPAS_PORT_RW_WRFBATCH`, `$WOOF_RW_WRFBATCH`, woof bridge directories, `PATH` |
| `rw_mpas_geometry` | `$WOOF_HEX_RW_MPAS_GEOMETRY`, `$RW_MPAS_GEOMETRY`, `$WOOF_RW_MPAS_GEOMETRY`, this wheel's `woof/hex/libexec`, woof bridge directories, `PATH` |
| `rw_mpas_stream` | `$WOOF_HEX_RW_MPAS_STREAM`, `$WOOF_RW_MPAS_STREAM`, this wheel's `woof/hex/libexec`, woof bridge directories, `PATH` |
| `rw_mpas_static` | `--static-exe`, `$WOOF_HEX_RW_MPAS_STATIC`, `$RW_MPAS_STATIC`, `$WOOF_RW_MPAS_STATIC`, this wheel's `woof/hex/libexec`, woof bridge directories, `PATH` |

The `GPUWM_HEX_*` spellings are preferred; the older spellings still work
and always will: a rename never invalidates an install line that already
works. "woof bridge directories" means a woof checkout's
`tools/rustwx/target/release`, `libexec/bridges` beside the installed
package, and `~/.woof/bridges` (where `woof fetch-bridges` stages).
Every engine is looked for in this wheel's own `woof/hex/libexec` just before
the woof bridge directories; the prebuilt wheel carries three there,
`rw_mpas_geometry`, `rw_mpas_stream` and `rw_mpas_static`, because no released
woof carries the versions this release needs ([engine bridges](../engine-bridges.md)).

Other variables: `WOOF_HEX_NO_LOCAL_GPU=1` / `GPUWM_NO_LOCAL_GPU=1` ban
device contact; `GPUWM_WPS_GEOG` names the geog root for `woof mesh`.

## 9.3 The mesh registry

`src/hexcore/drivers/mpas_mesh_binding.py`, one entry per runnable mesh: name, declared
`nCells`/`nEdges`, exact grid/static byte counts and SHA-256 digests, static
provenance, nominal spacing (compared FP32-bit-exactly against the static's
declaration), and the declared `dt_seconds`. Adding a mesh is adding a
row: data, not a code path. Chapter 4.1 tabulates the current entries and
their forecast status.

**A timestep passes two gates.** `src/hexcore/timestep_admission.py` is
the geometry gate: the versioned outer-step Courant policy, re-measured at
bind from the mesh's own complete `dcEdge` array and never from the nominal
spacing (4.1). `src/hexcore/dt_admission.py` is the evidence gate: an
earned-anchor registry keyed by CONFIGURATION (the timestep together with
the cumulus selection and the surface/PBL cadence) not by timestep alone,
and not per mesh digest, so a mesh finer than the one an anchor was earned
on inherits it. Seven rows stand across five timesteps: 120, 100, 75, 20
and 5 s with Grell-Freitas, and 20 and 5 s with convection off. Each names
its schedule receipt, its integration anchor (two byte-identical forecasts
on named hardware), and a measured `physics_health` verdict against a 120 s
control on the same card, mesh and init. Read that verdict: four of the
seven rows read `DIVERGES`, so an anchor certifies that a configuration
integrates finitely and deterministically at the cadences it names, and
nothing more. Only 120 s carries a native MPAS-A reference and only it can.
A configuration with no row is refused by name, with the roster and the
mint command. Rule of thumb for what a mesh wants: `dt ≈ 6 × dx(km)`, the
textbook 20 s at 3 km and 5 s at 750 m, which are exactly the timesteps
whose rows were earned on a 120 km mesh 35× and 140× below its own Courant
limit and diverge there.

## 9.4 Where receipts live

| receipt | written by | carries |
| --- | --- | --- |
| `<name>.cull.json` | cull door | the region row, the engine, one entry per cut file with its parent, its cell counts, its lineage and the engine's own `*.cull-receipt.json` |
| `<init>.provenance.json` | init door | SHA-256 of every input, engine binary, argv, engine receipt, output |
| `render-manifest.json` | render door | engine digests, weights/output digests, per-frame product results, exact invocations |
| `forecast-receipt.json` | forecast door | the resolved request, the admission decision with every number it was made from, the mesh-binding receipt, the driver's receipt whole, the history files, the render command for them |
| `cuda-v841-forecast-receipt.json` | forecast driver | claim, nonclaims, dropped guarantees, source/authority digests, per-step records, `gf_declared_divergence` |
| `swath-plan.json` / `threat-decision.json` | `swath plan` | the armed metric rows, every track and every drop with its reason, one mesh-spec and one cull-region row per admitted swath, priced or stamped `--no-size` |
| `cascade-receipt.json` | `cycle` | one block per cycle: what was admitted, what was declined or skipped and why, the churn (which slots moved and which stayed), and every slot's legs with their timings. `cycle-NN/` beside it holds that cycle's `swath-plan.json` and `swath-state.json` |
| `--receipt-json` bind receipt | registered-mesh runner | the mesh binding: names, digests, admitted dt, fingerprints |
| `demo.receipt.json` / static receipt | `woof mesh` | generation parameters, gates applied, output digests |

The measurements this manual quotes are stated where they are quoted: the
device-footprint model (chapter 4, `docs/device-memory-ledger.md`), the
timestep anchors and the convection threshold (chapter 6), the cycling loop
(6.9, `docs/cycle-door.md`), local time stepping (6.6,
`docs/local-timestep-lam.md`), restart identity (6.5) and the observation
referee (`docs/obs-referee.md`).

## 9.5 The test battery

Three tiers, split by what a machine must own for each tier's result to mean anything
([`tools/battery/README.md`](../../tools/battery/README.md)):

```sh
PYTHONPATH=src python -m pytest tests -q -m "not gpu and not bigcard and not assets"   # tier 1: anywhere
PYTHONPATH=src python -m pytest tests -q -m assets    # tier 2: ~6.9 GiB byte-pinned authority files
PYTHONPATH=src python -m pytest tests -q -m bigcard   # tier 3: capacity preflight for the big-card gates
```

Tests that cannot run skip with the missing thing named. Anything touching
CuPy is auto-marked `gpu` by AST inspection, so it cannot leak into tier 1
by omission.

## 9.6 Names that matter

- Distribution and command: `woof hex`. Import namespace: `woof.hex`,
  renamed from `mpas_port` at 0.2.0 and settled there (README, *The import
  namespace*). There is no alias shim.
- Engine range: `gpuwm>=2.9.12,<2.10` is what pip enforces, inside the 2.9
  minor, derived from a measured table in `woof.hex.engine_pin` rather than
  typed. The floor is the engine whose bytes this release measured against
  the sixteen-file seam manifest; an engine inside the minor whose radiation
  helper set no row admits is refused at the forecast door by name, and the
  whole-minor range means installing woof hex never downgrades an engine you
  already have. The floor clears, incidentally, the bundle rows that put the MPAS bridge
  binaries (`rw_mpas_init`, `rw_mpas_convert`, `rw_mpas_mesh`,
  `rw_mpas_static`, `rw_mpas_lbc`) within reach of `woof fetch-bridges`;
  published 2.5.2 carries none of them and would strand every door that
  drives one. The CEILING is exclusive and sits at the first engine not
  measured usable, so an engine that has not been compared against the
  manifest is never resolved onto, including ones not published yet. An
  engine that is NOT on PyPI (a local version such as `2.9.0+x.1`, which
  the range already resolves) is admitted only by a measured row of its
  own in `woof.hex.engine_pin.ADMITTED_ENGINES`, written by the same
  instrument's `--admit` from the engine's wheels and source tree and
  keyed by its exact version string and its sixteen digests together; the
  seam check, the door, `doctor` and the adapter name that row's bytes
  under that version as that engine. Since 2026-10-10 the sixteen-file seam
  manifest is evidence, not a wall: a forecast runs on an engine whose seam
  bytes match no measured row, and its receipt records the comparison
  (`arwen_engine_pin`: declared version, measured version, the moved and
  absent files). The pip range above is what an install resolves; a tree
  passed as `--gpuwm-checkout` may be any woof that carries
  `woof/core/mpas_column_batch.py`. `woof hex doctor` compares the
  installed engine's bytes against the manifest and names the version it
  found; since 2.5.8 all sixteen pinned paths resolve from the install,
  and since 0.3.2 the forecast runs on it with no clone. The x4
  full-physics proof keeps the exact-bytes requirement.
- Licence: Apache-2.0, with only permissive upstream terms for the ported
  portions: MPAS-Atmosphere and NCAR MMM-physics (BSD-3-Clause), WRF physics
  (public-domain notice), RRTMG (AER BSD-3-Clause), MT19937 (BSD-3-Clause)
  and the Arm and CORE-MATH maths (MIT). `NOTICE`, `THIRD-PARTY-NOTICES.txt`
  and `licenses/` state them. UGWP's CUDA ports ship and the RRFS stage runs
  the fork's NOAA GSL physics with no opt-in: the project treats them as US
  Government code (owner ruling 2026-10-14; `NOTICE` sections 8 and 9).
  Withheld from the public distribution: the Noah-MP v5.0 CUDA port (UCAR
  terms forbid sale). No default runs code that cannot be sold: a selection
  of `config_lsm_scheme='sf_noahmp'` is refused unless the run names the
  terms with `--licence-opt-in ucar-noahmp` (`WOOF_HEX_LICENCE_OPT_IN`), and
  such a run says so on its console and in `physics.licence_opt_in`. The Noah-MP that runs by default is the WOOF
  suite's WRF-lineage v4.x (WRF public-domain notice), not MPAS-bitwise. This
  is not the version available from LANS and UCAR, and neither they nor their
  contributors endorse it.
