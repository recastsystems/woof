# Research hardware and method contract

Development proposal, 2026-09-06. These are deliberately chosen configurations to compile and validate; none is yet a measured release recommendation. Every research leaf needs at least three scientifically distinct configurations, not three labels for the same run. Hardware variants implement those configurations without changing the question silently.

## Authoritative engine behavior

- `woof domain` creates complete experiment TOML and native companions. `--root-dx` is kilometres; finite positive values are required. The documented 0.05–200 km range is advisory in the actual parser. `--chain` requires integer ratios >=2; ratios >8 and more than eight nests warn and continue. Curated configurations should stay at ratios 2, 3 or 4 and at most three concurrent domains until deeper trees earn evidence. Do not turn advisory limits into claims that arbitrary deep refinement is scientifically demonstrated. See `woof/domain_wizard.py:2281`, `:3446`.
- Named `--card` values are `12gb`, `16gb`, `24gb`, `32gb`; `--vram-gib` covers 8 and 10 GiB too. An explicit card/capacity uses assumed free memory: capacity minus `max(0.75 GiB, 0.06*capacity)`. Omitting capacity measures the actual local card and its available memory. It must be the forecast host, because host RAM is separately priced.
- Point fitting grows a centered layout against the actual phase estimator. Its envelope budget is free memory minus the 0.5 GiB external margin. Its target then subtracts `max(0.25 GiB, 5% of envelope budget)`. CUDA context, selected kernel local-memory backing, allocator slack, transient radiation and nest/corridor costs already live inside the envelope; do not subtract those twice. Polygon fitting preserves the complete polygon plus requested buffers and refuses if it does not fit. It does not apply the point-bisection headroom subtraction. See `domain_wizard.sizing_budget_bytes`, `fit_headroom_bytes`, `fit_ladder`, `fit_polygon_ladder`.
- A supplied forcing archive may retain boundary intervals beyond the requested run end. Shortening hours alone does not remove those memory costs. Source, cadence, physics, vertical levels, host RAM and moving-store reconstruction must be included in the estimate.
- `--nz` resamples the native stretched eta ladder, retaining N+1 interfaces. It is a scientific change. Root time step starts from 5 s/km, reduced in the tropics; event alignment can shorten it. Child spacing/time step divide parent values exactly. History/radiation/cumulus events must land on the rational clock. Never repair cadence by rounding the user's value silently.
- Child `nx`/`ny` are mass-grid dimensions divisible by the parent grid ratio. Child southwest placement is 1-based parent cells. The loader owns boundary clearance, nesting geometry and clock validation. Geometry spans below are approximate mass-grid coverage `nx*dx`, useful for comparing configurations; native projected coordinates are the final coverage authority.
- `domain-fit TEMPLATE --point ... --write` preserves the template's complete scientific choices while fitting geometry. It maximizes the centered layout; it is not a request to preserve a particular geographic footprint. `--polygon` is the footprint-preserving route. `domain-tiles TEMPLATE --mode auto --write` preserves geometry and asks the shared planner to change execution road.

## Implemented candidate hardware profiles

The shared compiler/UI policy is [`research-hardware-profiles.json`](../../woof/data/tui/research-hardware-profiles.json), schema `arwen.research.hardware-profiles.v1`. These are candidate settings awaiting physical qualification. The compiler computes actual grid dimensions through the native planner; this table makes no fixed-shape fit claim. Every recipe keeps its duration, output cadence and minimum study extent. The selected hardware policy may change its horizontal ladder, so the receipt stores both the requested and effective geometry and explicitly reports a coarser-than-preferred finest spacing.

| Profile | Regional spacing | Mesoscale ladder | Storm ladder | Vertical mass levels |
| --- | --- | --- | --- | --- |
| 8 GiB | 12 km | 12 → 4 km | 4 → 1 km locally; 12 → 3 → 1 km for broad context | 49 |
| 12 GiB | 9 km | 12 → 4 km | 4 → 1 km locally; 12 → 3 → 1 km for broad context | 49 |
| 16 GiB | 6 km | 9 → 3 km | 12 → 3 → 1 km | 49 |
| 24 GiB | 3 km | 12 → 4 → 2 km | 12 → 3 → 1 km | 49 |
| 32 GiB | 3 km | 12 → 4 → 2 km | 12 → 3 → 1 km | 64 |

A recipe with no nests selects the regional intent. A nested recipe whose preferred finest spacing is at least 3 km selects mesoscale; finer requests select storm. For classes 8/12, a required root span below 500 km selects the compact pair, while a span of at least 500 km selects the three-domain context ladder. This rule follows CPU measurements: the old 4 → 1 km choice failed all 18 broader storm questions at the declared 8 GiB budget, while the 12 → 3 → 1 km alternative retained a 1512 × 1200 km root and a 276 × 216 km fine child for the four sampled rotation, derecho and tropical-core cases. These are native CPU estimates, not completed forecasts or physical 8 GiB tests. If the fitted root still falls below the recipe's numeric minimum span in either horizontal direction, creation refuses and prints its actual dimensions. A polygon retains its actual requested footprint. No profile disables longwave radiation or replaces the selected physics to fit.

`--hardware-class` chooses this policy only. `--vram-gib` alone declares an assumed-capacity budget. Without it, even an explicit class measures actual capacity and free memory once. Auto selects the highest class whose assumed-free allowance fits the measured free memory, capped at measured total memory. It does not compare the marketing capacity to CUDA's post-reservation total: the physical RTX 5070 Ti reports 15.5084 GiB total and 15.28 GiB free, which admits the 16 GiB profile's 15.04 GiB free-memory allowance. Neither measured value is rounded up. The floor is class 8 and the native planner can still refuse. A 32 GiB GPU with 8.5 GiB free selects the 8 GiB profile, while an explicit class 32 retains that geometry against the same 8.5 GiB free-memory budget.

Controlled warm-bubble comparisons retain their catalog's 12 → 4 km ladder and 49 levels at every class, so changing bubble amplitude or humidity behavior does not also change vertical discretization. Archived downscale and supplied-scenario entries remain parent-dependent: their profile view preserves the declared study intent, leaves vertical levels unresolved, and requires actual existing-state metadata. An independent archived-child vertical ladder needs explicit native `--child-levels N,STRETCH`; a hardware label never invents it.

Coverage retains upstream forcing and broad context; a compact moving/fixed child spends resolution on the feature; an archived child spends only one forecast domain on the GPU after its parent finishes. Those are distinct costs and questions. A 1 km nest is not a tornado-resolution or resolved-turbulence guarantee.

## Tracking, lifecycle and moving stores

`FollowConfig` supports `uh`, `reflectivity`, `pressure`, and the new native `attribute` registry (`woof/core/storm_tracking.py`). UH thresholds are m²/s² and require a separate reflectivity fallback threshold in dBZ. Reflectivity thresholds are dBZ and reject a fallback. Both use a diagnostic/history-cadence contract; the loader refuses consultation faster or off-lattice relative to the source fields. Pressure/height tracking derives from live state and is independent of history cadence.

Pressure tracking defaults to 850 hPa. `level_hpa` accepts a surface or supported multi-level steering mean in 200–1000 hPa. Thresholds on an isobaric surface are metres above the search minimum, not hPa. Explicit `level_hpa=0` requests sea-level pressure and then threshold must be 800–1100 hPa. Never reuse the numeric threshold while switching surface semantics. `radius_km` is 1–500 km, default50. Search margin is >=0 parent cells; min shift >=1; max shift >=min; cooldown is finite and >=0 seconds. Parent-edge keepout is five parent cells. Per-domain follow and global `[relocation.follow]` exist; the TUI already preserves these forms.

The reviewed catalog's tropical tracker uses 850 hPa and `threshold=20.0` m height depression, `radius_km=50`, search margin 10, min shift 1, max shift 4 parent cells, cooldown 300 s and consultation 300 s. Rotating-convection tracking uses 50 m²/s² UH with a separate 25 dBZ echo fallback; reflectivity tracking uses 35 dBZ. These are case-review detection settings, not severity thresholds. Time/signal-triggered birth, retirement and re-arm are separate lifecycle controls, not free memory or unlimited dynamic mesh creation.

Native attribute requests explicitly select `theta` (K), `qv`, `qc` or `qr` (kg/kg dry air), or `w` (m/s); `extremum=max|min`; and `reduction=column_max|column_min|column_mean|model_level`. Model-level reduction requires a zero-based integer level inside the actual source grid; other reductions refuse that key. Units come from `woof/core/attribute_tracking.py`, never a configurable TOML units key. The research catalog may display a units label, which the compiler checks against that registry before emitting the exact native request. The new attribute implementation has separate CPU tests and still needs its physical moving/restart evidence.

Moving children require sealed child-resolution statics over their movement corridor. Preparation must request `--statics-corridor` for affected children and retain its hash in the preparation receipt. New strips are reconstructed from the parent; overlapping evolved child state is retained. Moves do not invent fine-scale information in newly entered terrain. A moved checkpoint resumes through its own bound move/follower history, not a fresh stationary placement.

**Current source supports moving host-streamed children.** `woof/core/streamed_relocation.py` reserves bounded reconstruction storage and is wired by `prepared_domain_tree_forecast.py` and `core/streaming.py`. The older “keep moving child resident” sentence in public TILES.md, and adjacent-streamed refusal in HARDWARE.md, are stale. Existing `work/streamed-move-case-900/PROOF.md` records a 900×900×49 child whose resident requirement exceeds a physical10GiB3080, native moves and bitwise restart comparison; it is only a27-model-second proof on an older source. It does not qualify an hours-long new research configuration. New acceptance must include a due radiation call, movement, bounded memory and restart on the final revision.

Use `[tiles] mode="auto"` for an optional out-of-core alternative. `on` forces streaming. A moving host store needs admitted reconstruction reserve; both device and pinned host RAM remain finite. `auto` rejects pinned tile dimensions/buffer/halo hints; explicit dimensions require `on`. Forecast users must leave halo to the solver. More stages or streaming cannot recover absent parent scales, improve forcing cadence, or remove spin-up and boundary-zone contamination. `woof stream PLAN.toml` is a different feature: sealed hourly HRRR forecast legs, at least two domains and restart interval3600s; it is not the `[tiles]` memory method.

## Executable create/check recipes

Use the selected installed engine Python. These commands create configurations and estimate them; they do not start a forecast. Choose new output paths. Explicit declared free memory is an estimate, not device measurement.

```powershell
& $EnginePython -I -m woof.cli domain --point=35.3,-97.5 --source gfs --cycle 2026-09-05T18 --hours 6 --root-dx 12 --chain 3 --nz 49 --vram-gib 8 --history-interval 900 --nest-history-interval 300 --out .\research-8-focus.toml
& $EnginePython -I -m woof.cli check .\research-8-focus.toml --vram-gib 8 --free-gib 7.25 --json
& $EnginePython -I -m woof.cli domain-fit .\research-8-focus.toml --point=35.3,-97.5 --vram-gib 12 --out .\research-12-fitted.toml --write
& $EnginePython -I -m woof.cli domain-tiles .\research-12-fitted.toml --mode auto --out .\research-12-tiles.toml --write
```

The first command shows the underlying native domain mechanism. The research compiler uses the selected shared profile instead, writes diagnostic and review sidecars, and refuses existing output files:

```powershell
& $EnginePython -I -m woof.cli research catalog --json
& $EnginePython -I -m woof.cli research hardware --json
& $EnginePython -I -m woof.cli research create rotation.structure --point=35.3,-97.5 --source=gfs --cycle=2026-09-05T18 --hardware-class=8 --out .\new-rotation-study.toml --json
& $EnginePython -I -m woof.cli research create scenario-convection.gentle --point=35.3,-97.5 --source=gfs --cycle=2026-09-05T18 --hardware-class=8 --vram-gib=8 --out .\new-bubble-estimate.toml --json
```

On the actual execution host, omit capacity flags from creation to measure free memory, then `check CONFIG --json` to capture live readiness. `check --alloc` performs real zero-step GPU allocations and belongs to coordinated physical validation. Creation and a passed CPU estimate do not qualify preparation, allocations, forecast integration or scientific skill.

Ordinary regional/nested/moving creation resolves sources through the native registry and preserves source-specific companions. ERA5 keeps `[case_data]`, a configuration-specific Vtable and correctly rebased forcing/WPS paths; HRRR keeps its generated target and both native namelists. Controlled warm-bubble creation is currently restricted to GFS's explicitly supported fresh prepared-tree path. Fresh ERA5 preparation still refuses the perturbation block, and GDAS uses the mapped preparation runner, whose perturbation refusal is also explicit. Support in another runtime is not proof that a fresh source preparation route admits the same scenario.

Archived route (actual parent paths required):

```powershell
& $EnginePython -I -m woof.cli downscale $ParentHistory --parent-restart $ParentRestart --point=35.3,-97.5 --ratio 3 --child-size 252,204 --hours 3 --output-interval-seconds 300 --max-boundary-interval-seconds 900 --child-surface-from $ChildSurface --vram-gib 12 --preprocess-backend cpu --out .\new-child-run --dry-run
```

Dry-run validates and writes a derived configuration/plan; it is not filesystem read-only. A WOOF-ArWen parent needs its restart physics evidence. Stock WRF instead requires its namelist and an explicit standalone child RunConfig (`specified=true,nested=false`) with ratio and 1-based placement. Surface physics requires a child-grid land/soil warm start. Parent cadence acceptance must be explicit; a coarser archive cannot supply missing900s boundary evolution. Each sequential stage must produce the next stage's real history, restart identity and surface state before the next starts.

## Hardware evidence and compiler integration

Root observed a research node on2026-09-06: physical RTX5090,32607MiB total,8735MiB free, driver610.43.02 over existing strict SSH. That is about8.53GiB free and roughly7.63GiB point-fit envelope target after external margin/headroom; selecting the32GiB assumed-free budget would overstate current availability. Do not stop its existing work. Primary is the physical10GiB3080; a development machine is physical16GiBRTX5070Ti; a development machine has no nvidia-smi observation. Final receipts must capture each device UUID, OS/driver, total/free immediately before admission and final source revision. Historical5090 receipts are not current-device qualification.

Compiler contract: load one shared catalog, resolve one immutable sizing budget, pass it into the existing native wizard, compile into a fresh temporary stage, validate the final scientific TOML with the engine loader/check, and publish a create-only bundle of TOML/native WPS companions/recipe metadata/plot sidecar with correctly rebased paths. Expose profile, method, effective geometry, source, cycle, duration, levels, track semantics, output cadence and estimated memory before launch. Existing user files are never inputs to an overwrite path. Archived-downscale IDs must route to the existing Downscale guide and refuse ordinary `research create` until real parents are supplied. Controlled-scenario recipes explicitly declare their reviewed warm-bubble amplitude, size, height and humidity behavior; selecting one creates that exact perturbation at the user-selected centre in a configuration with at least two domains. Unknown scenario fields are refused. Preparation must still prove that the bubble touches model cells; a created configuration is not a claim of completed preparation or integration.

Every configuration gets a validation row. Distinguish physical10/16/32GiB tests, restricted-budget tests for8/12/24, CPU estimates, and refusals. Hardware size does not convert a pending science/forcing/movement check into PASS. Status must remain pending until the final configuration and final installed revision have passed their required preparation, forecast, diagnostics, checkpoint/resume and method-specific tests.
