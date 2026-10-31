# Moving nests: a fine region that follows a storm

`woof hex moving` keeps a fine region (for example 3 km, about 450 km in radius) inside a coarse
global mesh (for example 48 km) on a moving storm. It is something MPAS-Atmosphere does not have.
The dycore is the MPAS-exact port. The physics is whatever the case selects through the forecast
door: WOOF physics (the 48 h runs used it, with New Tiedtke convection) or an MPAS-A v8.4.2 suite
(see [MPAS physics on a moving region](#mpas-physics-on-a-moving-region)).

How a run works:

1. **Tracking.** The run-time tracker (`woof.hex.storm_tracker`) fixes the storm on the model's own
   history frames.
2. **Preparing ahead.** The next mesh member is prepared on the CPU while the card keeps
   integrating: generator spec, grid, static, init and mesh row.
3. **Moving.** At a move the forecast seals its state. The next forecast continues on the new member
   through a conservative remap that keeps the state in balance.
   - Mass and tracers are conserved to 1e-9 or better.
   - Restart across a move is bit-identical.

With no moving command and no `WOOF_HEX_MOVING_*` variable set, nothing changes: a fixed-region
forecast imports none of the moving state, remap or balance code.

## Check an install first (no inputs needed)

```bash
woof hex moving check --lts            # on a CUDA card: the production LTS executor
woof hex moving check --device cpu     # anywhere: the ordinary integrator on the CPU
```

On a card the check runs two cases: gentle (0-100 m terrain, 8 levels) and steep (a 2,700 m plateau
with a 16 K inversion over it, the production 55 levels), and passes only if both do. On the CPU it
runs the gentle case (8 levels); `--cases gentle,steep` adds the 55-level steep case. CPU operators
evaluate independent cells and edges together while retaining the source's accumulation order.
The three arms run in separate processes, limited by the available CPUs and any smaller
`OMP_NUM_THREADS` or `RAYON_NUM_THREADS` budget. The receipt records the process count and each
worker's stage times.

The check builds a 522-cell moving refinement family with `rw_mpas_mesh` and makes three moves, at
600, 1200 and 1800 s. It runs three arms through the production remap:

- `fixed`: no move, the reference.
- `baseline`: the remap without balance, i.e. the defect.
- `balanced`: the default path.

It prints the jolt after each move: the larger of the max |w| and max |dpsfc/dt| ratios against
the fixed arm over the next ten minutes. It passes when every balanced move is at most 1.5 and
conserves dry, theta and tracer mass to 1e-9. The receipt is `moving-check/moving-check.json`.
The cause and the fix are in [moving-jolt-report.md](moving-jolt-report.md).

## Run a case

A case is one JSON file. It is the only place a storm, a date or a position appears. It holds:

- the start time;
- the time step and level count;
- the vertical spec (a path, or the name of a packaged spec such as `tc55-v1.json`);
- the storm's position and motion at the analysis time;
- the mesh block: background, fine region, rings and an optional corridor of fixed rings;
- the tracker block;
- `physics_args` for the forecast door.

The source tree carries three worked cases in `verification/moving48/cases/`.

```bash
woof hex moving stage   --case CASE.json --casedir case/ --best-track    # GFS (or the case's init_source) -> WPS intermediate
woof hex moving prepare --case CASE.json --casedir case/                 # start member (corridor frame and template first)
woof hex moving run     --case CASE.json --casedir case/ --out run/ --hours 48 --device 0
woof hex moving fixed   --case CASE.json --casedir case/ --out fixed/ --hours 48 --device 0   # the matched control
```

**Inputs.**
- `stage` goes through the table-driven init-source path (`woof hex init-source --list`). Any row
  works, and `--staged DIR` reuses files already on disk.
- `--best-track` also fetches the storm's IBTrACS best track. It is used only for verification.
- `prepare` and the run's member preparation use `rw_mpas_mesh`, `rw_mpas_static` and the init
  door. `--geog` names a WPS_GEOG root; the default is `rw_mpas_static`'s own, which
  `woof fetch-geog` stages.

**Running.**
- `run` uses one card. `--device` sets `CUDA_VISIBLE_DEVICES`; by default the visible card is used.
- `run --cards K` (and `fixed --cards K`) splits the region across the first K visible cards of one
  host: every segment runs the multi-card class blocks of active local time stepping
  (`--samehost-devices 0,...,K-1`), and the door partitions each member from its own grid. The move
  hooks are the same as on one card: the stop request, a carrier with one entry per block, the
  remapped resume onto the blocks of the new member (a carrier from any card count moves onto any
  other), and one clock offset for every block. Do not also put `--samehost-*` flags in the case's
  `physics_args`; `--cards` refuses that by name.
- `--scripted-track` replaces the tracker's fixes, never the policy. It is for machinery tests, and
  the receipt says so.
- `--resume` continues a run whose controller died.
- `--engine` runs a woof checkout instead of the installed engine.

**Outputs.** In the run folder:
- `controller.json` and `controller.log`;
- `track-session.json`;
- one `segment-NNN/` per forecast;
- `summary.json`, which `woof hex moving summarize run/` rewrites.

Each segment's `fc/` is an ordinary forecast output folder. `woof hex render` and
`woof hex products` draw it.

**Acceptance evidence.** The source tree's `tools/moving48/acceptance.sh` gathers the evidence for a
moving run against its fixed control:
- storm inside the fine interior at every frame;
- post-move jolt against the fixed run;
- track error against the best track;
- frames and loops.

## MPAS physics on a moving region

A case whose `physics_args` select an MPAS-A v8.4.2 suite (`--physics-backend mpasphys` with a
physics namelist, as in a plain MPAS forecast) moves on that suite's own route
(`woof.hex.mpasphys.moving_v842`). The controller, the case file and the commands are the same.

- **What a move carries.** The dycore state goes through the same conservative, balanced remap as
  on the WOOF route. The suite's state goes with it: every physics field MPAS writes to its restart
  stream, plus the time-dependent surface diagnostics. The list is read from MPAS's own Registry,
  not kept by hand. So convection, boundary-layer, land and radiation memory continue across a move.
- **How it is mapped.** Fields on model levels are mapped in physical height. Land, soil and sea-ice
  state is taken whole from one donor column with the same surface role (land or water, ice, soil
  and vegetation class), so a column never mixes land and sea values. Wind tendencies are mapped
  as vectors.
- **After a move** the next member starts as an MPAS restart at the move time, so its physics
  initialises as MPAS restarts it, not as a cold start.
- **Not built:** zone carry (corridor cases) with an MPAS suite is refused by name; run such a case
  without a corridor.

Measured on one card (2025-11-08 12Z western Pacific storm, 240/60 km family, 55 levels,
convection_permitting with every scheme on the GPU, 2 h, one 357 km move at 1 h):
- The storm stayed inside the region at every frame (largest offset 249 km of the 300 km interior).
- Across the move, dry mass held to 2.5e-11 and every water species to 4.5e-10.
- MSLP, 10 m wind and reflectivity maps on one fixed lattice across the move show one continuous
  storm and no seam at the region edge.
- A stop and resume on the same mesh, physics carried, follows the uninterrupted run: the largest
  surface-pressure tendency and vertical velocity over the next 15 minutes agree within 0.4 % (not
  bit for bit).
- `woof hex moving check` (the same remap and dycore) passes: worst jolt 1.017.
- Post-move jolt against a fixed-region run of the same case, per 225 s step: the largest surface
  pressure tendency is 1.38 times the fixed run's over the first 15 minutes and 0.69 times over
  30 minutes, with its peak over the storm (13.0N 128.2E); vertical velocity 0.70. Mass, theta mass
  and every water species held to 4.9e-10 net of the declared saturation adjustment below.
- Two remap faults made the jolt on this case before, both in far-field cells: a donor extrapolated
  below its ground on steep terrain (22.3 times the fixed run), then, with that fixed, cloud water the
  remap left in subsaturated air (3.63 times). Mixing a foggy or cloudy donor with drier donors
  conserves vapour, cloud water and theta_m separately, and left 8.9 % of all cloud water on this case
  below saturation (the carried state itself holds 0.06 %). Thompson evaporated it on its first call
  at fixed density, so fog cells at 48.0N 126.3E and 23.5N 104.5E cooled 0.4-0.5 K in one step and
  their surface pressure fell and rebounded by 200-260 Pa. The remap now makes that evaporation itself,
  with its latent cooling, before the balanced initialization, for the cloud water beyond what the
  donors would evaporate themselves. Dry mass and each cell's total water are unchanged; the receipt
  states the cloud water turned to vapour (here 1.17e12 kg in 4,382 columns, largest cooling 1.19 K)
  and the theta-mass change it makes.

## Knobs

| Knob | Default | What it selects |
|---|---|---|
| `WOOF_HEX_MOVING_BALANCE` | `1` | `0` takes the previous thermodynamic remap path (the measured jolt) for comparison |
| `run --zone-carry` | off | frozen-far-field members built from the current member, with an exact carry outside the changed zone |
| `run --whole-domain-remap` | off | use the physical-height all-domain transfer on corridor or frozen-far-field geometry too; useful for qualifying its constrained balance correction on historical corridor cases. The controller records the route and refuses a resume with a different route. |
| `run --prelaunch` | on | start the next member's forecast before the move so its host preparation is hidden |
| `run --hold-for-member-seconds` | 1800 | rendezvous at every tracking frame; wait up to this many seconds for a decided member. A missing controller reply stops the run. `0` selects asynchronous decisions, which can land one check late. |
| `run --check-minutes` | case policy, else 60 | history and tracker interval in minutes |

## What is measured, and what is not

**Measured.**
- Three 48 h tropical-cyclone runs, five moves each. The storm stayed inside the 3 km interior at
  every output. Mass and tracers per move held to 9.7e-11. Restart across one move was
  bit-identical.
- These runs used the remap before the balance fix. Against the fixed-region run, two of 15 moves
  exceeded the 1.5 jolt limit: 1.5004 and 1.81.
- The balance fix brings the reduced test (`woof hex moving check`) from 16.2 to 1.01 on a card
  (LTS) and from 21.4 to 1.04 on the CPU (ordinary integrator), with mass, heat and tracer kept to
  1e-9 per move.
- One region split across two cards of one host (`--cards 2`, two RTX PRO 6000, every rate class but
  the finest cut across both cards):
  - `woof hex moving check --lts --cards 2`: worst jolt 1.013, mass, heat and tracer to 4.1e-10 per
    move, and every arm bit for bit the one-card check (all final state hashes equal).
  - The coarse smoke case (240/60 km, 12,381 cells, the 2025-11-08 12Z western Pacific storm, 3 h,
    one move at 1 h, scripted fixes): the model storm's track on two cards is the one-card track to
    0.09 km after 3 h and its central pressure to 0.01 hPa; surface pressure differed by 2.3 Pa RMS
    after 3 h (measured before the fix below). Mass and tracers held to 1.4e-10 at the move.
  - That difference was the multi-card executor, not the card count: its halo round before the
    microphysics read a finer class at mid-step instead of at the step end (fixed 2026-10-24). Now a
    one-card run and a two-card run of the same forecast are byte-identical without a move: the same
    storm on a 240/60 km fixed region (11,242 cells, rate classes 1/2/4, convection ntiedtke, 225 s
    base step, two RTX PRO 6000), 3 h, every 15-minute frame, 50 fields, 167 M values, 0 different.
  - A carrier sealed on one card resumes on two and the reverse; a same-mesh resume on two cards is
    bit-identical to the uninterrupted run (14.1 M words, 0 different).
- Moves over steep terrain on whole-mesh members. The remap no longer extrapolates a donor below its
  own ground: a donor keeps its nearest interface value outside its range, and the air it lends below
  its ground is its hydrostatic constant-theta extension. On the coarse smoke case (one card, 3 h, move
  at 1 h) the post-move surface-pressure tendency against the fixed run went from 7.66 to 0.73 times
  (15 min window) and from 6.06 to 0.82 times (30 min), the vertical-velocity ratio is 0.68, the peak
  is back over the storm instead of a far-field cell on the Tibetan escarpment, and mass, heat and
  water are held to 4.5e-10. The reduced check has a steep case (a 2,700 m plateau, the production 55
  levels, a 16 K inversion over the plateau): on a card (LTS) the previous remap jolts it 2.957 and the
  current one 1.029; the gentle case is 1.026.

**Refused by name.** On one card, a case whose mesh classes entirely at rate one (a uniform mesh, or
`--fullstep-lts-max-rate 1` in the case) runs the native single-class step, which has none of the move
hooks; such a segment is refused with that reason instead of running to its end without moving. A
same-mesh resume with no controller still runs there (tests/test_moving_inactive_noop.py).

**Not yet measured, so not claimed.**
- A full-cyclone replay of the two failing moves with the fix. It is scripted as a named step and
  needs a card with about 64 GB or more: the 470,000-cell member holds about 54 GB, and a move
  about 62 GB.
- The steep-terrain remap fix on two cards: measured on one card. The remap is the same host code for
  any card count, and before the fix one and two cards agreed to four digits.
- Several cards for a zone-carry (corridor) move: built on the same hooks, not yet run.
- Zone carry (a corridor case, or `moving run --zone-carry`) with an MPAS suite: not built. The
  frozen-far-field transfer carries no physics state, so such a case is refused by name; its moves
  take the overlap remap when run without a corridor.
- The RRFS suite takes the same route as the MPAS-A v8.4.2 suites but has not been run on a moving
  case.
- The 16:1 spacing contrast inside the reduced check.

The former absolute `--min-dc-edge-m` admission cutoff is retired. It came
from 48/3 km zone-member experiments and rejected valid 1 km members requested
by their case while admitting the same start mesh. Start, runtime and recovery
members now share preparation admission; minimum edge length remains recorded
as a diagnostic, not a universal stability bound. This does not certify every
mesh/timestep combination or retire the generator and numerical-failure checks.

Runtime anticipation extrapolates the last two valid model fixes, with the existing
motion speed limit; a single initial position does not launch a case-motion guess.
At each check, a revised target with no member within `min_move_km` starts a
preparation. At most two preparations run concurrently (`--max-preparing 1`
allows one). When both slots are busy, a next-move request replaces only the
furthest preparation, retaining the nearer work; speculative later moves never
evict work. Nearby in-flight targets are reused so small changes of snapped
centre do not restart preparation every check. The actual
current member remains the origin until a move executes. When a move is due, the
nearest ready member containing the current fix within `interior_radius_km -
core_margin_km` may be used; it need not match the ideal centre within 5 km.
`--chain-min-fit` is retained only as a deprecated compatibility option.

Start and runtime preparation default to every CPU allowed by process affinity
for Rust/OpenMP threads and static tile workers. Explicit thread/worker options
remain available. The static builder may further bound concurrent tiles by its
existing memory budget. Stage receipts record the requested thread and tile
worker counts; these defaults alone do not establish a preparation speedup.

### Conservative operator CPU parallelism

Overlap construction keeps the same per-row clipping arithmetic and ordered CSR
assembly. `WOOF_HEX_OVERLAP_THREADS` controls worker count (default: process CPU
affinity). Auto execution uses spawned processes when the interpreter's GIL is
enabled, threads when it is disabled, and serial execution for fewer than 64
recipient cells or one worker. Spawned workers reopen temporary read-only NumPy
mappings; geometry is not copied into each task, no CUDA state is forked, and the
mappings are removed after worker shutdown, including failures.
The `moving operator` CLI also unwinds these mappings when the controller
cancels its process group with SIGTERM (exit 143). This does not cover SIGKILL
or machine failure. An embedding application's custom SIGTERM handler is
preserved; direct Python callers own their signal and cleanup policy.
The unpublished operator archive is also retired if saving fails or unwinds;
the final path is replaced only after saving completes.

For an explicit comparison, `WOOF_HEX_OVERLAP_EXECUTOR` accepts `auto`, `serial`,
`thread`, or `process`. The operator receipt's `overlap_execution` records the
selected backend, GIL state, worker count, UTC start/end, stage wall time and
worker CPU time for overlap tasks (excluding worker startup). Thread count alone
is not evidence of parallel throughput. Existing native overlap selection via
`WOOF_HEX_SPHERICAL_OVERLAP_LIB` is unchanged.

The `moving operator` JSON log also carries `overlap_execution`, plus timed geometry
read, operator build and archive save stages with their CPU parallelism. Retain
that small log when deleting spent operator archives; worker CPU evidence does
not require keeping the NPZ.

The `moving operator` CLI sets OMP/OpenBLAS/MKL limits to one thread before
spawning workers. Overlap row workers still use the requested process count; this
avoids multiplying native thread pools across those workers. A coordinator
pool already initialized during CLI parsing is not resized. Direct Python
API callers should set those limits before importing NumPy.

### Physical-height execution defaults

A moving map uses CUDA mass integration and atmospheric rate interpolation by
default when a CUDA device is visible. The former
`GPUWM_HEX_MOVING_HEIGHT_GPU` and `GPUWM_HEX_MOVING_RATE_GPU` switches are
retired and ignored, including when set to zero. The map API retains
`backend="cpu"` for reference comparisons; hidden devices retain CPU-only
preparation without importing CuPy. CUDA execution errors propagate.

Physical-area marginal reconciliation and height-stencil setup use bounded host
workers controlled by `WOOF_HEX_HOST_COPY_THREADS` (default: up to 32 allowed
CPUs). Marginal sums keep each column's donor order; independent stencil levels
keep their original arithmetic. These changes preserve conservation checks and
prepared-area rounding limits. They do not establish a whole-move speedup on
every mesh; use the stage receipts for the actual case.

The spherical-overlap builder retains its independent polygon-coverage check
and declared-area gates (recipient relative error at most `1e-10`, donor at
most `2e-14`). Weakly connected overlaps can make alternating area scaling
stall. After eight ordinary passes, a positive Newton/CG correction accelerates
the same reconciliation, with component null modes projected out. It never adds
an overlap or relaxes either gate. Incompatible component totals still refuse.
Operator receipts record the original floor, accepted CG updates, final errors,
FP32 constant reproduction and reconciliation wall time. Progress goes to stderr;
the `moving operator` command keeps one final JSON document on stdout.
