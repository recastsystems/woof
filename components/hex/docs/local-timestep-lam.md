# Local time stepping on limited-area culls: what it is worth, measured

The local-time-stepping part of the 2026-09-13 performance campaign,
measured 2026-09-14 on an RTX 5090 (170 SMs, 32 GB) with public gpuwm 2.7.3 as the
engine. Two culls, both 1 h = 720 steps of 5 s, full physics, radiation
every 600 s, surface/PBL welded to every step:

| cull | cells x levels | finest edge | driving data | rings |
|---|---|---|---|---|
| point `q0.9375.120.43884.n41.95w94.79` (`mesh-plan --point`, 1.35x cut) | 43,884 x 55 | 841.3 m | HRRR 2026-09-13 15Z through `woof hex intermediate` | 7 (2,124 cells) |
| corridor `r0.9.120.40520` (`cull --region`, 125 km cap) | 40,520 x 55 | 869.3 m | GFS 2026-09-10 18Z through `woof hex init` on the 937.5 m parent, culled | 7 (2,468 cells) |

Seven arms ran in one session on the one card (05:39-06:17 UTC), the
baseline arms from the untouched 0.3.0 tree in the shared 2.7.3 venv and
the option arms from this change's tree in its own venv; every receipt
records which tree ran it (`repo` in `forecast-receipt.json`).

## The decision

**Local time stepping stays selectable and stays off by default on every
mesh class.** On the limited-area culls the doors make it is inert by
construction (one class, history byte-identical to the default run), and
when an interface is forced into the interior to see what the option would
do if it had a coarse class, the step gets 7 % slower, not faster, and the
fields move. Nothing here is a tuning problem; the reasons are structural.

| mesh class | cells earning rate 3 from spacing | where they sit | executed classes | s/step against the default, same card |
|---|---|---|---|---|
| point cull (7 rings, 1.35x cut) | 2,324 of 43,884 (5.3 %) | 2,065 in rings 1-7, 259 one cell inside ring 1 | one (identity permutation) | 0.992x, history byte-identical |
| corridor cull (7 rings, 125 km cap) | 692 of 40,520 (1.7 %) | all 692 in rings 3-7 | one (identity permutation) | 0.989x, history byte-identical |
| point cull, interface FORCED into the interior (instrument) | 6,206 at rate 3 (14.1 %), 1,456 interface edges | 99-135 km from the point | two | 1.071x (slower), fields differ |
| corridor cull, interface FORCED (instrument) | 4,737 at rate 3 (11.7 %), 1,573 interface edges | 99-125 km from the centre | two | 1.068x (slower), fields differ |
| published global VR `x4.163842` | 23.0 % acoustic saving admitted | interior | two | 0.988x (2026-08-24, RTX 5070 Ti, README) |
| quasi-uniform global `x1.40962` | none | -- | one | bit-identical (2026-08-24) |

Why the coarse class is empty on a cull: `mesh-plan` cuts the cull at 1.35
times the fine radius and the cull door cuts a cap, so the cell spacing only
reaches three times the finest edge in the last ~20 km before the cut
(chart 04, top row), and that band is exactly where the seven driven rings
are. Every driven cell is held at rate 1 (below), and the one-cell buffer
demotes the thin ring of interior cells next to ring 1. Nothing is left to
class, and the run executes the option-off arithmetic: both history frames
of `point-lts` carry the same SHA-256 as `point-baseline`
(`85cb8196...`, `4ed8001f...`), and both frames of `swath-lts` the same as
`swath-baseline` (`a7271333...`, `9c51ea33...`).

Why a coarse class would not pay even if the cut were wider: the forced
arms have the coarse class the shipped classing lacks, with an arithmetic
acoustic saving of 8.5 % (point) and 7.0 % (corridor) of the sub-step cell
work, and they are 7 % SLOWER per step. The card explains it. Every hex
dycore kernel on these meshes runs in a single wave (a cell kernel over
43,884 cells at 128 threads per block is 343 blocks on 170 SMs; measured by the
profile instrument), so a launch's wall is the per-thread column latency, not the
cell count. The fine class at 86 % of the cells (295 blocks) costs what the
whole mesh costs, and each coarse-class launch (49 blocks) is a further
wave of the same latency. The coarse class is active on 4 of the 10
acoustic sub-steps of a step, so the option adds about four sub-steps'
worth of cell-kernel time and removes none: measured +21 ms per step on the
point cull and +19 ms on the corridor, against an acoustic loop that is a
minority of the 0.30 s step. The README's box-mesh projection (0.979x on a
38,857-cell mesh) is the same finding from the other side, and the ceiling
argument there still holds: the released `(1, 3, 6)` schedule admits only
rate 3, so a coarse column still runs 4 of 10 sub-steps.

**Recommendation per mesh class:** default off everywhere. What would
change it is a mesh whose classes each hold several waves of blocks on the
card that runs it (an order of magnitude more cells than these culls, with
most of them coarse), or a way of overlapping the classes' launches on
concurrent streams; neither exists in the tree, and the instrument this
change ships (`tools/lts_forced_classing.py`) is what to re-run when one does.

## Measured

### Speed

Host wall per model step from the driver's step receipts (the composite
transaction, closed by a stream synchronize; the first step carries kernel
compilation and is excluded), median over steps 2-500, both meshes, one
card, one session. The corridor-forced arm's window stops at 500 because
another process arrived on the card at about step 560 (below).

| arm | tree, venv | median s/step (2-500) | ratio | receipt s/step after first (720 steps) | door wall | first step |
|---|---|---|---|---|---|---|
| point-baseline | untouched, shared | 0.2992 | 1.000 | 0.3155 | 329.5 s | 3.8 s |
| point-lts (option on, shipped classing) | change, own | 0.2967 | 0.992 | 0.3130 | 365.4 s | 35.0 s |
| point-lts-forced (instrument) | change, own | 0.3205 | 1.071 | 0.3370 | 344.2 s | 2.9 s |
| swath-baseline | untouched, shared | 0.2844 | 1.000 | 0.2991 | 311.0 s | 2.8 s |
| swath-lts (option on, shipped classing) | change, own | 0.2812 | 0.989 | 0.2956 | 306.0 s | 2.5 s |
| swath-lts-forced (instrument) | change, own | 0.3037 | 1.068 | 0.4049 (co-tenant) | 398.7 s | 2.6 s |

Host load at each arm's start was 0.8-1.1 (1-minute average) with no other
GPU process; utilisation during integration 88-95 % (busy mean) at
274-295 W. The option-on single-class arms are 1 % faster than the
untouched tree at the same arithmetic, which is inside what two runs of the
same tree show and is not claimed as a saving.

The 35 s first step of `point-lts` is the cold NVRTC compilation of the
derived local-timestep translation unit (global trio, regional trio,
damping, reflux, recovery: fourteen kernels) into the run's scratch cache;
the later option arms found it in the interpreter's kernel cache and paid
2.5-2.9 s. A cold `--local-timestep` run costs about half a minute before
its first step on this card.

Peak card memory from `nvidia-smi` every 2 s (whole card): 8,914 MiB
(point-baseline), 8,916 (point-lts), 8,944 (point-lts-forced), 8,872
(swath-baseline), 8,872 (swath-lts), 8,898 (swath-lts-forced before the
co-tenant). The option adds 2 MiB with one class and about 30 MiB with two
(the 55 x 131,957 float32 residual array and the index lists). One sample
of point-lts read 9,423 MiB and the next 8,916: an allocation that comes
and goes is not ours, because CuPy's pool keeps what a process touches.

**The co-tenant.** From 06:15:17 UTC (about step 560 of `swath-lts-forced`)
another process held 17-18.5 GB and 100 % utilisation on the card (whole
card 25.9-27.4 GB; chart 05). Steps 600-720 of that arm averaged 0.832 s
against 0.304 s before it, so the receipt's integration total (293.7 s,
0.408 s/step) is contaminated and the number of record for that arm is the
window median above. Nothing was killed or reniced; the process was not
part of this measurement.

### Fields at 1 h

`point-lts` and `swath-lts` are byte-identical to their baselines (every
frame's SHA-256 equal). The forced arms differ, as they must: a different
number of acoustic sub-steps in 12-14 % of the columns is different
arithmetic. What the difference looks like, forced minus baseline at 1 h,
from `lts-compare-full.json` and the per-cell tables (`cells-*.csv`):

| | point cull | corridor cull |
|---|---|---|
| domain dry mass, relative | -5.6e-6 | -5.4e-6 |
| psfc mean difference, interior | -0.97 Pa | -0.57 Pa |
| theta max abs (where) | 7.29 K (cell 12,930, level 30, fine interior) | 10.7 K (cell 13,509, level 28, fine interior) |
| w max abs (where) | 23.7 m/s (cell 3,979, level 29: the baseline's hot cell) | 16.2 m/s (cell 13,265, level 35) |
| theta column RMS: fine interior more than 10 km inside the interface | 0.101 K | 0.062 K |
| theta column RMS: interface cells | 0.098 K | 0.138 K |
| theta column RMS: coarse class | 0.093 K | 0.138 K |
| w RMS: fine interior / interface / coarse | 0.74 / 0.52 / 0.56 m/s | 0.56 / 0.75 / 0.79 m/s |
| psfc RMS: fine interior / interface / coarse | 8.3 / 9.9 / 8.3 Pa | 3.1 / 7.3 / 8.0 Pa |
| t2 RMS: fine interior / interface / coarse | 0.037 / 0.060 / 0.054 K | 0.044 / 0.292 / 0.274 K |
| rainnc max abs | 0.75 mm (fine interior) | 9.9 mm (coarse class; class RMS 0.39 against 0.04 in the fine interior) |
| rings 1-5 (relaxation), theta RMS | 0.012 K | 0.042 K |
| rings 6-7 (specified) | 0 in every field | 0 in every field |

Two different pictures (chart 04, bottom row). On the point cull the
difference peaks 80-95 km from the point, in the fine interior where the
HRRR-driven convection is (max |w| differences of 15-24 m/s there), and
falls off through the coarse class: a perturbation anywhere in a
convecting domain reaches this size in an hour, and the interface is not
distinguishable from that. On the corridor cull the interior inside 50 km
is quiet (theta RMS below 0.01 K) and the difference rises from 55 km to a
peak inside the coarse class: there the interface is the source, and the
coarse class carries two to six times the difference of the fine interior
in theta, psfc and t2, with the 2 m temperature and the hourly rain the
most affected surface fields. The domain mass and surface pressure are
lower in both forced arms by about five parts per million; with a driven
boundary the domain is not closed, so this cannot be split between the
reflux and the changed boundary fluxes from one pair of runs, and it is
recorded rather than explained.

None of this is graded against observations, and none of it needs to be:
the instrument arm is not a candidate default. Should a mesh class ever
make the option pay, the field change measured here is the size of the
thing that would then have to be graded.

### Card time

Chain 1 (05:39:37-05:47:05 UTC, 7 min): the two swath arms that ran
through the copied venv's console script, whose shebang still named the
shared venv's interpreter, and so ran the UNTOUCHED tree; `swath-baseline`
from that chain is the baseline of record, `swath-lts` from it reproduced
the untouched tree's refusal on a cull and was replaced. Chain 2
(05:47:18-06:17:09 UTC, 30 min): the four point arms and the two swath
option arms through `python -m hexcore` of each venv's own interpreter.
36 card minutes in all; the resumed session (2026-09-14, 16:00 UTC on) took
no card time, and the comparison, the audit and this document came from the
receipts. Budget was 40 minutes for each session.

## What changed in the tree

Before this change, `--local-timestep` on any limited-area cull failed, and
had it not failed it would have computed wrong arithmetic. Each item below
is tested in `tests/test_local_timestep.py`.

1. `lts_v841.classify_local_timestep` raised on the `0` a limited-area grid
   file writes for the missing second cell of a ring-7 edge
   ("cellsOnEdge references a cell outside the mesh"; the
   `point-untouched-lts-on` arm reproduces it, 41 s, rc 2). Fixed: a
   one-cell edge takes its present cell's rate and is never an interface.
2. `attach_local_timestep` compared the classing against the driver's cell
   and edge counts, which on a regional driver carry one padded garbage
   element each (`PaddedRegionalHostMesh`), so the classing would have been
   refused as "not the mesh the driver was built on". Fixed: the padding is
   counted, and the garbage cell and edge are appended to the rate-1 class
   so every launch list still covers the padded extent.
3. The attachment rebound the GLOBAL acoustic sub-step
   (`cuda_driver.advance_acoustic_step_cuda_v841`), the vertical
   coefficients, the divergence damping and the recovery. The pinned RK loop
   selects `cuda_regional_forecast_v841.advance_acoustic_step_regional_cuda_v841`
   for a regional driver (`cuda_driver.py:3745-3767`) and never calls the
   global one, so a regional run would have built class-rate vertical
   coefficients and divided the `ru_avg`/`ww_avg` time averages by
   class-rate counts around a sub-step still advancing every column at the
   global rate. The first sub-step's divergence damping would then have
   raised "ran without a preceding local-timestep acoustic sub-step", a
   refusal naming the wrong breakage. Fixed: the regional sub-step is a
   fourth rebinding, its three kernels derived from the regional
   translation unit by the same asserted gather (`REGIONAL_GATHERED`), so
   the specified-zone masking is byte for byte the pinned text.
4. Nothing kept a class interface out of the driven boundary zone. A
   specified-zone cell is advanced by the lateral-boundary tendency inside
   the acoustic loop, not solved, and the relaxation zone is relaxed on a
   schedule built for one global rate; a reflux residual settled into
   either is a mass source nothing accounts for. Fixed: every cell with
   `bdyMaskCell > 0` is held at rate 1 before the buffer pass, the held
   cells are recorded per ring in the receipt, and a classing whose
   interface still touches a driven cell is refused on the driver by name.
5. An explicit classing's summary reported the instrument's requested
   rates under `cells_qualified_by_spacing`. Fixed: that key is `None` for
   an explicit classing and the request goes out as `cells_requested`, so
   a receipt cannot present an instrument's choice as a property of the
   mesh. (The classing receipts kept with the release evidence predate this and
   carry the old key.)

The default path is untouched: with the switch off nothing in these
modules runs, and the contract deck's kernel-set digest is the same on this
change's tree as on the base tree, which is why the point cull's own contract
receipt admits both.

## The instrument: an interface in the interior

Because the shipped classing leaves one class on a cull, "what does a class
interface do to the fields" cannot be measured with the option's own
classing. `tools/lts_forced_classing.py` writes a per-cell rate file (rate 3
for interior cells at spacing ratio at least 2.0, rate 1 elsewhere) and
`--local-timestep-classing FILE` on the door and the tool runs it. The
driven-zone hold and the buffer still apply; the receipt records
`rate_source: explicit` so the arm cannot be mistaken for the option. The
coarse class runs at an acoustic Courant number of 0.17 (point) / 0.16
(corridor) against 0.11 for the fine class, inside the split-explicit
stability region; every step of both forced arms passed the health gate,
and the corridor arm's peak |w| at 1 h (27.55 m/s) is the baseline's own
hot cell (27.43 m/s).

Point cull: 6,932 requested, 726 demoted by the buffer, 6,206 classed
(14.1 %), spacing 1,688-2,752 m, 98.8-134.6 km from the point (median
115.4), 1,456 interface edges (1.10 % of edges), 738 coarse cells settle a
residual each stage. Corridor: 5,521 requested, 784 demoted, 4,737 classed
(11.7 %), spacing 1,742-2,150 m, 98.8-124.7 km (median 112.3), 1,573
interface edges (1.29 %), 795 settling cells. Launches per hour: 30,240
acoustic class launches and 6,480 reflux settles against 21,600 launches
and no settles with one class.

## Audit against the design it descends from

The implementation is Berger-Colella refluxing on the acoustic mass flux
with a one-ring buffer, applied to the split-explicit acoustic sub-step
only (the RK stages, the transport and the physics run at one rate). Native
MPAS-A v8.4.1 has no local time stepping (`Registry.xml:64-68`,
`mpas_atm_time_integration.F:2053-2092`), so the design is the tree's own;
the nearest published MPAS designs are the MPAS-Ocean shallow-water schemes
of Hoang et al. (2019) and Capodaglio and Petersen (2022), which reach
conservation by advancing two interface layers with predicted coarse fluxes
and need no reflux. Read against its own statements, with the pinned loop
open beside it (`cuda_driver.py:3454-3810`):

- **Stage detection.** `_StageTracker` infers the stage from
  `small_step == 1` and checks each stage's observed sub-step count against
  `(1, 3, 6)` when the next stage begins. The vertical coefficients pick the
  stage from `dts` instead (`_stage_steps_for_dts`), so RK1's single
  sub-step is never handed class-rate coefficients and RK2/RK3 share one set
  built for the smaller count. Both refuse on a schedule they do not know.
- **Class activity.** `_active_classes`: the rate-3 class advances on
  sub-step 1 of RK2 (one step of 3 dts) and sub-steps 1 and 4 of RK3 (two
  steps of 3 dts); RK1 has one sub-step and every class takes it. Four of
  ten sub-steps, as the ceiling argument assumes.
- **Buffer ring.** One pass of neighbour-minimum demotion after the
  driven-zone hold and before the edge rates and interfaces are derived,
  counted in the receipt; a one-cell edge contributes no demotion.
- **Edge rate.** The finer of the two cells' rates, so an interface edge is
  in the fine edge list and its `ru_p` is refreshed every fine sub-step.
- **Coupling within a coarse sub-step.** The fine side reads the coarse
  cell's `rho_pp` and `rtheta_pp` as constants until the coarse class next
  advances (no time interpolation), and the coarse cell's own update at
  its sub-step reads the interface `ru_p` the fine class just produced.
  This is the approximation the buffer ring exists for, and it is where
  this design differs from the interface-layer schemes above. Not a defect;
  its price is the corridor cull's coarse-class field difference measured
  above.
- **Reflux.** `lts_reflux_accumulate` runs after the momentum update and
  before the cell kernels of every sub-step, so both sides integrate the
  same `ru_p` samples the cell kernels apply: the fine side adds
  `+dts_fine*dvEdge*ru_p` on every fine sub-step, the coarse side
  `-dts_coarse*dvEdge*ru_p` on its own; `+-1` times a binary32 is exact,
  so the magnitudes are the ones `acoustic_rs_ts` used. `lts_reflux_settle`
  applies the residual to the coarse cell only, one thread per coarse cell
  over its own interface slots in ascending edge order (no atomics, so a
  dual run stays byte-identical), mirroring `acoustic_rs_ts` term for term
  including the `0.5*(theta_m[c1]+theta_m[c0])` theta flux with the same
  `saved_diag.theta_m` the sub-step's forcing carried
  (`cuda_driver.py:3706-3722` against `:3806`). In RK1 both sides
  accumulate the same product with opposite signs and the residual is
  exactly zero.
- **Divergence damping at the interface.** The pinned loop captures its
  own `rtheta_pp_old` over every cell before each sub-step
  (`cuda_driver.py:3726`, `cuda_horizontal.py:1563`), so on a fine sub-step
  where the coarse class rests the coarse cell's delta is zero, and the
  interface edge damps the coarse jump once, at the coarse sub-step, with
  the fine coefficient: over a stage the edge receives what the global-rate
  run would apply. An edge with both cells coarse damps once per coarse
  sub-step at one third of the coefficient, which is the damping a model
  running at 3 dts applies. Consistent with the per-class coefficient the
  module declares.
- **Time-average recovery.** `lts_recover_edges`/`lts_recover_interfaces`
  divide `ru_avg`/`ww_avg` by the count each entity ran, with the same
  `mpas_add(saved, mpas_div(avg, (float)count))` arithmetic as the pinned
  `recover_edges_f32`/`recover_interfaces_f32`, so rate-1 entities' fluxes
  are bitwise the pinned values. They run after the stock recovery, whose
  regional specified-zone overwrites touch `ru`, `u` and `w` and not the
  fluxes, so nothing is undone. With one class they do not run at all.
- **Regional sub-step.** Launch for launch the pinned regional sub-step
  (`cuda_regional_forecast_v841.py:1019`) with each launch per active class:
  the shared `acoustic_prepare` through the global gather, the three
  regional entrypoints through their own, `rs`/`ts` zero-initialised as
  the pinned code does. Every specified and relaxation cell is at rate 1.
- **Where it is NOT proved:** a class interface inside the relaxation zone
  (refused), a partitioned run (refused), a limiter-crossed interface
  (refused), and the run-to-run byte identity of a forced arm on a cull
  (the dual-run identity of the option is measured on the global x4 mesh,
  README; each forced arm here ran once). None of these refusals was
  changed.

Two things read and left alone: the multi-class recovery reruns the two
recovery kernels over the whole padded extent after the stock ones (a
harmless duplicate of a few hundred microseconds per stage, one of the
bookkeeping costs the 7 % is made of), and the derived translation unit's
half-minute cold compile.
