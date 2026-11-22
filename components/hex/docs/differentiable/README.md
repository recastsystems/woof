# The differentiable engine

Scope: the tangent-linear and adjoint model of the exact-mode hrrrv5 forecast step, dynamics and physics, reached
through four commands on three examples, on one card of compute capability 12.0: the 12 km cut of the storm box
(2,890 cells, 15 steps of 60 s) on a 32 GB GeForce RTX 5090, and the 3 km storm box (38,278 cells, 59 steps of
15 s) and a 120 km global example of three steps (36 minutes) on a 96 GB RTX PRO 6000 Blackwell. Start with the
12 km example ([FIRST-HOUR.md](FIRST-HOUR.md)); the 120 km tables below were this page's first measurements. On
the 12 km and 3 km examples `adjoint-test` ends PASS on the default route ([LIMITS.md](LIMITS.md)). The checks the commands print are consistency checks: they show that the adjoint is the transpose
of the tangent-linear model. They do not show how well the gradient predicts the forecast; that is measured
separately ("How far the gradient can be trusted", below). A measure built on composite radar reflectivity does not
pass the gradient check, and the commands refuse it by name. The derivative of the boundary layer is of MYNN-EDMF
c6d08709 and that of the surface layer is of the MYNN surface layer v6.0.0, where a plain hrrrv5 forecast runs v6.1.1 and
v6.1.0 ("What each row differentiates", below). On the 3 km storm row the default boundary-layer route differentiates more of that boundary layer than the exact route does (same section).

New here: [FIRST-HOUR.md](FIRST-HOUR.md) (which cards, what each example needs, the first command's output line
by line), [GUIDE.md](GUIDE.md) (your own measure, the checks, recordings, the C library, exact and fast mode, the
bit-for-bit claims) and [LIMITS.md](LIMITS.md) (every measured limit).

## Who it is for, and what to bring

The engine is built for labs and agencies with large NVIDIA cards, 48 to 96 GB, one to eight of them. What runs in
this release is narrower, and you should know it before you start:

- **One card.** A gradient runs on one card. Splitting one gradient over several cards (`--cards 0,1` on `adjoint-test` and `sensitivity`) is experimental: a split gradient has not yet been shown equal to the one-card gradient, nor faster than it, and the command says so before it runs and in its receipt.
- **The newest card family only.** The derivative images and the example plan are built for compute capability 12.0.
  The 12 km example has run on a 32 GB GeForce RTX 5090; the 120 km and 3 km examples on a 96 GB RTX PRO 6000
  Blackwell. On an A100, H100, L40S, RTX
  A6000 or RTX 6000 Ada the commands stop before they download anything and name the family they have no images for.
- **Free card memory, 120 km example: 73,156 MiB** for the default row, **80,000 MiB** for the full-physics row. A
  card with less free is refused before any download: every 48 GB card, and an 80 GB card for the full row.
  `--wait-card SECONDS` waits for memory another job is holding. The 12 km example needs 28,700 MiB and the 3 km
  storm box 96,600 MiB ([FIRST-HOUR.md](FIRST-HOUR.md), section 1, has each example's needs).
- **Free host memory, 120 km example: 81,920 MiB** (default row) or **110,592 MiB** (full row); 47,104 MiB for the
  12 km example. With less, the command stops before any download with exit code 3.
- **Linux, Python 3.14, and the `ad` extra.** The example replays three CUDA library files byte for byte: cuBLAS of
  cuda-toolkit 13.4.3, and the CUB and Thrust modules of cupy-cuda13x 14.2.0 as built for CPython 3.14. The `ad`
  extra pins both. In another Python, or with another extra, the commands stop before any download and say what to
  install.
- **About 2.4 GB of download** on first use (four archives, checked by SHA-256), and 5.6 GB in the hex cache
  once unpacked.

## Install and run

Install the `ad` extra in a Python 3.14 environment of its own, not on top of `[gpu]`: each of the two extras brings its own CuPy package, one environment holds one, and pip does not remove the other. `[ad]` installs the CUDA 13 CuPy, so it needs a driver that serves CUDA 13; the forecast commands of this release were run from a `[gpu]` environment. `woof hex doctor` reports two CuPy packages as a required gap and prints the command that leaves one. The runnable example scripts travel in the wheel: `woof hex examples` prints their folder,
and `woof hex examples --copy DIR` copies them.

```bash
python3.14 -m venv hexad && . hexad/bin/activate   # Debian and Ubuntu: sudo apt install python3-venv first
pip install "recast-woof[ad]"
woof hex adjoint-test --out at
woof hex tangent      --steps 3 --out tl
woof hex sensitivity  --steps 3 --metric theta_m --box 30,48,-105,-80 --levels 1-10 --out se
woof hex param-grad   --steps 3 --check --out pg
```

The first line needs a `python3.14`: Ubuntu 26.04's system Python is one; Ubuntu 24.04's is 3.12, and there the
line stops with `python3.14: command not found` (install a Python 3.14 first, or use Ubuntu 26.04).

Add `--row hrrrv5-full-x1.40962` to any of them for the full-physics row. Each command prints what it needs before
it starts and what it used at the end, and writes `receipt.json` in its `--out` folder: its numbers, every stage's
wall time, the card's and the host's peak memory, the runner and plan SHA-256, and whether the receipt is valid. It
exits 0 when it passes, 1 when it ran and did not pass, 3 when the host has too little memory, and 2 when it refused
for any other reason; a refusal is one sentence on its last line.

Offline: put the four pinned files in a directory and name it with `WOOF_HEX_AD_ASSETS_FROM`, or name an unpacked
asset root with `WOOF_HEX_AD_ASSETS`. `WOOF_HEX_NO_LOCAL_GPU=1` stops every command before it reads a card.

## What the commands use

The 120 km example, 3 steps, one 96 GB RTX PRO 6000 Blackwell card, a fresh install with an empty home, one run of
each command:

| command | row | card memory at peak | host memory at peak | seconds |
|---|---|---|---|---|
| `adjoint-test` (first command: includes the download) | `hrrrv5-x1.40962` | 71,156 MiB | 74,775 to 74,840 MiB | 193.8 to 194.6 |
| `tangent` | `hrrrv5-x1.40962` | 71,156 MiB | 74,822 to 74,867 MiB | 137.8 to 140.4 |
| `sensitivity` | `hrrrv5-x1.40962` | 71,156 MiB | 74,838 to 74,852 MiB | 158.9 to 159.4 |
| `param-grad --check` | `hrrrv5-x1.40962` | 71,282 MiB | 74,774 to 74,847 MiB | 165.4 to 168.0 |
| each of the four | `hrrrv5-full-x1.40962` | 78,206 to 78,328 MiB | 104,562 to 104,792 MiB | 177.8 to 267.3 |

Each command's seconds include its own strict check, the whole-state dot test and its proof, which run two extra
windows. After an `adjoint-test` on the same row, `--strict-from at/receipt.json` lets `tangent` and `sensitivity`
take that check instead of running it again. The backward pass alone of `sensitivity` took 22.8 s on the default row
and 33.0 s on the full row. The table's `adjoint-test` seconds are of an earlier build that ran the row's 3 steps
and one test direction, the bump. A later run of the same commands on a later build of this release (the README's
table and `examples/differentiable/expected.json`) read card 71,156 to 71,282 MiB, host 74,860 to 74,923 MiB,
`tangent` 158.1 s, `sensitivity` 175.2 s and `param-grad` 186.1 s on the default row, and 78,206 to 78,328 MiB,
104,871 to 105,017 MiB and 186.0 to 222.0 s on the full-physics row. `adjoint-test` now runs its own 1-step window
with every test direction (section 1): 271.9 to 307 s, card peak 60,734 MiB, host peak 73,994 to 74,072 MiB on the
default row (two runs), and 305.6 s, card peak 76,114 MiB, host peak 74,333 MiB on the full-physics row.

## How far the gradient can be trusted

The dot tests say nothing about this. It was measured by running the real model twice, from a start and from that
start plus a perturbation, and running the tangent-linear model once along the same perturbation. The score is the
fraction of the real model's change the tangent-linear model explains, over all levels (1 is perfect, 0 is no better
than predicting no change). This page takes 0.7 as the line.

| mesh and setting | trusted to | at that window | where it falls off |
|---|---|---|---|
| 120 km, three dates, mountain-drag closure held (as in the default row) | **about 3 hours** | temperature 0.78 to 0.82, wind 0.90 to 0.91, moisture 0.76 to 0.79 | 6 hours: temperature 0.61 to 0.64, moisture 0.54 to 0.59 |
| 120 km, mountain-drag closure differentiated (`param-grad` on either row, and rows made with `--suite hrrrv5-exact`) | **about 1 hour** | | after about an hour; for wind and near the ground from the first window on one of the three dates |
| 60 km, two places, one date; one place, two dates | **about 3 hours** | temperature 0.65 and 0.87 at 2 h 54 min, 0.63 and 0.81 at 3 hours | 6 hours: temperature 0.15 and 0.53 |
| 3 km storm box, one storm, low-level warm bump | **14 min 45 s** for temperature (0.92); not for moisture | moisture 0.30 | moisture 0.13 already at 5 minutes; not tried past 14 min 45 s |
| 3 km storm box, 1 K nudge aloft | temperature 5 minutes; moisture 10 minutes | temperature 0.98 and moisture 0.95 at 5 minutes; moisture 0.87 at 10 minutes | temperature 0.60 at 10 minutes (0.85 at half size) and -0.19 at 14 min 45 s; moisture 0.53 at 14 min 45 s (0.81 at half size) |

Six hours is not claimed on any mesh. The 120 km rows were measured with the cloud-microphysics tangent in single
precision; the default row runs it in double, and these rows were not rerun with it. The 60 km and 3 km rows are
measurements of the engine, not commands of this release (the 3 km storm-box example's own `adjoint-test` reading
is in [LIMITS.md](LIMITS.md)).

## Larger meshes: what box to bring, and how long a gradient takes

> **In WOOF 1.1.7** recording a plan from a start of your own (`--init FILE`, `--init example`) is not supported: the derivative commands run on the example rows' own plans (`--row`).

These were measured with the engine on one 96 GB RTX PRO 6000 Blackwell card. The 60 km mesh and the larger 3 km box are not commands of this release. The 3 km storm box (38,278 cells) ships as a second example: `--row hrrrv5-q3.96.38278.n40.60w91.30` downloads its mesh and a one-hour plan of the HRRR 2023-03-31 18Z start (about 2.4 GB more; 3 steps when `--steps` is not given, up to 239; the README's commands give `--steps 59`), and `--init example` records a plan of its own from that start (not yet run through to a gradient on a card in this release; [GUIDE.md](GUIDE.md), section 4). It needs one 96 GB card with 96,600 MiB free on the default route of `sensitivity` and of `adjoint-test` (93,000 MiB on the exact route, and for `tangent` and `param-grad`). A 59-step `sensitivity` is admitted on 258,057 MiB of host memory for a first call (the answer check; it held 241,895 MiB) and on 249,661 MiB as a later command, which keeps its forward for the next gradient (it held 233,216 MiB; 158,617 MiB with `--keep-forward off`), and the command states its need and stops on a host with less; the three-step windows need 110,592 MiB. On this row the boundary layer's backward runs without its tape by default (the tape-free route). From this build it is a different linear model from the exact route: it differentiates the boundary layer's turbulence closure, the plume's transport with the plume's own coefficients held, and the boundary-layer height the column returns, which the exact route holds at the forecast's values, so its answer differs from the exact route's by design; the size of that difference was not measured on this build. `--boundary-layer taped` runs the exact route. Through the command on one 96 GB card, `sensitivity` and the 59-step `adjoint-test` (PASS on the default route: the number cloud route and the live transport limiter; [LIMITS.md](LIMITS.md)) have run on this row.

| mesh (cells) | window | card memory at peak | host memory at peak | one gradient |
|---|---|---|---|---|
| 120 km global (40,962), the example | 3 steps, 36 min | 71,156 MiB | 74,774 to 74,923 MiB | 158.9 to 175.2 s for the whole `sensitivity` command (22.8 s backward) |
| 60 km global (163,842) | 29 steps, 2 h 54 min | 85,080 MiB, with the step tape streamed through the host | 161 to 193 GiB | 958 s from a recording already made; 1,163 s for the first gradient of a recording; recording 43 to 78 min |
| 3 km storm box (38,278) | 59 steps of 15 s, about 15 min | through the `sensitivity` command 89,966 MiB and 91,648 MiB on the default route, 89,040 MiB on the exact route | through the `sensitivity` command: default route with the forward kept, admitted on 249,661 MiB and peaked at 233,301 MiB; exact route, no forward kept, 158,617 MiB and 151,835 MiB; a first call with its answer check, admitted on 258,057 MiB, peaked at 241,895 MiB | through the `sensitivity` command, from the packed recording (measured once): default route, a first gradient 239.5 s, a further target in the same command 180.5 s, a later command that attached 181.1 s in the shell; exact route, 289.6 s first and 260.8 s further; a new forecast is first recorded, about 28 min |
| 3 km box of 135,318 cells | 4 steps only | 80,696 MiB, tape streamed | 317.2 GiB | not timed |

The card figures sit near the card's size because the step tape and the working areas are given budgets that fill
the card: they show the job fits a 96 GB card, not what a smaller card would need. On a 96 GB card a 3 km storm-box
window of 240 steps (60 minutes) has run with 275 GiB of host memory.

## What each row differentiates

With no `--row`, the commands run `hrrrv5-x1.40962`: the RRFS hrrrv5 suite on the 40,962-cell global mesh, 59
levels, on its example plan, a hex forecast from 2023-03-31 12Z recorded for four steps of 720 s, of which the
commands replay three. Every receipt carries the row's own words (`row_what`).

| | `hrrrv5-x1.40962` (default) | `hrrrv5-full-x1.40962` |
|---|---|---|
| radiation and cloud fraction | held: 64 kernel kinds are refused by name and listed in the receipt | in: one long-wave and short-wave call falls inside the three steps, and no kernel is refused |
| transport limiter | coefficients held | differentiated |
| boundary layer (MYNN-EDMF c6d08709, the code RRFS 44c0849 runs; not the v6.1.1 a plain hrrrv5 forecast runs) | turbulence closure and mass-flux plume held | moisture mixing live, plume speed on a ramp |
| land (RUC) | differentiated | bounded derivative |
| mountain-drag closure (GSL drag) | held | held (earlier builds of 0.6.0 differentiated it here, trusted to about 1 hour) |
| cloud microphysics (TEMPO) | tangent in double precision | tangent in single precision |
| identity kernels | two set-up kernels of the first step that read no model state (`mynnsfc_tables`, `w_filter_geometry_v841_f32`) | none |

On both rows 9,337,888 bytes of the transport limiter's work memory are left out of the whole-state dot test; each
command proves on the plan it runs that those bytes are never read, and prints the proof line. `param-grad` runs its
own scheme bundle in place of the row's: on both rows the boundary layer's turbulence closure and plume are held and the mountain-drag closure is
differentiated. The command prints this (`operator:`) and writes it in its receipt.

**Which boundary-layer and surface-layer code the derivative is of.** The tangent and the adjoint of the boundary
layer are generated from MYNN-EDMF at commit c6d08709, the code NOAA's RRFS 44c0849 runs (in a forecast,
`--physics config_mynn_version=pinned`), and those of the surface layer from the MYNN surface layer v6.0.0. A plain
hrrrv5 forecast of this release runs MYNN-EDMF v6.1.1 and the MYNN surface layer v6.1.0 by default; their derivatives are
not in this release (the derivative of v6.1.1 is planned for the next build). On the 3 km storm row the default
(tape-free) route differentiates the turbulence closure, the plume's heat, moisture and momentum and the returned
boundary-layer height of that c6d08709 boundary layer, which the exact route holds (README, "The default route is
fast"). What the older code means for a sensitivity:

- **Cloud-topped boundary layers.** c6d08709 leaves the latent heat of the scheme's own cloud mixing out of the
  boundary layer's temperature tendency, and v6.1.1 has it. Where the boundary layer mixes cloud (stratocumulus, low
  stratus under an inversion, a thin cloud layer at the boundary-layer top) a sensitivity is that of the older
  boundary layer, not of the one a plain forecast runs. The size of the term, recomputed on one 3 km state of 38,278
  columns (one instant and one boundary-layer step's tendency, not a forecast difference): over 0.1 K/h in 8.2 per
  cent of the columns and over 1 K/h in 1.6 per cent. v6.1.1 also carries ice in the plumes and retunes them; none of
  that is in the derivative.
- **The surface layer.** v6.0.0 and v6.1.0 differ only at the first step of a run: how the land surface moisture is
  set there, and which starts reset the surface fluxes.

Both rows hold the mountain-drag closure at the trajectory; held, the full row's tangent explains 0.59 to 0.70 of
the real temperature change and 0.76 to 0.83 of the wind change at 6 hours on the three dates (120 km, tangent-only
runs, 30 steps, one RTX 5090), where with the closure differentiated it ran away after about an hour. Rows made with
`--suite hrrrv5-exact` differentiate it (on a global mesh they hold radiation, the transport limiter's
coefficients and two set-up kernels), so on a global mesh read their gradients as gradients for windows of up to about an hour (not measured at
60 km). Below about 3.6 km grid spacing the drag
scheme skips the mesoscale drag and blocking the closure belongs to, so on a 3 km storm box it has nothing to act
on, held or live.

## 1. `woof hex adjoint-test`

The checks of the tangent and the adjoint against each other and against the real forecast. The default bump is 1 K
of potential temperature at 20.4 N, 26.2 W, model level 32, 1,200 km radius, 4 levels wide; the default measure is
the mean modified potential temperature over every level within 1.25 bump radii (1,500 km), at the window end.

```bash
woof hex adjoint-test --out at
```

On this build `adjoint-test` on the 120 km rows runs its own 1-step window (12 minutes) at eight step sizes; that
window read PASS on both rows on an earlier build of 0.6.0 (`examples/differentiable/expected.json`). With
`--steps 3` it ends NOT PASSED (UNRESOLVED), exit status 1, because over three steps only the bump resolves
([LIMITS.md](LIMITS.md)). What `--steps 3` printed on an earlier build, which ran one test direction, the
bump (paths shortened); this build prints one block more, the test directions below, before the `whole-state dot`
line, and its last line is the verdict over them:

```text
woof hex adjoint-test: row hrrrv5-x1.40962 needs 73,156 MiB of card memory (card 0: 97,252 MiB free); 81,920 MiB of host memory (123,945 MiB available)
adjoint-test: hrrrv5-x1.40962, 3 steps (0.6 h)
  dot     3.466e-08   bar 2e-05   PASS
  fd      1.061e-03   bar 1e-02   PASS
  taylor  1.061e-03   bar 1e-02   PASS
  zero    0 nonzero words   PASS
  real model against the tangent, per bump size (h x the perturbation):
       h     metric err    field |FD - TL| / |TL|
        1   3.144e-03    0.0859
      0.3   1.061e-03    0.108
      0.1   1.489e-03    0.265
     0.03   3.881e-03    0.723
     0.01   3.042e-03    2.21
  whole-state dot  9.217e-08 relative = 0.77 roundings   bar 4 roundings (4.77e-07)   PASS   (105 state allocations; own windows; test direction 1)
    left out of the dot: 9337888 bytes in 1 ranges of 1 allocations (...); NaN-poison proof on this plan: PASS (...)
PASS  (receipt at/receipt.json)
used: card peak 71,156 MiB, host peak 74,775 MiB, wall 193.8 s
```

| line | what it compares | passes when |
|---|---|---|
| `dot` | the tangent's change of the measure, `<w, M x>`, against the adjoint's, `<M^T w, x>`, for the bump `x` and the measure weights `w`; relative difference | at most 2e-5 |
| `fd` | the tangent's prediction against the real forecast rerun with the bump added at size h and differenced; the best h that a neighbouring h confirms | at most 1e-2 |
| `taylor` | the adjoint gradient against the same reruns, over the five sizes; the best confirmed h | at most 1e-2 |
| `zero` | the tangent of a zero perturbation after the whole window | 0 nonzero words |
| the h table | per bump size: the measure's error, and the tangent against the rerun over the whole measured field | no bar: read it. At the smallest sizes the forecast's own rounding shows |
| `whole-state dot` | the adjoint seeded with the tangent result itself, every model field at the window end at unit weight (every state allocation when the plan's field map does not hold the window end: the line then says so), so one wrong transpose in any field shows | within 4 single-precision roundings of the dot (4.77e-07 relative); with `--dot-directions` 8 or more, also no one-signed drift |
| `left out of the dot` | memory given zero weight in the whole-state dot, and the proof: the same window with that memory filled with NaN gives the same gradient bit for bit | proof PASS |

`dot`, `zero` and `whole-state dot` are consistency checks. `fd` and `taylor` compare against the real forecast over
this window and this bump only. One random direction of the whole-state dot proves little near the bar: run another
with `--seed 2`, or run eight together with `--dot-directions 8`. On the earlier build the whole-state dot read
9.217e-08, 9.587e-09, 9.924e-08 and 3.445e-08 in directions 1 to 4: 0.77, 0.08, 0.83 and 0.29 single-precision
roundings of the dot, against a bar of 4. The bars are in the wheel's `data/ad-doors.v1.json`.

**The test directions.** One direction says little: a gradient can read right along a mid-level temperature bump
and be wrong near the ground. After the bump's window `adjoint-test` runs one more window for each further
direction, on the same measure and step sizes, and prints for each step size the real model's change over the
gradient's prediction (1 is a gradient that predicts the change exactly):

| direction | what it is |
|---|---|
| `bump` | the temperature bump above; it always runs, and its window gives the `dot`, `fd`, `taylor` and `zero` lines |
| `descent` | minus the gradient itself, in temperature and vapour, made from the first window's own gradient: the direction a minimisation follows |
| `near-ground-moisture` | vapour only, on the seven lowest levels: a smooth pattern inside the bump's radius (seed 211) |
| `smooth-draw` | temperature and vapour at every level: a smooth pattern inside the bump's radius (seed 307) |

A step size counts where the real change per unit step is steady over three neighbouring step sizes. A direction
is printed INSIDE when a step size that counts reads, with its spread, within the bar (0.01 of 1), and OUTSIDE
when its steadiest three step sizes all read beyond the bar by more than their spread. Anything else is
UNRESOLVED: no step size is steady (the measure changes less along the direction than the forecast's own
roughness), or the steady ones read within their own spread of the bar. An unresolved direction never counts, as
a pass or as a fail. The last line is PASS when the first window's checks and the whole-state dot pass, no
direction is outside the bar and at least two independent directions are resolved (two whose cosine is 0.99 or
more count once); FAIL names the directions outside the bar, or says that a check failed; NOT PASSED (UNRESOLVED)
says that fewer than two resolved. The command exits 0 on PASS only. `--directions` takes a comma list of the names
above, or `all`; the default is the four, and on the 3 km storm row `bump` and `descent`. The receipt lists each
direction with its seed and its verdict, and its `checks_passed` is the verdict of the first window's checks and
the whole-state dot alone, which is what `--strict-from` takes.

Measured on the 12 km example (15 steps, one card). On the exact route (`--boundary-layer taped`), whose boundary
layer holds the turbulence closure and the plume's heat and moisture, an earlier build of 0.6.0, run without
the whole-state dot, read `bump` 1.0001 at its best step size, INSIDE; `descent` 0.954 to 0.956 at four step sizes,
OUTSIDE (the real change is 4.4 per cent short of the gradient's prediction); `near-ground-moisture` 0.850 to 0.882
at three, OUTSIDE (12 per cent short); `smooth-draw` UNRESOLVED. That is a FAIL where the bump alone read inside the
bar. On this build's default route (its boundary layer differentiates them; README, "The default route is fast")
the same command ends PASS: the bump inside (9.9e-04 from 1), the descent inside (5.0e-04 from 1), near-ground
moisture and the smooth draw unresolved (7.2e-03 and 1.5e-02 from 1), and near-ground moisture is a known limit,
about 2 to 4 per cent short ([LIMITS.md](LIMITS.md)). On the 3 km storm box the default route ends PASS on the bump
and the descent (3.65e-04 and 1.92e-03 from 1). On the 120 km rows the 1-step window read PASS on an earlier build
and `--steps 3` ends NOT PASSED (UNRESOLVED) ([LIMITS.md](LIMITS.md)).

## 2. `woof hex tangent`

The tangent-linear forecast of the same bump. It writes `tangent.nc`, with each field's tangent at the window end
beside the base forecast's own value, and prints the largest and root-mean-square tangent of each field.

```bash
woof hex tangent --steps 3 --out tl
```

On the earlier build the largest temperature response was 0.9965 K and the largest pressure response 37.44 Pa.
`--fields` picks fields; `--at LAT,LON`, `--level`, `--amplitude`, `--radius-km` and `--level-width` move and shape
the bump (the same options work on `adjoint-test`).

## 3. `woof hex sensitivity`

The gradient of a scalar forecast measure at the window end with respect to the start state, with a map.

```bash
woof hex sensitivity --steps 3 --metric theta_m --box 30,48,-105,-80 --levels 1-10 --out se
```

The measure is the mean modified potential temperature over levels 1 to 10 in the box 30 N to 48 N, 105 W to 80 W.
It writes `sensitivity.nc` (`dJ_dtheta` per K and `dJ_dqv` per kg/kg at the start, the gradient to every stored
start field, and `latCell`/`lonCell` in degrees) and a whole-globe map of `dJ_dtheta` at the level that carries most
of the field (`--map-level` draws another, `--no-map` draws none; a `qv` measure's map draws `dJ_dqv`). On the earlier build the map showed a red patch over the box, up to 3.8e-4 K/K at
model level 8, and nothing elsewhere, as a 36-minute window should.

## 4. `woof hex param-grad`

The gradient of a measure with respect to five physics constants at their shipped values: the Charnock constant over
sea (`coare30_charnock`), the Zilitinkevich factor over land (`mynnsfc_zil_kczil`), and three GSL drag coefficients
(`tofd_beljaars_a1`, `mesoscale_gwd_xlinv`, `flow_blocking_cd`). Its default measure is the mean modified potential
temperature over every cell and level at the window end.

```bash
woof hex param-grad --steps 3 --check --out pg
```

`--check` adds the parameter dot test: the parameter tangent against the parameter adjoint. Like the other dot tests
it shows consistency, not that the gradient predicts the forecast. Against reruns of the forward with one constant
moved (120 km, 3 steps): on the earlier build, the rerun slope for the Zilitinkevich factor was 0.98 times its
gradient at a move of 0.02 in ln p and 1.11 times at 0.1, the only constant of the five tried there; on an older build with another measure, three of four constants disagreed
with their reruns (gradient over rerun 0.31, 0.84, and 2.1 to 2.3). Read `param-grad` gradients as trusted only for
the constant and window checked, and for windows of up to about an hour. `--params FILE` names another parameter
table and reruns the window's forward with the constants it moves; `param-grad` does not take `--strict-from`.

## Options the four commands share

> **In WOOF 1.1.7** recording a plan from a start of your own (`--init FILE`, `--init example`) is not supported: the derivative commands run on the example rows' own plans (`--row`).

A card reading on this page names the command it came from; an option below that is not in that command was not
part of the reading, and this page claims no reading for it.

| option | what it does |
|---|---|
| `--row NAME` | the operator row (default: the table's default row, `hrrrv5-x1.40962`) |
| `--steps N` | window length in model steps (default: the row's, 3) |
| `--out DIR` | output folder (required) |
| `--card N` | which card; counted inside any `CUDA_VISIBLE_DEVICES` a scheduler set (default 0) |
| `--wait-card SECONDS` | wait for the row's free card memory instead of refusing at once |
| `--need-mib N`, `--need-host-mib N` | override the row's card or host memory need, at your own risk |
| `--seed N` | test direction of the whole-state dot (default 1) |
| `--strict-from RECEIPT` | `tangent` and `sensitivity` only: take the strict check from a passing `adjoint-test` receipt of the same plan, row, runner, steps, kernel set and refuse list |
| `--no-self-dot` | skip the strict check; the receipt's `valid` then reads null, never true |
| `--numerics fast` or `strict` | the kernels' numerics (default `fast`); the readings on this page are with the default |
| `--plan DIR`, `--init FILE` | replay another recorded plan, or record one from a start file on this card first (see "Other meshes and other card families") |
| `--mesh`, `--suite`, `--oro`, `--dt` | make a row for another mesh from the suite template |
| `--runner`, `--assets`, `--grid` | name another runner, asset root or grid file |
| `--cards A,B,...` | `adjoint-test` and `sensitivity` on a row with a rank split: the window runs as one rank on each card, from rank plans named with `--rank-plans PREFIX` (this release downloads none). Experimental: a split gradient is not yet shown equal to the one-card gradient, and the command says so |
| `--gradient-mesh native`, `auto` or `6km` | `sensitivity` on the 3 km storm box: the gradient on the mesh itself (`native`; `auto`, the default, is native today) or on its 6 km companion, carried back (see "A 3 km gradient taken on the 6 km mesh") |

The measure commands (`adjoint-test`, `sensitivity`, `param-grad`) take `--metric` (`theta`, `theta_m`, `qv`, `qc`,
`qr`, `pressure_p`, `rho`, `w`; default `theta_m`; `qr`, `qc` and `w` measure the storm itself, and no PASS of this
release covers them: [LIMITS.md](LIMITS.md), *Storm measures*), a region (`--box S,N,W,E` or `--circle LAT,LON,RADIUS_KM`) and
`--levels`. Without a region, `adjoint-test` measures a circle around its bump and the other two the whole mesh.
`adjoint-test` also takes `--taylor` (the bump sizes) and `--directions` (its test directions, section 1); `tangent`
takes `--fields`; `sensitivity` takes `--map-level` and `--no-map`; `param-grad` takes `--params` and `--check`.

## A 3 km gradient taken on the 6 km mesh (`--gradient-mesh`)

`sensitivity` can take the gradient of a regional window on a coarser cut of the same region and carry it back to the
mesh you asked about: `--gradient-mesh 6km` on the 3 km storm box (`q3.96.38278.n40.60w91.30`) runs the same command
on the 6 km mesh (10,014 cells against 38,278) and moves the result to the 3 km cells. Its files (the 6 km mesh, its
recorded plan and the map between the two meshes, about 1.2 GB) are one more pinned download, fetched only when the
route is asked for. `--gradient-mesh native` always takes it on the mesh itself, and the default, `auto`, is native
until the coarse route has been shown on more targets. The receipt and `sensitivity.nc` say which mesh the gradient
was made on, which mesh it was carried to, how, and that the answer check (the whole-state dot test) was made on the
6 km operator, not on the 3 km one. Only cell fields are carried (temperature, moisture and the other cell-centred
stored fields); the edge winds are not carried and are listed as left out. `tangent`, `adjoint-test` and `param-grad`
do not take a coarse mesh and say why.

**Where this was shown, and where it was not.** One storm: the HRRR start of 2024-05-21 15Z over eastern Iowa. One
target: the mean modified potential temperature (`--metric theta_m`) over the 150 km circle at 40.6N 91.3W, every
level. One window: 885 s (59 steps of 15 s on 3 km; 30 steps of 30 s on 6 km). One card class (RTX PRO 6000
Blackwell, 96 GB). On that case the 6 km gradient took 98 s and 95 s against 502 s and 499 s for the 3 km one on the
same card (an earlier build's times; on 0.6.0's final wheel the 3 km gradient takes 180 to 232 s, and the 6 km one
was not timed again); it points the same way (cosine above 0.99 for temperature and for moisture, sizes within 2 per cent); and
the real 3 km cost falls 0.995 as much along the carried 6 km direction as along the 3 km one. Not shown: storm-scale
targets (rain water, low-level wind, vertical velocity, a small circle), any other day, any other region, 12 km,
longer windows, the edge-wind fields. The command refuses a target outside the shown list by name.

## Other meshes and other card families

> **In WOOF 1.1.7** recording a plan from a start of your own (`--init FILE`, `--init example`) is not supported: the derivative commands run on the example rows' own plans (`--row`).

The rows in `data/ad-doors.v1.json` are a seed, not a whitelist. `--mesh NAME|GRID|DIR` makes the row for that mesh
from the suite template (`--suite`, default hrrrv5): cells, edges and levels from the mesh files, the timestep from
hex's rule (6 s per km of `nominalMinDc`, dividing the radiation step; `--dt` overrides, a given `--plan` wins), and
the memory need from the suite rule (`--need-mib` overrides). The made row is kept in the row cache
(`WOOF_HEX_AD_ROWS`, default `<hex cache>/ad/rows`) and `--row` finds it afterwards. Only the three examples' plans can be downloaded (120 km, the 3 km storm box and its 12 km cut), so another mesh needs a plan of its own: `--init FILE` records one on this card first, with the recorder
the `ad` extra installs (`WOOF_HEX_AD_RECORDER` names another recorder of the same contract). This route has not been
run from a plain install, and a plan recorded with `--init` has not yet passed the strict check.

A plan replays only on the card family it was recorded on (its forward kernel images are that card's code), so on a
new family record with `--init`; the command refuses a foreign plan by name. The derivative images must be built for
the card present too. The rows ship images for `sm_120` only, and the wheel ships no image builder. When the row's
images are not built for this card, the command runs the image builder named in `WOOF_HEX_AD_IMAGE_BUILDER` once and
caches the result; with none named it refuses before it downloads anything. No card family but compute capability
12.0 has run an adjoint command.

## Limits

> **In WOOF 1.1.7** recording a plan from a start of your own (`--init FILE`, `--init example`) is not supported: the derivative commands run on the example rows' own plans (`--row`).

- **Consistency is not accuracy.** See "How far the gradient can be trusted".
- **Reflectivity.** A composite-reflectivity measure does not pass the gradient check on any setup tried, because the
  forecast itself jumps there: under a nudge of one part in a thousand, ice nucleation in one cell changes a
  hundredfold. The commands refuse it by name. Use temperature and moisture measures.
- **One card, one family.** Nothing is claimed for a gradient split across several cards (the split is experimental, and says so when used), or for any card but compute
  capability 12.0 with the free memory above.
- **The shipped examples are the 12 km cut (15 steps), the 3 km storm box (59 steps) and three steps at 120 km.** The consistency check has also passed on longer windows of the engine (15 steps, 3 hours,
  at 120 km; 8 steps, 48 minutes, at 60 km); those are measurements of the engine, not commands of this release.
- **Plans you record** with `--init` have not yet passed the strict check; the receipt runs it and says so.
- **What is held** is named in each receipt (`row_what` and the refused kinds) and in the table above.

## The pieces

| piece | where | what |
|---|---|---|
| window runner `ad-binder` | `tree/rust/hexad/` (the packaged crate the wheel builds); upstream `tree/ad/phys/binder/tools/binder/phys/runner/`, sources in `tree/ad/phys/binder/runtime/src/binder/phys/` and `tree/ad/e2e/src/` | replay, tangent sweep, reverse sweep, dot, zero and Taylor checks, measure gradients. [RUNNER.md](RUNNER.md) |
| `libhexad-phys` | `tree/ad/phys/binder/tools/binder/phys/hexad/`; shipped as `woof/hex/libexec/libhexad_phys.so` with `woof/hex/include/hexad_phys.h` | the same runner as a C library: open a recorded window, tangent and adjoint one step per call |
| kernel set kset-int7f | `tree/ad/kset/int7f/` | 150 derivative images, their specs and derived stores, rebuildable from this tree with pinned toolchains. [KERNEL-SETS.md](KERNEL-SETS.md) |
| held manifest | `tree/ad/phys/window/schemes18.json` | the class of every kernel of the hrrrv5 window (held, permutation, constant, inactive, rule, live glue) |
| derivative generators | `tree/ad/phys/{tempo,mynn,p4,p7,typesub,p6}` | build-time source-to-source tangent and adjoint of the physics CUDA, each with a proof harness. [GENERATORS.md](GENERATORS.md) |
| scheme bundle builder | `tree/ad/phys/bind/` | assembles the generated physics images into a scheme bundle |
| parameter carriers | `tree/ad/phys/window/rung3/gen_params.py`, `gen_params_p3.py` | gradients with respect to physics constants |

What a gradient is checked against, and where the operator is deliberately not the exact derivative, are in
[CHECKS.md](CHECKS.md) and [FREEZES.md](FREEZES.md). Building the runner and the kernel set: [RUNNER.md](RUNNER.md)
and [KERNEL-SETS.md](KERNEL-SETS.md).

## Licences

Project code is Apache-2.0. The derivative kernels transcribe the forward kernels they differentiate and keep those
upstream terms (MPAS-Atmosphere BSD-3-Clause, AER RRTMG BSD-3-Clause, NCAR TEMPO Apache-2.0, and the NOAA GSL physics
under NOTICE section 9). NOTICE section 18 lists them.
