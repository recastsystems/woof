# 1. What woof hex is

## For someone who has never run an atmospheric model

An atmospheric model is a program that takes a snapshot of the atmosphere
(temperature, wind, moisture, pressure, everywhere at once) and advances it
forward in time using the equations of fluid motion. Feed it this morning's
atmosphere and it computes this evening's. The snapshot it starts from is
called the **initial conditions**; the ground it stands on (terrain height,
land use, soil type) is called the **static** data; and the arrangement of
points where the atmosphere is represented is called the **mesh**.

woof hex's mesh is not a flat rectangle over one region. It is a sphere tiled
with hexagonal cells (the whole planet at once) and the cells do not all
have to be the same size. You can make them 100 km wide over the oceans and
25 km wide over the region you care about, in one seamless mesh with no nest
boundaries. That is what "global variable-resolution" means.

The unusual part is where it runs: **one consumer graphics card**. The entire
model state lives in GPU memory and every step of the integration is CUDA.
There is no cluster, no MPI job, no supercomputer allocation. A 12 GiB gaming
card integrates the published 40,962-cell global mesh; a 32 GiB card
integrates the 163,842-cell mesh whose fine band is about 25 km; a 10 GiB
card runs a six-hour full-physics limited-area forecast cut out of a
global case.

## What it is, precisely

- **The dynamics** (the fluid-motion core) is a port of MPAS-Atmosphere
  v8.4.2, byte-identical to native MPAS built with the pinned strict recipe on
  the measured cases (dry, and with water vapour and five condensates), which
  `woof hex verify-dycore` re-checks to 0 differing words (see
  [compatibility](../mpas-compatibility.md) for the scope), and
  deterministic: two runs of the same case produce
  byte-for-byte identical output.
- **The physics** (radiation, clouds, rain, turbulence, land surface) comes in
  two kinds. By default (since 0.4.6) it is one of MPAS-A v8.4.2's own suites,
  picked by mesh (`convection_permitting`; `mesoscale_reference` by name once
  it runs on the GPU), ported scheme by scheme
  and checked bit for bit against native MPAS; a 24-hour forecast with every
  scheme of `convection_permitting` on the GPU matches native MPAS in every
  field hex writes. On the MPAS suites hex runs **correct mode** by default,
  which fixes five defects found in native MPAS; `--physics-mode exact`
  reproduces native bit for bit.
- **The tooling** is a set of doors: `doctor` reports what your install can
  reach, `mesh-check` validates a mesh, `cull` cuts a limited-area case out
  of a global one, `init` builds initial conditions, `forecast` runs the
  model, `swath` decides where a fine grid goes, `cycle` follows weather
  across cycles, `render` turns model output into product images. Each door
  refuses bad input by name rather than producing a plausible wrong answer.

## What it can do

Measured, not promised (the README states each measurement):

- 24-hour global forecasts on one card, deterministic across dual runs.
- Full-physics limited-area forecasts cut out of a global case with the
  `cull` door and driven at their boundary: six hours, 1,080 of 1,080
  steps on an 11,020-cell cut, a 6,224 MiB footprint on a 10 GiB card.
- Weather followed across forecast cycles: `swath` places fine grids from
  a coarse run's own fields, `cycle` cuts, forces and integrates them,
  two real cycles ran back to back, both fine forecasts completing
  1,080 of 1,080 steps.
- Initial conditions built from a WPS intermediate file (GFS, ERA5, or
  whatever you drive with) without running any native Fortran tool; a
  native-built "capsule" is an explicit compatibility mode, not a
  requirement (chapter 5).
- Product PNGs rendered from history files entirely through the Rust
  renderer, filed by domain, product, and valid day (chapter 7).
- New meshes at resolutions you choose, with their static fields, generated
  by the engine's `woof mesh` door with no native toolchain (chapter 4).
- Checkpoint/restart with bitwise-identical continuation, proven in the
  full-physics proof harness (chapter 6).

## What it cannot do, stated up front

- **Each identity claim has a scope.** The dycore identity is measured with
  physics off on the listed meshes; the all-GPU MPAS physics match is one
  24-hour forecast on the 480 km mesh, one process; every other MPAS scheme is
  bitwise on captured inputs. Correct mode differs from native wherever a fix
  acts, by design. Or it is the WOOF-ArWen engine's suite (`--physics-suite
  arwen_mpas_column`), reached through a pinned seam, so woof hex and the
  `woof` distribution share one physics implementation.
- **WOOF-ArWen's physics is not MPAS's.** On the WOOF-ArWen route a full-physics
  forecast is not identical to native MPAS, and the dycore check does not
  qualify one. Beyond the expected chaos-shaped divergence there are **three
  measured, one-signed differences**, the declared divergences of chapter 3.
  Read them before trusting a number from that route.
- **No multi-node.** Two-node execution is built and proven bitwise
  partition-invariant, but it is not part of this release. Several cards on
  one host is a door (chapter 12) whose qualification is not complete.
- **Forecasting runs from the wheel.** Since 0.3.2 `woof hex forecast`
  needs no checkout of either project: its drivers ship in the package and
  the physics seam is the installed `woof`, named in every receipt by
  version and `RECORD` digest (chapter 6.1). `cycle run` still needs a
  woof hex checkout for its per-cycle contract deck (chapter 6.9).
- **No data fetching.** woof hex ships no meshes, no static files, no
  meteorological data. `verify-dycore` downloads only its small public test
  mesh and builds its native reference. Chapter 2 states
  plainly where each asset comes from.
- **Not long-range on the WOOF-ArWen route.** The measured upper-level warm drift
  (chapter 3) disqualifies that route for multi-day stratospheric work; it is
  bounded and fine at 24 h.
- **A run can stop early.** When a step fails validation (for example a
  vertical-velocity spike past 200 m/s) the model refuses to publish the
  step and stops rather than writing output it does not trust. A refusal is
  designed behavior, not a crash (chapter 3).

## The differentiable engine: what it can and cannot do

woof hex 0.6.2 carries the tangent-linear and adjoint model of its forecast
step, reached through four commands (`tangent`, `adjoint-test`,
`sensitivity`, `param-grad`; [the differentiable engine](../differentiable/README.md)).
They are of the exact-mode forecast (`--physics-mode exact`), not of the
correct-mode forecast a plain `woof hex forecast` runs, and they run on Linux
on one NVIDIA card of compute capability 12.0 (GeForce RTX 50 series, RTX PRO
Blackwell). The dynamics are differentiated whole. Of the physics, the cloud
microphysics (TEMPO), the land surface (RUC) and the boundary layer are
differentiated, but the boundary layer's derivative is of MYNN-EDMF c6d08709
(the code NOAA's RRFS 44c0849 runs), not of the v6.1.1 a plain hrrrv5
forecast runs, and each example row holds named parts of it at the forecast's
values; on the 12 km example that is why the gradient falls about 2 to 4 per
cent short for moisture near the ground. The cumulus scheme (New Tiedtke) and
the mountain-drag closure are held, radiation is held on the default 120 km
row (it is in on that row's full-physics twin), and the lateral-boundary
driving data have a zero tangent. Every receipt names what its row holds
(`row_what`), and [the limits](../differentiable/LIMITS.md) list each measured
limit. The commands' own checks show that the adjoint is the transpose of the
tangent-linear model within 4 single-precision roundings, and compare the
gradient's prediction with the real model's change against a 1 per cent bar.
How far a gradient predicts the forecast was measured separately: about 3 hours
at 120 km, and on the 3 km storm box 5 to 15 minutes for temperature
(depending on the perturbation) and at most 10 minutes for moisture; six hours
is claimed on no mesh. The largest
example is the 120 km global one (40,962 cells, 3 steps, a card with 73,156 MiB
free); the 3 km storm box (38,278 cells, 59 steps) needs a 96 GB card and
about 250 GiB of host memory; the 12 km cut (2,890 cells, 15 steps) runs on a
32 GB card. A gradient runs on one card: splitting it over several
(`--cards`) is experimental and has not been shown equal to the one-card
gradient, and recordings as several rank threads on one card (`record --ranks
N`) are off by default and claimed for nothing. The learned-correction slot
(`WOOF_HEX_LEARNED_CORRECTION=checkpoint.pt`, with the `[torch]` extra) adds
a network's four tendencies (potential temperature, water vapour and the two
winds) to a forecast at every step, on one card. No weights ship. A network
whose weights are all zero gives the plain forecast word for word (the 120 km
global over 24 hours, 2,523,239,725 words). The slot's gradient has been
checked against finite differences only on the package's stand-in engine
(`python -m woof.hex.torch_bridge.proofs`), and training a network through the
real engine's adjoint is not a command of this release, so this release makes
no claim about what a trained correction does to a forecast.

## What hardware it needs

| | |
| --- | --- |
| GPU | Any CUDA device for the doors. For forecasts, memory is set by mesh size *and* by the card, and the two do not combine into a line: the footprint is a per-card core, plus two physics column workspaces charged at `min(cells, tile)`, plus a per-cell term (chapter 4). Measured at the merged tip on an RTX 5090, 2026-08-26: the 40,962-cell global mesh peaked at **8,874 MiB** (fits a 12 GiB card); the 163,842-cell mesh peaked at **20,446 MiB** and admits at the measured floor (the prediction plus that card's own margin, about 21.7 GiB free at x4, computed by `woof.hex.device_admission`) so it remains in practice a 32 GiB-card configuration. The core is a property of the card, and the door reads the card at the moment of the decision. |
| CPU/RAM | Modest. The doors are I/O-bound; the model state lives on the device. |
| Disk | Meshes, initial conditions, and history files are hundreds of MB to a few GB each. The optional byte-pinned test assets are about 6.9 GiB. |
| OS | Linux or Windows. Python 3.11 or newer. |
| CUDA | A driver whose CUDA major you know: the install chooses a matching CuPy wheel (chapter 2). |

Chapter 4 turns the footprint model into a sizing method for any card.
