# The first hour with the differentiable engine

This page is for someone who has just installed woof hex 0.6.1 and wants to know, within an hour, whether the
tangent-linear and adjoint run on their machine and what they show. The commands of sections 2 to 4 were run on a
pre-release build of 0.6.1, in a fresh Python 3.14 environment with an empty home directory, on one RTX 5090:
section 2's install line with that build's wheel file in place of the published package name, every other
command as printed. The output in section 3 is that run's. Section 5's commands were run on builds of 0.6.0 (one
96 GB RTX PRO 6000 Blackwell), and on that build of 0.6.1 on a 32 GB card, where they stop as section 5 says. The
12 km figures name the wheel they were measured on; the 120 km and 3 km figures are from builds of 0.6.0. What the
engine is and is not is in [GUIDE.md](GUIDE.md); every current limit is in [LIMITS.md](LIMITS.md).

## 1. Will it run on my machine?

| | forward forecasts (`woof hex forecast`, `global`, `regional`, ...) | the four adjoint commands (`tangent`, `adjoint-test`, `sensitivity`, `param-grad`) |
|---|---|---|
| NVIDIA Blackwell, compute capability 12.0 (GeForce RTX 50 series, RTX PRO Blackwell) | run; the strict-numerics proofs were measured on this family | run. The 12 km example has run on a 32 GB RTX 5090; the 3 km and 120 km examples on a 96 GB RTX PRO 6000 Blackwell |
| NVIDIA Ada (compute capability 8.9: RTX 40 series, L4, L40S, RTX 6000 Ada) and Ampere 8.6 (RTX 30 series, RTX A6000) | run; this release's numerical contract was measured again on 8.6 and 8.9 (README, *Requirements*) | stop before any download and say why. Run on a real RTX 4090: the command names the cards that run the row and exits 2 |
| NVIDIA A100 (8.0), H100 and H200 (9.0), Turing (7.5) | run; the receipt says the card is not anchored | stop before any download and say why (checked with a stand-in for an H100, not on a real one) |
| AMD | experimental, opt-in (`--backend amd`) | not available |

Why only one family runs the adjoint: the derivative images and the example recordings are compiled for compute
capability 12.0. A recording replays only on the card family it was recorded on, and part of the kernel set carries
code for that family alone. Building them for other families is planned work; nothing in this release has run an
adjoint command on another family.

What else to bring:

| example | card memory free | host memory | download (once) | first command |
|---|---|---|---|---|
| 12 km cut of the storm box (2,890 cells, 15 steps of 60 s) | 28,700 MiB | 47,104 MiB (the README command peaked at 41,190 to 44,109 MiB on 0.6.0's release builds and at 40,990 MiB on a pre-release build of 0.6.1; the 8-direction check at 46,681 MiB on a 0.6.0 build) | 1.20 GB (3.4 GB unpacked in the cache) | 320 to 645 s on 0.6.0's wheels, 418 to 708 s on builds of 0.6.1, 769 s on 0.6.2's with the host shared (5 to 13 minutes) |
| 120 km global (40,962 cells, 3 steps of 720 s) | 73,156 MiB (80,000 MiB for the full-physics row) | 81,920 MiB (110,592 MiB for the full-physics row) | about 2.4 GB | 271.9 to 307 s for its 1-step `adjoint-test` (about 4.5 to 5 minutes) |
| 3 km storm box (38,278 cells, 59 steps of 15 s) | 96,600 MiB | 258,057 MiB for a first `sensitivity` (it runs the answer check), 249,661 MiB later. `adjoint-test` on this row keeps every step boundary, so its need follows the host: at least 102,948 MiB available (on a smaller host than the runner's default suits, more of its store goes to the run folder), and it held 254,885 to 262,306 MiB on 512 GiB hosts, and 420,360 MiB on the run that read PASS at the row's five step sizes | about 2.7 GB | about 12 minutes for `sensitivity` |

Each command checks, before it downloads the example, first what only you can change (the card's family and
whole size, Python, the `ad` extra, a bump or region off the mesh) and then what is free on the machine now (card
memory, host memory, the run folder's disk), and stops with one sentence saying what is missing. Exit codes: 0
PASS, 1 ran and did not pass (FAIL, or NOT PASSED when fewer than two directions resolved), 2 refused, 3 not
enough host memory or disk.

## 2. Install

Linux x86_64, Python 3.14, and the `ad` extra (it pins the CUDA libraries the example recordings replay byte for
byte):

```bash
python3.14 -m venv hexad && . hexad/bin/activate   # Debian and Ubuntu: sudo apt install python3-venv first
pip install "recast-woof[ad]"
woof hex version
```

Install the `ad` extra in a Python 3.14 environment of its own, not on top of `[gpu]`: each of the two extras brings its own CuPy package, one environment holds one, and pip does not remove the other. `[ad]` installs the CUDA 13 CuPy, so it needs a driver that serves CUDA 13; the forecast commands of this release were run from a `[gpu]` environment. `woof hex doctor` reports two CuPy packages as a required gap and prints the command that leaves one.

Ubuntu 26.04's system Python is 3.14, so the first line runs there as printed. Ubuntu 24.04's is 3.12 and it has
no `python3.14`, so there the line stops with `python3.14: command not found`: use Ubuntu 26.04, or install a
Python 3.14 first and name it in place of `python3.14`. (The `[gpu]` forecast install runs on 24.04's `python3`.)

The package itself installs into Python 3.11 or newer (the forecast runs there), so `pip` does not stop you in
Python 3.12 or 3.13; the adjoint commands then refuse in one sentence, before any download, and say to make a
Python 3.14 environment.

`woof hex doctor` checks the whole forecast install. On a fresh install it exits 1 and names one missing item,
`WPS_GEOG`, the geography archive the forecast's static builder needs. The adjoint commands on this page do not
need it: their examples download everything they replay. The forecast commands do, `woof hex global` on its
published meshes included, because their static is built on your machine. Stage it once with `woof fetch-geog`
(16 datasets, about 2.5 GiB to download and about 31.3 GiB unpacked with gpuwm 3.0.3; `--explain` prints the plan
first); after it, doctor reported every check passed on the `[gpu]` install of the same build.

## 3. The first command: the 12 km gradient test (a 32 GB card)

```bash
woof hex adjoint-test --row hrrrv5-q12.96.2890.n40.60w91.30 --steps 15 --at 40.6,-91.3 --radius-km 150 --out cut12
```

It downloads the 12 km example once (1.20 GB), replays a recorded 15-minute window of an hrrrv5 forecast from the
HRRR 2024-05-21 15Z start over eastern Iowa, and tests the linear model a plain `sensitivity` on this row runs. What
it printed on one RTX 5090 (32 GB), from a fresh install with an empty home:

```text
adjoint-test: hrrrv5-q12.96.2890.n40.60w91.30, 15 steps (15 min)
  linear model: the boundary layer's tape-free route (this row's default: the one a plain sensitivity runs)
  dot     1.128e-08   bar 2e-05   PASS
  fd      9.906e-04   bar 1e-02   PASS
  taylor  9.906e-04   bar 1e-02   PASS
  zero    0 nonzero words   PASS
  ...
  directions: the real model's change over the gradient's prediction, per step size h (bar: within 0.01 of 1; ...)
    bump                  ...   INSIDE (9.91e-04 from 1)
    descent               ...   INSIDE (5.00e-04 from 1)
    near-ground-moisture  ...   UNRESOLVED (7.21e-03 from 1): ...; it does not count
    smooth-draw           ...   UNRESOLVED (1.47e-02 from 1): ...; it does not count
  known limits of this linear model (measured before this release, not by this run; ...):
    near-ground-moisture  [this run: UNRESOLVED] about 2 to 4 per cent short: ...
    window                the window's consistency check ... read +0.63, +0.33, -0.87, -0.25, +0.96, -0.34, +0.11 and +2.26 roundings on seeds 1 to 8 ..., each within the bar, with no one-signed drift ...
  whole-state dot  7.494e-08 relative = +0.63 roundings   bar 4 roundings (4.77e-07)   PASS   (...)
    the bar: the two sides of the dot must agree within 4 single-precision roundings of the dot (...)
PASS: 2 directions resolved, each inside the bar; not covered by this PASS: near-ground-moisture (a known limit, named above)
```

How to read it, line by line:

- **`linear model`**: which linear model was tested. On this row the default is the `tape-free` route; it is also
  what a plain `sensitivity` runs. `--boundary-layer taped` tests the exact route of the held boundary layer
  instead, and that ends FAIL (its descent and near-ground moisture miss the bar; the known-limits block names them).
- **`dot`, `zero`, `whole-state dot`**: consistency. They show the adjoint is the transpose of the tangent-linear
  model on this window. They say nothing about how well the gradient predicts the forecast. On this example the
  whole-state dot is scored on the window's memory allocations (134 of them), not on the model's fields, and the
  command prints why: the example's plan holds 15 step boundaries, so the 15-step window end has no field map
  ([LIMITS.md](LIMITS.md)). The 3 km example scores its 12 model fields.
- **`fd`, `taylor`**: the tangent and the adjoint against the real forecast rerun with the bump added, for the bump
  only.
- **`directions`**: for each test direction, the real model's change of the measure divided by the gradient's
  prediction, at five step sizes. 1 is a perfect gradient; the bar is 0.01. INSIDE and OUTSIDE need step sizes at
  which the forecast's own change is steady; where it is not, the direction is UNRESOLVED and does not count either
  way.
- **`known limits`**: what was measured about this linear model before the release, along the directions this
  command draws. It is printed so that a documented limit is not mistaken for a broken install. It never changes
  the verdict: a known limit is not a pass, and a direction read outside the bar still fails the command.
- **The last line**: PASS needs every check inside its bar, no direction outside, and at least two independent
  directions resolved. This PASS is of the bump and the gradient's steepest descent. It does not cover near-ground
  moisture, where the gradient is about 2 to 4 per cent short: the boundary layer's plume speed and condensation are
  held in this linear model.

Measured on a pre-release build of 0.6.1, one RTX 5090: 418 s on a first run (about 7 minutes, including
fetching the 1.20 GB from GitHub and unpacking it), card peak 24,766 MiB, host peak 40,990 MiB. On 0.6.0's published
wheel the same run took 458 s (host peak 41,190 MiB), and 320 to 645 s over the runs of 0.6.0's release builds. The receipt (`cut12/receipt.json`) holds every
number, each window's file, the runner's and the recording's SHA-256 and the peaks.

## 4. A gradient on the same window

```bash
woof hex sensitivity --row hrrrv5-q12.96.2890.n40.60w91.30 --steps 15 --circle 40.6,-91.3,150 --levels 1-10 --out cut12s
```

The gradient of the mean modified potential temperature over model levels 1 to 10 in a 150 km circle, at the
window's end, with respect to the start state. It writes `cut12s/sensitivity.nc` (`dJ_dtheta` in K per K,
`dJ_dqv` in K per kg/kg, the gradient to every stored start field, and each cell centre's latitude and longitude,
`latCell` and `lonCell`, in degrees) and a map. It takes the answer check from the
`adjoint-test` just run instead of running it again, and says so in one line. Measured: 108.8 s on the
pre-release build of 0.6.1 above (card peak 27,048 MiB, host peak 37,116 MiB), and 55 to 119 s over the runs of
0.6.0's release builds (card peak 27,048 MiB). The map shows a positive disc of `dJ_dtheta` over the 150 km circle at 40.6N 91.3W,
strongest on its south-west edge, and nothing outside it, as a 15-minute window should.

## 5. On a 96 GB card

The 3 km storm box is the size of mesh the engine is built for:

```bash
woof hex sensitivity --row hrrrv5-q3.96.38278.n40.60w91.30 --steps 59 --circle 40.6,-91.3,150 --levels 1-10 --out storm
```

and the 120 km global example is the quickest check that a large-card install works:

```bash
woof hex adjoint-test --out at
woof hex tangent      --steps 3 --out tl
woof hex sensitivity  --steps 3 --metric theta_m --box 30,48,-105,-80 --levels 1-10 --out se
woof hex param-grad   --steps 3 --check --out pg
```

On a 32 GB card each of these stops at once, before any download, in one sentence; the 3 km command's reads "card 0
has 32607 MiB in all; this row's window needs 96600 MiB, more than the whole card, so it cannot run on this card"
(the 120 km rows name their own need, 73,156 or 80,000 MiB). What they need and cost on a 96 GB card
is in the README (*The differentiable engine*). `adjoint-test` on the 3 km example takes about 45 minutes and ends
PASS on its default route today (the bump and the descent inside) and FAIL on the
exact route ([LIMITS.md](LIMITS.md), *3 km*). The wheel carries both sets as scripts, with the numbers to expect: `woof hex examples` prints their folder and
`woof hex examples --copy DIR` copies them ([examples/differentiable/](../../examples/differentiable/README.md)
in the source archive and the repository).

## 6. Where to go next

- What the tangent and adjoint are, what they hold, and how to take a gradient of your own measure:
  [GUIDE.md](GUIDE.md).
- Run the consistency and gradient checks on your own directions: [GUIDE.md](GUIDE.md), *Checking it yourself*.
- Everything that is not yet right, in one place: [LIMITS.md](LIMITS.md).
