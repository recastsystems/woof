# Using the tangent-linear and adjoint

A guide for a scientist who wants gradients from woof hex 0.6.1: what the tangent and adjoint are, what they hold,
how to take a gradient of your own measure, how to check it yourself, what a recording costs, how to drive the
linear model from a variational system, exact against fast mode, and what "bit for bit with NOAA's RRFS-MPAS" means.
Start with [FIRST-HOUR.md](FIRST-HOUR.md) if nothing has run yet. Every limit is in [LIMITS.md](LIMITS.md).

## 1. What they are, and what they are not

A hex forecast is recorded as a **plan**: every kernel launch of the forward, in order, with the state at each step
boundary. The **tangent-linear model** replays that plan and carries a perturbation of the start state forward,
launch by launch, through a derivative rule or a generated derivative kernel paired with each forward kernel. The
**adjoint** runs the same launches backward and carries the gradient of a measure at the window's end back to the
start state. Both are of the forecast the engine actually ran, not of a separate simplified model.

They are linearisations around one recorded forecast, over one window. What they are not:

- **Not the derivative of everything.** Some pieces of the forecast are held at the recorded values (their
  derivative is taken as zero). Which pieces depends on the row; the table below and each receipt's `row_what`
  name them.
- **Of exact mode's boundary layer, not the default forecast's.** The boundary layer's derivative is generated
  from MYNN-EDMF at commit c6d08709 (the code RRFS-MPAS 44c0849 runs, and what `--physics-mode exact` runs) and
  the surface layer's from the MYNN surface layer v6.0.0. A plain `hrrrv5` forecast of this release runs correct
  mode, with MYNN-EDMF v6.1.1 and the surface layer v6.1.0; a gradient of that forecast is refused before anything
  is downloaded, and the derivative of v6.1.1 is planned for the next build. Where the boundary layer mixes cloud
  (stratocumulus, low stratus under an inversion) a sensitivity is that of the older boundary layer. The bit-for-bit
  forward results (section 7) are of the forward model, not of the linear model.
- **Not evidence of accuracy by itself.** The dot tests show that the adjoint is the transpose of the tangent. How
  well the gradient predicts the real forecast is a separate measurement (section 3 and the README's *How far in
  time the gradient can be trusted*).
- **Not for a composite-reflectivity measure.** The forecast itself jumps there (ice nucleation in one cell changes a
  hundredfold under a nudge of one part in a thousand); the commands refuse it by name. Use temperature, moisture,
  pressure, density or vertical-wind measures.

### Scope, by row

| row | mesh and window | what is differentiated | what is held |
|---|---|---|---|
| `hrrrv5-q12.96.2890.n40.60w91.30` (the 12 km example) | 12 km cut of the storm box, 2,890 cells, 59 levels, 1 to 15 steps of 60 s, recorded from the HRRR 2024-05-21 15Z start | dynamics, the transport, cloud microphysics (TEMPO), and on the default route the boundary layer's turbulence closure, the plume's transport of heat, moisture and momentum, and the boundary-layer height the column returns, under a weak-shear rule | the mixing length; the plume's own speed, area and condensation; the transport limiter's coefficients; on this row only, the boundary layer's seven species outputs (its tendencies of cloud water, cloud ice, snow and four numbers) and, in a plan recorded with it, the cumulus scheme; number concentrations of ice, rain, cloud droplets and graupel carry no tangent (measured on this cut; [LIMITS.md](LIMITS.md)) |
| `hrrrv5-q3.96.38278.n40.60w91.30` (the 3 km storm box) | 3 km, 38,278 cells, 59 levels, 1 to 239 steps of 15 s, recorded from the HRRR 2023-03-31 18Z start | as the 12 km row, plus the transport limiter (live, its exact derivative) on windows up to 30 minutes, the example's 59 steps among them; the cloud derivative's `number` form by default (rain drop and cloud droplet number follow the forecast; `--cloud-derivative gated` holds them) | as the 12 km row, except that the transport limiter is held only on windows longer than 30 minutes (such a window runs the `gated` cloud route and says so), and without its species and cumulus holds; on the `number` route the snow riming ratio where riming is slow, the riming fall-speed boost where a trace of cloud water would carry it to its cap, and snow and cloud-ice deposition only where it would add the species' own mass within 200 s; the mountain-drag scheme has nothing to act on below about 3.6 km spacing |
| `hrrrv5-x1.40962` (the 120 km default) | global 120 km, 40,962 cells, 59 levels, 3 steps of 720 s | dynamics, land (RUC), cloud microphysics (tangent in double precision) | radiation and cloud fraction (64 kernel kinds, listed in the receipt); the transport limiter's coefficients; the boundary layer's turbulence closure, mass-flux plume and returned height; the mountain-drag closure |
| `hrrrv5-full-x1.40962` | as above | as above, plus radiation and cloud fraction, the transport limiter differentiated (not held), and the boundary layer with its moisture mixing and its plume live (the plume's speed on a ramp; the `floored` route by default) | the mountain-drag closure |

The 12 km row also takes `--boundary-layer guarded` (from 0.6.1): nothing of the boundary layer held, under the
weak-shear guard, as on the 3 km row. It is the route for a measure near the ground under rain or cloud, where the
default route's held plume and mixing length leave the gradient about 2 per cent short, and it is not the default:
the README's 12 km command ends NOT PASSED (UNRESOLVED) on it ([LIMITS.md](LIMITS.md), *Storm measures*).

Cumulus: no example of this release has a cumulus scheme in its recording, and no cumulus derivative ships; on the 12 km row a plan recorded with New Tiedtke on runs its 26 launch kinds as identity, named in `window.json`. The lateral
boundaries of a limited-area window are held at the recorded values.

## 2. A gradient of your own measure

### From the command line

`sensitivity` takes the gradient of the mean of one field over a region and a band of levels, at the window's end,
with respect to every stored field of the start state:

```bash
woof hex sensitivity --row hrrrv5-q12.96.2890.n40.60w91.30 --steps 15 \
    --metric qv --circle 40.6,-91.3,100 --levels 1-7 --out moist
```

- `--metric`: `theta_m` (default), `theta`, `qv`, `qc`, `qr`, `pressure_p`, `rho`, `w`.
  `qr`, `qc` and `w` measure the storm itself, and no PASS of this release covers them: on the 12 km example's
  storm band rain reacts to a one-rounding nudge of its start by more than the gradient's whole prediction at
  small steps, so its check cannot read 1 per cent (centred, 0.99 to 1.04 of the real change); the updraft reads
  inside along the bump and unresolved along the steepest descent, where the forecast's own switches move the
  real change 1 to 5 per cent at small steps (on a 3 km storm cell, read one-sided, 1.008 to 1.023). The map
  shows where and with what sign such a measure responds, not its size to 1 per cent. A temperature measure placed
  on a storm cell is not covered either: on a 3 km cell (a 40 km circle, levels 1-10) its check fails (0.245
  against 0.01). The PASSes are for the README's measures (low-level temperature and moisture averaged over circles of 100 to 150 km radius). For `qr`, `qc` and `w`
  the command prints the row's measured line before it runs; [LIMITS.md](LIMITS.md), 'Storm measures', has the
  readings.
- the region: `--box S,N,W,E` or `--circle LAT,LON,RADIUS_KM`; without one, the whole mesh.
- `--levels a-b` or `a,b` (1-based model levels); without it, every level.
- `--steps`: the window, 1 to 15 on the 12 km row, 1 to 239 on the 3 km row, 1 to 3 on the 120 km rows (3 by default; `adjoint-test` runs 1 step there unless `--steps` is given).

It writes `sensitivity.nc` (the gradient to each start field, with units, and the latitude and longitude of each
cell centre and edge midpoint in degrees, `latCell`, `lonCell`, `latEdge`, `lonEdge`, so the gradient is placed
without the mesh file), a map (`--map-level`, `--no-map`) and `receipt.json`. The map draws the gradient to the
measure's own start field: `dJ_dqv` for a `qv` measure, `dJ_dtheta` for `theta` and `theta_m`. For a measure of
another field it draws `dJ_dtheta` and says so; that field's own gradient is in `sensitivity.nc`. It draws the level
that carries most of the drawn field, which need not be one of the measure's levels. The command prints the field,
the level and the measure's levels in one line. Several measures in one command: `--objectives FILE`, a JSON list of objects with `metric`,
`levels`, `box` or `circle`, and `name`; each gets its own folder, answer and receipt.

### From Python

```python
from woof.hex.ad_session import SensitivitySession

with SensitivitySession("runs/opt", row="hrrrv5-q12.96.2890.n40.60w91.30", steps=15) as s:
    g = s.gradient("theta_m", circle="40.6,-91.3,150", levels="1-10")
    dJ_dtheta = g["fields"]["dJ_dtheta"]      # numpy array, [levels, cells]
    print(g["valid"], g["passed"], g["seconds"])
```

Each `gradient` call is the `sensitivity` command's own answer for that measure, in its own folder (`runs/opt/g0001`,
...), with the same answer check and the same bytes. On the 3 km storm row the session keeps one program loaded on
the card between calls, so later gradients skip the program's start and reuse the kept forward (a later gradient
took 180.5 s against 239.5 s for the first, measured once on a 96 GB card). On the 12 km row each call runs its own
program. [examples/differentiable/own_measure.py](../../examples/differentiable/own_measure.py) is a runnable
version with two measures.

### Any weighting of the end state

The runner takes an objective file with arbitrary sparse weights at the window's end,
`{"name": ..., "terms": [[allocation, word, weight], ...]}`, through `ad-binder window --objective FILE` (the runner
ships in the wheel as `woof/hex/libexec/ad-binder`; [CHECKS.md](CHECKS.md) shows a whole command and
[RUNNER.md](RUNNER.md) the passes). For a measure that is not a mean of one field (a weighted misfit to
observations, for example), drive the linear model through the library in section 5 and seed the adjoint with
your own gradient at the window's end.

## 3. Checking it yourself

`adjoint-test` runs every check a user can run, on the row and window you give it:

```bash
woof hex adjoint-test --row hrrrv5-q12.96.2890.n40.60w91.30 --steps 15 --at 40.6,-91.3 --radius-km 150 \
    --directions all --seed 2 --out check2
```

| what | how to ask for it | what passes |
|---|---|---|
| the dot test along a temperature bump | always (move the bump with `--at`, `--level`, `--amplitude`, `--radius-km`, `--level-width`) | relative difference of `<w, M x>` and `<M^T w, x>` at most 2e-5 |
| the finite-difference and Taylor checks | always, on the bump; `--taylor h1,h2,...` sets the step sizes | best confirmed step size within 1e-2 |
| the zero direction | always | 0 nonzero words |
| the whole-state dot test | always (`--seed N` draws another direction; `--dot-directions N` runs N together) | within 4 single-precision roundings of the dot (4.8e-07 relative); over 8 or more directions also no one-signed drift (`--dot-bar strict`: the earlier 1e-07) |
| the gradient against the real model along further directions | `--directions descent,near-ground-moisture,smooth-draw` or `all` | a direction INSIDE within 0.01 of 1; at least two independent directions resolved |
| the boundary-layer route tested | `--boundary-layer tape-free` (default) or `taped` | as above |
| a planted error, to watch the check fail | `--plant-error boundary-layer:1.01` (or `acoustic`, `advection`; default off) | must not pass: the whole-state dot FAILS, the verdict says PLANTED, exit status 4 |

**What the whole-state dot test is, exactly.** It moves every word of the start state at once and seeds the adjoint
with the tangent's own result at the window's end. The start direction is scaled word by word by the size of the
state itself, so a word that is zero in the start state (rain, snow or graupel in clear air; a number concentration
where it is zero) is not perturbed. The end weights are one over each field's root-mean-square, so every field
counts equally. It is a strict test of the transpose on every field the window carries; it is not a test with an
independent random weight at the end and an additive start direction, which this release does not run. One
direction proves little near the bar: run several with `--seed`, or `--dot-directions 8`.

**The answer check's bar, and why it is in roundings.** The check compares two numbers the engine computes from the same window: the tangent-linear model's `<y, M x>` and the adjoint's `<M^T y, x>`. If the adjoint is the transpose of the tangent-linear model they are equal, except for rounding. The model and its derivative run in single precision, so the two cannot agree to better than about one single-precision rounding of the dot (2^-23 of its size, 1.2e-07 relative). The bar is therefore stated in those roundings: the two sides must agree within 4 roundings (4.8e-07 relative) on every direction tested, and over 8 or more directions (`adjoint-test --dot-directions 8`, what the release check runs) their signed differences must not lean one way (the mean within 2 standard errors of zero). A missing or wrong derivative term reads tens to hundreds of roundings, or leans one way over many directions; rounding does neither. The bar before this build, 1e-07, was below one rounding, so a correct build passed or failed it by luck; `--dot-bar strict` still applies it. The check shows that the adjoint is the transpose of the tangent-linear model within 4 single-precision roundings; it is not a claim of exactness, and it says nothing about how well the gradient predicts the forecast. That is the separate gradient test against the real model (the `fd`, `taylor` and direction lines), with its own 1 per cent bar.

**Reading UNRESOLVED.** A forecast is not smooth: a branch taken in one run and not in the other moves the measure by
an amount that does not shrink with the step. Where the forecast's own change along a direction is not steady over
three neighbouring step sizes, the command cannot tell the gradient's error from that roughness and prints
UNRESOLVED. It is not a pass. The known-limits block printed under the directions says what was measured along
that direction with more step sizes.

**The last line and the exit status.** PASS (exit status 0) needs every check of the first window and the
whole-state dot inside its bar, every direction run, no direction OUTSIDE, and at least two independent directions
INSIDE. It is a statement about the directions printed INSIDE and no others. FAIL (exit status 1) names the check
or the direction that missed. NOT PASSED (UNRESOLVED) (exit status 1) means nothing failed and fewer than two
directions resolved. A refusal before the run is exit status 2, or 3 for host memory or disk; a run with
`--plant-error` ends 4 or 5 (below). Today the 12 km
example ends PASS on the bump and the descent, and the 3 km example ends PASS on its default route (the bump and the descent
inside) and FAIL on the exact route
([LIMITS.md](LIMITS.md) has each direction's reading).

**Break it on purpose.** A check that only ever passes shows nothing, so `adjoint-test --plant-error KERNEL:SCALE`
breaks the derivative on purpose: in the backward (adjoint) code of one kernel only (`boundary-layer`, `acoustic` or
`advection`), every change it makes is scaled by SCALE, so 1.01 is a 1 per cent error; the forward model and the
tangent are untouched. The command says so in one line before it runs; its verdict line says PLANTED, never PASS or
FAIL; its receipt is never valid and its answer check is never kept for later commands; it exits 4 when the check
caught the error and 5 when it did not; and `sensitivity` refuses the option. On the 12 km example (one RTX 5090,
16 Nov 2026), the plain command read the whole-state dot at +0.63 roundings, PASS. With
`--plant-error boundary-layer:1.01` it read 7.964e-06, 66.8 roundings against the bar of 4, exit status 4 (on a
pre-release build of 0.6.1, one RTX 5090: 638.8 s, card peak 24,766 MiB, host peak 46,486 MiB, inside the 12 km row's
47,104 MiB). With
`boundary-layer:1.001` (0.1 per cent) it read 8.545e-07, 7.2 roundings: caught, but by less than twice the bar, so an
error much smaller than 0.1 per cent in this code would hide under rounding. `acoustic:1.01` read 271.5 roundings,
exit status 4. `advection` has not been run on a card for this release; on a row where the kernel it names does not
run, the command says NOTHING WAS PLANTED and exits 5. In the boundary-layer planted runs the bump's own dot
(bar 2e-5) and the bump and descent against the real model still passed: a wrong adjoint leaves the tangent's checks
untouched, and the whole-state dot is the check that sees it.

**How far the gradient can be trusted in time** is measured differently: the real model is run twice, from a start
and from that start plus a perturbation, and the tangent once along the same perturbation; the score is the
fraction of the real change the tangent explains. The README's table has every row (about 3 hours at 120 km and 60
km, 14 min 45 s at 3 km for temperature, not for moisture).

## 4. Recordings: how they work and what they cost

> **In WOOF 1.1.7** recording a plan from a start of your own (`--init FILE`, `--init example`) is not supported: the derivative commands run on the example rows' own plans (`--row`).

The examples ship their recordings (plan packages), so nothing is recorded in the first hour. For another start, a
plan is recorded on your card first:

```bash
woof hex sensitivity --row hrrrv5-q3.96.38278.n40.60w91.30 --init example --steps 59 \
    --circle 40.6,-91.3,150 --levels 1-10 --out storm-own
```

`--init FILE` records from your own start file (with `--case` for a limited-area work directory: its static file,
boundaries and start time); `--init example` records the example's own start; `--plan DIR` replays a plan already
recorded. `--mesh` makes a row for another mesh from the suite template. No run of this release took the command
above, or any `--init`, through to a gradient on a card: without a card, `--init example` fetched the start (575 MB),
read the case and stopped at the recorder's first forward, naming the missing card. What to know:

- **A plan replays only on the card family it was recorded on**: its forward kernel images are that card's code.
  A foreign plan is refused by name.
- **A plan you record has not passed the strict check yet.** The receipt runs the whole-state dot test on it and
  says what it read.
- **Radiation's cloud sample is part of a recording** (from 0.6.2). The linear model holds the recorded forecast's
  sub-column clouds; the real forecasts a check runs beside it draw their own. Under MPAS's seed rule (the default)
  a perturbed start draws new clouds in every column it reaches; `--radiation-seed cell` with `--init` records
  under the cell rule (`config_radt_mcica_seed = cell`), so every forecast of the window draws the clouds the linear
  model holds. A plan replays under the rule it was recorded with; a command asked for the other rule refuses by
  name.
- **Recording is a forecast with every launch kept.** Measured: the 3 km storm box, about 28 minutes for its
  one-hour plan; the 60 km global mesh, 43 to 78 minutes for a 2 h 54 min window (96 GB card).

What a gradient costs against the forward it differentiates (one 96 GB RTX PRO 6000 Blackwell; the 3 km storm box,
59 steps, the build before this one, measured once each):

| | seconds | against the window's forward |
|---|---|---|
| the window's forward steps alone | 24.4 | 1 |
| a later gradient, the forward kept from the gradient before | 180.5 | about 7.4 times (arithmetic) |
| a first gradient, which also runs and keeps the forward | 239.5 | about 9.8 times (arithmetic) |
| the first command on a plan, with its answer check | 698.6 | the check runs once per plan and is kept |

Memory for that window: card 89,966 to 95,666 MiB; host 158,617 MiB with no forward kept, 249,661 MiB with it, and
258,057 MiB for a first call with its answer check (the command states its need before it starts). On the 12 km
cut a gradient took 22.6 s first and 14.7 s later on one RTX 5090.

## 5. As the linear model of a variational system

The wheel ships the window runner as a C library, `libhexad-phys` (ABI v2): `woof/hex/libexec/libhexad_phys.so` with
its header `woof/hex/include/hexad_phys.h` (`woof.hex.hexad.library()` and `woof.hex.hexad.header()` return their
paths). It is the interface a variational system's outer and inner loops call; the system itself (observation
operators, background error, minimiser) is yours.

The call order, one plan step per call (from the header):

| your linear-model step | the calls |
|---|---|
| initialise the tangent with an increment `dx` | `hexad_tl_init(s, dx)` |
| one tangent step | `hexad_tl_step(s)`, then `hexad_tl_get(s, dx)` |
| initialise the adjoint with `dx` at the window's end | `hexad_ad_init(s, dx)`, then zero `dx` |
| one adjoint step | `hexad_ad_put(s, dx)`, zero `dx`, `hexad_ad_step(s)` |
| finish the adjoint | `hexad_ad_get(s, dx)`: `dx` += the gradient at the start |

- `hexad_open(plan_dir, kset_dir, bundle_dir, manifest, options_json)` opens one session on one recorded plan: one
  session is one trajectory. A new outer loop records a new plan and opens a new session on it, passing
  `expect_manifest_sha256` so that an operator change between outer loops is refused.
- Every field pointer is a device pointer in the session's CUDA context (`{"context":"primary"}` shares the CUDA
  runtime's primary context with your program). Fields are named `theta` (K), `rho` (dry density), `u` (edge-normal
  wind), `w`, and the species of the plan (`qv`, `qc`, `qr`, `qi`, `qs`, `qg`, ...); the layout is given per field.
  Surface pressure is refused: the map from surface pressure to density belongs to your variable change.
- The first tangent sweep binds the trajectory: every boundary's hash is checked against the recording, and a
  mismatch is refused. There is no processor fallback. Return codes: 0 ok, -1 error, -2 refused
  (`hexad_last_error()` gives the sentence).
- `hexad_test_dot` runs the dot test inside the session, so your system can check the operator it was handed.

No example in this release drives the library end to end; the header is its contract, and the four commands run
the same runner. The cost per inner-loop pair (one tangent and one adjoint sweep) is the gradient cost in section 4.

## 6. Exact mode and fast mode

Two different switches carry the word "exact"; keep them apart.

- **Numerics** (forward and linear model): `--numerics fast` (the default) compiles with fused multiply-add and
  flush-to-zero; `--numerics strict` compiles every kernel the way the bit-for-bit proofs measured it. Strict
  exists to prove identity with native MPAS and RRFS-MPAS; nothing uses it by default. The
  adjoint readings in these pages are with the default.
- **The boundary layer's derivative route** (3 km and 12 km rows): `tape-free` (the default) runs the boundary
  layer's backward without its tape and differentiates more of the boundary layer (section 1); `taped` is the exact
  derivative of the held boundary layer, the answer to the byte, for windows up to about 30 minutes at 3 km. The two
  routes are two linear models: their answers differ by design, not by rounding. `taped` is the route for
  byte-for-byte comparisons of gradients; `tape-free` is the better linear model on the directions measured
  (FIRST-HOUR.md, section 3).

## 7. "Bit for bit with NOAA's RRFS-MPAS"

**What it means.** Every stored value of every field hex writes, in every file compared, is the same bit pattern as
the reference's: zero differing values, no tolerance. It is a claim about the forward model under
`--numerics strict`, on stated cases. The default (fast) forecast is not bit for bit.

**What is claimed in this release** (README, *RRFS-MPAS*; each with its case):

- `hrrrv5` on `x1.2562`, 2023-03-31 12Z, one process, every physics slot on the GPU, 6 hours: 0 of 129,420,310
  history, diagnostics and restart values different at every hour, f000 to f006 (steps 5 to 7 as written, on this
  release's wheel and one RTX 5090). Both runs read NOAA's published TEMPO tables and `ccn_activate.bin` (step 6
  names the native run folder); tables built with the strict recipe (step 2's note) change the forecast.
- `hrrrv5` on `x1.40962` (120 km, 59 levels), 2023-03-31 12Z, 24 hours: 8 same-host ranks on one card against native
  at 8 MPI ranks on the same partition, 0 of 1,958,644,100 history, diagnostics and restart values different; one
    card against one native rank, 0 of 786,309,080.
- `hrrrv5` through the plain door on two limited-area meshes, the HRRR 2024-05-21 15Z start, one hour, one process on
  one RTX 5090: the 12 km cut (2,890 cells, NOAA's 12 km cumulus on), 0 of 86,751,014 values different in 27 files;
  the 3 km storm box (38,278 cells), 0 of 624,931,668 in 11 files. The same 12 km hour on one RTX 4090: 0 of
  86,751,014.

Other meshes, dates, process counts and longer runs are not claimed. The steps below walk the six-hour `x1.2562`
case; this release gives no walk for the two limited-area hours. Native MPAS is not bit for bit across process
counts, so every claim fixes the count.

**The reference is patched, and says so.** It is RRFS-MPAS at 44c0849 built with a strict GNU recipe (single
precision, no floating-point contraction) plus exactly four labelled lines, each where native reads a value it never
set. `tools/rrfs_44c0849_reference/prepare.py` adds them and writes them to `source-correction.diff`; each has a
witness you can rerun with the steps below:

| Lines | File | Why | Witness |
|---|---|---|---|
| two `nullify(smoke_fine_fg, dust_fine_fg, dust_coarse_fg)` | `src/core_init_atmosphere/mpas_init_atm_cases.F` | case-7 init tests optional smoke and dust pointers it never nullified | the pristine `init_atmosphere_model` on step 4's case stops with signal 11 (exit 139); the patched one completes |
| `ilnb = 0` before `call snowsoil` | `src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_land.F90` | for snow thinner than `snth` (about 10 mm water) native reads the unset local `ilnb` to pick the `tsnav` formula | pristine against patched native on step 5's case: 55 of 129,420,310 stored values differ, all `tsnav` in the restart files (8 to 11 an hour), none in history or diagnostics; the patched build run twice: 0 different (the pristine count follows whatever the unset word holds) |
| `ilnb = 0` before `call snowseaice` | `.../RUCLSM/MPAS/module_ruc_ice.F90` | the same, over sea ice | the same run |

`RRFS_44C0849_REFEREE=pristine` builds the unpatched source for these witnesses. The patched build is the reference of
record and is never called pristine.

**Native outputs are not published.** You make them with the steps below; nothing in them is downloaded from us.
hex ships no RRFS-MPAS or MPAS source: `make_ref.sh` and `make_deck.py` fetch the model, its submodules, MPAS-Data and
NOAA's deck from their upstream repositories at the pins in `tools/rrfs_44c0849_reference/SOURCE-PINS.json`.

### How to reproduce it (the six-hour `x1.2562` case)

Steps 0 to 5 were walked from an empty folder on one Linux host (16 processors) with the 0.6.0 wheel installed;
the times are that walk's. The scripts are in the source tree
(`tools/rrfs_44c0849_reference/`, in the sdist beside `src/`).

**What you need.** Linux; git and network access to github.com; Python 3.11 or newer with `netCDF4` and `numpy`
(the hex environment has both); GNU Fortran and C; netCDF-C and netCDF-Fortran (`nf-config` on `PATH`; `NETCDF` names
their prefix, default `/usr`); MPICH and PnetCDF built with that GNU Fortran under one prefix `OPT` (`OPT/mpich`,
`OPT/pnetcdf`). The reference of record used GNU Fortran 15.2.0, MPICH 4.3.2 (ch4:ofi) and PnetCDF 1.14.1. With the
same three, your binaries are the reference's own bytes (step 1 gives their sums). With others they may differ, and
step 6 then measures your build, not the reference of record.

**0. Fetch REF** (18 s):

```sh
T=/path/to/gpuwm_hex-0.6.1/tools/rrfs_44c0849_reference   # the unpacked sdist
bash $T/make_ref.sh REF
```

REF then holds `SOURCE-PINS.json`, `src/MPAS-Model` (RRFSx/MPAS-Model 44c0849 with its nine submodules, each cloned at
its pin) and `MPAS-Data` (MPAS-Dev/MPAS-Data c57dbc7b, the WRF-physics data files).

**1. Build the patched reference** (80 s and 37 s with `make -j 8`):

```sh
export RRFS_FORK_CPU_OPT=OPT RRFS_44C0849_REFEREE=patched
bash $T/rebuild.sh REF WORK-atm atmosphere
bash $T/rebuild.sh REF WORK-init init_atmosphere
```

Each ends `BUILD-OK`. `WORK-atm/build/atmosphere_model` has sha256 `a128a1e86806c089...` and
`WORK-init/build/init_atmosphere_model` `fdfff07df75b280d...`; each build folder holds `SOURCE-PINS.json`
(`"referee": "patched"`), `source-correction.diff`, `flag-audit.json` (`all_ok`) and `toolchain.txt`.
`RRFS_BUILD_CPUS=8-23` pins the build to those processors.

**2. Fetch the TEMPO tables** (about 90 s). These are NOAA's published TEMPO v3 lookup tables, the files the
`rrfs-44c0849` pin set names; `--numerics strict` refuses any other TEMPO tables, in one sentence that gives the
pinned sums:

```sh
mkdir TABLES && cd TABLES
for f in freeze_water_data_tempo_v3 qr_acr_qs_data_tempo_v3 qr_acr_qg_data_tempo_v3; do
  curl -fO https://noaa-gsl-experimental-pds.s3.amazonaws.com/fix_rrfs2/v2.1.4-3.4/main/physics/hrrrv5/$f
done
```

Their sha256 begin `4624120418c25b9c`, `09e1037a6dd79bb8` and `537d784eb479ca98`. (Step 1 also builds
`build_tables_tempo_v3`, which makes tables from the same TEMPO source with the strict recipe: 270 s on 8 ranks,
sums `9cf579ec...`, `3b994bd6...`, `d319dc94...`. The six-hour run of record in the README named those; a 0.6.0
strict run refuses them, and this walk used NOAA's.)

**3. Render NOAA's deck** (1 s):

```sh
python3 $T/make_deck.py DECK --mesh x1.2562 --start 2023-03-31_12:00:00 --hours 6 --dt 720
```

It fetches `parm/hrrrv5/namelist.atmosphere`, `parm/streams.atmosphere` and `fix/stream_list/hrrrv5/` from
NOAA-EMC/rrfs-workflow 570c81f, fills the template, and prints its two changes for a global one-process run (no lateral
boundaries; the partition prefix). The namelist's sha256 begins `c5b84d079d3f72aa`.

**4. Make the case** (12 s for the init). The case is `x1.2562` with 59 levels and 9 soil levels, GFS 2023-03-31 12Z.
You need four standard MPAS inputs made from public data. This page gives no commands for the first four below:
they come from the standard MPAS static and invariant steps and WPS `ungrib`, and the walk of record used ones made
before it, so making them is yours:

- `static.nc` and `ugwp_oro_data.nc`: the static step of `init_atmosphere_model` on the `x1.2562` grid with the MPAS
  geography (`config_static_interp`, `config_native_gwd_gsl_static`);
- `invariant.nc`: the invariant stream for the same grid and levels;
- `GFS:2023-03-31_12`: WPS `ungrib` of NCEP's GFS 0.25-degree analysis for that time;
- from NOAA's public bucket (`https://noaa-gsl-experimental-pds.s3.amazonaws.com/fix_rrfs2/v2.1.4-3.4/main/physics/hrrrv5/`):
  `QNWFA_QNIFA_SIGMA_MONTHLY.dat` (sha256 `2f828eabd96a45f3...`) and `ccn_activate.bin` (`f2b8d3916560f904...`);
- `ZETA_LEVELS.txt`: `fix/vert_levels/L60.txt` from rrfs-workflow 570c81f.

In a folder with those, `GENPARM.TBL`, `LANDUSE.TBL`, `SOILPARM.TBL` and `VEGPARM.TBL` from
`WORK-init/build/src/core_atmosphere/physics/physics_wrf/files/`, and a `namelist.init_atmosphere` with
`config_init_case = 7`, start and stop `2023-03-31_12:00:00`, `config_nvertlevels = 59`, `config_nsoillevels = 9`,
`config_nfglevels = 58`, `config_nfgsoillevels = 4`, `config_met_prefix = 'GFS'`, `config_ztop = 25878.712`,
`config_specified_zeta_levels = 'ZETA_LEVELS.txt'`, `config_tc_vertical_grid = true`, `config_static_interp = false`,
`config_vertical_grid = true`, `config_met_interp = true`, `config_frac_seaice = true`,
`config_tsk_seaice_threshold = 271.4`, `config_lbc_hydrometeors_rrfs = true`, and a `streams.init_atmosphere` that reads
`static.nc` and writes `init.nc`, run:

```sh
OMP_NUM_THREADS=1 WORK-init/build/init_atmosphere_model
```

**5. Run native** (81 s, one process). In a folder NATIVE with `DECK/*` (namelist, streams, `stream_list/`),
`init.nc`, `invariant.nc`, `ugwp_oro_data.nc`, every file of `WORK-atm/build/src/core_atmosphere/physics/physics_wrf/files/`,
`WORK-atm/build/src/core_atmosphere/physics/physics_noahmp/parameters/NoahmpTable.TBL`, `ccn_activate.bin` and the three
TEMPO tables of step 2:

```sh
cd NATIVE && OMP_NUM_THREADS=1 ../WORK-atm/build/atmosphere_model
```

It writes `history`, `diag` and `restart` files for every hour, 12Z to 18Z.

**6. Run hex on the same case** (one card):

```sh
woof hex forecast --mesh x1.2562 --grid CASE/static.nc --init CASE/init.nc \
  --physics-namelist DECK/namelist.atmosphere --streams DECK/streams.atmosphere --stream-list-directory DECK \
  --physics-backend mpasphys --physics-pin-set rrfs-44c0849 \
  --orographic-statistics CASE/ugwp_oro_data.nc --stream-input invariant=CASE/invariant.nc \
  --hours 6 --physics-mode exact --numerics strict --no-products \
  --native-physics-tables NATIVE --out HEX
```

`--native-physics-tables` names the native run folder, so hex reads the same table files native read (the TEMPO
tables of step 2 among them). The folder named must hold every table file the run lists; hex refuses, in one
sentence, a folder that misses one.

Time and memory: on one RTX 5090 with its kernels already compiled and its tables already fetched, this step took
107 s. From an empty home, with the published 0.6.0 wheel and everything fetched from its public hosts, it took 113 s
on the same card: 63 s of that fetching the 1,000 MB of RRFS physics tables (once) and 20 s the prebuilt kernel pack
for the installed CUDA version (217 MB, once); host peak 3.4 GB, card peak 3.8 GB (3,758 MiB). Where a fresh install
finds no prebuilt kernel pack, it says so and compiles every CUDA module through NVRTC before the first step, which
takes far longer than the run and much more host memory.

**7. Compare every stored value:**

```sh
python $T/compare_outputs.py NATIVE HEX OUT.json
```

It reads each history, diagnostics, restart and tracer file of NATIVE, finds the same file in HEX, and compares the
storage words of every variable, NaN payloads included.

## 8. Where the rest is

- [README.md](README.md): every command and option, what each row differentiates, the memory and seconds tables.
- [CHECKS.md](CHECKS.md), [FREEZES.md](FREEZES.md): the runner's checks, and the freezes built into the runner and the derivative images, with their masks (the
  holds each row chooses: section 1).
- [RUNNER.md](RUNNER.md), [KERNEL-SETS.md](KERNEL-SETS.md), [GENERATORS.md](GENERATORS.md): building the runner, the
  kernel set and the derivative kernels from source.
- [LIMITS.md](LIMITS.md): what is not yet right.
