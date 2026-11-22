# Limits of the differentiable engine in 0.6.2

Every limit here was measured, and each states where. Nothing on this page is a limit nobody has reproduced. When a
limit is lifted it leaves this page in the release that lifts it. [GUIDE.md](GUIDE.md) says what the tangent and
adjoint are; the README's *The differentiable engine* has every number with its run.

## Hardware

- **One card family.** The adjoint commands run only on NVIDIA cards of compute capability 12.0 (GeForce RTX 50
  series, RTX PRO Blackwell). On every other family they stop before any download and say so (one RTX 4090 was
  refused so). The derivative
  images, part of the kernel set and the example recordings are compiled for that family, and a recording replays
  only on the family it was recorded on. Forward forecasts are not limited this way (README, *Requirements*).
- **One card a gradient.** Splitting one gradient over several cards (`--cards`) is experimental: a split gradient
  has not been shown equal to the one-card gradient (on the 3 km box its wind part read 7.6 per cent off), and the
  command says so before it runs.
- **Card memory.** 28,700 MiB free for the 12 km example, 73,156 MiB for the 120 km example (80,000 MiB on its
  full-physics row), 96,600 MiB for the 3 km storm box. A card whose whole memory is smaller is refused before any
  download; free memory is checked after the row's mesh file (2 MB for the 12 km cut) and before the rest.
- **Host memory on the 12 km example.** 47,104 MiB available. The README command peaked at 41,190 to 44,109 MiB on
  0.6.0's release builds and at 40,990 MiB on a pre-release build of 0.6.1, the same command with
  `--plant-error boundary-layer:1.01` at 46,486 MiB on another pre-release build of 0.6.1, and the check
  `adjoint-test --directions all --dot-directions 8` at 46,681 MiB on a 0.6.0 build (one RTX 5090 host).
- **Host memory on the 3 km storm box.** Every command states its need on this host before it downloads the
  example and stops in one sentence on a host with less: 258,057 MiB for a first `sensitivity` with its answer
  check on a large host. `adjoint-test` keeps every step boundary, in host memory up to 0.4 of what is available,
  so its need follows the host: it held 262,306 MiB and 254,885 MiB on its two card runs (512 GiB hosts), and
  420,360 MiB (450,866 MiB asked) on the run that read PASS at the row's five step sizes. On a
  host too small for that share the command gives the runner a smaller store in memory (`--bstore-ram-gib`, said
  in its output) and writes the rest to the run folder, so it runs from 102,948 MiB available at 59 steps, with
  up to about 587 GiB in the run folder, and is refused under that. On a 499 GiB host (2026-11-16, the 0.6.0
  candidate) it stated 303,410 MiB, kept 19 of 60 step boundaries in host memory, held 278,824 MiB and read PASS
  with the same digits. It has not been run on a smaller host, and the time the extra files cost there is not
  measured. A first `sensitivity --steps 59` with its answer check on the same host held 240,043 MiB, its reverse
  sweep 191.3 s and 630.4 s in all.

## What the linear model is of

- **Exact mode's boundary layer, not the default forecast's.** The derivative is of MYNN-EDMF c6d08709, the
  version NOAA's RRFS-MPAS 44c0849 pins, and of the MYNN surface layer v6.0.0: what `--physics-mode exact` runs.
  A plain `hrrrv5` forecast runs correct mode, with MYNN-EDMF v6.1.1 and the surface layer v6.1.0; a gradient of
  that forecast is refused before anything is downloaded, and the derivative of v6.1.1 is planned for the next
  build. c6d08709 leaves the latent
  heat of the scheme's own cloud mixing out of the temperature tendency: on one 3 km state of 38,278 columns that
  term is over 0.1 K/h in 8.2 per cent of the columns and over 1 K/h in 1.6 per cent. In cloud-topped boundary
  layers a sensitivity is that of the older scheme. The surface layers differ only at a run's first step.
- **Held pieces** (derivative taken as zero; each receipt names them in `row_what`): on the 12 km and 3 km rows the
  mixing length and the plume's own speed, area and condensation; the transport limiter's coefficients on the 12 km
  row and the 120 km default row, and on the 3 km row only on windows longer than 30 minutes (on the 3 km default
  route the limiter is live, its exact derivative, on windows up to 30 minutes, the example's 59 steps among them;
  a longer window holds it, runs the `gated` cloud route and says so); on the 120 km default row also radiation and
  cloud fraction (64 kernel kinds) and the boundary layer's turbulence closure and plume; the mountain-drag closure on both 120 km rows (the full-physics
  row differentiates the limiter and the plume, the plume's speed on a ramp). The lateral boundaries of a limited-area window are held at the recorded values.
- **The 3 km default cloud route is `number`.** It holds the snow riming ratio where riming is slow and the riming
  fall-speed boost where a trace of cloud water would carry it to its cap, lets rain drop number and cloud droplet
  number follow the forecast, and holds snow and cloud-ice deposition only where it would add the species' own mass
  within 200 s. `--cloud-derivative gated` holds both numbers at their forecast values; `--cloud-derivative shipped`
  runs the derivative of the builds before.
- **Number concentrations.** Ice, rain, cloud-droplet and graupel number carry no tangent at the window's end
  (measured on the 12 km cut at 2 and 15 steps, along two directions, where they had values). The aerosol numbers
  and graupel volume carry tangent and pass. A sensitivity to those four numbers reads zero.
- **No cumulus derivative.** No example recording has a cumulus scheme in it; on the 12 km row a plan recorded with
  New Tiedtke on runs its 26 launch kinds as identity in both sweeps, named in `window.json`.
- **Composite reflectivity** is not a measure the gradient can follow (the forecast itself jumps); the commands
  refuse it by name.

## What the checks show, and what they do not

- **Radiation's cloud sample sets a floor under how small a change of the start a real forecast follows**
  (from 0.6.2). Under MPAS's seed rule, the default, RRTMG seeds each column's sub-column clouds from the fractional
  digits of its lowest layer pressures, so any change of the start, once it reaches those pressures, draws new clouds
  at the next radiation call; no linear model predicts that. On the 12 km example's start (cumulus off, radiation
  every 30 minutes, the forecast door's defaults), the part of the real change in potential temperature on the lowest
  15 levels that keeps its sign when the change flips sign is 0.023 to 0.026 K (root mean square over cells) at
  +1 hour for warm bumps of 1, 0.1 and 0.001 K alike. With `config_radt_mcica_seed = cell` it is 0.0026 to
  0.0067 K, and the 0.1 K pair's plus and minus changes mirror each other with a correlation of 0.64 on cells and
  0.98 on boxes of about 165 km, against -0.49 and -0.06 under MPAS's rule. A 0.001 K bump stays under what is left
  (its own sign-flipping part is 0.0013 K). Up to +30 minutes the two rules read the same: the first radiation call,
  at the start, sees no cloud in that forecast. Measured once, on that start only, to +1 hour (its boundaries end
  there).
- **Near-ground moisture on the 12 km example is about 2 to 4 per cent short.** At ten step sizes of both signs the
  real change read 1.010 to 1.049 of the gradient's prediction (centred, 1.021 to 1.036: outside the 0.01 bar,
  2.6e-02 from 1, when read on centred differences). The cause is the
  boundary-layer plume's own speed and condensation, which the default route holds; with them live (a form not in
  this release) the centred readings were 0.998 to 1.013. The first command's five step sizes print this direction
  UNRESOLVED and its PASS does not cover it; the command names this limit under its directions.
- **The 120 km example tests on a 1-step window.** Its `adjoint-test` runs 1 step (12 minutes) on eight step
  sizes a factor of two apart and passes (the bump and near-ground moisture inside the bar); the steepest descent
  (1.008 to 1.047 of the real change) and the smooth draw stay UNRESOLVED. Over the row's 3 steps (36 minutes) only the
  bump resolves (`--steps 3` ends NOT PASSED (UNRESOLVED)), because the real model's change per unit step moves by 1 to
  6 per cent between neighbouring sizes. **Over 2 steps (24 minutes) the steepest descent reads OUTSIDE the bar:** it
  settles at 0.977 to 0.987 of the real change at the four smallest sizes (1.74e-02 from 1), so `--steps 2` fails;
  centred (each size at both signs) it reads 0.986 to 0.992 there. A gap of about 1 per cent along minus the
  gradient sits under the 1-step window's roughness and is not yet explained;
  at 2 steps the bump reads inside the bar and near-ground moisture is unresolved (0.944 to 0.980).
- **The 120 km full-physics row's own boundary-layer images fail its adjoint-test along the steepest descent**
  (`--boundary-layer taped`; the default, `floored`, passes: descent 1.4e-03 from 1, rc 0). On the same 1-step window
  (`--row hrrrv5-full-x1.40962 --boundary-layer taped`) the checks pass, the whole-state dot reads 1.0e-10 and the bump and near-ground
  moisture are inside the bar, but descent reads OUTSIDE (5.84e-02 from 1): 0.925 to 0.984 of the real change
  one-sided, and centred (each size at both signs, which cancels the curvature) 0.949 and 0.954 at the two largest
  sizes before the forecast's roughness takes over (0.971 to 1.084). The transpose is exact, so the tangent misses
  about 5 per cent of the real response along this direction. The command exits 1 and names this limit. The term
  is the boundary layer's live mixing (the diffusivities follow the turbulence speed): with the default row's
  boundary-layer images (mixing and plume held) the same window and direction read 1.030 and 1.035 centred, the
  default row's land derivative or the limiter held change nothing, and the closure-live images of the 3 km and
  12 km default route read 0.832 and 0.836. The gap sits in six coastal desert columns of West Africa (13 to 25 N,
  12 to 16 W) at the two lowest model levels, where the tangent moves theta_m by 7 to 31 K per unit step and the
  real model by -8 to +3 K. Its cause is the turbulence-energy path of the diffusivities (dK = K dq/q, dq = dqke/2q),
  unbounded where the turbulence speed q goes to zero. The default floors it with the factor q*q/(q*q + 0.5*0.5) of
  the forecast's own q: on the same window and direction the centred readings go to 0.995 and 1.000 (a floor of
  0.1 m/s gives 0.954 and 0.959), and the row's adjoint-test passes, exit status 0.
- **The window's consistency on further seeded directions, 12 km, 15 steps.** On the default route of this build seeds 1
  to 8 read +0.63, +0.33, -0.87, -0.25, +0.96, -0.34, +0.11 and +2.26 single-precision roundings of the dot
  (7.49e-08, 3.98e-08, 1.03e-07, 3.03e-08, 1.14e-07, 4.00e-08, 1.34e-08, 2.69e-07), each inside the bar of 4, with
  no one-signed drift: 5 of 8 positive, mean +0.35, standard error 0.34 (`adjoint-test --dot-directions 8`, one
  window, one RTX 5090, exit status 0; seeds 3 and 4 read the same through `--directions all --seed N`, and seed 1
  through the README command). Before the 12 km row's boundary-layer species hold, seed 3 read 7.16e-07 to
  9.80e-07, over the bar. On the exact route seeds 1 to 3 pass and seed 4 reads 1.74e-07.
  With the transport limiter's derivative live (not this row's default), twelve seeded directions read within 4 roundings: worst 3.38, mean -0.15, standard error 0.38, no one-signed lean.
- **Held on the 12 km row.** The boundary layer's seven species outputs (its tendencies of cloud water, cloud ice
  and snow, and of cloud-droplet, ice and the two aerosol numbers) are held in both sweeps: the part of a sensitivity
  that passes through the boundary layer's mixing of those species is zero. The cumulus scheme (New Tiedtke) is held by name: with
  its derivative live the 12 km window check misses by tens to hundreds of roundings, growing with the window. The
  12 km example's recorded forecast runs no cumulus scheme.
- **The exact route on the 12 km example fails** (`--boundary-layer taped`): the gradient's steepest descent reads
  0.954 to 0.959 of the real change and near-ground moisture 0.850 to 0.882. That is the held boundary layer.
- **3 km: `adjoint-test` ends PASS on the default route and FAIL on the exact route.** The
  default route on the storm example (number cloud route, live transport limiter, three double-arithmetic
  dynamics images): the window and whole-state dots passed (1.317e-08; 1.786e-07 relative, -1.50 roundings), the
  first window's finite-difference and Taylor checks read 3.653e-04 (bar 1e-02), and at the row's own step
  sizes (0.3, 0.2, 0.1, 0.06 and 0.03) the bump 0.9987 to 0.9996 (inside, 3.65e-04 from 1) and the steepest
  descent 0.9842 to 0.9981 (inside, 1.92e-03 from 1). At sizes 1 to 0.01 the descent read 0.9332 to 1.0044
  (unresolved): at 0.01 it moves the measure by only about 5 single-precision spacings of it. The exact route: the dots passed (2.825e-08, 1.313e-08), the finite-difference and
  Taylor checks read 1.017e-02, the bump 1.0102 to 1.0133 (unresolved), and the steepest descent 0.873 to 0.971
  (outside, 2.86e-02 from 1). Each was read once, on one 96 GB card. Two directions (the bump and the descent) are shown inside the bar
  at 3 km, on the default route only. In a storm direction (a 1 K warm bump under an updraft) the dot test reads 1.7e-06 to 2.2e-06 (14 to 18
  single-precision roundings of the dot), over the bar of 4.
- **Storm measures (rain water, cloud water, vertical velocity, and any measure placed on a storm cell) are not
  covered by any PASS of this release.** The PASSes above are for the README's measures (low-level temperature and moisture averaged over circles of 100 to 150 km radius).
  Read with `adjoint-test --metric` on measures placed on the storm itself, on 0.6.0's release build, 16 Nov:
  - **12 km rain water cannot be checked to 1 per cent** (15 steps, `qr` levels 1-20, a 60 km circle on the radar's
    58 dBZ band at 41.85N 92.92W). A rerun at step 0 is byte-equal to the forecast, but a nudge of 1e-06 of the 1 K
    bump moves the mean rain by up to 7.6e-08 kg/kg, about 13,000 times what the gradient predicts for it: the rain
    field reacts to a one-rounding change of its start. That roughness is as large as the linear change near a step
    of 0.01 and more than half the bar at every step size, while above a step of 0.1 rain answers the bump
    nonlinearly (0.83 to 0.85 at 1 and 0.3). Read centred (each size at both signs) the steps 0.1 to 0.003 give 0.99
    to 1.04 of the gradient. The command reads this roughness itself and ends NOT PASSED (UNRESOLVED), naming it
    (before it, the one-sided check printed FAIL, 0.118 against 0.01).
  - **12 km updraft** (`w` levels 12-30, the same circle): the bump is inside the bar read centred (0.9995 to 1.005
    of the gradient at steps 1, 0.6 and 0.3, where the roughness moves the ratio by under 0.25 per cent); the
    steepest descent reads 1.0005, 1.008, 1.012 and 1.016 at steps 1, 0.6, 0.3 and 0.1 and 1.04 to 1.05 at 0.03,
    while the bump falls the same way below 0.3 (0.983 at 0.1, 0.957 at 0.03). Every linear model tried shows the
    same rise of 1.5 to 2 per cent from step 1 to step 0.1 (the live limiter 0.998 to 1.016, the boundary layer's
    exact route 1.0008 to 1.019, the cloud route with rain and droplet number following and the live limiter 0.990
    to 1.011), and over a 5-step window the same
    direction reads 0.992 to 1.0004 at steps 0.3 to 0.03. So the rise is the forecast's own switches growing over
    the window (a branch taken at one sign and not the other, which does not shrink with the step), not a held
    piece of the linear model; the command reads this direction UNRESOLVED (8.0e-03 from 1, within its spread)
    and the bump INSIDE (4.9e-04), the first window's finite-difference check 4.9e-04.
    **What that reading is: the forecast's own melt-or-freeze switch, in one column.** Read again with the real
    model alone at 20 step sizes of both signs (one RTX 5090): in 55 of the circle's 56 cells the real change is
    0.997 to 1.009 of the gradient's prediction at every step from 0.07 to 1. One column near the melting level,
    where the cloud scheme either melts snow or sublimates it, adds 0.4 to 1.6 per cent to the whole-circle reading
    at those steps and more below them (the circle reads 1.049 at a step of 0.03 and 1.006 at 0.01). In that column
    the real model's vertical velocity is off the linear prediction by about 20 mm/s at every negative step from
    0.015 to 0.1 and by 0.1 mm/s inside a step of 0.005: a branch taken or not, not a slope. No derivative follows it and no
    step size removes it. Released one at a time, no held piece moves it: each changes the prediction by under
    0.6 per cent, the same at every step.
  - **12 km low-level temperature on the storm cell** (`theta_m` levels 1-10, the same circle, 15 steps): the real
    model is smooth along this measure's steepest descent, and read centred it gives 1.020 to 1.024 of the default
    route's prediction at steps 0.1 to 0.01: the gradient is 2.2 per cent short. The cause is the boundary layer's
    plume and mixing length, which the default route holds. With nothing of the boundary layer held
    (`--boundary-layer guarded`, an option on this row from 0.6.1) the same direction reads 0.996 to 1.000
    (measured on a 0.6.0 build with this row added); with
    the boundary layer held whole (the exact route) 1.040 to 1.043; the cloud numbers, the cloud route and the
    limiter change it by under 0.1 per cent. Through `adjoint-test` on the guarded route the measure reads INSIDE
    along its own descent (1.89e-04 from 1; the command as a whole prints FAIL on its bump, a bump at level 32
    against a measure at levels 1-10, which reads outside on both routes). **`guarded` is an option on this row,
    not its default:** the README's own 12 km command on it ends NOT PASSED (UNRESOLVED). Both dots pass (window
    4.2e-09, whole-state +0.11 roundings), the bump is inside (1.07e-03 from 1), and the steepest descent reads
    0.973, 0.980, 0.986, 0.997 and 1.005 at steps 1, 0.3, 0.1, 0.03 and 0.01, which the command cannot resolve;
    466 and 487 s, card peak 23,610 MiB, host peak 40,580 and 41,089 MiB through the command on two pre-release
    builds of 0.6.1, one RTX 5090 (the same readings, in 565 s, on the 0.6.0 wheel with this row's route added).
    The default route passes that command (352 and 398 s, 24,766 MiB, 40,775 and 41,307 MiB on the same two
    builds and card): the guarded route's minutes and memory would allow it as
    the default, and its unresolved descent is what keeps it an option. The updraft reading above is the same on the guarded route (0.9998,
    1.0071, 1.0134 and 1.0142 at steps 1, 0.6, 0.3 and 0.1).
  - **3 km updraft** (59 steps, `w` levels 12-30, a 40 km circle on a 66 dBZ cell at 39.85N 91.11W), read one-sided
    before the roughness reading: the steepest descent 1.008 to 1.023 of the gradient (1.8e-02 from 1), the bump
    0.953 to 0.998. **Low-level temperature (`theta_m`, levels 1-10) on the same cell fails too:** the first
    window's finite-difference check 0.245 against 0.01, the steepest descent 1.022 to 1.025 of the gradient's
    prediction (outside), the bump unresolved (0.60 to 1.68). So a temperature measure is covered only as the
    README poses it, not on a storm cell. Rain water was not read through the command at 3 km.
  - The transpose holds on every one of these windows (window dot 2e-08 to 2.5e-06, bar 2e-05): the backward code is
    the transpose of the tangent. The two pieces the 12 km row holds and the 3 km row runs live (rain and droplet
    number following the forecast, the transport limiter's derivative) were read on the 12 km measures and do not
    close them: with both live the rain gradient moves 4.6 per cent away from the real change, the updraft descent
    to 0.990 to 1.011 and the updraft bump to 0.985 to 0.991 (the live limiter alone: descent 0.998 to 1.016,
    bump unchanged; the boundary layer's exact route: descent 1.0008 to 1.019, bump unchanged). The 3 km readings
    above were read one-sided, before the command read the roughness; they have not been read again.
  - A map of a storm measure from this release shows where and with what sign the measure responds; its size is
    not shown to 1 per cent. `sensitivity`, `adjoint-test` and `param-grad` print the row's measured line before they
    run when a measure is `qr`, `qc` or `w` (on a mesh named with `--mesh`, that it has not been read there),
    and `adjoint-test` reads such a measure against the forecast's own
    roughness: the row's step sizes at both signs and a probe step of 1e-06 at both signs, every size read centred,
    and a size counts only where the roughness moves its ratio by at most half the bar and its neighbours agree.
    The probe's roughness is a lower bound (a larger step flips more switches), which is why the neighbours must
    agree as well.
- **On the 12 km example the whole-state dot is scored on memory allocations, not the model's fields.** The
  example's plan holds 15 step boundaries, so its 15-step window end has no field map; every 12 km command scores
  the dot on the window's memory allocations (134 of them) and prints "scored on memory allocations, not the model's
  fields" with the reason. The 3 km example scores the 12 model fields. Scoring the 12 km example on its fields needs
  the row recorded one step longer, which 0.6.1 does not carry (it keeps 0.6.0's recordings).
- **The limiter's own scratch words.** With the transport limiter live, the tangent and the adjoint disagree on
  the limiter's scratch memory (relative 5.85e-06, measured on the engine before this release); the model's fields
  pass on every window measured. The shipped checks are not affected: the 12 km and 120 km default rows hold the
  limiter, the 3 km example scores its fields (-1.50 roundings), and the 120 km full-physics row leaves that work
  memory out of the dot and proves on its plan that it is never read. A check you score on memory allocations
  (`AD_DOT_SELF_UNITS=pool`) on a window with the limiter live may read it. 0.6.1 does not carry a fix.
- **On NOAA's deck the cloud scheme's boundary-layer subgrid radii are not in the shipped cloud images.** The
  runner can take that launch down the taped route, but only with a cloud image built with it, and the images this
  release ships are not: a plan you record on NOAA's hrrrv5 deck (with `config_tempo_ml_for_bl_nc`) is refused at
  the first launch that runs the path, naming the image's missing part. The shipped example plans do not run it.
  0.6.1 keeps 0.6.0's cloud images.
- **The per-species check misses on rain at 12 km.** The runner's per-species form (`AD_DOT_PLANES=scalars`, [CHECKS.md](CHECKS.md)),
  read on the 12 km cut over 14 steps of 60 s with the transport limiter held: the rain plane's own dot is off by
  -2.2e-05 of itself (-42.9 roundings) and -1.7e-05 (-33.7) at two sizes on test direction 1, and +8.0e-06 (+18.8)
  on direction 2, against the bar of 4; on the same windows snow read +1.5 to +7.1 roundings and graupel -3.9 to
  -5.5. So the per-species check does not pass on the 12 km example in this release. The commands' whole-state dot
  scores the species together and does not run this form.
- **The whole-state dot test is of one form.** Its start direction is scaled word by word by the state, so words
  that are zero in the start state are not perturbed, and its end weights are the tangent's own result. A test with
  an independent random end weight and an additive start direction is not run by this release.
- **Consistency is not accuracy: a window is good to about 3 hours at 120 km and 60 km.** Measured as the share of
  the real model's change that the tangent-linear model explains (all levels, 0.7 as the line). At 120 km, on three
  dates, with the mountain-drag closure held as the default row holds it: at 3 hours temperature 0.78 to 0.82, wind
  0.90 to 0.91, moisture 0.76 to 0.79; at 6 hours temperature 0.61 to 0.64 and moisture 0.54 to 0.59. With the
  closure differentiated (`param-grad`, rows made with `--suite hrrrv5-exact`) about 1 hour. At 60 km: temperature
  0.65 and 0.87 in two places at 2 h 54 min, and 0.63 and 0.81 on two dates at 3 hours, so in one place of two the
  temperature is already under the line near 3 hours; at 6 hours 0.15 and 0.53. At 3 km: 14 min 45 s for
  temperature after a low-level warm bump, and not for moisture at any window tried. Moisture and vertical-wind
  sensitivities at 3 km are not validated. Six hours is not claimed on any mesh. The shipped 120 km example is three
  steps (36 minutes); the 3-hour figures are measurements of the engine on longer recordings, not a command of this
  release.
- **Near-ground wind on the exact route** reads about 1.37 times the real change along a box-wide low-level wind
  nudge (3 km, 12 steps): the held boundary-layer mixing. The default route differentiates the closure but has not
  been run along that nudge through the command. `--boundary-layer guarded` read 1.003 along it over 12 steps; over
  59 steps it has run through the command on a 96 GB card (answer check 1.304e-07, +1.09 roundings against the bar of 4, in 1,488 s on
    0.6.0's final wheel; 5.34e-08 with no one-value spike, in 1,478 s, on the build before) and has not been compared with the real model. It stays a named option: the exact route still holds the
  boundary layer.
- **The 120 km example's near-ground maps are an install check, not a forecast sensitivity.** Measured on that
  example over 3 steps, on the 87 of 348 box columns where the real forecast answers a nudge steadily: along a
  low-level east wind nudge of 0.01 to 0.3 m/s the gradient predicts 0.09 of the real model's change; along a
  low-level temperature nudge 0.77 of it; along a moisture nudge 1.4 to 1.9 times it. The cause is the boundary
  layer's mixing, held in that row's linear model. `sensitivity` prints this caution on the 120 km example.
- **`param-grad` is checked against the real model for one constant and one window.** Against reruns of the forward
  with one constant moved (120 km, 3 steps), on an earlier build of 0.6.0, the rerun slope for the
  Zilitinkevich factor was 0.98 times its gradient at a move of 0.02 in ln p and 1.11 times at 0.1; it was the only
  constant of the five tried. On an older build with another measure three of four constants disagreed with their
  reruns (gradient over rerun 0.31, 0.84, and 2.1 to 2.3). Read `param-grad` gradients as trusted only for the
  constant and window checked, and for windows of up to about an hour.
- **Near-ground wind is damped too fast.** The gradient damps a near-ground wind nudge about twice as fast as the
  real model does in its first physics step (measured; README, *The differentiable engine*).

## Recordings

> **In WOOF 1.1.7** recording a plan from a start of your own (`--init FILE`, `--init example`) is not supported: the derivative commands run on the example rows' own plans (`--row`).

- A plan recorded with `--init` has not passed the strict check; the receipt runs it.
- A plan replays only on the card family that recorded it.
- The 6 km gradient route (`--gradient-mesh 6km`) is shown on one storm, one measure and one window only; any other
  is refused by name. **No example this release ships runs it:** the 6 km plan that ships is of 2024-05-21 15Z, the
  3 km storm example is of 2023-03-31 18Z, and on the example's own day three of the route's four readings were
  under its bar of 0.99 (cosine with the 3 km gradient 0.992 for temperature and 0.986 for moisture on one measure,
  0.80 and 0.96 on a second), so there it stays refused. It runs only with your own 3 km recording of
  2024-05-21 15Z given with `--plan`.

## The variational interface

- `libhexad-phys` (the C library) ships with its header, but no example in this release drives it end to end.
- One session is one recorded trajectory: a new outer loop records a new plan.
- None of an assimilation system ships: no observation operators, no background-error model, no minimiser.
