# mp_physics = 28: the matched forecast trajectory

**Status of this document at the moment it was first committed: DESIGN ONLY.
Not one number below the line "MEASUREMENTS" had been produced when the
design, the metric set and the verdict rule were written.** That ordering is
the point. A comparison whose statistics are chosen after the numbers are
visible measures the person choosing them.

The registry entry for `thompson-aerosol-mp28` publishes
`forecast_trajectory_comparison: null` and warns, correctly, that the scheme
is **UNVERIFIED against a WRF forecast**: its evidence is 22 single-call
column fixtures plus a self-consistency smoke test. This document is the
attempt to close that one gap, and an accurate statement of how far it gets.

---

## 1. Why the case is idealized, single-domain and periodic

A matched *nested, real-data* forecast is not currently possible. Two
independent hard blockers, both already measured elsewhere in this tree:

1. **WRF's own `real.exe` refuses the configuration.** With
   `wif_input_opt = 0` and `mp_physics = 28`,
   `dyn_em/module_initialize_real.F:2734-2736` is a fatal error. There is no
   real-data WRF forecast to compare against without supplying a WIF aerosol
   input stream, which WOOF has no ingest for.
2. **There is no aerosol lateral boundary condition in WOOF.**
   `woof/ingest/lateral_bc.py` carries only `qv` from an external boundary
   snapshot; every other advected scalar (including `nc`, `nwfa`, `nifa`)
   takes WRF's `flow_dep_bdy` treatment, which is *zero on inflow*. Stock WRF
   forces `qnwfa`/`qnifa` at the boundary from the metgrid WIF stream.
   The measured consequence is a depletion front advancing at **1.005 of the
   wind speed**: a 100 km nest has lost its initial aerosol everywhere within
   83 minutes. Comparing that against WRF would be comparing two different
   problems and reporting the difference as a port error.

Both blockers are *boundary* blockers. Neither exists on a **periodic**
domain: there is no inflow face, so there is no aerosol boundary condition to
get wrong, and there is no `real.exe`. The idealized periodic case is
therefore not a weaker substitute chosen for convenience: it is the only
configuration in which the comparison is about the microphysics at all.

## 2. The case

A single domain, doubly periodic, unsheared, with one warm bubble in a
Weisman–Klemp (1982) sounding: WRF's own `em_quarter_ss` initializer with
the hodograph removed.

| | value | source |
|---|---|---|
| grid | 120 × 120 × 40 | - |
| dx = dy | 2000 m | `em_quarter_ss` stock namelist |
| ztop | 20000 m | `em_quarter_ss` stock namelist |
| eta | WRF's `stretch_grid` exponential, `z_scale = 8000/ztop` | `module_initialize_ideal.F:658-660` |
| dt | 12 s, `time_step_sound = 6` | `em_quarter_ss` stock namelist |
| length | 7200 s = 600 steps | - |
| history | every 600 s, 13 frames | - |
| sounding | WK82 analytic θ(z), RH(z); **u = v = 0** | `woof/verify/cases/wk82.py` → `input_sounding` |
| thermal | 3 K, cos² inside the unit ellipsoid, 10 km radius, centre z = 1500 m, half-depth 1500 m | `module_initialize_ideal.F:1089-1120` |
| lateral BC | `periodic_x = periodic_y = .true.` | - |
| physics | microphysics ONLY: no radiation, no surface layer, no LSM, no PBL, no cumulus | - |
| dissipation | `diff_opt=2, km_opt=4, c_s=0.25`; `diff_6th_opt=2, diff_6th_factor=0.12`; `khdif=kvdif=0` | WOOF implements WRF's `smag2d_km` |
| damping | `damp_opt=3, zdamp=5000, dampcoef=0.2`, `w_damping=0` | WRF Registry defaults |
| advection | `h_mom=5, v_mom=3, h_sca=5, v_sca=3`, `moist_adv_opt = scalar_adv_opt = 1` | matches WOOF's fixed orders |
| acoustic | `epssm=0.1, smdiv=0.1, emdiv=0.01`, `non_hydrostatic` | WRF Registry defaults |
| aerosol | neither model is given an aerosol field; both fill WRF's synthetic CCN/IN profile in their own `thompson_init` port | `module_mp_thompson.F:493-558` |

**No mean wind is deliberate.** With shear, a storm translates and on a
periodic domain re-enters its own outflow; the resulting flow is dominated by
a wrap-around interaction rather than by microphysics. Unsheared, the bubble
produces a pulsing multicellular cluster whose cold pool spreads outward
symmetrically. At ~15 m/s spread the cold pool needs about 8000 s to reach
the periodic edge from the domain centre, so the whole 7200 s window is free
of self-interaction.

### 2.1 The initial condition is WRF's, not a transcription of it

WOOF is initialised **from WRF's own `wrfinput_d01`**: the file `ideal.exe`
writes. `tools/mp28_matched/run_arwen.py` reads `U/V/W/PH/PHB/T/T_INIT/MU/
MUB/PB/ALB/ZNW/P_TOP` and every moisture and number species out of that file
and installs them as WOOF's base state and prognostics. This removes the
initialisation from the comparison entirely: the two models' t = 0 states are
the same numbers, to float32 storage.

That is a deliberate strengthening over the alternative (each model building
the state from the same analytic recipe), because WOOF's builders are
*transcriptions* of WRF's initializer and a transcription difference would
have entered the trajectory as if it were a microphysics difference.

## 3. Where the two models correspond, and where they do not

A comparison that overstates its own applicability is worth less than none.
Enumerated accurately:

**They correspond in:** the initial state (bit-for-bit as float32, §2.1);
the grid and the eta coordinate; the lateral boundary condition (periodic,
so no boundary data exists in either); the physics selection (microphysics
only); the advection orders; the dissipation and damping package and every
one of its constants; the acoustic-step parameters; the moist coupling
(`use_theta_m = 1` in WRF, `moist_cq` in WOOF: the same `cq` coupling);
and the aerosol initial condition, which both models synthesise from the same
`thompson_init` profile constants rather than reading.

**They do NOT correspond in, and this comparison cannot separate:**

1. **The dycore is not the same code.** WOOF is an independent FP32 GPU
   implementation. It is `wrf-matched-run` on `mp_physics = 8`, which is what
   makes the difference-in-differences design of §4 possible, but a
   trajectory difference between the two models is not attributable to
   microphysics without that control.
2. **Precision.** WRF here is a float32-storage / float32-arithmetic Fortran
   build on x86-64 with `-ftree-vectorize`; WOOF is float32 on sm_120, which
   **flushes float32 subnormals to zero in all arithmetic** and cannot be
   made not to. Divergence in a chaotic convective flow is expected and is
   not by itself evidence of a port defect.
3. **The Thompson lookup tables are not byte-identical.** WRF computes
   `freezeH2O.dat`, `qr_acr_qg_V4.dat` and `qr_acr_qsV2.dat` at first run
   from its own compiled source; the values depend on the optimization flags.
   WOOF ships tables built with `-O2 -fno-tree-vectorize` (the flag its
   column oracle pins); a stock WRF build uses `-O2 -ftree-vectorize
   -funroll-loops` and produces *different* files. §4 measures how much that
   alone is worth by running WRF both ways.
4. **This is one case.** Unsheared warm-bubble convection over a flat
   surface with no radiation, no surface fluxes and no shear exercises warm
   rain, ice initiation, riming and sedimentation. It does **not** exercise:
   supercell dynamics, aerosol advection across a boundary (there is none),
   surface aerosol emission over a heterogeneous surface, radiative feedback
   through the effective radii, or any interaction with a PBL scheme. A pass
   here is evidence about the scheme's forecast behaviour in this regime and
   is not a general licence.
5. **No observations are involved.** Nothing here says either model is
   *right*. It says whether WOOF's mp=28 does what WRF's mp=28 does.

## 4. The design: difference-in-differences, with two floors

The quantity of interest is not "does WOOF's trajectory equal WRF's", at
2 km resolution in deep convection, no two distinct implementations hold a
trajectory for two hours, and reporting their divergence would measure
chaos, not the port. The quantity of interest is the **aerosol scheme's
signature**: what changes when you go from `mp_physics = 8` to
`mp_physics = 28`, holding everything else fixed.

Six runs, one initial condition per microphysics option:

| run | model | build | mp |
|---|---|---|---|
| `wrf-vec-mp08` | WRF v4.6.1 | gfortran 13.3.0 `-O2 -ftree-vectorize -funroll-loops` (WRF's own default) | 8 |
| `wrf-vec-mp28` | WRF v4.6.1 | same | 28 |
| `wrf-novec-mp08` | WRF v4.6.1 | identical source, `-O2 -fno-tree-vectorize` | 8 |
| `wrf-novec-mp28` | WRF v4.6.1 | same | 28 |
| `arwen-mp08` | WOOF | CUDA / sm_120, FP32 | 8 |
| `arwen-mp28` | WOOF | same | 28 |

This yields three quantities per metric `M` and time `t`:

* **the signature** `Δ_X(M,t) = M_28(X,t) − M_8(X,t)` for `X ∈ {WRF, WOOF}`,
  what the aerosol scheme does to the forecast, in each model;
* **the implementation floor** `F(M,t) = |M_8(WOOF,t) − M_8(WRF,t)|`: how
  far apart the two models are on a scheme WOOF is already
  `wrf-matched-run` on. Nothing about mp=28 can be demanded tighter than
  this;
* **WRF's own ambiguity** `V(M,t) = |M(WRF-vec,t) − M(WRF-novec,t)|`: how
  much WRF's answer moves when one optimization flag changes, with the source
  bit-identical. This is the width of the target.

## 5. Metrics: declared before any run

Computed at every history time on the whole domain. `nc`, `nwfa`, `nifa`
exist only for mp=28 and are compared model-to-model, not 8-to-28.

**M1** domain-total accumulated surface precipitation, `sum(RAINNC)`
**M2** domain-max `RAINNC`
**M3** domain-max `w`
**M4** domain-mean and domain-max of `qc`, `qr`, `qi`, `qs`, `qg` (10 metrics)
**M5** domain-mean and domain-max of `nc` (droplet number) (the most direct
       aerosol signature there is
**M6** domain-mean `nwfa` and domain-mean `nifa`) aerosol consumed by
       activation, nucleation and scavenging
**M7** the vertical profile of domain-mean `qc` and domain-mean `qi`
**M8** normalised RMS field difference between the two models,
       `||A − W||₂ / ||W||₂`, for every 3-D prognostic, at every frame,
       the raw trajectory-divergence curve, reported for mp=8 and mp=28

**Primary analysis window: 0 – 5400 s.** 5400 – 7200 s is reported as
secondary, because a cold pool interacting with itself across the periodic
boundary near the end of the run is a regime the design does not control.

## 6. Verdict rule: declared before any run

mp=28 **ships in engine 1.5** (remaining `implemented-unverified`, never a
default, no new maturity tier) with this document populating
`forecast_trajectory_comparison`, if and only if all four hold:

* **V1: sign.** Over all (metric, time) pairs in M1–M6 for which
  `|Δ_WRF| > F` (i.e. WRF's aerosol signature is larger than the
  implementation floor and therefore actually resolvable), `sign(Δ_ArWen)`
  equals `sign(Δ_WRF)` in **at least 90%** of them.
* **V2: magnitude, floor-calibrated.** For the three headline aerosol
  metrics (domain-mean `nc`, domain-mean `nwfa`, domain-total `RAINNC`)
  at t = 1800, 3600 and 5400 s:
  `|Δ_ArWen − Δ_WRF| ≤ max(0.5·|Δ_WRF|, F)`.
  The test cannot demand better agreement than the dycore itself delivers,
  which is why `F` is in the bound; it also cannot be passed by a floor that
  swallows everything, which is why `F` is published alongside.
* **V3: no scheme-level amplification.** For every metric in M1–M4 and for
  M8, WOOF-vs-WRF disagreement on mp=28 is **not more than 3×** the same
  disagreement on mp=8 at the same time. mp=28 shares the dycore with mp=8;
  a much larger disagreement is the microphysics and nothing else.
* **V4: bounded and finite.** No non-finite value in any field of either
  WOOF run; `nwfa` and `nifa` stay strictly inside WRF's terminal clamp band
  and show **no monotone depletion trend** (with periodic boundaries there is
  no sink but activation and scavenging, so a depletion trend would be a
  conservation defect).

**Any failure holds mp=28 out of 1.5** and the failing metric names the
defect. Widening a bound after seeing a number is not available: the bounds
above are the ones committed with this file, and the git history of this
document is the receipt.

---

# MEASUREMENTS

*Everything above this line is byte-unchanged from commit `f87ca87f`, which
was made before `ideal.exe` had been run once at the case's real size.*

**Scope correction, 2026-10-02.** The frozen design above describes the
implementation and assumptions at registration. Since then WOOF has added
WRF's monthly WIF aerosol climatology, specified-boundary supply of `nwfa` and
`nifa`, and parent-to-child transport of those fields. A specified run without
the dataset is refused unless the synthetic source is deliberately selected.
No matched WRF forecast here covers those later paths; see
[PHYSICS.md](../PHYSICS.md).

The design's claim that sm_120 necessarily flushes every float32 subnormal
was also corrected later. The observed flushing came from the compile route
used for these runs, where CuPy appended `-ftz=true`, not from an unavoidable
hardware property. Routes that preserve IEEE subnormals exist; see
[HARDWARE.md](../HARDWARE.md#fp32-subnormals-and-gpu-model-caveats).
These corrections do not alter the frozen design, recorded values or HOLD
verdict. All comparisons here are code verification against WRF, not
validation against observations.

## 7. Provenance

**WRF.** Source: the v4.6.1 release tarball, sha256
`b8ec11b240a3cf1274b2bd609700191c6ec84628e4c991d3ab562ce9dc50b5f2`, unpacked
twice into two independent trees. `configure` option 32 (GNU gfortran/gcc,
**serial**), nesting option 0; `NETCDF=/usr`. Compiler gfortran/gcc 13.3.0
(Ubuntu 13.3.0-6ubuntu2~24.04.1), glibc 2.39 (Ubuntu GLIBC 2.39-0ubuntu8.8),
netCDF 4.9.2 with netCDF-Fortran 4.5.4, x86-64. Compile target
`em_quarter_ss`.

| | flags | `ideal.exe` sha256 | `wrf.exe` sha256 |
|---|---|---|---|
| build A ("vec") | `-O2 -ftree-vectorize -funroll-loops`, **WRF's own gfortran default, unedited** | `160a50a5eeeddb16809b4e401421877d14240ebbe4a930dbfa1ecbed1b27567f` | `cd611280c8385618ee10b1bf88b5f8f3e320de7a179a41e5db966cd6da5a09fd` |
| build B ("novec") | `-O2 -fno-tree-vectorize`, one line of `configure.wrf` changed, **no source change** | `57271e95c63b8e3c604f50b9cd04e288bca883500778e5f68dd40a680a50f1bc` | `e81d032f101bd86eedbebf4905530ba0596a5096059578c5d07c0662a94ef500` |

Source hashes, identical in both trees:
`phys/module_mp_thompson.F` =
`fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802`,
`phys/module_mp_radar.F` =
`aa99da858be41efa579966680708d230123a7417560af0eb2e24f4c94e253688`.
**Both match, byte for byte, the pinned reference copy that already existed
on this node** (`wrf461/phys/`), which this lane created two isolated
build trees rather than touch; its two files were re-hashed at the end of the
lane and are unchanged.

`CCN_ACTIVATE.BIN` in WRF's `run/` is
`f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd`: the
byte-identical file WOOF now ships, and the hash its loader pins.

**The Thompson lookup tables are build-dependent, and all three sets differ.**
WRF computes `freezeH2O.dat`, `qr_acr_qg_V4.dat` and `qr_acr_qsV2.dat` at
first run from its own compiled source:

| | `freezeH2O.dat` | `qr_acr_qg_V4.dat` | `qr_acr_qsV2.dat` |
|---|---|---|---|
| build A | `c7a01fa7…` | `441bd836…` | `910fb31d…` |
| build B | `a054a3a4…` | `bcef0ef3…` | `5e7438b6…` |
| WOOF (pinned) | `c235d1ce…` | `89b77985…` | `47350be2…` |

Three distinct table sets from the same physics source. Each model used its
own throughout; no table was moved between them. This is a real term in the
comparison and it is the reason build B exists.

**WOOF.** Branch `development branch` at the design commit,
CUDA/CuPy 14.1.1, Python 3.12.13, NVIDIA GeForce RTX 5090 (sm_120, no ECC).

**Cost.** Each 600-step forecast: WRF build A 425 s (mp=8) / 490 s (mp=28)
on one core; build B 700 s / 810 s. WOOF 24.0 s (mp=8) / 30.6 s (mp=28) on
one GPU. Everything in this document is under two hours of one node.

## 8. Screens that must pass before any number is read

**Dual-run byte comparison (the 5090 has no ECC).** Every WOOF
configuration was run twice into separate directories. All 13 frames of both
mp=8 runs and all 13 frames of both mp=28 runs are **byte-identical across
every field**. No silent corruption.

**t = 0 agreement, exactly zero in ten fields of thirteen, and 1e-8 in the
other three, across the hydrometeor mixing ratios and the wind field this
case integrates.** The normalised RMS difference between WOOF's t = 0
read-back and WRF's state, read out of the comparison receipt rather than
asserted, and re-measured per element under the certified ULP definition
by the full-carrier digest over the successor gate's staged pair, where the
`T` split/recombine round trip lands exact from a history frame: verdict
**PASS**, max 0 ULP in every staged array, both schemes
([mp=28 receipt](../../../woof/data/certification/mp28_matched_t0_readback_digest_mp28.json),
[mp=8 receipt](../../../woof/data/certification/mp28_matched_t0_readback_digest_mp08.json),
provenance in `woof/data/certification/README.md`):

| field | mp=8 | mp=28 |
|---|---|---|
| `QVAPOR, QCLOUD, QRAIN, QICE, QSNOW, QGRAUP, QNRAIN, QNICE, W` | **0.0** | **0.0** |
| `QNCLOUD` | - | **0.0** |
| `T` | 2.5098e-09 | 2.5098e-09 |
| `QNWFA` | - | 2.1765e-08 |
| `QNIFA` | - | 1.7313e-08 |

An earlier draft of this section asserted the difference was exact in all
thirteen. It was exact in ten, and is corrected here rather than quietly
dropped.
Both mechanisms are in `tools/mp28_matched/run_arwen.py` and neither is a
transcription of WRF's initializer sneaking back in:

* **`T`** is the only field that is split and recombined. WRF stores
  `theta − 300`; WOOF stores an absolute base `thb` and a perturbation, so
  `build_state` forms `thp = T + 300 − thb` and the frame writer forms
  `total_theta() − 300`. That round trip is exact in real arithmetic and
  costs one float32 rounding in each direction. It is identical for mp=8 and
  mp=28, which is why the number is the same in both columns.
* **`QNWFA`/`QNIFA`** are read straight across from `wrfinput_d01`, and are
  then *overwritten*, in both models, by `thompson_init`'s synthetic profile:
  WOOF's `microphysics_init_receipt` is
  `{'ccn': True, 'in': True}` (§8, and `evidence/arwen-mp28-a-run.json`).
  So what is being compared at t = 0 is not a copy against its source, it is
  **CUDA's evaluation of WRF's analytic CCN/IN profile against gfortran's**.
  2e-08 is what two float32 evaluations of the same closed-form expression
  cost. `QNCLOUD`, which the same profile does not set, is exactly 0.0.

Each residual is 1e-8 relative, against a perturbation growth that reaches
3.2e-02 in `W` by t = 600 s: six orders of magnitude larger. It changes no
conclusion below. Publishing an exactness the receipt does not record would
have been the kind of claim this document exists to avoid.

The substance of §2.1 stands: the two models start from the same numbers to
float32 storage in every field WOOF reads directly, and to float32 rounding
in the three it does not.

**The aerosol profile installed itself.** WOOF's
`microphysics_init_receipt` is `{'ccn': True, 'in': True}` and the domain
enters with mean `nwfa` = 5.5028e+07 kg⁻¹, peak 1.338e+08, mean `nifa` =
5.4930e+05, i.e. WRF's synthetic `thompson_init` profile, not zeros and not
the clamp floors.

## 9. The verdict, as declared

| condition | WOOF vs WRF (build A) | **CONTROL: WRF build B vs build A** |
|---|---|---|
| **V1** sign agreement | **PASS**, 43/47 = 91.5% (threshold 90%) | PASS, 108/109 = 99.1% |
| **V2** floor-calibrated magnitude | **PASS**, 9/9 rows | PASS, 9/9 rows |
| **V3** no scheme-level amplification | **FAIL**, 7 of 197 rows over 3×, worst 4.95e+04 | **FAIL**, 17 of 195 rows over 3×, worst 8.08e+02 |
| **V4** bounded and finite | **PASS**, 0 non-finite, 0 bound violations, no depletion trend | PASS |
| **declared outcome** | **HOLD** | **HOLD** |

**The control column is the finding.** The right-hand column applies the
identical machinery to **WRF against itself**: same source, same case, same
initial file, one optimization flag different. It fails V3 too. A gate that
unmodified WRF v4.6.1 cannot pass against its own recompilation is not
measuring the port; on this case it is measuring chaos, and V3 as written was
mis-specified. That is stated here rather than fixed, because the rule was
committed in advance and the accurate thing to do with a rule that fails is to
report the failure and name the reason, not to rewrite it.

The distribution behind V3 says the opposite of "mp=28 amplifies". V3 as
declared covers M1–M4 *and* M8, which is 197 rows here (111 scalar M1–M4
rows plus 86 M8 field-difference rows) and 195 in the control. Counted over
all of them, the **median** mp28/mp8 disagreement ratio is **0.691** and
**190 of 197** are ≤ 3; the control's median is **0.621** and 178 of 195 are
≤ 3. Counted over the 111 scalar rows alone the WOOF median is **0.628**
with 104 of 111 ≤ 3, against a control scalar median of **0.429**
with 93 of 109 ≤ 3. Every one of the seven failures is a scalar row; not one M8 row
fails. (An earlier draft printed the WOOF figures on the scalar subset
beside control figures on the full set: two different denominators in one
sentence. Both are given here, matched.) In other words
the median ratio is below one in both the WOOF comparison and the WRF
control. Most rows are in the decorrelated regime; this ratio summary does
not establish better agreement or overturn the failed worst-row condition.
The seven WOOF failures are `rainnc_max`/`rainnc_sum` at t = 600 s (where mp=8
rain is 5.8e-11 mm, so the ratio is a division by noise), `rainnc` at
t = 1200 s (3.4–3.5×), `qi_max` at 4200 s (5.1×), `qs_max` at 5400 s (4.9×)
and `w_max` at 5400 s (9.3×).

## 10. Why the long run cannot resolve more: measured, not asserted

M8, the normalised RMS field difference `||A − W||₂ / ||W||₂`:

| t (s) | WOOF−WRF `W` mp8 \| mp28 | WRF−WRF `W` mp8 \| mp28 | WOOF−WRF `QVAPOR` mp8 \| mp28 | WRF−WRF `QVAPOR` mp8 \| mp28 |
|---|---|---|---|---|
| 0 | 0.0000 \| 0.0000 | 0.0000 \| 0.0000 | 0.000000 \| 0.000000 | 0.000000 \| 0.000000 |
| 600 | 0.0317 \| 0.0316 | 0.0010 \| 0.0010 | 0.000157 \| 0.000157 | 0.000002 \| 0.000002 |
| 1200 | 0.0645 \| 0.0763 | 0.0010 \| 0.0009 | 0.001222 \| 0.001100 | 0.000013 \| 0.000010 |
| 1800 | 0.2075 \| 0.2595 | 0.0076 \| 0.0163 | 0.001990 \| 0.001702 | 0.000060 \| 0.000067 |
| 2400 | 0.2823 \| 0.3331 | 0.0465 \| 0.0310 | 0.002453 \| 0.003023 | 0.000099 \| 0.000064 |
| 3000 | 0.5085 \| 0.5451 | 0.0282 \| 0.0206 | 0.005493 \| 0.005505 | 0.000267 \| 0.000110 |
| 3600 | 2.1572 \| 0.9867 | 0.0594 \| 0.0133 | 0.008776 \| 0.007286 | 0.000452 \| 0.000238 |
| 4200 | 1.6891 \| 0.9586 | 0.0752 \| 0.0467 | 0.008456 \| 0.007518 | 0.000349 \| 0.000289 |
| 4800 | 2.3846 \| 0.7510 | 0.0546 \| 0.0481 | 0.009232 \| 0.006717 | 0.000360 \| 0.000342 |
| 5400 | 1.9532 \| 1.4258 | 0.0526 \| 0.0609 | 0.011973 \| 0.012316 | 0.000349 \| 0.000604 |
| 6000 | 5.0505 \| 1.2369 | 0.0475 \| 0.2091 | 0.014610 \| 0.016655 | 0.000369 \| 0.001254 |
| 6600 | 3.1645 \| 1.0975 | 0.0469 \| 0.2071 | 0.013479 \| 0.012378 | 0.000361 \| 0.001508 |
| 7200 | 3.2606 \| 1.5908 | 0.0460 \| 0.3109 | 0.014328 \| 0.014805 | 0.000356 \| 0.002290 |

All thirteen history times, uncurated. An earlier draft printed ten of them;
the three that were missing (4200, 6000, 6600) are the ones where WOOF's
mp=8 `W` difference is largest (5.05 at t = 6000 s) so their absence
flattered the table. They are the same story the surrounding text tells: past
saturation the number is a sample, not a measurement.

Read this as a perturbation-growth experiment. A `W` difference of RMS ≳ 1
means the two fields are unrelated: the updraft is in a different place.
WOOF's initial difference from WRF (3.2% in `W` at t = 600 s) is about 30×
WRF's own compiler-flag difference (0.10%), and it saturates by t = 3600 s.
WRF's own perturbation, 30× smaller, has not saturated by t = 7200 s. Both
grow with an e-folding time of order 10–15 minutes, which is what an
unsheared 2 km deep-convection case does.

**The consequence for the metrics: after roughly t = 2400 s the two models
are integrating different realisations of the same statistical problem, and
every scalar comparison past that point is a comparison of two samples, not
of two answers.** That shows directly in the floor: the WOOF–WRF mp=8
disagreement in domain-total rainfall grows from 0.02 mm at t = 1200 s to
4.9 mm at 1800 s, 226 mm at 3600 s and 656 mm at 5400 s, while WRF's own
flag-to-flag disagreement in the same quantity is 3.6e-05, 0.056, 1.53 and
6.35 mm. V2 therefore passes with room to spare, but it passes against a
bound the floor sets, and the floor is bigger than the aerosol signature
from t ≈ 3000 s onward. **The long run cannot resolve the aerosol signature
against the implementation floor past about 50 minutes.** It is reported
anyway, in full, because that limit is itself the measurement.

Two things in the long run are *not* limited this way, because they are not
chaotic:

* **`nwfa` conservation.** Domain-mean `nwfa` in kg⁻¹, at t = 0 and at
  t = 7200 s, to the precision the series receipts carry:

  | | t = 0 | t = 7200 s |
  |---|---|---|
  | WRF (build A) | 55028345.65 | 55087825.35 |
  | WOOF | 55028345.45 | 55079394.66 |

  The two models agree on the domain aerosol budget to **1.530e-04 relative
  at t = 7200 s** (`nifa`, on the same basis, to 2.451e-04), and both show
  the same slight *increase* (WRF's surface emission `nwfa2d` outrunning
  activation and scavenging) not the depletion an inflow boundary would
  have produced. This is the single cleanest number in the document and it
  is precisely the quantity the port exists to get right. An earlier draft
  typed a third figure that matches neither ratio above; both are now
  recomputed from the committed series by
  `tests/test_mp28_matched_trajectory_doc.py` rather than typed.
* **The sign of the aerosol effect.** Both models make mp=28 rain *more*
  than mp=8 at every output time, and both make mp=28 hold more cloud water
  aloft. WRF's Δ(domain-total rain) at t = 1800 s is +1.427 mm against
  WOOF's +2.090 mm: same sign, ratio 1.46, while the trajectories are
  still correlated.

## 11. Short-window paired test: a post-hoc addition, labelled as such

Its motive was in the measurement above, not in the answer: since the long
run decorrelates, the pre-declared M8 statistic was re-measured in the one
regime where "matched trajectory" can mean what it says. Both models are
started from the **same mature WRF state at t = 1800 s** (every
microphysical process fully active, condensate and ice present everywhere the
storm is) and run 50 steps (600 s). WRF reaches that state by running
continuously from t = 0 and beginning history output at t = 1800 s; WOOF
reads that history frame as its initial condition. The statistic (M8) and the
condition applied to it (V3) are the pre-declared ones; only the window is
new, and it was chosen after the long run had been read. That is disclosed
rather than hidden.

Normalised RMS field difference, WOOF vs WRF build A, `mp=8 | mp=28`:

| steps | t (s) | `W` | `T` | `QVAPOR` | `QCLOUD` | `QICE` | `QNWFA` | `QNIFA` |
|---|---|---|---|---|---|---|---|---|
| 0 | 1800 | 0 \| 0 | 0 \| 0 | 0 \| 0 | 0 \| 0 | 0 \| 0 |, \| 0 |, \| 0 |
| 5 | 1860 | 1.800e-2 \| 1.797e-2 | 4.91e-5 \| 4.55e-5 | 2.88e-4 \| 2.51e-4 | 6.65e-2 \| 8.41e-2 | 4.41e-2 \| 2.77e-2 |, \| 5.53e-4 |, \| 3.56e-4 |
| 10 | 1920 | 2.19e-2 \| 2.02e-2 | 1.01e-4 \| 9.62e-5 | 4.94e-4 \| 3.97e-4 | 1.06e-1 \| 9.33e-2 | 8.10e-2 \| 3.48e-2 |, \| 9.65e-4 |, \| 6.39e-4 |
| 25 | 2100 | 5.56e-2 \| 4.46e-2 | 2.67e-4 \| 2.58e-4 | 7.74e-4 \| 7.14e-4 | 1.57e-1 \| 1.19e-1 | 8.00e-2 \| 4.99e-2 |, \| 2.28e-3 |, \| 1.27e-3 |
| 50 | 2400 | 1.145e-1 \| 1.184e-1 | 3.62e-4 \| 3.87e-4 | 1.27e-3 \| 9.38e-4 | 4.28e-1 \| 2.66e-1 | 1.07e-1 \| 7.71e-2 |, \| 8.43e-3 |, \| 1.82e-3 |

**V3 on the short window: PASS.** 111 rows, worst ratio **2.195** (`QNRAIN`
at step 10). The next four are `QNRAIN` at step 2 (**1.501**), `QGRAUP` at
step 2 (1.266), `QCLOUD` at step 1 (1.263) and `QNRAIN` at step 7 (1.219);
everything else is at or below 1.14. (An earlier draft published a ceiling of
1.3 on the rest of the distribution, which the 1.501 row contradicts.) All 111 are
under the 3× threshold, and the two rows above 1.5 are both `QNRAIN`: rain
number, the field the aerosol scheme does *not* touch and the one whose
mp=8 norm is smallest.

The short-window WOOF runs were duplicated like the long ones, and the
byte comparison was made rather than assumed: `sw-arwen-mp08` against
`sw-arwen-mp08-b` and `sw-arwen-mp28` against `sw-arwen-mp28-b`, 11 frame
files each, SHA-256 per file, **11/11 identical in both configurations**.
`shortwindow.json` does not itself carry that screen (the long-run
comparison receipt does) so it was measured directly against the run
directories.

Three things are worth saying about this table.

1. **mp=28 and mp=8 diverge from WRF at the same rate.** In `W` the two
   columns agree to three significant figures at 5 steps (1.797e-2 vs
   1.800e-2) and remain within 4% of each other at 50 steps. Whatever the
   port's remaining disagreement with WRF is, it is already present with
   mp=8 at this scope. That is a relative statement, not evidence of small
   absolute disagreement. Section 10 measures 3.2% normalized RMS in `W` at
   600 s, about 30 times WRF's own one-flag recompilation difference of 0.10%.
   The successor short-window gate records mp=8 `W` disagreement of
   1.800e-02 after 60 s, about 2.4 times its WRF control's 7.555e-03 accumulated
   over the preceding 1800 s ([section 10](mp28-shortwindow-gate.md#10-the-verdict-as-declared)).
2. **The aerosol fields have the smallest normalized differences.**
   `QNWFA` is 8.4e-03 and `QNIFA` 1.8e-03 after 50 steps. These smooth fields
   have large background values, so normalized RMS differences are naturally
   small. Their ranking does not show that aerosol is carried more accurately
   than condensate.
3. **The condensate RMS numbers look large and are not.** `QCLOUD` at
   4.3e-01 after 50 steps is a spatially intermittent field whose norm is set
   by a handful of strong cells; a small displacement of one cell moves the
   RMS a long way. It is reported unmassaged.

`U`, `V`, `PH` and `MU` are absent from this table because the WOOF frame
writer does not dump them; that is a gap in the instrumentation, not a
result, and it is the one thing a repeat of this test should fix first.
`RAINNC` is excluded as meaningless here: WOOF's accumulator restarts at
zero while WRF's carries 1800 s of rain.

## 12. Verdict

**By the rule declared in §6, before any run: HOLD. V3 failed.**

That is the answer to the question as it was asked, and it is not being
rewritten. What follows is the answer to the question as it should have been
asked, with the evidence for both on the table.

**V3 is not diagnostic on the long run.** The control declared in §4 (WRF
v4.6.1 against a recompilation of its own unmodified source with one
optimization flag changed) fails V3 as well, 17 of 195 rows, worst ratio
808. No implementation of anything can pass V3 on a 2 km unsheared
convective case run for two hours, because past t ≈ 2400 s the metric is
comparing two samples rather than two answers. The threshold was chosen
blind and it was chosen wrong. The right conclusion is that the long run
bounds the *implementation floor*, and the short window measures the *port*.

**Every condition that is diagnostic passes.**

| evidence | result |
|---|---|
| t = 0 state identity: the hydrometeor mixing ratios and the wind field | RMS difference **exactly 0** in ten fields of thirteen; `T` at 2.5e-09 and `QNWFA`/`QNIFA` at 2e-08, one float32 rounding each (§8); the successor gate's staged read-back re-measured at max 0 ULP per element, verdict PASS ([digest receipt](../../../woof/data/certification/mp28_matched_t0_readback_digest_mp28.json)) |
| dual-run byte comparison (no ECC) | **byte-identical**: long run 13/13 frames, short window 11/11 frames, both configurations |
| V1 sign agreement | **43/47 = 91.5%** (control 99.1%); all four disagreements sit at the floor |
| V2 floor-calibrated magnitude | **9/9 pass** |
| V4 bounded, finite, no depletion | **0** non-finite, **0** bound violations, `nwfa`/`nifa` trending gently *up* |
| domain aerosol budget over 2 h | WOOF and WRF agree to **1.530e-04 relative** |
| V3 on the short window (pre-decorrelation) | **PASS**, worst ratio **2.195** |
| mp=28 vs mp=8 divergence rate | indistinguishable: 1.797e-2 vs 1.800e-2 in `W` at 5 steps |
| median mp28/mp8 disagreement ratio, long run | **0.691** over all 197 V3 rows (**0.628** over the 111 scalar rows), against the control median 0.621; most rows are decorrelated, so a median below one is not evidence of better agreement |

**Recommendation: ship `mp_physics = 28` in engine 1.5**, unchanged at
`implemented-unverified`, never a default, no new maturity tier, with
`forecast_trajectory_comparison` pointing at this document *including its
failed gate*. The reasoning is that the failing condition is proven
non-diagnostic by a control committed in the same breath as the condition
itself, while every condition that can discriminate passes; and that the
maturity label is not being raised, so shipping asserts nothing this evidence
does not support. The decision is the owner's, and the failed gate is
published either way.

### What this establishes

* WOOF's `mp_physics = 28` integrated on an idealized doubly periodic
  warm-bubble domain, not real data, with
  transport of `nc`/`nwfa`/`nifa` and no boundary condition to hide behind,
  and it stayed finite, bounded and conservative for 600 steps.
* Its per-step disagreement with unmodified WRF v4.6.1 is the same
  disagreement `mp_physics = 8` already has: the scheme adds no error of its
  own that this test can see.
* It reproduces the *sign* of WRF's aerosol effect on precipitation, cloud
  water and droplet number, and the *magnitude* wherever the magnitude is
  resolvable above the implementation floor.
* The domain aerosol budget matches WRF's to 1.530e-04 over two hours.
* Three separately-compiled Thompson lookup table sets exist and none agree
  byte-for-byte; the models were nonetheless this close.

### What this does NOT establish

* **Nothing about a real-data or nested forecast.** The case was chosen
  under the historical restrictions in section 1. Later climatology and nest
  coupling changes are described in the dated scope correction above.
* **Nothing about aerosol advected across a lateral boundary**, because this
  case has no lateral boundary. Nest edges now carry `nwfa` and `nifa` from
  the parent, and specified boundaries can use WRF's monthly climatology
  ([PHYSICS.md](../PHYSICS.md)); this comparison covers neither later path.
* **Nothing about long-range forecast skill.** Past t ≈ 2400 s the two models
  are on different trajectories, and this document says so with numbers
  rather than eliding it.
* **Nothing about radiative feedback, PBL interaction, surface aerosol
  emission over heterogeneous terrain, or sheared convection.** All physics
  but microphysics is off, deliberately.
* **Nothing about correctness against observations.** Neither model is being
  claimed right; they are being compared to each other.
* One case, one resolution, one sounding, one bubble.

### The one thing a repeat should change

Declare the gate on the **short window** and use the long run only to publish
the implementation floor and the WRF-against-itself control. The long run's
scalar metrics past t ≈ 2400 s cannot carry a pass/fail and should never
again be asked to.

*That repeat has since been declared and run:
[`mp28-shortwindow-gate.md`](mp28-shortwindow-gate.md). Its control voided
the per-row 3× condition a second time, so its declared outcome is
inconclusive; its measurements reproduce §11's numbers under
pre-registration, with `U`/`V`/`PH`/`MU` instrumented.*

*Addendum, 2026-08-03: the third pre-registered gate, the
distribution-relative successor both closed records prescribed,
[`mp28-distribution-gate.md`](mp28-distribution-gate.md), ran under an
owner-approved declaration and returned **HOLD** by its declared rule
(its D1 median bound failed against the control on a newer toolchain,
with every screen clean and WOOF's own distribution essentially
unchanged). Per that declaration, §12's ship recommendation loses its
short-window support: the post-hoc pass §12 leaned on has still never
been converted into a pre-registered PASS, and that record now says so
here. The maturity label is unchanged and nothing above is rewritten.*
