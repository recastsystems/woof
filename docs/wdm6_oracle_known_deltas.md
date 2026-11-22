# WDM6 (mp_physics=16): the column oracle and its documented deltas

Read this before quoting any ULP or relative-agreement number for mp=16.

**Status of the scheme:** implemented-unverified in the registry (no
forecast-trajectory comparison against WRF exists), and graded word for word
by a column oracle. `tools/wdm6_wrf461_oracle` drives WRF v4.6.1's own `wdm6`
entry of the byte-frozen `phys/module_mp_wdm6.F` (sha256 `78b7c148...`) at
gfortran 13.3 `-O0` (no FMA, scalar libm, glibc 2.39) over 84 columns in seven
configurations, and records two Fortran arms:

* **stock**: the unmodified file;
* **defined**: the same file with exactly the three defined behaviours below
  (sections 2, 5 and 6) substituted by `tools/wdm6_wrf461_oracle/defined_arm.py`.

Measured on an RTX PRO 6000 (NVRTC 12.9, CuPy 14.2) on 2026-10-07: under the
strict build (`GPUWM_WRF_EXACT=1`) every output word of the CUDA kernel --
theta, the six masses, nn/nc/nr, the three effective radii, `refl_10cm` and
the seven surface accumulators, every step of every configuration -- equals
the defined arm (`tests/test_wdm6_wrf461_oracle.py`). Against the stock arm
the kernel is bitwise in the 21 to 29 columns (of 84) where none of those
behaviours fires, and differs only where section 5 or section 6 fires; the
single-section arms (`run_wdm6_only{2,5,6}`) attribute every differing column.
The default build (no `GPUWM_WRF_EXACT`) compiles the unit without
multiply-add contraction too; what remains is the loader's flush-to-zero route
(section 8).

The fixture carries both `hail_opt` arms and both `xland` arms (land columns,
ocean columns with `xland == 2`, and one just below 2): WDM6 is the first woof
microphysics whose PROCESS RATES read a surface field
(`module_mp_wdm6.F:607-614`), so a land/sea mask that differed between the two
sides would be a trajectory difference with no other symptom.

The sections below were written before the campaign; each now carries what
the oracle measured.

The physics registry's `wdm6-mp16` warnings cite this file.

---

## 1. `_rgmma` runs in float64 where the Fortran runs in REAL(4) -- MEASURED, then CLOSED

**Closed by the oracle campaign (lane/verify-wdm6-mp16).** The floor was real:
the oracle's own `wdm6init` dump (`tests/data/oracles/wdm6/wdm6init-hail*.txt`)
put `pvtr` at 2998.4763 against the float64-derived 2998.4927 (67 ULP),
`pvtrn` 1410.880 against 1410.8845, `g4pbr`, `g7pbr`, `pvts`, `pvtg`,
`precr2`, `precs2`, `precg2`, `pacrc`, `pacrg` and `cdm2` likewise, and
single-rounding products off by one ULP as well (`pidnr = 4*pi*denr`,
`pidnc`, `pidn0g`, `precr1`, the `rslope*max` squares and cubes). The kernel
now bakes wdm6init's own float32 words (`kernels/wdm6.cu` `WDM6_*`,
`woof/core/wdm6_constants.py` `WDM6_WRF_WORDS`), pinned by
`tests/test_wdm6_wrf461_oracle.py`. That is not bit-exactness to a defect:
`wdm6init` declares these coefficients default REAL, so float32 arithmetic is
the definition, and the float64 evaluation of the same truncated product is
not more right in any physical sense -- the 10,000-term truncation it shares
(0.1 per cent) is two orders of magnitude larger than the 5e-6 accumulation
difference. The float64 derivation stays in `coefficients()` as the readable
formula, checked against the words to 1e-5.

The analysis below is the pre-campaign text, kept for the record.

`woof/core/wsm6_constants.py::_rgmma` is WRF's own truncated Weierstrass
product, ported verbatim, and WDM6's coefficient block reuses it
(`woof/core/wdm6_constants.py`). Getting the *truncation* right is the thing
that matters and the port has it: `wdm6init`'s inline comment at
`module_mp_wdm6.F:2130` claims `g4pbr = 17.837825` (the true `Gamma(4.8)`),
while the code's own 10000-term product returns 17.8173. Porting to the comment
would have been a 0.1-0.7% error in every fall speed. The port reproduces the
code.

What is NOT reproduced is the arithmetic width. The Fortran accumulates the
product in default `REAL` (single precision); woof accumulates in Python
float64 and only then rounds to FP32. With 10^4 sequential multiply-accumulates
the accumulated single-precision rounding can reach the 5th-6th significant
digit, so every derived coefficient can differ in its last FP32 digits -- and
these coefficients multiply *every* fall speed and collection rate.

Consequence for the campaign: a per-column ULP comparison against a
`gfortran -O0` oracle will show a nonzero floor that is NOT a transcription
error. Establish that floor first, by driving the oracle's own `wdm6init` and
diffing its coefficients against `woof/core/wdm6_constants.py`, before
attributing any process-rate difference to the process.

This is inherited from WSM6, which shares `_rgmma`. It is deliberate: float64 is
the defensible arithmetic, and WOOF-ArWen's standing rule is not to be bit-exact to a
rounding artifact.

## 2. The PLM remap's `kt` clamp -- REAL divergence, believed unreachable

`woof/core/kernels/wdm6.cu` clamps the remap top index (`kt = (kt > 0) ? kt - 1
: 0`) where `nislfv_rain_plmr:2629` and `nislfv_rain_plm6:2891` decrement
unconditionally. On the one input where they differ, the Fortran drops a
layer's mass and the port conserves it. The port takes the defined behaviour, as
the standing rule requires.

Believed unreachable for `nz >= 2`, because the `con1 = 0.05` limiter forces
`dza >= 0.95*dz`, and `nz = 1` is below the adapter's floor of 2. "Believed" is
an argument, not a measurement. If a column oracle ever shows a single-layer
mass difference in sedimentation with everything else matching, this is it. The
full reasoning is in a comment at the site.

**Measured:** the oracle's `only2` arm (the stock file with only this clamp
substituted) is bitwise equal to the stock arm on every word of all seven
configurations, so the clamp never fired on the fixture. It remains a
defined-behaviour guard, not an observed difference.

## 3. The rain-slope density inconsistency -- REAL, and it is WRF's

`wdm62D` consumes `ncr` as a VOLUMETRIC number (`:2251`) while `refl10cm_wdm6`
forms `nr = nr1d*rho` against `rr = qr1d*rho`, so the densities cancel and
`nr1d` enters as a PER-MASS number (`:3005`). The two rain slope definitions
differ by a factor `rho` inside a cube root, about 3% in `lamr` near the
surface. WRF does this; woof reproduces both as written. An oracle will
therefore agree -- this is listed so nobody "fixes" it into a disagreement.

## 4. The -35 dBZ floor is NOT a delta -- checked, and the suspicion is wrong

Recorded because it looks like one on a partial read, and a reviewer raised it.

`refl10cm_wdm6` initialises `dBZ(k) = -35.0` and then, in its final loop
(`:3126-3128`), overwrites every level unconditionally with
`10.*log10((ze_rain+ze_snow+ze_graupel)*1.d18)`. With the `1.e-22` floor on each
of the three `ze` accumulators, a hydrometeor-free column leaves the routine at
`10*log10(3e-22*1e18) = -35.23` dBZ, not -35.0. So the ROUTINE's floor really is
-35.23.

But the routine is not the boundary. The caller stores
`refl_10cm(i,k,j) = max(-35., dBZ(k))` (`:294`), and woof applies exactly that
clamp inside its kernel (`fmaxf(-35.0f, ...)`). The published field is identical
on both sides, including in hydrometeor-free columns. There is no 0.23 dB
inheritance.

Compare against `refl_10cm`, the field WRF publishes -- not against `dBZ`, the
routine's intermediate -- and this reads as agreement.

---

## 5. Conservative rain interfaces replace an inherited loss

The rain update in unmodified WRF v4.6.1 `module_mp_wdm6.F:810-839`
caps an arrival by the upstream cell's remaining state after that cell
has already lost its outgoing flux. The earlier CUDA transcription did
the same. It is not a translation error: both versions can discard the
difference between the amount removed and the smaller amount received.

The corrected `wdm6_rain_substep` transfers a single amount across each
interface. Its mass inventory is `rho * dz * qr` in kg/m2 and its number
inventory is `dz * nr` in 1/m2. An outgoing flux is limited by that donor's
inventory once; exactly that amount enters the next cell or the recorded
surface fallout. Incoming flux is never capped by the donor's remaining
state. The column sums therefore telescope, including unequal layer
depths and densities. There is no state clipping or column renormalization.
The existing separate mass and number fall speeds and their substep
refresh remain unchanged.

A retained 40-level, 2km warm column with a 1g/kg rain slab activates the
defect through production physics at 25, 60 and 120 second physics steps;
the existing mixed-phase fixture does not. The production regression is
`tests/test_wdm6_column.py::test_warm_rain_conserves_water_through_production_microphysics`.
`tests/test_wdm6_sedimentation.py` compiles the same substep for independent
mass/number budgets, true surface-flux controls and unequal density/depth.
Full microphysics has physical number sources and sinks, so number
conservation is asserted for sedimentation, not for the entire scheme.

This changes the numerical trajectory deliberately. The checkpoint
algorithm identity advances to v3 so an older checkpoint cannot silently
continue under the changed rain update. A WRF comparison should report
this conservation difference rather than reinstate the defective arrival
cap to obtain agreement. No other microphysics scheme is changed.

### The substep schedule covers newly wet receiving layers

A count derived from initial rain speeds does not bound refreshed speeds.
In a two-layer column with depths 1m and 100m and an initially dry lower
layer, it permits a later terminal Courant number above 100 while donor
limiting still conserves water. The arrival profile then depends strongly
on the outer interval. Conservation alone does not establish time accuracy.

The inverse rain slope is bounded by the existing 1e-3m PSD cap. Therefore
`2998.49272 * (1e-3)**0.8 * sqrt(1.28/rho)` bounds the mass fall speed for
every layer, including a dry receiver. The number speed is smaller by
the existing coefficient ratio 0.47053124. The new count uses this bound
and layer depth, so its terminal Courant stays below 0.4 throughout the
interval. It retains the existing slope refresh and conservative transfer.

For the active smooth PSD branch, write `p=0.8/3` and `b=0.47053124`.
The mass/number flux Jacobian divided by the mass speed has trace
`1+p+b*(1-p)` and determinant `b`; its larger eigenvalue is about 1.228805.
The 1.25 speed multiplier and 0.5 wave Courant give the 0.4 terminal bound.
The capped branch has constant speed. The inherited small-moment branch
switches remain unchanged; this calculation is not a smoothness claim
across those thresholds. The general wave-speed basis is described in the
[Clawpack time-step documentation](https://www.clawpack.org/dev/setrun.html).

The schedule stays first order in time. Tests separately check evolving
Courants, final profiles against a refined simultaneous flux calculation,
and convergence of the first arrival under actual interval refinement.
They cover both initially wet and dry unequal layers. The geometric
speed envelope also bounds the first dry-receiver step, where a budget
based only on current wet-cell rates can be coarse. It is not a universal
seconds cap, a state clip, a forecast-skill result or a complete WRF oracle.
The v3 checkpoint identity distinguishes this schedule from the interim
v2 conservative transfer with initial-speed substeps.

## 6. Rain with no number no longer evaporates through the condensation cap -- REAL, and it is WRF's

`module_mp_wdm6.F:1240-1256` computes the rain evaporation/condensation rate as
`prevp = (rh-1)*nr*(...)/work1`. A negative rate takes the evaporation branch,
bounded by the rain present (`max(prevp,-qr/dtcld)`, :1243) and by half the
saturation deficit. Any other rate takes the condensation branch,
`prevp = min(prevp, satdt/2)` (:1255), written for supersaturated air.

Rain that carries mass but no number (`nr = 0`, `qr > 0`) makes the rate
`-0` in subsaturated air. `-0 < 0` is false, so WRF takes the condensation
branch, and `min(-0, satdt/2)` with `satdt < 0` becomes an evaporation of half
the saturation deficit. Nothing bounds it by the rain: on the captured cell it
was 1.18e-5 kg/kg of evaporation from 1.3e-13 kg/kg of rain. `pidep`, `psdep`,
`pgdep` and `pigen` subtract `prevp` from `satdt` in their `supice` limits
(:1527 on), so ice deposits against that vapour. The rain limiter (:1670-1680)
then scales `prevp` back to the rain there is and never revisits the
deposition. Vapour goes negative.

Numberless rain reaches the process block from advection (mass and number are
transported separately), from melting snow or graupel below `qcrmin`, which
adds mass without number (:917-920, :940-943), and from freezing that consumes
the number before the mass.

WRF v4.6.1's own Fortran (the pinned source, driver and radar code removed,
`gfortran -O0 -ffp-contract=off`) was run on the two columns where both WDM6
convective cases failed the vapour check at step 4 (HRRR 2026-06-14 19Z,
West Texas, 3 km, 49 levels, dt 15 s; retained as
`tests/data/wdm6_numberless_rain_columns.npz`):

| column | WRF v4.6.1 | port before | port now = WRF with this rule |
|---|---|---|---|
| Grell-Freitas, cell (38,1,64) | qv -4.9046e-9; 9 levels negative, down to -9.88e-7 | the same bits | qv 1.0327e-6; minimum 1.30e-8 |
| no cumulus, cell (40,47,215) | 2 levels negative, down to -7.82e-7 | 8 levels negative, down to -8.74e-7 | qv 2.5168e-7; minimum 5.8e-17 |

The second column differs from WRF before the fix because WRF's lossy rain
transport (section 5) drops the tiny rain at that cell and the conservative
transfer keeps it; with the rule below both are nonnegative, and they agree
at the failing cell.

**The defined behaviour.** The condensation cap keeps its meaning and never
changes the rate's sign: `prevp = max(min(prevp, satdt/2), 0)`. It differs
from WRF only where the rate is exactly zero in subsaturated air; the
condensation branch gives the same bits wherever the rate is positive, since
a positive rate means `rh > 1` and so `satdt > 0`. Numberless rain then
neither evaporates through `prevp` nor feeds deposition. It is still removed
within the call: at `nr <= nrmin` the rain slope takes its maximum `lamdar`,
the mean diameter is below `di82`, and the rain joins the cloud water in the
collapse at :1938, where `pcond` evaporates it within its own `qc` bound.

What changes, per call, where it applies: vapour, cloud ice, snow, graupel and
theta in the cells where WRF would have deposited against the fictitious
evaporation (on the Grell-Freitas column above: levels 32 to 46, qv and qi up
to 5.5e-6 kg/kg, theta up to 0.024 K), and the tiny rain mass itself.

Measured in forecasts (the same case, 220 x 176 x 49, 1 h, RTX 5070 Ti; every
WDM6 call also ran the previous kernel on a copy of its inputs, a check that
found zero differences when both sides ran the shipped source): the first call
carries numberless rain in every column, because WDM6's cold start seeds no
rain number (as real.exe does not), and every later call carries it in all
3,860 columns of the 5-cell lateral boundary zone, where the hydrometeor edges
bring rain mass without number, plus about 160 interior columns. There the
previous kernel moved theta by up to 0.81 K and vapour, ice, snow and graupel
by up to 2.2e-4 kg/kg in one call, and left vapour negative in every one of
the 240 calls; the shipped kernel's smallest vapour over the hour was 5.5e-7.
A 1 h WDM6 forecast that passed before (Pacific Northwest, same date) still
passes; its 15-minute frames differ everywhere, up to 0.64 K in T and 6.7e-4
kg/kg in QVAPOR, the largest mostly along the domain edges.
`tests/test_wdm6_numberless_rain.py` compiles the kernel with WRF's statement
restored beside the shipped one: the captured and synthetic numberless
columns go negative with WRF's and stay nonnegative with the shipped one, and
rain that carries number gives identical bytes from both.

The same `-0` pattern exists in the ice deposition branches (`pidep`, `psdep`,
`pgdep`): their rates carry no number factor, so they are exactly zero only at
`rh = 1`, where `satdt` is zero up to rounding, or when a vanishing ice mass
underflows the product. A sign flip there is sublimation, which moves vapour
upward and is bounded afterwards by the ice limiter; it never takes vapour
below zero. They are left as WRF has them.

The checkpoint algorithm identity advances to v4.

**Measured footprint of sections 5 and 6 against stock WRF** (the oracle's
`only5` and `only6` arms against `stock`; one call unless stated):

| configuration | section 5 columns | section 6 columns | largest theta difference | largest qv difference |
|---|---|---|---|---|
| dt 20 s, hail_opt 0 | 54 of 84 | 38 of 84 | 0.41 K | 1.0e-4 kg/kg |
| dt 300 s (3 minor loops) | 57 | 38 | 0.40 K | 1.2e-4 kg/kg |
| dt 60 s, 10 calls | 58 | 39 | 0.68 K | 1.1e-4 kg/kg |
| dt 150 s, hail_opt 1, 3 calls | 59 | 36 | 1.06 K | 3.8e-4 kg/kg |

Section 5 moves the rain accumulator by up to 8.9 mm over ten 60 s calls in
the heaviest column; section 6 fires wherever rain mass carries no number,
which the fixture's Gaussian tails supply in most rainy columns.

**Section 5 split in two, and judged against a converged reference**
(`tools/wdm6_wrf461_oracle/measure_split.sh`, `split_arm.py`,
`convergence.py`). Section 5 changes two things: the transfer (the loss fix)
and the sub-step count (WOOF's geometric count instead of WRF's count from
the initial speeds). The `only5t` arm keeps the conservative transfer with
WRF's own count; `fine8` and `fine32` run section 5 with 8 and 32 times
WOOF's count. Measured 2026-10-07 on W1:

| configuration | transfer alone (stock vs only5t): words / columns | count alone (only5t vs only5): words / columns |
|---|---|---|
| dt 20 s, hail_opt 0 | 2,384 / 53 | 10,593 / 54 |
| dt 300 s | 5,363 / 54 | 12,038 / 57 |
| dt 150 s, hail_opt 1, 3 calls | 15,386 / 59 | 37,377 / 58 |
| dt 60 s, 10 calls | 53,012 / 55 | 121,533 / 58 |
| 72 levels, dt 30 s, 2 calls | 9,035 / 60 | 30,386 / 63 |

So most of section 5's word count against stock WRF comes from the count,
not the loss fix. The count is not arbitrary: mean last-step surface-rain
distance to the `fine32` reference (every arm differs from it only in the
rain sedimentation loop):

| configuration | stock WRF | transfer + WRF count | transfer + WOOF count (the kernel) | fine8 |
|---|---|---|---|---|
| dt 20 s, 1 call | 0.0018 mm | 0.0017 mm | 0.00047 mm | 0.000045 mm |
| dt 300 s, 1 call | 0.045 mm | 0.015 mm | 0.0033 mm | 0.00026 mm |
| dt 150 s, 3 calls | 0.078 mm | 0.037 mm | 0.0057 mm | 0.00050 mm |
| 72 levels, dt 30 s, 2 calls | 0.0088 mm | 0.0062 mm | 0.0013 mm | 0.00013 mm |

WOOF's count is four to fourteen times closer to the converged answer than
WRF's count, with or without WRF's loss. WRF's count with the conservative
transfer is the worst combination in the heaviest columns (conv_land_7 at
dt 150 s: 1.16 mm off; theta up to 1.86 K off). Over ten 60 s calls the column
physics feeds back and even `fine8` and `fine32` separate (3.8 mm in the
heaviest column), so that row is not a convergence measurement and is left
out of the table.

## 7. The 10 cm reflectivity is now WRF's word too -- the radar words were not

`refl10cm_wdm6` (and the radar module's melting-particle routines) are
evaluated in the TYPES the Fortran declares: temperature, density, the rain,
snow and graupel contents and the three `ze` arrays are default REAL; the
slopes and intercepts are DOUBLE PRECISION, with `lamr`'s right-hand side all
REAL. The radar constants are WRF's own words
(`woof.core.refl.wdm6_radar_words`), which the float64 `radar_init` mirror
does not reproduce: radar_init's `xcrg(4) = WGAMMA(8.)` is 5040.002 and
`xcsg(4)` 720.00006 (REAL `exp(gammln)`), `PI5` is a product of REAL
literals (306.018463 against 306.018392), `lamda_radar = 0.10` is a REAL
literal held in a DOUBLE PRECISION parameter (so `lamda4` is 1.00000006e-4),
and `m_w_0` is the square root of a default single-precision COMPLEX. Complex
division follows gfortran's Fortran-rules lowering (Smith's ratio form).
LOG10, ATAN2 and HYPOT stay the device library's binary64 functions; the
oracle measured no output word they moved. Morrison's and WSM6's
reflectivity (`kernels/refl.cu`) still use the float64 radar mirror; they are
outside this campaign.

## 8. Default build: flush-to-zero only, where subnormal values enter a call

The unit compiles with `--fmad=false` in every build (`_NO_FMAD_MODULES`),
so the strict and default builds differ only in the loader's flush-to-zero
route (docs/manual/04-gpu-numerics.md, 4.1). Measured: every default-build
difference in the STATE (theta, masses, numbers, radii, reflectivity) lies in
a column into which a subnormal moisture value enters a call, from the input
or from the scheme's own output of an earlier step (17 to 22 of 84 columns;
32 of 84 carry a subnormal input); WRF keeps those values, for instance
growing snow by deposition from a 1e-40 seed, and the default build treats
them as zero. Outside those columns, two columns in each of the two
multi-call configurations lose a vanishing surface fallout (under 5e-34 mm)
whose own product underflowed. Largest differences over
ten 60 s calls: theta 9.2e-4 K, qs 4.8e-7 kg/kg, refl_10cm 0.26 dBZ. The
strict build has no such difference.

For scale, stock WRF against itself: the unmodified file compiled at WRF's
own GNU physics flags (`-O2 -ftree-vectorize -funroll-loops`, which call
glibc's libmvec `expf`, `logf` and `log10` in vectorised loops;
`tools/wdm6_wrf461_oracle/measure_o2.sh`) differs from the `-O0` scalar
reference in 59 to 77 of the 84 columns, by up to 1.0e-3 K in theta,
3.1e-7 kg/kg in qv and 0.085 dBZ in refl_10cm (measured 2026-10-07). In theta and the
masses the default build's flush-to-zero difference is of the same order as
the spread of WRF's own production build around its reference, but it is
concentrated in fewer columns.

## The oracle, and how to rerun it

`tools/wdm6_wrf461_oracle/README.md`.
