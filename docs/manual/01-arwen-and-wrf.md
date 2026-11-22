# 1. What WOOF-ArWen is and how it relates to WRF-ARW

## 1.1 The relationship in one paragraph

WOOF-ArWen integrates a WRF-ARW-class compressible nonhydrostatic core in FP32
on CUDA. WRF-derived schemes use versioned WRF sources; the historical
comparison below uses v4.6.1, while newer ports identify v4.7.1 in their
rows. SASE and the RTE+RRTMGP coupling are original schemes or couplings
without a WRF counterpart. The physics registry records the evidence
status of each option; [the physics page](../public/PHYSICS.md) states
the measured scope. No end-to-end bit identity with WRF is claimed.

Deliberate divergences are recorded in PROVENANCE.md. Their intended
referee is skill against observations. The published observation-battery
receipts contain no scored forecast result for that programme; a physical
or numerical argument for a divergence is not an observation score.

The WRF reference build for the historical July four-domain comparison is v4.6.1 at the pinned
commit, built with GNU gfortran 15.2.0 (WRF `configure` option 34, dmpar; Intel
oneAPI supplies the MPI layer only), run on 48 MPI ranks against a single RTX 5090
GPU run [docs/public/VERIFICATION.md, historical comparison and limits].

## 1.2 What is kept from WRF-ARW

The core uses a WRF-style hybrid terrain-following, dry-mass vertical coordinate.
Prognostic state includes staggered horizontal wind, vertical velocity, perturbation
potential temperature, dry mass, geopotential, water vapor, and hydrometeors. A
three-stage Runge-Kutta outer integrator wraps split-explicit acoustic steps:
forward-backward horizontal acoustic updates, implicit vertical treatment, and
recoupling to the large step. Parent timesteps divide exactly down the nest tree
[docs/gpuwm-project-history.md:65].

The damping and stability stack is deliberately WRF-shaped rather than a generic
Laplacian safety net: `epssm` off-centering, external-mode divergence damping,
`smdiv`, sixth-order diffusion, Smagorinsky mixing, positive-definite scalar
transport, optional vertical-velocity damping, and the `damp_opt=3` upper sponge
applied to `w` rather than to every state. Lateral boundaries use
specified/relaxation zones with Davies-style weighting. Slow physics tendencies are
held through RK stages in WRF order; microphysics runs after the final stage
[docs/gpuwm-project-history.md:67]. Two solver details that WRF users know as
correctness traps are tracked as verification subjects in their own right: the
dry-mass flux accumulator `mudf` lifecycle across RK stages, and WRF's open-top and
moist pressure-coupling factors `cqu`/`cqv`/`cqw` [docs/gpuwm-project-history.md:71].

WRF's namelist is a first-class input: a WRF namelist imports through the
preprocessing front door, and options WOOF-ArWen does not implement are refused by name
rather than silently substituted (chapter 5).

## 1.3 The maturity ladder

The registry's canonical strings are `wrf-matched-run`,
`wrf-matched-run-candidate`, `supported`, `experimental-runtime`,
`implemented-unverified`, and `planned`. These describe WRF-conformance
evidence or its absence. None is validation against observations.
Historical spellings remain readable aliases; new selections use the
canonical names.

- **wrf-matched-run:** a historical multi-hour comparison against WRF
  exists for a named build and option set, with published tables. It
  does not record a pass against a calibrated tolerance. The July case
  failed its initial-state digest, and later kernel changes are not
  measured by that run.
- **wrf-matched-run-candidate:** executable with component checks and a
  reference comparison accepted by the project; a candidate for a full
  matched WRF comparison, not an observation-validation result.
- **supported:** an executable option exercised by the named runtime
  configurations, WRF-derived where a WRF counterpart exists. The label
  is not an accuracy guarantee.
- **experimental-runtime:** executable with a documented runtime
  restriction or composition warning.
- **implemented-unverified:** executable with the evidence its own row
  states. Some options have WRF column comparisons; MYJ, WDM6,
  Milbrandt-Yau and RRTM 1/1 have none. SASE has no WRF counterpart but
  can still be checked against its own equations and reference code.
- **planned / port-in-progress:** not selectable.

The current registry has 45 component options, 27 labelled
`implemented-unverified`. Most have component-level evidence rather than a matched forecast
comparison; an independent Fortran oracle is not implied by the label
`implemented-unverified`. See the current per-option inventory in
[PHYSICS.md](../public/PHYSICS.md#maturity-vocabulary).

The composition rule C2 takes the lowest component rung. The registry
also carries explicit exemptions; a template label granted by an exemption
does not establish a matched run of that exact tuple. In particular, the
historical Thompson comparison used legacy RRTMG and does not cover the
default RTE+RRTMGP radiation tuple. Read the exemption's scope, not just
the label.

`woof certify` checks a run capsule against the named acceptance band.
The historical `documented-margin` band is a margin around one old WRF
comparison table. Passing it is a regression check, not evidence of
statistical indistinguishability or forecast accuracy. A band calibrated
from reference-ensemble spread is a separate instrument; the current
[ensemble evidence](../public/receipts/wrf-consistency-20261002/SUMMARY.md)
does not silently replace that historical band. The source-admission
and component-oracle uses of certification elsewhere are identified by
their actual checks.

## 1.4 The intended observational standard for judging divergence

Validation asks how well forecasts represent the atmosphere, using
observations. Meteorology calls that scoring forecast verification; this
manual uses the computational-science terms defined in
[Verification and validation](../public/VERIFICATION.md#verification-and-validation).

The intended broad observation programme uses MRMS for reflectivity and
ASOS-class stations for near-surface state. Its archive and scoring tools
exist, but its published battery receipts contain no scored forecast
result. A multi-radar composite used as an input is a feed-space
diagnostic, not an independent grader. Separate limited case scores are
listed on the evidence page; they do not supply a broad validation record.

Matching WRF remains code verification. WRF's published validation record
can transfer only to the extent the models are statistically
indistinguishable for the configuration and quantities in question.
The old deterministic matched run does not establish that condition.
The recent WRF ensemble comparisons are the appropriate fidelity
instrument; their failures and coverage limits must be retained.

## 1.5 The divergence ledger

Deliberate runtime deviations from WRF v4.6.1 are numbered D1-D12 in
`PROVENANCE.md:250`. The standing rule, stated in the physics page:
implement the defined behaviour and document the divergence rather than reproduce an
undefined read, refusing with the Fortran line named where reproduction is refused
[docs/public/PHYSICS.md:367-371].

| id | subject |
|---|---|
| D1 | retired compatibility-mode microphysics / `h_diabatic` cadence |
| D2 | `REFL_10CM` computed on output-due microphysics steps only, where WRF's `nwp_diagnostics == 1` sets `diag_flag` every step |
| D3 | experiment-schema fail-loud rejections |
| D4 | integer-tick clock: WRF-recurrent `dtbc` / running seconds, exact-calendar scope |
| D5 | nest-interpolation (SINT) geometry precomputed FP64-on-host, stored FP32; bitwise-identical to WRF's per-op REAL construction for refinement ratios 1-4, proven in `tests/test_nest_interp.py`; diverges by exactly 1 ULP at ratio 5. Only geometry is precomputed; flux and limiter arithmetic is evaluated per field at force time |
| D6 | `adjust_tempqv` evaluates FP64 on device, stores FP32; the temperature chain cancels ~275 K of magnitude and the Magnus exponential amplifies the residue, so a REAL-internal kernel is irreducibly hundreds of ULPs from any FP64 mirror in qv |
| D7 | retired in 2.8.6: WRF's `seaice_noah` runs after Noah on sea-ice columns, graded against a WRF v4.6.1 column oracle |
| D8 | the vertical diffusion of vertical velocity (`vertical_diffusion_w_2`) takes the vertical momentum exchange coefficient where WRF hands it the horizontal one; taking WRF's is unstable at dx 250 m against dz 17 m, and the WRF oracle lane measured the two coefficients equal at all 589,824 points on the idealized case where it can matter (section 2.7) |
| D9 | Thompson aerosol-aware (mp=28) admission, aerosol ingest, lateral boundaries, PBL mixing |
| D10 | LES-nest inflow perturbation, a designed WOOF-over-WRF extension |
| D11 | RUC LSM returns the defined one-layer answer where WRF reads an uninitialised `ilnb` on thin snow, a real WRF defect in which the value depends on grid traversal order |
| D12 | configured initial-state theta bubbles (`[perturbation]`), an WOOF-over-WRF extension |

[PROVENANCE.md:250-1345]

Where WRF's own arithmetic is undefined, each case is decided on its consumers and
published rather than hidden: P3's first-step 0/0 supersaturation is floored to
exactly -1 (fully subsaturated, the intended meaning; default-on in all three arms,
inert from step 2 onward, the step-1 delta declared)
[docs/public/PHYSICS.md:256-264], while Shin-Hong reproduces WRF's own
`prfac2 = 0/0` NaN (every consumer tolerates it) and deliberately does not
perform WRF's `q2xk(kpbl+1)` out-of-bounds read [docs/public/PHYSICS.md:876-878].

Two schema extensions leave WRF-expressible territory on purpose and are named as
such: per-domain `isfflx` (a scalar in the WRF Registry, per-domain in WOOF-ArWen's TOML,
so such a configuration cannot round-trip back to a namelist)
[docs/public/LES.md:357-362], and per-domain radiation-cadence inheritance, which is
WRF's own namelist guidance made the default rather than a per-nest derivation
(section 3.9).

## 1.6 The arbitrary acceptance test

A design law shapes the input system: adding a future model must be metadata and
table work, not a new code path; a per-model adapter file fails the test. The 2.5.0
initialization engine (chapter 5) is built to that law: the domain wizard reads
cadence, horizon, and coverage off a registry row and names no model in code; a
synthetic registry row drives the real CLI in `tests/test_wizard_sources.py` with no
code added anywhere. Where a per-model fact is genuinely a correctness fact (an
upstream bucket publishing a different product under identical object keys, chapter
5), it is carried as table data, not as a code branch.
