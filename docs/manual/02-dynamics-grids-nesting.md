# 2. Dynamics, grids, nesting, and LES

## 2.1 Dynamical core

The core is a WRF-ARW-class compressible nonhydrostatic solver: three-stage
Runge-Kutta outer integration wrapping split-explicit acoustic steps
(forward-backward horizontal, implicit vertical, recoupled to the large step), on a
hybrid terrain-following dry-mass vertical coordinate, FP32 on CUDA
[docs/gpuwm-project-history.md:65; README.md, current release scope]. The RK stage table is a
config-visible knob (`rk_ord`, default 3) [docs/public/CONFIGURATION.md:983].

Advection is WRF's stencils: horizontal momentum and scalars are the WRF flux5
(5th-order) stencil, and the vertical faces take WRF's `vert_order` ladder,
3 (flux3 between 2nd-order faces) or 5 (flux3 two faces in, flux5 between),
selected by `v_sca_adv_order` for scalars, theta, TKE and w and by
`v_mom_adv_order` for u and v (`woof/core/kernels/advection.cu`; both default 3,
operational HRRR runs 5). Transported-scalar horizontal stencils are fixed at
5th order, so the importer accepts only the Registry default
`h_sca_adv_order = 5`; the configurable `h_sca_adv_order` (legacy default 2) feeds
the geopotential equation only [docs/public/CONFIGURATION.md:525]. Moist transport
runs WRF option 1 (positive-definite limiter) with `scalar_adv_opt` required to
match [docs/public/CONFIGURATION.md:529, 964]. At vertical order 5 its
low-order eta flux takes the operational fork's semi-Lagrangian sum of the
upstream cells above face Courant 1 (`woof/core/kernels/pd_vertical_sl.cu`)
and the upwind flux elsewhere, so the limiter stays positive definite.
The fork itself takes the downstream cell at face Courant numbers at most 1,
which drains an empty cell; the final-stage clamp then adds that mass back
as new scalar. Only the strict WRF verification build keeps the fork's
choice, to measure the rest of the transcription against it. Order 3 has
the upwind low-order flux at every face.

Lateral boundaries use specified/relaxation zones with Davies-style weighting;
`spec_bdy_width` defaults to 5 and must be at least `spec_zone + relax_zone`
[docs/public/CONFIGURATION.md:144]. The damping stack is described in section 1.2.

No symbolic statement of the governing equation set exists in the documentation
tree; the prose description above and the WRF-ARW technical-note lineage are the
reference. This is recorded as a documentation gap, not a claim.

## 2.2 Vertical coordinate and levels

`hybrid_opt` supports 0/1 (sigma, `B(eta)=eta`) and 2 (WRF cubic-B hybrid); anything
else is refused by name. The importer and the domain wizard default to 2 with
`etac = 0.2` [docs/public/CONFIGURATION.md:184-185; woof/namelist_import.py:3011-3012;
woof/domain_wizard.py:867]. That `etac` is the value asked for, not always the
value run: at `hybrid_opt = 2` the cubic orders a column only while its surface
pressure stays above a floor set by `etac` and `p_top` (46408 Pa, about 6082 m of
terrain, at the shipped pair), and WRF calls a column below it fatal. Preparation
surveys every terrain field the run can touch -- each declared domain at its own
resolution, and a following nest's whole statics corridor -- and lowers `etac` to
the largest value that orders the lowest surface pressure it finds
[woof/vertical_adaptation.py]. `p_top` is untouched. The chosen value is printed
in one line, recorded in `proof.json` under `vertical_coordinate`, and carried by
the prepared inputs the forecast integrates; every domain of the run takes that
one coordinate, including a streamed nest rebuilding its tile buffer and a nest
spawned mid-run, and a tree holding two is refused by name; terrain no positive
`etac` orders is still refused with the constraint and the column.

A preparation with nothing to derive still writes the block, at status
`NOT_APPLICABLE`, and its `why` names which of the three reasons applied: no eta
ladder is configured, `hybrid_opt` is an identity option (0 or 1), or the
configured cubic has no segment where `dB/deta` rises above 1 and so leaves
its surface-pressure floor at zero [woof/vertical_adaptation.py].

Eta levels are explicit, not generated: `eta_levels` is required for real runs,
automatic level generation (`auto_levels_opt`, `max_dz`, `dzbot`,
`dzstretch_s/u`) is not implemented, and with explicit `eta_levels` those keys
are inert in WRF too, so they import as dropped. `p_top` defaults on import to
the Registry's 5000 Pa [docs/public/CONFIGURATION.md:183].

Two hard properties a WRF user must plan around:

- **No vertical nesting in a LIVE tree, by construction.** The vertical grid of an
  inline nest tree is single-sourced from `ExperimentConfig.vertical` and per-domain
  vertical keys are rejected outright: a 250 m LES child inside a running tree takes
  its parents' level count [docs/public/LES.md:341-345]. The consequences for sub-km
  work are measured in section 2.8. **An OFFLINE child is not bound by this.**
  `woof downscale --child-levels N,STRETCH` builds a standalone child on its own
  eta ladder off archived parent history, through a conservative vertical remap
  (`woof/vertical_remap.py`): the child's initial state and its whole lateral
  boundary table set are rebinned once, at preparation, on the host, and the child
  then integrates normally. `p_top`, `hybrid_opt` and `etac` stay shared with the
  parent -- that is what makes the two ladders span the same column with coincident
  endpoints, so the remap conserves dry mass and every water substance to roundoff
  (measured: water 1.2e-16 relative, potential temperature 0.0, column dry mass
  closing to 0.0 Pa, and the child's model top landing on the parent's exactly).
  A child that names no ladder inherits its parent's, bitwise as before.
- **`nz <= 128` is what has been run; the solver now admits 256.** The acoustic
  solver's per-thread stack column bound is compiled from a tier ladder
  (129/193/257) chosen by `nz`; above 256 the host raises before any launch, a loud
  refusal rather than a skipped kernel. Nothing above 128 has a run receipt yet, and
  Kain-Fritsch's own bound still refuses `cu_physics=1` above 128
  [docs/public/LES.md:327-333]. The same ceiling binds every PBL scheme except
  MYNN: YSU, Shin-Hong, MYJ and SASE hold one column per thread at a compiled
  `KMAX = 128`, and `woof check` names that refusal before a run starts.

## 2.3 Projections

Implemented: Lambert conformal northern hemisphere (projection-oracle checks and one historical matched WRF case; code verification only), Lambert
conformal SH, Mercator, and polar stereographic at either pole (the latter three
implemented-unverified: binary64 oracle plus GPU smoke). Latitude-longitude and
rotated grids are not implemented, refused at load, never substituted. Domains
containing or touching a pole, and forcing footprints wider than 180 degrees of
longitude, are refused for every projection [docs/public/PHYSICS.md:1184-1194]. The
projection oracle pins per-quantity ULP ceilings against unmodified WRF v4.6.1
`share/module_llxy.F` [tests/test_projection_oracle.py].

## 2.4 Static one-way nesting

Static one-way nesting is the supported default. Children may start later on an
exact parent-step and forcing-cadence seam. Two-way feedback (`feedback = 1`) ships
as an experimental path: it runs, it is stamped as experimental in the run's own
provenance, and one-way consumers refuse a feedback-modified parent. It feeds back
dynamic state only, where WRF also feeds back hundreds of masked land-surface
fields, so it is not a WRF-equivalent claim [README.md, current release scope;
docs/public/CONFIGURATION.md:140-141]. `smooth_option` admits 0 only (the parent
smoother acts only under two-way feedback). No receipt, gate, or measurement exists
yet for the two-way path beyond its stamping; treat it accordingly.

Execution walks parent before child on a flat integer-tick schedule so no
floating-point clock drift can reorder coupling; child timestep derives exactly as
`dt_child = dt_parent / parent_time_step_ratio`
[docs/gpuwm-project-history.md:75; docs/public/CONFIGURATION.md:235].
Parent-to-child initialization uses WRF's SINT interpolation family with
stagger-aware geometry; lateral forcing is stored as value/tendency tables in WRF
`bdy_interp1` form [docs/gpuwm-project-history.md:77]. All 128 WRF boundary tables
became bit-identical after the staggered low-face U/V correction; the theta
successor residual was 2.59e-7 against a 1e-6 limit
[docs/gpuwm-project-history.md:121].

SINT donor geometry is precomputed FP64-on-host and stored FP32 (divergence D5):
bitwise-identical to WRF's per-op REAL construction for refinement ratios 1 through
4, machine-proven; exactly 1 ULP apart at ratio 5. The shipped configurations use
ratios 3 and 4 [PROVENANCE.md:576; tests/test_nest_interp.py].

## 2.5 Moving nests

### 2.5.1 The mechanism

A moving nest in WOOF-ArWen is a sequence of static nests joined by a re-grid at cycle
boundaries. WRF's continuous per-step motion is not adopted and its namelist keys
stay refused [docs/nest-relocation-identity-decision.md:29-31]. The design is exact
rather than approximate: a placement is a whole number of parent cells, so a move of
`di` parent cells is a shift of `di * ratio` child cells; substituting into WRF's
donor pickup shows the donor index and sub-cell offset are identical for an
overlapped cell before and after, so the overlap is a pure index-space copy and
anything the child derives from its parent by SINT is bitwise unchanged on it
[docs/nest-relocation-identity-decision.md:52-57].

Measured on an RTX 5090 (idealized ratio-3 tree, parent 168x168x60 at 1 km, child
120x120x60 at 333 m, moved 4 parent cells = 12 child cells, 90% overlap)
[docs/nest-relocation-identity-decision.md:60-80]:

| claim | result |
|---|---|
| null move leaves the child bitwise identical | child state sha256 unchanged, 25 fields stamped |
| rebuilds reproduce rather than copy | 8 fields (SINT base state, map factors), 58,081 cells, 0 mismatches |
| WRF `start_domain` RK post-condition re-established | 16 fields, 13,017,600 cells, 0 mismatches |
| parent bitwise unchanged, both moves | parent sha256 unchanged |
| overlap equals shifted outgoing child, bitwise | 0 mismatches over 18,714,960 cells, 25 restart-contract fields |
| treatment fired (not two identical arms) | 56% of overlap cells differ from a cold start; u 98.5%, v 98.6%, w 96.3% |
| post-move integration | 40 steps / 10 forces, finite everywhere, boundary metric under the frozen 0.5 threshold |

The purpose is capacity: a low-VRAM card can run a smaller nest that follows the
weather at a resolution a static nest of the same cost could never reach
[docs/nest-relocation-identity-decision.md].

### 2.5.2 What state moves with a nest (the 2.5.0 fix)

Two inventories move, both derived rather than hand-listed:

1. **Serialized model state**: exactly the restart layer's serialized-state
   contract (`woof.state_serialization_contract.STATE_SERIALIZED_ATTRS`), so a
   field added to the model reaches checkpointing and relocation at once
   [woof/core/nest_relocation.py:486-496].
2. **Driver-held physics continuation state** (new in 2.5.0): everything the
   physics driver holds per column that is cross-step memory nothing recomputes:
   Kain-Fritsch NCA hold timers, held cumulus rates, PRATEC/RAINCV, the
   RAINC/RAINNC precipitation accumulators, the W0AVG running trigger history. The
   shift set is derived from the restart registry
   (`woof.io.restart.SERIALIZED_SCRATCH_SLOTS`); a slot added to the registry
   tomorrow moves across relocations the day it is added
   [woof/core/physics_continuation.py:1-42].

Deliberately not carried: held PBL/surface/radiation tendencies and their timers,
because every one is recomputed from instantaneous state at the scheme's own cadence
(a rebuilt driver fires radiation immediately). Held cumulus tendencies are the
exception and are rebuilt from the carried raw rates after the move. The freshly
exposed strip takes each slot's documented cold value: 0 for every slot except the
KF `cu_nca` eligibility sentinel at -100 [woof/core/physics_continuation.py].

Before this fix, all of that continuation state was re-initialised from cold on the
whole child at every accepted move: convection cut off domain-wide, every column
became simultaneously re-eligible, the trigger memory restarted, and accumulated
precipitation reset mid-run. The defect ships in the released v2.4.1. Measured on an
ERA5-initialized 6 km moving child with KF active and four scripted moves (2 parent
cells east + 1 north, hourly from t=1.5 h)
[gallery:2026-08-17-moving-nest-kf/INDEX.md]:

| arm | behavior at the four moves | final RAINC | final RAINNC |
|---|---|---|---|
| unfixed moving+KF | RAINC -95.6% / -89% / -90% / -81% | 7,934 | 28,584 |
| fixed moving+KF | no wipe at any move | 35,239 | 111,458 |
| unfixed static+KF control (no moves) | n/a | 48,484 | 104,939 |
| released 2.4.1 wheel, moving+KF | -12,716 / -3,792 / -2,726 / -1,871 (mm-sum) | 8,023 | 28,484 |

The three KF arms are bitwise-identical until the first move, so the only downstream
difference is relocation behavior; reflectivity is continuous across the same move
because instantaneous fields ride the serialized transplant, which is what confines
the artifact to driver-held state. The fixed arm's small RAINC dips (-431 and -882
mm-sum) are strip-exit losses (heavy-rain columns leaving the domain), not resets
[gallery:2026-08-17-moving-nest-kf/INDEX.md;
tests/test_relocation_physics_continuation.py, 8 tests]. The fix ships default-on on
both moving-nest routes [CHANGELOG.md, engine 2.5.0].

### 2.5.3 What a move does not promise

A relocation invalidates any restart claim outright: placement is bound into the
prepared-cache identity and the tree restart fingerprint in three places, so moving
a nest one cell invalidates every prepared cache in the tree and breaks the tree
restart fingerprint. Two pinned tests hold this
[docs/nest-relocation-identity-decision.md:96-108]. Plan moving-nest work as
single-segment runs, or accept re-preparation on resume.

## 2.6 LES: the two closures

With `diff_opt=2`, two LES closures are selectable per domain
[docs/public/LES.md:24-32]:

| `km_opt` | closure | SGS TKE | selectable on a nest |
|---|---|---|---|
| 2 | 1.5-order prognostic TKE, `K = c_k sqrt(e) l` | prognostic, advected | yes, unless the parent is also `km_opt=2` |
| 3 | 3-D Smagorinsky, `K = (c_s l)^2 |S|` | diagnostic | yes |

`km_opt=2` additionally requires `bl_pbl_physics=0`: WRF will evolve TKE with the
PBL on, but that combination has no vertical TKE mixing (WRF's
`vertical_diffusion_2` is PBL-off-gated), so WOOF-ArWen refuses it rather than run a
half-wired scheme [docs/public/LES.md:24-32].

The property most often misread: with the PBL off, `km_opt` 2 or 3 give vertical
momentum and vertical scalar mixing; `km_opt` 1 or 4 give vertical momentum mixing
but no vertical scalar mixing. A 2-D Smagorinsky run with the PBL off does not mix
heat or moisture vertically by any route [docs/public/LES.md:44-55]. The `km_opt=2`
nest restriction exists because WRF gives `tke` no nest-interpolation and no
feedback Registry flag, so a child cold-starts its own TKE and never feeds it back
[docs/public/LES.md:346-355].

## 2.7 The anisotropic-mixing instability and its criterion

This is the single most important sub-km guidance in the manual.
`mix_isotropic = 0` (WRF's default) gives per-axis mixing lengths, and on a grid
whose layers are much deeper than the grid is wide that is a trap. WRF hands the
vertical-velocity horizontal-diffusion operator the vertical exchange coefficient;
`smag_km` builds and caps that coefficient on the layer depth
(`xkmv <= mix_upper_bound*dz^2/dt`), and the operator then differences it over
`dx`. Nothing compares the two, so the reachable diffusion number is

**`mix_upper_bound * (dz_max/dx)^2`, independent of `dt`**,

and an explicit Laplacian multiplies a 2*dx mode by `1 - 4*K*dt/dx^2` per step: past
1/4 the sign flips, past 1/2 the mode grows [docs/public/LES.md:385-394].

**Criterion: `mix_upper_bound * (dz_max/dx)^2 <= 0.25`.**

Measured ratios on this project's own trees at `mix_isotropic = 0`: a 250 m child at
0.702 (2.8x the limit), the 500 m domain of a 100 m tree at 0.169, the 100 m domain
at 4.23 (17x). The 4.23 tree aborted at step 5467 with `w = 239.48 m/s`,
bit-identical across three instrumented reproductions; the 0.702 trees completed
multi-hour runs, which is why the criterion is an advisory and not a refusal: it is
the worst case the cap admits, and completed runs at 2.8x the limit are the
receipt-backed evidence that a flow need not reach it [docs/public/LES.md:396-419].

**The remedy ships default-on.** A domain that leaves `mix_isotropic` unset (or
writes the sentinel `"auto"`) and violates the criterion runs `mix_isotropic = 1`
(isotropic `(dx dy dz)^(1/3)` length): selected once at config load, announced with
one line naming the ratio and the limit, and reported by the preflight checker
`woof check` (section 8.5) as what the run will do. A config that writes
`mix_isotropic = 0` keeps it, in the danger zone
too, and gets the advisory carrying the override state
[docs/public/LES.md:430-442; docs/public/CONFIGURATION.md:480]. Because
`mix_isotropic` is inside the restart fingerprint, a checkpoint written under the
old anisotropic default does not bit-continue under the auto-selected isotropic
form. A guard test fails if any shipped config arrives on the exposed path, and the
check resolves the level ladder from `nz`/`ztop` when a config leaves interfaces
implicit [tests/test_shipped_configs_mixing_stability.py; docs/public/LES.md:421-428].

Related divergence D8, which is a different operator from the one above: WOOF-ArWen
hands the *vertical* diffusion of vertical velocity (`vertical_diffusion_w_2`)
`xkmv` where WRF hands it `xkmh`. WRF's horizontal `w` operator
(`horizontal_diffusion_w_2`), the one the criterion above is about, takes `xkmv` in
WRF and WOOF-ArWen follows it unchanged. It can only matter where the two
coefficients differ (`km_opt` 2 or 3 with `mix_isotropic = 0`); taking WRF's choice
is unstable at dx 250 m against dz 17 m, and on the idealized convective
boundary-layer case the WRF oracle lane measured `XKMH == XKMV` at all 589,824
points, maximum difference exactly zero [docs/public/LES.md:363-381;
PROVENANCE.md:676].

Note the receipt scope: the measured nested numbers in the LES page were produced
with `mix_isotropic = 0`, before the auto-switch [docs/public/LES.md:444-449].

## 2.8 LES measurements

All on one RTX 5090 (sm_120, driver 575.57.08, CUDA 12.9, CuPy 14.1.1); dry
convective boundary layer 96x96x64 at dx = 100 m, ztop 2400 m, dt = 0.5 s, 2 h,
surface heat flux 0.24 K m/s [docs/public/LES.md:59-204]:

- 37 GPU kernel tests, each against an FP64 NumPy mirror of the WRF formula
  [tests/test_smag3d.py, tests/test_tke_km2.py, tests/test_tke_budget.py].
- Resolved buoyancy flux over surface flux: 0.838 (km2) and 0.844 (km3) on the base
  case, up to 0.94-0.97 on walled variants; entrainment ratios -0.143 to -0.282.
- TKE-based mixed-layer resolved fraction for `km_opt=2`: 0.894, above the
  conventional 0.8 threshold for a run to be called an LES; reported null for
  `km_opt=3`, which has no prognostic SGS TKE.
- Refinement to 50 m (192x192x96, dt 0.25 s, 28,800 steps): km2 resolved fraction
  0.894 to 0.932 and resolved flux fraction 0.838 to 0.896 in 858 s on one card;
  km3 0.844 to 0.904 in 753 s. This is a convergence direction measured at two
  spacings on one case; it is not a grid-convergence study and no order of
  convergence is claimed.
- WRF `em_les` head-to-head (WRF v4.6.1 built independently from pristine source):
  agreement at 100 m to 0.32% on z_i, 0.11% on w*, 0.15-0.29% on the resolved flux
  fraction, 0.27% on prognostic subgrid TKE. No pass band is cut; differences are
  reported, not judged.
- Realisation spread measured, not assumed (n=18 at 100 m, n=9 at 50 m): on the
  flux-fraction arm WOOF-ArWen's spread is 8.7x WRF's (0.01414 vs 0.00163 stdev at
  100 m), an open question recorded rather than diagnosed; on the TKE-based
  resolved fraction the two are at parity (0.00231 vs 0.00247).
- TKE budget closure over a 120-step window: relative residual 2.49e-9.
- Restart bit-identity: straight-through vs checkpoint-at-60-min-and-restore
  reproduces the end state exactly, 10 members (km2) and 9 (km3), none differing.

Nested, real-case, moist, terrain-following: a 250 m child inside an HRRR tree
carries up to 8.4x (`km_opt=3`) and 9.9x (`km_opt=2`) the 750 m parent's resolved
vertical-velocity variance over the same ground [docs/public/LES.md:12-15]. The
bound that must travel with that number: **the nested child is coarse LES at the
gray-zone edge**, because it runs its 3 km grandparent's 49 shared levels; measured,
18 levels inside the 1741 m boundary layer, an effective dz of 96.7 m. Vertical
resolution, not the 250 m spacing, is the binding constraint
[docs/public/LES.md:481-485]. LES enters the maturity ladder as
implemented-unverified; there is no separate LES-verified tier and none is claimed
[docs/public/LES.md:17-18].
