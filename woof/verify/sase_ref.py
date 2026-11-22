"""FP64 reference authority for the SASE-L1 dry turbulence core.

Design: docs/internal/specs/2026-07-20-sase-design.md.  This module is
the stage-1/2 evidence-ladder artifact: a NumPy, float64, CPU-only
implementation of the SASE-L1 sensor, dynamic closure, stress model, and
subgrid-energy budget on a horizontally periodic box.  The eventual CUDA
production implementation is gated against this module; nothing here may
import cupy or woof.core.kernels.

Array layout is woof's standard ``(nz, ny, nx)`` with x fastest.
Constants are Deardorff-synthesis values (spec section 10); the claimed
novel elements are the dynamic partition solve and the exact discrete
energy ledger, not these coefficients.

S3-6b amendment: anisotropic mixing lengths + implicit vertical
diffusion.  The v0 explicit step ``sase_ref_step`` uses the horizontal
filter scale for VERTICAL diffusion with explicit stepping, which is
linearly unstable on coarse domains (outer nest: delta = 12 km, dz1 ~ 50 m,
dt = 60 s gives 2*nu*dt/dz^2 = O(10^2) once sqrt(e) ~ 1; see
``physics._run_sase``).  S3-6c completed the retirement: the device
explicit path and its gates are gone, the device split step mirrors
:func:`sase_split_step`, and ``sase_ref_step`` stays bit-frozen as the
v0 historical authority with NO device consumer.  The amended
authority is :func:`sase_split_step`:

* Vertical mixing length: l_v = min(l_B, LS_COEF*sqrt(e)/N) on the
  N^2 > 0 branch, with the Blackadar (1962) neutral form
  l_B = k*(z+z0)/(1 + k*(z+z0)/lambda), lambda = BLACKADAR_LAMBDA
  = 150 m, k = KARMAN.  Vertical diffusivity K_v = C_KV*l_v*sqrt(e)
  with C_KV = C_E**(1/3) (log-layer consistency; derivation at the
  constant).  The dynamic weights (c_nu, f) deliberately do NOT enter
  K_v: the Germano identity is built from horizontal test filters
  (spec 4.1), so its weights calibrate the horizontal channel only,
  and a (c_nu, f)-free K_v preserves the exact log-layer constant and
  keeps the implicit coefficients per-column data.  f DOES enter the
  dissipation-length regime blend (:func:`dissipation_length`):
  l_d = min(delta**f * l_B**(1-f), l_s) -- GEOMETRIC since S3-9
  (originally the linear f*delta + (1-f)*l_B; the S3-9 section has
  the change and its evidence) -- recovering the v0 min(delta, l_s)
  bitwise at f = 1 (LES regime) and l_v at f = 0 (degenerate
  resolved field, RANS regime) under either form.
* Momentum: the full dynamic stress tau keeps the v0 machinery
  unchanged, but only its HORIZONTAL flux divergences step explicitly
  (du_i = -(d/dx tau_ix + d/dy tau_iy)); the vertical SGS flux is
  remodeled anisotropically as -K_v*d(phi)/dz and advanced with
  backward-Euler Thomas columns (zero-flux ends,
  :func:`implicit_vertical_diffusion`), likewise the vertical part of
  e-transport (coefficient 2*K_v) and -- through the same solver --
  the scalars' vertical mixing (K_v/PR_T; driver wiring is S3-6c).
  The e-budget buoyancy term rides the vertical channel:
  K_h,v = K_v/PR_T.  Dropping the d/dz(tau_i3) fluxes in favor of
  -K_v*d(phi)/dz is the standard mesoscale anisotropic closure (WRF's
  horizontal-Smagorinsky + PBL-column split does the same); the u<->w
  exchange is then modeled by different closures in the two equations,
  which is accepted and documented, and the 2e/3 isotropic part no
  longer exerts a vertical e-pressure force on w.

S3-6d amendment (analytic dissipation substep): the split step's e
update no longer carries dissipation as an explicit forward-Euler
source.  The substep order is (1) explicit sources
e* = e^n + dt*(P_h + P_v + buoyancy + T_h,e); (2) ANALYTIC decay
e* -> e*/(1 + b*dt)^2 with b = C_E*sqrt(max(e*, E_MIN))/(2*l_d) --
the EXACT solution of de/dt = -C_E*e^{3/2}/l_d over one dt from
initial value e* (l_d frozen at its e^n evaluation; verify by
substitution: e(t) = e0/(1+bt)^2 gives de/dt = -2b*e0/(1+bt)^3 =
-C_E*e^{3/2}/l_d iff b = C_E*sqrt(e0)/(2*l_d)); (3) clip to E_MIN;
(4) implicit vertical e-transport.  Explicit Euler dissipation is
unstable once dt*C_E*sqrt(e)/l_d > O(1) (outer-nest surface layer: l_B ~
kappa*z ~ 10 m, dt = 60 s puts the number at ~3 for e ~ e_eq), which
pinned surface-cell e in an E_MIN limit cycle under the live u*
source; the analytic substep is unconditionally stable, positivity-
preserving (0 <= e*/(1+b*dt)^2 <= e* for e* >= 0), and exact.

SPLIT-STEP LEDGER THEOREM (S3-6b statement, S3-6d decay channel).  On
a horizontally periodic box with the clamped zero-flux vertical
channel and UNIFORM dz, define for one :func:`sase_split_step` step
(per velocity component i):

    u_i* = u_i^n + dt*T_h,i     with T_h,i = -(ddx tau_ix + ddy tau_iy)
    u_i^{n+1} = (I - dt*D_v)^{-1} u_i*      [backward Euler, D_v the
                flux-form zero-flux operator with face diffusivities
                K_f = avg(K_v)]
    P_h = -(tau_xx*ddx u + tau_yy*ddy v + tau_xy*(ddy u + ddx v)
            + tau_xz*ddx w + tau_yz*ddy w)        [tau and grads at n]
    P_v = sum_i K_f*(dz u_i^{n+1})^2 at faces, split half to each
          neighbor cell (:func:`_vertical_production`)
    e*    = e^n + dt*(P_h + P_v + buoyancy + T_h,e)
    D     = e* - e*/(1 + b*dt)^2,  b = C_E*sqrt(max(e*, E_MIN))/(2*l_d)
            [the exact decay decrement over dt]
    e_clip = max(e* - D, E_MIN),  clip_gain = e_clip - (e* - D)
    dKE   = sum u^n.(u* - u^n) + sum u^{n+1}.(u^{n+1} - u*)
    dE    = sum(e^{n+1} - e^n) - dt*sum(buoyancy) - dt*sum(T_h,e)
            - sum(implicit e-transport increment)
    dHeat = sum(D - clip_gain).

Then dKE + dE + dHeat = 0 exactly in exact arithmetic and to relative
roundoff (< 1e-11, pinned by test) in FP64, because
(i)  the explicit pairing sum u^n . T_h = -sum P_h holds by horizontal
     summation-by-parts: the centered roll operators are skew-adjoint
     on the periodic directions (sum a*Db = -sum b*Da), and P_h pairs
     ONLY horizontal derivatives of u^n against tau; and
(ii) the implicit pairing sum u^{n+1}.(u^{n+1} - u*) =
     dt*sum u^{n+1}.D_v u^{n+1} = -dt*sum P_v holds by flux-form
     telescoping with zero end fluxes (production is deposited into e
     from the IMPLICIT-solved gradients -- the brief's "implicit-solved
     fluxes" pairing -- which is exactly what makes the identity hold;
     backward Euler damps each D_v eigenmode by 1/(1 + dt*lambda_m) in
     (0, 1], so the summed P_v = sum_m lambda_m a_m^2/(1+dt*lambda_m)
     is non-negative and the pointwise P_v is non-negative by
     construction).
The e channel closes by construction, and the S3-6d decay channel
SIMPLIFIES the algebra: e_clip - e^n = dt*(P_h + P_v + buoyancy +
T_h,e) - D + clip_gain identically (D is defined as the difference e*
minus its decayed value -- no discretization residual exists to
absorb), so with the theorem's exclusions
dE = dt*sum(P_h + P_v) - sum(D) + sum(clip_gain)
   = -dKE - dHeat.
The decay deposit into heat is exact by construction: D equals the
integral of C_E*e^{3/2}/l_d along the true decay trajectory, is
sign-definite (0 <= D <= e* wherever e* >= 0), and the floor can now
engage only where the SOURCES drive e* below E_MIN (decay alone
cannot cross the floor), so heat = D - clip_gain is >= 0 except in
those rare source-driven cells (v0's forward-Euler overshoot, which
the clip routinely caught, is structurally gone).  Unlike the v0
theorem this statement needs NO periodic vertical: (i) uses only
horizontal rolls and (ii) telescopes on any zero-flux column, so it
holds verbatim for the uniform clamped column.  Variable-``dz_col``
columns remain diagnostic-only under the unweighted ledger (the
1/thick_k factors break unweighted telescoping; a thickness-weighted
ledger would close on shared columns -- characterized here,
deliberately not pinned).

S3-6e amendment (RANS-limit horizontal governor + damping-layer
production taper; smoke-gate G2 adjudication 2026-07-20).  Two changes
to :func:`sase_split_step`, both formulation-level:

* HORIZONTAL GOVERNOR (:func:`governed_stress`).  The horizontal
  channel's viscosity becomes the f-blend
      nu = f*c_nu*delta*sqrt(e) + (1-f)*K_smag,
      K_smag = min((C_S*delta)^2*|D_h|, SMAG_KM_CAP*delta),
  with |D_h| the audited WRF 2-D Smagorinsky deformation (constants
  block above).  Rationale: the retired (1-f) momentum background
  (C_K/PR_T)*delta*sqrt(e) scales with the FILTER width in the RANS
  limit, where the across-cell strain is mesoscale gradient rather
  than turbulence -- at the outer nest (delta = 12 km) it converted synoptic
  deformation into subgrid-e production (the smoke G2/G7 4-6.6x
  surface-e excess).  The deformation form is WRF's own mesoscale
  closure for exactly this regime.  The SAME nu field now serves the
  stress, the e-transport K_m, and the scalar K_h = nu/PR_T --
  superseding, on the split path only, the registered v0 coefficient
  asymmetry (bare C_K in K_m vs C_K/PR_T in the stress background);
  the bit-frozen v0 machinery (``model_stress``/``e_rhs``) keeps it.
  ENERGETICS (registered decision, WRF-fidelity option adopted): the
  smag component's drain BYPASSES e and deposits directly into heat,
  mirroring WRF km_opt=4 semantics where horizontal Smagorinsky
  diffusion feeds no prognostic e (WRF simply drops the KE; we keep
  it, exactly, in the heat channel).  With r = nu_smag/max(nu,
  NU_BLEND_EPS) in [0, 1], the horizontal pairing splits as
      P_h,heat = r*(P_h,tot + (2/3)*e*div_h),   [= 2*nu_smag*G, the
                 smag share of the viscous pairing G]
      P_h,e    = P_h,tot - P_h,heat,            [dynamic + isotropic]
  and only P_h,e feeds e.  The dynamic solve is UNTOUCHED: it remains
  the partition sensor (its Germano bases are the v0 pair), while the
  applied RANS-limit closure is the audited deformation form --
  sensed-vs-applied stress mismatch at f < 1 is documented and
  accepted (the identical acceptance WRF makes by running Smagorinsky
  under any km_opt=4 flow).
* DAMPING-LAYER TAPER.  With ``zdamp`` given (the driver passes
  ``cfg.zdamp`` when damp_opt == 3), the e-equation production
  channels taper across the model-top damping layer with the SAME
  weight law as the audited damp_opt=3 KDH damper (acoustic.cu
  advance_w_phi: sin^2(pi/2*(z - (ztop - zdamp))/zdamp) above
  ztop - zdamp, per-column heights):
      g(z) = 1 - sin^2(pi/2*clip((z - (htop - zdamp))/zdamp, 0, 1)),
  evaluated at layer CENTERS with htop the per-column top interface
  (g = 1 below the layer, -> 0 at the top).  The tapered production is
  NOT discarded -- the withheld share redirects to heat so the ledger
  stays exact:
      x    = P_h,e + P_v,   src = g*x,   gb = g*buoyancy
      e*   = e + dt*((src + gb) + T_h,e)
      heat = D - clip_gain + dt*(P_h,heat + (x - src)).
  Buoyancy is tapered WITHOUT redirect (it is the PE-exchange channel,
  excluded from the closed triangle by the dE definition; the measured
  buoyancy channel becomes g*buoyancy).  e in the taper region then
  relaxes toward E_MIN on the dissipation timescale through the
  untouched S3-6d analytic decay substep -- no hard clamp exists.

RESTATED LEDGER THEOREM (S3-6e).  On the uniform-dz horizontally
periodic clamped box, with the definitions above and
    dE    = sum(e_clip - e^n) - dt*sum(gb) - dt*sum(T_h,e)
    dHeat = sum(D - clip_gain) + dt*sum(P_h,heat) + dt*sum(x - src)
    dKE   = dKE_expl + dKE_impl
the identity dKE + dE + dHeat = 0 holds exactly in exact arithmetic:
identity (i) pairs dKE_expl against the TOTAL P_h,tot = P_h,e +
P_h,heat (the momentum tendency rides the full blended tau, and the
split P_h,e + P_h,heat recombines to P_h,tot identically), identity
(ii) pairs dKE_impl against P_v as before, and the taper moves
production between the e and heat channels without creating or
destroying it (src + (x - src) = x identically).  At f = 1 with no
taper the step reduces BITWISE to the S3-6d formulation (0*K_smag,
r = 0, g = 1 are FP-exact no-ops), which the reduction test pins.
Heat is no longer pointwise sign-definite: P_h,heat inherits the
horizontal pairing's local sign (backscatter regions), exactly as
bookkeeping requires -- WRF drops this energy silently, the ledger
tracks it.

S3-6f amendment (partition cap + w-based resolved-fraction bound;
mesoscale-sensing concession, controller adjudication 2026-07-20).
The S3-6e review's root cause for the outer-nest surface-e excess: the
Germano solve returned f ~ 1 on ALL real domains, because HORIZONTAL-
velocity structure functions at mesoscale Delta sample balanced
(synoptic) motion -- they carry ZERO information about subgrid BL
turbulence there -- and the (then-linear, pre-S3-9)
l_d = f*delta + (1-f)*l_B blend then set the dissipation length to
the FILTER scale (12 km vs l_B ~ 9.4 m at the outer-nest surface:
dissipation throttled ~1280x, surface fixed point ~295 m^2/s^2).
This is a formulation-level concession, registered in
the spec (sections 3, 4.1, 10): flow-sensing from horizontal velocity
is provably uninformative in the mesoscale limit, so the sensed
partition must be BOUNDED there by prescribed scale information.  Two
bounds now cap the solved f INSIDE :func:`sase_split_step`, so the
capped value flows through every consumer (the l_d blend, the
governed-stress f-blend, and the (1-f) momentum-background weight):

* PARTITION CAP (primary; the Honnert 2011 / Shin-Hong 2015 class of
  prescribed Delta/z_i partitions, credited -- prior-art doc):
      f_used <= f_cap(rho),   rho = delta / z_i,
      f_cap = 1                          for rho <= F_CAP_KNEE (= 1)
      f_cap = exp(-((rho - F_CAP_KNEE)/F_CAP_WIDTH)^2)   above,
  a C^1 monotone ramp that is FP-exact 1 through the LES regime and
  Gaussian beyond the knee (:func:`partition_cap`).  FORM RATIONALE
  (documented decision; STATED FOR THE THEN-LINEAR BLEND -- S3-9's
  geometric l_d no longer needs the cap to reach 1e-3, and the
  registered Gaussian form deliberately stands unchanged): the
  (pre-S3-9) l_d blend is LINEAR in f, so l_d ~ l_B
  at the surface requires f_cap <~ l_B/delta ~ 1e-3 by rho ~ 10 --
  algebraic ramps of the 1/(1+x^2) class decay too slowly to get
  there; the Gaussian ramp reaches 1.6e-9 at rho = 10 with
  F_CAP_WIDTH = 2 while staying loose in the gray zone (0.78 at
  rho = 2, 0.37 at rho = 3 -- an UPPER bound there, deliberately
  above the Honnert subgrid-fraction data, because in the gray zone
  the W-SENSOR below supplies the estimate and the cap only guards
  the tail).  z_i is the bulk-Richardson boundary-layer height
  (:func:`bulk_richardson_zi`), the YSU-convention crossing
  (``npref.np_ysu_column`` ``diagnose``) collapsed to one registered
  critical value: Rib(k) = (theta_k - theta_1)*(g*z_k/theta_1)
  / max(|u_k|^2 + |v_k|^2, RIB_WSPD2_FLOOR), first crossing of
  RIB_CRIT = 0.25 (the classic Vogelezang-Holtslag-class radiosonde
  method; YSU's two-regime brcr with thermal excess is deliberately
  simplified -- the cap needs a SCALE, not a flux boundary, and the
  fixture bands tolerate factor-2 z_i shifts), linear interpolation
  in Rib between the bracketing layer centers.  Dry theta stands in
  for theta_v (L1 dry core).  FLOORS/FALLBACKS (documented): z_i
  floors at the FIRST INTERIOR layer center z[1] (a very stable BL
  whose Rib crosses immediately gets z_i = z[1] -- the strongest cap
  the geometry admits, which is the stable-BL fallback); a column
  with NO crossing (neutral through the top, e.g. weak winds over
  deep neutral stratification) returns the TOP layer center (the
  weakest cap -- deliberately permissive, a real profile crosses well
  below); nz = 1 returns z[0].  The step uses the domain-MEAN z_i
  (interior mean on the device path, matching the solve's
  specified-boundary exclusion).
* W-BASED RESOLVED-FRACTION BOUND (the gray-zone lever;
  :func:`w_resolved_bound`): resolved VERTICAL-velocity structure
  functions are the discriminating signal where horizontal ones
  saturate -- balanced mesoscale motion is quasi-horizontal, BL
  turbulence is not.  D2 of w rides the SAME r in {1, 2, 4}
  horizontal-increment machinery, N^2-SCREENED: cells with dry
  N^2 > N2_SCREEN (= 1e-4 s^-2, the free-troposphere Brunt-Vaisala
  benchmark N ~ 0.01 s^-1) are EXCLUDED from the accumulation --
  stratification at or above the tropospheric norm supports buoyancy
  oscillations at the sampled scales, so its w variance is prima
  facie gravity-wave motion, not turbulence (the screen is what
  keeps mountain/inertia-gravity waves from inflating the sensed
  resolved fraction).  Increments are anchored at screened cells
  (the anchor cell decides, mirroring the solve's boundary-mask
  convention); ``n2 is None`` (test boxes) treats every cell as
  neutral = passing, consistent with the module-wide n2-absent
  convention.  With E_r_w = 0.5*D2_w(2) (the w-component mirror of
  the sensor's e_res -- a one-component proxy that UNDERSTATES
  resolved energy ~3x under isotropy, i.e. biases the bound toward
  RANS, the conservative side):
      alpha_w = clip(e_mean/(e_mean + E_r_w), 0, 1),  f_w = 1 - alpha_w
  and f_used <= f_w wherever the screen passes ANY cell.  A SILENT
  screen (zero passing cells -- e.g. a fully stable domain at night)
  makes the w-sensor abstain: f_w = 1.0 (no constraint; alpha_w is
  reported at its degenerate all-subgrid value 1.0 and must not be
  read as a measurement) -- THE CAP GOVERNS REGARDLESS.

Combination (:func:`sase_split_step`):
    f_used = min(f_solved, f_cap, f_w),
where f_solved is the untouched Germano 2x2 solution -- the solve
retains full authority to LOWER f (it detects under-resolution
correctly); the caps only bound the claimed resolvedness from above
with scale/vertical-velocity information the horizontal identity
cannot supply.  c_nu is NOT rescaled: the dynamic eddy weight is
f_used*c_nu, so capping f attenuates the eddy channel and hands the
balance to the audited (1-f_used) RANS closures -- exactly the
concession's intent (at f_used -> 0 the step is WRF km_opt=4
horizontal Smagorinsky + Blackadar-length RANS dissipation, and the
gate-2a cap becomes the EXACT surface fixed point).  At
delta/z_i <= 1 with w-rich fields (f_cap = 1 FP-exact, f_w >=
f_solved) the min leaves f_solved bitwise and the S3-6d/6e behavior
is recovered (pinned).  The ledger gains the diagnostics f_solved,
f_cap, f_w, zi, w_coverage; ``f`` remains the value the step USED.
The LEDGER THEOREM IS UNAFFECTED: the cap changes coefficients
(f, hence nu and l_d), never channels -- every identity above is
f-pointwise, so closure holds verbatim (asserted with the cap
engaged).

S3-6g amendment (regime-consistent Prandtl number; smoke-c sealed-FAIL
diagnosis, controller ledger 2026-07-20 ~23:0x, registered before
data).  Smoke-c exonerated wave feedback (f_w flat) and TKE runaway
(bounded, decelerating) and converged on the scalar channel: the FIXED
PR_T = 1/3 is the LES-limit (Deardorff inertial-range) value, but the
S3-6f cap drives the real domains into the RANS/transition regime
(f_used ~ 0 at the outer nest), where the observed
neutral-surface-layer Prandtl number is ~0.7-1 (K_h ~ 1.2*kappa*u*z
against K_m ~ kappa*u*z -- registered approximation #1, now promoted
LAUNCH-BLOCKING).  Dividing
the RANS-regime K_v by the LES 1/3 tripled every scalar diffusivity:
the model mixed its own morning stratification down (inv_h rising,
theta gradient decaying) and the released stability showed up as the
field-wide 10 m wind amplification (inner-nest mean 4.3 -> 14.2 m/s over
90 min) with "healthy" TKE.  The fix is the registered regime blend

    Pr_t(f) = PR_RANS + f*(PR_LES - PR_RANS),
    PR_RANS = 0.85, PR_LES = 1/3 (the former PR_T),

implemented in the FP-exact two-product form f*PR_LES + (1-f)*PR_RANS
(:func:`prandtl_blend`; both endpoints bitwise, the module's blend
idiom) and evaluated at the step's USED f -- the SAME f_used that
blends l_d and the governed stress, so the Prandtl number rides the
identical regime judgment as every other S3-6b/6e/6f blend.  The
ledger theorem is unaffected (Pr enters coefficients only: the
buoyancy K_h, which the dE definition excludes from the closed
triangle).

PR_T-CONSUMER DECISION TABLE (S3-6g audit; every division or
multiplication by the former PR_T, with its adjudicated disposition):

===============================  =============================  ========
site                             role                           decision
===============================  =============================  ========
``model_stress`` nu_mom          (1-f) equilibrium momentum     FIXED ->
  (was C_K/PR_T)                 background viscosity           C_MOM_BG
``_solve_tail`` f recovery       Germano momentum-basis         FIXED ->
  (was C_K/PR_T)                 weight b = (1-f)*C_MOM_BG      C_MOM_BG
``e_rhs`` k_h = k_m/PR_T         v0 explicit buoyancy K_h       FROZEN
                                 (historical authority, no      (PR_T =
                                 device consumer since S3-6c)   PR_LES)
``scalar_mix`` kh_coef           v0 scalar-channel coefficient  FROZEN
  convention (doc)               (fixtures + CPU-shim seam      (blend at
                                 fallback -- the live fallback  the call
                                 call site blends)              site)
``sase_split_step`` buoyancy     split-path buoyancy K_h        BLENDED:
  (was kv/PR_T)                                                 kv/Pr_t(f)
driver scalar vertical kfac      theta/qv/qc/qi implicit        BLENDED:
  (was 1/PR_T)                   K_v/Pr vertical solves         1/Pr_t(f)
driver scalar horizontal         governed K_h = km_h/Pr         BLENDED:
  kh_fac (was 1/PR_T)            horizontal fluxes              1/Pr_t(f)
device ``launch_model_stress``   kernel momentum-background     FIXED ->
  (was DTYPE(C_K/PR_T))          constant (FP32-identical)      C_MOM_BG
device ``launch_sase_step``      fused e-source kernel's        BLENDED:
  (was DTYPE(PR_T))              ``pr_t`` scalar argument       Pr_t(f)
CFL derivation                   worst-case explicit scalar     PR_LES
  max(CNU_MAX, C_K)/PR_T         K_h bound (min over f of       (worst
                                 Pr_t(f) is PR_LES at f = 1)    case)
===============================  =============================  ========

C_MOM_BG RATIONALE (registered decision): the 0.30 momentum-background
coefficient inherited from the Task-5 design was WRITTEN as C_K/PR_T
but is the neutral-K_m momentum-channel calibration -- not a
scalar-channel Prandtl usage -- so the regime blend must NOT drag it.
It is renamed to the standalone registered constant C_MOM_BG (defined
as C_K/PR_LES so its FP64 value 0.30000000000000004 stays bit-identical
to every frozen v0 fixture and golden) and stays FIXED for all f.

The blend flows through every scalar-side consumer: the split step's
buoyancy term, the driver's implicit vertical scalar solves, and the
governed horizontal scalar channel (kh_fac).  The step exports
``pr_t`` in its ledger (diagnostic; the driver recomputes the same
value from the retained ``f`` so the CPU-shim seam needs no new key).
At f = 1 the whole amendment is FP-exact inert (Pr_t = PR_LES = the
old PR_T bitwise); at f < 1 the buoyancy channel shifts and the
SPLIT_* trajectory goldens re-pin deliberately (tests/sase_goldens.py
records the measured shift).  RED evidence obligation: the
inversion-persistence fixture (test_sase.py) demonstrates the failure
this amendment fixes -- with Pr forced to the old fixed 1/3 (PR_RANS
monkeypatched to PR_LES, which reproduces the pre-S3-6g formulation
bitwise at f = 0) the prescribed morning inversion erodes 1.7x faster
(0.529 vs 0.304 K of the 2.075 K jump in 1800 s) and its height rises
3.5x faster (29.8 vs 8.6 m) than with the blend -- the height signal
carries the mixdown feedback (eroded jump -> lower N^2 -> longer l_v
-> larger K_v) most cleanly.  (S3-6h note: the RED leg now also pins
the pre-S3-6h lengths -- see that section -- so the historical
measurements stay bitwise.)

S3-6h amendment (Bougeault-Lacarrere displacement lengths in the RANS
limb; jet-coupling diagnosis, ratified option 2, controller
ledger 2026-07-21 ~01:0x, registered before implementation).  The
smoke-c/d obs arbitration (the surface-observation arbitration record
filed with the campaign evidence; an 8-station surface-observation
table) plus an independent spun-up CPU WRF frame
established that the SASE inner-nest 10 m wind amplification (13:30Z
mean 14.03 m/s vs observed ~5.7, ramping while obs held) is SPURIOUS,
and
the surviving diagnosis is missing stability-aware suppression of
RANS-limb vertical momentum mixing: l_v = min(l_B, l_s) carries only
LOCAL stability information (l_s = 0.76*sqrt(e)/N), so a morning
column whose mixed layer erodes toward neutral under a capping
inversion sees l_s -> infinity locally, l_v -> l_B (the NEUTRAL
length, up to 150 m), K_v = O(10-100) m^2/s in the sheared stable
transition -- and the low-level jet couples to the ground.  The fix
is the Bougeault-Lacarrere (1989, MWR 117, 1872-1890) parcel-
energetics length pair, computed per column against the LIVE theta
profile:

    l_up(k):  first l with  int_{z_k}^{z_k+l} beta*(theta_v(z') -
              theta_v(z_k)) dz' = e(k),   beta = G_ACCEL/theta_v(z_k)
    l_down(k): the downward mirror, integrand beta*(theta_v(z_k) -
              theta_v(z')), bounded by the surface,

(:func:`bl89_displacement_lengths`; dry theta_v = theta, the L1 core's
registered simplification; the discrete integral is EXACT for
piecewise-linear theta_v -- layer-wise quadrature plus the quadratic
fractional-segment solve, derivations at the functions).  The lengths
are inversion-aware BY CONSTRUCTION: a parcel rising toward an
inversion is stopped by the inversion's integrated buoyancy however
weak the local N^2 below it -- exactly the information the coupling
failure lacked -- and they are flow-derived, not prescribed (the SASE
DNA; credited synthesis, AROME/Meso-NH lineage).  Combination forms
(:func:`bl89_combine`; constants block at BL89_MIX_EXP):
l_mix = the -2/3-power mean (Cuxart-Bougeault-Redelsperger 2000,
QJRMS 126, 1-30 -- the Meso-NH/AROME operational form, dominated by
the smaller displacement), l_eps = min(l_up, l_down) (BL89's
dissipation length; l_eps <= l_mix always).  Wiring
(:func:`bl89_rans_lengths` -> :func:`sase_split_step` step 2):

    l_mix_rans = min(l_B, l_s, l_mix_BL89)
    l_eps_rans = min(l_B,      l_eps_BL89)
    l_v  = f*min(l_B, l_s) + (1-f)*l_mix_rans     [K_v = C_KV*l_v*rt e]
    l_d  = min(delta**f * l_eps_rans**(1-f), l_s) [dissipation blend;
                                                   geometric since S3-9
                                                   -- was linear here]

* the l_B min is the kappa-z-class floor/match (BL89_KZ_MATCH): the
  NEUTRAL LOG-LAYER FIXTURE REMAINS A HARD CONSTRAINT, and in a
  neutral column the BL89 lengths are pure geometry (l_down = z,
  l_up = htop - z) >= l_B everywhere except within ~l_B of the model
  top, so the log-layer window is bitwise untouched (the fixture
  engine now runs the composed length to prove it);
* l_s is RETAINED as the outer stability min (BL89_LS_DECISION):
  the analytic uniform-stratification fixture pins BL89's stable
  limit at l_up = l_down = sqrt(2e)/N (Rodier et al. 2017, JAS 74)
  = 1.86x the audited Deardorff 0.76*sqrt(e)/N, so retirement would
  LENGTHEN the stable-limit RANS mixing length in exactly the regime
  under suppression; retention makes the amendment MONOTONE (no
  RANS-limb length ever exceeds its pre-S3-6h value);
* the LES limb is untouched BITWISE: at f = 1 the two-product blend
  gives 1.0*min(l_B, l_s) + 0.0*l_mix_rans and l_d = 1.0*delta +
  0.0*l_eps_rans, both FP-exact (pinned);
* the ledger theorem is UNAFFECTED: the lengths are coefficients
  (K_v faces, l_d), never channels -- every identity is pointwise in
  the coefficients, so closure holds verbatim (asserted with BL89
  binding).

RED evidence obligation (jet-decoupling fixture, test_sase.py): the
single-column scoring-box mean profile from the reference CPU WRF frame
(the trusted reference forecast frame at 13:08Z; the jet-profile
fixture module carries the code-generated constant table with
provenance) -- a
strongly stable surface-based morning layer (theta 286.9 -> 295.8 K
across the lowest 470 m) under a 17.6 m/s low-level jet at 472 m with
4.3 m/s at 8.8 m -- integrated 3600 s at dt = 60 with the driver's
exact per-step sequence.  RED = the pre-S3-6h formulation, reproduced
bitwise by monkeypatching :func:`bl89_rans_lengths` to return the v0
pair (the S3-6g pinning idiom): the jet couples and the 10 m
diagnostic exits the observed band (> 7.0 m/s; measured numbers at
the fixture).  GREEN = this amendment holds the 10 m wind inside the
obs-derived [4.5, 7.0] m/s band AND the jet-core wind within 15% of
its initial value (no erosion of the jet from below).  Obs anchor:
an 8-station surface-observation 13Z mean of 5.61 m/s (the
arbitration record filed with the campaign evidence).

S3-6i amendment (decoupled stable-limit diffusivity coefficient C_KS;
controller ledger 2026-07-21 ~01:5x, registered before
implementation).  The S3-6h jet fixture falsified the ratified
length-locality premise on its own trajectory (BL89 bitwise-inert on
the smooth strongly-stable frame profile) and isolated the true driver
of the spurious inner-nest morning-wind amplification: the STABLE-LIMIT
COEFFICIENT COMPOSITION.  Wherever l_s bound the RANS mixing length,
the S3-6b wiring gave

    K_v = C_KV*l_s*sqrt(e) = C_KV*LS_COEF*e/N = 0.742*e/N,

i.e. the neutral-wall log-layer constant C_KV = C_E^{1/3} (a
calibration valid ONLY in the l_B -> kappa*z wall regime -- its own
derivation block says so) riding the shared composition into the
stably-stratified regime: ~10x the Deardorff-consistent stable
composite 0.076*e/N (Deardorff 1980 pairs the 0.76*sqrt(e)/N length
with C_K = 0.1) and 3-5x the ~0.15-0.25*e/N operational band (S3-6h
report section 4, the registered finding).  The fix decouples the
COEFFICIENT and leaves the length composition untouched: the RANS
limb rides the min-aware blend

    rho = min(l_mix_rans/l_s, 1)               [0 where N^2 <= 0]
    C_r = C_KV + (C_KS/LS_COEF - C_KV)*rho**CKS_BLEND_EXP
    K_v = f*C_KV*l_les*sqrt(e) + (1-f)*C_r*l_mix_rans*sqrt(e)

(:func:`stable_limit_coefficient`; constants block at C_KS; the l_v
length blend becomes the equivalent two-product K_v blend so the
coefficient change is RANS-limb-only).  Properties, each pinned:

* STABLE LIMIT: where l_s binds, rho = 1 BITWISE (l_s is a term of
  the l_mix_rans min), so K_v = (C_KS/LS_COEF)*l_s*sqrt(e) =
  C_KS*e/N -- the registered decoupled asymptote.
* NEUTRAL WALL: unstable/neutral cells (n2 <= 0 or absent) keep
  C_r = C_KV FP-EXACT (x + dC*0.0 == x), so the neutral log-layer
  engine is bitwise untouched -- l_s never binds in a neutral column
  (asserted in the fixture).
* CONTINUITY ACROSS THE NEUTRAL<->STABLE TRANSITION (claim narrowed
  in S3-9c per the adversarial review, Minor 3): approaching N = 0 with
  l_B binding, rho = l_B*N/(LS_COEF*sqrt(e)) is linear in N, so with
  CKS_BLEND_EXP = 2 the coefficient deficit C_KV - C_r ~ N^2 = n2.
  Parameterized by the signed frequency N = sqrt(n2) the blend is
  C^1 (dK_v/dN -> 0 from the stable side, matching the
  N-independent neutral side); as a function of the MODEL INPUT n2
  it is C^0 ONLY -- the deficit onset is LINEAR in n2, so
  dK_v/d(n2) jumps from 0 to a finite slope at neutral.  The blend
  is continuous with no jump in K at n2 = 0; the linear-in-n2 onset
  is the registered behavior (pinned by the linear-onset witness in
  the closed-forms test).  The other slope breaks in K are the
  PRE-EXISTING min-switch kinks (l_B <-> l_s <-> l_BL89), which the
  blend co-locates with (rho reaches 1 exactly where the l_s min
  engages); kinks live in coefficients only and the ledger theorem
  is pointwise in the coefficients, so closure holds verbatim
  (asserted under stable binding).
* LES LIMB BITWISE: f = 1 gives 1.0*C_KV*l_les*sqrt(e) + 0.0*(...)
  FP-exact -- the stable decoupling is RANS-only (pinned).
* DISSIPATION UNTOUCHED: l_d keeps its S3-6b/S3-6h composition and
  C_E (one-stability-limit rule) -- C_KS is a DIFFUSIVITY
  coefficient, exactly as registered.

CALIBRATION (the jet-decoupling fixture, promoted to its registered
GREEN criteria): C_KS swept over {0.076, 0.10, 0.15, 0.20, 0.25} on
the frame-true column; the registered choice is the LARGEST value
that holds the 10 m wind inside [4.5, 7.0] m/s through 3600 s AND
the jet-core wind within 15% of initial, with margin (measured sweep
table at the fixture and the C_KS constant).

S3-6k amendment (decoupled stable-limb DISSIPATION coefficient C_ES;
gated by ``RunConfig.sase_stable_dissipation``, DEFAULT False, so HEAD
is bitwise unchanged until a run asks for it).  S3-6i decoupled the
stable-limit DIFFUSIVITY and recorded the omission in its own words --
"DISSIPATION UNTOUCHED: l_d keeps its S3-6b/S3-6h composition and C_E
(one-stability-limit rule) -- C_KS is a DIFFUSIVITY coefficient" (the
bullet above).  This amendment closes that omission on the identical
machinery.  THE DEFECT: where l_s binds l_d, the limb dissipates at

    eps = (C_E/LS_COEF)*e*N = 1.2237*e*N,

i.e. the neutral-wall constant C_E riding the Deardorff stability
length into the stably-stratified regime -- the same category of error
S3-6i fixed one slot over.  Deardorff's own pairing for that length is
c_eps -> 0.19 as lambda/Delta -> 0 (C_ES; provenance and the published
transcription at the constant).  MEASURED on the trusted MYNN reference
this session (reflen.py, box-B land, the frame the envelope doc names):
the reference dissipates at 0.2404*e*N (deck band) to 0.2765*e*N
(subcloud) on its own length -- 4.4-5.1x below HEAD -- while its
diffusivity ratio K_h/(e/N) = 0.3111-0.3578 sits within 6-18% of this
module's C_KS/PR_RANS = 0.29412.  One half of the pair is calibrated,
the other is not.

THE FORM, mirroring S3-6i's two-product K_v blend term for term:

    l_s    = LS_COEF*sqrt(max(e, E_MIN))/N            [N^2 > 0]
    rho    = min(l_d/l_s, 1)
    w      = rho**CKS_BLEND_EXP
    C_rans = (1 - w)*C_E + w*C_ES                     [two-product]
    C_eps  = f*C_E + (1 - f)*C_rans   where N^2 > 0    [two-product]
    C_eps  = C_E                      otherwise        [SELECTED]
    eps    = C_eps*e^{3/2}/l_d

(:func:`stable_dissipation_coefficient`).  Three properties, each
pinned, each essential:

* STABLE LIMIT BITWISE: :func:`dissipation_length` ends
  ``l = np.where(stable, np.minimum(l, ls), l)``, so l_d == l_s BITWISE
  wherever the stability limit binds and rho == 1.0 exactly there.  The
  TWO-PRODUCT form then returns C_ES bitwise (0.19,
  ``52b81e85eb51c83f``); the affine C_E + (C_ES - C_E)*w returns
  0.19000000000000006 (``54b81e85eb51c83f``) -- measured this session,
  which is why the form is two-product and not affine.  rho is built on
  l_d, the length the coefficient MULTIPLIES, not on l_mix_rans.
* NEUTRAL/UNSTABLE BITWISE, BY SELECTION AND NOT BY CANCELLATION: at
  n2 <= 0 (or ``n2 is None``) the coefficient is the SELECTED literal
  C_E -- the authority branches on the same predicate the device gate
  ``if (has_ces && ls_v > 0.0f)`` uses, because the arithmetic does NOT
  cancel.  The first S3-6k commit (fb67b9d) returned the unconditional
  blend f*C_E + (1-f)*C_rans and relied on C_rans == C_E there; measured
  this session over a uniform 10001-point f grid, 3661 of those f values
  make f*C_E + (1-f)*C_E return 0.9299999999999999 or
  0.9300000000000002 instead of 0.93 (``c3f5285c8fc2ed3f``), and the
  recorded production f = 4.1188928660938e-05 is one of them -- so the
  FP64 authority disagreed with its own FP32 mirror by 1 ulp on
  precisely the cells the amendment claims not to touch.  Measured at
  step level on a fully unstable column (n2 < 0 in every cell, live
  solved f = 0.6248773184883271): e and heat moved by 4.16e-17 with the
  switch ON.  With the gate transcribed the neutral log layer, the
  C_KV = C_E**(1/3) identity and every moist-unstable cell -- including
  the M1-substituted cells where N^2 goes negative, which is where the
  cloud-top over-entrainment defect lives -- are untouched, and this
  amendment is bitwise ABSENT from that defect and cannot deepen it.
  Pinned by ``test_stable_dissipation_coefficient_closed_forms`` over a
  DENSE f sweep (the old {0, 0.5, 1} sample is all-safe and could not
  see this) and at step level by
  ``test_sase_split_step_stable_dissipation_is_inert_where_unstratified``
  on a fixture whose solved f is asserted unsafe.
* LES LIMB BITWISE: the outer f-weighting makes the change RANS-only,
  restoring S3-6i's own pinned LES-limb property for the dissipation
  half: at f = 1 the expression is C_E bitwise for ANY stratification
  and any C_rans, so every LES fixture is inert with the switch ON as
  well as OFF.

DERIVED, registered so no later coefficient move shifts it silently:
the stable-limb critical Richardson number

    Ri* = C_KS/(C_eps/LS_COEF + C_KS/PR_RANS)

is 0.35385638343882664 pre-S3-6i (C_KV*LS_COEF limb), 0.16471188169301373
at HEAD, and 0.45945945945945943 under this switch.  Equivalently the
steady-state mixing efficiency Gamma_m = K_v*N^2/eps = C_KS*LS_COEF/C_eps
moves from 0.2043 -- Osborn's (1980, J. Phys. Oceanogr. 10:83) bound,
hard-coded into a PROGNOSTIC equation -- to 1.0.

WHAT Ri* ACTUALLY IS, and the LANE IT CLOSES (S3-6k landing amendment;
measured this session, ``ident.py``, and registered because it is a
NEGATIVE result that saves a future lane the trip).  Divide numerator
and denominator by C_KS/LS_COEF and the registered expression is the
HARMONIC MEAN of the mixing efficiency and the Prandtl number:

    Ri* = Gamma_m*PR_RANS/(Gamma_m + PR_RANS),   Gamma_m = C_KS*LS_COEF/C_eps

reproducing the three registered values to <= 1 ulp (0.16471188169301376
vs ...373 at HEAD and 0.3538563834388267 vs ...664 pre-S3-6i, both 1 ulp
from the associativity of the rewrite; 0.45945945945945943 BITWISE under
this switch).  A harmonic mean is strictly below both its arguments, so

    Ri* < min(Gamma_m, PR_RANS)     STRICTLY, at every admissible pair
                                    (0 violations in 200000 random pairs
                                    over [1e-2, 1e1]^2, measured)

and therefore PR_RANS CANNOT lift Ri* above Gamma_m: at HEAD's
Gamma_m = 0.2043 the supremum of Ri* over ALL PR_RANS is 0.2043
(PR_RANS = 0.4/0.85/2/10/1e6 give Ri* = 0.1352/0.1647/0.1854/0.2002/
0.2043).  A STABILITY-DEPENDENT PRANDTL NUMBER IS THEREFORE NOT A CURE
for the stable-limb collapse, and no length composition is either --
Ri* contains no length at all.

AND THE REFERENCE SUSTAINS ITS DECK ABOVE BOTH VALUES.  Measured this
session on the trusted MYNN 13:08Z frame (``ref.py``; dry gradient
Ri = N^2/S^2 on interior faces, land, 0.4-1.4 km AGL, BOTH bracketing
cells saturated on the model's own Tetens switch; the three scoring
regions are carried here as box A / box B / sector, smallest first --
their case identity lives with the campaign evidence, not here):

    box    n       Ri p50   p90     frac < 0.16471   frac < 0.45946
    box A   2,212  0.5968   1.7049      0.0118           0.3070
    box B  17,850  0.5450   1.2213      0.0202           0.3657
    sector 42,492  0.6037   1.4951      0.0305           0.3053

So 96.9-98.8% of the reference's own deck faces are SUPERCRITICAL under
HEAD's Ri* = 0.16471 -- the closure must set them to the E_MIN floor --
and 63.4-69.5% still are under C_ES's 0.45946.  Sustaining that deck
needs Gamma_m > 0.55, i.e. 2.7x Osborn's bound, and holding its top
decile needs Gamma_m > 1.2 AND PR_RANS > 1.2.  No admissible pair does
that.  The remaining move inside this closure is not a coefficient; it
is the EFB structure named below.

WHAT THIS IS NOT.  It does not remove the absorbing state.  Where l_s
binds, K, B and eps all stay exactly LINEAR in e, so the limb remains a
bang-bang switch with a hard critical Richardson number; only its value
moves.  Turbulence in the saturated layer will stay intermittent rather
than sustained.  The structural cure is a total-turbulent-energy/EFB
formulation in which buoyancy is a TKE<->TPE conversion and no critical
Ri exists (Mauritsen et al. 2007 JAS 64:4113; Zilitinkevich et al. 2007
BLM 125:167, 2013 QJRMS 139:1741) -- its own lane, not this one.
Reading this amendment as a well-posedness fix is a misreading.

CALIBRATION STATUS: **RED, AND THE SWITCH STAYS OFF.**  Measured this
session on the registered S3-6i calibration gate -- the frame-true
drag-free jet column, ``test_jet_decoupling_stable_coefficient_holds_
obs_band``, whose GREEN criteria are the obs-derived 10 m band
[4.5, 7.0] m/s at every step and jet-core hold within 15%:

    stable_dissipation   u10 min  u10 max  @900   @1800  @3600  jet dev
    False (registered)    5.701    6.580   6.423  6.479  6.580   0.063%
    True  (C_ES = 0.19)   5.701    8.394   7.917  8.263  8.394   0.218%
    [pre-S3-6i RED ref    5.857    7.356   7.103  7.220  7.356   0.19% ]

The switched-on leg exits the observation band, and by MORE than the
pre-S3-6i coupled formulation this whole line of work exists to fix.
Criterion 2 (jet-core hold) still passes; criterion 1 fails.

THE MECHANISM, measured, and it is NOT a 4.89x rescaling.  Below 500 m
in that column, l_s binds in 176 of 176 cells, and e rises from the
E_MIN floor (p50 1.000e-06 -- the absorbing state, i.e. the limb
produces no turbulence at all) to p50 9.670e-04, a factor of 216 at the
median and 1054 at the maximum, with K_v p50 going 0.000 -> 0.008
m^2/s.  That is the Ri* shift doing exactly what it says: the cell
population with 0.16471 < Ri < 0.45946 flips from "collapses to the
floor" to "sustains", and the extra mixing pulls the nocturnal jet
down.  The amendment therefore DOES remove the zero-TKE stable limb it
was designed to remove -- and the same act over-mixes the observed
morning stable boundary layer.

AT RUN SCALE THE SAME TWO-SIDEDNESS, MEASURED (S3-6k landing amendment;
``deck.py``/``ref.py`` on frames already on disk, intermediate nest,
land, saturated cells 0.4-1.4 km AGL on the model's own Tetens
switch).  The switch-on leg was run only with the M1 moist-N^2
substitution DISABLED
(the s36k M1-off run leg), so it is not a default-path result; but on
the discriminator that cannot be gamed by a scale factor -- the SHAPE
of the implied diffusivity profile -- it is the only configuration this
project has produced that matches.  Median implied
K_v = -SASE_FQV_DIFF/(rho dqv/dz) in 200 m bins, m2/s, box B:

    400-600 | 600-800 | 800-1000 | 1000-1200 | 1200-1400 m
    reference (Sh*EL*sqrt(QKE), Sh = 0.5)
      9.35  |   6.64  |   4.42   |   2.55    |   1.02     FALLS 9x
    C_ES on, M1 off, 12Z
     10.56  |   5.36  |   3.70   |   2.70    |   0.98     FALLS 11x
    C_ES on, M1 off, 13Z
     12.09  |   7.34  |   4.37   |   2.92    |   1.30     FALLS 9x
    DEFAULT PATH (M1 on, switch off), 12Z
      3.15  |  83.70  | 146.24   | 139.33    |  84.81     RISES to a
                                                          maximum at
                                                          deck top

Within a factor of 1.4 of the reference in EVERY bin, monotone falling,
against a default path that is two orders high and has the wrong
gradient SIGN.  It still does not ship: the same leg is RED on the
registered jet gate above (u10 8.394 against [4.5, 7.0]) and its 13Z
resolved w_colmax reaches 19.32 m/s over box B / the sector against an
acceptance criterion of ~1.4 m/s.  The record is therefore NOT "the
amendment produces nothing"; it is "the amendment's diffusivity profile
is right and its momentum behaviour is wrong, and no single coefficient
reconciles them" -- which is the same conclusion the Ri* bound reaches
analytically.

WHAT THIS FALSIFIES.  Not C_ES on its own, and NOT by an amount anyone
may tune away: it falsifies the PAIR.  C_KS = 0.25 was selected on this
fixture with C_E in the dissipation slot; C_ES = 0.19 is published for
Deardorff's length with Deardorff's own diffusivity.  Each is defensible
alone and together they over-mix.  Ri* is the joint quantity -- the pair
enters it as C_KS/(C_eps/LS_COEF + C_KS/PR_RANS) -- so re-registering
either member alone is not available.  Moving C_KS is outside this
amendment's authority and moving C_ES to make a gate pass would be
fitting a registered constant to a target, which this module does not
do.  The switch is therefore committed DEFAULT FALSE and INERT, with
the measurement above as the reason.

CORRECTION (S3-6k landing amendment).  The sentence above closed by
handing back "the joint (C_KS, C_ES) re-registration" as though it were
an open door.  It is a CLOSED one, and it is recorded closed here so no
later lane re-opens it by hope.  Measured this session (``ident.py``):

  * holding C_ES at the published 0.19 and sweeping C_KS across its own
    registered sweep, Ri* moves 0.2239 / 0.2720 / 0.3517 / 0.4121 /
    0.4595 at C_KS = 0.076 / 0.10 / 0.15 / 0.20 / 0.25 -- so C_KS alone
    cannot be re-registered without moving Ri*, which is the first
    reading of the sentence and it holds;
  * moving the PAIR TOGETHER along Gamma_m = 1 (i.e. C_eps = C_KS*
    LS_COEF) leaves Ri* at 0.45945945945945943 BITWISE at every one of
    those five C_KS values, because on that line Ri* collapses to
    PR_RANS/(1 + PR_RANS) and the diffusivity coefficient drops out
    entirely.  The whole Gamma_m = 1 line has ONE Ri*, and it is
    0.45946 -- which the jet gate above measures RED.

Neither reading unblocks the RED.  The joint re-registration does not
have a free parameter to spend on it.

A NOTE ON THE SPELLING, so a later reader does not mistake a
coincidence for a derivation: C_ES = 0.19 and C_KS*LS_COEF = 0.25*0.76
are BITWISE EQUAL (both ``52b81e85eb51c83f``, measured), because
C_KS = 2^-2 exactly makes the product a pure exponent shift of
LS_COEF's mantissa (``52b81e85eb51e83f`` -- identical mantissa, exponent
lower by 2).  That is a coincidence of two independently published
two-digit constants, NOT a derivation of Deardorff's asymptote, and the
literal 0.19 therefore STAYS the registered spelling.  Spelling C_ES as
C_KS*LS_COEF would make a published constant a silent dependent of this
module's own diffusivity calibration -- the same false coupling
``C_MOM_BG``'s comment already warns about one screen down -- and it
would hollow out the anti-tuning tripwire, whose
``C_ES/LS_COEF == 0.25`` assertion would degrade into a restatement of
C_KS.  The identity is registered as ALGEBRA (the Gamma_m = 1 line
above), never as source.

S3-6j amendment (surface momentum stress in the vertical solve;
missing-friction diagnosis, Probe-4 falsification, controller ledger
2026-07-21, registered before implementation).  THE HOLE: ``ust`` fed
ONLY the surface e source (``physics.sase_surface_e_source``); the
momentum Thomas columns ran zero-flux at BOTH ends, so the resolved
wind felt NO surface drag anywhere in the SASE path -- the whole
column accelerated freely under PGF forcing (Probe 4: jet slab +19%
over the hour, 10 m drift 1.91x vs obs 1.27x, unmoved by an engaged
~2.7x stable-coefficient mixdown, because no force opposed the
acceleration).  The fix is the standard implicit surface-stress
bottom boundary condition, exactly YSU's pattern (authority
``npref.np_ysu_column``: ``fric = ust*ust/wspd1 * rho*G/delp[0]*dt2
...; diag[0] = 1.0 + fric`` -- npref.py:6495-6497, transcribing WRF
bl_ysu tridi's implicit surface stress; kernels/ysu_default.cu:772-774 is the
device twin):

    tau      = u*^2 * (u1, v1)/|V1|          [surface kinematic stress]
    F_bottom = -(u*^2/|V1^n|) * u1^{n+1}     [implicit linearization]
    c        = u*^2 / max(|V1^n|, SFC_WSPD_FLOOR)
    diag_0  += dt*c/thick_0                  [folded into the Thomas
                                              bottom row: I - dt*D_v
                                              + dt*(c/thick_0)*P_0]

with |V1^n| the PRE-STEP level-1 speed (the linearization's frozen
coefficient, YSU's wspd1 convention) floored at SFC_WSPD_FLOOR (the
audited sfclay floor ``wspd = max(..., 0.1)``, npref.py np_sfclay /
kernels/sfclay.cu, transcribed as a registered constant).  The
augmented matrix adds a POSITIVE diagonal term, so it remains a
strictly diagonally dominant M-matrix: unconditionally stable for any
dt, and the solution magnitude is bounded by the input extrema (the
drag can only pull the bottom cell toward rest).  Scope (each choice
documented): u and v ONLY.  w keeps zero-flux (the stress is
horizontal; the kinematic surface BC owns w).  The scalars keep
zero-flux (their surface fluxes genuinely arrive through the
sfclay/Noah HFX/QFX pathways -- a drag row there would double-count).
e-transport keeps zero-flux (its surface source is the existing
``sase_surface_e_source``; see the ledger restatement below for why
that source IS this drag's energetic partner).  The drag applies at
ALL f -- friction is not regime-dependent -- making this the lane's
ONE intentional cross-limb (f = 1) change: the f = 1 reduction pins
move wherever a fixture passes ``ust`` (re-pinned with rationale);
``ust=None`` remains bitwise the S3-6i formulation (no drag term is
ever formed).

RESTATED LEDGER THEOREM (S3-6j).  With ``ust`` given, the u/v
implicit solves ride (I - dt*D_v + dt*(c/thick_0)*P_0) with P_0 the
bottom-row projector.  On the uniform-dz box the implicit pairing now
telescopes to the bottom-face flux instead of zero:

    dKE_impl = sum_i u_i^{n+1}.(u_i^{n+1} - u_i*)
             = -dt*sum(P_v) + dKE_sfc,
    dKE_sfc  = -dt*sum_cols (c/thick_0)*((u1^{n+1})^2 + (v1^{n+1})^2)
             <= 0                       [the MEASURED drag work]

(derivation: the bottom-row equation gains -dt*(c/thick_0)*u1^{n+1},
so u^{n+1}.(u^{n+1} - u*) = dt*u^{n+1}.D_v u^{n+1}
- dt*(c/thick_0)*(u1^{n+1})^2; the first term telescopes to -dt*P_v
exactly as before -- interior faces are untouched).  The closure
statement becomes BOUNDARY-CONSISTENT:

    dKE + dE + dHeat = dKE_sfc,
    residual := dKE + dE + dHeat - dKE_sfc   [closes to roundoff]

-- the identity's right side is exactly the resolved-KE flux through
the bottom face, which leaves the ledger's control volume.  PHYSICAL
DESTINATION (documented, diagnosed, NOT forced closed): the KE the
drag removes converts to near-surface turbulence -- which is exactly
what the EXISTING driver-side source u*^3/(kappa*0.5*dz1)
(``sase_surface_e_source``, deposited into e BEFORE the step) models.
The two are the two ends of one physical flux: resolved KE ->
surface-layer shear production of e.  They do NOT match numerically:
the source is the neutral-similarity production integral over the
sub-half-cell layer at the frozen u*, while dKE_sfc is the resolved
drag work at the solved level-1 wind (a similarity MODEL vs a
measurement).  The mismatch is recorded per step as the DIAGNOSED
channel

    dE_sfc_src    = dt*sum_cols u*^3/(kappa*0.5*thick_0)   [modeled]
    sfc_conv_resid = dE_sfc_src + dKE_sfc    [modeling residual;
                                              diagnostic only, never
                                              enters the identity]

and deliberately left open -- forcing it closed would replace the
audited similarity source with the resolved drag work, a different
(and grid-sensitive) model.  With ``ust=None`` all three channels are
exactly 0.0 and ``residual`` reduces bitwise to the S3-6i statement
(x - 0.0 == x).

S3-9 amendment (GEOMETRIC dissipation-length regime blend; F-Y1
cold-water-body over-coupling, column-mechanism investigation
recorded with the campaign evidence, controller ledger
2026-07-21).  ROOT CAUSE (measured on the intermediate-nest run): the
LINEAR l_d blend rode the DOMAIN-LEVEL f.  As the land CBL deepened,
the domain f_used rose to 0.12-0.19 and every stable marine column
inherited the convective land's partition, handing the blend a
linear term f*delta = 360-580 m that JUMPS PAST the Blackadar/BL89
dissipation bound entirely; the only remaining cap,
l_s = LS_COEF*sqrt(e)/N, is state-dependent and inflates with the
very e it should limit (l_s doubles per 4x in e, and diverges as the
mixing destroys N), so production exceeded dissipation (P/D
1.9-7.1) until e equilibrated an order of magnitude high: e =
1-2.5 m^2/s^2 (restart truth 2.03), K_v = 25-112 m^2/s through the
jet face over 16-K-colder water.  The -250 W/m^2 surface cooling
never formed an IBL, every stability suppression this module owns
(l_s, BL89, C_KS, the Pr blend) keys on an N^2 that never forms, and
the cold-water-body surface welded to the jet aloft (sp10/sp500
0.65-0.94, growing with fetch).  THE FIX -- the regime blend of
:func:`dissipation_length` moves to log space:

    l_d = min(delta**f * l_eps_rans**(1-f), l_s)
          [was  min(f*delta + (1-f)*l_eps_rans, l_s)]

with l_eps_rans = min(l_B, l_eps_BL89) the untouched S3-6h
composition, l_s the untouched outer min, f the step's used
partition weight, and the ``lb=None`` (v0/LES) branch untouched.
ZERO new constants.  A length-scale blend belongs in log space: the
geometric form keeps l_d within a factor (delta/l_eps_rans)**f of
the RANS bound (~3.6x at f = 0.19, l_eps = 3 m -- vs 175x for the
linear form) while remaining EXACTLY the v0 LES length at f = 1.
ENDPOINTS FP-EXACT AND PINNED: f = 0 gives delta**0.0 * lb**1.0 =
1.0*lb == lb bitwise (every RANS column fixture -- jet, inversion,
log layer, Ekman -- unchanged bitwise), f = 1 gives delta**1.0 *
lb**0.0 = delta*1.0 == delta bitwise (the LES-limb reduction pins);
monotone (log-linear) in f between them.  EXPONENTIATION DOMAIN:
both lengths are strictly positive wherever the blend is formed --
delta is a positive filter width and the live lb = min(l_B,
l_eps_BL89) has l_B > 0 at layer centers (z >= thick_0/2 > 0 even at
z0 = 0) and BL89 lengths strictly positive by the E_MIN floor
(:func:`bl89_displacement_lengths` docstring) -- documented rather
than asserted, the module convention.  THE BLEND WEIGHT DELIBERATELY
STAYS THE DOMAIN f_used: the geometric form makes the domain/column
partition mismatch benign (l_d stays O(l_eps_rans) at small f), so a
per-column partition is NOT required here and remains open as a
separate gray-zone refinement.  The S3-6f cap's Gaussian-form
rationale was stated against the linear blend and is preserved
above as the historical record; the cap itself stands unchanged.
LEDGER THEOREM UNAFFECTED: l_d is a coefficient, never a channel --
it enters the step ONLY through the analytic decay rate b =
C_E*sqrt(max(e*, E_MIN))/(2*l_d), and the decay decrement D is
DEFINED as e* - e*/(1 + b*dt)^2 for whatever l_d the step computed,
so the e-channel closes by construction and the production/
dissipation pairing (identities (i)/(ii)) never sees l_d: every
S3-6d/6e/6j closure statement holds verbatim (asserted by the
ledger-closure fixtures under the geometric blend).  DELIBERATE
RE-PINS (the linear formula's own pins, the only two; derivations at
their sites): the interior-f closed form of
test_dissipation_length_blend_les_and_rans_limits and the split-step
trajectory goldens (tests/sase_goldens.py; BOX e 7.57e-4 rel, COL e
1.81e-4 -- the interior-f fixture).  RED/GREEN evidence (the
water-surface jet-decoupling fixture, tests/test_sase.py, profile
table, the cold-water profile fixture module): GREEN holds the 13Z onset
column in the land band (sp10/sp500 0.539 in [0.35, 0.60], e below
500 m 0.011 <= 0.3) under the reconstructed domain-f schedule, and
the frozen-wind 18Z coupled state builds the 12.8 K IBL (>= 10 K
band) the run never formed; RED (linear blend pinned back) shows the
throttled equilibrium (IBL 2.75 K <= 4 K, e 2.21 >= 1.0 below
400 m) -- the defect this amendment removes.  Registered decision
string: LD_BLEND_FORM = "geometric" (config-ID-bound).  The device
twin ``launch_sase_step`` mirrors the blend in S3-9b; until then the
GPU parity suite mismatches the authority BY DESIGN.

S3-9c amendment (GUSTINESS-CORRECTED surface drag; the S3-6h/6i/6j
adversarial review IMPORTANT-1, task S3-9c).  THE DEFECT: sfclay's
u* is computed against its gust-ENHANCED speed
wspd = max(sqrt(wspd0^2 + vconv^2 + vsgd^2), 0.1) (convective
velocity + subgrid gustiness; authority ``npref.np_sfclay``,
npref.py:4257-4266, device twin kernels/sfclay.cu:221-231), but the
S3-6j drag row applied that u* against the RESOLVED level-1 wind
alone -- over-damping every column where gustiness inflates u* (the
review-measured SASE/YSU stress ratio exceeded 2 in 10.3% of
outer-nest interior cells at 13Z, 95th percentile 9.54 -- the
registered outer-nest low-side/calm-tail bias of the first 12 h run).
THE AUDITED FORM:
YSU's momentum bottom row multiplies the drag by the
resolved-over-enhanced ratio squared,

    fric = ust*ust/wspd1 * rho*G/delp[0]*dt2
           * (wspd1/max(wspd, 1.0e-9))**2     [npref.py:6495-6496]

with wspd1 = hypot(u1, v1) + 1e-9 the resolved level-1 speed
(npref.py:6145) and wspd the live sfclay enhanced speed.
TRANSCRIPTION (geometric-height conductance): the S3-6j row keeps
its registered base and gains exactly that factor,

    c = u*^2/spd1 * (spd1/max(wspd, 1e-9))^2
      = u*^2*spd1/wspd^2,   spd1 = max(|V1^n|, SFC_WSPD_FLOOR)

-- identical to YSU's u*^2*wspd1/wspd^2 except for the PRE-EXISTING
registered SFC_WSPD_FLOOR regularization of the resolved speed (YSU
regularizes wspd1 with +1e-9 instead).  The two differ only where
|V1| < 0.1 m/s; there our conductance u*^2*SFC_WSPD_FLOOR/wspd^2
sits between YSU's exact u*^2*|V1|/wspd^2 and the uncorrected S3-6j
u*^2/SFC_WSPD_FLOOR (wspd >= 0.1 always), i.e. the calm-column
over-damping shrinks by the full (0.1/wspd)^2 <= 1 while the
divide-by-vanishing-wind guard survives.  SEAM DESIGN: ``wspd_sfc``
is an optional (ny, nx)-broadcastable input beside ``ust``.  ABSENT,
no factor is formed and the S3-6j arithmetic is BITWISE unchanged
(the no-gustiness identity); SUPPLIED with a no-gust value
wspd_sfc == spd1, the factor is (spd1/spd1)^2 = 1.0 and c*1.0 == c
bitwise (the identity pin the neutral no-gust fixture asserts).
``wspd_sfc`` without ``ust`` is rejected -- there is no drag row to
correct.  The driver threads the live sfclay ``wspd`` field
(physics._run_sase; sfclay refreshes it in the same due step, before
the split step, exactly as it refreshes ust).  LEDGER UNTOUCHED: the
factor folds into the conductance c BEFORE every consumer -- the
Thomas bottom diagonal and the measured drag work dKE_sfc ride the
same corrected c -- so the boundary-consistent closure theorem holds
verbatim for any c >= 0.  The e source ``sase_surface_e_source``
deliberately keeps the uncorrected u*^3 similarity deposit (it
models the sub-half-cell production integral at sfclay's own u*
scale; sfc_conv_resid remains the diagnosed mismatch channel).

S3-11a amendment (surface scalar-flux deposit; G-LAKE residual root
cause, the cold-water momentum root-cause record in the campaign
evidence, adversarially
verified 2026-07-21, zero refutations).  THE HOLE: sfclay's surface
sensible/latent heat fluxes never reached the atmosphere.  The
S3-6j scope note above recorded the premise "their surface fluxes
genuinely arrive through the sfclay/Noah HFX/QFX pathways" -- that
premise is FALSE (the note stands above as the historical record):
sfclay only DIAGNOSES HFX/QFX, Noah consumes them in the GROUND
budget, the e source consumes them for TKE production (discarding
the stable part), and NO code path deposited them into theta/qv of
the lowest model level -- the scalar vertical channel ran zero-flux
at the ground.  Over the cold water body the -80..-224 W/m^2 downward
HFX therefore never cooled the marine air: the surface-based stable
layer could not form, N^2 stayed ~0 in the lowest ~100 m, the
vertical channel kept its neutral-wall coefficient, and jet
momentum mixed continuously to the surface (sp10/sp500 = 0.73 vs
the 0.30-0.65 physical band); the warm-sector qv likewise had no
surface evaporation source (the P2 Td2 miss, the moisture face of
the identical defect).

THE SEAM (:func:`surface_scalar_flux_deposit`): the standard
atmosphere-side deposit -- the audited YSU surface-rhs rows
(``npref.np_ysu_column`` heat_surface/q_surface, npref.py:6472 and
:6481: ``hfx/(CP/G)/delp[0]*dt2`` and ``qfx*G/delp[0]*dt2``, which
under delp[0] = rho*g*dz1 are exactly the increments below) --
applied as an EXPLICIT deposit into the lowest layer BEFORE the
implicit K_v/Pr_t vertical solve of the scalar channel:

    theta[0] += dt*HFX/(rho1*CP_AIR*thick_0)
    qv[0]    += dt*QFX/(rho1*thick_0)

with HFX [W m^-2] / QFX [kg m^-2 s^-1] the SAME fresh post-sfclay
fields the e source consumes and rho1 the same lowest-level moist
density the e source computes (``physics.sase_surface_e_source``;
the driver threads all three in S3-11b).  EXPLICIT over IMPLICIT
(registered choice; the root-cause file's fix candidate 1 over
candidate 2): the implicit bottom-flux-row variant is identical
physics but extends the Thomas assembly; the explicit deposit has
the smaller blast radius, and stiffness is not a concern at the
steps that consume it -- the deposit magnitude at the pinned defect
state is |dtheta_1| = dt*|HFX|/(rho1*CP_AIR*thick_0) ~ 0.14 K per
intermediate-nest step (dt = 15 s, HFX = -180.2 W/m^2,
rho1 = 1.174 kg/m^3,
thick_0 = 16.8 m), bounded, and the unconditionally stable implicit
solve runs immediately after it.

RESTATED LEDGER (S3-11a).  The split-step theorem is UNTOUCHED and
holds VERBATIM: theta enters :func:`sase_split_step` READ-ONLY (the
buoyancy coefficient field and the BL89 lengths), the deposit runs
OUTSIDE the step in the scalar channel, and no KE/e/heat channel
changes -- dKE + dE + dHeat = dKE_sfc exactly as stated in S3-6j
(re-pinned beside the live seam by the S3-11a ledger fixture).
What the column GAINS is a real heat/moisture source, and its
closure is the scalar twin of the S3-6j boundary-consistent form.
For the composed scalar update phi -> implicit_solve(deposit(phi))
on any zero-flux column -- uniform dz OR dz_col: unlike the KE
pairing, this closure needs no uniform spacing --

    dTH := sum_k thick_k*(theta^{n+1} - theta^n)_k
         = dt*HFX/(rho1*CP_AIR)            per column, exactly,
    dQV := sum_k thick_k*(qv^{n+1} - qv^n)_k
         = dt*QFX/rho1                     per column, exactly,

because (i) the deposit changes sum(thick*theta) by
thick_0*(dt*HFX/(rho1*CP_AIR*thick_0)) = dt*HFX/(rho1*CP_AIR) --
the thickness weight cancels the increment's 1/thick_0 -- and
(ii) the zero-flux implicit solve telescopes its face fluxes to the
(zero) end fluxes, conserving sum(thick*phi) to solver roundoff
(the pinned conservation property of
:func:`implicit_vertical_diffusion`).  The right sides are exactly
the physical boundary fluxes through the ground face: the
atmosphere's heat/moisture content changes by exactly what crossed
the surface, nothing hides in a residual, and the theorem is
STRENGTHENED (a new exact closure statement) rather than weakened
(pinned by the ledger fixture at rtol 1e-9 of the increment --
~1e-15 of the column content, pure FP64 roundoff).  ZERO-FLUX
IDENTITY: hfx = qfx = 0.0 (the DEFAULTS) adds literally +0.0 to the
bottom row -- x + 0.0 == x bitwise for every finite x except -0.0
(physical theta > 0, qv >= +0.0) -- NO existing code path calls the
seam, and no existing function's body changed, so every pre-S3-11a
fixture and golden is bitwise-untouched (pinned by the zero-flux
identity fixture).  NO RETUNING: every SASE constant (C_KV, C_KS,
the BL89 lengths, the drag BC, z0 handling) is untouched -- the
closure produced the correct K_v collapse when handed a cooled
profile (the root-cause single-column reproduction); it behaved
correctly for the inputs it received, so the fix is the missing
forcing, never the closure.  MEASURED (the S3-11a lake fixture: the
18Z coupled state under the observed fixture-point
HFX = -180.2 W/m^2, frozen wind, f = 0, dt = 15 s): N^2(k0)
crosses 1.0e-3 s^-2 at 2.75 min and reaches 1.37e-3 by 10 min,
K_v(26 m) collapses 11.3 -> 0.19 m^2/s (59x) by 10 min (the
first-step 11.3 rides the run's coupled-equilibrium restart e on
the still-neutral slab), theta_1 cools 2.7 K -- against the
hfx = 0 leg's N^2 ~ 8.3e-6 and K_v ~ 2.97 (15x separation),
matching the reproduction's mechanism and magnitude (N^2 3e-5 ->
1.6e-3; K_v 3.74 -> 0.26, 14x, from its own spun state).
Registered decision string: SFC_SCALAR_FLUX =
"explicit-deposit-v1" (config-ID-bound, with CP_AIR).  The device
twin and the driver wiring (``physics._run_sase`` scalar loop) are
S3-11b scope; until then the production run lacks the seam BY
DESIGN.

SASE-M1 amendment (moist stability core; SASE-M spec
docs/internal/specs/2026-07-22-sase-m-design.md section 3, plan
Task S4-1, registered before implementation).  THE DEFECT (confirmed
by experiment, spec section 1): at gray-zone dx a saturated boundary
layer that is dry-stable (+4..+9 K/km) but MOIST-unstable (theta_es
falling 4-22 K through 0.4-2.5 km) is abandoned by the closure --
every stability pathway in this module keys on the DRY
N^2 = (g/theta) d(theta)/dz, which reads the layer as stable
(1.3-2.0e-4 s^-2 in the 11Z amplifier specimen), so the l_s/C_KS
stable-limit strangle holds K at 0.00-0.02 m^2/s and TKE at the
E_MIN floor (early-ci-cap-audit.md) while the trusted MYNN reference
sustains TKE 0.5-1.6 m^2/s and K_h 3-40 m^2/s in the identical layer
(sase-m-target-envelope.md).  The instability's only outlet is the
resolved scale: x11.5/h areal amplification, premature CI.

THE FIX: in SATURATED air the stability machinery evaluates the
Durran-Klemp saturated moist Brunt-Vaisala frequency N^2_m instead
of the dry N^2.  Durran, D. R. and J. B. Klemp, 1982: On the effects
of moisture on the Brunt-Vaisala frequency.  J. Atmos. Sci., 39,
2152-2158, their Eq. (36):

    N^2_m = g * [ (a/b) * (d ln(theta)/dz + (L/(cp*T)) * dqs/dz)
                  - d(qw)/dz ]
    a = 1 + L*qs/(Rd*T),      b = 1 + eps*L^2*qs/(cp*Rd*T^2),
    eps = Rd/Rv,  qw = qv + qc  (total water),  qs = qs,liq(T, p).

DERIVATION (the algebra; e_s << p and q << 1 class approximations,
DK82's own).  (1) Saturated-adiabatic lapse: a saturated reversible
parcel obeys the first law with condensation heating,
cp dT + g dz + L dqs = 0.  With qs = eps*e_s/p (e_s << p) and
Clausius-Clapeyron d(e_s)/e_s = L dT/(Rv T^2), the parcel's
saturation gradient along hydrostatic ascent (dp/dz = -p g/(Rd T))
is dqs/dz = qs*L/(Rv T^2) dT/dz + qs*g/(Rd T).  Substituting and
collecting dT/dz (using L^2 qs/(Rv T^2) = eps L^2 qs/(Rd T^2)):

    dT/dz * (cp + eps*L^2*qs/(Rd*T^2)) = -g * (1 + L*qs/(Rd*T))
    =>  -dT/dz = Gamma_m = (g/cp) * a/b.

(2) The bracket identity: for the saturated ENVIRONMENT, using
d ln(theta)/dz = (1/T)(dT/dz + g/cp) and the same environment
saturation gradient dqs/dz = qs*L/(Rv T^2) dT/dz + qs*g/(Rd T),

    Q := d ln(theta)/dz + (L/(cp*T)) dqs/dz
       = (1/T) [ dT/dz * (1 + eps*L^2*qs/(cp*Rd*T^2))
                 + (g/cp) * (1 + L*qs/(Rd*T)) ]
       = (b/T) * [ dT/dz + Gamma_m ],

so (a/b)*Q = (a/T)*(dT/dz + Gamma_m): the Eq.-36 thermal term is
exactly the environment-minus-parcel temperature-gradient difference
(the parcel follows -Gamma_m), amplified by a = 1 + L*qs/(Rd*T) --
the factor carrying the induced saturation-vapor buoyancy difference
(DK82 section 2).  (3) Loading: the displaced parcel conserves total
water q_w (reversible, non-precipitating -- the spec's own scope),
so the parcel-vs-environment condensate+vapor loading difference per
unit displacement is the environment gradient -d(qw)/dz.  Summing
(2) and (3) gives Eq. (36).  EXECUTABLE VALIDATION (the fixture
set): the moist-adiabat-neutrality fixture integrates the q_w-const
saturated-adiabat ODE d ln(theta)/dz + (L/(cp T)) dqs/dz = 0
independently of the transcription and demands |N^2_m| <= 1e-6 with
dry N^2 > 1e-4 (measured interior residual 3.9e-10); the moist-lapse
witness pins -dT/dz == (g/cp)*(a/b) on the same column within 1%
(measured 0.28% -- a swapped a/b errs 2.8x, a missing eps in b
~40%); the condensate-loading witness pins the -g*d(qw)/dz term
EXACTLY (linear-ramp shift, measured 1e-18 class).

DISCRETE FORM (:func:`moist_n2`; authority op order, the device
mirror's parity target): T = theta*(p/P0_REF)**(RD_AIR/CP_AIR);
e_s = 1000*SVP1*exp(SVP2*(T - SVPT0)/(T - SVP3)) [Tetens liquid --
the model's own saturation constants, the SAME e_s that formed the
qc the switch consumes]; qs = EP2_RV*e_s/(p - e_s); every vertical
derivative rides the SAME clamped variable-``dz_col`` stencil as
every SASE vertical operator (:func:`_ddz_var` -- quadratic-exact
interior, one-sided linear-exact edges).

SATURATION SWITCH (registered convention MOIST_STABILITY_SWITCH =
"binary-qc-or-rh100-liquid"): a cell is saturated when qc > 0 OR
qv >= qs,liq(T, p) (monotone-equivalent to RH >= 100% with respect
to liquid).  v1 is the spec's binary cell test; the assumed-PDF
cloud-fraction blend is the registered later hook (spec section 3
"the exact blend is the implementer's to propose") and must
re-register the string when it lands.  The switch output feeds a
np.where whose FALSE branch is the LITERAL dry
:func:`brunt_vaisala_n2` field, so unsaturated cells carry the dry
bits VERBATIM -- the M1 unsaturated bitwise-identity contract
(pinned by the control-column fixture: a fully unsaturated column
through the M1 pipeline is tobytes()-identical to the pre-M1 path).

SUBSTITUTION POINTS (spec section 3 -- exactly these, nowhere else).
:func:`sase_split_step` gains the optional ``n2_moist`` seam (None =
bitwise the pre-M1 step; requires ``n2``).  With the seam engaged,
n2_eff = n2_moist feeds

    1. the stability length l_s = LS_COEF*sqrt(e)/N -- every l_s
       evaluation site: :func:`vertical_mixing_length` (the LES limb
       l_les and the l_s term of the RANS l_mix composition via
       :func:`bl89_rans_lengths`) and the outer l_s min of
       :func:`dissipation_length`;
    2. the buoyancy production/destruction term of the e budget:
       where the moist field DEPARTS from the dry field (subst =
       n2_moist != n2 -- moist_n2 constructs the unsaturated branch
       bitwise-dry, so inequality identifies exactly the cells where
       the moist closure claimed authority; a saturated cell whose
       DK82 value coincides bitwise with the dry value is
       substitution-inert by definition), the term becomes
       -(K_v/Pr_t)*n2_moist; elsewhere the LITERAL dry expression
       -(g/theta)*(K_v/Pr_t)*ddz(theta) stands unchanged (bitwise);
    3. the stability suppression of K_v/K_h -- the same N^2-keyed
       pathway as today: :func:`stable_limit_coefficient` (rho, C_r)
       and the l_s bounds of point 1 riding n2_eff.

NOT a substitution point: the S3-6f w-sensor gravity-wave screen
keeps the DRY n2 (it classifies resolved-w motion against the
free-troposphere dry-stratification benchmark N2_SCREEN; handing it
the moist field would widen M1 beyond the spec's three points --
asserted by the seam-contract fixture's spy).  BL89 displacement
integrals keep the registered dry-theta_v convention
(BL89_BETA_CONVENTION -- moist parcel thermodynamics is the M2/plume
scope, C10).  [S4-3b amendment: the DRY displacement integrals are
unchanged, but the SASE-M1b section below adds a SEPARATE moist
excursion family on the n2_eff field as an additional master-length
bound in substituted cells -- the spec-3b response to the G-M3
deck-clearing failure.]

LEDGER (C4/C5, sase-m-integration-points.md): the theorem holds
VERBATIM.  Points 1 and 3 change coefficients (K_v faces, l_d),
never channels -- every closure identity is pointwise in the
coefficients (the S3-6f/S3-9 rule).  Point 2 changes the VALUE of
the buoyancy source, which the dE definition EXCLUDES from the
closed triangle (dE = ... - dt*sum(gb) - ...; the PE-exchange
channel, S3-6e) -- a measured-channel change, not a closure change
(asserted by the moist-substitution ledger fixture at < 1e-11).

MEASURED on the reference 11Z amplifier specimen column (a saturated,
strongly capped continental column; the specimen, its fixture module
and its provenance are recorded with the campaign evidence, not here;
frozen-state column equilibrium at
f = 0, converged by 90 min): the deck's DK82 N^2_m runs
-8.9e-5..+1.0e-4 s^-2 against dry +0.66..+1.8e-4 (sign flips in 4
of 7 saturated cells); saturated-layer mean TKE 1e-6 -> 1.015
m^2/s^2 (INSIDE the G-M5 band [0.5, 1.6]) and mean K_h
2.5e-5 -> 102.6 m^2/s against the dry leg's 0.00-0.02 shortfall
(early-ci-cap-audit reproduced).
G-M5 K_h-CEILING DEVIATION (registered, carried as a strict xfail in
the suite): the plan band [3, 40] m^2/s is NOT reachable by this
in-scope formulation -- at in-band TKE the equilibrium K_h =
C_KV*l*sqrt(e)/Pr_t is set by the module's shortest registered
length in the deck, Blackadar l_B = 82-114 m at 456-1391 m AGL
(the BL89 dry-parcel lengths are LONGER there once the moist release
removes l_s), whereas the reference's 3-40 m^2/s class carries the
MYNN master length ~10-20 m in-deck (its TKE-integral/harmonic
blend, a length family this module does not own).  Reaching the
ceiling would need new length physics (a fourth substitution point)
or case tuning, both scope violations; the deviation is pinned,
measured, and handed to the task review + the S4-3 smoke (G-M3/G-M5
on the 3-D run, where supply/radiation self-limiting operates).
Registered decision strings: MOIST_STABILITY = "dk82-saturated-v1",
MOIST_STABILITY_SWITCH = "binary-qc-or-rh100-liquid" (config-ID-
bound, with RD_AIR/RV_AIR/EP2_RV/XLV/SVP1/SVP2/SVP3/SVPT0/P0_REF).
The device twin (``launch_sase_step`` n2_eff) and the driver wiring
are S4-2 scope; until then the production run lacks the seam BY
DESIGN.

SASE-M1b amendment (moist master-length limb; SASE-M spec section 3b,
plan Task S4-3b, added under the FROZEN S4-3 adjudication rule "G-M3
FAIL -> spec amendment for a moist master-length limb, NEVER constant
adjustment").  THE DEFECT (measured, S4-3 smoke,
the sase-m1-residual record in the campaign evidence): M1 confined the
amplifier (13->14Z 35-dBZ growth x1.15 vs the x11.5 baseline;
amp-conditional intermediate-nest LE 105-143 W/m^2, inside the
post-fix class) but FAILED G-M3 by
CLEARING the Sc deck (ampfrac 10.1% -> 4.6% vs reference 60-84%;
coverage <= 0.47 vs bar 0.8; LWP p50 < 130 g/m^2 by 14Z; SWDOWN
1.7-2.0x reference).  Mechanism: in a saturated moist-UNSTABLE layer
N^2_m < 0, so every l_s evaluation is inactive and the master length
entering the dissipation/K pathway falls back to the FREE
(dry-convective) length -- at f = 0 the Blackadar/dry-BL89 class
(l_B = 82-118 m in-deck; the deck-under-lid fixture pins l_mix ==
l_B bitwise at the lid-adjacent cell), at gray-zone f the delta**f
side of the geometric blend -- and eddy diffusion at the resulting
K_h ~1e2 m^2/s (measured deck p50 74-120, p90 136-150) entrains dry
air across the capping inversion and destroys the deck.  The trusted
reference sustains K_h 3-40 in the same deck because its master
length is bounded by the capping stability.

THE FIX (spec-3b envelope; the discretization is this module's):
in M1-substituted cells the RANS-limb composed lengths are
ADDITIONALLY bounded by a moist parcel-excursion length of the BL89
family,

    l_mix_rans -> min(l_mix_rans, l_m),  l_eps_rans -> min(l_eps_rans,
    l_m),  l_m = min(l_up_m, l_down_m)
    [MOIST_MASTER_LENGTH = "bl89-n2eff-excursion-min-v1"]

with (l_up_m, l_down_m) the :func:`bl89_moist_excursion_lengths`
up/down excursion integrals evaluated against the M1 moist buoyancy:
the integrand is R(z') = int N^2_eff ds from the parcel level, i.e.
the DK82 field of :func:`moist_n2` (condensate loading included) as
the stratification measure, which in the dry machinery's own terms is
exactly beta*(theta_v(z') - theta_v(z_k)) with N^2_beta substituted
by N^2_eff -- the same accumulate-until-exceed construction, the same
quadratic-per-segment quadrature (:func:`_bl89_first_crossing`), the
same E_MIN floor and geometric surface/top bounds.  A moist-unstable
deck spends nothing on the way up (N^2_m < 0 stretches contribute
negatively) and exhausts its budget INSIDE the moist-stable lid:
l_up_m = distance-to-lid + finite penetration, never the free
fallback.  The min member of the pair is the family's own l_eps
convention and the unique conservative choice for a master-length
CEILING (the defect is OVER-mixing: the K-side bound must never be
slacker than the tightest excursion direction).  Discrete form:
N^2_eff held piecewise-constant per segment at the arithmetic face
mean (the module's face convention) -- R piecewise linear, outer
integral exactly quadratic; constant-extension end segments carry
slope 0; the centered n2 stencil gives one-cell level bracketing at
the lid (fixture-pinned).  ZERO new tunable constants.

APPLICATION POINT (coefficients-not-channels): the bound lives inside
:func:`bl89_rans_lengths` behind the ``n2_dry`` seam -- None (every
pre-M1b caller) is bitwise the S3-6h composition; the split step
passes n2_dry = n2 exactly when the n2_moist seam is engaged, so the
limb rides into (a) K_v's RANS limb (1-f)*C_r*l_mix_rans*sqrt(e)
(C_r evaluated AT the bounded length -- the S3-6i blend keys on how
close the operating length sits to l_s, unchanged logic) and (b) the
l_d blend's RANS input lb = l_eps_rans.  IDENTITIES: unsaturated
cells keep their bits verbatim (np.where's FALSE branch; empty-mask
columns skip the machinery entirely -- poison-pinned by the control
and lake fixtures); f = 1 is bitwise the pre-limb path (the
two-product K_v blend multiplies the RANS limb by 0.0 and l_d's
lb**0.0 == 1.0, both FP-exact -- the S3-6h f = 1 argument, pinned by
the LES-inert fixture on a SATURATED column); the ledger theorem
holds verbatim (lengths are coefficients, never channels -- C4/C5).

MEASURED (the constructed deck-under-lid column, spec-3b fixture
class: saturated moist-unstable 0.4-1.4 km deck, N^2_m -8..-20e-5
while dry N^2 +7..+9e-5, capped by a +12 K inversion; frozen-state
equilibrium at f = 0): the lid-adjacent moist-unstable cell's master
length leaves the bitwise-l_B fallback (115.4 m) for the
distance-to-lid class (54.8 m = its upward excursion) and its
equilibrium K_h leaves the 1e2 class for the reference class
(101.9 -> 39.3 m^2/s) while deck TKE stays turbulent (mean 0.614 ->
0.561): the limb kills LID ENTRAINMENT -- the G-M3 mechanism -- not
deck turbulence.  On the 11Z amplifier specimen the limb is
measured SLACK (every moist excursion exceeds the operating length;
min margin 0.02 m at equilibrium): the 11Z deck is still strongly
dry-stable, so the S4-1 equilibrium (TKE 1.015, K_h 102.6) stands
BITWISE and the registered G-M5 K_h-ceiling deviation (strict xfail,
[3, 40] vs achieved ~102.6) is UNCHANGED by this amendment -- the
limb's bite lives on eroded (dry-neutralized) decks and lid-adjacent
cells, where the smoke's defect lives.  G-M3 adjudication is the
S4-3d smoke's; the device twin LANDED at S4-3c (`aba886a` feat(sase):
M1b device + driver moist master length) and the S4-3d smoke ran with
M1 + M1b live on the production path, so the interim "the production
device path lacks the limb BY DESIGN" note this paragraph carried is
RETIRED (spec-sweep ruling 12).

SASE-M2 amendment (conditional venting limb; SASE-M spec section 4,
plan Task S4-4, registered before implementation).  THE DEFECT
(measured; sase-m1-residual.md M1+M1b section, S4-3d adjudication):
M1 confined the broad-sector amplifier (growth x1.15 vs x11.5) but
the Sc deck is destroyed by UN-TERMINATED RESOLVED VENTING --
residual plume cores of conditional LE 0.61-2.28 kW/m^2 over
0.05-0.25% of the inner nest with a 1.9-2.7 km transport tail carrying
220-230 W/m^2 at the intermediate nest where the trusted reference
carries ~15 (x14), plus the
SWDOWN/HFX breakup feedback -- and the controlled M1b experiment
FALSIFIED eddy-diffusion entrainment as the deck-destruction channel,
so the fix is a transport/deposit-side limb: a DIAGNOSED per-column
plume (:func:`plume_vent_flux`; no new prognostic state) providing
non-local scalar transport exactly where the gray-zone amplifier
lives, with a hard neutral-buoyancy termination (kills the tail) and
a condense-don't-clear partition (returns the vented moisture to the
1.0-1.8 km deck band -- the G-M3 rescue channel, adjudicated at
S4-6).  Scalars theta/qv/qc only; non-precipitating; no plume
momentum (spec section 2 physics calls).  Registered decisions
(config-ID-bound): VENT_FORM = "nb-terminated-vent-v1",
VENT_ANCHOR_RULE = "cloud-base-face-standdown-v1".

MASK (VENT_MASK; trigger on STATE, never activity -- w/TKE/echo
triggers fire 3-5 e-foldings too late, amplifier-anatomy section 5):
the limb engages on the column's LOWEST contiguous run
(>= VENT_MIN_RUN_CELLS = 2 cells, S4-4 review Minor-1: registered,
previously prose-only) of MEMBER cells -- cells that are BOTH inside
the M1 saturation mask (the ``n2m_mask`` argument -- the switch is
never re-derived here; the M1 seam owns it, and it can only VETO) AND
saturated on the registered VENT_K_LID_MEMBERSHIP total-water test
(MEMBERSHIP below) -- that is moist-unstable under the registered
theta_es reading.  THE REGISTERED OBLIGATION
(S4-1 review): the spec's "saturated layer with d(theta_es)/dz < 0
through a finite depth" admits two readings, and the 11Z specimen
FLIPS between them -- over the member run k12..k15 the bulk (run-top
minus run-base) theta_es reading is -1.525 K (unstable) while
adjacent-level monotonicity FAILS (interior rise +0.222 K at
k12 -> k13).  (MEASURED round-4; over the wider M1 mask run k10..k16
the same flip holds with bulk -4.469 K and interior rises
+0.006/+0.222 K at k11 -> k12 / k12 -> k13 -- the flip is a property
of the specimen, not of which of the two runs the reading is
evaluated on.)  BOTH readings
are implemented behind VENT_MASK: "bulk-theta-es-v1" (CHOSEN) and
"per-level-theta-es-v1" (theta_es decreasing at every adjacent pair
of the run).  Bulk is chosen because it is the reading the plume's
own parcel CONFIRMS (the S4-1 consistency criterion): the root
parcel's buoyancy above its LFC rides the INTEGRATED theta_es
deficit from the root level, not the local sign of d(theta_es)/dz at
every interior pair -- and the per-level reading would VETO the very
specimen that measurably amplified x11.5/h.  The flip is pinned by
fixture so the convention can never drift silently.

MEMBERSHIP (VENT_K_LID_MEMBERSHIP = "qt-ge-qs-v1"; S4-4 review
round-3 for k_top, EXTENDED round-4 to the WHOLE structural layer):
a cell is a MEMBER of the venting layer iff it is inside
``n2m_mask`` AND total water qt = qv + qc >= qs.  The run's base
k_base, its contiguity, and its top k_top all ride that member flag;
the root k_r is then derived from the theta_es structure (ROOT
below).  ``n2m_mask`` is retained as a pure VETO -- the limb still
never engages where the M1 seam says unsaturated, and the seam is
never re-derived here (pinned by the poisoned-mask leg of
test_m2_mask_discrimination_bitwise) -- but it can no longer EXTEND a
run.  The veto is provably never BINDING when the caller passes the
M1 switch itself: qt >= qs implies qc > 0 or qv = qt - qc >= qs,
i.e. implies ``sat = (qc > 0) | (qv >= qs)``, so member == (qt >= qs)
there; measured true on every registered M2 column (round-4 probe
r4_p8).

DEFECT THIS FIXES (measured, S4-4 review rounds 1-4): the M1 mask's
``qc > 0`` limb fires on sub-1e-12 kg/kg condensate DUST -- on the
specimen, cell k16 (RH 95.48%, qv 0.4767 g/kg SHORT of qs) qualified
for ``n2m_mask`` SOLELY on qc = 5.68e-14 kg/kg (5.7e-11 g/kg,
numerically zero), and cell k10 on qc = 1.05e-13 kg/kg.  Round 3 moved
k_top alone onto the total-water test; the run's BASE, its contiguity,
and the ebar window still rode the raw mask, so round-off-scale
condensate still swung the amplitude.  MEASURED at HEAD 3367244 (the
round-3 build), specimen export/supply, against the THEN-frozen band
[0.4, 0.6] -- superseded by the round-5 re-derived pin
[0.3553, 0.5584] (BAND STATUS below); the in-band / OUT labels are the
round-3 build's own and are kept as the historical record:

    qc[9]  += 1e-12  ->  root k_r 10 -> 9    ->  0.43850  (in band)
    qc[10] -= 1e-12  ->  root k_r 10 -> 11   ->  0.38422  (OUT)
    qc[16] -= 1e-12  ->  ebar window shrinks ->  0.44666  (in band)
    qc[17] += 1e-12  ->  ebar window grows   ->  0.36918  (OUT)

THE ROUND-4 FORM: qt's margin to qs is O(0.01-0.5) g/kg
(4.478e-5 kg/kg is the smallest |qt - qs| anywhere on the specimen
column, 4.5e7 times a 1e-12 kg/kg perturbation) and theta_es is a
function of (theta, p) ONLY, so k_base, k_top, k_lid and k_r are all
INVARIANT under +-1e-12 kg/kg condensate shifts at every cell of the
column (98 perturbations, round-4 probe r4_p5).  The FLUXES are not
bitwise invariant and no accurate reading of the closure would make them
so: qt itself enters the entraining parcel, so the response is
CONTINUOUS and LINEAR in the perturbation -- max relative flux
movement 5.84e-10 at 1e-12 kg/kg (worst cell k10), and the response
scales exactly with epsilon over four decades: 3.045e-11 / 3.046e-10
/ 3.046e-9 / 3.046e-8 relative at eps = 1e-13 / 1e-12 / 1e-11 /
1e-10 kg/kg on cell k12 (round-4 probe r4_p8).  Cells outside the
plume's territory move the fluxes not at all (bitwise).  That is the
property the fixture test_m2_layer_structure_roundoff_insensitive
pins.  (An earlier revision of this section claimed the fluxes were
bitwise invariant "pinned by fixture"; both halves were false -- no
such fixture existed and no shift inside the territory leaves the
fluxes bitwise unchanged.)

On the specimen the member run is k12..k15 (k_top = 15; the M1 mask
run k10..k16 is wider at BOTH ends -- k10/k11 are subsaturated by
0.397/0.148 g/kg and carry only dust condensate, k16 by 0.477 g/kg).
k_lid = k_top + 2 = 17 (INVERSION BASE below) and k_r = 10 (ROOT
below), and every specimen flux is BITWISE identical to the round-3
build 3367244 -- verified face-by-face on all three rows across every
registered M2 column (round-4 probe r4_p4).

ROOT (spec section 4, "Root": the moist-instability source level =
the theta_es-decrease base / most-unstable parcel level, 0.2-0.7 km in
the specimen -- NOT the surface, C8), DEPTH-BOUNDED at round 5.
Discretely: k_r = the HIGHEST interior theta_es maximum at or below
the member run's own base k_base AND at or above the structural depth
floor k_base - (k_top - k_base) - 1 -- the level at which theta_es
stops increasing with height and the decrease layer the run sits in
begins, subject to the source layer being no deeper below the run's
base than the run's own depth.  theta_es depends on (theta, p) only,
so the root carries NO condensate dependence at all, which is the
point of moving it off the mask.  FALLBACK: when
theta_es never increases with height below k_base (no interior
maximum -- theta_es decreasing all the way to the ground, which is
every constructed deck column) the decrease layer has no interior base
and k_base itself IS the source level, the classical cloud-base root;
that fallback is what makes the round-4 form bitwise-neutral on the
deck fixtures.  Constant-free: only the already-computed ``thes``
array and the run's own base and top indices.  MEASURED (specimen):
the only interior theta_es maximum at or below k_base = 12 is k10
(theta_es 334.4085 K against 334.3545 below and 334.2976 above,
z = 455.7 m), and the floor is 12 - 3 - 1 = 8, so k_r = 10 -- the
same root the pre-round-4 mask run supplied, now derived from a
condensate-free quantity and depth-bounded.

DEPTH FLOOR, WHY (design doc SASE-M2 amendment "root / anchor
separation", clause 1; the round-4 verification lane's real-field
survey): an unbounded theta_es-peak root violates the spec's own "NOT
the surface" clause on real fields and lands below 100 m on 6.49% /
6.09% of firing intermediate-nest columns (11Z / 13Z, probe
v5_p2_pop.py); those
columns exported a median 1.0142x the frozen supply at 11Z (0.2968x at
13Z) with 31.9% / 37.4% of them in the over-strength class below,
against a whole-population median of 0.4131 / 0.3667.  With the floor,
the sub-100-m fraction is 1.15% / 3.80% and the over-strength class
(export >= 2.5e-4 kg/m^2/s, the spec's ~4x-too-strong G-M4-killing
class) falls from 3.13% / 5.67% to 0.80% / 3.67%.  A source layer
deeper than the layer it feeds is not a plume source.

AMPLITUDE (column-local state ONLY -- C7; the closure signature
carries NO surface flux or w* input at all -- C8: the specimen
amplified at HFX -9..-28 W/m^2 while the reference confined with its
EDMF mass flux OFF, MAXMF = 0 everywhere), ANCHORED AT CLOUD BASE
(round 5):

    M_base = VENT_MB_COEF * rho1 * sigma_w,
    sigma_w = sqrt(VENT_SIGW_SHARE * ebar),

ebar the thickness-weighted mean e_sgs from the SATURATED-LAYER BASE
k_base through the entrainment-zone cell k_top + 1, i.e.
k_base <= k < k_lid (the BL-integrated subgrid energy, E_MIN floor).
Both ends are condensate-noise-immune indices (MEMBERSHIP above),
which is what makes the AMPLITUDE noise-immune.  THE ANCHOR IS THE
CLOUD BASE, NOT THE ROOT (design doc amendment, clause 2): this is
classical mass-flux practice -- M_b is specified at cloud base, which
is the provenance VENT_MB_COEF already carries (Grant 2001) -- and it
stops ROOT DEPTH from multiplying the amplitude, the shared mechanism
behind both defects the amendment names.  The round-4 window ran from
k_r; MEASURED on real intermediate-nest columns with real TKE_SASE
whose base sits
>= 2 cells above the root (n = 9046 at 11Z, 8089 at 13Z, probe
v5_p3_window.py) the round-4 amplitude is a median 5.81x / 2.76x this
one, p95 35.98x / 57.32x, max 401.8x / 469.3x -- a spread no
registered column can see, because e_sgs is uniform over the plume's
territory on every one of them (the deck fixtures by construction;
the specimen's M1-equilibrium e is flat to 0.4% over k10..k16).  On
the specimen the anchored window is k12..k16 against the round-4
k10..k16, worth a factor sqrt(1.016596/0.996808) = 1.00990 in
amplitude; the shape normalization (SHAPE below) moving from z_f[11]
to z_f[12] is worth 1/1.081185, and the two together take the export
from 0.814e-4 to 0.76074e-4 kg/m^2/s.  VENT_MB_COEF = 0.03 is
Grant (2001, QJRMS 127, 407-421), the cumulus-BL cloud-base
mass-flux closure M_b = 0.03*w*, transcribed with the velocity scale
re-keyed from the surface-similarity w* (identically zero in the
negative-HFX regime and forbidden as the sole amplitude source) to
the column's own subgrid turbulent velocity sigma_w = the isotropic
w-share of the layer TKE (VENT_SIGW_SHARE = 2/3: e = (3/2)*
sigma_w^2).  In a convective BL e ~ 0.5*w*^2 puts sigma_w ~ 0.6*w*,
so the re-keyed form is Grant's closure evaluated on MEASURED
turbulence instead of a surface-flux estimate -- the coefficient is
Grant's verbatim, NOT fit to this case (registered deviation: the
velocity-scale substitution is the C8 requirement).  REGISTERED
BRANCH (S4-4 review Minor-2): the re-key is applied LITERALLY,
M_base = VENT_MB_COEF*rho1*sigma_w -- NOT M_base =
VENT_MB_COEF*rho1*(sigma_w/0.6), the alternative branch that would
explicitly compensate for the sigma_w ~ 0.6*w* approximation used
above to justify the substitution.  The two branches differ by the
factor-1.67 ratio 1/0.6 and are NOT interchangeable: the literal
branch is the one whose export number is measured in-band (fixture
:func:`test_m2_velocity_scale_rekey_branch_pinned`); the compensated
branch over-vents (out of the [0.7, 1.1]e-4 export class).  The
sigma_w ~ 0.6*w* relation is cited ONLY to explain why the literal
substitution is a reasonable evaluation of Grant's closure on
measured turbulence -- it is NOT algebraically applied as a second
factor, and the chosen (literal) branch is what ships.  ``rho1`` is
the S3-11a lowest-level moist density -- the seam's own density
convention, single-sourced with the surface deposit; strictly
positive, rejected otherwise.

ASCENT: entraining moist parcel conserving (theta_l, q_t) under the
eps ~ 1/z entrainment family, eps = VENT_ENT_COEF/z with
VENT_ENT_COEF = 0.4 = c_eps of Siebesma, Soares & Teixeira (2007,
JAS 64, 1230-1248), Eq. (16) ("...can be well fitted with [Eq. 16]
with c_eps ~ 0.4", p. 1236) -- S4-4 review Important-1 CORRECTION:
the value shipped at S4-4 authority (0.55) was misattributed to this
paper; 0.4 is the paper's own constant (verified from the PDF) and
0.55 has no located primary source.  Their approach-to-top member
c_eps/(z_top - z) is deliberately dropped -- the hard NB termination
plus full in-column detrainment below it supersede the divergence
that term models (registered simplification).  Launch state = the
ENVIRONMENT of the root cell.  Discrete update per center-to-center
segment with the
environment held at the arithmetic face mean (the module's face
convention):

    phi_p(k) = phi_f + (phi_p(k-1) - phi_f) * (z_{k-1}/z_k)**c_eps,

the EXACT solution of d(phi_p)/dz = -eps*(phi_p - phi_f) for
eps = c_eps/z and constant phi_f (int eps dz = c_eps*ln(z_k/z_{k-1}) --
the BL89 segment-exact quadrature discipline; c_eps = VENT_ENT_COEF
throughout this docstring -- S4-4 review round-3 Minor-3: the
lowercase "c_e" spelling used pre-round-3 collided with the module's
OWN C_E = 0.93 Deardorff dissipation constant, an unrelated symbol;
retired).  Parcel
thermodynamics REUSE the M1 primitives: the SAME Tetens/Exner
expressions and constants as :func:`moist_n2` (the e_s that formed
the qc the mask consumes), theta_l = theta - (L/(cp*Pi))*qc, and a
fixed-iteration (VENT_SAT_ADJUST_ITERS = 12, S4-4 review Minor-1:
registered, previously prose-only) Newton saturation adjustment
(:func:`_vent_saturation_adjust`, derivation there).  Buoyancy is
the LOADED moist theta_v in parcel form,

    B = g*(theta_v_p - theta_v_env)/theta_v_env,
    theta_v = theta*(1 + (Rv/Rd - 1)*qv - qc),

condensate loading included -- the DK82/M1 buoyancy convention
(:func:`moist_n2` is its stratification-measure form); the binary
saturation adjustment is the registered v1 stand-in for the
assumed-PDF hook, exactly the MOIST_STABILITY_SWITCH contract.

TERMINATION (hard NB -- the cap-preservation and CI-fidelity
mechanism, C9): LFC = the first center above the root, BELOW the
diagnosed inversion base (see INVERSION BASE below), with B > 0; NB
= the first center above the LFC, at or below the inversion base,
with B <= 0; every flux face at and above the NB cell's BOTTOM face
is exactly +0.0 -- one-cell level bracketing of the physical lid,
exactly the M1b excursion pin's convention.  STAND-DOWN: no LFC
below the inversion base, or the inversion base itself is beyond
VENT_DEPTH_CAP = 4000 m above the root, returns bitwise zeros -- a
parcel that would punch the lid stands down and leaves deep CI to
the grid (G-M6; VENT_DEPTH_CAP is the shallow-device scope guard of
amplifier-anatomy constraint 8, "detrainment capped ~2.5-4 km", and
the C13 scope line "SASE claims boundary-layer and resolved-storm
turbulence, not deep convective mass flux" -- a scale-separation
bound, not an amplitude tune).  A THIRD, undeclared-until-now
stand-down (S4-4 review round-3, x2 lane Minor finding, RE-VERIFIED
against the round-3 k_lid formula): if the qualifying run's own top
cell IS the column's own top cell (k_top = nz - 1), there is no room
left in the column for an entrainment-zone cell at all (k_top + 1 =
nz is out of bounds), so neither a natural NB nor the k_lid ceiling
can ever be reached by the k-loop (``range(nz)``) -- the column
returns bitwise zeros.  (A run reaching only k_top = nz - 2, one cell
short of the column top, does NOT stand down this way: the
entrainment-zone cell itself, nz - 1, still exists and is reachable,
so a natural NB can still be found there even though the k_lid
ceiling, nz, is itself out of bounds -- verified on a truncated deck
column.)  Harmless in production geometry (a saturated run reaching
the model top is not a physical inversion-capped deck), but a real,
reachable branch (verified on truncated deck columns at both depths)
the S4-5 device mirror must reproduce exactly.  A FOURTH stand-down --
a run based in the LOWEST MODEL LEVEL -- is registered under
VENT_ANCHOR_RULE and lives at step 1a; see SHAPE below for its
derivation and its measured basis.

INVERSION BASE (S4-4 review Important-2, AMENDED round-3 -- coordinator
ruling, design doc SASE-M2 amendment "discrete C9 reading").  THE
CONTRACT: the plume's territory is capped at the layer the mask
diagnosed PLUS one buffer cell -- the DISCRETE ENTRAINMENT ZONE,
standard mass-flux practice for the cell immediately above a
diagnosed saturated layer (the reference's own inversion-band tail is
nonzero there too) -- never the capping layer proper.  k_top is the
run's own last ROBUSTLY-saturated center (VENT_K_LID_MEMBERSHIP =
"qt-ge-qs-v1", the total-water test qt = qv + qc >= qs -- see
MEMBERSHIP above); the entrainment-zone cell is k_top+1;
k_lid = (k_top+1)+1 is that cell's own TOP face index, structure the
column analysis already computes from quantities step 1 needs
regardless (qt_env, qs_env, the run loop), zero new tunable constants.
The LFC/peak/NB search is bounded to centers k < k_lid (through and
including the entrainment-zone cell); if the parcel is still buoyant
when the search reaches k_lid (no natural B <= 0 center found through
the entrainment zone), k_lid itself becomes the NB by construction --
the cap proper is an automatic ceiling on termination, never crossed,
and receives bitwise +0.0 on every row (cap preservation holds by
construction).

WHY A BOUNDARY IS NEEDED AT ALL (measured, S4-4 review, RE-VERIFIED
round-3 -- the original "coarse-grid artifact" diagnosis is FALSE and
is retired): removing the k_lid bound entirely and refining the
specimen's capping-inversion cell up to 32x (dz ~ 8 m at the lid)
does NOT make the unbounded parcel stop at the deck top -- it keeps
terminating at 1793.9 m (1785.1 m at the finest refinement), a
GRID-CONVERGED result.  The entraining parcel is genuinely, physically
buoyant into the smoothed capping inversion; the pre-round-2 "coarse
face-mean dilutes the near-top dryness a finer grid resolves" story
tested this claim against a fixture that ALREADY ran the capped code
(:func:`test_m2_termination_grid_consistency`) and so could only ever
confirm the cap it was supposed to be testing -- a circular argument,
now retired along with the "artifact" diagnosis it supported.  The
REAL justification is the design contract itself: SASE-M spec section
4's C9 clause ("venting never erodes the inversion... reference holds
MLCIN ~= 0 / MLCAPE 2200-2900 J/kg for 3.5 h without CI") MANDATES the
hard boundary regardless of why the unbounded parcel stays buoyant --
cap preservation is a modeling requirement, not a numerical-noise
correction -- and the amended C9 reading additionally requires that
the boundary sit at the entrainment zone's own top, not the bare run's
top, so the legitimate one-cell detrainment buffer is honored.

MEASURED (specimen, UNCHANGED by the round-3 membership fix and by
the round-4 extension of it to the whole structural layer -- see
MEMBERSHIP above): k_top = 15 (member run k12..k15; was 16 under the
raw/noise-sensitive test), entrainment zone = k16, k_lid = 17,
termination face z_f[17] = 1511.653 m (the SAME numeric face the
pre-amendment code computed as z_f[16+1]; the two formulas coincide on
the specimen because k_top itself moved 16 -> 15 exactly offsetting
the k_top+1 -> k_top+2 formula change -- recipient/zero structure
UNCHANGED there; the pre-round-3 docstring's "1512.1 m" figure was
simply wrong -- 1511.653 m is, and always was, the measured value).
Before the S4-4 review fix 100% of the specimen's detrainment landed
in the clear (RH 52%) inversion cell (k17, 1512-1794 m); after it (and
unaffected by the round-3 membership refinement), termination is
capped at k_lid = 17, and the degenerate empty-taper case (module
docstring below) detrains fully into the entrainment-zone cell (k16)
-- at/below the cap, never inside it.  On CONSTRUCTED columns whose
top was never noise-classified (the deck fixtures), the entrainment
zone is a genuinely NEW search cell the pre-round-3 code never
searched (see test_m2_termination_grid_consistency): termination now
lands at the deck's own physical top OR one cell higher (at the
entrainment zone's own top face), never at the deck top ALWAYS as the
pre-round-3 fixture assumed -- a real, intended widening of legitimate
search territory, not a defect.

SHAPE: M(z) = M_used*M_hat(z) on faces.  Below the buoyancy-peak
face M_hat grows by pure entrainment, d(ln M)/dz = eps, discrete
(z_f[j]/z_f[k_base])**c_eps (exact for eps = c_eps/z), normalized on
the CLOUD-BASE face so M_hat = 1 there and M_used is literally the
cloud-base mass flux M_b of the Grant closure (round 5, AMPLITUDE
above; was z_f[k_r + 1], the root cell's own top face, which made the
whole profile scale as z_f_root**(-c_eps) -- a root-depth
amplification).

THE k_base = 0 CASE -- VENT_ANCHOR_RULE = "cloud-base-face-standdown-v1"
(design doc SASE-M section 4, amendment "a surface-based saturated layer
stands the limb down"; audit-wave project decision).  z_f[0] = 0
identically (the module's cumsum convention) and F[0] = +0.0 by the
interface contract, so face 0 can never normalize a shape: a run based
in the LOWEST MODEL LEVEL has no cloud base to anchor on, and the
eps = c_eps/z entrainment integral diverges at z = 0.  The
pre-amendment reading anchored such a column at face 1 -- the lowest
face the shape can occupy -- which is the lowest layer's THICKNESS, not
a height, so the amplitude became a function of the vertical GRID
rather than of the state.  MEASURED at the design point over whole
intermediate-nest frames (S4-5c survey, post-step-4 active set, no
sampling): the branch
was live on 91/19495 (11Z), 661/25626 (12Z), 307/22179 (13Z) columns;
the 11Z median anchor face is 610.38 m where k_base > 0 against 17.08 m
where k_base = 0; the shape factor at a fixed physical face multiplies
by exactly 2**VENT_ENT_COEF per refinement doubling on the branch; and
the frame MAXIMUM export was a k_base = 0 column in all three surveyed
frames.  That breaks the C9 grid-consistency contract by measurement,
so the column now STANDS DOWN (step 1a, bitwise +0.0 on all three rows)
-- M2 is a cloud-base mass-flux closure and a saturated layer sitting
on the ground is fog or surface-based stratus, ED-limb (M1) business.
CONSIDERED AND REJECTED: M_hat == 1 on the branch, which keeps the
column firing at the Grant amplitude but leaves a step discontinuity
against k_base = 1 columns that the C9 continuity clause forbids.
Bitwise-inert on the whole registered corpus by construction, which is
exactly why the convention string exists: sase_config_id() hashes
constant VALUES, not code.  From the peak
face to NB the plume detrains with the remaining-buoyancy weight

    M_hat(j) = M_hat(peak) * sum_{c >= j, c < NB} B_c*thick_c
                           / sum_{c > kb, c < NB} B_c*thick_c,

which reaches EXACTLY +0.0 at the NB face (empty sum): full
in-column detrainment with zero shape constants.  Fluxes
(face-registered, the S4-4 interface contract):

    F_phi[j] = M_used * M_hat[j] * facemean(phi_p - phi_env)_j,
    M_used   = (1 - f_blend) * M_base,

facemean the arithmetic face mean of the cell-center parcel excess
(the module's face convention) -- the (1-f) factor is the FP-exact
two-product blend idiom: f = 1 multiplies by literal +0.0 and every
face is bitwise zero (the S3-11a zero-flux identity class); domain
f GATES the regime only, per-column amplitude stays column-local
(C7, the S3-9 lake lesson).  F[0] = F[top] = +0.0 exactly, ALWAYS
(the surface flux is owned by the S3-11a deposit -- double-counting
ban).  Units are dynamic: F_theta [K kg m^-2 s^-1], F_qv/F_qc
[kg m^-2 s^-1].

MECHANICAL INSERTION CONTRACT (C1-C3; the S4-5 driver seam): the
fluxes deposit explicitly in the driver scalar loop from the frozen
pre-step state BEFORE ``launch_implicit_vertical_diffusion`` -- the
registered deposit-then-solve order generalizing SFC_SCALAR_FLUX =
"explicit-deposit-v1" -- as

    phi_k += (F[k] - F[k+1]) * dt / (rho1 * thick_k),

NEVER inside :func:`sase_split_step` (the ledger theorem reads theta
read-only) and NEVER in a Thomas row -- for three reasons that stand
on their own: the transport is NON-LOCAL by construction (one anchor
level to every level up to termination), so it has no nearest-neighbour
tridiagonal representation at all; the deposit must be computed from
the FROZEN pre-step state for the scalar ledger to telescope and for
the registered rate caps to bind a KNOWN quantity BEFORE the solve,
neither of which survives making transport a function of the SOLVED
state; and the spec's section 7 registers any change to the Thomas
solver as a v1 non-goal.  The Thomas kernels stay untouched.
(Rationale RE-FOUNDED -- spec-sweep ruling 5.  This read "the solver's
pinned max principle is exactly what counter-gradient transport must
violate", which rests on the clause the design's counter-gradient
amendment STRUCK: the reference's subgrid heat AND moisture fluxes are
measured everywhere DOWNGRADIENT, reproducible by a pure ED closure,
with no countergradient term -- sase-m-target-envelope.md sections 3
and 6.5.  The ban is unchanged; its reasons no longer depend on a
struck claim.)  LEDGER (C4/C5): the M2 term is
a diagnosed coefficient -- no e drain, no momentum -- so the
split-step theorem holds VERBATIM, and the S3-11a boundary-
consistent scalar ledger extends with a ZERO net-column term:

    sum_k thick_k*dphi_k = (dt/rho1) * sum_k (F[k] - F[k+1])
                         = (dt/rho1) * (F[0] - F[nz]) = 0

exactly in exact arithmetic -- interior faces telescope and both end
faces are exactly zero; rho1 is a per-column SCALAR (the S3-11a
single-density conversion), so the thickness weights cancel the
deposit's 1/thick_k exactly as in the S3-11a closure.  RATE CAP
(S4-4 review Important-4 FIX, EXTENDED round-3 to a CAP FAMILY):
VENT_THETA_STEP_CAP = 0.14 K/step -- the S3-11a measured
stable-deposit class (this docstring's S3-11a section: |dtheta_1| ~
0.14 K per intermediate-nest step at the pinned defect state) -- bounds
the per-step theta deposit; VENT_QT_STEP_CAP = VENT_THETA_STEP_CAP *
CP_AIR / XLV (the latent/sensible heat equivalence, DERIVED from the
SAME registered constant plus two constants already shared with
:func:`moist_n2` -- no second independent tunable) bounds qv and qc.
DEFECT THIS FIXES (S4-4 review, measured): the cap previously bounded
theta only -- the moisture channel scaled unbounded with the amplitude
(measured 0.047 g/kg/step shipped, x16.5 at the theta cap's own
headroom limit, with nothing stopping it going further).  ROUND-3 CAP
FAMILY (x2 lane finding): a SUM-form bound on |dqv + dqc| alone is
identically VACUOUS on any column doing pure phase conversion (parcel
qt == environment qt, so F_qv = -F_qc exactly and the sum cancels to
+0.0) while the INDIVIDUAL rows still scale with the amplitude --
measured on the specimen, three cells carry (|dqv|+|dqc|)/|dqt| =
3.2-19.2, i.e. a single row can run several times the summed quotient
while a sum-only test sees nothing.  THE FIX applies VENT_QT_STEP_CAP
to |dqv| and |dqc| SEPARATELY, and there is NO separate sum term --
the family is exactly {theta, qv, qc}, still no second tunable.
(S4-5b Item 5c: this sentence previously read "alongside the sum",
which no implementation ever matched -- the registered formula below,
:func:`vent_deposit_rescale` and the device
``sase_vent_deposit_scale`` all compute the three-quotient minimum and
agree with each other.  The prose was wrong, not the code, and a sum
term is not needed: the per-row caps bound the sum within a factor 2,
since |dqv + dqc| <= |dqv| + |dqc| <= 2*VENT_QT_STEP_CAP by the
triangle inequality, with equality only when the two rows move the
same way -- and the pure-phase-conversion case that motivated the
family at all is the opposite corner, where the sum cancels to +0.0
and only the per-row bound bites.)  The S4-5 seam
enforces ALL of theta/qv/qc by a UNIFORM per-column rescale of all
three flux profiles -- uniform so the telescoping, the zero ends, and
the qv/qc partition survive:

    s = min(1, VENT_THETA_STEP_CAP / |dtheta|_max,
               VENT_QT_STEP_CAP / |dqv|_max,
               VENT_QT_STEP_CAP / |dqc|_max),

each quotient over the column's own max (not the sum's), so the
tightest of the four constraints governs; F_phi *= s uniformly
preserves telescoping and F[0] = F[top] = +0.0 exactly (verified: the
uniform rescale keeps the ledger residual at roundoff class even when
driven at 40x amplitude, while a naive per-level clip of any one row
destroys it).  DIVIDE GUARD (S4-5 implementation note): on an
inactive/masked-off column every |d*|_max is exactly +0.0, so each
quotient is +inf -- min(1, inf, inf, inf) still correctly yields 1.0,
but the bare division emits a numpy RuntimeWarning, which this suite's
``-W error::RuntimeWarning`` policy treats as a test failure; the S4-5
implementation must guard the division (e.g. ``np.where(dmax > 0,
cap/dmax, np.inf)``) rather than divide unconditionally.  EXNER-FACTOR
LOOSENESS (documented, not tightened): a STRICT latent/sensible
equivalence at level k is L*dq = cp*Pi_k*dtheta (Pi_k = (p_k/P0_REF)**
(Rd/cp) at THAT level), which would make the cap Pi-DEPENDENT and
therefore per-LEVEL rather than a single registered scalar; the
shipped VENT_QT_STEP_CAP drops Pi (~0.90 near cloud base), making the
cap ~11% LOOSER than the exact per-level equivalence.  This is the
conservative direction (a looser moisture cap never falsely blocks a
theta-bounded deposit) and keeps VENT_QT_STEP_CAP a single
config-ID-bound constant instead of a field; kept as-is rather than
multiplying in the module's own Exner array, a derivation trade
documented here rather than silently absorbed.  Specimen headroom
(MEASURED, S4-4 review round-3): theta 18.72x, qv 7.24x, qc 5.42x, sum
qv+qc 3.20x -- all comfortably under cap; the individual rows have
LESS headroom than the sum precisely because the sum benefits from the
qv/qc cancellation the individual rows do not get (the reason this
channel needed its own per-row bound rather than inheriting the sum's
margin).  ENFORCEMENT SCOPE: none of this is applied in
:func:`plume_vent_flux` itself -- the caps are REGISTERED and
CONTRACTED (config-ID-bound, hash-sensitive, specimen headroom
pinned) but enforcement is explicitly the S4-5 deposit seam's job, the
same status VENT_THETA_STEP_CAP has carried since S4-4 authority.
RESPONSE TIME: the limb is DIAGNOSED from the instantaneous state
every step -- engagement latency is one physics step on the state mask
(well under the <= ~10 min bound against the resolved 10-18 min energy
e-folding, anatomy section 5); the only spin-up it inherits is M1's
own e_sgs uptake, which is the designed coupling (M1 seeds M2's
amplitude).

CONSIDERED AND REJECTED (registered): (a) a CIN gate (stand down
when the root-to-LFC negative-buoyancy integral exceeds the seed
kinetic energy) -- it double-adjudicates the mask decision with the
amplitude's own quantity and VETOES the measured specimen, whose
subsaturated-root negative-B band (0.55-0.98 km) IS the DOWNGRADIENT
subcloud lobe this regime carries -- a downward heat flux through a
stably stratified subcloud layer is downgradient by definition, and it
is the reference's own signature there (-20..-150 W/m^2 at 0.4-0.7 km,
sase-m-target-envelope.md section 6.5): the resolved amplifier
realized exactly this structure as a LAYER overturning, not a
sequence of independent CIN-limited parcels.  (Relabelled -- spec-sweep
ruling 5; this read "IS the counter-gradient lobe the residual
demands", the clause the design's counter-gradient amendment struck.
The band, the measurement and the rejection are unchanged.)  (b) a plume
vertical-velocity equation (Simpson-Wiggert family) -- it adds two
disputed constants and terminates on w = 0 OVERSHOOT past NB, which
is precisely the 1.9-2.7 km tail this limb exists to kill.

MEASURED (the 11Z amplifier specimen at the M1 frozen-state
equilibrium e, f = 0, post S4-4-review, round-3-amendment, round-4
noise-immunity AND round-5 root/anchor fixes -- every number below
re-measured on the ROUND-5 build this session; the STRUCTURE
(k_r/k_base/k_top/k_lid/k_nb/kb, the flux support, the C9 zeros) is
bitwise unchanged from 3367244 and df235f0, and every MAGNITUDE
carries the single round-5 amplitude factor 0.9340467 recorded at the
end of this paragraph):
root k_r = k10 (flux root face 502 m; the theta_es maximum below the
member run, above the round-5 depth floor k8 -- ROOT above), LFC
1167 m, buoyancy peak k16 (the
entrainment-zone cell -- MEMBERSHIP/INVERSION BASE above), NB =
k_lid = k17 (the entrainment zone's own top face; UNCHANGED numeric
value from the pre-round-3 "mask's own run top + 1" reading -- see
MEMBERSHIP) -> termination face 1511.653 m, the TOP of the
entrainment zone (zero tail above -- Important-2 FIX, AMENDED
round-3; was 1793.9 m, inside the capping-inversion cell, pre-S4-4);
export F_qv + F_qc = 0.76074e-4 kg/m^2/s (190 W/m^2 latent-equivalent)
at the 1270-m layer-top face = 0.386160 of the scoring-box supply
1.97e-4 kg/m^2/s -- inside the reference's 0.7-1.1e-4 export class AND
inside that class's exact derived image, the per-column ratio pin
[0.3553299, 0.5583756] (sase-m-target-envelope.md sections 3/4/6; NO
export fraction is coded anywhere -- the number EMERGES from the Grant
amplitude + SST07 dilution and the fixture checks it).  BAND STATUS,
ROUND 5 -- RESOLVED; the RED / BLOCKED status against a frozen
[0.4, 0.6] that this paragraph carried is SUPERSEDED (spec-sweep
ruling 12): the round-4 value was 0.814e-4 / 0.413427, and the round-3
MARGIN NOTE recorded that 0.413 sat only ~3.4% above the then-frozen
0.4 rail and that any further ~3% amplitude reduction would drop it
out.  The cloud-base anchor is a 6.6% reduction on this column and did
exactly that, to 0.386 -- which the design doc's own amendment
predicted in advance ("a cloud-base anchor may move the specimen below
the 0.4 rail; a first sketch measured 0.386") while ruling that the
bands do NOT widen.  They did not.  The round-5 adjudication (design
doc section 4, "the export ratio is not an independent criterion")
found the two frozen bands to be restatements of ONE quantity -- the
fixture's supply is the registered constant 1.97e-4 kg/m^2/s, so
export/supply is a linear rescale of export carrying no independent
information -- with rails that never agreed ([0.7, 1.1]e-4 maps to
[0.3553, 0.5584]; [0.4, 0.6] maps to [0.788, 1.182]e-4, a floor and a
ceiling both outside the export bar's own rails), and RE-DERIVED the
per-column ratio pin as the export class's exact image.  The export
class itself is untouched and remains the PRIMARY bar.  0.386160 is
inside the re-derived pin, the fixture asserts it GREEN, and the
40-60%-of-supply claim is the BOX-SCALE aggregate registered as G-M7
form 2, never a per-column invariant.
HEAT-CHANNEL SIGN STRUCTURE (the "counter-gradient heat lobe" label
this paragraph carried is struck -- spec-sweep ruling 5; the negative
lobe is DOWNGRADIENT, which is what the reference itself carries):
cp*F_theta = -3.51..-7.65 W/m^2 at faces 502-887 m -- downward through
the stably stratified subcloud layer, i.e. downgradient, inside the
reference's own -20..-150 W/m^2 subcloud class -- with the positive
limb aloft (faces 1063/1270 m: +3.87/+31.68 W/m^2; the former 1512-m
third aloft face is now the exact-zero termination face) landing
INSIDE the registered never-large-positive layer-top class
(+13..+35 W/m^2 = w'theta' +0.01..+0.03 K m/s,
sase-m-target-envelope.md section 6.5), which is the requirement that
replaced the struck clause; peak transport at the 1270-m face
(LE 190 W/m^2, inside the anatomy's intermediate-nest conditional
class); condensate deposit +9.70 mg/kg per
60-s step into the 1270-1512 m cell -- the entrainment-zone cell, NOT
the clear inversion cell above it (Important-3 FIX: was +16 mg/kg
into the clear 1512-1794 m cell, pre-fix); TRANSPORT IS EXACTLY
CONSERVED within THE PLUME'S OWN TERRITORY, the root through the
entrainment-zone cell (k_r..k_top+1 = k10..k16 on the specimen --
test_m2_condense_dont_clear_partition; it contains but is wider than
the member run k12..k15, and is wider than the ebar window k12..k16
the amplitude now uses), with
48.8% of the condensate funding the entrainment-zone gain drawn from
cells BELOW 1.0 km (k10-14), not the "1.0-1.8 km band" the pre-
round-3 report claimed; the recipient does NOT saturate on arrival --
it evaporates the deposit for ~28 min (measured -161.6 g/m2/h
transient) before the cumulative gain finally saturates it, i.e.
discrete cloud-top deepening, never instantaneous condensation; max
theta deposit 0.0069862 K/step (20.04x under VENT_THETA_STEP_CAP),
max qv+qc deposit 0.01643 g/kg/step (3.42x under the derived
VENT_QT_STEP_CAP), max|dqv| 7.27 mg/kg/step (7.74x under),
max|dqc| 9.70 mg/kg/step
(5.80x under) -- the round-3 CAP FAMILY: qv/qc individually bound,
not just their (cancellation-prone) sum.  Every round-5 number in this
paragraph is the round-4 number times the single amplitude factor
0.9340467 (the ebar window 1.009877 times the shape normalization
0.9249114), because the anchor rescales all three flux rows
uniformly on this column -- structure, faces and ratios unchanged.
The control column, a
masked-off column, and the f = 1 LES limit all return bitwise +0.0.
The device twin and the driver deposit wiring LANDED at S4-5
(`1c3edc2` feat(sase): M2 device + driver deposit seam), so the
production path carries the limb: the interim "until they land the
production run lacks the limb BY DESIGN" note this paragraph carried is
RETIRED (spec-sweep ruling 12).

S3-12 amendment (ADDITIVE e^{3/2} DISSIPATION CHANNEL; gated by
``RunConfig.sase_additive_dissipation``, DEFAULT True since the
2026-08-17 real-data confirmation leg -- see the DEFAULT section
below).  This is the
fix LD_STABILITY_LIMIT_REJECTED named and did not attempt: it bounds
the stable-limb amplitude by ADDING a dissipation channel rather than
by removing the e-linear one, so dissipation is nowhere weaker than
HEAD's.

THE DEFECT, restated in one line so this section stands alone.  Where
l_s = LS_COEF*sqrt(e)/N binds BOTH the mixing and the dissipation
length, K_v = LS_COEF*C_r*e/N and eps = (C_E/LS_COEF)*e*N are both
LINEAR in e, the subgrid-energy equation is exactly homogeneous, its
specific rate carries no e at all, and below

    Ri* = C_KS/(C_E/LS_COEF + C_KS/PR_RANS) = 0.16471188169301373

the energy grows exponentially with nothing in the closure to stop it.
Measured on real GFS at 12 km: l_s bound in 65-100% of live cells from
9.2 to 14.3 km, median l_s 0.2-3.2 m against l_B ~ 146 m, e reaching
1.62 m2/s2 at 13.1 km and still doubling every ~3 min at t+60 min.

THE FORM.  Deardorff's length-dependent coefficient, written out, is
already two additive channels -- c_eps = c_eps,1 + c_eps,2*lambda/Delta
multiplying e^{3/2}/lambda is

    eps = c_eps,1*e^{3/2}/lambda  +  c_eps,2*e^{3/2}/Delta

-- and only the second divides by a length that does not depend on the
state.  S3-6k took the FIRST member (C_ES = 0.19 replacing C_E) and
measured RED.  This amendment takes the SECOND (C_ED = 0.51) and ADDS
it:

    l_ref = delta**f * l_B(z + z0)**(1 - f)   [neutral_dissipation_length]
    rho   = min(l_d/l_s, 1)                   [0 where N^2 <= 0]
    w     = rho**CKS_BLEND_EXP
    C_eps = C_E + (1 - f)*w*C_ED*(l_d/l_ref)  where N^2 > 0
    C_eps = C_E                               otherwise   [SELECTED]
    eps   = C_eps*e^{3/2}/l_d

(:func:`additive_dissipation_coefficient`).  Returning the effective
COEFFICIENT and not the rate is deliberate: the sum stays of the form
-K*e^{3/2}, so the S3-6d analytic decay substep integrates it exactly
with no new integrator and the ledger theorem holds verbatim.

WHY l_ref DROPS THE BL89 MEMBER -- this is the entire content of the
amendment and the trap a later lane would fall into.  l_d's RANS input
is l_eps_rans = min(l_B, l_eps_BL89).  The BL89 displacement lengths
solve a parcel-energy integral, so in uniform stratification they are
sqrt(2*e)/N -- proportional to sqrt(e), exactly like l_s, only longer
(which is also why l_s always binds first).  Divide C_ED*e^{3/2} by a
length proportional to sqrt(e) and the "e^{3/2} channel" is e-LINEAR
again: it breaks no homogeneity, has no fixed point, and would look
like a fix while being a rescaling of the defect.  MEASURED, both ways,
in ``test_additive_channel_needs_a_state_independent_reference_length``:
l_eps_BL89/sqrt(e) is constant to 2.5e-5 across four decades, and the
specific rate built on it is flat to 1e-12 across eight decades of e,
while the same rate on l_B spreads by more than 100%.  Only l_B carries
no e, so only l_B is admissible -- and at f = 1 the blend is ``delta``,
Deardorff's own Delta, FP-exact.

THE ALGEBRA UNDER THE NEW FORM, re-derived to the same standard as the
old (``test_additive_dissipation_breaks_the_stable_limb_homogeneity``;
N^2 = 3.566e-5, the measured 12.2 km value, l_ref = BLACKADAR_LAMBDA):

* HOMOGENEITY BROKEN.  Writing HEAD's specific rate as ``a`` (constant
  in e), the limb becomes de/dt = a*e - (1-f)*C_ED*e^{3/2}/l_ref.  Over
  eight decades of e the specific rate now spreads by 1.7 (Ri = 0.05),
  6.2 (Ri = 0.10) and 62 (Ri = 0.15) relative -- against HEAD's
  6.7e-16 to 8.4e-15 on the same grid.
* A FINITE FIXED POINT EXISTS IN THE BINDING LIMB, and it is an
  ATTRACTOR: sqrt(e_eq) = a*l_ref/((1 - f)*C_ED), matched to a
  log-space bisection on the authority-evaluated rate to 1e-6, with
  the rate positive below it and negative above.

* A CORRECTION TO THE RECORD, MEASURED, WHICH THE PREVIOUS LANE'S
  FRAMING DID NOT CARRY.  "The stable limb has no equilibrium
  amplitude" is exactly true WHILE l_s BINDS, and false of the closure
  as a whole.  l_s = LS_COEF*sqrt(e)/N GROWS as sqrt(e), so a cell that
  runs away eventually pushes l_s past the Blackadar length; there
  ``dissipation_length``'s outer min stops selecting it, l_d becomes
  the state-independent l_B, and the e^{3/2} dissipation comes back BY
  ITSELF.  At N^2 = 3.566e-5 that crossover is at e = 1.347 m2/s2 --
  which is where the measured runaway had just reached (1.62 m2/s2 at
  13.1 km, t+60 min).  HEAD is therefore BOUNDED, at a fixed point of
  2.8 to 39 m2/s2, and the correct statement of the defect is not
  "unbounded" but "bounded two to three orders of magnitude above
  anything physical, after a transit through a homogeneous regime that
  has no amplitude of its own".

  Full-composition fixed points, solved through the LIVE length and
  coefficient functions at f = 0 on a uniform-N^2 column (NOT the
  binding-limb closed form, which is valid only where its own answer
  lands below the crossover, i.e. Ri >~ 0.115):

      Ri     e_eq HEAD   e_eq ADDITIVE   reduction   l_d at eq [m]
      0.020     38.96        38.20          1.02x      147.7
      0.050     14.42        13.63          1.06x      147.7
      0.080      8.245        7.391         1.12x      147.7
      0.100      6.161        5.238         1.18x      147.7
      0.120      4.744        0.9567        4.96x      124.5
      0.140      3.694        0.2147       17.2x        58.97
      0.160      2.840        0.005976    475x           9.839
      0.180      0            0             --          --

  READ THIS ACCURATELY.  The channel is worth 5x to 475x in the
  Ri = 0.12-0.16 band -- which is where the defect actually lives, a
  hair under Ri* = 0.16471, and where the census found 72% of live
  cells against a 0.36% base rate -- and worth only 2-18% at Ri <= 0.1.
  It is not a uniform rescaling and it is not advertised as one.  The
  reason for the split is structural and is the design working: at low
  Ri the equilibrium sits ABOVE the crossover, l_s is already slack,
  the stability gate w = rho**CKS_BLEND_EXP has tapered the added
  channel off, and l_d = l_B is carrying an e^{3/2} dissipation
  anyway.  The channel fires where l_s binds, which is exactly where
  nothing else does.
* A CRITICAL RICHARDSON NUMBER STILL EXISTS, AND IT NO LONGER BOUNDS A
  GROWTH REGIME.  The added term vanishes as sqrt(e), so at the E_MIN
  floor the onset threshold is HEAD's 0.16471 to better than 1e-3
  relative -- turbulence still cannot START above Ri*, which is the
  registered property the jet and lake fixtures depend on.  Above the
  floor the effective threshold FALLS monotonically with amplitude
  (0.16465 at E_MIN, 0.16278 at e = 1e-3, 0.11978 at e = 1): that
  falling threshold IS the saturation.  So the limb keeps its
  bang-bang character at the floor and loses its runaway above it,
  which is exactly the pair of properties that lets the fix land
  without moving Ri*, C_KS, C_ES or C_E.  The absorbing state is NOT
  removed (S3-6k's "WHAT THIS IS NOT" paragraph still stands; the
  structural cure remains the EFB/TTE formulation named there).

THE TWO FIXTURES THAT KILLED THE PREVIOUS TWO ATTEMPTS, RUN FIRST
(``test_additive_dissipation_holds_the_two_calibration_fixtures``;
same engines, same registered criteria, nothing widened):

    leg               u10 min  u10 max  @3600  jetdev%   ibl_d74
    HEAD                5.701    6.580  6.580    0.063    12.7995
    additive ON         5.701    6.485  6.485    0.061    10.9009
    S3-6k C_ES ON       5.701    8.394  8.394    0.218     1.6433
    both ON             5.701    7.101  7.101    0.144     1.9497

Both GREEN.  The direction is the reason: every previous attempt
WEAKENED dissipation, and both fixtures are held by dissipation.  The
jet band gains margin over HEAD itself (6.485 against 6.580, band top
7.0).  THE LAKE MARGIN NARROWS AND IS RECORDED, NOT ROUNDED OFF:
ibl_d74 falls 12.7995 -> 10.9009 against a floor of 10.0 (28% margin
to 9%), with the column's peak subgrid energy below 400 m falling
2.2013 -> 0.8059 m2/s2.  A later coefficient move that spends the rest
of that headroom should have to see this number.

DEFAULT: TRUE, AND WHY -- flipped 2026-08-17 on the real-data
confirmation leg the 08-02 session left named and unflown: the 08-01
operational configuration (the `woof domain` 12 km NA emission,
GFS front door, 306x244x49) flown with this channel on, on the
gate that session set -- run completes, no TKE regrowth above the
BL, health gate silent.  Under fixed-means-default the channel is a
correctness remedy for the stable-limb runaway and ships default-on.
The paragraph that follows is the 08-02 status that governed while
the default was still False, kept verbatim because its blast-radius
measurement is the flip's own test plan.  Flipping
the default was measured this session: 9 of 242 tests move, and every
one of them is either a bitwise golden (test_split_step_trajectory_
goldens) or a RED leg that pins a HISTORICAL formulation stack
(test_jet_decoupling_red_current_formulation_exits_obs_band,
test_jet_decoupling_stable_dissipation_exits_obs_band -- whose
``u10s.max() > 7.356`` assertion the composed switch turns into 7.101,
i.e. the S3-6k RED is PARTLY REPAIRED by this channel -- plus the lake
RED leg, the fixed-LES-Prandtl inversion RED and four M2 tests whose
plume machinery reads e).  None of those is a physics failure and none
of them may be silently re-pinned: the correct handling is the module's
own idiom (each RED leg pins ``additive_dissipation=False`` exactly as
the S3-6j RED legs pin ``apply_drag=False``), and that is a default
flip with its own evidence, taken against a stable-limb calibration
target rather than against the goldens it happens to move.

C_E STAYS PUT.  The registered TKE deficit (e/u*^2 = 1.052 against an
observed 3.3-5.5) is untouched here, deliberately.  The trap recorded
at C_E is that lowering it RAISES the free-troposphere equilibrium 30x
because the stable-limb fixed point scales as (C_r/C_E)^2 -- and that
argument is about the HOMOGENEOUS limb.  Under this channel the fixed
point is instead sqrt(e_eq) = a*l_ref/((1-f)*C_ED) with
a = N*[LS_COEF*C_r*(1/Ri - 1/Pr_t) - C_E/LS_COEF], so C_E now enters
e_eq only through ``a`` and lowering it still RAISES e_eq, now
quadratically in a rather than as (C_r/C_E)^2.  The direction of the
trap is unchanged; only its exponent moves.  A joint (C_E, C_KS) move
therefore still needs a stable-limb calibration target, and this
amendment does not supply one -- it supplies the amplitude BOUND that
makes such a target measurable at all.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from woof.core.sase_limits import (E_MIN,
                                    MAX_COLUMN_LEVELS)

import numpy as np

#: Deardorff dissipation and diffusion constants (documented synthesis).
#:
#: DO NOT LOWER C_E ALONE TO CLOSE THE REGISTERED TKE DEFICIT.  Read
#: this before acting on the "equilibrium e ~ 1.05*u*^2 vs the observed
#: 3.3-5.5" entry recorded at C_KV and in woof/core/physics.py.
#:
#: The tempting argument is sound but PARTIAL.  The log law forces
#: C_KV^3 = C_E -- one equation in two unknowns -- so C_E is free and
#: sets e/u*^2 = C_E^(-2/3).  Measured on this module's own neutral
#: constant-stress column: 1.052 as registered, 3.256 at
#: C_E = 2^1.5/16.6 = 0.1704 (the Mellor-Yamada B_1 = 16.6 equivalent,
#: the RANS calibration this length belongs to as against the
#: inertial-range LES calibration 0.93), landing inside the observed
#: band with K_v/(u* l_B) and the interior-face momentum flux unchanged
#: to 1e-3 across the whole sweep.  All of that is true.
#:
#: IT IS TRUE ONLY IN THE NEUTRAL WALL REGIME, where l = kappa*z carries
#: no e.  In the stable limb the mixing length is
#: l_s = LS_COEF*sqrt(e)/N, K_v = LS_COEF*C_r*e/N, and the stable-limb
#: fixed point scales as (C_r/C_E)^2.  Lowering C_E by 5.46x therefore
#: RAISES the free-troposphere equilibrium by 30x.  Measured at the
#: conditions this closure actually meets (N^2 = 3.566e-5 at 13.1 km,
#: Ri = 0.1): e_eq goes from 2.61 m2/s2 at C_E = 0.93 to 77.7 m2/s2 at
#: C_E = 0.1704 -- a 30x worsening of the free-troposphere runaway
#: recorded at LD_STABILITY_LIMIT_REJECTED.
#:
#: The arithmetic says what a joint move must do: to hold the
#: free-troposphere amplitude while C_E drops to 0.1704, C_KS must fall
#: by the SAME factor, 0.25 -> 0.04580.  That number is NOT registered,
#: because choosing it to hold one case's amplitude at one level would
#: be fitting, not calibration.  C_KS and C_ES are already recorded as
#: jointly falsified and requiring joint re-registration; the accurate
#: blocker is that no defensible stable-limb calibration target exists
#: yet.
#:
#: THAT BLOCKER IS NOW CLOSED, AND THE ANSWER IS NO (S3-12).  A
#: defensible stable-limb calibration target exists and has been
#: adopted: GABLS1, the GEWEX Atmospheric Boundary Layer Study first
#: case (:mod:`woof.verify.cases.gabls1`; Beare et al. 2006 BLM
#: 118:247 for the setup and the depth definition, Cuxart et al. 2006
#: BLM 118:273 for the single-column ensemble).  The joint move is
#: therefore CALIBRATION now and not fitting -- and the target REFUSES
#: IT.
#:
#: MEASURED on the prescribed case, 9 h, 64 x 6.25 m (LES reference
#: h = 177 +/- 16 m; Cuxart et al. report the LES near-ground TKE at
#: ~0.3 m2/s2 with participating models 0.2-0.5 -- the FIRST published
#: number this project has been able to score the deficit against):
#:
#:     C_E     C_KS    h [m]  dev/sigma   e_max   e/u*^2
#:     0.93    0.25    185.3    +0.52     0.0816   0.908   REGISTERED
#:     --- C_E alone, C_KV = C_E^(1/3) following the log law ---
#:     0.60    0.25    220.2    +2.70     0.1108   1.089
#:     0.40    0.25    250.4    +4.58     0.1570   1.421
#:     0.2734  0.25    276.6    +6.22     0.2145   1.832
#:     0.1704  0.25    305.7    +8.04     0.3077   2.492
#:     --- the JOINT move, C_KS scaled by the same factor as C_E ---
#:     0.60    0.1613  192.1    +0.94     0.1172   1.252
#:     0.40    0.1075  197.6    +1.29     0.1592   1.636
#:     0.2734  0.0735  202.6    +1.60     0.2082   2.065
#:
#: READ IT IN THREE PARTS.
#:
#: (1) C_E ALONE IS DECISIVELY REFUSED.  At the Mellor-Yamada value the
#: benchmark depth is EIGHT published standard deviations too deep and
#: lands in the operational first-order pack (396 +/- 60 m) that Cuxart
#: et al. diagnose as mixing far too much below the inversion; the
#: cross-isobar angle falls to 28.6 deg, which is that pack's own
#: signature (28 +/- 6) rather than the LES 35 +/- 3.  A second,
#: independent instrument agreeing with the (C_r/C_E)^2 argument above
#: from a completely different direction.
#:
#: (2) THE DEFICIT IS REAL, and is now measured against a published
#: number rather than a recollected observational band: the registered
#: constants give a peak subgrid energy of 0.0816 m2/s2 on this case
#: against an LES ~0.3 -- 3.7x low, independently reproducing the
#: registered "3-5x" claim.  It is not a bookkeeping artefact.
#:
#: (3) THE JOINT MOVE IS ADMISSIBLE AND STILL NOT WORTH TAKING.  Scaling
#: C_KS with C_E holds the depth inside the LES band all the way down,
#: exactly as the (C_r/C_E)^2 algebra predicts -- so the joint move IS
#: the right shape.  But it degrades the depth MONOTONICALLY
#: (+0.52 -> +0.94 -> +1.29 -> +1.60 sigma) while improving the energy
#: level monotonically (0.0816 -> 0.2082), and it never reaches the
#: observed e/u*^2 band at all (2.065 against 3.3-5.5) before the depth
#: has spent three quarters of its margin.  There is a real exchange
#: rate between the two, and the registered pair sits at the best point
#: on the quantity this benchmark measures most tightly.  The move is
#: REFUSED BY THE TARGET -- a measurement, not a caution.
#:
#: WHAT WOULD CHANGE THIS: a second published target that scores the
#: ENERGY LEVEL as tightly as GABLS1 scores the depth.  The exchange
#: rate above is what such a target would arbitrate, and it is
#: registered here so the arbitration is one step when one arrives.
#: Pinned by ``test_the_benchmark_refuses_the_tke_deficit_move``.
C_E = 0.93
C_K = 0.10
#: S3-6g regime-consistent turbulent Prandtl number (module docstring,
#: S3-6g section).  PR_LES is the LES-limit value -- the former fixed
#: PR_T = 1/3 of the Deardorff synthesis, correct where the filter
#: scale sits in an inertial range.  PR_RANS = 0.85 is the RANS-limit
#: value: the neutral-surface-layer / operational-PBL consensus
#: (observed K_h ~ 1.2*kappa*u*z against K_m ~ kappa*u*z gives
#: Pr ~ 0.85; MYJ/MYNN-class schemes carry 0.74-1.0 -- registered
#: approximation #1's own numbers).  The split path blends them on the
#: SAME f_used every other regime blend rides: Pr_t(f) =
#: f*PR_LES + (1-f)*PR_RANS (:func:`prandtl_blend`).
PR_LES = 1.0 / 3.0
PR_RANS = 0.85
#: Backward-compat alias (S3-6g): the v0 bit-frozen machinery
#: (``e_rhs``'s buoyancy K_h and the ``scalar_mix`` fixture
#: convention) keeps reading the historical name at the LES value.  NO
#: split-path consumer may read this alias -- the live channels take
#: :func:`prandtl_blend` at the step's used f (decision table, module
#: docstring).
PR_T = PR_LES
#: S3-6g momentum-background coefficient (decision table, module
#: docstring): the (1-f) equilibrium momentum background of
#: :func:`model_stress` and the Germano momentum-basis weight of
#: ``_solve_tail`` ride this FIXED constant -- the Task-5 neutral-K_m
#: calibration that was historically WRITTEN as C_K/PR_T.  That
#: spelling was a false coupling: 0.30 is a MOMENTUM-channel
#: calibration, not a scalar-channel Prandtl usage, so the regime
#: blend must not drag it.  Defined via the inherited expression
#: (C_K/PR_LES = 0.30000000000000004 in FP64) rather than a rounded
#: 0.3 literal so every frozen v0 fixture and golden stays bitwise;
#: registered standalone in the config ID.
C_MOM_BG = C_K / PR_LES
#: Realizability floors/ceilings (spec section 4.1).
#:
#: ``E_MIN`` and ``MAX_COLUMN_LEVELS`` are imported at the head of this
#: module from :mod:`woof.core.sase_limits` rather than restated, because
#: the config and state layers need them and cannot import this tree (see
#: that module).  One definition, and this module's constant registry and
#: configuration hash still bind them: ``sase_config_id`` resolves every
#: registered name against this namespace, and an imported name is in it.
CNU_MAX = 0.5
F_MIN, F_MAX = 0.0, 1.0
#: Stability-length coefficient l_s = LS_COEF*sqrt(e)/N (Deardorff).
LS_COEF = 0.76
#: Gravitational acceleration for the buoyancy source [m s^-2].
G_ACCEL = 9.81
#: von Karman constant (S3-6b vertical channel).  Single-sourced value
#: shared with the model side: ``woof.core.physics.KARMAN`` (WRF
#: share/module_model_constants.F ``karman`` = 0.4) and the sfclay/ysu
#: device kernels all use 0.4; a CPU test pins the two module constants
#: equal so they cannot drift.  (This module must stay cupy-free, so it
#: cannot import the physics module's copy directly.)
KARMAN = 0.4
#: Blackadar (1962) asymptotic mixing length [m]: the neutral
#: boundary-layer profile l_B = k*z/(1 + k*z/lambda) rises as k*z off
#: the surface and saturates at lambda aloft (S3-6b design envelope).
BLACKADAR_LAMBDA = 150.0
#: S3-6e horizontal-governor constants.  C_S is the WRF Smagorinsky
#: coefficient of the audited km_opt=4 path (``RunConfig.c_s`` default
#: 0.25, WRF module_diffusion_em smag2d; a CPU test pins the two values
#: equal so they cannot drift) and SMAG_KM_CAP the WRF K_m ceiling
#: factor (K_m <= 10*mlen, transcribed in ``npref.np_smag2d_km`` /
#: kernels/smag2d.cu).  Neither is re-derived here: the governor REUSES
#: the audited 2-D Smagorinsky deformation diffusivity
#: K_smag = min((C_S*delta)^2 * |D_h|, SMAG_KM_CAP*delta) with
#: |D_h| = sqrt((S_xx - S_yy)^2 + 4*S_xy^2) -- algebraically identical
#: to WRF's def2 = 0.25*(D11 - D22)^2 + D12^2 under D11 = 2*S_xx,
#: D22 = 2*S_yy, D12 = 2*S_xy -- evaluated on this module's audited
#: mass-point centered strain (the A-grid transcription of the C-grid
#: corner-averaged form; same continuum operator, same constants).
C_S = 0.25
SMAG_KM_CAP = 10.0
#: Guard for the smag-share weight r = nu_smag/max(nu, NU_BLEND_EPS) of
#: :func:`governed_stress`.  Since nu >= nu_smag >= 0 termwise, r is in
#: [0, 1] for ANY positive guard; the guard only prevents 0/0 where the
#: whole viscosity vanishes (and there the split it weights is 0 anyway).
NU_BLEND_EPS = 1.0e-30
#: S3-6e smoke-gate headroom factor: domain-max e_sgs must stay within
#: C_GATE times the 99.9th-percentile live-field equilibrium cap
#: (``woof.core.sase.sase_e_cap_stats`` documents the cap formula).
C_GATE = 3.0
#: Vertical-channel diffusion constant C_KV = C_E**(1/3) ~ 0.97613.
#: Log-layer consistency derivation: with K_v = C_KV*l_v*sqrt(e),
#: dissipation C_E*e^{3/2}/l_v, and l_v -> k*z near the surface, the
#: neutral constant-stress equilibrium (production K_v*S^2 =
#: dissipation, momentum flux K_v*S = u*^2) gives
#:     e   = (C_KV/C_E)*l_v^2*S^2,
#:     K_v = C_KV^{3/2}*C_E^{-1/2}*l_v^2*S,
#: and closing S = u*^2/K_v yields K_v = C_KV^{3/4}C_E^{-1/4}*l_v*u*.
#: K_v = k*u**z (the log law, kappa_eff = KARMAN) therefore requires
#: C_KV^3 = C_E exactly, whence e = C_E^{-2/3}*u*^2 ~ 1.05*u*^2 at
#: equilibrium.  The Deardorff LES constant C_K = 0.10 in this slot
#: would give K_v ~ 0.18*k*u**z (a 5.5x-too-steep log slope): the
#: RANS/wall channel is a different calibration regime from the
#: inertial-range LES channel and carries its own constant.  (Known
#: trade-off of one-constant e-l closures: matching the flux K_v puts
#: the e level at ~1.05 u*^2 vs the observed ~3.5 u*^2; the flux is
#: what acts dynamically, the e level is diagnostic.)
C_KV = C_E ** (1.0 / 3.0)
#: S3-6f partition-cap constants (module docstring, S3-6f section).
#: F_CAP_KNEE is the delta/z_i ratio below which the cap is FP-exact 1
#: (the LES/gray-zone side); F_CAP_WIDTH the Gaussian ramp width above
#: it.  Form rationale at the docstring: the (then-linear, pre-S3-9)
#: l_d blend demands f_cap <~ l_B/delta ~ 1e-3 by rho ~ 10, which the
#: algebraic 1/(1+x^2) class cannot reach; exp(-((rho-1)/2)^2) =
#: 1.6e-9 there.  (S3-9's geometric blend no longer needs that reach;
#: the registered Gaussian form deliberately stands unchanged.)
F_CAP_KNEE = 1.0
F_CAP_WIDTH = 2.0
#: S3-6f bulk-Richardson z_i constants: the single critical value
#: (Vogelezang-Holtslag-class; YSU's stable-BL brcr_sb, applied to all
#: regimes -- documented simplification, docstring) and the YSU wind
#: floor max(spd^2, 1 m^2 s^-2) transcribed from the audited
#: ``npref.np_ysu_column`` crossing.
RIB_CRIT = 0.25
RIB_WSPD2_FLOOR = 1.0
#: S3-6f w-sensor gravity-wave screen [s^-2]: cells with dry
#: N^2 > N2_SCREEN are excluded from the w-variance accumulation
#: (free-troposphere Brunt-Vaisala benchmark N ~ 0.01 s^-1; docstring).
N2_SCREEN = 1.0e-4
#: S3-6h Bougeault-Lacarrere (1989) combination-form constant: the
#: MIXING length combines the displacement pair through the negative
#: power mean l_mix = (0.5*(l_up^-p + l_down^-p))^(-1/p) with
#: p = BL89_MIX_EXP = 2/3 -- the Cuxart-Bougeault-Redelsperger (2000,
#: QJRMS 126, 1-30) form run operationally in Meso-NH/AROME (Seity et
#: al. 2011, MWR 139, 976-991).  Chosen over sqrt(l_up*l_down) (the
#: geometric mean = the p -> 0 limit): a negative-exponent mean is
#: dominated by the SMALLER displacement, so a parcel bounded on one
#: side (surface below, inversion above) mixes like its binding
#: constraint -- the conservative/suppressive side, which is the
#: amendment's purpose.  The DISSIPATION length takes the plain
#: minimum, l_eps = min(l_up, l_down) (BL89's l_epsilon; same
#: operational lineage) -- min <= the -2/3-power mean, so dissipation
#: is never weaker than mixing.  The module docstring (S3-6h section)
#: carries the full formulation.
BL89_MIX_EXP = 2.0 / 3.0
#: S3-6h registered CONVENTION strings (bound into the config ID like
#: numeric constants -- the gate evidence must pin these decisions):
#: * BL89_BETA_CONVENTION: the buoyancy factor of the displacement
#:   integral is beta = G_ACCEL/theta_v(z_k), the PARCEL level's own
#:   dry theta_v (= theta, L1 dry core; moist theta_v is SASE-M scope)
#:   -- no fixed reference temperature enters, matching the module's
#:   local-theta buoyancy convention (brunt_vaisala_n2 uses the same
#:   local normalization, so the uniform-stratification limit composes
#:   exactly: beta*dtheta/dz == N^2 levelwise).
#: * BL89_KZ_MATCH: the kappa-z-class near-surface floor/match is the
#:   min with the Blackadar length at ALL levels (l_B -> kappa*z near
#:   the surface enforces the log-layer match; l_B -> BLACKADAR_LAMBDA
#:   aloft retains the audited neutral asymptote).  In a neutral column
#:   the BL89 lengths are pure geometry (l_down = z, l_up = htop - z),
#:   both >= l_B except within ~l_B of the model top, so the neutral
#:   log-layer fixture window is BITWISE untouched.
#: * BL89_LS_DECISION: l_s = LS_COEF*sqrt(e)/N is RETAINED as the
#:   outer stability min in the RANS limb.  Adjudication (fixture
#:   test_bl89_uniform_stratification_sqrt_e_over_n_limit): the BL89
#:   integral reproduces sqrt(e)/N scaling in uniform stratification
#:   with coefficient sqrt(2) (Rodier et al. 2017, JAS 74: l_up =
#:   l_down = sqrt(2e)/N), which EXCEEDS the audited Deardorff 0.76 --
#:   retiring l_s would therefore LENGTHEN the stable-limit RANS
#:   mixing length 1.86x in exactly the regime this lane is
#:   suppressing.  Retaining it costs BL89 nothing new: the integral's
#:   fresh information is NON-LOCAL (a parcel is stopped by the
#:   integrated buoyancy of an inversion it has not yet reached --
#:   invisible to the local-N l_s), and the composition
#:   min(l_B, l_s, l_BL89) engages whichever bound is tightest.
BL89_BETA_CONVENTION = "g-over-local-parcel-thetav"
BL89_KZ_MATCH = "min-with-blackadar-all-levels"
BL89_LS_DECISION = "retain-ls-as-outer-min"
#: S3-6i decoupled stable-limit diffusivity coefficient (module
#: docstring, S3-6i section): where the Deardorff stability length
#: l_s = LS_COEF*sqrt(e)/N binds the RANS-limb mixing length, the
#: vertical diffusivity asymptotes to K_v = C_KS*e/N instead of the
#: coupled C_KV*LS_COEF*e/N = 0.742*e/N (the neutral-wall log-layer
#: constant riding l_s ~10x above Deardorff's own stable composite
#: 0.076*e/N -- the S3-6h-isolated driver of the inner nest's spurious
#: morning jet mixdown).  CALIBRATED against the jet-decoupling
#: fixture at its
#: registered GREEN criteria (10 m wind in the obs band [4.5, 7.0] m/s
#: through 3600 s AND jet-core hold within 15%); the registered sweep
#: {0.076, 0.10, 0.15, 0.20, 0.25} selects the LARGEST value passing
#: with margin -- the weakest intervention that meets the
#: observations.  MEASURED (frame-true E0, dt = 60, 3600 s; full
#: table + trajectory at the fixture
#: test_jet_decoupling_stable_coefficient_holds_obs_band): every
#: swept value passes both criteria -- u10 max 6.580/6.480/6.378/
#: 6.276/6.227 for C_KS = 0.25/0.20/0.15/0.10/0.076, jet-core
#: deviation <= 0.06% throughout -- so the selection rule lands on
#: 0.25 (u10 in [5.70, 6.58], 0.42 m/s of headroom below the band
#: top; the coupled RED reference exits the band at 480 s and ends at
#: 7.356).  0.25 is also the top of the operational-composite band,
#: i.e. the most conservative decoupling consistent with obs.
C_KS = 0.25
#: S3-6i coefficient-blend smoothness exponent: the coefficient
#: deficit grows as rho**CKS_BLEND_EXP with rho = min(l_mix_rans/l_s,
#: 1).  2.0 is the MINIMAL power that makes K_v C^1 in the signed
#: frequency N = sqrt(n2) across the neutral<->stable transition
#: (rho is linear in N near neutral, so the deficit ~ N^2 and
#: dK_v/dN -> 0 at N -> 0+, matching the N-independent neutral side;
#: exponent 1 would leave a kink in K(N) at N = 0).  In the MODEL
#: INPUT n2 the blend is C^0 only -- the deficit is LINEAR in n2 at
#: onset, so dK_v/d(n2) jumps at neutral (claim narrowed in S3-9c
#: per the adversarial review, Minor 3; module docstring, S3-6i section).
#: :func:`stable_limit_coefficient` has the form.
CKS_BLEND_EXP = 2.0
#: S3-6k DECOUPLED STABLE-LIMB DISSIPATION COEFFICIENT (module
#: docstring, S3-6k section; gated by RunConfig.sase_stable_dissipation,
#: DEFAULT False -- the constant is registered unconditionally but only
#: the switched-on path reads it).  Where the Deardorff stability length
#: l_s = LS_COEF*sqrt(e)/N binds the DISSIPATION length, S3-6i left the
#: coefficient at the neutral-wall C_E and said so in its own words
#: ("DISSIPATION UNTOUCHED").  That leaves eps = (C_E/LS_COEF)*e*N =
#: 1.2237*e*N in a regime C_E was never calibrated for.
#:
#: PROVENANCE (verified this session against a published implementation,
#: not remembered).  Deardorff (1980, Bound.-Layer Meteorol. 18:495-527)
#: pairs the 0.76*sqrt(e)/N stability length -- LS_COEF, already
#: registered above -- with the length-dependent SGS dissipation
#: coefficient c_eps = c_eps1 + c_eps2*lambda/Delta.  Transcribed in
#: Heus et al. (2010, Geosci. Model Dev. 3:415-444), the DALES
#: formulation paper: eq. (21) eps = (c_eps/lambda)*e^{3/2}, eq. (22)
#: lambda = min(Delta, c_N*sqrt(e)/N), eq. (24) c_eps = c_eps,1 +
#: c_eps,2*lambda/Delta, Table 2 c_eps,1 = 0.19, c_eps,2 = 0.51,
#: c_N = 0.76 -- introduced there as "Still following Deardorff (1980),
#: the SFS parameters are depending on the stability of the flow".  The
#: PALM model's SGS documentation carries the same asymptote with a
#: different filter-scale slope (0.19 + 0.74*l/Delta), so the two
#: independent transcriptions agree on exactly the member this constant
#: is: c_eps,1 = 0.19, the lambda << Delta limit -- which is precisely
#: the regime where l_s binds, and the SAME length composition
#: min(Delta, 0.76*sqrt(e)/N) this module's :func:`dissipation_length`
#: already implements.  ASSIGNED ONCE FROM THE CITATION; NEVER SWEPT.
#:
#: MEASURED CHECK, computed AFTER the value was chosen, not fitted to it
#: (a throwaway reflen.py under the judge's own temp directory, on the
#: trusted MYNN reference forecast frame, scoring box B
#: land, 4808 columns): the coefficient that reproduces the reference
#: scheme's OWN dissipation q^3/(B1*EL_PBL) when applied to SASE's OWN
#: l_s is p50 0.2101 (subcloud 100-600 m, n = 38464) and p50 0.1827
#: (deck band 400-1400 m, n = 28839), with B1 = 23.25 measured from the
#: reference's own surface layer.  0.19 lands between them.  The
#: diffusivity half of the pair needs no such move: the reference's
#: K_h/(e/N) is 0.3111/0.3578 against C_KS/PR_RANS = 0.29412 (6-18%).
#:
#: REGISTERED ALTERNATIVE, rejected so it can never land silently:
#: rescaling the published RATIO onto this module's own neutral
#: constant -- C_E*(0.19/(0.19 + 0.51)) = 0.25243 -- rather than
#: transcribing the published asymptote.  Rejected because it is a
#: derived quantity with no independent calibration, and because 0.19
#: is the value both published implementations state.
#:
#: DERIVED CONSEQUENCE, registered here so no future coefficient move
#: shifts it silently: the stable-limb critical Richardson number
#: Ri* = C_KS/(C_eps/LS_COEF + C_KS/PR_RANS) becomes 0.45945945945945943
#: (HEAD 0.16471188169301373; pre-S3-6i, when the limb carried
#: C_KV*LS_COEF, 0.35385638343882664).  This does NOT remove the limb's
#: linearity in e -- see the S3-6k docstring section.
#:
#: THE SPELLING IS THE LITERAL, DELIBERATELY (S3-6k landing amendment).
#: C_ES and C_KS*LS_COEF are BITWISE EQUAL -- both 0.19,
#: ``52b81e85eb51c83f``, measured this session -- because C_KS = 2^-2
#: exactly, so the product is a pure exponent shift of LS_COEF's
#: mantissa (LS_COEF is ``52b81e85eb51e83f``: same mantissa, exponent
#: lower by 2).  That is a COINCIDENCE of two independently published
#: two-digit constants, not a derivation, and C_ES is NOT spelled
#: ``C_KS * LS_COEF`` here: doing so would make Deardorff's published
#: asymptote a silent dependent of this module's own diffusivity
#: calibration (the false coupling ``C_MOM_BG``'s comment warns about)
#: and would degrade the anti-tuning tripwire's independent
#: ``C_ES/LS_COEF == 0.25`` check into a restatement of C_KS.  What the
#: coincidence buys is ALGEBRA, registered in the S3-6k docstring
#: section: it places this composition exactly on the Gamma_m = 1 line,
#: where Ri* = PR_RANS/(1 + PR_RANS) independently of C_KS.
C_ES = 0.19
#: S3-9 registered CONVENTION string (bound into the config ID like a
#: numeric constant -- the blend FORM is formulation-level gate
#: evidence, the S3-6h string idiom): the l_d regime blend of
#: :func:`dissipation_length` is the GEOMETRIC composition
#: delta**f * lb**(1-f) (log-space interpolation; the F-Y1
#: cold-water-body over-coupling amendment -- module docstring, S3-9
#: section).  The
#: pre-S3-9 decision this replaces was the linear f*delta + (1-f)*lb.
LD_BLEND_FORM = "geometric"
#: REJECTED option, recorded so a future flip can never land silently:
#: taking the Deardorff stability length l_s = LS_COEF*sqrt(e)/N OFF the
#: dissipation length and leaving it only on the mixing length.
#:
#: THE DEFECT IT WAS AIMED AT IS REAL.  l_s bounds BOTH the mixing
#: length (bl89_rans_lengths, BL89_LS_DECISION) and the dissipation
#: length (the trailing np.minimum(l, ls) of dissipation_length).
#: Where it binds both, the two competing terms are linear in e
#: TOGETHER:
#:
#:     K_v = C_r*l_s*sqrt(e)   = LS_COEF*C_r*e/N
#:     eps = C_E*e^{3/2}/l_s   = (C_E/LS_COEF)*e*N
#:
#: so the subgrid-energy equation is exactly HOMOGENEOUS -- its
#: specific rate
#:
#:     (1/e) de/dt = N*[LS_COEF*C_r*(1/Ri - 1/Pr_t) - C_E/LS_COEF]
#:
#: carries no e at all, and the closure has NO EQUILIBRIUM AMPLITUDE.
#: Below Ri_c = 1/(1/Pr_t + C_E/(LS_COEF^2*C_r)) = 0.1313 on the stable
#: limb (C_KS) and 0.3539 on the neutral one (C_KV), subgrid energy
#: grows EXPONENTIALLY, bounded only by the resolved shear being mixed
#: away and never by the closure.  Verified to relative 2.4e-15 across
#: eight decades of e, closed form matched to a bisection on the rate
#: itself to six decimals (the homogeneity pin in tests/test_sase.py).
#:
#: MEASURED on real GFS through `woof go` (d01 12 km, 2026-08-01T18):
#: l_s bound in 65-100% of the cells the closure had lit up at every
#: level from 9.2 to 14.3 km -- median l_s of 0.2-3.2 m against a
#: Blackadar length of ~146 m -- so the homogeneous regime is where the
#: closure actually operates.  Energy at 11-14 km reached 1.62 m2/s2,
#: 62% of the boundary-layer peak, still growing at t+60 min, in cells
#: 201-255x enriched in Ri < 0.25.  Those cells are 100% ICE-ONLY cloud
#: at k >= 33, which is why the liquid-keyed SASE-M1 switch
#: (MOIST_STABILITY_SWITCH, "qc > 0 OR RH100 liquid") is inert there and
#: why ablating it moved nothing above 8 km.
#:
#: WHY REMOVING THE MIN IS NOT THE FIX.  It does break the homogeneity
#: exactly as derived: with l_eps carrying no e, dissipation goes as
#: e^{3/2} against a production linear in e, the fixed point
#: sqrt(e_eq) = (l_eps/C_E)*N*LS_COEF*C_r*(1/Ri - 1/Pr_t) is finite for
#: every Ri < Pr_t, and NO Richardson number bounds a growth regime any
#: more -- the threshold moves to Ri = Pr_t and changes character, from
#: runaway onset to turbulence existence.  But removing a MIN can only
#: LENGTHEN l_d, and eps = C_E*e^{3/2}/l_d, so it can only WEAKEN
#: dissipation -- never strengthen it.  Measured weakening: 2000x at
#: e = 1e-6, 190x at 1e-4, 19x at 1e-2, 1.5x at 1.62 (N^2 = 1e-4).  It
#: therefore degrades every regime where the pre-change decay was
#: correct, and two registered calibration fixtures say so: the jet
#: decoupling fixture's 10 m wind leaves its obs band [4.5, 7.0] m/s
#: (7.93 -> 9.33 m/s) and the lake internal boundary layer collapses
#: (ibl_d74 10.0 -> 2.54).  Those are the regimes S3-6i and S3-6k were
#: built to hold; this trades them for the free troposphere.
#:
#: WHAT A REAL FIX HAS TO DO.  Bound the amplitude by ADDING an e^{3/2}
#: dissipation channel, not by removing the e-linear one -- e.g.
#: Deardorff's own length-dependent coefficient c_eps = C_1 + C_2*l/L
#: with C_1 = 0.19 (already registered as C_ES) -- so that dissipation
#: is nowhere weaker than it is today.  Not attempted here: it moves
#: C_ES into the default path, and C_ES/C_KS are recorded as jointly
#: falsified pending a stable-limb calibration target that does not yet
#: exist.  Same blocker as the C_E deficit above, and the same reason
#: not to invent a number to make one case land.
LD_STABILITY_LIMIT_REJECTED = "off-dissipation-length-v1"
#: S3-12 ADDITIVE e^{3/2} DISSIPATION CHANNEL -- the second (grid-scale)
#: member of Deardorff's length-dependent coefficient (module docstring,
#: S3-12 section; gated by ``RunConfig.sase_additive_dissipation``).
#:
#: PROVENANCE, the SAME citation chain C_ES above already carries and
#: therefore the same verified transcription: Deardorff (1980,
#: Bound.-Layer Meteorol. 18:495-527) pairs the 0.76*sqrt(e)/N stability
#: length with c_eps = c_eps,1 + c_eps,2*lambda/Delta; DALES (Heus et al.
#: 2010, Geosci. Model Dev. 3:415-444) eq. (24) and Table 2 give
#: c_eps,1 = 0.19 (registered as C_ES) and c_eps,2 = 0.51 -- THIS
#: constant.  Written out, that coefficient is two ADDITIVE dissipation
#: channels,
#:
#:     eps = c_eps,1*e^{3/2}/lambda  +  c_eps,2*e^{3/2}/Delta,
#:
#: and only the SECOND one carries no length that depends on e.  That is
#: the whole mechanism: Delta is the model's own scale, fixed by the
#: configuration and not by the state, so this channel is genuinely
#: e^{3/2} where the first is e-linear under a stability length
#: (LD_STABILITY_LIMIT_REJECTED).
#:
#: WHY THIS MEMBER AND NOT THE FIRST.  C_ES replaces C_E and can only
#: LOWER the e-linear channel (0.19 against 0.93): measured RED on the
#: registered jet gate, which is why RunConfig.sase_stable_dissipation
#: ships False.  C_ED is ADDED to it and is non-negative, so
#: dissipation under this switch is NOWHERE WEAKER than HEAD, pointwise,
#: and is bitwise EQUAL wherever the gate does not fire
#: (test_additive_dissipation_is_nowhere_weaker).
#:
#: THE SECOND PUBLISHED TRANSCRIPTION DISAGREES ON THIS MEMBER, recorded
#: rather than averaged away: PALM's SGS documentation carries the same
#: asymptote with a steeper filter-scale slope, 0.19 + 0.74*l/Delta, so
#: the two independent transcriptions agree on c_eps,1 = 0.19 and differ
#: on c_eps,2 (0.51 vs 0.74).  0.51 is registered because it is the
#: DALES/Deardorff pairing that also supplies C_ES and LS_COEF -- one
#: citation, one family -- and because it is the WEAKER intervention:
#: c_eps,2 enters the stable-limb fixed point as e_eq ~ (1/c_eps,2)^2,
#: so 0.51 gives the HIGHER equilibrium and therefore claims less.  The
#: 0.74 alternative is registered here and NOT swept: sweeping it against
#: a fixture is the fitting this module does not do.  ASSIGNED ONCE FROM
#: THE CITATION.
C_ED = 0.51
#: S3-12 registered CONVENTION string (bound into the config ID like a
#: numeric constant -- the FORM is formulation-level gate evidence, the
#: S3-6h/S3-9 string idiom).  Deardorff's second channel divides by
#: Delta -- "the model's own scale", which in an LES is the filter
#: width.  This closure spans two regimes on one partition weight f, so
#: the channel rides the SAME geometric blend :func:`dissipation_length`
#: rides, evaluated on the NEUTRAL member of each limb:
#:
#:     l_ref = delta**f * l_B(z + z0)**(1 - f)
#:
#: (:func:`neutral_dissipation_length`).  At f = 1 that is delta, the
#: filter width -- Deardorff's own Delta, FP-exact.  At f = 0 it is the
#: Blackadar length, this closure's registered neutral master length
#: (BLACKADAR_LAMBDA), which is what the RANS limb's energy-containing
#: scale actually is at mesoscale spacing.
#:
#: THE SELECTION RULE, stated so it is checkable: the additive channel
#: rides the STATE-INDEPENDENT member of the same length composition the
#: e-linear channel rides.  l_d's RANS input is
#: l_eps_rans = min(l_B, l_eps_BL89); of that pair only l_B carries no e
#: (BL89 displacement lengths solve an energy integral and scale as
#: sqrt(e), exactly like l_s, so a channel divided by them is e-linear
#: again and breaks NO homogeneity -- measured, the S3-12 docstring
#: section).  Dropping the state-dependent member is therefore not a
#: convenience: it is the entire content of the fix.
#:
#: CONSEQUENCE, registered: l_eps_rans <= l_B and l_d <= the blend, so
#: l_d/l_ref <= 1 always and the effective coefficient is bounded by
#: C_E + C_ED = 1.44 -- it can never exceed Deardorff's own neutral
#: total scaled onto this module's log-law-anchored C_E.
LD_ADDITIVE_CHANNEL = "deardorff-c2-neutral-reference-length-v1"
#: S3-6j surface-layer wind-speed floor [m/s]: the RESOLVED-speed
#: regularizer of the drag linearization c = u*^2/max(|V1|,
#: SFC_WSPD_FLOOR) (module docstring, S3-6j section).  The VALUE is
#: transcribed from the audited sfclay convention -- WRF
#: module_sf_sfclayrev ``wspd = amax1(wspd, 0.1)``, carried by
#: ``npref.np_sfclay`` (max(..., 0.1)) and kernels/sfclay.cu (fmaxf
#: against 0.1f) -- so the SASE drag row can never divide by a
#: vanishing surface wind.  S3-9c clarification (adversarial review,
#: IMPORTANT-1): sfclay's own floor lives on its gust-ENHANCED wspd,
#: which enters this module as the S3-9c correction-factor
#: denominator (``wspd_sfc``); this constant regularizes the resolved
#: |V1| in the linearization only (YSU regularizes its wspd1 with
#: +1e-9 instead -- the registered deviation, module docstring, S3-9c
#: section).  At |V1| >= 0.1 m/s the floor is inert and c carries the
#: exact YSU-class linearized stress coefficient.
SFC_WSPD_FLOOR = 0.1
#: S3-11a heat capacity of dry air at constant pressure
#: [J kg^-1 K^-1]: the sensible-heat conversion of the surface
#: scalar-flux deposit, HFX/(rho1*CP_AIR) [K m s^-1].  Single-sourced
#: VALUE: the model side's ``woof.core.constants.CP`` = 7*RD/2 =
#: 1004.5 (WRF share/module_model_constants cp) -- the same constant
#: sfclay used to FORM the HFX this seam consumes and the e source
#: uses to convert it; 7.0*287.0/2.0 is FP64-exact at 1004.5, so the
#: literal is bitwise the model constant, and a CPU test pins the two
#: equal so they cannot drift.  (This module stays cupy-free by
#: charter; the constants module is import-safe, but the module
#: convention -- KARMAN, C_S -- is an owned copy plus a pin test.)
CP_AIR = 1004.5
#: S3-11a registered CONVENTION string (config-ID-bound like the
#: S3-6h/S3-9 strings -- the seam FORM is formulation-level gate
#: evidence): the surface scalar fluxes enter the scalar vertical
#: channel as the EXPLICIT lowest-layer deposit
#: :func:`surface_scalar_flux_deposit` BEFORE the implicit K_v/Pr_t
#: solve (module docstring, S3-11a section; the audited YSU
#: surface-rhs rows npref.py:6472/6481 in pre-deposit form -- the
#: root-cause file's fix candidate 1, chosen over the implicit
#: bottom-flux-row candidate 2).
SFC_SCALAR_FLUX = "explicit-deposit-v1"
#: SASE-M1 moist-thermodynamics constants (module docstring, SASE-M1
#: section).  Values single-sourced with the model side
#: (``woof.core.constants`` = WRF share/module_model_constants),
#: exactly the KARMAN/C_S/CP_AIR owned-copy-plus-pin-test convention:
#: RD_AIR/RV_AIR are the dry-air/vapor gas constants (model RD, RV),
#: EP2_RV = RD_AIR/RV_AIR the epsilon = R_d/R_v ratio (model EP2 --
#: the SAME expression, so the FP64 value is bitwise the model
#: constant), XLV the latent heat of vaporization at 0 C (model XLV),
#: SVP1/SVP2/SVP3/SVPT0 the Tetens LIQUID saturation-vapor-pressure
#: constants (SVP1 in kPa; the audited model spelling is
#: es[Pa] = 1000*SVP1*exp(SVP2*(T - SVPT0)/(T - SVP3)) -- the SAME
#: e_s the microphysics used to FORM the qc the M1 switch consumes),
#: and P0_REF the Exner reference pressure (model P0).  A CPU test
#: pins each equal to the model constant so they cannot drift.
RD_AIR = 287.0
RV_AIR = 461.6
EP2_RV = RD_AIR / RV_AIR
XLV = 2.5e6
SVP1 = 0.6112
SVP2 = 17.67
SVP3 = 29.65
SVPT0 = 273.15
P0_REF = 1.0e5
#: SASE-M1 registered CONVENTION strings (config-ID-bound, the
#: S3-6h/S3-9/S3-11a idiom, registered BEFORE implementation --
#: module docstring, SASE-M1 section, has the formulation and the
#: DK82 derivation):
#: * MOIST_STABILITY: the saturated stability form is the Durran &
#:   Klemp (1982, JAS 39, 2152-2158) Eq.-(36) saturated N^2 with
#:   condensate loading (:func:`moist_n2`).
#: * MOIST_STABILITY_SWITCH: the v1 saturation switch is the BINARY
#:   cell test qc > 0 OR RH >= 100% with respect to liquid
#:   (implemented as qv >= q_s,liq(T, p), monotone-equivalent); the
#:   assumed-PDF cloud-fraction blend is the registered later hook
#:   (spec section 3) and must re-register this string when it lands.
MOIST_STABILITY = "dk82-saturated-v1"
MOIST_STABILITY_SWITCH = "binary-qc-or-rh100-liquid"
#: SASE-M1b registered CONVENTION string (spec section 3b, the frozen
#: S4-3 G-M3 amendment; module docstring, SASE-M1b section, has the
#: derivation): in M1-substituted cells the RANS-limb master lengths
#: are ADDITIONALLY min-bounded by the moist parcel-excursion length
#: l_m = min(l_up_m, l_down_m) of :func:`bl89_moist_excursion_lengths`
#: -- BL89-family up/down excursion integrals evaluated against the
#: n2_eff (moist_n2) field with piecewise-constant face-mean segment
#: stratification, combined by the family's min (l_eps) member.  Zero
#: new tunable constants by construction.
MOIST_MASTER_LENGTH = "bl89-n2eff-excursion-min-v1"
#: SASE-M2 registered CONVENTION strings + constants (module docstring,
#: SASE-M2 section, has the full formulation, derivations, and
#: provenance; registered BEFORE implementation, the S4-1 idiom):
#: * VENT_FORM: the conditional venting limb's registered form -- a
#:   diagnosed NB-terminated entraining-plume flux profile
#:   (:func:`plume_vent_flux`).
#: * VENT_MASK: the registered reading of the spec's "saturated layer
#:   with d(theta_es)/dz < 0 through a finite depth" -- the S4-1 mask
#:   obligation.  "bulk-theta-es-v1" (chosen: run-top minus run-base
#:   theta_es < 0 over the lowest >= 2-cell saturated run; the reading
#:   the plume's own parcel ascent confirms) vs the implemented
#:   alternative "per-level-theta-es-v1" (theta_es decreasing at EVERY
#:   adjacent pair -- vetoes the measured 11Z specimen, which carries
#:   interior theta_es rises; the flip is fixture-pinned).
VENT_FORM = "nb-terminated-vent-v1"
VENT_MASK = "bulk-theta-es-v1"
#: SASE-M2 amplitude coefficient: Grant (2001, QJRMS 127, 407-421)
#: cloud-base mass-flux closure M_b = 0.03*w, carried VERBATIM with the
#: velocity scale re-keyed to the column's subgrid sigma_w (C8: no
#: surface-w* dependence; module docstring, SASE-M2 AMPLITUDE).
VENT_MB_COEF = 0.03
#: SASE-M2 entrainment constant: c_eps of Siebesma, Soares & Teixeira
#: (2007, JAS 64, 1230-1248), Eq. (16) eps ~ c_eps*(1/z + 1/(z_top-z)),
#: "...can be well fitted with [Eq. 16] with c_eps ~ 0.4" (p. 1236) --
#: eps = c_eps/z of the established eps ~ 1/z family (their
#: approach-to-top member 1/(z_top-z) is a registered dropped
#: simplification -- the hard NB termination supersedes it).  S4-4
#: review Important-1 CORRECTION: the value shipped at S4-4 authority
#: (0.55) was misattributed to this paper -- SST07's own value is
#: 0.4 (verified from the paper PDF, journals.ametsoc.org bronze OA,
#: jas3888.1); 0.55 has no located primary source and is retired.
VENT_ENT_COEF = 0.4
#: SASE-M2 seed-velocity share: sigma_w^2 = (2/3)*e, the isotropic
#: w-variance share of TKE (e = (3/2)*sigma_w^2 by definition of
#: isotropy) -- the C8-compliant velocity scale the Grant closure rides.
VENT_SIGW_SHARE = 2.0 / 3.0
#: SASE-M2 shallow-device scope guard [m]: no NB within this depth of
#: the root -> the limb stands down (deep CI belongs to the resolved
#: grid).  Value: the amplifier-anatomy constraint-8 detrainment
#: ceiling ("~2.5-4 km") at its upper edge; a scale-separation bound
#: (C13 scope line), not an amplitude tune.
VENT_DEPTH_CAP = 4000.0
#: SASE-M2 registered per-step deposit bound [K/step]: the S3-11a
#: measured stable-deposit class (module docstring, S3-11a section:
#: |dtheta_1| ~ 0.14 K per intermediate-nest step at the pinned
#: defect state).  The
#: S4-5 deposit seam enforces it by a UNIFORM per-column rescale of
#: all three flux profiles (module docstring, SASE-M2 RATE CAP).
VENT_THETA_STEP_CAP = 0.14
#: SASE-M2 registered per-step deposit bound [kg/kg/step] for the
#: MOISTURE rows (qv, qc) -- S4-4 review Important-4 fix: the cap
#: previously bounded theta only, leaving the moisture deposit
#: unbounded (measured x16.5 headroom vs the theta cap at the theta
#: cap's own limit).  DERIVED, not a second independent tunable: the
#: SAME registered VENT_THETA_STEP_CAP converted through the latent-
#: sensible heat equivalence already used throughout this module
#: (L*dq = cp*dtheta for an equal energy deposit), using the two
#: constants moist_n2/plume_vent_flux already share (CP_AIR, XLV):
#:
#:     VENT_QT_STEP_CAP = VENT_THETA_STEP_CAP * CP_AIR / XLV
#:
#: The S4-5 seam applies the SAME uniform per-column rescale factor to
#: all three rows -- min(1, VENT_THETA_STEP_CAP/|dtheta|_max,
#: VENT_QT_STEP_CAP/|dqv|_max, VENT_QT_STEP_CAP/|dqc|_max) -- so
#: theta, qv, AND qc each bound INDIVIDUALLY (S4-4 review round-3 CAP
#: FAMILY fix: a sum-only |dqv+dqc| test is vacuous on pure-phase-
#: conversion columns, where F_qv = -F_qc cancels the sum while either
#: row alone still scales with the amplitude), all from the one
#: registered constant (module docstring, SASE-M2 RATE CAP -- that
#: section also documents the Exner-factor looseness this derivation
#: accepts and the divide-by-zero guard the S4-5 seam needs).
VENT_QT_STEP_CAP = VENT_THETA_STEP_CAP * CP_AIR / XLV
#: SASE-M2 registered numerics (S4-4 review Minor-1: previously
#: documented in prose but unregistered).  VENT_MIN_RUN_CELLS: the
#: minimum contiguous saturated-run depth for M2 engagement (module
#: docstring, SASE-M2 MASK -- "a finite depth", >= 2 cells so the
#: mask's bulk/per-level theta_es readings are even well-defined).
VENT_MIN_RUN_CELLS = 2
#: VENT_SAT_ADJUST_ITERS: the fixed Newton iteration count of
#: :func:`_vent_saturation_adjust` (deterministic op order for the
#: future device mirror; bit-stationary by 8 iterations on the
#: specimen, module docstring SASE-M2 ASCENT / the function's own
#: derivation).
VENT_SAT_ADJUST_ITERS = 12
#: VENT_K_LID_MEMBERSHIP: the registered convention for M2 LAYER
#: MEMBERSHIP (S4-4 review round-3, project decision -- design doc
#: SASE-M2 amendment "discrete C9 reading"; EXTENDED round-4 from the
#: run top alone to the whole structural layer).  A cell is a member
#: iff the noise-immune TOTAL-WATER test qt = qv + qc >= qs holds,
#: never on the M1 mask's bit-level qc > 0 limb (which also fires on
#: sub-1e-12 kg/kg condensate dust -- module docstring, SASE-M2
#: MEMBERSHIP, has the derivation and the measured amplitude swings).
#: The member run supplies the base, the contiguity and the top k_top
#: (hence the entrainment-zone cell and k_lid); the root k_r comes
#: from the theta_es structure and the ebar window from the root
#: through the entrainment-zone cell, so NOTHING structural rides
#: ``n2m_mask`` any more -- the M1 mask is kept only as a veto.  Only
#: one convention is implemented (unlike VENT_MASK's two); the string
#: is registered so a future alternative can never land silently.
VENT_K_LID_MEMBERSHIP = "qt-ge-qs-v1"
#: VENT_ANCHOR_RULE: the registered convention for the M2 SHAPE ANCHOR
#: and the stand-down it implies (S4-5c; design doc SASE-M section 4,
#: amendment "a surface-based saturated layer stands the limb down",
#: audit-wave project decision).  "cloud-base-face-standdown-v1": the
#: grow-zone shape normalizes on the cloud-base face z_f[k_base], and a
#: run based in the LOWEST MODEL LEVEL -- for which no such face exists,
#: since z_f[0] = 0 identically -- stands the limb down (bitwise +0.0 on
#: all three rows), the fourth registered stand-down condition alongside
#: no-LFC, VENT_DEPTH_CAP and k_lid past the column top.  Zero new
#: tunable constants: the rule is the index test k_base > 0.
#:
#: THIS STRING IS THE WHOLE REASON THE CONSTANT EXISTS.  The rule is
#: bitwise-inert on every registered CPU and device column (measured,
#: S4-5c: 78 corpus columns, 0 bytes moved), so nothing else would have
#: rotated the config ID -- ``sase_config_id`` hashes constant VALUES,
#: not code, and the semantic change would otherwise have shipped under
#: the pre-amendment identity and stamped it on new receipts.  The
#: alternative reading, "cloud-base-face-mhat1-v0" (M_hat == 1 on the
#: branch instead of a stand-down), is registered here as the rejected
#: option so a future flip can never land silently.
VENT_ANCHOR_RULE = "cloud-base-face-standdown-v1"

#: Config-ID registry: every physics constant in this module must appear
#: here.  ``sase_config_id`` resolves each name against the live module
#: namespace, so a constant added above but omitted from this tuple is
#: the only way to leave gate evidence unbound -- keep them adjacent.
_CONFIG_ID_CONSTANTS = ("C_E", "C_K", "PR_LES", "PR_RANS", "C_MOM_BG",
                        "E_MIN", "CNU_MAX",
                        "F_MIN", "F_MAX", "LS_COEF", "G_ACCEL",
                        "KARMAN", "BLACKADAR_LAMBDA", "C_KV",
                        "C_S", "SMAG_KM_CAP", "NU_BLEND_EPS", "C_GATE",
                        "F_CAP_KNEE", "F_CAP_WIDTH", "RIB_CRIT",
                        "RIB_WSPD2_FLOOR", "N2_SCREEN",
                        "BL89_MIX_EXP", "BL89_BETA_CONVENTION",
                        "BL89_KZ_MATCH", "BL89_LS_DECISION",
                        "C_KS", "CKS_BLEND_EXP", "C_ES", "LD_BLEND_FORM",
                        "C_ED", "LD_ADDITIVE_CHANNEL",
                        "SFC_WSPD_FLOOR", "CP_AIR", "SFC_SCALAR_FLUX",
                        "RD_AIR", "RV_AIR", "EP2_RV", "XLV",
                        "SVP1", "SVP2", "SVP3", "SVPT0", "P0_REF",
                        "MOIST_STABILITY", "MOIST_STABILITY_SWITCH",
                        "MOIST_MASTER_LENGTH",
                        "VENT_FORM", "VENT_MASK", "VENT_MB_COEF",
                        "VENT_ENT_COEF", "VENT_SIGW_SHARE",
                        "VENT_DEPTH_CAP", "VENT_THETA_STEP_CAP",
                        "VENT_QT_STEP_CAP", "VENT_MIN_RUN_CELLS",
                        "VENT_SAT_ADJUST_ITERS", "VENT_K_LID_MEMBERSHIP",
                        "VENT_ANCHOR_RULE")


def sase_config_id() -> str:
    """SHA-256 over the closure constants; binds gate evidence to them."""
    payload = {name: globals()[name] for name in _CONFIG_ID_CONSTANTS}
    payload["scheme"] = "sase-l1"
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


_FILTER_WEIGHTS = {
    2: np.array([0.25, 0.5, 0.25], dtype=np.float64),
    4: np.array([0.125, 0.25, 0.25, 0.25, 0.125], dtype=np.float64),
}


def _filter_axis(field: np.ndarray, weights: np.ndarray,
                 axis: int) -> np.ndarray:
    half = len(weights) // 2
    out = np.zeros_like(field)
    for offset, w in zip(range(-half, half + 1), weights):
        out += w * np.roll(field, -offset, axis=axis)
    return out


def box_filter(field: np.ndarray, width: int) -> np.ndarray:
    """Top-hat test filter of nominal width ``width``*grid, x then y.

    Horizontal directions wrap periodically (the reference box is
    periodic); the vertical direction is deliberately untouched -- the L1
    sensor and dynamic solve are horizontal-scale operators (spec 4.1).
    """
    weights = _FILTER_WEIGHTS[width]
    out = _filter_axis(np.asarray(field, dtype=np.float64), weights, axis=2)
    return _filter_axis(out, weights, axis=1)


def structure_functions(u, v, w) -> dict[int, float]:
    """Domain-mean D2(r), r in {1,2,4} cells, horizontal directions only."""
    out: dict[int, float] = {}
    for r in (1, 2, 4):
        total = 0.0
        for comp in (u, v, w):
            comp = np.asarray(comp, dtype=np.float64)
            dx_inc = np.roll(comp, -r, axis=2) - comp
            dy_inc = np.roll(comp, -r, axis=1) - comp
            total += 0.5 * (np.mean(dx_inc**2) + np.mean(dy_inc**2))
        out[r] = float(total)
    return out


@dataclass(frozen=True)
class SensorState:
    alpha: float
    slope: float
    e_res: float


def sensor_state(u, v, w, e_mean: float) -> SensorState:
    """Resolved-fraction and spectral-slope state from local increments."""
    d2 = structure_functions(u, v, w)
    e_res = 0.5 * d2[2]
    if min(d2.values()) <= 0.0:
        # Degenerate resolved field: everything is subgrid.
        return SensorState(alpha=1.0, slope=0.0, e_res=e_res)
    lr = np.log(np.array([1.0, 2.0, 4.0]))
    ld = np.log(np.array([d2[1], d2[2], d2[4]]))
    slope = float(np.polyfit(lr, ld, 1)[0])
    alpha = float(np.clip(e_mean / (e_mean + e_res), 0.0, 1.0))
    return SensorState(alpha=alpha, slope=slope, e_res=e_res)


# ---------------------------------------------------------------------------
# S3-6f: partition cap + w-based resolved-fraction bound (module
# docstring, S3-6f section, has the full formulation and rationale).
# ---------------------------------------------------------------------------


def partition_cap(delta_h: float, zi: float) -> float:
    """S3-6f prescribed partition cap f_cap(delta_h/z_i), scalar.

    FP-exact 1.0 for rho = delta_h/z_i <= F_CAP_KNEE (exp(-0.0) == 1.0:
    the gray zone and LES regime are untouched bitwise), the C^1
    Gaussian ramp exp(-((rho - F_CAP_KNEE)/F_CAP_WIDTH)^2) beyond it --
    monotone decreasing, 0.78 at rho = 2, 0.37 at rho = 3, 1.6e-9 at
    rho = 10 (form rationale: module docstring; Honnert/Shin-Hong-class
    prescribed Delta/z_i bound, credited).  ``zi`` must be positive
    (the z_i diagnosis floors at the first interior layer center).
    """
    rho = float(delta_h) / float(zi)
    x = max(rho - F_CAP_KNEE, 0.0) / F_CAP_WIDTH
    return float(np.exp(-(x * x)))


def bulk_richardson_zi(u, v, theta, z):
    """Per-column bulk-Richardson BL height (S3-6f, YSU convention).

    Rib(k) = (theta_k - theta_1)*(G_ACCEL*z_k/theta_1)
             / max(u_k^2 + v_k^2, RIB_WSPD2_FLOOR)
    against the level-1 thermal, first upward crossing of RIB_CRIT with
    linear interpolation in Rib between the bracketing layer centers --
    the audited ``npref.np_ysu_column`` ``diagnose`` crossing collapsed
    to the single registered critical value (documented simplification:
    module docstring, S3-6f section; dry theta stands in for theta_v).
    ``z`` are layer-center heights, ``(nz,) + trail`` or full shape (the
    :func:`_column_geometry` convention).  Floors at the first INTERIOR
    layer center z[1] (stable-BL fallback); a column with no crossing
    returns the top layer center (documented permissive fallback);
    nz = 1 returns z[0].  Returns the ``(ny, nx)`` FP64 height field.
    """
    u64, v64, th = (np.asarray(a, dtype=np.float64) for a in (u, v, theta))
    shape = th.shape
    z64 = np.broadcast_to(np.asarray(z, dtype=np.float64), shape)
    nz = shape[0]
    if nz == 1:
        return z64[0].copy()
    thermal = th[0]
    spd2 = np.maximum(u64 ** 2 + v64 ** 2, RIB_WSPD2_FLOOR)
    rib = (th - thermal) * (G_ACCEL * z64 / thermal) / spd2
    rib[0] = 0.0                               # own-level Rib, exact
    crossed = rib >= RIB_CRIT
    crossed[0] = False
    has = crossed.any(axis=0)
    kc = np.argmax(crossed, axis=0)            # first True (>= 1)
    kc = np.where(has, kc, nz - 1)
    kx = kc[None]
    rib_up = np.take_along_axis(rib, kx, axis=0)[0]
    rib_dn = np.take_along_axis(rib, kx - 1, axis=0)[0]
    z_up = np.take_along_axis(z64, kx, axis=0)[0]
    z_dn = np.take_along_axis(z64, kx - 1, axis=0)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (RIB_CRIT - rib_dn) / (rib_up - rib_dn)
    zi = np.where(has, z_dn + frac * (z_up - z_dn), z64[-1])
    return np.maximum(zi, z64[1])


@dataclass(frozen=True)
class WSensorState:
    f_w: float          #: upper bound on f (1.0 = sensor abstains)
    alpha_w: float      #: w-sensed subgrid fraction (degenerate 1.0)
    e_res_w: float      #: screened resolved-w energy 0.5*D2_w(2)
    coverage: float     #: screen-passing cell fraction


def w_structure_functions(w, n2=None):
    """N^2-screened horizontal D2 of w, r in {1, 2, 4}: ``(d2w, count)``.

    The sensor's r-machinery applied to the SINGLE component w, with the
    mean accumulated ONLY over cells passing the gravity-wave screen
    (dry n2 <= N2_SCREEN; increments anchored at the passing cell --
    module docstring, S3-6f section).  ``n2 is None`` treats every cell
    as neutral = passing (the module-wide n2-absent convention).  With
    zero passing cells every D2 is 0.0 and the caller must treat the
    sensor as SILENT (``count`` carries that state).
    """
    w64 = np.asarray(w, dtype=np.float64)
    if n2 is None:
        mask = np.ones(w64.shape, dtype=bool)
    else:
        mask = np.asarray(n2, dtype=np.float64) <= N2_SCREEN
    count = int(mask.sum())
    out: dict[int, float] = {}
    for r in (1, 2, 4):
        if count == 0:
            out[r] = 0.0
            continue
        dx_inc = np.roll(w64, -r, axis=2) - w64
        dy_inc = np.roll(w64, -r, axis=1) - w64
        out[r] = float(0.5 * (np.sum(dx_inc[mask] ** 2)
                              + np.sum(dy_inc[mask] ** 2)) / count)
    return out, count


def _w_bound_tail(d2w2: float, count: int, ncell: int,
                  e_mean: float) -> WSensorState:
    """Shared scalar tail of the w-sensor (host + device launcher).

    One copy of the E_r_w = 0.5*D2_w(2), alpha_w clip, f_w = 1 -
    alpha_w arithmetic and the silent-screen abstention (count == 0 ->
    f_w = 1.0, no constraint), so the device path cannot drift from the
    authority (the ``_solve_tail`` idiom).
    """
    if count == 0:
        return WSensorState(f_w=1.0, alpha_w=1.0, e_res_w=0.0,
                            coverage=0.0)
    e_res_w = 0.5 * float(d2w2)
    alpha_w = float(np.clip(e_mean / (e_mean + e_res_w), 0.0, 1.0))
    return WSensorState(f_w=1.0 - alpha_w, alpha_w=alpha_w,
                        e_res_w=e_res_w,
                        coverage=float(count) / float(ncell))


def w_resolved_bound(w, e_mean: float, n2=None) -> WSensorState:
    """S3-6f w-based resolved-fraction bound (module docstring).

    N^2-screened resolved-w structure functions -> alpha_w =
    e/(e + E_r_w) -> the upper bound f_w = 1 - alpha_w that
    :func:`sase_split_step` folds into f_used.  A silent screen (zero
    passing cells) abstains with f_w = 1.0 -- the partition cap governs
    regardless.
    """
    d2w, count = w_structure_functions(w, n2)
    ncell = int(np.asarray(w).size)
    return _w_bound_tail(d2w[2], count, ncell, float(e_mean))


def _ddx(f, dx):
    return (np.roll(f, -1, axis=2) - np.roll(f, 1, axis=2)) / (2.0 * dx)


def _ddy(f, dy):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dy)


def _ddz(f, dz, periodic: bool = False, dz_col=None):
    if dz_col is not None:
        return _ddz_var(f, dz_col, periodic)
    if periodic:
        return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dz)
    if f.shape[0] == 1:
        # Degenerate single-level column (S3-6b nz=1 decay fixture): a
        # one-cell clamped column has no vertical variation; derivative
        # is identically zero.  Pure extension -- nz >= 2 is untouched
        # (the nz = 1 clamped path previously raised IndexError).
        return np.zeros_like(f)
    out = np.empty_like(f)
    out[1:-1] = (f[2:] - f[:-2]) / (2.0 * dz)
    out[0] = (f[1] - f[0]) / dz
    out[-1] = (f[-1] - f[-2]) / dz
    return out


def strain(u, v, w, dx, dy, dz, periodic_z: bool = False, dz_col=None):
    """Resolved strain tensor [xx, yy, zz, xy, xz, yz], FP64 centered.

    ``periodic_z=True`` selects the roll-based (skew-adjoint) vertical
    operator; the energy-ledger step uses it so summation by parts is
    exact on the triply-periodic reference box.  The default keeps the
    clamped one-sided boundary rows for column work.  ``dz_col`` (layer
    thicknesses, see :func:`_ddz_var`) selects the variable-spacing
    clamped vertical stencil; it is incompatible with ``periodic_z``.
    """
    u, v, w = (np.asarray(a, dtype=np.float64) for a in (u, v, w))
    return [
        _ddx(u, dx),
        _ddy(v, dy),
        _ddz(w, dz, periodic=periodic_z, dz_col=dz_col),
        0.5 * (_ddy(u, dy) + _ddx(v, dx)),
        0.5 * (_ddz(u, dz, periodic=periodic_z, dz_col=dz_col)
               + _ddx(w, dx)),
        0.5 * (_ddz(v, dz, periodic=periodic_z, dz_col=dz_col)
               + _ddy(w, dy)),
    ]


_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def germano_lift(u, v, w):
    """Exact resolved lift L_ij at the width-2 test filter."""
    vel = [np.asarray(a, dtype=np.float64) for a in (u, v, w)]
    filt = [box_filter(a, 2) for a in vel]
    return [box_filter(vel[i] * vel[j], 2) - filt[i] * filt[j]
            for i, j in _PAIRS]


def model_stress(e, s_list, c_nu, f, delta_eddy, delta_mom):
    """SASE-L1 modeled SGS stress (spec 4.1 with algebraic tau_mom).

    The dynamic eddy term rides the local filter scale ``delta_eddy``;
    the equilibrium momentum background stays anchored to the grid scale
    ``delta_mom`` (the two coincide at the grid level -- production call
    sites pass ``delta, delta``).  The scale split is what makes the
    two-level Germano system of ``dynamic_solve`` full rank.

    The viscous term acts on the deviatoric strain S_ij - (1/3)S_kk d_ij
    so tau_kk = 2e holds exactly on the discrete (nonzero-divergence)
    velocity field -- the realizability contract the energy ledger uses.
    """
    e = np.maximum(np.asarray(e, dtype=np.float64), E_MIN)
    root_e = np.sqrt(e)
    nu_eddy = c_nu * delta_eddy * root_e
    # S3-6g: the momentum background rides the FIXED C_MOM_BG (the
    # historical C_K/PR_T spelling; bit-identical value) -- a
    # momentum-channel calibration, NOT a Prandtl consumer (decision
    # table, module docstring).  The intentional v0 asymmetry (plan
    # decision: 0.30 here vs bare C_K in e_rhs's K_m, 3x at f=0)
    # stands; do NOT harmonize without a registered decision.
    nu_mom = C_MOM_BG * delta_mom * root_e
    nu = f * nu_eddy + (1.0 - f) * nu_mom
    div3 = (s_list[0] + s_list[1] + s_list[2]) / 3.0
    tau = [-2.0 * nu * s for s in s_list]
    for k in range(3):
        tau[k] = tau[k] + 2.0 * nu * div3 + (2.0 / 3.0) * e
    return tau


def governed_stress(e, s_list, c_nu, f, delta):
    """S3-6e RANS-governed horizontal stress: ``(tau, nu, r)``.

    The split step's stress (module docstring, S3-6e section): the
    dynamic eddy viscosity f*c_nu*delta*sqrt(e) blended with the
    audited 2-D Smagorinsky deformation diffusivity
    K_smag = min((C_S*delta)^2*|D_h|, SMAG_KM_CAP*delta),
    |D_h| = sqrt((S_xx - S_yy)^2 + 4*S_xy^2), at weight (1-f) --
    replacing the v0 fixed-coefficient momentum background that scaled
    with delta*sqrt(e) in the RANS limit.  The tau construction (act on
    the deviatoric strain, restore the (2/3)e trace) is line-identical
    to :func:`model_stress`; ``nu`` is returned as the governed
    horizontal diffusivity FIELD (it also serves the e-transport K_m
    and, over Pr_t(f) (S3-6g), the scalar K_h -- one field, three
    consumers), and
    ``r = nu_smag/max(nu, NU_BLEND_EPS)`` is the smag share in [0, 1]
    (guaranteed: nu >= nu_smag >= 0 termwise) that the production
    split's heat bypass weights.  At f = 1 the smag term is an FP-exact
    zero (0.0*K_smag) and tau reduces bitwise to the v0
    ``model_stress`` at delta/delta.
    """
    e64 = np.maximum(np.asarray(e, dtype=np.float64), E_MIN)
    root_e = np.sqrt(e64)
    def_h = np.sqrt((s_list[0] - s_list[1]) ** 2 + 4.0 * s_list[3] ** 2)
    k_smag = np.minimum((C_S * delta) ** 2 * def_h, SMAG_KM_CAP * delta)
    nu_eddy = c_nu * delta * root_e
    nu_smag = (1.0 - f) * k_smag
    nu = f * nu_eddy + nu_smag
    r = nu_smag / np.maximum(nu, NU_BLEND_EPS)
    div3 = (s_list[0] + s_list[1] + s_list[2]) / 3.0
    tau = [-2.0 * nu * s for s in s_list]
    for k in range(3):
        tau[k] = tau[k] + 2.0 * nu * div3 + (2.0 / 3.0) * e64
    return tau, nu, r


def _deviatoric(s_list):
    """Deviatoric part of a [xx, yy, zz, xy, xz, yz] symmetric tensor."""
    div3 = (s_list[0] + s_list[1] + s_list[2]) / 3.0
    return [s_list[k] - div3 if k < 3 else s_list[k] for k in range(6)]


def _identity_rows(u, v, w, e, dx, dy, dz, delta, width,
                   manufactured_lift=None):
    """Per-component regression rows for one test width.

    Deviatoric residual: L_ij^dev = a*A_ij + b*B_ij with a = f*c_nu and
    b = (1-f)*C_MOM_BG.  The eddy basis A rides the test filter (its
    coarse level scales with width*delta); the momentum basis B is a
    fixed-Delta background anchored to the grid scale (delta at both
    levels).  Both share the refiltered fine-level term; the scale split
    is what makes the two-level Germano system full rank.  Note the
    grid-anchored column is observable only through spatial sqrt(e)
    structure: with uniform e it commutes with the test filter and
    vanishes from the lift.
    """
    vel = [np.asarray(x, dtype=np.float64) for x in (u, v, w)]
    filt = [box_filter(x, width) for x in vel]
    s_fine = _deviatoric(strain(*vel, dx, dy, dz))
    s_coarse = _deviatoric(strain(*filt, dx, dy, dz))
    e64 = np.maximum(np.asarray(e, dtype=np.float64), E_MIN)
    root_e = np.sqrt(e64)
    refilt = [box_filter(-2.0 * delta * root_e * s, width) for s in s_fine]

    def basis(coarse_delta):
        return [-2.0 * coarse_delta * root_e * sc - rf
                for sc, rf in zip(s_coarse, refilt)]

    a_basis = basis(width * delta)          # eddy: rides the test filter
    b_basis = basis(delta)                  # momentum: grid-anchored
    if manufactured_lift is not None:
        lift = manufactured_lift
    else:
        # Width-appropriate resolved lift (germano_lift is the width-2
        # special case of this construction).
        lift = [box_filter(vel[i] * vel[j], width) - filt[i] * filt[j]
                for i, j in _PAIRS]
    # Deviatoric part of the lift (remove the trace: the 2/3 e term).
    trace = (lift[0] + lift[1] + lift[2]) / 3.0
    lift_dev = [lift[k] - trace if k < 3 else lift[k] for k in range(6)]
    rows_a, rows_b, rhs = [], [], []
    for k in range(6):
        rows_a.append(a_basis[k].ravel())
        rows_b.append(b_basis[k].ravel())
        rhs.append(lift_dev[k].ravel())
    return (np.concatenate(rows_a), np.concatenate(rows_b),
            np.concatenate(rhs))


def _solve_tail(gram, proj):
    """Shared 2x2 dynamic-solve tail: cond gate, solve, clip/recovery.

    Extracted (S3-3 carry-forward) so the device launcher
    ``woof.core.sase.launch_dynamic_solve`` and :func:`dynamic_solve`
    share ONE copy of the tail arithmetic: the ``np.linalg.cond`` gate at
    1e12, the 2x2 ``np.linalg.solve``, and the clip/recovery order for
    (c_nu, f) from the raw weights a = f*c_nu, b = (1-f)*C_MOM_BG
    (S3-6g: the momentum-basis weight is the fixed momentum-channel
    constant, bit-identical to the former C_K/PR_T spelling -- decision
    table, module docstring).  Pure refactor: operations are
    line-identical to both former copies, so every golden (host and
    device) is unchanged bitwise.
    """
    if np.linalg.cond(gram) > 1.0e12:
        # Degenerate resolved field (alpha -> 1): fall back to the
        # equilibrium momentum weight, no eddy contribution.
        return 0.0, 0.0
    eddy_w, mom_w = np.linalg.solve(gram, proj)
    f = float(np.clip(1.0 - mom_w / C_MOM_BG, F_MIN, F_MAX))
    c_nu = float(np.clip(eddy_w / max(f, 1.0e-12), 0.0, CNU_MAX))
    return c_nu, f


def dynamic_solve(u, v, w, e, dx, dy, dz, delta,
                  manufactured_lifts=None):
    """Closed-form 2x2 least squares for (c_nu, f) across both test levels.

    Including both test widths (2 and 4) is the discrete invariance
    constraint: one parameter pair must satisfy the Germano identity at
    two cutoffs simultaneously.  The solve is linear in the weights
    a = f*c_nu and b = (1-f)*C_MOM_BG; (c_nu, f) are recovered from
    (a, b) and clipped to the realizable range.
    """
    cols_a, cols_b, rhs = [], [], []
    for width in (2, 4):
        lift = (manufactured_lifts[width]
                if manufactured_lifts is not None else None)
        a, b, r = _identity_rows(u, v, w, e, dx, dy, dz, delta, width,
                                 manufactured_lift=lift)
        cols_a.append(a)
        cols_b.append(b)
        rhs.append(r)
    a = np.concatenate(cols_a)
    b = np.concatenate(cols_b)
    r = np.concatenate(rhs)
    gram = np.array([[a @ a, a @ b], [a @ b, b @ b]])
    proj = np.array([a @ r, b @ r])
    return _solve_tail(gram, proj)


def prandtl_blend(f: float) -> float:
    """S3-6g regime-consistent turbulent Prandtl number Pr_t(f).

        Pr_t(f) = f*PR_LES + (1 - f)*PR_RANS

    -- the registered blend Pr_t = PR_RANS + f*(PR_LES - PR_RANS)
    (controller ledger 2026-07-20, S3-6g registration) rewritten in the
    two-product form every other regime blend of this module uses
    (:func:`dissipation_length`'s l_d, :func:`governed_stress`'s nu),
    so BOTH endpoints are FP-exact: f = 1 gives 1.0*PR_LES +
    0.0*PR_RANS = PR_LES bitwise (the LES regime and every f = 1
    reduction pin are untouched), f = 0 gives PR_RANS bitwise (the
    RANS fixed points are exact).  Affine, hence monotone, between.
    ``f`` is the step's USED partition weight (post-S3-6f caps): the
    same regime judgment that blends l_d and the governed stress.
    Reads the module globals, so instrumented runs can pin the regime
    -- monkeypatching PR_RANS = PR_LES reproduces the pre-S3-6g
    fixed-Pr formulation (bitwise at f = 0, the RANS-column fixtures'
    branch), which is how the inversion-persistence fixture produces
    its RED leg.
    """
    return f * PR_LES + (1.0 - f) * PR_RANS


def dissipation_length(e, delta, n2=None, lb=None, f=1.0):
    """Stability-limited dissipation length with the regime blend
    (GEOMETRIC since S3-9; module docstring, S3-9 section).

        l = min(delta**f * lb**(1-f), LS_COEF*sqrt(e)/N)    [N^2 > 0]

    ``lb`` is the RANS-limb dissipation composition (the live path
    passes min(l_B, l_eps_BL89) from :func:`bl89_rans_lengths`) and
    ``f`` the dynamic partition weight, blending the two regimes the
    length must serve IN LOG SPACE: at f = 1 (resolved, LES-like
    fields) the blend is exactly ``delta`` -- delta**1.0 * lb**0.0 =
    delta*1.0 is FP-exact -- so the v0 form min(delta, l_s) is
    recovered bitwise; at f = 0 (degenerate resolved field, the
    coarse-mesh RANS regime, where ``dynamic_solve`` falls back to
    (0, 0)) the blend is exactly ``lb`` (delta**0.0 * lb**1.0 =
    1.0*lb, FP-exact) so l = min(l_B, l_s) = l_v, the vertical
    mixing-length scale.  Both endpoints are bitwise IDENTICAL to the
    retired linear form f*delta + (1-f)*lb; between them the
    geometric form stays within (delta/lb)**f of lb where the linear
    form floored l_d at f*delta -- the F-Y1 defect (S3-9 section).
    EXPONENTIATION DOMAIN: delta and lb must be strictly positive --
    every live caller guarantees it (delta is a filter width; lb from
    bl89_rans_lengths is min of the positive Blackadar length at
    layer centers z >= thick_0/2 > 0 and the E_MIN-floored BL89
    lengths), documented rather than asserted per module convention.
    The stability limit enters ONCE through the outer min: l_v's own
    stability limit is the same l_s, so blending the NEUTRAL-limb lb
    avoids counting it twice.  ``lb=None`` (default) preserves the v0
    behavior unchanged bitwise; unstable/neutral cells (n2 <= 0 or
    ``n2 is None``) never engage the l_s branch.
    """
    e64 = np.maximum(np.asarray(e, dtype=np.float64), E_MIN)
    if lb is None:
        l = np.full_like(e64, float(delta))
    else:
        # S3-9 GEOMETRIC blend (was the linear f*delta + (1-f)*lb):
        # log-space interpolation keeps l_d within (delta/lb)**f of
        # the RANS bound.  Endpoints FP-exact in IEEE-754: x**1.0 == x,
        # x**0.0 == 1.0, 1.0*x == x, so f = 0 gives lb bitwise and
        # f = 1 gives delta bitwise -- identical to the linear form's
        # endpoint arithmetic (module docstring, S3-9 section).
        blend = (float(delta) ** f
                 * np.asarray(lb, dtype=np.float64) ** (1.0 - f))
        l = blend * np.ones_like(e64)
    if n2 is not None:
        n2 = np.asarray(n2, dtype=np.float64)
        stable = n2 > 0.0
        ls = LS_COEF * np.sqrt(e64) / np.sqrt(np.where(stable, n2, 1.0))
        l = np.where(stable, np.minimum(l, ls), l)
    return l


def e_rhs(e, u, v, w, theta, dx, dy, dz, delta, c_nu, f, n2=None,
          periodic_z: bool = False, dz_col=None):
    """SASE-L1 subgrid-energy budget terms (all FP64, same shape as e).

    ``periodic_z`` threads the roll-based vertical operator through the
    strain (hence production), buoyancy, and transport terms so the
    ledger's production pairing matches the momentum update exactly.
    ``dz_col`` threads the variable-spacing clamped vertical stencil
    (:func:`_ddz_var`) through the same three terms instead; it is
    incompatible with ``periodic_z``.
    """
    e64 = np.maximum(np.asarray(e, dtype=np.float64), E_MIN)
    s = strain(u, v, w, dx, dy, dz, periodic_z=periodic_z, dz_col=dz_col)
    tau = model_stress(e64, s, c_nu, f, delta, delta)
    production = -(tau[0] * s[0] + tau[1] * s[1] + tau[2] * s[2]
                   + 2.0 * (tau[3] * s[3] + tau[4] * s[4] + tau[5] * s[5]))
    l = dissipation_length(e64, delta, n2)
    dissipation = C_E * e64**1.5 / l
    k_m = (f * c_nu + (1.0 - f) * C_K) * delta * np.sqrt(e64)
    # S3-6g decision table: v0 FROZEN historical path (no device
    # consumer since S3-6c) -- the fixed LES Prandtl number stands
    # here so the v0 fixtures stay bitwise; the live split path
    # blends (sase_split_step).
    k_h = k_m / PR_T
    theta64 = np.asarray(theta, dtype=np.float64)
    buoyancy = -(G_ACCEL / theta64) * k_h * _ddz(theta64, dz,
                                                 periodic=periodic_z,
                                                 dz_col=dz_col)
    transport = (_ddx(2.0 * k_m * _ddx(e64, dx), dx)
                 + _ddy(2.0 * k_m * _ddy(e64, dy), dy)
                 + _ddz(2.0 * k_m * _ddz(e64, dz, periodic=periodic_z,
                                         dz_col=dz_col), dz,
                        periodic=periodic_z, dz_col=dz_col))
    return {"production": production, "buoyancy": buoyancy,
            "dissipation": dissipation, "transport": transport}


def _div_tau_row(tau_row_x, tau_row_y, tau_row_z, dx, dy, dz,
                 periodic_z: bool = False, dz_col=None):
    return (_ddx(tau_row_x, dx) + _ddy(tau_row_y, dy)
            + _ddz(tau_row_z, dz, periodic=periodic_z, dz_col=dz_col))


def sase_ref_step(u, v, w, theta, e, dx, dy, dz, delta, dt, n2=None,
                  dz_col=None):
    """One explicit SASE-L1 step on the periodic reference box.

    SUPERSEDED (S3-6b; device retirement completed S3-6c): this v0 step
    diffuses vertically with the horizontal filter scale, explicitly --
    linearly unstable at outer-nest parameters (module docstring).  It stays
    bit-frozen as the v0 HISTORICAL authority (its CPU fixtures pin the
    retained explicit machinery: strain, stress, solve, e_rhs); no
    device path steps it anymore -- ``launch_sase_step`` now runs the
    split scheme and gates against :func:`sase_split_step`.

    The step runs triply periodic (``periodic_z=True`` everywhere): the
    conservation theorem (spec 4.2) is stated on the closed periodic
    domain, and only the roll-based vertical operator is skew-adjoint
    under the discrete sum, which is what makes the production pairing
    close by parts against the momentum update to roundoff.

    ``dz_col`` (layer thicknesses, see :func:`_ddz_var`) selects the
    clamped-z model-column mode instead: every vertical operator becomes
    the variable-spacing clamped stencil (a periodic vertical is
    incompatible with ``dz_col``), matching real model columns for the
    stage-3 CUDA parity gate.  The clamped operators are not
    skew-adjoint, so the conservation theorem does not apply and the
    returned ledger is diagnostic only.  The dynamic solve keeps its
    uniform-``dz`` clamped strain either way (coefficients only, as
    above); ``dz_col=None`` preserves the periodic ledger path
    bit-identically.
    """
    u, v, w, e = (np.asarray(a, dtype=np.float64) for a in (u, v, w, e))
    pz = dz_col is None
    # Coefficients only: the solve intentionally stays clamped-z; do not
    # thread periodic_z into it.
    c_nu, f = dynamic_solve(u, v, w, e, dx, dy, dz, delta)
    s = strain(u, v, w, dx, dy, dz, periodic_z=pz, dz_col=dz_col)
    tau = model_stress(np.maximum(e, E_MIN), s, c_nu, f, delta, delta)
    # tau rows: (xx,xy,xz), (xy,yy,yz), (xz,yz,zz)
    du = -_div_tau_row(tau[0], tau[3], tau[4], dx, dy, dz, periodic_z=pz,
                       dz_col=dz_col)
    dv = -_div_tau_row(tau[3], tau[1], tau[5], dx, dy, dz, periodic_z=pz,
                       dz_col=dz_col)
    dw = -_div_tau_row(tau[4], tau[5], tau[2], dx, dy, dz, periodic_z=pz,
                       dz_col=dz_col)
    rhs = e_rhs(e, u, v, w, theta, dx, dy, dz, delta, c_nu, f, n2,
                periodic_z=pz, dz_col=dz_col)
    production = rhs["production"]
    dissipation = rhs["dissipation"]
    e_new = np.maximum(
        e + dt * (production + rhs["buoyancy"] - dissipation
                  + rhs["transport"]),
        E_MIN)
    clip_gain = e_new - (e + dt * (production + rhs["buoyancy"]
                                   - dissipation + rhs["transport"]))
    u_new, v_new, w_new = u + dt * du, v + dt * dv, w + dt * dw
    # Clipped energy exits as heat.  Note heat is NOT sign-definite: it
    # goes locally negative wherever clip_gain exceeds dt*dissipation
    # (bookkeeping-consistent -- the floor deposit is drawn back out).
    heat = dt * dissipation - clip_gain
    # Ledger: discrete-compatible pairing.  KE change from the tendency
    # inner product (first-order in dt, matching production's pairing),
    # not the quadratic finite difference.
    d_ke = float(np.sum(u * dt * du + v * dt * dv + w * dt * dw))
    d_e = float(np.sum(e_new - e)) - float(np.sum(dt * rhs["buoyancy"])) \
        - float(np.sum(dt * rhs["transport"]))
    d_heat = float(np.sum(heat))
    residual = d_ke + d_e + d_heat
    fields = {"u": u_new, "v": v_new, "w": w_new, "e": e_new, "heat": heat}
    ledger = {"dKE": d_ke, "dE": d_e, "dHeat": d_heat,
              "residual": residual, "c_nu": c_nu, "f": f}
    return fields, ledger


def column_free_convection(b0, zi, nz, dt, steps):
    """Quasi-steady CBL e column under a prescribed buoyancy-flux profile.

    S3-6b re-derivation: the column runs the amended VERTICAL channel in
    its RANS limit (a coarse-mesh column has no resolved field, so the
    blend weight is f = 0 and delta never enters): mixing/dissipation
    length l_B(z) = k*z/(1 + k*z/lambda) (the unstable column never
    engages the N^2 > 0 stability limit), dissipation C_E*e^{3/2}/l_B,
    and IMPLICIT vertical e-transport with face coefficient 2*K_v,
    K_v = C_KV*l_B*sqrt(e) -- replacing the v0 fixed l = 0.1*zi and the
    explicit np.gradient transport.  Closed form for the fixture regime
    (local balance, transport redistributes but nearly preserves the
    mixed-layer mean): flux = C_E*e^{3/2}/l_B gives
        e_lb(z) = (l_B(z)*flux(z)/C_E)^(2/3),
    returned as ``e_local_balance`` (mixed-layer mean) so the test can
    pin measured/predicted tightly instead of only an absolute band.
    """
    dz = zi / nz
    z = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    flux = np.maximum(b0 * (1.0 - z / zi), 0.0)
    lb = _blackadar_length(z)
    e = np.full(nz, E_MIN)
    for _ in range(steps):
        e64 = np.maximum(e, E_MIN)
        kv = C_KV * lb * np.sqrt(e64)
        diss = C_E * e64**1.5 / lb
        e = np.maximum(e + dt * (flux - diss), E_MIN)
        e = implicit_vertical_diffusion(e, 2.0 * _face_average(kv), dt,
                                        dz=dz)
    w_star = (b0 * zi)**(1.0 / 3.0)
    ml = z < 0.8 * zi
    e_ml = float(e[ml].mean())
    e_lb = float(np.mean((lb[ml] * flux[ml] / C_E)**(2.0 / 3.0)))
    return {"e_ml": e_ml, "w_star": w_star, "e_local_balance": e_lb}


def _z_centers(dz_col, shape):
    """Layer-center heights implied by layer thicknesses ``dz_col``.

    Convention (stage-3 model-column mode): ``dz_col`` holds layer
    thicknesses, shape ``(nz,)`` (a shared column) or ``(nz, ny, nx)``
    (per-column); interface positions are the cumulative thickness sums
    up from z = 0, and level ``k``'s value sits at its layer center,
    ``z_k = sum(dz_col[:k]) + 0.5*dz_col[k]``.
    """
    t = np.asarray(dz_col, dtype=np.float64)
    if t.ndim == 1:
        if t.shape[0] != shape[0]:
            raise ValueError(
                f"dz_col has {t.shape[0]} levels, field has {shape[0]}")
        t = t[:, None, None]
    elif t.shape != tuple(shape):
        raise ValueError(
            f"dz_col shape {t.shape} must be (nz,) or match field "
            f"shape {tuple(shape)}")
    return np.cumsum(t, axis=0) - 0.5 * t


def brunt_vaisala_n2(theta, dz, dz_col=None):
    """Moist-static N^2 = (g/theta) * d(theta)/dz, clamped-z stencil.

    Stage-3 Task 6 authority extension (new function; no existing default
    changes): the stability-length input the model driver computes from
    its own theta profile, on the SAME clamped (variable-``dz_col``)
    vertical stencil every SASE operator uses -- the discrete form is
    therefore pinned here, not re-derived at the call site.
    """
    theta64 = np.asarray(theta, dtype=np.float64)
    return (G_ACCEL / theta64) * _ddz(theta64, dz, dz_col=dz_col)


def moist_n2(theta, qv, qc, pressure, dz_col):
    """SASE-M1 effective stability N^2_eff (module docstring, SASE-M1
    section has the full DK82 derivation with the algebra): the
    Durran & Klemp (1982, JAS 39, 2152-2158, Eq. 36) saturated moist
    N^2_m in saturated cells, the dry :func:`brunt_vaisala_n2` field
    BITWISE in unsaturated cells:

        T     = theta*(pressure/P0_REF)**(RD_AIR/CP_AIR)     [Exner]
        e_s   = 1000*SVP1*exp(SVP2*(T - SVPT0)/(T - SVP3))   [Tetens,
                liquid -- the model's own saturation constants]
        q_s   = EP2_RV*e_s/(pressure - e_s)
        a     = 1 + XLV*q_s/(RD_AIR*T)
        b     = 1 + EP2_RV*XLV^2*q_s/(CP_AIR*RD_AIR*T^2)
        N^2_m = G_ACCEL*((a/b)*(ddz(theta)/theta
                                + (XLV/(CP_AIR*T))*ddz(q_s))
                         - ddz(qv + qc))
        sat   = (qc > 0) | (qv >= q_s)        [MOIST_STABILITY_SWITCH
                                               = binary-qc-or-rh100-
                                               liquid]
        out   = where(sat, N^2_m, brunt_vaisala_n2(theta, dz_col))

    Transcription validated executably by the moist-adiabat-
    neutrality fixture (|N^2_m| <= 1e-6 on an independently
    constructed q_w-const saturated adiabat while dry N^2 > 1e-4),
    the moist-lapse witness (-dT/dz == (g/cp)*(a/b) within 1% -- the
    a/b factor IS the textbook saturated-adiabatic lapse factor,
    derivation at the module docstring), and the condensate-loading
    witness (linear qc ramp shifts N^2_m by exactly -g*slope).

    Every vertical derivative rides the SAME clamped variable-
    ``dz_col`` stencil as every SASE vertical operator
    (:func:`_ddz_var`), and the where's FALSE branch is the LITERAL
    dry field, so unsaturated cells carry the dry bits VERBATIM (the
    M1 unsaturated identity contract, pinned by the control-column
    fixture).  ``pressure`` is the FULL pressure [Pa] at layer
    centers; inputs broadcast against ``theta``'s ``(nz, ny, nx)``
    layout; ``dz_col`` are the layer thicknesses
    (:func:`_column_geometry` conventions).  DOCUMENTED DOMAIN
    (module convention -- documented, not asserted): physical states
    have T > SVP3 and pressure > e_s; a state violating them would
    surface loudly as non-finite output rather than silently flip
    the switch.
    """
    th = np.asarray(theta, dtype=np.float64)
    qv64 = np.asarray(qv, dtype=np.float64)
    qc64 = np.asarray(qc, dtype=np.float64)
    p64 = np.asarray(pressure, dtype=np.float64)
    n2_dry = brunt_vaisala_n2(th, None, dz_col=dz_col)
    t = th * (p64 / P0_REF) ** (RD_AIR / CP_AIR)
    es = 1000.0 * SVP1 * np.exp(SVP2 * (t - SVPT0) / (t - SVP3))
    qs = EP2_RV * es / (p64 - es)
    a_fac = 1.0 + XLV * qs / (RD_AIR * t)
    b_fac = 1.0 + EP2_RV * XLV * XLV * qs / (CP_AIR * RD_AIR * t * t)
    bracket = (_ddz(th, None, dz_col=dz_col) / th
               + (XLV / (CP_AIR * t)) * _ddz(qs, None, dz_col=dz_col))
    n2m = G_ACCEL * ((a_fac / b_fac) * bracket
                     - _ddz(qv64 + qc64, None, dz_col=dz_col))
    sat = (qc64 > 0.0) | (qv64 >= qs)
    return np.where(sat, n2m, n2_dry)


def scalar_mix(s, e, kh_coef, dx, dy, dz, dz_col=None):
    """K_h down-gradient mixing tendency for one scalar (Task 6).

    ``div(K_h grad s)`` with ``K_h = kh_coef * sqrt(max(e, E_MIN))``,
    where the production call site supplies ``kh_coef =
    (f*c_nu + (1-f)*C_K) * delta / Pr_t(f)`` -- K_m's blend over the
    turbulent Prandtl number (S3-6g: the blended
    :func:`prandtl_blend` value; the frozen v0 fixtures pin the fixed
    PR_T = PR_LES special case), the same K_h the e-budget buoyancy
    term uses.  Discretization mirrors the e-transport grouping exactly:
    cell-centered fluxes first (centered periodic horizontal, clamped
    variable-``dz_col`` vertical), then the centered divergence of the
    fluxes.  New authority function; nothing existing changes.
    """
    e64 = np.maximum(np.asarray(e, dtype=np.float64), E_MIN)
    kh = kh_coef * np.sqrt(e64)
    s64 = np.asarray(s, dtype=np.float64)
    fx = kh * _ddx(s64, dx)
    fy = kh * _ddy(s64, dy)
    fz = kh * _ddz(s64, dz, dz_col=dz_col)
    return (_ddx(fx, dx) + _ddy(fy, dy) + _ddz(fz, dz, dz_col=dz_col))


def scalar_hmix(s, kh, dx, dy):
    """Horizontal K_h down-gradient mixing with a diffusivity FIELD.

    S3-6e governed scalar channel (horizontal-explicit half): the
    driver supplies ``kh = K_m_governed/Pr_t(f)`` from the split step's
    exported diffusivity field (S3-6g: the blended Prandtl number at
    the step's used f), and this helper pins the discrete form
    -- cell-centered periodic fluxes then the centered flux divergence,
    exactly :func:`scalar_mix`'s horizontal grouping of which the
    ``kh_coef*sqrt(e)`` form is the retired special case on the split
    path (v0 ``scalar_mix`` stays frozen for its own fixtures).
    """
    kh64 = np.asarray(kh, dtype=np.float64)
    s64 = np.asarray(s, dtype=np.float64)
    fx = kh64 * _ddx(s64, dx)
    fy = kh64 * _ddy(s64, dy)
    return _ddx(fx, dx) + _ddy(fy, dy)


def _ddz_var(f, dz_col, periodic: bool = False):
    """Clamped-z vertical derivative on variable layer thicknesses.

    Interior rows use the three-point Lagrange stencil on the nonuniform
    center-to-center spacings h_m = z[k]-z[k-1], h_p = z[k+1]-z[k]:

        f'(z_k) = -(h_p/(h_m*(h_p+h_m))) * f[k-1]
                  + ((h_p-h_m)/(h_p*h_m)) * f[k]
                  + (h_m/(h_p*(h_p+h_m))) * f[k+1]

    which is exact for quadratics in z (the two-sided form
    (f[k+1]-f[k-1])/(z[k+1]-z[k-1]) is only linear-exact on a stretched
    grid) and reduces to the centered (f[k+1]-f[k-1])/(2h) when
    h_p == h_m == h.  Edge rows are one-sided two-point over the edge
    center spacing (linear-exact), matching the uniform clamped path.

    ``periodic`` is accepted only to reject it: the roll-based periodic
    vertical is the conservation theorem's domain (the uniform periodic
    box) and must not be silently combined with variable spacing.
    """
    if periodic:
        raise ValueError(
            "dz_col is incompatible with a periodic vertical: the "
            "conservation ledger's periodic path is defined on the "
            "uniform box only")
    f = np.asarray(f, dtype=np.float64)
    if f.shape[0] == 1:
        # Degenerate single-level column: no vertical variation (see the
        # matching nz = 1 branch in ``_ddz``).
        return np.zeros_like(f)
    z = _z_centers(dz_col, f.shape)
    h_m = z[1:-1] - z[:-2]
    h_p = z[2:] - z[1:-1]
    out = np.empty_like(f)
    out[1:-1] = (-(h_p / (h_m * (h_p + h_m))) * f[:-2]
                 + ((h_p - h_m) / (h_p * h_m)) * f[1:-1]
                 + (h_m / (h_p * (h_p + h_m))) * f[2:])
    out[0] = (f[1] - f[0]) / (z[1] - z[0])
    out[-1] = (f[-1] - f[-2]) / (z[-1] - z[-2])
    return out


# ---------------------------------------------------------------------------
# S3-6b amendment: anisotropic vertical channel + implicit vertical
# diffusion.  Formulation, derivations, and the restated ledger theorem
# are in the module docstring; the v0 explicit machinery above is kept
# bit-frozen as the device parity target until S3-6c.
# ---------------------------------------------------------------------------


def _blackadar_length(z, z0=0.0):
    """Neutral Blackadar (1962) length l_B = k*(z+z0)/(1+k*(z+z0)/lambda).

    Rises as k*(z+z0) off the surface, saturates at BLACKADAR_LAMBDA
    aloft.  ``z`` is height above the surface at layer centers; ``z0``
    (surface roughness) is an argument rather than a registered
    constant because it is model surface data, not closure physics --
    layer-center evaluation (z >= dz1/2 >> z0) makes it subdominant and
    the default 0.0 is exact for the reference fixtures.
    """
    kz = KARMAN * (np.asarray(z, dtype=np.float64) + z0)
    return kz / (1.0 + kz / BLACKADAR_LAMBDA)


def vertical_mixing_length(z, e, n2=None, z0=0.0):
    """S3-6b vertical mixing length l_v = min(l_B, LS_COEF*sqrt(e)/N).

    The N^2 > 0 branch applies the Deardorff stability limit (the same
    l_s the dissipation length uses); neutral/unstable cells keep the
    Blackadar length.  Broadcasts ``z`` (per-level or full-shape)
    against ``e``; returns the full-shape FP64 length.

    S3-6h role: this frozen v0 form is now the LES LIMB (and one term
    of the RANS composition) of the split step's blended vertical
    length -- the live channel is l_v = f*THIS + (1-f)*
    min(THIS, l_mix_BL89) (:func:`bl89_rans_lengths`; module
    docstring, S3-6h section).  The function itself is unchanged
    bitwise.
    """
    e64 = np.maximum(np.asarray(e, dtype=np.float64), E_MIN)
    l = _blackadar_length(z, z0) * np.ones_like(e64)
    if n2 is not None:
        n2 = np.asarray(n2, dtype=np.float64)
        stable = n2 > 0.0
        ls = LS_COEF * np.sqrt(e64) / np.sqrt(np.where(stable, n2, 1.0))
        l = np.where(stable, np.minimum(l, ls), l)
    return l


def vertical_diffusivity(z, e, n2=None, z0=0.0):
    """S3-6b vertical momentum diffusivity K_v = C_KV*l_v*sqrt(e).

    The dynamic weights (c_nu, f) deliberately do NOT enter (module
    docstring rationale); the scalar/e channels ride the same K_v as
    K_v/Pr_t(f) and 2*K_v respectively at the call sites.  In the
    neutral surface-layer equilibrium e = C_E^{-2/3}*u*^2 this reduces
    exactly to K_v = u* * l_v -> KARMAN*u**z as z -> 0 (the log-layer
    identity pinned by the C_KV derivation and the log-layer fixtures).

    Registered approximations (controller ledger, 2026-07-20):
    (1) [RESOLVED by S3-6g, this module's final amendment section] the
    fixed PR_T = 1/3 put the RANS-regime scalar channel at
    K_v/PR_T ~ 3*kappa*u**z vs observed ~1.2*kappa*u**z; the blended
    Pr_t(f) now gives K_v/PR_RANS ~ 1.18*kappa*u**z at f = 0 --
    inside the observed band -- while the LES limit keeps PR_LES.
    (2) The one-constant closure's equilibrium e ~ 1.05*u*^2 sits
    ~3-5x below observed 3.3-5.5*u*^2 (sqrt(e) ~2x low), biasing l_s,
    horizontal K, buoyancy flux, and TKE products -- still open.

    S3-6i role: this frozen S3-6b form (the coupled C_KV coefficient
    on min(l_B, l_s)) is now the LES LIMB of the split step's vertical
    channel ONLY; the live RANS limb rides the decoupled stable-limit
    coefficient (:func:`stable_limit_coefficient` -- K_v -> C_KS*e/N
    where l_s binds; module docstring, S3-6i section).  The function
    itself is unchanged bitwise (v0 fixture surface).
    """
    e64 = np.maximum(np.asarray(e, dtype=np.float64), E_MIN)
    return C_KV * vertical_mixing_length(z, e, n2, z0) * np.sqrt(e64)


# ---------------------------------------------------------------------------
# S3-6h: Bougeault-Lacarrere (1989) parcel-energetics displacement
# lengths for the RANS-limb vertical channel (module docstring, S3-6h
# section, has the formulation, citations, and the jet-decoupling
# evidence chain).
# ---------------------------------------------------------------------------


def _bl89_first_crossing(c1, c2, rem, h):
    """Smallest s in [0, h] with c2*s^2 + c1*s == rem, else NaN.

    One segment of the BL89 displacement integral (all FP64, vectorized
    over columns).  DERIVATION: within a segment of extent h whose
    piecewise-linear theta_v runs from theta_a (at the parcel-side end)
    to theta_b, the integrand against a parcel of potential temperature
    theta_p is beta*(theta(s) - theta_p) = (c1 + 2*c2*s) with

        c1 = beta*(theta_a - theta_p),
        c2 = beta*(theta_b - theta_a)/(2*h),

    so the spent energy over a partial traversal s is EXACTLY

        I(s) = integral_0^s (c1 + 2*c2*sigma) d sigma = c1*s + c2*s^2

    -- quadratic in s, which is why the discrete integral is exact for
    every piecewise-linear theta_v profile (the quadratic-exactness
    class pinned by the analytic fixtures).  The parcel stops at the
    FIRST s where I(s) equals the remaining budget rem (>= 0): the
    smallest root of c2*s^2 + c1*s - rem = 0 inside [0, h].  Root
    finding is the numerically stable q-form (q = -(c1 +
    sign(c1)*sqrt(disc))/2, roots q/c2 and -rem/q), which degrades
    gracefully to the linear root rem/c1 as c2 -> 0; the exact c2 == 0
    branch is handled explicitly.  No root in [0, h] (including a
    negative discriminant) means the parcel traverses the whole
    segment: the caller subtracts I(h) from rem and continues.
    """
    c1 = np.asarray(c1, dtype=np.float64)
    c2, rem64, h64 = (np.broadcast_to(np.asarray(a, dtype=np.float64),
                                      c1.shape)
                      for a in (c2, rem, h))
    with np.errstate(divide="ignore", invalid="ignore"):
        disc = c1 * c1 + 4.0 * c2 * rem64
        sq = np.sqrt(np.maximum(disc, 0.0))
        q = -0.5 * (c1 + np.where(c1 >= 0.0, sq, -sq))
        root_a = np.where(c2 != 0.0, q / np.where(c2 != 0.0, c2, 1.0),
                          np.nan)
        root_b = np.where(q != 0.0, -rem64 / np.where(q != 0.0, q, 1.0),
                          # q == 0 <=> c1 == 0 and disc == 0 <=> the
                          # double root sits at s = 0 (rem == 0 with a
                          # locally flat or stabilizing integrand).
                          0.0)
        lin = np.where(c1 > 0.0, rem64 / np.where(c1 > 0.0, c1, 1.0),
                       np.nan)
    quad = c2 != 0.0
    cands = (np.where(quad, root_a, np.nan),
             np.where(quad, root_b, lin))
    tol = 1.0e-12 * h64
    best = np.full(c1.shape, np.nan)
    for cand in cands:
        ok = (np.isfinite(cand) & (disc >= 0.0)
              & (cand >= -tol) & (cand <= h64 + tol))
        cand = np.clip(cand, 0.0, h64)
        best = np.where(ok & (~np.isfinite(best) | (cand < best)),
                        cand, best)
    return best


def bl89_displacement_lengths(theta, e, z, thick):
    """S3-6h BL89 parcel displacement lengths ``(l_up, l_down)``.

    Per column and level k (Bougeault-Lacarrere 1989, MWR 117,
    1872-1890): ``l_up(k)`` is the FIRST upward displacement l at which

        integral_{z_k}^{z_k + l} beta * (theta_v(z') - theta_v(z_k)) dz'
            == e(k),        beta = G_ACCEL/theta_v(z_k)

    (the parcel's kinetic-energy budget exhausted against the
    integrated buoyancy resistance; the accumulate-until-exceed
    Meso-NH convention), bounded by the column top interface
    (l_up <= htop - z_k); ``l_down(k)`` is the mirror image downward,
    integrand beta*(theta_v(z_k) - theta_v(z')), bounded by the surface
    (l_down <= z_k).  Unstable stretches contribute NEGATIVELY (the
    parcel re-accelerates), which is the non-locality this amendment
    exists for: a parcel below an inversion is stopped by the
    inversion's INTEGRATED buoyancy regardless of the local N^2 --
    information the local l_s = LS_COEF*sqrt(e)/N cannot carry.

    DISCRETE INTEGRAL (exact for piecewise-linear theta_v): theta_v is
    taken linear between adjacent layer centers, held constant from the
    lowest center to the surface (z' in [0, z_0]) and from the highest
    center to the top interface -- the sub-half-layer profile is
    unresolved, so constant extension is the accurate discrete choice.
    Each full segment then contributes its exact quadrature
    (c1 + c2*h)*h and the fractional last segment solves the quadratic
    I(s) = rem exactly (:func:`_bl89_first_crossing` has the
    derivation), so the only approximation in the whole construction
    is the piecewise-linear profile representation itself.
    theta_v = dry theta (L1 dry core; registered v0 simplification --
    moist theta_v is SASE-M scope).  ``e`` is floored at E_MIN (module
    idiom), which keeps both lengths strictly positive.  ``z``/``thick``
    follow the :func:`_column_geometry` conventions ((nz,) + trail or
    full shape; scalar thick for uniform columns).  Returns two FP64
    full-shape arrays.
    """
    th = np.asarray(theta, dtype=np.float64)
    shape = th.shape
    nz = shape[0]
    e64 = np.broadcast_to(
        np.maximum(np.asarray(e, dtype=np.float64), E_MIN), shape)
    z64 = np.broadcast_to(np.asarray(z, dtype=np.float64), shape)
    t64 = np.broadcast_to(np.asarray(thick, dtype=np.float64)
                          * np.ones((1,) * th.ndim), shape)
    htop = z64[-1] + 0.5 * t64[-1]
    trail = shape[1:]
    l_up = np.empty(shape)
    l_down = np.empty(shape)
    for k in range(nz):
        beta = G_ACCEL / th[k]
        for down in (False, True):
            rem = e64[k].copy()
            acc = np.zeros(trail)
            active = np.ones(trail, dtype=bool)
            out_k = np.zeros(trail)
            prev_z, prev_t = z64[k], th[k]
            nodes = range(k - 1, -2, -1) if down else range(k + 1, nz + 1)
            for j in nodes:
                if down:
                    node_z = z64[j] if j >= 0 else np.zeros(trail)
                    node_t = th[j] if j >= 0 else th[0]
                    h = prev_z - node_z
                    c1 = beta * (th[k] - prev_t)
                    c2d = beta * (prev_t - node_t)
                else:
                    node_z = z64[j] if j < nz else htop
                    node_t = th[j] if j < nz else th[nz - 1]
                    h = node_z - prev_z
                    c1 = beta * (prev_t - th[k])
                    c2d = beta * (node_t - prev_t)
                pos = h > 0.0
                c2 = np.where(pos, c2d, 0.0) / np.where(pos, 2.0 * h, 1.0)
                s = _bl89_first_crossing(c1, c2, rem, h)
                hit = active & np.isfinite(s)
                out_k = np.where(hit, acc + s, out_k)
                seg = np.where(pos, (c1 + c2 * h) * h, 0.0)
                rem = np.where(active & ~hit,
                               np.maximum(rem - seg, 0.0), rem)
                acc = acc + np.where(active, h, 0.0)
                active = active & ~hit
                prev_z, prev_t = node_z, node_t
                if not active.any():
                    break
            bound = z64[k] if down else htop - z64[k]
            out_k = np.where(active, bound, out_k)
            if down:
                l_down[k] = out_k
            else:
                l_up[k] = out_k
    return l_up, l_down


def bl89_combine(l_up, l_down):
    """S3-6h combination forms ``(l_mix, l_eps)`` from the displacement
    pair (constants block at BL89_MIX_EXP has the choice rationale and
    citations):

        l_mix = (0.5*(l_up^-p + l_down^-p))^(-1/p),   p = BL89_MIX_EXP
        l_eps = min(l_up, l_down)

    Both inputs are strictly positive (bl89_displacement_lengths
    guarantees it), and min <= the negative power mean <= max, so
    l_eps <= l_mix always: dissipation is never weaker than mixing.
    """
    up64 = np.asarray(l_up, dtype=np.float64)
    dn64 = np.asarray(l_down, dtype=np.float64)
    p = BL89_MIX_EXP
    l_mix = (0.5 * (up64 ** (-p) + dn64 ** (-p))) ** (-1.0 / p)
    return l_mix, np.minimum(up64, dn64)


def bl89_moist_excursion_lengths(n2_eff, e, z, thick):
    """SASE-M1b moist parcel-excursion lengths ``(l_up, l_down)``
    (module docstring, SASE-M1b section; SASE-M spec section 3b).

    The BL89 displacement construction evaluated against the M1 MOIST
    buoyancy: per column and level k, ``l_up(k)`` is the FIRST upward
    displacement l at which

        integral_{z_k}^{z_k + l} R(z') dz' == e(k),
        R(z') = integral_{z_k}^{z'} N^2_eff(s) ds

    with ``n2_eff`` the :func:`moist_n2` effective-stability FIELD
    (DK82 saturated N^2_m in saturated cells, condensate loading
    included; the dry field elsewhere); ``l_down(k)`` is the mirror
    image downward (R(z') = integral_{z'}^{z_k} N^2_eff ds), bounded
    by the column top interface and the surface exactly as the dry
    lengths are.  DERIVATION OF THE FAMILY IDENTITY: in the dry
    machinery the integrand is beta*(theta_v(z') - theta_v(z_k)) with
    beta = g/theta_v(z_k), which IS R(z') for the parcel-frozen-beta
    stratification measure N^2_beta = beta*d(theta_v)/dz -- so this
    function is the SAME accumulate-until-exceed excursion integral
    with the M1 buoyancy substituted as the stratification measure,
    exactly the spec's "excursion integrand is the M1 buoyancy".
    Moist-UNSTABLE stretches (N^2_m < 0) contribute negatively (the
    saturated parcel re-accelerates), so a moist-unstable deck spends
    nothing until the moist-stable lid: l_up = distance-to-lid plus a
    finite penetration -- the geometry the dry-theta lengths cannot
    see (they read the deck's DRY stability and stop parcels inside
    it, hiding the lid, while the free Blackadar l_B fallback caps the
    master length at the wrong scale entirely; the G-M3 defect).

    DISCRETE FORM (exact for piecewise-constant N^2_eff): the M1
    buoyancy exists as a FIELD at layer centers (s^-2), not as a
    potential like theta, so within the segment between adjacent
    centers the stratification measure is held CONSTANT at the
    arithmetic face mean 0.5*(n2_eff[j] + n2_eff[j+1]) -- the module's
    face convention (:func:`_face_average`) -- making R(z') piecewise
    LINEAR and continuous (R accumulated across segments as
    c1_next = c1 + N^2_seg*h) and the outer integral exactly quadratic
    per segment: :func:`_bl89_first_crossing` solves the fractional
    last segment with c1 = R at the segment start and c2 = N^2_seg/2,
    identically the dry quadrature class.  End segments (surface to
    the lowest center, highest center to the top interface) carry
    slope 0 -- the constant-extension convention of the dry lengths.
    LEVEL BRACKETING (pinned by the deck-under-lid fixture): n2_eff
    rides the centered clamped stencil, so a lid at interface z_L is
    felt by the segment BELOW the lid-adjacent center -- excursions
    terminate within one cell of the geometric distance-to-lid,
    (z_L - z_k) - dz <= l_up <= (z_invtop - z_k) on the fixture.
    ``e`` is floored at E_MIN (module idiom); with rem > 0 and the
    strictly positive geometric bounds (z_k >= thick_0/2), both
    lengths are strictly positive.  Zero new constants: E_MIN, the
    face mean, and the crossing solver are all pre-existing machinery.
    Shapes follow :func:`bl89_displacement_lengths` (``n2_eff`` full
    (nz, ...) layout; ``e``/``z``/``thick`` broadcast against it).
    """
    n2 = np.asarray(n2_eff, dtype=np.float64)
    shape = n2.shape
    nz = shape[0]
    e64 = np.broadcast_to(
        np.maximum(np.asarray(e, dtype=np.float64), E_MIN), shape)
    z64 = np.broadcast_to(np.asarray(z, dtype=np.float64), shape)
    t64 = np.broadcast_to(np.asarray(thick, dtype=np.float64)
                          * np.ones((1,) * n2.ndim), shape)
    htop = z64[-1] + 0.5 * t64[-1]
    trail = shape[1:]
    l_up = np.empty(shape)
    l_down = np.empty(shape)
    for k in range(nz):
        for down in (False, True):
            rem = e64[k].copy()
            acc = np.zeros(trail)
            run = np.zeros(trail)              # R at the segment start
            active = np.ones(trail, dtype=bool)
            out_k = np.zeros(trail)
            prev_z = z64[k]
            nodes = range(k - 1, -2, -1) if down else range(k + 1, nz + 1)
            for j in nodes:
                if down:
                    node_z = z64[j] if j >= 0 else np.zeros(trail)
                    h = prev_z - node_z
                    slope = (0.5 * (n2[j] + n2[j + 1]) if j >= 0
                             else np.zeros(trail))
                else:
                    node_z = z64[j] if j < nz else htop
                    h = node_z - prev_z
                    slope = (0.5 * (n2[j - 1] + n2[j]) if j < nz
                             else np.zeros(trail))
                c1 = run
                pos = h > 0.0
                c2 = np.where(pos, 0.5 * slope, 0.0)
                s = _bl89_first_crossing(c1, c2, rem, h)
                hit = active & np.isfinite(s)
                out_k = np.where(hit, acc + s, out_k)
                seg = np.where(pos, (c1 + c2 * h) * h, 0.0)
                rem = np.where(active & ~hit,
                               np.maximum(rem - seg, 0.0), rem)
                acc = acc + np.where(active, h, 0.0)
                run = run + np.where(pos, slope * h, 0.0)
                active = active & ~hit
                prev_z = node_z
                if not active.any():
                    break
            bound = z64[k] if down else htop - z64[k]
            out_k = np.where(active, bound, out_k)
            if down:
                l_down[k] = out_k
            else:
                l_up[k] = out_k
    return l_up, l_down


def bl89_rans_lengths(theta, e, z, thick, n2=None, z0=0.0, n2_dry=None):
    """S3-6h RANS-limb vertical lengths ``(l_mix_rans, l_eps_rans)``.

    The composed lengths the split step's RANS limb rides (module
    docstring, S3-6h section):

        l_mix_rans = min(l_B, l_s, l_mix_BL89)   [the mixing length]
        l_eps_rans = min(l_B,      l_eps_BL89)   [the dissipation limb]

    SASE-M1b seam (module docstring, SASE-M1b section; spec section
    3b): with ``n2_dry`` given (the DRY N^2 field, so ``n2`` must be
    the M1 effective field), the M1-substituted cells (n2 != n2_dry --
    the seam's own mask, exactly the cells where the moist closure
    claimed authority) have BOTH composed lengths ADDITIONALLY bounded
    by the moist master-length limb

        l_m = min(l_up_m, l_down_m)     [MOIST_MASTER_LENGTH =
                                         "bl89-n2eff-excursion-min-v1"]

    from :func:`bl89_moist_excursion_lengths` -- the min member is the
    family's own dissipation convention (BL89's l_eps) and the unique
    conservative choice for a master-length CEILING: the amendment
    exists because K at the free fallback length OVER-mixes (the G-M3
    deck-clearing defect), so the bound on the K-side l_mix_rans must
    never be slacker than the tightest excursion direction (a lid
    60 m above must bound the master length at the 60 m class no
    matter how far the parcel could fall).  Unsubstituted cells keep
    their bits VERBATIM (np.where copies the FALSE branch), an empty
    mask skips the machinery entirely (the unsaturated-gate contract,
    poison-pinned by the control/lake fixtures), and ``n2_dry=None``
    (every pre-M1b caller) is the S3-6h composition unchanged bitwise.
    The S3-6h monotonicity statement survives verbatim: no RANS-limb
    length is ever LONGER than its pre-amendment value.

    * the l_B min is the registered kappa-z-class floor/match
      (BL89_KZ_MATCH): near the surface l_B -> kappa*(z + z0) enforces
      the neutral log-layer constant exactly (in a neutral column the
      BL89 lengths are pure geometry, l_down = z >= l_B and
      l_up = htop - z, so the min leaves l_B bitwise everywhere except
      within ~l_B of the model top), and aloft it retains the audited
      Blackadar asymptote;
    * the l_s min on the MIXING length is the registered retention
      decision (BL89_LS_DECISION, adjudicated at the uniform-
      stratification fixture) -- applied here through the frozen
      :func:`vertical_mixing_length` so the composition is literally
      "the v0 length, further bounded by BL89";
    * the dissipation limb takes NO l_s here because the l_d blend's
      OUTER min applies l_s once (:func:`dissipation_length`, the
      S3-6b double-counting rule).

    The whole amendment is therefore monotone: no RANS-limb length is
    ever LONGER than its pre-S3-6h value, so every suppression the v0
    formulation achieved is preserved and BL89 can only add more.
    Monkeypatching THIS function to return
    ``(vertical_mixing_length(z, e, n2, z0), broadcast l_B)``
    reproduces the pre-S3-6h formulation bitwise at f = 0 (the RED-leg
    idiom of the jet-decoupling fixture, mirroring S3-6g's
    PR_RANS-pinning argument).
    """
    e64 = np.maximum(np.asarray(e, dtype=np.float64), E_MIN)
    lb = _blackadar_length(z, z0) * np.ones_like(e64)
    l_les = vertical_mixing_length(z, e, n2, z0)      # min(l_B, l_s)
    l_up, l_dn = bl89_displacement_lengths(theta, e, z, thick)
    l_mix_bl, l_eps_bl = bl89_combine(l_up, l_dn)
    l_mix_rans = np.minimum(l_les, l_mix_bl)
    l_eps_rans = np.minimum(lb, l_eps_bl)
    if n2_dry is not None and n2 is not None:
        # SASE-M1b moist master-length limb (docstring above; module
        # docstring, SASE-M1b section): min-bound BOTH compositions by
        # the moist excursion length in the M1-substituted cells only.
        subst = (np.asarray(n2, dtype=np.float64)
                 != np.asarray(n2_dry, dtype=np.float64))
        if bool(subst.any()):
            l_up_m, l_dn_m = bl89_moist_excursion_lengths(n2, e, z,
                                                          thick)
            l_m = np.minimum(l_up_m, l_dn_m)
            l_mix_rans = np.where(subst, np.minimum(l_mix_rans, l_m),
                                  l_mix_rans)
            l_eps_rans = np.where(subst, np.minimum(l_eps_rans, l_m),
                                  l_eps_rans)
    return l_mix_rans, l_eps_rans


def stable_limit_coefficient(l_rans, e, n2=None):
    """S3-6i RANS-limb diffusivity coefficient C_r (module docstring,
    S3-6i section): the min-aware blend that decouples the stable-limit
    diffusivity from the neutral-wall calibration C_KV.

        rho = min(l_rans/l_s, 1),  l_s = LS_COEF*sqrt(e)/N  [N^2 > 0]
        C_r = C_KV + (C_KS/LS_COEF - C_KV)*rho**CKS_BLEND_EXP

    with rho = 0 where N^2 <= 0 or ``n2`` is None, so K_v =
    C_r*l_rans*sqrt(e) recovers

    * the wall/neutral calibration C_KV FP-EXACTLY where the
      stability length is slack (neutral/unstable cells: C_KV +
      dC*0.0 == C_KV bitwise -- the log-layer hard constraint), and
    * the decoupled stable limit C_KS*e/N exactly where l_s binds
      (there l_rans == l_s bitwise, since l_s is a term of the
      l_mix_rans min, so rho == 1.0).

    Between the regimes the coefficient interpolates smoothly on how
    close the operating length sits to the local stability limit --
    a bound-by geometry/inversion parcel (l_B or l_BL89 binding well
    under l_s) keeps most of the wall calibration; a parcel at the
    stability limit mixes at the stable calibration.  Continuity
    across the neutral<->stable transition (C^0 in the input n2 --
    linear deficit onset -- and C^1 in N = sqrt(n2); S3-9c narrowed
    claim) and the co-location of the remaining (pre-existing)
    min-switch kinks are derived in the
    module docstring; C_r is bounded in [C_KS/LS_COEF, C_KV] (the
    rho clip makes that hold for ANY l_rans, though every caller
    passes l_rans <= l_s by construction).  ``e`` is floored at E_MIN
    (module idiom); broadcasts like :func:`vertical_mixing_length`.
    """
    l64 = np.asarray(l_rans, dtype=np.float64)
    if n2 is None:
        return np.full(l64.shape, C_KV)
    e64 = np.maximum(np.asarray(e, dtype=np.float64), E_MIN)
    n2 = np.asarray(n2, dtype=np.float64)
    stable = n2 > 0.0
    ls = LS_COEF * np.sqrt(e64) / np.sqrt(np.where(stable, n2, 1.0))
    rho = np.where(stable, np.minimum(l64 / ls, 1.0), 0.0)
    return C_KV + (C_KS / LS_COEF - C_KV) * rho ** CKS_BLEND_EXP


def stable_dissipation_coefficient(l_d, e, n2=None, f=1.0):
    """S3-6k stable-limb DISSIPATION coefficient C_eps (module docstring,
    S3-6k section): the structural mirror of
    :func:`stable_limit_coefficient` on the dissipation half of the pair.

        l_s    = LS_COEF*sqrt(e)/N                        [N^2 > 0]
        rho    = min(l_d/l_s, 1)
        w      = rho**CKS_BLEND_EXP
        C_rans = (1 - w)*C_E + w*C_ES
        C_eps  = f*C_E + (1 - f)*C_rans                   [N^2 > 0]
        C_eps  = C_E                                      [otherwise]

    so eps = C_eps*e^{3/2}/l_d recovers

    * the neutral-wall calibration C_E BITWISE in neutral/unstable cells
      and where ``n2`` is None -- by SELECTION, the same
      ``if (has_ces && ls_v > 0.0f)`` gate the device carries, not by
      arithmetic cancellation: at N^2 <= 0 the blend would reduce to
      f*C_E + (1-f)*C_E, which is NOT C_E bitwise at general f (3661 of
      a uniform 10001-point f grid land 1 ulp off, measured this
      session; the recorded production f = 4.1188928660938e-05 is one
      of them).  So the log-layer engine, the C_KV = C_E**(1/3)
      identity and the M1-substituted cells are untouched, and the FP64
      authority and its FP32 mirror agree there bitwise;
    * the Deardorff stable asymptote C_ES BITWISE where l_s binds l_d --
      :func:`dissipation_length` ends with the same ``np.minimum(l, ls)``
      this function divides by, so l_d == l_s bitwise there and rho is
      exactly 1.0.  Measured this session with
      ``np.float64(x).tobytes().hex()``: the two-product form returns
      0.19 (``52b81e85eb51c83f``) while the affine
      C_E + (C_ES - C_E)*w returns 0.19000000000000006
      (``54b81e85eb51c83f``).  That is why the form is two-product;
    * C_E BITWISE at f = 1 for ANY stratification (1.0*C_E +
      0.0*C_rans), the S3-6i LES-limb property carried onto the
      dissipation half, which is what keeps every LES fixture inert with
      the switch ON as well as OFF.

    ``l_d`` is the DISSIPATION length -- the length the coefficient
    multiplies -- never ``l_mix_rans``: a coefficient must pair with its
    own length.  ``e`` must be the SAME field :func:`dissipation_length`
    consumed (it is floored at E_MIN here exactly as there, so a caller
    that passes the post-source e* can drive rho above 1 and would be
    silently clipped rather than caught).  Broadcasts like
    :func:`stable_limit_coefficient`.
    """
    l64 = np.asarray(l_d, dtype=np.float64)
    if n2 is None:
        return np.full(l64.shape, C_E)
    e64 = np.maximum(np.asarray(e, dtype=np.float64), E_MIN)
    n2 = np.asarray(n2, dtype=np.float64)
    stable = n2 > 0.0
    ls = LS_COEF * np.sqrt(e64) / np.sqrt(np.where(stable, n2, 1.0))
    rho = np.where(stable, np.minimum(l64 / ls, 1.0), 0.0)
    w = rho ** CKS_BLEND_EXP
    c_rans = (1.0 - w) * C_E + w * C_ES
    # The neutral/unstable return is the SELECTED literal C_E, never the
    # blend evaluated at a C_rans that happens to equal it: f*C_E +
    # (1-f)*C_E is NOT C_E bitwise at general f (measured this session
    # over a uniform 10001-point f grid, 3661 values return
    # 0.9299999999999999 or 0.9300000000000002 instead of 0.93
    # c3f5285c8fc2ed3f -- including the recorded production
    # f = 4.1188928660938e-05).  The device gate is
    # ``if (has_ces && ls_v > 0.0f)`` (sase.cu, S3-6k block), which
    # leaves the multiplicand the literal (double)c_e wherever no
    # stability length exists; this np.where IS that gate, so the
    # authority and its mirror agree bitwise on the cells the amendment
    # is documented not to touch.
    return np.where(stable, f * C_E + (1.0 - f) * c_rans, C_E)


def neutral_dissipation_length(z, delta, f=1.0, z0=0.0):
    """S3-12 STATE-INDEPENDENT reference length of the additive channel.

        l_ref = delta**f * l_B(z + z0)**(1 - f)

    the same geometric regime blend :func:`dissipation_length` runs,
    evaluated on the NEUTRAL member of each limb (LD_ADDITIVE_CHANNEL):
    ``delta`` at f = 1 -- Deardorff's own filter width, FP-exact -- and
    the Blackadar length at f = 0.  Carries NO ``e`` and no ``n2``,
    which is the property the whole amendment rests on: divided into
    ``C_ED*e**1.5`` it yields a dissipation channel that is genuinely
    e^{3/2}, where the e-linear channel's own RANS input
    (min(l_B, l_eps_BL89)) is not, because the BL89 displacement lengths
    solve a parcel-energy integral and scale as sqrt(e).

    Endpoints are FP-exact by the same IEEE-754 identities
    :func:`dissipation_length` documents (x**1.0 == x, x**0.0 == 1.0).
    Broadcasts over ``z`` like :func:`_blackadar_length`.
    """
    lb = _blackadar_length(np.asarray(z, dtype=np.float64), z0)
    return float(delta) ** f * lb ** (1.0 - f)


def additive_dissipation_coefficient(l_d, l_ref, e, n2=None, f=1.0,
                                     c_base=C_E):
    """S3-12 effective dissipation coefficient with Deardorff's SECOND
    channel added (module docstring, S3-12 section).

        l_s   = LS_COEF*sqrt(max(e, E_MIN))/N              [N^2 > 0]
        rho   = min(l_d/l_s, 1)
        w     = rho**CKS_BLEND_EXP
        C_eps = c_base + (1 - f)*w*C_ED*(l_d/l_ref)        [N^2 > 0]
        C_eps = c_base                                     [otherwise]

    so that eps = C_eps*e^{3/2}/l_d is EXACTLY

        eps = c_base*e^{3/2}/l_d  +  (1-f)*w*C_ED*e^{3/2}/l_ref,

    i.e. the HEAD channel untouched plus a second channel divided by a
    length that carries no ``e``.  Returning the effective COEFFICIENT
    rather than the rate keeps the split step's analytic decay substep
    exact: the sum is still of the form -K*e^{3/2}, so
    e = e*/(1 + K*sqrt(e*)*dt/2)^2 holds with no new integrator.

    Four properties, each pinned, each essential:

    * NOWHERE WEAKER.  Every factor of the added term is non-negative
      and l_d/l_ref is finite, so C_eps >= c_base POINTWISE and the
      dissipation rate is nowhere below HEAD's.  This is the property
      LD_STABILITY_LIMIT_REJECTED lacked -- removing a min can only
      lengthen l_d, and this term cannot shorten it.
    * NEUTRAL/UNSTABLE BITWISE, BY SELECTION.  At n2 <= 0 or
      ``n2 is None`` the SELECTED ``c_base`` is returned, exactly the
      S3-6k gate (``if (has_ced && ls_v > 0.0f)`` on the device), not an
      arithmetic cancellation: c_base + 0.0*x is c_base bitwise only
      because the multiply is selected away, and the np.where IS the
      selection.  The neutral log layer, the C_KV**3 = C_E identity and
      every M1-substituted cell are therefore untouched.
    * LES LIMB BITWISE.  The (1 - f) factor makes the change RANS-only:
      at f = 1 the added term is exactly 0.0 for ANY stratification, so
      every LES fixture is inert with the switch on.  This is not only
      the S3-6i/S3-6k pattern -- it is required, because at f = 1
      c_base = C_E = 0.93 already EXCEEDS Deardorff's own neutral total
      c_eps,1 + c_eps,2 = 0.70, so the LES limb is not the limb the
      published coefficient is missing from.
    * STABILITY-GATED.  w = rho**CKS_BLEND_EXP is 1.0 exactly where l_s
      binds l_d (:func:`dissipation_length` ends with the same
      ``np.minimum(l, ls)``, so l_d == l_s bitwise and rho == 1.0) and
      falls off quadratically as the stability limit goes slack.  A
      neutral RANS column has l_d == l_ref and would otherwise take the
      full C_E + C_ED = 1.44, a 55% dissipation increase that moves the
      log-law slope by ~10% -- outside the log-layer fixture's 2% band.
      The gate is what keeps the wall calibration.

    ``l_d`` is the DISSIPATION length -- the length the coefficient
    multiplies; ``l_ref`` the state-independent reference of
    :func:`neutral_dissipation_length`; ``e`` the SAME field
    :func:`dissipation_length` consumed.  ``c_base`` is the coefficient
    this channel is added TO (C_E on the default path; the S3-6k
    per-cell blend when that switch is also on -- the two amendments
    compose, and the composition is what the config ID records).
    """
    l64 = np.asarray(l_d, dtype=np.float64)
    lref = np.asarray(l_ref, dtype=np.float64)
    base = np.broadcast_to(np.asarray(c_base, dtype=np.float64),
                           np.broadcast(l64, lref).shape)
    if n2 is None:
        return np.array(base)
    e64 = np.maximum(np.asarray(e, dtype=np.float64), E_MIN)
    n2 = np.asarray(n2, dtype=np.float64)
    stable = n2 > 0.0
    ls = LS_COEF * np.sqrt(e64) / np.sqrt(np.where(stable, n2, 1.0))
    rho = np.where(stable, np.minimum(l64 / ls, 1.0), 0.0)
    w = rho ** CKS_BLEND_EXP
    added = (1.0 - f) * w * C_ED * (l64 / lref)
    return np.where(stable, base + added, base)


def _face_average(field):
    """Interior-face value by arithmetic mean of the neighbor centers.

    The face-diffusivity convention of the implicit channel (documented
    choice; harmonic averaging is the alternative for strongly layered
    K, not needed at the smooth K_v profiles of this closure).
    """
    return 0.5 * (field[:-1] + field[1:])


def _column_geometry(shape, dz=None, dz_col=None):
    """(z centers, face spacings h, layer thicknesses) for a column set.

    Exactly one of ``dz`` (uniform spacing; returns scalar h/thickness)
    and ``dz_col`` (layer thicknesses, shape (nz,) or matching
    ``shape``; the ``_z_centers`` cumsum-half-layer convention) must be
    given.  Shapes broadcast against fields of shape ``shape`` (any
    rank with z leading -- (nz,) columns and (nz, ny, nx) fields).
    """
    nz = shape[0]
    trail = (1,) * (len(shape) - 1)
    if (dz is None) == (dz_col is None):
        raise ValueError("exactly one of dz and dz_col must be given")
    if dz_col is None:
        z = ((np.arange(nz, dtype=np.float64) + 0.5) * float(dz))
        return z.reshape((nz,) + trail), float(dz), float(dz)
    t = np.asarray(dz_col, dtype=np.float64)
    if t.ndim == 1:
        if t.shape[0] != nz:
            raise ValueError(
                f"dz_col has {t.shape[0]} levels, field has {nz}")
        t = t.reshape((nz,) + trail)
    elif t.shape != tuple(shape):
        raise ValueError(
            f"dz_col shape {t.shape} must be (nz,) or match field "
            f"shape {tuple(shape)}")
    z = np.cumsum(t, axis=0) - 0.5 * t
    return z, z[1:] - z[:-1], t


def implicit_vertical_diffusion(phi, k_face, dt, dz=None, dz_col=None,
                                drag_bottom=None):
    """Backward-Euler vertical diffusion: one Thomas solve per column.

    Solves (I - dt*D_v) phi_new = phi with D_v the flux-form clamped
    operator: (D_v phi)_k = (F_{k+1/2} - F_{k-1/2})/thick_k, face flux
    F = K_f*(phi_{k+1} - phi_k)/h at the nz-1 interior faces and ZERO
    flux through both ends.  ``k_face`` holds the interior face
    diffusivities (leading dimension nz-1, broadcastable).  Properties
    (each pinned by test):

    * unconditionally stable -- the matrix is an irreducibly diagonally
      dominant M-matrix for any dt, K_f >= 0, so the solve exists and
      backward Euler damps every mode by 1/(1 + dt*lambda_m) in (0, 1];
    * max principle -- the inverse of an M-matrix with unit row sums of
      the diagonal/off-diagonal split is nonnegative with row sums 1,
      so min(phi) <= phi_new <= max(phi) elementwise;
    * conservation -- summing thick_k*(phi_new - phi)_k/dt telescopes
      the face fluxes to the (zero) end fluxes, so sum(thick*phi) is
      exact up to solver roundoff;
    * uniform-dz columns: cell-centered cosines cos(pi*m*(k+1/2)/nz)
      are the exact eigenvectors (DCT-II diagonalization of the
      zero-flux operator) with lambda_m = (4K/dz^2)*sin^2(pi*m/(2nz)),
      giving the closed-form fixture.

    S3-6j ``drag_bottom`` (module docstring, S3-6j section): a
    non-negative linearized surface-stress conductance c [m/s]
    (scalar or broadcastable to the trailing shape) replaces the
    bottom end's zero flux with the IMPLICIT flux
    F_{-1/2} = c*phi_new_0 -- the coefficient dt*c/thick_0 folds into
    the bottom diagonal (the YSU ``diag[0] = 1 + fric`` pattern,
    npref.py:6495-6497).  The augmented matrix adds a positive
    diagonal term, so it stays a strictly diagonally dominant
    M-matrix (nonnegative inverse, row sums <= 1): unconditional
    stability survives, the max principle weakens to the one-sided
    bound |phi_new| <= max|phi| (drag pulls the bottom cell toward
    rest, never past it), and conservation becomes flux-consistent --
    sum(thick*(phi_new - phi)) = -dt*c*phi_new_0 exactly (pinned).
    ``drag_bottom=None`` (and exactly-zero c) is bitwise the zero-flux
    solve.

    The Thomas sweep runs FP64, vectorized over the trailing
    dimensions; nz = 1 returns the input unchanged (no faces; a
    ``drag_bottom`` is deliberately NOT applied on this faceless
    branch -- documented, mirrored by the device kernel).
    """
    phi64 = np.asarray(phi, dtype=np.float64)
    nz = phi64.shape[0]
    if nz == 1:
        return phi64.copy()
    kf = np.asarray(k_face, dtype=np.float64)
    if kf.shape[0] != nz - 1:
        raise ValueError(
            f"k_face leading dimension {kf.shape[0]} must be nz-1 = "
            f"{nz - 1}")
    _, h, thick = _column_geometry(phi64.shape, dz=dz, dz_col=dz_col)
    tb = np.broadcast_to(np.asarray(thick, dtype=np.float64), phi64.shape)
    r = dt * kf / h                            # face conductances * dt
    sub = np.zeros_like(phi64)
    sup = np.zeros_like(phi64)
    sub[1:] = -(r / tb[1:])                    # couples k to k-1
    sup[:-1] = -(r / tb[:-1])                  # couples k to k+1
    diag = 1.0 - sub - sup
    if drag_bottom is not None:
        # S3-6j implicit surface stress: dt*c/thick_0 into the bottom
        # diagonal (docstring above; c = 0.0 is an FP-exact no-op).
        c = np.asarray(drag_bottom, dtype=np.float64)
        if np.any(c < 0.0):
            raise ValueError("drag_bottom must be non-negative")
        diag[0] = diag[0] + dt * c / tb[0]
    # Thomas forward elimination / back substitution, columns in bulk.
    cprime = np.empty_like(phi64)
    dprime = np.empty_like(phi64)
    cprime[0] = sup[0] / diag[0]
    dprime[0] = phi64[0] / diag[0]
    for k in range(1, nz):
        m = diag[k] - sub[k] * cprime[k - 1]
        cprime[k] = sup[k] / m
        dprime[k] = (phi64[k] - sub[k] * dprime[k - 1]) / m
    out = np.empty_like(phi64)
    out[-1] = dprime[-1]
    for k in range(nz - 2, -1, -1):
        out[k] = dprime[k] - cprime[k] * out[k + 1]
    return out


def surface_scalar_flux_deposit(theta, qv, dt, rho1, hfx=0.0, qfx=0.0,
                                dz=None, dz_col=None):
    """S3-11a explicit surface scalar-flux deposit (module docstring).

    Returns ``(theta_new, qv_new)`` -- fresh FP64 arrays, inputs never
    mutated -- with the audited YSU surface rows (npref.py:6472/6481,
    delp = rho*g*dz1 form) applied to the lowest layer and every
    level above bitwise untouched:

        theta_new[0] = theta[0] + dt*hfx/(rho1*CP_AIR*thick_0)
        qv_new[0]    = qv[0]    + dt*qfx/(rho1*thick_0)

    ``hfx`` [W m^-2] and ``qfx`` [kg m^-2 s^-1] are the DIMENSIONAL
    post-sfclay surface fluxes ((ny, nx)-broadcastable or scalar;
    positive UPWARD, so a positive hfx warms and a negative hfx cools
    the bottom layer); ``rho1`` [kg m^-3] the lowest-level moist air
    density (the same rho ``physics.sase_surface_e_source`` computes;
    strictly positive -- rejected otherwise, since a sign-flipped
    density silently inverts the flux); layer thicknesses come from
    ``dz``/``dz_col`` exactly as the implicit solver takes them
    (:func:`_column_geometry`).  The deposit is REGISTERED to run
    BEFORE the implicit K_v/Pr_t vertical solve of the same scalar
    fields (SFC_SCALAR_FLUX = "explicit-deposit-v1"); the composed
    update carries the boundary-consistent scalar ledger of the
    S3-11a section.  The default hfx = qfx = 0.0 adds literally +0.0
    to the bottom row: bitwise identity for every finite value except
    -0.0 (physical theta > 0, qv >= +0.0) -- the zero-flux seam-off
    contract, pinned by the identity fixture.  qc/qi deliberately
    take NO deposit (YSU's cloud/ice rows carry no surface source,
    npref.py:6485-6486).
    """
    th = np.asarray(theta, dtype=np.float64)
    q = np.asarray(qv, dtype=np.float64)
    if th.shape != q.shape:
        raise ValueError(
            f"theta shape {th.shape} and qv shape {q.shape} must match")
    rho = np.asarray(rho1, dtype=np.float64)
    if np.any(rho <= 0.0):
        raise ValueError("rho1 must be strictly positive")
    _, _, thick = _column_geometry(th.shape, dz=dz, dz_col=dz_col)
    t0 = np.broadcast_to(np.asarray(thick, dtype=np.float64),
                         th.shape)[0]
    hfx64 = np.asarray(hfx, dtype=np.float64)
    qfx64 = np.asarray(qfx, dtype=np.float64)
    th_new = th.copy()
    q_new = q.copy()
    th_new[0] = th[0] + dt * hfx64 / (rho * CP_AIR * t0)
    q_new[0] = q[0] + dt * qfx64 / (rho * t0)
    return th_new, q_new


def _vent_gather(arr, idx):
    """Per-column level gather: ``arr[idx[col], col]`` for a leading-z
    array and an index array over the trailing dims (clipped at 0; the
    callers guarantee out-of-range indices are consumed only under
    inactive-column masks)."""
    safe = np.maximum(np.asarray(idx, dtype=np.int64), 0)
    return np.take_along_axis(arr, safe[None], axis=0)[0]


def _vent_saturation_adjust(th_l, qt, pressure, iters=None):
    """SASE-M2 parcel saturation adjustment: ``(theta, qv, qc)`` of a
    parcel with liquid-water potential temperature ``th_l`` and total
    water ``qt`` at ``pressure`` (module docstring, SASE-M2 ASCENT).

    DERIVATION.  With Pi = (p/P0)**(Rd/cp) and theta_l = theta -
    (L/(cp*Pi))*qc (the standard linearized liquid-water potential
    temperature -- the conserved moist-parcel variable of the EDMF
    family), the parcel temperature solves

        f(T) = T - Pi*theta_l - (L/cp)*qc(T) = 0,
        qc(T) = max(qt - q_s(T, p), 0),

    with q_s the SAME Tetens-liquid saturation the M1 core uses
    (:func:`moist_n2`: e_s = 1000*SVP1*exp(SVP2*(T - SVPT0)/
    (T - SVP3)), q_s = EP2_RV*e_s/(p - e_s)).  Newton iteration with
    the ANALYTIC derivative

        de_s/dT = e_s*SVP2*(SVPT0 - SVP3)/(T - SVP3)^2,
        dq_s/dT = EP2_RV*p*de_s/dT/(p - e_s)^2,
        f'(T)   = 1 + (L/cp)*dq_s/dT   [qc > 0; else 1],

    from the unsaturated first guess T0 = Pi*theta_l.  f' >= 1 always
    (dq_s/dT > 0), so the update is well-defined everywhere; a FIXED
    iteration count (``iters`` = VENT_SAT_ADJUST_ITERS = 12, registered
    -- S4-4 review Minor-1 -- deterministic op order for the
    future device mirror) carries the quadratically convergent
    iteration far past FP64 stationarity (measured on the specimen:
    bit-stationary by 8).  ``iters=None`` (the default) resolves the
    registered module constant AT CALL TIME (S4-4 review round-3
    Minor-1(b)): the previous ``iters=VENT_SAT_ADJUST_ITERS`` default
    bound the value at function-DEFINITION time, so a test that
    monkeypatches the module constant after import changed the
    registry hash (``sase_config_id``) without changing behavior --
    the same runtime-live discipline :func:`plume_vent_flux` already
    gives VENT_MIN_RUN_CELLS (a plain global lookup in the loop body).
    Unsaturated states return qc = 0 and theta = theta_l exactly after
    the first residual vanishes.  DOCUMENTED DOMAIN (the
    :func:`moist_n2` convention): physical states have T > SVP3 and
    pressure > e_s; a violating state surfaces loudly as non-finite
    output.
    """
    if iters is None:
        iters = VENT_SAT_ADJUST_ITERS
    thl64 = np.asarray(th_l, dtype=np.float64)
    qt64 = np.asarray(qt, dtype=np.float64)
    p64 = np.asarray(pressure, dtype=np.float64)
    pi = (p64 / P0_REF) ** (RD_AIR / CP_AIR)
    t = pi * thl64
    for _ in range(iters):
        es = 1000.0 * SVP1 * np.exp(SVP2 * (t - SVPT0) / (t - SVP3))
        qs = EP2_RV * es / (p64 - es)
        qc_i = np.maximum(qt64 - qs, 0.0)
        res = t - pi * thl64 - (XLV / CP_AIR) * qc_i
        des = es * SVP2 * (SVPT0 - SVP3) / ((t - SVP3) * (t - SVP3))
        dqs = EP2_RV * p64 * des / ((p64 - es) * (p64 - es))
        slope = np.where(qc_i > 0.0, 1.0 + (XLV / CP_AIR) * dqs, 1.0)
        t = t - res / slope
    es = 1000.0 * SVP1 * np.exp(SVP2 * (t - SVPT0) / (t - SVP3))
    qs = EP2_RV * es / (p64 - es)
    qc_p = np.maximum(qt64 - qs, 0.0)
    return t / pi, qt64 - qc_p, qc_p


def plume_vent_flux(theta, qv, qc, p, dz_col, e_sgs, rho1, n2m_mask,
                    f_blend):
    """SASE-M2 conditional venting limb: face-registered scalar flux
    profiles ``(F_theta, F_qv, F_qc)``, each shape ``(nz+1,) +
    theta.shape[1:]``, with ``F[0] = F[top] = +0.0`` exactly.

    The complete formulation, every derivation, the registered
    constants with provenance, the mask-convention decision, the S4-5
    deposit/ledger/rate-cap contract, and the measured specimen
    numbers live in the module docstring (SASE-M2 section).  Summary
    of the algorithm (all FP64, vectorized over trailing dims):

    1. MASK/MEMBERSHIP: the lowest contiguous run (>=
       VENT_MIN_RUN_CELLS cells) of MEMBER cells -- inside
       ``n2m_mask`` (the M1 saturation/substitution mask, which is
       never re-derived here and can only VETO) AND saturated on the
       registered VENT_K_LID_MEMBERSHIP total-water test
       qt = qv + qc >= qs -- that is moist-unstable under the
       registered VENT_MASK theta_es reading; no qualifying run
       -> bitwise +0.0.  The run's base (k_base) and top (k_top) are
       retained; k_top feeds the entrainment-zone cell and the
       inversion-base diagnostic step 4 uses.
    1a. SURFACE-BASED STAND-DOWN (VENT_ANCHOR_RULE): a qualifying run
       based in the LOWEST MODEL LEVEL (k_base = 0) -> bitwise +0.0.
       There is no cloud base distinct from the ground for the step-5
       shape to normalize on (z_f[0] = 0 identically), and fog /
       surface-based stratus is ED-limb business.  The FOURTH
       registered stand-down condition.
    1b. ROOT: k_r = the highest interior theta_es maximum at or below
       k_base and at or above the structural depth floor
       k_base - (k_top - k_base) - 1 -- the theta_es-decrease base /
       most-unstable level of the spec's Root clause, bounded so the
       source layer is never deeper below the saturated run's base
       than that run's own depth (round-5 amendment, clause 1) --
       falling back to k_base when theta_es never increases with
       height below it.  theta_es is a function of (theta, p) only, so
       the root has no condensate dependence.
    2. AMPLITUDE, anchored at CLOUD BASE (round-5 amendment,
       clause 2): M_base = VENT_MB_COEF*rho1*sqrt(VENT_SIGW_SHARE*
       ebar), ebar the thickness-weighted mean e_sgs from the
       saturated-layer base to the entrainment-zone cell,
       k_base <= k <= k_top + 1;
       M_used = (1 - f_blend)*M_base (the FP-exact two-product blend:
       f = 1 -> bitwise zeros).  Root depth does not enter the
       amplitude at all.
    3. ASCENT from the run-base root: entraining (theta_l, q_t)
       parcel, eps = VENT_ENT_COEF/z, segment-exact update against
       face-mean environments; :func:`_vent_saturation_adjust`
       recovers (theta, qv, qc); loaded-theta_v buoyancy B.
    4. TERMINATION: LFC = first B > 0 center above the root and BELOW
       the inversion base k_lid = k_top + 2 -- the entrainment-zone
       cell (k_top + 1) and then that cell's own TOP face index
       (INVERSION BASE in the module docstring; this line read
       "k_top + 1", the entrainment-zone CELL rather than the lid index
       the code computes -- corrected, spec-sweep ruling 12);
       NB = first B <= 0 center
       above the LFC, at/below k_lid -- if the parcel is still buoyant
       when the search reaches k_lid, k_lid itself is the NB (S4-4
       review Important-2: termination never crosses into the capping
       layer above the mask's own run).  Stand down (bitwise zeros)
       without an LFC below k_lid, or if k_lid itself is beyond
       VENT_DEPTH_CAP of the root, or if k_lid is past the column top.
       Faces at/above NB carry exactly +0.0.
    5. SHAPE: M_hat grows as (z_f[j]/z_f[max(k_base, 1)])**
       VENT_ENT_COEF up to the buoyancy-peak face -- normalized on the
       cloud-base face, so M_hat = 1 there (round-5 amendment,
       clause 2; the max() is the structural statement that face 0
       (z_f[0] = 0) is never a normalization face, and since step 1a
       it can no longer BIND on an active column) -- then detrains on
       the remaining-buoyancy weight
       (exactly zero at the NB face).
    6. FLUX: F_phi[j] = M_used*M_hat[j]*facemean(phi_p - phi_env)_j
       for phi in (theta, qv, qc) -- dynamic units (F_theta
       [K kg m^-2 s^-1]; F_qv/F_qc [kg m^-2 s^-1]); the S4-5 deposit
       is phi_k += (F[k] - F[k+1])*dt/(rho1*thick_k).

    Inputs follow the module layout conventions (``(nz, ny, nx)`` with
    any trailing shape; ``dz_col`` per :func:`_column_geometry`);
    inputs are never mutated; ``rho1`` must be strictly positive (the
    S3-11a idiom) and ``f_blend`` in [0, 1].
    """
    th = np.asarray(theta, dtype=np.float64)
    shape = th.shape
    nz = shape[0]
    trail = shape[1:]
    qv64 = np.asarray(qv, dtype=np.float64)
    qc64 = np.asarray(qc, dtype=np.float64)
    p64 = np.asarray(p, dtype=np.float64)
    e_in = np.asarray(e_sgs, dtype=np.float64)
    mask = np.asarray(n2m_mask)
    for name, arr in (("qv", qv64), ("qc", qc64), ("p", p64),
                      ("e_sgs", e_in), ("n2m_mask", mask)):
        if arr.shape != shape:
            raise ValueError(
                f"{name} shape {arr.shape} must match theta shape "
                f"{shape}")
    mask = mask.astype(bool)
    if VENT_MASK not in ("bulk-theta-es-v1", "per-level-theta-es-v1"):
        raise ValueError(
            f"unknown VENT_MASK convention: {VENT_MASK!r}")
    f = float(f_blend)
    if not 0.0 <= f <= 1.0:
        raise ValueError(f"f_blend must be in [0, 1], got {f}")
    rho64 = np.asarray(rho1, dtype=np.float64)
    if np.any(rho64 <= 0.0):
        raise ValueError("rho1 must be strictly positive")
    rho_b = np.broadcast_to(rho64, trail)
    e64 = np.maximum(e_in, E_MIN)
    z, _, thick = _column_geometry(shape, dz_col=dz_col)
    zb = np.broadcast_to(z, shape)
    tbf = np.broadcast_to(np.asarray(thick, dtype=np.float64)
                          * np.ones((1,) * th.ndim), shape)

    def _zeros3():
        base = np.zeros((nz + 1,) + trail)
        return base, base.copy(), base.copy()

    # -- thermodynamic state (the M1 Tetens/Exner primitives) ----------
    t_env = th * (p64 / P0_REF) ** (RD_AIR / CP_AIR)
    es_env = 1000.0 * SVP1 * np.exp(SVP2 * (t_env - SVPT0)
                                    / (t_env - SVP3))
    qs_env = EP2_RV * es_env / (p64 - es_env)
    thes = th * np.exp(XLV * qs_env / (CP_AIR * t_env))
    qt_env = qv64 + qc64
    thl_env = th - (XLV / CP_AIR) * qc64 * th / t_env
    rvm1 = RV_AIR / RD_AIR - 1.0

    # -- step 1: lowest qualifying saturated moist-unstable run --------
    per_level = VENT_MASK == "per-level-theta-es-v1"
    # S4-4 review round-4 (Critical: NOISE-IMMUNE STRUCTURAL LAYER).
    # The ENTIRE structural layer -- run membership, hence the run's
    # base, its contiguity and its top k_top -- is the noise-immune
    # total-water test qt >= qs (VENT_K_LID_MEMBERSHIP), and the ROOT
    # k_r is then the theta_es-decrease base below it (step 1b);
    # NOTHING structural rides the M1 mask's bit-level ``qc > 0`` limb
    # any more.  Round 3 applied the total-water criterion to k_top
    # ALONE, leaving the run's base and contiguity on ``mask``, so a
    # round-off-scale condensate value still moved k_r and the ebar
    # window and swung the amplitude (module docstring, SASE-M2
    # MEMBERSHIP, has the measured swings).  ``mask`` is retained as a
    # VETO -- the limb still never engages where the M1 seam says
    # unsaturated (the poisoned-mask leg of
    # test_m2_mask_discrimination_bitwise REQUIRES this), and the seam
    # is never re-derived here -- but it can no longer EXTEND a run.
    # The veto is provably never BINDING when the caller passes the M1
    # switch itself, since qt >= qs implies (qc > 0) | (qv >= qs), so
    # it cannot re-admit condensate noise through the back door.  Zero
    # new tunable constants: qt_env, qs_env and thes are all already
    # computed above.
    robust_sat = qt_env >= qs_env
    member = mask & robust_sat
    in_run = np.zeros(trail, dtype=bool)
    run_s = np.zeros(trail, dtype=np.int64)
    base_thes = np.zeros(trail)
    prev_thes = np.zeros(trail)
    mono = np.zeros(trail, dtype=bool)
    chosen = np.zeros(trail, dtype=bool)
    k_base = np.full(trail, -1, dtype=np.int64)
    k_top = np.full(trail, -1, dtype=np.int64)
    for k in range(nz):
        mk = member[k]
        start = mk & ~in_run
        cont = mk & in_run
        run_s = np.where(start, k, run_s)
        base_thes = np.where(start, thes[k], base_thes)
        mono = np.where(start, True,
                        np.where(cont, mono & (thes[k] < prev_thes),
                                 mono))
        prev_thes = np.where(mk, thes[k], prev_thes)
        if k + 1 < nz:
            end_here = mk & ~member[k + 1]
        else:
            end_here = mk
        long_enough = (k - run_s) >= (VENT_MIN_RUN_CELLS - 1)
        if per_level:
            reading = mono
        else:
            reading = (thes[k] - base_thes) < 0.0
        take = end_here & long_enough & reading & ~chosen
        k_base = np.where(take, run_s, k_base)
        # k_top: the run's own top center.  Every member cell passes
        # qt >= qs by construction now, so this IS the last robustly-
        # saturated center the round-3 fix introduced -- reached
        # without the separate ``last_robust`` tracker that fix needed
        # while the run itself was still the raw masked run.
        k_top = np.where(take, k, k_top)
        chosen = chosen | take
        in_run = mk
    active = chosen
    # -- step 1a: SURFACE-BASED STAND-DOWN ----------------------------
    # VENT_ANCHOR_RULE = "cloud-base-face-standdown-v1" (design doc
    # SASE-M section 4, amendment "a surface-based saturated layer
    # stands the limb down"; audit-wave project decision).  The FOURTH
    # registered stand-down condition, alongside no-LFC, VENT_DEPTH_CAP
    # and k_lid past the column top -- and, like all three of those, it
    # returns bitwise +0.0 on all three rows.
    #
    # WHY: the shape normalizes on the CLOUD-BASE face z_f[k_base]
    # (SHAPE, step 5).  z_f[0] = 0 identically by the module's cumsum
    # convention, so a run based in the LOWEST MODEL LEVEL has no cloud
    # base to normalize on at all: the eps = c_eps/z entrainment
    # integral diverges at z = 0.  The pre-amendment structural guard
    # z_f[max(k_base, 1)] silently substituted the lowest layer's
    # THICKNESS for a height, making the amplitude a function of the
    # VERTICAL GRID rather than of the state -- MEASURED on real
    # intermediate-nest fields at the design point (S4-5c survey,
    # whole 11Z frame): the
    # median anchor face is 610.38 m on the 19404 active columns with
    # k_base > 0 against 17.08 m on the 91 surface-based ones, the
    # shape factor at a fixed physical face multiplies by exactly
    # 2**VENT_ENT_COEF per refinement doubling, and those columns carry
    # the population MAXIMUM export in all three surveyed frames.  That
    # violates the C9 grid-consistency contract by measurement.
    #
    # PHYSICS: M2 is a cloud-base mass-flux closure, and a saturated
    # layer sitting on the ground is fog or surface-based stratus -- a
    # regime whose vertical mixing belongs to the ED limb (M1), not to a
    # plume launched from a cloud base that does not exist.  The
    # alternative M_hat == 1 on the branch was considered and REJECTED:
    # it leaves a step discontinuity against k_base = 1 columns, which
    # the C9 continuity clause forbids.  Zero new tunable constants --
    # the rule is the index test k_base > 0, and the registered
    # CONVENTION STRING carries it into the config ID.
    active = active & (k_base > 0)
    if not bool(active.any()):
        return _zeros3()

    # -- step 1b: root = the theta_es-decrease base, depth-bounded ----
    # The spec's own definition (design doc SASE-M section 4, "Root"):
    # the moist-instability source level = the theta_es-decrease base /
    # most-unstable level, NOT the surface (C8).  Discretely: the
    # HIGHEST interior theta_es maximum at or below the run's own base
    # -- the level where theta_es stops increasing with height and the
    # decrease layer the run sits in begins.  ``thes`` is a function of
    # (theta, p) ONLY -- no qv, no qc -- so this root is EXACTLY
    # condensate-noise-immune, which is the whole point of moving it
    # off the mask.  When theta_es never increases with height below
    # the run's base (no interior maximum -- theta_es decreasing all
    # the way to the ground, the constructed deck columns) the decrease
    # layer has no interior base and the run's own base IS the source
    # level, the classical cloud-base root.  Constant-free: only the
    # already-computed thes array and the run's base index.
    #
    # ROUND-5 DEPTH FLOOR (design doc SASE-M2 amendment "root / anchor
    # separation", clause 1): the source layer lies no deeper below the
    # saturated run's base than that run's OWN depth,
    #
    #     k_r >= k_base - (k_top - k_base) - 1,
    #
    # i.e. no deeper than the run's own cell count (k_top - k_base + 1)
    # below its base -- a source layer deeper than the layer it feeds is
    # not a plume source.  Built ONLY from indices step 1 already
    # computed; zero new tunable constants and no numeric height floor.
    # No clamp at 0 is needed: the search itself starts at k = 1, so a
    # negative floor is vacuous.  MEASURED (real intermediate-nest
    # fields at 11Z/13Z, probe v5_p2_pop.py): the floor moves the root
    # on 21.92% / 17.54% of the columns firing in both builds and cuts the
    # sub-100-m root fraction 6.49% -> 1.15% (11Z) and 6.09% -> 3.80%
    # (13Z).
    k_r = k_base.copy()
    k_r_floor = k_base - (k_top - k_base) - 1
    for k in range(1, nz - 1):
        is_peak = (thes[k] > thes[k - 1]) & (thes[k] > thes[k + 1])
        k_r = np.where(active & is_peak & (k <= k_base) & (k >= k_r_floor),
                       k, k_r)

    # -- step 1c: BL-integrated ebar from CLOUD BASE up ----------------
    # ROUND-5 CLOUD-BASE ANCHOR (design doc SASE-M2 amendment "root /
    # anchor separation", clause 2): the amplitude is anchored at the
    # saturated-layer base, NOT at the root.  ebar is the
    # thickness-weighted mean e_sgs from k_base through the
    # entrainment-zone cell k_top + 1 (k_base <= k < k_lid) -- classical
    # mass-flux practice (M_b specified at cloud base; the Grant 2001
    # provenance is registered in VENT_MB_COEF), and it stops ROOT DEPTH
    # from multiplying the amplitude.  Both ends stay noise-immune
    # indices (MEMBERSHIP), so the amplitude stays noise-immune.  The
    # round-4 window ran from k_r instead; MEASURED on real
    # intermediate-nest columns with real TKE_SASE whose base sits
    # >= 2 cells above the root
    # (n = 9046 at 11Z, 8089 at 13Z -- probe v5_p3_window.py), the
    # round-4 window's amplitude is a median 5.81x / 2.76x the
    # cloud-base-anchored one, p95 35.98x / 57.32x, max 401.8x / 469.3x
    # -- a degree of freedom this anchor removes by construction.
    k_hi = np.minimum(k_top + 1, nz - 1)
    ebar_num = np.zeros(trail)
    ebar_den = np.zeros(trail)
    for k in range(nz):
        inw = active & (k >= k_base) & (k <= k_hi)
        ebar_num = np.where(inw, ebar_num + tbf[k] * e64[k], ebar_num)
        ebar_den = np.where(inw, ebar_den + tbf[k], ebar_den)

    # -- step 2: amplitude (module docstring, SASE-M2 AMPLITUDE) -------
    ebar = np.maximum(
        ebar_num / np.where(ebar_den > 0.0, ebar_den, 1.0), E_MIN)
    sigma_w = np.sqrt(VENT_SIGW_SHARE * ebar)
    m_base = VENT_MB_COEF * rho_b * sigma_w
    m_used = (1.0 - f) * m_base            # FP-exact two-product blend

    # -- step 3: entraining parcel ascent ------------------------------
    thl_p = np.zeros(trail)
    qt_p = np.zeros(trail)
    started = np.zeros(trail, dtype=bool)
    b = np.zeros(shape)
    dth = np.zeros(shape)
    dqv = np.zeros(shape)
    dqc = np.zeros(shape)
    for k in range(nz):
        if k > 0:
            adv = started & active
            thl_f = 0.5 * (thl_env[k - 1] + thl_env[k])
            qt_f = 0.5 * (qt_env[k - 1] + qt_env[k])
            fac = (zb[k - 1] / zb[k]) ** VENT_ENT_COEF
            thl_n = thl_f + (thl_p - thl_f) * fac
            qt_n = qt_f + (qt_p - qt_f) * fac
            th_p, qv_p, qc_p = _vent_saturation_adjust(
                np.where(adv, thl_n, 300.0), np.where(adv, qt_n, 0.0),
                p64[k])
            tv_p = th_p * (1.0 + rvm1 * qv_p - qc_p)
            tv_e = th[k] * (1.0 + rvm1 * qv64[k] - qc64[k])
            bk = G_ACCEL * (tv_p - tv_e) / tv_e
            b[k] = np.where(adv, bk, 0.0)
            dth[k] = np.where(adv, th_p - th[k], 0.0)
            dqv[k] = np.where(adv, qv_p - qv64[k], 0.0)
            dqc[k] = np.where(adv, qc_p - qc64[k], 0.0)
            thl_p = np.where(adv, thl_n, thl_p)
            qt_p = np.where(adv, qt_n, qt_p)
        init = active & (k_r == k)
        thl_p = np.where(init, thl_env[k], thl_p)
        qt_p = np.where(init, qt_env[k], qt_p)
        started = started | init

    # -- step 4: LFC / NB termination + inversion-base cap ------------
    # S4-4 review Important-2, AMENDED round-3 (project decision,
    # design doc SASE-M2 amendment): the search territory is bounded to
    # the (robustly-defined) saturated run PLUS its single DISCRETE
    # ENTRAINMENT ZONE cell (k_top + 1, the legitimate and only
    # above-deck detrainment recipient); k_lid = (k_top + 1) + 1 is
    # that cell's own TOP face index -- the C9 hard-zero boundary.  A
    # parcel still buoyant when the search reaches k_lid terminates
    # there (module docstring, SASE-M2 TERMINATION / INVERSION BASE) --
    # termination never crosses into the CAP proper above the
    # entrainment zone.  Was k_top + 1 (the run's own top + 1); the
    # semantics shifted from "last in-run cell + 1" to "entrainment-
    # zone cell + 1" when k_top itself became the robust (not raw)
    # run-top -- on the specimen the two formulas land on the SAME
    # numeric k_lid = 17, because k_top itself moved 16 -> 15 under the
    # membership fix (recipient/zero structure UNCHANGED there); on
    # constructed columns whose top was never noise-classified the
    # entrainment zone is a genuinely new (one-cell-higher) search cell
    # (see test_m2_termination_grid_consistency).
    k_lid = k_top + 2
    zr = _vent_gather(zb, k_r)
    lfc_found = np.zeros(trail, dtype=bool)
    nb_found = np.zeros(trail, dtype=bool)
    k_nb = np.full(trail, -1, dtype=np.int64)
    kb = np.full(trail, -1, dtype=np.int64)
    bmax = np.zeros(trail)
    for k in range(nz):
        above = active & (k > k_r)
        within_layer = above & (k < k_lid)
        incap = within_layer & ((zb[k] - zr) <= VENT_DEPTH_CAP)
        pos = b[k] > 0.0
        upd_lfc = incap & ~lfc_found & pos
        lfc_found = lfc_found | upd_lfc
        cand = incap & lfc_found & ~nb_found & pos
        upd_b = cand & (b[k] > bmax)
        bmax = np.where(upd_b, b[k], bmax)
        kb = np.where(upd_b, k, kb)
        upd_nb = incap & lfc_found & ~nb_found & ~pos
        k_nb = np.where(upd_nb, k, k_nb)
        nb_found = nb_found | upd_nb
        at_lid = (above & (k == k_lid) & lfc_found & ~nb_found
                  & ((zb[k] - zr) <= VENT_DEPTH_CAP))
        k_nb = np.where(at_lid, k, k_nb)
        nb_found = nb_found | at_lid
    active = active & nb_found
    if not bool(active.any()):
        return _zeros3()

    # -- step 5: face mass-flux shape M_hat ----------------------------
    karr = np.arange(nz).reshape((nz,) + (1,) * len(trail))
    w_taper = np.where(active[None] & (karr > kb[None])
                       & (karr < k_nb[None]), b * tbf, 0.0)
    ss = np.zeros((nz + 1,) + trail)
    ss[:nz] = np.flip(np.cumsum(np.flip(w_taper, axis=0), axis=0),
                      axis=0)
    zf = np.zeros((nz + 1,) + trail)
    np.cumsum(tbf, axis=0, out=zf[1:])
    # ROUND-5 CLOUD-BASE ANCHOR (design doc SASE-M2 amendment "root /
    # anchor separation", clause 2): the shape normalizes on the
    # CLOUD-BASE face z_f[k_base] -- the bottom face of the first
    # saturated cell -- so M_hat = 1 there and M_used IS the cloud-base
    # mass flux M_b the Grant closure specifies.  Was z_f[k_r + 1] (the
    # root cell's own top face), which made the whole profile scale as
    # (z_face/z_root)**c_eps and let ROOT DEPTH multiply the amplitude.
    # k_base = 0 GUARD, structural (no tuned floor): face 0 is the
    # ground, z_f[0] = 0 identically by the module's cumsum convention
    # and F[0] = +0.0 by the interface contract, so face 0 can never be
    # a normalization face.  KEPT, and now provably a NO-OP on every
    # column that reaches here: step 1a stands a surface-based run down
    # (VENT_ANCHOR_RULE), so k_base >= 1 on every active column and the
    # max() cannot bind.  It is retained as the structural statement
    # that face 0 is not a normalization face -- do NOT read it as
    # active protection, and do NOT delete it as dead code: it is what
    # keeps the gather in bounds for the INACTIVE columns (k_base = -1),
    # whose value the np.where below discards.  MEASURED on real
    # intermediate-nest fields before the amendment: k_base = 0 on
    # 91/19495 (11Z),
    # 661/25626 (12Z), 307/22179 (13Z) active columns at the design
    # point, so the branch was live, not hypothetical.
    zf_anchor = _vent_gather(zf, np.maximum(k_base, 1))
    zf_anchor = np.where(active, zf_anchor, 1.0)  # guard inactive cols
    den = _vent_gather(ss, kb + 1)
    mh_pk = (_vent_gather(zf, kb + 1) / zf_anchor) ** VENT_ENT_COEF
    mh = np.zeros((nz + 1,) + trail)
    for j in range(1, nz):
        # (j < k_nb) also covers the degenerate empty-taper case
        # kb + 1 == k_nb (buoyancy peak directly under the first
        # non-positive center): the NB face stays exactly +0.0 and the
        # plume detrains fully across the peak cell.
        grow = active & (j >= k_r + 1) & (j <= kb + 1) & (j < k_nb)
        tap = (active & (j > kb + 1) & (j < k_nb) & (den > 0.0))
        g = (zf[j] / zf_anchor) ** VENT_ENT_COEF
        r = ss[j] / np.where(den > 0.0, den, 1.0)
        mh[j] = np.where(grow, g, np.where(tap, mh_pk * r, 0.0))

    # -- step 6: face fluxes -------------------------------------------
    def _face_flux(dphi):
        out = np.zeros((nz + 1,) + trail)
        for j in range(1, nz):
            gate = (mh[j] > 0.0) & (m_used > 0.0)
            out[j] = np.where(
                gate,
                m_used * mh[j] * (0.5 * (dphi[j - 1] + dphi[j])),
                0.0)
        return out

    return _face_flux(dth), _face_flux(dqv), _face_flux(dqc)


def vent_deposit_rescale(f_theta, f_qv, f_qc, dt, rho1, dz_col=None,
                         dz=None):
    """SASE-M2 deposit seam: the registered CAP-FAMILY uniform rescale.

    The reference implementation of the enforcement the S4-5 driver seam
    performs (module docstring, SASE-M2 RATE CAP -- previously specified
    in prose only, which an S4-5 review flagged as a prose-only parity
    target for the device mirror; this closes it).  Takes the three
    face-registered flux profiles :func:`plume_vent_flux` returns and
    returns ``(d_theta, d_qv, d_qc, scale)`` -- the CAPPED per-level
    explicit deposits and the per-column rescale factor:

        d_phi_k = (Fs[k] - Fs[k+1]) * dt / (rho1 * thick_k),
        Fs      = s * F,
        s       = min(1, VENT_THETA_STEP_CAP / |d_theta|_max,
                         VENT_QT_STEP_CAP    / |d_qv|_max,
                         VENT_QT_STEP_CAP    / |d_qc|_max),

    each quotient over the column's OWN max of the UNSCALED deposit (not
    over the qv+qc sum, whose cancellation on a pure-phase-conversion
    column makes a sum-only bound vacuous -- module docstring).  The
    rescale is UNIFORM across all three rows and the whole column, which
    is what preserves the three structural properties a per-level clip
    destroys: the interior faces still telescope, both end faces stay
    exactly +0.0, and the qv/qc partition is unchanged.  MEASURED
    (module docstring): a naive per-level clip of one row drove
    sum thick*dtheta to -3.74 where the uniform rescale keeps it at 0.0.

    ``s`` is applied to the FLUXES and the deposits recomputed from the
    scaled fluxes (``Fs``), NOT formed as ``s*d_phi`` -- the two differ in
    the last ulp and the module docstring's registered wording is
    ``F_phi *= s``.

    DIVIDE GUARD (module docstring): an inactive column has every
    ``|d|_max`` exactly +0.0, so each quotient is +inf and ``min`` still
    correctly yields 1.0 -- but the bare division emits a numpy
    RuntimeWarning, which this suite's ``-W error::RuntimeWarning``
    policy treats as a failure.  The quotient is therefore formed under
    an explicit ``where`` on a guarded denominator.

    LEDGER (module docstring, S3-11a scalar ledger extended): with
    ``F[0] = F[nz] = +0.0`` from the interface contract,

        sum_k thick_k*d_phi_k = (dt/rho1) * sum_k (Fs[k] - Fs[k+1])
                              = (dt/rho1) * (Fs[0] - Fs[nz]) = 0

    exactly in exact arithmetic (at machine precision in FP64), for any
    ``s`` -- the uniform factor cancels out of the telescoping.  The
    surface flux is owned by the S3-11a deposit, so ``F[0] = 0`` here is
    the no-double-counting contract, not an approximation.

    Inputs: the three ``(nz+1,) + trail`` face profiles, the step ``dt``,
    the per-column lowest-level moist density ``rho1`` (the S3-11a
    convention, strictly positive), and the layer thicknesses through the
    :func:`_column_geometry` contract.  Inputs are never mutated.
    """
    fa = [np.asarray(a, dtype=np.float64)
          for a in (f_theta, f_qv, f_qc)]
    shape = fa[0].shape
    for name, arr in zip(("f_qv", "f_qc"), fa[1:]):
        if arr.shape != shape:
            raise ValueError(
                f"{name} shape {arr.shape} must match f_theta shape "
                f"{shape}")
    if shape[0] < 2:
        raise ValueError(
            "face profiles need at least 2 faces (nz >= 1)")
    nz = shape[0] - 1
    cell = (nz,) + shape[1:]
    rho = np.asarray(rho1, dtype=np.float64)
    if np.any(rho <= 0.0):
        raise ValueError("rho1 must be strictly positive")
    _, _, thick = _column_geometry(cell, dz=dz, dz_col=dz_col)
    tb = np.broadcast_to(np.asarray(thick, dtype=np.float64)
                         * np.ones((1,) * len(cell)), cell)
    dt64 = float(dt)

    def _deposit(face):
        return (face[:-1] - face[1:]) * dt64 / (rho * tb)

    def _quotient(cap, dep):
        dmax = np.max(np.abs(dep), axis=0)
        # DIVIDE GUARD: +0.0 max (inactive column) -> +inf, formed
        # without evaluating the division (module docstring).
        return np.where(dmax > 0.0,
                        cap / np.where(dmax > 0.0, dmax, 1.0),
                        np.inf)

    raw = [_deposit(a) for a in fa]
    scale = np.minimum(
        np.minimum(np.ones(cell[1:]),
                   _quotient(VENT_THETA_STEP_CAP, raw[0])),
        np.minimum(_quotient(VENT_QT_STEP_CAP, raw[1]),
                   _quotient(VENT_QT_STEP_CAP, raw[2])))
    out = [_deposit(a * scale) for a in fa]
    return out[0], out[1], out[2], scale


def _vertical_production(k_face, comps, h, thick):
    """Cell-centered vertical shear production from implicit-solved fields.

    Per interior face, eps_f = sum_i K_f*(delta phi_i)^2/h is the KE
    drain rate integrated over the face's unit column; splitting each
    face's eps half to each neighbor cell and dividing by the cell
    thickness gives P_k = 0.5*(eps_{k-1/2} + eps_{k+1/2})/thick_k with
    sum_k P_k = sum_f K_f*(delta phi)^2/(h*thick) -- on uniform columns
    exactly -sum phi.(D_v phi), the pairing the ledger theorem uses
    (module docstring, identity (ii)).  Pointwise non-negative.
    """
    shape = comps[0].shape
    p = np.zeros(shape)
    if shape[0] < 2:
        return p
    eps = np.zeros((shape[0] - 1,) + shape[1:])
    for comp in comps:
        d = comp[1:] - comp[:-1]
        eps = eps + k_face * d * d / h
    tb = np.broadcast_to(np.asarray(thick, dtype=np.float64), shape)
    p[:-1] += 0.5 * eps / tb[:-1]
    p[1:] += 0.5 * eps / tb[1:]
    return p


def damp_taper_weights(z, thick, zdamp):
    """S3-6e production-taper weights g(z) on the damp_opt=3 profile.

    ``z`` are layer-center heights ((nz,)+trail or full shape, the
    :func:`_column_geometry` convention), ``thick`` the layer
    thicknesses (scalar, (nz,)+trail, or full shape); the per-column
    top interface is htop = z[-1] + thick[-1]/2.  The weight law is the
    audited KDH damper's (acoustic.cu ``advance_w_phi``,
    sin^2(pi/2*(h - (htop - zdamp))/zdamp) above htop - zdamp),
    complemented and clipped:

        g = 1 - sin^2(pi/2 * clip((z - (htop - zdamp))/zdamp, 0, 1))

    so g = 1 below the damping layer, falls as cos^2 across it, and
    reaches 0 at the top interface.  All FP64.
    """
    z = np.asarray(z, dtype=np.float64)
    thick64 = np.asarray(thick, dtype=np.float64)
    top_half = 0.5 * (thick64 if thick64.ndim == 0 else thick64[-1])
    htop = z[-1] + top_half
    arg = np.clip((z - (htop - float(zdamp))) / float(zdamp), 0.0, 1.0)
    return 1.0 - np.sin(0.5 * np.pi * arg) ** 2


def sase_split_step(u, v, w, theta, e, dx, dy, dz, delta, dt, n2=None,
                    dz_col=None, z0=0.0, zdamp=None, ust=None,
                    wspd_sfc=None, n2_moist=None,
                    stable_dissipation: bool = False,
                    additive_dissipation: bool = False):
    """One S3-6b split SASE-L1 step: explicit horizontal, implicit vertical.

    The amended authority formulation (module docstring has the full
    derivation and the restated ledger theorem).  Order of operations
    -- pinned exactly; the S3-6c device mirror must reproduce it
    bit-compatibly (trajectory goldens in tests/sase_goldens.py):

    1. dynamic solve (unchanged v0 machinery, clamped uniform-dz
       strain, coefficients only) -> f_solved; then the S3-6f
       partition bounds (module docstring, S3-6f section):
       z_i = mean(:func:`bulk_richardson_zi`), f_cap =
       :func:`partition_cap`(delta, z_i), the N^2-screened w-sensor
       bound f_w (:func:`w_resolved_bound` at e_mean = mean(e64)),
       and f = min(f_solved, f_cap, f_w) -- the CAPPED f is what every
       downstream consumer sees (governed stress, l_d blend,
       (1-f) momentum background); c_nu is not rescaled; then the
       S3-6g regime-consistent Prandtl number pr_t =
       :func:`prandtl_blend`(f) at the SAME used f (module docstring,
       S3-6g section);
    2. column geometry (uniform ``dz`` or ``dz_col`` thicknesses; box
       mode assigns nominal heights z_k = (k+1/2)*dz with the box
       bottom as the wall) and the vertical channel at e^n
       (S3-6h/S3-6i): K_v = f*C_KV*l_les*sqrt(e) +
       (1-f)*C_r*l_mix_rans*sqrt(e) with l_les = min(l_B(z+z0), l_s)
       the frozen v0 length (bitwise at f = 1), l_mix_rans =
       min(l_B, l_s, l_mix_BL89) the BL89-bounded RANS-limb length
       (:func:`bl89_rans_lengths`; module docstring, S3-6h section),
       and C_r the S3-6i decoupled stable-limit coefficient
       (:func:`stable_limit_coefficient`: C_KV where the stability
       length is slack, C_KS/LS_COEF where l_s binds -- the stable
       limit K_v = C_KS*e/N); face values by arithmetic mean;
    3. full strain (CLAMPED vertical stencil in both modes -- the split
       theorem needs no periodic vertical) and the S3-6e GOVERNED
       stress (:func:`governed_stress`): nu = f*c_nu*delta*sqrt(e) +
       (1-f)*K_smag(deformation), with the smag share r and the
       governed diffusivity field nu returned alongside tau;
    4. explicit horizontal momentum channel du_i = -(ddx tau_ix +
       ddy tau_iy), u* = u + dt*du; P_h,tot from the horizontal pairing
       of tau against the SAME u^n gradients, split S3-6e into the
       heat-bypass smag share P_h,heat = r*(P_h,tot + (2/3)e*div_h)
       and the e-feeding remainder P_h,e (module docstring);
    5. implicit vertical momentum channel: backward-Euler Thomas per
       column for u, v, w with K_v faces; P_v from the implicit-solved
       gradients (:func:`_vertical_production`).  S3-6j (module
       docstring, S3-6j section): with ``ust`` given ((ny, nx) or
       scalar friction velocity), the u and v solves carry the
       IMPLICIT surface-stress bottom BC -- drag conductance
       c = u*^2/max(|V1^n|, SFC_WSPD_FLOOR) folded into the Thomas
       bottom diagonal (``implicit_vertical_diffusion`` drag_bottom;
       the YSU linearization).  S3-9c (module docstring, S3-9c
       section): with ``wspd_sfc`` ALSO given (sfclay's
       gust-enhanced speed, (ny, nx) or scalar), c gains the audited
       YSU gustiness factor (spd1/max(wspd_sfc, 1e-9))^2
       (npref.py:6495-6496); ``wspd_sfc=None`` forms no factor
       (S3-6j arithmetic bitwise), and ``wspd_sfc`` without ``ust``
       raises.  w, e-transport, and the scalar
       channel stay zero-flux (scope rationale at the docstring);
       the drag applies at ALL f (the lane's one intentional
       cross-limb change);
    6. explicit e sources: buoyancy -(g/theta)*(K_v/Pr_t(f))*ddz(theta)
       (the vertical-channel K_h at the S3-6g blended Prandtl number
       of step 1), horizontal e-transport
       ddx(2K_m ddx e) + ddy(2K_m ddy e) with K_m = the governed nu
       FIELD of step 3 (S3-6e harmonization -- one horizontal
       diffusivity serves stress, e-transport, and scalars), and the
       blended l_d (:func:`dissipation_length` with lb = the S3-6h
       RANS dissipation composition min(l_B, l_eps_BL89) and the live
       f) evaluated at e^n -- dissipation is NOT an explicit source
       any more (S3-6d);
    7. e update (S3-6d substep order, S3-6e taper): with the damping
       weight g (:func:`damp_taper_weights` when ``zdamp`` is given,
       else the FP-exact scalar 1.0), x = P_h,e + P_v, src = g*x,
       gb = g*buoyancy: e* = e + dt*((src + gb) + T_h) -- the withheld
       production (x - src) redirects to heat; ANALYTIC dissipation
       substep
       e* -> e*/(1 + b*dt)^2 with b = C_E*sqrt(max(e*, E_MIN))/(2*l_d)
       -- the exact solution of de/dt = -C_E*e^{3/2}/l_d over dt from
       initial value e* (module docstring; l_d frozen at its step-6
       e^n evaluation), unconditionally stable and positivity-
       preserving where forward Euler overshoots once
       dt*C_E*sqrt(e)/l_d > O(1); clip to E_MIN (clip_gain -> heat
       channel; the floor now engages only where the SOURCES drive e*
       below the floor); then implicit vertical e-transport with
       2*K_v faces (an M-matrix solve preserves the floor up to
       roundoff; a final floor folds any roundoff dip into the
       measured transport channel);
    8. heat = D - clip_gain + dt*(P_h,heat + (x - src)) with D =
       e* - e*/(1 + b*dt)^2 the EXACT per-substep decay decrement, the
       smag bypass, and the taper redirect (S3-6e; heat is no longer
       pointwise sign-definite -- module docstring); ledger with the
       split dKE pairing (explicit channel at u^n, implicit channel at
       u^{n+1}) plus the informational dKE_expl/dKE_impl breakdown;
       the measured buoyancy channel is the tapered gb.  S3-6j: with
       ``ust`` given the ledger carries the boundary channels --
       dKE_sfc (measured drag work at the solved level-1 winds),
       dE_sfc_src (the modeled u*^3 deposit), sfc_conv_resid (their
       diagnosed mismatch) -- and ``residual`` is the
       BOUNDARY-CONSISTENT dKE + dE + dHeat - dKE_sfc (module
       docstring, S3-6j section); all 0.0 with ``ust=None``.

    Scalars (theta, qv, qc, qi) are NOT advanced here -- their vertical
    channel is :func:`implicit_vertical_diffusion` with K_v/Pr_t(f)
    faces and their horizontal channel stays the explicit K_h machinery
    (km_h/Pr_t(f) since S3-6e/6g); the driver wiring is S3-6c scope.
    ``theta`` is read-only (buoyancy).

    SASE-M1 seam (module docstring, SASE-M1 section): ``n2_moist`` is
    the :func:`moist_n2` effective-stability field.  ``None`` (the
    default) is BITWISE the pre-M1 step; given (requires ``n2`` --
    the substitution mask and the w-sensor screen both need the dry
    field), it substitutes at EXACTLY the three spec points: n2_eff =
    n2_moist rides every l_s evaluation and the K_v/K_h stability
    suppression (steps 2b and 6's l_d -- points 1 and 3, coefficients
    only), and the step-6 buoyancy becomes -(K_v/Pr_t)*n2_moist WHERE
    n2_moist != n2 (point 2; elsewhere the literal dry expression
    stands bitwise -- moist_n2 constructs unsaturated cells bitwise-
    dry, so the inequality is exactly the cells where the moist
    closure claimed authority).  The step-1b w-sensor screen keeps
    the DRY n2 (not a substitution point).  The ledger theorem holds
    verbatim (points 1/3 are coefficients; point 2 is the PE-exchange
    channel the dE definition excludes -- C4/C5).  SASE-M1b (module
    docstring, SASE-M1b section): with the seam engaged, the step-2b
    RANS-limb lengths are additionally min-bounded by the moist
    excursion length in the substituted cells (the ``n2_dry`` seam of
    :func:`bl89_rans_lengths`); n2_moist=None remains bitwise the
    pre-M1 step and f = 1 remains bitwise the pre-limb path.

    S3-6k seam (module docstring, S3-6k section):
    ``stable_dissipation=False`` (the default) evaluates the LITERAL
    ``C_E*np.sqrt(...)`` decay coefficient of step 7 and forms no
    coefficient array at all -- bitwise the pre-S3-6k step by
    construction, not by cancellation.  True replaces that scalar with
    the per-cell :func:`stable_dissipation_coefficient` evaluated on the
    SAME (l_d, e, n2_eff, f) the step already holds, so where l_s binds
    l_d the limb dissipates at C_ES*e*N/LS_COEF instead of
    C_E*e*N/LS_COEF.  NOTHING else moves: not l_d, not K_v, not l_s, not
    the M1 seam, not the ledger's own channel definitions (a
    coefficient change inside the analytic decay substep is measured by
    D = e* - e_dec exactly as before, so the theorem holds verbatim).

    S3-12 seam (module docstring, S3-12 section):
    ``additive_dissipation=False`` (the default) leaves the line above
    untouched, so HEAD is bitwise the pre-S3-12 step by construction.
    True ADDS Deardorff's second, grid-scale dissipation channel to
    whichever base coefficient the S3-6k line selected --
    :func:`additive_dissipation_coefficient` on the SAME (l_d, e,
    n2_eff, f) plus the state-independent
    :func:`neutral_dissipation_length` -- so the decay coefficient is
    NOWHERE below the one this step would otherwise use and the
    stable limb gains a dissipation term that is genuinely e^{3/2}.
    The two switches COMPOSE (S3-6k selects the base, S3-12 adds to
    it); each is registered independently in the config ID.  Nothing
    else moves -- not l_d, not K_v, not l_s, not the M1 seam, not the
    ledger's channel definitions: the sum is still of the form
    -K*e^{3/2}, so the analytic decay substep and the theorem's
    D = e* - e_dec both hold verbatim.

    On the uniform-dz horizontally periodic box the returned ledger
    closes to relative roundoff (theorem, pinned < 1e-11); variable
    ``dz_col`` columns are diagnostic-only exactly as in v0.
    """
    u, v, w, e = (np.asarray(a, dtype=np.float64) for a in (u, v, w, e))
    theta64 = np.asarray(theta, dtype=np.float64)
    shape = u.shape
    # SASE-M1 seam (module docstring, SASE-M1 section): n2_eff is what
    # the stability machinery (points 1 and 3) consumes; the w-sensor
    # screen keeps the dry n2.  n2_moist=None leaves n2_eff the SAME
    # object as n2 -- the pre-M1 step bitwise.
    if n2_moist is not None:
        if n2 is None:
            raise ValueError(
                "n2_moist requires n2: the M1 substitution mask derives "
                "from the dry field and the w-sensor screen keeps it")
        n2_eff = np.asarray(n2_moist, dtype=np.float64)
    else:
        n2_eff = n2
    # 1. Coefficients only (v0 contract: clamped uniform-dz strain).
    c_nu, f_solved = dynamic_solve(u, v, w, e, dx, dy, dz, delta)
    # 2. Geometry + vertical channel at e^n.
    z, h, thick = _column_geometry(
        shape, dz=dz if dz_col is None else None, dz_col=dz_col)
    e64 = np.maximum(e, E_MIN)
    root_e = np.sqrt(e64)
    # 1b. S3-6f partition bounds: the Delta/z_i cap and the
    #     N^2-screened w-based resolved-fraction bound cap the SOLVED f
    #     before any consumer reads it (module docstring, S3-6f
    #     section); the solve keeps full authority to LOWER f.
    zi = float(np.mean(bulk_richardson_zi(u, v, theta64, z)))
    f_cap = partition_cap(delta, zi)
    wsens = w_resolved_bound(w, float(np.mean(e64)), n2=n2)
    f = min(f_solved, f_cap, wsens.f_w)
    # 1c. S3-6g regime-consistent Prandtl number at the SAME f_used
    #     (module docstring, S3-6g section): the buoyancy K_h below and
    #     the driver's scalar channels ride Pr_t(f) instead of the
    #     retired fixed PR_T; FP-exact PR_LES at f = 1.
    pr_t = prandtl_blend(f)
    # 2b. S3-6h/S3-6i vertical channel at e^n (module docstring, S3-6h
    #     and S3-6i sections): the RANS limb rides the BL89
    #     parcel-energetics lengths at the S3-6i DECOUPLED stable-limit
    #     coefficient C_r (stable_limit_coefficient: C_KV where the
    #     stability length is slack, C_KS/LS_COEF where l_s binds --
    #     K_v -> C_KS*e/N, the registered asymptote); the LES limb
    #     keeps the v0 C_KV*min(l_B, l_s) bitwise.  The former l_v
    #     length blend is restated as the equivalent two-product K_v
    #     blend so the coefficient change is RANS-only: f = 1 gives
    #     1.0*(C_KV*l_les*sqrt(e)) + 0.0*(...) = the pre-S3-6h channel
    #     FP-exact (the LES-limb pin), f = 0 gives C_r*l_mix_rans*
    #     sqrt(e) exactly (and, in neutral cells, bitwise the S3-6h
    #     value -- C_r == C_KV there).
    l_les = vertical_mixing_length(z, e, n2_eff, z0)
    # SASE-M1b (module docstring, SASE-M1b section): with the seam
    # engaged the RANS-limb compositions are additionally min-bounded
    # by the moist excursion length in the substituted cells (n2_dry
    # gates it inside bl89_rans_lengths); with n2_moist=None the call
    # is LITERALLY the pre-M1b call (no kwarg formed), so every
    # pre-change monkeypatch/RED-leg stand-in keeps its signature.
    # The LES limb l_les is untouched -- the f=1 products erase the
    # RANS limb FP-exactly, so the limb cannot reach the LES limit
    # (requirement 3, pinned).
    mlkw = {} if n2_moist is None else {"n2_dry": n2}
    l_mix_rans, l_eps_rans = bl89_rans_lengths(theta64, e, z, thick,
                                               n2_eff, z0, **mlkw)
    c_rans = stable_limit_coefficient(l_mix_rans, e64, n2_eff)
    kv = (f * (C_KV * l_les * root_e)
          + (1.0 - f) * (c_rans * l_mix_rans * root_e))
    k_face = _face_average(kv)
    # 3. Full strain, clamped vertical stencil in both modes, and the
    #    S3-6e governed stress (tau + the governed nu field + smag
    #    share r).
    s = strain(u, v, w, dx, dy, dz, dz_col=dz_col)
    tau, k_m, r_smag = governed_stress(e64, s, c_nu, f, delta)
    # 4. Explicit horizontal channel + its production pairing, split
    #    into the e-feeding and heat-bypass shares (S3-6e).
    du_h = -(_ddx(tau[0], dx) + _ddy(tau[3], dy))
    dv_h = -(_ddx(tau[3], dx) + _ddy(tau[1], dy))
    dw_h = -(_ddx(tau[4], dx) + _ddy(tau[5], dy))
    u_star = u + dt * du_h
    v_star = v + dt * dv_h
    w_star = w + dt * dw_h
    ux = _ddx(u, dx)
    vy = _ddy(v, dy)
    p_h = -(tau[0] * ux + tau[1] * vy
            + tau[3] * (_ddy(u, dy) + _ddx(v, dx))
            + tau[4] * _ddx(w, dx) + tau[5] * _ddy(w, dy))
    p_h_heat = r_smag * (p_h + (2.0 / 3.0) * e64 * (ux + vy))
    p_h_e = p_h - p_h_heat
    # 5. Implicit vertical momentum channel + implicit-flux production.
    #    S3-6j: with ust given, u and v carry the implicit surface
    #    stress (drag conductance from the PRE-STEP level-1 wind, the
    #    YSU wspd1 convention); w keeps zero-flux ends.
    vkw = {"dz": dz} if dz_col is None else {"dz_col": dz_col}
    if ust is not None:
        ust64 = np.asarray(ust, dtype=np.float64)
        spd1 = np.maximum(np.hypot(u[0], v[0]), SFC_WSPD_FLOOR)
        drag = ust64 * ust64 / spd1
        if wspd_sfc is not None:
            # S3-9c gustiness correction (module docstring, S3-9c
            # section): the audited YSU factor (wspd1/wspd)^2 --
            # npref.np_ysu_column ``fric = ust*ust/wspd1 * ... *
            # (wspd1/max(wspd, 1.0e-9))**2`` (npref.py:6495-6496,
            # wspd1 at npref.py:6145; sfclay's enhanced wspd at
            # npref.py:4257-4266) -- with the resolved speed
            # represented by the SAME floored spd1 the base
            # linearization uses (the registered SFC_WSPD_FLOOR
            # deviation, documented at the docstring).  No-gust
            # wspd_sfc == spd1 gives ratio 1.0 and c*1.0 == c
            # bitwise (the identity pin).
            wspd64 = np.maximum(
                np.asarray(wspd_sfc, dtype=np.float64), 1.0e-9)
            ratio = spd1 / wspd64
            drag = drag * (ratio * ratio)
    elif wspd_sfc is not None:
        raise ValueError(
            "wspd_sfc requires ust: the gustiness factor corrects the "
            "surface drag row, which does not exist without a friction "
            "velocity")
    else:
        drag = None
    u_new = implicit_vertical_diffusion(u_star, k_face, dt, **vkw,
                                        drag_bottom=drag)
    v_new = implicit_vertical_diffusion(v_star, k_face, dt, **vkw,
                                        drag_bottom=drag)
    w_new = implicit_vertical_diffusion(w_star, k_face, dt, **vkw)
    p_v = _vertical_production(k_face, (u_new, v_new, w_new), h, thick)
    # 6. Explicit e sources (dissipation advances analytically below).
    #    S3-6g: the buoyancy K_h is K_v/Pr_t(f) -- the blended Prandtl
    #    number of step 1c, not the retired fixed PR_T.
    buoyancy = -(G_ACCEL / theta64) * (kv / pr_t) * _ddz(
        theta64, dz, dz_col=dz_col)
    if n2_moist is not None:
        # SASE-M1 point 2 (module docstring, SASE-M1 section): where
        # the moist field departed from the dry field -- moist_n2
        # constructs unsaturated cells bitwise-dry, so the inequality
        # is exactly the substituted cells -- the buoyancy source
        # becomes -(K_v/Pr_t)*N^2_m; elsewhere the dry expression
        # above stands LITERALLY (np.where copies its bits).
        subst = n2_eff != np.asarray(n2, dtype=np.float64)
        buoyancy = np.where(subst, -(kv / pr_t) * n2_eff, buoyancy)
    # S3-6h: the l_d blend's RANS limb is the BL89 dissipation
    # composition min(l_B, l_eps_BL89) (the outer l_s min inside
    # dissipation_length is unchanged -- one stability limit, S3-6b
    # rule); f = 1 keeps l_d = delta FP-exact as before.  S3-9: the
    # blend itself is geometric, delta**f * l_eps_rans**(1-f) (module
    # docstring, S3-9 section; endpoints bitwise-unchanged).  SASE-M1:
    # the outer l_s min rides n2_eff (point 1).
    ld = dissipation_length(e, delta, n2_eff, lb=l_eps_rans, f=f)
    # S3-6e: the e-transport K_m IS the governed nu field of step 3
    # (harmonization; the v0 bare-C_K blend retired on this path).
    t_h = (_ddx(2.0 * k_m * _ddx(e64, dx), dx)
           + _ddy(2.0 * k_m * _ddy(e64, dy), dy))
    # 7. e update (S3-6d substeps, S3-6e taper): tapered explicit
    #    deposit, ANALYTIC decay substep, clip-to-heat, implicit
    #    transport.  g = 1.0 (scalar, FP-exact) without a zdamp.
    if zdamp is not None and float(zdamp) > 0.0:
        g = damp_taper_weights(z, thick, zdamp)
    else:
        g = 1.0
    x_prod = p_h_e + p_v
    src = g * x_prod
    gb = g * buoyancy
    e_star = e + dt * ((src + gb) + t_h)
    # S3-6k (module docstring, S3-6k section): the decay coefficient is
    # C_E everywhere UNLESS the stable-limb decoupling is switched on,
    # in which case it becomes the per-cell blend on the SAME l_d this
    # line divides by, the SAME e^n dissipation_length consumed (NOT
    # e_star -- rho would exceed 1 and clip silently), the SAME n2_eff
    # (so the seam is bitwise absent from unstable and M1-substituted
    # cells) and the SAME used f (so the LES limb is bitwise C_E).  With
    # the switch off the literal C_E scalar is evaluated and no array is
    # formed: bitwise the pre-S3-6k step by construction.
    # S3-12 (module docstring, S3-12 section): the additive channel then
    # ADDS Deardorff's second, grid-scale member to whichever base the
    # line above selected, on the SAME l_d, the SAME e^n, the SAME
    # n2_eff and the SAME used f -- so it inherits every bitwise
    # inertness property above and adds the (1-f) LES-limb gate of its
    # own.  With BOTH switches off the literal C_E scalar is evaluated
    # and no array is formed: bitwise the pre-S3-12 step by
    # construction.
    if stable_dissipation or additive_dissipation:
        c_eps = (stable_dissipation_coefficient(ld, e64, n2_eff, f=f)
                 if stable_dissipation else C_E)
        if additive_dissipation:
            l_ref = neutral_dissipation_length(z, delta, f=f, z0=z0)
            c_eps = additive_dissipation_coefficient(
                ld, l_ref, e64, n2_eff, f=f, c_base=c_eps)
        b = c_eps * np.sqrt(np.maximum(e_star, E_MIN)) / (2.0 * ld)
    else:
        b = C_E * np.sqrt(np.maximum(e_star, E_MIN)) / (2.0 * ld)
    decay_factor = 1.0 + b * dt
    e_dec = e_star / (decay_factor * decay_factor)
    decay = e_star - e_dec                     # exact decrement D
    e_clip = np.maximum(e_dec, E_MIN)
    clip_gain = e_clip - e_dec
    e_new = np.maximum(
        implicit_vertical_diffusion(e_clip, 2.0 * k_face, dt, **vkw),
        E_MIN)
    # 8. Heat (decay + smag bypass + taper redirect) + split ledger.
    heat = (decay - clip_gain) + dt * (p_h_heat + (x_prod - src))
    dke_expl = float(np.sum(u * (u_star - u) + v * (v_star - v)
                            + w * (w_star - w)))
    dke_impl = float(np.sum(u_new * (u_new - u_star)
                            + v_new * (v_new - v_star)
                            + w_new * (w_new - w_star)))
    d_ke = dke_expl + dke_impl
    d_e = (float(np.sum(e_new - e)) - float(np.sum(dt * gb))
           - float(np.sum(dt * t_h)) - float(np.sum(e_new - e_clip)))
    d_heat = float(np.sum(heat))
    # S3-6j surface channels (module docstring, S3-6j section):
    # dKE_sfc is the MEASURED drag work at the solved level-1 wind
    # (the boundary flux the closure identity now carries on its right
    # side); dE_sfc_src the modeled u*^3 similarity deposit the driver
    # feeds e; sfc_conv_resid their DIAGNOSED mismatch -- a modeling
    # residual, recorded per step, never forced closed.  All exactly
    # 0.0 with ust=None (residual then reduces bitwise: x - 0.0 == x).
    if drag is not None:
        tb0 = np.broadcast_to(
            np.asarray(thick, dtype=np.float64), shape)[0]
        dke_sfc = -dt * float(np.sum(
            drag * (u_new[0] ** 2 + v_new[0] ** 2) / tb0))
        de_sfc_src = dt * float(np.sum(
            (ust64 ** 3 / (KARMAN * 0.5 * tb0))
            * np.ones_like(u_new[0])))
    else:
        dke_sfc = 0.0
        de_sfc_src = 0.0
    fields = {"u": u_new, "v": v_new, "w": w_new, "e": e_new, "heat": heat}
    ledger = {"dKE": d_ke, "dE": d_e, "dHeat": d_heat,
              "residual": d_ke + d_e + d_heat - dke_sfc,
              "dKE_sfc": dke_sfc, "dE_sfc_src": de_sfc_src,
              "sfc_conv_resid": de_sfc_src + dke_sfc,
              "c_nu": c_nu, "f": f,
              "dKE_expl": dke_expl, "dKE_impl": dke_impl,
              # S3-6f partition-bound diagnostics ("f" above is the
              # value the step USED = min of the next three).
              "f_solved": f_solved, "f_cap": f_cap, "f_w": wsens.f_w,
              "zi": zi, "w_coverage": wsens.coverage,
              # S3-6g: the blended Prandtl number the step USED
              # (= prandtl_blend(f); diagnostic -- the driver
              # recomputes the same value from the retained f).
              "pr_t": pr_t}
    return fields, ledger


def column_neutral_log_layer(u_star, nz, dz, dt, steps, z0=0.0):
    """Neutral constant-stress column: the log-layer fixture engine.

    A 1-D column driven toward the neutral constant-flux equilibrium of
    the amended vertical channel: a prescribed momentum flux u*^2
    enters through the top face and exits through the surface (explicit
    end-face sources; the interior faces are the zero-flux Thomas
    solve), so the steady state carries flux u*^2 through every
    interior face and the profile shape satisfies K_v*du/dz = u*^2.
    The e budget runs P_v (implicit-flux pairing), dissipation over
    the RANS dissipation length, implicit 2*K_v e-transport, and the
    model's named surface e source (the shear term of
    ``physics.sase_surface_e_source``, u*^3/(k*0.5*dz)): the discrete
    face production only sees shear at z >= dz, so the source stands in
    for the sub-half-cell surface-layer production u*^3/(k*z) exactly
    as it does in the driver -- without it the surface cell's e (and
    hence K_v) sits ~50% low, which is the model-configuration defect
    the source exists to fix.  S3-6h: the engine runs the COMPOSED
    RANS-limb lengths (:func:`bl89_rans_lengths` at the f = 0 blend
    limit, uniform theta) instead of the bare l_B, so the fixture
    witnesses the live formulation's kappa-z match (the hard
    constraint).  In a neutral column the BL89 lengths are pure
    geometry (l_down = z >= l_B, l_up = htop - z; zero integrand means
    no crossing can occur and l_les = l_B exactly with n2 absent), so
    they are CONSTANT in time and hoisted out of the loop; the min
    leaves l_B bitwise below ~htop - l_B and only the top few cells
    shorten.  The neutral fixed point is unchanged where mixing and
    dissipation ride the same length (production = dissipation gives
    e = C_E^{-2/3}*u*^2 independent of l -- the C_KV derivation);
    measured window shifts vs the pre-S3-6h engine: K_v < 0.8%
    (inside the fixture's 2% band -- the hard-constraint receipt).
    Fixed point (derivation at the C_KV constant):
    e = C_E^{-2/3}*u*^2 uniformly, K_v = u* * l_B(z),
    du/dz = (u*/(k*z))*(1 + k*z/lambda).  Returns the final profiles
    for band assertions.
    """
    z = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    theta = np.full(nz, 300.0)                 # neutral: any uniform value
    e = np.full(nz, E_MIN)
    # S3-6h composed RANS lengths -- constant in a neutral column
    # (docstring invariance), computed once; l_v is the f = 0 blend.
    l_mix_r, l_eps_r = bl89_rans_lengths(theta, e, z, dz, None, z0)
    u = np.zeros(nz)
    dm = dt * u_star**2 / dz                   # end-face momentum sources
    de_sfc = dt * u_star**3 / (KARMAN * 0.5 * dz)   # named surface source
    for _ in range(steps):
        e64 = np.maximum(e, E_MIN)
        kv = C_KV * l_mix_r * np.sqrt(e64)
        k_face = _face_average(kv)
        u[-1] += dm                            # influx through the top
        u[0] -= dm                             # surface drag
        u = implicit_vertical_diffusion(u, k_face, dt, dz=dz)
        p = _vertical_production(k_face, (u,), dz, dz)
        diss = C_E * e64**1.5 / l_eps_r
        e = np.maximum(e + dt * (p - diss), E_MIN)
        e[0] += de_sfc
        e = implicit_vertical_diffusion(e, 2.0 * _face_average(kv), dt,
                                        dz=dz)
    e64 = np.maximum(e, E_MIN)
    kv = C_KV * l_mix_r * np.sqrt(e64)
    k_face = _face_average(kv)
    flux = k_face * (u[1:] - u[:-1]) / dz      # interior-face stress
    return {"z": z, "u": u, "e": e, "kv": kv, "flux": flux}


#: Monin-Obukhov stable-limb similarity slopes of the surface-layer
#: closure :func:`column_stable_cooling` runs (Businger-Dyer form,
#: phi_m = 1 + BETA_M*zeta, phi_h = 1 + BETA_H*zeta, hence
#: psi = -BETA*zeta).  TRANSCRIBED, not chosen: these are the constants
#: the GEWEX Atmospheric Boundary Layer Study first case PRESCRIBES for
#: every participating model, so that the intercomparison scores the
#: turbulence closure and not the surface layer -- Beare et al. (2006,
#: Bound.-Layer Meteorol. 118:247-272, section 2) and the case
#: description carried with it, "kappa = 0.4, beta_m = 4.8,
#: beta_h = 7.8".  They are NOT this closure's own surface layer (the
#: model runs sfclay) and no live path reads them: they exist so the
#: benchmark engine can be presented with the benchmark's own lower
#: boundary condition.
SFC_SIMILARITY_BETA_M = 4.8
SFC_SIMILARITY_BETA_H = 7.8


def _mo_surface_stable(spd1, theta1, theta_s, z1, z0, z0h=None,
                       iters=20):
    """Monin-Obukhov surface layer on the stable Businger-Dyer limb.

    Returns ``(u_star, theta_star, obukhov_L)`` for a prescribed
    surface potential temperature ``theta_s`` under the level-1 wind
    and temperature, iterating

        u*     = kappa*spd1/(ln(z1/z0)  - psi_m(z1/L) + psi_m(z0/L))
        theta* = kappa*(theta1 - theta_s)
                 /(ln(z1/z0h) - psi_h(z1/L) + psi_h(z0h/L))
        L      = u*^2*theta1/(kappa*G_ACCEL*theta*)

    with psi_m = -BETA_M*zeta and psi_h = -BETA_H*zeta (the linear
    stable limb; SFC_SIMILARITY_BETA_M/_H carry the provenance).  The
    neutral values seed the iteration and an unstable or vanishing
    theta* returns the neutral pair, so the routine is total.  Stable
    similarity is a contraction here for the depths this engine runs;
    ``iters`` is fixed rather than tolerance-tested so the result is a
    deterministic function of its inputs.
    """
    z0h = z0 if z0h is None else z0h
    lnm, lnh = math.log(z1 / z0), math.log(z1 / z0h)
    spd1 = max(float(spd1), SFC_WSPD_FLOOR)
    dth = float(theta1) - float(theta_s)
    ust = KARMAN * spd1 / lnm
    tst = KARMAN * dth / lnh
    if dth <= 0.0:
        return ust, 0.0, math.inf
    for _ in range(iters):
        if tst <= 0.0 or ust <= 0.0:
            return KARMAN * spd1 / lnm, 0.0, math.inf
        lmo = ust * ust * float(theta1) / (KARMAN * G_ACCEL * tst)
        dm = lnm + SFC_SIMILARITY_BETA_M * (z1 - z0) / lmo
        dh = lnh + SFC_SIMILARITY_BETA_H * (z1 - z0h) / lmo
        if dm <= 0.0 or dh <= 0.0:                # decoupled: no solution
            return 0.0, 0.0, math.inf
        ust = KARMAN * spd1 / dm
        tst = KARMAN * dth / dh
    lmo = (ust * ust * float(theta1) / (KARMAN * G_ACCEL * tst)
           if tst > 0.0 else math.inf)
    return ust, tst, lmo


def column_stable_cooling(geo_u, geo_v, fcor, nz, dz, theta_init,
                          dt, steps, theta_surface, cooling_rate,
                          z0=0.1, e_init=None, stable_dissipation=False,
                          additive_dissipation=False):
    """Stably stratified column under a PRESCRIBED SURFACE COOLING RATE:
    the closure's single-column benchmark engine.

    The regime every other column engine here lacks.  ``column_ekman_
    balance`` runs the drag+PGF balance NEUTRALLY (uniform theta, no
    heat channel), so it cannot score a stable boundary layer at all;
    the jet and lake fixtures run REAL frames, which pin what this
    closure does on one case rather than what it does against a
    published reference.  This engine is the third thing: a fully
    specified idealized stable column whose answer is published, so the
    closure's stable limb can be SCORED and not merely compared with
    itself.  The forcing is deliberately spelled as arguments -- the
    benchmark's own numbers live with the benchmark
    (:mod:`woof.verify.cases.gabls1`), never here.

    THE COLUMN.  An f-plane with a prescribed constant geostrophic wind
    (``geo_u``, ``geo_v``), the same simultaneous explicit forcing
    ``column_ekman_balance`` uses,

        du += dt*fcor*(v - geo_v),   dv += -dt*fcor*(u - geo_u),

    over a prognostic potential temperature whose lower boundary is a
    prescribed surface value falling linearly at ``cooling_rate``
    [K s^-1].  Surface momentum and heat both come from the SAME
    Monin-Obukhov solve (:func:`_mo_surface_stable`) on the stable
    Businger-Dyer limb, so the surface layer is the benchmark's, not
    this closure's: the momentum row enters the implicit solve as the
    drag conductance c = u*^2/max(|V1|, SFC_WSPD_FLOOR) (the live S3-6j
    bottom boundary condition) and the heat row as the explicit
    bottom-cell deposit dt*u**theta*/thick_0 (SFC_SCALAR_FLUX, the
    live convention).

    THE INTERIOR IS THE LIVE RANS LIMB, evaluated through the same
    authority functions the split step calls at f = 0 -- there is no
    second formulation here:

        (l_mix_rans, l_eps_rans) = bl89_rans_lengths(...)
        K_v  = stable_limit_coefficient(l_mix_rans, e, N^2)
               * l_mix_rans * sqrt(e)
        K_h  = K_v/prandtl_blend(0.0)
        l_d  = dissipation_length(e, delta=1, n2, lb=l_eps_rans, f=0)
        eps  : the S3-6d ANALYTIC decay substep at the coefficient the
               two switches select, exactly as sase_split_step forms it

    with e advanced by shear production on the solved faces, buoyancy
    destruction -K_h*N^2, the analytic decay, and the implicit 2*K_v
    e-transport -- the split step's own order of operations, minus the
    horizontal channel a single column does not have.  ``delta`` is
    irrelevant at f = 0 (the geometric blend returns l_eps_rans
    FP-exactly) and the additive channel's reference length is
    l_B(z + z0) there, likewise independent of it.

    Returns the final profiles plus the surface series (u*, theta*, the
    surface potential temperature and the kinematic heat flux per
    step), which is what a benchmark scores.  The boundary-layer depth
    is NOT computed here: its definition belongs to the benchmark
    (there are several, and they disagree), so the caller gets the
    stress profile and applies its own.
    """
    nz = int(nz)
    dz = float(dz)
    z = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    theta = np.array(theta_init, dtype=np.float64)
    e = (np.full(nz, E_MIN) if e_init is None
         else np.maximum(np.asarray(e_init, dtype=np.float64), E_MIN))
    u = np.full(nz, float(geo_u))
    v = np.full(nz, float(geo_v))
    pr_t = prandtl_blend(0.0)
    series = {"ust": np.empty(steps), "tstar": np.empty(steps),
              "theta_s": np.empty(steps), "hfx_kin": np.empty(steps)}
    th_s = float(theta_surface)
    for n in range(steps):
        spd1 = float(np.hypot(u[0], v[0]))
        ust, tst, _ = _mo_surface_stable(spd1, theta[0], th_s, z[0], z0)
        series["ust"][n] = ust
        series["tstar"][n] = tst
        series["theta_s"][n] = th_s
        series["hfx_kin"][n] = -ust * tst        # w'theta' at the surface
        n2 = brunt_vaisala_n2(theta, dz)
        e64 = np.maximum(e, E_MIN)
        l_mix_r, l_eps_r = bl89_rans_lengths(theta, e64, z, dz, n2, z0)
        kv = (stable_limit_coefficient(l_mix_r, e64, n2) * l_mix_r
              * np.sqrt(e64))
        k_face = _face_average(kv)
        # momentum: PGF/Coriolis at the pre-update state, then the
        # implicit solve with the live drag row
        du = dt * fcor * (v - float(geo_v))
        dv = -dt * fcor * (u - float(geo_u))
        u, v = u + du, v + dv
        c_drag = ust * ust / max(spd1, SFC_WSPD_FLOOR)
        u = implicit_vertical_diffusion(u, k_face, dt, dz=dz,
                                        drag_bottom=c_drag)
        v = implicit_vertical_diffusion(v, k_face, dt, dz=dz,
                                        drag_bottom=c_drag)
        # heat: explicit surface deposit then the implicit K_h channel
        theta[0] += dt * (-ust * tst) / dz
        theta = implicit_vertical_diffusion(theta, k_face / pr_t, dt,
                                            dz=dz)
        # subgrid energy: the split step's own sequence at f = 0
        p_v = _vertical_production(k_face, (u, v), dz, dz)
        buoy = -(kv / pr_t) * n2
        ld = dissipation_length(e64, 1.0, n2, lb=l_eps_r, f=0.0)
        e_star = e + dt * (p_v + buoy)
        c_eps = (stable_dissipation_coefficient(ld, e64, n2, f=0.0)
                 if stable_dissipation else C_E)
        if additive_dissipation:
            l_ref = neutral_dissipation_length(z, 1.0, f=0.0, z0=z0)
            c_eps = additive_dissipation_coefficient(
                ld, l_ref, e64, n2, f=0.0, c_base=c_eps)
        b = c_eps * np.sqrt(np.maximum(e_star, E_MIN)) / (2.0 * ld)
        e = np.maximum(e_star / (1.0 + b * dt) ** 2, E_MIN)
        e = np.maximum(implicit_vertical_diffusion(
            e, 2.0 * k_face, dt, dz=dz), E_MIN)
        th_s -= dt * float(cooling_rate)
    e64 = np.maximum(e, E_MIN)
    n2 = brunt_vaisala_n2(theta, dz)
    l_mix_r, _ = bl89_rans_lengths(theta, e64, z, dz, n2, z0)
    kv = (stable_limit_coefficient(l_mix_r, e64, n2) * l_mix_r
          * np.sqrt(e64))
    k_face = _face_average(kv)
    return {"z": z, "u": u, "v": v, "theta": theta, "e": e, "kv": kv,
            "z_face": 0.5 * (z[1:] + z[:-1]),
            "uw": -k_face * (u[1:] - u[:-1]) / dz,
            "vw": -k_face * (v[1:] - v[:-1]) / dz,
            "wt": -(k_face / pr_t) * (theta[1:] - theta[:-1]) / dz,
            **{k: val for k, val in series.items()}}


def column_ekman_balance(geo_u, geo_v, fcor, nz, dz, dt, steps, z0=0.1,
                         drag=True):
    """Ekman/PGF balance column: THE S3-6j missing-force fixture engine.

    The fixture class whose ABSENCE hid the missing-friction bug: every
    prior column fixture (log layer, free convection, inversion, jet)
    ran without a large-scale pressure-gradient force, so a momentum
    path with no surface drag could still look equilibrated.  This
    engine adds the minimal, test-only per-column PGF+Coriolis tendency
    hook: a prescribed CONSTANT geostrophic wind (geo_u, geo_v) on an
    f-plane, applied as the simultaneous explicit tendency

        du += dt*fcor*(v - geo_v),   dv += -dt*fcor*(u - geo_u)

    (both evaluated at the pre-update state -- the decomposition
    PGF = (-f*vg, +f*ug) plus Coriolis (+f*v, -f*u), the standard
    Ekman-layer forcing), followed each step by the LIVE S3-6j
    momentum channel: neutral-similarity u* diagnosed from the
    level-1 wind,

        u* = KARMAN*max(|V1|, SFC_WSPD_FLOOR)/ln((z1 + z0)/z0)

    (the sfclay stand-in of the log-layer engine, now closing the drag
    loop), the named surface e source, the composed neutral RANS
    vertical channel (the log-layer engine's hoisted BL89 geometry),
    and the implicit u/v solves WITH the drag_bottom conductance
    c = u*^2/max(|V1|, SFC_WSPD_FLOOR) (``drag=True``) or with the
    pre-S3-6j zero-flux ends (``drag=False``, the RED limb).

    STEADY STATE (drag=True), derivations the fixture asserts:

    * DISCRETE COLUMN MOMENTUM BUDGET -- exact at the fixed point.
      The step map is u^{n+1} = M^{-1}(u^n + dt*f*(v^n - vg)) with
      M = I - dt*D_v + dt*(c/thick_0)*P_0; at a fixed point,
      (M - I)u = dt*f*(v - vg), and thickness-weighting telescopes
      the D_v part to its (zero) end fluxes, leaving

          c*u_1 = fcor*sum_k dz*(v_k - geo_v)
          c*v_1 = -fcor*sum_k dz*(u_k - geo_u)

      -- integrated PGF equals surface stress componentwise, to FP
      tolerance limited only by residual spin-up transients.
    * SURFACE WIND / u* -- Rossby-number similarity.  The geostrophic
      drag law (kappa*G/u*)^2 = (ln(u*/(f*z0)) - A)^2 + B^2 with the
      classical neutral constants A ~ 1.8, B ~ 4.5 gives, at
      G = 10 m/s, f = 1e-4 s^-1, z0 = 0.1 m: u* ~ 0.40 m/s, and the
      similarity inversion |V1| = (u*/kappa)*ln((z1 + z0)/z0) then
      places the level-1 (z1 = dz/2) wind; the fixture's GREEN band
      widens that point for closure spread (band derivation at the
      test).
    * CROSS-ISOBAR ANGLE -- the surface wind turns toward low
      pressure (+v for geo = (G, 0)); observed/theoretical neutral
      barotropic range ~10-35 degrees (Ekman 45 deg is the constant-K
      idealization; similarity layers sit shallower).

    RED limb (drag=False): from rest the column stays HORIZONTALLY
    uniform (the PGF tendency is z-independent and D_v of a uniform
    profile is zero), so the trajectory is the exact discrete inertial
    spiral Z^{n+1} = (1 - i*fcor*dt)*Z^n, Z = (u - ug) + i*(v - vg),
    |Z^0| = G: the speed sweeps 0 -> ~2G over half an inertial period
    and the forward-Euler factor |1 - i*f*dt| > 1 amplifies secularly
    -- NO equilibrium exists without the drag term.  That unbounded
    drift is the regression tripwire for the missing-force class.

    Starts from REST (u = v = 0, e = E_MIN).  Returns the final
    profiles, the diagnosed u*, the drag conductance actually used on
    the last step, and the per-step level-1 speed series.
    """
    z = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    theta = np.full(nz, 300.0)                 # neutral column
    e = np.full(nz, E_MIN)
    # Neutral BL89 geometry: constant in time (log-layer engine
    # invariance), hoisted; l_v is the f = 0 blend.
    l_mix_r, l_eps_r = bl89_rans_lengths(theta, e, z, dz, None, z0)
    u = np.zeros(nz)
    v = np.zeros(nz)
    lnz = np.log((z[0] + z0) / z0)
    spd1_series = np.empty(steps)
    ust = 0.0
    c_drag = 0.0
    for n in range(steps):
        spd1 = max(float(np.hypot(u[0], v[0])), SFC_WSPD_FLOOR)
        ust = KARMAN * spd1 / lnz              # neutral-similarity u*
        e[0] += dt * ust ** 3 / (KARMAN * 0.5 * dz)
        e64 = np.maximum(e, E_MIN)
        kv = C_KV * l_mix_r * np.sqrt(e64)
        k_face = _face_average(kv)
        # PGF + Coriolis hook: simultaneous explicit tendencies at the
        # pre-update state (docstring; the fixed-point budget identity
        # depends on the simultaneous evaluation).
        du = dt * fcor * (v - geo_v)
        dv = -dt * fcor * (u - geo_u)
        u = u + du
        v = v + dv
        c_drag = (ust * ust / spd1) if drag else None
        u = implicit_vertical_diffusion(u, k_face, dt, dz=dz,
                                        drag_bottom=c_drag)
        v = implicit_vertical_diffusion(v, k_face, dt, dz=dz,
                                        drag_bottom=c_drag)
        p = _vertical_production(k_face, (u, v), dz, dz)
        diss = C_E * e64 ** 1.5 / l_eps_r
        e = np.maximum(e + dt * (p - diss), E_MIN)
        e = implicit_vertical_diffusion(e, 2.0 * _face_average(kv), dt,
                                        dz=dz)
        spd1_series[n] = float(np.hypot(u[0], v[0]))
    return {"z": z, "u": u, "v": v, "e": e, "ust": ust,
            "drag_c": (c_drag if drag else 0.0),
            "spd1_series": spd1_series}
