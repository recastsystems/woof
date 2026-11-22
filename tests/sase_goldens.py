"""Pinned SASE dynamic-solve goldens shared by the CPU and GPU suites.

S3-3 review fold-in: the FP64 authority goldens for the frozen seed-3
varying-e real-lift fixture (``test_dynamic_solve_real_lift_golden_values``
in ``test_sase.py`` and the device golden gate in ``test_sase_gpu.py``)
used to live as duplicated literals in the two files; one ``GOLDEN_*``
constants block means a deliberate re-pin edits exactly one site and the
suites can never drift apart.  They deliberately do NOT live in
``woof/verify/sase_ref.py``: the authority module is frozen and pinned
evidence about it belongs with the tests.

The FP64 pair freezes the authority's six-row layout, the off-diagonal
weight-1 convention, the cond threshold, and the clip/recovery order.
The DEVICE pair is the measured FP32 result of the same fixture
(deterministic block reduction + host np sum), and it is a property of
the card as well as of the code: ``GOLDEN_DEVICE_BY_CARD`` holds one pair
per card, keyed by the device name cupy reports, and the device golden
gate in ``test_sase_gpu.py`` skips with the reason on a card with no
pair rather than failing for the card and not the code.  Any toolchain
change that shifts a card's digits must arrive as a deliberate re-pin,
mirroring the CPU golden-pin policy.

READING, 2026-09-18, a development machine's RTX 4090 (compute capability 8.9, driver
610.57.04, CUDA runtime 13020, cupy 14.2.0, NVRTC 13.3.33): the 5090 pair
below fails the rtol 1e-9 gate there in ``f`` by rel 1.481e-08 (abs
1.381e-08) while ``c_nu`` holds at rel 4.844e-10; the 4090's own pair,
identical across two processes and across the test's second in-process
call, is rel 1.845e-07 (c_nu) and 1.039e-08 (f) from the FP64 authority,
inside the 5e-4 real-lift gate like the 5090's.  Receipt:
tests/data/receipts/pin-gates/extra-pin-readings-221f66f14-run1.json.

RE-RECORDED for A146, 2026-10-01: the 5090 pair was the sm_120 build of
sase.cu, whose five constant divisions NVRTC compiled as reciprocal
multiplies.  With them spelled ``__fdiv_rn`` a development machine's RTX 5090 (driver
13.3, cupy 14.2.0, NVRTC 13.4.92) reads exactly the 4090's pair, rel
1.845e-07 (c_nu) and 1.039e-08 (f) from the FP64 authority, in two
processes and in each process's repeat call; a development machine's RTX 5070 Ti (driver
13.2, same NVRTC), which had no pair, reads the same pair the same way.
So the pair is no longer per card on any card recorded here; it stays
keyed by card because a card nobody has read can still differ.
"""

GOLDEN_C_NU_FP64 = 0.001735249587725131
GOLDEN_F_FP64 = 0.9325844743877104

#: Was 0.0017352499069869236 / 0.9325844702650841 (rel 1.84e-7 / 4.42e-9),
#: the 5090's pair before A146.
GOLDEN_C_NU_DEVICE = 0.0017352499078274765     # rel 1.845e-7 vs FP64
GOLDEN_F_DEVICE = 0.932584484076787            # rel 1.039e-8 vs FP64

#: (c_nu, f) as the device solve produces them on each card the pair has
#: been recorded on: the 4090's (2026-09-18) and, after A146, the 5090's
#: and the 5070 Ti's (2026-10-01), all three the same pair.
GOLDEN_DEVICE_BY_CARD = {
    "NVIDIA GeForce RTX 5090": (GOLDEN_C_NU_DEVICE, GOLDEN_F_DEVICE),
    "NVIDIA GeForce RTX 5070 Ti": (GOLDEN_C_NU_DEVICE, GOLDEN_F_DEVICE),
    "NVIDIA GeForce RTX 4090": (GOLDEN_C_NU_DEVICE, GOLDEN_F_DEVICE),
}

# ---------------------------------------------------------------------------
# S3-6b split-step trajectory goldens: the S3-6c device-mirror parity
# targets.  Two-step FP64 ``sase_split_step`` trajectories on the frozen
# seed-20260720 band-limited fixture (see
# ``test_split_step_trajectory_goldens`` for the exact construction):
# BOX = uniform clamped dz=200 m, dt=0.5; COL = model mode on the
# geometric 1.08 thickness column from 50 m, dt=0.05.  Field sums are
# FP64 ``np.sum`` (deterministic pairwise order); C_NU/F are the
# step-2 ledger values.  S3-6c gates the fused device split step
# against the same fixture and inherits these literals.
#
# RE-PINNED for S3-6d (analytic dissipation substep): replacing the
# explicit forward-Euler dissipation e - dt*C_E*e^{3/2}/l_d with the
# exact decay e*/(1 + b*dt)^2 shifts each step's e by O((b*dt)^2 * e)
# -- the Euler-vs-exact Taylor gap, O(dt^2) per step -- and the step-2
# solve/momentum channels inherit it through e^n.  Measured deliberate
# shift vs the S3-6b pins: BOX e 3.8e-6 rel (u/v/w < 1e-15 -- momentum
# feels e only through the step-2 stress); COL e 2.5e-7, u/v/w
# <= 1.3e-11; F 6.4e-6 (BOX) / 8.5e-8 (COL).  Same fixture, same
# construction, arithmetic change registered in the S3-6d adjudication.
#
# RE-PINNED for S3-6e (RANS-limit horizontal governor): the (1-f)
# horizontal branch becomes the audited 2-D Smagorinsky deformation
# diffusivity (authority ``governed_stress``) and its production share
# bypasses e to heat, so on this fixture (c_nu = 0, f ~ 0.945: the
# (1-f) branch IS the live horizontal channel) the e trajectory and,
# through e^n, the step-2 solve shift deliberately.  Measured shift vs
# the S3-6d pins: BOX e 1.03e-5 rel, u/v/w <= 1.3e-15, F 1.8e-6; COL
# e 1.03e-6, u/v/w <= 1.5e-11, F 1.5e-6.  Same fixture, same
# construction; formulation change registered in the S3-6e
# adjudication (smoke-gate G2, ledger 2026-07-20).  The taper does not
# touch these goldens (no zdamp on the fixture).
#
# RE-PINNED for S3-6f (partition cap + w-based resolved-fraction
# bound): the step's f is now min(f_solved, f_cap(delta/z_i), f_w) and
# the pinned F is the value the step USED.  On this fixture the
# w-based bound BINDS (band-limited w against e ~ 0.05: f_w ~ 0.295 <
# f_solved ~ 0.945 -- the fixture's horizontal-velocity variance had
# been rated "resolved" while its w variance supports far less, which
# is exactly the S3-6f concession); the COL mode's shallow stretched
# column additionally engages the cap (z_i ~ 194 m, delta = 500 m ->
# f_cap ~ 0.54, still above the binding f_w).  Measured shift vs the
# S3-6e pins: BOX e 2.9e-4 rel, u/v/w <= 1e-15; COL e 3.7e-5, u/v/w
# <= 3e-12 (c_nu = 0 on this fixture, so momentum feels f only
# through the (1-f)*K_smag weight).  Same fixture, same construction;
# formulation change registered in the S3-6f adjudication (mesoscale
# sensing concession, ledger 2026-07-20).
#
# RE-PINNED for S3-6g (regime-consistent Prandtl number): the split
# step's buoyancy K_h becomes K_v/Pr_t(f) with Pr_t(f) = f*PR_LES +
# (1-f)*PR_RANS at the step's used f (~0.295 here), so Pr_t ~ 0.70
# replaces the retired fixed 1/3 -- the buoyancy channel WEAKENS ~2.1x
# on this live-theta fixture and the e trajectory (and, through e^n
# and the theta-gradient coupling, the step-2 solve) shifts
# deliberately.  Measured shift vs the S3-6f pins: BOX e 2.34e-3 rel,
# F 1.0e-3, u/v/w <= 8e-16 (momentum feels Pr not at all in step 1 --
# the shift enters only through step-2 e^n); COL e 4.3e-4, F 1.6e-4,
# u/v/w <= 4e-8.  Same fixture, same construction; formulation change
# registered in the S3-6g adjudication (smoke-c PR_T diagnosis,
# ledger 2026-07-20 ~23:0x).  C_MOM_BG (the renamed momentum
# background) is bit-identical to the former C_K/PR_T, contributing
# ZERO shift -- the whole delta is the buoyancy Prandtl blend.
#
# RE-PINNED for S3-6h (BL89 displacement lengths in the RANS limb):
# the vertical channel becomes l_v = f*min(l_B, l_s) +
# (1-f)*min(l_B, l_s, l_mix_BL89) and the l_d blend's RANS limb
# min(l_B, l_eps_BL89) (authority bl89_rans_lengths) -- on this
# live-theta fixture (f ~ 0.294, mixed-sign n2, e ~ 0.05-0.15) the
# BL89 bounds bind in a subset of cells, shifting K_v/l_d there and,
# through e^n, the step-2 solve.  Measured shift vs the S3-6g pins:
# BOX e 2.97e-3 rel, F 1.0e-3, u BITWISE-unchanged and v/w <= 4e-16
# (the RANS-side channel touches momentum only through the implicit
# K_v solve at step 1's unchanged e^n -- the step-1 K_v shift is
# sub-ulp on these sums -- and through step-2 e^n in the stress);
# COL e 5.1e-4, F 1.4e-4, u/v/w <= 1.1e-5 (the stretched column's
# shallow geometry engages the BL89 bounds harder at step 1).  Same
# fixture, same construction; formulation change registered in the
# S3-6h adjudication (jet-coupling diagnosis, ratified option 2,
# ledger 2026-07-21 ~01:0x).  The f = 1 LES limb is pinned bitwise
# elsewhere (test_split_step_les_limb_f1_bitwise_under_bl89): these
# RANS-side shifts are the expected and registered signature.
#
# RE-PINNED for S3-6j (surface momentum stress in the vertical solve)
# -- THE MISSING-FRICTION FIX, and its ONE INTENTIONAL CROSS-LIMB
# CHANGE (the drag applies at ALL f; flagged prominently per the
# S3-6j adjudication): the golden fixture now passes a uniform
# ust = 0.3 m/s field (constructed with np.full AFTER the rng draws --
# the frozen seed-20260720 field sequence is untouched), so the u/v
# Thomas solves carry the implicit surface-stress bottom row
# c = ust^2/max(|V1^n|, SFC_WSPD_FLOOR) folded into diag_0 (authority
# module docstring, S3-6j section; the YSU npref.py:6495-6497
# linearization).  Momentum SHIFTS BY DESIGN -- that is the entire
# point of the amendment: measured shift vs the S3-6i pins BOX
# u 1.56e-3, v 7.66e-4 rel (the drag decelerating the bottom row),
# w 8.0e-16 and e 7.7e-8 (step-2 trajectory coupling through P_v and
# the stress at the shifted e^n), F 1.8e-8; COL u 6.23e-4, v 3.06e-4,
# w 3.4e-13, e 2.1e-8, F 4.8e-9.  The BOX split-ledger residual with
# drag engaged closes to 4.4e-15 of scale under the restated
# BOUNDARY-CONSISTENT closure dKE + dE + dHeat - dKE_sfc (measured
# dKE_sfc = -1.58e-2, dE_sfc_src = +8.64e-2, sfc_conv_resid diagnosed
# open).  Same seeds, same construction otherwise; formulation change
# registered in the S3-6j adjudication (Probe-4 whole-column-
# acceleration falsification, ledger 2026-07-21).  ust=None remains
# bitwise the S3-6i step (pinned by the f = 1 LES-limb test's
# no-drag leg).
#
# RE-PINNED for S3-6i (decoupled stable-limit diffusivity coefficient
# C_KS): the vertical channel's RANS limb becomes
# C_r*l_mix_rans*sqrt(e) with C_r = C_KV + (C_KS/LS_COEF - C_KV)*
# rho^2, rho = min(l_mix_rans/l_s, 1) (authority
# stable_limit_coefficient), and the l_v length blend is restated as
# the equivalent two-product K_v blend (RANS-only coefficient change;
# f = 1 FP-exact as before).  On this fixture (f ~ 0.294, mixed-sign
# n2) the stable cells' K_v drops toward C_KS*e/N where l_s binds, so
# the e trajectory and, through e^n, the step-2 solve shift
# deliberately; the neutral-cell K-blend restatement contributes only
# roundoff-scale association noise.  Measured shift vs the S3-6h
# pins: BOX e 8.68e-4 rel, F 3.56e-4, u/v/w <= 1.3e-15 (association
# roundoff only -- momentum feels the coefficient through the step-1
# implicit solve, sub-ulp on these sums); COL e 2.44e-4, F 1.09e-4,
# u/v/w <= 5.1e-6 (the shallow stretched column engages the stable
# cells harder at step 1).  Same fixture, same construction;
# formulation change registered in the S3-6i adjudication
# (stable-limit coefficient decoupling, ledger 2026-07-21 ~01:5x);
# C_KS = 0.25 calibrated at the promoted jet-decoupling fixture.
#
# RE-PINNED for S3-9c (GUSTINESS-CORRECTED surface drag; the
# S3-6h/6i/6j adversarial review IMPORTANT-1, task S3-9c): the drag
# conductance gains the audited YSU factor (spd1/max(wspd, 1e-9))^2
# (authority module docstring, S3-9c section; npref.py:6495-6496),
# and the golden fixture now ALSO passes the gust-enhanced speed
# field wspd = max(sqrt(u1^2 + v1^2 + SPLIT_GOLDEN_GUST^2),
# SFC_WSPD_FLOOR) built from the frozen INITIAL level-1 winds
# (constructed AFTER the rng draws beside ust -- the S3-6j idiom --
# and held fixed across both steps exactly like ust; the driver
# refreshes both from sfclay each due step, the fixture freezes
# both).  SPLIT_GOLDEN_GUST = 0.5 m/s is comparable to the
# band-limited resolved speeds, so the factor lands mostly in
# (0.05, 0.95) and the fixture's sub-floor columns (|V1| <
# SFC_WSPD_FLOOR) exercise the calm-gusty branch (factor
# (0.1/wspd)^2 ~ 0.04 -- the over-damping class the correction
# removes).  Momentum shifts BY DESIGN through the weakened drag
# row: measured shift vs the S3-9 pins BOX u 1.19e-3, v 5.86e-4 rel
# (drag work dKE_sfc -1.583e-2 -> -5.054e-3, the correction backing
# the drag off ~3.1x on this gusty fixture), w 4.0e-16 and
# e 5.08e-8 (step-2 coupling through the solved winds), F 1.2e-8;
# COL u 4.75e-4, v 2.34e-4, w 2.3e-13, e 1.35e-8, F 3.2e-9
# (dKE_sfc -6.340e-3 -> -2.023e-3).  The wspd_sfc=None limb is
# BITWISE the S3-9 step (measured drift exactly 0.0 on both modes --
# the no-gustiness identity; the identity pin test asserts the
# supplied-no-gust case bitwise too).  Same seeds, same construction
# otherwise; tolerances untouched; formulation change registered in
# the S3-9c task (adversarial review fix; report
# docs/internal/sdd/task-s3-9c-report.md).
#
# RE-PINNED for S3-9 (geometric dissipation-length blend; F-Y1 Lake
# Michigan over-coupling): the l_d regime blend of the authority's
# ``dissipation_length`` becomes delta**f * l_eps_rans**(1-f) (was the
# linear f*delta + (1-f)*l_eps_rans; authority module docstring, S3-9
# section).  DERIVATION OF THE SHIFT: this fixture runs at INTERIOR
# f ~ 0.294 in both modes, so the pinned trajectories ride the blend
# between its endpoints -- the linear form floored l_d at f*delta =
# 147 m regardless of the RANS composition, while the geometric form
# stays within (delta/l_eps_rans)**f of it (e.g. l_eps_rans ~ 32 m at
# z ~ 100 m gives l_d ~ 71 m vs the linear 169 m), so dissipation
# STRENGTHENS, e falls, and through e^n the step-2 solve and stress
# shift with it; f rises slightly because the lower e_mean tightens
# the w-sensor bound f_w = 1 - e/(e + E_r_w).  Measured shift vs the
# S3-6j pins: BOX e 7.57e-4 rel, F 2.68e-4, u 3.7e-11, v 2.2e-11,
# w 4e-16; COL e 1.81e-4, F 6.4e-5, u/v/w <= 1.2e-9 (momentum feels
# l_d only through the step-2 stress/implicit solve at the shifted
# e^n).  Same seeds, same construction; formulation change registered
# in the S3-9 adjudication (docs/internal/sdd/yolo-lake-mechanism.md,
# controller ledger 2026-07-21).  The f = 0 and f = 1 endpoint
# arithmetic is BITWISE unchanged (pinned by
# test_dissipation_length_blend_les_and_rans_limits and every f = 1
# reduction test); the DEVICE mirror re-pins in S3-9b -- until then
# the GPU parity gates mismatch these literals BY DESIGN.
# ---------------------------------------------------------------------------

SPLIT_BOX_SUM_U = 19.283104304548182
SPLIT_BOX_SUM_V = -30.510957506475044
SPLIT_BOX_SUM_W = -8.869777026347748
SPLIT_BOX_SUM_E = 99.09186549754612
SPLIT_BOX_C_NU = 0.0
SPLIT_BOX_F = 0.2945709787233337

SPLIT_COL_SUM_U = 19.279134181991093
SPLIT_COL_SUM_V = -30.51376465510443
SPLIT_COL_SUM_W = -8.869629646489187
SPLIT_COL_SUM_E = 99.33572373692178
SPLIT_COL_C_NU = 0.0
SPLIT_COL_F = 0.29435385722235063

#: S3-6j: the uniform friction-velocity field of the golden fixture
#: (np.full AFTER the rng draws; both suites construct it from this
#: one literal so they cannot drift).
SPLIT_GOLDEN_UST = 0.3
#: S3-9c: the uniform gust augmentation [m/s] of the golden fixture's
#: sfclay-convention enhanced speed wspd = max(sqrt(u1^2 + v1^2 +
#: SPLIT_GOLDEN_GUST^2), SFC_WSPD_FLOOR), built from the frozen
#: initial level-1 winds AFTER the rng draws (derivation comment
#: above; both suites construct it from this one literal).
SPLIT_GOLDEN_GUST = 0.5
