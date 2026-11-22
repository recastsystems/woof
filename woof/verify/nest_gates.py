"""Pre-registered Phase-5 nesting milestone gate tables (N0-N6 + Phase-5b).

GATES-FIRST discipline (design decision D7; architecture doc section F):
every milestone-ladder gate -- definition AND numeric threshold -- is
registered here BEFORE any ARC-B implementing lane merges, and the rung
executors (Task 8(c) cases, Task 14) consume ONLY these records.  Numeric
thresholds are verbatim from the ratified architecture document

    docs/internal/specs/2026-07-16-phase5-nesting-architecture.md

section F ("Milestone ladder") AS AMENDED (the doc's F-numbered amendment
record is the ratification trail; F15 ratifies the external-review rows
below); each record's ``anchor`` names its rung
paragraph.  A threshold may be loosened ONLY by controller plan amendment
with the N1.5 attribution run in hand -- never against "physics differ"
hand-waving.  ``tests/test_nest_gates.py`` pins every number and the full
table inventory, so any edit to this ledger fails the suite and forces the
amendment to be visible in the diff.

Two thresholds required interpretation, both flagged in-record and both
RATIFIED by the 2026-07-16 controller amendment (post red-team):

- N1.5 registers an "~1e-6 band"; the pre-registered numeric value is
  ``1.0e-6`` (FP32-relative, order-of-ops differences only).
- N2(c) names a "boundary_zone_blowup-style w gate"; the N3 family value
  ``0.5`` is the ratified bound (architecture section F N2(c) as amended;
  plan Task 8(a)).

AMENDED 2026-07-16 (controller amendment after red-team verification):

- F10: executable comparator semantics per bound kind (:data:`COMPARATORS`);
  every kind BLOCKS except ``diagnostic``.
- F11: ``alloc_estimate_le_wddm_budget`` at N0 and the N5 re-check --
  the full enforced chain is measured <= estimate <= measured budget.
- F12: N6 blocking hard-gate records (completion/zero validator fires,
  output inventory, peak <= estimate, abort-threshold compliance).
- F13: the CLOSING milestone (Phase-5 closing gate, executed this phase)
  is distinct from P5B; feedback=0 bit-inertness is proven by a runtime
  FEEDBACK-call-path bypass A/B, not a pre/post-tests-only-merge diff.
- F8/F9: schedule-pin and N1.5 field-list conventions updated per the
  amended architecture (terminal feedback guard; moist AND scalar tables).

AMENDED 2026-07-16 (F15, external-review adjudication, controller-ratified
in the architecture doc's amendment record): comparator kind split
(``fp64_mirror_floor``/``fp32_state_equiv``/``real_emulation_mirror``) and
eleven additive blocking records (N5S matched-physics shadow, N5B
boundary-location invariance, ancestor inertness at every output,
validator invocation/frame counts, d04 budget/clipping residuals, N6
high-frequency time-max products, quantitative refl/updraft companions)
plus required evidence-manifest fields for N5/N6 blockers.  Additive or
strictly tightening only; the 73-row base ledger is byte-identical (proven
by the amendment's adversarial review, p5ext-gates-review.md).

All child dynamics gates inherit the ratified physics substitutions (the
reference ran ISHMAEL mp=55 + Shin-Hong bl_pbl=11; woof runs
Morrison+YSU), which is WHY the child gates are statistical while every
parent-protection gate is bitwise.

AMENDED 2026-07-16 (F17, controller amendment WITH the N1.5 attribution
run in hand -- the register's sanctioned loosening path, executed in
full): the N1/N3 blended HGT/MUB static thresholds move 0.5 m -> 1.5 m
and 10 Pa -> 17 Pa.  Evidence chain (out/n1-measure/attribution.json +
the ledger): (a) CHAIN ISOLATION IS BIT-EXACT -- pushing the bundle's
own inputs (geo_em child HGT_M + reference parent wrfout HGT) through
woof's GPU SINT+blend chain reproduces the reference blended child HGT
with max |diff| = 0.0 on the FULL GRID for d02/d03/d04, so the measured
miss is 100% input-attributed; (b) the input is the RATIFIED Phase-3
static terrain residual (unblended-vs-geo_em fine residual 2.918 m d02 /
2.166 m d03 / 0.175 m d04, cascading through the parent chain); (c) the
MUB miss is the HGT miss propagated hydrostatically (measured slope
-11.1..-11.5 Pa/m = -rho*g, correct sign).  New values = measured worst
(1.185 m / 13.17 Pa) + ~25% headroom; 17 Pa = 1.5 m x 11.4 Pa/m; the
input-residual ceiling (2.92 m) bounds any possible leak.  The
loosening is paired with three ADDITIVE TIGHTENING records
(``d0X_chain_isolation_bit_exact``): the machinery itself is now pinned
at bit-exactness, infinitely sharper than the numeric bound it
replaces as the chain-error detector.

AMENDED 2026-07-16 (F18, same attribution run): the N1.5 tendency
comparator treats the four ``state.t.*.tendency`` tables at the FP32
successor scale E = fl32(VALUE + fl32(cdt*TENDENCY)) -- exactly the
expression WRF's boundary consumer applies -- under the UNCHANGED 1e-6
band.  This is the registered treatment of the PROVENANCE "N1.5 harness
seam -- dumped-WRF theta restoration" (harness-only inverse
representation map, production-absent; empirically attributed by the
p5n15 numerics shadow: substituting direct t_2 coupling gives 882,000/
882,000 bit-identical theta elements).  All other tables keep the raw
registered comparator -- the u/v low-face misses the same run exposed
were a REAL production defect, FIXED (p5n15 80eebb4) to full bit
identity, and MUST NOT be absorbed by any tolerance.  A new additive
record pins the producer's stored-recurrence hard check.

AMENDED 2026-07-16 (F19, N2 identity-rung attribution -- p5n2attr
shadow report, adversarially derived + measured): the N2(a)/N2(b)
records re-kind fp32_state_equiv -> identity_oracle.  The
pre-registered post-t=0 full-state 8-ULP equivalence compared two
INTENTIONALLY DIFFERENT equations (periodic parent vs Davies child;
the specified operator branch reassociates FP32 arithmetic tilewide
from the first flux evaluation), so no correct implementation could
pass it; the measured divergence is physically negligible (t=0
bit-exact; boundary-frame <= 5e-3 relative at t=60) and every defect
hypothesis (w substeps, orientation/corners, tendency denominator,
the interior-first mup signal) was ruled out at bit level.  The
replacement composite oracle is STRICTLY SHARPER where it binds:
bit-exact force tables (measured 0.00 field-scale ULP), successor
reconstruction <= 2 field-scale ULP (measured 0.125), an independent
Davies-formula oracle, nested-w substep bit identity (measured 7/7
exact), and empirical raw specified-row caps at 2x run envelope.
Not a loosening: a correction of what the rung compares, paired with
oracles that catch every fault class the old comparison targeted,
loudly.

AMENDED 2026-07-17 (F20, first N3 hardware attribution -- controller
bisection to a one-line cause): the N3(i) d01 byte-ratchet text now
names its ratified exceptions explicitly and records the root-dtbc
adjudication.  The first N3 run failed d01-vs-Phase-4 with a whole-
domain 13:00 divergence; controller bisection proved t=0 byte-identity,
nested-vs-control byte-identity (the non-mutating-coupling property the
gate protects HOLDS), production-path trunk cleanliness (1-hour cardinal
re-run byte-identical, with and without mid-run checkpoint writes), and
step-1 divergence confined to the Davies relaxation zone with byte-
identical step inputs.  Cause: the tree build bound the root's external-
boundary clock, switching root Davies launches from the frozen Phase-4
elapsed-based dtbc to WRF's post-increment dtbc_launch_fp32 (one dt of
boundary target time).  Adjudication: root stays UNBOUND (frozen-bytes
authority); the WRF post-increment recurrence remains child-only where
the N1.5 oracle pins it; the root-side WRF deviation is registered in
PROVENANCE.md.  TITLE joins the ratified exception list as descriptive
metadata (value excluded, presence required, both values recorded in
evidence).  No numeric threshold changes; the byte comparator is
otherwise untouched.

AMENDED 2026-07-28 (Davies clock bind -- the F20 adjudication is
RETIRED): the production tree build now binds the root's external
boundary clock, so root Davies launches consume WRF's post-increment
dtbc (solve_em.F:371-372) and the bound final ring owns the OLD record
at dtbc=T_bdy on pre-seam steps (solve_em.F:4531-4639).  The frozen
Phase-4/-r2 anchor bytes encode the retired elapsed-based semantics and
can no longer serve as the byte authority; the d01/d02/d03 invariance
ratchets regenerate against the seam-closure anchor epoch
(reference-t7-final-r3) in the batched end-of-Wave-1 regeneration.  The
ratchet metric NAMES keep their historic d01_bitwise_vs_phase4_13z
identity; what they protect is unchanged (gpuwm-invariance --
non-mutating coupling -- not WRF identity).  Restart headers gained the
root_external_lbc_clock semantic identity so pre-bind checkpoints fail
closed (woof/io/restart.py).

AMENDED 2026-07-17 (F21, N3 FSS matched-physics attribution -- the
registered attribution path executed in full): the child REFL_10CM FSS
rows now score against a MATCHED-PHYSICS WRF reference where one is
registered in MATCHED_REFERENCE_FRAMES, at the UNCHANGED thresholds
(0.90/0.80/0.70).  Basis, three-way d02 comparison at 13:15Z with the
registered formula (out/rungs/N3-fss-attribution.json): woof vs
matched-WRF (Morrison mp=10 + YSU both sides) = 0.9818/0.9830/0.9734;
matched-WRF vs the mp55 bundle reference = 0.3935/0.0787/0.0050; woof
vs mp55 = 0.4114/0.0830/0.0073.  Real WRF under the woof physics suite
scores the same as woof against the mp55 reference, so THIS registered
d02/13:15 FSS miss at these thresholds and scales is attributable to
physics selection (the claim is scoped to this metric: it does not
assert absence of model error elsewhere), and like-for-like agreement
clears every threshold.  The matched reference is the node-generated bundle
gpuwm-wrf-matched-mp10-ysu-19740403-v1 (WRF v4.6.1 dmpar 48-rank,
mp=10/bl=1/sfclay=91/use_theta_m=0/do_radar_ref=1; deviations on
record: one-way nesting feedback=0 and MPI decomposition, both absorbed
by the 0.97+ like-for-like scores).  Frame consumption is SHA-256
pinned.  All other statistical rows (MSLP correlation, T500/T850 RMSE,
blowup, structure) keep the mp55 references untouched; domains without
a registered matched reference (d03/d04) keep the mp55 FSS reference
and BLOCK until a matched reference is registered -- extending the
matched run to max_dom=4 is the registered path.  Not a loosening:
thresholds, formula, mask, and family are unchanged; only the FSS
reference identity is corrected to like-for-like physics.

AMENDED 2026-07-17 (F22, controller geometry adjudication before any N5B
execution): the N5B shrink is 51 cells per side (498x498), not the
originally drafted 50 (500x500).  Exact 50-per-side is unrepresentable:
the d04 anchor lives on the ratio-3 parent lattice, and 50 is not
divisible by 3, so no integer i/j_parent_start can make the two centered
400x400 cores coincide (the machinery's coordinate-identity preflight
correctly refused to run).  51/side is the nearest symmetric realizable
shrink: anchor shift 51/3 = 17 parent cells exactly (151 -> 168), the
shrink stays parent-aligned (498 = 3 x 166, matching WRF's e_we-1
divisibility rule that 500 would violate), and the centered cores --
production [100:500]^2, shrink [49:449]^2 -- coincide cell-for-cell,
enforced by the pre-existing XLAT/XLONG SHA-equality preflight.  Not a
loosening: every predicate, threshold, and the pre-look envelope
procedure are untouched; only the B-geometry two numbers changed, and
they changed BEFORE any A/B run or metric was produced.

AMENDED 2026-07-17 (F23, first N4 execution -- canonical digest scope):
the rung-evidence canonical state digest gains a TRAJECTORY scope that
excludes CHILD_DUTY_SCRATCH_MEMBERS (currently the coupler's
scratch/nest_parent_field staging slot), with the scope baked into the
digest seed so scopes can never be compared silently.  Basis: N4's
d02_bitwise_vs_n3 gate failed on state hashes while every d02 wrfout
frame was byte-identical to the N3 ratchet; a member-wise probe showed
exactly 1 of 288 canonical members differing -- the parent-duty staging
slot, zero-filled when d02 has no child and working memory when it
forces d03.  Its contents are a pure function of parent duty, rebuilt
every FORCE and rebuilt on restart, and physically inert to the owning
domain (287/288 members plus all output bytes identical).  Cross-shape
comparisons (ratchet state evidence, ancestor inertness, restart-split
sampling) use the trajectory scope; the full scope remains available
and every other member keeps full coverage.  Not a loosening: the
excluded member is enumerated, evidenced, and definitionally
shape-dependent bookkeeping.

AMENDED 2026-07-17 (F24, matched-reference completion and pre-convective
adjudication): MATCHED_REFERENCE_FRAMES now registers bundles per domain.
d02 retains the v1 frame pin (SHA-256 65fc6f52205c14cdf618ddf19520e8ae541e
c74a9188de45524f9c12e2fb04aa, 333839612 bytes); d03 and d04 use
gpuwm-wrf-matched-mp10-ysu-19740403-v2 (same build/physics, max_dom=4) at
13:15Z, pinned respectively to
SHA-256 4382b72ed3fa76662030f92e91ac97b3229e771b8dfce1b2351b58bd4ad25754
(383352328 bytes) and be32aa0b1b3b8b79e2d9f5748c8becc6fdd57e579185cdd819131388306622d6
(488452790 bytes).  An FSS family row is meteorologically degenerate when
either candidate or matched-reference event coverage on the registered
interior mask is below FSS_DEGENERATE_EVENT_FLOOR (1e-4); it is a
PROVISIONAL pass held for the registered ensemble-envelope adjudicator,
never a mechanical model failure at this valid time.  The adjudicator uses
at least three same-domain 13:15 ensemble members, records all pairwise FSS
values and their min/max envelope, and confirms degeneracy only when a
majority of members are themselves below the coverage floor.  Otherwise it
marks the held row revoked, and that revocation MUST re-open the consuming
rung.  Run provenance: the previously pinned serial wrf.exe (sha
f0fb585b...) cannot initialize the 601x601x50 d04 (RSL layer sizes a
broadcast buffer as signed 32-bit; ~2.23 GB request overflows) and is retired
for 4-domain work; matched v2 and the ensemble use the registered dmpar build
sha bda5e5e69754b5f5c7eb700ff20bb0e2bb65145f4e1beeef492e9b10edb586e5.

AMENDED 2026-07-18 (F27, ratified by the project owner after F25
adjudication: "yeah amend lets move forward"): the d03/d04 REFL_10CM FSS
20 dBZ rows convert from blocking to DOCUMENTED-DEFICIENCY records.  Basis:
four ensemble members (unperturbed, member-00/01/02 of
gpuwm-n5s-ensemble-v1) each score FSS20 0.8558 vs the matched v2 reference
at d03 13:15 (point envelope; one-ULP twins, zero spread); woof scores
0.7047-0.7084 (stable across runs), below the envelope minimum, a genuine
deficiency; mechanism identified as the residual ~0.87 hPa acoustic ringing
(mslp-ringing report; stabilization campaign staged). The registered 0.90
minimum is simultaneously established as miscalibrated (members miss it
too); the row's calibrated reference bar is the envelope minimum 0.8558.
When a future N4/N5 evaluation measures the row at or above the envelope
minimum 0.8558, the documented-deficiency conversion self-revokes and the
row returns to blocking at 0.8558.

AMENDED 2026-07-18 (F28, ratified by the project owner after the N5S
four-member scoring: "btw yes amend"): all three CPU twin-pairs scored
exactly 0.0 on all 61 registered metrics.  The one-ULP twins were output-
identical at the registered 15-min cadence throughout the 75 pre-convective
minutes, so every measured E95 envelope is degenerate-zero.  A metric's
envelope is DEGENERATE exactly when E95 == 0.  For any such row,
N5S_matched_physics_wrf_shadow converts from blocking to
DOCUMENTED-EVIDENCE under adjudication ``f28-degenerate-envelope``; the
measured GPU distance and zero envelope remain in the full evidence record
but the row does not fail the compound verdict.  RE-BINDING is automatic
and row-wise: metrics whose E95 > 0 remain binding under the registered
gpu_distance <= E95 comparator, and a future ensemble with non-degenerate
envelopes -- the staged Phase-6 convective-window extension -- restores full
binding force row-by-row.  No thresholds are altered; nothing is deleted.
The FP32-format-precision floor alternative was considered and rejected
because format floors (e.g. ~0.008 Pa for MU vs measured implementation-scale
differences of 1-2 hPa) are orders of magnitude below
implementation-difference scale: it would convert the degenerate gate into
a differently-unpassable one, not an accurate bound.

CORRECTED 2026-09-03 (gate-integrity defect fix ver-05-01/nst-04-02; NOT an
owner amendment -- no threshold, no envelope and no adjudication label
changes): F28 as written above made a degenerate row report ``passed: true``,
and with every one of the 61 measured envelopes degenerate-zero the compound
N5S row accepted ANY gpuwm-vs-WRF distance.  "The twins never separated" is
the absence of a yardstick, not evidence that the candidate is inside one.
A degenerate row is therefore INCOMPLETE: it is neither accepted nor failed,
it carries ``status: "incomplete"`` beside its unchanged
``f28-degenerate-envelope`` adjudication and its measured distance, and it
stops the compound verdict reading PASS -- ``verdict`` is "fail" if any
non-degenerate row missed, else "incomplete" if any row was degenerate, else
"pass", the same three-valued reduction
:func:`woof.verify.spectral_receipt.evaluate_gates` already uses.  F28's
row-wise RE-BINDING is untouched: a future ensemble whose E95 > 0 scores
those rows under the original ``gpu_distance <= E95`` comparator and the
compound returns to PASS.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


#: The ratified architecture document all anchors point into.
ARCHITECTURE_DOC = ("docs/internal/specs/"
                    "2026-07-16-phase5-nesting-architecture.md")
#: The plan that mandates this table (Task 8 stage (a)).
PLAN_DOC = "docs/internal/plans/2026-07-16-gpuwm-phase5.md"

#: Ladder order.  "CLOSING" is the Phase-5 closing milestone (two-way
#: dormant closure, Task 16 -- executed THIS phase); "P5B" is the dormant
#: two-way (feedback) activation set registered now per design D4 and
#: executed in the Phase-5b follow-on (F13 amendment: the two are
#: distinct -- CLOSING must not sit in a milestone that never runs).
MILESTONES = ("N0", "N1", "N1.5", "N2", "N3", "N4", "N5", "N6",
              "CLOSING", "P5B")

#: Bound-kind vocabulary.  Numeric kinds carry a float ``threshold``; all
#: other kinds carry ``threshold=None`` (the pass criterion is identity,
#: a runtime-measured bound, or an adjudicated verdict -- never a number
#: invented outside this ledger).
BOUND_KINDS = frozenset({
    "max",             # pass iff metric <= threshold
    "min",             # pass iff metric >= threshold
    "strict_max",      # pass iff metric <  threshold
    "bitwise",         # bit/byte identity vs the named baseline
    "exact",           # exact discrete equality (schedules, ledgers, pins)
    "fp64_mirror_floor",       # FP32 output vs rounded FP64 mirror
    "fp32_state_equiv",        # peer FP32 states + distribution guards
    "identity_oracle",         # F19 composite identity-nest oracle
    "real_emulation_mirror",   # FP32 output vs per-op WRF REAL mirror
    "measured_bound",  # runtime-measured relational bound (memory)
    "structural",      # qualitative presence/verdict, adjudicated
    "diagnostic",      # recorded and reviewed, non-blocking
})
#: The kinds that require a numeric threshold.
NUMERIC_KINDS = frozenset({"max", "min", "strict_max"})

#: Comparator amendment (external-review adjudication): the old overloaded
#: ``fp32_floor`` ceiling is retained for every split comparator.  The N2
#: state comparator adds two distribution constraints below, so it is a
#: strict tightening rather than a reinterpretation of the 8-ULP ceiling.
FP64_MIRROR_MAX_ULPS = 8
FP32_STATE_EQUIV_MAX_ULPS = 8
FP32_STATE_EQUIV_P99_MAX_ULPS = 2
FP32_STATE_EQUIV_MEAN_SIGNED_ULP_MAX_ABS = 0.25
REAL_EMULATION_MIRROR_MAX_ULPS = 8

#: Compatibility alias for Task-10 operator tests written before the
#: comparator-kind split.  It is deliberately equal to, never wider than,
#: the FP64-mirror ceiling.
FP32_FLOOR_MAX_ULPS = FP64_MIRROR_MAX_ULPS

#: N5/N6 evidence values whose identity binds only when a controller run is
#: scheduled.  The manifest writer replaces the sentinel with the frozen
#: value before evaluating the gate.
CONTROLLER_EXECUTION_SENTINEL = "controller-fills-at-execution"
_EVIDENCE_FIELDS = (
    "evaluator_commit", "mask_hash", "baseline_hash", "cadence",
    "expected_samples")

#: F10 amendment: executable comparator semantics PER BOUND KIND.  Rung
#: executors implement EXACTLY these semantics; a record's ``convention``
#: adds the fixture identity (baseline, mask, valid time, field list) but
#: never overrides its kind's comparator.  Every kind BLOCKS the next
#: rung except ``diagnostic`` -- the ONLY non-blocking kind.
COMPARATORS: dict[str, str] = {
    "max": (
        "metric computed per the record's convention; pass iff "
        "metric <= threshold; a NaN/Inf metric FAILS."),
    "min": (
        "metric computed per the record's convention; pass iff "
        "metric >= threshold; a NaN/Inf metric FAILS."),
    "strict_max": (
        "metric computed per the record's convention; pass iff "
        "metric < threshold; a NaN/Inf metric FAILS."),
    "bitwise": (
        "byte identity of the named artifact(s) against the named "
        "baseline bytes; one differing byte FAILS; a missing artifact "
        "or baseline FAILS."),
    "exact": (
        "exact discrete equality of the named integers/structures "
        "(schedules, tick ledgers, counters, inventories); any "
        "mismatch FAILS."),
    "fp64_mirror_floor": (
        "elementwise comparison of the FP32 result against the FP64 "
        "mirror result rounded to FP32, on identical inputs including "
        "identical FP32-rounded geometry/weight tables; reduction = MAX "
        "ULP distance over all compared elements; pass iff that max "
        "<= FP64_MIRROR_MAX_ULPS (8); any NaN/Inf on either side FAILS."),
    "fp32_state_equiv": (
        "elementwise peer-state comparison after both states are stored as "
        "FP32.  Map each finite FP32 bit pattern to its monotone signed-ULP "
        "integer key; signed ULP = candidate key - reference key and ULP = "
        "abs(signed ULP).  For N elements, nearest-rank P99 is sorted_ULP["
        "ceil(0.99*N)-1].  Over all compared elements, pass iff MAX(ULP) "
        "<= FP32_STATE_EQUIV_MAX_ULPS (8), that P99(ULP) <= "
        "FP32_STATE_EQUIV_P99_MAX_ULPS (2), AND "
        "abs(arithmetic mean(signed ULP)) <= "
        "FP32_STATE_EQUIV_MEAN_SIGNED_ULP_MAX_ABS (0.25); an empty "
        "comparison or any NaN/Inf on either side FAILS."),
    "real_emulation_mirror": (
        "elementwise comparison of the FP32 result against a per-operation "
        "round-to-nearest-FP32 emulation of WRF REAL arithmetic on identical "
        "inputs (woof.verify.npref dtype=np.float32 mode; no contracted "
        "FMA); reduction = MAX ULP distance over all compared elements; pass "
        "iff that max <= REAL_EMULATION_MIRROR_MAX_ULPS (8); an empty "
        "comparison or any NaN/Inf on either side FAILS."),
    "identity_oracle": (
        "F19 composite identity-nest oracle (all components BLOCK; any "
        "NaN/Inf FAILS).  (1) t=0 full-state BIT identity of every "
        "registered field (initializer/restriction anchor).  (2) at EVERY "
        "force, table oracle: VALUE tables bit-exact to the correctly "
        "oriented child rows (Y-corner ownership and east/north distance "
        "reversal asserted); TENDENCY tables bit-exact to "
        "fl32(float64(fl32(parent-child))/float64(parent_dt_fp32)); the "
        "successor reconstruction fl32(V + fl32(dt*T)) within 2 "
        "parent-field-scale ULP -- abs err <= 2*spacing(fl32(max|parent "
        "FIELD|)), exact when that scale is zero (F19a errata: the "
        "formula originally wrote the per-SIDE max, contradicting the "
        "prose and the attribution run's measured basis; the shadow's "
        "0.125-ULP worst case and 16x headroom were FIELD-scale, and a "
        "near-zero side [early mup west] makes the per-side bound "
        "denormal-tight -- first hardware run measured 2.5 side-ULP = "
        "2.9e-10 absolute there, physically nothing, values and "
        "tendencies bit-exact).  (3) at each applicable "
        "RK stage, the applied Davies/specified tendency equals the "
        "independently evaluated five-point relaxation formula (row "
        "support and stage policy included), per element within "
        "atol + rtol*max|held tendency over that side table| -- rtol "
        "5e-7 against the TABLE-MAX held tendency, the attribution "
        "run's measured residual basis ('worst residual / max held "
        "tendency' ~2e-7; F19a errata: a per-element rtol misreads "
        "growing tendencies whose FP32-vs-FP64 residual scales with "
        "the table max -- first hardware run measured relative 2.6e-7, "
        "inside the envelope, while per-element flagged 38 checks) -- "
        "with the registered per-field coupled-unit atol floors (u/v "
        "8e-6, w 6e-6, "
        "theta 4e-6, phi 2e-3, mu 1e-7, qv 5e-10); EXACT ZERO on the "
        "zero-ramp outer row and on inactive species.  (4) the nested w "
        "substep update on the specified row is bit-exact to "
        "fl(w + fl(dtau*rw_t)) for every acoustic substep.  (5) raw "
        "specified-row end-to-end guardrail after final uncoupling: "
        "max |child-parent| within the registered per-field caps (u 8e-6, "
        "v 1e-6, w 4e-9, thp 6.5e-5, php 6.5e-5, mup 2e-6, qv 2.5e-10 raw "
        "units; inactive hydrometeors/numbers bit-exact; h_diabatic "
        "excluded -- no boundary table).  Post-t=0 full-state ULP "
        "equivalence against the periodic parent is NOT part of this "
        "oracle: the p5n2attr attribution proved it unattainable for any "
        "correct implementation (different boundary-value problems; "
        "specified-vs-periodic operator grouping differs tilewide from "
        "the first flux evaluation)."),
    "measured_bound": (
        "the relational inequality stated in the record's convention, "
        "evaluated EXACTLY (no tolerance) on runtime-measured "
        "quantities whose measurement procedure the convention names; "
        "a missing measurement FAILS."),
    "structural": (
        "BLOCKING adjudicated verdict: the executor produces the "
        "evidence artifacts named in the convention and the controller "
        "records a pass/fail verdict in the ledger; a missing verdict "
        "IS a fail -- the gate can never be silently skipped."),
    "diagnostic": (
        "recorded and reviewed, non-blocking; the ONLY non-blocking "
        "kind."),
}


@dataclass(frozen=True)
class NestGate:
    """One pre-registered milestone gate: an immutable ledger row."""

    #: Ladder rung: one of :data:`MILESTONES`.
    milestone: str
    #: Metric identifier, units suffixed reference-case style (``t500_rmse_k``).
    metric: str
    #: Bound kind: one of :data:`BOUND_KINDS`.
    kind: str
    #: Numeric bound for :data:`NUMERIC_KINDS`; ``None`` otherwise.
    threshold: float | None
    #: Comparison convention: baselines, masks, field lists, and any
    #: adoption note where section F names a family without a number.
    convention: str
    #: Anchor into :data:`ARCHITECTURE_DOC` (the section-F rung paragraph).
    anchor: str
    #: Evaluator source revision used to compute the evidence.
    evaluator_commit: str | None = None
    #: Hash of the exact spatial/field mask consumed by the evaluator.
    mask_hash: str | None = None
    #: Hash of the baseline or control artifact set.
    baseline_hash: str | None = None
    #: Sampling/accumulation cadence pin.
    cadence: str | None = None
    #: Expected sample-count pin; sentinel string when schedule-bound at run.
    expected_samples: int | str | None = None

    def __post_init__(self) -> None:
        """Materialize mandatory run-bound evidence pins for N5/N6 blockers."""
        if self.milestone in {"N5", "N6"} and self.kind != "diagnostic":
            for name in _EVIDENCE_FIELDS:
                if getattr(self, name) is None:
                    object.__setattr__(self, name,
                                       CONTROLLER_EXECUTION_SENTINEL)


# ---------------------------------------------------------------------------
# Named thresholds (verbatim, architecture section F).  The statistical
# family registered at N3(ii) for d02 is re-registered at the SAME values
# for d03 (N4) and d04 (N5): "same gate family, thresholds pre-registered
# at the N3 values pending N1.5 evidence" (section F N4; plan Task 8(a)).
# ---------------------------------------------------------------------------

#: N1 direct static oracle: blended HGT max |diff| vs reference wrfout +
#: geo_em HGT_M, over the blend zone, per child.  Meters.
#: F17 AMENDED 0.5 -> 1.5 with the attribution run in hand: chain
#: bit-exact on bundle inputs (0.0, full grid, all children), miss 100%
#: the ratified Phase-3 static terrain residual; measured worst 1.185 m
#: (d03) + ~25% headroom; input-residual ceiling 2.92 m.
HGT_BLEND_MAX_ABS_DIFF_M = 1.5
#: N1 direct static oracle: MUB max |diff| in the blend zone.  Pascals.
#: F17 AMENDED 10 -> 17: the MUB miss is the HGT miss propagated
#: hydrostatically (measured -11.1..-11.5 Pa/m = -rho*g); 17 = 1.5 m x
#: 11.4 Pa/m; measured worst 13.17 Pa (d03).
MUB_BLEND_MAX_ABS_DIFF_PA = 17.0
#: N1.5 instrumented-WRF table oracle: FP32-relative band ("~1e-6",
#: registered as 1.0e-6; order-of-ops differences only).
NEST_FORCE_FP32_RELATIVE_BAND = 1.0e-6
#: N3 statistical family, reused verbatim for d03 (N4) and d04 (N5).
MSLP_PATTERN_CORRELATION_MIN = 0.95
T500_RMSE_MAX_K = 2.0
T850_RMSE_MAX_K = 2.5
BOUNDARY_ZONE_BLOWUP_MAX = 0.5
#: N5 host-overhead profile gate: Python orchestration must be strictly
#: less than this fraction of wall clock before the 12 h flagship.
HOST_OVERHEAD_MAX_FRACTION = 0.10

#: Quantitative companions to the blinded REFL_10CM and updraft-structure
#: reviews.  These are new constraints, pre-registered before looking at the
#: candidate runs.  REFL rows are (event threshold dBZ, neighborhood radius
#: km, minimum FSS); updraft rows are nearest-rank percentiles of positive
#: column-max w whose reference-relative ratios must stay in the pinned band.
REFL_10CM_FSS_FAMILY = (
    (20.0, 5.0, 0.90),
    (30.0, 5.0, 0.80),
    (40.0, 5.0, 0.70),
)
#: F27: per-row ensemble-minimum bars for ratified standing deficiencies.
#: Values below the bar are documented and non-blocking; reaching the bar
#: self-revokes the conversion and restores normal blocking semantics there.
F27_DOCUMENTED_DEFICIENCY_ROWS = {
    ("d03", 20.0): 0.8558,
    ("d04", 20.0): 0.8558,
}
#: The floor under an F27 documented deficiency: the worst score F27 actually
#: measured (the amendment records "woof scores 0.7047-0.7084, stable across
#: runs").  Without it the "documented deficiency" branch and the "at or above
#: the bar" branch partition the finite reals and the row cannot fail for any
#: finite FSS20 -- a regression to 0.02 would report PASS under an
#: adjudication string that says "this is the known 0.70 deficiency".  A value
#: below this floor is a NEW deficiency, not the recorded one, and blocks.
F27_DOCUMENTED_DEFICIENCY_FLOOR = 0.7047
#: F24: fraction of interior grid points required to treat an FSS event
#: threshold as meteorologically measurable at the registered valid time.
FSS_DEGENERATE_EVENT_FLOOR = 1.0e-4
UPDRAFT_INTENSITY_PERCENTILES = (50, 90, 99)
UPDRAFT_INTENSITY_RATIO_MIN = 0.80
UPDRAFT_INTENSITY_RATIO_MAX = 1.25

#: Reference frames the child statistical gates score against (bundle
#: wrfout_reference; one frame each, +1:15 -- no 12 h nest references
#: exist, which is why N6 is a reasonability verdict).
CHILD_REFERENCE_FRAMES = {
    "d02": "wrfout_d02_1974-04-03_13_15_00",
    "d03": "wrfout_d03_1974-04-03_13_15_00",
    "d04": "wrfout_d04_1974-04-03_13_15_00",
}

#: F21/F24: matched-physics WRF references for the FSS rows ONLY (same
#: frame names, per-domain bundles).  Domains absent here have no matched
#: reference yet; their FSS rows keep the mp55 reference and BLOCK until
#: one is registered.  Values are (bundle, frame name, SHA-256, bytes) --
#: consumption must verify the pin and FAIL loudly on mismatch.
MATCHED_REFERENCE_FRAMES = {
    "d02": ("gpuwm-wrf-matched-mp10-ysu-19740403-v1",
            "wrfout_d02_1974-04-03_13_15_00",
            "65fc6f52205c14cdf618ddf19520e8ae541ec74a9188de45524f9c12e2fb04aa",
            333839612),
    "d03": ("gpuwm-wrf-matched-mp10-ysu-19740403-v2",
            "wrfout_d03_1974-04-03_13_15_00",
            "4382b72ed3fa76662030f92e91ac97b3229e771b8dfce1b2351b58bd4ad25754",
            383352328),
    "d04": ("gpuwm-wrf-matched-mp10-ysu-19740403-v2",
            "wrfout_d04_1974-04-03_13_15_00",
            "be32aa0b1b3b8b79e2d9f5748c8becc6fdd57e579185cdd819131388306622d6",
            488452790),
}


def _statistical_family(milestone: str, domain: str,
                        note: str, anchor: str) -> tuple[NestGate, ...]:
    """The N3(ii) statistical gate family for one child domain.

    Registered once for d02 at N3 and re-registered AT THE SAME VALUES for
    d03 (N4) and d04 (N5) per section F -- a single builder so the family
    cannot drift apart silently; the tests still pin every expanded record.
    """
    ref = CHILD_REFERENCE_FRAMES[domain]
    matched_reference = MATCHED_REFERENCE_FRAMES.get(domain)
    f27_bar = F27_DOCUMENTED_DEFICIENCY_ROWS.get((domain, 20.0))
    if f27_bar is None:
        fss_threshold_text = (
            "Pass iff FSS20 >= 0.90 AND FSS30 >= 0.80 AND FSS40 >= 0.70, "
            "exactly REFL_10CM_FSS_FAMILY")
    else:
        f27_floor = F27_DOCUMENTED_DEFICIENCY_FLOOR
        fss_threshold_text = (
            f"Pass iff FSS30 >= 0.80 AND FSS40 >= 0.70.  FSS20 >= "
            f"{f27_bar:.4f} under F27: a value below {f27_bar:.4f} but at or "
            f"above the measured floor {f27_floor:.4f} is a "
            f"non-blocking DOCUMENTED-DEFICIENCY; below {f27_floor:.4f} the "
            f"deficiency is not the one F27 recorded and the row BLOCKS; at "
            f"or above "
            f"{f27_bar:.4f} the conversion self-revokes and the row returns "
            f"to normal blocking at {f27_bar:.4f}")
    vs = f"{domain} vs reference {ref}. {note}"
    return (
        NestGate(milestone, f"{domain}_mslp_pattern_correlation", "min",
                 MSLP_PATTERN_CORRELATION_MIN,
                 f"MSLP pattern correlation >= 0.95; {vs}", anchor),
        NestGate(milestone, f"{domain}_t500_rmse_k", "max",
                 T500_RMSE_MAX_K,
                 f"T500 RMSE <= 2.0 K, interior convention (the section-F "
                 f"N3(ii) parenthetical governs the RMSE pair); {vs}",
                 anchor),
        NestGate(milestone, f"{domain}_t850_rmse_k", "max",
                 T850_RMSE_MAX_K,
                 f"T850 RMSE <= 2.5 K (interior convention); {vs}", anchor),
        NestGate(milestone, f"{domain}_boundary_zone_blowup", "max",
                 BOUNDARY_ZONE_BLOWUP_MAX,
                 f"boundary_zone_blowup <= 0.5, same metric definition as "
                 f"the frozen reference-case d01 gate; {vs}", anchor),
        NestGate(milestone, f"{domain}_refl_10cm_structure", "structural",
                 None,
                 f"REFL_10CM structure present (coherent precipitation "
                 f"structures, not a numeric score); {vs}", anchor),
        NestGate(
            milestone, f"{domain}_refl_10cm_fss", "measured_bound", None,
            f"QUANTITATIVE companion to the supplementary blinded structure "
            f"review.  On the pinned interior mask, form binary REFL_10CM "
            f"events at 20/30/40 dBZ and neighborhood fractions over grid "
            f"centres within 5 km.  FSS = 1-sum((gpu-ref)^2)/"
            f"sum(gpu^2+ref^2), with FSS=1 only when both fraction fields "
            f"are identically zero; missing/non-finite fields or any other "
            f"zero denominator FAIL.  {fss_threshold_text}, "
            + (f"vs the F21 MATCHED-PHYSICS reference {ref} from "
               f"{matched_reference[0]} (SHA-256 pinned in "
               f"MATCHED_REFERENCE_FRAMES; like-for-like Morrison+YSU "
               f"physics -- the mp55 reference stays authoritative for "
               f"every other row of this family)"
               if matched_reference is not None else
               f"vs reference {ref} (F21: no matched-physics reference is "
               f"registered for {domain} yet; this row BLOCKS until "
               f"MATCHED_REFERENCE_FRAMES registers one -- the mp55 "
               f"physics gap measured at d02 makes this reference "
               f"unpassable like-for-like)")
            + f".  This adds a blocker without replacing or "
            f"weakening the structural record. {note}", anchor),
    )


NEST_GATES: tuple[NestGate, ...] = (
    # -- N0: memory preflight (first; gates all dynamics-lane merges) ------
    NestGate(
        "N0", "alloc_fits_wddm_budget", "measured_bound", None,
        "woof check --alloc allocates all four domains + drivers + d01 "
        "LBC + coupler tables, zero integration; the measured allocation "
        "fits under the MEASURED WDDM free-VRAM budget (runtime-measured "
        "bound -- no fixed number is pre-registerable).",
        "section F, N0 (MEMORY PREFLIGHT)"),
    NestGate(
        "N0", "alloc_measured_le_estimate", "measured_bound", None,
        "measured <= estimate: the preflight estimator's number is an "
        "ENFORCED upper bound (design spec section 7, criterion 3).",
        "section F, N0 (MEMORY PREFLIGHT)"),
    NestGate(
        "N0", "alloc_estimate_le_wddm_budget", "measured_bound", None,
        "estimate <= measured budget: the 15%-headroom estimate itself "
        "fits under the measured WDDM free-VRAM budget minus the "
        "configured reserve -- design spec section 7, criterion 3 "
        "requires the FULL chain measured <= estimate <= budget; the "
        "middle leg alone would pass measured=20/estimate=40/budget=30 "
        "GiB.  BLOCKING at N0, re-checked at N5 (F11 amendment).",
        "section F, N0 (MEMORY PREFLIGHT)"),

    # -- N1: unit oracles ---------------------------------------------------
    NestGate(
        "N1", "sint_vs_fp64_mirror", "fp64_mirror_floor", None,
        "SINT donor interpolation (mass/x-stag/y-stag variants) vs the "
        "FP64 NumPy transliteration in verify/npref.py consuming the SAME "
        "FP32-rounded weight tables as the kernel.  SEMANTICS-PRESERVING "
        "kind rename: the ceiling remains 8 ULP and the FP64 fixtures are "
        "preconditioned bounded away from sign-changing cancellation.  A "
        "cancellation-sensitive force() fixture MUST instead use "
        "real_emulation_mirror (npref dtype=np.float32), never silently "
        "widen this FP64-oracle gate.",
        "section F, N1 (UNIT ORACLES)"),
    NestGate(
        "N1", "bdy_interp1_vs_fp64_mirror", "fp64_mirror_floor", None,
        "bdy_interp1 value/tendency tables vs the FP64 mirror, including "
        "the REAL*8 rdt=1.D0/cdt precision scheme "
        "(interp_fcn.F:2480/:2500) mirrored exactly.  SEMANTICS-PRESERVING: "
        "the same 8-ULP ceiling and zero-avoiding constructed fixtures are "
        "retained; any cancellation-sensitive force() table fixture uses "
        "real_emulation_mirror with npref dtype=np.float32.",
        "section F, N1 (UNIT ORACLES)"),
    NestGate(
        "N1", "blend_terrain_vs_fp64_mirror", "fp64_mirror_floor", None,
        "blend_terrain vs the FP64 mirror on FP32-rounded weights.  "
        "SEMANTICS-PRESERVING: this is the old FP64-mirror predicate renamed "
        "with its unchanged 8-ULP ceiling; fixtures use finite positive "
        "terrain and cannot introduce sign-changing cancellation.",
        "section F, N1 (UNIT ORACLES)"),
    NestGate(
        "N1", "adjust_tempqv_vs_fp64_mirror", "fp64_mirror_floor", None,
        "adjust_tempqv kernel vs the FP64 mirror at FP32 floors.  "
        "SEMANTICS-PRESERVING: the kernel intentionally evaluates this "
        "one-shot cancellation-prone initialization operator in FP64 and "
        "stores FP32, so the existing rounded-FP64 reference and unchanged "
        "8-ULP ceiling remain the correct predicate.",
        "section F, N1 (UNIT ORACLES)"),
    NestGate(
        "N1", "copy_fcn_vs_fp64_mirror", "fp64_mirror_floor", None,
        "copy_fcn BOTH parity branches (odd AND even 1/(nri*nrj) "
        "cell-average, interp_fcn.F:1565-1654; copy_fcnm/:1806 and "
        "copy_fcni/:1888 SW-corner nearest-neighbor on even ratios) vs "
        "FP64 mirrors -- dormant Phase-5b machinery oracled in N1.  "
        "SEMANTICS-PRESERVING: the 8-ULP ceiling is unchanged and the "
        "averaging fixtures are finite/nonnegative (zero-avoiding), so the "
        "existing rounded-FP64 oracle remains sound.",
        "section F, N1 (UNIT ORACLES)"),
    NestGate(
        "N1", "sint_quadratic_reproduction", "exact", None,
        "Property test: SINT reproduces quadratic fields.",
        "section F, N1 (UNIT ORACLES)"),
    NestGate(
        "N1", "sint_parent_coincident_odd_ratio", "exact", None,
        "Property test: parent-coincident points reproduced on odd "
        "ratios.",
        "section F, N1 (UNIT ORACLES)"),
    NestGate(
        "N1", "schedule_flat_vs_recursion", "exact", None,
        "Precomputed integer-tick flat schedule == reference recursion "
        "for chains (1,4,3,3), (1,3,3), (1,5,3), (1,4,4,2).  Both sides "
        "represent WRF's terminal stop-time feedback guard "
        "(module_integrate.F:439-445: no FEEDBACK when head/parent/child "
        "is at stop time), and the pin covers an interior period AND the "
        "final period (F8 amendment).",
        "section F, N1 (UNIT ORACLES)"),
    NestGate(
        "N1", "stepra_pin", "exact", None,
        "Clock/calendar pin: STEPRA 12/12/12/36 per domain.",
        "section F, N1 (UNIT ORACLES)"),
    NestGate(
        "N1", "chained_fp32_dt_pin", "exact", None,
        "Clock/calendar pin: chained-FP32 dt values per domain "
        "(set_timekeeping.F:368 semantics on any ratio chain).",
        "section F, N1 (UNIT ORACLES)"),
    NestGate(
        "N1", "diff_6th_coefficient_pin", "exact", None,
        "Clock/calendar pin: per-domain diff_6th coefficients.",
        "section F, N1 (UNIT ORACLES)"),
    NestGate(
        "N1", "child_spec_exp_rejection", "exact", None,
        "Clock/config pin: child spec_exp rejected (children force with "
        "spec_exp=0).",
        "section F, N1 (UNIT ORACLES)"),
    # Direct static oracle (graft, Judge 1): per child, blended HGT and
    # MUB vs the reference wrfout_d02/d03/d04 fields plus geo_em HGT_M.
    NestGate(
        "N1", "d02_hgt_blend_max_abs_diff_m", "max",
        HGT_BLEND_MAX_ABS_DIFF_M,
        "d02 blended HGT vs reference wrfout_d02_1974-04-03_13_15_00 + "
        "geo_em HGT_M: max |diff| <= 1.5 m in the blend zone (F17: 0.5 m "
        "pre-amendment; loosened with the attribution run in hand -- the "
        "miss is 100% the ratified Phase-3 static terrain residual, the "
        "chain itself is pinned bit-exact by the paired "
        "chain_isolation records); arbitrates blend-at-build-time and "
        "blend-vs-derive before any dynamics run.  Measured 1.053 m.",
        "section F, N1 (DIRECT STATIC ORACLE)"),
    NestGate(
        "N1", "d02_mub_blend_max_abs_diff_pa", "max",
        MUB_BLEND_MAX_ABS_DIFF_PA,
        "d02 MUB max |diff| <= 17 Pa in the blend zone vs reference "
        "wrfout_d02_1974-04-03_13_15_00 (F17: 10 Pa pre-amendment; the "
        "MUB miss is the HGT miss propagated hydrostatically, measured "
        "-11.1..-11.5 Pa/m = -rho*g).  Measured 11.70 Pa.",
        "section F, N1 (DIRECT STATIC ORACLE)"),
    NestGate(
        "N1", "d03_hgt_blend_max_abs_diff_m", "max",
        HGT_BLEND_MAX_ABS_DIFF_M,
        "d03 blended HGT vs reference wrfout_d03_1974-04-03_13_15_00 + "
        "geo_em HGT_M: max |diff| <= 1.5 m in the blend zone (F17; "
        "measured 1.185 m, the worst child).",
        "section F, N1 (DIRECT STATIC ORACLE)"),
    NestGate(
        "N1", "d03_mub_blend_max_abs_diff_pa", "max",
        MUB_BLEND_MAX_ABS_DIFF_PA,
        "d03 MUB max |diff| <= 17 Pa in the blend zone vs reference "
        "wrfout_d03_1974-04-03_13_15_00 (F17; measured 13.17 Pa).",
        "section F, N1 (DIRECT STATIC ORACLE)"),
    NestGate(
        "N1", "d04_hgt_blend_max_abs_diff_m", "max",
        HGT_BLEND_MAX_ABS_DIFF_M,
        "d04 blended HGT vs reference wrfout_d04_1974-04-03_13_15_00 + "
        "geo_em HGT_M: max |diff| <= 1.5 m in the blend zone (F17; "
        "measured 0.725 m).",
        "section F, N1 (DIRECT STATIC ORACLE)"),
    NestGate(
        "N1", "d04_mub_blend_max_abs_diff_pa", "max",
        MUB_BLEND_MAX_ABS_DIFF_PA,
        "d04 MUB max |diff| <= 17 Pa in the blend zone vs reference "
        "wrfout_d04_1974-04-03_13_15_00 (F17; measured 8.50 Pa).",
        "section F, N1 (DIRECT STATIC ORACLE)"),
    # F17 additive tightening: the chain-error detector the loosened
    # numeric bounds previously stood in for, pinned at bit-exactness.
    NestGate(
        "N1", "d02_chain_isolation_bit_exact", "exact", None,
        "F17: the bundle's own inputs (geo_em.d02 HGT_M + reference "
        "wrfout_d01 HGT) pushed through woof's GPU SINT+blend chain "
        "must reproduce the reference blended d02 HGT with max |diff| "
        "== 0.0 on the FULL grid (executable procedure: the controller "
        "attribution runner, evidence out/n1-measure/attribution.json; "
        "measured 0.0 on 2026-07-16).  Any nonzero element FAILS.",
        "section F, N1 (DIRECT STATIC ORACLE, F17)"),
    NestGate(
        "N1", "d03_chain_isolation_bit_exact", "exact", None,
        "F17: geo_em.d03 HGT_M + reference wrfout_d02 HGT through the "
        "chain == reference blended d03 HGT bit-exactly, full grid "
        "(measured 0.0 on 2026-07-16).",
        "section F, N1 (DIRECT STATIC ORACLE, F17)"),
    NestGate(
        "N1", "d04_chain_isolation_bit_exact", "exact", None,
        "F17: geo_em.d04 HGT_M + reference wrfout_d03 HGT through the "
        "chain == reference blended d04 HGT bit-exactly, full grid "
        "(measured 0.0 on 2026-07-16).",
        "section F, N1 (DIRECT STATIC ORACLE, F17)"),

    # -- N1.5: instrumented-WRF table oracle --------------------------------
    NestGate(
        "N1.5", "bdy_value_tables_fp32_relative", "max",
        NEST_FORCE_FP32_RELATIVE_BAND,
        "woof nest_force on the restored inputs vs the patched v4.6.1 "
        "med_nest_force dump of the child bdy_xs/xe/ys/ye VALUE tables "
        "for u/v/w/t/ph/mu/moist AND all active scalar species incl. "
        "Morrison qni/qns/qnr/qng (Registry.EM_COMMON:3026 scalar set, "
        "bdy_interp:dt at :519-542 -- F9 amendment); FP32-relative, "
        "order-of-ops differences only.  METRIC (F10): elementwise pass "
        "|woof - wrf| <= band * (|wrf| + max|wrf| over that field's "
        "table) -- relative with an absolute fallback at the table's own "
        "scale, so zero/near-zero entries compare at table magnitude; "
        "NaN/Inf anywhere FAILS; aggregation covers every element of "
        "every field's four-side tables.  Section F registers an '~1e-6 "
        "band'; 1.0e-6 is the pre-registered number -- widening it is a "
        "plan amendment.  This rung is MANDATORY before any "
        "pre-registered threshold loosens.",
        "section F, N1.5 (INSTRUMENTED-WRF TABLE ORACLE)"),
    NestGate(
        "N1.5", "bdy_tendency_tables_fp32_relative", "max",
        NEST_FORCE_FP32_RELATIVE_BAND,
        "Same oracle and F10 metric (relative with per-table absolute "
        "fallback; NaN/Inf fails; all elements), TENDENCY tables for "
        "u/v/w/t/ph/mu/moist AND all active scalar species incl. "
        "Morrison qni/qns/qnr/qng (F9); FP32-relative ~1e-6 band "
        "registered as 1.0e-6.  F18 AMENDMENT (attribution run in "
        "hand): the four state.t.*.tendency tables are compared at the "
        "FP32 successor scale E = fl32(VALUE + fl32(cdt*TENDENCY)) -- "
        "the exact expression WRF's boundary consumer applies -- under "
        "the SAME unchanged band, with cdt taken from the dump metadata "
        "parent_dt.  This is the registered treatment of the PROVENANCE "
        "'N1.5 harness seam -- dumped-WRF theta restoration' "
        "(production-absent inverse representation map; measured "
        "successor metrics 2.06e-7..2.59e-7).  Every other family/field "
        "keeps the raw metric -- the u/v low-face defect the same run "
        "exposed was FIXED to bit identity, never absorbed by tolerance.",
        "section F, N1.5 (INSTRUMENTED-WRF TABLE ORACLE)"),
    NestGate(
        "N1.5", "producer_stored_recurrence_fp32_relative", "max",
        NEST_FORCE_FP32_RELATIVE_BAND,
        "F18 additive: the producer hard-fails unless woof's own stored "
        "tables reproduce woof's raw FP32 SINT successor -- "
        "fl32(V_gpuwm + fl32(cdt*T_gpuwm)) vs the raw successor under "
        "the same band and coupled-value scale (measured <= 4.4e-10; "
        "solely the unavoidable stored-tendency inversion loss).  "
        "Protects the recurrence identity the successor comparison "
        "relies on.",
        "section F, N1.5 (INSTRUMENTED-WRF TABLE ORACLE, F18)"),

    # -- N2: identity and idealized nests ------------------------------------
    NestGate(
        "N2", "null_nest_r1_dry_restriction", "identity_oracle", None,
        "(a) DRY same-grid ratio-1 null nest, physics off.  F19 AMENDED "
        "(p5n2attr attribution in hand): the pre-registered post-t=0 "
        "full-state fp32_state_equiv against the parent was proven "
        "unattainable for ANY correct implementation -- the parent "
        "advances a PERIODIC initial-value problem while the child "
        "advances the Davies boundary-value problem, and the "
        "specified-operator branch groups FP32 arithmetic differently "
        "from the periodic branch tilewide from the first flux "
        "evaluation (measured: t=0 bit-exact; post-step differences "
        "physically tiny [<=5e-3 relative at t=60] but unbounded in raw "
        "near-zero ULP).  Replaced by the SHARPER composite "
        "identity_oracle (t=0 bit anchor + bit-exact force tables + "
        "Davies-tendency formula oracle + nested-w substep bit identity "
        "+ raw specified-row caps -- this run measured 16x headroom on "
        "the successor bound and 2x on the raw caps; an orientation "
        "error measures O(1e3..7e5) coupled units = loud).  Fixture: "
        "verify/cases/nest_null_r1.py.",
        "section F, N2(a)"),
    NestGate(
        "N2", "identity_nest_r1_moist_wk82", "identity_oracle", None,
        "(b) MOIST ratio-1 identity nest on the Phase-2 Weisman-Klemp "
        "supercell: exercises the nested w tables and "
        "hydrometeor/number-concentration plumbing a dry null nest "
        "misses (inactive condensate/number fields must stay BIT-EXACT; "
        "qv under the registered oracle bounds).  F19 AMENDED with the "
        "same attribution and composite oracle as N2(a).  Fixture: "
        "verify/cases/nest_ideal_r1_moist.py.",
        "section F, N2(b)"),
    NestGate(
        "N2", "wk_r3_boundary_zone_blowup", "max",
        BOUNDARY_ZONE_BLOWUP_MAX,
        "(c) ratio-3 WK nest: no boundary reflection.  Section F names a "
        "'boundary_zone_blowup-style w gate' without restating a number; "
        "the N3 family value 0.5 is the bound -- RATIFIED by the "
        "2026-07-16 controller amendment (architecture section F N2(c); "
        "plan Task 8(a)) (F15).  Same metric definition as the frozen "
        "reference-case d01 gate.  Fixture: verify/cases/nest_ideal_r3.py.",
        "section F, N2(c)"),
    NestGate(
        "N2", "wk_r3_interface_crossing", "structural", None,
        "(c) the storm crosses the nest interface without ringing.  "
        "Fixture: verify/cases/nest_ideal_r3.py.  Evidence (F10): the "
        "per-output-time series of max|w| over the child boundary frame "
        "(spec+relax+blend rows) vs the child interior max|w| through "
        "the crossing window, plus w cross-sections at the interface; "
        "verdict adjudicated per the structural comparator (missing "
        "verdict = FAIL).",
        "section F, N2(c)"),
    NestGate(
        "N2", "wk_r3_interior_vs_uniform_highres", "structural", None,
        "(c) child interior statistically matches a uniform high-res "
        "single-domain run of the same WK82 case.  Fixture: "
        "verify/cases/nest_ideal_r3.py vs the co-located restriction of "
        "the uniform high-res run.  Evidence (F10): interior-convention "
        "statistics (w, theta perturbation, hydrometeor fields) at the "
        "pinned valid time; verdict adjudicated per the structural "
        "comparator (missing verdict = FAIL).",
        "section F, N2(c)"),

    # -- N3: real d01+d02, 12:00 -> 13:15 ------------------------------------
    NestGate(
        "N3", "d01_bitwise_vs_phase4_13z", "bitwise", None,
        "(i) GPUWM-INVARIANCE GATE: d01 trajectory FP32-BIT-IDENTICAL "
        "to the anchor run at 13:00 (same wrfout state bytes); "
        "this protects the registered non-mutating-coupling deviation, "
        "not WRF identity.  Ratified exceptions (F20): REFL_10CM values "
        "(D2 seam), GPUWM_WRITE_COMPLETE (T15 publication attribute), "
        "TITLE (case-descriptive metadata; value excluded, presence "
        "required).  Davies clock bind (2026-07-28, retires the F20 "
        "adjudication): the tree build binds the root's external-"
        "boundary clock, so root Davies launches consume WRF's post-"
        "increment recurrence (solve_em.F:371-372) on root and children "
        "alike; the byte authority is the seam-closure anchor epoch "
        "(reference-t7-final-r3) -- the metric keeps its historic Phase-4 "
        "name, while the original Phase-4/-r2 bytes encode the retired "
        "pre-bind elapsed-based dtbc and remain historical evidence "
        "only.",
        "section F, N3(i)"),
    *_statistical_family(
        "N3", "d02",
        "(ii) pre-registered d02 thresholds; a miss triggers N1.5 "
        "attribution BEFORE any loosening.",
        "section F, N3(ii)"),
    NestGate(
        "N3", "d02_hgt_blend_recheck_m", "max", HGT_BLEND_MAX_ABS_DIFF_M,
        "(iii) HGT blend gate re-checked in the output frame, same bound "
        "as the N1 static oracle (F17: 1.5 m; 0.5 m pre-amendment -- "
        "HGT is static during integration, so the N1 attribution "
        "applies verbatim).",
        "section F, N3(iii)"),
    NestGate(
        "N3", "d02_mub_blend_recheck_pa", "max", MUB_BLEND_MAX_ABS_DIFF_PA,
        "(iii) MUB blend gate re-checked in the output frame, same bound "
        "as the N1 static oracle (F17: 17 Pa; 10 Pa pre-amendment).",
        "section F, N3(iii)"),
    NestGate(
        "N3", "d02_blend_zone_t2_tsk_bias", "diagnostic", None,
        "(iv) blend-zone T2/TSK bias diagnostic -- the soil blind-spot "
        "monitor (soil is initialized on unblended fine terrain, never "
        "re-adjusted, mirroring WRF; flagged and diagnosed, not bound).",
        "section F, N3(iv)"),
    NestGate(
        "N3", "restart_split_bit_identity", "bitwise", None,
        "(v) restart split bit-identity: 30 min write + resume + 45 min "
        "== straight-through, per domain.",
        "section F, N3(v)"),
    NestGate(
        "N3", "two_domain_alloc_check", "measured_bound", None,
        "(vi) 2-domain --alloc check under the N0 contract.",
        "section F, N3(vi)"),

    # -- N4: d01+d02+d03 to 13:15 --------------------------------------------
    NestGate(
        "N4", "d01_bitwise_vs_phase4_13z", "bitwise", None,
        "RATCHET: d01 bitwise vs the anchor epoch (historic Phase 4 "
        "metric name; reference-t7-final-r3 bytes since the 2026-07-28 "
        "Davies clock bind) protects the registered "
        "non-mutating-coupling deviation when d03 is added.",
        "section F, N4 (RATCHETS)"),
    NestGate(
        "N4", "d02_bitwise_vs_n3", "bitwise", None,
        "RATCHET: d02 bitwise vs N3's d02 protects the registered "
        "non-mutating-coupling deviation when d02 gains a child.",
        "section F, N4 (RATCHETS)"),
    *_statistical_family(
        "N4", "d03",
        "Same gate family, thresholds pre-registered at the N3 values "
        "pending N1.5 evidence (section F, N4).",
        "section F, N4"),
    NestGate(
        "N4", "d03_w_cfl_health", "structural", None,
        "w/CFL health at 1 km grid spacing.",
        "section F, N4"),

    # -- N5: full chain to 13:15 ----------------------------------------------
    NestGate(
        "N5", "d03_bitwise_vs_n4", "bitwise", None,
        "RATCHET: d03 bitwise vs N4's d03 protects the registered "
        "non-mutating-coupling deviation.",
        "section F, N5"),
    *_statistical_family(
        "N5", "d04",
        "d04 statistical thresholds pre-registered at the N3 family "
        "values (plan Task 8(a)); section F additionally names "
        "domain-mean profiles, MSLP correlation, updraft-intensity "
        "distribution sanity, REFL_10CM structure.",
        "section F, N5"),
    NestGate(
        "N5", "d04_domain_mean_profiles", "structural", None,
        "d04 domain-mean profile sanity vs "
        "wrfout_d04_1974-04-03_13_15_00.",
        "section F, N5"),
    NestGate(
        "N5", "d04_updraft_intensity_distribution", "structural", None,
        "SUPPLEMENTARY blinded review of d04 updraft-intensity "
        "distribution sanity; a recorded PASS verdict is required in "
        "addition to the quantitative percentile-band companion.",
        "section F, N5"),
    NestGate(
        "N5", "d04_updraft_intensity_percentile_band", "measured_bound",
        None,
        "QUANTITATIVE companion vs reference "
        "wrfout_d04_1974-04-03_13_15_00.  On the pinned interior mask, "
        "sample finite positive column-max w values >= 1 m s-1; empty or "
        "non-finite candidate/reference samples FAIL.  For nearest-rank "
        "P50/P90/P99, define Pq=sorted_sample[ceil(q*N/100)-1]; pass iff EACH "
        "gpu/reference percentile ratio is in [0.80, 1.25], exactly "
        "UPDRAFT_INTENSITY_PERCENTILES and the pre-registered "
        "UPDRAFT_INTENSITY_RATIO_MIN/MAX band.  The blinded structural "
        "record remains independently blocking.",
        "section F, N5"),
    NestGate(
        "N5", "N5S_matched_physics_wrf_shadow", "measured_bound", None,
        "CONTROLLER-RUN shadow gate (WSL; CPU WRF and GPU candidate are run "
        "sequentially so there is no GPU conflict), BLOCKING N6.  From "
        "identical restored inputs, run woof and the stock instrumented "
        "WRF v4.6.1 Morrison+YSU T8b build for >= 30 min on all four "
        "domains.  Before scoring woof, run M >= 3 CPU-WRF members whose "
        "inputs differ only by a documented single 1-ULP perturbation.  For "
        "every domain/field low-pass state RMSE distance, every applied-"
        "boundary-increment error, d04 reflectivity FSS distance (1-FSS), "
        "and every storm-object timing absolute difference, form all "
        "unordered CPU-member pair distances and E95 = nearest-rank "
        "sorted_pair_distance[ceil(0.95*n_pairs)-1].  Missing/non-finite "
        "samples FAIL.  These are the GPU-vs-WRF distances: pass iff EACH "
        "gpuwm-vs-unperturbed-WRF distance <= "
        "its like-for-like E95; aggregation may not average a miss away.  "
        "F28: an envelope is degenerate exactly when E95 == 0; that row is "
        "DOCUMENTED-EVIDENCE under f28-degenerate-envelope, which is the "
        "absence of a yardstick and not evidence of agreement, so the row is "
        "INCOMPLETE -- neither a pass nor a fail -- and the compound verdict "
        "is INCOMPLETE, never PASS, while any such row stands.  All "
        "non-degenerate rows remain binding "
        "without threshold changes.  RE-BINDING is automatic row-by-row: "
        "the staged Phase-6 convective-window extension restores full binding "
        "force for every row whose future E95 > 0.",
        "section F, N5 (external-review matched-physics shadow)"),
    NestGate(
        "N5", "N5B_d04_boundary_location_invariance", "measured_bound",
        None,
        "BLOCKING N6 boundary-location A/B: execute two sequential 75-min "
        "runs from identical inputs.  A uses production d04 600x600; B "
        "shrinks d04 by 51 cells per side to 498x498; the verification core "
        "is the central 400x400 common region (F22: exact 50-per-side is "
        "unrepresentable on the ratio-3 parent-anchor lattice; 51/side "
        "keeps the shrink parent-aligned at 498 = 3x166 with anchor "
        "151 -> 168 and the centered cores exactly coincident).  With the "
        "evaluator/mask "
        "manifest pins applied, pass iff core REFL_10CM>=40-dBZ FSS at 5 km "
        ">= 0.90, cold-pool-edge median symmetric distance <= 3 km, "
        "gust-front arrival-time MAE <= 5 min, unmatched boundary-seeded CI "
        "object count == 0, AND inflow-fetch resolved-TKE(B)/resolved-TKE(A) "
        "is in [0.80, 1.25].  A boundary-seeded CI object has >=40 dBZ "
        "within 10 km of A's boundary, persists >=20 min, covers >=25 km^2, "
        "and is unmatched when no B object overlaps the pinned object mask "
        "within 10 km and 10 min.  Missing/non-finite inputs FAIL.  PRE-LOOK "
        "ENVELOPE PROCEDURE: before any boundary-shift metric is exposed, "
        "freeze hashes/results from an M>=3 same-geometry 1-ULP ensemble.  "
        "Express the five predicates as discrepancies 1-FSS, edge distance, "
        "arrival MAE, unmatched count, and lower/upper TKE-ratio excursion; "
        "for each, freeze allowed=max(the stated limit, nearest-rank P95 "
        "same-geometry discrepancy).  If roundoff spread exceeds a stated "
        "limit, this larger envelope is therefore frozen before looking; "
        "the manifest records ensemble hashes and freeze timestamp and any "
        "post-look change FAILS.",
        "section F, N5 (external-review boundary-location invariance)"),
    NestGate(
        "N5", "ancestor_inertness_every_output", "bitwise", None,
        "For d01/d02/d03, hash the canonical mutable state (fixed field "
        "order, dtype, shape and C-order bytes pinned by evaluator_commit) at "
        "EVERY scheduled output frame.  For each ancestor, the baseline is "
        "the otherwise-identical no-younger-child control.  Pass iff the "
        "schedule-derived frame-key inventory equals "
        "expected_samples AND every candidate hash equals its control hash; "
        "a missing/extra frame or hash FAILS.  This extends, never replaces, "
        "the ratchet-instant records.",
        "section F, N5 (external-review ancestor inertness)"),
    NestGate(
        "N5", "full_tree_restart_bit_identity", "bitwise", None,
        "Full-tree restart bit-identity (write + resume == "
        "straight-through, every domain).",
        "section F, N5"),
    NestGate(
        "N5", "tick_exact_sync_ledger", "exact", None,
        "Tick-exact sync ledger across 25,920 d04-equivalent steps.",
        "section F, N5"),
    NestGate(
        "N5", "memory_peak_le_estimate", "measured_bound", None,
        "Measured memory peak vs estimate -- the estimate is an enforced "
        "upper bound (N0 contract at full chain).",
        "section F, N5"),
    NestGate(
        "N5", "estimate_le_wddm_budget", "measured_bound", None,
        "Full-chain re-check of the N0 contract's estimate <= measured "
        "WDDM budget leg (F11 amendment): the enforced upper bound must "
        "still fit the device at N5's measured baseline.",
        "section F, N5"),
    NestGate(
        "N5", "host_overhead_fraction", "strict_max",
        HOST_OVERHEAD_MAX_FRACTION,
        "HOST-OVERHEAD PROFILE gate: Python orchestration (op-table "
        "walk, alarms, launch trains across 25,920 d04 steps) must be "
        "< 10% of wall before the 12 h run; CUDA-graph capture per "
        "domain-step is the Phase-6 hook if it fails.",
        "section F, N5 (also section E, MEASURED BEFORE COMMITTING (6))"),

    # -- N6: Super Outbreak 12 h flagship --------------------------------------
    NestGate(
        "N6", "scientific_reasonability", "structural", None,
        "COMPOUND blocker (no 12 h nest references exist): pass iff every "
        "new quantitative reference-frame companion has passed "
        "(d02/d03/d04 refl_10cm_fss AND d04 "
        "updraft_intensity_percentile_band) AND an independently blinded "
        "review records PASS for discrete warm-sector supercells, "
        "right-movers, hook/UH tracks on d03/d04, and cold pools under the "
        "parent design's section 10.4 contract.  Any quantitative miss, "
        "missing blinded verdict, or blinded FAIL makes this record FAIL; "
        "the review supplements rather than substitutes for numbers.",
        "section F, N6"),
    NestGate(
        "N6", "d01_13z_gates_unchanged", "bitwise", None,
        "d01 13:00 gates unchanged in the flagship (the frozen anchor-"
        "epoch bytes contract of N3(i) still holds; reference-t7-final-r3 "
        "since the 2026-07-28 Davies clock bind -- the historic Phase-4 "
        "bytes encode the retired pre-bind semantics).",
        "section F, N6"),
    NestGate(
        "N6", "ancestor_inertness_every_output", "bitwise", None,
        "Across the FULL 12 h, hash d01/d02/d03 canonical mutable state "
        "(fixed field order, dtype, shape and C-order bytes pinned by "
        "evaluator_commit) at EVERY scheduled output frame.  For each "
        "ancestor, compare with the otherwise-identical no-younger-child "
        "control.  Pass iff the schedule-derived frame-key inventory equals "
        "expected_samples AND every candidate hash equals its control hash; "
        "a missing/extra frame or mismatch FAILS.  This strictly extends the "
        "13:00 and ratchet-instant bitwise gates.",
        "section F, N6 (external-review ancestor inertness)"),
    NestGate(
        "N6", "completion_zero_validator_fires", "exact", None,
        "BLOCKING (F12 amendment; plan Task 17): the flagship completes "
        "the full 12 h (elapsed_ticks == 129,600) NaN-free with health-"
        "validator fire count == 0.",
        "section F, N6"),
    NestGate(
        "N6", "health_validator_invocations_exact", "exact", None,
        "BLOCKING schedule-accounting predicate.  Materialize "
        "expected_count from the frozen integer-tick op schedule and all "
        "registered validator call sites, and expected_frames from the exact "
        "diagnostic-frame schedule before execution.  Pass iff the exact "
        "tuple (validator_fire_count, validator_invocation_count, "
        "diagnostic_frames_processed) == (0, expected_count, "
        "expected_frames).  Missing counters, missing/extra calls or frames, "
        "and any unequal integer FAIL.",
        "section F, N6 (external-review health accounting)"),
    NestGate(
        "N6", "output_inventory_complete", "exact", None,
        "BLOCKING (F12 amendment; plan Task 17): exact inventory "
        "equality vs the configured schedule -- per-domain wrfouts at "
        "the configured cadences (d01 3600 s; d02-d04 900 s), REFL_10CM "
        "present in d03/d04 outputs, composite-reflectivity + hook/UH-"
        "track products for d03/d04, MSLP/T2/500 hPa/precip products "
        "for d01/d02.",
        "section F, N6"),
    NestGate(
        "N6", "n6_high_frequency_products", "exact", None,
        "For every 900 s d04 output interval, accumulator kernels update at "
        "EVERY d04 STEP (WRF nwp_diagnostics style), not only at output "
        "times: time-max UH 2-5 km, time-max column w, time-max 10 m wind, "
        "and time-min 2 m theta-e.  Pass iff the exact field-name inventory "
        "is present in every scheduled 900 s frame, every value is finite, "
        "each frame's recorded accumulator-update count equals the integer-"
        "tick schedule's d04-step count for that interval (540 for each full "
        "900 s interval at dt=5/3 s), and reset/emission frame keys exactly "
        "equal expected_samples.  Missing/extra fields, a non-finite value, "
        "output-only accumulation, or any cadence/count mismatch FAILS.",
        "section F, N6 (external-review high-frequency products)",
        cadence="every-d04-step accumulation; 900-s emission",
        expected_samples=CONTROLLER_EXECUTION_SENTINEL),
    NestGate(
        "N6", "d04_budget_records", "measured_bound", None,
        "For EACH 15-min window and the full 12 h, record d04 storage "
        "changes, signed boundary transports, physical sources/sinks, and "
        "R = delta_storage + net_boundary_outflow - net_sources for dry mass "
        "and total water, using the pinned evaluator/mask/baseline.  Pass iff "
        "for every window and full run |R_dry| <= max(2*|R_dry_oracle|, "
        "1e-5*dry_mass_throughput) AND |R_water| <= "
        "max(2*|R_water_oracle|, 0.005*water_throughput), with nonnegative "
        "throughput defined as storage exchanged plus absolute boundary and "
        "source/sink transports by the evaluator.  Also require "
        "cumulative_clipped_condensate_fraction <= max(2*oracle_fraction, "
        "1e-8).  Missing/non-finite terms or zero-throughput with nonzero "
        "residual FAIL.  Evidence MUST retain per-species and boundary-vs-"
        "interior residual/clipping breakdowns even though the conjunction "
        "is evaluated on dry mass, total water, and cumulative clipping.",
        "section F, N6 (external-review d04 budgets)"),
    NestGate(
        "N6", "memory_peak_le_estimate", "measured_bound", None,
        "BLOCKING (F12 amendment): flagship measured memory peak <= "
        "estimate -- design spec section 7, criterion 3 covers N6, not "
        "only N0/N5; the diagnostic telemetry record below cannot "
        "enforce it.",
        "section F, N6"),
    NestGate(
        "N6", "mempool_abort_threshold_compliance", "measured_bound",
        None,
        "BLOCKING (F12 amendment): no crossing of the mempool abort "
        "threshold during the 12 h soak.  The threshold itself is "
        "controller-set at run time from the measured N5 peak/WDDM "
        "baseline (plan Task 17 PRE-FLIGHT) -- a runtime-measured "
        "bound, so no fixed number is registered here.",
        "section F, N6"),
    NestGate(
        "N6", "mempool_soak_telemetry", "diagnostic", None,
        "Hourly cupy mempool used/total high-water telemetry logged by "
        "the runner (the flagship's first attempt doubles as the "
        "fragmentation soak) with an abort threshold; the abort number "
        "is controller-set at run time from the measured WDDM baseline, "
        "not pre-registered here.",
        "section F, N6 (also section E, MEASURED BEFORE COMMITTING (5))"),
    NestGate(
        "N6", "restart_3h_bitwise_resume", "bitwise", None,
        "3 h restart write -> resume -> bitwise-equal 6 h outputs.",
        "section F, N6"),
    NestGate(
        "N6", "wall_clock_vs_goal_h", "diagnostic", None,
        "Wall-clock reported against the 1-2 h goal (d04's 25,920 steps "
        "dominate); reported, not bound.",
        "section F, N6"),

    # -- CLOSING: Phase-5 two-way dormant closure (design D4; Task 16) ---------
    NestGate(
        "CLOSING", "feedback0_bitwise_inert", "bitwise", None,
        "Phase-5 closing milestone (Task 16), executed THIS phase (F13 "
        "amendment -- moved out of P5B, which never runs this phase): "
        "with all feedback code merged (kernels Task 10, hook Task 13), "
        "feedback=0 bit-inertness is proven by a REAL control -- a "
        "runtime FEEDBACK-call-path bypass A/B on the same N5 state: "
        "run A executes the dormant prepare/commit/finalize transaction "
        "exactly as merged (no-ops at feedback=0), run B bypasses the "
        "FEEDBACK call path "
        "via a test-only executor switch; whole-output bytes identical. "
        "A pre-merge-vs-post-merge diff around the tests-only Task-16 "
        "merge is NOT a control: both sides already contain all "
        "feedback code.",
        "section F, TWO-WAY HOOKS"),

    # -- Phase-5b dormant two-way activation set (design D4) -------------------
    NestGate(
        "P5B", "feedback1_child_region_dry_mass", "structural", None,
        "feedback=1 conserves child-region dry mass on the parent; "
        "section F registers the conservation contract without a numeric "
        "tolerance -- the number is registered by the Phase-5b plan "
        "before activation, never improvised here.",
        "section F, TWO-WAY HOOKS"),
    NestGate(
        "P5B", "ht_coarse_bookkeeping", "exact", None,
        "ht_coarse bookkeeping reproduced (parent-terrain save/restore "
        "around feedback, WRF v4.6.1 semantics).",
        "section F, TWO-WAY HOOKS"),
    NestGate(
        "P5B", "even_ratio_d02_branch_in_scope", "structural", None,
        "Even-ratio (d02, nri=4) branch explicitly in scope: copy_fcn's "
        "EVEN branch cell-averages 1/(nri*nrj) exactly like the odd "
        "branch (interp_fcn.F:1565-1654); copy_fcnm/:1806 and "
        "copy_fcni/:1888 use SW-corner nearest-neighbor for "
        "masked/integer fields.",
        "section F, TWO-WAY HOOKS"),
)


def gates_for(milestone: str) -> tuple[NestGate, ...]:
    """All gates registered for one ladder rung, in ledger order."""
    if milestone not in MILESTONES:
        raise KeyError(f"unknown milestone {milestone!r}; "
                       f"expected one of {MILESTONES}")
    return tuple(g for g in NEST_GATES if g.milestone == milestone)


def gate(milestone: str, metric: str) -> NestGate:
    """Look up one gate record; KeyError if it is not in the ledger."""
    for g in gates_for(milestone):
        if g.metric == metric:
            return g
    raise KeyError(f"no gate {metric!r} registered for {milestone}")


def _check_table() -> None:
    """Ledger integrity, enforced at import (and re-pinned in tests)."""
    if set(COMPARATORS) != BOUND_KINDS:
        raise ValueError("COMPARATORS must define every bound kind "
                         "exactly (F10 amendment)")
    seen = set()
    for g in NEST_GATES:
        key = (g.milestone, g.metric)
        if key in seen:
            raise ValueError(f"duplicate gate record {key}")
        seen.add(key)
        if g.milestone not in MILESTONES:
            raise ValueError(f"unknown milestone in {key}")
        if g.kind not in BOUND_KINDS:
            raise ValueError(f"unknown bound kind {g.kind!r} in {key}")
        if (g.kind in NUMERIC_KINDS) != (g.threshold is not None):
            raise ValueError(f"threshold/kind mismatch in {key}")
        if not g.convention or not g.anchor:
            raise ValueError(f"empty convention/anchor in {key}")
        for name in _EVIDENCE_FIELDS:
            value = getattr(g, name)
            valid = (value is None
                     or isinstance(value, str) and bool(value)
                     or name == "expected_samples"
                     and isinstance(value, int) and not isinstance(value, bool)
                     and value >= 0)
            if not valid:
                raise ValueError(f"invalid evidence pin {name} in {key}")
        if g.milestone in {"N5", "N6"} and g.kind != "diagnostic":
            missing_pins = [name for name in _EVIDENCE_FIELDS
                            if getattr(g, name) is None]
            if missing_pins:
                raise ValueError(
                    f"N5/N6 blocker {key} lacks evidence pins {missing_pins}")
    missing = set(MILESTONES) - {g.milestone for g in NEST_GATES}
    if missing:
        raise ValueError(f"milestones without gates: {sorted(missing)}")
    assert {f.name for f in fields(NestGate)} == {
        "milestone", "metric", "kind", "threshold", "convention", "anchor",
        "evaluator_commit", "mask_hash", "baseline_hash", "cadence",
        "expected_samples"}


_check_table()
