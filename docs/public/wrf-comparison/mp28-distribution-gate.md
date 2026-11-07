# mp_physics = 28: the distribution gate, declared before its run

**Status of this document at the moment it was first committed: DESIGN
ONLY, AWAITING OWNER APPROVAL. Not one number below the line
"MEASUREMENTS" existed when the statistic, the margins and the verdict
rule were written, and unlike the two declarations before it, this one
does not fire on commit: no build starts and no run is scheduled until
the owner approves the declaration.** The one amendment permitted above
the MEASUREMENTS line after this commit is a single line recording that
approval. Nothing else moves, approval or not.

**Approval recorded: the owner approved this declaration as committed at
`0d69a648`, 2026-08-03; the runs fire under it unchanged.**

This is the third declaration in this lane, and it is the one the second
prescribed. The record it stands on:

* `mp28-matched-trajectory.md`: **HOLD, V3 failed**, a closed record.
  Its per-row 3.0× amplification bound, chosen blind, was failed by
  WOOF at 7 of 197 rows and then failed by WRF against its own
  single-flag recompilation at 17 of 195 rows, worst ratio 808:
  non-diagnostic on the long run, proven by its own control.
* `mp28-shortwindow-gate.md`: **INCONCLUSIVE, by its own §5 rule**, a
  closed record. On the pre-decorrelation window the same per-row 3.0×
  was passed by WOOF at 0 of 140 rows, worst 2.195, and failed by the
  identical control at 8 of 140 rows, worst 4.966. The condition voided
  itself a second time, at a second amplitude.
* Its closing section: *"State the amplification condition on the
  distribution of ratios, not on every row, declare the median bound
  and a percentile bound in advance, keep the binding control exactly as
  it is here, and keep the fields, window and screens of this document.
  The control has now voided a per-row 3× twice; that constant has been
  measured into retirement."*

This document is that declaration. Neither closed record is reopened by
anything here.

## 1. Everything inherited, by reference

The case, the six run slots, the control quartet, the gate fields, the
window and the screens are §1–§4 of `mp28-shortwindow-gate.md`,
unchanged and consumed by reference:

* the same banked case (`input_sounding` byte-for-byte, namelists from
  the banked `make-namelists.sh`, WRF v4.6.1 tarball sha256
  `b8ec11b240a3cf1274b2bd609700191c6ec84628e4c991d3ab562ce9dc50b5f2`,
  configure option 32 GNU serial, the two flag variants of
  `configure-vec.txt` / `configure-novec.txt`);
* the same runs: WRF builds A ("vec") and B ("novec", the control),
  each at mp=8 and mp=28, continuous 0 → 2400 s with history every 60 s
  from 1800 s; WOOF initialised from build A's t = 1800 s history frame
  via `run_arwen.py --restart-from`, 50 steps × 12 s, each configuration
  run twice into separate directories;
* the same statistic per cell, M8: `d(F, s) = ||A − W||₂ / ||W||₂`;
* the same 14 gate fields (`U V W PH MU T QVAPOR QCLOUD QRAIN QICE
  QSNOW QGRAUP QNRAIN QNICE`); `RAINNC` and the mp28-only fields
  published, never gated;
* the same screens, binding as before: **G0** (step-0 installation
  screen, tested slot, ≤ 1.0e-8 or the run is void), **G2** (dual-run
  byte screen, one re-run allowance), **G3** (finite everywhere;
  `QNWFA`/`QNIFA` inside the terminal clamp band);
* the same binding control: **WRF build B in the tested slot against
  build A**, unmodified source, one optimization flag, the yardstick
  for what an unimpeachably correct implementation difference does to
  this exact statistic on this exact window.

What changes is only the amplification condition, exactly as the second
gate's closing section prescribed.

## 2. The statistic: declared before any run

Per arm, the ratio rows of the prior gates' G1 convention, unchanged:

* rows are `r(F, s) = d_mp28(F, s) / d_mp08(F, s)` over the 14 gate
  fields and steps `s ∈ 1..10`, nominally **140 rows per arm**;
* the **tested arm** compares WOOF to build A in both schemes; the
  **control arm** compares build B to build A in both schemes: the
  same four WRF runs feed both arms' denominators;
* a row with `d_mp08 = 0` and `d_mp28 = 0` carries no information and
  drops out; a row with `d_mp08 = 0` and `d_mp28 > 0` is **+∞** and
  participates in the order statistics, sorting above every finite
  value. No other row leaves the set. Each arm's row count is
  published; a shortfall from a missing frame or field voids the gate
  (G0 already screens the installation that would cause one).

Two order statistics per arm, defined exactly:

* **median**, ascending sort; the mean of the two middle order
  statistics for even n, the middle one for odd n;
* **p95**, nearest-rank: the order statistic at rank `ceil(0.95·n)`,
  1-based, ascending.

A statistic that lands on +∞ is +∞. The reference implementation of
both definitions is `tools/mp28_matched/derive_distribution_margins.py`
(`order_statistics`), committed with this document; the gate evaluator
transcribes it.

## 3. The gate: relative to the control, margins fixed here

* **D1: median.** `median_tested ≤ 1.7 × median_control`.
* **D2: tail.** `p95_tested ≤ 2.1 × p95_control`.

**Where 1.7 and 2.1 come from, and where they do not.** Both margins
are derived from the CONTROL arm of the second gate's committed receipt
(WRF against its own recompilation, public, and containing no WOOF
number) by `tools/mp28_matched/derive_distribution_margins.py` at its
pinned seed (20260802, 20 000 replicates): field-block bootstrap the
control's 140 rows (resampling whole fields, because the 10 steps
within a field are serially correlated), form the ratio of two
independent replicates of each statistic (the spread two *equally
benign* arms would show against each other) and take the 99th
percentile: **1.663** for the median statistic, **2.048** for the p95
statistic. The margins are those values rounded up to one decimal.
They encode "indistinguishable from a correct implementation
difference, within the sampling noise of this row set", and nothing
looser.

**Why the relative form survives the objection that every prior number
is public.** It is public: anyone choosing an absolute bound today
knows WOOF's worst short-window ratio was 2.195 and the control's was
4.966, and no absolute constant chosen with those numbers in hand can
be called blind. The relative form does not use them. Its bound is
`M × median_control` and `P × p95_control` measured on the **new** run
set: numbers that will not exist until the runs fire, on fresh builds,
on different hardware. There is nothing to tune toward: the only way
the tested arm passes is by actually being no worse than WRF's own
benign recompile on data neither arm has seen. The margins themselves
are fixed above from control-side noise alone, and their derivation is
a committed, seeded script rather than a judgment call.

**Disclosure, so nobody finds it later and calls it concealed:** on the
second gate's run set, closed, public, and not evidence for this gate,
the declared rule would have read median 0.9492 vs 1.0135 (ratio
0.937 ≤ 1.7) and p95 1.1198 vs 3.1083 (ratio 0.360 ≤ 2.1). This gate's
verdict reads only its own fresh runs.

**Control validity, declared in advance.** The relative form has a
denominator, so the gate says now what makes it meaningless:

* `median_control = 0`, or a `p95_control` of +∞, **VOID**
  (non-diagnostic denominator; a certification cannot be bought with a
  broken control any more than it can be lost to one);
* the cross-node replication screen of §5 fails: **VOID** (the
  build/hardware noise floor exceeds what the margin derivation
  assumed).

## 4. Realizations per arm

**n = 1 fresh trajectory set per arm decides the verdict.** Both codes
are deterministic per build (the second pass measured bit-identical
executables and bit-identical Thompson tables per flag set, and
byte-identical WOOF duplicates) so repeated runs of the same bytes
are not realizations and add no information. Variation across builds
and machines is measured instead of simulated, by two declared
replications that are screens, not extra draws:

* the **WOOF dual-run** (G2): each configuration twice, byte-compared,
  the corruption detector on a card with no ECC;
* the **WRF cross-node replication**: the full WRF quartet is built and
  run independently on both weather nodes from the banked recipes. The
  gate set is **test host 1's quartet, fixed here in advance**.
  test host 2's quartet re-computes the control arm's median and p95; if
  either differs from test host 1's by more than **2 % relative**, the gate
  is VOID. The published cross-pass reproducibility of these cells is
  1.373e-3 relative, so 2 % is ≈15× headroom: a screen that only a real
  environment difference can trip, and one that no amount of tripping
  can convert into a pass.

## 5. The run plan: owned hardware only

No rentals. CPU work on the weather nodes, GPU work on the local
RTX 5090, per standing allocation.

**test host 1 and test host 2 (CPU, independently, same steps).**
Work root `$T` (only the root path may differ from the banked scripts,
as before). Stage the pinned tarball, `input_sounding`, `mknml.sh`;
then, in order, the four banked scripts from
`docs/public/receipts/mp28-shortwindow-gate/`:

    bash build-vec.sh      # WRF-4.6.1,      -O2 -ftree-vectorize -funroll-loops
    bash build-novec.sh    # WRF-4.6.1-novec, -O2 -fno-tree-vectorize
    bash stage-sw.sh       # ideal.exe ICs: runs/ic-mp08, runs/ic-mp28
    bash run-sw-wrf.sh     # the quartet: sw-wrf-{,novec-}mp{08,28}

Wall measured by the second pass: vec 597 + 216 s, novec 823 + 373 s,
plus ICs, under 40 minutes a node, ~2.2 GB a node, both nodes in
parallel. Provenance per §6 before anything moves off-node.

**local RTX 5090 (WOOF + the gate).** test host 1's `runs/` tree is
transferred with hashes verified at both ends, then:

    bash run-sw-arwen.sh   # 4 runs: mp{8,28} × duplicates, ~2 min GPU total
    python tools/mp28_matched/distribution_gate.py \
        --runs $T/runs --node2-runs $T/runs-node2 \
        --out $T/out/distribution-gate.json

`distribution_gate.py` does not exist yet and is written only after
this declaration is approved; it transcribes this document's constants
the way `shortwindow_gate.py` transcribed the second's, and evaluates
G0, G2, G3, D1, D2, the control validity conditions and the cross-node
screen in one pass. Its receipt, the runs' SHA256SUMS, and the node
provenance are committed beside this document under
`docs/public/receipts/mp28-distribution-gate/`.

GPU sessions here are short and bursty; nothing in this plan needs a
detached long run.

## 6. Provenance requirements

Section 6 of `mp28-shortwindow-gate.md` applies unchanged and by
reference: the tarball, `module_mp_thompson.F` / `module_mp_radar.F`
and `CCN_ACTIVATE.BIN` pins; both builds' executable hashes; each
build's generated Thompson table hashes; both `wrfinput_d01` hashes;
compiler, glibc and netCDF versions; WOOF source snapshot commit, CuPy
version, driver and device; the FP32-subnormal (FTZ) disclosure for
sm_120. Added for this gate: both weather nodes' identities and
toolchains, recorded separately, and the hash-verified transfer
manifest for every hop the run data takes.

## 7. Verdict rule: declared before any run

| outcome | condition | consequence |
|---|---|---|
| **CERTIFY** | G0, G2, G3 hold; both arms carry full row sets; control valid; cross-node screen passes; **D1 and D2 both hold** | the registry transition of §8, exactly and only it |
| **HOLD** | screens and control all valid, and D1 or D2 fails | the failing statistic and both arms' values are published; the scheme stays `implemented-unverified`; the first document's ship recommendation loses its short-window support and says so |
| **VOID / INCONCLUSIVE** | G0 fails; or G2 cannot be brought clean within its one declared re-run; or a row-set shortfall; or the control is degenerate (§3); or the cross-node screen fails (§4) | published as such. **No fourth per-window gate is declared without a design review of the whole gate family**: three declarations is enough to conclude the instrument, not the luck, needs examining. The real-data tier (aerosol lateral BC, then a matched real-data run with decay tables) proceeds regardless |

No bound, margin, rank definition or row-set rule above moves after a
number exists. The git history of this file is the receipt.

## 8. What CERTIFY moves, exactly

* `woof/physics_registry_v2.json`,
  `/components/microphysics/options/thompson-aerosol-mp28`:
  `maturity` moves `implemented-unverified` → **`wrf-matched-run-candidate`**
  (rank 2 → 5). That rung is defined in the registry's own ladder as
  "executable and gated, with a ratified reference comparison, and
  deliberately not the default: the next candidate for a full matched
  run", which is precisely, and only, what a CERTIFY here evidences.
  NSSL-2/MP18 is the precedent occupant of the rung.
* The same entry's `warnings[0]` is rewritten to cite this document and
  its receipt, and to keep saying (because it stays true) that no
  matched REAL-DATA or NESTED trajectory exists, that both remain
  blocked on an aerosol lateral boundary condition, and that the scheme
  is never a default. `scientific_evidence` stays `"none"`: the ladder
  measures agreement with WRF, not skill against observations.
* `extensions.column_oracle_evidence.forecast_trajectory_comparison`
  gains this document and receipt beside the two closed records.
* `docs/public/PHYSICS.md`: the MP28 maturity line moves to
  `wrf-matched-run-candidate`, pointing here.
* Every test that pins mp28's maturity string moves in the same commit,
  named in its message.

What CERTIFY does **not** move: mp=8's tier; the first document's HOLD;
the second's INCONCLUSIVE; the `wrf-matched-run` requirements (real
case, decay tables, aerosol LBC), all unchanged. A certification here
is the statement "on the declared window, WOOF's mp28-vs-mp8 behaviour
is statistically no worse than WRF's own benign recompile", made by a
rule fixed before the runs, and it is not a statement about anything
else.

---

# MEASUREMENTS

*Nothing below this line existed at the design commit. Everything above
it is byte-unchanged from that commit except the §0 approval line.*

## 9. Provenance

**Weather nodes.** `test host 1` and `test host 2`, each 24 cores,
Ubuntu 26.04 LTS, gfortran/gcc 15.2.0 (Ubuntu 15.2.0-16ubuntu1), glibc
2.43, netCDF-Fortran 4.6.2: a newer toolchain than either prior pass
(13.3.0 / 2.39), disclosed, equality never required. Both nodes ran the
em_les oracle campaign's WRF jobs concurrently (94–99 % of one core
each, snapshots in each `chain.log`); nothing was stopped or displaced.
Tarball sha256 equals the §1 pin on both nodes;
`phys/module_mp_thompson.F`, `phys/module_mp_radar.F` and
`run/CCN_ACTIVATE.BIN` equal the inherited §6 pins in both trees on both
nodes. Three adaptations to the banked scripts, the declared work-root
change plus two toolchain-plumbing fixes this OS generation forces (a
netCDF shim prefix and a GCC-15 legacy-C dialect flag on `SCC`; the
em_les lane's committed recipe applies the same two fixes to the same
nodes), are recorded in full, with the executed scripts, in
`docs/public/receipts/mp28-distribution-gate/`. `SFC`, both `FCOPTIM`
variants, configure option 32 and the Fortran source are byte-unchanged
from the banked recipes; the four run namelists are byte-identical to
the second gate's committed ones (`a2eca705…`, `a355dd83…`).

**Artifacts.** vec `ideal.exe` `3814e5a7…` / `wrf.exe` `a7f4209a…`,
novec `325845ac…` / `6bcc015e…`, all four differ from both prior
passes, as a compiler generation change must. Thompson tables:
`qr_acr_qg_V4` and `qr_acr_qsV2` are **bit-identical to the published
hashes** in both flag variants; `freezeH2O.dat` differs per variant
(`05497b0e…` vec, `256259ef…` novec), the freezing-table derivation is
libm-heavy and gfortran 15 rounds it differently, the same disease this
lane measured twice elsewhere today (the Exner re-pin and the
derived-constant canonicalization). Three table sets in play, each
model on its own, disclosed as before. ICs: `wrfinput_d01` mp=8
`231a2dc8…`, mp=28 `41fc5aad…`.

**Cross-node determinism, measured.** All eight wrfouts (both builds,
both schemes) are **bit-identical between the two independently
building and running nodes**, so the control arm's statistics agree
across nodes to exactly 0.0 relative and the §4 screen passed with all
of its 2 % slack unused.

**Local node (WOOF + evaluation).** The RTX 5090 (sm_120, no ECC),
driver 610.74 / cc 12.0, CuPy 14.0.1, numpy 2.2.6, Python 3.13.7, WOOF
source = this lane at `d66ae147`, tracked tree clean; the pinned table
set under its committed hashes; the canonical derived-constant asset
verified against both of its pins at import, with this host's recorded
libm drift (`t_Nc`) inert because the canonical bytes are what load. A
convective-boundary-layer job was cycling on the card (48 % util at
launch); the four gate runs (2.1–2.9 s each) ran beside it, and every
transferred input was hash-verified against the producing node's
manifest before use. Node walls: builds 479+190 s (test host 1) /
889+185 s (test host 2), quartet 447 s / 453 s.

## 10. Screens

* **G0**, all 28 step-0 rows exactly `0.0`, third staging in a row.
* **G2**: 11/11 frames byte-identical, both schemes, both duplicates.
* **G3**: 0 non-finite values, 0 clamp-band violations.
* **Row sets**: 140/140 cells present in both arms; 0 both-zero drops;
  0 infinite ratios.
* **Control validity**: median finite and positive; p95 finite.
* **Cross-node**: 0.0 relative on both statistics (bound 2 %).

## 11. The two arms, and the verdict

| statistic | tested (WOOF) | control (WRF novec) | bound | condition |
|---|---|---|---|---|
| median ratio | **0.9346** | 0.4833 | ≤ 1.7 × 0.4833 = **0.8216** | **D1 FAILS** |
| p95 ratio | **1.1433** | 1.7462 | ≤ 2.1 × 1.7462 = 3.6670 | D2 passes |
| worst row | 2.458 (`QNRAIN`, step 10) | 5.992 (`QRAIN`, step 2) | - | published |
| rows over the retired 3.0× | 0 of 140 | 4 of 140 | - | published |

**Outcome: HOLD, by the rule in §7.** D1 fails: the tested arm's median
is not within 1.7× of the control's on this run set. No screen is
invoked to soften it and no bound moved after the numbers arrived. Per
§7, the scheme stays `implemented-unverified`, the registry does not
move, and the first document's ship recommendation loses its
short-window support: recorded there in a dated addendum.

**What is measured regardless, with the verdict letter unchanged.**
The tested arm's median is **0.93**, below 1: on the identical
statistic, WOOF's mp=28 again sits *closer* to WRF's mp=8-to-mp=28
behaviour than to any amplification, its worst row is again `QNRAIN`
shaped, and it clears the retired per-row 3.0× at 0 of 140 rows while
the control breaks that condition a third time (4 rows, worst 5.992,
the distribution form did exactly what it was declared to do: it
survived a control that voids per-row gates, and returned a decisive
verdict). What moved is the yardstick's centre: the control's median
halved against its public prior (0.4833 here vs 1.0135 on
gfortran 13.3) while WOOF's held (0.9346 vs 0.9492). The margin M was
derived from the old control's *sampling noise* around a centre that
this toolchain then shifted by 2×: a sensitivity of the
relative-to-control construction to the control's own toolchain, which
is a measured finding about the instrument, published here as one. The
verdict is still the word in bold; evidence that the rule was
imperfectly designed is a matter for a future declaration, never for
this one's outcome.

**Standing consequences.** Maturity label unchanged
(`implemented-unverified`); registry untouched; both prior records
unchanged. This was a decisive HOLD, not a void, so the §7 no-fourth-
gate-without-review clause is not triggered, but any successor
declaration must confront what this one measured: a control centre that
moves with the compiler. The obvious repair candidates (declaring a
control-median validity band in advance, or margins derived from
control replicates across toolchains rather than within one) are a
future design's choices, made before ITS runs. The real-data tier
(aerosol lateral BC, then a matched real-data run with decay tables)
remains the path that retires this whole gate family, and proceeds
regardless.
