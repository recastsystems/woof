# mp_physics = 28: the short-window gate, declared before its run

**Status of this document at the moment it was first committed: DESIGN ONLY.
Not one number below the line "MEASUREMENTS" had been produced when the
fields, the thresholds and the verdict rule were written.** The runs this
document gates had not happened; the node they run on had just been
provisioned and carried no WRF build and no WOOF-ArWen checkout.

This is the successor gate that `mp28-matched-trajectory.md` prescribes in
its closing section: *"Declare the gate on the **short window** and use the
long run only to publish the implementation floor and the WRF-against-itself
control."* That document's short-window test (its §11) was added after its
long runs were read and is labelled post-hoc wherever it appears; its own
receipt says so in its `note` field. A post-hoc pass is evidence, but it is
not a pre-registered gate, and the difference is the entire reason that
lane's V3 failed: a condition chosen blind was chosen wrong, and the accurate
repair is not to reuse the post-hoc window's numbers, it is to declare the
gate first and run second. This file is that declaration.

It also closes the one instrumentation gap the first pass named (its §11):
*"`U`, `V`, `PH` and `MU` are absent from this table because the WOOF-ArWen frame
writer does not dump them; that is a gap in the instrumentation, not a
result, and it is the one thing a repeat of this test should fix first."*
The frame writer dumps them now, in the same commit as this design, so the
comparison below covers every prognostic both models carry.

The first document's declared verdict (**HOLD, V3 failed**) is a closed
record and is not rewritten by anything here. Its rule was committed before
its runs, its gate failed by that rule, and its control proved the failing
condition non-diagnostic; all three facts stand. This gate does not reopen
it. What this gate does is supply, with the ordering done right, the
evidence the first document's closing recommendation leaned on.

Provenance chain of the first pass: design `f87ca87f`, measurements
`2af4087b`, evidence bank `41fc2563`. The fleet those runs executed on is
gone; every run below is re-derived on a fresh node from the banked recipe
in `docs/public/receipts/mp28-matched-trajectory/`.

---

## 1. The case: unchanged, by reference

Identical to §2 of `mp28-matched-trajectory.md`, consumed from the banked
receipts rather than re-decided: the same `input_sounding` byte-for-byte,
namelists produced by the banked `make-namelists.sh`, the same WRF v4.6.1
release tarball (sha256
`b8ec11b240a3cf1274b2bd609700191c6ec84628e4c991d3ab562ce9dc50b5f2`), the
same `configure` recipe (option 32, GNU serial, nesting 0) with the two
optimization-flag variants recorded in `configure-vec.txt` and
`configure-novec.txt`. A doubly periodic single domain, 120 × 120 × 40 at
dx = 2 km, unsheared WK82 sounding, one 3 K warm bubble, microphysics the
only physics, dt = 12 s, `time_step_sound = 6`.

This is a different physical machine than the first pass ran on. The
executable and lookup-table hashes are republished under MEASUREMENTS and
compared against the first pass's; **equality is not required** (the
Thompson tables are already published as build-dependent) and whichever
way it lands is disclosed, not absorbed.

## 2. The runs

| run | model | build | mp | span |
|---|---|---|---|---|
| `sw-wrf-mp08` | WRF v4.6.1 | A ("vec": `-O2 -ftree-vectorize -funroll-loops`) | 8 | continuous 0 → 2400 s, history every 60 s beginning at 1800 s |
| `sw-wrf-mp28` | WRF v4.6.1 | A | 28 | same |
| `sw-wrf-novec-mp08` | WRF v4.6.1 | B ("novec": `-O2 -fno-tree-vectorize`, identical source) | 8 | same, the control |
| `sw-wrf-novec-mp28` | WRF v4.6.1 | B | 28 | same, the control |
| `sw-arwen-mp08`, `sw-arwen-mp08-b` | WOOF-ArWen | CUDA / sm_120, FP32 | 8 | from build A's t = 1800 s history frame, 50 steps × 12 s, frame every 5 steps |
| `sw-arwen-mp28`, `sw-arwen-mp28-b` | WOOF-ArWen | same | 28 | same |

WRF reaches the mature state by running continuously from t = 0 (no
restart file is involved) and WOOF-ArWen reads build A's first history frame as
its initial condition via `run_arwen.py --restart-from` (prognostics from
the frame, time-invariant base state from `wrfinput_d01`). Each WRF build
generates and uses its own Thompson lookup tables; WOOF-ArWen uses its pinned
shipped set. Three table sets again, as before, disclosed.

Each WOOF-ArWen configuration runs twice into separate directories: the card has
no ECC, and the byte comparison of independent runs is the corruption
detector.

The window is 1800 → 2400 s because that is the regime the first pass
measured as pre-decorrelation: perturbations grow with a 10–15 minute
e-folding time on this case, so 600 s from a common mature state compares
two answers rather than two samples. The choice of window is inherited from
that measurement, not from any number this gate will produce.

## 3. Statistic and fields

The statistic is **M8 as declared in the first document**: the normalised
RMS field difference `d(F, s) = ||A − W||₂ / ||W||₂` over the whole domain,
at each compared frame `s = 0..10` (t = 1800 + 60·s seconds).

**Gate comparison set: 14 fields:** `U`, `V`, `W`, `PH`, `MU`, `T`,
`QVAPOR`, `QCLOUD`, `QRAIN`, `QICE`, `QSNOW`, `QGRAUP`, `QNRAIN`, `QNICE`.
The first four are the ones the first pass could not measure; they are in
the gate precisely because their absence was named as the repeat's first
job.

**Published, never gated:**

* `RAINNC`: WOOF-ArWen's accumulator restarts at zero while WRF's carries
  1800 s of rain, so the statistic compares two different quantities. The
  first pass's gate machinery nonetheless counted its rows; this one
  excludes them from the verdict in both the tested and control slots and
  publishes them anyway.
* `QNCLOUD`, `QNWFA`, `QNIFA`: mp=28-only, so no mp=8 denominator exists.
  Published as model-to-model divergences; the aerosol fields being the
  best-agreeing 3-D fields was the first pass's headline and the repeat
  reports the same columns.

## 4. The gate: declared before any run

* **G0: installation screen.** At step 0, `d(F, 0) ≤ 1.0e-8` for every
  gate field, both schemes, tested slot. Both sides of step 0 are the same
  file read back; a residual above float32 round-off means the installation
  is broken and the run is **void**, not failing.
* **G1: no scheme-level amplification** (the condition a shipping decision
  reads). For every gate field `F` and step `s ∈ 1..10`:
  `d_mp28(F, s) / d_mp08(F, s) ≤ 3.0`.
  A row with `d_mp08 = 0` and `d_mp28 = 0` carries no information and drops
  out; a row with `d_mp08 = 0` and `d_mp28 > 0` is an infinite ratio and
  **fails**. Nominally 140 rows. The threshold is the first document's V3
  constant, applied in the one regime its own control showed it can
  discriminate in.
* **G2: dual-run byte screen.** All 11 frames byte-identical between each
  WOOF-ArWen configuration and its duplicate. On a mismatch the configuration is
  re-run once; a second mismatch withholds the result and the hardware
  finding is the report.
* **G3: finite and bounded.** Every field of every WOOF-ArWen frame finite in
  both schemes; `QNWFA` within `[11.1e6, 9999.0e6]` and `QNIFA` within
  `[5.0e3, 9999.0e6]` in every mp=28 frame: the same terminal clamp-band
  constants the first pass's V4 applied to the same fields. (No depletion
  -trend condition here: 600 s is too short a lever arm for the trend test
  the 7200 s run already passed.)
* **Control: diagnosticity, binding on the verdict.** The identical
  statistic and the identical G1 condition with **WRF build B in the tested
  slot** against build A, over the same window. The control's step-0
  divergence is 1800 s of accumulated one-flag drift: that is the point:
  it shows what an unimpeachably correct implementation difference does to
  this exact statistic on this exact window. **If the control fails G1, G1
  is non-diagnostic here and the outcome is INCONCLUSIVE**: the same
  reading the long run's control earned, applied symmetrically and decided
  in advance this time.

## 5. Verdict rule: declared before any run

* **PASS**: G0 through G3 all hold and the control passes G1. Consequence:
  this document populates the short-window slot the first document
  prescribed. The scheme remains `implemented-unverified`, never a default;
  no maturity tier moves on this evidence.
* **FAIL**: any of G0, G1, G3 fails for WOOF-ArWen while the control passes G1.
  The failing field and step are named. The first document's closing
  recommendation rested in part on its post-hoc short window; a FAIL here
  withdraws that support and says so.
* **INCONCLUSIVE**: the control fails G1, or G2 cannot be brought to a
  clean state within its declared re-run allowance. Inconclusive is a valid
  outcome and is published as one.

No bound above moves after a number exists. The git history of this file is
the receipt.

## 6. Provenance requirements

MEASUREMENTS must record, before any verdict is read: the tarball sha256
(must equal the pin in §1); sha256 of `phys/module_mp_thompson.F` and
`phys/module_mp_radar.F` in both trees (must equal
`fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802` and
`aa99da858be41efa579966680708d230123a7417560af0eb2e24f4c94e253688`);
`CCN_ACTIVATE.BIN` sha256 (must equal
`f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd`); both
builds' `ideal.exe`/`wrf.exe` hashes; each build's generated Thompson table
hashes; both `wrfinput_d01` hashes; compiler, glibc and netCDF versions;
and for WOOF-ArWen the source snapshot commit, CuPy version, driver and device.
sm_120 flushes FP32 subnormals to zero in all arithmetic, as published for
the first pass; the same disclosure applies unchanged.

---

# MEASUREMENTS

*Nothing below this line existed at the design commit. Everything above it
is byte-unchanged from that commit.*

**Scope correction, 2026-10-02.** The frozen design's sm_120 flushing claim
was corrected later: CuPy's compile route appended `-ftz=true` for these runs.
It is a toolchain property, not an unavoidable property of sm_120 hardware;
see [HARDWARE.md](../HARDWARE.md#fp32-subnormals-and-gpu-model-caveats).
The original design, measurements and INCONCLUSIVE verdict remain unchanged.
The comparisons are code verification against WRF, not validation against
observations.

## 7. Provenance

**Node.** A second RTX 5090 (not the card the first pass ran on) driver
580.126.09, cc 12.0, CUDA 13.0, on Ubuntu 24.04.4, gcc/gfortran 13.3.0
(Ubuntu 13.3.0-6ubuntu2~24.04.1), glibc 2.39, Python 3.12.3, NumPy 2.5.1,
netCDF4 1.7.4, CuPy 14.1.1, the same compiler, glibc and CuPy versions as
the first pass, on different hardware.

**WRF.** Tarball sha256 equals the §1 pin. `phys/module_mp_thompson.F` and
`phys/module_mp_radar.F` equal the §6 pins in both trees;
`run/CCN_ACTIVATE.BIN` equals the shipped table's pin. Configure records
match `configure-vec.txt`/`configure-novec.txt` field for field.

| artifact | this pass | first pass | |
|---|---|---|---|
| vec `ideal.exe` | `160a50a5…` | `160a50a5…` | **bit-identical** |
| novec `ideal.exe` | `57271e95…` | `57271e95…` | **bit-identical** |
| vec `wrf.exe` | `317cf098…` | `cd611280…` | differ |
| novec `wrf.exe` | `a9835fbf…` | `e81d032f…` | differ |
| vec tables (`freezeH2O`, `qr_acr_qg_V4`, `qr_acr_qsV2`) | `c7a01fa7…`, `441bd836…`, `910fb31d…` | same three | **bit-identical** |
| novec tables | `a054a3a4…`, `bcef0ef3…`, `5e7438b6…` | same three | **bit-identical** |

The `wrf.exe` difference has a measured cause: the binary embeds its build
root (`strings` finds the node's work-root path once; the first pass built under a
different root). `ideal.exe` carries no such bytes and reproduced exactly.
All six regenerated Thompson tables are bit-for-bit the first pass's:
table generation is deterministic per flag set on this compiler, which
retires the residual worry that the published table hashes were
node-luck. Three table sets in play again (vec, novec, WOOF-ArWen's pinned
set), each model on its own, as declared.

**Initial conditions.** `ideal.exe` (vec) per banked namelists:
`wrfinput_d01` mp=8 `598911a4…`, mp=28 `c36ce7f8…`.

**WOOF-ArWen.** `git archive` of this document's design commit (`arwen-src`
sha256 `180fd880…`), run in-tree; CCN table loaded under its pinned hash.

**Cost.** WRF walls: vec 597 s (mp=8, includes its build's one-time table
computation) / 216 s (mp=28); novec 823 s / 373 s (same asymmetry, same
cause). WOOF-ArWen 50-step runs: 57.1 s (first, includes kernel JIT), then
2.7 / 11.5 / 3.3 s. Node total for the registered comparison, from first
`ideal.exe` to gate receipt: under 40 minutes wall.

## 8. Screens

* **G0, exactly zero.** Every one of the 28 step-0 rows (14 gate fields ×
  both schemes) is exactly `0.0`, including `T`, whose split/recombine
  round trip lands exact from a history frame, as the first pass's take 3
  also measured, and including the four newly instrumented fields.
* **G2: dual-run byte screen.** 11/11 frames byte-identical, both
  configurations. No silent corruption.
* **G3: finite and bounded.** 0 non-finite values; 0 clamp-band
  violations.

## 9. The measurements

Normalised RMS field difference, WOOF-ArWen vs WRF build A, `mp=8 | mp=28`,
at the compared frames (frame = 60 s):

| frame | t (s) | `U` | `V` | `W` | `PH` | `MU` | `T` | `QVAPOR` | `QCLOUD` | `QNWFA` | `QNIFA` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1800 | 0 \| 0 | 0 \| 0 | 0 \| 0 | 0 \| 0 | 0 \| 0 | 0 \| 0 | 0 \| 0 | 0 \| 0 |, \| 0 |, \| 0 |
| 1 | 1860 | 2.081e-02 \| 2.049e-02 | 2.081e-02 \| 2.049e-02 | 1.800e-02 \| 1.797e-02 | 6.458e-03 \| 6.479e-03 | 3.116e-03 \| 3.110e-03 | 4.911e-05 \| 4.551e-05 | 2.878e-04 \| 2.509e-04 | 6.654e-02 \| 8.405e-02 |, \| 5.530e-04 |, \| 3.558e-04 |
| 2 | 1920 | 2.609e-02 \| 2.551e-02 | 2.608e-02 \| 2.551e-02 | 2.186e-02 \| 2.021e-02 | 1.193e-02 \| 1.192e-02 | 4.293e-03 \| 4.219e-03 | 1.010e-04 \| 9.621e-05 | 4.944e-04 \| 3.968e-04 | 1.059e-01 \| 9.331e-02 |, \| 9.654e-04 |, \| 6.394e-04 |
| 5 | 2100 | 4.929e-02 \| 4.697e-02 | 4.930e-02 \| 4.702e-02 | 5.559e-02 \| 4.458e-02 | 2.421e-02 \| 2.450e-02 | 7.790e-03 \| 7.904e-03 | 2.670e-04 \| 2.581e-04 | 7.742e-04 \| 7.135e-04 | 1.573e-01 \| 1.191e-01 |, \| 2.275e-03 |, \| 1.267e-03 |
| 10 | 2400 | 7.252e-02 \| 7.157e-02 | 7.251e-02 \| 7.155e-02 | 1.145e-01 \| 1.184e-01 | 2.941e-02 \| 3.049e-02 | 9.782e-03 \| 9.976e-03 | 3.622e-04 \| 3.871e-04 | 1.265e-03 \| 9.378e-04 | 4.282e-01 \| 2.663e-01 |, \| 8.434e-03 |, \| 1.815e-03 |

Three observations, in evidence order:

1. **The re-derived runs reproduce the dead fleet's numbers to 1.373e-03
   relative, and no tighter.** Over all 48 non-zero cells this window
   shares with the first pass's short-window receipt, the worst relative
   difference is **1.373e-03**, at mp=8 `QICE`, frame 10: 1.0719e-01 in
   the first pass's receipt, 1.0704e-01 here. The `W` cells match to all
   four digits the first pass printed (1.800e-2 \| 1.797e-2 at frame 1,
   1.145e-1 \| 1.184e-1 at frame 10), and the worst G1 row is the same
   row at the same printed ratio (`QNRAIN`, frame 10, 2.195). The
   trajectories are **not** bit-identical across passes (this is a
   different physical GPU and a WRF rebuilt from the tarball) and the
   claim is exactly the measured bound above, nothing rounder.
2. **The four newly instrumented fields found nothing hiding.** `U`, `V`,
   `PH`, `MU` (absent from the first pass) are the tightest-matched
   dynamical fields in the table, and their mp=28/mp=8 ratios sit between
   0.983 and 1.068 (worst `U`/`V`/`PH` ratio 1.037). The instrumentation
   gap is closed and the answer behind it was "the same as everything
   else".
3. **The aerosol fields again have the smallest normalized differences.**
   `QNWFA` is 8.434e-03 and `QNIFA` 1.815e-03 after 50 steps. Smooth fields
   with large background values naturally give small normalized RMS
   differences; the ranking is not evidence that aerosol is carried more
   accurately than condensate.

## 10. The verdict, as declared

| condition | WOOF-ArWen (tested slot) | **CONTROL: WRF build B in the tested slot** |
|---|---|---|
| **G0** installation screen | **PASS**, 28/28 rows exactly 0.0 | (not applicable by design, its step 0 is 1800 s of accumulated drift, published: `W` 7.555e-03 \| 1.631e-02) |
| **G1** no amplification, ≤ 3.0× | **PASS**, 0 of 140 rows over, worst **2.195** (`QNRAIN` at frame 10; runner-up **1.501**, `QNRAIN` at frame 2) | **FAIL**, 8 of 140 rows over, worst **4.966** (`QRAIN` at frame 2) |
| **G2** dual-run byte screen | **PASS**, 11/11 + 11/11 | (GPU-only screen) |
| **G3** finite and bounded | **PASS**, 0, 0 | - |

**Outcome: INCONCLUSIVE, by the rule in §5.** The control fails the
identical condition, so G1 is non-diagnostic on this window, and §5 says
that outcome in advance. It is not softened here: the gate this document
declared cannot certify mp=28 on this evidence, and no bound is being
adjusted after the fact to make it do so.

The control's eight failing rows are `MU` at frames 1, 2 and 9 (3.555,
3.120, 3.108), `QCLOUD` at 2 and 3 (4.093, 4.050), `QRAIN` at 2 (4.966)
and `QNRAIN` at 2 and 8 (4.647, 3.160). Six of the eight sit in the first
three frames, where both of the control's divergences are small
accumulated-drift quantities and their ratio is at its noisiest; and three
of the eight are in `MU`, a field measured here for the first time. The
diagnosis is the long run's, recurring at smaller amplitude: a
**per-row worst-case ratio** is brittle against ratio noise wherever the
denominator is small, even pre-decorrelation. A condition stated on the
distribution (a median and a high percentile, say) would very likely
discriminate here, but that is a third declaration's design choice, made
before ITS runs, not a repair applied to this one after the numbers
arrived.

**What is measured regardless, with the verdict unchanged:** the ratio
statistic asks whether mp=28 adds disagreement beyond mp=8. WOOF-ArWen has no
rows over 3x (worst 2.195), while the control has eight (worst 4.966).
This does not mean WOOF-ArWen is closer to WRF than recompiled WRF is. The ratio
is steadier partly because WOOF-ArWen's mp=8 divergence, its denominator, is larger.
For mp=8, WOOF-ArWen's normalized `W` difference after 60 s (1.800e-02) is about
2.4 times the recompiled WRF difference accumulated over the preceding
1800 s (7.555e-03). For mp=28 the two are similar: 1.797e-02 after 60 s
against 1.631e-02 after 1800 s. These compare different elapsed windows;
they show why the ratio ranking is not a ranking of absolute agreement.
The declared outcome remains **INCONCLUSIVE**.

**Standing consequences.** The first document's HOLD record is unchanged.
Its closing recommendation is neither strengthened nor withdrawn by an
inconclusive outcome: the post-hoc short-window pass it leaned on has now
been reproduced under pre-registration (the same numbers at their printed
precision, the same worst row), but the gate built on those numbers voided
itself through its own control.
The maturity label stays `implemented-unverified`; nothing here moves it.

### What a third declaration should change

State the amplification condition on the distribution of ratios, not on
every row: declare the median bound and a percentile bound in advance,
keep the binding control exactly as it is here, and keep the fields,
window and screens of this document. The control has now voided a per-row
3× twice, at two different amplitudes; that constant has been measured
into retirement.
