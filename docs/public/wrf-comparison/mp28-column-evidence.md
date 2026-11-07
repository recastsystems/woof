# mp_physics = 28 (Thompson aerosol-aware): what is actually measured

Status of this page: **evidence statement, deliberately conservative.**
It records what has been measured against unmodified WRF v4.6.1, what has
only been measured against something weaker, and what has not been measured
at all. Where a residual would not close, the measured number is printed
rather than absorbed into a tolerance.

The maturity label WOOF publishes for this option is
**`implemented-unverified`**. It is **not** `wrf-matched-run-candidate` and
**not** `wrf-matched-run`, which older records spell `validation-candidate`
and `model-validated`. Those are registry labels for WRF comparison
evidence, not validation against observations. The scope is:

* No forecast-scale comparison has qualified mp=28 for either higher rung.
  Three declared comparisons on an idealized doubly periodic warm bubble
  returned [HOLD](mp28-matched-trajectory.md),
  [INCONCLUSIVE](mp28-shortwindow-gate.md) and
  [HOLD](mp28-distribution-gate.md).
* No real-data matched WOOF-versus-WRF run with published decay tables exists.
  The 150-step and 600-step self-consistency runs in section 5 were not WRF
  comparisons; the separate idealized runs linked above were.

The label's evidence is the measurements in this page, not an assertion that
no comparison exists. Of 22 committed WRF column fixtures, 21 of 22 clear
the flat gate on all 23 compared quantities, with no allowance and nothing
held out; the 22nd, `aero-cold-overlap`, misses by a declared divergence
(WOOF does not reproduce WRF's out-of-bounds read of its rain-graupel
collision tables; section 3) and clears the gate when that read is emulated
in a measurement copy of the tree. Section 3 publishes the table field by
field and the history of every residual that closed. The 153-column oracle
of section 3.6 compares every output word bit for bit: outside the columns
where rain meets graupel the port is WRF v4.6.1, word for word, in the
strict and the default build at 20 s and 5 s, and with WRF's out-of-bounds
read reproduced in a measurement copy every word of every column is. These
are component code verification results. The idealized forecast comparisons
above have not produced a declared PASS, and no observation-based validation
is claimed. A clean single-call deck is not a forecast validation.

The idealized comparisons do not establish agreement for real-data or nested
forecasts. Staying finite and inside WRF's clamps for two hours also does not
establish correctness or skill against observations (see section 5).

If you only read one section, read
[What you would get wrong today](#6-what-you-would-get-wrong-today).

---

## 1. Measurement environment

Most original measurements were made on an RTX 5090 on 2026-08-01.
Sections 5 and 6.1 were re-measured on 2026-09-28; the wp08-freeze row and
classic-path comparison changed on 2026-09-23; the per-card rows and stripped
section 6.1 run were re-measured on 2026-09-30, and section 6.1 again on
2026-10-05 for the 2.8.6 accumulator rework. Each update below states its
scope. The 2026-09-28 forecasts were bitwise identical on NVRTC 13.0.48 and
13.4.92, and on an RTX 5070 Ti with NVRTC 13.0.88; that comparison does not
establish identity for every GPU or compiler.

| item | value |
| --- | --- |
| reference physics | unmodified WRF v4.6.1 `phys/module_mp_thompson.F`, commit `d66e442fccc04111067e29274c9f9eaccc3cef28` |
| Fortran | GNU Fortran 13.3.0, `-O2 -ffree-form -ffree-line-length-none`, baseline x86-64 (**no FMA instruction**) |
| GPU | NVIDIA GeForce RTX 5090 (cc 12.0), CUDA runtime 13.0, cupy 14.1.1, nvrtc defaults (`--fmad=true`) |
| host | Ubuntu 24.04.4 LTS on WSL2, Python 3.12.3, NumPy 2.5.1 |
| aerosol table | `CCN_ACTIVATE.BIN`, 35,288 bytes, sha256 `f2b8d391…c82a3dbd`, **redistributed with WOOF** since 2026-08-01 (deviation D9i in `PROVENANCE.md`; it stays outside the classic mp=8 table set, §7) |
| date of measurement | mostly 2026-08-01; sections 5 and 6.1 on 2026-09-28; wp08-freeze and the classic-path comparison on 2026-09-23; per-card rows and the stripped section 6.1 run on 2026-09-30; section 6.1 on 2026-10-05 (RTX 5090, test host 2: Ubuntu, Python 3.13, cupy 14.2.0, NumPy 2.5.3, NVRTC 13.4.92) |
| §5 and §6.1 | re-measured 2026-09-28 on the same RTX 5090, now on Linux (Ubuntu, Python 3.12, cupy 14.2.0, NumPy 2.5.3), once on NVRTC 13.0.48 and once on 13.4.92: every value bitwise identical between the two |

Arithmetic is float32 on the GPU and REAL(4) in the Fortran reference, which
is the comparison that matters; the WOOF kernels raise selected
sub-expressions to double exactly where WRF does and nowhere else.

---

## 2. Evidence classes, strongest first

Not all "it matches" claims are the same claim. This page grades them, and
a reader is entitled to discount the lower classes.

| class | what it is | why it is weaker than the one above | how much of mp=28 rests on it |
| --- | --- | --- | --- |
| **A: committed WRF column fixtures** | A whole column stepped by unmodified `mp_gt_driver`, dumped to CSV, committed, SHA-256'd. The comparison is against WRF's own answer for the complete scheme. | - | 22 scenarios, each a 24-level column with a `before` and an `after` row (48 data rows × 22 numeric fields) plus a 10-field surface file. This is the gate the maturity label rests on. |
| **B: committed scratch-driver Fortran output** | Fortran programs *written for this port* that call WRF's routines directly and tabulate intermediates or scalar functions. The physics is WRF's; the driver, the argument list and the choice of what to print are WOOF's. | A driver bug can make a wrong port look right, and there is no second implementation to catch it. Intermediate rates are also not something WRF itself ever exposes, so the "reference" has never been exercised by anyone else. | The five probe tables (`probe-activncloud.csv` and four siblings), the three per-kernel rate oracles and the five instrumented intermediate tables. |
| **C: host NumPy transcription** | A Python re-reading of the WRF formula, compared against the CUDA kernel. | If the transcription is wrong, both sides are wrong the same way and the test is green. This class can only find CUDA/Python disagreements, never a misreading of WRF. | Bit-exactness gates on individual device helpers and several bound/monotonicity properties. |
| **D: self-consistency, no external reference** | Runs, stays finite, stays inside WRF's own clamps, restores what it must restore. | It compares WOOF to WOOF. A scheme that is uniformly 20% wrong satisfies every one of these. | The whole of G4, the forecast smoke. Everything in §5. |

---

## 3. Class A: the committed WRF column fixtures

**Gate:** each fixture is driven end to end through the shipped adapter
(`woof/core/microphysics_aerosol.py::_apply_thompson_aerosol`) and compared
against WRF's own after-state on 15 column fields (`qv`, `qc`, `qr`, `qi`,
`qs`, `qg`, `ni`, `nr`, `nc`, `nwfa`, `nifa`, `temp`, `effc`, `effi`,
`effs`), seven surface accumulations (`RAINNC`, `RAINNCV`, `SNOWNC`,
`SNOWNCV`, `GRAUPELNC`, `GRAUPELNCV`, `SR`) and `REFL_10CM` against WRF's
own `calc_refl10cm` (**23 quantities**) at a **2.0e-6 maximum relative
difference** and **2.0e-4 dB**. The tables below use the 16-field subset
(the 15 column fields plus `RAINNC`) that the registry publishes.

**The deck is twenty-two columns, not nineteen.** The gate globs
`woof/data/thompson/oracle-aero/*-column.csv`: the 19 scenarios
`MP28_PORT_SPEC.md` names (ids 101-119, all `aero-*`) plus three `wp08-*`
columns (ids 120-122) that the same `build_aero.sh` invocation produced in
the same format, which pin every reachable `nu_c` (3..15) and both branches
of the terminal phase cleanup. Earlier revisions of this page, the registry
and `PROVENANCE.md` all said "nineteen" while the gate drove twenty-two,
so two residuals (`wp08-freeze`, `wp08-nusweep`) were in no published class
at all. They are below.

**Result: 21 of 22 clear the flat 2.0e-6 / 2.0e-4 dB gate on every
compared quantity, with nothing held out and no allowance anywhere**,
18 of the 19 spec'd `aero-*` fixtures and the three `wp08-*` columns. The
22nd, `aero-cold-overlap`, is the declared rain-graupel divergence: WRF
v4.6.1 allocates its rain-graupel collision tables with a graupel-density
axis of extent 1 when the scheme is not hail aware, fills that one slab,
and reads it with index 5 (`module_mp_thompson.F:465`, `:607-615`,
`:2527-2545`), an out-of-bounds read; WOOF reads the slab the tables hold.
With WRF's read emulated in a measurement copy of the tree (never shipped)
the fixture clears the gate on all 23 quantities, so nothing else in it
differs. Five fixtures
(`aero-ccn-activate`, `aero-ccn-sweep`, `aero-init-profile`,
`aero-sfc-emit`, `wp08-melt`) are bit-exact against WRF on every quantity at
every level, and eight more are bit-exact on all 16 fields of the table
below. This is the 2.8.6 accumulator rework (§3.0); before it the count was
18 of 22 with one allowance, and the history of how it got to 18 is kept in
§3.2 and §3.3.

### 3.0 What closed the last four: WRF's tendencies, applied once

WRF's `mp_thompson` never writes a hydrometeor during the call. Every
process adds to a running tendency (`qcten`, `qrten`, `nrten`, `qiten`,
`niten`, ...), every stage reads the working value `X1d + Xten*DT`, and the
terminal apply (`module_mp_thompson.F:3972-4053`) rounds each species once,
with the size bounds after the `:3943-3966` phase cleanup. The port applied
each stage to the state in turn, so a species was rounded once per stage,
and the classic rain and ice fallout it shared with mp=8 folded the
terminal size bounds in before the freeze. Every one of the last misses was
that, measured stage by stage against WRF's own running tendencies
(`tools/thompson_wrf461_oracle/build_aero_instrumented.sh` with
`STAGE_DUMP=1`; its fidelity proof regenerates all 44 committed fixtures
byte for byte, and `compare_port_stages_aero.py` reports where the port
first leaves WRF):

* `qcten` is carried through the source networks (`:2987`), the
  condensation (`:3480`), the cloud fallout (`:3832`) and the cleanup
  (`:3949`, `:3962`) and applied once (`:3975`). The working cloud is now
  bitwise WRF's after the condensation, the fallout and the cleanup on all
  22 columns.
* `qrten`, `nrten`, `qiten` and `niten` likewise, on the v4.6.1 generation:
  the source-stage ice and rain balances (`:3033-3055`, `:3070-3091`) run in
  WRF's tendency form, the rain evaporation adds `:3562`/`:3564`, mp=28 has
  its own tendency-form rain and ice fallout (`:3611-3640`, `:3664-3698`,
  `:3790-3870`), and a terminal kernel applies them once with WRF's
  `:4023-4053` bounds. The fork generation keeps its in-place rain and ice.
* The cold network's rain-conservation ratio is REAL, as `:1615` declares
  `sump`, `rate_max` and `ratio`; a double ratio sat 3e-8 away, which at a
  level the limiter drains is 2.48e-05 of what survives.

| cell | before | after |
| --- | --- | --- |
| `aero-cloud-freeze-nc` `qc`, level 4 | was 4.926e-06 (one extra rounding) | **0 (bit-exact)** |
| `aero-cold-overlap` `qc` / `nc` / `effc`, level 4 | was 1.0 / 1.0 / 0.810 (one ULP flipping `:4007`) | **0 / 0 / 0** |
| `aero-cold-overlap` `qr` / `nr`, level 6 | was 4.443e-05 / 1.261e-04 | **0 / 0** |
| `aero-reduces-to-classic` `qr` / `nr`, level 6 (held by the allowance) | was 0.585 / 0.159 entry-ULP | **0 / 0, every level bit-exact** |
| `wp08-nusweep` `qr`, level 12 | was 4.642e-06 (60 ULP) | **5.532e-07 (6 ULP, level 11)** |
| `wp08-freeze` `nr`, level 0 | was 4.006e-07 (5 ULP) | **0 (bit-exact)** |
| worst ULP of any cell inside the gate | 20.0 | **6.0** |

The table is the 2.8.6 measurement. On this tree `wp08-nusweep` reads 0.0
on every quantity and, outside the declared divergence, the worst ULP of any
cell inside the gate is 1.0 (a `temp_k`), under the gate's 6.0 ceiling, on an RTX 4090 (sm_89) and an RTX
5070 Ti (sm_120) alike. The two `aero-cold-overlap` rows are not current:
2.8.8 moved rain collecting graupel onto the one slab WRF's tables hold, and
since then the fixture misses by the declared rain-graupel divergence (§3.1)
and by nothing else.

The table below gives, for every spec'd fixture, the worst of the 16
fields and its measured relative difference. `PASS`/`MISS` is against the
uniform 2.0e-6 gate.

| fixture | verdict | worst field | worst relative difference | what it pins |
| --- | --- | --- | --- | --- |
| `aero-ccn-activate` | PASS | - | 0.0 | `activ_ncloud`, both clamp ends of `ta_Na`/`ta_Ww` |
| `aero-ccn-sweep` | PASS | - | 0.0 | activation over 10 updraft × 5 CCN cells |
| `aero-init-profile` | PASS | - | 0.0 | `thompson_init`'s synthetic CCN/IN fill and the `nwfa2d` derivation |
| `aero-sfc-emit` | PASS | - | 0.0 | surface emission lands only on k=kts and is unclamped |
| `aero-drop-evap` | PASS | - | 0.0 | the aerosol-only `tnc_wev` droplet-evaporation branch |
| `aero-nc-accrete` | PASS | - | 0.0 | nu_c-driven accretion, live `t_Efrw` second index |
| `aero-nc-auto` | PASS | - | 0.0 | nu_c-driven Berry-Reinhardt autoconversion |
| `aero-nc-cap` | PASS | - | 0.0 | the `Nt_c_max` caps and the `2/rho` floor |
| `aero-nc-effrad` | PASS | - | 0.0 | all three `inu_c` branches of `calc_effectRad` |
| `aero-nc-sed` | PASS | - | 0.0 | number-weighted cloud sedimentation |
| `aero-scav-rain` | PASS | - | 0.0 | rain scavenging of CCN and IN |
| `aero-warm-overlap` | PASS | - | 0.0 | **cross-network `ncten`/`nwfaten` reconciliation, warm half** |
| `aero-scav-frozen` | PASS | - | 0.0 | snow/graupel aerosol scavenging, `Eff_aero`; bit-exact since the scheme's own WRF-transcribed graupel fallout and graupel number |
| `aero-cloud-freeze-nc` | PASS | temp_k | 6.363e-08 | Bigg freezing with the `nc`-driven cap: **closed by the accumulator rework; see §3.0** |
| `aero-ice-demott-idxin` | PASS | temp_k | 6.359e-08 | the only fixture that reads a `freezeH2O` slice other than 27 |
| `aero-reduces-to-classic` | PASS | temp_k | 6.967e-08 | the bridge to mp=8, which has a historical matched WRF run; `qr` and `nr` bit-exact at every level, no allowance |
| `aero-ice-koop` | PASS | temp_k | 6.634e-08 | **homogeneous haze freezing; see §3.4** |
| `aero-ice-demott-dep` | PASS | temp_k | 6.358e-08 | `iceDeMott` replacing Cooper nucleation |
| `aero-cold-overlap` | MISS | qr | 5.070e-04 | **cross-network reconciliation, cold half; misses by the declared rain-graupel divergence only (emulating WRF's out-of-bounds table read clears it)** |

The three columns outside the spec'd nineteen, measured the same way:
`wp08-melt` PASS (bit-exact on every quantity), `wp08-freeze` PASS
(`temp_k` 6.636e-08; its `nr` is bit-exact), `wp08-nusweep` PASS (0.0 on
every quantity; it read `qr` 5.532e-07 after the 2.8.6 rework).

**The numbers are per card class, and the two measured classes agree.** The
table above is sm_120's (RTX 5090). An RTX 4090 (sm_89) reads every row,
every ULP pin and every per-fixture worst ULP identically with the rework,
so the per-card table that used to list the rows reading differently is
empty. The tests still hold each card class to its own row, keyed by
compute capability; a card class with no row is held to the gate's verdicts
only, and the test output says so.

| card class | fixture | verdict | worst field | worst relative difference |
| --- | --- | --- | --- | --- |

### 3.1 Every field that misses, with its number

| fixture | fields above 2.0e-6 |
| --- | --- |
| `aero-cold-overlap` | declared rain-graupel divergence: `qr` 5.070e-04, `nr` 5.066e-04, `qi` 1.618e-04, `qg` 8.776e-05, `effi` 5.394e-05, `ni` 1.464e-05, `effs` 6.574e-06, reflectivity 1.585e-03 dB |

Only the declared rain-graupel divergence, read identically on an RTX 4090
(sm_89) and an RTX 5070 Ti (sm_120).
`tests/test_thompson_aerosol_adapter.py::_G3_RESIDUALS` carries that one row
and is asserted to name exactly `_G3_DECLARED_DIVERGENCE`, its attribution
table is empty and asserted empty, and
`test_no_residual_survives_and_none_needs_a_regime` asserts the
unexceptioned table (every level, every field, the seven surface
diagnostics included) has nothing above the flat gate outside
`_G3_DECLARED_DIVERGENCE`. `RAINNC`, `RAINNCV` and `SR` are bitwise
identical to WRF on all 22 columns.

The text that used to stand here, for the record of what closed: the four
missing fixtures were `aero-cold-overlap` (`qc` 1.000e+00, `nc` 1.000e+00,
`effc` 8.102e-01 at level 4, one float32 ULP of the entry cloud flipping
`:4007`'s `qc1d <= R1` branch; `nr` 1.261e-04 and `qr` 4.443e-05 at level 6,
99.97% consumed), `aero-cloud-freeze-nc` (`qc` 4.926e-06 at level 4, the
second float32 rounding of `qc` inside one step), `wp08-nusweep` (`qr`
4.642e-06 at level 12, created from zero, ill-conditioned) and, held clean
only by the allowance below, `aero-reduces-to-classic` (`qr` / `nr`
1.238e-04 at level 6 measured relatively). §3.0 is what closed each.

### 3.2 The allowances: all retired

No allowance remains. There is no departure from the flat gate anywhere in
the deck.
`test_every_g3_allowance_is_retired_and_none_is_needed` asserts the
allowance registry and every carve-out constant are empty and that the
gated and the flat verdicts are the same 22 fixtures, so a re-added
carve-out fails rather than passing silently.

| allowance | what it did | retired |
| --- | --- | --- |
| ~~`_NEAR_CANCELLATION_LEVELS`~~ | `aero-reduces-to-classic` level 6 held to 32 ULP of the entry value instead of a relative bound (99.958% of the level's rain evaporates in the step) | **by the 2.8.6 accumulator rework: the level is bit-exact against WRF in `qr` and `nr`** |
| ~~`_END_TO_END_BOUNDS`~~ | `nr` held to 1.0e-5 instead of 2.0e-6 | at the 1.4.1 merge (level 5's `nr` fell to 4.146e-07) |
| ~~`_REFL_DB_BOUNDS`~~ | reflectivity held to 1.0e-3 dB instead of 2.0e-4 dB | by WP-13a (3.242e-05 dB, then 9.537e-06 dB) |

Nothing was ever widened: each bound tightened or retired when the
mechanism it existed for was found and closed.

### 3.3 What moved since the previous revision, in both directions

**The clean count went 15 of 22 → 17 of 22 and the miss count six → four.**
Two production changes did it, both in mp=28-owned kernels, and one of them
also cost a cell.

* **WRF's sedimentation density is level-wise and WOOF's was not.** WRF
  forms the working rain mass and number that sedimentation consumes
  **twice**: at `module_mp_thompson.F:3237-3238` from the `:3193` τ+1
  density, for every level with rain, and again at `:3568`/`:3570` from the
  `:3490` post-condensation density, but only inside the `:3501-3502`
  gate. `thompson_aerosol_sat.cu` wrote the post-condensation density into
  its `reference_density` output unconditionally, so every level got the
  `:3568` answer including the levels WRF never rewrote, and
  `microphysics_aerosol.py` hands that same buffer to the rain
  sedimentation launcher. Measured closures: **`aero-drop-evap` left this
  table entirely** (`rainnc` / `rainncv` 5.165e-04 → 0.000e+00, `qr`
  3.533e-05 → 7.35e-08, `nr` 2.258e-05 → 3.92e-07) and so did
  **`aero-ice-demott-idxin`** (`sr` and `rainnc` / `rainncv` 1.279e-04 →
  0.000e+00 after the second change below, `qr` 2.894e-05 → 6.42e-07);
  `aero-cloud-freeze-nc` lost five of its six rows (`qr` 2.800e-05 →
  8.97e-08, `nr` 1.797e-05 → 2.59e-07, `rainnc` / `rainncv` / `sr`
  1.162e-05 → 0.000e+00) and keeps only `qc`; `aero-reduces-to-classic`
  went `qr` 7.813e-05 → 1.788e-07 and `nr` 4.832e-05 → 5.700e-06, and its
  reflectivity 5.283e-04 → 3.242e-05 dB, which **retired the third
  allowance**.
* **WRF's terminal apply cannot be an FMA and nvrtc made it one.**
  `:3973-4023` is `q1d(k) = q1d(k) + qten(k)*DT`, and the `gfortran -O2`
  baseline-x86-64 oracle has no FMA instruction, so `qten*DT` is rounded to
  `REAL(4)` first; nvrtc contracted the same expression in the cold and warm
  source networks and never rounded it. Both now use the same pinned
  add/sub/mul the rain-evaporation apply already used. **Verified at the
  stage, not only end to end**: the post-source-network `qc` at
  `wp08-freeze` level 0 is now 2.7275411412119865e-05 where the fused form
  gave 2.7275422326056287e-05, and at `aero-cold-overlap` level 4 it is
  1.5486410120502114e-05 where the fused form gave 1.5486406482523307e-05,
  both of which are the rounded-product values WRF itself produces. It made
  `aero-nc-cap`'s `qc` and `nc` and `aero-ice-demott-idxin`'s surface
  accumulators **bitwise** exact.
* **What it cost, recorded rather than absorbed.** `aero-cold-overlap`'s
  `qr` at level 6 **grew**, 3.667e-05 → 4.443e-05 (1.477 → 1.789 ULP of the
  entry value, which is the scale that cell is carried at because 99.5% of
  the level's rain is consumed in the step). Bisected to a single line, the
  cold network's own `qr` apply: reverting that pin alone restores 3.667e-05
  and simultaneously loses all four `aero-ice-demott-idxin` improvements
  above, two of which are exact. The pin is kept because it makes the
  instruction sequence WRF's at every cell rather than at the cells that
  flatter this table.

**What moved in the revision before this one**, kept here because a reader
checking the direction of travel needs more than one step of it:

* **`aero-ice-koop` is closed, and it was the measuring stick that was
  wrong, not the physics.** Its `qi` 1.612e-03 / `ni` 1.764e-03 / `effi`
  5.093e-05 (which this page called "the one genuine physics gap" and
  which three auditors called the port's largest) now measure 1.534e-07 /
  3.396e-07 / 1.886e-07 and the fixture clears the flat gate. **No kernel
  changed.** See §3.4: the closure is entirely the correction of the oracle
  harness's Exner constant. A port that reports "we fixed our measuring
  stick" is more trustworthy than one that reports a physics fix it did not
  make, and homogeneous haze freezing was never measured to be wrong.
* `aero-cloud-freeze-nc` lost its `effc` row (5.018e-06 → 1.619e-06) and
  its `qc` fell 1.478e-05 → 4.926e-06; `aero-ice-demott-idxin` lost its
  `qc` row (6.031e-06 → 7.556e-08).
* **Against that: `aero-cold-overlap` is worse than it was published.** It
  gained the `qc`/`nc`/`effc` rows at level 4 described in §3.1, and its
  `qr` rose 1.174e-05 → 3.667e-05, and then rose again in this revision.
  Recorded, not absorbed.
* **And two residuals had never been published at all**, because the
  narrative counted 19 fixtures while the gate drove 22.

An evidence document that only ever moves in the author's favour is not
evidence. Of the seven bullets above, five are improvements, one is a
regression published as grown, and one is a disclosure of residuals that
were never published; the registry, `PROVENANCE.md` and
`docs/public/PHYSICS.md` now carry the same seven.

### 3.4 What actually closed `aero-ice-koop`: the oracle harness, not a kernel

This subsection exists because the alternative was to let a true sentence
("`aero-ice-koop` is clean") carry a false implication ("WOOF fixed the
homogeneous-freezing kernel"). It did not. **Nothing in
`woof/core/kernels/` changed for this fixture.** What changed is the
Fortran program that produces the reference answer.

`tools/thompson_wrf461_oracle/run_column_aero.F90` builds the Exner
function the way WRF's own `phy_prep` does
(`dyn_em/module_big_step_utilities_em.F:4854`,
`pi_phy = (p_phy/p1000mb)**rcp`). Until 2026-08-01 it spelled `rcp` as
`287.0/1004.0`. WRF's `rcp` is `r_d/cp` with `r_d = 287.` and
`cp = 7.*r_d/2.` (**1004.5, not 1004**)
`share/module_model_constants.F:19`, `:20`, `:31`. In float32 WRF's value
is `0x3E924925`, bit-identical to `2./7.` and to `woof/core/constants.py`'s
`RCP`; the harness's was `0x3E925BCB`, **4774 ULP away**. (The literal
`287./1004.` does occur once in the stock WRF tree, at
`phys/module_fdda_psufddagd.F:1247`, a PSU FDDA diagnostic. It is not the
model's Exner function and nothing in the dynamics or physics prep uses
it.)

The consequence was not that the fixtures were "slightly warm". It was
that the `(p, theta)` pair the harness handed `mp_gt_driver` could no
longer be **inverted** on the WOOF side. A fixture records `p_pa` and
`temp_k`; the adapter must solve for the float32 `theta` that reproduces
`temp_k` bitwise under `woof`'s own `RCP`. Two Exner constants 4774 ULP
apart do not agree about whether that solve has a solution: **dozens of
the deck's 528 entry levels, 47 as measured by every current
environment, 40 as first published; see the correction below, admit an
exact `theta` under one and none under the other.** `_reconstruct_entry_state` therefore perturbed the entry
pressure by up to 15 ULP at those levels, and a perturbed entry pressure
drives genuinely different microphysics. Seven fixtures carried perturbed
levels; `aero-ice-koop` was one of them.

Be precise about which half of that is checkable here and which is
historical. **Checkable now, from the committed bytes:** the shipped deck
inverts exactly under WRF's `r_d/cp` at all 528 levels and fails at 47 of
them under `287.0/1004.0`: the table below, re-derived by a test on every
run.

**Correction, 2026-08-03 (owner-ratified), re-pinning 40 → 47.** This
section first published the superseded-constant failure count as 40, and
the pinning test asserted it. The deck's bytes never changed, the count
did, because it is not a property of the deck alone: the inversion check
runs the host's float32 `power`, and hosts disagree at the last bit.
The original 2026-08-01 environment measured 40; every environment since
measures 47, with identical per-fixture counts across toolchains as
different as glibc 2.43 / numpy 2.5.1 and MSVC / numpy 2.2.6, and the
count is re-pinned to what is measurable rather than to a dead
environment's libm. What never moved, on any environment: **528 of 528
levels invert under WRF's own constant**, and the affected fixtures are
**the same seven** either way. Those two facts carry the attribution;
the exact count under a constant this project already retired never did. **Historical, and taken from the harness's own rebuild receipt:**
that the fixtures *before* the correction were the ones with the perturbed
levels, and that regenerating them under WRF's constant is what moved the
numbers. The old fixtures are not on this tree, so that half is a
citation, not a re-measurement.

**Verifiable from this tree, without rebuilding anything.**
`tests/test_physics_md_aerosol_claims.py::test_the_committed_deck_carries_wrfs_own_exner_constant`
reads the 22 committed CSVs and re-solves the inversion under both
constants:

| Exner constant | entry levels that invert exactly | fixtures affected |
| --- | --- | --- |
| WRF's `r_d/cp` (`2/7`, `0x3E924925`): what the deck carries | **528 of 528** | none |
| the superseded `287.0/1004.0` (`0x3E925BCB`) | 481 of 528 | 7, including `aero-ice-koop` |

The seven are `aero-cloud-freeze-nc`, `aero-cold-overlap`,
`aero-ice-demott-dep`, `aero-ice-demott-idxin` (8 levels each),
`aero-ice-koop` and `wp08-freeze` (6 each) and `aero-reduces-to-classic`
(3). Six of them are `aero-*`; the seventh, `wp08-freeze`, is outside the
nineteen the harness README tabulates, which is why that file's table names
six. Its level counts are also not the same integers as the ones above,
and they should not be: the README counts levels the adapter actually
perturbed, this table counts levels at which no exact `theta` exists at
all. The first is bounded below by the second.

And it is verifiable by rebuild: the harness's own
`tools/thompson_wrf461_oracle/README-AEROSOL.md` records the differential
regeneration (same tree, same kernels, only the fixtures regenerated)
in which `aero-ice-koop`'s worst field went `ni_per_kg` 1.764e-03 →
3.40e-07 and its perturbed-level count went 8 → 0. The gate's
`_ENTRY_STATE_PERTURBATION` table now records `(0, 0.0)` for **every one of
the twenty-two fixtures** and asserts it exactly, so no G3 residual
anywhere in this document can be attributed to the reconstruction any more.

Two things follow, and the second is the uncomfortable one:

1. The port's advertised "largest genuine physics gap" was, for two
   waves, an artifact of WOOF's own reference harness. Every publication
   that quoted it (this page, `PHYSICS.md`, `PROVENANCE.md` and the
   registry warning) quoted a number produced against a yardstick that
   was not WRF's.
2. The same correction is what made `aero-cold-overlap` **worse** (§3.1).
   The regeneration is not a favourable edit that happened to help; it
   moved fixtures in both directions, which is the only reason it can be
   trusted at all.

### 3.5 What the fixtures do NOT cover

* Every fixture supplies its own aerosol, so **no fixture exercises the
  initialisation path a real run takes** (§6.1). That path is now wired and
  separately gated, but not by anything in this section.
* Fixture states are deliberately chosen away from `activ_ncloud`'s bin
  edges. `nc` is a step function of state there (the temperature bin is
  nearest-neighbour on 10 K and `idx_d`/`idx_c`/`idx_n` are `INT`
  truncations) so near an edge an FP32 GPU and the Fortran reference can
  select different bins and differ by tens of percent in `nc` while every
  mass field agrees. Measured at the 248.15 K edge: a 1.526e-05 K change
  moves `nc` by 9.729e-02 relative (8.198490e+08 → 7.400879e+08). This is
  documented rather than absorbed into a looser gate, and **no fixture sits
  on an edge**, so the port's behaviour there is unmeasured.
* Single call, single column, no transport, no accumulation. Everything
  multi-step is class D.

### 3.6 The column oracle: every output word against WRF, bit for bit

The fixtures above are graded at a 2.0e-6 relative gate.  The column oracle
(`tools/thompson_aerosol_column_oracle`, README there) grades the port at 0
ULP: WOOF's shipped adapter on the GPU and unmodified WRF v4.6.1
`module_mp_thompson.F` (gfortran 13.3, `-O2 -fno-tree-vectorize`, scalar
libm) on the CPU, on identical float32 columns, every one of the 23 output
words compared bit for bit: the eleven moments, theta, 10 cm reflectivity,
the three effective radii and the seven surface accumulations.  153 columns
of 49 levels (42 real convective columns, eight synthetic regimes raw and
after WRF's own 10- and 45-step spin-up, 15 edge cases), 121,023 words per
run.  RTX PRO 6000 Blackwell (sm_120), NVRTC 12.9, measured 2026-10-07 at
the development branch.

| run | strict 20 s | default 20 s | strict 5 s | default 5 s |
|---|---:|---:|---:|---:|
| shipped tree: differing words | 3,267 | 3,267 | 2,787 | 2,787 |
| shipped tree: bit-identical columns | 105 / 153 | 105 / 153 | 105 / 153 | 105 / 153 |
| measurement copy reproducing WRF's rain-graupel read: differing words | **0** | **0** | **0** | **0** |

Every differing word of the shipped tree sits in one of 48 columns that
carry rain and graupel at a common level: the declared rain-graupel
divergence above, and nothing else.  On a measurement copy of the tree
whose `aaf_racg_index` reproduces WRF's out-of-bounds read
(`make_racg_read_copy.sh`; never shipped) every word of every column is
WRF's, in both arithmetic builds; the copy faults rather than read past the
arrays' end, so its completed runs also show WRF never read past the end on
these columns.  The strict build (`GPUWM_WRF_EXACT=1`) and the default build
give the same words: every mp=28 operation is pinned, so contraction has
nothing to fuse.

The same holds beyond the 153.  The four residue edge columns (157 in all:
whole-cloud evaporation, the cloud limiter under rain and under riming, the
freeze below HGFR and the melt above 0 C) are bit-identical on the shipped
tree.  The stress sets of `make_stress_columns.py` (every synthetic regime at
2, 17, 49 and 73 levels; isothermal columns at 190 to 310 K and RH 0.001 to
1.08; thin air; condensate from the smallest float32 to 0.02 kg/kg), run at
0.001, 1, 60 and 300 s, and the bounded set at 20, 5, 0.001 and 300 s, read
0 differing words on the measurement copy in all 40 runs (strict and
default), and on the shipped
tree differ only in columns carrying rain and graupel together.

What closed the last differences, in the order they were found: WRF's
`qsten`/`qgten`/`ngten` accumulators (the snow and graupel fallout add to
the sources' REAL tendencies and `:4054-4059` applies each sum once; the
port rounded twice: 592 `qs` and 325 `qg` words at strict 20 s), the phase
cleanup's latent heat (`:3943-3973` adds `lfus*ocp(k)*xri*odt` and
`lfus2*ocp(k)*xrc*odt` to `tten` with the `ocp(k)` and `lvap(k)` WRF last
formed, and returns `t1d + tten*DT`), and the wet-bulb `twet` above WRF's
melting level (`:1971-2013`: a level at exactly 273.15 K keeps `twet = temp`
unless a warmer level sits at or above it).

`tests/test_thompson_aerosol_column_oracle_gpu.py` holds the result: the
157 columns and WRF's 23 words at 20 s and 5 s are committed
(`tests/data/mp28_column_oracle_wrf461.npz`), and the 109 columns outside
the divergence must stay bit-identical in both builds. All 48 declared
columns also pin every strict WOOF output word. A cut requires the same
inputs to pass the measurement-copy check, recorded in the fixture.

The melting-level repair has a separate full-column gate,
`tests/data/mp28_melting_level_wrf461.npz`, with two exactly freezing
columns and a warm neighbor at 20 s and 5 s. The real legacy-mask negative
control must differ in cloud number; the shipped adapter equals all 23
WRF outputs in strict and default arithmetic.

---

## 4. Class B: committed scratch-driver Fortran output

Three programs in `tools/thompson_wrf461_oracle/` link the unmodified WRF
module and print things WRF never prints: scalar helper functions over a
grid, per-process rates inside the source loop, and the state at chosen
anchors inside the column sweep. Receipts, digests and reproduction
commands are in `tools/thompson_wrf461_oracle/PROBE_ORACLE_RECEIPTS.md`.

| table | rows verified | status |
| --- | --- | --- |
| `_WARM_RATE_ORACLE` (`test_thompson_aerosol_warm_gpu.py`) | 124 × 26 fields | reproduced |
| `_NCTEN_BALANCE_ORACLE` (same) | 68 × 8 fields | reproduced |
| `_WRF_COLD_WARM_LOOP` (`test_thompson_aerosol_cold_gpu.py`) | 54 × 20 fields | reproduced |
| `_WRF_COLD_REFERENCE` (same) | 360 values | reproduced |
| `SED_AERO_NC_SED` (`test_thompson_aerosol_sed_gpu.py`) | 384 values | reproduced bitwise |
| `CLEAN_CLASSIC` (same) | 408 values | reproduced bitwise |
| `SED_NU_SWEEP`, `CLEAN_MELT`, `CLEAN_FREEZE` (same) | - | **not verified.** The three scratch scenarios that produced them are not in `run_column_aero.F90` and the driver was lost. The literals are asserted against; their provenance is not reproducible from committed sources. |

The five device-helper probe tables (`probe-activncloud.csv` 1320 rows,
`probe-icekoop.csv` 480, `probe-icedemott.csv` 320, `probe-effaero.csv` 48,
`probe-effectrad.csv` 14) come from `probe_aero_functions.F90` and are
committed. The device helpers reproduce them exactly where the gates say
"bitwise" and to a documented FP32 distance elsewhere.

**Why this is class B and not class A.** `probe_aero_functions.F90` and its
siblings are WOOF code. If one of them passes an argument WRF's own driver
would never pass (a `nifa` in per-kg where WRF uses per-m3, say) the probe
and the kernel can agree perfectly and both be describing a call WRF never
makes. Only the class A fixtures, which go through `mp_gt_driver`, are
immune to that.

---

## 5. Class D: the forecast (G4), and what it does and does not show

Before WP-12b, **no multi-step mp=28 forecast had ever been run.** There is
now one: `tests/test_mp28_forecast_smoke.py`.

**Configuration.** 28 × 16 × 24 cells, dx = dy = 2 km, ztop = 16 km,
dt = 12 s, 150 RK3 steps (1800 s), specified lateral boundaries
(`spec_zone = 1`, `relax_zone = 4`), WK82's analytic sounding entered
through WRF's own moist rebalance, a 3 K / 8 km thermal, and a uniform
20 m/s zonal inflow. The domain is taken through the real
`initialize_physics`, so it carries `thompson_init`'s CCN/IN profile:
this is the configuration a user gets, not a stripped one. Attaching the
full `PhysicsDriver` is separately measured to be **bitwise irrelevant** to
this forecast (same domain-total `RAINNC` to every digit either way), which
is what makes §6.1's counterfactual a clean single-variable comparison.

**What holds, on every one of the 150 steps:**

* no NaN or Inf in any prognostic, any effective radius, `thp`, `php`,
  `mup`, `u`, `v`, `w` or `h_diabatic`;
* in the microphysics-updated interior, every bound WRF's terminal apply
  establishes, on the columns each call updates: `nwfa` ∈ [1.11e7, 9.999e9],
  `nifa` ∈ [5.0e3, 9.999e9], `nc·ρ` ≤ 1.999e9, `effc` ∈ [2.49, 50] µm,
  `effi` ∈ [4.99, 125] µm, `effs` ∈ [9.99, 999] µm;
* every column WRF returns from at `:2020` (entry cloud, ice, rain, snow and
  graupel all at or below R1, nowhere supersaturated over ice) leaves the
  call with `nwfa` and `nifa` above k = 0 **bitwise** as they entered and
  `nc` zero, which is all WRF does to it: that exit comes before the
  terminal apply, and the surface emission afterwards lands on k = 0 only.
  The call's own per-column flag is read back for the check, and in this
  run 232 to 364 of the 364 interior columns take the exit on each step;
* the specified-zone ring is **bit-restored** by every microphysics call:
  checked around the call itself, not across the step, because the
  lateral-boundary path also writes that ring during RK3;
* the production health validator accepts the final state, and `nwfa`,
  `nifa` and `nc` are in its census (so that acceptance means something);
* **no persistent scratch slot is carried between steps.** All fifteen
  `mp_thompson_aero_*` slots are poisoned before every one of 40
  consecutive microphysics calls and the entire prognostic state stays
  **bitwise identical** to an unpoisoned run. This is the failure a column
  test structurally cannot find: `state.scratch` buffers survive across
  steps by design, so an unzeroed accumulator would feed the previous
  step's tendency in as a live physics input and drift plausibly forever.

**The G4 detectors were checked against injected faults** rather than
trusted: a one-row write to the ring, an out-of-band `nwfa` and `effi`, a
single NaN, scoring the 10 m/s run against the 20 m/s wind, and disabling
the accumulator zeroing (which moves 9 of 12 prognostic fields). Every one
turns the corresponding gate red.

**A 2-hour run holds too.** The same configuration integrated for 600 steps
(7200 s, 2.6 domain ventilation times) gives 0 non-finite values, 0 bound
violations, 0 changed cells in a column WRF returned from and 0 ring
violations across all 600 microphysics calls, peak |w| 56.20 m/s,
2.0655 mm of total `RAINNC`, and ends with the domain-interior mean `nwfa`
at 3.1338e5 kg⁻¹ and `nifa` at 2.1e-11 kg⁻¹, both far below WRF's floors
of 1.110e7 and 5.000e3: entirely inflow air, which carries no aerosol, as
§5.1's law predicts. On the last step no interior column holds condensate,
so every one of them takes WRF's `:2020` exit and keeps what transport
delivered; what is left is the surface emission at k = 0. This is the
longest mp=28 integration among this page's measurements. [PHYSICS.md](../PHYSICS.md)
reports later specified-domain runs of about 3.5 and 4 hours forced from
the WIF climatology.

**What this does not show.** Nothing here compares WOOF to WRF. A scheme
with a systematically wrong activation rate would satisfy every bullet
above, for two hours, without a single violation. The bounds are WRF's, but
they are *clamps*, not answers. Class D is a statement about robustness and
nothing else.

### 5.1 The lateral-boundary aerosol depletion, measured

The runs measured here used the earlier external-boundary policy: only `qv`
was supplied from the external snapshot, while `nc`/`nwfa`/`nifa` received
`flow_dep_bdy` zero-gradient outflow and zero inflow. WRF instead carries
`qnwfa`/`qnifa` through its boundary fields. The measurements below quantify
that zero-inflow path.

**Current scope, 2026-10-02.** Specified boundaries can now couple `nwfa` and
`nifa` from WRF's monthly WIF climatology. Without the dataset, the run door
refuses a specified domain unless `mp28_aerosol_source = "synthetic"` was
selected deliberately. Nest edges carry these fields from the parent.
The depletion measured here therefore describes the zero-inflow path, not
a run with the climatology boundary supply (see [PHYSICS.md](../PHYSICS.md)).

It cannot NaN and cannot go negative. Where the scheme runs, WRF's
terminal floors hold the air at `nwfa = 1.11e7`, `nifa = 5.0e3`; a column
with no condensate that is nowhere supersaturated over ice leaves
`mp_thompson` at `:2020`, before the terminal apply, so clear inflow air
keeps its zero aerosol, exactly as WRF would leave it. It therefore trips
nothing. Until now there was no number for how fast it happens. There is
one now, measured on a deliberately **cloud-free** run
(`qc = qr = qi = qs = qg ≡ 0` on every step is asserted, so the aerosol has
no microphysical source or sink and every kilogram lost is the boundary
policy):

| metric | value |
| --- | --- |
| `front_speed_ms` | 20.0909 |
| `front_speed_ratio` | 1.00454 |
| `nwfa_retained` | 0.33139 |
| `nifa_retained` | 0.32719 |
| `ventilation_time_s` | 2800.0 |
| `swept_fraction` | 0.6428571428571429 |
| `surface_emission_per_kg_s` | 5540.14 |

Read that as: with a 20.0 m/s inflow the depletion front advanced at
**20.09 m/s**, i.e. at the wind speed. Over 1800 s the domain-interior mean
`nwfa` fell to **0.331** of its initial value and `nifa` to **0.327**.
Every column of this run takes WRF's `:2020` exit on every step (no
condensate, nowhere supersaturated over ice), so no floor applies anywhere
in it: the inflow air arrives with zero aerosol and keeps it.

The same experiment at 10 m/s gives a front speed of **9.96 m/s** and a
retained fraction of **0.679**, so the behaviour is a law and not one
number:

> **The upstream `U·t` of your domain has lost its initial aerosol after
> time `t`, and the whole domain after `L/U`.**

For a 1000 km operational domain in a 20 m/s flow that is **13.9 hours** to
sweep the domain, and 72 km of the upstream edge has already lost it after
the first hour. For a 100 km nest in the same flow it is **83 minutes**.
Clear inflow air carries no aerosol at all; where the scheme runs it holds
the air at the floor, `nwfa = 1.11e7 kg⁻¹`, about 1/4.5 of the 5.0e7 kg⁻¹
that WRF's own synthetic profile installs in the free troposphere and about
1/17 of the 1.909e8 kg⁻¹ it installs at the surface.

The only aerosol *source* in the scheme is the fixed surface emission
`nwfa2d` at k = 0, measured here at **5540 kg⁻¹ s⁻¹**, which replaces
9.97e6 kg⁻¹ over the 1800 s run against an initial k = 0 loading of
1.909e8 kg⁻¹: about 5%. It cannot keep up, and it acts on the lowest model
level only.

`nifa` and `nwfa` deplete almost alike (0.327 vs 0.331 retained): with no
floor applied anywhere in this run, advection alone sets both, and `nwfa`
has the surface emission besides. Until the port took WRF's `:2020` exit
the clamp ran in every column and held both at their floors, which is why
earlier revisions of this page published 0.457 and 0.336.

### 5.2 The specified-zone ring itself ends at exactly zero aerosol

Sharper, and separate from §5.1, which is about the interior. WRF's clipped
microphysics tiles never touch the `spec_zone` ring, and WOOF reproduces
that bit for bit (§5), so the terminal clamp that holds
`nwfa >= 1.11e7 kg⁻¹` in the columns microphysics updates **does not run
there**. The ring carries whatever `flow_dep_bdy` left, and with no aerosol
in the boundary file that is exactly `0.0`. The same zero then reaches every
clear interior column, which WRF's `:2020` exit leaves alone (§5.1). WRF
itself forces `qnwfa`/`qnifa` at the boundary from the monthly dataset, so
in WRF that inflow air carries aerosol and neither zero arises.

Measured on the same purely zonal 20 m/s run, as the fraction of each face
that ends at exactly zero:

| face | zero fraction | why |
| --- | --- | --- |
| west (inflow) | 1.000 | zero inflow |
| south (tangential, v = 0) | 1.000 | not strictly outbound, so treated as inflow |
| north (tangential, v = 0) | 1.000 | same |
| east (outflow) | 0.125 | retains the interior value except at the two corner cells |

So **three of the four faces**, not one. Microphysics is unaffected (it
never reads the ring) but three consumers are: `QNWFA`/`QNIFA` written to
`wrfout`, any nest whose boundary is fed from this ring, and any diagnostic
that reads the full array. Pinned by
`tests/test_mp28_forecast_smoke.py::test_the_specified_zone_ring_ends_at_exactly_zero_aerosol`.
It is a consequence of D9c, not a separate defect, and it closes with it.

---

## 6. What you would get wrong today

Ordered by how much it would change a forecast. Every item was measured,
not inferred. §6.7 lists what left this section since the previous
revision, so a reader can see the direction of travel without taking it on
trust.

### 6.1 Removing the aerosol initial condition moves this case's surface rain by 56.6%

This is a **sensitivity**, not a defect, but it is the first thing to
understand about mp=28, because it is the largest single number the port
has measured about its own behaviour.

WRF's `thompson_init` fills a *synthetic* CCN/IN profile whenever the
aerosol arrays arrive unset: CCN at `module_mp_thompson.F:493-515`, ice
nuclei at `:531-551`, and the surface emission `nwfa2d` derived from the
filled surface value at `:510`. WOOF ports that fill as
`woof.core.microphysics.microphysics_init`, gates it against WRF's own
post-`thompson_init` snapshot, and **calls it once per domain** from
`woof/core/physics.py::initialize_physics`: the seam where WRF's
`phy_init` calls `mp_init` (`phys/module_physics_init.F:1635`). It is
presence-gated exactly as WRF's is, so a nest that inherited its parent's
aerosol and a run about to restore a checkpoint are both left alone.

Measured, as two otherwise identical 150-step forecasts of the convective
case in §5: one taking the production init path, one with the profile
removed:

| quantity | with the profile (what a run does today) | with it removed | change |
| --- | --- | --- | --- |
| initial mean `nwfa` | 6.653e+07 kg⁻¹ | 0 | - |
| final interior `nwfa` | 2.174e+07 kg⁻¹ | 4.274e+06 kg⁻¹ | floor where the scheme runs, zero in clear columns |
| peak `nc` over the run | 1.593e+08 kg⁻¹ | 2.848e+07 kg⁻¹ | **5.6× fewer droplets** |
| domain-total `RAINNC` | 2.077 mm | 3.253 mm | **+56.6%** |
| peak `RAINNC` | 0.847 mm | 1.056 mm | +24.7% |

Re-measured 2026-09-30 on the RTX 5070 Ti (sm_120) when A146 made Blackwell
cards divide by compile-time constants IEEE-correctly: the stripped run's
rain moved from 3.207 to 3.286 mm, so removing the profile then added 67.8%
rather than 63.8%.

Re-measured 2026-10-05 on an RTX 5090 (sm_120), the published card class,
for the 2.8.6 accumulator rework (§3.0), with the same script the gate runs
(the measurement in
`test_the_published_aerosol_sensitivity_is_a_live_measurement`). At the
2.8.6 staging tip before the rework (ed2b14e7d) this card already read
1.994 and 3.165 mm (+58.7%, 5.6 times fewer droplets): the 2.8.6 changes
merged since 2026-09-30 had moved the trajectory, and the table was stale.
The rework, which rounds the cloud, rain and ice once per call as WRF
does, then moved the run with the profile from 1.994 to 2.066 mm and the
run without it from 3.165 to 3.158 mm. Over 150 steps of a convective
bubble that is a trajectory difference grown from rounding, not a change
in what the aerosol does: the droplet ratio and the peak rain change are
the same to the printed precision.

Re-measured 2026-10-07 on an RTX PRO 6000 Blackwell (sm_120), the same card
class, with the same script, when the cold and warm source networks were
re-transcribed in WRF's arithmetic order and the rain-graupel divergence was
declared (§3). The tree before that change already read 1.989 and 3.167 mm
(+59.2%; the table above it was stale); the change moved the run with the
profile to 1.967 mm and the run without it to 3.258 mm. Rain collecting
graupel is the largest single difference: the collision rates now come from
the table slab WRF built rather than from four rain-intercept bins away.

Re-measured again 2026-10-07 for development branch, same card class, same
script: the four fix lanes merged, WRF's snow and graupel accumulators
applied once, the phase cleanup's latent heat on WRF's own ocp(k)/lvap(k)
and twet above the melting level moved the run with the profile to
2.077 mm and the run without it to 3.253 mm (+56.6%; the table above).
The one-step scheme is now WRF's word for word outside the declared
rain-graupel divergence (§3.6), so this is the 150-step trajectory those
repairs make, not a change in what the aerosol does: the droplet ratio is
5.6 as before.

Both runs are re-executed and this table rebuilt by
`tests/test_physics_md_aerosol_claims.py::test_the_published_aerosol_sensitivity_is_a_live_measurement`,
compared at the precision printed here. The comparison is exact rather
than toleranced because both forecasts were repeated end to end on the
measurement machine and every value was bit-identical across repeats; if
that stops holding, the right response is to publish the spread.

Read that precisely: removing the CCN loading raises domain-total surface
precipitation by 56.6% over half an hour and cuts the peak droplet count by
a factor of 5.6. (Both forecasts were re-run for the 2.8 line. Measured
commit by commit, most of the move from the earlier 74% rise came from the
2026-09-24 mp=28 Thompson repairs; WRF's `:2020` column exit, which stopped
the port clamping aerosol in the columns WRF leaves alone, and the later
Thompson repairs moved it the rest of the way. The base-state geopotential
leaving the FP32 equation of state moved only the third decimal. Merged-in
changes reaching a multi-step trajectory are republished, never rounded
back.) That is not a rounding
difference; it is a different forecast. Note also that §6.2 drives the
domain to exactly the right-hand column given enough time.

**This was the port's largest measured error until 2026-08-01.** The fill
was implemented, proven against WRF and called by nothing, so every mp=28
run integrated from `nwfa = nifa = 0` and the terminal apply clamped both
to WRF's floors (`:3979-3982`) for the whole run, with no NaN, no bound
violation, no warning, and no column gate able to see it, because every
fixture supplies its own aerosol. The same two forecasts measured the cost
of that gap and now measure the value of the profile. Two
gates keep it closed: one scans for the call site and asserts there is
**exactly one** (once-per-domain silently becoming per-step would overwrite
an advected, activated and scavenged aerosol field with the synthetic one),
the other asserts the filled state on a real `DomainState` taken through
the real `initialize_physics`.

### 6.2 Historical zero-aerosol lateral boundary

**Current scope, 2026-10-02.** The missing boundary supply described here is
closed when WRF's monthly WIF climatology is present. A specified domain
without it is refused unless the synthetic source is deliberately selected;
that synthetic path retains zero aerosol inflow. No matched WRF forecast
comparison of the climatology or nested supply is documented here.

Sections 5.1 and 5.2 quantify the historical zero-inflow path. A 6-hour specified-BC run in a 20 m/s flow on a
200 km domain has been ventilated three times over: nothing of the initial
aerosol field remains anywhere; clear air holds none and the columns the
scheme runs in sit at the CCN floor. The scheme keeps working and nothing
warns. Separately, the `spec_zone` ring itself ends at *exactly zero*
aerosol on three of its four faces (§5.2), because the clipped tile means
the clamp never runs there, and that is what a nest or a `wrfout` reader
sees.

**The forecast impact of this deviation is NOT directly measured, and here
is why.** The obvious experiment (re-impose the driving aerosol on the
`spec_zone` ring after every step, as WRF's `bdy_interp` would) does not
work: `flow_dep_bdy` runs *inside* the RK loop and re-zeroes the ring before
transport reads it, so the emulation moves the domain-interior mean `nwfa`
by only **0.05%** over 150 steps. Measured, and reported as a failed
attempt rather than as a null result. At that time the aerosol LBC ingest
was absent. It now exists for WIF climatology, but this failed emulation
does not measure its forecast effect.

What *is* known is the endpoint, and §6.1 now measures it directly: after
`L/U` the whole domain holds inflow air, aerosol-free where it is clear and
at the CCN floor where the scheme runs, which is the right-hand column of
the section 6.1 table: 5.6 times fewer droplets and +56.6% domain-total surface rain.
That is an endpoint magnitude inferred from a different experiment, not a
measured trajectory difference, and it should be read as an order of
magnitude for this zero-inflow experiment, not a measured effect of the
current boundary supply. The initial-only QNWFA/QNIFA ingest was later
extended to the climatology boundary supply described above. The numerical
and observational effects of that supplied-boundary path need their own
matched comparisons; this extrapolation does not qualify them.

### 6.3 One column residual remains, and it is the declared divergence

§3.1, in full: `aero-cold-overlap`, the one fixture where rain meets
graupel, misses by the declared rain-graupel divergence alone (`qr`
5.070e-04, `nr` 5.066e-04, `qi` 1.618e-04, `qg` 8.776e-05, `effi`
5.394e-05, `ni` 1.464e-05, `effs` 6.574e-06, reflectivity 1.585e-03 dB),
and no surface accumulation misses on any fixture. It is a different table
read, not a rounding residual: with WRF's out-of-bounds read emulated in a
measurement copy of the tree the fixture clears all 23 quantities. What this
section said before the 2.8.6 accumulator rework, kept as history: four
column residuals remained and `aero-cold-overlap` had got **worse**, its
level-6 `qr` growing 3.667e-05 → 4.443e-05 as the direct price of pinning
the terminal apply's contraction, while four others got better.

### 6.4 MYNN mixes the `qn` family when it is asked to: **D9d CLOSED**

This section used to record that `woof/core/mynn_pbl.py` passed
`flag_qnc`/`flag_qnwfa`/`flag_qnifa` as literal `False`, so WRF's
`bl_mynn_mixscalars > 0` mixing had no WOOF counterpart and the divergence
was latent rather than active.

`bl_mynn_mixscalars` is now admitted at `{0, 1}` and implemented:
`woof/core/mynn_scalar_mix.py` and its device twin carry WRF's own qn solves
(`module_bl_mynn.F:4654-4860`) plus the `scalar_opt > 0` DMP updraft-flux
terms, and `woof/core/mynn_pbl_runtime.py` drives the `flag_qn*` flags true
for the five stock species when the switch is 1. The default remains
`bl_mynn_mixscalars = 0` (WRF's Registry default) so the shipped default
trajectory is unchanged; what changed is that lighting the switch now mixes
the qn family as WRF does instead of silently doing nothing.

The `1` arm is admitted only under the combination its anchored oracle
fixtures were generated at and the runtime was wired for: `bl_pbl_physics = 5`
(MYNN), `mp_physics = 28` (the one scheme whose state carries the qn family),
and `bldt = 0`. Anything else refuses by name. (Snow *is* mixed:
`flag_qs` is true for mp=28, matching `Registry.EM_COMMON:3036`, which was
a separate defect and is closed.)

Update (2026-10-06): the "requires `mp_physics = 28`" refusal named no
breakage for a microphysics that carries no number species, where WRF's
flag-gated solves (`module_bl_mynn.F` `mynn_tendencies`: each solve is
`bl_mynn_mixscalars > 0 .AND. FLAG_QNx`) mix nothing and the run proceeds.
The gating is now ported as a table (`MYNN_QN_FLAG_SPECIES`,
`woof/config.py`): `bl_mynn_mixscalars = 1` mixes the whole family under
mp=28, is admitted as WRF's no-op under mp 0/1/6 (the driver is handed 0 and
the key-0 path runs bit for bit, measured on the card by
`tests/test_mynn_mixscalars_inert_gpu.py`), and is refused by name under
the schemes WRF would mix only in part (8, 9, 10, 16, 18, 50), because this
port's solve runs all five species or none. The `bldt = 0` restart
invariant binds the mixing case only.

### 6.5 WOOF now runs the configuration `real.exe` admits: **D9a CLOSED**

This section used to say the opposite. `dyn_em/module_initialize_real.F:2734-2736`
fatals with `wif_input_opt=0 but mp_physics=28`, and WOOF had no WIF ingest,
so it ran the case WRF's initializer refuses and a WOOF mp=28 run was
**not** directly comparable to a WIF-initialised WRF mp=28 run.

That is no longer the configuration WOOF runs. `woof/ingest/wif_climatology.py`
ports WRF's global monthly QNWFA/QNIFA climatology
(`QNWFA_QNIFA_SIGMA_MONTHLY.dat`) (metgrid's `four_pt` horizontal
interpolation, `real.exe`'s `monthly_interp_to_date` temporal weighting, and
its `vert_interp` onto the dry eta pressure) matched to `real.exe` to 1e-5,
and a real-data mp=28 run takes it **by default**
(`RunConfig.mp28_aerosol_source = "auto"`). That is the state `real.exe`
reaches through `use_aero_icbc = .true.` → `aer_init_opt = 1` with
`wif_input_opt = 1`, so the two initial conditions are the same one and the
runs **are** directly comparable. The constant that carried the old warning,
`woof.config.MP28_AEROSOL_SOURCE_DEVIATION`, was retired with the deviation;
`MP28_AEROSOL_SOURCE_DEFAULT` and `MP28_AEROSOL_SYNTHETIC_FALLBACK` replace it.

**What survives.** When no dataset can be located, `"auto"` falls back to
`thompson_init`'s synthetic CCN/IN profile (a real initial condition, but a
different experiment) and says so by name in the run receipt. A run that fell
back is not comparable to a WIF-initialised WRF run;
`mp28_aerosol_source = "climatology"` refuses rather than falling back, and
`"synthetic"` selects the fallback deliberately. (The `real.exe` citation was
**re-verified** against a full stock WRF v4.6.1 tree: `:2734-2735` is the
`ELSE IF (config_flags%mp_physics .EQ. THOMPSONAERO .and.
config_flags%wif_input_opt .EQ. 0 )` test and `:2736` is the
`CALL wrf_error_fatal`. Earlier revisions of this page said it had not
been checked, because the reference tree used then held only `phys/`.)

### 6.6 mp=28 and mp=8 are deliberately not bit-identical

`thompson_aerosol_common.cuh` contraction-pins the `RSLF`/`RSIF` Horner
chains; `thompson.cu` leaves them contracted. nvrtc defaults to `--fmad=true`
and the gfortran baseline-x86-64 reference has no FMA instruction, so the
unpinned chain lands one ULP low. `module_mp_thompson.F:3401` opens the
condensation/CCN-activation block on `ssatw > 1.E-15`, so one ULP flips a
branch. The mp=28 kernels hold these two chains to WRF's rounding; mp=8 keeps its
FMA-contracted saturation chains. mp=8 itself changed after its 2026-07-28
matched run against WRF, in 2.7.4 and on 2026-09-23 (section 7), so that run
describes an earlier build.
Deviation **D9g**. This is intended, and "make them agree" is the wrong
fix in both directions. (The `:3401` citation was `:3400` in every earlier
publication of this deviation, including this page; `:3400` is
`orho = 1./rho(k)` and the branch is one line lower. Re-verified for this
revision against both reference WRF trees, which hold byte-identical copies
of the file.)

**It is also measured, and it was RE-measured for this revision**, by
substituting the unpinned Horner chains into the header the loader hands
nvrtc, with no file on disk edited, and re-driving all 22 columns. Removing
the pin took the unexceptioned clean count, when it was last re-measured, from
17 of 22 to **4 of 22**
(only `aero-ice-demott-dep`, `aero-scav-frozen`, `aero-sfc-emit` and
`wp08-melt` survive), and the damage is not subtle: `aero-nc-accrete`,
`aero-nc-auto`, `aero-nc-cap`, `aero-nc-sed`, `aero-scav-rain` and
`aero-warm-overlap` all go to relative differences above 1e+20 in `qc` and
above 1e+36 in `nc` (the branch flip, not a rounding change) and
`aero-ice-koop` returns to `qi` 7.06e-03 / `ni` 8.79e-03. The pin is not a
style choice.

### 6.7 What left this section since the previous revision

Five items that were published here as open are closed, and each was
re-checked for this page rather than taken on report. **One of the five
was never open in the first place**, and it is marked as such rather than
counted as a win:

| was | now |
| --- | --- |
| "The synthetic aerosol profile is never installed, **largest**" | closed by wiring the call: `woof/core/physics.py::initialize_physics`. §6.1 is the same measurement, reframed from a cost into a sensitivity |
| "An mp=28 run cannot be checkpointed, therefore cannot be resumed" | closed: `MICROPHYSICS_ALGORITHM_IDENTITIES` carries a 28 row, and a running mp=28 forecast now checkpoints |
| "`REFL_10CM` is never computed in a runtime-driven mp=28 forecast" | closed: `woof/runtime.py`'s `REFL_10CM_MICROPHYSICS` is `(1, 6, 8, 10, 18, 28)` and one admission constant replaced three open-coded tuples |
| "Homogeneous haze freezing (Koop) is the weakest process" | **withdrawn, not fixed.** It was never measured against WRF's own Exner function; correcting `run_column_aero.F90`'s `287.0/1004.0` and regenerating the deck removed the residual with no kernel change at all. §3.4 |
| "Packaging: `CCN_ACTIVATE.BIN` would reach a wheel" | closed, then **superseded the same day**: the exclusion entry landed and the three packaging gates went green, after which the owner reversed the do-not-ship decision. The entry was removed, the gates were inverted to assert the table *does* reach the wheel, and all three are green in that direction |
| "`aero-drop-evap` and `aero-ice-demott-idxin` miss the gate on their surface accumulators" | closed **in this revision**, on the physics: WRF's level-wise `:3237` / `:3568` sedimentation density was restored and both fixtures now clear the flat gate on all 23 quantities. §3.3 |
| "`_REFL_DB_BOUNDS`, a third allowance on `aero-reduces-to-classic`" | **retired in this revision**: the residual it existed for fell to 3.242e-05 dB, inside the flat 2.0e-4 dB gate, and the constant is now an empty dict. §3.2 |

That is six repairs and one withdrawn claim, against two recorded
regressions in `aero-cold-overlap` (the level-4 rows, which the fixture
regeneration caused, and the level-6 `qr` growth, which the contraction pin
caused and which is published as grown rather than reverted) and one
newly-published pair of residuals (`wp08-*`). It is recorded this way so
the next reader can check the direction of travel rather than infer it,
and so that a correction to WOOF's *measuring instrument* is never
reported as a correction to WOOF's *physics*.

---

## 7. The mp=8 non-regression receipt

mp=28 lives in six new CUDA translation units plus one standalone helper
probe (seven `.cu` files, sharing one new `.cuh`) and ten new Python
modules.
`woof/core/kernels/thompson.cu` and `woof/core/thompson.py` are
byte-frozen, and mp=8's numerics are unchanged **by construction**: the
kernel loader compiles one module per `.cu` file from
`_preamble() + <name>.cu`, so an unedited file is an unchanged source string
and therefore unchanged PTX.

**This receipt is historical.** `thompson.cu` stopped being byte-frozen
when mp=8's own rain concentration and condensation history was corrected
(2.7.4), and it has since carried WRF rules both schemes share (the
process-rate comparison's snow-cloud table bin and fall-speed gates, and on
2026-09-23 the rules that brought the classic kernels and `_apply_thompson`
to the ones the mp=28 units follow). `tests/test_mp8_frozen.py` re-pins its
digests with each such change, with the record of what moved. What holds
mp=8 now is WRF's own Fortran: `tools/thompson_real_column_parity --mp 8` on the
seven saved real-data frames, and the committed classic companion fixture
`tests/data/thompson_real_columns_wrf461_mp8.npz`, which
`tests/test_thompson_real_column_host_parity.py` grades with every process
rate and final-state quantity held to WRF.

Measured rather than asserted (2026-08-01):

| receipt | result |
| --- | --- |
| `thompson.cu` sha256 | unchanged from the pin captured at commit `789f611` |
| assembled nvrtc source string for module `thompson` | unchanged (catches a preamble or `CUDA_DEFINES` change the file hash alone would not) |
| `woof/core/thompson.py` sha256 | unchanged |
| `CLASSIC_TABLE_ASSETS` / `TABLE_SET_ID` | unchanged; `CCN_ACTIVATE.BIN` absent |
| clean rebuild of the mp=8 oracle from a pristine WRF v4.6.1 tree | **re-run for this revision on 2026-08-01.** `tests/test_mp8_frozen.py` goes from 21 passed / 1 skipped to **22 passed** with `WOOF_MP8_ORACLE_REBUILD_DIR` set: 4/4 generated `.dat` SHA-256s match the pins, and an independent byte comparison of the regenerated column oracle against the committed deck gives **88 of 92 CSVs byte-identical** |

The four CSVs that differ are `warm-column.csv`, `ice-column.csv`,
`mixed-column.csv` and `mixed-surface.csv`, and only in the *input* `p_pa`
and the `before` rows that follow from it: a one-float32-ULP
input-seed provenance drift that predates this port (max relative difference
7.26e-06, 1.30e-02, 4.06e-06 and 1.13e-07 respectively; the `mixed-column`
figure is a single near-zero field). This is the documented exception in
`tools/mp8_freeze_receipt.py::ORACLE_REBUILD_EXCEPTIONS`, and this port did
not introduce it and did not widen it.

Re-run it with:

```sh
bash tools/thompson_wrf461_oracle/build.sh /path/to/WRF-v4.6.1 /empty/dir
WOOF_MP8_ORACLE_REBUILD_DIR=/empty/dir python -m pytest tests/test_mp8_frozen.py -q
```

---

## 8. Skip census: every test in the mp=28 suite that can decline to run

A skipped test is an unmeasured claim wearing a green tick. Seven silent
skips are how two critical defects survived wave 2 of this port, so the
complete census is published and machine-checked:
`tests/test_thompson_aerosol_gpu.py::test_the_mp28_suite_has_no_unaudited_skip_site`
fails if a skip site is **added** anywhere in the suite, as loudly as if one
were removed. Nothing in the suite uses `xfail`, and that too is asserted.

The suite is these thirteen modules: `test_kernel_loader_inert.py`,
`test_mp28_forecast_smoke.py`, `test_mp28_runnable.py`, `test_mp8_frozen.py`,
`test_thompson_aerosol_adapter.py`, `test_thompson_aerosol_cold_gpu.py`,
`test_thompson_aerosol_contract.py`,
`test_thompson_aerosol_device_helpers.py`, `test_thompson_aerosol_gpu.py`,
`test_thompson_aerosol_sat_gpu.py`, `test_thompson_aerosol_sed_gpu.py`,
`test_thompson_aerosol_state_gpu.py`, `test_thompson_aerosol_warm_gpu.py`.

**One mp=28 gate is deliberately outside that census, and its skips are
stated here instead.** `tests/test_physics_md_aerosol_claims.py` is the
publication gate (it checks this page and `PHYSICS.md` against the
adapter gate) so it is not part of the physics suite the census covers.
Twenty of its twenty-two tests are host-only and never skip. The two that
drive the device (`test_the_published_aerosol_sensitivity_is_a_live_measurement`
and `test_the_page_republishes_the_measured_depletion_numbers`) each carry
a `pytest.importorskip("cupy")`, a device-count check and the same
`CCN_ACTIVATE.BIN` guard the suite uses; on a machine without a device or
without the table they skip, and the sixteen documentary assertions still
run and still fail closed. Adding the module to `_SUITE_MODULES` and
`_SKIP_SITES` would make that machine-checked rather than stated, and is
filed as an integration request against
`tests/test_thompson_aerosol_gpu.py`, which this package does not own.

### 8.1 What actually skipped on the measurement machine: 14 of 679

| count | module | reason | is it an unmeasured claim? |
| --- | --- | --- | --- |
| 12 | `test_kernel_loader_inert.py` | the six `thompson_aerosol_*` translation units are allow-listed to receive `thompson_aerosol_common.cuh`, so "assembled source == preamble + file" is **false for them by construction** | **No.** The same file asserts the positive property for those six directly, and asserts byte-identity for every other module including `thompson` itself. The skip excludes the six from a test that is about the *other* modules. |
| 1 | `test_mp8_frozen.py::test_clean_oracle_rebuild_reproduces_the_committed_oracle` (named `test_clean_oracle_rebuild_matches_except_the_four_documented_files` before 2.9.1) | opt-in: needs gfortran, the pristine WRF tree and ~380 MB of regenerated tables, gated on `WOOF_MP8_ORACLE_REBUILD_DIR` | **It was, until this page.** WP-12b built the oracle and ran it; the result is §7. It skips again on a machine without gfortran. |
| 1 | `test_thompson_aerosol_device_helpers.py::test_local_reimplementations_agree_with_the_shared_definition` | "no renamed local re-implementations remain": the test lifts every duplicated device function out of the `.cu` files and compares it against the shared header's definition, and there are none left to lift | **No, and this is the desired end state.** The uniqueness of the shared definition is asserted positively by `test_shared_helpers_are_defined_exactly_once_and_only_in_the_header` in the same file, which does not skip. The skip fires because the duplication it existed to police is gone. |

### 8.2 Skip sites that exist but did not fire here

These are the conditional paths a different machine would take. Each is
audited and pinned in `_SKIP_SITES`.

| condition | what stops being measured | how bad |
| --- | --- | --- |
| `CCN_ACTIVATE.BIN` absent (`_tables_or_skip`, 4 modules; two `skipif` markers in `test_thompson_aerosol_device_helpers.py`; one in `test_thompson_aerosol_contract.py`) | **every device gate for mp=28**, including all of class A | Severe, but **no longer expected to fire**: the asset ships with WOOF as of 2026-08-01 (deviation D9i, reversed), so a clean checkout can run the mp=28 device gates against WRF's column fixtures. The guards are kept as defence for a tree where the file was deleted or `WOOF_THOMPSON_CCN_ACTIVATE` points elsewhere. The skip names the one file rather than swallowing every load failure, so it can never be mistaken for a pass. |
| no CUDA device (`_require_device`, `pytest.importorskip("cupy")`, and `conftest`'s automatic `gpu` marker) | every device gate | Expected. The host-only gates (call-graph order, guards, registry, namelist, contract parsing) still run and still fail closed. |
| the arithmetic leg (`test_thompson_aerosol_adapter.py::test_reused_classic_launchers_receive_the_mp8_argument_shape_default` / `_strict`, since 2.9.1) | nothing: one of the pair runs on every leg | None. A default process runs 2.9.1's mp=8 adapter, which still calls the reused launchers, so `_default` compares the two adapters live; under `GPUWM_WRF_EXACT=1` mp=8 is the classic unit and `_strict` holds the pinned shape. |
| fixture or scratch-oracle file absent (`test_thompson_aerosol_sed_gpu.py::test_embedded_seed_matches_the_committed_fixture`, one in `test_thompson_aerosol_device_helpers.py`) | individual bitwise cross-checks | Narrow. Each is a second opinion on something a class A fixture already covers. |

**Two skip sites were RETIRED from this census, not relaxed.**
`test_thompson_aerosol_state_gpu.py`'s
`test_state_finalize_rounds_every_real4_subexpression_that_feeds_a_double`
and `test_effective_radius_is_bitwise_against_a_fortran_faithful_host_sweep`
both began `if _LIBM is None: pytest.skip("libm.so.6 unavailable")`. They
need the glibc `powf` that gfortran lowered the oracle's `REAL(4) ** REAL(4)`
to, and the suite reached it with `ctypes.CDLL("libm.so.6")`, so on any host
that is not a glibc host, two bitwise gates silently did not run. They now
call `woof.core.noahmp_libm.powf`, a bit-exact transcription of the same
glibc 2.39 `sysdeps/ieee754/flt-32/e_powf.c`. What is compared is unchanged;
the condition that could stop it being compared is gone, so the skips are
gone with it and the two gates run everywhere.

### 8.3 Suite result at the time of measurement

**679 collected: 664 passed, 14 skipped, 1 failed.**

The §3 gate, `test_thompson_aerosol_adapter.py::test_g3_end_to_end_against_all_nineteen_oracle_fixtures`,
was red here until the 2.8.6 accumulator rework closed the last three
misses and the one allowance (§3.0); it is green with no allowance now.
The 2.8.6 staging tree carried it for a while as a declared exception in
`tools/battery/must_run_gates.txt` (6e26fd29c); the same branch as the
rework deletes that entry, so no 2.8.6 build skips it.

Two gates that were red when this page was last written are green now: the
mp=8/mp=28 sedimentation bridge ratchet and the reflectivity-residual count
ratchet in `test_mp28_runnable.py`.

The **publication** gate that reads this page and `PHYSICS.md`,
`tests/test_physics_md_aerosol_claims.py`, is **18 passed, 0 skipped, 0
failed** on the same machine, including the two device tests that re-run
§6.1's pair of forecasts and §5.1's depletion measurement.

Outside the mp=28 suite, three groups of gates are red on this machine and
all three are named rather than left for a reader to find. **None is caused
by this port**, and saying so is only worth anything with the cause
attached:

* `test_line_ending_stability.py::test_every_hashed_file_matches_its_committed_blob`
  compares tracked files against their committed blobs, which cannot hold on
  a working tree with uncommitted integration edits in it.
* `test_real74_rungs.py` (29 tests) and `test_clock.py` (18 collection
  errors) both load `configs/real74_4dom.toml`, whose `[case_data]` names an
  ERA5 GRIB file under the case data root that this machine does not have. The
  error is `forcing file … does not exist`, raised by `woof/case_data.py`
  at config load, before any physics runs.
* `test_native_wrf_distribution.py` (2 tests) raises
  `importlib.metadata.PackageNotFoundError: No package metadata was found
  for woof`, this virtualenv runs the tree from `sys.path` rather than
  from an installed distribution. Same class as the `setuptools` skips
  above.

Two further modules are excluded from every count on this page, on the
same grounds and with the same accuracy: `tests/test_flagship_tools.py`
cannot be **collected** (no `matplotlib` in this virtualenv) and every test
in `tests/test_rrtmg_sw_cuda.py` fails with `NVRTC_ERROR_INVALID_OPTION`
because this cupy appends `-ftz=true` to a compile line that already
carries `--ftz=false`. All three RRTMG source files are byte-identical to
the branch point, so that is a toolchain fact, not a port fact.

**What was NOT run.** The whole-tree suite (~7,400 tests) was not carried
to completion for this revision. What was run is the 13-module mp=28 suite
in full, the publication gate in full, and every one of the 24 test modules
that reads a file this work edited: 888 passed, 12 skipped, and no failure
outside the three environment groups above and the §3 gate itself.

**The unsatisfiable pair the previous revision recorded here is resolved.**
`test_physics_md_aerosol_claims.py::test_the_workaround_the_page_prints_is_real`
required `docs/public/PHYSICS.md` to contain the literal
`woof.core.microphysics.microphysics_init(state, cfg)`, while its sibling
`test_physics_md_does_not_claim_the_synthetic_profile_is_installed`
required (once `microphysics_init` had a production caller) that the page
NOT contain `microphysics_init(state, cfg)`. The second string contains the
first, so no page satisfied both. Neither test was deleted and neither
constraint was relaxed. The workaround test became
`test_the_production_call_site_the_page_names_is_real`, which asserts
strictly more than it did: the hook still has to exist, still has to take
`(state, cfg)` and still has to be the documented no-op away from mp=28,
*and* the caller the page names has to be real, has to be the only one, and
the page must print no manual workaround while it exists. Both halves still
fail in both directions, with the call site removed, the workaround
requirement comes straight back.

The packaging defect of the previous revision's §6.9 is closed: the
`[tool.setuptools.exclude-package-data]` entry landed, and
`tests/test_package_data_coverage.py` is 3 passed / 4 skipped here rather
than red. Stated exactly, because the skip is the interesting part: the
gate that runs everywhere -- the one that reads the declaration out of
`tomllib` -- is GREEN, and the four that measure real wheel contents SKIP in
this virtualenv because it has no `setuptools`. That is the same asymmetry
`PROVENANCE.md` D9j records, and it is why the declaration gate exists: a
suite that only had the setuptools ones would have been silently inert in
the environment this port is developed in.

---

## 9. Reproducing everything on this page

```sh
# Class A, all twenty-two fixtures, end to end (and the ratchet on §3):
python -m pytest tests/test_thompson_aerosol_adapter.py \
                 tests/test_thompson_aerosol_gpu.py -q

# Class D, the forecast and the depletion measurement (§5, §6.1):
python -m pytest tests/test_mp28_forecast_smoke.py -q -s

# The mp=8 freeze, including the empirical rebuild (§7):
bash tools/thompson_wrf461_oracle/build.sh /path/to/WRF-v4.6.1 /empty/dir
WOOF_MP8_ORACLE_REBUILD_DIR=/empty/dir \
  python -m pytest tests/test_mp8_frozen.py -q

# This page and docs/public/PHYSICS.md against the gate that owns the
# numbers, including §3.4's Exner re-derivation and §6.1's two forecasts:
python -m pytest tests/test_physics_md_aerosol_claims.py -q -s
```

Every number in §3 and §5 is recomputed by
`tests/test_thompson_aerosol_gpu.py` and
`tests/test_mp28_forecast_smoke.py` and compared against what is printed
here. If this page and the code disagree, those tests go red, including
when a residual **improves**, because an evidence document that understates
the port is still a document nobody re-read.

**And the publication layer is now derived rather than transcribed.**
`tests/test_physics_md_aerosol_claims.py` imports the adapter gate's own
pinned partition (`_G3_UNEXCEPTIONED_CLEAN`, `_G3_GATED_CLEAN`,
`_G3_RESIDUALS`, `_END_TO_END_BOUNDS`) and rebuilds this page's and
`PHYSICS.md`'s counts, residual tables and carve-out from it, in both
directions: a fixture the gate closes may not stay in a miss table, a
fixture it misses may not leave one, and a residual may not be quoted at a
value the gate does not measure. The §6.1 sensitivity table is re-run on
the device rather than cross-referenced, and §3.4's Exner claim is
re-derived from the committed CSVs. Restating a number in prose is the
defect that recurred in every wave of this port; it is now a gate rather
than a review item.

---

## 10. What would move the label

| to reach | what is required | what is missing |
| --- | --- | --- |
| a clean `implemented-unverified` | all twenty-two class A fixtures inside 2.0e-6 with no allowance | nothing: reached by the 2.8.6 accumulator rework, section 3.0 |
| `wrf-matched-run-candidate` | an accepted forecast-scale reference comparison | the idealized gates have not produced a declared PASS; section 5 is self-consistency only |
| `wrf-matched-run` | a matched multi-hour real-data WOOF-versus-WRF forecast with published decay tables | no real-data matched run exists. The three idealized gates returned HOLD, INCONCLUSIVE and HOLD; none qualifies the current specified-boundary or nested aerosol paths |
