# Device memory ledger, where the CUDA port's VRAM goes

> ## EVERY ROW ON THIS PAGE IS RETIRED: read this first
>
> **The account of record is not a line in cell count and is not on this
> page.** Since 2026-08-27 it is the shaped model in
> `woof.hex.device_admission`:
>
> ```
> peak = core(card, configuration)
>      + GF workspace  at min(cells, SMs × 4 × 64)
>      + YSU workspace at min(cells, SMs × 16 × 32)
>      + bytes_per_cell × cells
> ```
>
> Grell-Freitas and YSU size a global device workspace to the threads the CARD
> holds in flight, so those terms stop growing past a knee, 43,520 and 87,040
> cells on a 170 SM part, and RRTMG's column chunk never grows with the mesh
> at all. Ask `device_admission.model_for_card(card, configuration)`; never
> multiply a fixed term by hand, and never quote one card's at another.
>
> **Every dated section below fitted `fixed + slope × cells`, and every one of
> them is retired: the 2026-08-26 merged-tip re-fit included.** They are kept
> because the peaks they measured are real and are the before arms of the moves
> between them.
>
> **Everything under the 2026-08-20 rule below was measured at the WOOF-ArWen seam
> pin `629ddb6f0`, which is BEFORE the Grell-Freitas local-memory frame cut.**
> The pin moved to `0d04db712` (measured 2026-08-24) and then converged with
> the engine release line as `26daaab7e` (measured 2026-08-25).
>
> | | this document (pre-cut, 08-20) | last affine row, RETIRED 2026-08-27 (merged tip, 08-26, 170 SM) |
> |---|---|---|
> | fixed term | 9,797.8 MiB | 5,016.5 MiB |
> | per-cell slope | 86,630 B/cell | 98,748 B/cell |
> | widest launched local frame | `gf_gfdrv_stage`, 29,264 B | **`wsm6_column`, 7,216 B** |
>
> That right-hand column is computable, coefficients and all, at
> `device_admission.RETIRED_AFFINE_ROW_20260826` and
> `retired_affine_row_floor_bytes`. It was retired for SHAPE, not for
> provenance: it charged the card-sized GF, YSU and RRTMG-SW knees as per-cell
> growth, its error across the seventeen recorded peaks spans **−41.42 % to
> +27.19 %**, and **six runs exceeded what it demanded**, four limited-area
> culls short by 1,056 / 1,104 / 860 / 716 MiB, plus 96 MiB on `v6.75.112676`
> and 28 MiB on `v20.80.151649`. The shaped row on that same card and tip is
> **core 3,681.5 MiB + 96,581.58 B/cell** plus the two tiled terms, under a
> **1,756.8 MiB** margin (this card's 1,745.6 MiB shortwave workspace plus
> 11.2 MiB of instrument convention); it reproduces both published peaks
> exactly and covers all seventeen.
>
> **The saving this document identifies has been TAKEN, not left on the
> table.** Section 1 calls the `gf_gfdrv_stage` frame "the single largest
> available saving in the whole footprint". That was true when written and it
> is why the cut was made: the pin move released **3,251.3 MiB** of fixed
> term. `gf_gfdrv_stage` has since left the launched set entirely. Nobody
> should read section 1 as an opportunity that is still open.
>
> **These measurements are kept, and deliberately.** They are the BEFORE arm
> of the pin move's own delta: the 3,251.3 MiB is a subtraction against the
> numbers on this page, so deleting them would delete the evidence for the
> improvement. Read the whole document in the past tense.
>
> **One apparent contradiction, resolved, because it will otherwise be
> re-discovered as a defect.** The 2026-08-24 GF pin-move measurement's
> verdict ledger reports `ysu_column` as the widest local frame in BOTH the
> old and the new arm, and a local-frame reservation delta of 0.0 MiB. That
> does not contradict this document, it is exactly the instrument limitation
> section 5 of this page documents in advance: a post-run `gc.get_objects()`
> scan cannot see the `gf_gfdrv_stage` frame at all, because
> `woof/core/gf.py:166` rebinds the `RawKernel` on every call and the
> 29,264 B frame is unreachable by the end of a run. The scan understates the
> largest single allocation by 4,990 MiB and reports 9,232 B instead. The
> campaign used that scan-based row, so its 0.0 MiB delta measures the
> instrument, not the cut. The fixed term falling 3,251.3 MiB in the same
> session is the number that reflects the cut.

---

## 2026-08-26, the merged-tip re-fit (RTX 5090, 170 SM), RETIRED AS A SHAPE 2026-08-27

*(This section was the of-record row until 2026-08-27, when the affine SHAPE
itself was retired: see the banner. Its two measured peaks are unchanged and
the shaped model reproduces both exactly; what is retired is the two-point
line drawn through them, computable at
`device_admission.RETIRED_AFFINE_ROW_20260826`. Past tense throughout.)*

Decided 2026-08-26: the 08-25 row below was measured at hex
**`7fe514b`** and then quoted at every later tree. This section is the
named follow-up from the 08-26 floor re-proof, run: one per-allocation ledger session, both
published meshes, same card, same engine pin `26daaab7e`, same protocol
(6 steps x 120 s full physics, peak = this process's `nvidia-smi` row), at
the merged tip **`2009db7`**.

| mesh | cells | peak at `7fe514b` | peak at `2009db7` | delta |
| --- | --- | --- | --- | --- |
| `x1.40962` | 40,962 | 8,390.0 MiB | **8,874.0 MiB** | **+484.0** |
| `x4.163842` | 163,842 | 20,542.0 MiB | **20,446.0 MiB** | **−96.0** |

Two-point fit: **fixed 5,016.5 MiB + 98,748 B/cell** on this card.

**The tip did not shift the footprint uniformly, and that is the finding.**
A tree that simply cost more would have moved the fixed term alone. This one
moved both terms in opposite directions (a mesh-independent +484 MiB and a
per-cell −4,948 B) so the two rows CROSS at about 143,554 cells. Below the
crossing the new row asks for more; above it, less. Quoting the 08-25 row at
this tip left `x1.40962`'s requirement only **27.9 MiB** above its measured
peak: 5 % of the shared headroom, on the smallest published mesh. Both rows
stay computable in one place (`device_admission.RETIRED_AFFINE_ROW_20260826`
and `device_admission.RETIRED_ROW_20260825`; this section named the first of
those `FOOTPRINT_MODEL`, which since 2026-08-27 holds the SHAPED row instead).

**Admitted cells, at each card's own measured free memory:**

| card | free at the reading | 08-25 row | merged-tip row | |
| --- | --- | --- | --- | --- |
| RTX 5090 (170 SM) | 31,642.6 MiB | 270,915 | **277,297** | +6,382 |
| RTX 5070 Ti (70 SM) | 15,880 MiB | 111,524 | **109,919** | −1,605 |
| RTX 3080 (desktop) | 9,097.0 MiB | 42,934 | **37,892** | −5,042 |

No registered mesh changes status on the two larger cards. On the 10 GiB
desktop card **two do**: `x1.40962` (40,962) and `v15.150.38857` (38,857)
move from admitted to refused **on the default row**. That refusal is
conservative rather than correct (the same card was measured running
`x1.40962` at a 6,340.5 MiB peak) and the remedy is the card's own row at
the door (`CARD_TIER_ROWS["10gib-68sm"]`, now fixed **2,483.0 MiB** with the
slope re-borrowed from this tip: 64,795 cells, `x1.40962` admitted with
2,244 MiB to spare). It is still a capability a bare default run on that
card no longer reaches, and it is recorded here as one.

> **[Corrected 2026-08-27, the borrowed-row defect.]** That remedy did not exist.
> `CARD_TIER_ROWS` is now a retired alias for
> `RETIRED_CARD_TIER_ROWS_20260826`, and when this paragraph was written
> **nothing read it**: `forecast_door._resolve_model` returned the 170 SM row
> unless the user typed `--device-fixed-mib` and `--device-bytes-per-cell` by
> hand. The 10 GiB entry carried a second defect beside that: its 2,483.0 MiB
> fixed term came from a WDDM device-view peak that includes the desktop's own
> 1,142.5 MiB baseline, which the door's free-memory comparison has already
> taken out, so the row charged the desktop twice. The live surface is
> `model_for_card(KNOWN_CARDS["10gib-68sm"], configuration)` and the door reads
> the driver's multiprocessor count at decision time, default-on. On that
> card's own shaped row (core **1,239.7 MiB**, margin **884.0 MiB**)
> `x1.40962` needs **6,566.0 MiB** and `v15.150.38857` **6,372.1 MiB** against
> 9,097.0 MiB free, both ADMITTED, confirmed on the real RTX 3080.
> The capability this
> paragraph recorded as lost is fixed, not worked around.

**What the re-fit did NOT fix.** The row's one graded-mesh point,
`v20.80.151649` (151,649 cells), peaked 19,838.0 MiB against this row's
19,297.8 MiB prediction: **+540.2 MiB (+2.80 %)**, and **28.2 MiB above its
own admission requirement**. The 08-25 row missed the same point by
+2.60 %. Both fitted points are quasi-uniform global meshes; the only graded
point measured sits above the line through them, and the re-fit ISOLATED
that residue rather than removing it: this row is exact on both uniform
meshes, so the excess is now attributable to the mesh shape instead of being
tangled with a stale fixed term. At most 11 MiB of it is the whole-device
sampling convention (measured side by side in this measurement's x4 arm).
**CLOSED 2026-08-27, and the answer was not the mesh shape.** The per-allocation ledger arm
above ran: the SAME graded
mesh at this row's own 6-step protocol measures **19,255.25 MiB against a
19,297.8 MiB prediction (the row OVER-predicts it by 0.22 %**) and a
whole-device sampler running beside the probe agreed to 0.75 MiB, so neither
mesh shape nor sampling convention explains the +2.80 %. The placed-mesh footprint spread is
answered: a graded mesh is not more expensive per cell, and its peak-instant
site census is identical to the uniform mesh's scaled by cell count.

**The guard this paragraph carried is RETIRED with it.** "Do not spend the
512 MiB headroom, and size graded meshes with margin above what the formula
returns" was installed for a defect that turned out not to exist. The flat
512 MiB headroom is retired too, for a different and real reason: it named
no breakage and it failed by 96 MiB on `v6.75.112676`. The margin is now
priced from the card and names what it absorbs:
`woof.hex.device_admission.ShapedFootprintModel.margin_terms`.

## 2026-08-25, the converged-stack arm (RTX 5090, 170 SM), SUPERSEDED 2026-08-26

The seam pin converged with the engine release line: hex
**`7fe514b`** + engine **`26daaab7e`** (`pin/mpas-port-arwen-seam-v4`,
the seam-converge merge into the engine release line), which brings
the device-sized radiation chunk, the release line's frame-cut
wave, and the refl10cm/q2 history stream into one stack. Measured
2026-08-25 (UTC 16:46–16:52Z) on an **RTX 5090
(32,607 MiB, 170 SM)**, the per-allocation ledger instrument
(`tools/device_memory_ledger/hex_ledger_probe.py`), 6 steps × 120 s
full physics, peak = this process's `nvidia-smi` row.

| mesh | cells | peak (process nvidia-smi) |
| --- | --- | --- |
| `x1.40962` | 40,962 | **8,390 MiB** |
| `x4.163842` | 163,842 | **20,542 MiB** |

Two-point fit: **fixed 4,339.1 MiB + 103,696 B/cell** on this card.
*(Superseded 2026-08-26 by the merged-tip re-fit above; the retired arm
is computable at `device_admission.RETIRED_ROW_20260825`. Read this
section in the past tense.)*
Against the superseded 170 SM row (`0d04db712`, 2026-08-24: 6,296.5 MiB
+ 93,474 B/cell; x1 9,948, x4 20,902) the fixed term falls
**1,957.4 MiB** (device-sized radiation chunk narrows the LW chunk on a 170 SM part; the frame
wave shrinks the local-memory reservation: widest launched frame is now
`wsm6_column` 7,216 B, predicted reservation upper bound 1,797.0 MiB)
and the slope rises **10,222 B/cell** (the refl10cm/q2 history
publication and the release line's seam-file evolution ride per-cell).
Net: x1 −1,558 MiB, x4 −360 MiB. x1.40962 fits a 12 GiB budget with
3,898 MiB of headroom; x4.163842 remains in practice a 32 GiB-card
configuration under the RE-DERIVED `NATIVE_DEVICE_FLOOR` (2026-08-25: the floor is now this row plus 512 MiB headroom
(20.6 GiB at x4) through `woof.hex.device_admission`; the 24 GiB
assertion is retired, see the 2026-08-25 per-card section below). The
fixed term is per-card: none of this transfers to smaller parts, whose
rows are measured, never derived.

Same-session corruption check on this no-ECC part: the step-16 restart divergence
gate ran first on the same card and stack, two independent processes
(uninterrupted parent, fresh-process restart worker) produced
byte-identical history (`118332d5…` both), which is the dual-run byte
comparison this node's hardware history requires.

## 2026-08-26: the floor re-proved, and the row's first out-of-sample point

Re-derived 2026-08-26.
The re-proof found no new value to derive (the constant has been the measured row since 08-25)
and found the tree's CITATIONS stale instead: the graded-mesh work branched
from a pre-08-25 base, measured its capacity boundaries against the retired
proxy on hardware, and merged those conclusions beside the re-derived floor.
Three registry rows and STATE.md section 6 described a retired constant as
governing. Retired here and pinned by test; the retired arm is computable in
one place (`device_admission.retired_linear_floor_bytes`) so a before/after
table is never hand-typed again.

Measured on the RTX 5090 (free 31,642.6 MiB at the reading):

| leg | measurement |
| --- | --- |
| `v15.60.224210` (224,210 cells) | door preflight **ADMITTED** on memory: 26,511.7 MiB predicted + 512 headroom against 31,642.6 free. The retired proxy demanded 32.84 GiB: above the card's total either way it is read. Still refused by the 75 s timestep, and the preflight now reports BOTH |
| `v20.80.151649`, 1 h full physics | rc **0**, 30/30 steps, `memory_admission.minimum` **20,812,141,738**: the shared sum, 19.38 GiB. The same gate at the graded-mesh work's base, same mesh and card and day, demanded the retired proxy's 22.21 GiB |
| cells admitted, 32 GiB part | 210,952 → **270,915** |

**The row's first out-of-sample point, and it is not a clean confirmation.**
`v20.80.151649` is a cell count the two-point fit never saw and a graded mesh
shape it was never fitted on. Predicted 19,336.0 MiB, measured **19,838.0 MiB
(+2.60 %)**, steady for the last 50 s of the run: inside its own 19,847.7 MiB
requirement by **9.7 MiB**, entirely on the shared headroom. The same mesh
measured 19,226 MiB at the graded-mesh work's base earlier the same day, so the
merged tip costs **+612 MiB (+3.2 %)** on this mesh against the tree the row
was fitted at (`7fe514b`). **NAMED FOLLOW-UP: re-fit the of-record row at the
merged tip**, one per-allocation ledger session, both published meshes, 170 SM card. Until it
runs, the fixed term is a `7fe514b` number quoted at a later tree and the
error direction is under-prediction.

> **[Corrected 2026-08-27: the follow-up ran and the sign reversed.]** The
> re-fit is the 2026-08-26 section at the top of this page, and the per-allocation ledger arm on
> the graded mesh itself ran with the four-swaths measurement: the SAME
> mesh at the row's
> own 6-step protocol measures **19,255.25 MiB against 19,297.8 predicted, the
> row OVER-predicts it by 0.22 %**, and a whole-device sampler beside the probe
> agreed to 0.75 MiB. So the error direction was not under-prediction and the
> mesh shape was not the cause. The 19,838.0 MiB peak stands as measured: it
> carries both a different run schedule and a different engine pin from the
> 19,255.25 figure and which of the two moved it is NOT SEPARATED, 582.75 MiB
> apart. Under the shaped model that mesh requires **21,079.8 MiB**, which
> covers the 19,838.0 MiB run with 1,241.8 MiB to spare where the affine row
> demanded 19,809.8 and was overrun by 28.2.

## 2026-08-25, per-card rows at the converged stack, and the floor re-derived

The `NATIVE_DEVICE_FLOOR` re-derivation (2026-08-25) made the
of-record row above THE
admission gate: every free-memory decision (forecast door, `--preflight`,
the binding's per-mesh floor, the driver's `MIN_FREE_DEVICE_BYTES`) is
`round(4,339.1 MiB + 103,696 B/cell) + 512 MiB headroom`, one sum in
`woof.hex.device_admission`. The retired linear floor
(`24 GiB * cells / 163,842`) both refused meshes this row says fit and
admitted x1.40962 at 6,144 MiB free against its measured 8,390 MiB peak.
`FLOOR_DERIVATION` in that module is the machine-readable derivation.

Two per-card per-allocation ledger rows were measured the same day at the converged stack
(hex `5252421` + engine `26daaab7e`):

| card | mesh | peak | row |
| --- | --- | --- | --- |
| RTX 3080 (10,240 MiB, 68 SM, sm_86, WDDM, desktop) | `x1.40962` | **6,340.5 MiB** device-view (run-start baseline 1,142.5 MiB included: WDDM publishes no per-process `nvidia-smi` row) | fixed **2,289.6 MiB**, slope **borrowed** from the 170 SM fit and said so |
| RTX 5070 Ti (16,303 MiB, 70 SM) | `x1.40962` / `u96.64002` | **6,272.0** / **8,802.0 MiB** (per-process `nvidia-smi` rows) | two-point fit **1,774.0 MiB + 115,143 B/cell**, same caveat as the 08-24 fit on this card: the two peaks land at different instants, so quote the per-mesh peaks |

No 12 GiB card exists in the fleet: the 12 GiB figure is a DECLARED
DERIVATION from the of-record model (about 75,000 cells at a full
12,288 MiB budget), labeled `DERIVED, NOT MEASURED` in
`CARD_TIER_ROWS["12gib"]` and test-pinned to stay labeled.

> **[Corrected 2026-08-27.]** `CARD_TIER_ROWS` is a retired alias
> (`RETIRED_CARD_TIER_ROWS_20260826`), and its `"12gib"` entry keyed a
> derivation to a card's MEMORY SIZE, which is the one card property the
> shaped model does not read. The answer follows the multiprocessor count: at a
> full 12,288 MiB budget the shaped rows carry **57,433 cells on a 170 SM part
> and 103,085 on a 68 SM part**, the same budget and a 1.8x spread, so there is
> no single "12 GiB figure" to declare. `model_for_card` derives an unmeasured
> card's row from the SM count the driver reports and labels it `DERIVED, NOT
> MEASURED`; checked against the two cards that do have their own measurement
> it over-predicts by 182.7 MiB on the 70 SM part and 490.8 MiB on the 68 SM
> part, which is the direction a gate needs. No 12 GiB card exists in the fleet
> either way.

What the re-derivation changed on real cards, measured at decision time:
the 3080 (free 9,097.0 MiB) went from 25,672 admitted cells on the
superseded door row to 42,934 on the of-record row and **63,659 on its own
measured row**, x1.40962 is now ADMITTED on the 10 GiB desktop card
(predicted 8,389.9 + 512 vs 9,097.0 free, `preflight_passed`) where the superseded row refused it at 9,948.0; u96.64002
still refuses there by 33.9 MiB on the card's own row, by name. The 16 GiB
card (free 15,880 MiB) went 101,762 → 111,524 → **123,796** cells, and its u96
leg ran to rc 0 under the new floor: admitted-and-completes, on hardware.

## 2026-08-24: the merged-tip arm (RTX 5070 Ti, 70 SM)

The 08-24 pin-move campaign left one open row: the integration tip
carries both the GF frame cut AND the copy-elision merge, and no
ledger had run on that combination. This section is that measurement.
It is a NEW dated arm, not a revision of anything below; the 2026-08-20
content stays untouched as the pre-cut record.

Measured 2026-08-24 (UTC 2026-08-25T04:33–04:58Z) on an **RTX 5070 Ti
(16,303 MiB, 70 SM)**, hex tree **`9a87d27`** (the merged
the port's integration branch tip), engine pin **`0d04db712`**, the per-allocation ledger
instrument (`tools/device_memory_ledger/hex_ledger_probe.py`, selftest
rc 0), 6 steps × 120 s full physics, peak = this process's `nvidia-smi`
row.

| arm | mesh | cells | peak MiB | peak phase | pool MiB | non-pool MiB |
|---|---|---:|---:|---|---:|---:|
| merged tip | `x1.40962` | 40,962 | **6,724.0** | after_step_1 | 5,414.0 | 1,310.0 |
| merged tip | `u96.64002` | 64,002 | **9,344.0** | after_step_6 | 8,126.3 | 1,217.7 |
| + device-sized radiation chunk narrowing | `x1.40962` | 40,962 | **5,604.0** | after_step_2 | 4,274.3 | 1,329.7 |
| + device-sized radiation chunk narrowing | `u96.64002` | 64,002 | **8,664.0** | after_step_6 | 7,386.6 | 1,277.4 |

Two-mesh affine fit at the merged tip, THIS CARD: 2,066.0 MiB +
119,239 B/cell, with the caveat that the two peaks land at different
instants (x1 in step 1's radiation; u96 at the step-6 radiation +
history coincidence), so the pair does not separate card constant from
slope cleanly. Quote the per-mesh peaks. **The fixed term is per-card:
nothing here moves the 170 SM model of record (6,296.5 MiB +
93,474 B/cell at `0d04db712`, before the copy-elision merge), and the merged tip on that
card class remains NOT MEASURED.**

> **[Superseded 2026-08-25, stale-guard audit finding 10.]** The
> bolded sentence above dated: the 170 SM model of record is now the
> converged-stack row (4,339.1 MiB + 103,696 B/cell, the top section of
> this document), and the merged-and-converged tip on the 170 SM card
> WAS measured 2026-08-25, that converged row IS it. This card's own
> per-card row was also re-measured at the converged stack the same day
> (1,774.0 MiB + 115,143 B/cell).
> The paragraph is kept as the 08-24 record it is.

What sits at the peak: **2,832.8 MiB of RRTMG/McICA chunk transients,
identical at both mesh sizes**, the mesh-independent radiation
workspace this document's tier-2 section identified (`rrtmg_sw.py`
spcvmc workspace 1,745.6 MiB and friends), priced by
`legacy_radiation_vram_bytes` at 2,832.4 MiB for SW 2048 / LW 4096.

**the device-sized radiation chunk took it** (engine commit `1a665e3fc`): the chunk width now defaults
to the smallest 256-multiple saturating the device's resident threads,
capped at the old constants.
On this 70 SM part that is SW 1024 / LW 768: x1 releases 1,120.0 MiB
(the peak moves off radiation entirely, RRTMG live at the new peak is
0.4 MiB), u96 releases 680.0 MiB (its step-6 pool event shrinks from
+1,745.7 to +872.8 MiB, exactly the halved SW workspace). Output is
byte-identical (135/135 history surfaces, 31/31 per-step fingerprints,
flip-validated comparator) at zero measured wall cost. On a 170 SM part
the fix halves only LW (4096 → 2048; SW's ceiling binds), and no arm
ran there.

**12 GiB, on this card, measured:** x1.40962 6,724.0 → 5,604.0 MiB and
u96.64002 9,344.0 → 8,664.0 MiB. Both meshes fit a 12 GiB budget at the
merged tip with ≥ 2.9 GiB headroom (≥ 3.5 GiB with the device-sized radiation chunk). `x4.163842`
was not attempted: its 24 GiB admission floor exceeds the card.
*(That 24 GiB floor was the asserted constant in force on 08-24; it was
re-derived 2026-08-25 to the measured 20.6 GiB, which still exceeds this 16 GiB
card, so the non-attempt stands.)*

---

Measured 2026-08-20 on one RTX 5090 (32,607 MiB, 170 SM, sm_120), float32,
nVertLevels=55, full physics, 6 steps, at two mesh sizes so the fixed and
per-cell terms separate by subtraction rather than by assumption.

**Tense warning:** every present-tense claim below describes the pre-cut
stack at pin `629ddb6f0`. It is left in its original wording rather than
rewritten, so that it remains quotable as the before arm.

Runs: `x1.40962` (40,962 cells / 122,880 edges) and `x4.163842`
(163,842 cells / 491,520 edges), both from GFS 2026-08-12 06Z, both rc 0,
`status: passed`.

## The measured model

```
footprint(cells) = 9,798 MiB + 86,630 B/cell        (this card, this build)
```

| tier | fixed MiB | B/cell | what it is |
|---|---:|---:|---|
| non-pool | 7,730.9 | 1,193 | CUDA context, kernel local-memory backing store, module images |
| pool tier 1, resident | 0.5 | 37,064 | 385 arrays live between steps |
| pool tier 2, transient | 1,253.0 | 47,728 | step workspace live at the peak instant |
| pool tier 3, allocator | 813.3 | 645 | pool blocks held above the live peak |
| **total** | **9,797.8** | **86,630** | |

Whole-process anchors (this process's `nvidia-smi` row, not device-wide):
x1 = 13,182.0 MiB, x4 = 23,334.0 MiB.

Prognostic state is 4,628 B/cell: **5.3 %** of the slope. The other 94.7 %
is resident physics tendency storage, step workspace and allocator headroom,
itemised below.

## 1. The fixed term is a property of the CARD, not of the model

Half of a 10 GiB card disappears before the first cell exists, and 7,034.0 MiB
of the 7,777.5 MiB non-pool total is **one allocation**: the CUDA local-memory
backing store.

| class | x1 MiB | x4 MiB | scales with cells | dtype | lifetime | allocated at |
|---|---:|---:|---|---|---|---|
| CUDA context + CuPy/NVRTC baseline | 498.0 | 498.0 | no | - | process | first CUDA touch |
| local-memory frame, `gf_gfdrv_stage` (29,264 B/thread), **HISTORICAL: cut at pin `0d04db712`, no longer in the launched set** | 4,990.0 | 4,990.0 | no | local | process | `woof/core/gf.py:168` |
| local-memory frame, `ysu_column` (9,232 B/thread) | 1,790.0 | 1,790.0 | no | local | process | `woof/core/ysu.py:104` |
| local-memory frame, `rlw_rtrn_march` (2,048 B/thread) | 254.0 | 254.0 | no | local | process | `woof/core/rrtmg_lw.py:3979` |
| non-pool taken during device-stack construct | 142.4 | 207.3 | yes (554 B/cell) | cubin + | process | `cuda_backend/runtime.py:270` |
| **unattributed residue** | **103.1** | **178.0** | 639 B/cell | | | |
| non-pool total, measured | 7,777.5 | 7,917.3 | 1,193 B/cell | | | |

The three local-memory rows are **increments to one pool**, not three pools.
CUDA sizes a single per-context local backing store to the largest single
kernel demand and never returns it while the context lives:

* launched alone, `gf_gfdrv_stage` reserves **7,034.0 MiB**, exactly
  254.0 + 1,790.0 + 4,990.0, the sum of the three in-run increments;
* launch `gf_gfdrv_stage` first and `rlw_rtrn_march`'s own increment is
  **0.0 MiB**.

Consequences worth acting on:

* The store is sized from the card's resident-thread capacity, not from the
  mesh, so a "fixed term" quoted without naming the card is meaningless. Its
  size on any other card is NOT MEASURED, only this 170 SM part was run.
* It is driven by **one kernel**. `gf_gfdrv_stage` carries a 29,264 B
  per-thread frame; cutting that frame is worth up to 4,990 MiB on this card
  and is the single largest available saving in the whole footprint.
  **TAKEN, 2026-08-24.** This is the finding the pin move acted on. The cut
  landed at `0d04db712` and released 3,251.3 MiB of fixed term (9,547.8 →
  6,296.5 MiB); `gf_gfdrv_stage` is gone from the launched set and the widest
  launched frame is now `ysu_column` at 9,232 B. This bullet is a record of
  why the cut happened, NOT a saving still on offer.
* `gf_deep_stage` (26,880 B) and `gf_shallow_stage` (18,944 B) are compiled
  but never launched from the shipped package: they would cost 6,438.0 and
  4,462.0 MiB if a future call site launched them first.

Module images are not the story: all eight physics modules together load
**4.0 MiB** of device code (`module-images.json`).

## 2. The slope: 86,630 B/cell against 4,628 B/cell of state

### Tier 1. Resident, 37,064 B/cell (8.0 × the prognostic state)

1,448.4 MiB at x1 → 5,791.8 MiB at x4, over 385 named arrays, all `run`
lifetime, and with essentially **zero fixed part** (0.5 MiB): this tier is
pure per-cell. The largest rows:

| array | x1 MiB | x4 MiB | B/cell | dtype | allocated at |
|---|---:|---:|---:|---|---|
| `atm.terrain.zb3_cell` | 87.5 | 350.0 | 2,240 | float32 | `cuda_backend/containers.py:736` |
| `atm.terrain.zb_cell` | 87.5 | 350.0 | 2,240 | float32 | `cuda_backend/containers.py:735` |
| `atm.state.scalars` | 51.6 | 206.3 | 1,320 | float32 | `cuda_transport_v841.py:612` |
| (unnamed) | 34.4 | 137.5 | 880 | float32 | `cuda_arwen_physics_v841.py:1328` |
| `atm.saved.normal_velocity` | 25.8 | 103.1 | 660 | float32 | `cuda_backend/recovery.py:267` |
| `atm.state.rho_u` | 25.8 | 103.1 | 660 | float32 | `cuda_driver.py:3188` |
| `atm.vertical.zxu` | 25.8 | 103.1 | 660 | float32 | `cuda_backend/containers.py:433` |

The shape of this tier is the finding: after the seven rows above it is a long
flat tail. **104 of the 385 arrays sit at 220 B/cell**: one full
`nVertLevels × nCells` float32 field each, 8.6 MiB at x1 and 34.4 MiB at x4.

Counted by owner at x4:

| owner | arrays | x4 MiB |
|---|---:|---:|
| `woof/core/physics.py` | 174 | 1,273.8 |
| `woof/core/mpas_column_batch.py` | 74 | 1,531.9 |
| `cuda_backend/containers.py` | 50 | 1,231.3 |
| `cuda_gwdo_v841.py` | 19 | 214.4 |
| `woof/core/ysu.py` | 15 | 279.4 |
| `cuda_backend/recovery.py` | 7 | 310.0 |
| `cuda_arwen_physics_v841.py` | 6 | 291.3 |
| `cuda_driver.py` | 6 | 275.6 |
| remainder (7 files) | 34 | 383.9 |

**The ArWen physics seam owns 265 of the 385 arrays and 3,085.4 MiB of the
5,791.7 MiB resident tier: 53.3 %.** The dycore state the port itself holds
is the smaller half. What that seam keeps live between steps is four parallel
tendency sets (`tendencies`, `pbl_tendencies`, `radiation_tendencies`,
`cumulus_tendencies`, `physics.py:1270-1414`), a `last_ysu` result set
(`ysu.py:93`), a `_last_gwdo_result` set (`cuda_gwdo_v841.py:1523-1530`), a
WSM6 scratch set (`mpas_column_batch.py:372`) and the seam's `_in`/`_out`
mirrors (`mpas_column_batch.py:552-562`).

The ledger records all 385 arrays under `resident_arrays`, with dtype, x4
shape, B/cell and site.

### Tier 2. Transient, 47,728 B/cell (the largest slope contributor)

3,117.5 MiB at x1 → 8,710.7 MiB at x4, on top of the resident tier, live
simultaneously at the global peak (both peaks land in `step_integrate`).

The peak lands in a **different scheme at each mesh**, which is itself the
point:

* x1 peak is radiation-bound: `rrtmg_sw.py:3466` alone holds 1,745.6 MiB,
  plus `rrtmg_mcica.py:617` 392.0 and `rrtmg_sw.py:3471` 299.2. These are
  fixed-chunk column batches: they do **not** grow from x1 to x4, so they
  belong to the fixed term in disguise and dominate only small meshes.
* x4 peak is dycore-bound: `cuda_driver.py:3222/3253`, `cuda_horizontal.py:991`
  and `cuda_backend/recovery.py:267` at 412.5 MiB each, then six
  `cuda_horizontal_v841.py` operator temporaries at 309.4 MiB each.

So the slope is set by horizontal-operator and acoustic-substep temporaries,
not by state. Full per-site lists for both peaks are in `LEDGER-TABLE.json`
under `peak_instant_sites` (139 sites at x1, 188 at x4).

### Tier 3. Allocator, 645 B/cell + 813.3 MiB fixed

`pool.total_bytes()` sits 838.6 MiB (x1) / 914.2 MiB (x4) above the live
peak: CuPy best-fit rounding plus free blocks the pool keeps rather than
returns. Nearly mesh-independent, and it is the cheapest MiB in the ledger to
recover: a pool policy change reaches it without touching any physics.

## 3. What is not attributed

**103.1 MiB at x1, 178.0 MiB at x4** of the non-pool term (1.3 % and 2.2 %),
scaling at 639 B/cell. Ruled out by measurement, not by argument:

* module images, all eight physics modules measured at 4.0 MiB total;
* kernel local frames, every kernel in the four physics modules had its
  `local_size_bytes` read from the driver, and every frame > 0 was launched
  and measured; the two zero-local controls reserved exactly 0.0 MiB;
* pool allocations, the hook sees every one and they are in tiers 1-3;
* sibling-process contamination: every number here is this process's own
  `nvidia-smi` row, which another process cannot move.

That it scales with cells says the remainder is a small per-cell allocation
taken outside the CuPy pool. It is not identified.

## Instrument

`tools/device_memory_ledger/`, validated against known answers in both
directions before any number here was believed:

| instrument | validation |
|---|---|
| `kernel_reservation.py` | 512 MiB array → hook, pool and `nvidia-smi` all move by 512 MiB; a view moves nothing; an 8,192 B local frame reserves 1,784.0 MiB by `cudaMemGetInfo` **and** 1,784.0 MiB by this process's `nvidia-smi` row; a zero-local kernel reserves 0; a second launch reserves 0 |
| `reservation_probe.py` | positive control `gf_gfdrv_stage` standalone = 7,034.0 MiB = the sum of the three in-run increments; negative control (zero-local kernel) = 0.0 MiB; reverse-order control drives `rlw_rtrn_march` to 0.0 MiB |
| non-perturbation | traced vs untraced x1 run differ in 1 of 1,935 receipt leaf fields, and that field is `memory_admission/free_bytes` (free VRAM at admission, moved by other processes on the card) |

Two traps this instrument had to survive, both of which silently produce
wrong ledgers:

1. **`cudaMemGetInfo` is device-wide.** On a shared card another process's
   allocation lands inside any window measured that way; one contended run
   here reported a 7,052 MiB radiation delta and a *negative* surface-layer
   delta. Every number in this ledger comes from the per-process row instead.
2. **A post-run `gc.get_objects()` kernel scan misses the biggest frame.**
   `RawModule.get_function` returns a fresh `RawKernel` that
   `woof/core/gf.py:166` rebinds on every call, so by the end of the run the
   29,264 B frame is unreachable and a scan reports a maximum local frame of
   9,232 B, understating the largest single allocation in the process by
   4,990 MiB.

A third route stays out of reach by design: RRTMG compiles through
`compile_using_nvrtc` + `cupy.cuda.function.Module` to avoid CuPy's appended
`-ftz=true`. Proxying that class is not an option: CuPy's own Cython
elementwise path constructs and type-checks it, and a proxy kills every cupy
expression in the process with `Cannot convert TracedCudaModule to
cupy.cuda.function.Module`. Those kernels are measured out of band by
`reservation_probe.py` instead.

## Correction to commit ff28ef5

That commit's subject says "the pool sees 89% of the rest". The figure is not
supported by anything measured here and no number in this document produces
it. The measured shares are: the CuPy pool accounts for **98.6 %** of the
slope (85,437 of 86,630 B/cell) and **21.1 %** of the fixed term (2,066.8 of
9,797.8 MiB); as a share of the whole footprint it is 41.0 % at x1 and
66.1 % at x4. Forward commits only, so the subject stands as written and this
note is the correction of record.

Commit f531d04's own body repeats the same fixed-term share as 16.4 %. That
is arithmetic error, not measurement: 2,066.8 / 9,797.8 = 21.1 %. The four
percentages in the paragraph above are the checked ones.
