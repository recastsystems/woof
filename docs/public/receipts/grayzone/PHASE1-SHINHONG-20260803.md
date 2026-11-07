# Gray-zone campaign Phase 1: Shin-Hong (bl_pbl_physics=11) on the frozen ladder

Registered expectation:
`2026-08-03-shinhong-grayzone-expectation.md`
(commit `31275170`), written and committed BEFORE any run with
`--pbl 11` existed.  Frozen baseline and bands:
`PHASE0-BASELINE-20260803.md` at `67ac01b1`.  Neither document ships --
both are development-campaign records; the README beside this file says
what each one held and where its load-bearing content is restated.
Instrument: commit
`82c8155b` (D1, `gpuwm.verify.cases.cbl_dry.partition_run` driven by
`tools/grayzone_phase0.py`), scoring path byte-identical to the frozen
baseline's -- `gpuwm/verify/gray_zone.py` diffs ZERO against `82c8155b`,
and the only `cbl_dry.py` change against it is a comment block declaring
the `km_opt=4` fall-through.  Tree under test: `61488333`, the merge of
the shipped `integration/release-1.5.1` (`cf159eb2`) into
`development branch`.  Card: the local RTX 5090 (no ECC; determinism
pairs are the corruption screen).  Every number below is reproducible
from `shinhong_runs.jsonl` beside this file (42 JSON receipts, npz
profile stacks under `npz/`) via
`python tools/grayzone_phase0.py score --ledger <that file>`, plus spec
I1's `sigma = max(frozen, own)` step applied on top.

## Protocol facts (as run, deviations declared)

* Registered budget spent EXACTLY: 36 scored runs (6 rungs x 6 seeds
  20260801..20260806, all at `repeat 0`) + 6 determinism rerolls (seed
  20260801 a second time at each rung, `repeat 1`).  42 runs, no more.
  No run crashed; no cell is missing; nothing was re-run.
* 64x64 columns, nz=40 to 2 km, the registered `cbl_dry` sounding and
  surface contrast; dx ladder 3200/1600/800/400/200/100 m; dt per rung
  15/7.5/4/2.4/1.2/0.6 s.  4 h runs; scored scalar = mixed-layer
  (0.05 <= z/h <= 0.85) mean SUBGRID TKE fraction over 12 snapshots at
  300 s in the final hour.  The resolved fraction is one minus this, as
  Honnert defines it; the ladder is reported in the subgrid convention
  the instrument and the bands are written in.
* `sweep_config(dx, 11)` resolves `km_opt=4` -- vertical transport from
  the PBL scheme, horizontal Smagorinsky -- the reading declared in the
  spec before the runs.  `supplies_own_mixing` untouched.
* h is the run's own S3-6f bulk-Richardson depth; x = dx/h at the
  candidate's own x, per the band rule.  Shin-Hong's h is NOT the
  baseline's: it matches SASE at 800 m and finer (1029.5 m vs the
  control's 1029.2 m) and shallows at the two coarsest rungs (803.3 m at
  1600, 663.8 m at 3200, against the control's 986.1 m at 3200).  Every
  rung is therefore scored LEFT or RIGHT of HEAD's column exactly as the
  "run's own x" rule requires; the coarse rungs move RIGHT.

## Scored ladder (spec I1 rule, candidate's own x)

`sigma_used = max(frozen HEAD sigma at the rung, this leg's own)`;
band = envelope +/- 2*sigma_used/sqrt(6).  HEAD column is the frozen
Phase-0 seed-mean, quoted for contrast only -- it is not a band input.

| dx [m] | x (own) | h [m] | SH seed-mean | sigma_own | sigma_used | eq. (9) | envelope | band (I1) | verdict | HEAD mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 3200* | 4.8211 | 663.8 | 0.999984 | 0.000001 | 0.005399 | 0.9882 | [0.9775, 0.9990] | [0.9731, 1.0034] | IN | 0.446239 |
| 1600 | 1.9919 | 803.3 | 0.979367 | 0.000203 | 0.002029 | 0.9562 | [0.9248, 0.9876] | [0.9232, 0.9893] | **IN** | 0.419366 |
| 800 | 0.7770 | 1029.5 | 0.827530 | 0.001929 | 0.001929 | 0.8329 | [0.7632, 0.9026] | [0.7616, 0.9042] | **IN** | 0.351014 |
| 400 | 0.3718 | 1075.9 | 0.525066 | 0.002470 | 0.002470 | 0.6152 | [0.5135, 0.7168] | [0.5115, 0.7189] | **IN** | 0.293873 |
| 200 | 0.1843 | 1085.2 | 0.303247 | 0.001633 | 0.001633 | 0.3734 | [0.2545, 0.4924] | [0.2531, 0.4937] | **IN** | 0.248733 |
| 100 | 0.0919 | 1088.3 | 0.204203 | 0.001908 | 0.003008 | 0.2083 | [0.0939, 0.3228] | [0.0914, 0.3252] | **IN** | 0.202912 |

*3200 m is advisory, never gated (spec I1).  Recorded: it lands inside
its band too, but on 2x2-block-degenerate ground and at an x the frozen
baseline never occupied.  It is not part of the verdict.

Every gated rung (1600/800/400/200/100) is INSIDE its band.

## The 36 scored cells

Seed-mean is the scored value; the six columns are the six seeds
(20260801..06).  These 36 numbers are the whole scored leg.

| dx [m] | ..01 | ..02 | ..03 | ..04 | ..05 | ..06 | seed-mean |
|---|---|---|---|---|---|---|---|
| 3200 | 0.999983 | 0.999984 | 0.999983 | 0.999984 | 0.999984 | 0.999984 | 0.999984 |
| 1600 | 0.979172 | 0.979332 | 0.979116 | 0.979574 | 0.979401 | 0.979610 | 0.979367 |
| 800 | 0.827918 | 0.824730 | 0.828522 | 0.827975 | 0.825901 | 0.830133 | 0.827530 |
| 400 | 0.527128 | 0.522552 | 0.522415 | 0.523788 | 0.526336 | 0.528176 | 0.525066 |
| 200 | 0.302418 | 0.304870 | 0.300735 | 0.304901 | 0.302584 | 0.303971 | 0.303247 |
| 100 | 0.203208 | 0.203494 | 0.206939 | 0.205394 | 0.201459 | 0.204726 | 0.204203 |

## The registered criteria, line by line

1. **Monotone climb (the qualitative kill criterion): MET.**  Ordered by
   ascending x (dx = 100, 200, 400, 800, 1600), the seed-mean subgrid
   fraction reads 0.204203 -> 0.303247 -> 0.525066 -> 0.827530 ->
   0.979367.  Monotone at every step, and the steps are enormous
   relative to the noise: max-min across the gated rungs is 0.775164
   against an F1-flatness threshold of 2x the largest sigma_seed =
   0.004941, i.e. the response is 157x the width that would have been
   called flat.  This is the opposite of the F1 signal; no wiring
   investigation is triggered.  For contrast, HEAD moved 0.2029 ->
   0.4194 (0.2165) over the same span, and every closure in the tree,
   including WRF's own km_opt=2 run coarse, sat flat.
2. **Band verdicts (the scored claim): every gated rung IN.**  The
   registered "earned expectation" was the two coarsest gated rungs
   (1600, 800) landing inside; both do.  The spec declined to make a
   strong claim at 400 and 200 m, where it recorded the literature as
   genuinely uncertain -- both land inside as well, and are reported
   here as the spec required, exactly as they fell.  The 200 m rung sits
   at x = 0.1843, still just below the 0.2 gray-zone knee, as the
   Phase-0 note anticipated for the 4 h window.
3. **The anchor holds (F2 inherited verbatim): HELD.**  dx = 100 m gives
   0.204203 at its own x = 0.0919, inside [0.0914, 0.3252].  F2 asked
   whether the candidate broke the one point SASE already had: it did
   not -- it reproduces it to within 0.0013 (HEAD 0.202912) while
   changing every other rung.  The candidate improves the gray zone
   without paying for it at LES resolution.
4. **Instrument tripwire: NOT tripped.**  Largest sigma_seed anywhere is
   0.002470 (400 m), two orders below the 0.03 stop.
5. **Determinism pairs: BITWISE IDENTICAL at every rung.**  SHA-256 over
   the full per-sample profile stacks, same seed twice:

   | dx [m] | seed | runs | digest (first 16) | verdict |
   |---|---|---|---|---|
   | 3200 | 20260801 | 2 | `c69864c1a32eeeee` | bitwise identical |
   | 1600 | 20260801 | 2 | `c533cb28c441a42d` | bitwise identical |
   | 800 | 20260801 | 2 | `0e47e268c1b49886` | bitwise identical |
   | 400 | 20260801 | 2 | `31f92359e9d6a9fb` | bitwise identical |
   | 200 | 20260801 | 2 | `5953d42481386a0a` | bitwise identical |
   | 100 | 20260801 | 2 | `1355628a97c66adb` | bitwise identical |

   The 3200 m cell additionally carries a THIRD match from a separate
   process run later for the VRAM measurement
   (`c69864c1a32eeeea8736bd7b569857249a2609843132273a6fbe1d9c820c418`,
   full digest equal across all three).  The no-ECC dual-run screen is
   green; no corruption signature anywhere in Phase 1.

**Spec I2 PASS condition -- every gated rung inside its band -- is MET.**

## Controls: the delta is the closure, not the tree

The merged tree still reproduces the frozen SASE baseline.  Two
non-scored `--pbl 900` control runs (separate ledger, never in the
scored leg, seed 20260801):

| dx [m] | control | frozen HEAD seed-mean | frozen sigma | agreement |
|---|---|---|---|---|
| 3200 | 0.439331 (x 3.2450) | 0.446239 (x 3.2403) | 0.005399 | 1.3 sigma |
| 800 | 0.349112 (x 0.7773) | 0.351014 (x 0.7782) | 0.001533 | 1.2 sigma |

Both are single seeds against a six-seed mean, so ~1 sigma is what
agreement looks like.  The merge did not move the baseline; the ladder's
0.20 -> 0.98 climb is Shin-Hong.

Dispatch is independently proven, not inferred from the scores:
`tests/test_shinhong_runtime.py` is 6/6 green on this tree, including
`test_driver_compute_dispatches_11_to_run_shinhong` and
`test_shinhong_step_publishes_its_tke_into_e_sgs` -- the publication
step that puts the scheme's own TKE into `state.e_sgs`, which is the
field the instrument scores.

## Parity sanity gate (the stop condition)

`tests/test_shinhong_wrf461_parity.py`: **22 passed**, run on the merged
tree before the ladder.  The merge moved no Shin-Hong pin; max ULP 0
against the byte-frozen `module_bl_shinhong.F` still holds.

## Non-regression as measured (NOT adjudicated here)

Full CPU suite on the merged tree, `pytest -q -m "not gpu"`:
**7 failed, 7888 passed, 130 skipped, 2982 deselected, 1 xfailed in
53:43**.

Green, as the spec's registered non-regression predicted: the YSU and
MYNN parity suites, `tests/sase_goldens.py` values,
`tests/data/config_freeze_golden.json`, and the OFF-path bitwise
inertness tests.  No lane file touches YSU.  Also green, and expected to
be after the merge: `test_certify_hygiene` and all four FTZ tests, which
the merged-in `ce81f975`/`f2c58067`/`7f2f4f21` fixed.

Of the 7 reds, 3 are the environmental baseline (`gpuwm` is not
pip-installed in this interpreter, so `PackageNotFoundError`):
`test_import.py::test_version_is_the_installed_distribution_version`
and both `test_native_wrf_distribution.py` version tests.  The three
`test_render_rust` failures in the 1.5.1 proof pass are now GREEN.

The remaining **4 are lane-owned bookkeeping debt, and they pre-date the
merge**.  Ownership established by comparing the pins across all three
tips rather than by argument: `_ROUTED_COMBINATIONS = 33` and
`len(report["rows"]) == 1386` are IDENTICAL at `cf159eb2` (release),
`cfcfa9a9` (pre-merge lane) and `61488333` (merged).  The release line
never moved them; the lane moved the TREE under them by making
`bl_pbl_physics=11` routable, and did not re-measure the counts.  The
only count-feeding file the merge touched is `physics_compat.py`, and
its 37 added lines are an unrelated Thompson export-string helper.

| test | measured | cause | owner |
|---|---|---|---|
| `test_health_integer_policy::test_the_routed_cross_product_is_the_size_it_was_measured_at` | `assert 41 == 33` | scheme 11 joins the routed cross-product | lane, pre-merge |
| `test_health_field_census::test_noahmp_slice_matches_the_current_wrf_authority` | `assert 1722 == 1386` | same cause, census rows | lane, pre-merge |
| `test_evidence_axes::test_the_walk_receipt_regenerates` | `docs/public/receipts/F5-maturity-surface-walk.json` stale: `"component-option"` 28 -> 29 | same cause, one new component option, public walk receipt not regenerated | lane, pre-merge |
| `test_release_snapshot_machine_paths::test_the_staged_release_tree_carries_no_machine_paths` | 4 developer-absolute POSIX home-directory paths (pattern deliberately not reproduced here, so this receipt does not become a fifth offender) | 3 lines of `gpuwm/data/shinhong/oracle/oracle-sha256sums.txt` + `tools/shinhong_wrf461_oracle/README.md:26`, all lane-added files absent at `cf159eb2` | lane, pre-merge |

None of the four touches Shin-Hong physics, the parity pins, the
instrument, or the ladder.  Three of them are the arithmetic consequence
of the feature working -- a new scheme IS routable now -- against pins
that were never re-measured, and the fourth is oracle provenance text.

Recorded, not adjudicated: the spec's registered non-regression bullet
claims "the full CPU suite (`pytest -m "not gpu"`) ... pass unchanged on
the lane tip".  As measured, that bullet is FALSE, and F3 reads "any
non-regression gate N1-N4 REDs" kills the candidate.  The scored ladder
above is reported on its own terms and is not softened by it.

### Addendum, 2026-08-03, written after the ladder

All four bookkeeping items were re-measured on this branch and closed,
each in its own commit stating its own derivation.  No pin was widened:

| item | commit | movement |
|---|---|---|
| routed cross-product | `c8bbdf03` | 33 -> 41, scheme 11's own 8 admitted (sfclay {1, 91} x lsm {0, 2, 3, 4}); 8 refused |
| F5 maturity-surface walk | `9d55d19d` | one component option, counted on the four axes it touches; no measured value moved |
| four-domain health census | `d2c1c0c5` | 1722 rows / 1638 rejected, scheme 11's own 336 of each; peak 632 and headroom 392 unchanged |
| oracle developer paths | `32c2bbb4` | 4 path strings relativized, every digest byte-untouched |

Three of the four were the arithmetic of the feature working, as the
table above already said; the fourth was oracle provenance text.  None
of them touched Shin-Hong physics, the parity pins, the instrument or
the ladder, and the tree says so rather than the argument: the 3200 m /
seed 20260801 cell was re-run twice more, once at `32c2bbb4` and once
after every commit above, and both return scored value
`0.9999831472581168`, h `663.7594056419367`, x `4.821023962598628` and
digest `c69864c1a32eeeea8736bd7b569857249a2609843132273a6fbe1d9c820c418`
-- bit for bit the ledger's.  With the pair in the table above and the
VRAM-measurement run, that cell now has FIVE independent bitwise
matches.  `tests/test_shinhong_wrf461_parity.py` is 22 passed and
`tests/test_shinhong_runtime.py` 6 passed on the post-fix tree, the same
counts recorded above.

The three remaining reds are the environmental baseline named above
(`gpuwm` not pip-installed in that interpreter); they are not tree
defects and nothing in this branch can close them.

## Cost

* Wall: 42 scored/reroll runs in 3099.1 s of run time = 51.7 min,
  co-tenant with the full CPU suite on the same box for its whole
  duration (16:13-17:05 local).  Per rung: 50.4 / 98.2 / 209.4 / 428.7 /
  796.5 / 1515.9 s coarse-to-fine.
* VRAM: 3016 MiB idle (desktop/WDDM) -> 6781 MiB peak during a run, so
  ~3.7 GB for the case on a 32.6 GB card.  Grid is 64x64x40 at every
  rung, so the footprint does not vary with dx.
* Receipts: `shinhong_runs.jsonl` 112 KB + 36 npz profile stacks 592 KB
  = 0.7 MB, well under the ~50 MB declared before launch.  No node time.
