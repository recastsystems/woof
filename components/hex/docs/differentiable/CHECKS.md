# The checks

Every check a window result is graded by: the bar, and where the number is in the receipt (`window.json` in
`--out`). RUNNER.md has the passes that produce them.

| check | bar | where in the receipt |
|---|---|---|
| **dot** | rel_plain or rel_aligned <= **2e-5** (float32 window, `DOT_F32`), and 0 non-finite tangent words. Per-launch rows: **1e-10** (float64, `AD_NATIVE_TOL`) | `dots[].{y_Mx, x_MTy, rel_plain, rel_aligned, pass}` |
| **zero direction** | **0** nonzero words after the whole window from x = 0 | `zero_direction.{nonzero_words, pass}` |
| **two-process bitwise** | the second reverse in-process differs in **0** words, and a second process with the same arguments gives the same `result_sha256`. Needs `--stable-dir 1` (x and y identical across processes) | `dots[0].reverse[1].differing_words_vs_first_reverse`; `dots[0].reverse[*].result_sha256` across two receipts |
| **boundary hashes** | at every step of every forward the window runs (all seeds, the zero direction, the redo of every reverse step), every carried allocation's sha256 equals the recording's ordinary-forward reference (`reference.jsonl`). Each reverse step's restore + redo differs from the stored boundary in **0** words. Any miss fails the run | stderr `window_tangent_step ... "boundary_hashes":"equal"`; `dots[].reverse[].steps[].redo.differing_words` |
| **Taylor V** | ratio = dJ / (h <x', g>) and err = abs(ratio - 1) over the h curve: **min err <= 1e-2 for a whole short forecast** (1e-3 for smooth unit cases) inside a **resolved interval** (err falls with h, then rises below the float32 floor: a V), and the finite-difference quotients must converge. A flat or non-converging FD quotient is a property of the objective, not a pass or fail of the gradient | `taylor.{rows[].{h, dJ, fd_quotient, ratio, err, error}, best_err, fd_convergence}`; with `--tangent-taylor-dir 1`, `tangent_dir.{w_Mx, x_MTw, rel}` |
| **gradients off** | (1) the forecast trees (`tree/src`, `tree/rust`, `tree/tools`) are unchanged by the differentiable engine; (2) the strict replay `ad-binder replay`, run in A B B A order (A = the previous binary, B = the candidate), has **0** differing process-stable hashes; (3) B is not slower than A | replay stdout `state_sha256` rows, `rc`, wall time |

`milestone_a_numbers_pass` in the receipt is true when every dot passes, the zero direction did not fail and the
second reverse differs in 0 words.

## The forms of the window dot

The dot row above is one test with a choice of second vector. The receipt names the form that ran (`check_form`).

| form | words | what it is | where in the receipt |
|---|---|---|---|
| **self** | `AD_DOT_SELF=1` | y is the tangent's own result at unit weight per unit, so <y, M x> is the number of units that carry tangent and nothing cancels. The bar is the program's: within **4 float32 roundings of the dot**, one rounding being 2^-23 x max(|<y, M x>|, |<M^T y, x>|) with both sums in binary64, so 4.77e-07 on `rel_aligned` (`AD_DOT_BAR`, which the door sets from its table's `thresholds.self_dot_bar`; `relative:1e-7` is the strict setting; each dot's signed reading is `dots[].roundings`). Over 8 or more directions the door also asks that the readings do not lean one way (mean within 2 standard errors of zero) The units are the **model's prognostic fields at the window end** (the package's `alloc-map.json`, boundary steps + 1; boundary 1 is the recorded start). `AD_DOT_SELF_UNITS=pool` scores memory allocations instead: a diagnostic, since one allocation holds scratch words that weigh as much as a field. When the map does not hold the window end (a 15-step window on a package recorded with 15 boundaries), cannot place every field in a tangent-carrying allocation, or the plan is a rank plan, the units are the pool's and `units_why` says so: nothing changes silently and the default never stops a run. A map named with `AD_DOT_SELF_FIELDS` that cannot give the units stops it | `self_dot.{units, fields_spec, units_why, allocations, words}`, `dots[]` |
| **independent** | no `AD_DOT_SELF`, `--direction fieldxy --y-scale 1` | y is a seeded random vector that is not made from the tangent, and x is additive (u times the field's rms, not the word's own value). It answers the doubt that a y built from M x could hide a fault. The receipt applies the same bars to it as to any dot (`pass` at 2e-5, `valid_derivative` at the program's bar, 4 roundings of the dot, on `rel_aligned`); a random y lets the two sums cancel, so read it beside `miss_roundings` (`AD_DOT_ROUNDINGS=1`: the miss in float32 roundings of the sums' absolute terms) | `dots[].{rel_aligned, miss_roundings}`, `direction`, `valid_derivative` |
| **per species** | `AD_DOT_SELF=1 AD_DOT_PLANES=scalars`, with `AD_DOT_SELF_SPLIT=planes` | the 13 species of the stacked `scalars` field each get their own weight and their own unit, at both ends of the window, and each plane's own dot is read with one more reverse. Without it the species share one unit set by the largest words (the number concentrations), and a wrong term on a rain, snow or graupel path can pass unseen | `planes`, `self_dot.planes[]`, `self_dot.planes_without_tangent`, `self_split[]` |

What the per-species form says about the shipped hrrrv5 set: the number concentrations of cloud ice, rain, graupel and
cloud droplets (ni, nr, ng, nc) hold values at the window end and **carry no tangent** there, because the shipped
linear model of the cloud scheme holds them. The window check therefore tests no number-concentration path; the
runner names those planes in `self_dot.planes_without_tangent` and on stderr (`window_planes_without_tangent`). The
six mass species, the two aerosols and graupel volume carry tangent and are tested. On the 12 km cut (14 steps of 60 s,
the transport limiter held) this form misses on rain: the rain plane's own dot is off by -2.2e-05 of itself
(-42.9 roundings) on test direction 1 and +8.0e-06 (+18.8) on direction 2, against the bar of 4 ([LIMITS.md](LIMITS.md)).

The per-launch transpose check of a replay (`AD_DOT_PERSIST=1`) builds every topology table from the run's own mesh.
A stored table that was captured on another mesh is refused by name (`AD_DOT_LOCAL_TOPOLOGY=0` asks for the stored
tables), so a per-launch reading is never one mesh's tables on another mesh.

Per-launch checks (one captured launch, not a window) use `ad-binder capture-dot` (ad.entry/v1 rows) and
`ad-binder capdot` (scheme rows), plus each generator's proof harness (GENERATORS.md). Their bar is the dot at 1e-10
in float64 or 2e-5 in float32, the zero direction 0, and the repeated reverse bitwise.

## Example arguments

A dot leg (seed 1, two reverses, no physics scheme held; the kernels the window's manifest classes as held,
permutation or constant still run as identity, [RUNNER.md](RUNNER.md) section 6) and a Taylor leg on a recorded
18-step hrrrv5 window. The working directory is the recorded plan package.

```bash
K=KSET            # build_kset_int7f.py --out KSET
COMMON="--config plan.json --steps 18 --kernels $K/kernels.json --cubins $K/cubins --tape-gib 2 --scratch-gib 1 \
        --schemes tree/ad/phys/window/schemes18.json --anchors 2 --fct-frozen 1"
DIRS="--native-schemes 1 --direction field --stable-dir 1 --x-scale 1e-12 --y-scale 1e-20"
export CUDA_MODULE_LOADING=EAGER AD_SCHEMES=BUNDLE      # tree/ad/phys/bind/build_bundle.sh output
ad-binder window $COMMON --out runs/dot --bstore bstore/dot --seeds 1 --repeat-reverse 1 --zero 0 $DIRS
ad-binder window $COMMON --out runs/taylor --bstore bstore/taylor --seeds 1 --tangent-only 1 --zero 0 \
    --objective objective.json --taylor 1,3e-1,1e-1,3e-2,1e-2 --tangent-taylor-dir 1 --taylor-dir dir.json $DIRS
```

An objective file is `{"name", "terms":[[allocation, word, weight], ..]}` (sparse weights at the window end) or a
`form` objective; a Taylor direction file has the same shape.
