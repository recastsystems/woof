# The window runner (`ad-binder window`)

How the window runner is built, what each pass does, the anchors and segments of the reverse, the derivative rule of
every command, and every flag and environment switch.

| file | what |
|---|---|
| `tree/ad/phys/binder/runtime/src/binder/phys/main.rs` | the replay runner (`Execution`, `finish_step`), subcommand dispatch |
| `tree/ad/phys/binder/runtime/src/binder/phys/window.rs` | the window: passes, boundary store, reverse segments, directions, dots, Taylor, dumps |
| `tree/ad/phys/binder/runtime/src/binder/phys/wrules.rs` | the derivative rule of every non-entry command (enum `R`) |
| `tree/ad/phys/binder/runtime/src/binder/phys/schemes.rs` | the L2 scheme rows (`AD_SCHEMES` bundle): P3, P4, P6, P7 LW/SW taumol, RUC land |
| `tree/ad/phys/binder/runtime/src/binder/phys/capi.rs` | `libhexad-phys`, the C ABI over the window |
| `tree/ad/e2e/src/{deriv,dev,dot,expr,recipes,held}.rs` | ad.entry/v1 binding (deriv.rs), device helpers, the held manifest (held.rs); compiled in by `#[path]` |

## 1. Build

Crate `ad-binder-phys`, directory `tree/ad/phys/binder/tools/binder/phys/runner/`; the binary `ad-binder` is
`tree/ad/phys/binder/runtime/src/binder/phys/main.rs`. Dependencies are pinned exactly in `Cargo.lock` (serde,
serde_json, sha2, flate2 with zlib-rs).

`build.rs`:
- refuses unless `step_submit.c` has its pinned sha256 (the hex step submitter the recording used);
- compiles `step_submit.c`, `native_cub_bridge.cc` and `native_thrust_bridge.cc` with the system `cc`/`c++` (-O2);
- **only when `NVCC` is set**, builds the fatbins that `include_bytes!` embeds: `e2e_support.fatbin`
  (`tree/ad/e2e/cuda/e2e_support.cu`), `p8_rules.fatbin` (`tree/ad/phys/p8/cuda/p8.cu`), `window_support.fatbin`
  (`tree/ad/phys/window/cuda/window.cu`) and `hexad_t.fatbin` (`tree/ad/phys/window/cuda/hexad_t.cu`), each
  `-fatbin -O3 --fmad=false --prec-div=true --prec-sqrt=true` for sm_120, sm_89 and compute_80. Without `NVCC`
  they are empty and `window` fails at module load, so a window binary always needs `NVCC`.

```bash
cd tree/ad/phys/binder/tools/binder/phys/runner
NVCC=/path/to/nvcc cargo build --release
cargo test --release
```

The bytes depend on the rustc, the C compiler and nvcc versions (the fatbins embed nvcc's output), not on the
checkout path, the target directory or the job count.

## 2. Subcommands

`ad-binder replay | window | capture-dot | capdot | linearize` (main.rs `run()`).
- `replay --config PLAN --steps N [--trace P] [--reference P] [--poison P] [--checkpoint-bindings-at STEPS
  --checkpoint-hook P] [--word-reference-socket P --word-image P --word-image-sha256 HEX]`: the strict
  gradients-off replay. Physics plans need the ordinary-forward reference. This is the path the ABBA check runs.
- `window ARGS`: this page.
- `capture-dot` (capentry.rs): the per-launch dot of one ad.entry/v1 row on one captured launch.
- `capdot` (capdot.rs): the per-launch dot of one scheme row on one captured launch.
- `linearize < driver.jsonl`: plan linearisation from a driver log.

## 3. Argument form

- Every window argument is a `--key value` pair (window.rs `run`). A key without a value is an error, there are no
  bare switches, and on/off flags take `1` / `0`. The parser keeps the last value of a repeated key.
- The process cwd must be the plan package directory: `--config plan.json` is relative, and the runner reads its
  neighbours from there:
  - `catalog-a.json` (the fixed allocations);
  - the plan's `reference` (reference.jsonl);
  - the held blobs.

## 4. The passes, in order

`run()` in window.rs:

0. **Load.**
   - The plan must be `ad.binder.plan/v1` with `complete: true`, then it goes through `preflight`.
   - Fixed argument sources and copy-indexer sources are verified against the plan.
   - The ordinary-forward reference is required.
   - **Kernel-set admission** (`check_spec_hashes`): every row's recorded spec sha256 is checked against its file, on
     all cores, before any image loads. One mismatch refuses the run (KERNEL-SETS.md section 4).
   - `Win::new` then does the following:
     - loads the kernel set (`--kernels`, `--cubins`) and applies the P9 manifest (`--schemes`, `--hold`);
     - builds the `--refuse` list;
     - allocates **one float32 shadow arena** over every plan allocation, mid-step `Allocate` ids included (each id
       at a 64-word-aligned offset);
     - loads the P8 rule and window support modules, the L2 bundle (`--native-schemes 1`) and the TEMPO images;
     - allocates the tape, scratch and TEMPO arenas;
     - marks as **fixed** every allocation whose catalog shows the same sha256 at boundary 0 and at every step
       boundary. Directions put no weight there.
   - It logs `window_loaded`: kernels, dispatchable, shadow bytes, held, refused, native rows and the TEMPO image shas.
1. **Augmented forward with the tangent, per seed** (`--seeds`, default `1,2`):
   - x is filled from seed `0x5eed00000000 + 2s + 1`, and y from `+ 2s + 2` (directions: section 7).
     `--tangent-taylor-dir 1` seeds the first tangent from the `--taylor-dir` terms instead.
   - Each step runs every command through the runner's own `Execution::command` (the strict replay) plus its rule
     (section 6).
   - `finish_step` then checks every carried allocation's sha256 against the reference. Any mismatch fails the leg.
     These are the **boundary hashes** (CHECKS.md).
   - On the first seed only, each step boundary is **stored**:
     - every live allocation goes to `--bstore/b{s}.bin`, written by a background thread;
     - the host side stays in RAM: the host allocations (the physics frame included), the TEMPO cache and bindings,
       the pointer-translation history, the held pointer words, the CUDA events and the live frame.
   - <y, M x> is formed in two forms, plain and aligned (M - I).
2. **Reverse from y** (skipped with `--tangent-only 1`):
   - The steps run last to first through `reverse_step` (section 5).
   - The first seed runs two reverses (`--repeat-reverse`, default on). The second reports
     `differing_words_vs_first_reverse`, which is the in-process half of the bitwise check.
   - `result_sha256` is the hash of the whole cotangent arena.
   - A seed **passes** when the tangent has no non-finite word and either rel_plain or rel_aligned is at most 2e-5
     (`DOT_F32 = 2e-5`).
3. **Zero direction** (`--zero`, default on): restore boundary 0, zero the tangent and run the forward N steps with
   the tangent. `nonzero_words` must be 0.
4. **Objective reverse** (`--objective a.json[,b.json..]`):
   - y is the first objective's sparse weights. For a `form` objective (the smooth log-sum-exp reflectivity, `Form`
     in window.rs), y is dJ/dx_N at the base window end, computed in float64 and rounded once.
   - One reverse runs. `--dump-grad id,..` writes `grad-<id>.f32`, and <x', dJ/dx0> is reported.
5. **Taylor rows** (`--taylor h1,h2,..`). For each h:
   - restore boundary 0;
   - perturb by h x', where x' is seed 1's relative direction with the address-bearing and fixed allocations
     masked, or by h times the `--taylor-dir` terms;
   - run the unchecked nonlinear forward.
   - The row reports dJ, dJ/h, ratio = dJ / (h <x', g>) and err = |ratio - 1|. It also reports every objective's
     own FD quotient, the changed words, the `--regime` flips and the `--dump-state` fields.
   - A forward error at step s becomes the row's `error`, and the leg continues.
6. **Receipt.**
   - `window.json` is rewritten after every seed, after the objective and after every Taylor row.
   - It holds: the args, stages, dots, zero_direction, objective, taylor (rows, best_err, fd_convergence),
     commands_by_rule, refused, held_upload_dropped_cotangent_sq, tempo_live stats, the boundary store size, the tape
     and scratch peaks, and the wall time.
   - `milestone_a_numbers_pass` is true when every dot passes, the zero direction did not fail, and the second
     reverse differs in 0 words. It is false on `--tangent-only` legs, which run no dots.
   - The boundary files are deleted unless `--keep-bstore 1`.

## 5. Reverse step: anchors, restore + redo, segments

`reverse_step` (window.rs) reverses one model step, given the cotangent at its end:
- **Pass 1: anchors, then restore + redo.**
  - Restore the step's start boundary and replay all n commands of the step. Along the way, keep up to `--anchors`
    whole-device snapshots in RAM, at stride ceil(n / (anchors + 1)).
  - At the end, compare every live word with the stored end boundary. One differing word refuses the step
    ("restore + redo differs").
  - The receipt row `redo` carries differing_words, residue_words and anchors.
- **Segments, last first.** For the range [lo, end):
  - restore the nearest anchor at or before lo (else the start boundary) and replay up to lo;
  - tape lo..end into the tape arena (`--tape-gib`): the pre-launch primal spans, the `post` copies, the masks and
    the max inputs;
  - walk the tape back.
  - If the tape overflows at command k, the segment is split there and retried from the nearest anchor. A single
    command that alone does not fit is an error; raise `--tape-gib`.
- **Host branches** replay their recorded outcome from the flat command list.
- **Fresh allocations:** an allocation made inside a step starts from zeros in every pass (G2), so no pass reads what
  an earlier pass left behind.
- **Prefetch (G5):** the next step's blobs are decoded on all cores in the background, and its start boundary is
  read ahead.
- **After the step,** the P8 failure flag must be 0; otherwise the step fails with "row transpose saw a non-finite
  cotangent".
- **Receipt row per step:** commands, segments, overflow_splits, commands_replayed, tape_peak_bytes, adjoint_entry_s,
  pass1_s, wall_s and redo.

## 6. Rules: what each command does to the derivative

wrules.rs `R`. Every command gets exactly one rule. The tangent and the reverse use the same rule, so the composed
reverse is the exact transpose of the composed tangent.

| rule | applies to |
|---|---|
| `Entry` | a kernel whose kernel-set row has a loaded ad.entry/v1 derivative (deriv.rs binds `tl`/`ad`/`aug` to the shadows) |
| `Rows` | P8 copy rows: post-transport copy, glue copy plan, cuBLAS transpose, broadcasting view copy |
| `View`, `ZeroView` | elementwise float copy; overwrite of a strided view with state-independent data |
| `Max`, `Where` | out = max(c, x) and dst = mask ? src : dst, with x or the mask taped before the launch |
| `Scale`, `Copy`, `Zero` | out = c in; dst <- src; overwrite by a constant (clear, fill, immutable upload, fresh allocation) |
| `Held` | a held upload (a held scheme's step result, a host value): tangent 0, cotangent dropped, summed per origin in `held_upload_dropped_cotangent_sq` |
| `Identity(label)` | no floating state written, or a held physics launch: identity both ways |
| `Refused(name)` | no exact rule, or listed in `--refuse`: identity both ways, counted under `REFUSED:<name>` in the receipt |
| `FctMinmax` | `--fct-frozen 1`: fct_minmax_source with the limiter bounds frozen (FREEZES.md) |
| `Native(row)` | `--native-schemes 1`: an L2 bundle row (P3, P4, P6, P7 taumol, RUC land) through the float32-shadow adapter (widen to float64, run in place, round back once) |
| `Tempo`, `TempoSerial`, `TempoSmall` | TEMPO warp_steps (L5 tempo_adw), tempo_precip and tempo_radar (P5 per-thread entry) |

Classes are assigned before the rules, by held.rs from the P9 manifest (`--schemes`):
- held, permutation and constant kernels get class `held`, which gives the identity rule;
- inactive kernels get class `output-diagnostic`;
- copies and glue are never held;
- kernels with no entry and no rule are refused by name.

For example, an 18-step hrrrv5 receipt with no physics scheme held lists 15 held kernels and 4 `refused_without_rule`: cupy_copy__float32_float64,
cupy_copy__float64_float64, cupy_copy_where__float32_bool_float32 and cupy_maximum__float_float32_float32.

## 7. Directions and scales

| flag | value | meaning |
|---|---|---|
| `--direction` | (absent) relative | x = u |primal at boundary 0|, y = u |primal at the window end|. Words that are not finite or above `WEIGHT_MAX` 2e5 (stale scratch, address words) get 0 |
| | `plain` | uniform [-1, 1) on every shadow word. Taylor refuses it (address words would be perturbed) |
| | `field` | x relative, y = u / rms_f (rms of allocation f's valid words at the window end: finite, at most 2e5, above 1e-30; invalid words 0) |
| | `fieldxy` | also x = u rms_f(boundary 0): an additive x. Without `AD_DOT_SELF` this is the independent form of the window check (CHECKS.md). The operator is unchanged |
| `--x-scale c`, `--y-scale c` | finite c > 0, default 1 | uniform factor on the x or y weights. rel is invariant to c; it keeps a 1e18-fold tangent inside float32. Typical: 1e-12 and 1e-20 |
| `--stable-dir 1` | default off | zero weight on translated address words, relocation slots, address-bearing tables and declared residue ranges, so x and y are the same in every process (needed for the two-process bitwise check) |

## 8. Every window flag

Required: `--config` (plan, relative to cwd), `--steps N`, `--out DIR`, `--bstore DIR`, `--kernels kernels.json`,
`--cubins DIR`.

| flag | default | effect |
|---|---|---|
| `--seeds a,b` | `1,2` | seeds of the dot pass. Seed 1 also drives the Taylor direction |
| `--anchors n` | 3 | RAM snapshots per step in the reverse (2 is enough for 18-step hrrrv5 windows) |
| `--tape-gib g` | 4 | reverse tape arena (2 is enough for 18-step hrrrv5 windows) |
| `--scratch-gib g` | 1 | scratch arena |
| `--repeat-reverse 0/1` | on (anything but `0`) | a second reverse on the first seed, compared word for word |
| `--zero 0/1` | on (anything but `0`) | the zero-direction pass |
| `--tangent-only 1` | off | no reverse and no dots for the seeds (Taylor and objective legs) |
| `--schemes FILE` | none | the P9 held manifest (`tree/ad/phys/window/schemes18.json` for the hrrrv5 x1.2562 window) |
| `--hold s1,s2` | none | with `--schemes`: hold these schemes (`physics` = all; `tempo` holds every TEMPO mechanism) |
| `--refuse f[=why],..` | none | refuse these functions by name for this run (identity both ways) |
| `--fct-frozen 1` | off | frozen FCT limiter (FREEZES.md). Windows longer than a few steps need it |
| `--native-schemes 1` | off | the L2 rows live (needs `AD_SCHEMES`) |
| `--tempo all` (or `1`) / `0` | off | every TEMPO derivative live: warp_steps, precip and radar. Conflicts with `--hold tempo` |
| `--tempo-live 1` | off | tempo_warp_steps only (needs `--tempo-cubin` or `AD_WIN_TEMPO_CUBIN` / `AD_DOT_TEMPO_CUBIN`) |
| `--tempo-cubin P` | - | the sm_120 tempo_adw image built by `tempo-ad-gen` (GENERATORS.md) |
| `--tempo-arena-gib g` | 6 | TEMPO work arena (4 is enough for hrrrv5 x1.2562) |
| `--tempo-precip 1` | off | tempo_precip live (needs `AD_DOT_TEMPO_SERIAL_CUBIN`) |
| `--tempo-small tempo_radar` | off | tempo_radar live (needs `AD_DOT_TEMPO_SERIAL_CUBIN`); any other name is refused |
| `--objective a.json,..` | none | objective reverse. The first file is the gradient objective; every file's dJ/h is reported on each Taylor row |
| `--dump-grad id,..` | none | write the objective reverse's gradient for these allocations |
| `--taylor h1,h2,..` | none | the Taylor rows (relative directions or `--taylor-dir` only) |
| `--taylor-dir FILE` | none | fixed Taylor direction `{"name","terms":[[alloc,word,value],..]}` |
| `--tangent-taylor-dir 1` (alias `--tangent-dir 1`) | off | needs `--taylor-dir` and `--tangent-only 1`, not `--fd-only`: the tangent starts from the Taylor direction, and <w, Mx> is reported beside <x, M^T w> |
| `--fd-only 1` | off | forward only: base plus Taylor rows, FD quotients only. Needs `--taylor-dir`, `--objective` and `--taylor` |
| `--regime FILE` | none | regime words captured at every base boundary and compared on each Taylor row (`eq` / `sign` / `pos`) |
| `--dump-state id[:off:bytes],..` | none | window-end base state, tangent (`tl-<id>.f32`) and per-row perturbed state and FD fields, scored against the tangent |
| `--trace-norm 1` | off | per-step tangent norms and the growth trace |
| `--keep-bstore 1` | off | keep the boundary files |

## 9. Environment switches (all default off)

- **Inputs:**
  - `AD_SCHEMES=DIR`: the L2 bundle (schemes.json `ad.scheme.rows/v1` and its images; KERNEL-SETS.md).
  - `AD_DOT_TEMPO_SERIAL_CUBIN`: the P5 serial TEMPO image.
  - `AD_WIN_TEMPO_CUBIN` / `AD_DOT_TEMPO_CUBIN`: fallbacks for `--tempo-cubin`.
  - `CUDA_MODULE_LOADING=EAGER`: recommended, so module loads do not land inside a timed step.
- **L2 row knobs:**
  - `AD_NATIVE_ONLY=f,..`: only these rows live (one-scheme attribution).
  - `AD_NATIVE_TOL`: the per-launch dot tolerance for rows (default 1e-10).
  - `AD_P3_TILE`: the P3 column tile (default max(row tile, 384)). Different tiles give bitwise equal results.
  - `AD_RUC_LAND_TAPE`, `AD_RUC_LAND_CHUNK`, `AD_RUC_LAND_ADCHECK=1`.
- **The window check's form** (CHECKS.md, "The forms of the window dot"):
  - `AD_DOT_SELF=1`: the self form. Its units are the model's prognostic fields at the window end (the package's
    `alloc-map.json`, boundary steps + 1). `AD_DOT_SELF_UNITS=pool` scores memory allocations instead;
    `AD_DOT_SELF_FIELDS=<alloc-map.json>:<boundary>` names another map or boundary.
  - `AD_DOT_PLANES=scalars` (or the full form `X_ALLOC@X_OFF,Y_ALLOC@Y_OFF/NxPLANE_BYTES[/name,..]`): each species
    plane of the stacked scalars field gets its own weight and its own unit; `AD_DOT_SELF_SPLIT=planes[:name,..]`
    reads each plane's own dot with one more reverse a plane. `AD_DOT_PLANES_CAP=1` keeps the 2e5 cap in the planes.
  - `AD_DOT_LOCAL_TOPOLOGY=0` (replay, per-launch dots): the stored topology tables instead of the run's own mesh's;
    a stored table of another mesh is refused by name.
- **Diagnostics** (never a proof):
  - `AD_ZERO_DERIV=f,..`: a launch's outputs act as constants both ways (bisection).
  - `AD_SUBNORMAL=f,..`, with `AD_CENSUS_DUMP=DIR`: a shadow census before and after a launch.
  - `AD_UNPACK_ZERO=n,..` and `AD_UNPACK_ZERO_OPS=N:i,j`: radiation-unpack rows become fill rows.
  - `AD_TRACE_ALL=step:lo:hi`, `AD_TRACE_GROWTH`, `AD_TRACE_HITS`, `AD_TRACE_STATE=1`, `AD_STEP_NORMS=1` and
    `AD_DOT_ALLOC=1`: traces and norms.
  - `AD_G6_WATCH=f,..`, `AD_TEMPO_SR_EXACT=1`, `AD_TEMPO_PROFILE_LOG`.
- **Replay** (main.rs):
  - `AD_BINDER_HASH=device`: the old one-thread device digest. The default host SHA runs on all cores.
  - `AD_BINDER_QUICK=0`: the spin-wait referee.
  - `AD_REPLAY_SERIAL_BLOBS=1`: the old in-command blob decode (the ABBA referee).
  - `AD_BINDER_DIAGNOSTIC=1`, with `AD_BINDER_CONTINUE`, `AD_BINDER_DUMP_DIR/IDS`, `AD_BINDER_TRACE_ALLOCS` or
    `AD_BINDER_TRACE_WORD`: diagnostic runs, never proofs.

## 10. Size of a window

The size of a window depends on the mesh, the number of steps and what is live. Each `woof hex` command prints the card
and host memory its row needs before it starts and what it used at the end; [README.md](README.md) ("What the
commands use" and "Larger meshes") gives the peaks measured on the shipped 120 km example.
