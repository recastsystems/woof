# The derivative generators

The build-time generators that turn the forward's own captured CUDA source into tangent (TL) and adjoint (AD) code.
No generator runs in a step loop. Their outputs are compiled once, and the window loads them through the L2 bundle
(`AD_SCHEMES`), the kernel set (`--kernels` / `--cubins`) or a TEMPO image (KERNEL-SETS.md says which).

## What they share

- **Input is pinned.** Every generator reads the exact text the forward image was compiled from, as captured by the
  recorder, and refuses on a sha256 mismatch.
- **Type substitution.** `float` / `double` in the physics code become derivative carriers. TL: the value is the
  forward's own float operation (a bitwise primal) and the tangent is in double. AD: the value plus a node on a
  per-thread linear tape, swept in reverse.
- **Exact executed branch.** Every comparison, min/max, threshold, limiter and sign decides on the primal value; the
  derivative is that of the arm the forward took. Loop and iteration counts are integer control and replay exactly.
- **Text checks.** A textual pattern that matches the wrong number of times stops generation, so a changed source is
  never differentiated, or frozen, at the wrong statement.
- **Freezes** are generator options (`--freeze closure` for TEMPO, MYNN-EDMF and the GSL drag); FREEZES.md has what
  each one freezes.
- **Proof harnesses** sit beside each generator, built on libcuda and NVRTC only. Each checks identity (primal =
  forward, every word), the dot test, the zero direction, the repeated reverse bitwise, and TL against centred finite
  differences of the unchanged image.

## The generators

| unit | generator | output | freeze | where the output goes |
|---|---|---|---|---|
| TEMPO `tempo_warp_steps` | `tempo-ad-gen` (Rust, `tree/ad/phys/tempo/gen`) | one .cu: forward text, `adreal*.cuh`, the active copy, `driver*.cuh` | `--freeze none\|closure`, `--freeze-mask` (default `0xc6u`) | `--tempo-cubin` |
| MYNN-EDMF `rrfs_mynnedmf_run_l16` | `tree/ad/phys/mynn/gen_whole.py` | one .cu with `mynn_whole_tl`, `mynn_whole_count`, `mynn_whole_ad` (+ gen-info.json on stdout) | `--freeze none\|closure\|closure+cloud`, default closure; `--weakshear [T]` (with `--freeze none --l25-quot exact`; T in s-2, default 1e-4; off unless given) | bundle images `p3/rel.{0,1,2}.cubin` |
| MYNN-EDMF parameter carrier | `tree/ad/phys/window/rung3/gen_params_p3.py` | derivative .cu with a B1 parameter leaf, forward .cu with the parameter slot | inherits | parameter-gradient bundles |
| MYNN surface layer, GSL drag | `p4-gen` (Rust, `tree/ad/phys/p4/gen`), driven by `tree/ad/phys/p4/build.sh` | `gen-out/{mynnsfc,ugwp}.p4.cu` | ugwp `--freeze=closure` (also `closure+froude`) | bundle images `p4/{mynnsfc,ugwp}.{tl,rev}.sm_120.cubin` |
| surface-layer / drag parameter carrier | `tree/ad/phys/window/rung3/gen_params.py` | the same with parameter slots: Charnock, TOFD a1, xlinv, flow-blocking cd | inherits | parameter-gradient bundles |
| RRTMG-LW gas optics `rlw_taumol_batched` | `tree/ad/phys/p7/gen/gen_taumol.py` | the 16 bands in forward, TL and AD form | none | bundle row (`tree/ad/phys/bind/p7_taumol_row.py`) |
| RRTMG-SW gas optics `rsw_taumol_b` | `tree/ad/phys/p7/gen/gen_rsw_taumol.py` | TL and AD of the 14 active per-layer inputs | none | bundle row (`tree/ad/phys/bind/p7_rsw_taumol_row.py`) |
| general type substitution | `tsgen` (Rust, `tree/ad/phys/typesub`) | the reachable functions re-emitted in TL / tape form | none | style rules `tsad_mynn.cuh` / `tsad_tempo.cuh`, reused by the MYNN and TEMPO generators |
| RUC sea ice | `tree/ad/phys/p6` (`build.sh`, NVRTC) | the whole-launch image | none | bundle row (`tree/ad/phys/bind/p6_row.py`) |

## Commands

The capture root `CAPTURES` is the recorder's output for a configuration (for example hrrrv5 x1.2562).

**TEMPO:**

```
tempo-ad-gen --source tempo_fwd.cu --sha256 <pinned source sha256> \
  --gen-dir tree/ad/phys/tempo/gen --out tempo_adw.cu --manifest gen.json --warp --freeze closure
tempo-ad-proof build --cu tempo_adw.cu --nvrtc libnvrtc.so.13 --out tempo_adw_sm120.cubin
```

`--warp` keeps the forecast's warp shape (one 32-lane warp per column); without it the generator uses the serial
shape. `--promote` is the float64-promoted grading build. The freeze mask is the output's first line; a bisection
build is `--freeze-mask 0x..u` or `-DAD_FRZ_MASK=..`, with no regeneration.

**MYNN-EDMF** (`CAPTURES=... NVRTC=libnvrtc.so.13 tree/ad/phys/mynn/run_p3.sh SCRATCH`, the whole recipe):

```
python3 gen_whole.py <captured forward .cu> $S/gen/mynn_whole.gen.cu > $S/gen/gen-info.json
p3-mynn build --src $S/gen/mynn_whole.gen.cu --include tree/ad/phys/mynn/cuda --nvrtc libnvrtc.so.13 --arch sm_120 --out $S/gen/rel
p3-mynn prove --cubin $S/gen/rel --gen-info $S/gen/gen-info.json --capture ... --tests replay,identity,dot,fd --seeds 2
```

Its derivative rules are `cuda/mw_ad_rules.inc` and its tape is `cuda/mynn_tape.cuh`; `gen-info.json`'s `frozen`
block lists every freeze point.

The exact form with the weak-shear rule (the three images behind `--boundary-layer guarded` on the 3 km storm row):
`gen_whole.py --freeze none --fused m8 --l25-quot exact --weakshear`, then `integrated/compose.sh BASE OUT auto`.
`tree/ad/phys/mynn/weakshear/README.md` says what the rule holds, what was shown and what was not;
`integrated/test_weakshear.sh` is its byte test (without the option the generator writes what it wrote before).

**MYNN surface layer and GSL drag** (`tree/ad/phys/p4/build.sh <capture root> <scratch> <libnvrtc.so.13> sm_120`):

```
p4-gen mynnsfc <captured mynnsfc .cu> <sha256> cuda/p4_ad.cuh cuda/mynnsfc_entries.cuh gen-out/mynnsfc.p4.cu
p4-gen ugwp    <captured ugwp .cu>    <sha256> cuda/p4_ad.cuh cuda/ugwp_entries.cuh    gen-out/ugwp.p4.cu --freeze=closure
python3 gen/p4_frame.py --body gen-out/ugwp.p4.cu
p4-proof build gen-out <scratch>/cubins libnvrtc.so.13 sm_120
p4-proof run <scratch>/cubins sm_120 <capture root> receipt.json
```

The entry ABI, branch policy and inactive rules are in `tree/ad/phys/p4/P4-ENTRIES.json`.

**RRTMG gas optics:** `python3 gen_taumol.py <captured LW .cu> <out.cu>` and
`python3 gen_rsw_taumol.py <captured SW .cu> <out.cu>`; bundle rows from `tree/ad/phys/bind/p7_taumol_row.py` and
`p7_rsw_taumol_row.py`.

**tsgen:**

```
tsgen --source CAP.cu --out GEN.cu --rules tsad.cuh --rules tsad_<style>.cuh --root NAME [--leaf a,b] \
      [--inactive S,T] [--drop-kernels] [--manifest M.json]
```

**Parameter carriers:**
- `gen_params.py <mynnsfc|ugwp> <gen-out/x.p4.cu> <out.cu>`
- `gen_params_p3.py <mynn_whole.gen.cu> <captured forward .cu> <out_deriv.cu> <out_forward.cu>`
- Parameters are module globals, zero by default, so the default build computes exactly the literal, with the same
  float operation on the same value. The adjoint accumulates per column in a fixed order and the host sums the
  columns in index order in float64.

**The L2 bundle:**

```
CAPTURES=... NVRTC=libnvrtc.so.13 NVCC=nvcc P6_TABLE=vilka-table.bin tree/ad/phys/bind/build_bundle.sh IMG OUT plan.json
```

It builds the P3 and P4 images (`build_images.sh`), the P8 and L2 rule kernels and the P6 images, then writes
`schemes.json` from the captures, P3's gen-info, `P4-ENTRIES.json` and P8's rule table, checked against the plan.
`P6_TABLE` is the forward sea-ice module's Vilka saturation table, checked against its pinned sha256.
