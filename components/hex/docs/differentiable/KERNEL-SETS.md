# Specs and kernel sets

What the window loads besides the recorded plan: the ad.entry/v1 specs, the kernel set (`kernels.json` + images),
the L2 scheme bundle, the held manifest, how a set is built, and the sha records that keep a set honest.

## 1. The four inputs and their flags

| input | flag / env | layout | in this tree |
|---|---|---|---|
| kernel set | `--kernels DIR/kernels.json --cubins DIR/cubins` | `kernels.json`, `specs/*.json`, `cubins/<function>.fatbin`, `derived/<sha256>.bin` (content-addressed: a file is used only if its name starts with the sha256 of its bytes), optional `derived.pv/` | kset-int7f: `tree/ad/kset/int7f/`, built by `build_kset_int7f.py --out DIR` (185 rows, 150 derivative images, 153 spec files) |
| L2 scheme bundle | `AD_SCHEMES=DIR` + `--native-schemes 1` | `schemes.json` (schema `ad.scheme.rows/v1`) and its images (P8 rules, L2 rules, P3 tl/ad/count, P4 mynnsfc/ugwp tl/rev, P6, RUC land, P7 LW/SW taumol) | built by `tree/ad/phys/bind/build_bundle.sh IMG OUT PLAN` from a capture root (GENERATORS.md) |
| held manifest | `--schemes FILE` | class per kernel: held / permutation / constant / inactive / rule / live-glue | `tree/ad/phys/window/schemes18.json` (hrrrv5 x1.2562) |
| TEMPO images | `--tempo-cubin`, `AD_DOT_TEMPO_SERIAL_CUBIN` | single cubins | built by `tempo-ad-gen` + `tempo-ad-proof build` |

Fixed allocations are a plan input: they come from `catalog-a.json` beside the plan, not from the kernel set.

## 2. ad.entry/v1: the spec of one derivative entry

A spec is JSON. Its top level describes the derivative:
- `kernel`, `engine`, `pin`, `file` / `line`, `route`, `mode` (`exact`), `affine` with its reason;
- `forward_capture`: source and image sha256, source lines, captured steps;
- `launch`: grid, block, shared bytes;
- `args`: per argument the name, role (in / out / inout / scalar), dtype, shape (an expression in the launch scalars,
  e.g. `[ncol, nl+1]`), stagger, `active` (with an `inactive_reason` when inactive), `overwrite`, `alias_group` and
  `field_id`;
- `saved`, `branches`, `limiters`, `tables`, `refusal`, `precondition`, `precision`, `binder_roles`.

The runner reads `launch_abi`:
- `schema` must be `"ad.entry/v1"`; otherwise the row is not dispatchable ("no ad.entry/v1 launch_abi").
- `entries.tl` and `entries.ad` are required. `entries.aug` is optional and runs on the pre-launch primal copies
  before tl/ad. Each entry has a `function` (a symbol in `cubins/<fn>.fatbin`) and `args`, an ordered binding list.
- Multi-pass orders: `launch_order.{tl,ad}`, `tangent_launch_order` / `adjoint_launch_order` and `reverse_order`.
- `derived_scalars`: scalars computed from expressions in the launch's own scalars (`tree/ad/e2e/src/expr.rs`).

Binding kinds (deriv.rs `prepare_with`):

| bind | meaning |
|---|---|
| `scalar` | a launch scalar by `arg` / `scalar_arg`, or a `value`, `derived_from` or `value_from` |
| `primal` | forward storage; the pre-launch snapshot by default, `post: true` for the live post-launch state; `indirect` (with `mirror: true`) for a mirrored pointer table |
| `tangent` / `adjoint` | the shadow of the argument at the same byte offset; `io` is `in` / `out` / `inout` / `accumulate` |
| `workspace` | a per-launch zeroed buffer with a `bytes` expression (inactive mesh constants use it as a zero source or a discarded sink) |
| `failure` | zeroed failure-flag slots |
| `branch_record` | a zeroed buffer for the executed-branch record |
| `derived` | a content-addressed array: its `sha256` must be in `derived/`, or the binding must carry a recipe |
| `saved` | a named buffer from the bound launch's `saved` set |
| `fill_tangent_pointers` / `fill_adjoint_pointers` | fill-scalar derivative pointers, passed as null (fills are fixed) |

A row with any unsatisfied binding is loaded as **unsupported with a named reason** and its launches get `Refused`;
a row is never skipped silently.

## 3. kernels.json: one row per kernel

Each row names the `function`, its `class` (rowed, glue, physics-radiation, output-diagnostic, inactive-entry,
inactive, copy-or-fill, radiation, linear-scale, physics-glue), its `arguments`, and a `derivative` block with the
spec (`path`, `sha256`), the CUDA source and the nvcc flags. `derivative.spec.sha256` is the **spec record**: the
sha256 of the spec file the image was built against. Some rows also carry a fatbin record. Extra keys are
provenance notes; the runner does not read them.

A fixed numeric spec extent (such as `tab: [104303]`) is never undercut by the row's capture shape.

## 4. Admission: the sha records are checked before anything loads

`check_spec_hashes` (window.rs) runs before the kernel set loads:
- every row whose `derivative.spec` has a `path` and a `sha256` is hashed on all cores, and the file's sha256 is
  compared with the record;
- **any mismatch refuses the run**, naming each row (`f: spec sha256 H != recorded W (path)`);
- rows with no recorded sha256 are counted as `specs_unrecorded`, not refused;
- the result is in the receipt as `spec_sha256` (`{"specs_checked":153,"specs_unrecorded":0}` for kset-int7f).

It prevents a row whose spec changed after the set recorded it from being bound with a launch ABI its image was not
built from.

## 5. Rebuilding kset-int7f

`python3 tree/ad/kset/int7f/build_kset_int7f.py --out DIR [--toolchain 13.2.86] [--only f1,f2] [--jobs N]`:
- `recipe.json`: per image, the toolchain that built it, the working directory, the source, the exact flags, the
  image sha256 and the sha256 of every source and header it reads. A source that differs from its record refuses the
  build.
- `toolchains.json`: the NVIDIA PyPI wheels (nvcc, nvvm, crt, runtime; cccl for 13.2.86) pinned by sha256, installed
  into a cache (`--tc`, default `~/.cache/ad-kset-tc`).
- `Containerfile.ubuntu-24.04`: the host for the rows built with CUDA 13.0 (whose headers do not parse against glibc
  2.41 or later), run with podman or docker.
- `src/`: the vendored sources and headers of rows that have no generator directory in the tree.
- The set is assembled in DIR (cubins/, specs/ with the spec paths rewritten into DIR, derived/, derived.pv/,
  kernels.json), every spec record is checked, and `DIR/REBUILD.json` is the receipt.

Result from a fresh checkout: 149 of 150 images byte-equal and 153 spec records checked. The one image that is not
byte-equal, rlw_rtrn_march_coalesced, is not byte-reproducible by nvcc itself (13.5.37 output changes run to run with
the address-space layout); its rebuilt images give the same tangent and adjoint words on every captured launch.

## 6. Extending a set

- A hand row (`tree/ad/phys/held/<function>/`) has the derivative kernels in `cuda/<fn>.ad.cu` or `<fn>.ad.cu`,
  which include the forward's own helpers so the primal stays bitwise, an ad.entry/v1 spec and a `kset-row.json`.
- A row with an `aug` entry must either tape what it overwrites or recompute every primal it reads: an `aug` that
  rewrites an inout primal that tl/ad then read untaped corrupts both by the same amount, so every dot still passes.
- Fixed mesh constants (`inv_area_cell`, `rdzw`, `acoustic_sign`, `mesh_density`, and the surface spacing's
  `density` and `length`) are inactive: in every spec that binds them with a derivative the tangent reads a zeroed
  workspace and the adjoint accumulates into a discarded sink. A constant bound as a written tangent, or as a
  non-accumulate adjoint, is refused.
- The L2 bundle rows carry `p8_class`; schemes.rs refuses the whole table if a row disagrees with P8's class.
