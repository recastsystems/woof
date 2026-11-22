CFE source at `f9182dfb3d81c407c66c5b782114173893c30cf2`: “Apache License”; “Version 2.0, January 2004”, Apache-2.0.

This lane owns CFE fixture ingestion, replay orchestration and module tests for
the NextGen port. The native reference/captures belong to the referee; numerical
CUDA implementation belongs to the CFE port owner. Shared raw I/O, hashing and
word comparison belong to `tools/nextgen_replay` (overland-replay owner).
No withdrawn WRF-Hydro code, fixture or ABI is used here.

Select the capture by its sealed index, not by a notification calling it fresh:

| Index | Boundary and initialization | Existing result |
|---|---|---|
| `fixture-index-ngen-shipped.json`, SHA256 `f38581bf7552b43f44e8cce57d055e8b54a9b50460b7240654a5a735aff1ba22` | Original 2026-10-17 23:48 SLoTH/PET/CFE capture, 1157 files; initialization audit pending | Diagnostic only; no numerical verdict |
| `fixture-index-ngen-shipped-defined.json`, SHA256 `487fb68c217be43bca1570e3d4a7a96bcde3ae87d42f0b30289b7cbf7e2d4d66` | Explicit native allocation patch; step39 core entry/exit after rainfall conversion | Fast and strict **0/411** at numerical candidate `0a3e47ba610d2b7858c9e9857437437e5b1b84f7` |
| `fixture-index-ngen-shipped-noah-coupled.json`, SHA256 `777a8b86b45fdef1b568a13276b37a6780860454cbbfccdfaad221f3fbc19f63` | Separate defined Noah-fed BMI capture; step35 entry/exit includes clock, conversion and ponded depth | Prepared-factory default-fast **0/426** at executed revision `9253f006cf2551acbf4c046d5a71b37c59d40009` |

The later captures do not validate undefined members in the original capture.
The packer rejects a missing or explicitly pending initialized-state audit before
reading capture payloads, in both replay and history modes. This enforces the
referee's recorded pending status; it does not independently certify the audit
or turn a payload hash into proof of initialization.
The defined-state allocation patch and configured storage fractions remain
disclosed native-tool adaptations; none of these results means untouched
upstream identity, active-option coverage or whole-trajectory acceptance.

The delivery's `code/capture-command-index-current.json` indexes this lane's
retained core GPU argv, environment, host/cwd, packing and shared Rust comparison
commands, with request/verdict seals. Its original staging/build directories
were cleared after receipts. Reconstruct the pinned sources and sealed inputs
in your own scratch before a new run; archived argv is execution provenance,
not permission to write into a peer's scratch or substitute current sources.
The separate BMI proof remains in `code/first-boundary-35/`.

For the speed owner, use the existing fresh-instance loader; no second packer
or replay harness is needed. Select fast numerics before importing CFE:

```python
import os
os.environ["GPUWM_NUMERICS"] = "fast"  # fresh process, before module import
from woof.hydro.cfe import load_prepared
component, manifest, rust_library = load_prepared(
    CORE_MANIFEST, RAW_LIBRARY, payload_root=CORE_INPUT, strict=False)
# The retained native fixture has period_seconds == 3600.
launch = component.launch  # launch(3600.0, stream=stream), enqueue only
```

`CORE_MANIFEST` is the retained `replay-fixed/manifest.json`, and `CORE_INPUT`
is its `input/` directory in this lane's delivered evidence. This is core
entry/exit replay after rainfall conversion; its fast and strict verdicts are
0/411 at the recorded numerical revision. A new loader call allocates separate
resident state; do not treat repeated advancement of this captured forcing as
a newly validated trajectory. All setup, uploads and compilation precede timing.

For the full BMI boundary, reuse `run_bmi.py --factory` with the retained
`first-boundary-35/fixture/manifest.json` and its `input/` directory. Exact
original argv, catchment order, time origin and native witness seals are in
`first-boundary-35/actual/run.json` and `receipt.json`; the existing default-fast
result is 0/426. The delivery's `code/speed-state-handoff-current.json` publishes
absolute source/state paths, loader arguments, numerics selection and both
verdict seals. Speed owns timing and `bind_driver`; this lane supplies no
second speed executor. The fixture's one-hour CFE period does not resolve
case(b)'s routing cadence or supply its physical catchment parameters.

`audit_source.py` inspects source metadata from the exact Git objects in an
existing CFE checkout. It does not build/run the native model and is not a
numerical replay. It emits the native struct leaf inventory, advertised legacy
serialization table, actual pointer assignments and dynamic-size assignments.
Reading an exact Git object avoids accidentally auditing a locally instrumented
working file instead of the source pin.

```
PYTHONDONTWRITEBYTECODE=1 python3 tools/cfe_replay/audit_source.py \
  /path/to/existing/cfe-checkout \
  /path/to/source-audit.json
```

Add `--layout /path/to/port-owner/field-schema.json` to check that the published
device field/option paths resolve to native types and report explicit setup
conversions. This validates metadata; it does not read numerical state.

The output must be a new file. JSON/text inspection is metadata work only;
Python never decodes or transforms hydrologic arrays. See
[CAPTURE-REQUIREMENTS.md](CAPTURE-REQUIREMENTS.md) for concrete capture hazards
and the boundary discussion with the owners. The emitted inventory includes
unused and potentially uninitialized members; it is **not** permission to read
all member bytes or compare unused native memory.

`cfe-capture-pack` ingests the referee's sealed `nextgen.native.capture/v1`
JSONL descriptors and raw little-endian snapshots. It checks each selected
descriptor/payload hash with the shared Rust library, preserves floating words,
promotes native integer `time_step_size` exactly, and packs the published
97 f64 / 18 i32 / five-offset layout. The eighteenth option is pointer presence:
GIUH NULL is witnessed by pinned `src/bmi_cfe.c:1123-1124`; active surface Nash
requires its complete positive-length captured storage. Missing inactive soil
depth vectors receive explicitly reported binding slots. Native captured words
are never masked. Queue length is exactly N+1.

Build outside the worktree, using the vendored crates:

```sh
CARGO_HOME="$SCRATCH/cargo-home" CARGO_TARGET_DIR="$SCRATCH/target" TMPDIR="$SCRATCH/tmp" \
  cargo build --release --offline --locked --manifest-path tools/cfe_replay/Cargo.toml
RAYON_NUM_THREADS=12 "$SCRATCH/target/release/cfe-capture-pack" \
  "$INDEX" "$LAYOUT" 39 "$SCRATCH/replay" "$CANDIDATE_REVISION"
```

The optional last argument relocates the capture root without changing any
original index seal. A delivery needs only the selected model-0/1/2 step-39
core-entry/core-exit `.jsonl` and `.bin` files. `wet` instead selects the step
with maximum summed captured rainfall from the shipped 3-catchment/48-step
case; this is case selection, not a numerical parameter sweep. Outputs are new
files only. The manifest records selection, native pin, layout, period, scope
and the referee's complete recipe/audit. Original undefined-state frames remain
diagnostic; the defined-state reference explicitly records its native patch.

`history:48` extracts six native BMI-exit scalar histories for the driver's
single continuous case: runoff depth, soil storage, groundwater storage,
rainfall depth, potential ET and actual ET. Each file is f64 `[time,catchment]`
in C order. Catchment order comes from sealed config paths. Entry/exit clocks,
3600-second periods and unit time-step fractions are checked for every frame;
the separate i64 clock file contains post-update seconds relative to the native
initial clock. No UTC origin is inferred. `manifest.json` retains the original
index seal and recipe, names, physical units and payload seals. This is native
expected data only: it defines no resident output ABI and claims no GPU result.
Run it with the corrected-fraction Noah-coupled index, not the earlier malformed
percent-configuration trajectory:

```sh
RAYON_NUM_THREADS=12 "$SCRATCH/target/release/cfe-capture-pack" \
  "$INDEX" "$LAYOUT" history:48 "$SCRATCH/native-history" "$CANDIDATE_REVISION"
```

The positional layout is retained for CLI compatibility and is unused by this
native-only extraction. Supply the extractor source revision in the revision
argument; the manifest records it with the actual argv and executable seal.
The history manifest is not itself a
comparison request; bind the actual driver's command, revision, numerics and
exported fields through the shared comparator when its run exists. CFE runoff
stays in metres per step; no catchment area or discharge is invented.

The initialized 48-hour case(a) chain at driver revision
`11c6bb0fc66b89107c2507d39eab5ba5073c02ca` now has a strict continuous
comparison: five f64 `[48,3]` histories (runoff, soil storage, groundwater
storage, rainfall and PET rate), **0/720 differing words**, through the shared
Rust comparator. The independently packed CFE histories use primary index
`777a8b86b45fdef1b568a13276b37a6780860454cbbfccdfaad221f3fbc19f63`;
framework index `5c5b23e6050e304fbfd98cb0096924e31ac030e99230ded3b83105cfa5547461`
explicitly seals that primary index. All five expected payload hashes also
equal the framework getter histories. PET rate binds to the PET module's
`water_potential_evaporation_flux` in m/s; CFE's `POTENTIAL_ET` BMI getter
is depth per step and cannot substitute for it.

The receipt is delivered in `code/continuous-casea/`: exact commands, request,
verdict, field bindings, driver archive and five inspected Rust-rendered plots.
This reuses the sole driver run. Shared RTX 4090 timing was 0.263366 ms per
hourly step for three catchments; its CFE stage was 0.013600 ms including
enqueue gaps, one catchment per thread in a 128-thread block. These measurements
do not establish the LowerColorado whole-network/terrain speed gate or USGS
skill. Focused default-fast and strict module replay evidence remains separate
from this continuous strict trajectory.

The initialized standalone restart at driver revision
`43f8cea2e84ce3e3eb28bcacce0e807fa22f43b2` now covers a 48-hour strict run
versus 24 hours plus a fresh-process restore and another 24 hours. This lane
independently compared the resumed five CFE/PET histories against its sealed
native extraction: **0/720 differing words**, no tolerance or exclusions,
5.095093 ms with 12 Rust workers (14.156861 ms comparator process).
`code/resumed-casea-01/` retains the actual request, verdict, native provenance,
commands, inspection record and five inspected Rust plots. Runoff overlaps
native without a jump at the 24-hour boundary; PET and Noah pulses align.
The comparison command is:

```sh
RAYON_NUM_THREADS=12 nextgen-replay \
  "$SCRATCH/resumed-casea-01/request.json" "$SCRATCH/resumed-casea-01/verdict.json"
```

The driver reports equality for all 466 declared carriers (465 nonempty,
232708 logical bytes), the complete 427024-byte restart and the 63268-byte
history. This lane verified the retained resumed container seals. The owner
cleared its continuous baseline after recording the comparison, so whole-state
equality is attributed to that contemporaneous owner receipt; this lane's
independent numerical comparison covers the five histories above.

The owner measured 0.189708957 ms per step on a shared RTX 4090, 47 strict
steps after one warmup, with two batch-boundary waits and no production events
or graphs. Its separate diagnostic CFE stage was 0.013312 ms for three
catchments in one 128-thread block, including enqueue gaps. This closes the
standalone initialized restart blocker for this tested composition. Actual
parent/atmosphere, network, terrain and event continuation, USGS skill and the
whole-hydro speed gate remain open. Fast remains the default; no fast
whole-chain parity is claimed. No duplicate GPU run or serializer was added.

The later PET-v2 default-fast chain at driver
`15b52b6157be07baa3e2db594c9415d4de91997c` differs from native in **83/720**
CFE/PET history words: runoff, soil and groundwater each 26, rainfall 5,
PET 0. The first observed difference is at flat word 105 (hour 36, cat-67),
also the first native/fast Noah QINSUR difference. The owner's six-field
comparison includes five QINSUR differences, totaling 88/864. These are
measured differences, with no last-ulp or math-library exemption.

A focused Rust conversion diagnostic applied the published f32 m/s binding
(`f64(rate) * 3600000.0000000005 / 1000`, fraction 1) independently to actual
and native QINSUR. Shared exact comparison against their respective captured
CFE rainfall histories gives **0/288 differing words**. The observed rainfall
difference is therefore supplied by upstream QINSUR, rather than a mismatch
in this conversion. This does not prove reservoir arithmetic on changed
inputs. Requests, verdicts, opaque inputs, converter source and commands are
retained in `code/fast-input-trace-01/`. The five current Rust plots were
inspected and visually overlap; the word differences remain declared.

The native comparison took 3.240406 ms and the conversion comparison
0.668331 ms, each with 12 Rust workers. No new GPU trajectory was run.
The driver's shared RTX 4090 fast batch measured 0.172824618 ms per hourly
step, 48 steps with first enqueue included; its separate cold CFE diagnostic
was 0.013216 ms for three catchments in one 128-thread block. These remain
owner timings, with no whole-network/terrain speed claim. The earlier strict
restart proof above remains at its PET-v1 pin.

The current PET-v2 default-fast standalone continuation at the same driver pin
now has a separate owner proof: 24 hours plus fresh-process restore for another
24 hours equals its continuous 48-hour run for all 466 declared carriers and
the complete restart/history containers. This lane independently compared the
resumed five histories with the retained continuous histories: **0/720 differing
words**, 1.143722 ms with 12 Rust workers (2.341888 ms process). Comparison with
the sealed native extraction still gives **83/720 differences**, 1.999244 ms
with 12 workers (3.228031 ms process). Restart equality does not retire the
native differences. `code/current-v2-resumed-01/` retains exact requests,
verdicts, owner commands and scope. Full-state equality is attributed to the
owner's contemporaneous proof; this lane verified the canonical final container
seals without copying the deleted resumed duplicates. The five previously
inspected continuous-run plots describe these identical resumed histories;
no new rendering or image inspection is claimed.

One additional CFE module replay isolates the first differing interval, native
step 35 (hour 36). `bmi-noah:35` preserves captured native CFE entry state and
Noah QINSUR inputs, with the sealed framework conversion witness. The current
CFE Python/CUDA sources match driver `15b52b6157be07baa3e2db594c9415d4de91997c`.
This is a CFE implementation identity, not whole-package byte identity. The
executed replay uses its own sealed `woof/numerics.py`, which lacks the driver's
unused `select()` helper; the remaining source AST is identical. Neither CFE
nor this runner calls that helper. The raw CUDA bridge is replay-owned. Actual
execution sources remain pinned to `9253f006cf2551acbf4c046d5a71b37c59d40009`
and the recorded staging seals, with the source-scope audit retained beside
the replay receipt.
Default-fast through the published factory gives **0/426 differing words**,
including status zero, with no masks or tolerance. The first launch took
0.033984 ms on a pooled RTX 5070 Ti, three catchments in one 128-thread block;
shared Rust comparison took 3.410052 ms with 12 workers (6.730125 ms process).
`code/first-boundary-35/` retains the fixture, exact command, source seals,
lease and verdict. This clears CFE on the captured native state and inputs at
that interval. It does not establish equality of later CFE state after receiving
different upstream runoff, identify the internal Noah operation, or prove the
whole-network/terrain speed gate. No second continuous GPU run was added.

The Noah replay owner's subsequent native-entry step36 fast diagnostic gives
171/1767 differing words, status zero, on a pooled RTX 5070 Ti. Its stored-word
trace reproduces each side's dew and QINSUR from that side's energy flux;
EVG/EVB/FGEV differ while latent heat matches. This is upstream-owner evidence,
not a new CFE replay or a cause established for the earlier continuous RTX4090
run. The precise energy-flux arithmetic remains unresolved. Source pins, commands,
timings and scope are retained in `code/upstream-noah36-review/`.

On a GPU host, use its pool mutex and set TMPDIR, CUPY_CACHE_DIR,
CUDA_CACHE_PATH and XDG_CACHE_HOME under the lane's data-disk scratch. In a
fresh process select `GPUWM_NUMERICS=strict` for the native word comparison
(`fast` remains the product default):

```sh
python3 tools/cfe_replay/run_gpu.py "$SCRATCH/replay/manifest.json" \
  "$RAW_LIBRARY" "$SCRATCH/gpu-result"
python3 tools/cfe_replay/make_request.py "$SCRATCH/replay/compare.json" \
  "$SCRATCH/gpu-result" "$SCRATCH/request.json"
RAYON_NUM_THREADS=12 nextgen-replay "$SCRATCH/request.json" "$SCRATCH/verdict.json"
```

`RAW_LIBRARY` is the shared `libnextgen_replay.so`. `--payload-root` remaps
input basenames when staging on another host; hashes remain binding. After
downloading a run, `make_request.py` binds its recorded command/numerics and
local actual paths to the comparison template. It handles JSON metadata only.
`run_gpu.prepare(manifest, library, payload_root=None, strict=None)` returns
`(CFEGPU, manifest, library)` for the speed owner; select process numerics before
calling. Rust loads sealed bytes into CuPy pinned host buffers, CUDA copies and
advances state, and Rust writes the downloaded bytes. No Python array values
are decoded, packed, hashed or compared.

The runtime packet loader now lives in `woof.hydro.cfe.load_prepared`, owned
by the CFE port lane; `run_gpu.prepare` delegates to it. Its packet boundary is
still the core state. The shared C ABI takes caller-owned host allocations;
CuPy supplies and releases pinned memory. No private allocator or bit comparer
is implemented by this module.

For a focused native BMI entry/exit replay, select `bmi:39` instead of `39`:

```sh
RAYON_NUM_THREADS=12 "$SCRATCH/target/release/cfe-capture-pack" \
  "$INDEX" "$LAYOUT" bmi:39 "$SCRATCH/bmi" "$CANDIDATE_REVISION"
python3 tools/cfe_replay/run_bmi.py "$SCRATCH/bmi/manifest.json" \
  "$RAW_LIBRARY" "$SHARED_RAW_CUDA_PY" "$SCRATCH/bmi-result"
python3 tools/cfe_replay/make_request.py "$SCRATCH/bmi/compare.json" \
  "$SCRATCH/bmi-result" "$SCRATCH/bmi-request.json"
RAYON_NUM_THREADS=12 nextgen-replay "$SCRATCH/bmi-request.json" "$SCRATCH/bmi-verdict.json"
```

`SHARED_RAW_CUDA_PY` is the shared owner's unchanged
`tools/nextgen_replay/raw_cuda.py` (introduced at `7c69b0b689097b813718b80c257ae65c91e70c9e`).
It remains owned by overland-replay. This runner imports it explicitly, records
its Rust-computed source hash, and uses `RawCuda` for all byte staging/export.
`--payload-root` relocates input basenames while preserving seals. Rust packs
the captured native BMI rainfall amount divided by native period into canonical
mm/s; GPU multiplication by period then follows native `/1000` and fraction
order. The native BMI label `mm h-1` is not silently treated as mm/s.
Common captured clock/fraction values configure the published constructor;
CUDA restores their exact captured bytes and previous ponded depth at setup.
The same single launch advances core, clock and ordered ponded sum.

`--factory --initial-manifest PATH --initial-payload-root DIR` instead loads
the production Rust parameter packet (`cfe.initial.pack/v1`). This is restricted
to BMI step zero. Captured rain/PET are bound inputs; captured core state,
clock, fraction and ponded depth are not restored over the initializer. The
ordinary factory creates those values. The receipt seals the initial manifest.
This proves one initialized update, rather than an entire continuous trajectory.

`bmi-noah:39 --framework-index INDEX` selects the actual captured upstream
Noah QINSUR f32 m/s words. The framework index must name the same primary
private-state index hash and its native exchange audit. Ordered catchment IDs
come from sealed native `config-path.json`; public Noah/CFE descriptors and
payloads are verified with shared Rust I/O. Public precipitation/PET inputs must
equal the private CFE entry words before combining these boundaries. No inverse
conversion from CFE rain is used. Both indices are retained as comparison
provenance. The named native UDUNITS coefficient is `3600000.0000000005`,
whose rounding differs from literal `3600000` on all 15 nonzero native links.
The same published factory/runner then tests that conversion on the GPU. Its
upstream vectors are captured native values; live GPU Noah/PET composition
remains a separate driver proof.

Focused execution at `93180ba7b682136aca5918e0a2dde7fb8f7f9726` (runtime
`e7a7a7b95a843d9b9684bef24a43f72436a62d17`): production parameter packet
`prepared-config-02` to native BMI0 is **default fast 0/426 different words**.
No captured core state or carrier values are restored over that initializer.
Actual captured Noah m/s wet39 to CFE is **strict 0/426 and fast 0/426**.
Framework index SHA256 is
`5c5b23e6050e304fbfd98cb0096924e31ac030e99230ded3b83105cfa5547461`.
The shared comparator returns exit0 without masks or tolerances. RTX5070Ti
pool, three catchments/one 128-thread block: first launches 0.040960ms
(initialized fast), 0.034112ms (wet strict), 0.033760ms (wet fast).
These timings cover tiny first launches, not a full 4090 routing step.
The subsequent optional canonical mm/s compatibility extension `28119241426`
was merged after this proof; that distinct input option has no GPU witness here.

For an initialization proof, pass `--initial-manifest LOCAL_MANIFEST` also to
`make_request.py`. It seals the actual initial manifest hash recorded by the
GPU run in shared comparison provenance, and preserves an unset numerics
environment as `env -u GPUWM_NUMERICS` in the reproducible command.

Measured defined-state step39, candidate CUDA `f91e0a0d22c029432b93a099f369eb68fc9b3217`
with runtime loader source `5f2b7fc70141c175aaa30ddfebad0f1220f5eb12`:
**strict 0/426 and fast 0/426 different words**, no tolerance or mask.
These extend the 411 core/check words with 15 explicit resident BMI carriers.
First launches: strict 0.029408ms, fast 0.031392ms, RTX5070Ti,
one 128-thread block/three catchments, no graphs. The specialized runner binds
the constructor directly; it does not prove the production composition factory.

The original example's percentage storage strings are parsed as raw values by
the pinned native source. Existing step0 and step39 packets are valid module
replay inputs, but their trajectory is not an acceptable physical short run.
Referee has published `fixture-index-ngen-shipped-noah-coupled.json`, SHA256
`777a8b86b45fdef1b568a13276b37a6780860454cbbfccdfaad221f3fbc19f63`.
Its explicit configuration adaptations use soil `0.667[m/m]`, groundwater
`0.5[m/m]`, and map native Noah QINSUR through ngen UDUNITS into CFE's mm/h.
Native clean/observed outputs are byte-identical across six CSVs. This remains
an adapted defined-state reference, with all changes retained in the recipe.

Select `bmi-mm-h:39` to preserve captured CFE precipitation bytes directly;
its resident array is named `precipitation_mm_h`. With that packet, invoke:

```sh
python3 tools/cfe_replay/run_bmi.py "$SCRATCH/fraction/manifest.json" \
  "$RAW_LIBRARY" "$SHARED_RAW_CUDA_PY" "$SCRATCH/fraction-result" \
  --factory --catchment-ids cat-67,cat-27,cat-52 --time-origin 2015-12-01T00:00:00Z
```

The roster above is audited from this case's captured `config-path.json`,
never a generic default. `--factory` invokes the published `cfe.prepare` with
row-selected precipitation component/field/units and PET component. Its source
components bind captured native arrays; they do not advance upstream physics.
It writes only a JSON core-buffer descriptor for the existing runtime loader;
all payload reads still use the shared Rust bridge. Exact native clock,
fraction and previous ponded bytes are restored at setup through CUDA.

Corrected fraction fixture step39, port candidate
`aeb27e450d99d64b0427ecd80f8ff8238ca238ab`: strict and fast both **0/426**,
no tolerance/mask, through the actual factory. First launches 0.034432ms strict,
0.033664ms fast, RTX5070Ti, one 128-thread block/three catchments. This proves
the factory and native mm/h boundary on this captured state. It does not prove
the m/s unit option, live Noah/PET GPU chain, continuous state, routing or a
physical short-run picture. Corrected step0 core state is packed separately
for the port/forcing/driver owners' single continuous run. This lane does not
produce a competing driver or relabel an original malformed trajectory as a
successful physical picture.

For named scalar diagnostics, `split_fields MANIFEST REQUEST NEW_OUTPUT`
extracts packed raw rows in Rust and writes `NEW_OUTPUT/compare.json` for the
same shared comparator. It implements no numerical comparison. Whole-buffer
and named-row verdicts must agree on the total differing words.
The optional last comma-separated field list emits individual catchment words
for those named fields, explicitly labeled a selected-word diagnostic. It
supplements the full verdict and cannot replace it.

Current measured fixture: defined native ngen-shipped core step39, three
catchments, candidate CFE `0a3e47ba610d2b7858c9e9857437437e5b1b84f7`,
RTX5070Ti. Strict and fast both match **411/411 words exactly**, with no mask or
tolerance. First-call device event times were 0.038080ms strict and 0.039456ms
fast. These are neither steady-state benchmarks nor the whole-step 4090 speed
gate. See the lane REPORT and sealed verdicts.

The retained predecessor `3cd3702222a8681a1c538f7ae1ac4b8e8572cc97` differed
in six words, only groundwater/total discharge. Its isolated witness identifies
glibc2.43 exp versus NVRTC13.4 device exp on the same argument as the first
one-ULP difference. The forward port fix uses WOOF-ArWen's existing MIT Arm math
helper at that calculation, default-on in both modes. The same native fixture
then passed; the original failing verdicts remain evidence of the exact defect.
The referee's initialization audit subsequently confirmed undefined native
members in these original captures. They remain diagnostic only. Its separately
sealed, explicitly patched native tool defining initial state was then replayed:
same step39, same6/411 differences, 0.033120ms first launch. That reference uses
calloc for native state/profile allocation and retains identical six output
CSVs; untouched-upstream identity is not claimed. Its index has a `cfe/` module
namespace, handled explicitly by the packer. The manifest carries the full
native recipe, including audit, patches and observation-neutrality evidence.
An actual short coupled run/picture and whole-step speed receipt remain
pending; this module replay result does not accept the complete unit.
