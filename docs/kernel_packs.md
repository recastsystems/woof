# Compiler-family kernel packs

The normal kernel-plan startup also installs the prebuilt-image loader. It
selects `woof/data/kernel-pack-cuda12.json` or `kernel-pack-cuda13.json` from
the loaded NVRTC major version, with `kernel-pack.json` as the legacy fallback.
`WOOF_AOT_PACK` selects an explicit pack. The filename alone never establishes
compatibility: the loader checks the compiler packages, NVRTC library digest,
source, numerical options, target architecture, request metadata and image
digest. A miss uses the ordinary compiler.

The codec is the same `gpuwm.kernel-pack/v1` contract used by hex. Both CuPy's
cached module route, WOOF's direct NVRTC route and the final NVRTC funnel use it. Template symbol
aliases are restored on load. Strict and fast recipes remain separate, even
when they are built in one pool whose environment defaults to fast.
`GPUWM_WRF_EXACT=1` also normalizes its arithmetic controls before direct-cache,
warm-plan and pack lookup. A fast image cannot satisfy that request. Packs or
catalogs recorded without those controls need separate exact-mode entries;
otherwise the loader falls back to compilation.

Final-funnel recipes use `raw-nvrtc`. Their options already contain CuPy's
additions: the builder preserves them and the exact program filename rather
than appending flags or recomputing a cache name. Ordered template names and
their lowered mappings are preserved. PTX, linking and memory-only requests
retain the ordinary compiler path. A retained plan from a different compiler
is an input seed; it must be reconstructed or captured for the target identity
and target filenames before it becomes that compiler's production catalog.

## Build and stage

Record the actual selected forecast or `warm-kernels` requests in each pinned
compiler environment. Use a data-disk scratch directory for every output,
including `TMPDIR`, `CUPY_CACHE_DIR` and `WOOF_KERNEL_PLAN_DIR`. A `{pid}`
filename template gives parallel workers separate catalogs. For example,
with `$scratch` and a valid `$PROFILE` already selected:

```sh
WOOF_AOT_RECORD="$scratch/strict-{pid}.json" GPUWM_NUMERICS=strict \
  python -m woof warm-kernels --profile "$PROFILE" --json
WOOF_AOT_RECORD="$scratch/fast-{pid}.json" GPUWM_NUMERICS=fast \
  python -m woof warm-kernels --profile "$PROFILE" --json
python tools/build_kernel_aot.py \
  --recipes "$scratch"/strict-*.json "$scratch"/fast-*.json \
  --output "$scratch/kernel-pack-cuda12.json"
```

Repeat with the CUDA 13 environment and a separate scratch directory to build
`kernel-pack-cuda13.json`. Catalogs are compiler-bound; a CUDA 13 catalog is
not a CUDA 12 build recipe. The CPU-only builder uses spawned NVRTC processes,
defaults to all available CPUs, and builds sm_89/90/100/120. `--workers` caps
build concurrency; `--base-pack` reuses matching verified images. Stage the
two completed family packs beside each other in package data for the release
build. Broad generated packs and catalogs stay outside the source checkout.
The Thompson startup packs are versioned package data so a fresh source
or wheel install also has the large first-use units. Rebuild them with
`tools/build_thompson_startup_pack.py` in each compiler family; it captures the
production loaders in strict and fast modes and uses the same parallel builder
for sm_89/90/100/120. Pass real first-step catalogs with `--recipes` to include
their small startup requests. The shipped entries bind complete include trees
with the retained-image cache's existing verifier, including conditional CCCL
headers. Compiler, source, option and header mismatches retain normal fallback.

## Warm plan and receipts

Supported cubin requests, including first-start static captures, compile in
CPU-only processes. GPU module loads stay in the parent thread pool. Every
process compiles the parent's final options and exact program name. The pool
honors `WOOF_KERNEL_PLAN_WORKERS`. Rebinding a configuration drains its old
jobs before replacing the state used by their callbacks.

Core CUDA handle factories install the same prebuilt-image loader, so direct
library callers use shipped images without a forecast supervisor or a process
pool. The forecast's parallel pool stays open for requests discovered during
execution; interpreter exit stops its workers.
Recorded requests register the process cancellation proxy directly in the
single-flight table, leaving GPU warm-up threads available while CPU compilers
run. A foreground request cancels a queued job only when the process job itself
can be cancelled; otherwise it joins that job's image and symbol mapping.

Pack-covered recorded requests and static captures are skipped before scheduling. This is only a
scheduling hint: the actual loader still verifies the complete pack contract.
`kernel_plan.receipt()` reports `pool_skipped_aot`, `static_skipped_aot`, worker CPU seconds and
per-process work; `kernel_aot.receipt()` reports image hits and JIT misses.
These records describe startup and add no work to a warm simulation step.

Build coverage is the recorded request inventory. Before release, verify the
intended configuration loads its images without JIT misses, replay its strict
outputs, and capture a separate device timeline to check step launch counts.

## Automatic warm images and compiler metadata

The ordinary compile plan retains successful cubins under
`WOOF_KERNEL_CACHE/images/<binding digest>/index.json`. The binding includes
the full loaded NVRTC library digest, CuPy package/cache identity and target
SM. Each request includes the source digest, compiler program name, ordered
final options, header contents and symbol aliases; run tags do not partition images. Blobs and index rows carry digests. First use
uses the existing parallel compiler; background image writes retain its
actual results. Missing or corrupt entries warn and use that same compiler.
Concurrent publishers merge index entries under a process lock.

An unknown shape reads only its own plan filename. It does not enumerate
other shapes' plans. Grid dimensions and floating configuration values are
part of the tag. Composed Noah-MP and legacy radiation units use their
runtime factories during static warming.

CuPy's empty-source NVRTC preprocessing and the device-probe compiler banner
also have persistent metadata. These records contain measured output, not
inferred version strings. Exact options/target/compiler bytes bind empty
preprocessing; loaded compiler bytes bind the banner. The bare subprocess
loads stdlib-only helpers by path. If a standalone preparation package lacks
those helpers, its original measurement path remains available.

These caches are enabled with the existing compile-plan defaults. A cache
hit changes neither a kernel's source nor its arithmetic flags. Component
zero-NVRTC checks do not establish full CLI startup or forecast-word parity;
those require the corresponding forecast receipts.

A shape first encountered with an existing CuPy disk cache can have a partial
automatic image index: requests served before the NVRTC funnel produce no new
compiler result to retain. Removing that CuPy cache can therefore trigger
compilation of those missing requests. Their successful results populate the
index. The missing-image fallback must not be counted as a zero-NVRTC warm
run. Both caches contribute to ordinary warm startup.

External-header invalidation: automatic images now bind complete explicit
include trees by relative filename and file digest, covering macro-selected
and conditional includes. Forced includes and dependencies outside those
trees still warn and use ordinary compilation. The older shipped AOT format
has no external-header binding and declines such requests. Historical AC
measurements above predate this persistent-image change; they do not establish
the new format's complete forecast coverage.

## Persistent images and box provisioning

`WOOF_KERNEL_CACHE` selects the stable cache root (default
`$XDG_CACHE_HOME/woof/kernels`, or `~/.cache/woof/kernels`). Images are shared
across run tags, cycles and shapes. A request retains its source digest,
ordered final compiler options/defines, symbol names, compiler identity and SM.
Dimensions embedded in the source or compiler defines still distinguish images.
The NVRTC library digest and version distinguish compiler builds; cubins need
no driver-version key because no driver JIT produced them. PTX/linking routes
keep their existing compiler path.

CuPy's generated program filename originally included installation paths through
its cache hash. The compile funnel now gives NVRTC a source-derived basename,
so the actual compiler input agrees across installations. Explicit program names
are preserved. Include roots are represented by ordered content snapshots, with
relative filenames, including extensionless C++ headers. This covers guarded
inlined includes, conditional discovery and CCCL/Thrust macro dispatch. Header
roots are immutable during a compiler session, as are its loaded libraries;
restart a process after changing an installed toolkit/header tree. External
computed dependencies that escape those roots fall back with a warning.

To prepare a box kit, warm the intended configuration on each compiler/SM pair,
then export its actual images:

```sh
woof kernels pack /work/kit/kernels-sm120.tar.gz --arch sm_120
# On an Ada box with the intended compiler:
woof kernels pack /work/kit/kernels-sm89.tar.gz --arch sm_89
```

Copy each archive **and its `.sha256` file** with the private box kit. During
bring-up, after installing the pinned compiler environment, run:

```sh
woof kernels seed /work/kit/kernels-sm120.tar.gz
# Or pass a separately retained expected hash:
woof kernels seed /work/kit/kernels-sm89.tar.gz --sha256 EXPECTED_SHA256
```

Pack/seed need no GPU. A pack may contain multiple SMs; `--arch` is repeatable
and omitting it exports every recorded SM. Seeding verifies the archive,
manifest, binding, row and image digests before publishing any entry, then merges
with existing local images under the publisher's process lock. A compiler/SM
mismatch cannot satisfy a runtime request. Damaged runtime images warn and
recompile; damaged packs are refused explicitly, leaving ordinary compilation
available. Compiler metadata is packed too, avoiding empty NVRTC probes on a
new box. `WOOF_KERNEL_PLAN_DIR` can still relocate replay plans independently;
it no longer partitions images or compiler metadata.

The parallel pre-warm inventory includes selected physics generations and both
the declared domain and synthetic-column addressing shapes. Independent CuPy
validator reductions warm concurrently at the column's vertical, surface and
volume sizes. An automatic tile layout can introduce additional addressing
shapes; its actual first run must populate those entries before exporting a
pack qualified for that layout.

Pre-warm publishes completed images before returning its receipt. The receipt
reports image hits, misses, stored, invalid and unbound counts, plus distinct
unbound reasons. A real-size forecast is still needed to demonstrate that the
pre-warm inventory covers every shape-dependent first-step kernel.

The kit can call `tools/kernels/seed-box.sh /work/kit/kernels-*.tar.gz`
after installing its pinned environment. `WOOF_PYTHON` selects that environment's
interpreter; seeding does not acquire a card. Include both sm_120 and sm_89 packs
when the same kit serves Blackwell RTX and Ada boxes.
