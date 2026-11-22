# Determinism, and what the no-ECC dual-run screen detects

Consumer GeForce cards have no ECC memory. WOOF-ArWen's answer to that is
running the same forecast twice and comparing the bytes. This page
states exactly what that comparison detects and what it does not,
because the difference bounds what you may conclude from a WOOF-ArWen
result.

Short version: **dual-run byte comparison is a transient-fault screen
inside a fixed numerical environment. It is not a replacement for ECC.**
Equality of two runs cannot detect a fault that is identical in both
runs.

## 1. The claim, stated precisely

> Two independent executions in one pinned numerical environment are
> compared at the same declared sample instants. A mismatch in the
> compared inventory, scalars, state digest, or output bytes is a
> failure. Equality proves only that all compared bytes were equal at
> those instants, modulo SHA-256 collision. It does not prove numerical
> correctness, and it does not exclude common-mode, repeatable, latent,
> or out-of-scope corruption.

Every clause is essential. "Independent executions" excludes
comparing a run against itself or against a cached artifact.
"Pinned numerical environment" is defined in section 3. "Declared
sample instants" matters because state between samples is only observed
through effects that survive into a later compared byte. "Compared
bytes" matters because the compared surfaces are not the whole live
model state: section 4.

## 2. What equality does and does not detect

**Detected.** An independent transient event (a single-bit upset in
VRAM, a register or execution-unit glitch, a bad DMA) that changes any
byte covered by a compared surface before the next sample, in one run
and not the other. This is the fault class the screen exists for, and
it is the fault class ECC-less consumer memory actually exhibits.

Also detected, though these are environment errors rather than hardware
faults: a driver, library, configuration, or input that differs between
the two arms, because that changes bytes too. The screen cannot tell
you *which* of the two it found; it tells you the arms disagree.

**Not detected.** Equality is silent about every fault that is the same
in both runs:

- a persistent VRAM cell fault hit at the same allocation and the same
  address in both runs: likely, not unlikely, when two sequential runs
  on one idle card allocate the same pool in the same order;
- a repeatable execution-unit, driver, or compiler fault under identical
  instruction and data stimulus;
- corruption already present in a shared input file or a shared
  checkpoint, which both arms then read faithfully;
- a host-memory or hash-computation fault that produces the same wrong
  digest twice;
- a fault in memory that no compared surface covers, which has not yet
  propagated into a compared byte by the last sample;
- any corruption after the final read;
- a SHA-256 collision (cryptographically negligible, listed for
  completeness).

**Also not detected, and worth saying separately: equality is not
correctness.** Two runs that agree exactly may both be wrong: wrong
input, wrong configuration, a physics error, a numerical error. The
screen observes divergence. Nothing on this page is a statement about
forecast quality. [VERIFICATION.md](VERIFICATION.md) distinguishes scoped
code-verification comparisons against WRF from the separate, limited
observation-based validation results. Byte identity supplies neither an
observation score nor a general forecast-accuracy claim.

### The independence trade

Running both arms sequentially on one card is what makes byte identity
achievable at all: it holds the reduction order, compiled image, and
subnormal behaviour fixed (section 3). The same choice maximises
common-mode exposure: identical hardware, identical allocation pattern,
identical instruction stream. Two physically independent cards of the
same model, under one pinned software stack, make the hardware failures
independent and are the stronger configuration where they are
available. Two cards of *different* models are not a stronger
configuration for this purpose: byte identity across architectures is
not claimed by this project and would have to be established route by
route before a mismatch could be read as corruption rather than as
expected architectural difference.

## 3. The fixed environment: the pin set

"Fixed numerical environment" means every item below is identical
between the two arms. Change any of them and a mismatch no longer means
what section 1 says it means.

| Pin | Why it is in the set |
|---|---|
| Physical GPU (model, SM version, and (for a same-card screen) the same UUID) | Reduction tile sizing and math-library results are architecture properties (subnormal handling follows the compiled instruction, so the toolkit and CuPy rows below are what pin it) |
| Driver version | Owns JIT/PTX translation and runtime behaviour |
| CUDA toolkit / NVRTC used to compile kernels | Kernels are compiled at runtime; a different NVRTC is a different compiled image |
| CuPy version, and `CUPY_ACCELERATORS` | Owns the reduction and scan implementations WOOF-ArWen calls (below) |
| NumPy version | Owns host reductions and ufunc code generation |
| netCDF4 / HDF5 versions | Own `wrfout` byte serialization and layout |
| WOOF-ArWen version and git commit | The code and the packaged kernels |
| Configuration file bytes | `woof run` records this as `config_digest` |
| Every input artifact's bytes | Section 5 |
| Runner route and I/O mode | `woof run` and the prepared-runner tools sample different things |
| Output and diagnostic mode, history cadence, checkpoint cadence | These change what is computed and when it is observed |

WOOF-ArWen does not pin these for you. The distribution declares dependency
*lower bounds* (`numpy>=2.0`, `netCDF4>=1.6`, `cupy-cuda12x>=13.0`), not
exact versions, and ships no lockfile. If you intend to use the dual-run
screen, pin the stack yourself (an exact-version environment file, a
container image, or both) and record what you pinned beside the result.

What WOOF-ArWen records automatically today: the supervised `woof run` route
computes a SHA-256 config digest and per-input hashes before the worker
starts, and passes the GPU UUID, GPU name, and driver version to it.
Those input hashes and that GPU identity are written to a durable
artifact only on failure, in `failure-capsule.json`; a *successful* run
leaves `run-progress.json` (run id, config digest, progress) and its
stdout log, which prints the resolved configuration and the input
catalog fingerprint. A successful run additionally writes
`certification-capsule.json` beside its output, which carries one entry
per item of the table above with an explicit resolved/unavailable status,
the code identity, the compiled kernel manifest, and the emitted frames
with their SHA-256. Section 7.

### Three mechanisms that make the pin set necessary

These are the specific reasons "same answer on a different stack" is not
promised. Each is a property of WOOF-ArWen as built, not a hypothetical.

1. **Library-owned reduction order.** The RRTMGP solar-spectrum
   normalization uses a NumPy float64 sum on the host, an order that
   belongs to NumPy. It is fixed for a fixed build, shape, and device;
   NumPy publishes no cross-version bit-reproducibility contract for it.
   The vertical mass-flux construction in the dynamical core (WRF
   `calc_ww_cp`) used CuPy's `sum` and `cumsum` over the vertical axis
   through 2.7.5; since 2.7.6 it is one thread per column in the
   Fortran's own operation order, compiled with `-fmad=false` and
   explicit round-to-nearest intrinsics and checked bit-for-bit against
   a float32 scalar transcription, so that order is the project's and no
   longer part of this item.
2. **FMA contraction and CUDA math functions.** Kernels are compiled
   with `-std=c++17` and no contraction restriction, so the compiler may
   fuse a multiply-add: a different operation sequence, not a different
   rounding of the same one. Live kernels call `expf`, `powf`, `logf`,
   `sinf`, `sqrtf`, `cbrtf`, and `tgammaf`, which carry ULP bounds
   rather than a correctly-rounded result. (Nest interpolation and the
   fused dycore column kernels, couple-momentum, the scalar update and
   the Omega column scan, are the deliberate exceptions: they compile
   with `-fmad=false`.)
3. **FP32 subnormal flushing.** Subnormal handling is a property of
   the compile route rather than of the device, and the mitigations
   WOOF-ArWen applies are per-route. See
   [HARDWARE.md § FP32, subnormals, and GPU-model caveats](HARDWARE.md#fp32-subnormals-and-gpu-model-caveats)
   and the per-scheme records in [PHYSICS.md](PHYSICS.md).

None of these are defects. They are the reason the guarantee is scoped
to one environment rather than to arithmetic in general.

## 4. What is compared, and what each surface covers

### Canonical state digest

`woof.state_digest.canonical_state_digest(state, clock)` is the
strongest surface. At its default `scope="trajectory"` it hashes, in
sorted member order, the name, dtype, shape, and full contiguous bytes
of every restart-serialized state array, serialized scratch accumulator,
and serialized driver/surface/tendency array, plus registered lazy
nest/LBC/reflectivity members: together with elapsed time, the exact
FP32 bits of `dtbc`, driver call counts, the YSU guard count, and the
microphysics update count.

It hashes the entire *stored* array shape, so prognostic boundary rows
and allocated halo cells are included, not cropped.

It does not cover:

- live setup arrays and scalars: base state, map factors, coordinate
  coefficients, vertical coordinates, terrain, Coriolis/rotation arrays;
- external lateral-boundary source tables (these appear in the setup
  fingerprint, which restart validates but canonical state receipts do
  not carry);
- frame scratch `refl_t`, and the two child-duty buffers
  `scratch/nest_parent_field` and `scratch/nest_child_field`, which
  `scope="trajectory"` excludes by design;
- rebuild-only working arrays.

Consequence to hold onto: a bit flip in a base-state or map-factor array
can be latent at one sample and only alter covered state later. A
*final-only* sample cannot attest that the live setup was uncorrupted
throughout.

### `wrfout` files

`wrfout` writes fixed model and start times and a deterministic field
order; it adds no wall-clock timestamp, so two identical runs produce
byte-identical files on one netCDF/HDF5 build. It carries winds, theta
perturbation, geopotential and mass perturbations, selected base and
setup fields, moisture, optional pressure, microphysics moments,
accumulated precipitation, and selected PBL and land diagnostics.

It is a selected *history* surface, not the model state. It omits
cross-step carriers such as `al`, `alt`, `h_diabatic`, held
radiation/PBL/cumulus tendency arrays, scheduler counters, many surface
driver fields, and most scratch state. Whole-file equality proves
equality of what was serialized.

Whole-file SHA-256 also binds the netCDF/HDF5 serialization build, not
only the science. Across different netCDF4/HDF5 versions it can report a
mismatch on layout alone. Within one pinned build (which the pin set
already requires) that strictness is a feature.

**Compare field inventories before you compare hashes.** Two files whose
variable lists differ are not a valid comparison, whatever their digests
do. The `woof.verify` case comparators do this in the right order:
schedule, then identity, then inventory, then scalars, then SHA-256.

### What is never a comparison surface

- **Checkpoint archives.** Restart headers carry the current UTC time,
  and tree restart members carry a fresh UUID in the header and the
  filename. Two identical model states do not produce identical
  checkpoint bytes, by construction.
- **Run reports and receipts.** They contain wall time, timings, memory
  peaks, and absolute paths.
- **`run-progress.json`.** It records the run id, the config digest, the
  progress and the terminal status, none of which is model state, and
  its timestamps and pid differ between runs by construction.

"Bit-identical restart" therefore means the *downstream* state digests
and regenerated outputs match under the same build, cadence, and
diagnostics, never that the checkpoint files match.

## 5. Input identity

Both arms must consume the same input bytes; otherwise a mismatch is a
staging error and, worse, a *shared* corrupt input makes both arms agree
on the same wrong answer.

Declared file inputs are content-hashed with SHA-256. Declared
*directory* inputs (in practice the static geography tree) have two
modes, chosen with `--directory-input-hash` on `woof run` / `woof
resume`, or the `WOOF_DIRECTORY_INPUT_HASH` environment variable:

| Mode | Binds each file by | Cost | Known behaviour |
|---|---|---|---|
| `inventory` (default) | relative path, size, mtime | Stat only | A byte-identical copy staged separately compares **different**. A changed file that preserves path, size, and mtime compares **equal**. |
| `content` | relative path, size, SHA-256 of the bytes | Reads every file | Neither of the above. Costs a full read of a multi-GB tree before every launch. |

The default is `inventory` because it runs before every launch on a
static geography tree large enough that re-reading it is a real cost.
Use `content` when the two arms stage their geography separately, or
when an mtime-preserving change to that tree must not go unnoticed. Each
recorded hash carries the algorithm that produced it
(`sha256-directory-inventory` or `sha256-directory-content`), so digests
computed under different modes are distinguishable and must not be
compared to each other.

`inventory` digests are unchanged from earlier releases: no domain tag
was added to the record layout, so an older recorded digest still
compares equal.

## 6. Running the screen

`woof dual-run` is the fail-closed comparator for this screen. It reads
the two successful-run certification capsules, compares their complete
leaf inventories in deterministic path order, exits nonzero on any
difference, and can write a durable JSON report:

```bash
woof dual-run \
  --capsule-a out/run-a/certification-capsule.json \
  --capsule-b out/run-b/certification-capsule.json \
  --out-report out/dual-run-report.json
```

It has three outcomes, not two:

| Exit | Meaning |
|---|---|
| 0 | identical: `dual-run: capsules are identical field for field (71 compared quantities)` |
| 1 | a divergence, naming the first field the two arms disagree on |
| 2 | there was nothing to compare, or an arm could not be read |

Read the count on the success line. "Identical" is the same sentence
whether the screen compared one quantity or a hundred, and on this
screen the size *is* the result: a pair of capsules that agree on 71
leaves is a corruption screen, and a pair that agree on two is not.
`--out-report` records the same number as `compared_count`.

Exit 2 is what a screen over nothing gets. A zero-byte capsule, an empty
one, or a pair that between them carry no leaf at all is refused by arm
and by name with its byte count: this command used to answer two `{}`
documents with `identical field for field` and exit 0, which is a green
on nothing from the only detector that stands in for the ECC this card
does not have. A real capsule against an empty one is still a
divergence (exit 1), not a refusal: that pair has something to compare
and it does not match.

The comparison has no ignore list. The only normalization is for the
pure output-location leaves `output.frames[i].path` and
`receipts.<name>.path`: their final filenames are compared because two
arms must use different output directories. Frame bytes and SHA-256,
trajectory digests, inputs, numerical-stack pins, GPU identity, code,
and every other capsule leaf remain literal. A missing leaf and a leaf
whose value is `null` are different claims.

The procedure:

1. Pin the environment (section 3) and record the pins.
2. **Prepare once.** Both arms read one prepared root, one geography
   tree, one config, exactly as they read one set of GRIB files. The
   subject of this screen is the forecast integration, and a second
   preparation is a second source of variation that is not it. It is
   also not reproducible even in principle: a preparation receipt
   records its own wall-clock timings and the staging directory the
   decoder used, so two preparations of one configuration never produce
   the same receipt bytes. `woof dual-run` compares every input
   artifact's bytes verbatim and always will (that comparison is what a
   swapped or corrupted input trips) so two independently prepared arms
   cannot agree and the screen returns nothing usable. Use
   `--directory-input-hash content` if the arms do not share one staged
   geography tree.
3. Run the forecast twice from that one prepared root into two different
   `--outdir` directories, on the same card, with nothing else on the
   GPU. Two directories are required: one would overwrite the other.
   `woof dual-run` normalizes the fields that record *where* a run
   wrote (`output.frames[i].path`, `receipts.<name>.path`) to their
   leaf name and still compares them, so a run that wrote the wrong
   domain, the wrong valid time or a different number of frames still
   diverges. Every byte that carries physics is compared verbatim.
4. Compare, in this order, and stop at the first difference:
   - **schedule**, the two runs produced the same number of frames with
     the same names at the same model instants;
   - **environment identity**, the pin set matches, from your own
     record;
   - **inventory**, each pair of `wrfout` files declares the same
     variables with the same dtypes and shapes;
   - **bytes**: SHA-256 of each `wrfout` file pair.

   `woof dual-run --capsule-a A/certification-capsule.json --capsule-b
   B/certification-capsule.json` does the schedule, environment-identity
   and bytes rows of that list from the capsules, and exits 0 only when
   every leaf agrees. The `wrfout` variable inventory is the row it does
   not cover.
5. Any difference is a failure. Investigate it; do not average it away.

If you use the prepared-runner tools in `tools/`, they additionally emit
a final canonical trajectory digest per domain plus per-file SHA-256 at
the end of a run, which gives you a state surface as well as an output
surface. They emit it *at the end only*, so an intermediate transient
that has been overwritten by the final sample is out of reach; the
`wrfout` frame comparison is what covers the run's interior.

The reference implementation of the comparison ordering (schedule
binding, then inventory, then scalars, then hashes, recording canonical
state at every history instant and every checkpoint) is the
historical-reference d01 verification case under
`woof/verify/cases/`. Read it if you are
building your own comparator.

## 7. Known detector limits

The fail-closed comparator and mechanical environment-identity refusal
described above are shipped. The remaining limits are listed so that a
dual-run PASS is not mistaken for evidence outside the detector's scope.

- **A comparator that opens the output files.** `woof dual-run`
  compares the capsules, which carry each frame's SHA-256 but not its
  variable list, dtypes or shapes. A run pair that diverged in the
  header would be caught by the frame hash without being *named* as a
  header difference, and a pair with no capsules cannot be compared at
  all. The variable-level inventory ordering exists inside the
  verification case, not as an ordinary run-path contract.
- **A dual-run surface for the preparation stage.** Two arms of the
  screen share one preparation (section 6), so the preparation itself is
  not screened. Screening it would need a receipt whose bytes can repeat:
  today a preparation receipt records its own wall-clock timings and
  staging directory, so its digest cannot. The content digests inside it
  (`prepared_cache.content_sha256`, the input-manifest digest) do repeat
  and are the surface a preparation screen would be built on.
- **Project-owned reduction order.** The dycore's vertical mass-flux
  construction left the library-version pin in 2.7.6 (section 3,
  mechanism 1); replacing the host solar normalization with a
  fixed-order implementation would move the last such item out of it.
- **A detector-scope state digest.** A scope that additionally hashes
  live setup and LBC bytes and the full child-duty buffers, emitted at
  every history and checkpoint boundary rather than at stop.
- **Cryptographic checkpoint content identity.** Per-member and
  aggregate SHA-256 in the checkpoint header, fsync-and-reopen
  validation before publication, and a strict current-format identity
  mode that refuses compatibility migrations and diagnostic-mode
  changes.
- **`content` as the default directory-input mode**, once the cost of
  hashing a large static tree is amortized by a cached, content-bound
  manifest.

## 8. Not claimed

- Dual-run byte equality is **not** an ECC substitute and does not
  detect every hardware fault.
- No cross-GPU, cross-driver, or cross-library bit identity is claimed.
- No claim that byte equality implies numerical or meteorological
  correctness.
- No claim that checkpoint files are byte-reproducible.
- No claim that the compared surfaces cover all live device memory.
- No claim about a run's interior between sample instants, except
  through effects that survive into a later compared byte.

## Related

- [VERIFICATION.md](VERIFICATION.md): what is verified against WRF
  v4.6.1, and what is not.
- [HARDWARE.md](HARDWARE.md): FP32, subnormals, VRAM sizing, and
  platform notes.
- [PHYSICS.md](PHYSICS.md), per-scheme maturity and the registry of
  known numerical divergences.
