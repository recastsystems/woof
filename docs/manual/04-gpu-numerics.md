# 4. GPU numerics a researcher must know

The model state is FP32, like WRF's default REAL; no end-to-end bit-identity with
WRF is claimed anywhere [README.md, current release scope]. Four numerics facts govern how to read
WOOF-ArWen results: subnormal handling follows the compile route, FMA contraction and
library reduction orders scope the determinism claim, dual-run byte comparison is a
transient-fault screen (not ECC), and restart identity is a published contract.

## 4.1 FP32 subnormal flush is a property of the compile route, not the SM

Measured on an RTX 5090 (compute capability 12.0, CUDA driver API version 13.3
(13030) as the receipt records it; not the display driver's 5xx.xx number)
across the six
compile routes the model uses, crossed with six arithmetic mechanisms; the receipt
statement is generated into the hardware page and the raw evidence (bit-pattern
tables, cubins, SASS) is committed [docs/public/HARDWARE.md, the block between
the `BEGIN GENERATED ftz-statement: hardware-fp32-subnormals` and
`END GENERATED` markers; tools/ftz_receipt/receipt/]:

| route | what it is | verdict (6 mechanisms) |
|---|---|---|
| R1 | loader RawModule, `-ftz=true` | flush-to-zero 6/6 |
| R1-ftztrue | R1 + explicit `--ftz=true` (control) | flush-to-zero 6/6 |
| R2 | direct NVRTC with the shortwave option tuple, `--ftz=false` (the shortwave site left `RawModule` for the bypass R3 uses when NVRTC 13 rejected the duplicated flag, so no `-ftz=true` follows it) | ieee-agreement 6/6 |
| R3 | direct NVRTC + module load, `--ftz=false` | ieee-agreement 6/6 |
| R4 | CuPy ReductionKernel | flush-to-zero 6/6 |
| R5 | inline PTX without `.ftz`, riding R1's compile | ieee-agreement 4/6, not-applicable 2 |

The decisive observation: R5 and R1 are kernels inside one compiled object, same
device, same flags, one compile, and they did not measure alike, so on this device
the outcome follows the instruction the compiler emitted rather than the hardware
alone [the `R5` entry of that generated block]. The control arm matters: the three
distinct bit tables among the six arms are what shows the pipeline responds to the
flag at all. This corrects earlier project lore that "sm_120 flushes subnormals in
all arithmetic and `--ftz=false` is ineffective"; on the direct-NVRTC route
`--ftz=false` survives and all six mechanisms score ieee-agreement. The consequence
that reaches the science is a branch flip on physically negligible inputs, not a
change in a resolved quantity [the first bullet after that generated block].

## 4.2 The shortwave subnormal countermeasure

The RRTMG shortwave transmittance chain really produces subnormals: the
exponential table floors at 1e-20, so two clamped layers give a 1e-40 transmittance
product. Measured by an instrumented witness build over a diurnal batch: 1,659,113
subnormal results and 2,166,356 subnormal operands in 304,204,382 macro calls, all
inside the direct-beam transmittance chain and what consumes it; the preparation
stages witness none [woof/core/kernels/rrtmg_sw.cu:77-85].

The shipped countermeasure is inline PTX without `.ftz`: FP32 add/sub/mul in that
chain are written as `add.rn.f32` / `sub.rn.f32` / `mul.rn.f32`, one instruction
each, full IEEE gradual underflow, immune to the compile option tuple; the same
idiom the dycore, acoustic, and MYNN kernels already use for their rounding-tree
discipline. The reasoning is route-independence: the file's own compile route today
would keep subnormals, but that is not something to build a numerics contract on;
PTX is the only mechanism correct under every route, so the compile route stays a
performance choice rather than a correctness one
[woof/core/kernels/rrtmg_sw.cu:67-91].

An FP64-emulation form of the same three operations is retained verbatim as the
reference semantics and as the witness build's cross-check arm. The two are exactly
equal, not merely close: the product of two binary32 values is exact in binary64,
and for sums the double-rounding is provably innocuous (Figueroa), while a sum
landing in the binary32 subnormal range is itself exact in binary64. The proof
chain: a CPU identity suite with exhaustive low-order and cancellation sweeps
[tests/test_rrtmg_sw_rn_identity.py], the on-card witness (calls where the PTX arm
and the FP64 arm disagree must be zero), and the max-ULP-0 oracle and batched gates
end to end. Two conversion helpers stay live because the compiler's own conversions
flush on this route; division and sqrt stay on hardware `__fdiv_rn`/`__fsqrt_rn`
under a named invariant that every operand reaching them is non-subnormal, with the
two audited exposure sites listed [woof/core/kernels/rrtmg_sw.cu:93-133]. The
aerosol route is closed for `aer_opt` 1 and 2 by refusal; `aer_opt = 3` reaches
the division with optics whose range is proven normal for any layer of nonzero
mass [woof/core/rrtmg_aerosol_optics.py, tests/test_rrtmg_aerosol_optics.py].

Other schemes carry subnormal-class records in the registry rather than silent
behavior: YSU's two FTZ-class branch disagreements are closed (the
subnormal-`br` branch by the same double-compare countermeasure, and the
flushed subnormal `ust` by removing the zero-flux short circuit it reached),
leaving only WRF's own `prfac2` 0/0 recorded as a declared divergence; Shin-Hong
records the subnormal-flush branch closed by a double-compare countermeasure with
72 residual flush lanes pinned as counts; the radiation preparation path routes one
subnormal-sensitive block through the host by design
[docs/public/PHYSICS.md:868-881; docs/public/HARDWARE.md:552-553].

## 4.3 FMA contraction and library-owned reduction order

Two further mechanisms scope the determinism claim [docs/public/DETERMINISM.md:128-150]:

1. The RRTMGP solar-spectrum normalization uses a NumPy float64 host sum, an
   order that belongs to NumPy. The vertical mass-flux construction (WRF
   `calc_ww_cp`, `woof/core/dycore.py::_omega_ref`) used CuPy's `sum` and
   `cumsum` over the vertical axis until 2.7.6; it is now one thread per column
   in the Fortran's own operation order (`divv = (msftx*dnw)*(rdx*du + rdy*dv)`,
   `dmdt` summed sequentially from the surface up, `ww(k) = (ww(k-1) -
   (dnw*c1h)*dmdt) - divv(k-1)` as two subtractions) with explicit
   round-to-nearest intrinsics under `-fmad=false`, bit-identical to a float32
   scalar transcription of the loop [tests/test_omega_column_scan.py], so that
   order is the project's. The readings of the move, taken on the seeded 64x64x32
   state of `tools/benchmark_seeded_step.py` (Morrison, seed 20260731, RTX 5070
   Ti, receipts under `tests/data/receipts/omega-column-scan/`): one call of Omega differs
   from the retired construction by at most 6.0e-7 of max |Omega| (6.5 ULP of
   the field maximum) on the flat route and 5.0e-7 (5.5 ULP) with map factors,
   with 73 percent of the words moving: 17 percent of all words by 1 ULP of their
   own magnitude, 14 percent by 2 and 42 percent by 3 or more, the last group
   being small-magnitude words, since no word moves by more than 6.5 ULP of the
   field maximum in absolute terms; after 60 steps u, v,
   w, theta', phi' and mu' differ by 1.7e-4 to 7.2e-4 of each field's maximum,
   dry core and Morrison alike, and every moisture mass field and number moment
   is unchanged except Morrison's cloud droplet number `nc`, which follows the
   thermodynamic state and moves by 2.5e-6 of its maximum. The whole step got 5.1 to
   7.0 percent cheaper (1.07, 2.02 and 5.33 ms per step at 250x200x49, 320x256x49
   and 480x384x49); the routine alone went from 1,933 to 164 microseconds at
   480x384x49, of which `cp.cumsum` had been 1,054. The phase-2 step capture
   [tests/data/phase2_step_regression.npz] was re-pinned with its readings in the
   ledger of tests/test_coriolis_map.py.
2. Kernels compile with no contraction restriction, so the compiler may fuse a
   multiply-add: a different operation sequence, not a different rounding of the
   same one. Live kernels call `expf`, `powf`, `logf`, `sinf`, `sqrtf`, `cbrtf`,
   `tgammaf`, which carry ULP bounds rather than correctly-rounded results. Nest
   interpolation and the fused dycore column kernels (couple-momentum, the
   scalar update, the Omega column scan) are the deliberate exceptions and
   compile with `-fmad=false`.

None of these are defects; they are the reason the guarantee is scoped to one
environment rather than to arithmetic in general. A concrete published consequence:
mp=28 and mp=8 are deliberately not bit-identical thermodynamics. mp=28's
saturation Horner chains are contraction-pinned while mp=8's stay FMA-contracted;
the two saturation vapour pressures differ by one ULP, and WRF's Thompson opens the
condensation/CCN activation block on `ssatw > 1.E-15`, so one ULP flips a branch.
Neither is a defect, and "make them agree" is the wrong fix in both directions
[docs/public/PHYSICS.md:748-756].

## 4.4 Determinism and dual-run practice

The claim, exactly as scoped: dual-run byte comparison is a transient-fault screen
inside a fixed numerical environment; it is not a replacement for ECC, and equality
of two runs cannot detect a fault that is identical in both runs
[docs/public/DETERMINISM.md:9-22]. It detects an independent transient event (a
single-bit VRAM upset, a register or execution-unit glitch, a bad DMA) that changes
any covered byte in one run and not the other, which is the fault class ECC-less
consumer memory actually exhibits. It does not detect persistent same-address cell
faults hit in both runs (likely, not unlikely, when two sequential runs on one idle
card allocate the same pool in the same order), repeatable
execution-unit/driver/compiler faults, corruption already in a shared input,
faults in memory no compared surface covers, or a SHA-256 collision
[docs/public/DETERMINISM.md:34-61]. Two physically independent cards of the same
model under one pinned stack are the stronger configuration where available; two
cards of different models are not [docs/public/DETERMINISM.md:71-82].

The pin set is eleven items (GPU model/SM/UUID, driver, CUDA/NVRTC, CuPy and
`CUPY_ACCELERATORS`, NumPy, netCDF4/HDF5, WOOF-ArWen version and commit, config bytes,
every input artifact's bytes, runner route and I/O mode, output/diagnostic/cadence
settings). WOOF-ArWen does not pin these for you: the distribution declares dependency
lower bounds and ships no lockfile [docs/public/DETERMINISM.md:90-108].

What is compared [docs/public/DETERMINISM.md:154-224]:

- **Canonical state digest**, the strongest surface: at trajectory scope it hashes
  every restart-serialized state array, serialized scratch accumulator, and
  serialized driver array plus clock and counter state. It does not cover live
  setup arrays (base state, map factors, terrain, Coriolis), so a bit flip there
  can be latent at one sample and only alter covered state later.
- **`wrfout` files**: deterministic field order, no wall-clock timestamp, so
  identical runs produce byte-identical files on one netCDF/HDF5 build. A selected
  history surface, not the model state.
- **Never a comparison surface**: checkpoint archives (headers carry UTC time and
  fresh UUIDs), run reports, progress files. "Bit-identical restart" means the
  downstream state digests and regenerated outputs match, never that checkpoint
  files match.

The comparator `woof dual-run` has three outcomes: 0 identical (71 compared
quantities), 1 divergence naming the first disagreeing field, 2 nothing to compare.
Exit 2 exists because the command once answered two empty documents with
"identical" and exit 0, a green on nothing from the only detector standing in for
ECC [docs/public/DETERMINISM.md:269-290]. Procedure: pin and record; prepare once
(a second preparation is a second source of variation that is not the subject); run
twice from the one prepared root into two output directories on the same card;
compare schedule, environment identity, inventory, bytes, stopping at the first
difference. Any difference is a failure: investigate it, do not average it away
[docs/public/DETERMINISM.md:300-338].

Determinism results actually measured:

- The reference matched run survived two external process kills; frames produced
  before each kill were byte-compared when regenerated after relaunch:
  SHA256-identical. WOOF-ArWen reproduces its own trajectory bit-for-bit under
  restart-free relaunch on the same hardware and build
  [docs/public/VERIFICATION.md, historical comparison and limits].
- LES: two independent 14,400-step `km_opt=2` integrations produced byte-identical
  state and receipts identical in every field but wall time; since extended across
  hardware, the same seed is bit-identical on three different cards (two of them
  different sm_120 silicon) and seed-equivalent against a different architecture
  (sm_89), where the worst receipt field differs by 1.81 difference-sigma against
  the n=18 realisation spread [docs/public/LES.md:193-203].
- A measurement-hygiene example: one receipt field did not reproduce because it
  measured whole-device VRAM occupancy at exit rather than the run's own
  footprint; the field was split into a compared pool figure and an excluded
  device figure rather than tolerated [docs/public/LES.md:132-144].

## 4.5 Restart identity

The restart contract is a small tolerated-difference set held in one place
[woof/core/model.py:307-370]:

- Experiment fields a restart may differ in: `run_seconds`,
  `restart_interval_s`, `acknowledgements`, `relocation`, `tiles`. `tiles` is
  excluded because a domain integrated resident and the same domain tile-streamed
  (section 6.5 introduces streaming) produce the same bytes, proven carrier by
  carrier at every physics rung and
  across a checkpoint in four legs (streamed-file-streamed,
  streamed-file-monolithic, monolithic-file-streamed, all bit-exact); binding the
  mode would refuse exactly the operation streaming exists for, a forecast that
  outgrew its card resuming on the card it outgrew.
- Domain fields: `history_interval_s`. Run fields: `run_seconds`,
  `output_interval_s`, `restart_interval_s`.
- Scheme-scoped knobs drop out of the identity under a different scheme (WDM6 and
  NSSL option sets): a field no path reads cannot move a trajectory, so binding it
  would buy no safety; under their own scheme the fields bind value for value.
- Absent-stays-absent: `[perturbation]`, per-domain `spawn`, and the
  auto-selected-mixing provenance label keep every pre-feature fingerprint
  byte-identical, while the value of `mix_isotropic` itself does bind (section
  2.7).
- A relocation invalidates the restart claim outright: a checkpoint resumes only
  into the run that wrote it [woof/io/restart.py:2460, quoted in
  docs/nest-relocation-identity-decision.md].
- The radiation engine firewall binds: a restart written under RTE+RRTMGP refuses
  to resume under legacy RRTMG and vice versa; P3 binds its lookup table's SHA-256
  into the checkpoint [docs/public/PHYSICS.md:1040-1041, 220-224].
