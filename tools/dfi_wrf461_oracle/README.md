# DFI source and operator evidence

Status: **DFL, DDFI, TDFI and filters 0 through 7 are connected to resident
model initialization**. Stock namelists reach the same configuration and
runner as prepared inputs. Filter 8 remains held for witnessed undefined
native RHOFIL indexing; no substitute filter is installed.

`SOURCE.json` pins the original public WRF v4.6.1 files. `NOTICE.txt` records
the upstream public-domain notice, checked before translation. No external
tables or competitor implementation is used.

The Rust crate at `tools/dfi-runtime` translates filters 0 through 7 and native
odd/even coefficient mirroring. It also expresses the three native time
windows and exposes a CUDA-driver submission primitive. The CUDA module
`woof/core/kernels/dfi.cu` accumulates descriptor-selected device arrays and
normalizes them, with the native nonnegative clamp for moisture, scalars and
effective radii. Fast numerics is the default; strict proof opts in with
`GPUWM_NUMERICS=strict`. The production bridge is `core/dfi_model.py` and
`core/dfi_initialization.py`. `core/dfi_graph.py` records each actual signed
model solve with its own clock, boundary and radiation arguments; one Rust
`aw_dfi_submit_compiled_pass` submits the complete finite pass. Python runs
during graph construction, never between the submitted weather solves.

`woof/core/dfi.py` now owns a complete explicitly supplied native carrier
inventory, descriptor tables and saved arrays. `dfi_reset.cu` implements
surface snapshots, snow maxima, snow-dependent soil restoration, optional
RUC fields, hydrometeor/scalar restoration, Thompson RSLF saturation and
column-parallel native rebalance. It requires actual native fields and rejects
missing, noncontiguous or overlapping writable carriers. It does not infer
native pressure, theta reference, vertical mass flux or radius units from a
DomainState. `ModelDfiState` supplies these fields and their native units.

`DeviceDfiState` uses process-default fast numerics; strict proof requires
`GPUWM_NUMERICS=strict` before process startup. Its graph resources and buffers
remain owned through completion. `clear()` follows the literal native clear:
LIVE effective radii become zero while their accumulation arrays are retained.
It does not silently change that native inter-pass behavior.

The Rust `aw_dfi_submit_tree` validates a connected integer-ratio nest tree,
signed timesteps and exact pass duration before its first CUDA submission.
It forces all siblings before integrating any sibling, suppresses feedback
through both DFI passes and samples before each solve and at each recursive
return. Child endpoint coefficients repeat on re-entry. Accumulation graphs
must not advance the coefficient cursor; a separate advance graph does that
once per actual model solve. There is no readback, wait, per-step allocation or
Python callback in this dispatcher. The caller must supply real model graphs
with advancing signed clocks, boundary slots and physics cadence. The event
contract test is `check_tree_gpu.py`; it does not supply those model graphs.

Fetch the pinned original source into a scratch directory, then run:

```sh
curl -fL https://raw.githubusercontent.com/wrf-model/WRF/v4.6.1/share/dfi.F -o /scratch/dfi.F
python3 tools/dfi_wrf461_oracle/check.py --source /scratch/dfi.F --output /scratch/dfi-check
python3 tools/dfi_wrf461_oracle/check_gpu.py --source /scratch/dfi.F --output /scratch/dfi-check
cargo test --offline --manifest-path tools/dfi-runtime/Cargo.toml
python3 tools/dfi_wrf461_oracle/check_admission.py --output /scratch/dfi-admission
```

The GPU check needs CuPy, NumPy, NVRTC, the CUDA driver and a CUDA device.
`check.py` needs Rust and gfortran. Neither needs a Python forecast loop.
The scripts record their native compiler commands and receipts. Generated
binary fixtures and executables are disposable after recording the results.

`check.py` extracts complete original `dfcoef` and filter subroutines without
rewriting their arithmetic. Native O0 and O2 must first agree. It compares
968 chronological coefficient words for filters 0 through 7, including both
sample parities. This is a bounded operator check with synthetic time-window
inputs, not a real-run capture. The same executable built with bounds checking
witnesses original filter 8 writing outside `FROW(0:NOSIZE)` in `RHOFIL`.
The Rust API returns -8 for this undefined native case, not substitute weights.

`check_gpu.py` extracts the complete original `dfi_accumulate` routine and
the original dynamics/moisture normalization block from `dfi_array_reset`.
Minimal domain modules allocate synthetic arrays with two moisture and scalar
species and a non-block-aligned dimension of 1031. The real original statements
run for 60 samples. Both effective-radius branches are checked. A synthetic
CUDA graph advances arrays by fixed increments; one Rust FFI call submits
59 graph advances and accumulation operations after the initial sample.
It validates the accumulated arrays, sequential coefficient total, and reset
arrays against native Fortran words. There is no Python callback or device
readback between samples. One final stream completion precedes comparison.
This proves the primitive's GPU residency, not a model dispatcher or forecast.
Fast-mode differences are recorded, not represented as strict identity.

`instrument_native.py --build /private/strict-wrf --receipt /scratch/instrumentation.json`
checks the original `share/dfi.F` pin before adding four observation-only calls.
It verifies that removing those calls recovers every original source byte.
The appended writer captures a fixed interior 8 x 4 x 8 tile, with separate
files before and after accumulation and before and after array normalization.
Use a private copy of a strict build; never instrument another run's tree.
The existing WRF 4.6.1 build has the pinned original DFI source.
The native executable must then be rebuilt and run on actual weather inputs.
The instrument requires domain 1 to include global i/j=31..38 and k=2..5;
one rank must own the entire horizontal tile. It observes savehydmeteors=0.

After that run, copy its `lane-dfi-*.bin` files to a private scratch directory:

```sh
python3 tools/dfi_wrf461_oracle/replay_capture.py --capture /scratch/capture --output /scratch/replay --dt 15 --filter 4 --cutoff 3600
```

Supply the actual captured namelist's timestep, filter and cutoff. The check
verifies native continuity and unchanged inputs, compares Rust coefficients,
preloads the complete captured trajectory on the GPU and submits each pass
through Rust. It compares every sample's accumulation, the final coefficient
total and the normalization output. Both strict and default fast results are
recorded. This is a replay of real native operator inputs, not GPU dynamics.
Full surface restoration and savehydmeteors=1 have a separate full-column
oracle below. This tile check still does not integrate a weather model.

The first 2026-11-01 real-input receipt records 155,160 strict words with
zero differences on an RTX 5090: nine backward samples and thirteen forward
samples from native TDFI, filter 4, dt 15 seconds, cutoff 3600 seconds. Each
sample contains the 15 dynamics carriers, seven moisture species, one scalar
and three effective-radius arrays on the captured tile. The count includes
22 coefficients, 142,210 accumulation/total words and 12,928 normalization
words. Native WRF 4.6.1 completed both DFI passes and a four-minute forecast;
its original DFI source matches the 4.6.1 pin byte for byte. This is neither
a GPU forecast. Later DFL, DDFI and native nested operator receipts are
described below. These historical module replays are separate from the later
three-hour production forecasts.
The report and receipts are under the requested `WRF-PORTS-2026-11-01`
Downloads directory, in `port-dfi.md` and `evidence/port-dfi/`.

`check_admission.py` calls the real namelist importer with the repository's
stock two-domain fixture and both feedback modes. It verifies that opt=1/2/3
are admitted in each domain. This checks admission only, not disabled-option output,
a forecast or restart.

`instrument_nested.py` adds sequence-numbered captures for every domain,
so repeated endpoint samples cannot overwrite each other. The real nested
TDFI run (parent dt 15 seconds, child dt 5 seconds, ratio 3) produced 212
records. Replay compared 685,290 strict words with zero differences, including
seven repeated backward and eleven repeated forward child endpoint samples.
Supply `--domain-dt 2=5` to `replay_capture.py` for this case. These are actual
native nested operator inputs, replayed independently by domain; they do not
prove a coupled GPU nest forecast. `check_tree_gpu.py` separately matched 178
real CUDA-driver events in both clock directions, including a grandchild and
siblings, and rejects a bad signed ratio before the first write.

`instrument_reset.py` observes complete save, clear, reset and rebalance on an
8 x 8 tile of full native columns. `instrument_all.py` composes that writer
with the all-domain accumulation writer. Each instrument asserts that removing
its observation calls recovers the preceding source byte for byte, and the
composition starts only from the pinned original source. Rebuild the whole
private native WRF executable before running it. Then use:

```sh
python3 tools/dfi_wrf461_oracle/replay_reset.py --capture /scratch/capture --output /scratch/reset
GPUWM_NUMERICS=strict python3 tools/dfi_wrf461_oracle/replay_device_state.py --capture /scratch/capture --output /scratch/state
python3 tools/dfi_wrf461_oracle/check_tree_gpu.py --output /scratch/tree
```

The real TDFI savehydmeteors=1 WSM6/Noah capture compared 239,104 strict words
with zero differences in the full save/reset/rebalance replay. The actual
DeviceDfiState save/reset CUDA graphs independently compared 201,088 words,
zero different. Counts overlap and must not be summed as unique model state.
The first executed reset instrument captured modes 4,5,6,7. Mode 8 was absent
because its inserted call followed a preprocessor endif on the same line.
The source now puts each call on its own line. That earlier replay uses the
immediately following complete-reset mode 6 as rebalance's post-state; no
native arithmetic occurs between the rebalance return and that observation.
The exact executed instrument and receipt remain with the evidence.

The composed instrument subsequently captured real DFL and DDFI with
savehydmeteors=0, including before/after clear states. Their accumulation
replays compare 64,650 and 174,556 strict words, respectively; the complete
reset replays compare 86,848 and 89,984, and the DeviceDfiState API replays
compare 255,616 and 427,520. Every comparison has zero differing words.
These counts overlap. DDFI's backward accumulation has no native reset;
the replay requires a normalization pair for every other captured pass.

`check_clear.py` additionally extracts the complete original native clear
routine and seeds every live and saved field with nonzero values. Its
230,948-word synthetic GPU control covers both radius inventories and both
savehydmeteors modes, including an extent of 1031. It witnesses that native
clear zeros live radii while retaining their accumulation arrays. It is
separate from the real-input proof and does not increase its word counts.

A strict option-off ten-step forecast was also byte-identical to the candidate:
26 fields, 116,762 stored values and identical whole NetCDF SHA256. This is
an off-path regression proof, not an option-on DFI forecast or model timing.

Production initialization now retains actual WW/PHP/PH0 diagnostics, packed
Registry moisture/scalars, original geography and live soil/surface inputs.
Backward integration disables native physics/diffusion selectors, retains
negative boundary time on the device, and restores the forecast configuration.
Both feedback modes run the coupled tree; feedback is suppressed during DFI.
Each ordinary ensemble member filters its own perturbed initial state. A saved
forecast checkpoint resumes its filtered state without filtering it again.
Checkpoint publication occurs only after initialization completes.

History snapshots are graph device copies, downloaded at pass completion.
Filtered input uses the original stock input template, or the existing stock
WRF exporter for prepared caches. The initialization memory ledger includes
saved carriers, duplicate physics initialization, calendar tables and snapshots.
The signed passes use the resident road, with ordinary VRAM admission before
restore; they do not page the backward atmosphere through a host tile store.

The 2026-11-01 lane evidence under `evidence/port-dfi/` includes three-hour
stock `woof go` forecasts for selectors 1, 2 and 3; coupled feedback 0/1;
Rust temperature/reflectivity maps at 10/11/12 UTC alongside native WRF;
ASOS comparisons; stage CUDA events; and the disabled-option byte comparison.
These are the WSM6/YSU/Noah 301x301x50 native-input case and a 97x97x50 child.
Operator counts above overlap and do not establish whole-model bitwise parity.
`replay_coefficients.py` binds the actual captured time-window parameters to
native DFCOEF and checks all eight defined selectors (72 coefficient words,
zero different). The older 968-word test remains synthetic supplementary proof.

Native lifecycle anchors are `main/module_wrf_top.F:2133` and
`share/set_timekeeping.F:91`. DFL filters one forward window and starts the
forecast at its midpoint. DDFI runs backward without a reset, then filters
forward and starts the forecast at the original analysis. TDFI filters and
resets the backward window, then filters forward from its backward midpoint.
The accumulation ordering is in `frame/module_integrate.F:365,490`: sample
before each solve and at every recursive interval return. A child endpoint
is sampled again on the next recursive interval entry with the same coefficient
index. Treating those repeats as new model timesteps gives the wrong filter.
