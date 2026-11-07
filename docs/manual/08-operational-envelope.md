# 8. Operational envelope

Every number here carries its conditions; a number without its grid, lead, or
software version is a different claim. Note the version caveat on the reference
envelope: it was measured 2026-08-03 with the published gpuwm 1.5.0 wheel (a WRF
v4.6.1 dmpar arm ran alongside it in the same receipt). These are not 2.5.0-line
numbers, and no newer throughput measurement exists; the 2.5.0-line measurements
in this chapter are the first-contact walk (section 8.3) and the memory-gate
calibration (section 8.5).

## 8.1 Throughput, capacity, and memory: the reference envelope

Receipt: [receipt:ARWEN-WRF-16GB-REFERENCE-20260803.md]. Method: WRF memory is
kernel VmHWM per rank summed across ranks; WRF timing from its own per-step rsl
lines with the cold first step split from the warmed mean; WOOF memory is the
reported pool peak plus nvidia-smi sampling. Same case class both sides: 3 km
HRRR init, Thompson/YSU/MM5/Noah/Dudhia SW, dt 15 s. The WRF radiation bracket
moves timing only 5.5%; WRF timing stability 0.62% across three repeats; WRF
24-to-48-rank scaling 98.5% efficient.

Operational throughput, the WOOF side of the receipt: an RTX 5090 32 GB runs a
551x551x50 domain (14.82 M mass cells) at 0.871 warmed seconds per simulated
minute in 12.35 GiB of VRAM; an RTX 5070 Ti 16 GB runs 568x454x49 (12.6 M
cells) at 1.27 s per simulated minute in 9.94 GiB. Both runs are dual-run
byte-identical on both wrfout frames (the no-ECC corruption screen, clean). The
same receipt times WRF v4.6.1 dmpar on one 24-core node and a 48-core two-node
pair over the same case class; the per-card and per-cell comparisons, each
valid only under its own conditions (matched-pair against cell-normalized, and
which card a circulating figure belongs to), live in the receipt for whoever
needs them.

The compile tax, a cost WRF does not pay: cold-kernel one-time NVRTC compile is
53.8 s on the 5090 (98 kernels), about 22.6 s on the 5070 Ti; 3.6% of a 1-hour
forecast, zero on the next run (a cache hit drops the cold step to 1.6 s
against WRF's 3.6 s). WRF's cold step is effectively free.

Capacity at 16 GB, same receipt: GPU 15.7-18.5 M cells (about 550-575 squared at
50 levels) at full speed. CPU 24 ranks strict (summed RSS inside 16 GiB) 3.46 M
cells = 263 squared measured at 16.01 GiB and 3.0 s per sim-min; CPU at a
realistic node level about 6 M cells = about 350 squared, estimated (the receipt
marks it as an estimate and the marker travels with it). The asymmetry: CPU
capacity shrinks as cores are added (+9/+25/+61/+130% total footprint at
2/4/8/16 ranks); GPU capacity does not.

Memory-model fits for sizing: WOOF VRAM 0.74-0.84 KiB/cell plus a 2.9-3.4 GiB
intercept; WRF wrf.exe serial 0.88 KiB/cell + 0.26 GiB (Morrison; Thompson
+9.6%, WSM6 -5.0%); WRF 24 ranks 0.98 KiB/cell + 12.7 GiB summed (convex,
interpolate only); WRF real.exe 0.80 KiB/cell + 0.14 GiB, and it allocates all
nests at once. Standing caveats: single-domain, quilting off, desktop reality
takes about 1.5 GB off any RAM ceiling
[receipt:ARWEN-WRF-16GB-REFERENCE-20260803.md].

## 8.2 Radiation and physics cost levers

- Radiation cadence is the largest single lever on nested sub-km chains: the
  pre-2.5.0 per-nest derivation spent 82% of wall clock on radiation on a
  6/2/0.5 km chain; inheritance cut the same case's wall clock from 1435.1 s to
  391.0 s with d01 bit-identical (section 3.9)
  [gallery:radt-subkm-fix-20260817/].
- Legacy RRTMG is the costlier radiation choice: 34.8 wall-s per simulated
  minute where RTE+RRTMGP runs 18.7 on the same three-domain stack (radt
  12/3/1, same card) [docs/public/PHYSICS.md:1040-1048].
- The RRTM+Dudhia 1/1 pair peaks at 1.67 GiB at `column_chunk = 4096` and 53
  layers on an RTX 5090; byte-identical across chunk sizes, so chunk sizing is a
  throughput lever only [docs/public/PHYSICS.md:1089-1099].
- RUC LSM full device residency: 0.47 s per call at 360,000 columns, snow-free
  [docs/public/PHYSICS.md:996-1020].

## 8.3 Launch-to-first-step and fetch

Prep scaling (single domain, GFS, 6 h forcing, wizard defaults, warm caches,
cold download, test host 1): 15 s at 0.14 M cells, 37 s at 4.39 M, 79 s at
12.6 M, 100 s at the largest card-resident 16.8 M, 373 s at a tile-streamed
41.6 M; 24 h forcing at mid size +68%; a 3-domain tree 61 s for 13.3 M cells;
first run on a box +31 s flat [receipt:ARWEN-PRESIM-SCALING-2026-08-16.md].
Parallel fetch is the default: ICON-EU's 252 objects complete in 71.22 s cold
under the pool, and the NOMADS pair stays politeness-capped by design. Those are
the only two measured serial-versus-pooled A/B pairs, and their serial arms live
in the receipt; two further published many-file walls stand against a modelled
serial arm, not a measured one (section 5.2)
[docs/public/receipts/fetch-pool-cold-measured.json; docs/public/DATA.md:409-411].
Decode and static-field build times are in section 6.6.

A 2.5.0-line first-contact receipt exists on the smallest authorized card: on
an RTX 3080 10 GiB Windows/WDDM desktop, the forecast chain reached its first
plot 2 m 45 s after `woof go` launch and rendered 1,155 `rw_wrfbatch` PNGs to
its own "forecast validity" verdict of PASS (health, stability and input-identity
checks in report.json, not a check of forecast accuracy)
[receipt:RELEASE-CANDIDATE-2P5-2026-08-18.md;
receipt:ux-walks-replay/gpu-walk-3080.html]. That walk predates the
memory-gate recalibration and ran the go leg under the gate workaround flag the
old floor forced; the replay at the recalibrated tip completes the same config
bare-default with the gate on (section 8.5)
[receipt:ux-walks-replay/gpu-walk-3080-fixed.html]. The complete cold walk
around the chain (venv, install, CuPy, the one-time 3 m 07 s WPS_GEOG
download, wizard, forecast, render) reached its first plot 7 m 03 s after the
first command. Host memory no longer bounds the largest preparations by the
LENGTH of the forcing window: the mapped engine decodes and writes one valid
time at a time, so the RRFS 7-valid-time prep that peaked near 107 GiB of host
RSS now peaks at 26.9 GiB and rises 0.04 GiB per forcing hour (section 5.7).

## 8.4 Tile streaming and moving nests as capacity tools

Tile streaming trades 1.2-1.4x wall (measured dry at three tile sizes, RTX 4090)
for out-of-core capacity, with per-cell store costs and the no-dry-numbers
governor in section 6.5. Moving nests exist for the same envelope reason: a
low-VRAM card runs a smaller nest that follows the weather at a resolution a
static nest of the same cost could not reach (section 2.5). The DA demo fits a
full six-cycle, ten-member nowcast in 16 GB (727 s on a rented RTX 4080; peak
near 15.9 GB, nearly flat in ensemble size because members advance sequentially)
[docs/da-nowcast-quickstart.md].

## 8.5 The memory gate: one measured envelope

The sizing gates in `woof domain`, `woof check`, and `woof go` price one
formula (`woof/core/preflight.py`): the itemized alloc estimate plus CUDA
context and local-memory backing store, 0.50 GiB unmodelled, 5% of the
estimate per nest beyond the root, and, on Windows only, 20% of the estimate
as WDDM pool slack. The Windows term is measured, not modelled: six whole
bare-default `woof go` forecasts on an RTX 3080 10 GiB Windows 11 WDDM
desktop (60x48 through 240x192 at 12 km, RTE+RRTMGP and legacy-RRTMG suites,
machine-wide nvidia-smi sampling at 0.25 s beside the runtime's own peak
watcher) landed within -0.20 to +0.95 GiB of estimate plus itemized non-pool
residency, the only positive residuals legacy-RRTMG pool retention, worst
+0.30x of the estimate [docs/public/HARDWARE.md;
docs/public/receipts/wddm/rtx3080-wddm-calibration-20260819.json].

Two prior models are retired by that measurement: the footprint x1.75 WDDM
multiplier, taken from one 32 GiB campaign run (it predicted 9.91 GiB for a
walk run whose own contribution measured 2.6 GiB), and the experimental
small-Windows-card tier that stood in for it. Envelope pricing no longer
switches model by card size, so the wizard, `woof check`, and `woof go`
cannot disagree on the same bytes; Linux keeps its previously measured affine
form. The refusals are true refusals now: go's memory refusal keeps the
measured free-VRAM sentence and drops a remedy that recursed into the gate
that refused, and the wizard's no-layout refusal ranks lighter profiles by
priced envelope, so the legacy-RRTMG suites that measured 2.1x heavier than
the default can no longer be advised as lighter [commit 6775450d9;
tests/test_memory_gate_calibration.py]. `woof check` still does not stop a
later run; an observed peak above the WDDM budget is exit 4, not a green exit
with a warning in it [docs/public/HARDWARE.md].

Proven where a user stands: the first-contact walk's 110x88 config, refused at
every grid size by the old floor, passes bare `woof check` (4.73 GiB peak
envelope against a 7.09 GiB budget, 2.36 GiB to spare) and completes
bare-default `woof go` with the gate on; the walk's machine-wide VRAM sampler
peaked at 6930 MiB on the 10 GiB card, desktop share included
[receipt:ux-walks-replay/gpu-walk-3080-fixed.html]. The fix commit's message
carries its own commit-time re-run of the same config with smaller figures
(3.57 GiB priced against a 7.12 GiB budget, 5722 MiB machine peak); the walk
receipt above is the one measured where a user stands [commit 6775450d9].

## 8.6 Platform support

- **Operating systems.** Current released platforms and native-tool availability
  are stated in [README.md](../../README.md); this historical manual does not
  establish that every assembled release ships a Windows native build. The
  older Windows GPU walkthrough used a wheel plus CuPy on an RTX 3080
  (section 8.3), and applies only to that recorded artifact.
- **No GPU.** Everything upstream of the forecast runs without a card: install,
  doctor, fetch, a wizard sizing against a declared budget, and preparation on
  the CPU backend; the forecast loop is CUDA-only. The measured
  command-by-command walkthrough is `docs/public/WITHOUT-A-GPU.md`; this manual
  does not duplicate it.
- **Python.** The package floor is `requires-python >= 3.11` [pyproject.toml:21].
  The full install including the render extra is measured to resolve on Python
  3.10 through 3.14 since wrf-rust 0.2.39 published cp310-cp314 wheels on all
  five platforms; 2.5.0's suites are exercised against 0.2.39, and the runtime
  window still accepts >= 0.2.35 [CHANGELOG.md, engine 2.5.0]. The supported
  window to state is therefore 3.11 to 3.14.
- **GPU.** CUDA via CuPy (`cupy-cuda12x >= 13.0` lower bound
  [docs/public/DETERMINISM.md:104-108]); measured cards in this manual's receipts
  span sm_89 (RTX 4080), sm_120 (RTX 5070 Ti, RTX 5090), plus RTX 4090 and
  RTX 3080 arms on specific studies. Driver-only CUDA 13 boxes install with
  `pip install 'recast-woof[gpu-cu13]'`, toolkit included; measured on a driver-only
  node: bare wheel fails cuBLAS/kernel probes, with `[ctk]` all three pass
  [CHANGELOG.md, engine 2.5.0].
- **Determinism scope.** Bit-reproducibility is scoped to one pinned
  environment; the distribution declares lower bounds and ships no lockfile, so
  pinning is the user's task (section 4.4).
- **Estate checking.** `woof doctor` verifies the installed estate against the
  real binaries and says `untested` where it cannot probe (section 6.7).

## 8.7 What has no published number

So the envelope is not over-read: no full-physics tile-streaming capacity
multiplier exists (section 6.5); no 2.5.0-line throughput reference exists (the
section 8.1 receipt is 1.5.0-era); no effective-resolution number exists at any spacing
other than 500 m (section 7.3); no run receipt exists above nz = 128 (section
2.2); rrfs decode was not re-benched after the parallel-assembly work (section
6.6). Where this manual is silent on a size or speed, no receipt exists.
