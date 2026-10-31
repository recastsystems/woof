# Ian, 07:05Z analysis, 12 hours before landfall

This case branches independent forecasts from a 36-parent GEFS ensemble.
The target is 100 descendants through 2022-09-29 01:05Z (18 forecast hours).
The measured engineering proof is eight descendants through 10:05Z (three
forecast hours), on an RTX 5070 Ti box and an RTX 4090 box. This is the explicitly
requested **NO-DA** proof. It does not establish hurricane forecast skill or
the performance of the private DA-fed version.

After seven hours of GEFS spin-up, the reviewed members are around
965–970 hPa and 47 m/s, versus NHC verification around 943–940 hPa and
64–69 m/s at the same times: roughly 25 hPa too weak, with a 17–22 m/s
wind deficit. Model instantaneous wind and NHC one-minute sustained wind
remain different quantities. This weak spun-up vortex is the measured reason
to require the DA-fed version; DA improvement still needs to be demonstrated.
**Do not spend the eight-card 100×18 h run until the private DA posterior
parents exist.** The sequence below describes the launcher, but that production
run must use the sibling's validated posterior manifest.

`configs/GEFS-ROSTER.json` names every source cycle, member, forcing lead and
configuration SHA-256. `configs/GEFS-ARCHIVE.json` retains the actual acquired
archive-object and composed-pair hashes, URLs and coverage. The 31 current-cycle parents are GEFS 2022-09-28 00Z
c00 and p01–p30. Five lagged parents are 2022-09-27 18Z p01–p05, initialized
from f006 at 00Z. Each spins up independently to 07:05Z. Boundary files reach
03Z on September 29, bracketing the 01:05Z forecast end. The root is
200×160×49 at 12 km; its following child is 480×420×49 at 3 km, dt 15 s.
Fast numerics and no CUDA graphs are used. The child follows its own minimum
smoothed MSLP, with 850 hPa circulation as fallback. Observed tracks never
steer the model.

The authored physics are Morrison (10), YSU (1), surface layer 91, Noah (2),
and RRTMGP radiation (4/4). Coarse-grid cumulus is on (1); fine-grid cumulus
is off. These measurements do not use the plan's Thompson stepping reference.

## Full sequence on an eight-card box

Run this from a checkout of the delivered branch, using its Python environment
and Rust bridges. `python` below means that environment's interpreter. The
geography directory and host-wide GPU mutex must be the ones installed on the
box. Keep the same installed engine and source bytes for parent spinup and
branching: native restart validation checks them.

```bash
cd /work/gpuwm
export PYTHONPATH="$PWD"
export IAN_BASE="$HOME/trailer-ian"
mkdir -p "$IAN_BASE"/{tmp,cache,bin}
export TMPDIR="$IAN_BASE/tmp" TMP="$IAN_BASE/tmp" TEMP="$IAN_BASE/tmp"
export XDG_CACHE_HOME="$IAN_BASE/cache" MPLCONFIGDIR="$IAN_BASE/cache/matplotlib"
export CUPY_CACHE_DIR="$IAN_BASE/cache/cupy" CUDA_CACHE_PATH="$IAN_BASE/cache/cuda"
export PYTHONDONTWRITEBYTECODE=1
export CARGO_HOME="$IAN_BASE/cache/cargo" CARGO_TARGET_DIR="$IAN_BASE/build-target"
export GEOG_ROOT="${GEOG_ROOT:?Set GEOG_ROOT to the installed geography directory}"
export GPU_MUTEX="${GPU_MUTEX:?Set GPU_MUTEX to the existing host-wide GPU mutex}"
(cd tools/rustwx && cargo build --release --locked --offline \
  -p rw-wrfbatch --bin rw_wrfbatch -j "$(nproc)")
export WOOF_RW_WRFBATCH="$CARGO_TARGET_DIR/release/rw_wrfbatch"
rustc --edition 2024 -O cases/ian-20220928/compact_owned.rs \
  -o "$IAN_BASE/bin/compact-owned"
python cases/ian-20220928/author.py \
  --out "$IAN_BASE/case" --geog-root "$GEOG_ROOT"
python cases/ian-20220928/measure.py \
  --out "$IAN_BASE/measure-parents" --mutex "$GPU_MUTEX" \
  --timeout-seconds 86400 -- \
  python cases/ian-20220928/prepare_parents.py \
    --case "$IAN_BASE/case" --out "$IAN_BASE/parents" \
    --geog-root "$GEOG_ROOT" --cards all --compact-owned "$IAN_BASE/bin/compact-owned"
python cases/ian-20220928/seal_pool.py \
  --case "$IAN_BASE/case" --parents "$IAN_BASE/parents" \
  --evidence "$IAN_BASE/evidence"
python cases/ian-20220928/radar.py \
  --valid 2022-09-29T01:05:00Z --out "$IAN_BASE/radar-end"
python - <<'PY'
import json, os
from pathlib import Path
run_base = Path(os.environ["IAN_BASE"])
source = Path('cases/ian-20220928')
request = json.loads((source/'products.json').read_text())
request['best_track'] = str((source/'bal092022.dat.gz').resolve())
request['observed_radar'] = str(run_base/'radar-end/observed/frame.json')
(run_base/'products.json').write_text(json.dumps(request, indent=2)+'\n')
PY
export IAN_POSTERIOR_PARENTS="${IAN_POSTERIOR_PARENTS:?Set the validated private DA posterior manifest before the eight-card run}"
python cases/ian-20220928/measure.py \
  --out "$IAN_BASE/measure-100" --mutex "$GPU_MUTEX" \
  --timeout-seconds 86400 -- \
  python -m woof ensemble --branch-from "$IAN_POSTERIOR_PARENTS" \
    --members 100 --branch-hours 18 --cards all --ensemble-seed 73 \
    --cyclone-products "$IAN_BASE/products.json" --outdir "$IAN_BASE/ensemble-100"
```

All generated files, build targets and caches use the data-disk lane directory.
The existing host-wide mutex is reused so other lanes still coordinate.
The GEFS preparation above reproduces the measured NO-DA 07:05Z pool.
The sibling's separate 05:05Z priors and private eight-window cycle must produce
the validated posterior manifest before the final command can run.

On exactly eight visible cards the forecast takes **13 waves**: twelve waves
of eight and a final four, with completion-driven assignment. Each card owns
one long-lived serial worker. Each child restores one complete ancestor,
including its following-domain placement and its own external boundaries.
No mean vortex is constructed. The deterministic allocation gives 28 families
three children and eight families two. Each family's total probability stays
1/36; children have weights 1/108 or 1/72 accordingly. Uniform wind offsets
are seeded engineering perturbations; their hurricane spread is not calibrated.

Preparation reruns skip parents whose saved checkpoint and configuration still
match their receipts. For a failed forecast, repeat the ensemble command with
`--resume-ensemble "$IAN_BASE/ensemble-100/run-<the-original-stamp>"`.
The exact run directory is printed by the launcher. Completed member receipts
and their diagnostic payloads are reused; failures get a new numbered attempt.
The parent roster, member recipe, forecast length and product request must agree.
The measurement output directory must be fresh for each measured invocation.

## Products and interpretation

Weather fields, common-grid regridding, probability reductions and map pixels
use Rust `rw_wrfbatch` through `woof.rustwx`. Matplotlib makes only the two
intensity time-series charts. Output follows `maps/d02/product/valid-day/`.

The track plume and animation use the member's MSLP minimum and show a colored
ancestor key; black is NHC best-track verification. Animation frames also show
each member's actual nest footprint. The ancestor key occupies a separate
figure-space side panel, outside the map. Plotted positions use a 30-minute
centroid of raw fixes within ±15 minutes, with truncated, renormalized endpoint
windows. The receipt retains both `fixes` (raw) and `plotted_fixes`; smoothing
changes only displayed positions, leaving intensity, wind swaths, steering and
native nest footprints unchanged. NHC verification is not smoothed.
Pressure and maximum instantaneous 10 m
wind charts show members, ancestry-weighted mean and full ensemble range.
NHC one-minute sustained winds and model instantaneous winds are different
quantities and are labelled as such.

Hurricane-wind probability is the weighted fraction whose **every-step,
per-member time maximum** reaches 32.92 m/s. CUDA accumulates it on a fixed
Earth grid before relocation; a cell missing coverage at any member's step
remains missing in the full probability. It is not a maximum of instantaneous
ensemble probabilities. The short proof's probabilities are conditional on
its represented ancestors, not an estimate from all 36 parents.

Reflectivity stamps and the observed panel share one Earth frame, valid time,
units and color scale. Verification uses KTBW, KBYX and KAMX Level-II scans:
the latest complete volume per radar ending by the nominal valid time, at most
15 minutes old. Receipts retain actual scan intervals, QC and coverage. Missing
offshore coverage stays missing; it is not clear air. Radar is verification
only in this NO-DA arm.

Coastal products show earliest sampled member onset and ancestry-weighted
duration of onshore 10 m wind component ≥10 m/s. The coast normal follows the
smoothed landmask gradient in physical east/north coordinates. Duration uses
right-endpoint fine-step integration. These are atmospheric forcing indicators,
not surge height or inundation. The landmask stencil also includes inland lake
shores, visible in the proof; it is not an ocean-only coastline mask. No coupled
surge model is demonstrated. Black on the duration map is a valid zero; missing
onset means the threshold was not reached within the measured window.

Captions carry analysis, observation cutoff, actual product issue time and
remaining lead to the 19:05Z landfall reference. These are retrospective runs:
their actual issue dates are in 2026 and remaining historical lead is negative.
The nominal analysis-to-landfall interval is 12 hours. No 2022 operational
availability claim is inferred from a hindcast.

Best-track source: [NHC ATCF archive](https://ftp.nhc.noaa.gov/atcf/archive/2022/bal092022.dat.gz).
The small archive is included for reproducibility. GEFS archive object hashes
and coverage receipts are retained with the case evidence.

## Measured proof and remaining work

The actual proof uses `run_proof.py`, which writes the resolved public
`woof ensemble` command into `command.json` and takes the host mutex before
launching. Each host has one visible card and runs four children serially.
On the 5070 Ti box, `IAN_BASE=<the box's data root>` and the mutex is
`/tmp/gpu-mutex.lock`; on the 4090 box, `IAN_BASE=<the box's data root>` and the
mutex is `<the 4090 box's mutex file>`.

```bash
# the 5070 Ti box; use its installed engine interpreter
python "$IAN_BASE/engine/cases/ian-20220928/run_proof.py" \
  --parents "$IAN_BASE/parents" \
  --parent-ids 20220928T00-p03,20220928T00-p04,20220928T00-p05,20220928T00-p06 \
  --out "$IAN_BASE/proof-four" --mutex /tmp/gpu-mutex.lock --seed 73

# the 4090 box; its interpreter also needs its installed CUDA headers/libraries
python "$IAN_BASE/engine/cases/ian-20220928/run_proof.py" \
  --parents "$IAN_BASE/parents" \
  --parent-ids 20220928T00-p08,20220928T00-p09,20220928T00-p10,20220928T00-p12 \
  --out "$IAN_BASE/proof-four" --mutex <the 4090 box's mutex file> \
  --ledger <the box's gpu-mutex OWNER ledger> --seed 173
```

The two shards are collected with `collect_products.py --ensemble SET1
--ensemble SET2 --relocate OLD1=LOCAL1 --relocate OLD2=LOCAL2 --request
REQUEST --out OUT`. Only owned diagnostic fields move for rendering;
checkpoint and boundary provenance keep their original identities. The eight
represented ancestors are reweighted to 1/8 each for these conditional proof
products. The full target retains all 36 prior weights.

The two-host commands and checkpoint paths in their original receipts record
the placement at execution time. After proof and resume, owned parents move to
the audited central pool; original receipts preserve their historical paths.
Use the final pool's `PARENTS.json` for new work. The desktop evidence includes
the transport receipts, full diagnostic archive and a DA handoff with the live
pool and compatible installed-engine locations.

The measured walls below include real subprocess startup. The two four-child
jobs ran at different times; their sum is not a measured simultaneous two-host
makespan. Native model timing overlaps restoration and setup, so those columns
must not be added together.

| Measured stage | A development machine, RTX 4090 | A development machine, RTX 5070 Ti |
|---|---:|---:|
| Four children × three forecast hours, command wall | 754.685 s | 909.214 s |
| Public launcher wall within that command | 751.722 s | 906.901 s |
| Individual member wall range | 183.117–195.065 s | 223.336–229.952 s |
| Restore/setup to branch initialization, range | 17.238–25.024 s | 15.067–16.490 s |
| First member's parent load | 3.029 s | 2.616 s |
| Public resume, four reused / zero new, command wall | 4.140 s | 2.059 s |
| Sampled process-tree RSS peak | 8.733 GiB | 6.608 GiB |
| Sampled whole-card memory peak | 10,314 MiB | 9,950 MiB |

Each child produced 13 diagnostic times, 07:05–10:05Z at 15-minute intervals,
and 721 every-fine-step wind samples including the restored analysis. Actual
nest relocations occurred. These walls include that work; they do not isolate
its cost. The the 4090 box proof ran 17:52:16–18:04:51Z and the 5070 Ti box ran
18:12:47–18:27:56Z on October 17. Their real resume commands reused all eight
successful children. The retained the 5070 Ti box ensemble receipt is the later resume
receipt: its `previous_invocations[0]` preserves the 906.901 s forecast run.

The final native release build took 52.44 s with 24 Cargo jobs. Case authoring
and cold archive-network acquisition were not isolated as separate timers.

Radar verification at 10:05Z used three complete volumes ending by that time.
Cached archive acquisition took 0.527 s; native decode/grid took 1.008 s across
24 workers. This does not measure a cold network download. The final combined
eight-member product collection and rendering took 1.725 s internally and
2.068 s including command startup. It produced 26 native maps and one two-panel
analysis chart. That short CPU command's sampler missed its RSS; the reported
22,738 MiB whole-card sample includes concurrent GPU work and is not a
renderer memory measurement.

Preparation uses Rust decode/static worker pools and CUDA initialization;
the measured hierarchy/static and child-init host intervals described below
remain filed to their owner. Parent spinup and child forecasts use one GPU
per serial card worker. Checkpoint/header auditing uses eight independent CPU
workers; byte-identical owned-file consolidation uses 24 native workers.

All 36 parents completed their real independent 00–07:05Z spinups. The final
pool audit verified 72 restart files, 36 distinct ancestor checkpoint hashes,
36 distinct sealed forcing heads, the exact GEFS member/cycle/lead roster,
and forcing through September 29 03Z. It took **18.246 s command wall**
(17.139 s internally) with eight audit workers. The actual public 100×18-h
dry plan has IDs 0–99, 100 distinct seeds, 28 families of three and eight
of two, and probability mass 1/36 for every family. This is a manifest/recipe
audit, not an executed 100-member forecast.

| Parent production host | Parents | Fetch + preparation per parent | GPU spinup per parent |
|---|---:|---:|---:|
| A development machine, RTX 5070 Ti | 10 | 91.300–155.764 s | 490.804–514.476 s |
| A development machine, RTX 4090 | 11 | 139.422–212.346 s | 383.469–425.266 s |
| A development machine, RTX 5090, contended | 15 | 133.118–163.928 s | 284.209–358.128 s |

These are measured stage ranges, not cold-start/network benchmarks. c00,
p01 and p02 reuse earlier sealed preparation while their compatible spinups
were remeasured; c00 preparation was retained at 0.1-s precision. p12 prep
wall was reconstructed from log mtimes (approximately ±1 s), excluding a
manual pause. The delivered `PARENT-TIMINGS.csv` and `MEASURED-PARENTS.json`
retain all 36 rows, hashes and qualifications. Parent-production queues
include the 4090 box's final seven in 3,865.752 s, the 5090 box's final three lagged
members in 1,438.084 s and the 5070 Ti box's final two in 1,263.687 s. These ran
overlapping other work and must not be added as a measured makespan.

Final corrected prep queues contain 13 sustained 30-s one-core alerts,
0.960–1.042 CPU cores with GPU utilization 0%. Their owner, raw process
samples and exact commands are retained in `CPU-OWNER.md` and
`FINAL-PREP-ALERTS.json`; this bottleneck remains unresolved outside this
lane. The public preparation-resume check reused the actual twice-moved
p23 parent in 2.054 s with no new preparation or forecast. A local archive
now hashes directly instead of requiring self-SSH.

Only owned spent downloads/spinup histories were retired after successful
checkpointing: 178,427,740,558 logical bytes. Native full-byte verification
and hardlink consolidation separately saved 78,352,454,097 logical bytes;
these are not filesystem free-space measurements. Verified source copies
were retired after transport; final sealed inputs, both restart domains and
native identities remain in the central pool. A further 63 MiB of superseded
local render outputs/executables was cleared; final images, raw diagnostics,
development receipts and build logs remain.

The the 5090 box twelve-parent queue measured 5,738.925 s including preparation,
spinup, consolidation and intermediate archive transfers. It ran under
`<the 5090 box's mutex file>`, but a separate release-test process was observed
sharing that GPU (14,252 MiB at 18:51Z, process started 18:00Z). Its whole-card
peak was 32,130 MiB. This is a contended production receipt, not an isolated
5090 performance measurement. That queue's older sampler omitted threaded
subprocesses, so its CPU/RSS values are invalid and are not used here.

All 27 final pictures were opened and inspected. The eye is offshore southwest
Florida, but modeled minimum pressure is roughly 963–973 hPa against the
interpolated NHC verification around 939–944 hPa. Modeled reflectivity has a
broader, stronger eyewall/shield than the tighter observed ring. These are
observed shortcomings of this NO-DA proof, not evidence of hurricane skill.
The probability ring is conditional on eight ancestors and three hours; it
does not establish the later landfall wind footprint. The original maps showed
cell-scale MSLP-minimum jitter and a misplaced geographic legend; the revised
plots smooth positions and put the legend outside the map. Raw fixes remain
available for inspection. The common map clips most of the large nest footprints.
Gray observed areas lack radar coverage and remain distinct from clear air.

Inspection exposed a native projection defect: full-domain maps built their
forward frame with the native projection but used the presentation projection
for its inverse, leaving valid weather planes blank. The delivered default fix
uses the resolved native frame for both directions, with a regression test.
Final pictures were regenerated with the fixed Rust executable. The visual
receipt records each final PNG SHA-256 and its review finding.

The final evidence receipt records the exact two-host proof commands, per-stage
wall times, GPU mutexes, parent identities, native renderer identity and visual
inspection notes. Parent spinup and forecast use GPUs; decode, static fields,
archive prefetch and rendering use native worker pools. Passive process/GPU
sampling adds no CUDA synchronization. An unlabelled prep interval observed at
roughly one to two CPU cores is filed with the prep/static owner with process
measurements; it is not silently priced as acceptable single-core overhead.

Not measured by an eight-member, three-hour run: the private DA cycle;
100 descendants through landfall and the full 18 hours; full-set posterior
restoration and moving-nest overhead; eight-card end-to-end availability.
The three-hour proof ends nine hours before landfall, so its intensity and
wind products do not demonstrate evolution through landfall.

The private DA sibling supplies posterior parents using the same `PARENTS.json`
shape: parent ID and weight, checkpoint and SHA-256, prepared root and sealed
head SHA-256, experiment configuration and SHA-256, source/member/cycle/lag and
physics provenance. Root and child checkpoints must belong to the same saved
tree. Each posterior must retain its own parent boundary stream and native
restart/source identity. Replace the manifest path in the ensemble command
after those posterior artifacts exist; the public launcher imports no DA code.

## Storage after the disk ruling

At 19:29Z on October 17, 4,173 scratch files were copied to
`$IAN_BASE/runtime` and SHA-256 checked with eight workers
before removing their root copies. The measured root allocation removed was
816,074,752 bytes (778.3 MiB); copy and verification took 1.256 s and the full
relocation took 1.347 s. The former scratch directory is a compatibility symlink
for historical receipt paths. New commands must use the canonical data path.
The compatible no-.git engine now lives at that data path's `engine/`.

After verifying all payloads against the delivered archive and product files,
458,383,360 bytes (437.1 MiB) of duplicate proof diagnostics, pictures and radar
scratch were retired in 0.781 s. Retained runtime allocation is 357,691,392 bytes
(341.1 MiB). The measured root free-space increase was 812,716,032 bytes, a
shared-filesystem observation rather than an attribution of other agents' work.
Both storage receipts accompany the evidence. The existing 217G parent pool
remains on its original data disk with unchanged native identities and manifest;
the delivery remains on the data disk.

No new forecast was needed for storage relocation. The required eight-child
proof and every final picture were already measured and reviewed. The DA cycle,
full 100×18 h run and full posterior/move overhead remain unmeasured.

## Render correction measured on the existing proof

The October 17 19:41Z product issue uses the same archived eight-member
diagnostics, without a new forecast. The track plume and all 13 animation
frames now place ancestor keys in a Rust-composed figure side panel. Displayed
positions use a 30-minute geographic centroid (±15 minutes, truncated at the
ends); the receipt stores both `fixes` and `plotted_fixes`. All 104 raw fixes,
member identities, pressure/wind values and NHC verification fixes compare
exactly with the preceding delivery. This changes plotted positions only.

All 27 regenerated pictures were opened and inspected. The misplaced keys are
gone from the geography, and the centroid reduces the cell-scale zigzags while
retaining the ancestor spread. The weak vortex and broad modeled reflectivity
ring remain visible. This rendering correction supplies no DA or skill result.

| Measured stage | Wall |
| --- | ---: |
| Eight-worker release-cache staging onto the data disk | 0.922 s |
| Restore the existing proof archive | 0.541 s |
| Three native product regression tests, including compilation | 75.048 s |
| Native release executable build, 24 jobs | 52.770 s |
| All eight-member products, internal collection/rendering | 2.130 s |
| Measured render command, including sampler completion | 4.122 s |

The three native tests passed. Passive two-second CPU/process-tree and GPU
sampling recorded no sustained one-core alerts in these build/test/render
commands. These short stages do not establish an all-core speedup; whole-card
GPU memory belongs to concurrent jobs, not this CPU renderer. The thirteen
previously filed prep alerts remain unresolved with the prepared-tree/static
owner. Commands, samples and raw-fix equality receipt accompany the evidence.

After checking all delivered products and the final native executable by
SHA-256, the spent build target, restored diagnostics and duplicate radar/maps
were retired: 2,466,615,296 allocated bytes in 0.170 s. The sealed raw proof
archive, final pictures and compatible engine remain available. Request paths
in the measured render receipt describe the restored workspace at run time;
those temporary copies were deliberately retired after verification.

Do not launch the eight-card 100×18 h forecast until the private sibling has
produced validated posterior parents. The seven-hour GEFS spin-up's
965–970 hPa / approximately 47 m/s versus NHC 943–940 hPa / 64–69 m/s is
the measured reason for that DA-fed continuation. The DA cycle, its effect on
this deficit, and full-set posterior restoration/move overhead remain unmeasured.
