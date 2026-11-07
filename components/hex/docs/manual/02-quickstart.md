# 2. Quickstart

This chapter goes from nothing to rendered forecast products. Every command
in it was run, in this order, on a fresh install before it was printed here:
the door commands against the installed 0.2.0 wheel, the forecast against the
repository checkout, the GPU steps on an RTX 5070 Ti. Where output is shown
it is that run's output, except where a block is marked as a *form*: the
shape of a line rather than a captured one. The card transcripts in 2.6 come
from a second machine, a 10 GiB RTX 3080; their `ADMISSION` lines are
re-derived on the model the door decides with today, from that card's own
driver reading, because
the model the original run printed has since been retired (6.2).

The true shape of the journey: the wheel installs in one command; the
doors then need three estates the wheel cannot carry (a CUDA-matched CuPy,
the Rust engines, the mesh assets), and `woof hex doctor` walks you through
closing each one by name.

## 2.1 Install

```sh
pip install recast-woof
```

That installs the Python package, its dependencies (numpy, netCDF4, scipy)
and the `woof` engine distribution at a version this port has measured.

**No engine constraint of your own is needed, and that is new.** Until
2026-08-27 this page told you to type an engine version beside the package,
because the declared dependency was `gpuwm>=2.5.5` with no ceiling: a bare
install took the newest published engine, and the forecast lane pins sixteen
individual woof source files by SHA-256. A bare install reached an engine the
pin refused, and the forecast door refused at launch with a digest mismatch
that named neither the version that works nor the fact that pip chose it.

woof hex 0.4.14 declares `gpuwm>=2.9.12,<2.10`, inside the 2.9 minor, so
installing it never downgrades an engine you already have. An engine inside
that range whose radiation helper set this release has not measured is refused
at the forecast door by name, with the version to install. The range is
derived from the measured table in `woof.hex.engine_pin`, not typed (README,
*The engine pin*). The next minor is excluded until it is measured.
`woof hex doctor` checks installed bytes and names a differing engine. The
run transcripts below are dated evidence from earlier engines.

Then add the CUDA lane:

```sh
pip install "recast-woof[gpu]"     # or, spelled out, "recast-woof[gpu-cu12]"
```

**A CUDA 12 or a CUDA 13 driver runs this port.** `[gpu]` installs
`cupy-cuda12x`, which runs on either, because a driver runs its own CUDA
major and every older one. On a CUDA 13 driver you may instead install
`"recast-woof[gpu-cu13]"` (`cupy-cuda13x`, with NVRTC 13.5.37 at most so a
prebuilt kernel pack matches it). `woof hex doctor` warns when the loaded
NVRTC matches no pinned pack. Do not install `[gpu-cu13]` on a
CUDA 12 driver: it imports, and then every device call stops with
`cudaErrorInsufficientDriver`.

Check the driver:

```sh
nvidia-smi        # the CUDA version in the header is your driver's major
```

(On recent drivers that header field is spelled `CUDA UMD Version`, driver
610.74 reads `CUDA UMD Version: 13.3`. A driver below CUDA 12 cannot open this
port's CUDA lane at all, and no pip command changes that; `woof hex doctor`
reads the driver itself, names the extra for it, and says so rather than
offering an install that cannot work. A compute capability 12.0 card, such as
an RTX PRO 6000 or an RTX 50-series card, needs CUDA 12.8 or newer; both
extras install that.)

Confirm what landed:

```sh
woof hex version
```

```json
{
  "distribution": "woof hex",
  "package": ".../site-packages/hexcore",
  "source_checkout": null,
  "version": "0.2.0"
}
```

Note `"package": .../hexcore`, the import namespace is `woof.hex`, not
`gpuwm_hex`. It was renamed from `mpas_port` at 0.2.0 (README, "The import
namespace"); everything you *type* is `woof hex`.

## 2.2 Doctor first

```sh
woof hex doctor
```

On a fresh install with the CUDA extra, the report looks like this:

```
INFO    distribution: woof hex 0.2.0
INFO    interpreter: Python 3.14.4 on Linux x86_64
OK      numpy (arrays; 48 modules import it at line one): imported, 2.5.2
OK      netCDF4 (reads and writes every mesh, static and history file): imported, 1.7.4
OK      scipy (the regridder's spatial index): imported, 1.18.1
PRESENT woof (the physics seam): gpuwm 2.5.2 is installed
INFO    woof git checkout (the forecast lane only): ...
OK      cupy (the CUDA lane): imported, 14.2.0
INFO    woof fetch-bridges coverage: the woof installed here bundles no rw_mpas_convert, rw_mpas_init, ...
MISSING rw_mpas_init (the initial-condition builder): not found on any rung. ...
          cargo build --release --locked --offline -p rw-mpas --bin rw_mpas_init
MISSING rw_mpas_convert (the history converter): not found on any rung. ...
          cargo build --release --locked --offline -p rw-mpas --bin rw_mpas_convert
OK      rw_wrfbatch (the product renderer): found via woof's bridge directories: ~/.woof/bridges/rw_wrfbatch
INFO    mesh grid + static pair: external assets. ...

2 required item(s) missing: rw_mpas_init, rw_mpas_convert.
Run `woof hex doctor --explain` for the full remedy for each.
```

Doctor exits 1 while any required item is missing, so it works as a gate in
a script. `woof hex doctor --explain` prints the whole pasteable remedy
block for each finding; `woof hex doctor --json` emits the same findings as
data. Run doctor after every step below until it says `Every check passed.`

## 2.3 Close the gaps doctor names

### The Rust engines

The doors are orchestration only, all field data is handled by Rust
binaries built from the `woof` source tree. Two roads:

**The staged road.** `woof` publishes prebuilt bundles:

```sh
woof fetch-bridges
```

This stages verified binaries into `~/.woof/bridges`, which woof hex reads
directly (the run that produced the transcript above staged 59 files, every
one verified against its packaged pin). What it can supply depends on the
woof you have: the transcript above was taken against the published woof
2.5.2 wheel, whose bundle carries `rw_wrfbatch` and **not** `rw_mpas_init`
or `rw_mpas_convert`, which is exactly the shortfall the doctor reports
there. The four MPAS bridge binaries enter the bundle at gpuwm 2.5.3, and
this distribution's declared range is `gpuwm>=2.9.12,<2.10` (the floor is
far past 2.5.3, for the seam bytes the port pins): a conforming install cannot
land on the engine that transcript was taken from. Doctor
asks the woof you actually have what its bundle declares, so it never sends
you to a staging command that cannot deliver the file.

**The build road.** From a `woof` source checkout, at `tools/rustwx`:

```sh
cd <gpuwm-checkout>/tools/rustwx
cargo build --release --offline --locked -p rw-mpas       # rw_mpas_init, rw_mpas_convert, rw_mpas_mesh, rw_mpas_static
cargo build --release --offline --locked -p rw-wrfbatch   # rw_wrfbatch
```

(Both commands were run for this manual; they finished in 36 s and 34 s on a
warm target directory.) Then point the doors at what you built:

```sh
export WOOF_HEX_RW_MPAS_INIT=<gpuwm-checkout>/tools/rustwx/target/release/rw_mpas_init
export WOOF_HEX_RW_MPAS_CONVERT=<gpuwm-checkout>/tools/rustwx/target/release/rw_mpas_convert
export WOOF_HEX_RW_WRFBATCH=<gpuwm-checkout>/tools/rustwx/target/release/rw_wrfbatch
```

A flag or environment variable naming a missing file is a hard error, never
a silent fall-through to the next rung of the search ladder (chapter 9 has
the full ladder). When everything is closed:

```sh
woof hex doctor
```

```
...
OK      rw_mpas_init (the initial-condition builder): found via $WOOF_HEX_RW_MPAS_INIT: .../rw_mpas_init
OK      rw_mpas_convert (the history converter): found via $WOOF_HEX_RW_MPAS_CONVERT: .../rw_mpas_convert
OK      rw_wrfbatch (the product renderer): found via woof's bridge directories: ~/.woof/bridges/rw_wrfbatch
...
Every check passed.
```

## 2.4 Obtain the assets, stated plainly

**There is no fetch path in woof hex for any of these.** No
`woof hex fetch` command exists. If a document ever implies otherwise, it
is wrong. You need four things:

1. **A mesh grid file** (`x1.40962.grid.nc`, `x4.163842.grid.nc`, …), from
   the MPAS-Atmosphere project's published mesh downloads, from the
   `MPAS-Tools` generator, or from the engine's own `woof mesh` door
   (chapter 4). This quickstart uses the published 40,962-cell global mesh
   `x1.40962`: the one a 10 GiB card holds.
2. **The matching static file** (`x1.40962.static.nc`), carrying terrain,
   land use, soil and vegetation for that exact mesh. The published meshes
   have published statics; a generated mesh gets its static from the same
   `woof mesh` run. Grid and static must be the same mesh: the doors
   cross-check and refuse a mismatched pair by name.
3. **For the init door: a vertical-grid declaration.** The normal path is
   a `--vertical-spec` JSON (`gpuwm-hex.vertical-spec/v1`), from which the
   door constructs the vertical grid itself: no native toolchain anywhere
   in the loop. The compatibility mode reads the vertical out of a
   native-minted init-class file (`--capsule`); chapter 5 explains both
   and when to prefer which.

   **The example specs are not in the wheel.** They live at
   `components/hex/verification/vertical-specs/` in the repository, and
   `pip install` places a wheel, so the path 2.5 prints
   below does not exist on a fresh install: the door refuses with the
   absolute path it could not find, which is correct and is not a hint about
   where to get one. Take them from the repository and put them where
   2.5 expects.
4. **Meteorological input**: a WPS intermediate file from `ungrib` (GFS,
   ERA5, whatever you drive with), valid at your start time.

For 1 and 2, on this mesh, the public download and the registered row are the
same bytes: verified 2026-08-24 by fetching both archives fresh and hashing
what came out:

| file | from | bytes | SHA-256 |
| --- | --- | --- | --- |
| `x1.40962.grid.nc` | `x1.40962.tar.gz` | 56,039,332 | `9a9e1909a755dac209462ceb0bfffd77ac1b37503169568b7f296707ee612bb9` |
| `x1.40962.static.nc` | `x1.40962_static.tar.gz` | 94,766,584 | `cf1a47d4168327f06a8403555d6ed8b2fe1aff7f8b916bb7f6a754c34a10ac82` |

Both match `src/hexcore/drivers/mpas_mesh_binding.py` exactly, so the forecast door's byte
check passes on what the published downloads give you. That is a fact about
those archives, not a fetch path: woof hex still fetches nothing.

Sanity-check the mesh pair before anything expensive touches it:

```sh
woof hex mesh-check --grid assets/x1.40962.grid.nc --static assets/x1.40962.static.nc
```

```json
{
  "connectivity_indexing": "zero-based with -1 padding",
  "dimensions": { "maxEdges": 10, "nCells": 40962, "nEdges": 122880, "nVertices": 81920 },
  "grid": ".../assets/x1.40962.grid.nc",
  "grid_sha256": "9a9e1909a755dac209462ceb0bfffd77ac1b37503169568b7f296707ee612bb9",
  "passed": true,
  "static": ".../assets/x1.40962.static.nc",
  "static_sha256": "cf1a47d4168327f06a8403555d6ed8b2fe1aff7f8b916bb7f6a754c34a10ac82"
}
```

## 2.5 First initial conditions

Build an init from your WPS intermediate file. Every physics switch is
required and has no default, each one changes the numbers in a file that
would open cleanly either way, so a default would be a silent wrong answer.
Each refusal prints the native namelist key it corresponds to (chapter 5
maps them all).

```sh
woof hex init \
  --met     <path-to-WPS-intermediate> \
  --grid    assets/x1.40962.grid.nc \
  --static  assets/x1.40962.static.nc \
  --vertical-spec verification/vertical-specs/tc55-v1.json \
  --out     work/x1.40962.init.nc \
  --start-time 2026-08-12_06:00:00 \
  --nfglevels 34 --nfgsoillevels 4 \
  --extrap-airtemp lapse-rate --use-spechumd no \
  --theta-adv-order 3 --coef-3rd-order 0.25 \
  --virtual-factor reproduce-fortran \
  --deep-soil-moisture reproduce-fortran \
  --landuse-table MODIFIED_IGBP_MODIS_NOAH \
  --frac-seaice yes --tsk-seaice-threshold 100.0 \
  --oned-underflow preserve
```

This is the native-free mint: `--vertical-spec` names a versioned JSON
vertical declaration and no native artifact appears anywhere in the
lineage (the receipt records `native_runtime_dependency: false`). The
compatibility mode passes `--capsule` + `--reference` naming a
native-minted init-class file instead; chapter 5.1 has both modes and
their receipts. `--nfglevels` must cover the levels actually in your met
file; declare too few and the door refuses with the real count (the
proving run's met file, a GFS intermediate, carried 34). The run above
wrote a 385 MB `x1.40962.init.nc` plus
`x1.40962.init.nc.provenance.json`: the SHA-256 of every input, the
engine binary, the argv, the engine's own receipt, and the output. The
engine step takes about 2 s; the first mint for a mesh spends minutes in
the door's geometry solve, and the keyed cache brings a re-mint down to
about a minute.

## 2.6 Ask the card first

The forecast lane is the one step whose answer depends on your hardware, so
ask before you spend anything on it. It runs on what you installed: the
drivers ship in the wheel and the physics seam is the installed `woof`
(2.1), so no checkout of either project is needed.

```sh
woof hex forecast --preflight \
  --mesh    x1.40962 \
  --grid    assets/x1.40962.grid.nc \
  --static  assets/x1.40962.static.nc \
  --init    work/x1.40962.init.nc \
  --init-source "GFS 2026-08-12 06Z" \
  --hours 1.0 --history-every-minutes 30 \
  --out work/fc-01
```

Through 0.3.1 this command also needed `--repo <gpuwm-hex-checkout>/tree`
(the drivers were not in the wheel) and `--gpuwm-checkout` (a woof git
clone, only so receipts could name a commit). Both flags are still
accepted: `--repo` is reported and ignored, and a clone passed as
`--gpuwm-checkout` is recorded by commit instead of by the install's
version and `RECORD` digest.

On a card that holds the mesh, preflight binds it, admits it, runs the
driver's own source-pin and host checks, and exits 0. This transcript is
from a 10 GiB RTX 3080 with a desktop session on it, run 2026-08-25; the `ADMISSION` and
`ARCHITECTURE` lines are the ones the door prints for that card and that
free reading today:

```
[mesh-binding] bound x1.40962: nCells=40962, nEdges=122880, dx=120000.0 m, dt=120.0 s
BIND mesh=x1.40962 rebound=True dt=120.0 s
ADMISSION mesh=x1.40962 cells=40,962 card=68 SM row=measured global predicted=5,682.0 MiB margin=884.0 MiB free=9,097.0 MiB of 10,239.5 MiB -> admitted
ARCHITECTURE sm=sm_86 -> admitted, anchored: the 2026-08-25 anchor (evidence/sm86-tier-20260825/RECEIPT.md, evidence pin 75437c663c9894dc...)
PREFLIGHT mesh=x1.40962 problems=0 status=preflight_passed
```

The card is read at the moment of the decision (`card=68 SM`, and the row
selected for it) so a 10 GiB desktop is priced as itself. On this free
reading it admits `x1.40962`, `v15.150.38857` and `u96.64002`; the refusals
earlier releases printed for those three were a 32 GiB card's arithmetic
quoted at a 10 GiB part, and 6.2 has the whole story.

On a request the card really does not hold, it says so with numbers. The
same card and the same free reading, asked for the 163,842-cell
`x4.163842`:

```
ADMISSION mesh=x4.163842 cells=163,842 card=68 SM row=measured global predicted=17,000.2 MiB margin=884.0 MiB free=9,097.0 MiB of 10,239.5 MiB -> REFUSED
PREFLIGHT REFUSED device memory admission refused --mesh x4.163842: the fitted
footprint for 163,842 cells is 17,000.2 MiB and the decision holds back 884.0
MiB, so it needs 17,884.2 MiB free; this device reports 9,097.0 MiB free of
10,239.5 MiB, short by 8,787.2 MiB. ... This card fits the registered mesh(es)
conus-x1.2971, r4.75.11020, ... u96.64002, v15.150.38857, x1.40962 at this
moment (68,440 cells fit); re-run --mesh with one of them, or free the memory
the shortfall names and re-run.
```

Preflight exits 1 on a refusal and touches no CUDA beyond the memory and
device-properties queries; a missing input is *reported* alongside the card
verdict rather than hiding it. Chapter 6.2 explains the model it decides
with.

What the run checks before CUDA is touched, stated here rather than
discovered mid-launch: the sixteen pinned `woof` seam files are verified by
SHA-256 against the pin and a mismatch is refused naming the file and both
digests, and the drivers verify their own executing modules the same way.
Both read the installed copies, which are the copies that execute.

## 2.7 First forecast

Same command without `--preflight`:

```sh
woof hex forecast \
  --mesh    x1.40962 \
  --grid    assets/x1.40962.grid.nc \
  --static  assets/x1.40962.static.nc \
  --init    work/x1.40962.init.nc \
  --init-source "GFS 2026-08-12 06Z" \
  --start-time 2026-08-12_06:00:00 \
  --hours 1.0 --history-every-minutes 30 \
  --out work/fc-01 \
  --case-label quickstart
```

The same integration, from the init built in 2.5, took **97.8 s of
integration and 146 s wall on a 16 GiB RTX 5070 Ti**: 1 hour, 30 steps at
dt = 120 s, full physics, three history files, exit 0. That measurement was
taken through `tools/run_cuda_v841_forecast_mesh.py`, the driver this door
now drives; the door adds the admission decision in front of it and the
receipt and render command behind it, and changes nothing in between.

Re-run on 2026-08-27 on the 10 GiB RTX 3080 in a desktop, from the installed
wheel and a published `woof` 2.5.6 clone: **165.4 s of integration and 244 s
wall**, same 30/30 steps, same three frames, `status=passed`.

When it finishes, `--out` holds three history files (analysis, +30 min,
+60 min), `cuda-v841-forecast-receipt.json` (the driver's receipt, stating
exactly what this run claims and, at equal prominence, what it does not)
and `forecast-receipt.json`, the door's own, carrying the admission
decision, the mesh binding, the driver's receipt embedded whole, and the
render command. The door's last two lines have the form:

```
DOOR mesh=<row> steps=<n> frames=<n> status=passed out=<out>
NEXT woof hex render --history <out>/cuda-history.<valid-time>.nc --mesh <grid> --out <out>/png --simulation-start <start>
```

## 2.8 First rendered products

Spell it out, and add `--window focus`, which the `NEXT` line does not
carry. The door's default window is `mesh`, and the `rw_mpas_convert` in
every published bundle answers `unknown render window 'mesh' (known: focus,
global)`, so the pasted `NEXT` line fails on a published engine (7.1).
`render` needs no checkout and no GPU:

```sh
woof hex render \
  --history work/fc-01/cuda-history.2026-08-12_07.00.00.nc \
  --mesh    assets/x1.40962.grid.nc \
  --out     work/png \
  --simulation-start 2026-08-12_06:00:00 \
  --window  focus \
  --products all
```

```
CONVERT CONVERTED  .../frames/wrfout_d01_2026-08-12_07_00_00  0.053s  107377492 bytes  35 written  3 absent
FRAME 2026-08-12T07:00:00 rendered=41 skipped=0 failed=0
MANIFEST work/png/render-manifest.json
SCRATCH cleared work/png.render-scratch
DOOR rendered=41 skipped=0 failed=0 out=work/png
```

41 product PNGs, filed under `work/png/<domain>/<product>/<valid-day>/`
(for this mesh the focus window is `d01-22km`), with a
`render-manifest.json` carrying engine digests, per-frame results, and the
exact invocations. 3.6 seconds on the proving node. That is the whole loop:
install → doctor → assets → init → forecast → pictures.

Re-run 2026-08-27 on a desktop against the published 2.5.7 bridge bundle:
**49 rendered, 0 skipped, 0 failed, 7 s**, same `d01-22km` tree. The count
moved because the renderer's catalog grew, not because anything here changed.
End to end (empty machine to these pictures, including every command that
failed on the way) **16 min 46 s**, with the mesh pair and the met file
already on disk.

## Where to next

- Something refused? Chapter 8 indexes the refusals you will actually meet.
- Want your own mesh, or to know what fits your card? Chapter 4.
- Want to understand what you just ran before trusting it? Chapter 3.
