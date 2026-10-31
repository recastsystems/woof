# Install and verify RW-WPS

## Without an NVIDIA GPU

The base `pip install recast-woof` works without CuPy. Fetching, CPU preparation
(`woof prep --preprocess-backend cpu`), doctor, rendering existing output,
and extracting soundings do not require an NVIDIA GPU. NumPy setup states
support preparation and export; there is no CPU forecast step.

Forecast allocation and `woof.core.dycore.step` require an NVIDIA GPU and
a driver supporting CUDA 12.x with `pip install 'recast-woof[gpu-cu12]'` (CuPy
>=14.0 and its CUDA toolkit). Use `recast-woof[gpu-cu13]` on a CUDA-13-only
system. VRAM depends on the domain and physics; `woof check` sizes the
configuration. `woof doctor` reports the detected runtime and matching
extra, and explains this requirement when a device is unavailable.

Direct integrations can display the same named error:

```python
from woof.forecast_gpu import ForecastGPURequired, require_forecast_gpu

try:
    require_forecast_gpu()  # before constructing a forecast domain
except ForecastGPURequired as error:
    print(error)  # present this message in the application's interface
```

`DomainState` forecast allocation and `dycore.step` also check this
requirement. Catch `ForecastGPURequired` around either call to handle an
unavailable device. CLI forecast commands print the message and exit 2.
The [CPU walkthrough](public/WITHOUT-A-GPU.md) covers preparing inputs for
a forecast run on another machine.

RW-WPS does not yet publish a community package. There are two useful
installation modes: a developer checkout for inspection/tests and a sealed
Linux or Windows x86-64 runtime archive for real preprocessing. Neither mode
changes the scientific support matrix.

## Developer checkout

### One command

`install.sh` (POSIX) and `install.ps1` (PowerShell) at the repository root
perform the whole developer install: create `.venv` if absent, on
free-threaded CPython 3.14t (see "Interpreter" below), install
the matching data companion with `pip install -e recast-woof-data`, then install
`-e '.[gpu-cu12,render]'` (or `gpu-cu13`) into it, stage the externalized Thompson tables
with `woof fetch-tables` (downloads only what is absent -- ~243 MiB
from a checkout -- SHA-256-verified against the packaged pins before
install; `--no-fetch-tables`/`-NoFetchTables` or
`WOOF_INSTALL_NO_FETCH_TABLES=1` defers it, and `woof fetch-tables
--from DIR` stages them offline), offer to install rustup when `cargo` is
missing (they prompt first; `--yes`/`-Yes` or `WOOF_INSTALL_YES=1`
consents non-interactively, and a non-interactive run without consent
fails with instructions instead of hanging), build the vendored Rust
GRIB bridges (`tools/grib1_bridge`) and the production render engine
(`tools/rustwx`) with `cargo build --release --locked --offline`
(`--no-render`/`-NoRender` or `WOOF_INSTALL_NO_RENDER=1` skips the
renderer build; until a render engine exists `woof render` REFUSES,
naming `woof fetch-bridges`: weather-field product plots come from
`rw_wrfbatch`, and `--engine matplotlib` is a named workaround that
announces itself, not an automatic fallback), build the terminal workspace
in the Zarr reader in `tools/zarr_bridge`, the default
mapped-source decode engine in `tools/rw_wps` and the default velocity
dealiasing library in `tools/region_global_dealias` with the same locked
offline command, build the hydro payload (`tools/nextgen_hydrofabric`),
forcing (`tools/nextgen_forcing`), gauge verification (`tools/nextgen_verify`) and
hydro fabric preparation (`tools/hydro_world`, `tools/hydro_world_params`,
`tools/nextgen_reservoir_prepare`, `tools/hydro_world_init`) and ocean
preparation (`tools/ocean_prepare`, `tools/rw-bathy`), the wind-turbine table decoder (`tools/wind-table`) and hydro output (`tools/nextgen_output`), and the Icechunk store writer (`tools/rw_icechunk`)
workspaces (fetching their locked dependencies first), and finish with `woof doctor`, whose exit status the script propagates.

A pip install on Windows x86-64 or Linux x86-64 needs no toolchain for
either half: the platform wheels for those two carry all of the prebuilt
native tools, the GRIB bridges and the render engine among them. A
pure-Python wheel or an sdist install on those two platforms carries none,
and `woof fetch-bridges` (or `woof setup`, which runs it first) stages
this release's prebuilt bundle under the SHA-256 pins the package
carries. Other platforms have no published bundle and build the native
tools from a source clone, as described below. The renderer's Natural
Earth and US Census map assets arrive with the `recast-woof-data` package every
install pulls, and `woof` hands them
to the renderer, because a renderer without them draws plots with no
coastlines or borders and reports success. A run that finds none says so
in a `render_basemap_missing` warning event. Both scripts are idempotent: re-running reuses the existing
`.venv` and the incremental cargo build. Run them from the checkout
root, or standalone (piped from the raw URL), in which case they clone
https://github.com/recastsystems/woof into `./gpuwm`
(`WOOF_REPO_URL` overrides the clone source).

**Interpreter.** A new `.venv` is made on free-threaded CPython 3.14t:
a `python3.14t` already on PATH, else the one uv finds or installs
(`uv python install 3.14t`), else, with consent (`--yes`/`-Yes` or the
prompt), uv itself from https://astral.sh/uv first. Multi-card
`[devices]` forecasts step each card from its own Python thread, and under
a standard Python those threads take turns on the interpreter lock: four
cards measured 289 s per forecast hour with the lock and 130 s on 3.14t.
The `woof` command line and both prepared runners re-run themselves once
with `PYTHON_GIL=0`, so nothing is exported by hand. On a free-threaded
`.venv` the scripts install with `--prefer-binary` (cftime 1.6.6 has no
cp314t wheel; 1.6.5 does), and the `[render]` extra's `wrf-rust`, which
publishes no cp314t wheel, builds from its sdist with the Rust toolchain
the scripts already require, which is why the Rust check runs before the
`.venv` step. `--python PY`/`-Python PY` (or `WOOF_PYTHON`) picks another
interpreter. When no 3.14t can be had, or a dependency will not install
under it, the install goes on with `python3`/`python` and prints a warning
naming the cost and the remedy; `woof doctor` reports the same. A
`python3.14t` that cannot make a venv (a distribution build without its
venv package has no ensurepip; deadsnakes splits it out as
`python3.14-venv`) gives way to the 3.14t uv provides, then to
`python3`/`python` with that warning, and the half-made `.venv` is removed.
A pip install on 3.14t outside the scripts needs `--prefer-binary` too, or
pip builds cftime 1.6.6 from source with a C compiler. A
re-run keeps an existing `.venv` and warns when it has the lock; remove
`.venv` and re-run to remake it on 3.14t.

### Manual steps

Use Python 3.11 or newer and Rust 1.94 or newer (`rustc --version`). The
terminal and Zarr reader workspaces refuse an older compiler, and a
distribution's packaged Rust can be older; `rustup update stable` brings a
rustup install up to date. An editable developer checkout still installs the
monorepo metadata and its forecast/plot dependencies. The sealed release
builder instead creates a dedicated `rw-wps` wheel containing the source
adapters, initialization/export state, required setup kernels, and HRRR
preprocessing helpers; forecast executors and forecast-only physics data are
absent.

POSIX:

```bash
git clone https://github.com/recastsystems/woof woof && cd woof
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e recast-woof-data
python -m pip install -e '.[gpu-cu12,render]'   # or gpu-cu13
woof fetch-tables
for workspace in tools/grib1_bridge tools/rustwx tools/zarr_bridge tools/rw_wps tools/region_global_dealias tools/nextgen_hydrofabric tools/nextgen_forcing tools/nextgen_verify tools/hydro_world tools/hydro_world_params tools/nextgen_reservoir_prepare tools/hydro_world_init tools/ocean_prepare tools/rw-bathy tools/wind-table tools/nextgen_output tools/rw_icechunk; do
  (cd "$workspace" && cargo fetch --locked && cargo build --release --locked --offline)
done
woof doctor
```

Windows (PowerShell):

```powershell
git clone https://github.com/recastsystems/woof woof; cd woof
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e recast-woof-data
python -m pip install -e '.[gpu-cu12,render]'   # or gpu-cu13
woof fetch-tables
foreach ($workspace in 'tools\grib1_bridge', 'tools\rustwx', 'tools\zarr_bridge', 'tools\rw_wps', 'tools\region_global_dealias', 'tools\nextgen_hydrofabric', 'tools\nextgen_forcing', 'tools\nextgen_verify', 'tools\hydro_world', 'tools\hydro_world_params', 'tools\nextgen_reservoir_prepare', 'tools\hydro_world_init', 'tools\ocean_prepare', 'tools\rw-bathy', 'tools\wind-table', 'tools\nextgen_output', 'tools\rw_icechunk') {
  Push-Location $workspace; cargo fetch --locked; cargo build --release --locked --offline; Pop-Location
}
woof doctor
```

### After a pull, rebuild

`git pull` moves the Rust half of the tree and leaves the binaries you
built before it exactly where they were, so every `target/release` in
the checkout can be a release behind the Python beside it. Rebuild
after every pull.

The one step that covers all of them is the installer. Both scripts are
idempotent and the cargo builds are incremental, so a pull that touched
no Rust costs seconds:

```bash
./install.sh          # Windows (PowerShell): .\install.ps1
```

They build nine cargo workspaces, one for every compiled artifact a
bundle carries, and the gate below judges each one it finds built:

| Workspace | What it builds |
|---|---|
| `tools/grib1_bridge` | the GRIB decoders and the CPU preprocessing library |
| `tools/rustwx` | the render engine, `rw_netcdf`, and the rest of the Rust data path |
| `tools/zarr_bridge` | `rw_zarr` |
| `tools/rw_wps` | `gpuwm_mapped_engine`, the default mapped-source decode engine |
| `tools/region_global_dealias` | the default velocity dealiasing library |
| `tools/nextgen_hydrofabric` | hydro payload transport and source preparation |
| `tools/nextgen_forcing` | external hydro forcing preparation |
| `tools/nextgen_verify` | gauge verification |
| `tools/hydro_world` | `rw_hydro_world`, the hydro door's fabric builder (`--hydro on`) |
| `tools/hydro_world_params` | `rw_hydro_params`, hydro catchment parameters |
| `tools/nextgen_reservoir_prepare` | `nextgen-reservoir-prepare`, hydro reservoirs |
| `tools/hydro_world_init` | `rw_hydro_init`, hydro initial water state |
| `tools/ocean_prepare` | `rw_ocean`, ocean bathymetry and atmospheric forcing preparation |
| `tools/rw-bathy` | `rw_bathy`, storm-surge bathymetry ingest and products |
| `tools/wind-table` | `rw-wind-table`, wind-turbine inventories and WRF Fitch tables (`--wind-farms`) |
| `tools/nextgen_output` | `rw_nextgen_output`, coupled hydro history, restart and products |
| `tools/rw_icechunk` | `rw_icechunk`, the optional Icechunk output store writer |

By hand, POSIX:

```bash
for workspace in tools/grib1_bridge tools/rustwx tools/zarr_bridge tools/rw_wps tools/region_global_dealias tools/nextgen_hydrofabric tools/nextgen_forcing tools/nextgen_verify tools/hydro_world tools/hydro_world_params tools/nextgen_reservoir_prepare tools/hydro_world_init tools/ocean_prepare tools/rw-bathy tools/wind-table tools/nextgen_output tools/rw_icechunk; do
  (cd "$workspace" && cargo fetch --locked && cargo build --release --locked --offline)
done
```

Windows (PowerShell):

```powershell
foreach ($workspace in 'tools\grib1_bridge', 'tools\rustwx', 'tools\zarr_bridge', 'tools\rw_wps', 'tools\region_global_dealias', 'tools\nextgen_hydrofabric', 'tools\nextgen_forcing', 'tools\nextgen_verify', 'tools\hydro_world', 'tools\hydro_world_params', 'tools\nextgen_reservoir_prepare', 'tools\hydro_world_init', 'tools\ocean_prepare', 'tools\rw-bathy', 'tools\wind-table', 'tools\nextgen_output', 'tools\rw_icechunk') {
  Push-Location $workspace; cargo fetch --locked; cargo build --release --locked --offline; Pop-Location
}
```

From 2.7.6 nothing depends on remembering. A checkout's own build that
is older than the sources that build it is refused before the run
starts, naming the binary, when it was built, the source revision it
was built from, the file that moved past it and the exact command for
the workspace that builds it. `woof doctor` reports the same line.
The sources are the files cargo recorded for that binary in the `.d`
file it writes beside it, plus the crate manifests and the lock, so a
change to a file only another binary of the same crate reads does not
condemn this one.

The judgement is about the file, not about how it was named. A wheel or
bundle install has no sources here to be compared with and is untouched,
and so is a binary you built outside this checkout and pointed an
environment variable at. A variable pointing into this checkout's own
`target/release` or `target/debug` names this checkout's build, and that
is measured like any other.

**Which GPU extra.** CuPy ships one wheel per CUDA major, and a pip
extra cannot detect the major of the box it is installing on -- there is
no environment marker for it. So the extra names the major, and you pick
the one that matches:

| Your box | Extra | Wheel |
|---|---|---|
| CUDA 12.x | `[gpu-cu12]`, or `[all-cu12]` with the renderer | `cupy-cuda12x[ctk]` |
| CUDA 13, no 12.x runtime libraries | `[gpu-cu13]`, or `[all-cu13]` | `cupy-cuda13x[ctk]` |

**Why `[ctk]`, and how to skip it.** A CuPy wheel carries NVRTC -- the
compiler -- and no CUDA headers, while CuPy's own kernels include the
CUDA runtime headers. On a box that has a driver and no toolkit (stock
Ubuntu with the archive's NVIDIA driver; almost every fresh rental) the
wheel installs, `import cupy` succeeds, and the first device call of any
kind dies with `Failed to find CUDA headers`. Measured on a driver-only
CUDA 13 node, 2026-08-17:

| Install | Venv | cuBLAS / kernel / reduction |
|---|---|---|
| `pip install cupy-cuda13x` | 274 MiB | all three fail on missing headers |
| `pip install 'cupy-cuda13x[ctk]'` | 1867 MiB | all three ok |

So the GPU extras ask for `[ctk]`, which is CuPy's own extra for a
matching toolkit delivered as wheels (runtime + headers, NVRTC, cuBLAS,
cuFFT, cuRAND, cuSOLVER, cuSPARSE): about 1.6 GiB, and it is the
difference between a GPU box that runs and one that cannot execute a
kernel. The floor moved to CuPy 14.0 with it, because `[ctk]` is a
CuPy 14 extra and pip only *warns* when a resolved version does not
provide a requested extra.

If this box already has a matching CUDA toolkit, install the two pieces
separately and skip the 1.6 GiB:

```bash
pip install recast-woof && pip install cupy-cuda13x    # or cupy-cuda12x
```

That is the smaller install, not the safer one: CuPy resolves the wheels
ahead of a system toolkit, so the `[ctk]` set is also the self-consistent
one. `woof doctor` judges whichever you chose by compiling a kernel
from a cold cache and loading cuBLAS, and when the headers are the gap it
prints `pip install 'cupy-cuda13x[ctk]'` -- the command that closes it --
rather than a wheel reinstall that cannot.

`nvidia-smi` prints the CUDA version in its header. Getting it wrong is
quiet until it is not: the cu12 wheel on a CUDA-13-only box imports
cleanly, compiles kernels, passes an import probe, and then fails at the
first cuBLAS load -- which on a real run is the first matmul, hours in.
`woof doctor` reads the major straight off the driver, with or without
CuPy installed, and names the extra that matches; `install.sh` /
`install.ps1` do the same detection and take `--cuda 12|13` /
`-Cuda 12|13` to override it.

`[gpu]` and `[all]` are kept and still mean cu12, so every install that
already names them keeps working; they are aliases for the cu12 pair,
not a separate pin.

`[render]` = `wrf-rust` in its certified window (`>=0.2.39,<0.3` -- a
window, not an equality, so installing woof never downgrades a newer core
you already run; 2.5.0's suites are exercised against 0.2.39) + matplotlib
+ the demo gallery's shapefile reader for `woof render`; add `[dev]` for
the test suite:

**Python 3.10 through 3.14: this extra installs everywhere, and the floor
is why.** wrf-rust 0.2.39 publishes cp310-cp314 wheels on all five
platforms (macOS x86_64 and arm64, manylinux x86_64 and aarch64,
win_amd64). 0.2.38 and earlier stopped at cp313, so on a 3.14 box pip
fell back to their sdist, that sdist did not build (`the configured
Python interpreter version (3.14) is newer than PyO3's maximum supported
version (3.13)`), and because pip fails a whole resolution when one
requirement fails, `pip install 'recast-woof[gpu-cu13,render]'` installed **no
woof at all** there. The floor is the remedy: every supported
interpreter resolves a real wheel, and no environment marker silently
skips the package on any of them. If you pin an older core yourself on an
interpreter it has no wheel for, `woof doctor` reports that gap by name
rather than printing a pip line that cannot work -- the runtime window
still accepts `>=0.2.35`, so a 0.2.38 already on your box keeps
rendering.

The free-threaded build (`python3.14t`, which `install.sh` and
`install.ps1` use by default) is the exception: 0.2.39 publishes no cp314t
wheel, so pip builds it from its sdist, which needs `cargo` on PATH. Its
PyO3 supports 3.14t: the build takes about 20 s on a 24-core box and the
module imports without switching the interpreter lock back on (measured
2026-10-08).

```bash
python -m pip install -e '.[dev]'
python -m pytest -q \
  tests/test_source_adapters.py \
  tests/test_namelist_compat.py \
  tests/test_native_wrf_distribution.py
```

**Tables stage on first use; some installs need the native bundle.** A
platform wheel (Windows x86-64, Linux x86-64) already contains all of
the compiled Rust tools and needs no `woof fetch-bridges`. A
pure-Python wheel or an sdist install contains none, so until
`woof fetch-bridges` stages them, `woof check`/`run` (ERA5 route) and
the `rw-wps` GFS/HRRR front doors have nothing to decode GRIB with.
The first Thompson forecast stages its missing tables automatically, and
`woof doctor` says which assets are still missing:

```bash
pip install 'recast-woof[all-cu12]'   # or 'recast-woof[all-cu13]' on a CUDA-13-only box
woof fetch-bridges   # pure-Python wheel or sdist only
woof fetch-tables   # optional: stage Thompson tables ahead of the first run
woof doctor
```

`woof fetch-bridges` downloads one release bundle for this platform --
the five GRIB decoders, the CPU preprocessing library, the `rw_fetch`
backbone, the `rw_wrfbatch` renderer and the `gpuwm_mapped_engine`
decode engine every mapped source runs on -- verifies every artifact
against the exact size + SHA-256 pins packaged in the wheel, and stages
them into `~/.woof/bridges`, the directory the resolver already
searches.  It is re-run safe (present, pin-valid artifacts are left
alone), an interrupted download resumes or restarts, and `--from DIR`
stages the same bundle -- or the loose artifacts -- from a local
directory with identical verification, for air-gapped installs.
`--dest DIR` stages elsewhere; `WOOF_BRIDGE_ASSET_URL_BASE` points it
at a mirror. Two setups aimed at one folder (two terminals, or a run's
automatic refresh beside a manual `woof fetch-bridges`) take turns: the
second waits for the first to finish, then finds the bundle staged and
fetches nothing. A folder you cannot write is only checked: a complete
estate there is reported as verified, and an incomplete one is refused
with the reason.

The release workflow generates those pins before it builds the supported
PyPI wheel and sdist. The GitHub-generated source `.zip` and `.tar.gz`
archives deliberately retain an unpinned document, because their bytes were
not assembled with the target-native bundles. Consequently,
`woof fetch-bridges` from an automatic source archive refuses with the
source-build remedy instead of borrowing another release's hashes. Use the
PyPI wheel/sdist for prebuilt bundles, or clone the exact tag and build the
vendored Rust workspaces for a source installation.

Bundles are published for Windows x86-64 and Linux x86-64.  On any
other platform, and on any release that published no bundle, the
command says so by name and the route is the same one-time
`cargo build --release --locked --offline` from a source clone: point
the install at the built executables with the per-artifact environment
variables (`WOOF_GRIB1_BRIDGE`, `WOOF_GFS_GRIB2_BRIDGE`, ...) or copy
them into `~/.woof/bridges/`.  `woof doctor` prints whichever of the
two remedies is true for the machine it runs on.

The wheel and sdist also exclude the two externalized Thompson tables
(freezeH2O.dat, 243 MiB, and qr_acr_qg_V4.dat, 71 MiB -- together they
would put the artifacts over PyPI's per-file limit). The first
`woof go` run selecting Thompson downloads and checks the two tables
(about 315 MiB) from the pinned release assets into
`~/.woof/tables/thompson`. `woof fetch-tables` stages them ahead of
time or offline with `--from DIR`. An explicit
`WOOF_THOMPSON_TABLE_ROOT` must already contain the tables; automatic
staging does not fill a named root. The default staging location is
outside the install, beside `~/.woof/bridges`, because staging inside
site-packages meant the next `pip install --upgrade` deleted the
download without saying so. The staged directory holds the whole
four-asset set (the two the wheel does carry are copied in beside the
two it fetches), since a run reads one complete table root. A checkout
whose packaged root already has all four -- and any wheel that staged
into site-packages before 1.4 -- keeps reading that one and stages
nothing.
`woof doctor` reports exactly which pieces are missing and prints the
command that fixes each one.

Downstream programs can check tables without starting CUDA or downloading:

```python
from woof import tables

report = tables.status({"mp_physics": 50, "sf_surface_physics": 2})
print(report.needed, report.present, report.missing, report.corrupt)
tables.ensure({"mp_physics": 50}, fetch=False)  # raise a named table error
tables.ensure({"mp_physics": 8})  # retain verified Thompson first-use staging
```

These public checks inspect the current bytes on every call. Physics loaders
reuse verification within a process while the resolved path, size, nanosecond
modification/change times, inode and device remain unchanged; replacing a table
invalidates that cache. Admission and checkpoint identities share those digests.

Both helpers accept a `RunConfig`, an experiment, a mapping of physics
selectors, or family names such as `["p3", "noah", "tempo", "rrtmgp"]`.
An experiment also includes the tables its real-data land-use initialization reads.
`present` means the size and SHA-256 match; `ready` means every required file
is present. Corrupt data raises `tables.CorruptTableAssets`; missing data
raises `tables.MissingTableAssets`. No scheme is substituted. TEMPO retains
its verified automatic acquisition and announces each staged file.
Acquisition failures raise `tables.TableAssetError` with the attempted source
and the offline staging command.

`woof fetch-tables --schemes p3 noah --from DIR` stages just those families.
The directory can mirror the package's data layout, contain family
subdirectories, or hold the files directly. Add `--replace-damaged` to repair
a corrupt destination: the replacement must pass its pin before the atomic
install, and a bad source leaves the destination untouched. `WOOF_TABLE_ROOT`
selects an isolated packaged-table root with the package's relative data
layout; existing Thompson, TEMPO and P3 root overrides remain supported.
For a pinned uncompressed mirror, `WOOF_TABLE_ASSET_URL_BASE` plus each
packaged table's relative data path supplies its URL. Failed or interrupted
downloads never publish a partial table; rerunning safely starts a new
transfer. Offline failures name the attempted URL and the `--from DIR` route.

High-resolution static geography needs no extra. Its engine is the Rust
`static-fields` library, which ships in the bridge bundle every install
line on this page stages, and `woof doctor` reports it as `static
builder (default static-field engine)`. The `[geog]` extra carries
rasterio and pyproj, which are the pure-Python **parity fallback** that
`WOOF_STATIC_PYTHON=1` runs on -- install it to bisect a divergence
against the reference implementation, not to get the feature. Doctor
reports those two as `geography stack (rasterio + pyproj, the highres
fallback)` and says `info`, not `missing`, when they are absent.
See `docs/public/HIGHRES-TERRAIN.md` for the worked example. Note that the
international terrain path is a shipped feature, while the US full-stack
raster overlay (land use and soil as well as terrain) remains a prototype
and is not a stock-WRF-certified static source.

Build the vendored Rust decoder and CPU preprocessing workspace without
network access:

```bash
(
  cd tools/grib1_bridge
  cargo test --locked --offline
  cargo build --release --locked --offline
)
```

The crate emits the generic GRIB1/GRIB2 tools, source-specific bridges, and
the CPU shared library. Platform filenames differ (`.so`, `.dylib`, `.dll`).
Cargo must run from `tools/grib1_bridge` so it discovers the checked-in
`.cargo/config.toml` and uses `vendor/crates-io`; a populated developer Cargo
cache is not accepted as proof that the offline build is reproducible. The
Linux archive supplies CPU and CUDA setup paths. The Windows x86-64 archive
is deliberately CPU-only and uses PowerShell launchers; the named HRRR shell
pipeline is not portable to it yet.

## Optional extras

Every extra `woof` declares, and what naming it adds. A bare
`pip install recast-woof` is enough for preprocessing, static geography
including high-resolution terrain, radar velocity dealiasing, the
observation battery and the demo gallery's basemaps.

| Extra | Adds | Needed for |
|---|---|---|
| `recast-woof[gpu-cu12]` | `cupy-cuda12x[ctk]` | running the model on a CUDA 12.x box. The `[ctk]` half is a matching CUDA toolkit as wheels, ~1.6 GiB; without it a driver-only box cannot compile a kernel or load cuBLAS |
| `recast-woof[gpu-cu13]` | `cupy-cuda13x[ctk]` | running the model on a CUDA-13-only box |
| `recast-woof[gpu]` | alias of `gpu-cu12` | kept so existing install lines keep working |
| `recast-woof[render]` | `wrf-rust>=0.2.39` | `woof render`'s matplotlib engine, `woof enprod`, and derived quantities. The default rust engine needs none of it. The floor is 0.2.39 because that is the oldest release with wheels for every supported interpreter (cp310-cp314); no environment marker, nothing skipped |
| `recast-woof[dev]` | `pytest`, `pyyaml`, `setuptools` | running the test battery (setuptools because the wheel-gate tests run `setup.py` with the battery's interpreter) |
| `recast-woof[publish]` | `huggingface_hub` | maintainers only: publishing the WPS_GEOG mirror snapshot. Needs write credentials nobody else has, so it is deliberately outside `[all]` |
| `recast-woof[all-cu12]` | `gpu-cu12` + `render` | one line for a CUDA 12.x forecasting box |
| `recast-woof[all-cu13]` | `gpu-cu13` + `render` | one line for a CUDA-13-only forecasting box |
| `recast-woof[all]` | alias of `all-cu12` | kept so existing install lines keep working |
| `recast-woof[geog]` | `rasterio`, `pyproj` | the pure-Python **parity fallback** for the high-resolution warp (`WOOF_STATIC_PYTHON=1`). The default engine is the Rust `static-fields` library a bare install stages, so this changes no default and unlocks no product; deliberately outside `[all]` because 118.8 MiB of GDAL stack does not belong in the recommended one-liner for a debugging aid |
| `recast-woof[obs]` | nothing | **empty as of 2.4.1.** scipy moved into the base install, so scoring a forecast against observations (`woof.verify.obs`) works from `pip install recast-woof`; the name is kept so the old line does not fail |
| `recast-woof[dealias]` | nothing | **empty as of 2.4.1.** scipy moved into the base install, so the `vad-region` dealiasing engine is selectable from `pip install recast-woof`; the name is kept so the old line does not fail |

`pyshp` moved into the base install alongside `scipy` for the same
reason: `tools/da_nowcast_render.py` and the launcher's basemap endpoint
import it bare, at the very end of a DA nowcast run, and it is 74 kB of
pure Python. Nobody should have to choose an extra to get a coastline.

CuPy is an extra and not a dependency for one reason: it ships one wheel
per CUDA major and pip cannot tell which major a box serves, so the choice
has to be named rather than guessed. `woof doctor` reads the major off
the driver and prints the extra that matches.

## Sealed Linux archive

From a clean Linux x86-64 checkout, build the complete artifact with one
command. The Python wheel build and vendored Rust build do not access the
network; the Rust workspace is locked and built from its vendored sources.

```bash
python tools/build_rw_wps_release.py \
  --output-dir /tmp/rw-wps-0.1.1-linux-x86_64 \
  --archive /tmp/rw-wps-0.1.1-linux-x86_64.tar.gz
```

The underlying release builder accepts and independently probes one
source-matched wheel, every Rust bridge executable the bundle declares, and
the Rust CPU library.
It refuses a dirty Git tree, wheel/source drift, missing bridge ABI markers,
CRLF shell payloads, and developer-specific absolute paths in installed text
files. It emits a deterministic gzip archive, `manifest.json`, and
`SHA256SUMS`.

Exercise the resulting archive from a temporary directory and a new virtual
environment with no CuPy, CUDA, or Matplotlib installation:

```bash
python tools/smoke_rw_wps_cpu_install.py \
  --archive /tmp/rw-wps-0.1.1-linux-x86_64.tar.gz \
  --receipt /tmp/rw-wps-clean-cpu-install.json
```

The smoke command installs only NumPy, netCDF4 and threadpoolctl, extracts with traversal and
non-regular-entry checks, invokes `install.sh --skip-gpu`, runs the public CLI
from outside the checkout, and retains a receipt. Supply `--wheelhouse DIR`
for an offline dependency install.

After extracting an archive into a new directory:

```bash
python -m venv /opt/rw-wps-venv
/opt/rw-wps-venv/bin/python -m pip install 'numpy>=1.26' 'netCDF4>=1.6' 'threadpoolctl>=3.1'

WOOF_PYTHON=/opt/rw-wps-venv/bin/python ./install.sh --skip-gpu
./bin/rw-wps --version
./bin/rw-wps --show-support-matrix
./bin/rw-wps --namelist-support-report \
  --wps-namelist /case/namelist.wps \
  --namelist-input /case/namelist.input
```

`--skip-gpu` skips only CuPy/device verification. All wheel RECORD hashes,
archive hashes, decoder identities, GRIB2 tabular ABIs, CPU-library ABI,
helper inventories, an actual serial-versus-parallel FP32 interpolation
through the native CPU library, and the library's masked surface chain (soil,
snow, skin temperature and sea ice, which every backend maps through it) and
the native HRRR route's soil stencil and the lake search and water blends at
one and three workers remain checked. ERA5, GFS, and mapped runs with
explicit `--preprocess-backend cpu` can retain setup/export state in NumPy and
do not import CuPy. `--preprocess-backend auto` selects CUDA only when a device
and a certified CUDA runtime major (12.x with CuPy 13 or newer, 13.x with CuPy
14 or newer) are available; otherwise it uses the Rust/NumPy path, prints one
line saying why, and records the same reason in the preparation receipt's
`selection` block. HRRR's public driver does not yet expose the common
backend selector.

Auto also reads live card load through the fit probe. It selects CPU when
GPU utilization is at least 50%, or free device memory is at most 25% of
total memory. The receipt records the measurements, thresholds, and reason.
A card CUDA cannot open in the probe (too full for a context, or held by
another program in exclusive-process mode) also selects CPU; the receipt
names the CUDA error and the NVML reading taken before it. In a GFS domain
tree, each child prepared on the root's backend carries the root selection.
An idle card keeps the CUDA choice. Unavailable load readings keep the prior
runtime-based choice.
CUDA preparation receipts include the measured runtime and CuPy versions.

For the CUDA path, prepare the environment with the CuPy wheel matching
the box's CUDA major -- `cupy-cuda12x[ctk]>=14.0` on CUDA 12.x,
`cupy-cuda13x[ctk]>=14.0` on a CUDA-13-only box -- omit
`--skip-gpu`, and verify the runtime receipt generated at install. The
launcher repeats archive, wheel, bridge, dependency, and selected-backend
checks rather than trusting an old receipt.

## Sealed Windows x86-64 CPU archive

From a clean Windows x86-64 checkout, build the dedicated wheel, every PE Rust
bridge the bundle declares in `BUNDLED_BRIDGES`
(`woof/native_wrf_distribution.py`, one row per bridge naming the cargo
workspace it builds in), and the Rust CPU DLL without network access to Cargo:

```powershell
python tools/build_rw_wps_windows_release.py `
  --output-dir C:\release\rw-wps-0.1.1-windows-x86_64 `
  --archive C:\release\rw-wps-0.1.1-windows-x86_64.zip
```

The ZIP writer fixes timestamps, modes, ordering, compression, and its single
top-level root. Rust uses the locked vendored workspace and remaps the source
checkout path. The distribution builder executes every PE bridge identity
probe, loads the CPU DLL, checks its ABI, and runs byte-identical one- and
three-worker FP32 interpolation before publishing the archive.

Exercise the archive in a new venv outside the checkout:

```powershell
python tools/smoke_rw_wps_cpu_install.py `
  --archive C:\release\rw-wps-0.1.1-windows-x86_64.zip `
  --receipt C:\release\rw-wps-windows-clean-cpu-install.json
```

The smoke extracts only regular ZIP entries beneath one root, rejects path
traversal and case-insensitive duplicate paths, installs only NumPy,
netCDF4 and threadpoolctl, and proves CuPy and Matplotlib are absent. Manual installation is:

```powershell
$env:WOOF_PYTHON = "C:\absolute\venv\Scripts\python.exe"
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -SkipGpu
.\bin\rw-wps.cmd --version
.\bin\rw-wps.cmd --show-support-matrix
```

The PowerShell installer and launcher re-hash every declared payload, reject
rooted/traversing/duplicate checksum paths, reject reparse points, refuse
existing install targets, bind the exact Python interpreter, and reject the
CUDA backend. The portable Windows envelope is the public CLI, descriptor and
namelist authoring, and pure Python/Rust CPU routes selected with
`--preprocess-backend cpu` or `auto`. The Bash-based named HRRR driver, a
Windows CUDA release, and real-source/stock-WRF Windows certification remain
open gates.

## What a successful install does not prove

Installation proves artifact integrity and runtime availability. It does not
prove that a new source mapping is meteorologically complete, that an
arbitrary namelist is supported, or that emitted files advance unchanged WRF.
Preserve the run's proof JSON, input manifest, static receipts, export
manifest, output hashes, exact WRF executable hash, and WRF smoke log as one
evidence set.

Before a public release, second clean Linux and Windows machines must
reproduce the platform archives and representative CPU/CUDA cases, and the
project owner must select a repository license. The dedicated wheel retains the
existing internal `woof.*` module namespace for source compatibility, but its
distribution identity is `rw-wps` and its sealed payload excludes the model
driver, supervisor, dycore/physics executors, verification suite, and
forecast-only data tables.
