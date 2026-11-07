# Setup details

The [README](../../README.md#get-running) gives the package installation and
first forecast. This page collects the platform, data and offline details.

## CUDA and native tools

Local forecasting requires a compatible NVIDIA GPU and driver. A card's VRAM
capacity alone does not establish driver or CUDA compatibility.

For a manual Python installation on a supported CUDA 13 GPU and driver, replace
`all-cu12` with `all-cu13` in the README installation command. Install only one CuPy/CUDA major in each environment.
A driver reporting CUDA 13 support can also run CUDA 12 applications; that
display does not require switching a working CUDA 12 installation. See
[NVIDIA's compatibility guidance](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html).

| | `all-cu12` | `all-cu13` |
| --- | --- | --- |
| Minimum NVIDIA driver (NVIDIA's, per CUDA major) | R525: 525.60.13 Linux, 527.41 Windows | R580 |
| GPU architectures the forecast compiles for | sm_50 to sm_121 | sm_75 to sm_121 (not V100 or P100) |
| GRIB2 products on the GPU | sm_60 and newer | sm_60 and newer |

The driver minimums are NVIDIA's rules for each CUDA major, not runs of ours.
A card also needs a driver that supports its architecture (Blackwell, for
example, needs R570 or newer).

The two extras compile the forecast with different NVRTC builds, so the same
case on the same card gives different output words (on a 12 km case after
1 h, 2 m temperature differed by up to 0.34 K). Compare runs within one
extra. Details, and what was measured on which driver, are in
docs/public/HARDWARE.md under "Drivers, CUDA majors and GPU architectures".

`woof doctor` reports each remedy as a command or a `#` comment explaining
the manual step.

The GPU extra installs CuPy and the required user-space NVIDIA CUDA components
through their dependency packages. It does not install Python or the NVIDIA
device driver. Install one CuPy/CUDA major per environment. See the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) and
[WOOF hardware guidance](https://github.com/recastsystems/woof/blob/v2.9.2/docs/public/HARDWARE.md).

Most scientific lookup data and the renderer's map assets (coastlines,
borders, state lines and counties) arrive with the automatically installed
`recast-woof-data` package. The first Thompson forecast downloads and checks two
additional tables, about 315 MiB combined, from the pinned release assets.
`woof fetch-tables` stages them ahead of time or offline with `--from DIR`.
An explicit `WOOF_THOMPSON_TABLE_ROOT` must already contain the tables;
automatic staging does not fill a named root. Geography and forecast inputs are
separate: configure an existing geography installation or use `woof fetch-geog`.
ERA5 through Copernicus CDS requires the user's CDS configuration and dataset
access. Deterministic ERA5 reanalysis also has a public, keyless ARCO route:
select `--era5-provider arco` when creating the domain or fetching ERA5.
Ensemble-member retrievals still require CDS. Remote operation requires an SSH
client and a configured, reachable forecast computer.

Local forecast integration runs on CUDA; it has no CPU fallback. The desktop,
terminal, native weather processing, and remote-control workflows can run on a
computer without a local CUDA installation.

Only an intentionally viewer-only or remote-control installation should use
`recast-woof[render]` without a GPU extra. It cannot execute local forecasts.

## Geography and offline setup

`woof setup --with-geog` stages native tools, physics tables and geography.
The full geography set is about 2.2 GB compressed and 35 GB unpacked.
Use `woof fetch-geog --list` to inspect the datasets before downloading.
`woof fetch-geog --datasets wrf` stages only the regional set; the doctor
can still report missing mesh geography for global workflows.

`woof setup --from DIR` verifies and stages native tools and physics tables
from a local directory. It does not redirect the geography download.
See [data paths](DATA.md) for an existing geography installation and caches.

## Python and source builds

For a source checkout, run `bash install.sh` on Linux or
`powershell -File install.ps1` on Windows. The scripts build the native
components, install the package and run the doctor.
Use the [developer installation guide](../install.md) for toolchain requirements.

The scripts prefer free-threaded Python 3.14t. Multi-card forecasts use one
Python thread per card, so the interpreter lock affects their scheduling.
For a manual free-threaded environment:

```bash
uv venv --seed --python 3.14t .venv
.venv/bin/python -m pip install --prefer-binary 'recast-woof[all-cu12]'
```

On Windows, use `.venv\Scripts\python.exe` for the second line.
The render extra's `wrf-rust` dependency builds from source on 3.14t and
needs Rust. `--prefer-binary` selects an available binary dependency version
when a newer version has only a source distribution.

## Desktop application

The desktop GUI is separate from the `woof` package. Choose its archive
from the [release assets](https://github.com/recastsystems/woof/releases)
and retain the application resources and license notices beside it.
Windows starts with `WOOF.exe`; Linux uses `Start WOOF.sh --python PATH`
with the forecast environment's Python. The
[desktop guide](GUI.md) covers the interface and remote operation.
