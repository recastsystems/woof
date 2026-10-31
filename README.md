# Recast WOOF

**Weather Observations and Open Forecasting: a GPU-native weather model you run on your own card.**

WOOF is a regional weather model with WRF-class physics that runs on one
NVIDIA GPU, with its own preprocessing, data assimilation and plots. One
install carries two models:

| Model | Command | Status |
| --- | --- | --- |
| Regional (WRF-ARW class, nests, DA) | `woof` | Production |
| Hex (MPAS-Atmosphere v8.4.2 on the GPU, regional and global) | `woof hex`, `woof hex global` | Preview |

Preview means the model is in the package, documented and tested on CPU, and
its command says preview each time it runs; its config and checkpoint formats
may change in a minor release. The major-version promise covers the regional
model.

Production means the regional commands and their config and checkpoint
formats are covered by that promise. It is not a statement about forecast
accuracy; see Science and limits below.

## Install

Linux on x86_64, Python 3.11 or newer, an NVIDIA GPU and driver. WOOF has
no Windows build of the native tools yet.

```bash
python3 -m venv ~/woof-env
~/woof-env/bin/python -m pip install 'recast-woof[all-cu12]'
~/woof-env/bin/woof fetch-tables
~/woof-env/bin/woof fetch-geog --datasets wrf
~/woof-env/bin/woof doctor
```

For a CUDA 13 setup use `all-cu13`. The hex model runs on either extra:
`all-cu12` works on a CUDA 12 or CUDA 13 driver, `all-cu13` needs a CUDA 13
driver, and `woof hex doctor` names the one for your driver. Install one
CuPy/CUDA major per environment.
The Linux wheel carries the native tools (GRIB and NetCDF decoding, the
WPS-equivalent preprocessor, the renderer).

`fetch-geog` downloads the global geography every forecast builds its
terrain, land use and soil from: about 1.3 GB, 17 GB unpacked, once per
computer. `fetch-tables` adds two Thompson microphysics tables (about 314 MiB).
Reference tables and map assets arrive with the `recast-woof-data` package.
`woof doctor` reports each remedy as a command or a `#` comment explaining
the manual step.

For a developer installation from a clone of this repository, use
`bash install.sh`: it installs the checkout's `recast-woof-data` companion and
the package with its extras, builds the native tools (this needs Rust and a C
toolchain) and ends with `woof doctor`. The explicit `bash install.sh` form
also works when an extracted source archive has lost its executable bits; use
it if `./install.sh` reports `Permission denied`.

## Run a forecast

```bash
woof domain --point 35.3,-97.5 --source hrrr --cycle latest --hours 3 \
     --root-dx 3 --point-extent-km 300 --out forecast.toml
woof go forecast.toml --dry-run
woof go forecast.toml
```

`--dry-run` reviews the route, sizes it for your card and names anything
missing; the second line fetches, prepares, runs and draws. Pictures are drawn
as each history frame lands, beside the running forecast.

Coupled hydro is opt-in. `woof go forecast.toml --hydro on` couples NextGen
hydro wherever a hydrofabric covers the domain; a forecast without it has no
hydro.

Compare two runs, any product, as run A minus run B:

```bash
woof render --diff runA/ runB/ --out diff/
```

The pair must share the grid and the valid times; a mismatch is refused by
name.

## Documentation

- `docs/public/`: the CLI manual, physics, hardware guidance and data sources.
  The [code verification record](docs/public/VERIFICATION.md) documents
  historical WRF v4.6.1 comparisons and their configuration and coverage
  limits.
- [Stage contract](docs/public/PIPELINE-STAGES.md) for running preprocessing,
  the simulation and rendering on your own terms.
- [Smoke, dust and air quality](docs/public/SMOKE-AND-AIR-QUALITY.md): wildfire
  smoke, dust, sea salt, carbon and CAMS air-quality species are an opt-in
  product, off unless a configuration names `chem_sets`.
- `components/hex/`: the hex model's manual, tests and tools.
- `woof --help-all` lists every command; `woof hex --help` lists the hex
  model's commands.

## Science and limits

WOOF implements a WRF-ARW-class regional model and WRF-derived physics on the
GPU. Comparisons against WRF Fortran, ULP and byte-identity checks, and the
matched-run and ensemble consistency comparisons are **code verification**:
evidence that the tested code computes its intended equations. Each result
belongs to the build, reference version and configuration it tested. None
establishes general equivalence with WRF or skill for every combination of
physics, input data and hardware.

**Solution verification** estimates numerical error in a particular run.
Routine numerical-error budgets for weather forecasts remain a gap.
**Validation** compares model results with observations. Meteorology calls
that scoring forecast verification; in computational-science V&V terms it is
validation. Matching WRF is code verification and never validates a model.
WOOF can inherit WRF's published validation record only to the extent the
two are statistically indistinguishable for the configuration in question.

Published observation evidence is limited to specific runs and variables,
including the ASOS snow/soil comparison, the station/ceilometer fog-suite
comparison and an MRMS model-top comparison. The observation scoreboard is
in development. These do not establish a broad validation record across
regions, seasons or forecast lengths. The
[evidence page](docs/public/VERIFICATION.md) documents historical WRF v4.6.1
code comparisons and their configuration and coverage limits.

The hex model is a GPU port of MPAS-Atmosphere v8.4.2, with MPAS-A's own
physics suites as its default and this engine's physics selectable; it is not
MPAS and is not endorsed by UCAR. Use official
meteorological services for forecasts and warnings.

WOOF continues an engine released before under another name: WOOF 1.0 is
that engine's release 2.8.0 and the fixes made since, under the WOOF name,
with the hex model inside it. Settings spelled `GPUWM_*` keep
working beside the new `WOOF_*` spelling.

## Licence

Apache License 2.0. Third-party code, tables and datasets keep their own
terms; see [NOTICE](NOTICE) and [licenses/](licenses/).
