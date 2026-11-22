# Recast WOOF

**Weather Observations and Open Forecasting: GPU-native weather models you run on your own card.**

WOOF is two weather models that run on NVIDIA GPUs, in one install with
their own preprocessing and maps: WOOF-ArWen (the `woof` command) and
WOOF-HEX-A (hexagonal mesh; A for Atmosphere, as in MPAS-A), the `woof hex`
command.

| Model | What it is | Command |
| --- | --- | --- |
| WOOF-ArWen | The WRF engine: WRF-ARW on GPUs for regions you draw, with nests and storm-following runs. Fast numerics by default; an exact mode is checked against native WRF 4.8.0 on the cases its verification page names. | `woof` |
| WOOF-HEX-A | The MPAS engine: MPAS-Atmosphere v8.4.2 and NOAA's hrrrv5 (RRFS v2) configuration on GPUs, on hexagonal meshes: whole globes, zoom globes, limited-area and storm-following regions, with a tangent-linear and adjoint for sensitivity work. | `woof hex`, `woof hex global` |

WRF and MPAS are developed at NCAR; WOOF is not affiliated with or endorsed
by NCAR or UCAR.

Each engine keeps its own version number. WOOF-ArWen is also released on its
own as the `gpuwm` package, and WOOF-HEX-A as `gpuwm-hex`; WOOF 1.1.7
carries gpuwm 3.0.4 and gpuwm-hex 0.6.2. Install `recast-woof`, or `gpuwm`
and `gpuwm-hex`, not both in one environment: they install the same
top-level packages and the same `rw-wps` command.

The WOOF site's API keeps its own values for the two models:

| API `model` value | Model |
| --- | --- |
| `regional`, `cyclone` (and `kind: arwen` in run lists) | WOOF-ArWen |
| `hex` | WOOF-HEX-A |

Separate projects, not part of this package: woof-nam
([recastsystems/woof-nam](https://github.com/recastsystems/woof-nam)) and
ArWen Global
([FahrenheitResearch/gpuwm-global](https://github.com/FahrenheitResearch/gpuwm-global)).

WOOF-ArWen is the production model: its commands and their config and
checkpoint formats are covered by the major-version promise. That is not a
statement about forecast accuracy; see Science and limits below.

WOOF-HEX-A is a preview: it is in the package, documented and tested on CPU,
and its command says preview each time it runs; its config and checkpoint
formats may change in a minor release.

## Install

Linux on x86_64, Python 3.11 or newer, an NVIDIA GPU and driver. WOOF has
no Windows build of the native tools yet.

```bash
python3 -m venv ~/woof-env
. ~/woof-env/bin/activate       # puts woof and its tools on PATH; again in each new shell
python -m pip install 'recast-woof[all-cu12]'
woof fetch-tables
woof fetch-geog --datasets wrf
woof doctor
```

For WOOF-HEX-A, add the geography its meshes read, then check it:

```bash
woof fetch-geog --datasets mesh
woof hex doctor
```

For a CUDA 13 setup use `all-cu13`. WOOF-HEX-A runs on either extra:
`all-cu12` works on a CUDA 12 or CUDA 13 driver, `all-cu13` needs a CUDA 13
driver, and `woof hex doctor` names the one for your driver. Install one
CuPy/CUDA major per environment.
The Linux wheel carries the native tools (GRIB and NetCDF decoding, the
WPS-equivalent preprocessor, the renderer).

`fetch-geog --datasets wrf` downloads the global geography every forecast
builds its terrain, land use and soil from: about 2.6 GB, about 26 GiB
unpacked, once per computer. `fetch-tables` adds two Thompson microphysics tables (about 314 MiB).
Reference tables and map assets arrive with the `recast-woof-data` package.
`woof doctor` reports each remedy as a command or a `#` comment explaining
the manual step.

For a developer installation from a clone of this repository, use
`bash install.sh`: it installs the checkout's `recast-woof-data` companion and
the package with its extras, builds the native tools (this needs Rust, a C
toolchain and CMake) and ends with `woof doctor`. The explicit `bash install.sh` form
also works when an extracted source archive has lost its executable bits; use
it if `./install.sh` reports `Permission denied`.

## Run a forecast

WOOF-ArWen, a 3 km regional forecast:

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

WOOF-HEX-A, a 120 km global forecast in one command (it fetches its mesh,
start and kernels itself):

```bash
woof hex global --row global-120km --cycle 2026-10-11T12 --work-dir RUN
```

## The derivative commands (WOOF-HEX-A)

`woof hex tangent`, `adjoint-test`, `sensitivity` and `param-grad` run the
tangent-linear and adjoint of the exact-mode hrrrv5 forecast
(`--physics-mode exact`), not of the correct-mode forecast `woof hex
forecast` runs by default. They run on NVIDIA cards of compute capability
12.0 (GeForce RTX 50 series, RTX PRO Blackwell) and need the `ad` extra, in
a Python 3.14 environment of its own (the extra brings its own CuPy package,
and one environment holds one):

```bash
python3.14 -m venv ~/woof-ad
~/woof-ad/bin/python -m pip install 'recast-woof[ad]'
~/woof-ad/bin/woof hex adjoint-test --row hrrrv5-q12.96.2890.n40.60w91.30 --steps 15 \
     --at 40.6,-91.3 --radius-km 150 --out cut12
```

The last command is the 12 km example: one card with 28,700 MiB free (a 32 GB
GeForce RTX 5090 runs it) and 47,104 MiB of host memory. It fetches its own
1.20 GB of inputs and ends PASS when the adjoint passes its transpose check
over the window and the gradient test (the real model rerun from a nudged
start, against the 1 per cent bar) finds no test direction outside the bar
and two independent directions inside it; a direction it cannot resolve is
named. No PASS covers a measure of the storm itself (rain water, cloud water
or updraft), and a measure built on radar reflectivity is refused. What each example row holds, and every
measured limit, is in `components/hex/docs/differentiable/` (`LIMITS.md`).

The learned column correction (`WOOF_HEX_LEARNED_CORRECTION`, from a
checkpoint of your own; no weights ship) and the two network options on the
hrrrv5 suite (the surface-flux network and the neural boundary layer) need
PyTorch, from the `torch` extra, which adds it to the `ad` environment:
`~/woof-ad/bin/python -m pip install 'recast-woof[torch]'`. Each is off
unless named, and no skill is claimed for them; the User Manual's chapter 11
(`components/hex/docs/manual/11-new-in-0.6.2.md`) says what each does.

## Documentation

- [FAMILY.md](FAMILY.md): the two models, their packages, commands and
  versions, and which installs can share an environment.
- `docs/public/`: the CLI manual, physics, hardware guidance and data sources.
  The [WRF comparison](docs/public/README-EVIDENCE.md#wrf-comparison) names
  the grids, physics and fields of WOOF-ArWen's exact-mode cases, and the
  [code verification record](docs/public/VERIFICATION.md) keeps the earlier
  WRF v4.6.1 comparisons with their configuration and coverage limits.
- [Stage contract](docs/public/PIPELINE-STAGES.md) for running preprocessing,
  the simulation and rendering on your own terms.
- [Smoke, dust and air quality](docs/public/SMOKE-AND-AIR-QUALITY.md): wildfire
  smoke, dust, sea salt, carbon and CAMS air-quality species are an opt-in
  product, off unless a configuration names `chem_sets`.
- `components/hex/`: WOOF-HEX-A's manual, tests and tools.
- `woof --help-all` lists every command; `woof hex --help` lists WOOF-HEX-A's
  commands.

## Science and limits

WOOF-ArWen implements a WRF-ARW-class regional model and WRF-derived physics
on the GPU.
Comparisons against WRF Fortran, ULP and byte-identity checks, and the
matched-run and ensemble consistency comparisons are **code verification**:
evidence that the tested code computes its intended equations. Each result
belongs to the build, reference version and configuration it tested. None
establishes general equivalence with WRF or skill for every combination of
physics, input data and hardware.

In strict numerics, a build of gpuwm 3.0.3 (the release before gpuwm 3.0.4,
which WOOF-ArWen carries here) matched stock WRF 4.8.0 in every stored value on two one-hour, 3 km
cases (40 x 40 x 50 points), on one card and one process. With
`--physics-mode exact --numerics strict`, gpuwm-hex 0.6.0's one-hour hrrrv5
forecasts on a 12 km and a 3 km limited-area mesh matched the patched native
RRFS-MPAS 44c0849 reference in every stored value, on one card and one
process. The default forecasts (fast numerics, and correct mode on
WOOF-HEX-A) are not bit for bit, by design, and other meshes, dates and
lengths are not claimed by these lines.

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
regions, seasons or forecast lengths.

WOOF-HEX-A is a GPU port of MPAS-Atmosphere v8.4.2, with MPAS-A's own
physics suites as its default and ArWen physics selectable; it is not MPAS.
Use official meteorological services for forecasts and warnings.

WOOF 1.0 was the engine's release 2.8.0 and the fixes made since, under the
WOOF name, with WOOF-HEX-A inside it. Settings spelled `GPUWM_*` keep working
beside the new `WOOF_*` spelling.

## Licence

Apache License 2.0. Third-party code, tables and datasets keep their own
terms; see [NOTICE](NOTICE) and [licenses/](licenses/).
