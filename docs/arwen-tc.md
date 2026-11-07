# Opt-in tropical cyclone bundle

Generate an ordinary experiment with a fixed 9 km parent and following 3 km
and 1 km nests:

```sh
woof tc init --start 2022-09-26T00 --latitude 22 --longitude -83 --arm barr --out tc.toml
woof go tc.toml --prepare-only --outdir prepared
woof go tc.toml --prepared-root prepared/chain/prep --outdir forecast --products 10m_wind_speed_and_direction
```

For a storm merge through `woof go` or `woof prep`, the initial frame uses
the donor and receiving domains to choose its sampling lattice automatically.
A fine donor is not first collapsed onto a coarser background grid: the chosen
spacing is the coarser of donor spacing and finest receiving spacing, over the
receiving footprint and the merge disk. Rust resamples the background before
insertion. Later boundary frames remain on their original grids. An explicit
`--storm-merge-grid` still selects the initialization lattice.

The perturbation receipt distinguishes its analytical central-pressure target
from the minimum actually sampled on that lattice. Native initialization and
forecast adjustment still need inspection; matching the analytical target alone
does not establish that the model starts with it.

Review the initial nest positions against the input vortex before running.
The generator initially centres the nests inside their parents. `i_parent_start`
and `j_parent_start` place the initial nests; the existing pressure tracker then
relocates them without observed-track steering. Use `--les` to add 200 m and
40 m nests with the existing three-dimensional Smagorinsky closure and no PBL
parameterization. Their small footprints must cover the feature being studied.
This configuration is an experiment, not an assertion of demonstrated skill.

`--arm stock`, `green-zhang`, `bao` and `barr` select stock isftcflx=2,
the option-2 neutral curve, that curve plus the 2000 spray parameterization,
and that curve plus the 2023 sea-state parameterization. `woof tc curves`
prints all equation rows, observational anchors and extrapolation assumptions.
Neutral Ch and Cq are distinct. Enthalpy Ck is their temperature/humidity
weighted combination, not automatically equal to either one.

The `[shared]` selectors, also overridable per domain, are:

| Setting | Inert default | Meaning |
|---|---|---|
| `tc_flux_curve` | `"off"` | Named neutral exchange table |
| `tc_spray` | `"off"` | `bao2000` or `barr2023` |
| `tc_spray_partition` | `1.0` | 2000 evaporation cooling partition, 0 to 1 |
| `tc_wave_roughness` | `false` | Wave-dependent momentum roughness |
| `tc_wave_source` | `"fetch"` | Local parametric `fetch` spectrum |
| `tc_fetch_m` | `100000.0` | Fetch in metres for the parametric spectrum |
| `tc_horizontal_length_m` | `0.0` | Fixed horizontal turbulence length; zero retains stock |

Surface options support the MM5 surface layer (1 or 91) and no LSM, thermal
diffusion, or Noah. They affect open sea only, excluding lakes and ice.
Horizontal length requires diff_opt=2 and km_opt=3 or 4; vertical mixing stays
with the selected closure. `--horizontal-length-m` exposes this in the generator.
`--wave-roughness` requires a named scalar exchange curve.

The fetch model integrates a local deep-water JONSWAP spectrum to obtain
wave height, phase speed, dissipation and mean-square slope. It has no swell,
currents, propagation or storm memory.

The 2000 spray multipliers cap wind at 30 m/s as in the paper. The 2023
implementation integrates 48 logarithmic radius bins over 2 micrometres to
2 mm and iterates spray feedback. It uses constant air properties and an
approximate mean ventilation factor. It presently retains interfacial exchange
and flags cells outside the positive radius-relaxation domain, including
condensation above saline saturation, or failed feedback convergence. The
published signed relaxation-time expression is not silently replaced by an
absolute value. These restrictions require scrutiny in the hurricane proof.
This is not yet a validated full reproduction of the paper.

At forecast completion, `tc` receipt blocks give surface CUDA-event intervals
and diagnostic flags without adding a blocking synchronization per step.
The intervals bracket launch calls and include stream gaps between enqueues;
they are not isolated kernel instruction times or a measured counterfactual
wall-time difference. Mixing cost and separate CPU dispatch cost are not
included. Field arithmetic runs in CUDA, but the current surface and wave
methods still assemble launch arguments in Python each step. Native launch
attachment remains unfinished; this implementation does not yet meet the
no-new-Python-step-work requirement.
The all-off configuration creates no TC surface state, launches no TC kernels
and removes inert TC keys from existing config/restart fingerprints. A complete
base-versus-off forecast identity check is still required. The delivered short
fast-mode check matches all 204 decoded arrays across three domains at the
initial time and 90 seconds; it does not establish full-run equivalence.

Landfall wind-decay diagnostics consume a CSV with `time_hours`, `max_wind_ms`
and `centre_land_fraction`. They bracket the crossing between frames and end
an inland episode on a return to sea:

```sh
woof tc decay --input track.csv --out decay.json
woof storm-diagnostics --input-list frames.txt --out storm --heights 500,1000,3000 --gravity 9.81 --jobs 6 --render
woof tc vortices --input storm/wrfout_d05_FRAME --height 500 --dx 40 --dy 40 --latitude 26 --out candidates.json
```

The last command runs GPU neighbourhood detection on the existing Rust
diagnostic fields. It refuses grids coarser than 100 m. Cyclonic vorticity
above 0.2 s^-1 and a nearby updraft above 20 m/s identify candidates below
3 km AGL. Review circulation diameter and lifetime in wind frames; this is
not an automatic tornado forecast. Render weather fields only through
`woof.rustwx` and the Rust renderer. Wind-decay time series may use ordinary
charting tools.

Equation sources and observational curve provenance are cited beside each
implementation. Spectral-wave coupling belongs to a separate capability.

For hurricane wind maps, the stock 60 kt legend is too narrow. The native
renderer theme accepts an explicit `colormaps.levels` row in displayed units;
use the same row for every arm and time, for example:

```json
{"text":{"model_label":"WOOF"},"colormaps":{"levels":{
  "wind_speed_10m_agl":[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160]
}}}
```

The direct wind field uses the canonical key `wind_speed_10m_agl` in themes,
while the CLI product selector remains `10m_wind_speed_and_direction`.
Pass this file through the existing renderer `--theme` option. This requires
the renderer built from this branch. Omitted levels preserve the existing
render. This changes only the legend range, never the field values.
