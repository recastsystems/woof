# PWP ocean on native atmospheric cells

Select `woof hex forecast ... --ocean-mode pwp` on either physics route. With
WOOF physics the columns run in its Noah-MP surface hook. With MPAS physics
(the hex default) the same columns, kernel and parameter table take the OML1D
slot's place and timing in the MPAS driver (once per `dt_pbl`, after the surface
layer), forced by MPAS's own `gsw`, `glw`, `hfx`, `lh`, `ust`, `qfx` and
`raincv + rainncv` and the lowest-level wind; they write `sst`, `skintemp`,
`t_oml` and `h_oml`. On that route PWP steps open water only and sea ice stays
with MPAS's sea-ice scheme, so ice keeps one owner. Do not also name
`config_oml1d`: that would be two mixed-layer oceans in one slot, and the run
refuses it. Storm-following moves and checkpoints carry the columns on one card
and on several. The existing `prescribed` and `prognostic` choices retain
their meanings and defaults. PWP is opt-in. This foundation delivery does not
implement KPP, a reduced-gravity ENSO ocean, MPAS-Ocean or ALE.

The column has T, S, eastward velocity and northward velocity on uniform
layers. Surface heat, freshwater and stress drive it at the existing surface
cadence. Its new SST feeds the next surface-layer call. CUDA handles all column
arithmetic in the existing ocean launch. No new Python step routine or device
synchronization is added. Options and the profile identity are established
once at construction. All carried arrays belong to the existing surface
checkpoint and rollback pool.

The implementation follows the accessible HYCOM PWP technical specification:
https://www.hycom.org/attachments/067_pwp.pdf. The original foundation paper
was unavailable. Citation: Price, Weller and Pinkel (1986),
DOI 10.1029/JC091iC07p08411. No implementation code was copied.

The defaults use 40 layers of 5 m, bulk Ri 0.65, gradient Ri 0.25 and a
relaxation target of 0.30. Surface instability, entrainment and pair relaxation
mix temperature, salinity and momentum conservatively. Exact inertial rotation
avoids explicit rotational energy growth. The density relation is linear;
sunlight has one exponential attenuation length and freshwater uses a virtual
salt flux. These are declared idealizations. There is no horizontal transport
or background diffusivity. Initially icy columns retain the baseline bulk
ice model for the whole run. Initially open-water columns have no new ice
formation closure, so this is a warm-water research option.

`--pwp-options '{"levels":80,"layer_m":2.5}'` overrides table parameters.
`--pwp-profile profile.npz` supplies `columns[nCells,4,nLevels]` in global native
cell order, with T in kelvin, S in PSU and u/v in m/s. The constructor gathers
this profile through the same cell-order map as the atmosphere. Without a
profile, the initial SST anchors a configured 30 m mixed layer and 0.05 K/m
thermocline prior. That prior is not a subsurface observation. Do not interpret
agreement with SST as validation of the assumed subsurface heat reservoir.

History includes SST, mixed depth, full-column heat residual and
`ocean_pwp_mixing_failures`. That last field must stay zero: a bounded iterative
mixing solve records each exhausted iteration limit rather than hiding it.
The full T/S/u/v profile is in checkpoints, not in ordinary weather history.
The residual is full-column heat change minus absorbed boundary heat. Radiation
that passes through the bottom is excluded from the absorbed heat budget.

Executable demo, after preparing native grid/static/init inputs:

```sh
python tools/run_coupled_ocean_case.py \
  --config examples/coupled-ocean-ian-2022.json \
  --inputs INPUTS --work-dir RUN --mode prescribed
python tools/run_coupled_ocean_case.py \
  --config examples/coupled-ocean-ian-2022.json \
  --inputs INPUTS --work-dir RUN --mode pwp
```

The input directory contains the files named in the case table, mesh-rows.json
with paths valid on the run host, and librw_surface.so. Run the demo under the
site's GPU pool scheduler. The wrapper caps all arms together at 20 GB and
records the command, wall time, return code and peak disk size. It samples
process CPU and device allocation outside the model's step path.

The supplied case is a five-day September 2022 Atlantic hurricane period on
the 120 km globe with 55 atmospheric levels. Compare SST response with NOAA
OISST and buoy observations. This grid cannot resolve eyewall intensity. A
higher-resolution coupled run with observed subsurface initialization remains
necessary for hurricane intensity qualification. KPP and dynamic ocean stages
remain later work.

The restart rejection fix validates every array layout and ocean identity
before copying any live array. Tests cover rejected mode, options, shape and
dtype. GPU tests cover actual column conservation and deterministic continuation
from saved arrays. These are separate from the published hl-3 whole-model gate;
no test name implies that gate ran. Consult the delivered case receipt for
actual forecast and gate outcomes.

PWP selects hex as the sole sea-ice owner during construction. gpuwm 2.9.4
otherwise also advances a separate ice model after the hex hook, changing
XICE after the coupled surface census. `--ice-owner hex` gives prescribed-SST
controls the same owner; the demo sets it for both arms. The unselected
published path is unchanged. This setup removes the duplicate owner's
cold-seeded fields before checkpoint manifests bind, without adding step work.

Reduce each completed history directory with
`python tools/reduce_coupled_ocean_history.py RUN/ARM RUN/ARM-reduced`.
Then fill the file paths in `examples/coupled-ocean-evaluation.json` and run
`python tools/evaluate_coupled_ocean.py --config EVALUATION.json`.
The evaluation matches OISST's nominal noon daily analyses to sparse history,
uses the nearest native ocean cell, and reports latitude-weighted SST errors.
Buoy water temperatures and best-track winds are independent observations.
The storm wind statistic is a coarse-grid maximum near the observed center,
not a tracked or resolved eyewall. Weather maps use the Rust renderer with a
shared physical colour scale; matplotlib is used only for time series.
The evaluation table also declares a smaller storm-transit box. SST scores
are reported both over the full region and over that box, so broad Atlantic
agreement cannot hide a missing local cooling response.
