TC forecast evidence helpers
===========================

These scripts run outside the forecast step. Stage copies in an owned lane's
`runs/` directory, set `TC_FLUX_ROOT` to that lane root, and activate its engine
venv. `bind-published-bridges.py` binds the published helper pack on the atlas
boxes without changing the shared installation. It also includes the native
MPAS stream writer, whose EngineSpec is outside the ordinary engine list.

`f-box-forecast.py ARM CONFIG` launches one complete prepared WOOF forecast,
records source/device/config identity, samples resource use asynchronously,
and runs the collector. The prepared root is `runs/prepared/chain/prep`.
ARM `base` reads the detached base source in `runs/base-source`; other arms read
`engine/`. No arm can silently overwrite an earlier launch marker. Input
configuration files are separate case rows.

`box-products.py` consumes durable history markers. It retains field hashes,
physical surface fields, derived MSLP and total enthalpy flux; six-hourly inner
volumes remain on the box. Failed reductions retain their source. A 40 GB lane
cap is recorded and a guard pauses writers below that limit, or when shared
filesystem free space falls below 8 GB, resuming above 14 GB. The operator
must render and retire retained volumes as the run proceeds.

`box-render.py --arm ARM` renders common-scale wind and enthalpy maps through
`woof.rustwx`. Inspect every selected time and redraw any clipped scale.
`box-sections.py --arm ARM --radius-km R` renders retained inner volumes as
scalar azimuthal means at common MSL heights, then retires the source after
durable picture and hash receipts. Select a radius supported by the domain;
incomplete circles remain missing. This requires the azimuthal Rust renderer.
`compare_off.py --runs RUNS --start UTC --hours HOURS --out RESULT` refuses
missing frames, failed producers, nonfinite states and unequal inventories.
Its passing result concerns all published history arrays, not unrecorded
internal state or restart bytes.

The native hex collector and resource guard live in the companion hex tree,
`tools/tc_flux/`. Neither collector is a data-assimilation background exporter.

`render-flux-detail.py SCRATCH --maximum 1000` adds a tighter upward enthalpy
map from the native collector's actual converted frame. Invoke before the
collector retires its `render-scratch` directory, after `enthalpy_picture`
adds TC_ENTHALPY and binds terrain. Its receipt records the source and scale;
it neither changes fields nor reruns the forecast. The original common-scale
view remains available. This helper requires the paired hex installation.
