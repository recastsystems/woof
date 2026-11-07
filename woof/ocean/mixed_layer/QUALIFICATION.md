# Whole forecast qualification

A real 72-hour Ian 2022 forecast (2022-09-26 00 UTC to September 29 00 UTC)
completed 8,640 atmosphere steps and 25 history frames on a shared RTX 5090:
182 × 146 × 49, 12 km Mercator, 30 s atmospheric step, 30 ocean levels.
Thompson, YSU, MM5, Noah and legacy RRTMG were active. All PWP circulation,
transport and mixing terms remained enabled; CUDA graphs were disabled.

Initialization used GFS SST plus NOAA WOA18 September mixed-layer depth,
35 PSU, resting currents and the configured thermocline gradient. This named
fallback lacks contemporary ocean anomalies. Attempts to acquire historical
HYCOM profiles failed at the remote subset service with HTTP 500. Supplied
RTOFS/HYCOM/CF analysis files use the same Rust reader/CUDA remapper door.

The full command took 289.260 s, model execution 280.585 s, forecast with
async output 278.006 s. PID-matched sampled peak GPU allocation was 5,234 MiB,
with an 8,192 MiB shared pool reservation and two-second sampling. These are
shared-card measurements, not an isolated performance comparison.

All 50 SST and SST-change maps were produced by gpuwm.rustwx/rw_wrfbatch and
visually inspected. They show development of a cold wake west of Florida,
approaching 2.4 K cooling by hour 72, with missing land and lake values.
Early transient diagonal bands and automatic per-frame colour scales are
visible; the fields are not tuned or smoothed for the demonstration.

The native eight-thread verifier scored 25 frames in 0.530 s. Daily
left-interval time-weighted SST means against NOAA OISST give RMSE
0.491/0.502/0.531 K on September 26/27/28 (20,513 wet cells per day,
cosine-latitude weighted). Change from the first complete day has RMSE
0.176/0.299 K on September 27/28. NDBC 2022 station positions were taken from
the corresponding historical NetCDF metadata. Twenty-five WTMP pairs per
station give SST RMSE 1.019 K at 42036 (distance 3.88 km) and 0.155 K at
42056 (4.42 km); change-from-start RMSE is 0.458/0.332 K. The station 42036
cold bias is 0.989 K, including a 0.602 K initial SST offset. These are
measured errors, not acceptance thresholds or a broad skill endorsement.

The ocean is flat-depth (5 m layers), with no bathymetry or sea-ice energy
balance. Coastal mapping borrows complete profiles within one degree.
The fallback cannot initialize mesoscale anomalies. OISST includes buoy SST,
so the two comparisons are not independent. Model top-layer, foundation SST
and buoy sensor depths differ without correction. Spatial matching is
nearest-cell, not conservative. No bitwise WRF parity is claimed.

Full provenance, source licences, resource samples, per-stage timings,
observation pairs, source hashes and pictures are retained in the private
slab-ocean port delivery. Qualification does
not establish prognostic sea ice, ice-sheet runoff or seasonal ocean skill.

The installed duration-default path was then run for the complete same
72 hours with `sf_ocean_physics` omitted. Preparation was regenerated from
the cached GFS inputs to preserve the exact-TOML source-manifest binding
(215.34 s, 170% CPU on the test host). All 8,640 steps and 25 frames passed on the
shared RTX 5090. Command wall was 599.078 s, model wall 588.922 s, sampled
peak 4,702 MiB within the 8,192 MiB pool reservation. No products were
duplicated in this second run. Its canonical final-state digest, covering
206 arrays and scalar state, exactly matches the explicit-selection run:
`bac09317e6e750c5d6978381d2f1a1ed1e906620c27c1cf704f71772f408e0a1`.
The two shared-card walls have different contention and are not a speed
comparison. Both runs used eager CUDA without CUDA graphs. The default
policy retains an explicit option 0 opt-out and explicit option 1 Pollard.
