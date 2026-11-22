# Real initial conditions: WOOF-ArWen already does this. Do not rebuild it.

Every state in `tilestream/` so far is a seeded synthetic draw from
`harness.make_state`. That is deliberate: a bit-exact gate needs two runs to
start from an identical, reproducible condition, and a seeded state gives that
for free with no download.

**It is not a statement about WOOF-ArWen's capabilities.** WOOF-ArWen is a full-featured
model with a production ingest for GFS, HRRR, ERA5 and 20CR. An agent that
concludes "real data is a gap" has misread the absence of real data in the
STREAMING TESTS as the absence of real data in the MODEL.

## Where it lives

```
woof/ingest/real.py          the real-data initialisation path
woof/ingest/grib.py          GRIB decode
woof/ingest/hrrr.py          HRRR, plus hrrr_surface / hrrr_physics / hrrr_target
woof/ingest/lateral_bc.py    lateral boundary series -- ALREADY USED by the
                              tiled boundary work, see driver/spec
woof/ingest/vert.py horiz.py vertical and horizontal interpolation
woof/ingest/soil.py ruc_soil.py wrf_ozone.py water_overlay.py
woof/ingest/prepared_cache.py  cached preprocessed input
woof/ingest/nest_init.py nest_spawn_init.py relocation_init.py

tools/download_gfs_native_subset.py    tools/download_hrrr_native_subset.py
tools/prepare_gfs_cpu_wrf.py           tools/prepare_era5_cpu_wrf.py
tools/hrrr_pipeline.py                 tools/hrrr_two_domain_forecast.py
```

## The only thing streaming actually needs from it

A domain larger than VRAM cannot be initialised by building a monolithic
`DomainState` on the device and then copying it out: the whole premise is that
no such state fits. So the one piece of work is routing the ingest's output
into a **pinned host store** field by field, never materialising the full domain
on the GPU.

Everything else (decode, interpolation, soil, ozone, lateral boundary series)
is existing, tested, production code. Call it. Do not reimplement it, and do not
report its absence as a finding.

## Why the gate still uses seeded states

Bit-exactness does not care what the numbers are: if the tiled run reproduces a
monolithic run digit-for-digit on synthetic air, the transport is correct for
real air too, because the transport never inspects a value. Seeded states are
also rougher than the real atmosphere, so they stress the halo and the boundary
zone harder than a real analysis would. Keep using them for the gate; use real
data for forecasts, plots, and anything a human will look at.
