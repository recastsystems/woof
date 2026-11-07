# WRF surface updates

`sst_update=1`, `sst_skin=1` and `tmn_update=1` are independent WRF selectors.
All default to zero. The namelist importer preserves them. Native WRF enforces
`lagday=150` when TMN updating is enabled, and WOOF does the same.

A stock `namelist.input` directory with `wrfinput_dNN`, `wrfbdy_d01` and
`wrflowinp_dNN` runs through `woof go DIRECTORY`. Auxiliary stream 4 uses
`auxinput4_inname`, `io_form_auxinput4` and the declared per-domain interval.
Each domain gets its own Rust-prepared, coordinate-bound stream. A met-em
run produces stream 4 with the shared rw-wps implementation:

```sh
rw-wps wrflowinp --input met_em.d01.TIME1.nc --input met_em.d01.TIME2.nc \
  --sf-surface-physics 2 --output wrflowinp_d01
```

The producer writes WRF-compatible CDF-2 records: SST, SEAICE, VEGFRA and
ALBBCK, coordinates, static ownership, valid times and projection metadata.
Monthly GREENFRAC and ALBEDO12M follow WRF's day-15 interpolation in float32.
Existing SST is authoritative; otherwise initialized TSK or metgrid SKINTEMP
supplies the bulk temperature.
The SKINTEMP fallback applies native process_soil_real's terrain lapse
correction when FLAG_SOILHGT and the selected land-surface scheme require
it, preserving its land mask, float32 order and height guards. The met-em
launch passes the configured surface scheme; the standalone producer defaults
to Noah (2). A native real.exe reference stream from the recorded case matches
all 52,560 generated field words and all valid times. Inputs needing other WRF surface initialization
repairs must first supply initialized surface fields. Production does not
invoke real.exe. See the producer's explicit input checks and source pins.

For downloaded public analyses, prepare before launch:

```sh
python -m woof.surface_analysis --grid wrfinput_d01 \
  --input oisst=analysis-earlier.nc --input oisst=analysis-later.nc \
  --out surface-stream
```

Use the manifest as `surface_analysis`. The source table declares variable
names, units, missing values, geographic ownership and bounded remapping.
OISST, GHRSST L4, GLSEA, OSI SAF and WRF stream 4 share one Rust reader.
New sources are metadata rows. Reader ABI `surface_analysis_v4` and
`wrflowinp_v1` are required. Coordinates and source hashes enter restart
identity. Moving a grid requires a newly prepared stream for that footprint.
WRF stream 4 preserves all four fields across the full grid without an
analysis temperature filter; the native surface-driver kernel owns its
250 to 350 K water-temperature guard. For public analyses, missing source cells preserve the last valid model value. Selected records
are the latest non-future records; they expire at their declared next update.

CUDA applies the original WRF surface-driver order on every physics call:
analysis loads, lake/sea-ice ownership, concentration adjustments, freeze,
water SST replacement and melt. Freeze/melt resets include land and soil
categories, all soil layers, snow, vegetation, albedo and emissivity. Under
fractional sea ice, WRF recomputes the temperature mixture only when ice
concentration changes. An active lake model retains its own temperature;
the WRF sea-ice threshold controls its ice clearing. RUC mosaic selectors mix parameter tables and consume these same
grid carriers; they do not create a separate prognostic surface temperature.

Skin SST evolves SSTSK and DTW with WRF's incoming and outgoing surface
fluxes, then writes TSK over water. Bulk SST remains separate. Fast CUDA
math is the default; `GPUWM_WRF_EXACT=1` selects strict replay math. WRF
initializes allocated REAL state to zero in tools/gen_allocs.c. SSTSK has
no later cold-start initializer, so WOOF preserves the native zero seed.
A supplied SSTSK or checkpoint state takes precedence.

TMN updating keeps TLAG, TYR, TYRA, TDLY, NDAY and NYEAR on the device.
The 150-day sum preserves native float32 ordering. Daily shifting and yearly
means follow WRF's fractional Julian day and leap-year decisions. Domain
counters update in their own ordered CUDA launch, without a host read.

History exposes SST, SSTSK, TSK and TMN; active skin runs also expose DTW.
Checkpoints preserve the complete active state. Native wrfrst import transposes
the TLAG bank from WRF's `(y, lagday, x)` to `(lagday, y, x)` and restores
its domain counters. Nest feedback uses Registry `copy_fcnm/copy_fcni` for
fields carrying a feedback flag. Restart-only state and domain counters do
not feed back. Relocation preserves history and overlap state. Ensembles
with these selectors use independent member CUDA drivers; their mutable
surface categories and TMN clocks cannot use the existing shared packed
Noah binding.

The pinned oracle in `tools/sst_surface_wrf461_oracle` isolates each original
WRF block and both unmodified native modules. The recorded strict replay
compares 24,592 words per option, with zero differences, from eight captures
of a real six-hour native forecast. An additional 24,592-word end-of-block
comparison agrees with that actual forecast. The warm coastal case crosses
midnight but does not contain freeze/melt or a year boundary. Supplementary
branch and state tests are distinct from this forecast-input replay.

The same six-hour case runs through `woof go` with all three selectors on.
Maps at 0, 3 and 6 hours are rendered through gpuwm.rustwx/rw_wrfbatch beside
native WRF and a daily observed SST analysis. This is a visual check of this
case, not a forecast skill score. See the lane evidence for exact configuration,
commands, timing samples, field hashes and limitations.
