# WRF V3.5 windspec oracle

`windturbines_spec` and the nine-field windspec file belong to WRF V3.5,
before the modern `windfarm_opt` table format. This port keeps its analytic
`dragcof`, base-geopotential rotor areas and full advected-TKE copy. It does
not replace the analytic power curve with interpolated modern tables.

The immutable public source and each distributed sample are pinned in
`native-pin.json`. Read the pinned README public-domain statement before
obtaining the sources. Its complete notice is retained in `NOTICE.txt`.
No competitor code or manufacturer curve is used.

Prepare an owned strict WRF v4.7.1 source copy with a clean model build. A reused build must remove both `.o` and `.mod` files together, including nested `phys/physics_mmm` objects: the added Registry keys change Fortran interfaces. Preserve its strict configuration and external libraries. Then run:

```
python tools/wind_legacy_wrf35_oracle/prepare_native.py OWNED_WRF REVIEWED_CACHE
cd OWNED_WRF
J='-j 2' ./compile em_real
```

The patch adds the unchanged V3.5 generic reader and turbine module under a
distinct module name, the native V3.5 Registry keys, initialization and a call
at the modern driver turbine position. It records the first real module call.
This is a native V3.5 module inside the v4.7.1 real forecast driver; it is not
a claim that the entire forecast uses WRF V3.5 meteorology. No physics
expression is changed. The driver adaptations are explicit in `capture.patch`.

Use `-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`, default real32,
and `-fconvert=big-endian` for the native build. Run `real.exe` and `wrf.exe`
with a real meteorological case, MYNN5/surface5, `bldt=0`, native
`windturbines_spec="windspec.in"`, `bl_mynn_tkeadvect=.true.`, and
`windfarm_opt=0`. Keep the captured `legacy-native.in`, `legacy-native.out`
and exact `windspec.in` together. Replay through the production Rust decoder
and resident CUDA class:

```
GPUWM_WRF_EXACT=1 GPUWM_WRF_GLIBC=2.43 PYTHONPATH=. \
  python tools/wind_legacy_wrf35_oracle/replay.py CAPTURE_DIRECTORY
```

The replay compares the geographic cell lookup, all six analytic turbine
parameters, both wind tendencies, QKE and the complete QKE_ADV field as raw
float32 words. The case and compiler receipts accompany the lane evidence.
Fast numerics remain the normal forecast default.

The ideal initialization and non-unit-density leg reuses this captured real
weather with the unchanged V3.5 modules compiled separately. Run in strict
mode (with the production decoder on `WOOF_RW_WIND_TABLE`):

```
GPUWM_WRF_EXACT=1 GPUWM_WRF_GLIBC=2.43 PYTHONPATH=. \
  python tools/wind_legacy_wrf35_oracle/ideal_replay.py CAPTURE_DIRECTORY REVIEWED_CACHE NEW_SCRATCH
```

`ideal_driver.f90` supplies only constants, native namelist getter stubs and
binary IO. It requests a two-by-two rectangle with 2.5 turbines per cell.
It compares native initialized cells/parameters and all 595,680 output words
through the production Rust ideal expansion and CUDA kernel. It does not
claim a second whole ideal-weather forecast.

The documented pseudo-real file interpretation has a native defect: with
all-zero geographic coordinates, V3.5 leaves `%i` and `%j` at the reader's
`-999` sentinels and never activates turbines. The GPU door names this
native limitation. The supported `ideal` selector uses the native rectangular
farm parameters and density directly. Geographic files retain their own grid
IDs, including one shared file used by nested domains.
