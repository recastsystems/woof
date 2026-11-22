# gwd_opt = 1 (KIM orographic drag) WRF v4.8.0 replay oracle

WRF v4.8.0 rewrote `phys/physics_mmm/bl_gwdo.F90` (MMM-physics 550b5b49:
Hong et al. 2025, Koo and Hong 2025, the nonhydrostatic factor of Xu et al.
2024) and passes it ELVMAX, `gwd_dx_factor` and `gwd_if_nonhyd`. WOOF-ArWen runs it
as `gwdo_version = "wrf_480"` (the default; kernel
`woof/core/kernels/terrain_drag_gwdo480.cu`). `"wrf_471"` keeps the WRF
v4.6.1/v4.7.1 kernel byte for byte.

This oracle replays the scheme as **the native WRF v4.8.0 strict referee's own
machine code**. `build.sh` links `run_gwdo_kim480.F90` against the object files
of `WRF-4.8.0-serial-pristine`: `phys/module_bl_gwdo.o`,
`phys/physics_mmm/bl_gwdo.o`, `phys/ccpp_kind_types.o` and
`share/module_model_constants.o`. `OBJECTS.sha256` pins those object files and
the sources they were built from. The referee recipe is GNU Fortran 13.3.0 and
glibc 2.39, with FCOPTIM `-O2 -fno-fast-math -ffp-contract=off
-fno-tree-vectorize`, run in the container `localhost/wrfref2404:1` on a development machine.

`build.sh` does the same for `WRF-4.7.1-serial-pristine`, linked with the
v4.6.1 oracle's driver. That gives a two-sided replay of the frozen kernel on
the same columns. The drivers only read inputs and copy arrays.

## Inputs (`make_inputs.py`)

- **`synthetic`**: two sets:
  - the 149 regime columns of the v4.6.1 oracle on 50 and 90 levels;
  - the same regimes on 60 levels topped at 3 hPa, which reaches the
    mesospheric limiter.

  ELVMAX is added to every column: the terrain plus 0.5-4.5 VAR, plus three
  overrides (30 m, terrain + 8 km, terrain). Each set runs six (dx, dt) cases
  twice, once with v4.8.0's defaults (2, true) and once with (1, false).
- **`real`**: every column of four WOOF-ArWen HRRR-recipe history frames from
  `datasets/era5-hrrr-arwen-pairs/v1` (3 km, 244 x 234 x 50 over Oklahoma
  and its surroundings):
  - 2023-12-10 18Z and 2023-12-11 06Z;
  - 2024-04-25 18Z and 2024-04-26 00Z.

  The atmosphere is built as the PBL driver hands it to gwdo. VAR, CON and
  OA/OL come from NCEP's HRRR static (`hrrr_geo_em.d01.nc`, nearest cell).
  ELVMAX is the 30" terrain's maximum over each cell, as WPS v4.8 writes it.
  There are three cases: 3 km / 20 s at the defaults and at (1, false), and
  9 km / 54 s at the defaults.

## Run (a development machine builds; any card for the grade)

```bash
python tools/gwdo_kim_wrf480_oracle/make_inputs.py synthetic $D/inputs
python tools/gwdo_kim_wrf480_oracle/make_inputs.py real $D/inputs \
    --wrfout .../wrfout_d01_2023-12-10_18_00_00 [...] \
    --static-grid .../static/grid.nc \
    --hrrr-geo ~/WPS_GEOG/static_sources/hrrr-conus-v4/hrrr_geo_em.d01.nc \
    --dem ~/WPS_GEOG/gwd_dem_30s.npy
podman run --rm -v $HOME/wrf480-port:/w -v $TREE:/t -v $D:/d \
    localhost/wrfref2404:1 bash /t/tools/gwdo_kim_wrf480_oracle/build.sh \
    /w/build/WRF-4.8.0-serial-pristine /w/build/WRF-4.7.1-serial-pristine \
    /d/inputs /d/out
GPUWM_WRF_EXACT=1 python tools/gwdo_kim_wrf480_oracle/compare_gpu.py \
    $D/inputs $D/out [--write-fixture tests/data/oracles/gwdo_kim480]
python tools/gwdo_kim_wrf480_oracle/engine_smoke.py OUT.npz --gwd-opt 1 \
    [--gwdo-version wrf_471]        # run from each tree's root, then --compare
```

`compare_gpu.py` exits 0 only when every word of every set is bit-identical
for both kernels. It also times one launch of each kernel over the largest
set.
