"""Setting names: ``WOOF_*`` is the spelling, ``GPUWM_*`` still works.

The engine was published as ArWen, whose settings were spelled ``GPUWM_*``.
Every setting the Python side reads is spelled ``WOOF_*`` now.  So that no
existing script or shell profile breaks, :func:`apply` runs once when
``woof`` is imported and copies an old ``GPUWM_X`` onto ``WOOF_X`` when
``WOOF_X`` is not set.

A few names are read by the native (Rust) programs themselves, which keep
their spelling so their bytes do not change.  For those the direction is the
other way: a ``WOOF_X`` setting is copied onto the ``GPUWM_X`` name the
program reads, so both spellings reach it.
"""
from __future__ import annotations

import os

#: Settings a native program reads under its own name.
NATIVE_NAMES = frozenset({
    'GPUWM_CASE_DATA_ROOT',
    'GPUWM_EARTH2_TEST_PYTHON',
    'GPUWM_ENGINE_VERSION',
    'GPUWM_EXTREME_INTERVAL_SECONDS',
    'GPUWM_FETCH_NO_IDLE_CONNECTIONS',
    'GPUWM_FIRE_INDICES_V1',
    'GPUWM_FIRE_OBSERVATIONS',
    'GPUWM_GFS_BRIDGE_THREADS',
    'GPUWM_GFS_SNOW_CONVENTION',
    'GPUWM_GLIBC_TRIG_FLT32_CUH',
    'GPUWM_GLOBE_ZLIB',
    'GPUWM_GRIB1_BRIDGE_THREADS',
    'GPUWM_GRID_LABEL',
    'GPUWM_GRID_NAME',
    'GPUWM_HIGHRES_DEM_CACHE',
    'GPUWM_HISTORY_COMPRESSION',
    'GPUWM_HISTORY_COMPRESSION_LEVEL',
    'GPUWM_HISTORY_COMPRESSION_THREADS',
    'GPUWM_HISTORY_PRESET',
    'GPUWM_HOST_RATES',
    'GPUWM_HYDROFABRIC_REV',
    'GPUWM_HYDRO_WORLD_TILES',
    'GPUWM_INITIAL_CONDITION_CYCLE',
    'GPUWM_INITIAL_CONDITION_SOURCE',
    'GPUWM_INITIAL_FORECAST_LEAD_HOURS',
    'GPUWM_LAKE_SUPPORT',
    'GPUWM_LAM_ORACLE_DIR',
    'GPUWM_MAPPED_ENGINE',
    'GPUWM_MAPPED_ENGINE_LANES',
    'GPUWM_MAPPED_ENGINE_MEMORY_BUDGET_BYTES',
    'GPUWM_MAPPED_ENGINE_THREADS',
    'GPUWM_MC_STRICT',
    'GPUWM_MESH_PROFILE',
    'GPUWM_MODEL_GAUNTLET_STAGING',
    'GPUWM_MODEL_LABEL',
    'GPUWM_MPAS_PUBLISHED_GRID',
    'GPUWM_MPAS_REBUILT_GRID',
    'GPUWM_MUTATE_MOIST_N2_FORCE_DRY',
    'GPUWM_NEXTGEN_RESERVOIRS_CUH',
    'GPUWM_NEXTGEN_RESERVOIR_STRICT',
    'GPUWM_NOAHMP_ATAN_GLIBC243',
    'GPUWM_NOAHMP_NIGHT_AGE',
    'GPUWM_NO_LOCAL_GPU',
    'GPUWM_NPMATH_SWEEP',
    'GPUWM_NUMERICS',
    'GPUWM_PORTABLE_LIBM64_CUH',
    'GPUWM_PREPARATION_EOS',
    'GPUWM_PREP_THREADS',
    'GPUWM_REFC_NOTE',
    'GPUWM_RENDER_NETCDF4_DEFLATE',
    'GPUWM_RUC_SNOW_V461',
    'GPUWM_RW_ZARR_WHOLE_CHUNKS',
    'GPUWM_RW_ZARR_WORKERS',
    'GPUWM_SATMEDMF_HOST',
    'GPUWM_SFIRE_LIBM_CUH',
    'GPUWM_SFIRE_PHYS_CUH',
    'GPUWM_SFIRE_RENDERER_CONTROL_BIN',
    'GPUWM_SMAG_DIRECT_W_REFERENCE',
    'GPUWM_SMAG_INDEX32',
    'GPUWM_SNOW_SINCE',
    'GPUWM_SNOW_WRF461',
    'GPUWM_SOILPROP_WRF461',
    'GPUWM_SOUNDINGS_RENDERER',
    'GPUWM_STATIC_LANE1_GOLDENS',
    'GPUWM_STATIC_PARITY_GEOG',
    'GPUWM_STATIC_SEAM_PANIC_CHILD',
    'GPUWM_STATIC_WPS32_OUTPUT',
    'GPUWM_TERRAIN_AUTHORITY',
    'GPUWM_TERRAIN_AUTHORITY_SHA256',
    'GPUWM_TERRAIN_POLICY',
    'GPUWM_URBAN_GROUND_WATER_480',
    'GPUWM_URBAN_ROOT_UPTAKE_480',
    'GPUWM_VERSION',
    'GPUWM_WAVES_DATA_REV',
    'GPUWM_WPS_GEOG',
    'GPUWM_WPS_SMOOTH_ORACLE',
    'GPUWM_WRF_CFL_PROBE',
    'GPUWM_WRF_EXACT',
    'GPUWM_WRF_EXACT_C_ADVECTION',
    'GPUWM_WRF_EXACT_C_BIGSTEP',
    'GPUWM_WRF_EXACT_C_DIFFUSION',
    'GPUWM_WRF_EXACT_DIFFUSION',
    'GPUWM_WRF_EXACT_D_DIAGNOSTICS',
})

_OLD, _NEW = "GPUWM_", "WOOF_"


def apply(environ=None) -> None:
    env = os.environ if environ is None else environ
    for key in [k for k in env if k.startswith(_OLD)]:
        if key in NATIVE_NAMES:
            continue
        env.setdefault(_NEW + key[len(_OLD):], env[key])
    for key in [k for k in env if k.startswith("HEXCORE_")]:
        env.setdefault(_NEW + "HEX_" + key[len("HEXCORE_"):], env[key])
    for name in NATIVE_NAMES:
        twin = _NEW + name[len(_OLD):]
        if twin in env:
            env[name] = env[twin]
    # A setting that kept its GPUWM_ spelling in the code is reached by its
    # WOOF_ spelling too.
    for key in [k for k in env if k.startswith(_NEW) and not k.startswith(_NEW + "HEX_")]:
        env.setdefault(_OLD + key[len(_NEW):], env[key])
