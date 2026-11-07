"""Pinned WRF v4.6.1 contract for the RUC LSM lane, at its ORACLE geometry.

This module records source identity, table identity, geometry, persistent
Registry state, and the first supported namelist-option identity.  It does
not make ``sf_surface_physics=3`` executable by itself.

"Nine-level" is still the right word for what is PINNED here and for
:data:`CONTRACT_ID`: every asset in :data:`REFERENCE_ASSETS` was generated at
``nzs = 9`` and each ``run_*.F90`` harness hardcodes it, so the oracle this
contract binds has exactly one geometry.  It is no longer the right word for
what the forecast column CAN run.  :data:`WRF_SUPPORTED_NUM_SOIL_LAYERS` is
the pair WRF tabulates, :data:`NUM_SOIL_LAYERS` is the one woof has an
oracle for, and the difference between those two facts is the whole subject
of ``docs/wrf_ruc_runtime_admission.md``.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from woof.core.noahmp_mynn_contract import (
    ReferenceAsset,
    WRF_REFERENCE_COMMIT,
    WRF_REFERENCE_VERSION,
)
from woof.ingest.ruc_soil import RUC_LEVEL_DEPTHS_M as _RUC_LEVEL_DEPTHS_M
from woof.core.ruc_spp import RUC_SPP_VALUES


CONTRACT_ID = "wrf-v4.6.1-ruc3-nine-level-defaults-v1"
SF_SURFACE_PHYSICS = 3
NUM_SOIL_LAYERS = 9
#: Every soil geometry WRF's RUC defines, read from the level-depth table
#: rather than written again.  ``share/module_check_a_mundo.F:3574-3581``
#: resolves ``num_soil_layers`` to 6 or to 9 for ``sf_surface_physics=3`` and
#: coerces every other request to 6; ``init_soil_depth_3`` has a ``zs`` table
#: for exactly those two lengths and leaves ``zs`` UNINITIALISED for any
#: other, which is why the set is closed and not a range.
WRF_SUPPORTED_NUM_SOIL_LAYERS = tuple(sorted(_RUC_LEVEL_DEPTHS_M))


# share/module_soil_pre.F:1153-1194 (``init_soil_depth_3``).  These are soil
# *level* (node) depths, not layer-centre depths.  RUC deliberately includes a
# zero-depth surface level.
#
# NOT transcribed here.  ``init_soil_depth_3`` tabulates one ``zs`` array per
# admitted geometry and woof holds that table in exactly one place --
# :data:`woof.ingest.ruc_soil.RUC_LEVEL_DEPTHS_M`, the module that is
# oracle-matched against the routine.  A second transcription in this file is
# how the six-level geometry came to exist in the ingest path and not in the
# forecast path, which is the defect this lane is retiring: a geometry must be
# a row in one table, never a constant that some modules happen to carry.
RUC_SOIL_LEVELS_BY_COUNT: Mapping[int, tuple[float, ...]] = MappingProxyType({
    count: tuple(float(depth) for depth in depths)
    for count, depths in sorted(_RUC_LEVEL_DEPTHS_M.items())
})

#: The level depths of :data:`NUM_SOIL_LAYERS`, the geometry this contract
#: pins.  An alias for one row of the table above, kept because the contract
#: receipt names a single geometry; it is not a second table.
RUC_SOIL_LEVELS_M = RUC_SOIL_LEVELS_BY_COUNT[NUM_SOIL_LAYERS]


RUC_SOURCE_ASSETS = (
    ReferenceAsset(
        "phys/module_sf_ruclsm.F",
        288_509,
        "3265f810d08dcbddfaf198371dc7f652e78e8d3a788f703a515c555a3bbb2a12",
    ),
    ReferenceAsset(
        "share/module_soil_pre.F",
        139_826,
        "f981a2cce2cd1a8ca5b8ced100bf2d34eb2725d1d0bc7c42daf05fd05aa2c074",
    ),
    ReferenceAsset(
        "Registry/Registry.EM_COMMON",
        436_138,
        "28cf7a2d369e898217d7231f4ed19310af7f0bec920f3cb57831e3dc4c80beec",
    ),
)

RUC_TABLE_ASSETS = (
    ReferenceAsset(
        "run/VEGPARM.TBL",
        23_037,
        "ed5478afbe49af51492256c1eb6cf88b3948590308525b32df1ddec28687b40e",
    ),
    ReferenceAsset(
        "run/SOILPARM.TBL",
        6_557,
        "1e2275a32d8cd3b48ca693d22c0816df0013f83b6594ac632716361db337d58f",
    ),
    ReferenceAsset(
        "run/GENPARM.TBL",
        261,
        "9c02832a0e4a2ecaf47fcee485539aad95cd732c379c5c258161a88eb3d25ea2",
    ),
)

PINNED_REFERENCE_ASSETS = RUC_SOURCE_ASSETS + RUC_TABLE_ASSETS


# Registry/Registry.EM_COMMON:3147, normalized exactly as the package line.
RUC_PACKAGE_STATE_FIELDS = (
    "smfr3d",
    "keepfr3dflag",
    "soilt1",
    "rhosnf",
    "snowfallac",
    "precipfr",
    "acrunoff",
)
RUC_PACKAGE_STATE_SHA256 = (
    "e3b868534f12165d78ee03ea25735d6b934dffb81ab8db412540b6134b750b8f"
)


# Direct RUC controls from Registry.EM_COMMON:2535-2537 and the stochastic
# physics controls documented in run/README.namelist:1218-1220.  The first
# executable defaults retain WRF dominant categories; mosaics use source
# category fractions. Six levels and the separately versioned hydraulic SPP
# operator are admitted.
RUC_NAMELIST_DEFAULTS: Mapping[str, int] = MappingProxyType({
    "num_soil_layers": 9,
    "mosaic_lu": 0,
    "mosaic_soil": 0,
    "flag_sm_adj": 0,
    "spp_lsm": 0,
})

#: Keys whose override is admitted against a SET rather than a single
#: validated default. WRF defines two
#: RUC geometries and woof's forecast column now compiles at both, so pinning
#: it to one would be woof's pin, not WRF's. Mosaic modes admit 0 and 1;
#: enabled modes require source fractions. SPP restores the historical
#: hydraulic-conductivity operator under its own source identity. Other keys
#: retain their defaults.
#:
#: Admitted is not validated.  Six levels has no WRF forecast oracle; what it
#: has is a compiled column, host/device agreement and the runtime warning
#: woof.physics_compat raises.  See docs/wrf_ruc_runtime_admission.md.
RUC_NAMELIST_ADMITTED: Mapping[str, tuple[int, ...]] = MappingProxyType({
    "num_soil_layers": WRF_SUPPORTED_NUM_SOIL_LAYERS,
    "mosaic_lu": (0, 1),
    "mosaic_soil": (0, 1),
    "spp_lsm": RUC_SPP_VALUES,
})


RUC_INIT_INPUT_FIELDS = (
    "tslb",
    "smois",
    "isltyp",
    "ivgtyp",
    "xice",
)
RUC_INIT_OUTPUT_FIELDS = (
    "sh2o",
    "smfr3d",
    "mavail",
    "znt",
)

RUC_INIT_ORACLE_ASSET = ReferenceAsset(
    "init.csv",
    3_614,
    "f953b2cbe4990374101d427ee13b5f085cc763ef5a0b4a486f800b2a288208bf",
)
RUC_INIT_ORACLE_CASES = (
    "warm_land",
    "frozen_land",
    "water",
    "sea_ice",
)
RUC_SURFACE_ORACLE_ASSET = ReferenceAsset(
    "soilvegin.csv",
    1_600,
    "594bfe285c32c6bb2dfab64b3e272aed0ee024bfdf880d32fa8222489ef08cc2",
)
RUC_SURFACE_ORACLE_CASES = (
    "evergreen_cold",
    "evergreen_warm",
    "crop_midseason",
    "water_preserve_znt",
    "lai2d_preserve",
    "grass_short_season",
)
#: The two mosaic fixtures were rebuilt on test host (development branch) with
#: gfortran 13.3.0 on glibc 2.39, the toolchain of every other RUC WRF
#: oracle, by tools/ruc_mosaic_wrf461_oracle/build.sh.  The earlier build
#: (gfortran 15.2 on another C library) recorded that library's float32
#: words, which WOOF's libm does not reproduce (grdflx 1-2 ULP).
RUC_MOSAIC_SURFACE_ORACLE_ASSET = ReferenceAsset(
    "mosaic_surface.csv",
    2_967,
    "0308b9ec425db77501fe2687f81b404d8f2fedcd0d45cb18c82389e147421787",
)
RUC_MOSAIC_DRIVER_ORACLE_ASSET = ReferenceAsset(
    "mosaic_driver.csv",
    583_954,
    "16698d9e498b0933feca6afbf215fce10a80d1c5c62b765302c56a279100a0dc",
)

RUC_SOILPROP_ORACLE_ASSET = ReferenceAsset(
    "soilprop.csv",
    10_137,
    "a8c08e47f09b62d9a07635435455ea06f181ec472cc2850a32e2d7e1ca239040",
)
RUC_SOILPROP_ORACLE_CASES = (
    "warm_wet",
    "warm_dry",
    "frozen",
    "deep_frozen_keep",
)
RUC_TRANSF_ORACLE_ASSET = ReferenceAsset(
    "transf.csv",
    7_499,
    "c1eb25f901611721e1dcd66e743557900d32a6406fa43dd50e630b9dba4a6003",
)
RUC_TRANSF_ORACLE_CASES = (
    "wet_forest",
    "wilt_dark_grass",
    "mixed_hot_crop",
    "bare_high_sun",
)
RUC_SOILMOIST_ORACLE_ASSET = ReferenceAsset(
    "soilmoist.csv",
    16_834,
    "8a531d722cf8e3e18f7cf9556b7e62897953a446e99708d5901d98c2480d5751",
)
RUC_SOILMOIST_ORACLE_CASES = (
    "rain_wet",
    "dry_evap",
    "dew",
    "frozen_melt",
)
RUC_SOILTEMP_ORACLE_ASSET = ReferenceAsset(
    "soiltemp.csv",
    17_365,
    "caecf80b6d04c1758de180c8cf6c7a801bf2518f73919a430ca7de2bc5906b98",
)
RUC_SOILTEMP_ORACLE_CASES = (
    "moist_saturated",
    "dry_second_solve",
    "warm_rain",
    "humid_condensing",
)
RUC_TBQ_ORACLE_ASSET = ReferenceAsset(
    "tbq.csv",
    84_334,
    "22d7c553692ac1b6b277d1498fb48ad3cd4de79e0d9460ae8d9f311aed31cf12",
)
RUC_SOIL_ORACLE_ASSET = ReferenceAsset(
    "soil.csv",
    32_332,
    "efd113c359c216762ad4c178e90b8267790adc7c984e5f55184ee7cb6469e22c",
)
RUC_SOIL_ORACLE_CASES = (
    "warm_forest_rain",
    "dry_grass",
    "frozen_crop_rain",
    "humid_bare_dew",
)
RUC_STEP_ORACLE_ASSET = ReferenceAsset(
    "step.csv",
    14_042,
    "acc2e33eb2ed79c5381a2aeff256860d649ed5b7945e124dd925771a897bd49a",
)
RUC_STEP_ORACLE_CASES = (
    "warm_rain",
    "cold_snow",
    "water",
    "sea_ice",
)


def _sha256_file(path: Path, *, block_bytes: int = 8 * 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def validate_ruc_reference_assets(
    wrf_root: str | Path,
    assets: tuple[ReferenceAsset, ...] = PINNED_REFERENCE_ASSETS,
) -> tuple[ReferenceAsset, ...]:
    """Fail closed on a non-canonical WRF v4.6.1 reference tree."""

    root = Path(wrf_root)
    for asset in assets:
        path = root / asset.relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing WRF RUC reference asset {path}")
        if path.stat().st_size != asset.bytes:
            raise ValueError(
                f"WRF RUC reference asset {path} has {path.stat().st_size} "
                f"bytes; expected {asset.bytes}"
            )
        digest = _sha256_file(path)
        if digest != asset.sha256:
            raise ValueError(
                f"WRF RUC reference asset {path} SHA-256 {digest}; "
                f"expected {asset.sha256}"
            )
    return assets


def resolve_default_ruc_options(
    supplied: Mapping[str, int] | None = None,
) -> Mapping[str, int]:
    """Resolve omissions and reject every not-yet-validated RUC mode."""

    supplied = supplied or {}
    unknown = sorted(set(supplied) - set(RUC_NAMELIST_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown RUC options: {', '.join(unknown)}")
    changed: list[str] = []
    for key, expected in RUC_NAMELIST_DEFAULTS.items():
        if key not in supplied:
            continue
        actual = supplied[key]
        admitted = RUC_NAMELIST_ADMITTED.get(key)
        if admitted is not None:
            # A SET, not a single value: WRF tabulates two RUC soil
            # geometries and the forecast column compiles at both.
            if type(actual) is not int or actual not in admitted:
                changed.append(f"{key}={actual!r} (admitted {admitted!r})")
            continue
        if type(actual) is not int or actual != expected:
            changed.append(
                f"{key}={actual!r} (validated default {expected!r})"
            )
    if changed:
        raise ValueError(
            "RUC first executable lane is pinned to WRF v4.6.1 defaults; "
            "unsupported overrides: " + "; ".join(changed)
        )
    resolved = dict(RUC_NAMELIST_DEFAULTS)
    resolved.update(supplied)
    return MappingProxyType(resolved)


def ruc_reference_contract_receipt(wrf_root: str | Path) -> dict[str, object]:
    """Validate the authority tree and return a serializable receipt."""

    root = Path(wrf_root).resolve()
    assets = validate_ruc_reference_assets(root)
    return {
        "schema": "gpuwm.ruc-reference-contract/v1",
        "status": "PASS",
        "contract_id": CONTRACT_ID,
        "wrf_version": WRF_REFERENCE_VERSION,
        "wrf_commit": WRF_REFERENCE_COMMIT,
        "root": str(root),
        "selector": SF_SURFACE_PHYSICS,
        "num_soil_layers": NUM_SOIL_LAYERS,
        "soil_level_depths_m": list(RUC_SOIL_LEVELS_M),
        "package_state_fields": list(RUC_PACKAGE_STATE_FIELDS),
        "package_state_sha256": RUC_PACKAGE_STATE_SHA256,
        "options": dict(RUC_NAMELIST_DEFAULTS),
        "assets": [
            {
                "relative_path": asset.relative_path,
                "bytes": asset.bytes,
                "sha256": asset.sha256,
            }
            for asset in assets
        ],
    }


__all__ = [
    "CONTRACT_ID",
    "NUM_SOIL_LAYERS",
    "PINNED_REFERENCE_ASSETS",
    "RUC_INIT_INPUT_FIELDS",
    "RUC_INIT_ORACLE_ASSET",
    "RUC_INIT_ORACLE_CASES",
    "RUC_INIT_OUTPUT_FIELDS",
    "RUC_NAMELIST_DEFAULTS",
    "RUC_PACKAGE_STATE_FIELDS",
    "RUC_PACKAGE_STATE_SHA256",
    "RUC_SOIL_LEVELS_BY_COUNT",
    "RUC_SOIL_LEVELS_M",
    "RUC_SOILPROP_ORACLE_ASSET",
    "RUC_SOILPROP_ORACLE_CASES",
    "RUC_SOILMOIST_ORACLE_ASSET",
    "RUC_SOILMOIST_ORACLE_CASES",
    "RUC_SOIL_ORACLE_ASSET",
    "RUC_SOIL_ORACLE_CASES",
    "RUC_SOILTEMP_ORACLE_ASSET",
    "RUC_SOILTEMP_ORACLE_CASES",
    "RUC_SOURCE_ASSETS",
    "RUC_SURFACE_ORACLE_ASSET",
    "RUC_SURFACE_ORACLE_CASES",
    "RUC_STEP_ORACLE_ASSET",
    "RUC_STEP_ORACLE_CASES",
    "RUC_TABLE_ASSETS",
    "RUC_TBQ_ORACLE_ASSET",
    "RUC_TRANSF_ORACLE_ASSET",
    "RUC_TRANSF_ORACLE_CASES",
    "SF_SURFACE_PHYSICS",
    "WRF_SUPPORTED_NUM_SOIL_LAYERS",
    "resolve_default_ruc_options",
    "ruc_reference_contract_receipt",
    "validate_ruc_reference_assets",
]
