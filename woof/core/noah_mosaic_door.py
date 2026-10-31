"""Attach Noah mosaic tiles after each door finishes its surface initialization.

WRF v4.7.1 builds the tile state in ``phy_init``, after ``LSMINIT``
(module_physics_init.F:3366-3398), from the grid state the door has just
written.  Every woof door that can carry LANDUSEF calls
:func:`attach_noah_mosaic_to_driver` as its LAST surface write, so the tiles
are copies of exactly the state the dominant-category column would have
started from.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from woof.core.noah_mosaic import MosaicCategories


def mosaic_xice_threshold(fractional_seaice: bool) -> float:
    """WRF's XICE_THRESHOLD for a ``fractional_seaice`` setting.

    lsm_mosaic_init derives it itself (module_sf_noahdrv.F:5031-5035) and the
    surface driver hands lsm_mosaic the same value, so the tile elimination at
    initialisation and the land / sea-ice split every step agree.
    """
    return 0.02 if fractional_seaice else 0.5


@dataclass(frozen=True)
class NoahMosaicSetup:
    """What a driver needs to run mosaic Noah; rebuilt by the door on resume."""

    mosaic_cat: int
    categories: MosaicCategories
    #: XICE at or above this is sea ice for the tile loop.  The same value
    #: :func:`woof.core.noah_mosaic.lsm_mosaic_init` eliminated ice tiles by.
    xice_threshold: float


def attach_noah_mosaic_to_driver(driver, cfg, *, landusef, processed,
                                 landuse_attrs, fractional_seaice: bool,
                                 landmask=None):
    """Build the tile state on ``driver``; a no-op when mosaic is off.

    ``processed`` says whether ``landusef`` already carries real.exe's edits
    (a wrfinput's LANDUSEF does; geogrid/metgrid fractions do not).
    ``fractional_seaice`` is the setting THIS door initialised its land use
    with, so the tiles and the categories the driver runs agree about which
    cells are sea ice.
    """
    if getattr(cfg, "sf_surface_mosaic", 0) == 0:
        return
    from woof.config import validate_noah_mosaic_config
    validate_noah_mosaic_config(cfg)
    if landusef is None:
        raise ValueError("sf_surface_mosaic=1 requires LANDUSEF at this door; "
                         "without fractions the run would silently integrate "
                         "the dominant category")
    from woof.core.noah_mosaic import (
        attach_noah_mosaic, load_mosaic_categories, real_exe_landusef)

    categories = load_mosaic_categories(
        str(landuse_attrs["MMINLU"]), isurban=_door_isurban(landuse_attrs),
        isice=int(landuse_attrs["ISICE"]),
        iswater=int(landuse_attrs["ISWATER"]))
    if not processed:
        xice = driver.fields["xice"]
        xice = xice.get() if hasattr(xice, "get") else xice
        landusef = real_exe_landusef(
            landusef, landmask=landmask, xice=xice,
            iswater=categories.iswater, islake=int(landuse_attrs["ISLAKE"]),
            isice=categories.isice, fractional_seaice=bool(fractional_seaice))
    attach_noah_mosaic(
        driver.fields, landusef=landusef, mosaic_cat=cfg.mosaic_cat,
        categories=categories, fractional_seaice=bool(fractional_seaice),
        lucats=driver.noah_params.lucats,
        urban=int(getattr(cfg, "sf_urban_physics", 0)) == 1)
    driver.noah_mosaic = NoahMosaicSetup(
        cfg.mosaic_cat, categories,
        mosaic_xice_threshold(bool(fractional_seaice)))
    if cfg.mosaic_urban_canopy == "every_tile":
        apply_every_tile_urban_canopy(driver, cfg, fractional_seaice)


def apply_every_tile_urban_canopy(driver, cfg, fractional_seaice: bool) -> None:
    """The town rule (``mosaic_urban_canopy = "every_tile"``) on a driver
    whose urban state and tiles are built.

    Only FRC_URB2D moves: the urban type map, and with it the 10 m wind and
    surface-layer overrides WRF applies on dominant-urban cells
    (module_surface_driver.F:3001-3021), stay WRF's.  The default rule
    ("dominant") never reaches here and leaves urban_var_init's fractions.
    """
    urban = getattr(driver, "urban", None)
    if urban is None or int(getattr(urban, "option", 0)) != 1:
        raise ValueError(
            "mosaic_urban_canopy = 'every_tile' reached a domain with no "
            "single-layer urban canopy state (sf_urban_physics = 1): the town "
            "fractions would have no canopy to run")
    from woof.core.noah_mosaic import (every_tile_urban_fraction,
                                        mosaic_tile_utype_lookup)

    def host(a):
        return a.get() if hasattr(a, "get") else np.asarray(a)

    f = driver.fields
    land = ((host(f["xland"]) < np.float32(1.5))
            & (host(f["xice"]) < np.float32(
                mosaic_xice_threshold(bool(fractional_seaice)))))
    frc = every_tile_urban_fraction(
        host(urban.fields["frc_urb2d"]),
        utype_urb2d=host(urban.fields["utype_urb2d"]),
        mosaic_cat_index=host(f["mosaic_cat_index"]),
        landusef2=host(f["landusef2"]), land=land,
        utype_lookup=mosaic_tile_utype_lookup(
            driver.noah_mosaic.categories, int(cfg.use_wudapt_lcz)),
        frc_table=urban.params.FRC_URB_TBL)
    target = urban.fields["frc_urb2d"]
    if hasattr(target, "get"):
        import cupy as cp
        target[...] = cp.asarray(frc)
    else:
        target[...] = frc


def _door_isurban(landuse_attrs) -> int:
    """ISURBAN from the door's land-use identity, else the Noah launcher's.

    Older prepared identities, and the HRRR prepared-tree door, carry no
    ISURBAN; the dominant-category column then takes it from the land-use
    dataset (:func:`woof.core.noah.launch_noah` with ``isurban=None`` reads
    :data:`woof.core.noah.NOAH_LANDUSE_CATEGORY_IDS`), and the tiles must
    use the same category.  Named breakage: since launch_noah's default
    became ``None`` (30f396c9f6), reading that default here raised
    ``int() argument ... not 'NoneType'`` and every ``sf_surface_mosaic = 1``
    run on the HRRR route failed before its first step.
    """
    if "ISURBAN" in landuse_attrs:
        return int(landuse_attrs["ISURBAN"])
    from woof.core.noah import NOAH_LANDUSE_CATEGORY_IDS
    mminlu = str(landuse_attrs["MMINLU"]).strip()
    try:
        return int(NOAH_LANDUSE_CATEGORY_IDS[mminlu][0])
    except KeyError:
        raise ValueError(
            f"Noah mosaic has no ISURBAN for land-use dataset {mminlu!r}: add "
            "its pair to woof.core.noah.NOAH_LANDUSE_CATEGORY_IDS") from None


def attach_wrfinput_noah_mosaic(driver, cfg, restored, *,
                                fractional_seaice: bool):
    """Read tile fractions on demand through Rust, preserving the off reader.

    real.exe already edited wrfinput LANDUSEF; running its edits twice would
    change the tile ranking before lsm_mosaic_init
    (module_physics_init.F:3366-3398).
    """
    from woof import netcdf_bridge
    from woof.ingest.wrfinput import _read_numeric

    landusef = restored.raw.get("LANDUSEF")
    attrs = restored.global_attributes
    if landusef is None:
        with netcdf_bridge.open_dataset(restored.path) as dataset:
            variable = dataset.variables.get("LANDUSEF")
            if variable is not None:
                dimensions = tuple(variable.dimensions)
                if dimensions and dimensions[0] == "Time":
                    dimensions = dimensions[1:]
                # real.exe writes land_cat_stag (Registry stagger Z; see
                # woof.ingest.ruc_mosaic), WPS's geo_em land_cat.
                if dimensions not in (
                        ("land_cat_stag", "south_north", "west_east"),
                        ("land_cat", "south_north", "west_east")):
                    raise ValueError("LANDUSEF has wrong axes; tile fractions "
                                     "would be assigned to the wrong cells")
                landusef = _read_numeric(variable)
    attach_noah_mosaic_to_driver(
        driver, cfg, landusef=landusef, processed=True, landuse_attrs=attrs,
        fractional_seaice=fractional_seaice)
