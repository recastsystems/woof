"""Static and cold-start surface records the WRF-file door takes by name.

Three groups of real.exe output used to stop the door as "unmapped WRF
variable(s)" although WRF itself either never reads them under the options
this door admits, or reads them only as static data this module hands to
the physics (combo sweep round 2, 2026-10-07: 30 urban and Noah-mosaic
runs, 16 GSL drag runs, 14 topo_wind / gwd_opt runs).

* Noah mosaic cold-start state (``sf_surface_mosaic = 1``).  real.exe
  writes the tile arrays as zeros; on a non-restart launch WRF's
  ``lsm_mosaic_init`` (module_sf_noahdrv.F:4948-5350, under
  ``IF(.not.restart)``) rebuilds LANDUSEF2 and MOSAIC_CAT_INDEX from
  LANDUSEF and every ``*_MOSAIC`` carrier from the grid-cell state, which is
  what :func:`woof.core.noah_mosaic_door.attach_wrfinput_noah_mosaic` does
  here.  The file's words are never read.
* Urban morphology (``sf_urban_physics > 0``).  ``urban_var_init``
  (module_sf_urban.F:2555-3050) keeps the gridded morphology only on a city
  column whose BUILD_HEIGHT (HGT_URB2D) is positive, and resets LP, LB,
  HGT, MH, STDH, LF and HI to zero everywhere else.  The port's
  ``urban_var_init_host`` takes that zero branch on every column, so the
  door accepts the morphology records and refuses, by name, a file that
  carries a positive BUILD_HEIGHT on a city column
  (:func:`require_table_urban_morphology`).  AHE, Z0_URB2D and LF_URB2D_S
  are read only by ``distributed_ahe_opt > 0`` and
  ``slucm_distributed_drag``, both refused at the namelist door.
* Sub-grid orography statistics (``topo_wind``, ``gwd_opt``).  real.exe
  copies them from geo_em into wrfinput; :func:`terrain_drag_static` reads
  the ones the selected options need and hands them to the terrain drag as
  its static field set.
"""

from __future__ import annotations

from woof.config import ConfigurationRefused

from types import MappingProxyType
from typing import Mapping

import numpy as np

MOSAIC_COLD_START_WRFINPUT = (
    "LANDUSEF2", "MOSAIC_CAT_INDEX", "TSK_MOSAIC", "QSFC_MOSAIC",
    "TSLB_MOSAIC", "SMOIS_MOSAIC", "SH2O_MOSAIC", "CANWAT_MOSAIC",
    "SNOW_MOSAIC", "SNOWH_MOSAIC", "SNOWC_MOSAIC",
)
_MOSAIC_REASON = (
    "Noah mosaic cold-start carrier; real.exe writes zeros and WRF's "
    "lsm_mosaic_init rebuilds it on a non-restart launch "
    "(module_sf_noahdrv.F:4948-5350), as the mosaic door does here")

URBAN_MORPHOLOGY_WRFINPUT = (
    "BUILD_AREA_FRACTION", "BUILD_HEIGHT", "BUILD_SURF_RATIO",
    "HEIGHT_HISTOGRAMS", "MH_URB2D", "STDH_URB2D", "LF_URB2D",
)
_URBAN_MORPHOLOGY_REASON = (
    "gridded urban morphology; urban_var_init keeps it only on a city column "
    "with BUILD_HEIGHT > 0 and zeroes it elsewhere (module_sf_urban.F:2555-"
    "3050); the door refuses a positive BUILD_HEIGHT on a city column")
URBAN_UNREAD_WRFINPUT = ("AHE", "Z0_URB2D", "LF_URB2D_S")
_URBAN_UNREAD_REASON = (
    "read only by distributed_ahe_opt > 0 or slucm_distributed_drag, which "
    "the namelist door refuses; WRF does not read it under the admitted "
    "options")


def surface_static_dispositions(cfg) -> Mapping[str, str]:
    """The records above that ``cfg``'s surface options make WRF ignore."""
    table: dict[str, str] = {}
    if cfg is None:
        return MappingProxyType(table)
    if int(getattr(cfg, "sf_surface_mosaic", 0) or 0) == 1:
        table.update(dict.fromkeys(MOSAIC_COLD_START_WRFINPUT, _MOSAIC_REASON))
    if int(getattr(cfg, "sf_urban_physics", 0) or 0) > 0:
        table.update(dict.fromkeys(URBAN_MORPHOLOGY_WRFINPUT,
                                   _URBAN_MORPHOLOGY_REASON))
        table.update(dict.fromkeys(URBAN_UNREAD_WRFINPUT, _URBAN_UNREAD_REASON))
    return MappingProxyType(table)


def _read_mass_2d(restored, names) -> dict[str, np.ndarray]:
    """``names`` from the file behind ``restored``, each (ny, nx) float32."""
    from woof import netcdf_bridge
    from woof.ingest.wrfinput import _read_numeric

    shape = (int(restored.dimensions["south_north"]),
             int(restored.dimensions["west_east"]))
    out: dict[str, np.ndarray] = {}
    with netcdf_bridge.open_dataset(restored.path) as dataset:
        for name in names:
            variable = dataset.variables.get(name)
            if variable is None:
                continue
            dimensions = tuple(variable.dimensions)
            if dimensions[:1] == ("Time",):
                dimensions = dimensions[1:]
            value = np.asarray(_read_numeric(variable), dtype=np.float32)
            if dimensions != ("south_north", "west_east") or value.shape != shape:
                raise ConfigurationRefused(
                    f"{restored.path} {name} is {value.shape} on {dimensions}; "
                    f"the mass grid is {shape} on ('south_north', 'west_east'), "
                    "so it would be read onto the wrong columns")
            if not np.all(np.isfinite(value)):
                raise ConfigurationRefused(f"{restored.path} {name} holds non-finite words")
            out[name] = value
    return out


def terrain_drag_static(restored, cfg) -> dict[str, np.ndarray] | None:
    """The static field set the terrain drag reads, from the file itself.

    ``None`` with ``topo_wind = 0`` and ``gwd_opt = 0``.  A statistic the
    options need and the file lacks is left out, so the drag refuses it by
    name (``build_terrain_drag``); WRF itself stops on ``topo_wind = 1``
    without VAR_SSO (start_em.F:222-223).  ``HGT_M`` is the file's HGT, the
    column check ``_attach_terrain_drag`` makes.
    """
    from woof.static.orographic import drag_static_fields, gwdo_version_of

    topo_wind = int(getattr(cfg, "topo_wind", 0) or 0)
    gwd_opt = int(getattr(cfg, "gwd_opt", 0) or 0)
    if topo_wind == 0 and gwd_opt == 0:
        return None
    static = _read_mass_2d(restored, drag_static_fields(
        topo_wind, gwd_opt, gwdo_version_of(cfg)))
    static["HGT_M"] = np.asarray(restored.raw["HGT"], dtype=np.float32)
    return static


def require_table_urban_morphology(restored, utype) -> None:
    """Refuse a file whose gridded morphology WRF would run on a city column.

    The breakage it prevents: WRF's urban_var_init keeps BUILD_AREA_FRACTION,
    BUILD_HEIGHT, BUILD_SURF_RATIO and the height histogram (or MH, STDH and
    LF for the single-layer canopy) on every city column whose BUILD_HEIGHT
    is positive; the port's urban init takes the URBPARM table morphology
    there, so those columns would run a different building geometry from
    WRF's with nothing said.  Zero words, real.exe's output without gridded
    urban data, are WRF's table branch and pass.
    """
    height = _read_mass_2d(restored, ("BUILD_HEIGHT",)).get("BUILD_HEIGHT")
    if height is None:
        return
    city = np.asarray(utype) > 0
    gridded = city & (height > np.float32(0.0))
    if np.any(gridded):
        j, i = (int(v[0]) for v in np.nonzero(gridded))
        raise ConfigurationRefused(
            f"{restored.path} carries gridded urban morphology on "
            f"{int(gridded.sum())} city column(s) (BUILD_HEIGHT "
            f"{float(height[j, i]):g} m at j={j}, i={i}).  WRF's urban_var_init "
            "keeps it there (module_sf_urban.F:2555-3050) and this port's urban "
            "init takes the URBPARM table morphology, so those columns would "
            "run a different building geometry from WRF's.  Use a wrfinput "
            "made without gridded urban data, or a prepared forecast.")


__all__ = [
    "MOSAIC_COLD_START_WRFINPUT", "URBAN_MORPHOLOGY_WRFINPUT",
    "URBAN_UNREAD_WRFINPUT", "require_table_urban_morphology",
    "surface_static_dispositions", "terrain_drag_static",
]
