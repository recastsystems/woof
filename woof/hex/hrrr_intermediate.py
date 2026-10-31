"""A projected regional source onto the regular lat-lon WPS intermediate the
init and boundary engines read: ``woof hex intermediate``.

THE GAP THIS CLOSES, MEASURED.  ``docs/source-matrix.md`` (2026-08-24) drove
every registered source through the hex chain and recorded that a Lambert
regional product (HRRR, RAP, RRFS) is refused at TWO independent layers:
the engine's ``met_intermediate`` writer refuses to mint a projected grid as
``iproj 0`` ("would silently mis-georeference every point"), and
``rw_mpas_init`` / ``rw_mpas_lbc`` invert projection code 0 only.  So the
HRRR bytes the engine already fetches, decodes and interpolates for its own
WRF road never reached a hex init.

THE ROAD, and what is borrowed from where.  Everything with numbers in it
is the engine's own, driven, not re-implemented:

* DECODE -- ``hrrr_grib2_bridge`` (Rust, staged by the engine's bundle,
  resolved through ``woof.bridges.resolve_source_decoder``), the same
  fail-closed decoder ``woof prep`` runs, over the same ``wrfnat`` + soil
  pair ``woof fetch --source hrrr`` writes, into the engine's own native
  window format that ``woof.ingest.hrrr`` maps;
* REGRID -- the rw_wps projected-intermediate entry builds the engine's
  FP64 Lambert donor geometry and keeps donors separate from FP32 fractions.
  It calls the selected engine library's exact-donor WPS overlapping-parabolic
  operator. Rust rotates source winds to earth coordinates before movement;
* SOIL -- Rust interpolates the source row's depth nodes to the engine's
  declared layer midpoints in FP64, rounds once to FP32 and moves by nearest;
* WRITE -- Rust writes WPS version-5 records and checks the complete file with
  the engine Rust reader before publishing the per-hour completion marker.

WHAT THE RESULT IS, STATED PLAINLY.  The intermediate is HRRR RESAMPLED onto
a 0.025-degree regular lat-lon grid over the domain (2.8 km meridionally,
finer zonally at mid-latitudes -- the source is not coarsened).  Fifty
hybrid levels are carried as level-indexed slabs with the 3-D ``PRESSURE``
field beside them, which is the MPAS convention for model-level first
guesses (ERA5 ML uses it) and the branch ``rw_mpas_init`` takes when a 3-D
pressure is present.

AN HRRR START IS HRRR'S OWN STATE.  The row carries what HRRR analysed, under
the WPS names the RAP/HRRR Vtable gives them and the RRFS fork's case 7 and
case 9 read (``mpas_init_atm_cases.F:4995-5080`` at 1d6e370e):

* cloud water, rain, cloud ice, snow and graupel (``QC QR QI QS QG``) and
  HRRR's own number concentrations of cloud droplets, cloud ice and rain
  (``QNC QNI QNR``; NCONCD, NCCICE, SPNCR), with its analysed aerosol numbers
  (``QNWFA QNIFA``) when the cycle publishes them -- numbers diagnosed from
  the mass are not HRRR's state;
* the 2 m relative humidity (``RH`` at 200100), the only record case 7 forms
  its first-guess q2 from (without it q2 is zero everywhere);
* the soil as HRRR has it: for a producer whose land surface is RUC
  (``--physics-suite hrrrv5`` and the other RRFS suites) the nine RUC nodes
  themselves (``SOILT000 .. SOILT300``), which the fork copies straight onto
  its nine levels; otherwise the four Noah layers interpolated from them.

The MPAS-A init (``rw_mpas_init``) reads the 2 m humidity, and the
hydrometeor and number records too under the init door's
``--first-guess-hydrometeors`` (default yes, no under strict numerics, where
the init is native case 7's and starts with no condensate).

THE ARBITRARY ACCEPTANCE TEST.  The source is a ROW (:data:`SOURCE_ROWS`):
the engine decoder key, the loader, the level count, the field map, the
soil nodes, the file names ``woof fetch`` writes.  Nothing below this
docstring branches on the string ``hrrr``; a second projected source with an
engine decoder and loader is a second row.  The module carries the name of
the one row it ships because the plan of record names the file that way.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import socket
import sys
from threading import Condition, Event
import time
import warnings
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .errors import MpasPortError

INTERMEDIATE_SCHEMA = "gpuwm-hex.intermediate/v1"
WPS_VERSION = 5
SURFACE_LEVEL = 200100.0
#: Meridional 3 km at the equator, a hair finer than HRRR's 3 km cells, so a
#: 3 km source is never coarsened by the target.
DEFAULT_SPACING_DEG = 0.025
#: Ground kept past the cull cut, so the four-point parabolic stencil and
#: the seven boundary rings never reach the edge of the intermediate.
DEFAULT_MARGIN_KM = 30.0
KM_PER_DEG = 111.195


class IntermediateRefusal(MpasPortError):
    """The intermediate cannot be written, and the message says why."""


# ---------------------------------------------------------------------------
# the source table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceRow:
    """One projected source: how the engine decodes and maps it, what it carries."""

    name: str
    #: The key ``woof.bridges.resolve_source_decoder`` resolves.
    decoder_key: str
    #: The engine module whose loader maps the decoder's output.
    loader_module: str
    #: How the fetch names the two GRIB files per hour: (atmosphere, soil).
    grib_names: tuple[str, str]
    #: 3-D fields, by the engine's window name -> (WPS name, units, description).
    atmosphere_3d: Mapping[str, tuple[str, str, str]]
    #: The 3-D wind pair on the source grid basis: (u, v).
    wind_3d: tuple[str, str]
    #: Surface fields at the WPS surface level tag.
    surface: Mapping[str, tuple[str, str, str, str]]
    #: The 10 m wind pair.
    wind_10m: tuple[str, str]
    #: Categorical surface fields, moved by nearest neighbour.
    nearest: tuple[str, ...]
    #: Soil nodes and the Noah layer names they become.
    soil_temperature: str
    soil_moisture: str
    soil_layer_names: tuple[tuple[str, str], ...]
    levels: int
    map_source: str
    #: How long the decoder holds its output open per hour, for the halo.
    parabolic_halo_cells: int = 2
    #: What a refusal calls the product.
    description: str = ""
    #: The engine's fetch transport, used for a missing hour while decode runs.
    fetch_function: str = ""
    source_grid_function: str = ""
    loader_function: str = ""
    producer_class: str = ""
    decoder_workers_max: int = 1
    decoded_directories: tuple[str, str] = ("", "")
    soil_depth_metadata: tuple[str, str] = ("", "")
    #: Records the decoder writes only on request (its ``--extras`` tables).
    decoder_extras: tuple[str, ...] = ()
    #: Whether the decoder selects the source's analysed aerosol numbers.
    analyzed_aerosol: bool = False
    #: The source's own soil nodes, as (temperature, moisture) WPS names in
    #: node order: the names a first-guess reader knows them by.  Written
    #: instead of the interpolated layers for a producer that reads the
    #: source's nodes (``first_guess_soil`` in its init-producer row).
    soil_node_names: tuple[tuple[str, str], ...] = ()
    #: WPS field names that are never negative (mixing ratios, numbers).
    nonnegative: tuple[str, ...] = ()

    def hour_files(self, root: Path, cycle: datetime, hour: int) -> tuple[Path, Path]:
        atmosphere, soil = self.grib_names
        return (
            root / atmosphere.format(cycle=cycle, hour=hour),
            root / soil.format(cycle=cycle, hour=hour),
        )


SOURCE_ROWS: Mapping[str, SourceRow] = {
    "hrrr": SourceRow(
        name="hrrr",
        decoder_key="hrrr",
        loader_module="woof.ingest.hrrr",
        grib_names=(
            "hrrr.t{cycle:%H}z.wrfnatf{hour:02d}.grib2",
            "hrrr.t{cycle:%H}z.soilf{hour:02d}.grib2",
        ),
        atmosphere_3d={
            "PRES": ("PRESSURE", "Pa", "Pressure"),
            "HGT": ("GHT", "m", "Height"),
            "TT": ("TT", "K", "Temperature"),
            "SPFH": ("SPECHUMD", "kg kg-1", "Specific humidity"),
            "QC": ("QC", "kg kg-1", "Cloud water mixing ratio"),
            "QR": ("QR", "kg kg-1", "Rain water mixing ratio"),
            "QI": ("QI", "kg kg-1", "Cloud ice mixing ratio"),
            "QS": ("QS", "kg kg-1", "Snow mixing ratio"),
            "QG": ("QG", "kg kg-1", "Graupel mixing ratio"),
            "QNC": ("QNC", "kg-1", "Cloud droplet number concentration"),
            "QNI": ("QNI", "kg-1", "Cloud ice number concentration"),
            "QNR": ("QNR", "kg-1", "Rain number concentration"),
        },
        wind_3d=("U_MASS", "V_MASS"),
        surface={
            "T2": ("TT", "K", "Temperature", "parabolic"),
            "Q2": ("SPECHUMD", "kg kg-1", "Specific humidity", "parabolic"),
            "RH2": ("RH", "%", "Relative Humidity", "parabolic"),
            "PSFC": ("PSFC", "Pa", "Surface pressure", "parabolic"),
            "SKINTEMP": ("SKINTEMP", "K", "Skin temperature", "parabolic"),
            "SOILHGT": ("SOILHGT", "m", "Terrain field of source analysis", "parabolic"),
            "SNOW": ("SNOW", "kg m-2", "Water equivalent snow depth", "bilinear"),
            "LANDSEA": ("LANDSEA", "proprtn", "Land/Sea flag (1=land, 0 or 2=sea)", "nearest"),
            "XICE": ("SEAICE", "proprtn", "Ice flag", "nearest"),
        },
        wind_10m=("U10_MASS", "V10_MASS"),
        nearest=("LANDSEA", "XICE"),
        soil_temperature="SOILT",
        soil_moisture="SOILW",
        soil_layer_names=(
            ("ST000010", "SM000010"),
            ("ST010040", "SM010040"),
            ("ST040100", "SM040100"),
            ("ST100200", "SM100200"),
        ),
        levels=50,
        # The label ungrib writes for NCEP's HRRR, which the RRFS fork's case 7 keys RUC's 1 cm node on
        # (mpas_init_atm_cases.F:5658-5661); the grid itself is described by each record's projection.
        map_source="NCEP HRRR Model",
        description="the 3 km Lambert CONUS HRRR (native hybrid wrfnat + wrfprs soil)",
        fetch_function="woof.fetch:fetch_hrrr",
        source_grid_function="hrrr_source_grid",
        loader_function="load_hrrr_pipeline_ready_window",
        producer_class="tools.hrrr_pipeline:HrrrPipelineProducer",
        decoder_workers_max=13,
        decoded_directories=("atmosphere-f{hour:02d}", "soil-f{hour:02d}"),
        soil_depth_metadata=("HRRR_SOIL_NODE_DEPTHS_M", "NOAH_LAYER_MIDPOINTS_M"),
        decoder_extras=("QNC", "QNI", "QNR", "RH2"),
        analyzed_aerosol=True,
        # RUC's nine nodes (0, 1, 4, 10, 30, 60, 100, 160, 300 cm) as the RAP/HRRR Vtable names them.
        soil_node_names=tuple((f"SOILT{cm:03d}", f"SOILM{cm:03d}")
                              for cm in (0, 1, 4, 10, 30, 60, 100, 160, 300)),
        nonnegative=("QC", "QR", "QI", "QS", "QG", "QNC", "QNI", "QNR"),
    ),
}


def source_row(name: str) -> SourceRow:
    key = str(name).strip().lower()
    if key not in SOURCE_ROWS:
        raise IntermediateRefusal(
            f"--source {name!r} is not a projected source this door knows.  "
            f"Known rows: {', '.join(sorted(SOURCE_ROWS))}.  A source is a row "
            f"in woof.hex.hrrr_intermediate.SOURCE_ROWS naming the engine's "
            f"decoder, its loader and its field map; a uniform lat-lon source "
            f"does not need this door at all (docs/source-matrix.md)"
        )
    return SOURCE_ROWS[key]


# ---------------------------------------------------------------------------
# the target grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LatLonTarget:
    """A regular lat-lon grid, south-west corner first, x fastest."""

    south: float
    west: float
    dlat: float
    dlon: float
    nx: int
    ny: int

    @property
    def north(self) -> float:
        return self.south + self.dlat * (self.ny - 1)

    @property
    def east(self) -> float:
        return self.west + self.dlon * (self.nx - 1)

    def latitudes(self) -> np.ndarray:
        return self.south + self.dlat * np.arange(self.ny, dtype=np.float64)

    def longitudes(self) -> np.ndarray:
        return self.west + self.dlon * np.arange(self.nx, dtype=np.float64)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """``(lat, lon)`` as ``(ny, nx)`` arrays, the engine's row-major shape."""

        lat = np.repeat(self.latitudes()[:, None], self.nx, axis=1)
        lon = np.repeat(self.longitudes()[None, :], self.ny, axis=0)
        return lat, lon

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "regular-latlon",
            "south": self.south, "west": self.west, "north": self.north, "east": self.east,
            "dlat": self.dlat, "dlon": self.dlon, "nx": self.nx, "ny": self.ny,
        }


def target_for_cap(
    centre: tuple[float, float], radius_km: float, *,
    margin_km: float = DEFAULT_MARGIN_KM, spacing_deg: float = DEFAULT_SPACING_DEG,
) -> LatLonTarget:
    """The lat-lon box that covers a cap plus a margin, on a fixed spacing."""

    lat, lon = centre
    if not spacing_deg > 0.0:
        raise IntermediateRefusal(f"--spacing-deg {spacing_deg} is not positive")
    reach_km = float(radius_km) + float(margin_km)
    half_lat = reach_km / KM_PER_DEG
    cos_lat = max(math.cos(math.radians(lat)), 0.05)
    half_lon = reach_km / (KM_PER_DEG * cos_lat)
    ny = int(math.ceil(2.0 * half_lat / spacing_deg)) + 1
    nx = int(math.ceil(2.0 * half_lon / spacing_deg)) + 1
    south = round(lat - 0.5 * (ny - 1) * spacing_deg, 6)
    west = round(lon - 0.5 * (nx - 1) * spacing_deg, 6)
    if south < -90.0 or south + (ny - 1) * spacing_deg > 90.0:
        raise IntermediateRefusal(
            f"a {reach_km:g} km reach around {lat:.4f} N crosses a pole; a "
            f"regular lat-lon intermediate cannot carry it"
        )
    return LatLonTarget(south=south, west=west, dlat=spacing_deg, dlon=spacing_deg, nx=nx, ny=ny)


def east_longitude(value: float) -> float:
    """Degrees east, 0 <= value < 360 (the convention of MPAS cell longitudes)."""

    return round(float(value) % 360.0, 6) % 360.0


def west_longitude(value: float) -> float:
    """Degrees in -180 <= value < 180, the convention this door writes a box's west edge in (MPAS's met
    reader folds a start longitude above 180 back into it anyway)."""

    return round((float(value) + 180.0) % 360.0 - 180.0, 6)


def mesh_cell_degrees(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Every cell centre of an MPAS grid, init or static file: (latitude, longitude east 0..360) in degrees."""

    from netCDF4 import Dataset

    try:
        with Dataset(Path(path)) as dataset:
            lat = np.degrees(np.asarray(dataset.variables["latCell"][:], dtype=np.float64))
            lon = np.degrees(np.asarray(dataset.variables["lonCell"][:], dtype=np.float64)) % 360.0
    except (OSError, KeyError) as error:
        raise IntermediateRefusal(f"--grid {path} is not an MPAS mesh with latCell/lonCell: {error}") from error
    if lat.size == 0 or not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise IntermediateRefusal(f"--grid {path} holds no finite cell centres")
    return lat, lon


def _relative_longitudes(lon: np.ndarray) -> tuple[float, np.ndarray]:
    """A reference longitude and every longitude as degrees east of it, unwrapped across 0/360."""

    angle = np.radians(lon)
    centre = float(np.degrees(np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())) % 360.0)
    return centre, (lon - centre + 180.0) % 360.0 - 180.0


#: Target points a mesh cell needs on each side of it inside the box: MPAS case 7's sixteen-point
#: interpolation reads two columns and two rows either side of the cell's position.
STENCIL_POINTS = 2


def target_from_mesh(
    grid: Path, row: SourceRow, *, margin_km: float = DEFAULT_MARGIN_KM,
    spacing_deg: float = DEFAULT_SPACING_DEG,
) -> tuple[LatLonTarget, dict[str, Any]]:
    """The lat-lon box over EVERY cell of a mesh (its relaxation and specified rings are cells too), with
    ``margin_km`` of ground past them, trimmed to what the source can interpolate.

    THE BREAKAGE THIS PREVENTS (hex HRRR start, Vermont 2023-07-10 09Z, the NE 3 km cut of conus3km): a box
    built around a centre and a reach is symmetric, so near the source's edge it either left HRRR (refused) or,
    narrowed by hand, left 1,750 of the cut's cells outside the intermediate.  Those cells interpolated
    nothing, and case 7 stopped on "temperature extrapolation above the first-guess column top" and "1528
    nonpositive soil temperatures".  The box here is asymmetric: each side keeps the margin where the source
    has ground and gives it up where it does not, and the door refuses, by count, a mesh whose own cells the
    source does not cover.
    """

    if not spacing_deg > 0.0:
        raise IntermediateRefusal(f"--spacing-deg {spacing_deg} is not positive")
    lat, lon = mesh_cell_degrees(grid)
    centre, relative = _relative_longitudes(lon)
    stencil = STENCIL_POINTS * spacing_deg
    lat_margin = float(margin_km) / KM_PER_DEG
    cos_lat = max(math.cos(math.radians(min(89.0, float(np.abs(lat).max()) + lat_margin))), 0.05)
    lon_margin = float(margin_km) / (KM_PER_DEG * cos_lat)
    south = math.floor((float(lat.min()) - stencil - lat_margin) / spacing_deg) * spacing_deg
    north = math.ceil((float(lat.max()) + stencil + lat_margin) / spacing_deg) * spacing_deg
    west_rel = math.floor((float(relative.min()) - stencil - lon_margin) / spacing_deg) * spacing_deg
    east_rel = math.ceil((float(relative.max()) + stencil + lon_margin) / spacing_deg) * spacing_deg
    if south < -90.0 or north > 90.0:
        raise IntermediateRefusal(f"--grid {grid} reaches a pole; a regular lat-lon intermediate cannot carry it")
    target = LatLonTarget(south=round(south, 6), west=west_longitude(centre + west_rel),
                          dlat=spacing_deg, dlon=spacing_deg,
                          nx=int(round((east_rel - west_rel) / spacing_deg)) + 1,
                          ny=int(round((north - south) / spacing_deg)) + 1)
    # The box the cells themselves need: every cell with its interpolation stencil.
    need = {"south": float(lat.min()) - stencil, "north": float(lat.max()) + stencil,
            "west_rel": float(relative.min()) - stencil, "east_rel": float(relative.max()) + stencil,
            "centre": centre}
    try:
        target, trimmed = clip_to_source(target, row, need)
    except IntermediateRefusal as refusal:
        source_grid, x0, x1, y0, y1 = _source_window_limits(row)
        x, y = _native_positions(source_grid, lat, lon)
        beyond = int(((x < x0) | (x > x1) | (y < y0) | (y > y1)).sum())
        if not beyond:
            raise
        raise IntermediateRefusal(
            f"{beyond} of the {lat.size} cells of {grid} lie outside what {row.name} can interpolate (its "
            f"native grid less the decoder's halo); an init over them would read no meteorology there.  Cut the "
            f"mesh inside {row.description or row.name}") from refusal
    outside = cells_outside(target, lat, lon)
    if outside:
        raise IntermediateRefusal(
            f"{outside} of the {lat.size} cells of {grid} lie outside what {row.name} can interpolate (its "
            f"native grid less the decoder's halo); an init over them would read no meteorology there "
            f"(case 7 stops on temperature extrapolation or nonpositive soil temperatures).  Cut the mesh "
            f"inside {row.description or row.name}")
    basis = {"from_grid": str(grid), "cells": int(lat.size),
             "cells_deg": {"south": float(lat.min()), "north": float(lat.max()),
                           "west": west_longitude(centre + float(relative.min())),
                           "east": west_longitude(centre + float(relative.max()))},
             "trimmed_to_source": trimmed}
    return target, basis


def cells_outside(target: LatLonTarget, lat: np.ndarray, lon: np.ndarray, *,
                  stencil_points: int = STENCIL_POINTS) -> int:
    """How many cells lack ``stencil_points`` target points on every side of them inside the box."""

    x = ((lon - target.west) % 360.0) / target.dlon
    y = (lat - target.south) / target.dlat
    eps = 1e-9
    inside = ((x >= stencil_points - eps) & (x <= target.nx - 1 - stencil_points + eps)
              & (y >= stencil_points - eps) & (y <= target.ny - 1 - stencil_points + eps))
    return int((~inside).sum())


def mesh_cells_outside_met(met: Path, mesh: Path) -> dict[str, Any] | None:
    """For a regular lat-lon WPS intermediate: how many of a mesh's cells sit outside its box (with the
    interpolation stencil), or None for a projection this check does not read.

    THE BREAKAGE THIS NAMES (hex HRRR start, Vermont 2023-07-10 09Z): 1,750 cells outside the intermediate
    interpolated nothing and case 7 stopped on "temperature extrapolation above the first-guess column top" and
    "1528 nonpositive soil temperatures", messages that never said the box was short."""

    from .wps_intermediate import WpsIntermediateReader

    with WpsIntermediateReader(Path(met)) as reader:
        first = next(reader.iter_fields(load_values=False), None)
    if first is None or int(first.projection.code) != 0:
        return None
    projection = first.projection
    dlat, dlon = float(projection.delta_latitude), float(projection.delta_longitude)
    nx, ny = int(first.nx), int(first.ny)
    if not dlon > 0.0 or dlat == 0.0 or nx * dlon >= 360.0 - 1.5 * dlon:
        return None  # a global ring (or a layout this check does not read) covers every longitude
    south = min(float(projection.start_latitude), float(projection.start_latitude) + (ny - 1) * dlat)
    target = LatLonTarget(south=south, west=east_longitude(projection.start_longitude), dlat=abs(dlat), dlon=dlon,
                          nx=nx, ny=ny)
    lat, lon = mesh_cell_degrees(mesh)
    # Case 7 needs a cell inside the box; its sixteen-point stencil falls back to four points near the edge.
    outside = cells_outside(target, lat, lon, stencil_points=0)
    return {"cells": int(lat.size), "outside": outside, "box": target.as_dict()}


def uncovered_cells_refusal(met: Path, mesh: Path) -> str | None:
    """The refusal text when a lat-lon first guess does not cover every cell of the mesh, else None."""

    try:
        coverage = mesh_cells_outside_met(met, mesh)
    except (MpasPortError, OSError, ValueError):
        # An unreadable first guess or a mesh without cell centres fails in the producer with its own message.
        return None
    if not coverage or not coverage["outside"]:
        return None
    return (f"{coverage['outside']} of the {coverage['cells']} cells of {mesh} lie outside the first guess {met} "
            f"(box {coverage['box']}); the init would read no meteorology there and case 7 stops on "
            f"temperature extrapolation or nonpositive soil temperatures.  Write the intermediate over the mesh "
            f"itself: woof hex intermediate --grid {mesh}")


def _source_window_limits(row: SourceRow) -> tuple[Any, float, float, float, float]:
    """The source grid and the range of zero-based native positions a target point may take: every point
    needs its bilinear donor cell (x and x + 1) on the native grid.  The decoder's crop widens by its halo and
    extra cells up to the grid's edge and stops there, where the geometry interpolates bilinearly
    (``gpuwm_projected_window_v1`` / ``ProjectionSeed::build`` in the engine)."""

    import importlib

    loader = importlib.import_module(row.loader_module)
    grid = getattr(loader, row.source_grid_function)()
    nx, ny = int(grid.e_we) - 1, int(grid.e_sn) - 1
    return grid, 0.0, nx - 1.0 - 1e-6, 0.0, ny - 1.0 - 1e-6


def _native_positions(grid: Any, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-based native (x, y) of lat-lon points on a Lambert source, the formula of the engine's window
    entry (``ProjectionSeed::points``), so the trim below and the decoder's crop agree."""

    hemi, cone, rebydx = float(grid.hemi), float(grid.cone), float(grid.rebydx)
    polei, polej, truelat1, stand_lon = (float(grid.polei), float(grid.polej), float(grid.truelat1),
                                         float(grid.stand_lon))
    radians = math.pi / 180.0
    denominator = math.tan((90.0 * hemi - truelat1) * radians / 2.0)
    radius = rebydx * math.cos(truelat1 * radians) / cone * (
        np.tan((90.0 * hemi - lat) * radians / 2.0) / denominator) ** cone
    delta = lon - stand_lon
    delta = np.where(delta > 180.0, delta - 360.0, np.where(delta < -180.0, delta + 360.0, delta))
    angle = cone * delta * radians
    return hemi * (polei + hemi * radius * np.sin(angle)) - 1.0, hemi * (polej - radius * np.cos(angle)) - 1.0


def clip_to_source(target: LatLonTarget, row: SourceRow,
                   need: Mapping[str, float] | None = None) -> tuple[LatLonTarget, dict[str, int]]:
    """Trim a box, a row or column at a time from the side that has ground to spare, until every target point
    maps inside what the source's decoder crop can carry.  ``need`` is the part the box must keep (south, north,
    west_rel, east_rel about ``centre``); trimming into it is a refusal, never a silent cut."""

    grid, x0, x1, y0, y1 = _source_window_limits(row)
    south, west, nx, ny = target.south, target.west, target.nx, target.ny
    trimmed = {"south": 0, "north": 0, "west": 0, "east": 0}
    if need is not None:
        centre = float(need["centre"])
        floor_rows = {
            "south": lambda: (float(need["south"]) - south) / target.dlat,
            "north": lambda: (south + (ny - 1) * target.dlat - float(need["north"])) / target.dlat,
            "west": lambda: (float(need["west_rel"]) - ((west - centre + 180.0) % 360.0 - 180.0)) / target.dlon,
            "east": lambda: (((west - centre + 180.0) % 360.0 - 180.0) + (nx - 1) * target.dlon
                             - float(need["east_rel"])) / target.dlon,
        }
    for _ in range(4 * (target.nx + target.ny)):
        lats = south + target.dlat * np.arange(ny, dtype=np.float64)
        lons = west + target.dlon * np.arange(nx, dtype=np.float64)
        # The box's edge, where a smooth projection takes its extremes.
        la = np.concatenate([np.full(nx, lats[0]), np.full(nx, lats[-1]), lats, lats])
        lo = np.concatenate([lons, lons, np.full(ny, lons[0]), np.full(ny, lons[-1])])
        index = np.arange(la.size)
        on_side = {"south": index < nx, "north": (index >= nx) & (index < 2 * nx),
                   "west": (index >= 2 * nx) & (index < 2 * nx + ny), "east": index >= 2 * nx + ny}
        x, y = _native_positions(grid, la, lo)
        # Native x runs with longitude and y with latitude, so an x overshoot is the east or west side's to
        # give up and a y overshoot the north or south side's, whichever edge point carries it.
        kinds = {"east": x - x1, "west": x0 - x, "north": y - y1, "south": y0 - y}
        bad = {side: float(over.max()) for side, over in kinds.items() if over.max() > 0.0}
        if not bad:
            return LatLonTarget(south=round(south, 6), west=west_longitude(west), dlat=target.dlat,
                                dlon=target.dlon, nx=nx, ny=ny), trimmed
        spare = {side: (floor_rows[side]() if need is not None else float("inf")) for side in on_side}
        candidates = [side for side in bad if spare[side] >= 1.0]
        if not candidates:
            # Past the projection's turn the overshooting edge point sits on a perpendicular side.
            worst = np.maximum.reduce([np.maximum(over, 0.0) for over in kinds.values()]) > 0.0
            candidates = [side for side, mask in on_side.items() if (worst & mask).any() and spare[side] >= 1.0]
        side = max(candidates, key=lambda name: bad.get(name, 0.0)) if candidates else next(iter(bad))
        if spare[side] < 1.0 or nx < 2 * STENCIL_POINTS + 2 or ny < 2 * STENCIL_POINTS + 2:
            raise IntermediateRefusal(
                f"the box {target.as_dict()} leaves {row.name}'s native grid on its {side} side and has no "
                f"ground left to give there; the region is not inside {row.description or row.name}")
        trimmed[side] += 1
        if side == "south":
            south += target.dlat
            ny -= 1
        elif side == "north":
            ny -= 1
        elif side == "west":
            west += target.dlon
            nx -= 1
        else:
            nx -= 1
    raise IntermediateRefusal(f"the box {target.as_dict()} did not settle inside {row.name}'s native grid")


def target_from_plan(plan_path: Path, *, margin_km: float, spacing_deg: float) -> tuple[LatLonTarget, dict[str, Any]]:
    """The target grid a point plan's cull region implies."""

    try:
        document = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntermediateRefusal(f"--from-plan {plan_path} is not readable JSON: {error}") from error
    plan = document.get("plan", document)
    region = plan.get("cull_region") or {}
    cull = plan.get("cull") or {}
    halo = float(cull.get("halo_km") or 0.0)
    if region.get("kind") == "polygon":
        # THE BREAKAGE THIS PREVENTS (shape1-mesh, 2026-10-29): `woof hex regional --shape` / `--ellipse` cut a
        # drawn core, whose cull_region is the grown outline (a polygon, woof.hex.mesh_shape.shape_cull_region), and
        # this door refused it ("carries no cap cull_region"), so every drawn limited-area region stopped at the
        # intermediate stage after its mesh was built. The source window only has to COVER the cut, so a drawn cut
        # takes the cap that encloses it: about its vertex centroid, out to its farthest point along the ring.
        from . import mesh_shape

        try:
            centre = mesh_shape.centroid(region)
            ring = mesh_shape._densify(mesh_shape._xyz(*mesh_shape._ring(region).T), 5.0 / mesh_shape.EARTH_RADIUS_KM)
        except mesh_shape.ShapeRefusal as error:
            raise IntermediateRefusal(f"--from-plan {plan_path}: its polygon cull_region {error}") from error
        cosine = np.clip(ring @ mesh_shape._xyz(*centre), -1.0, 1.0)
        radius = float(np.degrees(np.arccos(cosine.min())) * KM_PER_DEG)
        target = target_for_cap(centre, radius + halo, margin_km=margin_km, spacing_deg=spacing_deg)
        return target, {"centre_deg": list(centre), "cut_radius_km": radius, "halo_km": halo,
                        "cut_kind": "polygon", "cut_radius_basis": "the cap about the polygon's vertex centroid "
                        "that encloses its ring"}
    if region.get("kind") != "cap":
        raise IntermediateRefusal(
            f"--from-plan {plan_path} carries no cap or polygon cull_region; it is not a "
            f"gpuwm-hex.point-plan/v1 or point-generate/v1 document"
        )
    centre = (float(region["center_deg"][0]), float(region["center_deg"][1]))
    radius = float(region["radius_km"])
    target = target_for_cap(centre, radius + halo, margin_km=margin_km, spacing_deg=spacing_deg)
    return target, {"centre_deg": list(centre), "cut_radius_km": radius, "halo_km": halo}


# ---------------------------------------------------------------------------
# the source window
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceWindow:
    i_start: int
    i_end: int
    j_start: int
    j_end: int

    @property
    def nx(self) -> int:
        return self.i_end - self.i_start + 1

    @property
    def ny(self) -> int:
        return self.j_end - self.j_start + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "zero_based_inclusive": {"i": [self.i_start, self.i_end], "j": [self.j_start, self.j_end]},
            "shape": [self.ny, self.nx],
        }


def _projection_seed(grid, target, window=None):
    seed = {name: float(getattr(grid, name)) for name in
            ("hemi", "cone", "rebydx", "polei", "polej", "truelat1", "stand_lon")}
    seed.update(native_nx=int(grid.e_we)-1, native_ny=int(grid.e_sn)-1, target=target.as_dict())
    seed.update(i_start=window.i_start if window else 0, j_start=window.j_start if window else 0,
                source_nx=window.nx if window else seed["native_nx"],
                source_ny=window.ny if window else seed["native_ny"])
    return seed


def source_window_for(target: LatLonTarget, row: SourceRow, *, extra_cells: int = 2) -> SourceWindow:
    """Rust projects the target and bounds the source crop before decode."""
    import importlib
    loader = importlib.import_module(row.loader_module)
    grid = getattr(loader, row.source_grid_function)()
    seed = _projection_seed(grid, target)
    seed.update(halo=row.parabolic_halo_cells, extra_cells=extra_cells)
    encoded = json.dumps(seed, separators=(",", ":")).encode("utf-8")
    library, _ = _native_projected_entry()
    entry = library.gpuwm_projected_window_v1
    entry.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int64),
                      ctypes.c_void_p, ctypes.c_size_t]
    entry.restype = ctypes.c_int
    bounds = (ctypes.c_int64 * 4)()
    error = ctypes.create_string_buffer(4096)
    code = entry(encoded, len(encoded), bounds, error, len(error))
    if code:
        raise IntermediateRefusal(f"the target interpolation halo leaves {row.name} coverage: "
                                  + error.value.decode("utf-8", errors="replace"))
    return SourceWindow(*bounds)

def loader_envelope(loader: Any) -> tuple[float, float, float, float]:
    try:
        from woof.ingest.hrrr_target import hrrr_coverage_envelope

        return tuple(float(v) for v in hrrr_coverage_envelope())  # type: ignore[return-value]
    except Exception:  # pragma: no cover - only the refusal text degrades
        return (float("nan"),) * 4


# ---------------------------------------------------------------------------
# the decoder
# ---------------------------------------------------------------------------
def parse_cycle(text: str) -> datetime:
    for pattern in ("%Y-%m-%dT%H", "%Y-%m-%dT%H:%M", "%Y-%m-%d_%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d%H"):
        try:
            return datetime.strptime(str(text).strip(), pattern)
        except ValueError:
            continue
    raise IntermediateRefusal(
        f"--cycle {text!r} is not a cycle; write it as YYYY-MM-DDTHH (UTC)"
    )


def parse_hours(text: str) -> tuple[int, ...]:
    hours: set[int] = set()
    for piece in str(text).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            first, last = piece.split("-", 1)
            hours.update(range(int(first), int(last) + 1))
        else:
            hours.add(int(piece))
    ordered = tuple(sorted(hours))
    if not ordered:
        raise IntermediateRefusal(f"--hours {text!r} names no forecast hour")
    if ordered != tuple(range(ordered[0], ordered[-1] + 1)):
        raise IntermediateRefusal(
            f"--hours {text!r} is not contiguous; the boundary producer admits "
            f"files by valid time and a gap would freeze the boundary across it"
        )
    return ordered


def resolve_decoder(row: SourceRow, explicit: Path | None = None) -> Path:
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if not candidate.is_file():
            raise IntermediateRefusal(f"--decoder {candidate} is not a file")
        return candidate
    try:
        from woof import bridges
    except ImportError as error:
        raise IntermediateRefusal(
            f"woof is not importable ({error}); this door drives the engine's "
            f"own {row.decoder_key} decoder and cannot find it without the "
            f"engine installed"
        ) from error
    try:
        return Path(bridges.resolve_source_decoder(row.decoder_key))
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise IntermediateRefusal(
            f"the engine's {row.decoder_key} decoder could not be resolved: "
            f"{error}.  `woof fetch-bridges` stages it"
        ) from error


def decode_window(
    row: SourceRow,
    *,
    decoder: Path,
    grib_dir: Path,
    cycle: datetime,
    hours: Sequence[int],
    window: SourceWindow,
    out_dir: Path,
    workers: int,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run the engine's decoder over the fetched hours into its native window."""

    grib_dir = Path(grib_dir).expanduser().absolute()
    series = out_dir / "series.tsv"
    rows: list[str] = []
    inputs: list[dict[str, Any]] = []
    for hour in hours:
        atmosphere, soil = row.hour_files(grib_dir, cycle, int(hour))
        for path in (atmosphere, soil):
            if not path.is_file():
                raise IntermediateRefusal(
                    f"{path} is not a file.  `woof fetch --source {row.name} "
                    f"--cycle {cycle:%Y-%m-%dT%H} --hours {max(hours)} --out "
                    f"{grib_dir}` writes it"
                )
        rows.append(f"{int(hour)}\t{atmosphere}\t{soil}\n")
        inputs.append({"hour": int(hour), "atmosphere": str(atmosphere), "soil": str(soil)})
    series.write_text("".join(rows), encoding="utf-8", newline="\n")
    native = out_dir / "native"
    if native.exists():
        raise IntermediateRefusal(
            f"{native} exists; the decoder refuses to overwrite a published "
            f"window.  Pass a fresh --out-dir"
        )
    workers = max(1, min(13, int(workers)))
    argv = [
        str(decoder), "--series-workers", str(workers), str(series), str(native),
        f"{cycle:%Y-%m-%d %H:00:00}",
        str(window.i_start), str(window.i_end), str(window.j_start), str(window.j_end),
    ]
    log(f"DECODE {row.decoder_key}: {len(hours)} hour(s), window "
        f"{window.ny}x{window.nx}, {workers} worker(s) ...")
    started = time.perf_counter()
    completed = subprocess.run(argv, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs" / "decode.log").write_text(
        f"$ {' '.join(argv)}\n[{elapsed:.1f} s, exit {completed.returncode}]\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise IntermediateRefusal(
            f"{decoder.name} exited {completed.returncode}; its own message is the "
            f"reason and it is in {out_dir / 'logs' / 'decode.log'}"
            + (f".  Last line: {tail[-1]}" if tail else "")
        )
    log(f"DECODE done in {elapsed:.1f} s -> {native}")
    manifest = _write_manifest(native)
    return {
        "decoder": str(decoder),
        "decoder_sha256": sha256_file(decoder),
        "argv": argv,
        "seconds": round(elapsed, 2),
        "series": str(series),
        "inputs": inputs,
        "native": str(native),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
    }


def _write_manifest(root: Path) -> Path:
    manifest = root / "SHA256SUMS"
    payloads = sorted(p for p in root.rglob("*") if p.is_file() and p != manifest)
    lines = [f"{sha256_file(p)}  ./{p.relative_to(root).as_posix()}" for p in payloads]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# the regrid
# ---------------------------------------------------------------------------
def _engine_pieces(row: SourceRow) -> dict[str, Any]:
    """The engine functions this road drives, resolved once and named once."""

    import importlib

    try:
        loader = importlib.import_module(row.loader_module)
        soil = importlib.import_module("woof.ingest.soil")
        backend_module = importlib.import_module("woof.ingest.preprocess_backend")
        producer_path, producer_name = row.producer_class.split(":", 1)
        producer_module = importlib.import_module(producer_path)
    except ImportError as error:
        raise IntermediateRefusal(
            f"the engine's ingest is not importable ({error}); this door drives "
            f"the engine's native decoder, projected metadata and interpolation operator and "
            f"has no road without them"
        ) from error
    needed = {
        "source_grid": getattr(loader, row.source_grid_function, None),
        "soil_nodes_m": getattr(soil, row.soil_depth_metadata[0], None),
        "noah_midpoints_m": getattr(soil, row.soil_depth_metadata[1], None),
        "resolve_backend": getattr(backend_module, "resolve_preprocess_backend", None),
        "producer": getattr(producer_module, producer_name, None),
    }
    missing = sorted(name for name, value in needed.items() if value is None)
    if missing:
        raise IntermediateRefusal(
            f"the installed engine's ingest does not expose {missing}; this "
            f"road drives the operator the engine ships and refuses to "
            f"substitute one of its own"
        )
    return needed


def _producer_requests(producer: Any, row: SourceRow) -> dict[str, Any]:
    """The row's decoder requests (analysed aerosol, request-only records), as the engine producer takes them.

    An engine whose producer cannot pass the requests would decode HRRR without its numbers and 2 m humidity,
    the start this row exists to stop (q2 = 0, numbers diagnosed from mass), so it is refused by name."""

    import inspect

    requests: dict[str, Any] = {}
    parameters = inspect.signature(producer).parameters
    if row.analyzed_aerosol:
        requests["analyzed_aerosol"] = True
    if row.decoder_extras:
        requests["extras"] = tuple(row.decoder_extras)
    # A producer that takes **kwargs takes every request by name (stand-ins and wrappers do).
    takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    missing = [] if takes_any else sorted(name for name in requests if name not in parameters)
    if missing:
        raise IntermediateRefusal(
            f"the installed engine's {row.producer_class} takes no {missing}; it cannot ask the {row.decoder_key} "
            f"decoder for {', '.join(row.decoder_extras)}, and a start without them has q2 = 0 and numbers "
            f"diagnosed from mass.  Install the woof release that carries the HRRR start fields")
    return requests


def _native_projected_entry():
    from woof.bridges import artifact_candidates

    name = ("rw_projected_intermediate.dll" if os.name == "nt" else
            "librw_projected_intermediate.dylib" if sys.platform == "darwin" else
            "librw_projected_intermediate.so")
    candidates = list(artifact_candidates("WOOF_PROJECTED_INTERMEDIATE_BRIDGE", name))
    # The entry is built in rw_wps, alongside its projected-source frontend.
    import woof
    root = Path(woof.__file__).resolve().parent.parent
    candidates.extend(root / "tools" / "rw_wps" / "target" / arm / name
                      for arm in ("release", "debug"))
    explicit = os.environ.get("WOOF_PROJECTED_INTERMEDIATE_BRIDGE")
    if explicit and not Path(explicit).is_file():
        raise IntermediateRefusal(f"WOOF_PROJECTED_INTERMEDIATE_BRIDGE={explicit} is not a file")
    for path in candidates:
        if path.is_file():
            try:
                library = ctypes.CDLL(str(path))
                entry = library.gpuwm_projected_intermediate_v1
                library.gpuwm_projected_geometry_v1
                library.gpuwm_projected_geometry_release_v1
                library.gpuwm_projected_window_v1
            except (OSError, AttributeError) as error:
                raise IntermediateRefusal(
                    f"{path} has no usable projected MET entry ({error}); "
                    "wind rotation, soil layering and MET writing cannot run. "
                    "Build tools/rw_wps with cargo build --release -p projected-intermediate"
                ) from error
            entry.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            entry.restype = ctypes.c_int
            return library, entry
    raise IntermediateRefusal(
        f"{name} was not found; projected MET fields cannot be regridded or written. "
        "Build tools/rw_wps with cargo build --release -p projected-intermediate; "
        "stage the library with the bridge bundle or name WOOF_PROJECTED_INTERMEDIATE_BRIDGE"
    )


def _field_spec(source: str, name: str, units: str, description: str, method: str,
                nonnegative: Sequence[str] = ()) -> dict[str, str]:
    transform = {"SPECHUMD": "nonnegative", "SNOW": "nonnegative",
                 "LANDSEA": "land", "SEAICE": "fraction"}.get(name, "none")
    if name in nonnegative:
        # The parabolic operator overshoots below zero beside a sharp cloud edge; a mixing ratio or a number
        # concentration is never negative, and the fork's case 7 copies what it is given.
        transform = "nonnegative"
    return dict(source=source, field=name, units=units, description=description,
                method=method, transform=transform)


class ProjectedMetPlan:
    """Retain the engine's immutable donor geometry across hourly native calls.

    Only geometry and table metadata cross Python. Rust reads window fields,
    rotates winds, interpolates soil nodes, regrids, clips, writes and verifies.
    The engine's FP64 donor decisions are kept separate from FP32 fractions.
    """
    def __init__(self, pieces, snapshot, target, backend):
        import weakref
        self.library, self.entry = _native_projected_entry()
        if not backend.indexed_donor_interp:
            raise IntermediateRefusal("the CPU bridge lacks gpuwm_indexed_interp_f32; exact projected MET donors cannot be evaluated. Rebuild tools/grib1_bridge")
        self.backend = backend
        self.key = (snapshot.i_start, snapshot.j_start, snapshot.ny, snapshot.nx, target)
        grid = pieces["source_grid"]()
        seed = _projection_seed(grid, target,
                                SourceWindow(snapshot.i_start, snapshot.i_start+snapshot.nx-1,
                                             snapshot.j_start, snapshot.j_start+snapshot.ny-1))
        encoded = json.dumps(seed, separators=(",", ":")).encode("utf-8")
        error = ctypes.create_string_buffer(4096)
        interpolate = ctypes.cast(backend._native._library.gpuwm_indexed_interp_f32, ctypes.c_void_p)
        create = self.library.gpuwm_projected_geometry_v1
        create.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        create.restype = ctypes.c_void_p
        self.geometry = create(encoded, len(encoded), interpolate, error, len(error))
        if not self.geometry:
            raise IntermediateRefusal(error.value.decode("utf-8", errors="replace"))
        release = self.library.gpuwm_projected_geometry_release_v1
        release.argtypes = [ctypes.c_void_p]
        release.restype = None
        self._release = weakref.finalize(self, release, self.geometry)
        self.workers = int(backend.workers)

    def close(self):
        self._release()


@dataclass(frozen=True)
class WindowFile:
    filename: Path


@dataclass(frozen=True)
class DecodedWindowFiles:
    valid_time: datetime
    forecast_hour: int
    i_start: int
    j_start: int
    ny: int
    nx: int
    fields: Mapping[str, WindowFile]


def decoded_window_files(row, root, cycle, hour, window):
    """Describe producer-owned files; Rust opens and validates every field."""
    if not all(row.decoded_directories):
        raise IntermediateRefusal("the source row has no decoded directory metadata; MET fields cannot be located")
    atmosphere, soil = (Path(root) / pattern.format(hour=hour) for pattern in row.decoded_directories)
    names = set(row.atmosphere_3d) | set(row.surface) | set(row.wind_3d) | set(row.wind_10m)
    fields = {name: WindowFile(atmosphere / f"{name}.f32le") for name in names}
    fields.update({name: WindowFile(soil / f"{name}.f32le")
                   for name in (row.soil_temperature, row.soil_moisture)})
    for name in ("QNWFA", "QNIFA"):
        path = atmosphere / f"{name}.f32le"
        if path.is_file():
            fields[name] = WindowFile(path)
    return DecodedWindowFiles(cycle + timedelta(hours=hour), hour, window.i_start, window.j_start,
                              window.j_end - window.j_start + 1, window.i_end - window.i_start + 1, fields)

#: The first-guess soil sets a source row can write: the interpolated Noah layers, or the source's own nodes.
SOIL_SETS = ("layers", "nodes")


def soil_spec(row: SourceRow, pieces: Mapping[str, Any], soil_set: str) -> dict[str, Any]:
    """The soil block of the native job: one set only, because a first-guess reader indexes both sets into the
    same soil levels (ST000010 and SOILT000 both land on level 1 of the RRFS fork's column)."""

    if soil_set not in SOIL_SETS:
        raise IntermediateRefusal(f"soil set {soil_set!r} is not one of {SOIL_SETS}")
    common = {"temperature": row.soil_temperature, "moisture": row.soil_moisture,
              "nodes": pieces["soil_nodes_m"].tolist(), "preserve_nodes": True}
    if soil_set == "nodes":
        if len(row.soil_node_names) != len(common["nodes"]):
            raise IntermediateRefusal(
                f"source row {row.name} names {len(row.soil_node_names)} soil nodes and its decoder carries "
                f"{len(common['nodes'])}; the nodes cannot be written under names a first-guess reader knows")
        return {**common, "midpoints": [], "names": [], "node_names": [list(pair) for pair in row.soil_node_names]}
    return {**common, "midpoints": pieces["noah_midpoints_m"].tolist(), "names": row.soil_layer_names}


def first_guess_levels(row: SourceRow, soil_set: str) -> tuple[int, int]:
    """(--nfglevels, --nfgsoillevels) of an intermediate written with ``soil_set``."""

    return row.levels + 1, (len(row.soil_node_names) if soil_set == "nodes" else len(row.soil_layer_names))


def soil_set_for_suite(row: SourceRow, suite: str | None) -> str:
    """The soil set an init for ``suite`` reads: the source's own nodes when the suite's init producer reads
    first-guess nodes (``soil.first_guess_soil`` = ``source-nodes`` in data/profiles/init-producers.v1.json; the
    RRFS fork's RUC column) and the row carries them, else the interpolated layers."""

    if not suite:
        return "layers"
    from .mpasphys.rrfs.cold_start_rrfs import producer_for_suite

    producer = producer_for_suite(suite)
    if producer is None:
        return "layers"
    wanted = (producer[1].get("soil") or {}).get("first_guess_soil")
    return "nodes" if wanted == "source-nodes" and row.soil_node_names else "layers"


def regrid_hour(
    row: SourceRow, pieces: Mapping[str, Any], snapshot: Any, target: LatLonTarget,
    *, backend: Any, path: Path, plan: ProjectedMetPlan | None = None, soil_set: str = "layers",
) -> tuple[dict[str, Any], ProjectedMetPlan]:
    """Drive the native projected MET entry. There is no Python field route."""
    if plan is None:
        plan = ProjectedMetPlan(pieces, snapshot, target, backend)
    if plan.key != (snapshot.i_start, snapshot.j_start, snapshot.ny, snapshot.nx, target):
        raise IntermediateRefusal("source window changed between hours; reusing its donors would misplace MET fields")
    if not plan._release.alive:
        raise IntermediateRefusal("projected geometry has been released; it cannot supply MET donors")
    required = set(row.atmosphere_3d) | set(row.surface) | set(row.wind_3d) | set(row.wind_10m)
    required.update((row.soil_temperature, row.soil_moisture))
    aerosol = [name for name in ("QNWFA", "QNIFA") if name in snapshot.fields]
    required.update(aerosol)
    sources = {}
    for name in sorted(required):
        value = snapshot.fields[name]
        filename = getattr(value, "filename", None)
        if filename is None:
            raise IntermediateRefusal(f"{name} is not a decoder window file; the native entry reads authenticated window bytes")
        sources[name] = str(filename)
    receipt_path = path.with_name(path.name + ".native.json")
    spec = {
        "output": str(path), "receipt": str(receipt_path),
        "valid_time": snapshot.valid_time.strftime("%Y-%m-%d_%H:%M:%S"),
        "forecast_hour": float(snapshot.forecast_hour), "map_source": row.map_source,
        "target": target.as_dict(), "levels": row.levels, "workers": plan.workers,
        "sources": sources,
        "atmosphere": [_field_spec(source, *metadata, "parabolic", row.nonnegative)
                       for source, metadata in row.atmosphere_3d.items()] +
                      [_field_spec(name, name, "kg-1", "Analysis aerosol number", "parabolic", (name,))
                       for name in aerosol],
        "surface": [_field_spec(source, *metadata) for source, metadata in row.surface.items()],
        "wind_3d": row.wind_3d, "wind_10m": row.wind_10m,
        "soil": soil_spec(row, pieces, soil_set),
        "admission": {"land": next(k for k, v in row.surface.items() if v[0] == "LANDSEA"),
                      "ice": next(k for k, v in row.surface.items() if v[0] == "SEAICE"),
                      "soil_temperature": row.soil_temperature, "soil_moisture": row.soil_moisture,
                      "humidity": [k for k, v in row.atmosphere_3d.items() if v[0] == "SPECHUMD"]
                                  + [k for k, v in row.surface.items() if v[0] == "SPECHUMD"]},
    }
    encoded = json.dumps(spec, separators=(",", ":")).encode("utf-8")
    error = ctypes.create_string_buffer(4096)
    code = plan.entry(encoded, len(encoded), plan.geometry, error, len(error))
    if code:
        raise IntermediateRefusal(error.value.decode("utf-8", errors="replace"))
    return json.loads(receipt_path.read_text(encoding="utf-8")), plan

# ---------------------------------------------------------------------------
# the writer
# ---------------------------------------------------------------------------
def _text(value: str, width: int) -> bytes:
    raw = value.encode("ascii", errors="strict")
    if len(raw) > width:
        raise IntermediateRefusal(f"{value!r} does not fit a {width}-character WPS field")
    return raw.ljust(width, b" ")


def _record(payload: bytes) -> bytes:
    marker = struct.pack(">i", len(payload))
    return marker + payload + marker


def wps_record_bytes(
    *,
    hdate: str,
    xfcst: float,
    map_source: str,
    field_name: str,
    units: str,
    description: str,
    level: float,
    target: LatLonTarget,
    values: np.ndarray,
    earth_radius_km: float = 6371.229,
) -> bytes:
    """One WPS version-5 field group on a regular lat-lon grid, big-endian."""

    values = np.asarray(values, dtype=np.float32)
    if values.shape != (target.ny, target.nx):
        raise IntermediateRefusal(
            f"{field_name} slab is {values.shape}, the target is ({target.ny}, {target.nx})"
        )
    if not np.isfinite(values).all():
        raise IntermediateRefusal(
            f"{field_name} at level {level:g} carries a non-finite value; a "
            f"structurally perfect record with NaN in it is exactly the file "
            f"this door refuses to write"
        )
    header = (
        _text(hdate, 24) + struct.pack(">f", float(xfcst)) + _text(map_source, 32)
        + _text(field_name, 9) + _text(units, 25) + _text(description, 46)
        + struct.pack(">f", float(level)) + struct.pack(">iii", target.nx, target.ny, 0)
    )
    projection = (
        _text("SWCORNER", 8)
        + struct.pack(">fffff", float(target.south), float(target.west),
                      float(target.dlat), float(target.dlon), float(earth_radius_km))
    )
    # The record is x-fastest (Fortran ``(nx, ny)``), which is exactly the
    # C-order byte stream of the row-major ``(ny, nx)`` slab; the reader
    # reshapes it as ``(nx, ny), order="F"`` and gets this array's transpose.
    slab = np.ascontiguousarray(values, dtype=">f4").tobytes(order="C")
    return (
        _record(struct.pack(">i", WPS_VERSION))
        + _record(header)
        + _record(projection)
        + _record(struct.pack(">i", 0))
        + _record(slab)
    )


def write_intermediate(
    path: Path, records: Sequence[Mapping[str, Any]], *, valid_time: datetime,
    forecast_hour: float, map_source: str, target: LatLonTarget,
) -> dict[str, Any]:
    hdate = valid_time.strftime("%Y-%m-%d_%H:%M:%S")
    with open(path, "wb") as handle:
        for item in records:
            handle.write(wps_record_bytes(
                hdate=hdate, xfcst=forecast_hour, map_source=map_source,
                field_name=str(item["field"]), units=str(item["units"]),
                description=str(item["description"]), level=float(item["level"]),
                target=target, values=item["values"],
            ))
    # Read back through the tree's own reader before claiming anything.
    from .wps_intermediate import inventory

    seen = inventory(path)
    if int(seen["field_records"]) != len(records):
        raise IntermediateRefusal(
            f"{path} reads back as {seen['field_records']} records after "
            f"{len(records)} were written; the writer and the reader disagree"
        )
    for got, want in zip(seen["records"], records):
        if got["field"] != want["field"] or got["nx"] != target.nx or got["ny"] != target.ny \
                or abs(float(got["level"]) - float(want["level"])) > 1e-3:
            raise IntermediateRefusal(
                f"{path}: record {want['field']}@{want['level']:g} read back as "
                f"{got['field']}@{got['level']:g} ({got['nx']}x{got['ny']})"
            )
    return {
        "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path),
        "valid_time": hdate, "forecast_hour": forecast_hour,
        "field_records": int(seen["field_records"]), "fields": seen["fields"],
        "record_endian": seen["record_endian"],
    }


# ---------------------------------------------------------------------------
# the door
# ---------------------------------------------------------------------------
@dataclass
class IntermediateRequest:
    source: SourceRow
    grib_dir: Path
    cycle: datetime
    hours: tuple[int, ...]
    target: LatLonTarget
    out_dir: Path
    decoder: Path | None = None
    workers: int = 4
    prefix: str = "MET"
    target_basis: Mapping[str, Any] = field(default_factory=dict)
    #: One of SOIL_SETS: the interpolated layers, or the source's own nodes (soil_set_for_suite).
    soil_set: str = "layers"
    physics_suite: str | None = None


_LEDGER_WARNED: set[str] = set()


def _ledger_emit(path: Path, stage: str, phase: str, attrs: Mapping[str, Any]) -> None:
    """C1 v1, one O_APPEND write, never a model failure on telemetry failure."""
    try:
        from woof.timeline.ledger import emit
    except ImportError:
        pass  # The frozen hex bootstrap predates the separately delivered C1 package.
    else:
        try:
            emit(stage, phase, path=path, product="hex", attrs=attrs)
        except BaseException:
            pass
        return
    destination = os.environ.get("WOOF_TIME_LEDGER") or str(path)
    try:
        event = {"v": 1, "t_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                 "t_mono": time.monotonic(), "host": socket.gethostname(),
                 "rental_id": os.environ.get("WOOF_RENTAL_ID"), "run_id": os.environ.get("WOOF_RUN_ID"),
                 "product": "hex", "stage": stage, "phase": phase, "card": None,
                 "pid": os.getpid(), "attrs": dict(attrs)}
        data = (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            if os.write(fd, data) != len(data):
                raise OSError("short append")
        finally:
            os.close(fd)
    except Exception as error:
        if destination not in _LEDGER_WARNED:
            _LEDGER_WARNED.add(destination)
            # A warnings filter set to 'error' must not turn telemetry into a
            # forecast failure either.
            try:
                warnings.warn(f"timeline append failed for {destination}: {error}", RuntimeWarning)
            except Exception:
                pass


def _fetch_requests(fetch: Callable[..., Any], row: SourceRow) -> dict[str, Any]:
    """The decoder's requests (analysed aerosol, request-only records), asked of the row's fetch as well.

    THE BREAKAGE THIS PREVENTS (mesh-gen-fast polygon from HRRR 2026-10-28T12, an RTX 4090 host, 2026-10-29): the fetch took
    the plain 561-record index subset, without NCONCD/NCCICE/SPNCR or RH at 2 m, while the decoder was asked for
    QNC/QNI/QNR/RH2, so it refused "missing required field QNC hybrid level 1" and every HRRR start without whole
    GRIB on disk failed at the intermediate stage.  A fetch that cannot take the requests is refused by name."""

    import inspect

    requests: dict[str, Any] = {}
    if row.analyzed_aerosol:
        requests["analyzed_aerosol"] = True
    if row.decoder_extras:
        requests["extras"] = tuple(row.decoder_extras)
    parameters = inspect.signature(fetch).parameters
    takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    missing = [] if takes_any else sorted(name for name in requests if name not in parameters)
    if missing:
        raise IntermediateRefusal(
            f"the installed engine's {row.fetch_function} takes no {missing}; its download would lack "
            f"{', '.join(row.decoder_extras) or 'the analysed aerosol'} and the decoder would refuse the hour.  "
            f"Install the woof release that carries the HRRR start fields")
    return requests


def _ensure_hour(request: IntermediateRequest, hour: int, log: Callable[[str], None]) -> Path:
    """Fetch I/O only. This runs in the one prefetch thread, never a process pool."""
    row = request.source
    root = Path(request.grib_dir).expanduser().absolute()
    if all(path.is_file() for path in row.hour_files(root, request.cycle, hour)):
        return root
    if not row.fetch_function:
        raise IntermediateRefusal(f"source row {row.name} has no fetch transport for missing hour {hour}")
    import importlib
    from woof.fetch import select_fetch_engine

    module, function = row.fetch_function.split(":", 1)
    fetch = getattr(importlib.import_module(module), function)
    requests = _fetch_requests(fetch, row)
    choice = select_fetch_engine("rust", progress=log)
    root = Path(request.out_dir).absolute() / "fetch" / f"f{hour:02d}"
    root.mkdir(parents=True, exist_ok=True)
    ledger = Path(request.out_dir).absolute() / "timeline.jsonl"
    _ledger_emit(ledger, "fetch.intermediate.hour", "begin", {"hour": hour})
    started = time.perf_counter()
    try:
        fetch(cycle=request.cycle, hours=(hour,), area=None, out=root, workers=request.workers,
              file_workers=1, engine=choice.engine, engine_bin=choice.binary,
              engine_selection=choice.selection, transport="s3", progress=log,
              cache_dir=Path(request.out_dir).absolute() / "fetch-cache", **requests)
    except Exception:
        _ledger_emit(ledger, "fetch.intermediate.hour", "end",
                     {"hour": hour, "seconds": time.perf_counter() - started, "status": "failed"})
        raise
    _ledger_emit(ledger, "fetch.intermediate.hour", "end",
                 {"hour": hour, "seconds": time.perf_counter() - started, "status": "complete"})
    if not all(path.is_file() for path in row.hour_files(root, request.cycle, hour)):
        raise IntermediateRefusal(f"fetch completed without both source files for hour {hour}; no decode can start")
    return root


def _admit_hour(row, grib_root, cycle, hour, admissions):
    """Bind the bytes admitted to the one live Rust producer before it reads them."""
    atmosphere, soil = row.hour_files(grib_root, cycle, hour)
    inputs = {"hour": hour, "atmosphere": str(atmosphere), "soil": str(soil),
              "atmosphere_sha256": sha256_file(atmosphere), "soil_sha256": sha256_file(soil)}
    path = admissions / f"f{hour:02d}.admitted"
    temporary = path.with_suffix(".partial")
    temporary.write_text(f"forecast_hour\t{hour}\natmosphere\t{atmosphere}\nsoil\t{soil}\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, path)
    return inputs


def _cancel_producer(producer):
    """Use a caller's owned-process stop helper when its host requires one."""
    command = os.environ.get("WOOF_CHILD_STOP_COMMAND")
    process = getattr(producer, "process", None)
    if command and process is not None and process.poll() is None:
        argv = json.loads(command)
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
            raise IntermediateRefusal("WOOF_CHILD_STOP_COMMAND must name an argv list for the owned decoder stop helper")
        subprocess.run([*argv, str(process.pid)], check=True, capture_output=True, text=True)
        process.wait(timeout=30)
    producer.cancel()


def build_intermediates(request: IntermediateRequest, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    row = request.source
    out_dir = Path(request.out_dir).expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob(f"{request.prefix}:*")):
        raise IntermediateRefusal(f"{out_dir} holds intermediate hours; pass a fresh --out-dir so completion markers cannot describe an earlier run")
    pieces = _engine_pieces(row)
    decoder = resolve_decoder(row, request.decoder)
    window = source_window_for(request.target, row)
    if not request.hours or tuple(sorted(set(request.hours))) != tuple(request.hours):
        raise IntermediateRefusal("forecast hours must be unique and ordered; per-hour completion cannot be ambiguous")
    # Resolve the native entry before starting fetch. No fallback keeps the
    # Python field path alive when the native artifact is absent.
    _native_projected_entry()
    backend = pieces["resolve_backend"]("cpu", workers=request.workers)
    files: list[dict[str, Any]] = []
    decoded_hours: list[dict[str, Any]] = []
    plan = None
    operator = None
    ledger = out_dir / "timeline.jsonl"
    if len(request.hours) < 2 or request.hours != tuple(range(request.hours[0], request.hours[-1]+1)):
        raise IntermediateRefusal("the decoder series requires at least two contiguous hours; a missing boundary interval cannot be published")
    admissions = out_dir / "admissions"
    admissions.mkdir()
    series = out_dir / "series.tsv"
    predicted = {}
    for hour in request.hours:
        root = Path(request.grib_dir).expanduser().absolute()
        if not all(path.is_file() for path in row.hour_files(root, request.cycle, hour)):
            root = out_dir / "fetch" / f"f{hour:02d}"
        predicted[hour] = root
    series.write_text("".join(f"{hour}\t{a}\t{s}\n" for hour in request.hours
                              for a, s in [row.hour_files(predicted[hour], request.cycle, hour)]),
                      encoding="utf-8", newline="\n")
    native = out_dir / "native"
    producer = pieces["producer"](
        decoder=decoder, series=series, output=native, signals=out_dir / "signals",
        admissions=admissions, cycle=request.cycle.strftime("%Y-%m-%d %H:00:00"),
        window=(window.i_start, window.i_end, window.j_start, window.j_end),
        workers=str(max(1, min(row.decoder_workers_max, request.workers))), log=out_dir / "logs" / "decode.log",
        **_producer_requests(pieces["producer"], row))
    # One I/O thread fetches one missing lead ahead of ordered consumption.
    # Existing immutable input pairs are admitted immediately so inventory and
    # decode stay parallel on warm packs. Rust owns the shared field budget.
    condition = Condition()
    stop = Event()
    admitted = {}
    consuming = [0]

    def fetch_series():
        try:
            for index, hour in enumerate(request.hours):
                if predicted[hour] != Path(request.grib_dir).expanduser().absolute():
                    with condition:
                        while index > consuming[0] + 1 and not stop.is_set():
                            condition.wait(0.02)
                if stop.is_set():
                    return
                root = _ensure_hour(request, int(hour), log)
                if root != predicted[hour]:
                    raise IntermediateRefusal("fetched root differs from the admitted series; refuses a different source window")
                started = time.perf_counter()
                attrs = {"hour": int(hour), "decode_s": 0.0, "regrid_s": 0.0}
                _ledger_emit(ledger, "prep.intermediate.hour", "begin", attrs)
                try:
                    inputs = _admit_hour(row, root, request.cycle, int(hour), admissions)
                except Exception:
                    _ledger_emit(ledger, "prep.intermediate.hour", "end", {**attrs, "status": "failed"})
                    raise
                with condition:
                    admitted[hour] = (inputs, started)
                    condition.notify_all()
        finally:
            with condition:
                condition.notify_all()
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="intermediate-fetch") as prefetch:
            producer.start()
            fetched = prefetch.submit(fetch_series)
            for index, hour in enumerate(request.hours):
                hour = int(hour)
                with condition:
                    consuming[0] = index
                    condition.notify_all()
                    while hour not in admitted and not fetched.done():
                        condition.wait(0.02)
                    if hour not in admitted:
                        fetched.result()
                        raise IntermediateRefusal(f"source hour {hour} was never admitted; no completion can be published")
                    inputs, started = admitted[hour]
                attrs = {"hour": hour, "decode_s": 0.0, "regrid_s": 0.0}
                try:
                    if index == 0:
                        producer.wait_preflight()
                    staging = producer.wait_hour(hour)
                    attrs["decode_s"] = time.perf_counter() - started
                    decoded_hours.append({"native": str(native), "native_during_consume": str(staging),
                                          "seconds": attrs["decode_s"], "inputs": [inputs]})
                    snapshot = decoded_window_files(row, staging, request.cycle, hour, window)
                    valid = request.cycle + timedelta(hours=hour)
                    path = out_dir / f"{request.prefix}:{valid:%Y-%m-%d_%H}"
                    regrid_start = time.perf_counter()
                    written, plan = regrid_hour(row, pieces, snapshot, request.target,
                                               backend=backend, path=path, plan=plan,
                                               soil_set=request.soil_set)
                    attrs["regrid_s"] = time.perf_counter() - regrid_start
                    written["regrid_seconds"] = attrs["regrid_s"]
                    operator = written["operator"]
                    marker = path.with_name(path.name + ".ready.json")
                    temporary = marker.with_name(marker.name + ".partial")
                    completion = {"schema": "gpuwm-hex.intermediate.hour/v1", "hour": hour,
                                  **written, **attrs}
                    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                        handle.write(json.dumps(completion, sort_keys=True) + "\n")
                    os.replace(temporary, marker)
                    written["completion_marker"] = str(marker)
                    files.append(written)
                    _ledger_emit(ledger, "prep.intermediate.hour", "end", {**attrs, "status": "complete"})
                    _ledger_emit(ledger, "prep.intermediate.hour", "mark",
                                 {**attrs, "path": str(path), "marker": str(marker), "sha256": written["sha256"]})
                    log(f"WROTE {path.name}: {written['field_records']} records, "
                        f"{written['bytes'] / 1e6:.1f} MB, {attrs['regrid_s']:.3f} s")
                    del snapshot
                except Exception:
                    _ledger_emit(ledger, "prep.intermediate.hour", "end", {**attrs, "status": "failed"})
                    stop.set()
                    raise
            fetched.result()
        # Record the producer's generated staging paths before its normal cleanup.
        staging_files = [{"path": str(p), "bytes": p.stat().st_size}
                         for p in staging.rglob("*") if p.is_file() and not p.is_symlink()]
        (out_dir / "staging-cleanup.json").write_text(json.dumps(staging_files, indent=2)+"\n", encoding="utf-8")
        pipeline = producer.finish()
    except Exception as error:
        # This is the existing producer's owned-process failure unwind. Its
        # in-memory tail and exact PID-owned staging are captured before cleanup.
        _cancel_producer(producer)
        if plan is not None:
            plan.close()
        raise IntermediateRefusal(f"posted decoder/MET pipeline failed: {error}; log {out_dir / 'logs' / 'decode.log'}") from error
    manifest = _write_manifest(native)
    decoded = {"seconds": sum(item["seconds"] for item in decoded_hours), "hours": decoded_hours,
               "native": str(native), "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
               "pipeline": pipeline, "decoder": str(decoder), "decoder_sha256": sha256_file(decoder)}
    if plan is not None:
        plan.close()
    receipt = {
        "schema": INTERMEDIATE_SCHEMA,
        "source": row.name,
        "source_description": row.description,
        "cycle": f"{request.cycle:%Y-%m-%d_%H:%M:%S}",
        "hours": list(request.hours),
        "grib_dir": str(Path(request.grib_dir).absolute()),
        "target": request.target.as_dict(),
        "target_basis": dict(request.target_basis),
        "source_window": window.as_dict(),
        "decode": decoded,
        "regrid": {
            "operator": operator,
            "engine": "rw_wps projected-intermediate, using gpuwm_indexed_interp_f32 "
                      "and the engine projected plan's FP64 donors and rotation seed",
            "soil": ("the source's own soil nodes, moved by nearest neighbour, under the row's node names"
                     if request.soil_set == "nodes" else
                     "Rust FP64 node interpolation to the declared layer midpoints, "
                     "moved by nearest neighbour"),
            "categorical": "nearest neighbour (LANDSEA, SEAICE)",
        },
        "soil_set": request.soil_set,
        "physics_suite": request.physics_suite,
        "levels": {
            "count": row.levels + 1,
            "convention": "level-indexed hybrid slabs (xlvl 1..N) with the 3-D PRESSURE "
                          "field carried, plus the surface level 200100.0",
            "init_switches": {
                "--nfglevels": first_guess_levels(row, request.soil_set)[0],
                "--nfgsoillevels": first_guess_levels(row, request.soil_set)[1],
                "--use-spechumd": "yes", "--extrap-airtemp": "constant",
                "why_constant": "the source column tops below the 30 km model top; "
                                "lapse-rate extrapolation above the first-guess top is "
                                "the Fortran's own fatal (docs/source-matrix.md, IFS row)",
            },
        },
        "files": files,
        "carried": {"atmosphere": sorted(spec[0] for spec in row.atmosphere_3d.values()),
                    "surface": sorted({spec[0] for spec in row.surface.values()}),
                    "decoder_extras": list(row.decoder_extras),
                    "analyzed_aerosol_requested": bool(row.analyzed_aerosol)},
        "not_carried": ["SNOWH", "PMSL"],
        "why_not_carried": "the row maps no snow depth and the engine's window carries no "
                           "mean-sea-level pressure; neither init requires one.  Both inits read the "
                           "hydrometeor and number records: the RRFS producer always, the MPAS-A init "
                           "(rw_mpas_init) under the init door's --first-guess-hydrometeors (default yes; "
                           "no under strict numerics, native case 7, no condensate at hour zero)",
    }
    receipt_path = out_dir / "intermediate-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    log(f"RECEIPT {receipt_path}")
    return receipt


def add_intermediate_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "intermediate",
        help="write regular lat-lon WPS intermediates from a projected regional "
             "source (HRRR) through the engine's own decoder and operator",
        description=(
            "Resample a projected regional source onto a regular lat-lon WPS "
            "intermediate the init and boundary engines read.  The decode, the "
            "interpolation operator and the soil layering are woof's own; the "
            "record layout is the inverse of this tree's reader and every file "
            "is read back before the receipt is written."
        ),
    )
    parser.add_argument("--source", default="hrrr", metavar="ROW",
                        help=f"a row of SOURCE_ROWS ({', '.join(sorted(SOURCE_ROWS))})")
    parser.add_argument("--grib-dir", type=Path, required=True, metavar="DIR",
                        help="where `woof fetch --source hrrr` wrote the cycle")
    parser.add_argument("--cycle", required=True, metavar="YYYY-MM-DDTHH",
                        help="the cycle the files came from (UTC)")
    parser.add_argument("--hours", default="0-3", metavar="SPEC",
                        help="forecast hours to write, contiguous (default 0-3)")
    parser.add_argument("--from-plan", type=Path, default=None, metavar="JSON",
                        help="derive the target box from a mesh-plan --point receipt's cull region")
    parser.add_argument("--grid", type=Path, default=None, metavar="MESH.nc",
                        help="derive the target box from every cell of an MPAS mesh (grid, static or init "
                             "file; relaxation and specified rings included), trimmed to the source")
    parser.add_argument("--physics-suite", default=None, metavar="SUITE",
                        help="the suite the init is for: an RRFS suite (hrrrv5, ...) reads the source's own "
                             "soil nodes (RUC's nine), every other suite the interpolated Noah layers")
    parser.add_argument("--point", default=None, metavar="LAT,LON",
                        help="centre of the target box (with --radius-km) instead of --from-plan")
    parser.add_argument("--radius-km", type=float, default=None, metavar="KM",
                        help="half-reach of the target box around --point")
    parser.add_argument("--margin-km", type=float, default=DEFAULT_MARGIN_KM, metavar="KM",
                        help=f"ground kept past the reach (default {DEFAULT_MARGIN_KM:g})")
    parser.add_argument("--spacing-deg", type=float, default=DEFAULT_SPACING_DEG, metavar="DEG",
                        help=f"target lat-lon spacing (default {DEFAULT_SPACING_DEG:g})")
    parser.add_argument("--out-dir", type=Path, required=True, metavar="DIR")
    parser.add_argument("--prefix", default="MET", metavar="TEXT",
                        help="file prefix, ungrib style PREFIX:YYYY-MM-DD_HH (default MET)")
    parser.add_argument("--decoder", type=Path, default=None, metavar="FILE",
                        help="the engine's decoder binary (default: woof's bridge ladder)")
    parser.add_argument("--workers", type=int, default=4, metavar="N",
                        help="decoder workers, 1..13 (default 4)")
    parser.set_defaults(handler=run_intermediate)


def request_from_arguments(arguments: argparse.Namespace) -> IntermediateRequest:
    row = source_row(arguments.source)
    given = [flag for flag, value in (("--from-plan", arguments.from_plan), ("--point", arguments.point),
                                      ("--grid", getattr(arguments, "grid", None))) if value is not None]
    if len(given) > 1:
        raise IntermediateRefusal(f"{' and '.join(given)} were all given; the box comes from one")
    if getattr(arguments, "grid", None) is not None:
        target, basis = target_from_mesh(Path(arguments.grid).expanduser(), row, margin_km=arguments.margin_km,
                                         spacing_deg=arguments.spacing_deg)
    elif arguments.from_plan is not None:
        target, basis = target_from_plan(
            arguments.from_plan, margin_km=arguments.margin_km, spacing_deg=arguments.spacing_deg,
        )
        basis["from_plan"] = str(arguments.from_plan)
    elif arguments.point is not None:
        from .mesh_point import parse_point

        if arguments.radius_km is None:
            raise IntermediateRefusal("--point needs --radius-km: the box has no reach without one")
        centre = parse_point(arguments.point)
        target = target_for_cap(centre, float(arguments.radius_km),
                                margin_km=arguments.margin_km, spacing_deg=arguments.spacing_deg)
        basis = {"centre_deg": list(centre), "reach_km": float(arguments.radius_km)}
    else:
        raise IntermediateRefusal(
            "none of --grid, --from-plan or --point was given; the intermediate covers a "
            "box and nothing here guesses one"
        )
    basis["margin_km"] = float(arguments.margin_km)
    return IntermediateRequest(
        source=row, grib_dir=arguments.grib_dir, cycle=parse_cycle(arguments.cycle),
        hours=parse_hours(arguments.hours), target=target, out_dir=arguments.out_dir,
        decoder=arguments.decoder, workers=int(arguments.workers), prefix=str(arguments.prefix),
        target_basis=basis, soil_set=soil_set_for_suite(row, getattr(arguments, "physics_suite", None)),
        physics_suite=getattr(arguments, "physics_suite", None),
    )


def run_intermediate(arguments: argparse.Namespace) -> int:
    request = request_from_arguments(arguments)
    receipt = build_intermediates(request)
    print(json.dumps({
        "files": [item["path"] for item in receipt["files"]],
        "target": receipt["target"],
        "source_window": receipt["source_window"],
        "next": (
            f"woof hex init --physics-suite {receipt['physics_suite']} --met {receipt['files'][0]['path']} "
            f"... --nfglevels {receipt['levels']['init_switches']['--nfglevels']} --nfgsoillevels "
            f"{receipt['levels']['init_switches']['--nfgsoillevels']}"
            if receipt.get("physics_suite") else
            f"woof hex init --met {receipt['files'][0]['path']} ... --nfglevels "
            f"{receipt['levels']['init_switches']['--nfglevels']} --nfgsoillevels "
            f"{receipt['levels']['init_switches']['--nfgsoillevels']} "
            f"--use-spechumd yes --extrap-airtemp constant"
        ),
    }, indent=2))
    return 0


# ---------------------------------------------------------------------------
# the boundary door: rw_mpas_lbc over the hourly intermediates
# ---------------------------------------------------------------------------
def intermediate_valid_times(met_dir: Path) -> list[tuple[datetime, Path]]:
    """Every WPS intermediate in a directory, by the valid time in its own header."""

    from .init_door import _probe_wps_header
    from .wps_intermediate import WpsIntermediateReader

    found: list[tuple[datetime, Path]] = []
    for path in sorted(Path(met_dir).iterdir()):
        if not path.is_file() or not _probe_wps_header(path):
            continue
        with WpsIntermediateReader(path) as reader:
            first = next(reader.iter_fields(load_values=False), None)
        if first is None:
            continue
        found.append((datetime.strptime(first.valid_time[:19], "%Y-%m-%d_%H:%M:%S"), path))
    if not found:
        raise IntermediateRefusal(
            f"{met_dir} holds no WPS intermediate; `woof hex intermediate` writes them"
        )
    found.sort()
    return found


def add_lbc_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "lbc",
        help="build lateral boundary files with rw_mpas_lbc from hourly WPS intermediates",
        description=(
            "Drive rw_mpas_lbc on its wps-intermediate route: one boundary file per "
            "intermediate valid time between --start-time and --stop-time.  The "
            "first-guess switches have no defaults, exactly as the engine says."
        ),
    )
    parser.add_argument("--grid", type=Path, required=True, metavar="INIT.nc",
                        help="the child's initial-conditions file (the culled init)")
    parser.add_argument("--met-dir", type=Path, required=True, metavar="DIR",
                        help="directory of WPS intermediates, one per boundary time")
    parser.add_argument("--out-dir", type=Path, required=True, metavar="DIR")
    parser.add_argument("--start-time", required=True, metavar="YYYY-MM-DD_HH:MM:SS")
    parser.add_argument("--stop-time", required=True, metavar="YYYY-MM-DD_HH:MM:SS")
    parser.add_argument("--nfglevels", type=int, required=True)
    parser.add_argument("--extrap-airtemp", choices=("constant", "linear", "lapse-rate"), required=True)
    parser.add_argument("--use-spechumd", choices=("yes", "no"), required=True)
    parser.add_argument("--theta-adv-order", type=int, default=3)
    parser.add_argument("--coef-3rd-order", type=float, default=0.25)
    parser.add_argument("--oned-underflow", choices=("preserve", "reproduce-ifx-ftz"), default="preserve")
    parser.add_argument("--physics-mode", choices=("correct", "exact"), default="correct",
                        help="correct (default): boundary winds extrapolate like the initial conditions "
                             "(IN4) and every boundary file must carry its valid time (IN5); exact: "
                             "native init_atmosphere v8.4.2 case 9.  An rw_mpas_lbc without "
                             "--physics-mode writes case 9 and the door receipt names IN4/IN5 as gaps")
    parser.add_argument("--lbc-exe", type=Path, default=None, metavar="FILE")
    parser.add_argument("--workers", type=int, default=None, metavar="N",
                        help="rw_mpas_lbc processes writing boundary times at once (default: one per time, "
                             "bounded by CPUs, 16 and half the available host memory)")
    parser.add_argument("--stream", action="store_true",
                        help="publish each hash-checked LBC hour as its MET marker lands")
    producer = parser.add_argument_group(
        "suites with their own producer (RRFS-MPAS: hrrrv5, rrfs_*)",
        "the producer's case 9 writes the boundaries, on the grid and fields of its own case-7 init (--grid)")
    producer.add_argument("--physics-suite", default=None,
                          help="the run's suite; one with an init producer row gets that producer's boundaries")
    producer.add_argument("--physics-pin-set", default=None,
                          help="source pin set of the suite route whose producer writes the boundaries")
    producer.add_argument("--rrfs-worker", type=Path, default=None,
                          help="rw_mpas_init_v831 (default $WOOF_HEX_RW_MPAS_INIT_V831, then the engine ladder)")
    producer.add_argument("--wif-climatology", type=Path, default=None,
                          help="TEMPO's monthly aerosol climatology when the first guess carries none")
    parser.set_defaults(handler=run_lbc)


def lbc_workers(count: int, grid: Path, mets: Sequence[Path], requested: int | None = None) -> int:
    """How many rw_mpas_lbc processes write a boundary series at once: one per time, bounded by the CPUs this
    process may use, 16, and host memory (each process held 1.1 GiB for a 0.27 GB init plus a 0.82 GB GFS
    intermediate, about the two files' bytes; half of ``MemAvailable`` is the budget). ``--workers`` caps it;
    with no memory reading the series stays one process, as it always was."""
    from .cpu_budget import available_cpus
    from .products_door import available_memory_bytes

    if count <= 1:
        return 1
    limit = min(count, available_cpus(), 16)
    if requested is not None:
        return max(1, min(limit, int(requested)))
    memory = available_memory_bytes()
    if memory is None:
        return 1
    try:
        each = Path(grid).stat().st_size + max(Path(path).stat().st_size for path in mets)
    except OSError:
        return 1
    return max(1, min(limit, int(memory // 2 // max(each, 1))))


def _run_producer_lbc(arguments: argparse.Namespace, producer, grid: Path) -> int:
    """``woof hex lbc --physics-suite <producer suite>``: the producer's own case-9 boundaries.

    THE BREAKAGE IT PREVENTS: a region on hrrrv5 got rw_mpas_lbc's MPAS-A v8.4.2 boundaries (Noah-era
    fields, no hydrometeor or aerosol rows) for an init on the RRFS fork's L60 grid."""

    from .mpasphys.rrfs.cold_start_rrfs import RrfsInitError, run_rrfs_lbc

    start = datetime.strptime(arguments.start_time, "%Y-%m-%d_%H:%M:%S")
    stop = datetime.strptime(arguments.stop_time, "%Y-%m-%d_%H:%M:%S")
    times = [(t, p) for t, p in intermediate_valid_times(arguments.met_dir) if start <= t <= stop]
    if not times or times[0][0] != start or times[-1][0] != stop:
        raise IntermediateRefusal(
            f"the intermediates in {arguments.met_dir} do not cover {arguments.start_time}..{arguments.stop_time} "
            "at both ends; a boundary series with a missing end freezes the boundary there without saying so")
    out_dir = Path(arguments.out_dir).expanduser().absolute()
    try:
        record = run_rrfs_lbc(producer, init=grid, intervals=[(f"{t:%Y-%m-%d_%H:%M:%S}", p) for t, p in times],
                              nfglevels=int(arguments.nfglevels), out_dir=out_dir,
                              worker=getattr(arguments, "rrfs_worker", None),
                              wif_climatology=getattr(arguments, "wif_climatology", None))
    except RrfsInitError as error:
        raise IntermediateRefusal(str(error)) from error
    files = [Path(path) for path in record["files"]]
    door_receipt = {
        "schema": "gpuwm-hex.lbc-door/v1", "route": "init-producer case 9",
        "physics_suite": arguments.physics_suite, "grid": str(grid), **record,
        "intermediates": [{"valid_time": f"{t:%Y-%m-%d_%H:%M:%S}", "path": str(p), "sha256": sha256_file(p)}
                          for t, p in times],
        "files": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files],
    }
    (out_dir / "lbc-door-receipt.json").write_text(
        json.dumps(door_receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"files": [str(p) for p in files], "seconds": record["seconds"],
                      "receipt": str(out_dir / "lbc-door-receipt.json")}, indent=2))
    print(f"NEXT woof hex forecast ... --lbc-dir {out_dir}")
    return 0


def run_lbc(arguments: argparse.Namespace) -> int:
    from .cycle.chain import resolve_lbc_engine
    from .cycle.errors import CycleRefusal

    grid = Path(arguments.grid).expanduser().absolute()
    if not grid.is_file():
        raise IntermediateRefusal(f"--grid {grid} is not a file")
    suite = getattr(arguments, "physics_suite", None)
    if suite:
        from .mpasphys.rrfs.cold_start_rrfs import RrfsInitError, producer_for_suite
        try:
            producer = producer_for_suite(suite, pin_set=getattr(arguments, "physics_pin_set", None))
        except RrfsInitError as error:
            raise IntermediateRefusal(str(error)) from error
        if producer is not None:
            if getattr(arguments, "stream", False):
                raise IntermediateRefusal(f"--stream publishes rw_mpas_lbc hours; {suite}'s boundaries come from "
                                          "its producer's case 9, which writes the whole series at once")
            return _run_producer_lbc(arguments, producer, grid)
    try:
        engine = resolve_lbc_engine(arguments.lbc_exe)
    except CycleRefusal as error:
        raise IntermediateRefusal(str(error)) from error
    if getattr(arguments, "stream", False):
        from .lbc_producer import run_stream
        return run_stream(arguments, engine)
    start = datetime.strptime(arguments.start_time, "%Y-%m-%d_%H:%M:%S")
    stop = datetime.strptime(arguments.stop_time, "%Y-%m-%d_%H:%M:%S")
    if stop <= start:
        raise IntermediateRefusal("--stop-time is not after --start-time")
    times = [(t, p) for t, p in intermediate_valid_times(arguments.met_dir) if start <= t <= stop]
    if not times or times[0][0] != start or times[-1][0] != stop:
        held = ", ".join(t.strftime("%Y-%m-%d_%H:%M:%S") for t, _ in intermediate_valid_times(arguments.met_dir))
        raise IntermediateRefusal(
            f"the intermediates in {arguments.met_dir} do not cover "
            f"{arguments.start_time}..{arguments.stop_time} at both ends; they hold "
            f"[{held}].  A boundary series with a missing end freezes the boundary "
            f"there without saying so"
        )
    gaps = {int((b - a).total_seconds()) for (a, _), (b, _) in zip(times, times[1:])}
    if len(gaps) != 1:
        raise IntermediateRefusal(
            f"the intermediates are not evenly spaced in time ({sorted(gaps)} s); "
            f"--fg-interval-seconds is one number"
        )
    interval = gaps.pop()
    from .lbc_producer import lbc_physics_mode
    physics, correct_record = lbc_physics_mode(engine, getattr(arguments, "physics_mode", "exact"))
    out_dir = Path(arguments.out_dir).expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt = out_dir / "lbc-receipt.json"

    def engine_argv(part: Sequence[tuple[datetime, Path]], part_out: Path, part_receipt: Path) -> list[str]:
        command = [
            str(engine), "--grid", str(grid), "--out-dir", str(part_out),
            "--start-time", arguments.start_time, "--stop-time", arguments.stop_time,
        ]
        for moment, path in part:
            command += ["--interval", f"{moment:%Y-%m-%d_%H:%M:%S}={path}"]
        return command + [
            "--nfglevels", str(int(arguments.nfglevels)),
            "--fg-interval-seconds", str(interval),
            "--extrap-airtemp", arguments.extrap_airtemp,
            "--use-spechumd", arguments.use_spechumd,
            "--theta-adv-order", str(int(arguments.theta_adv_order)),
            "--coef-3rd-order", repr(float(arguments.coef_3rd_order)),
            "--oned-underflow", arguments.oned_underflow,
            *physics,
            "--receipt", str(part_receipt),
        ]

    argv = engine_argv(times, out_dir, receipt)
    # One engine process per group of boundary times, at once. THE SINGLE-CORE STAGE THIS REMOVES: rw_mpas_lbc
    # writes its times one after another on one core (111 % CPU, 16.9 s for the 7 hourly times of a 6 h 96->3 km
    # region, an RTX 5090 host, 2026-10-29), and hourly table-source boundaries (init_sources.boundary_step) multiply them.
    # Each time's file depends only on the grid and that time's intermediate: one process per time wrote the same
    # bytes as one process for all seven (sha256, all 7 files) in 4.1 s.
    workers = lbc_workers(len(times), grid, [path for _, path in times], getattr(arguments, "workers", None))
    parts = [list(times[index::workers]) for index in range(workers)]
    started = time.perf_counter()
    if workers == 1:
        runs = [(argv, subprocess.run(argv, capture_output=True, text=True), 0.0)]
    else:
        def run_part(index: int):
            part_out = out_dir / f".part-{index:03d}"
            part_out.mkdir(parents=True, exist_ok=True)
            command = engine_argv(parts[index], part_out, part_out / "lbc-receipt.json")
            begun = time.perf_counter()
            return command, subprocess.run(command, capture_output=True, text=True), time.perf_counter() - begun

        with ThreadPoolExecutor(max_workers=workers) as pool:
            runs = list(pool.map(run_part, range(workers)))
    elapsed = time.perf_counter() - started
    if workers == 1:
        runs = [(runs[0][0], runs[0][1], elapsed)]
    (out_dir / "lbc.log").write_text("".join(
        f"$ {' '.join(command)}\n[{seconds:.1f} s, exit {completed.returncode}]\n--- stdout ---\n"
        f"{completed.stdout}\n--- stderr ---\n{completed.stderr}\n" for command, completed, seconds in runs)
        + (f"[{workers} processes, {elapsed:.1f} s wall]\n" if workers > 1 else ""), encoding="utf-8")
    for _, completed, _ in runs:
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "").strip().splitlines()
            raise IntermediateRefusal(
                f"rw_mpas_lbc exited {completed.returncode}; its log is {out_dir / 'lbc.log'}"
                + (f".  Last line: {tail[-1]}" if tail else "")
            )
    if workers > 1:
        engine_receipts = []
        for index in range(workers):
            part_out = out_dir / f".part-{index:03d}"
            for produced in sorted(part_out.glob("lbc.*.nc")):
                produced.replace(out_dir / produced.name)
            part_receipt = part_out / "lbc-receipt.json"
            engine_receipts.append(json.loads(part_receipt.read_text(encoding="utf-8")) if part_receipt.is_file() else None)
            for leftover in part_out.iterdir():
                leftover.unlink()
            part_out.rmdir()
        receipt.write_text(json.dumps({"schema": "gpuwm-hex.lbc-engine-parts/v1", "parts": engine_receipts},
                                      indent=2) + "\n", encoding="utf-8", newline="\n")
    files = sorted(out_dir.glob("lbc.*.nc"))
    if not files:
        raise IntermediateRefusal(f"rw_mpas_lbc exited 0 and wrote no lbc.*.nc into {out_dir}")
    door_receipt = {
        "schema": "gpuwm-hex.lbc-door/v1",
        "engine": str(engine), "engine_sha256": sha256_file(engine),
        "grid": str(grid), "argv": argv, "seconds": round(elapsed, 2), "processes": workers,
        "interval_seconds": interval,
        "intermediates": [{"valid_time": f"{t:%Y-%m-%d_%H:%M:%S}", "path": str(p), "sha256": sha256_file(p)}
                          for t, p in times],
        "files": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files],
        "engine_receipt": str(receipt) if receipt.is_file() else None,
        # Correct mode only (an exact door receipt is the one written before the key existed).
        **({"correct_mode": correct_record} if correct_record else {}),
    }
    (out_dir / "lbc-door-receipt.json").write_text(
        json.dumps(door_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n",
    )
    print(json.dumps({"files": [str(p) for p in files], "seconds": round(elapsed, 2),
                      "receipt": str(out_dir / "lbc-door-receipt.json")}, indent=2))
    print(f"NEXT woof hex forecast ... --lbc-dir {out_dir}")
    return 0


__all__ = [
    "SOIL_SETS",
    "cells_outside",
    "clip_to_source",
    "east_longitude",
    "west_longitude",
    "first_guess_levels",
    "mesh_cell_degrees",
    "mesh_cells_outside_met",
    "soil_set_for_suite",
    "target_from_mesh",
    "uncovered_cells_refusal",
    "DEFAULT_MARGIN_KM",
    "DEFAULT_SPACING_DEG",
    "INTERMEDIATE_SCHEMA",
    "SOURCE_ROWS",
    "SURFACE_LEVEL",
    "IntermediateRefusal",
    "IntermediateRequest",
    "LatLonTarget",
    "SourceRow",
    "SourceWindow",
    "add_intermediate_parser",
    "add_lbc_parser",
    "build_intermediates",
    "decode_window",
    "intermediate_valid_times",
    "parse_cycle",
    "parse_hours",
    "regrid_hour",
    "resolve_decoder",
    "run_intermediate",
    "run_lbc",
    "source_row",
    "source_window_for",
    "target_for_cap",
    "target_from_plan",
    "wps_record_bytes",
    "write_intermediate",
]
