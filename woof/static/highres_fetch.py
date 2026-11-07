"""Tiled, cached, provenance-recorded fetch for high-resolution geography.

This module turns one model-domain footprint (plus the geogrid processing
halo) into the concrete source artifacts :mod:`woof.static.highres`
consumes: terrain tiles (USGS 3DEP, Copernicus DEM, SRTM), one land-cover
raster (CGLC-MODIS-LCZ by default, or one Annual NLCD year), and SoilGrids
v2 sand/silt/clay WCS windows.  It is deliberately footprint-parametric --
nothing in here knows about any particular case or place; every
geographic number arrives from the caller's grid.

Contract:

- Sources that publish whole artifacts are fetched whole.  Terrain is
  fetched as complete 1x1-degree GeoTIFF tiles, CGLC-MODIS-LCZ as its one
  global GeoTIFF (pinned by size, MD5 and SHA-256) and Annual NLCD as the
  complete published CONUS year bundle; none is ever range-subset from
  the network.  SoilGrids is served by ISRIC's own WCS windowing service,
  which is the pilot-proven route for that source.
- Land-cover sources are table rows (:data:`LANDCOVER_SOURCES`): the
  fetch kind, the crosswalk into WRF's MODIS 21 categories, the water
  rule and the reference years are data, so another collection is a row,
  not a code path.
- Every fetched byte is hashed (SHA-256) at fetch time and the digest is
  recorded in a JSON sidecar next to the cached payload; receipts carry
  those digests.  Arbitrary user windows cannot be pre-pinned, so recorded
  provenance -- not a pinned manifest -- is the contract.
- A cache hit is a payload whose byte count and SHA-256 match its
  sidecar. Anything else is refetched (process-owned resumable staging,
  fsync and atomic rename), by one writer at a time per cached file; a
  preparation that finds another downloading the file waits for as long
  as that download keeps growing.
- Incomplete tile coverage refuses loudly, naming the missing tiles
  (:class:`CoverageError`).  A transient network fault is asked again
  with a bounded backoff, resuming the staged bytes; a network failure
  that outlasts it is :class:`HighresFetchRefusal`.  Transport failures
  are infrastructure faults, not coverage facts, and must never be
  converted into a silent fallback. The former ``fallback-30s`` option
  is retired; an all-water footprint succeeds without a fallback.
"""
from __future__ import annotations

# One remedy string for the whole geography stack; see geog_stack.
from .geog_stack import geog_unavailable_detail

import hashlib
import http.client
import json
import math
import os
import re
import secrets
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np

from woof import fetch_endpoints, fetch_guard
from woof.ingest.source_coverage import PreparationRefusal

from .highres import (CGLC_MODIS_LCZ_TO_MODIS21, NLCD_TO_MODIS21_INLAND,
                      WATER_FROM_SOURCE, WATER_RULES,
                      WATER_SPLIT_BY_BASELINE, sha256_file)
from .highres_refusal import HighresRefusal
from .projection import _wrap180

#: Read/stream chunk for downloads and hashing.
_CHUNK = 8 * 1024 * 1024

#: USGS 3DEP seamless 1/3 arc-second staged products (whole 1x1-degree
#: GeoTIFF tiles, public domain).
THREE_DEP_TILE_URL = ("https://prd-tnm.s3.amazonaws.com/StagedProducts/"
                      "Elevation/13/TIFF/current/{tile}/USGS_13_{tile}.tif")
THREE_DEP_SOURCE_URL = "https://www.usgs.gov/3d-elevation-program"
THREE_DEP_LICENSE = ("US-PD",
                     "https://www.usgs.gov/information-policies-and-"
                     "instructions/copyrights-and-credits")

#: Annual NLCD Collection 1 land cover, one whole published CONUS year
#: bundle per fetch (public domain).
ANNUAL_NLCD_URL = ("https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/"
                   "data-bundles/Annual_NLCD_LndCov_{year}_CU_C1V1.zip")
ANNUAL_NLCD_SOURCE_URL = "https://www.mrlc.gov/data"
ANNUAL_NLCD_LICENSE = ("US-PD", "https://www.mrlc.gov/data")
#: Years the Annual NLCD collection publishes.  Cases before the first
#: year take the earliest map and the receipt names the anachronism.
ANNUAL_NLCD_FIRST_YEAR = 1985
ANNUAL_NLCD_LAST_YEAR = 2024

#: CGLC-MODIS-LCZ, the hybrid 100 m global land cover WRF and WPS ship
#: from version 4.5 as tiled WPS data: the Copernicus Global Land Service
#: LC100 v3 map of 2018 in the MODIS IGBP legend, with the global Local
#: Climate Zone map's urban classes.  One GeoTIFF on the same grid as the WPS tiles,
#: fetched whole (anonymous HTTPS, resumable) and pinned.
CGLC_MODIS_LCZ_URL = ("https://zenodo.org/records/7670653/files/"
                      "CGLC_MODIS_LCZ.tif?download=1")
CGLC_MODIS_LCZ_SOURCE_URL = "https://doi.org/10.5281/zenodo.7670653"
CGLC_MODIS_LCZ_LICENSE = ("CC-BY-4.0",
                          "https://creativecommons.org/licenses/by/4.0/")
CGLC_MODIS_LCZ_ATTRIBUTION = (
    "CGLC-MODIS-LCZ, Demuzere M., He C., Martilli A. and Zonato A. (2023), "
    "doi:10.5281/zenodo.7670653, CC BY 4.0; built from the Copernicus "
    "Global Land Service LC100 v3 (Buchhorn et al. 2020) and the global "
    "Local Climate Zone map (Demuzere et al. 2022, Earth Syst. Sci. Data "
    "14, 3835).")
CGLC_MODIS_LCZ_BYTES = 2_281_913_068
CGLC_MODIS_LCZ_MD5 = "a757712949c23e5a3967aec69529a19e"
CGLC_MODIS_LCZ_SHA256 = (
    "4f8ea602a84ffc45ce82ee61d55970da22323618f9e8ece3b15c91de91942b01")
#: The map represents 2018 everywhere.
CGLC_MODIS_LCZ_YEAR = 2018

#: SoilGrids v2 250 m WCS (ISRIC; CC-BY-4.0).  The pilot-proven route.
SOILGRIDS_WCS_URL = (
    "https://maps.isric.org/mapserv?map=/map/{component}.map"
    "&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
    "&COVERAGEID={component}_{depth}_Q0.5&FORMAT=GEOTIFF_INT16"
    "&SUBSET=X({x0:.0f},{x1:.0f})&SUBSET=Y({y0:.0f},{y1:.0f})"
    "&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/152160"
    "&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/152160")
SOILGRIDS_SOURCE_URL = "https://www.isric.org/explore/soilgrids"
SOILGRIDS_LICENSE = ("CC-BY-4.0",
                     "https://creativecommons.org/licenses/by/4.0/")
#: Interrupted Goode Homolosine, SoilGrids' native grid ("EPSG" 152160 is
#: ISRIC's own registry entry, outside the GeoTIFF registry, so the
#: delivered tile cannot name it: until 2026-10-14 it carried no CRS keys
#: at all, and since 2026-10-15 it carries user-defined keys (GTModelType
#: 32767, an ESRI PE string naming Goode_Homolosine). Either way this
#: override, recorded on every bound raster, is the CRS of record).
SOILGRIDS_CRS = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
SOILGRIDS_COMPONENTS = ("clay", "sand", "silt")
SOILGRIDS_DEPTHS = ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm")
#: SoilGrids raw units are g/kg; the USDA triangle wants percent.
SOILGRIDS_SCALE = 0.1
#: The WCS delivers masked (water/ice/urban-core) pixels as 0 g/kg, which
#: is not a physical soil composition; treat it as the nodata sentinel.
SOILGRIDS_NODATA = 0.0
#: Fetch margin around the model footprint, metres on the IGH plane.
_SOILGRIDS_MARGIN_M = 2000.0
#: Snap WCS windows to whole kilometres so repeated preparations of the
#: same domain hit the cache instead of minting near-duplicate windows.
_SOILGRIDS_SNAP_M = 1000.0


#: Copernicus DEM GLO-30 (COG GeoTIFF, ~30 m, near-global), AWS Open Data.
#: Anonymous HTTP GET; no account, no token, no signing.  Tile ids name the
#: SOUTH-WEST corner: ``N39_00_W105_00`` spans 39..40 N, 105..104 W.
COPERNICUS_DEM_TILE_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{tile}_DEM/Copernicus_DSM_COG_10_{tile}_DEM.tif")
COPERNICUS_DEM_SOURCE_URL = (
    "https://registry.opendata.aws/copernicus-dem/")
COPERNICUS_DEM_LICENSE = (
    "Copernicus-DEM-EULA-free-open",
    "https://spacedata.copernicus.eu/documents/20123/121286/"
    "CSCDA_ESA_Mission-specific+Annex.pdf")
COPERNICUS_DEM_ATTRIBUTION = (
    "produced using Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and (c) "
    "Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by "
    "the European Union and ESA; all rights reserved")
#: Published tile labels run S90..N83 (south-west corners), so the product
#: reaches 84 N and 90 S -- not the whole globe.  Verified against the
#: bucket's own tileList.txt, not assumed.
COPERNICUS_DEM_LAT_MAX = 84.0
COPERNICUS_DEM_LAT_MIN = -90.0
#: Latitude spacing is 1/3600 degree everywhere.  LONGITUDE spacing is
#: latitude-banded (verified against the published tiles): 3600 columns
#: below 50 deg, 2400 in 50-60, 1800 in 60-70, and coarser toward the
#: poles.  A mosaic therefore has to declare its own output resolution
#: instead of inheriting the first tile's.
COPERNICUS_DEM_LAT_STEP_DEG = 1.0 / 3600.0
#: Elevations are metres above the EGM2008 geoid -- the same vertical sense
#: as WPS terrain, so no datum shift is applied or needed.
COPERNICUS_DEM_VERTICAL_DATUM = "EGM2008 geoid (orthometric metres)"

#: SRTM 1 arc-second (SRTMGL1 v3).  Wired as a named alternative because
#: users ask for it by name.  NASA's own LP DAAC distribution needs an
#: Earthdata login, which this program will not require of anyone; the
#: OpenTopography S3-compatible mirror serves the same raw v3 tiles
#: anonymously and is what is fetched here.  Tile ids name the south-west
#: corner, 3601x3601 int16, nodata -32768, 1/3600 degree in BOTH axes at
#: every latitude (unlike Copernicus).
SRTM_GL1_TILE_URL = ("https://opentopography.s3.sdsc.edu/raster/SRTM_GL1/"
                     "SRTM_GL1_srtm/{tile}.tif")
SRTM_GL1_SOURCE_URL = "https://portal.opentopography.org/raster?opentopoID=OTSRTM.082015.4326.1"
SRTM_GL1_LICENSE = ("US-PD", "https://lpdaac.usgs.gov/data/data-citation-"
                             "and-policies/")
SRTM_GL1_ATTRIBUTION = (
    "NASA Shuttle Radar Topography Mission Global 1 arc second "
    "(SRTMGL1 v3), NASA JPL 2013, doi:10.5067/MEaSUREs/SRTM/SRTMGL1.003; "
    "distributed by OpenTopography, doi:10.5069/G9445JDF")
SRTM_GL1_STEP_DEG = 1.0 / 3600.0
SRTM_GL1_NODATA = -32768.0
#: SRTM heights are metres above the EGM96 geoid; Copernicus uses EGM2008.
#: The two differ by up to a few metres regionally, which is why a run
#: names its terrain source in the receipt instead of calling them
#: interchangeable.
SRTM_GL1_VERTICAL_DATUM = "EGM96 geoid (orthometric metres)"


class CoverageError(RuntimeError):
    """A source does not cover the requested footprint; names what is
    missing."""


# ---------------------------------------------------------------------------
# The Rust seam (fixed-means-default).  This module is the network
# DRIVER -- URL loops, resumable staging, sha256 sidecars, cache
# admission -- and that stays Python: the bytes it moves are opaque
# payloads written to disk verbatim.  What is NOT orchestration is the
# derivation of a cached window from those payloads (decode, mosaic,
# void fill, clip, re-emit) and the projection of a footprint onto a
# source CRS.  Both route to the static-fields cdylib by default; the
# rasterio/pyproj bodies stay as the parity reference and the reported
# fallback (WOOF_STATIC_PYTHON=1 or an unloadable library).
# ---------------------------------------------------------------------------

def _static_rust(operation: str):
    """The loaded bridge module, or None with the workaround reported."""
    from . import rust_bridge

    return rust_bridge.route(operation)


@dataclass(frozen=True)
class FootprintBBox:
    """Geographic bounding box of one model domain plus processing halo."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def __post_init__(self):
        if not (math.isfinite(self.lat_min) and math.isfinite(self.lat_max)
                and math.isfinite(self.lon_min)
                and math.isfinite(self.lon_max)):
            raise ValueError(f"footprint bbox is not finite: {self}")
        if self.lat_min >= self.lat_max or self.lon_min >= self.lon_max:
            raise ValueError(f"footprint bbox is degenerate: {self}")

    def padded(self, degrees: float) -> "FootprintBBox":
        return FootprintBBox(self.lat_min - degrees, self.lat_max + degrees,
                             self.lon_min - degrees, self.lon_max + degrees)

    def contains(self, other: "FootprintBBox") -> bool:
        return (self.lat_min <= other.lat_min
                and self.lat_max >= other.lat_max
                and self.lon_min <= other.lon_min
                and self.lon_max >= other.lon_max)

    def intersects(self, other: "FootprintBBox") -> bool:
        return (self.lat_min < other.lat_max
                and other.lat_min < self.lat_max
                and self.lon_min < other.lon_max
                and other.lon_min < self.lon_max)

    def as_dict(self) -> dict[str, float]:
        return {"lat_min": self.lat_min, "lat_max": self.lat_max,
                "lon_min": self.lon_min, "lon_max": self.lon_max}


@dataclass(frozen=True)
class SourceCoverage:
    """Where one named source is published, and under what terms.

    Coverage is a property of the source, not of the program: each source
    declares its own envelope and a footprint is checked against the source
    actually selected, so a refusal can say *which* dataset does not reach
    *where*.  ``global_lon`` marks sources published for every longitude,
    which is the normal case outside the US collections.
    """

    source_id: str
    role: str
    envelope: FootprintBBox
    nominal_resolution: str
    source_url: str
    license_id: str
    license_url: str
    attribution: str = ""
    note: str = ""
    #: The source is published for EVERY longitude, so no footprint can
    #: leave it east or west.  Without this a dateline footprint, whose
    #: longitude range is continued past 180 by a fraction of a degree
    #: (see :func:`domain_footprint`), would be reported as leaving an
    #: envelope that in truth wraps around and meets itself.
    global_lon: bool = False

    def outside(self, bbox: FootprintBBox) -> dict[str, float]:
        """Per-edge overshoot in degrees; empty when fully covered."""
        env, out = self.envelope, {}
        if bbox.lat_min < env.lat_min:
            out["south_by_deg"] = round(env.lat_min - bbox.lat_min, 6)
        if bbox.lat_max > env.lat_max:
            out["north_by_deg"] = round(bbox.lat_max - env.lat_max, 6)
        if not self.global_lon:
            if bbox.lon_min < env.lon_min:
                out["west_by_deg"] = round(env.lon_min - bbox.lon_min, 6)
            if bbox.lon_max > env.lon_max:
                out["east_by_deg"] = round(bbox.lon_max - env.lon_max, 6)
        return out

    def reaches(self, bbox: FootprintBBox) -> bool:
        """True when any part of ``bbox`` lies inside the envelope.

        A footprint that only partly overlaps is built: the cells the
        source does not cover take the 30-arc-second baseline
        (:mod:`woof.static.highres`).  Only a footprint wholly outside
        has nothing to take from the source.
        """
        env = self.envelope
        if not (bbox.lat_min < env.lat_max and env.lat_min < bbox.lat_max):
            return False
        if self.global_lon:
            return True
        return bbox.lon_min < env.lon_max and env.lon_min < bbox.lon_max

    def require_reach(self, bbox: FootprintBBox) -> None:
        """Raise :class:`CoverageError` when ``bbox`` lies wholly outside."""
        if self.reaches(bbox):
            return
        raise CoverageError(
            f"source {self.source_id!r} ({self.role}, "
            f"{self.nominal_resolution}) is published over "
            f"{self.envelope.as_dict()}; the domain+halo footprint "
            f"{bbox.as_dict()} lies wholly outside it (by "
            f"{self.outside(bbox)}), so it has no cell of this domain to "
            "supply. "
            + (f"{self.note} " if self.note else "")
            + f"Publication terms: {self.source_url}")

    def check(self, bbox: FootprintBBox) -> None:
        """Raise :class:`CoverageError` naming source, footprint, overshoot."""
        overshoot = self.outside(bbox)
        if not overshoot:
            return
        raise CoverageError(
            f"source {self.source_id!r} ({self.role}, "
            f"{self.nominal_resolution}) is published over "
            f"{self.envelope.as_dict()}; the domain+halo footprint "
            f"{bbox.as_dict()} leaves it by {overshoot}. "
            + (f"{self.note} " if self.note else "")
            + f"Publication terms: {self.source_url}")

    def echo(self) -> dict[str, object]:
        return {"source_id": self.source_id, "role": self.role,
                "envelope": self.envelope.as_dict(),
                "nominal_resolution": self.nominal_resolution,
                "source_url": self.source_url,
                "license_id": self.license_id,
                "license_url": self.license_url,
                "attribution": self.attribution, "note": self.note,
                "global_lon": self.global_lon}


#: Envelope where the two US collections are JOINTLY published: 3DEP staged
#: 1/3 arc-second tiles and the Annual NLCD conterminous-US collection.
_US_ENVELOPE = FootprintBBox(lat_min=24.0, lat_max=49.5,
                             lon_min=-125.0, lon_max=-66.5)
#: Copernicus GLO-30's published extent.  Individual tiles can still be
#: absent (the product does not publish all-water tiles); that is a
#: tile-level fact checked at fetch time, not an envelope fact.
_COPERNICUS_ENVELOPE = FootprintBBox(
    lat_min=COPERNICUS_DEM_LAT_MIN, lat_max=COPERNICUS_DEM_LAT_MAX,
    lon_min=-180.0, lon_max=180.0)

#: Terrain sources, keyed by the id users write in ``terrain_source``.
TERRAIN_SOURCES: dict[str, SourceCoverage] = {
    "usgs-3dep-13as": SourceCoverage(
        source_id="usgs-3dep-13as", role="terrain",
        envelope=_US_ENVELOPE,
        nominal_resolution="1/3 arc-second (~10 m)",
        source_url=THREE_DEP_SOURCE_URL,
        license_id=THREE_DEP_LICENSE[0], license_url=THREE_DEP_LICENSE[1],
        attribution="USGS 3D Elevation Program (public domain).",
        note="3DEP staged 1/3 arc-second tiles are a United States "
             "collection; outside the US use terrain_source = "
             "\"copernicus-dem-glo30\"."),
    "copernicus-dem-glo30": SourceCoverage(
        source_id="copernicus-dem-glo30", role="terrain",
        envelope=_COPERNICUS_ENVELOPE,
        nominal_resolution="1 arc-second latitude (~30 m)",
        source_url=COPERNICUS_DEM_SOURCE_URL,
        license_id=COPERNICUS_DEM_LICENSE[0],
        license_url=COPERNICUS_DEM_LICENSE[1],
        attribution=COPERNICUS_DEM_ATTRIBUTION,
        note="Published south-west tile labels run S90..N83, so the "
             "product reaches 84 N and 90 S.  All-water tiles are not "
             "published, which this path treats as sea level only where "
             "the domain's own baseline land mask already says water.",
        global_lon=True),
    "srtm-gl1": SourceCoverage(
        source_id="srtm-gl1", role="terrain",
        envelope=FootprintBBox(lat_min=-56.0, lat_max=60.0,
                               lon_min=-180.0, lon_max=180.0),
        nominal_resolution="1 arc-second (~30 m)",
        source_url=SRTM_GL1_SOURCE_URL,
        license_id=SRTM_GL1_LICENSE[0], license_url=SRTM_GL1_LICENSE[1],
        attribution=SRTM_GL1_ATTRIBUTION,
        note="SRTM stops at 60 N / 56 S: it does not reach Canada, "
             "Scandinavia, Alaska or most of Russia.  Fetched from the "
             "anonymous OpenTopography mirror because NASA's own "
             "distribution requires an Earthdata login.",
        global_lon=True),
}

#: Land-cover fetch kinds.  Each is a way a collection is PUBLISHED, not a
#: collection: another source published the same way is a table row.
#: ``yearly-zip-bundle``: one zip per year around one GeoTIFF (``{year}``
#: in the URL).  ``whole-geotiff``: one GeoTIFF fetched whole, pinned by
#: size, MD5 and SHA-256.
LANDCOVER_FETCH_KINDS = ("yearly-zip-bundle", "whole-geotiff")


@dataclass(frozen=True)
class LandcoverSource:
    """One land-cover collection, as data the production path reads.

    Where it is published and under what terms (``coverage``), how it is
    fetched, how its raw classes reach WRF's MODIS 21 categories
    (``crosswalk``), how its water reaches WRF ocean and lake (``water``,
    one of :data:`woof.static.highres.WATER_RULES`) and which years it
    represents.  A case outside ``first_year..last_year`` takes the nearest
    year, and the receipt names the anachronism.
    """

    coverage: SourceCoverage
    fetch: str
    url: str
    cache_dir: str
    crosswalk: Mapping[int, int]
    water: str
    first_year: int
    last_year: int
    #: Local file name of a ``whole-geotiff`` payload.
    file_name: str = ""
    pinned_bytes: int | None = None
    pinned_md5: str | None = None
    pinned_sha256: str | None = None
    #: Short name used in console lines and anachronism statements.
    label: str = ""
    #: The raw class list a reader can check the crosswalk against.
    legend: str = ""
    #: Raw value that marks an unclassified pixel when the file carries no
    #: nodata tag of its own.  Such pixels reach no model cell, so the
    #: cells they cover take the 30-arc-second baseline.
    nodata: float | None = None
    #: The WPS ``geog_data_res`` tokens (GEOGRID.TBL ``rel_path`` keys)
    #: that select this same collection in WPS.  A namelist.wps naming
    #: one is honoured through ``[static.highres]`` with this row as its
    #: ``landcover_source``: ``woof import-namelist`` writes that block,
    #: and the static builder admits the token where the block selects
    #: this row (:class:`woof.static.build.GeogSelection`).  Empty for a
    #: collection WPS has no token for.  Not part of :meth:`echo`: it says
    #: how a WPS namelist names the collection, not what is built.
    wps_geog_tokens: tuple[str, ...] = ()

    def __post_init__(self):
        if self.fetch not in LANDCOVER_FETCH_KINDS:
            raise ValueError(
                f"land-cover source {self.source_id!r}: fetch kind "
                f"{self.fetch!r} is not one of {list(LANDCOVER_FETCH_KINDS)}")
        if self.water not in WATER_RULES:
            raise ValueError(
                f"land-cover source {self.source_id!r}: water rule "
                f"{self.water!r} is not one of {list(WATER_RULES)}")
        if self.fetch == "whole-geotiff" and not self.file_name:
            raise ValueError(
                f"land-cover source {self.source_id!r} is fetched whole "
                "and names no file")
        if self.first_year > self.last_year:
            raise ValueError(
                f"land-cover source {self.source_id!r}: first year "
                f"{self.first_year} is after last year {self.last_year}")

    @property
    def source_id(self) -> str:
        return self.coverage.source_id

    def year_for(self, case_date: date) -> tuple[int, int]:
        """(represented year nearest the case date, anachronism in years)."""
        year = min(max(int(case_date.year), self.first_year),
                   self.last_year)
        return year, abs(int(case_date.year) - year)

    def bound_id(self, year: int) -> str:
        """The source id a bound raster and the receipt carry."""
        return f"{self.source_id}-{int(year)}"

    def echo(self) -> dict[str, object]:
        return {**self.coverage.echo(), "fetch": self.fetch,
                "url": self.url, "water": self.water,
                "years": [self.first_year, self.last_year],
                "pinned_bytes": self.pinned_bytes,
                "pinned_md5": self.pinned_md5,
                "pinned_sha256": self.pinned_sha256,
                "legend": self.legend, "nodata": self.nodata,
                "crosswalk": {str(raw): int(target) for raw, target
                              in sorted(self.crosswalk.items())}}


#: CGLC-MODIS-LCZ's published extent: the GeoTIFF runs from 78 N to 60 S
#: at every longitude.  WRF's tiled dataset fills poleward of that from
#: MODIS; here the 30-arc-second MODIS baseline does the same through the
#: coverage fallback.
_CGLC_ENVELOPE = FootprintBBox(lat_min=-60.0, lat_max=78.0,
                               lon_min=-180.0, lon_max=180.0)

#: Land-cover sources, keyed by the id users write in ``landcover_source``.
LANDCOVER_SOURCES: dict[str, LandcoverSource] = {
    "cglc-modis-lcz": LandcoverSource(
        coverage=SourceCoverage(
            source_id="cglc-modis-lcz", role="landcover",
            envelope=_CGLC_ENVELOPE,
            nominal_resolution="100 m (0.000898 degree)",
            source_url=CGLC_MODIS_LCZ_SOURCE_URL,
            license_id=CGLC_MODIS_LCZ_LICENSE[0],
            license_url=CGLC_MODIS_LCZ_LICENSE[1],
            attribution=CGLC_MODIS_LCZ_ATTRIBUTION,
            note="CGLC-MODIS-LCZ is published from 60 S to 78 N at every "
                 "longitude; poleward of that the land-use fields take the "
                 "30-arc-second MODIS baseline.",
            global_lon=True),
        fetch="whole-geotiff", url=CGLC_MODIS_LCZ_URL,
        cache_dir="cglc_modis_lcz", file_name="CGLC_MODIS_LCZ.tif",
        pinned_bytes=CGLC_MODIS_LCZ_BYTES, pinned_md5=CGLC_MODIS_LCZ_MD5,
        pinned_sha256=CGLC_MODIS_LCZ_SHA256,
        crosswalk=CGLC_MODIS_LCZ_TO_MODIS21, water=WATER_FROM_SOURCE,
        first_year=CGLC_MODIS_LCZ_YEAR, last_year=CGLC_MODIS_LCZ_YEAR,
        label="CGLC-MODIS-LCZ",
        legend="MODIS IGBP 1-20, 17 sea, 21 inland water, 51-61 Local "
               "Climate Zones LCZ 1-10 and LCZ E, 0 unclassified (the "
               "open sea past the collection's coastal zone)",
        nodata=0.0,
        # WPS v4.6.0 geogrid/GEOGRID.TBL.ARW_LCZ: a priority-2 LANDUSEF
        # entry "rel_path = cglc_modis_lcz:CGLC_MODIS_LCZ_global/" (the
        # same Zenodo 7670653 collection, tiled for geogrid) over the
        # priority-1 MODIS entry, which fills where it has no data.
        wps_geog_tokens=("cglc_modis_lcz",)),
    "annual-nlcd": LandcoverSource(
        coverage=SourceCoverage(
            source_id="annual-nlcd", role="landcover",
            envelope=_US_ENVELOPE, nominal_resolution="30 m",
            source_url=ANNUAL_NLCD_SOURCE_URL,
            license_id=ANNUAL_NLCD_LICENSE[0],
            license_url=ANNUAL_NLCD_LICENSE[1],
            attribution="Annual NLCD Collection 1, MRLC (public domain).",
            note="Annual NLCD is a conterminous-United-States collection; "
                 "for a domain outside it use landcover_source = "
                 "\"cglc-modis-lcz\" (the default)."),
        fetch="yearly-zip-bundle", url=ANNUAL_NLCD_URL,
        cache_dir="annual_nlcd", crosswalk=NLCD_TO_MODIS21_INLAND,
        water=WATER_SPLIT_BY_BASELINE,
        first_year=ANNUAL_NLCD_FIRST_YEAR, last_year=ANNUAL_NLCD_LAST_YEAR,
        label="Annual NLCD",
        legend="NLCD Anderson classes 11-95"),
}

#: What ``landcover_source = "auto"`` selects: the global collection,
#: everywhere.  Annual NLCD stays selectable by name inside the US.
DEFAULT_LANDCOVER_SOURCE = "cglc-modis-lcz"


def landcover_source(source_id: str) -> LandcoverSource:
    """Look up one land-cover source (``auto`` is the default), refusing
    an unknown id by name."""
    if source_id == "auto":
        source_id = DEFAULT_LANDCOVER_SOURCE
    try:
        return LANDCOVER_SOURCES[source_id]
    except KeyError:
        raise CoverageError(
            f"unknown land-cover source {source_id!r}; known sources are "
            f"{sorted(LANDCOVER_SOURCES)}") from None


def landcover_source_ids_by_wps_token() -> dict[str, str]:
    """``{WPS geog_data_res token: landcover_source id}`` for every row of
    :data:`LANDCOVER_SOURCES` that WPS names (``wps_geog_tokens``).

    Tokens are matched lower-case, the way
    :meth:`woof.static.build.GeogSelection.from_tokens` normalizes them.
    A collection reaches this map by its table row, never by a code path.
    """
    return {token.lower(): source_id
            for source_id, row in LANDCOVER_SOURCES.items()
            for token in row.wps_geog_tokens}


def terrain_source_coverage(source_id: str) -> SourceCoverage:
    """Look up one terrain source, refusing an unknown id by name."""
    try:
        return TERRAIN_SOURCES[source_id]
    except KeyError:
        raise CoverageError(
            f"unknown terrain source {source_id!r}; known sources are "
            f"{sorted(TERRAIN_SOURCES)}") from None


def _continued_longitude_range(lon) -> tuple[float, float]:
    """(lon_min, lon_max) of a corner mesh, continued across the dateline.

    ``ij_to_latlon`` returns longitudes cut into (-180, 180], so a domain
    straddling 180 degrees comes back bimodal against a gap and a plain
    min/max collapses to the whole planet.  Each longitude is re-expressed
    as the shortest signed offset from one member of the mesh
    (:func:`woof.static.projection._wrap180`, the existing helper for that
    arithmetic), which recovers the true range exactly for any footprint
    narrower than 180 degrees, and the range is then shifted so ``lon_min``
    lands in (-180, 180].  A dateline domain therefore reports a CONTINUED
    range such as 179.28 .. 180.72 rather than -180 .. 180.

    A footprint genuinely 180 degrees or wider in longitude (a domain that
    encloses a projection pole is the realistic case) has no continued
    representation at all, and is reported as the full longitude band so
    the per-source coverage check sees what it really is.
    """
    lon = np.asarray(lon, dtype=np.float64)
    anchor = float(lon.ravel()[0])
    continued = anchor + _wrap180(lon - anchor)
    lon_min = float(np.min(continued))
    lon_max = float(np.max(continued))
    if lon_max - lon_min >= 180.0:
        return -180.0, 180.0
    while lon_min <= -180.0:
        lon_min, lon_max = lon_min + 360.0, lon_max + 360.0
    while lon_min > 180.0:
        lon_min, lon_max = lon_min - 360.0, lon_max - 360.0
    return lon_min, lon_max


def domain_footprint(grid, halo: int, margin_deg: float = 0.03
                     ) -> FootprintBBox:
    """Geographic bbox of the grid extended by ``halo`` cells + margin.

    Uses the cell-corner mesh of the extended grid (the same support the
    static builder samples) so the fetched sources cover every source pixel
    any halo cell can accumulate.

    The longitude range is CONTINUED across the dateline
    (:func:`_continued_longitude_range`): a domain on 180 degrees reports
    the degree and a half it actually occupies, and ``lon_max`` may exceed
    180.  Every consumer -- the coverage envelopes, the one-degree tile
    enumerators, the window mosaic -- reads the same continued frame.
    """
    if halo < 0:
        raise ValueError("halo must be non-negative")
    nx, ny = grid.e_we - 1, grid.e_sn - 1
    xc, yc = np.meshgrid(
        np.arange(0.5 - halo, nx + halo + 1.0, dtype=np.float64),
        np.arange(0.5 - halo, ny + halo + 1.0, dtype=np.float64))
    lat, lon = grid.ij_to_latlon(xc, yc)
    lon_min, lon_max = _continued_longitude_range(lon)
    return FootprintBBox(
        float(np.min(lat)), float(np.max(lat)),
        lon_min, lon_max).padded(margin_deg)


@dataclass(frozen=True)
class FetchedFile:
    """One cached payload with fetch-time provenance."""

    path: Path
    url: str
    sha256: str
    bytes: int
    fetched_utc: str
    cache_hit: bool

    def receipt(self) -> dict[str, object]:
        return {
            "path": str(Path(self.path).resolve()),
            "url": self.url,
            "sha256": self.sha256,
            "bytes": int(self.bytes),
            "fetched_utc": self.fetched_utc,
            "cache_hit": bool(self.cache_hit),
        }


class SourceAbsent(RuntimeError):
    """The source authoritatively reports the artifact does not exist."""


class RangeExhausted(RuntimeError):
    """A resume offset sits at/after the payload end (HTTP 416)."""


#: What to do about :class:`HighresFetchRefusal` when nothing more
#: specific is known. Network outages remain infrastructure failures;
#: the former coverage fallback is retired (see the module contract).
HIGHRES_FETCH_REMEDY = (
    "remedy: check this computer's connection to the host named above and "
    "prepare again; the high-resolution files already downloaded stay in "
    "their cache and a partly downloaded file resumes where it stopped.  "
    "To prepare without them, leave [static.highres] disabled and run on "
    "the 30-arc-second baseline.")


class HighresFetchRefusal(PreparationRefusal):
    """A high-resolution geography file could not be downloaded.

    The concrete breakage it replaces: one TLS connection reset from the
    terrain tile host ended whole preparations as a raw ``URLError``
    traceback at exit 1.  A preparation door owns this class (two
    lines, the message and the remedy), and it is deliberately not a
    :class:`HighresRefusal`: a transport failure must not be represented
    as a source-coverage fact or silently replaced by baseline terrain.
    """

    remedy = HIGHRES_FETCH_REMEDY


#: Transfers of one file before a network fault ends the preparation:
#: the schedule every source's fetch shares
#: (:data:`woof.fetch_endpoints.TRANSIENT_ATTEMPTS`, 2, 4, 8 and 16 s
#: apart).  A preparation fetches dozens of tiles back to back, and losing
#: any one of them ends the whole preparation, so it never takes fewer
#: tries than a source fetch does.
FETCH_ATTEMPTS = fetch_endpoints.TRANSIENT_ATTEMPTS

#: The longest ``Retry-After`` a host may ask for and still be waited out.
FETCH_RETRY_WAIT_LIMIT_S = fetch_endpoints.TRANSIENT_WAIT_LIMIT_S

#: The pause between attempts.  A module attribute so a test records the
#: backoff instead of sleeping through it.
_sleep = time.sleep

#: Failures that are the network's, as opposed to this computer's (a full
#: disk or an unwritable cache is an ``OSError`` too, and propagates).
_NETWORK_FAULTS = (urllib.error.URLError, http.client.HTTPException,
                   ConnectionError, TimeoutError)

#: The ``fetch_guard`` lock kind that makes one process the writer of one
#: cached file.
_FETCH_LOCK_KIND = "highres-fetch"


def _network_reason(error: BaseException) -> str:
    """One line naming what the network did."""
    if isinstance(error, http.client.IncompleteRead):
        missing = getattr(error, "expected", None)
        return ("the response ended early" if not missing else
                f"the response ended {missing} bytes early")
    reason = fetch_endpoints.fault_reason(error)
    if reason is not None:
        return reason
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTP {error.code}"
    return f"{type(error).__name__}: {error}"


def _staged_bytes(partial: Path) -> int:
    try:
        return partial.stat().st_size
    except OSError:
        return 0


def _default_urlopen(url: str, offset: int):
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        return urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        if error.code in (403, 404):
            # S3 buckets report absent keys as 403 without list permission;
            # both codes mean "this artifact is not published here".
            raise SourceAbsent(f"{url} -> HTTP {error.code}") from error
        if error.code == 416 and offset:
            match = re.fullmatch(r"bytes \*/(\d+)",
                                 error.headers.get("Content-Range", ""))
            if match and int(match.group(1)) == offset:
                raise RangeExhausted(
                    f"{url} -> HTTP 416 at complete offset {offset}") from error
        raise


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256.json")


def _read_sidecar(path: Path) -> dict | None:
    sidecar = _sidecar(path)
    if not path.is_file() or not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        if payload.get("bytes") != path.stat().st_size:
            return None
        digest = payload.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return None
        if sha256_file(path) != digest:
            return None
    except (OSError, ValueError):
        return None
    return payload


def _write_sidecar(path: Path, payload: dict) -> None:
    fetch_guard.atomic_write_text(
        _sidecar(path), json.dumps(payload, indent=2, sort_keys=True) + "\n",
        tag="sha256")


def _new_partial(path: Path) -> Path:
    """Exclusively claim a process-owned sibling of a cached payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        partial = path.with_name(
            f"{path.name}.partial-{os.getpid()}-{secrets.token_hex(6)}")
        try:
            with partial.open("xb"):
                pass
        except FileExistsError:
            continue
        return partial


def _partial_progress(path: Path) -> tuple:
    """Observe private staging files without opening or changing them."""
    try:
        stages = tuple(sorted(
            (partial.name, _staged_bytes(partial))
            for partial in path.parent.glob(path.name + ".partial-*")))
    except OSError:
        stages = ()
    return stages, _staged_bytes(path)


def _publish_partial(partial: Path, path: Path, *, url: str,
                     expected_bytes: int | None = None,
                     expected_sha256: str | None = None,
                     expected_md5: str | None = None) -> FetchedFile:
    """Validate and fsync private bytes before exposing a cached payload."""
    size = partial.stat().st_size
    consequence = ("; the staged payload was rejected before publication; "
                   "the next preparation fetches it again")
    if not size:
        raise ValueError(f"{path.name} received an empty payload" + consequence)
    if expected_bytes is not None and size != expected_bytes:
        raise ValueError(f"{path.name} is pinned at {expected_bytes} bytes "
                         f"and the staged payload holds {size}" + consequence)
    digest = sha256_file(partial)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"{path.name} is pinned at SHA-256 {expected_sha256} "
                         f"and the staged payload hashes to {digest}" + consequence)
    if expected_md5 is not None:
        observed_md5 = _md5_file(partial)
        if observed_md5 != expected_md5:
            raise ValueError(f"{path.name} is published with MD5 {expected_md5} "
                             f"and the staged payload hashes to {observed_md5}" + consequence)
    with partial.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, path)
    fetch_guard._fsync_dir(path.parent)
    record = FetchedFile(
        path=path, url=url, sha256=digest, bytes=size,
        fetched_utc=datetime.now(timezone.utc).isoformat(), cache_hit=False)
    _write_sidecar(path, {
        "url": url, "sha256": digest, "bytes": size,
        "fetched_utc": record.fetched_utc})
    return record


@contextmanager
def _derivation_writer(path: Path):
    """Serialize a derived cache entry and own its unpublished staging file."""
    with _one_writer(path):
        partial = _new_partial(path)
        try:
            yield partial
        finally:
            partial.unlink(missing_ok=True)


def record_local_artifact(path: Path, *, url: str,
                          cache_hit: bool = False) -> FetchedFile:
    """Hash one locally produced payload and write its sidecar."""
    path = Path(path)
    digest = sha256_file(path)
    record = FetchedFile(
        path=path, url=url, sha256=digest, bytes=path.stat().st_size,
        fetched_utc=datetime.now(timezone.utc).isoformat(),
        cache_hit=cache_hit)
    _write_sidecar(path, {
        "url": url, "sha256": digest, "bytes": record.bytes,
        "fetched_utc": record.fetched_utc})
    return record


def _cache_hit(path: Path, url: str) -> FetchedFile | None:
    cached = _read_sidecar(path)
    if cached is None:
        return None
    return FetchedFile(
        path=path, url=str(cached.get("url", url)),
        sha256=cached["sha256"], bytes=int(cached["bytes"]),
        fetched_utc=str(cached.get("fetched_utc", "")), cache_hit=True)


def _declared_length(response) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("Content-Length")
        return None if raw is None else int(raw)
    except (AttributeError, TypeError, ValueError):
        return None


def _transfer(opener, url: str, partial: Path) -> int:
    """One request, appended to what ``partial`` already holds.

    Returns once ``partial`` holds the whole payload.  A body shorter
    than the ``Content-Length`` the host declared raises
    :class:`http.client.IncompleteRead`: ``HTTPResponse.read(n)`` reports
    a connection closed mid-body as a quiet end of data, and a tile cut
    short that way used to be renamed into the cache and hashed as if it
    were whole.
    """
    offset = _staged_bytes(partial)
    try:
        response = opener(url, offset)
    except RangeExhausted:
        # The production opener accepts this only when the host's total
        # byte count equals our offset.
        return offset
    status = int(getattr(response, "status", 200) or 200)
    mode = "ab" if (offset and status == 206) else "wb"
    declared = _declared_length(response)
    total = declared if mode == "wb" else (
        None if declared is None else offset + declared)
    if status == 206:
        headers = getattr(response, "headers", {})
        content_range = headers.get("Content-Range", "")
        if content_range:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if (match is None or int(match.group(1)) != offset
                    or int(match.group(2)) + 1 != int(match.group(3))
                    or (declared is not None and
                        int(match.group(2)) - offset + 1 != declared)):
                response.close()
                raise http.client.HTTPException(
                    f"invalid Content-Range {content_range!r} at offset {offset}")
            total = int(match.group(3))
    received = 0
    with response, partial.open(mode) as stream:
        while True:
            block = response.read(_CHUNK)
            if not block:
                break
            stream.write(block)
            received += len(block)
        stream.flush()
        os.fsync(stream.fileno())
    if declared is not None and received != declared:
        if received < declared:
            raise http.client.IncompleteRead(b"", declared - received)
        raise http.client.HTTPException(
            f"response delivered {received} bytes, declared {declared}")
    size = partial.stat().st_size
    if total is not None and size != total:
        raise http.client.HTTPException(
            f"staged payload holds {size} bytes, response declared {total}")
    return size


def _resume_sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".resume.json")


def _remember_partial(path: Path, partial: Path, url: str) -> None:
    """Record a verified private partial for a later preparation to copy."""
    fetch_guard.atomic_write_text(_resume_sidecar(path), json.dumps({
        "url": url, "partial": partial.name,
        "bytes": _staged_bytes(partial), "sha256": sha256_file(partial),
    }, sort_keys=True) + "\n", tag="resume")


def _resume_partial(path: Path, partial: Path, url: str) -> None:
    """Copy abandoned bytes into this process's own staging file under lock."""
    resume = _resume_sidecar(path)
    try:
        saved = json.loads(resume.read_text(encoding="utf-8"))
        name = saved["partial"]
        if (saved.get("url") != url or not isinstance(name, str)
                or Path(name).name != name
                or not name.startswith(path.name + ".partial-")):
            return
        previous = path.parent / name
        if (previous == partial or previous.stat().st_size != saved["bytes"]
                or sha256_file(previous) != saved["sha256"]):
            return
    except (OSError, ValueError, KeyError, TypeError):
        # Missing or damaged abandoned staging is never published.
        return
    with previous.open("rb") as source, partial.open("wb") as target:
        shutil.copyfileobj(source, target, _CHUNK)
        target.flush()
        os.fsync(target.fileno())
    previous.unlink()
    resume.unlink(missing_ok=True)


def _fetch_refusal(path: Path, url: str, partial: Path, error: BaseException,
                   *, attempts: int, waited_s: float) -> HighresFetchRefusal:
    host = urllib.parse.urlsplit(url).netloc or url
    message = (f"[static.highres] could not download {path.name} from "
               f"{host}: {_network_reason(error)}")
    if attempts > 1:
        message += (f" (the last of {attempts} attempts, "
                    f"{waited_s:g} s apart in total)")
    kept = _staged_bytes(partial)
    remedy = (f"remedy: check this computer's connection to {host} and "
              f"prepare again; the files already downloaded stay in "
              f"{path.parent}")
    remedy += (f", and the {kept} verified staged bytes of {path.name} "
               "resume where they stopped" if kept else "")
    remedy += (".  To prepare without them, leave [static.highres] "
               "disabled and run on the 30-arc-second baseline.")
    return HighresFetchRefusal(message, remedy=remedy, folders=(path.parent,))


def _lock_refusal(path: Path,
                  busy: fetch_guard.FetchLockBusy) -> HighresFetchRefusal:
    """Another preparation holds ``path`` and this one stops waiting."""
    who = f"another preparation ({busy.holder or 'unidentified'})"
    if not busy.budget_s:
        state = (f"and {fetch_guard.LOCK_TIMEOUT_ENV} = 0 tells this one "
                 "not to wait for it")
    else:
        # The idle time is at least the budget; whole seconds, floored,
        # so the line never claims more than was measured.
        idle = int(busy.idle_s if busy.idle_s is not None
                   else busy.budget_s)
        waited = int(busy.waited_s if busy.waited_s is not None else idle)
        state = (f"and its download has not grown for {idle} s, with "
                 f"{sum(size for _, size in _partial_progress(path)[0])} "
                 "bytes staged; this one waited "
                 f"{waited} s for it")
    return HighresFetchRefusal(
        f"[static.highres] {who} is downloading {path.name} into "
        f"{path.parent}, {state}",
        remedy=("remedy: check that preparation and its connection; once "
                "it has finished or been stopped, prepare again (a "
                "finished file is read from the cache and a stopped "
                "download resumes from the bytes it received), or raise "
                f"{fetch_guard.LOCK_TIMEOUT_ENV} when that host is known "
                "to pause for longer"),
        folders=(path.parent,))


def _one_writer(path: Path):
    """The cross-process writer lock over one cached file.

    The concrete breakage it prevents: preparations sharing one cache
    (four at once in a multi-area build) staged the same uncached tile
    into one ``.partial``, both wrote into it, and the second rename
    failed with ``FileNotFoundError``; a resume read the other process's
    bytes as its own. The loser waits for the holder and then finds the
    file in the cache. Every writer owns an exclusively claimed staging
    name; a resumable failure records its bytes for the next writer to
    copy into a new private file only after acquiring the same lock.

    The waiter watches the holder's ``.partial`` (and the published
    file) grow, and the lock's wait budget runs from the last growth, so
    it waits out a slow but live download and refuses only a stalled
    one.  With the flat budget it had, any link slower than about
    3.8 MB/s ended parallel preparations on a fresh cache while the
    2.28 GB default land-cover file was still arriving.
    """
    return fetch_guard.hold(
        _FETCH_LOCK_KIND, path,
        progress=lambda line: print(f"[static.highres] {line}",
                                    file=sys.stderr, flush=True),
        holder_progress=lambda: _partial_progress(path))


def fetch_file(url: str, path: Path, *, urlopen=None,
               expected_bytes: int | None = None,
               expected_sha256: str | None = None,
               expected_md5: str | None = None) -> FetchedFile:
    """Fetch ``url`` to ``path`` (cached, resumable, hashed at fetch).

    A transient network fault (a reset connection, a timeout, a body cut
    short, HTTP 408/429/5xx) is asked again up to :data:`FETCH_ATTEMPTS`
    times with the tree's shared backoff
    (:func:`woof.fetch_endpoints.retry_delay`), each attempt resuming
    the staged bytes.  A network failure that outlasts that, or one no
    retry can change, is :class:`HighresFetchRefusal`.  ``SourceAbsent``
    and failures on this computer propagate unchanged.
    """
    path = Path(path)
    def admitted(cached):
        return cached is not None and (
            expected_bytes is None or cached.bytes == expected_bytes) and (
            expected_sha256 is None or cached.sha256 == expected_sha256) and (
            expected_md5 is None or _md5_file(path) == expected_md5)

    cached = _cache_hit(path, url)
    if admitted(cached):
        return cached

    opener = _default_urlopen if urlopen is None else urlopen
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = _one_writer(path).acquire()
    except fetch_guard.FetchLockBusy as error:
        raise _lock_refusal(path, error) from error
    partial = None
    keep_partial = False
    try:
        cached = _cache_hit(path, url)
        if admitted(cached):
            # Another preparation fetched it while this one waited.
            return cached
        partial = _new_partial(path)
        _resume_partial(path, partial, url)
        waited_s = 0.0
        for attempt in range(1, FETCH_ATTEMPTS + 1):
            try:
                total_bytes = _transfer(opener, url, partial)
                break
            except Exception as error:  # noqa: BLE001 - classified below
                delay = fetch_endpoints.retry_delay(
                    error, attempt, wait_limit_s=FETCH_RETRY_WAIT_LIMIT_S)
                if delay is None and not isinstance(error, _NETWORK_FAULTS):
                    raise
                if delay is None or attempt == FETCH_ATTEMPTS:
                    if _staged_bytes(partial):
                        _remember_partial(path, partial, url)
                        keep_partial = True
                    raise _fetch_refusal(path, url, partial, error,
                                         attempts=attempt,
                                         waited_s=waited_s) from error
                kept = _staged_bytes(partial)
                print(f"[static.highres] {path.name}: "
                      f"{_network_reason(error)}; asking again in "
                      f"{delay:g} s (attempt {attempt + 1} of "
                      f"{FETCH_ATTEMPTS})"
                      + (f", resuming after {kept} bytes" if kept else ""),
                      file=sys.stderr, flush=True)
                _sleep(delay)
                waited_s += delay
        record = _publish_partial(
            partial, path, url=url,
            expected_bytes=(total_bytes if expected_bytes is None
                            else expected_bytes),
            expected_sha256=expected_sha256, expected_md5=expected_md5)
        _resume_sidecar(path).unlink(missing_ok=True)
        return record
    finally:
        if partial is not None and not keep_partial:
            partial.unlink(missing_ok=True)
        writer.release()


# ---------------------------------------------------------------------------
# Whole-tile sets (terrain)
# ---------------------------------------------------------------------------

#: Tiles one terrain fetch downloads at once.  A public tile host serves
#: one stream at about 8-10 MB/s to a rented box, and Copernicus GLO-30
#: over a 1 km CONUS footprint is about 1,000 published tiles (40 GB): one
#: stream at a time took over an hour, and the 1 km preparation kit grew a
#: parallel prefill script to get around it.  Each tile has its own cache
#: entry, lock and staging file (:mod:`woof.fetch_guard`), so concurrent
#: tiles never share a partial, and the result keeps the enumeration
#: order, so receipts and the mosaic read tiles in the same order as one
#: stream would.
#: Raised from 16 to 32 on 2026-10-21: one 1 km corridor over the
#: contiguous US (~1,000 GLO-30 tiles) still left every core and the card
#: waiting on the transfer (a rented box).
TILE_FETCH_WORKERS = 32


def _fetch_tile_set(tiles, url_for, path_for, *, urlopen=None
                    ) -> tuple[tuple[FetchedFile, ...], tuple[str, ...]]:
    """Fetch ``tiles`` (``url_for(tile)`` into ``path_for(tile)``),
    :data:`TILE_FETCH_WORKERS` at a time; return ``(fetched, absent)`` in
    tile order.  The first failure (other than an unpublished tile) is
    raised after the tiles already being fetched finish; queued tiles are
    not started."""
    order = list(tiles)

    def one(tile):
        try:
            return fetch_file(url_for(tile), path_for(tile), urlopen=urlopen)
        except SourceAbsent:
            return None

    workers = max(1, min(int(TILE_FETCH_WORKERS), len(order)))
    if workers == 1:
        results = [one(tile) for tile in order]
    else:
        with ThreadPoolExecutor(workers, thread_name_prefix="tile-fetch") as pool:
            futures = [pool.submit(one, tile) for tile in order]
            try:
                results = [future.result() for future in futures]
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    fetched = tuple(item for item in results if item is not None)
    absent = tuple(tile for tile, item in zip(order, results) if item is None)
    return fetched, absent


# ---------------------------------------------------------------------------
# USGS 3DEP terrain tiles
# ---------------------------------------------------------------------------

def three_dep_tile_ids(bbox: FootprintBBox) -> tuple[str, ...]:
    """1x1-degree staged-tile ids covering ``bbox`` (n..w.. naming).

    Tile ``n40w084`` covers latitudes [39, 40] x longitudes [-84, -83].
    Only the northern/western quadrant is enumerable because that is where
    the jointly declared sources exist; anything else must already have
    been refused by the coverage gate.
    """
    if bbox.lat_min < 0.0 or bbox.lon_max > 0.0:
        raise CoverageError(
            "3DEP staged 1/3 arc-second tiles are enumerated for the "
            f"northern/western quadrant only; footprint {bbox.as_dict()} "
            "leaves it")
    tiles = []
    for north in range(int(math.floor(bbox.lat_min)) + 1,
                       int(math.ceil(bbox.lat_max)) + 1):
        for west_edge in range(int(math.floor(bbox.lon_min)),
                               int(math.ceil(bbox.lon_max))):
            tiles.append(f"n{north:02d}w{-west_edge:03d}")
    return tuple(tiles)


def fetch_three_dep_tiles(bbox: FootprintBBox, cache_root: Path, *,
                          urlopen=None
                          ) -> tuple[tuple[FetchedFile, ...], tuple[str, ...]]:
    """Fetch every published whole 3DEP tile covering ``bbox``.

    Returns ``(fetched, absent)``, the same contract as
    :func:`fetch_copernicus_dem_tiles`.  3DEP stages no tile over open
    sea or wholly outside the United States, so an absent tile is a
    square this source does not cover: its cells take the 30-arc-second
    baseline terrain (:mod:`woof.static.highres`), and the ids are
    named in the receipt.  ``fetched`` is empty when no tile is staged.
    """
    cache = Path(cache_root) / "usgs3dep_13as"
    return _fetch_tile_set(
        three_dep_tile_ids(bbox), lambda tile: THREE_DEP_TILE_URL.format(tile=tile),
        lambda tile: cache / f"USGS_13_{tile}.tif", urlopen=urlopen)


# ---------------------------------------------------------------------------
# Copernicus DEM GLO-30 terrain tiles (near-global)
# ---------------------------------------------------------------------------

def copernicus_dem_tile_ids(bbox: FootprintBBox) -> tuple[str, ...]:
    """1x1-degree GLO-30 tile ids covering ``bbox`` (south-west corner names).

    ``N39_00_W105_00`` spans 39..40 N and 105..104 W; ``S34_00_E018_00``
    spans 34..33 S and 18..19 E.  Unlike the US enumerator this one is
    valid in all four quadrants, which is the whole point of the source.

    A continued longitude range from :func:`domain_footprint` (a dateline
    domain reporting, say, 179.25 .. 180.75) enumerates correctly with no
    special case: the integer-degree loop names each tile through
    ``((lon_sw + 180) % 360) - 180``, so degree 180 comes out as
    ``W180_00`` and the two sides of the line are simply adjacent tiles.
    """
    lat_lo = max(-90, int(math.floor(bbox.lat_min)))
    lat_hi = min(90, int(math.ceil(bbox.lat_max)))
    lon_lo = int(math.floor(bbox.lon_min))
    lon_hi = int(math.ceil(bbox.lon_max))
    tiles: list[str] = []
    for lat_sw in range(lat_lo, lat_hi):
        ns = "N" if lat_sw >= 0 else "S"
        for lon_sw in range(lon_lo, lon_hi):
            wrapped = ((lon_sw + 180) % 360) - 180
            ew = "E" if wrapped >= 0 else "W"
            tiles.append(f"{ns}{abs(lat_sw):02d}_00_"
                         f"{ew}{abs(wrapped):03d}_00")
    if not tiles:
        raise CoverageError(
            f"footprint {bbox.as_dict()} enumerates no Copernicus DEM tile")
    return tuple(tiles)


def srtm_tile_ids(bbox: FootprintBBox) -> tuple[str, ...]:
    """SRTMGL1 tile ids covering ``bbox`` (``N39W105`` style, SW corner)."""
    return tuple(tile.replace("_00_", "").removesuffix("_00")
                 for tile in copernicus_dem_tile_ids(bbox))


def one_degree_tile_bbox(tile: str) -> FootprintBBox:
    """Geographic box of one 1x1-degree tile id, either naming style.

    Accepts ``N39_00_W105_00`` (Copernicus) and ``N39W105`` (SRTM); both
    name the south-west corner.
    """
    match = re.fullmatch(r"([NS])(\d{2})(?:_00)?_?([EW])(\d{3})(?:_00)?",
                         tile)
    if match is None:
        raise ValueError(
            f"tile id {tile!r} is not a 1x1-degree south-west-corner id "
            "(expected N39_00_W105_00 or N39W105)")
    lat = int(match.group(2)) * (1 if match.group(1) == "N" else -1)
    lon = int(match.group(4)) * (1 if match.group(3) == "E" else -1)
    return FootprintBBox(lat_min=float(lat), lat_max=float(lat + 1),
                         lon_min=float(lon), lon_max=float(lon + 1))


#: Backward-compatible alias.
copernicus_tile_bbox = one_degree_tile_bbox


def fetch_srtm_gl1_tiles(bbox: FootprintBBox, cache_root: Path, *,
                         urlopen=None
                         ) -> tuple[tuple[FetchedFile, ...], tuple[str, ...]]:
    """Fetch every published SRTMGL1 tile covering ``bbox``.

    Same ``(fetched, absent)`` contract as
    :func:`fetch_copernicus_dem_tiles`: SRTM publishes no all-water tiles
    either, so absence is handed back rather than being read as terrain.
    """
    cache = Path(cache_root) / "srtm_gl1"
    return _fetch_tile_set(
        srtm_tile_ids(bbox), lambda tile: SRTM_GL1_TILE_URL.format(tile=tile),
        lambda tile: cache / f"{tile}.tif", urlopen=urlopen)


def fetch_copernicus_dem_tiles(bbox: FootprintBBox, cache_root: Path, *,
                               urlopen=None
                               ) -> tuple[tuple[FetchedFile, ...],
                                          tuple[str, ...]]:
    """Fetch every published GLO-30 tile covering ``bbox``.

    Returns ``(fetched, absent)``.  The product does not publish
    all-water tiles (nor a few withheld land tiles), so an absent tile
    is a square this source does not cover: the derived window keeps it
    as no data and its cells take the 30-arc-second baseline terrain
    (:mod:`woof.static.highres`), whether the baseline calls them sea
    or land.  ``fetched`` is empty when every tile is absent.
    """
    cache = Path(cache_root) / "copernicus_dem_glo30"
    return _fetch_tile_set(
        copernicus_dem_tile_ids(bbox),
        lambda tile: COPERNICUS_DEM_TILE_URL.format(tile=tile),
        lambda tile: cache / f"Copernicus_DSM_COG_10_{tile}_DEM.tif",
        urlopen=urlopen)


#: Second way out of :func:`_require_cut_frame_window`, valid for both
#: footprints it refuses: the 30-arc-second baseline is written in no
#: window at all and so has no cut frame to run past.
_BASELINE_WAY_OUT = ("or leave [static.highres] disabled and run on the "
                     "30-arc-second baseline, which has no such limit")


def _require_cut_frame_window(bbox: FootprintBBox) -> None:
    """One statement, every door: the mosaic frame is still cut.

    :func:`domain_footprint` continues a dateline domain's longitude range
    past 180 and the tile enumerators and per-source coverage envelopes
    read it in that frame.  The window writers below (Rust bridge and the
    rasterio parity body alike) still emit the derived GeoTIFF in the cut
    -180..180 frame, so a continued window is the one thing they cannot
    express.

    Two different footprints arrive here and they are told two different
    things, because the fact and the way out differ:

    - A domain ON the line has a narrow continued range (179.28 .. 180.72
      and the like).  The tiles either side of the line would be pasted a
      planet apart, and moving the domain off 180 degrees builds.
    - A domain whose corners span 180 degrees of longitude or more has no
      continued range at all: :func:`_continued_longitude_range` reports
      it as the full -180..180 band, which the footprint margin then
      pushes past both ends of the cut frame.  Such a footprint occupies
      every longitude, and a domain wrapped around its projection pole is
      what produces it (the polar-stereographic domain that encloses the
      pole, the conic domain whose corners fan more than half a turn
      about the cone apex).  There is no line here and nothing to move
      off it, so the way out is a smaller domain, or one further from the
      projection pole, until its corners span less than 180 degrees.

    The two are told apart by width: the band is 360 degrees wide or more
    once padded, and a continued range is by construction narrower than
    180.

    This fires at PLAN REVIEW:
    :func:`woof.static.highres_production._apply` calls it on the
    footprint it has just computed, before the plan is resolved and
    before one byte is requested, so neither domain enumerates or
    downloads a tile it cannot mosaic.  Both window writers call it
    again as a backstop, so no door can disagree with another about one
    footprint.
    """
    if not (bbox.lon_max > 180.0 or bbox.lon_min < -180.0):
        return
    if bbox.lon_max - bbox.lon_min >= 360.0:
        raise HighresRefusal(
            "dateline-window-unbuilt",
            f"the domain+halo footprint {bbox.as_dict()} occupies every "
            "longitude: its corners span 180 degrees or more, which has "
            "no continued range, so it reduces to the whole band.  A "
            "domain wrapped around its projection pole is what produces "
            "that, and the mosaic window is one rectangle in the cut "
            "-180..180 frame, so the only window covering this footprint "
            "is the whole planet at source resolution, thousands of "
            "tiles, rather than the domain.  Shrink the domain, or move "
            "it away from the projection pole, until its corners span "
            f"less than 180 degrees of longitude, {_BASELINE_WAY_OUT}")
    raise HighresRefusal(
        "dateline-window-unbuilt",
        f"the domain+halo footprint {bbox.as_dict()} is continued past "
        "180 degrees, and the mosaic window is still written in the "
        "cut -180..180 frame, so the tiles either side of the line "
        "would be pasted a whole planet apart and the terrain would "
        "arrive shifted.  Tile enumeration and source coverage already "
        "handle the crossing; the window writer does not yet.  Move "
        f"the domain off 180 degrees, {_BASELINE_WAY_OUT}")


def derive_global_terrain_window(tiles, bbox: FootprintBBox,
                                 cache_root: Path, *,
                                 sea_level_fill: float | None = 0.0,
                                 source_nodata: float | None = None,
                                 resolution_deg: float
                                 = COPERNICUS_DEM_LAT_STEP_DEG
                                 ) -> tuple[FetchedFile, dict[str, object]]:
    """Mosaic GLO-30 tiles onto one uniform grid over the whole footprint.

    Two things differ from :func:`derive_terrain_window` and both are
    forced by the source:

    - GLO-30's longitude sampling is latitude-banded (3600 columns per
      degree below 50, 2400 in 50-60, 1800 in 60-70), so tiles from
      different bands do not share a resolution.  The output resolution is
      declared, not inherited, and the resampling is nearest so no
      elevation value is invented -- coarser bands are replicated, and the
      subsequent area-average to the model grid is what actually reduces
      them.
    - Unpublished tiles leave holes (and SRTM additionally carries an
      in-band void sentinel).  With ``sea_level_fill = None`` -- what the
      production overlay passes -- they stay no data (NaN), so the model
      cells under them take the 30-arc-second baseline terrain and are
      counted as outside the source's coverage.  A number fills them with
      that height instead (0 m on the EGM2008 geoid, the source's own
      vertical datum), and the filled pixel count is returned.

    The derivation itself -- decode, mosaic, void fill, re-emit -- runs
    in the Rust static-fields library by default; the rasterio body is
    the parity reference and the reported fallback.
    """
    tiles = list(tiles)
    if not tiles:
        raise ValueError("terrain window derivation requires >= 1 tile")
    _require_cut_frame_window(bbox)
    keep_holes = sea_level_fill is None
    identity = hashlib.sha256(json.dumps(
        {"tiles": sorted(item.sha256 for item in tiles),
         "bbox": bbox.as_dict(), "res": resolution_deg,
         "fill": sea_level_fill, "src_nodata": source_nodata,
         # v2: cut on the fixed lattice (_terrain_lattice), not at the
         # footprint's own edge, so a v1 window is never reused.
         "kind": "global-terrain-window-v2"},
        sort_keys=True).encode("utf-8")).hexdigest()[:20]
    out_dir = Path(cache_root) / "derived"
    out_path = out_dir / f"terrain_global_{identity}.tif"
    sidecar_path = out_dir / f"terrain_global_{identity}.audit.json"
    derivation_url = ("derived:mosaic+fill+clip of "
                      + ",".join(sorted(item.path.name for item in tiles)))
    with _derivation_writer(out_path) as partial:
        cached = _read_sidecar(out_path)
        if cached is not None and sidecar_path.is_file():
            return (FetchedFile(
                path=out_path, url=derivation_url, sha256=cached["sha256"],
                bytes=int(cached["bytes"]),
                fetched_utc=str(cached.get("fetched_utc", "")), cache_hit=True),
                json.loads(sidecar_path.read_text(encoding="utf-8")))

        out_dir.mkdir(parents=True, exist_ok=True)
        margin_deg = 0.01
        bounds = (bbox.lon_min - margin_deg, bbox.lat_min - margin_deg,
                  bbox.lon_max + margin_deg, bbox.lat_max + margin_deg)
        bridge = _static_rust("derive_global_terrain_window")
        if bridge is not None:
            request = {
                "kind": "global-terrain-window",
                "tiles": [str(item.path) for item in tiles],
                "bounds": list(bounds),
                "resolution_deg": float(resolution_deg),
                "source_nodata": (None if source_nodata is None
                                  else float(source_nodata)),
                "out_path": str(partial),
            }
            if keep_holes:
                request["keep_holes"] = True
            else:
                request["sea_level_fill"] = float(sea_level_fill)
            audit = bridge.highres_derive_window(request)
            artifact = _publish_partial(partial, out_path, url=derivation_url)
            holes = int(audit.get("hole_pixels",
                                  audit.get("sea_level_filled_pixels", 0)))
            audit = {
                "output_resolution_deg": float(resolution_deg),
                "output_shape": [int(v) for v in audit["output_shape"]],
                "sea_level_filled_pixels": 0 if keep_holes else holes,
                "no_data_pixels_outside_coverage": holes if keep_holes else 0,
                "total_pixels": int(audit["total_pixels"]),
                "sea_level_fill_m": (None if keep_holes
                                     else float(sea_level_fill)),
                "source_nodata": (None if source_nodata is None
                                  else float(source_nodata)),
                "resampling": str(audit["resampling"]),
                # The source's own vertical datum: provenance the crate has
                # no business asserting, because it is a fact about which
                # PRODUCT was fetched, not about the bytes.
                "vertical_datum": COPERNICUS_DEM_VERTICAL_DATUM,
            }
            fetch_guard.atomic_write_text(sidecar_path,
                json.dumps(audit, indent=2, sort_keys=True) + "\n")
            return artifact, audit

        try:
            import rasterio
            from rasterio.enums import Resampling
            from rasterio.merge import merge as rasterio_merge
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                geog_unavailable_detail()
            ) from exc
        datasets = [rasterio.open(item.path) for item in tiles]
        try:
            crs = datasets[0].crs
            bounds = _lattice_bounds(bounds, *_terrain_lattice(
                None, resolution_deg))
            mosaic, transform = rasterio_merge(
                datasets, bounds=bounds,
                res=(resolution_deg, resolution_deg),
                resampling=Resampling.nearest,
                nodata=np.nan, dtype="float32")
        finally:
            for dataset in datasets:
                dataset.close()
        values = np.asarray(mosaic[0], dtype=np.float32)
        holes = ~np.isfinite(values)
        if source_nodata is not None:
            # SRTM carries an in-band void sentinel; Copernicus carries none.
            holes |= values == np.float32(source_nodata)
        filled = int(np.count_nonzero(holes))
        values[holes] = (np.float32(np.nan) if keep_holes
                         else np.float32(sea_level_fill))
        with rasterio.open(
                partial, "w", driver="GTiff", height=values.shape[0],
                width=values.shape[1], count=1, dtype="float32", crs=crs,
                transform=transform, nodata=None, compress="deflate",
                predictor=3, tiled=True) as target:
            target.write(values, 1)
        artifact = _publish_partial(partial, out_path, url=derivation_url)
        audit = {
            "output_resolution_deg": float(resolution_deg),
            "output_shape": [int(values.shape[0]), int(values.shape[1])],
            "sea_level_filled_pixels": 0 if keep_holes else filled,
            "no_data_pixels_outside_coverage": filled if keep_holes else 0,
            "total_pixels": int(values.size),
            "sea_level_fill_m": (None if keep_holes
                                 else float(sea_level_fill)),
            "source_nodata": (None if source_nodata is None
                              else float(source_nodata)),
            "resampling": "nearest (latitude-banded source resolutions)",
            "vertical_datum": COPERNICUS_DEM_VERTICAL_DATUM,
        }
        fetch_guard.atomic_write_text(sidecar_path,
            json.dumps(audit, indent=2, sort_keys=True) + "\n")
        return artifact, audit


# ---------------------------------------------------------------------------
# Land cover (table-driven: LANDCOVER_SOURCES)
# ---------------------------------------------------------------------------

def nlcd_year_for(case_date: date) -> tuple[int, int]:
    """(published year nearest the case date, anachronism in years)."""
    return LANDCOVER_SOURCES["annual-nlcd"].year_for(case_date)


def fetch_landcover(source: LandcoverSource, year: int, cache_root: Path,
                    *, urlopen=None
                    ) -> tuple[tuple[FetchedFile, ...], FetchedFile]:
    """Fetch one land-cover source's raster for ``year``.

    Returns ``(downloaded, raster)``: the artifacts exactly as published
    (hashed at fetch time) and the GeoTIFF the window step decodes.  The
    fetch kind is the row's, so the dispatch below is over publication
    shapes, never over collections.
    """
    if source.fetch == "yearly-zip-bundle":
        bundle, raster = _fetch_yearly_zip_bundle(source, year, cache_root,
                                                  urlopen=urlopen)
        return (bundle,), raster
    if source.fetch == "whole-geotiff":
        raster = _fetch_whole_geotiff(source, cache_root, urlopen=urlopen)
        return (raster,), raster
    raise ValueError(  # pragma: no cover - __post_init__ refuses it first
        f"land-cover fetch kind {source.fetch!r} has no fetcher")


def _human_bytes(count: int) -> str:
    return (f"{count / 1e9:.2f} GB" if count >= 1e9
            else f"{count / 1e6:.0f} MB")


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch_whole_geotiff(source: LandcoverSource, cache_root: Path, *,
                         urlopen=None) -> FetchedFile:
    """Fetch a whole-GeoTIFF source once per cache root and hold it to
    its pins.

    The download is resumable (:func:`fetch_file` stages a ``.partial``
    and continues it with an HTTP range) and happens once per
    ``cache_root``.  A fresh payload must match the pinned size, MD5 and
    SHA-256; a cached one is held to the size and SHA-256 its sidecar
    recorded when it was fetched.
    """
    path = Path(cache_root) / source.cache_dir / source.file_name
    if _read_sidecar(path) is None:
        size = ("" if source.pinned_bytes is None
                else f" ({_human_bytes(source.pinned_bytes)})")
        print(f"[static.highres] fetching {source.label or source.source_id}"
              f"{size} once into {path.parent}; later preparations with "
              "this cache_root read it from there")
    try:
        fetched = fetch_file(
            source.url, path, urlopen=urlopen,
            expected_bytes=source.pinned_bytes,
            expected_sha256=source.pinned_sha256,
            expected_md5=source.pinned_md5)
    except SourceAbsent as error:
        raise CoverageError(
            f"land-cover source {source.source_id!r} is not published at "
            f"{source.url}") from error
    return fetched


def fetch_annual_nlcd(year: int, cache_root: Path, *, urlopen=None
                      ) -> tuple[FetchedFile, FetchedFile]:
    """Fetch one whole Annual NLCD year bundle; return (zip, extracted tif)."""
    return _fetch_yearly_zip_bundle(LANDCOVER_SOURCES["annual-nlcd"], year,
                                    cache_root, urlopen=urlopen)


def _fetch_yearly_zip_bundle(source: LandcoverSource, year: int,
                             cache_root: Path, *, urlopen=None
                             ) -> tuple[FetchedFile, FetchedFile]:
    """Fetch one whole year bundle; return (zip, extracted tif).

    The published artifact is a zip around one GeoTIFF; both the bundle
    exactly as fetched and the extracted raster are hashed and
    sidecar-recorded, so the receipt can bind the raster actually decoded
    back to the bytes actually downloaded.
    """
    name = source.label or source.source_id
    if not (source.first_year <= int(year) <= source.last_year):
        raise CoverageError(
            f"{name} publishes {source.first_year}..{source.last_year}; "
            f"there is no year {year}")
    cache = Path(cache_root) / source.cache_dir
    url = source.url.format(year=int(year))
    try:
        bundle = fetch_file(url, cache / Path(url).name, urlopen=urlopen)
    except SourceAbsent as error:
        raise CoverageError(
            f"{name} year {year} is not published at {url}") from error

    members: list[str] = []
    with zipfile.ZipFile(bundle.path) as archive:
        members = [member for member in archive.namelist()
                   if member.lower().endswith(".tif")]
        if len(members) != 1:
            raise ValueError(
                f"{name} bundle {bundle.path} contains {len(members)} .tif "
                f"members ({members}); expected one")
        raster_path = cache / Path(members[0]).name
        with _derivation_writer(raster_path) as partial:
            cached = _read_sidecar(raster_path)
            if cached is None:
                with archive.open(members[0]) as stream, \
                        partial.open("wb") as target:
                    while True:
                        block = stream.read(_CHUNK)
                        if not block:
                            break
                        target.write(block)
                    target.flush()
                    os.fsync(target.fileno())
                raster = _publish_partial(
                    partial, raster_path, url=f"{url}!{members[0]}",
                    expected_bytes=archive.getinfo(members[0]).file_size)
            else:
                raster = FetchedFile(
                    path=raster_path, url=str(cached.get("url", url)),
                    sha256=cached["sha256"], bytes=int(cached["bytes"]),
                    fetched_utc=str(cached.get("fetched_utc", "")),
                    cache_hit=True)
    return bundle, raster


# ---------------------------------------------------------------------------
# SoilGrids v2 WCS windows
# ---------------------------------------------------------------------------

def _soilgrids_window_m(bbox: FootprintBBox) -> tuple[float, float,
                                                      float, float]:
    """Snap the footprint to a whole-km window on SoilGrids' IGH plane.

    The projection of the perimeter mesh onto the Interrupted Goode
    Homolosine plane runs in the Rust static-fields library by default;
    the pyproj body below is the parity reference and the reported
    fallback.  Snapping and the margin are arithmetic on the four
    extrema and stay here.
    """
    lons = np.linspace(bbox.lon_min, bbox.lon_max, 25)
    lats = np.linspace(bbox.lat_min, bbox.lat_max, 25)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    bridge = _static_rust("soilgrids window transform")
    if bridge is not None:
        x, y = bridge.highres_transform_points(
            SOILGRIDS_CRS, grid_lon, grid_lat)
        return _snap_window_m(x, y)

    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise RuntimeError(
            geog_unavailable_detail()
        ) from exc
    transformer = Transformer.from_crs("EPSG:4326", SOILGRIDS_CRS,
                                       always_xy=True)
    x, y = transformer.transform(grid_lon, grid_lat)
    return _snap_window_m(x, y)


def _snap_window_m(x, y) -> tuple[float, float, float, float]:
    """Margin then snap to whole kilometres, so re-preparing a domain
    hits the WCS cache instead of minting a near-duplicate window."""
    snap = _SOILGRIDS_SNAP_M
    x0 = math.floor((float(np.min(x)) - _SOILGRIDS_MARGIN_M) / snap) * snap
    x1 = math.ceil((float(np.max(x)) + _SOILGRIDS_MARGIN_M) / snap) * snap
    y0 = math.floor((float(np.min(y)) - _SOILGRIDS_MARGIN_M) / snap) * snap
    y1 = math.ceil((float(np.max(y)) + _SOILGRIDS_MARGIN_M) / snap) * snap
    return x0, x1, y0, y1


def fetch_soilgrids(bbox: FootprintBBox, cache_root: Path, *, urlopen=None
                    ) -> dict[tuple[str, str], FetchedFile]:
    """Fetch SoilGrids Q0.5 windows for every component x depth."""
    x0, x1, y0, y1 = _soilgrids_window_m(bbox)
    key = f"x{x0:.0f}_{x1:.0f}_y{y0:.0f}_{y1:.0f}"
    cache = Path(cache_root) / "soilgrids_v2"
    out: dict[tuple[str, str], FetchedFile] = {}
    for component in SOILGRIDS_COMPONENTS:
        for depth in SOILGRIDS_DEPTHS:
            url = SOILGRIDS_WCS_URL.format(
                component=component, depth=depth, x0=x0, x1=x1, y0=y0, y1=y1)
            name = f"{component}_{depth}_Q0.5_{key}.tif"
            try:
                out[(component, depth)] = fetch_file(
                    url, cache / name, urlopen=urlopen)
            except SourceAbsent as error:
                raise CoverageError(
                    f"SoilGrids WCS refused {component} {depth} for window "
                    f"{key}: {error}") from error
    return out


# ---------------------------------------------------------------------------
# Derived per-footprint windows (local derivations of fetched payloads)
# ---------------------------------------------------------------------------

#: Metres per degree of latitude on the mean sphere, for turning the
#: window margin into degrees on a geographic raster.
_METRES_PER_DEGREE = 111_320.0


def margin_degrees(lat_min: float, lat_max: float, margin_m: float
                   ) -> tuple[float, float]:
    """(longitude, latitude) margin in degrees for a metre margin.

    The window margin is stated in metres.  On a projected raster that is
    its own unit; on a geographic (EPSG:4326) raster the same number read
    as degrees turned a 2 km margin into 2000 degrees, so the window was
    the whole global raster.  The longitude margin is widened by the
    footprint's most poleward latitude (floored at cos 87 degrees), so it
    is at least ``margin_m`` everywhere in the footprint.  The Rust window
    step computes the same two numbers.
    """
    extreme = max(abs(float(lat_min)), abs(float(lat_max)))
    shrink = max(math.cos(math.radians(min(extreme, 90.0))),
                 math.cos(math.radians(87.0)))
    return (margin_m / (_METRES_PER_DEGREE * shrink),
            margin_m / _METRES_PER_DEGREE)


def _densified_bounds(bbox: FootprintBBox, dst_crs, margin_m: float
                      ) -> tuple[float, float, float, float]:
    """Footprint bounds in ``dst_crs``, sampled along the perimeter.

    ``margin_m`` is metres; a geographic ``dst_crs`` takes it in degrees
    through :func:`margin_degrees`.
    """
    try:
        from pyproj import CRS, Transformer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            geog_unavailable_detail()
        ) from exc
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    lons = np.linspace(bbox.lon_min, bbox.lon_max, 41)
    lats = np.linspace(bbox.lat_min, bbox.lat_max, 41)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    x, y = transformer.transform(grid_lon, grid_lat)
    margin_x = margin_y = margin_m
    if CRS.from_user_input(dst_crs).is_geographic:
        margin_x, margin_y = margin_degrees(bbox.lat_min, bbox.lat_max,
                                            margin_m)
    return (float(np.min(x)) - margin_x, float(np.min(y)) - margin_y,
            float(np.max(x)) + margin_x, float(np.max(y)) + margin_y)


def _terrain_lattice(transform, resolution_deg):
    """The fixed pixel lattice a terrain crop is cut on, as the native
    library chooses it: ``((res_x, res_y), (origin_x, origin_y))``.

    Every footprint cut from a source must sample the same source pixel
    for the same ground, or a moving nest's statics corridor disagrees
    with the nest's own statics and the move is refused.  An inherited
    resolution keeps the first tile's own lattice; a declared one puts
    pixel centres on whole multiples of the resolution, where the
    point-sampled DEMs put their samples.
    """
    if resolution_deg is not None:
        r = float(resolution_deg)
        return (r, r), (-0.5 * r, 0.5 * r)
    return ((float(transform.a), -float(transform.e)),
            (float(transform.c), float(transform.f)))


def _lattice_bounds(bounds, resolution, origin):
    """``bounds`` grown outward to whole pixels of the lattice."""
    west, south, east, north = bounds
    (rx, ry), (ox, oy) = resolution, origin
    return (ox + math.floor((west - ox) / rx) * rx,
            oy - math.ceil((oy - south) / ry) * ry,
            ox + math.ceil((east - ox) / rx) * rx,
            oy - math.floor((oy - north) / ry) * ry)


def derive_terrain_window(tiles, bbox: FootprintBBox,
                          cache_root: Path) -> FetchedFile:
    """Mosaic the fetched whole tiles and clip to the footprint.

    The derivation is cached by the SHA-256 of its inputs (tile digests +
    footprint), so re-preparing a domain reuses it byte-identically.

    Decode, mosaic and re-emit run in the Rust static-fields library by
    default; the rasterio body is the parity reference and the reported
    fallback.
    """
    tiles = list(tiles)
    if not tiles:
        raise ValueError("terrain window derivation requires >= 1 tile")
    _require_cut_frame_window(bbox)
    identity = hashlib.sha256(json.dumps(
        {"tiles": sorted(item.sha256 for item in tiles),
         # v2: cut on the source's own pixel lattice, not at the
         # footprint's own edge, so a v1 window is never reused.
         "bbox": bbox.as_dict(), "kind": "terrain-window-v2"},
        sort_keys=True).encode("utf-8")).hexdigest()[:20]
    out_dir = Path(cache_root) / "derived"
    out_path = out_dir / f"terrain_{identity}.tif"
    with _derivation_writer(out_path) as partial:
        cached = _read_sidecar(out_path)
        derivation_url = ("derived:mosaic+clip of "
                          + ",".join(sorted(item.path.name for item in tiles)))
        if cached is not None:
            return FetchedFile(
                path=out_path, url=derivation_url, sha256=cached["sha256"],
                bytes=int(cached["bytes"]),
                fetched_utc=str(cached.get("fetched_utc", "")), cache_hit=True)

        out_dir.mkdir(parents=True, exist_ok=True)
        margin_deg = 0.01
        bounds = (bbox.lon_min - margin_deg, bbox.lat_min - margin_deg,
                  bbox.lon_max + margin_deg, bbox.lat_max + margin_deg)
        bridge = _static_rust("derive_terrain_window")
        if bridge is not None:
            bridge.highres_derive_window({
                "kind": "terrain-window",
                "tiles": [str(item.path) for item in tiles],
                "bounds": list(bounds),
                "out_path": str(partial),
            })
            artifact = _publish_partial(partial, out_path, url=derivation_url)
            return artifact

        try:
            import rasterio
            from rasterio.merge import merge as rasterio_merge
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                geog_unavailable_detail()
            ) from exc
        datasets = [rasterio.open(item.path) for item in tiles]
        try:
            crs = datasets[0].crs
            bounds = _lattice_bounds(bounds, *_terrain_lattice(
                datasets[0].transform, None))
            mosaic, transform = rasterio_merge(datasets, bounds=bounds)
            nodata = datasets[0].nodata
        finally:
            for dataset in datasets:
                dataset.close()
        with rasterio.open(
                partial, "w", driver="GTiff", height=mosaic.shape[1],
                width=mosaic.shape[2], count=1, dtype=mosaic.dtype, crs=crs,
                transform=transform, nodata=nodata, compress="deflate",
                predictor=2 if np.issubdtype(mosaic.dtype, np.integer) else 3,
                tiled=True) as target:
            target.write(mosaic[0], 1)
        artifact = _publish_partial(partial, out_path, url=derivation_url)
        return artifact


def _landcover_audit_path(window_path: Path) -> Path:
    return Path(window_path).with_name(
        Path(window_path).stem + ".audit.json")


def landcover_window_audit(window: FetchedFile) -> dict | None:
    """The window step's audit (window, clip, raw category pixel counts),
    or None for a window derived before the audit was recorded."""
    path = _landcover_audit_path(window.path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def derive_landcover_window(raster: FetchedFile, bbox: FootprintBBox,
                            cache_root: Path) -> FetchedFile:
    """Clip a whole land-cover raster to the footprint (cached).

    Decode, the footprint densification into the raster's own CRS, the
    window arithmetic and the re-emit run in the Rust static-fields
    library by default; the rasterio body is the parity reference and
    the reported fallback.  The 2 km margin is metres on a projected
    raster and the same distance in degrees on a geographic one
    (:func:`margin_degrees`).  A footprint that runs past the raster's
    extent is clipped to it: the model cells beyond receive no source
    pixel and take the 30-arc-second baseline land use
    (:mod:`woof.static.highres`).  A footprint wholly outside the
    raster is a :class:`CoverageError` (it crosses the seam as its own
    return code, so a decode fault stays a fault), which the production
    shell answers with the baseline land use on every cell.

    The step's audit, including the pixel count of every raw category in
    the window, is kept beside the window (:func:`landcover_window_audit`)
    and copied into the receipt.
    """
    identity = hashlib.sha256(json.dumps(
        {"source": raster.sha256, "bbox": bbox.as_dict(),
         "kind": "landcover-window-v1"},
        sort_keys=True).encode("utf-8")).hexdigest()[:20]
    out_dir = Path(cache_root) / "derived"
    out_path = out_dir / f"landcover_{identity}.tif"
    audit_path = _landcover_audit_path(out_path)
    derivation_url = f"derived:clip of {raster.path.name}"
    with _derivation_writer(out_path) as partial:
        cached = _read_sidecar(out_path)
        if cached is not None and audit_path.is_file():
            return FetchedFile(
                path=out_path, url=derivation_url, sha256=cached["sha256"],
                bytes=int(cached["bytes"]),
                fetched_utc=str(cached.get("fetched_utc", "")), cache_hit=True)

        out_dir.mkdir(parents=True, exist_ok=True)
        bridge = _static_rust("derive_landcover_window")
        if bridge is not None:
            from .rust_bridge import StaticCoverageRefusal
            try:
                audit = bridge.highres_derive_window({
                    "kind": "landcover-window",
                    "source": str(raster.path),
                    "bounds_lonlat": [bbox.lat_min, bbox.lat_max,
                                      bbox.lon_min, bbox.lon_max],
                    "margin_m": 2000.0,
                    "out_path": str(partial),
                })
            except StaticCoverageRefusal as refusal:
                partial.unlink(missing_ok=True)
                raise CoverageError(str(refusal)) from refusal
            artifact = _publish_partial(partial, out_path, url=derivation_url)
            fetch_guard.atomic_write_text(audit_path, json.dumps(audit, indent=2, sort_keys=True)
                                  + "\n")
            return artifact

        try:
            import rasterio
            from rasterio.windows import from_bounds
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                geog_unavailable_detail()
            ) from exc
        with rasterio.open(raster.path) as source:
            left, bottom, right, top = _densified_bounds(
                bbox, source.crs, margin_m=2000.0)
            window = from_bounds(left, bottom, right, top,
                                 transform=source.transform)
            # Explicit outward rounding (rasterio's round_offsets/round_lengths
            # signatures drifted across 1.x releases; the arithmetic is fixed).
            col_off = math.floor(window.col_off)
            row_off = math.floor(window.row_off)
            window = rasterio.windows.Window(
                col_off, row_off,
                math.ceil(window.width + (window.col_off - col_off)),
                math.ceil(window.height + (window.row_off - row_off)))
            full = rasterio.windows.Window(0, 0, source.width, source.height)
            clipped = window.intersection(full)
            if clipped.width <= 0 or clipped.height <= 0:
                raise CoverageError(
                    f"footprint {bbox.as_dict()} lies outside the land-cover "
                    f"raster extent of {raster.path.name}")
            values = source.read(1, window=clipped)
            transform = source.window_transform(clipped)
            with rasterio.open(
                    partial, "w", driver="GTiff", height=values.shape[0],
                    width=values.shape[1], count=1, dtype=values.dtype,
                    crs=source.crs, transform=transform, nodata=source.nodata,
                    compress="deflate", predictor=2, tiled=True) as target:
                target.write(values, 1)
            categories, counts = np.unique(values, return_counts=True)
            audit = {
                "output_shape": [int(values.shape[0]), int(values.shape[1])],
                "window": [int(clipped.col_off), int(clipped.row_off),
                           int(clipped.width), int(clipped.height)],
                "requested_window": [int(window.col_off), int(window.row_off),
                                     int(window.width), int(window.height)],
                "clipped_to_raster": (int(clipped.width) != int(window.width)
                                      or int(clipped.height)
                                      != int(window.height)),
                "nodata": source.nodata,
                "category_pixels": {
                    str(int(category)): int(count)
                    for category, count in zip(categories, counts)
                    if np.isfinite(category)},
            }
        artifact = _publish_partial(partial, out_path, url=derivation_url)
        fetch_guard.atomic_write_text(audit_path, json.dumps(audit, indent=2, sort_keys=True) + "\n")
        return artifact


__all__ = [
    "ANNUAL_NLCD_FIRST_YEAR", "ANNUAL_NLCD_LAST_YEAR", "ANNUAL_NLCD_URL",
    "CGLC_MODIS_LCZ_ATTRIBUTION", "CGLC_MODIS_LCZ_BYTES",
    "CGLC_MODIS_LCZ_LICENSE", "CGLC_MODIS_LCZ_MD5", "CGLC_MODIS_LCZ_SHA256",
    "CGLC_MODIS_LCZ_SOURCE_URL", "CGLC_MODIS_LCZ_URL", "CGLC_MODIS_LCZ_YEAR",
    "DEFAULT_LANDCOVER_SOURCE", "LANDCOVER_FETCH_KINDS", "LandcoverSource",
    "fetch_landcover", "landcover_source",
    "landcover_source_ids_by_wps_token", "landcover_window_audit",
    "margin_degrees",
    "COPERNICUS_DEM_ATTRIBUTION", "COPERNICUS_DEM_LAT_STEP_DEG",
    "COPERNICUS_DEM_LICENSE", "COPERNICUS_DEM_SOURCE_URL",
    "COPERNICUS_DEM_TILE_URL", "COPERNICUS_DEM_VERTICAL_DATUM",
    "CoverageError", "FETCH_ATTEMPTS", "FetchedFile", "FootprintBBox",
    "HIGHRES_FETCH_REMEDY", "HighresFetchRefusal", "LANDCOVER_SOURCES",
    "SOILGRIDS_COMPONENTS", "SOILGRIDS_CRS", "SOILGRIDS_DEPTHS",
    "SOILGRIDS_NODATA", "SOILGRIDS_SCALE", "SRTM_GL1_LICENSE",
    "SRTM_GL1_ATTRIBUTION", "SRTM_GL1_NODATA", "SRTM_GL1_SOURCE_URL",
    "SRTM_GL1_STEP_DEG", "SRTM_GL1_TILE_URL", "SRTM_GL1_VERTICAL_DATUM",
    "SourceAbsent", "SourceCoverage",
    "TERRAIN_SOURCES", "THREE_DEP_TILE_URL", "copernicus_dem_tile_ids",
    "copernicus_tile_bbox", "derive_global_terrain_window",
    "fetch_srtm_gl1_tiles", "one_degree_tile_bbox", "srtm_tile_ids",
    "derive_landcover_window", "derive_terrain_window", "domain_footprint",
    "fetch_annual_nlcd", "fetch_copernicus_dem_tiles", "fetch_file",
    "fetch_soilgrids", "fetch_three_dep_tiles", "nlcd_year_for",
    "record_local_artifact", "terrain_source_coverage", "three_dep_tile_ids",
]
