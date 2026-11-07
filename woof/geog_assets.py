"""``woof fetch-geog``: download and stage the WPS_GEOG static tree.

The WRF static builder opens nine WPS_GEOG dataset directories by
default (:data:`woof.domain_wizard.GEOG_DATASETS`); until this command existed,
staging them was entirely manual.  ``fetch-geog`` downloads the dataset
tarballs, verifies every byte, extracts them into the standard layout,
and validates each dataset the same way ``woof doctor`` does (the
``index`` file must exist -- here it must also parse).

The pin table is wider than those nine: the MPAS static builder
(``rw_mpas_static``) also reads the Noah-MP soil products, which arrive
as one ``soilgrids`` archive holding seven dataset directories under a
common parent.  A pin declares such children in ``index_subdirs`` so
validation looks for the index where it actually is, which keeps a
multi-dataset product table work rather than a second code path.

WHICH DOOR needs a pin is a column of that table too
(:data:`GEOG_CONSUMERS`, and each pin's ``required_by``), not a branch
in the code that reads it.  ``wrf`` is the WRF static builder's nine;
``mesh`` is what ``woof mesh`` must open to write the static half of
its pair: six of those nine, the soil archive, and (since its default
became MPAS-A's own static in 2.9.0) the lake-free MODIS land use and
the USGS land use native ``init_atmosphere`` reads.  Every
reader -- the default fetch set, ``--datasets wrf``, and each of
``woof doctor``'s two completeness verdicts -- is that column read at
a different scope, so adding a dataset for a future door is a row and a
consumer name rather than a new code path.

The default set carries only permissively licensed sources.  Soil categories come from the SoilGrids250m
tiles (CC BY 4.0) in the ``soilgrids`` archive, and deep-soil temperature
from ``soiltemp_era5_1deg``, which is not downloaded at all: it is BUILT
into the root from ERA5 by ``rw_zarr soiltemp-climatology``
(:data:`GEOG_PRODUCERS`).  WPS's own defaults for those two fields, the
FAO-UNESCO soil tiles and ISLSCP-I's ``soiltemp_1deg``, are pins with a
``licence_hold``: fetched and selected by name only, with their terms
printed whenever they are.

The dust door ``chem-dust`` adds three upstream-only archives. They are
not in the mirror or mandatory bundle and are fetched only when selected.
The default fetch set stays the WRF and mesh set.

The soil archive is a REQUIRED download for the mesh door and no part
of a WRF-only install.  It is large (864 MB compressed, ~13 GB
unpacked) and the default fetch stages it anyway, because a tree the
installer called complete and ``woof mesh`` then refuses is worse than
a long download nobody was surprised by.  ``--datasets wrf`` is the
spelling for an operator who knows the mesh door is not theirs.

Two download sources serve byte-identical archives:

* ``hf`` -- a byte-for-byte mirror on Hugging Face
  (:data:`HF_MIRROR_BASE_URL`), a verbatim copy of the NCAR per-dataset
  tarballs republished for CDN bandwidth with attribution.  Every
  archive is verified against the size + SHA-256 pins in
  :data:`GEOG_ARCHIVES` before extraction.
* ``ncar`` (default) -- the upstream NCAR server (:data:`NCAR_BASE_URL`), a single
  host that can be slow.  NCAR publishes no checksums, so the same pins
  apply; they were computed from a TLS download from UCAR on
  2026-07-29, which is the accurate limit of first-download integrity for
  this source (see docs/public/DATA.md). If NCAR republishes an
  archive the pin mismatch refuses loudly. The native production route
  refuses ``--allow-upstream-drift`` because another dataset version
  would change the prepared fields.

The optional lake-depth and BNU soil archives are available from NCAR only. Source
availability is declared per archive; a request through the mirror uses
the declared upstream source for a dataset the mirror does not carry.

``--bundle`` fetches NCAR's single ``geog_high_res_mandatory.tar.gz``
(2.6 GB compressed, ~29 GB unpacked) instead of the per-dataset
tarballs and extracts only the requested datasets from it.  The bundle
exists only on NCAR.  The per-dataset route downloads half as many
bytes (~1.3 GB) and is the default.

Rust retries stalled or interrupted HTTP ranges during a transfer. A new
process restarts an incomplete object; explicit injected legacy transports
retain their recorded on-disk resume contract. A
complete verified archive or an already-valid dataset directory is
never re-downloaded, and every fetch appends to a local
``geog-fetch-manifest.json`` recording url/bytes/sha256 per archive for
future re-verification.

No case is named here; WPS_GEOG is a public static dataset, not a case.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from woof import fetch_guard
from woof.timeline.ledger import span

GEOG_FETCH_MANIFEST_SCHEMA = "gpuwm-geog-fetch-manifest-v1"
GEOG_FETCH_MANIFEST_NAME = "geog-fetch-manifest.json"

#: The WPS_GEOG mirror (--source hf): a Hugging Face dataset repo holding the
#: nine NCAR per-dataset tarballs byte-for-byte (plus a MANIFEST.sha256
#: and provenance README).  Resolve URLs are plain HTTPS behind a CDN;
#: no account or hub library is needed to download.
HF_MIRROR_REPO = "deepguess/wps-geog-arwen"
HF_MIRROR_BASE_URL = (
    f"https://huggingface.co/datasets/{HF_MIRROR_REPO}/resolve/main")

#: Upstream: the NCAR WPS geographical-data server (single host; the
#: index page is get_sources_wps_geog.html).  No checksums published.
NCAR_BASE_URL = "https://www2.mmm.ucar.edu/wrf/src/wps_files"

#: Override the download base URL (private mirrors, tests).  The archive
#: filename is appended; the override wins over --source's base.
GEOG_URL_BASE_ENV = "WOOF_GEOG_URL_BASE"

GEOG_SOURCES = ("hf", "ncar")

#: The doors a pin can be required by.  A door a user can type, never a
#: builder's internal name: ``wrf`` is the WRF static path (``woof
#: prepare``/``woof go``, via ``rw-wps``) and ``mesh`` is ``woof
#: mesh``'s static half (``rw_mpas_static``).  A consumer name is also
#: accepted by ``--datasets``, so the scoping is selectable and not just
#: declarative.
GEOG_CONSUMER_WRF = "wrf"
GEOG_CONSUMER_MESH = "mesh"
GEOG_CONSUMER_CHEM_DUST = "chem-dust"
GEOG_CONSUMERS = (GEOG_CONSUMER_WRF, GEOG_CONSUMER_MESH, GEOG_CONSUMER_CHEM_DUST)


def default_geog_consumers() -> tuple[str, ...]:
    """Doors in a default install, derived from the pin scope and opt-in column."""
    return tuple(c for c in GEOG_CONSUMERS
                 if any(a.fetch_by_default and c in a.required_by for a in GEOG_ARCHIVES))

#: The line that sets up the geography a WRF-grid forecast needs: the
#: datasets the WRF static builder opens, and not the land-use pins only
#: ``woof mesh`` reads.  Every place that tells a person to set the
#: tree up quotes this one line (the run door's missing-input refusal, the
#: web page's start buttons).
WRF_FETCH_COMMAND = f"woof fetch-geog --datasets {GEOG_CONSUMER_WRF}"

#: Subdirectory of the geog root that holds in-flight and verified
#: archives (kept only under --keep-archives after extraction).
ARCHIVE_SUBDIR = ".fetch-geog"

#: With --allow-upstream-drift, an unpinned NCAR archive is still
#: refused outside [pin/2, pin*2] -- wide enough for a republished
#: dataset, narrow enough to catch an error page or truncation.
DRIFT_SIZE_BAND = (0.5, 2.0)

_BLOCK_BYTES = 8 * 1024 * 1024
_USER_AGENT = "gpuwm-fetch-geog/1"
_TIMEOUT_S = 120


@dataclass(frozen=True)
class GeogArchive:
    """One WPS_GEOG dataset tarball, pinned by size and SHA-256.

    ``dataset`` is the extracted top-level directory name (equal to the
    entry in :data:`woof.domain_wizard.GEOG_DATASETS`); ``filename`` is
    the tarball name, identical on NCAR and the mirror.  The pins were
    computed from the NCAR bytes on 2026-07-29 (sizes independently
    confirmed against the server's Content-Length); the mirror serves
    the same bytes, so one pin covers both sources.
    ``extracted_bytes`` is the measured sum of member sizes -- sizing
    information for humans, not a verification bar.

    ``index_subdirs`` names the child directories that carry the WPS
    ``index`` when one archive ships several datasets under a common
    top-level directory.  Most archives are a single dataset and leave it
    empty, which keeps the index at ``<dataset>/index``.  Declaring the
    children here rather than special-casing an archive name is what lets
    a future multi-dataset product be table work instead of a new code
    path.

    ``in_mandatory_bundle`` records whether NCAR's single mandatory-fields
    bundle carries this dataset.  That is a measured fact about that one
    archive, so a pin whose presence in it nobody has verified says False
    and ``--bundle`` refuses the dataset up front instead of extracting
    nothing and calling it done.

    ``required_by`` names the doors that cannot open without this pin,
    from :data:`GEOG_CONSUMERS`.  It replaced a single ``optional`` flag,
    which could only say "somebody needs this" and "nobody does" -- and
    the archive nine-tenths of installs never open is the one the mesh
    door hard-requires, so that flag made ``woof doctor`` choose between
    demanding 13 GB of a WRF-only box and calling a tree complete that
    ``woof mesh`` refuses.  Scoped by consumer, each verdict reads its
    own column and both are right. ``optional_for`` names consumers that
    read a dataset only when a selected model option needs it. It does not
    change the consumer's mandatory geography set.

    ``available_sources`` lists the hosts that carry these exact bytes.
    The first serves requests whose preferred host does not carry the file.

    ``producer`` names a row of :data:`GEOG_PRODUCERS` for a dataset that
    is BUILT into the root from a public source rather than downloaded as
    a tarball (no filename, no tarball pin; ``archive_bytes`` is what the
    build reads over the network, so the size quoted before a fetch is
    right).  ``licence_hold`` is non-empty for a dataset whose licence
    does not clear the public-package bar (permissive terms only on any
    default path; WPS_GEOG soil sources that are not permissive are held
    out of the default geog set): it is
    then never in the default fetch and never required by a door, only
    fetched or selected by name, and the text -- the source's own terms --
    is printed whenever it is staged or selected.
    """

    dataset: str
    filename: str
    archive_bytes: int
    archive_sha256: str
    extracted_bytes: int
    index_subdirs: tuple[str, ...] = ()
    in_mandatory_bundle: bool = True
    required_by: tuple[str, ...] = (GEOG_CONSUMER_WRF,)
    optional_for: tuple[str, ...] = ()
    available_sources: tuple[str, ...] = GEOG_SOURCES
    # Optional consumers must opt in without changing the default fetch bill.
    fetch_by_default: bool = True
    # None uses the invocation's source. Some upstream products are not mirrored.
    download_source: str | None = None
    pin_date: str = "2026-07-29"
    pin_host: str = "www2.mmm.ucar.edu"
    producer: str | None = None
    licence_hold: str = ""


#: The terms of the WPS_GEOG sources held out of the default set, quoted
#: from the datasets' own documentation: the licence table carries the
#: source and licence lines; a source that is not permissive is said so and
#: held out of the default geog set.
FAO_DSMW_HOLD = (
    "STATSGO (USDA NRCS, public domain) inside CONUS + the FAO-UNESCO "
    "Digital Soil Map of the World elsewhere, whose notes (Version 3.6, "
    "Jan 2003, https://swat.tamu.edu/media/116411/notes.txt) say "
    "\"(C) FAO/UNESCO, 1995 All rights reserved worldwide\" and \"The user "
    "shall not reproduce for the benefit of third parties the data\" "
    "without permission from FAO's Publications Division: NOT PERMISSIVE, "
    "held out of the default geography; the default soil categories are "
    "SoilGrids250m (soilgrids/texture_top, texture_bot, CC BY 4.0)")
ISLSCP_SOILTEMP_HOLD = (
    "1-degree annual-mean 2 m air temperature from ISLSCP-I (ECMWF "
    "1987-88 analyses, adjusted to 1000 mb) per the Noah-MP TBOT READ_ME; "
    "the ISLSCP-I Vol. 4 record at CEDA "
    "(https://catalogue.ceda.ac.uk/uuid/b9c3effaab9fe9a2206b58af500f157d) "
    "states Permitted Use \"Unstated\" and NCAR states none: licence "
    "UNCLEAR, so not cleared and held out of the default geography; the "
    "default deep-soil temperature is soiltemp_era5_1deg (ERA5, Licence "
    "to Use Copernicus Products)")


@dataclass(frozen=True)
class GeogProducer:
    """A WPS_GEOG dataset built from a public source by a native command.

    ``command`` is the ``rw_zarr`` subcommand, ``request`` the JSON it
    reads (sources, reduction constants); ``licence`` and ``attribution``
    travel into the dataset's ``PROVENANCE.json`` and the fetch manifest.
    """

    command: str
    request: dict
    source: str
    licence: str
    attribution: str


#: Deep-soil temperature, SOILTEMP: ERA5's 1990-2019 hourly climatology
#: (WeatherBench 2's conservative 512x256 copy on Google Cloud, keyless),
#: averaged over every hour and day (day 366 weighted 7/30, the leap days
#: in 1990-2019) and reduced to sea level with the 0.0065 K/m lapse rate
#: WRF's TMN = SOILTEMP - 0.0065 * HGT undoes -- the reduction the
#: ISLSCP-I field it replaces carries.  Same 1-degree layout as
#: soiltemp_1deg, so every reader of that dataset reads this one.
GEOG_PRODUCERS: dict[str, GeogProducer] = {
    "era5-annual-t2m-sea-level": GeogProducer(
        command="soiltemp-climatology",
        request={
            "schema": "arwen.geog-soiltemp-climatology.request.v1",
            "climatology_store": (
                "https://storage.googleapis.com/weatherbench2/datasets/"
                "era5-hourly-climatology/"
                "1990-2019_6h_512x256_equiangular_conservative.zarr"),
            "temperature": "2m_temperature",
            "temperature_units": "K",
            "static_store": (
                "https://storage.googleapis.com/weatherbench2/datasets/era5/"
                "1959-2022-6h-512x256_equiangular_conservative.zarr"),
            "geopotential": "geopotential_at_surface",
            "land_sea_mask": "land_sea_mask",
            "lapse_rate_k_per_m": 0.0065,
            "gravity_m_s2": 9.80665,
            "leap_day_weight": 7.0 / 30.0,
            "land_fraction_min": 0.001,
            "description": ("Annual mean 2 m air temperature reduced to sea "
                            "level, ERA5 1990-2019 climatology (Copernicus "
                            "Climate Change Service)"),
        },
        source=("ERA5 hourly climatology 1990-2019 (WeatherBench 2, Google "
                "Cloud public bucket gs://weatherbench2)"),
        licence=("Licence to Use Copernicus Products (https://cds.climate."
                 "copernicus.eu/api/v2/terms/static/licence-to-use-"
                 "copernicus-products.pdf): "
                 "free, worldwide, commercial use allowed, attribution "
                 "required"),
        attribution=("Contains modified Copernicus Climate Change Service "
                     "information 2026 (ERA5, Hersbach et al. 2020, "
                     "doi:10.1002/qj.3803), 1990-2019 climatology as "
                     "processed by WeatherBench 2 (Rasp et al. 2024, "
                     "doi:10.1029/2023MS004019). Neither the European "
                     "Commission nor ECMWF is responsible for any use of "
                     "this information.")),
}


#: The directories the WRF static builder's ``default`` selection opens,
#: relative to the geog root (woof/static/build.py ``_DEFAULT_GEOG_DIRS``;
#: a test binds the two, and their owning pins to the ``wrf`` column).
#: Soil categories and deep-soil temperature are the permissive ones
#: (2026-10-21): SoilGrids250m tiles inside ``soilgrids`` and the ERA5
#: climatology fetch-geog builds.
WRF_DEFAULT_GEOG_DIRS = (
    "topo_gmted2010_30s", "modis_landuse_20class_30s_with_lakes",
    "soilgrids/texture_top", "soilgrids/texture_bot", "greenfrac_fpar_modis",
    "lai_modis_10m", "albedo_modis", "maxsnowalb_modis", "soiltemp_era5_1deg",
)

#: Both doors need this pin; only the WRF static builder does; only the
#: mesh one does.  Spelled out here so each row's scoping reads as a
#: column of the table rather than as an argument list.
_WRF_AND_MESH = (GEOG_CONSUMER_WRF, GEOG_CONSUMER_MESH)
_WRF_ONLY = (GEOG_CONSUMER_WRF,)
_MESH_ONLY = (GEOG_CONSUMER_MESH,)

#: The pin inventory, with opt-in consumers after the default rows.
#: A test binds the ``wrf``-scoped names to
#: woof.domain_wizard.GEOG_DATASETS and the ``mesh``-scoped ones to
#: woof.rustwx_static.REQUIRED_GEOG_DATASETS through the child
#: declarations, so neither column can drift from the builder it speaks
#: for.
GEOG_ARCHIVES: tuple[GeogArchive, ...] = (
    GeogArchive(
        "topo_gmted2010_30s", "topo_gmted2010_30s.tar.bz2", 182721518,
        "20e02ef61b16c8d66753db8f1ede0edc0ac74e6e1b87961cdbdc483040ff22d7",
        1884949355, required_by=_WRF_AND_MESH),
    # The WRF static builder's land use. The mesh door's native default
    # reads the two rows after the soil archive, but `rw_mpas_static
    # --legacy-static` opens this one, and the static builder woof hex
    # 0.4.0 ships refuses an archive without it ("missing 1 dataset(s) a
    # static needs"). Scoped WRF-only in 2.9.0, `woof fetch-geog
    # --datasets mesh` -- the line hex's own refusal prints -- staged an
    # archive every hex point mesh then refused (2.9.1, 32 MB).
    GeogArchive(
        "modis_landuse_20class_30s_with_lakes",
        "modis_landuse_20class_30s_with_lakes.tar.bz2", 32532921,
        "fe052cc1000638c32ac7191c779ad070efae458ff0751bc4aa2a716e38bec578",
        933120358, required_by=_WRF_AND_MESH),
    GeogArchive(
        "greenfrac_fpar_modis", "greenfrac_fpar_modis.tar.bz2", 990624380,
        "a31d44c7b0c3dfcbaf933b3c30b4c6e52762a3451cf4e1951d7b49f3751b4d03",
        11197440257, required_by=_WRF_AND_MESH),
    GeogArchive(
        "lai_modis_10m", "lai_modis_10m.tar.bz2", 2604794,
        "7dfe8e2a7c48c085c8d180ffda993ce878b3c054ed7e5c185414c1993b913ebf",
        30863207, required_by=_WRF_ONLY),
    GeogArchive(
        "albedo_modis", "albedo_modis.tar.bz2", 77919797,
        "cf0d3df8e6b07a2dc7222a0cbc023205a991859395292dc6161597faf02f8e6f",
        628316651, required_by=_WRF_AND_MESH),
    GeogArchive(
        "maxsnowalb_modis", "maxsnowalb_modis.tar.bz2", 4446019,
        "eca29b1707e184c942eb8f1bfdbafbb4e6650cb97c15784c293fda39e24c1371",
        52359972, required_by=_WRF_AND_MESH),
    # The default deep-soil temperature, built at fetch time from ERA5
    # (GEOG_PRODUCERS) in soiltemp_1deg's exact layout.  459 MB is the
    # network read (the 2 m temperature cube's 13 chunks on GCS, measured
    # 2026-10-21; built in 8.4 s on a rented box); the dataset itself is
    # two 69 kB tiles, its index and PROVENANCE.json.
    GeogArchive(
        "soiltemp_era5_1deg", "", 459_000_000, "", 143_000,
        in_mandatory_bundle=False, required_by=_WRF_AND_MESH,
        available_sources=GEOG_SOURCES, pin_date="2026-10-21",
        pin_host="storage.googleapis.com",
        producer="era5-annual-t2m-sea-level"),
    # The Noah-MP soil products.  One upstream archive carries seven
    # dataset directories, each with its own WPS index, so the pin
    # declares the children rather than assuming an index at the top.
    # rw_mpas_static reads soilcomp for --soilcomp and texture_layer1..4
    # for --soilcl1..4; texture_top and texture_bot are the surface and
    # bottom composites that ship alongside them.  Since 2026-10-21 the
    # WRF static builder's default soil categories are texture_top and
    # texture_bot (permissive, replacing the FAO-based soiltype_*_30s
    # tiles), so both doors require this row.  For the mesh door it was
    # always REQUIRED -- the one static writer emits the five
    # soil-composition variables and cannot invent them.
    # Pinned from a TLS download from UCAR on 2026-08-23, the same way
    # and from the same host as the nine above.  Unlike them it carries
    # terms of its own: NCAR generated the tiles from ISRIC's SoilGrids250m,
    # which is CC BY 4.0 and needs its credit wherever the archive is
    # redistributed (NOTICE; tools/publish_geog_mirror.py writes it onto
    # the mirror's page).
    GeogArchive(
        "soilgrids", "soilgrids.tar.bz2", 864090244,
        "3d42737a68a52a2be10281f0505c6cb6c05c03707e9382b3a823495afc7937bb",
        13138518107,
        ("soilcomp", "texture_top", "texture_bot",
         "texture_layer1", "texture_layer2",
         "texture_layer3", "texture_layer4"),
        in_mandatory_bundle=False, required_by=_WRF_AND_MESH),
    # The mesh door's native static (2.9.0 default) reads land use exactly
    # as MPAS-A v8.4.2 init_atmosphere does: MODIFIED_IGBP_MODIS_NOAH from
    # the 20-class archive WITHOUT lakes (a lake is water, category 17;
    # mpas_init_atm_static.F:403), and the gravity-wave-drag statistics over
    # the USGS 30 s water mask (mpas_init_atm_gwd.F). Neither is on the
    # mirror or verified in the mandatory bundle. Pinned from official TLS
    # downloads on 2026-10-14 (sizes equal the server's Content-Length).
    GeogArchive(
        "modis_landuse_20class_30s", "modis_landuse_20class_30s.tar.bz2",
        32334661,
        "b21ca154d1038ec271abaa1be2fd38a0cd055b8a4ddfaab520719478ac48d326",
        933120348, in_mandatory_bundle=False, required_by=_MESH_ONLY,
        available_sources=("ncar",), pin_date="2026-10-14"),
    GeogArchive(
        "landuse_30s", "landuse_30s.tar.bz2", 20988479,
        "143cd195ae91f64011a43eae52ca00228709672c6a2ba614cb437eeb4cd41160",
        933120263, in_mandatory_bundle=False, required_by=_MESH_ONLY,
        available_sources=("ncar",), pin_date="2026-10-14"),
    # CLM lake physics reads depth only when sf_lake_physics=1. This NCAR
    # archive is outside the mandatory bundle and is not on the mirror:
    # woof fetch-geog --datasets lake_depth --source ncar
    # Pinned from the official TLS download on 2026-10-03.
    GeogArchive(
        "lake_depth", "lake_depth.tar.bz2", 2425674,
        "7f016173f4999e67d9757bddb9e80f1b1f3794bfe820fd78b215eb599d59fd76",
        1884949370, in_mandatory_bundle=False,
        required_by=(), optional_for=_WRF_ONLY, available_sources=("ncar",)),
    # GSL terrain drag reads all twenty children together. Verified from
    # the official TLS archive on 2026-10-12; optional to the baseline.
    # In the default fetch because the packaged CONUS HRRR recipe
    # (woof/data/recipes/hrrr_v4_gsd41.toml, gwd_opt = 3) reads it.
    # Defect (2.9.3 door 6): it was fetch_by_default=False, so `woof setup
    # --with-geog` staged a tree `woof go` then refused for that recipe
    # ("missing geography indexes ... orogwd3_10m/conls/index").  14 MB.
    GeogArchive(
        "orogwd3_10m", "orogwd3_10m.tar.bz2", 14957055,
        "0ffb24427670e4114a8a64dc7106c260152e35ad0f3176e1da4149ed9eb10c84",
        93319204,
        tuple(f"{stat}{scale}" for scale in ("ls", "ss")
              for stat in ("con", "var", "oa1", "oa2", "oa3", "oa4",
                           "ol1", "ol2", "ol3", "ol4")),
        in_mandatory_bundle=False, required_by=(), optional_for=_WRF_ONLY,
        available_sources=("ncar",), fetch_by_default=True,
        download_source="ncar", pin_date="2026-10-12"),
    # TLS pins measured 2026-09-30 on www2.mmm.ucar.edu. These products
    # are absent from the Hugging Face mirror and the mandatory bundle.
    GeogArchive("erod", "erod.tar.bz2", 131569,
        "a508b15bc7bc85ba3c34bdf01f0e8c0c582666bd739331333d289a5772edccd7",
        6221082, in_mandatory_bundle=False,
        required_by=(GEOG_CONSUMER_CHEM_DUST,), fetch_by_default=False,
        download_source="ncar", available_sources=("ncar",), pin_date="2026-09-30"),
    GeogArchive("clayfrac_5m", "clayfrac_5m.tar.bz2", 443388,
        "9475bcc76119474ff4fe36215b7101c478d381097904d7c9b5e56a121470c2da",
        20250288, in_mandatory_bundle=False,
        required_by=(GEOG_CONSUMER_CHEM_DUST,), fetch_by_default=False,
        download_source="ncar", available_sources=("ncar",), pin_date="2026-09-30"),
    GeogArchive("sandfrac_5m", "sandfrac_5m.tar.bz2", 462249,
        "53e169f64237d1ade90bd40813e40181e0da640413cc27c32ce093d2c3444815",
        20250288, in_mandatory_bundle=False,
        required_by=(GEOG_CONSUMER_CHEM_DUST,), fetch_by_default=False,
        download_source="ncar", available_sources=("ncar",), pin_date="2026-09-30"),
    # Held out of the default geography (licence, not physics): the FAO-
    # based soil categories and the ISLSCP-I deep-soil temperature WPS
    # ships as its defaults.  Selectable by name only -- geog_data_res
    # '30s' (WPS's own selector for these soil tiles) or 'islscp_soiltemp'
    # for the WRF builder, --soilcat-data STATSGO / --soiltemp-data ISLSCP
    # for the mesh door -- and fetched from NCAR, never from the mirror.
    # Pins and bundle membership as measured 2026-07-29.
    GeogArchive(
        "soiltype_top_30s", "soiltype_top_30s.tar.bz2", 1639561,
        "6d45d93cc0d14ac4019f1a2857e56fb1fbc860f135e12d89fc0a569278242315",
        933120270, required_by=(), optional_for=_WRF_AND_MESH,
        available_sources=("ncar",), fetch_by_default=False,
        download_source="ncar", licence_hold=FAO_DSMW_HOLD),
    GeogArchive(
        "soiltype_bot_30s", "soiltype_bot_30s.tar.bz2", 1919818,
        "818febf2d141a46152f0990877e3ca15fd23a61961d978f71f5fdd0ea80bb335",
        933120273, required_by=(), optional_for=_WRF_ONLY,
        available_sources=("ncar",), fetch_by_default=False,
        download_source="ncar", licence_hold=FAO_DSMW_HOLD),
    GeogArchive(
        "soiltemp_1deg", "soiltemp_1deg.tar.bz2", 31057,
        "c407c0b1d9d864ff49708fa3b6ba2f17078535ccfc05a944024ad052e5f914d4",
        138645, required_by=(), optional_for=_WRF_AND_MESH,
        available_sources=("ncar",), fetch_by_default=False,
        download_source="ncar", licence_hold=ISLSCP_SOILTEMP_HOLD),
    # Alternative 30 arc-second, 16-category soil textures selected by
    # geog_data_res=bnu_soil_30s. Pinned from official TLS downloads.
    GeogArchive(
        "bnu_soiltype_top", "bnu_soiltype_top.tar.bz2", 8198836,
        "7a7eb86d585c3dc6b32297f1ea4622d929aacadc70f16e14e0193a06f233038a",
        933120270, in_mandatory_bundle=False,
        required_by=(), optional_for=_WRF_ONLY, available_sources=("ncar",)),
    GeogArchive(
        "bnu_soiltype_bot", "bnu_soiltype_bot.tar.bz2", 8078617,
        "49306298749e3ed2a6172dfe7af0023a5be0e9f5f5230a2cfd6421ebce3e81b1",
        933120273, in_mandatory_bundle=False,
        required_by=(), optional_for=_WRF_ONLY, available_sources=("ncar",)),
    # 30 arc-second MODIS LAI selected by geog_data_res = 'modis_lai' (WPS
    # GEOGRID.TBL.ARW's rel_path for that token; the operational HRRR
    # namelist names it).  The token was admitted with no pin, so a namelist
    # naming it imported and then stopped at the missing dataset.  Not on
    # the mirror or in the mandatory bundle; pinned from the official TLS
    # download on 2026-10-15 (sha256 of the bytes, extracted size the sum of
    # member sizes).
    GeogArchive(
        "lai_modis_30s", "lai_modis_30s.tar.bz2", 749633975,
        "b8188f06717767d5ed8898e5afb3a9b8b5ad46b888a530177f161ef3623acf75",
        11197440254, in_mandatory_bundle=False,
        required_by=(), optional_for=_WRF_ONLY, available_sources=("ncar",),
        fetch_by_default=False, download_source="ncar", pin_date="2026-10-15"),
    # The native MPAS static path's other selectors (rw_mpas_static,
    # MPAS-A v8.4.2 mpas_init_atm_static.F): --topo-data GTOPO30,
    # --landuse-data MODIFIED_IGBP_MODIS_NOAH_15s, and the NCEP vegetation,
    # albedo and snow-albedo choices, each read by its native directory name.
    # The two land-use archives the native DEFAULT reads are required rows
    # above; these are read only when a namelist or flag selects them, so
    # they are optional for the mesh door and not in the default fetch.
    # None was fetchable before 2.9.0, so selecting one stopped at "WPS_GEOG
    # index ... is unreadable". Not on the mirror or in the mandatory bundle;
    # downloaded from NCAR. Pinned from the official TLS downloads on
    # 2026-10-14 (sha256 of the bytes, extracted size the sum of member sizes).
    GeogArchive(
        "topo_30s", "topo_30s.tar.bz2", 167515986,
        "c4ce6bac221767e7946fe3e544a97c1fa5b10cc0c5ee02d9fee5319ddc12a51b",
        1884949330, in_mandatory_bundle=False,
        required_by=(), optional_for=_MESH_ONLY, available_sources=("ncar",),
        fetch_by_default=False, download_source="ncar", pin_date="2026-10-14"),
    # Also the WRF static builder's land use where a namelist.wps names
    # geog_data_res = 'modis_15s' (WPS GEOGRID.TBL.ARW's rel_path for that
    # token; the operational HRRR namelist names it).
    GeogArchive(
        "modis_landuse_20class_15s", "modis_landuse_20class_15s.tar.bz2", 65224208,
        "9dd85b97221062c9a69ece3fe1dfd964702d516c35ff25cdc06c16aea6411028",
        3732480347, in_mandatory_bundle=False,
        required_by=(), optional_for=_WRF_AND_MESH, available_sources=("ncar",),
        fetch_by_default=False, download_source="ncar", pin_date="2026-10-14"),
    GeogArchive(
        "greenfrac", "greenfrac.tar.bz2", 2377411,
        "38f4d78f386094d73bb384d7bdce99cf5dffb4231b245b6dad8d847fd2a38079",
        37861126, in_mandatory_bundle=False,
        required_by=(), optional_for=_MESH_ONLY, available_sources=("ncar",),
        fetch_by_default=False, download_source="ncar", pin_date="2026-10-14"),
    GeogArchive(
        "albedo_ncep", "albedo_ncep.tar.bz2", 2074979,
        "d13ad7b44ffaadaf9fab8a731eb0cc9f1db0a2a03e4254a4a3dbe6c7a4d54461",
        37861126, in_mandatory_bundle=False,
        required_by=(), optional_for=_MESH_ONLY, available_sources=("ncar",),
        fetch_by_default=False, download_source="ncar", pin_date="2026-10-14"),
    GeogArchive(
        "maxsnowalb", "maxsnowalb.tar.bz2", 8886,
        "62267cec1ea4b2b6868d646a27911bf097527e602abfc06c4ce006f1c86d6d1c",
        69404, in_mandatory_bundle=False,
        required_by=(), optional_for=_MESH_ONLY, available_sources=("ncar",),
        fetch_by_default=False, download_source="ncar", pin_date="2026-10-14"),
    # The other two orographic drag statistics, GEOGRID.TBL.ARW's
    # ``default`` rows (woof/static/orographic.py): orogwd_10m carries
    # VAR/CON/OA1-4/OL1-4 for gwd_opt 1 and 3, varsso_10m the small-scale
    # VAR_SSO.  A run that selects orographic drag reads them (the CONUS
    # HRRR recipe's gwd_opt = 3 and WOOF Global's default suite read
    # orogwd_10m beside orogwd3_10m), so the default fetch stages them or
    # a fresh machine's default run refuses for want of the statistics.
    # NCAR-only (the mirror does not carry them); pinned from TLS
    # downloads from UCAR on 2026-10-06 (development branch).
    GeogArchive(
        "orogwd_10m", "orogwd_10m.tar.bz2", 3332672,
        "0e8c292c901e5148cb80849dc154d19cc885393658c5725741da597859777e5c",
        46658851,
        ("con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4", "var"),
        in_mandatory_bundle=False, required_by=(), optional_for=_WRF_ONLY,
        available_sources=("ncar",), fetch_by_default=True,
        download_source="ncar", pin_date="2026-10-06"),
    GeogArchive(
        "varsso_10m", "varsso_10m.tar.bz2", 783440,
        "a2420ea9698a30afdb46a82a0c13961ca17603fb11a4e657c973cf2dff95c727",
        6221332, in_mandatory_bundle=False, required_by=(),
        optional_for=_WRF_ONLY, available_sources=("ncar",),
        fetch_by_default=True, download_source="ncar", pin_date="2026-10-06"),
)

#: NCAR's "highest resolution mandatory fields" bundle: the fallback
#: when the per-dataset tarballs are unavailable.  NCAR-only.
MANDATORY_BUNDLE_FILENAME = "geog_high_res_mandatory.tar.gz"
MANDATORY_BUNDLE_BYTES = 2772782816
MANDATORY_BUNDLE_SHA256 = \
    "89b026b9db0a03c0c995e53b4a1d99663af1f6bda21b3b34c3c2c07386da5493"
#: Datasets of the nine that the bundle actually contains (verified by
#: walking the archive on 2026-07-29: 22,783 files, 30.7 GB unpacked,
#: everything nested under one WPS_GEOG/ directory the extractor
#: strips).  A requested dataset outside this set cannot be served by
#: --bundle and refuses up front.
MANDATORY_BUNDLE_DATASETS: frozenset[str] = frozenset(
    archive.dataset for archive in GEOG_ARCHIVES
    if archive.in_mandatory_bundle)


class GeogFetchError(RuntimeError):
    """An operational fetch failure with its remedy in the message."""


def geog_datasets() -> tuple[str, ...]:
    """Every dataset ``fetch-geog`` can stage, in canonical order."""
    return tuple(archive.dataset for archive in GEOG_ARCHIVES)


def datasets_required_by(consumer: str) -> tuple[str, ...]:
    """The pins ``consumer``'s door cannot open without, in table order.

    The scoped answer replaced a single "required set" that had to be
    true for every door at once.  There is no such set: the WRF static
    builder never opens the soil container and the mesh static builder
    hard-requires it, so one list could only overstate one door's needs
    or understate the other's.  ``woof doctor`` asks per consumer and
    reports two verdicts that do not contradict each other.
    """

    if consumer not in GEOG_CONSUMERS:
        raise ValueError(f"unknown geography consumer {consumer!r}; "
                         f"expected one of {', '.join(GEOG_CONSUMERS)}")
    return tuple(archive.dataset for archive in GEOG_ARCHIVES
                 if consumer in archive.required_by)


def packaged_recipe_datasets() -> tuple[str, ...]:
    """Pins the packaged recipes' terrain-drag options read, in table order.

    The recipes under ``woof/data/recipes`` select ``gwd_opt`` and
    ``topo_wind`` in ``[shared]`` or per ``[[domain]]``; each selection
    reads orographic datasets that no consumer column requires.  ``woof
    doctor`` checks these so it agrees with ``woof go``'s geography
    refusal on a default tree (defect: doctor said "all 9 WRF datasets
    indexed" while go refused the packaged CONUS HRRR recipe for
    orogwd3_10m).  Empty where the recipes or the static package are not
    installed (the preprocessing-only wheel).
    """

    import tomllib
    recipes = sorted((Path(__file__).parent / "data" / "recipes").glob("*.toml"))
    try:
        from woof.static.orographic import (orographic_paths,
                                             required_static_fields)
    except ImportError:
        return ()
    required: set[str] = set()
    for recipe in recipes:
        raw = tomllib.loads(recipe.read_text(encoding="utf-8-sig"))
        shared = raw.get("shared", {})
        for table in [shared, *raw.get("domain", [])]:
            names = required_static_fields(
                int(table.get("topo_wind", shared.get("topo_wind", 0)) or 0),
                int(table.get("gwd_opt", shared.get("gwd_opt", 0)) or 0))
            if names:
                required.update(path.as_posix() for path in
                                orographic_paths(names, Path(".")).values())
    owners = archive_names_for_paths(required)
    return tuple(a.dataset for a in GEOG_ARCHIVES if a.dataset in owners)


def size_phrase(datasets: tuple[str, ...] | None = None) -> str:
    """``'~2.2 GB compressed, ~30 GB unpacked'`` for a set of pins.

    ``None`` means the default fetch set. Decimal
    GB, matching every other published size in this product.

    Published because three surfaces quote these numbers before the
    first byte moves -- ``woof setup --with-geog``, the notice printed
    when setup skips the tree, and ``woof doctor``'s remedy -- and all
    three carried the figure as a literal.  Adding a pin left every one
    of them understating the download by most of its size, in the exact
    place a reader is deciding whether to start it.
    """

    chosen = (tuple(a for a in GEOG_ARCHIVES if a.fetch_by_default) if datasets is None
              else tuple(archive_for(name) for name in datasets))
    compressed = sum(archive.archive_bytes for archive in chosen) / 1e9
    unpacked = sum(archive.extracted_bytes for archive in chosen) / 1e9
    return (f"~{compressed:.1f} GB compressed, "
            f"~{unpacked:.0f} GB unpacked")


def archive_for(dataset: str) -> GeogArchive:
    for archive in GEOG_ARCHIVES:
        if archive.dataset == dataset:
            return archive
    raise ValueError(f"unknown WPS_GEOG dataset {dataset!r}")


def parse_datasets(raw: str) -> tuple[str, ...]:
    """Parse ``--datasets`` into dataset names, in canonical order.

    ``default`` selects fetch_by_default rows; ``all`` selects every pin.
    Other spellings are a consumer name from
    :data:`GEOG_CONSUMERS` standing for that door's whole set, and an
    explicit comma list.  They mix, and duplicates collapse.

    ``--datasets wrf`` is the subset spelling for an operator who knows
    the mesh door is not theirs and does not want its ~13 GB.  It is a
    selector over the scoping column rather than a ``--skip`` flag,
    because a flag would have to be re-argued for every future door and
    this stays a row in the table.
    """

    known = geog_datasets()
    if raw.strip().lower() == "default":
        return tuple(a.dataset for a in GEOG_ARCHIVES if a.fetch_by_default)
    if raw.strip().lower() == "all":
        return known
    requested = [part.strip() for part in raw.split(",") if part.strip()]
    if not requested:
        raise ValueError("--datasets must be 'all', a consumer name "
                         f"({', '.join(GEOG_CONSUMERS)}), or a "
                         "comma-separated list of dataset names")
    picked: set[str] = set()
    unknown: list[str] = []
    for name in requested:
        if name in GEOG_CONSUMERS:
            picked.update(datasets_required_by(name))
        elif name in known:
            picked.add(name)
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(
            f"unknown dataset(s) {', '.join(sorted(set(unknown)))}; expected "
            f"'all', a consumer name ({', '.join(GEOG_CONSUMERS)}), or names "
            f"from: {', '.join(known)}")
    return tuple(name for name in known if name in picked)


def resolve_source(source: str | None, bundle: bool) -> str:
    """Concrete source for one invocation; ``--bundle`` is NCAR-only."""

    if source is not None and source not in GEOG_SOURCES:
        raise ValueError(f"unknown --source {source!r}; expected one of "
                         f"{GEOG_SOURCES}")
    if bundle:
        if source == "hf":
            raise ValueError(
                "--bundle is served by NCAR only (the mirror republishes "
                "the per-dataset tarballs); drop --bundle or use "
                "--source ncar")
        return "ncar"
    return source if source is not None else "ncar"


def archive_url(filename: str, source: str) -> str:
    """Download URL for ``filename`` from ``source`` (env override wins)."""

    override = os.environ.get(GEOG_URL_BASE_ENV)
    if override and override.strip():
        return f"{override.strip().rstrip('/')}/{filename}"
    if source == "hf":
        return f"{HF_MIRROR_BASE_URL}/{filename}"
    if source == "ncar":
        return f"{NCAR_BASE_URL}/{filename}"
    raise ValueError(f"unknown source {source!r}")


#: The ``--geog-root`` help every door that prepares static fields shows.
GEOG_ROOT_HELP = ("override the geography tree (default: the config's "
                  "[case_data].geog_root or [static].geog_root, "
                  "otherwise the staged WPS_GEOG tree)")


def add_geog_root_argument(parser, *, help: str = GEOG_ROOT_HELP):
    """Add the shared ``--geog-root DIR`` option to a door's parser.

    One definition for the doors that prepare static fields, so their
    spelling, type and default cannot drift.  Breakage (2.9.4 landing,
    test host 2026-10-20): ``woof earth2`` had no ``--geog-root`` while
    ``woof go`` had one, so an Earth-2 nest could only read the default
    tree.
    """

    return parser.add_argument("--geog-root", type=Path, default=None,
                               metavar="DIR", help=help)


def default_geog_root() -> Path:
    """The root fetch-geog stages into when ``--root`` is not given.

    Exactly the tree the rest of woof already reads:
    ``$GPUWM_CASE_DATA_ROOT/WPS_GEOG`` -- the default ``geog_root`` in
    wizard-emitted configs and the path ``woof doctor`` checks -- with
    the same unset-variable fallback as every other case-data path
    (:func:`woof.case_data.case_data_root`).
    """

    from woof.case_data import case_data_root

    return case_data_root() / "WPS_GEOG"


def geog_root_from_tables(raw, *, base_dir: Path) -> Path:
    """The saved geography root, independent of the forcing source."""
    from woof.case_data import expand_path_variables

    value = raw.get("case_data", {}).get("geog_root")
    if value is None:
        value = raw.get("static", {}).get("geog_root")
    if value is None:
        return default_geog_root()
    if not isinstance(value, str) or not value.strip():
        raise ValueError("geog_root must be a non-empty path string")
    path = Path(expand_path_variables(value, "geog_root", str(base_dir))).expanduser()
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def config_geog_root(config: Path) -> Path:
    """Resolve a configuration's geography root beside its TOML file."""
    import tomllib

    config = Path(config).resolve()
    raw = tomllib.loads(config.read_text(encoding="utf-8-sig"))
    return geog_root_from_tables(raw, base_dir=config.parent)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    return _rw_asset("hash", input=path)["sha256"]


def dataset_corpus(root: Path, dataset: str) -> tuple[int, int]:
    """``(regular file count, total bytes)`` of one installed dataset.

    A cheap stat walk: the tile trees run to gigabytes, so re-hashing
    them at every use is not a bar anybody would leave switched on.
    """

    files = 0
    total = 0
    for path in (root / dataset).rglob("*"):
        if path.is_file():
            files += 1
            total += path.stat().st_size
    return files, total


def installed_receipt(root: Path, dataset: str) -> dict | None:
    """What the local manifest says was installed for ``dataset``.

    Returns the ``{"files": n, "bytes": n}`` entry recorded when the
    dataset was extracted, or ``None`` when no manifest claims it (an
    install from before this receipt existed, or a directory some other
    tool staged).
    """

    manifest = _load_manifest(root)
    for entry in manifest.get("archives", {}).values():
        if not isinstance(entry, dict):
            continue
        datasets = entry.get("datasets")
        if not isinstance(datasets, dict):
            continue
        recorded = datasets.get(dataset)
        if (isinstance(recorded, dict)
                and isinstance(recorded.get("files"), int)
                and isinstance(recorded.get("bytes"), int)):
            return recorded
    return None


def validate_dataset_dir(root: Path, dataset: str, *,
                         check_receipt: bool = True) -> tuple[bool, str]:
    """(ok, detail) for one staged dataset directory.

    The bar is doctor's own -- the directory exists and carries its WPS
    ``index`` file -- plus the index must actually parse
    (:func:`woof.static.geog.parse_index`), so a zero-byte or truncated
    index cannot pass, and it must declare a projection the reader can
    resolve (:func:`woof.static.geog.unsupported_projection_detail`),
    so a tree of projected tiles is named here rather than at dataset
    construction inside a run.

    And, when the local manifest recorded what was installed, the tile
    corpus must still match that receipt.  Index-only validation was the
    whole bar until now, which meant a directory whose *tiles* were
    corrupt, truncated, deleted or added to was classified valid beside
    an intact ``index`` and reused indefinitely -- fetch returned zero
    work and never looked at the manifest.  Comparing the file count and
    total byte count against the extraction receipt catches missing,
    extra and short tiles for the cost of a stat walk.  It does not
    catch a same-size mutation of a tile's contents: that needs a full
    content digest of a multi-gigabyte tree, which is a deliberate
    verification mode rather than something to run on every command.

    ``check_receipt=False`` asks for the historical index-only bar, for
    callers validating a tree they have just staged themselves.
    """

    directory = root / dataset
    if not directory.is_dir():
        return False, f"{directory} does not exist"
    try:
        subdirs = archive_for(dataset).index_subdirs
    except ValueError:
        subdirs = ()
    # A single-dataset archive carries one index at the top; a container
    # archive carries one per declared child.  Every index named here has
    # to pass, so a container missing one child is a finding, not a pass
    # earned by its siblings.
    for index in ([directory / "index"] if not subdirs
                  else [directory / name / "index" for name in subdirs]):
        if not index.is_file():
            return False, f"{index.parent} lacks its WPS `index` file"
        try:
            from woof.static.geog import (parse_index,
                                           unsupported_projection_detail)
            parsed = parse_index(index)
            # parse_index defaults absent keys to None; a truncated or
            # foreign file "parses" vacuously, so require the fields no
            # real WPS index omits, and a decodable word size.
            absent = [key for key in ("dx", "dy", "known_lat", "known_lon",
                                      "wordsize", "tile_x", "tile_y")
                      if getattr(parsed, key) is None]
            if absent:
                return False, (f"{index} lacks required WPS index key(s): "
                               + ", ".join(absent))
            parsed.dtype  # noqa: B018 -- raises on an undecodable wordsize
            # A tree whose index declares a projection the reader cannot
            # resolve parses perfectly and is unreadable.  Name it here,
            # at the inventory every fetch door already consults, so it
            # is named before anything downloads or builds rather than
            # at dataset construction inside a run.  Same predicate the
            # reader raises on, so the two doors cannot disagree.
            projection_detail = unsupported_projection_detail(
                parsed.projection)
            if projection_detail is not None:
                return False, f"{index}: {projection_detail}"
        except Exception as error:  # any parse failure is the finding
            return False, f"{index} does not parse as a WPS index: {error}"
    if not check_receipt:
        return True, f"{directory} present, index parses"
    receipt = installed_receipt(root, dataset)
    if receipt is None:
        return True, (f"{directory} present, index parses (no local "
                      f"{GEOG_FETCH_MANIFEST_NAME} entry to check the "
                      "tile corpus against)")
    files, total = dataset_corpus(root, dataset)
    if (files, total) != (receipt["files"], receipt["bytes"]):
        return False, (
            f"{directory} no longer matches what "
            f"{GEOG_FETCH_MANIFEST_NAME} records was installed: "
            f"{files} files / {total:,} B on disk, "
            f"{receipt['files']} files / {receipt['bytes']:,} B recorded "
            "(a tile is missing, truncated, or added)")
    return True, (f"{directory} present, index parses, {files} tiles / "
                  f"{total:,} B match the recorded install")


def _quarantine(path: Path, progress, label: str) -> Path:
    """Move a refused file aside (never deleted, never reused).

    The name is proven free first: ``.rejected-<time_ns>`` on its own
    collides inside a clock tick, and ``os.replace`` onto the collision
    destroys the older evidence.
    """

    aside = fetch_guard.quarantine(path)
    progress(f"fetch-geog {label}: moved rejected file aside to "
             f"{aside.name}")
    return aside


# ---------------------------------------------------------------------------
# Download (resumable)
# ---------------------------------------------------------------------------

def _default_urlopen(request: Request):
    return urlopen(request, timeout=_TIMEOUT_S)


#: Sidecar recording which object a partial archive is a prefix OF.
_RESUME_SIDECAR_SUFFIX = ".fetch-resume.json"


def _resume_sidecar(dest: Path) -> Path:
    return dest.with_name(dest.name + _RESUME_SIDECAR_SUFFIX)


def _resume_identity(response, url: str) -> dict:
    headers = getattr(response, "headers", None)
    getter = headers.get if headers is not None else (lambda _key: None)
    return {
        "url": url,
        "etag": getter("ETag"),
        "last_modified": getter("Last-Modified"),
    }


def _partial_note(dest: Path, sidecar: Path, *, strict_size: bool) -> str:
    """What a failed transfer leaves, in the terms the next run acts on.

    Mirrors :func:`download_archive`'s own resume rule: a partial is
    resumed when a resume record binds it, or on the pinned path, where
    the exact size and SHA-256 pin catches any mixture; otherwise the
    next run sets it aside and starts again.  The old text promised a
    resume after every failure, including ones that left nothing on disk.
    """

    try:
        size = dest.stat().st_size
    except OSError:
        size = 0
    if size <= 0:
        return "nothing was downloaded, so a re-run starts from the beginning"
    if strict_size or sidecar.is_file():
        return (f"the partial file ({size:,} B) is kept and a re-run "
                "resumes from it")
    return (f"the partial file ({size:,} B) has no resume record, so a "
            "re-run sets it aside and downloads it from the start")


def _content_range_start(value: str) -> int | None:
    """The first byte a ``206`` says it is sending, or None."""

    text = (value or "").strip()
    if not text.lower().startswith("bytes "):
        return None
    span = text[6:].split("/", 1)[0].strip()
    first = span.split("-", 1)[0].strip()
    try:
        return int(first)
    except ValueError:
        return None


def _rw_asset(kind: str, *, on_progress=None, **options) -> dict:
    """Drive the existing Rust bridge; no transport or hashing in Python."""
    from woof.rustwx_fetch import find_fetch_bin, fetch_remedy, probe_fetch_bin, _run
    binary = find_fetch_bin()
    if binary is None:
        raise GeogFetchError("rw_fetch is missing; static downloads cannot be "
                             "verified and published through Rust. " + fetch_remedy())
    ok, evidence = probe_fetch_bin(binary)
    if not ok:
        raise GeogFetchError(evidence)
    command = [os.fspath(binary), kind]
    for name, value in options.items():
        command.extend(["--" + name.replace("_", "-"), str(value)])
    try:
        return _run(command, what=kind, on_progress=on_progress)
    except (RuntimeError, OSError) as error:
        raise GeogFetchError(str(error)) from error


def download_archive(url: str, dest: Path, *, expected_bytes: int,
                     progress, label: str, strict_size: bool = True,
                     urlopen_fn=_default_urlopen,
                     report_every_bytes: int | None = None,
                     expected_sha256: str | None = None) -> dict | None:
    """Default transport: bounded Rust range GETs with an in-stream pin."""
    # Retain the explicit injected legacy transport for pre-existing callers.
    # Production callers always supply the hash and use the Rust door.
    if urlopen_fn is not _default_urlopen:
        return _download_archive_serial(
            url, dest, expected_bytes=expected_bytes, progress=progress,
            label=label, strict_size=strict_size, urlopen_fn=urlopen_fn,
            report_every_bytes=report_every_bytes)
    if not strict_size:
        raise GeogFetchError(f"{label}: --allow-upstream-drift is refused by "
                             "the pinned static transport; unpinned geography "
                             "would change the prepared fields under the same dataset version")
    if expected_sha256 is None:
        raise GeogFetchError(f"{label}: no SHA-256 pin was supplied; "
                             "unverified geography would change the prepared run")
    if dest.exists():
        try:
            return _rw_asset("hash", input=dest, bytes=expected_bytes,
                             sha256=expected_sha256)
        except GeogFetchError:
            _quarantine(dest, progress, label)
    with span("prep.geog.fetch", attrs={"bytes": expected_bytes,
                                       "datasets": [label]}):
        try:
            record = _rw_asset("asset", url=url, output=dest,
                               bytes=expected_bytes, sha256=expected_sha256,
                               on_progress=lambda received, total: progress(
                                   f"fetch-geog {label}: {received:,} of {total or expected_bytes:,} B"))
        except GeogFetchError as error:
            raise GeogFetchError(f"{label}: {url}: {error}") from error
    progress(f"fetch-geog {label}: pin-verified {record['bytes']:,} B through rw_fetch")
    return record


def _download_archive_serial(url: str, dest: Path, *, expected_bytes: int,
                     progress, label: str, strict_size: bool = True,
                     urlopen_fn=_default_urlopen,
                     report_every_bytes: int | None = None) -> None:
    """Download ``url`` to ``dest`` with Range resume and a size bar.

    ``report_every_bytes`` adds a progress line each time the file grows
    past another multiple of it (a single file of a gigabyte or more is
    minutes of silence otherwise); ``None`` keeps the start and end lines
    only.

    A ``dest`` left by an interrupted run resumes from its byte count
    (both NCAR and the mirror serve ranges).  A server that ignores the
    Range header restarts cleanly.  With ``strict_size`` (the pinned
    path) the final size must equal ``expected_bytes`` -- a mismatch
    refuses and quarantines rather than handing a short file to the
    hash check with a confusing story.  ``--allow-upstream-drift``
    turns ``strict_size`` off, deferring to the verify step's sanity
    band (a republished archive rarely keeps its byte count).

    Resume is bound to an *object*, not just to a byte count.  A partial
    file used to be resumed from its length alone: nothing checked that
    the bytes about to be appended came from the same URL, the same
    version, or even the requested offset, so a republished or
    reconfigured upstream could produce a chimera stitched out of two
    generations.  The exact pin normally catches that; under
    ``--allow-upstream-drift`` -- where the size band *is* the whole
    integrity policy -- a same-size chimera passed.  Now the first
    response's identity (URL, ``ETag``, ``Last-Modified``) is recorded
    in a sidecar, a resume must match it and is sent with ``If-Range``,
    and a ``206`` whose ``Content-Range`` does not start at the
    requested offset is refused.  A mismatch restarts from zero after
    setting the partial aside; nothing is deleted.
    """

    offset = dest.stat().st_size if dest.exists() else 0
    sidecar = _resume_sidecar(dest)
    recorded: dict = {}
    if offset:
        try:
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
            recorded = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            recorded = {}
        bound_to_another = bool(recorded) and recorded.get("url") != url
        # A partial with no record at all is either a pre-1.1.3 resume
        # or something else's file.  On the pinned path the exact
        # size+SHA-256 pin is a complete integrity policy and catches any
        # mixture, so such a partial is still worth resuming.  Under
        # --allow-upstream-drift there is no pin -- the size band IS the
        # policy, and a same-size chimera passes it -- so an unbound
        # partial has to start again.
        unbound_and_unpinned = not recorded and not strict_size
        if bound_to_another or unbound_and_unpinned:
            progress(
                f"fetch-geog {label}: the partial archive on disk is not a "
                f"recorded prefix of {url} "
                + (f"(it was recorded against {recorded.get('url')!r})"
                   if bound_to_another else
                   "(no resume record, and --allow-upstream-drift has no "
                   "pin to catch a mixed archive)")
                + "; setting it aside and downloading from the start")
            _quarantine(dest, progress, label)
            sidecar.unlink(missing_ok=True)
            offset = 0
            recorded = {}
    if strict_size and offset > expected_bytes:
        _quarantine(dest, progress, label)
        sidecar.unlink(missing_ok=True)
        offset = 0
        recorded = {}
    if strict_size and offset == expected_bytes:
        return
    headers = {"User-Agent": _USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        validator = recorded.get("etag") or recorded.get("last_modified")
        if validator:
            # If the object changed, the server answers 200 with the whole
            # new body and the restart below is taken -- which is exactly
            # the outcome wanted, decided by the origin rather than
            # guessed at here.
            headers["If-Range"] = validator
        progress(f"fetch-geog {label}: resuming at "
                 f"{offset / (1024 * 1024):.1f} MiB")
    request = Request(url, headers=headers)
    started = time.perf_counter()
    try:
        with urlopen_fn(request) as response:
            status = getattr(response, "status", 200)
            if offset and status == 206:
                response_headers = getattr(response, "headers", None)
                raw_range = ("" if response_headers is None
                             else response_headers.get("Content-Range") or "")
                start = _content_range_start(raw_range)
                if start != offset:
                    raise GeogFetchError(
                        f"{label}: resume asked for bytes {offset}- and the "
                        f"server answered 206 with Content-Range "
                        f"{raw_range!r}, which does not start there; the "
                        "partial file is kept and untouched -- move it "
                        "aside and re-run to download from the start")
            mode = "ab" if (offset and status == 206) else "wb"
            if offset and status != 206:
                progress(f"fetch-geog {label}: server ignored the resume "
                         "range (or the object changed); restarting the "
                         "download")
                recorded = {}
            if not recorded:
                try:
                    fetch_guard.atomic_write_text(
                        sidecar,
                        json.dumps(_resume_identity(response, url),
                                   indent=2, sort_keys=True) + "\n",
                        tag="geog")
                except OSError as error:
                    raise fetch_guard.LocalWriteFailed(
                        sidecar, error) from error
            on_bytes = None
            if report_every_bytes:
                base = offset if mode == "ab" else 0
                said = {"step": base // report_every_bytes}

                def on_bytes(written: int) -> None:
                    total = base + written
                    step = total // report_every_bytes
                    if step > said["step"]:
                        said["step"] = step
                        progress(
                            f"fetch-geog {label}: {total / 2**20:,.0f} of "
                            f"{expected_bytes / 2**20:,.0f} MiB "
                            f"({100.0 * total / expected_bytes:.0f}%)")
            fetch_guard.receive(response, dest, mode,
                                block_bytes=_BLOCK_BYTES, on_bytes=on_bytes)
    except fetch_guard.LocalWriteFailed as failure:
        # Caught before the network handlers below: a folder Windows
        # will not write into used to be reported as "download failed
        # from https://huggingface.co/...; the partial file is kept and a
        # re-run resumes" when nothing had been written, the mirror was
        # fine and every re-run failed the same way.
        raise GeogFetchError(fetch_guard.local_write_refusal(
            label, failure.path, failure.error,
            _partial_note(dest, sidecar, strict_size=strict_size))
        ) from failure.error
    except HTTPError as error:
        hint = ""
        # HF answers 401 for a repo that does not exist (or is private
        # to anonymous callers), 404 for a missing file in a live repo.
        if error.code in (401, 403, 404) and "huggingface.co" in url:
            hint = ("  (the mirror archive is absent -- if the mirror "
                    "repo is not published yet, use --source ncar)")
        raise GeogFetchError(
            f"{label}: HTTP {error.code} from {url}{hint}; "
            + _partial_note(dest, sidecar, strict_size=strict_size)
        ) from error
    except (URLError, OSError, TimeoutError) as error:
        raise GeogFetchError(
            f"{label}: download failed from {url}: {error}; "
            + _partial_note(dest, sidecar, strict_size=strict_size)
        ) from error
    size = dest.stat().st_size
    if strict_size and size != expected_bytes:
        _quarantine(dest, progress, label)
        sidecar.unlink(missing_ok=True)
        raise GeogFetchError(
            f"{label}: downloaded {size:,} B from {url}, expected "
            f"{expected_bytes:,} B; the file was moved aside -- re-run "
            "to fetch fresh (if this repeats, the upstream archive has "
            "changed; see --allow-upstream-drift)")
    # The archive is whole: the resume record has nothing left to bind.
    sidecar.unlink(missing_ok=True)
    seconds = time.perf_counter() - started
    rate = (size - offset) / (1024 * 1024) / seconds if seconds > 0 else 0.0
    progress(f"fetch-geog {label}: {dest.name} {size:,} B in "
             f"{seconds:.1f} s ({rate:.1f} MiB/s)")


def verify_archive(dest: Path, *, expected_sha256: str, expected_bytes: int,
                   source: str, allow_drift: bool, progress,
                   label: str) -> tuple[str, bool]:
    """(observed sha256, pinned?) for a completely downloaded archive.

    Pinned means the bytes match the packaged pin.  A mismatch refuses
    (quarantining the file) unless ``--allow-upstream-drift`` was given
    for an NCAR download whose size stays inside
    :data:`DRIFT_SIZE_BAND`; the mirror is our own publication, so
    drift there is corruption by definition and is never acceptable.
    """

    observed = sha256_file(dest)
    if observed == expected_sha256:
        return observed, True
    if source == "ncar" and allow_drift:
        size = dest.stat().st_size
        low = int(expected_bytes * DRIFT_SIZE_BAND[0])
        high = int(expected_bytes * DRIFT_SIZE_BAND[1])
        if not low <= size <= high:
            _quarantine(dest, progress, label)
            raise GeogFetchError(
                f"{label}: --allow-upstream-drift refused: {size:,} B is "
                f"outside the sanity band [{low:,}, {high:,}] around the "
                "pinned size -- that is an error page or truncation, not "
                "a republished dataset")
        progress(f"fetch-geog {label}: WARNING: accepting UNPINNED bytes "
                 f"under --allow-upstream-drift (sha256 {observed[:12]}..., "
                 f"pin {expected_sha256[:12]}...); recorded as unpinned in "
                 f"{GEOG_FETCH_MANIFEST_NAME}")
        return observed, False
    _quarantine(dest, progress, label)
    if source == "hf":
        raise GeogFetchError(
            f"{label}: sha256 {observed[:12]}... does not match the pin "
            f"{expected_sha256[:12]}... -- the mirror serves exactly the "
            "pinned bytes, so this is a corrupted transfer; the file was "
            "moved aside, re-run to fetch fresh")
    raise GeogFetchError(
        f"{label}: sha256 {observed[:12]}... does not match the pin "
        f"{expected_sha256[:12]}... computed from NCAR's bytes on "
        "2026-07-29.  Either the transfer corrupted (re-run fetches "
        "fresh; the mismatched file was moved aside) or NCAR republished "
        "the archive.  --source hf serves the pinned bytes; "
        "--allow-upstream-drift accepts NCAR's new bytes and records "
        "them as unpinned")


# ---------------------------------------------------------------------------
# Extraction (safe members only, staged then atomically placed)
# ---------------------------------------------------------------------------

def _safe_member(member: tarfile.TarInfo, datasets: tuple[str, ...],
                 archive_name: str) -> tuple[str, str] | None:
    """(dataset, relative path to extract) for a member, or None to skip.

    Only plain files and directories whose paths are relative, ``..``
    free, and rooted in a requested dataset directory are extracted;
    anything else (symlink, device, absolute path, traversal) is
    refused loudly -- these archives contain nothing of the sort, so
    their presence means a corrupted or hostile archive.

    The per-dataset tarballs carry their dataset directory at the top
    level (``soiltemp_1deg/index``); the mandatory bundle nests
    everything under one ``WPS_GEOG/`` directory
    (``WPS_GEOG/soiltemp_1deg/index`` -- verified by walking the
    archive).  Both shapes land identically: the extracted path always
    starts at the dataset directory.
    """

    name = member.name
    parts = [part for part in name.split("/") if part not in ("", ".")]
    if name.startswith(("/", "\\")) or ".." in parts or ":" in name:
        raise GeogFetchError(
            f"{archive_name} contains an unsafe member path {name!r}; "
            "refusing to extract")
    if not parts:
        return None
    if not (member.isfile() or member.isdir()):
        raise GeogFetchError(
            f"{archive_name} contains a non-regular member {name!r} "
            f"(type {member.type!r}); refusing to extract")
    # NCAR packed the mandatory bundle on macOS: it carries AppleDouble
    # ``._*`` sidecars (212 B resource-fork remnants) beside every
    # directory and tile.  They are not data; skipping them makes the
    # bundle and per-dataset routes produce byte-identical trees.
    if parts[-1].startswith("._") or parts[-1] == ".DS_Store":
        return None
    if parts[0] in datasets:
        return parts[0], "/".join(parts)
    if len(parts) > 1 and parts[0] == "WPS_GEOG" and parts[1] in datasets:
        return parts[1], "/".join(parts[1:])
    return None


def _extraction_staging(root: Path) -> Path:
    """The folder one extraction stages into, beside what it publishes.

    Compact for the reason :func:`woof.fetch_guard._staging_path` is:
    every staged tile sits one folder deeper than where it lands, and the
    old ``.fetch-geog-extract-<time_ns>`` put the deepest WRF tile 101
    characters below the geography folder against 61 once published.
    """

    return root / f"{ARCHIVE_SUBDIR}-{fetch_guard._staging_token()}"


def extract_datasets(archive: Path, root: Path, datasets: tuple[str, ...],
                     *, progress, label: str) -> dict[str, dict]:
    """Native codec and safe Rust tar reader, then validated atomic rename."""
    tmp = _extraction_staging(root)
    with span("prep.geog.unpack", attrs={"bytes": archive.stat().st_size,
                                        "datasets": list(datasets)}):
        try:
            counts = _rw_asset("unpack", input=archive, output=tmp,
                               datasets=",".join(datasets))
            for dataset in datasets:
                ok, detail = validate_dataset_dir(tmp, dataset, check_receipt=False)
                if not ok:
                    raise GeogFetchError(f"{label}: {detail}; incomplete tiles must not be published")
            for dataset in datasets:
                final = root / dataset
                if final.exists():
                    _quarantine(final, progress, label)
                os.replace(tmp / dataset, final)
                progress(f"fetch-geog {label}: staged {final} ({counts[dataset]['bytes']:,} B)")
            return counts
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)


def _extract_datasets_serial(archive: Path, root: Path, datasets: tuple[str, ...],
                     *, progress, label: str) -> dict[str, dict]:
    """Extract ``datasets`` from ``archive`` into ``root``.

    Members stream into a temporary directory beside the target; each
    completed dataset must carry a parsing ``index`` before it is
    renamed into place, so ``root`` never holds a half-extracted or
    index-less dataset directory.  Returns per-dataset
    ``{"files": n, "bytes": n}``.

    A write this computer refuses (a folder too deep for Windows, a full
    disk) removes the half-extracted staging tree, keeps the verified
    archive, and says which path failed and by how many characters.
    """

    tmp = _extraction_staging(root)
    counts: dict[str, dict] = {name: {"files": 0, "bytes": 0}
                               for name in datasets}
    started = time.perf_counter()
    try:
        try:
            tmp.mkdir(parents=True)
        except OSError as error:
            raise fetch_guard.LocalWriteFailed(tmp, error) from error
        with tarfile.open(archive, "r:*") as tar:
            for member in tar:
                placed = _safe_member(member, datasets, archive.name)
                if placed is None:
                    continue
                dataset, relative = placed
                target = tmp / relative
                if member.isdir():
                    try:
                        target.mkdir(parents=True, exist_ok=True)
                    except OSError as error:
                        raise fetch_guard.LocalWriteFailed(
                            target, error) from error
                    continue
                source = tar.extractfile(member)
                if source is None:
                    raise GeogFetchError(
                        f"{archive.name}: unreadable member {member.name}")
                # Only the writes are this computer's; a read that fails
                # is the archive, and keeps reaching its own handling.
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    sink = target.open("wb")
                except OSError as error:
                    source.close()
                    raise fetch_guard.LocalWriteFailed(
                        target, error) from error
                with source, sink:
                    while block := source.read(_BLOCK_BYTES):
                        try:
                            sink.write(block)
                        except OSError as error:
                            raise fetch_guard.LocalWriteFailed(
                                target, error) from error
                counts[dataset]["files"] += 1
                counts[dataset]["bytes"] += member.size
    except tarfile.TarError as error:
        raise GeogFetchError(
            f"{label}: {archive.name} failed to extract: {error} -- "
            "the archive is corrupt; move it aside and re-run") from error
    except fetch_guard.LocalWriteFailed as failure:
        # The half-extracted tree is this run's own and worth nothing;
        # left behind, every retry would add another beside it.
        shutil.rmtree(tmp, ignore_errors=True)
        raise GeogFetchError(fetch_guard.local_write_refusal(
            label, failure.path, failure.error,
            f"the verified archive is kept at {archive}")
        ) from failure.error

    for dataset in datasets:
        staged = tmp / dataset
        # Index-only here on purpose: this tree was just written by this
        # process and no receipt describes it yet -- the corpus check
        # belongs to the reuse path, against the receipt published below.
        ok, detail = validate_dataset_dir(tmp, dataset, check_receipt=False)
        if not ok:
            raise GeogFetchError(
                f"{label}: extracted dataset failed validation: {detail} "
                f"(staged under {tmp}, left for inspection)")
        final = root / dataset
        if final.exists():
            _quarantine(final, progress, label)
        os.replace(staged, final)
        progress(f"fetch-geog {label}: staged {final} "
                 f"({counts[dataset]['files']} files, "
                 f"{counts[dataset]['bytes']:,} B) in "
                 f"{time.perf_counter() - started:.1f} s")
    try:
        tmp.rmdir()  # only ever empty on success
    except OSError:
        pass
    return counts


# ---------------------------------------------------------------------------
# Local manifest
# ---------------------------------------------------------------------------

def _load_manifest(root: Path) -> dict:
    path = root / GEOG_FETCH_MANIFEST_NAME
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (isinstance(payload, dict)
                    and payload.get("schema") == GEOG_FETCH_MANIFEST_SCHEMA
                    and isinstance(payload.get("archives"), dict)):
                return payload
        except (ValueError, UnicodeDecodeError, OSError):
            pass
    return {"schema": GEOG_FETCH_MANIFEST_SCHEMA, "archives": {}}


def _write_manifest(root: Path, payload: dict) -> Path:
    payload["updated"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    return fetch_guard.atomic_write_text(
        root / GEOG_FETCH_MANIFEST_NAME,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        tag="geog")


def publish_archive_entry(root: Path, filename: str, entry: dict) -> Path:
    """Record one installed archive in the manifest, immediately.

    Two defects closed at once, and both need the read to happen here
    rather than at the start of the run.

    *Provenance survived nothing.*  Entries used to accumulate in memory
    and be published once, after every archive, and the verified archive
    was normally deleted before that.  A kill between the directory
    install and the final write lost the only record of where those
    tiles came from -- and the rerun classified the directory valid and
    returned before it could rebuild the entry.  Publishing per archive
    means the receipt is never further behind the disk than one archive.

    *Concurrent fetches lost each other's updates.*  Two processes each
    loaded the manifest at startup and each replaced it with its own
    version, so whichever finished last dropped the other's datasets
    while both directories sat installed and unrecorded.  Re-reading the
    canonical manifest here, under the geog-root lock, merges instead.
    """

    with fetch_guard.hold("fetch-geog", root):
        manifest = _load_manifest(root)
        manifest["archives"][filename] = entry
        return _write_manifest(root, manifest)


PRODUCED_PROVENANCE_NAME = "PROVENANCE.json"
PRODUCED_PROVENANCE_SCHEMA = "arwen.geog-produced-dataset.v1"


def produce_dataset(root: Path, dataset: str, *, progress=print):
    """Build a :data:`GEOG_PRODUCERS` dataset into ``root``.

    The native reader (``rw_zarr <command>``) reads the public source,
    reduces it and writes the WPS tiles and index; this function only
    writes its request, relays its progress, records its receipt with the
    source's licence and attribution in ``<dataset>/PROVENANCE.json`` (the
    WPS readers open tiles by name, so the extra file is inert), and
    moves the finished directory into place in one rename.  Returns the
    ``(manifest key, entry, None, label, 1)`` tuple the fetch engine
    publishes, so the dataset is receipted and re-validated exactly like
    a downloaded one.
    """

    import subprocess

    from woof.zarr_bridge import resolve_zarr_bin

    archive = archive_for(dataset)
    producer = GEOG_PRODUCERS[archive.producer]
    work = root / ARCHIVE_SUBDIR / f"{dataset}.produce"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    request_path = work / "request.json"
    request_path.write_text(json.dumps(producer.request, indent=2,
                                       sort_keys=True) + "\n",
                            encoding="utf-8")
    built = work / dataset
    executable = resolve_zarr_bin()
    progress(f"fetch-geog {dataset}: building from {producer.source}; "
             f"{producer.licence}")
    started = time.monotonic()
    process = subprocess.Popen(
        [os.fspath(executable), producer.command, os.fspath(request_path),
         os.fspath(built)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    tail: list[str] = []
    assert process.stderr is not None and process.stdout is not None
    for line in process.stderr:
        line = line.rstrip()
        tail = (tail + [line])[-20:]
        progress(f"fetch-geog {dataset}: {line}")
    stdout = process.stdout.read()
    if process.wait():
        raise GeogFetchError(f"building {dataset} failed: "
                             + " | ".join(tail))
    receipt = json.loads(stdout)
    if receipt.get("schema") != "arwen.geog-soiltemp-climatology.v1":
        raise GeogFetchError(f"building {dataset}: the native reader "
                             "returned an unknown receipt schema")
    produced_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tiles = {path.name: sha256_file(path) for path in sorted(built.iterdir())}
    provenance = {
        "schema": PRODUCED_PROVENANCE_SCHEMA, "dataset": dataset,
        "producer": archive.producer, "source": producer.source,
        "licence": producer.licence, "attribution": producer.attribution,
        "request": producer.request, "receipt": receipt,
        "sha256": tiles, "produced_at": produced_at,
        "wall_s": round(time.monotonic() - started, 1),
    }
    (built / PRODUCED_PROVENANCE_NAME).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    target = root / dataset
    if target.exists():
        _quarantine(target, progress, dataset)
    os.replace(built, target)
    shutil.rmtree(work, ignore_errors=True)
    files, total = dataset_corpus(root, dataset)
    out = receipt.get("output", {})
    progress(f"fetch-geog {dataset}: built in {provenance['wall_s']} s, "
             f"{out.get('land_cells')} land cells, "
             f"{out.get('min_k', 0):.2f}..{out.get('max_k', 0):.2f} K")
    entry = {
        "producer": archive.producer, "source": producer.source,
        "licence": producer.licence, "attribution": producer.attribution,
        "request": producer.request, "sha256": tiles,
        "datasets": {dataset: {"files": files, "bytes": total}},
        "fetched_at": produced_at,
    }
    return f"produced:{dataset}", entry, None, dataset, 1


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

def _print_listing(datasets: tuple[str, ...], source: str,
                   root: Path, *, explain: bool = False) -> None:
    # The total is what a fetch of this root would DOWNLOAD, so it sums
    # the datasets marked "needed" and nothing else -- the same skip
    # decision fetch_geog takes.  It used to sum every requested dataset
    # whether staged or not, so a fully staged root listed seven "staged"
    # rows and then ended "total 1.21 GiB download": a figure for a
    # download that would not happen.
    total_archive = 0
    total_extracted = 0
    needed = 0
    print(f"fetch-geog: root {root}")
    print(f"fetch-geog: source {source} "
          f"({archive_url('<archive>', source)})")
    for name in datasets:
        archive = archive_for(name)
        ok, _ = validate_dataset_dir(root, name)
        if not ok:
            needed += 1
            total_archive += archive.archive_bytes
            total_extracted += archive.extracted_bytes
        state = "staged" if ok else "needed"
        if archive.producer:
            producer = GEOG_PRODUCERS[archive.producer]
            print(f"  {state}  {name}: built from {producer.source}, "
                  f"~{archive.archive_bytes / (1024 * 1024):.0f} MiB read; "
                  f"{producer.licence}")
            continue
        # The host the fetch itself will use (stage_one's choice), so the
        # plan names the URL the download will actually open.
        actual_source = archive.download_source or (
            source if source in archive.available_sources
            else archive.available_sources[0])
        source_note = (f"; {actual_source} only, absent from mirror"
                       if actual_source != source else f"; source {source}")
        print(f"  {state}  {name}: {archive.filename} "
              f"{archive.archive_bytes / (1024 * 1024):.1f} MiB "
              f"(~{archive.extracted_bytes / (1024 * 1024):.0f} MiB "
              f"unpacked){source_note}")
        if archive.licence_hold:
            print(f"           held out of the default geography: "
                  f"{archive.licence_hold}")
        if explain:
            needed_by = ", ".join(archive.required_by) or "no door by default"
            optional = (f"; optional for {', '.join(archive.optional_for)}"
                        if archive.optional_for else "")
            print(f"           required by: {needed_by}{optional}"
                  f"; from {archive_url(archive.filename, actual_source)}")
    if not needed:
        print(f"fetch-geog: all {len(datasets)} requested dataset(s) are "
              "staged; nothing to download")
        return
    print(f"fetch-geog: total {total_archive / (1024 ** 3):.2f} GiB "
          f"download, ~{total_extracted / (1024 ** 3):.1f} GiB unpacked "
          f"for the {needed} of {len(datasets)} dataset(s) still needed; "
          f"bundle alternative {MANDATORY_BUNDLE_BYTES / (1024 ** 3):.2f} "
          "GiB download (--bundle, NCAR only)")


def _human_size(n: int) -> str:
    """``4096`` -> ``'4.0 KiB'``; adaptive binary units, one decimal."""
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")  # pragma: no cover


def _bytes_label(n: int) -> str:
    """Exact bytes first (they match Content-Length sums), human after."""
    exact = f"{n:,} B"
    return exact if n < 1024 else f"{exact} (~{_human_size(n)})"


def fetch_geog(*, root: Path, datasets: tuple[str, ...], source: str,
               bundle: bool = False, keep_archives: bool = False,
               allow_drift: bool = False, progress=print,
               urlopen_fn=_default_urlopen) -> int:
    """Stage ``datasets`` under ``root``; the fetch-geog engine.

    Returns the count of datasets newly staged (0 = everything was
    already present and valid).

    One writer per geog root: the classification, the archive
    downloads, the extractions and every manifest publication run under
    an exclusive OS lock, so two concurrent ``fetch-geog`` runs cannot
    quarantine each other's freshly installed directories or race on the
    same archive path.
    """

    root.mkdir(parents=True, exist_ok=True)
    with fetch_guard.hold("fetch-geog", root, progress=progress):
        return _fetch_geog_locked(
            root=root, datasets=datasets, source=source, bundle=bundle,
            keep_archives=keep_archives, allow_drift=allow_drift,
            progress=progress, urlopen_fn=urlopen_fn)


def _fetch_geog_locked(*, root: Path, datasets: tuple[str, ...], source: str,
                       bundle: bool, keep_archives: bool, allow_drift: bool,
                       progress, urlopen_fn) -> int:
    """The fetch-geog engine, with the geog-root lock already held."""

    needed: list[str] = []
    for name in datasets:
        ok, detail = validate_dataset_dir(root, name)
        if ok:
            progress(f"fetch-geog {name}: {detail} -- skipped")
        else:
            needed.append(name)
    if not needed:
        progress(f"fetch-geog: all {len(datasets)} requested datasets "
                 f"already staged and valid at {root}")
        return 0

    for name in needed:
        if archive_for(name).licence_hold:
            progress(f"fetch-geog {name}: held out of the default "
                     f"geography -- {archive_for(name).licence_hold}")
    produced = tuple(name for name in needed if archive_for(name).producer)
    if bundle:
        outside = sorted(set(needed) - MANDATORY_BUNDLE_DATASETS
                         - set(produced))
        if outside:
            raise ValueError(
                f"--bundle cannot serve {', '.join(outside)}: the "
                "mandatory bundle does not contain them; use the "
                "per-dataset route (drop --bundle)")

    # The whole bill, announced before the first byte moves.  `woof
    # setup --with-geog` promises "the size is printed before anything
    # downloads", and the engine both entry points share is the only
    # place that can keep that promise for `woof fetch-geog` too --
    # the first field run watched 16 GB arrive with sizes printed only
    # as each dataset landed.  Covers exactly the datasets that still
    # need staging (the ones above were skipped as already valid);
    # under --bundle the download is the one bundle archive however few
    # datasets are extracted from it.
    download = ((MANDATORY_BUNDLE_BYTES
                 + sum(archive_for(name).archive_bytes for name in produced))
                if bundle else
                sum(archive_for(name).archive_bytes for name in needed))
    unpacked = sum(archive_for(name).extracted_bytes for name in needed)
    progress(f"fetch-geog: {len(needed)} of {len(datasets)} requested "
             f"datasets need staging: {_bytes_label(download)} download, "
             f"~{_human_size(unpacked)} unpacked")

    archive_dir = root / ARCHIVE_SUBDIR
    archive_dir.mkdir(parents=True, exist_ok=True)
    staged = 0

    def run_archive(filename: str, expected_bytes: int,
                    expected_sha256: str, targets: tuple[str, ...],
                    label: str, actual_source: str = source):
        url = archive_url(filename, actual_source)
        dest = archive_dir / filename
        if dest.exists() and dest.stat().st_size == expected_bytes \
                and sha256_file(dest) == expected_sha256:
            progress(f"fetch-geog {label}: {filename} already downloaded "
                     "and pin-verified -- reusing")
            observed, pinned = expected_sha256, True
        else:
            record = download_archive(url, dest, expected_bytes=expected_bytes,
                             progress=progress, label=label,
                             strict_size=not (allow_drift
                                              and actual_source == "ncar"),
                             urlopen_fn=urlopen_fn,
                             expected_sha256=expected_sha256)
            if record is not None:
                observed, pinned = record["sha256"], True
            else:
                observed, pinned = verify_archive(
                dest, expected_sha256=expected_sha256,
                expected_bytes=expected_bytes, source=actual_source,
                allow_drift=allow_drift, progress=progress, label=label)
        extract = (extract_datasets if urlopen_fn is _default_urlopen
                   else _extract_datasets_serial)
        counts = extract(dest, root, targets,
                                  progress=progress, label=label)
        # Published NOW, before the verified archive is removed: the
        # receipt is the only record of where these tiles came from, and
        # holding it in memory until every archive finished meant a kill
        # in between lost it with nothing able to rebuild it.
        entry = {
            "url": url, "source": actual_source,
            "archive_bytes": dest.stat().st_size,
            "archive_sha256": observed, "pinned": pinned,
            "datasets": {target: counts[target] for target in targets},
            "fetched_at": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
        }
        return filename, entry, dest, label, len(targets)

    def publish_finished(result):
        nonlocal staged
        filename, entry, dest, label, count = result
        # The main thread owns the re-entrant root lock. Workers never reacquire it.
        publish_archive_entry(root, filename, entry)
        staged += count
        if dest is None:
            return
        if keep_archives:
            progress(f"fetch-geog {label}: kept archive {dest}")
        else:
            dest.unlink()
            progress(f"fetch-geog {label}: removed verified archive "
                     f"{dest.name} (--keep-archives retains it)")

    if bundle:
        for name in produced:
            publish_finished(produce_dataset(root, name, progress=progress))
        downloaded = tuple(name for name in needed if name not in produced)
        if downloaded:
            publish_finished(run_archive(
                MANDATORY_BUNDLE_FILENAME, MANDATORY_BUNDLE_BYTES,
                MANDATORY_BUNDLE_SHA256, downloaded, "bundle"))
    else:
        def stage_one(name):
            archive = archive_for(name)
            if archive.producer:
                return produce_dataset(root, name, progress=progress)
            actual_source = (source if source in archive.available_sources
                             else archive.available_sources[0])
            if actual_source != source:
                progress(f"fetch-geog {name}: unavailable on {source}; "
                         f"using {actual_source}")
            return run_archive(archive.filename, archive.archive_bytes,
                        archive.archive_sha256, (name,), name, actual_source)
        # Four codecs and four bounded network pools. No unbounded CPU fanout.
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(stage_one, name) for name in needed]
            failure = None
            for future in as_completed(futures):
                try:
                    publish_finished(future.result())
                except Exception as error:
                    failure = failure or error
            if failure is not None:
                raise failure

    failures = []
    for name in datasets:
        ok, detail = validate_dataset_dir(root, name)
        if not ok:
            failures.append(detail)
    if failures:
        raise GeogFetchError(
            "post-stage validation failed:\n  " + "\n  ".join(failures))
    progress(f"fetch-geog: {staged} dataset(s) staged; all "
             f"{len(datasets)} requested datasets valid at {root} "
             "(each carries its parsing WPS `index`); `woof doctor` "
             "re-checks the full estate")
    return staged


def archive_names_for_paths(required) -> set[str]:
    """Resolve dataset children using the archive table's container metadata.

    Defect: a static reader asks for a child dataset directory, while the
    fetcher stages its whole container. Comparing those names directly refused
    a supported full-physics recipe before any input preparation.
    """
    owners = {archive.dataset: archive.dataset for archive in GEOG_ARCHIVES}
    for archive in GEOG_ARCHIVES:
        for child in archive.index_subdirs:
            owners[f"{archive.dataset}/{child}"] = archive.dataset
    missing = set(required) - set(owners)
    if missing:
        raise GeogFetchError(f"selected datasets {sorted(missing)} have no archive pins; fetching another resolution would change the static fields")
    return {owners[name] for name in required}


def geography_for_run(config_path: Path) -> dict:
    """Read the same selectors and grid applicability as the static builder.

    A complete published source supplies the unchanged default legend metadata.
    Partial sources retain the entire baseline because the builder builds it all.
    """
    from woof.case_data import optional_case_data_from_tables
    from woof.experiment import load_experiment
    from woof.static.build import GeogSelection, _DEFAULT_GEOG_DIRS, _BASE_STATIC_FIELDS
    from woof.static.external_source import (off_grid_tokens, sampling_window,
                                             served_names)
    from woof.static.projection import grids_from_wps_namelist, grids_from_projection_config
    from woof.static.highres_production import resolve_static_highres
    from woof.wps_domain_ids import domain_ids_from_wps_text
    from types import SimpleNamespace
    import tomllib

    config_path = Path(config_path).resolve()
    raw = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))
    data = optional_case_data_from_tables(raw, source=str(config_path), base_dir=config_path.parent)
    fetch = raw.get("fetch", {})
    carrier = resolve_static_highres(raw, source=str(fetch.get("source") or "configuration"), base_dir=config_path.parent)
    experiment = load_experiment(config_path)
    # `woof go` hands its preparation the WPS twin beside the configuration
    # when one is there (woof.companion_domains.configuration_wps_namelist),
    # so its geog_data_res is what the run builds.  Defect: this reader
    # ignored the twin and named the default datasets, so fetch-geog
    # --config staged the wrong land use and LAI for an imported
    # operational namelist (modis_15s, modis_lai) and go's preflight passed
    # a tree its preparation then could not build from.
    beside = config_path.with_name(f"{config_path.stem}.namelist.wps")
    if data is None and beside.is_file():
        data = SimpleNamespace(geog_root=Path("."), wps_namelist=beside,
                               static_highres=carrier)
    if data is not None:
        data = SimpleNamespace(geog_root=data.geog_root, wps_namelist=data.wps_namelist,
                               static_highres=carrier)
        grids = grids_from_wps_namelist(data.wps_namelist)
        ids = domain_ids_from_wps_text(Path(data.wps_namelist).read_text(encoding="utf-8-sig"), len(grids))
    else:
        # The run door emits default WPS selectors for a declarative experiment.
        grids = grids_from_projection_config(experiment)
        ids = [domain.grid_id for domain in experiment.domains]
    domains = {domain.grid_id: domain for domain in experiment.domains}
    required, statics = set(), set()
    for domain_id, grid in zip(ids, grids):
        if data is not None:
            selection = GeogSelection.from_case_data(data, domain_id=domain_id)
        else:
            from dataclasses import replace
            from woof.static.external_source import static_source_for
            selection = replace(GeogSelection.fallback(Path(".")), static_source=static_source_for(carrier))
        source = selection.static_source
        served = frozenset()
        if source is not None:
            if sampling_window(source.row, grid) is not None:
                served = served_names(source)
                statics.add(source.row.id)
            else:
                # A native source's child can rebuild from different WPS
                # datasets. Resolve its row's recipe exactly as build_static
                # does, before fetch-geog stages or go checks those datasets.
                selection = selection.with_off_grid_tokens(
                    off_grid_tokens(source, grid))
        if _BASE_STATIC_FIELDS <= served:
            from woof.static.geog import DEFAULT_LANDUSE_ATTRS as _DEFAULT_LANDUSE_ATTRS
            attrs = source.row.attrs_map
            keys = ("MMINLU", "ISWATER", "ISLAKE", "ISICE", "ISURBAN", "NUM_LAND_CAT")
            if (selection.landuse != _DEFAULT_GEOG_DIRS["landuse"]
                    or any(attrs.get(key) != _DEFAULT_LANDUSE_ATTRS[key] for key in keys)):
                required.add(selection.landuse)
        else:
            # A grid the static source does not describe rebuilds its statics
            # from the source's own off-grid datasets (woof.static.build.
            # _with_off_grid_recipe), so the plan names those.  Defect
            # (2026-10-24, WOOF HRRR 3 km/1 km runs): this plan named the
            # default inventory, go's preflight passed, and preparation then
            # refused at the 1 km nest for bnu_soiltype_top, bnu_soiltype_bot
            # and lai_modis_30s, which `fetch-geog --config` never staged.
            rebuilt = selection
            if source is not None and selection.resolution_tokens == ("default",):
                from woof.static.external_source import off_grid_tokens
                tokens = off_grid_tokens(source, grid)
                if tokens:
                    rebuilt = GeogSelection.from_tokens(selection.root, tokens)
            required.update(getattr(rebuilt, role) for role in _DEFAULT_GEOG_DIRS)
        cfg = domains[domain_id].run
        if getattr(cfg, "sf_lake_physics", 0) and "LAKE_DEPTH" not in served:
            required.add("lake_depth")
        from woof.static.orographic import required_static_fields, orographic_paths
        drag = tuple(name for name in required_static_fields(
            getattr(cfg, "topo_wind", 0), getattr(cfg, "gwd_opt", 0)) if name not in served)
        if drag:
            paths = orographic_paths(drag, selection.root, selection.resolution_tokens)
            required.update(path.relative_to(selection.root).as_posix() for path in paths.values())
            required.add(selection.landuse)
    required = archive_names_for_paths(required)
    datasets = [a.dataset for a in GEOG_ARCHIVES if a.dataset in required]
    from woof.static.external_source import static_source_row
    return {"datasets": datasets, "static_sources": sorted(statics),
            "archive_bytes": sum(archive_for(name).archive_bytes for name in datasets),
            "static_bytes": sum(static_source_row(name).bytes for name in statics)}


def _print_only(args) -> bool:
    """``--list`` and ``--explain`` print the plan and touch nothing.

    Defect (hex global run, 2026-10-16): ``--explain`` is the CLI-wide
    "say why" modifier, and fetch-geog ignored it, so ``woof fetch-geog
    --explain`` started a 13 GB download for a reader who asked what it
    would do.  On this command both flags mean: print, no network, no
    writes -- on the --config, --static-source and dataset paths alike.
    """
    from woof.explain import explain_enabled

    return bool(getattr(args, "list", False)) or explain_enabled(args)


def fetch_geog_main(args) -> int:
    from woof.explain import explain_enabled

    print_only = _print_only(args)
    config_path = getattr(args, "config", None)
    if config_path is not None:
        plan = geography_for_run(Path(config_path))
        root = Path(args.root) if args.root is not None else config_geog_root(Path(config_path))
        if print_only:
            print(json.dumps(plan, indent=2))
            return 0
        root.mkdir(parents=True, exist_ok=True)
        from woof.static.external_source import fetch_static_source
        for source_id in plan["static_sources"]:
            fetch_static_source(source_id, root)
        fetch_geog(root=root, datasets=tuple(plan["datasets"]),
                   source=resolve_source(args.source, args.bundle),
                   bundle=args.bundle, keep_archives=args.keep_archives,
                   allow_drift=args.allow_upstream_drift)
        return 0
    static_source = getattr(args, "static_source", None)
    if static_source is not None:
        # A published static file a configuration names with [static]
        # source (woof.static.external_source), staged beside the
        # datasets under <root>/static_sources/<id>/.
        from woof.static.external_source import fetch_static_source
        root = Path(args.root) if args.root is not None else default_geog_root()
        if print_only:
            from woof.static.external_source import static_source_row
            row = static_source_row(static_source)
            print(f"fetch-geog: would stage static source {row.id} "
                  f"({row.filename}, {row.bytes:,} B) from {row.url} "
                  f"into {root / 'static_sources' / row.id}; nothing fetched")
            return 0
        fetch_static_source(static_source, root)
        return 0
    datasets = parse_datasets(args.datasets)
    source = resolve_source(args.source, args.bundle)
    if args.root is not None:
        root = Path(args.root)
    else:
        root = default_geog_root()
        print(f"fetch-geog: --root not given; staging into {root} "
              "(the $GPUWM_CASE_DATA_ROOT/WPS_GEOG default every other "
              "command reads)")
    if print_only:
        _print_listing(datasets, source, root, explain=explain_enabled(args))
        if explain_enabled(args):
            print("fetch-geog: --explain prints the plan only; nothing was "
                  "downloaded or written.  Run the same command without "
                  "--explain to stage it")
        return 0
    fetch_geog(root=root, datasets=datasets, source=source,
               bundle=args.bundle, keep_archives=args.keep_archives,
               allow_drift=args.allow_upstream_drift)
    return 0


def register_cli(subparsers) -> None:
    # Both totals are read off the table.  The unpacked figure used to be
    # the literal "~16 GiB", which stopped being true the moment a pin was
    # added and left the command understating its own download by most of
    # its size.
    defaults = tuple(a for a in GEOG_ARCHIVES if a.fetch_by_default)
    total_gib = sum(a.archive_bytes for a in defaults) / (1024 ** 3)
    unpacked_gib = sum(a.extracted_bytes for a in defaults) / (1024 ** 3)
    parser = subparsers.add_parser(
        "fetch-geog",
        help=f"download and stage the {len(defaults)} default WPS_GEOG static "
             f"datasets (~{total_gib:.1f} GiB download, "
             f"~{unpacked_gib:.0f} GiB unpacked), SHA-256-verified against "
             "the packaged pins, resumable, idempotent when everything is "
             "already staged")
    parser.add_argument(
        "--config", type=Path, default=None, metavar="FILE",
        help="resolve only the geography this experiment's domains read; includes selected published static sources")
    parser.add_argument(
        "--root", type=Path, default=None, metavar="DIR",
        help="geog root to stage into (default: "
             "$GPUWM_CASE_DATA_ROOT/WPS_GEOG, exactly what woof doctor "
             "checks and wizard configs reference)")
    wrf_count = len(datasets_required_by(GEOG_CONSUMER_WRF))
    mesh_gib = sum(archive_for(name).extracted_bytes
                   for name in datasets_required_by(GEOG_CONSUMER_MESH)
                   if GEOG_CONSUMER_WRF not in archive_for(name).required_by)
    parser.add_argument(
        "--datasets", default="default", metavar="default|all|CONSUMER|NAME,NAME",
        help="which datasets to stage (default keeps the WRF and mesh set; "
             "chem-dust opts into the three upstream dust datasets; all selects every pin). A consumer name stands for one "
             f"door's whole set: '{GEOG_CONSUMER_WRF}' is the {wrf_count} "
             f"the WRF static builder opens, '{GEOG_CONSUMER_MESH}' is what "
             "woof mesh needs for the static half of its pair.  Use "
             f"'--datasets {GEOG_CONSUMER_WRF}' to skip the "
             f"~{mesh_gib / (1024 ** 3):.1f} GiB of land-use pins that "
             "only woof mesh reads")
    parser.add_argument(
        "--source", default=None, choices=GEOG_SOURCES,
        help="download host: 'ncar' (default) is the upstream NCAR "
             "server (no upstream checksums; the packaged pins are "
             "enforced); 'hf' is a byte-for-byte mirror of the same "
             "tarballs on Hugging Face (CDN bandwidth, pinned bytes)")
    parser.add_argument(
        "--bundle", action="store_true",
        help="fetch NCAR's single geog_high_res_mandatory.tar.gz "
             f"({MANDATORY_BUNDLE_BYTES / (1024 ** 3):.1f} GiB) instead "
             "of the per-dataset tarballs and extract the requested "
             "datasets from it (fallback; NCAR only)")
    parser.add_argument(
        "--keep-archives", action="store_true",
        help="keep the verified tarballs under <root>/"
             f"{ARCHIVE_SUBDIR} after extraction (default: remove each "
             "one after its datasets validate)")
    parser.add_argument(
        "--allow-upstream-drift", action="store_true",
        help="legacy spelling; refused by the native production route "
             "because unpinned geography changes prepared fields under "
             "the same dataset version")
    parser.add_argument(
        "--static-source", default=None, metavar="ID",
        help="stage only the published static file of this static-source "
             "row (woof/data/static_sources/static-sources.v1.toml), the "
             "file a configuration names with [static] source; verified "
             "against the row's size and SHA-256 and staged under "
             "<root>/static_sources/<ID>/")
    parser.add_argument(
        "--list", action="store_true",
        help="print the dataset/size/source table and per-dataset "
             "staged state, then exit without touching the network "
             "(--explain does the same and adds which door needs each "
             "dataset and its URL)")
    parser.set_defaults(func=fetch_geog_main)
    return parser


__all__ = [
    "ARCHIVE_SUBDIR", "DRIFT_SIZE_BAND", "GEOG_ARCHIVES",
    "GEOG_CONSUMERS", "GEOG_CONSUMER_MESH", "GEOG_CONSUMER_WRF", "GEOG_CONSUMER_CHEM_DUST",
    "GEOG_FETCH_MANIFEST_NAME", "GEOG_FETCH_MANIFEST_SCHEMA",
    "GEOG_SOURCES", "GEOG_URL_BASE_ENV", "GeogArchive", "GeogFetchError",
    "HF_MIRROR_BASE_URL", "HF_MIRROR_REPO", "MANDATORY_BUNDLE_BYTES",
    "MANDATORY_BUNDLE_DATASETS", "MANDATORY_BUNDLE_FILENAME",
    "MANDATORY_BUNDLE_SHA256", "NCAR_BASE_URL", "archive_for",
    "archive_url", "datasets_required_by", "default_geog_root", "default_geog_consumers",
    "download_archive", "extract_datasets", "fetch_geog", "fetch_geog_main",
    "geog_datasets", "packaged_recipe_datasets", "parse_datasets",
    "register_cli", "resolve_source",
    "sha256_file", "size_phrase", "validate_dataset_dir",
    "WRF_FETCH_COMMAND",
]
