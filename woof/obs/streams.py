"""The observation streams of the data assimilation (moved into the engine
from woof global ``arwen_global.obs_streams`` at b54373992587; the door rows it
needs are the engine's, :data:`woof.obs.frontdoor.STREAM_DOORS`): what each is,
which Rust front door fetches and decodes it, how far behind real time it
arrives and in which latency class that puts it, how an hour's tables are
cut for the analysis under an information cutoff, and how the volumes are
recorded.

Design item 2 of the Arwen Global DA program (2026-09-06, with the design
amendments of 01:10 UTC).  Everything on the data path is Rust: the
fetchers, the decoders (GRIB through the mapped engine, NetCDF-4/HDF5
through netcrust and hdf5-reader, the fixed-width and CSV archives through
the ``rw-obs`` front doors) and the regridding.  This module is
orchestration: it names the streams, drives the binaries through
:mod:`woof.obs.frontdoor`, cuts the decoded tables into analysis windows,
applies the information cutoff, thins to a stated grid when asked, and
writes the manifests.  Python decodes nothing here.

The interface every other lane reads
------------------------------------
* **The table.**  ``gpuwm-obs.table.v2``
  (:data:`woof.obs.neutral_table.TABLE_HEADER`): one CSV row per
  observation, the ten v1 columns ``source, station_id, latitude_deg,
  longitude_deg, elevation_m, level_pa, valid_time, variable, value,
  error`` and the five bookkeeping columns ``measurement, nominal_time,
  published_time, received_time, revision``.  A surface row has an empty
  ``level_pa`` and is anchored at ``elevation_m``; an aloft row carries
  its pressure; a ``refractivity_n`` row is anchored at ``elevation_m``
  (the tangent height) and carries the retrieval's dry pressure in
  ``level_pa`` for the column gates only.  Times are ISO-8601 UTC with a
  ``Z``.  :func:`woof.obs.neutral_table.load_obs` reads v2 and v1
  files (the neutral header is recognised before the declared source
  tables), so the engine's local data assimilation, woof global's cycle door and the
  ensemble filter's ``batches_from_rows`` take these files unchanged.
* **Observations at their own times.**  Every row keeps the instant it
  was measured (a METAR at 12:08 is a 12:08 row; a radiosonde level is at
  its release-plus-elapsed time with the nominal hour beside it in
  ``nominal_time``; a motion vector at its image mid-point).  An hourly
  cycle is a window, not an instant: :func:`cut_hours` writes the rows of
  ``(t - cycle/2, t + cycle/2]`` per analysis instant and never moves a
  row's time; the filter compares each report with the trajectory at the
  report's own time (the ensemble lane's contract).
* **The information cutoff.**  ``fresh`` returns the analysis constructible
  from information available by a declared cutoff, so
  :func:`apply_information_cutoff` keeps a row only when its
  ``received_time`` is at or before the cutoff; a row without a receipt
  time (decoded from files whose arrival was not recorded) is kept and
  counted as ``latency_unverified``, and the hours manifest says how many
  such rows each hour carries.  A historical case fetched days later is
  therefore labelled, never passed off as a real-time replay.
* **The variables** are :data:`woof.obs.neutral_table.VARIABLE_TABLE`:
  ``surface_pressure_pa``, ``temperature_k``, ``dewpoint_k``,
  ``wind_u_m_s``, ``wind_v_m_s``, ``refractivity_n``.  A stream that wants
  a new one adds it there, with units and gross bounds, before any row is
  written; the ``measurement`` column
  (:data:`woof.obs.neutral_table.MEASUREMENT_TABLE`) says which
  physical quantity a row is (a station pressure recovered from the
  altimeter setting, a sea-level pressure, a buoy wind reduced from 5 m).
* **The errors** are in every row (the Rust doors state them per stream
  and variable in their records; :data:`STREAMS` repeats them here).
* **Latency classes.**  Every stream carries a measured latency behind
  real time with its basis and a class (:data:`LATENCY_CLASSES`): ``fast``
  streams feed the hourly cycle, ``replay`` streams feed the delayed
  replay whose lag follows the measured latency, ``retrospective``
  streams exist only for past cases, ``gated`` sources need an account
  and are reported, never fetched.  The numbers are in the entries below
  with the date they were measured; :func:`stream_manifest` carries the
  measurement of each fetch.
* **Thinning.**  :func:`thin_to_grid` keeps one row per (source,
  variable, latitude band, longitude band, pressure layer) per window,
  the row nearest the analysis instant; the bands are the model grid's
  own spacing and the layer is ``ln p`` at ``layer_ln_p`` (default 0.1).
  It is a pre-thinning for a stated grid and is counted, never silent;
  the ensemble filter thins again to its own cell at analysis time (the
  ensemble lane's quality-control order), and the AMV door has already
  kept one vector per 0.5 degree box and 50 hPa layer.
* **The external analysis is not a stream.**  The GDAS or IFS analysis is
  a weak low-pass background constraint on the largest scales
  (:data:`ANCHOR_SOURCES`), because an external analysis already contains
  many of the observations assimilated here and offering it as millions of
  independent pseudo-reports would double-count them.  This module records
  the source, its cycle, its measured availability latency and the
  mapping row that reads it; the analysis lanes implement the constraint
  and record its increment separately.  It constrains selected scales only
  and is never claimed to make the cycle unable to drift.
* **The manifest** of a fetched stream (``gpuwm-obs.stream.v2``) wraps the
  door's own fetch and table records: URLs, bytes, SHA-256 per source
  file, rows per variable and per hour, the QC counters, and the latency
  behind real time with its basis and class.

Streams (:data:`STREAMS`)
-------------------------
``iem-metar``  worldwide METAR through the Iowa Environmental Mesonet ASOS
archive (266 networks, hourly), ``rw_asos networks``, ``rw_asos fetch
--networks`` and ``rw_asos table``: station pressure from the altimeter
setting, 2 m temperature and dewpoint, 10 m wind.  ``igra2`` the NCEI
IGRA2 year-to-date archive, ``rw_igra2``: every pressure level's
temperature, dewpoint and wind at its own time and the surface level's
pressure.  ``goes-dmw`` GOES-18 and GOES-19 ABI L2 derived motion winds,
``rw_amv``: u and v at the assigned pressure, DQF gated, zenith screened,
thinned.  ``ndbc`` NDBC buoys and coastal stations, ``rw_ndbc``.
``cdaac-ro`` the UCAR CDAAC near-real-time occultation retrievals
(COSMIC-2, PAZ, KOMPSAT-5), ``rw_gnssro cdaac-fetch`` and ``table
--tarballs``: one daily tarball per mission, anonymous, published about
five hours after the day it covers ends (the delayed-replay class), the
refractivity at tangent heights with the retrieval's dry pressure for the
ln p metric; ``gnss-ro`` the AWS ``gnss-ro-data`` archive, ``rw_gnssro
list/fetch/table --files``: the same rows for past cases (the bucket holds
nothing after 2025-07-29).  ``wis2`` the WMO Information System 2 global
broker and caches, ``rw_wis2``: the notification stream subscribed over
MQTT, every original message archived on receipt, the coverage and
latency measured; the BUFR payloads are archived and counted and are not
yet decoded into the table (a named open item).  ``awc-metar`` the
Aviation Weather Center METAR cache as the complementary surface source,
``rw_asos awc``.  ``madis-acars``, ``madis-raob`` and ``madis-mesonet`` the
NOAA MADIS public files (commercial aircraft, radiosondes worldwide, the
public part of the mesonet collection), ``rw_madis fetch/table``: anonymous,
from the real-time directory (about five and a half days, each hour file
rewritten while its reports arrive) and the dated archive behind it, MADIS's
own QC summary applied and GSI's regional error table per row.  An earlier
version of this table called MADIS account-gated; the public files never
were (the account covers the restricted providers, which are absent from the
public files by MADIS policy).

No stream here needs an account.  Reachable but not decoded here: the NWS
upper-air text feed (FM-35 TEMP parts, US sites, about three minutes
behind the nominal hour); IGRA2 covers the same soundings a day later and
is the sounding stream of record until a TEMP decoder exists.

The door's streams (:func:`fetch_stream`) are what ``woof obs streams
fetch`` runs and what the DA door's ``--stream NAME`` fetches per window
(woof global's ``da_streams`` and the engine's local data assimilation): one function, two callers, so the
cycle and the command line cannot drift.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from woof.obs.neutral_table import (
    MEASUREMENT_TABLE,
    TABLE_HEADER,
    TABLE_SCHEMA,
    VARIABLE_TABLE,
    ObsRow,
    load_obs,
    parse_valid_time,
)

STREAM_MANIFEST_SCHEMA = "gpuwm-obs.stream.v2"
HOURS_MANIFEST_SCHEMA = "gpuwm-obs.hours.v2"
ANCHOR_RECORD_SCHEMA = "gpuwm-obs.anchor.v1"
DEFAULT_CYCLE_S = 3600
DEFAULT_LAYER_LN_P = 0.1

#: Where a stream's measured latency puts it.  The classes are the fresh
#: door's vocabulary: the fast hourly analysis takes ``fast`` streams, the
#: separately versioned delayed replay takes ``replay`` streams at a lag
#: that follows their measured latency, a ``retrospective`` stream is
#: available for past cases only, and a ``gated`` source is reported.
LATENCY_CLASSES: dict[str, str] = {
    "fast": "arrives within the hour; feeds the hourly analysis",
    "replay": "arrives hours to days late; feeds the delayed replay at its measured lag",
    "retrospective": "available for past cases only; no live route without an account",
    "gated": "needs an account; reported, never fetched",
}


@dataclass(frozen=True)
class StreamSpec:
    """One observation stream: its identity, door, stated errors and
    measured latency class."""

    name: str
    door: str | None
    subject: str
    sources: tuple[str, ...]
    variables: tuple[str, ...]
    errors: dict[str, float]
    cadence_s: int
    public: bool
    latency_class: str
    latency_basis: str
    measurements: tuple[str, ...] = ()
    account_gated: bool = False
    decoder_built: bool = True
    #: The door subscribes to a live feed and archives what arrives, which
    #: is work a user can ask for even where no decoder writes the neutral
    #: table yet.  Without this field that work is reachable only by running
    #: the binary by hand, and a binary this package pins, stages and
    #: verifies but publishes no command for is not shipped.
    subscribes: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        unknown = [v for v in self.variables if v not in VARIABLE_TABLE]
        if unknown:
            raise ValueError(
                f"stream {self.name!r} names variables outside the neutral "
                f"vocabulary: {unknown}; add them to VARIABLE_TABLE first"
            )
        if self.latency_class not in LATENCY_CLASSES:
            raise ValueError(
                f"stream {self.name!r} names latency class {self.latency_class!r}; "
                f"the classes are {sorted(LATENCY_CLASSES)}"
            )
        bad = [m for m in self.measurements if m not in MEASUREMENT_TABLE]
        if bad:
            raise ValueError(
                f"stream {self.name!r} names measurements outside MEASUREMENT_TABLE: {bad}"
            )
        if self.account_gated and self.latency_class != "gated":
            raise ValueError(f"stream {self.name!r} is account-gated and must be class 'gated'")


#: The refractivity error of both occultation routes, as the Rust door
#: writes it into every row (``rw_obs::table::refractivity_error_fraction``):
#: a fraction of N by tangent height and latitude, the Kuo et al. (2004)
#: shape.
REFRACTIVITY_ERRORS = {
    "refractivity_fraction_floor": 0.003,
    "refractivity_fraction_surface": 0.020,
    "refractivity_scale_height_m": 3000.0,
    "refractivity_tropical_factor_below_8km_inside_30deg": 1.5,
    "refractivity_fraction_per_km_above_25km": 0.0007,
}

#: Every stream this program knows, fetched or reported.  Latencies are
#: measured numbers with their date; the class follows the number.
STREAMS: dict[str, StreamSpec] = {
    "iem-metar": StreamSpec(
        name="iem-metar", door="rw_asos",
        subject="worldwide METAR through the IEM ASOS archive",
        sources=("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
                 "https://mesonet.agron.iastate.edu/geojson/networks.geojson"),
        variables=("surface_pressure_pa", "temperature_k", "dewpoint_k",
                   "wind_u_m_s", "wind_v_m_s"),
        errors={"surface_pressure_pa": 100.0, "temperature_k": 1.5,
                "dewpoint_k": 1.5, "wind_u_m_s": 2.5, "wind_v_m_s": 2.5},
        cadence_s=3600, public=True,
        latency_class="fast",
        latency_basis="fetch instant minus the latest report kept (an upper bound): "
                      "443 s at 01:01 and 137 s at 01:42 UTC on 2026-09-06 for ten networks",
        measurements=("station_pressure_from_altimeter", "screen_temperature_2m",
                      "screen_dewpoint_2m", "anemometer_wind_10m"),
        notes="266 ASOS networks; ten networks per request, paced 2 s; a 31 h "
              "global window is 25 requests, 18.6 MB, 90 s",
    ),
    "igra2": StreamSpec(
        name="igra2", door="rw_igra2",
        subject="IGRA2 radiosondes (year-to-date archive), every level at its own time",
        sources=("https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-y2d/",),
        variables=("surface_pressure_pa", "temperature_k", "dewpoint_k",
                   "wind_u_m_s", "wind_v_m_s"),
        errors={"surface_pressure_pa": 100.0, "temperature_k": 1.0,
                "dewpoint_k": 2.5, "wind_u_m_s": 2.5, "wind_v_m_s": 2.5},
        cadence_s=12 * 3600, public=True,
        latency_class="replay",
        latency_basis="archive file Last-Modified minus the latest nominal hour it holds: "
                      "77,802 to 120,998 s (21.6 to 33.6 h) over four stations on 2026-09-06",
        measurements=("station_pressure", "sonde_level"),
        notes="807 station files, 631 MB, refreshed once a day near 21:36 UTC; "
              "soundings appear about a day after their nominal hour",
    ),
    "goes-dmw": StreamSpec(
        name="goes-dmw", door="rw_amv",
        subject="GOES-18 and GOES-19 ABI L2 derived motion winds (full disk)",
        sources=("https://noaa-goes18.s3.amazonaws.com/ABI-L2-DMWF/",
                 "https://noaa-goes19.s3.amazonaws.com/ABI-L2-DMWF/"),
        variables=("wind_u_m_s", "wind_v_m_s"),
        errors={"wind_low_m_s": 3.0, "wind_mid_m_s": 4.0, "wind_high_m_s": 5.0},
        cadence_s=3600, public=True,
        latency_class="fast",
        latency_basis="bucket LastModified minus the scan end in the filename: 619 to 1,861 s, "
                      "median 921 s, over 365 granules of 2026-08-31 17Z to 09-02 01Z",
        measurements=("amv_assigned_pressure",),
        notes="six bands per satellite per hour (the visible and 3.9 um bands are "
              "empty products by night); DQF 0 only; zenith 68 deg; thinned to 0.5 "
              "deg boxes and 50 hPa layers in the door; GOES-16 publishes none",
    ),
    "ndbc": StreamSpec(
        name="ndbc", door="rw_ndbc",
        subject="NDBC moored buoys and coastal stations (realtime2 feed)",
        sources=("https://www.ndbc.noaa.gov/data/realtime2/",
                 "https://www.ndbc.noaa.gov/data/stations/station_table.txt"),
        variables=("surface_pressure_pa", "temperature_k", "dewpoint_k",
                   "wind_u_m_s", "wind_v_m_s"),
        errors={"surface_pressure_pa": 100.0, "temperature_k": 1.5,
                "dewpoint_k": 2.0, "wind_buoy_m_s": 3.0, "wind_fixed_m_s": 2.5},
        cadence_s=3600, public=True,
        latency_class="fast",
        latency_basis="file Last-Modified minus the latest report in the file: median 1,242 s, "
                      "maximum 2,135 s over 61 platforms of a live 70-platform fetch on "
                      "2026-09-06 01:42 UTC (8 platforms answered 404)",
        measurements=("sea_level_pressure", "platform_temperature", "platform_dewpoint",
                      "anemometer_wind_10m", "anemometer_wind_5m_reduced_to_10m"),
        notes="45-day rolling files, 943 platforms, 526 MB; a platform without a "
              "realtime2 file answers 404 and is a named absence in the fetch record",
    ),
    "gnss-ro": StreamSpec(
        name="gnss-ro", door="rw_gnssro",
        subject="GNSS radio-occultation refractivity (AWS gnss-ro-data archive)",
        sources=("https://gnss-ro-data.s3.amazonaws.com/contributed/v2.0/",),
        variables=("refractivity_n",),
        errors=REFRACTIVITY_ERRORS,
        cadence_s=3600, public=True,
        latency_class="retrospective",
        latency_basis="the COSMIC-2 UCAR L2a collection ends at 2025-07-29 and no 2026 object "
                      "exists in any collection (measured 2026-09-06); update cadence monthly",
        measurements=("ro_refractivity_tangent_point",),
        notes="one NetCDF-4 per occultation, about 350 kB, 3,841 levels at 10 m; the past-case "
              "route; the recent days come through cdaac-ro",
    ),
    "cdaac-ro": StreamSpec(
        name="cdaac-ro", door="rw_gnssro",
        subject="GNSS radio-occultation refractivity, the UCAR CDAAC near-real-time daily tarballs "
                "(COSMIC-2, PAZ, KOMPSAT-5)",
        sources=("https://data.cosmic.ucar.edu/gnss-ro/cosmic2/nrt/level2/",
                 "https://data.cosmic.ucar.edu/gnss-ro/paz/nrt/level2/",
                 "https://data.cosmic.ucar.edu/gnss-ro/kompsat5/nrt/level2/"),
        variables=("refractivity_n",),
        errors=REFRACTIVITY_ERRORS,
        cadence_s=24 * 3600, public=True,
        latency_class="replay",
        latency_basis="the daily tarball's Last-Modified minus the end of the day it covers: 4 h 42 min "
                      "to 5 h 05 min over the six tarballs of 2026-09-01, 09-02 and 09-05 (measured "
                      "2026-09-06); the day's first occultation therefore waits up to 29 h",
        measurements=("ro_refractivity_tangent_point",),
        notes="anonymous (the earlier note that the feed needs an account was wrong: the directory and "
              "the tarballs answer 200 with no credential); one atmPrf tarball per mission and day, "
              "COSMIC-2 6,227 occultations in 2.23 GB gzipped for 2026-09-01, PAZ 46 MB, KOMPSAT-5 9 MB; "
              "the tarball is streamed member by member, a profile with CDAAC's bad flag set is refused "
              "by name (17 percent on the case day), one row per 200 m to 30 km; the case window "
              "(2026-09-01 17:30 to 09-02 00:30) held 1,610 profiles and 237,297 rows",
    ),
    "wis2": StreamSpec(
        name="wis2", door="rw_wis2",
        subject="WMO Information System 2: the global broker's notification stream and the global caches "
                "(SYNOP, SHIP and TEMP in BUFR, decoded)",
        sources=("mqtts://globalbroker.meteo.fr:8883",
                 "mqtts://globalbroker.noaa.gov:8883"),
        variables=("surface_pressure_pa", "temperature_k", "dewpoint_k",
                   "wind_u_m_s", "wind_v_m_s"),
        errors={"surface_pressure_pa": 100.0, "temperature_surface_k": 1.5, "dewpoint_surface_k": 1.5,
                "wind_surface_m_s": 2.5, "temperature_sonde_k": 1.0, "dewpoint_sonde_k": 2.5,
                "wind_sonde_m_s": 2.5},
        cadence_s=60, public=True,
        latency_class="fast",
        latency_basis="notification pubtime minus the report's datetime (median 388 s, "
                      "min 160 s over 2,892 messages) plus receipt minus pubtime (median 158 s, "
                      "max 558 s): 546 s, measured 2026-09-06 02:03 to 02:13 UTC on "
                      "globalbroker.meteo.fr",
        measurements=("station_pressure", "sea_level_pressure", "screen_temperature_2m",
                      "screen_dewpoint_2m", "platform_temperature", "platform_dewpoint",
                      "anemometer_wind_10m", "sonde_level"),
        subscribes=True,
        notes="core surface, upper-air and marine data in BUFR under "
              "cache/a/wis2/+/data/core/weather/surface-based-observations/# (synop, temp, "
              "ship); ten minutes carried 2,920 notifications from 29 centres, 292 a minute, "
              "2,918 payloads of 1.8 MB archived with 2,891 integrity digests verified; the "
              "subscriber archives every message and payload on receipt and measures coverage "
              "per centre, and `rw_wis2 table` decodes the archive (FM 94 BUFR against the "
              "vendored WMO master tables, version 46; the SYNOP land templates, SHIP 308009 "
              "and TEMP 309052 / 309056 to rows); the global caches keep 24 hours, so a case "
              "older than a day has no WIS2 rows: the stream is live only",
    ),
    "awc-metar": StreamSpec(
        name="awc-metar", door="rw_asos",
        subject="Aviation Weather Center METAR cache (complementary surface source)",
        sources=("https://aviationweather.gov/data/cache/metars.cache.csv.gz",),
        variables=("surface_pressure_pa", "temperature_k", "dewpoint_k",
                   "wind_u_m_s", "wind_v_m_s"),
        errors={"surface_pressure_pa": 100.0, "temperature_k": 1.5,
                "dewpoint_k": 1.5, "wind_u_m_s": 2.5, "wind_v_m_s": 2.5},
        cadence_s=60, public=True,
        latency_class="fast",
        latency_basis="fetch instant minus the latest observation_time in the cache: 134 s, "
                      "the cache's Last-Modified 69 s after its latest report, on 2026-09-06 "
                      "02:07 UTC (4,925 reports, 13 corrected)",
        measurements=("station_pressure_from_altimeter", "screen_temperature_2m",
                      "screen_dewpoint_2m", "anemometer_wind_10m"),
        notes="one gzip CSV of the last hour, worldwide, rewritten every minute; "
              "`rw_asos awc` fetches and converts it",
    ),
    "madis-acars": StreamSpec(
        name="madis-acars", door="rw_madis",
        subject="MADIS commercial aircraft reports (AMDAR, ACARS, TAMDAR), public files",
        sources=("https://madis-data.ncep.noaa.gov/madisPublic1/data/point/acars/netcdf/",
                 "https://madis-data.ncep.noaa.gov/madisPublic1/data/archive/"),
        variables=("temperature_k", "dewpoint_k", "wind_u_m_s", "wind_v_m_s"),
        errors={"gsi_report_type_temperature_humidity": 133.0, "gsi_report_type_wind": 233.0},
        cadence_s=3600, public=True,
        latency_class="fast",
        latency_basis="minutes from report to its appearance in the public hour file: median 14 "
                      "over the 11-12 UTC hour of 2026-10-14 (live watch 10:44-15:05 UTC); the "
                      "07 UTC file was still being rewritten at 08:13:30 UTC on 2026-10-16",
        measurements=("aircraft_level",),
        notes="anonymous; real-time directory first (about 5.5 days), dated archive behind it "
              "(about two days late); level at the ISA pressure of the pressure altitude; MADIS QC "
              "summary V S C G K kept; errors per row from GSI's regional table (133/233); a later "
              "fetch of a growing hour file stamps its rows with the later arrival, so an "
              "information cutoff keeps them out of an analysis that ran before",
    ),
    "madis-raob": StreamSpec(
        name="madis-raob", door="rw_madis",
        subject="MADIS radiosondes (worldwide), public files",
        sources=("https://madis-data.ncep.noaa.gov/madisPublic1/data/point/raob/netcdf/",
                 "https://madis-data.ncep.noaa.gov/madisPublic1/data/archive/"),
        variables=("temperature_k", "dewpoint_k", "wind_u_m_s", "wind_v_m_s"),
        errors={"gsi_report_type_temperature_humidity": 120.0, "gsi_report_type_wind": 220.0},
        cadence_s=3600, public=True,
        latency_class="fast",
        latency_basis="minutes after the nominal hour: from 13, still arriving 128 after 12 UTC "
                      "(2026-10-14 live watch); IGRA2 carries the same soundings a day later",
        measurements=("sonde_level",),
        notes="anonymous; mandatory, significant, tropopause and maximum-wind levels with a "
              "pressure; every level dated to the release time (the MADIS file has no "
              "per-level elapsed time); an hour nobody launched is absent and counted",
    ),
    "madis-mesonet": StreamSpec(
        name="madis-mesonet", door="rw_madis",
        subject="MADIS mesonets (the public part of the LDAD collection), public files",
        sources=("https://madis-data.ncep.noaa.gov/madisPublic1/data/LDAD/mesonet/netCDF/",
                 "https://madis-data.ncep.noaa.gov/madisPublic1/data/archive/"),
        variables=("surface_pressure_pa", "temperature_k", "dewpoint_k",
                   "wind_u_m_s", "wind_v_m_s"),
        errors={"gsi_report_type_temperature_humidity": 188.0, "gsi_report_type_wind": 288.0},
        cadence_s=3600, public=True,
        latency_class="fast",
        latency_basis="minutes from report to its appearance in the public hour file: median 23 "
                      "over the 11-12 UTC hour of 2026-10-14; a 15-min window holds about 95 % of "
                      "its reports 45 min after it ends",
        measurements=("screen_temperature_2m", "screen_dewpoint_2m", "station_pressure",
                      "platform_wind"),
        notes="anonymous; five-minute reports from tens of thousands of stations (18,141 in a "
              "30-minute window on 2026-10-16); restricted providers are absent by MADIS policy; "
              "wind is platform_wind (anemometer heights vary and are not stated); "
              "--bbox S,N,W,E keeps a region",
    ),
    "nws-upper-air-text": StreamSpec(
        name="nws-upper-air-text", door=None,
        subject="NWS upper-air TEMP text (FM-35, US sites)",
        sources=("https://tgftp.nws.noaa.gov/data/observations/upperair/",),
        variables=("temperature_k", "dewpoint_k", "wind_u_m_s", "wind_v_m_s"),
        errors={}, cadence_s=12 * 3600, public=True,
        latency_class="fast",
        latency_basis="Last-Modified 00:02:46Z for the 06 Sep 00Z soundings (measured "
                      "2026-09-06): about three minutes behind the nominal hour",
        decoder_built=False,
        notes="reachable without an account; no TEMP decoder in the tree, IGRA2 is the "
              "sounding stream of record",
    ),
}


def account_gated_sources() -> list[StreamSpec]:
    """The streams that need an account: reported, never worked around."""
    return [s for s in STREAMS.values() if s.account_gated]


def fetchable_streams() -> list[StreamSpec]:
    """The streams a fetch can run today: a door and a decoder."""
    return [s for s in STREAMS.values() if s.door is not None and s.decoder_built]


# ---------------------------------------------------------------- anchors

@dataclass(frozen=True)
class AnchorSource:
    """An external analysis as a weak low-pass background constraint: the
    source, how it is read onto the model grid, and how late it becomes
    available.  Never observation rows; the analysis lanes apply it on the
    largest scales with a configured weight and record its increment
    separately, and it constrains those scales only."""

    name: str
    subject: str
    url_template: str
    mapping: str
    cadence_s: int
    availability_basis: str
    notes: str = ""

    def record(self, cycle: dt.datetime, *, published_time: dt.datetime | None,
               received_time: dt.datetime | None, path: str | None = None,
               sha256: str | None = None, bytes_: int | None = None) -> dict[str, object]:
        """The ``gpuwm-obs.anchor.v1`` record of one cycle's object: what
        the analysis carries beside its increment."""
        cycle = _utc(cycle)
        latency = (
            None if published_time is None
            else (_utc(published_time) - cycle).total_seconds()
        )
        return {
            "schema": ANCHOR_RECORD_SCHEMA,
            "anchor": self.name,
            "role": "weak low-pass background constraint on the largest scales; "
                    "never pseudo-observations; increment recorded separately",
            "mapping": self.mapping,
            "source_cycle_utc": cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "url": self.url_template.format(
                yyyymmdd=cycle.strftime("%Y%m%d"), hh=cycle.strftime("%H")
            ),
            "published_time_utc": None if published_time is None
            else _utc(published_time).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "received_time_utc": None if received_time is None
            else _utc(received_time).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "availability_latency_s": latency,
            "availability_basis": self.availability_basis,
            "path": path, "sha256": sha256, "bytes": bytes_,
        }


ANCHOR_SOURCES: dict[str, AnchorSource] = {
    "gdas": AnchorSource(
        name="gdas",
        subject="the GDAS 0.25 degree analysis (f000) from the NOAA open-data bucket",
        url_template="https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.{yyyymmdd}/{hh}/atmos/gdas.t{hh}z.pgrb2.0p25.f000",
        mapping="gdas-global",
        cadence_s=6 * 3600,
        availability_basis="the object's LastModified on the bucket minus the cycle time",
        notes="about 470 MB per cycle; the cold start's own source, read through the "
              "mapped GRIB engine",
    ),
    "ifs-open-data": AnchorSource(
        name="ifs-open-data",
        subject="the ECMWF IFS open-data step-0 object (0.25 degree)",
        url_template="https://storage.googleapis.com/ecmwf-open-data/{yyyymmdd}/{hh}z/ifs/0p25/oper/{yyyymmdd}{hh}0000-0h-oper-fc.grib2",
        mapping="ecmwf-open-data-global-forecast",
        cadence_s=12 * 3600,
        availability_basis="the object's Last-Modified on the mirror minus the cycle time",
        notes="about 135 MB per cycle; 00 and 12 UTC carry the full step set",
    ),
}


# ---------------------------------------------------------------- windows

def _utc(moment: dt.datetime) -> dt.datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=dt.timezone.utc)
    return moment.astimezone(dt.timezone.utc)


def _stamp(moment: dt.datetime | None) -> str:
    return "" if moment is None else _utc(moment).strftime("%Y-%m-%dT%H:%M:%SZ")


def analysis_window(
    analysis_time: dt.datetime, cycle_s: int = DEFAULT_CYCLE_S
) -> tuple[dt.datetime, dt.datetime]:
    """``(t - cycle/2, t + cycle/2]``: the reports one analysis takes.

    Half-open at the start so an hourly cycle never offers one report to
    two analyses; a report exactly on the boundary belongs to the later
    hour.
    """
    if cycle_s <= 0:
        raise ValueError("cycle_s must be positive")
    centre = _utc(analysis_time)
    half = dt.timedelta(seconds=cycle_s / 2.0)
    return centre - half, centre + half


def rows_in_window(
    rows: list[ObsRow], start: dt.datetime, end: dt.datetime
) -> list[ObsRow]:
    """Rows with ``start < valid_time <= end``."""
    start = _utc(start)
    end = _utc(end)
    return [r for r in rows if start < _utc(r.valid_time) <= end]


def apply_information_cutoff(
    rows: list[ObsRow], cutoff: dt.datetime | None
) -> tuple[list[ObsRow], dict[str, int]]:
    """Keep the rows available by ``cutoff``: a row whose ``received_time``
    is after the cutoff is dropped and counted; a row with no receipt time
    is kept and counted as ``latency_unverified`` (its arrival was never
    recorded, so nothing can be said about when it was available).  With
    ``cutoff`` None every row is kept and only the unverified count is
    taken."""
    kept: list[ObsRow] = []
    counters = {"offered": len(rows), "kept": 0, "after_cutoff": 0, "latency_unverified": 0}
    limit = None if cutoff is None else _utc(cutoff)
    for row in rows:
        if row.received_time is None:
            counters["latency_unverified"] += 1
        elif limit is not None and _utc(row.received_time) > limit:
            counters["after_cutoff"] += 1
            continue
        kept.append(row)
    counters["kept"] = len(kept)
    return kept, counters


# --------------------------------------------------------------- thinning

def thin_to_grid(
    rows: list[ObsRow],
    *,
    analysis_time: dt.datetime,
    nlat: int,
    nlon: int,
    layer_ln_p: float = DEFAULT_LAYER_LN_P,
) -> tuple[list[ObsRow], dict[str, object]]:
    """One row per (source, variable, cell, layer): the one nearest the
    analysis instant, ties by input order.

    The cell is the model grid's spacing in latitude (``180 / nlat``) and
    longitude (``360 / nlon``); the layer is ``floor(ln p / layer_ln_p)``
    for aloft rows and a single surface layer for surface rows.  Rows from
    different sources never compete: a buoy under a METAR station's cell
    is a second instrument, not a duplicate.  Returns the kept rows in
    input order and counters (``offered``, ``kept``, ``thinned`` and
    ``thinned_by_source``).
    """
    if nlat <= 0 or nlon <= 0:
        raise ValueError("nlat and nlon must be positive")
    if layer_ln_p <= 0.0:
        raise ValueError("layer_ln_p must be positive")
    centre = _utc(analysis_time)
    dlat = 180.0 / nlat
    dlon = 360.0 / nlon
    best: dict[tuple, tuple[float, int]] = {}
    for order, row in enumerate(rows):
        lat_band = int(math.floor((row.latitude_deg + 90.0) / dlat))
        lon_band = int(math.floor(((row.longitude_deg + 180.0) % 360.0) / dlon))
        if row.level_pa is None:
            layer = "sfc"
        else:
            layer = int(math.floor(math.log(row.level_pa) / layer_ln_p))
        key = (row.source, row.variable, lat_band, lon_band, layer)
        distance = abs((_utc(row.valid_time) - centre).total_seconds())
        held = best.get(key)
        if held is None or (distance, order) < held:
            best[key] = (distance, order)
    keep = sorted(order for _, order in best.values())
    kept = [rows[i] for i in keep]
    thinned_by_source: dict[str, int] = {}
    keep_set = set(keep)
    for order, row in enumerate(rows):
        if order not in keep_set:
            thinned_by_source[row.source] = thinned_by_source.get(row.source, 0) + 1
    counters = {
        "offered": len(rows),
        "kept": len(kept),
        "thinned": len(rows) - len(kept),
        "thinned_by_source": thinned_by_source,
        "nlat": nlat, "nlon": nlon, "layer_ln_p": layer_ln_p,
    }
    return kept, counters


# ----------------------------------------------------------------- tables

def write_table(rows: list[ObsRow], path: Path) -> dict[str, object]:
    """Write rows as a ``gpuwm-obs.table.v2`` file (every bookkeeping
    column carried through); returns its record.  Values and errors are
    written as Python floats whatever scalar the caller built the row
    from (a numpy float64's repr is ``np.float64(...)``, which the reader
    counts as malformed and drops)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(TABLE_HEADER)]
    by_variable: dict[str, int] = {}
    by_source: dict[str, int] = {}
    unverified = 0
    for row in rows:
        lon = ((row.longitude_deg + 180.0) % 360.0) - 180.0
        level = "" if row.level_pa is None else f"{row.level_pa:.1f}"
        lines.append(
            f"{row.source},{row.station_id},{row.latitude_deg:.5f},{lon:.5f},"
            f"{row.elevation_m:.1f},{level},{_stamp(row.valid_time)},{row.variable},"
            f"{float(row.value)!r},{float(row.error)!r},{row.measurement},{_stamp(row.nominal_time)},"
            f"{_stamp(row.published_time)},{_stamp(row.received_time)},{row.revision}"
        )
        by_variable[row.variable] = by_variable.get(row.variable, 0) + 1
        by_source[row.source] = by_source.get(row.source, 0) + 1
        if row.received_time is None:
            unverified += 1
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return {
        "schema": TABLE_SCHEMA,
        "path": str(path),
        "rows": len(rows),
        "bytes": len(text.encode("utf-8")),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "rows_by_variable": dict(sorted(by_variable.items())),
        "rows_by_source": dict(sorted(by_source.items())),
        "rows_latency_unverified": unverified,
    }


def read_tables(paths: list[str | Path]) -> tuple[list[ObsRow], list[dict]]:
    """Every table decoded once (neutral or declared-source layout)."""
    rows: list[ObsRow] = []
    provenance: list[dict] = []
    for path in paths:
        _, decoded, record = load_obs(str(path))
        rows.extend(decoded)
        provenance.append(record)
    return rows, provenance


def coverage(rows: list[ObsRow]) -> dict[str, dict[str, int]]:
    """Rows per source per variable."""
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        inner = out.setdefault(row.source, {})
        inner[row.variable] = inner.get(row.variable, 0) + 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def cut_hours(
    tables: list[str | Path],
    *,
    start: dt.datetime,
    end: dt.datetime,
    out_dir: Path,
    cycle_s: int = DEFAULT_CYCLE_S,
    cutoff: dt.datetime | None = None,
    thin_grid: tuple[int, int] | None = None,
    layer_ln_p: float = DEFAULT_LAYER_LN_P,
) -> dict[str, object]:
    """One neutral table per analysis instant from ``start`` to ``end`` at
    ``cycle_s``, each holding the rows of its window that the information
    cutoff admits (thinned to ``thin_grid = (nlat, nlon)`` when given), plus
    the ``gpuwm-obs.hours.v2`` manifest (rows per hour per source per
    variable, the cutoff counters, the thinning counters).  This is the
    file set the ensemble and the door read.  ``cutoff`` may be a fixed
    instant or the string ``"analysis"``, meaning each hour's own analysis
    instant (what a real-time cycle would have held)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, provenance = read_tables(tables)
    start = _utc(start)
    end = _utc(end)
    if end < start:
        raise ValueError("end precedes start")
    hours: list[dict] = []
    moment = start
    while moment <= end:
        w0, w1 = analysis_window(moment, cycle_s)
        chosen = rows_in_window(rows, w0, w1)
        hour_cutoff = moment if cutoff == "analysis" else cutoff
        chosen, cutoff_counters = apply_information_cutoff(chosen, hour_cutoff)
        thinning = None
        if thin_grid is not None:
            chosen, thinning = thin_to_grid(
                chosen, analysis_time=moment, nlat=thin_grid[0], nlon=thin_grid[1],
                layer_ln_p=layer_ln_p,
            )
        stamp = moment.strftime("%Y-%m-%dT%H")
        record = write_table(chosen, out_dir / f"{stamp}.csv")
        hours.append({
            "analysis_time": _stamp(moment),
            "window": [_stamp(w0), _stamp(w1)],
            "information_cutoff": None if hour_cutoff is None else _stamp(hour_cutoff),
            "cutoff_counters": cutoff_counters,
            "thinning": thinning,
            "table": record,
            "rows_by_source_variable": coverage(chosen),
        })
        moment += dt.timedelta(seconds=cycle_s)
    manifest = {
        "schema": HOURS_MANIFEST_SCHEMA,
        "table_schema": TABLE_SCHEMA,
        "cycle_s": cycle_s,
        "window_rule": "(t - cycle/2, t + cycle/2]",
        "cutoff_rule": (
            "a row is kept when its received_time is at or before the cutoff; a row "
            "without a received_time is kept and counted latency_unverified"
        ),
        "information_cutoff": (
            "each hour's analysis instant" if cutoff == "analysis"
            else (None if cutoff is None else _stamp(cutoff))
        ),
        "thin_grid": None if thin_grid is None else list(thin_grid),
        "inputs": provenance,
        "rows_total": len(rows),
        "hours": hours,
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "hours.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


# ---------------------------------------------------------------- manifest

def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stream_manifest(
    stream: str,
    *,
    table_record: Path,
    fetch_records: list[Path] = (),
    out: Path | None = None,
) -> dict[str, object]:
    """Wrap a door's table record (and its fetch records) into the
    ``gpuwm-obs.stream.v2`` manifest: volumes, digests, counters and the
    latency behind real time with its basis and class, read straight from
    the records the binary printed.  Nothing here recomputes a number the
    door already stated."""
    spec = STREAMS[stream]
    table = _read_json(table_record)
    fetches = [_read_json(p) for p in fetch_records]
    latency = table.get("latency_behind_real_time_s")
    basis = table.get("latency_basis")
    if latency is None and "latency_upper_bound_s" in table:
        latency = table["latency_upper_bound_s"]
        basis = table.get("latency_basis")
    if latency is None:
        for f in fetches:
            if f.get("latency_behind_real_time_s") is not None:
                latency = f["latency_behind_real_time_s"]
                basis = f.get("latency_basis")
    # Most doors state the table's path, digest, rows and bytes at the top of
    # their record; rw_madis nests them under "table" and calls its counters
    # "counts".  Read either, so a manifest never reports a written table as
    # rows=None (the first MADIS manifests did).
    inner = table.get("table") if isinstance(table.get("table"), dict) else {}
    manifest = {
        "schema": STREAM_MANIFEST_SCHEMA,
        "stream": stream,
        "door": spec.door,
        "subject": spec.subject,
        "table_schema": TABLE_SCHEMA,
        "table": {
            "path": table.get("path", inner.get("path")),
            "sha256": table.get("sha256", inner.get("sha256")),
            "rows": table.get("rows", inner.get("rows")),
            "bytes": table.get("bytes", inner.get("bytes")),
            "status": table.get("status"),
        },
        "errors": spec.errors,
        "measurements": list(spec.measurements),
        "counters": table.get("counters", table.get("counts")),
        "files": table.get("files"),
        "fetch_records": fetches,
        "latency_behind_real_time_s": latency,
        "latency_basis": basis,
        "latency_class": spec.latency_class,
        "latency_class_meaning": LATENCY_CLASSES[spec.latency_class],
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    if out is not None:
        Path(out).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def door_state(spec: StreamSpec) -> dict[str, object]:
    """What THIS install can do with ``spec``'s door, read from the install.

    ``decoder_built`` is true only when the door's binary resolves here now.
    It used to be the hard-coded declaration that the tree holds a decoder,
    so ``woof obs streams`` called buoys, radiosondes, motion vectors,
    occultations and WIS2 built on installs that carried none of their
    binaries.  ``door_route`` says how the next fetch reaches the door:
    ``installed``, ``on-demand`` (the first fetch stages the pinned binary
    from this release's bundle), ``build-from-source`` (nothing this install
    can download supplies it) or ``none`` (no decoding door exists).
    Resolution here never downloads (:func:`woof.bridges.inspection_only`).
    """

    from woof import bridges

    if spec.door is None or not spec.decoder_built:
        return {"decoder_built": False, "door_path": None, "door_route": "none"}
    door = _door(spec.door)
    with bridges.inspection_only():
        try:
            found = door.find()
        except FileNotFoundError:
            found = None
    if found is not None:
        return {"decoder_built": True, "door_path": str(found), "door_route": "installed"}
    route = "on-demand" if bridges.on_demand_fetch_offered(spec.door) else "build-from-source"
    return {"decoder_built": False, "door_path": None, "door_route": route}


def streams_table() -> list[dict[str, object]]:
    """Every stream with its door, class, latency basis and state: the
    ``streams`` command and the receipt's stream roster.  The door state
    (:func:`door_state`) is read from this install, not declared."""
    rows = []
    for spec in STREAMS.values():
        state = door_state(spec)
        rows.append({
            "stream": spec.name,
            "door": spec.door,
            "public": spec.public,
            "account_gated": spec.account_gated,
            **state,
            "latency_class": spec.latency_class,
            "latency_basis": spec.latency_basis,
            "cadence_s": spec.cadence_s,
            "variables": list(spec.variables),
            "measurements": list(spec.measurements),
            "notes": spec.notes,
        })
    return rows


# --------------------------------------------------------------------- CLI

def _door(name: str):
    """One observation front door by binary name, from the engine's own table
    (:func:`woof.obs.frontdoor.door_by_binary`).  A name the engine has no row
    for refuses by naming every door that does exist."""

    from woof.obs.frontdoor import door_by_binary

    return door_by_binary(name)


def _run_door(name: str, arguments: list[str]) -> dict:
    """Run one front door and return its JSON record (stdout).

    An unresolvable door is relayed as the engine's own resolution failure
    (it names the environment variable, the bridge directories and the build
    from a checkout), with the stream it stops; the exit code a workspace acts
    on is unchanged.
    """

    door = _door(name)
    try:
        binary = door.require()
    except Exception as reason:                     # the engine's own class
        raise RuntimeError(
            f"the Rust door {name} is not resolvable on this install, so the stream "
            f"that needs it cannot be fetched: {reason}") from None
    result = subprocess.run([str(binary), *arguments], capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        # rw-http's retry notes ("rw-http: HTTP 429 ... retrying") come before the door's own error.
        tail = [l for l in (result.stderr or "").splitlines() if l.strip() and not l.startswith("rw-http:")]
        raise RuntimeError(f"{name} {arguments[0]}: {tail[0] if tail else result.returncode}")
    return json.loads(result.stdout)


def _write_json(path: Path, record: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return path


def _refuse(record: dict, reason: str, remedy: str = "") -> int:
    """A stream door declining, in the shape this distribution publishes.

    The machine record stays on stdout so a workspace polling the door still
    gets its answer; the reason is the first line of stderr and the remedy
    follows it; the exit code is 1.

    Exit 1 and not 3: 3 is reserved for a Rust door that is missing or fails
    its pin, and it is separate precisely so a caller can act on it without a
    human by staging a bundle.  An archive that holds nothing in the window,
    and a feed that needs an account, are neither of those -- the door ran,
    verified and answered.  Reporting them as 3 sends an operator to restage
    binaries that are already staged, which is the shape the exit-code table
    in the command line exists to prevent.  Exit 2 is argparse's, and the
    command line was not the thing that was wrong either.
    """

    print(json.dumps(record, indent=1))
    print(reason, file=sys.stderr)
    if remedy:
        print(remedy, file=sys.stderr)
    return 1


#: The MADIS streams and the ``rw_madis --kind`` each one reads.
MADIS_KINDS = {"madis-acars": "acars", "madis-raob": "raob", "madis-mesonet": "mesonet"}


class StreamEmpty(RuntimeError):
    """A stream that holds nothing for the window, with the reason (the
    archive's latest day, the empty granule set); a caller states it and
    carries on, never fabricates rows.

    ``record`` is the door's OWN record where it wrote one.  The sentence is
    for a person; the record is for a caller, and a caller that has to parse
    the newest day back out of an English sentence to ask for a window the
    archive covers is a caller that will get it wrong.  So the fields the door
    reported travel beside the sentence, and the command merges them into the
    refusal it prints.
    """

    def __init__(self, message: str, record: dict | None = None) -> None:
        super().__init__(message)
        self.record = dict(record or {})


@dataclass
class StreamFetch:
    """What :func:`fetch_stream` hands back: the decoded table for the
    window and the manifest wrapping the door's fetch and table records."""

    stream: str
    table: Path
    manifest: dict
    fetch_records: list[Path]

    @property
    def latency_behind_real_time_s(self):
        return self.manifest.get("latency_behind_real_time_s")


def _day(text: str) -> str:
    return str(text)[:10]


def fetch_stream(
    stream: str, start: str, end: str, out_root: Path, *, networks: str | None = None,
    stations: str | None = None, satellites: str | None = None, seconds: int | None = None,
    missions: str | None = None, archive: str | Path | None = None,
    bbox: str | None = None,
) -> StreamFetch:
    """Fetch and decode one stream for the window ``[start, end]`` (ISO-8601
    UTC) through its Rust door into ``<out_root>/<stream>/<stream>.csv``,
    the fetch records beside it, the ``gpuwm-obs.stream.v2`` manifest
    written and returned.  The DA door and ``woof obs streams fetch``
    both call this.  A gated stream is refused by name; a window a
    retrospective archive does not cover raises :class:`StreamEmpty` with
    the archive's latest day."""
    spec = STREAMS[stream]
    if spec.account_gated:
        raise ValueError(f"stream {spec.name} needs an account and is reported, never fetched: {spec.notes}")
    if spec.door is None or not spec.decoder_built:
        raise ValueError(f"stream {spec.name} has no decoding door here: {spec.notes}")
    out = Path(out_root) / spec.name
    out.mkdir(parents=True, exist_ok=True)
    table = out / f"{spec.name}.csv"
    fetch_records: list[Path] = []
    if spec.name == "igra2":
        fetch_args = ["fetch", "--out", str(out / "raw")]
        if stations:
            fetch_args += ["--stations", stations]
        _run_door("rw_igra2", fetch_args)
        fetch_records.append(out / "raw" / "fetch.json")
        _run_door("rw_igra2", ["table", "--zips", str(out / "raw"), "--start", start, "--end", end,
                               "--fetch-record", str(out / "raw" / "fetch.json"), "--out", str(table)])
    elif spec.name == "ndbc":
        fetch_args = ["fetch", "--out", str(out / "raw")]
        if stations:
            fetch_args += ["--stations", stations]
        _run_door("rw_ndbc", fetch_args)
        fetch_records.append(out / "raw" / "fetch.json")
        _run_door("rw_ndbc", ["table", "--dir", str(out / "raw"), "--start", start, "--end", end,
                              "--fetch-record", str(out / "raw" / "fetch.json"), "--out", str(table)])
    elif spec.name == "iem-metar":
        networks = networks or "all"
        if networks == "all":
            listed = _run_door("rw_asos", ["networks", "--out", str(out / "networks.json")])
            networks = ",".join(listed["networks"])
        record = _run_door("rw_asos", ["fetch", "--networks", networks, "--start", start, "--end", end,
                                       "--out", str(out / "obs.csv")])
        fetch_records.append(_write_json(out / "fetch.json", record))
        _run_door("rw_asos", ["table", "--obs", str(out / "obs.csv"), "--start", start, "--end", end,
                              "--fetch-record", str(out / "fetch.json"), "--out", str(table)])
    elif spec.name == "awc-metar":
        _run_door("rw_asos", ["awc", "--out", str(table)])
    elif spec.name == "goes-dmw":
        records = []
        for satellite in (satellites or "G18,G19").split(","):
            record = _run_door("rw_amv", ["fetch", "--satellite", satellite.strip(), "--start", start,
                                          "--end", end, "--cache", str(out / "cache")])
            fetch_records.append(_write_json(out / f"fetch-{satellite.strip()}.json", record))
            records.append(record)
        by_hour: dict[str, list[str]] = {}
        for record in records:
            for f in record["files"]:
                hour = f["scan_start"][:13]
                by_hour.setdefault(hour, []).append(f["path"])
        if not by_hour:
            raise StreamEmpty(f"no DMW granules in the window {start} to {end}")
        hour_tables = []
        for hour in sorted(by_hour):
            hour_table = out / "hours" / f"{hour}.csv"
            arguments = ["table", "--files", ",".join(by_hour[hour]), "--out", str(hour_table)]
            for path in fetch_records:
                arguments += ["--fetch-record", str(path)]
            hour_tables.append(_run_door("rw_amv", arguments))
        # The stream table is the concatenation of the hours (one header).
        rows: list[str] = []
        for record in hour_tables:
            lines = Path(record["path"]).read_text(encoding="utf-8").splitlines()
            if not rows:
                rows.append(lines[0])
            rows.extend(lines[1:])
        text = "\n".join(rows) + "\n"
        table.write_text(text, encoding="utf-8", newline="\n")
        combined = dict(hour_tables[0])
        combined.update({
            "path": str(table), "rows": len(rows) - 1, "bytes": len(text.encode("utf-8")),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "hours": [{"path": r["path"], "rows": r["rows"], "sha256": r["sha256"]} for r in hour_tables],
            "counters": {"hours": len(hour_tables),
                         **{k: sum(int(r["counters"].get(k, 0)) for r in hour_tables)
                            for k in ("files_read", "files_empty", "vectors_in_files", "dqf_good",
                                      "zenith_rejected", "candidates", "thinned_away", "vectors_kept")}},
            "latency_behind_real_time_s": max(
                (r["latency_behind_real_time_s"] for r in hour_tables
                 if r.get("latency_behind_real_time_s") is not None), default=None),
        })
        _write_json(table.with_suffix(".json"), combined)
    elif spec.name == "gnss-ro":
        record = _run_door("rw_gnssro", ["list", "--start", _day(start), "--end", _day(end)])
        _write_json(out / "list.json", record)
        if record.get("status") == "EMPTY":
            raise StreamEmpty(
                f"the AWS occultation archive holds nothing for {_day(start)} to {_day(end)}; its latest "
                f"day is {record.get('latest_day_in_bucket')} (the recent days come through cdaac-ro)",
                {"latest_day_in_bucket": record.get("latest_day_in_bucket")})
        fetched = _run_door("rw_gnssro", ["fetch", "--start", _day(start), "--end", _day(end),
                                          "--cache", str(out / "cache")])
        fetch_records.append(_write_json(out / "fetch.json", fetched))
        _run_door("rw_gnssro", ["table", "--files", ",".join(f["path"] for f in fetched["files"]),
                                "--fetch-record", str(out / "fetch.json"), "--out", str(table)])
    elif spec.name == "cdaac-ro":
        arguments = ["cdaac-fetch", "--start", _day(start), "--end", _day(end), "--cache", str(out / "cache")]
        if missions:
            arguments += ["--missions", missions]
        fetched = _run_door("rw_gnssro", arguments)
        fetch_records.append(_write_json(out / "fetch.json", fetched))
        if fetched.get("status") == "EMPTY" or not fetched.get("files"):
            raise StreamEmpty(
                f"no CDAAC tarball published for {_day(start)} to {_day(end)} ({len(fetched.get('missing') or [])} "
                f"URLs answered 404): a day's tarball appears about five hours after the day ends")
        _run_door("rw_gnssro", ["table", "--tarballs", ",".join(f["path"] for f in fetched["files"]),
                                "--start", start, "--end", end,
                                "--fetch-record", str(out / "fetch.json"), "--out", str(table)])
    elif spec.name in MADIS_KINDS:
        kind = MADIS_KINDS[spec.name]
        raw = out / "raw"
        fetched = _run_door("rw_madis", ["fetch", "--kind", kind, "--start", start, "--end", end,
                                         "--out", str(raw)])
        record = raw / f"fetch-{kind}.json"
        fetch_records.append(record)
        if fetched.get("status") == "EMPTY" or not fetched.get("files"):
            raise StreamEmpty(
                f"MADIS has no {kind} hour file for {start} to {end} in real time or in the archive "
                f"({len(fetched.get('hours_not_in_archive') or [])} hours answered 404)",
                {"hours_not_found": len(fetched.get("hours_not_in_archive") or [])})
        arguments = ["table", "--kind", kind, "--files",
                     *[str(raw / f["name"]) for f in fetched["files"]],
                     "--fetch-record", str(record), "--start", start, "--end", end, "--out", str(table)]
        if bbox:
            arguments += ["--bbox", bbox]
        _run_door("rw_madis", arguments)
    elif spec.name == "wis2":
        # A subscriber already running beside the cycle (`rw_wis2 subscribe
        # --seconds 86400 --out DIR`) is the live route: its archive is
        # decoded for the window.  Without one, subscribe for ``seconds``
        # first (the broker hands over what is published meanwhile, not
        # the hour behind: a ten-minute listen is a sixth of the hour).
        if archive is None:
            arguments = ["subscribe", "--out", str(out / "archive"), "--seconds", str(seconds or 600)]
            record = _run_door("rw_wis2", arguments)
            fetch_records.append(_write_json(out / "subscribe.json", record))
            if not (record.get("coverage") or {}).get("payloads_downloaded"):
                raise StreamEmpty(
                    f"wis2 subscribed for {seconds or 600} s on {record.get('broker')}: "
                    f"{record.get('coverage', {}).get('messages')} messages and no payload archived")
            archive = out / "archive"
        _run_door("rw_wis2", ["table", "--archive", str(archive), "--start", start, "--end", end,
                              "--out", str(table)])
    else:
        raise ValueError(f"stream {spec.name} is not fetched by this function: {spec.notes}")
    manifest = stream_manifest(spec.name, table_record=table.with_suffix(".json"),
                               fetch_records=fetch_records, out=out / "manifest.json")
    return StreamFetch(stream=spec.name, table=table, manifest=manifest, fetch_records=fetch_records)


def _cmd_fetch(args) -> int:
    """Fetch and decode one stream for a window through its Rust door.

    The library function raises; this command turns each refusal into the
    exit code this distribution publishes.  Both refusals are exit 1 and
    NEITHER is exit 3, which is reserved for a Rust door that is missing or
    fails its pin -- the one refusal a caller can act on without a person, by
    staging a bundle.  A window an archive does not cover and a feed that
    needs an account are neither: the door ran, verified and answered.
    Reporting them as 3 sends an operator to restage binaries that are
    already staged.  Exit 2 is argparse's, and the command line was not the
    thing that was wrong either.
    """

    spec = STREAMS[args.stream]
    if spec.account_gated:
        return _refuse(
            {"stream": spec.name, "status": "ACCOUNT_GATED", "notes": spec.notes},
            f"stream {spec.name} is behind an account and this door fetches nothing "
            f"anonymously",
            f"the sources are {', '.join(spec.sources)}; register there and feed the "
            f"files in as tables, or use a stream in the same measurement class",
        )
    if spec.door is None or not spec.decoder_built:
        remedy = ""
        if spec.subscribes:
            remedy = (f"the door's subscriber is reachable with `woof obs streams "
                      f"subscribe --stream {spec.name}`, which archives what arrives "
                      f"and reports coverage without claiming a table")
        return _refuse(
            {"stream": spec.name, "status": "NO_DECODER", "notes": spec.notes},
            f"stream {spec.name} has no decoding door here: {spec.notes}",
            remedy,
        )
    try:
        fetched = fetch_stream(
            args.stream, args.start, args.end, Path(args.out), networks=args.networks,
            stations=args.stations, satellites=args.satellites, seconds=args.seconds,
            missions=args.missions, archive=args.archive, bbox=args.bbox,
        )
    except StreamEmpty as empty:
        record = {"stream": spec.name, "status": "EMPTY",
                  "latency_class": spec.latency_class,
                  "reason": str(empty),
                  "notes": spec.notes}
        record.update({k: v for k, v in empty.record.items() if v is not None})
        return _refuse(
            record,
            f"stream {spec.name} holds no files in {args.start} .. {args.end}: "
            f"the door listed the collection and it is empty for that window",
            str(empty),
        )
    manifest = fetched.manifest
    print(json.dumps({k: manifest[k] for k in ("stream", "table", "latency_behind_real_time_s",
                                               "latency_basis", "latency_class")}, indent=1))
    return 0


def _cmd_subscribe(args) -> int:
    """Subscribe to a stream's live feed through its Rust door.

    Separate from `fetch` because it is a different verb with a different
    product: `fetch` asks an archive for a window and writes the neutral
    table, and returns when the window is decoded; `subscribe` opens a live
    connection for a stated number of seconds, archives every message and
    payload it receives with its integrity digest, and reports coverage.  A
    stream whose payloads are BUFR has no table door yet, and folding the two
    verbs together would either make `fetch` return without the table it
    promises or leave the subscriber reachable only by running the binary,
    which is where this command found it.
    """

    spec = STREAMS[args.stream]
    if not spec.subscribes:
        raise SystemExit(
            f"stream {spec.name} has no subscriber: its door reads an archive, not a "
            f"live feed; the streams that subscribe are "
            f"{', '.join(sorted(s.name for s in STREAMS.values() if s.subscribes))}")
    out = Path(args.out) / spec.name
    out.mkdir(parents=True, exist_ok=True)
    seconds = args.seconds or 600
    record = _run_door(spec.door, ["subscribe", "--out", str(out / "archive"),
                                   "--seconds", str(seconds)])
    _write_json(out / "subscribe.json", record)
    # The summary carries the door's OWN keys, and a key the door does not
    # write is REPORTED missing rather than printed as a null.  The first
    # version of this line named four counters this door has never emitted
    # (`messages`, `payloads_archived`, `digests_verified`, `centres`); the
    # command exited 0 and printed four nulls beside a real archive, which
    # reads as "the subscriber received nothing" when it had just archived 68
    # messages from 5 centres.  A summary that can silently describe nothing
    # is worse than no summary.
    wanted = ("schema", "status", "broker", "listened_s", "coverage",
              "publication_delay_s", "transport_delay_s",
              "latency_behind_real_time_s", "latency_basis")
    summary = {"stream": spec.name, "seconds": seconds,
               "archive": str(out / "archive"),
               "record": str(out / "subscribe.json")}
    summary.update({k: record[k] for k in wanted if k in record})
    missing = [k for k in wanted if k not in record]
    if missing:
        summary["not_reported_by_the_door"] = missing
    print(json.dumps(summary, indent=1))
    return 0


def _cmd_hours(args) -> int:
    cutoff: dt.datetime | str | None
    if args.cutoff is None:
        cutoff = None
    elif args.cutoff == "analysis":
        cutoff = "analysis"
    else:
        cutoff = parse_valid_time(args.cutoff)
        if cutoff is None:
            raise SystemExit("--cutoff must be an ISO-8601 instant or 'analysis'")
    thin_grid = None
    if args.thin_grid:
        parts = args.thin_grid.split(",")
        if len(parts) != 2:
            raise SystemExit("--thin-grid takes NLAT,NLON")
        thin_grid = (int(parts[0]), int(parts[1]))
    manifest = cut_hours(
        args.tables,
        start=parse_valid_time(args.start), end=parse_valid_time(args.end),
        out_dir=Path(args.out), cycle_s=args.cycle_s, cutoff=cutoff, thin_grid=thin_grid,
    )
    for hour in manifest["hours"]:
        print(hour["analysis_time"], hour["table"]["rows"],
              "unverified", hour["cutoff_counters"]["latency_unverified"],
              json.dumps(hour["rows_by_source_variable"]))
    return 0


def _cmd_summary(args) -> int:
    """Volumes and latency per stream from the manifests under a directory."""
    root = Path(args.dir)
    for manifest_path in sorted(root.glob("*/manifest.json")):
        m = _read_json(manifest_path)
        print(f"{m['stream']:16s} rows={m['table'].get('rows')} bytes={m['table'].get('bytes')} "
              f"class={m.get('latency_class')} latency_s={m.get('latency_behind_real_time_s')} "
              f"basis={m.get('latency_basis')}")
    for spec in account_gated_sources():
        print(f"{spec.name:16s} ACCOUNT-GATED: {spec.notes}")
    return 0


def _cmd_streams(args) -> int:
    print(json.dumps(streams_table(), indent=1))
    return 0


def _cmd_anchors(args) -> int:
    print(json.dumps([{
        "anchor": a.name, "subject": a.subject, "mapping": a.mapping,
        "cadence_s": a.cadence_s, "availability_basis": a.availability_basis,
        "url_template": a.url_template, "notes": a.notes,
    } for a in ANCHOR_SOURCES.values()], indent=1))
    return 0


def add_stream_commands(sub) -> None:
    """The stream subcommands on an argparse subparsers object: the module's
    own ``main`` and ``woof obs streams`` share them, so the product door
    and the module entry point cannot drift."""
    fetch = sub.add_parser("fetch", help="fetch and decode one stream for a window through its Rust door")
    fetch.add_argument("--stream", required=True, choices=sorted(STREAMS))
    fetch.add_argument("--start", required=True, help="window start, ISO-8601 UTC")
    fetch.add_argument("--end", required=True, help="window end, ISO-8601 UTC")
    fetch.add_argument("--out", required=True, help="directory; the stream lands under <out>/<stream>/")
    fetch.add_argument("--networks", help="iem-metar: comma-separated IEM networks, or 'all' (default)")
    fetch.add_argument("--stations", help="igra2 or ndbc: comma-separated station ids (default all)")
    fetch.add_argument("--satellites", help="goes-dmw: comma-separated, default G18,G19")
    fetch.add_argument("--seconds", type=int, help="wis2: how long to subscribe (default 600)")
    fetch.add_argument("--missions", help="cdaac-ro: comma-separated missions (default cosmic2,paz,kompsat5)")
    fetch.add_argument("--archive", help="wis2: a subscriber's archive directory to decode for the window "
                                         "instead of subscribing now")
    fetch.add_argument("--bbox", help="madis-*: keep reports inside S,N,W,E (degrees); default worldwide")
    fetch.set_defaults(func=_cmd_fetch)
    subscribe = sub.add_parser(
        "subscribe",
        help="open a stream's live feed through its Rust door for a stated number of "
             "seconds, archiving every message and payload with its digest")
    subscribe.add_argument(
        "--stream", required=True,
        choices=sorted(s.name for s in STREAMS.values() if s.subscribes),
        help="the stream to subscribe to; only doors that read a live feed are listed")
    subscribe.add_argument("--out", required=True,
                           help="directory; the archive lands under <out>/<stream>/archive/")
    subscribe.add_argument("--seconds", type=int, default=600,
                           help="how long to stay subscribed (default 600)")
    subscribe.set_defaults(func=_cmd_subscribe)
    hours = sub.add_parser("hours", help="cut stream tables into one table per analysis instant")
    hours.add_argument("--tables", nargs="+", required=True, help="neutral tables (v2 or v1)")
    hours.add_argument("--start", required=True, help="first analysis instant, ISO-8601 UTC")
    hours.add_argument("--end", required=True, help="last analysis instant, ISO-8601 UTC")
    hours.add_argument("--cycle-s", type=int, default=DEFAULT_CYCLE_S, help="cycle length in seconds")
    hours.add_argument("--cutoff", help="information cutoff: an ISO-8601 instant, or 'analysis' for "
                                        "each hour's own instant; rows received later are dropped and "
                                        "counted, rows without a receipt time are kept and counted")
    hours.add_argument("--thin-grid", help="NLAT,NLON: keep one row per source, variable, cell and "
                                           "ln p layer, the one nearest the analysis instant")
    hours.add_argument("--out", required=True, help="directory for the hourly tables and hours.json")
    hours.set_defaults(func=_cmd_hours)
    summary = sub.add_parser("summary", help="volumes and latencies from the manifests under a directory")
    summary.add_argument("--dir", required=True, help="the directory `fetch` wrote the streams under")
    summary.set_defaults(func=_cmd_summary)
    streams = sub.add_parser("streams", help="every stream with its door, latency class and state")
    streams.set_defaults(func=_cmd_streams)
    anchors = sub.add_parser("anchors", help="the external analyses usable as the weak background constraint")
    anchors.set_defaults(func=_cmd_anchors)


def register_cli(obs_subparsers) -> None:
    """``woof obs streams <fetch|hours|summary|streams|anchors>``: the
    product door onto this module."""
    parser = obs_subparsers.add_parser(
        "streams",
        help="the global data-assimilation observation streams: fetch and decode a "
             "stream through its Rust door, cut hourly tables, list streams and anchors")
    sub = parser.add_subparsers(dest="obs_streams_command", required=True)
    add_stream_commands(sub)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="woof obs streams",
        description="the observation streams of the global data assimilation",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    add_stream_commands(sub)
    args = parser.parse_args(argv)
    return args.func(args)


__all__ = [
    "ANCHOR_RECORD_SCHEMA",
    "ANCHOR_SOURCES",
    "AnchorSource",
    "DEFAULT_CYCLE_S",
    "DEFAULT_LAYER_LN_P",
    "HOURS_MANIFEST_SCHEMA",
    "LATENCY_CLASSES",
    "MADIS_KINDS",
    "REFRACTIVITY_ERRORS",
    "STREAMS",
    "STREAM_MANIFEST_SCHEMA",
    "StreamEmpty",
    "StreamFetch",
    "StreamSpec",
    "account_gated_sources",
    "add_stream_commands",
    "analysis_window",
    "apply_information_cutoff",
    "coverage",
    "cut_hours",
    "fetch_stream",
    "fetchable_streams",
    "main",
    "read_tables",
    "register_cli",
    "rows_in_window",
    "stream_manifest",
    "streams_table",
    "thin_to_grid",
    "write_table",
]


if __name__ == "__main__":
    sys.exit(main())
