"""Acquire initialization/boundary data: the ``woof fetch`` front door.

This module is transport only.  It downloads (or, for ERA5, templates and
validates) the exact source inventory the fail-closed Rust GRIB bridges
consume downstream; it never decodes scientific payloads itself.  Every
published file is envelope-verified, sha256-summed, and recorded in a
``fetch-manifest.json`` the preparation step can consume.

Per-source transport:

Every NCEP source here is asked for along an ENDPOINT LADDER declared
in ``authorities/rw-wps-fetch-routes.v1.json`` and resolved by
:mod:`woof.fetch_endpoints`: the operational server
(``nomads.ncep.noaa.gov``) while it still holds the cycle, the AWS Open
Data archive behind it.  The two publish the same relative key with the
same bytes and differ in three ways only -- the operational server has
a latest cycle hours before any mirror does, it keeps a bounded window,
and it paces bulk transfers where the archive does not.

Retention decides which rungs are ASKED; throughput decides which one
SERVES.  Inside the retention window each requested object is HEADed on
the archive first, and the archive takes any object it has already
mirrored, because the operational server's head start is spent the
moment both hosts have the same bytes.  An object the archive has not
caught up with comes from the operational server, which is what that
host is for.  Promotion reorders the ladder and never shortens it, so
fall-through is unchanged.  ``--transport`` pins one rung, disables
fall-through and skips the probe.

* ``gfs`` -- three byte transports.  The default is the NOMADS
  ``filter_gfs_0p25.pl`` subsetter (spatial subregion + the exact
  variable/level selection): rate-governed, bandwidth-frugal, re-encoded
  by NOMADS to south-to-north simple packing.  ``--mode full-file``
  takes the whole ``pgrb2.0p25`` objects along the ladder instead --
  whole-globe north-to-south (scan 0x00) complex-packed (DRT 5.3)
  grids, both certified in ``gfs_grib2_bridge`` by committed matched
  pairs (the scan-order flip and the SOILW missing-value proof; see
  ``tests/fixtures/gfs-scan-order/README.md``) -- through either the
  Rust backbone's parallel range GETs or the stdlib transport.
  ``--mode idx-subset`` takes from those same objects only the records
  the grib-filter asks for, by the byte ranges of the ``.idx`` beside
  each one (:data:`GFS_RECORDS_MODE`): the records are copied unchanged,
  so the decode reads the bytes it reads from the whole object, about a
  sixth of them.  It is what the hosted AWS policy and a cycle only the
  archive keeps take; an object whose host keeps no index moves whole.
  ``--cycle latest`` walks the same ladder with anonymous HEAD probes
  (no HTML scraping), so it resolves the newest cycle that EXISTS
  rather than the newest one the archive has caught up with.
* ``hrrr`` -- NOAA ``.idx`` byte-range subsetting, reusing the proven
  record inventory and range transport in
  :mod:`tools.download_hrrr_native_subset` (native hybrid ``wrfnat``
  atmosphere plus the soil records of ``wrfprs``, the inputs
  ``hrrr_grib2_bridge`` requires), over either of two hosts serving the
  identical production files and indexes: the NOMADS operational server
  (``nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod``, roughly the
  newest 48 h, where each hour publishes first) and the AWS Open Data
  S3 archive ``noaa-hrrr-bdp-pds``.  ``--transport auto`` (default)
  asks the archive for the requested window first and takes it when it
  already serves it, falling back to the operational server for a
  window the archive has not caught up with, and skipping the doomed
  probe entirely for a cycle older than the operational window.
  NOMADS
  has no grib-filter route for these products -- its HRRR filter
  scripts cover the 2-D ``wrfsfc`` file only -- so subsetting stays
  ``.idx`` byte ranges on both hosts and the exact 561/18 record
  contracts are unchanged.  By default (:data:`HRRR_DEFAULT_MODE`,
  ``auto``) only the selected records move, through the Rust
  backbone's parallel range GETs, and an object whose index is absent
  or short moves whole; ``--mode full-file`` takes every object whole.  ``--wait-for`` polls (at most every 30 s)
  and downloads each forecast hour as it publishes, so preparation can
  start before a live cycle finishes publishing.  HRRR objects are
  CONUS-wide; ``.idx`` subsetting selects records, not areas, so
  ``--area`` is validated against CONUS coverage rather than used to
  crop.
* ``era5`` -- no CDS download is implemented (the CDS API requires a user
  account and key).  ``fetch`` emits the precise ``cdsapi`` request
  template for the variables/levels/times/area woof ingest expects and,
  with ``--validate``, checks a user-supplied GRIB1 file set against that
  expectation (transport envelopes via
  :func:`woof.ingest.grib.inspect_grib1_envelopes` plus a
  parameter/level/valid-time census).  The Rust bridge remains the decode
  authority at ingest time.

No case is named here; sources (GFS/HRRR/ERA5) are public data products,
not cases.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import contextlib
import functools

from woof import (explain, fetch_bars, fetch_endpoints, fetch_guard,
                   fetch_pool, fetch_routes, source_adapters)
# ALIASED.  `progress` is the name of the per-route reporting callable on
# most signatures in this module, so importing the module under its own
# name would be shadowed by the parameter inside every one of them.
from woof import progress as progress_mod
from woof.config_keys import KeyRow, key_rows
from woof.explain import layered
from woof.nomads_governor import paced_urlopen
from woof.filesystem_paths import DOWNLOAD_DEPTH_BUDGET, deep_io_path


FETCH_MANIFEST_SCHEMA = "gpuwm-fetch-manifest-v1"
FETCH_MANIFEST_NAME = "fetch-manifest.json"

#: Every receipt a fetched directory publishes, manifest first.
#:
#: All four NAME PAYLOADS: the manifest carries a digest per file, the
#: checksum list is what the prep door consumes verbatim, ``inputs.txt``
#: is a list of resolved payload paths, and ``prep-command.txt`` binds
#: the series.  So all four can outlive the bytes they describe, which
#: is the one directory state the fetch state machine refuses to leave
#: behind, and the force sweep has to move all four aside before it
#: touches a payload rather than only the two it used to know about.
#: Manifest first inside that class because it is the file the front
#: door reads and refuses on: while it is canonical the directory still
#: presents itself as a completed fetch.
FETCH_RECEIPT_NAMES = (FETCH_MANIFEST_NAME, fetch_routes.SHA256SUMS_NAME,
                       fetch_routes.INPUT_LIST_NAME,
                       fetch_routes.PREP_COMMAND_NAME)

#: The GFS front door (``rw-wps --source gfs``) verifies its inputs
#: against this manifest schema (woof/gfs_direct.py
#: ``_verify_input_manifest``).  ``woof fetch --source gfs
#: --author-front-door-manifest`` writes it, so the value is mirrored
#: here to keep fetch importable on a base install (no ingest imports);
#: a test binds the two constants together.
GFS_FRONT_DOOR_MANIFEST_SCHEMA = "gpuwm-gfs-direct-input-manifest-v1"
GFS_INPUT_MANIFEST_NAME = "gfs-input-manifest.json"

#: Margin (degrees) the suggested GFS fetch crop adds beyond the outer
#: domain.  Two terms, both anchored in the front door's donor-coverage
#: proof (woof/gfs_direct.py ``_source_coverage_receipt``):
#:
#: * the deterministic parabolic/masked interpolation stencil reaches
#:   floor-based [-1, +2] source cells -- 2 cells = 0.5 deg at the
#:   0.25-deg GFS resolution -- so the crop needs at least that halo;
#: * lake initialization takes each model lake's nearest source-water
#:   donor from the crop, which is the nearest GFS water only when the
#:   crop edge is farther from the lake than that donor.  Interior
#:   continental lakes can sit many degrees from the nearest
#:   GFS-resolved water, so the suggested crop allows
#:   :data:`GFS_LAKE_DONOR_MARGIN_DEG` for that search.  A lake the crop
#:   cannot show its nearest donor for, or holds no water for at all,
#:   prepares and is counted in the coverage receipt.
GFS_SOURCE_RESOLUTION_DEG = 0.25
GFS_DONOR_HALO_CELLS = 2
GFS_LAKE_DONOR_MARGIN_DEG = 15.0


def gfs_suggested_fetch_margin_deg() -> float:
    """Fetch-crop margin (deg) sized for the GFS donor-coverage proof.

    Used by ``woof domain`` to compute its suggested ``--area`` so the
    wizard's own hint passes the front door's coverage check instead of
    being rejected downstream.
    """
    return max(GFS_DONOR_HALO_CELLS * GFS_SOURCE_RESOLUTION_DEG,
               GFS_LAKE_DONOR_MARGIN_DEG)

#: The three sources whose transport predates the route table still
#: read their endpoints FROM it (``legacy_ladders``), so "which host,
#: and in what order" is one table fact for every NCEP source rather
#: than a constant here and a row there.  The names below are kept
#: because callers and receipts have always spelled them this way.
#:
#: Both HRRR hosts serve byte-identical files and ``.idx`` indexes
#: (HEAD Content-Length, ``Accept-Ranges: bytes`` and HTTP 206 range
#: responses verified 2026-07-29; the equal-Content-Length pairing
#: re-verified across the whole NCEP family 2026-08-24).  The
#: operational server publishes each hour first and keeps a bounded
#: window; the archive lags and keeps everything.
GFS_S3_BASE = fetch_endpoints.endpoint_named("gfs", "s3").base
GFS_NOMADS_BASE = fetch_endpoints.endpoint_named("gfs", "nomads").base
HRRR_S3_BASE = fetch_endpoints.endpoint_named("hrrr", "s3").base
HRRR_NOMADS_BASE = fetch_endpoints.endpoint_named("hrrr", "nomads").base
#: Approximate NOMADS retention (hours).  Older cycles live on S3 only.
HRRR_NOMADS_RETENTION_HOURS = int(
    fetch_endpoints.endpoint_named("hrrr", "nomads").retention_hours)
HRRR_TRANSPORTS = ("auto", "nomads", "s3")

#: Every host name ``--transport`` accepts across all routes: HRRR's
#: three plus whatever the packaged route table's rows declare.  The
#: union lives here rather than in argparse literals so a new row in the
#: table teaches the front door its host without a code edit.
FETCH_TRANSPORTS = tuple(sorted(
    set(HRRR_TRANSPORTS)
    | {row["transport"] for row in fetch_endpoints.archive_document()["layouts"]}
    | {host.name
       for source_id in fetch_routes.route_ids()
       for host in fetch_routes.route_for(source_id).hosts}))

#: Which downloader moves the bytes.  ``rust`` is the vendored
#: ``rw_fetch`` backbone (parallel range GETs, ``.idx``
#: coalescing, the cross-process NOMADS rate governor, a disk cache);
#: ``python`` is the stdlib ``urllib`` transport in :mod:`tools`, which
#: stays as the always-available fallback.  ``auto`` uses the backbone
#: when it is built and the Python transport when it is not.
FETCH_ENGINES = ("auto", "rust", "python")

#: HOW the downloader in a receipt was chosen, which ``engine`` alone
#: cannot say.  ``engine: "python"`` covers two different situations --
#: an operator who asked for the stdlib transport, and an install that
#: inherited it because the backbone was not there -- and only the
#: second one is a measured tax somebody would want to see in a
#: receipt after a slow run.  So the receipt carries both fields.
#:
#: This is deliberately NOT called ``transport``: on this front door
#: ``--transport`` already names the HOST (nomads or s3), a separate
#: axis, and a second meaning for the word in the same document would
#: be worse than a longer key.
FETCH_ENGINE_SELECTIONS = ("rust", "python-requested", "python-fallback")
PYTHON_FALLBACK_SELECTION = "python-fallback"

#: The one sentence an install gets when it inherits the slow transport.
#: One line, at SELECTION time, so every caller of the front door says it
#: -- the HRRR command said something like it and the GFS full-file
#: command, the streamer's preflight and every library caller said
#: nothing at all.
_PYTHON_TRANSPORT_TAX = (
    "woof fetch is using the Python transport ({reason}).  It has no "
    "whole-file branch: every object is pulled as hundreds of serial "
    ".idx range GETs, measured at 560 s for one 419 MB HRRR file "
    "against 27-35 s for the same file taken whole through the rust "
    "backbone -- roughly a 16x tax.  Install the bridges bundle "
    "(`woof setup`, or `woof fetch-bridges`) to get the fast path.")

_PYTHON_TRANSPORT_TAX_WHY = (
    "The Python transport is the always-available fallback and it is "
    "correct; it is simply the slow one, and an install should not "
    "discover that after the download rather than before it.  On NOMADS "
    "it is worse than 16x: the cross-process rate governor allows one "
    "worker per NOMADS URL with a 2.5 s minimum interval, so the "
    "degraded path is one thread pausing between every range request.  "
    "The receipt records engine_selection='python-fallback' so a run "
    "that paid this can be recognised afterwards without guessing.")


@dataclass(frozen=True)
class FetchEngineChoice:
    """Which downloader was chosen, and whether anybody chose it."""

    engine: str
    binary: Path | None
    selection: str
    reason: str | None = None

    @property
    def degraded(self) -> bool:
        return self.selection == PYTHON_FALLBACK_SELECTION

#: ``--transport`` picks the *host*; ``--mode`` picks the *byte
#: transport*, which is a separate axis: whether to pull the whole
#: object or only the ``.idx``-selected byte ranges out of it.  ``auto``
#: is the probe rule -- object present, and its ``.idx`` absent,
#: malformed, or provably shorter than the object => take the whole
#: file.  No time constants are involved; both named modes are
#: first-class and either can be forced.
FETCH_MODES = ("auto", "full-file", "idx-subset")


def archive_only_cycle(source: str, cycle: datetime, now: datetime | None = None) -> bool:
    """Whether only a source's archive endpoints still hold ``cycle``.

    Read from the endpoint table: every endpoint with a rolling
    retention is too young for the cycle, and at least one archive
    endpoint (no retention) remains.
    """

    rungs = fetch_endpoints.ladder(source) if fetch_endpoints.has_ladder(source) else ()
    age = fetch_endpoints.cycle_age_hours(cycle, now)
    rolling = [entry for entry in rungs if not entry.archive]
    return (bool(rolling) and any(entry.archive for entry in rungs)
            and not any(entry.covers(age) for entry in rolling))

#: What ``woof fetch --source hrrr`` does when nobody says otherwise.
#:
#: ``auto``: the backbone's probe rule takes the ``.idx`` record subset
#: whenever the host publishes a complete index, and the whole object
#: when the index is absent, malformed or shorter than the object, and
#: each file's line says which it took and why.  A whole ``wrfnat`` is
#: 66-70 % read and a whole ``wrfprs`` 7 % (18 soil records of 708), so
#: the whole-file default moved 2.5x the bytes the preparation reads.
#: Measured over AWS, 2026-10-29 12Z f00-f03: whole files
#: 4,601 MB in 176 s, the subset 1,823 MB in 137 s.  A 12 h hourly start
#: is about 15.1 GB whole and 5.6 GB subset, on the start WOOF and the
#: DA cycles make most.
#:
#: The 2026-07-31 reason for the whole-file default was the PYTHON
#: transport's serial range requests (560 s for one 419 MB file); that
#: transport only ever subsets and is unchanged.  The Rust backbone
#: coalesces a selection into about 108 ranges per wrfnat and pulls them
#: in parallel, and the subset carries every record the bridge reads
#: from a whole file (``OPTIONAL_SURFACE_FIELDS`` in
#: :mod:`tools.download_hrrr_native_subset`), so the prepared start is
#: byte-identical either way.  ``--mode full-file`` and ``--mode
#: idx-subset`` still force one byte mode.
HRRR_DEFAULT_MODE = "auto"

#: ``woof fetch --transport`` host names to ``rw_fetch --source``
#: registry names.  Same two hosts, different vocabularies: ArWen has
#: said ``s3`` since before there was a registry, and rustwx calls the
#: same bucket ``aws``.
RW_FETCH_SOURCES = {"nomads": "nomads", "s3": "aws"}

#: ``rw_fetch --model``/``--product`` for each HRRR product ArWen wants.
RW_FETCH_HRRR_PRODUCTS = {"atmosphere": "nat", "soil": "prs"}

#: ``rw_fetch --product`` for the raw GFS object.
RW_FETCH_GFS_PRODUCT = "pgrb2.0p25"

#: ``--wait-for`` polling cadence ceiling (seconds between probe rounds).
HRRR_WAIT_POLL_SECONDS = 30
#: ``--wait-for`` default patience: 90 min covers a live HRRR cycle's
#: full f00..f18 publication spread with margin.
HRRR_WAIT_TIMEOUT_DEFAULT_MINUTES = 90.0

#: The HRRR front door's flags no fetch can bind, in the order the
#: handoff names them.  ``woof domain --source hrrr`` writes the files
#: behind the first four beside the config it emits
#: (:func:`woof.hrrr_route_inputs.route_input_paths`); the geography
#: root and the output root are the reader's.  The handoff used to name
#: four of these six, and a line completed with exactly those four was
#: refused at the door: ``invalid or missing run arguments:
#: --namelist-input, --domain-spec (required with --geog-root)``.
HRRR_CALLER_SUPPLIES = ("--domain-spec", "--namelist-input",
                        "--wps-namelist", "--experiment-config",
                        "--geog-root", "--output-root")

GFS_CYCLE_HOURS = (0, 6, 12, 18)
GFS_MAX_FORECAST_HOUR = 384

#: The last GFS 0.25-degree ``pgrb2`` lead published EVERY hour.
#:
#: NCEP publishes that product hourly through f120 and 3-hourly from
#: f120 to f384.  f121, f122 and f124 are not late -- they do not exist
#: and never will, and a HEAD against the archive returns 404 for each
#: while f120 and f123 return 200.  Modelling that break here is what
#: lets a window crossing it be refused for what it is: an availability
#: probe alone reported the permanent gap as "not published yet" and
#: sent a reader off to wait for data no cycle will ever carry.
GFS_HOURLY_MAX_FORECAST_HOUR = 120

#: Native publication intervals, not a whitelist of requested subsets.
GFS_PUBLISHED_CADENCES_H = (1, 3)


def gfs_cadence_break_refusal(start: int, last: int, cadence: int) -> str:
    """Name a requested lead absent from the published source ladder."""
    later_spacing = GFS_PUBLISHED_CADENCES_H[-1]
    missing = next(lead for lead in range(start, last + 1, cadence)
                   if lead > GFS_HOURLY_MAX_FORECAST_HOUR and lead % later_spacing)
    return layered(
        f"--cadence {cadence} requests f{missing:03d}, which GFS does not publish: "
        f"the product is published every hour only through f{GFS_HOURLY_MAX_FORECAST_HOUR:03d} "
        f"and every {later_spacing} hours afterward through f{GFS_MAX_FORECAST_HOUR}.\n"
        f"  What to do: end at f{GFS_HOURLY_MAX_FORECAST_HOUR:03d} or earlier, or use "
        f"--cadence {later_spacing} with start/end leads on the {later_spacing} h grid"
        + (f" (near this start: f{start - start % later_spacing:03d} or "
           f"f{start + later_spacing - start % later_spacing:03d})"
           if start % later_spacing else "") + ".",
        "  Why: this is an absent source object, not a late cycle. Positive uniform "
        "subsets of the published leads are accepted without changing their spacing.")


#: GDAS is the GFS assimilation cycle's own output, in the *same*
#: pgrb2.0p25 container: same 0.25-degree regular lat/lon grid, same
#: variable and level codes, same 124-record census under the certified
#: selector, same originating centre and table versions.  Verified
#: against a live cycle before this lane was wired -- 124 records, scan
#: 0x40, DRT 5.0, shape 6, centre 7, master table 2, local table 1,
#: PDT 4.0, generating process 81 at f000.  The v1.1 proof corpus adds
#: real f003/f006/f009 subsets with generating process 96; fetch declares
#: the expected process ID per row so the bridge never infers that
#: capability from an hour or a source name.
#:
#: **Fetch and decode, not a front door.**  v1.0.1 scoped this source to
#: f000 because the fail-closed ``gfs_grib2_bridge`` -- which selects by
#: exact field identity and never guesses -- was certified only against
#: the process the analysis carries, and it said that widening the gate
#: would be a re-certification event rather than a flag.  That event has
#: happened: real NOMADS f000/f003/f006/f009 subsets of
#: ``gdas.20260729/12`` are committed under
#: ``tests/fixtures/gdas-process-id/`` -- the last of them the published
#: endpoint itself, so the ceiling rests on bytes and not on this
#: constant -- all 124 messages each frozen at the envelope above, and
#: the bridge now
#: verifies a *declared* process ID against its certified ``{81, 96}``
#: set (each forecast sample is also required to fail under the
#: undeclared analysis-only policy).  So the fetch/decode span is
#: f000..f009 again.
#:
#: The packaged composed profile supplies the preparation mapping.
#: Container acquisition publishes its ordered inputs and bound surface
#: role in the same structured handoff as the table acquisition paths.
GDAS_MAX_FORECAST_HOUR = 9

#: The GDAS forecast-hour ladder this ArWen is certified for: NOMADS
#: publishes the assimilation cycle's short forecast hourly.
GDAS_PUBLISHED_HOURS = tuple(range(GDAS_MAX_FORECAST_HOUR + 1))

#: Sources that ride the certified GFS pgrb2.0p25 container.
GFS_CONTAINER_SOURCES = ("gfs", "gdas")

#: woof source name -> S3 object prefix and directory stem.
GFS_CONTAINER_PREFIX = {"gfs": "gfs", "gdas": "gdas"}
#: One NOMADS-subset GFS pgrb2.0p25 file carries exactly the 124 records
#: the fail-closed ``gfs_grib2_bridge`` selects (21 pressure levels x
#: {GHT, T, RH, U, V} + 11 surface/near-surface + 8 soil-layer records).
#: This is now the **certified tripwire** rather than the bar itself:
#: the bar applied to a download is derived from the live inventory, and
#: a disagreement with this constant is a loud, explicitly acknowledged
#: re-certification event.  See :mod:`woof.fetch_bars`.
GFS_SUBSET_RECORD_COUNT = fetch_bars.CERTIFIED_RECORD_BARS["gfs"]

#: What a GFS-container fetch takes from the whole-object hosts (the AWS
#: archive, and the operational server's own directory) when no mode is
#: named: the records the grib-filter crop asks for, copied unchanged by
#: the byte ranges of the ``.idx`` published beside each object.  The
#: hosted AWS policy and a cycle only the archive still keeps take it.
#: Measured 2026-10-29 12Z f000/f003/f006: 1,556 MB of whole
#: objects against 266.5 MB of selected records (124 per lead, 17 %).
#: An object whose host keeps no index is moved whole, and the log says
#: so; ``--mode full-file`` still asks for the whole objects.
GFS_RECORDS_MODE = "idx-subset"


#: wgrib2's ``.idx`` names for the GRIB2 (discipline, category, parameter)
#: codes a packaged NCEP mapping binds.  A mapping selector with a code
#: outside this table cannot be named in an index, so its source moves
#: whole objects rather than risk a selection that leaves a bound record
#: out.
NCEP_IDX_NAMES = {
    (0, 0, 0): "TMP", (0, 1, 0): "SPFH", (0, 1, 1): "RH",
    (0, 1, 22): "CLMR", (0, 1, 23): "ICMR", (0, 1, 24): "RWMR",
    (0, 1, 25): "SNMR", (0, 1, 32): "GRLE", (0, 2, 2): "UGRD",
    (0, 2, 3): "VGRD", (0, 3, 0): "PRES", (0, 3, 1): "PRMSL",
    (0, 3, 5): "HGT", (2, 0, 0): "LAND", (2, 0, 2): "TSOIL",
    (2, 0, 192): "SOILW", (10, 2, 0): "ICEC",
}


def mapped_record_levels(source: str) -> dict[str, set[str]] | None:
    """``{VAR: {LEVEL, ...}}`` a source's packaged mapping binds, by ``.idx`` name.

    Empty for a source with no packaged mapping (GFS decodes through
    ``gfs_grib2_bridge``, which reads the grib-filter's records).  None
    when a selector cannot be named (see :data:`NCEP_IDX_NAMES`).
    """

    adapter = source_adapters.get_source_adapter(source)
    if adapter.packaged_profile is None:
        return {}
    from woof.source_authorities import packaged_authorities

    mapping = json.loads(packaged_authorities(
        adapter.packaged_profile)["mapping"].read_bytes())
    vertical = (mapping.get("coordinates") or {}).get("vertical") or {}
    ladder = (vertical.get("levels") or ()
              if vertical.get("kind") == "pressure" and vertical.get("units") == "Pa"
              else ())
    wanted: dict[str, set[str]] = {}
    for field in (mapping.get("fields") or {}).values():
        for selector in field.get("selectors") or ():
            if selector.get("format") != "grib2":
                return None
            name = NCEP_IDX_NAMES.get((selector.get("discipline"),
                                       selector.get("category"),
                                       selector.get("parameter")))
            kind = selector.get("level_type")
            if name is None:
                return None
            if kind == 100:
                published = (field.get("published_levels") or {}).get("levels")
                pressures = published or ladder
                if not pressures:
                    return None
                levels = {f"{float(value) / 100.0:g} mb" for value in pressures}
            elif kind == 1:
                levels = {"surface"}
            elif kind == 101:
                levels = {"mean sea level"}
            elif kind == 103:
                levels = {f"{float(selector['level_value']):g} m above ground"}
            elif kind == 106 and selector.get("second_level_type") == 106:
                levels = {f"{float(selector['level_value']):g}-"
                          f"{float(selector['second_level_value']):g} m below ground"}
            else:
                return None
            wanted.setdefault(name, set()).update(levels)
    return wanted


def gfs_record_selection(source: str,
                         levels_hpa: tuple[float, ...]) -> dict | None:
    """The records a preparation reads, as an ``.idx`` record declaration.

    The first ``[VAR, LEVEL]`` pair of full-match alternations is exactly
    the grib-filter's variable-by-level cross product: a record is kept
    when its variable is one the filter asks for AND its level is one the
    filter asks for.  Both alternations come from the declaration the
    crop request and its record bar already read
    (:func:`woof.fetch_bars.nomads_selector_pairs` over
    :mod:`tools.download_gfs_native_subset`), so the two transports take
    the same records.  A source that prepares through a packaged mapping
    (GDAS) also keeps every record its mapping binds, one pair per
    variable: GDAS's mapping reads the five hydrometeor mixing ratios,
    and a GDAS start without them is refused at prepare ("lacks required
    fields ['cloud_ice_mixing_ratio', ...]", 2026-10-29).  The
    grib-filter asks for the same variables
    (:func:`tools.download_gfs_native_subset.nomads_variables` reads the
    same mapping), so the crop and the archive's records agree.  None
    when the mapping binds a record this table cannot name in an index.
    """

    import re

    from woof.record_index import validate_declaration
    from tools import download_gfs_native_subset as transport

    pairs = fetch_bars.nomads_selector_pairs(
        transport.nomads_variables(source), transport.NOMADS_LEVELS,
        tuple(levels_hpa))
    mapped = mapped_record_levels(source)
    if mapped is None:
        return None
    variables = sorted({variable for variable, _level in pairs})
    levels = sorted({level for _variable, level in pairs})
    select = [["|".join(re.escape(name) for name in variables),
               "|".join(re.escape(level) for level in levels)]]
    select += [[re.escape(name), "|".join(re.escape(level)
                                          for level in sorted(wanted))]
               for name, wanted in sorted(mapped.items())]
    return validate_declaration({
        "format": "ncep-idx",
        "select": select,
        "why": (f"the {source} grib-filter selection (every requested "
                "variable on every requested level, "
                f"{len(levels_hpa)} isobaric levels)"
                + (" and every record its packaged mapping binds"
                   if mapped else "")
                + ", by the .idx beside each pgrb2.0p25 object; the bytes "
                "are copied unchanged"),
    }, f"{source} records")

# HRRR CONUS coverage is NOT a constant here.  It is derived from the
# native Lambert grid definition -- the single source of truth in
# :func:`woof.ingest.hrrr_target.hrrr_coverage_envelope` -- via
# :func:`source_coverage_envelope` below.  The hand-held box this
# replaces (lat 21.1..52.7, lon -134.2..-60.8) was a second definition
# of coverage beside the one `woof domain` sized against, and the two
# disagreed in both directions: its 52.70 cap admitted latitudes north
# of the grid's real 52.6157 top, and refused the wizard's own emitted
# next command for a legal, source-coverable 3 km CONUS root.

#: The full ERA5 pressure-level ladder (hPa).  Requesting all 37 levels
#: keeps the template independent of any one experiment's p_top.
ERA5_PRESSURE_LEVELS_HPA = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250,
    300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850,
    875, 900, 925, 950, 975, 1000,
)

ERA5_REQUEST_NAME = "era5-cds-request.json"
#: The runnable retrieval written beside the request.  A printed snippet
#: has to be retyped, and every retype is a chance to lose the paths the
#: request just bound; a file is copied by running it.
ERA5_RETRIEVE_NAME = "era5-cds-retrieve.py"
ERA5_COMBINED_NAME = "era5-combined.grib"
#: The file each ERA5 provider PUBLISHES.  The two providers hand back two
#: containers -- the CDS a concatenated GRIB1 file, the keyless ARCO reader a
#: regular NetCDF one -- so the name of the published object is a function of
#: the provider and of nothing else.
ERA5_COMBINED_NAMES = {"cds": ERA5_COMBINED_NAME, "arco": "era5-combined.nc"}


def era5_combined_name(provider: str | None = None) -> str:
    """The file name an ERA5 fetch publishes for ``provider``.

    THE DEFECT THIS EXISTS TO END.  Every emitter of a ``[case_data]``
    table spelled the CDS name as a literal, and the ARCO provider
    publishes a different one.  A config written by ``woof domain
    --era5-provider arco`` therefore declared a forcing file its own
    ``[fetch]`` table could never produce: the download succeeded, wrote
    ``era5-combined.nc``, and the next command refused with "[fetch].out
    does not produce the file named by [case_data].forcing".  The door was
    dead by default and the only way through was to hand-edit the
    generated TOML.

    So the name is DERIVED here, once, from the same table the fetch
    publishes through, and every emitter asks instead of spelling.  The
    two publishers ask as well -- :mod:`woof.era5_acquisition` for the
    CDS container and :mod:`woof.era5_arco` for the ARCO one -- so there
    is no copy of the name to fall out of step with the table.  A third
    provider added to :data:`ERA5_COMBINED_NAMES` is carried by all of
    them with no further edit.
    """

    name = ERA5_COMBINED_NAMES.get(provider or "cds")
    if name is None:
        raise ValueError(
            f"era5_provider = {provider!r} publishes no ERA5 combined file, "
            "so no [case_data].forcing name can be derived for it. WOOF's "
            f"ERA5 providers are {sorted(ERA5_COMBINED_NAMES)}; select one "
            "of those with --era5-provider, or omit the flag for the CDS "
            "default.")
    return name


def era5_forcing_name_disagreement(declared, *, provider: str | None) -> str | None:
    """Why ``declared`` is not the file this provider's fetch will publish.

    THE BREAKAGE IT NAMES, and the reason it is called at AUTHORING time
    and not only at launch: a config whose declared forcing name and whose
    fetch recipe disagree is accepted by every loader, passes every
    ``[fetch]`` validator, downloads gigabytes, and only then refuses --
    at which point the bytes are on disk and the config is still wrong.
    The wizard already round-trips its ``[fetch]`` table through the real
    fetch validators before writing, for exactly this reason; this is the
    same guarantee for the one ``[case_data]`` key whose value the fetch
    determines.

    ``None`` when they agree, which includes a caller that declares no
    forcing at all -- that is a different config shape, not this defect.
    """

    from pathlib import Path

    expected = era5_combined_name(provider)
    names = [Path(str(item)).name for item in (declared or ())]
    if not names or all(name == expected for name in names):
        return None
    return (f"[case_data].forcing names {names[0]!r}, which an ERA5 fetch "
            f"with era5_provider = {provider or 'cds'!r} does not publish: "
            f"that provider writes {expected!r}. Keep both on the file the "
            "fetch produces, or select the provider whose container you "
            "declared.")

# ERA5 GRIB1 parameter expectations, grounded in what ingest consumes
# (woof/ingest/grib.py _CANONICAL_SPECS; woof/ingest/real.py requires
# TT/RH/GHT/UU/VV/PSFC/T2/D2-or-RH2/U10/V10; woof/ingest/soil.py requires
# LANDSEA/SKINTEMP plus the four ST/SM layers and reads SNOW_EC/SST/
# SEAICE; woof/ingest/horiz.py converts invariant geopotential to source
# orography).  Keys are (grib1_parameter, cds short name).
ERA5_REQUIRED_PRESSURE = {
    129: "z", 130: "t", 131: "u", 132: "v", 157: "r",
}
ERA5_REQUIRED_SURFACE = {
    134: "sp", 165: "10u", 166: "10v", 167: "2t", 168: "2d", 172: "lsm",
    235: "skt", 141: "sd",
    139: "stl1", 170: "stl2", 183: "stl3", 236: "stl4",
    39: "swvl1", 40: "swvl2", 41: "swvl3", 42: "swvl4",
}
#: Invariant geopotential doubles as the source orography.  woof's
#: ingest can substitute a per-domain source-orography supplement, so its
#: absence is reported as a failure with that escape hatch named.
ERA5_OROGRAPHY_PARAMETER = 129
ERA5_OPTIONAL_SURFACE = {151: "msl", 31: "ci", 34: "sst"}
#: Native CDS GRIB1 encodes soil layers as level type 112; the
#: CDO-normalized form flattens them to level type 1.  Ingest accepts
#: both (woof/ingest/grib.py _NATIVE_LEVEL_ALIASES), so the census does
#: too.
ERA5_SOIL_PARAMETERS = frozenset({139, 170, 183, 236, 39, 40, 41, 42})

_USER_AGENT = "gpuwm-fetch/1"


# ---------------------------------------------------------------------------
# Area / cycle / hours parsing
# ---------------------------------------------------------------------------

def _wrap_lon(value: float) -> float:
    """Wrap a longitude into [-180, 180) (west-edge convention)."""
    return (value + 180.0) % 360.0 - 180.0


def _wrap_lon_east(value: float) -> float:
    """Wrap a longitude into (-180, 180]: an east edge sitting exactly
    on the antimeridian reads as +180, not -180."""
    wrapped = _wrap_lon(value)
    return 180.0 if wrapped == -180.0 else wrapped


@dataclass(frozen=True)
class Area:
    """A geographic bounding box, south/west/north/east in degrees.

    ``lon_west > lon_east`` (after wrapping into [-180, 180)) denotes a
    box crossing the antimeridian -- the eastward walk from west to
    east passes 180E.
    """

    lat_south: float
    lon_west: float
    lat_north: float
    lon_east: float

    @property
    def crosses_antimeridian(self) -> bool:
        return _wrap_lon(self.lon_west) > _wrap_lon_east(self.lon_east)

    @property
    def longitude_span_degrees(self) -> float:
        """Eastward longitude width represented by the two stored edges."""

        span = (self.lon_east - self.lon_west) % 360.0
        if span == 0.0 and self.lon_east != self.lon_west:
            return 360.0
        return span

    @property
    def nomads_longitude_amplification(self) -> float | None:
        """Full-band/requested-span ratio when one NOMADS box widens."""

        box = self.as_nomads()
        if (box["left_lon"] == 0.0 and box["right_lon"] == 360.0
                and self.longitude_span_degrees < 360.0):
            return 360.0 / self.longitude_span_degrees
        return None

    def as_manifest(self) -> dict[str, float]:
        return {
            "lat_south": self.lat_south, "lon_west": self.lon_west,
            "lat_north": self.lat_north, "lon_east": self.lon_east,
        }

    def as_cds(self) -> list[float]:
        """CDS ``area`` convention: [north, west, south, east].

        Longitudes are wrapped into the signed convention; the CDS API
        reads ``west > east`` as an antimeridian-crossing box.
        """
        return [self.lat_north, _wrap_lon(self.lon_west),
                self.lat_south, _wrap_lon_east(self.lon_east)]

    def as_nomads(self) -> dict[str, float]:
        """NOMADS subregion in [0, 360] longitudes with left < right.

        A box crossing the prime meridian cannot be expressed with one
        ``0 <= left < right <= 360`` request; it widens to the full
        longitude band.  The fetch path must disclose that amplification.
        """
        left = self.lon_west % 360.0
        right = self.lon_east % 360.0
        if right == 0.0 and self.lon_east != 0.0:
            right = 360.0
        if not left < right:
            left, right = 0.0, 360.0
        return {"left_lon": left, "right_lon": right,
                "bottom_lat": self.lat_south, "top_lat": self.lat_north}


def parse_area(raw: str) -> Area:
    """Parse ``lat0,lon0,lat1,lon1`` (corner order free) into an Area.

    A longitude pair spanning more than 180 degrees is read as the
    complementary box crossing the antimeridian: ``170,-170`` (or
    ``-170,170``) is the 20-degree Pacific box over 180E, never the
    340-degree box that excludes it.  Boxes genuinely wider than 180
    degrees must be requested as the full band (span 360) or split.
    """

    parts = raw.split(",")
    if len(parts) != 4:
        raise ValueError(
            "--area must be lat0,lon0,lat1,lon1 (two corners, degrees)")
    try:
        lat0, lon0, lat1, lon1 = (float(part) for part in parts)
    except ValueError as error:
        raise ValueError("--area corners must be decimal degrees") from error
    lat_south, lat_north = sorted((lat0, lat1))
    lon_west, lon_east = sorted((lon0, lon1))
    if not (-90.0 <= lat_south and lat_north <= 90.0):
        raise ValueError("--area latitudes must lie within [-90, 90]")
    if not (-180.0 <= lon_west and lon_east <= 360.0):
        raise ValueError("--area longitudes must lie within [-180, 360]")
    if lat_south == lat_north or lon_west == lon_east:
        raise ValueError("--area must span a nonzero box")
    span = lon_east - lon_west
    if 180.0 < span < 360.0:
        # The complement crossing the antimeridian is the intended box.
        lon_west, lon_east = lon_east, lon_west
    return Area(lat_south, lon_west, lat_north, lon_east)


def area_from_point(raw_point: str, radius_km: float) -> Area:
    """Convert ``--point lat,lon --radius-km N`` to a bounding Area."""

    parts = raw_point.split(",")
    if len(parts) != 2:
        raise ValueError("--point must be lat,lon in decimal degrees")
    try:
        lat, lon = (float(part) for part in parts)
    except ValueError as error:
        raise ValueError("--point must be lat,lon in decimal degrees") from error
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 360.0:
        raise ValueError("--point lies outside [-90,90] x [-180,360]")
    if not math.isfinite(radius_km) or radius_km <= 0.0:
        raise ValueError("--radius-km must be positive")
    km_per_degree = 111.195  # mean meridional degree
    dlat = radius_km / km_per_degree
    cos_lat = math.cos(math.radians(lat))
    if cos_lat * km_per_degree * 360.0 <= 2.0 * radius_km or cos_lat <= 0.0:
        raise ValueError(
            "--radius-km circles the pole at this latitude; "
            "pass an explicit --area instead")
    dlon = radius_km / (km_per_degree * cos_lat)
    lat_south = max(-90.0, lat - dlat)
    lat_north = min(90.0, lat + dlat)
    # Wrap the walked-out edges into the signed convention; near the
    # antimeridian this produces a lon_west > lon_east crossing box.
    return Area(lat_south, _wrap_lon(lon - dlon),
                lat_north, _wrap_lon_east(lon + dlon))


#: Decimal places of an emitted ``--area`` hint: the wizard's printed
#: command and its [fetch] table both carry this fixed-point form, and
#: :func:`area_bounds_inward` quantizes coverage bounds to the same
#: precision so a formatted hint can never round back out of coverage.
AREA_HINT_DECIMALS = 2


def source_coverage_envelope(source: str
                             ) -> tuple[float, float, float, float] | None:
    """``(south, west, north, east)`` the source's native grid actually
    covers, or ``None`` for a source with no coverage box (global).

    Data, not policy, and ONE definition per source: the source's registry
    row declares its native grid (:mod:`woof.source_coverage`), and both
    sides of the area contract -- this module's ``--area`` gate and
    ``woof domain``'s suggested fetch box -- consume it here, so they
    cannot drift apart the way the retired hand-held CONUS box did.  A
    source with no declared window is global and gets no bound; a name the
    registry does not know gets none either, because this function answers
    "how far does it reach", not "does it exist" (the callers' own
    validators own that question and phrase it better).

    The import is deferred so a base install stays importable without the
    registry's projection machinery loaded.
    """

    from woof.source_adapters import get_source_adapter
    from woof.source_coverage import window_envelope

    try:
        adapter = get_source_adapter(source)
    except ValueError:
        return None
    return window_envelope(adapter.coverage_window)


def fetch_front_door_sources() -> tuple[str, ...]:
    """The sources ``woof fetch`` can actually download today.

    Named as a seam because another front door has to ask: `woof domain`
    emits a ``[fetch]`` hint table only for a source whose bytes this
    module can go and get, and prints the accurate acquisition route for the
    rest.  Before the seam existed the wizard had no way to ask, so it
    simply did not offer the other sources at all.

    DERIVED, never listed.  There are exactly two ways a fetch runs: a row
    in the packaged acquisition-route document, or one of the four
    hand-written transports that predate it -- which is precisely what
    :func:`woof.fetch_routes.all_fetchable_sources` answers, and it is
    the same answer ``woof fetch`` itself dispatches on.  The seam spelled
    the four legacy names by hand until 2026-08-17, so the ten routes that
    landed that day were invisible here: `woof domain --source rrfs`
    printed "stage the bytes yourself" for a model whose route was live,
    and a hand-written ``[fetch]`` table naming it was refused at config
    load as an unknown source.  A future model's row now reaches this door
    with the row, which is the whole point of the route table being data.
    """

    owners = {row.source_id for row in source_adapters.source_adapters() if row.acquisition_owner}
    return tuple(sorted(set(fetch_routes.all_fetchable_sources()) | owners))


def native_cf_fetch_contract(source: str) -> str | None:
    """Capability probe for annual CF acquisition and bound native preparation."""
    from woof.cf_archive_fetch import contract
    return contract(fetch_routes.canonical_source(source))


def fetch_accepts_area(source: str) -> bool:
    """Can ``woof fetch --source SOURCE`` be handed a crop box?

    The second half of the same seam, and derived from the same split.
    Only the four hand-written transports subset: GFS/GDAS through the
    NOMADS grib-filter, HRRR through its ``.idx`` byte ranges, ERA5
    through the retrieval request.  A table route takes whole published
    objects because there is no subsetting service in front of them, and
    ``woof fetch`` refuses ``--area`` on one by name -- so a front door
    that emits an ``area`` hint for a routed source prints a step 1 that
    exits 2.  The crop for those sources happens at `woof prep`, where
    the namelist geometry is the crop.
    """

    from woof.cf_archive_fetch import sources
    adapter = source_adapters.get_source_adapter(source)
    if adapter.acquisition_owner:
        from importlib import import_module
        return bool(getattr(import_module(adapter.acquisition_owner), "FETCH_ACCEPTS_AREA", False))
    return fetch_routes.canonical_source(source) in (
        fetch_routes.LEGACY_ROUTE_SOURCES + sources())


def fetch_accepts_cadence(source: str) -> bool:
    """Can ``woof fetch --source SOURCE`` be handed a ``--cadence``?

    The third question in the same seam as
    :func:`fetch_front_door_sources` and :func:`fetch_accepts_area`, and
    asked by everything that writes or reads a ``cadence``: the fetch's
    own argument validation, the ``[fetch]`` table's config-load check,
    and the front door that emits such a table.
    """

    row = source_adapters.get_source_adapter(fetch_routes.canonical_source(source))
    return (not row.fetch_entire_window or
            (row.forcing_interval_seconds is not None and row.forcing_interval_seconds % 3600 == 0))


def validate_fetch_cadence(source: str, cadence: int | None) -> None:
    """Keep native complete-window acquisition on its declared frame spacing."""
    if cadence is None:
        return
    row = source_adapters.get_source_adapter(fetch_routes.canonical_source(source))
    if row.fetch_entire_window and cadence * 3600 != row.forcing_interval_seconds:
        raise ValueError(cadence_inapplicable_refusal(source))


def cadence_inapplicable_refusal(source: str) -> str:
    """Why a cadence cannot apply to SOURCE, and what to write instead.

    The publisher's spacing is read off the registry row rather than
    written here, so the sentence stays true for the next source that
    declares acquisition of its entire native window.
    """

    spacing_h = int(
        source_adapters.source_forcing_interval_seconds(source) // 3600)
    spacing = "hourly" if spacing_h == 1 else f"{spacing_h}-hourly"
    return layered(
        f"{source} is {spacing} and the fetch takes every frame it "
        "publishes inside the window.\n"
        f"  What to do: use cadence {spacing_h}, or omit cadence and set the window with --hours.",
        "  Why: a cadence is the spacing this fetch would subsample the "
        "publisher's ladder at, and a preparation reads boundary "
        "conditions at every frame this source publishes inside the "
        "window.  Skipping any of them would hand the preparation a "
        "series with holes in it, so there is no spacing to accept here "
        "and the value is refused rather than ignored.")


def preparation_cadence_refusal(source: str, cadence: int) -> str | None:
    """Why SOURCE's preparation cannot take boundaries CADENCE hours apart.

    ``None`` when it can, and for every source whose preparation is not a
    packaged mapped profile.  Asked by :func:`validate_fetch_hints`, so
    ``woof domain``, the ``[fetch]`` table's config-load check and
    ``woof fetch`` all put the question the decode puts, in its own
    function (:func:`woof.source_authorities.boundary_interval_refusal`)
    and against the same packaged mapping.  A cadence the decode refuses
    used to be accepted at every door and written into ``interval_seconds``,
    and the refusal came after the whole download.
    """

    from woof.source_authorities import (
        BOUNDARY_MULTIPLES_KEY, boundary_interval_refusal,
        boundary_interval_takes, packaged_mapping_target)

    row = source_adapters.get_source_adapter(fetch_routes.canonical_source(source))
    if row.runner != "mapped_composition_v1" or not row.packaged_profile:
        return None
    target = packaged_mapping_target(row.packaged_profile)
    if boundary_interval_refusal(target, cadence * 3600) is None:
        return None
    spacing_h = int(target["boundary_interval_seconds"]) / 3600
    takes = boundary_interval_takes(target)
    return layered(
        f"cadence {cadence} gives {row.source_id} boundaries {cadence} h "
        f"apart, and its preparation takes {takes}.\n"
        f"  What to do: use cadence {spacing_h:g}, or omit cadence.",
        f"  Why: the packaged {row.packaged_profile} mapping declares "
        f"boundary_interval_seconds = {int(target['boundary_interval_seconds'])}"
        + (f" with {BOUNDARY_MULTIPLES_KEY}"
           if target.get(BOUNDARY_MULTIPLES_KEY) is True else "")
        + ", and the decode refuses a series at any other spacing.  Accepting "
          "this cadence here would download the whole window and then "
          "refuse it at preparation.")


def area_bounds_inward(envelope: tuple[float, float, float, float],
                       decimals: int = AREA_HINT_DECIMALS
                       ) -> tuple[float, float, float, float]:
    """ENVELOPE rounded INWARD to DECIMALS places.

    The tightest ``(south, west, north, east)`` box that both lies
    inside the envelope and survives fixed-point formatting: a value
    clamped to these bounds and printed at the same precision parses
    back inside the true envelope, whereas clamping to the exact
    envelope and then rounding to the printed form can cross it (e.g.
    52.615653 prints as 52.62, north of the grid).
    """

    south, west, north, east = envelope
    scale = 10.0 ** decimals
    return (math.ceil(south * scale) / scale,
            math.ceil(west * scale) / scale,
            math.floor(north * scale) / scale,
            math.floor(east * scale) / scale)


#: Per-source remedy sentence for a coverage refusal.  Source-specific
#: words live in data, next to the sources this module already names.
_COVERAGE_REMEDY = {
    "hrrr": "use --source gfs for domains outside HRRR's CONUS coverage",
}


def validate_fetch_area(source: str, area: Area) -> None:
    """The per-source ``--area`` coverage gate ``woof fetch`` applies.

    One seam for every front door that wants to prove an area before
    paying for a download (the wizard proves each emitted hint through
    it).  A source with a coverage envelope refuses, fail-closed, any
    box extending beyond what its native grid carries; global sources
    accept every parseable box.  An antimeridian-crossing box cannot
    lie inside a non-crossing envelope, so it is refused too (the old
    corner-order arithmetic waved a Pacific-crossing box through the
    HRRR gate).
    """

    envelope = source_coverage_envelope(source)
    if envelope is None:
        return
    south, west, north, east = envelope
    if (area.crosses_antimeridian
            or area.lat_south < south or area.lat_north > north
            or area.lon_west < west or area.lon_east > east):
        # Printed bounds are quantized INWARD so the remedy box the
        # message names is itself accepted.
        say_s, say_w, say_n, say_e = area_bounds_inward(envelope)
        remedy = _COVERAGE_REMEDY.get(
            source, "choose a source whose coverage includes the request")
        raise ValueError(
            f"requested area extends beyond {source.upper()} coverage: "
            f"the native grid's own lat/lon envelope is "
            f"lat {say_s:.2f}..{say_n:.2f}, lon {say_w:.2f}..{say_e:.2f} "
            f"(derived from the grid definition, not a hand-held box); "
            f"{remedy}")


def parse_cycle(raw: str, source: str) -> datetime:
    """Parse ``YYYY-MM-DDTHH`` and enforce the source's cycle cadence."""

    try:
        cycle = datetime.strptime(raw, "%Y-%m-%dT%H")
    except ValueError as error:
        raise ValueError(
            f"--cycle {raw!r} must be YYYY-MM-DDTHH (UTC) or 'latest'"
        ) from error
    if source in GFS_CONTAINER_SOURCES and cycle.hour not in GFS_CYCLE_HOURS:
        raise ValueError(
            f"{source.upper()} cycles run at 00/06/12/18 UTC only")
    return cycle


def _forecast_start_hour(start: int | None) -> int:
    """The lead a fetch window begins at: 0, or a checked positive lead.

    A window that starts at f000 is the analysis and its short forecast;
    a window that starts at f{K} is the run whose initial condition is
    GFS's own K-hour forecast.  Both are legitimate; the second is what
    a user wanting the f174..f240 window needs, and fetching f000..f240
    to reach it is the workaround this closes.

    The cadence spaces the leads from this one; it does not restrict
    where the window begins.  Whether each lead of the window is
    published is the source ladder's question, asked by its caller.
    """

    if start is None:
        return 0
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError(
            "--forecast-start-hour must be a nonnegative forecast lead")
    return start


def gfs_forecast_hours(hours: int, cadence: int,
                       start: int | None = None) -> tuple[int, ...]:
    """The f{start}..f{start+NNN} ladder the GFS series contract accepts.

    ``start`` defaults to 0, which is the f000..fNNN ladder every prior
    release fetched, byte for byte.  ``--hours`` stays what it always
    was: the LENGTH of the window, not its final lead, so a window is
    described the same way wherever it begins.
    """

    if type(cadence) is not int or cadence <= 0:
        raise ValueError("--cadence must be a positive whole number of hours")
    if type(hours) is not int or hours < 0 or hours % cadence:
        raise ValueError(
            f"--hours must be a nonnegative integer multiple of the {cadence} h cadence")
    # A uniform window can start on any actual source lead, even when the
    # lead is not a multiple of the chosen spacing (for example f001/f003).
    start = _forecast_start_hour(start)
    if start + hours > GFS_MAX_FORECAST_HOUR:
        raise ValueError(
            f"The GFS publication horizon is f{GFS_MAX_FORECAST_HOUR}; this window ends "
            f"at f{start + hours:03d}. Shorten the window or start earlier.")
    leads = tuple(range(start, start + hours + 1, cadence))
    if any(lead > GFS_HOURLY_MAX_FORECAST_HOUR
           and lead % GFS_PUBLISHED_CADENCES_H[-1] for lead in leads):
        raise ValueError(gfs_cadence_break_refusal(start, start + hours, cadence))
    return leads


def gdas_capability_refusal(requested_hour: int) -> str:
    """Why a GDAS request past the published ladder is refused, and what to do.

    Publication wording: the limit is the assimilation cycle's own
    output, not a claim about what has been proved here.  Written in the
    two halves this project layers everywhere -- ``What to do`` is the
    action, ``Why`` is the mechanism -- so the remedy reaches a reader at
    the default width and the mechanism waits for ``--explain`` instead
    of arriving on top of it.
    """

    return layered(
        f"GDAS publishes f{GDAS_PUBLISHED_HOURS[0]:03d}.."
        f"f{GDAS_MAX_FORECAST_HOUR:03d}: the assimilation cycle's analysis "
        "and the short forecast that carries it to the next cycle.  "
        f"f{requested_hour:03d} names no object it publishes.\n"
        f"  What to do: stay inside --hours 0..{GDAS_MAX_FORECAST_HOUR}, or "
        f"use --source gfs, which publishes leads through "
        f"f{GFS_MAX_FORECAST_HOUR}.",
        f"  Why: a lead past f{GDAS_MAX_FORECAST_HOUR:03d} is not a cycle "
        "still uploading -- the object is never written, on this cycle or "
        "any other.  Starting the fetch anyway would move the same refusal "
        "behind a download that cannot complete, so it is made here, before "
        "any bytes.")


def gdas_cadence_refusal(hours: int, cadence: int, start: int) -> str:
    """Why a GDAS window and its cadence cannot both be served.

    What is accepted is derived from :data:`GDAS_PUBLISHED_HOURS`, so a
    ladder change moves this refusal with it rather than leaving a
    literal behind to contradict it.
    """

    published = set(GDAS_PUBLISHED_HOURS)
    accepted = tuple(
        step for step in range(1, hours + 1)
        if not hours % step
        and set(range(start, start + hours + 1, step)) <= published)
    last = start + (hours // cadence) * cadence
    return layered(
        f"--cadence {cadence} does not divide --hours {hours}, so the "
        f"f{start:03d}..f{start + hours:03d} window would stop at "
        f"f{last:03d} and the run would be bounded by a shorter series "
        "than the one requested.\n"
        "  What to do: pass --hours a whole multiple of the cadence, or "
        + (f"--cadence {' or '.join(str(step) for step in accepted)}."
           if accepted else
           f"a window inside f{GDAS_PUBLISHED_HOURS[0]:03d}.."
           f"f{GDAS_MAX_FORECAST_HOUR:03d}."),
        "  Why: the cadence is the spacing of the boundary frames and the "
        "window's final hour is a frame like any other, so a cadence that "
        "does not divide the window drops it in silence -- the fetch "
        "succeeds, the manifest is complete for what it holds, and the "
        "forecast simply ends early.  GDAS publishes "
        f"f{GDAS_PUBLISHED_HOURS[0]:03d}..f{GDAS_MAX_FORECAST_HOUR:03d} at "
        f"{GDAS_PUBLISHED_HOURS[1] - GDAS_PUBLISHED_HOURS[0]}-hour spacing, "
        "so every whole-hour cadence that divides the window is on its "
        "ladder.")


def container_handoff_binding(source: str) -> str:
    """The in-band surface role a container source's prep handoff binds.

    ONE function for both doors: the fetch's own plan review calls it
    before a byte moves, and the publisher calls it again as it writes
    the handoff, so the two can never disagree about whether this source
    is preparable from what the fetch brings.  It used to be asked only
    inside the publisher, which put the refusal after the whole download.
    """

    role = fetch_routes.in_band_supplement_role(source)
    if role is None:
        raise ValueError(layered(
            f"{source} declares no in-band surface binding, so a fetch "
            "would bring its bytes and then have no preparation handoff to "
            "publish with them.\n"
            f"  What to do: prepare it explicitly -- `woof prep --source "
            f"{source}` with the ordered --input-list and one --supplement "
            "ROLE=PATH for each role its contract names -- or fetch a "
            "source whose packaged composition binds its surface fields in "
            "band.",
            "  Why: this container writes its preparation arguments from "
            "the packaged composition's terrain role, and that role is "
            "bindable here only when it selects its fields from the same "
            "input files this fetch downloads.  A composition naming a "
            "separate donor leaves the publisher nothing to bind, and the "
            "refusal belongs before the transfer rather than after it."))
    return role


#: What NOMADS keeps, roughly, for the 0.25-degree pgrb2 product.
#:
#: Approximate on purpose: it is a rolling window NCEP manages, and this
#: number is only ever used to name the shape of a refusal the transport
#: has already made -- never to predict one.  DATA.md states the same
#: figure.
NOMADS_RETENTION_DAYS = 10


def nomads_reach_refusal(source: str, cycle: datetime, hour: int,
                         error: HTTPError) -> str:
    """Why the NOMADS grib filter would not serve one cycle.

    Two probes disagree in this product, and a user in the gap between
    them met the disagreement as a 42-line ``urllib.error.HTTPError``
    traceback.  Cycle completeness is checked against the AWS S3 archive,
    which holds years; the download is the NOMADS grib-filter crop, which
    holds about :data:`NOMADS_RETENTION_DAYS` days.  Every cycle in
    between passes the check and then dies in the transport -- measured
    on one node: 7 days old fetched, 10 days old 404, 31 days old 403.

    The archive's own answer is the ground truth, so it is translated
    here rather than predicted by a second probe: whatever NCEP's
    retention is today, this is what it just said.
    """

    age = datetime.now(timezone.utc).replace(tzinfo=None) - cycle
    days = age.total_seconds() / 86400.0
    if error.code in (403, 404) and days > 1.0:
        return layered(
            f"{source.upper()} cycle {cycle:%Y-%m-%dT%H}Z is "
            f"{days:.0f} days old and the NOMADS grib filter no longer "
            f"serves it (HTTP {error.code} for f{hour:03d}).\n"
            "  What to do: fetch a cycle inside NOMADS' rolling window "
            f"-- about {NOMADS_RETENTION_DAYS} days -- or use "
            "--source hrrr, whose S3 archive this WOOF reads directly.",
            "  Why: cycle completeness is probed against the AWS S3 "
            "archive, which holds years, while the GFS/GDAS download is "
            "the NOMADS grib-filter crop, which holds days.  A cycle in "
            "between passes the probe and is then refused by the "
            "transport, which is what this is.  Reading the raw S3 "
            "objects instead is not a substitute: they are GRIB2 "
            "template 5.3 (complex packing), and the certified bridge "
            "admits 5.0 only -- see docs/public/DATA.md.")
    return layered(
        f"NOMADS returned HTTP {error.code} for {source.upper()} cycle "
        f"{cycle:%Y-%m-%dT%H}Z f{hour:03d} ({error.reason}).",
        f"  The request URL was {error.url}")


def hrrr_reach_refusal(host: str, cycle: datetime, hour: int, kind: str,
                       error: URLError, *,
                       now: datetime | None = None) -> str:
    """Why one HRRR host did not hand over one product, in plain words.

    The Python range transport lets urllib's own error out, and it left
    ``woof fetch`` as a raw ``HTTPError`` traceback.  When the host's
    declared retention no longer covers the cycle's age, the refusal
    says so and names the hosts that still keep a cycle that old.
    """

    what = f"HRRR cycle {cycle:%Y-%m-%dT%H}Z f{hour:02d} {kind}"
    kept = "files already verified on disk are kept"
    if not isinstance(error, HTTPError):
        return layered(
            f"could not reach {host} for {what}.\n"
            f"  What to do: check the network and re-run the same "
            f"command; {kept}.",
            f"  The network library said: {error.reason}")
    rungs = fetch_endpoints.ladder("hrrr")
    served = next((rung for rung in rungs if rung.name == host), None)
    age = fetch_endpoints.cycle_age_hours(cycle, now)
    if served is not None and not served.covers(age):
        keepers = [rung.name for rung in rungs
                   if rung.name != host and rung.covers(age)]
        said = (f"{host} answered HTTP {error.code} for {what}: it keeps "
                f"only about the newest {served.retention_hours:g} h of "
                f"HRRR cycles, and this one is {age:.0f} h old")
        if keepers:
            return layered(
                f"{said}; {' and '.join(keepers)} still "
                f"{'keeps' if len(keepers) == 1 else 'keep'} it.\n"
                f"  What to do: pass --transport {keepers[0]}, or leave "
                "--transport off and the fetch asks the host that has it; "
                f"{kept}.",
                f"  The request URL was {error.url}")
        return layered(
            f"{said}, and no other HRRR host keeps a cycle that old.\n"
            "  What to do: fetch a newer cycle.",
            f"  The request URL was {error.url}")
    return layered(
        f"{host} answered HTTP {error.code} ({error.reason}) for {what}.\n"
        f"  What to do: re-run the same command; {kept}.",
        f"  The request URL was {error.url}")


def container_default_cadence(source: str) -> int:
    """The spacing, in hours, a container fetch takes when none is named.

    The registry row's own forcing interval, which is also the spacing the
    front door writes into ``[fetch]`` and the one the source's
    preparation declares.  It was a literal 3 for both container sources,
    so a bare GDAS fetch took a 3 h ladder the row does not declare and
    refused ``--hours 1`` over a cadence nobody had asked for.
    """

    return int(source_adapters.source_forcing_interval_seconds(
        fetch_routes.canonical_source(source)) // 3600)


def gdas_forecast_hours(hours: int, cadence: int | None = None,
                        start: int | None = None) -> tuple[int, ...]:
    """The GDAS ladder inside the published span, or a refusal.

    The cadence is checked the way :func:`gfs_forecast_hours` checks its
    own -- a whole number of hours that divides the window -- because a
    cadence that does not divide it truncated the series in silence: the
    fetch succeeded, the manifest was internally complete, and the run
    was bounded by a window shorter than the one asked for.  What is
    accepted is derived from :data:`GDAS_PUBLISHED_HOURS` rather than
    written here as a literal, so the ladder stays the only place the
    publisher's spacing is recorded.  An omitted cadence is the
    registry row's (:func:`container_default_cadence`).
    """

    if cadence is None:
        cadence = container_default_cadence("gdas")
    if isinstance(cadence, bool) or not isinstance(cadence, int) or cadence < 1:
        raise ValueError("--cadence must be a positive whole number of hours")
    if isinstance(hours, bool) or not isinstance(hours, int) or hours < 0:
        raise ValueError("--hours must be a nonnegative integer")
    # As on the GFS ladder, the window may begin on any published lead:
    # f001 with a 3 h cadence is f001 and f004, both of which GDAS writes.
    start = _forecast_start_hour(start)
    if start + hours > GDAS_MAX_FORECAST_HOUR:
        raise ValueError(gdas_capability_refusal(start + hours))
    if hours == 0:
        return (start,)
    if hours < cadence or hours % cadence:
        raise ValueError(gdas_cadence_refusal(hours, cadence, start))
    ladder = tuple(range(start, start + hours + 1, cadence))
    off = [hour for hour in ladder if hour not in GDAS_PUBLISHED_HOURS]
    if off:
        raise ValueError(gdas_capability_refusal(off[0]))
    return ladder


def container_forecast_hours(source: str, hours: int,
                             cadence: int | None = None,
                             start: int | None = None) -> tuple[int, ...]:
    """The published ladder for one GFS-container window.

    The dispatcher BOTH doors call: ``woof fetch``'s own argument
    validation and the ``[fetch]`` table's config-load check ask this
    one function, so a window accepted when a config is loaded cannot be
    refused when it is fetched.  The config door used to plan every
    container source on the GFS ladder, and only when the cadence was
    one of two values it named itself, so a GDAS table carrying any
    other cadence reached the download unplanned.
    """

    if source not in GFS_CONTAINER_SOURCES:
        raise ValueError(f"container_forecast_hours serves "
                         f"{GFS_CONTAINER_SOURCES}, not {source!r}")
    if source == "gdas":
        return gdas_forecast_hours(hours, cadence, start)
    return gfs_forecast_hours(
        hours, container_default_cadence(source) if cadence is None else cadence,
        start)


def hrrr_forecast_hours(hours: int, cycle: datetime,
                        start: int | None = None) -> tuple[int, ...]:
    """The contiguous f{start}..f{start+NN} window, checked against the horizon.

    ``start`` defaults to 0, which is the f00..fNN ladder every prior
    release fetched, byte for byte.  ``--hours`` stays what it always
    was -- the LENGTH of the window, not its final lead -- so a window is
    described the same way here as on the GFS and GDAS ladders above.

    ``_forecast_start_hour`` is what checks the value, so a negative or
    non-integer lead is refused in the same words on every source.
    """

    from woof.hrrr_forecast import validate_hrrr_source_forecast_hours

    if isinstance(hours, bool) or not isinstance(hours, int) or hours < 0:
        raise ValueError("--hours must be a nonnegative integer")
    start = _forecast_start_hour(start)
    return validate_hrrr_source_forecast_hours(
        range(start, start + hours + 1), cycle=cycle, allow_single_frame=True)


# ---------------------------------------------------------------------------
# Latest-cycle resolution (anonymous S3 HEAD probes)
# ---------------------------------------------------------------------------

def _head_ok(url: str) -> bool:
    """Availability probe, governed when it is aimed at NOMADS.

    ``--wait-for`` polls this every 30 s per product per host,
    ``--transport auto`` probes a whole window with it, and the
    throughput selection asks it once per object, so it is a real
    request stream and belongs under the same node-wide pacer as the
    payload transfers -- not beside them.  Probes at S3 pass straight
    through, unpaced.

    ONE implementation, in :mod:`woof.fetch_endpoints`, because the
    table routes ask the same question of the same hosts: two HEADs
    with two timeouts and two error vocabularies would eventually
    disagree about whether a host has an object, and the two halves of
    this package would then choose different hosts for the same file.
    """

    return fetch_endpoints.object_available(url)


def _head_answer(url: str) -> bool | None:
    """The publication question for one object: True, False or None.

    False only when the host said the object is not there (404 or 410);
    None when the host could not be heard even after asking again (a
    timeout, a refused connection, a throttle).  A named cycle's check
    (:func:`cycle_publication_check`) asks this rather than
    :func:`_head_ok`, because there a timeout read as "not there"
    refused a published start as "not published yet" (GS-05: three of
    about 170 HEADs timed out once, and the same URLs answered 200 a
    few seconds later).
    """

    return fetch_endpoints.settled_object_answer(url)


def objects_published(urls, probe=_head_ok, *, workers: int | None = None) -> bool:
    """True when ``probe`` answers every one of ``urls`` present.

    The question every publication check asks of one endpoint: a cycle
    counts only when every object of its final lead is there, and no
    URLs at all is not a published cycle.  The objects are asked as
    :func:`_rung_answer` asks them, side by side; an object the host
    could not be heard about is not counted present here.
    """

    return _rung_answer(urls, probe, workers=workers)[0] is True


def _rung_answer(urls, probe, *, workers: int | None = None
                 ) -> tuple[bool | None, str | None]:
    """One rung's answer for ``urls``, and the URL that decided it.

    ``probe`` answers True, False, or None when the host could not be
    heard; a probe that answers only True or False is read as it always
    was.  True when every object is there, and no URLs at all is not a
    published cycle (False).

    The objects are asked side by side, as many at once as the fetch's
    own transfers (:data:`woof.fetch_pool.DEFAULT_FILE_WORKERS`) and
    never more than a host's cap in the table allows, and every NOMADS
    request still passes the node-wide governor, so no host sees more in
    flight than a download already puts there.  Asked one after another,
    the final lead of a GEM cycle (about 174 objects) or an ICON-EU cycle
    (about 127) held a run's start for one to three minutes before a
    byte moved.

    Once an object is not answered present nothing more is sent; the
    HEADs already in flight finish on their own.  The rung is False when
    an object the host said is not there is among the answers in hand by
    then, and otherwise None at the object it could not be heard about,
    since nothing after it would change the rung from "not heard" to
    "holds them all".
    """

    urls = tuple(dict.fromkeys(urls))
    if not urls:
        return False, None
    width = workers or fetch_pool.DEFAULT_FILE_WORKERS
    for host in {fetch_pool.host_key(url) for url in urls}:
        width = fetch_pool.host_worker_cap(host, width)
    width = min(width, len(urls))
    if width <= 1:
        for url in urls:
            found = probe(url)
            if found is None:
                return None, url
            if not found:
                return False, url
        return True, None
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    from itertools import islice

    order = {url: index for index, url in enumerate(urls)}
    pool = ThreadPoolExecutor(max_workers=width, thread_name_prefix="gpuwm-publication-probe")
    waiting = iter(urls)
    flying: dict = {}
    try:
        flying = {pool.submit(probe, url): url for url in islice(waiting, width)}
        while flying:
            done, _ = wait(flying, return_when=FIRST_COMPLETED)
            answers = sorted(((flying.pop(future), future.result()) for future in done),
                             key=lambda pair: order[pair[0]])
            missing = next((url for url, found in answers
                            if found is not None and not found), None)
            if missing is not None:
                return False, missing
            unheard = next((url for url, found in answers if found is None), None)
            if unheard is not None:
                return None, unheard
            flying.update({pool.submit(probe, url): url
                           for url in islice(waiting, len(done))})
        return True, None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _probe_object_ladders(ladder, *, keys, source: str,
                          pinned: str | None, workers: int | None,
                          progress, probe=None
                          ) -> dict[str, tuple]:
    """Per key, the endpoint order its transfer will actually walk.

    Retention decides which hosts are ASKED; throughput decides which
    of them should serve, and the archive earns that only for an object
    it provably already holds.  The measured cost of not asking: at
    peak hours the operational server paced whole-file transfers at
    about 3 MB/s per file, so a 3.4 GB request took ~20 min where the
    archive had served the same volume in ~3.

    The probes run AHEAD of the transfers, through the same pool and
    under the same per-host caps, so none of them ever waits behind a
    download.  Promotion is a reorder: every rung stays behind the
    chosen one, so fall-through and the whole-ladder refusal are
    untouched, and a probe that 404s or throttles costs the transfer
    nothing.

    Returns only the keys whose order CHANGED; everything absent keeps
    the ladder unchanged.
    """

    if pinned is not None or not keys:
        return {}
    if not fetch_endpoints.transfer_probes(ladder):
        return {}
    if probe is None:
        probe = _head_ok
    preferred = fetch_endpoints.transfer_probes(ladder)[0]

    def ask(key: str) -> dict:
        return {"key": key, "ladder": fetch_endpoints.transfer_ladder(
            ladder, (key,), probe=probe)}

    entries, _receipt = fetch_pool.run_transfers(
        [fetch_pool.TransferJob(name=key, url=preferred.url(key),
                                action=functools.partial(ask, key))
         for key in keys],
        workers=fetch_pool.resolve_file_workers(workers))
    promoted = {entry["key"]: entry["ladder"] for entry in entries
                if entry["ladder"][0] is not ladder[0]}
    if promoted:
        progress(
            f"fetch {source}: mirrored: taking the archive for throughput "
            f"-- {len(promoted)} of {len(keys)} object"
            f"{'' if len(keys) == 1 else 's'} "
            f"{'is' if len(promoted) == 1 else 'are'} already on "
            f"{preferred.name} ({preferred.host})"
            + (f"; the rest from {ladder[0].name}, which publishes before "
               "the mirrors" if len(promoted) < len(keys) else ""))
    else:
        progress(
            f"fetch {source}: {preferred.name} has not caught up with this "
            f"cycle -- using {ladder[0].name}, which publishes before the "
            "mirrors")
    return promoted


def gfs_object_key(cycle: datetime, hour: int, source: str = "gfs") -> str:
    """The host-independent key of one ``pgrb2.0p25`` forecast hour.

    The same relative key on either endpoint: the operational server
    and the archive answer it with the same ``Content-Length`` (HEAD
    verified 2026-08-24 for both GFS and GDAS), which is what makes the
    endpoint ladder an append rather than a re-plan.
    """

    prefix = GFS_CONTAINER_PREFIX[source]
    return (f"{prefix}.{cycle:%Y%m%d}/{cycle:%H}/atmos/"
            f"{prefix}.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}")


def gfs_object_url(cycle: datetime, hour: int, source: str = "gfs",
                   transport: str = "s3") -> str:
    """One ``pgrb2.0p25`` forecast hour on one endpoint.

    Availability probes, the live index behind the record-count bar,
    and -- since the raw complex-packed north-to-south form earned its
    certification (see ``tests/fixtures/gfs-scan-order/README.md``) --
    the payload of ``--mode full-file`` itself.
    """

    endpoint = fetch_endpoints.endpoint_named(source, transport)
    return endpoint.url(gfs_object_key(cycle, hour, source))


def _hrrr_transport_base(transport: str) -> str:
    if transport == "s3":
        return HRRR_S3_BASE
    if transport == "nomads":
        return HRRR_NOMADS_BASE
    raise ValueError(
        f"unknown HRRR transport {transport!r}; expected 'nomads' or 's3'")


#: The HRRR file each rw_fetch product name stands for.
HRRR_PRODUCT_FILES = {"nat": "wrfnat", "prs": "wrfprs", "sfc": "wrfsfc"}


def hrrr_lead_products() -> tuple[str, ...]:
    """Every HRRR file one lead's fetch reads, so all its probe must ask.

    ``wrfnat`` (atmosphere) and ``wrfprs`` (soil), plus the file of each
    runtime surface field the hrrr adapter declares (VEGFRA is cut from
    ``wrfsfc``).  The breakage this prevents: NOAA posts a lead's
    ``wrfsfc`` up to ~10 min after its ``wrfnat``/``wrfprs`` (07Z f07 on
    2026-10-20: wrfprs 08:03:26, wrfsfc 08:13:00), so a probe of the pair
    alone called the lead posted and the VEGFRA transfer was refused with
    "the GRIB object is not present at this source", failing the run.
    """

    from woof.source_adapters import get_source_adapter

    extra = tuple(HRRR_PRODUCT_FILES[row[1]] for row in
                  get_source_adapter("hrrr").runtime_surface_fields)
    return tuple(dict.fromkeys(("wrfnat", "wrfprs", *extra)))


def hrrr_lead_urls(cycle: datetime, hour: int,
                   transport: str = "s3") -> tuple[str, ...]:
    """The URLs of :func:`hrrr_lead_products` for one lead on one host."""

    return tuple(hrrr_object_url(cycle, hour, product, transport=transport)
                 for product in hrrr_lead_products())


def hrrr_object_url(cycle: datetime, hour: int, product: str,
                    transport: str = "s3") -> str:
    if product not in HRRR_PRODUCT_FILES.values():
        raise ValueError(f"unknown HRRR product {product!r}")
    return (f"{_hrrr_transport_base(transport)}/hrrr.{cycle:%Y%m%d}/conus/"
            f"hrrr.t{cycle:%H}z.{product}f{hour:02d}.grib2")


def require_cycle_grid(source: str):
    """This source's declared initialization grid, or a refusal saying why.

    THE REFUSAL IS DERIVED.  ``--cycle latest`` used to be answered by a
    branch on three model names, so a reader asking for RAP or ICON-EU
    was told "latest is only meaningful for gfs/gdas/hrrr" and then told
    about ERA5's latency, which they had not asked about -- and ERA5
    itself, whose publication delay is a KNOWN NUMBER, was refused for
    having one.  A list of names cannot say anything true about a
    registry of thirty-two sources.

    What is said instead names the missing declaration, so the sentence
    stays true as the row grows and stops being said the moment it does.
    """

    from woof.source_cycles import cycle_grid_for

    grid = cycle_grid_for(source)
    if grid is not None:
        return grid
    raise ValueError(layered(
        f"--cycle latest cannot be resolved for {source!r}: nothing in "
        "this build declares when that source initializes.",
        "`latest` means the newest init a source can serve, which needs "
        "the UTC hours the producer runs on and how long after each one "
        "its bytes land.  A source with a fetch route declares both in "
        "the route table (`woof sources` lists what is registered); a "
        "source without one declares them on its registry row.  This "
        "source has neither, so name the cycle you want as "
        "YYYY-MM-DDTHH (UTC)."))


def cycle_is_probeable(source: str) -> bool:
    """Can this source's publication be settled by asking a server?

    Two shapes answer yes and one answers no, and the no is not a
    restriction: the CDS is a keyed JOB API, so there is no object to
    HEAD for ERA5 and no probe to run.  A source that answers no
    resolves ``--cycle latest`` from its declared publication delay
    instead, and the fetch's own completeness contract reports anything
    the delay was optimistic about.
    """

    if source in GFS_CONTAINER_SOURCES or source == "hrrr":
        return True
    try:
        fetch_routes.route_for(source)
    except (ValueError, KeyError):
        return False
    return True


def _route_probe_urls(source: str, cycle: datetime, last_hour: int,
                      transport: str | None, *, cadence: int | None = None,
                      start_hour: int = 0, member: str | None = None) -> tuple[str, ...]:
    """The selected member's final objects on ONE endpoint.

    The caller walks endpoints. Flattening all mirror URLs here would require
    every mirror to have caught up before treating the primary as published.
    """
    plan = fetch_routes.resolve_request(
        source, cycle=cycle, hours=last_hour - start_hour, host=transport,
        cadence=cadence, start_hour=start_hour, member=member)
    final = plan.leads[-1]
    urls = tuple(obj.url for obj in plan.objects if obj.lead == final or obj.lead is None)
    if not urls:
        raise ValueError(f"{source}: no publication objects declared for f{final:03d}")
    return urls


def cycle_probe_urls(source: str, cycle: datetime, last_hour: int,
                     transport: str | None = None, *,
                     cadence: int | None = None, start_hour: int = 0,
                     member: str | None = None) -> tuple[str, ...]:
    """The objects whose existence proves one cycle covers ``last_hour``.

    ``transport`` names the endpoint to ask.  It defaults to the head of
    the source's ladder for that cycle's age, which for a recent cycle
    is the operational server -- and that is the whole point of the
    default: the archive lags by minutes to hours, so probing it first
    resolves ``--cycle latest`` to a cycle that is already stale by the
    time the fetch starts.

    The legacy transports build their own URLs because they have no
    route row to derive from; everything else is derived from the route
    table.  A source that can be probed at all is
    :func:`cycle_is_probeable`, and asking one that cannot is refused by
    naming what its row lacks -- never by not being on a list.
    """

    if source in GFS_CONTAINER_SOURCES:
        if transport is None:
            transport = fetch_endpoints.serving_ladder(
                source, cycle=cycle)[0].name
        return (gfs_object_url(cycle, last_hour, source,
                               transport=transport),)
    if source == "hrrr":
        if transport is None:
            transport = fetch_endpoints.serving_ladder(
                source, cycle=cycle)[0].name
        return hrrr_lead_urls(cycle, last_hour, transport=transport)
    if cycle_is_probeable(source):
        return _route_probe_urls(source, cycle, last_hour, transport,
                                 cadence=cadence, start_hour=start_hour, member=member)
    raise ValueError(layered(
        f"{source!r} publishes no object a completeness probe can ask "
        "for, so this cycle's publication cannot be settled by probing.",
        "A probe needs a file server: a URL that answers HEAD once the "
        "bytes are there.  This source is acquired over a transport that "
        "has none -- a keyed job API, or a route this build does not "
        "carry -- so `--cycle latest` resolves from the publication "
        "delay its registry row declares and the fetch reports what it "
        "could not serve."))


def probe_cycle_window(source: str, cycle: datetime, leads, *,
                       now: datetime | None = None, probe=None,
                       transport: str | None = None, cadence: int | None = None,
                       member: str | None = None) -> dict:
    """Check the exact requested lead set through the declared URL owners.

    One endpoint must contain the complete requested set. The final lead is
    checked first, followed by every preceding required frame and invariant.
    This proves object availability only; preparation still verifies payload,
    source member, field inventory and donor identity before integration.

    ``probe`` answers True, False, or None for a host that could not be
    heard (:func:`_head_answer`).  ``available`` is False only when every
    rung that was asked answered and none holds the set; when a host was
    not heard it is None, as for a source with nothing to probe, because
    a timeout says nothing about whether the objects are there.
    """
    values = tuple(leads)
    if (not values or any(type(hour) is not int or hour < 0 for hour in values)
            or tuple(sorted(set(values))) != values):
        raise ValueError('The publication probe requires sorted unique nonnegative forecast leads')
    if not cycle_is_probeable(source):
        return dict(probeable=False, available=None, checks=[])
    probe = _head_answer if probe is None else probe
    checks = []
    unheard = False
    for endpoint in fetch_endpoints.serving_ladder(source, cycle=cycle, now=now, pinned=transport):
        seen = set()
        complete = True
        for lead in (values[-1], *values[:-1]):
            urls = cycle_probe_urls(source, cycle, lead, transport=endpoint.name,
                                    cadence=cadence, start_hour=values[0], member=member)
            for url in urls:
                if url in seen:
                    continue
                seen.add(url)
                found = probe(url)
                available = None if found is None else bool(found)
                checks.append(dict(endpoint=endpoint.name, lead=lead, url=url, available=available))
                if available is not True:
                    complete = False
                    unheard = unheard or available is None
                    break
            if not complete:
                break
        if complete and seen:
            return dict(probeable=True, available=True, endpoint=endpoint.name, checks=checks)
    return dict(probeable=True, available=None if unheard else False, checks=checks)


@dataclass(frozen=True)
class PublicationCheck:
    """What a named cycle's publication check found, from :func:`cycle_publication_check`.

    ``state`` is ``"published"`` (one rung holds every object),
    ``"not-published"`` (every rung asked answered, and none holds them
    all), ``"unchecked"`` (no rung was heard to hold them all and at
    least one host could not be heard, so the fetch goes ahead and each
    object is checked as it downloads) or ``"unprobeable"`` (no public
    object to ask).  ``why`` is the sentence for ``"not-published"``
    (the refusal) and ``"unchecked"`` (the hosts not heard), else None.
    """

    state: str
    why: str | None = None


def cycle_publication_check(source: str, cycle: datetime, last_hour: int, *,
                            now: datetime | None = None,
                            probe=_head_answer,
                            transport: str | None = None,
                            cadence: int | None = None, start_hour: int = 0,
                            member: str | None = None) -> PublicationCheck:
    """Whether a named cycle is published through ``last_hour``, as a :class:`PublicationCheck`.

    THE question a fetch asks before it moves a byte: does one endpoint
    already hold every object for the final requested lead?  The date
    guidance (:mod:`woof.source_availability`) asks it through this same
    function, so a start the guidance calls available is one the fetch
    accepts.  A source with no public object to probe is
    ``"unprobeable"``: the fetch cannot settle it either, and reports
    what it could not serve.

    ``probe`` answers True, False, or None when the host could not be
    heard; a probe that answers only True or False is read as it always
    was.  "Not published" is said only when every rung asked answered.
    A rung that was not heard might hold the cycle, so the check then
    says which host could not be reached, and the fetch goes ahead and
    checks each object as it downloads (GS-05: one connect timeout among
    about 170 HEADs refused a published start as "not published yet").
    """
    if not cycle_is_probeable(source):
        return PublicationCheck("unprobeable")  # A keyed job API has no public object to HEAD.
    options = dict(cadence=cadence, start_hour=start_hour, member=member)
    ladder = fetch_endpoints.serving_ladder(source, cycle=cycle, now=now, pinned=transport)
    unheard: list[str] = []
    for endpoint in ladder:
        urls = cycle_probe_urls(source, cycle, last_hour, transport=endpoint.name, **options)
        if not urls:
            continue
        answer, url = _rung_answer(urls, probe)
        if answer is True:
            return PublicationCheck("published")
        if answer is None:
            host = urlsplit(url).netloc or endpoint.name
            if host not in unheard:
                unheard.append(host)
    if transport is not None:
        # A pinned host past its declared retention does not keep the
        # cycle, whether or not it answered this time.
        retention = _pinned_retention_refusal(source, cycle, last_hour,
                                              transport, now=now)
        if retention is not None:
            return PublicationCheck("not-published", retention)
    selection = f" member {member}" if member is not None else ""
    named = f"{source.upper()}{selection} cycle {cycle:%Y-%m-%dT%H}Z"
    if unheard:
        return PublicationCheck(
            "unchecked",
            f"could not reach {' or '.join(unheard)} to check whether {named} "
            f"is published through f{last_hour:03d}; the fetch goes ahead "
            "and checks each file as it downloads")
    newest = None
    try:
        newest = resolve_latest_cycle(source, last_hour, now=now, probe=probe,
                                       transport=transport, **options)
        remedy = (f"the newest complete {source.upper()} cycle covering "
                  f"f{last_hour:03d} is {newest:%Y-%m-%dT%H}Z -- pass that, "
                  "or --cycle latest to resolve it automatically")
    except (RuntimeError, ValueError) as error:
        remedy = f"and no complete cycle could be resolved either ({error})"
    gone = _passed_cycle_refusal(source, cycle, last_hour, ladder, named=named,
                                 newest=newest, now=now, pinned=transport)
    if gone is not None:
        return PublicationCheck("not-published", f"{gone}; {remedy}")
    return PublicationCheck(
        "not-published", f"{named} is not published through f{last_hour:03d} yet; {remedy}")


def _passed_cycle_refusal(source: str, cycle: datetime, last_hour: int,
                          asked, *, named: str, newest: datetime | None,
                          now: datetime | None,
                          pinned: str | None) -> str | None:
    """Why a cycle the server does not hold will not appear by waiting, or None while it still may.

    Two facts settle it, both read from the source's rows and neither
    from its name.  The declared retention of every host the source
    publishes on: past all of them, and with no archive behind them, the
    cycle has aged off.  And the newest complete cycle: a cycle older
    than it is not still publishing.  Without this an ICON cycle a day
    and a half old, long gone from DWD's rolling door, was refused as
    "not published through f003 yet" and the user was left to wait for
    a cycle that would never appear.
    """

    rungs = fetch_endpoints.ladder(source)
    age = fetch_endpoints.cycle_age_hours(cycle, now)
    if rungs and not any(rung.covers(age) for rung in rungs):
        hosts = list(dict.fromkeys(rung.host for rung in rungs))
        kept = max(float(rung.retention_hours) for rung in rungs)
        one = len(hosts) == 1
        return (f"{named} is no longer on the server: {' and '.join(hosts)} "
                f"{'keeps' if one else 'keep'} only about the newest {kept:g} h "
                f"of {source.upper()} cycles and this one is {age:.0f} h old, "
                f"with no archive behind {'it' if one else 'them'}, so it will "
                "not appear by waiting")
    if newest is None or not cycle < newest:
        return None
    hosts = list(dict.fromkeys(rung.host for rung in asked)) or [source.upper()]
    said = (f"{named} is not on {' or '.join(hosts)} through f{last_hour:03d}, "
            "and waiting will not bring it: a newer cycle is already complete")
    keepers = [rung.name for rung in rungs
               if pinned is not None and rung.name != pinned and rung.covers(age)]
    if keepers:
        said += (f"; {' and '.join(keepers)} also "
                 f"{'keeps' if len(keepers) == 1 else 'keep'} {source.upper()} "
                 f"cycles this old -- pass --transport {keepers[0]}, or leave "
                 "--transport off and the fetch asks every host")
    return said


def cycle_publication_refusal(source: str, cycle: datetime, last_hour: int, *,
                              now: datetime | None = None,
                              probe=_head_answer,
                              transport: str | None = None,
                              cadence: int | None = None, start_hour: int = 0,
                              member: str | None = None) -> str | None:
    """Why a named cycle cannot be fetched yet, or None when it can.

    :func:`cycle_publication_check`'s refusal: its sentence when the
    cycle is not published, None otherwise.  A host that could not be
    heard is not a refusal: the fetch goes ahead and checks each object
    as it downloads.
    """
    check = cycle_publication_check(
        source, cycle, last_hour, now=now, probe=probe, transport=transport,
        cadence=cadence, start_hour=start_hour, member=member)
    return check.why if check.state == "not-published" else None


def _pinned_retention_refusal(source: str, cycle: datetime, last_hour: int,
                              pinned: str, *,
                              now: datetime | None = None) -> str | None:
    """Why a pinned host that has let this cycle go cannot serve it.

    None while the host's declared retention still covers the cycle's
    age.  Past it, the host did not answer because it no longer keeps
    the cycle, not because the cycle is still publishing, so the refusal
    says so and names the hosts this source publishes on that still keep
    a cycle that old.  Without it an old cycle pinned to the operational
    server was told it was "not published yet" and pointed at a cycle
    from today.
    """

    rungs = fetch_endpoints.ladder(source)
    host = next((rung for rung in rungs if rung.name == pinned), None)
    if host is None:
        return None
    age = fetch_endpoints.cycle_age_hours(cycle, now)
    if host.covers(age):
        return None
    keepers = [rung.name for rung in rungs
               if rung.name != pinned and rung.covers(age)]
    refusal = (f"--transport {pinned}: {pinned} keeps only about the "
               f"newest {host.retention_hours:g} h of {source.upper()} "
               f"cycles, and cycle {cycle:%Y-%m-%dT%H}Z is {age:.0f} h "
               f"old, so it no longer serves f{last_hour:03d}")
    if keepers:
        return (f"{refusal}; {' and '.join(keepers)} still "
                f"{'keeps' if len(keepers) == 1 else 'keep'} it.  Pass "
                f"--transport {keepers[0]}, or leave --transport off and "
                "the fetch asks the host that has it")
    return (f"{refusal}, and no other host {source.upper()} publishes on "
            "keeps a cycle that old.  Leave --transport off and the fetch "
            "asks every host")


def require_published_cycle(source: str, cycle: datetime, last_hour: int, *,
                            now: datetime | None = None,
                            probe=_head_answer, progress=print,
                            transport: str | None = None,
                            cadence: int | None = None, start_hour: int = 0,
                            member: str | None = None) -> None:
    """Check a named cycle's requested member/end before moving payload bytes.

    One complete endpoint is sufficient. An explicitly pinned endpoint is the
    only one asked; a control member or a different mirror cannot authorize
    downloading the selected member from a still-incomplete pinned endpoint.
    The question itself is :func:`cycle_publication_check`: a cycle not
    published is refused with its sentence, and a host that could not be
    heard is said on ``progress`` before the fetch goes ahead.
    """
    check = cycle_publication_check(
        source, cycle, last_hour, now=now, probe=probe, transport=transport,
        cadence=cadence, start_hour=start_hour, member=member)
    if check.state == "not-published":
        raise RuntimeError(check.why)
    if check.state == "unchecked":
        progress(f"fetch {source}: {check.why}")


def analysis_window_reference(source: str, grid, last_hour: int,
                             now: datetime) -> datetime:
    """The instant one window's worth of analyses is resolved at.

    An analysis source publishes no forecast leads, so an ``last_hour``
    hour window is that many hours of successive ANALYSES: the newest
    start it can have is the newest published analysis minus the window,
    and a later start asks for valid times the provider has not published
    yet.  A forecast source covers its window with leads from one cycle,
    so its reference is the caller's own instant, unchanged.

    Two doors resolve the same latest and must agree about one request:
    :func:`resolve_latest_cycle`, which serves ``--cycle latest`` and
    every planner that calls it, and :mod:`woof.source_availability`,
    whose calendar publishes ``latest_candidate`` and resolves the Latest
    button.  Each used to subtract the window itself, so on that path it
    came off twice: an era5 240-hour window selected a cycle ten days
    before the same document's own ``latest_candidate``, silently.

    The subtraction happens AFTER ``newest()``, because a rolling
    publication delay and a closed archive's ``record_end`` both bound
    the LAST analysis requested and only ``newest()`` applies both;
    subtracting from ``now`` alone gets a closed archive wrong.  The
    delay is added back so the answer is a reference INSTANT rather than
    a cycle: ``grid.newest()`` of it is the newest usable start, and
    ``grid.candidates()`` of it walks the same search window back from
    there, which is what a probing route needs.
    """

    if _source_reaches_forecast_leads(source):
        return now
    start = grid.newest(now) - timedelta(hours=last_hour)
    return start + timedelta(hours=grid.delay(start))


def provider_cycle_grid(source: str, provider: str | None, *,
                        now: datetime | None = None,
                        published=None) -> tuple[object, str]:
    """``(grid, basis)``: the cycle grid ``latest`` resolves on for one provider.

    A source whose registry row declares archive windows per transport
    (ERA5: the keyed CDS and Google's keyless ARCO copy) can be served
    later by one of them than by another, and ``latest`` must resolve
    against the one the fetch will ask.  The ARCO copy trails the CDS by
    a day or two, so resolving an ARCO request on the CDS delay asked
    for hours the store does not hold and the fetch refused in 4 s
    ("outside the coverage this store declares").

    The window's own row decides, in this order: the provider's own
    bounds document when it publishes one and it can be read (its last
    published hour IS the answer, as the Rust reader checks the same
    attribute), then the window's declared ``publication_lag_hours``,
    then the source's cycle grid unchanged.  ``published`` reads a
    bounds document (:func:`woof.source_availability.published_stop`
    by default).
    """

    from dataclasses import replace

    grid = require_cycle_grid(source)
    if provider is None:
        return grid, "the declared publication delay"
    try:
        adapter = source_adapters.get_source_adapter(source)
    except ValueError:
        return grid, "the declared publication delay"
    window = next((row for row in adapter.archive_windows
                   if row.transport == provider), None)
    if window is None:
        return grid, "the declared publication delay"
    if window.bounds_url and window.bounds_stop_keys:
        if published is None:
            from woof.source_availability import published_stop as published
        stop = published(window)
        if stop is not None:
            return (replace(grid, delay_hours=0.0, delays=(), record_end=stop),
                    f"the {provider} store's own declared end, {stop:%Y-%m-%dT%H}Z")
    lag = getattr(window, "publication_lag_hours", None)
    if lag is not None:
        return (replace(grid, delay_hours=float(lag), delays=()),
                f"the {provider} provider's declared publication delay of "
                f"about {float(lag):g} h")
    return grid, "the declared publication delay"


def resolve_latest_cycle(source: str, last_hour: int, *,
                         now: datetime | None = None,
                         probe=_head_ok, transport: str | None = None,
                         cadence: int | None = None, start_hour: int = 0,
                         member: str | None = None,
                         provider: str | None = None,
                         as_posted: bool | None = None) -> datetime:
    """Newest cycle whose final requested objects are actually published.

    With ``as_posted`` (the default of every front door, DESIGN A136
    2.2) ``latest`` is instead the newest cycle whose START needs are
    posted -- the analysis, the first boundary lead and whatever the
    route fetches with them -- and whose ladder covers the window
    (:func:`woof.source_readiness.resolve_startable_cycle`): the run
    starts on those and fetches the rest as they post.  A source whose
    cycle cannot be waited on (a keyed job, an archive) resolves as
    before.

    Under the ordinary whole-cycle rule, a cycle qualifies only when
    every probed object for forecast hour
    ``last_hour`` is already published, so a partially uploaded cycle
    never wins and the fetched window is complete by construction.  For
    HRRR that means BOTH the final ``wrfnat`` (atmosphere) and the final
    ``wrfprs`` (soil-record source) objects: fetching needs both per
    hour, and during a live publication ``wrfnat`` can appear before its
    ``wrfprs`` sibling, which must not make the cycle win.  For a table
    route that declares a donor (the hybrid AI routes and their
    same-cycle GDAS analysis) the donor's final lead has to be published
    too, because the fetch downloads both and refuses without it.

    Without the hosted AWS policy, endpoints are asked in ladder order,
    and the operational server
    heads it.  That IS the answer to "latest": the archive lags the
    operational server by minutes to hours, so resolving against the
    archive returned an older cycle than the one already published --
    a run initialized an hour behind the best available state, with
    nothing in the receipt to say why.  The archive is still asked when
    the operational server yields no complete cycle at all.
    """

    if native_cf_fetch_contract(source) is not None:
        from woof.cf_archive_fetch import latest_cycle
        if start_hour or member is not None or provider is not None or transport is not None:
            raise ValueError("Annual CF analyses have no forecast lead, member or alternate provider axis")
        return latest_cycle(fetch_routes.canonical_source(source),last_hour,cadence)
    transport = fetch_endpoints.policy_transport(source, transport)
    if as_posted is None:
        as_posted = fetch_endpoints.aws_fetch_policy()
    if isinstance(last_hour, bool) or not isinstance(last_hour, int) or last_hour < 0:
        raise ValueError("the final requested hour must be a nonnegative integer")
    if (isinstance(start_hour, bool) or not isinstance(start_hour, int)
            or not 0 <= start_hour <= last_hour):
        raise ValueError("the start hour must be an integer between zero and the final hour")
    if as_posted and provider is None and _startable_rule_applies(source):
        from woof.source_readiness import resolve_startable_cycle

        return resolve_startable_cycle(
            source, start_hour=start_hour, hours=last_hour - start_hour,
            cadence=cadence, member=member, transport=transport, now=now,
            probe=probe)
    route = None
    if source in fetch_routes.route_ids():
        route = fetch_routes.route_for(source)
        fetch_routes.resolve_member(route, member)
        if transport is not None:
            route.host(transport)
        if cadence is None:
            # ONE spacing for the whole walk: the finest any cycle hour
            # publishes over this window, the rule `woof domain` writes
            # into [fetch].  Chosen per candidate instead, ICON-EU's short
            # 03/09/15/21Z runs (hourly to f030, then 6-hourly) won a 36 h
            # window at 6 h spacing over the hourly main cycle, and a
            # caller that then fetched at its own hourly cadence (a saved
            # setup's start) was refused the cycle `latest` had handed it.
            cadence = fetch_routes.window_cadence(
                route, start_hour, last_hour - start_hour)
    if now is None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
    elif now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)
    # ``provider`` is a transport the source's archive windows name (the
    # keyless ARCO copy of ERA5): it can trail the source's usual delay.
    grid, _basis = provider_cycle_grid(source, provider, now=now)
    # ONE owner of the analysis back-off, for every door that resolves a
    # latest cycle.  When the calendar applied it too, a 240-hour era5
    # window resolved ten days before the same document's own
    # latest_candidate, with nothing said about it.
    now = analysis_window_reference(source, grid, last_hour, now)
    if (not cycle_is_probeable(source)
            and not _source_reaches_forecast_leads(source)):
        return grid.newest(now)
    # A cycle that does not reach the end of the window is not a
    # candidate at all -- it is a cycle that cannot serve the request.
    # The rule is the row's or the route's; nothing here knows which
    # producer runs a short off-synoptic cycle.
    # The walk starts at the newest cycle whose declared delay for the
    # window's LAST lead has passed: a cycle posts its leads over hours,
    # and the probe below asks about the last one.
    candidates = tuple(
        cycle for cycle in grid.candidates(now, last_hour)
        if grid.horizon(cycle) is None or last_hour <= grid.horizon(cycle))
    if not candidates:
        raise RuntimeError(layered(
            f"no {source} cycle in the last {grid.search_hours} h "
            f"forecasts as far as f{last_hour:03d}.",
            f"The declared horizons are {list(grid.horizons)} "
            f"(cycle hours, through-hour), and this window needs "
            f"f{last_hour:03d}.  Shorten --hours, or name a cycle whose "
            "own ladder reaches it."))
    if route is not None:
        admitted, errors = [], []
        for cycle in candidates:
            try:
                fetch_routes.resolve_leads(route, cycle, last_hour - start_hour,
                                           cadence=cadence, start_hour=start_hour)
                admitted.append(cycle)
            except ValueError as error:
                errors.append(error)
        if not admitted:
            raise errors[-1]
        candidates = tuple(admitted)
    if not cycle_is_probeable(source):
        # No file server to ask, so the declared publication delay IS
        # the answer.  Reported as resolved rather than refused: the
        # newest published analysis of a reanalysis is a well-defined
        # time, and the fetch's own completeness contract is what
        # reports a delay that turned out optimistic.
        return candidates[0]
    ladder = fetch_endpoints.serving_ladder(
        source, cycle=candidates[0], now=now, pinned=transport)
    donors = route.donors if route is not None else ()
    donors_published: dict[datetime, bool] = {}

    def request_complete(cycle: datetime) -> bool:
        # A declared donor is part of the same request: the hybrid AI
        # routes take their land surface from the same-cycle GDAS
        # analysis, which publishes later than the AI atmosphere.  Asked
        # only about the primary, latest picked a cycle whose donor was
        # not out yet and the fetch then refused it, although the cycle
        # before had everything.  Asked once per cycle, whichever of the
        # primary's endpoints held it.
        if cycle not in donors_published:
            donors_published[cycle] = all(
                _donor_published(donor.source, cycle, max(donor.leads),
                                 now=now, probe=probe)
                for donor in donors)
        return donors_published[cycle]

    for endpoint in ladder:
        for cycle in candidates:
            urls = cycle_probe_urls(source, cycle, last_hour,
                                    transport=endpoint.name, cadence=cadence,
                                    start_hour=start_hour, member=member)
            if objects_published(urls, probe) and request_complete(cycle):
                return cycle
    tried = " or ".join(endpoint.name for endpoint in ladder)
    needs = "".join(
        f" (with its same-cycle {donor.source.upper()} "
        f"f{max(donor.leads):03d})" for donor in donors)
    raise RuntimeError(
        f"no complete {source.upper()} cycle covering f{last_hour:03d}"
        f"{needs} was found on {tried} within the last "
        f"{grid.search_hours} h; pass an explicit --cycle")


def resolve_cycle_for_valid_start(source: str, valid_start: datetime, hours: int,
                                  cadence: int, *, now: datetime | None = None,
                                  probe=_head_ok, transport: str | None = None,
                                  as_posted: bool = True) -> tuple[datetime, int]:
    """Newest startable cycle preserving a fixed valid start and duration."""

    from woof import source_readiness

    if type(hours) is not int or hours < 0 or type(cadence) is not int or cadence < 1:
        raise ValueError("the window needs nonnegative integer hours and a positive integer cadence")
    if valid_start.tzinfo is not None:
        valid_start = valid_start.astimezone(timezone.utc).replace(tzinfo=None)
    if valid_start.minute or valid_start.second or valid_start.microsecond:
        raise ValueError("the valid start must be on an exact UTC hour")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)
    grid, _basis = provider_cycle_grid(source, None, now=now)
    if not cycle_is_probeable(source):
        raise ValueError(f"{source} publishes no objects to verify for a fixed valid start")
    transport = fetch_endpoints.policy_transport(source, transport)
    cycle = grid.snap(valid_start)
    newest = cycle
    while newest - cycle <= timedelta(hours=grid.search_hours):
        start = int((valid_start - cycle).total_seconds() // 3600)
        last = start + hours
        horizon = grid.horizon(cycle)
        if cycle <= now and (horizon is None or last <= horizon):
            try:
                window = source_readiness.resolve_window(
                    source, cycle, hours, cadence=cadence, start_hour=start,
                    transport=transport, now=now)
            except ValueError:
                cycle = grid.snap(cycle - timedelta(hours=1))
                continue
            needs = source_readiness.start_needs(window)
            if not as_posted:
                expected, late = source_readiness._due(source, cycle, last, None, None)
                needs += (source_readiness.Need("whole_cycle", source, last, expected, late),)
            if all(source_readiness.need_answer(window, need, probe=probe, now=now)["answer"]
                   == source_readiness.POSTED for need in needs):
                return cycle, start
        cycle = grid.snap(cycle - timedelta(hours=1))
    raise RuntimeError(
        f"no {'startable' if as_posted else 'complete'} {source} cycle covers "
        f"{valid_start:%Y-%m-%dT%H}Z for {hours} h within {grid.search_hours} h")


def _startable_rule_applies(source: str) -> bool:
    """Whether ``latest`` under ``as_posted`` asks start needs for ``source``.

    Only a source whose cycle is waited on (its posting row's shape) and
    that a probe can ask; an analysis source's window is analyses, not
    leads of one cycle.
    """

    from woof.source_posting import posting

    try:
        return (cycle_is_probeable(source)
                and _source_reaches_forecast_leads(source)
                and posting(source).waited)
    except ValueError:
        return False


#: ``[fetch] as_posted`` when nothing says otherwise (DESIGN A136 3.2):
#: fetch, prepare and run as the leads post.
DEFAULT_AS_POSTED = True


def requested_as_posted(args) -> bool:
    """Whether this fetch runs as posted: ``--as-posted``/``--wait-for``
    (True), ``--whole-cycle`` (False), or the default."""

    value = getattr(args, "as_posted", None)
    return DEFAULT_AS_POSTED if value is None else bool(value)


def _posting_flag_refusal(args, as_posted: bool) -> None:
    for flag, key in (("--late-after-minutes", "late_after_minutes"),
                      ("--wait-timeout-minutes", "wait_timeout_minutes")):
        value = getattr(args, key, None)
        if value is None:
            continue
        if not as_posted:
            # Breakage it prevents: a budget typed beside --whole-cycle
            # would be accepted and then read by nothing, since a whole
            # cycle is asked once and never waited on lead by lead.
            raise ValueError(
                f"{flag} is the wait budget of an as-posted fetch, and "
                "--whole-cycle waits for nothing; drop one of the two")
        if (isinstance(value, bool) or not math.isfinite(float(value))
                or float(value) <= 0):
            # A zero or negative budget fails every lead a second late,
            # and NaN or infinity is a wait that never ends.
            raise ValueError(f"{flag} must be positive and finite")
    if getattr(args, "no_probe", False) and not getattr(args, "readiness", False):
        # Breakage it prevents: a probe switch on a fetch would be
        # accepted and ignored, while the fetch asked hosts anyway.
        raise ValueError("--no-probe belongs to --readiness")


def _posting_loop(source: str, cycle: datetime, args, *, leads,
                  plan=None, transport: str | None = None):
    """The as-posted loop for this window, or None where there is nothing to wait on.

    A source whose posting row is not waited on (a keyed job, an
    archive, private bytes), or that has no object to ask, moves as a
    one-shot fetch does.  A cycle aged off every host is refused here:
    it can never start, so it is not waited for.
    """

    from woof import fetch_as_posted
    from woof import source_readiness as readiness
    from woof.source_posting import posting

    if not cycle_is_probeable(source) or not posting(source).waited:
        return None
    window = readiness.Window(
        source=source, cycle=cycle, leads=tuple(leads),
        cadence=getattr(args, "cadence", None),
        member=(plan.member if plan is not None
                else getattr(args, "member", None)),
        transport=transport, plan=plan, hours=int(args.hours))
    return fetch_as_posted.PostingLoop(
        window, args.out,
        late_after_minutes=getattr(args, "late_after_minutes", None),
        wait_timeout_minutes=getattr(args, "wait_timeout_minutes", None))


def readiness_for_fetch(args, *, no_probe: bool | None = None,
                        nest_start_leads=None) -> tuple[dict, int]:
    """``gpuwm.readiness.v1`` and its exit code for one parsed fetch request.

    The one reader of a request for ``woof fetch --readiness``, ``woof
    go --readiness`` and ``woof run-plan --readiness``, so the three
    doors answer one window the same way.
    """

    from woof import source_readiness as readiness

    source = args.source
    if args.cycle is None or args.hours is None:
        raise ValueError("--readiness needs --cycle and --hours: it answers "
                         "for one window")
    provider = getattr(args, "era5_provider", None) if source == "era5" else None
    transport = (args.transport if source in fetch_routes.route_ids()
                 else pinned_host(getattr(args, "transport", None)))
    return readiness.readiness(
        source, args.cycle, args.hours, cadence=args.cadence,
        start_hour=args.forecast_start_hour,
        member=getattr(args, "member", None), transport=transport,
        provider=provider, as_posted=requested_as_posted(args),
        late_after_minutes=getattr(args, "late_after_minutes", None),
        nest_start_leads=nest_start_leads,
        no_probe=bool(getattr(args, "no_probe", False)
                      if no_probe is None else no_probe))


def _readiness_main(args, source: str) -> int:
    """``woof fetch --readiness``: print ``gpuwm.readiness.v1``, run nothing."""

    from woof import source_readiness as readiness

    document, code = readiness_for_fetch(args)
    readiness.print_document(document)
    print(f"fetch {source}: readiness {document['state']}"
          + (f" ({document['refusal']})" if document.get("refusal") else "")
          + (f"; expected ready at {document['expected_ready_at']}"
             if document.get("expected_ready_at") else ""),
          file=sys.stderr)
    return code


def _donor_published(source: str, cycle: datetime, last_hour: int, *,
                     now: datetime, probe) -> bool:
    """Whether one endpoint of a donor's own ladder holds its final lead.

    The primary's rule, asked of the donor's source: one rung holding
    every object of the final lead, and a host that could not be heard
    does not count as holding it.  A donor with no public object to ask
    cannot be settled here, and the fetch's own pre-transfer check is
    what reports it.
    """

    if not cycle_is_probeable(source):
        return True
    for endpoint in fetch_endpoints.serving_ladder(source, cycle=cycle,
                                                   now=now):
        urls = cycle_probe_urls(source, cycle, last_hour,
                                transport=endpoint.name)
        if urls and objects_published(urls, probe):
            return True
    return False


class _HrrrPrefixNotPosted(RuntimeError):
    """Every endpoint said an object of a native prefix is missing; its posting loop waits."""


def resolve_hrrr_transport(cycle: datetime, requested: str, *,
                           last_hour: int, now: datetime | None = None,
                           probe=None, progress=print, required_leads=None) -> str:
    """Pick the concrete HRRR transport for one fetch invocation.

    Both hosts serve byte-identical HRRR files and ``.idx`` indexes, so
    the choice is never about the data.  It is about two things the
    hosts do NOT share, and both are declared in the packaged endpoint
    ladder (``legacy_ladders.hrrr``): the operational server publishes
    each forecast hour before the cloud mirrors do and keeps only about
    :data:`HRRR_NOMADS_RETENTION_HOURS`; the S3 archive lags and keeps
    everything.

    ``auto`` asks the THROUGHPUT rung first and takes it when it
    already serves the requested window: the operational server's whole
    advantage is having the cycle first, and once the archive has the
    same object that advantage is spent.  What is left is throughput,
    and the archive wins it -- measured on one box, one cycle, the same
    four objects through the same backbone: 348/209/418/255 s from the
    operational server against 69/34/45/44 s from S3, and measured
    again at peak hours as ~3 MB/s per file against the archive serving
    the same 3.4 GB in roughly a sixth of the wall clock.

    When the archive has NOT caught up -- publication lag, which is the
    one thing the operational server exists for -- the operational
    server is probed and taken, and it says so.  A cycle past its
    retention window skips the doomed probe entirely.

    Without ``required_leads`` this keeps the historical final-hour
    selection. Native as-posted transfers supply their complete prefix:
    every wrfnat, wrfprs, runtime surface file and index must be available
    on one endpoint,
    because a later mirrored lead can arrive before an earlier soil pair.
    An endpoint that said one of them is missing (404 or 410) is never
    taken.  When none holds the whole prefix but one could not be heard
    about an object, that endpoint is taken and its transfer's own index,
    inventory and digest checks decide (GS-05): a probe with no answer
    is not a missing object.

    One decision per invocation, so a fetch never silently mixes hosts;
    the manifest records every file's actual URL and transport either
    way.  An explicit ``nomads`` that the operational server cannot
    serve refuses with the retention story rather than failing file by
    file mid-download.
    """

    requested = fetch_endpoints.policy_transport("hrrr", requested)
    if required_leads is not None:
        if requested not in HRRR_TRANSPORTS:
            raise ValueError(f"unknown HRRR transport {requested!r}")
        leads = tuple(required_leads)
        if not leads or leads[-1] != last_hour:
            raise ValueError("required HRRR leads must end at last_hour")
        selected_probe = _head_answer if probe is None else probe

        def with_index(url):
            found = selected_probe(url)
            if found is not True:
                return False if found is False else None
            found = selected_probe(url + ".idx")
            return found if found is True or found is False else None

        ladder = fetch_endpoints.serving_ladder(
            "hrrr", cycle=cycle, now=now,
            pinned=None if requested == "auto" else requested)
        asked = set()
        unheard = []
        for endpoint in (*fetch_endpoints.transfer_probes(ladder), *ladder):
            if endpoint.name in asked:
                continue
            asked.add(endpoint.name)
            verdict = probe_cycle_window(
                "hrrr", cycle, leads, now=now, probe=with_index,
                transport=endpoint.name)
            if verdict["available"] is True:
                progress(f"fetch hrrr: using {endpoint.name} for the complete "
                         f"verified prefix f{leads[0]:03d}..f{leads[-1]:03d}")
                return endpoint.name
            if verdict["available"] is None:
                unheard.append(endpoint.name)
        if unheard:
            progress(f"fetch hrrr: {unheard[0]} could not be heard about every "
                     f"object of f{leads[0]:03d}..f{leads[-1]:03d} and said "
                     "none is missing; its transfer's own checks decide")
            return unheard[0]
        raise _HrrrPrefixNotPosted(
            f"every HRRR endpoint said a {', '.join(hrrr_lead_products())} or index of "
            f"f{leads[0]:03d}..f{leads[-1]:03d} is missing; a final lead "
            "alone cannot authorize downloading a missing intermediate object")
    if requested == "s3":
        return "s3"
    if requested not in ("auto", "nomads"):
        raise ValueError(
            f"unknown HRRR transport {requested!r}; expected one of "
            f"{HRRR_TRANSPORTS}")
    if probe is None:
        probe = _head_ok
    if now is None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
    age_hours = fetch_endpoints.cycle_age_hours(cycle, now)
    if requested == "nomads":
        urls = hrrr_lead_urls(cycle, last_hour, transport="nomads")
        if all(probe(url) for url in urls):
            return "nomads"
        detail = (
            f" -- the cycle is {age_hours:.0f} h old and NOMADS keeps only "
            f"about the newest {HRRR_NOMADS_RETENTION_HOURS} h"
            if age_hours > HRRR_NOMADS_RETENTION_HOURS else
            " (still publishing, or the window's final hour is not up yet;"
            " an as-posted fetch, the default, downloads hours as they"
            " appear)")
        raise ValueError(
            f"--transport nomads: NOMADS is not serving cycle "
            f"{cycle:%Y-%m-%dT%H}Z through f{last_hour:02d}{detail}; use "
            "--transport s3 (the full archive) or auto")
    ladder = fetch_endpoints.serving_ladder("hrrr", cycle=cycle, now=now)

    def window_urls(name: str) -> tuple[str, ...]:
        return hrrr_lead_urls(cycle, last_hour, transport=name)

    # The throughput rung first: an hour the archive already mirrors has
    # nothing left to gain from the slower host.  Every final-hour
    # object must answer, because a fetch needs them all.
    for endpoint in fetch_endpoints.transfer_probes(ladder):
        if all(probe(url) for url in window_urls(endpoint.name)):
            progress(
                f"fetch hrrr: mirrored: taking the archive for throughput "
                f"-- {endpoint.name} already serves cycle "
                f"{cycle:%Y-%m-%dT%H}Z through f{last_hour:02d}")
            return endpoint.name

    for position, endpoint in enumerate(ladder):
        if position == len(ladder) - 1:
            # The last rung is the fallback; probing it would only
            # duplicate the refusal the transfer itself would give --
            # and as the throughput rung it has already been asked
            # above.
            break
        if all(probe(url) for url in window_urls(endpoint.name)):
            # EARNED, not assumed: this prints only after the archive
            # was asked and did not have the window, which is exactly
            # the publication lag the sentence claims.
            progress(
                f"fetch hrrr: using {endpoint.name} -- {endpoint.why}")
            return endpoint.name
        progress(
            f"fetch hrrr: {endpoint.name} does not serve cycle "
            f"{cycle:%Y-%m-%dT%H}Z through f{last_hour:02d} yet -- "
            f"asking {ladder[position + 1].name}")
    last = ladder[-1]
    if len(ladder) == 1 and age_hours > HRRR_NOMADS_RETENTION_HOURS:
        progress(
            f"fetch hrrr: cycle {cycle:%Y-%m-%dT%H}Z is {age_hours:.0f} h "
            f"old, beyond the ~{HRRR_NOMADS_RETENTION_HOURS} h NOMADS "
            "retention -- using the AWS S3 archive")
    return last.name


def _posted_hrrr_transport(loop, cycle, leads):
    """Choose one host for the prefix, waiting within the existing posting budget."""

    while True:
        try:
            return resolve_hrrr_transport(
                cycle, "auto", last_hour=leads[-1], required_leads=leads)
        except _HrrrPrefixNotPosted as error:
            # Every host said an object of the prefix is missing, though
            # individual leads may be present on different hosts. Never
            # fall back to a host known to lack a prefix object; re-ask
            # within this lead's existing deadline instead.
            loop.wait_again((leads[-1],), error)
            loop(leads[-1])


def _hrrr_pacing_advice(*, pinned: str | None, mode: str, cached: bool,
                        say=print):
    """``advise(host)``: what a resolved NOMADS whole-file transfer costs, said once.

    Said BEFORE the first byte moves from the operational server, and
    only when the host was RESOLVED rather than named: an operator who
    typed ``--transport nomads`` made a decision, and a decision does not
    get advice.  The archive was already asked first and did not hold the
    window, so this is not a nudge towards ``--transport s3``, which
    would only 404; it is the cost of the freshness that was the only
    thing on offer.  Measured on one box, one cycle, the same four
    objects through the same backbone: 348/209/418/255 s from the
    operational server against 69/34/45/44 s from S3.

    A whole-cycle fetch decides its host once, before the transfer; an
    as-posted fetch decides it for each verified prefix, so ``advise``
    is called with each prefix's host and speaks before the first one
    the operational server serves, and never again in the same fetch.
    """

    said = []

    def advise(host: str | None) -> None:
        if (said or pinned is not None or cached
                or mode not in ("full-file", "auto") or host != "nomads"):
            return
        said.append(host)
        say("fetch hrrr: the operational server paces whole-file "
            "transfers -- expect several times the wall clock of "
            "the S3 archive.  It is serving "
            "this fetch because it is the only host that has "
            "this window yet; once the archive catches up, a "
            "re-run takes it from there without being asked.")

    return advise


# ---------------------------------------------------------------------------
# Shared transport helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Only fresh Rust publications enter this memo. Existing/resumed files still
# undergo independent identity checks. Mutation, append and rename change keys.
_STREAM_FACTS = OrderedDict()
_STREAM_FACTS_LOCK = threading.Lock()


def _stream_entry(identity):
    # Stat clocks can have coarser resolution on overlay filesystems. A
    # same-size write must invalidate the stream digest even inside one tick.
    with _STREAM_FACTS_LOCK:
        entry = _STREAM_FACTS.get(identity)
        if entry is not None and entry['_change_watch'].unchanged():
            return entry
        if entry is not None:
            _STREAM_FACTS.pop(identity)['_change_watch'].close()
    return None


def _publication(identity: tuple) -> dict:
    """The fields of a :func:`_stat_identity` a Rust receipt's ``publication`` names."""
    return dict(zip(("dev", "ino", "bytes", "mtime_ns", "ctime_ns"), identity[1:]))


#: Output holds open in this process; see :func:`_fetch_out`.
_FETCH_OUT_HOLDS = 0


@contextlib.contextmanager
def _fetch_out(out, **options):
    """The ``fetch-out`` lock; the last one to end closes held payload files.

    A Windows mutation notice keeps its payload open, and an open file
    refuses a rename of any folder above it and a MoveFileEx replacement
    of itself.  The fetch that published a payload owns its folder until
    it lets go.  When no fetch in this process holds an output any more,
    the fresh facts that keep a file open are dropped, and later checks
    read the disk as a resumed process does.  An inotify watch holds no
    file and stays.
    """
    global _FETCH_OUT_HOLDS
    # A second writer of this folder waits as long as the holder's bytes
    # keep arriving and refuses only a holder that stalls for the budget
    # (fetch_guard.DirectoryProgress names the breakage).
    options.setdefault("holder_progress", fetch_guard.DirectoryProgress(out))
    with fetch_guard.hold("fetch-out", out, **options):
        with _STREAM_FACTS_LOCK:
            _FETCH_OUT_HOLDS += 1
        try:
            yield
        finally:
            with _STREAM_FACTS_LOCK:
                _FETCH_OUT_HOLDS -= 1
                if not _FETCH_OUT_HOLDS:
                    for identity in [key for key, value in _STREAM_FACTS.items()
                                     if value['_change_watch'].holds_file]:
                        _STREAM_FACTS.pop(identity)['_change_watch'].close()


def _remember_stream_facts(path: Path, entry: dict, *, emit_last_byte=True) -> None:
    if (not isinstance(entry.get("grib2_messages"), int)
            or not isinstance(entry.get("sha256"), str)):
        return  # Compatibility with a backbone predating stream receipts.
    published = entry.get("publication")
    watch = None
    if isinstance(published, dict):
        # Watch before the identity is read, so a change after the read is
        # noticed even when it lands inside the same change-time tick.
        from woof.file_change_watch import FileChangeWatch
        watch = FileChangeWatch(path)
    try:
        identity = _stat_identity(path)
        if entry.get("bytes") != identity[3]:
            raise ValueError(f"{path.name}: Rust stream receipt byte count differs from the published file; refuses a changed payload")
        if watch is not None and any(published.get(key) != value
                                     for key, value in _publication(identity).items()):
            raise ValueError(f"{path.name}: payload identity changed after Rust publication; refuses binding a digest to changed bytes")
    except BaseException:
        if watch is not None:
            watch.close()
        raise
    if watch is not None:
        with _STREAM_FACTS_LOCK:
            previous = _STREAM_FACTS.pop(identity, None)
            if previous is not None:
                previous['_change_watch'].close()
            _STREAM_FACTS[identity] = {**entry, '_change_watch': watch}
            while len(_STREAM_FACTS) > 4096:
                _STREAM_FACTS.popitem(last=False)[1]['_change_watch'].close()
    # Platforms without a change-time identity retain independent disk checks.
    if emit_last_byte and isinstance(entry.get("last_byte_unix"), (int, float)):
        from woof.timeline.ledger import emit
        emit("fetch.last_byte", attrs={"name": path.name,
             "range_completed_unix": entry["last_byte_unix"],
             "receipt_delay_seconds": time.time() - entry["last_byte_unix"],
             "boundary": "verified range response completion"})


def _rename_stream_payload(landed: Path, dest: Path, entry: dict) -> None:
    """Verify the published inode, rename it, then rebind its identity.

    A rename changes ctime on Unix and ChangeTime on Windows. It may
    change neither the bytes nor the inode, so every other identity
    component must survive it.
    """
    _remember_stream_facts(landed, entry, emit_last_byte=False)
    os.replace(landed, dest)
    published = entry.get("publication")
    if isinstance(published, dict):
        actual = _publication(_stat_identity(dest))
        if any(actual[key] != published.get(key) for key in actual if key != "ctime_ns"):
            raise ValueError(f"{dest.name}: payload identity changed during rename; refuses binding changed bytes")
        entry = {**entry, "publication": actual}
    _remember_stream_facts(dest, entry)


def _fetched_digest(path: Path) -> str:
    entry = _stream_entry(_stat_identity(path))
    return entry["sha256"] if entry is not None else sha256_file(path)


def _fetched_records(path: Path) -> int:
    entry = _stream_entry(_stat_identity(path))
    return entry["grib2_messages"] if entry is not None else count_grib2_messages(path)


#: sha256 of files this process has already read whole, keyed by every
#: stat field that moves when a file's bytes can.  A finished folder's
#: re-run asks "are these still the bytes the receipt recorded?" at the
#: front door, again in the route that lets the receipt stand in for the
#: live index, and a third time in the verify-skip; without this each
#: asking read every file again.
_DIGEST_MEMO: OrderedDict[tuple, str] = OrderedDict()
_DIGEST_MEMO_LOCK = threading.Lock()
_DIGEST_MEMO_ENTRIES = 4096
#: A file changed this recently can still change again inside its
#: timestamps' resolution without either one moving, so its digest is
#: read afresh next time instead of being remembered.
_DIGEST_SETTLED_NS = 2_000_000_000
#: The wall clock the settling rule reads.
_digest_clock_ns = time.time_ns


def _stat_identity(path: Path) -> tuple:
    # Not os.stat on Windows, whose st_ctime is the creation time; see
    # woof.file_change_watch.stat_identity.
    from woof.file_change_watch import stat_identity
    return (os.path.abspath(path), *stat_identity(path))


def existing_file_digest(path: Path) -> str:
    """sha256 of a file already on disk, read once while it is unchanged.

    For the checks that ask whether an existing file still holds the
    bytes a receipt recorded.  The answer is remembered under the file's
    path, device, inode, size, modification and change times, so any
    write to it (and any replacement of it) is a new question; a file
    written in the last two seconds is never remembered at all.  A file
    just downloaded is hashed with :func:`sha256_file`, because it has
    never been asked about before.
    """

    identity = _stat_identity(path)
    # Same-process manifest and supplement binding may ask again immediately
    # after publication. The trusted stream already hashed these bytes. A
    # resumed process has no stream facts, and any file mutation changes the key.
    streamed = _stream_entry(identity)
    if streamed is not None:
        return streamed["sha256"]
    with _DIGEST_MEMO_LOCK:
        known = _DIGEST_MEMO.get(identity)
        if known is not None:
            _DIGEST_MEMO.move_to_end(identity)
            return known
    digest = sha256_file(path)
    settled = (_digest_clock_ns() - max(identity[4], identity[5])
               >= _DIGEST_SETTLED_NS)
    if settled and _stat_identity(path) == identity:
        with _DIGEST_MEMO_LOCK:
            _DIGEST_MEMO[identity] = digest
            while len(_DIGEST_MEMO) > _DIGEST_MEMO_ENTRIES:
                _DIGEST_MEMO.popitem(last=False)
    return digest


def count_grib2_messages(path: Path) -> int:
    """Walk and validate every GRIB2 envelope; return the message count.

    Fail-closed transport check mirroring the GRIB1 envelope walk in
    :mod:`woof.ingest.grib`: every message must declare edition 2, its
    exact length, and close with ``7777``; the messages must tile the
    file exactly.
    """

    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"GRIB2 file {path} is empty")
    count = 0
    with path.open("rb") as stream:
        offset = 0
        while offset < size:
            stream.seek(offset)
            header = stream.read(16)
            if len(header) != 16 or header[:4] != b"GRIB":
                raise ValueError(
                    f"invalid GRIB2 file {path}: message {count} at byte "
                    f"{offset} lacks a GRIB indicator")
            if header[7] != 2:
                raise ValueError(
                    f"unsupported GRIB edition {header[7]} in {path}, "
                    f"message {count} at byte {offset}")
            length = int.from_bytes(header[8:16], "big")
            end = offset + length
            if length < 20 or end > size:
                raise ValueError(
                    f"truncated GRIB2 file {path}: message {count} at byte "
                    f"{offset} declares {length} bytes, file has {size}")
            stream.seek(end - 4)
            if stream.read(4) != b"7777":
                raise ValueError(
                    f"invalid GRIB2 file {path}: message {count} at byte "
                    f"{offset} lacks the 7777 terminator")
            count += 1
            offset = end
    return count


def _atomic_write_text(path: Path, text: str) -> None:
    """Publish a receipt whole, or leave the previous one alone.

    The staging name used to be a fixed ``<name>.tmp``, which is exactly
    the file two publishers collide on -- one could be renaming the
    other's half-written bytes onto a canonical receipt.  The shared
    helper stages under a per-process, per-call name and fsyncs before
    the rename, so a crash leaves either the old receipt or the new one
    and never a torn or foreign one.
    """

    fetch_guard.atomic_write_text(path, text, tag="fetch")


def write_fetch_manifest(out: Path, payload: dict) -> Path:
    path = out / FETCH_MANIFEST_NAME
    _atomic_write_text(
        path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def fetch_throughput(out: Path) -> dict | None:
    """What the fetch in ``out`` moved, and how fast, or ``None``.

    Read back out of ``fetch-manifest.json`` -- the artifact, never a
    printed line -- so a chain reporting fetch bandwidth is relaying a
    receipt rather than re-deriving one.

    ``bytes_per_second`` IS BANDWIDTH AND ONLY BANDWIDTH: it is computed
    over the files this run actually downloaded, and it is ``None`` when
    this run downloaded nothing.  The distinction is not pedantry.  A
    re-run against an existing ``--data-dir`` skips every download and
    only re-hashes what is on disk, and dividing those bytes by those
    seconds produced **1.09 GB/s** on the reference box -- a true
    number about sha256, presented under a name that means the network.
    An instrument that confidently reports a wrong-by-two-orders answer
    on the most ordinary re-run there is would be worse than no
    instrument, so the verified bytes are reported separately, by name.

    ``None`` when there is no readable manifest.  Per-file ``seconds``
    is absent from manifests written before it existed, and absent is
    said as ``None`` rather than as zero.
    """

    payload = _load_fetch_manifest(Path(out))
    if payload is None:
        from woof.fetch_adoption import route_throughput
        return route_throughput(Path(out))
    files = payload.get("files")
    if not isinstance(files, list):
        return None
    total_bytes = 0
    seconds = 0.0
    timed = 0
    downloaded_bytes = 0
    downloaded_seconds = 0.0
    downloaded = 0
    for entry in files:
        if not isinstance(entry, dict):
            continue
        size = entry.get("bytes")
        size = int(size) if isinstance(size, (int, float)) else 0
        total_bytes += size
        elapsed = entry.get("seconds")
        elapsed = float(elapsed) if isinstance(elapsed, (int, float)) else None
        if elapsed is not None:
            seconds += elapsed
            timed += 1
        # A manifest that predates this key says nothing either way, so
        # it is not counted as a download; its bytes still show up in
        # `bytes`, and `bytes_per_second` stays None, which is accurate.
        if entry.get("downloaded") is True:
            downloaded += 1
            downloaded_bytes += size
            if elapsed is not None:
                downloaded_seconds += elapsed
    concurrency = payload.get("concurrency")
    return {
        "files": len(files),
        "bytes": total_bytes,
        # The SERIAL MODEL of the stage: the sum of per-file seconds.
        # Under the pooled default transfers overlap, so the wall the
        # caller actually waited is `concurrency.wall_seconds`; this sum
        # is what the same request would have cost one file at a time.
        "seconds": round(seconds, 6) if timed else None,
        "files_timed": timed,
        "downloaded_files": downloaded,
        "adopted_files": 0,
        "reused_files": len(files) - downloaded,
        "downloaded_bytes": downloaded_bytes,
        "downloaded_seconds": (round(downloaded_seconds, 6)
                               if downloaded else None),
        "bytes_per_second": (round(downloaded_bytes / downloaded_seconds, 1)
                             if downloaded and downloaded_seconds > 0.0
                             else None),
        "verified_files": len(files) - downloaded,
        "verified_bytes": total_bytes - downloaded_bytes,
        # The pool receipt (files, bytes, workers, host caps, wall,
        # modeled serial seconds, effective speedup); None when the
        # manifest predates it or the run was interrupted.
        "concurrency": (dict(concurrency)
                        if isinstance(concurrency, dict) else None),
    }


def _load_fetch_manifest(out: Path) -> dict | None:
    """The prior fetch manifest payload in ``out``, or None.

    Malformed or foreign JSON yields None rather than an error: the
    per-file completeness bars still apply, so an unreadable manifest
    only ever loses the request-identity comparison, never safety.
    """

    path = out / FETCH_MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError, OSError):
        return None
    if (not isinstance(payload, dict)
            or payload.get("schema") != FETCH_MANIFEST_SCHEMA):
        return None
    return payload


def check_prior_request(out: Path, *, source: str, cycle: datetime,
                        area: Area | None) -> None:
    """Refuse resuming into ``out`` unless the recorded request matches.

    The per-file resume check verifies envelopes and record counts, but a
    GFS subset carries the same 124 records for ANY area, so re-fetching
    a different area (or cycle) into the same ``--out`` would silently
    keep the old files.  The fetch manifest records the request; a
    source, cycle, or area difference refuses with the exact mismatch
    and the remedy.  Forecast-hour changes alone stay resumable: files
    are per-hour and byte-identical for the same source/cycle/area, so
    extending the window is safe by construction.

    A nonempty ``out`` WITHOUT a readable manifest refuses too: with no
    recorded request there is nothing to tie the existing files to (a
    legacy interrupted fetch from before incremental manifests, a
    corrupted manifest, or a directory some other tool wrote), and the
    per-file bars are area-blind, so resuming would bless files this
    request cannot verify.  Only a directory that is absent or empty may
    be fetched into without a manifest.
    """

    prior = _load_fetch_manifest(out)
    if prior is None:
        # The as-posted loop's own posting/ folder is written before the
        # first byte moves (its schedule comes first), so it is not a
        # stranger's file.
        if out.is_dir() and any(entry.name != "posting"
                                for entry in out.iterdir()):
            raise ValueError(layered(
                f"--out {out} is not empty but carries no readable "
                f"{FETCH_MANIFEST_NAME}, so its files are UNVERIFIED for "
                "this request and will not be resumed.\n"
                "  remedy: fetch into a different --out, or pass "
                "--force-refetch to move the existing files aside "
                "(nothing is deleted) and re-download this request.",
                "  why: a missing manifest is a legacy interrupted fetch, "
                "a corrupted manifest, or files another tool put there.  "
                "The existing files cannot be tied to any recorded "
                "source/cycle/area, and the per-file resume check is "
                "area-blind, so resuming onto them would publish a "
                "receipt describing bytes nobody recorded."))
        return
    requested = {
        "source": source,
        "cycle": cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "area": None if area is None else area.as_manifest(),
    }
    existing = {key: prior.get(key) for key in requested}
    differences = [
        f"  {key}: requested {requested[key]!r}, but {out} was fetched "
        f"with {existing[key]!r}"
        for key in requested if requested[key] != existing[key]]
    if differences:
        raise ValueError(layered(
            f"--out {out} already holds a fetch for a different request:\n"
            + "\n".join(differences)
            + "\n  remedy: fetch into a different --out, or pass "
            "--force-refetch to move the existing files aside (nothing "
            "is deleted) and re-download this request.",
            "  why: the per-file resume check cannot tell the difference "
            "-- a subset file passes its record-count bar for any area -- "
            "so resuming would silently mix two requests' bytes under one "
            "manifest."))


def require_matching_request(out: Path, *, source: str, cycle: datetime,
                             area: Area | None,
                             mode: str | None = None) -> None:
    """The request-identity and transfer-mode check, for every door.

    :func:`check_prior_request` plus the transfer mode.  Each public
    fetch API calls this inside its own output lock, so a library caller
    gets the same refusal the command line does: a GFS or HRRR file name
    carries the cycle HOUR but not the date, so without this a second
    day's request into the same folder found the first day's files,
    passed their per-file bars, and published them under the new date.

    ``mode`` is compared only when given.  The GFS transports name and
    verify their files differently, so resuming one onto the other would
    mix two requests' bytes; HRRR's modes share names and bars and pass
    None.
    """

    check_prior_request(out, source=source, cycle=cycle, area=area)
    if mode is None:
        return
    prior = _load_fetch_manifest(out)
    if prior is None:
        return
    recorded_mode = prior.get("mode") or "nomads-cgi-subset"
    if recorded_mode != mode:
        raise ValueError(layered(
            f"--out {out} already holds a {recorded_mode} fetch and this "
            f"request is {mode}.\n"
            "  remedy: fetch into a different --out, or pass "
            "--force-refetch to move the existing files aside (nothing is "
            "deleted) and re-download this request.",
            "  why: the two transports name their files differently and "
            "verify them against different bars, so resuming one onto the "
            "other would publish a manifest mixing two requests' bytes."))


def cached_request_complete(out: Path, *, source: str, cycle: datetime,
                            area: Area | None, hours: tuple[int, ...],
                            mode: str | None = None,
                            progress=None,
                            refuse_changed: bool = False) -> bool:
    """Does ``out`` already hold every file this exact request needs?

    Answered from the local receipt alone, before any provider is asked:
    the recorded source, cycle, area and (when given) mode must match,
    every requested hour must be one the receipt declares complete, and
    every payload file it lists for those hours must still be on disk
    holding the bytes whose sha256 it recorded.  A yes lets the fetch
    skip the publication probe and keep the host the receipt names, so a
    finished download stays usable after the provider has rolled the
    cycle off or while the network is down.

    The digests are part of the answer, not left to the transfer: a yes
    means nothing will be downloaded, and a file damaged in place at its
    own size was otherwise re-fetched from the receipt's host without
    anyone asking whether that host still keeps the cycle, so a folder
    fetched from the operational server met a 404 once the cycle aged
    off it.  A no sends the fetch the way a fresh one goes.

    ``progress`` hears what is being checked, when every file is there
    to check, and which file failed.

    ``refuse_changed`` is for the GFS routes.  They refuse a file whose
    bytes moved rather than fetch it again, so for them a damaged file
    is not "ask the provider after all": that question could not change
    the answer, and offline it was answered as "not published yet".
    With it set, a changed file raises the refusal the route itself
    gives, before any provider is asked.
    """

    prior = _load_fetch_manifest(out)
    if prior is None or not hours:
        return False
    try:
        require_matching_request(out, source=source, cycle=cycle, area=area,
                                 mode=mode)
    except ValueError:
        return False
    recorded = prior.get("forecast_hours")
    if not isinstance(recorded, list) or not set(hours) <= set(recorded):
        return False
    wanted = set(hours)
    payload = [entry for entry in prior.get("files") or ()
               if isinstance(entry, dict)
               and entry.get("forecast_hour") in wanted]
    if {entry.get("forecast_hour") for entry in payload} != wanted:
        return False
    if source == "hrrr" and _hrrr_receipt_has_old_subsets(prior, payload):
        # Every file is here, but a subset predates the optional surface
        # records, so the fetch takes it again; it goes the way a fresh
        # one does.
        if progress is not None:
            progress(f"fetch {source}: {out} holds .idx subsets fetched "
                     "before subsets took PBLH/LCDC/MCDC/HCDC; the fetch "
                     "takes them again")
        return False
    vouched: list[tuple[Path, str]] = []
    for entry in payload:
        name = entry.get("name")
        digest = entry.get("sha256")
        if (not isinstance(name, str) or not name
                or not isinstance(digest, str)):
            return False
        path = out / name
        if not path.is_file() or path.stat().st_size != entry.get("bytes"):
            return False
        vouched.append((path, digest))
    if progress is not None:
        progress(f"fetch {source}: every file of this request is already "
                 f"in {out}; checking them here without asking the "
                 "provider")
    changed = _first_changed_file(vouched)
    if changed is not None:
        if refuse_changed:
            _refuse_changed_file(changed)
        if progress is not None:
            progress(f"fetch {source}: {changed.name} no longer holds the "
                     "bytes its receipt recorded, so the provider is "
                     "asked about this cycle after all")
        return False
    return True


def resume_digest_refusal(name: str) -> str:
    """Why a GFS file on disk cannot be resumed for this request."""

    return (f"existing {name} does not match the sha256 recorded in the "
            "prior fetch manifest, so it cannot be resumed for this "
            "request; pass --force-refetch to move the existing files "
            "aside (nothing is deleted) and re-download")


def refuse_changed_on_disk(out: Path, names, prior_digests: dict[str, str]
                           ) -> None:
    """Refuse, before anything is published, a file whose bytes moved.

    For the GFS routes, which refuse such a file rather than fetch it
    again.  They compare each file already on disk with the receipt as
    its transfer runs, and publish the receipt again as the verified
    prefix grows, so a damaged later hour was refused only after the
    receipt had been rewritten without it; the next run, with no
    recorded digest left to compare, took the damaged file as its own.
    Checked here first, every file the receipt binds is compared while
    the receipt still binds it: a refusal leaves the receipt as it was
    and stays a refusal.
    """

    changed = _first_changed_file([
        (out / name, prior_digests[name]) for name in names
        if name in prior_digests and (out / name).is_file()])
    if changed is not None:
        _refuse_changed_file(changed)


def _refuse_changed_file(path: Path) -> None:
    """Refuse a GFS file whose bytes moved, naming what moved.

    A file that is no longer whole GRIB is refused for that, with the
    envelope walk's finding, as the crop route's own check always
    refused it; any other change is refused as a file that no longer
    matches its receipt.  Both carry the remedy, and name the file
    rather than the machine path it sits at.
    """

    try:
        count_grib2_messages(path)
    except ValueError as error:
        finding = str(error).replace(str(path), path.name)
        raise ValueError(
            f"existing {path.name} is no longer a whole GRIB2 file "
            f"({finding}), so it cannot be resumed for this request; "
            "pass --force-refetch to move the existing files aside "
            "(nothing is deleted) and re-download") from None
    raise ValueError(resume_digest_refusal(path.name))


def _first_changed_file(files: list[tuple[Path, str]]) -> Path | None:
    """The first ``(path, sha256)`` whose file no longer has that digest.

    Read on the fetch's own file-worker count, since a finished request
    can be tens of gigabytes; every digest read here is remembered for
    the verify-skip that follows.
    """

    if not files:
        return None

    def differs(item: tuple[Path, str]) -> bool:
        path, recorded = item
        try:
            return existing_file_digest(path) != recorded
        except OSError:
            return True

    workers = min(len(files), fetch_pool.resolve_file_workers(None))
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="gpuwm-verify") as pool:
        verdicts = list(pool.map(differs, files))
    return next((path for (path, _recorded), changed
                 in zip(files, verdicts) if changed), None)


def latest_cycle_request(args) -> tuple[str, int, dict]:
    """What ``--cycle latest`` asks, from arguments the fetch parser read.

    Returns ``(source, last_hour, options)`` for
    :func:`resolve_latest_cycle`.  The fetch command and every planner
    that resolves ``latest`` ahead of it ask through this one function,
    with the namespace the real parser produced, so the source, the end
    of the window and the pinned host cannot be read two ways: a planner
    that searched the raw argument list for ``--source`` missed
    ``--source=gfs`` and resolved another source's cycle.
    """

    def stated(**options) -> dict:
        # Only what the request says: an unset option is the resolver's
        # own default, so it is left for the resolver to supply.
        return {key: value for key, value in options.items()
                if value is not None}

    source = args.source
    start = args.forecast_start_hour
    if args.hours is None:
        raise ValueError(
            "--cycle latest needs --hours: the newest cycle is the newest "
            "one published through the end of the window, and the window "
            "has no end without it")
    # As posted, latest is the newest cycle whose START needs are out
    # (DESIGN A136 2.2); the whole-cycle rule is --whole-cycle's.
    as_posted = requested_as_posted(args)
    if not as_posted and not fetch_endpoints.aws_fetch_policy():
        as_posted = None
    if source in fetch_routes.route_ids():
        # A table route names its own hosts, and the fetch refuses any
        # other word in the route's own terms; the resolver asks the
        # route the same way, so a plan is refused as the fetch would be.
        begin = 0 if start is None else start
        return source, begin + args.hours, stated(
            cadence=args.cadence, start_hour=begin,
            member=getattr(args, "member", None),
            transport=getattr(args, "transport", None), as_posted=as_posted)
    transport = fetch_endpoints.policy_transport(
        source, pinned_host(getattr(args, "transport", None)))
    if source in GFS_CONTAINER_SOURCES:
        hours = container_forecast_hours(source, args.hours, args.cadence,
                                         start)
        return source, hours[-1], stated(
            transport=transport, as_posted=as_posted,
            start_hour=hours[0] if as_posted else None,
            cadence=(hours[1] - hours[0] if as_posted and len(hours) > 1
                     else None))
    if source == "hrrr":
        begin = _forecast_start_hour(start)
        return (source, begin + args.hours,
                stated(transport=transport, as_posted=as_posted,
                       start_hour=begin if as_posted else None))
    # A provider the source's archive windows name (ERA5's keyless ARCO
    # copy) can trail the source's own delay; the resolver asks about it.
    return source, args.hours, stated(
        provider=getattr(args, "era5_provider", None))


def pinned_host(transport: str | None) -> str | None:
    """The one host a ``--transport`` value pins, or None for none.

    ``auto`` is the unpinned default written out: the parser accepts it,
    the HRRR refusals recommend it, and the HRRR transfer treats it as
    "walk the ladder".  Every check that asks a host whether a cycle is
    there must read it the same way, because handing ``auto`` on as a
    host name refused every HRRR fetch that spelled the default out.
    """

    return None if transport in (None, "auto") else transport


def _recorded_hrrr_transport(out: Path) -> str:
    """The host a complete HRRR folder's files came from ('s3' if unsaid)."""

    prior = _load_fetch_manifest(out) or {}
    for entry in prior.get("files") or ():
        if (isinstance(entry, dict)
                and entry.get("transport") in HRRR_TRANSPORTS[1:]):
            return entry["transport"]
    return "s3"


def _prior_manifest_entries(out: Path) -> dict[str, dict]:
    """``name -> file entry`` from the prior fetch manifest, else empty.

    A file already on disk is verified rather than moved, and when its
    sha256 matches the entry recorded for it, that entry still says
    where its bytes came from and what census they were admitted
    against.  Carrying those forward keeps a re-run's receipt naming the
    host that actually served each file, and needs no host to be asked.
    """

    prior = _load_fetch_manifest(out) or {}
    return {entry["name"]: entry for entry in prior.get("files") or ()
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
            and isinstance(entry.get("sha256"), str)}


def _complete_request_receipt(out: Path, *, force: bool, source: str,
                              cycle: datetime, area: Area | None,
                              hours: tuple[int, ...],
                              mode: str) -> dict | None:
    """The prior receipt when ``out`` already holds this whole request.

    None under ``force`` or when any file is missing; a file whose bytes
    moved is refused here, before the live index is read, with the
    refusal the route's own check gives it.  A GFS route uses
    it to take the level ladder and record bar the receipt recorded when
    the files were fetched, instead of reading the live index again: the
    files are pinned by the digests in that same receipt, so a second
    read of the index could not change what they contain, and on a
    network that drops traffic each read waited out its full timeout
    before the files on disk were used.
    """

    if force or not cached_request_complete(
            out, source=source, cycle=cycle, area=area, hours=hours,
            mode=mode, refuse_changed=True):
        return None
    return _load_fetch_manifest(out)


def _recorded_published_levels(receipt: dict | None
                               ) -> tuple[float, ...] | None:
    """The published isobaric ladder a receipt recorded, or None."""

    if receipt is None:
        return None
    levels = receipt.get("published_pressure_levels_hpa")
    if (not isinstance(levels, list) or not levels
            or not all(isinstance(level, (int, float))
                       and not isinstance(level, bool) for level in levels)):
        return None
    return tuple(float(level) for level in levels)


#: What :func:`_recorded_derived_bar` answers when the receipt cannot
#: stand in for the live index.  Not None: None is a recorded answer
#: ("the index could not be read, the certified count stood in").
_UNRECORDED = object()


def _recorded_derived_bar(receipt: dict | None, kind: str,
                          levels: tuple[float, ...]):
    """The live census a receipt recorded for this exact selection.

    Only a receipt whose decode ladder is the one this request asks for
    can answer, because the count is a function of the ladder; anything
    else returns :data:`_UNRECORDED` and the caller reads the index.
    """

    if not _recorded_levels_match(receipt, levels):
        return _UNRECORDED
    for bar in receipt.get("record_bars") or ():
        if not isinstance(bar, dict) or bar.get("kind") != kind:
            continue
        derived = bar.get("derived")
        if derived is None or (isinstance(derived, int)
                               and not isinstance(derived, bool)):
            return derived
    return _UNRECORDED


def _recorded_levels_match(receipt: dict | None,
                           levels: tuple[float, ...]) -> bool:
    """Does the receipt's decode ladder equal the one this request asks?"""

    if receipt is None:
        return False
    recorded = receipt.get("pressure_levels_hpa")
    if not isinstance(recorded, list):
        return False
    try:
        return ([float(level) for level in recorded]
                == [float(format(float(level), "g")) for level in levels])
    except (TypeError, ValueError):
        return False


def _engine_selection(engine: str, selection: str | None) -> str:
    """How the downloader was chosen, for a caller that did not say.

    A caller that resolved the engine through
    :func:`select_fetch_engine` passes the answer.  A caller that
    resolved it some other way -- a library, a test, an older script --
    gets the accurate default: rust was found, or python was named.  It
    never guesses "python-fallback", because claiming a degrade that did
    not happen would make the field useless for the one thing it exists
    to answer.
    """

    if selection is not None:
        if selection not in FETCH_ENGINE_SELECTIONS:
            raise ValueError(
                f"unknown engine selection {selection!r}; expected one of "
                f"{FETCH_ENGINE_SELECTIONS}")
        return selection
    return "rust" if engine == "rust" else "python-requested"


def _manifest_payload(*, source: str, cycle: datetime,
                      hours: tuple[int, ...], area: Area | None,
                      files: list[dict]) -> dict:
    return {
        "schema": FETCH_MANIFEST_SCHEMA,
        "source": source,
        "cycle": cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecast_hours": list(hours),
        "area": None if area is None else area.as_manifest(),
        "files": files,
        "payload_bytes": sum(item["bytes"] for item in files),
        "created": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
    }


# ---------------------------------------------------------------------------
# GFS
# ---------------------------------------------------------------------------

def _gfs_file_levels(path: Path, source: str) -> tuple[tuple[float, ...], dict]:
    """Read the archived object's complete pressure levels without decoding fields.

    An unavailable live index cannot establish an old archive's upper ladder.
    In particular, a 2020 object carries temperature at 40 hPa but no humidity
    or wind there. The modern fallback table used to declare that partial level.
    """
    import csv
    import io
    import subprocess
    from woof.bridges import find_bridge, bridge_remedy

    executable = find_bridge("grib2_inventory")
    if executable is None:
        raise RuntimeError(bridge_remedy("grib2_inventory"))
    started = time.perf_counter()
    result = subprocess.run([str(executable), str(path)], capture_output=True,
                            text=True, check=True)
    # GRIB2 discipline/category/parameter codes, not display-name aliases.
    required = {(0, 3, 5), (0, 0, 0), (0, 1, 0 if source == "gdas" else 1),
                (0, 2, 2), (0, 2, 3)}
    fields = {key: set() for key in required}
    rows = csv.DictReader(io.StringIO("\n".join(
        line for line in result.stdout.splitlines() if not line.startswith("#"))), delimiter="\t")
    for row in rows:
        key = tuple(int(row[name]) for name in ("discipline", "category", "parameter"))
        if key in fields and int(row["level_type"]) == 100:
            level = float(row["level_value"]) / 100
            if math.isfinite(level) and level > 0:
                fields[key].add(level)
    levels = tuple(sorted(set.intersection(*fields.values())))
    if not levels:
        raise ValueError(f"{path}: no complete {source} pressure level in native inventory")
    return levels, dict(method="native-file-inventory", file=path.name,
        inventory_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
        wall_seconds=round(time.perf_counter() - started, 6))


def gfs_live_index(cycle: datetime, *, progress=print, opener=None,
                   source: str = "gfs") -> str | None:
    """The live ``.idx`` behind one GFS/GDAS cycle, or None.

    The NOMADS CGI subset has no index of its own -- it *is* the subset
    -- so the live inventory is the ``.idx`` of the corresponding full
    ``pgrb2.0p25`` object on S3.  Two questions are answered from this
    one document: how many records the selection yields (the record
    bar) and which isobaric levels the product publishes (the ladder a
    requested model top is resolved against).  Reading it once means
    both answers describe the same generation of the same object.

    Returns None when it cannot be read; callers stand the certified
    constants in and say so.  A transient S3 blip must not stop a fetch
    whose own record count is checked anyway.
    """

    try:
        url = f"{gfs_object_url(cycle, 0, source)}.idx"
    except ValueError as error:
        progress(f"fetch {source}: no index layout for this cycle ({error}); "
                 "the declared pressure ladder stands in; Rust validates the fields")
        return None
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with (opener or paced_urlopen)(request, timeout=120) as response:
            return response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, OSError, ValueError) as error:
        progress(f"fetch {source}: could not read the live inventory at "
                 f"{url} ({error}); the certified constants stand in")
        return None


def gfs_available_levels(cycle: datetime, *, progress=print, opener=None,
                         source: str = "gfs",
                         index_text: str | None = None
                         ) -> tuple[float, ...]:
    """Which isobaric levels this cycle publishes for every 3-D field.

    The live index first; the captured certified ladder only when it
    cannot be read.  Deciding which levels a requested model top needs
    is a claim about what the product carries, and the product is the
    authority on that -- but a fetch must not fail because S3 hiccuped,
    so the certified fallback is named out loud when it is used.
    """

    from tools import download_gfs_native_subset as transport

    if index_text is None:
        index_text = gfs_live_index(cycle, progress=progress, opener=opener,
                                    source=source)
    if index_text is not None:
        levels = transport.available_levels_from_index(index_text, model=source)
        if levels:
            return levels
        progress(f"fetch {source}: the live inventory names no isobaric "
                 "level carrying all of "
                 f"{', '.join(transport.PRESSURE_FIELDS)}; the certified "
                 "ladder stands in")
    return transport.CERTIFIED_AVAILABLE_LEVELS_HPA


#: The GRIB2 (discipline, category, number) of each 3-D field the selection
#: takes on every isobaric level (``PRESSURE_FIELDS``; GDAS carries SPFH
#: where GFS carries RH).
_GFS_PRESSURE_PARAMETERS = {
    "HGT": (0, 3, 5), "TMP": (0, 0, 0), "RH": (0, 1, 1), "SPFH": (0, 1, 0),
    "UGRD": (0, 2, 2), "VGRD": (0, 2, 3),
}


def gfs_object_levels(path: Path, *, source: str = "gfs") -> tuple[float, ...]:
    """The isobaric levels (hPa) one downloaded GFS/GDAS object carries for
    every 3-D field, read from the object by the engine's Rust GRIB2
    inventory; ``()`` when the inventory cannot be run or names none.

    The object is the authority when its live ``.idx`` cannot be read: an
    archive copy with no index (NCAR GDEX d084001) of a cycle from before
    the 2021-03-22 12Z GFS v16 implementation carries 1-70 hPa as 1, 2, 3,
    5, 7, 10, 20, 30, 50, 70, not v16's 15 and 40 hPa, so the certified
    v16 ladder declared levels the file does not hold and the bridge
    refused it ("missing required GFS field RH at 4000 Pa").
    """

    from woof.bridges import find_bridge
    from woof.mapped_source import _grib2_inventory
    from tools import download_gfs_native_subset as transport

    executable = find_bridge("grib2_inventory")
    if executable is None:
        return ()
    try:
        rows = _grib2_inventory(Path(path), executable)
    except (ValueError, RuntimeError, OSError):
        return ()
    names = tuple("SPFH" if source == "gdas" and name == "RH" else name
                  for name in transport.PRESSURE_FIELDS)
    per_field: dict[str, set[float]] = {name: set() for name in names}
    for row in rows:
        try:
            if int(row["level_type"]) != 100:
                continue
            key = (int(row["discipline"]), int(row["category"]),
                   int(row["parameter"]))
            level = float(format(float(row["level_value"]) / 100.0, "g"))
        except (KeyError, ValueError):
            continue
        for name in names:
            if _GFS_PRESSURE_PARAMETERS[name] == key:
                per_field[name].add(level)
    if not all(per_field.values()):
        return ()
    return tuple(sorted(set.intersection(*per_field.values())))


def gfs_derived_record_bar(cycle: datetime, *, progress=print,
                           opener=None, source: str = "gfs",
                           levels_hpa: tuple[float, ...] | None = None,
                           index_text: str | None = None) -> int | None:
    """How many records the GFS selection yields in the live inventory.

    The NOMADS CGI subset has no index of its own -- it *is* the subset
    -- so the live inventory is the ``.idx`` of the corresponding full
    ``pgrb2.0p25`` object on S3, and the bar is the count of records in
    it whose ``(variable, level)`` the CGI query asks for.  Both sides
    come from the same declaration in
    :mod:`tools.download_gfs_native_subset`, so there is no second table
    to drift.

    ``levels_hpa`` is the ladder this request actually asks for; the
    count is derived against it, so a run with a deeper model top is
    measured against its own selection rather than the default one.
    ``index_text`` lets the caller hand over an index it has already
    read, so the ladder and the count describe one generation of one
    object rather than two reads that could straddle a publication.

    Returns ``None`` when the index cannot be read; the caller then
    stands the certified constant in and says so.  A transient S3 blip
    must not stop a fetch whose own record count is checked anyway.
    """

    from woof.fetch_bars import count_index_selection, nomads_selector_pairs
    from tools import download_gfs_native_subset as transport

    if index_text is None:
        index_text = gfs_live_index(cycle, progress=progress, opener=opener,
                                    source=source)
    if index_text is None:
        return None
    if levels_hpa is None:
        levels_hpa = transport.PRESSURE_LEVELS_HPA
    return count_index_selection(index_text, nomads_selector_pairs(
        transport.nomads_variables(source), transport.NOMADS_LEVELS, levels_hpa))


def container_subset_levels(source: str, *,
                            top_pressure_pa: float | None = None,
                            all_levels: bool = False,
                            available: tuple[float, ...] | None = None
                            ) -> tuple[float, ...]:
    """The isobaric ladder a grib-filter request of ``source`` selects.

    One decision for the fetch and for the price of the fetch: every
    level the product publishes under ``--all-levels`` (and for GDAS when
    no top is named), otherwise the certified ladder extended upward to
    ``top_pressure_pa``.  ``available`` is the ladder the live index
    publishes; the captured inventory stands in when none is given.
    """

    from tools import download_gfs_native_subset as transport

    if available is None:
        available = transport.CERTIFIED_AVAILABLE_LEVELS_HPA
    if _subset_takes_whole_ladder(source, top_pressure_pa, all_levels):
        return tuple(float(level) for level in available)
    return transport.levels_for_top(top_pressure_pa, available=available)


def _subset_takes_whole_ladder(source: str, top_pressure_pa, all_levels) -> bool:
    return bool(all_levels or source == "gdas" and top_pressure_pa is None)


def _ladder_flags(top_pressure_pa, all_levels) -> list[str]:
    """The ``woof fetch`` flags that ask for this request's ladder again."""

    if all_levels:
        return ["--all-levels"]
    if top_pressure_pa is not None:
        return ["--p-top-pa", f"{float(top_pressure_pa):g}"]
    return []


def _existing_crop_refusal(name: str, observed: int, expected: int,
                           levels: tuple[float, ...],
                           source: str = "gfs") -> str:
    """Why a crop already in the folder cannot serve this request.

    A file whose record count is exactly another ladder's was fetched
    for another model top (a folder downloaded for a 100 hPa top, handed
    to a run whose top is 50 hPa), so the refusal says that rather than
    only the two counts; decoding it would stop the source atmosphere
    under the model top, or carry levels the manifest does not record.
    """

    from tools import download_gfs_native_subset as transport

    words = (f"existing {name} carries {observed} GRIB2 messages, "
             f"expected {expected}")
    ladder = transport.CERTIFIED_AVAILABLE_LEVELS_HPA
    held = next((count for count in range(1, len(ladder) + 1)
                 if container_record_count(source, ladder[-count:]) == observed),
                None)
    if held is None or held == len(levels):
        return words + "; move it aside and re-fetch"
    return (words + f": this request takes {len(levels)} isobaric levels "
            f"up to {min(levels):g} hPa and the file holds {held}, so it "
            "was fetched for another model top. Fetch into a new folder, "
            "or move it aside and re-fetch")


def container_record_count(source: str, levels: tuple[float, ...]) -> int:
    """Records one grib-filter crop of ``source`` carries on ``levels``.

    The certified record bar the fetch holds every file to.  The declared
    selection gives five records per isobaric level plus its single-level
    records (``record_count_for_levels``).  A variable the crop asks for
    because the source's packaged mapping binds it adds one record per
    level the mapping says the publisher writes it on that the request
    asks for: GDAS's five hydrometeors are written on 22 levels
    (``published_levels``: 50..1000 hPa, no 70 hPa), 110 records on its
    whole 41-level ladder, 334 in all.
    """

    from tools import download_gfs_native_subset as transport

    declared = {name for name, _level in fetch_bars.nomads_selector_pairs(
        transport.declared_variables(source), ("lev_surface",), ())}
    mapped = {name: published for name, published
              in (mapped_record_levels(source) or {}).items()
              if name not in declared}
    asked = fetch_bars.nomads_selector_pairs(
        tuple(fetch_bars.filter_variable(name) for name in mapped),
        transport.NOMADS_LEVELS, tuple(levels))
    return transport.record_count_for_levels(len(levels)) + sum(
        (name, level) in asked
        for name, published in mapped.items() for level in published)


def container_subset_record_count(source: str, *,
                                  top_pressure_pa: float | None = None,
                                  all_levels: bool = False) -> int:
    """Records one grib-filter file of ``source`` carries for this request.

    :func:`container_record_count` on the ladder of
    :func:`container_subset_levels`.  Read against the captured
    inventory, so it answers before any index is read (the download
    price asks it).
    """

    return container_record_count(source, container_subset_levels(
        source, top_pressure_pa=top_pressure_pa, all_levels=all_levels))


def fetch_gfs(*, cycle: datetime, hours: tuple[int, ...], area: Area,
              out: Path, progress=print, force: bool = False,
              accept_inventory_change: bool = False,
              derived_bar=gfs_derived_record_bar,
              source: str = "gfs",
              top_pressure_pa: float | None = None,
              all_levels: bool = False,
              file_workers: int | None = None,
              available_levels=gfs_available_levels) -> Path:
    """Download the exact GFS pgrb2.0p25 subset series into ``out``.

    Single writer per ``--out``: the whole flow -- reading the prior
    receipt, moving files aside under ``force``, transferring, and
    publishing the new receipt -- runs under an exclusive OS lock on the
    output root, so two concurrent fetches cannot interleave into a
    manifest that describes the other one's bytes.  A second run
    announces the wait and then refuses loudly rather than proceed.

    ``top_pressure_pa`` is the model top the fetched atmosphere must
    reach.  Left ``None`` the certified 21-level ladder is fetched
    exactly as before (a 100 hPa / 10000 Pa source top); given a value
    the ladder is extended upward along whatever the live inventory says
    the product publishes, until a level sits at or above it.  A top the
    product genuinely cannot serve refuses, naming the deepest it can.
    ``all_levels`` takes every level the product carries instead.

    See :func:`_fetch_gfs_locked` for the transfer itself.
    """

    with _fetch_out(out, progress=progress):
        return _fetch_gfs_locked(
            cycle=cycle, hours=hours, area=area,
            out=deep_io_path(out, DOWNLOAD_DEPTH_BUDGET), progress=progress,
            force=force, accept_inventory_change=accept_inventory_change,
            derived_bar=derived_bar, source=source,
            top_pressure_pa=top_pressure_pa, all_levels=all_levels,
            file_workers=file_workers,
            available_levels=available_levels)


def _fetch_gfs_locked(*, cycle: datetime, hours: tuple[int, ...], area: Area,
                      out: Path, progress=print, force: bool = False,
                      accept_inventory_change: bool = False,
                      derived_bar=gfs_derived_record_bar,
                      source: str = "gfs",
                      top_pressure_pa: float | None = None,
                      all_levels: bool = False,
                      file_workers: int | None = None,
                      available_levels=gfs_available_levels) -> Path:
    """The GFS transfer, with the output-root lock already held.

    Reuses the certified NOMADS query builder and downloader in
    :mod:`tools.download_gfs_native_subset` (single source for the
    124-record selection), adds resumability (a present, envelope-valid
    subset is never re-downloaded), and writes the ``gfs-series.tsv``
    that ``rw-wps --source gfs --gfs-series`` / ``gfs_grib2_bridge``
    consume, plus the fetch manifest.  The series and manifest are
    atomically refreshed after every verified forecast hour, so an
    interrupted fetch records its complete contiguous prefix and the
    same command resumes it.  An existing file must pass the envelope
    walk, the exact 124-record count, AND -- when the prior fetch
    manifest recorded its digest -- that same sha256; a swapped file is
    never re-blessed by the area-blind count alone.  ``force`` moves
    every existing file in ``out`` aside -- receipts first, so an
    interrupted force leaves no manifest claiming replaced bytes -- and
    re-downloads.  Nothing is ever deleted.
    """

    from woof.fetch_bars import resolve_bar
    from tools import download_gfs_native_subset as transport

    if source not in GFS_CONTAINER_SOURCES:
        raise ValueError(f"fetch_gfs serves {GFS_CONTAINER_SOURCES}, not "
                         f"{source!r}")
    # The capability boundary sits here as well as at the CLI: a caller
    # reaching the library directly must hit the same refusal.
    if source == "gdas":
        beyond = [hour for hour in hours if hour > GDAS_MAX_FORECAST_HOUR]
        if beyond:
            raise ValueError(gdas_capability_refusal(beyond[0]))
    if top_pressure_pa is not None and all_levels:
        raise ValueError(
            "--p-top-pa names the model top the ladder must reach and "
            "--all-levels takes every level the product carries; they "
            "are two answers to the same question, so pass one")
    prefix = GFS_CONTAINER_PREFIX[source]
    if not force:
        # Inside the output lock, before any provider is asked: a file
        # name carries the cycle hour but not the date, so another day's
        # files would otherwise pass every per-file bar here.
        require_matching_request(out, source=source, cycle=cycle, area=area,
                                 mode="nomads-cgi-subset")
    if source == "gdas" and not force and any(out.glob("gdas.*.subset.grib2")):
        # Before the receipt answers the record bar: a folder cut to
        # another variable selection is refused for that, not as an
        # upstream inventory change.
        previous_path = out / FETCH_MANIFEST_NAME
        previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else {}
        wanted = list(transport.nomads_variables(source))
        recorded = previous.get("requested_variables")
        if recorded != wanted:
            missing = [name.removeprefix("var_") for name in wanted
                       if name not in (recorded or ())]
            raise ValueError(
                f"{out}: these GDAS crops were fetched "
                + (f"without {', '.join(missing)}"
                   if recorded is not None and missing
                   else "with another variable selection")
                + ", and the packaged GDAS mapping reads every variable "
                "this request asks for.  Fetch into a new folder, or pass "
                "--force-refetch to move the old files aside and fetch "
                "them again.")
    out.mkdir(parents=True, exist_ok=True)
    # A folder that already holds this whole request answers both index
    # questions from the receipt its files were admitted under.
    receipt = _complete_request_receipt(
        out, force=force, source=source, cycle=cycle, area=area,
        hours=hours, mode="nomads-cgi-subset")
    index_text = None
    available = _recorded_published_levels(receipt)
    from_receipt = available is not None
    if not from_receipt:
        # One read of the live index answers both questions below, so
        # the ladder and the record count describe the same generation
        # of the same object rather than two reads that could straddle
        # a publication.
        index_text = gfs_live_index(cycle, progress=progress,
                                    source=source)
        available = available_levels(cycle, progress=progress,
                                     source=source, index_text=index_text)
    levels = container_subset_levels(
        source, top_pressure_pa=top_pressure_pa, all_levels=all_levels,
        available=available)
    if _subset_takes_whole_ladder(source, top_pressure_pa, all_levels):
        progress(f"fetch {source}: --all-levels takes the whole published "
                 f"ladder, {len(levels)} isobaric levels "
                 f"({min(levels):g}..{max(levels):g} hPa)")
    elif top_pressure_pa is not None:
        extra = len(levels) - len(transport.PRESSURE_LEVELS_HPA)
        progress(
            f"fetch {source}: model top {float(top_pressure_pa):g} Pa "
            f"needs {len(levels)} isobaric levels, source top "
            f"{min(levels) * 100.0:g} Pa"
            + (f" ({extra} level(s) above the certified 100 hPa "
               "ladder)" if extra else " (the certified ladder "
               "already reaches it)"))
    source_top_pa = min(levels) * 100.0
    # One record bar for the whole request: the selection is
    # instantaneous fields only, so its census does not vary by hour.
    # The certified count is a function of THIS request's ladder --
    # five records per level plus the single-level records -- so a
    # deeper top is not mistaken for an upstream inventory change.
    derived = (_recorded_derived_bar(receipt, "gfs", levels)
               if from_receipt else _UNRECORDED)
    if derived is _UNRECORDED:
        derived = derived_bar(cycle, progress=progress, source=source,
                              levels_hpa=levels, index_text=index_text)
    bar = resolve_bar("gfs", derived,
                      accept_inventory_change=accept_inventory_change,
                      progress=progress,
                      certified=container_record_count(source, levels))
    if force:
        # Receipts first, then every other existing file: an interrupted
        # force must never leave a manifest behind that still claims a
        # payload it has already replaced.
        _force_quarantine_output(out, progress, source)
    prior_digests = _prior_manifest_digests(out)
    box = area.as_nomads()
    longitude_amplification = area.nomads_longitude_amplification
    longitude_note = None
    if longitude_amplification is not None:
        requested = (
            f"lat {area.lat_south:g}..{area.lat_north:g}, "
            f"lon {area.lon_west:g}..{area.lon_east:g}")
        fetched = (
            f"lat {box['bottom_lat']:g}..{box['top_lat']:g}, "
            f"lon {box['left_lon']:g}..{box['right_lon']:g}")
        longitude_note = (
            f"requested box {requested} crosses 0 degrees longitude, "
            "which one NOMADS [0,360] subregion cannot express; fetched "
            f"band {fetched}, a {longitude_amplification:g}x "
            "longitude-span amplification (compressed-byte "
            "amplification is data-dependent); informational only -- the "
            "ingest interpolates the domain out of the wider band, so the "
            "only cost is download size and the run continues unchanged")
        progress(f"fetch {source}: NOTE {longitude_note}")
    files: list[dict] = []
    pool_summary: dict = {}

    def publish_manifest() -> Path:
        # Another preparation may already be reading this completed series.
        # A cached recheck must not replace it with its first verified hour.
        # Fresh/interrupted downloads still publish each resumable prefix.
        if receipt is not None and len(files) < len(hours):
            return out / FETCH_MANIFEST_NAME
        # Relative names: gfs_grib2_bridge resolves them against the
        # TSV's own directory, so the fetched directory stays relocatable.
        series = out / f"{prefix}-series.tsv"
        _atomic_write_text(series, "".join(
            f"{item['forecast_hour']}\t{item['name']}\t"
            f"{81 if item['forecast_hour'] == 0 else 96}\n"
            for item in files))
        entries = files + [{
            "name": series.name, "role": "series", "forecast_hour": None,
            "bytes": series.stat().st_size, "sha256": sha256_file(series),
            "url": None,
        }]
        entries += _write_gfs_front_door_files(
            out, source=source, cycle=cycle, files=files, series=series)
        recorded_hours = tuple(
            int(item["forecast_hour"]) for item in files)
        payload = _manifest_payload(
            source=source, cycle=cycle, hours=recorded_hours,
            area=area, files=entries)
        payload["notes"] = (
            "NOMADS filter subsets (south-to-north 0.25-degree grids); raw "
            "noaa-gfs-bdp-pds S3 objects are north-to-south and are "
            "accepted by gfs_grib2_bridge only after its declared "
            "scan-order flip")
        payload["nomads_area"] = box
        if longitude_note is not None:
            payload["notes"] += f"; {longitude_note}"
            payload["longitude_span_amplification"] = (
                longitude_amplification)
        payload["engine"] = "python"
        # Not a degrade: the CGI subset route has no rust transport to
        # fall back FROM, so nobody inherited anything here.
        payload["engine_selection"] = "python-requested"
        payload["mode"] = "nomads-cgi-subset"
        payload["requested_variables"] = list(transport.nomads_variables(source))
        payload["record_bars"] = [bar.as_manifest()]
        if pool_summary:
            # The completed run's concurrency receipt: files, bytes,
            # workers, host caps, wall, and the effective speedup
            # against the serial model.  Absent from interrupted
            # manifests, which measured no complete run.
            payload["concurrency"] = dict(pool_summary)
        # The ladder is request state, not a constant, so the receipt
        # carries it: the front-door manifest passes it to the bridge,
        # and the vertical contract needs the source top to decide
        # whether the case's p_top is reachable at all.
        payload["pressure_levels_hpa"] = [
            float(format(float(level), "g")) for level in levels]
        payload["source_top_pressure_pa"] = source_top_pa
        # Everything the product published, which a re-run of this
        # finished folder resolves its ladder against without reading
        # the index again.
        payload["published_pressure_levels_hpa"] = [
            float(format(float(level), "g")) for level in available]
        return write_fetch_manifest(out, payload)

    def resume_command() -> str:
        cadence = hours[1] - hours[0] if len(hours) > 1 else 3
        area_arg = ",".join(format(value, "g") for value in (
            area.lat_south, area.lon_west,
            area.lat_north, area.lon_east))
        command = [
            "woof", "fetch", "--source", source,
            "--cycle", cycle.strftime("%Y-%m-%dT%H"),
            # --hours is the window LENGTH, so a window that begins at a
            # forecast lead resumes to the same set it was cut from.
            "--hours", str(hours[-1] - hours[0]),
            "--cadence", str(cadence),
            "--area", area_arg, "--out", str(out),
        ]
        if hours[0]:
            command.extend(("--forecast-start-hour", str(hours[0])))
        # The ladder too: the files already on disk were cut to it, and a
        # resume on the certified ladder refuses every one of them.
        command.extend(_ladder_flags(top_pressure_pa, all_levels))
        if accept_inventory_change:
            command.append("--accept-inventory-change")
        return shlex.join(command)

    planned = [
        (hour,
         f"{prefix}.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}.subset.grib2",
         transport.nomads_query(cycle, hour, model=source,
                                pressure_levels_hpa=levels, **box))
        for hour in hours]

    def transfer(hour: int, name: str, url: str) -> dict:
        path = out / name
        # The stopwatch starts on the WHOLE hour, not on the
        # download alone: a verify-skip re-hashes the file on disk
        # and that is real wall clock a reader of the manifest is
        # entitled to see.  The terminal printed a per-file
        # "N B in X.X s" and threw it away; the manifest recorded
        # bytes and sha256 and no seconds at all, so bandwidth --
        # the number that says whether the network or the service
        # was the limiter -- existed nowhere on disk.
        file_started = time.perf_counter()
        downloaded = not path.exists()
        # No per-hour force quarantine here: the whole directory was
        # swept before the transfers, receipts first.  Moving a payload
        # aside mid-run is what let an old manifest outlive the
        # bytes it claimed.
        if path.exists():
            observed = count_grib2_messages(path)
            if observed != bar.expected:
                raise ValueError(_existing_crop_refusal(
                    name, observed, bar.expected, levels, source))
            digest = existing_file_digest(path)
            recorded = prior_digests.get(name)
            if recorded is not None and digest != recorded:
                raise ValueError(resume_digest_refusal(name))
            progress(f"fetch {source} f{hour:03d}: {name} exists, "
                     f"{path.stat().st_size:,} B verified -- skipped")
        else:
            started = time.perf_counter()
            try:
                transport._download(url, path)
            except HTTPError as error:
                raise RuntimeError(nomads_reach_refusal(
                    source, cycle, hour, error)) from None
            observed = count_grib2_messages(path)
            if observed != bar.expected:
                raise ValueError(
                    f"NOMADS returned {observed} GRIB2 messages for "
                    f"f{hour:03d}, expected {bar.expected}; the "
                    "upstream inventory has drifted")
            digest = sha256_file(path)
            progress(f"fetch {source} f{hour:03d}: {name} "
                     f"{path.stat().st_size:,} B in "
                     f"{time.perf_counter() - started:.1f} s")
        return {
            "name": name, "role": f"{source}-subset",
            "forecast_hour": hour, "bytes": path.stat().st_size,
            "seconds": round(time.perf_counter() - file_started, 6),
            # Said, not inferred from the seconds: a reader of this
            # manifest must be able to tell a download from a
            # verify-skip, because dividing bytes by seconds means
            # bandwidth for one and sha256 throughput for the other.
            "downloaded": downloaded,
            "sha256": digest, "url": url,
        }

    def admit(index: int, entry: dict) -> Path:
        # On the caller's thread, in hour order, as the verified prefix
        # grows -- the manifest keeps its exact serial publication
        # semantics under any pool size.
        files.append(entry)
        return publish_manifest()

    refuse_changed_on_disk(out, [name for _hour, name, _url in planned],
                           prior_digests)
    monitor = progress_mod.TransferMonitor(f"fetch {source}")
    try:
        _entries, receipt = fetch_pool.run_transfers(
            [fetch_pool.TransferJob(
                name=name, token=f"f{hour:03d}", path=out / name,
                # A file already here is checked and never fetched
                # again (a failed check refuses), so it asks no host
                # and is held under no host's cap.
                url=None if (out / name).exists() else url,
                on_disk=(out / name).exists(),
                action=functools.partial(transfer, hour, name, url))
             for hour, name, url in planned],
            workers=file_workers, on_admitted=admit, monitor=monitor)
        pool_summary.update(receipt)
        publish_manifest()
    except KeyboardInterrupt:
        # The downloader atomically promotes .part only after checking
        # the GRIB envelope, and pooled hours may have completed beyond
        # the admitted prefix before SIGINT landed.  Walk the remaining
        # hours in order and extend the prefix with every file that
        # passes the full envelope/count/digest bars, stopping at the
        # first that does not.
        for hour, name, url in planned[len(files):]:
            path = out / name
            if not path.is_file():
                break
            try:
                observed = count_grib2_messages(path)
                digest = sha256_file(path)
            except (OSError, ValueError):
                break
            recorded = prior_digests.get(name)
            if observed != bar.expected or (
                    recorded is not None and digest != recorded):
                break
            files.append({
                "name": name, "role": f"{source}-subset",
                "forecast_hour": hour,
                "bytes": path.stat().st_size,
                "sha256": digest, "url": url,
            })
        publish_manifest()
        verified = (
            ", ".join(
                f"f{item['forecast_hour']:03d} {item['name']} "
                f"({item['bytes']:,} B, sha256 {item['sha256']})"
                for item in files)
            if files else "none")
        verified_names = {item["name"] for item in files}
        unverified_paths = sorted(
            path for path in out.iterdir()
            if path.is_file()
            and (path.name.endswith(".part")
                 or (path.name.endswith(".grib2")
                     and path.name not in verified_names)))
        unverified = (
            ", ".join(f"{path.name} ({path.stat().st_size:,} B)"
                      for path in unverified_paths)
            if unverified_paths else "none")
        raise RuntimeError(
            f"interrupted. Verified complete GRIB files on disk and "
            f"recorded in {out / FETCH_MANIFEST_NAME}: {verified}. "
            f"Unverified partial/incomplete GRIB files on disk (not "
            f"recorded): {unverified}.\n"
            f"  resume exactly with: {resume_command()}") from None
    finally:
        monitor.close()
    return out / FETCH_MANIFEST_NAME


def _gfs_index_record_count(index_url: str, *, progress, label: str
                            ) -> int | None:
    """Message count the live ``.idx`` declares for one whole object.

    The full-file transfer's independent census: every index line names
    one GRIB2 message, so the downloaded object must walk to exactly
    this many envelopes.  ``None`` when the index cannot be read -- the
    envelope walk and the fail-closed bridge remain the completeness
    gates, the same doctrine the HRRR full-file route applies when an
    index cannot vouch for an object.
    """

    return _gfs_index_census(index_url, progress=progress, label=label)[0]


def _gfs_index_census(index_url: str, *, progress, label: str,
                      selection: dict | None = None
                      ) -> tuple[int | None, int | None]:
    """``(messages in the whole object, messages the selection keeps)``.

    Read from one copy of the live ``.idx``, so the two counts describe
    the same object.  The second is ``None`` without a ``selection``;
    both are ``None`` when the index cannot be read.
    """

    request = Request(index_url, headers={"User-Agent": _USER_AGENT})
    try:
        with paced_urlopen(request, timeout=120) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, OSError, ValueError) as error:
        progress(f"fetch {label}: could not read {index_url} ({error}); "
                 "the GRIB2 envelope walk and the fail-closed bridge "
                 "remain the completeness gates")
        return None, None
    count = sum(1 for line in text.splitlines() if line.strip())
    if not count:
        return None, None
    if selection is None:
        return count, None
    from woof.record_index import count_ncep

    try:
        return count, count_ncep(text.encode("utf-8"), selection["select"])
    except ValueError:
        return count, None


def _rw_fetch_gfs_fullfile(*, binary: Path, cycle: datetime, hour: int,
                           source: str, out: Path,
                           cache_dir: Path | None, progress,
                           transport: str = "s3",
                           streams: int | None = None,
                           byte_relay=None) -> dict:
    """One whole ``pgrb2.0p25`` object through the Rust backbone.

    No selectors: ``--mode full-file`` takes the object in parallel
    range GETs, and the backbone names it after the URL, which is
    already the name the series records.  ``transport`` is the ladder
    rung being asked, in this front door's vocabulary; the backbone has
    its own name for the same host.  ``streams`` and ``byte_relay`` are
    as for :func:`_rw_fetch_hrrr`.
    """

    from woof import rustwx_fetch

    endpoint = fetch_endpoints.endpoint_named(source, transport)
    record = rustwx_fetch.run_fetch(
        binary, model=source, date=f"{cycle:%Y%m%d}", cycle=cycle.hour,
        hours=(hour,), product=endpoint.product,
        source=endpoint.rust_source,
        mode="full-file", out=out, cache_dir=cache_dir, streams=streams,
        on_progress=None if byte_relay is None else byte_relay())
    if len(record["files"]) != 1:
        raise RuntimeError(
            f"rw_fetch returned {len(record['files'])} files for one "
            "forecast hour")
    entry = record["files"][0]
    canonical = f"{GFS_CONTAINER_PREFIX[source]}.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}"
    if entry["name"] != canonical:
        _rename_stream_payload(out / entry["name"], out / canonical, entry)
        entry = {**entry, "name": canonical, "path": str(out / canonical),
                 "publication": _publication(_stat_identity(out / canonical))}
    else:
        _remember_stream_facts(out / entry["name"], entry)
    progress(f"fetch {source} f{hour:03d}: {entry['name']} "
             f"{entry['bytes']:,} B in {entry['wall_seconds']:.1f} s "
             f"({entry['source']}, {entry['mode']} -- "
             f"{entry['mode_reason']})")
    return entry


#: What a selected-record file's receipt entry carries beside the common
#: fields: the index it was selected by and the object it was cut from.
_RECORD_FILE_KEYS = ("selected_records", "index_url", "index_sha256",
                     "index_format", "object_bytes", "object_etag",
                     "requests")


def fetch_gfs_fullfile(*, cycle: datetime, hours: tuple[int, ...],
                       area: Area | None, out: Path, progress=print,
                       force: bool = False, source: str = "gfs",
                       engine: str = "python", engine_bin: Path | None = None,
                       engine_selection: str | None = None,
                       cache_dir: Path | None = None,
                       top_pressure_pa: float | None = None,
                       all_levels: bool = False,
                       file_workers: int | None = None,
                       transport: str | None = None,
                       available_levels=gfs_available_levels,
                       records: bool = False) -> Path:
    """Download ``pgrb2.0p25`` objects along the endpoint ladder.

    The full-file transport for the GFS container sources, holding the
    same output-root lock discipline as :func:`fetch_gfs`.  The raw
    objects are whole-globe north-to-south DRT 5.3 grids; both forms
    are certified in ``gfs_grib2_bridge`` (see
    ``tests/fixtures/gfs-scan-order/README.md``), so no area crop and
    no NOMADS CGI round trip are involved -- ``area``, when given, is
    recorded as request identity only.  The decode ladder is still a
    request property: the manifest records it and the front door
    declares it to the bridge, so a whole-globe object never drags the
    mesosphere into a tornado-scale decode.

    ``records`` (``--mode idx-subset``, :data:`GFS_RECORDS_MODE`) takes
    only the records the grib-filter crop asks for from each object, by
    the byte ranges its ``.idx`` lists (:func:`gfs_record_selection`),
    instead of the whole object.  The records are copied unchanged, so
    the decode reads the same bytes either way.  An object whose host
    keeps no index is moved whole, and the log says so.
    """

    transport = fetch_endpoints.policy_transport(source, transport)
    with _fetch_out(out, progress=progress):
        return _fetch_gfs_fullfile_locked(
            cycle=cycle, hours=hours, area=area,
            out=deep_io_path(out, DOWNLOAD_DEPTH_BUDGET), progress=progress,
            force=force, source=source, engine=engine, engine_bin=engine_bin,
            engine_selection=engine_selection,
            cache_dir=cache_dir, top_pressure_pa=top_pressure_pa,
            all_levels=all_levels, file_workers=file_workers,
            # ``auto`` is the unpinned default written out, here as on
            # every check that asks a host before the transfer.
            pinned_host=pinned_host(transport),
            available_levels=available_levels, records=records)


def _fetch_gfs_fullfile_locked(*, cycle: datetime, hours: tuple[int, ...],
                               area: Area | None, out: Path, progress,
                               force: bool, source: str, engine: str,
                               engine_bin: Path | None,
                               engine_selection: str | None,
                               cache_dir: Path | None,
                               top_pressure_pa: float | None,
                               all_levels: bool,
                               file_workers: int | None = None,
                               pinned_host: str | None = None,
                               available_levels=gfs_available_levels,
                               records: bool = False
                               ) -> Path:
    """The whole-object (or selected-record) transfer, lock held.

    Per object, three independent bars before the manifest admits it:
    the GRIB2 envelope walk (every message declares edition 2 and its
    exact length, and the messages tile the file), the live ``.idx``
    message census when the index can be read, and -- on resume -- the
    sha256 the prior manifest recorded.  The series and manifest are
    refreshed after every verified hour, so an interrupted fetch
    records its complete prefix and the same command resumes it.
    """

    from tools import download_gfs_native_subset as transport

    if source not in GFS_CONTAINER_SOURCES:
        raise ValueError(f"fetch_gfs_fullfile serves {GFS_CONTAINER_SOURCES}, "
                         f"not {source!r}")
    if source == "gdas":
        beyond = [hour for hour in hours if hour > GDAS_MAX_FORECAST_HOUR]
        if beyond:
            raise ValueError(gdas_capability_refusal(beyond[0]))
    if top_pressure_pa is not None and all_levels:
        raise ValueError(
            "--p-top-pa names the model top the ladder must reach and "
            "--all-levels takes every level the product carries; they "
            "are two answers to the same question, so pass one")
    prefix = GFS_CONTAINER_PREFIX[source]
    # The request's byte transport: its identity in the receipt, so a
    # folder of whole objects and a folder of selected records are never
    # resumed onto each other.
    mode = GFS_RECORDS_MODE if records else "full-file"
    if not force:
        # Inside the output lock, before any provider is asked: a file
        # name carries the cycle hour but not the date, so another day's
        # files would otherwise pass every per-file bar here.
        require_matching_request(out, source=source, cycle=cycle, area=area,
                                 mode=mode)
    out.mkdir(parents=True, exist_ok=True)
    # The ladder is a DECODE declaration here, not a transfer selection:
    # the whole object carries every published level either way.  It is
    # resolved exactly as the subset route resolves it, recorded in the
    # manifest, and handed to the bridge by the front door.  A folder
    # that already holds this whole request resolves it against the
    # published ladder its receipt recorded, and reads no index.
    receipt = _complete_request_receipt(
        out, force=force, source=source, cycle=cycle, area=area,
        hours=hours, mode=mode)
    available = _recorded_published_levels(receipt)
    # True while the ladder is the certified stand-in (no live index): the
    # first downloaded object then names what it really carries.
    ladder_from_object = False
    if available is None:
        index_text = gfs_live_index(cycle, progress=progress,
                                    source=source)
        available = available_levels(cycle, progress=progress,
                                     source=source, index_text=index_text)
        ladder_from_object = index_text is None
    if all_levels:
        levels = tuple(float(level) for level in available)
        progress(f"fetch {source}: --all-levels declares the whole "
                 f"published ladder for the decode, {len(levels)} isobaric "
                 f"levels ({min(levels):g}..{max(levels):g} hPa)")
    else:
        levels = transport.levels_for_top(top_pressure_pa,
                                          available=available)
        if top_pressure_pa is not None:
            extra = len(levels) - len(transport.PRESSURE_LEVELS_HPA)
            progress(
                f"fetch {source}: model top {float(top_pressure_pa):g} Pa "
                f"needs {len(levels)} isobaric levels, source top "
                f"{min(levels) * 100.0:g} Pa"
                + (f" ({extra} level(s) above the certified 100 hPa "
                   "ladder)" if extra else " (the certified ladder "
                   "already reaches it)"))
    source_top_pa = min(levels) * 100.0
    # The records this request takes, selected against the decode ladder
    # just resolved.  Without a live index the ladder is the certified
    # stand-in until the first object names what it carries, and a
    # selection made against a stand-in could leave out a level the
    # object really publishes, so those objects move whole.
    selection = None
    if records and ladder_from_object:
        progress(f"fetch {source}: no live index to select records by; "
                 "moving the whole pgrb2.0p25 objects instead")
    elif records:
        # The grib-filter's own ladder, which is not always the decode
        # ladder declared below: GDAS takes every published level (its
        # mapped preparation reads air temperature up to 1 hPa), so a
        # selection cut to the declared ladder was refused at prepare
        # ("air_temperature vertical coverage mismatch",
        # 2026-10-29).  The crop route and this one take the same records.
        selection = gfs_record_selection(source, container_subset_levels(
            source, top_pressure_pa=top_pressure_pa, all_levels=all_levels,
            available=tuple(float(level) for level in available)))
        if selection is None:
            progress(f"fetch {source}: its packaged mapping binds a record "
                     "no index name here covers; moving the whole "
                     "pgrb2.0p25 objects instead")
    records_role = f"{source}-records"
    whole_role = f"{source}-full-file"
    #: Which role each file moved as, for the interrupt path's receipt.
    file_roles: dict[str, str] = {}
    if force:
        _force_quarantine_output(out, progress, source)
    prior_digests = _prior_manifest_digests(out)
    prior_entries = _prior_manifest_entries(out)
    files: list[dict] = []
    pool_summary: dict = {}
    file_ladder_receipt = None

    def publish_manifest() -> Path:
        nonlocal available, levels, source_top_pa, ladder_from_object
        # Keep a completed shared series stable while a second member checks
        # cached files. Only a fresh/interrupted fetch needs prefix receipts.
        if receipt is not None and len(files) < len(hours):
            return out / FETCH_MANIFEST_NAME
        if ladder_from_object and files:
            first = min(files, key=lambda item: int(item["forecast_hour"]))
            carried = gfs_object_levels(out / first["name"], source=source)
            if carried:
                ladder_from_object = False
                if tuple(carried) != tuple(available):
                    progress(f"fetch {source}: no live index; {first['name']} "
                             f"carries {len(carried)} isobaric levels "
                             f"({min(carried):g}..{max(carried):g} hPa), and the "
                             "decode ladder is resolved against them")
                available = carried
                levels = (tuple(float(level) for level in available) if all_levels
                          else transport.levels_for_top(top_pressure_pa,
                                                        available=available))
                source_top_pa = min(levels) * 100.0
        # Relative names: gfs_grib2_bridge resolves them against the
        # TSV's own directory, so the fetched directory stays relocatable.
        series = out / f"{prefix}-series.tsv"
        _atomic_write_text(series, "".join(
            f"{item['forecast_hour']}\t{item['name']}\t"
            f"{81 if item['forecast_hour'] == 0 else 96}\n"
            for item in files))
        entries = files + [{
            "name": series.name, "role": "series", "forecast_hour": None,
            "bytes": series.stat().st_size, "sha256": sha256_file(series),
            "url": None,
        }]
        entries += _write_gfs_front_door_files(
            out, source=source, cycle=cycle, files=files, series=series)
        payload = _manifest_payload(
            source=source, cycle=cycle,
            hours=tuple(int(item["forecast_hour"]) for item in files),
            area=area, files=entries)
        payload["notes"] = (
            ("The records the grib-filter selection names, copied unchanged "
             "from the archive GRIB2 objects by the byte ranges of their "
             ".idx (an object whose host keeps no index is whole); no area "
             "crop. "
             if records else "Whole archive GRIB2 objects; no area crop. ")
            + "The endpoint layout "
            "declares the product resolution. Rust verifies the GRIB envelope "
            "and the ingest bridge validates the grid and required fields.")
        if selection is not None:
            payload["record_selection"] = dict(selection)
        payload["archive_products"] = {
            name: {"product": fetch_endpoints.endpoint_named(source, name).product,
                   "layouts": list(fetch_endpoints.endpoint_named(source, name).layouts)}
            for name in sorted({str(item["endpoint"]) for item in files
                                if item.get("endpoint")})}
        payload["engine"] = engine
        payload["engine_selection"] = _engine_selection(
            engine, engine_selection)
        payload["mode"] = mode
        # WHERE THE BYTES CAME FROM, per file and in summary.  This was
        # the constant "s3" and is now what actually served: a receipt
        # that names one host while the ladder fell through to another
        # is a claim about intent, not provenance.
        payload["endpoints"] = {
            "considered": [endpoint.name for endpoint in ladder],
            # The throughput order the table declares, beside the
            # retention order above: they answer different questions,
            # and a receipt that showed only one could not explain why
            # a fresh cycle came off the archive.
            "transfer_preference": [
                endpoint.name
                for endpoint in fetch_endpoints.transfer_order(ladder)],
            "served": sorted({
                str(item["endpoint"]) for item in files
                if item.get("endpoint")}),
            "ladder": [
                {"name": endpoint.name, "base": endpoint.base,
                 "host": endpoint.host,
                 "retention_hours": endpoint.retention_hours,
                 "transfer_rank": endpoint.transfer_rank,
                 "why": endpoint.why}
                for endpoint in ladder],
        }
        payload["transport"] = (payload["endpoints"]["served"] or
                                [ladder[0].name])[0]
        # The decode ladder this request declares; the front-door
        # manifest passes it to the bridge, and the vertical contract
        # needs the source top to decide whether the case's p_top is
        # reachable at all.
        payload["pressure_levels_hpa"] = [
            float(format(float(level), "g")) for level in levels]
        payload["source_top_pressure_pa"] = source_top_pa
        payload["published_pressure_levels_hpa"] = [
            float(format(float(level), "g")) for level in available]
        if file_ladder_receipt is not None:
            payload["pressure_ladder_basis"] = file_ladder_receipt
        if pool_summary:
            # The completed run's concurrency receipt; absent from
            # interrupted manifests, which measured no complete run.
            payload["concurrency"] = dict(pool_summary)
        return write_fetch_manifest(out, payload)

    def resume_command() -> str:
        cadence = hours[1] - hours[0] if len(hours) > 1 else 3
        command = [
            "woof", "fetch", "--source", source,
            "--cycle", cycle.strftime("%Y-%m-%dT%H"),
            "--hours", str(hours[-1] - hours[0]),
            "--cadence", str(cadence),
            "--mode", mode, "--out", str(out),
        ]
        if area is not None:
            command.extend(("--area", ",".join(
                format(value, "g") for value in (
                    area.lat_south, area.lon_west,
                    area.lat_north, area.lon_east))))
        if hours[0]:
            command.extend(("--forecast-start-hour", str(hours[0])))
        # The decode ladder the manifest records; dropping it on resume
        # would record the certified ladder for the same objects.
        command.extend(_ladder_flags(top_pressure_pa, all_levels))
        return shlex.join(command)

    # The endpoints this CYCLE will be asked for, in order.  Resolved
    # once for the request, from the cycle's age: an initialization
    # inside the operational window is taken from the server that
    # published it first, and one older than that window goes straight
    # to the archive without paying for an attempt that was certain to
    # 404.  Both hosts answer the same relative key with the same
    # Content-Length, so falling through is appending one key to
    # another base.
    ladder = fetch_endpoints.serving_ladder(source, cycle=cycle,
                                            pinned=pinned_host)
    planned = [
        (hour, f"{prefix}.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}",
         gfs_object_key(cycle, hour, source))
        for hour in hours]
    # WHICH rung moves each object, decided before any of them move.
    # Retention says who is ASKED; throughput says who should serve,
    # and the archive only earns that when it provably has the object
    # -- one HEAD per hour, run ahead of the transfers through the same
    # pool.  See woof.fetch_endpoints.transfer_ladder.  An hour already
    # on disk is verified here rather than moved, so no host is asked
    # about it: a finished folder re-runs without a provider.
    object_ladders = _probe_object_ladders(
        ladder, keys=[key for _hour, name, key in planned
                      if not (out / name).exists()],
        source=source, pinned=pinned_host, workers=file_workers,
        progress=progress)
    # The fetch's chunk-stream budget, split over the files in flight,
    # and the monitor the transfers report in-flight bytes to once the
    # pool starts.
    streams = fetch_pool.chunk_streams_per_file(
        file_workers, files=len(planned))
    monitor = None

    def transfer(hour: int, name: str, key: str) -> dict:
        path = out / name
        rungs = object_ladders.get(key, ladder)
        url = rungs[0].url(key)
        endpoint_name = rungs[0].name
        # The whole hour, index probe included: see the subset
        # route's note.  A manifest that recorded only the download's
        # seconds would under-report the wall a caller waited.
        file_started = time.perf_counter()
        downloaded = not path.exists()
        role = whole_role
        # What a selected-record file's receipt carries beside the
        # common fields: its index, the object it was cut from.
        extra: dict = {}
        if path.exists():
            digest = existing_file_digest(path)
            recorded = prior_digests.get(name)
            if recorded is not None and digest != recorded:
                raise ValueError(resume_digest_refusal(name))
            prior = prior_entries.get(name) if recorded is not None else None
            if prior is not None:
                # The bytes the receipt vouched for: their census and
                # the host that served them are the recorded ones, so
                # no host is asked, and a re-run's receipt keeps naming
                # where the bytes actually came from.
                idx_records = prior.get("idx_records")
                if isinstance(prior.get("url"), str):
                    url = prior["url"]
                endpoint_name = prior.get("endpoint")
                census = "its receipt recorded"
                if prior.get("role") == records_role:
                    role = records_role
                    extra = {field: prior[field] for field in _RECORD_FILE_KEYS
                             if field in prior}
                observed = count_grib2_messages(path)
            else:
                whole, kept = ((_gfs_index_census(
                    url + ".idx", progress=progress,
                    label=f"{source} f{hour:03d}", selection=selection))
                    if rungs[0].idx_available else (None, None))
                # Nothing records which host served a file the receipt
                # never admitted, and naming the ladder head would be a
                # claim.
                endpoint_name = None
                census = "the live index lists"
                observed = count_grib2_messages(path)
                # A file left by an interrupted selected-record fetch
                # carries the selection's census; one this request moved
                # whole (its host kept no index) carries the object's.
                if kept is not None and observed == kept:
                    role, idx_records = records_role, kept
                else:
                    idx_records = whole
            if idx_records is not None and observed != idx_records:
                raise ValueError(
                    f"existing {name} carries {observed} GRIB2 "
                    f"messages where {census} {idx_records}; move it "
                    "aside and re-fetch")
            progress(f"fetch {source} f{hour:03d}: {name} exists, "
                     f"{path.stat().st_size:,} B verified -- skipped")
        else:
            # A selected-record transfer reads its index itself and holds
            # the file to it; a whole object is held to the index's
            # message count, read here.
            idx_records = (_gfs_index_record_count(
                url + ".idx", progress=progress,
                label=f"{source} f{hour:03d}")
                if rungs[0].idx_available and selection is None else None)
            started = time.perf_counter()
            taken: dict = {}

            def move_whole(endpoint: fetch_endpoints.Endpoint) -> None:
                if engine == "rust":
                    entry = _rw_fetch_gfs_fullfile(
                        binary=engine_bin, cycle=cycle, hour=hour,
                        source=source, out=out, cache_dir=cache_dir,
                        transport=endpoint.name, progress=progress,
                        streams=streams,
                        byte_relay=(None if monitor is None else
                                    functools.partial(monitor.relay,
                                                      name)))
                    landed = out / entry["name"]
                    if landed != path:
                        raise RuntimeError(
                            f"rw_fetch landed {entry['name']}, expected "
                            f"{name}")
                else:
                    transport._download(endpoint.url(key), path)

            def move(endpoint: fetch_endpoints.Endpoint) -> None:
                taken.clear()
                if selection is not None:
                    from woof.record_index import IndexAbsent, download_records

                    if endpoint.idx_available:
                        try:
                            taken.update(download_records(
                                endpoint.url(key), endpoint.url(key) + ".idx",
                                path, declaration=selection,
                                streams=max(1, int(streams or 1))))
                            return
                        except IndexAbsent as error:
                            progress(f"fetch {source} f{hour:03d}: "
                                     f"{endpoint.name} keeps no index beside "
                                     f"{name} ({error}); moving the whole "
                                     "object instead")
                    else:
                        progress(f"fetch {source} f{hour:03d}: "
                                 f"{endpoint.name} publishes no index for "
                                 f"{name}; moving the whole object instead")
                move_whole(endpoint)

            try:
                # The tree's one shared retry: a transfer the network cut
                # off is asked again 2, 4, 8 and 16 s later.  The Python
                # transport retries inside itself on the same shared
                # classification and attempt budget, so it gets one
                # round here rather than five rounds of five.  A
                # selected-record transfer does not retry inside itself.
                chosen, _ = fetch_endpoints.ask_along_ladder(
                    rungs, move, label=f"fetch {source} f{hour:03d}",
                    name=name, progress=progress,
                    discard=lambda _endpoint: path.with_name(
                        path.name + ".part").unlink(missing_ok=True),
                    attempts=(fetch_endpoints.TRANSIENT_ATTEMPTS
                              if engine == "rust" or selection is not None
                              else 1))
            except fetch_endpoints.TransferRefusal as error:
                raise RuntimeError(str(error)) from None
            endpoint_name = chosen.name
            url = chosen.url(key)
            if taken:
                # download_records refused anything but whole messages,
                # one per selected record, from one object version.
                role = records_role
                idx_records = int(taken["selected_records"])
                observed = count_grib2_messages(path)
                digest = str(taken["sha256"])
                extra = {field: taken[field] for field in _RECORD_FILE_KEYS
                         if field in taken}
                progress(f"fetch {source} f{hour:03d}: {name} "
                         f"{idx_records} records, {path.stat().st_size:,} of "
                         f"{int(taken.get('object_bytes') or 0):,} B in "
                         f"{time.perf_counter() - started:.1f} s "
                         f"({endpoint_name}, {GFS_RECORDS_MODE})")
            else:
                observed = _fetched_records(path)
                if idx_records is not None and observed != idx_records:
                    _quarantine_rejected(path, progress,
                                         f"{source} full-file")
                    raise ValueError(
                        f"downloaded {name} carries {observed} GRIB2 "
                        f"messages where its live .idx lists "
                        f"{idx_records}; the file has been moved aside, "
                        "nothing was deleted")
                digest = _fetched_digest(path)
                if engine != "rust":
                    progress(f"fetch {source} f{hour:03d}: {name} "
                             f"{path.stat().st_size:,} B in "
                             f"{time.perf_counter() - started:.1f} s "
                             f"({endpoint_name}, full-file)")
        file_roles[name] = role
        return {
            "name": name, "role": role,
            "forecast_hour": hour, "bytes": path.stat().st_size,
            "seconds": round(time.perf_counter() - file_started, 6),
            "downloaded": downloaded,
            "sha256": digest, "url": url, "endpoint": endpoint_name,
            "grib2_messages": observed, "idx_records": idx_records,
            **extra,
        }

    def admit(index: int, entry: dict) -> Path:
        nonlocal available, levels, source_top_pa, file_ladder_receipt, receipt
        if index == 0 and entry.get("idx_records") is None and (
                top_pressure_pa is not None or all_levels):
            available, file_ladder_receipt = _gfs_file_levels(out / entry["name"], source)
            file_ladder_receipt["file_sha256"] = entry["sha256"]
            levels = (available if all_levels else
                      transport.levels_for_top(top_pressure_pa, available=available))
            source_top_pa = min(levels) * 100.0
            if receipt is not None and tuple(receipt.get(
                    "pressure_levels_hpa", ())) != tuple(levels):
                # A corrected declaration must reach as-posted consumers
                # before they start decoding the first admitted lead.
                receipt = None
            progress(f"fetch {source}: archived file inventory declares {len(levels)} "
                     f"complete pressure levels, source top {source_top_pa:g} Pa")
        files.append(entry)
        return publish_manifest()

    def vouched(name: str) -> bool:
        # On disk AND in the receipt: checked against what the receipt
        # recorded, so no host is asked, not even for its index, and a
        # failed check refuses rather than fetching the file again.  A
        # file on disk the receipt never admitted still has its census
        # read from the host's index.
        return (out / name).exists() and name in prior_entries

    def job_url(name: str, key: str) -> str | None:
        # No host for a vouched file, so it is held under no host's cap.
        if vouched(name):
            return None
        return object_ladders.get(key, ladder)[0].url(key)

    refuse_changed_on_disk(out, [name for _hour, name, _key in planned],
                           prior_digests)
    monitor = progress_mod.TransferMonitor(f"fetch {source}")
    try:
        _entries, receipt = fetch_pool.run_transfers(
            [fetch_pool.TransferJob(
                name=name,
                # The host this object will ACTUALLY be asked first:
                # counting a mirrored transfer against the operational
                # server's cap of 2 would throttle the fetch to the
                # pace of the host it just avoided.
                url=job_url(name, key),
                on_disk=vouched(name),
                token=f"f{hour:03d}", path=out / name,
                action=functools.partial(transfer, hour, name, key))
             for hour, name, key in planned],
            workers=file_workers, on_admitted=admit, monitor=monitor)
        pool_summary.update(receipt)
        publish_manifest()
    except KeyboardInterrupt:
        # Same admission bars as the transfers: envelope walk and the
        # prior manifest's digest.  Pooled hours may have completed
        # beyond the admitted prefix; extend it in order and stop at
        # the first file that is absent or fails a bar.  A partial file
        # never reaches the manifest.
        for hour, name, key in planned[len(files):]:
            path = out / name
            if not path.is_file():
                break
            # The role the transfer recorded; a file whose transfer never
            # said how it moved could be either, so the prefix ends there.
            role = file_roles.get(name, None if selection is not None
                                  else whole_role)
            if role is None:
                break
            try:
                observed = count_grib2_messages(path)
                digest = sha256_file(path)
            except (OSError, ValueError):
                break
            recorded = prior_digests.get(name)
            if recorded is not None and digest != recorded:
                break
            files.append({
                "name": name, "role": role,
                "forecast_hour": hour,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "url": object_ladders.get(key, ladder)[0].url(key),
                # The interrupt path never saw which endpoint served
                # this file, and guessing the head would be a claim.
                "endpoint": None,
                "grib2_messages": observed,
                "idx_records": None,
            })
        publish_manifest()
        verified = (
            ", ".join(
                f"f{item['forecast_hour']:03d} {item['name']} "
                f"({item['bytes']:,} B, sha256 {item['sha256']})"
                for item in files)
            if files else "none")
        verified_names = {item["name"] for item in files}
        unverified_paths = sorted(
            path for path in out.iterdir()
            if path.is_file()
            and (path.name.endswith(".part")
                 or (".pgrb2." in path.name
                     and path.name not in verified_names)))
        unverified = (
            ", ".join(f"{path.name} ({path.stat().st_size:,} B)"
                      for path in unverified_paths)
            if unverified_paths else "none")
        raise RuntimeError(
            f"interrupted. Verified complete GRIB files on disk and "
            f"recorded in {out / FETCH_MANIFEST_NAME}: {verified}. "
            f"Unverified partial/incomplete GRIB files on disk (not "
            f"recorded): {unverified}.\n"
            f"  resume exactly with: {resume_command()}") from None
    finally:
        # Its ticker thread otherwise outlives the fetch and repeats the
        # last "N of N files done" line for the rest of the process,
        # which runs the whole forecast when the fetch is a plan stage.
        monitor.close()
    return out / FETCH_MANIFEST_NAME


def _write_gfs_front_door_files(out: Path, *, source: str, cycle: datetime,
                                files: list[dict], series: Path
                                ) -> list[dict]:
    """The four-file front door, on the GFS container routes too.

    DATA.md promises every fetched directory ``inputs.txt`` +
    ``prep-command.txt`` + ``SHA256SUMS`` + ``fetch-manifest.json``, and
    the table routes keep that promise; the GFS route left three grib2
    files, a tsv and a manifest -- a pile, not a front door (UX finding
    N11).  Written BEFORE each manifest publication and returned as
    manifest rows, so an interrupted fetch's files describe exactly the
    verified prefix the manifest records, and the manifest claims every
    canonical file the fetch put in the directory.

    The order and the binding are the contract, not decoration.  These
    three landed after the manifest and unlisted, which left the audited
    receipt under-claiming its own directory -- three canonical,
    undigested files a reader had to take on trust -- and left a window
    between the manifest rename and theirs where a kill published a
    complete-looking receipt beside the PREVIOUS run's checksum list and
    input list.  Publishing them first and hashing them into ``files``
    makes the manifest the publication barrier the rest of this module
    already treats it as, and matches the HRRR route, which has always
    written ``SHA256SUMS`` first and carried it as a ``checksums`` row.

    ``prep-command.txt`` carries the BOUND half only, like every table
    route's: the namelist, config, geography and output root are the
    reader's.  The digest binding is not a fifth file to author by hand
    -- ``woof prep --source gfs`` authors and binds the input manifest
    itself when ``--source-manifest`` is omitted.
    """

    lines = [f"{item['sha256']}  {item['name']}" for item in files]
    lines.append(f"{sha256_file(series)}  {series.name}")
    sums = out / fetch_routes.SHA256SUMS_NAME
    inputs = out / fetch_routes.INPUT_LIST_NAME
    command_path = out / fetch_routes.PREP_COMMAND_NAME
    _atomic_write_text(sums, "\n".join(lines) + "\n")
    _atomic_write_text(inputs, "".join(
        f"{(out / item['name']).resolve()}\n" for item in files))
    header = [
        f"# {source} 0.25-degree isobaric container fetch",
        f"# cycle {cycle:%Y-%m-%dT%H}Z"
        + (f", forecast hours f{files[0]['forecast_hour']:03d}.."
           f"f{files[-1]['forecast_hour']:03d}" if files else ""),
        "#",
    ]
    handoff = None
    # Fork on the CAPABILITY, not on the container's name.  A caller asks
    # `fetch_routes.publishes_prep_handoff` whether this path publishes
    # bound prep arguments, and that predicate reads the packaged
    # composition; a second test written here as a source name answers
    # differently the moment a composition binds its surface fields in
    # band, and the predicate would then promise a document nothing had
    # written.  A container that HAS a composed profile but binds no
    # in-band role still reaches `container_handoff_binding` below, which
    # is where that refusal belongs.
    if not fetch_routes.prepares_through_packaged_composition(source):
        header.extend((
            "# The prep door authors and digest-binds the input manifest",
            "# itself when --source-manifest is omitted (it binds this",
            "# directory's fetch manifest, the resolved bridge, and the",
            "# namelist/config you pass).",
            "#",
            "# yours to supply: --wps-namelist, --experiment-config,",
            "#                  --geog-root, --output-root",
        ))
        body = [
            "woof prep \\",
            "  --source gfs \\",
            f"  --gfs-series {shlex.quote(str(series.resolve()))} \\",
            f"  --cycle {cycle:%Y-%m-%d_%H:%M:%S}",
        ]
    else:
        role = container_handoff_binding(source)
        tokens = ["--source", source, "--input-list", str(inputs.resolve())]
        for item in files:
            tokens += ["--supplement", f"{role}={(out / item['name']).resolve()}"]
        tokens += ["--author-input-manifest", str(out.resolve() / "inputs.json")]
        handoff = fetch_routes.write_prep_arguments(
            out, source=source, prep_source=source, cycle=cycle, tokens=tokens)
        header += ["# Supply --wps-namelist, --experiment-config,",
                   "# --geog-root and --output-root."]
        body = ["woof prep " + shlex.join(tokens)]
    _atomic_write_text(command_path,
                       "\n".join(header + ([""] if body else []) + body)
                       + "\n")
    published = [("checksums", sums), ("input-list", inputs),
                 ("prep-command", command_path)]
    if handoff is not None:
        published.append(("prep-arguments", handoff))
    return [{"name": path.name, "role": role, "forecast_hour": None,
             "bytes": path.stat().st_size, "sha256": sha256_file(path),
             "url": None}
            for role, path in published]


def preparation_manifest_path(output_root: Path) -> Path:
    """Where one preparation's own GFS front-door manifest is written.

    Beside that preparation's output root and named for it, never inside
    the download it binds.  The manifest binds the preparation's own
    namelist, experiment and bridge, so every preparation from one
    download used to write one shared ``<download>/gfs-input-manifest.json``:
    a second preparation started from that download replaced the first
    one's manifest while the first was still preparing, and the first
    then failed its own digest check (the front door verifies the
    manifest when it starts and again when it publishes).  An output root
    belongs to one preparation (the front door refuses to prepare into an
    existing one), so a name keyed to it is never another preparation's.
    """

    root = Path(output_root)
    return root.parent / f"{root.name}.{GFS_INPUT_MANIFEST_NAME}"


def author_gfs_front_door_manifest(
        *, out: Path, bridge: Path, wps_namelist: Path,
        experiment_config: Path, static_input: Path | None = None,
        static_receipt: Path | None = None,
        manifest_out: Path | None = None,
        source: str = "gfs",
        forecast_start_hour: int | None = None,
        progress=print) -> tuple[Path, str]:
    """Write the exact input manifest the GFS front door verifies.

    Bridges ``woof fetch`` to ``rw-wps --source gfs``: the front door
    (woof/gfs_direct.py ``_verify_input_manifest``) demands a
    ``gpuwm-gfs-direct-input-manifest-v1`` document binding every input
    role -- ``series``/``bridge``/``wps_namelist``/``experiment_config``
    (plus the optional static pair) and one ``grib-fNNN`` per forecast
    hour -- to its basename and sha256, INCLUDING the bridge
    executable's own hash, plus a ``source`` identity block.  This
    function derives the cycle and hour inventory from the fetch
    manifest in ``out``, hashes every role from disk, writes the
    document, and returns ``(path, sha256-of-the-document)`` -- the pair
    ``--source-manifest``/``--source-manifest-sha256`` wants verbatim.
    """

    # Fetches atomically publish a resumable prefix after each lead, under
    # this same lock. A second member can be rechecking the shared download
    # when the first reaches this stage. Wait for its complete publication,
    # and keep the inventory, series and hashes in one locked transaction.
    with _fetch_out(out, progress=progress):
        return _author_gfs_front_door_manifest_locked(
            out=out, bridge=bridge, wps_namelist=wps_namelist,
            experiment_config=experiment_config, static_input=static_input,
            static_receipt=static_receipt, manifest_out=manifest_out,
            source=source, forecast_start_hour=forecast_start_hour,
            progress=progress)


def _author_gfs_front_door_manifest_locked(
        *, out, bridge, wps_namelist, experiment_config, static_input,
        static_receipt, manifest_out, source, forecast_start_hour, progress):
    """Author the front-door manifest with the download lock held."""

    if source not in GFS_CONTAINER_SOURCES:
        raise ValueError(
            f"the GFS front door serves {GFS_CONTAINER_SOURCES}, not "
            f"{source!r}")
    prior = _load_fetch_manifest(out)
    if prior is None or prior.get("source") != source:
        raise ValueError(
            f"{out / FETCH_MANIFEST_NAME} is not a completed "
            f"`woof fetch --source {source}` output; run the fetch first "
            "(the front-door manifest binds the fetched cycle and "
            "forecast-hour inventory)")
    if (static_input is None) != (static_receipt is None):
        raise ValueError(
            "--static-input and --static-receipt must be supplied "
            "together (or neither, when the front door builds statics "
            "from --geog-root)")
    cycle_text = str(prior.get("cycle", ""))
    try:
        cycle = datetime.strptime(cycle_text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError(
            f"fetch manifest in {out} carries an unreadable cycle "
            f"{cycle_text!r}") from error
    hours = prior.get("forecast_hours")
    if (not isinstance(hours, list) or not hours
            or not all(isinstance(hour, int) for hour in hours)):
        raise ValueError(
            f"fetch manifest in {out} lacks a forecast-hour inventory")
    if len(hours) < 2:
        raise ValueError(
            "a forecast manifest needs at least two forcing times: lateral "
            "boundaries are interpolated BETWEEN frames, so one frame leaves "
            "every boundary interval empty and the run has nothing to force "
            "its edges with.  Fetch one more forcing time, or use the "
            "analysis on its own without a manifest.")
    prefix = GFS_CONTAINER_PREFIX[source]
    # Either transport's payload rows: the NOMADS CGI crop and the
    # whole-object S3 route feed the same front door and the same
    # bridge, which selects by exact field identity either way.
    payload_roles = {f"{source}-subset", f"{source}-full-file",
                     f"{source}-records"}
    subset_names = {
        item.get("forecast_hour"): item.get("name")
        for item in prior.get("files", ())
        if isinstance(item, dict)
        and item.get("role") in payload_roles}
    # Author over a TAIL of what was fetched, when asked.  A directory
    # already holding f000..f240 does not have to be re-downloaded for a
    # run that starts at f174: the manifest and its series are cut to
    # f174..f240, so the front door decodes only that window.  The cut is
    # by absolute lead, because that is what the fetch recorded.
    series_name = f"{prefix}-series.tsv"
    if forecast_start_hour is not None:
        if (isinstance(forecast_start_hour, bool)
                or not isinstance(forecast_start_hour, int)
                or forecast_start_hour < 0):
            raise ValueError(
                "--forecast-start-hour must be a nonnegative forecast lead")
        if forecast_start_hour not in hours:
            raise ValueError(
                f"--forecast-start-hour {forecast_start_hour} is not a "
                f"forecast hour {out} carries.  It holds "
                + ", ".join(f"f{hour:03d}" for hour in hours))
        hours = [hour for hour in hours if hour >= forecast_start_hour]
        if len(hours) < 2:
            raise ValueError(
                f"--forecast-start-hour {forecast_start_hour} leaves "
                f"{len(hours)} forecast hour(s) in {out}; a run needs its "
                "initial condition and at least one lateral boundary time")
        series_name = f"{prefix}-series-f{forecast_start_hour:03d}.tsv"
        # A real, hash-bound series over the tail, written beside the
        # fetch's own.  The full series is never edited: both remain
        # readable, and the manifest names exactly one of them.  Written
        # under the same condition that named it: an explicit f000 used
        # to be named here and written only for a nonzero lead, so the
        # manifest bound a file that did not exist.
        _atomic_write_text(out / series_name, "".join(
            f"{hour}\t{subset_names[hour]}\t{81 if hour == 0 else 96}\n"
            for hour in hours
            if isinstance(subset_names.get(hour), str)))
    roles: dict[str, Path] = {
        "series": out / series_name,
        "bridge": Path(bridge),
        "wps_namelist": Path(wps_namelist),
        "experiment_config": Path(experiment_config),
    }
    if static_input is not None:
        roles["static_input"] = Path(static_input)
        roles["static_receipt"] = Path(static_receipt)
    for hour in hours:
        name = subset_names.get(hour)
        if not isinstance(name, str):
            raise ValueError(
                f"fetch manifest in {out} lists forecast hour {hour} "
                f"without a {source}-subset or {source}-full-file entry")
        roles[f"grib-f{hour:03d}"] = out / name
    missing = sorted(
        f"{role}: {path}" for role, path in roles.items()
        if not path.is_file())
    if missing:
        raise ValueError(
            "front-door manifest inputs are missing:\n  "
            + "\n  ".join(missing))
    identity = {
        # The front door verifies schema, roles and digests, not the
        # model string, so the tag is provenance: which container
        # this series came out of, in one place a receipt can read.
        "model": source.upper(),
        "product": "pgrb2.0p25",
        "cycle": cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # The pressure ladder the fetch actually took, and the source top it
    # implies.  The front door validates the case's p_top against the
    # source top BEFORE the bridge runs, so it has to learn it from the
    # receipt rather than from a constant -- a constant is exactly what
    # capped every GFS run at 10000 Pa.  Absent for a directory fetched
    # by an older ArWen, where the certified 100 hPa ladder is the only
    # thing it can have been.
    levels = prior.get("pressure_levels_hpa")
    if isinstance(levels, list) and levels and all(
            isinstance(level, (int, float)) for level in levels):
        identity["pressure_levels_hpa"] = [float(level) for level in levels]
        identity["top_pressure_pa"] = float(min(levels)) * 100.0
    payload = {
        "schema": GFS_FRONT_DOOR_MANIFEST_SCHEMA,
        "source": identity,
        "files": {
            role: {"name": path.name, "sha256": _fetched_digest(path)}
            for role, path in roles.items()
        },
    }
    path = (Path(manifest_out) if manifest_out is not None
            else out / GFS_INPUT_MANIFEST_NAME)
    # A preparation's own manifest sits beside an output root that does
    # not exist yet, and its parent may not either.
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(path, text)
    # The digest of the bytes this call wrote, not a re-read of the path:
    # a re-read hashes whatever another writer put there in between, and
    # the pair returned would then bind that writer's roles.
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    # Pasteable exactly as printed: no placeholder a user has to fill
    # in, because every value is already known here.  The geography root
    # is the one woof reads everywhere else (and `woof fetch-geog`
    # stages into by default); the output root is a sibling of the
    # download, so the line runs from any directory and does not write
    # into the inputs.
    from woof.geog_assets import default_geog_root

    def printed(value) -> str:
        """One printed argument, quoted if a shell would split it.

        POSIX display form, as `source_cli._quote_command` renders argv:
        the certified runtime is Linux/CUDA, and forward slashes are
        accepted by every path API on Windows too.  A valid `--out` or
        config path containing a space used to be split the moment this
        command -- whose entire value is that it can be pasted -- was.
        """

        return shlex.quote(str(value).replace("\\", "/"))

    static_args = (
        f" --static-input {printed(static_input)}"
        f" --static-receipt {printed(static_receipt)}"
        if static_input is not None
        else f" --geog-root {printed(default_geog_root())}")

    # --physics-profile, when this config HAS one the front door names.
    #
    # It is spelled "optional" in rw-wps' own help and is not: absent,
    # `woof.source_cli` substitutes WSM6_PROFILE_ID and then compares
    # the experiment's physics against that, so a pasted command with no
    # profile refuses every config except a wsm6-no-radiation one.  The
    # command this function prints is the one FIRST-LIGHT.md section 3a
    # tells a reader to paste, and its own worked example uses the
    # Morrison profile -- so the documented chain failed at this step
    # for exactly the case it documents.  Found by running it.
    #
    # Derived through the same authority the front door asks
    # (`identify_single_domain_profile`), so this cannot name a profile
    # the runner would then reject.  A config matching no shipped
    # profile adds nothing: the front door prints its own explanation of
    # that case after preparation, and inventing a flag here would only
    # move the refusal earlier without making it truer.
    profile_arg = ""
    corridor_arg = ""
    try:
        from woof.experiment import load_experiment
        from woof.physics_compat import identify_single_domain_profile
        from woof.static.corridor import config_declares_follow_source

        experiment = load_experiment(Path(experiment_config))
        matched = identify_single_domain_profile(experiment.root.run)
        if matched is not None:
            profile_arg = f" --physics-profile {matched}"
        # A config that declares a [relocation] follow source needs the
        # sealed statics corridor prepared, or the tree runner refuses
        # the very bundle this command builds.  The predicate itself
        # lives in the corridor module, so the pasted line, `woof go`'s
        # driven line and run-plan's refusal cannot drift apart on it.
        if config_declares_follow_source(experiment):
            corridor_arg = " --statics-corridor"
    except Exception:
        # A config this process cannot load is not a reason to withhold
        # the rest of a correct command.
        profile_arg = ""
        corridor_arg = ""
    output_root = out.resolve() / "prepared"
    progress(f"fetch {source}: front-door manifest {path}")
    progress(f"fetch {source}: front-door manifest sha256 {digest}")
    if hours[0]:
        lead_start = cycle + timedelta(hours=hours[0])
        progress(
            f"fetch {source}: this manifest binds forecast hours "
            f"f{hours[0]:03d}..f{hours[-1]:03d}.  An experiment whose "
            f"start_time is {lead_start:%Y-%m-%d %H:%M:%S} is initialized "
            f"from f{hours[0]:03d} -- a {hours[0]} h forecast, not an "
            "analysis -- with its lateral boundaries from the hours "
            "after it")
    if source != "gfs":
        # A command that ends in a refusal is worse than no command.
        # This container has exactly one certified ingest route and it
        # is named for the product it was certified on; printing it here
        # under another source's series would be a source-identity lie,
        # and printing `--source {source}` would be a dead end.  See
        # GDAS_MAX_FORECAST_HOUR above and docs/public/DATA.md.
        progress(
            f"fetch {source}: the native mapped preparation route uses "
            "an --input-list and each file's in-band terrain supplement. "
            "Use the complete pressure ladder and specific humidity. "
            f"See `woof prep --show-source {source}` for the contract.")
        return path, digest
    progress(
        "fetch gfs: feed the GFS front door with:\n"
        f"  rw-wps --source gfs --gfs-series {printed(roles['series'])}"
        f" --cycle {cycle:%Y-%m-%d_%H:%M:%S}"
        f" --bridge {printed(bridge)}"
        f" --wps-namelist {printed(wps_namelist)}"
        f" --experiment-config {printed(experiment_config)}"
        f" --source-manifest {printed(path)}"
        f" --source-manifest-sha256 {digest}"
        f"{profile_arg}{corridor_arg}{static_args} "
        f"--output-root {printed(output_root)}")
    return path, digest


# ---------------------------------------------------------------------------
# HRRR
# ---------------------------------------------------------------------------

def _prior_manifest_digests(out: Path) -> dict[str, str]:
    """``name -> sha256`` from an existing fetch manifest, else empty.

    A malformed or foreign manifest yields no digests rather than an
    error: the record-count bar below still applies to every existing
    file, so the digest map only ever *adds* strictness.
    """

    path = out / FETCH_MANIFEST_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {item["name"]: item["sha256"]
                for item in payload.get("files", ())
                if isinstance(item.get("name"), str)
                and isinstance(item.get("sha256"), str)}
    except (ValueError, TypeError, AttributeError):
        return {}


def _prior_manifest_records(out: Path) -> dict[str, int]:
    """``name -> GRIB2 message count`` from an existing fetch manifest.

    The resume bar has to know what a file was *supposed* to contain,
    and since the probe rule can land either a subset or a whole object
    the certified subset count is no longer that answer on its own.  A
    manifest written before this key existed simply yields nothing here
    and the caller falls back to the certified constant, so old fetch
    directories still resume.
    """

    path = out / FETCH_MANIFEST_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {item["name"]: item["records"]
                for item in payload.get("files", ())
                if isinstance(item.get("name"), str)
                and isinstance(item.get("records"), int)}
    except (ValueError, TypeError, AttributeError):
        return {}


def _prior_manifest_bars(out: Path) -> dict[str, object]:
    """``kind -> RecordBar`` reconstructed from an existing manifest.

    A resume that finds every file already present downloads nothing and
    therefore resolves no bars -- and the manifest it republishes was
    being written from an empty map, so ``record_bars`` came out ``[]``
    and an ``inventory_change_accepted: true`` from the original fetch
    vanished.  DATA promises that acceptance stays recorded, and the
    directory's files really were fetched under that bar, so the prior
    bars seed this run's map and any kind actually re-resolved replaces
    its own entry.

    A malformed or foreign manifest yields nothing, exactly as the digest
    and record maps above do: worst case the provenance is no worse than
    it is today.
    """

    from woof.fetch_bars import RecordBar

    path = out / FETCH_MANIFEST_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        bars: dict[str, object] = {}
        for item in payload.get("record_bars", ()):
            kind = item["kind"]
            bars[kind] = RecordBar(
                kind=kind, expected=int(item["expected"]),
                certified=int(item["certified"]),
                derived=None if item.get("derived") is None
                else int(item["derived"]))
        return bars
    except (ValueError, TypeError, AttributeError, KeyError):
        return {}


#: The byte modes an HRRR file can land in; ``auto`` only chooses
#: between them, per file.
HRRR_LANDED_MODES = ("full-file", "idx-subset")


def _hrrr_landed_mode(entry: dict | None, records: int | None,
                      subset_ceiling: int) -> str | None:
    """Is a kept HRRR file a whole object or an ``.idx`` record subset?

    A receipt says so per file (``mode``).  One written before that key
    existed is read by its census: a subset carries at most the
    selection's records, a whole ``wrfnat`` or ``wrfprs`` hundreds more.
    ``None`` when there is nothing to read it from.
    """

    recorded = (entry or {}).get("mode")
    if recorded in HRRR_LANDED_MODES:
        return recorded
    if isinstance(records, bool) or not isinstance(records, int):
        return None
    return "full-file" if records > subset_ceiling else "idx-subset"


def _hrrr_receipt_has_old_subsets(prior: dict, entries) -> bool:
    """Does a receipt claim an atmosphere subset that predates the optional records?

    A subset fetched before it took PBLH/LCDC/MCDC/HCDC
    (``OPTIONAL_SURFACE_FIELDS``) carries no ``optional_surface`` key.
    Next to a whole file or a current subset its hour selects a different
    record set, which hrrr_grib2_bridge refuses, so the fetch sets it
    aside and takes it again.  Whole files and current subsets mix
    freely: they carry the same records for the bridge.
    """

    from tools import download_hrrr_native_subset as range_transport

    ceiling = (range_transport.atmosphere_record_count(
        analyzed_aerosol=True,
        extras=tuple(prior.get("native_extras") or ()))
        + len(range_transport.OPTIONAL_SURFACE_FIELDS))
    return any(
        entry.get("role") == "atmosphere"
        and _hrrr_landed_mode(entry, entry.get("records"), ceiling)
        == "idx-subset"
        and "optional_surface" not in entry
        for entry in entries)


def _existing_hrrr_digest(dest: Path, *, expected_count: int,
                          prior_digest: str | None, progress,
                          label: str) -> str | None:
    """sha256 of a verified existing subset, or None to force re-download.

    An existing file passes the SAME completeness bar as a fresh
    download: the full GRIB2 envelope walk plus the exact expected
    record count (a nonempty prefix truncated at a message boundary
    walks clean but fails the count).  When a prior fetch manifest
    recorded a digest for this name, the file must also still match it;
    a failing file is never digest-blessed -- it is re-downloaded.
    """

    try:
        observed = count_grib2_messages(dest)
    except ValueError as error:
        progress(f"fetch hrrr {label}: existing {dest.name} failed "
                 f"envelope validation ({error}); re-downloading")
        return None
    if observed != expected_count:
        progress(f"fetch hrrr {label}: existing {dest.name} carries "
                 f"{observed} GRIB2 messages, expected {expected_count} "
                 "(truncated or drifted); re-downloading")
        return None
    digest = existing_file_digest(dest)
    if prior_digest is not None and digest != prior_digest:
        progress(f"fetch hrrr {label}: existing {dest.name} does not "
                 "match the sha256 recorded in the prior fetch manifest; "
                 "re-downloading")
        return None
    return digest


def _quarantine_rejected(dest: Path, progress, label: str) -> Path:
    """Move a failed existing file aside (never deleted, never reused).

    The name is proven free first.  ``.rejected-<time_ns>`` alone is
    not: two quarantines inside one clock tick -- or two processes --
    produce the same name, and ``os.replace`` onto it silently destroys
    the earlier evidence, which is the one thing quarantine promises not
    to do.
    """

    aside = fetch_guard.quarantine(dest)
    progress(f"fetch {label}: moved rejected file aside to "
             f"{aside.name}")
    return aside


def _quarantine_inventory_payload(dest: Path, out: Path, entry: dict,
                                  progress, label: str) -> str:
    """Set aside a payload the record bar is about to refuse.

    Returns the sentence the refusal prints about the disk.  Two things
    have to be true afterwards: no unverified GRIB is left where a
    consumer would read it as a fetch product, and the message says
    where the bytes went -- nothing is ever deleted, so an operator who
    accepts the change keeps the evidence of what arrived.

    The ``.idx`` the backbone kept beside the object goes with it: it is
    the very index whose census disagreed, and it is what a later
    ordinary run would otherwise resume against.
    """

    moved: list[str] = []
    for path in (dest, out / entry["idx_name"] if entry.get("idx_name")
                 else None):
        if path is None or not path.is_file():
            continue
        aside = fetch_guard.quarantine(path, tag="inventory-change")
        moved.append(aside.name)
    if not moved:
        return "Nothing was left on disk."
    progress(f"fetch {label}: quarantined {', '.join(moved)}")
    return (f"The transfer had already completed, so the payload is on "
            f"disk; it has been moved aside as {', '.join(moved)} in "
            f"{out} and no manifest was written, so nothing downstream "
            f"will read it as a fetch product.  Nothing was deleted.")


#: Suffix fragments that mark a file as *already* set aside.  A force
#: sweep leaves these alone: re-quarantining evidence only renames it,
#: and the audited property that quarantine artifacts are never treated
#: as canonical and never recursively quarantined is worth keeping.
_QUARANTINE_MARKS = (".rejected-", ".inventory-change-")


def _force_quarantine_output(out: Path, progress, label: str) -> list[str]:
    """``--force-refetch``: move every existing file in ``out`` aside.

    Two properties, and the order between them is the whole point.

    *The receipt goes first.*  Force used to move each requested payload
    aside one at a time, as its turn came round in the fetch loop, while
    the previous ``fetch-manifest.json`` stayed canonical -- so a kill
    (or a network failure) part-way through left a readable manifest
    claiming a digest for bytes that had just been renamed away.  That
    directory lies until something re-reads it.  Quarantining every
    receipt in :data:`FETCH_RECEIPT_NAMES` -- and the series -- *before*
    touching a single payload means an interrupted force leaves a
    directory with payloads and no receipt, which the front door already
    refuses accurately.  The receipt class is the whole front door, not
    just the manifest and the checksum list: ``inputs.txt`` is a list of
    resolved payload paths and ``prep-command.txt`` binds the series, so
    sweeping either of them at payload rank could leave a readable file
    naming bytes that had already been renamed aside.

    *Every file, as advertised.*  The CLI has always said force moves
    every existing file in ``--out`` aside; it actually moved only the
    payload paths the new request selected.  Forecast hours outside a
    shortened window, ``.idx`` indexes, selector files, stale ``.part``
    files and unrelated files all stayed canonical -- old payloads that
    the new manifest does not list, and sidecars that can block the
    recovery force was invoked to perform.  Now the sweep matches the
    sentence.

    Nothing is deleted and nothing already set aside is touched again.
    Subdirectories are left alone: fetch writes no directories into
    ``--out``, so anything that is one belongs to the operator.
    """

    if not out.is_dir():
        return []
    def order(path: Path) -> tuple[int, int, str]:
        if path.name in FETCH_RECEIPT_NAMES:
            # Manifest first, then the rest of the front door, in the
            # order the constant declares rather than alphabetically:
            # `SHA256SUMS` sorts before `fetch-manifest.json`, so a
            # by-name sort left the one receipt the front door reads
            # canonical for longer than the ones it does not.
            return 0, FETCH_RECEIPT_NAMES.index(path.name), path.name
        if path.name.endswith("-series.tsv"):
            return 1, 0, path.name
        return 2, 0, path.name

    candidates = sorted(
        (path for path in out.iterdir()
         if path.is_file()
         and not any(mark in path.name for mark in _QUARANTINE_MARKS)),
        key=order)
    moved: list[str] = []
    for path in candidates:
        aside = fetch_guard.quarantine(path)
        moved.append(aside.name)
    if moved:
        progress(f"fetch {label}: --force-refetch moved {len(moved)} "
                 f"existing file(s) in {out} aside (receipts first, so no "
                 "manifest survives claiming replaced bytes); nothing was "
                 "deleted: " + ", ".join(moved))
    return moved


def _degrade_to_python_transport(reason: str) -> FetchEngineChoice:
    """Say the tax out loud, once per degrade, and record it."""

    explain.warn(_PYTHON_TRANSPORT_TAX.format(reason=reason),
                 _PYTHON_TRANSPORT_TAX_WHY)
    return FetchEngineChoice(
        "python", None, PYTHON_FALLBACK_SELECTION, reason)


def select_fetch_engine(requested: str, *, progress=print
                        ) -> FetchEngineChoice:
    """Resolve ``--engine``, and say when the answer was not asked for.

    ``rust`` is explicit and fails loudly when the backbone is not built;
    ``python`` never looks; ``auto`` prefers the backbone and falls
    through to the Python transport when it is absent or unusable.

    That fall-through used to be silent on the missing-binary branch --
    the branch an ordinary install without the bridges bundle takes every
    time.  It is the expensive one: the Python transport has no
    whole-file mode at all, so degrading also silently converts a
    whole-file request into idx subsetting, and the source has carried
    the measurement of that difference (16x) since before this warning
    existed.  Under warn-not-block it is still not a refusal: the
    transport is correct, the run continues, and the reader is told what
    it costs and how to stop paying it before the bytes move rather than
    after.
    """

    if requested not in FETCH_ENGINES:
        raise ValueError(f"unknown fetch engine {requested!r}; expected one "
                         f"of {FETCH_ENGINES}")
    if requested == "python":
        return FetchEngineChoice("python", None, "python-requested")

    from woof import rustwx_fetch

    binary = rustwx_fetch.find_fetch_bin()
    if binary is None:
        if requested == "rust":
            raise ValueError(
                "--engine rust needs the vendored fetch backbone, which is "
                f"not built.\n  {rustwx_fetch.fetch_remedy()}")
        return _degrade_to_python_transport(
            "the vendored rw_fetch backbone is not installed")
    ok, evidence = rustwx_fetch.probe_fetch_bin(binary)
    if not ok:
        if requested == "rust":
            raise ValueError(f"--engine rust: {binary} -- {evidence}")
        progress(f"fetch: the rust backbone at {binary} is unusable "
                 f"({evidence}); using the Python transport")
        return _degrade_to_python_transport(
            f"the rust backbone at {binary} is unusable ({evidence})")
    return FetchEngineChoice("rust", binary, "rust")


def resolve_fetch_engine(requested: str, *, progress=print
                         ) -> tuple[str, Path | None]:
    """``select_fetch_engine`` for the callers that only want the pair.

    Kept at its original arity on purpose: widening it would break every
    two-value unpack in and outside this repository for a field the
    receipt writers reach through :func:`select_fetch_engine` anyway.
    """

    choice = select_fetch_engine(requested, progress=progress)
    return choice.engine, choice.binary


def _rw_fetch_hrrr(*, binary: Path, cycle: datetime, hour: int, kind: str,
                   host: str, mode: str, out: Path,
                   cache_dir: Path | None, progress,
                   retries: int = 0, shown_name: str | None = None,
                   streams: int | None = None, byte_relay=None,
                   analyzed_aerosol: bool = False, extras=()) -> dict:
    """One HRRR product through the Rust backbone; returns its record.

    The backbone names each object after the URL it came from, so the
    atmosphere lands under the name WOOF already uses and only the soil
    product -- carved out of ``wrfprs`` -- needs renaming afterwards.

    ``retries`` is how many more times a transfer the network cut off is
    asked for, the same budget the Python transport spends.  The Rust
    route used to spend none: one object failing after the backbone's
    own chunk retries ended the whole fetch, however many hours of other
    files had already landed.

    ``shown_name`` is the name the file is filed under, which the one
    completion line says; the soil product lands as ``wrfprs`` and is
    renamed afterwards, and the line used to name the transient file.

    ``streams`` is this file's share of the fetch's chunk-stream budget
    (:func:`woof.fetch_pool.chunk_streams_per_file`).  ``byte_relay``
    makes one ``(received, total)`` sink per attempt
    (:meth:`woof.progress.TransferMonitor.relay`), which is how the
    bytes of an object still in the backbone's memory reach the progress
    line and the run's ``fetch_progress`` events.
    """

    from tools import download_hrrr_native_subset as range_transport
    from woof import rustwx_fetch

    selectors = (range_transport.atmosphere_selectors(
                     extras, analyzed_aerosol=analyzed_aerosol)
                 if kind == "atmosphere" else range_transport.soil_selectors())
    optional: tuple[tuple[str, str], ...] = ()
    if kind == "atmosphere" and mode != "full-file":
        # The backbone refuses a selector the index does not list, and
        # these records are read only where published, so the index says
        # which of them to ask for before the transfer is planned.
        optional = _hrrr_optional_surface(
            cycle=cycle, hour=hour, host=host, retries=retries)
        selectors += range_transport.optional_surface_selectors(optional)
    patterns = out / f".rw-fetch-{kind}-f{hour:02d}.selectors"
    rustwx_fetch.write_pattern_file(patterns, selectors)
    attempt = 0
    try:
        while True:
            try:
                record = rustwx_fetch.run_fetch(
                    binary, model="hrrr", date=f"{cycle:%Y%m%d}",
                    cycle=cycle.hour, hours=(hour,),
                    product=RW_FETCH_HRRR_PRODUCTS[kind],
                    source=RW_FETCH_SOURCES[host], mode=mode, out=out,
                    pattern_file=patterns,
                    exclusions=(range_transport.ACCUMULATION_EXCLUSION,),
                    cache_dir=cache_dir, keep_idx=True, streams=streams,
                    on_progress=(None if byte_relay is None
                                 else byte_relay()))
                break
            except rustwx_fetch.RwFetchError as error:
                # The tree's shared schedule (at most four more asks, 2,
                # 4, 8 and 16 s apart), not an immediate re-ask: the
                # network that just cut this transfer off is still the
                # network the next one meets.
                budget = min(retries,
                             fetch_endpoints.TRANSIENT_ATTEMPTS - 1)
                wait = (fetch_endpoints.retry_delay(
                    error, attempt + 1,
                    wait_limit_s=fetch_endpoints.TRANSIENT_WAIT_LIMIT_S)
                    if attempt < budget else None)
                if wait is None:
                    raise
                attempt += 1
                # Counted against the asks this loop will make, not the
                # budget it was handed: "1 of 5" from a loop capped at
                # four promised the reader an ask that never came.
                progress(f"fetch hrrr f{hour:02d} {kind}: {error.reason}; "
                         f"asking again ({attempt} of {budget}) in "
                         f"{wait:g} s")
                fetch_pool.sleep_unless_stopped(wait, sleep=time.sleep)
    except RuntimeError as error:
        # A selector that matches nothing is this host publishing an
        # inventory we do not recognise, not a network fault; the caller
        # may legitimately try the next host.
        if "matched no index record" in str(error):
            raise range_transport.IndexInventoryError(str(error)) from error
        raise
    finally:
        patterns.unlink(missing_ok=True)
    if len(record["files"]) != 1:
        raise RuntimeError(
            f"rw_fetch returned {len(record['files'])} files for one "
            "forecast hour")
    entry = record["files"][0]
    # The cache's own accounting is a record-level fact, and the caller
    # only ever sees the file entry; carry it across rather than widen
    # every return in this route.  A backbone predating the key simply
    # says nothing, and the manifest reports nothing for it.
    dedup = record.get("dedup")
    if isinstance(dedup, dict):
        entry["dedup"] = dedup
    if entry.get("mode") == "idx-subset" and kind == "atmosphere":
        entry["optional_surface"] = list(
            range_transport.optional_surface_selectors(optional))
    progress(f"fetch hrrr f{hour:02d} {kind}: {shown_name or entry['name']} "
             f"{entry['bytes']:,} B in {entry['wall_seconds']:.1f} s "
             f"({entry['source']}, {entry['mode']} -- "
             f"{entry['mode_reason']})")
    return entry


def _hrrr_host_byte_mode(mode: str, url: str) -> str:
    """The byte mode ``mode`` means on the host serving ``url``.

    ``auto`` takes the record subset from the archive, but the whole
    object from a host the NOMADS governor paces: that host is held to
    one request every 2.5 s node-wide (:mod:`woof.nomads_governor`), and
    a wrfnat subset is about 108 coalesced ranges where the whole object
    is about 45 chunks of 16 MiB, so there the request count, not the
    bytes, sets the wall clock.  It serves a cycle only until the archive
    holds it.  A named mode is taken as named.
    """

    from woof.nomads_governor import is_nomads_url

    if mode == "auto" and is_nomads_url(url):
        return "full-file"
    return mode


def _hrrr_optional_surface(*, cycle: datetime, hour: int, host: str,
                           retries: int = 0) -> tuple[tuple[str, str], ...]:
    """Which ``OPTIONAL_SURFACE_FIELDS`` this hour's wrfnat ``.idx`` lists.

    One small GET (about 70 kB) through the NOMADS governor, before the
    backbone plans its ranges.  An index the host does not have (403,
    404, 410) lists nothing: under ``auto`` the backbone then takes the
    whole object, under ``idx-subset`` it refuses, as it always did.  A
    transient fault is asked again on the tree's shared schedule.
    """

    from tools import download_hrrr_native_subset as range_transport

    url = hrrr_object_url(cycle, hour, "wrfnat", transport=host) + ".idx"
    budget = min(retries, fetch_endpoints.TRANSIENT_ATTEMPTS - 1)
    attempt = 0
    while True:
        try:
            request = Request(url, headers={"User-Agent": "rw-wps-hrrr-range/1"})
            with paced_urlopen(request, timeout=60) as response:
                text = response.read().decode("ascii", "replace")
            return range_transport.optional_surface_published(text)
        except HTTPError as error:
            if error.code in (403, 404, 410):
                return ()
            failure: BaseException = error
        except (URLError, OSError) as error:
            failure = error
        wait = (fetch_endpoints.retry_delay(
            failure, attempt + 1,
            wait_limit_s=fetch_endpoints.TRANSIENT_WAIT_LIMIT_S)
            if attempt < budget else None)
        if wait is None:
            raise failure
        attempt += 1
        fetch_pool.sleep_unless_stopped(wait, sleep=time.sleep)


#: Cache-accounting keys the Rust backbone reports per transfer.
_DEDUP_FIELDS = ("cache_bytes_written", "cache_bytes_deduplicated",
                 "reference_entries")


def _cache_dedup_summary(reports) -> dict:
    """Sum what this run's download cache wrote versus what it reused.

    One full-file object reaches the backbone's cache under two key
    shapes and used to land as two whole copies; the second is now a
    reference to the first, and the receipt says so in bytes.  A
    backbone built before the key existed reports nothing, so
    ``transfers`` is 0 and the byte columns stay accurate rather than
    claiming a saving that was never measured.
    """

    totals = {"transfers": 0}
    totals.update({field: 0 for field in _DEDUP_FIELDS})
    for report in reports:
        if not isinstance(report, dict):
            continue
        totals["transfers"] += 1
        for field in _DEDUP_FIELDS:
            value = report.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                totals[field] += value
    return totals


def _expand_selector(selector: str):
    """``A|B:LEVEL`` -> ``[('A', 'LEVEL'), ('B', 'LEVEL')]``.

    The alternation is how one selector covers two provider spellings of
    the same record (``CLMR`` on AWS, ``CLWMR`` on NOMADS).
    """

    variable, _, level = selector.partition(":")
    return [(spelling.strip(), level if _ else None)
            for spelling in variable.split("|")]


def count_selectors_in_index(index_text: str, selectors: tuple[str, ...],
                             exclusion: str | None = None) -> int:
    """How many ``.idx`` records the exact ``VAR:LEVEL`` selectors take.

    Exact on both columns, and skipping any line carrying ``exclusion``
    -- the same rule the Rust backbone applies, so the derived bar and
    the transfer agree by construction.
    """

    wanted = {f"{spelling}:{level}" if level is not None else spelling
              for selector in selectors
              for spelling, level in _expand_selector(selector)}
    matched = 0
    for line in index_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if exclusion is not None and exclusion in stripped:
            continue
        fields = stripped.rstrip(":").split(":")
        if len(fields) < 6:
            continue
        if f"{fields[3]}:{fields[4]}" in wanted:
            matched += 1
    return matched


def _hrrr_derived_bar(entry: dict, *, kind: str, out: Path,
                      analyzed_aerosol: bool = False,
                      extras=()) -> int | None:
    """The live selection count behind one Rust HRRR transfer.

    In ``idx-subset`` mode the backbone selected the records itself and
    reports how many.  In ``full-file`` mode nothing was selected -- the
    whole object landed -- so the selection census is recomputed from
    the index the backbone kept beside it, but **only** when that index
    was proven to cover the whole object.  A short index is exactly why
    the full file was taken; counting a selection out of it would report
    a deficit that is an artefact of the index, not of the data.  With
    no complete index the bar cannot be derived; the certified constant
    stands in, :func:`woof.fetch_bars.resolve_bar` says so, and the
    fail-closed ``hrrr_grib2_bridge`` remains the completeness gate --
    it selects by exact field identity and refuses a file whose
    inventory it cannot satisfy.
    """

    if entry["mode"] == "idx-subset":
        return entry.get("selected_record_count")
    if entry.get("probe", {}).get("idx_covers_object") is not True:
        return None
    index_name = entry.get("idx_name")
    if not index_name:
        return None
    index_path = out / index_name
    if not index_path.is_file():
        return None
    from tools import download_hrrr_native_subset as range_transport

    selectors = (range_transport.atmosphere_selectors(
                     extras, analyzed_aerosol=analyzed_aerosol)
                 if kind == "atmosphere" else range_transport.soil_selectors())
    return count_selectors_in_index(
        index_path.read_text(encoding="utf-8"), selectors,
        range_transport.ACCUMULATION_EXCLUSION)


def _download_one_hrrr_product(
        *, engine: str, engine_bin: Path | None, mode: str,
        cycle: datetime, hour: int, kind: str, host: str, url: str,
        dest: Path, dest_name: str, source_name: str, out: Path,
        label: str, cache_dir: Path | None, workers: int, retries: int,
        bar_kind: str, certified: int, accept_inventory_change: bool,
        progress, dedup: list[dict] | None = None,
        streams: int | None = None, byte_relay=None,
        analyzed_aerosol: bool = False, extras=(),
        mode_sink: dict | None = None):
    """Download one HRRR product from one host; ``(bar, url)``.

    ``dedup`` collects each Rust transfer's cache accounting, which the
    receipt sums; the download's own answer stays the two-value pair
    every caller unpacks.  ``mode_sink`` hears the byte mode the file
    landed in (``mode_sink["mode"]``): under ``auto`` the backbone's
    probe picks it per file, and the Python transport only takes
    subsets.  ``streams`` and ``byte_relay`` reach the Rust backbone
    only (see :func:`_rw_fetch_hrrr`).

    Raises
    :class:`tools.download_hrrr_native_subset.IndexInventoryError` when
    *this host's* published index does not carry the expected inventory,
    which the caller answers by trying the next host.
    """

    from woof.fetch_bars import resolve_bar
    from tools import download_hrrr_native_subset as range_transport

    if engine == "rust":
        # The backbone names its output after the URL, so the soil
        # product lands as `wrfprs` and Python renames it to `soil`
        # afterwards.  A kill in that gap leaves a canonical orphan
        # under the source name: the next Rust run refuses because its
        # own destination already exists, and force never selected that
        # name, so the directory was unrecoverable without hand
        # intervention.  Set the orphan aside first -- it is unclaimed
        # by any receipt and nothing is deleted.
        landing = out / source_name
        if landing != dest and landing.exists():
            _quarantine_rejected(
                landing, progress,
                f"hrrr {label} orphaned {source_name}")
        entry = _rw_fetch_hrrr(
            binary=engine_bin, cycle=cycle, hour=hour, kind=kind,
            host=host, mode=mode, out=out, cache_dir=cache_dir,
            progress=progress, retries=retries, shown_name=dest_name,
            streams=streams, byte_relay=byte_relay,
            # Said only when asked, so every other request keeps the
            # backbone call it always made.
            **({"analyzed_aerosol": True} if analyzed_aerosol else {}),
            **({"extras": extras} if extras else {}))
        if dedup is not None and isinstance(entry.get("dedup"), dict):
            dedup.append(entry["dedup"])
        optional_surface = entry.get("optional_surface")
        if mode_sink is not None:
            mode_sink["mode"] = entry.get("mode")
            if optional_surface is not None:
                mode_sink["optional_surface"] = list(optional_surface)
        url = entry["grib_url"]
        landed = out / entry["name"]
        if landed != dest:
            # Only the soil product needs renaming: it is carved out of
            # wrfprs and ArWen files it under its own name.
            _rename_stream_payload(landed, dest, entry)
        else:
            _remember_stream_facts(dest, entry)
        # The census can only be read after the object has landed, so a
        # tripped tripwire here refuses AFTER a payload exists.  Quarantine
        # it first, then let the refusal say what is actually on disk --
        # the alternative is a manifestless directory the next ordinary
        # run also refuses, under a message swearing nothing was
        # downloaded.
        bar = resolve_bar(
            bar_kind, _hrrr_derived_bar(
                entry, kind=kind, out=out,
                **({"analyzed_aerosol": True} if analyzed_aerosol else {}),
                extras=extras),
            accept_inventory_change=accept_inventory_change,
            # The optional records ride beside the certified selection
            # where the index lists them; the bar stays the required set.
            certified=certified + len(optional_surface or ()),
            progress=progress,
            on_refusal=lambda: _quarantine_inventory_payload(
                dest, out, entry, progress, f"hrrr {label}"))
        if entry["mode"] == "idx-subset":
            expected_messages: int | None = bar.expected
        else:
            # The whole object landed: its census is the index's record
            # count, not the selection's.
            expected_messages = (
                entry.get("probe", {}).get("idx_record_count") or None)
        observed = _fetched_records(dest)
        if expected_messages is not None and observed != expected_messages:
            _quarantine_rejected(dest, progress, f"hrrr {label}")
            raise ValueError(
                f"downloaded {dest_name} carries {observed} GRIB2 messages, "
                f"expected {expected_messages} ({entry['mode']} transfer); "
                "the file has been moved aside, nothing was deleted")
        return bar, url

    if mode_sink is not None:
        mode_sink["mode"] = "idx-subset"
    request = range_transport.ProductRequest(
        url=url,
        index_url=url + ".idx",
        index_path=out / f"{source_name}.idx",
        destination=dest,
        kind=kind,
        analyzed_aerosol=analyzed_aerosol,
    )
    # The selection itself is the derivation here: with the count clause
    # left on, an inventory change is refused inside the transport with
    # a message naming exactly what moved; with it accepted, the live
    # selection count becomes the bar.
    try:
        range_transport._download_product(
            request, workers=workers, retries=retries,
            expected_count=None if accept_inventory_change else certified,
            **({"extras": extras} if extras else {}))
    except URLError as error:
        raise RuntimeError(hrrr_reach_refusal(
            host, cycle, hour, kind, error)) from None
    observed = count_grib2_messages(dest)
    optional = ()
    if kind == "atmosphere" and request.index_path.is_file():
        # The selection took the optional records this index lists
        # (OPTIONAL_SURFACE_FIELDS); the bar stays the required set.
        optional = range_transport.optional_surface_published(
            request.index_path.read_text(encoding="ascii", errors="replace"))
        if mode_sink is not None:
            mode_sink["optional_surface"] = list(
                range_transport.optional_surface_selectors(optional))
    # An unaccepted change is normally refused inside the transport,
    # before any range GET.  Should one ever reach here the payload has
    # landed, so this refusal quarantines it and tells the truth too.
    bar = resolve_bar(bar_kind, observed,
                      accept_inventory_change=accept_inventory_change,
                      certified=certified + len(optional),
                      progress=progress,
                      on_refusal=lambda: _quarantine_inventory_payload(
                          dest, out, {"idx_name": f"{source_name}.idx"},
                          progress, f"hrrr {label}"))
    if observed != bar.expected:
        raise ValueError(
            f"downloaded {dest_name} carries {observed} GRIB2 messages, "
            f"expected {bar.expected}; the upstream .idx inventory has "
            "drifted")
    return bar, url


def _wait_for_hrrr_product(*, cycle: datetime, hour: int, product: str,
                           candidates: tuple[str, ...], probe, clock,
                           deadline: float, sleeper, progress,
                           label: str) -> str | None:
    """Block until a transport serves the object AND its ``.idx``.

    Returns the transport name, or None when the deadline expires.
    Rounds are separated by at most :data:`HRRR_WAIT_POLL_SECONDS`.
    Both the object and its index must answer: the ``.idx`` can lag its
    GRIB by a moment, and the range transport needs both.

    Every candidate is polled each round, in ladder order, so the
    operational server keeps its head start -- watching it is the whole
    point of ``--wait-for``, and it is the host that will see the hour
    first.  What the poll DECIDES, though, is throughput: among the
    candidates that answered this round, the quickest one takes the
    transfer.  A file the archive has already mirrored has nothing left
    to gain from the paced host.

    It stops waiting the moment another file of the same download
    fails (:func:`woof.fetch_pool.sleep_unless_stopped`).  It used to
    poll on to publication or the deadline, which held the refusal for
    as long as the latest hour of a live cycle took to appear.
    """

    ranked = {endpoint.name: endpoint.transfer_rank
              for endpoint in fetch_endpoints.ladder("hrrr")}
    announced = False
    while True:
        published = []
        for name in candidates:
            url = hrrr_object_url(cycle, hour, product, transport=name)
            if probe(url) and probe(url + ".idx"):
                published.append(name)
        if published:
            chosen = min(published,
                         key=lambda name: (ranked.get(name, 0),
                                           candidates.index(name)))
            if announced:
                progress(f"fetch hrrr {label}: published on {chosen}")
            return chosen
        remaining = deadline - clock()
        if remaining <= 0:
            return None
        if not announced:
            progress(f"fetch hrrr {label}: not yet published on "
                     f"{'/'.join(candidates)}; polling every "
                     f"{HRRR_WAIT_POLL_SECONDS} s (up to "
                     f"{remaining / 60.0:.0f} more min)")
            announced = True
        fetch_pool.sleep_unless_stopped(
            min(HRRR_WAIT_POLL_SECONDS, remaining), sleep=sleeper)


def fetch_hrrr(*, cycle: datetime, hours: tuple[int, ...],
               area: Area | None, out: Path, workers: int = 8,
               retries: int = 5, progress=print,
               force: bool = False, transport: str = "s3",
               wait: bool = False,
               wait_timeout_s: float = HRRR_WAIT_TIMEOUT_DEFAULT_MINUTES * 60,
               probe=None, sleeper=time.sleep,
               clock=time.monotonic,
               engine: str = "python", engine_bin: Path | None = None,
               engine_selection: str | None = None,
               mode: str = "auto", cache_dir: Path | None = None,
               accept_inventory_change: bool = False,
               file_workers: int | None = None,
               transport_fallback: tuple[str, ...] = (),
               on_hour_ready=None,
               runtime_surface: str | None = None,
               analyzed_aerosol: bool = False, extras=()) -> Path:
    """Byte-range download the native HRRR subset series into ``out``.

    Single writer per ``--out``: the prior-receipt read, the ``force``
    sweep, the transfers and every receipt publication run under an
    exclusive OS lock on the output root, so two concurrent HRRR fetches
    cannot publish receipts describing each other's bytes.

    See :func:`_fetch_hrrr_locked` for the transfer itself.
    ``on_hour_ready(hour, manifest)`` runs after each complete hour's
    ordinary checks and durable manifest publication, in lead order.
    """

    if fetch_endpoints.policy_uses_aws("hrrr"):
        transport = fetch_endpoints.policy_transport("hrrr", transport)
        transport_fallback = ()
    with _fetch_out(out, progress=progress):
        return _fetch_hrrr_locked(
            cycle=cycle, hours=hours, area=area,
            out=deep_io_path(out, DOWNLOAD_DEPTH_BUDGET), workers=workers,
            retries=retries, progress=progress, force=force,
            transport=transport, wait=wait, wait_timeout_s=wait_timeout_s,
            probe=probe, sleeper=sleeper, clock=clock, engine=engine,
            engine_bin=engine_bin, engine_selection=engine_selection,
            mode=mode, cache_dir=cache_dir,
            accept_inventory_change=accept_inventory_change,
            file_workers=file_workers,
            transport_fallback=transport_fallback,
            on_hour_ready=on_hour_ready, runtime_surface=runtime_surface,
            analyzed_aerosol=analyzed_aerosol, extras=extras)


def _fetch_hrrr_locked(*, cycle: datetime, hours: tuple[int, ...],
                       area: Area | None, out: Path, workers: int = 8,
                       retries: int = 5, progress=print,
                       force: bool = False, transport: str = "s3",
                       wait: bool = False,
                       wait_timeout_s: float = (
                           HRRR_WAIT_TIMEOUT_DEFAULT_MINUTES * 60),
                       probe=None, sleeper=time.sleep,
                       clock=time.monotonic,
                       engine: str = "python",
                       engine_bin: Path | None = None,
                       engine_selection: str | None = None,
                       mode: str = "auto", cache_dir: Path | None = None,
                       accept_inventory_change: bool = False,
                       file_workers: int | None = None,
                       transport_fallback: tuple[str, ...] = (),
                       on_hour_ready=None,
                       runtime_surface: str | None = None,
                       analyzed_aerosol: bool = False, extras=()) -> Path:
    """The HRRR transfer, with the output-root lock already held.

    ``analyzed_aerosol``: the configuration's preparation reads HRRR's
    analyzed aerosol numbers (QNWFA/QNIFA), so the atmosphere subset
    selects them too and a kept subset without them is fetched again.

    Reuses the proven ``.idx`` selection/range transport in
    :mod:`tools.download_hrrr_native_subset` per product (atmosphere
    ``wrfnat`` subset + soil records of ``wrfprs``), adds resumability,
    and writes the fetch manifest plus ``SHA256SUMS``.  An existing file
    is skipped only when it passes the same completeness bar as a fresh
    download (envelope walk + the exact 561/18 record counts) and, when
    a prior manifest recorded its digest, still matches that digest;
    anything else is moved aside and re-downloaded, never re-blessed.
    HRRR files are CONUS-wide: ``--area`` is a coverage check, not a
    crop.

    ``transport`` selects the host ('s3' or 'nomads'; both serve
    byte-identical files and indexes, so every bar above is
    host-independent and a directory fetched over one host resumes over
    the other).  ``wait`` is the live-cycle mode: hours are fetched in
    order, each product polled (at most every
    :data:`HRRR_WAIT_POLL_SECONDS` seconds) until it publishes; under
    ``transport='auto'`` each round tries NOMADS first, then S3.  On
    timeout the manifest is still written for the contiguous complete
    prefix -- so a re-run of the same command resumes instead of
    refusing -- and a ``RuntimeError`` reports exactly what was and was
    not fetched.

    The manifest is republished after every **completed hour**, and it
    claims only the files of the hours it declares complete.  Both
    halves matter: without the first, an ordinary kill after many good
    hours leaves a fresh output with valid payloads and no receipt at
    all; without the second, the timeout path publishes the
    half-fetched hour's atmosphere in ``files`` and ``SHA256SUMS``
    while ``forecast_hours`` names only the earlier prefix -- one
    receipt with two definitions of complete.  A half-fetched hour
    stays on disk unclaimed and is re-verified under the ordinary bars
    on the next run.

    ``engine`` selects the downloader: ``'python'`` is the stdlib
    byte-range transport in :mod:`tools.download_hrrr_native_subset`,
    ``'rust'`` the vendored ``rw_fetch`` backbone (parallel range GETs,
    the cross-process NOMADS rate governor, a disk cache).  ``mode`` is
    the byte transport the backbone uses -- under ``'auto'`` a lagging
    or short ``.idx`` lands the **whole** object instead of a subset,
    which is bigger on disk and still exactly what
    ``hrrr_grib2_bridge`` wants, because that bridge selects by field
    identity rather than by file size.

    ``transport_fallback`` lists further hosts to try, in order, when
    the chosen one publishes an index whose inventory this WOOF does
    not recognise.  The two hosts do **not** publish identical index
    vocabularies -- see ``download_hrrr_native_subset.FIELD_ALIASES`` --
    and a host that has genuinely changed something is a reason to move
    on with an explanation rather than to abort a fetch the other host
    can serve.  Network faults remain the transport's own to retry.
    """

    from woof.fetch_bars import resolve_bar
    from tools import download_hrrr_native_subset as range_transport

    if engine not in FETCH_ENGINES or engine == "auto":
        raise ValueError(
            "fetch_hrrr needs a resolved engine ('rust' or 'python'); the "
            "CLI resolves 'auto' first via resolve_fetch_engine")
    if mode not in FETCH_MODES:
        raise ValueError(f"unknown fetch mode {mode!r}; expected one of "
                         f"{FETCH_MODES}")
    if engine == "rust" and engine_bin is None:
        raise ValueError("engine 'rust' needs the resolved rw_fetch binary")
    bar_kinds = {"atmosphere": ("hrrr-atmosphere-aerosol" if analyzed_aerosol
                                else "hrrr-atmosphere"),
                 "soil": "hrrr-soil"}
    from woof.ingest.native_extras import requested_extra_fields
    extras = requested_extra_fields(extras)
    expected_counts = {"atmosphere": range_transport.atmosphere_record_count(
                           analyzed_aerosol=analyzed_aerosol, extras=extras),
                       "soil": range_transport.SOIL_RECORD_COUNT}
    from woof.source_adapters import get_source_adapter
    runtime_adapter = get_source_adapter("hrrr")
    runtime_rows = runtime_adapter.runtime_surface_fields
    from woof.runtime_surface_fetch import declared_runtime_surface_fields
    runtime_declared = declared_runtime_surface_fields(
        runtime_surface, runtime_adapter)
    # Seeded, not empty: a completed resume downloads nothing and would
    # otherwise republish record_bars as [], erasing an accepted
    # inventory change the directory's files really were fetched under.
    bars: dict[str, object] = {}
    candidates = (("nomads", "s3") if transport == "auto"
                  else (transport,))
    for name in candidates:
        _hrrr_transport_base(name)  # unknown transports fail before I/O
    if transport == "auto" and not wait:
        raise ValueError(
            "transport 'auto' reaches fetch_hrrr only in --wait-for mode "
            "(per-file polling); a plain fetch resolves it first via "
            "resolve_hrrr_transport, as the CLI does")
    if wait and (not math.isfinite(wait_timeout_s) or wait_timeout_s <= 0):
        raise ValueError("the --wait-for timeout must be positive")
    if probe is None:
        probe = _head_ok
    if area is not None:
        validate_fetch_area("hrrr", area)
    if not force:
        # Inside the output lock, before any provider is asked: a file
        # name carries the cycle hour but not the date, so another day's
        # files would otherwise pass every per-file bar here.
        require_matching_request(out, source="hrrr", cycle=cycle, area=area,
                                 mode=None)
    out.mkdir(parents=True, exist_ok=True)
    if force:
        # Receipts first, then every other existing file -- including the
        # `.idx` indexes, whose byte-identity guard could otherwise block
        # the very host failover force was invoked to unblock.
        _force_quarantine_output(out, progress, "hrrr")
    prior_digests = _prior_manifest_digests(out)
    prior_entries = _prior_manifest_entries(out)
    prior_records = _prior_manifest_records(out)
    previous = out / FETCH_MANIFEST_NAME
    previous_extras = ()
    if previous.is_file():
        previous_extras = tuple(json.loads(previous.read_text()).get("native_extras", ()))
    if previous_extras != extras:
        # A full-file census alone cannot prove which subset was requested.
        prior_digests = {}
        prior_entries = {}
        prior_records = {}
    bars.update(_prior_manifest_bars(out))
    # One record set per series: hrrr_grib2_bridge refuses a series whose
    # hours select different records.  A subset used to lack PBLH, LCDC,
    # MCDC and HCDC, which the bridge reads from a whole file, so
    # `--mode idx-subset` over an interrupted full-file folder was refused
    # 80-146 s into the preparation (updraft factory, 2023-08-07 and
    # 2019-06-09).  A subset now takes them wherever the index lists them
    # (receipt key ``optional_surface``), so whole files and current
    # subsets mix; a subset from before that is set aside and fetched
    # again.  What each file landed as is read from the receipt key, else
    # its census.
    subset_ceiling = {
        "atmosphere": range_transport.atmosphere_record_count(
            analyzed_aerosol=True, extras=extras)
        + len(range_transport.OPTIONAL_SURFACE_FIELDS),
        "soil": range_transport.SOIL_RECORD_COUNT + len(runtime_rows)}
    # What a download lands as: the Python transport only takes subsets;
    # the backbone takes the named mode, and under 'auto' its probe
    # decides per file.
    lands = ("idx-subset" if engine == "python"
             else None if mode == "auto" else mode)
    series_mode: dict[str, str | None] = {}

    def kept_index_optional(dest_name: str):
        """The optional surface records the ``.idx`` kept beside an
        atmosphere file lists, or None when no index was kept."""
        index = out / f"{dest_name}.idx"
        if not index.is_file():
            return None
        try:
            return range_transport.optional_surface_published(
                index.read_text(encoding="ascii", errors="replace"))
        except OSError:
            return None

    def kept_mode(dest_name: str, kind: str) -> str | None:
        dest = out / dest_name
        if not dest.exists():
            return None
        records = prior_records.get(dest_name)
        if records is None:
            # On disk but claimed by no receipt (a half-fetched hour):
            # its own census says what it is.
            try:
                records = count_grib2_messages(dest)
            except ValueError:
                return None
        return _hrrr_landed_mode(prior_entries.get(dest_name), records,
                                 subset_ceiling[kind])

    deadline = clock() + wait_timeout_s
    files: list[dict] = []
    complete_hours: list[int] = []
    cache_dedup: list[dict] = []
    pool_summary: dict = {}
    # Set once the pool starts; the transfers read it to report the bytes
    # of an object still in flight (the wait mode runs without one).
    monitor = None
    # The fetch's chunk-stream budget, split over the files in flight;
    # wait mode moves one file at a time and gives it the whole budget.
    streams = fetch_pool.chunk_streams_per_file(
        1 if wait else file_workers, files=2 * len(hours),
        host=fetch_pool.host_key(hrrr_object_url(
            cycle, hours[0] if hours else 0, "wrfnat",
            transport=candidates[0])))

    def publish_manifest(recorded_hours: tuple[int, ...]) -> Path:
        # A receipt claims only files belonging to a COMPLETE hour.  An
        # hour is complete when both its products landed and verified;
        # `files` grows a product at a time, so publishing it whole
        # against an earlier `forecast_hours` prefix -- which is exactly
        # what the wait-timeout branch did -- produced one receipt with
        # two contradictory definitions of completeness: `forecast_hours`
        # said [0] while `files` and `SHA256SUMS` carried the half-done
        # hour 1.  A half-fetched hour stays on disk, unclaimed, and the
        # next run re-verifies it under the ordinary bars.
        wanted = set(recorded_hours)
        claimed = [item for item in files
                   if item["forecast_hour"] in wanted]
        sums = out / "SHA256SUMS"
        _atomic_write_text(sums, "".join(
            f"{item['sha256']}  {item['name']}\n"
            for item in sorted(claimed, key=lambda item: item["name"])))
        entries = claimed + [{
            "name": sums.name, "role": "checksums", "forecast_hour": None,
            "bytes": sums.stat().st_size, "sha256": sha256_file(sums),
            "url": None, "transport": None,
        }]
        payload = _manifest_payload(source="hrrr", cycle=cycle,
                                    hours=recorded_hours, area=area,
                                    files=entries)
        payload["notes"] = (
            "native hybrid-level wrfnat subsets plus wrfprs soil records, "
            "the exact inventory hrrr_grib2_bridge requires; CONUS-wide "
            "(idx subsetting selects records, not areas); NOMADS and S3 "
            "serve byte-identical files, so per-file transports may mix "
            "across resumed runs without weakening the digest bars")
        payload["engine"] = engine
        payload["engine_selection"] = _engine_selection(
            engine, engine_selection)
        # The mode the claimed files ARE in (each file also says its
        # own), not only the one asked for: a resume that kept another
        # fetch's files labelled a whole + subset folder "full-file".
        in_modes = {item.get("mode") for item in claimed}
        payload["mode"] = (in_modes.pop() if len(in_modes) == 1
                           and None not in in_modes else mode)
        payload["requested_mode"] = mode
        if extras:
            payload["native_extras"] = list(extras)
        payload["record_bars"] = [
            bar.as_manifest() for bar in bars.values()]
        # What the backbone's disk cache cost this run.  A full-file
        # object is stored under two key shapes, and until those two
        # entries shared one content-addressed payload the cache held a
        # multiple of what was fetched with nothing on the receipt to
        # say so.
        payload["dedup"] = _cache_dedup_summary(cache_dedup)
        if pool_summary:
            # The completed run's concurrency receipt; wait mode has
            # none, because publication-following is serial by design.
            payload["concurrency"] = dict(pool_summary)
        return write_fetch_manifest(out, payload)

    def transfer_product(hour: int, kind: str, source_name: str,
                         dest_name: str, product: str) -> dict:
        dest = out / dest_name
        # A prior full-file transfer recorded its own census; only
        # fall back to the certified subset count when the manifest
        # predates that key.
        expected = prior_records.get(dest_name, expected_counts[kind])
        if kind == "soil" and runtime_rows and dest_name not in prior_records:
            expected += len(runtime_rows)
        unclaimed_optional = None
        if kind == "atmosphere" and dest_name not in prior_records:
            # An unclaimed subset also carries the optional records its
            # kept index lists.
            unclaimed_optional = kept_index_optional(dest_name)
            expected += len(unclaimed_optional or ())
        label = f"f{hour:02d} {kind}"
        # The stopwatch starts on the WHOLE product, not on the download
        # alone: a verify-skip re-hashes the file on disk and walks its
        # GRIB envelope, and that is real wall clock a reader of the
        # manifest is entitled to see.
        file_started = time.perf_counter()
        digest = None
        runtime_binding = None
        mode_sink: dict = {}
        if (dest.exists() and not force and kind == "atmosphere"
                and analyzed_aerosol
                and prior_records.get(dest_name, 0) < expected_counts[kind]):
            # A subset kept from a fetch whose preparation did not read the
            # aerosol pair: it lacks the records this one needs, so it is
            # set aside (never deleted) and fetched again.  Said in the
            # line: an idx-subset folder fetched without --analyzed-aerosol
            # for an mp=28 aerosol-aware run is moved aside file by file and
            # downloaded again, and the bare "moved rejected file aside"
            # read as a broken fetch (StormNet-plus factory, 2026-10-27).
            _quarantine_rejected(
                dest, progress,
                f"hrrr {label} (subset lacks the analyzed aerosol records "
                "QNWFA/QNIFA this preparation reads; fetch subsets for an "
                "aerosol-aware mp=28 run with `woof fetch --mode idx-subset "
                "--analyzed-aerosol`)")
        if dest.exists() and not force and previous_extras != extras:
            _quarantine_rejected(dest, progress,
                "native extra request changed; prevents reusing a different tracer inventory")
        if dest.exists() and not force:
            digest = _existing_hrrr_digest(
                dest, expected_count=expected,
                prior_digest=prior_digests.get(dest_name),
                progress=progress, label=label)
            if kind == "soil" and runtime_rows:
                prior_runtime = (prior_entries.get(dest_name) or {}).get("runtime_surface")
                if (not isinstance(prior_runtime, list)
                        or {row.get("field") for row in prior_runtime} != {row[0] for row in runtime_rows}
                        # A recorded fallback for a field this request
                        # declares is not a reusable file: the fetch runs
                        # again and refuses that field by name.
                        or any(row.get("fallback_id") and row.get("field") in runtime_declared
                               for row in prior_runtime)):
                    digest = None
                else:
                    runtime_binding = prior_runtime
        # Decided HERE, while `digest` still means "the existing file
        # passed every bar", and not inferred later from the seconds:
        # dividing bytes by seconds means bandwidth for a download and
        # sha256 throughput for a verify-skip, and a receipt that cannot
        # tell them apart reports the second as the first.
        downloaded = digest is None
        if digest is not None:
            chosen = candidates[0]
            url = hrrr_object_url(cycle, hour, product,
                                  transport=chosen)
            prior = prior_entries.get(dest_name)
            if (prior is not None and prior.get("sha256") == digest
                    and prior.get("transport") in HRRR_TRANSPORTS[1:]
                    and isinstance(prior.get("url"), str)):
                # The receipt vouched for these bytes and says which
                # host served them; this run moved nothing, so it keeps
                # saying that rather than naming this run's host.
                chosen, url = prior["transport"], prior["url"]
            progress(f"fetch hrrr {label}: {dest_name} exists, "
                     f"{dest.stat().st_size:,} B / {expected} records "
                     "verified -- skipped")
        else:
            if dest.exists():
                _quarantine_rejected(dest, progress, f"hrrr {label}")
            chosen = candidates[0]
            if wait:
                found = _wait_for_hrrr_product(
                    cycle=cycle, hour=hour, product=product,
                    candidates=candidates, probe=probe, clock=clock,
                    deadline=deadline, sleeper=sleeper,
                    progress=progress, label=label)
                if found is None:
                    publish_manifest(tuple(complete_hours))
                    fetched = (
                        f"complete hours f{complete_hours[0]:02d}.."
                        f"f{complete_hours[-1]:02d} are on disk and "
                        "recorded in the fetch manifest"
                        if complete_hours else
                        "no complete forecast hour was fetched")
                    raise RuntimeError(
                        f"--wait-for timed out after "
                        f"{wait_timeout_s / 60.0:.0f} min: {dest_name} "
                        f"(cycle {cycle:%Y-%m-%dT%H}Z) never appeared "
                        f"on {'/'.join(candidates)}.  {fetched}; "
                        "re-running the same command resumes the "
                        "verified files and extends the window.")
                chosen = found
            # Hosts do not publish identical index vocabularies (see
            # download_hrrr_native_subset.FIELD_ALIASES), so a host
            # whose inventory this ArWen does not recognise is a
            # reason to move to the next one with an explanation --
            # not to abort a fetch the other host can serve.
            attempts = (chosen,) + tuple(
                host for host in transport_fallback if host != chosen)
            started = time.perf_counter()
            for position, host in enumerate(attempts):
                remaining = attempts[position + 1:]
                chosen = host
                url = hrrr_object_url(cycle, hour, product,
                                      transport=host)
                try:
                    bar, url = _download_one_hrrr_product(
                        engine=engine, engine_bin=engine_bin,
                        # The series' mode when one was settled above, so
                        # a file fetched again lands like its neighbours.
                        mode=_hrrr_host_byte_mode(
                            series_mode.get(kind) or mode, url),
                        mode_sink=mode_sink,
                        cycle=cycle, hour=hour, kind=kind, host=host,
                        url=url, dest=dest, dest_name=dest_name,
                        source_name=source_name, out=out, label=label,
                        cache_dir=cache_dir, workers=workers,
                        retries=retries, bar_kind=bar_kinds[kind],
                        certified=expected_counts[kind],
                        **({"extras": extras} if extras else {}),
                        accept_inventory_change=accept_inventory_change,
                        progress=progress, dedup=cache_dedup,
                        **({"analyzed_aerosol": True}
                           if analyzed_aerosol else {}),
                        streams=streams,
                        byte_relay=(None if monitor is None else
                                    functools.partial(monitor.relay,
                                                      dest_name)))
                except range_transport.IndexInventoryError as error:
                    if not remaining:
                        raise
                    # The refused host's .idx is already on disk and
                    # the next host's will differ; move it aside so
                    # the byte-identity guard has a clean slate.
                    stale = out / f"{source_name}.idx"
                    if stale.is_file():
                        _quarantine_rejected(
                            stale, progress, f"hrrr {label} index")
                    progress(
                        f"fetch hrrr {label}: {host} publishes an index "
                        f"this WOOF does not recognise ({error}); "
                        f"falling back to {remaining[0]}")
                    continue
                bars[bar_kinds[kind]] = bar
                break
            if kind == "soil" and runtime_rows:
                from woof.runtime_surface_fetch import append_runtime_surface_records
                runtime_binding = append_runtime_surface_records(
                    dest, adapter=runtime_adapter, cycle=cycle, lead=hour, host=chosen,
                    binary=engine_bin if engine == "rust" else None,
                    cache_dir=cache_dir, progress=progress, streams=streams,
                    declared=runtime_declared)
            digest = _fetched_digest(dest)
            if engine != "rust":
                # The Rust route has already said this, with the
                # transport it used; a second line per file said it
                # twice under two names.
                progress(f"fetch hrrr {label}: {dest_name} "
                         f"{dest.stat().st_size:,} B in "
                         f"{time.perf_counter() - started:.1f} s "
                         f"({chosen})")
        records = (_fetched_records(dest) if downloaded
                   else count_grib2_messages(dest))
        optional_surface = (
            mode_sink.get("optional_surface") if downloaded
            else (prior_entries.get(dest_name) or {}).get("optional_surface"))
        if (optional_surface is None and not downloaded
                and unclaimed_optional is not None):
            optional_surface = list(
                range_transport.optional_surface_selectors(unclaimed_optional))
        return {
            "name": dest_name, "role": kind, "forecast_hour": hour,
            "bytes": dest.stat().st_size, "sha256": digest,
            "url": url, "transport": chosen,
            "records": records,
            # The byte mode this file IS in, per file: a resume may keep
            # files another fetch landed.
            "mode": (mode_sink.get("mode") if downloaded else
                     _hrrr_landed_mode(prior_entries.get(dest_name),
                                       records, subset_ceiling[kind])),
            "seconds": round(time.perf_counter() - file_started, 6),
            # Said, not inferred.  Without it `fetch_throughput` read
            # every HRRR fetch as zero downloads -- so a re-run that
            # verified 3 GiB in seconds and a first run that pulled 3
            # GiB over the network published the same receipt, and a
            # caller could not tell a user which one had happened.
            "downloaded": downloaded,
            **({"runtime_surface": runtime_binding} if runtime_binding else {}),
            # The optional surface records an atmosphere subset took
            # (OPTIONAL_SURFACE_FIELDS); its presence marks a subset as
            # current, so a resume keeps it beside whole files.
            **({"optional_surface": optional_surface}
               if optional_surface is not None else {}),
        }

    products = []
    for hour in hours:
        atmosphere = f"hrrr.t{cycle:%H}z.wrfnatf{hour:02d}.grib2"
        pressure = f"hrrr.t{cycle:%H}z.wrfprsf{hour:02d}.grib2"
        soil = f"hrrr.t{cycle:%H}z.soilf{hour:02d}.grib2"
        products.append((hour, "atmosphere", atmosphere, atmosphere,
                         "wrfnat"))
        products.append((hour, "soil", pressure, soil, "wrfprs"))

    # Settle each product's series mode before anything moves.  A series
    # already whole on disk in one mode is reused as it is, whatever was
    # asked (refetching it would buy nothing).  Otherwise the named mode
    # wins, and a kept file of the other mode is set aside (nothing is
    # deleted) and fetched again in it.  Under 'auto' nothing is
    # settled: a whole object and a current subset carry the same
    # records for hrrr_grib2_bridge (the subset takes the optional
    # surface records wherever the index lists them), so hours may land
    # in either mode, each file as its own index allows.
    for kind in ("atmosphere", "soil"):
        if force:
            series_mode[kind] = lands
            continue
        members = [(hour, dest_name) for hour, of_kind, _source, dest_name,
                   _product in products if of_kind == kind]
        if kind == "atmosphere":
            for hour, dest_name in members:
                if not ((out / dest_name).exists()
                        and kept_mode(dest_name, kind) == "idx-subset"):
                    continue
                claimed = prior_entries.get(dest_name)
                if claimed is not None:
                    current = "optional_surface" in claimed
                else:
                    # A half-fetched hour no receipt claims: its census
                    # against the index kept beside it says whether it
                    # took the optional records that index lists.
                    listed = kept_index_optional(dest_name)
                    try:
                        census = count_grib2_messages(out / dest_name)
                    except ValueError:
                        census = None
                    current = (listed is not None and census
                               == expected_counts[kind] + len(listed))
                if not current:
                    # Fetched before the subset took the optional surface
                    # records: next to a whole file or a current subset
                    # its hour selects a different record set, which the
                    # bridge refuses (and alone it starts PBLH at zero).
                    _quarantine_rejected(
                        out / dest_name, progress,
                        f"hrrr f{hour:02d} {kind} (subset fetched before "
                        "subsets took PBLH/LCDC/MCDC/HCDC; fetching it "
                        "again with them)")
        present = {dest_name: (hour, kept_mode(dest_name, kind))
                   for hour, dest_name in members
                   if (out / dest_name).exists()}
        found = {landed for _hour, landed in present.values()
                 if landed is not None}
        settled = (len(present) == len(members) and len(found) == 1
                   and all(name in prior_entries for name in present))
        if settled:
            target = next(iter(found))
        else:
            target = lands
        series_mode[kind] = target
        if target is None:
            continue
        for dest_name, (hour, landed) in present.items():
            if landed is not None and landed != target:
                _quarantine_rejected(
                    out / dest_name, progress,
                    f"hrrr f{hour:02d} {kind} ({landed} file kept from an "
                    f"earlier fetch; this series is {target}, and "
                    "hrrr_grib2_bridge refuses a series whose hours carry "
                    f"different record sets; fetching it again as {target})")

    def hour_checkpoint(index: int, entry: dict) -> None:
        # Checkpoint per completed hour, as GFS already did.  Publishing
        # only after every product meant an ordinary SIGKILL after many
        # good hours left a fresh output with valid payloads and no
        # receipt at all, which the front door then refused -- verified
        # data, unusable, and nothing to resume from.  An hour is
        # complete when its SECOND product (soil) has verified; entries
        # arrive in submission order either way, so the claimed prefix
        # is contiguous by construction.
        files.append(entry)
        hour, kind = products[index][0], products[index][1]
        if kind == "soil":
            complete_hours.append(hour)
            manifest = publish_manifest(tuple(complete_hours))
            if on_hour_ready is not None:
                # Both files passed the normal inventory and digest bars,
                # and the ordered prefix is durable. A preparation may now
                # admit this lead while the remaining files still download.
                on_hour_ready(hour, manifest)

    if wait:
        # Live-cycle mode follows publication by definition: each
        # product is polled until the mirrors serve it, then fetched.
        # The polling is the pacing, so the transfers stay serial and
        # in publication order regardless of the pool default.
        for index, (hour, kind, source_name, dest_name,
                    product) in enumerate(products):
            hour_checkpoint(index, transfer_product(
                hour, kind, source_name, dest_name, product))
    else:
        # The Rust fetch bridge holds each object in memory until it is
        # whole, so its in-flight bytes come from its own progress lines
        # (`byte_relay` above); `path` is what the Python transport's
        # growing file and every landed file are counted by.
        monitor = progress_mod.TransferMonitor("fetch hrrr")
        try:
            _entries, receipt = fetch_pool.run_transfers(
                [fetch_pool.TransferJob(
                    name=dest_name,
                    url=hrrr_object_url(cycle, hour, product,
                                        transport=candidates[0]),
                    # Checked before anything moves; one that fails the
                    # check is fetched again from this host, so the job
                    # keeps its url and stays under the host's cap.
                    on_disk=(out / dest_name).exists(),
                    token=f"f{hour:02d} {kind}", path=out / dest_name,
                    action=functools.partial(
                        transfer_product, hour, kind, source_name,
                        dest_name, product))
                 for hour, kind, source_name, dest_name, product in products],
                workers=file_workers, on_admitted=hour_checkpoint,
                monitor=monitor)
        finally:
            monitor.close()
        pool_summary.update(receipt)
    return publish_manifest(hours)


# ---------------------------------------------------------------------------
# ERA5: cdsapi template + user-file validation
# ---------------------------------------------------------------------------

def _era5_times(cycle: datetime, hours: int,
                cadence: int) -> tuple[datetime, ...]:
    from woof.era5_member import validate_selection

    validate_selection(cadence=cadence)
    if (isinstance(hours, bool) or not isinstance(hours, int)
            or hours < 0 or hours % cadence):
        raise ValueError(
            f"--hours must be a nonnegative integer multiple of the {cadence} h "
            "cadence")
    if not isinstance(cycle, datetime):
        raise ValueError("ERA5 cycle must be a UTC date and hour")
    if cycle.tzinfo is not None:
        cycle = cycle.astimezone(timezone.utc).replace(tzinfo=None)
    if cycle.minute or cycle.second or cycle.microsecond:
        raise ValueError("ERA5 cycle must fall on an exact UTC hour")
    return tuple(cycle + timedelta(hours=lead)
                 for lead in range(0, hours + 1, cadence))


def era5_request_template(*, cycle: datetime, hours: int, area: Area,
                          cadence: int = 6, out: Path | None = None) -> dict:
    """Exact CDS pressure/surface requests for the requested valid times.

    CDS combines every date with every clock time in one request. Split by
    day so partial days, and cadences that do not divide 24, cannot acquire
    extra times. Each response has its own target, bound to ``out`` rather
    than the retrieval process's working directory.
    """

    times = _era5_times(cycle, hours, cadence)
    by_day: dict[str, list[str]] = {}
    for moment in times:
        by_day.setdefault(moment.strftime("%Y-%m-%d"), []).append(
            moment.strftime("%H:%M"))
    shared = {
        "product_type": "reanalysis",
        "data_format": "grib",
        "area": area.as_cds(),
    }
    pressure = dict(shared)
    pressure["variable"] = [
        "geopotential", "temperature", "u_component_of_wind",
        "v_component_of_wind", "relative_humidity",
    ]
    pressure["pressure_level"] = [
        str(level) for level in ERA5_PRESSURE_LEVELS_HPA]
    single = dict(shared)
    single["variable"] = [
        "geopotential", "surface_pressure", "mean_sea_level_pressure",
        "10m_u_component_of_wind", "10m_v_component_of_wind",
        "2m_temperature", "2m_dewpoint_temperature", "land_sea_mask",
        "skin_temperature", "sea_surface_temperature", "sea_ice_cover",
        "lake_mix_layer_temperature", "lake_ice_temperature", "lake_ice_depth",
        "snow_depth",
        "soil_temperature_level_1", "soil_temperature_level_2",
        "soil_temperature_level_3", "soil_temperature_level_4",
        "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
        "volumetric_soil_water_layer_3", "volumetric_soil_water_layer_4",
    ]
    def target(name: str) -> str:
        return name if out is None else str((out / name).resolve())

    requests = []
    for day, clock in by_day.items():
        # Keep the existing names for a one-day request. More than one day
        # needs distinct files so retrieval cannot overwrite an earlier day.
        suffix = "" if len(by_day) == 1 else "-" + day
        for dataset, leaf, selection in (
                ("reanalysis-era5-pressure-levels", "pressure", pressure),
                ("reanalysis-era5-single-levels", "single", single)):
            requests.append({"dataset": dataset,
                "target": target(f"era5-{leaf}{suffix}.grib"),
                "request": dict(selection, date=[day], time=clock)})
    combined = target(ERA5_COMBINED_NAME)
    return {
        "schema": "gpuwm-era5-cds-request-v1",
        "requires": "CDS account + ~/.cdsapirc key; pip install cdsapi",
        "requests": requests,
        "combine": ("concatenate all GRIB1 targets into one file "
                    "(byte concatenation preserves every message): "
                    f"{combined}"),
        "combine_target": combined,
        "area_requested": area.as_cds(),
        "validate": ("woof fetch --source era5 --validate "
                     f"{combined} --area "
                     f"{area.lat_south:g},{_wrap_lon(area.lon_west):g},"
                     f"{area.lat_north:g},{_wrap_lon_east(area.lon_east):g}"),
    }


#: Where cdsapi reads the personal CDS key, exactly as
#: :data:`ERA5_INSTRUCTIONS` step 1 tells the reader to write it.  Named
#: once so the instruction and the check cannot come to disagree about
#: which file they are talking about.
CDSAPIRC_NAME = ".cdsapirc"


def cds_credentials_path() -> Path:
    """The ``~/.cdsapirc`` cdsapi would read on this machine."""

    override = os.environ.get("CDSAPI_RC")
    return Path(override) if override is not None else Path.home() / CDSAPIRC_NAME


def cds_credentials_present() -> bool:
    """Is a CDS key file in place?  Existence only -- never read.

    The ERA5 route is the one front door whose first step happens
    outside woof entirely, and the failure mode it produces is a
    cdsapi exception several commands later with nothing pointing back
    at the missing file.  Answering "is it there" costs a ``stat`` and
    lets the wizard say so while the reader is still deciding what to
    run next.

    Presence, not validity: a key's correctness is the CDS server's
    verdict to give, and guessing at its format here would produce
    confident wrong advice about a file this project does not own.
    """

    try:
        return bool(os.environ.get("CDSAPI_KEY")) or cds_credentials_path().is_file()
    except OSError:
        return False


def wsl_path(path: Path) -> str:
    """``C:\\dir\\file`` as the ``/mnt/c/dir/file`` WSL can open.

    A Windows path handed to an interpreter running inside WSL names
    nothing: the retrieval silently writes a file called ``C:\\...`` in
    whatever directory it started in, or fails to open the request at
    all.  Paths that are already POSIX come back unchanged.
    """

    text = str(path).replace("\\", "/")
    if len(text) > 1 and text[1] == ":" and text[0].isalpha():
        return f"/mnt/{text[0].lower()}{text[2:]}"
    return text


#: The retrieval, as a file rather than a snippet.  It resolves every
#: path from its OWN location, so it produces the same files whether it
#: is run by this box's Python, by a Linux Python, or by a WSL Python
#: that sees the same directory under a different name.
ERA5_RETRIEVE_SCRIPT = '''\
"""Retrieve the ERA5 request `woof fetch --source era5` wrote beside me.

Run with any Python that has cdsapi installed and a CDS key in ITS home
directory (on Windows the retrieval is commonly run inside WSL, whose
home is not the Windows one).  Every file lands in this script's own
directory, so the working directory does not matter.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, {request_name!r})
COMBINED = os.path.join(HERE, {combined_name!r})


def _beside_me(target):
    """The declared target's leaf, in this script's directory.

    The request records absolute targets so a hand-run retrieval cannot
    scatter them; taking the leaf here keeps the script correct when the
    same directory is reached under another name.
    """
    return os.path.join(HERE, target.replace("\\\\", "/").rsplit("/", 1)[-1])


def main():
    import cdsapi

    with open(SPEC, encoding="utf-8") as stream:
        spec = json.load(stream)
    client = cdsapi.Client()
    parts = []
    for item in spec["requests"]:
        target = _beside_me(item["target"])
        print("retrieve", item["dataset"], "->", target, flush=True)
        client.retrieve(item["dataset"], item["request"], target)
        parts.append(target)
    # GRIB is a concatenation of self-delimiting messages, so joining the
    # two retrievals byte for byte preserves every one of them.
    with open(COMBINED, "wb") as combined:
        for part in parts:
            with open(part, "rb") as stream:
                combined.write(stream.read())
    print("wrote", COMBINED, flush=True)
    print("now run:", spec["validate"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


ERA5_INSTRUCTIONS = """\
ERA5 acquisition is manual: the Copernicus CDS API requires a personal
account and key, which woof will not embed.

1. Create an account at https://cds.climate.copernicus.eu and write your
   key to ~/.cdsapirc as documented there.  "~" is the home directory of
   the interpreter that runs step 3, which on Windows is usually a WSL
   python3 -- then the key belongs in the WSL home, not this box's.
2. pip install cdsapi   (for that same interpreter)
3. Retrieve and combine, in one command:
       {retrieve_command}
{wsl_note}\
4. Validate the result -- including that it covers the box you asked for:
       {validate_command}
"""


def era5_retrieve_commands(script: Path) -> tuple[str, str | None]:
    """The command that runs the retrieval, plus the WSL form on Windows."""

    native = f"python {script}"
    if os.name != "nt":
        return native, None
    posix = wsl_path(script)
    return native, f'wsl sh -c "python3 -u {posix}"'


def write_era5_request(*, cycle: datetime, hours: int, area: Area,
                       out: Path, cadence: int = 6,
                       progress=print) -> Path:
    template = era5_request_template(
        cycle=cycle, hours=hours, area=area, cadence=cadence, out=out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ERA5_REQUEST_NAME
    _atomic_write_text(
        path, json.dumps(template, indent=2, sort_keys=True) + "\n")
    script = out / ERA5_RETRIEVE_NAME
    _atomic_write_text(script, ERA5_RETRIEVE_SCRIPT.format(
        request_name=ERA5_REQUEST_NAME, combined_name=ERA5_COMBINED_NAME))
    native, under_wsl = era5_retrieve_commands(script)
    progress(f"fetch era5: wrote {path}")
    progress(f"fetch era5: wrote {script} (runs the retrieval)")
    progress(ERA5_INSTRUCTIONS.format(
        retrieve_command=native,
        wsl_note=("" if under_wsl is None else
                  "   or, if cdsapi and your key live in WSL:\n"
                  f"       {under_wsl}\n"),
        validate_command=template["validate"]))
    return path


@dataclass(frozen=True)
class Grib1Grid:
    """The geographic identity of one regular lat/lon GRIB1 grid.

    Read from the Grid Definition Section, which is where the answer to
    "is this file over my domain?" lives.  Only data representation type
    0 (equidistant cylindrical) is produced here; that is what the CDS
    serves for ERA5, and a grid of any other type is reported as absent
    rather than guessed at.
    """

    ni: int
    nj: int
    lat_first: float
    lon_first: float
    lat_last: float
    lon_last: float
    #: Declared increments in degrees, or ``None`` when the GDS says
    #: they are not given (octet 17 bit 1 clear, or 0xFFFF).
    di_deg: float | None
    dj_deg: float | None

    @property
    def lat_south(self) -> float:
        return min(self.lat_first, self.lat_last)

    @property
    def lat_north(self) -> float:
        return max(self.lat_first, self.lat_last)

    @property
    def lon_west(self) -> float:
        """West edge in the signed convention, as the GDS scans it."""

        return _wrap_lon(self.lon_first)

    @property
    def longitude_span_degrees(self) -> float:
        """Eastward width from the first to the last meridian."""

        span = (self.lon_last - self.lon_first) % 360.0
        if span == 0.0 and self.ni > 1:
            return 360.0
        return span

    @property
    def lon_east(self) -> float:
        return _wrap_lon_east(self.lon_west + self.longitude_span_degrees)

    @property
    def lon_step(self) -> float:
        """Longitude increment, declared or derived from the corners."""

        if self.di_deg:
            return self.di_deg
        if self.ni > 1:
            return self.longitude_span_degrees / (self.ni - 1)
        return 0.0

    @property
    def lat_step(self) -> float:
        if self.dj_deg:
            return self.dj_deg
        if self.nj > 1:
            return abs(self.lat_north - self.lat_south) / (self.nj - 1)
        return 0.0

    def describe(self) -> str:
        step = (f", {self.lon_step:g} x {self.lat_step:g} deg"
                if self.lon_step and self.lat_step else "")
        return (f"grid {self.ni}x{self.nj}{step}, "
                f"lat [{self.lat_south:.2f}, {self.lat_north:.2f}] "
                f"lon [{self.lon_west:.2f}, {self.lon_east:.2f}]")


@dataclass(frozen=True)
class Grib1Record:
    """Transport-level identity of one GRIB1 message."""

    parameter: int
    level_type: int
    level: int
    valid_time: datetime
    #: The message's own grid, when it carries a GDS this reader
    #: understands.  ``None`` means the geography was not stated in a
    #: form that can be read, never that the message is ungridded.
    grid: Grib1Grid | None = None
    table_version: int | None = None
    center: int | None = None
    grid_definition_sha256: str | None = None


_GRIB1_TIME_UNITS = {
    0: timedelta(minutes=1), 1: timedelta(hours=1), 2: timedelta(days=1),
    10: timedelta(hours=3), 11: timedelta(hours=6),
    12: timedelta(hours=12), 254: timedelta(seconds=1),
}


def _grib1_signed_millideg(raw: bytes) -> float:
    """GRIB1 sign-magnitude millidegrees, as degrees."""

    value = int.from_bytes(raw, "big")
    if value & 0x800000:
        return -float(value & 0x7FFFFF) / 1000.0
    return float(value) / 1000.0


def read_grib1_grid(gds: bytes) -> Grib1Grid | None:
    """The geographic identity in a GRIB1 Grid Definition Section.

    ``None`` for anything that is not a regular lat/lon grid, or for a
    section too short to read: a guessed extent is worse than no extent,
    because a reader would act on it.
    """

    if len(gds) < 32 or gds[5] != 0:      # octet 6: data representation
        return None
    ni = int.from_bytes(gds[6:8], "big")
    nj = int.from_bytes(gds[8:10], "big")
    if ni in (0, 0xFFFF) or nj in (0, 0xFFFF):
        return None

    def increment(raw: bytes) -> float | None:
        value = int.from_bytes(raw, "big")
        if value in (0, 0xFFFF) or not gds[16] & 0x80:
            return None
        return value / 1000.0

    return Grib1Grid(
        ni=ni, nj=nj,
        lat_first=_grib1_signed_millideg(gds[10:13]),
        lon_first=_grib1_signed_millideg(gds[13:16]),
        lat_last=_grib1_signed_millideg(gds[17:20]),
        lon_last=_grib1_signed_millideg(gds[20:23]),
        di_deg=increment(gds[23:25]), dj_deg=increment(gds[25:27]))


def read_grib1_records(path: Path) -> tuple[Grib1Record, ...]:
    """Envelope-validate ``path`` and read each message's PDS identity.

    Reuses :func:`woof.ingest.grib.inspect_grib1_envelopes` for the
    strict transport walk, then reads only Product Definition Section
    header bytes -- parameter, level, and reference/valid time -- plus
    the Grid Definition Section's corners when the PDS declares one.  No
    scientific payload is decoded.
    """

    from woof.ingest.grib import inspect_grib1_envelopes

    envelopes = inspect_grib1_envelopes(path)
    records: list[Grib1Record] = []
    with path.open("rb") as stream:
        for envelope in envelopes:
            stream.seek(envelope.offset + 8)
            pds = stream.read(28)
            if len(pds) < 28:
                raise ValueError(
                    f"GRIB1 message {envelope.index} in {path} has a "
                    "truncated PDS")
            parameter = pds[8]
            level_type = pds[9]
            level = int.from_bytes(pds[10:12], "big")
            year_of_century, month, day, hour, minute = pds[12:17]
            unit, p1, p2, time_range = pds[17], pds[18], pds[19], pds[20]
            century = pds[24]
            year = (century - 1) * 100 + year_of_century
            reference = datetime(year, month, day, hour, minute)
            if unit not in _GRIB1_TIME_UNITS:
                raise ValueError(
                    f"unsupported GRIB1 forecast time unit {unit} in "
                    f"{path} message {envelope.index}")
            if time_range in (0, 1):
                lead = p1 if time_range == 0 else 0
            elif time_range == 10:
                lead = (p1 << 8) | p2
            else:
                raise ValueError(
                    f"unsupported GRIB1 time range indicator {time_range} "
                    f"in {path} message {envelope.index}; ERA5 analyses "
                    "are instantaneous")
            valid_time = reference + lead * _GRIB1_TIME_UNITS[unit]
            grid = None
            grid_definition_sha256 = None
            if pds[7] & 0x80:      # octet 8 bit 1: a GDS follows the PDS
                pds_length = int.from_bytes(pds[0:3], "big")
                stream.seek(envelope.offset + 8 + pds_length)
                header = stream.read(3)
                if len(header) == 3:
                    gds_length = int.from_bytes(header, "big")
                    if 32 <= gds_length <= envelope.length:
                        grid_bytes = header + stream.read(gds_length - 3)
                        grid = read_grib1_grid(grid_bytes)
                        grid_definition_sha256 = hashlib.sha256(grid_bytes).hexdigest()
            records.append(
                Grib1Record(parameter, level_type, level, valid_time, grid,
                            table_version=pds[3], center=pds[4],
                            grid_definition_sha256=grid_definition_sha256))
    return tuple(records)


@dataclass(frozen=True)
class Era5ValidationReport:
    """Census of a user-supplied ERA5 GRIB1 set vs ingest expectations."""

    failures: tuple[str, ...]
    checks: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.failures

    def format(self) -> str:
        lines = [f"era5 validation: "
                 f"{'PASS' if self.ok else 'FAIL'}"]
        lines.extend(f"  ok: {check}" for check in self.checks)
        lines.extend(f"  FAIL: {failure}" for failure in self.failures)
        return "\n".join(lines)


def _grid_coverage_failures(
        grid: Grib1Grid, expected: Area) -> list[str]:
    """Edges where the delivered grid falls short of the requested box.

    The tolerance is ONE grid cell per edge, and it is the provider's
    rounding, not slack: the CDS snaps a requested area onto its native
    grid, which can move an edge inward by up to one increment.  A
    shortfall larger than that is a different box -- the mistake this
    check exists to catch, because such a file validates, prepares, and
    only refuses at ``woof check`` or mid-run.
    """

    failures: list[str] = []
    lat_tolerance = grid.lat_step or 0.0
    lon_tolerance = grid.lon_step or 0.0
    for edge, shortfall, tolerance, got, want in (
            ("south", grid.lat_south - expected.lat_south, lat_tolerance,
             grid.lat_south, expected.lat_south),
            ("north", expected.lat_north - grid.lat_north, lat_tolerance,
             grid.lat_north, expected.lat_north)):
        if shortfall > tolerance + 1e-9:
            failures.append(
                f"the delivered grid stops {shortfall:.2f} deg short of the "
                f"requested {edge} edge (grid {got:.2f}, requested "
                f"{want:.2f}); more than the {tolerance:g} deg the "
                "provider's grid snap can account for")
    grid_span = grid.longitude_span_degrees + (grid.lon_step or 0.0)
    if grid_span < 360.0 - 1e-9:
        west_gap = ((grid.lon_west - expected.lon_west) + 180.0) % 360.0 - 180.0
        east_gap = ((_wrap_lon_east(expected.lon_east) - grid.lon_east)
                    + 180.0) % 360.0 - 180.0
        for edge, shortfall, got, want in (
                ("west", west_gap, grid.lon_west, _wrap_lon(expected.lon_west)),
                ("east", east_gap, grid.lon_east,
                 _wrap_lon_east(expected.lon_east))):
            if shortfall > lon_tolerance + 1e-9:
                failures.append(
                    f"the delivered grid stops {shortfall:.2f} deg short of "
                    f"the requested {edge} edge (grid {got:.2f}, requested "
                    f"{want:.2f}); more than the {lon_tolerance:g} deg the "
                    "provider's grid snap can account for")
    return failures


def validate_era5_files(
        paths: tuple[Path, ...], *,
        expected_times: tuple[datetime, ...] | None = None,
        expected_area: Area | None = None,
) -> Era5ValidationReport:
    """Validate a user-supplied ERA5 GRIB1 file set for woof ingest.

    Covers: strict GRIB1 transport envelopes (edition, declared lengths,
    ``7777`` terminators, exact EOF coverage), the required
    pressure-level and surface parameter inventory at every valid time,
    identical pressure-level ladders across variables and times, soil
    encodings at level type 112 or 1, invariant orography presence, the
    delivered geographic extent (one grid for the whole set, and -- when
    the caller supplies ``expected_area`` -- that it covers the box that
    was asked for), and (when the caller supplies ``expected_times``)
    valid-time coverage.  It does not decode data values -- the Rust
    GRIB1 bridge re-validates and decodes at ingest time.
    """

    failures: list[str] = []
    checks: list[str] = []
    records: list[Grib1Record] = []
    for path in paths:
        if not path.is_file():
            failures.append(f"missing input file {path}")
            continue
        try:
            found = read_grib1_records(path)
        except ValueError as error:
            failures.append(str(error))
            continue
        checks.append(f"{path.name}: {len(found)} valid GRIB1 envelopes")
        records.extend(found)
    if failures:
        return Era5ValidationReport(tuple(failures), tuple(checks))

    times = sorted({record.valid_time for record in records})
    if not times:
        return Era5ValidationReport(
            ("no GRIB1 messages found in the supplied files",),
            tuple(checks))
    checks.append(
        f"{len(times)} valid times {times[0].isoformat()} .. "
        f"{times[-1].isoformat()}")

    # Geographic extent.  A census that never says WHERE the bytes are
    # cannot catch the most likely retrieval mistake -- a file cropped to
    # a different box than the domain needs -- and that mistake survives
    # preparation and costs a whole run.
    grids = {record.grid for record in records if record.grid is not None}
    if not grids:
        checks.append(
            "geographic extent: not stated by these messages (no readable "
            "lat/lon grid definition), so the box was NOT checked")
    elif len(grids) > 1:
        described = sorted(grid.describe() for grid in grids)
        failures.append(
            f"the supplied messages carry {len(grids)} different grids "
            f"({'; '.join(described)}); ERA5 for one domain is retrieved "
            "with ONE area, so two grids means the two CDS requests were "
            "not made with the same one and the fields cannot be composed "
            "onto a single domain")
    else:
        grid = next(iter(grids))
        checks.append(grid.describe())
        if expected_area is None:
            checks.append(
                "note: the extent above was not checked against any "
                "requested box -- pass --area (the same one `woof domain` "
                "printed) to have this check it")
        else:
            shortfalls = _grid_coverage_failures(grid, expected_area)
            if shortfalls:
                failures.extend(shortfalls)
            else:
                checks.append(
                    "the delivered grid covers the requested box "
                    f"lat [{expected_area.lat_south:.2f}, "
                    f"{expected_area.lat_north:.2f}] "
                    f"lon [{_wrap_lon(expected_area.lon_west):.2f}, "
                    f"{_wrap_lon_east(expected_area.lon_east):.2f}]")

    if expected_times is not None:
        missing = sorted(set(expected_times) - set(times))
        if missing:
            failures.append(
                "missing valid times: "
                + ", ".join(when.isoformat() for when in missing))
        else:
            checks.append("requested cycle/hours window fully covered")

    # Pressure-level census: required variables at identical ladders.
    ladders: dict[tuple[int, datetime], set[int]] = {}
    for record in records:
        if record.level_type == 100:
            ladders.setdefault(
                (record.parameter, record.valid_time), set()
            ).add(record.level)
    reference_ladder: set[int] | None = None
    for parameter, short in sorted(ERA5_REQUIRED_PRESSURE.items()):
        per_time = [ladders.get((parameter, when)) for when in times]
        if any(levels is None for levels in per_time):
            failures.append(
                f"pressure-level {short} (GRIB1 parameter {parameter}) is "
                "absent at one or more valid times")
            continue
        if len({frozenset(levels) for levels in per_time}) != 1:
            failures.append(
                f"pressure-level {short} ladder differs across valid times")
            continue
        if reference_ladder is None:
            reference_ladder = per_time[0]
        elif per_time[0] != reference_ladder:
            failures.append(
                f"pressure-level {short} ladder differs from the other "
                "variables")
    if reference_ladder is not None:
        checks.append(
            f"pressure ladder: {len(reference_ladder)} levels "
            f"{min(reference_ladder)}..{max(reference_ladder)} hPa, "
            "identical across required variables and times")

    # Surface census.  Soil layers may arrive as level type 112 (native
    # CDS) or 1 (CDO-normalized); everything else must be level type 1.
    surface: dict[int, set[datetime]] = {}
    invariant_orography = False
    for record in records:
        accepted_types = (
            (1, 112) if record.parameter in ERA5_SOIL_PARAMETERS else (1,))
        if record.level_type not in accepted_types:
            continue
        surface.setdefault(record.parameter, set()).add(record.valid_time)
        if (record.parameter == ERA5_OROGRAPHY_PARAMETER
                and record.level_type == 1):
            invariant_orography = True
    for parameter, short in sorted(ERA5_REQUIRED_SURFACE.items()):
        present = surface.get(parameter, set())
        if not present >= set(times):
            failures.append(
                f"surface {short} (GRIB1 parameter {parameter}) is absent "
                "at one or more valid times")
    if not invariant_orography:
        failures.append(
            "invariant geopotential (parameter 129 at the surface) is "
            "absent: request 'geopotential' in the single-levels dataset, "
            "or declare a per-domain source-orography supplement in "
            "[case_data]")
    optional = sorted(
        short for parameter, short in ERA5_OPTIONAL_SURFACE.items()
        if parameter in surface)
    if optional:
        checks.append(f"optional fields present: {', '.join(optional)}")
    required_present = sorted(
        short for parameter, short in ERA5_REQUIRED_SURFACE.items()
        if surface.get(parameter, set()) >= set(times))
    if required_present:
        checks.append(
            f"required surface fields at every time: "
            f"{', '.join(required_present)}")
    return Era5ValidationReport(tuple(failures), tuple(checks))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_area(args) -> Area | None:
    if args.area is not None and args.point is not None:
        raise ValueError("--area and --point are mutually exclusive")
    if args.area is not None:
        if args.radius_km is not None:
            raise ValueError("--radius-km belongs to --point, not --area")
        return parse_area(args.area)
    if args.point is not None:
        if args.radius_km is None:
            raise ValueError("--point requires --radius-km")
        return area_from_point(args.point, args.radius_km)
    if args.radius_km is not None:
        raise ValueError("--radius-km requires --point")
    return None


#: Flags that belong to one of the four hand-written transports and mean
#: nothing on a table route, each with the sentence that says why.
_LEGACY_ONLY_FLAGS = {
    "--validate": "the ERA5 manual-retrieval checker",
    "--p-top-pa": "the GFS/GDAS isobaric ladder",
    "--all-levels": "the GFS/GDAS isobaric ladder",
    "--engine": "the Rust range-GET backbone, which the GFS and HRRR "
                "whole-file routes drive",
    "--cache-dir": "the Rust backbone's disk cache",
    "--author-front-door-manifest": "the GFS front-door input manifest",
}


@functools.lru_cache(maxsize=32)
def _stream_capable(binary, identity):
    import subprocess
    try:
        result = subprocess.run([str(binary), "--help"], capture_output=True,
                                text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "rw_fetch object --url URL --output FILE" in result.stdout


_ROUTE_OBJECT_TRANSPORT = fetch_routes._download_object


def _route_stream_downloader(plan, args):
    """Use the Rust streaming door for a declared GRIB2 profile.

    Other formats retain their established transport. No source names or
    provider-specific URL algebra appear in this data path.
    """
    adapter = source_adapters.get_source_adapter(plan.source_id)
    # Preserve a caller's substituted transport, including offline replays.
    if fetch_routes._download_object is not _ROUTE_OBJECT_TRANSPORT:
        return None
    if "grib2" not in str(adapter.packaged_profile).lower():
        return None
    from woof import rustwx_fetch
    binary = rustwx_fetch.find_fetch_bin()
    if binary is None:
        return None
    # Older installed artifacts keep their former route until rebuilt.
    # The new object door is additive to the existing fetch ABI.
    if not _stream_capable(binary, _stat_identity(binary)):
        return None
    from woof.fetch_connections import max_connections
    if (max_connections() <= rustwx_fetch.LEGACY_IDLE_CONNECTIONS and
            not rustwx_fetch._idle_capable(str(binary), _stat_identity(binary))):
        # An older bridge can retain idle sockets across redirected hosts.
        # The generic Python transport can honor even a one-connection box.
        return None
    def download(url, dest, *, magic, opener=None):
        # The ahead probe can promote a mirror above the plan's first
        # endpoint. Size this transfer for its actual host, not that
        # earlier endpoint's (possibly one-stream) politeness limit.
        streams = fetch_pool.chunk_streams_per_file(args.fetch_workers,
            files=len(plan.objects), host=fetch_pool.host_key(url))
        if opener is not None or magic != "GRIB":
            return fetch_routes._download_object(url, dest, magic=magic,
                                                opener=opener, streams=streams)
        env = dict(os.environ)
        env["RUSTWX_DOWNLOAD_STREAMS"] = str(streams)
        dest = Path(dest)
        if dest.exists():
            _quarantine_rejected(dest, print,
                "verified reuse failed; prevents replacing changed input bytes")
        job = fetch_pool.current_job()
        relay = getattr(job, "byte_relay", None)
        entry = rustwx_fetch._run([str(binary), "object", "--url", url,
            "--output", str(dest)], what="object", out=dest.parent, env=env,
            on_progress=None if relay is None else relay())
        _remember_stream_facts(dest, entry)
        # Preparation binds the durable source manifest. Publication identity
        # and transfer clocks belong to the memo/ledger, not that content key.
        return {"name": dest.name, "bytes": entry["bytes"],
                "sha256": entry["sha256"], "url": url}

    # The route's own transport: a file row's record selection still applies.
    download.route_transport = True
    return download


def _route_fetch_main(args, source: str) -> int:
    """Fetch a table route with the same cycle/member/window used by its probe."""
    supplied = []
    for flag, owner in _LEGACY_ONLY_FLAGS.items():
        value = getattr(args, flag.lstrip("-").replace("-", "_"), None)
        if value is not None and value is not False and value != []:
            supplied.append((flag, owner))
    if supplied:
        raise ValueError(
            f"{', '.join(flag for flag, _ in supplied)}: --source {source} is a table-driven route.\n"
            f"  why: {supplied[0][0]} belongs to {supplied[0][1]}, which this route does not use; "
            "it transfers whole published objects and composes the declared profile inputs.")
    author_roles = ("bridge", "wps_namelist", "experiment_config", "static_input", "static_receipt", "manifest_out")
    extras = ["--" + key.replace("_", "-") for key in author_roles if getattr(args, key, None) is not None]
    if extras:
        raise ValueError(f"{', '.join(extras)} belong to --author-front-door-manifest; "
                         f"{source} writes its table-route preparation handoff instead")
    if args.fetch_workers is not None:
        fetch_pool.resolve_file_workers(args.fetch_workers)
    fetch_routes.resolve_mode(source, args.mode)
    if args.cycle is None or args.hours is None or args.out is None:
        raise ValueError(f"fetch --source {source} requires --cycle, --hours and --out")
    # Dry validation precedes publication probes, locks, directories and bytes.
    hints = {"source": source, "cycle": args.cycle, "hours": args.hours}
    for key in ("cadence", "forecast_start_hour", "member", "area", "point", "radius_km"):
        value = getattr(args, key, None)
        if value is not None:
            hints[key] = value
    validate_fetch_hints(hints, source=COMMAND_LINE_HINTS)
    route = fetch_routes.route_for(source)
    if args.transport is not None:
        route.host(args.transport)
    start = 0 if args.forecast_start_hour is None else args.forecast_start_hour
    options = dict(cadence=args.cadence, start_hour=start, member=args.member, transport=args.transport)
    last = start + args.hours
    as_posted = requested_as_posted(args)
    if args.cycle == "latest":
        cycle = resolve_latest_cycle(source, last, as_posted=as_posted, **options)
        evidence = ("publication probe confirmed" if cycle_is_probeable(source)
                    else "declared publication delay; not probed")
        if as_posted and _startable_rule_applies(source):
            print(f"fetch {source}: latest startable cycle is "
                  f"{cycle:%Y-%m-%dT%H}Z (its start needs are posted; later "
                  "leads are fetched as they post)")
        else:
            print(f"fetch {source}: latest complete cycle is {cycle:%Y-%m-%dT%H}Z ({evidence})")
    else:
        cycle = parse_cycle(args.cycle, source)
    plan = fetch_routes.resolve_request(source, cycle=cycle, hours=args.hours,
        cadence=args.cadence, start_hour=start, host=args.transport, member=args.member, out=args.out)
    with _fetch_out(args.out):
        if not args.force_refetch:
            fetch_routes.check_prior_request(args.out, plan)
        loop = None
        if as_posted and (args.force_refetch
                          or not fetch_routes.request_cached(plan, args.out)):
            loop = _posting_loop(source, cycle, args, leads=plan.leads,
                                 plan=plan, transport=args.transport)
        if loop is not None:
            return _route_fetch_as_posted(plan, args, loop, last=last,
                                          options=options)
        # A folder that already holds every object of this exact request,
        # byte-verified, needs nothing from the provider: it stays usable
        # offline and after the cycle has left the provider's retention.
        # Anything else asks whether the named cycle is published before a
        # byte moves, as the latest path did while resolving it.
        if args.cycle != "latest" and (
                args.force_refetch
                or not fetch_routes.request_cached(plan, args.out)):
            require_published_cycle(source, cycle, last, **options)
        # A declared donor is a second download of the same request.
        # Resolving latest already chose a cycle whose donor was
        # published, but publication is asked again here on either path
        # before the first byte moves: a named cycle was never asked, and
        # a cached donor folder is checked for completeness instead.
        for donor in plan.donors:
            if args.force_refetch or not _donor_cached(donor, args):
                try:
                    require_published_cycle(
                        donor.source, donor.cycle, max(donor.leads))
                except RuntimeError as error:
                    raise RuntimeError(
                        f"--source {source} takes part of its start from the "
                        f"{donor.source.upper()} analysis of its own cycle, "
                        f"and {error}") from None
        donor_files = _fetch_route_donors(plan, args)
        bound = False

        def bind_ready(lead, entries, composed):
            nonlocal bound
            if not bound:
                fetch_routes.write_handoff(plan, args.out, donor_files=donor_files)
                bound = True

        def bind_completed(index, entry):
            # Probes also use the pool, but carry no payload digest.
            if entry.get("sha256"):
                bind_ready(None, None, None)

        # Lead callbacks compose and publish verified prefixes while later
        # inputs download. The handoff carries planned paths; complete=true
        # remains the preparation barrier for an ordinary whole-window run.
        with fetch_pool.completion_hook(bind_completed):
            fetch_routes.run_plan(plan, out=args.out, force=args.force_refetch,
                file_workers=args.fetch_workers, on_lead=bind_ready,
                downloader=_route_stream_downloader(plan, args))
        fetch_routes.write_handoff(plan, args.out, donor_files=donor_files)
    print(f"fetch {source}: manifest {args.out / fetch_routes.MANIFEST_NAME}")
    for line in fetch_routes.handoff_lines(plan, args.out):
        print(line)
    return 0


def _route_fetch_as_posted(plan, args, loop, *, last: int, options: dict) -> int:
    """A table route's window as its source posts it (DESIGN A136 2.3).

    The schedule goes out first, then the start needs (the first leads,
    their invariants and step-0 objects, a same-cycle donor) are waited
    for, the donor is fetched with them, and each lead moves the moment
    one host holds all of it, its marker published in lead order.  A
    window whose final lead is already posted (a late launch, a past
    cycle) moves as one request under the host caps, with no per-lead
    asking, and still publishes every marker in lead order.
    """

    from woof import fetch_as_posted

    source = plan.source_id
    if args.force_refetch:
        fetch_routes._quarantine(args.out, print)
    loop.start()
    # The question a whole-cycle fetch asks first, through the same
    # function: a window whose final lead is out needs no waiting.
    whole = (False if (fetch_endpoints.policy_uses_aws(source)
                       or any(fetch_endpoints.policy_uses_aws(donor.source)
                              for donor in plan.donors)) else
             _window_posted(loop.window, lambda said: require_published_cycle(
                 source, plan.cycle, last, progress=said, **options)))
    if whole:
        for donor in plan.donors:
            try:
                require_published_cycle(donor.source, donor.cycle,
                                        max(donor.leads))
            except RuntimeError as error:
                if _donor_still_due(loop, donor.source):
                    # The donor is a start need (DESIGN A136 1b, 3.3): a
                    # donor-gated window whose own leads are all out
                    # starts when its donor posts, as its readiness
                    # answer (75, waiting) tells the site to launch it.
                    # Refusing here skipped the cycle the site launched.
                    loop.wait_start()
                    break
                raise RuntimeError(
                    f"--source {source} takes part of its start from the "
                    f"{donor.source.upper()} analysis of its own cycle, "
                    f"and {error}") from None
    else:
        loop.wait_start()
    try:
        donor_files = _fetch_route_donors(plan, args)
        # The handoff goes out once the start needs and donors are in,
        # before later leads move, so preparation can run beside the fetch.
        fetch_routes.write_handoff(plan, args.out, donor_files=donor_files,
                                   posting=loop.folder)

        def published(lead, entries, composed):
            loop.publish(lead, fetch_as_posted.route_objects(entries),
                         composed=fetch_as_posted.composed_objects(composed))

        fetch_routes.run_plan(plan, out=args.out, force=False,
                              file_workers=args.fetch_workers,
                              downloader=_route_stream_downloader(plan, args),
                              lead_gate=None if whole else loop,
                              on_lead=published)
    finally:
        loop.close()
    fetch_routes.write_handoff(plan, args.out, donor_files=donor_files,
                               posting=loop.folder)
    print(f"fetch {source}: manifest {args.out / fetch_routes.MANIFEST_NAME}")
    for line in fetch_routes.handoff_lines(plan, args.out):
        print(line)
    return 0


def _donor_still_due(loop, donor_source: str) -> bool:
    """Whether a donor start need not posted yet is still within its budget.

    Before its late time a donor that is not posted is waited for (the
    readiness answer says ``waiting``); past it the donor will not appear
    by waiting, and the whole-cycle refusal, which names the newest donor
    cycle that is posted, stands.
    """

    now = loop.now()
    return any(need.role == f"donor:{donor_source}"
               and need.late_at is not None and now < need.late_at
               for need in loop.needs)


def _route_donor_out(out: Path, donor) -> Path:
    """The subdirectory a route's declared donor is fetched into."""
    return Path(out) / f"donor-{donor.source}"


def _donor_is_table_route(donor) -> bool:
    """Whether a declared donor is itself a row of the fetch-route table."""
    return (donor.source not in GFS_CONTAINER_SOURCES
            and fetch_routes.canonical_source(donor.source)
            in fetch_routes.table_route_sources())


def _donor_route_plan(donor, args):
    """The donor's own table-route plan: its source, its cycle, its leads.

    Refuses by name when the donor has neither a table row nor the
    GFS-container ladder, because then nothing in this door can move
    its bytes and the handoff would name a supplement nobody fetched.
    """
    if not _donor_is_table_route(donor):
        raise ValueError(
            f"the declared {donor.source} donor ({donor.role}) is neither "
            f"a row of {fetch_routes.ROUTE_TABLE_NAME} nor one of the "
            f"GFS-container sources {', '.join(GFS_CONTAINER_SOURCES)}; "
            "nothing in this door can download it, so the handoff would "
            "name a supplement nobody fetched")
    start = min(donor.leads)
    # The primary's --transport pin reaches the donor: a user who pinned
    # a host asked for every byte of this fetch from it, and a donor that
    # cannot serve that host refuses in its route's own words instead of
    # quietly walking its own ladder to another host.
    plan = fetch_routes.resolve_request(
        donor.source, cycle=donor.cycle, hours=max(donor.leads) - start,
        start_hour=start, host=getattr(args, "transport", None),
        out=_route_donor_out(args.out, donor))
    if getattr(donor, "records", None):
        # The donor row names the records the composition binds: only the
        # bound primary object moves, and only those records of it
        # (hrrr-native's soil donor: 29 MB of a 381 MB wrfprs, and not the
        # 134 MB wrfsfcf00 the donor route would also take).
        plan = fetch_routes.donor_records_plan(plan, donor.records)
    return plan


def _donor_route_primary(donor_plan, donor) -> Path:
    """The one primary object the donor role binds (its lowest lead)."""
    if not donor_plan.primary_files:
        raise ValueError(
            f"the {donor.source} donor route plans no primary object for "
            f"leads {list(donor.leads)}")
    return Path(donor_plan.primary_files[0])


#: The receipt a record-selected donor folder carries.
DONOR_RECORDS_RECEIPT = "donor-records.json"
DONOR_RECORDS_SCHEMA = "gpuwm-donor-records-v1"


def _donor_records_request(donor) -> dict:
    return {"source": donor.source, "cycle": f"{donor.cycle:%Y-%m-%dT%H}",
            "leads": list(donor.leads), "records": dict(donor.records)}


def _donor_records_cached(donor, donor_out: Path) -> list[Path] | None:
    """The donor's record files when ``donor_out`` already holds exactly them."""
    try:
        receipt = json.loads((Path(donor_out) / DONOR_RECORDS_RECEIPT)
                             .read_text(encoding="utf-8"))
        if (receipt.get("schema") != DONOR_RECORDS_SCHEMA
                or receipt.get("request") != json.loads(json.dumps(
                    _donor_records_request(donor)))):
            return None
        paths = []
        for entry in receipt["files"]:
            path = Path(donor_out) / entry["name"]
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                return None
            with path.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != entry["sha256"]:
                    return None
            paths.append(path)
        return paths or None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _fetch_donor_records(donor, args, donor_out: Path) -> list[Path]:
    """Move only the records a donor binds, by the donor object's index.

    The four ECMWF-family routes bind one field from the same cycle's GFS
    analysis (sea_ice_fraction from ICEC).  Moving the whole f000 for it
    took about 490 MB and 25 s for a 0.1 MB record; hex already
    took the same record by its ``.idx``.  The donor object's endpoints are
    the GFS ladder (the hosted policy's pin first), throughput order.
    """
    from woof.record_index import download_records

    donor_out = Path(donor_out)
    if not args.force_refetch:
        cached = _donor_records_cached(donor, donor_out)
        if cached is not None:
            print(f"fetch {donor.source} donor: {len(cached)} record file"
                  f"{'' if len(cached) == 1 else 's'} already present")
            return cached
    (donor_out / DONOR_RECORDS_RECEIPT).unlink(missing_ok=True)
    ladder = fetch_endpoints.serving_ladder(
        donor.source, cycle=donor.cycle,
        pinned=fetch_endpoints.policy_transport(donor.source))
    rungs = tuple(entry for entry in fetch_endpoints.transfer_order(ladder)
                  if entry.idx_available)
    if not rungs:
        raise ValueError(f"no {donor.source} endpoint for {donor.cycle:%Y-%m-%dT%H} "
                         "publishes an index")
    entries = []
    paths = []
    for lead in donor.leads:
        key = gfs_object_key(donor.cycle, lead, donor.source)
        dest = donor_out / f"{key.rsplit('/', 1)[-1]}.records.grib2"

        def transfer(endpoint, key=key, dest=dest):
            return download_records(endpoint.url(key), endpoint.url(key) + ".idx",
                                    dest, declaration=donor.records)

        endpoint, entry = fetch_endpoints.ask_along_ladder(
            rungs, transfer, label=f"fetch {donor.source} donor", name=dest.name,
            progress=print,
            discard=lambda _endpoint, dest=dest: dest.with_name(
                dest.name + ".part").unlink(missing_ok=True))
        entries.append({**entry, "lead": lead, "endpoint": endpoint.name})
        paths.append(dest)
        print(f"fetch {donor.source} donor: f{lead:03d} {entry['selected_records']} "
              f"record(s), {entry['bytes']:,} of {entry['object_bytes'] or 0:,} bytes "
              f"from {endpoint.name}")
    _atomic_write_text(donor_out / DONOR_RECORDS_RECEIPT, json.dumps(
        {"schema": DONOR_RECORDS_SCHEMA, "request": _donor_records_request(donor),
         "files": entries}, indent=2) + "\n")
    return paths


def _donor_cached(donor, args) -> bool:
    """Whether the donor folder already holds this exact donor request."""
    donor_out = _route_donor_out(args.out, donor)
    if getattr(donor, "records", None) and donor.source in GFS_CONTAINER_SOURCES:
        return _donor_records_cached(donor, donor_out) is not None
    if _donor_is_table_route(donor):
        return fetch_routes.request_cached(_donor_route_plan(donor, args),
                                           donor_out)
    return cached_request_complete(
        donor_out, source=donor.source, cycle=donor.cycle, area=None,
        hours=tuple(donor.leads), mode="full-file", refuse_changed=True)


def _fetch_route_donors(plan, args) -> dict:
    """Fetch the cross-source analysis a hybrid profile declares.

    The atmosphere-only AI products publish no land surface at all;
    their packaged compositions bind the missing canonicals to the
    same-cycle GDAS analysis.  Leaving that to the reader is what made
    them unreachable, so the front door fetches the declared donor with
    the same command, into its own subdirectory, and binds it in the
    handoff.
    """

    donor_files: dict[str, Path] = {}
    for donor in plan.donors:
        donor_out = _route_donor_out(args.out, donor)
        print(f"fetch {plan.source_id}: fetching the declared "
              f"{donor.source} donor into {donor_out}")
        print(f"  why: {donor.why}")
        if donor.source not in GFS_CONTAINER_SOURCES:
            # A donor that is itself a table route (hrrr-native's
            # same-cycle hrrr-prs soil column) is fetched the way its
            # own ``woof fetch --source`` line would fetch it: the
            # route's plan for the donor's leads, into the donor
            # folder.  Only the two GFS-container sources have no row
            # and keep the whole-file ladder below; a donor with
            # neither is refused by name because nothing here can
            # move its bytes.
            donor_plan = _donor_route_plan(donor, args)
            fetch_routes.run_plan(donor_plan, out=donor_out,
                                  force=args.force_refetch,
                                  file_workers=args.fetch_workers,
                                  downloader=_route_stream_downloader(donor_plan, args))
            donor_files[donor.role] = donor_out / _donor_route_primary(
                donor_plan, donor)
            continue
        if donor.records:
            try:
                donor_files[donor.role] = _fetch_donor_records(
                    donor, args, donor_out)[0]
                continue
            except (ValueError, OSError) as error:
                # A per-run fallback that says so: the whole object holds
                # the same record, so the start still gets its field.
                print(f"fetch {plan.source_id}: WARNING: the {donor.source} "
                      f"donor's records could not be taken by index ({error}); "
                      "moving the whole object instead")
        choice = select_fetch_engine("auto")
        manifest = fetch_gfs_fullfile(
            cycle=donor.cycle, hours=tuple(donor.leads), area=None,
            out=donor_out, force=args.force_refetch, source=donor.source,
            engine=choice.engine, engine_bin=choice.binary,
            engine_selection=choice.selection, cache_dir=None,
            top_pressure_pa=None, all_levels=False,
            file_workers=args.fetch_workers,
            transport=fetch_endpoints.policy_transport(donor.source))
        document = json.loads(Path(manifest).read_text(encoding="utf-8"))
        names = [entry["name"] for entry in document.get("files", [])]
        if not names:
            raise ValueError(
                f"the {donor.source} donor fetch published no file")
        donor_files[donor.role] = donor_out / names[0]
    return donor_files


def _retrieve_inapplicable_refusal(source: str) -> str:
    return ("--retrieve applies only to sources whose default fetch writes a request template; "
            f"{source} has no request template to retrieve")


def fetch_main(args) -> int:
    """``woof fetch``; exit 75 when an as-posted source fell behind its budget."""

    from woof.earth2_licensing import check_namespace
    check_namespace(args)
    from woof.fetch_as_posted import SourceBehind
    from woof.timeline.ledger import emit

    from woof import fetch_connections
    old_cap = os.environ.get(fetch_connections.MAX_CONNECTIONS_ENV)
    configured_cap = getattr(args, "fetch_max_connections", None)
    if configured_cap is not None:
        os.environ[fetch_connections.MAX_CONNECTIONS_ENV] = str(fetch_connections.max_connections(configured_cap))
    fetch_connections.max_connections()
    emit("fetch.request", "begin", attrs={"source": args.source,
         "worker_row": fetch_pool.worker_profile()["id"]})
    try:
        return _fetch_main(args)
    except SourceBehind as error:
        print(f"fetch {args.source}: {error}", file=sys.stderr)
        return error.exit_code
    finally:
        emit("fetch.request", "end", attrs={"source": args.source})
        if configured_cap is not None:
            if old_cap is None:
                os.environ.pop(fetch_connections.MAX_CONNECTIONS_ENV, None)
            else:
                os.environ[fetch_connections.MAX_CONNECTIONS_ENV] = old_cap


def _fetch_main(args) -> int:
    args.source = source_adapters.source_for_cycle(args.source, getattr(args, "cycle", None))
    source = args.source
    if fetch_endpoints.policy_uses_aws(source):
        args.transport = fetch_endpoints.policy_transport(source)
        if source in GFS_CONTAINER_SOURCES and args.mode != "full-file":
            # The policy forbids the grib-filter host, so the bytes come
            # from the archive's objects: the records the decode reads, by
            # their .idx (1,556 MB -> 266.5 MB for a 6 h 3-lead start,
            # 2026-10-29); '--mode full-file' still takes them whole.
            args.mode = GFS_RECORDS_MODE
    _posting_flag_refusal(args, requested_as_posted(args))
    if getattr(args, "readiness", False):
        return _readiness_main(args, source)
    if getattr(args, "wif", False):
        from woof.table_assets import stage_wif_dataset

        if stage_wif_dataset() != 0:
            from woof.config import MP28_AEROSOL_LATERAL_FORCING_PRECONDITION

            print("fetch: " + MP28_AEROSOL_LATERAL_FORCING_PRECONDITION
                  + " Automatic acquisition failed before forcing transfer. "
                    "Offline: woof fetch-tables --wif --from DIR.", file=sys.stderr)
            return 2
    if (getattr(args, "retrieve", False)
            and not source_adapters.get_source_adapter(source).fetch_requires_retrieve):
        raise ValueError(_retrieve_inapplicable_refusal(source))
    era5_provider = getattr(args, "era5_provider", None)
    if era5_provider is not None and source != "era5":
        raise ValueError("--era5-provider applies to --source era5 only")
    era5_provider = era5_provider or "cds"
    era5_product = getattr(args, "era5_product", None)
    if era5_product is not None and source != "era5":
        raise ValueError("--era5-product applies to --source era5 only")
    era5_product = era5_product or "reanalysis"
    if native_cf_fetch_contract(source) is not None:
        from woof.cf_archive_fetch import cli
        return cli(args)
    if source in fetch_routes.route_ids():
        # A table route refuses the flags it does not take, then validates
        # its window with the same validator.
        return _route_fetch_main(args, source)
    acquisition_owner = source_adapters.get_source_adapter(source).acquisition_owner
    if acquisition_owner is not None:
        from importlib import import_module
        return import_module(acquisition_owner).fetch(args)
    if getattr(args, "model", None) is not None:
        raise ValueError("--model applies to a source whose row runs a model "
                         "(--source earth2)")
    if args.validate is not None and source != "era5":
        raise ValueError("--validate applies to --source era5 only")
    if args.fetch_workers is not None:
        if source == "era5":
            raise ValueError(
                "--fetch-workers does not apply to ERA5; its provider "
                "controls retrieval concurrency")
        # Refused here, before any network round trip, in the pool's own
        # words (a zero-or-negative count names no schedulable pool).
        fetch_pool.resolve_file_workers(args.fetch_workers)

    if args.transport is not None and not fetch_endpoints.has_ladder(source):
        raise ValueError(
            f"--transport: --source {source} has no host to choose "
            "between (ERA5 is a manual CDS retrieval; the template is "
            "written locally)")
    if source not in ("hrrr",) + GFS_CONTAINER_SOURCES:
        transported = sorted(
            flag for flag, value in (
                ("--engine", args.engine),
                ("--mode", args.mode),
                ("--cache-dir", args.cache_dir),
            ) if value is not None)
        if transported:
            raise ValueError(
                f"{', '.join(transported)}: --source hrrr or "
                f"{'/'.join(GFS_CONTAINER_SOURCES)} only (ERA5 is a "
                "manual CDS retrieval)")
    # The GFS container sources have three byte transports, and --mode
    # chooses between them: the NOMADS grib-filter crop (the default;
    # spatial subregion + exact record selection), --mode idx-subset
    # (the same records, copied unchanged from the whole pgrb2.0p25
    # objects by their .idx byte ranges) and --mode full-file (the whole
    # objects, the same first-class whole-file doctrine as HRRR).  The
    # last two read the archive's objects; a host that keeps no index
    # moves its objects whole under idx-subset, and says so.  'auto' asks
    # for the default choice, so it takes exactly the path an omitted
    # --mode takes: reuse of verified crops, the crop, and the archive's
    # records for a cycle the crop host no longer keeps.
    gfs_fullfile = False
    gfs_records = False
    gfs_named_mode = None if args.mode == "auto" else args.mode
    if source in GFS_CONTAINER_SOURCES:
        pinned_rung = (
            fetch_endpoints.endpoint_named(source, args.transport)
            if args.transport not in (None, "auto") else None)
        pinned_archive = pinned_rung is not None and bool(pinned_rung.layouts)
        gfs_fullfile = (gfs_named_mode in ("full-file", GFS_RECORDS_MODE)
                        or pinned_archive)
        # An archive host named with --transport and no mode takes the
        # records when it publishes an index, its whole objects when not.
        gfs_records = gfs_named_mode == GFS_RECORDS_MODE or (
            gfs_named_mode is None and pinned_archive
            and pinned_rung.idx_available)
        if not gfs_fullfile:
            cgi_extras = sorted(
                flag for flag, value in (
                    ("--engine", args.engine),
                    ("--cache-dir", args.cache_dir),
                    ("--transport", args.transport),
                ) if value is not None)
            if cgi_extras:
                raise ValueError(
                    f"{', '.join(cgi_extras)}: these choose how the "
                    "archive's objects move and belong to '--mode "
                    "full-file' (or '--mode idx-subset', the same objects' "
                    "records); the default NOMADS grib-filter crop has "
                    "exactly one transport (governed stdlib HTTP)")
    if source not in GFS_CONTAINER_SOURCES:
        gfs_only = sorted(
            flag for flag, value in (
                ("--p-top-pa", args.p_top_pa),
                ("--all-levels", args.all_levels or None),
            ) if value is not None)
        if gfs_only:
            raise ValueError(
                f"{', '.join(gfs_only)}: --source "
                f"{'/'.join(GFS_CONTAINER_SOURCES)} only.  HRRR is fetched "
                "on its native hybrid levels (no isobaric ladder to "
                "choose), and the ERA5 request template carries its own "
                "level list.")
    # A separate refusal, with its own reason.  --forecast-start-hour used
    # to ride the level-ladder bundle above, so `--source hrrr
    # --forecast-start-hour 6` was declined for having asked about an
    # isobaric ladder -- a sentence about a flag the user had not typed,
    # for a source whose whole decode path is already lead-aware.  What
    # the flag actually needs is a source with forecast leads in it, and
    # the registry row says which sources have them, as it does for the
    # [fetch] table.
    if (args.forecast_start_hour is not None
            and not _source_reaches_forecast_leads(source)):
        raise ValueError(
            f"--forecast-start-hour: {source} declares max_forecast_hour = 0 "
            "and publishes analyses, not forecasts, so there is no forecast "
            "lead for a window to begin at.  Name the analysis time you want "
            "with --cycle instead.")
    # `<= 0` alone let NaN and infinity through: NaN compares false
    # against every level of the ladder, infinity asks for no top at all.
    if args.p_top_pa is not None and not (
            math.isfinite(args.p_top_pa) and args.p_top_pa > 0):
        raise ValueError("--p-top-pa must be a positive, finite pressure in Pa")

    # Every flag this source does not take is refused above, before the
    # window is read: a window refusal would name a problem the stray flag
    # was never going to fix.
    hints = {"source": source}
    # --transport was checked above with --mode in view; a table cannot
    # carry a mode, so its transport check would refuse the full-file host.
    for key in FETCH_HINT_KEYS - {"source", "source_root", "transport"}:
        if key in {"era5_provider", "era5_product"} and source != "era5":
            continue
        if key == "retrieve" and not source_adapters.get_source_adapter(source).fetch_requires_retrieve:
            continue
        value = getattr(args, key, None)
        if value is not None:
            hints[key] = str(value) if isinstance(value, Path) else value
    validate_fetch_hints(hints, source=COMMAND_LINE_HINTS)
    if getattr(args, "member", None) is not None and source != "era5":
        raise ValueError(
            f"--member: --source {source} is not an ensemble route")
    area = _resolve_area(args)

    author_roles = {
        "--bridge": args.bridge,
        "--wps-namelist": args.wps_namelist,
        "--experiment-config": args.experiment_config,
        "--static-input": args.static_input,
        "--static-receipt": args.static_receipt,
        "--manifest-out": args.manifest_out,
    }
    if args.author_front_door_manifest:
        if source not in GFS_CONTAINER_SOURCES:
            raise ValueError(
                "--author-front-door-manifest applies to --source "
                f"{'/'.join(GFS_CONTAINER_SOURCES)} only (the HRRR front "
                "door consumes the fetched SHA256SUMS directly; its "
                "handoff line is printed after every HRRR fetch)")
        required = ("--wps-namelist", "--experiment-config")
        absent = [flag for flag in required if author_roles[flag] is None]
        if absent or args.out is None:
            raise ValueError(
                "--author-front-door-manifest requires --out plus "
                + ", ".join(required)
                + " (the front-door manifest binds each file's sha256, "
                "including the bridge executable's)")
        if args.bridge is None:
            # `woof go` has always resolved this through
            # woof.bridges; the stage-by-stage route demanded a path
            # instead, and the one FIRST-LIGHT printed
            # (tools/grib1_bridge/target/release/...) exists only in a
            # checkout -- so a wheel user following the documented long
            # form met "front-door manifest inputs are missing: bridge"
            # after paying for the fetch.  Same resolver, same answer,
            # whichever door they came through.  Resolved after the
            # flags above so a usage mistake still reads as one.
            args.bridge = author_roles["--bridge"] = _resolve_manifest_bridge(
                source)
    else:
        supplied = sorted(
            flag for flag, value in author_roles.items()
            if value is not None)
        if supplied:
            raise ValueError(
                f"{', '.join(supplied)} belong to "
                "--author-front-door-manifest")

    if (args.author_front_door_manifest and args.cycle is None
            and args.hours is None):
        # Author-only: convert an already-completed fetch directory.
        author_gfs_front_door_manifest(
            out=args.out, bridge=args.bridge,
            wps_namelist=args.wps_namelist,
            experiment_config=args.experiment_config,
            static_input=args.static_input,
            static_receipt=args.static_receipt,
            manifest_out=args.manifest_out, source=source,
            forecast_start_hour=args.forecast_start_hour)
        return 0
    if source == "era5" and args.validate:
        from woof.era5_member import validate_selection, check_member
        member = validate_selection(product_type=era5_product, member=getattr(args, "member", None),
            provider=era5_provider, cadence=args.cadence if args.cadence is not None else 6,
            cycle=parse_cycle(args.cycle, source) if args.cycle is not None else None)
        if member is not None:
            for path in args.validate:
                check_member(path, member)
        expected = None
        if args.cycle is not None and args.hours is not None:
            cycle = parse_cycle(args.cycle, source)
            cadence = args.cadence if args.cadence is not None else 6
            expected = _era5_times(cycle, args.hours, cadence)
        report = validate_era5_files(
            tuple(args.validate), expected_times=expected,
            expected_area=area)
        print(report.format())
        return 0 if report.ok else 1

    if args.cycle is None or args.hours is None or args.out is None:
        raise ValueError(
            "fetch requires --cycle, --hours, and --out (the exceptions: "
            "era5 --validate mode, and gfs --author-front-door-manifest "
            "on an already-fetched --out, which needs neither --cycle "
            "nor --hours)")
    if args.hours < 0:
        raise ValueError("--hours cannot be negative")
    if args.hours == 0 and args.author_front_door_manifest:
        raise ValueError(
            "--hours 0 fetches one analysis, and a forecast manifest needs at "
            "least two forcing times: lateral boundaries are interpolated "
            "BETWEEN frames, so one frame leaves every boundary interval "
            "empty.  Raise --hours to one cadence step, or drop "
            "--author-front-door-manifest and keep the analysis.")
    validate_fetch_cadence(source, args.cadence)
    if args.cadence is not None and not fetch_accepts_cadence(source):
        # Plan review, and through the same function the [fetch] table's
        # config-load check asks, so the flag and the table cannot hold
        # two answers about one source.  It used to live inside the HRRR
        # dispatch branch, which is after every other source's plan
        # review and reachable only by name.
        raise ValueError(cadence_inapplicable_refusal(source))

    if source == "era5":
        if area is None:
            raise ValueError("era5 fetch requires --area or --point")
        cadence = args.cadence if args.cadence is not None else 6
        # Validate the duration before selecting a date or creating a template.
        _era5_times(datetime(2000, 1, 1), args.hours, cadence)
        cycle = (resolve_latest_cycle(source, args.hours, provider=era5_provider)
                 if args.cycle == "latest" else parse_cycle(args.cycle, source))
        from woof.era5_member import validate_selection
        member = validate_selection(product_type=era5_product, member=getattr(args, "member", None),
            provider=era5_provider, cadence=cadence, cycle=cycle)
        if args.cycle == "latest":
            end = cycle + timedelta(hours=args.hours)
            _grid, basis = provider_cycle_grid(source, era5_provider)
            print(f"fetch era5: latest analysis window {cycle:%Y-%m-%dT%H}Z..{end:%Y-%m-%dT%H}Z "
                  f"from {basis}, not a live completeness probe")
        if member is not None and not getattr(args, "retrieve", False):
            raise ValueError("ERA5 EDA requires --retrieve so native member verification runs before publication")
        if getattr(args, "retrieve", False) or era5_provider == "arco":
            if era5_provider == "arco":
                from woof.era5_arco import retrieve_era5_arco as retrieve
            else:
                from woof.era5_acquisition import retrieve_era5 as retrieve
            retrieve(cycle=cycle, hours=args.hours,
                area=area, out=args.out, cadence=cadence, force=args.force_refetch,
                **({"product_type": era5_product, "member": member} if era5_provider == "cds" else {}))
        else:
            write_era5_request(
                cycle=cycle, hours=args.hours,
                area=area, out=args.out, cadence=cadence)
        return 0

    if source in GFS_CONTAINER_SOURCES:
        if area is None and not gfs_fullfile:
            raise ValueError(
                f"{source} fetch requires --area or --point --radius-km: "
                "the NOMADS subsetter needs a subregion (--mode full-file "
                "takes the whole-globe objects instead, and there --area "
                "is optional request identity)")
        hours = container_forecast_hours(
            source, args.hours, args.cadence, args.forecast_start_hour)
        if fetch_routes.prepares_through_packaged_composition(source):
            # Plan review for the preparation handoff this fetch will
            # publish.  Asked here, before the transfer, because the
            # publisher asks the same question after it and a refusal
            # that arrives then has already spent the download -- and
            # asked through the SAME predicate the publisher forks on, so
            # a container that gains a composed profile cannot have its
            # review skipped here while the writer still demands one.
            container_handoff_binding(source)
        if hours[0]:
            print(f"fetch {source}: window begins at forecast lead "
                  f"f{hours[0]:03d}; a model initialized there starts from "
                  f"a {hours[0]} h forecast, not an analysis")
        if args.cycle == "latest":
            # The host the transfer is pinned to is the host asked: a
            # cycle only another host has cannot be downloaded from this
            # one.
            query, last, options = latest_cycle_request(args)
            cycle = resolve_latest_cycle(query, last, **options)
            print(f"fetch {source}: latest "
                  f"{'startable' if options.get('as_posted') else 'complete'} "
                  f"cycle is {cycle:%Y-%m-%dT%H}Z")
        else:
            cycle = parse_cycle(args.cycle, source)
        # A folder that already holds this whole request as grib-filter
        # crops keeps its transport: the switch below is about where to
        # DOWNLOAD an old cycle, and re-running a finished fetch after the
        # crop host rolled the cycle off must reuse the crops rather than
        # be refused for holding the other transport's files.
        #
        # A crop damaged in place is refused here with the refusal the
        # crop route gives it, before an old cycle would be switched to
        # the archive's whole objects and refused instead for the mode
        # the folder recorded, a refusal that never named the file.
        # Both reuse checks speak through one printer that says each
        # line once: the second, under the lock, asks the same question.
        said: set[str] = set()

        def say_once(line: str) -> None:
            if line not in said:
                said.add(line)
                print(line)

        subset_cached = (
            not gfs_fullfile and gfs_named_mode is None
            and args.cycle != "latest" and not args.force_refetch
            and area is not None
            and cached_request_complete(
                args.out, source=source, cycle=cycle, area=area,
                hours=hours, mode="nomads-cgi-subset", progress=say_once,
                refuse_changed=True))
        if (not gfs_fullfile and gfs_named_mode is None and not subset_cached
                and archive_only_cycle(source, cycle)):
            # The crop host keeps a rolling window; an older cycle exists
            # only as whole objects in the archive.  Asking the crop host
            # for it failed with a 403 and a refusal, so a date the
            # archive holds could not be fetched without knowing a flag.
            # The archive's .idx gives the crop's records without the
            # rest of each object.
            gfs_fullfile = True
            gfs_records = True
            print(f"fetch {source}: {cycle:%Y-%m-%dT%H}Z is older than the "
                  "grib-filter host keeps; reading the same records from the "
                  f"archive's objects by their .idx (--mode {GFS_RECORDS_MODE})")
        if (gfs_records and not args.force_refetch and args.cycle != "latest"
                and cached_request_complete(
                    args.out, source=source, cycle=cycle, area=area,
                    hours=hours, mode="full-file", progress=say_once,
                    refuse_changed=True)):
            # Whole objects carry every record the selection would take,
            # so a folder that already holds this request whole is used
            # as it is rather than refused for its transport.
            gfs_records = False
            say_once(f"fetch {source}: {args.out} already holds this request "
                     "as whole objects, which carry every record; using them")
        # The request-identity guard and the transfer it authorises are
        # one decision: taking the lock around BOTH is what stops two
        # writers from passing the guard together and then publishing
        # incompatible receipts into the same directory.  The library
        # call re-enters the same lock (it is re-entrant per process).
        requested_gfs_mode = (
            (GFS_RECORDS_MODE if gfs_records else "full-file") if gfs_fullfile
            else "nomads-cgi-subset")
        with _fetch_out(args.out):
            if not args.force_refetch:
                require_matching_request(args.out, source=source,
                                         cycle=cycle, area=area,
                                         mode=requested_gfs_mode)
            loop = None
            if requested_as_posted(args) and (
                    args.force_refetch or not cached_request_complete(
                        args.out, source=source, cycle=cycle, area=area,
                        hours=hours, mode=requested_gfs_mode,
                        progress=say_once, refuse_changed=True)):
                loop = _posting_loop(source, cycle, args, leads=hours,
                                     transport=pinned_host(args.transport))
            if args.cycle != "latest" and loop is None:
                # A folder that already holds every file of this exact
                # request needs nothing from the provider, so it is
                # checked before the provider is asked: a finished
                # download stays usable offline and after the cycle has
                # left the provider's retention.
                if args.force_refetch or not cached_request_complete(
                        args.out, source=source, cycle=cycle, area=area,
                        hours=hours, mode=requested_gfs_mode,
                        progress=say_once, refuse_changed=True):
                    require_published_cycle(
                        source, cycle, hours[-1],
                        transport=pinned_host(args.transport))
            if gfs_fullfile:
                choice = select_fetch_engine(
                    args.engine if args.engine is not None else "auto")
                engine, engine_bin = choice.engine, choice.binary
                print(f"fetch {source}: engine {engine}"
                      + (f" ({engine_bin})" if engine_bin is not None
                         else "")
                      + (" [inherited, not chosen]" if choice.degraded
                         else "")
                      + (f", mode {GFS_RECORDS_MODE} (the grib-filter records "
                         "of each pgrb2.0p25 object by its .idx, "
                         if gfs_records else
                         ", mode full-file (whole pgrb2.0p25 objects, ")
                      + " then ".join(
                          endpoint.name for endpoint in
                          fetch_endpoints.serving_ladder(
                              source, cycle=cycle,
                              pinned=pinned_host(args.transport)))
                      + ")")
                def transfer(window, force):
                    return fetch_gfs_fullfile(
                        cycle=cycle, hours=window, area=area, out=args.out,
                        force=force, source=source,
                        engine=engine, engine_bin=engine_bin,
                        engine_selection=choice.selection,
                        cache_dir=args.cache_dir,
                        top_pressure_pa=args.p_top_pa,
                        all_levels=args.all_levels,
                        transport=pinned_host(args.transport),
                        file_workers=args.fetch_workers,
                        records=gfs_records)
            else:
                def transfer(window, force):
                    return fetch_gfs(
                        cycle=cycle, hours=window, area=area, out=args.out,
                        force=force, source=source,
                        accept_inventory_change=args.accept_inventory_change,
                        top_pressure_pa=args.p_top_pa,
                        all_levels=args.all_levels,
                        file_workers=args.fetch_workers)
            manifest = _legacy_transfer(loop, transfer, hours,
                                        force=args.force_refetch)
    elif source == "hrrr":
        transport = args.transport if args.transport is not None else "auto"
        # Only an unpinned request may wander between hosts; an operator
        # who named --transport gets that host or an error.
        transport_fallback = (
            tuple(HRRR_TRANSPORTS[1:]) if transport == "auto" else ())
        choice = select_fetch_engine(
            args.engine if args.engine is not None else "auto")
        engine, engine_bin = choice.engine, choice.binary
        # The default ('auto': the record subset, whole where the index
        # cannot carry it) is the same on both engines; the Python
        # transport only does .idx range subsets.  What that transport
        # costs is said at selection time, by select_fetch_engine, so the
        # GFS full-file command, the streamer's preflight and every
        # library caller of the front door hear it too.
        if args.mode is not None:
            mode = args.mode
        else:
            mode = HRRR_DEFAULT_MODE
        if engine == "python" and mode != "auto":
            # The build line comes from the shared shell rule: a literal
            # `&&` here was a Windows PowerShell 5.1 parser error.
            from woof import bridges
            raise ValueError(
                f"--mode {mode} needs the rust fetch backbone: the Python "
                "transport only does .idx range subsets.  Build the "
                "backbone ("
                + bridges.cargo_build_one_liner(
                    bridges.RUSTWX_CRATE_RELATIVE)
                + ") or drop --mode.")
        # The lead is checked before any network round trip: `--cycle
        # latest` probes for a cycle complete through the END of the
        # window, and a bad lead should not have to pay for a probe to
        # be refused.  --hours stays the window LENGTH on every source,
        # so the window's final lead is lead + length.
        start_hour = _forecast_start_hour(args.forecast_start_hour)
        if start_hour:
            print(f"fetch hrrr: window begins at forecast lead "
                  f"f{start_hour:02d}; a model initialized there starts "
                  f"from a {start_hour} h forecast, not an analysis")
        if args.cycle == "latest":
            # Asked of the pinned host when there is one: the transfer
            # below downloads from that host only.  As posted, the newest
            # cycle whose start needs are out; its later hours are
            # fetched as they post.
            query, last, options = latest_cycle_request(args)
            cycle = resolve_latest_cycle(query, last, **options)
            print(f"fetch hrrr: latest "
                  f"{'startable' if options.get('as_posted') else 'complete'} "
                  f"cycle is {cycle:%Y-%m-%dT%H}Z")
        else:
            cycle = parse_cycle(args.cycle, source)
        hours = hrrr_forecast_hours(args.hours, cycle, start_hour)
        # One lock over the guard and the transfer it authorises; see the
        # GFS branch above.
        with _fetch_out(args.out):
            if not args.force_refetch:
                require_matching_request(args.out, source="hrrr",
                                         cycle=cycle, area=area)
            # A yes means every file still holds the bytes its receipt
            # recorded, so nothing will move and no host is asked; a
            # file damaged in place sends the fetch the uncached way,
            # where the host that serves it is one that still has it.
            cached = not args.force_refetch and cached_request_complete(
                args.out, source="hrrr", cycle=cycle, area=area,
                hours=hours, progress=print)
            loop = None
            whole = None
            if not cached and requested_as_posted(args):
                # As posted, each hour is waited for by the loop and then
                # fetched; a window already whole moves in one call.
                loop = _posting_loop(source, cycle, args, leads=hours,
                                     transport=pinned_host(args.transport))
                if loop is not None:
                    # The named cycle's own check still runs for what it
                    # says: the refusal of a host that no longer keeps the
                    # cycle, and the sentence naming a host that could not
                    # be heard (GS-05). Its answer does not make the window
                    # whole: a posted final lead does not prove the
                    # intervening atmosphere, soil and runtime files reached this
                    # host, so each lead is checked before it joins the
                    # downloaded prefix.
                    if not fetch_endpoints.policy_uses_aws(source):
                        _window_posted(
                            loop.window, lambda said: require_published_cycle(
                                source, cycle, hours[-1], progress=said,
                                transport=pinned_host(args.transport)))
                    whole = False
            if not cached and args.cycle != "latest" and loop is None:
                require_published_cycle(
                    source, cycle, hours[-1],
                    transport=pinned_host(args.transport))
            if cached:
                # Every file matched its recorded digest above, so
                # nothing will be downloaded; the files keep the host
                # they were fetched from.
                transport = (pinned_host(args.transport)
                             or _recorded_hrrr_transport(args.out))
            elif loop is None or whole:
                # One transport decision per invocation; 'auto' probes
                # NOMADS for the window's final hour pair, falls back S3.
                transport = resolve_hrrr_transport(
                    cycle, transport, last_hour=hours[-1])
            # Name who chose the byte mode.  The backbone cannot: it
            # sees `--mode full-file` on its command line and cannot
            # tell a typed flag from this front door's own default,
            # which is where every unqualified `woof fetch` gets it.
            mode_chooser = "you" if args.mode is not None else "the default"
            print(f"fetch hrrr: engine {engine}"
                  + (f" ({engine_bin})" if engine_bin is not None else "")
                  + (f", mode {mode} ({mode_chooser})"
                     if engine == "rust" else ""))
            advise_pacing = _hrrr_pacing_advice(
                pinned=pinned_host(args.transport), mode=mode, cached=cached)
            advise_pacing(transport)
            requested_transport = transport

            def transfer(window, force, *, on_hour_ready=None):
                # The host is decided per call: under the loop the
                # window's final hour is the newest posted one.
                host = requested_transport
                if loop is not None and not whole:
                    host = (pinned_host(args.transport)
                            or _posted_hrrr_transport(loop, cycle, window))
                    # As posted, the transport above stays unresolved, so
                    # the cost of a resolved NOMADS whole-file transfer
                    # is said here, before the first prefix it serves.
                    advise_pacing(host)
                return fetch_hrrr(
                    cycle=cycle, hours=window, area=area, out=args.out,
                    force=force, transport=host,
                    engine=engine, engine_bin=engine_bin,
                    engine_selection=choice.selection, mode=mode,
                    cache_dir=args.cache_dir,
                    accept_inventory_change=args.accept_inventory_change,
                    file_workers=args.fetch_workers,
                    transport_fallback=transport_fallback,
                    runtime_surface=getattr(args, "runtime_surface", None),
                    analyzed_aerosol=bool(getattr(args, "analyzed_aerosol", False)),
                    **({} if on_hour_ready is None else {
                        "on_hour_ready": on_hour_ready}))

            manifest = _legacy_transfer(loop, transfer, hours,
                force=args.force_refetch, whole=whole,
                incremental_transfer=lambda window, force, ready: transfer(
                    window, force, on_hour_ready=ready))
    else:
        raise ValueError(f"unknown fetch source {source!r}")
    print(f"fetch {source}: manifest {manifest}")
    if source == "hrrr":
        # The trailing `...` this used to print was on a command line
        # after a "front door:" label, and the consumer refuses it:
        # `woof-wrf-init: error: unrecognized arguments: ...`.  A
        # successful producer must not print a command that fails before
        # it can look at what was just fetched.  So the half this step
        # knows is a bound command, and the half it cannot know is a
        # comment naming every flag the door still needs -- the same
        # shape the 20CRv3 authoring step uses.
        # Absolute, as every table route prints them, so the line runs
        # from whatever directory the reader pastes it in.
        sums = args.out.resolve() / "SHA256SUMS"
        bound = ["--source", "hrrr", "--source-root", str(sums.parent),
                 "--source-manifest", str(sums),
                 "--source-manifest-sha256", sha256_file(sums),
                 "--valid-time", f"{cycle:%Y-%m-%d_%H:%M:%S}"]
        if hours[0]:
            bound += ["--forecast-start-hour", str(hours[0])]
        print("fetch hrrr: next: feed the HRRR front door, source "
              "already bound:")
        print("  " + fetch_routes.render_prep_command(bound))
        print("  # fetching cannot bind the run's own flags: "
              + fetch_routes.named_flags(HRRR_CALLER_SUPPLIES)
              + " are yours to supply.\n"
              "  # `woof domain --source hrrr --out CONFIG.toml` writes "
              "the first four beside\n"
              "  # each other (CONFIG.d01-target.json, "
              "CONFIG.namelist.input,\n"
              "  # CONFIG.namelist.wps, CONFIG.toml) and prints the whole "
              "chain.  The\n"
              "  # --valid-time above is the CYCLE these files came from; "
              "model time zero\n"
              "  # is cycle + the lead, and every stage derives it.\n"
              "  # `woof prep --show-source hrrr` lists the full argument "
              "contract.")
    elif args.author_front_door_manifest:
        author_gfs_front_door_manifest(
            out=args.out, bridge=args.bridge,
            wps_namelist=args.wps_namelist,
            experiment_config=args.experiment_config,
            static_input=args.static_input,
            static_receipt=args.static_receipt,
            # No tail cut here: the download that just ran already
            # STARTS at --forecast-start-hour, so its series and manifest
            # are the window.  Cutting again would only be a second,
            # redundant statement of the same lead.
            manifest_out=args.manifest_out, source=source,
            forecast_start_hour=None)
    elif fetch_routes.prepares_through_packaged_composition(source):
        # The container writer published prep-arguments.json for this
        # source (it forks on the same predicate), so the handoff is the
        # table routes' own block, read back from that document.  What
        # stood here was a sentence with no command in it, telling the
        # reader to fetch with --all-levels, which the default ladder
        # already takes for this container.
        for line in fetch_routes.prep_handoff_lines(source, args.out):
            print(line)
    else:
        # A template with GFS_GRIB2_BRIDGE_EXE, NAMELIST_WPS and
        # EXPERIMENT_TOML in it was presented as "next" and does not run
        # as printed.  Same shape as the HRRR handoff above: the bound
        # half is a real command, the three values only the user has are
        # named in comments.
        print(f"fetch {source}: next: author the front-door input "
              "manifest.  This half is bound:")
        print(f"  woof fetch --source {source} "
              f"--author-front-door-manifest --out "
              f"{shlex.quote(str(args.out))}")
        print("  # and these two are yours to point at: --wps-namelist "
              "and\n"
              "  # --experiment-config.  The bridge resolves itself "
              "(--bridge PATH\n"
              "  # overrides); `woof doctor` names the one this "
              "install found.")
    return 0


def _window_posted(window, check) -> bool:
    """Whether a whole window is already posted, asked as a whole-cycle fetch asks.

    ``check(progress)`` is :func:`require_published_cycle` for the
    window's final lead.  Not posted is not a refusal under as-posted:
    the loop waits.  A host that could not be heard is not counted as
    holding the window either (the check says so on ``progress``), so the
    loop asks lead by lead, and lets the transfer try a lead whose host
    it still cannot hear (GS-05).  A cycle every host has let go (and no
    archive keeps), or that a pinned host no longer keeps, will never
    post there, so the whole-cycle refusal stands, in its own words.
    """

    from woof import source_readiness as readiness

    unheard: list[str] = []
    try:
        check(unheard.append)
    except RuntimeError:
        if readiness.retention_refusal(window, now=None) is not None:
            raise
        return False
    if unheard:
        print(unheard[0])
        return False
    return True


def _legacy_transfer(loop, transfer, hours, *, force: bool,
                     whole: bool | None = None, incremental_transfer=None):
    """One legacy transport's window, whole or as it posts.

    Without a loop, ``transfer(hours, force)`` once, as before.  With one
    (:func:`woof.fetch_as_posted.fetch_legacy_as_posted`) the transport
    is called for each verified prefix; a forced refetch sets the old
    files aside once, before the first call, and never again.
    ``incremental_transfer(leads, force, ready)`` may call
    ``ready(lead, manifest)`` before returning, after that lead's ordinary
    checks and manifest publication. Each marker is then published once;
    repeated prefix checks cannot change what the preparation already bound.
    """

    if loop is None:
        return transfer(hours, force)
    from woof.fetch_as_posted import (
        fetch_legacy_as_posted, legacy_objects, marker_name)

    window = loop.window
    if (incremental_transfer is not None
            or fetch_endpoints.policy_uses_aws(window.source)):
        # Native hourly completions belong to an individually gated
        # prefix, even when a final-lead probe called the window whole.
        # Otherwise a lagging intermediate object fails the fetch before
        # any start lead is admitted to the preparation.
        whole = False
    published = {}

    def publish_verified(lead, objects):
        lead = int(lead)
        objects = [dict(item) for item in objects]
        if lead in published:
            if published[lead] != objects:
                raise ValueError(
                    f"verified lead f{lead:03d} changed after its posted "
                    "marker was written; the preparation has already "
                    "bound the earlier objects")
            return loop.folder / marker_name(lead)
        if len(published) >= len(hours) or lead != hours[len(published)]:
            raise ValueError(
                f"verified lead f{lead:03d} arrived outside the ordered "
                "prefix; publishing it would leave an earlier lead "
                "unbound while the preparation waits for it")
        path = loop.publish(lead, objects)
        published[lead] = objects
        return path

    def ready(lead, manifest):
        document = json.loads(Path(manifest).read_text(encoding="utf-8"))
        publish_verified(lead, legacy_objects(document, lead))

    def move(leads, fresh):
        if incremental_transfer is None:
            return transfer(leads, fresh)
        return incremental_transfer(leads, fresh, ready)
    # A window whose final lead is out needs no waiting, and moves as one
    # request with its markers written in lead order.
    if whole is None:
        whole = _window_posted(window, lambda said: require_published_cycle(
            window.source, window.cycle, hours[-1], progress=said,
            transport=window.transport))
    if whole:
        # Started (and, forced, stripped of its lead markers) before the
        # transfer sets the old payloads aside, so no marker describes a
        # file that is gone while the new one downloads.
        loop.start(fresh=force)
        manifest = move(hours, force)
        document = json.loads(Path(manifest).read_text(encoding="utf-8"))
        for lead in hours:
            publish_verified(lead, legacy_objects(document, lead))
        return manifest
    forced = [force]

    def prefix(window):
        manifest = move(window, forced[0])
        forced[0] = False
        return manifest

    if force:
        # The receipts and payloads go aside before the schedule is
        # written, and the loop starts with no lead marker (fresh), so the
        # posting folder is this run's: a marker kept from before would
        # say a lead is ready whose file was just moved aside.
        _force_quarantine_output(loop.out, print, loop.window.source)
        forced[0] = False
    return fetch_legacy_as_posted(loop, prefix, fresh=force,
                                 publish_lead=publish_verified)


def _resolve_manifest_bridge(source: str) -> Path:
    """The built decoder ``--author-front-door-manifest`` should bind.

    Through :mod:`woof.bridges`, which is where every other consumer
    looks: the environment override, then a checkout's own build, then
    ``libexec/bridges``, then the ``~/.woof/bridges`` that ``woof
    setup`` / ``woof fetch-bridges`` stage into.  Resolving here rather
    than defaulting in argparse keeps the flag's absence meaningful --
    an explicit ``--bridge`` still wins, and still fails loudly when it
    names a file that is not there.
    """

    from woof import bridges

    found = bridges.find_bridge(bridges.SOURCE_DECODERS[source])
    if found is None:
        raise ValueError(
            f"--author-front-door-manifest needs the built "
            f"{bridges.SOURCE_DECODERS[source]}, and none is resolvable on "
            "this install -- run `woof fetch-bridges` (or pass --bridge "
            "PATH), then re-run this command; `woof doctor` prints the "
            "exact steps for this install")
    return found


#: Keys an advisory ``[fetch]`` table may carry (mirroring the CLI flags).
#: The table is emitted by ``woof domain`` and validated -- never silently
#: ignored -- by the experiment loaders, which split it off before the
#: strict experiment schema runs.
#:
#: Each key is one declared row (:mod:`woof.config_keys`): no dataclass
#: carries these hints, so the rows are where a front end reads their
#: types, and :func:`validate_fetch_hints` checks every value against its
#: row.  The key set is read off the rows.
FETCH_HINT_ROWS = key_rows(
    KeyRow("fetch_max_connections", "integer", None,
           "box-wide connection cap for engine fetches; default 30, or WOOF_FETCH_MAX_CONNECTIONS"),
    KeyRow("source", "string", None,
           "the source to acquire, a registry id or alias", required=True),
    KeyRow("initial_source", "string", None,
           "separate mapped analysis source at the forecast start; go fetches its analysis"),
    KeyRow("boundary_cycle_offset_hours", "integer", 0,
           "boundary cycle offset from the nominal forecast cycle"),
    KeyRow("boundary_lead_offset_hours", "integer", 0,
           "boundary start lead offset; cancels the boundary cycle offset"),
    KeyRow("cycle", "string", None,
           "the forcing cycle, YYYY-MM-DDTHH or 'latest'; absent means "
           "any cycle the source publishes"),
    KeyRow("hours", "integer", None,
           "forecast hours of boundaries to fetch after the start lead"),
    KeyRow("area", "string", None,
           "the crop box, south,west,north,east in degrees"),
    KeyRow("point", "string", None,
           "the crop centre, lat,lon in degrees; needs radius_km"),
    KeyRow("radius_km", "number", None,
           "the crop radius around point, in kilometres; needs point"),
    KeyRow("out", "string", None,
           "the download directory"),
    KeyRow("wif", "boolean", False,
           "stage the pinned monthly aerosol climatology before forcing transfer"),
    KeyRow("cadence", "integer", None,
           "hours between boundary times; absent takes the source's own"),
    KeyRow("forecast_start_hour", "integer", 0,
           "the forecast lead the run starts from; 0 is the analysis"),
    KeyRow("source_root", "string", None,
           "the directory holding a local source's input files"),
    KeyRow("era5_provider", "string", "cds",
           "ERA5 only: 'cds' (Copernicus, keyed) or 'arco' (public store)"),
    KeyRow("era5_product", "string", "reanalysis",
           "ERA5 only: 'reanalysis' or 'ensemble_members'"),
    KeyRow("reuse_fetch", "string", None,
           "a completed generated fetch root to verify and reuse when it covers this request"),
    KeyRow("member", ("string", "integer"), None,
           "the ensemble member to fetch; absent is the route's control"),
    KeyRow("retrieve", "boolean", False,
           "ERA5 only: download and verify now instead of writing a "
           "retrieval template"),
    KeyRow("mode", "string", None,
           "the byte transport, the value `woof fetch --mode` takes: "
           "'full-file', 'idx-subset' or 'auto'; absent keeps the fetch "
           "default (whole files).  Lets the preparation route's own fetch "
           "take the .idx-selected records, as `woof fetch --mode "
           "idx-subset` does"),
    KeyRow("transport", "string", None,
           "the one host of the source's endpoint ladder to download "
           "from, the value `woof fetch --transport` takes; absent walks "
           "the ladder"),
    KeyRow("as_posted", "boolean", True,
           "fetch, prepare and run as the leads post, and take 'latest' "
           "as the newest cycle whose first leads are posted; false is "
           "the whole-cycle rule (`woof fetch --whole-cycle`)"),
    KeyRow("late_after_minutes", "number", None,
           "how far past its scheduled time a lead may be before the run "
           "stops with exit 75; absent takes the source table's posting "
           "late_after_minutes"),
    KeyRow("runtime_surface", "string", None,
           "comma-separated runtime surface fields (e.g. 'VEGFRA') the "
           "start must read from the cycle itself; absent, a field the "
           "source table declares starts from its recorded fallback when "
           "a cycle publishes no record for it"),
    *(KeyRow(*hint) for hint in dict.fromkeys(
        hint for adapter in source_adapters.source_adapters()
        for hint in adapter.acquisition_hints)),
)
FETCH_HINT_KEYS = frozenset(FETCH_HINT_ROWS)


def transport_refusal(source: str, transport: object) -> str | None:
    """Why ``transport`` cannot be pinned for SOURCE, or None when it can.

    The answers ``woof fetch --transport`` gives, asked of the same
    tables, so a ``[fetch] transport`` key and a ``woof go --transport``
    flag are refused at config load instead of by the fetch stage after
    the chain has started.  The command line keeps its own checks, which
    see ``--mode``; a table has no mode key, so a GFS or GDAS table always
    means the grib-filter crop, which takes no host.
    """

    from woof.source_drivability import drivability_for

    if not isinstance(transport, str) or transport not in FETCH_TRANSPORTS:
        return (f"transport = {transport!r} is not a host `woof fetch "
                f"--transport` takes; it takes one of {list(FETCH_TRANSPORTS)}")
    if (drivability_for(source) or {}).get("requires_source_root"):
        return ("transport names a download host but these inputs are "
                "already local. what to do: remove transport.")
    name = fetch_routes.canonical_source(source)
    if name in fetch_routes.route_ids():
        try:
            fetch_routes.route_for(name).host(transport)
        except ValueError as error:
            return str(error)
        return None
    if not fetch_endpoints.has_ladder(name):
        return (f"transport: {name} has no host to choose between, so there "
                "is nothing to pin. what to do: remove transport.")
    if name in GFS_CONTAINER_SOURCES:
        try:
            endpoint = fetch_endpoints.endpoint_named(name, transport)
        except ValueError as error:
            return str(error)
        if endpoint.layouts or fetch_endpoints.policy_uses_aws(name):
            return None
        return ("transport pins whole archive objects; choose an archive "
                "host declared in the source table or remove transport.")
    if name == "hrrr":
        if transport not in HRRR_TRANSPORTS:
            return (f"unknown HRRR transport {transport!r}; expected one of "
                    f"{HRRR_TRANSPORTS}")
        return None
    pinned = pinned_host(transport)
    if pinned is not None:
        try:
            fetch_endpoints.endpoint_named(name, pinned)
        except ValueError as error:
            return str(error)
    return None


def _fetch_hint_sources() -> tuple[str, ...]:
    """Sources a ``[fetch]`` table may name -- one definition, derived.

    The same seam the wizard emits through
    (:func:`fetch_front_door_sources`), so the validator cannot refuse a
    table the emitter just wrote.  It was a hand-typed 4-tuple, and that
    is exactly the drift that shipped: ten table routes opened and every
    hand-written ``[fetch]`` table naming one of them failed to load.
    """

    from woof.source_drivability import intent_drivability
    local = {name for name, verdict in intent_drivability().items()
             if verdict.get("requires_source_root")}
    return tuple(sorted(set(fetch_front_door_sources()) | local))


def _source_reaches_forecast_leads(source: str) -> bool:
    """Does SOURCE publish forecast leads, or only analyses?

    The registry's ``max_forecast_hour`` is the whole answer, so
    ``forecast_start_hour`` is gated on the row rather than on a spelled
    ``{"gfs", "gdas", "hrrr"}`` -- which refused RAP a lead RAP publishes
    51 hours of.  An unregistered name is not this function's question;
    the caller has already proved the source is fetchable.
    """

    try:
        return source_adapters.get_source_adapter(
            source).max_forecast_hour > 0
    except ValueError:
        return False


#: What a caller passes as ``source`` when the hints came from FLAGS.  The
#: same validator serves the ``[fetch]`` table and the command line so the
#: two doors cannot disagree about one window, and it names where the hints
#: came from; naming a table on a command line reads as a diagnostic about a
#: file the user never wrote.
COMMAND_LINE_HINTS = "command line"


def validate_fetch_hints(table: dict, *, source: str) -> None:
    """Validate advisory acquisition hints without network or filesystem I/O.

    Partial hints remain legal, but every supplied window must have a possible
    interpretation on the source's actual ladder. A named cycle is checked
    exactly; `latest` or an omitted cycle may use any declared cycle hour.
    """
    prefix = ("the fetch command line" if source == COMMAND_LINE_HINTS
              else f"[fetch] of {source}")
    if not isinstance(table, dict):
        raise ValueError(f"{prefix} must be a table of scalar hint keys")
    unknown = sorted(set(table) - FETCH_HINT_KEYS)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in {prefix}; known keys: {sorted(FETCH_HINT_KEYS)}")
    if table.get("mode") is not None and table["mode"] not in FETCH_MODES:
        raise ValueError(f"mode = {table['mode']!r} in {prefix} is not a fetch mode; "
                         f"what to do: write one of {', '.join(repr(m) for m in FETCH_MODES)}.")
    from woof.initial_source import boundary_fetch_request
    table = boundary_fetch_request(table)
    # Common spellings precede every source, product and local-input branch.
    for key, minimum in (("hours", 0), ("forecast_start_hour", 0), ("cadence", 1)):
        value = table.get(key)
        if value is not None and (type(value) is not int or value < minimum):
            example = minimum or 6
            try:
                numeric = float(value)
                if math.isfinite(numeric) and numeric.is_integer() and numeric >= minimum:
                    example = int(numeric)
            except (ValueError, TypeError, OverflowError):
                pass
            raise ValueError(f"{key} = {value!r} in {prefix} must be a whole number of hours, "
                "written without quotes or a decimal point. "
                f"The window requires {'positive' if minimum else 'nonnegative'} integers; "
                f"what to do: write {key} = {example}.")
    # Every value against its declared row, before any source reasoning:
    # the rows are what a front end is told these keys take.
    for key, value in table.items():
        FETCH_HINT_ROWS[key].check(value, where=prefix)
    if table.get("fetch_max_connections") is not None:
        from woof.fetch_connections import max_connections
        max_connections(table["fetch_max_connections"])
    if table.get("runtime_surface") is not None and isinstance(table.get("source"), str):
        from woof.runtime_surface_fetch import declared_runtime_surface_fields
        from woof.source_adapters import get_source_adapter
        try:
            adapter = get_source_adapter(table["source"])
        except (KeyError, ValueError):
            adapter = None
        try:
            declared_runtime_surface_fields(table["runtime_surface"], adapter)
        except ValueError as error:
            raise ValueError(f"{prefix}: {error}") from None
    late = table.get("late_after_minutes")
    if late is not None and not (math.isfinite(float(late)) and float(late) > 0):
        # A zero budget fails every lead a second late (DESIGN A136 2.1).
        raise ValueError(
            f"late_after_minutes = {late!r} in {prefix} must be a positive "
            "number of minutes")
    if late is not None and table.get("as_posted") is False:
        raise ValueError(
            f"late_after_minutes in {prefix} is the wait budget of an "
            "as-posted fetch, and as_posted = false waits for nothing")
    known = _fetch_hint_sources()
    local_source = False
    if isinstance(table.get("source"), str):
        from woof.source_drivability import local_input_requested
        local_source = local_input_requested(table)
        # An unregistered name is refused below with the hint table's own
        # "not one of" list; resolving it here raised the registry's
        # lookup error first, so the [fetch] table's refusal never fired.
        try:
            owner = source_adapters.get_source_adapter(table["source"]).acquisition_owner
        except ValueError:
            owner = None
        # An owner can author an acquisition request and still prepare an
        # existing local directory. Only explicit local tables prohibit crops.
        if owner and not table.get("source_root"):
            local_source = False
    if "source" not in table:
        raise ValueError(f"{prefix} must carry source = {'|'.join(known)}")
    name = fetch_routes.canonical_source(str(table["source"]))
    if name not in known or (name not in fetch_front_door_sources() and not local_source):
        raise ValueError(f"source = {name!r} in {prefix} is not one of {known}")
    try:
        model = table.get("model")
        adapter = source_adapters.get_source_adapter(name)
        if model is not None:
            from importlib import import_module
            resolve = (getattr(import_module(adapter.acquisition_owner), "acquisition_hints", None)
                       if adapter.acquisition_owner else None)
            if resolve is None:
                raise ValueError("model requires a registry acquisition owner with model selection")
            resolve(str(table["source"]), model)
        cadence = table.get("cadence")
        for key in ("cycle", "area", "point", "out", "source_root"):
            if key in table and not table[key].strip():
                raise ValueError(f"{key} must be a nonempty string")
        raw_cycle = table.get("cycle")
        cycle = parse_cycle(raw_cycle, name) if raw_cycle not in (None, "latest") else None
        if "retrieve" in table and not source_adapters.get_source_adapter(name).fetch_requires_retrieve:
            raise ValueError(_retrieve_inapplicable_refusal(name))
        era5_keys = {"era5_provider", "era5_product"} & table.keys()
        if era5_keys and name != "era5":
            raise ValueError(f"{sorted(era5_keys)} apply to ERA5 only")
        if name == "era5":
            from woof.era5_member import validate_selection
            validate_selection(
                product_type=FETCH_HINT_ROWS["era5_product"].get(table, where=prefix),
                member=table.get("member"),
                provider=FETCH_HINT_ROWS["era5_provider"].get(table, where=prefix),
                cadence=6 if cadence is None else cadence, cycle=cycle)
            if table.get("era5_product") == "ensemble_members" and table.get("retrieve") is not True:
                raise ValueError("ERA5 EDA requires retrieve = true (--retrieve) so native member "
                                 "verification checks the selected payload before publication.")
        elif name in fetch_routes.route_ids():
            fetch_routes.resolve_member(fetch_routes.route_for(name), table.get("member"))
        elif "member" in table:
            adapter = source_adapters.get_source_adapter(name)
            if adapter.acquisition_owner:
                from importlib import import_module
                validator = getattr(import_module(adapter.acquisition_owner), "validate_member", None)
            else:
                validator = None
            if validator is None:
                raise ValueError(f"{name} has no acquisition member axis. Omit member or select an ensemble product.")
            selection = adapter.selection_owner
            if selection and table.get("earth2_members"):
                getattr(import_module(selection), "validate_member")(table["member"], cycle=cycle)
            else:
                validator(name, table["member"])

        area, point, radius = (table.get(key) for key in ("area", "point", "radius_km"))
        crop_keys = sorted(key for key in ("area", "point", "radius_km") if key in table)
        if local_source:
            if crop_keys:
                raise ValueError(f"{', '.join(crop_keys)} names a crop on {name} local inputs. "
                    "what to do: remove these keys from [fetch]; woof prep maps the files onto the namelist geometry.")
            if "out" in table:
                raise ValueError("out names a download destination but these inputs are already local. "
                    "what to do: remove out and set source_root to the input directory.")
        elif "source_root" in table:
            raise ValueError("source_root names local inputs but this source is downloaded. "
                "what to do: remove source_root and use out or --data-dir for the download destination.")
        if "transport" in table:
            refusal = transport_refusal(name, table["transport"])
            if refusal is not None:
                raise ValueError(refusal)
        validate_fetch_cadence(name, cadence)
        if cadence is not None and not fetch_accepts_cadence(name):
            raise ValueError(cadence_inapplicable_refusal(name))
        if cadence is not None:
            refusal = preparation_cadence_refusal(name, cadence)
            if refusal is not None:
                raise ValueError(refusal)
        if crop_keys and not fetch_accepts_area(name):
            raise ValueError(f"{', '.join(crop_keys)} names a crop `woof fetch --source {name}` refuses. "
                "This source publishes whole objects without a subsetting service; "
                "`woof prep` maps them onto the namelist geometry.")
        if area is not None and (point is not None or radius is not None):
            raise ValueError("--area and --point/--radius-km are mutually exclusive")
        if (point is None) != (radius is None):
            raise ValueError("--point requires --radius-km, and --radius-km requires --point")
        if area is not None:
            validate_fetch_area(name, parse_area(area))
        elif point is not None:
            validate_fetch_area(name, area_from_point(point, float(radius)))

        hours = table.get("hours")
        start = FETCH_HINT_ROWS["forecast_start_hour"].get(table, where=prefix)
        if start and not _source_reaches_forecast_leads(name):
            raise ValueError(f"forecast_start_hour applies to forecast leads; {name} declares "
                             "max_forecast_hour = 0 and publishes analyses, not forecasts")
        if name in fetch_routes.route_ids():
            route = fetch_routes.route_for(name)
            # Pure grammar only. resolve_request's host/retention decision is
            # deliberately left to acquisition, when a real cycle is selected.
            cycles = ((cycle,) if cycle is not None else
                      fetch_routes.planning_cycles(route))
            # A table that names no start opens at the route's first
            # published lead: the GEFSv12 reforecast (first_lead = 3,
            # lane/cap-gefs-archive) publishes no f000, so checking f000
            # refused every bare [fetch] source of it.
            route_start = (start if "forecast_start_hour" in table
                           else max(start or 0, route.first_lead))
            errors = []
            for candidate in cycles:
                try:
                    fetch_routes.resolve_cycle(route, candidate)
                    fetch_routes.resolve_leads(route, candidate, 0 if hours is None else hours,
                                               cadence=cadence, start_hour=route_start)
                    break
                except ValueError as error:
                    errors.append(error)
            else:
                raise errors[-1]
        elif name == "era5":
            step = 6 if cadence is None else cadence
            if hours is not None:
                _era5_times(cycle or datetime(2000, 1, 1), hours, step)
        elif native_cf_fetch_contract(name) is not None and not local_source:
            from woof.cf_archive_fetch import validate_window
            if hours is not None:
                validate_window(name, cycle or datetime(2000, 1, 1), hours, cadence)
        elif name in GFS_CONTAINER_SOURCES:
            step = container_default_cadence(name) if cadence is None else cadence
            if name == "gdas":
                gdas_forecast_hours(step if hours is None else hours, step, start)
            else:
                gfs_forecast_hours(step if hours is None else hours, step, start)
        elif name == "hrrr":
            if cadence is not None and cadence != 1:
                raise ValueError("HRRR is hourly; --cadence must be 1 or omitted")
            # A latest/unspecified request may select a long synoptic cycle.
            hrrr_forecast_hours(1 if hours is None else hours, cycle or datetime(2000, 1, 1), start)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{prefix}: {error}") from error


def source_argument(value: str) -> str:
    """``--source`` as a registry id, or the refusal that names why not.

    argparse ``choices`` would answer a registered-but-unfetchable name
    with "invalid choice", which says nothing about WHY -- and the two
    reasons are different: a private archive wants ``--source-root``, a
    non-runnable row wants nothing at all.  Resolving here keeps each
    refusal in the route table's own words, and it makes every registry
    alias (``gdps``, ``ifs``, ``hrrr-wrfprs``) spell its source.
    """

    import argparse as _argparse

    name = str(value).strip().lower().replace("_", "-")
    try:
        source_id = source_adapters.get_source_adapter(name).source_id
    except ValueError:
        source_id = name
    if source_id in fetch_routes.all_fetchable_sources():
        return source_adapters.canonical_source_token(value)
    try:
        if source_adapters.get_source_adapter(source_id).acquisition_owner:
            # Its bytes are produced by running something (a model), not
            # downloaded; the row's owner module is the door.
            return source_id
    except ValueError:
        pass
    try:
        fetch_routes.route_for(source_id)
    except ValueError as error:
        raise _argparse.ArgumentTypeError(str(error)) from error
    raise _argparse.ArgumentTypeError(
        f"--source {value}: no fetch route in this WOOF")


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "fetch",
        help="download initialization/boundary data for any registered "
             "source with public bytes; native GDAS uses its mapped preparation "
             "profile; retrieve ERA5 from the public ARCO store without a key, "
             "or from Copernicus CDS with configured credentials")
    parser.add_argument("--reuse-fetch", default=None, metavar="DIR",
        help="reuse a verified completed generated fetch covering this request before loading its model")
    parser.add_argument("--model", default=None, metavar="MODEL",
        help="--source earth2: which NVIDIA Earth-2 model to run through "
             "earth2studio (fcn3, sfno, pangu, aurora, fengwu, graphcast, aifs: "
             "a row of woof/authorities/earth2-models.v1.json); the forecast "
             "and the ERA5 analysis at --cycle are written to --out")
    parser.add_argument("--earth2-members", help="complete comma-separated Earth-2 noise-id roster")
    parser.add_argument("--earth2-batch-root", help="shared generated-member batch folder")
    parser.add_argument("--earth2-batch-size", type=int, help="members per admitted GPU sub-batch")
    parser.add_argument("--earth2-weights-package", help="local pinned Earth2Studio weight package")
    parser.add_argument("--prepare-only", action="store_true",
        help="Earth-2 sources: run only the CPU stage (weights, CPU model construction, initial "
             "conditions, donor) with no GPU, so it can run outside a card lease; the same "
             "command with a card then generates from it. Implied when CUDA devices are hidden")
    parser.add_argument("--retrieve", action="store_true",
        help="ERA5: download and validate with the selected provider (default CDS); otherwise write a CDS retrieval template")
    parser.add_argument("--era5-provider", choices=("cds", "arco"), default=None,
        help="ERA5 provider: cds uses Copernicus credentials; arco downloads Google's public hourly ERA5 Zarr archive without a key")
    parser.add_argument("--era5-product", choices=("reanalysis", "ensemble_members"), default=None,
        help="ERA5 product: reanalysis (default), or ten-member EDA with explicit --member 0..9 --retrieve, on a cadence that is a whole multiple of its three-hourly clock")
    parser.add_argument(
        "--source", required=True, type=source_argument, metavar="MODEL",
        help="public data source: "
             + ", ".join(fetch_routes.all_fetchable_sources())
             + ".  Registry aliases work too (gdps, ifs, hrrr-wrfprs).  A "
               "registered source with no public bytes -- the 20CRv3 "
               "every-member archive, the generic 'mapped' adapter -- "
               "refuses by name and points at `woof prep --source-root`")
    parser.add_argument(
        "--member", default=None, metavar="ID",
        help="ERA5 EDA: required encoded member 0..9. Ensemble routes (gefs, aigefs): which member to fetch "
             "(default the control).  Member identity is a PATH component "
             "for these products, so the files land under their declared "
             "upstream-relative paths and `woof-member-prep --inputs` "
             "reads the directory as published")
    parser.add_argument(
        "--cycle", default=None, metavar="YYYY-MM-DDTHH|latest",
        help="model cycle (UTC); 'latest' resolves the newest cycle this "
             "source can serve, from the initialization grid and "
             "publication lag its registry row or route declares -- "
             "probed against the mirrors where the source publishes "
             "objects to probe, and taken from the declared lag where it "
             "does not (a reanalysis published on a delay has a latest, "
             "and it is that delay).  A source that declares neither is "
             "refused by name")
    parser.add_argument(
        "--analyzed-aerosol", action="store_true",
        help="the preparation reads the source's analyzed aerosol numbers "
             "(QNWFA/QNIFA), so a subset fetch selects them too; implied by "
             "the native forecast chain when an mp=28 domain asks for the "
             "analyzed aerosol")
    parser.add_argument(
        "--wif", action="store_true",
        help="stage the SHA-256-verified monthly aerosol climatology in its "
             "shared cache before forcing transfer; implied by the native "
             "forecast chain when its selected physics needs that dataset")
    parser.add_argument(
        "--hours", type=int, default=None, metavar="N",
        help="forecast window length: hours 0..N are fetched.  gdas is "
             "certified for fetch and decode through "
             f"f{GDAS_MAX_FORECAST_HOUR:03d}; native mapped GDAS preparation "
             "uses the complete pressure ladder and specific humidity. --hours 0 is "
             "one analysis on each acquisition route; it is also how a "
             "hybrid source's donor is fetched. Forecast preparation still "
             "needs at least two forcing times. A "
             "window past the cycle's own horizon refuses and names both "
             "the horizon and which cycles reach farther")
    parser.add_argument(
        "--area", default=None, metavar="LAT0,LON0,LAT1,LON1",
        help="bounding box corners in degrees (order free); allow several "
             "degrees of margin beyond the outer domain -- for gfs, "
             f"{GFS_LAKE_DONOR_MARGIN_DEG:g} deg, so every model lake's "
             "nearest source-water donor lies inside the crop (a lake "
             "whose nearest donor may lie outside it is counted; `woof "
             "domain` suggests areas with this margin built in)")
    parser.add_argument(
        "--point", default=None, metavar="LAT,LON",
        help="center point; requires --radius-km")
    parser.add_argument(
        "--radius-km", type=float, default=None, metavar="KM",
        help="half-width of the box around --point")
    parser.add_argument(
        "--out", type=Path, default=None, metavar="DIR",
        help="output directory (created; complete files are skipped on "
             "re-run)")
    parser.add_argument(
        "--cadence", type=int, default=None, metavar="HOURS",
        help="forecast-hour cadence: gfs any positive whole-hour spacing whose "
             f"requested leads are published (default "
             f"{container_default_cadence('gfs')}); gdas any "
             "whole number of hours that divides --hours, on its hourly "
             f"f{GDAS_PUBLISHED_HOURS[0]:03d}..f{GDAS_MAX_FORECAST_HOUR:03d} "
             f"ladder (default {container_default_cadence('gdas')}; --hours 0 "
             "is a single lead, which a "
             "cadence has nothing to space); era5 any positive whole number of "
             "hours that divides --hours (default 6; the EDA product "
             "publishes 3-hourly, so it takes multiples of 3); hrrr is "
             "hourly.  On a table route any whole number of hours at "
             "which the cycle's ladder publishes every lead of the window "
             "and the preparation takes the series, default the row's own "
             "-- a cadence off the publisher's ladder refuses and names "
             "the ladder")
    parser.add_argument(
        "--validate", type=Path, nargs="+", default=None, metavar="GRIB",
        help="era5 only: validate user-supplied GRIB1 file(s) against "
             "what woof ingest expects instead of fetching")
    parser.add_argument(
        "--transport", default=None, choices=FETCH_TRANSPORTS,
        help="pin one rung of the source's endpoint ladder.  Every NCEP "
             "source declares an ORDERED ladder -- the operational "
             "server (nomads.ncep.noaa.gov) while it still holds the "
             "cycle, the AWS archive behind it -- and the default walks "
             "it.  Retention decides which rungs are asked: a cycle "
             "older than the operational window goes straight to the "
             "archive.  Throughput decides which one serves: each "
             "requested object is HEADed on the archive first and taken "
             "there when the archive already has it, because the "
             "operational server's head start is spent once both hosts "
             "have the same bytes; an object the archive has not caught "
             "up with comes from the operational server.  A refusal, a "
             "403/503 or a Retry-After moves to the next rung either "
             "way.  Where a source's ladder carries both hosts they "
             "serve byte-identical objects under identical keys, so the "
             "choice never changes the data, except for AI-GEFS: NOMADS "
             "marks each member with ensemble type 6 where the AWS copy "
             "of the same member says 3, the AWS surface files carry an "
             "extra surface pressure record, and the AWS pressure-level "
             "files are repacked copies whose heights sit within 0.08 "
             "gpm of the NOMADS ones.  Preparation reads AI-GEFS from "
             "either host the same way and derives surface pressure "
             "itself on both.  Naming a host here is a decision: it "
             "skips the probe, disables fall-through, and refuses in "
             "that host's own words.  A host a source does not carry "
             "refuses and lists the ones it does, because for some "
             "products the second "
             "copy is a DIFFERENT product (see `woof fetch --source "
             "aigfs`)")
    posting = parser.add_mutually_exclusive_group()
    posting.add_argument(
        "--as-posted", dest="as_posted", action="store_const", const=True,
        default=None,
        help="fetch each lead the moment one host holds all of it, in lead "
             "order, writing <out>/posting/ (schedule.json and one "
             "fNNN.json per verified lead) as the manifest grows; the "
             "default.  --cycle latest is then the newest cycle whose "
             "first leads are posted.  A lead later than its budget stops "
             "the fetch with exit 75 and posting/failed.json; a re-run "
             "resumes from the fetched prefix")
    posting.add_argument(
        "--wait-for", dest="as_posted", action="store_const", const=True,
        help="the same as --as-posted, for every source")
    posting.add_argument(
        "--whole-cycle", dest="as_posted", action="store_const", const=False,
        help="the old rule: --cycle latest is the newest cycle whose final "
             "lead is posted, and a named cycle needs its final lead "
             "before anything moves")
    parser.add_argument(
        "--late-after-minutes", type=float, default=None, metavar="MIN",
        help="how far past its scheduled time a lead may be before the "
             "fetch stops with exit 75 (default: the source table's "
             "posting late_after_minutes)")
    parser.add_argument(
        "--wait-timeout-minutes", type=float, default=None, metavar="MIN",
        help="a cap on the whole window, for every source: exit 75 if the "
             "window is not all in by then, keeping the fetched prefix")
    parser.add_argument(
        "--readiness", action="store_true",
        help="print gpuwm.readiness.v1 for this window on stdout and fetch "
             "nothing: exit 0 ready (or nothing to probe), 75 not yet "
             "(with expected_ready_at and retry_after_seconds), 2 refused "
             "(the window can never start)")
    parser.add_argument(
        "--no-probe", action="store_true",
        help="with --readiness: compute the schedule from the table only "
             "and ask no host")
    parser.add_argument(
        "--force-refetch", action="store_true",
        help="move every existing file in --out aside (nothing is "
             "deleted) and re-download this request.  The receipts go "
             "first -- fetch-manifest.json, SHA256SUMS, the series -- so "
             "an interrupted force can never leave a manifest behind "
             "claiming payloads it has already replaced; then payloads, "
             ".idx indexes, stale parts and anything else in the "
             "directory.  Files already set aside by an earlier "
             "quarantine are left untouched, and subdirectories are "
             "yours.  Required when re-fetching a different area/cycle "
             "into the same --out")
    parser.add_argument(
        "--p-top-pa", type=float, default=None, metavar="PA",
        help="gfs/gdas only: the model top (Pa) the fetched atmosphere "
             "must reach.  The pressure ladder is extended upward along "
             "whatever the live inventory publishes until a level sits "
             "at or above it, so --p-top-pa 5000 fetches the 70 and 50 "
             "hPa levels the certified 100 hPa ladder stops short of.  "
             "Omitted, the certified 21-level ladder is fetched exactly "
             "as before (a 10000 Pa source top).  woof go, run-plan "
             "and the desktop pass the config's own [shared].p_top "
             "when the certified ladder stops below it.  A top the "
             "product cannot serve refuses and names the deepest it can")
    parser.add_argument(
        "--all-levels", action="store_true",
        help="gfs/gdas only: take every isobaric level the product "
             "publishes instead of choosing a ladder.  On the default "
             "NOMADS grib-filter transport this selects every level; "
             "with --mode full-file the whole object already carries "
             "every level and this declares them all for the decode.  "
             "Either way level subsetting stays an opt-in bandwidth "
             "saver rather than a ceiling on the model top")
    parser.add_argument(
        "--engine", default=None, choices=FETCH_ENGINES,
        help="hrrr, and gfs/gdas --mode full-file: which downloader "
             "moves the bytes.  'rust' is "
             "the vendored rw_fetch backbone (parallel range "
             "GETs, .idx coalescing, the cross-process NOMADS rate "
             "governor, a disk cache); 'python' is the stdlib transport "
             "and always works; 'auto' (default) uses the backbone when "
             "it is built")
    parser.add_argument(
        "--mode", default=None, choices=FETCH_MODES,
        help="the byte transport.  hrrr: "
             f"'{HRRR_DEFAULT_MODE}' "
             "is the default -- the .idx record subset the preparation "
             "reads (about 40 %% of the bytes), and the whole object for "
             "a file whose .idx is absent, malformed, or provably "
             "shorter than the object, each file's line saying which; "
             "'full-file' takes every object whole (--engine rust); "
             "'idx-subset' takes only subsets and refuses rather than "
             "degrading when the index cannot carry the selection "
             "(--engine rust).  gfs/gdas: 'full-file' takes the "
             "whole pgrb2.0p25 objects from the S3 archive (either "
             "engine); 'idx-subset' takes from those objects only the "
             "records the grib-filter asks for, by their .idx byte ranges "
             "(an object whose host keeps no index moves whole); omitted "
             "or 'auto', the NOMADS grib-filter crop remains the default "
             "(the archive's records for a cycle the crop host no longer "
             "keeps, and under the hosted AWS policy).  The other "
             "forecast sources take whole objects, and "
             "'auto' there is that same full-file default")
    parser.add_argument(
        "--cache-dir", type=Path, default=None, metavar="DIR",
        help="--engine rust only (hrrr, gfs/gdas --mode full-file): "
             "wx-core disk cache root, keyed "
             "by URL and byte range, so a re-run or an overlapping "
             "window re-reads bytes instead of re-downloading them")
    parser.add_argument(
        "--fetch-max-connections", type=int, default=None, metavar="N",
        help="total engine fetch connections on this box (default 30, or "
             "WOOF_FETCH_MAX_CONNECTIONS). Shared across engine processes; "
             "use the same cap for every process. File and range widths are "
             "clipped to it; receipts record the cap and stream limits.")
    parser.add_argument(
        "--fetch-workers", type=int, default=None, metavar="N",
        help="how many FILES are in flight at once (default "
             f"{fetch_pool.DEFAULT_FILE_WORKERS}; every source but era5, "
             "which is a manual CDS retrieval).  Bounded per host on top of "
             "the pool: NOMADS is capped at "
             f"{fetch_pool.NOMADS_FILE_WORKER_CAP} in-flight requests "
             "and every request still passes the node-wide 2.5 s "
             "spacing governor, so concurrency overlaps service time "
             "without raising the request rate against a fragile "
             "public host.  Every file keeps the exact serial "
             "verification -- envelope walk, record bar, sha256 -- and "
             "one failed file still refuses by name.  1 is the serial "
             "transport: a knob, not a workaround.  The manifest "
             "receipts files, bytes, workers, wall and the effective "
             "speedup under 'concurrency'")
    parser.add_argument(
        "--runtime-surface", default=None, metavar="FIELDS",
        help="comma-separated runtime surface fields (e.g. VEGFRA) the "
             "start must read from this cycle itself.  A field the source "
             "table declares but this list omits starts from its recorded "
             "fallback when the cycle publishes no record for it; a listed "
             "field the cycle does not publish is refused by name")
    parser.add_argument(
        "--accept-inventory-change", action="store_true",
        help="proceed when the live provider inventory yields a "
             "different record count than this WOOF was certified "
             "against.  Without it such a mismatch is a refusal naming "
             "both counts; with it the live count becomes the bar and "
             "the fetch manifest records the acceptance")
    front = parser.add_argument_group(
        "GFS front-door manifest authoring",
        "gfs only: write the gpuwm-gfs-direct-input-manifest-v1 document "
        "the rw-wps GFS front door verifies (name + sha256 for every "
        "role, including the bridge executable) and print the "
        "ready-to-run command.  Runs after the download, or standalone "
        "on an already-fetched --out when --cycle/--hours are omitted.")
    front.add_argument(
        "--author-front-door-manifest", action="store_true",
        help="author the front-door input manifest for the fetched "
             "series; requires --wps-namelist and --experiment-config "
             "(--bridge defaults to the built decoder this install "
             "resolves)")
    front.add_argument(
        "--bridge", type=Path, default=None, metavar="EXE",
        help="built gfs_grib2_bridge executable; omit it and the same "
             "resolver `woof go` uses finds the one this install has "
             "(checkout build, libexec, then ~/.woof/bridges -- see "
             "woof doctor)")
    front.add_argument(
        "--wps-namelist", type=Path, default=None, metavar="WPS",
        help="the namelist.wps the front door will consume (e.g. the "
             "woof domain output)")
    front.add_argument(
        "--experiment-config", type=Path, default=None, metavar="TOML",
        help="the experiment TOML the front door will consume")
    front.add_argument(
        "--static-input", type=Path, default=None, metavar="NPZ",
        help="optional prebuilt static cache (with --static-receipt); "
             "omit when the front door builds statics from --geog-root")
    front.add_argument(
        "--static-receipt", type=Path, default=None, metavar="JSON",
        help="receipt for --static-input")
    front.add_argument(
        "--manifest-out", type=Path, default=None, metavar="JSON",
        help=f"manifest path (default <out>/{GFS_INPUT_MANIFEST_NAME})")
    parser.add_argument(
        "--forecast-start-hour", type=int, default=None, metavar="K",
        help="every forecast source: the forecast lead the window BEGINS at "
             "(default f000, the analysis).  --hours stays the window "
             "length, so --forecast-start-hour 174 --hours 66 fetches "
             "f174..f240 and nothing before it; an experiment whose "
             "start_time is cycle+K is then initialized from f{K} with "
             "its boundaries from f{K+i}.  With "
             "--author-front-door-manifest on an already-fetched --out, "
             "this authors the manifest over that tail of the existing "
             "series instead of re-downloading it")
    parser.set_defaults(func=fetch_main)
    return parser


__all__ = [
    "AREA_HINT_DECIMALS", "Area", "Era5ValidationReport", "FETCH_HINT_KEYS",
    "FETCH_HINT_ROWS",
    "area_bounds_inward", "source_coverage_envelope", "validate_fetch_area",
    "FETCH_ENGINE_SELECTIONS", "FETCH_MANIFEST_SCHEMA",
    "FetchEngineChoice", "GFS_FRONT_DOOR_MANIFEST_SCHEMA",
    "GFS_INPUT_MANIFEST_NAME", "GFS_LAKE_DONOR_MARGIN_DEG",
    "HRRR_DEFAULT_MODE", "FETCH_ENGINES", "FETCH_MODES",
    "HRRR_NOMADS_BASE", "HRRR_NOMADS_RETENTION_HOURS", "HRRR_TRANSPORTS",
    "HRRR_WAIT_POLL_SECONDS", "HRRR_WAIT_TIMEOUT_DEFAULT_MINUTES",
    "validate_fetch_hints", "transport_refusal",
    "GFS_SUBSET_RECORD_COUNT", "Grib1Record", "area_from_point",
    "author_gfs_front_door_manifest", "check_prior_request",
    "preparation_manifest_path",
    "cached_request_complete", "latest_cycle_request", "pinned_host",
    "refuse_changed_on_disk", "resume_digest_refusal",
    "require_matching_request",
    "count_grib2_messages", "era5_request_template", "fetch_gfs",
    "fetch_gfs_fullfile",
    "fetch_hrrr", "fetch_main", "gfs_forecast_hours", "gfs_object_url",
    "gfs_suggested_fetch_margin_deg",
    "hrrr_forecast_hours", "hrrr_lead_products", "hrrr_lead_urls",
    "hrrr_object_url", "parse_area", "parse_cycle",
    "read_grib1_records", "register_cli", "resolve_fetch_engine",
    "select_fetch_engine", "resolve_hrrr_transport",
    "cycle_probe_urls", "objects_published",
    "PublicationCheck", "cycle_publication_check", "cycle_publication_refusal",
    "require_published_cycle",
    "analysis_window_reference",
    "resolve_latest_cycle", "resolve_cycle_for_valid_start",
    "sha256_file", "validate_era5_files", "write_era5_request",
    "read_grib1_grid", "wsl_path", "era5_retrieve_commands", "Grib1Grid",
    "write_fetch_manifest",
]
