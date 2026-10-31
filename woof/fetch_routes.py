"""The one acquisition engine every table-driven fetch route runs through.

``woof fetch --source`` used to accept four names.  Ten sources had a
runnable packaged profile, a passing 6 h forecast and no way to get their
bytes -- the RRFS arm's 6.3 GiB came down by hand-written ``curl``.  A
capability with no front door is engine-proven, not shipped.

This module closes that, and it closes it as TABLE WORK.  Everything
model-shaped -- which host publishes the bytes, what the key looks like,
which cycles the producer runs, how far each one forecasts and at what
spacing, which downloaded objects are the primaries and which are the
composition's supplement or its cross-source donor -- is rows in the
packaged document ``authorities/rw-wps-fetch-routes.v1.json``.  Nothing
here branches on a model name.  Adding ICON-D2, RAP-AK or a hires Euro
model is a row in that file; if it ever needs a function in this one, the
row grammar was wrong.

Three stages, in order:

* **resolve** -- :func:`resolve_request` turns ``--cycle/--hours`` into an
  ordered, deterministic object list with no network at all, so every
  refusal a request has coming (a cycle the producer does not run, a lead
  past the horizon, a member outside the declared set) is paid for in
  milliseconds rather than after a download.
* **transfer** -- :func:`run_plan` moves the objects through
  :mod:`woof.fetch_pool`, so every table route is parallel by default with
  the same bounded, host-capped, in-order-admission semantics the GFS and
  HRRR routes already have.
* **compose** -- the declared per-lead concatenations (GEFS's disjoint
  ``pgrb2a``+``pgrb2b`` pair, GDPS's one-message-per-file valid time) run
  after the bytes land and verify, producing the exact files
  ``woof prep`` consumes.

The output directory is then a front door, not a pile: it carries the
ordered ``--input-list``, the supplement bindings the composition
declares, ``SHA256SUMS``, and a ``prep-command.txt`` whose bound half runs
as printed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import time
from time import perf_counter
from types import MappingProxyType
from typing import Mapping, Sequence

from woof import fetch_endpoints, fetch_pool, source_adapters
from woof.fetch_endpoints import Endpoint
from woof.filesystem_paths import io_path as _io_path
from woof.filesystem_paths import replace_file_with_retry
# Imported as a NAME, not as the module: `run_plan` takes a parameter
# called ``progress`` (the route's status-line sink), which would shadow
# a module of that name inside exactly the function that needs it.
from woof import progress as progress_mod
from woof.progress import ByteCounter
from woof.source_cycles import PublicationRule


#: The packaged acquisition-route document, beside the decode authorities
#: it complements: a mapping says how to READ a source's bytes, a route
#: says where they LIVE.
ROUTE_TABLE_NAME = "rw-wps-fetch-routes.v1.json"
ROUTE_TABLE_SCHEMA = "gpuwm-fetch-routes-v1"

#: SHA-256 of the packaged route table, pinned so a wheel whose data file
#: has drifted from this engine fails loudly at load rather than
#: resolving a key shape nothing was measured against.  Kept in sync by
#: ``tests/test_fetch_routes.py``.
ROUTE_TABLE_SHA256 = (
    # Moved by A136 L3: the NOMADS host row's missing_path (403 for a path
    # under a cycle directory not created yet), merged over the L1
    # follow-ups' lateness budgets, then the HRRR NOMADS rung's "why",
    # which named --wait-for on every default as-posted fetch.  Moved by
    # A159: hrrr-prs, rap, rrfs and icon-d2 offer cadences 1, 3 and 6,
    # every one a spacing their hourly ladders publish and their mappings
    # take.  Hosted AWS endpoint allowances also date deadlines from the
    # 2026-10-02 metadata comparisons without delaying positive probes.
    # Then A154: the measured publication_lag rows name the posting
    # watch without the private machine it ran on (a host name in a wheel).
    # Moved by A173: every row's cadences list is retired (it refused
    # spacings the publisher posts and the decode takes, naming no
    # breakage), and cadence_note states the grammar that replaces it.
    # Native-level atmosphere and analysis-soil routes are table entries,
    # merged with the ecmwf-ens and rrfs-ens input-ensemble rows.
    # Moved by the capabilities landing: the ecmwf-ens historical-archive rows
    # merged over the gfs/gefs archive publication rows.
    # Moved by fix/go-posted-cycle-default: the rap-native and hrrr-native
    # publication_lag rows are measured (they were an unmeasured 1.5 h
    # estimate, so a default `woof go` waited ~40 min for a posted cycle).
    # rrfs-ens fetches its declared companion (the deterministic prslev and
    # 2dfld pair of the same cycle) as non-primary rows bound to the
    # composition's companion data role.
    # Moved by the ecmwf-open-data-0p1 route (IFS HRES open data at 0.1
    # degree, published from the 2026-10-20 cycles).
    # Both, merged for 2.9.7.
    # Moved by fix/U-E8: hrrr-native's soil donor counts its leads from the
    # request's start (lead_origin window_start), so a time-lagged member
    # takes its own run's soil at the start.  Moved by
    # fix/ifs-0p1-first-cycle: ecmwf-open-data-0p1 begins at 2026-10-20 06Z
    # (no 00Z cycle on any mirror) and that first cycle, which publishes no
    # soil layers, is refused before any transfer.  Moved by
    # fix/ifs-open-data-sea-ice: both ECMWF open-data routes declare the
    # same cycle's GFS f000 as their sea-ice donor (seaice_data).  Moved by
    # fix/ecmwf-seaice-aifs-ens: the aifs and ecmwf-ens routes declare it too.
    # Moved by fix/ecmwf-record-subset: the ECMWF open-data file rows take
    # the mapped records by the .index beside each object (record_index),
    # and the four GFS sea-ice donors take ICEC alone by the .idx (records).
    # Moved by fix/ecmwf-ens-record-subset: the ecmwf-ens member row also
    # declares record_index, so a member moves only its mapped records.
    # Moved by fix/hrrr-native-record-subset: the hrrr-native file rows
    # take the mapped records by the .idx (record_index) and its hrrr-prs
    # soil donor takes only its bound soil and terrain records (records).
    # Moved by fix/gefs-record-subset: both halves of the GEFS member pair
    # take the mapped records by the .idx beside them (record_index).
    # Moved by fix/rrfs-record-subset: the rrfs prslev/2dfld rows and the
    # rrfs-ens member and companion rows take the mapped records by .idx
    # line (record_index), the nine RUC soil nodes line by line.
    # Moved by fix/hrrr-prs-record-subset: the hrrr-prs file rows take the
    # records the packaged profile and composition bind by the .idx beside
    # each object (record_index), and record_subset is supported.
    # Moved by fix/aifs-record-subset: the aifs file row takes the mapped
    # records by the ECMWF .index (record_index, gh preferred over the
    # pressure-level z the way the mapping reads them), and record_subset
    # is supported.
    "77e2d33b334057fce82a61da4eca03889dfca1833cc86c80d82f45d0c6e615b8"
)

#: Sources whose acquisition predates the route table and keeps its own
#: hand-written transport in :mod:`woof.fetch`: the certified GFS
#: container pair (a NOMADS grib-filter crop plus an S3 full-file route)
#: and HRRR (two hosts, a live-cycle wait mode, an ``.idx`` subsetter).
#: They are named here so the coverage gate can tell "handled elsewhere"
#: from "not handled".
LEGACY_ROUTE_SOURCES = ("gfs", "gdas", "hrrr", "era5")

#: Receipt schemas.
ROUTE_MANIFEST_SCHEMA = "gpuwm-fetch-route-manifest-v1"

#: Every token a route's path template may spell.  A template naming
#: anything else is a table error, caught at load rather than at the
#: first 404.
PATH_TOKENS = frozenset({
    "YYYY", "MM", "DD", "HH", "YYYYMMDD", "YYYYMMDDHH", "YYYYMMDDHHMMSS",
    "F", "FF", "FFF", "MEMBER", "FIELD", "field_lower", "leveltype",
    "LEVEL", "LEVEL_SUFFIX",
})

_TOKEN_RE = re.compile(r"\{([A-Za-z_]+)\}")


# --------------------------------------------------------------------------
# Table loading
# --------------------------------------------------------------------------

def _table_path() -> Path:
    return Path(__file__).with_name("authorities") / ROUTE_TABLE_NAME


def packaged_route_table_sha256() -> str:
    """SHA-256 of the packaged route table as installed."""

    return hashlib.sha256(_table_path().read_bytes()).hexdigest()


def _load_table() -> Mapping[str, object]:
    document = json.loads(_table_path().read_text(encoding="utf-8"))
    schema = document.get("schema")
    if schema != ROUTE_TABLE_SCHEMA:
        raise ValueError(
            f"{ROUTE_TABLE_NAME} declares schema {schema!r}; this WOOF "
            f"reads {ROUTE_TABLE_SCHEMA!r}")
    return document


def unknown_tokens(template: str) -> tuple[str, ...]:
    """Tokens in ``template`` this engine cannot fill."""

    return tuple(sorted(
        name for name in _TOKEN_RE.findall(template)
        if name not in PATH_TOKENS))


# --------------------------------------------------------------------------
# Row types
# --------------------------------------------------------------------------

#: A route's host row IS an endpoint row: name, base, the retention
#: window it serves, and what it is for.  :mod:`woof.fetch_endpoints`
#: owns the shape and the ladder rules; this alias keeps the older
#: spelling readable where a route talks about "its hosts".
Host = Endpoint


@dataclass(frozen=True)
class FileRow:
    """One family of objects in a route's file set."""

    role: str
    path: str
    primary: bool
    #: ``all`` -- one per requested lead; ``step0`` -- one object at the
    #: CYCLE's step 0 whatever lead the window starts at (the
    #: once-per-cycle invariants a producer publishes at analysis time
    #: alone), fetched with the window's first lead and composed into its
    #: first valid time; ``none`` -- lead-independent.
    leads: str
    axis: str | None
    idx_sidecar: str | None
    #: Leading magic every published object of this family carries.  It is
    #: the cheapest bar that separates a payload from an error page a
    #: proxy served with HTTP 200, and it is declared rather than guessed
    #: from a suffix because ``gec00.t00z.pgrb2a.0p50.f000`` has none.
    magic: str
    #: A JSON-lines index selecting a member from a multi-member object.
    #: The byte ranges are copied unchanged, then verified by Rust inventory.
    member_index: Mapping[str, object] | None = None
    index_suffix_replaces: str | None = None
    #: The records the packaged mapping binds, selected by the publisher's
    #: index beside each object (:mod:`woof.record_index`); None moves the
    #: whole object.
    record_index: Mapping[str, object] | None = None
    lead_min: int = 0
    lead_max: int | None = None


@dataclass(frozen=True)
class ComposeRow:
    kind: str
    roles: tuple[str, ...]
    name: str
    primary: bool
    why: str


DONOR_LEAD_ORIGINS = ("cycle", "window_start")


def _donor_lead_origin(row) -> str:
    origin = str(row.get("lead_origin", "cycle"))
    if origin not in DONOR_LEAD_ORIGINS:
        raise ValueError(f"donor lead_origin {origin!r} is not one of {DONOR_LEAD_ORIGINS}")
    return origin


@dataclass(frozen=True)
class DonorRow:
    source: str
    role: str
    leads: tuple[int, ...]
    cycle: str
    why: str
    #: What ``leads`` count from: ``cycle`` (the donor's own leads, the
    #: same for every request) or ``window_start`` (offsets from the
    #: request's first lead, so a time-lagged request that starts at f03
    #: takes the donor's f03 record, valid when the forecast starts).
    lead_origin: str = "cycle"
    #: The donor records the composition binds, selected by the donor
    #: object's index (:mod:`woof.record_index`); None moves it whole.
    records: Mapping[str, object] | None = None


@dataclass(frozen=True)
class PublicationEra:
    """A dated file layout and its preparation compatibility, all table facts."""

    label: str
    valid_from: datetime
    valid_until: datetime | None
    cycle_hours: tuple[int, ...]
    resolution_degrees: float
    files: tuple[FileRow, ...]
    prep_refusal: str
    steps: tuple[tuple[int, int], ...] = ()


#: One ``publication_lag`` row: the cycle grid's own rule type, so the
#: table and the ``latest`` walk read one shape.  ``hours`` and
#: ``per_lead_hours`` are the EARLIEST posting seen on the ladder head
#: (lead ``L`` at ``hours + per_lead_hours * L`` after the cycle), which
#: is where the probed walk starts; ``late_hours`` is how much later the
#: latest posting seen came, which an answer no probe checks waits for;
#: ``from_lead`` starts a rule part way up the ladder (GEFS's 00Z
#: extension past f384 posts about a day after the rest).
LagRule = PublicationRule


@dataclass(frozen=True)
class Route:
    source_id: str
    label: str
    measured: str
    hosts: tuple[Host, ...]
    host_note: str
    coverage_note: str
    cycle_hours: tuple[int, ...]
    #: MEASURED time between a nominal cycle and this producer's bytes
    #: appearing on the ladder head, per cycle hour and per lead
    #: (:class:`LagRule`, most specific first).  It is what lets
    #: ``--cycle latest`` start its walk at a cycle that plausibly exists
    #: instead of HEAD-ing its way down from a cycle nobody has published
    #: yet.  One number per producer was the defect it replaces: ECMWF
    #: posts its 00/12Z runs an hour after its 06/18Z ones, and a single
    #: 12 h figure made ``latest`` skip a cycle posted five hours earlier.
    publication_lag: tuple[LagRule, ...]
    ladders: tuple[tuple[tuple[int, ...] | None, tuple[tuple[int, int], ...]], ...]
    #: The spacing a window that names no cadence starts from.  Every
    #: other spacing the ladder publishes and the preparation takes is
    #: offered too (A173, :func:`window_spacings`): the row carries no
    #: list of them.
    default_cadence: int
    layout: str
    members: Mapping[str, object] | None
    axes: Mapping[str, tuple[Mapping[str, object], ...]]
    files: tuple[FileRow, ...]
    compose: tuple[ComposeRow, ...]
    donors: tuple[DonorRow, ...]
    record_subset_supported: bool
    record_subset_why: str
    prep: Mapping[str, object]
    first_lead: int = 0
    publication_eras: tuple[PublicationEra, ...] = ()
    #: How a cycle posts and how long a run waits for a lead
    #: (:class:`PostingRow`).  Every loaded route has one: the loader
    #: refuses a row without it.
    posting: "PostingRow | None" = None

    def lag_rule(self, cycle_hour: int, lead: int = 0) -> LagRule:
        """The publication rule for one UTC cycle hour and lead."""

        for rule in self.publication_lag:
            if rule.matches(cycle_hour, lead):
                return rule
        raise ValueError(                      # pragma: no cover - load checks
            f"{self.source_id}: no publication lag for the {int(cycle_hour):02d}Z cycle")

    def publication_lag_hours(self, cycle_hour: int, lead: int = 0, *,
                              settled: bool = False) -> float:
        """Hours after a ``cycle_hour`` cycle until its ``lead`` is published.

        The earliest posting seen, or with ``settled`` the latest.
        """

        return self.lag_rule(cycle_hour, lead).at(lead, settled=settled)

    def host(self, name: str | None) -> Host:
        """One named endpoint, or the head of the ladder.

        The head is what a request with no ``--transport`` starts at;
        which endpoint it IS depends on the cycle's age, so callers
        that have a cycle should ask :func:`endpoint_ladder` instead of
        assuming the table's first row.
        """

        if name is None:
            return self.hosts[0]
        for host in self.hosts:
            if host.name == name:
                return host
        offered = ", ".join(host.name for host in self.hosts)
        raise ValueError(
            f"--transport {name}: --source {self.source_id} publishes on "
            f"{offered}.  {self.host_note or ''}".strip())


#: The lead rules a file row may declare (see :attr:`FileRow.leads`).
FILE_LEAD_RULES = frozenset({"all", "step0", "none"})

#: Where a route's prep supplement comes from.  ``every_input`` names
#: every primary file of the window; ``step0`` names the primary rows'
#: objects at the CYCLE's step 0, fetched in addition (and bound only as
#: the supplement) when the window starts later -- the shape of a
#: producer that publishes its statics in the analysis-step object
#: alone; ``role:NAME`` names a declared file role.  ``first_input`` (the
#: window's first file) is retired: it equalled the step-0 object only
#: for a window starting at f000, and AIFS, its one route, found 0 of
#: 122 messages matching past f000.
SUPPLEMENT_ORIGINS = frozenset({"every_input", "step0"})


def _file_row(raw: Mapping[str, object]) -> FileRow:
    leads = str(raw.get("leads", "all"))
    if leads not in FILE_LEAD_RULES:
        # Named breakage: the planner treats any rule it does not know as
        # "one object per requested lead", so a misspelt or retired rule
        # (``first``, which rendered the window's first lead instead of
        # the analysis step) would plan a 404 at every later start.
        raise ValueError(
            f"{ROUTE_TABLE_NAME}: file role {raw.get('role')!r} declares lead "
            f"rule {leads!r}; the planner reads {sorted(FILE_LEAD_RULES)}")
    return FileRow(
        lead_min=int(raw.get("lead_min", 0)),
        lead_max=int(raw["lead_max"]) if raw.get("lead_max") is not None else None,
        role=str(raw["role"]),
        path=str(raw["path"]),
        primary=bool(raw.get("primary", False)),
        leads=leads,
        axis=(str(raw["axis"]) if raw.get("axis") else None),
        idx_sidecar=(str(raw["idx_sidecar"]) if raw.get("idx_sidecar")
                     else None),
        magic=str(raw.get("magic", "GRIB")),
        member_index=(MappingProxyType(dict(raw["member_index"]))
                      if raw.get("member_index") else None),
        index_suffix_replaces=(str(raw["index_suffix_replaces"])
                               if raw.get("index_suffix_replaces") else None),
        record_index=_record_index(raw),
    )


def _record_index(raw: Mapping[str, object]) -> Mapping[str, object] | None:
    """A file row's ``record_index``, checked at load.

    Named breakage: a selection with no index sidecar would plan range
    requests against an index URL the table never built.  Beside a member
    index the selection narrows that member's records (both read the same
    JSON-lines index, so one pass selects); it cannot sit beside a wgrib2
    lead index or select by an NCEP ``.idx``, which would read a second,
    different index of the same object.
    """
    if not raw.get("record_index"):
        return None
    from woof import record_index

    label = f"{ROUTE_TABLE_NAME}: file role {raw.get('role')!r}"
    if not raw.get("idx_sidecar"):
        raise ValueError(f"{label} selects records but declares no idx_sidecar")
    declaration = record_index.validate_declaration(raw["record_index"], label)
    member = raw.get("member_index")
    if member and (member.get("format") == "wgrib2"
                   or declaration["format"] != "ecmwf-jsonl"):
        raise ValueError(f"{label} narrows a member by record selection, which needs a "
                         "JSON-lines member index and an ecmwf-jsonl selection")
    if member and declaration.get("prefer"):
        # The member narrowing keeps every record a clause keeps; it would
        # move both of the records a prefer order chooses between.
        raise ValueError(f"{label} orders a member's records by prefer, which the "
                         "member index transfer does not apply")
    return MappingProxyType(declaration)


def _donor_records(source_id: str, raw: Mapping[str, object]
                   ) -> Mapping[str, object] | None:
    """A donor row's ``records`` selection, checked at load."""
    if not raw.get("records"):
        return None
    from woof import record_index

    return MappingProxyType(record_index.validate_declaration(
        raw["records"], f"{ROUTE_TABLE_NAME}: route {source_id} donor "
        f"{raw.get('role')!r}"))


def _lag_rules(source_id: str, raw: Mapping[str, object]) -> tuple[LagRule, ...]:
    """A route's ``publication_lag`` rows, checked at load.

    Named breakage: a cycle hour no rule matches would make ``latest``
    raise inside a page request for that one hour, a negative lag would
    start the walk at a cycle that has not been run yet, and a rule that
    an earlier one shadows for every hour it names would read as a
    measured fact that nothing ever asks.
    """

    rows = raw.get("publication_lag")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"{ROUTE_TABLE_NAME}: route {source_id} declares no publication_lag "
            "rows, so `--cycle latest` has no cycle to start its walk at")
    rules = []
    for row in rows:
        hours = row.get("cycle_hours")
        try:
            rule = LagRule(
                cycle_hours=None if hours is None else tuple(int(hour) for hour in hours),
                hours=float(row["hours"]),
                per_lead_hours=float(row.get("per_lead_hours", 0.0)),
                late_hours=float(row.get("late_hours", 0.0)),
                from_lead=int(row.get("from_lead", 0)),
                measured=str(row.get("measured", "")))
        except ValueError as error:
            raise ValueError(
                f"{ROUTE_TABLE_NAME}: route {source_id} declares an unreadable "
                f"or negative publication_lag row ({error})") from None
        rules.append(rule)
    for index, rule in enumerate(rules):
        named = raw["cycle_hours"] if rule.cycle_hours is None else rule.cycle_hours
        if not any(next(earlier for earlier in rules
                        if earlier.matches(hour, rule.from_lead)) is rule
                   for hour in named):
            raise ValueError(
                f"{ROUTE_TABLE_NAME}: route {source_id}'s publication_lag row "
                f"{index} is shadowed by an earlier row for every cycle hour "
                "it names; put the rule for the later leads first")
    unmatched = [hour for hour in raw["cycle_hours"]
                 if not any(rule.matches(hour, 0) for rule in rules)]
    if unmatched:
        raise ValueError(
            f"{ROUTE_TABLE_NAME}: route {source_id} declares no publication "
            f"lag for its {', '.join(f'{int(h):02d}' for h in unmatched)}Z cycles")
    return tuple(rules)


# --------------------------------------------------------------------------
# Posting rows: how a source's cycle can be waited on
# --------------------------------------------------------------------------

#: What a cycle's posting looks like to a reader who wants to run it as
#: it posts.  ``rolling``: lead by lead, in lead order, so a run can
#: start on its first leads and wait for the rest.  ``whole_cycle``:
#: every lead appears at once.  ``donor_gated``: the source's own leads
#: are out before a same-cycle donor the preparation needs, so the donor
#: decides the start.  ``archive``: every time already exists or never
#: will.  ``brokered``: a keyed job produces the file, so there is no
#: object to ask for.  ``private``: the bytes are not published.
POSTING_SHAPES = frozenset({
    "rolling", "whole_cycle", "donor_gated", "archive", "brokered", "private"})
#: The shapes a run waits on, lead by lead or for the whole cycle: the
#: ones that need a readiness check, a lateness budget and a poll ceiling.
WAITED_POSTING_SHAPES = frozenset({"rolling", "whole_cycle", "donor_gated"})
#: How one lead is asked about.  ``objects``: HEAD 2xx on every object of
#: the lead, on one host.  ``objects_and_index``: the same plus each
#: object's declared index sidecar.
READY_CHECKS = frozenset({"objects", "objects_and_index"})
#: A readiness check named for a later round (one directory listing per
#: lead, for the hosts that publish one object per field) and not built.
LATER_READY_CHECKS = frozenset({"listing"})
#: The fastest a lead may be asked about again, seconds.  NOMADS answers
#: a client that polls faster with its rate limiter (``host_policy``).
MIN_POLL_SECONDS = 5.0
#: How far past its row's measured late spread (:func:`late_spread_minutes`)
#: a shipped lateness budget reaches, minutes.  The spreads come from
#: watches of 5 to 32 cycles over two days, and one late day (30 Sep,
#: NCEP) set most of them, so a later day can post later than any cycle
#: seen; the margin keeps that cycle running, at the cost of a lead that
#: never posts holding its run this much longer before it is called
#: late.  Each shipped budget is the spread plus this margin, rounded up
#: to 5 min, and never under the design's adopted budget (60 min; 90
#: for rrfs, gdas, aigfs and aigefs), which is a floor
#: (tests/test_source_posting_rows.py holds every row to that rule).
#: The load check refuses only a budget under the spread itself.
LATE_BUDGET_MARGIN_MINUTES = 30.0


def late_spread_minutes(rules: Sequence[LagRule]) -> float:
    """The longest a lead of these rules posted after its scheduled line, minutes.

    The largest ``late_hours`` over the rules: how much later than the
    earliest posting seen (each lead's ``expected_at``, from which the
    budget counts) the latest posting seen came.
    """

    return max((float(rule.late_hours) for rule in rules), default=0.0) * 60.0


@dataclass(frozen=True)
class PostingRow:
    """How one source's cycle posts, and how long a run waits for a lead.

    ``late_after_minutes`` is counted from a lead's scheduled time (the
    source's ``publication_lag`` rule for that lead); ``poll_seconds`` is
    the longest a waiting reader leaves between asks once that time has
    passed.  ``why`` is the evidence for the shape, or the named reason a
    cycle cannot be waited on lead by lead.
    """

    shape: str
    why: str
    ready_check: str | None = None
    late_after_minutes: float | None = None
    poll_seconds: float | None = None

    @property
    def streams(self) -> bool:
        """Whether a cycle can be waited on lead by lead."""

        return self.shape == "rolling"

    @property
    def waited(self) -> bool:
        """Whether a run waits on this source at all."""

        return self.shape in WAITED_POSTING_SHAPES

    def declaration(self) -> dict[str, object]:
        return {"shape": self.shape, "streams": self.streams,
                "ready_check": self.ready_check,
                "late_after_minutes": self.late_after_minutes,
                "poll_seconds": self.poll_seconds, "why": self.why}


@dataclass(frozen=True)
class LegacyPosting:
    """The posting facts of a source whose transport predates the table.

    ``publication_lag`` is the same rule shape a route row carries, read
    into the source's cycle grid by :func:`woof.source_cycles.cycle_grid_for`;
    ``idx_sidecar`` is the index suffix its transport's objects carry,
    which an ``objects_and_index`` check asks for; ``providers`` holds
    the posting of an alternate provider transport (a keyless archive
    copy of a keyed job API), which a caller naming that provider reads.
    """

    source_id: str
    publication_lag: tuple[LagRule, ...]
    posting: PostingRow
    idx_sidecar: str | None = None
    # A factory, not a shared MappingProxyType({}): Python 3.11's dataclasses
    # refuse an unhashable default, and a mappingproxy is hashable only from 3.12.
    providers: Mapping[str, PostingRow] = field(
        default_factory=lambda: MappingProxyType({}))


def _posting_row(owner: str, raw: object, *, rules: Sequence[LagRule],
                 index_files: Sequence[tuple[str, str | None]],
                 donors: Sequence[DonorRow], probeable: bool) -> PostingRow:
    """One ``posting`` block, checked at load.

    Every refusal names the breakage it prevents (DESIGN A136, 2.1).
    """

    if not isinstance(raw, Mapping):
        # Load check 1.
        raise ValueError(
            f"{ROUTE_TABLE_NAME}: {owner} declares no posting block, so a "
            "run of it could never wait for its leads as they post and "
            "nothing would say why")
    shape = str(raw.get("shape", ""))
    why = str(raw.get("why", "")).strip()
    if shape not in POSTING_SHAPES:
        raise ValueError(
            f"{ROUTE_TABLE_NAME}: {owner} declares posting shape {shape!r}; "
            f"the schedule reads {sorted(POSTING_SHAPES)}, and an unknown "
            "shape would be waited on as if it were one of them")
    if not why:
        raise ValueError(
            f"{ROUTE_TABLE_NAME}: {owner}'s posting block does not say why "
            "its cycles post as they do; the schedule, the readiness answer "
            "and progress print that reason, and without it a source that "
            "cannot stream would say nothing about why")

    def number(key: str) -> float | None:
        value = raw.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{ROUTE_TABLE_NAME}: {owner}'s posting {key} is {value!r}, "
                "not a number")
        return float(value)

    ready_check = raw.get("ready_check")
    late = number("late_after_minutes")
    poll = number("poll_seconds")
    if shape in WAITED_POSTING_SHAPES:
        if ready_check in LATER_READY_CHECKS:
            raise ValueError(
                f"{ROUTE_TABLE_NAME}: {owner} asks for its leads by "
                f"{ready_check!r}, which is named for a later round and not "
                "built; the probe would have nothing to ask")
        if ready_check not in READY_CHECKS:
            raise ValueError(
                f"{ROUTE_TABLE_NAME}: {owner} is waited on ({shape}) but its "
                f"ready_check is {ready_check!r}; the probe reads "
                f"{sorted(READY_CHECKS)}, and without one a waiting run "
                "could not tell a posted lead from a missing one")
        if late is None or poll is None:
            raise ValueError(
                f"{ROUTE_TABLE_NAME}: {owner} is waited on ({shape}) but "
                "declares no late_after_minutes or poll_seconds, so a lead "
                "that never posts would hold its run forever")
        if not probeable:
            # Load check 4.
            raise ValueError(
                f"{ROUTE_TABLE_NAME}: {owner} is waited on ({shape}) but has "
                "no object a probe can ask for, so every run of it would "
                "wait out its budget and fail")
    if shape == "whole_cycle":
        # Load check 2.
        sloped = [rule for rule in rules if rule.per_lead_hours]
        if sloped:
            raise ValueError(
                f"{ROUTE_TABLE_NAME}: {owner} posts whole cycles but a "
                "publication_lag rule gives its leads a per-lead slope, so the "
                "schedule would promise leads before the cycle posts and "
                "progress would show a wait no host can end early")
    if shape == "donor_gated" and not any(donor.cycle == "same"
                                          for donor in donors):
        # Load check 3.
        raise ValueError(
            f"{ROUTE_TABLE_NAME}: {owner} is gated by a donor but declares no "
            "same-cycle donor, so the shape would claim a gate the start "
            "probe never asks about")
    if ready_check == "objects_and_index":
        # Load check 5.
        bare = [role for role, sidecar in index_files if not sidecar]
        if bare or not index_files:
            raise ValueError(
                f"{ROUTE_TABLE_NAME}: {owner} asks for each object's index but "
                f"{', '.join(bare) or 'its objects'} declare no idx_sidecar, "
                "so the probe would ask for an index URL the table never built")
    # Load check 6.
    if late is not None and late <= 0:
        raise ValueError(
            f"{ROUTE_TABLE_NAME}: {owner}'s late_after_minutes is {late:g}; a "
            "budget of zero or less fails every run whose lead is a second "
            "past its scheduled time")
    # Load check 7 (A136 L1 follow-ups): the budget is no shorter than the
    # row's own measured late spread.
    spread = late_spread_minutes(rules)
    if late is not None and shape in WAITED_POSTING_SHAPES and late < spread:
        raise ValueError(
            f"{ROUTE_TABLE_NAME}: {owner}'s late_after_minutes is {late:g}, "
            f"shorter than its own publication_lag rows' late spread: a lead "
            f"was seen posting {spread:g} min after its scheduled time.  An "
            "as-posted run of an ordinary late cycle, no later than one "
            "already seen, would stop with exit 75 (source behind).  The "
            f"budget must be at least {spread:g} min; the shipped rows carry "
            f"the spread plus {LATE_BUDGET_MARGIN_MINUTES:g} min")
    if poll is not None and poll < MIN_POLL_SECONDS:
        raise ValueError(
            f"{ROUTE_TABLE_NAME}: {owner}'s poll_seconds is {poll:g}; asking "
            f"more often than every {MIN_POLL_SECONDS:g} s trips the NOMADS "
            "rate limiter host_policy names")
    return PostingRow(shape=shape, why=why,
                      ready_check=None if ready_check is None else str(ready_check),
                      late_after_minutes=late, poll_seconds=poll)


def _route_posting(route: Route, raw: Mapping[str, object]) -> PostingRow:
    files = route.files + tuple(
        row for era in route.publication_eras for row in era.files)
    return _posting_row(
        f"route {route.source_id}", raw.get("posting"),
        rules=route.publication_lag,
        index_files=tuple((row.role, row.idx_sidecar) for row in files),
        donors=route.donors,
        # Every route row plans an object per lead on a declared host,
        # which is what woof.fetch.cycle_is_probeable answers yes for.
        probeable=bool(route.hosts) and any(row.leads == "all" for row in files))


def _build_legacy_posting() -> Mapping[str, LegacyPosting]:
    document = _load_table()
    rows = dict(document.get("legacy_posting", {}))
    rows.pop("note", None)
    stray = sorted(set(rows) - set(LEGACY_ROUTE_SOURCES))
    if stray:
        raise ValueError(
            f"{ROUTE_TABLE_NAME}: legacy_posting names {stray}, which have no "
            "legacy transport; a route row carries its own posting block")
    ladders = dict(document.get("legacy_ladders", {}))
    found: dict[str, LegacyPosting] = {}
    for source_id in LEGACY_ROUTE_SOURCES:
        raw = rows.get(source_id)
        owner = f"legacy source {source_id}"
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"{ROUTE_TABLE_NAME}: {owner} declares no legacy_posting row, "
                "so a run of it could never wait for its leads as they post "
                "and nothing would say why")
        rules: tuple[LagRule, ...] = ()
        if raw.get("publication_lag") is not None:
            grid = source_adapters.get_source_adapter(source_id).cycle_grid
            hours = [] if grid is None else list(grid.hours)
            rules = _lag_rules(source_id, {
                "cycle_hours": hours, "publication_lag": raw["publication_lag"]})
        sidecar = (str(raw["idx_sidecar"]) if raw.get("idx_sidecar") else None)
        # The legacy transports probe exactly the sources that declare an
        # endpoint ladder here (woof.fetch.cycle_is_probeable); a keyed
        # job API declares none.
        probeable = isinstance(ladders.get(source_id), list)
        posting = _posting_row(
            owner, raw.get("posting"), rules=rules,
            index_files=(("objects", sidecar),), donors=(),
            probeable=probeable)
        if posting.waited and not rules:
            raise ValueError(
                f"{ROUTE_TABLE_NAME}: {owner} is waited on ({posting.shape}) "
                "but declares no publication_lag rows, so its leads would "
                "have no scheduled time to wait from")
        providers = {
            str(name): _posting_row(
                f"{owner} provider {name}", dict(entry).get("posting"),
                rules=(), index_files=(), donors=(), probeable=False)
            for name, entry in dict(raw.get("providers", {})).items()}
        found[source_id] = LegacyPosting(
            source_id=source_id, publication_lag=rules, posting=posting,
            idx_sidecar=sidecar, providers=MappingProxyType(providers))
    return MappingProxyType(found)


def _build_routes() -> Mapping[str, Route]:
    document = _load_table()
    routes: dict[str, Route] = {}
    for source_id, raw in dict(document["routes"]).items():
        ladders = tuple(
            (tuple(entry["cycle_hours"]) if entry.get("cycle_hours") else None,
             tuple((int(step[0]), int(step[1])) for step in entry["steps"]))
            for entry in raw["ladders"])
        routes[source_id] = Route(
            first_lead=int(raw.get("first_lead", 0)),
            source_id=source_id,
            label=str(raw["label"]),
            measured=str(raw.get("measured", "")),
            hosts=fetch_endpoints.ladder(source_id),
            host_note=str(raw.get("host_note", "")),
            coverage_note=str(raw.get("coverage_note", "")),
            cycle_hours=tuple(int(hour) for hour in raw["cycle_hours"]),
            publication_lag=_lag_rules(source_id, raw),
            ladders=ladders,
            default_cadence=int(raw["default_cadence"]),
            layout=str(raw.get("layout", "flat")),
            members=(MappingProxyType(dict(raw["members"]))
                     if raw.get("members") else None),
            axes=MappingProxyType({
                name: tuple(MappingProxyType(dict(group)) for group in groups)
                for name, groups in dict(raw.get("axes", {})).items()}),
            files=tuple(_file_row(row) for row in raw["files"]),
            compose=tuple(
                ComposeRow(kind=str(row["kind"]),
                           roles=tuple(str(role) for role in row["roles"]),
                           name=str(row["name"]),
                           primary=bool(row.get("primary", False)),
                           why=str(row.get("why", "")))
                for row in raw.get("compose", [])),
            donors=tuple(
                DonorRow(source=str(row["source"]), role=str(row["role"]),
                         leads=tuple(int(lead) for lead in row["leads"]),
                         cycle=str(row.get("cycle", "same")),
                         why=str(row.get("why", "")),
                         lead_origin=_donor_lead_origin(row),
                         records=_donor_records(source_id, row))
                for row in raw.get("donors", [])),
            record_subset_supported=bool(
                raw["record_subset"].get("supported", False)),
            record_subset_why=str(raw["record_subset"].get("why", "")),
            prep=MappingProxyType(dict(raw.get("prep", {}))),
            publication_eras=tuple(
                PublicationEra(
                    label=str(era["label"]),
                    valid_from=datetime.fromisoformat(era["valid_from"]),
                    valid_until=(datetime.fromisoformat(era["valid_until"])
                                 if era.get("valid_until") else None),
                    cycle_hours=tuple(era.get("cycle_hours", raw["cycle_hours"])),
                    resolution_degrees=float(era["resolution_degrees"]),
                    files=tuple(_file_row(row) for row in era.get("files", raw["files"])),
                    prep_refusal=str(era.get("prep_refusal", "")),
                    steps=tuple((int(last), int(step)) for last, step in era.get("steps", [])),
                ) for era in raw.get("publication_eras", [])),
        )
    for route in routes.values():
        if not route.hosts:
            raise ValueError(
                f"{ROUTE_TABLE_NAME}: route {route.source_id} names no "
                "endpoint, so there is nowhere to fetch it from")
        for host in route.hosts:
            if not host.why.strip():
                raise ValueError(
                    f"{ROUTE_TABLE_NAME}: route {route.source_id} endpoint "
                    f"{host.name} does not say what it is for; a ladder "
                    "whose refusal cannot name each rung is not a ladder")
        for era in route.publication_eras:
            if ((era.valid_until is not None and era.valid_until <= era.valid_from)
                    or era.resolution_degrees <= 0 or not era.files):
                raise ValueError(
                    f"{route.source_id}: publication era {era.label} has an invalid "
                    "date interval, nonpositive resolution or empty file set; "
                    "cannot resolve its dated grid and file layout")
        files = route.files + tuple(
            row for era in route.publication_eras for row in era.files)
        for row in files:
            bad = unknown_tokens(row.path)
            if bad:
                raise ValueError(
                    f"{ROUTE_TABLE_NAME}: route {route.source_id} file "
                    f"{row.role} spells unknown token(s) {list(bad)}")
        supplement_specs = tuple(route.prep.get("extra_supplements") or ())
        if route.prep.get("supplement"):
            supplement_specs = (route.prep["supplement"], *supplement_specs)
        for supplement in supplement_specs:
            origin = str(supplement.get("from", ""))
            roles = {row.role for row in files}
            if not (origin in SUPPLEMENT_ORIGINS
                    or (origin.startswith("role:")
                        and origin.split(":", 1)[1] in roles)):
                # Named breakage: an origin the planner cannot resolve
                # used to surface only when a request was planned, so a
                # table typo shipped and refused the first user's fetch.
                raise ValueError(
                    f"{ROUTE_TABLE_NAME}: route {route.source_id} takes its "
                    f"supplement from {origin!r}; the planner reads "
                    f"{sorted(SUPPLEMENT_ORIGINS)} or role:NAME of a "
                    f"declared file role ({sorted(roles)})")
    for source_id, raw in dict(document["routes"]).items():
        routes[source_id] = replace(
            routes[source_id], posting=_route_posting(routes[source_id], raw))
    return MappingProxyType(routes)


#: The file formats a ``source_root`` row may declare, as the leading bytes
#: of a file say them (:func:`sniff_format`).
SOURCE_ROOT_FORMATS = frozenset({"grib1", "grib2", "netcdf"})


def _source_root_row(source_id: str, raw: Mapping[str, object]
                     ) -> Mapping[str, object]:
    """One refusal row's ``source_root`` layout, checked at load.

    The layout is how a folder of bytes this WOOF cannot download binds
    to the source's preparation: which files are the ordered inputs and
    which file each supplement role is.  A malformed row fails here, at
    import, rather than as a wrong binding on a user's folder.
    """

    label = f"{ROUTE_TABLE_NAME}: refusal {source_id} source_root"

    def patterns(value, key):
        if (not isinstance(value, list) or not value
                or any(not isinstance(item, str) or not item for item in value)):
            raise ValueError(f"{label}.{key} must be a non-empty list of names")
        return tuple(value)

    def file_format(value, key):
        if value not in SOURCE_ROOT_FORMATS:
            raise ValueError(f"{label}.{key} must be one of "
                             f"{sorted(SOURCE_ROOT_FORMATS)}")
        return str(value)

    inputs = raw.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError(f"{label}.inputs must be an object")
    supplements = []
    for index, row in enumerate(raw.get("supplements") or ()):
        key = f"supplements[{index}]"
        if not isinstance(row, Mapping) or not str(row.get("role", "")):
            raise ValueError(f"{label}.{key} must name a role")
        fetch = row.get("fetch")
        if fetch is not None and not (isinstance(fetch, Mapping)
                                      and str(fetch.get("source", ""))):
            raise ValueError(f"{label}.{key}.fetch must name a source")
        if fetch is not None:
            # Named breakage: the printed fetch line for this file would
            # name a source the fetch door refuses as unknown.
            try:
                source_adapters.get_source_adapter(str(fetch["source"]))
            except ValueError as error:
                raise ValueError(f"{label}.{key}.fetch: {error}") from error
            # Named breakage: a key the printed fetch line does not read
            # is dropped from it without a word.  Whether that line needs
            # --retrieve is the donor adapter's fetch_requires_retrieve,
            # the fact the fetch door itself gates the flag on, so a row
            # restating it could only disagree with the door.
            unread = sorted(set(fetch) - {"source"})
            if unread:
                raise ValueError(
                    f"{label}.{key}.fetch carries {unread}, which no printed "
                    "fetch line reads; it names only the donor source, whose "
                    "adapter states whether its fetch needs --retrieve")
        supplements.append(MappingProxyType({
            "role": str(row["role"]),
            "match": patterns(row.get("match"), f"{key}.match"),
            "format": file_format(row.get("format"), f"{key}.format"),
            "input": bool(row.get("input", False)),
            "fetch": (MappingProxyType(dict(fetch)) if fetch else None),
            "why": str(row.get("why", "")),
        }))
    request = raw.get("request")
    if request is not None and not (
            isinstance(request, Mapping) and str(request.get("dataset", ""))
            and str(request.get("keywords", ""))):
        raise ValueError(f"{label}.request must name a dataset and keywords")
    if request is not None and request.get("lattice_deg") is not None and not (
            isinstance(request["lattice_deg"], (int, float))
            and request["lattice_deg"] > 0):
        raise ValueError(f"{label}.request.lattice_deg must be a positive "
                         "grid spacing in degrees")
    return MappingProxyType({
        "why": str(raw.get("why", "")),
        "inputs": MappingProxyType({
            "match": patterns(inputs.get("match"), "inputs.match"),
            "exclude": (patterns(inputs["exclude"], "inputs.exclude")
                        if inputs.get("exclude") else ()),
            "format": file_format(inputs.get("format"), "inputs.format"),
        }),
        "supplements": tuple(supplements),
        "request": (MappingProxyType(dict(request)) if request else None),
    })


def _build_refusals() -> Mapping[str, Mapping[str, object]]:
    document = _load_table()
    refusals = {}
    for source_id, raw in dict(document.get("refusals", {})).items():
        row = dict(raw)
        if row.get("source_root") is not None:
            row["source_root"] = _source_root_row(source_id, row["source_root"])
        refusals[source_id] = MappingProxyType(row)
    return MappingProxyType(refusals)


_ROUTES = _build_routes()
_REFUSALS = _build_refusals()
_LEGACY_POSTING = _build_legacy_posting()


def legacy_posting(source: str) -> LegacyPosting | None:
    """The ``legacy_posting`` row of a source outside the route table, or None."""

    return _LEGACY_POSTING.get(_canonical(source))


def legacy_publication_lag(source: str) -> tuple[LagRule, ...]:
    """A legacy source's ``publication_lag`` rules; empty where it declares none.

    :func:`woof.source_cycles.cycle_grid_for` reads them into the
    source's cycle grid, so a legacy source answers when a lead is due
    through the same :meth:`~woof.source_cycles.CycleGrid.delay` a
    route does.
    """

    row = legacy_posting(source)
    return () if row is None else row.publication_lag


def posting_for(source: str, provider: str | None = None) -> PostingRow:
    """How ``source``'s cycles post: its route's block or its legacy row.

    ``provider`` names an alternate provider transport a legacy row
    declares (ERA5's keyless archive copy); an unnamed or undeclared one
    reads the source's own block.  A source with neither a route nor a
    legacy row is refused with the reason the table gives for it.
    """

    source_id = _canonical(source)
    route = _ROUTES.get(source_id)
    if route is not None and route.posting is not None:
        return route.posting
    row = _LEGACY_POSTING.get(source_id)
    if row is not None:
        return row.providers.get(provider or "", row.posting)
    route_for(source_id)              # refuses in the table's own words
    raise ValueError(                 # pragma: no cover - route_for refuses
        f"--source {source_id}: no posting row")


def posting_sources() -> tuple[str, ...]:
    """Every source with a posting row: the routes, then the legacy sources."""

    return tuple(_ROUTES) + tuple(_LEGACY_POSTING)


def route_ids() -> tuple[str, ...]:
    """Registry ids with a table-driven acquisition route, in table order."""

    return tuple(_ROUTES)


def refusal_ids() -> tuple[str, ...]:
    """Registry ids the table refuses by name, in table order."""

    return tuple(_REFUSALS)


def acquisition_refusal(source: str) -> Mapping[str, object] | None:
    """The table's declared refusal row for ``source``, or ``None``.

    Public because a caller outside this module has to be able to read
    WHY the table registers no acquisition route -- the run plan says
    so in the sentence it prints about a local-input source -- and a
    second copy of that sentence written at the reading end would
    drift from the table the moment a row's ``why`` changed.
    """

    return _REFUSALS.get(_canonical(source))


def acquisition_refusal_reason(source: str) -> str:
    """One sentence: why no acquisition route is registered for ``source``.

    Empty for a source this fetch door DOES serve, so a caller cannot
    print a reason for an absence that is not there.
    """

    source_id = _canonical(source)
    if source_id in all_fetchable_sources():
        return ""
    why = str((acquisition_refusal(source_id) or {}).get("why", "")).strip()
    return why or "No automatic acquisition route is registered for it."


def source_root_layout(source: str) -> Mapping[str, object] | None:
    """How a hand-staged folder binds to ``source``'s preparation, or None.

    Read from the source's refusal row: a source this fetch door cannot
    download still has a folder its bytes arrive in, and which of those
    files are the ordered inputs and which is each supplement role is a
    table fact about the source, the same kind of fact a route's file
    rows state for the folder a download writes.
    """

    from woof.cf_archive_fetch import sources, row
    canonical = _canonical(source)
    if canonical in sources() and row(canonical).get("source_root") is not None:
        return _source_root_row(canonical,row(canonical)["source_root"])
    return (acquisition_refusal(source) or {}).get("source_root")


def supplement_fetch_retrieves(fetch: Mapping[str, object]) -> bool:
    """Does a supplement's printed ``woof fetch`` line need ``--retrieve``?

    The donor source's own adapter says so (``fetch_requires_retrieve``),
    the one statement of the fact that the fetch door, the wizard and the
    fetch hints read too: that source's default fetch writes a request and
    a retrieval script rather than the file, so a line without the flag
    leaves the supplement's file absent and the folder's preparation
    refuses it, and a line carrying it for any other source is refused.
    """

    return source_adapters.get_source_adapter(
        str(fetch["source"])).fetch_requires_retrieve


def local_prep_line(source: str) -> str:
    """The ``woof prep`` line a folder laid out as the row says runs as."""

    source_id = _canonical(source)
    return (f"woof prep --source {source_id} --source-root DIR "
            "--experiment-config CONFIG.toml "
            "--wps-namelist CONFIG.namelist.wps")


def local_input_remedy(source: str) -> str:
    """What to do with bytes this door cannot download, as a runnable line.

    Every line printed here runs as written once the capitalized
    placeholders are filled in.  A row that declares its folder layout
    gets the short ``--source-root`` line, because the preparation binds
    that folder itself; a row that does not carries its own remedy text.
    """

    source_id = _canonical(source)
    row = acquisition_refusal(source_id) or {}
    if row.get("remedy"):
        return str(row["remedy"])
    layout = row.get("source_root")
    if layout is None:
        return (f"bring the files yourself and name each one to `woof prep "
                f"--source {source_id}`; `woof sources {source_id}` lists "
                "the products its preparation reads.")
    donors = [
        f"{supplement['match'][0]} is what `woof fetch --source "
        f"{supplement['fetch']['source']} --cycle CYCLE --hours HOURS "
        "--area AREA"
        + (" --retrieve" if supplement_fetch_retrieves(supplement["fetch"])
           else "")
        + " --out DIR` writes there"
        for supplement in layout["supplements"] if supplement["fetch"]]
    return (
        "bring the bytes yourself into one folder DIR"
        + (f" ({'; '.join(donors)})" if donors else "")
        + f"; `{local_prep_line(source_id)}` binds that folder's files "
        f"itself, and `woof go CONFIG.toml --data-dir DIR` runs the whole "
        f"chain from a `woof domain --source {source_id}` config.")


def sniff_format(path: Path) -> str | None:
    """``grib1``, ``grib2`` or ``netcdf`` from a file's leading bytes.

    The edition octet sits at the same offset in both GRIB editions, and
    a NetCDF file is either the classic ``CDF`` header or the HDF5
    signature NetCDF-4 writes.  Anything else, or a file that cannot be
    read, answers None.
    """

    try:
        with Path(path).open("rb") as stream:
            head = stream.read(8)
    except OSError:
        return None
    if head[:4] == b"GRIB" and len(head) == 8:
        return {1: "grib1", 2: "grib2"}.get(head[7])
    if head[:3] == b"CDF" or head == b"\x89HDF\r\n\x1a\n":
        return "netcdf"
    return None


def all_fetchable_sources() -> tuple[str, ...]:
    """Every ``--source`` the fetch front door accepts, sorted."""
    from woof.cf_archive_fetch import sources
    return tuple(sorted(set(LEGACY_ROUTE_SOURCES) | set(_ROUTES) | set(sources())
                        | {row.source_id for row in source_adapters.source_adapters()
                           if row.acquisition_owner is not None}))


def table_route_sources() -> frozenset[str]:
    """The source ids that are rows of the route table (no aliases)."""
    return frozenset(_ROUTES)


def _canonical(source: str) -> str:
    """The registry id for a name or alias, or the name unchanged."""

    try:
        return source_adapters.get_source_adapter(source).source_id
    except ValueError:
        return source.strip().lower().replace("_", "-")


def canonical_source(source: str) -> str:
    """The registry id for a name or alias, for callers outside this module.

    Public because the front doors that ask "is this fetchable" must ask
    it about the SAME name this module dispatches on: an alias answered
    yes here and no there is the drift that makes a printed next-step
    refuse.
    """

    return _canonical(source)


def table_route(source: str) -> Route | None:
    """The route-table row for ``source``, or None for a source outside the table (never a refusal).

    The date guidance (:mod:`woof.source_availability`) reads the row's
    lead spacing and ladder through :func:`resolve_leads`, which asks no
    server, so the lengths and start hours it offers are ones the fetch
    takes: a 3-hour window on a source whose files come every 6 hours once
    reached this resolver inside a page request and failed the whole
    source list.  A source outside the table has its own transport, which
    sets its own spacing.
    """

    return _ROUTES.get(_canonical(source))


def route_for(source: str) -> Route:
    """The route for ``source``, or the refusal that names why there is none."""

    source_id = _canonical(source)
    route = _ROUTES.get(source_id)
    if route is not None:
        return route
    refusal = _REFUSALS.get(source_id)
    if refusal is not None:
        # The remedy is a line that runs.  It used to name a
        # SHA256SUMS manifest pair beside --source-root, which the mapped
        # preparation refuses outright ("--source-root is not used") and
        # the member route accepts only while authoring.
        raise ValueError(
            f"--source {source_id}: no fetch route.\n"
            f"  why: {refusal['why']}\n"
            f"  remedy: {local_input_remedy(source_id)}")
    if source_id in LEGACY_ROUTE_SOURCES:
        raise ValueError(
            f"--source {source_id} has its own transport in woof.fetch and "
            "does not run through the route table")
    try:
        adapter = source_adapters.get_source_adapter(source_id)
    except ValueError as error:
        raise ValueError(str(error)) from error
    # TWO different absences, and they had one sentence between them.
    # "No route" for an unrunnable row means the decode is missing, so a
    # download would buy nothing.  "No route" for a RUNNABLE row means
    # only the transport is missing -- this ArWen reads those bytes and
    # prepares them, it just cannot go and get them -- and telling such
    # a reader "the registry row is not runnable" is a false statement
    # of cause that stops them at a door that is actually open.
    if adapter.runnable:
        why = (
            "no acquisition route is registered for this source, so this "
            f"WOOF has no way to download its bytes.  {adapter.source_id} "
            "IS runnable: bring the files yourself and the prepared route "
            "reads them")
        remedy = f"  remedy: {local_input_remedy(adapter.source_id)}\n"
    else:
        why = (
            f"the registry row is not runnable ({adapter.status.value}); "
            "nothing in this WOOF could read the bytes a download "
            "produced")
        remedy = ""
    raise ValueError(
        f"--source {adapter.source_id}: no fetch route.\n"
        f"  why: {why}.\n"
        f"{remedy}"
        "  see: `woof sources` for what each registered source can do "
        "today, or `woof sources " + adapter.source_id + "` for this "
        "row in full.")


# --------------------------------------------------------------------------
# Cycle and lead grammar
# --------------------------------------------------------------------------


def publication_era(route: Route, cycle: datetime) -> PublicationEra | None:
    """Select the UTC half-open date interval without probing a server."""

    if not route.publication_eras:
        return None
    if cycle.tzinfo is not None:
        cycle = cycle.astimezone(timezone.utc).replace(tzinfo=None)
    matches = [era for era in route.publication_eras
               if era.valid_from <= cycle
               and (era.valid_until is None or cycle < era.valid_until)
               and cycle.hour in era.cycle_hours]
    if len(matches) != 1:
        first = min(era.valid_from for era in route.publication_eras)
        if cycle < first:
            raise ValueError(
                f"{route.label} begins {first:%Y-%m-%dT%H} UTC; this cycle predates it")
        reason = "no publication era" if not matches else "overlapping publication eras"
        raise ValueError(
            f"--source {route.source_id} --cycle {cycle:%Y-%m-%dT%H}: {reason}; "
            "the route table cannot select a file layout for this cycle")
    return matches[0]


def _era_cycles(era: PublicationEra) -> tuple[datetime, ...]:
    """The first instance of each cycle hour inside an era's interval."""
    cycles = []
    for hour in era.cycle_hours:
        cycle = era.valid_from.replace(hour=hour, minute=0, second=0, microsecond=0)
        if cycle < era.valid_from:
            cycle += timedelta(days=1)
        if era.valid_until is None or cycle < era.valid_until:
            cycles.append(cycle)
    return tuple(cycles)


def first_preparable_cycle(route: Route) -> datetime | None:
    """One authority for the calendar's bound and the fetch's remedy."""
    return min((cycle for era in route.publication_eras if not era.prep_refusal
                for cycle in _era_cycles(era)), default=None)


def planning_cycles(route: Route) -> tuple[datetime, ...]:
    """Stand-in cycles for pricing the newest preparable layout at each hour."""
    if not route.publication_eras:
        return tuple(datetime(2001, 1, 1, hour) for hour in route.cycle_hours)
    cycles = [cycle for era in route.publication_eras if not era.prep_refusal
              for cycle in _era_cycles(era)]
    return tuple(max(cycle for cycle in cycles if cycle.hour == hour)
                 for hour in route.cycle_hours if any(cycle.hour == hour for cycle in cycles))


def _lead_steps(route: Route, cycle: datetime) -> tuple[tuple[int, int], ...]:
    era = publication_era(route, cycle)
    if era is not None and era.steps:
        return era.steps
    for cycle_hours, steps in route.ladders:
        if cycle_hours is None or cycle.hour in cycle_hours:
            return steps
    raise ValueError(f"--source {route.source_id}: no lead ladder for the {cycle:%H}Z cycle")


def ladder_for(route: Route, cycle: datetime) -> tuple[int, ...]:
    """Every forecast lead ``cycle`` publishes, in order."""

    steps = _lead_steps(route, cycle)
    leads: list[int] = []
    previous = 0
    for index, (through, step) in enumerate(steps):
        start = 0 if index == 0 else previous + step
        leads.extend(range(start, through + 1, step))
        previous = leads[-1]
    return tuple(lead for lead in leads if lead >= route.first_lead)


def _extended_cycle_hours(route: Route) -> tuple[int, ...]:
    """Cycle hours whose ladder reaches farthest (for the refusal text)."""

    best_hours: tuple[int, ...] = ()
    best_last = -1
    for cycle_hours, steps in route.ladders:
        last = steps[-1][0]
        if last > best_last and cycle_hours:
            best_last, best_hours = last, tuple(cycle_hours)
    return best_hours


def resolve_cycle(route: Route, cycle: datetime) -> datetime:
    """Refuse a cycle hour the producer does not run."""

    if cycle.hour not in route.cycle_hours:
        offered = ", ".join(f"{hour:02d}" for hour in route.cycle_hours)
        raise ValueError(
            f"--source {route.source_id} --cycle {cycle:%Y-%m-%dT%H}: "
            f"{cycle:%H}Z is not a cycle this producer runs.\n"
            f"  it runs: {offered} (UTC).")
    return cycle


def _is_whole_hours(cadence: object) -> bool:
    """A cadence is a whole number of hours between boundary times, at least 1."""

    return (not isinstance(cadence, bool) and isinstance(cadence, int)
            and cadence >= 1)


def _window_ladders(route: Route, cycle: datetime | None
                    ) -> tuple[frozenset[int], ...]:
    """CYCLE's published leads, or every planning cycle's when it is None."""

    cycles = (cycle,) if cycle is not None else planning_cycles(route)
    ladders = []
    for each in cycles:
        try:
            ladders.append(frozenset(ladder_for(route, each)))
        except ValueError:
            continue
    return tuple(ladders)


def _preparation_target(route: Route) -> tuple[str, Mapping[str, object]] | None:
    """The route's packaged mapping profile and target, or None.

    None for a route whose preparation is not a packaged mapped profile:
    nothing here can say what it takes, so it is never named as a reason.
    """

    row = source_adapters.get_source_adapter(
        canonical_source(str(route.prep.get("source", route.source_id))))
    if row.runner != "mapped_composition_v1" or not row.packaged_profile:
        return None
    from woof.source_authorities import packaged_mapping_target

    return row.packaged_profile, packaged_mapping_target(row.packaged_profile)


def window_spacings(route: Route, start_hour: int, hours: int, *,
                    cycle: datetime | None = None, floor: int = 1,
                    round_up: bool = False, first: bool = False
                    ) -> tuple[int, ...]:
    """Every spacing ROUTE offers over a whole window, finest first.

    A173: a table route's cadence is any whole multiple of the spacing its
    publisher posts.  A spacing is offered when it is a whole multiple of
    ``floor``, the cycle's ladder (``cycle``'s, or any planning cycle's
    when it is None) publishes every lead of the window at it, and the
    route's packaged preparation takes it.  Those are the only two things
    that can break a spacing: a lead the publisher never posts, and a
    decode that refuses the series after the download.  The per-route
    ``cadences`` list this replaces left out spacings that broke neither
    (2 h on an hourly publisher, 12 h on every global) and said so.

    ``round_up`` extends ``hours`` to a whole number of steps the way the
    domain door writes ``[fetch]``; without it only spacings that divide
    ``hours`` are offered.  ``first`` stops at the finest one.
    """

    floor = int(floor)
    if floor < 1:
        raise ValueError(f"a spacing floor is a whole number of hours, not {floor}")
    ladders = _window_ladders(route, cycle)
    if not ladders:
        return ()
    from woof.source_authorities import boundary_interval_refusal

    start, hours = int(start_hour), int(hours)
    prepared = _preparation_target(route)
    # A spacing past the farthest lead serves only a window of one lead,
    # which the floor itself already serves.
    limit = max(floor, max(max(published) for published in ladders) - start)
    offered: list[int] = []
    for cadence in range(floor, limit + 1, floor):
        if round_up:
            span = max(cadence, -(-hours // cadence) * cadence)
        elif hours % cadence:
            continue
        else:
            span = hours
        wanted = range(start, start + span + 1, cadence)
        if not any(all(lead in published for lead in wanted)
                   for published in ladders):
            continue
        if prepared is not None and boundary_interval_refusal(
                prepared[1], cadence * 3600) is not None:
            continue
        offered.append(cadence)
        if first:
            break
    return tuple(offered)


def window_cadence(route: Route, start_hour: int, hours: int, *,
                   cycle: datetime | None = None, floor: int | None = None,
                   round_up: bool = False) -> int | None:
    """The finest spacing the route offers over a whole window, or None.

    The ladder rows ARE the source's cadence by lead: IFS's 00/12Z row
    says every 3 h to f144 and every 6 h to f360, GEFS's every 3 h to
    f240 and every 6 h past it.  A window that runs past the lead where
    the finer spacing ends cannot be taken at that spacing, and the
    boundary series a forecast reads is one uniform spacing, so the
    window takes the coarsest spacing it crosses into, from its start.
    That used to be a flag every caller had to know to pass (``--cadence
    6`` for a 240 h IFS run), and the default refused the window with
    "does not publish f147".

    Only whole multiples of ``floor`` (the route's default) are tried,
    finest first (:func:`window_spacings`), so a window the default
    serves keeps it and a window that names no cadence never lands on a
    spacing finer than, or off the grid of, the default.  ``cycle`` asks
    one cycle's ladder; without it, any cycle hour whose ladder serves
    the window answers.  ``round_up`` is :func:`window_spacings`'s.  None
    means no spacing serves the window, and the caller's own refusal
    names why.
    """

    floor = route.default_cadence if floor is None else int(floor)
    spacings = window_spacings(route, start_hour, hours, cycle=cycle,
                               floor=floor, round_up=round_up, first=True)
    return spacings[0] if spacings else None


def resolve_leads(route: Route, cycle: datetime, hours: int, *,
                  cadence: int | None = None,
                  start_hour: int = 0) -> tuple[int, ...]:
    """The ordered leads a window asks for, checked against the ladder.

    With no ``cadence`` the window takes :func:`window_cadence`: the
    route's default spacing where the cycle's ladder publishes it over
    the whole window, and the coarser spacing it runs into otherwise.  A
    named cadence is any whole number of hours (A173), refused only for
    what breaks it: a final time the spacing would omit, a lead the
    cycle's ladder does not publish, or a series the route's packaged
    preparation would refuse after the download.
    """

    if isinstance(hours, bool) or not isinstance(hours, int) or hours < 0:
        raise ValueError("--hours must be a nonnegative integer")
    if (isinstance(start_hour, bool) or not isinstance(start_hour, int)
            or start_hour < 0):
        raise ValueError("--forecast-start-hour must be a nonnegative integer")
    named = cadence is not None
    if cadence is None:
        cadence = (window_cadence(route, start_hour, hours, cycle=cycle)
                   or route.default_cadence)
    if not _is_whole_hours(cadence):
        raise ValueError(cadence_refusal(
            route, cadence, cycle=cycle, start_hour=start_hour))
    if hours % cadence:
        raise ValueError(
            f"--hours must be an exact multiple of the {cadence} h cadence; "
            "the requested final time must not be silently omitted")
    ladder = ladder_for(route, cycle)
    last = start_hour + hours
    if last > ladder[-1]:
        era = publication_era(route, cycle)
        extended = () if era is not None and era.steps else _extended_cycle_hours(route)
        reach = ""
        if extended:
            extended_last = max(
                steps[-1][0] for cycle_hours, steps in route.ladders
                if cycle_hours == extended)
            reach = (f"  the {', '.join(f'{h:02d}' for h in extended)}Z "
                     f"cycles reach f{extended_last:03d}.")
        raise ValueError(
            f"--source {route.source_id}: the {cycle:%H}Z cycle forecasts "
            f"through f{ladder[-1]:03d}; this window ends at f{last:03d}.\n"
            f"{reach}".rstrip())
    wanted = [lead for lead in ladder
              if start_hour <= lead <= last
              and (lead - start_hour) % cadence == 0]
    missing = [lead for lead in range(start_hour, last + 1, cadence)
               if lead not in set(ladder)]
    if missing:
        served = (window_cadence(route, start_hour, hours, cycle=cycle,
                                 floor=1)
                  if named else None)
        remedy = ""
        if served is not None and served != cadence:
            # Worded for every door that names a cadence: the flag, and
            # the [fetch] key a configuration or a saved setup carries.
            remedy = (f"\n  remedy: cadence {served} (--cadence {served}, or "
                      f"cadence = {served} in [fetch]) is published over this "
                      "whole window; a window that names no cadence takes the "
                      "spacing its ladder publishes by itself.")
        raise ValueError(
            f"--source {route.source_id}: the {cycle:%H}Z cycle does not "
            f"publish f{missing[0]:03d}"
            + (f" at --cadence {cadence}" if named else "")
            + f"; its ladder runs {_ladder_words(route, cycle)}.{remedy}")
    refusal = cadence_refusal(route, cadence, cycle=cycle,
                              start_hour=start_hour)
    if refusal is not None:
        raise ValueError(refusal)
    if not wanted:
        raise ValueError(
            f"--source {route.source_id}: the requested window resolves to "
            "no forecast lead at all")
    return tuple(wanted)


def cadence_refusal(route: Route, cadence: object, *, cycle: datetime,
                    start_hour: int = 0) -> str | None:
    """Why ROUTE cannot take CADENCE whatever the window, or None.

    The window-free half of :func:`resolve_leads`'s check, by the
    breakage each refusal prevents: a cadence that is not a whole number
    of hours names no boundary times at all, and a spacing the route's
    packaged preparation refuses would be downloaded whole and then
    refused by the decode.  A lead the cycle's ladder does not publish is
    the window's question, answered by :func:`resolve_leads` in the
    ladder's words.  A173 retired the third reason this used to give,
    "the route's own row" leaving out a spacing the publisher posts and
    the decode takes: a refusal that names no breakage does not exist.
    """

    head = (f"--cadence {cadence}: --source {route.source_id} takes any whole "
            "number of hours at which the cycle's ladder publishes every "
            f"lead of the window (default {route.default_cadence} h).")
    remedy = ("\n  remedy: name another (--cadence N, or cadence = N in "
              "[fetch]), or name none: a window that names no cadence takes "
              f"{route.default_cadence} h, or the coarser spacing its ladder "
              "runs into.")
    if not _is_whole_hours(cadence):
        return (head + "\n  why: a cadence is a whole number of hours between "
                "boundary times, at least 1." + remedy)
    prepared = _preparation_target(route)
    if prepared is None:
        return None
    from woof.source_authorities import (
        boundary_interval_refusal, boundary_interval_takes)

    profile, target = prepared
    if boundary_interval_refusal(target, cadence * 3600) is None:
        return None
    takes = boundary_interval_takes(target)
    return (head + f"\n  why: {route.source_id}'s preparation (the packaged "
            f"{profile} mapping) takes {takes}, so a {cadence} h series "
            "would be downloaded whole and then refused by the decode."
            + remedy.replace("name another", "name a spacing it takes", 1))


def _ladder_words(route: Route, cycle: datetime) -> str:
    parts = []
    previous = 0
    for index, (through, step) in enumerate(_lead_steps(route, cycle)):
        # The first segment begins at the route's first published lead
        # (ladder_for drops the leads below it), not at f000.
        start = (-(-route.first_lead // step) * step if index == 0
                 else previous + step)
        parts.append(f"f{start:03d}..f{through:03d} every {step} h")
        previous = through
    return ", then ".join(parts)


# --------------------------------------------------------------------------
# Members
# --------------------------------------------------------------------------

def member_tokens(route: Route) -> Mapping[str, str]:
    """``member name -> filename/path token`` for the declared member set.

    Two vocabularies, because the producers use two: GEFS's control is
    ``c00`` in the member grammar and ``gec00`` in the key, while every
    AIGEFS member spells the same string in both.  The table carries the
    pair so neither is inferred from the other.
    """

    if route.members is None:
        return MappingProxyType({})
    default = str(route.members["default"])
    names: dict[str, str] = {
        default: str(route.members.get("default_token", default))}
    low, high = route.members.get("perturbed_range", (1, 0))
    name_fmt = route.members.get("perturbed_name_format")
    token_fmt = route.members.get("perturbed_token_format", name_fmt)
    if name_fmt:
        for number in range(int(low), int(high) + 1):
            names.setdefault(str(name_fmt) % number,
                             str(token_fmt) % number)
    return MappingProxyType(names)


def resolve_member(route: Route, member: str | None) -> tuple[str, str]:
    """``(member_name, filename_token)`` for the requested member."""

    if route.members is None:
        if member is not None:
            raise ValueError(
                f"--member {member}: --source {route.source_id} is not an "
                "ensemble; it publishes one deterministic state.")
        return "", ""
    name = str(route.members["default"]) if member is None else str(member)
    known = member_tokens(route)
    if name not in known:
        listed = tuple(known)
        raise ValueError(
            f"--member {name}: --source {route.source_id} publishes "
            f"{listed[0]}..{listed[-1]} -- "
            f"{len(listed)} members in all.")
    return name, known[name]


# --------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannedObject:
    """One published object, and every endpoint that publishes it.

    ``key`` is the host-independent part -- the rendered path template
    -- and it is what makes the ladder possible at all: the NCEP
    operational server and the AWS archive answer the SAME relative key
    with the same bytes (HEAD-verified for every source in the family),
    so falling through hosts is appending one key to another base, not
    re-planning the request.  ``url`` is that key on the ladder's head,
    kept as a field because a plan is read for what it WILL do.
    """

    name: str
    url: str
    relpath: str
    role: str
    lead: int | None
    idx_url: str | None
    key: str = ""
    member_index: Mapping[str, object] | None = None
    member_ordinal: int | None = None
    index_key: str | None = None
    record_index: Mapping[str, object] | None = None

    def urls(self, ladder: Sequence[Endpoint]) -> tuple[str, ...]:
        if not self.key:
            return (self.url,)
        return tuple(endpoint.url(self.key) for endpoint in ladder)


@dataclass(frozen=True)
class ComposePart:
    role: str
    relpath: str


@dataclass(frozen=True)
class ComposeStep:
    kind: str
    name: str
    primary: bool
    lead: int
    parts: tuple[ComposePart, ...]


@dataclass(frozen=True)
class DonorRequest:
    source: str
    role: str
    cycle: datetime
    leads: tuple[int, ...]
    why: str
    records: Mapping[str, object] | None = None


@dataclass(frozen=True)
class FetchPlan:
    route: Route
    files: tuple[FileRow, ...]
    host: Host
    cycle: datetime
    leads: tuple[int, ...]
    member: str
    objects: tuple[PlannedObject, ...]
    compose: tuple[ComposeStep, ...]
    donors: tuple[DonorRequest, ...]
    primary_files: tuple[Path, ...]
    supplement_files: tuple[Path, ...]
    supplement_role: str | None
    member_set: str | None
    out: Path | None = None
    #: The endpoints this cycle may be asked for, in RETENTION order:
    #: the operational server while it still holds the cycle, the
    #: archive behind it.  One entry when ``--transport`` pinned a
    #: host.  Which of them actually moves each object is settled at
    #: transfer time, by availability -- planning stays network-free.
    ladder: tuple[Endpoint, ...] = ()
    #: The host ``--transport`` named, if any.  A typed host is a
    #: decision: it disables fall-through and it IS the request's
    #: identity, where an unpinned request's identity is the ladder.
    pinned_host: str | None = None

    @property
    def source_id(self) -> str:
        return self.route.source_id

    @property
    def extra_supplement_files(self):
        bindings = []
        for spec in self.route.prep.get("extra_supplements") or ():
            origin = str(spec["from"])
            if not origin.startswith("role:"):
                raise ValueError("additional supplements must name a declared file role")
            file_role = origin.split(":", 1)[1]
            bindings.extend((str(spec["role"]), Path(obj.relpath))
                            for obj in self.objects if obj.role == file_role)
        return tuple(bindings)


def _cycle_context(cycle: datetime) -> dict[str, str]:
    return {
        "YYYY": f"{cycle:%Y}", "MM": f"{cycle:%m}", "DD": f"{cycle:%d}",
        "HH": f"{cycle:%H}", "YYYYMMDD": f"{cycle:%Y%m%d}",
        "YYYYMMDDHH": f"{cycle:%Y%m%d%H}",
        "YYYYMMDDHHMMSS": f"{cycle:%Y%m%d%H}0000",
    }


def _render(template: str, context: Mapping[str, str]) -> str:
    def replace(match: re.Match) -> str:
        token = match.group(1)
        if token not in context:
            raise ValueError(
                f"path template {template!r} wants {{{token}}}, which this "
                "route's axes do not provide")
        return context[token]

    return _TOKEN_RE.sub(replace, template)


def _axis_entries(route: Route, axis: str) -> tuple[dict[str, str], ...]:
    """Expand one declared axis into its per-object token dictionaries."""

    entries: list[dict[str, str]] = []
    for group in route.axes[axis]:
        leveltype = str(group.get("leveltype", ""))
        levels = group.get("levels")
        literal = group.get("level_literal")
        prefix = str(group.get("level_prefix", ""))
        fmt = str(group.get("level_format", "%d"))
        for name in group["names"]:
            if levels:
                for level in levels:
                    text = prefix + (fmt % int(level))
                    entries.append({
                        "FIELD": str(name),
                        "field_lower": str(name).lower(),
                        "leveltype": leveltype,
                        "LEVEL": text,
                        "LEVEL_SUFFIX": f"_{text}",
                    })
            elif literal:
                entries.append({
                    "FIELD": str(name), "field_lower": str(name).lower(),
                    "leveltype": leveltype, "LEVEL": str(literal),
                    "LEVEL_SUFFIX": f"_{literal}",
                })
            else:
                entries.append({
                    "FIELD": str(name), "field_lower": str(name).lower(),
                    "leveltype": leveltype, "LEVEL": "", "LEVEL_SUFFIX": "",
                })
    return tuple(entries)


def _relpath(route: Route, key: str) -> str:
    if route.layout == "upstream":
        return f"upstream/{key}"
    return key.rsplit("/", 1)[-1]


def resolve_mode(source: str, mode: str | None) -> str:
    """The byte transport a table route runs with.

    Full files are the default and the pipeline; record subsetting is an
    opt-in bandwidth saver, and a route that cannot honour it refuses in
    its own words rather than silently degrading.  ``auto`` asks for the
    default, so it takes the full-file route like an omitted mode does.
    """

    route = route_for(source)
    if mode is None or mode in ("full-file", "auto"):
        return "full-file"
    if mode == "idx-subset":
        if route.record_subset_supported:
            return "idx-subset"
        raise ValueError(
            f"--mode idx-subset: --source {route.source_id} takes whole "
            "objects.\n"
            f"  why: {route.record_subset_why}.\n"
            "  remedy: drop --mode (full-file is the default and the "
            "pipeline).")
    raise ValueError(f"--mode {mode}: unknown transport")


def endpoint_ladder(route: Route, cycle: datetime, *,
                    host: str | None = None,
                    now: datetime | None = None) -> tuple[Endpoint, ...]:
    """The endpoints ``cycle`` will be asked for, in order.

    The cycle is resolved BEFORE this is called -- that is the whole
    point.  A latest initialization is hours old, so the operational
    server (which published it first) heads the ladder; a cycle older
    than that server's measured window is not asked for there at all,
    and the archive is asked directly.
    """

    return fetch_endpoints.serving_ladder(
        route.source_id, cycle=cycle, now=now, pinned=host,
        throttle_fallthrough=True)


def resolve_request(source: str, *, cycle: datetime, hours: int,
                    cadence: int | None = None, start_hour: int = 0,
                    host: str | None = None, member: str | None = None,
                    area: str | None = None,
                    out: Path | None = None,
                    now: datetime | None = None) -> FetchPlan:
    """Everything a fetch will do, decided before a single byte moves."""

    source = source_adapters.source_for_cycle(source, cycle)
    route = route_for(source)
    host = fetch_endpoints.policy_transport(route.source_id, host)
    if area is not None:
        raise ValueError(
            f"--area/--point: --source {route.source_id} publishes whole "
            "objects and there is no subsetting service in front of them.\n"
            "  where the crop happens: `woof prep` maps the source onto "
            "your domain, so the namelist geometry is the crop.")
    resolve_cycle(route, cycle)
    era = publication_era(route, cycle)
    if era is not None and era.prep_refusal:
        first = first_preparable_cycle(route)
        remedy = (f" Use a cycle from {first:%Y-%m-%dT%H} UTC onward, the first "
                  "publication this preparation can read." if first is not None else "")
        raise ValueError(
            f"--source {route.source_id} --cycle {cycle:%Y-%m-%dT%H}: "
            f"{era.label}: {era.prep_refusal}{remedy}")
    files = era.files if era is not None else route.files
    leads = resolve_leads(route, cycle, hours, cadence=cadence,
                          start_hour=start_hour)
    if host is not None:
        route.host(host)          # refuses in the route's own words
    ladder = endpoint_ladder(route, cycle, host=host, now=now)
    chosen = ladder[0]
    member_name, member_token = resolve_member(route, member)
    member_ordinal = None
    if any(row.member_index is not None for row in files):
        from woof.member_grammar import load_member_grammar
        from woof.source_authorities import packaged_member_grammar
        grammar_id = source_adapters.get_source_adapter(route.source_id).member_set
        if grammar_id is None:
            raise ValueError("Indexed member fetch needs a byte-verification member grammar")
        member_ordinal = load_member_grammar(
            packaged_member_grammar(grammar_id)).member(member_name).ordinal

    context = _cycle_context(cycle)
    context["MEMBER"] = member_token

    objects: list[PlannedObject] = []
    by_role: dict[str, list[tuple[int, str]]] = {}
    seen: set[str] = set()
    supplement_spec = route.prep.get("supplement")
    supplement_origin = (str(supplement_spec["from"]) if supplement_spec
                         else None)
    #: Objects planned only because the supplement is the cycle's step-0
    #: object and the window starts later: fetched, bound as the
    #: supplement, never an input of the forcing series.
    step0_only: list[str] = []

    def emit(row: FileRow, lead: int | None) -> None:
        if lead is not None and (lead < row.lead_min or (row.lead_max is not None and lead > row.lead_max)):
            return
        lead_context = dict(context)
        if lead is not None:
            lead_context.update({
                "F": str(lead), "FF": f"{lead:02d}", "FFF": f"{lead:03d}"})
        for entry in (_axis_entries(route, row.axis) if row.axis else ({},)):
            key = _render(row.path, {**lead_context, **entry})
            identity = (key, lead) if row.member_index and row.member_index.get("format") == "wgrib2" else key
            if identity in seen:
                continue
            seen.add(identity)
            relpath = _relpath(route, key)
            if row.member_index is not None:
                relpath = (f"members/{member_name}/leads/f{lead:03d}/{relpath}"
                           if row.member_index.get("format") == "wgrib2"
                           else f"members/{member_name}/{relpath}")
            index_base = key
            if row.index_suffix_replaces:
                if not key.endswith(row.index_suffix_replaces):
                    raise ValueError(f"Index suffix does not match object key {key}")
                index_base = key[:-len(row.index_suffix_replaces)]
            index_key = f"{index_base}{row.idx_sidecar}" if row.idx_sidecar else None
            objects.append(PlannedObject(
                name=relpath, url=chosen.url(key), relpath=relpath,
                role=row.role, lead=lead, key=key,
                idx_url=chosen.url(index_key) if index_key else None,
                member_index=row.member_index, member_ordinal=member_ordinal,
                index_key=index_key, record_index=row.record_index))
            by_role.setdefault(row.role, []).append(
                (lead if lead is not None else -1, relpath))

    # Lead-major, so the pool's in-order admitted prefix is a contiguous
    # run of COMPLETE valid times: an interrupted fetch leaves a series a
    # shorter window can still be prepared from, never half of every hour.
    # A step-0 object belongs to the window's FIRST valid time, whatever
    # lead that is: its statics are what that time (and every later one)
    # is prepared with, so it travels in the first lead's group.
    for row in files:
        if row.leads == "none":
            emit(row, None)
    for lead in leads:
        for row in files:
            if row.leads == "none":
                continue
            if row.leads == "step0":
                if lead == leads[0]:
                    emit(row, 0)
                continue
            emit(row, lead)
        if (lead == leads[0] and lead != 0
                and supplement_origin == "step0"):
            for row in files:
                if row.primary and row.leads == "all":
                    before = len(objects)
                    emit(row, 0)
                    step0_only.extend(
                        obj.relpath for obj in objects[before:])

    step0_roles = {row.role for row in files if row.leads == "step0"}
    compose: list[ComposeStep] = []
    for row in route.compose:
        for lead in leads:
            parts = tuple(
                ComposePart(role=role, relpath=relpath)
                for role in row.roles
                for entry_lead, relpath in by_role.get(role, ())
                if (entry_lead == lead and role not in step0_roles)
                or (role in step0_roles and lead == leads[0]))
            if not parts:
                continue
            lead_context = dict(context)
            lead_context.update({
                "F": str(lead), "FF": f"{lead:02d}", "FFF": f"{lead:03d}"})
            compose.append(ComposeStep(
                kind=row.kind, name=_render(row.name, lead_context),
                primary=row.primary, lead=lead, parts=parts))

    donors = tuple(
        DonorRequest(source=row.source, role=row.role, cycle=cycle,
                     leads=(row.leads if row.lead_origin == "cycle"
                            else tuple(leads[0] + lead for lead in row.leads)),
                     why=row.why, records=row.records)
        for row in route.donors)

    if compose:
        primary = tuple(Path(step.name) for step in compose if step.primary)
    else:
        primary = tuple(
            Path(obj.relpath) for obj in objects
            if obj.relpath not in step0_only
            and any(row.primary and row.role == obj.role for row in files))

    supplements: tuple[Path, ...] = ()
    supplement_role: str | None = None
    if supplement_spec:
        supplement_role = str(supplement_spec["role"])
        origin = str(supplement_origin)
        select = supplement_spec.get("select")
        if origin == "every_input":
            supplements = primary
        elif origin == "step0":
            # The cycle's step-0 object of every primary row: the first
            # inputs when the window starts there, the objects planned
            # beside the window otherwise.
            supplements = tuple(
                Path(obj.relpath) for obj in objects
                if obj.lead == 0
                and any(row.primary and row.leads == "all"
                        and row.role == obj.role for row in files))
        elif origin.startswith("role:"):
            role = origin.split(":", 1)[1]
            supplements = tuple(
                Path(obj.relpath) for obj in objects
                if obj.role == role
                and (select is None or f"_{select}." in obj.relpath
                     or f"_{select}_" in obj.relpath))
        else:
            raise ValueError(
                f"{ROUTE_TABLE_NAME}: route {route.source_id} declares an "
                f"unknown supplement origin {origin!r}")

    return FetchPlan(
        route=route, files=files, host=chosen, cycle=cycle, leads=leads,
        member=member_name, objects=tuple(objects), compose=tuple(compose),
        donors=donors, primary_files=primary, supplement_files=supplements,
        supplement_role=supplement_role,
        member_set=(str(route.prep["member_prep"])
                    if route.prep.get("member_prep") else None),
        out=out, ladder=ladder, pinned_host=host)


def donor_records_plan(plan: FetchPlan, records: Mapping[str, object]) -> FetchPlan:
    """The part of a donor's own plan a record-selected donor moves.

    A donor binds its primary object (one per lead) and nothing else, so
    the route's other rows are not moved, and each primary object takes
    only ``records``, the donor row's selection, by its index.  Measured
    cost this removes: hrrr-native's hrrr-prs soil donor moved the whole
    381 MB wrfprsf00 for 29 MB of soil records, plus a 134 MB wrfsfcf00
    the composition never reads.

    Named breakage: a donor route that composes its primary from several
    objects, or whose primary row declares no index, cannot be cut this
    way; it refuses here rather than handing the preparation a file with
    records missing.
    """

    if plan.compose:
        raise ValueError(f"the {plan.source_id} donor composes its primary from "
                         "several objects; its records cannot be selected by index")
    primary_roles = {row.role for row in plan.files if row.primary}
    objects = tuple(replace(obj, record_index=records) for obj in plan.objects
                    if obj.role in primary_roles)
    if not objects or any(obj.index_key is None for obj in objects):
        raise ValueError(f"the {plan.source_id} donor's primary objects declare no "
                         "index; its records cannot be selected by index")
    kept = {obj.relpath for obj in objects}
    return replace(plan, objects=objects,
                   supplement_files=tuple(path for path in plan.supplement_files
                                          if path.as_posix() in kept))


# --------------------------------------------------------------------------
# Transfer
# --------------------------------------------------------------------------

MANIFEST_NAME = "fetch-manifest.json"
SHA256SUMS_NAME = "SHA256SUMS"
INPUT_LIST_NAME = "inputs.txt"
PREP_COMMAND_NAME = "prep-command.txt"

#: The same bound half, as machine-readable argv tokens.  The text file
#: above is a reader's; this one is a CALLER'S -- ``woof run-plan``'s
#: staged chain composes its preparation from it, so the binding is
#: relayed from the fetch's own artifact instead of being re-derived
#: from a second copy of the route table.  Tokens, not a command
#: string, so no quoting convention has to round-trip Windows paths.
PREP_ARGUMENTS_NAME = "prep-arguments.json"
PREP_ARGUMENTS_SCHEMA = "gpuwm-fetch-prep-arguments-v1"

_USER_AGENT = "gpuwm-fetch/2.5 (+https://github.com/arwenweather)"
_RECOVERY_REQUEST_NAME = "fetch-recovery-request.json"
_RECOVERY_SCHEMA = "gpuwm-fetch-recovery-v1"
_RECOVERY_DIRECTORY = ".fetch-verified"
#: Rounds per object: the one number every source's fetch shares
#: (:data:`woof.fetch_endpoints.TRANSIENT_ATTEMPTS`).
_TRANSFER_ATTEMPTS = fetch_endpoints.TRANSIENT_ATTEMPTS
_RETRY_WAIT_LIMIT = fetch_endpoints.TRANSIENT_WAIT_LIMIT_S


def _magic_for(plan: FetchPlan, role: str) -> str:
    for row in plan.files:
        if row.role == role:
            return row.magic
    return "GRIB"


def _verify_payload(path: Path, *, magic: str, label: str) -> None:
    """The cheapest accurate completeness bar for a downloaded object.

    Leading magic separates a payload from an HTML error page a proxy
    served with HTTP 200; the GRIB2 end marker separates a complete
    object from a truncated transfer, which is the failure a
    length-only check misses when the server closes the connection
    early and reports no length at all.
    """

    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"{label}: the server returned an empty object")
    with path.open("rb") as handle:
        head = handle.read(len(magic))
        if head != magic.encode("latin1"):
            raise ValueError(
                f"{label}: expected a {magic} payload and the first bytes "
                f"are {head!r} -- the host answered with something that is "
                "not this product")
        if magic == "GRIB":
            handle.seek(max(0, size - 4))
            if handle.read(4) != b"7777":
                raise ValueError(
                    f"{label}: the GRIB2 end marker is missing, so the "
                    f"transfer is truncated at {size} bytes")


def _object_token(obj: PlannedObject) -> str | None:
    """What to call one planned object to a person.

    ``f01 atmosphere`` rather than the relpath alone: with six transfers
    in flight, the lead and the role are what tell a reader which part
    of the request is moving, and the legacy routes have printed exactly
    this token for as long as they have printed anything.
    """

    parts = []
    if getattr(obj, "lead", None) is not None:
        parts.append(f"f{int(obj.lead):02d}")
    if getattr(obj, "role", None):
        parts.append(str(obj.role))
    return " ".join(parts) or None


def _admission_reporter(label: str, *, total: int, progress):
    """The per-file completion line, as an ``on_admitted`` callback.

    Named rather than inlined so its TEXT can be pinned by a test.  This
    is the line a reader has been parsing since the serial loop, and the
    start and in-flight lines added around it must not disturb it.
    """

    def landed(index: int, entry: dict) -> None:
        note = ("adopted" if entry.get("adopted") else
                "already present" if entry.get("reused")
                else f"{entry.get('bytes', 0) / (1024 * 1024):.1f} MiB")
        progress(progress_mod.format_transfer_done_line(
            label=label, index=index, total=total,
            name=entry.get("relpath", entry.get("name")), note=note))

    return landed


def _download_object(url: str, dest: Path, *, magic: str, opener=None,
                     timeout: float = 300.0, progress=None,
                     declared_size=None, streams=1) -> dict:
    """Move one object, verify it, and return its manifest entry.

    ``progress`` is called with the size of each chunk as it lands.  The
    table routes move objects that are hundreds of megabytes each, and
    without this the whole transfer was one silent gap between the
    route's opening line and its manifest (UX finding N10).
    """

    from woof.fetch_ranges import download

    dest = _io_path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        written, digest = download(url, part, user_agent=_USER_AGENT,
            streams=streams, timeout=timeout, opener=opener,
            progress=progress, declared_size=declared_size)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    _verify_payload(part, magic=magic, label=dest.name)
    part.replace(dest)
    return {"name": dest.name, "bytes": written, "sha256": digest,
            "url": url}


def _retry_delay(error: BaseException, attempt: int) -> float | None:
    """Bound transient recovery without repeatedly asking for absent objects.

    The network faults are the shared classification
    (:func:`woof.fetch_endpoints.retry_delay`).  A ``ValueError`` is this
    route's own: a payload that did not verify, which a fresh transfer
    can repair.
    """
    if isinstance(error, ValueError):
        return 2.0 ** attempt
    return fetch_endpoints.retry_delay(error, attempt,
                                       wait_limit_s=_RETRY_WAIT_LIMIT)


#: (source, endpoint) pairs already told that a paced host moves record-
#: selected objects whole: one line per run, not one per object.
_PACED_WHOLE_SAID: set[tuple[str, str]] = set()


def _paced_host(url: str) -> bool:
    """Does the shared NOMADS governor pace this URL's requests?"""

    from woof.nomads_governor import is_nomads_url

    return is_nomads_url(url)


def _download_along_ladder(plan: FetchPlan, obj: PlannedObject, dest: Path, *,
                           magic: str, fetch, opener, progress,
                           ladder: Sequence[Endpoint] | None = None,
                           records: bool = True) -> dict:
    """Move one object, asking each endpoint in turn until one serves.

    The rounds, the waits between them and the refusal once they are
    spent are the tree's one shared retry
    (:func:`woof.fetch_endpoints.ask_along_ladder`): a transient fault
    (a reset connection, a body cut short, HTTP 408, 429 or 5xx) is
    asked again 2, 4, 8 and 16 s later, a permanent refusal falls
    through to the next endpoint once, and a fault on this computer
    propagates unchanged.  This route adds one fault of its own: a
    payload that did not verify, which a fresh transfer can repair.

    ``ladder`` is this OBJECT's order, which is the request's ladder
    with any rung that provably already holds the object promoted to
    the head (see :func:`_probe_transfer_ladders`).  It is a reorder,
    never a shorter list, so everything below is unchanged by it.
    Exhaustion names each endpoint and preserves completed files.
    """

    part = dest.with_name(dest.name + ".part")

    def transfer(endpoint: Endpoint) -> dict:
        url = endpoint.url(obj.key) if obj.key else obj.url
        if obj.member_index is not None:
            from woof.member_index import download_indexed_member
            # The row's record_index narrows the member to the records the
            # mapping binds; a caller-substituted transport keeps the whole
            # member, as it keeps whole objects elsewhere.
            return download_indexed_member(
                url, endpoint.url(obj.index_key), dest,
                declaration=obj.member_index, member_ordinal=obj.member_ordinal,
                source=plan.source_id, member=plan.member, opener=opener,
                forecast_hour=obj.lead,
                records=obj.record_index if records else None)
        if obj.record_index is not None and records and _paced_host(url):
            # The NOMADS governor spaces every request a node sends 2.5 s
            # apart (woof.nomads_governor), so a selection read as tens
            # of byte ranges takes minutes there where the whole object
            # takes seconds: a GEFS member-lead is 85 ranges, about 3.5
            # min against ~40 s whole at NOMADS's measured 3 MB/s.  The
            # paced host moves the object whole; the decode reads the
            # same records from it.
            said = (plan.source_id, endpoint.name)
            if said not in _PACED_WHOLE_SAID:
                _PACED_WHOLE_SAID.add(said)
                progress(f"fetch {plan.source_id}: {endpoint.name} paces each request "
                         "2.5 s apart, so its objects move whole rather than as "
                         "selected records")
        elif obj.record_index is not None and records:
            from woof.record_index import IndexAbsent, download_records
            try:
                return download_records(
                    url, endpoint.url(obj.index_key), dest,
                    declaration=obj.record_index, opener=opener)
            except IndexAbsent as error:
                # The object may still be there: move it whole (the decode
                # reads the same records from it) rather than fail it.
                progress(f"fetch {plan.source_id}: {obj.relpath}: "
                         f"{endpoint.name} cannot serve the record selection "
                         f"({error}); moving the whole object")
        return fetch(url, dest, magic=magic, opener=opener)

    endpoint, entry = fetch_endpoints.ask_along_ladder(
        ladder or plan.ladder or (plan.host,), transfer,
        label=f"fetch {plan.source_id}", name=obj.relpath,
        progress=progress, delay=_retry_delay,
        discard=lambda _endpoint: part.unlink(missing_ok=True),
        attempts=_TRANSFER_ATTEMPTS,
        # No new round for a request another file has already failed,
        # and no waiting out this round's backoff to find that out.
        pause=lambda seconds: fetch_pool.sleep_unless_stopped(
            seconds, sleep=time.sleep),
        tail=(" Completed files are kept; start this forecast again to "
              "retry the remaining files."))
    return {**entry, "endpoint": endpoint.name}


def _probe_transfer_ladders(
        plan: FetchPlan, objects: Sequence[PlannedObject], *, probe,
        workers: int, progress) -> tuple[dict[str, tuple[Endpoint, ...]],
                                         dict | None]:
    """Ask the throughput rung which of these objects it already has.

    The measured cost this exists to remove: at peak hours the
    operational server paced whole-file transfers at about 3 MB/s per
    file, so a 3.4 GB request took ~20 min where the archive had served
    the same volume in ~3.  The archive is the same bytes under the
    same key; the only reason to pay that is an object the archive does
    not have YET, which is exactly what a HEAD settles.

    One HEAD per object -- milliseconds against a multi-hundred-megabyte
    transfer -- and they run AHEAD of the transfers through the same
    pool, under the same per-host caps, so no probe ever waits behind
    a download.  Probing decides ORDER only: the returned ladder for an
    object is the request's ladder reordered, so every rung is still
    behind the chosen one and a probe that says no (a 404 because the
    mirror lags, a 503 because it is throttling) costs the transfer
    nothing at all.

    Returns ``(per-object ladders, probe receipt)``; both are empty when
    there was nothing to choose between -- a pinned ``--transport``, a
    one-rung ladder, or a source whose ladder head is already its
    quickest host.
    """

    ladder = plan.ladder or (plan.host,)
    if plan.pinned_host is not None or not objects:
        return {}, None
    candidates = fetch_endpoints.transfer_probes(ladder)
    if not candidates:
        return {}, None

    def ask(obj: PlannedObject) -> dict:
        rungs = (fetch_endpoints.transfer_ladder(
            ladder, (obj.key,), probe=probe) if obj.key else ladder)
        return {"relpath": obj.relpath, "ladder": rungs}

    entries, _receipt = fetch_pool.run_transfers(
        [fetch_pool.TransferJob(
            name=obj.relpath, url=candidates[0].url(obj.key),
            action=functools.partial(ask, obj))
         for obj in objects],
        workers=workers)

    ladders = {entry["relpath"]: entry["ladder"] for entry in entries
               if entry["ladder"][0] is not ladder[0]}
    preferred = candidates[0]
    available = len(ladders)
    if available:
        progress(
            f"fetch {plan.source_id}: mirrored: taking the archive for "
            f"throughput -- {available} of {len(objects)} object"
            f"{'' if len(objects) == 1 else 's'} "
            f"{'is' if available == 1 else 'are'} already on "
            f"{preferred.name} ({preferred.host})"
            + (f"; the rest from {ladder[0].name}, which publishes before "
               "the mirrors" if available < len(objects) else ""))
    else:
        progress(
            f"fetch {plan.source_id}: {preferred.name} has not caught up "
            f"with this cycle -- using {ladder[0].name}, which publishes "
            "before the mirrors")
    return ladders, {"endpoint": preferred.name, "objects": len(objects),
                     "available": available}


def _prior_entries(out: Path) -> dict[str, dict]:
    out = _io_path(out)
    manifest = out / MANIFEST_NAME
    if not manifest.is_file():
        return {}
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(entry["relpath"]): entry
            for entry in document.get("files", [])
            if isinstance(entry, dict) and entry.get("relpath")}


def _recovery_path(out: Path, obj: PlannedObject) -> Path:
    token = hashlib.sha256(obj.relpath.encode("utf-8")).hexdigest()
    return _io_path(out / _RECOVERY_DIRECTORY / f"{token}.json")


def has_recovery_request(out: Path) -> bool:
    """Whether an interrupted managed download recorded its input identity."""
    out = _io_path(out)
    try:
        record = json.loads((out / _RECOVERY_REQUEST_NAME).read_text(encoding="utf-8"))
        return (record.get("schema") == _RECOVERY_SCHEMA
                and isinstance(record.get("request"), dict)
                and all(key in record["request"]
                        for key in ("source", "cycle", "host", "member")))
    except (OSError, ValueError, AttributeError, TypeError):
        return False


def _recovery_entry(out: Path, plan: FetchPlan, obj: PlannedObject) -> dict | None:
    try:
        record = json.loads(_recovery_path(out, obj).read_text(encoding="utf-8"))
        wanted = _request_identity(plan)
        if (record.get("key") != obj.key
                or any(record.get("request", {}).get(key) != wanted[key]
                       for key in ("source", "cycle", "host", "member"))):
            return None
        entry = record.get("file")
        return entry if isinstance(entry, dict) else None
    except (OSError, ValueError, AttributeError, TypeError):
        return None


def _verified_reuse(dest: Path, entry: dict | None, *, magic: str) -> bool:
    dest = _io_path(dest)
    if not entry or not dest.is_file() or dest.stat().st_size != entry.get("bytes"):
        return False
    try:
        _verify_payload(dest, magic=magic, label=dest.name)
    except ValueError:
        return False
    with dest.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return digest == entry.get("sha256")


def _reusable_entry(out: Path, plan: FetchPlan, obj: PlannedObject,
                    prior: Mapping[str, dict]) -> dict | None:
    """The receipt entry OBJ is reused from, or None when it must move."""
    known = _recovery_entry(out, plan, obj) or prior.get(obj.relpath)
    if _verified_reuse(out / obj.relpath, known, magic=_magic_for(plan, obj.role)):
        return known
    return None


def request_cached(plan: FetchPlan, out: Path) -> bool:
    """Whether OUT already holds every object of PLAN, byte-verified.

    Read-only: nothing is created, moved or written.  A yes means
    :func:`run_plan` would reuse every object and ask no host for any of
    them, so a caller may skip the provider's publication check.  A
    directory recorded for a different request answers no.
    """
    out = _io_path(Path(out))
    if not out.is_dir():
        return False
    try:
        check_prior_request(out, plan)
    except ValueError:
        return False
    prior = _prior_entries(out)
    with ThreadPoolExecutor(max_workers=max(1, min(len(plan.objects), os.cpu_count() or 1))) as pool:
        return all(pool.map(lambda obj: _reusable_entry(out, plan, obj, prior) is not None,
                            plan.objects))


#: What an unpinned request records where it used to record one host.
#:
#: The guard below exists to stop two different CYCLES publishing one
#: ``SHA256SUMS``.  Recording the SERVED endpoint as request identity
#: would have made an ordinary fall-through -- the operational server
#: throttling for fifteen minutes, the archive finishing the job --
#: look like exactly that, and refuse the resume that was the remedy.
LADDER_IDENTITY = "ladder"


def _request_identity_fields(source: str, cycle: datetime, host: str | None,
                             member: str | None, leads) -> dict:
    return {
        "source": source,
        "cycle": f"{cycle:%Y-%m-%dT%H}Z",
        "host": host or LADDER_IDENTITY,
        "member": member,
        "leads": list(leads),
    }


def _request_identity(plan: FetchPlan) -> dict:
    return _request_identity_fields(plan.source_id, plan.cycle, plan.pinned_host,
                                    plan.member, plan.leads)


def check_prior_request(out: Path, plan: FetchPlan | None = None, *,
                        source: str | None = None, cycle: datetime | None = None,
                        host: str | None = None, member: str | None = None) -> None:
    """Refuse two requests in one directory, also during managed cache selection.

    The explicit identity form checks the same source/cycle/host/member
    contract without resolving transfer endpoints or building object lists.
    """
    out = _io_path(out)
    if plan is not None:
        if any(value is not None for value in (source, cycle, host, member)):
            raise ValueError("Supply a fetch plan or request identity, not both")
        wanted = _request_identity(plan)
        route = plan.route
    else:
        if source is None or cycle is None:
            raise ValueError("Request identity requires source and cycle")
        route = route_for(source)
        resolve_cycle(route, cycle)
        if host is not None:
            route.host(host)
        resolved_member, _ = resolve_member(route, member)
        wanted = _request_identity_fields(route.source_id, cycle, host, resolved_member, ())

    def differs(recorded: dict, key: str) -> bool:
        if recorded.get(key) == wanted[key]:
            return False
        if key != "host":
            return True
        # An unpinned request is compatible with a directory whose
        # prior receipt recorded whichever endpoint happened to serve
        # it -- including one written before this ArWen had a ladder.
        return not (wanted[key] == LADDER_IDENTITY
                    and recorded.get(key) in
                    {host.name for host in route.hosts})

    for name in (MANIFEST_NAME, _RECOVERY_REQUEST_NAME):
        try:
            prior = json.loads((out / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        recorded = prior.get("request", {})
        differing = [key for key in ("source", "cycle", "host", "member")
                     if differs(recorded, key)]
        if differing:
            detail = ", ".join(
                f"{key} {recorded.get(key)!r} -> {wanted[key]!r}"
                for key in differing)
            raise ValueError(
                f"--out {out} already holds a different request ({detail}).\n"
                "  remedy: fetch into a different --out, or pass "
                "--force-refetch to move the existing files aside (nothing is "
                "deleted) and re-download this request.\n"
                "  why: one directory publishes one SHA256SUMS and one input "
                "list, and a mixed directory would hand `woof prep` a series "
                "spanning two cycles.")


def _quarantine(out: Path, progress) -> Path | None:
    """Move an existing fetch aside; nothing is deleted."""

    entries = [entry for entry in sorted(out.iterdir())
               if not entry.name.startswith("quarantine-")]
    if not entries:
        return None
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    aside = out / f"quarantine-{stamp}"
    generation = 0
    while True:
        try:
            aside.mkdir()
            break
        except FileExistsError:
            # A fast retry or a repeated clock value must preserve every
            # earlier generation, including directories of source objects.
            generation += 1
            aside = out / f"quarantine-{stamp}-{generation}"
    moved = 0
    for entry in entries:
        entry.replace(aside / entry.name)
        moved += 1
    if moved:
        progress(f"fetch: --force-refetch moved {moved} entr"
                 f"{'y' if moved == 1 else 'ies'} to {aside}")
        return aside
    return None


def run_plan(plan: FetchPlan, *, out: Path, force: bool = False,
             file_workers: int | None = None, progress=print,
             opener=None, downloader=None, probe=None,
             lead_gate=None, on_lead=None) -> dict:
    """Move the planned objects, compose the primaries, write the receipts.

    Every file rides :mod:`woof.fetch_pool`, so a table route is
    parallel by default with bounded, host-capped, in-order admission.
    Completed objects get individual recovery receipts even after an
    earlier object fails. The preparation manifest is published only
    when the whole request completes.

    ``probe`` is the availability question the transfer host is chosen
    with -- ``url -> bool``, defaulting to one governed HEAD.  It runs
    ahead of the transfers, through the same pool, and only for the
    objects this run has still to download.

    The as-posted fetch (``woof fetch --as-posted``, DESIGN A136 2.3)
    passes two hooks.  ``lead_gate`` is called with an object's lead
    group before it moves and blocks until one host holds that whole
    lead, returning the rung that does (which then heads the object's
    ladder); it raises when the lead passes its late time, and its
    ``refetch(lead, error)`` is asked when a transfer did not verify, to
    wait a round and name the rung to fetch it from again.  The ahead
    probe is not run under it: it would ask hosts about leads not yet
    due.  ``on_lead(lead, entries, composed)`` fires in lead order once
    every object of a lead group is admitted, after that lead's
    composition and after the manifest is republished for the verified
    prefix with ``complete: false``; the last manifest says
    ``complete: true``.  Without the hooks nothing here changes.
    """

    out = _io_path(Path(out))
    out.mkdir(parents=True, exist_ok=True)
    if force:
        _quarantine(out, progress)
    else:
        check_prior_request(out, plan)
    prior = _prior_entries(out)
    # The request and per-object receipts survive a failed pool. They do
    # not publish a preparation manifest for an incomplete forecast.
    _write_json(out / _RECOVERY_REQUEST_NAME, {
        "schema": _RECOVERY_SCHEMA, "request": _request_identity(plan)})
    (out / _RECOVERY_DIRECTORY).mkdir(exist_ok=True)
    # ONE byte counter for the whole request, not one per object: the
    # pool keeps several transfers in flight, and six interleaved
    # counters read worse than one.  Injected downloaders keep the
    # signature they always had -- the counter is bound to the default
    # transport only, so a route test's fake is called exactly as before.
    counter = ByteCounter(f"fetch {plan.source_id}")
    # WHICH FILES ARE MOVING, while they are moving.  The counter above
    # says how many bytes the request as a whole has moved and cannot say
    # whose; once the pool put six transfers in flight that became the
    # only per-file signal, arriving at completion.  Default-on: no flag
    # turns this on, and a bare run stops going quiet.
    monitor = progress_mod.TransferMonitor(f"fetch {plan.source_id}")
    workers = fetch_pool.resolve_file_workers(file_workers)
    fetch = downloader
    if fetch is None:
        # ``declared_size`` is bound to the DEFAULT transport only, for
        # the same reason ``progress`` is: an injected downloader keeps
        # the four-argument signature it has always had, so a route
        # test's fake is called exactly as before.
        fetch = functools.partial(_download_object,
                                  progress=counter.advance,
                                  declared_size=monitor.declare_for_path,
                                  streams=fetch_pool.chunk_streams_per_file(
                                      workers, files=len(plan.objects)))

    ladder = plan.ladder or (plan.host,)
    # A file row's record selection rides the route's own transports (the
    # default and the Rust object door, which says so); a caller that
    # substitutes its own transport (an offline replay, a standby store)
    # keeps moving whole objects, as it always has.
    records = downloader is None or bool(getattr(downloader, "route_transport", False))

    # Verify files across the available cores. Unrecorded files can be
    # adopted against the host; their HEAD/index responses are capped at
    # 16 open connections independently of the hashing width.
    from woof.fetch_adoption import adopt

    def local_entry(obj):
        known = _reusable_entry(out, plan, obj, prior)
        if known is not None:
            return {**known, "adopted": False, "reused": True,
                    "downloaded": False}
        dest = out / obj.relpath
        # A failed recorded checksum is damage, not a new unmanaged file.
        if (not dest.is_file() or _recovery_entry(out, plan, obj)
                or prior.get(obj.relpath)):
            return None
        entry = adopt(dest, obj, ladder=ladder, magic=_magic_for(plan, obj.role),
                      verify_payload=_verify_payload, opener=opener, progress=progress)
        if entry is not None:
            entry.update(relpath=obj.relpath, role=obj.role, lead=obj.lead,
                         adopted=True, reused=False, downloaded=False)
            _write_json(_recovery_path(out, obj), {
                "request": _request_identity(plan), "key": obj.key, "file": entry})
        return entry

    reuse: dict[str, dict] = {}
    pending: list[PlannedObject] = []
    verification_workers = max(1, min(len(plan.objects), os.cpu_count() or 1))
    verification_started = perf_counter()
    with ThreadPoolExecutor(max_workers=verification_workers) as pool:
        local = list(pool.map(local_entry, plan.objects))
    verification_seconds = perf_counter() - verification_started
    for obj, known in zip(plan.objects, local):
        if known is not None:
            reuse[obj.relpath] = known
        else:
            pending.append(obj)

    # AHEAD of the transfers, not behind them: the throughput rung is
    # asked which of the pending objects it already holds, and each one
    # that it does takes it.  See _probe_transfer_ladders.
    if lead_gate is None:
        promoted, probe_receipt = _probe_transfer_ladders(
            plan, pending, workers=workers, progress=progress,
            probe=(fetch_endpoints.object_available if probe is None
                   else probe))
    else:
        promoted, probe_receipt = {}, None
    # The lead each object is fetched and published with: a lead-free
    # object and a step-0 object travel with the window's first lead,
    # as the plan orders them.
    groups = [obj.lead if obj.lead in plan.leads else plan.leads[0]
              for obj in plan.objects]

    def _headed(rungs, name):
        if not name:
            return rungs
        return (tuple(rung for rung in rungs if rung.name == name)
                + tuple(rung for rung in rungs if rung.name != name))

    adopted = sum(bool(entry.get("adopted")) for entry in reuse.values())
    reused = len(reuse) - adopted
    jobs = []
    for obj in plan.objects:
        dest = out / obj.relpath
        magic = _magic_for(plan, obj.role)
        known = reuse.get(obj.relpath)
        if known is not None:

            def _reuse(entry=known, relpath=obj.relpath, role=obj.role,
                       lead=obj.lead) -> dict:
                return {**entry, "relpath": relpath, "role": role,
                        "lead": lead}

            jobs.append(fetch_pool.TransferJob(
                name=obj.relpath, url=None, action=_reuse,
                token=_object_token(obj),
                expected_bytes=known.get("bytes")))
            continue

        rungs = promoted.get(obj.relpath, ladder)

        def _get(obj=obj, dest=dest, magic=magic, relpath=obj.relpath,
                 role=obj.role, lead=obj.lead, rungs=rungs,
                 group=groups[len(jobs)]) -> dict:
            if lead_gate is not None:
                rungs = _headed(rungs, lead_gate(group))
            if dest.exists():
                from woof.fetch_guard import quarantine
                aside = quarantine(dest)
                progress(f"fetch {plan.source_id}: moved rejected file aside to {aside.name}")
            while True:
                try:
                    entry = _download_along_ladder(
                        plan, obj, dest, magic=magic, fetch=fetch,
                        opener=opener, progress=progress, ladder=rungs,
                        records=records)
                except ValueError as error:
                    # A payload that did not verify, or every rung
                    # refusing it once its own retries are spent.  A
                    # HEAD can answer before the server has finished
                    # writing, so under the gate it is asked again next
                    # round until the lead's late time.
                    if lead_gate is None:
                        raise
                    rungs = _headed(rungs, lead_gate.refetch(group, error))
                    continue
                break
            entry = {**entry, "relpath": relpath, "role": role, "lead": lead,
                     "reused": False, "adopted": False, "downloaded": True}
            _write_json(_recovery_path(out, obj), {
                "request": _request_identity(plan), "key": obj.key, "file": entry})
            return entry

        # The politeness key is the host this object will ACTUALLY be
        # asked first, not the ladder's head: counting a mirrored
        # transfer against the operational server's cap of 2 would
        # throttle the fetch to the pace of the host it just avoided.
        jobs.append(fetch_pool.TransferJob(
            name=obj.relpath, url=rungs[0].url(obj.key) if obj.key
            else obj.url, action=_get,
            token=_object_token(obj),
            # WHERE IT LANDS, so the in-flight byte count can be read off
            # the growing file when the transport says nothing of its
            # own; it costs a stat().
            path=dest))

    # WHERE THE BYTES ARE ACTUALLY COMING FROM, not where the ladder
    # starts.  Naming the ladder's head here would contradict the
    # mirrored note printed a moment ago, and an opening line that
    # disagrees with the receipt is worse than no opening line.
    heads: list[Endpoint] = []
    for obj in plan.objects:
        head = promoted.get(obj.relpath, ladder)[0]
        if head not in heads:
            heads.append(head)
    serving = (f"{heads[0].name} ({heads[0].base})" if len(heads) == 1
               else " and ".join(entry.name for entry in heads))
    behind = ", then ".join(entry.name for entry in ladder
                            if entry not in heads)
    progress(
        f"fetch {plan.source_id}: {len(plan.objects)} object"
        f"{'' if len(plan.objects) == 1 else 's'} from {serving}, cycle "
        f"{plan.cycle:%Y-%m-%dT%H}Z, leads "
        f"f{plan.leads[0]:03d}..f{plan.leads[-1]:03d}"
        + (f", member {plan.member}" if plan.member else "")
        + (f"; {behind} behind it" if behind else "")
        + (f"; {adopted} adopted" if adopted else "")
        + (f"; {reused} already present" if reused else ""))

    # WHAT IT IS DOING, per object, as the verified prefix grows.  The
    # measured shape of this finding: a 792 MB hrrr-prs request printed
    # its opening line and then nothing at all until the manifest, so a
    # slow link and a hung command looked identical (UX finding N10).
    _landed = _admission_reporter(f"fetch {plan.source_id}", total=len(jobs),
                                  progress=progress)

    admitted: list[dict] = []
    composed_so_far: list[dict] = []
    remaining: dict[int, int] = {}
    for group in groups:
        remaining[group] = remaining.get(group, 0) + 1

    def _admitted(index: int, entry: dict) -> None:
        _landed(index, entry)
        if on_lead is None:
            return
        admitted.append(entry)
        group = groups[index]
        remaining[group] -= 1
        if remaining[group]:
            return
        from dataclasses import replace as _replace

        lead_steps = tuple(step for step in plan.compose if step.lead == group)
        composed_now = (_run_compose(_replace(plan, compose=lead_steps), out,
                                     progress=progress)
                        if lead_steps else [])
        composed_so_far.extend(composed_now)
        _write_route_manifest(out, _plan_manifest(
            plan, list(admitted), list(composed_so_far), None,
            complete=False))
        _write_sha256sums(out, admitted, composed_so_far)
        on_lead(group, [item for item, owner in zip(admitted, groups)
                        if owner == group], composed_now)

    try:
        entries, receipt = fetch_pool.run_transfers(
            jobs, workers=workers, on_admitted=_admitted, monitor=monitor)
    finally:
        monitor.close()
        counter.close()

    composed = (composed_so_far if on_lead is not None
                else _run_compose(plan, out, progress=progress))
    receipt.update(verification_workers=verification_workers,
                   verification_seconds=round(verification_seconds, 6),
                   adoption_connection_cap=16)
    payload = _plan_manifest(plan, entries, composed, receipt,
                             probe_receipt=probe_receipt,
                             complete=True if on_lead is not None else None)
    _write_route_manifest(out, payload)
    _write_sha256sums(out, entries, composed)
    return payload


def _plan_manifest(plan: FetchPlan, entries, composed, receipt, *,
                   probe_receipt=None, complete: bool | None = None) -> dict:
    """The route manifest for ``entries``; ``complete`` only on an as-posted fetch."""

    acquisition = {}
    for kind, flag in (("adopted", "adopted"), ("reused", "reused"),
                       ("downloaded", "downloaded")):
        selected = [entry for entry in entries if entry.get(flag)]
        acquisition[f"{kind}_files"] = len(selected)
        acquisition[f"{kind}_bytes"] = sum(entry.get("bytes", 0) for entry in selected)
    payload = {
        "schema": ROUTE_MANIFEST_SCHEMA,
        "route_table_sha256": packaged_route_table_sha256(),
        "request": _request_identity(plan),
        "label": plan.route.label,
        "mode": "full-file",
        # WHERE THE BYTES CAME FROM.  A receipt that named only the
        # ladder's head would be a claim about intent, not provenance:
        # a fall-through mid-request is normal and must be readable
        # afterwards, per file and in summary.
        "endpoints": _endpoint_receipt(plan, entries, probe=probe_receipt),
        "files": entries,
        "composed": composed,
        "concurrency": {**(receipt or {}), **acquisition},
        "donors": [
            {"source": donor.source, "role": donor.role,
             "cycle": f"{donor.cycle:%Y-%m-%dT%H}Z",
             "leads": list(donor.leads), "why": donor.why}
            for donor in plan.donors],
        "prep": {
            "source": str(plan.route.prep.get("source", plan.source_id)),
            "member_set": plan.member_set,
            "supplement_role": plan.supplement_role,
            "primary_files": [str(path) for path in plan.primary_files],
            "supplement_files": [str(path)
                                 for path in plan.supplement_files],
        },
    }
    if complete is not None:
        # The as-posted fetch republishes this file as each lead lands;
        # a reader tells the verified prefix from the whole window here.
        payload["complete"] = bool(complete)
    return payload


def _endpoint_receipt(plan: FetchPlan,
                      entries: Sequence[Mapping[str, object]],
                      *, probe: Mapping[str, object] | None = None) -> dict:
    """Which endpoints this request considered, and which ones served.

    ``transfer_preference`` is the throughput order the table declares
    and ``probe`` is what the availability question actually answered,
    so a reader can tell "the mirror served because it had the object"
    from "the mirror served because the operational server failed" --
    two very different runs that name the same host.
    """

    ladder = plan.ladder or (plan.host,)
    served: list[str] = []
    for entry in entries:
        name = entry.get("endpoint")
        if isinstance(name, str) and name not in served:
            served.append(name)
    return {
        "pinned": plan.pinned_host,
        "considered": [endpoint.name for endpoint in ladder],
        "transfer_preference": [
            endpoint.name
            for endpoint in fetch_endpoints.transfer_order(ladder)],
        "served": served,
        "probe": (dict(probe) if probe is not None else None),
        "ladder": [
            {"name": endpoint.name, "base": endpoint.base,
             "host": endpoint.host,
             "retention_hours": endpoint.retention_hours,
             "transfer_rank": endpoint.transfer_rank,
             "why": endpoint.why}
            for endpoint in ladder],
    }


def _composed_already(dest: Path, out: Path, parts) -> tuple[str, int] | None:
    """``(sha256, bytes)`` when ``dest`` already holds exactly the parts' concatenation, else None."""

    try:
        if not dest.is_file():
            return None
        want = hashlib.sha256()
        size = 0
        for part in parts:
            data = (out / part.relpath).read_bytes()
            want.update(data)
            size += len(data)
        if dest.stat().st_size != size:
            return None
        have = hashlib.sha256()
        with dest.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                have.update(chunk)
    except OSError:
        return None
    return (want.hexdigest(), size) if have.hexdigest() == want.hexdigest() else None


def _run_compose(plan: FetchPlan, out: Path, *, progress=print) -> list[dict]:
    """Concatenate each composed valid time from its parts.

    A composed file that already holds exactly its parts' bytes is left as
    it is, and a changed one is written beside it and renamed into place.
    THE BREAKAGE THIS PREVENTS (WOOF sensitivity smoke, 2026-10-25,
    three gem-gdps members of one cycle with different physics): members of
    one ensemble share one download folder (recipe_door._SharedDownloads)
    and each member's chain runs this fetch again; the in-place rewrite of
    ``composed/gdps_*.grib2`` by a later member moved the file an earlier
    member's preparation had hashed, and that member's deferred re-read
    refused with "authority changed while it was re-read". Every composed
    route (gem-gdps, ecmwf-open-data-0p1, the GEFS routes) was exposed.
    """
    composed: list[dict] = []
    for step in plan.compose:
        if step.kind != "concat_per_lead":
            raise ValueError(
                f"{ROUTE_TABLE_NAME}: unknown compose kind {step.kind!r}")
        dest = out / step.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        already = _composed_already(dest, out, step.parts)
        if already is not None:
            composed.append({
                "name": step.name, "bytes": already[1],
                "sha256": already[0], "lead": step.lead,
                "parts": [part.relpath for part in step.parts]})
            continue
        digest = hashlib.sha256()
        written = 0
        temporary = dest.with_name(dest.name + ".composing")
        with temporary.open("wb") as handle:
            for part in step.parts:
                data = (out / part.relpath).read_bytes()
                handle.write(data)
                digest.update(data)
                written += len(data)
        _verify_payload(temporary, magic="GRIB", label=step.name)
        replace_file_with_retry(temporary, dest)
        composed.append({
            "name": step.name, "bytes": written,
            "sha256": digest.hexdigest(), "lead": step.lead,
            "parts": [part.relpath for part in step.parts]})
    if composed:
        progress(f"fetch {plan.source_id}: composed {len(composed)} valid "
                 f"time{'' if len(composed) == 1 else 's'} from "
                 f"{sum(len(item['parts']) for item in composed)} objects")
    return composed


def _write_route_manifest(out: Path, payload: Mapping[str, object]) -> None:
    # Preparation binds the bytes of this authority document. Transfer timing
    # and pool width are observations, not input content: putting them here
    # invalidates an otherwise identical preparation on every cached fetch.
    material = dict(payload)
    receipt = material.pop("concurrency", None)
    # A cached object has the same authority as a freshly acquired object.
    # Keep reuse observations in the returned payload and recovery receipts.
    if "files" in material:
        material["files"] = [
            {key: value for key, value in entry.items()
             if key not in {"reused", "adopted", "downloaded", "adoption"}}
            for entry in material["files"]
        ]
    if receipt is not None:
        _write_json(out / "fetch-pool-receipt.json", receipt)
    _write_json(out / MANIFEST_NAME, material)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path = _io_path(path)
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    replace_file_with_retry(temporary, path)


def _write_sha256sums(out: Path, entries: Sequence[Mapping[str, object]],
                      composed: Sequence[Mapping[str, object]]) -> None:
    lines = [f"{entry['sha256']}  {entry['relpath']}" for entry in entries]
    lines.extend(f"{item['sha256']}  {item['name']}" for item in composed)
    (out / SHA256SUMS_NAME).write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------
# The handoff: what makes a fetched directory a front door
# --------------------------------------------------------------------------

def write_handoff(plan: FetchPlan, out: Path, *,
                  donor_files: Mapping[str, Path] | None = None,
                  posting: Path | None = None
                  ) -> tuple[Path, Path]:
    """Write the ordered ``--input-list`` and the bound prep command.

    A directory of verified bytes is not yet a front door: the caller
    still has to know which of them are the primaries, in what order,
    which one the composition binds as its surface supplement, and under
    which role.  All four are table facts, so they are written down here
    rather than left for a reader to reconstruct -- and the
    ``--input-list`` spelling is what keeps a field-per-file source's
    hundreds of inputs inside the 32 KB Windows command line.

    ``posting`` is the as-posted fetch's ``posting/`` folder: the handoff
    is then written before the window's leads move (every path in it is
    a table fact, the donors are in), so a preparation can start beside
    the fetch and wait on each lead's marker there (DESIGN A136 2.3
    step 2).
    """

    out = _io_path(Path(out))
    inputs = out / INPUT_LIST_NAME
    inputs.write_text(
        "".join(f"{(out / path).resolve()}\n" for path in plan.primary_files),
        encoding="utf-8", newline="\n")

    prep_source = str(plan.route.prep.get("source", plan.source_id))
    tokens: list[str] = ["--source", prep_source,
                        "--input-list", str(inputs.resolve())]
    for path in plan.supplement_files:
        binding = (f"{plan.supplement_role}={(out / path).resolve()}"
                   if plan.supplement_role else str((out / path).resolve()))
        tokens += ["--supplement", binding]
    for role, path in plan.extra_supplement_files:
        tokens += ["--supplement", f"{role}={(out / path).resolve()}"]
    unfetched: list[DonorRequest] = []
    for donor in plan.donors:
        supplied = (donor_files or {}).get(donor.role)
        if supplied is None:
            unfetched.append(donor)
            continue
        tokens += ["--supplement", f"{donor.role}={Path(supplied).resolve()}"]
    tokens += ["--author-input-manifest", str(out.resolve() / "inputs.json")]
    # The text rendering below derives from the SAME tokens, so the two
    # spellings of this handoff cannot disagree.
    arguments = [f"{flag} {_q(value)}"
                 for flag, value in zip(tokens[::2], tokens[1::2])]

    member_step = None
    if plan.member_set:
        member_step = {
            "set": plan.member_set, "member": plan.member,
            "cycle": f"{plan.cycle:%Y-%m-%dT%H}",
            "steps": list(plan.leads),
            "inputs": str((out / "upstream").resolve()),
            "output": str((out / "members").resolve()),
            "input_list_after": str((out / "member-input-list.txt").resolve()),
        }
    verification_set = (source_adapters.get_source_adapter(
        plan.source_id).member_set if plan.member is not None else None)
    member_verification = ({"set": verification_set, "member": plan.member}
                           if verification_set is not None else None)
    write_prep_arguments(
        out, source=plan.source_id, prep_source=prep_source,
        cycle=plan.cycle, tokens=tokens,
        unbound_roles=[donor.role for donor in unfetched],
        member=plan.member, member_set=verification_set,
        member_prep=member_step, member_verification=member_verification,
        posting=posting)

    header = [
        f"# {plan.route.label}",
        f"# cycle {plan.cycle:%Y-%m-%dT%H}Z, leads "
        f"f{plan.leads[0]:03d}..f{plan.leads[-1]:03d}, endpoints "
        + " then ".join(entry.name
                        for entry in (plan.ladder or (plan.host,)))
        + f" (see {MANIFEST_NAME} for which one served each file)",
    ]
    if plan.member:
        header.append(f"# member {plan.member}")
    if plan.member_set:
        header.append("#")
        header.append(
            "# Automatic prepared chains verify and select this member before prep.")
        header.append(
            "# For a standalone preparation, first run")
        header.append(
            f"#   woof-member-prep --member-set {_q(plan.member_set)} "
            f"--member {_q(plan.member)} \\")
        header.append(
            f"#     --cycle {plan.cycle:%Y-%m-%dT%H} "
            f"--steps {','.join(str(lead) for lead in plan.leads)} --inputs "
            f"{_q(out.resolve() / 'upstream')} "
            f"--output {_q(out.resolve() / 'members')}")
        header.append(
            "# first, and point --input-list at the verified member tree "
            "it publishes.")
    for donor in unfetched:
        header.append("#")
        header.append(
            f"# STILL NEEDED: --supplement {donor.role}=<a {donor.source} "
            f"{'/'.join(f'f{lead:03d}' for lead in donor.leads)} analysis "
            f"for {plan.cycle:%Y-%m-%dT%H}Z>")
        header.append(f"#   why: {donor.why}")
    header.append("#")
    header.append("# yours to supply: --wps-namelist, --experiment-config,")
    header.append("#                  --geog-root, --output-root")

    body = ["woof prep \\"]
    body.extend(f"  {argument} \\" for argument in arguments[:-1])
    body.append(f"  {arguments[-1]}")

    command = out / PREP_COMMAND_NAME
    command.write_text("\n".join(header + [""] + body) + "\n",
                       encoding="utf-8", newline="\n")
    return inputs, command


def write_prep_arguments(out: Path, *, source: str, prep_source: str,
                         cycle: datetime, tokens: Sequence[str],
                         unbound_roles: Sequence[str] = (),
                         member: str | None = None,
                         member_set: str | None = None,
                         member_prep: Mapping[str, object] | None = None,
                         member_verification: Mapping[str, object] | None = None,
                         posting: Path | None = None) -> Path:
    """Publish the bound preparation arguments from any acquisition path.

    ``posting``: the window is fetched as posted, and its lead markers land
    in that folder; the document says so (``as_posted``, ``posting``) so a
    chain preparing beside the fetch knows where to wait.
    """
    document = {
        "schema": PREP_ARGUMENTS_SCHEMA,
        "source": source,
        "prep_source": prep_source,
        "cycle": f"{cycle:%Y-%m-%dT%H}",
        "argv": list(tokens),
        "caller_supplies": ["--wps-namelist", "--experiment-config",
                            "--geog-root", "--output-root"],
        "unbound_supplement_roles": sorted(set(unbound_roles)),
        "member": member,
        "member_set": member_set,
    }
    if member_prep is not None:
        document["member_prep"] = dict(member_prep)
    if member_verification is not None:
        document["member_verification"] = dict(member_verification)
    if posting is not None:
        document["as_posted"] = True
        document["posting"] = str(Path(posting).resolve())
    path = _io_path(Path(out)) / PREP_ARGUMENTS_NAME
    _write_json(path, document)
    return path


def in_band_supplement_role(source: str) -> str | None:
    """Read an in-band terrain binding from the packaged composition."""
    from woof.source_authorities import packaged_composition, packaged_profile
    adapter = source_adapters.get_source_adapter(source)
    if adapter.packaged_profile is None:
        return None
    profile = packaged_profile(adapter.packaged_profile)
    if profile["composition_state"] != "composed":
        return None
    composition = packaged_composition(adapter.packaged_profile)
    terrain = composition.get("supplements", {}).get("terrain_height")
    if (terrain and terrain.get("format") == profile["source_format"]
            and terrain.get("selector_authority") == "mapping_field_exact"
            and not composition.get("field_sources")):
        return str(terrain["data_role"])
    return None


def prepares_through_packaged_composition(source: str) -> bool:
    """Whether a packaged composition drives this source's preparation.

    The fact the container writer forks on.  It used to fork on the
    container's NAME, which is the same test only for as long as exactly
    one of the two containers has a composed profile: a name cannot say
    whether a composition is there to write a handoff from, and
    :func:`publishes_prep_handoff` already promises callers a document
    from this fact.
    """

    from woof.source_authorities import packaged_profile

    adapter = source_adapters.get_source_adapter(source)
    if adapter.packaged_profile is None:
        return False
    try:
        return packaged_profile(
            adapter.packaged_profile)["composition_state"] == "composed"
    except (ValueError, KeyError):
        return False


def publishes_prep_handoff(source: str) -> bool:
    """Whether the implemented acquisition path publishes bound prep arguments."""
    adapter = source_adapters.get_source_adapter(source)
    if adapter.acquisition_owner and adapter.acquisition_prep_handoff:
        return True
    source = canonical_source(source)
    adapter = source_adapters.get_source_adapter(source)
    if adapter.acquisition_owner:
        from importlib import import_module
        return bool(getattr(import_module(adapter.acquisition_owner),
                            "PUBLISHES_PREP_HANDOFF", False))
    from woof.cf_archive_fetch import sources
    if source in sources():
        return True
    if source in route_ids():
        return True
    # The container writer emits a mapped handoff when its composition
    # selects its surface fields from the same input files.
    return source in LEGACY_ROUTE_SOURCES and in_band_supplement_role(source) is not None


def _q(value) -> str:
    text = str(value)
    return text if all(ch not in text for ch in ' \t"\'') else shlex.quote(text)


def render_prep_command(argv: Sequence[str]) -> str:
    """One ``woof prep`` line from a bound argument vector.

    Every printed handoff renders through here, so a route's printed line
    and its written document are two spellings of the same tokens.
    """

    return "woof prep " + " ".join(_q(token) for token in argv)


def named_flags(flags: Sequence[str]) -> str:
    """``--a, --b and --c``: the flags a handoff leaves to its reader."""

    flags = list(flags)
    if len(flags) < 2:
        return "".join(flags)
    return ", ".join(flags[:-1]) + " and " + flags[-1]


def prep_handoff_lines(label: str, out: Path) -> tuple[str, ...]:
    """The ``next:`` block for a fetch that published its prep arguments.

    Read back from the published ``prep-arguments.json``, not rebuilt:
    that document is what the chained doors compose from, so the printed
    line is its argv verbatim and the flags named as the reader's are
    its ``caller_supplies``.  The line this replaced printed
    ``--source`` and ``--input-list`` only and pointed at
    ``prep-command.txt`` for the rest, so pasting it with the four flags
    it named was refused for the ``--supplement`` binding and the
    manifest flag it had left out.
    """

    out = Path(out)
    document = json.loads(
        _io_path(out / PREP_ARGUMENTS_NAME).read_text(encoding="utf-8"))
    lines = [
        f"fetch {label}: next: feed the mapped front door, source already "
        "bound:",
        "  " + render_prep_command(document["argv"]),
        f"  # {named_flags(document['caller_supplies'])} are yours.",
    ]
    lines.extend(
        f"  # still needed: --supplement {role}=FILE"
        for role in document.get("unbound_supplement_roles") or ())
    lines.append(
        "  # the same command, one flag per line, with the route's notes: "
        f"{(out / PREP_COMMAND_NAME).resolve()}")
    return tuple(lines)


def handoff_lines(plan: FetchPlan, out: Path) -> tuple[str, ...]:
    """The ``next:`` block the fetch front door prints when it finishes."""

    return prep_handoff_lines(plan.source_id, out)


__all__ = [
    "ComposePart", "ComposeRow", "ComposeStep", "DonorRequest", "DonorRow",
    "Endpoint", "FetchPlan", "FileRow", "Host", "INPUT_LIST_NAME",
    "LADDER_IDENTITY", "LEGACY_ROUTE_SOURCES", "MANIFEST_NAME", "PATH_TOKENS",
    "PREP_ARGUMENTS_NAME", "PREP_ARGUMENTS_SCHEMA",
    "PREP_COMMAND_NAME", "PlannedObject", "PublicationEra", "ROUTE_MANIFEST_SCHEMA",
    "ROUTE_TABLE_NAME", "ROUTE_TABLE_SCHEMA", "ROUTE_TABLE_SHA256", "Route",
    "SHA256SUMS_NAME", "all_fetchable_sources", "cadence_refusal",
    "window_cadence", "window_spacings",
    "check_prior_request",
    "LATER_READY_CHECKS", "LegacyPosting", "MIN_POLL_SECONDS", "POSTING_SHAPES",
    "LATE_BUDGET_MARGIN_MINUTES", "late_spread_minutes",
    "PostingRow", "READY_CHECKS", "WAITED_POSTING_SHAPES", "legacy_posting",
    "legacy_publication_lag", "posting_for", "posting_sources",
    "endpoint_ladder", "first_preparable_cycle", "handoff_lines", "ladder_for",
    "member_tokens", "named_flags", "packaged_route_table_sha256", "planning_cycles",
    "prep_handoff_lines", "publication_era", "refusal_ids", "render_prep_command",
    "resolve_cycle",
    "resolve_leads", "resolve_member", "resolve_mode", "resolve_request",
    "route_for", "route_ids", "run_plan", "table_route", "unknown_tokens",
    "write_handoff", "SOURCE_ROOT_FORMATS", "local_input_remedy",
    "local_prep_line", "sniff_format", "source_root_layout",
]
