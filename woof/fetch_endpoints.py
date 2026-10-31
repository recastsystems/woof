"""Which endpoints a cycle is asked for, and which one moves the bytes.

Every NCEP source publishes on two hosts that serve byte-identical
objects under byte-identical keys -- verified by HEAD against the live
services: the operational server and the AWS Open Data archive answer
the same relative key with the same ``Content-Length`` for GFS, GDAS,
GEFS, HRRR, RAP and RRFS alike.  So the choice between them is never
about the data.  It is about three things, and only three:

* **Freshness.**  The operational server publishes each object hours
  before any mirror has it.  A run initialized from the newest cycle
  is asking for exactly that.
* **Retention.**  The operational server keeps a bounded window --
  measured by counting the day directories the live tree serves, two
  days for HRRR/RAP/RRFS, four for GEFS, ten for the GFS tree -- and
  the archive keeps everything.
* **Throughput.**  The operational server paces bulk transfers and the
  archive does not.  Measured at peak hours: about 3 MB/s per file
  against the archive serving the same 3.4 GB in roughly a sixth of
  the wall clock.

All three are DATA, declared per endpoint in
``authorities/rw-wps-fetch-routes.v1.json``, and this module is the one
engine that reads them.  Adding a model, or moving a host, is a row
there; nothing here branches on a source name, and the ladder shape is
identical for a route whose transport is table-driven and for the three
whose transport predates the table.

**Two orders over the same rungs.**  :func:`serving_ladder` answers
"which endpoints is this cycle asked for, and in what order" from
retention alone: a latest initialization keeps both, a reanalysis-era
cycle goes straight to the archive without paying for an attempt that
was certain to 404.  :func:`transfer_order` answers a different
question -- "which of them should actually move the bytes" -- from the
declared ``transfer_rank``.

They differ for exactly one reason.  The operational server's
advantage is having the cycle FIRST, and that advantage is spent the
moment the mirror has the same object; what is left after that is
throughput, and the mirror wins it.  So for each requested object
inside the retention window the caller asks the throughput rung
whether it HAS that object -- one HEAD, milliseconds against a
multi-hundred-megabyte transfer -- and :func:`promote` moves it to the
head when it does.  Promotion REORDERS the ladder and never shortens
it: every other rung stays behind the chosen one, so fall-through, the
fault vocabulary and the whole-ladder refusal are unchanged, and a
probe that 404s or errors costs the transfer nothing.

Retention is an optimisation, never a bar.  When NO endpoint's window
covers the age -- a source with no archive behind it -- the whole
ladder is still tried, because a host that MIGHT still hold the cycle
beats a refusal that never asked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import errno
import functools
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import re
import socket
from types import MappingProxyType
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

#: The packaged acquisition authority.  One document: a source's cycle
#: and lead grammar, its file keys, and -- here -- its endpoints.
TABLE_NAME = "rw-wps-fetch-routes.v1.json"

#: Hosted workers opt into this policy; ordinary installs keep their defaults.
FETCH_POLICY_ENV = "WOOF_FETCH_POLICY"
#: Requires strict AWS pins while preserving the default streaming contract.
FETCH_POLICY_CONTRACT = "aws-streaming-v1"


def aws_fetch_policy() -> bool:
    """Whether this process requires AWS copies wherever the table has one."""

    return os.environ.get(FETCH_POLICY_ENV) == "aws"


def aws_transport_for_source(source_id: str) -> str | None:
    """The AWS endpoint's own transport name, or None when none is declared."""

    for endpoint in ladder(source_id):
        host = urlsplit(endpoint.base).hostname or ""
        if host == "amazonaws.com" or host.endswith(".amazonaws.com"):
            return endpoint.name
    return None


def policy_transport(source_id: str, requested: str | None = None) -> str | None:
    """The hosted policy's AWS pin, otherwise the caller's existing choice."""

    if aws_fetch_policy():
        return aws_transport_for_source(source_id) or requested
    return requested


def policy_uses_aws(source_id: str) -> bool:
    """Whether the hosted policy selects an AWS rung for this source."""

    return aws_fetch_policy() and aws_transport_for_source(source_id) is not None


def policy_posting_delay_minutes(source_id: str, lead: int) -> float:
    """The selected mirror's measured planning allowance, otherwise zero.

    Host rows carry lead-ranged allowances.  These metadata estimates shift
    the deadline, never postpone a positive availability probe.
    """

    if not policy_uses_aws(source_id):
        return 0.0
    selected = aws_transport_for_source(source_id)
    for endpoint in _rows(source_id):
        if endpoint["name"] != selected:
            continue
        for rule in endpoint.get("posting_delay_minutes", ()):
            first = int(rule.get("from_lead", 0))
            through = rule.get("through_lead")
            if int(lead) >= first and (through is None or int(lead) <= int(through)):
                return float(rule["minutes"])
    return 0.0

#: What an availability probe identifies itself as.  Its own string, so
#: a provider reading their logs can tell a HEAD that moved nothing
#: from the transfers that follow it.
PROBE_USER_AGENT = "gpuwm-fetch-probe/2.5 (+https://github.com/arwenweather)"

#: HTTP statuses that mean "ask the next endpoint", each for its own
#: reason: the host is refusing this client (403), does not have this
#: object (404), is rate limiting (429), or is failing/overloaded
#: (5xx).  None of them means the OBJECT is wrong -- the other host
#: publishes the same key -- so every one of them is a reason to move
#: on rather than to end the fetch.
FALLTHROUGH_STATUSES = MappingProxyType({
    403: "the host refused this client",
    404: "the host does not serve this object",
    429: "the host is rate limiting this client",
    500: "the host failed on its side",
    502: "the host's gateway failed",
    503: "the host is unavailable or throttling",
    504: "the host's gateway timed out",
})


def _table_path() -> Path:
    return Path(__file__).with_name("authorities") / TABLE_NAME


@functools.lru_cache(maxsize=1)
def document() -> Mapping[str, object]:
    """The packaged acquisition authority, parsed once."""

    return json.loads(_table_path().read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def archive_document() -> Mapping[str, object]:
    """Dated archive object grammars, shared with the Rust downloader."""
    path = _table_path().with_name("rw-wps-archive-layouts.v1.json")
    return json.loads(path.read_text(encoding="utf-8"))


def archive_rows(source_id: str):
    return tuple(row for row in archive_document()["layouts"]
                 if row["source_id"] == source_id)


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Endpoint:
    """One rung of a source's ladder.

    ``retention_hours`` is how far back this endpoint serves, measured
    against the live tree; ``None`` is an archive -- effectively
    unbounded.  ``transfer_rank`` is where this rung sits in the
    THROUGHPUT order (lower is quicker), which is a different order
    from the ladder and decides only which host moves the bytes; a row
    that declares none inherits its ladder position, so a source with
    no measured throughput difference transfers in table order.
    ``why`` is the sentence a refusal quotes, so a reader who has just
    been told every endpoint failed also learns what each one was for.
    """

    name: str
    base: str
    retention_hours: float | None
    why: str
    transfer_rank: int = 0
    layouts: tuple = field(default=(), hash=False)
    #: An archive that takes a cycle some hours after it runs (its layout row's ``available_after_hours``): it does
    #: not cover a younger cycle.  THE BREAKAGE THIS PREVENTS (WOOF production run_6452d3110cf0705f, 2026-10-26):
    #: GFS 06Z asked for at 07:29Z, before it posted, ended on NCAR GDEX's 404, an archive about two days behind.
    available_after_hours: float | None = None

    @property
    def host(self) -> str:
        """The politeness key: this endpoint's lower-cased netloc."""

        return urlsplit(self.base).netloc.lower()

    @property
    def archive(self) -> bool:
        return self.retention_hours is None

    def covers(self, age_hours: float) -> bool:
        """Does this endpoint still serve a cycle ``age_hours`` old?"""

        if self.available_after_hours is not None and age_hours < self.available_after_hours:
            return False
        if self.retention_hours is None:
            return True
        return age_hours <= float(self.retention_hours)

    def url(self, key: str) -> str:
        """The absolute URL for one rendered object key."""

        if self.layouts:
            match = re.fullmatch(self.layouts[0]["key_pattern"], key)
            if match is None:
                raise ValueError(f"{self.name}: object key does not match its declared archive grammar: {key}")
            tokens = {k: v or "" for k, v in match.groupdict().items()}
            stamp = f'{tokens["YYYY"]}-{tokens["MM"]}-{tokens["DD"]}T{tokens["HH"]}'
            for row in self.layouts:
                if (stamp >= row["record_start"]
                        and (not row.get("cycle_until") or stamp < row["cycle_until"])):
                    lead = int(tokens["FFF"])
                    segments = row.get("lead_segments", ())
                    if segments and not any(
                            lead <= seg["through"] and lead % seg["step"] == 0
                            and (i == 0 or lead > segments[i-1]["through"])
                            for i, seg in enumerate(segments)):
                        raise ValueError(f"{self.name}: lead {lead} is outside the archive's declared forecast ladder")
                    return row["base"] + "/" + row["path_template"].format(**tokens) + tokens.get("SUFFIX", "")
            raise ValueError(f"{self.name}: no declared archive layout for {stamp}")
        return f"{self.base}/{key.lstrip('/')}"

    def supports_cycle(self, cycle: datetime) -> bool:
        stamp = cycle.strftime("%Y-%m-%dT%H")
        return not self.layouts or any(
            stamp >= row["record_start"] and
            (not row.get("cycle_until") or stamp < row["cycle_until"])
            for row in self.layouts)

    @property
    def rust_source(self) -> str:
        return self.layouts[0]["rust_source"] if self.layouts else {"s3": "aws"}.get(self.name, self.name)

    @property
    def product(self) -> str:
        return self.layouts[0]["product"] if self.layouts else "pgrb2.0p25"

    @property
    def idx_available(self) -> bool:
        return not self.layouts or bool(self.layouts[0]["idx_available"])


def _endpoint(raw: Mapping[str, object], position: int) -> Endpoint:
    retention = raw.get("retention_hours")
    rank = raw.get("transfer_rank")
    after = [float(row["available_after_hours"]) for row in raw.get("layouts", ())
             if row.get("available_after_hours") is not None]
    return Endpoint(
        available_after_hours=max(after) if after else None,
        name=str(raw["name"]),
        base=str(raw["base"]),
        retention_hours=(None if retention is None else float(retention)),
        why=str(raw.get("why", "")),
        layouts=tuple(raw.get("layouts", ())),
        # No declared rank means "transfer in ladder order", which is
        # what every source outside the NCEP family does: the ladder
        # and the transfer order are then the same tuple, and nothing
        # is ever probed.
        transfer_rank=(position if rank is None else int(rank)),
    )


def _rows(source_id: str) -> tuple[Mapping[str, object], ...]:
    routes = dict(document().get("routes", {}))
    legacy = dict(document().get("legacy_ladders", {}))
    raw = (dict(routes[source_id])["hosts"] if source_id in routes
           else legacy.get(source_id, ()))
    hosts = [dict(row) for row in raw] if isinstance(raw, (list, tuple)) else []
    for row in archive_rows(source_id):
        if row.get("append_host", True) and not any(h["name"] == row["transport"] for h in hosts):
            hosts.append({"name": row["transport"], "base": row["base"],
                          "retention_hours": None, "transfer_rank": row["transfer_rank"],
                          "why": row["why"]})
    for host in hosts:
        host["layouts"] = tuple(row for row in archive_rows(source_id)
                                if row["transport"] == host["name"])
    return tuple(hosts)


@functools.lru_cache(maxsize=None)
def ladder(source_id: str) -> tuple[Endpoint, ...]:
    """``source_id``'s endpoints, in the order they are asked.

    Empty for a source with no declared endpoints at all (ERA5 is a
    manual CDS retrieval; there is no host to prefer).
    """

    rows = _rows(source_id)
    endpoints = tuple(_endpoint(row, position)
                      for position, row in enumerate(rows))
    for row, endpoint in zip(rows, endpoints):
        if row.get("default") and endpoint is not endpoints[0]:
            raise ValueError(
                f"{TABLE_NAME}: {source_id} marks {endpoint.name} as its "
                "default host but lists it behind another endpoint.  The "
                "ladder is its order, so the default must be its head.")
    return endpoints


def has_ladder(source_id: str) -> bool:
    return bool(ladder(source_id))


def endpoint_named(source_id: str, name: str) -> Endpoint:
    for endpoint in ladder(source_id):
        if endpoint.name == name:
            return endpoint
    offered = ", ".join(entry.name for entry in ladder(source_id))
    raise ValueError(
        f"--transport {name}: --source {source_id} publishes on {offered}")


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def cycle_age_hours(cycle: datetime, now: datetime | None = None) -> float:
    """How old ``cycle`` is, in hours, on a naive-UTC clock."""

    if now is None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
    if cycle.tzinfo is not None:
        cycle = cycle.astimezone(timezone.utc).replace(tzinfo=None)
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)
    return (now - cycle).total_seconds() / 3600.0


def serving_ladder(source_id: str, *, cycle: datetime,
                   now: datetime | None = None,
                   pinned: str | None = None,
                   throttle_fallthrough: bool = False) -> tuple[Endpoint, ...]:
    """The endpoints this cycle will actually be asked for, in order.

    A typed ``--transport`` is a DECISION, not a preference: it yields
    that endpoint alone, so an operator who named a host is never
    quietly served from the other one.  Otherwise the ladder is filtered
    to the endpoints whose retention covers the cycle's age -- and, when
    that filter would empty it, kept whole (see the module note:
    retention is an optimisation, never a bar).

    ``throttle_fallthrough`` is the TRANSFER ladder's question (asked by
    :func:`woof.fetch_routes.endpoint_ladder`): under the hosted AWS
    policy, a pinned mirror whose ``host_policy`` row declares a
    ``throttle`` keeps the cycle's other rungs behind it, so a file it
    answers "slow down" for is moved from the next rung.  Cycle and
    readiness questions leave it off and stay on the pinned mirror.
    """

    pinned = policy_transport(source_id, pinned)
    rungs = ladder(source_id)
    if not rungs:
        return ()
    age = cycle_age_hours(cycle, now)
    dated = tuple(entry for entry in rungs if entry.supports_cycle(cycle))
    covering = tuple(entry for entry in dated if entry.covers(age))
    if pinned is not None:
        head = endpoint_named(source_id, pinned)
        if (throttle_fallthrough and policy_uses_aws(source_id)
                and throttle_policy(head.host) is not None):
            # The hosted policy's AWS pin is kept at the head, but a mirror
            # whose host_policy row says it answers "slow down" keeps the
            # cycle's other rungs behind it.  Named breakage: ECMWF's AWS
            # mirror answered HEAD 200 and GET 503 SlowDown to ~90% of asks
            # on 2026-10-29 while Google and data.ecmwf.int served the same
            # keys, and the AWS-only ladder ended every IFS 9 km run on WOOF
            # at fetch (exit 75).  NOAA's AWS buckets declare no throttle
            # row, so they stay pinned alone and never fall through to NOMADS.
            return (head,) + tuple(
                entry for entry in (covering or dated) if entry is not head)
        return (head,)
    return covering or dated


# --------------------------------------------------------------------------
# Which rung moves the bytes: throughput, probed per object
# --------------------------------------------------------------------------

def transfer_order(rungs: Sequence[Endpoint]) -> tuple[Endpoint, ...]:
    """``rungs`` in THROUGHPUT order: quickest bulk host first.

    A second order over the same endpoints, and the only thing it
    decides is which host is asked to move an object it provably has.
    Ties keep ladder order, and a source that declares no rank has the
    two orders equal -- so this is the identity function for every
    publisher outside the NCEP family.
    """

    indexed = list(enumerate(rungs))
    indexed.sort(key=lambda item: (item[1].transfer_rank, item[0]))
    return tuple(entry for _position, entry in indexed)


def transfer_probes(serving: Sequence[Endpoint]) -> tuple[Endpoint, ...]:
    """The rungs worth asking "do you already have this object?".

    Only the rungs strictly quicker than the one the ladder would use
    anyway, in throughput order.  Empty means there is nothing to gain
    -- a one-rung ladder, a pinned ``--transport``, or a source whose
    ladder head is already its quickest host -- and an empty tuple is
    the instruction to probe NOTHING, which is what keeps an
    archive-era cycle and every non-NCEP route exactly as they were.
    """

    if len(serving) < 2:
        return ()
    head = serving[0]
    return tuple(entry for entry in transfer_order(serving)
                 if entry.transfer_rank < head.transfer_rank)


def promote(serving: Sequence[Endpoint],
            endpoint: Endpoint) -> tuple[Endpoint, ...]:
    """``serving`` with ``endpoint`` moved to the head, nothing dropped.

    The concrete breakage this prevents: choosing the mirror by
    REPLACING the ladder would leave a promoted object with one host
    and no fall-through, so a mirror that answered a HEAD and then
    threw a 503 at the transfer would refuse a fetch the operational
    server could have finished.  Promotion is a reorder.
    """

    rest = tuple(entry for entry in serving if entry is not endpoint)
    if len(rest) == len(serving):
        raise ValueError(
            f"{endpoint.name} is not one of the endpoints this cycle is "
            "served by, so it cannot be promoted ahead of them")
    return (endpoint,) + rest


def transfer_ladder(serving: Sequence[Endpoint], keys: Sequence[str], *,
                    probe) -> tuple[Endpoint, ...]:
    """One object's endpoint order: the quickest rung that HAS it, first.

    ``keys`` is everything the object needs from one host at once --
    usually one, but a route whose hour is a PAIR of published objects
    (HRRR's ``wrfnat`` + ``wrfprs``) must not promote a rung that has
    only half of it.  A rung is promoted when it answers for all of
    them.

    This is the whole selection rule, and it is deliberately one
    sequential function: each caller runs it inside its OWN pool, so
    the probes ride the concurrency and per-host caps that caller
    already has rather than a second scheduler invented here.

    A probe that raises is a probe that answered no.  It is not an
    endpoint failure -- nothing is recorded, nothing is spent, and the
    ladder simply stays in retention order with every rung intact.
    """

    for endpoint in transfer_probes(serving):
        try:
            if all(probe(endpoint.url(key)) for key in keys):
                return promote(serving, endpoint)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except BaseException:                 # noqa: BLE001 - see docstring
            continue
    return tuple(serving)


def object_available(url: str, *, opener=None, timeout: float = 60.0) -> bool:
    """Does ``url`` exist right now?  One HEAD, governed where it must be.

    True only for a 2xx.  Every other answer -- a 404 because the
    mirror has not caught up, a 503 because it is throttling, a refused
    connection -- is False, because to the ladder's promotion all of
    them mean the same thing: this rung has not earned the transfer.
    None of them is an endpoint FAILURE in the ladder's sense, and none
    is recorded as one; a probe that says no simply leaves the ladder
    in retention order, with every rung still behind it.

    Whether a cycle is PUBLISHED is a different question, and there a
    probe that was not answered is not a no: that question is
    :func:`object_answer` (and :func:`settled_object_answer`, which
    asks again before giving up).
    """

    from urllib.request import Request
    from woof.nomads_governor import paced_urlopen

    request = Request(url, method="HEAD",
                      headers={"User-Agent": PROBE_USER_AGENT})
    try:
        with paced_urlopen(
                request, timeout=timeout,
                **({"opener": opener} if opener is not None else {})
        ) as response:
            return 200 <= int(response.status) < 300
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except BaseException:                     # noqa: BLE001 - see docstring
        return False


#: A probe's answer for a status the host's ``missing_path`` row names
#: (:func:`missing_path_statuses`): the object is not there, or this
#: client was refused, and only another host can say which.  Returned
#: only to a caller that asks for it (``missing_path=True``): every other
#: reader of :func:`object_answer` takes a truthy answer as "present".
ABSENT_OR_REFUSED = "absent_or_refused"


@functools.lru_cache(maxsize=None)
def missing_path_statuses(host: str) -> frozenset[int]:
    """The statuses ``host`` answers for a path it has not created yet.

    Its ``host_policy`` ``missing_path`` row, or none.  404 and 410 are
    always "not there" and are never listed.  The breakage the row
    prevents: NOMADS answers 403 for every lead of a cycle whose
    directory does not exist yet, which read as a host not heard, so an
    as-posted fetch tried a transfer of each such lead every round and
    a lead that never posted was reported as a host not heard.
    """

    raw = _policy().get(str(host).lower(), {}).get("missing_path")
    if not raw:
        return frozenset()
    statuses = frozenset(int(code) for code in raw["statuses"])
    # Named breakage: a 2xx or 5xx here would count a served object or a
    # failing host as absent, and a row without its evidence could not be
    # re-checked when the host changes its answer.
    if (not statuses or any(not 400 <= code < 500 or code in (404, 410)
                            for code in statuses)
            or not str(raw.get("why", "")).strip()):
        raise ValueError(
            f"{TABLE_NAME}: host_policy {host} missing_path names 4xx "
            "statuses other than 404 and 410, and says why")
    return statuses


#: The bytes a GET asks for when it confirms a refused HEAD: the first
#: one, and nothing past the status is read even from a host that
#: ignores the range and answers the whole object.
CONFIRM_RANGE = "bytes=0-0"
#: Statuses that ask this client to slow down.  A HEAD answered with one
#: is not asked again at once by GET, which would ignore the host.
_SLOW_DOWN_STATUSES = frozenset({429, 503})


def _confirms_by_get(host: str, status: int) -> bool:
    """Whether a HEAD that ``host`` answered with ``status`` is asked again by GET.

    The breakage this prevents: a mirror or cache that refuses HEAD (405
    or 501, or any status that says nothing about the object) but serves
    GET read as "not posted", so a native HRRR as-posted fetch waited out
    its whole posting budget and exited 75 on files a GET serves.  Not
    asked again: 404 and 410 (the host said the object is missing), a
    status the host's ``missing_path`` row already reads, and a status
    asking this client to slow down (its ``throttle`` row, 429, 503).
    """

    if status in (404, 410) or status in _SLOW_DOWN_STATUSES:
        return False
    if status in missing_path_statuses(host):
        return False
    policy = throttle_policy(host)
    return policy is None or status not in policy.statuses


def object_answer(url: str, *, timeout: float, max_wait_s: float | None = None,
                  opener=None, missing_path: bool = False) -> bool | str | None:
    """:func:`object_available`'s question with a third answer: no answer.

    True for a 2xx and False for a 404 or 410 -- the host said the
    object is not there.  None for everything that is not an answer
    about the object: a timeout, a refused connection, a throttling
    status, or a NOMADS turn further away than ``max_wait_s`` (nothing
    is sent then; see :func:`woof.nomads_governor.pace`).  A caller
    that must not wait long, such as a page listing which sources hold
    a start, can then say "not checked" instead of "not published".
    With ``missing_path``, a status the host's ``missing_path`` row
    names answers :data:`ABSENT_OR_REFUSED` instead of None.

    A HEAD the host refused with any other status (:func:`_confirms_by_get`)
    is asked once more as a GET of the first byte (:data:`CONFIRM_RANGE`),
    and that answer is read the same way.  A timeout or refused
    connection is not: it stays None, and the caller's transfer, whose
    own checks decide, is let try the object (GS-05).
    """

    from urllib.error import HTTPError
    from urllib.request import Request
    from woof.nomads_governor import PaceBudgetExceeded, paced_urlopen

    host = urlsplit(url).netloc
    asks = [Request(url, method="HEAD",
                    headers={"User-Agent": PROBE_USER_AGENT})]
    while asks:
        request = asks.pop()
        try:
            with paced_urlopen(
                    request, timeout=timeout, max_wait_s=max_wait_s,
                    **({"opener": opener} if opener is not None else {})
            ) as response:
                return 200 <= int(response.status) < 300
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except PaceBudgetExceeded:
            return None
        except HTTPError as error:
            status = int(error.code)
            if status in (404, 410):
                return False
            if missing_path and status in missing_path_statuses(host):
                return ABSENT_OR_REFUSED
            if request.get_method() == "HEAD" and _confirms_by_get(host, status):
                asks.append(Request(url, headers={
                    "User-Agent": PROBE_USER_AGENT, "Range": CONFIRM_RANGE}))
                continue
            return None
        except BaseException:                 # noqa: BLE001 - see docstring
            return None
    return None


#: Pauses before each further ask of an object a publication check had
#: no answer about.  A connect timeout on a busy link is usually gone a
#: few seconds later: three of about 170 HEADs timing out once was
#: enough to refuse a published start as "not published yet".
SETTLE_BACKOFF_S = (2.0, 8.0)
#: How long one of those further asks waits for its NOMADS turn.  A
#: cooldown the host asked for is far longer, and then the object stays
#: unanswered rather than holding the fetch for the whole cooldown.
SETTLE_PACE_BUDGET_S = 30.0


def settled_object_answer(url: str, *, timeout: float = 60.0,
                          backoff_s: Sequence[float] | None = None,
                          opener=None, sleep=None,
                          missing_path: bool = False) -> bool | str | None:
    """:func:`object_answer`, asked again after each pause in ``backoff_s`` while it has no answer.

    True or False as soon as the host answers either way (or
    :data:`ABSENT_OR_REFUSED` with ``missing_path``); None only when
    every ask went unanswered.  The first ask waits for its NOMADS turn
    as long as the governor says, as every fetch probe does; the ones
    after it wait at most :data:`SETTLE_PACE_BUDGET_S`.  ``backoff_s``
    defaults to :data:`SETTLE_BACKOFF_S`, read when called.
    """

    import time

    pause = time.sleep if sleep is None else sleep
    answer = object_answer(url, timeout=timeout, opener=opener,
                           missing_path=missing_path)
    for seconds in (SETTLE_BACKOFF_S if backoff_s is None else backoff_s):
        if answer is not None:
            break
        pause(seconds)
        answer = object_answer(url, timeout=timeout, max_wait_s=SETTLE_PACE_BUDGET_S, opener=opener,
                               missing_path=missing_path)
    return answer


# --------------------------------------------------------------------------
# Host politeness, from the table
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _policy() -> Mapping[str, Mapping[str, object]]:
    return MappingProxyType({
        str(host).lower(): MappingProxyType(dict(raw))
        for host, raw in dict(document().get("host_policy", {})).items()})


@functools.lru_cache(maxsize=1)
def host_caps() -> Mapping[str, int]:
    """``netloc -> in-flight request cap`` for every host that declares one.

    A host absent from the table has no cap and keeps whatever pool the
    caller asked for.
    """

    return MappingProxyType({
        host: int(raw["concurrent"])
        for host, raw in _policy().items()
        if raw.get("concurrent") is not None})


def host_cap_why(host: str) -> str:
    """Why ``host`` is capped where it is -- the table's own sentence."""

    return str(_policy().get(host.lower(), {}).get("why", ""))


@dataclass(frozen=True)
class Throttle:
    """How long one host is waited out when it answers "slow down": a table row.

    ``statuses`` are the HTTP answers that can mean it, and ``codes`` the
    error codes the answer body must name for it to count (S3 names
    ``SlowDown`` in an XML ``<Code>``); empty ``codes`` counts every
    answer with one of those statuses.  Each further ask of a file waits
    a doubling interval from ``first_wait_s`` up to ``max_wait_s``, the
    upper half of it drawn at random so parallel files do not come back
    in step, until ``budget_s`` seconds have been waited for that file.
    """

    host: str
    statuses: tuple[int, ...]
    codes: tuple[str, ...]
    first_wait_s: float
    max_wait_s: float
    budget_s: float
    why: str

    def wait(self, answers: int, jitter: float) -> float:
        """Seconds before the next ask, after ``answers`` throttle answers in a row."""

        ceiling = min(self.max_wait_s,
                      self.first_wait_s * 2.0 ** max(0, int(answers) - 1))
        return ceiling / 2.0 + (ceiling / 2.0) * min(1.0, max(0.0, float(jitter)))


@functools.lru_cache(maxsize=None)
def throttle_policy(host: str) -> Throttle | None:
    """The throttle row ``host`` declares in ``host_policy``, or None."""

    raw = _policy().get(str(host).lower(), {}).get("throttle")
    if not raw:
        return None
    policy = Throttle(
        host=str(host).lower(),
        statuses=tuple(int(code) for code in raw["statuses"]),
        codes=tuple(str(code) for code in raw.get("codes", ())),
        first_wait_s=float(raw["first_wait_s"]),
        max_wait_s=float(raw["max_wait_s"]),
        budget_s=float(raw["budget_s"]),
        why=str(raw.get("why", "")))
    # Named breakage: a zero or inverted row would either never wait or
    # wait longer per ask than the whole budget, so the file would be
    # refused on its first throttle answer exactly as before the row.
    if not (0 < policy.first_wait_s <= policy.max_wait_s <= policy.budget_s):
        raise ValueError(
            f"{TABLE_NAME}: host_policy {host} throttle needs 0 < first_wait_s "
            "<= max_wait_s <= budget_s")
    return policy


_ERROR_CODE = re.compile(rb"<Code>\s*([A-Za-z0-9_.-]+)\s*</Code>")


def answer_code(error: BaseException) -> str | None:
    """The error code an HTTP answer's body names (S3's ``<Code>``), or None.

    Read once and kept on the error, because a body can only be read
    once and both the wait and the log line ask for it.
    """

    if not isinstance(error, HTTPError):
        return None
    if hasattr(error, "_gpuwm_answer_code"):
        return error._gpuwm_answer_code
    code = None
    try:
        body = error.read(4096) if error.fp is not None else b""
        match = _ERROR_CODE.search(body or b"")
        code = match.group(1).decode("ascii") if match else None
    except Exception:                          # noqa: BLE001 - a body is optional
        code = None
    try:
        error._gpuwm_answer_code = code
    except AttributeError:                     # pragma: no cover
        pass
    return code


_BACKBONE_STATUS = re.compile(r"\bhttp status:?\s*(\d{3})\b", re.IGNORECASE)


def answer_status(error: BaseException) -> int | None:
    """The HTTP status a failed transfer answered with, or None.

    Python's own transfers raise :class:`HTTPError`; the Rust backbone's
    raise a transient error whose reason ends ``http status: NNN`` and
    carries no answer body.
    """

    if isinstance(error, HTTPError):
        return int(error.code)
    if getattr(error, "transient", False) is True:
        match = _BACKBONE_STATUS.search(str(getattr(error, "reason", error)))
        if match is not None:
            return int(match.group(1))
    return None


def throttled(endpoint: Endpoint, error: BaseException) -> Throttle | None:
    """The throttle row this answer from ``endpoint`` falls under, or None.

    The Rust backbone's answer has no body, so a row's ``codes`` cannot be
    read from it and its status alone counts.  The breakage this prevents
    (WOOF 2026-10-29, run_19793057ff2979e3): every IFS 0.1-degree file
    moves through the backbone, so ECMWF's AWS mirror's 503 SlowDown was
    asked 5 times in 30 s and refused instead of being waited out on the
    host's own row.
    """

    status = answer_status(error)
    if status is None:
        return None
    policy = throttle_policy(endpoint.host)
    if policy is None or status not in policy.statuses:
        return None
    if (isinstance(error, HTTPError) and policy.codes
            and answer_code(error) not in policy.codes):
        return None
    return policy


def host_worker_cap(host: str, workers: int) -> int:
    """How many of ``workers`` may target ``host`` at once."""

    if aws_fetch_policy() and host.lower() == "nomads.ncep.noaa.gov":
        return min(1, workers)
    cap = host_caps().get(host.lower())
    if cap is None:
        return workers
    return min(cap, workers)


# --------------------------------------------------------------------------
# Faults: which ones mean "ask the next endpoint"
# --------------------------------------------------------------------------

def fault_reason(error: BaseException) -> str | None:
    """One line naming why this endpoint did not serve, or ``None``.

    ``None`` means the failure is NOT an endpoint's fault and the next
    one would fail identically -- an interrupt, a programming error, a
    full disk.  Those propagate unchanged; walking a ladder over them
    would only multiply the damage and bury the real refusal.

    A ``Retry-After`` the host sent is carried into the sentence.  The
    node-wide governor has already extended its cooldown by it (see
    :func:`woof.nomads_governor.mark_rate_limited`), so by the time
    this is read the host's own retry discipline is spent for this
    request -- what is left to decide is whether to wait fifteen
    minutes or to ask the archive, and the archive has the same bytes.
    """

    if isinstance(error, (KeyboardInterrupt, SystemExit, MemoryError)):
        return None
    if isinstance(error, HTTPError):
        detail = FALLTHROUGH_STATUSES.get(error.code)
        if detail is None:
            return None
        from woof.nomads_governor import retry_after_seconds
        wait = retry_after_seconds(error)
        asked = (f", and asked for Retry-After {wait:g} s" if wait else "")
        return f"HTTP {error.code} -- {detail}{asked}"
    if isinstance(error, URLError):
        return f"the connection failed -- {error.reason}"
    if isinstance(error, TimeoutError):
        return "the connection timed out"
    if isinstance(error, OSError):
        return f"the connection failed -- {error}"
    if getattr(error, "transient", False) is True:
        # The Rust backbone's transfer the network cut off after its own
        # retries: the next endpoint serves the same key.
        return f"the connection failed -- {getattr(error, 'reason', error)}"
    if isinstance(error, ValueError):
        # A payload that does not verify: an error page served with 200,
        # a truncated object, a declared length the host did not
        # deliver.  The other endpoint publishes the same key, so this
        # is a reason to ask it rather than to end the fetch.
        return f"the object did not verify -- {error}"
    return None


#: Rounds of asking for one file before a transient network fault ends
#: the fetch: the first try and four more, 2, 4, 8 and 16 s apart (30 s
#: in all), or longer where the host's own ``Retry-After`` asks for it.
#:
#: The concrete breakage it prevents: an IFS fetch pinned to the AWS
#: mirror met HTTP 503 on 3 of its 5 files, gave up after three tries 2
#: and 4 s apart, and the same command run 20 s later completed.  A
#: throttling or restarting host answers like that for longer than 6 s;
#: 30 s outlasts that episode and still ends a fetch from a host that
#: stays down in well under a minute.  Every source's fetch reads this
#: one number (:func:`ask_along_ladder`).
TRANSIENT_ATTEMPTS = 5

#: The longest ``Retry-After`` a host may ask for and still be waited
#: out inside one fetch.  A longer ask ends that endpoint's turn at once,
#: because the other endpoint, or a fetch started later, serves sooner.
TRANSIENT_WAIT_LIMIT_S = 30.0


def retry_delay(error: BaseException, attempt: int, *,
                wait_limit_s: float) -> float | None:
    """Seconds to wait before asking again after a transient transfer fault.

    ``None`` means the fault is not worth another attempt: an HTTP
    status that will not change (a missing object, a refused client), a
    host name that does not resolve, a ``Retry-After`` longer than
    ``wait_limit_s``, or anything that is not the network at all.  A
    reset or aborted connection, a timeout, a response that ended early
    and a 408/429/5xx answer wait ``2 ** attempt`` seconds, or the
    host's own ``Retry-After`` when that is longer.  Every download
    route that retries reads this one classification, so a fault one of
    them waits out is never a refusal on another.
    """

    if isinstance(error, HTTPError):
        if error.code not in {408, 429, 500, 502, 503, 504}:
            return None
        from woof.nomads_governor import retry_after_seconds
        requested = retry_after_seconds(error) or 0.0
        if requested > wait_limit_s:
            return None
        return max(2.0 ** attempt, requested)
    if isinstance(error, URLError):
        reason = error.reason
        if (isinstance(reason, socket.gaierror)
                and reason.errno != socket.EAI_AGAIN):
            return None
        return 2.0 ** attempt
    if isinstance(error, (TimeoutError, ConnectionError, IncompleteRead)):
        return 2.0 ** attempt
    if isinstance(error, OSError) and error.errno in {
            errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNABORTED,
            errno.ECONNREFUSED, errno.ENETUNREACH, errno.EHOSTUNREACH}:
        return 2.0 ** attempt
    if getattr(error, "transient", False) is True:
        # The Rust backbone's transfer the network cut off once its own
        # sub-second retries were spent (``RwFetchError.transient``).
        return 2.0 ** attempt
    return None


def ladder_refusal(label: str, attempts: Sequence[tuple[Endpoint, str]],
                   *, rounds: int = 1, waited_s: float = 0.0) -> str:
    """The refusal when every endpoint failed: each one, and why.

    The concrete breakage this prevents: a two-host fetch that failed
    on both used to report only the last host's error, so a reader saw
    "403 from the archive" and never learned the operational server had
    been rate limiting them for fifteen minutes.

    An endpoint asked more than once is one line with its last reason
    and how many times it was asked, and a request that retried says
    how many rounds it took and how long it waited between them.
    """

    order: list[Endpoint] = []
    reasons: dict[Endpoint, list[str]] = {}
    for endpoint, reason in attempts:
        if endpoint not in reasons:
            order.append(endpoint)
            reasons[endpoint] = []
        reasons[endpoint].append(reason)
    retried = (f" in {rounds} rounds over {waited_s:g} s" if rounds > 1
               else "")
    lines = [f"{label}: every endpoint refused{retried}.  Tried, in order:"]
    for endpoint in order:
        asked = reasons[endpoint]
        times = f" (asked {len(asked)} times)" if len(asked) > 1 else ""
        lines.append(
            f"  {endpoint.name} ({endpoint.host}): {asked[-1]}{times}")
        if endpoint.why:
            lines.append(f"    what it is for: {endpoint.why}")
    lines.append(
        "  remedy: name one host with --transport to see its own refusal "
        "in full, or pass a cycle inside an endpoint's retention window.")
    return "\n".join(lines)


class TransferRefusal(ValueError):
    """One file no endpoint served once every transient retry was spent.

    ``str()`` is :func:`ladder_refusal`'s text: the file, each endpoint
    with its last reason, how many rounds were asked and how long was
    waited.  ``name`` is the file, ``attempts`` every
    ``(endpoint, reason)`` in the order they happened.
    """

    def __init__(self, message: str, *, name: str,
                 attempts: Sequence[tuple[Endpoint, str]], rounds: int,
                 waited_s: float) -> None:
        super().__init__(message)
        self.name = name
        self.attempts = tuple(attempts)
        self.rounds = rounds
        self.waited_s = waited_s


def transfer_reason(error: BaseException) -> str | None:
    """:func:`fault_reason`, with the two faults a transfer names itself."""

    if isinstance(error, IncompleteRead):
        return "the response ended early"
    if isinstance(error, HTTPError) and error.code == 408:
        return "HTTP 408 -- the request timed out"
    return fault_reason(error)


def ask_along_ladder(ladder: Sequence[Endpoint], transfer, *, label: str,
                     name: str, progress, delay=None, discard=None,
                     attempts: int = TRANSIENT_ATTEMPTS, pause=None,
                     tail: str = "", jitter=None) -> tuple[Endpoint, object]:
    """Move one file, asking each endpoint in turn and in rounds.

    ``transfer(endpoint)`` moves the file from one endpoint and returns
    what the caller records; the result comes back with the endpoint
    that served it.  The endpoints publish the same key with the same
    bytes, so a host that refuses, throttles, or serves something that
    does not verify is a reason to ask the next one, not a reason to
    end the fetch.

    This is the one retry every source's fetch runs through: the table
    routes' transfers and the Rust backbone's GFS and HRRR transfers.
    After a round in which every endpoint failed, the ones whose fault
    ``delay(error, attempt)`` calls transient (default
    :func:`retry_delay`: a reset or dropped connection, a timeout, a
    body cut short, HTTP 408, 429 or 5xx) are asked again after the
    longest of their waits, for at most ``attempts`` rounds
    (:data:`TRANSIENT_ATTEMPTS`).  A permanent refusal falls through
    once and is not asked again.  Faults that are not an endpoint's (an
    interrupt, a full disk) propagate unchanged, because the next
    endpoint would fail identically and walking the ladder over them
    would bury the real refusal.  ``discard(endpoint)`` removes what a
    failed attempt staged.  ``pause(seconds)`` waits between rounds and
    by default stops early once another file has failed the request.

    A host that declares a :class:`Throttle` row and answers with it
    ("slow down", S3's ``503 SlowDown``) is not held to the round count:
    it is asked again after the row's doubling, jittered wait until the
    row's time budget is spent, and the log says which host is throttling
    and how much of the budget is gone.  ECMWF's AWS mirror answers about
    half of all requests that way, and 5 rounds in 30 s refused files a
    plain GET then served.  ``jitter()`` draws the random part of each
    wait (default :func:`random.random`).

    When every round is spent, :class:`TransferRefusal` names the file,
    each endpoint and why, the rounds and the seconds waited, followed
    by ``tail``.
    """

    if delay is None:
        def delay(error, attempt):
            return retry_delay(error, attempt,
                               wait_limit_s=TRANSIENT_WAIT_LIMIT_S)
    if pause is None:
        def pause(seconds):
            import time
            from woof import fetch_pool
            fetch_pool.sleep_unless_stopped(seconds, sleep=time.sleep)
    active = tuple(ladder)
    if not active:
        raise ValueError(f"{label}: {name} has no endpoint to ask")
    if jitter is None:
        import random
        jitter = random.random
    attempts = max(1, int(attempts))
    tried: list[tuple[Endpoint, str]] = []
    last_error: BaseException | None = None
    waited = 0.0
    rounds = 0
    throttle_answers: dict[Endpoint, int] = {}
    spent: dict[Endpoint, Throttle] = {}
    attempt = 0
    while True:
        attempt += 1
        rounds = attempt
        again: list[tuple[Endpoint, float, Throttle | None]] = []
        for position, endpoint in enumerate(active):
            try:
                return endpoint, transfer(endpoint)
            except BaseException as error:        # noqa: BLE001 - classified
                seconds = delay(error, attempt)
                policy = throttled(endpoint, error)
                if policy is not None:
                    throttle_answers[endpoint] = throttle_answers.get(endpoint, 0) + 1
                    seconds = policy.wait(throttle_answers[endpoint], jitter())
                # A failure on this computer (a full disk, an unwritable
                # folder) cannot be repaired by another endpoint.
                if (isinstance(error, OSError)
                        and not isinstance(error, (URLError, TimeoutError,
                                                   ConnectionError))
                        and seconds is None):
                    raise
                reason = transfer_reason(error)
                if reason is None:
                    raise
                if policy is not None:
                    code = answer_code(error)
                    reason = (f"HTTP {answer_status(error)}{f' {code}' if code else ''} -- "
                              "the host is throttling this client")
                last_error = error
                tried.append((endpoint, reason))
                if seconds is not None:
                    again.append((endpoint, seconds, policy))
                if discard is not None:
                    discard(endpoint)
                remaining = active[position + 1:]
                progress(
                    f"{label}: {endpoint.name} did not serve {name} "
                    f"({reason})"
                    + (f"; asking {remaining[0].name}" if remaining else ""))
        # A throttling host is held to its row's time budget, every other
        # transient fault to the round count.
        kept: list[tuple[Endpoint, float, Throttle | None]] = []
        for endpoint, seconds, policy in again:
            if policy is None:
                if attempt < attempts:
                    kept.append((endpoint, seconds, None))
                continue
            left = policy.budget_s - waited
            if left >= 1.0:
                kept.append((endpoint, min(seconds, left), policy))
            else:
                spent[endpoint] = policy
        if not kept:
            break
        wait = max(seconds for _endpoint, seconds, _policy in kept)
        throttling = [(endpoint, policy) for endpoint, _seconds, policy in kept
                      if policy is not None]
        if throttling:
            endpoint, policy = throttling[0]
            progress(
                f"{label}: {endpoint.name} ({endpoint.host}) is throttling "
                f"this client; waiting {wait:.1f} s before asking for {name} "
                f"again (ask {attempt + 1}; {waited:.0f} s of the "
                f"{policy.budget_s:g} s the route table allows this host "
                "spent); completed files are kept")
        else:
            progress(f"{label}: retrying {name} in {wait:g} s (attempt "
                     f"{attempt + 1}/{attempts}); completed files are kept")
        pause(wait)
        waited += wait
        active = tuple(endpoint for endpoint, _seconds, _policy in kept)
    budget = "".join(
        f"\n  {endpoint.name} ({endpoint.host}) was still throttling when "
        f"the {policy.budget_s:g} s the route table allows it were spent."
        for endpoint, policy in spent.items())
    raise TransferRefusal(
        ladder_refusal(f"{label}: {name}", tried, rounds=rounds,
                       waited_s=waited) + budget + tail,
        name=name, attempts=tried, rounds=rounds,
        waited_s=waited) from last_error


__all__ = [
    "FETCH_POLICY_ENV", "FETCH_POLICY_CONTRACT", "aws_fetch_policy", "aws_transport_for_source",
    "policy_transport", "policy_uses_aws", "policy_posting_delay_minutes",
    "ABSENT_OR_REFUSED", "missing_path_statuses",
    "Endpoint", "FALLTHROUGH_STATUSES", "PROBE_USER_AGENT", "TABLE_NAME",
    "cycle_age_hours", "document", "endpoint_named", "fault_reason",
    "has_ladder", "host_cap_why", "host_caps", "host_worker_cap", "ladder", "object_answer",
    "ladder_refusal", "object_available", "promote", "retry_delay",
    "serving_ladder",
    "SETTLE_BACKOFF_S", "SETTLE_PACE_BUDGET_S", "settled_object_answer",
    "TRANSIENT_ATTEMPTS", "TRANSIENT_WAIT_LIMIT_S", "Throttle", "TransferRefusal",
    "answer_code", "answer_status", "throttle_policy", "throttled",
    "ask_along_ladder", "transfer_ladder", "transfer_order",
    "transfer_probes", "transfer_reason",
]
