"""Whether a source's window can start, and which cycle ``latest`` is, as it posts.

A run that starts on its source's first hours and waits for the rest
(DESIGN A136, the as-posted chain) asks three questions of a host, and
every door asks them here so no two answers can disagree:

* :func:`lead_answer`: does one host hold every object of one lead?
  ``posted``, ``not_posted`` or ``not_heard``.  A host that could not be
  heard is never counted as not holding the lead.
* :func:`start_needs` and :func:`readiness`: which leads must be posted
  before the run can start (the analysis, the first boundary interval,
  the route's invariant and step-0 objects, a same-cycle donor, a whole
  cycle's final lead), and the ``gpuwm.readiness.v1`` answer a site
  launches on.
* :func:`resolve_startable_cycle`: ``latest`` under ``as_posted``, the
  newest cycle whose start needs are posted and whose ladder covers the
  window.

Nothing here names a source.  Which objects a lead is made of is the
route table's plan (or the legacy transport's own URL builder, the same
ones the completeness probe asks), when a lead is due is the source's
``publication_lag`` rows, and how a run waits is its ``posting`` block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Mapping, Sequence

from woof import fetch_endpoints
from woof import source_posting as rows

READINESS_SCHEMA = "gpuwm.readiness.v1"

#: ``--readiness`` exit codes (DESIGN A136 3.4): run-plan's query-mode
#: convention (0 printed, 2 refused) with 75 for "not yet".
READY_EXIT = 0
NOT_YET_EXIT = 75
REFUSED_EXIT = 2

#: A lead is not asked about before its scheduled time minus this: an
#: earlier ask can only answer no, and NOMADS rate-limits clients that
#: poll it for nothing (DESIGN A136 2.2, R2).
ASK_AHEAD = timedelta(minutes=5)

POSTED = "posted"
NOT_POSTED = "not_posted"
NOT_HEARD = "not_heard"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _naive(moment: datetime) -> datetime:
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment


def parse_instant(text: str | None) -> datetime | None:
    if not text:
        return None
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


class ReadinessRefusal(ValueError):
    """A window that can never start: exit 2 with the sentence."""


@dataclass(frozen=True)
class Window:
    """One fetch window, resolved: the leads it asks for and how.

    ``plan`` is the route table's plan for a table route and None for a
    legacy transport, whose objects come from its own URL builder.
    """

    source: str
    cycle: datetime
    leads: tuple[int, ...]
    cadence: int | None
    member: str | None = None
    transport: str | None = None
    provider: str | None = None
    plan: object | None = None
    hours: int = 0

    @property
    def start_lead(self) -> int:
        return self.leads[0]

    @property
    def final_lead(self) -> int:
        return self.leads[-1]

    @property
    def spacing(self) -> int | None:
        if len(self.leads) > 1:
            return self.leads[1] - self.leads[0]
        return self.cadence


def resolve_window(source: str, cycle: datetime, hours: int, *,
                   cadence: int | None = None, start_hour: int | None = None,
                   source_parameters: str | None = None,
                   member: str | None = None, transport: str | None = None,
                   provider: str | None = None,
                   now: datetime | None = None) -> Window:
    """The window's leads, refused only by what can never start.

    The ladder decides the leads through the same function the fetch
    plans with, so a lead beyond the ladder is refused here in the
    ladder's words (exit 2) and never waited for.
    """

    from woof import fetch
    from woof import fetch_routes

    start = 0 if start_hour is None else int(start_hour)
    cycle = _naive(cycle)
    if source not in fetch.fetch_front_door_sources():
        # Breakage it prevents: an unknown name would read as a source
        # with nothing to probe and be answered "ready".
        raise ReadinessRefusal(
            f"--source {source!r} is not a source `woof fetch` acquires; "
            f"it acquires {', '.join(fetch.fetch_front_door_sources())}")
    try:
        if source in fetch_routes.route_ids():
            plan = fetch_routes.resolve_request(
                source, cycle=cycle, hours=hours, cadence=cadence,
                start_hour=start, host=transport, member=member, source_parameters=source_parameters, now=now)
            return Window(source=source, cycle=cycle, leads=tuple(plan.leads),
                          cadence=cadence, member=plan.member,
                          transport=transport, provider=provider, plan=plan,
                          hours=hours)
        if source in fetch.GFS_CONTAINER_SOURCES:
            leads = fetch.container_forecast_hours(source, hours, cadence, start)
        elif source == "hrrr":
            leads = fetch.hrrr_forecast_hours(hours, cycle, start)
        else:
            leads = tuple(range(start, start + hours + 1,
                                max(1, cadence or 1)))
    except (KeyError, ValueError) as error:
        raise ReadinessRefusal(str(error)) from None
    return Window(source=source, cycle=cycle, leads=tuple(leads),
                  cadence=cadence, member=member, transport=transport,
                  provider=provider, hours=hours)


# --------------------------------------------------------------------------
# The objects of one lead, on one rung
# --------------------------------------------------------------------------

def _route_groups(plan) -> dict[int, list]:
    """Each planned object under the lead its group is fetched with.

    A lead-free object (an invariant) and a step-0 object planned beside
    a window that starts later travel with the window's first lead, as
    the plan itself orders them.
    """

    first = plan.leads[0]
    groups: dict[int, list] = {lead: [] for lead in plan.leads}
    for obj in plan.objects:
        lead = obj.lead if obj.lead in groups else first
        groups[lead].append(obj)
    return groups


def _idx_suffix(plan, role: str) -> str | None:
    for row in plan.files:
        if row.role == role:
            return row.idx_sidecar
    return None


def _with_index(window: Window) -> bool:
    return rows.posting(window.source, window.provider).ready_check == \
        "objects_and_index"


def lead_urls(window: Window, lead: int, endpoint: str, *,
              objects: Sequence | None = None) -> tuple[str, ...]:
    """Every URL whose answer says ``lead`` is posted on ``endpoint``.

    A table route asks the plan's objects of that lead group (and each
    one's index when the posting row's check is ``objects_and_index``);
    a legacy transport asks the objects its own completeness probe
    asks, plus their ``.idx`` under the same check.
    """

    from woof import fetch

    with_index = _with_index(window)
    if window.plan is not None:
        plan = window.plan
        rung = next((entry for entry in (plan.ladder or ())
                     if entry.name == endpoint), None)
        if rung is None:
            rung = fetch_endpoints.endpoint_named(window.source, endpoint)
        chosen = objects if objects is not None else _route_groups(plan)[lead]
        urls: list[str] = []
        for obj in chosen:
            urls.append(rung.url(obj.key) if obj.key else obj.url)
            suffix = _idx_suffix(plan, obj.role) if with_index else None
            if suffix and obj.key:
                index_key = getattr(obj, "index_key", None) or f"{obj.key}{suffix}"
                urls.append(rung.url(index_key))
        return tuple(urls)
    urls = list(fetch.cycle_probe_urls(window.source, window.cycle, lead,
                                       transport=endpoint))
    if with_index and fetch_endpoints.endpoint_named(window.source, endpoint).idx_available:
        urls += [url + ".idx" for url in list(urls)]
    return tuple(urls)


def _ladder(window: Window, *, now: datetime | None = None,
            source: str | None = None, cycle: datetime | None = None):
    return fetch_endpoints.serving_ladder(
        source or window.source, cycle=cycle or window.cycle, now=now,
        pinned=window.transport if source in (None, window.source) else None)


def _first_object_answer(url: str):
    """The first object's question: a missing-path answer counts.

    A host whose ``missing_path`` row names a status (NOMADS's 403 for a
    path under a cycle directory it has not created yet) answers
    :data:`woof.fetch_endpoints.ABSENT_OR_REFUSED` for it; every later
    object of the lead is asked the plain question, since by then the
    directory exists.
    """

    return fetch_endpoints.settled_object_answer(url, missing_path=True)


def _first_probe_for(probe):
    """The first object's probe beside ``probe``, when that is the engine's.

    Only the engine's own HEAD question (``woof.fetch._head_answer``)
    has a missing-path twin; a probe a caller hands in (a replay, a test)
    answers every object itself.
    """

    if (getattr(probe, "__module__", None) == "woof.fetch"
            and getattr(probe, "__name__", None) == "_head_answer"):
        return _first_object_answer
    return None


def _ask(urls: Sequence[str], probe, first_probe=None
         ) -> tuple[bool | str | None, str | None]:
    """One rung's answer, asked one object after another.

    Stops at the first object not there, so a lead that has not posted
    costs one HEAD per round (DESIGN A136 2.2); once the first object is
    there the rest are asked side by side under the host caps.  The
    first object may answer ``ABSENT_OR_REFUSED`` (``first_probe``).
    """

    from woof import fetch

    urls = tuple(dict.fromkeys(urls))
    if not urls:
        return False, None
    first = (probe if first_probe is None else first_probe)(urls[0])
    if first is None:
        return None, urls[0]
    if first == fetch_endpoints.ABSENT_OR_REFUSED:
        return first, urls[0]
    if not first:
        return False, urls[0]
    if len(urls) == 1:
        return True, None
    return fetch._rung_answer(urls[1:], probe)


def _ladder_answer(answers) -> dict[str, object]:
    """The lead's answer from each rung's ``(found, url, endpoint)``.

    ``posted`` when one rung holds every object.  ``not_posted`` when
    every rung answered and at least one said an object is not there; a
    rung that answered only ``ABSENT_OR_REFUSED`` (not there, or this
    client refused) counts as not there beside such a rung, and as not
    heard without one, so a blocked client is never reported as the
    publisher being late.  ``not_heard`` otherwise.
    """

    unheard_url = refused_url = last_url = None
    absent = False
    for found, url, endpoint in answers:
        if found is True:
            return {"answer": POSTED, "endpoint": endpoint, "url": None}
        if found is None:
            unheard_url = unheard_url or url
        elif found == fetch_endpoints.ABSENT_OR_REFUSED:
            refused_url = refused_url or url
        else:
            absent = True
        last_url = url
    if unheard_url is not None:
        return {"answer": NOT_HEARD, "endpoint": None, "url": unheard_url}
    if refused_url is not None and not absent:
        return {"answer": NOT_HEARD, "endpoint": None, "url": refused_url}
    return {"answer": NOT_POSTED, "endpoint": None, "url": last_url}


def lead_answer(window: Window, lead: int, *, probe=None,
                now: datetime | None = None,
                objects: Sequence | None = None) -> dict[str, object]:
    """Whether one host holds every object of ``lead``.

    ``{"answer": "posted" | "not_posted" | "not_heard", "endpoint": NAME
    or None, "url": the URL that decided}``.  The rungs are asked one at
    a time in serving order (the operational server, which posts first,
    heads it).  ``not_posted`` only when every rung asked answered; a
    rung that could not be heard makes the answer ``not_heard``, never
    absent.
    """

    from woof import fetch

    probe = fetch._head_answer if probe is None else probe
    first_probe = _first_probe_for(probe)
    answers = []
    for rung in _ladder(window, now=now):
        try:
            urls = lead_urls(window, lead, rung.name, objects=objects)
        except ValueError:
            # A declared date/lead hole is absent, never a positive probe.
            answers.append((False, None, rung.name))
            continue
        found, url = _ask(urls, probe, first_probe)
        answers.append((found, url, rung.name))
        if found is True:
            break
    return _ladder_answer(answers)


def donor_answer(donor_source: str, cycle: datetime, lead: int, *, probe=None,
                 now: datetime | None = None) -> dict[str, object]:
    """:func:`lead_answer` for a same-cycle donor, on the donor's own ladder."""

    from woof import fetch

    probe = fetch._head_answer if probe is None else probe
    first_probe = _first_probe_for(probe)
    answers = []
    for rung in fetch_endpoints.serving_ladder(donor_source, cycle=cycle, now=now):
        urls = fetch.cycle_probe_urls(donor_source, cycle, lead,
                                      transport=rung.name)
        found, url = _ask(urls, probe, first_probe)
        answers.append((found, url, rung.name))
        if found is True:
            break
    answer = _ladder_answer(answers)
    if answer["answer"] == NOT_POSTED:
        answer["url"] = None
    return answer


# --------------------------------------------------------------------------
# Start needs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Need:
    """One thing that must be posted before a run can start."""

    role: str
    source: str
    lead: int | None
    expected_at: datetime | None
    late_at: datetime | None
    #: The route objects that make up this need (None: the lead's whole
    #: group, or a legacy transport's objects for the lead).
    objects: tuple = field(default=(), compare=False)

    def row(self) -> dict[str, object]:
        return {"role": self.role, "source": self.source, "lead": self.lead,
                "expected_at": rows.instant(self.expected_at),
                "late_at": rows.instant(self.late_at)}


def _due(source: str, cycle: datetime, lead: int,
         late_after_minutes: float | None, provider: str | None):
    try:
        expected = rows.expected_at(source, cycle, lead)
    except ValueError:
        return None, None
    late = rows.late_at(source, cycle, lead,
                        late_after_minutes_override=late_after_minutes,
                        provider=provider)
    return expected, late


def start_needs(window: Window, *, late_after_minutes: float | None = None,
                nest_start_leads: Mapping[str, int] | None = None
                ) -> tuple[Need, ...]:
    """What must be posted before ``window`` can start (DESIGN A136 1, 3.3).

    ``analysis`` (the first lead) and ``first_boundary`` (the second);
    ``invariant`` for lead-free route objects and ``supplement:<role>``
    for the cycle's step-0 objects, both fetched with the first lead;
    ``donor:<source>`` for each same-cycle donor; ``nest_start:dNN`` for
    each delayed nest's start lead (``nest_start_leads``, from the
    experiment); and ``whole_cycle``, the final lead, for a source whose
    cycle posts whole.
    """

    source, cycle = window.source, window.cycle
    posting = rows.posting(source, window.provider)
    needs: list[Need] = []

    def add(role: str, lead: int | None, objects=(), *, owner: str = source):
        due_lead = window.start_lead if lead is None else lead
        expected, late = _due(owner, cycle, due_lead, late_after_minutes,
                              window.provider if owner == source else None)
        needs.append(Need(role=role, source=owner, lead=lead,
                          expected_at=expected, late_at=late,
                          objects=tuple(objects)))

    first = window.start_lead
    if window.plan is not None:
        plan = window.plan
        group = _route_groups(plan)[first]
        primary_roles = {row.role for row in plan.files if row.leads == "all"}
        own = [obj for obj in group if obj.lead == first
               and obj.role in primary_roles]
        add("analysis", first, own or group)
        invariants = [obj for obj in group if obj.lead is None]
        if invariants:
            add("invariant", None, invariants)
        step0 = [obj for obj in group
                 if obj.lead is not None and obj.lead != first]
        step0 += [obj for obj in group if obj.lead == first
                  and obj.role not in primary_roles]
        for role in dict.fromkeys(obj.role for obj in step0):
            add(f"supplement:{role}", 0 if first != 0 else first,
                [obj for obj in step0 if obj.role == role])
    else:
        add("analysis", first)
    if len(window.leads) > 1:
        add("first_boundary", window.leads[1])
    if window.plan is not None:
        for donor in window.plan.donors:
            add(f"donor:{donor.source}", max(donor.leads), owner=donor.source)
    for name, lead in sorted((nest_start_leads or {}).items()):
        if int(lead) not in (need.lead for need in needs):
            add(f"nest_start:{name}", int(lead))
    if posting.shape == "whole_cycle" and window.final_lead not in (
            need.lead for need in needs):
        add("whole_cycle", window.final_lead)
    return tuple(needs)


def need_answer(window: Window, need: Need, *, probe=None,
                now: datetime | None = None) -> dict[str, object]:
    if need.source != window.source:
        return donor_answer(need.source, window.cycle, int(need.lead),
                            probe=probe, now=now)
    lead = window.start_lead if need.lead is None else need.lead
    objects = need.objects or None
    return lead_answer(window, lead, probe=probe, now=now, objects=objects)


# --------------------------------------------------------------------------
# The readiness document
# --------------------------------------------------------------------------

def _refusal_document(source: str, cycle: datetime | None, basis: str,
                      hours: int | None, as_posted: bool, why: str,
                      checked: datetime) -> dict[str, object]:
    return {
        "schema": READINESS_SCHEMA, "checked_at": rows.instant(checked),
        "source": source, "member": None,
        "cycle": None if cycle is None else cycle.strftime("%Y-%m-%dT%H"),
        "cycle_basis": basis, "window": {"hours": hours},
        "as_posted": bool(as_posted), "posting": None, "state": "refused",
        "ready": False, "start_needs": [], "expected_ready_at": None,
        "expected_final_at": None, "leads": [], "retry_after_seconds": None,
        "refusal": why,
    }


def retention_refusal(window: Window, *, now: datetime | None) -> str | None:
    """Why ``window``'s cycle has aged off every host it posts on, or None.

    A pinned host (``--transport``) is the whole ladder: past that host's
    retention the cycle can never start from it, whatever other hosts
    keep, and the refusal is the one-shot fetch's own
    (:func:`woof.fetch._pinned_retention_refusal`), which names the host
    that still keeps it.  Without it an old cycle pinned to the
    operational server was waited for and stopped with exit 75 as late,
    where the whole-cycle fetch refuses it with exit 2 and names S3.
    """

    from woof import fetch

    rungs = fetch_endpoints.ladder(window.source)
    if window.transport is not None and any(
            rung.name == window.transport for rung in rungs):
        return fetch._pinned_retention_refusal(
            window.source, window.cycle, window.final_lead, window.transport,
            now=now)
    if not rungs:
        return None
    age = fetch_endpoints.cycle_age_hours(window.cycle, now)
    if any(rung.covers(age) for rung in rungs):
        return None
    hosts = list(dict.fromkeys(rung.host for rung in rungs))
    kept = max(float(rung.retention_hours) for rung in rungs)
    return (f"{window.source.upper()} cycle {window.cycle:%Y-%m-%dT%H}Z is "
            f"{age:.0f} h old and {' and '.join(hosts)} keep only about the "
            f"newest {kept:g} h, with no archive behind them, so it will "
            "not appear by waiting")


def archive_end_refusal(window: Window, *, now: datetime) -> str | None:
    """Why an archive window runs past the archive's declared end, or None."""

    from woof import fetch

    try:
        grid, basis = fetch.provider_cycle_grid(window.source, window.provider,
                                                now=now)
    except ValueError:
        return None
    end = getattr(grid, "record_end", None)
    last = window.cycle + timedelta(hours=window.final_lead)
    if end is not None and last > end:
        return (f"{window.source} {window.cycle:%Y-%m-%dT%H}Z + "
                f"{window.final_lead} h ends at {last:%Y-%m-%dT%H}Z, past "
                f"{basis} ({end:%Y-%m-%dT%H}Z)")
    return None


def readiness(source: str, cycle: datetime | str, hours: int, *,
              source_parameters: str | None = None,
              cadence: int | None = None, start_hour: int | None = None,
              member: str | None = None, transport: str | None = None,
              provider: str | None = None, as_posted: bool = True,
              late_after_minutes: float | None = None,
              nest_start_leads: Mapping[str, int] | None = None,
              no_probe: bool = False, probe=None,
              now: datetime | None = None) -> tuple[dict[str, object], int]:
    """The ``gpuwm.readiness.v1`` document and its exit code (DESIGN A136 3.4).

    ``cycle`` may be ``"latest"``: under ``as_posted`` that is
    :func:`resolve_startable_cycle`, otherwise the whole-cycle rule.
    ``no_probe`` asks no host: every answer is null and the state is the
    schedule's (``ready`` once ``expected_ready_at`` has passed).
    """

    from woof import fetch

    # Generated sources own acquisition, including the inputs they need.
    # They have no posted forecast-file route to ask about.
    from importlib import import_module
    from woof.source_adapters import get_source_adapter

    try:
        adapter = get_source_adapter(source)
    except ValueError:
        adapter = None
    if adapter is not None and adapter.acquisition_owner:
        owner = import_module(adapter.acquisition_owner)
        return owner.readiness(
            source, cycle, hours, cadence=cadence, start_hour=start_hour,
            member=member, transport=transport, provider=provider,
            as_posted=as_posted, late_after_minutes=late_after_minutes,
            nest_start_leads=nest_start_leads, no_probe=no_probe,
            probe=probe, now=now)

    checked = _naive(now) if now is not None else _utc_now()
    transport = fetch_endpoints.policy_transport(source, transport)
    basis = "named"
    try:
        if isinstance(cycle, str):
            if cycle == "latest":
                start = 0 if start_hour is None else int(start_hour)
                if no_probe:
                    resolved = startable_by_schedule(
                        source, start_hour=start, hours=hours,
                        cadence=cadence, source_parameters=source_parameters, now=checked)
                    basis = "latest (the posting schedule; no host asked)"
                elif as_posted:
                    resolved = resolve_startable_cycle(
                        source, start_hour=start, hours=hours, cadence=cadence,
                        member=member, source_parameters=source_parameters, transport=transport, now=checked,
                        probe=probe)
                    basis = "latest (newest cycle whose start needs are posted)"
                else:
                    resolved = fetch.resolve_latest_cycle(
                        source, start + hours, now=checked,
                        **({} if probe is None else {"probe": probe}),
                        transport=transport, cadence=cadence,
                        start_hour=start, member=member, source_parameters=source_parameters, provider=provider)
                    basis = "latest (newest cycle whose final lead is posted)"
                cycle = resolved
            else:
                cycle = fetch.parse_cycle(cycle, source)
        window = resolve_window(source, cycle, hours, cadence=cadence,
                                start_hour=start_hour, member=member, source_parameters=source_parameters,
                                transport=transport, provider=provider,
                                now=checked)
    except (ReadinessRefusal, ValueError, RuntimeError) as error:
        named = cycle if isinstance(cycle, datetime) else None
        return (_refusal_document(source, named, basis, hours, as_posted,
                                  str(error), checked), REFUSED_EXIT)
    posting = rows.posting(source, provider)
    refusal = retention_refusal(window, now=checked) or archive_end_refusal(
        window, now=checked)
    lead_rows = []
    try:
        lead_rows = rows.lead_rows(
            source, window.cycle, window.leads,
            late_after_minutes_override=late_after_minutes, provider=provider)
    except ValueError:
        lead_rows = [{"lead": lead, "valid_time": rows.instant(
            window.cycle + timedelta(hours=lead)), "expected_at": None}
            for lead in window.leads]
    rule = None
    try:
        candidate = rows.publication_rule(source, window.cycle, window.start_lead)
    except ValueError:
        candidate = None
    if candidate is not None:
        rule = {"cycle_hours": (None if candidate.cycle_hours is None
                                else list(candidate.cycle_hours)),
                "hours": candidate.hours,
                "per_lead_hours": candidate.per_lead_hours,
                "measured": candidate.measured}
    budget = rows.late_after_minutes(source, late_after_minutes,
                                     provider=provider)
    from woof import fetch_routes

    document: dict[str, object] = {
        "schema": READINESS_SCHEMA,
        "checked_at": rows.instant(checked),
        "source": source,
        "member": window.member,
        "cycle": window.cycle.strftime("%Y-%m-%dT%H"),
        "cycle_basis": basis,
        "window": {"start_lead": window.start_lead, "hours": hours,
                   "cadence": window.spacing,
                   "final_lead": window.final_lead},
        "as_posted": bool(as_posted),
        "posting": {"shape": posting.shape, "streams": posting.streams,
                    "why": posting.why, "late_after_minutes": budget,
                    "poll_seconds": posting.poll_seconds, "rule": rule,
                    "table_sha256": fetch_routes.packaged_route_table_sha256()},
        "state": None, "ready": False, "start_needs": [],
        "expected_ready_at": None,
        "expected_final_at": lead_rows[-1].get("expected_at") if lead_rows else None,
        "leads": [{"lead": row["lead"], "valid_time": row["valid_time"],
                   "expected_at": row.get("expected_at"), "answer": None}
                  for row in lead_rows],
        "retry_after_seconds": None,
        "refusal": None,
    }
    if refusal is not None:
        document.update(state="refused", refusal=refusal)
        return document, REFUSED_EXIT
    probeable = fetch.cycle_is_probeable(source) and posting.waited
    needs = start_needs(window, late_after_minutes=late_after_minutes,
                        nest_start_leads=nest_start_leads)
    if not as_posted:
        # Every requested boundary matters, including interior archive holes.
        for lead in window.leads:
            if lead not in (need.lead for need in needs):
                expected, late = _due(source, window.cycle, lead,
                                     late_after_minutes, provider)
                needs += (Need(role="whole_cycle", source=source,
                               lead=lead, expected_at=expected, late_at=late),)
    stamps = [need.expected_at for need in needs if need.expected_at is not None]
    ready_at = max(stamps) if stamps else None
    document["expected_ready_at"] = rows.instant(ready_at)
    need_rows = []
    if not probeable:
        for need in needs:
            need_rows.append({**need.row(), "answer": None, "endpoint": None})
        document.update(state="unprobeable", ready=True,
                        start_needs=need_rows)
        return document, READY_EXIT
    answers: dict[int | None, str] = {}
    waiting = False
    for need in needs:
        if no_probe:
            answer = {"answer": None, "endpoint": None}
        else:
            answer = need_answer(window, need, probe=probe, now=checked)
        need_rows.append({**need.row(), "answer": answer["answer"],
                          "endpoint": answer["endpoint"]})
        if need.source == source:
            answers[need.lead if need.lead is not None
                    else window.start_lead] = answer["answer"]
        if answer["answer"] != POSTED:
            waiting = True
    for row in document["leads"]:
        row["answer"] = answers.get(row["lead"])
    document["start_needs"] = need_rows
    poll = posting.poll_seconds or 30.0
    if no_probe:
        ready = ready_at is None or checked >= ready_at
    else:
        ready = not waiting
    if ready:
        document.update(state="ready", ready=True)
        return document, READY_EXIT
    retry = poll
    if (ready_at is not None and ready_at - ASK_AHEAD > checked
            and (no_probe or not any(fetch_endpoints.policy_uses_aws(need.source)
                                     for need in needs))):
        retry = max(poll, (ready_at - ASK_AHEAD - checked).total_seconds())
    document.update(state="waiting", ready=False,
                    retry_after_seconds=round(float(retry), 1))
    return document, NOT_YET_EXIT


# --------------------------------------------------------------------------
# latest, as posted
# --------------------------------------------------------------------------

def _candidates(source: str, *, start_hour: int, hours: int,
                cadence: int | None, now: datetime) -> tuple:
    from woof import fetch
    from woof import fetch_routes

    grid = fetch.require_cycle_grid(source)
    last = start_hour + hours
    # The walk starts at the newest cycle whose first boundary lead can
    # be out: a cycle posts its leads over hours, and this asks about
    # the first ones.
    first_boundary = start_hour + (cadence or 1 if hours else 0)
    route = (fetch_routes.route_for(source)
             if source in fetch_routes.route_ids() else None)
    if route is not None and cadence is None:
        cadence = fetch_routes.window_cadence(route, start_hour, hours)
        first_boundary = start_hour + (cadence or 1 if hours else 0)
    candidates = tuple(
        cycle for cycle in grid.candidates(now, first_boundary)
        if grid.horizon(cycle) is None or last <= grid.horizon(cycle))
    return grid, route, cadence, candidates


def startable_by_schedule(source: str, *, start_hour: int, hours: int,
                          source_parameters: str | None = None,
                          cadence: int | None = None,
                          now: datetime | None = None) -> datetime:
    """``latest`` from the table alone: the newest cycle whose start needs are due."""

    now = _utc_now() if now is None else _naive(now)
    _grid, _route, cadence, candidates = _candidates(
        source, start_hour=start_hour, hours=hours, cadence=cadence, now=now)
    for cycle in candidates:
        try:
            window = resolve_window(source, cycle, hours, cadence=cadence,
                                    start_hour=start_hour, source_parameters=source_parameters, now=now)
        except ReadinessRefusal:
            continue
        needs = start_needs(window)
        if all(need.expected_at is None or need.expected_at <= now
               for need in needs):
            return cycle
    raise RuntimeError(f"no {source} cycle's start needs are due by the "
                       "posting schedule within the search window")


def resolve_startable_cycle(source: str, *, start_hour: int = 0, hours: int,
                            cadence: int | None = None,
                            source_parameters: str | None = None,
                            member: str | None = None,
                            transport: str | None = None,
                            now: datetime | None = None, probe=None,
                            nest_start_leads: Mapping[str, int] | None = None
                            ) -> datetime:
    """``latest`` under ``as_posted``: the newest cycle whose start needs are posted.

    The walk is :func:`woof.fetch.resolve_latest_cycle`'s -- endpoints
    in serving order, cycles newest first -- asking each cycle's start
    needs (:func:`start_needs`) instead of its final lead, and keeping
    only cycles whose ladder covers the window.  A source a probe cannot
    ask resolves as the whole-cycle rule does, from its declared delay.
    """

    from woof import fetch

    now = _utc_now() if now is None else _naive(now)
    if not fetch.cycle_is_probeable(source) or not \
            rows.posting(source).waited:
        return fetch.resolve_latest_cycle(
            source, start_hour + hours, now=now, transport=transport,
            cadence=cadence, start_hour=start_hour, member=member, source_parameters=source_parameters)
    probe = fetch._head_ok if probe is None else probe
    grid, _route, cadence, candidates = _candidates(
        source, start_hour=start_hour, hours=hours, cadence=cadence, now=now)
    if fetch_endpoints.policy_uses_aws(source):
        # Predictions date the deadline; an already posted AWS start must
        # not be excluded because the mirror's predicted lag has not passed.
        newest = grid.snap(now)
        candidates = []
        candidate = newest
        while newest - candidate <= timedelta(hours=grid.search_hours):
            if (grid.horizon(candidate) is None
                    or start_hour + hours <= grid.horizon(candidate)):
                candidates.append(candidate)
            candidate = grid.snap(candidate - timedelta(hours=1))
    windows: list[Window] = []
    errors: list[str] = []
    for cycle in candidates:
        try:
            windows.append(resolve_window(
                source, cycle, hours, cadence=cadence, start_hour=start_hour,
                member=member, source_parameters=source_parameters, transport=transport, now=now))
        except ReadinessRefusal as error:
            errors.append(str(error))
    if not windows:
        raise RuntimeError(errors[-1] if errors else (
            f"no {source} cycle in the last {grid.search_hours} h forecasts "
            f"as far as f{start_hour + hours:03d}"))
    ladder = fetch_endpoints.serving_ladder(
        source, cycle=windows[0].cycle, now=now, pinned=transport)
    donors_seen: dict[tuple, bool] = {}
    for rung in ladder:
        for window in windows:
            needs = start_needs(window, nest_start_leads=nest_start_leads)
            complete = True
            for need in needs:
                if need.source != source:
                    key = (need.source, window.cycle, need.lead)
                    if key not in donors_seen:
                        donors_seen[key] = donor_answer(
                            need.source, window.cycle, int(need.lead),
                            probe=probe, now=now)["answer"] == POSTED
                    if not donors_seen[key]:
                        complete = False
                        break
                    continue
                lead = window.start_lead if need.lead is None else need.lead
                urls = lead_urls(window, lead, rung.name,
                                 objects=need.objects or None)
                found, _url = _ask(urls, probe)
                if found is not True:
                    complete = False
                    break
            if complete:
                return window.cycle
    tried = " or ".join(entry.name for entry in ladder)
    raise RuntimeError(
        f"no {source.upper()} cycle whose start needs (f{start_hour:03d} and "
        f"its first boundary lead) are posted was found on {tried} within "
        f"the last {grid.search_hours} h; pass an explicit --cycle")


def print_document(document: Mapping[str, object], stream=None) -> None:
    import json
    import sys

    (stream or sys.stdout).write(json.dumps(document, indent=2) + "\n")


__all__ = [
    "ASK_AHEAD", "NOT_HEARD", "NOT_POSTED", "NOT_YET_EXIT", "Need", "POSTED",
    "READINESS_SCHEMA", "READY_EXIT", "REFUSED_EXIT", "ReadinessRefusal",
    "Window", "donor_answer", "lead_answer", "lead_urls",
    "need_answer", "parse_instant", "print_document", "readiness",
    "resolve_startable_cycle", "resolve_window", "retention_refusal",
    "start_needs", "startable_by_schedule",
]
