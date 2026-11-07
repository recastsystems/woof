"""When a source has an initialization, and when its bytes land.

``--cycle latest`` asks one question -- what is the newest init this
source can serve -- and three model names used to answer it.
:func:`woof.fetch.resolve_latest_cycle` branched on gfs/gdas/hrrr and
refused everything else with a sentence about ERA5's publication delay,
which a reader asking for RAP or ICON-EU got verbatim.  In a registry of
thirty-two sources that is the per-model bandaid the arbitrary
acceptance test bans, and it made a reanalysis unaskable for a time it
publishes perfectly well.

The question is answered from DECLARED FACTS instead:

``hours``          the UTC hours of day this producer initializes on.
``delay_hours``    how long after a nominal init its bytes are on a
                   server.  Zero where a completeness PROBE decides
                   publication -- the probe IS the answer there, and a
                   declared delay would only start the walk-back late.
``delays``         the same fact per cycle hour and per lead, where the
                   producer's timing differs between its cycles (a fetch
                   route's ``publication_lag`` rows, :class:`PublicationRule`).
                   ``delay_hours`` is then the longest first-object delay,
                   for a reader that asks without naming a cycle.  Each
                   rule is the EARLIEST posting seen, where the probed
                   ``latest`` walk starts, plus how much later the latest
                   posting seen came: a reader that no probe checks
                   (``settled``) waits for that one instead.
``usual_delay_hours``  where the probe decides (``delay_hours`` zero),
                   how long after a nominal init the whole run is
                   usually on the server, measured.  The page's answer
                   while no probe has answered: a start this old is
                   taken as published unless a check found it missing.
                   Unset, it reads as ``delay_hours``.
``search_hours``   how far back ``latest`` walks the grid before it
                   gives up and says so.
``record_end``     the last init of a CLOSED archive, or ``None`` for a
                   producer still running.  An archive that ended still
                   has a newest init, and it is not "now minus a delay".

A source with a fetch route already declares its grid there
(``woof.fetch_routes.Route.cycle_hours``, measured against the
producer), so :func:`cycle_grid_for` READS that rather than asking this
module to repeat it: a model added as a route-table row gets ``latest``
with no edit here at all.  What this module carries is the sources whose
schedule nothing else declares, and the shape every answer is given in.

A row that declares no grid and has no route is not on a list of the
unsupported -- :func:`cycle_grid_for` returns ``None`` and the caller
refuses by naming what the row lacks, which is a sentence that stays
true when the row grows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta


@dataclass(frozen=True)
class PublicationRule:
    """When one set of a producer's leads lands, measured: a table row.

    Lead ``L`` of a cycle whose UTC hour is in ``cycle_hours`` (``None``:
    every hour no earlier rule names) and with ``L >= from_lead`` was
    first seen on the ladder head ``hours + per_lead_hours * L`` after
    the cycle, and never more than ``late_hours`` after that.  The first
    rule that matches answers, so a rule for a later part of the ladder
    (GEFS's 00Z extension past f384, posted about a day later) comes
    before the rule for the rest of it.

    The earliest line is where the probed ``latest`` walk starts: later
    than a real posting makes it skip a cycle that is out, earlier costs
    one probe that answers no.  The latest line (``settled``) is for an
    answer no probe checks, where earlier claims a lead that is not out.
    """

    cycle_hours: tuple[int, ...] | None
    hours: float
    per_lead_hours: float = 0.0
    late_hours: float = 0.0
    from_lead: int = 0
    measured: str = ""

    def __post_init__(self) -> None:
        if (self.hours < 0.0 or self.per_lead_hours < 0.0
                or self.late_hours < 0.0 or self.from_lead < 0):
            raise ValueError(
                "a publication rule cannot be negative: a cycle is not "
                "published before it runs")

    def matches(self, cycle_hour: int, lead: int = 0) -> bool:
        return ((self.cycle_hours is None or int(cycle_hour) in self.cycle_hours)
                and max(0, int(lead)) >= self.from_lead)

    def at(self, lead: int = 0, *, settled: bool = False) -> float:
        return (float(self.hours) + float(self.per_lead_hours) * max(0, int(lead))
                + (float(self.late_hours) if settled else 0.0))


#: The walk-back a source declares nothing better than.  Two days is the
#: retention the operational NCEP servers publish for their own
#: directories, so it is the window in which a probe can still get an
#: answer rather than a guess about how stale a cycle may be.
DEFAULT_SEARCH_HOURS = 48


@dataclass(frozen=True)
class CycleGrid:
    """One source's initialization schedule, as data.

    Constructed from a registry row or derived from a fetch route; the
    resolver never learns which, which is the point.
    """

    hours: tuple[int, ...]
    delay_hours: float = 0.0
    search_hours: int = DEFAULT_SEARCH_HOURS
    basis: str = ""
    record_end: datetime | None = None
    #: Per-cycle-hour forecast horizon, most specific rule first:
    #: ``((cycle_hours_or_None, through_hour), ...)``.  ``None`` matches
    #: any hour, so a trailing ``(None, N)`` is the default rule.  The
    #: shape is the fetch route table's own ``ladders`` shape,
    #: deliberately: that is where a new model declares this, and a
    #: reader who has seen one has seen both.  Empty means the producer
    #: declares no per-cycle variation and no candidate is filtered on
    #: this ground.
    horizons: tuple[tuple[tuple[int, ...] | None, int], ...] = ()
    #: How long after a nominal init the whole run is usually on the
    #: server, measured, where ``delay_hours`` is zero because the probe
    #: decides publication.  ``None`` reads as ``delay_hours``: a source
    #: whose walk already starts at its measured lag.  See
    #: :attr:`usual_delay`.
    usual_delay_hours: float | None = None
    #: Per-cycle-hour and per-lead publication rules, most specific
    #: first (:class:`PublicationRule`, the fetch route table's
    #: ``publication_lag`` rows).  Empty means one delay for every cycle
    #: and lead: ``delay_hours``.
    delays: tuple[PublicationRule, ...] = ()
    #: UTC hours at which the source holds an ANALYSIS valid time, where
    #: that set is wider than the grid ``latest`` snaps to.  Empty means
    #: :attr:`hours`.  ERA5 keeps ``latest`` on 00/06/12/18 (whole
    #: 6-hourly files, the cheapest fetch) but publishes an analysis
    #: every hour, and the ordinary fetch takes any of them.  Breakage
    #: it prevents: a WOOF ensemble from ERA5 2011-04-27T16 (12 h) was
    #: refused by the WoFS ladder with "era5 2011-04-27T16Z does not
    #: cover the requested window" because its knots 16Z/22Z/04Z were
    #: checked against the ``latest`` grid.  Read through
    #: :attr:`analysis_valid_hours`.
    analysis_hours: tuple[int, ...] = ()

    @property
    def analysis_valid_hours(self) -> tuple[int, ...]:
        """The hours an analysis valid time may fall on."""
        return self.analysis_hours or self.hours

    def __post_init__(self) -> None:
        if self.analysis_hours and (
                sorted(set(self.analysis_hours)) != list(self.analysis_hours)
                or not all(0 <= hour <= 23 for hour in self.analysis_hours)):
            raise ValueError(
                f"CycleGrid analysis_hours {self.analysis_hours} must be "
                "sorted, unique UTC hours of day")
        if not self.hours:
            raise ValueError(
                "a CycleGrid must declare at least one UTC hour; a source "
                "with no cycle concept declares no grid at all, so the "
                "refusal can name the absence instead of an empty table")
        if sorted(set(self.hours)) != list(self.hours):
            raise ValueError(
                f"CycleGrid hours {self.hours} must be sorted and unique")
        if not all(0 <= hour <= 23 for hour in self.hours):
            raise ValueError(
                f"CycleGrid hours {self.hours} must be UTC hours of day")
        if self.delay_hours < 0.0:
            raise ValueError("CycleGrid delay_hours cannot be negative")
        if not all(isinstance(rule, PublicationRule) for rule in self.delays):
            raise ValueError("CycleGrid delays are PublicationRule rows")
        if self.usual_delay_hours is not None and self.usual_delay_hours < 0.0:
            raise ValueError("CycleGrid usual_delay_hours cannot be negative")
        if self.search_hours <= 0:
            raise ValueError(
                "CycleGrid search_hours must be a positive window; zero "
                "would refuse every cycle without looking at one")

    def horizon(self, cycle: datetime) -> int | None:
        """How far this cycle forecasts, or ``None`` if undeclared.

        A cycle that does not reach the end of the requested window is
        not the ``latest`` anything: it is a cycle that cannot serve the
        request.  RAP's 07Z run stops at f021 while its 09Z run reaches
        f051, and HRRR's off-synoptic hours stop at f018 -- filtering on
        a declared horizon is what stops ``latest`` returning one of
        those for a window that needs more.
        """

        for hours, through in self.horizons:
            if hours is None or cycle.hour in hours:
                return through
        return None

    def delay(self, cycle: datetime, lead: int = 0, *,
              settled: bool = False) -> float:
        """Hours after ``cycle`` until its ``lead`` is published, declared.

        The rule for the cycle's hour where :attr:`delays` has one, and
        :attr:`delay_hours` otherwise.  ECMWF posts its 06/18Z runs an
        hour before its 00/12Z ones and GEFS takes two hours longer to
        reach f384 than f000, so one number per producer either starts
        ``latest`` a cycle late or claims leads that are not out yet.

        The rule is the earliest posting seen, which is where a probed
        walk starts: earlier costs one probe that answers no, later skips
        a cycle that is out.  ``settled`` asks for the latest posting seen
        instead, for a reader that takes the schedule's word with no probe
        to check it (a plan made offline, a page before its check answers,
        a DA cycle's wait), where too early claims a lead that is not out.
        """

        for rule in self.delays:
            if rule.matches(cycle.hour, lead):
                return rule.at(lead, settled=settled)
        return float(self.delay_hours)

    @property
    def usual_delay(self) -> float:
        """Hours after a nominal init by which the whole run is usually published.

        A page takes a start at least this old as published while no
        check has answered for it, and opens on the newest such start
        until a check confirms a newer one.  The probe still decides:
        a start a check found missing is not taken, however old.  Where
        :attr:`delays` declares the timing per lead it is the latest
        posting seen of the last lead EVERY cycle hour publishes, for the
        slowest hour: the longest first-object delay took GEFS as whole
        at 3.7 h where its f384 lands after six.  A longer ladder some
        hours add (GEFS's 00Z extension, posted a day later) is left to
        the page's lead-aware ``due``, which asks about the window's own
        last lead.
        """

        if self.usual_delay_hours is not None:
            return float(self.usual_delay_hours)
        if self.delays:
            cycles = [datetime(2001, 1, 1, hour) for hour in self.hours]
            common = min(self.horizon(cycle) or 0 for cycle in cycles)
            return max(self.delay(cycle, common, settled=True) for cycle in cycles)
        return float(self.delay_hours)

    def declaration(self) -> dict[str, object]:
        """This grid as JSON-safe fields, for a manifest or a front end."""

        value = {
            "hours": list(self.hours),
            "delay_hours": float(self.delay_hours),
            "search_hours": int(self.search_hours),
            "basis": self.basis,
            "record_end": (None if self.record_end is None
                           else self.record_end.strftime("%Y-%m-%dT%H")),
            "horizons": [[None if hours is None else list(hours), through]
                         for hours, through in self.horizons],
        }
        if self.usual_delay_hours is not None:
            value["usual_delay_hours"] = float(self.usual_delay_hours)
        if self.delays:
            value["delays"] = [
                {"cycle_hours": (None if rule.cycle_hours is None
                                 else list(rule.cycle_hours)),
                 "from_lead": int(rule.from_lead), "hours": float(rule.hours),
                 "per_lead_hours": float(rule.per_lead_hours),
                 "late_hours": float(rule.late_hours)}
                for rule in self.delays]
        return value

    def snap(self, moment: datetime) -> datetime:
        """The newest grid point at or before ``moment``.

        Hour-by-hour rather than by arithmetic on a period: a grid is a
        SET of hours, not a spacing, and ICON-EU's 00/03/06/09/12/15/18/21
        and GEM-GDPS's 00/12 are both irregular enough that a period
        would have to be inferred and would be wrong for the next
        producer that runs an odd schedule.
        """

        candidate = moment.replace(minute=0, second=0, microsecond=0)
        for _step in range(24):
            if candidate.hour in self.hours:
                return candidate
            candidate -= timedelta(hours=1)
        raise AssertionError(              # pragma: no cover - hours is nonempty
            "a nonempty hour set is reached within one day")

    def newest(self, now: datetime, lead: int = 0, *,
               settled: bool = False) -> datetime:
        """The newest init this grid says exists through ``lead``, optimistically.

        The newest grid point whose declared delay for ``lead``
        (:meth:`delay`) has passed by ``now``, and never later than a
        closed archive's last init.  With one delay for every cycle this
        is ``now`` minus that delay, snapped back onto the grid.  This is
        the whole answer where no probe transport exists; where one does,
        it is the first candidate the probe is asked about.  ``settled``
        waits for the latest posting seen instead (:meth:`delay`), for an
        answer no probe will check.
        """

        newest = self.snap(now)
        longest = max([self.delay_hours] + [rule.at(lead, settled=True)
                                            for rule in self.delays])
        for _step in range(int(longest) + 48):
            if newest + timedelta(hours=self.delay(newest, lead, settled=settled)) <= now:
                break
            newest = self.snap(newest - timedelta(hours=1))
        if self.record_end is not None and newest > self.record_end:
            return self.snap(self.record_end)
        return newest

    def candidates(self, now: datetime, lead: int = 0) -> tuple[datetime, ...]:
        """Every cycle ``latest`` may resolve to through ``lead``, newest first."""

        newest = self.newest(now, lead)
        found = [newest]
        while True:
            earlier = self.snap(found[-1] - timedelta(hours=1))
            if (newest - earlier) > timedelta(hours=self.search_hours):
                return tuple(found)
            found.append(earlier)


def route_cycle_grid(source_id: str) -> CycleGrid | None:
    """The grid a source's FETCH ROUTE declares, or ``None``.

    The route table is measured against the producer and is where a new
    model is added, so a row there is the reason ``latest`` needs no
    per-source code: adding ICON, AIFS or a Canadian model is a route
    row, and this reads the ``cycle_hours`` it already had to carry.
    """

    try:
        from woof import fetch_routes

        route = fetch_routes.route_for(source_id)
    except (ImportError, ValueError, KeyError):
        return None
    hours = tuple(sorted({int(hour) for hour in route.cycle_hours}))
    if not hours:
        return None
    # The default host's retention IS the walk-back: it is how far back
    # the server this resolver probes still holds directories, so a
    # candidate older than that cannot be confirmed and asking about it
    # only spends HEADs.  A route that declares no retention takes the
    # module default rather than an unbounded walk.
    retention = next(
        (float(host.retention_hours) for host in route.hosts
         if getattr(host, "retention_hours", None)), None)
    return CycleGrid(
        hours=hours,
        # The longest first-object delay, for a reader that names no
        # cycle; every reader that has one asks :meth:`CycleGrid.delay`.
        delay_hours=max(rule.at(0) for rule in route.publication_lag
                        if rule.from_lead == 0),
        delays=tuple(route.publication_lag),
        search_hours=int(retention) if retention else DEFAULT_SEARCH_HOURS,
        # The last rung of each ladder rule IS the horizon: the steps are
        # (through_hour, spacing) pairs in increasing order, so the final
        # pair's first element is how far that cycle forecasts.
        horizons=tuple(
            (None if cycle_hours is None else tuple(cycle_hours), steps[-1][0])
            for cycle_hours, steps in route.ladders if steps),
        basis=f"the fetch route table's measured cycle_hours and "
              f"publication_lag rows for {source_id!r} "
              f"(measured {route.measured})")


def cycle_grid_for(source_id: str, *, posting: bool = False) -> CycleGrid | None:
    """The initialization grid for one source id, or ``None``.

    ``None`` means nothing in this product declares when this source
    initializes -- NOT that the source is unsupported.  The caller turns
    that into a refusal naming the missing declaration, so a row which
    later grows a fetch route starts resolving ``latest`` with no edit
    here at all.

    The registry row's own ``cycle_grid`` wins where it declares one:
    that column carries the sources whose schedule the route table
    cannot state (the legacy transports, and a keyed job API like the
    CDS, which has no object to probe and so needs its publication delay
    written down).  Everything else derives.

    ``posting`` asks for the grid a posting schedule reads: a legacy
    transport's measured lead times are the route table's
    ``legacy_posting`` rows, in a route's :class:`PublicationRule` shape,
    and they become that grid's :attr:`CycleGrid.delays`, so every source
    answers when a lead is due through :meth:`CycleGrid.delay`.  Without
    it a legacy grid keeps no delay: its completeness probe decides
    publication object by object from now, which is what ``--cycle
    latest`` and the date page's due start are built on for those
    sources until the as-posted ``latest`` (DESIGN A136 2.2) replaces
    them.  A route's grid is the same either way.
    """

    try:
        from woof.source_adapters import get_source_adapter

        adapter = get_source_adapter(source_id)
    except (ImportError, ValueError):
        return route_cycle_grid(source_id)
    grid = adapter.cycle_grid
    if grid is None:
        return route_cycle_grid(adapter.source_id)
    if not posting:
        return grid
    from woof import fetch_routes

    rules = fetch_routes.legacy_publication_lag(adapter.source_id)
    if rules:
        grid = replace(grid, delays=tuple(rules),
                       basis=f"{grid.basis}; lead timing from the fetch route "
                             f"table's legacy_posting rows for "
                             f"{adapter.source_id!r}")
    return grid


__all__ = ["DEFAULT_SEARCH_HOURS", "CycleGrid", "cycle_grid_for",
           "route_cycle_grid"]
