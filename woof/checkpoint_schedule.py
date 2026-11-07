"""Write only the checkpoints a later stage restarts from.

A forecast checkpoints at every ``restart_interval`` instant.  A caller that
chains legs (a cycling driver, a forecast launched from one analysis) reads
one of them -- the instant the next leg restarts from -- and deletes the
rest unread.  Measured on a 300 x 300 x 49 member with restart every 15 min:
four 818 MB archives an hour, of which a 15-minute cycle reads one.

``--checkpoint-at`` names the instants to keep, as WRF valid times
(``2022-06-13_22:00:00``) or ``end`` for the run's stop instant.  The
schedule itself does not move: the forecast still reaches every
``restart_interval`` instant, and at the ones not named it writes nothing.
An instant that is not a ``restart_interval`` instant of the run is refused
before the forecast starts, because it would never be written and the leg
that needs it would find nothing.  A worker's own restart request
(``WOOF_RESTART_REQUEST_FILE``) is always honoured.

Absent -- the default -- every scheduled checkpoint is written, as before.
"""

from __future__ import annotations

import contextlib
import contextvars
from datetime import datetime, timedelta
from fractions import Fraction

_SELECTION: contextvars.ContextVar = contextvars.ContextVar(
    "gpuwm_checkpoint_instants", default=None)

#: The spelling of the run's stop instant.
END = "end"


def parse(text: str | None) -> tuple[str, ...] | None:
    """The instants of one ``--checkpoint-at`` value, or ``None``."""
    if text is None:
        return None
    items = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if not items:
        raise ValueError("--checkpoint-at names no instant; give WRF valid "
                         "times (2022-06-13_22:00:00) or 'end'")
    for item in items:
        if item != END:
            _valid_time(item)
    return items


def _valid_time(item: str) -> datetime:
    for pattern in ("%Y-%m-%d_%H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(item, pattern)
        except ValueError:
            continue
    raise ValueError(f"--checkpoint-at {item!r} is not a WRF valid time "
                     "(YYYY-MM-DD_HH:MM:SS) or 'end'")


class Selection:
    """The model ticks at which a scheduled checkpoint is written."""

    def __init__(self, ticks: frozenset[int], names: tuple[str, ...]):
        self.ticks = ticks
        self.names = names

    def keeps(self, ticks: int) -> bool:
        return int(ticks) in self.ticks


def resolve(items, *, start_time: datetime, run_ticks: int,
            restart_ticks: int | None, tick_den: int,
            restart_origin_ticks: int = 0) -> Selection:
    """Ticks for ``items`` in a run, refusing an instant never written.

    ``start_time`` is the simulation start the checkpoint clock counts
    from, ``run_ticks`` the run's length from that start, ``restart_ticks``
    the restart interval (``None`` when the run writes no checkpoint), all
    in ticks of ``1 / tick_den`` s.
    """
    if restart_ticks is None:
        raise ValueError(
            "--checkpoint-at selects among the run's scheduled checkpoints, "
            "and this run schedules none (restart_interval is off)")
    start = start_time.replace(tzinfo=None)
    ticks = set()
    for item in items:
        if item == END:
            at = int(run_ticks)
        else:
            offset = Fraction(int((_valid_time(item) - start)
                                  / timedelta(microseconds=1)), 10 ** 6)
            at = offset * int(tick_den)
            if at.denominator != 1:
                raise ValueError(
                    f"--checkpoint-at {item}: not a whole tick of this run's "
                    "clock, so no checkpoint is ever written there")
            at = int(at)
        if at <= 0 or at > int(run_ticks):
            raise ValueError(
                f"--checkpoint-at {item}: outside this run, which ends "
                f"{float(Fraction(int(run_ticks), int(tick_den))):g} s after "
                f"{start:%Y-%m-%d_%H:%M:%S}; that checkpoint would never be "
                "written")
        if ((at-int(restart_origin_ticks)) % int(restart_ticks) != 0
                and not (item == END and restart_origin_ticks)):
            raise ValueError(
                f"--checkpoint-at {item}: not a restart_interval "
                f"({float(Fraction(int(restart_ticks), int(tick_den))):g} s) "
                "instant of this run, so no checkpoint is ever written there")
        ticks.add(at)
    return Selection(frozenset(ticks), tuple(items))


@contextlib.contextmanager
def selected(items):
    """Run the enclosed forecast with ``items`` as its checkpoint instants."""
    token = _SELECTION.set(None if items is None else tuple(items))
    try:
        yield
    finally:
        _SELECTION.reset(token)


def current():
    """The instants set by :func:`selected`, or ``None`` (write every one)."""
    return _SELECTION.get()


def restart_requested() -> bool:
    """Whether a worker's own restart request is pending."""
    from woof.core.restart_request import RestartRequest
    return RestartRequest().pending()


__all__ = ["END", "Selection", "current", "parse", "resolve",
           "restart_requested", "selected"]
