"""The operational HRRR namelist as tests import it since 2026-10-12.

Two decisions of the namelist door changed what NOAA's own namelist
(NOAA-EMC/HRRR v4.1.21 ``parm/conus/hrrr_wrf.nl``) becomes:

* its HRRR-Smoke chemistry (``chem_opt = 18``) is no longer imported to
  run with none.  It maps onto WOOF-ArWen's ``smoke`` set, and the four smoke
  settings WOOF-ArWen does not carry are refused by name
  (tests/test_operational_namelist_door.py pins that refusal).  A test
  that is about something else imports it through :func:`importable`,
  which changes exactly those four settings. HAILCAST remains enabled.
* the namelist omits ``&dynamics top_lid``, which now imports as WRF's
  open top instead of the profile's rigid lid.  A control pinned to the
  bytes or the hash of an earlier import reads the one lid line through
  :func:`with_parent_lid` / :func:`as_open_top`.
"""
from __future__ import annotations

import re

#: (pattern, replacement) -- each must match the namelist exactly once.
_SMOKE_EDITS = (
    (re.compile(r"^(\s*aer_drydep_opt\s*=\s*)111,", re.M), r"\g<1>0,"),
    (re.compile(r"^(\s*enh_vermix\s*=\s*)\.true\.,", re.M), r"\g<1>.false.,"),
    (re.compile(r"^\s*flam_part\s*=\s*0\.75,[^\n]*\n", re.M), ""),
    (re.compile(r"^(\s*simple_dir_fdb\s*=\s*)1,", re.M), r"\g<1>0,"),
)

OPEN_TOP_LINE = "\ntop_lid = false\n"
RIGID_LID_LINE = "\ntop_lid = true\n"


def importable(text: str) -> str:
    """NOAA's namelist with the four refused smoke settings changed to the
    spelling WOOF-ArWen carries (smoke dry deposition off, no enhanced mixing,
    no flaming fraction, no direct radiative feedback)."""
    for pattern, replacement in _SMOKE_EDITS:
        text, count = pattern.subn(replacement, text)
        assert count == 1, pattern.pattern
    return text


_SMOKE_SELECTOR = re.compile(r"^(\s*chem_opt\s*=\s*)18,", re.M)


def without_smoke(text: str) -> str:
    """NOAA's namelist with its chemistry switched off (``chem_opt = 0``):
    the meteorology alone, for a door that cannot carry chemistry."""
    text, count = _SMOKE_SELECTOR.subn(r"\g<1>0,", text)
    assert count == 1
    return text


def write_importable(source, target):
    """Write :func:`importable` of ``source`` to ``target``; returns it."""
    target.write_text(importable(source.read_text(encoding="utf-8")),
                      encoding="utf-8", newline="\n")
    return target


def with_parent_lid(emitted: str) -> str:
    """``emitted`` with its open top written as the rigid lid an earlier
    importer resolved for the same omitted key (for a hash pin)."""
    assert emitted.count(OPEN_TOP_LINE) == 1, "one open-top line expected"
    return emitted.replace(OPEN_TOP_LINE, RIGID_LID_LINE)


def as_open_top(pinned: bytes) -> bytes:
    """Pinned bytes of an earlier import, their rigid lid read as the open
    top today's importer writes for the same omitted key."""
    rigid, open_top = RIGID_LID_LINE.encode(), OPEN_TOP_LINE.encode()
    assert pinned.count(rigid) == 1, "one rigid-lid line expected"
    return pinned.replace(rigid, open_top)
