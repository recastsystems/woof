"""``physics.mode``: the matched arithmetic or the corrected arithmetic.

The shared key of every route:

``"exact"``
    The matched arithmetic (native MPAS-A v8.4.2, the engine's WRF/MMM
    transcriptions), defects included.  With it the door, the driver, the
    sealed constructor and the engine seam are built exactly as before the
    key existed, so the byte-identity claims hold.  Every identity proof and
    parity configuration names ``exact`` explicitly.
``"correct"``
    Every correction the run's code carries, minus ``physics.correct_skip``.
    This is the default of every entry point hex ships (correct mode is
    the standard; exact stays one flag away).

ONE TABLE OF FINDINGS.  :data:`FINDINGS` names every switchable finding id
the hex routes know and where its correction lives.  The WOOF-ArWen column route's
water-budget corrections live in the woof engine (``woof.correct_mode``);
the IAU corrections in :mod:`woof.hex.iau_v842`; the boundary corrections in
the engine's ``rw_mpas_lbc``.  :func:`known_findings` adds the MPAS-suite
driver's own table (``woof.hex.mpasphys.driver_v842.CORRECTIONS_V842``) and
the imported engine's registry when they exist, so a skip list is checked
against one set of names whichever route runs.

A CORRECTION THE RUN CANNOT APPLY IS A LOUD NAMED GAP, NEVER A REFUSAL
(since 0.4.0).  When a correct-mode run reaches code
that does not carry a correction (an engine without ``woof.correct_mode``,
an engine seam that takes no mode keywords, an ``rw_mpas_lbc`` without
``--physics-mode``), the run goes ahead with the matched arithmetic for that
finding, prints one ``CORRECT MODE: named gap`` line per finding on stderr,
raises a :class:`CorrectModeGapWarning`, and lists the finding under
``correct_mode.gaps`` in its receipt ({finding, slot, reason, owner}, the
shape the MPAS-suite route writes).  The receipt never says a finding was
applied when it was not: ``applied`` lists only what the code carried.

A misspelt mode or an unknown finding id is still refused at the door, before
any input is read: a typo must never silently mean either mode.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

PHYSICS_MODE_EXACT = "exact"
PHYSICS_MODE_CORRECT = "correct"
PHYSICS_MODES = (PHYSICS_MODE_EXACT, PHYSICS_MODE_CORRECT)
#: The default of every shipped entry point (door, presets, boundary door).
DEFAULT_PHYSICS_MODE = PHYSICS_MODE_CORRECT

#: The findings that act outside the MPAS suite route, read from the ONE
#: finding table (``woof.hex.mpasphys.driver_v842.CORRECTIONS_V842``, the rows
#: that carry ``hex_route``).  ``route``: the code path the finding changes.
#: ``lives``: the code that carries the correction.  ``scheme``: the scheme it
#: acts in, when one.  ``slot``: the physics or door slot it changes.
#: ``owner``: who closes a gap in it.  ``what``: the correction, in one line.
def _findings_from_table() -> dict[str, Mapping[str, str]]:
    from .mpasphys.driver_v842 import CORRECTIONS_V842

    return {finding: dict(row["hex_route"]) for finding, row in CORRECTIONS_V842.items()
            if "hex_route" in row}


FINDINGS: Mapping[str, Mapping[str, str]] = _findings_from_table()

#: Back-compatible view: id -> "lives: what".
KNOWN_FINDINGS: Mapping[str, str] = {
    finding: f"{row['lives']}: {row['what']}" for finding, row in FINDINGS.items()}

#: The findings the ArWen column route expects the engine to carry.
ENGINE_COLUMN_FINDINGS: tuple[str, ...] = tuple(
    finding for finding, row in FINDINGS.items() if row["route"] == "arwen-column")
#: The findings the boundary door expects rw_mpas_lbc to carry.
LBC_FINDINGS: tuple[str, ...] = tuple(
    finding for finding, row in FINDINGS.items() if row["route"] == "lbc")

GAP_SCHEMA = "gpuwm-hex.correct-mode-gap/v1"


class CorrectModeRefusal(ValueError):
    """A physics.mode request that is misspelt (never a missing correction)."""


class CorrectModeGapWarning(UserWarning):
    """A correct-mode run that reached code without one of its corrections."""


@dataclass(frozen=True)
class CorrectionGap:
    """One correction a correct-mode run asked for and did not get.

    Its receipt row is the shape the MPAS-suite route writes
    (``correct_mode.gaps``): ``finding``, ``slot``, ``reason``, ``owner``, plus
    the detail keys ``what``, ``where`` and ``applied`` (always False).
    """

    finding: str
    where: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        row = FINDINGS.get(self.finding, {})
        return {"finding": self.finding, "slot": row.get("slot", ""),
                "reason": self.reason, "owner": row.get("owner", ""),
                "what": row.get("what", ""), "where": self.where,
                "applied": False, "schema": GAP_SCHEMA}


def _engine_registry() -> Mapping[str, Any] | None:
    try:
        from woof import correct_mode as engine_mode
    except ImportError:
        return None
    return getattr(engine_mode, "CORRECTIONS_BY_FINDING", None)


def _port_registry() -> Mapping[str, Any]:
    try:
        from .mpasphys.driver_v842 import CORRECTIONS_V842
    except ImportError:
        return {}
    return CORRECTIONS_V842


def known_findings() -> frozenset[str]:
    """Every finding id a skip list may name: this table, the MPAS-suite
    driver's table and the imported engine's registry (one set of names)."""

    names = set(FINDINGS) | set(_port_registry())
    try:
        from .mpasphys.rrfs.fork_corrections import FORK_CORRECTIONS
        names |= set(FORK_CORRECTIONS)
    except ImportError:
        pass
    engine = _engine_registry()
    if engine:
        names |= set(engine)
    return frozenset(names)


def route_findings(route: str) -> frozenset[str]:
    """The finding ids one route applies: this table's rows for that route, plus,
    for the WOOF-ArWen column route, every finding the imported engine carries."""

    names = {finding for finding, row in FINDINGS.items() if row.get("route") == route}
    if route == "arwen-column":
        engine = _engine_registry()
        if engine:
            names |= set(engine)
    return frozenset(names)


def validate(mode: Any, correct_skip: Iterable[str] | None = ()) -> tuple[str, tuple[str, ...]]:
    """``(mode, skip)`` normalised, or a refusal naming the bad spelling."""

    if not isinstance(mode, str) or mode not in PHYSICS_MODES:
        raise CorrectModeRefusal(
            f"physics.mode = {mode!r} is not a physics mode; known modes: "
            f"{list(PHYSICS_MODES)}")
    if isinstance(correct_skip, str):
        correct_skip = (correct_skip,)
    known = known_findings()
    skip: list[str] = []
    for finding in correct_skip or ():
        if finding not in known:
            raise CorrectModeRefusal(
                f"physics.correct_skip names {finding!r}, which this tree "
                f"cannot switch; known: {sorted(known)}")
        if finding in skip:
            raise CorrectModeRefusal(f"physics.correct_skip names {finding!r} twice")
        skip.append(finding)
    if mode == PHYSICS_MODE_EXACT and skip:
        raise CorrectModeRefusal(
            "physics.correct_skip is set while physics.mode = 'exact', which "
            "applies no correction; name the skip list on a correct-mode arm")
    return mode, tuple(skip)


def terminal_record(value: Any) -> Any:
    """``value`` as a forecast prints it in the terminal: every correct-mode gap row without its ``owner`` field.

    The receipt file keeps the field (other programs read it); the line printed when a forecast finishes names each
    gap, its slot and its reason, which is what a user acts on. The value given is not changed."""
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if key == "gaps" and isinstance(item, list):
                out[key] = [{k: v for k, v in row.items() if k != "owner"} if isinstance(row, Mapping)
                            else terminal_record(row) for row in item]
            else:
                out[key] = terminal_record(item)
        return out
    if isinstance(value, list):
        return [terminal_record(item) for item in value]
    return value


def report_gaps(gaps: Iterable[CorrectionGap], *, context: str = "") -> list[dict[str, Any]]:
    """Print each gap loudly on stderr, warn once, return the receipt rows."""

    rows = []
    for gap in gaps:
        rows.append(gap.as_dict())
        row = FINDINGS.get(gap.finding, {})
        where = f" ({context})" if context else ""
        print(f"CORRECT MODE: named gap {gap.finding} NOT APPLIED{where}: "
              f"{row.get('what', gap.finding)}; slot {row.get('slot', '?')}; "
              f"{gap.where}: {gap.reason}",
              file=sys.stderr, flush=True)
    if rows:
        warnings.warn(
            "physics.mode = 'correct' ran WITHOUT "
            + ", ".join(row["finding"] for row in rows)
            + " (named gaps in the run receipt)",
            CorrectModeGapWarning, stacklevel=2)
    return rows


def receipt_block(mode: str, *, applied: Iterable[str] = (), skipped: Iterable[str] = (),
                  gaps: Iterable[Mapping[str, Any]] = (),
                  engine: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The ``correct_mode`` receipt block, the same keys on every route:
    ``mode``, ``applied``, ``skipped``, ``gaps`` (plus ``engine``, the engine's
    own resolution, when an engine resolved the corrections)."""

    block: dict[str, Any] = {"mode": mode, "applied": list(applied),
                             "skipped": list(skipped), "gaps": [dict(g) for g in gaps]}
    if engine is not None:
        block["engine"] = dict(engine)
    return block


#: What this process's engine resolutions decided, for the run receipt the
#: forecast door writes after the run (the door runs the driver and its
#: physics backends in its own process; a forked worker keeps its own copy).
#: THE BREAKAGE THIS PREVENTS: the backends' own receipts are not persisted on
#: every route (the full-step LTS route writes none of them), so without this a
#: correct-mode run's named gaps reached stderr and never its run receipt.
_RUN_RECORD: dict[str, Any] = {"resolutions": 0, "applied": [], "gaps": []}


def _record(applied: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> None:
    _RUN_RECORD["resolutions"] += 1
    for finding in applied:
        if finding not in _RUN_RECORD["applied"]:
            _RUN_RECORD["applied"].append(finding)
    seen = {(g["finding"], g["where"], g["reason"]) for g in _RUN_RECORD["gaps"]}
    for row in rows:
        key = (row["finding"], row["where"], row["reason"])
        if key not in seen:
            seen.add(key)
            _RUN_RECORD["gaps"].append(dict(row))


def run_record(mode: str, correct_skip: Iterable[str] = ()) -> dict[str, Any]:
    """The ``correct_mode`` block for this process's run receipt: what every
    engine resolution in this process applied and every named gap it hit."""

    block = receipt_block(mode, applied=_RUN_RECORD["applied"], skipped=tuple(correct_skip),
                          gaps=_RUN_RECORD["gaps"])
    block["engine_resolutions"] = int(_RUN_RECORD["resolutions"])
    if mode != PHYSICS_MODE_EXACT and not _RUN_RECORD["resolutions"]:
        block["note"] = ("no engine resolution ran in this process (physics off, or the physics ran in "
                         "another process, whose own receipt holds its correct_mode block)")
    return block


def reset_run_record() -> None:
    """Forget this process's record (a new run in the same process)."""

    _RUN_RECORD.update(resolutions=0, applied=[], gaps=[])


def _rows(gaps: list[CorrectionGap], report: bool) -> list[dict[str, Any]]:
    if report:
        return report_gaps(gaps, context="WOOF-ArWen column physics")
    return [gap.as_dict() for gap in gaps]


def engine_accepts(target: Any) -> bool:
    """True when an engine seam's signature takes the mode keywords (or
    cannot be inspected, in which case the call itself decides)."""

    import inspect

    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return True
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return True
    return {"physics_mode", "physics_correct_skip"} <= set(parameters)


def engine_settings(mode: str, correct_skip: Iterable[str] = (), *,
                    seam: Any = None, seam_label: str = "",
                    active_schemes: Iterable[str] | None = None,
                    report: bool = True) -> dict[str, Any]:
    """The imported engine's resolution for the WOOF-ArWen column route.

    Returns ``{"settings", "receipt", "gaps", "engine_takes_mode"}``.  Exact
    mode returns no settings and no receipt, and imports no engine module.

    Correct mode resolves the engine's corrections.  An engine without
    ``woof.correct_mode``, or a ``seam`` (the engine callable the run
    constructs) whose signature takes no mode keywords, cannot apply them:
    each expected finding not skipped becomes a named gap, the settings are
    empty (so the engine computes the matched arithmetic) and
    ``engine_takes_mode`` is False, telling the caller to keep the mode
    keywords out of the engine call.  ``active_schemes``, when given, limits
    the gaps to findings whose scheme the run actually runs.  ``report=False``
    computes the gaps without printing them (a sealing step whose backend
    reports them once, in its receipt).
    """

    mode, skip = validate(mode, correct_skip)
    if mode == PHYSICS_MODE_EXACT:
        return {"settings": {}, "receipt": None, "gaps": [], "engine_takes_mode": True}
    schemes = None if active_schemes is None else {str(s) for s in active_schemes}

    def relevant(finding: str) -> bool:
        scheme = FINDINGS.get(finding, {}).get("scheme")
        return schemes is None or scheme is None or scheme in schemes

    expected = [f for f in ENGINE_COLUMN_FINDINGS if f not in skip and relevant(f)]
    try:
        from woof import correct_mode as engine_mode
    except ImportError:
        engine_mode = None
    reason = None
    if engine_mode is None:
        reason = "the imported woof engine has no correct mode (woof.correct_mode)"
    elif seam is not None and not engine_accepts(seam):
        reason = (f"the engine's {seam_label or 'column seam'} takes no "
                  "physics_mode/physics_correct_skip keywords")
    if reason is not None:
        gaps = [CorrectionGap(f, "engine", reason) for f in expected]
        receipt = receipt_block(mode, applied=(), skipped=skip, gaps=_rows(gaps, report))
        if report:
            _record((), receipt["gaps"])
        return {"settings": {}, "receipt": receipt, "gaps": gaps,
                "engine_takes_mode": False}
    registry = getattr(engine_mode, "CORRECTIONS_BY_FINDING", None)
    if registry is not None:
        engine_skip = tuple(f for f in skip if f in registry)
    else:
        engine_skip = tuple(f for f in skip if f in ENGINE_COLUMN_FINDINGS)
    resolution = engine_mode.resolve(mode, engine_skip, source="physics.mode")
    carried = (set(registry) if registry is not None else
               set(getattr(resolution, "applied", ())) | set(engine_skip))
    missing = [f for f in expected if f not in carried]
    gaps = [CorrectionGap(f, "engine", "the imported engine's woof.correct_mode "
                          "does not carry this finding") for f in missing]
    receipt = receipt_block(mode, applied=tuple(getattr(resolution, "applied", ())),
                            skipped=skip, gaps=_rows(gaps, report),
                            engine=dict(resolution.receipt()))
    if report:
        _record(receipt["applied"], receipt["gaps"])
    return {"settings": dict(resolution.settings), "receipt": receipt, "gaps": gaps,
            "engine_takes_mode": True}


def door_argv(mode: str, correct_skip: Iterable[str] = ()) -> list[str]:
    """The driver flags for a resolved mode; EMPTY in exact mode, so an
    exact request hands the driver the argument vector it always did."""

    mode, skip = validate(mode, correct_skip)
    if mode == PHYSICS_MODE_EXACT:
        return []
    argv = ["--physics-mode", mode]
    for finding in skip:
        argv += ["--physics-correct-skip", finding]
    return argv


#: Keys a sealed constructor mapping MAY carry beyond its exhaustive set.
#: Present only in correct mode, so an exact-mode mapping -- and its sealed
#: identity digest -- is the one built before the key existed.
OPTIONAL_CONSTRUCTOR_KEYS = frozenset(("physics_mode", "physics_correct_skip"))


def sealed_scalars(values: Mapping[str, Any]) -> dict[str, Any]:
    """The correct-mode scalars a sealed constructor stores: none in exact
    mode or when the mapping names no mode, else the mode and skip list."""

    if "physics_mode" not in values and "physics_correct_skip" not in values:
        return {}
    mode, skip = validate(values.get("physics_mode", PHYSICS_MODE_EXACT),
                          values.get("physics_correct_skip", ()))
    if mode == PHYSICS_MODE_EXACT:
        return {}
    return {"physics_mode": mode, "physics_correct_skip": skip}


def without_mode_keywords(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """An engine call's keywords with the mode keys removed (a named-gap run
    on an engine seam that takes none)."""

    return {k: v for k, v in kwargs.items() if k not in OPTIONAL_CONSTRUCTOR_KEYS}


__all__ = [
    "CorrectModeGapWarning", "CorrectModeRefusal", "CorrectionGap",
    "DEFAULT_PHYSICS_MODE", "ENGINE_COLUMN_FINDINGS", "FINDINGS", "GAP_SCHEMA",
    "KNOWN_FINDINGS", "LBC_FINDINGS", "OPTIONAL_CONSTRUCTOR_KEYS",
    "PHYSICS_MODES", "PHYSICS_MODE_CORRECT", "PHYSICS_MODE_EXACT",
    "door_argv", "engine_accepts", "engine_settings", "known_findings", "terminal_record",
    "receipt_block", "report_gaps", "reset_run_record", "route_findings", "run_record",
    "sealed_scalars", "validate",
    "without_mode_keywords",
]
