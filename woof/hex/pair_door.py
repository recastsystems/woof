"""``woof hex pair`` -- one authored forecast, run twice, differenced.

WHAT THIS DOOR IS FOR, and it is not convenience.

A question of the form *"what did the model do differently when this point
source table was applied?"* is answered by two runs, and the answer is only
worth reading when the two runs differ in ONE thing.  Every part of that is
easy to get wrong by hand: two configuration files drift, one leg picks up a
stale kernel cache from the other, the two output trees land beside each
other with no record of which was which, and the difference that comes out is
attributed to the treatment when it was really a differing timestep, a
differing mixing lane or a differing init.

THE BREAKAGE THIS DOOR PREVENTS is exactly that: a difference published
against a control that was never the same experiment.  So:

* **There is ONE authored request.**  The user writes the forecast argument
  vector once, on this command line, and adds ``--source-table``.  The
  treatment leg is that vector plus the table; the control leg is that vector
  with the table absent.  The control is DERIVED.  There is no second
  configuration to keep in step, because there is no second configuration.
* **The derivation is re-proved rather than trusted.**
  :func:`assert_pair_identity` reads both argument vectors back as tokens and
  refuses unless they differ in exactly ``--source-table`` and the per-leg
  destinations.  Every other differing token is NAMED.  A derivation that
  cannot be checked is a derivation nobody has to keep correct.
* **The legs run through the forecast door**, as subprocesses, in the order
  control then treatment.  This door reimplements no part of the model, no
  part of admission and no part of the receipt: it drives the same front door
  a user drives, so a pair cannot be running a configuration the single door
  would have refused.
* **Both trees are fresh and separate.**  ``--pair-out`` must be empty, and
  each leg gets its own output tree and its own kernel cache derived the way
  the forecast door derives its own defaults, so neither leg can read the
  other's temporaries.

WHAT THE OUTPUT IS, AND WHAT IT IS NOT.  ``summary.txt`` opens with the
sentence this door exists to keep attached to its own numbers: a difference
between the two legs is a MODEL RESULT.  It states what this model did when
the table was present.  It is not a measurement of the atmosphere and it is
not evidence that the atmosphere would have done the same.  The manifest
carries digests so a reader can check that the numbers came from the files
they name; nothing here makes a claim about the world.

This module is stdlib-only at import.  numpy and netCDF4 are pulled inside
the summary, so ``woof hex pair --help`` and every argument refusal work on
a box with no scientific stack at all.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as _dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from .host_identity import host_record
from .errors import MpasPortError
from .forecast_door import add_forecast_arguments, default_scratch, history_files

__all__ = [
    "EXTRA_FILE_PATTERNS",
    "LEG_NAMES",
    "MANIFEST_NAME",
    "PAIR_SCHEMA",
    "PER_LEG_FLAGS",
    "RESULT_SENTENCE",
    "SOURCE_TABLE_FLAG",
    "SUMMARY_NAME",
    "LegPlan",
    "PairRefusal",
    "PairRequest",
    "add_pair_arguments",
    "add_pair_parser",
    "assert_pair_identity",
    "authored_forecast_argv",
    "collect_leg",
    "extra_files",
    "forecast_option_table",
    "forecast_subprocess_runner",
    "leg_plans",
    "resolve_pair_request",
    "run_pair",
    "run_pair_door",
    "sha256_file",
    "write_summary",
]

#: The manifest this door writes, and the identity a reader keys on.
PAIR_SCHEMA = "gpuwm-hex.pair/v1"

MANIFEST_NAME = "pair_manifest.json"
SUMMARY_NAME = "summary.txt"

#: The one flag the treatment leg carries and the control leg does not.
SOURCE_TABLE_FLAG = "--source-table"

#: Control first, always.  A control that runs second has already had the
#: card warmed, the cache populated and the disk filled by its own treatment,
#: and the cheapest way to keep that out of the comparison is to fix the
#: order and record it.
LEG_NAMES: tuple[str, str] = ("control", "treatment")

#: Forecast-door destinations this door derives per leg rather than
#: forwarding.  They are the ONLY tokens allowed to differ between the two
#: legs besides the source table itself.
PER_LEG_FLAGS: tuple[str, ...] = ("--out", "--scratch", "--icechunk-store")

#: Forecast-door arguments the pair door owns and therefore refuses on its
#: own command line, mapped to what a user should have written instead.  A
#: forwarded ``--out`` would put both legs in one tree; a forwarded
#: ``--receipt`` would have the second leg overwrite the first one's receipt.
OWNED_BY_THIS_DOOR: dict[str, str] = {
    "out": (
        "--out names ONE destination and a pair has two.  This door derives "
        "<pair-out>/control/out and <pair-out>/treatment/out so neither leg "
        "can write over the other's frames.  Pass --pair-out instead"
    ),
    "receipt": (
        "--receipt names ONE file and a pair writes two receipts.  Each leg "
        "writes its own at <leg>/out/forecast-receipt.json, which is the "
        "forecast door's own default, and the pair manifest records both "
        "paths"
    ),
}

#: Files a leg may drop beside its history that this door carries forward
#: without reading.  They are OPAQUE here on purpose: whatever a leg records
#: about its own accounting is that leg's business, and a differencing door
#: that started parsing them would acquire an opinion about a format it does
#: not own.  Paths and digests go into the manifest so a reader can find them
#: and prove they have not moved.
EXTRA_FILE_PATTERNS: tuple[str, ...] = (
    "*budget*.json",
    "*ledger*.csv",
    "*ledger*.json",
)

#: The first line of every summary this door writes.  It is a constant rather
#: than a sentence assembled at write time because it is the one line that
#: must never vary with the run, and a reader quoting the summary quotes it.
RESULT_SENTENCE = (
    "A difference between these two legs is a MODEL RESULT, not evidence "
    "about the atmosphere: it states what this model did when the "
    "point-source table was present, and says nothing about what the real "
    "atmosphere would have done."
)


class PairRefusal(MpasPortError):
    """A named pair-door refusal: what breaks, then the remedy."""


def _refuse(message: str) -> PairRefusal:
    return PairRefusal(message)


#: What ``write_summary`` returns when a leg left no frame to difference.
NO_FRAMES_WEIGHTING = "no frames to weight"


def sha256_file(path: Path) -> str:
    """Digest a file without pulling it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# the forecast door's own argument surface, read rather than restated
# ---------------------------------------------------------------------------
def _forecast_actions() -> tuple[argparse.Action, ...]:
    """Every action ``add_forecast_arguments`` installs, in its own order.

    Read off a throwaway parser rather than transcribed.  A transcription is
    a second list of the forecast door's flags, and the two doors would drift
    the first time one flag was added to that door and not to this one --
    silently, because a pair whose legs both lack a flag still looks like a
    matched pair.
    """

    probe = argparse.ArgumentParser(add_help=False)
    add_forecast_arguments(probe)
    return tuple(probe._actions)


def forecast_option_table() -> dict[str, bool]:
    """``{option string: takes a value}`` for the forecast door plus this one.

    The table is what :func:`assert_pair_identity` tokenises argument vectors
    against, so an unknown token is NAMED instead of being silently paired
    with whatever followed it.
    """

    table: dict[str, bool] = {}
    for action in _forecast_actions():
        takes_value = action.nargs != 0
        for option in action.option_strings:
            table[option] = takes_value
    table[SOURCE_TABLE_FLAG] = True
    return table


def authored_forecast_argv(arguments: argparse.Namespace) -> list[str]:
    """The ONE authored forecast vector, rebuilt from the parsed request.

    Every forecast option resolves to an explicit token here, defaults
    included, for two reasons.  The manifest then records WHAT RAN rather
    than what was typed, and the two legs are explicit in the same way, so
    the identity proof compares tokens rather than an absence against a
    default that could later change.

    The per-leg destinations are excluded: they are derived, not authored.
    """

    argv: list[str] = []
    for action in _forecast_actions():
        if action.dest in OWNED_BY_THIS_DOOR or action.dest in ("scratch", "icechunk_store"):
            continue
        if SOURCE_TABLE_FLAG in action.option_strings:
            # The one authored-but-not-shared token: the treatment leg adds
            # it, the control leg never carries it (leg_plans).
            continue
        value = getattr(arguments, action.dest, None)
        if value is None:
            continue
        if action.nargs == 0:
            if value:
                argv.append(action.option_strings[0])
            continue
        argv += [action.option_strings[0], str(value)]
    return argv


# ---------------------------------------------------------------------------
# the identity proof
# ---------------------------------------------------------------------------
def _tokenise(argv: Sequence[str], leg: str) -> dict[str, str | None]:
    """``argv`` as ``{flag: value}``, or a refusal that names the bad token."""

    table = forecast_option_table()
    seen: dict[str, str | None] = {}
    tokens = list(argv)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token not in table:
            raise _refuse(
                f"the {leg} argument vector carries {token!r}, which is not a "
                f"flag of the forecast door or of this one.  This door proves "
                f"the two legs differ in exactly {SOURCE_TABLE_FLAG} and "
                f"{', '.join(PER_LEG_FLAGS)}, and it cannot prove that about "
                f"a token it cannot read: an unrecognised flag would be paired "
                f"with whatever followed it and a real difference between the "
                f"legs would go unreported.  Known flags: "
                f"{', '.join(sorted(table))}"
            )
        if token in seen:
            raise _refuse(
                f"the {leg} argument vector gives {token!r} twice.  The last "
                f"one would win inside argparse and the identity proof would "
                f"compare the wrong value, so a repeated flag is refused "
                f"rather than resolved"
            )
        if table[token]:
            if index + 1 >= len(tokens):
                raise _refuse(
                    f"the {leg} argument vector ends at {token!r}, which takes "
                    f"a value.  A trailing flag with no value cannot be "
                    f"compared against the other leg"
                )
            seen[token] = tokens[index + 1]
            index += 2
        else:
            seen[token] = None
            index += 1
    return seen


def assert_pair_identity(
    control_argv: Sequence[str], treatment_argv: Sequence[str]
) -> dict[str, Any]:
    """Re-prove that the two legs differ in exactly the one thing they may.

    Returns the token diff, which the manifest carries whole.  Raises
    :class:`PairRefusal` NAMING every token that differs and is not allowed
    to, because "the legs drifted" is not an answer anybody can act on.

    This is a check on the ARGUMENT VECTORS, not on the derivation that built
    them, and that is the point: a derivation is code, and code changes.  The
    proof is applied to the output no matter where the output came from.
    """

    control = _tokenise(control_argv, "control")
    treatment = _tokenise(treatment_argv, "treatment")

    problems: list[str] = []

    if SOURCE_TABLE_FLAG in control:
        problems.append(
            f"{SOURCE_TABLE_FLAG} {control[SOURCE_TABLE_FLAG]!r} appears in "
            f"the CONTROL leg.  The control is the leg without the table; a "
            f"control that carries one is not a control, and the difference "
            f"between the legs would measure nothing"
        )
    if SOURCE_TABLE_FLAG not in treatment:
        problems.append(
            f"{SOURCE_TABLE_FLAG} is absent from the TREATMENT leg.  With no "
            f"table on either side the two legs are one run performed twice, "
            f"and every reported difference would be numerical noise "
            f"presented as a treatment effect"
        )

    per_leg: dict[str, dict[str, Any]] = {flag: {} for flag in PER_LEG_FLAGS}
    for flag in PER_LEG_FLAGS:
        in_control = flag in control
        in_treatment = flag in treatment
        if in_control != in_treatment:
            present, absent = (
                ("control", "treatment") if in_control else ("treatment", "control")
            )
            problems.append(
                f"{flag} is given for the {present} leg and not for the "
                f"{absent} leg.  Each leg needs its own destination, and one "
                f"derived beside one defaulted is how two legs come to share "
                f"a directory"
            )
            continue
        if not in_control:
            continue
        if control[flag] == treatment[flag]:
            problems.append(
                f"{flag} is {control[flag]!r} for BOTH legs.  Two legs writing "
                f"one destination overwrite each other's output, and the "
                f"forecast door refuses the second one only after the first "
                f"has already run"
            )
            continue
        per_leg[flag] = {"control": control[flag], "treatment": treatment[flag]}

    allowed = set(PER_LEG_FLAGS) | {SOURCE_TABLE_FLAG}
    value_differs: dict[str, list[str | None]] = {}
    control_only: list[str] = []
    treatment_only: list[str] = []
    drifted = (
        "The legs then differ in {flag} as well as the source table, and no "
        "difference between them can be attributed to the table alone"
    )
    for flag in sorted(set(control) | set(treatment)):
        if flag in allowed:
            continue
        if flag not in treatment:
            control_only.append(flag)
            problems.append(
                f"{flag} {control[flag]!r} is given for the control leg and "
                f"not for the treatment leg.  " + drifted.format(flag=flag)
            )
            continue
        if flag not in control:
            treatment_only.append(flag)
            problems.append(
                f"{flag} {treatment[flag]!r} is given for the treatment leg "
                f"and not for the control leg.  " + drifted.format(flag=flag)
            )
            continue
        if control[flag] != treatment[flag]:
            value_differs[flag] = [control[flag], treatment[flag]]
            problems.append(
                f"{flag} is {control[flag]!r} for the control leg and "
                f"{treatment[flag]!r} for the treatment leg.  "
                + drifted.format(flag=flag)
            )

    if problems:
        raise _refuse(
            f"the two legs are not one authored request run twice.  "
            f"{len(problems)} problem(s):\n  - "
            + "\n  - ".join(problems)
            + f"\nA paired run is readable only when the legs differ in "
            f"exactly {SOURCE_TABLE_FLAG} and the per-leg destinations "
            f"{', '.join(PER_LEG_FLAGS)}.  Author the forecast arguments ONCE "
            f"on the `woof hex pair` command line and let this door derive "
            f"the control."
        )

    return {
        "proved": True,
        "source_table_flag": SOURCE_TABLE_FLAG,
        "token_diff": {
            "treatment_only": [SOURCE_TABLE_FLAG, treatment[SOURCE_TABLE_FLAG]],
            "control_only": control_only,
            "value_differs": {
                flag: [per_leg[flag]["control"], per_leg[flag]["treatment"]]
                for flag in PER_LEG_FLAGS
                if per_leg[flag]
            },
        },
        "identical_flags": {
            flag: control[flag] for flag in sorted(control) if flag not in allowed
        },
    }


# ---------------------------------------------------------------------------
# the request
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LegPlan:
    """One leg: where it writes, what it is told, and how it is invoked."""

    name: str
    directory: Path
    out: Path
    scratch: Path
    log: Path
    #: The forecast door's own argument vector for this leg.
    argv: tuple[str, ...]
    #: The full command the default runner spawns.
    command: tuple[str, ...]


@dataclass(frozen=True)
class PairRequest:
    """One resolved paired request, with its identity already proved."""

    pair_out: Path
    source_table: Path
    #: The forecast vector as authored, before the per-leg destinations.
    authored_argv: tuple[str, ...]
    control_argv: tuple[str, ...]
    treatment_argv: tuple[str, ...]
    control_out: Path
    treatment_out: Path
    control_scratch: Path
    treatment_scratch: Path
    identity: Mapping[str, Any]
    #: ``--grid``, kept for the area weights when the frames carry none.
    grid: Path | None


def _leg_scratch(explicit: Path | None, leg: str, out: Path) -> Path:
    """This leg's kernel cache.

    With no ``--scratch`` the forecast door's own rule applies to the leg's
    own destination: a sibling of ``--out``, never inside it, so the cache is
    not published as product.  With one given, each leg gets its own
    subdirectory of it -- a shared cache would have the second leg load
    kernels the first leg compiled, and the forecast door refuses an existing
    cache directory anyway.
    """

    if explicit is None:
        return default_scratch(out)
    return Path(explicit).expanduser().absolute() / leg


def leg_plans(request: PairRequest) -> tuple[LegPlan, ...]:
    """Both legs, control first, ready to run."""

    plans: list[LegPlan] = []
    for name, argv, out, scratch in (
        (
            "control",
            request.control_argv,
            request.control_out,
            request.control_scratch,
        ),
        (
            "treatment",
            request.treatment_argv,
            request.treatment_out,
            request.treatment_scratch,
        ),
    ):
        directory = request.pair_out / name
        plans.append(
            LegPlan(
                name=name,
                directory=directory,
                out=out,
                scratch=scratch,
                log=directory / "forecast.log",
                argv=tuple(argv),
                command=tuple([sys.executable, "-m", "woof.hex", "forecast", *argv]),
            )
        )
    return tuple(plans)


def _require_fresh(pair_out: Path) -> None:
    if pair_out.exists() and not pair_out.is_dir():
        raise _refuse(
            f"--pair-out {pair_out} exists and is not a directory.  A pair "
            "writes two output trees, two logs, a manifest and a summary "
            "under it."
        )
    if not pair_out.parent.is_dir():
        raise _refuse(
            f"--pair-out {pair_out} cannot be created: its parent "
            f"{pair_out.parent} does not exist.  Create the parent first; "
            "this door refuses to build a deep path for a run that costs two "
            "forecasts, because a mistyped one then looks like a successful "
            "new directory."
        )
    if not pair_out.is_dir():
        return
    contents = sorted(entry.name for entry in pair_out.iterdir())
    if not contents:
        return
    shown = ", ".join(contents[:10])
    if len(contents) > 10:
        shown += f", ... ({len(contents)} entries in all)"
    raise _refuse(
        f"--pair-out {pair_out} is not empty; it holds {shown}.  A pair is "
        "read by differencing two trees, and a directory that already carries "
        "frames, logs or a manifest cannot say which run they came from -- a "
        "second pair landing beside a first one silently produces a summary "
        "built from a mixture of the two.  Give an unused path, or move the "
        "existing one aside."
    )


def resolve_pair_request(arguments: argparse.Namespace) -> PairRequest:
    """Every check this door can make before either leg costs anything."""

    for dest, why in OWNED_BY_THIS_DOOR.items():
        if getattr(arguments, dest, None) is not None:
            raise _refuse(why)

    table_argument = getattr(arguments, "source_table", None)
    if table_argument is None:
        raise _refuse(
            f"{SOURCE_TABLE_FLAG} was not given.  It is the ONE thing the two "
            "legs differ in, so without it this door would run the same "
            "forecast twice and report the difference between a run and "
            "itself as a treatment effect.  Pass the point-source table the "
            "treatment leg is to carry."
        )
    source_table = Path(table_argument).expanduser().absolute()
    if source_table.is_symlink():
        raise _refuse(
            f"{SOURCE_TABLE_FLAG} {source_table} is a symbolic link.  The "
            "manifest pins the table by SHA-256, and the bytes a manifest "
            "names must be the bytes that were read."
        )
    if not source_table.is_file():
        raise _refuse(
            f"{SOURCE_TABLE_FLAG} names a missing file: {source_table}.  The "
            "treatment leg is defined by the table it carries; there is no "
            "default and no empty table."
        )

    pair_argument = getattr(arguments, "pair_out", None)
    if pair_argument is None:
        raise _refuse(
            "--pair-out was not given.  A pair writes two output trees, two "
            "logs, a manifest and a summary, and this door will not scatter "
            "them into the current directory by default."
        )
    pair_out = Path(pair_argument).expanduser().absolute()
    _require_fresh(pair_out)

    authored = authored_forecast_argv(arguments)

    control_out = pair_out / "control" / "out"
    treatment_out = pair_out / "treatment" / "out"
    explicit_scratch = getattr(arguments, "scratch", None)
    control_scratch = _leg_scratch(explicit_scratch, "control", control_out)
    treatment_scratch = _leg_scratch(explicit_scratch, "treatment", treatment_out)

    control_argv = [
        *authored,
        "--out", str(control_out),
        "--scratch", str(control_scratch),
    ]
    treatment_argv = [
        *authored,
        "--out", str(treatment_out),
        "--scratch", str(treatment_scratch),
        SOURCE_TABLE_FLAG, str(source_table),
    ]
    from woof.output_formats import options
    _, store = options(arguments)
    if store is not None:
        for leg, argv in (("control", control_argv), ("treatment", treatment_argv)):
            argv.extend(("--icechunk-store", str(store).rstrip("/") + "/" + leg))
    identity = assert_pair_identity(control_argv, treatment_argv)

    grid = getattr(arguments, "grid", None)
    return PairRequest(
        pair_out=pair_out,
        source_table=source_table,
        authored_argv=tuple(authored),
        control_argv=tuple(control_argv),
        treatment_argv=tuple(treatment_argv),
        control_out=control_out,
        treatment_out=treatment_out,
        control_scratch=control_scratch,
        treatment_scratch=treatment_scratch,
        identity=identity,
        grid=Path(grid).expanduser().absolute() if grid is not None else None,
    )


# ---------------------------------------------------------------------------
# running the legs
# ---------------------------------------------------------------------------
def forecast_subprocess_runner(plan: LegPlan) -> int:
    """Run one leg through the forecast door, logging to ``<leg>/forecast.log``.

    A subprocess rather than an in-process call, deliberately: the forecast
    door loads drivers by path, compiles kernels and holds device memory, and
    a second leg in the same interpreter would inherit whatever the first one
    left behind.  Two processes make "the legs did not contaminate each
    other" a property of the operating system rather than of a cleanup
    routine somebody has to keep correct.
    """

    plan.log.parent.mkdir(parents=True, exist_ok=True)
    with plan.log.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(" ".join(plan.command) + "\n")
        handle.flush()
        completed = subprocess.run(
            list(plan.command), stdout=handle, stderr=subprocess.STDOUT
        )
    return int(completed.returncode)


def extra_files(out: Path) -> list[Path]:
    """Accounting files a leg dropped beside its history, carried unread."""

    found: set[Path] = set()
    for pattern in EXTRA_FILE_PATTERNS:
        for path in Path(out).rglob(pattern):
            if path.is_file():
                found.add(path.resolve())
    return sorted(found)


def collect_leg(plan: LegPlan, *, rc: int, seconds: float) -> dict[str, Any]:
    """What one leg left behind, with a digest against every file named."""

    receipt = plan.out / "forecast-receipt.json"
    return {
        "name": plan.name,
        "argv": list(plan.argv),
        "command": list(plan.command),
        "out": str(plan.out),
        "scratch": str(plan.scratch),
        "log": str(plan.log),
        "rc": int(rc),
        "wall_seconds": round(float(seconds), 3),
        "receipt": str(receipt) if receipt.is_file() else None,
        "receipt_sha256": sha256_file(receipt) if receipt.is_file() else None,
        "history": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in history_files(plan.out)
        ],
        "extra_files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in extra_files(plan.out)
        ],
    }


def run_pair(
    request: PairRequest,
    *,
    runner: Callable[[LegPlan], int] | None = None,
) -> dict[str, Any]:
    """Run both legs, then write the manifest and the summary.

    ``runner`` is the seam.  The default drives the forecast door as a
    subprocess; a test substitutes a runner that writes small frames, so
    everything this door decides -- the derivation, the identity proof, the
    freshness refusal, the collection and the arithmetic in the summary -- is
    exercised without a card and without an hour of integration.
    """

    from . import __version__

    if runner is None:
        runner = forecast_subprocess_runner

    request.pair_out.mkdir(parents=True, exist_ok=True)
    legs: dict[str, Any] = {}
    started_all = time.monotonic()
    for plan in leg_plans(request):
        plan.directory.mkdir(parents=True, exist_ok=True)
        # The leg's kernel-cache scratch is this door's derivation.  The
        # forecast door refuses a scratch that EXISTS (a stale kernel cache
        # must never be loaded) and its driver creates the leaf itself with
        # no parents, so what this door owes is the PARENT: a missing parent
        # refused after the mesh bind, 26 s into the first card run.
        plan.scratch.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        rc = int(runner(plan))
        seconds = time.monotonic() - started
        legs[plan.name] = collect_leg(plan, rc=rc, seconds=seconds)
        print(
            f"LEG {plan.name} rc={rc} {seconds:.2f}s "
            f"{len(legs[plan.name]['history'])} frame(s) {plan.out}",
            flush=True,
        )
        if rc != 0:
            raise _refuse(
                f"the {plan.name} leg exited {rc}.  A pair is two legs of one "
                f"experiment: with the {plan.name} leg unfinished there is no "
                f"comparison to make, and running the remaining leg would "
                f"spend a whole forecast producing a tree nothing can be "
                f"differenced against.  The leg's own output is at "
                f"{plan.log}, and it names what it could not do."
            )

    # Re-proved against the vectors that WERE RUN rather than the ones that
    # were planned.  The two are the same object today; the check costs
    # nothing and stops being free the day a runner is allowed to edit a leg.
    identity = assert_pair_identity(
        legs["control"]["argv"], legs["treatment"]["argv"]
    )

    summary_path = request.pair_out / SUMMARY_NAME
    weighting = write_summary(request, legs, summary_path)

    manifest = {
        "schema": PAIR_SCHEMA,
        "tool": "woof hex pair",
        "door_version": __version__,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "host": host_record(),
        "pair_out": str(request.pair_out),
        "authored_argv": list(request.authored_argv),
        "source_table": {
            "path": str(request.source_table),
            "sha256": sha256_file(request.source_table),
        },
        "identity": dict(identity),
        "legs": {name: legs[name] for name in LEG_NAMES},
        "leg_order": list(LEG_NAMES),
        "summary": str(summary_path),
        "area_weighting": weighting,
        "result_sentence": RESULT_SENTENCE,
        "pair_seconds": round(time.monotonic() - started_all, 3),
    }
    manifest_path = request.pair_out / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"MANIFEST {manifest_path}", flush=True)
    print(f"SUMMARY {summary_path}", flush=True)
    if weighting == NO_FRAMES_WEIGHTING:
        # THE BREAKAGE THIS PREVENTS (hex parity run group 7, #27): a pair whose legs left no frames this door
        # can read printed "NO COMPARISON" and still exited 0, so a caller counted an empty experiment as done.
        empty = [name for name in LEG_NAMES if not legs[name]["history"]]
        raise _refuse(
            f"NO COMPARISON: the {' and '.join(empty)} leg(s) exited 0 but left no history frame in "
            f"{', '.join(str(legs[name]['out']) for name in empty)}, so there is nothing to difference.  "
            f"The manifest ({manifest_path}) and the legs' logs say what each leg wrote."
        )
    return manifest


# ---------------------------------------------------------------------------
# the summary
# ---------------------------------------------------------------------------
def _open_dataset(path: Path):
    from netCDF4 import Dataset

    dataset = Dataset(str(path), "r")
    dataset.set_auto_mask(False)
    return dataset


def _area_weights(control: Any, treatment: Any, grid: Path | None):
    """Cell areas and the sentence saying where they came from.

    Order: the frames' own ``areaCell`` first, because a weight read out of
    the file being summarised cannot belong to a different mesh; then
    ``--grid``, which is the mesh the run was bound to; then nothing -- and
    the summary SAYS SO rather than presenting an unweighted mean as an area
    mean.
    """

    import numpy

    for dataset, where in ((treatment, "treatment"), (control, "control")):
        if "areaCell" in dataset.variables:
            values = numpy.asarray(dataset.variables["areaCell"][...], dtype=float)
            return values, (
                f"area-weighted by areaCell carried in the {where} leg's own "
                f"frames"
            )
    if grid is not None and Path(grid).is_file():
        with _open_dataset(Path(grid)) as dataset:
            if "areaCell" in dataset.variables:
                values = numpy.asarray(
                    dataset.variables["areaCell"][...], dtype=float
                )
                return values, f"area-weighted by areaCell from --grid {grid}"
    return None, (
        "UNWEIGHTED: the frames carry no areaCell and no --grid was given, so "
        "every mean below is a plain average over cells and levels.  On a "
        "variable-resolution mesh that over-weights the refined region"
    )


def _cell_major(values: Any, dims: Sequence[str], record: int) -> Any:
    """One record, reshaped to ``(nCells, ...)`` however the file ordered it."""

    import numpy

    axes = list(dims)
    array = values
    if axes and axes[0] == "Time":
        array = array[record]
        axes = axes[1:]
    return numpy.moveaxis(
        numpy.asarray(array, dtype=float), axes.index("nCells"), 0
    )


def _rank(dims: Sequence[str]) -> int:
    """Field rank counting cells and levels, ignoring ``Time``."""

    return len([name for name in dims if name != "Time"]) + 1


def _comparable(dataset: Any, name: str) -> bool:
    """Is this a per-cell floating-point field this door reports on?

    Mesh geometry and connectivity are excluded by name: they are identical
    in both legs by construction -- the legs bind the same mesh -- and a row
    of zeroes for every one of them would bury the fields a reader came for.
    """

    import numpy

    from .output import HISTORY_MESH_VARIABLES

    if name in HISTORY_MESH_VARIABLES:
        return False
    variable = dataset.variables[name]
    if "nCells" not in variable.dimensions:
        return False
    if not numpy.issubdtype(numpy.dtype(variable.dtype), numpy.floating):
        return False
    return _rank(variable.dimensions) in (2, 3)


def _mean_and_max(array: Any, weights: Any) -> tuple[float, float]:
    import numpy

    flat = array.reshape(array.shape[0], -1)
    if weights is None:
        return float(flat.mean()), float(flat.max())
    column = numpy.asarray(weights, dtype=float).reshape(-1, 1)
    total = float(column.sum()) * flat.shape[1]
    return float((flat * column).sum() / total), float(flat.max())


def _cells_above_zero(array: Any) -> tuple[int, int]:
    """``(cells carrying any value above zero, cells)``.

    Counted per CELL rather than per value, so the number means the same
    thing for a two-dimensional accumulator and for a three-dimensional
    scalar: how much of the domain the treatment leg touched at all.
    """

    import numpy

    flat = array.reshape(array.shape[0], -1)
    return int(numpy.count_nonzero((flat > 0.0).any(axis=1))), int(flat.shape[0])


def _number(value: float) -> str:
    return f"{value:+.6e}"


def write_summary(
    request: PairRequest, legs: Mapping[str, Any], destination: Path
) -> str:
    """Difference the matched frames and write ``summary.txt``.

    Returns the area-weighting sentence, which the manifest also records so
    that a reader of the manifest alone knows what the means in the summary
    are means of.
    """

    control_history = [Path(row["path"]) for row in legs["control"]["history"]]
    treatment_history = [Path(row["path"]) for row in legs["treatment"]["history"]]

    lines: list[str] = [RESULT_SENTENCE, ""]
    lines += [
        f"pair               {request.pair_out}",
        f"control out        {request.control_out}",
        f"treatment out      {request.treatment_out}",
        f"point-source table {request.source_table}",
        f"                   sha256 {sha256_file(request.source_table)}",
        f"frames             control {len(control_history)}, "
        f"treatment {len(treatment_history)}",
        "",
    ]

    if not control_history or not treatment_history:
        lines.append(
            "NO COMPARISON: one leg wrote no history frames, so there is "
            "nothing to difference.  Read the legs' own logs."
        )
        destination.write_text(
            "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
        )
        return NO_FRAMES_WEIGHTING

    control_names = [path.name for path in control_history]
    treatment_names = [path.name for path in treatment_history]
    if control_names != treatment_names:
        raise _refuse(
            "the two legs wrote different history frames: control "
            f"{control_names} against treatment {treatment_names}.  A "
            "difference is meaningful only between frames at the same valid "
            "time, and pairing frames by position across two different "
            "schedules would subtract one hour from another and report the "
            "result as a treatment effect."
        )

    weighting = ""
    body: list[str] = []
    for index, (control_path, treatment_path) in enumerate(
        zip(control_history, treatment_history), start=1
    ):
        control = _open_dataset(control_path)
        treatment = _open_dataset(treatment_path)
        try:
            weights, weighting = _area_weights(control, treatment, request.grid)
            records = (
                int(treatment.dimensions["Time"].size)
                if "Time" in treatment.dimensions
                else 1
            )
            control_records = (
                int(control.dimensions["Time"].size)
                if "Time" in control.dimensions
                else 1
            )
            if records != control_records:
                raise _refuse(
                    f"{treatment_path.name} holds {records} time record(s) in "
                    f"the treatment leg and {control_records} in the control "
                    f"leg.  Records are matched by position, and two files "
                    f"with different record counts cannot be matched that way."
                )
            shared = sorted(
                name
                for name in treatment.variables
                if name in control.variables
                and _comparable(treatment, name)
                and _rank(treatment.variables[name].dimensions) == 3
            )
            only = sorted(
                name
                for name in treatment.variables
                if name not in control.variables and _comparable(treatment, name)
            )
            for record in range(records):
                body.append(
                    f"frame {index}/{len(treatment_history)} "
                    f"{treatment_path.name} record {record}"
                )
                body.append(
                    "  shared 3-D scalars, treatment minus control"
                    + ("" if shared else "   (none)")
                )
                for name in shared:
                    difference = _cell_major(
                        treatment.variables[name][...],
                        treatment.variables[name].dimensions,
                        record,
                    ) - _cell_major(
                        control.variables[name][...],
                        control.variables[name].dimensions,
                        record,
                    )
                    mean, peak = _mean_and_max(difference, weights)
                    body.append(
                        f"    {name:<28} mean {_number(mean)}  "
                        f"max {_number(peak)}"
                    )
                body.append(
                    "  declared extra scalars, present in the treatment leg only"
                    + ("" if only else "   (none)")
                )
                for name in only:
                    array = _cell_major(
                        treatment.variables[name][...],
                        treatment.variables[name].dimensions,
                        record,
                    )
                    mean, peak = _mean_and_max(array, weights)
                    above, cells = _cells_above_zero(array)
                    body.append(
                        f"    {name:<28} mean {_number(mean)}  "
                        f"max {_number(peak)}  "
                        f"cells above zero {above} of {cells}"
                    )
                body.append("")
        finally:
            control.close()
            treatment.close()

    lines.append(f"weighting          {weighting}")
    lines.append("")
    lines += body
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return weighting


# ---------------------------------------------------------------------------
# the handler and the argument surface
# ---------------------------------------------------------------------------
def run_pair_door(arguments: argparse.Namespace) -> int:
    run_pair(resolve_pair_request(arguments))
    print(RESULT_SENTENCE, flush=True)
    return 0


def add_pair_arguments(parser: argparse.ArgumentParser) -> None:
    """Every forecast flag, then the two this door owns.

    ``add_forecast_arguments`` is CALLED rather than mirrored so the two doors
    cannot drift: a flag added to the forecast door is a flag a pair carries
    the same day, and one removed from it disappears from here too.
    """

    add_forecast_arguments(parser)
    # --source-table is the forecast door's own flag now (it is what the
    # treatment leg passes through); here it is REQUIRED and it is the ONE
    # thing the two legs differ in -- the control leg is this same command
    # with the table absent.  resolve_pair_request refuses its absence.
    parser.add_argument(
        "--pair-out", type=Path, default=None, metavar="DIR",
        help="fresh directory for both legs, the manifest and the summary.  "
             "The legs land at <pair-out>/control/out and "
             "<pair-out>/treatment/out; a non-empty directory is refused by "
             "name, because a summary built from a mixture of two pairs "
             "cannot say which run it read")


def add_pair_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "pair",
        help="run one authored forecast twice -- with and without a "
             "point-source table -- and difference the two",
        description=(
            "Run ONE authored forecast as a matched pair: a control leg, and "
            "a treatment leg that is the same request plus --source-table.  "
            "The control is DERIVED from the treatment rather than authored "
            "separately, and the door re-proves that the two argument vectors "
            "differ in exactly the source table and the per-leg destinations "
            "before either leg starts.  Both legs run through the forecast "
            "door.  The output is a manifest pinning every file by SHA-256 "
            "and a summary differencing the matched frames -- and a "
            "difference between the legs is a MODEL RESULT, not evidence "
            "about the atmosphere."
        ),
    )
    add_pair_arguments(parser)
    parser.set_defaults(handler=run_pair_door)
