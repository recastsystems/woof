"""One run, one folder -- the level above the render layout.

The defect this closes was reported by a collaborator running his own
pipeline: he runs the same configuration repeatedly, and every run wrote
into the same directory as the last one.  Renders were the visible half
(a second render drops its PNGs among the first's, and the only way to
tell them apart is a file mtime), but the run artifacts were the
dangerous half -- ``wrfout``, ``report.json``, ``progress.jsonl``, the
stage receipts.  Two runs interleaved in one tree publish receipts that
describe neither, and the create-only refusals that stand in front of
that turn "run it again" into "invent a new directory name, by hand,
every time".

So a run-stamped level is inserted at the CASE ROOT, above everything a
run writes::

    <out>/<run-stamp>/<domain>/<product>/<valid-day>/<file>.png
    <out>/<run-stamp>/run/wrfout/...
    <out>/<run-stamp>/run/report.json
    <out>/<run-stamp>/prepared/, authority/

It EXTENDS WOOF's 2026-08-06 render ruling rather than replacing any
part of it: ``<domain>/<product>/<valid-day>`` survives byte for byte
underneath, in the same order, spelled by the same module
(:mod:`woof.render_layout`).  Nothing here inverts that tree back to
flat and nothing here reorders it.

**The stamp, exactly.**

::

    run-<YYYYMMDD>-<HHMMSS>Z[_i<YYYYMMDD><HHMM>Z][-<NN>]
        \\_______ launch _______/  \\____ init ____/  \\ordinal/

``launch``
    The wall-clock instant the front door was invoked, in UTC.  It leads
    because it is the field that makes two runs of ONE configuration
    different, and putting it first makes a plain directory listing sort
    into run order.  Seconds, not minutes: two back-to-back runs of a
    short configuration land inside the same minute, which is exactly
    the case this exists for.

``init``
    The model's initialisation time -- the cycle a forecast is launched
    from -- in UTC, to the minute.  It is what a forecaster looking at
    two folders actually wants to know, and it is the field that keeps
    two DIFFERENT cycles apart in the listing even when they were
    launched a second apart.

    It is present when the front door holds that fact and ABSENT when it
    does not.  ``woof go`` reads it from the config's ``[fetch] cycle``;
    ``woof sim`` reads it from the prepared bundle's own document;
    ``woof render`` reads ``SIMULATION_START_DATE`` out of the first
    wrfout that carries it.  A door that can read none of those omits
    the component rather than stamping a guess -- a wrong init time in a
    folder name is worse than a missing one, because a reader believes
    it.

``ordinal``
    ``-02``, ``-03`` ... appended ONLY when the directory the first two
    fields name already exists.  Two runs launched inside one second
    would otherwise share a stamp, and the whole point is that they
    cannot share a tree.  The allocation is a create-exclusive
    ``mkdir``, so two processes racing for the same second get different
    folders rather than one folder and a corrupted pair of receipts.

Everything but the ordinal is a pure function of ``(launch, init)``, so
a caller that knows both can compute the folder name before the run
starts -- the same predictability contract the render layout keeps.

**What is NOT stamped.**  The download directory.  ``woof go`` puts it
at ``<out>/data`` and leaves it there across runs on purpose: it holds
INPUTS, it is meant to be reused, and stamping it would re-download
gigabytes of identical GRIB on every re-run.  Artifacts are stamped,
inputs are cached.

**It is the default.**  ``--run-stamp off`` restores the unstamped tree
for a consumer written against it; that is a documented workaround, not
a supported alternative, and each front door says so in ``--help``.

**A path that is already INSIDE a run folder is honoured verbatim.**
Point ``--out``/``--outdir`` at an existing ``run-...`` directory, or at
anything beneath one, and no second stamp level is inserted -- that
path already belongs to a run, and the run owns exactly one folder.
The ancestor half of that is not a nicety: ``woof go`` claims
``<case>/run-.../`` and then hands its render stage
``<case>/run-.../png``, so a render door that asked only "is my own
name a stamp?" allocated a SECOND stamp inside the first, and the two
renders of one run -- the early one on the first committed frame and
the finalize one at the end -- landed in two different ``run-...``
folders minutes apart.  A run folder is the run's identity, never the
calling stage's clock.  What happens next is each door's own business
and each one states it: ``woof go`` and ``woof sim`` REFUSE when the
folder already holds a run, because their stages are create-only and
merging two runs into one tree publishes receipts describing neither;
``woof render`` RESUMES, drawing into the tree that is there, because
rendering more products into an existing run's folder is a workflow and
rendering is idempotent per filename.
"""

from __future__ import annotations

import datetime
from contextlib import contextmanager
import os
import re
from pathlib import Path
import tempfile

#: Every run folder begins with this.  One word, so a reader scanning a
#: case directory can tell a run from ``data`` at a glance.
RUN_PREFIX = "run-"

#: ``strftime`` for the launch half of the stamp.
LAUNCH_FORMAT = "%Y%m%d-%H%M%S"

#: ``strftime`` for the init half.  Minutes, because a config may start
#: at a half hour and an hour-resolution stamp would collapse two.
INIT_FORMAT = "%Y%m%d%H%M"

#: The whole grammar, as one expression.  THE definition of "is this a
#: run folder": every door asks this and none of them re-spells it.
STAMP_RE = re.compile(
    r"^run-(?P<launch>\d{8}-\d{6})Z(?:_i(?P<init>\d{12})Z)?"
    r"(?:-(?P<ordinal>\d{2,}))?$")

#: How many same-second collisions are resolved before giving up.  A
#: bound rather than an unbounded loop: past this, something is wrong
#: with the clock or the filesystem and a run should say so rather than
#: spin.
MAX_ORDINAL = 99

#: The pointer a script reads instead of globbing.  One line, the NAME
#: of the newest run folder under this root, rewritten when a run's
#: folder is PUBLISHED -- at allocation for ``go``/``sim``, whose
#: stages write receipts from the first moment, and only after the
#: first PNG lands for ``render``, whose folder holds nothing until
#: something is drawn (UX finding N5).  A name and not a path so the
#: file survives the tree being moved.
LATEST_POINTER = "latest-run.txt"

#: What ``--run-stamp`` is when nobody says otherwise.
DEFAULT_RUN_STAMP = True

#: The ``--run-stamp`` vocabulary.
RUN_STAMP_CHOICES = ("on", "off")

#: Keys a preparation document may carry its initialisation time under,
#: in the order they are consulted.  All are ISO-8601 instants written by
#: this tree's own runners; the list is ordered most-specific first so a
#: document carrying several is read at its most exact.
BUNDLE_INIT_KEYS = ("initial_valid_time", "model_start_time", "start_time",
                    "valid_time", "source_cycle", "cycle")


class RunStampError(RuntimeError):
    """A run folder could not be allocated."""


def utcnow() -> datetime.datetime:
    """The launch instant, UTC and timezone-aware.

    One place, so a test can monkeypatch the clock and every door moves
    together.
    """

    return datetime.datetime.now(datetime.timezone.utc)


def _as_utc(when: datetime.datetime) -> datetime.datetime:
    """``when`` in UTC.  A naive value is READ as UTC, not localised.

    Naive datetimes reach this from config tables and proof documents,
    where every timestamp in this tree is already UTC by contract.
    Localising one would shift a cycle by the box's offset and stamp a
    folder with a time the run never had.
    """

    if when.tzinfo is None:
        return when.replace(tzinfo=datetime.timezone.utc)
    return when.astimezone(datetime.timezone.utc)


def parse_time(value) -> datetime.datetime | None:
    """A ``datetime`` from any init-time spelling this tree writes, or None.

    ``None`` is not an error: it means the caller holds no readable
    initialisation time and the stamp must omit that component.  The
    spellings are the ones the front doors actually hand over --
    ``2026-05-17T18`` from a wizard config's ``[fetch] cycle``,
    ``2026-05-17T18:00:00Z`` from a proof document,
    ``2026-05-17_18:00:00`` from a WRF ``Times``/``SIMULATION_START_DATE``
    record -- plus a ``datetime`` passed straight through.
    """

    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return _as_utc(value)
    if isinstance(value, datetime.date):
        return datetime.datetime(value.year, value.month, value.day,
                                 tzinfo=datetime.timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    # WRF's own stamp separates date from time with an underscore and
    # `fromisoformat` does not read that; the trailing Z is ISO but only
    # from 3.11, and this tree still states its floor at 3.10.
    text = text.replace("_", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    for candidate in (text, f"{text}:00:00", f"{text}:00"):
        try:
            return _as_utc(datetime.datetime.fromisoformat(candidate))
        except ValueError:
            continue
    return None


def format_stamp(*, launch: datetime.datetime,
                 init: datetime.datetime | None = None,
                 ordinal: int = 1) -> str:
    """The folder name for one run.  Pure, given its three arguments.

    ``ordinal`` 1 is the unsuffixed name; 2 and up append ``-NN``.  It is
    a parameter rather than a private detail of :func:`allocate` so a
    caller can predict the collision spelling too.
    """

    when = _as_utc(launch)
    name = RUN_PREFIX + (f"{when.year:04d}{when.month:02d}{when.day:02d}-"
                         f"{when.hour:02d}{when.minute:02d}{when.second:02d}Z")
    if init is not None:
        start = _as_utc(init)
        name += (f"_i{start.year:04d}{start.month:02d}{start.day:02d}"
                 f"{start.hour:02d}{start.minute:02d}Z")
    if ordinal > 1:
        name += f"-{ordinal:02d}"
    return name


def parse_stamp(name) -> dict | None:
    """``{launch, init, ordinal}`` for a run folder name, or None.

    ``None`` means the name is not one of ours, which is the same answer
    :func:`is_run_folder` gives -- one grammar, read once.
    """

    match = STAMP_RE.match(Path(str(name)).name)
    if match is None:
        return None
    try:
        launch = datetime.datetime.strptime(
            match.group("launch"), LAUNCH_FORMAT)
    except ValueError:
        # Syntactically a stamp, but naming an instant that does not
        # exist.  It is still one of our folders -- it was written by
        # this grammar -- so the shape is reported and the impossible
        # field is not.
        launch = None
    init = None
    if match.group("init"):
        try:
            init = _as_utc(datetime.datetime.strptime(
                match.group("init"), INIT_FORMAT))
        except ValueError:
            init = None
    return {
        "launch": _as_utc(launch) if launch is not None else None,
        "init": init,
        "ordinal": int(match.group("ordinal") or 1),
    }


def is_run_folder(path) -> bool:
    """True when ``path``'s own name is a run stamp.

    Asked of the NAME, not of the contents: a caller that points a front
    door at ``.../run-20260817-041233Z`` has named a run folder whether
    or not anything has been written into it yet, and a door that
    inspected the contents would answer differently before and after the
    first stage.
    """

    return STAMP_RE.match(Path(str(path)).name) is not None


def owning_run(path) -> Path | None:
    """The run folder ``path`` already belongs to, or ``None``.

    ``path`` itself counts, so this is :func:`is_run_folder` widened by
    one question: *is a run's identity already established at or above
    this path?*  A door asking only about its own name got that question
    wrong in the one place it matters most.  ``woof go`` claims
    ``<case>/run-.../`` and hands its render stage
    ``<case>/run-.../png``; ``png`` is not a stamp, so the render door
    allocated a second stamp INSIDE the first and produced
    ``<case>/run-.../png/run-.../``.  Two renders of one run -- the
    early render of the first committed frame and the finalize render
    at the end -- then filed into two different ``run-...`` folders,
    minutes apart, and the byte-identity the finalize skip is licensed
    by could not even be stated, let alone checked.

    Asked of the NAME chain, not of any directory's contents, for the
    same reason :func:`is_run_folder` is: the answer must not change
    between the moment a door plans a path and the moment it writes it.
    """

    candidate = Path(str(path))
    for node in (candidate, *candidate.parents):
        if is_run_folder(node):
            return node
    return None


def stage_flags() -> list[str]:
    """``--run-stamp`` argv for a sub-command INSIDE an existing run.

    ``woof go`` claims its run folder once, at the top, and every stage
    it spawns writes into that one tree.  The render stage is spawned
    twice -- once on the first committed frame
    (:mod:`woof.first_products`) and once at finalize -- and a stage
    that stamped for itself would allocate a fresh wall-clock folder on
    each of those calls: the same run's pictures split across two
    ``run-...`` directories named seconds or minutes apart, with the
    finalize stage unable to prove the early render's frame was already
    drawn.

    So the stage says out loud that the run folder is claimed above it.
    ``off`` here is not the front-door workaround wearing a disguise:
    the workaround is a READER choosing an unstamped case tree, while
    this is a stage declining to stamp a second time inside a tree that
    is already stamped.  Spelled once, here, so the two call sites in
    :func:`woof.go_cli.render_command` cannot drift apart.
    """

    return ["--run-stamp", "off"]


@contextmanager
def _latest_lock(root: Path):
    """Serialize pointer selection on one permanent lock inode."""
    with (root / ".latest-run.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def record_latest(root, run_dir) -> None:
    """Atomically publish the newest launch among completed publications.

    Best effort on purpose: a read-only case root is a reason not to
    have a pointer, never a reason to fail a run that is otherwise
    fine.
    """

    temporary = None
    try:
        root = Path(root)
        candidate = Path(run_dir).name
        if not is_run_folder(candidate) or not (root / candidate).is_dir():
            return
        with _latest_lock(root):
            pointer = root / LATEST_POINTER
            try:
                previous = pointer.read_text(encoding="utf-8").strip()
            except OSError:
                previous = ""
            if (is_run_folder(previous) and (root / previous).is_dir()
                    and previous >= candidate):
                return
            with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=root,
                    prefix=".latest-run-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(candidate + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, pointer)
            temporary = None
    except OSError:
        pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def latest(root) -> Path | None:
    """The newest run folder under ``root``, or None if there is none.

    The pointer first, the listing as the fallback -- so a tree whose
    pointer was never written (a run from before this landed, a
    read-only root) still answers, and a tree whose pointer names a
    folder that has since been moved away does not lie about it.

    The listing fallback sorts by NAME, which is run order: the launch
    field leads the stamp precisely so that this is true.
    """

    root = Path(root)
    pointer = root / LATEST_POINTER
    try:
        named = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        named = ""
    if named and is_run_folder(named) and (root / named).is_dir():
        return root / named
    if not root.is_dir():
        return None
    found = sorted((child for child in root.iterdir()
                    if child.is_dir() and is_run_folder(child)),
                   key=lambda child: child.name)
    return found[-1] if found else None


def allocate(root, *, init=None, launch=None, publish: bool = True) -> Path:
    """Create and return a fresh run folder under ``root``.

    Create-EXCLUSIVE: the directory is made with ``mkdir`` and no
    ``exist_ok``, and a collision bumps the ordinal.  Two processes
    launching inside one second therefore get two folders; a check-then-
    create would give them one folder and two runs' receipts in it,
    which is the defect this module exists to prevent, reintroduced at
    the one moment it matters.

    ``publish=False`` allocates the folder WITHOUT moving the
    ``latest-run.txt`` pointer; the caller then publishes with
    :func:`record_latest` once the run has produced something.  This is
    ``woof render``'s arm: a failed render used to point the pointer at
    an empty stamped folder, so a script following the documented
    pointer could not tell a failed render from a successful empty one
    (UX finding N5).  ``go``/``sim`` keep the default -- their stages
    write receipts into the folder from the first moment, so for them
    allocation IS production.
    """

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    when = utcnow() if launch is None else launch
    start = parse_time(init)
    for ordinal in range(1, MAX_ORDINAL + 1):
        candidate = root / format_stamp(launch=when, init=start,
                                        ordinal=ordinal)
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        if publish:
            record_latest(root, candidate)
        return candidate
    raise RunStampError(
        f"{MAX_ORDINAL} run folders already exist under {root} for the "
        f"same launch second, so a distinct one cannot be named.  This "
        f"means runs are being started faster than the stamp can "
        f"separate them, and two runs sharing a tree would publish "
        f"receipts describing neither.\n"
        f"  remedy: give each run its own --out, or run them apart in "
        f"time")


def resolve(root, *, init=None, launch=None, enabled: bool = DEFAULT_RUN_STAMP,
            create: bool = True, publish: bool = True) -> Path:
    """Where this run's artifacts go, under the caller's ``root``.

    THE single answer every front door asks, and the reason there is
    only one: three doors deciding independently whether to stamp is
    three chances for ``woof go``'s render stage to write somewhere
    ``woof render`` would not look.

    Three outcomes, all of them stated rather than inferred:

    * ``enabled=False`` -- ``root`` itself, unchanged.  The documented
      workaround.
    * ``root`` is already inside a run folder (:func:`owning_run`) --
      ``root`` itself, unchanged.  The run's identity is established at
      or above this path; a second stamp level would bury the caller's
      own path and would give two stages of one run two folders.
    * otherwise -- a fresh ``root/run-...`` folder.
    """

    root = Path(root)
    if not enabled or owning_run(root) is not None:
        if create:
            root.mkdir(parents=True, exist_ok=True)
        return root
    if not create:
        # The name allocate() would take right now: the first ordinal no
        # folder holds yet.  Always naming ordinal 1 predicted a folder
        # an earlier run of the same second already owned, and the claim
        # then took the next ordinal under a name nobody had announced.
        when = utcnow() if launch is None else launch
        start = parse_time(init)
        for ordinal in range(1, MAX_ORDINAL + 1):
            candidate = root / format_stamp(launch=when, init=start,
                                            ordinal=ordinal)
            if not candidate.exists():
                return candidate
        return root / format_stamp(launch=when, init=start)
    return allocate(root, init=init, launch=launch, publish=publish)


def bundle_init(payload) -> datetime.datetime | None:
    """The initialisation time a preparation document carries, or None.

    Reads :data:`BUNDLE_INIT_KEYS` in order and returns the first that
    parses.  Nothing is computed and nothing is defaulted: a bundle that
    records no initialisation time yields ``None``, and the stamp then
    carries no init component.
    """

    if not isinstance(payload, dict):
        return None
    for key in BUNDLE_INIT_KEYS:
        parsed = parse_time(payload.get(key))
        if parsed is not None:
            return parsed
    return None


def wrfout_init(path) -> datetime.datetime | None:
    """``SIMULATION_START_DATE`` from one wrfout, or None.

    The model's own record of when it was initialised, read from the
    file rather than inferred from its name -- a wrfout is named for the
    valid time of its FIRST FRAME, which is the initialisation only for
    the first file of a run.

    Read through the Rust NetCDF route (:mod:`woof.netcdf_bridge`), the
    metadata pass only.  It read through netCDF4 until D-10 (2.8.8
    acceptance): that import switches a free-threaded interpreter's lock
    back on for the rest of the process, and NetCDF reads belong in Rust.

    Every failure is ``None``: no decoder, an unreadable file, an
    attribute that is not a timestamp.  A render whose input proves no
    initialisation time gets a launch-only stamp, which is still a
    distinct folder per run.
    """

    try:
        from woof.netcdf_bridge import open_dataset

        with open_dataset(path) as dataset:
            if "SIMULATION_START_DATE" not in dataset.ncattrs():
                return None
            raw = dataset.getncattr("SIMULATION_START_DATE")
    except Exception:
        return None
    return parse_time(raw)


def first_wrfout_init(paths) -> datetime.datetime | None:
    """The first readable ``SIMULATION_START_DATE`` across ``paths``.

    First rather than earliest: every file of one run records the same
    initialisation, so the first that answers is the answer, and reading
    the rest would open a whole render's worth of NetCDF to learn
    nothing new.
    """

    for path in paths or ():
        found = wrfout_init(path)
        if found is not None:
            return found
    return None


def run_stamp_enabled(args, attribute: str = "run_stamp") -> bool:
    """``--run-stamp`` off a parsed namespace, defaulting to ON.

    Read through one function so a door that forgets to declare the flag
    gets the DEFAULT behaviour rather than the workaround -- the failure
    mode of a bare ``getattr(args, "run_stamp", False)`` is a silent
    regression to the layout this module replaces.
    """

    value = getattr(args, attribute, None)
    if value is None:
        return DEFAULT_RUN_STAMP
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() != "off"


def add_argument(parser, *, artifacts: str = "output",
                 option: str = "--out") -> None:
    """Declare ``--run-stamp`` on one front door, spelled once.

    ``artifacts`` names what this door writes and ``option`` is that
    door's own output flag, so the help line reads in the reader's
    vocabulary without any door re-describing the scheme.
    """

    parser.add_argument(
        "--run-stamp", choices=RUN_STAMP_CHOICES,
        default="on" if DEFAULT_RUN_STAMP else "off",
        dest="run_stamp",
        help=f"put this run's {artifacts} in its own timestamped folder "
             f"under {option} (default on): {describe(option)}.  "
             f"Successive runs of one configuration then never "
             f"overwrite or interleave each other.  'off' writes "
             f"straight into {option}, which is what releases up to "
             f"2.4.1 did; it is kept only for a consumer still written "
             f"against that and is a workaround, not a supported "
             f"alternative")


def describe(root: str = "<--out>") -> str:
    """The one sentence a script author needs, for --help and docs."""

    return (f"{root}/run-<YYYYMMDD>-<HHMMSS>Z_i<YYYYMMDD><HHMM>Z/ "
            f"(launch instant UTC, then the model initialisation time; "
            f"the _i part is omitted when the run's init time cannot be "
            f"read)")


def relative_to_case(run_dir, root) -> str:
    """``run_dir`` said relative to ``root`` when it is under it.

    For the one line a front door prints: a reader wants the folder
    NAME, not the absolute path they just typed with one component
    added.
    """

    run_dir, root = Path(run_dir), Path(root)
    try:
        return os.fspath(run_dir.relative_to(root))
    except ValueError:
        return os.fspath(run_dir)


__all__ = [
    "BUNDLE_INIT_KEYS", "DEFAULT_RUN_STAMP", "INIT_FORMAT",
    "LATEST_POINTER", "LAUNCH_FORMAT", "MAX_ORDINAL", "RUN_PREFIX",
    "RUN_STAMP_CHOICES", "RunStampError", "STAMP_RE", "add_argument",
    "allocate", "bundle_init", "describe", "first_wrfout_init",
    "format_stamp", "is_run_folder", "latest", "owning_run",
    "parse_stamp", "parse_time", "record_latest", "relative_to_case",
    "resolve", "run_stamp_enabled", "stage_flags", "utcnow",
    "wrfout_init",
]
