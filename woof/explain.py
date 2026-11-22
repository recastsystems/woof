"""One layering convention for everything the CLI prints.

Every message this project prints earned its words.  The refusals name
what they refused and why the rail exists; the remedies are pasteable
whole; the receipts say what was verified against what.  None of that is
being cut.  What changed is that all of it used to print at once, so the
one line a reader needed -- the next command -- arrived in the middle of
a wall and was read as part of the wall.  A field exhibit: a wizard run
whose correct ``woof fetch`` line sat at line 15 of 20, under a
gray-zone advisory and above a nine-name dataset inventory, and whose
user's verdict was "still can't get it working".

So the words are layered rather than shortened:

* **default** -- what happened, and the single next action.  One line
  per item wherever items repeat.
* **``--explain``** -- the same output with the mechanism prose, the
  alternate routes, and the per-item evidence restored *verbatim*.

The convention is one flag with one name on every subcommand
(:func:`add_explain_flag` puts it on all of them, so the pointer this
module appends is never a lie), and one composition helper
(:func:`layered`) for messages that carry both halves.

Why a text sentinel rather than a structured exception.  The refusals
travel as ``ValueError``/``NotImplementedError`` through call chains
this package does not own, and are asserted on by tests that read
``str(error)``.  :func:`layered` keeps both halves inside that one
string, so every existing content assertion still holds and only the
*print boundary* -- :func:`render` -- decides which half reaches the
terminal.  Nothing is deleted; a layer is chosen.
"""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import sys

#: Separates a message's ACTION half from its WHY half inside one string.
#:
#: A sentinel rather than a blank line because prose contains blank
#: lines and a heuristic split would eventually cut a paragraph in half.
#: It is written only by :func:`layered` and removed by :func:`render` in
#: BOTH modes, so it cannot reach a terminal even on a path that forgets
#: to ask which layer it wanted.
EXPLAIN_MARK = "\n[[explain]]\n"

#: Appended to a layered message printed without ``--explain``.
_POINTER = "  (run {command} --explain for the reason)"

#: The tokens of the invocation in progress, as the reader typed them
#: (without the leading ``woof``), or ``None`` outside the CLI.
#:
#: Why the pointer needs this: ``--explain`` is a MODIFIER, so a tail
#: that printed only ``run woof domain --explain`` handed the reader a
#: line that is itself an argparse usage error -- measured in the UX
#: persona walks (finding N8), on five doors in one session.  The tail
#: must be the reader's own full invocation plus the one flag, and only
#: the front door knows that invocation, so it records it here (through
#: :func:`explain_scope`, which bounds it to the one call) and
#: :func:`render` reads it at the print boundary.
_INVOCATION: tuple[str, ...] | None = None


def set_invocation(tokens) -> None:
    """Record the invocation in progress, for the pointer tail.

    Called by ``woof.cli`` once per dispatch, inside the
    :func:`explain_scope` that door opened, with the tokens exactly as
    the reader typed them.  ``None`` forgets it.
    """

    global _INVOCATION
    _INVOCATION = (None if tokens is None
                   else tuple(str(token) for token in tokens))


def invocation() -> tuple[str, ...] | None:
    """The tokens :func:`set_invocation` recorded, or ``None``."""

    return _INVOCATION


def _shell_word(token: str) -> str:
    """``token`` spelled so a shell reads it back as one word."""

    if token and not any(char.isspace() for char in token) \
            and '"' not in token:
        return token
    return '"' + token.replace('"', '\\"') + '"'


def reinvocation(fallback: str | None = None) -> str | None:
    """The reader's own command line, re-runnable, else ``fallback``.

    ``fallback`` is the command NAME a call site holds (``woof
    render``); it is the whole pointer only on a path no front door
    recorded -- the ``python -m`` doors, an embedder calling a handler
    directly, a test driving one function.  Recorded beats named,
    because the name alone produces the dead-end tail this exists to
    close.
    """

    if _INVOCATION is None:
        return fallback
    return " ".join(("woof", *map(_shell_word, _INVOCATION)))


def layered(action: str, why: str) -> str:
    """Compose a message from its action half and its explanation half.

    ``action`` is what was refused or what happened, plus the exact
    remedy: everything a reader needs in order to act.  ``why`` is the
    mechanism -- the paragraph that says what the rail is protecting,
    which route was withdrawn, what the alternative costs.

    A caller with no explanation half gets its action back unchanged, so
    wrapping a message that has not been split yet is a no-op rather
    than a message with an empty section.

    Surrounding blank lines are removed from both halves; leading
    INDENTATION on the first line is not.  These messages are written
    as indented blocks -- ``  What to do:`` above ``  Why:`` -- and a
    normalizer that reached for ``strip()`` would left-align the
    explanation while the action kept its gutter, so the two halves
    would print as if they came from different messages.
    """

    action = action.strip("\n").rstrip()
    why = why.strip("\n").rstrip()
    if not why.strip():
        return action
    return action + EXPLAIN_MARK + why


def split(message: str) -> tuple[str, str]:
    """``(action, why)`` for a message; ``why`` is ``""`` when unlayered."""

    text = str(message)
    head, mark, tail = text.partition(EXPLAIN_MARK)
    return (head, tail) if mark else (text, "")


def render(message: str, *, explain: bool, command: str | None = None) -> str:
    """The layer of ``message`` that ``explain`` asked for.

    With ``explain`` the two halves are rejoined with a blank line and
    printed in full -- that is the whole promise of the flag, and it is
    why the explanation half is stored verbatim rather than summarized.

    Without it the action half stands alone, followed by a pointer at
    the flag that produces the rest.  The pointer is the reader's OWN
    invocation with ``--explain`` appended -- re-runnable exactly as
    printed -- whenever the front door recorded one
    (:func:`set_invocation`); ``command`` is the fallback name for the
    paths no door recorded.  When a caller cannot even name the command
    the pointer is omitted rather than guessed, because a pointer at a
    command that does not take ``--explain`` is worse than no pointer.
    """

    action, why = split(message)
    if not why:
        return action
    if explain:
        return f"{action}\n\n{why}"
    line = reinvocation(command)
    if line:
        return action + "\n" + _POINTER.format(command=line)
    return action


def add_explain_flag(parser: argparse.ArgumentParser, *,
                     nested: bool = False) -> None:
    """Register ``--explain`` on one parser, idempotently.

    Idempotent because the CLI adds the flag by sweeping every
    registered subparser, and two registrars share a parser (``check``
    is built by the ingest preflight and extended by the memory
    estimator; ``run``/``resume`` are extended by the supervisor).  A
    second registration would be ``argparse.ArgumentError`` at import
    time, which is a startup crash rather than a message-layer bug.

    ``nested`` is for a subcommand of a command that takes the flag
    itself (``woof research hardware`` under ``woof research``).
    argparse copies every value the selected child parsed, defaults
    included, over the parent's, so a child that defaulted to False
    erased a ``--explain`` typed before the subcommand's name.  A nested
    child therefore supplies no default at all and records the flag
    only when it is typed after the name; the parent keeps the default.
    """

    if any("--explain" in action.option_strings
           for action in parser._actions):  # noqa: SLF001 - argparse's only API
        return
    parser.add_argument(
        "--explain", action="store_true",
        default=argparse.SUPPRESS if nested else False,
        help="print the full reasoning, alternate routes and per-item "
             "evidence behind this command's output, instead of the "
             "default one-line-per-item summary")


def explain_enabled(args) -> bool:
    """Did the caller ask for the full layer?  Absent flag means no."""

    return bool(getattr(args, "explain", False))


#: Record of whether the INVOCATION IN PROGRESS asked for the full
#: layer.  Library code that emits warnings has no ``args`` in reach;
#: a front door stamps the flag here once, right after parsing.
#:
#: Module state, so its lifetime has to be the invocation's rather than
#: the interpreter's -- see :func:`explain_scope`, which is how every
#: front door sets it.
_EXPLAIN_ACTIVE = False


def set_explain(enabled: bool) -> None:
    """Record --explain for the invocation in progress.

    Called by a front door right after it parses its arguments, from
    inside the :func:`explain_scope` that door opened.  Setting it
    outside a scope leaves it set for the rest of the interpreter, which
    is the defect that context manager exists to prevent.
    """

    global _EXPLAIN_ACTIVE
    _EXPLAIN_ACTIVE = bool(enabled)


@contextlib.contextmanager
def explain_scope(enabled: bool = False):
    """Bound one front door's --explain state to that one invocation.

    ``woof.cli.main`` and the ``tools/`` doors are ordinary functions,
    and plenty of callers reach them without spawning a process: the
    console script, an embedder scripting the CLI, and the test suite
    most of all.  A door that only ever CALLS :func:`set_explain` leaves
    the flag standing after it returns, so the NEXT invocation in the
    same interpreter silently inherits an ``--explain`` it never asked
    for, and every :func:`warn` it emits grows a mechanism continuation
    under its one contracted line.

    Measured, not theorised: ``pytest tests/test_doctor.py
    tests/test_fetch_engine_degrade_guard.py`` put two lines on stderr
    where the guard contracts one, and the same two files in the
    opposite order put one -- because ``woof doctor --explain`` ran
    first and never gave the flag back.  A battery is free to pack its
    shards either way, so the same tree was red or green by luck of the
    ordering.

    Entering forces ``enabled`` (a fresh invocation must not start from
    the last one's answer) and leaving restores whatever was in place
    before, so scopes nest and a door called from inside another door
    does not corrupt its caller's layer.

    The recorded invocation (:func:`set_invocation`) has exactly the
    same lifetime problem -- a pointer naming the LAST invocation's
    arguments is a lie about this one -- so it is bounded here too:
    cleared on entry, restored on exit.
    """

    global _EXPLAIN_ACTIVE, _INVOCATION
    previous = _EXPLAIN_ACTIVE
    previous_invocation = _INVOCATION
    _EXPLAIN_ACTIVE = bool(enabled)
    _INVOCATION = None
    try:
        with warning_scope():
            yield
    finally:
        _EXPLAIN_ACTIVE = previous
        _INVOCATION = previous_invocation


#: Callables that also receive every warning, as a typed mapping.
#:
#: The stderr sentence is the reader's interface and does not change.
#: This is the interface for a PROGRAM driving woof as a subprocess:
#: it needs the same facts as fields rather than as prose it would have
#: to recognize by shape.  A generic list of callables rather than one
#: named consumer, so the next machine-facing surface reuses it instead
#: of adding a second hook beside it.
#:
#: Observers are called inside the same call that prints, so a warning
#: is never observed later than it is printed.  One that raises would
#: turn an advisory into a failure, so each is called defensively.
_WARNING_OBSERVERS: list = []


def add_warning_observer(observer) -> None:
    """Also deliver every :func:`warn` to ``observer(record)``.

    ``record`` is ``{"action": ..., "why": ...}`` -- the two halves the
    layering convention already splits every message into, unjoined, so
    a consumer chooses its own layer the way :func:`render` does.
    """

    if not callable(observer):
        raise TypeError("warning observer must be callable")
    _WARNING_OBSERVERS.append(observer)


def remove_warning_observer(observer) -> None:
    """Detach an observer; absent is not an error."""

    try:
        _WARNING_OBSERVERS.remove(observer)
    except ValueError:
        pass


_WARNING_KEYS = contextvars.ContextVar("warning_keys", default=None)


@contextlib.contextmanager
def warning_scope():
    """Deduplicate keyed advisories within one review, including nested calls."""
    if _WARNING_KEYS.get() is not None:
        yield
        return
    token = _WARNING_KEYS.set(set())
    try:
        yield
    finally:
        _WARNING_KEYS.reset(token)


#: True inside :func:`muted_warnings`: the warnings raised there are
#: about a configuration nobody asked for, so they go nowhere.
_WARNINGS_MUTED = contextvars.ContextVar("warnings_muted", default=False)


@contextlib.contextmanager
def muted_warnings():
    """Drop every :func:`warn` raised inside, printed and observed alike.

    For a door that loads configurations it only PRICES: the memory
    refusal's candidate suites and shallower ladders are built and
    loaded to price them, and each one's loader warnings described a
    file the reader never asked for and will not get.  Measured on a
    6 GiB card: ``woof domain --ladder 12-3-1-0.5 --source gfs`` printed
    four SASE and km_opt nest warnings from pricing suites the refusal
    then named or dropped, above the refusal itself.  A muted warning
    spends no ``once`` or keyed slot either, so the same sentence about
    the file that IS written still prints.
    """

    token = _WARNINGS_MUTED.set(True)
    try:
        yield
    finally:
        _WARNINGS_MUTED.reset(token)


#: The sentences already printed inside :func:`each_warning_once`, or None
#: outside it.
_WARNINGS_SAID = contextvars.ContextVar("warnings_said", default=None)


@contextlib.contextmanager
def each_warning_once():
    """Print each distinct :func:`warn` sentence once inside, however often it is raised.

    For a door that loads the file it writes more than once (``woof
    domain`` round-trips its emitted bytes through the loader, then hands
    the same file to ``woof check``): every load raised the file's loader
    warnings again.  Observers still receive every call.  Nested uses share
    the outermost record.
    """

    if _WARNINGS_SAID.get() is not None:
        yield
        return
    token = _WARNINGS_SAID.set(set())
    try:
        yield
    finally:
        _WARNINGS_SAID.reset(token)


def warn_once(key: str, action: str, why: str = "") -> None:
    """Emit a keyed advisory once in the current review, or once per direct call."""
    if _WARNINGS_MUTED.get():
        return
    keys = _WARNING_KEYS.get()
    if keys is not None:
        if key in keys:
            return
        keys.add(key)
    warn(action, why)


#: Sentences a ``warn(..., once=True)`` call has already printed in this
#: process.
_PRINTED_ONCE: set[str] = set()


def warn(action: str, why: str = "", *, once: bool = False) -> None:
    """Print one warning sentence and keep going.

    This is the voice of every check that found something worth saying
    but nothing worth stopping for: the run continues, and the reader
    gets exactly one ``warning:`` line saying what happened and (when
    there is one) what to do.  The mechanism prose goes in ``why`` and
    prints only when the invocation carried ``--explain`` -- same
    layering contract as the refusals, same flag.

    Warnings go to stderr so a piped stdout (JSON reports, command
    relays) stays clean.

    Every warning is additionally delivered to each observer registered
    with :func:`add_warning_observer`, which is how a machine consumer
    receives it as a field rather than as a line to recognize.

    ``once`` prints a sentence at most once per process, for a check
    that runs each time the same input is loaded (one ``woof run``
    loads its config several times).  Observers still receive every
    call, so one attached after the first print keeps its record.
    """

    if _WARNINGS_MUTED.get():
        return
    action = " ".join(str(action).split())
    said = _WARNINGS_SAID.get()
    if not ((once and action in _PRINTED_ONCE)
            or (said is not None and action in said)):
        if once:
            _PRINTED_ONCE.add(action)
        if said is not None:
            said.add(action)
        print(f"warning: {action}", file=sys.stderr)
        if why and _EXPLAIN_ACTIVE:
            for line in str(why).strip("\n").splitlines():
                print(f"  {line}", file=sys.stderr)
    if not _WARNING_OBSERVERS:
        return
    record = {"action": action, "why": str(why)}
    for observer in tuple(_WARNING_OBSERVERS):
        try:
            observer(record)
        except Exception:  # noqa: BLE001 - an advisory never fails a run
            continue


__all__ = [
    "EXPLAIN_MARK", "add_explain_flag", "add_warning_observer",
    "each_warning_once",
    "explain_enabled", "explain_scope", "layered", "muted_warnings",
    "reinvocation",
    "remove_warning_observer", "render", "set_explain",
    "set_invocation", "split", "warn",
]
