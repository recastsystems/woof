"""The short version of ``woof domain``, for a person at a terminal.

``woof domain`` takes three required flags, so typing it bare answers
with argparse's usage dump and exit 2.  That is the correct answer to a
script and the wrong one to a person: the owner's reaction to it was
"so we don't have an easier way of doing it?".  There is now.  On a
terminal, bare ``woof domain`` asks four questions and runs.

Two rules shape everything here.

**The prompts collect values; they do not build anything.**  A session
produces an ordinary ``argv`` list of the same flags a user could have
typed, and that list goes through the same parser into the same
:func:`woof.domain_wizard.domain_main`.  There is no second sizing
path, no second validator, and no way for the two front doors to drift
into emitting different files -- the equivalence is by construction
rather than by a test that has to keep noticing.  The session also
prints the command it assembled, so the flag form is learned by using
the short one.

**Interactivity is layered exactly like verbosity.**  A hidden prompt
that a CI job can block on is a worse failure than a usage error, so
the prompt session requires BOTH stdin and stdout to be terminals.
Redirected, piped, or under a scheduler, bare ``woof domain`` keeps
today's behaviour to the letter: usage on stderr, exit 2.  Any flag at
all also means the caller knows what they want, and the old path runs
untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

from woof.hrrr_route_inputs import (
    ROUTE_DEFAULT_PHYSICS_PROFILE as _HRRR_ROUTE_DEFAULT_PHYSICS_PROFILE,
)

#: The sources the wizard accepts, in the order the prompt offers them.
#: GFS leads because it is the only one of the three that a new install
#: can fetch with no account and no manual step: HRRR is CONUS-only and
#: ERA5's default wizard provider, Copernicus CDS, needs a personal key.
#: The flag default stays era5 --
#: changing it would edit every documented example -- so this is a
#: default for the person who did not state a preference, not a change
#: of the project's default.
SOURCES = ("gfs", "hrrr", "era5")
DEFAULT_SOURCE = "gfs"

#: Default forecast length, matching ``--hours``.
DEFAULT_HOURS = 6

#: Where the emitted TOML goes when the reader just presses Enter.
DEFAULT_OUT_DIR = "configs"

#: The nest ladder a bare session emits: one 12 km domain.
#:
#: ``auto`` -- the deepest ladder that fits the card, four domains down
#: to 500 m on a large one -- is the right answer for someone who asked
#: for a ladder; it is the wrong DEFAULT, because ``woof go`` drives
#: the SINGLE-domain runner and refuses a tree, so a default-following
#: reader was handed a file the other new front door would not take.
#: Two features that do not compose is not a feature.  This door ruled
#: that first; the flags door's ``--ladder`` default was still ``auto``
#: until the 4090 user-zero run met the same refusal there, and now
#: both doors default to this shape (register_cli in
#: :mod:`woof.domain_wizard`).  Nests are explicit opt-in on either:
#: a deeper preset, ``auto``, or ``--root-dx``/``--chain``.
#:
#: This is exactly the shape ``docs/public/FIRST-LIGHT.md`` section 3a's
#: worked first run uses (``--ladder 12``), so the short path emits
#: the documented first run rather than a private variant of it.
DEFAULT_LADDER = "12"

#: The physics a bare session emits, per source.
#:
#: Without one the wizard emits the product default suite, which is not
#: one of the shipped runner profiles, so the prepared single-domain
#: runner refuses it and ``woof go`` refuses it earlier and says so.
#: The short front door therefore names a profile, and names the
#: strongest one its source's route can actually RUN.  For gfs and era5
#: that is ``morrison-mp10-...``, the registry's only ``wrf-matched-run``
#: template with full lw+sw radiation and FIRST-LIGHT section 3a's own
#: worked example.
#:
#: For hrrr it is not.  This table gave hrrr the Morrison suite too, and
#: that suite is refused at emission by the nested HRRR route's own
#: physics gate -- ``d01 cu_physics=1 (the route requires 0)``, because
#: HRRR's 3 km grid is convection permitting and Kain-Fritsch would
#: parameterize convection it already resolves.  A bare hrrr session
#: therefore emitted a config its own route would not take.  The entry is
#: now the strongest ROUTE-ADMISSIBLE full-radiation suite: Thompson mp8
#: with RTE+RRTMGP longwave and shortwave and no cumulus, which matches
#: :data:`woof.domain_wizard.HRRR_DEFAULT_PROFILE` -- the two doors
#: agree, as they must.  The same composition on the legacy RRTMG
#: engines, which is what this entry named while that was the route's
#: default, stays selectable by id.
#:
#: ``tests/test_domain_interactive.py`` re-derives every entry from the
#: generated registry -- offered by that source's route, admitted by that
#: route's physics gate, and the strongest maturity that survives both --
#: so this table cannot quietly outlive the facts it is quoting.
DEFAULT_PHYSICS_PROFILE_BY_SOURCE = {
    "gfs": "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1",
    "hrrr": _HRRR_ROUTE_DEFAULT_PHYSICS_PROFILE,
    "era5": "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1",
}


def default_physics_profile(source: str) -> str:
    """The profile a bare session emits for ``source``.

    The table above names the three sources this prompt OFFERS, each with
    the strongest suite its own route admits.  Any other registered source
    a reader types gets the flags door's own default, so the two doors
    cannot hand out different physics for the same source -- and a source
    reachable by flags is never unreachable here for want of a table row.
    """

    from woof.domain_wizard import resolved_physics_profile

    key = source.lower()
    if key in DEFAULT_PHYSICS_PROFILE_BY_SOURCE:
        return DEFAULT_PHYSICS_PROFILE_BY_SOURCE[key]
    return resolved_physics_profile(key, None)


class PromptAborted(Exception):
    """The reader stopped the session (Ctrl-C, or stdin reached EOF)."""


def is_interactive(argv: list[str], *, stdin=None, stdout=None) -> bool:
    """Should ``argv`` start a prompt session?

    Exactly ``["domain"]`` -- one token, no flags.  ``woof domain
    --explain`` is a caller who has started stating what they want, and
    answering that with questions would be a surprise; it takes the
    usage error it has always taken.

    Both streams must be terminals.  stdin alone is not enough: a run
    whose output is piped into a file or a log is a run nobody is
    watching, and prompting into that pipe hangs a job on a question no
    one will ever see.
    """

    if argv != ["domain"]:
        return False
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    try:
        return bool(stdin.isatty() and stdout.isatty())
    except (AttributeError, ValueError):
        # A closed or substituted stream that cannot answer is not a
        # terminal.  Fail toward the non-interactive path, which is the
        # one that cannot hang.
        return False


def _ask(prompt: str, default: str | None, validate) -> str:
    """Ask until the answer validates; return the accepted raw string.

    ``validate`` raises ``ValueError`` with the message the reader
    should see.  Re-prompting rather than exiting is the whole point of
    a prompt: a typo in the fourth answer must not throw away the first
    three.

    EOF is not a typo -- it means there is no one there (a closed pipe,
    a Ctrl-D) -- so it aborts instead of looping forever on a stream
    that will never produce another line.
    """

    label = f"{prompt} [{default}]: " if default is not None else f"{prompt}: "
    while True:
        try:
            raw = input(label).strip()
        except EOFError as error:
            raise PromptAborted("stdin closed") from error
        except KeyboardInterrupt as error:
            raise PromptAborted("interrupted") from error
        if not raw and default is not None:
            raw = default
        if not raw:
            print("  (this one has no default -- a value is needed)")
            continue
        try:
            validate(raw)
        except ValueError as error:
            print(f"  {error}")
            continue
        return raw


def _validate_point(raw: str) -> None:
    from woof.domain_wizard import _parse_point, swapped_point_refusal

    # The source is asked next, so only the pole case is asked here; the
    # flags door asks the source's coverage too.
    swapped = swapped_point_refusal(*_parse_point(raw), source=None)
    if swapped is not None:
        raise ValueError(swapped)


def _validate_source(raw: str) -> None:
    """Accept any source the flags door can plan for, not only the three
    this prompt OFFERS.

    :data:`SOURCES` is the shortlist the prompt suggests -- the three a
    new install is most likely to want -- and it stays short on purpose.
    It was also the accepted SET, so a reader who knew the product ships
    a RAP profile and typed ``rap`` here was told "source must be one of
    gfs, hrrr, era5", which is a false statement about the product.  The
    registry decides, in the same words the flags door uses.
    """

    from woof.domain_wizard import resolve_source

    resolve_source(raw)


def _cycle_validator(source: str):
    """A cycle validator bound to the source, because the rules differ.

    Whether ``latest`` has an answer is a property of the source's
    declared initialization grid, asked here while the reader is still
    typing rather than after the point and hours are already collected.
    """

    def validate(raw: str) -> None:
        if raw.strip().lower() == "latest":
            # WHETHER `latest` HAS AN ANSWER is the resolver's question,
            # asked here rather than restated.  This door used to answer
            # it with a branch on era5 and a fetch-door check, so a
            # reanalysis whose publication delay is a declared number was
            # told it had no latest at all.
            from woof.fetch import require_cycle_grid

            require_cycle_grid(source)
            return
        from woof.fetch import parse_cycle

        parse_cycle(raw, source)

    return validate


def _validate_hours(raw: str) -> None:
    try:
        hours = int(raw)
    except ValueError as error:
        raise ValueError("forecast hours must be a whole number") from error
    if hours < 1:
        raise ValueError("forecast hours must be at least 1")


def _validate_out(raw: str) -> None:
    if not raw.strip():
        raise ValueError("an output path is needed")


def detected_vram_gib() -> float | None:
    """Physical VRAM of the card, or None when nothing can be read.

    Reuses the estimator's own NVML capacity probe -- the one that is
    deliberately answerable without standing up a CUDA context -- so the
    prompt session asks the same authority the memory preflight will ask
    later, rather than introducing a second opinion about the card.

    None is a real answer and is handled by the caller: a machine whose
    GPU cannot be read gets the documented ``--card`` default, and is
    told so.
    """

    try:
        from woof.core.preflight import device_physical_total_bytes

        total = device_physical_total_bytes()
    except Exception:
        return None
    if not total or total <= 0:
        return None
    return total / (1024 ** 3)


def _default_out_path(lat: float, lon: float) -> str:
    from woof.domain_wizard import _default_name

    return str(Path(DEFAULT_OUT_DIR) / f"{_default_name(lat, lon)}.toml")


def default_cycle_answer(source: str) -> str | None:
    """What the cycle prompt offers for ``source``, or ``None``.

    ``latest`` is offered where something can resolve it, and that is
    ONE question: does this source declare when it initializes.  The
    predicate is the resolver's own, so the default can never offer an
    answer the next prompt refuses -- which is what it did while two
    hand-written rules (a branch naming era5, and a fetch-door check)
    stood in for it.
    """

    from woof.source_cycles import cycle_grid_for

    return "latest" if cycle_grid_for(source) is not None else None


def collect(*, printer=print) -> list[str]:
    """Run the prompt session; return the ``argv`` it assembled.

    The returned list is exactly what the reader could have typed.
    Nothing else in this module knows how to build a config, and this
    function knows nothing about projections, ladders or budgets.
    """

    from woof.domain_wizard import _parse_point, resolve_source

    printer("woof domain: no arguments, so here is the short version.")
    printer("Enter accepts the default in [brackets].  Ctrl-C stops.")
    printer("")

    point = _ask("  center point, lat,lon", None, _validate_point)
    lat, lon = _parse_point(point)

    source = resolve_source(
        _ask("  source (gfs/hrrr/era5, or any registered source)",
             DEFAULT_SOURCE, _validate_source))
    # What this source needs configured, from its registry row -- not an
    # arm for one id.  A row that declares nothing prints nothing.
    from woof.domain_wizard import source_credential_notes

    for note in source_credential_notes(source):
        printer(f"  note: {note}")

    cycle_default = default_cycle_answer(source)
    cycle = _ask("  cycle, YYYY-MM-DDTHH or latest", cycle_default,
                 _cycle_validator(source))

    hours = _ask("  forecast hours", str(DEFAULT_HOURS), _validate_hours)
    out = _ask("  output file", _default_out_path(lat, lon), _validate_out)

    profile = default_physics_profile(source)
    argv = ["domain", f"--point={point}", "--source", source,
            "--cycle", cycle, "--hours", str(int(hours)),
            "--ladder", DEFAULT_LADDER, "--physics-profile", profile,
            "--out", out]
    # Said out loud, because these two are the only answers the session
    # supplied that the reader did not: a default that changes what runs
    # has to be visible at the moment it is chosen, not discovered in
    # the emitted file.
    printer(f"  ladder: {DEFAULT_LADDER} km, one domain -- the shape "
            "`woof go` runs end to end (pass --ladder for nests).")
    printer(f"  physics: {profile} (pass --physics-profile for another).")
    if source in {"gfs", "gdas"}:
        # Not a sixth question: the analysis start is right for almost
        # every first run, and a prompt for an advanced choice would
        # lengthen the shortest path to one.  But a session that never
        # MENTIONS the choice is a session in which the feature does not
        # exist, so it is named where the other defaults are named.
        printer("  start: the cycle's f000 analysis (pass "
                "--forecast-start-hour K to start from the f{K} forecast "
                "lead instead -- the way to reach a window deep in a "
                "forecast without integrating to it).")

    vram = detected_vram_gib()
    if vram is not None:
        argv += ["--vram-gib", f"{vram:.2f}"]
        printer(f"  card: {vram:.2f} GiB detected, used as the budget.")
    else:
        # Not a failure: the flag path's own default applies, and the
        # reader is told which number is about to size their domain
        # rather than discovering it in the sizing line.
        from woof.domain_wizard import CARD_VRAM_GIB

        printer(f"  card: no GPU readable, assuming "
                f"{CARD_VRAM_GIB['24gb']:g} GiB (pass --card or "
                "--vram-gib to state it).")
    return argv


def printable_command(argv: list[str]) -> str:
    """The assembled session as a command line the reader can re-run.

    Printed before the run so the short path teaches the long one: the
    second time, this is the line to paste.
    """

    import shlex

    return "woof " + " ".join(shlex.quote(token) for token in argv)


__all__ = [
    "DEFAULT_HOURS", "DEFAULT_LADDER", "DEFAULT_OUT_DIR",
    "DEFAULT_PHYSICS_PROFILE_BY_SOURCE", "DEFAULT_SOURCE", "SOURCES",
    "PromptAborted", "collect", "default_physics_profile",
    "detected_vram_gib", "is_interactive", "printable_command",
]
