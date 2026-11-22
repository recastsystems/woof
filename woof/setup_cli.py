"""``woof setup``: the staging steps a wheel install still needs.

``pip install recast-woof`` lands the Python half and nothing else.  The wheel
ships no compiled Rust -- every GRIB decode goes through the fail-closed
bridges -- and the two largest Thompson tables are published as release
assets because they exceed PyPI's per-file cap.  Both gaps already had a
one-command fix; what they did not have was ONE command.  A first
install therefore read as a list of chores whose order and necessity the
reader had to infer, which is the half of the install complaint that no
amount of message layering fixes.

So this is a wrapper, deliberately thin.  Each step is
:mod:`woof.bridge_assets` or :mod:`woof.table_assets` doing exactly
what it does when invoked directly -- same verification, same pins, same
refusals, same exit codes -- and this module adds only sequencing, one
line of status per step, and a closing doctor summary.  Nothing new
downloads anything, so there is no second implementation to drift.

**Static geography is opt in.**  ``woof fetch-geog`` unpacks tens of
gigabytes, which is not a thing to start on someone's behalf because
they typed a command called ``setup``.  ``--with-geog`` asks for it, and
prints the size -- read off the pin table, never a literal -- before the
first byte moves.

**Re-run safe**, because every step is: staged artifacts that match
their pins are verified and skipped, so a second ``woof setup`` is a
verification pass.  The wrapper adds no state of its own -- no marker
file, no "already done" short circuit -- precisely so that a re-run
after a partial or corrupted first run does the checking rather than
trusting a flag.
"""

from __future__ import annotations

import argparse
import contextlib
import io

from woof.explain import explain_enabled

#: One setup step: its label and the module whose ``register_cli``
#: declares the subcommand this step IS.  Ordered so that a reader who
#: stops after step one still has the half that unblocks every data
#: route: the decoders, then the tables.
_STEPS: tuple[tuple[str, str], ...] = (
    ("bridges", "woof.bridge_assets"),
    ("tables", "woof.table_assets"),
)


def step_namespace(module_name: str) -> argparse.Namespace:
    """The exact defaults ``woof <that subcommand>`` would run with.

    Read out of the real registrar rather than transcribed here.  A
    transcribed copy is a second declaration of every default -- the
    geog mirror choice, the archive retention, the dataset list -- and
    the failure mode is silent: setup would keep running yesterday's
    defaults while the documented command ran today's, and no test that
    exercises either one alone would notice.
    """

    from importlib import import_module

    parser = argparse.ArgumentParser(prog="woof")
    subparsers = parser.add_subparsers(dest="command")
    import_module(module_name).register_cli(subparsers)
    [name] = list(subparsers.choices)
    return parser.parse_args([name])


def _run_step(module_name: str, overrides: dict) -> tuple[int, str]:
    """Run one step's real handler; return its exit code and its output.

    The output is captured rather than streamed so the wrapper can print
    one line per step by default and still hand over every captured word
    when the step fails or when ``--explain`` asks.  Capturing is not
    discarding: a failing step replays its full text, because the reason
    a step refused is the one thing a setup command must never swallow.

    A step that RAISES rather than returning a code is handled here, in
    the one place that holds its captured output.  Letting the exception
    escape would have thrown away everything the step had already
    printed before it died -- which, for a download that got most of the
    way through, is the whole diagnosis.
    """

    namespace = step_namespace(module_name)
    for key, value in overrides.items():
        setattr(namespace, key, value)
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            code = namespace.func(namespace)
    except Exception as error:
        return 2, buffer.getvalue() + f"{type(error).__name__}: {error}\n"
    return code, buffer.getvalue()


def _replay(text: str) -> None:
    """Print a step's captured output, indented under its status line."""

    for line in text.splitlines():
        print(f"    {line}" if line.strip() else "")


def _geog_size_phrase() -> str:
    """The default fetch set's sizes, from the pin table."""

    from woof.geog_assets import size_phrase

    return size_phrase()


def _geog_size_notice() -> str:
    """Printed before ``--with-geog`` starts, because 30 GB is a decision.

    The sizes come from the pin table
    (:func:`woof.geog_assets.size_phrase`), which is also where
    :data:`woof.doctor.GEOG_HINT` reads them, so a reader cannot be
    told two different sizes -- and neither text can go stale when a
    dataset is added, which is exactly what a pair of literals did.
    """

    from woof.geog_assets import GEOG_ARCHIVES, size_phrase

    # The default fetch set, which is what the geog step stages (defect:
    # this counted every pin, 25, while 15 were fetched).
    count = sum(1 for archive in GEOG_ARCHIVES if archive.fetch_by_default)
    # The default set is the permissive one (2026-10-21): SoilGrids250m soil
    # categories (CC BY 4.0) and a deep-soil temperature BUILT here from
    # ERA5 (Licence to Use Copernicus Products), so the notice names both
    # credits and says one dataset is built rather than downloaded.
    built = [archive.dataset for archive in GEOG_ARCHIVES
             if archive.fetch_by_default and archive.producer]
    built_phrase = (f" ({', '.join(built)} is built on this machine from "
                    "ERA5)" if built else "")
    return (f"setup: --with-geog will stage the {count} default "
            f"WPS_GEOG datasets{built_phrase}: {size_phrase()}.  "
            "Every default source is permissively licensed (SoilGrids250m "
            "CC BY 4.0, ERA5 under the Licence to Use Copernicus Products; "
            "credits in NOTICE); the FAO-UNESCO soil tiles and ISLSCP-I "
            "soiltemp_1deg are fetched only when named.  `woof "
            "fetch-geog --datasets wrf` stages the WRF set alone.  "
            "Resumable and re-run safe; omit --with-geog to skip it and "
            "run `woof fetch-geog` later.")


def _geog_skipped_notice() -> str:
    """Printed when ``setup`` finishes WITHOUT ``--with-geog``.

    Excluding the tree is deliberate -- the download is a decision, not
    a default -- but staying silent about it moved the surprise to prep
    time, several commands downstream of the one that would have closed
    the gap.  So the door that skipped it says so, with the size that
    makes it a decision and the command that ends it.
    """

    from woof.geog_assets import size_phrase

    return (f"setup: not staged -- WPS_GEOG static geography "
            f"({size_phrase()}), which preprocessing reads.  Run `woof "
            "fetch-geog` when you want it, or re-run setup with "
            "--with-geog.")


GEOG_SIZE_NOTICE = _geog_size_notice()
GEOG_SKIPPED_NOTICE = _geog_skipped_notice()


def setup_main(args) -> int:
    """Run every staging step, then report the estate.

    Returns the first failing step's exit code, or doctor's own verdict
    when every step succeeded.  "First failing" rather than "last"
    because the earlier step is the one further upstream, and its
    refusal is usually why the later one had nothing to do.
    """

    explain = explain_enabled(args)
    steps = list(_STEPS)
    if getattr(args, "with_geog", False):
        print(GEOG_SIZE_NOTICE)
        steps.append(("geog", "woof.geog_assets"))

    failed = 0
    asked_failed = False
    for label, module_name in steps:
        overrides = {}
        if getattr(args, "from_dir", None) is not None and label != "geog":
            overrides["from_dir"] = args.from_dir
        if label == "tables":
            # The mixed-layer ocean (sf_ocean_physics = 1 or 2; an omitted
            # key is 0 since 3.0.2) starts from the WOA18 climatology, so it
            # is staged with the tables.  Breakage it
            # prevents: NCEI THREDDS down for an hour (2026-10-22) killed
            # every forecast with water at its first step on boxes and
            # images that setup had called complete.
            overrides["ocean"] = True
        code, output = _run_step(module_name, overrides)
        if code == 0:
            print(f"  ok      {label}")
            if explain:
                _replay(output)
        else:
            # Never swallow a refusal.  A failed step prints everything
            # it said, whether or not --explain was asked for: a
            # wrapper that hides why a step refused is worse than no
            # wrapper, because the reader now has to work out which
            # underlying command to re-run before they can even read
            # the error.
            print(f"  FAILED  {label} (exit {code})")
            _replay(output)
            failed = failed or code

    # Every step still runs.  They are independent -- staging tables
    # does not need the decoders, and both are idempotent -- so
    # stopping at the first refusal would leave a second, unrelated gap
    # undiscovered until the reader re-ran and hit it next.
    #
    # The Earth2Studio interpreter and TempestExtremes, which tc-tracks and
    # earth2-diagnose run.  Breakage (2.9.3 door 6, 2026-10-20): no setup
    # path built them, so a fresh install had no working tc-tracks backend.
    if getattr(args, "with_earth2", False):
        from woof import earth2_env
        try:
            receipt = earth2_env.ensure(progress=print)
        except (ValueError, RuntimeError, OSError) as error:
            print("  FAILED  earth2")
            _replay(str(error))
            failed = failed or 2
            # Asked for by name, so no complete estate excuses it. Breakage (2.9.4 R15, RTX 4090,
            # 2026-10-22): the earth2 build failed, setup still exited 0 through the estate-complete
            # note below, and the FCN3 fetch after it failed on "earth2studio is not installed".
            asked_failed = True
        else:
            built = ", ".join(receipt["built"]) or "already complete, verified"
            print(f"  ok      earth2 ({built}; {receipt['root']})")

    # Close with the estate report either way, because after a partial
    # setup "where do I stand" is exactly the question, and it is the
    # same function `woof doctor` runs rather than a summary of it.
    from woof.doctor import (blocking_gaps, collect_checks, format_brief,
                              format_report)

    # Before the estate report, not after: the report is about to list
    # WPS_GEOG as a gap, and a reader who has just been told setup
    # deliberately skipped it reads that line as expected rather than as
    # a failure of the command they typed.
    if not getattr(args, "with_geog", False):
        print(GEOG_SKIPPED_NOTICE)

    checks = collect_checks()
    print(format_report(checks) if explain else format_brief(checks))
    if failed:
        if not asked_failed and not blocking_gaps(checks):
            # Field case: a checkout install has no published bundle
            # pins, so fetch-bridges refuses -- while the bridges it
            # would have fetched are already built and verified (the
            # doctor brief above says so).  A step that could not run
            # against an estate that is already complete is a note,
            # not a failure.
            print("setup: a step above refused, but the estate is "
                  "already complete (no blocking gaps); nothing was "
                  "missing.")
            return 0
        print("setup: a step above refused; its own output is quoted "
              "under it.  The remaining steps still ran.")
        return failed
    # Every step succeeded: setup did its whole job.  Only a broken or
    # integrity-suspect finding fails the exit; a documented optional
    # absence (cupy on a base install, the render extra, a WPS_GEOG
    # tree nobody opted into) reports in the brief above and exits 0.
    return 1 if blocking_gaps(checks) else 0


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "setup",
        help="stage what a pip install still needs: the prebuilt Rust "
             "artifacts (woof fetch-bridges) then the externalized "
             "physics tables and ocean climatology (woof fetch-tables "
             "--ocean), then print the doctor "
             "summary.  Idempotent; static geography is opt-in via "
             "--with-geog")
    parser.add_argument(
        "--with-geog", action="store_true", dest="with_geog",
        help="also stage the WPS_GEOG static geography ("
             + _geog_size_phrase() + "); the size is printed before "
             "anything downloads")
    parser.add_argument(
        "--with-earth2", action="store_true", dest="with_earth2",
        help="also build the managed Earth2Studio environment (torch, "
             "earth2studio with the fcn3, cyclone and diagnostic extras) and the "
             "TempestExtremes tools that woof tc-tracks and woof "
             "earth2-diagnose run; several GB of wheels, built at "
             "WOOF_EARTH2_ENV (default ~/.local/share/woof/earth2-env)")
    parser.add_argument(
        "--from", dest="from_dir", metavar="DIR", default=None,
        help="stage the bridge and table artifacts from a local "
             "directory instead of downloading (offline installs); "
             "verification is identical.  Does not apply to --with-geog, "
             "which has its own --source")
    parser.set_defaults(func=setup_main)
    return parser


__all__ = ["GEOG_SIZE_NOTICE", "register_cli", "setup_main"]
