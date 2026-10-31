"""Write ``docs/public/CLI-OPTIONS.md`` from the parsers themselves.

The complete option surface of every documented door, in one page, built
by reading the argparse parsers rather than by remembering to write a
line.  ``tests/test_docs_extras_agree_with_code.py`` holds the committed
page against those same parsers in both directions, so the page cannot
silently fall behind the code and cannot name a flag the code dropped.

Run it after changing any CLI option::

    python -m tools.build_cli_options_doc

It rewrites the generated half of the page and leaves the hand-written
notes above each table alone -- those are keyed by door name in
:data:`NOTES` here, which is version-controlled prose, not scraped text.
"""

from __future__ import annotations

import argparse
import importlib
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "public" / "CLI-OPTIONS.md"

#: Hand-written orientation for the doors whose flags gate whole
#: features.  Everything else is served by its help text alone; these
#: are the ones where a reader needs to know that the flag is the ONLY
#: way in.  Keyed by door name.
NOTES: dict[str, str] = {
    "woof downscale": (
        "`--parent-namelist` (with `--parent-namelist-domain`) is the "
        "entire stock-WRF-parent route this command's own summary "
        "advertises: without it, only a woof parent run can be "
        "downscaled.  `--tiles {on,auto}` and `--child-size` are the "
        "only way to stream a `--point`-derived child, because this "
        "command authors the child TOML itself and a `[tiles]` table "
        "you wrote by hand would be overwritten.  A child builds its own "
        "terrain, land use and soil at its own spacing by default; "
        "`--parent-terrain` keeps its parent's, interpolated, and "
        "`--geog-root` names the WPS_GEOG tree the child is built from."),
    "woof go": (
        "`--no-memory-gate` is the only escape from the pre-fetch memory "
        "gate.  The gate runs before the chain downloads anything, and "
        "on a box whose card it cannot see it declines to refuse rather "
        "than blocking a run that would have worked."),
    "woof multi-run": (
        "`--preflight {estimate,alloc,off}` is the only override of the "
        "plan's own preflight mode."),
    "woof run": (
        "`--allow-shared-gpu`, `--gpu-uuid`, `--prep-timeout` and "
        "`--supervisor-max-restarts` are the command-line spellings of "
        "settings STREAMING.md documents only as run-plan keys."),
    "woof render": (
        "`--pair-labels`, `--pair-subtitle` and `--pair-title` title and "
        "label the paired CPU-vs-GPU figure `--pair` composes."),
    "woof-prepared-forecast": (
        "`--tiles JSON` is the only way to stream this route: its "
        "hash-bound experiment cannot carry a `[tiles]` table, so the "
        "table rides on the flag.  `--render-products` (with "
        "`--render-dir`) is render-on-first-committed-frame, off by "
        "absence.  `--materialize-authorities` and `--show-capabilities` "
        "each select a DIFFERENT program with its own options and must "
        "be the first argument on the line."),
    "woof-prepared-tree-forecast": (
        "`--sealed-forcing-extension` selects the append-only "
        "forcing-prefix checkpoint contract."),
    "rw-wps": (
        "`--validate-physics-plan`, `--canonical-physics-plan-output`, "
        "`--extend-root-preparation`, `--sealed-prepared-cache`, "
        "`--domain-source-orography`, `--validate-hrrr-domain` and "
        "`--no-stock-wrf-export` are gates on the preprocessing route; "
        "each is off unless named."),
}

#: Doors that are the same program reached by a second name.
ALIASES = {"woof-wrf-init": "rw-wps"}

#: Console-script modules whose parser factory is not ``build_parser``.
#: Everything absent from this table is asked for ``build_parser``, and a
#: module that answers neither is a refusal below rather than a silent
#: omission.
PARSER_FACTORY = {"woof.source_cli": "_parser"}

#: Doors that are NOT console scripts: a mode flag that selects a second
#: program inside one script, with its own parser and its own options.
#: Name -> ``(module, factory)``.
MODE_FLAG_DOORS = {
    "woof-prepared-forecast --materialize-authorities": (
        "woof.prepared_single_domain_forecast", "build_materialize_parser"),
}

#: What the page prints for an option whose parser declares no help.
#: The flag is still NAMED, which is the reachability contract; the
#: missing sentence is a separate, smaller debt, and the parity test
#: ratchets the count down rather than letting it grow.
NO_HELP = "_(the parser declares no help text for this option)_"

#: What the page prints for an option whose parser hides it from --help
#: (``help=argparse.SUPPRESS``).  The reference page lists every option a
#: door accepts; printing the sentinel itself would put ``==SUPPRESS==``
#: on a public page.
HIDDEN = "_(accepted, but not listed by --help)_"


def portable(text: str) -> str:
    """Strip machine-specific absolute paths out of a help string.

    Some help strings interpolate a resolved path so the terminal shows
    the reader the real file on THEIR box.  Committed to a document that
    every reader shares, the same string names one developer's checkout
    -- so the page would ship a path nobody else has, and would churn in
    the diff on every machine that regenerated it.  The package-relative
    form says the same thing and belongs to no one.

    The tail after the stand-in is spelled POSIX on every platform: the
    interpolated path carries the host's separator, and a page generated
    on Windows read ``<woof package>\\authorities\\...`` where the same
    tree on Linux read ``<woof package>/authorities/...``, so the
    byte-for-byte staleness gate split by platform on a separator alone.
    """

    import woof

    for root, stand_in in (
            (str(Path(woof.__file__).resolve().parent), "<woof package>"),
            (str(REPO_ROOT), "<repository root>")):
        for spelling in (root, root.replace("\\", "/")):
            text = text.replace(spelling, stand_in)
    return _POSIX_TAIL.sub(
        lambda match: match.group(0).replace("\\", "/"), text)


#: A stand-in followed by the path tail the help string interpolated
#: after it: everything up to the first whitespace or table separator.
_POSIX_TAIL = re.compile(r"<(?:woof package|repository root)>[^\s|]*")


def _positional_spelling(action: argparse.Action) -> str:
    """How a positional is written on the command line.

    ``metavar`` when the door declares one, otherwise ``dest`` -- the
    same two argparse falls through for its own usage line, so the page
    and ``--help`` name the argument identically.  ``nargs`` then
    decorates it exactly as a usage line does, because "one config path"
    and "any number of wrfout files" are different instructions and the
    bare name says neither.

    The decoration is spelled here rather than borrowed from argparse's
    formatter: the private formatter's rendering of ``nargs='*'``
    changed between interpreter versions, and this page is committed and
    compared byte for byte, so a borrowed spelling would make the same
    tree generate two different documents on two Python versions.
    """

    name = action.metavar or action.dest
    nargs = action.nargs
    if nargs is None:
        return name
    if nargs == argparse.OPTIONAL:
        return f"[{name}]"
    if nargs == argparse.ZERO_OR_MORE:
        return f"[{name} ...]"
    if nargs == argparse.ONE_OR_MORE:
        return f"{name} [{name} ...]"
    if nargs in (argparse.REMAINDER, argparse.PARSER):
        return f"{name} ..."
    if isinstance(nargs, int):
        return " ".join([name] * nargs)
    return name


def _arguments(parser: argparse.ArgumentParser) -> list[tuple[str, str]]:
    """``(spelling, help)`` for every positional, in command-line order.

    Positionals were omitted from this page for its whole life, and the
    omission hid the most basic fact about the config-driven doors:
    `woof run` was listed with every optional knob it has and nothing
    about the config path it cannot run without, so the page named no
    way to pass a config at all.  A flag nobody can find and an argument
    nobody can find are the same defect.

    Declaration order is kept rather than sorted -- for a positional the
    order IS the calling convention, and `woof import-namelist WPS
    INPUT` sorted alphabetically would instruct a reader wrongly.

    ``choices`` are deliberately not expanded the way an option's are,
    which is the one place this page reads differently from a usage
    line.  A positional's choice list here is a discovered registry
    rather than a fixed vocabulary, so printing it would rewrite this
    committed page whenever the tree gained an unrelated entry; the
    door's own ``--help`` prints the live list, and the page says so.
    """

    rows: list[tuple[str, str]] = []
    for action in parser._actions:
        if action.option_strings:
            continue
        if isinstance(action, argparse._SubParsersAction):
            continue  # the subcommand tree; each branch is its own door
        text = portable(" ".join((action.help or "").replace("%%", "%").split()))
        rows.append((_positional_spelling(action), text or NO_HELP))
    return rows


def _options(parser: argparse.ArgumentParser) -> list[tuple[str, str]]:
    """``(spelling, help)`` for every option except ``--help``."""

    rows: list[tuple[str, str]] = []
    for action in parser._actions:
        names = [o for o in action.option_strings if o.startswith("--")]
        if not names or "--help" in names:
            continue
        spelling = ", ".join(sorted(names))
        if action.choices and not isinstance(action.choices, dict):
            spelling += " {" + ",".join(str(c) for c in action.choices) + "}"
        elif action.metavar:
            spelling += f" {action.metavar}"
        if action.help == argparse.SUPPRESS:
            text = HIDDEN
        else:
            text = portable(" ".join((action.help or "").replace("%%", "%").split()))
        rows.append((spelling, text or NO_HELP))
    return sorted(rows)


def console_scripts() -> dict[str, str]:
    """``[project.scripts]`` from ``pyproject.toml``, name -> target.

    The door list used to be a hand-written literal, and the failure mode
    it has is the one every hand-written mirror of a real table has:
    `woof-member-prep` shipped as an installed console script for a
    whole release with no section on the reference page and no document
    naming any of its options, because adding the entry point and adding
    the line here were two separate acts and only the first happened.

    Reading the same table setuptools reads makes them one act.
    """

    import tomllib

    manifest = REPO_ROOT / "pyproject.toml"
    if not manifest.exists():
        raise SystemExit(
            f"{manifest} is not there, so the console-script table this "
            "page is generated from cannot be read.  This generator is a "
            "repository tool; run it from a source checkout.")
    scripts = tomllib.loads(
        manifest.read_text(encoding="utf-8"))["project"]["scripts"]
    return dict(scripts)


def doors() -> dict[str, argparse.ArgumentParser]:
    """Every documented command door, name -> parser."""

    from woof.cli import build_parser

    # The root parser is a door too: `woof --help-all` is an option of
    # `woof` itself, and a walk that began at the root's children left
    # the one page promising every option without it.
    root = build_parser()
    out: dict[str, argparse.ArgumentParser] = {"woof": root}

    def walk(parser: argparse.ArgumentParser, prefix: str) -> None:
        """Every command under ``parser``, at every depth.

        Recursive rather than one level deep.  ``woof obs`` is the first
        door with a subcommand tree of its own -- `woof obs radar pack
        --file ...` is three levels -- and a one-level walk resolved every
        one of those to the `woof obs` parser, which owns none of their
        flags.  The docs-parity instrument then reported that the European
        radar page passed nine flags `woof obs` does not define, all nine
        of which the command it actually names defines.
        """

        for action in parser._actions:
            if not isinstance(action, argparse._SubParsersAction):
                continue
            for name, sub in sorted(action.choices.items()):
                out[f"{prefix} {name}"] = sub
                walk(sub, f"{prefix} {name}")

    walk(root, "woof")

    def built(module: str, attr: str = "build_parser"):
        loaded = importlib.import_module(module)
        factory = getattr(loaded, attr, None)
        if factory is None:
            raise SystemExit(
                f"{module} is a console-script door and declares no "
                f"{attr}(), so this page cannot read its options and "
                f"every one of them would ship undocumented.  Give the "
                f"module a build_parser() that returns the parser its "
                f"main() uses, or name its factory in PARSER_FACTORY.")
        return factory()

    # Driven by the entry-point table rather than by a literal, so a new
    # console script is a documented door on the day it is installable.
    for name, target in sorted(console_scripts().items()):
        if name in ALIASES:
            continue  # the same program, printed once under its own name
        module = target.split(":")[0]
        if module == "woof.cli":
            continue  # the subcommand walk above IS this door
        out[name] = built(module, PARSER_FACTORY.get(module, "build_parser"))
        walk(out[name], name)
    for name, (module, attr) in sorted(MODE_FLAG_DOORS.items()):
        out[name] = built(module, attr)
    return out


def _cell(text: str) -> str:
    """One table cell's text, with the pipes GFM would read as borders.

    A pipe is escaped in BOTH columns, not just the help one.  The option
    column carries argparse ``metavar`` strings verbatim and a metavar may
    spell its own alternatives with a pipe (``--parent-restart
    PATH|latest``); unescaped, that pipe ends the cell and the row renders
    with its help text in the wrong column and a stray third column after
    it.  GFM reads ``\\|`` as a literal pipe even inside a code span, so
    the escape is invisible in the rendered page.
    """

    return text.replace("|", "\\|")


def render() -> str:
    lines = [
        "# Every option, every door",
        "",
        "The complete command-line surface, read off the parsers "
        "themselves.  It exists because a flag that appears in no "
        "document is a feature nobody can reach: `--parent-namelist` "
        "gated the whole stock-WRF-parent route, `--tiles` gated the "
        "only streamed prepared route, and neither was written down "
        "anywhere.",
        "",
        "`tests/test_docs_extras_agree_with_code.py` holds this page "
        "against the parsers in both directions, so it cannot fall "
        "behind the code and cannot name a flag that was removed.  "
        "Regenerate it with `python -m tools.build_cli_options_doc` "
        "after changing any option.",
        "",
        "Everything is listed with the help text the tool itself "
        "prints.  A door's positional arguments come first, in the "
        "order they are written on the command line and under the "
        "names its `--help` usage line gives them: `[NAME]` is "
        "optional, `NAME [NAME ...]` repeats.  Where an argument is "
        "restricted to a fixed set of values, run that door with "
        "`--help` for the list -- it is read from the tree at run time "
        "rather than pinned here.  `--help` itself is omitted.",
        "",
    ]
    built = doors()
    for name in sorted(built):
        arguments = _arguments(built[name])
        rows = _options(built[name])
        lines.append(f"## `{name}`")
        lines.append("")
        note = NOTES.get(name)
        if note:
            lines.append(note)
            lines.append("")
        if name in ALIASES:
            lines.append(f"Same program as `{ALIASES[name]}`.")
            lines.append("")
        if arguments:
            lines.append("| argument | what it does |")
            lines.append("|---|---|")
            for spelling, text in arguments:
                lines.append(f"| `{_cell(spelling)}` | {_cell(text)} |")
            lines.append("")
        if not rows:
            lines.append("Takes no options of its own.")
            lines.append("")
            continue
        lines.append("| option | what it does |")
        lines.append("|---|---|")
        for spelling, text in rows:
            lines.append(f"| `{_cell(spelling)}` | {_cell(text)} |")
        lines.append("")
    for alias, target in sorted(ALIASES.items()):
        lines.append(f"## `{alias}`")
        lines.append("")
        lines.append(f"The same program as `{target}`, under its other "
                     f"installed name; every option above applies.")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.build_cli_options_doc",
        description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true",
        help="exit 1 if the committed page is stale instead of writing it")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
        if current != text:
            print(f"{DOC} is stale; run python -m tools.build_cli_options_doc")
            return 1
        print(f"{DOC} is current")
        return 0
    DOC.parent.mkdir(parents=True, exist_ok=True)
    # newline="" so `\n` reaches the file as `\n`.  Python's text mode
    # translates it to `\r\n` on Windows, and the repository promises
    # `* -text` -- committed bytes are checkout bytes on every platform.
    # This tool ran on Windows once and put the only CRLF copy of this
    # page into the object database; a generator that re-creates the
    # defect every time it runs is the defect.
    DOC.write_text(text, encoding="utf-8", newline="")
    print(f"wrote {DOC} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
