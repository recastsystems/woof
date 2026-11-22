"""The commands the first-run pages print are ones this release's doors accept, with the file names they write.

THE BREAKAGE (the 0.6.1 User Manual's quickstart, typed as printed on a fresh Ubuntu 26.04): its
``pip download --no-binary :all: woof hex`` built every dependency from source and stopped after 152 s in cmake's
build ("Could not find a make program"), and its ``woof hex render --history work/fc-01/cuda-history....nc``
named a file the default forecast route does not write (that route writes ``history.*.nc``), exit 2.  The page
said every command in it had been run before it was printed.

These tests read the printed commands only; running them is a separate check before each release.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
#: The command's words. Compared word by word with shlex's tokens, so a distribution that spells the command in
#: two words (a renamed fold of this package) is read too; one string never equals two tokens.
DOOR = "woof hex".split()
FIRST_RUN_PAGES = (
    "README.md",
    "docs/manual/02-quickstart.md",
    "docs/differentiable/FIRST-HOUR.md",
    "docs/differentiable/README.md",
)
FENCE = re.compile(r"^```(sh|bash)\s*\n(.*?)^```", re.S | re.M)


def _commands(page: Path):
    """Each simple command of the page's sh and bash blocks: (block text, tokens)."""

    for block in FENCE.finditer(page.read_text(encoding="utf-8")):
        text = block.group(2).replace("\\\n", " ")
        for line in text.splitlines():
            try:
                tokens = shlex.split(line, comments=True)
            except ValueError:
                continue
            command: list[str] = []
            for token in tokens + ["&&"]:
                if token in ("&&", ";", "||", "|"):
                    if command:
                        yield block.group(2), command
                    command = []
                else:
                    command.append(token)


def _all_pages():
    return [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


@pytest.mark.parametrize("page", FIRST_RUN_PAGES)
def test_every_printed_door_command_parses(page):
    from woof.hex.cli import build_parser, join_negative_coordinates

    checked = 0
    for _, tokens in _commands(ROOT / page):
        if tokens[:len(DOOR)] != DOOR or len(tokens) <= len(DOOR) or tokens[len(DOOR)].startswith("-"):
            continue
        if any("..." in token for token in tokens):
            continue  # a shape, not a command (the page says so beside it)
        argv = join_negative_coordinates(tokens[len(DOOR):])
        try:
            build_parser(argv[0]).parse_args(argv)
        except SystemExit as stop:
            assert stop.code == 0, f"{page}: the door refuses its own printed command: {' '.join(tokens)}"
        checked += 1
    assert checked, f"{page} prints no woof hex command"


INLINE_PIP = re.compile(r"`(pip [^`\n]+)`")


def test_no_printed_pip_line_builds_every_dependency_from_source():
    for page in _all_pages():
        inline = [span.split() for span in INLINE_PIP.findall(page.read_text(encoding="utf-8"))]
        for tokens in [tokens for _, tokens in _commands(page)] + inline:
            if tokens[:1] == ["pip"] and ":all:" in tokens:
                assert "--no-deps" in tokens, (
                    f"{page.relative_to(ROOT)}: {' '.join(tokens)} builds every dependency from source")


def test_every_printed_render_names_a_history_file_the_default_route_writes():
    for page in _all_pages():
        for block, tokens in _commands(page):
            if tokens[:len(DOOR) + 1] != [*DOOR, "render"] or "arwen_mpas_column" in block:
                continue
            for flag, value in zip(tokens, tokens[1:]):
                if flag == "--history":
                    assert not Path(value).name.startswith("cuda-history."), (
                        f"{page.relative_to(ROOT)}: {value}: the default (MPAS) route writes history.*.nc")


def test_every_printed_render_of_several_frames_names_the_run_start():
    """THE BREAKAGE: the transport products page's render of ``RUN/history.*.nc`` had no ``--simulation-start``;
    every frame was filed as lead 0, so three frames rendered 21 pictures and 7 were kept, under exit 0."""

    for page in _all_pages():
        for _, tokens in _commands(page):
            if tokens[:len(DOOR) + 1] != [*DOOR, "render"]:
                continue
            frames = [value for flag, value in zip(tokens, tokens[1:]) if flag == "--history"]
            if any("*" in value for value in frames) or "--series" in tokens:
                assert "--simulation-start" in tokens, (
                    f"{page.relative_to(ROOT)}: {' '.join(tokens)} files every frame as lead 0")
