"""A command a release removed answers with one line, not argparse's usage dump.

THE BREAKAGE THIS PREVENTS: WOOF 1.1.7 removed ``woof local-da``.  Without an
answer, a user following an older page or script gets ``invalid choice`` and
the list of every command, and is left to guess whether the install is
broken.  The door says in one line that the command was removed, in which
version and why, and exits 2.
"""

from __future__ import annotations

import pytest

from woof import cli
from woof.removed_commands import REMOVED_COMMANDS, removed_command


@pytest.mark.parametrize("argv", [
    ["local-da"],
    ["local-da", "--point", "35.5,-90.5", "--epoch", "2025-03-14T22:00:00Z", "--dry-run"],
    ["local-da", "--help"],
    ["--explain", "local-da", "--status", "plan.json"],
])
def test_the_removed_local_da_door_says_so_in_one_line(capsys, argv):
    assert cli.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == REMOVED_COMMANDS["local-da"]
    assert "removed in WOOF 1.1.7" in captured.err
    assert len(captured.err.strip().splitlines()) == 1


def test_a_command_that_exists_is_not_answered_as_removed():
    assert removed_command(["go", "local-da"]) is None
    assert removed_command(["--backend", "cuda"]) is None
    assert removed_command([]) is None


def test_the_removed_door_is_not_a_registered_command():
    parser = cli.build_parser()
    commands = next(action for action in parser._actions
                    if action.__class__.__name__ == "_SubParsersAction")
    assert "local-da" not in commands.choices
