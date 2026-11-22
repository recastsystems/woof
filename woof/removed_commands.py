"""Commands an earlier release had and this one does not.

A user who types one gets one plain line instead of argparse's list of every
command.  ``woof.cli`` answers the command; ``gpuwm.tui_cli`` answers the
terminal's old opt-in switch with the same words.  Kept free of imports so
any door can read it.
"""

from __future__ import annotations

REMOVED_COMMANDS = {
    "local-da": ("woof local-da was removed in WOOF 1.1.7: it was "
                 "experimental and is no longer maintained."),
}


def removed_command(tokens: list[str]) -> str | None:
    """The removal line when the command word names a removed command."""
    command = next((token for token in tokens if not token.startswith("-")), None)
    return REMOVED_COMMANDS.get(command)


__all__ = ["REMOVED_COMMANDS", "removed_command"]
