"""Command remedies through the invoking front door, including embedded hex.

An embedding CLI sets argv[0] to its component command (for example
``weather hex``). Standalone executables whose helpers belong to another
distribution declare that relationship in the table below.
"""

from pathlib import Path
import re
import sys


FRONT_DOORS = {
    "woof hex": ("woof hex", "woof"),
    "woof": ("woof hex", "woof"),
}
# Helper programs are shared by distributions. Only command tokens in their
# diagnostics are translated; file paths and the rest of the refusal stay put.
HELPER_REMEDIES = ("fetch-geog",)
# The prebuilt helpers (the hex release's archive) print the commands of the
# distribution they were built for. The rename moved the table above to woof, so
# the translation looked for "woof fetch-geog" and never fired: a hex run without
# its mesh geography told the user to run a command WOOF does not install.
HELPER_PROGRAMS = ("gpuwm-hex", "gpuwm", *FRONT_DOORS)


def command_name(*, engine: bool = False) -> str:
    invoked = Path(sys.argv[0]).name.removesuffix(".exe")
    if invoked in FRONT_DOORS:
        model, parent = FRONT_DOORS[invoked]
    elif invoked.endswith(" hex"):
        model, parent = invoked, invoked.removesuffix(" hex")
    else:
        model, parent = FRONT_DOORS["woof hex"]
    return parent if engine else model


def helper_remedies(text: str) -> str:
    for command in HELPER_REMEDIES:
        programs = "|".join(re.escape(name) for name in sorted(HELPER_PROGRAMS, key=len, reverse=True))
        text = re.sub(r"(?<![\w/.-])(?:" + programs + r") " + re.escape(command) + r"\b",
                      lambda _: f"{command_name(engine=True)} {command}", text)
    return text
