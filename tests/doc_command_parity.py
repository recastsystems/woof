"""One extractor, one door registry, for every docs-to-code binding.

There were three partial implementations of the same idea in this tree
before this module existed, each correct and each blind to what the
others saw:

* ``test_verification_recipe_flags.py`` extracts the fenced block under
  VERIFICATION.md's reproduce heading and feeds each command to the
  parser that would receive it.  Strongest check, narrowest scope: one
  section of one document.
* ``test_certify_verdict.py`` binds CERTIFICATION.md's condition table
  to the ``CONDITIONS`` the code declares, in the direction
  docs-must-match-code.
* ``test_docs_extras_agree_with_code.py`` holds every documented install
  line against the extras ``pyproject.toml`` declares, in both
  directions.

They agree on the shape -- read a document, read the code, hold the two
against each other -- and disagreed on the mechanism, so a fourth was
not wanted.

There are two BINDING KINDS in play, and it is worth being exact about
which one this module serves, because putting them in one abstraction
would help nobody:

* **Command bindings** -- a document tells a reader to RUN something,
  and the parser is the authority on what runs.  This module is that
  mechanism: the door registry, the fragment reader, and the resolver
  that decides whether a piece of a document is a command at all.  Its
  consumers are ``test_docs_extras_agree_with_code.py`` (every
  user-facing page, membership) and ``test_verification_recipe_flags.py``
  (one recipe, full ``parse_args``).  A consumer may be as strict as it
  likes on top of the shared floor; the recipe's parse catches a missing
  required argument that membership cannot, and the two are asserted to
  resolve the same door from the same registry.
* **Enumeration bindings** -- the code declares a SET and a document
  presents a table of it as complete.  ``test_certify_verdict.py``'s
  condition table and this file's consumers' ``[case_data]`` key rules
  are both of that kind: read the set from the code, read the table from
  the page, compare.  They need no extractor and deliberately do not
  import one.

A new binding belongs in whichever of those two it is.  A new
mechanism needs a reason neither covers.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A shell prompt a document may print before the command itself.
PROMPT = re.compile(r"^\s*(?:\$|>|PS[^>]*>)\s+")

#: A long option, not counting a bare ``--`` or a negative number.
FLAG = re.compile(r"(?<![\w-])(--[A-Za-z][\w-]*)")

#: How a document spells a door argparse knows by another name.
DOC_SPELLINGS = {
    "python -m woof.prepared_single_domain_forecast":
        "woof-prepared-forecast",
    "python -m woof.prepared_domain_tree_forecast":
        "woof-prepared-tree-forecast",
    "woof-wrf-init": "rw-wps",
}

#: A first-position flag that selects a different program, and the door
#: whose parser owns that program's options.
MODE_DOORS = {
    "--materialize-authorities":
        "woof-prepared-forecast --materialize-authorities",
}


def doors() -> dict:
    """Every documented door, name -> argparse parser.

    Single-sourced from ``tools.build_cli_options_doc`` so the generated
    reference page, this module's consumers and the generator itself
    cannot disagree about what a door is.
    """

    from tools.build_cli_options_doc import doors as _doors
    return _doors()


def door_options(parser) -> set[str]:
    """Every long option a parser defines, ``--help`` included."""

    out: set[str] = set()
    for action in parser._actions:
        out.update(o for o in action.option_strings if o.startswith("--"))
    return out


def code_fragments(text: str):
    """``(lineno, fragment)`` for every fenced line and inline span."""

    inline = re.compile(r"`([^`\n]+)`")
    fence = re.compile(r"^\s*(?:```|~~~)")
    inside = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if fence.match(line):
            inside = not inside
            continue
        if inside:
            yield lineno, line
        else:
            for match in inline.finditer(line):
                yield lineno, match.group(1)


def resolve_door(body: str, known: dict):
    """The door a fragment invokes, or ``None`` if it is not one.

    A fragment counts as an invocation only when it STARTS with a door
    name, after an optional shell prompt.  Prose that merely contains
    the program name -- "woof implements bl_mynn_mixlength=1 only" --
    is not a command and is not checked, which is why this rule needs
    no allowlist of English words.
    """

    body = PROMPT.sub("", body).strip()
    for spelling, canonical in DOC_SPELLINGS.items():
        if body == spelling or body.startswith(spelling + " "):
            # A MODE flag selects a different program with its own
            # option set, so the door is not decided by the program
            # name alone.
            for mode, mode_door in MODE_DOORS.items():
                if mode in body.split() and mode_door in known:
                    return mode_door, body
            return canonical, body
    if body == "woof" or body.startswith("woof "):
        rest = body[len("woof"):].strip()
        token = rest.split()[0] if rest.split() else ""
        # A placeholder (`woof <command>`, `woof SUBCOMMAND`) is not a
        # claim that a subcommand exists.
        if not token or not re.fullmatch(r"[a-z][a-z0-9-]*", token):
            return None
        # LONGEST known door, not the first one.  `woof obs` owns a
        # subcommand tree -- `woof obs radar pack --file ...` is three
        # levels -- and stopping at the first token resolved every one of
        # those to the `woof obs` parser, which defines none of their
        # flags.  Descend only through tokens the registry actually knows,
        # so an argument that happens to be a bare word (`woof render
        # wrfout_d01`) still resolves to the door that takes it.
        door = f"woof {token}"
        # An UNKNOWN first token is still returned, unchanged: a document
        # naming a subcommand the parser does not have is exactly what the
        # consumer exists to report, and swallowing it here would make that
        # test silently green.  Only the DESCENT is gated on the registry.
        for word in (rest.split()[1:] if door in known else []):
            if not re.fullmatch(r"[a-z][a-z0-9-]*", word):
                break
            if f"{door} {word}" not in known:
                break
            door = f"{door} {word}"
        return door, body
    for name in known:
        if name.startswith("woof ") or " " in name:
            continue
        if body == name or body.startswith(name + " "):
            return name, body
    return None


def user_facing_docs() -> list[Path]:
    """The pages a user is expected to read.

    README, the top level of ``docs/``, and ``docs/public/`` -- and
    deliberately NOT ``docs/internal/``, ``docs/ports/`` or
    ``docs/public/validation/``, which are working records and may
    legitimately quote a command as it stood on the day.  The release
    development records (``docs/X.Y.Z-development-record.md``) are the same
    kind of page, kept verbatim: 2.7.5's quotes ``woof local-da``, which
    3.0.3 removed.
    """

    import subprocess

    out = subprocess.run(["git", "ls-files", "-z", "docs", "README.md"],
                         cwd=REPO_ROOT, capture_output=True, check=True)
    found = []
    for name in out.stdout.decode("utf-8").split("\0"):
        if not name.endswith(".md"):
            continue
        if name == "README.md":
            found.append(REPO_ROOT / name)
        elif name.startswith("docs/public/") and name.count("/") == 2:
            found.append(REPO_ROOT / name)
        elif name.startswith("docs/") and name.count("/") == 1:
            if name.endswith("-development-record.md"):
                continue
            found.append(REPO_ROOT / name)
    return sorted(found)
