"""Content rules shared by the full export scan and its in-cut hygiene rewrite.

Scan the proposed export, including unchanged files. Findings are handed to
hygiene.plan inside the cut; executable data that cannot be rewritten stays
an explicit finding in that stage, without creating another pass.

This file ships, so it carries one generic rule only.  The rules that name
what must not leave the project's own machines are data, read at run time
from a pattern file that is never packaged: the path in the environment
variable ``HEX_PRIVATE_PATTERNS`` (or ``configure(path)``).  THE BREAKAGE
THIS PREVENTS (source packages of three candidates, found on 2026-11-15):
the rules were written here, split across string literals, so every
source package carried the very names its scans refuse.  Without a pattern
file the scan runs the generic rule alone and says so (``PRIVATE_LOADED``);
the release seal and the export cut refuse to run that way.

Pattern file (JSON): ``content_rules`` is a list of objects with ``kind``
and ``pattern`` (a regular expression, matched case-insensitively) and,
optionally, ``sealed_ok`` (not a hit inside recorded compiler bytes),
``skip_line`` with ``skip_line_suffix`` (a line that also matches it, in a
file with that suffix or one of that list of suffixes, is not a hit), ``historical`` (a line that matches
it is not a hit), ``public_contracts`` (public interoperability names are
blanked first) and ``artifact`` (the rule also reads binary members);
``public_contracts`` lists ``pattern`` objects; ``artifact_rules`` is read
by release_hygiene.py.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path, PurePosixPath

PRIVATE_ENV = "HEX_PRIVATE_PATTERNS"

# The one generic rule: a delivery folder in a path names a person's machine, whatever the project.
GENERIC_RULES = (
    ('delivery path', re.compile(r'Down' r'loads[\\/]'), {}),
)
EXAMPLE_PATH = re.compile(r'[Cc]:[\\/]Users[\\/]someone[\\/]Down' r'loads[\\/]')

RULES: tuple = GENERIC_RULES
PUBLIC_CONTRACT_ALLOWLIST: tuple = ()
PRIVATE: dict = {}
PRIVATE_LOADED = False
PRIVATE_SOURCE: str | None = None


def configure(path: str | os.PathLike | None) -> bool:
    """Load the private pattern file at ``path`` (None: the generic rule alone). Returns whether one loaded.

    An unreadable or malformed file raises: a scan that silently lost its private rules would pass what it exists
    to refuse."""
    global RULES, PUBLIC_CONTRACT_ALLOWLIST, PRIVATE, PRIVATE_LOADED, PRIVATE_SOURCE
    if path is None or str(path) == "":
        RULES, PUBLIC_CONTRACT_ALLOWLIST, PRIVATE = GENERIC_RULES, (), {}
        PRIVATE_LOADED, PRIVATE_SOURCE = False, None
        return False
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        rules = tuple((row["kind"], re.compile(row["pattern"], re.I), row) for row in raw.get("content_rules", ()))
        public = tuple((row.get("group", ""), re.compile(row["pattern"], re.I), row.get("reason", ""))
                       for row in raw.get("public_contracts", ()))
        for row in raw.get("content_rules", ()):
            for key in ("skip_line", "historical"):
                if row.get(key):
                    re.compile(row[key], re.I)
    except (OSError, ValueError, KeyError, TypeError, re.error) as error:
        raise RuntimeError(f"the private pattern file {path} cannot be read: {error}") from error
    RULES, PUBLIC_CONTRACT_ALLOWLIST, PRIVATE = GENERIC_RULES + rules, public, raw
    PRIVATE_LOADED, PRIVATE_SOURCE = True, str(path)
    return True


configure(os.environ.get(PRIVATE_ENV))


def private_recipe_text(line: str) -> str:
    for _, pattern, _ in PUBLIC_CONTRACT_ALLOWLIST:
        line = pattern.sub("public_contract", line)
    return line


def _sealed_path(path: str) -> bool:
    return any(part in path for part in ("compiler-recipes", "/mpasphys-maths/")) or path.endswith("rkind_libm.py")


def scan(path: str, data: bytes) -> list[tuple[str, int, str, str]]:
    """(path, line, kind, context), with one finding per rule and line."""
    if PurePosixPath(path).suffix == '.pyc' or '__pycache__' in PurePosixPath(path).parts:
        return [(path, 0, 'compiled Python', path)]
    lines = [(0, path)]
    if b'\0' not in data[:4096]:
        lines.extend(enumerate(data.decode('utf-8', 'replace').splitlines(), 1))
    sealed = _sealed_path(path)
    hits = []
    for number, line in lines:
        for kind, pattern, options in RULES:
            checked = private_recipe_text(line) if options.get("public_contracts") else line
            if not pattern.search(checked):
                continue
            if sealed and options.get("sealed_ok"):
                continue
            skip = options.get("skip_line")
            suffix = options.get("skip_line_suffix", "")
            if skip and path.endswith(tuple(suffix) if isinstance(suffix, list) else suffix) and re.search(skip, line, re.I):
                continue
            if options.get("historical") and re.search(options["historical"], line, re.I):
                continue
            if kind == 'delivery path' and path.startswith(('tests/', 'tree/tests/')):
                # Only the made-up Windows example used to test path refusal.
                if not pattern.search(EXAMPLE_PATH.sub('', line)):
                    continue
            hits.append((path, number, kind, line.strip()[:220]))
    return hits


def artifact_rules() -> list[tuple[str, re.Pattern[bytes], dict]]:
    """The pattern file's byte rules for release artifacts, and the content rules marked ``artifact``."""
    rows = []
    for row in PRIVATE.get("artifact_rules", ()):
        flags = 0 if row.get("case_sensitive") else re.I
        rows.append((row["kind"], re.compile(row["pattern"].encode(), flags), row))
    for kind, pattern, row in RULES:
        if row.get("artifact"):
            rows.append((kind, re.compile(pattern.pattern.encode(), re.I), row))
    return rows


def sealed_exception(path: str, kind: str) -> bool:
    """Temporary patch exception for exact recorded compiler and arithmetic bytes."""
    if not _sealed_path(path):
        return False
    return any(row.get("sealed_ok") and row["kind"] == kind
               for row in tuple(PRIVATE.get("artifact_rules", ())) + tuple(PRIVATE.get("content_rules", ())))
