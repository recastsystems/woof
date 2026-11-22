"""``woof sources``: the human door onto the source registry.

Every fetch/prep/run refusal that has to say "this WOOF-ArWen cannot do that
with that source" needs somewhere to send the reader, and until now the
only place the whole registry was legible was ``woof run-plan
--sources`` -- a JSON query flag on the machine-facing execution front
door.  ``woof.fetch_routes`` said so out loud and got it wrong: its
no-route refusal ended ``see: `woof sources` for what each registered
source can do today``, and there was no such command.  The real parser
answered ``invalid choice: 'sources'`` at exit 2, so the one sentence a
stuck reader was handed sent them nowhere.

Handing that reader a JSON query flag instead would have been the other
half of the same mistake (ship-only-what-users-can-reach), so the
command in the sentence is the command that now exists.  It is a VIEW,
not a second source of truth: every fact printed here is copied out of
the document :func:`woof.runplan.source_inventory` builds, which reads
``woof.source_adapters._ADAPTERS`` and the chemistry source registry.  ``--json``
re-emits that document byte for byte, so the human and machine
spellings cannot drift.

Nothing in this module knows a model's name.  The listing's columns are
registry FIELD names, the detail view walks whatever keys the row
carries, and a row appended to the registry appears in both with no
edit here -- which is the arbitrary acceptance test, and what
``tests/test_sources_command.py`` grafts a row to prove.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Mapping, Sequence

#: Registry field names -- NOT model names -- and the heading each one
#: gets in the listing.  A row appended to the registry is printed by
#: this same table; a column is added by naming another registry field.
#: ``(heading, path)`` where ``path`` walks the row document.
_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("id", ("source_id",)),
    ("kind", ("source_kind",)),
    ("family", ("file_family",)),
    ("maturity", ("maturity", "status")),
    ("runnable", ("maturity", "runnable")),
    ("intent", ("run_plan", "intent_chain")),
    ("fetch", ("fetch", "kind")),
    ("posting", ("fetch", "posting", "shape")),
)


def _walk(row: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value) or "-"
    return str(value) or "-"


def _rows_by_id(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("source_id")): row
            for row in (*document.get("sources", ()), *document.get("composition_sources", ()))
            if row.get("source_id")}


def _resolve(document: Mapping[str, Any], wanted: str) -> Mapping[str, Any]:
    """The row a reader's spelling names, id or alias, or a refusal.

    Alias resolution reads the row's own ``aliases`` list rather than a
    lookup table here, for the same reason the columns are field names:
    a source that grows an alias grows it in the registry.
    """

    rows = _rows_by_id(document)
    row = rows.get(wanted)
    if row is not None:
        return row
    for candidate in rows.values():
        if wanted in {str(alias) for alias in candidate.get("aliases", ())}:
            return candidate
    count = document.get("source_count")
    known = len(rows) if count is None else count
    raise ValueError(
        f"{wanted!r} is not a registered source id or alias.\n"
        f"  why: the source registry carries {known} rows and this is not "
        "one of them, so nothing in this WOOF-ArWen could fetch, prepare or run "
        "it -- a typo here becomes a refusal three commands later.\n"
        "  remedy: run `woof sources` with no argument for the whole "
        "list, then name a row's id.")


def _format_listing(document: Mapping[str, Any]) -> str:
    rows = [*document.get("sources", ()), *document.get("composition_sources", ())]
    cells = [[_cell(_walk(row, path)) for _heading, path in _COLUMNS]
             for row in rows]
    headings = [heading for heading, _path in _COLUMNS]
    widths = [len(heading) for heading in headings]
    for line in cells:
        widths = [max(width, len(value))
                  for width, value in zip(widths, line)]

    def _render(values: Sequence[str]) -> str:
        return "  " + "  ".join(value.ljust(width)
                                for value, width in zip(values, widths)).rstrip()

    total = document.get("source_count")
    runnable = document.get("runnable_source_count")
    lines = [
        f"woof sources: {total if total is not None else len(rows)} "
        f"registered, {runnable if runnable is not None else '?'} runnable "
        f"on this release "
        f"(registry {document.get('registry_schema')}, "
        f"woof {document.get('gpuwm_version')})",
        _render(headings),
    ]
    if document.get("composition_sources"):
        lines.append(f"plus {len(document['composition_sources'])} composition supplements (event proof is separate)")
    lines.extend(_render(line) for line in cells)
    for label, key in (("readiness", "readiness_rule"),
                       ("certification", "certification_rule")):
        if document.get(key):
            lines.append(f"{label}: {document[key]}")
    if document.get("composition_error"):
        lines.append(f"composition registry error: {document['composition_error']}")
    if document.get("error"):
        # A query mode reports; it does not raise.  The envelope's own
        # error is printed rather than swallowed, because a short list
        # that looks complete is worse than a list that says it is not.
        lines.append(f"registry error: {document['error']}")
    lines.append(
        "one row in full: `woof sources ID`; the same facts as JSON: "
        "`woof sources --json` (identical to `woof run-plan --sources`)")
    return "\n".join(lines)


def _format_value(value: Any, indent: str) -> list[str]:
    """One registry value, printed by SHAPE and never by field name."""

    if isinstance(value, Mapping):
        lines = []
        for key, nested in value.items():
            rendered = _format_value(nested, indent + "  ")
            if len(rendered) == 1:
                lines.append(f"{indent}{key}: {rendered[0].strip()}")
            else:
                lines.append(f"{indent}{key}:")
                lines.extend(rendered)
        return lines or [f"{indent}(none)"]
    if isinstance(value, (list, tuple)):
        if not value:
            return [f"{indent}(none)"]
        if all(not isinstance(item, (Mapping, list, tuple)) for item in value):
            return [f"{indent}{', '.join(str(item) for item in value)}"]
        lines = []
        for item in value:
            lines.extend(_format_value(item, indent + "  "))
        return lines
    # A registry value may itself be a multi-line refusal -- the fetch
    # route's own sentence, relayed verbatim.  Its continuation lines
    # are indented to this block so the row stays readable instead of
    # spilling back to column zero and reading as a second field.
    parts = _cell(value).splitlines() or ["-"]
    return [f"{indent}{parts[0]}"] + [f"{indent}  {part.strip()}"
                                      for part in parts[1:]]


def _format_row(row: Mapping[str, Any], document: Mapping[str, Any]) -> str:
    lines = [f"woof sources: {row.get('source_id')} "
             f"(registry {document.get('registry_schema')}, "
             f"woof {document.get('gpuwm_version')})"]
    for key, value in row.items():
        rendered = _format_value(value, "    ")
        if len(rendered) == 1:
            lines.append(f"  {key}: {rendered[0].strip()}")
        else:
            lines.append(f"  {key}:")
            lines.extend(rendered)
    return "\n".join(lines)


def sources_main(args: argparse.Namespace) -> int:
    """``woof sources [ID] [--json]``.

    Exit codes: 0 on a printed listing or row; 2 on a refusal, which the
    front door's own boundary prints (this raises ``ValueError``, the
    convention every refusal in this product follows).
    """

    import contextlib

    from woof.runplan import source_inventory

    # The registry imports print on some rows, and in --json mode stdout
    # is the machine channel.  Same redirect, same reason, as
    # `woof run-plan --sources`.
    with contextlib.redirect_stdout(sys.stderr):
        document = source_inventory()

    wanted = getattr(args, "source", None)
    if getattr(args, "json", False):
        if wanted is not None:
            # Refuse BEFORE printing: a machine consumer that asked for
            # one row must not receive the whole registry and mistake it
            # for an answer.
            row = _resolve(document, wanted)
            document = dict(document)
            document["sources"] = [row]
            # ``source_count`` stays the REGISTRY's count, because that
            # is what it means.  This says the rows were narrowed, so a
            # consumer never reads a one-row reply as a one-row
            # registry.
            document["requested_source"] = wanted
        print(json.dumps(document, indent=2, sort_keys=True))
        return 0
    if wanted is None:
        print(_format_listing(document))
        return 0
    print(_format_row(_resolve(document, wanted), document))
    return 0


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    """Register the human-facing registry door."""

    parser = subparsers.add_parser(
        "sources",
        help="list every registered forcing source and what this "
             "release can do with each one -- the human view of the "
             "same registry `woof run-plan --sources` serves as JSON")
    parser.add_argument(
        "source", nargs="?", default=None, metavar="ID",
        help="print ONE row in full, named by its registry id or any "
             "alias it declares (omit for the listing)")
    parser.add_argument(
        "--json", action="store_true",
        help="emit the registry document instead of the table -- the "
             "gpuwm.run-plan.sources.v1 schema, narrowed to the one row "
             "when ID is given")
    parser.set_defaults(func=sources_main)


__all__ = ["register_cli", "sources_main"]
