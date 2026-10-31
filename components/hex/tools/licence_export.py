"""Apply and check the licence holds of tools/licence_holds.json.

THE BREAKAGE THIS PREVENTS.  A port whose upstream terms forbid sale
(NOTICE section 4: UCAR's Noah-MP v5.0) must not reach the public
distribution, and leaving it out must not break what remains.  Removing a module by hand
breaks things silently: a tier list still names its test (tests/
test_tier_membership.py then fails only on the export), or another module
still imports it (the export then fails on import, not at the selection that
needs it).  This tool removes the held paths and the tier rows that name
them, and then checks that nothing left imports a held module.

    python tools/licence_export.py apply SRC_TREE DST_TREE
        copy SRC_TREE (the distribution root, the directory holding
        pyproject.toml) to DST_TREE without the held paths and tier rows
    python tools/licence_export.py check TREE
        verify TREE: no held path present, no tier row naming a missing
        file, no import of a held module; exit 1 and name each finding

Registry rows that name a held module by string (the MPAS suite table's
cuda_entries) are not findings: device_pools_v842.register_cuda_slots
refuses that selection by name, and the native backend still runs the
scheme from the user's own MPAS build.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from pathlib import Path

HOLDS = Path(__file__).resolve().parent / "licence_holds.json"
TIER_LISTS = ("tools/battery/cpu_files.txt", "tools/battery/asset_gates.txt",
              "tools/battery/gpu_gates.txt")
SCANNED = ("src", "tests", "tools")


def load_holds(path: Path = HOLDS) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "gpuwm-hex.licence-holds/v1":
        raise SystemExit(f"{path}: unknown schema {data.get('schema')!r}")
    return data["holds"]


def held_paths(holds: list[dict]) -> list[str]:
    return [p for hold in holds for p in hold["paths"]]


def is_held(relative: str, paths: list[str]) -> bool:
    return any(relative == p or (p.endswith("/") and relative.startswith(p))
               for p in paths)


def held_modules(paths: list[str]) -> set[str]:
    """Fully qualified imports, so another package's same-named module is allowed."""
    return {p[4:-3].replace("/", ".").removesuffix(".__init__")
            for p in paths if p.startswith("src/") and p.endswith(".py")}


def imports(path: Path, relative: str) -> list[tuple[int, str]]:
    module = relative.removeprefix("src/")[:-3].split("/")
    package = module[:-1]
    found = []
    for node in ast.walk(ast.parse(path.read_bytes(), filename=relative)):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[:len(package) - node.level + 1] if node.level else []
            if node.module:
                base.extend(node.module.split("."))
            name = ".".join(base)
            found.append((node.lineno, name))
            found.extend((node.lineno, ".".join(base + [alias.name])) for alias in node.names)
    return found


def apply(src: Path, dst: Path, holds: list[dict]) -> dict:
    paths = held_paths(holds)
    if dst.exists():
        raise SystemExit(f"{dst} exists; give a new directory")
    removed = []

    def ignore(directory: str, names: list[str]) -> set[str]:
        skip = set()
        base = Path(directory).relative_to(src).as_posix()
        for name in names:
            relative = name if base == "." else f"{base}/{name}"
            probe = relative + ("/" if (Path(directory) / name).is_dir() else "")
            if name == "__pycache__" or is_held(probe, paths):
                skip.add(name)
                if name != "__pycache__":
                    removed.append(probe)
        return skip

    shutil.copytree(src, dst, ignore=ignore, symlinks=True)
    dropped_rows = []
    for tier in TIER_LISTS:
        path = dst / tier
        if not path.is_file():
            continue
        kept = []
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
            entry = line.strip()
            if entry and not entry.startswith("#") and is_held(entry, paths):
                dropped_rows.append(f"{tier}: {entry}")
                continue
            kept.append(line)
        path.write_text("".join(kept), encoding="utf-8")
    return {"removed": sorted(removed), "tier_rows_dropped": dropped_rows}


def check(tree: Path, holds: list[dict]) -> list[str]:
    paths = held_paths(holds)
    findings = []
    for p in paths:
        if (tree / p.rstrip("/")).exists():
            findings.append(f"held path present: {p}")
    for tier in TIER_LISTS:
        path = tree / tier
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = line.strip().split()[0] if line.strip() else ""
            if entry.endswith(".py") and not entry.startswith("#") and not (tree / entry).exists():
                findings.append(f"{tier} names a missing file: {entry}")
    modules = held_modules(paths)
    for top in SCANNED:
        for path in sorted((tree / top).rglob("*.py")):
            relative = path.relative_to(tree).as_posix()
            try:
                names = imports(path, relative)
            except (SyntaxError, ValueError) as error:
                findings.append(f"cannot inspect imports: {relative}: {error}")
                continue
            for line, name in sorted(set(names)):
                if any(name == held or name.startswith(held + ".") for held in modules):
                    findings.append(f"imports a held module: {relative}:{line}")
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("apply")
    a.add_argument("src", type=Path)
    a.add_argument("dst", type=Path)
    c = sub.add_parser("check")
    c.add_argument("tree", type=Path)
    args = parser.parse_args(argv)
    holds = load_holds()
    if args.command == "apply":
        report = apply(args.src.resolve(), args.dst.resolve(), holds)
        findings = check(args.dst.resolve(), holds)
        report["findings"] = findings
        print(json.dumps(report, indent=1))
        return 1 if findings else 0
    findings = check(args.tree.resolve(), holds)
    print(json.dumps({"findings": findings}, indent=1))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
