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

tools/export-rules.txt, when present, names paths the export leaves out
(private prefixes with the paths under them that ship, and never paths); apply
leaves them out and drops the tier rows that name a never path, and check
refuses an export that carries one, a module that imports one, or a ship rule
that names nothing.

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


RULES = Path(__file__).resolve().parent / "export-rules.txt"


def load_export_rules(path: Path = RULES) -> dict:
    """tools/export-rules.txt: private prefixes, the ship lines under them, and never paths."""
    rules = {"private": [], "ship": [], "never": []}
    if not path.is_file():
        return rules
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        kind, _, target = line.partition(" ")
        target = target.strip()
        if kind not in rules or not target or target.startswith("/") or ".." in target.split("/"):
            raise SystemExit(f"{path}:{number}: cannot read rule {line!r}")
        rules[kind].append(target)
    for ship in rules["ship"]:
        if not any(_under(ship, prefix) for prefix in rules["private"]):
            raise SystemExit(f"{path}: ship {ship} is not under a private prefix")
        if any(_under(ship, never) or _under(never, ship) for never in rules["never"]):
            raise SystemExit(f"{path}: ship {ship} overlaps a never path")
    return rules


def _under(relative: str, prefix: str) -> bool:
    """relative (a file, or a directory with a trailing /) is prefix or lies under it."""
    if prefix.endswith("/"):
        return relative.startswith(prefix) or relative == prefix.rstrip("/")
    return relative == prefix or relative.rstrip("/") == prefix


def rule_excluded(relative: str, rules: dict) -> bool:
    """True when the export leaves this path out. A directory is given with a trailing /."""
    if any(_under(relative, never) for never in rules["never"]):
        return True
    if not any(_under(relative, prefix) for prefix in rules["private"]):
        return False
    if any(_under(relative, ship) for ship in rules["ship"]):
        return False
    # a directory that holds a shipped path is walked, its other members are left out one by one
    return not (relative.endswith("/") and any(ship.startswith(relative) for ship in rules["ship"]))


def held_paths(holds: list[dict]) -> list[str]:
    return [p for hold in holds for p in hold["paths"]]


def is_held(relative: str, paths: list[str]) -> bool:
    return any(relative == p or (p.endswith("/") and relative.startswith(p))
               for p in paths)


def held_modules(paths: list[str]) -> set[str]:
    """Fully qualified imports, so another package's same-named module is allowed."""
    modules = {p[4:-3].replace("/", ".").removesuffix(".__init__")
               for p in paths if p.startswith("src/") and p.endswith(".py")}
    # a left-out package directory (src/hexcore/ad/) is its package
    return modules | {p[4:].rstrip("/").replace("/", ".") for p in paths if p.startswith("src/") and p.endswith("/")}


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


def never_paths(rules: dict) -> list[str]:
    return list(rules.get("never", []))


def apply(src: Path, dst: Path, holds: list[dict], rules: dict | None = None) -> dict:
    rules = load_export_rules() if rules is None else rules
    paths = held_paths(holds) + never_paths(rules)
    private_dropped = []
    if dst.exists():
        raise SystemExit(f"{dst} exists; give a new directory")
    removed = []

    def ignore(directory: str, names: list[str]) -> set[str]:
        skip = set()
        base = Path(directory).relative_to(src).as_posix()
        for name in names:
            relative = name if base == "." else f"{base}/{name}"
            probe = relative + ("/" if (Path(directory) / name).is_dir() else "")
            generated = name in {"__pycache__", "target", "build", "dist", ".pytest_cache", ".git"} or name.endswith(".egg-info")
            if generated or is_held(probe, paths):
                skip.add(name)
                if not generated:
                    removed.append(probe)
            elif rule_excluded(probe, rules):
                skip.add(name)
                private_dropped.append(probe)
        return skip

    shutil.copytree(src, dst, ignore=ignore, symlinks=True)
    dropped_rows = []
    for tier in TIER_LISTS:
        path = dst / tier
        if not path.is_file():
            continue
        kept = []
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        for number, line in enumerate(lines):
            entry = line.strip()
            if entry and not entry.startswith("#") and is_held(entry, paths):
                dropped_rows.append(f"{tier}: {entry}")
                # a comment that introduced only this row goes with it (the next line opens a new group)
                following = lines[number + 1].strip() if number + 1 < len(lines) else ""
                if not following or following.startswith("#"):
                    while kept and kept[-1].strip().startswith("#"):
                        dropped_rows.append(f"{tier}: {kept.pop().strip()}")
                continue
            kept.append(line)
        path.write_text("".join(kept), encoding="utf-8")
    return {"removed": sorted(removed), "tier_rows_dropped": dropped_rows,
            "private_dropped": len(private_dropped)}


def check_rules(tree: Path, rules: dict) -> list[str]:
    """An export carries no never path and no unshipped private path, and every ship line names something."""
    findings = []
    for prefix in rules["private"]:
        base = tree / prefix.rstrip("/")
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_dir() and not path.is_symlink():
                continue
            relative = path.relative_to(tree).as_posix()
            if rule_excluded(relative, rules):
                findings.append(f"private path present: {relative}")
    for ship in rules["ship"]:
        # only where the tree carries the private directory at all (a tree without it is not this distribution)
        if not any(_under(ship, prefix) and (tree / prefix.rstrip("/")).is_dir() for prefix in rules["private"]):
            continue
        if not (tree / ship.rstrip("/")).exists() and not (tree / ship.rstrip("/")).is_symlink():
            findings.append(f"ship rule names nothing: {ship}")
    return findings


def check(tree: Path, holds: list[dict]) -> list[str]:
    rules = load_export_rules()
    paths = held_paths(holds) + never_paths(rules)
    findings = check_rules(tree, rules)
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
    a.add_argument("--engine-source", type=Path, required=True, help="apply content hygiene inside this export using the engine cut rules")
    a.add_argument("--private-patterns", type=Path,
                   help="the private pattern file the content hygiene reads (default: $HEX_PRIVATE_PATTERNS); never packaged")
    c = sub.add_parser("check")
    c.add_argument("tree", type=Path)
    args = parser.parse_args(argv)
    holds = load_holds()
    if args.command == "apply":
        import importlib.util
        spec = importlib.util.spec_from_file_location("hex_release_hygiene", Path(__file__).with_name("release_hygiene.py"))
        hygiene = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = hygiene
        spec.loader.exec_module(hygiene)
        # THE BREAKAGE THIS PREVENTS: the cut rewrites the lines its content rules find; with no private pattern
        # file those rules are the generic one alone, and every private line would pass into the export.
        try:
            if args.private_patterns is not None:
                hygiene.configure(args.private_patterns)
        except RuntimeError as error:
            print(f"licence_export apply: {error}", file=sys.stderr)
            return 2
        if not hygiene._content.PRIVATE_LOADED:
            print("licence_export apply: no private pattern file (--private-patterns or $HEX_PRIVATE_PATTERNS); "
                  "the export's content hygiene cannot run without it", file=sys.stderr)
            return 2
        report = apply(args.src.resolve(), args.dst.resolve(), holds)
        findings = check(args.dst.resolve(), holds)
        report["content_hygiene"] = hygiene.fix_export(args.dst.resolve(), args.engine_source.resolve())
        remaining = hygiene.export_tree_findings(args.dst.resolve())
        findings.extend({"path": path, "line": line, "reason": kind} for path, line, kind, _ in remaining)
        report["findings"] = findings
        print(json.dumps(report, indent=1))
        return 1 if findings else 0
    findings = check(args.tree.resolve(), holds)
    print(json.dumps({"findings": findings}, indent=1))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
