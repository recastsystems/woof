#!/usr/bin/env python3
"""Refuse a release artifact that carries private material.

    python tools/release_hygiene.py dist/gpuwm_hex-*.whl dist/gpuwm_hex-*.tar.gz
    python tools/release_hygiene.py --tree .          # the files git tracks here

THE BREAKAGE THIS GATE PREVENTS (release review, 2026-10-13): the 0.4.0
candidate wheel and sdist carried the owner's name in 16 and 18 files, home
directory paths in 23, internal cloud host names in 76, private workstation
folders, and the live root-ssh address of a rented GPU box in 9 sdist files.
Published 0.3.3 carried none of these.  The sdist's ``verification/`` had grown
from a curated 915 KB of contracts to 31 MB of lane receipts.  Nothing looked
at the artifacts' members before an upload would have shipped them.

Every member of every artifact is scanned (binaries too: a Rust build embeds
the build directory).  The rules are generic and carry no private literal, so
this file can ship: the identity tokens (the checkout's ``git config
user.name``/``user.email``, the login name, the home folder's name, and any
``--owner``) are read where the gate runs.  Exit 0 clean, 1 with every hit
listed, 2 when an artifact cannot be read.
"""

from __future__ import annotations

import argparse
import getpass
import ipaddress
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
import importlib.util

_content_spec = importlib.util.spec_from_file_location("hex_export_content", Path(__file__).with_name("export_content.py"))
_content = importlib.util.module_from_spec(_content_spec)
sys.modules[_content_spec.name] = _content
_content_spec.loader.exec_module(_content)
content_scan = _content.scan
from typing import Iterable, Iterator

# Generic rules: (name, pattern).  Each names a class of private material.
RULES: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("home-path", re.compile(rb"(?i)\b[a-z]:[\\/]{1,2}users[\\/]{1,2}(?!public\b|default\b)[a-z0-9._-]+")),
    ("home-path", re.compile(rb"(?<![\w.])/(?:home|Users)/(?!runner\b|user\b|username\b|you\b|me\b)[A-Za-z0-9._-]+/")),
    ("home-path", re.compile(rb"\bC--Users-[A-Za-z0-9._-]+")),
    ("downloads-path", re.compile(rb"(?<![\w.])Downloads[\\/]")),
    ("cloud-host-name", re.compile(rb"\bip-\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3}\b")),
    # split literals: the shipped gate itself should not spell a lab's naming
    ("lab-host-name", re.compile(rb"\bweather" rb"-node-\d+\b")),
    ("workstation-folder", re.compile(rb"\bcodex" rb"-\d{3,4}\b")),
    ("workstation-folder", re.compile(rb"\bagent" rb"-scratch\b")),
    ("private-case", re.compile(rb"(?i)\b" rb"u" rb"ae\b")),
    ("private-da-name", re.compile(rb"(?i)\bcuda\s+" rb"je" rb"di\b")),
    ("private-da-name", re.compile(rb"(?i)\bsyn" rb"radar\b")),
    ("private-da-name", re.compile(rb"(?i)\bsynthetic-radar\s+da\s+cycle\b")),
    ("private-worktree", re.compile(rb"(?i)\.claude[/\\]worktrees")),
    ("private-host", re.compile(rb"(?i)\bno" rb"de-2\b")),
    ("private-pool", re.compile(rb"(?i)\bpool-" rb"v3\b")),
    ("private-ruling-file", re.compile(rb"(?i)\bd" rb"rew-ruling\.json\b")),
)

# Apply the same private-name rules to embedded strings in binary members.
RULES += tuple((kind, re.compile(pattern.pattern.encode(), re.I))
               for kind, pattern in _content.RULES
               if kind in {"private assimilation name", "private assimilation tooling"})

IPV4 = re.compile(rb"(?<![\w.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?![\w.])")

TEXT_LIMIT = 64 * 1024 * 1024  # members larger than this are scanned in part


def identity_tokens(extra: Iterable[str], where: Path) -> list[str]:
    """Owner identity read where the gate runs, never written into this file."""

    tokens: set[str] = {t for t in extra if t}
    for key in ("user.name", "user.email"):
        try:
            value = subprocess.run(["git", "-C", str(where), "config", key],
                                   capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            value = ""
        if value:
            tokens.add(value)
    for value in (os.environ.get("USER"), os.environ.get("USERNAME")):
        if value:
            tokens.add(value)
    # THE BREAKAGE (hex 0.4.12 publish.yml dry run 42033510796, 2026-11-05): a GitHub container job runs with
    # HOME=/github/home, so the home folder's NAME was "home" and the capitalised rule refused every "HOME"
    # (os.environ["HOME"], $HOME, the Rust standard library's home_dir) as the owner's name: 14 wheel and
    # 6 sdist false hits, none of them private.  Only a per-user home (/home/<name>, /Users/<name>,
    # C:/Users/<name>) names a person.
    home = Path.home()
    if home.parent.name.lower() in {"home", "users"}:
        tokens.add(home.name)
    try:
        tokens.add(getpass.getuser())
    except Exception:  # noqa: BLE001 - no login name is not a hit
        pass
    # A login such as "root" or a one-letter name is not an identity.
    return sorted(t for t in tokens if len(t) >= 3 and t.lower() not in {"root", "runner", "user", "admin", "home"})


def identity_rules(tokens: Iterable[str]) -> list[tuple[str, re.Pattern[bytes]]]:
    rules = []
    for token in tokens:
        if "@" in token:
            rules.append(("owner-email", re.compile(re.escape(token).encode(), re.IGNORECASE)))
            continue
        for part in re.split(r"\s+", token):
            if len(part) < 3:
                continue
            # The capitalised form only: a lower-case login name is often an
            # English word ("drew" the verb), and paths are caught above.
            shown = part[:1].upper() + part[1:]
            rules.append(("owner-name", re.compile(rb"\b" + (b"(?:" + re.escape(shown).encode() + b"|" + re.escape(shown.upper()).encode() + b")") + rb"\b")))
    return rules


def public_addresses(data: bytes) -> Iterator[bytes]:
    for match in IPV4.finditer(data):
        octets = [int(g) for g in match.groups()]
        if any(o > 255 for o in octets):
            continue
        address = ipaddress.ip_address(".".join(map(str, octets)))
        if (address.is_private or address.is_loopback or address.is_unspecified
                or address.is_link_local or address.is_multicast or address.is_reserved):
            continue
        yield match.group(0)


def members(artifact: Path) -> Iterator[tuple[str, bytes]]:
    name = artifact.name
    if name.endswith((".whl", ".zip")):
        with zipfile.ZipFile(artifact) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    with archive.open(info) as handle:
                        yield info.filename, handle.read(TEXT_LIMIT)
    elif name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(artifact) as archive:
            for info in archive.getmembers():
                if info.isfile():
                    handle = archive.extractfile(info)
                    if handle is not None:
                        yield info.name, handle.read(TEXT_LIMIT)
    else:
        raise ValueError(f"{artifact}: not a wheel, zip or tar archive")


def tree_members(root: Path) -> Iterator[tuple[str, bytes]]:
    listed = subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                            capture_output=True, check=True).stdout.split(b"\0")
    for raw in listed:
        if raw:
            path = root / raw.decode()
            if path.is_file():
                yield raw.decode(), path.read_bytes()[:TEXT_LIMIT]


def _scan_member(arguments):
    item, rules = arguments
    hits: list[tuple[str, str, str]] = []
    for member, data in [item]:
        hits.extend((member, kind, f"line {line}: {context}")
                    for _, line, kind, context in content_scan(member, data))
        seen: set[tuple[str, bytes]] = set()
        for rule, pattern in rules:
            # A short case token occurs by chance in compressed image bytes.
            # Other rules still inspect binaries for embedded paths.
            if rule == "private-case" and b"\0" in data[:8192]:
                continue
            if _content.sealed_exception(member, rule):
                continue
            for match in pattern.finditer(data):
                key = (rule, match.group(0))
                if key not in seen:
                    seen.add(key)
                    hits.append((member, rule, match.group(0).decode("utf-8", "replace")))
        for address in public_addresses(data):
            key = ("public-ipv4", address)
            if key not in seen:
                seen.add(key)
                hits.append((member, "public-ipv4", address.decode()))
    return hits



def scan(items: Iterable[tuple[str, bytes]],
         rules: list[tuple[str, re.Pattern[bytes]]]) -> list[tuple[str, str, str]]:
    rows = list(items)
    arguments = [(item, rules) for item in rows]
    if len(rows) > 8 and sum(len(data) for _, data in rows) > 1024 * 1024:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        with ProcessPoolExecutor(max_workers=8, mp_context=get_context("fork")) as pool:
            groups = pool.map(_scan_member, arguments, chunksize=8)
            return [hit for group in groups for hit in group]
    return [hit for argument in arguments for hit in _scan_member(argument)]


from functools import lru_cache

@lru_cache(maxsize=8)
def _export_rules(root: Path):
    return list(RULES) + identity_rules(identity_tokens([], root))


def export_findings(name: str, data: bytes, root: Path):
    findings = content_scan(name, data)
    covered = {hit[1] for hit in findings}
    for path, kind, context in scan([(name, data)], _export_rules(root)):
        if context.startswith("line "):
            continue
        offset = data.find(context.encode())
        line = data[:max(0, offset)].count(b"\n") + 1
        if line not in covered:
            findings.append((path, line, kind, context))
            covered.add(line)
    return findings


def _export_member(arguments):
    name, data, root = arguments
    return export_findings(name, data, root)


def export_tree_findings(root: Path):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    rows = [(p.relative_to(root).as_posix(), p.read_bytes(), root)
            for p in sorted(root.rglob("*")) if p.is_file()
            and ".git" not in p.relative_to(root).parts]
    if len(rows) <= 8:
        return [hit for row in rows for hit in _export_member(row)]
    with ProcessPoolExecutor(max_workers=8, mp_context=get_context("fork")) as pool:
        return [hit for group in pool.map(_export_member, rows, chunksize=8) for hit in group]


def fix_export(root: Path, engine_source: Path) -> dict:
    """Apply the engine cut's recorded hygiene rules to this proposed export."""
    import tempfile
    spec = importlib.util.spec_from_file_location('hex_cut_hygiene',
        engine_source / 'tools/release/cut/hygiene.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    findings = export_tree_findings(root)
    blobs = {name: (root / name).read_bytes() for name in {hit[0] for hit in findings}}
    with tempfile.TemporaryDirectory(prefix='export-hygiene-', dir=root.parent) as scratch:
        repo = Path(scratch)
        subprocess.run(['git', 'init', '-q', str(repo)], check=True)
        oids = {name: module.write_blob(repo, data) for name, data in blobs.items()}
        rewrites, refused = module.plan(blobs, oids,
            hits=[module.Hit(path, line, kind) for path, line, kind, _ in findings])
        expected = module.apply(repo, {name: ('100644', oid) for name, oid in oids.items()}, blobs, rewrites)
        for name in blobs:
            path = root / name
            if name not in expected:
                path.unlink()
            else:
                data = subprocess.check_output(['git', '-C', str(repo), 'cat-file', 'blob', expected[name][1]])
                path.write_bytes(data)
    remaining = export_tree_findings(root)
    return {'hits_before': findings, 'rewrites': [item.to_json() for item in rewrites],
            'unrewritable': refused, 'remaining': remaining}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fix-export", type=Path, help="rewrite the proposed export inside the current cut")
    parser.add_argument("--engine-source", type=Path, help="engine checkout supplying the cut hygiene rules")
    parser.add_argument("artifacts", nargs="*", type=Path)
    parser.add_argument("--tree", type=Path, help="scan the files git tracks under this folder")
    parser.add_argument("--owner", action="append", default=[],
                        help="another identity token to refuse (repeatable)")
    parser.add_argument("--allow-address", action="append", default=[],
                        help="a public IPv4 address that is meant to ship (repeatable)")
    args = parser.parse_args(argv)
    if args.fix_export is not None:
        if args.engine_source is None:
            parser.error('--fix-export requires --engine-source')
        import json
        report = fix_export(args.fix_export.resolve(), args.engine_source.resolve())
        print(json.dumps(report, indent=2))
        return 1 if report['remaining'] or report['unrewritable'] else 0
    if not args.artifacts and args.tree is None:
        parser.error("give artifacts or --tree")
    where = args.tree or Path(__file__).resolve().parent
    rules = list(RULES) + identity_rules(identity_tokens(args.owner, where))
    allowed = set(args.allow_address)
    sources: list[tuple[str, Iterable[tuple[str, bytes]]]] = []
    if args.tree is not None:
        sources.append((str(args.tree), tree_members(args.tree)))
    for artifact in args.artifacts:
        sources.append((str(artifact), members(artifact)))
    total = 0
    for label, items in sources:
        try:
            hits = [h for h in scan(items, rules)
                    if not (h[1] == "public-ipv4" and h[2] in allowed)]
        except (OSError, ValueError, zipfile.BadZipFile, tarfile.TarError) as error:
            print(f"HYGIENE cannot read {label}: {error}", file=sys.stderr)
            return 2
        total += len(hits)
        for member, rule, text in hits:
            print(f"HYGIENE {label} :: {member} :: {rule} :: {text}")
        print(f"HYGIENE {label}: {len(hits)} hit(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
