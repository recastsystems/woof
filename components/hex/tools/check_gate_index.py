#!/usr/bin/env python3
"""Check the public gate index: every reference relative, every pin consistent.

    python tools/check_gate_index.py verification/gates           # structure only (no data needed)
    python tools/check_gate_index.py verification/gates --data    # also every data file under data/

THE BREAKAGE THIS PREVENTS: a gate index that names a file by an absolute
path works only on the machine that wrote it.  The public index
(verification/gates/) pins the same decks, inputs and ratchet as the
maintainers' gate with relative paths; this check refuses an absolute path, a
deck whose bytes no longer match its pin, a data reference the manifest does
not list (or lists with another sha256), a landed deck G2 omits, and, with
--data, a data file that is missing or changed.  Exit 0 clean, 1 with every
finding listed.

The MPAS route's own resources (the native framework library, the Registry
pool reader, the pinned MPAS-Model tree, the hex-cuda resources packet) are
not data: a reader builds them with `woof hex physics-resources build`, and a
deck names them `{resource:KEY}` (woof.hex.verify_mpas.InstalledResources).  A
deck that names one by a path again, or names an unknown key, is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

# woof.hex.mpasphys.resources_build.RESOURCE_KEYS (tests/test_physics_route_resources.py holds them equal).
RESOURCE_KEYS = ("physics-resources", "thompson-tables", "pool-reader", "mpas-source", "native-library",
                 "mpas-source-rrfs", "native-library-rrfs",
                 "native-library-rrfs-44c0849", "mpas-source-rrfs-44c0849")
RESOURCE_OPTIONS = {"--native-physics-library": "native-library", "--native-pool-reader": "pool-reader",
                    "--mpas-source": "mpas-source", "--physics-resources": "physics-resources"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_absolute(text: str) -> bool:
    return text.startswith(("/", "~")) or (len(text) > 2 and text[1] == ":" and text[2] in "\\/")


def refs(value, where="$"):
    """Every {path, sha256} reference inside a JSON value, with its location."""
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and "sha256" in value:
            yield where, value
        for key, item in value.items():
            if key != "path":
                yield from refs(item, f"{where}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from refs(item, f"{where}[{i}]")


def strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, str):
        yield value


def check(root: Path, with_data: bool) -> list[str]:
    findings: list[str] = []
    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    if index.get("schema") != "mpas-gate-index-v1":
        findings.append("index.json: schema is not mpas-gate-index-v1")
    manifest = json.loads((root / index.get("data_manifest", "data-manifest.json")).read_text())
    listed = manifest.get("files", {})
    deck_ids: dict[str, set] = {}
    for gate, entries in index.get("gates", {}).items():
        if not entries:
            findings.append(f"index.json: {gate} has no decks")
        deck_ids[gate] = set()
        for entry in entries:
            if is_absolute(entry["path"]):
                findings.append(f"index.json: {gate} names {entry['path']} by an absolute path")
                continue
            deck_path = (root / entry["path"])
            if not deck_path.is_file():
                findings.append(f"index.json: {gate} deck {entry['path']} is missing")
                continue
            if sha(deck_path) != entry["sha256"]:
                findings.append(f"index.json: {gate} deck {entry['path']} does not match its pin")
            deck = json.loads(deck_path.read_text())
            deck_ids[gate].add(deck.get("id"))
            if deck.get("schema") != "mpas-verification-deck-v1":
                findings.append(f"{entry['path']}: schema is not mpas-verification-deck-v1")
            if not isinstance(deck.get("process_count"), int) or deck["process_count"] < 1:
                findings.append(f"{entry['path']}: no fixed process count")
            for where, ref in refs(deck):
                text = ref["path"]
                if is_absolute(text):
                    findings.append(f"{entry['path']} {where}: absolute path {text}")
                    continue
                resolved = PurePosixPath(entry["path"]).parent / text
                parts = []
                for part in resolved.parts:
                    if part == "..":
                        if parts:
                            parts.pop()
                    elif part != ".":
                        parts.append(part)
                relative = "/".join(parts)
                if relative.startswith("data/"):
                    item = listed.get(relative[len("data/"):])
                    if item is None:
                        findings.append(f"{entry['path']} {where}: data/{relative[5:]} is not in the data manifest")
                    elif item.get("sha256") != ref["sha256"]:
                        findings.append(f"{entry['path']} {where}: pin differs from the data manifest")
            for text in strings(deck.get("hex_command", [])):
                if is_absolute(text):
                    findings.append(f"{entry['path']}: model command names an absolute path {text}")
            command = [text for text in deck.get("hex_command", []) if isinstance(text, str)]
            for i, text in enumerate(command[:-1]):
                key = RESOURCE_OPTIONS.get(text)
                if key is not None and command[i + 1] != "{resource:" + key + "}":
                    findings.append(f"{entry['path']}: {text} names a path, not {{resource:{key}}}")
            # THE BREAKAGE THIS PREVENTS: the published GPU headline demanded a site-built native MPAS library
            # (and its Fortran runtime) that its route never loads, so a PyPI user could not rerun the proof.
            # A headline with every slot on hex-cuda runs the ported framework (runtime_v842.framework_only).
            backends = deck.get("headline_backends") or {}
            if (deck.get("role") == "headline" and backends and set(backends.values()) == {"hex-cuda"}
                    and "--native-physics-library" in command):
                findings.append(f"{entry['path']}: an all-GPU headline names --native-physics-library; "
                                "its route loads no native library")
            for key in re.findall(r"\{resource:([^}]*)\}", " ".join(command)):
                if key not in RESOURCE_KEYS:
                    findings.append(f"{entry['path']}: unknown resource key {key}")
            for text in strings(deck):
                if text.startswith(("/home/", "/Users/")) or "\\Users\\" in text:
                    findings.append(f"{entry['path']}: names a home folder: {text[:80]}")
    missing = set(index.get("landed_decks", [])) - deck_ids.get("G2", set())
    if missing:
        findings.append(f"index.json: G2 omits landed deck(s) {sorted(missing)}")
    ratchet = json.loads((root / "ratchet.json").read_text())
    if ratchet.get("schema") != "mpas-ratchet-v1" or not isinstance(ratchet.get("entries"), list):
        findings.append("ratchet.json: schema is not mpas-ratchet-v1 with entries")
    for text in strings({k: v for k, v in ratchet.items() if k != "entries"}):
        if is_absolute(text):
            findings.append(f"ratchet.json: absolute path {text[:80]}")
    if with_data:
        data = root / "data"
        absent = changed = 0
        for relative, item in listed.items():
            path = data / relative
            if item.get("directory"):
                if not path.is_dir():
                    findings.append(f"data/{relative}/: folder missing")
                continue
            if not path.is_file():
                absent += 1
                if absent <= 20:
                    findings.append(f"data/{relative}: missing")
            elif item.get("sha256") and sha(path) != item["sha256"]:
                changed += 1
                findings.append(f"data/{relative}: sha256 differs from the manifest")
        if absent > 20:
            findings.append(f"... {absent - 20} more data files missing")
    return findings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parents[1] / "verification/gates")
    parser.add_argument("--data", action="store_true", help="also check every data file under ROOT/data")
    args = parser.parse_args(argv)
    findings = check(args.root, args.data)
    for finding in findings:
        print("GATE-INDEX", finding)
    print(f"GATE-INDEX {args.root}: {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
