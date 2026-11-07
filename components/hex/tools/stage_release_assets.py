#!/usr/bin/env python3
"""Stage byte-pinned release assets and print publish commands without running them.

Sources are local directories. No GitHub command or upload is performed.
The optional HTTP check verifies <base>/<name>, with one base per release tag.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shlex
import tempfile
import threading
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PIN_NAMES = ("engine-bridges.json", "physics-resource-assets.json", "on-demand-assets.json")
SCHEMAS = ("gpuwm-hex.engine-bridges/v1", "gpuwm-hex.physics-resource-assets/v1",
           "gpuwm-hex.on-demand-assets/v1")


def safe_component(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.+-]*", value):
        raise ValueError(f"unsafe asset name or release tag: {value!r}")
    return value


def load_pins(paths: list[Path]) -> list[dict]:
    rows = {}
    for path in paths:
        document = json.loads(path.read_text())
        schema = document.get("schema")
        if schema not in SCHEMAS:
            raise ValueError(f"{path}: unsupported pin schema {schema!r}")
        release = document["release_assets"]
        tag = safe_component(release["tag"])
        repository = release["repository"]
        if repository not in {"FahrenheitResearch/gpuwm-hex", "recastsystems/gpuwm-hex"}:
            raise ValueError(f"{path}: unexpected repository {repository!r}")
        table = document["bridges" if schema == SCHEMAS[0] else "assets"]
        entries = list(table.items() if isinstance(table, dict) else ((row["name"], row) for row in table))
        if schema == SCHEMAS[0]:
            # Source-owned prebuilts (rw_wrfbatch, rw_sfc_series) ride the bridge release: publish.yml
            # downloads them by SHA-256 because the public floor tag cannot build them.
            entries += list(document.get("source_prebuilts", {}).get("files", {}).items())
        for name, pin in entries:
            name = safe_component(name)
            size, digest = pin["bytes"], pin["sha256"]
            if type(size) is not int or size < 0 or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"{path}: invalid size or sha256 for {name}")
            row_tag = safe_component(pin.get("release_tag", tag))
            row = dict(tag=row_tag, repository=repository, name=name, bytes=size, sha256=digest)
            key = (row_tag, name)
            if key in rows and rows[key] != row:
                raise ValueError(f"conflicting pins for {row_tag}/{name}")
            rows[key] = row
    # Each of these releases must exist before the main version release.
    return sorted(rows.values(), key=lambda row: (row["tag"], row["name"]))


def stage_one(pin: dict, sources: dict[str, list[Path]], out: Path) -> dict:
    destination = out / pin["tag"] / pin["name"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = dict(pin, staged_path=str(destination), found=False, status="missing")
    errors = []
    for source in sources.get(pin["name"], []):
        temporary = None
        try:
            digest, size = hashlib.sha256(), 0
            with source.open("rb") as stream, tempfile.NamedTemporaryFile(
                    dir=destination.parent, prefix=".stage-", delete=False) as target:
                temporary = Path(target.name)
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    size += len(block)
                    digest.update(block)
                    target.write(block)
            actual = digest.hexdigest()
            if size != pin["bytes"] or actual != pin["sha256"]:
                errors.append(f"{source}: bytes={size}, sha256={actual}")
                continue
            os.replace(temporary, destination)
            temporary = None
            row.update(found=True, status="verified", source=str(source))
            return row
        except OSError as error:
            errors.append(f"{source}: {error}")
        finally:
            if temporary is not None:
                temporary.unlink()
    if errors:
        row.update(status="mismatched", errors=errors)
    return row


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def serve_check(out: Path, rows: list[dict], workers: int) -> dict:
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(out)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def check(row):
        url = f"{base}/{quote(row['tag'])}/{quote(row['name'])}"
        if not row["found"]:
            return dict(url=url, status="skipped", name=row["name"], reason=row["status"])
        try:
            digest, size = hashlib.sha256(), 0
            with urlopen(url, timeout=30) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
            passed = digest.hexdigest() == row["sha256"] and size == row["bytes"]
            return dict(url=url, status="passed" if passed else "failed", name=row["name"],
                        bytes=size, sha256=digest.hexdigest())
        except OSError as error:
            return dict(url=url, status="failed", name=row["name"], error=str(error))

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            checks = list(pool.map(check, rows))
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    counts = {status: sum(row["status"] == status for row in checks)
              for status in ("passed", "failed", "skipped")}
    for row in checks:
        print(f"serve-check {row['status']}: {row['url']}")
    print("serve-check: " + ", ".join(f"{count} {status}" for status, count in counts.items()))
    return dict(base=base, checks=checks, **counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", type=Path, action="append",
                        help="Pin file, repeatable; replaces the three default package pins")
    parser.add_argument("--source", type=Path, action="append", default=[],
                        help="Local asset directory, repeatable; also searches release-assets/kernel-packs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8, help="Concurrent hashing, copying and HTTP workers")
    parser.add_argument("--serve-check", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        pins = load_pins(args.pin or [ROOT / "src/hexcore/data" / name for name in PIN_NAMES])
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"pin error: {error}")
        return 1
    sources = {}
    for directory in [ROOT / "release-assets/kernel-packs", *args.source]:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                sources.setdefault(path.name, []).append(path)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda pin: stage_one(pin, sources, out), pins))
    commands = ["# Asset releases must be created before the v0.4.12 release.",
                "# Run only after every manifest asset is verified."]
    for tag in sorted({row["tag"] for row in rows}):
        group = [row for row in rows if row["tag"] == tag]
        sums = out / tag / "SHA256SUMS"
        sums.write_text("".join(f"{row['sha256']}  {row['name']}\n" for row in group if row["found"]))
        command = ["gh", "release", "create", tag, "--repo", group[0]["repository"],
                   "--title", tag, "--notes", "Byte-pinned woof hex release assets.",
                   *[row["staged_path"] for row in group], str(sums)]
        commands.append(shlex.join(command))
    (out / "PUBLISH-ASSETS.txt").write_text("\n".join(commands) + "\n")
    manifest = dict(schema="gpuwm-hex.staged-release-assets/v1", assets=rows)
    for row in rows:
        print(f"{row['status']}: {row['tag']}/{row['name']}")
        for error in row.get("errors", []):
            print(f"  {error}")
    if args.serve_check:
        manifest["serve_check"] = serve_check(out, rows, args.workers)
    (out / "assets-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return int(any(not row["found"] for row in rows)
               or manifest.get("serve_check", {}).get("failed", 0) != 0)


if __name__ == "__main__":
    raise SystemExit(main())
