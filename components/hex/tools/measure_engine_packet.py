#!/usr/bin/env python3
"""Measure a sealed, prepared public engine packet without declaring it published.

The default pin remains the published engine. A packet row accepts exactly the
sealed wheel's version and all its seam bytes, verified against its public
source commit. It never changes the dependency floor or a published row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import measure_engine_verdicts as measurement
import repin_source_tables as repin

BEGIN = "# --- BEGIN GENERATED: PACKET_ENGINES ---"
END = "# --- END GENERATED: PACKET_ENGINES ---"
SCHEMA = "gpuwm-hex.prepared-engines/v1"


def packet_identity(assets: Path, wheel: Path) -> tuple[str, str]:
    data = json.loads(assets.read_text())
    github = data.get("github", {})
    version = str(github.get("target_version", "")).removeprefix("v")
    commit = data.get("engine_source_revision", "")
    if (data.get("schema") != "arwen.publication-assets.v1"
            or data.get("status") != "PREPARED_NOT_PUBLISHED"
            or data.get("published") is not False
            or github.get("repository") not in {"recastsystems/arwen", "FahrenheitResearch/arwen"}
            or not re.fullmatch(r"\d+\.\d+\.\d+", version)
            or not re.fullmatch(r"[0-9a-f]{40}", commit)):
        raise SystemExit("packet assets do not identify a prepared public engine")
    rows = [row for row in data.get("pypi", {}).get("artifacts", [])
            if row.get("filename") == wheel.name]
    if (len(rows) != 1 or rows[0].get("bytes") != wheel.stat().st_size
            or rows[0].get("sha256") != measurement.sha256_bytes(wheel.read_bytes())):
        raise SystemExit("packet wheel does not match the sealed assets digest and size")
    if measurement.wheel_version(wheel) != version:
        raise SystemExit("packet wheel version disagrees with the assets release")
    return version, commit


def render_rows(rows: list[dict]) -> str:
    return measurement.render_admitted_rows(rows).replace("ADMITTED_ENGINES", "PACKET_ENGINES")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--engine-tree", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    version, commit = packet_identity(args.assets, args.wheel)
    origin = measurement._git(args.engine_tree, "remote", "get-url", "origin").removesuffix(".git")
    if origin not in {"https://github.com/recastsystems/arwen", "https://github.com/FahrenheitResearch/arwen"}:
        raise SystemExit("packet source must be a clone of the public engine repository")
    manifest = measurement.shipped_manifest(measurement.MANIFEST_MODULE)
    source = repin.verify_engine_tree(args.engine_tree, commit, version, sorted(manifest))
    wheel, absent = measurement.wheel_digests(args.wheel, sorted(manifest))
    if absent or source != wheel:
        raise SystemExit(f"packet public source and sealed wheel disagree: absent {absent}; moved "
                         f"{sorted(p for p in source if source[p] != wheel.get(p))}")
    engine_pin = measurement._engine_pin_module()
    if not (measurement.version_key(engine_pin.gpuwm_floor()) <= measurement.version_key(version)
            < measurement.version_key(engine_pin.gpuwm_ceiling())):
        raise SystemExit("packet is outside the existing dependency range")
    glacier = measurement.glacier_digest_from_wheel(args.wheel)
    if glacier != measurement._glacier_digest_from_root(args.engine_tree):
        raise SystemExit("packet wheel and public source glacier digests disagree")
    # The adapter's separate surface guard must work on both engines too.
    if repin.contract_surface_digest(args.engine_tree, wheel) != repin.contract_surface_digest(
            args.engine_tree, manifest):
        raise SystemExit("packet contract surface differs from the published pin")
    helpers = repin.helper_set_row(args.engine_tree, commit, version)
    helper_wheel, missing = measurement.wheel_digests(args.wheel, sorted(helpers["sources"]))
    if missing or helpers["sources"] != helper_wheel:
        raise SystemExit("packet radiation helper source and sealed wheel disagree")
    row = {"version": version, "base": version, "build_commit": commit,
           "digests": wheel, "moved_from_base": sorted(p for p in manifest if manifest[p] != wheel[p]),
           "glacier_composed_tu_sha256": glacier,
           "artifacts": [{"name": args.wheel.name, "kind": "wheel",
                          "sha256": measurement.sha256_bytes(args.wheel.read_bytes())}]}
    document = {"schema": SCHEMA, "instrument": "tools/measure_engine_packet.py",
                "instrument_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "assets_sha256": measurement.sha256_bytes(args.assets.read_bytes()),
                "engines": [row]}
    target = repin.TREE / "verification/engine-verdicts/prepared-engines.json"
    module = repin.TREE / "src/hexcore/engine_pin.py"
    text = module.read_text()
    rendered = render_rows(document["engines"])
    if args.write:
        target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        start, end = text.index(BEGIN), text.index(END) + len(END)
        module.write_text(text[:start] + rendered + text[end:])
        repin.write_helper_set(helpers)
    elif not target.is_file() or json.loads(target.read_text()) != document or rendered not in text:
        raise SystemExit("packet measurement differs from the recorded contract")
    print(f"packet woof {version}: {len(wheel)} seam paths verified at public commit {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
