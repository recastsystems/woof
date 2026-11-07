#!/usr/bin/env python3
"""Verify official WRF bytes at the immutable pin and record each licence.

Network access is explicit. No code from the competitor is fetched or read.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((HERE / "sources.json").read_text())
    commit = manifest["wrf"]["commit"]
    notice = HERE / "LICENSE-WRF.txt"
    notice_sha = hashlib.sha256(notice.read_bytes()).hexdigest()

    def check(row):
        url = f"https://raw.githubusercontent.com/{row.get('repository','wrf-model/WRF')}/{row.get('commit',commit)}/{row['path']}"
        with urlopen(url, timeout=60) as response:
            remote = response.read()
        digest = hashlib.sha256(remote).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"immutable source differs: {row['path']}")
        if "local_path" in row:
            local = HERE / row["local_path"]
        elif row["path"] == "LICENSE.txt":
            local = notice
        elif row["path"].startswith(("Registry/", "tools/")):
            local = HERE / "upstream" / row["path"]
        else:
            local = HERE / "upstream" / Path(row["path"]).name
        # README.namelist is audit evidence rather than a build input.
        if local.exists() and hashlib.sha256(local.read_bytes()).hexdigest() != digest:
            raise ValueError(f"local source differs: {local}")
        return {**row, "immutable_url": url, "immutable_bytes_equal": True,
                "licence": "WRF public-domain statement for this version and release",
                "licence_url": f"https://raw.githubusercontent.com/wrf-model/WRF/{commit}/LICENSE.txt",
                "licence_sha256": notice_sha,
                "tables": "none added by these six diagnostic modules"}

    # Licence is verified before downloading any remaining source.
    license_row = next(r for r in manifest["sources"] if r["path"] == "LICENSE.txt")
    checked = [check(license_row)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        checked += list(pool.map(check, [r for r in manifest["sources"] if r is not license_row]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"wrf": manifest["wrf"], "files": checked}, indent=2) + "\n")
    print(json.dumps({"verified": len(checked), "receipt": str(args.output)}))


if __name__ == "__main__":
    main()
