"""Verify the exact public WRF source bytes before compiling the PX oracle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


def verify_sources(source_root: Path) -> dict:
    pin_path = Path(__file__).with_name("source-pin.json")
    pin = json.loads(pin_path.read_text())
    source_root = source_root.resolve()
    head = None
    if (source_root / ".git").exists():
        head = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        if head not in pin["revisions"].values():
            raise ValueError(f"WRF revision {head} is not in the PX source pin")
    files = []
    for relative, expected in pin["files"].items():
        raw = (source_root / relative).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected["sha256"]:
            raise ValueError(f"{relative}: SHA-256 {digest}, expected {expected['sha256']}")
        files.append(dict(path=relative, sha256=digest, bytes=len(raw),
                          license=expected["license"]))
    return dict(source_root=str(source_root), head=head,
                pin_sha256=hashlib.sha256(pin_path.read_bytes()).hexdigest(),
                files=files)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_sources.py WRF_SOURCE_ROOT")
    try:
        receipt = verify_sources(Path(sys.argv[1]))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"PX source verification failed: {exc}") from exc
    print(json.dumps(receipt, indent=2))
