#!/usr/bin/env python3
"""Run the real forecast door and retain small graph/identity/timing receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from woof.hex import card_queue
from pathlib import Path
import statistics
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--history-minutes", type=int, default=60)
    parser.add_argument("--fingerprint-every", type=int, default=720)
    parser.add_argument("--graphs", choices=("0", "1"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.cache.mkdir(parents=True, exist_ok=True)
    # Driver scratch admission requires a destination that does not yet exist.
    os.environ.update(
        WOOF_TIME_LEDGER=str(args.output / "ledger.jsonl"),
        CUPY_CACHE_DIR=str(args.cache / "cupy"),
        WOOF_HEX_PREP_CACHE=str(args.cache / "prep"),
        WOOF_CACHE_ROOT=str(args.cache / "engine"),
        XDG_CACHE_HOME=str(args.cache / "xdg"),
        WOOF_HEX_MESH_ROWS=str(args.input / "mesh" / "mesh-rows.json"),
        WOOF_HEX_REGIONAL_CONTRACT_DIR=str(args.input / "contract"),
        WOOF_HEX_CUDA_GRAPHS=args.graphs,
        WOOF_HEX_CUDA_GRAPHS_REQUIRE_REPLAY=args.graphs,
    )
    from woof.timeline.ledger import Ledger
    card = card_queue.get(card_queue.CARDS).split(",")[0]
    if not card.isdecimal():
        raise RuntimeError("measurement requires a granted physical card index from the card queue")
    os.environ["WOOF_CARD"] = card
    ledger = Ledger(product="hex", card=int(card) if card.isdecimal() else None)
    ledger.emit("proxy.start", attrs={"basis": "owned forecast command envelope, queue excluded"})
    started = time.monotonic()
    config = json.loads((args.input / "case.json").read_text())
    argv = [item.format(input=args.input, output=args.output, cache=args.cache)
            for item in config["forecast_argv"]]
    for flag, value in (("--hours", args.hours), ("--history-every-minutes", args.history_minutes)):
        argv[argv.index(flag) + 1] = str(value)
    argv += ["--repo", str(Path(__file__).resolve().parents[2]),
             "--gpuwm-checkout", str(args.engine),
             "--fingerprint-every", str(args.fingerprint_every)]
    request = {"argv": [sys.executable, "-m", "woof.hex.cli", *argv],
               "config": vars(args), "mutex_cards": card_queue.get(card_queue.CARDS, None),
               "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "case_sha256": sha(args.input / "case.json"),
               "pin_sha256": sha(args.input / "pin.json")}
    (args.output / "request.json").write_text(json.dumps(request, default=str, indent=2))
    from woof.hex.cli import main as forecast
    rc = forecast(argv)
    ledger.emit("proxy.end")
    if rc:
        raise RuntimeError(f"forecast door returned {rc}")
    root = args.output / "forecast"
    receipt = json.loads((root / "cuda-v841-forecast-receipt.json").read_text())
    steps = json.loads((root / "step-receipts.json").read_text())["receipts"]
    identity = json.loads((root / "carried-identity.json").read_text())
    artifacts = args.output / "artifacts"
    artifacts.mkdir()
    histories = {path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
                 for path in sorted(root.glob("cuda-history.*.nc"))}
    for name in histories:
        os.link(root / name, artifacts / name)
    snapshots = json.loads((root / "snapshot-hashes.json").read_text())
    evidence = {"final_state_digest": identity["final_state_digest"],
                "prepared_content_sha256": config["prepared_content_sha256"],
                "hex_snapshots": snapshots, "backend": "cuda",
                "history_sha256": {name: row["sha256"] for name, row in histories.items()},
                "restart_inventory": []}
    (artifacts / "evidence.json").write_text(json.dumps(evidence, sort_keys=True))
    # Preserve every step's full-field fingerprint in a bounded receipt. The
    # gate also compares original records on the box before their cleanup.
    with (root / "boundary-fingerprints.jsonl").open() as source:
        fingerprints = [{"step": row["step"],
                         "sha256": hashlib.sha256(json.dumps(row, sort_keys=True,
                                  separators=(",", ":")).encode()).hexdigest()}
                        for row in map(json.loads, source)]
    summary = {
        "returncode": rc, "graphs_requested": args.graphs,
        "steps": len(steps), "door_wall_seconds": time.monotonic() - started,
        "step_median_seconds": statistics.median(row["seconds"] for row in steps[1:]),
        "first_step_seconds": steps[0]["seconds"],
        "cuda_graphs": receipt["forecast"].get("cuda_graphs"),
        "history_files": histories, "fingerprints": fingerprints,
        "final_state_digest": identity["final_state_digest"]["sha256"],
        "identity_class": "matched forecast, full carried state and history",
        "restart_files": 0,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    subprocess.run([sys.executable, "-m", "woof.cli", "timeline", "report",
                    str(args.output / "ledger.jsonl"), "--output", str(args.output / "timeline.json"),
                    "--png", str(args.output / "timeline.png")], check=True)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("fingerprints", "history_files")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
