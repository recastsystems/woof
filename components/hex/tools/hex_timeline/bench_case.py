"""Run the real hex door with a common, read-only C4 boundary observer.

The pre-ledger reference source receives minimal integration timestamps. The
candidate uses its own default C1 events. Both receive the same final-state
observer, after the final history capture, and two boundary synchronizations.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import dataclasses
import hashlib
import json
import os
from woof.hex import card_queue
from pathlib import Path
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def carried_digest(driver, backend, result):
    import numpy as np
    from woof.state_digest import _canonical_digest_document
    from woof.hex.cuda_dualrun import STATE_FIELDS, SIDECAR_FIELDS

    arrays, scalars = {}, {}
    def walk(path, value):
        if isinstance(value, np.ndarray) or hasattr(value, "__cuda_array_interface__"):
            arrays[path] = value
        elif isinstance(value, Mapping):
            for key, item in sorted(value.items()):
                walk(f"{path}/{key}", item)
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                walk(f"{path}/{field.name}", getattr(value, field.name))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                walk(f"{path}/{index}", item)
        elif isinstance(value, np.generic):
            scalars[path] = value.item()
        elif value is None or isinstance(value, (str, int, float, bool)):
            scalars[path] = value
        else:
            raise TypeError(f"unregistered carried state {path}: {type(value).__name__}")
    atmosphere = driver.atmosphere
    for name in STATE_FIELDS:
        walk(f"atmosphere/state/{name}", getattr(atmosphere.state, name))
    for name in SIDECAR_FIELDS:
        walk(f"atmosphere/saved/{name}", getattr(atmosphere.saved, name))
    walk("backend", backend.restart_state())
    walk("forcing", result.dynamics_tendencies)
    walk("surface", result.committed.surface_updates)
    scalars["model_time_seconds"] = float(atmosphere.state.time_seconds)
    return _canonical_digest_document(arrays, scalars, "hex")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.input / "case.json").read_text())
    from woof.timeline.ledger import Ledger
    value = (os.environ.get("WOOF_CARD") or card_queue.get(card_queue.CARDS).split(",")[0]
             or os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0])
    ledger = Ledger(product="hex", card=int(value) if value.isdecimal() else None)
    ledger.emit("boot.command", "begin", attrs={"scope": "hex CLI imports and startup"})
    from woof.hex.drivers import run_cuda_v841_full_physics_x4 as proof
    from woof.hex import cli
    reference = os.environ.get("BENCH_REFERENCE") == "1"
    last_result, final_digest, calls = None, None, 0
    planned = int(config["model_seconds"] / config["dt_seconds"])
    original_step, original_capture = proof.execute_composite_step, proof.capture_snapshot

    def step(**kwargs):
        nonlocal calls, last_result
        calls += 1
        if calls == 1:
            ledger.emit("boot.command", "end")
            if reference:
                ledger.emit("forecast.integrate", "begin", attrs={"basis": "reference observer"})
                ledger.emit("forecast.first_step", "begin", attrs={"basis": "reference observer"})
        result = original_step(**kwargs)
        # Common to both sources, only at the two observed boundaries.
        if calls in (1, planned):
            kwargs["driver"].cp.cuda.get_current_stream().synchronize()
        if reference and calls == 1:
            ledger.emit("forecast.first_step", "end", attrs={"basis": "reference observer"})
        if reference and calls == planned:
            ledger.emit("forecast.integrate", "end", attrs={"basis": "reference observer"})
        last_result = result
        return result

    def capture(**kwargs):
        nonlocal final_digest
        snapshot = original_capture(**kwargs)
        if int(kwargs["step"]) == planned:
            ledger.emit("post.state_digest", "begin", attrs={"scope": "common C4 observer"})
            final_digest = carried_digest(kwargs["driver"], kwargs["backend"], last_result)
            ledger.emit("post.state_digest", "end", attrs={"scope": "common C4 observer"})
        return snapshot

    proof.execute_composite_step, proof.capture_snapshot = step, capture
    variables = {"input": str(args.input), "output": str(args.output), "cache": str(args.cache)}
    argv = [item.format_map(variables) for item in config["forecast_argv"]]
    started = time.monotonic()
    try:
        rc = cli.main(argv)
    finally:
        proof.execute_composite_step, proof.capture_snapshot = original_step, original_capture
    if rc or calls != planned or final_digest is None:
        raise RuntimeError(f"forecast incomplete: rc={rc}, steps={calls}/{planned}, final digest={final_digest is not None}")
    forecast = args.output / "forecast"
    artifacts = args.output / "artifacts"
    artifacts.mkdir()
    histories = sorted(forecast.glob("cuda-history.*.nc"))
    if len(histories) != config["history_frames"]:
        raise RuntimeError(f"history inventory incomplete: {len(histories)} frames")
    hashes = {}
    for path in histories:
        os.link(path, artifacts / path.name)
        hashes[path.name] = sha(path)
    projection = json.loads((forecast / "snapshot-hashes.json").read_text())
    evidence = {"schema": "hex-bench-evidence-v1", "final_state_digest": final_digest,
        "prepared_content_sha256": config["prepared_content_sha256"],
        "hex_snapshots": projection, "backend": "cuda", "history_sha256": hashes,
        "restart_inventory": []}
    (artifacts / "evidence.json").write_text(json.dumps(evidence, sort_keys=True, indent=2))
    (args.output / "case-summary.json").write_text(json.dumps({
        "steps": calls, "door_wall_seconds": time.monotonic() - started,
        "reference_observer": reference, "history_sha256": hashes,
        "final_state_sha256": final_digest["sha256"], "restart_files": 0,
        "configuration_sha256": sha(args.input / "case.json")}, indent=2))
    ledger.emit("finish", attrs={"scope": "hex case observer"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
