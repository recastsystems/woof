"""Copy pinned WRF source and correct the coupled fire clock locally."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

DRIVER = "phys/module_fr_fire_driver.F"
PINNED_DRIVER_SHA256 = "7662f29bb003697d08ccee585e272a3ed7362edc73bcef5ea8dc95d100d9e578"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def correct(source: Path, destination: Path, wind_patch: Path | None = None):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists() or destination == source or source in destination.parents:
        raise ValueError("the corrected reference requires a new directory outside the pinned source")
    original = (source / DRIVER).read_bytes()
    if sha(original) != PINNED_DRIVER_SHA256:
        raise ValueError("the fire driver is not the pinned WRF 4.7.1 source")
    shutil.copytree(source, destination, symlinks=True,
                    ignore=shutil.ignore_patterns(".git", ".env", ".env.local", "__pycache__"))
    text = original.decode()
    old = "time_start = itimestep * dt"
    if text.count(old) != 1:
        raise ValueError("the pinned time assignment changed")
    text = text.replace(old, "time_start = max(itimestep - 1, 0) * dt")
    loop = "itimestep = grid%itimestep + istep"
    if text.count(loop) != 1:
        raise ValueError("the pinned diagnostic test loop changed")
    text = text.replace(loop, loop + "\n      time_start = max(itimestep - 1, 0) * dt")
    (destination / DRIVER).write_bytes(text.encode())
    source_changes = {DRIVER: (original, (destination / DRIVER).read_bytes())}
    corrections = ["fire clock begins at the actual atmosphere step start",
                   "diagnostic fire_test_steps recomputes its starting time on every extra step"]
    if wind_patch is not None:
        patch_text = wind_patch.read_text()
        paths = {line[6:].split()[0] for line in patch_text.splitlines() if line.startswith(("--- a/", "+++ b/"))}
        if paths != {DRIVER}:
            raise ValueError("the supplied wind correction must change only the copied fire driver")
        subprocess.run(["patch", "--batch", "--forward", "-p1", "-d", str(destination)],
                       input=patch_text, text=True, check=True)
        corrections.append("separate pinned wind interpolation correction")
    corrected = (destination / DRIVER).read_bytes()
    source_changes[DRIVER] = (original, corrected)
    patch = "".join("".join(difflib.unified_diff(old.decode().splitlines(keepends=True),
                                        new.decode().splitlines(keepends=True),
                                        fromfile=f"a/{name}", tofile=f"b/{name}"))
                    for name, (old, new) in source_changes.items())
    (destination / "SFIRE_CORRECTIONS.patch").write_text(patch)
    receipt = {"schema": "sfire-native-correction-v1", "pinned_driver_sha256": sha(original),
               "corrected_driver_sha256": sha(corrected), "corrections": corrections,
               "changed_sources": {name: {"original_sha256": sha(old), "corrected_sha256": sha(new)}
                                   for name, (old, new) in source_changes.items()},
               "scope": "constant native timestep reference; WOOF-ArWen uses the actual integer calendar for variable timesteps",
               "cause": "solve_em increments itimestep before physics, while fire_model advances from time_start to time_start+dt"}
    (destination / "SFIRE_CORRECTIONS.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--wind-patch", type=Path)
    args = parser.parse_args()
    print(json.dumps(correct(args.source, args.destination, args.wind_patch), sort_keys=True))
