#!/usr/bin/env python3
"""Re-read the selected Noah-MP frames that ``noahmp_selected_frames`` prices.

  python tools/measure_selected_noahmp_frames.py            # read and print
  python tools/measure_selected_noahmp_frames.py --write    # read, then rewrite READINGS

Every row of ``woof/core/noahmp_selected_frames.py`` (and its SNICAR
``snow_papers.cu`` row) is bound to the identity of the source it was read
from.  Any edit to a selected unit withdraws the reading and prices that unit
at the assumed frame bound (65,136 B/thread today), which priced a 90x90
``opt_alb=3`` domain at 14-16 GiB instead of about 3 and refused it on 8-24 GB
cards.  ``tests/test_noahmp_selected_frames.py`` fails while any row is stale;
this tool is the one command that clears it.

Each mode (fast, then strict with ``GPUWM_WRF_EXACT=1``) is read in a FRESH
process through the production compile route (``compile_runtime_unit`` /
``_compile_loaded_module``), from ``get_function(...).attributes`` on every
exported kernel; no launches.  The row takes the larger of the two modes.
Run it on a card of the platform in ``MEASURED_PLATFORM`` (sm_120, the NVRTC
build the package resolves); one visible device.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "woof" / "core" / "noahmp_selected_frames.py"
SNOW = ROOT / "woof" / "core" / "kernels" / "snow_papers.cu"


def _read_here() -> dict:
    sys.path.insert(0, str(ROOT))
    import cupy as cp
    from woof.certify.compile_platform import compile_platform_fingerprint
    from woof.core.noahmp_kernel_sources import (
        _compile_loaded_module, compile_runtime_unit, exported_kernels, runtime_unit)
    from woof.core.noahmp_selected_frames import READINGS
    if cp.cuda.runtime.getDeviceCount() != 1:
        raise SystemExit("one visible CUDA device, please (CUDA_VISIBLE_DEVICES)")
    fp = compile_platform_fingerprint()
    rows = {}
    for name, key, _identity, _frame in READINGS:
        unit = runtime_unit(name, selected_options=True)
        module = compile_runtime_unit(name, module_key=f"selected_frame_read:{name}",
                                      selected_options=True)
        local = {s: int(module.get_function(s).attributes["local_size_bytes"])
                 for s in exported_kernels(unit.source)}
        rows[name] = {"key": key, "identity": unit.identity()["identity_sha256"],
                      "frame": max(local.values()), "functions": local}
    source = SNOW.read_text()
    module = _compile_loaded_module(source, ("-std=c++17", "--ftz=false"),
                                    "selected_frame_read:snow_papers")
    local = {s: int(module.get_function(s).attributes["local_size_bytes"])
             for s in exported_kernels(source)}
    # Two images share the source: the default kernel (with the packing and
    # probe exports) and the Toon image snow_papers_albedo_layers, priced only
    # when snicar_solver_opt=2 selects it (noahmp_selected_frames, snow_layers).
    layers = {"snow_papers_albedo_layers", "snow_column_probe"}
    rows["snow_papers"] = {"key": "snow_papers",
                           "identity": hashlib.sha256(SNOW.read_bytes()).hexdigest(),
                           "frame": max(v for s, v in local.items() if s not in layers),
                           "frame_layers": max(v for s, v in local.items() if s in layers),
                           "functions": local}
    return {"platform": [str(fp.get("device_compute_capability")), str(fp.get("nvrtc_build"))],
            "strict": os.environ.get("GPUWM_WRF_EXACT", "0") == "1", "rows": rows}


def _fresh(strict: bool) -> dict:
    env = dict(os.environ, GPUWM_WRF_EXACT="1" if strict else "0")
    out = subprocess.run([sys.executable, __file__, "--worker"], env=env, cwd=ROOT,
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def _write(rows: dict) -> None:
    text = TABLE.read_text()
    for name, row in rows.items():
        if name == "snow_papers":
            text, n = re.subn(r'(if digest == ")[0-9a-f]{64}(":\n\s*frames\["snow_papers"\] = )\d+ if snow_layers else \d+',
                              rf'\g<1>{row["identity"]}\g<2>{row["frame_layers"]} if snow_layers else {row["frame"]}', text)
        else:
            text, n = re.subn(rf"(\('{name}', '{row['key']}',\s*')[0-9a-f]{{64}}(', )\d+(\))",
                              rf"\g<1>{row['identity']}\g<2>{row['frame']}\g<3>", text)
        if n != 1:
            raise SystemExit(f"could not place the {name} row in {TABLE}")
    TABLE.write_text(text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        print(json.dumps(_read_here()))
        return 0
    fast, strict = _fresh(False), _fresh(True)
    sys.path.insert(0, str(ROOT))
    from woof.core.noahmp_selected_frames import MEASURED_PLATFORM
    if tuple(fast["platform"]) != tuple(MEASURED_PLATFORM):
        print(f"note: read on {fast['platform']}, table says {MEASURED_PLATFORM}")
    rows = {}
    for name, row in fast["rows"].items():
        other = strict["rows"][name]
        assert other["identity"] == row["identity"], name
        rows[name] = dict(row, frame=max(row["frame"], other["frame"]),
                          fast=row["frame"], strict_frame=other["frame"])
        if "frame_layers" in row:
            rows[name]["frame_layers"] = max(row["frame_layers"], other["frame_layers"])
        print(f"{name:18s} {row['identity'][:12]}  fast {row['frame']:5d}  strict {other['frame']:5d}")
    print(json.dumps({"platform": fast["platform"], "rows": rows}, indent=1),
          file=open(ROOT / "selected-frames-reading.json", "w"))
    if args.write:
        _write(rows)
        print(f"wrote {TABLE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
