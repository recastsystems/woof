"""Render every frame of a cells case with its overlay into an evidence gallery.

Usage: python tools/cells_gallery.py <series-root> <wrfout-glob> <gallery-dir>
           [--products refl] [--no-render]

<series-root> is the `<case>/<domain>/cells/<day>/` folder woof cells analyze
wrote; each overlay under overlays/ is matched to its wrfout by valid stamp.
A frame whose PNG already sits in <gallery-dir> (named <stamp>_*.png) is not
rendered again; `--no-render` only assembles what is there.  Renders go
through `woof render --overlays` (the Rust renderer, rw_wrfbatch).  Writes
the PNGs, a captions README.md and an LF-terminated SHA256SUMS.txt that
`sha256sum -c` accepts; prints per-frame wall time.
"""

from __future__ import annotations

import glob
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
root = Path(sys.argv[1])
pattern = sys.argv[2]
gallery = Path(sys.argv[3])
products = "refl"
if "--products" in sys.argv:
    products = sys.argv[sys.argv.index("--products") + 1]
no_render = "--no-render" in sys.argv
gallery.mkdir(parents=True, exist_ok=True)

catalog = json.loads((root / "catalog.json").read_text("utf-8"))
rows = catalog["rows"]
by_stamp: dict[str, list[dict]] = {}
for row in rows:
    by_stamp.setdefault(row["valid_time"], []).append(row)

overlays = {re.search(r"cells_(\d{8}T\d{6}Z)\.json", p.name).group(1): p
            for p in (root / "overlays").glob("cells_*.json")}
wrfouts = sorted(glob.glob(pattern))


def stamp_of(path: str) -> str | None:
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})[_T](\d{2})[:_-](\d{2})[:_-](\d{2})", Path(path).name)
    if not m:
        return None
    return f"{m.group(1)}{m.group(2)}{m.group(3)}T{m.group(4)}{m.group(5)}{m.group(6)}Z"


captions = ["# Storm cells over WOOF-ArWen reflectivity\n",
            f"Series root: `{root}`\n",
            "Each image is WOOF-ArWen's composite reflectivity for one frame, drawn by the "
            "Rust renderer (rw_wrfbatch), with the storm cells titan identified in "
            "that frame outlined in colour by track (thick line = the cell's footprint "
            "now; thin line in the same colour = titan's forecast footprint at the first "
            "lead). The label at each centroid is the track id, the cell's lifetime so "
            "far in minutes, and WOOF-ArWen's peak updraft inside the footprint in m/s.\n",
            "| frame | cells | tracks | strongest updraft | caption |",
            "|---|---|---|---|---|"]
timings = []
written = []
rendered_here = 0
for path in wrfouts:
    stamp = stamp_of(path)
    if stamp is None or stamp not in overlays:
        continue
    existing = sorted(gallery.glob(f"{stamp}_*.png"))
    seconds = None
    if existing:
        pngs = [str(p) for p in existing]
    elif no_render:
        continue
    else:
        overlay = overlays[stamp]
        out = gallery / "render"
        tick = time.perf_counter()
        cmd = [PY, "-m", "woof.cli", "render", path, "--products", products,
               "--overlays", str(overlay), "--out", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        seconds = time.perf_counter() - tick
        pngs = [line.split(": ", 1)[1].strip() for line in proc.stdout.splitlines()
                if line.startswith("render: ") and line.strip().endswith(".png")]
        if proc.returncode != 0 or not pngs:
            print("FAILED", path, proc.stderr[-400:])
            continue
        rendered_here += 1
    valid = rows and next((r["valid_time"] for r in rows
                           if r["valid_time"].replace("-", "").replace(":", "") == stamp), None)
    frame_rows = by_stamp.get(valid, []) if valid else []
    tracks = sorted({r["track_id"] for r in frame_rows if r["track_id"] is not None})
    peak = max((r["peak_w_mps"] for r in frame_rows if r.get("peak_w_mps") is not None), default=None)
    strongest = max(frame_rows, key=lambda r: r.get("peak_w_mps") or -1) if frame_rows else None
    for png in pngs:
        src = Path(png)
        dst = src if src.parent == gallery else gallery / f"{stamp}_{src.name}"
        if dst != src:
            shutil.copyfile(src, dst)
        written.append(dst)
        text = (f"{valid}: titan found {len(frame_rows)} cells on {len(tracks)} tracks. "
                + (f"The strongest updraft is {peak:.1f} m/s ({peak * 196.85:.0f} ft/min) in track "
                   f"{strongest['track_id']} (lifetime {strongest['lifetime_so_far_s'] / 60:.0f} min, "
                   f"area {strongest['projected_area_km2']:.0f} km2, max {strongest['max_dbz']:.0f} dBZ, "
                   f"cloud top {strongest['cloud_top_m_msl']:.0f} m MSL at {strongest['cloud_top_temperature_c']:.0f} C)."
                   if strongest and peak is not None else "No cell met the threshold in this frame."))
        captions.append(f"| {dst.name} | {len(frame_rows)} | {len(tracks)} | "
                        f"{'' if peak is None else f'{peak:.1f} m/s'} | {text} |")
    if seconds is not None:
        timings.append(seconds)
    print(f"{stamp}: {len(frame_rows)} cells, "
          f"{'reused' if seconds is None else f'{seconds:.1f} s'}")

shutil.rmtree(gallery / "render", ignore_errors=True)
shutil.rmtree(gallery / "render.render-scratch", ignore_errors=True)
captions.append("")
if timings:
    captions.append(f"Render wall time on {platform.node()}: {sum(timings):.1f} s over "
                    f"{len(timings)} frames (mean {sum(timings) / max(len(timings), 1):.1f} s/frame).")
(gallery / "README.md").write_text("\n".join(captions) + "\n", encoding="utf-8", newline="\n")
with open(gallery / "SHA256SUMS.txt", "w", encoding="utf-8", newline="\n") as handle:
    for dst in sorted(written):
        handle.write(f"{hashlib.sha256(dst.read_bytes()).hexdigest()}  {dst.name}\n")
print(f"gallery: {len(written)} PNGs ({rendered_here} rendered now), {sum(timings):.1f} s")
