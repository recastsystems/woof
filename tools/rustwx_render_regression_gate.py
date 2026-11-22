#!/usr/bin/env python3
"""Pixel-regression gate for the vendored Rust renderer's EXISTING products.

WHAT BREAKAGE THIS PREVENTS (gate law): a shared change made for
a NEW lane -- a ``StaticPlotDesign`` tweak, a style-ladder entry, a frame
aspect-ratio adjustment, an extra field published by the import -- silently
restyles or renames every product ``rw_wrfbatch`` already draws.  Nothing
else in the tree compares rendered PIXELS across a code change: the render
tests assert names, sizes and catalog counts, all of which survive a
restyle untouched.

It is the ``tools/nc_rewrite_render_parity.py`` pattern (drive the real
exe, byte-compare every PNG) turned ninety degrees: that one holds the
BINARY fixed and varies the input file; this one holds the INPUT fixed and
varies the binary.

    # before touching the renderer
    python tools/rustwx_render_regression_gate.py record \
        --rw-wrfbatch tools/rustwx/target/release/rw_wrfbatch.exe \
        --work work/render-gate

    # after every change, with the rebuilt binary
    python tools/rustwx_render_regression_gate.py check \
        --rw-wrfbatch tools/rustwx/target/release/rw_wrfbatch.exe \
        --work work/render-gate

``record`` writes ``baseline.json`` (one SHA-256 per rendered PNG plus the
renderer's own ``CATALOG``/``FINISHED`` tallies); ``check`` re-renders and
refuses on any hash, any missing product, and any new product.  A NEW
product appearing is a failure too: ``--products all`` expands from the
store catalog, so an extra 2-D plane published for a new lane joins every
existing caller's render silently.

The fixture is the repo's own render fixture -- the two-frame wrfout
``tests/test_render_rust.py`` builds with the production ``WrfoutWriter``
and the production global-attribute profile, whose second frame is on the
half hour so the exact-time axis is covered.  It is rebuilt here rather
than imported so the gate runs without pytest.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

_NZ, _NY, _NX = 4, 12, 16
_STAMPS = ("1974-04-03_18:00:00", "1974-04-03_18:30:00")


def _frame(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    lat = np.tile(np.linspace(38.0, 40.0, _NY)[:, None], (1, _NX))
    lon = np.tile(np.linspace(-98.0, -95.0, _NX)[None, :], (_NY, 1))
    return {
        "T": np.zeros((_NZ, _NY, _NX), np.float32),
        "MU": np.zeros((_NY, _NX), np.float32),
        "REFL_10CM": rng.uniform(-20.0, 65.0,
                                 (_NZ, _NY, _NX)).astype(np.float32),
        "T2": rng.uniform(280.0, 300.0, (_NY, _NX)).astype(np.float32),
        "Q2": rng.uniform(0.004, 0.012, (_NY, _NX)).astype(np.float32),
        "PSFC": rng.uniform(96000.0, 98000.0, (_NY, _NX)).astype(np.float32),
        "U10": rng.uniform(-10.0, 10.0, (_NY, _NX)).astype(np.float32),
        "V10": rng.uniform(-10.0, 10.0, (_NY, _NX)).astype(np.float32),
        "RAINC": rng.uniform(0.0, 5.0, (_NY, _NX)).astype(np.float32),
        "RAINNC": rng.uniform(0.0, 30.0, (_NY, _NX)).astype(np.float32),
        "XLAT": lat.astype(np.float32),
        "XLONG": lon.astype(np.float32),
        "HGT": np.zeros((_NY, _NX), np.float32),
        "SINALPHA": np.zeros((_NY, _NX), np.float32),
        "COSALPHA": np.ones((_NY, _NX), np.float32),
    }


def build_fixture(path: pathlib.Path) -> pathlib.Path:
    """The exact wrfout ``tests/test_render_rust.py::wrfout`` builds."""

    from woof.io.wrfout import WrfoutWriter, wrf_global_attrs

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    grid = SimpleNamespace(truelat1=38.5, truelat2=39.5, stand_lon=-96.5,
                           ref_lat=39.0, ref_lon=-96.5)
    attrs = wrf_global_attrs(
        grid, datetime.datetime(1974, 4, 3, 18), grid_id=2, parent_id=1,
        i_parent_start=5, j_parent_start=5, parent_grid_ratio=3, dt=6.0)
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=1000.0, dy=1000.0,
                      global_attrs=attrs) as writer:
        for index, stamp in enumerate(_STAMPS):
            writer.write_frame(stamp, _frame(seed=7 + index))
    return path


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def render(exe: str, source: pathlib.Path, work: pathlib.Path, tag: str,
           width: int, height: int, products: str) -> tuple[pathlib.Path, str,
                                                            str]:
    store = work / f"store_{tag}"
    out = work / f"out_{tag}"
    for folder in (store, out):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
    command = [
        exe,
        "--store-root", str(store),
        "--out-dir", str(out),
        "--products", products,
        "--frames", "all",
        "--width", str(width),
        "--height", str(height),
        # Pinned: the default label carries the executing woof version on
        # the woof side, but rw_wrfbatch's own default is a constant.  It
        # is stated anyway so the gate cannot drift with a default.
        "--source-label", "WOOF-ArWen",
        str(source),
    ]
    result = subprocess.run(command, capture_output=True, text=True,
                            errors="replace")
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-4000:])
        sys.stderr.write(result.stderr[-4000:])
        raise SystemExit(
            f"rw_wrfbatch failed on {source} (exit {result.returncode})")
    catalog = ""
    finished = ""
    for line in result.stdout.splitlines():
        if line.startswith("CATALOG "):
            catalog = line
        elif line.startswith("FINISHED "):
            finished = line
    return out, catalog, finished


def snapshot(exe: str, work: pathlib.Path, width: int, height: int,
             products: str) -> dict:
    fixture = build_fixture(work / "fixture" /
                            "wrfout_d02_1974-04-03_18-00-00.nc")
    out, catalog, finished = render(exe, fixture, work, "current", width,
                                    height, products)
    pngs = sorted(p for p in out.rglob("*.png"))
    if not pngs:
        raise SystemExit("the fixture rendered no PNGs; the gate has no "
                         "subject")
    return {
        "fixture_sha256": sha256(fixture),
        "width": width,
        "height": height,
        "products": products,
        "catalog": catalog,
        "finished": finished,
        "pngs": {p.relative_to(out).as_posix(): sha256(p) for p in pngs},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("record", "check"))
    parser.add_argument("--rw-wrfbatch", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--baseline", default=None,
                        help="baseline JSON (default: <work>/baseline.json)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--products", default="all")
    args = parser.parse_args(argv)

    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    baseline_path = (pathlib.Path(args.baseline) if args.baseline
                     else work / "baseline.json")

    current = snapshot(args.rw_wrfbatch, work, args.width, args.height,
                       args.products)
    if args.action == "record":
        baseline_path.write_text(json.dumps(current, indent=2, sort_keys=True),
                                 encoding="utf-8")
        print(f"RECORDED pngs={len(current['pngs'])} -> {baseline_path}")
        print(f"  {current['catalog']}")
        print(f"  {current['finished']}")
        return 0

    if not baseline_path.is_file():
        raise SystemExit(f"no baseline at {baseline_path}; run `record` "
                         "with the pre-change binary first")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    problems: list[str] = []
    for key in ("fixture_sha256", "width", "height", "products"):
        if baseline.get(key) != current.get(key):
            problems.append(
                f"{key}: baseline {baseline.get(key)!r} vs now "
                f"{current.get(key)!r} -- the gate is comparing two "
                "different questions")
    before = baseline.get("pngs", {})
    after = current["pngs"]
    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    for name in missing:
        problems.append(f"{name}: rendered before, absent now")
    for name in extra:
        problems.append(
            f"{name}: NEW product in `--products all` -- every existing "
            "caller now renders it too")
    identical = 0
    for name in sorted(set(before) & set(after)):
        if before[name] == after[name]:
            identical += 1
        else:
            problems.append(
                f"{name}: {before[name][:16]} vs {after[name][:16]}")

    print(f"COMPARED pngs={len(before)} identical={identical}")
    print(f"  baseline {baseline.get('catalog', '')}")
    print(f"  now      {current['catalog']}")
    if problems:
        print(f"RENDER REGRESSION FAIL differences={len(problems)}")
        for problem in problems[:40]:
            print(f"  - {problem}")
        return 1
    print(f"RENDER REGRESSION PASS all {identical} existing PNG(s) are "
          "byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
