#!/usr/bin/env python3
"""Radar reflectivity on the model grid, one product per 15-minute window.

The radar latent heating forcing reads NOAA's ``ref2tten`` reflectivity
field four times an hour ahead of each forecast leg.  This tool makes those
fields from raw NEXRAD Level-II volumes for a list of window end times:

* discovery, listing and the causal volume choice are
  ``tools/nexrad_archive.py``'s, reused rather than retold: every
  NEXRAD site whose 250 km disc touches the grid (the decoder's own site
  table), one S3 listing per site-day, candidates whose key time lies in
  ``[end - 900 s, end]``, no ``_MDM`` objects, and no object whose
  LastModified is after the window end (``causal_object``);
* the newest few candidates per site are downloaded into ``OUT/raw`` (the
  same bucket/key layout and size check as ``download``), so overlapping
  windows reuse the same files;
* one ``rw_nexrad grid-ref`` call per window decodes and grids them all.
  It applies the measured-end half of the causal rule itself
  (``--window-end``, ``--max-age-s 900``: per site, the newest complete
  volume that ended at or before the window end and at most 900 s before
  it), so the choice is made on the volume's own radial clocks.

The grid is either a wrfout (``--grid-wrfout``) or a prepared root
(``--grid-prepared``).  A prepared root is read by
``woof.da.forecast_heating.grid_for_prepared``, the same function the
deterministic door computes its grid identity with, so windows made for a
prepared forecast pass the door's strict reader by construction.  Windows
made from a wrfout carry that file's identity and are refused by a run on a
prepared root whose columns are not the same.

Output::

    OUT/grid/grid.json, grid.z_w.f64      the grid descriptor, written once
    OUT/raw/<bucket>/<key>                cached Level-II volumes
    OUT/<YYYYmmddTHHMMZ>/ref.f32, ref.json the product and its receipt
    OUT/windows.json                       the index

Python here only lists, downloads and calls; every gate is handled in Rust.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.nexrad_archive import (MAX_AGE_S, RADAR_BUCKET, RANGE_KM,  # noqa: E402
                                         MAX_ELEVATION_DEG, download, iso,
                                         list_objects, object_time, save, sha,
                                         utc)

INDEX_SCHEMA = "gpuwm-obs.radar-tten-windows.v1"
#: Candidates per site handed to grid-ref, newest key first.  A volume
#: still being scanned at the window end is refused on its measured end, so
#: the second newest is the usual pick; three covers a refused pair.
CANDIDATES_PER_SITE = 3


def stamp(when: dt.datetime) -> str:
    return when.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%MZ")


def window_ends(args) -> list[dt.datetime]:
    if args.windows:
        ends = [utc(value) for value in args.windows]
    else:
        start, end = utc(args.start), utc(args.end)
        step = dt.timedelta(minutes=args.step_minutes)
        ends = []
        when = start
        while when <= end:
            ends.append(when)
            when += step
    if not ends:
        raise SystemExit("no window end times")
    return sorted(set(ends))


def candidates_for(listing, end, *, per_site: int):
    """Causal candidates for one site and window, newest key first."""
    from tools.nexrad_archive import causal_object
    rows = [row for row in listing
            if object_time(row["key"]) is not None
            and end - dt.timedelta(seconds=MAX_AGE_S) <= object_time(row["key"]) <= end
            and not row["key"].endswith("_MDM")]
    rows.sort(key=lambda row: object_time(row["key"]), reverse=True)
    kept, refused = [], []
    for row in rows:
        try:
            basis = causal_object(row, end)
        except ValueError as error:
            refused.append({"key": row["key"], "reason": str(error)})
            continue
        kept.append({**row, "causal_basis": basis})
    return kept[:per_site], refused


def target_grid(args):
    """``(grid, source)`` from ``--grid-wrfout`` or ``--grid-prepared``."""
    from woof.obs.target_grid import TargetGrid

    if getattr(args, "grid_prepared", None):
        from woof.da.forecast_heating import grid_for_prepared
        root = Path(args.grid_prepared).resolve()
        return grid_for_prepared(root), {"prepared_root": str(root)}
    path = Path(args.grid_wrfout).resolve()
    return TargetGrid.from_wrfout(path), {"wrfout": str(path)}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    grid_source = parser.add_mutually_exclusive_group(required=True)
    grid_source.add_argument("--grid-wrfout",
                             help="wrfout/wrfinput whose grid the product is placed on")
    grid_source.add_argument("--grid-prepared",
                             help="prepared root (the door's --prepared-root) whose "
                                  "start-state grid the product is placed on")
    parser.add_argument("--out", required=True)
    parser.add_argument("--windows", nargs="*",
                        help="window end times, e.g. 2026-10-01T19:15:00Z")
    parser.add_argument("--start", help="first window end (with --end)")
    parser.add_argument("--end", help="last window end")
    parser.add_argument("--step-minutes", type=int, default=15)
    parser.add_argument("--reduce", choices=("mean", "max"), default="mean")
    parser.add_argument("--threads", type=int, default=None,
                        help="grid-ref worker threads (default: the binary's)")
    parser.add_argument("--download-workers", type=int, default=32)
    parser.add_argument("--candidates-per-site", type=int, default=CANDIDATES_PER_SITE)
    parser.add_argument("--raw-cache", default=None,
                        help="read-only volume tree with the same <bucket>/<key> layout")
    parser.add_argument("--limit-sites", type=int, default=0)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.windows and not (args.start and args.end):
        parser.error("give --windows or --start/--end")

    from woof.obs.coverage import read_site_table, sites_covering
    from woof.obs.nexrad import find_nexrad_bin, nexrad_remedy
    from woof.obs.radar_tten_grid import (grid_reflectivity,
                                           write_grid_descriptor)

    t_all = time.perf_counter()
    binary = find_nexrad_bin()
    if binary is None:
        raise SystemExit(nexrad_remedy())
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw_root = out / "raw"
    ends = window_ends(args)

    t0 = time.perf_counter()
    grid, grid_path = target_grid(args)
    descriptor_path = out / "grid" / "grid.json"
    write_grid_descriptor(grid, descriptor_path)
    table = read_site_table(binary)
    found = sites_covering(grid, table, max_range_km=RANGE_KM, min_coverage_fraction=0.0)
    sites = [entry.site.id for entry in found]
    if args.limit_sites:
        sites = sites[:args.limit_sites]
    grid_seconds = time.perf_counter() - t0
    print(f"grid {grid.nx}x{grid.ny}x{grid.nz}; {len(sites)} of {len(table)} sites reach it",
          flush=True)

    t0 = time.perf_counter()
    days = sorted({(end - dt.timedelta(seconds=MAX_AGE_S)).date() for end in ends}
                  | {end.date() for end in ends})
    listing: dict[str, list] = {site: [] for site in sites}
    listing_errors = []
    with cf.ThreadPoolExecutor(max_workers=48) as pool:
        jobs = {pool.submit(lambda s, d: list(list_objects(RADAR_BUCKET, f"{d:%Y/%m/%d}/{s}/")),
                            s, d): (s, d) for s in sites for d in days}
        for job in cf.as_completed(jobs):
            s, d = jobs[job]
            try:
                listing[s].extend(job.result())
            except OSError as error:
                listing_errors.append({"site": s, "day": str(d), "error": str(error)})
    listing_seconds = time.perf_counter() - t0
    print(f"listing {listing_seconds:.1f} s, {sum(len(v) for v in listing.values())} objects",
          flush=True)

    windows = []
    for end in ends:
        t_window = time.perf_counter()
        work = out / stamp(end)
        work.mkdir(parents=True, exist_ok=True)
        rows, refused_objects, no_candidates = [], [], []
        for site in sites:
            kept, refused = candidates_for(listing[site], end, per_site=args.candidates_per_site)
            rows.extend(kept)
            refused_objects.extend(refused)
            if not kept:
                no_candidates.append(site)
        t0 = time.perf_counter()
        volumes, fetched_bytes, download_errors = [], 0, []
        with cf.ThreadPoolExecutor(max_workers=args.download_workers) as pool:
            jobs = {pool.submit(download, row, raw_root, cache=args.raw_cache): row for row in rows}
            for job in cf.as_completed(jobs):
                row = jobs[job]
                try:
                    path, _, fetched = job.result()
                except OSError as error:
                    download_errors.append({"key": row["key"], "error": str(error)})
                    continue
                volumes.append(path)
                if fetched:
                    fetched_bytes += row["bytes"]
        download_seconds = time.perf_counter() - t0
        volumes.sort()
        t0 = time.perf_counter()
        _, receipt = grid_reflectivity(
            grid, volumes, work / "ref", reduce=args.reduce, max_range_km=RANGE_KM,
            max_elevation_deg=MAX_ELEVATION_DEG, threads=args.threads,
            window_end=iso(end), max_age_s=MAX_AGE_S,
            grid_descriptor=descriptor_path, out_data=work / "ref.f32", binary=binary)
        grid_ref_seconds = time.perf_counter() - t0
        counts = receipt["counts"]
        entry = {
            "window_end": iso(end),
            "dir": work.name,
            "ref": "ref.f32",
            "receipt": "ref.json",
            "ref_sha256": receipt["data"]["sha256"],
            "shape": receipt["shape"],
            "reduce": args.reduce,
            "volumes_offered": len(volumes),
            "volumes_used": counts["volumes_used"],
            "volumes_refused": counts["volumes_refused"],
            "sites_used": sorted({v["site"] for v in receipt["volumes"]}),
            "sites_without_candidates": no_candidates,
            "objects_refused_last_modified": refused_objects,
            # Archive replay (da_recent_observations.ARCHIVE_RESTAMPS): offered
            # objects whose causality was judged by their key scan time.
            "objects_causal_by_scan_time": sum(
                row.get("causal_basis") == "archive-replay-scan-time" for row in rows),
            "download_errors": download_errors,
            "echo_cells": counts["echo_cells"],
            "clear_cells": counts["clear_cells"],
            "no_coverage_cells": counts["no_coverage_cells"],
            "bytes_fetched": fetched_bytes,
            "seconds": {"download": round(download_seconds, 2),
                        "grid_ref": round(grid_ref_seconds, 2),
                        "grid_ref_internal": receipt["seconds"],
                        "window": round(time.perf_counter() - t_window, 2)},
        }
        windows.append(entry)
        print(f"{work.name}: {counts['volumes_used']} volumes from {len(entry['sites_used'])} "
              f"sites, echo {counts['echo_cells']} clear {counts['clear_cells']} cells, "
              f"download {download_seconds:.1f} s, grid-ref {grid_ref_seconds:.1f} s", flush=True)
        save(out / "windows.json", _index(args, grid, grid_path, descriptor_path, binary,
                                          sites, days, listing_errors, windows, {
                                              "grid_and_discovery": round(grid_seconds, 2),
                                              "listing": round(listing_seconds, 2),
                                              "total": round(time.perf_counter() - t_all, 2)}))
    return 0


def _index(args, grid, grid_path, descriptor_path, binary, sites, days, listing_errors,
           windows, seconds):
    return {
        "schema": INDEX_SCHEMA,
        "builder": "tools/radar_tten_windows.py",
        "grid_source": grid_path,
        "grid_identity_sha256": grid.identity_sha256(),
        "grid_descriptor": str(descriptor_path),
        "shape": [int(grid.nz), int(grid.ny), int(grid.nx)],
        "binary": str(binary),
        "binary_sha256": sha(binary),
        "bucket": RADAR_BUCKET,
        "days_listed": [str(d) for d in days],
        "listing_errors": listing_errors,
        "sites_reaching_grid": sites,
        "range_km": RANGE_KM,
        "max_elevation_deg": MAX_ELEVATION_DEG,
        "max_age_s": MAX_AGE_S,
        "candidates_per_site": args.candidates_per_site,
        "reduce": args.reduce,
        "convention": {"echo": "dBZ", "observed_no_echo": -99.0, "no_coverage": -99999.0,
                       "dtype": "<f4", "order": "C", "dims": ["bottom_top", "south_north",
                                                              "west_east"]},
        "publication_policy": "archive-replay; LastModified is object metadata, not first publication proof",
        "windows": windows,
        "seconds": seconds,
    }


if __name__ == "__main__":
    os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
    raise SystemExit(main())
