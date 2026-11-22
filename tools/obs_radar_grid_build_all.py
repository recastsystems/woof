"""Build ONE observation file for ONE analysis time from EVERY radar that sees it.

The sibling of ``tools/obs_radar_grid_build.py``, which builds the same
file from one named site.  The rule that tool states -- one invocation,
one observation file, one analysis time, one georeference -- is kept here
exactly.  What changes is that the site list is discovered from coverage
rather than typed, and that a site failing is a line in the receipt rather
than the end of the run.

    python -m tools.obs_radar_grid_build_all \
        --valid-time 2026-08-01T11:30:00Z \
        --grid-wrfout wrfout_d01 --out obs.nc --work-dir work/

Discovery uses the analysis grid itself as the domain: a site is in when
some fraction of the grid's mass points lies within its usable range.
``--domain-polygon`` overrides that with a GeoJSON ring, for the callers
that discover before they have a georeference.

Exit status is 0 when at least ``--min-radars`` sites contributed, so a
cycle that silently lost its radars fails loudly instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _parse_iso(text: str) -> datetime:
    value = text.strip().replace("Z", "+00:00")
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def _iso(stamp: datetime) -> str:
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _ring_from_geojson(path: Path):
    """The outer ring of the first polygon in a GeoJSON document."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("type") == "FeatureCollection":
        doc = doc["features"][0]
    if doc.get("type") == "Feature":
        doc = doc["geometry"]
    kind = doc.get("type")
    if kind == "Polygon":
        return doc["coordinates"][0]
    if kind == "MultiPolygon":
        return doc["coordinates"][0][0]
    raise SystemExit(f"{path}: need a Polygon or MultiPolygon, got {kind!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tools.obs_radar_grid_build_all",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--valid-time", required=True)
    p.add_argument("--grid-wrfout", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--work-dir", required=True, type=Path)
    p.add_argument("--receipt", type=Path, default=None,
                   help="write the ingest receipt here as well as stdout")

    disc = p.add_argument_group("discovery")
    disc.add_argument("--domain-polygon", type=Path, default=None,
                      help="GeoJSON ring to discover against, instead of "
                           "the analysis grid's own mass points")
    disc.add_argument("--min-coverage-fraction", type=float, default=0.0,
                      help="keep only sites seeing at least this fraction "
                           "of the domain (default: any overlap at all)")
    disc.add_argument("--min-radars", type=int, default=1,
                      help="fail the build if fewer than this many sites "
                           "CONTRIBUTED (default: 1)")
    disc.add_argument("--max-radars", type=int, default=None,
                      help="cap the roster at the best-covering N sites")
    disc.add_argument("--sites", default=None,
                      help="comma-separated ICAO ids: skip discovery and "
                           "use exactly these (still coverage-scored)")
    disc.add_argument("--discovery-stride", type=int, default=4,
                      help="sample every Nth grid point when scoring "
                           "coverage (default: 4)")

    fetch = p.add_argument_group("fetch")
    fetch.add_argument("--workers", type=int, default=None,
                       help="native site pipelines (default: site count bounded by RAM)")
    fetch.add_argument("--compute-workers", type=int, default=None,
                       help="optional lower bound on concurrent native superob stages")
    fetch.add_argument("--source", default="auto",
                       choices=("archive", "auto", "live"))
    fetch.add_argument("--bucket", default=None)
    fetch.add_argument("--live-bucket", default=None)
    fetch.add_argument("--cache-dir", type=Path, default=None)
    fetch.add_argument("--allow-partial", action="store_true")
    fetch.add_argument("--min-chunks", type=int, default=None)
    fetch.add_argument("--max-offset-seconds", type=float, default=480.0)

    sup = p.add_argument_group("superob")
    sup.add_argument("--max-range-km", type=float, default=None)
    sup.add_argument("--max-elevation-deg", type=float, default=20.0)
    sup.add_argument("--z-reduce", default="mean", choices=("max", "mean"))
    sup.add_argument("--overwrite", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from woof.obs.allradar import (discover_sites, grid_sample_points,
                                    ingest_domain, polygon_sample_points)
    from woof.obs.nexrad import find_nexrad_bin, nexrad_remedy, run_sites
    from woof.obs.radar_grid import write_radar_grid
    from woof.obs.superob import SuperobParams
    from woof.obs.target_grid import TargetGrid

    binary = find_nexrad_bin()
    if binary is None:
        raise SystemExit(f"no rw_nexrad front door: {nexrad_remedy()}")

    target = _parse_iso(args.valid_time)
    range_km = (args.max_range_km if args.max_range_km is not None
                else SuperobParams().max_range_km)
    params = SuperobParams(max_range_km=range_km,
                           max_elevation_deg=args.max_elevation_deg)
    params.validate()

    grid = TargetGrid.from_wrfout(args.grid_wrfout)

    # ---- discovery ----------------------------------------------------
    if args.domain_polygon is not None:
        ring = _ring_from_geojson(args.domain_polygon)
        sample_lat, sample_lon = polygon_sample_points(ring)
        domain = {"kind": "polygon", "path": str(args.domain_polygon)}
    else:
        sample_lat, sample_lon = grid_sample_points(
            grid, stride=max(1, args.discovery_stride))
        domain = {"kind": "analysis-grid",
                  "stride": max(1, args.discovery_stride)}
    domain["sample_points"] = int(sample_lat.size)

    catalog = run_sites(binary)["sites"]
    if args.sites:
        want = {s.strip().upper() for s in args.sites.split(",") if s.strip()}
        missing = want - {str(s["id"]).upper() for s in catalog}
        if missing:
            raise SystemExit(
                f"not in the site table: {', '.join(sorted(missing))}")
        catalog = [s for s in catalog if str(s["id"]).upper() in want]

    coverages = discover_sites(
        catalog, sample_lat, sample_lon, range_km=range_km,
        min_coverage_fraction=args.min_coverage_fraction,
        limit=args.max_radars)
    if not coverages:
        raise SystemExit(
            "no radar in the site table covers this domain at "
            f"--max-range-km {range_km} and --min-coverage-fraction "
            f"{args.min_coverage_fraction}")

    print(f"discovered {len(coverages)} sites covering the domain",
          file=sys.stderr)

    from tools.nexrad_archive import (require_native_observations,
        observation_workers, observation_memory, native_observation_threads)
    from woof.obs.coverage import read_site_table
    require_native_observations()
    site_table = read_site_table(binary)
    native_threads = native_observation_threads(len(coverages), requested=args.workers)
    retained = observation_memory(grid, [site_table[c.site] for c in coverages], params)
    sizing = observation_workers(len(coverages), requested=args.workers,
                                 retained_slot_bytes=retained)
    sizing['native_threads'] = native_threads
    compute_workers = min(sizing['workers'], args.compute_workers or sizing['workers'])
    if compute_workers < 1:
        raise ValueError('compute workers must be positive; zero would suppress radar processing')

    def _progress(outcome):
        mark = "ok " if outcome.contributed else "FAIL"
        print(f"  [{mark}] {outcome.site} {outcome.status}"
              + (f": {outcome.reason}" if outcome.reason else ""),
              file=sys.stderr)

    result = ingest_domain(
        binary, coverages, grid=grid, work_dir=args.work_dir,
        valid_time=target, params=params, workers=sizing['workers'],
        compute_workers=compute_workers,
        z_reduce=args.z_reduce, progress=_progress,
        source=args.source, bucket=args.bucket,
        live_bucket=args.live_bucket, cache_dir=args.cache_dir,
        max_offset_seconds=args.max_offset_seconds,
        allow_partial=args.allow_partial, min_chunks=args.min_chunks,
        max_elevation_deg=args.max_elevation_deg)

    payload = result.to_payload()
    payload['worker_sizing'] = sizing
    payload["requested_valid_time"] = _iso(target)
    payload["domain"] = domain
    payload["range_authority_km"] = range_km
    payload["discovery"] = {
        "min_coverage_fraction": args.min_coverage_fraction,
        "max_radars": args.max_radars,
        "catalog_size": len(catalog),
        "sites": [c.to_payload() for c in coverages],
    }

    contributing = result.contributing
    if len(contributing) < args.min_radars:
        payload["status"] = "REFUSED"
        payload["reason"] = (
            f"{len(contributing)} of {len(coverages)} discovered sites "
            f"contributed, below --min-radars {args.min_radars}")
        _emit(payload, args.receipt)
        return 1

    receipt = write_radar_grid(
        args.out, result.observations, grid,
        valid_time=_iso(target), params=params,
        provenance={
            "builder": "tools/obs_radar_grid_build_all.py",
            "sites_contributing": len(contributing),
            "sites_discovered": len(coverages),
            "sites": [o.to_payload() for o in result.outcomes],
            "domain": domain,
            "grid_wrfout": str(args.grid_wrfout),
            "range_authority_km": range_km,
            "requested_valid_time": _iso(target),
        },
        overwrite=args.overwrite)
    payload["status"] = "OK"
    payload["writer_receipt"] = receipt
    payload["pairing"] = {"obs": str(args.out),
                          "grid_wrfout": str(args.grid_wrfout)}
    _emit(payload, args.receipt)
    return 0


def _emit(payload: dict, receipt: Path | None) -> None:
    text = json.dumps(payload, indent=2, default=str)
    print(text)
    if receipt is not None:
        Path(receipt).parent.mkdir(parents=True, exist_ok=True)
        Path(receipt).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
