"""Any packaged regional source onto the regular lat-lon WPS box the init and
boundary engines read: the second road of ``woof hex intermediate``.

The first road (:mod:`woof.hex.hrrr_intermediate`) drives HRRR's own native
decoder.  This one serves every source woof ships as a PACKAGED profile with
a table fetch route -- ECCC's HRDPS 2.5 km and RDPS 10 km (rotated lat-lon),
and any later regional row -- with no per-model code:

1. **fetch** -- ``woof fetch --source ROW`` over the cycle's hours (the
   route table names the objects, the hosts and the composed per-lead files);
2. **compose + resample** -- ``woof source-intermediate --fetch-dir
   --target-latlon``: the mapped engine composes the fetch once, decoding the
   atmosphere over the box's window only, rotates grid-relative winds to the
   earth basis, and writes one WPS v5 file per hour on the box (bilinear over
   finite corners, nearest for the land mask and sea ice, soil carried to
   coasts).  No weather array enters Python.

What a row needs is a row: ``woof.fetch_routes`` knows the source and its
``woof.source_adapters`` adapter names a packaged profile.  The init-source
table row (``data/init-sources.v1.json``, ``coverage: limited_area``) states
the first-guess level counts the receipt here is checked against.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

PACKAGED_SCHEMA = "gpuwm-hex.packaged-intermediate/v1"


def handles(name: str) -> bool:
    """Whether ``name`` is a packaged source with a table fetch route."""

    try:
        from woof import fetch_routes
        from woof.source_adapters import get_source_adapter
    except ImportError:
        return False
    try:
        if fetch_routes.table_route(name) is None:
            return False
        return bool(getattr(get_source_adapter(name), "packaged_profile", None))
    except Exception:  # an unknown name is simply not this road's
        return False


def source_table_row(source: str) -> dict[str, Any]:
    from .init_sources import load_table, resolve_source
    return resolve_source(load_table(), source, coverage="limited_area")


def coverage_window(row: dict[str, Any]):
    """Geometry declared by a row, available before a mesh or download exists."""
    from woof.source_coverage import LambertGridWindow, RegularLatLonWindow
    spec = dict(row.get("coverage_window") or {})
    if not spec:
        return None
    family = spec.pop("family")
    kinds = {"lambert_conformal": LambertGridWindow, "latitude_longitude": RegularLatLonWindow}
    if family not in kinds:
        raise ValueError(f"unknown coverage window family {family!r}")
    return kinds[family](**spec)


def coverage_refusal(row: dict[str, Any], centre: tuple[float, float], radius_km: float,
                     *, margin_km: float = 0.0) -> str | None:
    """Check a spherical region before mesh work, using its row's finite map.

    Breakage: a limited-area start otherwise downloads and builds a mesh before
    case 7 finds cells for which the input has no meteorology.
    """
    import numpy as np
    window = coverage_window(row)
    if window is None:
        return None
    angle = (float(radius_km) + float(margin_km)) / 6371.229
    if angle < 0 or not np.isfinite(angle):
        raise ValueError("coverage radius must be finite and nonnegative")
    bearing = np.linspace(0, 2 * np.pi, 1441)
    lat0, lon0 = np.radians(centre)
    lat = np.arcsin(np.sin(lat0) * np.cos(angle) + np.cos(lat0) * np.sin(angle) * np.cos(bearing))
    lon = lon0 + np.arctan2(np.sin(bearing) * np.sin(angle) * np.cos(lat0),
                           np.cos(angle) - np.sin(lat0) * np.sin(lat))
    lat = np.append(np.degrees(lat), centre[0])
    lon = np.append(np.degrees(lon), centre[1])
    if not window.outside(lat, lon).any():
        return None
    if hasattr(window, "grid"):
        grid = window.grid()
        x, y = grid.latlon_to_ij(lat, lon)
        edge_lat, edge_lon = grid.ij_to_latlon(np.clip(x, 1, window.nx), np.clip(y, 1, window.ny))
        a, b, c, d = map(np.radians, (lat, lon, edge_lat, edge_lon))
        distance = 2 * 6371.229 * np.arcsin(np.minimum(1, np.sqrt(
            np.sin((a-c)/2)**2 + np.cos(a)*np.cos(c)*np.sin((b-d)/2)**2)))
        overshoot = max(1, round(float(distance.max())))
    else:
        overshoot = max(1, round(float(radius_km)))
    label = row.get("coverage_label", row.get("display_name", row["id"]) + "'s map")
    return (f"This region reaches past {label} by about {overshoot} km. "
            "Move it inside the map or start from GFS.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_complete(grib_dir: Path, source: str, cycle, hours: tuple[int, ...], profile=None) -> bool:
    manifest = grib_dir / "fetch-manifest.json"
    if not manifest.is_file():
        return False
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    request = document.get("request") or {}
    return (document.get("complete", True) and request.get("source") == source
            and str(request.get("cycle", "")).startswith(cycle.strftime("%Y-%m-%dT%H"))
            and set(hours) <= set(int(lead) for lead in request.get("leads") or ())
            and (not profile or request.get("record_profile") == profile))


def _target_support(target_basis, box: str, out_dir: Path) -> tuple[Path, str] | None:
    """Seal the reader's mesh and lattice; Rust derives its cell and edge stencils."""
    basis = target_basis or {}
    mesh = basis.get("from_grid")
    if not mesh and basis.get("from_plan"):
        plan = json.loads(Path(basis["from_plan"]).read_text(encoding="utf-8"))
        mesh = (plan.get("cull") or {}).get("grid")
    if not mesh:
        return None  # a box without a mesh still requires every rectangle point
    mesh = Path(mesh).resolve(strict=True)
    path = out_dir / "target-support.json"
    document = {"schema": "rw-wps.mesh-support.v1", "consumer": "mpas-case7-f32",
                "target_latlon": box, "mesh": {"path": str(mesh), "sha256": _sha256(mesh)}}
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path, _sha256(path)


def build_packaged_intermediates(
    *, source: str, grib_dir: Path, cycle, hours: tuple[int, ...], target, out_dir: Path,
    workers: int, prefix: str = "MET", target_basis: dict[str, Any] | None = None,
    log: Callable[[str], None] = print, physics_suite: str | None = None,
    soil_set: str | None = None,
) -> dict[str, Any]:
    from .hrrr_intermediate import IntermediateRefusal, INTERMEDIATE_SCHEMA

    out_dir = Path(out_dir).expanduser().absolute()
    grib_dir = Path(grib_dir).expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob(f"{prefix}:*")):
        raise IntermediateRefusal(f"{out_dir} holds intermediate hours; pass a fresh --out-dir so the receipt "
                                  "cannot describe an earlier run")
    if not hours or hours != tuple(range(hours[0], hours[-1] + 1)) or hours[0] != 0:
        raise IntermediateRefusal("a packaged source writes the cycle's hours from 0, contiguous; "
                                  f"{list(hours)} would leave a boundary interval with no file")
    from .init_sources import admit_cycle
    from .hrrr_intermediate import soil_set_for_suite
    row = source_table_row(source)
    source = row["engine_source_id"]
    admit_cycle(row, cycle)
    spec = row.get("intermediate") or {}
    profile = spec.get("profile")
    soil_set = soil_set or soil_set_for_suite(row, physics_suite)
    stages: dict[str, float] = {}
    started = time.perf_counter()
    if not _fetch_complete(grib_dir, source, cycle, hours, profile):
        grib_dir.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, "-m", "woof.cli", "fetch", "--source", source,
                "--cycle", cycle.strftime("%Y-%m-%dT%H"), "--hours", str(hours[-1]), "--out", str(grib_dir)]
        if profile:
            argv += ["--record-profile", profile]
        log(f"FETCH {' '.join(argv[3:])}")
        with (out_dir / "fetch.log").open("w", encoding="utf-8") as handle:
            if subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT).returncode:
                raise IntermediateRefusal(f"woof fetch --source {source} failed; see {out_dir / 'fetch.log'}")
    stages["fetch_s"] = time.perf_counter() - started
    box = f"{target.south!r},{target.west!r},{target.dlat!r},{target.dlon!r},{target.nx},{target.ny}"
    work = out_dir / "work"
    argv = [sys.executable, "-m", "woof.cli", "source-intermediate", "--source", source,
            "--fetch-dir", str(grib_dir), "--target-latlon", box, "--out-dir", str(out_dir),
            "--prefix", prefix, "--work-dir", str(work), "--workers", str(max(1, int(workers)))]
    support = _target_support(target_basis, box, out_dir)
    if support is not None:
        path, digest = support
        argv += ["--target-support", str(path), "--target-support-sha256", digest]
    argv += ["--soil-set", soil_set]
    if profile:
        argv += ["--profile", profile]
    for hour in hours:
        argv += ["--valid-time", f"{cycle.replace(minute=0, second=0) + _hours(hour):%Y-%m-%dT%H:%M:%S}"]
    log(f"COMPOSE {source} over the {target.nx} x {target.ny} box, {len(hours)} hours")
    middle = time.perf_counter()
    completed = subprocess.run(argv, capture_output=True, text=True)
    (out_dir / "source-intermediate.log").write_text(completed.stderr + completed.stdout, encoding="utf-8")
    if completed.returncode:
        prefix_text = "woof source-intermediate: "
        plain = next((line[len(prefix_text):] for line in reversed(completed.stderr.splitlines())
                      if line.startswith(prefix_text)), None)
        raise IntermediateRefusal(plain or f"Could not prepare {source}; see {out_dir / 'source-intermediate.log'}")
    stages["compose_export_s"] = time.perf_counter() - middle
    written = json.loads(completed.stdout.splitlines()[-1])["written"]
    if support is not None:
        document = json.loads(support[0].read_text(encoding="utf-8"))
        mesh = document["mesh"]
        if _sha256(support[0]) != support[1] or _sha256(Path(mesh["path"])) != mesh["sha256"]:
            raise IntermediateRefusal("The mesh support changed while preparing the source; run the region again")
        for item in written:
            export = json.loads(Path(item["receipt"]).read_text(encoding="utf-8"))
            authenticated = export.get("target_support") or {}
            if (authenticated.get("contract_sha256") != support[1]
                    or authenticated.get("mesh_sha256") != mesh["sha256"]):
                raise IntermediateRefusal("The source export did not authenticate this mesh's support")
    # The composed frameset is spent once every hour is written (it holds the
    # whole source grid's surface planes); the receipts name what it was.
    shutil.rmtree(work / "composed", ignore_errors=True)
    files = []
    for item in written:
        path = Path(item["output"]["path"])
        files.append({"path": str(path), "valid_time": item["valid_time"], "bytes": path.stat().st_size,
                      "sha256": _sha256(path), "export_receipt": item["receipt"]})
    levels = {item["nfglevels"] for item in written}
    soil = {item["nfgsoillevels"] for item in written}
    if len(levels) != 1 or len(soil) != 1:
        raise IntermediateRefusal(f"{source}'s hours disagree on their first-guess levels ({levels}, {soil})")
    receipt = {
        "schema": INTERMEDIATE_SCHEMA,
        "road": PACKAGED_SCHEMA,
        "source": source,
        "cycle": f"{cycle:%Y-%m-%d_%H:%M:%S}",
        "hours": list(hours),
        "grib_dir": str(grib_dir),
        "physics_suite": physics_suite, "soil_set": soil_set, "profile": profile,
        "target": target.as_dict(),
        "target_basis": dict(target_basis or {}),
        "stages": {**stages, "parallelism": f"mapped engine, {max(1, int(workers))} workers"},
        "files": files,
        "levels": {"count": next(iter(levels)), "init_switches": {"--nfglevels": next(iter(levels), row["first_guess"]["nfglevels"]), "--nfgsoillevels": soil.pop()}},
        "argv": argv,
    }
    path = out_dir / "intermediate-receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8", newline="\n")
    log(f"RECEIPT {path}")
    return receipt


def _hours(hour: int):
    from datetime import timedelta
    return timedelta(hours=int(hour))


__all__ = ["PACKAGED_SCHEMA", "build_packaged_intermediates", "handles"]
