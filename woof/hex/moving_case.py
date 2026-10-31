"""A moving-region case file and the mesh-family specs derived from it.

A case is a JSON document (``woof hex moving`` reads it; nothing else does):
the start time and its initial-condition source, the storm's position and
motion at the analysis time (both known at run time), the mesh block (the
background spacing, the fine region and the rings around it, optionally a
corridor of fixed rings along the initial motion) and the tracker block.
The case file is the only place a storm, a date or a position appears: no
default, identifier or code path names one.

Every member of a family is built from this file and a centre:

* :func:`spec` -- the generator spec of one member: concentric caps about the
  centre whose spacings and radii come from the mesh block.  With a corridor,
  rings marked ``fixed`` are caps about the corridor centres instead and do
  not move with the region;
* :func:`frame_spec` -- the fixed corridor rings alone (the far field every
  stitched member shares);
* :func:`stitch` -- the radii of a stitched member
  (:func:`woof.hex.moving_zone.build_insert_member`);
* :func:`register_member` -- after grid and static exist, the member's mesh
  row and its entry in the family file the controller reads.

Moved here from the source tree's ``tools/moving48/member.py`` so the
installed ``woof hex moving`` door needs no checkout.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .errors import MpasPortError

EARTH_RADIUS_KM = 6371.229
CASE_KEYS = ("start_time", "dt_seconds", "n_levels", "vertical_spec", "storm", "mesh")


class MovingCaseError(MpasPortError):
    """A case file that cannot drive a moving run, named with what is missing."""


def load_case(path: str | Path) -> dict[str, Any]:
    """Read and check a case file; the result carries its own resolved path."""

    path = Path(path).expanduser().absolute()
    try:
        case = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise MovingCaseError(f"case file {path} does not exist") from None
    except ValueError as error:
        raise MovingCaseError(f"case file {path} is not JSON: {error}") from None
    missing = [key for key in CASE_KEYS if key not in case]
    if missing:
        raise MovingCaseError(
            f"case file {path} lacks {', '.join(missing)}; a moving case names its start time, "
            "time step, level count, vertical spec, storm block and mesh block "
            "(see docs/moving-nest.md)")
    storm = case["storm"]
    if "initial_center_deg" not in storm:
        raise MovingCaseError(f"case file {path}: storm.initial_center_deg is required")
    for key in ("background_km", "fine_km", "fine_radius_km", "fine_transition_km", "rings"):
        if key not in case["mesh"]:
            raise MovingCaseError(f"case file {path}: mesh.{key} is required")
    # The hex default physics (owner ruling 2026-10-25): a case whose physics_args name no physics runs MPAS-A's
    # own suite for its fine spacing (woof.hex.physics_default).  Written here, the one reader of case files, so the
    # roster, the move hooks and every segment's forecast door decide the same route.
    case["_requested_physics_args"] = list(case.get("physics_args") or [])
    case["physics_args"] = resolve_physics(case)
    case["_path"] = str(path)
    return case


def resolve_physics(case, extra=(), *, source=None):
    """Resolve once from caller flags, so later explicit overrides also supersede defaults."""
    from .physics_default import ensure
    requested = case.get("_requested_physics_args", case.get("physics_args", ()))
    fine = case.get("mesh", {}).get("fine_km")
    return ensure([*requested, *extra], None if fine is None else 1000.0 * float(fine),
                  context={"geometry": "storm-following",
                           "source": source or case.get("init_source") or "gfs", "start": "cold"})


def cycle_of(case: dict[str, Any]) -> str:
    """``YYYY-MM-DDTHH`` of the case's start time (the init-source door's cycle)."""

    start = str(case["start_time"])
    return f"{start[:10]}T{start[11:13]}"


def resolve_vertical_spec(case: dict[str, Any]) -> Path:
    """The case's vertical spec: absolute, beside the case, or a packaged spec of that name.

    A case written in the source tree names ``verification/vertical-specs/<name>``
    relative to the tree; from an install the same file ships as package data
    (``woof/hex/data/countries/<name>``), so the name alone is enough there.
    """

    declared = Path(str(case["vertical_spec"]))
    if declared.is_absolute():
        if declared.is_file():
            return declared
        raise MovingCaseError(f"vertical spec {declared} does not exist")
    base = Path(case.get("_path", ".")).parent
    for folder in (base, *base.parents):
        candidate = folder / declared
        if candidate.is_file():
            return candidate
    packaged = Path(__file__).resolve().with_name("data") / "countries" / declared.name
    if packaged.is_file():
        return packaged
    raise MovingCaseError(
        f"vertical spec {declared} was found neither beside the case ({base} and its parents) nor "
        f"in the package ({packaged.parent}); pass an absolute path in the case's vertical_spec")


def member_name(lat: float, lon: float) -> str:
    """A member's name from its centre alone (a centre names its geometry)."""

    lon = lon % 360.0
    return f"m{'n' if lat >= 0 else 's'}{abs(lat) * 100:05.0f}e{lon * 100:05.0f}"


def great_circle_km(a, b) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    c = math.sin(la1) * math.sin(la2) + math.cos(la1) * math.cos(la2) * math.cos(lo2 - lo1)
    return EARTH_RADIUS_KM * math.acos(max(-1.0, min(1.0, c)))


def corridor_path(case: dict[str, Any]) -> list[tuple[list[float], float]]:
    """(centre, along-path km) of the fixed rings.

    The case's explicit ``centers_deg``, or points along the storm's motion at
    the analysis time (heading and speed are run-time inputs, like the initial
    position) over the run length.
    """

    corridor = case["mesh"]["corridor"]
    if corridor.get("centers_deg"):
        return [([float(a), float(b)], 0.0) for a, b in corridor["centers_deg"]]
    lat0, lon0 = (math.radians(float(v)) for v in case["storm"]["initial_center_deg"])
    motion = case["storm"]["initial_motion"]
    heading = math.radians(float(motion["heading_deg"]))
    length = float(motion["speed_kmh"]) * float(case["hours"]) * float(corridor.get("length_factor", 1.0))
    step = float(corridor.get("step_km", 300.0))
    count = max(1, int(math.ceil(length / step)))
    out = []
    for k in range(count + 1):
        d = length * k / count / EARTH_RADIUS_KM
        lat = math.asin(math.sin(lat0) * math.cos(d) + math.cos(lat0) * math.sin(d) * math.cos(heading))
        lon = lon0 + math.atan2(math.sin(heading) * math.sin(d) * math.cos(lat0),
                                math.cos(d) - math.sin(lat0) * math.sin(lat))
        out.append(([round(math.degrees(lat), 4), round((math.degrees(lon) + 180.0) % 360.0 - 180.0, 4)],
                    length * k / count))
    return out


def stitch(case: dict[str, Any]) -> dict[str, float]:
    """Radii of a stitched member, from the mesh block."""

    mesh = case["mesh"]
    fine = float(mesh["fine_radius_km"])
    moving = [r for r in mesh["rings"] if not r.get("fixed")]
    fixed = [r for r in mesh["rings"] if r.get("fixed")]
    if not fixed:
        raise MovingCaseError("a corridor mesh needs at least one fixed ring")
    inner = fine + max([float(r["extra_radius_km"]) + float(r["transition_km"]) for r in moving]
                       or [2.0 * float(mesh["fine_transition_km"])])
    spacing = min(float(r["spacing_km"]) for r in fixed)
    outer = inner + float(mesh["corridor"].get("seam_rows", 6)) * spacing
    return {"inner_radius_km": inner, "outer_radius_km": outer, "fixed_spacing_km": spacing}


def frame_spec(case: dict[str, Any], name: str = "frame") -> dict[str, Any]:
    """The fixed rings alone: caps of each fixed ring about every corridor centre.

    A cap's radius is the ring's own radius plus the cross-track allowance,
    which grows along the path (``spread_km_per_km``): an extrapolated motion
    is less certain the further it is carried.
    """

    mesh = case["mesh"]
    fine = float(mesh["fine_radius_km"])
    cross = float(mesh["corridor"].get("cross_track_km", 450.0))
    spread = float(mesh["corridor"].get("spread_km_per_km", 0.0))
    regions = []
    for ring in sorted((r for r in mesh["rings"] if r.get("fixed")), key=lambda r: -r["spacing_km"]):
        for centre, along in corridor_path(case):
            regions.append({"shape": {"kind": "cap", "center_deg": centre,
                                      "radius_km": fine + float(ring["extra_radius_km"]) + cross + spread * along},
                            "spacing_km": float(ring["spacing_km"]),
                            "transition_km": float(ring["transition_km"])})
    return {"name": name, "background_km": float(mesh["background_km"]), "regions": regions}


def spec(case: dict[str, Any], name: str, lat: float, lon: float) -> dict[str, Any]:
    """The generator spec of one member centred at (lat, lon)."""

    mesh = case["mesh"]
    fine = float(mesh["fine_radius_km"])
    corridor = bool(mesh.get("corridor"))
    regions = frame_spec(case)["regions"] if corridor else []
    # Outer rings first, as the generator applies them in order.  With a
    # corridor, rings marked "fixed" stay on the corridor for the whole run
    # and only the others move with the region.
    for ring in sorted(mesh["rings"], key=lambda r: -r["spacing_km"]):
        if corridor and ring.get("fixed"):
            continue
        regions.append({"shape": {"kind": "cap", "center_deg": [lat, lon],
                                  "radius_km": fine + float(ring["extra_radius_km"])},
                        "spacing_km": float(ring["spacing_km"]),
                        "transition_km": float(ring["transition_km"])})
    regions.append({"shape": {"kind": "cap", "center_deg": [lat, lon], "radius_km": fine},
                    "spacing_km": float(mesh["fine_km"]),
                    "transition_km": float(mesh["fine_transition_km"])})
    return {"name": name, "background_km": float(mesh["background_km"]), "regions": regions}


def interior_km(case: dict[str, Any]) -> float:
    mesh = case["mesh"]
    return float(mesh["fine_radius_km"]) - float(mesh.get("fine_transition_km", 0.0))


def register_member(case: dict[str, Any], name: str, lat: float, lon: float, folder: Path) -> dict[str, Any]:
    """Write NAME-rows.json and NAME-case.json and add NAME to the family file.

    Runs once grid, static and init exist in ``folder``.  The family file is
    updated under a lock: the controller prepares several members at once in
    one folder, and without it two finishing members read the same family and
    the second write drops the first member's row.
    """

    from . import mesh_rows

    folder = Path(folder)
    mesh = case["mesh"]
    row = mesh_rows.describe_generated(
        name=name, grid=folder / f"{name}.grid.nc", static=folder / f"{name}.static.nc",
        spec_path=folder / f"{name}.spec.json", generator_receipt=folder / f"{name}-mesh.json",
        static_receipt=folder / f"{name}-static.json", point_deg=(lat, lon),
        fine_dx_m=float(mesh["fine_km"]) * 1000.0, core_radius_km=float(mesh["fine_radius_km"]),
        background_km=float(mesh["background_km"]), dt_seconds=float(case["dt_seconds"]),
        n_levels=int(case["n_levels"]),
        admission={"scope": "moving-region family member; geometry admissions measured by the normal bind"})
    rows_path = folder / f"{name}-rows.json"
    mesh_rows.write_rows(rows_path, [row])
    member = {"name": name, "center_deg": [lat, lon], "mesh": name,
              "grid": str(folder / f"{name}.grid.nc"), "static": str(folder / f"{name}.static.nc"),
              "init": str(folder / f"{name}.init.nc"), "mesh_rows": str(rows_path),
              "dt_seconds": float(case["dt_seconds"]), "n_cells": int(row.n_cells)}
    (folder / f"{name}-case.json").write_text(json.dumps(member, indent=2) + "\n")
    family_path = folder / "family.json"
    with _locked(folder / ".family.lock"):
        family = json.loads(family_path.read_text()) if family_path.exists() else []
        family = [m for m in family if m["name"] != name] + [member]
        temporary = family_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(family, indent=2) + "\n")
        temporary.replace(family_path)
    return member


class _locked:
    """An exclusive advisory lock on ``path`` (POSIX); a no-op where fcntl is absent."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("w")
        try:
            import fcntl
        except ImportError:  # pragma: no cover - not a POSIX host
            return self
        fcntl.flock(self.stream, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        self.stream.close()
        return False


def locked(path: Path) -> _locked:
    return _locked(path)


def best_track_from_ibtracs(source: Path, sid: str) -> dict[str, Any]:
    """One storm's best track from an IBTrACS v04 CSV (verification only, never read at run time)."""

    import csv

    def number(text):
        text = (text or "").strip()
        try:
            return float(text)
        except ValueError:
            return None

    rows, name = [], None
    with open(source, newline="") as stream:
        for record in csv.DictReader(stream):
            if record.get("SID") != sid:
                continue
            name = name or record.get("NAME")
            rows.append(dict(time=record["ISO_TIME"], lat=number(record["LAT"]), lon=number(record["LON"]),
                             nature=record.get("NATURE"), wmo_wind_kt=number(record.get("WMO_WIND")),
                             wmo_pres_hpa=number(record.get("WMO_PRES")), usa_wind_kt=number(record.get("USA_WIND")),
                             usa_pres_hpa=number(record.get("USA_PRES")), tokyo_lat=number(record.get("TOKYO_LAT")),
                             tokyo_lon=number(record.get("TOKYO_LON")), tokyo_pres_hpa=number(record.get("TOKYO_PRES"))))
    if not rows:
        raise MovingCaseError(f"no IBTrACS rows for {sid} in {source}")
    return dict(source="IBTrACS v04r01 (provisional for the latest season)", sid=sid, name=name, rows=rows)
