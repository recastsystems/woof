"""Vortex relocation and intensity correction of a mapped source's initial frame (TC-VI1).

Breakage it prevents (the storm 2024-10-08, box M, 2026-10-27): WOOF's vortex initialization existed only on the native
GFS route (source_cli refused --vortex-init for every other source), so the 36 GEFS members of the moving-nest
data-assimilation cycle started from their own broad, weak vortices against a 909 hPa, 77 m/s, 9 km-RMW storm. The first analysis
then had a first guess so far from the recon and TDR core winds that its gross check rejected them, and the forecast
ran 40-47 kt weak and 35-56 hPa high.

On any mapped preparation (every packaged profile, GEFS members included) ``--vortex-init relocate --vortex-vitals
VITALS.json`` diagnoses THIS member's own vortex on its canonical initial frame (centre: the sea-level pressure minimum
within 300 km of the vitals position; RMW: the maximum of its azimuthal-mean 10 m wind) and moves and corrects it with
woof.tc.frame_transform.relocate_frame toward the vitals: centre, maximum wind and RMW (the published 15 % per-cycle
size bound), with the vitals minimum pressure bounding the common wind gain. Each member keeps its own vortex
structure, asymmetries and environment; outside the 500 km disks the frame is copied bit for bit; later forcing frames
are unchanged. This is the HWRF/HAFS order: relocation and intensity correction before the first analysis.

Storm size (TC-VI2). Breakage it prevents (the storm hex 72 h from the 2024-10-08 00Z hex data-assimilation analysis, 2026-10-27):
the common wind gain (capped at 2) multiplies the member's whole axisymmetric vortex out to the 500 km disk while the
size stays inside the per-cycle bound, so a broad 998 hPa GEFS 0.5 deg vortex (10 m azimuthal maximum 19 m/s at 60 km)
became a 978 hPa vortex whose 34 kt wind reached about 400 km at 1.5 km height against the vitals R34 of 130 km. The
analysis kept that outer vortex (R34 395 km, GFS analysis 185 km), and the forecast storm moved at 7-9 kt inside
11-12 kt of its own deep-layer steering from 24 h on, landing 18 h late. ``size_fit="vitals_r34"`` (the default when
the vitals carry R34) relocates and gains the member once at its own size, measures the 34 kt radius of the result's
azimuthal-mean 10 m wind, and relocates it again scaled radially by one factor, RMW and outer disk alike (a linear
radius map cannot fold), so that radius lands on the vitals' mean R34. The
per-cycle bound is for a model's own cycled vortex (DOI 10.3390/atmos11090968, section 3.2); a cold start from
another model's vortex is fitted once, within [SIZE_FIT_MIN, SIZE_FIT_MAX].

Equations: woof/tc/README.md (Kurihara et al. 1993; DOI 10.3390/atmos11090968). Apache-2.0.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

SCHEMA = "gpuwm-prepared-vortex-init-v1"
SEARCH_RADIUS_M = 300000.0
SEPARATION_RADIUS_M = 500000.0
PROFILE_RADII_M = (5000.0, 400000.0, 5000.0)   # first, last, step of the azimuthal-mean wind profile
AZIMUTHS = 72
EARTH_RADIUS_M = 6371000.0
R34_M_S = 34.0 * 1852.0 / 3600.0
SIZE_FIT_MIN, SIZE_FIT_MAX = 0.35, 1.5   # one-time cold-start radius factor bounds (TC-VI2)
SIZE_FIT_TOLERANCE = 0.10                # 34 kt radius already within 10 % of the vitals: no size change


def add_arguments(parser, *, with_mode=True):
    if with_mode:
        parser.add_argument("--vortex-init", choices=("off", "relocate"), default="off",
                            help="TC initial vortex: relocate = relocation and intensity correction toward TC vitals")
    parser.add_argument("--vortex-vitals", type=Path, metavar="VITALS.json",
                        help="tc-vitals/1 document (rw_tc_vitals) holding the record at the initial time")
    parser.add_argument("--vortex-size-change-limit", type=float, default=None, metavar="FRACTION",
                        help="largest RMW change of one relocation (default 0.15, the published per-cycle bound)")


def declaration(args):
    mode = getattr(args, "vortex_init", "off") or "off"
    vitals = getattr(args, "vortex_vitals", None)
    if mode == "off":
        if vitals is not None:
            raise ValueError("--vortex-vitals needs --vortex-init relocate")
        return None
    if mode != "relocate":
        raise ValueError("mapped preparations take --vortex-init relocate (vitals-da is the private data-assimilation cycle)")
    if vitals is None:
        raise ValueError("--vortex-init relocate on a mapped source needs --vortex-vitals VITALS.json")
    path = Path(vitals).resolve(strict=True)
    doc = {"mode": "relocate", "vitals": str(path),
           "vitals_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    limit = getattr(args, "vortex_size_change_limit", None)
    if limit is not None:
        if not (math.isfinite(limit) and 0.0 <= limit < 1.0):
            raise ValueError("--vortex-size-change-limit must be in [0, 1)")
        doc["size_change_limit"] = float(limit)
    return doc


def command_arguments(args):
    doc = declaration(args)
    if doc is None:
        return []
    return ["--vortex-init", "relocate", "--vortex-vitals", doc["vitals"],
            *(["--vortex-size-change-limit", repr(doc["size_change_limit"])] if "size_change_limit" in doc else [])]


def normalize_arguments(tokens, base):
    import argparse
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    add_arguments(parser)
    args, extra = parser.parse_known_args(tokens)
    if extra or args.vortex_init != "relocate":
        raise ValueError("vortex run options require a complete relocate declaration")
    if args.vortex_vitals is not None and not args.vortex_vitals.is_absolute():
        args.vortex_vitals = Path(base) / args.vortex_vitals
    return command_arguments(args)


def vitals_record(path, valid_time):
    """The one tc-vitals/1 record valid at the initial time (exact; no interpolation of intensity targets)."""
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != "tc-vitals/1":
        raise ValueError(f"{path}: vortex vitals must be a tc-vitals/1 document (rw_tc_vitals)")
    want = valid_time.removesuffix("Z").replace(" ", "T")
    rows = [r.get("vitals", r) for r in doc.get("records", [])
            if str(r.get("vitals", r).get("valid_time", "")).removesuffix("Z") == want]
    if len(rows) != 1:
        raise ValueError(f"{path}: needs exactly one vitals record at {want}, found {len(rows)}")
    v = rows[0]
    need = ("latitude_deg", "longitude_deg", "minimum_pressure_pa", "maximum_wind_m_s", "radius_maximum_wind_m")
    if any(v.get(k) is None or not math.isfinite(float(v[k])) for k in need):
        raise ValueError(f"{path}: the vitals record at {want} lacks one of {need}")
    return v


def library_name():
    """The platform's spelling of the TC frames library (the bundle's own)."""
    return ("tc_frames_io.dll" if os.name == "nt" else
            "libtc_frames_io.dylib" if sys.platform == "darwin" else
            "libtc_frames_io.so")


def library_path():
    from woof import bridges
    found = bridges.find_artifact("WOOF_TC_FRAMES_LIB", library_name(),
                                  crate_relative="tools/tc_frames_io", executable=False)
    if found is None:
        raise FileNotFoundError("install this release's TC bridge library or build tools/tc_frames_io")
    return found


def _distance(lat, lon, centre, xp):
    la, lo = xp.radians(lat), xp.radians(lon)
    lc, oc = math.radians(centre[0]), math.radians(centre[1])
    h = xp.sin((la - lc) / 2) ** 2 + math.cos(lc) * xp.cos(la) * xp.sin((lo - oc) / 2) ** 2
    return 2.0 * EARTH_RADIUS_M * xp.arcsin(xp.sqrt(xp.clip(h, 0.0, 1.0)))


def diagnose(*, library, input_directory, valid_time, record):
    """This member's own vortex on its canonical frame: centre, RMW, boundary-layer pressure (all on the card)."""
    import cupy as cp
    from . import frames
    from .frame_transform import MSLP, _axis, derived_mslp
    doc = json.loads((Path(input_directory) / "frames.json").read_bytes())
    chosen = [f for f in doc["frames"] if f["valid_time"] == valid_time]
    if len(chosen) != 1:
        raise ValueError("vortex diagnosis needs exactly one initial frame")
    have = {f["name"] for f in chosen[0]["fields"]}
    names = ("surface_pressure", "terrain_height", "air_temperature_2m", "specific_humidity_2m",
             "eastward_wind_10m", "northward_wind_10m") + ((MSLP,) if MSLP in have else ())
    arrays, meta, digest = frames.read(library=library, input_directory=input_directory,
                                       valid_time=valid_time, fields=names)
    mslp = arrays[MSLP] if MSLP in arrays else derived_mslp(
        arrays["surface_pressure"], arrays["terrain_height"], arrays["air_temperature_2m"],
        arrays["specific_humidity_2m"])
    lat1 = cp.asarray(_axis(meta, "latitude"), cp.float64)
    lon1 = cp.asarray(_axis(meta, "longitude"), cp.float64)
    lat, lon = cp.meshgrid(lat1, lon1, indexing="ij")
    seed = (float(record["latitude_deg"]), float(record["longitude_deg"]))
    near = _distance(lat, lon, seed, cp) <= SEARCH_RADIUS_M
    if not bool(near.any()):
        raise ValueError("the vitals position is not inside this source frame")
    masked = cp.where(near, mslp, cp.inf)
    k = int(cp.argmin(masked))
    j, i = divmod(k, mslp.shape[1])
    ny, nx = mslp.shape

    def vertex(a, b, c):  # parabola vertex offset in grid units, clamped to half a cell
        d = a - 2 * b + c
        return 0.0 if d <= 0 else max(-.5, min(.5, .5 * (a - c) / d))
    m = cp.asnumpy(mslp[max(j - 1, 0):j + 2, max(i - 1, 0):i + 2])
    dj = vertex(*m[:, min(i, 1)]) if m.shape[0] == 3 else 0.0
    di = vertex(*m[min(j, 1), :]) if m.shape[1] == 3 else 0.0
    dlat = float(lat1[1] - lat1[0])
    dlon = float(lon1[1] - lon1[0])
    centre = (float(lat1[j]) + dj * dlat, math.remainder(float(lon1[i]) + di * dlon, 360.0))
    # environment: the mean sea-level pressure on the 300-500 km annulus; a member with no closed low refuses
    dist = _distance(lat, lon, centre, cp)
    ring = (dist > SEARCH_RADIUS_M) & (dist <= SEPARATION_RADIUS_M)
    deficit = float(cp.mean(mslp[ring]) - mslp[j, i])
    if not deficit > 200.0:
        raise ValueError(f"no member vortex near the vitals position (central deficit {deficit:.0f} Pa against the "
                         "300-500 km annulus); relocation needs a vortex to move")
    # RMW: azimuthal mean of the 10 m wind speed on rings, bilinear on the regular lattice
    speed = cp.hypot(arrays["eastward_wind_10m"], arrays["northward_wind_10m"])
    r0, r1, dr = PROFILE_RADII_M
    radii = cp.arange(r0, r1 + .5 * dr, dr, dtype=cp.float64)
    theta = cp.arange(AZIMUTHS, dtype=cp.float64) * (2 * math.pi / AZIMUTHS)
    ang = radii[:, None] / EARTH_RADIUS_M
    la0, lo0 = math.radians(centre[0]), math.radians(centre[1])
    plat = cp.arcsin(math.sin(la0) * cp.cos(ang) + math.cos(la0) * cp.sin(ang) * cp.cos(theta[None, :]))
    plon = lo0 + cp.arctan2(cp.sin(theta[None, :]) * cp.sin(ang) * math.cos(la0),
                            cp.cos(ang) - math.sin(la0) * cp.sin(plat))
    fy = (cp.degrees(plat) - float(lat1[0])) / dlat
    fx = cp.remainder(cp.degrees(plon) - float(lon1[0]), 360.0) / dlon
    y0 = cp.clip(cp.floor(fy).astype(cp.int64), 0, ny - 2)
    x0 = cp.floor(fx).astype(cp.int64) % nx
    x1 = (x0 + 1) % nx
    wy = cp.clip(fy - y0, 0, 1)
    wx = fx - cp.floor(fx)
    sample = ((1 - wy) * ((1 - wx) * speed[y0, x0] + wx * speed[y0, x1]) +
              wy * ((1 - wx) * speed[y0 + 1, x0] + wx * speed[y0 + 1, x1]))
    profile = sample.mean(axis=1)
    rmw = float(radii[int(cp.argmax(profile))])
    psfc = float(arrays["surface_pressure"][j, i])
    return {"centre_deg": list(centre), "rmw_m": rmw, "max_azimuthal_mean_10m_wind_m_s": float(profile.max()),
            "profile_10m": {"radii_m": [float(x) for x in cp.asnumpy(radii)],
                            "speed_m_s": [float(x) for x in cp.asnumpy(profile)]},
            "minimum_sea_level_pressure_pa": float(mslp[j, i]), "central_deficit_pa": deficit,
            "boundary_layer_pressure_pa": 5000.0 + .85 * (psfc - 5000.0),
            "sea_level_pressure": "source" if MSLP in have else "derived (frame_transform.derived_mslp)",
            "frame_manifest_sha256": digest,
            "method": "centre: sea-level pressure minimum within 300 km of the vitals position, parabolic sub-cell; "
                      "RMW: maximum of the 72-azimuth mean 10 m wind on 5 km rings to 400 km; boundary layer: "
                      "sigma 0.85 of the centre surface pressure above a 5000 Pa top"}


def vitals_r34_m(record):
    """mean of the vitals' positive R34 quadrant radii (m), or None when the record carries none"""
    values = [float(v) for v in (record.get("r34_m") or []) if v is not None and math.isfinite(float(v)) and float(v) > 0]
    return sum(values) / len(values) if values else None


def r34_of(member):
    """outermost 34 kt crossing of the diagnosed azimuthal-mean 10 m wind beyond its maximum (m), linear between rings;
    the profile's last radius when the wind is still at or above 34 kt there, 0 when the maximum is below 34 kt"""
    radii, speed = member["profile_10m"]["radii_m"], member["profile_10m"]["speed_m_s"]
    peak = max(range(len(speed)), key=speed.__getitem__)
    if speed[peak] < R34_M_S:
        return 0.0
    out = radii[-1]
    for k in range(peak, len(speed) - 1):
        if speed[k] >= R34_M_S > speed[k + 1]:
            out = radii[k] + (speed[k] - R34_M_S) / (speed[k] - speed[k + 1]) * (radii[k + 1] - radii[k])
    return out


def fit_size(probe, record):
    """TC-VI2: the one radial factor that puts the relocated member's 34 kt radius on the vitals mean R34.

    ``probe`` is the diagnosis of the member relocated and gained at its own size (the transform's gain multiplies the
    axisymmetric vortex only, so the 34 kt radius is measured on that output, environment included, not predicted).
    The radius map is linear, so the vortex's 34 kt radius scales with the factor. Returns None when the vitals carry
    no R34 or the relocated vortex has no 34 kt wind (nothing to fit)."""
    target = vitals_r34_m(record)
    source = r34_of(probe)
    if target is None or not source > 0:
        return None
    factor = min(max(target / source, SIZE_FIT_MIN), SIZE_FIT_MAX)
    return {"mode": "vitals_r34", "vitals_r34_m": target, "relocated_r34_m": source,
            "relocated_r34_beyond_profile": source >= PROFILE_RADII_M[1], "requested_factor": target / source,
            "factor": factor, "bounds": [SIZE_FIT_MIN, SIZE_FIT_MAX],
            "measure": "azimuthal-mean 10 m wind speed, 72 azimuths on 5 km rings to 400 km"}


def relocate(*, library, input_directory, output_directory, valid_time, vitals_path, size_change_limit=.15,
             maximum_wind_gain=None, size_fit="vitals_r34"):
    from .frame_transform import relocate_frame, REQUIRED
    if size_fit not in (None, "vitals_r34"):
        raise ValueError("size_fit is None or 'vitals_r34'")
    record = vitals_record(vitals_path, valid_time)
    member = diagnose(library=library, input_directory=input_directory, valid_time=valid_time, record=record)
    target_rmw = float(record["radius_maximum_wind_m"])
    source_radius = target_radius = SEPARATION_RADIUS_M
    fit = None
    if size_fit and vitals_r34_m(record) is not None:
        # probe: this member relocated and gained at its own size (the transform without TC-VI2), then its 34 kt radius
        import shutil
        probe_dir = Path(output_directory).with_name(Path(output_directory).name + ".size-probe")
        if probe_dir.exists():
            raise ValueError(f"{probe_dir} exists")
        try:
            relocate(library=library, input_directory=input_directory, output_directory=probe_dir,
                     valid_time=valid_time, vitals_path=vitals_path, size_change_limit=size_change_limit,
                     maximum_wind_gain=maximum_wind_gain, size_fit=None)
            probe = diagnose(library=library, input_directory=probe_dir, valid_time=valid_time, record=record)
        finally:
            shutil.rmtree(probe_dir, ignore_errors=True)
        fit = fit_size(probe, record)
        if fit is not None and abs(1.0 - fit["factor"]) <= SIZE_FIT_TOLERANCE:
            fit = dict(fit, applied=False, reason=f"34 kt radius within {SIZE_FIT_TOLERANCE:.0%} of the vitals")
        elif fit is not None:
            # one factor on RMW and the outer disk: a linear radius map (vortex.radial_map b = 0) never folds; the
            # core is then taken toward the vitals RMW on the target grid (hex: TC-VP1 on the mesh), not on this
            # source lattice
            fit = dict(fit, applied=True)
            target_rmw = fit["factor"] * member["rmw_m"]
            target_radius = fit["factor"] * SEPARATION_RADIUS_M
            size_change_limit = max(size_change_limit, abs(1.0 - fit["factor"]) + 1e-9)
    doc = json.loads((Path(input_directory) / "frames.json").read_bytes())
    descriptor = next(f for f in doc["frames"] if f["valid_time"] == valid_time)
    carried = sorted({f["name"] for f in descriptor["fields"] if f["axes"] == ["vertical", "y", "x"]}
                     - set(REQUIRED) - {"relative_humidity", "air_pressure"})
    provenance = {"vitals_sha256": hashlib.sha256(Path(vitals_path).read_bytes()).hexdigest(),
                  "storm_id": record.get("atcf_id"), "vitals_valid_time": record["valid_time"],
                  "member_diagnosis": member,
                  "radius_policy": ("500 km separation disk; the vortex scaled radially by one factor so the gained "
                                    "34 kt radius meets the vitals mean R34 (TC-VI2)" if fit and fit.get("applied") else
                                    "500 km separation disks; R34 diagnosed, not fitted"),
                  "size_fit": fit,
                  "intensity_policy": "vitals maximum wind, gain bounded by the vitals minimum pressure"}
    manifest, receipt = relocate_frame(
        library=library, input_directory=input_directory, output_directory=output_directory,
        valid_time=valid_time, old_centre=tuple(member["centre_deg"]),
        new_centre=(float(record["latitude_deg"]), float(record["longitude_deg"])),
        source_rmw_m=member["rmw_m"], target_rmw_m=target_rmw,
        source_radius_m=source_radius, target_radius_m=target_radius,
        maximum_wind_m_s=float(record["maximum_wind_m_s"]),
        boundary_layer_pressure_pa=member["boundary_layer_pressure_pa"],
        minimum_pressure_pa=float(record["minimum_pressure_pa"]), carried_fields=tuple(carried),
        provenance=provenance, relative_humidity_convention="gfs_mixed", size_change_limit=size_change_limit,
        maximum_wind_gain=maximum_wind_gain)
    receipt["size_fit"] = fit
    return manifest, receipt, member, record


def apply_to_bundle(bundle, doc, *, valid_time, scratch):
    """Transform the initial frame once, before every root/child initialization interpolates (as storm merge)."""
    import tempfile
    from woof.mapped_engine_bridge import FrameSet
    from woof.storm_merge import _MergedBundle
    if not isinstance(bundle.frames, FrameSet):
        raise ValueError("vortex initialization requires the native canonical frame writer")
    if hashlib.sha256(Path(doc["vitals"]).read_bytes()).hexdigest() != doc["vitals_sha256"]:
        raise ValueError("vortex vitals changed after the run was declared")
    work = tempfile.TemporaryDirectory(prefix="gpuwm-vortex-init-", dir=scratch)
    try:
        root = Path(work.name)
        library = library_path()
        manifest, receipt, member, record = relocate(
            library=library, input_directory=bundle.frames.directory, output_directory=root / "relocated",
            valid_time=valid_time, vitals_path=doc["vitals"], size_change_limit=doc.get("size_change_limit", .15))
        frames = FrameSet(root / "relocated")
        if frames.valid_times != bundle.frames.valid_times:
            raise ValueError("vortex initialization changed the forcing clock")
        evidence = {
            "declaration.json": (json.dumps(doc, sort_keys=True) + "\n").encode(),
            "vitals.json": Path(doc["vitals"]).read_bytes(),
            "member-diagnosis.json": (json.dumps(member, sort_keys=True, allow_nan=False) + "\n").encode(),
            "tc-transform.json": (root / "relocated/tc-transform.json").read_bytes(),
            "background-frames.json": (bundle.frames.directory / "frames.json").read_bytes(),
            "relocated-frames.json": manifest.read_bytes(),
            "library.sha256": (hashlib.sha256(library.read_bytes()).hexdigest() + "\n").encode()}
        binding = {"schema": SCHEMA, "valid_time": valid_time, "mode": "relocate",
                   "evidence_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in evidence.items()}}
        applied = receipt.get("declaration", {})
        print(f"vortex-init: member centre {[round(x, 2) for x in member['centre_deg']]} "
              f"{member['minimum_sea_level_pressure_pa'] / 100:.1f} hPa RMW {member['rmw_m'] / 1000:.0f} km -> vitals "
              f"{record['latitude_deg']},{record['longitude_deg']} {record['minimum_pressure_pa'] / 100:.0f} hPa "
              f"{record['maximum_wind_m_s']:.0f} m/s RMW {record['radius_maximum_wind_m'] / 1000:.0f} km "
              f"(applied RMW {applied.get('target_rmw_m', float('nan')) / 1000:.0f} km)", flush=True)
        return _MergedBundle(bundle, frames, work), binding, evidence
    except BaseException:
        work.cleanup()
        raise


def write_evidence(evidence_root, evidence):
    root = Path(evidence_root) / "vortex-init"
    root.mkdir()
    for name, data in evidence.items():
        (root / name).write_bytes(data)


def validate_evidence(prepared_root, binding, *, valid_time):
    if (not isinstance(binding, dict) or set(binding) != {"schema", "valid_time", "mode", "evidence_sha256"}
            or binding["schema"] != SCHEMA or binding["valid_time"] != valid_time or binding["mode"] != "relocate"):
        raise ValueError("invalid prepared vortex initialization binding or clock")
    root = Path(prepared_root) / "source-evidence/vortex-init"
    need = {"declaration.json", "vitals.json", "member-diagnosis.json", "tc-transform.json",
            "background-frames.json", "relocated-frames.json", "library.sha256"}
    if set(binding["evidence_sha256"]) != need:
        raise ValueError("prepared vortex initialization evidence inventory differs")
    payloads = {}
    for name, digest in binding["evidence_sha256"].items():
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"vortex initialization evidence changed: {name}")
        payloads[name] = data
    transform = json.loads(payloads["tc-transform.json"])
    if (transform["declaration"]["valid_time"] != valid_time
            or transform["output_manifest_sha256"] != binding["evidence_sha256"]["relocated-frames.json"]
            or transform["declaration"]["background_manifest_sha256"]
            != binding["evidence_sha256"]["background-frames.json"]):
        raise ValueError("vortex initialization evidence chain differs")
