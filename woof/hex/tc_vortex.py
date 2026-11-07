"""TC inner-core initialization for hex, for any table source (TC-VI1, hex port).

Breakage it prevents (the storm 2024-10-08, box M, 2026-10-27): hex had no vortex step. The init door's only TC route
took a hand-sealed GFS canonical request, so GEFS members (and GFS itself, through the Vtable decode) reached the hex
mesh with their own weak, broad vortices (gep01 18Z: 988 hPa, RMW 55 km against a 909 hPa, 9 km-RMW storm), and

:func:`stage_source` replaces the plain decode of a staged source with the WOOF canonical composition of the same
files (the Rust mapped engine; the row's ``tc_composition`` table declaration: packaged profile, the extra fields
a WPS export needs, in-band supplement role, surface contract) followed, on that canonical frame, by

* ``--storm-merge ...`` (optional, the HAFS-A or other inner core merged into the member: woof/storm_merge.py), and
* ``--vortex-init relocate --vortex-vitals VITALS`` (woof/tc/mapped_vortex.py: this member's own vortex diagnosed,
  moved to the vitals centre and corrected toward the vitals wind, RMW and minimum pressure, Kurihara et al. 1993),

and then the Rust WPS v5 export the hex init door interpolates. Adding a source is table work (a ``tc_composition``
block on its row), never a new code path.

:func:`mesh_vortex` diagnoses the vortex the hex init actually holds ON THE MESH (minimum sea-level pressure within
300 km of the vitals centre, maximum 10 m wind within 150 km of that minimum) and refuses an init whose centre
landed more than ``max_centre_error_km`` from the vitals or whose central low vanished, so the interpolation from the
source frame to the cells cannot silently lose the corrected vortex.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

EARTH_RADIUS_M = 6371000.0


class TcVortexRefusal(ValueError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def composition(row: dict[str, Any]) -> dict[str, Any]:
    spec = row.get("tc_composition")
    if not spec:
        raise TcVortexRefusal(
            f"{row['id']}: no tc_composition declaration in the init-source table; TC vortex initialization needs the "
            "row's canonical composition (engine source and packaged profile) so the vortex is moved on the source "
            "frame before the mesh interpolates it")
    if not spec.get("profile"):
        raise TcVortexRefusal(f"{row['id']}: tc_composition lacks profile")
    return spec


def _merge_namespace(storm_merge: Sequence[str]):
    """The storm-merge declaration from its CLI words, through WOOF's own parser (woof/storm_merge.py)."""
    import argparse
    from woof.storm_merge import add_arguments, declaration
    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)
    return declaration(parser.parse_args(list(storm_merge)))


def _wps_contract(policy: str, work: Path) -> Path:
    """The WPS field contract for a TC export. native-gfs-skin: SKINTEMP is the initial surface temperature and no
    separate SST slab is exported (the native MPAS GFS initial dispatch; woof/tc/intermediate.py), so an ensemble
    source without an SST field (GEFS) exports; every other binding stays."""
    import woof
    contract = Path(woof.__file__).with_name("authorities") / "wps-intermediate-fields.v1.json"
    if policy == "complete":
        return contract
    if policy != "native-gfs-skin":
        raise TcVortexRefusal(f"unknown TC surface policy {policy!r} (complete or native-gfs-skin)")
    doc = json.loads(contract.read_text())
    doc["bindings"] = [row for row in doc["bindings"] if row["wps"] != "SST"]
    out = work / "wps-contract.native-gfs-skin.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    return out


def stage_source(row: dict[str, Any], inputs: Sequence[str], cycle, out_dir: Path, *, vitals: Path | None = None,
                 storm_merge: Sequence[str] = (), size_change_limit: float | None = None, workers: int = 8,
                 member: int | str | None = None, max_wind_gain: float | None = None) -> dict[str, Any]:
    """Source files -> canonical frame -> (storm merge) -> (vitals relocation) -> WPS intermediate for hex init.

    Every array stays in Rust (composition, WPS export) or on the card (the transform). The row's tc_composition
    names the packaged profile whose mapping and composition it extends: ``mapping_fields`` adds the fields a WPS
    export needs that the profile does not decode (GEFS: snow water equivalent and sea-level pressure), and
    ``surface_policy`` selects the export contract. Returns the decode record of :func:`woof.hex.init_sources.decode`
    plus ``first_guess`` (the export's level counts) and ``vortex_initialization``."""
    from woof.ingest.preparation_workers import worker_environment
    from woof.mapped_authoring import author_input_manifest
    from woof.mapped_engine_bridge import require_engine, run_engine
    from woof.source_authorities import (packaged_authorities, packaged_contributing_mappings,
                                          packaged_provenance_files)

    spec = composition(row)
    if vitals is None and not storm_merge:
        raise TcVortexRefusal("TC initialization needs --vortex-vitals and/or a storm merge; neither was given")
    out_dir = Path(out_dir).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    met = out_dir / f"MET:{cycle.when.strftime('%Y-%m-%d_%H')}"
    work = out_dir / "tc-composition"
    if work.exists():
        raise TcVortexRefusal(f"{work} exists: a TC source stage is never reused partially; use a new directory")
    work.mkdir()
    valid = cycle.when.strftime("%Y-%m-%dT%H:%M:%S")
    began = time.perf_counter()
    timings = {}
    profile = spec["profile"]
    authorities = packaged_authorities(profile)
    mapping = authorities["mapping"]
    if spec.get("mapping_fields"):
        doc = json.loads(mapping.read_text())
        clash = sorted(set(spec["mapping_fields"]) & set(doc["fields"]))
        if clash:
            raise TcVortexRefusal(f"{row['id']}: tc_composition.mapping_fields redeclares {clash} of {profile}")
        doc["fields"].update(spec["mapping_fields"])
        mapping = work / "mapping.json"
        mapping.write_text(json.dumps(doc, indent=2) + "\n")
    role = spec.get("in_band_supplement_role")
    files = [Path(p) for p in inputs]
    supplements = {role: files} if role else {}
    provenance = packaged_provenance_files(profile)
    contributors = packaged_contributing_mappings(profile)
    manifest = author_input_manifest(work / "inputs.json", mapping_path=mapping,
        composition_path=authorities["composition"], primary_files=files,
        supplement_files=supplements, provenance_files=provenance, contributing_mappings=contributors)
    engine = require_engine()
    tick = time.perf_counter()
    composed = run_engine("compose", mapping=mapping, files=files,
        composition=authorities["composition"], supplements=supplements, provenance=provenance,
        contributing_mappings=contributors, input_manifest=manifest["manifest"]["path"],
        input_manifest_sha256=manifest["manifest"]["sha256"], output=work / "composed", engine=engine,
        threads=int(workers), lead_batch=True)
    timings["compose"] = {"seconds": round(time.perf_counter() - tick, 2), "parallelism": f"Rust, {workers} threads"}
    frame_dir = work / "composed"
    merge_receipt = None
    if storm_merge:
        from woof.storm_merge import compose_donor, run_merge
        tick = time.perf_counter()
        merge = _merge_namespace(storm_merge)
        donor, donor_execution, _evidence = compose_donor(merge, work / "donor", int(workers))
        merge_receipt, _request = run_merge(frame_dir, donor, work / "merged", valid_time=valid,
                                            declaration=merge, workers=int(workers))
        merge_receipt["donor_compose"] = donor_execution["receipt"]
        frame_dir = work / "merged"
        timings["storm_merge"] = {"seconds": round(time.perf_counter() - tick, 2), "parallelism": "Rust"}
    binding = None
    if vitals is not None:
        from woof.tc.mapped_vortex import library_path, relocate
        tick = time.perf_counter()
        library = library_path()
        limit = .15 if size_change_limit is None else float(size_change_limit)
        relocated, transform, diagnosis, record = relocate(
            library=library, input_directory=frame_dir, output_directory=work / "relocated", valid_time=valid,
            vitals_path=Path(vitals).resolve(strict=True), size_change_limit=limit,
            maximum_wind_gain=None if max_wind_gain is None else float(max_wind_gain))
        timings["vortex_relocation"] = {"seconds": round(time.perf_counter() - tick, 2), "parallelism": "GPU (CuPy)"}
        binding = dict(mode="relocate", vitals=str(Path(vitals).resolve()), vitals_sha256=_sha256(Path(vitals)),
                       member_diagnosis=diagnosis, vitals_record=record, size_change_limit=limit,
                       max_wind_gain=max_wind_gain, wind_gain=transform.get("wind_gain"),
                       requested_wind_gain=transform.get("requested_wind_gain"), size_fit=transform.get("size_fit"),
                       resolved_minimum_pressure_pa=transform.get("resolved_minimum_pressure_pa"),
                       resolved_maximum_wind_m_s=transform.get("resolved_maximum_wind_m_s"),
                       applied_target_rmw_m=transform.get("declaration", {}).get("target_rmw_m"),
                       background_manifest_sha256=_sha256(frame_dir / "frames.json"),
                       relocated_manifest_sha256=_sha256(relocated), library_sha256=_sha256(library))
        print(f"tc-vortex: member centre {[round(float(x), 2) for x in diagnosis['centre_deg']]} "
              f"{diagnosis['minimum_sea_level_pressure_pa'] / 100:.1f} hPa RMW {diagnosis['rmw_m'] / 1000:.0f} km -> "
              f"vitals {record['latitude_deg']:.2f},{record['longitude_deg']:.2f} "
              f"{record['minimum_pressure_pa'] / 100:.0f} hPa {record['maximum_wind_m_s']:.0f} m/s", flush=True)
        frame_dir = work / "relocated"
    contract = _wps_contract(spec.get("surface_policy", "complete"), work)
    pending = work / "intermediate.part"
    argv = [str(engine), "wps", "--frameset", str(frame_dir), "--composition", str(authorities["composition"]),
            "--contract", str(contract), "--valid-time", valid, "--out", str(pending),
            # the native surface-wind locator needs a 0 E first column (a -180 E frame left every western-
            # hemisphere core cell without a 10 m wind in the the storm hex init)
            "--longitude-origin", "zero"]
    if member is not None:
        # an ensemble member's frames carry its GRIB member number; the export selects it
        argv += ["--member", str(member)]
    tick = time.perf_counter()
    completed = subprocess.run(argv, capture_output=True, text=True, env=worker_environment(int(workers)))
    (work / "wps.stdout.json").write_text(completed.stdout)
    (work / "wps.stderr.txt").write_text(completed.stderr)
    if completed.returncode:
        raise TcVortexRefusal(f"TC WPS export of {row['id']} refused: {completed.stderr.strip()[-2000:]}")
    native = json.loads(completed.stdout.splitlines()[-1])
    timings["wps_export"] = {"seconds": round(time.perf_counter() - tick, 2), "parallelism": f"Rust, {workers} threads"}
    pending.replace(met)
    return {"argv": argv, "met": str(met), "met_sha256": _sha256(met), "met_bytes": met.stat().st_size,
            "route": "tc-canonical-composition", "profile": profile, "mapping_sha256": _sha256(mapping),
            "contract_sha256": _sha256(contract), "surface_policy": spec.get("surface_policy", "complete"),
            "compose": composed["receipt"], "native": native,
            "first_guess": {"nfglevels": int(native["nfglevels"]), "nfgsoillevels": int(native["nfgsoillevels"]),
                            "use_spechumd": spec.get("use_spechumd", "yes")},
            "vortex_initialization": binding, "storm_merge": merge_receipt,
            "stages": timings, "wall_seconds": round(time.perf_counter() - began, 2)}


def _dist_m(lat, lon, lat0, lon0):
    import numpy as np
    la, lo = np.radians(lat), np.radians(lon)
    l0, o0 = math.radians(lat0), math.radians(lon0)
    h = np.sin((la - l0) / 2) ** 2 + math.cos(l0) * np.cos(la) * np.sin((lo - o0) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def mesh_vortex(init: Path, vitals_record: dict[str, Any], *, static: Path | None = None,
                search_km: float = 300.0, wind_km: float = 150.0) -> dict[str, Any]:
    """The vortex the hex init holds on its cells (no smoothing): centre, minimum sea-level pressure, 10 m Vmax.

    Sea-level pressure is the surface pressure reduced hypsometrically through the lowest model layer's virtual
    temperature with a 6.5 K/km lapse over the terrain (exactly the surface pressure over open water, where TC centres
    are fixed). The 10 m wind is the init's u10/v10 when present and finite (missing sentinels refuse the wind)."""
    import numpy as np
    from netCDF4 import Dataset

    with Dataset(init) as nc:
        v = nc.variables
        lat = np.degrees(np.asarray(v["latCell"][:], np.float64))
        lon = np.degrees(np.asarray(v["lonCell"][:], np.float64))
        psfc = np.asarray(v["surface_pressure"][0] if v["surface_pressure"].ndim == 2 else v["surface_pressure"][:],
                          np.float64)
        theta = np.asarray(v["theta"][0, :, 0], np.float64)
        qv = np.asarray(v["qv"][0, :, 0], np.float64) if "qv" in v else np.zeros_like(theta)
        p = None
        if "pressure_base" in v and "pressure_p" in v:
            p = np.asarray(v["pressure_base"][0, :, 0], np.float64) + np.asarray(v["pressure_p"][0, :, 0], np.float64)
        elif "pressure" in v:
            p = np.asarray(v["pressure"][0, :, 0], np.float64)
        if "ter" in v:
            ter = np.asarray(v["ter"][:], np.float64)
        elif static is not None:
            with Dataset(static) as st:
                ter = np.asarray(st.variables["ter"][:], np.float64)
        else:
            ter = np.zeros_like(psfc)
        u10 = np.asarray(v["u10"][0], np.float64) if "u10" in v else None
        v10 = np.asarray(v["v10"][0], np.float64) if "v10" in v else None
        land = np.asarray(v["landmask"][:], np.int32) if "landmask" in v else None
    temperature = theta * (p / 1.0e5) ** (287.04 / 1004.5) if p is not None else theta
    tv = temperature * (1.0 + 0.608 * qv)
    mslp = psfc * np.exp(9.80665 * ter / (287.04 * (tv + 0.5 * 0.0065 * ter)))
    lat0, lon0 = float(vitals_record["latitude_deg"]), float(vitals_record["longitude_deg"])
    near = _dist_m(lat, lon, lat0, lon0) <= search_km * 1000.0
    if land is not None:
        near &= land == 0
    if not near.any():
        raise TcVortexRefusal(f"{init}: no water cell within {search_km} km of the vitals centre")
    k = int(np.argmin(np.where(near, mslp, np.inf)))
    out = dict(centre_deg=[round(float(lat[k]), 3), round(float(lon[k]), 3)],
               minimum_sea_level_pressure_hpa=round(float(mslp[k]) / 100.0, 2),
               centre_error_km=round(float(_dist_m(lat[k:k + 1], lon[k:k + 1], lat0, lon0)[0]) / 1000.0, 1),
               vitals_pressure_hpa=round(float(vitals_record["minimum_pressure_pa"]) / 100.0, 1),
               vitals_wind_m_s=float(vitals_record["maximum_wind_m_s"]), cells_searched=int(near.sum()))
    ring = _dist_m(lat, lon, lat[k], lon[k])
    annulus = (ring > 300e3) & (ring <= 500e3)
    out["central_deficit_hpa"] = round(float(np.mean(mslp[annulus]) - mslp[k]) / 100.0, 2) if annulus.any() else None
    if u10 is not None and v10 is not None:
        speed = np.hypot(u10, v10)
        disk = ring <= wind_km * 1000.0
        good = disk & np.isfinite(speed) & (np.abs(u10) < 1e3) & (np.abs(v10) < 1e3)
        if good.sum() == disk.sum() and disk.any():
            j = int(np.argmax(np.where(disk, speed, -1.0)))
            out.update(maximum_10m_wind_m_s=round(float(speed[j]), 2),
                       radius_maximum_wind_km=round(float(ring[j]) / 1000.0, 1))
        else:
            out["maximum_10m_wind_m_s"] = None
            out["wind_note"] = f"{int(disk.sum() - good.sum())} of {int(disk.sum())} cells within {wind_km} km lack a finite 10 m wind"
    else:
        out["maximum_10m_wind_m_s"] = None
        out["wind_note"] = "init carries no u10/v10"
    return out


def check_mesh_vortex(diagnosis: dict[str, Any], *, max_centre_error_km: float = 60.0,
                      min_deficit_hpa: float = 2.0) -> None:
    if diagnosis["centre_error_km"] > max_centre_error_km:
        raise TcVortexRefusal(f"the hex init's vortex centre is {diagnosis['centre_error_km']} km from the vitals "
                              f"(limit {max_centre_error_km} km): the relocated vortex did not reach the mesh")
    deficit = diagnosis.get("central_deficit_hpa")
    if deficit is not None and deficit < min_deficit_hpa:
        raise TcVortexRefusal(f"the hex init holds no closed low at the vitals centre (deficit {deficit} hPa)")


def vitals_record(path: Path, valid_time: str) -> dict[str, Any]:
    from woof.tc.mapped_vortex import vitals_record as record
    return record(path, valid_time)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="python -m woof.hex.tc_vortex",
                                     description="diagnose the TC vortex a hex init holds on its mesh")
    parser.add_argument("init", type=Path)
    parser.add_argument("--vitals", type=Path, required=True, help="tc-vitals/1 document")
    parser.add_argument("--valid-time", required=True, help="YYYY-MM-DDTHH:MM:SS")
    parser.add_argument("--static", type=Path)
    parser.add_argument("--check", action="store_true", help="refuse a misplaced or vanished vortex")
    args = parser.parse_args(argv)
    diagnosis = mesh_vortex(args.init, vitals_record(args.vitals, args.valid_time), static=args.static)
    print(json.dumps(diagnosis))
    if args.check:
        check_mesh_vortex(diagnosis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
