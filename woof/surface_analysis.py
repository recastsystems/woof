"""Prescribed SST and ice: source selection and CUDA driver orchestration.

Run ``rw_netcdf surface SPEC.json OUTPUT_DIR`` before the forecast. Sources,
packing, coordinates and bounded remapping are owned by that Rust command.
The resulting manifest is the only runtime input. No Python decode or regrid.
"""
from __future__ import annotations
from bisect import bisect_right
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

SOURCE_TABLE = Path(__file__).with_name("data") / "surface_analyses.json"


def _instant(value):
    value = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def selected_records(records, instant):
    """One latest non-future record per target and geographic ownership mask."""
    groups = {}
    for row in records:
        groups.setdefault((row["target"], row["mask"]), []).append(row)
    selected = []
    for key, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: _instant(r["valid_time"]))
        times = [_instant(row["valid_time"]) for row in rows]
        if len(times) != len(set(times)):
            raise ValueError(f"duplicate surface analysis time for {key}; source precedence is ambiguous")
        index = bisect_right(times, instant) - 1
        if index < 0:
            raise ValueError(f"surface analysis {key} starts after the model time; using it would read future data")
        row = rows[index]
        interval = row["interval_seconds"]
        if type(interval) is not int or interval <= 0:
            raise ValueError("surface analysis interval must be a positive integer")
        if instant - times[index] >= interval:
            raise ValueError(f"surface analysis {key} expired at its next scheduled update; holding it would use stale SST or ice")
        selected.append(row)
    return selected


def _read_plane(root, row, shape):
    path = (root / row["filename"]).resolve()
    if path.parent != root.resolve():
        raise ValueError("surface plane escapes its manifest directory")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError(f"surface analysis plane changed: {path.name}")
    if len(data) != int(np.prod(shape)) * 4:
        raise ValueError(f"surface analysis plane has wrong byte count: {path.name}")
    # Transport of Rust's decoded f32 plane, without a numerical transform.
    return np.frombuffer(data, dtype="<f4").reshape(shape)


class SurfaceAnalysis:
    def __init__(self, manifest, *, shape, start_time, cfg, latitude=None, longitude=None):
        if start_time is None:
            raise ValueError("SST cycling needs an absolute model start time to select the correct analysis")
        self.path = Path(manifest).resolve()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.document = json.loads(self.path.read_text(encoding="utf-8"))
        if self.document.get("schema") != "gpuwm-surface-analysis-v1" or self.document.get("time_policy") != "hold-latest-no-lookahead":
            raise ValueError("SST cycling needs a Rust surface-analysis v1 manifest")
        if list(shape) != self.document.get("shape"):
            raise ValueError("surface analysis grid shape differs from the model; remap the analysis with Rust before launch")
        if latitude is None or longitude is None:
            raise ValueError("SST cycling needs model latitude and longitude to prevent applying an analysis from another grid")
        def host(value):
            if hasattr(value, "get"):
                value = value.get()
            return np.asarray(value, dtype="<f4").tobytes()
        coordinates = hashlib.sha256(host(latitude) + host(longitude)).hexdigest()
        if coordinates != self.document.get("coordinates_sha256"):
            raise ValueError("surface analysis coordinates differ from the model; a same-shaped grid is not the same location")
        self.shape = tuple(shape)
        self.start = _instant(start_time)
        self.cfg = cfg
        self.records = self.document["records"]
        if not self.records or not any(r["target"] == "SST" for r in self.records):
            raise ValueError("sst_update=1 needs at least one SST record; an ice-only stream cannot update water temperature")
        for row in self.records:
            if row["target"] not in ("SST", "SEAICE", "VEGFRA", "ALBBCK") or row["mask"] not in ("water", "ocean", "lake", "all"):
                raise ValueError("surface analysis has an unknown target or ownership mask")
        selected_records(self.records, self.start)
        self.water = _read_plane(self.path.parent, self.document["water"], self.shape)
        self.lake = _read_plane(self.path.parent, self.document["lake"], self.shape)
        if not np.isin(self.water, (0, 1)).all() or not np.isin(self.lake, (0, 1)).all():
            raise ValueError("surface ownership masks must be binary")
        import cupy as cp
        categories = self.document.get("surface_categories", {})
        if any(r["target"] == "SEAICE" for r in self.records) and not all(
                type(categories.get(k)) in (int, float) and categories[k] > 0
                and int(categories[k]) == categories[k] for k in ("water", "ice")):
            raise ValueError("sea-ice updates require the grid's ISWATER and ISICE category identities")
        self.categories = categories
        self.held = {}
        self.events = []
        self.device = (cp.asarray(self.water), cp.asarray(self.lake))
        self.uploaded = cp.empty(self.shape, dtype=cp.float32)
        # Pinned transport buffers are prepared before stepping. The Rust
        # producer owns decode, units, masking and range validation. Runtime
        # copies and column launches share a CUDA stream, without host reads.
        self.planes = {}
        for row in self.records:
            data = _read_plane(self.path.parent, row, self.shape)
            memory = cp.cuda.alloc_pinned_memory(data.nbytes)
            host = np.frombuffer(memory, dtype="<f4", count=data.size).reshape(self.shape)
            np.copyto(host, data)
            self.planes[row["filename"]] = (memory, host)
        self.next_update = self.start

    def advance(self, fields, elapsed_seconds):
        import cupy as cp
        from woof.core.kernels import get_kernel
        instant = self.start + float(elapsed_seconds)
        n = int(np.prod(self.shape))
        if instant >= self.next_update:
            selected = selected_records(self.records, instant)
            selected.sort(key=lambda row: row["target"] != "SST")
            for row in selected:
                for other in selected:
                    if other is not row and other["target"] == row["target"] and (
                            "all" in (other["mask"], row["mask"]) or
                            "water" in (other["mask"], row["mask"])):
                        raise ValueError("overlapping surface source ownership")
                key = (row["target"], row["mask"])
                if self.held.get(key) == row["filename"]:
                    continue
                self.uploaded.set(self.planes[row["filename"]][1], stream=cp.cuda.get_current_stream())
                target = {"SST": "sst", "SEAICE": "xice", "VEGFRA": "vegfra", "ALBBCK": "albbck"}[row["target"]]
                get_kernel("surface_analysis", "load_surface_field")(
                    ((n+255)//256,), (256,),
                    (self.uploaded, *self.device, fields[target],
                     np.int32({"water": 0, "ocean": 1, "lake": 2, "all": 3}[row["mask"]]), np.int32(n)))
                self.held[key] = row["filename"]
                self.events.append({"elapsed_seconds": float(elapsed_seconds),
                                    "manifest_sha256": self.sha256, **row})
            deadlines = [_instant(row["valid_time"]) + row["interval_seconds"] for row in selected]
            deadlines.extend(_instant(row["valid_time"]) for row in self.records
                             if _instant(row["valid_time"]) > instant)
            self.next_update = min(deadlines)
        names = ("sst", "lakemask", "xice", "xicem", "xice_save", "xland", "ivgtyp", "isltyp", "vegfra",
                 "tmn", "albedo", "albbck", "emiss", "embck")
        tail = ("tsk", "tsk_save", "snow", "snowc", "snowh", "tslb", "smois", "sh2o")
        get_kernel("surface_analysis", "surface_update_column")(
            ((n+255)//256,), (256,),
            (*(fields[k] for k in names), fields.get("albsi", fields["albbck"]),
             *(fields[k] for k in tail), np.int32(self.cfg.sf_lake_physics),
             np.int32(self.cfg.fractional_seaice),
             np.int32(self.categories.get("water") or 1), np.int32(self.categories.get("ice") or 1),
             np.int32(getattr(self.cfg, "seaice_albedo_opt", 0)), np.float32(self.cfg.seaice_albedo_default),
             np.int32(fields["tslb"].shape[0]), np.int32(n)))


def bind_surface_analysis(driver, cfg, start_time, latitude=None, longitude=None):
    if not cfg.sst_update:
        return
    if not cfg.surface_analysis:
        raise ValueError("sst_update=1 needs surface_analysis pointing to a Rust-prepared manifest; without a source stream the run would hold its initial SST")
    # RUC mosaic_lu/mosaic_soil mix parameter tables. They do not create
    # separate prognostic surface temperatures (core/ruc_mosaic.py).
    # The common WRF SST update therefore consumes the same grid carriers.
    driver.surface_analysis = SurfaceAnalysis(cfg.surface_analysis, shape=driver.fields["tsk"].shape,
                                              start_time=start_time, cfg=cfg,
                                              latitude=latitude, longitude=longitude)
    import cupy as cp
    analysis = driver.surface_analysis
    if cfg.sf_lake_physics:
        new_lakes = (analysis.device[1] > .5) & (driver.fields["lakemask"] < .5)
        if bool(cp.any(new_lakes).item()):
            raise ValueError("surface analysis declares lake columns without initialized lake model state")
    else:
        driver.fields["lakemask"][...] = cp.maximum(driver.fields["lakemask"], analysis.device[1])


def prepare_analysis(grid, inputs, output, *, table=SOURCE_TABLE, interval_seconds=None):
    """Prepare table rows through the existing Rust binary, returning its manifest."""
    import subprocess
    from woof.netcdf_bridge import resolve_netcdf_bin
    output = Path(output).resolve()
    binary = resolve_netcdf_bin()
    abi = subprocess.run([str(binary), "--abi"], check=True, capture_output=True, text=True).stdout
    if "surface_analysis_v4" not in abi:
        from woof.netcdf_bridge import netcdf_remedy
        raise RuntimeError("the installed Rust reader predates revision 4 surface ownership; it cannot verify grid-aligned auxiliary inputs without coordinates and could apply forcing to the wrong grid. Rebuild the reader before preparing this SST stream.\n" + netcdf_remedy())
    spec = output.with_suffix(".spec.json")
    if spec.exists():
        raise ValueError(f"surface spec already exists: {spec}")
    document = {"grid": str(Path(grid).resolve()), "table": str(Path(table).resolve()),
                "inputs": [{"source": source, "path": str(Path(path).resolve()),
                            **({"interval_seconds": int(interval_seconds)} if interval_seconds is not None else {})}
                           for source, path in inputs]}
    spec.write_text(json.dumps(document, indent=2), encoding="utf-8")
    subprocess.run([str(binary), "surface", str(spec), str(output)], check=True)
    return output / "manifest.json"


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Prepare SST, lake temperature and ice with metadata rows and Rust")
    parser.add_argument("--grid", type=Path, required=True, help="destination wrfinput or geographic grid NetCDF")
    parser.add_argument("--input", action="append", required=True, help="source-row=local-NetCDF, repeat for each source time")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--table", type=Path, default=SOURCE_TABLE)
    args = parser.parse_args(argv)
    inputs = []
    for text in args.input:
        source, separator, path = text.partition("=")
        if not separator or not source or not path:
            parser.error("--input must be source-row=local-NetCDF")
        inputs.append((source, path))
    print(prepare_analysis(args.grid, inputs, args.out, table=args.table))


if __name__ == "__main__":
    main()


def produce_wrflowinp(inputs, output, *, surface_physics=2):
    """WRF stream 4, produced by the shared rw-wps Rust implementation."""
    import subprocess
    from woof.netcdf_bridge import resolve_netcdf_bin
    binary = resolve_netcdf_bin()
    abi = subprocess.run([str(binary), "--abi"], check=True, capture_output=True, text=True).stdout
    if "wrflowinp_v1" not in abi:
        raise RuntimeError("rebuild rw-netcdf with the rw-wps stream-4 producer")
    result = subprocess.run([str(binary), "wrflowinp", "--sf-surface-physics", str(surface_physics), str(output), *map(str, inputs)],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)
