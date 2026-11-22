"""Deposition series for snow light-absorbing particles (snicar_lap_opt=2).

Re-implemented from published descriptions, not bitwise to upstream Noah-MP.
Python only fetches and orchestrates; decoding and regridding are the Rust
``rw_netcdf snow-lap-deposition`` subcommand, one file per worker on all
cores. Sources are rows of ``woof/data/snow_papers/lap_sources.json``.

    python -m woof.snow_lap_deposition --source geos-fp \\
        --start 2023-04-01T00 --end 2023-04-12T00 \\
        --domain wrfinput_d01 --out lapdep/

then run with ``&noah_mp opt_alb=3, snicar_lap_opt=2,
snicar_lap_series='lapdep/receipt.json' /``. The part of the series before
the forecast start is summed into the snowpack at the first step (spin-up).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

SPECIES = ("bc_hydrophilic", "bc_hydrophobic", "oc_hydrophilic", "oc_hydrophobic",
           "dust1", "dust2", "dust3", "dust4", "dust5")
TABLE = Path(__file__).resolve().parent/"data"/"snow_papers"/"lap_sources.json"
SPECIES_TABLE = Path(__file__).resolve().parent/"data"/"snow_papers"/"lap_species.json"


def source_row(name: str) -> dict:
    rows = json.loads(TABLE.read_text())["sources"]
    if name not in rows:
        raise SystemExit(f"unknown LAP source {name!r}; table rows: {', '.join(rows)}")
    return rows[name]


def overlap_matrix(source_diameter, target_diameter):
    """Mass fraction of each source bin falling in each target bin, for mass
    log-uniform in diameter inside a source bin."""
    out = []
    for lo, hi in source_diameter:
        row = []
        for a, b in target_diameter:
            ov = math.log(min(hi, b)) - math.log(max(lo, a)) if min(hi, b) > max(lo, a) else 0.0
            row.append(ov / (math.log(hi) - math.log(lo)))
        out.append(row)
    return out


def channels(row: dict) -> list[dict]:
    """The 18 output channels (9 species dry, then 9 wet) as linear terms."""
    def terms(names, coef=1.0):
        return [{"name": n, "coef": coef} for n in names]

    def shared(total, members, share, coef=1.0):
        return [{"name": total, "coef": coef, "split_by": members, "share": share}] if total else []

    dry, wet = {}, {}
    for family, (phil, phob) in (("bc", ("bc_hydrophilic", "bc_hydrophobic")),
                                 ("oc", ("oc_hydrophilic", "oc_hydrophobic"))):
        f = row[family]
        members = [f["hydrophobic"]["wet"][0], f["hydrophilic"]["wet"][0]]
        for key, part in ((phil, "hydrophilic"), (phob, "hydrophobic")):
            dry[key] = terms(f[part]["dry"])
            wet[key] = terms(f[part]["wet"]) + shared(f.get("shared_wet"), members, f[part]["wet"][0])
    target = [s["diameter_um"] for s in json.loads(SPECIES_TABLE.read_text())["species"][4:]]
    d = row["dust"]
    source = [[2 * lo, 2 * hi] for lo, hi in d["radius_um"]]
    frac = overlap_matrix(source, target)
    members = [w[0] for w in d["wet"]]
    for t in range(len(target)):
        key = f"dust{t+1}"
        dry[key], wet[key] = [], []
        for s in range(len(source)):
            if frac[s][t] <= 0:
                continue
            dry[key] += terms(d["dry"][s], frac[s][t])
            wet[key] += terms(d["wet"][s], frac[s][t]) + shared(d.get("shared_wet"), members, d["wet"][s][0], frac[s][t])
    return ([{"key": f"{k}_dry", "terms": dry[k]} for k in SPECIES]
            + [{"key": f"{k}_wet", "terms": wet[k]} for k in SPECIES])


def file_times(row: dict, start: datetime, end: datetime) -> list[datetime]:
    """Stored-time centres whose averaging windows overlap [start, end)."""
    step = timedelta(minutes=row["times"]["step_minutes"])
    half = timedelta(seconds=row["window_seconds"] / 2)
    t = datetime(start.year, start.month, start.day) + timedelta(minutes=row["times"]["first_offset_minutes"])
    out = []
    while t - half < end:
        if t + half > start:
            out.append(t)
        t += step
    return out


def urls(row: dict, start: datetime, end: datetime) -> list[str]:
    if row.get("credentials"):
        raise SystemExit(f"this source needs {row['credentials']} credentials, which this door does "
                         "not hold; download the files yourself and pass them with --files")
    seen, out = set(), []
    for t in file_times(row, start, end):
        u = row["url"].format(Y=f"{t.year:04d}", m=f"{t.month:02d}", d=f"{t.day:02d}",
                              H=f"{t.hour:02d}", M=f"{t.minute:02d}")
        if u not in seen:
            seen.add(u); out.append(u)
    return out


def fetch(url: str, cache: Path) -> Path:
    path = cache/url.rsplit("/", 1)[1]
    if path.exists() and path.stat().st_size > 0:
        return path
    part = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    part.rename(path)
    return path


def target_columns(domain: Path, out: Path) -> int:
    from woof.netcdf_bridge import open_dataset
    ds = open_dataset(domain)
    lat = np.asarray(ds.variables["XLAT"][...], dtype=np.float64)
    lon = np.asarray(ds.variables["XLONG"][...], dtype=np.float64)
    lat, lon = lat.reshape(-1, *lat.shape[-2:])[0], lon.reshape(-1, *lon.shape[-2:])[0]
    lat.ravel().astype("<f8").tofile(out/"target_latitude.f64")
    lon.ravel().astype("<f8").tofile(out/"target_longitude.f64")
    return lat.size


def produce(source: str, start: datetime, end: datetime, domain: Path, out: Path, *,
            cache: Path | None = None, files: list[Path] | None = None, jobs: int = 0,
            keep_downloads: bool = False, fetch_workers: int = 8) -> dict:
    from woof.netcdf_bridge import resolve_netcdf_bin
    row = source_row(source)
    out.mkdir(parents=True, exist_ok=True)
    cache = cache or out/"downloads"
    cache.mkdir(parents=True, exist_ok=True)
    fetched = []
    if files is None:
        with ThreadPoolExecutor(fetch_workers) as pool:
            files = list(pool.map(lambda u: fetch(u, cache), urls(row, start, end)))
        fetched = files
    columns = target_columns(domain, out)
    manifest = {"schema": "gpuwm-snow-lap-deposition-v1", "source": source,
                "files": [str(p) for p in files], "latitude": row["latitude"],
                "longitude": row["longitude"], "time": row["time"],
                "window_seconds": row["window_seconds"], "units": "kg m-2 s-1",
                "channels": channels(row),
                "target_latitude": str(out/"target_latitude.f64"),
                "target_longitude": str(out/"target_longitude.f64")}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=1))
    subprocess.run([str(resolve_netcdf_bin()), "snow-lap-deposition", str(out/"manifest.json"),
                    str(out), str(jobs)], check=True)
    if fetched and not keep_downloads:
        for p in fetched:
            p.unlink(missing_ok=True)
    receipt = json.loads((out/"receipt.json").read_text())
    assert receipt["columns"] == columns
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m woof.snow_lap_deposition", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="geos-fp")
    ap.add_argument("--start", required=True, type=datetime.fromisoformat)
    ap.add_argument("--end", required=True, type=datetime.fromisoformat)
    ap.add_argument("--domain", required=True, type=Path, help="wrfinput_d0N or geo_em with XLAT/XLONG")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--files", type=Path, nargs="*", help="already-downloaded source files")
    ap.add_argument("--jobs", type=int, default=0, help="decode workers, 0 = all cores")
    ap.add_argument("--keep-downloads", action="store_true")
    a = ap.parse_args(argv)
    r = produce(a.source, a.start, a.end, a.domain, a.out, cache=a.cache, files=a.files,
                jobs=a.jobs, keep_downloads=a.keep_downloads)
    print(f"{len(r['times'])} times x {len(r['channels'])} channels x {r['columns']} columns -> {a.out/'receipt.json'}")


if __name__ == "__main__":
    main()
