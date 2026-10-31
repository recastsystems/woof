"""Fetch and prepare the two-hour radar-band example with native preprocessing.

All paths are explicit and contained under --work, apart from the read-only
--geography input. Run forecast stages inside the host's GPU lease.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
SOURCE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20240507/conus/"
OBJECTS = tuple(f"hrrr.t00z.wrfprsf{hour:02d}.grib2" for hour in range(3)) + ("hrrr.t00z.wrfsfcf00.grib2",)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(work):
    inputs = work/"inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    def one(name):
        target = inputs/name
        with urlopen(SOURCE+name, timeout=120) as response:
            size = int(response.headers["Content-Length"])
            if not target.exists() or target.stat().st_size != size:
                with target.with_suffix(".part").open("wb") as stream:
                    while block := response.read(1024*1024):
                        stream.write(block)
                if target.with_suffix(".part").stat().st_size != size:
                    raise RuntimeError("incomplete public input: "+name)
                target.with_suffix(".part").replace(target)
        print("Fetched "+name,flush=True)
        return dict(url=SOURCE+name,path=str(target),bytes=size)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        objects = list(pool.map(one, OBJECTS))
    (work/"fetch-receipt.json").write_text(json.dumps(dict(objects=objects,seconds=time.perf_counter()-start,
        license="Public NOAA US-government forecast data; exact content digests are sealed by native preparation"),indent=2)+"\n")


def prepare(work,geography,workers):
    (work/"case.toml").write_text((HERE/"simulated_radar_bands.toml").read_text().replace("<geography>",str(geography)))
    (work/"case.namelist.wps").write_text((HERE/"simulated_radar_bands.namelist.wps").read_text())
    command = [sys.executable,"-m","woof","prep","--source","hrrr-prs",
               "--author-input-manifest",str(work/"inputs.json")]
    for name in OBJECTS[:3]:
        path = str(work/"inputs"/name)
        command += ["--input",path,"--supplement","hrrr_prs_in_band_surface="+path]
    command += ["--supplement","vegetation_surface_data="+str(work/"inputs"/OBJECTS[-1]),
        "--experiment-config",str(work/"case.toml"),"--wps-namelist",str(work/"case.namelist.wps"),
        "--output-root",str(work/"prepared"),"--preprocess-backend","cpu",
        "--preprocess-workers",str(workers),"--no-stock-wrf-export","--geog-root",str(geography)]
    (work/"prepare-command.json").write_text(json.dumps(command,indent=2)+"\n")
    subprocess.run(command,check=True,timeout=1200)
    root = work/"prepared"
    header = json.loads((root/"prepared-cache/header.json").read_text())
    if header["status"] != "READY":
        raise RuntimeError("native preparation did not produce a READY cache")
    binding = dict(source="hrrr-prs",source_manifest_sha256=digest(root/"source-evidence/input-manifest.json"),
        proof_sha256=digest(root/"proof.json"),prepared_content_sha256=header["content_sha256"],
        experiment_config=str(work/"case.toml"),wps_namelist=str(work/"case.namelist.wps"),
        run_seconds=7200.,history_interval_seconds=1800.)
    (work/"binding.json").write_text(json.dumps(binding,indent=2)+"\n")


def forecast(work,arm,smoke=False,label=None):
    label = label or arm
    binding = json.loads((work/"binding.json").read_text())
    if smoke:
        binding["run_seconds"] = 24.
    command = [sys.executable,"-m","woof.prepared_single_domain_forecast",
        "--prepared-root",str(work/"prepared"),"--outdir",str(work/label),
        "--tiles",'{"mode":"off"}',"--stream-init","resident","--render-products","none","--io-mode","history"]
    for key,value in binding.items():
        command += ["--"+key.replace("_","-"),str(value)]
    if arm == "on":
        command += ["--simulated-radar-table",json.dumps(dict(enabled=True,sites=["TOKC"],
            scan_strategy="custom",elevations_deg=[0.5],range_km=150.,gate_spacing_m=500.,
            azimuth_step_deg=1.,formats=["cfradial1"],fields=["reflectivity","zdr","rhohv","kdp","phidp"]))]
    (work/(label+"-command.json")).write_text(json.dumps(command,indent=2)+"\n")
    started = time.perf_counter()
    result = subprocess.run(command,timeout=1800)
    (work/(label+"-wall.json")).write_text(json.dumps(dict(returncode=result.returncode,
        wall_seconds=time.perf_counter()-started,command=command),indent=2)+"\n")
    result.check_returncode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("fetch","prepare","off","on","smoke"))
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--geography",type=Path)
    p.add_argument("--workers",type=int,default=os.cpu_count())
    p.add_argument("--run-label",help="output subdirectory for a fresh run")
    a = p.parse_args()
    if a.run_label is not None and (Path(a.run_label).name != a.run_label or a.run_label in ("", ".", "..")):
        p.error("--run-label must be one subdirectory name, so a run cannot escape --work")
    work = a.work.resolve()
    work.mkdir(parents=True,exist_ok=True)
    if a.stage == "fetch": fetch(work)
    elif a.stage == "prepare":
        if a.geography is None: p.error("prepare requires --geography")
        prepare(work,a.geography.resolve(),a.workers)
    else: forecast(work,a.stage,smoke=a.stage=="smoke",label=a.run_label)


if __name__ == "__main__": main()
