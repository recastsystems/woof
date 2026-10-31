"""Scan a saved forecast at one coordinate-defined site in S, C and X bands.

Use CfRadial for lossless floating-point polarimetric moments. The source
histories are read only; this example does not step a forecast.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

from woof.simulated_radar import RadarSite, simulate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("histories",type=Path)
    p.add_argument("--outdir",type=Path,required=True)
    p.add_argument("--site",required=True)
    p.add_argument("--latitude",type=float,required=True)
    p.add_argument("--longitude",type=float,required=True)
    p.add_argument("--height-m",type=float,required=True)
    p.add_argument("--beamwidth-deg",type=float,default=.55)
    p.add_argument("--elevation-deg",type=float,default=.5)
    p.add_argument("--range-km",type=float,default=150.)
    p.add_argument("--bands",nargs="+",choices=("S","C","X"),default=["S","C","X"])
    a=p.parse_args()
    if a.histories.is_dir():
        from woof.io.wrfout import iter_wrfout_files
        histories=list(iter_wrfout_files(a.histories,include_temporaries=False))
    else: histories=[a.histories]
    if not histories: p.error("no full history frames found")
    receipts=[]
    for band in a.bands:
        site=RadarSite(a.site,a.latitude,a.longitude,a.height_m,band=band,beamwidth_deg=a.beamwidth_deg)
        started=time.perf_counter()
        result=simulate(histories,site=site,outdir=a.outdir/band,scan_strategy="custom",
            elevations_deg=[a.elevation_deg],formats=["cfradial1"],
            fields=["reflectivity","zdr","kdp"],range_km=a.range_km,
            gate_spacing_m=500.,azimuth_step_deg=1.)
        elapsed=time.perf_counter()-started
        receipt=dict(band=band,wall_seconds=elapsed,volumes=len(result.volumes),
            wall_seconds_per_volume=elapsed/len(result.volumes),manifest=str(result.manifest_path),
            input_transport=result.input_transport,
            outputs=[dict(time=v.volume_time.isoformat(),paths={k:str(p) for k,p in v.paths.items()},
                          native_seconds=v.metadata["elapsed_seconds"]) for v in result.volumes])
        receipts.append(receipt)
        print(json.dumps(receipt),flush=True)
    a.outdir.mkdir(parents=True,exist_ok=True)
    (a.outdir/"timing.json").write_text(json.dumps(receipts,indent=2)+"\n")


if __name__=="__main__":main()
