"""Hash a short real-input run with prescribed analysis forcing disabled."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    import cupy as cp
    from woof.config import RunConfig
    from woof.ingest.wrfinput import read_wrfinput,read_wrfbdy,restore_domain_state,initialize_wrfinput_physics
    from woof.ingest.lateral_bc import attach_lateral_boundaries
    from woof.core.dycore import step
    from tilestream.physics_inventory import carrier_manifest
    cfg=RunConfig(nx=40,ny=40,nz=50,dx=3000,dy=3000,ztop=20000,dt=1,run_seconds=5,
                  moist=True,mp_physics=1,sf_sfclay_physics=91,sf_surface_physics=2,bl_pbl_physics=1,
                  ra_physics=0,cu_physics=0,km_opt=4,terrain_opt=1,hybrid_opt=2,
                  specified=True,spec_bdy_width=5,spec_zone=1,relax_zone=4)
    dimensions=dict(west_east=40,west_east_stag=41,south_north=40,south_north_stag=41,
                    bottom_top=50,bottom_top_stag=51,soil_layers_stag=4)
    restored=read_wrfinput(args.inputs/"wrfinput_d01",expected_dimensions=dimensions,cfg=cfg)
    state=restore_domain_state(restored,cfg)
    start=datetime.fromisoformat(restored.global_attributes["START_DATE"].replace("_","T")).replace(tzinfo=timezone.utc)
    initialize_wrfinput_physics(state,restored,cfg,radiation_start_time=start,
        radiation_latitude=restored.raw["XLAT"],radiation_longitude=restored.raw["XLONG"],constant_glw_wm2=300)
    boundary=read_wrfbdy(args.inputs/"wrfbdy_d01",restored=restored,cfg=cfg,run_seconds=5,
                        forcing_interval_seconds=3600,spec_bdy_width=5)
    attach_lateral_boundaries(state,boundary)
    frames=[]
    for index in range(6):
        if index:step(state,cfg)
        cp.cuda.runtime.deviceSynchronize()
        arrays=carrier_manifest(state)
        hashes={}
        for name,value in sorted(arrays.items()):
            if not bool(cp.isfinite(value).all().item()):
                raise ValueError(f"nonfinite real-run field at step {index}: {name}")
            hashes[name]=hashlib.sha256(cp.asnumpy(value).tobytes()).hexdigest()
        frames.append(dict(step=index,fields=hashes))
    args.output.write_text(json.dumps(dict(inputs=str(args.inputs),shape=[40,40,50],dt=1,steps=5,
        physics=dict(mp=1,surface_layer=91,land_surface=2,pbl=1,radiation=0,cumulus=0,constant_glw_wm2=300),
        frames=frames),indent=2))

if __name__=="__main__":main()
