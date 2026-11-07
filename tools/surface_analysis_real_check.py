"""Run a native-input forecast and verify each public analysis landing."""
import argparse
from dataclasses import replace, asdict
from datetime import timedelta
import hashlib
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("config",type=Path);p.add_argument("inputs",type=Path)
    p.add_argument("manifest",type=Path);p.add_argument("output",type=Path)
    args=p.parse_args()
    import cupy as cp
    import numpy as np
    from woof.experiment import load_experiment
    from woof.ingest.wrfinput import read_wrfinput, read_wrfbdy, restore_domain_state, initialize_wrfinput_physics
    from woof.ingest.lateral_bc import attach_lateral_boundaries
    from woof.core.dycore import step
    from woof.io.wrfout import state_frame, WrfoutWriter
    from woof.surface_analysis import selected_records, _read_plane
    from tilestream.physics_inventory import carrier_manifest
    exp=load_experiment(args.config)
    cfg=replace(exp.root.run,sst_update=1,surface_analysis=str(args.manifest.resolve()))
    dimensions=dict(west_east=cfg.nx,west_east_stag=cfg.nx+1,south_north=cfg.ny,
                    south_north_stag=cfg.ny+1,bottom_top=cfg.nz,bottom_top_stag=cfg.nz+1,
                    soil_layers_stag=cfg.num_soil_layers)
    restored=read_wrfinput(args.inputs/"wrfinput_d01",expected_dimensions=dimensions,cfg=cfg)
    state=restore_domain_state(restored,cfg)
    initialize_wrfinput_physics(state,restored,cfg,radiation_start_time=exp.start_time,
        radiation_latitude=restored.raw["XLAT"],radiation_longitude=restored.raw["XLONG"])
    boundary=read_wrfbdy(args.inputs/"wrfbdy_d01",restored=restored,cfg=cfg,
        run_seconds=cfg.run_seconds,forcing_interval_seconds=3600,spec_bdy_width=cfg.spec_bdy_width)
    attach_lateral_boundaries(state,boundary)
    driver=state.physics; stream=driver.surface_analysis
    args.output.mkdir()
    frames=[]
    count=round(cfg.run_seconds/cfg.dt)
    cadence=round(exp.root.history_interval_s/cfg.dt)
    for index in range(count+1):
        if index:step(state,cfg)
        if index % cadence and index != count:continue
        cp.cuda.runtime.deviceSynchronize()
        hashes={}
        for name,value in sorted(carrier_manifest(state).items()):
            host=cp.asnumpy(value)
            if not np.isfinite(host).all():raise ValueError(f"nonfinite {name} at step {index}")
            hashes[name]=hashlib.sha256(host.tobytes()).hexdigest()
        sst=cp.asnumpy(driver.fields["sst"])
        # The physics call consumes the beginning of the current step.
        applied=max(0,index-1)*cfg.dt
        checks=[]
        if index:
            for row in selected_records(stream.records,stream.start+applied):
                if row["target"] != "SST":continue
                expected=_read_plane(stream.path.parent,row,stream.shape)
                valid=np.isfinite(expected)
                if valid.sum()==0:raise ValueError(f"no source SST cells for {row['source']}")
                if sst[valid].tobytes()!=expected[valid].tobytes():
                    raise ValueError(f"SST differs from the prescribed source at step {index}")
                checks.append(dict(source=row["source"],valid_time=row["valid_time"],
                    valid_cells=int(valid.sum()),mean=float(sst[valid].astype('f8').mean()),
                    source_words_match=True))
        time=(exp.start_time+timedelta(seconds=index*cfg.dt)).strftime('%Y-%m-%d_%H:%M:%S')
        frame=state_frame(state,include_diagnostic_pressure=True)
        frame.update({k:restored.raw[k] for k in ("XLAT","XLONG","LANDMASK","LU_INDEX")})
        frame["SST"]=sst
        with WrfoutWriter(args.output/f"wrfout_d01_{time}",nx=cfg.nx,ny=cfg.ny,nz=cfg.nz,
            dx=cfg.dx,dy=cfg.dy,soil_layers=cfg.num_soil_layers,
            global_attrs=dict(restored.global_attributes)) as writer:
            writer.write_frame(time,frame)
        frames.append(dict(step=index,elapsed_seconds=index*cfg.dt,fields=hashes,SST=checks))
        print(json.dumps(dict(step=index,elapsed_seconds=index*cfg.dt,SST=checks)),flush=True)
    updates=stream.events
    if len(set(r["valid_time"] for r in updates if r["target"]=="SST"))<2:
        raise ValueError("forecast did not consume both daily public analyses")
    receipt=dict(status="PASS",config=asdict(cfg),start=exp.start_time.isoformat(),
        steps=count,frames=frames,updates=updates,manifest_sha256=stream.sha256,
        inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (args.inputs/"wrfinput_d01",args.inputs/"wrfbdy_d01")})
    (args.output/"receipt.json").write_text(json.dumps(receipt,indent=2,default=str))

if __name__=="__main__":main()
