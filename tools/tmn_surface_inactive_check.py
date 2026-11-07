"""Hash every resident carrier of an SST-off public-input continuation."""
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    import cupy as cp
    import numpy as np
    from woof.wrfinput_door import resolve_wrfinput_run
    from woof.ingest.wrfinput import (read_wrfinput, read_wrfbdy,
        restore_domain_state, initialize_wrfinput_physics)
    from woof.ingest.lateral_bc import attach_lateral_boundaries
    from woof.core.dycore import step
    from tilestream.physics_inventory import carrier_manifest
    run = resolve_wrfinput_run(args.inputs)
    exp = run.experiment
    overrides = dict(run_seconds=args.steps*exp.root.run.dt)
    if hasattr(exp.root.run, "sst_update"):
        overrides["sst_update"] = 0
    cfg = replace(exp.root.run, **overrides)
    dimensions = dict(west_east=cfg.nx, west_east_stag=cfg.nx+1,
        south_north=cfg.ny, south_north_stag=cfg.ny+1,
        bottom_top=cfg.nz, bottom_top_stag=cfg.nz+1,
        soil_layers_stag=cfg.num_soil_layers)
    restored = read_wrfinput(args.inputs/"wrfinput_d01", expected_dimensions=dimensions, cfg=cfg)
    state = restore_domain_state(restored, cfg)
    initialize_wrfinput_physics(state, restored, cfg,
        radiation_start_time=exp.start_time,
        radiation_latitude=restored.raw["XLAT"], radiation_longitude=restored.raw["XLONG"])
    boundary = read_wrfbdy(args.inputs/"wrfbdy_d01", restored=restored, cfg=cfg,
        run_seconds=cfg.run_seconds, forcing_interval_seconds=run.coverage.forcing_interval_seconds,
        spec_bdy_width=cfg.spec_bdy_width)
    attach_lateral_boundaries(state, boundary)
    frames = []
    for index in range(args.steps+1):
        if index:
            step(state, cfg)
        cp.cuda.runtime.deviceSynchronize()
        hashes = {}
        for name, value in sorted(carrier_manifest(state).items()):
            host = cp.asnumpy(value)
            if not np.isfinite(host).all():
                raise ValueError(f"nonfinite carrier {name} at step {index}")
            hashes[name] = dict(dtype=str(host.dtype), shape=list(host.shape),
                sha256=hashlib.sha256(host.tobytes()).hexdigest())
        frames.append(dict(step=index, elapsed_seconds=float(state.elapsed_seconds), fields=hashes))
    args.output.write_text(json.dumps(dict(status="PASS", tmn_update=cfg.tmn_update,
        sst_update=getattr(cfg, "sst_update", 0), dt=cfg.dt, frames=frames), indent=2))

if __name__ == "__main__":
    main()
