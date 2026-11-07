"""Bounded integration smoke and case preparation; never a whole-run receipt.

Run on the assigned node. GPU mode must be inside its pool lease.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "smoke", "render"))
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(cfg["output"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        from woof.ocean.waves.prepared import prepare_mesh
        source = json.loads((Path(cfg["source_prepared"])/"manifest.json").read_text())
        start = datetime.fromtimestamp(source["initial_valid_time"], timezone.utc)
        path = prepare_mesh(mesh=None, grid_path=cfg["grid"], start_time=start,
            end_time=start+timedelta(seconds=cfg.get("prepare_seconds", 3600)),
            output_root=root, options={"source_prepared": cfg["source_prepared"]})
        (root/"prepared-path.json").write_text(json.dumps({"path":str(path)})+"\n")
        print(json.dumps({"prepared":str(path)}), flush=True)
        return
    if args.mode == "render":
        from woof.hex.wave_products import render
        print(json.dumps(render(root/"smoke",cfg["grid"],root/"pictures",domain="global")),flush=True)
        return
    import cupy as cp
    import numpy as np
    from woof.hex.mesh import Mesh
    from woof.hex.wave_options import configuration
    from woof.hex.wave_prepared import load
    from woof.hex.waves import HexWaveService
    from woof.hex.mpasphys import sf_mynn_v842
    from woof.hex.mpasphys.rrfs import sf_mynnsfc_v600
    started = time.perf_counter()
    receipt = {"schema":"gpuwm-hex.waves.smoke/v1","status":"RUNNING",
               "whole_run":False,"config":cfg,"stages":{}}
    target = root/"smoke-receipt.json"
    def save():
        receipt["wall_seconds"] = time.perf_counter()-started
        target.write_text(json.dumps(receipt,indent=2)+"\n")
    def stage(name):
        cp.cuda.get_current_stream().synchronize()
        receipt["stages"][name] = time.perf_counter()-started
        save()
        print(json.dumps({"stage":name,"elapsed":receipt["stages"][name]}),flush=True)
    save()
    try:
        # Both feedback variants are compiled and their lookup tables initialized.
        # The real atmosphere run is still required to prove complete flux behavior.
        sf_mynn_v842._module(waves=True)
        sf_mynnsfc_v600.module(waves=True)
        stage("surface_variants_compiled")
        mesh = Mesh.from_netcdf(cfg["grid"])
        if cfg.get("synthetic_spectrum", False):
            from woof.hex.wave_prepared import Prepared
            n = int(mesh.dimensions["nCells"])
            epoch = 0.
            partitions = cp.zeros((n,4,4),cp.float32)
            partitions[:,0,:] = cp.asarray([2.,18.,315.,1.],cp.float32)
            prepared = Prepared(root,{"grid":{"ncell":n},"initial_valid_time":epoch,
                "boundary_times":[0.,3600.],"stations":[]}, {
                "lat":cp.rad2deg(cp.asarray(mesh.arrays["latCell"])).astype(cp.float32),
                "lon":cp.rad2deg(cp.asarray(mesh.arrays["lonCell"])).astype(cp.float32),
                "depth":cp.full(n,1000.,cp.float32),"wet":cp.ones(n,cp.float32),
                "initial_partitions":partitions,"boundary_cells":cp.empty(0,cp.int32),
                "boundary_partitions":cp.empty((2,0,4,4),cp.float32),
                "u10_east":cp.full(n,10.,cp.float32),"v10_north":cp.zeros(n,cp.float32)},
                "synthetic-uniform-deep-water-smoke")
            receipt["input_scope"] = "real native mesh; synthetic all-water depth and spectrum; no observation validation"
        else:
            prepared = load(json.loads((root/"prepared-path.json").read_text())["path"])
            receipt["input_scope"] = "shared real-case prepared input"
        options = configuration("on", cfg.get("options",{}))
        epoch = prepared.manifest["initial_valid_time"]
        service = HexWaveService(mesh,prepared,options,output_root=root/"smoke",
            start_epoch=epoch,end_seconds=600)
        if cfg.get("diagnose_stages", False):
            original_stage = service.solver._stage
            receipt["stage_probes"] = []
            receipt["profiling_scope"] = "separate failure diagnosis with per-stage synchronizations; not performance evidence"
            def probe_action(name, a):
                bad = ~cp.isfinite(a) | (a < -1.e-8)
                cells = cp.flatnonzero(cp.any(bad, axis=(1,2)))
                probe = dict(name=name,minimum=float(cp.nanmin(a)),maximum=float(cp.nanmax(a)),
                    nonfinite_count=int(cp.count_nonzero(~cp.isfinite(a))),
                    bad_cells=int(cells.size),bad_latitudes=cp.asnumpy(service.backend.lat[cells[:8]]).tolist())
                receipt["stage_probes"].append(probe)
                save()
                print(json.dumps(probe),flush=True)
                if cells.size:
                    raise RuntimeError("first invalid action stage: "+name)
            def checked_stage(name, function):
                original_stage(name, function)
                probe_action(name, service.solver.state)
            service.solver._stage = checked_stage
            original_advect = service.backend.advect
            def checked_advect(*args, **kwargs):
                result = original_advect(*args, **kwargs)
                probe_action("voronoi_advect", result)
                return result
            service.backend.advect = checked_advect
            original_launch = service.solver._launch
            def checked_launch(name, *args, **kwargs):
                original_launch(name, *args, **kwargs)
                if name in ("refract", "wave_frequency"):
                    probe_action(name, service.solver.state)
            service.solver._launch = checked_launch
        ids = cp.arange(service.n,dtype=cp.int32)
        binding = service.bind(local=ids,global_cells=ids,all_global_cells=ids,label="smoke")
        receipt["wave_pool_used_bytes"] = cp.get_default_memory_pool().used_bytes()
        receipt["wave_pool_held_bytes"] = cp.get_default_memory_pool().total_bytes()
        receipt["solver_allocation"] = service.solver.allocation_receipt()
        receipt["wet_cells"] = service.backend.ncell
        receipt["cells"] = service.n
        stage("service_constructed")
        service.output(0.)
        fields = {"u10":service.buffers["u10"].copy(),"v10":service.buffers["v10"].copy(),
                  "psfc":cp.full(service.n,101325,cp.float32),"ust":cp.full(service.n,.3,cp.float32),
                  "rho_air":cp.full(service.n,1.225,cp.float32)}
        before = {k:v.copy() for k,v in fields.items()}
        service.handoff(binding,fields,300.)
        stage("first_coupled_interval")
        saved = service.snapshot()
        service.handoff(binding,fields,600.)
        expected = service.solver.state.copy()
        service.restore(saved)
        restored = service.solver.checkpoint_arrays()
        receipt["restart_load_bitwise"] = all(
            bool(cp.array_equal(restored[k],cp.asarray(v)))
            for k,v in saved["solver"]["arrays"].items())
        receipt["restart_loaded_arrays"] = len(restored)
        receipt["restart_outputs_bitwise"] = all(bool(cp.array_equal(v,cp.asarray(saved["outputs"][k])))
            for k,v in vars(service.solver.outputs).items())
        service.handoff(binding,fields,600.)
        receipt["restart_action_bitwise"] = bool(cp.array_equal(expected,service.solver.state))
        receipt["restart_action_relative_max"] = float(cp.max(cp.abs(expected-service.solver.state)) /
            cp.maximum(cp.max(cp.abs(expected)),cp.float32(1.e-30)))
        receipt["restart_continuation_tolerance"] = 1.e-6
        receipt["restart_continuation_pass"] = receipt["restart_action_relative_max"] <= 1.e-6
        receipt["forcing_inputs_bitwise"] = all(bool(cp.array_equal(v,before[k])) for k,v in fields.items())
        receipt["action_finite_nonnegative"] = bool(cp.all(cp.isfinite(service.solver.state)&(service.solver.state>=0)))
        receipt["action_minimum"] = float(cp.nanmin(service.solver.state))
        receipt["action_nonfinite_count"] = int(cp.count_nonzero(~cp.isfinite(service.solver.state)))
        service.output(600.)
        receipt["service"] = service.receipt()
        receipt["wave_pool_final_bytes"] = cp.get_default_memory_pool().used_bytes()
        receipt["checks_scope"] = "two surface variants compile; native global topology; 600 s initialized waves; exact restart loading and tolerance-based continuation because DIA uses atomic sums; input read-only check is not an atmosphere identity proof; no forecast or observation skill claim"
        if not all(receipt[k] for k in ("restart_load_bitwise","restart_outputs_bitwise","restart_continuation_pass","forcing_inputs_bitwise","action_finite_nonnegative")):
            raise RuntimeError("wave integration smoke failed; see boolean checks in receipt")
        receipt["status"] = "DONE"
        stage("complete")
    except BaseException as error:
        receipt.update(status="FAILED",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
