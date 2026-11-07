"""Observe WRF and GPUWM at the same concentration-space stage boundaries.

The official oracle remains the referee. A separate, hashed diagnostic build
adds output-only calls. Its final fields must equal the official oracle before
any trace is used. Trace arrays stay on the box; only hashes and summaries are
receipts. No production kernel is replaced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PHASES = ("sediment", "gs", "nucond")
TRACE_FIELDS = ("temperature", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
                "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh")
SOURCE_SHA = "1eb1b138b75ff3b0cfe33c23779f4ec9b72e57a5455a53ef11c9e55ae0f42722"


def read_trace(path, *, nx, nz, nsteps, it0):
    outputs = {p: np.empty((nsteps, len(TRACE_FIELDS), nx, nz), np.float32) for p in PHASES}
    seen = set()
    with Path(path).open() as stream:
        for row in csv.reader(stream):
            if len(row) != 6 + len(TRACE_FIELDS):
                raise ValueError("trace row does not contain every field")
            phase = row[0].strip()
            step, loop, jy, ix, k = map(int, row[1:6])
            if phase not in PHASES or not (it0 <= step < it0+nsteps):
                raise ValueError("unexpected phase or step")
            if loop != 1 or jy != 1 or not (1 <= ix <= nx and 1 <= k <= nz):
                raise ValueError("unexpected trace coordinates")
            key = (phase, step, ix, k)
            if key in seen:
                raise ValueError("duplicate trace row")
            seen.add(key)
            words = np.array([int(word, 16) for word in row[6:]], np.uint32)
            values = words.view(np.float32)
            if not np.isfinite(values).all():
                raise ValueError("nonfinite trace")
            outputs[phase][step-it0, :, ix-1, k-1] = values
    if len(seen) != len(PHASES) * nsteps * nx * nz:
        raise ValueError("incomplete trace")
    return outputs


def summarize(outputs):
    result = {}
    for phase, array in outputs.items():
        if array.dtype != np.float32 or array.ndim != 4 or not np.isfinite(array).all():
            raise ValueError("invalid stage array")
        result[phase] = {}
        for index, field in enumerate(TRACE_FIELDS):
            values = np.ascontiguousarray(array[:, index])
            result[phase][field] = {
                "shape": list(values.shape),
                "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                "by_step_column": [[hashlib.sha256(np.ascontiguousarray(col).tobytes()).hexdigest()
                                    for col in step] for step in values],
                "min": float(values.min()), "max": float(values.max()),
            }
    return result


HELPER = r'''
  ! Output-only lane diagnostic. Called after each stage, never from numerics.
  subroutine nssl_lane_trace(phase,nx,ny,nz,na,an,temp,istep,iloop,ij)
    implicit none
    character(len=*), intent(in) :: phase
    integer, intent(in) :: nx,ny,nz,na,istep,iloop,ij
    real, intent(in) :: an(nx,ny,nz,na),temp(nx,ny,nz)
    character(len=1024) :: trace_path
    integer :: status,unit,ix,k,f
    integer :: slots(16)
    real :: values(17)
    call get_environment_variable('NSSL2_VARIANT_TRACE',trace_path,status=status)
    if (status /= 0 .or. len_trim(trace_path) == 0) return
    slots=(/lv,lc,lr,li,ls,lh,lhl,lnc,lnr,lni,lns,lnh,lnhl,lccn,lvh,lvhl/)
    open(newunit=unit,file=trim(trace_path),status='unknown',position='append',action='write')
    do ix=1,nx
      do k=1,nz
        values=0.0
        values(1)=temp(ix,1,k)
        do f=1,16
          if (slots(f)>0 .and. slots(f)<=na) values(f+1)=an(ix,1,k,slots(f))
        enddo
        write(unit,'(A,5(",",I0),17(",",Z8.8))') &
          trim(phase),istep,iloop,ij,ix,k,values
      enddo
    enddo
    close(unit)
  end subroutine nssl_lane_trace
'''


def build(wrf, target):
    if not target.exists():
        subprocess.run(["bash", str(HERE/"build.sh"), str(wrf), str(target)], check=True)
    elif not all((target/name).is_file() for name in (
            "SOURCE_SHA256", "FLAGS.txt", "stub_wrf.o", "variant_column.o", "nssl2_variant_column")):
        raise ValueError("existing trace directory is not a complete official oracle build")
    module = target / "module_mp_nssl_2mom.F"
    source = module.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA:
        raise ValueError("trace source is not the official WRF v4.6.1 module")
    # Unique complete statement tails preserve all original arithmetic.
    sites = {
        "sediment": "     &   ,timesed1,timesed2,timesed3,zmaxsed,timesetvt)\n",
        "gs": "     &   ,elec2,its,ids,ide,jds,jde          &\n     & )\n",
        "nucond": "     &  ,ssat,t00,t77,flag_qndrop)\n",
    }
    anchors = {}
    for phase, anchor in sites.items():
        if source.count(anchor) != 1:
            raise ValueError(f"trace anchor is ambiguous: {phase}")
        anchors[phase] = source[:source.index(anchor)].count("\n") + 1
        source = source.replace(anchor, anchor +
            f"      call nssl_lane_trace('{phase}',nx,ny,nz,na,an,t0,itimestep,loopcnt,jy)\n")
    end = "END MODULE module_mp_nssl_2mom"
    if source.count(end) != 1:
        raise ValueError("ambiguous module end")
    source = source.replace(end, HELPER + "\n" + end)
    module.write_text(source)
    flags = (target / "FLAGS.txt").read_text().split()
    subprocess.run(["gfortran", "-c", *flags, "-cpp", "-DWRF_CHEM=0", module.name], cwd=target, check=True)
    subprocess.run(["gfortran", "-O2", "-o", "nssl2_variant_column", "module_mp_nssl_2mom.o",
                    "stub_wrf.o", "variant_column.o"], cwd=target, check=True)
    (target/"TRACE_SOURCE.json").write_text(json.dumps({"official_sha256": SOURCE_SHA,
        "instrumented_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "anchors": anchors, "compiler_flags": flags}, indent=2)+"\n")


def wrf_trace(exe, official_out, out, label, nsteps=3):
    import cases
    import run_oracle
    mode = cases.MODE_BY_LABEL[label]
    trace = out / "stages.csv"
    out.mkdir(parents=True, exist_ok=False)
    os.environ["NSSL2_VARIANT_TRACE"] = str(trace.resolve())
    run_oracle.run_mode(exe.resolve(), out, mode, nsteps, 6.0, 2)
    arrays = read_trace(trace, nx=6, nz=cases.NZ, nsteps=nsteps, it0=2)
    with np.load(official_out/f"{label}.npz") as ref, np.load(out/f"{label}.npz") as traced:
        checks = {key: bool(np.array_equal(ref[key].view(np.uint32), traced[key].view(np.uint32)))
                  for key in ref.files if key.startswith("wrf_")}
    if len(checks) != 53 or not all(checks.values()):
        raise ValueError("diagnostic instrumentation changed official final outputs")
    receipt = {"mode": label, "settings": {"nsteps":nsteps,"dt":6.0,"itimestep0":2},
               "official_output_identity": checks, "stages": summarize(arrays)}
    (out/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    np.savez(out/"stages.npz", **arrays)


def gpu_trace(out, label, nsteps=3):
    import cupy as cp
    import cases
    import run_gpu
    from woof.core import nssl2_fused_gs as gs, nssl2_nucond as nu
    from woof.core import nssl2_production_coordinator as coordinator
    from woof.certify.kernel_manifest import kernel_manifest
    mode = cases.MODE_BY_LABEL[label]
    resolved = run_gpu.admit(mode)
    records = {p: [] for p in PHASES}
    current = []
    normal_gather = coordinator.gather_initialize_and_sediment
    normal_gs, normal_nu = gs.launch_fused_gs, nu.launch_nucond

    def capture(phase, workspace, temperature):
        state = cp.asnumpy(workspace.state)[:, :, 0, :].transpose(0, 2, 1)
        temp = cp.asnumpy(temperature)[:, 0, :].T
        records[phase].append(np.concatenate((temp[None], state), axis=0))

    def gather(*args, **kw):
        workspace = normal_gather(*args, **kw)
        current[:] = [workspace]
        capture("sediment", workspace, kw["temperature_k"])
        return workspace

    def fused(*args, **kw):
        result = normal_gs(*args, **kw)
        capture("gs", args[0], args[1] * args[4])
        return result

    def condensation(*args, **kw):
        result = normal_nu(*args, **kw)
        capture("nucond", current[0], args[0] * args[3])
        return result

    coordinator.gather_initialize_and_sediment = gather
    gs.launch_fused_gs, nu.launch_nucond = fused, condensation
    try:
        output, inputs = run_gpu.run_mode(mode, resolved, nsteps, 6.0, 2)
    finally:
        coordinator.gather_initialize_and_sediment = normal_gather
        gs.launch_fused_gs, nu.launch_nucond = normal_gs, normal_nu
    arrays = {p: np.stack(rows) for p, rows in records.items()}
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"mode": label, "arm": "strict" if os.environ.get("GPUWM_WRF_EXACT")=="1" else "default",
               "settings":{"nsteps":nsteps,"dt":6.0,"itimestep0":2}, "input_sha256":inputs,
               "output_sha256":{n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in output.items()},
               "stages":summarize(arrays), "kernel_manifest":kernel_manifest()}
    np.savez(out/"stages.npz", **arrays)
    (out/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    b=sub.add_parser("build"); b.add_argument("wrf",type=Path); b.add_argument("out",type=Path)
    w=sub.add_parser("wrf"); w.add_argument("exe",type=Path); w.add_argument("official",type=Path); w.add_argument("out",type=Path)
    w.add_argument("--mode",default="mp18-default")
    w.add_argument("--nsteps",type=int,default=3)
    g=sub.add_parser("gpu"); g.add_argument("out",type=Path); g.add_argument("--mode",default="mp18-default")
    g.add_argument("--nsteps",type=int,default=3)
    args=parser.parse_args()
    if args.command in ("wrf", "gpu") and args.nsteps < 1:
        parser.error("nsteps must be positive")
    if args.command=="build": build(args.wrf,args.out)
    elif args.command=="wrf": wrf_trace(args.exe,args.official,args.out,args.mode,args.nsteps)
    else: gpu_trace(args.out,args.mode,args.nsteps)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
