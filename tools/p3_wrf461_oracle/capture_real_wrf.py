#!/usr/bin/env python3
"""Add I/O-only P3 wrapper capture to an OWNED copy of native WRF.

The pinned arithmetic stays unchanged. At the requested model timestep,
each MPI tile records its central j slab before and after the wrapper.
The .bin/.ref layouts match run_woof.py and compare.py. KEEP the .bin,
.ref, .meta and index.json behind every replay claim in the lane's paper
evidence. Hashes and a comparison receipt do not replace the real capture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PIN = "716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6"
IN3D = ["th_3d", "qv_3d", "qc_3d", "qr_3d", "qnr_3d", "qi1_3d",
        "qni1_3d", "qir1_3d", "qib1_3d", "th_old_3d", "qv_old_3d",
        "pii", "p", "dz", "w", "nc_3d", "qzi1_3d"]
OUT3D = IN3D[:11] + ["diag_zdbz_3d", "diag_effc_3d", "diag_effi_3d",
                     "diag_vmi_3d", "diag_di_3d", "diag_rhopo_3d"] + IN3D[-2:]
SURFACE = ["rainnc", "rainncv", "sr", "snownc", "snowncv"]


def writes(names, *, surface=False):
    sub = "(its:ite,cap_j)" if surface else "(its:ite,kts:kte,cap_j)"
    return "\n".join(f"      write(cap_u) {name}{sub}" for name in names)


def instrument(source: Path, step: int, variant: int = 51):
    inputs = IN3D if variant == 53 else IN3D[:-1]
    outputs = OUT3D if variant == 53 else OUT3D[:-1]
    guard = "present(qzi1_3d)" if variant == 53 else ".not.present(qzi1_3d)"
    original = source.read_bytes()
    if hashlib.sha256(original).hexdigest() != PIN:
        raise ValueError(f"{source} is not the unmodified pinned WRF P3 source")
    text = original.decode()
    start = text.index("   scpf_on=.false. ! cloud fraction version not used with WRF")
    declaration = """
! GPUWM_REAL_CAPTURE: I/O only, before any wrapper arithmetic.
   integer :: cap_u, cap_j
   logical :: cap_on
   character(len=96) :: cap_base

"""
    text = text[:start] + declaration + text[start:]
    start = text.index("   scpf_on=.false. ! cloud fraction version not used with WRF")
    before = f"""
   cap_on = present(nc_3d).and.{guard}.and.itimestep.eq.{step}
   if (cap_on) then
      cap_j = (jts+jte)/2
      write(cap_base,'(A,I5.5,A,I5.5,A,I8.8)') 'p3_capture_',its,'_',jts,'_',itimestep
      open(newunit=cap_u,file=trim(cap_base)//'.meta',status='replace',action='write')
      write(cap_u,*) ite-its+1,kte-kts+1,dt,itimestep,its,cap_j
      close(cap_u)
      open(newunit=cap_u,file=trim(cap_base)//'.bin',status='replace',access='stream', &
           form='unformatted',convert='little_endian',action='write')
{writes(inputs)}
{writes(SURFACE, surface=True)}
      close(cap_u)
      open(newunit=cap_u,file=trim(cap_base)//'.ref',status='replace',access='stream', &
           form='unformatted',convert='little_endian',action='write')
{writes(outputs)}
{writes(SURFACE, surface=True)}
      close(cap_u)
   endif

"""
    text = text[:start] + before + text[start:]
    end = text.index("   END SUBROUTINE mp_p3_wrapper_wrf")
    after = f"""
   if (cap_on) then
      open(newunit=cap_u,file=trim(cap_base)//'.ref',status='old',access='stream', &
           form='unformatted',convert='little_endian',action='write',position='append')
{writes(outputs)}
{writes(SURFACE, surface=True)}
      close(cap_u)
   endif

"""
    text = text[:end] + after + text[end:]
    source.write_text(text)
    print(json.dumps(dict(source=str(source), original_sha256=PIN,
                          instrumented_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          capture_step=step, variant=variant, change="wrapper I/O only"), sort_keys=True))


def index(directory: Path, variant: int = 51):
    extras = ["nc", "qzi"] if variant == 53 else ["nc"]
    from tools.p3_wrf461_oracle.run_woof import IN3D as INPUTS, OUT3D as OUTPUTS, OUT2D
    cases = []
    for meta in sorted(directory.glob("p3_capture_*.meta")):
        ncol, nk, dt, step, x, y = meta.read_text().split()
        name = meta.stem
        path = directory / (name + ".bin")
        reference = directory / (name + ".ref")
        cases.append(dict(name=name, ncol=int(ncol), nk=int(nk), dt=float(dt.replace("D", "E")),
                          nsteps=1, native_itimestep=int(step), x=int(x), y=int(y),
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          input_bytes=path.stat().st_size,
                          reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                          reference_bytes=reference.stat().st_size,
                          origin="captured native WRF v4.7.1 forecast wrapper", in_claim=True))
    if not cases:
        raise ValueError(f"no native capture metadata in {directory}")
    payload = dict(variant=variant, predict_nc=True, in3d=INPUTS + extras,
                   out3d=OUTPUTS + extras, out2d=OUT2D, cases=cases,
                   source_sha256=PIN)
    (directory / "index.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(dict(cases=len(cases), columns=sum(c["ncol"] for c in cases))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("instrument")
    p.add_argument("source", type=Path)
    p.add_argument("--step", type=int, default=121)
    p = sub.add_parser("index")
    p.add_argument("directory", type=Path)
    for command in sub.choices.values():
        command.add_argument("--variant", type=int, choices=(51, 53), default=51)
    args = ap.parse_args()
    if args.mode == "instrument":
        instrument(args.source, args.step, args.variant)
    else:
        index(args.directory, args.variant)


if __name__ == "__main__":
    main()
