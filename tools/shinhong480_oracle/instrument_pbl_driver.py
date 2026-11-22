#!/usr/bin/env python3
"""Capture native WRF v4.8.0 Shin-Hong inputs and outputs.

    python instrument_pbl_driver.py WRF_TREE

Patches phys/module_pbl_driver.F of a COPY of a strict build (never the
referee tree): around CALL shinhong it writes, at the listed ITIMESTEPs, one
little-endian stream file per tile before the call and one after.  The
scheme objects (physics_mmm/bl_shinhong.o, module_bl_shinhong.o) are not
touched; build.sh checks they are byte-identical to the referee's.

File captures/shinhong-<step>-<its>-<jts>-in.bin:
  int32[10]  nx ny nz itimestep flag_qc flag_qi nonlocal scu dissip 0
  float32    dtbl
  3-D planes (nx, nz, ny), Fortran order: u v th t qv qc qi p pi dz8w tke rthraten
  3-D (nx, nz+1, ny): p8w
  2-D (nx, ny): psfc znt ust fm fhh xland hfx qfx wspd br u10 v10 uoce voce
                ctopo ctopo2 dx2d f
File ...-out.bin:
  int32[4]   nx ny nz itimestep
  3-D: rublten rvblten rthblten rqvblten rqcblten rqiblten exch_h exch_m tke el
  2-D: pblh wstar delta u10 v10;  int32 2-D: kpbl
"""
import pathlib
import sys

p = pathlib.Path(sys.argv[1]) / "phys/module_pbl_driver.F"
s = p.read_text()
STEPS = "(/1,2,50,100,240/)"
IN3 = ["u_phytmp", "v_phytmp", "th_phy", "t_phy", "qv_curr", "qc_curr",
       "qi_curr", "p_phy", "pi_phy", "dz8w", "tke_pbl", "RTHRATEN"]
IN2 = ["psfc", "znt", "ust", "fm", "fhh", "xland", "hfx", "qfx", "wspd", "br",
       "u10", "v10", "uoce", "voce", "ctopo", "ctopo2", "dx2d", "f"]
OUT3 = ["rublten", "rvblten", "rthblten", "rqvblten", "rqcblten", "rqiblten",
        "exch_h", "exch_m", "tke_pbl", "el_pbl"]
OUT2 = ["pblh", "wstar", "delta", "u10", "v10"]


def head(tag):
    return [
        f"             IF (ANY(ITIMESTEP == {STEPS})) THEN",
        f"              WRITE(cap_file,'(A,I0,A,I0,A,I0,A)') 'captures/shinhong-',ITIMESTEP,'-',its,'-',jts,'-{tag}.bin'",
        "              OPEN(newunit=cap_unit,file=trim(cap_file),access='stream',form='unformatted',"
        "status='replace',convert='little_endian')",
    ]


def w3(n, top="kte"):
    return f"              WRITE(cap_unit) {n}(its:ite,kts:{top},jts:jte)"


def w2(n):
    return f"              WRITE(cap_unit) {n}(its:ite,jts:jte)"


def block_in():
    lines = head("in")
    lines.append("              WRITE(cap_unit) ite-its+1,jte-jts+1,kte-kts+1,itimestep,"
                 "merge(1,0,flag_qc),merge(1,0,flag_qi),merge(1,0,shinhong_nonlocal_flux),"
                 "merge(1,0,shinhong_scu_mixing),merge(1,0,shinhong_ke_dissipation),0")
    lines.append("              WRITE(cap_unit) dtbl")
    lines += [w3(n) for n in IN3]
    lines.append(w3("p8w", "kte+1"))
    lines += [w2(n) for n in IN2]
    lines += ["              CLOSE(cap_unit)", "             ENDIF"]
    return "\n".join(lines) + "\n"


def block_out():
    lines = head("out")
    lines.append("              WRITE(cap_unit) ite-its+1,jte-jts+1,kte-kts+1,itimestep")
    lines += [w3(n) for n in OUT3]
    lines += [w2(n) for n in OUT2]
    lines.append(w2("kpbl"))
    lines += ["              CLOSE(cap_unit)", "             ENDIF"]
    return "\n".join(lines) + "\n"


call = "             CALL shinhong(                                         &\n"
assert s.count(call) == 1, s.count(call)
i = s.index(call)
end = "              ,ITS=its,ITE=ite,JTS=jts,JTE=jte,KTS=kts,KTE=kte      &\n" \
      "                                                                    )\n"
j = s.index(end, i) + len(end)
s = s[:i] + block_in() + s[i:j] + block_out() + s[j:]
decl = "   IMPLICIT NONE\n"
k = s.index(decl, s.index("SUBROUTINE pbl_driver(")) + len(decl)
s = s[:k] + "   INTEGER :: cap_unit\n   CHARACTER(len=256) :: cap_file\n" + s[k:]
p.write_text(s)
print("patched", p)
