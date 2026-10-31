"""Insert wrapper capture around WRF 4.6.1 MYJSFC and MYJPBL calls.

usage: patch_drivers.py SRC_PBL SRC_SFC OUT_PBL OUT_SFC
"""
import re
import sys

SFC3 = ["dz8w", "p_phy", "p8w", "th_phy", "t_phy", "qv_curr", "qc_curr",
        "u_phy", "v_phy", "tke_pbl"]
SFC2_IN = ["ht", "tsk", "qsfc", "thz0", "qz0", "uz0", "vz0", "xland",
           "ust", "znt", "z0", "pblh", "mavail", "rmol", "akhs", "akms",
           "br", "chs", "chs2", "cqs2", "hfx", "qfx", "lh", "flhc", "flqc",
           "qgh", "cpm", "ct"]
SFC2_OUT = ["tsk", "qsfc", "thz0", "qz0", "uz0", "vz0", "ust", "znt", "z0",
            "pblh", "rmol", "akhs", "akms", "br", "chs", "chs2", "cqs2",
            "hfx", "qfx", "lh", "flhc", "flqc", "qgh", "cpm", "ct", "u10",
            "v10", "t2", "th2", "tshltr", "th10", "q2", "qshltr", "q10",
            "pshltr", "u10e", "v10e"]
SFCI = ["lowlyr", "ivgtyp"]

PBL3_IN = ["dz8w", "p_phy", "p8w", "th_phy", "t_phy", "pi_phy", "qv_curr",
           "qc_curr", "u_phy", "v_phy", "rho", "tke_pbl", "exch_h"]
PBL3_OUT = ["tke_pbl", "exch_h", "el_pbl", "rublten", "rvblten", "rthblten",
            "rqvblten", "rqcblten"]
PBL2_IN = ["ht", "tsk", "qsfc", "chklowq", "thz0", "qz0", "uz0", "vz0",
           "xland", "xice", "snow", "ust", "znt", "pblh", "ct", "akhs",
           "akms", "lh", "mixht"]
PBL2_OUT = ["qsfc", "thz0", "qz0", "uz0", "vz0", "ust", "znt", "pblh",
            "ct", "akhs", "akms", "mixht"]


def r3(name, ind):
    return (f"{ind}CALL cap_r3('{name}',{name},ims,ime,kms,kme,jms,jme,"
            f"ci0,ci1,kms,kme,cj0,cj1)")


def r2(name, ind):
    return f"{ind}CALL cap_r2('{name}',{name},ims,ime,jms,jme,ci0,ci1,cj0,cj1)"


def i2(name, ind):
    return f"{ind}CALL cap_i2('{name}',{name},ims,ime,jms,jme,ci0,ci1,cj0,cj1)"


def block(tag, phase, ind, ib, ie, jb, je, scal, f3, f2, fi, extra=()):
    o = [f"{ind}IF (cap_want(itimestep)) THEN", f"{ind}BLOCK",
         f"{ind}  INTEGER :: ci0, ci1, cj0, cj1",
         f"{ind}  ci0 = {ib}; ci1 = MIN({ie}, ide-1)",
         f"{ind}  cj0 = {jb}; cj1 = MIN({je}, jde-1)",
         f"{ind}  CALL cap_open('{tag}','{phase}',itimestep,ci0,cj0)",
         f"{ind}  CALL cap_is('itimestep',itimestep)"]
    o += [f"{ind}  {s}" for s in scal]
    o += [r3(n, ind + "  ") for n in f3]
    o += [r2(n, ind + "  ") for n in f2]
    o += [i2(n, ind + "  ") for n in fi]
    o += [f"{ind}  {s}" for s in extra]
    o += [f"{ind}  CALL cap_close()", f"{ind}END BLOCK", f"{ind}ENDIF"]
    return o


def patch(lines, use_line, call_re, end_re, before, after):
    out = []
    i_use = next(i for i, l in enumerate(lines) if l.strip() == use_line)
    i_call = next(i for i, l in enumerate(lines) if re.search(call_re, l))
    i_end = next(i for i in range(i_call, len(lines))
                 if re.search(end_re, lines[i]))
    for i, l in enumerate(lines):
        if i == i_call:
            out += before
        out.append(l)
        if i == i_use:
            out.append("   USE module_cap_io")
        if i == i_end:
            out += after
    return out


def main():
    src_pbl, src_sfc, out_pbl, out_sfc = sys.argv[1:5]
    rq = ("IF (PRESENT({0})) CALL cap_r3('{0}',{0},ims,ime,kms,kme,jms,jme,"
          "ci0,ci1,kms,kme,cj0,cj1)")
    lines = open(src_pbl).read().split("\n")
    before = block("myjpbl", "in", "             ", "its", "ite", "jts", "jte",
                   ["CALL cap_rs('dt',dt)", "CALL cap_is('stepbl',stepbl)"],
                   PBL3_IN, PBL2_IN, ["lowlyr"], [rq.format("qi_curr")])
    after = block("myjpbl", "out", "             ", "its", "ite", "jts", "jte",
                  [], PBL3_OUT, PBL2_OUT, ["kpbl"], [rq.format("rqiblten")])
    lines = patch(lines, "USE module_bl_myjpbl", r"CALL myjpbl\(DT=dt",
                  r"^\s*\)\s*$", before, after)
    open(out_pbl, "w").write("\n".join(lines))

    lines = open(src_sfc).read().split("\n")
    before = block("myjsfc", "in", "            ", "i_start(ij)", "i_end(ij)",
                   "j_start(ij)", "j_end(ij)",
                   ["CALL cap_is('isurban',isurban)",
                    "CALL cap_is('iz0tlnd',iz0tlnd)"],
                   SFC3, SFC2_IN, SFCI)
    after = block("myjsfc", "out", "            ", "i_start(ij)", "i_end(ij)",
                  "j_start(ij)", "j_end(ij)", [], [], SFC2_OUT, [])
    lines = patch(lines, "USE module_sf_myjsfc",
                  r"^\s*CALL MYJSFC\(itimestep,ht,dz8w,",
                  r"i_start\(ij\),i_end\(ij\), j_start\(ij\),j_end\(ij\), kts,kte\s*\)",
                  before, after)
    open(out_sfc, "w").write("\n".join(lines))
    print("patched")


if __name__ == "__main__":
    main()
