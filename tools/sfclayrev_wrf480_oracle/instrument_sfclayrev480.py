"""Capture native WRF v4.8.0 SFCLAYREV wrapper inputs and outputs (kim_tofd).

Patches phys/module_surface_driver.F of a COPY of the strict build (never the
referee tree): around the non-fractional CALL SFCLAYREV it writes, at the
listed ITIMESTEPs, one little-endian stream file before and one after the call.
The scheme objects (sf_sfclayrev.o, module_sf_sfclayrev.o) are not touched.

  python instrument_sfclayrev480.py WRF_TREE
Header: nx ny itimestep isfflx isftcflx iz0tlnd kim_tofd (7 x int32), tofd_factor (float32)
in : 22 planes  u v t qv p dz8w psfc tsk pblh mavail xland lakemask dx znt ust ustm mol hfx qfx qsfc zol var2d
out: 34 planes  woof SFCLAY_OUTPUTS order
"""
import pathlib
import sys

p = pathlib.Path(sys.argv[1]) / "phys/module_surface_driver.F"
s = p.read_text()
STEPS = "(/1,2,3,15,30/)"
IN = ["u_phytmp(its:ite,kts,jts:jte)", "v_phytmp(its:ite,kts,jts:jte)", "t_phy(its:ite,kts,jts:jte)",
      "qv_curr(its:ite,kts,jts:jte)", "p_phy(its:ite,kts,jts:jte)", "dz8w(its:ite,kts,jts:jte)",
      "psfc", "tsk", "pblh", "mavail", "xland", "lakemask", "dx2d", "znt", "ust", "ustm", "mol",
      "hfx", "qfx", "qsfc", "zol", "var2d"]
OUT = ("znt ust ustm mol hfx qfx qsfc zol regime psim psih fm fhh lh u10 v10 th2 t2 q2 chs chs2 "
       "cqs2 flhc flqc qgh rmol wspd br gz1oz0 cpm ck cka cd cda").split()


def block(tag, names):
    two = lambda n: n if "(" in n else n + "(its:ite,jts:jte)"
    lines = [
        f"         IF (ANY(ITIMESTEP == {STEPS})) THEN",
        "          its_c = i_start(ij); ite_c = i_end(ij); jts_c = j_start(ij); jte_c = j_end(ij)",
        f"          WRITE(cap_file,'(A,I0,A,I0,A,I0,A)') 'captures/sfclayrev-',ITIMESTEP,'-',its_c,'-',jts_c,'-{tag}.bin'",
        "          OPEN(newunit=cap_unit,file=trim(cap_file),access='stream',form='unformatted',status='replace',convert='little_endian')",
        "          WRITE(cap_unit) ite_c-its_c+1,jte_c-jts_c+1,itimestep,isfflx,isftcflx,iz0tlnd,merge(1,0,kim_tofd)",
        "          WRITE(cap_unit) tofd_factor",
    ]
    for n in names:
        lines.append("          WRITE(cap_unit) " + two(n).replace("its:ite", "its_c:ite_c").replace("jts:jte", "jts_c:jte_c"))
    lines += ["          CLOSE(cap_unit)", "         ENDIF"]
    return "\n".join(lines) + "\n"


call = "         CALL SFCLAYREV(u_phytmp,v_phytmp,t_phy,qv_curr,&\n"
assert s.count(call) == 1, s.count(call)
end = "               scm_force_flux,errmsg,errflg                        ) \n"
i = s.index(call)
j = s.index(end, i) + len(end)
s = s[:i] + block("in", IN) + s[i:j] + block("out", OUT) + s[j:]
# declarations: after the subroutine's IMPLICIT NONE of surface_driver
sub = s.index("SUBROUTINE surface_driver")
imp = s.index("IMPLICIT NONE", sub)
eol = s.index("\n", imp) + 1
s = s[:eol] + "   INTEGER :: cap_unit, its_c, ite_c, jts_c, jte_c\n   CHARACTER(len=160) :: cap_file\n" + s[eol:]
p.write_text(s)
print("instrumented", p)
