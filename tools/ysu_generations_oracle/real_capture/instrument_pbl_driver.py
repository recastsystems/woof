"""Instrument a COPY of a native WRF 4.8.0 build so it records YSU columns.

    instrument_pbl_driver.py WRF_COPY/phys/module_pbl_driver.F STEPS

Inserts, around the one ``CALL ysu(`` in the YSUSCHEME branch, writes of
every column's arguments (before the call) and every word the call returned
(after it), at the listed time steps, in the record layout of
tools/ysu_wrf461_oracle (make_wide_columns.write_records for inputs,
run_bl_ysu_wide.F90 for outputs).  Only reads are added: no argument and no
YSU arithmetic changes, and bl_ysu.o is not recompiled (receipt checks its
sha256 before and after).  The referee binary itself is never edited.

Output files in the run directory, per captured step N:
  ysucap-stepN-in.bin   inputs (hdr nz/topdown/regime=0/topo=0, 18 scalars,
                        u v theta pi tx qv qc qi p dz rthraten, p8w[nz+1])
  ysucap-stepN-out.bin  kpbl; hpbl wstar delta u10 v10; utnp vtnp ttnp(=0,
                        not returned by the driver) rthblten qvtnp qctnp
                        qitnp exch_h exch_m qmixtnp(=0, no tracer arm)
"""
import sys
from pathlib import Path

path = Path(sys.argv[1])
steps = [int(s) for s in sys.argv[2].split(",")]
src = path.read_text()
assert "ysucap_u" not in src, "already instrumented"

decl_anchor = "   IMPLICIT NONE\n"
assert src.count(decl_anchor) >= 1
first = src.index(decl_anchor) + len(decl_anchor)
src = src[:first] + ("   INTEGER :: ysucap_u, ysucap_k\n"
                     "   CHARACTER(LEN=64) :: ysucap_name\n"
                     "   REAL :: ysucap_ws, ysucap_de\n") + src[first:]

call = "             CALL ysu(                                              &\n"
assert src.count(call) == 1, src.count(call)
steplist = ",".join(str(s) for s in steps)
pre = f"""             IF ( ANY( itimestep == (/{steplist}/) ) ) THEN
               WRITE(ysucap_name,'(A,I0,A)') 'ysucap-step', itimestep, '-in.bin'
               OPEN(NEWUNIT=ysucap_u, FILE=TRIM(ysucap_name), ACCESS='stream', &
                    FORM='unformatted', STATUS='replace', CONVERT='little_endian')
               DO j = jts, jte
               DO i = its, ite
                 WRITE(ysucap_u) kte-kts+1, ysu_topdown_pblmix, 0, 0
                 WRITE(ysucap_u) dtbl, psfc(i,j), znt(i,j), ust(i,j), hfx(i,j), qfx(i,j), &
                      wspd(i,j), br(i,j), fm(i,j), fhh(i,j), xland(i,j), u10(i,j), v10(i,j), &
                      ctopo(i,j), ctopo2(i,j), uoce(i,j), voce(i,j), 0.0
                 WRITE(ysucap_u) (u_phytmp(i,ysucap_k,j), ysucap_k=kts,kte), &
                      (v_phytmp(i,ysucap_k,j), ysucap_k=kts,kte), &
                      (th_phy(i,ysucap_k,j), ysucap_k=kts,kte), &
                      (pi_phy(i,ysucap_k,j), ysucap_k=kts,kte), &
                      (t_phy(i,ysucap_k,j), ysucap_k=kts,kte), &
                      (qv_curr(i,ysucap_k,j), ysucap_k=kts,kte), &
                      (qc_curr(i,ysucap_k,j), ysucap_k=kts,kte), &
                      (qi_curr(i,ysucap_k,j), ysucap_k=kts,kte), &
                      (p_phy(i,ysucap_k,j), ysucap_k=kts,kte), &
                      (dz8w(i,ysucap_k,j), ysucap_k=kts,kte), &
                      (rthraten(i,ysucap_k,j), ysucap_k=kts,kte)
                 WRITE(ysucap_u) (p8w(i,ysucap_k,j), ysucap_k=kts,kte+1)
               ENDDO
               ENDDO
               CLOSE(ysucap_u)
             ENDIF
"""
src = src.replace(call, pre + call)
# The statement after the call's closing paren: the FASDAS comment block.
tail_anchor = "                                                                   )\n!\n! FASDAS\n"
assert src.count(tail_anchor) >= 1
pos = src.index(tail_anchor, src.index(pre)) + len("                                                                   )\n")
post = f"""             IF ( ANY( itimestep == (/{steplist}/) ) ) THEN
               WRITE(ysucap_name,'(A,I0,A)') 'ysucap-step', itimestep, '-out.bin'
               OPEN(NEWUNIT=ysucap_u, FILE=TRIM(ysucap_name), ACCESS='stream', &
                    FORM='unformatted', STATUS='replace', CONVERT='little_endian')
               DO j = jts, jte
               DO i = its, ite
                 ysucap_ws = 0.0
                 ysucap_de = 0.0
                 IF (PRESENT(wstar)) ysucap_ws = wstar(i,j)
                 IF (PRESENT(delta)) ysucap_de = delta(i,j)
                 WRITE(ysucap_u) kpbl(i,j)
                 WRITE(ysucap_u) pblh(i,j), ysucap_ws, ysucap_de, u10(i,j), v10(i,j)
                 WRITE(ysucap_u) (rublten(i,ysucap_k,j), ysucap_k=kts,kte), &
                      (rvblten(i,ysucap_k,j), ysucap_k=kts,kte), &
                      (0.0, ysucap_k=kts,kte), &
                      (rthblten(i,ysucap_k,j), ysucap_k=kts,kte), &
                      (rqvblten(i,ysucap_k,j), ysucap_k=kts,kte), &
                      (rqcblten(i,ysucap_k,j), ysucap_k=kts,kte), &
                      (rqiblten(i,ysucap_k,j), ysucap_k=kts,kte), &
                      (exch_h(i,ysucap_k,j), ysucap_k=kts,kte), &
                      (exch_m(i,ysucap_k,j), ysucap_k=kts,kte), &
                      (0.0, ysucap_k=kts,kte)
               ENDDO
               ENDDO
               CLOSE(ysucap_u)
             ENDIF
"""
src = src[:pos] + post + src[pos:]
path.write_text(src)
print("instrumented", path, "steps", steps)
