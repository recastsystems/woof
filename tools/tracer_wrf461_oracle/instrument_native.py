#!/usr/bin/env python3
"""Add stream captures around the unchanged WRF tracer arithmetic.

Apply only to a private copy of the pinned native tree. Captures cover the
first root/nest step's scalar advection and the first tracer diff4d call.
The MPI/tile bounds remain in every file; no native result is overwritten.
"""
from pathlib import Path
import hashlib
import json
import sys


def instrument(root):
    manifest = json.loads(Path(__file__).with_name("source_manifest.json").read_text())
    pins = {row["path"]: row["sha256"] for row in manifest["sources"]}
    receipts = []

    def read(name):
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != pins[name]:
            raise ValueError(f"source pin differs: {name}")
        return data.decode()

    def write(name, text):
        path = root / name
        path.write_text(text)
        receipts.append({"path": name, "original_sha256": pins[name],
                         "instrumented_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    name = "dyn_em/module_advect_em.F"
    text = read(name)
    anchor = "  USE module_wrf_error\n\nCONTAINS"
    assert text.count(anchor) == 1
    text = text.replace(anchor, "  USE module_wrf_error\n"
                        "  INTEGER :: tracer_capture_key = 0\n\nCONTAINS")
    flags = ["periodic_x", "periodic_y", "specified", "nested", "open_xs", "open_xe",
             "open_ys", "open_ye", "symmetric_xs", "symmetric_xe", "symmetric_ys",
             "symmetric_ye", "polar"]
    for mode, routine, first in ((1, "advect_scalar", "   specified = .false."),
                                 (5, "advect_scalar_pd", "    IF (config_flags%polar) THEN")):
        begin = text.index("SUBROUTINE " + routine + " ")
        end = text.index("END SUBROUTINE " + routine, begin)
        section = text[begin:end]
        decl = "   INTEGER :: tracer_capture_unit, tracer_capture_flags\n   CHARACTER(256) :: tracer_capture_file\n"
        # Declarations precede statement functions, which are executable-region delimiters.
        at = section.index("! definition of flux operators")
        section = section[:at] + decl + section[at:]
        capture = "\n   IF (tracer_capture_key > 0) THEN\n"
        capture += "     WRITE(tracer_capture_file,'(A,I9.9,A,I4.4,A,I4.4)') 'tracer-',tracer_capture_key,'-',its,'-',jts\n"
        capture += "     tracer_capture_flags = 0\n"
        for bit, flag in enumerate(flags):
            capture += f"     IF (config_flags%{flag}) tracer_capture_flags = IBSET(tracer_capture_flags,{bit})\n"
        capture += "     OPEN(newunit=tracer_capture_unit,file=TRIM(tracer_capture_file)//'.in.bin', &\n       form='unformatted',access='stream',status='replace')\n"
        capture += f"     WRITE(tracer_capture_unit) 1,{mode},ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme, &\n       its,ite,jts,jte,kts,kte,time_step,config_flags%h_mom_adv_order,config_flags%v_mom_adv_order, &\n       config_flags%h_sca_adv_order,config_flags%v_sca_adv_order,tracer_capture_flags,"
        capture += "MERGE(1,0,tenddec)\n" if mode == 5 else "0\n"
        capture += f"     WRITE(tracer_capture_unit) rdx,rdy,{('dt' if mode == 5 else '1.0')}\n"
        capture += "     WRITE(tracer_capture_unit) field,field_old,tendency,ru,rv,rom,rom\n"
        capture += "     WRITE(tracer_capture_unit) mut," + ("mub,mu_old" if mode == 5 else "mut,mut") + ",msfux,msfuy,msfvx,msfvy,msftx,msfty\n"
        capture += "     WRITE(tracer_capture_unit) c1,c2,fzm,fzp,rdzw,rdzw\n     CLOSE(tracer_capture_unit)\n   END IF\n"
        assert section.count(first) == 1
        section = section.replace(first, capture + first)
        section += "\n   IF (tracer_capture_key > 0) THEN\n     OPEN(newunit=tracer_capture_unit,file=TRIM(tracer_capture_file)//'.out.bin', &\n       form='unformatted',access='stream',status='replace')\n     WRITE(tracer_capture_unit) tendency\n     CLOSE(tracer_capture_unit)\n   END IF\n"
        text = text[:begin] + section + text[end:]
    write(name, text)

    name = "dyn_em/solve_em.F"
    text = read(name)
    text = text.replace("   USE module_em\n", "   USE module_em\n   USE module_advect_em, ONLY: tracer_capture_key\n")
    text = text.replace("         tracer_variable_loop: DO ic = PARAM_FIRST_SCALAR, num_tracer\n",
                        "         tracer_variable_loop: DO ic = PARAM_FIRST_SCALAR, num_tracer\n"
                        "           tracer_capture_key = 0\n"
                        "           IF (grid%itimestep == 1) tracer_capture_key = 100000*grid%id + 100*rk_step + ic\n")
    text = text.replace("       ENDDO tracer_variable_loop\n", "       ENDDO tracer_variable_loop\n       tracer_capture_key = 0\n")
    write(name, text)

    name = "phys/module_pbl_driver.F"
    text = read(name)
    begin = text.lower().index("          subroutine diff4d(")
    end = text.lower().index("      end subroutine diff4d", begin)
    section = text[begin:end]
    first = "      s1d=0."
    at = section.index(first)
    section = section[:at] + "      LOGICAL, SAVE :: tracer_captured = .false.\n      LOGICAL :: tracer_capture_now\n      INTEGER :: tracer_capture_unit\n" + section[at:]
    capture = """      tracer_capture_now = .not. IS_SCALAR .and. .not. tracer_captured
      IF (tracer_capture_now) THEN
        OPEN(newunit=tracer_capture_unit,file='tracer-diff4d.in.bin', &
          form='unformatted',access='stream',status='replace')
        WRITE(tracer_capture_unit) 1,NUM_SCALAR,PARAM_FIRST_SCALAR, &
          IDS,IDE,JDS,JDE,KDS,KDE,IMS,IME,JMS,JME,KMS,KME,ITS,ITE,JTS,JTE,KTS,KTE
        WRITE(tracer_capture_unit) DT,SCALAR,DZ,RHO,EXCH_H,SCALAR_TEND
        CLOSE(tracer_capture_unit)
      END IF
"""
    section = section.replace(first, capture + first)
    section += """      IF (tracer_capture_now) THEN
        OPEN(newunit=tracer_capture_unit,file='tracer-diff4d.out.bin', &
          form='unformatted',access='stream',status='replace')
        WRITE(tracer_capture_unit) SCALAR_TEND
        CLOSE(tracer_capture_unit)
        tracer_captured = .true.
      END IF
"""
    write(name, text[:begin] + section + text[end:])
    (root / "tracer-instrumentation.json").write_text(json.dumps(receipts, indent=2) + "\n")


if __name__ == "__main__":
    instrument(Path(sys.argv[1]))
