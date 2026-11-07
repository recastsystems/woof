#!/usr/bin/env python3
"""Capture bounded raw NWP inputs and native outputs in an owned WRF build.

The two first calls on each MPI patch retain 4x4 horizontal columns and all
vertical levels. Original expressions are unchanged. Run only on one domain
to keep the patch-coordinate filenames unique.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = ("wspd10max", "up_heli_max", "w_up_max", "w_dn_max", "w_mean",
          "grpl_max", "refd_max", "hail_maxk1", "hail_max2d", "w_colmean",
          "numcolpts", "grpl_colint")
OPTIONAL = ("qh_curr", "ng_curr", "nh_curr", "qvolg_curr", "qr_curr", "nr_curr")
START = "    idump = (history_interval * 60.) / dt"
END = "   END SUBROUTINE diagnostic_output_nwp"
DECLARATIONS = """    ! Input/output capture only; native expressions below are unchanged.
    INTEGER, SAVE :: gpuwm_capture_count = 0
    INTEGER :: gpuwm_unit, gpuwm_status, gpuwm_is, gpuwm_ie, gpuwm_js, gpuwm_je
    CHARACTER(LEN=2048) :: gpuwm_directory, gpuwm_filename
    LOGICAL :: gpuwm_capture_active

"""


def writes(fields, dims):
    return "".join(f"       WRITE(gpuwm_unit) {field}({dims})\n" for field in fields)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkout", type=Path)
    p.add_argument("--receipt", required=True, type=Path)
    args = p.parse_args()
    target = args.checkout.resolve() / "phys/module_diag_nwp.F"
    manifest = json.loads((HERE / "sources.json").read_text())
    row = next(r for r in manifest["sources"] if r["path"] == "phys/module_diag_nwp.F")
    original = target.read_bytes()
    if hashlib.sha256(original).hexdigest() != row["sha256"]:
        raise ValueError("refusing to instrument a source outside the native pin")
    text = original.decode()
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("native insertion points are not unique")
    xy = "gpuwm_is:gpuwm_ie,gpuwm_js:gpuwm_je"
    xyz = "gpuwm_is:gpuwm_ie,kms:kme,gpuwm_js:gpuwm_je"
    mass = "gpuwm_is:gpuwm_ie,kms:kme-1,gpuwm_js:gpuwm_je"
    capture = """    gpuwm_capture_active = .FALSE.
    CALL GET_ENVIRONMENT_VARIABLE('WOOF_DIAG_CAPTURE_DIR', gpuwm_directory, STATUS=gpuwm_status)
    IF (gpuwm_status == 0 .AND. LEN_TRIM(gpuwm_directory) > 0 .AND. gpuwm_capture_count < 2) THEN
       gpuwm_capture_active = .TRUE.
       gpuwm_capture_count = gpuwm_capture_count + 1
       gpuwm_is = i_start(1)
       gpuwm_js = j_start(1)
       gpuwm_ie = MIN(gpuwm_is+3,i_end(1),ide-1)
       gpuwm_je = MIN(gpuwm_js+3,j_end(1),jde-1)
       WRITE(gpuwm_filename,'(A,"/nwp-",I6.6,"-",I6.6,"-",I2.2,".bin")') &
           TRIM(gpuwm_directory),gpuwm_is,gpuwm_js,gpuwm_capture_count
       OPEN(NEWUNIT=gpuwm_unit,FILE=TRIM(gpuwm_filename),ACCESS='STREAM', &
            FORM='UNFORMATTED',STATUS='NEW',CONVERT='LITTLE_ENDIAN')
       WRITE(gpuwm_unit) 47102,gpuwm_ie-gpuwm_is+1,gpuwm_je-gpuwm_js+1,kme-kms, &
           mphysics_opt,history_interval,itimestep,MERGE(1,0,adaptive_ts),max_time_step, &
           gsfcgce_hail,gsfcgce_2ice,mpuse_hail,config_flags%nssl_2moment_on,config_flags%nssl_hail_on, &
           MERGE(1,0,PRESENT(qh_curr)),MERGE(1,0,PRESENT(ng_curr)),MERGE(1,0,PRESENT(nh_curr)), &
           MERGE(1,0,PRESENT(qvolg_curr)),MERGE(1,0,PRESENT(qr_curr)),MERGE(1,0,PRESENT(nr_curr)), &
           gpuwm_is,gpuwm_ie,gpuwm_js,gpuwm_je,kms,kme
       WRITE(gpuwm_unit) dt,xtime,g,nssl_cnoh,nssl_cnohl,nssl_rho_qh,nssl_rho_qhl,nssl_alphah,nssl_alphahl
"""
    for field in ("w", "p8w", "qg_curr", "rho", "ph", "phb", "refl_10cm"):
        capture += writes((field,), xyz if field in ("w", "p8w", "ph", "phb") else mass)
    capture += "       WRITE(gpuwm_unit) znw(kms:kme)\n" + writes(("u10", "v10"), xy)
    for field in OPTIONAL:
        capture += f"       IF (PRESENT({field})) THEN\n" + writes((field,), mass) + "       END IF\n"
    capture += writes(FIELDS, xy) + "    END IF\n\n"
    finish = "    IF (gpuwm_capture_active) THEN\n" + writes(FIELDS, xy)
    finish += "       CLOSE(gpuwm_unit)\n    END IF\n\n"
    changed = text.replace(START, DECLARATIONS + capture + START).replace(END, finish + END).encode()
    target.write_bytes(changed)
    args.receipt.write_text(json.dumps({
        "wrf": manifest["wrf"], "original_sha256": row["sha256"],
        "instrumented_sha256": hashlib.sha256(changed).hexdigest(),
        "instrumenter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": str(target), "kind": "raw native NWP entry and output; no arithmetic changes",
        "limit": "two calls per MPI patch; first tile; at most 4x4 columns; all levels",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
