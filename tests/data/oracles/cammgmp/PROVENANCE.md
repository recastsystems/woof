# CAMMGMP (mp_physics = 11) column-oracle fixtures

Written by `tools/cammgmp_wrf461_oracle/build.sh` on the sprint port box
(2026-10-08): gfortran 13.3.0, glibc 2.39, x86-64 (AMD EPYC 9R14), WRF v4.6.1
d66e442 (noahmp 848f54a), linked against the SCALAR build's own
`main/libwrflib.a` (FCOPTIM `-O2 -fno-fast-math -ffp-contract=off
-fno-tree-vectorize`), every CAM source byte-identical to the pinned tree.

| file | content |
|---|---|
| `cases.bin` | the input deck, `make_cases.py` (88 columns, 5 groups, 3-4 steps) |
| `pristine-out.bin` | every step's CAMMGMP outputs from the unmodified archive |
| `fixed-out.bin` | the same from the archive with `module_cam_mp_ndrop.F` replaced by the declared cloud-borne correction (`fix_ndrop.py`), compiled with the archive's flags |
| `helpers.bin` | WRF's estblf, polysvp, CAM erf and MG GAMMA on argument grids, and the modal-aerosol tables CAM_INIT published |
| `words.txt` | the CAM modules' run-time words read out of the linked objects after CAM_INIT and CAMMGMP_INIT (`dump_words.c`) |
| `toolchain.txt`, `wrf-commit.txt` | the build's identity |

`tests/test_cammgmp_wrf461_parity.py` pins these bytes by sha256.

This is a recovered generated-input diagnostic deck. Its changed-ndrop file
is an altered-source attribution instrument only. The stock column kernel
uses pristine indexing and WRF's imported error_function module. No record
here is a real-forecast capture or forecast qualification.
