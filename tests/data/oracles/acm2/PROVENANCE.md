# ACM2 PBL oracle fixtures (WRF v4.6.1, bl_pbl_physics = 7)

Written by `tools/acm2_wrf461_oracle/build.sh` on the sprint CPU box,
2026-10-08, from the pristine WRF v4.6.1 tree (d66e442).

* Source: `phys/module_bl_acm.F`, sha256
  `f18685f7ea091f567c3f37e983f26357dcd7e9ec2eea9a6e70635c506a66d60c`
  (build.sh refuses any other file).
* Preprocessing: WRF's own phys/ rule (sed, cpp -P -nostdinc
  -traditional-cpp with WRF's define set, tools/standard.c, cpp). The
  resulting `module_bl_acm.f90` is byte-identical to the one in the
  combo-sweep referee build (`referee.txt`).
* Compile: WRF's serial gfortran FCOPTIM and FCBASEOPTS
  (`-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`, `toolchain.txt`:
  gfortran 13.3.0, glibc 2.39). A second driver linked against the referee's
  own `phys/module_bl_acm.o` wrote byte-identical fixtures (`referee.txt`).
* libm surface: five `powf` call sites in WRF's object (`libm-report.txt`);
  `(1-ZF/PBL)**P` with `P = 2.0` is folded to a multiply at -O2. An -O0
  build keeps it as `powf(x, 2.0)` and differs in a few bytes
  (`O0-report.txt`), so -O0 is not WRF's program and is evidence only.

| file | contents |
| --- | --- |
| `acm2-l40.in.bin` / `.out.bin` / `.index.json` | 74 columns, 40 levels, DTPBL 60 s, six steps from itimestep 1 |
| `acm2-l60.in.bin` / `.out.bin` / `.index.json` | 74 columns, 60 levels, DTPBL 180 s, four steps from itimestep 41 |
| `fatal.in.bin`, `fatal.stderr.txt` | two columns; WRF stops on column 2 with "RIBX never exceeds RIC" |
| `oracle-sha256sums.txt` | sources, the preprocessed module and every fixture |

Layouts are documented in `tools/acm2_wrf461_oracle/run_acm2.F90`; the
reader is `gpuwm/verify/acm2_oracle.py`. Machine path prefixes in the
receipts are replaced by `<wrf-4.6.1>`, `<oracle-tools>` and
`<referee-build>`; every digest is the byte the box computed.
