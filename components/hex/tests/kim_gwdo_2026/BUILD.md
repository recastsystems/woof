# Native KIM GWDO oracle (test-only)

Sources, all fetched read-only:
- `bl_gwdo.F90`: NCAR/MMM-physics tag `20260319-WRFv4.8.0` (63c806deef6a), sha256 d6211b83...12f0 (BSD-3-Clause).
- `module_bl_gwdo.F`: MPAS-A v8.4.2 `src/core_atmosphere/physics/physics_wrf/module_bl_gwdo.F` (sha256 9784f201...52e6)
  with exactly the MPAS-Model PR #1409 hunk in `module_bl_gwdo.pr1409.diff` applied (result sha256 3b11d30d...777b).
- `kinds.F90`: `../ysu_gwdo_v842/kinds.F90`. `oracle.F90`: this folder (C binding of the driver_gwdo bl_kim_gwdo slice).

GNU strict recipe (gfortran 15.2, the pinned MPAS strict flags):

    F="-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -fPIC"
    gfortran $F -c kinds.F90 && gfortran $F -c bl_gwdo.F90 && gfortran $F -ffree-form -c module_bl_gwdo.F
    gfortran $F -c oracle.F90 && gfortran -shared -o libkim_oracle.so kinds.o bl_gwdo.o module_bl_gwdo.o oracle.o

Then `KIM_GWDO_ORACLE=$PWD/libkim_oracle.so WOOF_HEX_NUMERICS=strict pytest tests/test_kim_gwdo_2026.py` on a GPU.
