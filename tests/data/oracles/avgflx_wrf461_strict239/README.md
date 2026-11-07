# Native WRF averaged-flux captures

WRF 4.6.1 commit d66e442fccc04111067e29274c9f9eaccc3cef28,
Ubuntu 24.04, gfortran 13.3, glibc 2.39, 8 MPI processes (4 by 2).
Flags: -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize.
No libmvec calls. Real NCAR CONUS input for 2019-11-26 12 UTC,
425 by 300 by 50 parent staggered points, 100 by 100 by 50 child,
12 km / 4 km, 36 s / 12 s, history 15 / 10 minutes.
Physics: Thompson 8, RRTMG 4/4, MYJ 2, Eta 2, Noah 2, Tiedtke 6.
The input and boundary working copies use dry theta. The child input
was captured from native parent interpolation before its first solve.

Each stream is big-endian: one int32 count followed by nine float32
8 by 8 by 8 blocks in Fortran (i,k,j) order. Blocks are old means
RUM/RVM/WWM, input RU_M/RV_M/WW_M, then new means RUM/RVM/WWM.
They cover counts 0, 1, 24, 49 and the first history reset on the child.
The filename's first number is native ide, distinguishing the two domains.
The serial spans both domains on the first MPI process. Capturing did not
change the module's update expressions.

Commands, capture source changes, source provenance, full-forecast results
and inspection receipts are retained under
`<reports>/WRF-SETTINGS-2026-10-31/wrfs-avgflx/`.
The executable identity is recorded there in `native-toolchain.txt`.
