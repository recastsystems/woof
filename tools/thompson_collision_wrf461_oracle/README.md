# Classic Thompson collision initialization

The GPU setup quadrature ports public WRF v4.6.1 `qr_acr_qg`, native lines
4186–4291, source pin `d66e442fccc04111067e29274c9f9eaccc3cef28`.
WRF distributes this source under its public-domain notice. The source SHA-256
is checked before instrumentation. No extra notice occurs in this module.

`build.sh WRF_PHYS_DIR NATIVE_TABLE_DIR EMPTY_BUILD_DIR` compiles the original
native initialization with an appended printing procedure and checks the exact
words in `seeds-wrf461.txt`. `make_header.py seeds-wrf461.txt OUTPUT.cuh`
reproduces the device constant header. Use the repository's oracle compiler/libm
pin (gfortran 13.3, glibc 2.39, scalar math, contraction disabled) when replaying.
The seed dumper calls only unmodified native initialization.

A changed classic graupel coefficient regenerates all five 37^4 collision tables
on the GPU once during setup, using the native 100 by 100 quadrature and ordered
FP64 accumulations. It is cached by coefficient words, native pin, algorithm,
quadrature and CUDA source identity. Rain collision velocity uses the native
fixed polynomial, independently of the rain sedimentation coefficient.

Native WRF 4.6.1 allocates one classic density slab and later indexes it with 5.
The existing `thompson_wrf461_oracle/active_collision_fixture.py` documents the
reference repair of its eight reads to `MIN(idx_bg(k),dimNRHG)`. The GPU reads
the one classic slab. Real collision-table replay compares every FP64 word,
and the column replay uses this documented native reference repair.
