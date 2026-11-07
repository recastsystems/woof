# EEPS synthetic regression fixtures

These files were recovered from the qualified EEPS merge
`522d7639884` in the archived `r290/ports` history. They were produced by
`tools/eeps_wrf461_oracle/build.sh`, using generated columns from
`make_columns.py` and the native Fortran module. They contain no inputs
captured from a real WRF forecast and do not satisfy that acceptance gate.

The original fixture platform was gfortran 13.3.0 and glibc 2.39. The
strict flags are `-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`.
The source and constants pins, official source URLs and public-domain
license receipt are in `tools/eeps_wrf461_oracle/SOURCE.json`.
No external EEPS table is required.

There are 72 columns in eight generated regimes. Set A has 44 levels,
eight calls and a 60 second step. Set B has six calls at 180 seconds and
omits the optional snow and graupel arguments. Set C has four calls at
90 seconds, starts at timestep seven, and uses supplied carried state.
Between calls, a fixed vertical blend stands in for scalar advection.

The fixture hashes are checked by `tests/test_eeps_port.py`.
The oracle adds exactly one initialization, `radsum(j) = 0.` at the native
module's initialization block. Native line 748 accumulates RADSUM without
initializing it. The builder also runs stock-zero and stock-snan versions
to witness that undefined value. The CUDA kernel reports an active
out-of-bounds SZV stencil at native line 734 instead of clamping it.

GPU replay and whole-model tests must be run on the current candidate.
The archived qualification is not evidence for the current fast default,
current driver integration, or a real forecast comparison.
