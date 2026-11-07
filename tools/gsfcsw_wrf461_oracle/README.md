# Original GSFC shortwave oracle

This oracle uses WRF's own `phys/module_ra_gsfcsw.F`. The v4.7.1 source
(commit `f52c197ed39d12e087d02c50f412d90d418f6186`) and v4.6.1 source have
SHA-256 `da257cf060c385c3aa4d6c3a737ec8c10773bd52034786d954bab0a0dc290766`.
WRF distributes this file under its public-domain statement. All shipped
coefficient words come from DATA statements in that module. There are no
external optical tables in this option. Source and licence identities are
also recorded in the installed coefficient manifest.

Run `bash tools/gsfcsw_wrf461_oracle/build.sh /path/to/clean/WRF /path/to/build`
to build the strict native library. An exported exact module file is
accepted as the first argument too. The build records compiler, libc,
source, wrapper and library identities as text files in that build directory. The fixture
receipt records those identities and the resulting fixture hash. The
recording tool preserves all existing fixture input words when refreshing
native outputs for a different compiler or libc.

`generate.py` transcribes the public WRF source through f2c and moves
automatic column arrays into a disjoint device workspace. `arena.py`
retains shifted array origins and coalesces adjacent columns in 32-column
tiles. This changes storage, while retaining the source's scalar
operations. The device log10 helper computes a double logarithm and rounds
to float to match the witnessed native float results; no external libm
implementation is included.

`real_replay.py` reads the bounded binary capture format emitted by the
lane's native-run instrumentation. It passes the incoming RTHRATEN field
through, checks all five defined output fields by integer word comparison,
and excludes unused padding slots. For the 2026-11-01 proof, four 64-column,
32-layer native tiles covered night and daylight. All 25,088 compared
output words matched under `GPUWM_NUMERICS=strict`, with no undefined
residue invoked. Capture hashes and native settings belong in the replay
receipt. Raw captures are disposable after that receipt is retained.

The production adapter supports 2 to 128 layers. Its non-chemistry scope,
adapter timing and forecast evidence are described in
`docs/public/PHYSICS.md`. A matching column replay does not establish
whole-forecast bitwise parity.

The default fast image (`gsfcsw_fast.cuh`) is generated alongside the strict
translation. It exposes the packed entry's m=n=ndim=1 dimensions, scalarizes
five one-column temporaries and fuses independent per-level flux passes.
Every spectral term and cloudy-sky case remains. Under the same production
compiler with contraction disabled, the specialization matches all 11,008
defined fixture output words. The strict production image itself remains
byte-identical to the real captured-input replay image.
