# Native EM passive tracer replay

The native pin is WRF v4.6.1, commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`. The public source URLs and
SHA-256 values are in `source_manifest.json`. `NOTICE` retains WRF's
public-domain notice. No competitor implementation or external table is used.
The Registry.EM package and scalar advection source are byte-identical at
the audited WRF v4.7.1 pin `f52c197ed39d`.

`instrument_native.py PRIVATE_WRF_COPY` checks the original source hashes,
then inserts stream capture statements around unchanged native arithmetic.
Use a strict build with `-fno-fast-math -ffp-contract=off
-fno-tree-vectorize`, one MPI rank and one tile. The first native step
captures all eight tracers at three RK stages on each active domain. The
first non-scalar `diff4d` call captures tracer PBL diffusion. These are real
native forecast inputs, including native halos and incoming tendencies.

Run the instrumented `real.exe` and `wrf.exe` from a stock prepared case,
with `tracer_opt=2`, `tracer_adv_opt=1`, and `tracer_pblmix=1`. Then:

```
GPUWM_WRF_EXACT=1 GPUWM_WRF_GLIBC=2.43 python \
  tools/tracer_wrf461_oracle/replay.py CASE RECEIPT.json
```

The replay calls production CUDA scalar transport and column diffusion.
The receipt records hashes, bounds, word counts and differences for every
capture. Native halo and excluded boundary words are checked separately
from the physical comparison region. Keep the actual native capture inputs and reference outputs permanently, with
a MANIFEST.md containing commands, native and engine commits, hashes and sizes.
A receipt without the corresponding native bytes cannot support an exact-match
claim. `benchmark.py RECEIPT.json` times the same production
operators under fast numerics on the forecast's parent grid dimensions.
Timing synchronizes once after each event batch, outside production stepping.

The lane receipt records 18,961,536 compared words with zero differences,
plus 873,660 untouched guard words. This is a module replay, not a claim
that the entire forecast matches native WRF bit for bit. It covers ordinary
and positive-definite advection and the first parent tracer PBL solve.
It does not qualify every diffusion, PBL or boundary combination.

Runtime integration is described in `docs/public/PHYSICS.md`. `traj_opt` has its own [trajectory oracle](../trajectory_wrf461_oracle/README.md).
