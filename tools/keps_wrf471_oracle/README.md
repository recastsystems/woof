# KEPS native WRF 4.7.1 replay

The authority is WRF v4.7.1, commit
`f52c197ed39d12e087d02c50f412d90d418f6186`,
`phys/module_bl_keps.F`, SHA-256
`a79058405affcd293740b15137002e788d168a5eba51586d14cee41342ffc204`.
The release's [public-domain statement](https://github.com/wrf-model/WRF/blob/v4.7.1/LICENSE.txt)
was checked before intake. Its SHA-256 is
`b30611d35ad1922bc741e86d21e1a897234c1a9c9e2af28076e021f565dfbec6`.
This module needs no external coefficient table. The constants asset contains
the module's parameters and the scalar compiler's folded float32 words.
The unmodified notice ships at `woof/data/keps/WRF-NOTICE.txt`.

`instrument.py SOURCE` verifies that pin, then inserts stream writes before
and after the native three-dimensional call. It changes no arithmetic and
does not initialize native undefined variables. Use it on a private source
copy before compiling a full WRF executable. The measured native build uses
GNU Fortran 13.3, `-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`.
The earlier `keps_wrf461_oracle` patched synthetic fixture is historical
evidence and is not this port's numerical authority.

Set `WOOF_KEPS_CAPTURE=/absolute/own/capture/keps` for the native run.
The hook captures a 9 by 8 column tile at steps 1, 2, 3, 4, 60, 120 and
180 of a real 250 by 200 by 49 forecast with 60-second steps. Every capture
contains native entry fields, entry state, and exit words. Replay each call
from its own native entry state:

```sh
GPUWM_WRF_EXACT=1 GPUWM_NUMERICS=strict python tools/keps_wrf471_oracle/replay.py \
  woof/data/keps/wrf471/original/keps-*.bin.xz --receipt original-replay.json
```

The comparator counts every bit in 60,048 float32 words per capture:
five tendencies, three defined turbulent fluxes, two exchange coefficients,
seven state fields, and PBL height. It compares NaN payloads and signed
zeros too. WQ_TUR is never assigned by native WRF and is excluded. The GPU
keeps that diagnostic zero. Native `ttke` and `radsum` are uninitialized;
the GPU uses computed TKE and a zero radiative-sum seed. A difference in
either affected output must be reported, rather than hidden by a tolerance.
The native first-level WV_TUR formula reads U and is preserved.

The seven unmodified captures contain 420,336 compared float32 words,
with 15,372 differences. A single-call Fortran harness using the same
compiler reproduces every original full-run capture byte for byte.
`native_replay.py SOURCE DIRECTORY --defined-undefined` then emits a
diagnostic copy that changes only `ttke` to the computed `tke` and seeds
`radsum` with zero. All seven diagnostic captures replay with zero
differences. Their entry arrays are byte-identical to the original
captures. `woof/data/keps/wrf471/manifest.json` records the source-output
hashes and the precise two changes. Reproduce the paired witness with:

```sh
GPUWM_WRF_EXACT=1 GPUWM_NUMERICS=strict python -m pytest -q \
  tests/test_keps_wrf471_parity.py
```

The Bougeault length search runs only where the native dissipation repair
consumes its result. The condition reads unchanged TE/D words and the
consuming branch retains the complete native search. This removes unused
quadratic work without replacing a physics calculation.

For a bounded stage comparison on the same tiled native columns, leave
strict proof settings unset and run `benchmark.py --receipt timing.json`.
It measures KEPS, MYJ and YSU with CUDA events, three warmups and five
samples at 250 by 200 columns. `--input`, `--nx` and `--ny` are explicit
overrides. The recorded RTX 4090 medians are 2.639872, 1.697888 and
1.690624 ms respectively. This historical KEPS microbenchmark is 1.5548 times MYJ. Its physics-cost
exception is withdrawn: it predates the optimized image and is not a
production forecast measurement. Current qualification uses the same-commit,
interleaved real forecasts described in
[the PBL and land stage account](../../docs/public/pbl-land-stage.md).

Fast numerics remain the runtime default. The strict environment is an
explicit proof setting. The column kernel uses a bounded global arena,
one column per thread, with no host value checks or synchronizations in its
step path. The production loader measured zero per-thread local bytes in
both numerics modes on RTX 4090 with NVRTC 12.9.86.

The lane report records the whole forecast, the initial-history handoff
needed after the shared original input was removed, timing, figures,
observations and any unfinished qualification. Module replay alone does
not establish forecast skill or complete qualification.
