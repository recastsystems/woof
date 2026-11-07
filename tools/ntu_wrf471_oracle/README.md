# NTU native source and real-input qualification

`source.json` pins the official WRF v4.7.1 module and its public-domain
statement. Its in-file AERO_CONST tables need no external dataset. No
competitor source is used. The native arithmetic block has no source
corrections: the recovered prototype's F1–F6 alterations are removed.

## Reproduce the source and table translation

```sh
python tools/ntu_wrf471_oracle/generate.py module_mp_ntu.F LICENSE.txt build/ntu
python tools/ntu_wrf471_oracle/export_aero.py --help
python tools/ntu_wrf471_oracle/parallelize.py --out build/ntu_parallel.cuh
```

The generators check the source and licence hashes before using them. The
canonical generated block has SHA256
`cff4807966d0ee63217616278536bedb58a0fa60446238f0327af95cd29c1e0b`.
`parallelize.py` preserves its arithmetic while mapping independent levels
to GPU threads. The ordered sedimentation recurrence uses one thread per
column. Three launches share 28 device work volumes. All physics stays on
the device; tables and pointer tables are cached, with no per-step host
status read. Fast CUDA mathematics is the default. Strict mathematics is a
separate proof arm, including the native compiler's promoted LOG10.

## Real native replay

`capture.py` inserts observation-only input/return writers into a copied
native tree. The reference forecast used a strict WRF 4.6.1 build with the
unmodified WRF 4.7.1 NTU module; it is not a whole WRF 4.7.1 comparison.
The public January 2000 real-data case ran 12–15 UTC on 24 January, with
73 × 60 × 34 mass cells, 30 km spacing and a 60 s time step. MP56 used
CCNTY=2, YSU, classic MM5, Noah and legacy RRTMG.

```sh
python tools/ntu_wrf471_oracle/capture.py --help
GPUWM_NUMERICS=strict PYTHONPATH=. python tools/ntu_wrf471_oracle/replay.py \
  native-first.in native-first.out build/ntu/ntu_gen.cuh replay.json
PYTHONPATH=. python tools/ntu_wrf471_oracle/lifecycle.py \
  native-first.in build/lifecycle lifecycle.json
```

The first native call's captured tile contains 71 × 29 × 34 cells. The
strict shipping three-stage GPU layout compared all 29 mass-grid outputs
and nine surface outputs: **0 differing words of 2,048,705**, including
signed zeros, with no nonfinite output. Input SHA256:
`2bc7f68616ea9252773e5d66d156e09e465a35c0da46770951bc165aa7fc9be2`.
Native return SHA256:
`1141ffb1559493d9b16b7fa85fd2abe6878b0818560bc661e141dc3a42e4e7bc`.

This is one real module call, not a claim that the two forecast systems
have identical dynamics. A completed three-hour `woof go --wrfinput`
forecast was rendered with `woof.rustwx`, beside native WRF and real ASOS
temperatures at four hourly times. The lane report carries the figures,
commands, observation method and numerical differences. The corrected run
from the installed package wrote three hourly checkpoints. An installed
`woof go --restart` continuation from hour two reproduced all 224 final
history fields, including 7,044,496 stored float32 words, with zero
differences. `qualification.json` records these results and the hashes of
the retained paper evidence.

## Native undefined reads

The standalone signaling-NaN diagnostic (`build.py`, `run.py`) witnesses
an unfilled top density read at module_mp_ntu.F:1618: NTU_INIT fills local
RHO through KTE-1, then INIT_AEROSOL reads it through KTE. The engine cold
start supplies the analytically evaluated top density. The diagnostic is
analytic; it does not replace the real-input replay above.

A signaling-NaN replay of the real step-60 capture separately traps at
module_mp_ntu.F:5695, `RATIO=RHOGW/DNGRM`, where DNGRM is uninitialized.
A fresh standalone native invocation differs from the original native
forecast's return at 1,399 words on that input. The trap witnesses an
undefined read; it does not identify the cause of every differing word.
No wet-graupel parity claim is made for that call.

## Reach and limits

Stock `mp_physics=56` and `ccnty=1..4` reach the device port. All 28 moist
and scalar carriers are transported, written to history, restored through
restart/checkpoint, and included in same-scheme nests and ensemble member
slabs. Both nest feedback modes have GPU carrier checks. `lifecycle.py`
uses the actual real-capture physics to check restart continuation and
two independent member slabs; it is not a second dycore forecast.

WRF NTU supplies no REFL_10CM; the engine's Rayleigh diagnostic is its own
moment-based product. RTE+RRTMGP is refused because qualified
moment-derived effective radii are missing. Mixed NTU/other-scheme nests
are refused because their aerosol/shape transition closure is missing.
Same-scheme NTU nests and native legacy-RRTMG coupling are admitted.

On the captured 70,006-cell tile the optimized default takes 66.8 ms on an
RTX 4090, versus 0.189 ms for the nearest existing multi-moment MY2 port.
The 1.2× bar is exceeded. Native operation counters record 2,685,059
GAMLN evaluations and 274,625 adaptive microsteps; the stiffest cell needs
397 sequential iterations. The lane evidence measures the critical cell
separately (40.0 ms on the same RTX 4090), profiles all three stages, and records rejected compiler and
storage alternatives. No adaptive iteration is capped or omitted.

The speed audit is in [SPEED.md](SPEED.md). `ntu-mp56` is opt-in only,
with its measured cost stated; it is never a default scheme.
