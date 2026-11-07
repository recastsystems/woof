# SBU-YLin native WRF oracle

`phys/module_mp_sbu_ylin.F` is the sole scheme source. The WRF v4.7.1
and v4.6.1 files are byte-identical:
`d06a0b8fa01465c09ac218f94e2afc409b4ef3a65cf5782fc79cbe8a9d25e06f`.
No external coefficient table is required.

Sources inspected before porting:

- https://raw.githubusercontent.com/wrf-model/WRF/v4.7.1/phys/module_mp_sbu_ylin.F
- https://raw.githubusercontent.com/wrf-model/WRF/v4.6.1/phys/module_mp_sbu_ylin.F
- https://raw.githubusercontent.com/wrf-model/WRF/v4.7.1/LICENSE.txt

The source has no separate restrictive license. The release public-domain
notice is reproduced in `WRF-LICENSE.txt`, SHA-256
`b30611d35ad1922bc741e86d21e1a897234c1a9c9e2af28076e021f565dfbec6`.

The production target is the unmodified native module. The historical
`declared` output and its patch generator are retained solely as negative
controls. They are never the reference for native equivalence.

Strict synthetic replay:

```
GPUWM_WRF_EXACT=1 python tools/sbu_ylin_wrf461_oracle/validate_sbu_ylin_oracle.py --json strict.json
```

The capture and real-forecast receipts are supplied with the port lane's
review evidence. The 89-column synthetic fixture is a regression test and
must not be presented as real-input forecast proof.

Real-input proof uses the native driver around its unchanged column call:

```
python tools/sbu_ylin_wrf461_oracle/capture_native.py /path/to/public/module_mp_sbu_ylin.F /path/to/own/WRF/phys/module_mp_sbu_ylin.F
# Rebuild the owned native WRF copy with strict FP32 numerics, then run
# a 301 by 301, 49-mass-level, three-hour forecast at a 15-second step.
# Keep its replay-*-in.txt and replay-*-out.txt capture files.
GPUWM_WRF_EXACT=1 python tools/sbu_ylin_wrf461_oracle/replay_native.py /path/to/captures --out native-replay.json
```

The capture tool verifies the public source SHA-256 before editing. It
records hexadecimal FP32 words immediately before and after the native
`clphy1d_ylin` call, at 36 interior locations on calls 1, 241, 481 and 720.
The real replay requires all 144 columns and compares every theta, water,
RIMI, accumulated-rain and per-call-rain output word. Captured files and
the native build/run command belong in the evidence receipt, rather than
in the distributed package.

WRF declares RIMI as persistent, unadvected `state` with input, restart and
history IO. The transported fields are QVAPOR, QCLOUD, QRAIN, QICE and QSNOW.
Stock RRTMG uses its native legacy wrapper. The scheme produces no effective
radii and supplies no native reflectivity; the optional engine reflectivity
uses the source's snow/rimed-snow partition and is an engine diagnostic.

The completed port proof, single-case forecast limits and stage timings are
recorded in [the public receipt](../../docs/public/receipts/sbu-ylin-mp13-20261101.json).
