Selector 31 replay from native model inputs

The pinned 4.6.1 forcing files are byte-identical in WRF 4.7.1. Their
SHA-256 hashes and source licence are in `licenses/NOTICE-WRF-HS31.txt`.
No competitor source or coefficient tables are used.

Copy an existing strict native build before instrumenting. Then run:

```
python3 tools/relaxation_wrf471_oracle/capture_native.py "$PRIVATE_NATIVE"
cd "$PRIVATE_NATIVE"
./compile -j 2 em_heldsuarez
```

Use the stock `test/em_heldsuarez/namelist.input`, with a bounded three-hour
duration and only initial/end history frames. Run `ideal.exe`, then `wrf.exe`
with one MPI rank and `WOOF_HS_CAPTURE` naming an existing capture directory.
The hooks capture HSRAD call 2 and momentum call 7, preserving all original
expressions and loop bounds. These calls contain evolved native model inputs.

After checking free memory on a shared GPU:

```
GPUWM_NUMERICS=strict PYTHONPATH=. python3 \
  tools/relaxation_wrf471_oracle/replay.py "$CAPTURE" "$RECEIPT"
```

For the stock 64 x 36 x 40 mass grid, the replay compares 92,160 temperature
rate words, 93,600 u-tendency words and 94,720 v-tendency words. The last
count includes both untouched polar faces. The v replay uses the production
masked kernel: its persistent staggered mask also carries physical boundaries
through tilestream windows, without suppressing internal window faces. Exact comparison includes signed
zeros and NaN payloads. No tolerance is applied. The receipt pins each raw
capture by its hash and bounds, and pins the assembled kernel via the shared
manifest. Retain losslessly compressed captures, their hashes, the receipt and
run logs; decompress into private scratch when replaying, then remove that
regenerable copy.

This replay proves the forcing operators. The additional native-input
global proofs are `replay_initializer`, `replay_polar`, `replay_pole`,
`replay_hydrostatic`, `replay_hydrostatic_acoustic`, `replay_hydrostatic_w`,
`replay_rotation`, `replay_advection`, `replay_transport`, `replay_geometry`
and `replay_physics_prepare`. Run each as a module under
`GPUWM_NUMERICS=strict`, with its retained native binary capture (or capture
directory and schema) and a receipt path. `--help` gives each exact command.
The initializer also accepts `--undefined-witness`: native local surface
weights are uninitialized, witnessed by a signalling-NaN native rebuild.
Defined output words must all match; the two affected inactive skin fields
are explicitly separated. Production uses the defined grid weights.

The lane evidence retains the non-arithmetic native taps, their source
hashes, raw capture hashes, strict compiler settings and native run logs.
The captures come from evolved native WRF calls, not generated test profiles.
Each proof calls the production CUDA operator and compares unsigned words.
These module proofs do not claim whole-model bitwise parity. Whole native
and public `woof go` forecasts, hourly Rust map inspection, actual restart
continuation, concurrent-member isolation and CUDA-event stage timings are
recorded separately. The public global runtime is the stock dry single-domain
hydrostatic case; regional selector-31 nests exercise both feedback modes.
