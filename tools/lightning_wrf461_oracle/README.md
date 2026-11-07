# Native lightning oracles

Reference: public-domain WRF v4.6.1, commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`, and its
[license](https://github.com/wrf-model/WRF/blob/v4.6.1/LICENSE.txt).
The lightning modules and driver are byte identical in WRF v4.7.1.
No external tables or competitor source are used.

`build.sh` and `validate_lightning.py` compile the original native modules
and exercise prescribed column and patch inputs. The strict 360-row
check compares 66,960 words. This is constructed-input evidence.

For a real native forecast, first make a private copy of WRF's source.
The capture tool checks the original driver's SHA-256 before changing it:

```sh
python tools/lightning_wrf461_oracle/capture_wrf461.py \
  --wrf-source PRIVATE_WRF_COPY --patch instrumented-native.patch
```

Build that copy with strict native numerics, including
`-O0 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations`.
Run an actual forecast with `lightning_option=3`. After the first active
call at or beyond one forecast hour, the monitor rank writes
`lane-real-lightning.bin`. The capture contains a patch around that
rank's LPI maximum, the native memory-level inputs and its LPI output.
It does not change the calculation or insert constructed inputs.

```sh
GPUWM_WRF_EXACT=1 python tools/lightning_wrf461_oracle/replay_capture.py \
  lane-real-lightning.bin --out lightning-real.json
```

The lane's 2026-10-16 09 UTC case, 300 by 300 columns, 49 mass levels,
3 km spacing, 15 s step and MP6, captured 256 LPI words at 3,645 s.
All 256 matched bit for bit; 253 were nonzero and the maximum was
0.0533673093. The input stream SHA-256 was
`e38d7f04f1a311fdc355e539535af52e22c4dc44be216cb0ab35553804addc80`.
The JSON receipt records the strict CUDA compiler options and image hash.
Raw inputs are deleted after the receipt is recorded.

That original real capture proves only option 3's column-local LPI output.
Activation-clock values are recorded separately and are not counted as
compared words. It does not establish a real-input replay for options
1, 2 or 11, forecast agreement, observations, nests or ensembles. The continuation
below supplies separate evidence for those paths. Fast
numerics are the default; strict numerics are the proof opt-in.

The 2026-11-01 continuation adds `capture_native_all.py` and
`replay_native_all.py` for complete real driver inputs and the activation clock.
The capture tool requires a pristine owned native tree and checks the public
licence before modifying it. Format 47104 captures only the allocated first
IC:CG input plane; legacy 47103 receipts also contain unused native residue.
`iccg_method=4` is held: Registry allocates two-dimensional inputs while the
native routine reads twelve monthly planes. Admitted methods do not read them.
Use `--wrf-source TREE --receipt JSON` to instrument, build the whole native WRF,
run with `WOOF_LTNG_CAPTURE_DIR` set, then replay its files with
`GPUWM_WRF_EXACT=1 .../replay_native_all.py FILES --month M --out RECEIPT`.
Real options 1/2/11: 1,026,054 output/clock words, zero differences; this case has
zero flash rates. The earlier real option-3 capture has 253 nonzero LPI values.
The option-2 replay splits the output into two tiles after one domain reduction.
This proves the array ABI tiling, separately from streamed model decomposition.

The continuation also runs stock `woof go` forecasts for all four options,
checkpoint continuation, both nest feedback modes and ordinary production
ensemble members. Hourly maps at 19, 20 and 21 UTC compare the 2024-05-25 case
with the same native selector and surface observations. The initial replay
calls have zero flash rates; option 11 develops nonzero counts by 21 UTC.
Its whole-forecast amounts and some patches differ from native WRF. Surface
observations do not establish flash forecast skill.
The separate 2026-10-16 option-3 forecast has active LPI. The lane report records
commands, word counts, figures, timing scopes and the native monthly-input wall.
