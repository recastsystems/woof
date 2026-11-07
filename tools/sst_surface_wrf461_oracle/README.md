# Native surface replay

Authority is public WRF v4.7.1, pinned in `sources.json`. Its SST-skin and
TMN modules are byte-identical to v4.6.1, the reused strict forecast build.
`WRF-LICENSE.txt` is the upstream public-domain notice. No extra numerical
tables are copied by these ports. No competitor source is used.

`instrument.py OWN_WRF_COPY` instruments the surface-driver block in an
owned native build. It captures the first 16 columns of each MPI tile at
steps 1, 360, 361 and 720, before and after all three operations. Real
forecast state, fluxes, counters and the complete TLAG bank are retained.
The capture format is defined by `read()` in `replay.py`. The instrumenter
also sets strict compiler arithmetic; build with no more than eight jobs.
Run the instrumented native real-data forecast with all three selectors on.

`build.py` verifies its numerical source pins before compiling. It extracts
the original SST block and the original skin/TMN call blocks, retaining
their numerical statements. The two module files compile unmodified. An
empty unused WRF error namespace and message-only logging routines complete
the standalone wrapper. Fatal input errors stop the oracle.

```sh
python tools/sst_surface_wrf461_oracle/build.py PUBLIC_WRF_SOURCE oracle-build
GPUWM_WRF_EXACT=1 GPUWM_WRF_GLIBC=2.43 python \
  tools/sst_surface_wrf461_oracle/replay.py \
  --captures OWN_NATIVE_CASE --oracle oracle-build/libsurface_stage.so \
  --receipt replay.json
```

Each native pre-capture enters both the native standalone blocks and CUDA.
Each stage is compared independently by uint32 word, including held state.
The final CUDA state is also compared with the actual native forecast's
post-capture. This witnesses that isolating the blocks preserved the native
forecast operations. A missing capture set fails the proof.

The recorded case is 73 × 60 × 44, dt 30 seconds, 2019-09-04 21 UTC to
2019-09-05 03 UTC. Eight captures give 24,592 compared words per stage,
73,776 stage words total, plus 24,592 forecast-witness words. All match.
The case includes land, water and midnight. Sea-ice transitions and the
annual boundary require supplementary branch tests and are not claimed
as branches observed in this forecast.

Cold-start SSTSK is zero: WRF's tools/gen_allocs.c sets initial_data_value
and the generated SSTSK allocation uses it. No later initializer appears
in real, physics-init or start-domain. WOOF uses the same zero seed, and
the first native forecast captures witness it. A supplied SSTSK or restart
state overrides cold initialization.

The lane report retains the exact native configuration, compiler command,
capture hashes, forecast log, renderer receipts, observed SST source receipt,
CUDA event samples, options-off carrier hashes and checkpoint state hashes.
Raw forecast outputs and prepared inputs are disposable after those receipts.
