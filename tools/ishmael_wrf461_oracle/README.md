# Native mp=55 replay

Authority is WRF v4.7.1. The retained directory name reflects the recovered
v4.6.1 work. The module and three table blobs are identical at both tags.
See source-pin.json and WRF-LICENSE.txt. No competitor implementation is used.

Current component status is supported. qualification.json records the real
native capture, strict 213,156-word replay (zero differences), default 180-step
forecast, Rust maps, timings and lifecycle receipts from 2026-11-01. The
former synthetic-column campaign is not qualifying evidence. build.sh builds
one pristine strict reference and optionally replays one captured call. It
runs no sweep.

In an authorized private native WRF copy, instrument only the pinned module:

```sh
python3 capture_native.py WRF/phys/module_mp_jensen_ishmael.F captured.F --receipt capture-source.json
```

Use captured.F in that private copy and rebuild native WRF with strict
numerics, contraction disabled, REAL32 and at most eight make jobs. Preserve
the configure file and build log. During a real forecast set
WOOF_CAPTURE_MP55_PREFIX to a path inside the run folder and
WOOF_CAPTURE_MP55_STEP to the explicit model step to record. The tool
captures the center west-east row of the first monitor tile at that step, with all
levels, before and after the unchanged numerical routine. The prefix files
must not exist. Unset the prefix for ordinary native runs.

The input packet carries 19 input volumes and two accumulated precipitation
fields. The expected packet carries all 35 output volumes and four surface
fields from exactly one native call. The file order is documented in
make_columns.py and compare.py. Never label synthetic packets as captures.

Replay that captured call through the production launcher:

```sh
GPUWM_WRF_EXACT=1 python3 run_gpu.py capture-inputs.bin gpu.bin 1 DT --want capture-want.bin --json replay.json
```

DT must be the native timestep in the packet. A qualifying report records
0 differing words, the number of compared words, input/output hashes,
source pin, compiler recipe, native forecast log and capture source receipt.
Any native-undefined residue needs a separate witnessed native diagnosis.
Do not silently accept nonfinite or differing words. Strict mode retains
the pristine native limiter; default mode applies the declared rain-riming
limiter correction. Default mode uses fast CUDA elementary functions and
FMA; both modes retain subnormal habit volumes.

The completed default forecast runs from 1974-04-03 12:00 to 15:00 UTC on
250 x 200 x 49 mass cells at 12 km and dt=60 s, using stock namelist physics
55/YSU/91/Noah/legacy RRTMG/Kain-Fritsch. All nine paired Rust maps at 13,
14 and 15 UTC were inspected beside native output; temperature maps carry
public station observations. Shared RTX 5090 timings are 110.271 ms/step
against Morrison's 183.312; microphysics sections are 17.438 and 52.797 ms.
These are shared-device measurements, not an intrinsic ranking.

Both same-scheme nest feedback modes exercise nonzero three-habit state.
The native two-minute restart imports and advances two full GPU steps; the
default initializer floors witnessed native negative transport residues and
records the per-field counts. Checkpoint continuation matches 83,325,438
history words. Two complete real-input ensemble members match ordinary
references and Rust history hashes over two steps. Option-off matches
71,075,438 candidate history words in the final frame; all four complete
history files are also byte-identical by SHA-256. Mixed-scheme nest edges retain the
named missing habit-volume closure refusal. No general observation-based
forecast skill or full native trajectory bit identity is claimed.

Commands, configurations, logs, native capture packets and figures live in
<reports>/WRF-PORTS-2026-11-01/evidence/port-mp55-ishmael/. Raw full
forecast frames are removed after reporting. thin_history.rs,
station_overlay.rs and history_words.rs provide Rust data-path evidence
helpers; render_evidence.py in the receipts drives woof.rustwx.
