# OSAS column check

This runs WRF v4.6.1 CU_OSAS, including OSASCNV, SHALCV, MSTADBT3,
TRIDI2T3 and the gfuncphys tables. Source and preprocessed files are
byte-pinned in build.sh. Only random-input and stage/table recording hooks
change the referee. The compiler flags prohibit fast math, contraction and
vector math lowering.

The 96 soundings cover deep, severe, shallow, capped, dry and near-saturated
columns, land and sea, two vertical grids and six calls at STEPCU 1, 3 and 5.
Chained calls use the preceding Fortran tendencies to advance the state.
The 12 by 8 and 1 by 96 layouts must have identical output words.

On a CPU sprint box, from the engine root:

    bash tools/osas_wrf461_oracle/build.sh "$WRF_SOURCE_ROOT" "$L/oracle-build"
    python tools/osas_wrf461_oracle/make_cases.py "$L/cases"
    for grid in k44 k29 k44-default k29-default k44-single; do
      "$L/oracle-build/run_osas" "$L/cases/cases-$grid.bin" "$L/cases/out-$grid.bin" "$L/cases/stage-$grid.bin" "$L/cases/tables.bin"
    done
    python tools/osas_wrf461_oracle/verify_fixtures.py "$L/cases" tests/data/oracles/osas "$L/fixture-receipt.json"

On a GPU sprint box, under the required card queue, set PYTHONPATH to the
engine root and tests directory:

    GPUWM_WRF_EXACT=1 python -m pytest -q tests/test_cu_osas_wrf461_parity.py
    GPUWM_WRF_EXACT=1 python tools/osas_wrf461_oracle/validate_cuda.py "$L/column-receipt.json"
    GPUWM_WRF_EXACT=0 python -m pytest -q tests/test_cu_osas_wrf461_parity.py

Strict mode uses injected XKT2 and preserves every WRF backout word.
Without an injection it uses the WRF REGTEST value 0.1. Fast mode uses CUDA binary64 arithmetic with contraction and a
counter draw keyed by model step and column. Its declared condensate
correction restores QL on columns whose rain fully evaporates. The corrected-backout comparison uses strict arithmetic and supplies WRF the
same counter draw; all other fields have zero word tolerance, and corrected
condensate tendencies have a stated absolute
tolerance of 1e-4 kg kg-1 s-1 plus an exact restored-word check.
This is column conformance evidence. It establishes no forecast skill.

The restart gate compares every serialized array after 20 steps, restore
and 20 further steps against 40 uninterrupted steps. It also compares an
immediate restore/write roundtrip. The test pins only the archive's created provenance stamp and then
requires whole-archive byte identity as well as every numeric member and
the restart identity. Set WOOF_OSAS_RECEIPTS to a lane receipt directory
to save all four archive hashes and every member hash.

A one-bit change to C0 must be caught in RAINCV. The table buffer must hash
to 46b826ff2e7519cd89149ca9c351bc8aa5819f5e1a1812dea1f306f5e6735a61.

The source license for each module and the derived tables is recorded in
`source-pins.json`; the original public-domain notice is `WRF-LICENSE.txt`.
Dewpoint-table segments begin at native binary64 predecessor roots generated
by `table_seeds.py`; the full 94,662-word table hash is checked before use.

For real-input capture, use a clean, writable copy of the pinned native WRF
source. The hook checks its source hash before writing and refuses a second
application:

    python tools/osas_wrf461_oracle/instrument_native.py "$WRF_SOURCE_ROOT"

Build that copy with GNU strict flags `-O2 -fno-fast-math -ffp-contract=off
-fno-tree-vectorize`. The instrumentation tool does not edit the build flags.
Run the real forecast with one MPI rank and `OMP_NUM_THREADS=1`. The first
active native row writes `osas-native-capture.bin`, `osas-native-output.bin`
and `osas-native-selection.txt` into the run directory. A first-step fallback
is replaced if a later row becomes active. The hook preserves native inputs,
random draws and arithmetic. Replay the retained row on the GPU:

    GPUWM_WRF_EXACT=1 python tools/osas_wrf461_oracle/replay_native.py \
      --capture "$RUN/osas-native-capture.bin" \
      --native-output "$RUN/osas-native-output.bin" --receipt "$L/live-replay.json"

`evidence/native-live-replay.json` records 60,000 real-input output words
with zero differences. The source pin also records byte equivalence of OSAS,
its GFS support modules and WRF preprocessing between v4.6.1 and v4.7.1.
