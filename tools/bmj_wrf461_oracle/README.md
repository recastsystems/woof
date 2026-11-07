# BMJ WRF 4.6.1 check

The CUDA source ports BMJDRV, BMJ, TTBLEX, BMJINIT and SPLINE from the pinned public WRF 4.6.1 source. One thread handles a column. Registry metadata selects the module, table builder, field bindings and restart state. No scheme-specific Python adapter is used.

Build the standalone sounding oracle from a public WRF v4.6.1 checkout:

    bash tools/bmj_wrf461_oracle/build.sh "$WRF_SOURCE_ROOT" "$ORACLE_BUILD_DIR"

The builder checks source SHA-256 values, compiles byte-unmodified WRF source at O0 and with the scalar WRF optimization flags, and requires the two captures to agree byte for byte. PRIVATE is exposed for table capture without changing arithmetic statements. The 96-column fixture has 12 sounding families, 45 levels, and four calls in each of two feedback/cadence arms. The second reference executable binds the settled correctly rounded routine. Both reference arms are retained and the limited stock-arm cloud delta is declared in the tests.

CPU and planted one-bit check:

    CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1 python -m pytest -q tests/test_bmj_wrf461_parity.py tests/test_bmj_driver_contract.py

GPU, inside the assigned card queue only:

    GPUWM_WRF_EXACT=1 python -m pytest -q tests/test_bmj_wrf461_parity_gpu.py
    env -u GPUWM_WRF_EXACT python -m pytest -q tests/test_bmj_wrf461_parity_gpu.py

Both modes use the same kernel through the shared direct NVRTC cache with FTZ disabled. BMJINIT must hash to the registry pin before any column runs. All 11 output fields and all 87,972 table words grade at 0 ULP against the declared reference. The stock feedback-off arm also grades at 0 ULP. The stock feedback-on cloud fraction differs by at most 4 ULP, cloud water and ice by at most 6 ULP; the remaining fields agree.

BMJ cadence is retained from the namelist. The callable replaces PRATEC and cloud diagnostics on due calls; the existing model-clock precipitation path accumulates the held PRATEC on every step. CLDEFI and the convective clouds live in checkpoint-covered driver fields. The restart test compares every serialized state array, driver field and serialized scratch array after an 8-step run against 3 steps plus checkpoint plus 5 steps, with MYJ/Eta/Noah, WSM6 and both legacy RRTMG and RTE+RRTMGP. It also checkpoints both oracle-driver arms.

The feedback merge applies WRF module_radiation_driver.F:1583-1587 to temporary radiation inputs and preserves prognostic condensate. RRTM/Dudhia, legacy RRTMG, RTE+RRTMGP and New Goddard are supported, including mixed or disabled streams. Feedback with a radiation adapter that lacks this merge is refused with the concrete reason. No forecast skill or full WRF forecast trajectory identity is claimed by these checks.


For real native-run input capture, use a private copy of WRF. The
instrumentation checks the source, constants and public-domain licence
hashes before changing the copy. Build the full native model with
`-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`, and use one
WRF tile per MPI rank with `OMP_NUM_THREADS=1`:

    python tools/bmj_wrf461_oracle/instrument_native.py "$PRIVATE_WRF_ROOT" \
        --receipt "$EVIDENCE_DIR/native-instrumentation.json"
    # Rebuild the private native copy, then run its wrf.exe with cu_physics=2.
    # Use the compiler/MPI toolchain recorded with that build.
    GPUWM_NUMERICS=strict GPUWM_WRF_EXACT=1 \
        python tools/bmj_wrf461_oracle/replay_native.py "$NATIVE_RUN_DIR" \
        --receipt "$EVIDENCE_DIR/native-replay-strict.json"

The capture writes the middle horizontal row of each MPI tile at the first
BMJ call. `bmj-native-*.bin` follows `bmj_oracle_io.read_arm(nstep=1)`:
seven level inputs, interface pressure, land mask, incoming CLDEFI, KPBL,
five level outputs, six surface outputs and integer activity flags.
All words are little endian float32 or int32. The companion `.txt` records
columns, levels, DT, STEPCU, feedback, timestep and tile coordinates.
The `.tables` file holds the 87,972 BMJINIT words, without expansion scratch.
The replay grades bit patterns, including signed zero. Its receipt records
branch counts, precipitation counts and hashes so quiet rows cannot be
mistaken for evidence of deep convection. Preserve that receipt and delete
raw captures once it is written.

WRF v4.6.1 and v4.7.1 distribute this same BMJ module byte for byte:
`62d2805aae152b58093d07100000b581458b536164d90f9cecc27a157e2bd92f`.
No external BMJ coefficient file is needed. The WRF public-domain notice
is retained in `licenses/LICENSE-WRF-public-domain.txt` and `NOTICE`.
