# CAM5 stock-module diagnostics

These tools pin and replay the public WRF column code for MP11, CU7 and
SHCU2. The options are wired to the forecast runtime, namelist import,
state inventory and IO. Generated soundings remain diagnostics. Real-input
proof uses driver-boundary captures from a strict native forecast. See
[the suite qualification record](../../docs/public/cam5-suite.md) for results
and limits; runtime admission does not assert completed qualification.

`SOURCE.json` pins WRF 4.7.1 at
`f52c197ed39d12e087d02c50f412d90d418f6186`. Every entry is hash checked before
compilation. The WRF release public-domain statement is retained in
`LICENSE-WRF.txt`. Source-level notices were inspected before recovery. The
inline notice in `module_cam_mp_microp_aero.F:1511-1513` permits use, copying
and modification for any purpose without a fee. That file also imports the
separate `error_function` module. Its CALERF routine is ported directly from
the WRF release under that release's public-domain statement. No competitor code is used.
The CAM column physics sources used here are byte-identical to the legacy
4.6.1 pins; the coupling files changed between releases.

The stock CU7 kernel preserves the native XLAND branch and momtran expression.
The stock MP11 kernel preserves native ndrop indexing and CALERF evaluation. Corrected-source legacy
fixtures and the private corrected-ndrop kernel are attribution instruments.
They cannot replace the unmodified native referee.

Build with a fresh output directory and a public source copy containing all
pinned files:

```sh
python3 tools/cam5_wrf471_oracle/build.py WRF_SOURCE cu7 CU7_OUT --case CASES
python3 tools/cam5_wrf471_oracle/build.py WRF_SOURCE shcu2 SHCU_OUT --case CASES
GPUWM_WRF_EXACT=1 GPUWM_NUMERICS=strict python3 -m tools.cam5_wrf471_oracle.diagnose \
  --cu7 CU7_OUT/native --shcu2 SHCU_OUT/native \
  --mp11 MP11_OUT --out diagnostic.json
```

The builder compiles sequentially with scalar strict flags and unchanged
physics. Framework stubs supply non-physics WRF interfaces. MP11 currently
uses the legacy 4.6.1 generated deck, explicitly marked in the receipt. Its
full native archive builder remains in `tools/cammgmp_wrf461_oracle`.
Raw fixture streams and prepared case files are disposed after their hashes
and diagnostic results are recorded. Regenerate them when needed. Legacy
tests accept `CAMZM_ORACLE_DIR`, `CAMMGMP_ORACLE_DIR` and `UWSHCU_ORACLE_DIR`;
an absent stream skips that diagnostic and grants no qualification.
The comparator counts all uint32 words, including signed zeros. One CUDA-event
sample and one host duration include status readback and output work; they are
not an integrator stage benchmark or a speed qualification.

Runtime buffers and pointer tables persist per domain. Device status joins the
existing health ledger in forecast mode; standalone oracles read it at their
explicit boundary. Strict numerics is opt-in. Fast mode uses native CUDA
arithmetic while retaining WRF's binary64 CAM working precision.

`capture.install` instruments only a private, source-pinned native build.
It writes paired little-endian streams at the three driver boundaries for
bounded columns and steps. `replay_capture` checks every output word,
including integers and signed zeros. Raw differences remain in its receipt. `--undefined-witness` accepts only a
source-pinned native initialization witness and separately counts inactive
IDEep/JT storage; floating-point and active-column differences are never exempted.
Use `witness_cu7_undefined` on a real captured input to compile two native
driver variants with distinct initial storage seeds. In inactive CU7,
IDEep and JT retain those seeds; MAXG is initialized to one and is defined.
The witness must leave every other native output word unchanged. This is
an attribution instrument, separate from the untouched native replay.

The CAMMGMP GAMMA helper follows the positive Cody branch distributed in
`module_cam_mp_cldwat2m_micro.F`, rather than the different generic gamma
helper. Active cloud/rain captures from native steps 1, 2 and 10 replayed
with zero differing words. Captured CU7 and SHCU2 columns at these steps
were inactive and do not establish active-convection coverage. Results
belong in the lane evidence. Full-tile +1 h captures additionally exercise
active deep and shallow convection. MP11 has 71 raw defined differences;
CU7 has zero defined differences after its native storage witness, and
SHCU2 is bitwise exact.

All 35 MP columns containing the 71 full-tile differences were also
replayed through an untouched native standalone call, reproducing their
forecast outputs exactly. General MIT CORE-MATH cosine and arccosine
interposers reduce those 71 differences to zero; they change native math
and do not count as an unmodified-native parity pass. The coordinator's
correctly rounded math ruling accepts an exact witness of native rounding
errors. The production MIT CORE-MATH functions remain unchanged. The system
glibc implementations are never copied; no LGPL code ships.

`rounding_witness.py` records original and correctly rounded trajectories
without modifying WRF source. It independently bounds cos/acos with system
MPFR at 256 bits (directed down/up), requires both endpoints to round to one
binary64 ties-to-even word, and compares the production CUDA helpers.
It checks all 1,970 distinct arguments from 2,478 calls in each trajectory.
CUDA has zero mismatches. Native has 37 distinct cosine and two arccosine
errors, each one ULP. Exact argument/native/correct bits and bounding values
are retained. All 71 downstream output differences disappear in the
37,905-word selected replay when only correctly rounded trig is substituted.
The receipt retains raw native failure counts and never applies a tolerance.
MPFR is an external development oracle, not an engine runtime dependency.

Run against the recorded real captured-column deck and native executable:

```bash
GPUWM_WRF_EXACT=1 GPUWM_NUMERICS=strict python -m tools.cam5_wrf471_oracle.rounding_witness \
  --native-exe NATIVE_REPLAY --deck REAL_CAPTURED_COLUMNS \
  --adapter MIT_CORE_MATH_HOST_ADAPTER --native-reference UNTOUCHED_NATIVE_OUTPUT \
  --work OWNED_SCRATCH --out ROUNDING_WITNESS.json
```

The adapter is the recorded host transcription of the same production
CORE-MATH functions; its notices and digest are checked and retained with
the command. The native trace must reproduce the untouched captured output
SHA256. No fixture input list is consulted by production math.
