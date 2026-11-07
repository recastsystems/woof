# MP97 column implementation and historical qualification

Current real-input replay, forecast, timing and lifecycle measurements are
in [REAL-INPUT.md](REAL-INPUT.md). The historical measurements below are
retained with their original scope. The current default-switch radar
replay is bitwise, including melting particles.

The measurements below belong to the October implementation, before its
forward merge onto 2.9.12. They use synthetic inputs, not inputs captured
from a real native forecast. They do not qualify this merged engine under
the current real-input replay, forecast, lifecycle and speed requirements.
The registry therefore keeps `implemented-unverified` maturity.

The GCE source is byte-identical in WRF 4.6.1 and 4.7.1. The shared radar
source changed. See [SOURCE.md](SOURCE.md) for the source and licence audit.

Strict and default modes match 2,727,648 state elements and every per-step
increment against the repaired scalar WRF 4.6.1 oracle at 0 ULP. The
fixtures contain 66 columns, 40 levels, all eight switch combinations,
12 steps at dt=12 seconds, and 6 steps at dt=60 seconds.

Default and strict CUDA output files are bitwise equal, including radar.
Unmelted radar is bitwise equal to the referee. Only radar values entering
the melting-particle binary64 complex backscatter and exponential chain
differ from scalar Fortran. This chain is reused from kernels/refl.cu;
GCE integration is kernels/gsfcgce_refl.cu:120-148. It is not claimed
bitwise. The phased full-column integer-bin hashes prove an absolute
difference below 0.01 dBZ. This is a bound, not an exact maximum.

The one-bit planted rain-freezing coefficient changes 6,665 state
elements with maximum 8,320 ULP. compare.py rejects it. Missing required
cases and nonfinite outputs fail qualification. State hashes distinguish
signed zero. Equal consecutive states and increments under the same float32
time step also prove equal derived per-second tendencies.

## Referee and declared repairs

WRF source: d66e442fccc04111067e29274c9f9eaccc3cef28.
module_mp_gsfcgce.F SHA256:
43e841f78e97a35e261f7d91d83c6ad54c7bf30a74625e933603d2e8d6126ef1.

The oracle links the scalar build's own module objects. Flags:
-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize.
The recompiled source arm reproduces the referee disassembly.

The historical October implementation applied three repairs: graupel sedimentation for ice2=2, warm-branch
transport coefficients for ice2=3, and zero graupel in the ice2=1 radar
column. Stock and repaired arms agree on every full three-ice state
and radar value. The stock ice-plus-snow state also agrees.

The current implementation preserves native ice2=2 graupel fallout. Only
the other two repairs remain; both have signaling-NaN trap witnesses on
real captured inputs. Historical synthetic hashes above describe the old
three-repair arm, not the current two-repair source. See REAL-INPUT.md.

phys_tot and acphys* in microphysics_driver.F:2103-2132 belong to the
MP7 case that begins at :2066. The MP97 call at :2022-2055 does not pass
them. Old process-heating writes in module_mp_gsfcgce.F:2830-2842 are
comments. MP97 does not advertise those outputs.

## Commands

Run every build and test on a sprint box. With T set to this worktree
and L set to its lane folder, CPU:

    source /work/env.sh
    source /work/wrf/env.sh
    bash tools/gsfcgce_wrf461_oracle/build.sh "" "/oracle-build"
    python tools/gsfcgce_wrf461_oracle/make_columns.py "/fixtures"
    bash tools/gsfcgce_wrf461_oracle/run_fortran.sh "/oracle-build" "/fixtures" ref fix src
    python tools/gsfcgce_wrf461_oracle/hash_receipt.py "/fixtures" "/oracle-build/out/fix" tools/gsfcgce_wrf461_oracle/oracle-hashes.json

Through the GPU card queue, tag pl-port-mp97-goddard-gce:

    export T="/src"
    bash "/tools/gsfcgce_wrf461_oracle/run_gpu.sh" "/fixtures" "/gpu"
    python tools/gsfcgce_wrf461_oracle/hash_receipt.py "/fixtures" "/gpu/strict" "/receipts/strict-hashes.json" --reference tools/gsfcgce_wrf461_oracle/oracle-hashes.json --grade-out "/receipts/strict-grade.json"

Repeat for default and plant. Strict and default must pass; plant must fail.
For ULP measurements with arrays on one box, use compare.py. Its
--report-only option reports measurements without accepting a candidate.

prepare_cli.py follows the delivered MP7 Rust NetCDF fixture method.
It reads masses and switches from the table-column metadata, preserves
analyzed source fields, and cold-starts missing masses and lateral values
at zero. The public namelist, adaptations, source and produced hashes
are recorded. This is a real CLI integration check, not a matched WRF
forecast or observation skill experiment.

WOOF's own float32 math drives the column network. The shared direct
loader retains --ftz=false. Strict numerics retains --fmad=false; the
current engine's fast default applies its normal contraction policy. On
sm_120/NVRTC 12.9.86 the column local frame is zero and the
unspecialized GCE radar entry is 13,056 bytes. Its price retains the
18,432-byte ceiling of the composed radar exports.
