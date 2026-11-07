# Ferrier-Aligo with advected rime: real native calls

Selector 15 uses `FER_HIRES_ADVECT` from public WRF. Its transported fields
are QVAPOR, QCLOUD, QRAIN, QICE and QRIMEF. QRIMEF is ice mass times rime
factor, not another water mass. The shared Ferrier CUDA column derives
`QRIMEF/QICE` at entry and writes `QICE*F_RIMEF_PHY` at exit. QT and the
three fractions are local, unlike selector 5's saved fraction arrays.

The native pin is WRF v4.7.1
`f52c197ed39d12e087d02c50f412d90d418f6186`. The Ferrier module and expanded
rain table are byte-identical to v4.6.1
`d66e442fccc04111067e29274c9f9eaccc3cef28`. WRF's public-domain notice is
retained in `WRF-LICENSE.txt`; neither inspected file adds a restriction.
Source hashes and the original failure are in `NATIVE-STARTUP.md`. No
competitor code or tables are used.

Unchanged native WRF fails this option before integration. Three caller
corrections are required to obtain a valid native trajectory:

1. `native-init-correction.patch`: option 15 omits the optional fraction
   arguments whose actual allocations are each one element. This prevents
   domain-shaped writes beyond the allocations.
2. `native-driver-correction.patch`: its presence guard requires rain and
   drops unused legacy option-5 restart tables. The advected wrapper never
   reads those tables.
3. `native-solve-correction.patch`: `solve_em` passes the allocated QRIMEF
   scalar and its presence flag to the microphysics driver.

Apply these patches to a private copy of the pin, then
`native-capture.patch`, which writes input and output words around
`FER_HIRES_ADVECT` at steps 1, 11, 21 and 41. It changes no column arithmetic.
Build the complete `em_real` target with strict flags (`-O0`,
`-ffp-contract=off`, `-fno-fast-math`) and at most eight make jobs. The
first build used eight jobs; the final caller rebuilds used two. Execution
used four MPI ranks.

The real case uses the public `wps_30km.tar.gz` January 2000 data, 73 by 60
mass cells, 32 levels, 180-second steps, and 2000-01-24 12Z through 15Z.
Stock `real.exe` supplies the input and boundaries. Radiation is 4/4,
surface layer 1, Noah 2, YSU 1 and KF 1. The exact namelist and all commands
are retained with the lane evidence; `cu_rad_feedback` is false.

Replay all captured tile calls with the production adapter:

```sh
GPUWM_WRF_EXACT=1 PYTHONPATH=. python \
  tools/ferrier_hires_wrf461_oracle/advect/replay_real.py \
  /path/to/case/mp15-capture-*.bin --out real-replay.json
```

The 2026-11-01 receipt compares 3,212,040 FP32 words in sixteen calls,
covering theta, four moisture fields, QRIMEF, two rain fields and snow
ratio: **zero differing words**, no nonfinite words and no undefined-value
allowance. Capture hashes are included. Synthetic CSV fixtures in this
directory's harness are separate regression material, not this proof.

The fast default was also run through a complete three-hour `woof go
--wrfinput` forecast. Rust-rendered temperature, dewpoint, wind and hourly
precipitation panels were inspected beside corrected native WRF at 13Z,
14Z and 15Z, with thirteen reporting surface stations. The short case is
not a general forecast-skill certification. Existing dry-theta and
fractional-sea-ice import divergences remain explicit. Native Ferrier's
wrapper does not export its internal reflectivity diagnostic, so no
native reflectivity parity is claimed. Mixed-scheme nest edges still
require an explicit rime closure and are refused by name.

The report, figures, source hashes, actual commands, per-stage CUDA-event
timings, checkpoint continuation and lifecycle checks are retained under
`WRF-PORTS-2026-11-01/evidence/port-mp15-fer-advect/`; the row report is
`WRF-PORTS-2026-11-01/port-mp15-fer-advect.md`. Raw run output is removed
after delivery. Do not describe the corrected caller reference as stock
unchanged native WRF or the whole forecast as bitwise native WRF parity.
