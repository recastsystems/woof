# EEPS option 16 evidence

This port implements WRF's public-domain `phys/module_bl_eepsilon.F`.
The v4.6.1 and v4.7.1 module and constants are byte-identical. Source URLs,
SHA-256 pins and licence records are in
`tools/eeps_wrf461_oracle/SOURCE.json`. There is no external EEPS table.
The implementation was recovered from the archived qualified EEPS work,
then integrated and checked on `next/2.9.12`.

## Arithmetic and native replay

Fast CUDA arithmetic is the default. `GPUWM_WRF_EXACT=1` selects the
shared strict compiler contract, pinned scalar math, no FMA contraction
and no flush to zero. `GPUWM_NUMERICS=strict` alone is not a claim of
native bitwise physics arithmetic.

The native accumulator `RADSUM` is uninitialized at module line 748.
The port initializes it to zero. Three independently rebuilt synthetic
traces match a stock build with zero-initialized locals; a stock
signaling-NaN build differs by 3,040, 2,241 and 1,718 record words.
An active native `SZV(KMINRAD-2)` access below the allocated range at
line 734 is recorded in the device health ledger. It is never clamped.

The archived synthetic regression has 72 columns, 44 levels and 18 calls
across three traces. All 689,472 output words match in the strict GPU
replay. A 216-column replay also crosses the workspace tile boundary and
checks the partial final tile.

Real inputs were captured before and after EEPS in a strict native WRF
v4.7.1 forecast. Each capture includes 64 interior columns, 49 levels,
all thermodynamic and surface inputs, every output and both native
carried scalar pairs. The replay restores the captured native inputs
for each call, including the actual advected state. It does not replace
native advection with a synthetic blend. The capture instrumentor is
`tools/eeps_wrf461_oracle/capture_native.py`; its generated source was
byte-identical to the module used in the native build.

Captures at native calls 1, 301, 601 and 901 compare 151,552 output
words with zero differences. A separate real native Eta/EEPS first-call
capture compares another 37,888 words with zero differences. The total
historical real-input replay was 189,440 words, independently of the
synthetic count. Its raw streams were deleted, so that historical count
is unsupported as a rerunnable qualification. The retained 2026-11-02
recapture below supersedes it.

For reproduction, use a private native build and the pinned public source:

```sh
python tools/eeps_wrf461_oracle/capture_native.py \
  --source module_bl_eepsilon.F \
  --output PRIVATE_WRF/phys/module_bl_eepsilon.F \
  --receipt native-capture.json
# Rebuild the module and native executable with SOURCE.json strict flags.
# The strip i=40:103, j=75 must lie inside one MPI tile.
GPUWM_WRF_EXACT=1 python tools/eeps_wrf461_oracle/replay.py \
  eeps-real-000001.bin --receipt replay.json
```

## Whole forecast and cost

The real-input case starts at 2024-05-25 18 UTC and runs six hours at
3 km spacing, with 150 x 150 mass columns, 49 levels and a 12 s step.
Its namelist selects EEPS 16, revised MM5 surface layer 1, Noah 2,
Thompson 8, RRTMG 4/4 and no cumulus. The whole forecast ran through:

```sh
GPUWM_NUMERICS=fast python -m woof go --wrfinput CASE \
  --outdir OUTPUT --no-supervise --benchmark --products none
```

On a shared RTX 5090 it completed all 1,800 steps in 106.474 s wall time.
The reported median whole-model step was 16.439 ms. This includes other
physics and is separate from the EEPS stage measurement.

The native reference was intentionally stopped after the third-hour
capture had passed. It supplies matched 19, 20 and 21 UTC histories;
it is not a completed six-hour native forecast. The six-hour WOOF
forecast has seven saved hours, with finite temperature, wind, PBL height,
TKE and dissipation fields.

Maps were produced only by `rw_wrfbatch`, through `woof.rustwx`, and
inspected beside native WRF and observed surface stations at all three
hours. Both models retain the warmer southwestern region, the deeper
western PBL and the shallower eastern PBL. The third-hour maximum PBL
height difference is 667 m in the southwest; the first two maxima are
18.7 and 59.3 m. The temperature maxima are 0.150, 0.427 and 2.08 F
of difference, and wind-speed maxima are 0.297, 0.280 and 1.21 kt.
These are whole-forecast differences, not native module replay errors.

| Valid UTC | Temperature RMSE, WOOF / native, K | Wind RMSE, WOOF / native, m/s | WOOF temperature / dewpoint bias, K |
| --- | --- | --- | --- |
| 19 | 1.29594 / 1.29615 | 1.19401 / 1.19393 | +0.16335 / -1.32240 |
| 20 | 1.53938 / 1.53986 | 1.16683 / 1.16617 | +0.79135 / -2.08823 |
| 21 | 2.59255 / 2.59295 | 1.43178 / 1.42605 | +1.69477 / -2.76891 |

Temperature and dewpoint scores use 65, 64 and 65 station samples;
wind scores use 64, 63 and 64. The native forecast shares the growing
warm and dry bias. These scores do not establish a skill advantage.
No direct PBL-height observation or radar comparison was checked.

The fast stage benchmark repeats real captured columns to the same
49 x 150 x 150 shape. Fifteen CUDA-event samples after three warmups
include launcher allocation and device health recording, and exclude
input preparation, state restoration and host reads. Median EEPS time
is 0.997376 ms; YSU is 0.736192 ms, a ratio of 1.354777. YSU receives
neutral similarity denominators and a fixed 0.1 m roughness for this
comparison; it is not a native YSU replay or a comparison of every PBL.

That benchmark repeats a few captured columns against a YSU with neutral
similarity denominators, so it is not a forecast cost. The same holds for
the 0.811516 ratio once claimed for the `ea830c91` implementation: on a
real forecast `ea830c91` costs 1.46 times YSU, over the 1.2x target.

The forecast of record is the release timing case: GFS 0.25 degree,
2026-11-01 00 UTC, 12 km Lambert, 251 x 201 x 49, dt 60 s (three native
EEPS substeps), 3 h, Morrison, Kain-Fritsch, Noah, RRTMG 4/4 and surface
layer 91 for both PBLs, fast numerics, one quiet RTX 4090. The stage is
the CUDA-event time of the surface layer plus the PBL, the median over
steps 7-180. The arms ran back to back from the same prepared inputs in
the order YSU, `ea830c91`, fused, fused, `ea830c91`, YSU.

| Arm | Stage ms, two runs | Stage / YSU | Whole step ms, two runs | Step / YSU |
| --- | --- | --- | --- | --- |
| YSU (bl 1) | 2.621, 2.648 | 1 | 89.57, 89.32 | 1 |
| EEPS `ea830c91` | 3.830, 3.848 | 1.457 | 93.37, 91.15 | 1.031 |
| EEPS fused (this kernel) | 2.745, 2.732 | 1.039 | 90.18, 91.55 | 1.016 |

The fused kernel forms each tridiagonal row inside the elimination sweep
that consumes it, computes the Richardson numbers and stability functions
in the column-packing pass, reads U, V and T from the inputs instead of
copying them, and forms the top-down TKE source where it is read. Every
value is the native expression in WRF's operand order; only the loops
changed. The workspace holds 22 column arrays instead of 35 (227 MB
instead of 361 MB at this shape), and the kernel uses 64 registers and no
local frame at 49 levels. On one captured call of this forecast (step 90)
the EEPS kernel alone takes 1.730 ms against 2.871 ms. The strict builds
of the two kernels agree in all 29,866,992 output and carried words of that
call. The fast builds differ in 230,330 words by FMA contraction; the
largest difference is 2.6e-6 K/s in RTHBLTEN.

On the fused kernel the retained native recapture below replays 151,552
words with zero differences, max ULP zero, and still rejects the planted
one-bit PBLH error. The full hour-three call replays 13,320,000 words with
zero differences. EEPS evolves two prognostic turbulence equations and
solves them twice per call at dt=12 and three times at dt=60, which YSU's
diagnostic closure does not; on the forecast above that costs 4 percent
of the surface-layer and PBL stage.

A separate intact-kernel phase profile uses the archived synthetic trace
at 44 x 150 x 150 and dt=12. It inserts fenced `clock64` reads into a copy
of the source, using the production fast compiler options. Every output
and carried value is unchanged; in the fast build 5,000 RVBLTEN words are
-0.0 where the production build writes +0.0, a zero tendency formed by a
contracted subtraction. The production kernel's median time is 0.644 ms
(the pre-fusion kernel took 0.956 ms on the same trace) and the
instrumented kernel's 0.675 ms.

| Phase | Fraction of mean column cycles |
| --- | --- |
| Column packing, thermodynamics and Richardson numbers | 26.50% |
| PBL height, surface values and top-down source | 5.96% |
| Two native turbulence substeps, including epsilon and TKE solves | 22.17% |
| Diffusivities, six atmospheric solves, output and health recording | 45.37% |

Column-cycle fractions include scheduling and cache effects; they are not
an additive attribution of device wall time. The forecast stage above is
the comparison of record. Reproduce the diagnostic with:

```sh
GPUWM_NUMERICS=fast python tools/eeps_wrf461_oracle/profile.py \
  --receipt phase-profile.json
```

## Integration and limits

`pek_adv` and `pep_adv` use the existing positive-definite scalar
transport, boundary, nest and member-local paths. `PEK_PBL` and `PEP_PBL`
remain separate history/restart carriers. Checkpoint and restart include
both pairs. GPU checks cover live nest forcing and feedback modes 0 and
1, two member-local physics executions against independent calls, and
continuous versus restarted coupled trajectories.

The revised MM5 (1), Eta (2), MYNN (5) and classic MM5 (91) pairings are
admitted. Under Eta, its own Registry package independently allocates a
zero `TKE_PBL` field. A native one-step Eta/EEPS run completed and its
initial history contains 1,125,000 zero TKE words. The GPU retains this
separate zero column after three coupled steps; it never substitutes
EEPS's positive PEK field. The native witness wrote only initial TKE,
so later preservation is supported by the native read-only surface
call and the absence of TKE_PBL from EEPS's argument list.

With option 16 off, the candidate and port produce byte-identical
serialized payloads for all 169 fields, 87,904 bytes, after four coupled
YSU steps. That check covers the tested trajectory, rather than every
possible option-off configuration.

This is one real-input case. It does not establish broad scientific
skill, native full-model bitwise agreement, or a full nested or ensemble
forecast comparison. The native-input door retains its existing
multi-member limitation; resident member execution is checked separately.

## Retained native recapture and PBL-height diagnosis (2026-11-02)

The original-format captures at calls 1, 301, 601 and 901 were regenerated
from the original WRF 4.7.1 case, strict build and instrument. All four raw
SHA-256 hashes reproduce the originals. Lane `9399d8cd93b` and release
snapshot `8f0b2cf1d39d` each compare **151,552 output words, zero different,
max ULP zero**. Both reject a one-bit PBLH reference error with one word
different, max ULP one and exit status one. The separately witnessed native
undefined RADSUM-zero interpretation is unchanged; no words are excluded.
The previous separate Eta capture is not included in this new count.

Inputs, native outputs, executable, source instrument, original IC/BC,
commands and hardware are retained permanently at
`datasets/paper/wrf-ports/port-bl16-eeps/recapture-20261102/` under the data
root. `MANIFEST.json`, `REPRODUCE.md` and `sk3-replay-retained.py` provide
hashes and a replay procedure that writes only to separate scratch.

The first state difference occurs at initialization before EEPS: EOS
pressure differs in 1,056,334 of 1,102,500 words, by at most 3.125 Pa.
Initial PB, T, PH, PHB and QVAPOR agree. Reconstructing with the shipped
default EOS reproduces every GPU pressure word, while the strict EOS
reproduces every native word. This identifies the existing stabilized
hypsometric arithmetic in `woof/core/kernels/diagnostics.cu:205`, compared
with native FP32 `LOG(pfd/pfu)`, as the first departure. This is separate
from the EEPS column implementation. It does not establish that EOS
arithmetic alone accounts for every later coupled-model difference.

The new RTX 4090 diagnostic gives a 591.531494 m hour-three maximum at
WRF cell (12,17), rather than reproducing the old RTX 5090 maximum of
667 m exactly. The native 21Z history hash is identical to the historical
reference. The supplementary observer preserves all 22,500 native PBLH
words, and replay of its complete native hour-three call compares
13,320,000 output words with zero differences, max ULP zero.

At that cell the coupled inputs select different Richardson crossings:
native PBLH is 4658.058594 m, KPBL 17, and GPU PBLH is 4066.527100 m,
KPBL 16. Native virtual-potential-temperature bulk Ri at levels 16/17 is
approximately -0.152/0.661 after surface thermal excess; GPU values are
0.454/0.772. The critical value is 0.25. Native sensible heat flux is
122.721 W/m2 with negative inverse Monin-Obukhov length, while GPU flux
is -28.604 W/m2 with positive inverse length. The native column receives
about 0.251 K thermal excess; the GPU column receives none. These Ri
values are explanatory float64 profiles, not the strict word oracle.

With the same CUDA implementation, substituting only native temperature
into GPU inputs raises PBLH to 4564.877441 m and KPBL to 17. Substituting
all surface inputs gives 4289.276855 m; substituting every input gives
exactly 4658.058594 m. Fast and strict PBLH are identical at this cell
for either fixed input set. Over all 22,500 cells their largest same-input
PBLH difference is 0.103516 m for native inputs and 0.106689 m for GPU
inputs. The large reproduced discrepancy is therefore a coupled input
profile and surface-stability difference, rather than a PBLH arithmetic
error in EEPS. The historical 667 m magnitude remains a historical map
measurement, not a newly reproduced word-level result.

Evidence is retained with the dataset: `receipts/lane-strict-four-calls.json`,
`receipts/release-strict-four-calls.json`, `negative-control.json`,
`receipts/lane-strict-full-hour3.json`, `receipts/observer-control.json`,
`receipts/causal-000900-{fast,strict}.json`,
`receipts/domain-pblh-{fast,strict}.json`, and the initial EOS receipts.
The standalone diagnostic snapshots intentionally synchronize to the host
and provide no production timing. No runtime arithmetic was changed by
this closure.
