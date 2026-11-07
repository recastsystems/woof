# CAM5 suite: MP11, CU7 and SHCU2

The native WRF CAM5 suite is selectable from a stock WRF namelist:
`mp_physics = 11`, `cu_physics = 7`, `shcu_physics = 2`. The registry maturity
is `implemented-unverified`. The ports execute column physics in CUDA,
retain CAM's binary64 working precision, and use the existing Rust input,
history and rendering paths. Default fast arithmetic and opt-in strict
arithmetic use the same physics. Required PBL/surface-layer pairings are
validated from WRF's driver contract.

The source authority is public WRF 4.7.1 commit
`f52c197ed39d12e087d02c50f412d90d418f6186`, pinned in
`tools/cam5_wrf471_oracle/SOURCE.json`. Its CAM column sources are identical
to the pinned WRF 4.6.1 sources; coupling differences are recorded in the lane's
source audit. The WRF public-domain statement and the inline unrestricted
FFT notice are retained. No competitor source is used.

The real-input strict replay captures both MPI tiles at +1 h of a native
1974-04-03 12–15 UTC forecast, on 250 by 200 columns and 49 mass levels:

| selector | compared 32-bit words | result |
|---|---:|---|
| MP11 | 53,179,632 | 71 raw native differences; exactly witnessed native trig rounding residue, zero unexplained differences |
| CU7 | 111,100,000 | zero defined differences; 89,864 raw IDEEP/JT differences are separately witnessed native undefined storage on inactive columns |
| SHCU2 | 44,600,000 | zero differences |

The CU witness compiles two private native initialization instruments.
Only inactive IDEEP/JT retain the distinct seeds; active diagnostics and
all floating-point outputs remain checked. The replay records raw counts
and witness counts separately. It never converts a floating-point failure
into an undefined-storage allowance.

All 35 real MP columns containing the 71 differences were isolated. The
untouched native standalone call reproduces their forecast capture exactly.
A general MIT CORE-MATH cosine interposer reduces the differences from 71
to five; adding its general arccosine reduces them to zero over 37,905
output words. Neither interposer contains fixture argument keys. These
instruments change native math: the untouched native referee still differs
in 71 words. They are not counted as an unmodified-native parity pass.

The production helpers already use MIT CORE-MATH correctly rounded cosine
and arccosine. The exact witness in
`tools/cam5_wrf471_oracle/rounding_witness.py` traces both trajectories of
all 35 affected real columns: 2,478 trig calls per trajectory, 1,970 distinct
arguments. Independently directed 256-bit MPFR bounds round at both ends
to the same binary64 ties-to-even result. CUDA matches that result on every
argument. Native glibc 2.43 is one ULP wrong on 37 distinct cosine inputs
and two arccosine inputs. Exact input/result bits and bounds are retained
in `mp-correct-rounding-witness-results.json` in the lane evidence. The
37,905-word selected replay retains 71 raw native differences and has zero
differences when only these correctly rounded functions are used by native.
This is the declared native rounding residue authorized by the coordinator;
it is not native undefined storage or a relaxed numerical tolerance.
No glibc source is copied, no LGPL implementation ships, and no
argument-specific substitution enters the production path. MPFR is a system
development oracle, with no shipped code or engine runtime dependency.

A complete three-hour `woof go --wrfinput` forecast and the corresponding
strict native forecast were run with MP11/CU7/SHCU2, UW PBL 9, surface layer
1, Noah 2 and legacy RRTMG 4/4. Rust temperature, dewpoint and accumulated
precipitation maps at 13, 14 and 15 UTC were examined beside native WRF
and historical station temperatures. The warm sector, moisture corridor
and main rain band agree in placement; light precipitation differs in
intensity and individual cells. Seventeen station temperature RMSEs at
13/14/15 UTC are 3.06/3.04/3.79 F for the final GPU forecast and
3.04/3.16/3.84 F for native WRF. One case does not establish general
forecast skill.

CAM cloud fraction, effective radii, detrainment, mass flux, number
moments, CUSH memory and held CU rates are carried through allocation,
physics composition, IO and serialization. Native CAM signed moment
residuals are preserved on warm restart; finiteness and ceiling checks
remain. Native CAM radii feed the radiation coupling. Checkpoint
continuation, actual native restart, both nest feedback modes and
independent ensemble member ownership have dedicated lane receipts.
The nested continuation preserves the solar diagnostic producer flag:
parent and child history files at the resumed time are byte-identical
to the continuous run. Older checkpoints without that optional flag keep
the previous fallback until the radiation producer runs.
The optimized packed ensemble bootstrap still supports its existing
Thompson/YSU composition; this port uses ordinary member physics.

The native binary64 CAM kernels cost more than Morrison and New SAS.
CU7 now reuses a pressure logarithm only for bit-identical inputs and
uses the exact zero of log(1). SHCU2 omits unused saturation derivatives
and stops repeated saturation iterations only at a bit-identical fixed
point. Both active native tiles still pass their strict replay. No
precision reduction or tolerance-based iteration truncation is selected.
An alternating RTX 4090 comparison reduced CU7 from 19.23 to 17.77 ms;
SHCU2 changed from 214.45 to 218.51 ms there. The shared RTX 5090 shallow
comparison improved from 505.98 to 453.77 ms. Card/compiler effects and
contention are disclosed rather than combining these into one speedup.

The same-input RTX 4090 stage comparison gives CU7 17.76 ms versus
New SAS including NSCV 2.69 ms, MP11 129.13 ms versus Morrison 6.00 ms,
and SHCU2 409.05 ms. These shared-card medians are not exclusive timings;
the shallow samples were particularly variable. A later speed pass measures
NSCV's standalone shallow stage as the nearest existing shallow mass-flux
scheme. On one RTX PRO 6000 Blackwell card, interleaved fast samples give
CU7 16.72 ms versus New SAS including NSCV 2.79 ms (6.00x), and SHCU2
74.48 ms versus NSCV alone 0.762 ms (97.76x). The 1.2x target is missed.
CU7 and UW shallow remain named opt-in selectors, off by default.

A separate native WRF 4.7.1 task-clock profile measures ZM's entropy
inversion alone at about 2.7x the entire New SAS call, and UW's saturation
adjustment alone at about 13x NSCV. These are additional native parcel
and implicit-CIN solves, not CUDA launch or storage costs. The CAM sample
was interrupted after 12 completed steps; normalizing its recorded work
by all 15 scheduled calls conservatively understates its cost relative
to the completed 15-step New SAS sample. Both closed profiles lost zero
samples. This is evidence for the named physics-cost exception, not a
GPU lower bound. GPU register spills and poor active-warp occupancy are
still implementation costs. A staged environment and fast divide
candidate slowed both GPU stages and was reverted. Commands, raw
samples, profile limits and replay packets are retained with
`spd-cam5-zm-uw.md` in the lane report folder.

The original lane also counted the native routine work:
25,000 CU columns perform 3,424,014 entropy inversions and 1,149,616
entropy evaluations; SH performs 4,244,705 saturation solves and 575,916
buoyancy evaluations. MP couples aerosol activation and nucleation to
two-moment cloud physics. Instrumentation preserves every output word.
These measurements explain substantial native work; they do not prove
that no further faithful GPU optimization is possible.

The original port report and small evidence are kept with the maintainers
(`port-cam5-mg-zm.md` and
`evidence/port-cam5-mg-zm/`). The later CU7 and SHCU2 speed report is
`spd-cam5-zm-uw.md`, with evidence under `evidence/spd-cam5-zm-uw/`.
These two rows qualify only for the named opt-in physics-cost exception;
neither meets the 1.2x speed target. MP11 has a separate speed lane.
Raw native differences and performance limits remain declared.
Registry maturity remains unchanged: one observed case does not establish
broad forecast skill. Numerical production code did not change when this
rounding witness closed the qualification; the reviewed forecast, timing
and lifecycle results apply to the same physics source.
