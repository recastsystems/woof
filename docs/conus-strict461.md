# Strict CONUS comparison with WRF 4.6.1

This is a measured implementation comparison on one case, not forecast skill
against observations or a claim for every WRF configuration.

The June 2021 WRF-JAX case uses the supplied 500 x 500 input grid, 499 x 499
mass columns, 44 mass levels, 3 km Mercator spacing, an 18 s step and four
acoustic substeps. Thompson 8, RRTMG 4/4, MYJ 2, Eta surface layer 2, Noah 2
and modified Tiedtke 6 remain active. Radiation is called every 9.9 minutes.

## Two referees

The bitwise referee is WRF 4.6.1 at
`d66e442fccc04111067e29274c9f9eaccc3cef28`, Ubuntu 24.04, gfortran 13.3,
glibc 2.39, with `-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`
and no libmvec. It runs on 12 MPI ranks with decomposition 3 x 4. WOOF runs
one RTX 5090 with strict numerics, default `GPUWM_WRF_GLIBC=2.39`, libmvec
off and a logged per-run radiation column batch of 512. This batching keeps
every scheme active. No second strict decomposition was run.

The paper accuracy metric uses a separate vectorized gfortran 15.2/glibc
2.43 WRF 4.6.1 pair, 3 x 4 versus 4 x 3. Its words are not the strict target.
`GPUWM_WRF_GLIBC` and `WOOF_WRF_LIBMVEC` remain independent selectors for
named referee builds. Fast numerics remains the default.

## Native wrapper replays and whole forecast

On WRF's captured real-step wrapper arguments, differing/compared words are:

| Scheme | Steps | Words |
|---|---|---:|
| Thompson, sedimentation and rain | 67, 100 | 0/132149815 each |
| RRTMG LW/SW, clouds and driver fields | 67, 100 | 0/56523227 each |
| Noah | 67, 100 | 0/11454046 each |
| MYJ and Eta surface layer | 1, 2, 3, 50, 100 | 0/545312190 total |
| Tiedtke 6 | 100 | 0/66483267 |

One corrected 400-step whole forecast matches all 204 common floating WRF
history variables at both +1 h and +2 h: **0/265734210 words differ at each
lead**, no missing reference variables, shape mismatches or nonfinite common
values. Model-only diagnostics are extensions outside this comparison.

The coupled QRAIN/RAINNC divergence closed by preserving the independent
map inverse, the final RK stage's coupled RQVFTEN and the physical boundary
WW seed that WRF's sumflux consumes. Standalone Thompson sedimentation and
rain replays were already exact. Strict history also exposes the four native
MYJ fields. The unused top TKE/length interface carries WRF's zero.

## Fast cost and accuracy

One 30000 MiB pool lease ran the base `57182636fd` and then the corrected
production source `bdfe96916b`, six model hours each on the same RTX 5090.
All physics, input words, history cadence, runtime and native bridges were
held fixed. Reported simulation walls are 177.1 and 179.4 s, or 29.517 and
29.900 s/model hour (1.012987 times). Whole process walls are 187.758 and
189.204 s (1.007701 times). Rendering and observation verification are
excluded. These measurements pass the 1.2 times cost limit.

Scaling the supplied 58.4 s/model-hour A100 baseline by the measured ratio
gives an estimate of 59.158 s/model hour and 1.787 times the original WRF-JAX
speed. Neither value is an after-port A100 measurement.

Against the vectorized accuracy referee across +1 to +6 h, fast after-port
U/V/T/QVAPOR/T2 ratios are 1.012-1.166 times the decomposition RMSE baseline.
QRAIN is 0.781-2.610 and RAINNC is 1.106-2.079. It is not uniformly closer
than the original fast run. The worst ratio is QRAIN at +4 h. The original
accuracy states ran on an A100; the after-port states ran on a 5090, so that
comparison does not isolate hardware effects. Six-lead Rust difference
maps, full-range checks and +4/+6 h zooms were inspected. They retain the
broad offshore rain structure while showing core displacement and intensity
differences, with larger southwest Florida differences at later leads.

## Receipts and compatibility

The lane's patch, commands, toolchain and executable identities, per-field
word counts, test logs, timings, metric JSON and inspected maps are under
`PAPER-EVIDENCE/woof-regional/conus-suite-exact/`. The companion
`WRF-JAX-2026-10-29/ACCURACY-AFTER-PORT.md` contains every main-variable ratio
beside the saved WRF-JAX and WOOF-before values. This candidate is untagged;
the evidence is not a released-tree qualification.

Checkpoint format 7 includes the independently stored `MF_VX_INV` map
inverse in setup identity. Format 6 is refused by name rather than guessed.
The CPU restart tests verify the new binding and refusal. A full GPU
checkpoint/resume forecast and other cards, decompositions or cases were
not checked here. The low-memory automatic streaming route's existing
RTHRATEN frame-classification refusal was not part of this resident proof.
