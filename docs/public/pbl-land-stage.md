# PBL and land forecast-stage qualification

The November 2026 retiming uses complete, evolving forecasts through the real
`woof go` CLI. It does not tile one replay call into a performance case.
The retained machine-readable receipt is
`receipts/retime-pbl-forecast.json`. Each row records its measured source,
card, candidate and nearest-scheme stage time, and sample counts.

The case starts at 2024-05-25 18 UTC, has 150 by 150 mass columns and 49
levels, uses 3 km spacing and 12 s steps, and advances one hour (300 steps). The separate EEPS cause comparison advances
three hours. Its
remaining physics is Thompson, legacy RRTMG, no cumulus and Noah, apart
from the surface-layer pairing each PBL requires and SSiB's native
three-layer soil input. SSiB's input was produced by a complete native
WRF REAL run from the matching metgrid files. It was not made by changing
the dimension labels of a four-layer input.

Each arm retains its own evolving state. Initialization runs in sequence.
The driver advances one complete model step per arm, reversing the order
on alternate rounds. Every arm synchronizes its device before yielding.
CUDA events surround the complete production PhysicsDriver stage. The
first call and steps 300, 600 and 900 are excluded in every arm, as
declared before sampling; those steps carry saved EEPS observations.
All other samples are retained. A second summary uses only same-step
pairs with zero involuntary thread context switches in both stage calls.
No sample is selected by its duration. Raw medians, p10 and minima are
reported alongside the clean pairs so host stalls remain visible.

QNSE is tested with its native default `mfshconv=1`. GBM and KEPS are
compared with MYJ, EEPS with YSU, and SSiB with Noah. QNSE's complete
stage includes its shallow mass flux. These are stage costs inside
whole forecasts, not ratios of whole-model runtime. They do not establish
forecast skill or timing on another card or case.

## Measured forecast stage

All seven real forecasts completed 300 steps on the same timed RTX PRO 6000
Blackwell card and source 5200f1a286deb433a80fd64c313bbf1b03838db3.
Zero involuntary switches alone did not remove the MYJ and Noah timing spread.
The audit therefore uses the largest of the clean median, p10 and minimum
ratios. This conservative diagnostic was chosen after seeing that spread; it
is not a typical-time estimate or a formal upper bound. It avoids claiming the
favorable median alone. No duration filter removes samples.

| Option | Median ratio | p10 ratio | Minimum ratio | Audit ratio |
| --- | ---: | ---: | ---: | ---: |
| QNSE / MYJ | 0.484 | 0.724 | 1.034 | 1.034 |
| GBM / MYJ | 0.288 | 0.438 | 0.621 | 0.621 |
| EEPS / YSU | 0.730 | 0.722 | 0.686 | 0.730 |
| KEPS / MYJ | 0.198 | 0.284 | 0.388 | 0.388 |
| SSIB / NOAH | 0.511 | 1.942 | 1.390 | 1.942 |

SSiB remains an explicit opt-in, with a measured 1.942x Noah lower-tail
qualification cost. Its median 0.511x alone would misstate the evidence.
The unmodified strict REAL32 native profile shows the additional canopy
radiative transfer, stomatal stress, coupled temperature solve and repeated
aerodynamic solve. The retained gprof records 42 million RASIT5 calls for
5.6 million land calls, alongside TEMRS1, RADAB and STOMA1. Noah uses its
bulk surface budget and canopy-resistance path instead. This is native
physical work, not a double-precision implementation path. CPU profile
seconds are not used as a GPU cost estimate. The full profile inputs,
outputs, source, compiler flags and commands are retained with the native
paper packet. This single case does not establish forecast skill.

All fifteen Rust temperature/wind/PBLH maps were inspected. Broad wind
patterns are coherent. QNSE has a deep northwestern mixed layer; GBM has a
shallower, patchier diagnosis; EEPS and KEPS have broad west-to-east PBLH
gradients. SSiB has a much warmer western temperature field. That contrast
is recorded, not treated as an observation-skill qualification.

## EEPS PBL-height investigation

A new complete strict WRF 4.7.1 reference and fast GPU forecast were run
from the same original input pair. The maximum PBLH differences are
48.8956, 84.6561 and 335.5396 m at hours one, two and three. The hour-three
PBLH RMSE is 12.2707 m. This run does not reproduce the older 667 m value.

Actual GPU EEPS entry records at steps 1, 300, 600 and 900 were passed to
the unchanged strict native CPU routine. The maximum PBLH differences
from the recorded fast GPU results are 0.0198, 0.0530, 0.1455 and 0.0789 m;
KPBL agrees in every saved column. At the new largest-gap column,
native EEPS on the actual GPU entry gives 3821.2280 m and the GPU gives
3821.2085 m. The whole native forecast gives 3485.6689 m there.

A counterfactual using the native end-history temperature and pressure
profile lowers native EEPS's diagnosis on the saved GPU column by
407.177 m. Replacing only surface fluxes changes it by 0.0513 m.
Both profiles cross the same model level in this new run; the diagnosed
height changes through the bulk-Richardson interpolation. The native
history is 12 s later than the PBL entry, and lacks three surface fields
that were held at their saved GPU values. This establishes profile
sensitivity rather than exact native-entry agreement for that
counterfactual. The retained original-case recapture below supplies actual native-entry controls for the older investigation.

The exact module replay, complete native histories, actual GPU entries,
native CPU replay inputs and outputs, counterfactuals, scripts, hashes and
build configuration are permanent paper evidence under
`~/weather/data/datasets/paper/wrf-ports/spd-retime-pbl/bl16/`.
The complete cause packet is `new-wrf471/eeps-complete-cause.tar.zst`,
SHA-256 `ec6fba4418aeef3de3c9ba04858832dd3bf8374a056507a5170bf081a589928f`.
Rust-rendered temperature, wind and PBLH sheets were inspected. The broad
patterns agree; differences occupy small southern and western patches.
The lane report gives the commands, configurations and evidence paths.

The original-case recapture reproduces the native hour-three history hash and
all 22,500 PBLH words in the supplementary native observer. On an RTX 4090 the
new maximum is 591.5315 m at cell (12,17), rather than the historical RTX 5090
667 m. At that cell native and GPU PBLH/KPBL are 4658.0586 m / 17 and
4066.5271 m / 16. Both fast and strict EEPS give the recorded PBLH on either
fixed input set. Swapping only the native temperature into the GPU entry gives
4564.8774 m / 17; swapping the native surface fields gives 4289.2769 m / 16.
Across all columns, same-input fast/strict PBLH differences are at most
0.1036 m on native entries and 0.1067 m on GPU entries. The large reproduced
gap therefore comes from the evolved input state and its native Richardson
crossing, rather than an EEPS PBLH translation defect.

The first departure is the existing fast EOS pressure arithmetic before EEPS,
at most 3.125 Pa at initialization. Strict EOS reproduces the native pressure
words. This identifies the first difference, not a unique cause of all later
coupled-state growth. The historical exact 667 m magnitude remains unreproduced.
The original native history, full actual entry/output records, all four original
captures, both mode controls, counterfactuals and hashes are retained in
`bl16/old-run-cause/` under the permanent lane dataset. The detailed source and
replay account is in `wrf-comparison/eeps-port.md`.
