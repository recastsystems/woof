# Noah-MP selected-option development referee

This builds verbatim BTRAN and RSURF/RHSUR blocks from WRF v4.6.1's
Noah-MP submodule, commit `848f54ad3d28c4303151fe5ad83724e232694422`.
`build.py` refuses any other source SHA256, records the extracted blocks,
compiler and flags, and rejects vector-libm calls. The source and tables are
part of WRF's public-domain distribution. The original notice is retained in
`licenses/LICENSE-WRF-public-domain.txt`.
`source_pin.json` records the public distribution, source and table URLs,
sizes and hashes. Official WRF v4.7.1 pins the older-layout Noah-MP commit
`e5c0859874407859936739e8be8741f9aed369ee`; its BTRAN and RSURF/RHSUR blocks
are byte-identical to this referee's pin. Neither inspected v4 LSM contains
SNICAR. The separately recovered v5 SNICAR and split-runoff implementations
are excluded because their source pin carries sale restrictions. That hold
does not apply to the eligible v4 option branches.

Download the original `src/module_sf_noahmplsm.F` from that exact NCAR/noahmp
commit into a development scratch directory. Run from the engine root:

```sh
GPUWM_WRF_EXACT=1 WOOF_NOAHMP_NATIVE_LSM=/absolute/path/module_sf_noahmplsm.F \
  python -m pytest -q -s tests/test_noahmp_btran_options.py
```

The 35 constructed columns vary root depth, soil properties, water below and
above saturation, dry caps, urban caps and snow cover. BTR=1/2/3 each compare
105 defined native words. The 70 inactive BTRANI words per selector are left
unassigned by native ENERGY, witnessed with a sentinel. The existing device
convention writes zero there. RSF=2/3/4 each compare 70 defined words. A
negative control requires BTR=2 and BTR=3 to produce different answers.

The leaf checks above use constructed inputs. The real-input replay covers
BTR2/RSF2, ALB1/BTR3/RSF3, RSF4, each SNF2–5 and each DVEG1/2/3/5/6/7/8/9 in separate
captured native configurations. Every configuration compares all 153 formal
OUT/INOUT fields: 980 words from four strict native NOAHMP_SFLX calls,
zero differing words and zero exclusions. Inactive land HCPCT slots are
initialized by native and are included. The observer borrows already produced
arrays only when explicitly requested. Expected native outputs never supply
GPU inputs. The tool validates the native source, captured option globals and
stock namelist hash before running the slab.

```sh
GPUWM_WRF_EXACT=1 python -m tools.noahmp_options_wrf461_oracle.replay \
  --metadata /absolute/evidence/real-noahmp-observer.json \
  --captures /absolute/evidence/native-captures \
  --namelist /absolute/evidence/native-captures/namelist.input \
  --receipt /absolute/evidence/replay-result.json
```

The replay admits every implemented RUN and DVEG selection, INF1/2, CROP0/1,
and IRR0/1/2/3 with every method. Its SNF4 input
uses all actual native microphysics precipitation species. `atm.py` additionally
builds the complete verbatim native ATM procedure and compares 884 words per
SNF selector across threshold adjacencies and mixed precipitation species.

The original option configurations completed a three-hour 2000-01-24 forecast on a
73×60×34 real WRF grid. Crop and irrigation use a separate 2024-05-25
150×150×49 real grid; GLA2 uses a 2026-10-24 Greenland 60×60×49 grid.
The lane report identifies each qualified configuration and its exact evidence. Hourly temperature, wind and downward-shortwave Rust
comparison sheets were examined beside native forecasts and archived station
observations. This establishes a limited real-case comparison. It does not
establish general forecast skill. Station winds are often calmer than either
model; later local temperature, wind and offshore cloud differences remain.
There are no station shortwave observations in this comparison.

Selected classification and parameter gathering use fixed-grid device kernels
with current land, ice, vegetation and soil identities. Complete vegetation/soil
table pools are prepared once. Urban masks and native soil remapping stay on
the device. Masked glacier columns use a fused resident CUDA driver. Device
health checks join the existing deferred ledger. There is no selected per-step
host census or category readback. Normal default kernels and numerics stay
unchanged; strict numerics are the proof opt-in.

Production restart/history round trips, checkpoint option seals, actual
PreparedEnsembleSession two-member trajectories and both coupled nest feedback
modes are covered. Nest tests include mixed glacier/land CUDA graph capture,
current category mutations and all 56 allocated native-feedback carriers.
The whole real single-domain checkpoint resume matches all 492 compared
history arrays and 480 checkpoint arrays. A complete real nested BTR2/RSF2
forecast with output3 also completes three hours in both feedback modes,
180 parent and 540 child steps. Checkpoint continuation matches all 341 history
and 335 checkpoint arrays per domain/hour/mode. Cold native child input is
admitted only at START_DATE, preserving the stepped-state refusal. The lane
report retains native completion, exact arrays and directly reviewed Rust sheets.

The full real land-stage timer includes layout, parameter gathering, column
physics and write-back, with two warmups per arm and eight paired samples.
Its reference is the existing canonical option configuration on the same
saved real input. No selected physics-cost exception is claimed. SNF3 first
exceeded 1.2×; profiling found repeated urban classification launches. Fusing
classification removed that overhead without changing forecast array words.
Per-configuration timing samples, native captures, source notices, forecast
commands, maps and visual-review hashes are retained in the coordinator's
`WRF-PORTS-2026-11-01/evidence/port-noahmp-options` handoff. Runtime frame
prices bind the complete actual translation-unit identity and measured
sm_120/NVRTC12.9.86 fast/strict attributes; changed units use an assumed bound.

The stage timer is bounded to 65,536 constructed columns and 20 CUDA-event
repetitions per branch. It queues each repetition without a host sync between
them and reports all samples, the median and the ratio to option 1:

```sh
GPUWM_WRF_EXACT=0 GPUWM_NUMERICS=fast \
  python -m tools.noahmp_options_wrf461_oracle.benchmark
```

The native Dynamic VIC diagnostic uses one constructed column and verbatim
procedures from the same pin. With only the compiler's initialization policy
changed, zero initialization completes and signaling-NaN initialization traps
in RR1 when it reads YD before assignment. Both run the invalid-operation trap.
The tool requires that precise outcome and records the compiler, flags,
source and block hashes, inputs, native exit status and backtrace:

```sh
python -m tools.noahmp_options_wrf461_oracle.dvic_undefined \
  --lsm /absolute/path/module_sf_noahmplsm.F --work /absolute/owned/scratch
```

This constructed witness identifies the native unassigned local passed at
LSM:8263 and read at RR1:8421. The complete Dynamic VIC port below retains a
named quiet-NaN residue there. Real rainy native calls compare every formal
output without exclusions; their separate captures qualify each RUN8 method.

INF2 uses the native nonlinear frozen-soil conductivity with actual liquid
water and column maximum ice throughout SRT. RUN4 uses the complete native
BATS surface saturation, repeated soil solve and frozen-fraction bottom
drainage. Both real captures compare all 980 words without exclusions, and
both stock namelists complete the same three-hour forecast and twelve
examined Rust comparison sheets. Each has actual nested feedback 0/1,
checkpoint continuation, next-step restart, Rust history and two-member
ensemble evidence. They share the standard soil and runoff carriers; neither
needs a new prognostic state. Default WATER source and launch stay unchanged.


RUN6 and RUN7 implement the complete native VIC and Xinanjiang runoff
branches, including every repeated fine soil-step evaluation after soil
moisture changes. They use actual transferred BVIC, SMCREF, AXAJ, BXAJ and
XXAJ parameters. The pinned native top-soil storage uses cumulative ZSOIL
weights; this ordering is preserved. Xinanjiang can raise a positive base
to exponent zero when BXAJ=1; the selected extension handles that native
powf case as exactly one. The unchanged default translation unit is retained.
Each selected real native configuration compares all 980 words without
exclusions. Additional complete WATER boundary replays cover INF1/2 crossed
with RUN3/4/6/7, 2,556 compared words each, including frozen, wet and urban
cases. Forecast, hourly Rust visual, timing, actual ensemble, nested feedback,
checkpoint, restart and history receipts are delivered with the lane report.

RAD2 implements native dense-canopy gap fractions in the full ordered two-stream
calculation. VAI=0 keeps the native open-canopy value. The stock RAD2 namelist
uses four actual sunlit columns (COSZ > 0.1), comparing all 980 formal output
words with no exclusions. The observer records OPT_RAD independently of the
namelist. Additional complete native TWOSTREAM/ALBEDO boundaries compare 720
words without differences. CLASS and BATS combinations both have actual nest,
restart, checkpoint and Rust history coverage. RAD2 adds no prognostic state
or parameter table; the default radiation translation unit stays unchanged.

RAD1 uses the complete native crown geometry gap calculation, with actual transferred RC/HVT/HVB. The selected radiation unit appends the already shipped MIT Arm/CORE-MATH trigonometric helpers and retains their notices. Four actual sunlit native calls match all 980 formal output words; 24 two-stream and ten albedo boundary cases additionally match 720 words. The complete three-hour forecast, twelve directly reviewed Rust comparison sheets, complete-stage timing and actual lifecycle/ensemble receipts are in the lane report. The default radiation source is unchanged.

FRZ2 uses the complete native iterative FRH2O calculation, including its iteration bound and fallback, in selected resident PHASECHANGE. The default thermal unit is unchanged. Exact source and real-capture receipts are retained in the lane report.


STC2 ports the complete fully implicit HRT top boundary, leaves the energy
iteration unreset until TSNOSOI, and applies the native post-solve snow surface
temperature correction. STC3 carries actual fractional snow cover into both
flux leaves. TBOT1 supplies native positive zero at the lower heat boundary.
The resident glacier driver carries both thermal selectors. Default CUDA
translation units and unselected runtime identities remain unchanged. Each
selector has a real strict module replay and a complete stock-namelist forecast;
the retained snowy STC3 capture selects actual above-freezing snow surfaces.
The lane report lists commands, timings, maps and persistence evidence.

Native proof may explicitly select `--tanh-profile glibc_2_43` or
`--atan-profile glibc_2_43` when the captured native binary uses that generation.
The receipt records both selections. The existing 2.39 arithmetic remains the
default independently of the execution host. Both profiles are also explicit
`NoahmpRuntimeParameters` metadata and are sealed across restart/checkpoint.
The newer tanh uses the already retained MIT CORE-MATH source, whose original
notice and source revision are in `sfire_libm.cuh` and this oracle's source pin.

CRS2 ports the complete native CANRES/CALHUM canopy-resistance branch in both
sunlit and shaded foliage, using actual transferred RGL/RSMIN/RSMAX/TOPT/HS
values. Photosynthesis follows the native -999.99 diagnostic for this branch.
The four real native calls compare all 980 formal words with no exclusions.
The lane report retains the stock-namelist forecast, maps and complete land-stage timing.

SFC2 ports native SFCDIF2 into vegetation and bare-ground stability iterations,
using actual THAIR and transferred CZIL, native wind-speed coefficient rescaling
and the bare-ground snow cap. Its Paulson functions cover the entire active
native branch (ILECH is assigned zero unconditionally). Four real native calls
match all 980 formal words with no exclusions. Default SFCDIF1 source is unchanged.

RUN1 and RUN2 retain the complete pinned SOILWATER/SRT/WATER groundwater branches, including native double-precision S_NODE, aquifer recharge/discharge and the equilibrium water-table solve. FSATMX, TIMEAN and PSISAT are actual transferred parameters; active ZWT/WT/WA are carried through cold initialization, resident write-back, history, nests and restart. Each strict real four-call replay compares all 980 formal output words without exclusions. Forecast, observation-map, timing and ensemble receipts are listed in the lane handoff.

SOIL2 gathers the four current SOILCL1–4 categories on the GPU and applies
WRF's layer-specific table values and top-layer scalar parameters. SOIL3
executes the complete native SR2006 pedotransfer with the forty actual
MPTABLE coefficients, the eight SOILCOMP layers and native zero organic
matter. Urban handling and native missing-texture defaults are retained.
These inputs are required from stock wrfinput, retained by conditional
history/checkpoint/restart inventory and passed through both nest feedback
modes. Default SOIL1 allocates none of them. The native driver parameter
comparison is separate from the all-formal SFLX replay. The additional
192-word pedotransfer comparison covers nonzero textures and organic matter,
including missing-input boundaries. Production fast numerics and the default SOIL1 image are unchanged.

The newer native LOG10 proof profile is selected explicitly with
`--log10-profile glibc_2_43`. It reuses the existing CORE-MATH MIT log10
implementation and changes only THERMOPROP's saturation-ratio logarithm.
The default old-native profile and production fast profile are retained.
The exact-input native/CUDA trace in the lane evidence identifies the
first differing conductivity word before this profile is selected.


DVEG2/5/6 use the complete WRF v4 CARBON and CO2FLUX procedures in resident
CUDA after WATER. The six prognostic pools, LAI/SAI updates and NEE/GPP/NPP
fluxes retain native initialization, actual vegetation parameters, both nest
feedback modes, restart/checkpoint and Rust history/ensemble ownership. All
three real module replays compare 980 words with zero differences; an
additional sunlit DVEG2 capture independently exercises positive photosynthesis.
The default prescribed-vegetation source and output words remain unchanged.
Exact complete-stage timings and directly inspected hourly Rust maps are in
the lane evidence. No crop or GECROS branch is implied by this qualification.

Positive Noah-MP `soiltstep` follows the pinned v4 driver: the binary32
`soiltstep/DT` ratio is rounded with native NINT, and absolute domain steps
select soil updates and accumulator resets. Canopy, snow and phase change
still execute each atmosphere step. Nine native accumulator arrays, including
four-layer ACC_ETRANI, carry the intervening energy and water inputs. Each
domain uses its own DT and clock; restart/checkpoint and native copy feedback
retain all accumulator words. The zero default retains its prior allocation
and CUDA image. Source/licence anchors are in the option oracle source pin.


TDRN1 and TDRN2 execute the complete pinned WRF v4 simple and Hooghoudt tile
procedures in resident CUDA, including all six simple layer-selection modes,
water-table finding, equivalent-depth series and native rate conversions.
Actual MPTABLE parameters are transferred per land/soil category. Native
RUN3 and fraction thresholds govern calls; selecting another RUN leaves
these routines inactive exactly as WRF does. TD_FRACTION is decoded by the
Rust NetCDF path; ZWT and QTDRAIN remain resident through both nest modes,
checkpoint/restart and Rust history. QTDRAIN retains the native driver's
`QTDRAIN += QTLDRN` arithmetic. The default allocates none of these carriers.
The optional Rust geography row accepts an explicit average_gcell ratio and
uses the existing WPS REAL sampler before interpolation fallbacks. Other rows
retain their previous sampler. The proof inputs use UCAR's actual drainage
map, masked by native XLAND; this data is not packaged with the physics.
A strict real capture exposed the CANWATER signed-zero MIN tie at L:6353;
the complete selected routine retains native second-on-equality behavior.
Per-row replay counts, figures and timings are in the lane report.

Native output level3 now carries all58 native accumulators and their complete diagnostic inputs through resident CUDA, both atmospheric epilogues, restart/checkpoint, nests, ensembles and Rust history. Levels0/1/2 keep the native standard path. `noahmp_acc_dt` is a nonnegative real interval in minutes; zero disables native accumulation. The separate pinned `module_diag_misc.F` replay compares1,856 words from32 actual captured columns at start, pre-reset, reset and post-reset steps, with zero differences and no exclusions. `generate_output.py` reads only the pinned public-domain WRF source. The lane report retains the whole real forecast, twelve directly inspected Rust panels, lifecycle/ensemble checks and1.143123× complete stage timing.


CROP1 translates the complete native GROWING_GDD, CO2FLUX_CROP and CARBON_CROP
procedures, with actual crop-table parameters, corn/soy selection, calendar,
phenology, all growth pools and grain/GDD. Actual five-plane UCAR crop geography
and seasonal GDD are read by the Rust data path; native cold initialization and
vegetation fraction overrides are retained. Six actual positive-crop calls match
all 153 formal outputs, 1,470 words, zero differences or exclusions. The complete
three-hour May forecast, twelve directly examined Rust sheets, two independent
ensemble trajectories, both nest modes and restart/checkpoint/history qualify
this row. Complete stage timing is 10.697952/9.576592 ms, 1.117094×.

IRR1/2/3 implement every native trigger and method, including mapped fractions
at IRRM0, sprinkler interception/evaporation, drip and flood injection. Integer
event counters and all amounts/rates remain resident and persist as native
Registry fields specify. Mapped and both sprinkler-trigger configurations each
match 2,940 formal words from twelve active irrigated crop calls; drip and flood
each match 980 words from four active calls. The lane report binds each completed
native/GO forecast, direct twelve-sheet map review, lifecycle and ensemble proof.
Selected negative velocity squares use native multiplication. Explicit strict
LOG10/tanh/atan profiles select the witnessed native libm where required;
production arithmetic stays unchanged.

GLA2 runs the complete pinned native glacier procedure in the resident driver.
Four actual ice-sheet calls match all 39 formal outputs, 244 words, with zero
differences and no exclusions. The real Greenland case contains 2,906 active
glacier cells and completes 180 steps beside a clean native forecast. All twelve
Rust sheets at 00/01/02/03Z were examined with actual coastal observations.
Nighttime shortwave is zero in both; no interior observational skill is claimed.
Complete stage timing is 19.029936/18.012448 ms, 1.056488× against GLA1. Actual
nest feedback, checkpoint, restart, history and independent ensembles pass.


RUN5 contains the complete pinned Miguez-Macho/Fan spatial groundwater model:
eight-neighbor transmissivity and head gradients, river exchange, deep recharge,
UPDATEWTD, SHALLOWWATERTABLE and native cold equilibrium/river initialization.
Real FDEPTH/EQZWT/RECHCLIM/RIVERBED geography is required and enters through the
Rust WRF reader. WTDDT is read from &physics in minutes and uses native positive
NINT on the absolute model timestep. All evolving fields and SMOISEQ persist
through restart/checkpoint/history and independent ensembles; child feedback
copies only the subset carrying native Registry feedback flags. Neighbor flow
uses resident full-grid device arrays without per-step host synchronization.
The strict column replay compares 980 real formal words, and the separate WTABLE
replay checks all twelve formal output arrays, including captured MPI halos.
`generate_groundwater.py` checks all three public source hashes and reproduces
the complete CUDA header using its retained spatial translation. The handoff
records the complete native/GO forecasts, actual observations, twelve directly
examined Rust sheets and cadence-weighted complete-stage timing.
