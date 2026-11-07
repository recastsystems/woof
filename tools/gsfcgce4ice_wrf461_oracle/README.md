# WRF 4.6.1 MP7 oracle

The active non-chemistry `gsfcgce_4ice_nuwrf` entry comes from public WRF
v4.6.1, commit `d66e442fccc04111067e29274c9f9eaccc3cef28`. Source digests,
licences and the identical v4.7.1 source are recorded in [SOURCE.md](SOURCE.md).
The runtime contains all active column processes, separate graupel and hail,
latent heating and accumulators, six effective radii, five sedimentation fluxes,
and the entry's hail-inclusive `saticel_s` reflectivity.

[qualification-skeptic2.json](qualification-skeptic2.json) records the current
retained native capture, exact replay and stock moist-theta integration repair.
The original [qualification.json](qualification.json) records the earlier native capture,
strict replay, six-hour `woof go --wrfinput` forecast, restart continuation,
option-off byte comparison and timing evidence. The associated lane report
retains Rust comparison maps with native WRF, MRMS and surface observations.
This is implementation qualification. The observed storm-initiation errors
in the case do not establish observation skill or a validated forecasting
configuration.

## Real native captures

Instrument a **copy** of the pinned module. Use one tile per MPI rank and
`nproc_x=1`, so exactly one rank's tile crosses the selected j index. Capture
clocks are native timestep indices, not independently generated columns:

```sh
python tools/gsfcgce4ice_wrf461_oracle/instrument_capture.py \
    WRF_SOURCE_ROOT/phys/module_mp_gsfcgce_4ice_nuwrf.F \
    OWN_BUILD/module_mp_gsfcgce_4ice_nuwrf.F \
    --j-index 75 --timesteps 1 300 600 900 1200 1500 1800
```

Compile this module into an owned native WRF build with
`-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`, preserving the
build's WRF service modules and IO flags, then link `wrf.exe`. Initialize
MP7 with native `real.exe`, including its actual hail and lateral-boundary
fields. Run native WRF normally. The instrumentation records the complete
native tile at entry and exit without changing a physics expression.
The original run used two MPI ranks. The retained second-skeptic capture
uses four MPI ranks; the new tile packets and reference outputs are their own
native trajectory. Other reused WRF objects retain their stock build flags.
[The current receipt](qualification-skeptic2.json) records capture locations,
hashes, strict replay and the stock moist-theta integration repair. Never
delete the native packets behind a replay claim.

Copy completed `case-real-*.bin` and `case-real-*.expected.out` files to
`ORACLE_DIR/cases`, then replay the production launcher:

```sh
GPUWM_WRF_EXACT=1 GPUWM_WRF_GLIBC=2.43 \
    python tools/gsfcgce4ice_wrf461_oracle/run_gpu.py \
    ORACLE_DIR --json strict.json
```

`GPUWM_WRF_GLIBC` names the native referee's libm generation. Strict mode
uses no FTZ or fused multiply-add and compares every returned word, including
signed zero. For glibc 2.41 and newer, reflectivity's `log10f` uses the
MIT-licensed CORE-MATH authority also used by Rust `rw-libm`. Its notice is
retained in `gsfcgce4ice_log10.cuh`. Default forecasts use CUDA `powf`, `expf`
and `log10f`; strict arithmetic is an explicit proof option.

## Generated regression fixtures

`build.sh` links an existing WRF scheme object and checks the pinned source.
`build_standalone.sh` compiles the two pinned modules and compares their
complete outputs with the **historical** `scalar-reference.json` artifact.
Those historical hashes depend on the original toolchain/libm; a changed
platform can fail that artifact check and must supply a fresh native real-run
capture instead of treating the old output as a new oracle.

```sh
bash tools/gsfcgce4ice_wrf461_oracle/build_standalone.sh WRF_SOURCE_ROOT ORACLE_DIR
GPUWM_WRF_EXACT=1 python tools/gsfcgce4ice_wrf461_oracle/run_gpu.py ORACLE_DIR --json strict.json
GPUWM_WRF_EXACT=1 python tools/gsfcgce4ice_wrf461_oracle/run_gpu.py ORACLE_DIR --plant --json plant.json
python tools/gsfcgce4ice_wrf461_oracle/run_gpu.py ORACLE_DIR --json default.json
```

Generated fixtures have 80 columns, 40 levels and 13 calls covering hail
convection, stratiform cloud, snow, warm rain, dry columns, high terrain and
edge values. Their planted-change check moves the splinter-process constant
by one float32 bit and requires rejection. Default-mode tolerances are in
`check.py`; they grade port arithmetic, not forecast skill. Historical
reflectivity residuals on glibc 2.43 preceded the corrected strict logarithm
and are superseded by the real-run replay receipt.

Admission, transported species, history names/units and radiation aliases
come from the registry. Same-scheme nests transport hail in both feedback
modes. Mixed-scheme edges retain a named refusal because hail-water folding
between unlike schemes has not been validated. The production member driver
keeps each ensemble member's hail and carried heating independent. MP7 uses
the original GPU member path; it does not claim packed microphysics execution.
Native fused reflectivity is not a stand-alone differentiable observation
operator. Initial history preserves WRF's allocated zero reflectivity and
never advances microphysics to manufacture a diagnostic.

`prepare_cli.py` remains an older 60-second smoke helper. It adds missing
hydrometeors to an existing initialization through Rust IO; it is not the
native `real.exe` initialization used for qualification. `inspect_forecast.py`
uses Rust decoding to retain field hashes/ranges and check requested output.

## Standalone helper scope

WRF comments out `negcor` calls at lines 410–433. `auto_conversion` callers
are inside WRF_CHEM sections. `refl10cm_gsfc` is defined at line 4781 and is
never called by this entry. These helpers have separate launchers; no disabled
call or chemistry coupling is added to a forecast.

```sh
bash tools/gsfcgce4ice_wrf461_oracle/build_helpers.sh WRF_SOURCE_ROOT ORACLE_DIR
GPUWM_WRF_EXACT=1 python tools/gsfcgce4ice_wrf461_oracle/check_helpers.py ORACLE_DIR --json helpers-strict.json
python tools/gsfcgce4ice_wrf461_oracle/check_helpers.py ORACLE_DIR --json helpers-default.json
GPUWM_WRF_EXACT=1 python tools/gsfcgce4ice_wrf461_oracle/check_helpers.py ORACLE_DIR --plant --json helpers-plant.json
```

The helper build widens module visibility only and checks unchanged active
entry output bytes. The helper fixtures include 80 radar columns, 160
conversion inputs and five correction tiles. WRF leaves effective radius
undefined on five early-return helper inputs; those unwritten words are named
and excluded, while rates remain graded. Standalone radar preserves WRF's
unfloored clear-air result and rain/snow/graupel species. Its table is built
once per device from pinned source-initialization constants, with no external
table files. The active entry continues to use its hail-inclusive reflectivity.
