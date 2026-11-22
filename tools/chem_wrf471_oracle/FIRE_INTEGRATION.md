# Fire and large-scale wet-removal integration

Reference sources are pinned in SOURCES-smoke.sha256. WRF is
f52c197ed39d12e087d02c50f412d90d418f6186. GSL ccpp-physics is
3e6660c6df54e95a0871e990c2294dd397ae3860, Apache-2.0:
https://github.com/ufs-community/ccpp-physics/blob/3e6660c6df54e95a0871e990c2294dd397ae3860/physics/smoke_dust/rrfs_smoke_wrapper.F90#L855
https://github.com/ufs-community/ccpp-physics/blob/3e6660c6df54e95a0871e990c2294dd397ae3860/physics/smoke_dust/module_add_emiss_burn.F90#L70

Run `bash tools/chem_wrf471_oracle/build_fire_oracles.sh SOURCE_ROOT BUILD_ROOT`.
The staging wrapper prevents build.sh's sources-*.list/extract-*.list globs
from consuming GSL's incompatible lists. Shared harness files are unchanged.
The KIND=4 gate uses gfortran 15.2.0, glibc 2.43. KIND=8 is a reported
precision comparison, including precision changes in driver input generation.

## Interface additions needed by the other lanes

SourceFrames supplies an aware UTC `reference_time` and
`available_hours(source, field)`, and the specified
`at(source, field, valid_time)` returns remapped host float32 arrays.
Every fire-owned source variable has metadata like:

```json
{"fire": {"frp": "related_field_key", "diagnostic": true,
          "history_source": "prior_model_source_key",
          "history_hwp": "prior_hwp_field_key",
          "history_rain": "prior_rain_field_key",
          "minimum": 0.001, "state_factors": {"ug kg-1": 1}}}
```

The default minimum is 1e-3 ug m-2 s-1 (wrapper.F90:250), held in the rule data.
Keys are data and never compared to a product or species name. Only variables
with a `fire` mapping are consumed. A gas requires its positive unit factor
as source-row data, also used to undo the factor in the emission diagnostic.
No emission source or transported species row is added by this sublane.
The source row and species row owners must declare them.

The plume lane calls `chem_fire.prepare_columns(ctx)` before its plume solve.
The driver must run plume preparation/solve before injection consumes cached bounds, including the first step.
The return mapping has kpbl, kpbl_thetav, uspdavg2d, windgustpot, hpbl2d.
It does not require HWP-only inputs. The outputs use GSL's 1-based indices.
Publish cached k_min/k_max/flam_frac and a host boolean plume_ran through met.
Export `ebu_distribute(ctx, emission_flux, output_volume)`. The distribution
must preserve the column flux within levels 1..51 because GSL's injection
limit is 51. Fire reruns distribution on either a plume call or an hourly
source-frame change. GSL only reruns it on plume calls; hour-aligned
60-minute plume calls give the same cadence.

ChemContext must expose model map factors msftx/msfty, bottom-face w, and
terrain oro. HWP maps t2m to physics.fields['t2'] (physics.py:339),
dswsfc to swdown, snow to the engine's SWE field in mm, and pb2d to pblh.
Explicit dpt2m in Kelvin is required. The engine's diagnostic at
surface_moisture_ledger.py:100-123 is binary64 Bolton dewpoint; this lane
refuses its use as an unqualified float32 GSL input. FV3 wetness is not
Noah volumetric smois, so no substitution is made. totprcp and the prior
rain history are metres, as in the GSL HWP precipitation expression.

LANDUSEF exists in metem.py:80 and static/build.py:1565. This export does
not carry it in the physics driver's fields, which have dominant ivgtyp.
Publish category-first vegtype_frac and landuse_dataset='IGBP-MODIS' to
admit daily-cycle fire classification. Dominant categories never become
fabricated one-hot fractional land use. All category sets, thresholds,
coordinate bounds and curve constants are in fire_type_rules.json.
Its SHA256 must join the prepared-state process-input identity; the current
catalog digest only covers the species/sets/sources/diagnostics directories.

The Rust array reducer follows:
https://raw.githubusercontent.com/ufs-community/ufs-srweather-app/2ad2bc5731819cdc5aef95372a8b27ffe417bf40/ush/smoke_dust/core/cycle.py
Lines 53-57, 228-230 select 24 hours from initialization minus 25 hours
through minus 2 hours. Lines 252-307 use prior model HWP/rain and latest
positive-FRP fire time. Lines 330-376 divide by emitting-hour count, using
two for one emitting hour. This is not an all-24-hour arithmetic mean.
The already remapped emission mass replaces FRE-to-emission conversion,
which belongs to the source ingest lane. Invalid positive-FRP masses are
refused instead of contaminating a reduction. Finite HWP samples are
averaged; positive finite rain samples are summed. Unlike the reference's
nanmean, nonfinite HWP values including infinities are omitted.

The source preprocessing at lines 240-250 refuses absent prior model
history, or explicitly writes dummy emissions if requested. This port
refuses a cold daily-cycle start to prevent invented HWP or silently zero
emissions. The array-only reduction is `gpuwm_fire_daily` in the CPU
preprocessing bridge (tools/grib1_bridge/src/chem_fire_reduce.rs, built and
staged with libgpuwm_preprocess_cpu like every other preprocessing export),
found through the bridge's own resolution ladder. No Python decode, remap or reduction
is introduced; NumPy stack/contiguous copies only marshal bridge arguments.

## Numerical choices and omitted wrapper lines

wetdep_ls, chem/module_wetdep_ls.F:49-91, initializes every scratch read.
chem_driver.F:778 gives nz mass levels. Sum levels 1..nz-1 and remove
1..nz-2, using moist rho, rainncv mm/step and bottom-face w. Alpha is a row
parameter; zero is skipped. The oracle prelude uses chem_opt=300's
GOCART_SIMPLE order (registry.chem:4022), numgas=4. The added removed-mass
output is dry-air mass, and never books the driver's ledger a second time.

rrfs_smoke_wrapper.F90:855-864 leaves kpbl unwritten above the column.
Poisoning it to -777/+777 exposes this. Line 956 reads kpbl+1, so a
source kpbl=kte reaches outside mass-level winds. Initialize/cap kpbl to
nz-1 in both the port and defined fixture. Keep the source's comparison
of z_at_w against pbl and its divisor real(kpbl); do not reinterpret either.
The fixture publishes raw_kpbl as evidence separate from defined output.
The prep driver poisons other scratch both signs; defined outputs match.
The compiler-poison build uses signalling NaNs and integer -98765 and all
678 fixture files match the baseline, including manifests and both kinds.

GSL add_emis_burn:132-159 carries coef_bb_dc for type zero and throughout
ebb_dcycle=1; fire_hist carries its night caps. Initialize both to one at
ktau=1, preserve them thereafter and across restart. conv is undefined
outside ebb_dcycle 1/2, so those are the only admitted arms. kfire_max=51
is preserved with a shorter-column refusal. Debug-only dc_gp/dc_fn are
unwritten in the source's enabled debug logging; the oracle disables
dbg_opt and the port has no such logging. icall's saved debug counter has
no physical effect. q_vap is untouched with add_fire_moist_flux false.

Wrapper extraction takes 365-370, 375-403, 552-556, 620, 855-864, 922-974 and 985-1013.
Every other wrapper line is omitted: 1-364, 371-374, 404-551, 557-619 and 621-659
are the CCPP/MPI driver, other processes, plume/coupling calls and diagnostics; 660-854 and
865-921 are declarations, initialization and FV3-to-column preparation
(the engine already supplies column met); 975-984 loads prior rain through
source frames instead; 1014-1071 is source loading, initial carry and chem
copying implemented in host orchestration/ktau initialization. Remaining
1072..EOF code is outside the fire prep routine. The source's peak_hr
bands are retained as data but are unused by the active cycle arithmetic.

The source's curve denominators are the folded float32 words represented
exactly in JSON; CUDA reads them, and never refolds the expressions.
CPU transcendentals use noahmp_libm's scalar glibc float32 transcriptions.
The measured fixtures match glibc 2.43 despite that module's 2.39 origin.
CUDA uses glibc_flt32.cuh and explicit rounding. No fast math is used.

GSL rho_phy is p/(Rd*T), wrapper.F90:889, not WRF-Chem moist rho.
For production kg-to-ug/kg-dry mass identity, injection uses dryrho as its
denominator. The transcribed conv/add/clamp association is unchanged;
oracle tests use the explicitly supplied Fortran rho. This density choice
is a WOOF-ArWen addition and needs the integrated mass-ledger test on a node.
The cached ebu output carries coef scaling, restored through the source's 1e-4 floor before the next call (wrapper.F90:365-370,552-556). Below that floor, reuse loses flux exactly as GSL does.
The 5000 clamp can discard source mass, and changes to an existing tracer
below the 0 clamp also affect the net emitted diagnostic. Conservation
tests apply where neither clamp clips the prescribed emission.

EBU_IN is the GSL scalar input name, used as a WOOF-ArWen 2-D diagnostic.
WRF Registry's ebu_in at line 421 is a container; its PM2.5 mass field is
ebu_in_pm25 at line 433. This lane does not claim a WRF scalar smoke output
of that spelling. FRP_MEAN follows GSL's coefficient-scaled diagnostic at :620, retaining the MW-to-W-to-MW rounding association. The plume input at :451-455 separately uses the previous coefficient, a type-4 sc_factor and an FRP cap; the plume lane must retain that behavior.
Plume orchestration must convert
to W if its plume input requires the GSL conv_frp=1e6 (wrapper.F90:735).

## Shared integration rows

Digest/frame modules: chem_fire (chem_fire_prep, chem_fire_classify,
chem_fire_cycle, chem_fire_inject, chem_fire_units, chem_fire_carry, chem_fire_frp_diag); chem_wetdep_ls
(chem_wetdep_ls). Record device frames on sm_89 and sm_120; CPU-only NVRTC
compile receipts are not frame recordings or device parity.

Allocation bounds use C=ny*nx, R=active rows, Z=nz. Every float32/int32
2d plane is 4*C bytes; rows_2d is 4*R*C; rows_3d is 4*R*Z*C.
All allocation names and restart/output classes are declared in ALLOCATES.
Fire ebu, coef_carry and hist_carry each use rows_3d; the carry plane index
is the emission reference index and is refused beyond Z. This spends two
full row volumes on usually sparse carry; reducing it needs an allocation
shape for reference planes in aq-core. Wet work is rows_3d; alpha and
removed_mass are rows_2d. Five fire constant tables total 1,984 device
constant bytes, with bounds 64/48/160/160/64 float32 words. Classification
has 16 local float32 group words per thread. There are no transient CuPy
array allocations in either production step.

Census tests: tests/test_chem_fire_emissions.py and
tests/test_chem_wetdep_ls_wrf471_parity.py. Imports/core packaging:
woof.core.chem_fire, woof.core.chem_wetdep, woof.chem_fire_reduce (its Rust half in libgpuwm_preprocess_cpu);
dependencies datetime, json, pathlib, numpy, ctypes, os, and lazy cupy
through the shared kernel loader. Verify imports add noahmp_libm and
chem_oracle. Pin fire_type_rules.json in process-input identity. Ship the
Rust bridge and GSL Apache attribution with these modules.

CHANGELOG text:
Add table-driven fire emission timing, GSL float32 fire preparation and
diurnal injection, and WRF large-scale wet removal with pinned CPU oracles,
restart carry, conservative source units, and explicit refusals for missing
fire inputs.

NOTICE text: Fire preparation and diurnal injection transcribe NOAA GSL
ccpp-physics smoke_dust at 3e6660c6df54e95a0871e990c2294dd397ae3860,
Copyright 2017 NOAA, UCAR/NCAR CU/CIRES, licensed under Apache-2.0.

## Verification limits

Device launches, kernel parity mutations, timings, integrated SourceFrames,
plume-cache interaction, restart round trips, and driver ledger closure
were not run. The corresponding sibling/core modules are absent in this
export. No node GPU or local GPU was used. Only a development machine CPU work was used.
No frozen core file or shared harness file was edited. The kernel loader's
single extra-header row and its closed allow-list test row are the only existing code-file changes. No commits,
tags, pushes or uploads were made.
