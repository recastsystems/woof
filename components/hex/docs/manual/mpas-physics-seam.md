# MPAS-A v8.4.2 physics seam

Authority is the pinned MPAS-Model commit f34984b and pinned MMM/UGWP sources.
Both suites with every slot on the native backend are bitwise against native
MPAS-A v8.4.2 in every written field (`x1.2562`, 24 h, one process), and so is
the `convection_permitting` forecast with all ten active slots on hex-cuda;
the gate decks are in `verification/gates/`. The WOOF default and its pinned
source/receipts retain their existing contract.

## Slots and registration

`woof.hex.mpasphys.driver_v842.resolve_namelist` reads `&physics`, takes Registry
defaults, resolves `suite` selectors from the existing MPAS suite table and
applies `mpas_atmphys_control.F:92-332` checks. The table block `mpasphys_v842`
routes `mesoscale_reference`, `convection_permitting`, and `none` to the new
backend. `route_namelist` selects an explicit native suite through this table,
returning `None` for the existing default or WOOF suites. The public forecast door consumes these explicit suite routes.

| Slot | MPAS selector | Position |
| --- | --- | --- |
| cloudiness | config_radt_cld_scheme | before radiation, either radiation alarm |
| radiation_sw | config_radt_sw_scheme | before LW, SW alarm |
| radiation_lw | config_radt_lw_scheme | after SW, LW alarm |
| surface_layer | config_sfclayer_scheme | every physics-driver call |
| oml | config_oml1d (logical) | after surface layer, before land surface |
| land_surface | config_lsm_scheme | after surface layer and OML |
| pbl | config_pbl_scheme | after sea ice and surface diagnostics |
| gwdo | config_gwdo_scheme | after PBL |
| convection | config_convection_scheme | after convection step1, convection alarm |
| microphysics | config_microp_scheme | after RK transport, dry-theta recovery and clamp |

Each slot's backend is `off`, `native-cpu`, or `hex-cuda`. At the forecast door a
v8.4.2 suite runs every slot on `hex-cuda` (the table row's `default_backend`). A
slot whose selected scheme this install cannot run on the GPU (no GPU slot, a port
withheld from the distribution, or a resource packet `woof hex physics-resources
build` does not make) runs `native-cpu` instead, and the run names it at start-up
(lines beginning `PHYSICS ON CPU:`) and in its receipt (`cpu_slots`).
`--physics-default-backend native-cpu` puts every slot that names no backend of its
own on `native-cpu`: the native referee. The library resolvers called without a
`default_backend` (module replays) keep every slot on `native-cpu` and record no
`cpu_slots`. Missing active registrations fail before any physics initialization or
field mutation.

Backend overrides are read from `&hex_mpasphys`; native `&physics` retains the
MPAS option names. Each table row names its `backend_selector`, such as
`config_convection_backend='hex-cuda'`. Without overrides every slot takes the
route's default above. An explicit adapter override must agree with the namelist when
both name the same slot. The real forecast door passes the native namelist and backend group to this
resolver before the existing WOOF path.

A selection of Noah-MP v5.0 runs only with `--licence-opt-in ucar-noahmp`
(`woof.hex.licence_gate`); UGWP and the fork's NOAA GSL physics are not gated
(owner ruling 2026-10-14: US Government code). The library resolvers are not
gated, so module replays are unchanged.

Register a launcher with `registry.register(slot, mpas_scheme_name, 'hex-cuda',
launcher)`. The callable accepts one immutable `SlotCall`: slot, scheme, backend,
native one-based `itimestep`, elapsed `time_seconds`, scheme `dt`, manager flags,
shared `pools`, and all resolved MPAS `options`. It updates that slot's named
pools and tendencies in place. Keep persistent state in `pools`, so the common
restart adapter can preserve it. A callback must not retain hidden mutable
numerical state absent from restart. It must not compute another slot's physics.

Framework glue uses `DriverHooks.invoke(operation, call)`. `NativeDriverHooks`
binds the f3 ABI with exact named arguments and registers native slot callbacks
from each scheme's table `native_entries` row. The argument provider synchronizes
input pools; `after_call` writes changed pools back. Native array allocation and
release operations remain explicit in MPAS order. Microphysics allocation is
inside its pristine driver. f3 operations not in the ABI require explicit
external glue; missing operations fail, with no silent substitute.
Preparation/coupling/update kernels supply CUDA glue. The slot switch
changes only a scheme callback. Native glue must not call the all-scheme native
driver as a substitute: that would execute hex-cuda slots a second time.

`NativePoolArguments` supplies exact ABI arguments from populated pools,
one-based solve extents, step-start time, and native calendar metadata. It
selects time level 1 for pre-dynamics preparation and level 2 for microphysics.
`NativeCalendarState` calls `native_manager_state` in a shim linked to the same
strict native library. The shim uses native timekeeping and manager arithmetic,
sets the eight shared logical flags, and returns timestamp/Julian/leap metadata.
Advance the native clock exactly once at the native outer-step clock boundary.

## Time ownership

`physics_timetracker_state` updates native calendar globals and logical flags
from the manager before surface updates; native framework glue owns its date
arithmetic. The pre-dynamics driver gate excludes microphysics alone, cloudiness
and GWDO (driver.F:179-184); manager and post-transport glue still run whenever
native `moist_physics` is true.

The manager creates independent LW, SW and convection alarms at model start.
`none` for these intervals means every step with the dynamical dt. Thresholds
use the framework `<= now` rule, not timestep modulus; reset skips missed rings
while preserving phase (`mpas_timekeeping.F:490-566,876-985`). PBL interval sets
`dt_pbl`, but pristine manager creates its alarm without checking it: PBL runs
every driver call (`driver.F:319-330`, `driver_pbl.F:841,908`).

Greenness and optional surface boundary updates precede `MPAS_to_physics`.
CAM absorption updates start immediately; CAM restart and rain/radiation bucket
alarms start after one interval and query/reset with dt lookahead. Diagnostics
reset at now and query with lookahead. Bucket flags are passed to schemes and
update glue. Radiation accumulation runs every driver call when bucket updates
are enabled and threshold is positive, even when radiation alarms do not ring.

Native driver elapsed `xtime_s` is step-start time, including for post-RK
microphysics. Native Fortran NINT and RKIND rounding control microphysics
substeps and `n_cu`; native adapter must preserve them (manager.F:681-708).

## Arrays and precision

The real-door input boundary reads the initialized mesh over the grid, including
`fEdge` and `fVertex`. Rust prepares every package-selected Registry scalar in
Fortran order, including carriers absent from the initialization file. The seam
copies those complete words into contiguous CUDA storage before driver creation;
native microphysics initialization then supplies its prescribed initial values.
Scheme launchers must use the selected scalar row rather than require each
constituent to exist as an input variable (Registry.xml:1701-1754).

Before native physics or dynamics initialization, the seam resolves the native
core's horizontal-length assignment: a positive `config_len_disp` sets
`mesh.nominalMinDc`; otherwise a positive input `nominalMinDc` sets the live
native config. These are scalar copies preserving RKIND words, with authenticated
namelist bytes retained (mpas_atm_core.F:156-200). CUDA binders read the resolved
native config rather than calculate a replacement.

CUDA public mass arrays are contiguous float32 `[level, cell]`, interface arrays
`[interface, cell]`, scalar block `[species, level, cell]`, edge arrays
`[level, edge]`, surface arrays `[cell]`, and counters int32 `[cell]`. Vertical
order is surface to top. Solve extents exclude halos; total extents include
resident halos. MPAS's final dummy cell/edge is adapter-owned. Scalar ordering
comes from the model's species row; native indexes are converted once in glue.
The native bridge packs to Fortran order without changing any float word.
RRTMG kind_rb=8 arrays remain float64 inside radiation's own adapter.

Reuse `cuda_physics_prep_v841.py` for its existing generalized preparation and
geometry; do not reproduce preparation formulas in scheme launchers. The
adapter owns units, mass/hydrostatic conversions, normal-wind reconstruction,
halo communication, and writeback. Slot drivers own exactly the conversions
in their MPAS `*_from_MPAS`/`*_to_MPAS` source; carry additions by native names.

`prepare_mpas_number_carriers_v842` reuses the generalized preparation’s
max(+0,q) species scratch, then applies native MYNN zeroing and grouped
`f_nc/f_nifa/f_nwfa` selection. Its return mapping omits unallocated fields;
this number scratch is separate from the prognostic species row.

Keep the tendency pools distinct:

| Pool | Fields |
| --- | --- |
| PBL | rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten; MYNN also rqsblten, rniblten; aerosol Thompson also rncblten, rnifablten, rnwfablten |
| convection | rucuten, rvcuten, rthcuten, rqvcuten, rqccuten, rqicuten; KF also rqrcuten, rqscuten |
| LW | rthratenlw |
| SW | rthratensw |
| dynamics handoff | rthdynten, rqvdynten |

`CudaMpasCouplingV842` binds persistent unweighted edge caches, invokes
`physics:blten` then `physics:cuten` halo exchanges on the first RK stage of
the first split, and returns `CudaMpasStageSourcesV842` in dycore carrier names.
Pass the common device validation flag to it.

`couple_mpas_physics_tendencies_v842` in `cuda_physics_v841.py` reproduces
`todynamics.F:98-498` independently of the existing v8.4.1 aggregate path. Mass
is `rho_zz` at time level 2; theta_m and scalars are time level 1. The dycore
assembles it once before the RK/substep loops
(`mpas_atm_time_integration.F:2140-2176`). It starts physics
sources at zero, adds PBL, convection, LW and SW in source order, and then
converts dry theta to moist theta. Keep the separately weighted source sums in
native order, then hold the assembled arrays for all RK stages and substeps.
`PhysicsDriver.physics_get_tend` caches the first call's result until the next
outer step. The brief's per-stage reassembly is absent from the pinned source;
the native-defect policy requires the pinned source's behavior.

`project_mpas_momentum_v842` reuses the existing projection kernel with unit
mass, preserving the native unweighted edge cache. Project PBL and
Tiedtke/NTiedtke momentum separately after their named halo
exchanges at `rk_step=1, dynamics_substep=1`. Pass unweighted cached edge fields
to coupling; multiply each by the outer-step mass before adding. GWDO changes the
PBL tendency pool, rather than acquiring an additional RK source. Preserve
`tend_u_phys` separately (`todynamics.F:380-462`). IAU follows this insertion;
the f4 seam leaves its implementation to d3.

`prepare_mpas_post_transport_v842` implements the native dynamic tendency
handoff and species clamp for `post_transport_prepare`. GF/NTiedtke handoff uses
unclamped transported qv first; vapor tendency and scalar clamp include halos,
dry-theta tendency recovery uses solve cells only. Negative zero and NaN survive
the native `< 0` clamp. `pre_microphysics` synchronizes host/device fields in
OpenACC builds; it does no numerical work in the strict CPU build. The native
microphysics driver owns dry-state conversion in `microphysics_from_MPAS`.
Scheme launchers update dry theta and scalar carriers. The driver’s
`microphysics_to_MPAS` recovers moist theta and commits at native time level 2;
`post_microphysics` only synchronizes in OpenACC builds and is a CPU no-op. Existing generalized v8.4.1
writeback kernels retain their contract; carrier-specific extensions must be
verified against `interface.F:598-1138` before native full-model admission.

## Restart and receipts

Persist all active MPAS Registry fields in `diag_physics`, `tend_physics`,
`sfc_input`, `atm_input`, `ngw_input`, Noah-MP pools, and scalar state. This
includes held radiation/convection rates, effective radii, precipitation and
radiation accumulators, integer bucket counters, CAM absnst/absnxt/emstot,
soil/sea-ice/ocean state, MYNN prognostics and scheme work arrays that native
marks persistent. The native Registry and scheme package declarations determine
allocation; unallocated fields must not be reported as numerical zero fields.

Manager restart payload preserves rational alarm phase, elapsed time and the
exact resolved configuration. Driver restart is allowed only after finishing a
step; adapter field restore must be transactional. A configuration or alarm-set
mismatch refuses restore, preventing a phase-shifted restarted forecast.

`PhysicsDriver.calls` records operations, slots, backends, native step and time.
These are dispatch diagnostics, not G2 receipts. Whole-model verification must
compare every written field/output through f2 with missing fields in the ratchet.
Standalone tests and schedule diffs support debugging; they do not establish M0.

The standalone replay accepts Rust-decoded initialization columns through
`MPAS_PHYSICS_REAL_COLUMNS`. The test reader emits raw stored float32 words;
strict native `atm_init_coupled_diagnostics` statements (:7546-7547) form
`rho_zz` and `theta_m` before coupling comparisons. Scheme rates and edge
masses in this fixture are synthetic. Captured native driver arguments and
full forecasts remain separate proof requirements.

## Pinned NOAA fork provider

The additive `mpasphys_rrfs` table section selects
`woof.hex.mpasphys.rrfs.driver_rrfs`, pin set `rrfs`, and dycore path
`v8.3.1-noaa`. Authority is RRFSx/MPAS-Model
`1d6e370e7b43a332074809f170532a216c2212fb`, with the eight submodule pins
recorded in the row. It is separate from the existing WOOF RRFS preset and
MPAS-A v8.4.2 provider. Existing table sections and the default are unchanged.
The native `hrrrv5` spelling routes to the fork. The explicit
`rrfs_mesoscale_reference`, `rrfs_convection_permitting`, and `rrfs_none`
routes resolve to the fork's original suite spellings inside its native config
pool; unqualified shared suite spellings retain their existing routes.

Control order follows the fork's `mpas_atmphys_control.F:94-386`: validate
cloud input, fill an unset radiation cloud, validate MYNN pairing, then apply
TEMPO's internal `cld_tempo` override. The override changes cloud-PDF to `-2`
only for `bl_mynn`, as the pin specifies. MYNN-EDMF and MYJ force their own
surface layers after input validation. `physics` defaults and the included
Noah-MP, TEMPO and chemistry namelist groups come from the fork Registry.
Smoke/dust/anthropogenic driver options remain available; MUSICA is excluded.

The fork provider reuses the common alarm and lifecycle transactions. PBL
still runs each driver step, even when its interval alarm exists. Microphysics
subcycling (`n_microp`, `dt_microp`) remains in pristine
`mpas_atmphys_manager.F:737-754`: TEMPO shares Thompson's 90-second rule;
NSSL uses its own 75-second threshold. Physics sources are assembled once
before the first RK stage (`dynamics/mpas_atm_time_integration.F:1076-1084`)
and retained for the outer timestep.

The pre-dynamics order is the full fork driver (`mpas_atmphys_driver.F:127-419`):
cloud, SW, LW, surface layer, OML, LSM/sea ice/surface diagnostics, emissions,
PBL, drag, and convection. RUC uses the same allocation boundary as Noah.
PBL receives `state`; cloud receives `itimestep`; sea ice receives
`itimestep` and `mesh`. NGW is embedded inside native UGWP's drag call and
is not called twice. A separate NGW provider requires its own component
boundary and disables embedded native NGW during allocation and drag.
There is no pristine standalone `driver_ngw` routine in the fork: that seam
entry must be supplied externally for a split backend.

`NativePoolArguments` binds every argument by its actual fork ABI name.
The domain/session owner must supply `emission_input`, `output_noahmp`,
`ngw_input`, native `domain`/`block`/`dminfo`, and the real halo-exchange
procedure. Convection passes `state`, `diag`, `dminfo`, the callback and the
block (`mpas_atmphys_driver.F:379-385`). The callback must use the actual
native local/halo/dummy indices and registered `physics:sub3d_cuten` fields,
including `cellsOnCellsOnCell` and `ncellsOnCellsOnCell`; a Python no-op is
not an implementation. r2's ABI may bind the pristine exchange procedure
inside its compiled trampoline instead of accepting a procedure handle.
`physics_run_init` additionally receives `emission_input` and `diag_physics`
(`mpas_atmphys_manager.F:382-832`). Surface updates receive
`config_gvf_update`, `config_dust_scheme`, and `diag_physics`
(`mpas_atmphys_update_surface.F:54-143`). Native initialization owns DA-cycling
land/soil preservation, RUC mosaics, constants, and scheme-private state.

Scalar allocation follows Registry package predicates in original order.
The row also publishes `dycore.variant=mpas-v8.3.1-noaa` for r3's
factory. Explicit existing `--physics-suite` aliases can select a fork provider
while retaining an original shared-suite namelist byte for byte; aliases resolve
through `suite_routes`. Unqualified shared suite names still select v8.4.2.

The row's `streams_parser` names an adapter to the pinned native `ezxml`
parser. The original GSL streams declare `filename_interval` twice; native
accepts that declaration and reads its first value. The adapter copies only
stream declarations and leaves the original XML bytes unchanged. The
shared provider keeps its existing parser. Weather payloads still pass
through Rust.

The fork sea-ice slot inherits the resolved land scheme (including RUC), keeps
its own backend, and runs before surface diagnostics. Mixed UGWP/NGW backends
retain native NGW allocation and climate packages, mask only the parent call,
and execute both callbacks before GWDO deallocation. The pre-drag snapshot is
shared with f4. A separate NGW native entry still needs its provider; the fork
only exposes it embedded in the complete UGWP driver.

The full Registry retains associated zero-length chemistry arrays when
`num_chem=0`. Rust transport and the native ABI must preserve those shapes,
time levels and empty payload hashes; dropping the fields or changing their
dimension to one changes the native pool contract. This keeps MUSICA disabled.

TEMPO adds `nc`, `nifa`, `nwfa` when aerosol-aware, and `ng`, `volg` when
hail-aware. The pin has no `qb` scalar. `scalar_registry` and
`package_selection` describe the physics predicates; the full Registry
provider must also allocate chemistry groups when enabled. The table's
`persistent_pools` inventory lists changed/new fork state, physics tendencies
and diagnostics, input fields, and emission storage. It supplements the full
Registry, and does not replace it. Restart snapshots must include every
allocated pool/time level, accumulators, and native scheme-private state;
restoring public tables alone is insufficient. The fork top-level restart
schema and resolved pin receipt distinguish it from the shared provider.

After every output cycle, use the stream manager's actual diagnostics output
alarm, captured before that alarm is reset, to invoke `after_output`. The
native `atm_reset_diagnostics` callback zeroes `refl10cm_1km_max`,
`max_hail_diameter_sfc_acc`, and `max_hail_diameter_column_acc` after writes
(`mpas_atm_core.F:883-894,971-1002`). It must run even when a diagnostics field
is not written. The physics manager's earlier diagnostic lookahead is a
different alarm decision. The fork table's `native_core_module` binds
`reset_native_diagnostics` to the session operation; initialization-time output
is not a timestep reset. It zeros every associated allocation, including dummy
and halo storage, and leaves absent fields unallocated.

After scheme initialization and before output or timesteps, that provider's `initialize_native_core` binds
the complete Rust second-ring traversal from r3 into native mesh pools
`nCellsOnCellsOnCell` and `cellsOnCellsOnCell` (fork `mpas_atm_core.F:598,1481-1584`).
The bridge retains native traversal order; the adapter translates active
zero-based neighbors into one-based neighbors. It copies only owned counts
and active neighbor entries, preserving halo, unused padding and native dummy
storage exactly as the fork does. Its executable must be installed by r3.

The optional coupling and number-carrier wrappers select word-identical
shared arithmetic statements for MYNN-EDMF and aerosol-aware TEMPO
(`mpas_atmphys_todynamics.F:251-484`, `mpas_atmphys_interface.F:374-506`).
Their selectors never substitute a shared scheme implementation for a fork
scheme. Their cache compiles directly through NVRTC with `--fmad=false`,
`--ftz=false`, `--prec-div=true`, and `--prec-sqrt=true`; CuPy RawModule's
forced FTZ would change native subnormals. WOOF/dynamics compilation is
unaffected. This local cache can be replaced by f4's identical strict cache
when that module lands.

`required_extensions` in a driver receipt states the matched-run constraint:
WOOF, WOOF physics, moving domains, and regional extras must be disabled by
the forecast owner. It does not claim that the real door has enforced them.
Only a completed real-door forecast receipt may state the actual resolved
extension settings and scientific identity. Source/orchestration unit tests
and synthetic CUDA replays do not establish RRFS-M0.

## Public native forecast and hydration

`woof hex forecast --physics-namelist namelist.atmosphere --streams
streams.atmosphere --init input.nc --out run` selects explicit native suites.
Bind --native-physics-library, --native-pool-reader and --mpas-source (or their
MPAS_NATIVE_LIBRARY/MPAS_NATIVE_POOL_READER/MPAS_SOURCE environment variables).
One that is not bound comes from this install (`woof hex physics-resources
build`, built on first use; docs/physics-resources.md). Every suite needs all
three, the all-GPU ones too: the library hosts the pools, clock, streams and
driver glue. The native library is test-only; the reader is the owned Rust
`rw_mpas_physics_pool` ABI gpuwm-rw-mpas-physics-pool-v1. Stream field list files
resolve beside streams.atmosphere, or under --stream-list-directory. Native table
directories can be supplied repeatedly through --native-physics-tables.

RegistryLayout resolves active packages, scalar order/indices, original kinds,
aliases and independent time levels. Rust reads only input-stream members and
preserves the native default for fields absent from inputs; it emits immutable
hash-checked raw Fortran buffers. NativePoolHydration validates all metadata and
payloads before populating actual native pools. Successful --preflight certifies
routing/metadata preparation only. It does not certify scientific initialization.

## Selected CUDA callbacks and device pools

Each slot table row can publish cuda_entries[scheme] containing module and call.
The shared session imports only selected hex-cuda callbacks and registers their
exact slot/scheme keys. Unselected scheme modules are never required. The native
callback remains selected for every other active native-cpu slot. A descriptor
may provide profiles (argument alias -> prepared native variable), constant_aliases
(argument alias -> compiled native constant), and bind (module function accepting
provider, SlotCall and returning the scheme mapping). Binders associate storage
and copy ABI buffers; all formulas belong to CUDA.

NativeDevicePoolViews supplies mapping keys mesh, diag, diag_physics, state,
sfc_input and other Registry pools; configs contains native Registry config values.
scalar_state reads native manager globals without recalculating them. constants
reads f3 get_constant; absent compiled constants fail explicitly. prepared reads
native module globals after allocation/conversion glue. Atmospheric i,k,one-j
native arrays become contiguous k,i CUDA views. MYNN profile aliases use exact
driver_sfclayer.F:1016-1047 arguments. Microphysics uses state time level 2; other
slots use level 1. Solve copies exclude halos and MPAS dummy. After a successful
callback buffers are copied back into original native storage, retaining halo
and dummy words. A failed callback does not commit changes. Scheme binders may
associate specific views/tables; they retain native precision/allocation ownership.

The runtime uses existing CUDA conversion kernels. It synchronizes outer state
before physics, holds sources through all RK stages, and hands unclamped transport
to native microphysics at the pinned insertion. Native writes requested streams;
unpopulated diagnostics are missing explicitly for the integrator ratchet. Pool
snapshots do not certify persistence of private/SAVE/derived native scheme state.

Sea ice has its own backend selector `config_seaice_backend`, with the scheme
inherited from `config_lsm_scheme`. Disabling it preserves land surface's
selector/backend. Disabling land surface also disables its sea-ice call. The
native slot calls `driver_seaice` once between allocation and deallocation,
then runs surface diagnostics, with `dt_pbl` for either Noah selector
(mpas_atmphys_driver.F:285-321; mpas_atmphys_driver_seaice.F:495-536).
Its table descriptor binds `seaice_shared`: t3d=t_p, qv3d=qv_p,
p8w3d=pres2_hyd_p, dz8w=dz_p and the surface-layer *_sea flux arrays.
Native i,j surface storage becomes a contiguous solve-cell vector; native
i,k,j profiles become k,i. Copies preserve halos/dummy values. Fractional
sea-ice arrays only exist when native allocates them; scheme launchers must
only request them on branches where native consumes them.

The logical `config_ngw_scheme` selects `ngw/ugwp_ngw` within
`gwdo/bl_ugwp_gwdo`. `config_ngw_backend` selects its backend separately.
When both are native, `driver_gwdo` contains the NGW call and no second callback
runs (mpas_atmphys_driver_gwdo.F:692-834). Mixed execution snapshots pre-drag
rublten/rvblten, masks the actual native config_ngw_scheme during the parent
call, restores it even on errors, then invokes NGW before deallocating GWDO.
Both receive dt_pbl. The NGW provider consumes independent pre-both wind copies;
rubldiff/rvbldiff subtract them once after the two components. Native NGW with a
CUDA parent needs a separate provider entry; it cannot replay the whole parent.
Thermodynamic/climate preparation is still owned by native/provider and the
scheme lane. The driver supplies storage copies without re-evaluating them.

The real MPAS door accepts `--streams` and repeatable `--stream-input NAME=FILE`.
Input files enter Rust pool hydration. Input-only streams read those explicit
files; streams that also write receive owned byte copies at the start-time
filename. MPAS retains its output template and filename interval. Never bind a
writable stream directly to a reference file. Initial input defaults to `--init`;
UGWP orographic input can bind through `--orographic-statistics`. Other input
streams need explicit files, including NGW climate input. Recurring records are
selected by native stream I/O, not by Python interpolation.

Thompson's table descriptor uses `bind_native_pools` and installs persistent
`provider.thompson_v842_resources` before any callback. `--physics-resources`
names a JSON object keyed by scheme (`mp_thompson`); `woof hex physics-resources build` makes it from the install (docs/physics-resources.md). Its entry contains:
`rust_decoder={path,sha256}`, `table_directory`, `auxiliary`, and
`device_math_source={path,sha256}`. The decoder implements the published Rust
lookup-record ABI; its certificate is explicit. The four native DBL hashes come
from the scheme, and the six-array auxiliary packet is pinned to the published
strict preparation certificate. Rust decodes records/endian words; Python
marshals the buffers into Fortran-ordered arrays without scheme mathematics.
All 33 shapes/precisions remain the scheme's declarations. The shared device
math source must be approved by its owner; no local fallback is installed.
Resources survive provider.begin(), which clears only per-call field views.

MPAS carrier preparation uses its own `MpasPhysicsKernelCache`. It compiles
through direct NVRTC with `--fmad=false --ftz=false --prec-div=true
--prec-sqrt=true` and records actual compiler entry evidence. Its scope is
physics conversion and carriers; the dynamics cache and default WOOF route
retain their existing compiler behavior. When the integrator's additive
`KernelCache.raw_kernel(..., mpas_strict=False)` API is present, the MPAS cache
explicitly passes `mpas_strict=True`. The default remains false. Scheme
launchers retain responsibility for their own strict compiler and device maths.

The MPAS cache binds the existing common carrier kernels to the shared GNU
float sine, cosine, logarithm and FMA-path exponential headers. This binding
does not change their archived source or the default physics cache.
Native output `theta` consumes the raw prognostic `qv`, including negative
words; scheme preparation keeps its separate clipped carrier. `rho` and
`pressure` follow the native output products/sums. Cold initialization computes
both Exner and pressure pairs using the shared GNU FMA-path power function,
then publishes those resident base-state words to the native pools.
Source: mpas_atm_core.F:925-966; mpas_atm_time_integration.F:7628-7650;
mpas_vector_reconstruction.F:303-323; mpas_atmphys_interface.F:504-519.

## Explicit native runtime auxiliaries

The caller supplies immutable supplemental tables with repeatable
`--native-physics-tables DIRECTORY`. This includes Noah-MP's separate
`NoahmpTable.TBL`; physics initialization retains its native relative lookup
inside the owned output directory. Input-only runtime streams use
`--stream-input NAME=FILE`, including `ugwp_ngw_in` for native NGW's limb
flux input. These files enter Rust hydration before Registry dimension
resolution, and native stream input binds to the same read-only source.
The driver never fabricates dimensions or fields absent from those inputs.
Source: mpas_atmphys_init.F:451-475; mpas_stream_manager.F:3187-3190.

The strict non-MPAS_DEBUG framework starts with its normal SILENT optional
pool lookup policy (mpas_pool_routines.F:108); DEBUG uses WARN
(mpas_framework.F:107-109). A stand-in which imposes FATAL instead changes
optional absent-field behavior, including landuse's null/associated handling
(mpas_atmphys_landuse.F:154-164,290-310). Match the pinned native runtime
policy rather than allocating inactive-package fields to bypass the error.

## Lagged microphysics heating across timesteps

`CudaMpasStageSourcesV842.diabatic_theta` is the resident RKIND view of
`tend.rt_diabatic_tend`, retained from the previous microphysics update. It
stays separate from the four physics_get_tend sources. Dynamics adds
`rho_zz * diabatic_theta` after forming rthdynten and before Euler mixing and
held physics sources at every RK stage. At RK3 recovery it subtracts
`((rk_dt * rho_zz) * diabatic_theta)` from the updated rtheta_p before
recovering theta_m, Exner and perturbation pressure. Intermediate RK stages
retain that heating so it influences acoustic pressure and momentum.
Native-cpu supplies the native pool words; hex-cuda microphysics owns the same
persistent field, including restart hydration. No Python field arithmetic is
used. The dycore's optional carrier defaults to None for existing callers.
Source: mpas_atmphys_interface.F:912-926;
mpas_atm_time_integration.F:6931-6948,4436-4445; Registry.xml:2041-2042.

After all ringing streams have written, the diagnostics output alarm clears
`diag_physics.refl10cm_1km_max` to zero for the next accumulation period.
Initial output does not run this reset. Do not clear it between diagnostics
and restart writes; both must see the same pre-reset words.
Source: mpas_atm_core.F:877-890,968-990.

### RRFS resident halo metadata

The table-selected fork layout preserves `config_num_halos` from native
bootstrapping, including empty rings on MPI1. `NativePoolHydration.register`
passes its cumulative `halo_ends` to the native decomposition ABI; layouts
without this selected metadata retain their existing registration behavior.
An MPI1 global entity has repeated ends equal to its owned count. A distributed
provider must supply actual per-layer cumulative ends; the driver never invents
a partition. Authority is fork `mpas_bootstrapping.F:144-156` and
`mpas_atm_halos.F:181-186` at `1d6e370e`.

A halo-aware slot obtains resident views including both native rings. Its
six-field GFL transaction commits device carriers, exchanges native
`physics:sub3d_cuten` at time level 1 / layer 2, and refreshes the resident
views before the slot continues. Empty MPI1 rings retain their index and the
native exchange still executes. Prepared rank-4 `(i,k,j,species)` storage is
exposed as `(species,k,i)` and copied back without touching inactive padding.
Scheme arithmetic remains in the owning lane.

### RRFS native initialization tables

The fork's `native_layout_prepare` provider restores constituent counts and
indices for inactive var arrays before native hydration. Registry-generated
allocation retains `num_<array>=0` and inactive indices/group bounds `-1` even
when an array package is off (fork `src/tools/registry/gen_inc.c:1321-1489`).
The manager reads emission counts unconditionally during initialization
(`mpas_atmphys_manager.F:461-485,707-726`). Active fields/counts and input-file
dimensions stay authoritative; this hook does not allocate inactive weather
fields or invent missing land/soil category dimensions. The shared session
must invoke this table-selected provider before `NativePoolHydration.populate`.

The fork row declares `native_table_selector` and `native_table_names`. Its
provider collects the existing MPAS tables plus TEMPO's three extensionless
lookup tables, `ccn_activate.bin`, the monthly aerosol climatology and UGWP's
limb input. These inputs remain opaque native files: Python only selects paths
and hashes. Publication JSON and logs are excluded. Duplicate native filenames
with identical SHA256 bytes represent one input; conflicting base bytes are
refused before installation. The row's `native_table_override_keyword` separates
explicit `--native-physics-tables` roots from the built-in root. Those explicit
roots replace entries in argument order, matching r0's native `--tables` staging;
the runtime records every selected file hash. Table variants must match the
actual reference manifest before any word-identity claim. No file extension or
precision is changed. Absent files retain the pristine scheme's initialization
behavior. The shared runtime consumer is an explicit integration proposal;
without that consumer, listing these names alone does not provide them to a run.
The existing v8.4.2 table selection is preserved when no provider is selected.

The RRFS row also selects `native_registry_prepare`. Invoke it immediately
after parsing the included Registry and before any field, dimension or package
allocation. It reproduces native `push_attributes`, pool/carrier merging and
stream merging (`src/tools/registry/parse.c:54-57`, `gen_inc.c:2227-2521` at the
fork pin). Included chemistry declarations extend the original scalar carrier;
they must not produce duplicate native fields or reset moist constituent indices.

`native_table_stream_selector` binds active table-backed input streams before
Rust inspects input dimensions. The declared NGW stream uses the selected
`ugwp_limb_tau.nc` bytes, including ordered overrides; Rust supplies its actual
`lat` and `days` dimensions and weather words. The same paths bind stream input.
`native_initialization_namelist` stages the original namelist in the native
working directory, preserving internal non-Registry scheme groups. NSSL opens
this file itself at `physics_wrf/module_mp_nssl_2mom.F:1227-1245`. An existing
different file is rejected because it would initialize different options.
These consumers remain exact shared integration proposals; no v8.4.2 provider
or default selects them.

### RRFS delivery proof audit

`tree/tests/rrfs_proof_contract.py` audits the fresh reproduction directory
against both immutable gate indexes and the immutable missing-field ratchet.
It requires every indexed forecast receipt, the delivered clean commit, final
output steps and nonzero compared storage. A passing gate with no receipts,
column evidence, another commit, missing samples or differing words fails.

The complete driver proof additionally requires `rrfs/driver-witnesses.json`
(schema `rrfs-driver-forecast-witnesses-v1`) generated during that reproduction.
This is a required evidence contract; the shared door does not yet produce it.
The file contains `source_commit`, one `decks` entry per RRFS gate deck, and
`broken_slot`. Each deck entry names its `deck`, baseline receipt
`forecast_sha256`, and pinned `{path, sha256}` references `runtime`,
`native_alarm_trace`, and `hex_alarm_trace`. The runtime sidecar must complete
all requested steps, match the original resolved namelist and report the
actual `resolved_extensions`. The verifier must capture its hash as
`native-physics-receipt.json` and the normalized hex trace hash as
`physics-alarm-trace.json` in `provenance.hex_output_sha256` before disposing
of forecast output.

Both normalized traces identify the exact comparison deck, strict
`native_manifest_sha256`, process count, duration and `evidence_kind=FORECAST`;
ordered `events` contain `operation` and integer `step`. All active radiation,
PBL, land-surface and convection branches must be observed. The full matrix
must also observe both hail accumulator reset operations. Native and Hex events
must agree exactly. Standalone clock traces do not satisfy this contract.

`broken_slot` names `baseline_deck`, `slot`, pinned comparison `receipt` and
`runtime` references, and `expected_first_difference` (`stream`, `field`,
`step`). Its actual runtime must flip exactly one active native slot to
`hex-cuda`, preserve all other physics and resolved extensions, and finish the
forecast. Its comparison must preserve the baseline output/missing inventory
and show the first computed difference at that field and step. The runtime
sidecar must be hash-bound to that comparison as above. An admission refusal,
missing output or unchanged field cannot prove the switch.

The native trace collector, actual extension resolution sidecar, fault runner
and verifier sidecar retention are still required integration work. Until they
produce fresh evidence, the lane's reproduce script cannot return success. Unit
fixtures validate this audit contract only; they are not forecast receipts.

The visual proof rule also applies to this driver lane. The maintainers' delivery
helper (`rrfs_forecast_sheets.py verify-mpas`, kept with the proof-lane harness and
not distributed) runs inside the proof GPU queue when sheets need the fresh
outputs: it delegates to the unchanged verifier and real model door, records the
ordinary receipt, and retains owned output until comparison and inspection
finish. Run `mpas-sheets` separately with the Rust renderer, then retire the raw
outputs. The helper changes no model selection or numerical acceptance rule.

`rrfs_sheet_contract.audit_sheets` requires `sheet-inspections.json` with schema
`rrfs-sheet-inspections-v1`, the clean source commit, and at least one actual RRFS
proof deck. Each deck entry binds its forecast receipt SHA, `mpas-sheets/v1`
manifest through a file reference, fixed difference-table SHA, observations for
every hash-bound PNG, and an evolution observation. All eight products must exist
at 6, 12 and 24 hours; additional produced frames and images must also be inspected.
Manifest deck/native/output hashes must match the actual field-verification
receipt. This is an artifact audit: it cannot perform visual judgment. The lane
opens every image and records what it saw; the integrator opens them before
landing. Renderer maxima never substitute for those observations. Missing sheets
or inspections fail the complete delivery proof audit.

### Real RRFS driver boundary replay

The integrator's 2026-10-14 03:00 ruling supersedes the per-lane forecast
matrix as a landing requirement. The matrix and witness audits above retain
their historical scope. Landing now requires complete bitwise module replay
from real native-tap inputs, a clean-checkout run of the lane's replay script, and one
24-hour forecast with the implementation enabled and inspected native comparison
sheets. The integrator owns the headline exact whole-forecast configuration.

`tests/rrfs_driver_dump_replay.py` consumes the complete `physics_get_tend`
boundary (fork `mpas_atmphys_todynamics.F:65-550`). Rust decodes and writes
typed records and compares every POST payload, bounds, global IDs, boundary
identity and alias. Output emission uses PRE metadata and actual CUDA results;
expected POST values are available only to the comparator. A producer manifest
binds the original fork source, binary, rank count and every captured file.
Its schema is `r4.real-driver-bank/v1`, with `model_commit`, `source_sha256`,
`native_binary_sha256`, `process_count=1`, `input_origin='native-tap'`, and
`captures[prefix].files` mapping every PRE/POST filename to SHA-256. The
consumer refuses missing or extra records before numerical execution.

This consumer currently admits closed global MPI1 pools only; distributed or
regional replay needs the actual captured halo callback context. It projects
the recorded native vectors and then calls the production RRFS coupling
adapter. Pass the native mesh dimension as `n_edges`, independently of the
allocated diagnostic width: native overwrites `tend_u_phys` only through
`nEdges`, preserving its persistent dummy edge (fork
`mpas_atmphys_todynamics.F:327-333,434-442`). Other coupled output arrays retain
the native zeroed padded extent. Nonfinite native words remain in the comparison.

A successful coupling replay covers this entire boundary, including unchanged
pool records; it does not establish uncaptured control, manager, initialization,
interface or diagnostic-reset boundaries. Those contexts must accompany the
whole-driver replay before this unit can claim complete module acceptance.

The RRFS coupling external observer in
`tests/rrfs_driver_external_watch.rs` uses r1/r12's GNU descriptor and ptrace
mechanism against the pristine pinned executable. It records entry arguments
and the native buffers returned by direct pool getters, including the momentum
projection helper. Its scope is one complete caller boundary, rather than every
pool member or the outer manager/initialization boundary. Record axes without
observed global IDs stay unmapped; absent pointers stay explicitly absent.
Observer admission includes an ordinary-versus-observed native comparison with
output at the last captured timestep, including restart. The supplied-checkout
replay uses Rust for typed record transport and word comparison, and directly
compiles the shared strict CUDA arithmetic. Native hidden state and callbacks
outside the captured boundary require separate evidence.

The RRFS real-input driver replay also observes the pristine native lifecycle
from control checks through finalization. The current stated case covers one
first model timestep, with complete pre-dynamics call order, native manager
schedule scalars, coupling, microphysics delegation and the successful
post-output diagnostics reset. Invocation recording verifies orchestration;
it does not replace numerical replay of a delegated scheme or initialization.
The untouched native executable runs ordinarily and under observation, and all
nine NetCDF files, including evolved output and restart, match word for word.

`PhysicsManager.native_timestep_state()` exposes the fork's RKIND schedule
metadata, including its microphysics substeps and convection ratio
(`mpas_atmphys_manager.F:737-762` at the fork pin). Native initialization still
owns its module variables. This metadata does not change the v8.4.2 scheduler.
The complete number-carrier slice of `MPAS_to_physics` is separately captured
on real native input and replayed on CUDA. Native rectangular scratch includes
KME, while those assignments write only KTS:KTE; the observed interface tail
must survive from PRE. Every carrier and captured context word, including that
tail, is compared. Temperature/geometry preparation and hidden numerical
initialization state outside these packets remain separate proof scopes.

The real preparation packet now includes the complete `MPAS_to_physics` routine
at fork `mpas_atmphys_interface.F:284-640`: every allocated module scratch array
and every directly accessed native pool record, not just number carriers.
`prepare_mpas_to_physics` consumes native reconstructed diagnostic winds,
uses the shared mass/interpolation/hydrostatic CUDA kernels with the shared GNU
device exp/log routines, and retains native's left-associated surface-pressure
sum. Interpolation weights are returned; their untouched boundary levels and
all allocation tails survive from PRE. The same source statements are identical
in the v842 native file; the arithmetic adaptation is local to the fork wrapper
and leaves the existing shared/default kernel unchanged. Expected POST storage
is comparison-only, never an input to generated results.

The clean-checkout replay includes full preparation, whole coupling, the
independent number-carrier observation and the real schedule/call-order case.
Counts include unchanged captured context and a separately observed carrier
case; they are not unique prognostic fields or a completed Hex forecast.
Numerical native
initialization/private state and the implementation-enabled visual forecast
remain outside this packet's acceptance claim.

The RRFS real control replay observes the entire `physics_namelist_check`
configuration boundary (fork `mpas_atmphys_control.F:87-386` at `1d6e370e`).
It captures all twelve configuration pointers and module `moist_physics`,
resolves from actual PRE values through the existing table-driven resolver,
and compares every typed POST record with the Rust codec. Character records
include all 512 bytes, including native padding; their comparator word count
is one per character record. This successful real case is separate from the
existing standalone branch/control matrix. Logging/error-message storage,
other control-module numerical initialization and private state are outside
this boundary's claim. A wrong forced surface selection must fail this replay.


The complete `microphysics_from_mpas` and `microphysics_to_mpas` handoffs
(fork `mpas_atmphys_interface.F:643-1129,1132-1625` at `1d6e370e`) now have
resident CUDA entry points. They reuse only identical common thermodynamic
statements from the existing microphysics interface kernels. No Kessler scheme
is invoked. The fork exit covers all physical levels rather than the v842
microphysics top-level truncation. Local glue copies the configured number,
aerosol, hail and NSSL carriers/diagnostics and computes native tendency rates.
NSSL entry seeds its number tendencies, while native exit leaves those number
seeds unchanged. Native `w` has an extra interface level; handoff scratch writes
only KTS:KTE. Allocation tails and dummy cells survive from actual PRE.

The stated real replay covers both complete handoffs in the measured TEMPO
case with aerosol and hail awareness: all allocated scratch and direct getter
records are compared, including untouched context. Other selector branches are
implemented but are not claimed as real-replayed. Shared native power maths is
qualified here against this actual strict GNU/glibc 2.43 case, not by a generic
all-domain math claim. A deliberately wrong timestep fails the returned heating
and tendencies. The clean supplied-checkout replay includes these boundaries;
numerical initialization/private state and the own implementation-enabled
visual forecast remain separate acceptance requirements.


The complete physics output phase has separate real reference-host capture
and replay: all actual calls to fork `atm_compute_output_diagnostics`
(`mpas_atm_core.F:919-968`) and `atm_reset_diagnostics` (`:971-1002`) during the
stated first timestep. `compute_native_output_diagnostics` reuses the byte-
identical native diagnostic statements through the existing shared CUDA
kernel: prognostic qv is unclipped, and only physical field extents overwrite
PRE storage. `reset_native_diagnostics` clears each whole associated allocation,
including the fork's two hail accumulators. The real reset replay binds observed
allocation metadata to the native pool setter contract; no POST supplies input.
Successful-write ordering remains in the complete driver lifecycle replay.

The reproduction controller selects a complete reference-host capture
packet through `RRFS_DRIVER_BANK_ROOT`. Historical banks from an earlier capture
host remain separately bound evidence. The aggregate records the authenticated capture host and
manifest hash of every phase; it never infers cross-host forecast identity
from the GPU replay host. Wrong temperature and reset implementations fail the
corresponding output fields. These are complete module boundary measurements,
not a completed own-implementation forecast or hidden numerical initialization
proof.


RRFS surface-layer handoffs own complete pool packing and diagnostic writeback
around the sibling scheme callbacks. `surface_layer_from_mpas` and
`surface_layer_to_mpas` follow the fork
`MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:434-1052`
at 1d6e370e. The adapter supplies resident pool arrays in MPAS layouts and the
complete allocated `_p`/`_sea` scratch in native layouts, plus actual tile bounds
and the initialized `xice_threshold`. Untouched allocation tails and inactive
scratch survive. Scheme kernels and allocation/initialization remain separate
owner hooks; this handoff does not infer their private state.

The entry duplicates the main/sea fields and applies native fractional-sea-ice
surface conditions. The exit reproduces the exact selected blend fields;
`sf_mynnsfclay` blends `ch` but retains main `cqs`, and the native flux fields
that are not blended remain main. All multiply/subtract/add operations compile
through the strict fork cache. Mesh spacing reuses the shared GNU positive
powf implementation; the native strict ELF calls powf then single-precision
division. No scheme CUDA or device math is copied or replaced.

The stated real reference-host hrrrv5 first720s/MPI1 packet contains both actual handoffs,
including99 mixed sea-ice cells. It compares249 complete POST records and589,347
words exactly. Other surface selectors are implemented from the fork but are
not claimed as real-replayed by this packet. The immutable producer, passivity,
input and member hashes are in `real-surface-bank`; no expected POST is an
input. The landing runner includes this phase with the prior complete driver
packet and records each bank's native host separately.

RRFS PBL handoffs preserve the complete resident native scratch layouts and
MPAS pool layouts through the pbl_from_mpas and pbl_to_mpas launchers (fork
MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:461-1283
at 1d6e370e). The caller supplies actual integer bounds, package flags,
chemistry dimensions and resolved options. The MYNN-EDMF branch retains
tkepbl_p, rmol_p and kbl_plume_p; the MYNN branch copies or clears them
as native does. Native writeback assigns kzq from kzh_p. Radiation sums
use explicit rounded single-precision addition, and mesh spacing reuses the
shared strict GNU positive powf routine. Number tendencies obey the native
package conjunctions. Smoke/deposition carrier plumbing is preserved;
MUSICA remains outside this campaign. Scheme initialization and scheme
kernels are separate hooks. The stated real first-step PBL packet qualifies
both handoffs, every captured POST allocation and unchanged context; it does
not certify other selectors, uncaptured private state or the forecast route.

The fork convection entry and exit handoffs are `convection_from_mpas` and
`convection_to_mpas` (fork driver_convection.F:820-1474 at 1d6e370e). They retain
full native allocation indices and tails. NTiedtke's water forcing adds dynamics
then PBL; temperature forcing adds dynamics, PBL, longwave, shortwave in that
order, then multiplies by the existing Exner word. Mesh spacing shares the
strict native power/division implementation used by the surface and PBL glue.
Native writeback leaves rainncv unchanged. Non-LI GF reads and writes the native
rvcuten(k,k) diagonal; writeback ends with the last owned column's value.

GFL's numerical subsidence spreading remains r13's source-identical routine,
passed as `subsidence_spreader` or imported from its module. Positive sub3d
requires `exchange_halo_group('physics:sub3d_cuten', pools=out)` to exchange the
newly written resident pools. That real exchange completes before the arrays
are staged for r13's `_spread_after_halo`; it must bind the runtime domain and
this supplied pool storage. No placeholder exchange qualifies a neighbor-read
replay. The module replay packet currently qualifies both complete NTiedtke
handoffs on one stated native option companion, including every observed POST
record. Other selector and GFL-halo branches have no real replay claim.


RRFS LSM pool handoffs are `lsm_from_mpas` and `lsm_to_mpas`, complete fork
`mpas_atmphys_driver_lsm.F:350-642` and `:645-853` at `1d6e370e`.
They consume resident native pool and full scratch allocations, explicit
`num_soils`, `num_landcat`, `num_soilcat` and native cell/tile bounds. The
entry returns new scratch storage; exit returns new pool storage. Retain
inactive words and native increasing-tile write order. RUC's `qsg_p` entry
uses the source's rebound `qcg` pointer; the exit writes distinct `qsg` and
`qcg` fields. Rain forcing uses rounded zero-plus-micro then convection
addition; downward solar flux uses rounded subtraction then division.
`dzs_p` and RUC `zs_p` use the first selected cell. Do not write back the
entry-only vegetation categories, shade bounds or direct/diffuse solar
arrays. Allocation, LSM initialization, private state and scheme calls
remain their existing native or sibling hooks. The real RUC replay proves
these two handoffs only; it does not qualify their initialization.

The fork longwave handoffs implement `mpas_atmphys_driver_radiation_lw.F:300-834`
at `1d6e370e`: full entry staging, diagnostic/tendency writeback, and the CAM
restart writeback. They consume resident complete PRE pool and scratch allocations
and preserve inactive cells, levels and tiles. RRTMG radius selection includes
TEMPO and NSSL. Ozone interpolation reuses the source-identical landed RRTMG
kernel; the driver retains native sequence association at `o3clim_p(1,1,j)`.
CAM carries aerosols, climatology, the first-cell hybrid coordinate, and its
`absnxt`/`abstot`/`emstot` restart storage. Compile-time `degrad` is supplied as
native constant metadata. Allocation, numerical initialization, private state and
radiation scheme execution remain their existing native or sibling hooks.
The stated real original RRTMG case qualifies both observed main handoffs; CAM
and other option combinations are source implementations without real replay
qualification in this packet.


RRFS shortwave handoffs (`radiation_sw_from_mpas` / `radiation_sw_to_mpas`)
retain the fork's complete surface diagnostics, TEMPO radii and aerosol
preparation, ozone inputs, CAM persistent absorption and aerosol storage,
and full untouched allocation tails. Entry returns prepared resident scratch,
updated native pools and the native `aer_opt` scalar. The runtime passes the
native single-precision `degrad` constant and native dimensions. TEMPO-aware
shortwave reuses the source-identical p23 aerosol kernels, with the original
`qv_p`, explicit rounded vertical reduction and native active-level bounds.
The original caller's native getter/record descriptors remain the authority;
expected POST words are comparison only. RRTMG handoffs are real-qualified on
the stated March hrrrv5 first720seconds/MPI1 reference-host packet; CAM and other
selectors are source implemented but not qualified by that packet.


The fork driver also exposes `initialize_physics_pools` for its complete
numerical preparation before the first ozone/landuse initializer
(`MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_init.F:84-396,480-605`).
It normalizes east/north on `nCells`, resolves all fifteen scalar flags from
actual indices, and initializes counters, radiation accumulators, surface
memory, ocean mixed-layer fields and lagged soil temperature on `nCellsSolve`.
It preserves allocated tails and restart/DA-cycling storage. Trigonometry uses
the existing shared glibc device header and explicit rounded operations.
Ozone, landuse, scheme tables and private/SAVE state remain native delegates;
a preparation replay does not establish that private-state closure or wire
this additive entry into the protected runtime. Negative mixed-layer depth
retains the native contradictory missing-value predicate's no-write behavior.

## Correct mode and exact mode

On an MPAS suite route (`--physics-suite`, or `config_physics_suite` in the
`--physics-namelist`), `woof hex forecast` runs **correct mode** unless the run
names a mode. In correct mode hex fixes 5 defects found in native MPAS, each with
a test that fails on the native behaviour and passes with the fix, and each
checked against its cluster's native referee; exact mode reproduces native bit
for bit.

Choosing the mode:

- `--physics-mode correct|exact`, or `config_physics_mode = 'correct'|'exact'` in
  `&nhyd_model` (the two must agree). `--physics-correct-skip ID` (repeatable or
  comma-separated) or `config_physics_correct_skip` leaves one correction out of
  a correct-mode run, for an observation A/B arm.
- With no mode named, the door uses correct for an MPAS suite route. On the
  WOOF physics route the same option applies the corrections the installed
  engine carries (`woof.hex.correct_mode`) and lists the rest as named gaps; a
  skip list naming a finding that route never applies is refused, because the
  arm would be identical to its control.
- The library default of `mpasphys.driver_v842.resolve_namelist` and
  `mpasphys.forecast_v842.native_route` stays exact, so replay and reproduction
  scripts that call them directly are unchanged.
- Every proof names exact. `woof hex verify-mpas` runs
  `--physics-mode exact` when its deck names no mode (in `hex_command` or in the
  deck's namelist), records `physics_mode` (mode, source, whether it appended the
  flag) and `provenance.hex_command_run` in its receipt, and leaves the deck file
  unchanged, so the landing gates run the numerics they ran before the switch.
- An exact-mode run prints nothing new and writes the physics receipt it wrote
  before the switch existed (no `physics_mode`, `corrections` or `correct_mode`
  keys).

The findings (`CORRECTIONS_V842`, clusters mpas-rad-pbl-lsm-seaice and mpas-water-budget):

| Id | Native defect | On this tree | Follow-up owner of the gap |
|---|---|---|---|
| RP6 | `config_pbl_interval` stretches the surface/PBL step but the schemes are still called every step, so they integrate interval/dt times too long | Fixed: the surface layer, ocean mixed layer, land surface, sea ice, PBL and GWDO are called when the PBL alarm rings, and the land surface and sea ice receive every step's precipitation since their last call. Gaps: Noah-MP (it must step by `dt_pbl`), and a native-file restart (the window cannot be held in the pristine restart stream) | p17-noahmp; the MPAS integrator with f3's checkpoint provider |
| RP4 | `config_mynn_tkeadvect=true` resets TKE to a field MPAS never advects | Fixed by refusal: correct mode refuses the option, as the native correct-mode build does | none |
| RP1 | MYNN removes cloud water, ice and snow on every call | Fixed in the hex-cuda MYNN slot (p12): the exact inverse of the forward conversion, word for word the native correct-mode build. Gap: native-cpu MYNN, which runs pristine v8.4.2 | p12-bl-mynn (select the hex-cuda slot) |
| WB3 | Precipitable water divides the dry density by (1+qv) a second time (0.6 % low on average, up to 1.3 %) | Fixed in the hex-cuda WSM6, Thompson and Kessler slots: each compiles its CUDA text with the one density statement corrected (exact mode compiles the original text); the words equal the native correct-mode build's loop. Gaps: native-cpu microphysics, and Thompson-aerosols, whose hex-cuda slot is not in this tree | p2-wsm6, p3-thompson, p1-kessler, p4-thompson-aerosols (select the hex-cuda slot) |
| WB5 | WSM6 books the below-ground part of the falling cell at the flat mean while the remap moves it with a sloped profile, so column plus surface water is not conserved | Fixed in the hex-cuda WSM6 slot: the two rain-out statements use the remap's profile; whole columns equal the native correct-mode build word for word (surface precipitation changes, the atmosphere does not). Gap: native-cpu WSM6 | p2-wsm6 (select the hex-cuda slot) |
| RP3 | Fractional sea ice loses its ice temperature (the blended skin temperature is fed back); active whenever `config_frac_seaice`, which defaults to true | Named gap on the sea-ice and surface-layer slots | p18-seaice; p9-sfclay-mm5 (`sf_monin_obukhov`, `sf_monin_obukhov_rev`); p10-sf-mynn (`sf_mynn`) |

Named gaps are never refusals (coordinator ruling 2026-10-14). A finding is
applied whole or not at all: when any active slot it lives in has an
implementation that does not declare it (a native-cpu slot never can; a hex-cuda
slot declares the ids in its table descriptor's `corrections`), the run keeps the
native behaviour for that finding and lists it. Each gap appears in the run
record (`native-physics-request.json` and `native-physics-receipt.json`, under
`physics.correct_mode`: `mode`, `applied`, `skipped` and `gaps`, each gap with
`finding`, `slot`, `reason` "slot does not implement it", `owner` (the follow-up
owner), `scheme`, `backend` and `breakage`) and in a stderr warning at startup that
begins `CORRECT MODE: named gaps` and names every gap.

Not claimed: a combined correct-mode forecast verified against a combined native
referee. Each cluster built and checked against its own native referee; one
native build carrying every cluster's corrections is a named follow-up (owner:
the MPAS integrator).

The fork provider accepts the same mode controls. Its library default and all
native parity launches remain exact. Correct mode records the table's named
`RRFS-CORRECTION-INVENTORY` gap and retains selected fork behavior: the v8.4.2
finding inventory is not a qualified fork correction inventory. No v8.4.2
scientific correction is applied implicitly to the GSL driver. The fork owners
and integrator must qualify that separate inventory before claiming corrections.
