# MPAS column-batch physics seam

`woof.core.physics.run_mpas_column_batch` builds a persistent physics
seam for an MPAS CUDA dynamical core. The seam runs woof's ARW physics
orchestration on the caller's columns and hands back raw A-grid
tendencies. Implementation: `woof/core/mpas_column_batch.py`.

## Data contract

- Every atmospheric, tendency, surface, soil, and microphysics floating
  array is C-contiguous CuPy float32. Category and index arrays are
  int32.
- Layout is `[level, column]`, column fastest. `k = 0` is the lowest
  model level. Interface fields are `[level + 1, column]`.
- Cadence and elapsed-time bookkeeping stays exact: integer step counts
  plus float64 scalar seconds. Cadence state never enters a float32
  field array.
- Species set follows the constructor's `microphysics_scheme`
  (default `"wsm6"`, everything below unchanged; `"p3"` selects
  mp_physics=50, `"thompson_aero"` selects mp_physics=28):
  - `"wsm6"`: the full WSM6 six, `qv/qc/qr/qi/qs/qg`. The MPAS init
    carries `qv/qc/qr`; the seam's state zero-initializes the frozen
    categories once at construction and phase 2 evolves all six in the
    caller's arrays from then on.
  - `"p3"` (2026-08-31): WRF's own mp=50 transport, EIGHT scalars --
    `qv/qc/qr/qi` plus `ni/nr` (ice and rain number) and the rime pair
    `qir/qib` (rime mass/volume). P3 has one ice category: `qs`/`qg`
    are refused by name in both phases, phase 2 refuses `rho_dry`
    (P3 derives density from the EOS pressure in-kernel), receipts are
    the five-slot set (`rainncv/snowncv/sr`, no graupel key), and the
    cross-step supersaturation carriers `th_old`/`qv_old` ride the
    restart payload as seam-owned state. A `"p3"` restart identity
    carries `"microphysics": "p3"`; the `"wsm6"` identity is unchanged,
    so stored WSM6 payloads keep restoring and cross-scheme restores
    refuse on the identity gate.
  - `"thompson_aero"` (2026-09-01): WRF's own mp=28 transport, ELEVEN
    scalars -- WSM6's six masses `qv/qc/qr/qi/qs/qg` plus `ni/nr/nc`
    (ice, rain and cloud droplet number; `nc` is prognostic here) and
    the aerosol number tracers `nwfa/nifa` (water- and ice-friendly).
    No `nbca` (WRF zeroes black carbon without `wif_input_opt=2`,
    which WOOF-ArWen refuses by name) and no rime pair; both directions of
    slack refuse by name. Phase 2 refuses `rho_dry` (Thompson builds
    density from the EOS pressure in-kernel), receipts are the
    seven-slot WSM6 shape (`rainncv/snowncv/graupelncv/sr`; the
    buckets carry `GRAUPELNC`), the WSM6 radius triple rides the
    payload, and the surface aerosol emission pair `nwfa2d/nifa2d`
    (INTENT(IN) to every microphysics call) rides it as seam-owned
    state. FIRST CONTACT: the first `run_phase2` of a freshly
    constructed seam runs thompson_init on the caller's bound arrays
    with the caller's own `z_interface` -- WRF's two presence tests,
    the synthetic CCN/IN profile for whatever the caller did not bring,
    and `nwfa2d` from the lowest level either way (thompson_init's form
    when the fill ran, real.exe's climatology form when the caller
    brought aerosol; `nifa2d` is zero in every WRF branch).
    `seam.aerosol_init_receipt` reports which branches ran and the
    payload's `scalars["aerosol_init"]` carries it, so a restored seam
    never re-fills a checkpointed field. The seam stages no WIF
    climatology and requires none: a batch that brings no aerosol data
    is a valid batch, and a batch that brings its own is never
    overwritten. A `"thompson_aero"` identity carries `"microphysics":
    "thompson_aero"`.

## Construction

```python
from woof.core.physics import run_mpas_column_batch

seam = run_mpas_column_batch(
    n_levels=55, n_columns=16384, dt=120.0,
    radiation_seconds=600.0,          # legacy RRTMG cadence
    surface_pbl_seconds=120.0,        # revised MO + Noah-MP + YSU
    cumulus_seconds=120.0, cumulus_scheme="gf",   # or "kf", or None
    start_time=utc_datetime, latitude_deg=lat, longitude_deg=lon,
    terrain_height_m=hgt, z_interface_nominal_m=z_nominal,
    p_top_pa=p_top, dx_m=cell_spacing_m,
    landmask=..., ivgtyp=..., isltyp=..., vegfra=..., tsk=...,
    tmn=..., soil_temperature=..., soil_moisture=...)
```

- The constructor validates everything before touching the device. A
  misspelled keyword refuses with its name.
- `xland=` is the native surface classification and wins VERBATIM when
  supplied: the seam never re-derives it. The pinning case is MPAS
  category-15 fractional sea ice carrying native `xland = 1` on columns
  whose landmask is 0 - the landmask derivation would classify them
  open water and run no surface at all, while the native value routes
  them to the ice surface. Omitting `xland` keeps the documented
  fallback: WPS landmask (1=land/0=water) derives WRF XLAND
  (1=land/2=water). `seam.surface_classification` is the receipt: it
  names which source decided (`native` or `derived-from-landmask`) and
  counts every class - sea ice, open water, NOAHMP_SFLX land,
  NOAHMP_GLACIER land.
- `xice_threshold=` is the Noah-MP sea-ice classification threshold.
  WRF's Registry default 0.5 stands unless the caller's ice analysis is
  fractional (the collaborator case runs 0.02). Columns at or above the
  threshold take WRF's sea-ice skip whatever XLAND says; the value
  joins the seam identity, so a restart across a different threshold
  refuses.
- Glacier columns run. An active land column whose `ivgtyp` equals
  `ISICE_TABLE` (category 15) dispatches to the ported NOAHMP_GLACIER
  column (`woof/core/noahmp_glacier.py` host authority,
  `woof/core/kernels/noahmp_glacier.cu` device batch, bitwise twins),
  never to NOAHMP_SFLX and never silently to the water path.
  `seam.last_noahmp_census` reports the per-class counts of the most
  recent surface step plus `glacier_path`, the execution provenance of
  the glacier dispatch.
- The seam owns the sub-cadences. Each cadence must be a positive
  integer multiple of `dt`. Radiation follows WRF's
  `MOD(ITIMESTEP, STEPRA) == 1` calendar; surface/PBL and cumulus
  follow `MOD(ITIMESTEP, STEP*) == 0` with the mandatory step-1 call.
- `cumulus_scheme="gf"` requires `cumulus_seconds == dt`. That is WRF's
  own pinned `cudt = 0` law for Grell-Freitas.
- `gf_ishallow=` toggles GF's shallow scheme (CUP_gf_sh). The default is
  ON for `cumulus_scheme="gf"` because native MPAS v8.4.1 hardwires
  `ishallow = 1` (mpas_atmphys_vars.F:340) -- shallow OFF was the
  pre-parity behaviour and remains reachable only as an explicit
  `gf_ishallow=0` A/B arm. Refused for non-GF schemes.
- `dx_column_m=` optionally supplies per-column dx (metres, one value
  per column) for GF's scale-awareness on variable-resolution meshes,
  where the scalar `dx_m` is wrong for most of the globe. Native
  feeds `len_disp / meshDensity**0.25` per cell
  (mpas_atmphys_driver_convection.F:718); WRF's GFDRV itself takes
  dx(i,j), so this is the WRF interface, not an extension. Omitting it
  keeps the scalar `dx_m` for every column. GF is the only consumer;
  the scalar `dx_m` still feeds everything else.
- `z_interface_nominal_m` is the 1-D nominal vertical mesh. It supplies
  the WRF `fnm/fnp` interface-interpolation weights the legacy RRTMG
  temperature prep consumes.

## Phase 1: pre-RK tendencies

```python
out = seam.run_phase1(dt=120.0, u=u, v=v, theta=theta,
                      pressure=p, pressure_interface=p8w,
                      z_interface=z8w, w=w, rho_dry=rho,
                      qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg,
                      rthdynten=rthdynten, rqvdynten=rqvdynten)
```

- Runs legacy RRTMG, the revised-MO surface layer, Noah-MP, YSU, and
  GF/KF in WRF order, each on its own cadence, through the same
  `PhysicsDriver.compute` the ARW step calls.
- Returns `MpasColumnPhysicsTendencies`: raw A-grid `du/dv` (m s-2),
  `dtheta` (K s-1), `dqv/dqc/dqr/dqi/dqs` (kg kg-1 s-1), `dqg`
  (always zero, no phase-1 producer), and `h_diabatic` (K s-1).
- The returned set is the complete currently-held forcing: on non-due
  steps the held radiation and cumulus rates are returned unchanged,
  which is exactly what the ARW step integrates.
- No ARW mass coupling, no A-to-C face interpolation, no `cp.roll`, no
  map factors. The batch state pins `c1h = 1`, `c2h = 0`, total dry
  mass 1.0 and unit map factors, so the driver's mass-coupled scalar
  stacks are bit-identical to the raw physical rates. Momentum comes
  from the driver's retained raw YSU output, before
  `couple_ysu_tendencies` interpolates faces.
- `h_diabatic` is the previous phase-2 call's retained microphysics
  heating. ARW adds it to every RK theta tendency
  (`woof/core/dycore.py`, `add_h_diabatic_tendency`). It is reported
  separately, never folded into `dtheta`; apply it or decline it
  explicitly.
- `rthdynten=`/`rqvdynten=` ([nz, ncol] float32, K s-1 dry theta and
  kg kg-1 s-1) are GF's RTHFTEN/RQVFTEN advective forcing lanes --
  native MPAS v8.4.1's own construction (mpas_atm_time_integration.F:
  6936 advective theta_m rate, :2789 moist-to-dry conversion), computed
  at the END of the caller's previous dynamics step. `None` zeroes the
  lane, which is native's t=0 tend_physics state and the pre-upgrade
  behaviour. The seam copies both into seam-owned persistent buffers;
  GF's RTHBLTEN/RQVBLTEN lanes come from the driver's own retained raw
  PBL-slot rates and need no caller input.

### Which lanes each path actually fills

The four lanes are driver-held, so what reaches GFDRV depends on which
driver is bound. Recorded so the ARW and MPAS answers are not assumed to
be the same:

- `gf_rthblten` / `gf_rqvblten`
  - MPAS seam: the retained raw YSU rates.
  - ARW: the retained raw rates of whichever scheme holds the PBL
    slot.
- `gf_rthdynten` / `gf_rqvdynten`
  - MPAS seam: the caller's exported advective rates.
  - ARW: the dycore's own export. It captures pure theta and qv
    advection at RK stage 1 of every step into
    `state.rthften`/`state.rqvften`, which the driver binds at
    construction: the same one-step producer/consumer lag `h_diabatic`
    has, and pure advection on both halves (`module_cumulus_driver.F:867`
    pre-folds RTHRATEN+RTHBLTEN for G3/NTiedtke only, never for
    GFSCHEME).
- `gf_dx_column`
  - MPAS seam: per-column dx.
  - ARW: the scalar `cfg.dx`.

Every PBL scheme the driver dispatches -- YSU (1), MYJ (2), MYNN (5),
Shin-Hong (11) and SASE -- hands `couple_ysu_tendencies` the same raw
`dtheta`/`dqv` mapping, and those ARE WRF's RTHBLTEN/RQVBLTEN whichever
scheme filled them, so `PhysicsDriver._couple_pbl_slot` retains them at
one shared site rather than per scheme. None is a special case and none
is left feeding zeros.

RTHFTEN/RQVFTEN on the ARW path WAS the one remaining lane where WRF fed
something and this engine fed nothing. It was a dycore-export gap rather
than an adapter gap (the adapter read the lane the moment a driver set
it) and the dycore now sets it. Every lane GFDRV takes is live on both
paths.

The ARW pair is restart-carried state (`state/rthften`,
`state/rqvften`), because the producer is a dycore stage that has not
run yet when a resume reaches its first cumulus call. The MPAS seam's
buffers stay REBUILT: the caller refills them inside every
`run_phase1`, so a resumed seam overwrites whatever a checkpoint could
have carried.
- Read-only guarantee: phase 1 leaves every input array byte-identical.
  Negative `qv` values are accepted and clamped `max(qv, 0)` in seam
  scratch, matching MPAS bulk physics. The seam never refuses them.
- The returned arrays are seam-owned buffers, valid until the next
  phase-1 call. Copy them to hold them longer.

## Phase 2: post-RK WSM6, in place

```python
receipt = seam.run_phase2(theta=theta, qv=qv, qc=qc, qr=qr, qi=qi,
                          qs=qs, qg=qg, pressure=p, rho_dry=rho,
                          z_interface=z8w, refl_10cm_due=False)
```

- Runs WSM6 through `woof.core.microphysics.apply` plus
  `PhysicsDriver.accept_microphysics`, the ARW post-RK pair.
- Updates `theta` and all six species in the caller's memory through
  zero-copy views. `pressure`, `rho_dry`, `z_interface` are read-only.
- The caller's integrator clamps scalar state before this call (pinned
  contract); the seam applies no clamp here.
- The theta update carries WRF's `mp_tend_lim` clamp and refreshes the
  retained `h_diabatic`.
- Returns per-call `[column]` copies: `rainncv`, `snowncv`,
  `graupelncv` (mm) and `sr`.
- `refl_10cm_due=True` is WRF's history-step `diagflag`: the scheme
  adapter computes REFL_10CM inside the same microphysics call from its
  post-call temperature and the unchanged prepared pressure
  (`woof/core/refl.py`, PROVENANCE D2 -- the point where WRF and
  native MPAS-A compute `refl10cm`), and the receipt gains
  `refl_10cm`, a seam-owned `[nz, column]` float32 device copy.
  Computing it outside the call would pair post-scheme temperature
  with re-derived post-scheme pressure -- a different field than WRF
  defines. Never carried across steps; not restart state.
- Strict alternation: phase 1 then phase 2, every step. Out-of-order
  calls refuse.

## Persistence

Held radiation rates, held cumulus rates with KF NCA holds and expiry,
KF `w0avg` trigger history, Noah-MP surface and soil state, surface
layer state, precipitation buckets (`RAINNC/SNOWNC/GRAUPELNC` from
phase 2, `RAINC` from cumulus), WSM6 effective radii, `h_diabatic`,
call counters and the model clock all persist on the seam across calls.
`seam.accumulated_precipitation()` reads the buckets;
`seam.call_counts` reads the counters.

## Restart

- `seam.export_state()` returns a host-side payload of every persisted
  item. Legal only at a step boundary (before a phase-1 call).
- Payload schema `mpas-column-batch-v2` (2026-08-31): the scalars carry
  `"carriers"`, the radiation CarrierContract provenance, beside the
  held radiation buffers in the array manifest. Without it a restored
  seam refused its first radiation-not-due step ("GLW has no
  producer"); the schema string rides the identity, so a v1 payload
  refuses by name at the identity gate.
- `seam.restore_state(payload)` on a freshly constructed seam with the
  identical configuration restores everything in place. Identity
  mismatches, unknown keys and missing keys refuse.
- A `"thompson_aero"` payload's scalars also carry `"aerosol_init"`,
  the first-contact receipt; an mp=28 payload without it refuses by
  name. The `"wsm6"` and `"p3"` payload layouts are unchanged (gated
  by `test_wsm6_and_p3_payload_layouts_are_unchanged_by_the_mp28_arm`).
- A restored seam continues bit-identically. Gated by
  `tests/test_mpas_column_batch_gpu.py::
  test_restart_round_trip_continues_bit_identically`.
- The transported species and `theta` live on the MPAS side and restart
  with the caller's own state; the payload deliberately excludes them.

## Stated copies and documented divergences

- Phase 1 copies each input once into persistent seam buffers and the
  harvested tendencies once into the output buffers. Phase 2 copies
  nothing in or out of theta/species.
- Phase 2 feeds WSM6 `alt = 1/rho_dry` and `php = z_interface*g`; the
  WSM6 adapter's own `rho = 1/alt` and `z = php/g` round trips cost at
  most 1 ULP.
- Physics density is `(1 + qv)*rho_dry` where WRF phy_prep spells
  `(1 + qv)/alt`; one multiply against the caller's native dry density
  instead of two reciprocals.
- `fnm/fnp` come from the nominal 1-D vertical mesh, not from a
  per-column mesh. WRF's own weights are 1-D by construction.

## Parity proofs (node 5, RTX 4080, CUDA 13.1, 2026-08-10)

`tools/mpas_seam_proofs.py`; receipt at
`evidence/mpas-seam-proofs/mpas_seam_proofs.json`, provenance at
`evidence/mpas-seam-proofs/provenance.json`.  The harness runs from a
`git archive` tree with no repository, so the receipt's own `git_sha`
field is empty; the provenance file names the commit, tree, woof
subtree and harness blob the numbers belong to, and records that a
separate re-run of the committed harness on node 5 reproduced the
receipt byte for byte.  The harness runs the
seam against the REAL ARW physics step (a genuine mass-coordinate
`DomainState` driven exactly as `woof/core/dycore.py:2324-2325` and
`:2500-2512` drive it) on identical column states, 40 levels x 32
columns, dt 120 s, 90 steps (3 h), both KF (600 s) and GF (120 s)
cumulus arms, radiation 600 s, surface/PBL 120 s.

- Determinism control: the ARW leg run twice is bit-repeatable on the
  device (full driver manifest), so bitwise comparison is meaningful.
- Parity: on an f-plane column set with unit map factors, flat terrain,
  x-uniform winds and a uniform coordinate, every omitted coupling is
  the identity or a pure factor.  6368 (KF) / 6365 (GF) comparisons over
  3 h, zero failures: per-component `fl(chm_ARW * seam_raw)` equals the
  ARW coupled stacks bit for bit, retained raw YSU momentum and raw
  radiation rates equal bit for bit, and the full raw driver manifest
  (surface, soil, holds, KF trigger history, RAINBL, buckets) equals bit
  for bit at steps 1, 45, and 90.
- Cadence: radiation due at ITIMESTEP 1, 6, ..., 86 (18 calls in 3 h),
  held byte-identical between dues, changed at every due; KF cumulus 19
  calls with NCA holds; buckets monotone and bit-equal to the ARW
  accumulators; counters equal at every step.
- Restart: export at the step-45 boundary, restore into a fresh seam,
  45 further steps bit-identical to the uninterrupted seam.  Proven on
  the stock seam (765 comparisons) and the bound arm.
- Phase 2: after a per-step transport stand-in perturbation, species,
  h_diabatic, per-call receipts, effective radii, and buckets are bit
  identical to the ARW post-RK pair on the same perturbed state.  The
  two documented marshalling round trips moved 96/1280 (alt) and
  192/1312 (ph) input elements by exactly 1 ULP, and the WSM6 adapter's
  inverse derivations (`rho = 1/alt`, `z = php/g`) cancel them: measured
  output drift over the full 3 h is zero on every tracked field.
- Instrument validation: five deliberate breakages each go red (held
  radiation drop, bucket corruption, one-ULP silent restart corruption
  of `fields/tsk` detected as a 1-element divergence on the first
  post-restore step, identity-coupling break `c2h = 1e-4`, and a
  radiation cadence corruption caught by both schedule and counters).

Single-pressure callers: a caller that omits ``exner=`` gets exner
derived from the supplied pressure.  ARW derives pi_phy from the EOS
pressure while feeding physics the hydrostatic pressure; feeding one
hydrostatic pressure for both moves layer temperature by up to ~0.37 K
(exner max delta 1.2e-3) and materially changes surface-layer and PBL
rates (measured: LW heating max delta 4.5e-5 K/s on a 3.5e-4 K/s scale).
An MPAS caller is self-consistent with its own single pressure; pass
``exner`` explicitly when matching ARW behaviour is the goal.

## Physics mode (2.9.0)

`MpasColumnBatchPhysics` and `MpasColumnSuitePhysics` take
`physics_mode` (`"correct"` or `"exact"`) and `physics_correct_skip`
(finding ids). `physics_mode=None`, the default, is the engine default:
`"correct"` from 2.9.0, and `"exact"` (forced, recorded) in a strict
process (`GPUWM_WRF_EXACT=1`). The resolution sets every correction key of
the seam's RunConfig explicitly, so `"exact"` builds the configuration and
the restart identity this seam built before the keys existed, whatever the
RunConfig defaults are. A caller proving identity against native MPAS
passes `physics_mode="exact"` explicitly: passing nothing now runs
correct. `correct_mode_receipt` records the mode and the findings it
changed; the restore identity carries `physics_correct` only in correct
mode, so a restore across modes is refused.

## Orchestration identity

`MpasColumnBatchPhysics._PHASE1_ORCHESTRATION` is
`PhysicsDriver.compute` and `_PHASE2_MICROPHYSICS` is
`woof.core.microphysics.apply`, asserted by object identity in
`tests/test_mpas_column_batch.py`. The seam is not a per-kernel wrapper
set and cannot silently become one.
