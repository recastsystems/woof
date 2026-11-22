# Grell-Freitas (`cu_physics = 3`): what a release integrator needs to know

Lane: `lane/gf-port`. Target: `integration/release-1.6`. This is the short
form. The conformance argument lives in the registry entry
(`woof/physics_registry_v2.json`,
`components.cumulus.options.grell-freitas`) and in the parity suites; this
file is only the set of things that will bite someone folding the lane in.

## 1. The label is `implemented-unverified`, and a run does not change that

`scientific_evidence: none`. Grell-Freitas is bitwise against WRF v4.6.1
at the whole-driver boundary, all 216 branch-targeted oracle columns on
both the float32 CPU authority and the CUDA translation unit, and all 280
realistic regime columns on the CUDA translation unit with nothing pinned
(WRF linked with the correctly rounded gamma WOOF ships), and that is
**conformance evidence, not scientific validation**. No woof/WRF forecast comparison (code
verification) is recorded for this scheme, and no GF forecast score
against observations (validation) is published here.

A first real-case trajectory (below) establishes that the scheme
integrates a real domain without diverging, and that a dual run of it is
byte-reproducible. It does **not** establish forecast skill, and it is not
grounds to move the label. Moving to `supported` needs a scored
comparison against obs (MRMS/ASOS), on the project's own
obs-skill-is-the-referee terms.

## 2. Selection surface

* `cu_physics = 3`, per domain, on the tree route only
  (`reachability: component-override`, the Shin-Hong/SASE posture).
* `cudt_minutes = 0` is **enforced, not chosen**:
  `validate_run_config` refuses a nonzero `cudt` for `cu_physics = 3` by
  name. GF runs on the model step (WRF's usual GF configuration,
  `STEPCU = 1`) and carries no NCA hold, so the KF cadence knob has no
  meaning here.
* `clos_choice = 0` (the 16-member ensemble closure) is the arm the
  oracle covers. 1-16 run one closure member alone, WRF's own code
  path: the kernel matches the float32 reference on all sixteen
  (`tests/test_gf_closure_choice_cuda.py`) and the load warns that no
  WRF run covers them. Other values are refused (above 16 reads past
  the 16-member closure array). `ishallow` is 0 or 1, both covered, 0
  is the WRF Registry default.
* Both Grell-family keys are refused wherever no Grell scheme is selected,
  so unrelated configs cannot carry them.
* **WOOF-ArWen structural seam:** 390 (`pbl = 0`, `cu = 3`) cells of the WRF
  v4.6.1 compatibility matrix are admitted by WRF and refused by WOOF-ArWen.
  WRF reads `KPBL = 0` there and indexes below the column base; WOOF-ArWen
  refuses rather than reproduces. This is the second named structural
  seam and it is deliberate.

## 3. The corrected-k22 default, and the flag that is not a config key

WRF's shallow k22 trigger is a `MAXLOC` over the array section
`heo_cup(2:kbmax)` whose result `module_cu_gf_sh.F` uses as an absolute
level index without adding the section offset. **The shipped kernel uses
the corrected indexing** (owner ruling, no inherited WRF bugs).

The WRF-faithful off-by-one exists only behind a launch argument
(`k22_wrf_faithful`) that the parity suites set directly. It has **no
`RunConfig` spelling**: deliberately, and there is a test that asserts
so. An integrator should not add one.

The restart identity binds the choice:
`grell-freitas-wrf461-gfdrv-corrected-k22-v2`. A checkpoint written under
the shipped identity cannot resume under a WRF-faithful build. Version 2 also binds the shallow closure top bound and lower-face W, so a version 1 checkpoint cannot silently resume with the changed defaults.

Measured ledger for the difference: k22 moves on 3 of 18 fixture cases,
all three rejected under both modes with identical `ierr`, and **zero**
output words differ at either boundary.

## 4. Engine-seam deviations (the plumb-list for any label upgrade)

The kernel is bitwise; these are about what the engine can hand it today.

1. **Forcing tendencies: CLOSED.** GFDRV sums advective
   (`RTHFTEN`/`RQVFTEN`), radiative (`RTHRATEN`) and boundary-layer
   (`RTHBLTEN`/`RQVBLTEN`) forcing into its forced state, and all four
   lanes are now fed, all read off the driver through `bind_driver`. The
   boundary-layer pair is the PBL slot's own raw dry-theta/qv rates,
   retained by `PhysicsDriver._couple_pbl_slot` whichever of YSU, MYJ,
   MYNN, Shin-Hong or SASE holds the slot, which is how WRF's own
   cumulus driver reads RTHBLTEN/RQVBLTEN. The advective pair is the
   integrator's export: on the ARW path the dycore captures pure theta
   and qv advection at RK stage 1 of every step, uncouples it to
   K s<sup>-1</sup> and kg kg<sup>-1</sup> s<sup>-1</sup>, and leaves it
   in `state.rthften`/`state.rqvften` for the next step's cumulus call,
   the same one-step producer/consumer lag `h_diabatic` has. (The MPAS
   seam's caller supplies its own.) The rate-of-change closure family
   therefore sees its advective forcing, the instability closures see
   the current state, the omega/moisture-convergence closures are fully
   fed, and with `ishallow = 1` the shallow `blqe` member sees a real
   `dhdt`.

   The export is **pure advection**, deliberately.
   `module_cumulus_driver.F:867` pre-folds `RTHRATEN + RTHBLTEN` into
   `RTHFTEN` for `G3SCHEME` and `NTIEDTKESCHEME` and **not** for
   `GFSCHEME`, which sums the three lanes itself
   (`kernels/gf.cu:4428`). A pre-folded RTHFTEN here would make GF
   integrate the boundary layer and the sky twice.

   **Restart consequence, stated plainly:** the pair is serialized
   state, so a `cu_physics = 3` checkpoint written before this landed
   carries two fewer state arrays than this build expects and is
   refused by name. Resume from a checkpoint this build wrote, or start
   the run again from its prepared state. No other configuration's
   checkpoints move.
2. **Convective momentum tendencies.** The kernel computes GF's
   `dudt`/`dvdt`; `CumulusResult` carries no momentum slots, so they are
   not coupled. Stock WRF 4.6.1's `module_physics_addtendc.F`
   `CASE(GDSCHEME,G3SCHEME,GFSCHEME)` adds the scalar slots only.
   The computed momentum rates still need diagnostic history output.
3. **Vertical velocity** is lower-face `w[k]` in strict and default arithmetic, matching GFDRV. The former adjacent-face average changed closure forcing and rain.

**MEASURED regime behaviour (2026-08-17).** Two 12 km single-domain 6 h
real-case twin pairs (150x120x49, tree route, KF control differing only
in `cu_physics`/`cudt`): under strong synoptic forcing (1974-04-03
12-18Z) GF's domain-mean RAINC is ~40% of KF's, ordinary inter-scheme
spread; under weak forcing (1999-05-03 12-18Z, Ohio valley) it is 1-2%
of KF's, nearly silent where KF still rains (KF RAINC max 2.2 mm, GF
0.18 mm). A column-level probe of the weak-forcing state (18,000 real
columns through `gf_gfdrv_stage`, four arms: shipped zero forcing,
HFX/QFX-reconstructed RTHBLTEN/RQVBLTEN, ±1 K/day radiative) found the
deep trigger rejecting every column in every arm, so the weak-forcing
silence is the bitwise scheme's own trigger/closure response to these
inputs, no fed forcing arm revives it, and feeding reconstructed BL
rates alone REDUCED convecting columns on the strongly forced state
(3162 → 2068 of 18,000; the dicycle closure subtracts BL-driven
instability by design). Any seam upgrade must therefore be validated
against obs skill, not against "more rain"; a user report of "GF is
broken / makes no rain" on a weakly forced case is this measured shape,
not an adapter defect.

## 5. VRAM: GF's backing store, and the cut that removed it

RETIRED 2026-08-21, and kept here because the mechanism is worth reading.

One GF thread owns one whole GFDRV column, so the per-thread local frame
USED TO BE the deep+shallow local-array stack: **22,416 B** at the nz=40
tier, rising 456 B per level. By the reservation law in
`docs/kernel_local_memory_bounds.md` the driver sizes the backing store to
the device's whole resident-thread capacity on the kernel's first launch
and holds it for the life of the process, so that frame cost

```
(22,416 - 1,024) * 1,536 *  70  =  2.14 GiB          [RTX 5070 Ti]
(22,416 - 1,024) * 1,536 * 170  =  5.20 GiB          [RTX 5090]
```

while the kernel only ever had 384 threads/SM in flight.

The column arrays now live in a global workspace `woof/core/gf.py` sizes
to the columns IN FLIGHT (`woof/core/kernels/gf.cu`, section "Per-thread
column workspace").  MEASURED on a development machine: the frame is 72 B, the
reservation is 4.0 MiB, and the workspace costs 422.1 MiB at the shipped
tile.  `woof check` prices the workspace under `NON-POOL` where it used
to price the frame; section 7 of `docs/kernel_local_memory_bounds.md`
carries the full before/after.

**Selecting GF is no longer what sets the local-memory term.** On a
`cu_physics = 3` tree the widest frame is now some other module's -- the
same one-thread-one-column shape in `ysu`, `shinhong`, `nssl2` and `refl`.

## 6. CLOSED: the frame that did not fit is gone

`configs/real74_4dom_gf.toml` did not fit, and the route registered here
was a code change. **It landed on 2026-08-21**, and by a better road than
the one this section proposed.

The proposal was to specialize `gf.cu`'s per-thread frame the way
`KF_KMAX`/`REFL_KMAX` were specialized. That would have divided the
reservation by the level ratio and left it proportional to `nz`. What
landed instead moves the column arrays **out of the local frame entirely**,
into a global workspace sized to the threads actually in flight
(`docs/kernel_local_memory_bounds.md` sections 7 and 8). The difference
matters: the reservation does not shrink, it **stops existing**, and it
stops moving with `nz` at all.

Measured on a development machine (RTX 5070 Ti, 70 SM x 1,536, sm_120):

| kernel | frame before | frame after | reservation before | after |
|---|---|---|---|---|
| `gf_gfdrv_stage` | 22,416 B | 88 B (NVRTC 13.0) / 72 B (13.3) | 2,200.0 MiB | 4.0 MiB |
| `ysu_column` | 9,232 B | 0 B (both builds) | 842.0 MiB | none |

YSU is in the table because it is what the GF cut uncovered: with GF out of
the way, YSU held the widest frame a **bare default run** launched, and
`bl_pbl_physics = 1` is the wizard's default.

### What the four-domain arm costs now

`woof check`, re-measured 2026-08-21:

| config | device-footprint projection |
|---|---|
| `real74_4dom.toml` (KF base) | 21.52 GiB |
| `real74_4dom_gf.toml` | **22.17 GiB** |
| `real74_2dom_gf.toml` | 11.05 GiB |

The GF arm is now **0.65 GiB** over the KF base, where this section
recorded 3.67 GiB (27.79 against 24.12). **Stated plainly: the acceptance
criterion written here (GF back *under* the four-domain base's footprint)
is NOT met.** It is 0.65 GiB over, because GF's column workspace (0.47 GiB
at this configuration) is a real allocation that the local frame's
reservation used to hide. The 3.2 GiB obstruction is gone; a smaller,
accurate one is in its place.

These three rows are measured on the **current Windows host, an RTX 3080
with 68 SMs**. The 32,607 MiB card the original table was measured on moved
to a development machine on 2026-08-16, so that box no longer exists and the two
tables are not the same instrument. The *delta* between configs is the
comparable quantity, not the absolute GiB.

Disk remains an independent and unchanged limit: four domains at these
cadences is ~244 GB of wrfout per arm, ~488 GB for a dual run, against
242 GB free on the largest volume. Not even one arm fits, and nothing may
be deleted. **Whether the four-domain GF arm is now runnable is therefore
still NOT MEASURED**: the memory obstruction is closed, the disk one is
not, and no four-domain GF run has been attempted.

GF parity is unaffected by the cut: the frame was allocation size, not a
value any loop reads, and the interleaved A/B in
`docs/kernel_local_memory_bounds.md` section 7 records 141,480 graded
words with 0 differing, both controls firing.

## 7. Restart: `cu_physics = 3` could not checkpoint until a4c06235

The GF selectable-wiring commit (`ea0cada3`) taught the restart manifest
GF's algorithm identity and its coefficient-table absence, but not GF's
**callable shape**. The adapter stores the `PhysicsDriver` that
`bind_driver` hands it (that is how it reads the held radiative rates)
and `_callable_state_check` walks one level into object attributes, so the
driver's own arrays appeared under the cumulus callable unclassified.
Every checkpoint of a GF run raised `RestartManifestError`. The first
real-case GF trajectory integrated 179 outer steps and died at its first
restart.

Fixed in `a4c06235`: `_driver` is classified rebuild-on-load in
`CUMULUS_CALLABLE_CONTAINERS`, on the KF `_history_state` precedent. Both
are back-references, not state. **Nothing of GF's is serialized as new
restart state, because GF holds none**: `cu_physics = 3` is a stateless
Task-1 callable with no NCA persistence, no trigger history (no `W0AVG`),
and no closure memory. Every GF quantity with memory between calls already
lived on the driver and was already serialized there: held rates through
`cu_rates`' `cu_rthcuten`/`cu_rqvcuten`/`cu_rqccuten`/`cu_rqicuten`
scratch slots, precipitation accumulators through
`cu_rainc`/`cu_raincv`/`cu_pratec`.

Gated by `tests/test_restart.py::
test_short_grell_freitas_restart_is_bit_identical` (20 steps + restart +
20 steps == 40 steps, FP32-bit-exact, on a state where GF actually
convects) and by the CPU-tier
`test_every_cumulus_adapter_attribute_is_classified`, which walks every
selectable cumulus adapter live.

**Integrator note:** a GF adapter that ever grows real state must be
serialized, not added to the container allowlist. The classification
comment in `woof/io/restart.py` says so at the point of use.

## 8. Test battery

```
tests/test_gf_deep_cuda.py      tests/test_gf_deep_parity.py
tests/test_gf_driver_parity.py  tests/test_gf_engine_smoke.py
tests/test_gf_gfdrv_cuda.py     tests/test_gf_shallow_cuda.py
tests/test_gf_shallow_parity.py tests/test_gf_wrf461_parity.py
                                -> 765 passed
tests/test_restart.py           -> 96 passed, 1 skipped
```

Cross-lane touchpoints that also carry GF assertions and must stay green:
`test_namelist_import.py` (native import of namelist `cu_physics = 3`),
`test_wrf461_compatibility.py` (the cumulus axis: 3,840 -> 5,760 matrix
cells, and the `pbl = 0` structural seam), `test_config_freeze.py`,
`test_experiment.py`, `test_physics_vertical_preflight.py`.
