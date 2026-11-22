# TEMPO microphysics port (lane tempo-mp)

TEMPO (Thompson-Eidhammer Microphysics Parameterization for Operations) is
the successor to Thompson microphysics and the microphysics of RRFS v2. This
lane ports it to WOOF as a GPU option. The target is one scheme that both
regional WOOF and hex can call, so hex can score an RRFS-v2-style physics
profile beside the HRRR-family profile.

## Licence gate (passed)

- Source: https://github.com/NCAR/TEMPO.
- Port target: commit `9adf9ef2827104640125f0c322e1218fce74e94c`
  (tag `tempo_v3.0.3`). This is the commit RRFS v2 pins. The chain is:
  NOAA-EMC/rrfs-workflow branch `rrfs-mpas-jedi` (tip `85ea55c0`),
  `sorc/MPAS-Model`, RRFSx/MPAS-Model `1d6e370e`,
  `src/core_atmosphere/physics/physics_noaa/TEMPO`, then `9adf9ef`.
- Licence: Apache-2.0, "Copyright 2026, NOAA, UCAR/NCAR CU/CIRES".
  - It has been Apache-2.0 since `e41f020` (v3.0.0). Before that it was MIT,
    which is also compatible.
  - The source files carry no per-file headers that say otherwise.
  - The incomplete-gamma routines are written from DLMF 8.7.1 and the
    modified Lentz algorithm. Numerical Recipes is cited only as "see also";
    its code is not transcribed.
  - GitHub's API reports the licence as "Other", only because of the custom
    copyright line.
- Main has moved on to v3.2.5 (`c62699fb`, 2026-09-25), with hail tuning and
  version bumps. The port follows the RRFS pin. Moving to a newer TEMPO is a
  re-diff of `module_mp_tempo_main.F90` against `9adf9ef`.

## What TEMPO adds or changes against WOOF's Thompson

WOOF has two Thompson schemes:
- classic mp=8, WRF v4.6.1, one-moment graupel, fixed droplet number;
- aerosol-aware mp=28, which adds prognostic nc, nwfa and nifa, Koop and
  DeMott ice nucleation, and aerosol scavenging by precipitation, still with
  one-moment, fixed-density graupel.

TEMPO 3.0.3 adds or changes the following.

1. **Hail-aware graupel (Jensen et al. 2023).** This is the main difference.
   - Graupel carries a prognostic number `ng` and a rime volume `qb`, so
     density is predicted.
   - Density falls into 9 classes from 50 to 800 kg m-3 (`rho_g`, `idx_bg`).
   - Graupel fallspeed comes from a Heymsfield Best-Reynolds relation that
     depends on density and air viscosity (`a_coeff`, `b_coeff`).
   - The rain-graupel collection tables gain a density axis
     (37x37x9x37x37).
   - Size limits: graupel median volume diameter is clamped to 25.4 mm; the
     floor is the minimum rain-drop diameter `d0r`.
   - New volume tendencies (`pbg_*`) cover riming, rain freezing, collection
     and melting.
   - mp=8 and mp=28 have none of this.
2. **Aerosol-aware by default.** The 2-moment cloud water, nwfa/nifa and
   CCN-activation table match mp=28 in kind. The process code is TEMPO's own
   refactor, not WRF 4.6.1's.
3. **Explicit cloud condensation and evaporation** (`cloud_condensation`)
   after the source and sink update, with its own droplet activation pass.
4. **Rain number from autoconversion** is scaled by `autocon_nr_factor`
   (10). Larger values produce fewer drops.
5. **Rime above freezing turns into rain** when dt >= 120 s
   (`timestep_conversion_rime_to_rain`).
6. **Snow fallspeed changes:**
   - while melting, snow falls at a mix of snow and rain speed weighted by
     the mass fraction;
   - the riming boost (`vtboost`) is applied in the fallspeed routine.
7. **Optional semi-Lagrangian sedimentation** (Juang and Hong 2010,
   `semi_sedi_flag`, default off).
8. **Optional machine-learning cloud droplet number** (`ml_for_nc`,
   `ml_for_bl_nc`).
   - A two-layer neural network predicts nc in one-moment mode and for
     boundary-layer clouds in the effective-radius calculation.
   - It is off in the RRFS configuration and not yet ported.
9. **New diagnostics:**
   - maximum hail diameter, at the surface and in the column;
   - freezing-rain precipitation;
   - frozen fraction;
   - median volume diameters for rain and graupel;
   - optional 10 cm reflectivity from melting snow and graupel
     (`refl10cm_from_melting_flag`).
10. **Structure.** TEMPO is a host-agnostic, column-based Fortran module
    (CCPP, MPAS, WRF), with one `tempo_main` per column and
    double-precision lookup tables.
    - The tables are `qr_acr_qg`, `qr_acr_qs` and `freeze_water`, about
      970 MB in total, plus `ccn_activate.bin`.
    - Every rate is a named `ty_tend` array, summed in `sum_tendencies` after
      the over-depletion limiter.
11. **First-call state.**
    - A Fortran SAVE flag, `first_call_main`, runs `graupel_init` and the
      synthetic aerosol initialization on the first call only.
    - The port takes this from the host (`TempoCtx.first_call_main`) and
      applies it to the first column only, as the Fortran does.

## Port design

- Kernels live in `woof/core/kernels/tempo/`, and the same headers compile
  for both the CPU and the GPU:
  - `tempo_common.cuh`: the blob layout, accessors and utilities;
  - `tempo_processes.cuh`: process rates and aerosols;
  - `tempo_diags.cuh`: diagnostics;
  - `tempo_column.cuh`: `tempo_main`, check/update, sedimentation and
    fallspeeds.
  - Device builds use nvcc or NVRTC, which is the engine path through CuPy
    RawModule. The CPU float64 reference is the host build of the same
    source.
- The port is a line-by-line transliteration of TEMPO 9adf9ef:
  - Fortran names and 1-based indices are kept;
  - quirks are kept and commented, for example `cloud_condensation` returns
    rather than cycling on |ssatw| < eps, and the aerosol-init height uses
    `dz1d(k)`.
- All parameters and tables sit in one float64 blob laid out as
  `struct TempoBlob`. Today the blob is filled from TEMPO's own `tempo_init`
  state (`tools/tempo_ref/tempo_ref_driver.F90`). This checks the kernel port
  separately from the table builders.
- Execution is one thread per column. Each column's work arrays live in
  global memory (`TempoWork`, 143 KB per column at nz <= 100).

## Proof (box B, RTX 5090, 2026-10-05)

- **Reference:** unmodified TEMPO 9adf9ef built with gfortran 13.3,
  `-O0 -fcheck=all -fdefault-real-8`, driven through `tempo_run` on the
  59-level MPAS test column. There are six scenarios:
  - S1: as given, dt 20 s, 90 steps;
  - S2: qv raised 15% on levels 8-30 and qc seeded on levels 5-20;
  - S3: S2 with semi-Lagrangian sedimentation;
  - S4: S2 at dt 60 s;
  - S5: S2 with the column 8 K warmer;
  - S6: S2 with the column 18 K colder.
  The reference reproduced bitwise on a second box (same SHA-256).
- **CPU float64 port vs Fortran:** bitwise identical on S1, S2, S4, S5 and
  S6 for all 17 state fields, the 7 surface diagnostics, and reflectivity and
  effective radii at every step.
- **GPU port:**
  - 4096 replicated columns are bitwise identical to each other;
  - two full runs are bitwise identical (reproducible);
  - against Fortran, the worst relative difference over all fields, levels
    and steps is 1.9e-10 (S1, S2, S4, S5, S6).
  - Stated tolerance: 1e-6 relative, with per-field floors (see
    `compare_columns.py`).
  - The difference comes from CUDA libm against glibc pow/exp in the last
    ulp. It starts at 1e-15 and stays near 1e-10 over 90 steps.
- **S3 (semi-Lagrangian):** the GPU differs from Fortran by order 1. The
  same CPU port that matches Fortran in S1, S2 and S4 diverges just as far
  when the initial graupel is scaled by (1 + 1e-15). TEMPO's semi-Lagrangian
  option therefore amplifies rounding by itself. It is off by default and in
  RRFS. This is a finding to report upstream, not a port defect.
- **Engine compile path:** the same headers built through NVRTC (CuPy
  RawModule, `tempo_nvrtc_check.py`) give bitwise the nvcc result on all six
  scenarios. `woof/core/tempo_runtime.py` (3-D pack, launch and unpack)
  runs a 256x256x59 domain:
  - 0.26 s per call (4.0 us per column) on one RTX 5090, with float32 or
    float64 storage;
  - batched to an 8 GB work cap.
- **Tables:** tables built from the pinned source (`build_tables`, -O2) match
  the tempo_v3.1.0 release assets to 2.2e-12 relative, which is compiler
  rounding.
- **Speed:** first cut, no optimization yet:
  - 6.9 us per column-step in float64 with 4096 columns;
  - 4.0 us with 65k columns.

## Column oracle (W1, RTX PRO 6000, 2026-10-07)

The 2026-10-05 numbers above are superseded: the GPU is now TEMPO's Fortran
word for word.

- **Harness:** `tools/tempo_ref/oracle/`.
  - `build_reference.sh` clones TEMPO at 9adf9ef, builds it unmodified with
    its own `Makefile.gfortran` flags (gfortran 13.3, glibc 2.39), runs
    `tempo_init` on the pinned tables and writes the reference.
  - `make_columns.py` builds 127 columns in three sets: dt 20 s (60 steps),
    60 s (30 steps) and 150 s (12 steps, the rime-to-rain path). Regimes:
    convective (supercell, MCS stratiform, pulse, elevated hail), stable
    night (clear inversion, fog, stratus drizzle), snow and ice (snow,
    freezing rain, sleet, lake effect, arctic ice fog), ocean (tropical warm
    rain, marine stratocumulus, eyewall), high terrain (70 to 78 kPa
    surface), and 19 edge cases. The edge cases include first-call aerosol
    and graupel initialisation, masses at r1, mass without number, graupel
    density at and outside its bounds, hail above 25.4 mm, w from -25 to
    +55 m/s and negative mixing ratios.
  - `tempo_oracle_driver.F90` runs the referee (all columns of a set in one
    `tempo_run` call per step).
  - `run_woof_oracle.py` runs the engine's `TempoRuntime` on the same input.
    It compares all 26 outputs of every step bitwise: 14 updated fields,
    8 surface diagnostics and 4 column diagnostics.
  - `tempo_cpu_oracle.cpp` runs the port's host build, which separates a port
    logic difference from a device arithmetic difference.
- **Result:** 6,763,008 values, 0 ULP. This holds free-running and per-step
  replay, under `GPUWM_WRF_EXACT=1` and default arithmetic, and with the
  Fortran init dump and with the engine's Rust `tempo_init`.
- **Two port defects found and fixed** (`tempo_common.cuh`,
  `tests/test_tempo_device_libm.py`):
  1. `tnint` was `floor(x + 0.5)`, which rounds 0.49999999999999994 to 1.
     Graupel at the 50 kg m-3 density bound takes the next density class,
     which made a 1.1 K error in one step on the CPU and the GPU alike.
  2. The device build called CUDA's exp, log, pow and log10. They now go to
     WOOF's own float64 routines (`kernels/glibc_flt64.cuh`, plus a log10 on
     its log) through `tempo_exp` / `tempo_log` / `tempo_pow` /
     `tempo_log10`.
     - With the NINT fix alone, the CUDA functions left 120 of 127 columns
       off: up to 9e-10 relative per step in the regular regimes.
     - Free-running, threshold branches flipped (convective columns 0.59
       relative after 60 steps).
- **Documented TEMPO defect, not copied:** `frozen_fraction` is never
  assigned when `tempo_main` returns early (no condensate, no
  supersaturation: `module_mp_tempo_main.F90` line 386 returns before
  line 747). `tempo_run` therefore copies an undefined value (5e-324 on W1).
  WOOF returns 0, which is TEMPO's own formula for zero precipitation. The
  oracle counts these 342 cells apart and requires WOOF's +0 there.
- **Scenarios S1 to S6** (`tests/test_tempo_runtime_gpu.py`) are now required
  to be bitwise. The 1e-6 tolerance existed only because of the two defects
  and is retired.
- **Not covered:** `refl10cm_from_melting` (off in RRFS v2) uses complex
  arithmetic with CUDA's hypot, atan2, sin and cos. The never-reached
  `cos` in the aerosol initialisation is also outside the oracle (`hgt(1)`
  is always 0, so `h_01` is 0.8).

## Plan from here

1. **Engine option.**
   - Add an mp selector for TEMPO (the number needs a ruling: WRF has no
     TEMPO number, MPAS calls it `mp_tempo`).
   - State fields are mp=28's (`nc`, `nr`, `ni`, `nwfa`, `nifa`) plus
     prognostic `ng` and `qb`, advected like the other moist scalars, plus
     the physics registry rows.
   - The adapter packs (nz, ny, nx) into column order. Batch the columns so
     `TempoWork` fits the memory budget, or slim it (step 4).
   - It returns `MicrophysicsDiagnostics` (rain, snow, graupel and ice
     accumulations, SR from frozen fraction) and stashes REFL_10CM and the
     effective radii.
2. **Init without Fortran.** Port the TEMPO init (`initialize_parameters`,
   bins, `compute_efrw`, `compute_efsw`, `compute_drop_evap`, `qi_aut_qs`)
   and the table readers to Rust, per the Python boundary. Check the result
   bitwise against the Fortran blob dump.
3. **Table builder.** Port `build_tables` to Rust or the GPU, checked against
   the pinned build at about 1e-12, or ship the pinned tables as recast-woof-data
   assets with SHA-256.
4. **Performance.**
   - Slim `TempoWork`: about 40 of its arrays are temporaries.
   - Fuse the check/update passes.
   - Move to a warp-per-column or k-parallel layout.
   - Add a float32 production path, gated against the float64 reference at
     a stated tolerance; double stays the reference.
5. **Real case.** Small regional WOOF domain, a few hours, TEMPO beside mp=8
   and mp=28, scored against MRMS (QPE and composite reflectivity). Maps go
   through `woof.rustwx`/rw_wrfbatch.
6. **Hex.** The same kernel behind the hex physics seam (lane h4a's
   interface), MPAS column order.
7. **Optional features** (not used by RRFS v2): ML cloud number, reflectivity
   from melting. Port them after the default path is in the engine.

## TEMPO v3.2.5 generation (upstream latest-as-default, 2026-11-09)

Since 2026-11-08 the latest upstream release of each scheme is
the default, and the bitwise pins stay selectable. For TEMPO:

- `tempo_version = "v3.2.5"` (NCAR/TEMPO c62699f, released 2026-10-06) is the
  default: `woof/core/kernels/tempo325/`. It carries every change since
  9adf9ef, including hail-hyperaware (`tempo_hailhyperaware`, on by default as
  upstream and the RRFS fork have it), the ice-sedimentation sub-step fix,
  the Koop `delta_aw` fix, the `pbg_rci` volume fix, the `pni_inu` aerosol
  sink, `roverrv = rdry/rv` and `is_first_step` on every column.
- `tempo_version = "v3.0.3"` (9adf9ef, the RRFS v2 pin) is
  `woof/core/kernels/tempo/`, unchanged byte for byte, and compiles with the
  options it always had. RRFS bitwise proofs name it.
- v3.2.5 reads `final_weight` in `sum_tendencies` where it may be unset
  (main.F90:1752-1755). The port defines it as 1 (no scaling); the referee
  (`tools/tempo_ref/oracle/build_reference_v325.sh`) carries the same one
  line and is labelled patched.
- The v3.2.5 init is the v3.0.3 Rust init plus `roverrv = rdry/rv`
  (`tempo_runtime.with_v325_parameters`): TEMPO's own v3.2.5 tempo_init dump
  differs from its v3.0.3 dump in that one record of 249, on the same tables.
- Column oracle, 127 built columns in three sets (6,763,008 words): the host
  build and the GPU runtime under strict numerics are both 0 words different
  from the patched native v3.2.5 Fortran, free-running and replayed; the
  hail split is active in 2,270 of the 6,216 column-steps.

## Fast numerics for v3.2.5 (woof.core.tempo_fast, 2026-11-09)

A fast-numerics process (the default) runs TEMPO v3.2.5 as binary32 with a tiled work area. A strict process
(`GPUWM_NUMERICS=strict` or `GPUWM_WRF_EXACT=1`) compiles `kernels/tempo325` exactly as before, and the pinned
v3.0.3 is never transformed.

- **Why:** the default ran the strict column with contraction allowed: binary64 throughout (1/64 rate on GeForce
  and RTX PRO cards), the glibc-word libm, and a per-column work area 40 to 80 kB long, so every warp access touched
  32 separate lines. On the 3 km HRRR recipe (Iowa 92x92x50, RTX 4090) a TEMPO call took 18.8 ms, against
  3.7 ms for a Thompson mp=28 call, and the step ran about 2x Thompson's. A clock64 segment profile of the binary32
  column put 70 % of the time in work-area traffic (zeroing alone was 19 %) and 8 % in the process rates.
- **What changes:** one checked source transform of the same headers, with no second copy of the physics:
  - `double` becomes `float` and every literal becomes a binary32 literal;
  - the blob is converted to binary32, and a table value that overflows is refused by name;
  - exp, log, log10 and pow use CUDA's binary32 intrinsics (the maths hex's fast TEMPO uses);
  - the work area and column state are tiled (32 columns per tile, 128-byte level stride), reached through a
    view shifted by the lane. The tiling alone leaves every word unchanged.
  - The pointer tables are cached per pointer set (the three pageable copies per call waited for queued kernels).
  - `WOOF_TEMPO_FAST_FLOAT=0` keeps the binary64 column in a fast process.
- **Proof:** strict column oracle vs patched native v3.2.5: 0 of 6,763,008 words different, free and replay.
  Fast words change by design. On the HRRR recipe at +1 h against MRMS, fast and binary64 score the same: FSS at
  27 km for 20 and 35 dBZ is 0.610/0.499 vs 0.605/0.489, and T2 RMSE is 2.17 vs 2.18 K.
- **Speed:** one TEMPO call takes 18.8 ms -> 5.5 ms. On the step, fast-step (10th-percentile) time is about
  29.7 ms vs 26.0 ms for Thompson mp=28, i.e. 1.14x; it was 50.7 ms before (2.0x).
