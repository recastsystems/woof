# WRF 4.8.0 MYNN-EDMF column (`bl_mynn_version = "wrf_480"`)

The source is WRF v4.8.0 `06d4240ae989cc3e50af412bb472df3d9048783c`, whose
MYNN-EDMF is the submodule NCAR/MYNN-EDMF `8f8e8b59021249949c04d1686edc5a02ca325627`.
That commit carries no licence file of its own; it is kept exactly as WRF
ships it, under WRF's public-domain statement
(licenses/LICENSE-WRF-public-domain.txt; NOAA GSL code, US government work):

| file | sha256 |
|---|---|
| `phys/MYNN-EDMF/module_bl_mynnedmf.F90` | `4aee355aefaaa06c7d0838f97f69b64eda9d0402885c1c35b24454070df25a83` |
| `phys/MYNN-EDMF/WRF/module_bl_mynnedmf_common.F90` | `20c0e0c7c3ee0c08e4de67a6479049ef948022c11c70cff37e1a22b0bfd398ba` |
| `phys/MYNN-EDMF/WRF/module_bl_mynnedmf_driver.F90` | `b3e0646a53ad917521cf858818897c4eed019101b77686e32c8488326cee2cd4` |

`model.cu`, `schema49.json` and `schema50.json` are generated offline by
`tools/mynn_wrf480_port` with the 4.7.1 recipe: `lower.py` (native REAL
constants dumped from the native modules, F90 lowered to fixed-form F77),
f2c, `lower_cuda.py --global-scratch`, `generalize.py` (nz 49/50 affine
dimensions) and `coalesce.py` (private arrays interleaved across columns,
native array arguments bound directly to the packed slabs). Two generator
changes were needed for 8f8e8b59: array sections with nested calls in the
subscript (`minval(rthraten(max(kts,k-1):k+1))`) and the `nscalars` dimension.
`speed.py` then writes the shipped `model.cu` and schemas from the coalesced
column (the 4.7.1 speed transforms matched that generation's text statement
by statement and are not reused). Each step changes no output word:

- `deadstore.py`: assignments to routine-private values nothing enabled
  reads are dropped (121 values), or guarded by the one routine option
  their only readers sit under;
- `zerosink.py`: `dmp_mf`'s eight-plume carrier clears move into the
  updraft branch, the only code that reads them (94% of columns on the
  1974 case never enter it);
- `driverzero.py`: driver clears of arrays `dmp_mf` clears again before any
  use are guarded by `bl_mynn_edmf <= 0`;
- `fold_locals.py`, `recurrence.py`: the plume folds and the tridiagonal
  sweeps keep their running value in a register (same additions, same order);
- `phases.py`: the driver is cut at its native call boundaries into seven
  phase kernels launched in order (cross-phase scalars in a 152-byte
  per-column reserve; per-phase register caps for phases 2 and 3);
- `overlay.py`: the private arrays are laid out as a call stack, so the
  column arena is 48.1 KB instead of 95.8 KB (nz 49);
- inner call-free loops are unrolled by four (no reassociation);
- `initialised.py`: the arguments the driver clears before reading are
  listed in the schema (`initialised_by_column`) and not packed.

Strict runs the single kernel `mynn471_step` (phase -1); fast runs the
phase kernels. `MYNN480_PHASE_PROOF=1` runs the phases under strict numerics.

No Fortran runtime, f2c runtime or native library is used by a forecast.
The `wrf_461` default and the `wrf_471` route do not import this directory.

## Qualification

Native referee: WRF 4.8.0 strict serial pristine build
(`WRF-4.8.0-serial-pristine`, wrf.exe f6f47f86...; module objects
`module_bl_mynnedmf.o` 903cd737..., `module_bl_mynnedmf_common.o` b457ded0...)
in `localhost/wrfref2404:1` (GNU Fortran 13.3.0, glibc 2.39,
`-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`).

Captures: a private I/O-only hook copy of the same tree
(`tools/mynn_wrf480_port/capture.py`), GNU dmpar, 12 ranks, real case
1974-04-03 12Z, 251 x 201 x 50 at 12 km, dt 60 s, one hour, mp_physics 8,
bl_pbl_physics 5, sf_sfclay_physics 5, Noah, RRTMG, 4.8.0 MYNN defaults.
80 columns (i, j = 12 mod 25) at PBL calls 1, 2, 3, 10, 30 and 60: 480 calls.

- Native witness: the standalone referee (pristine serial objects) reproduces
  every in-run output file byte for byte, 480 of 480.
- Raw lowering, GPU strict (sm_89): 0 of 1,276,320 output words differ.
- Shipped `model.cu`, prebuilt strict images: sm_89 on an RTX 4090 and
  sm_120 on an RTX 5090, each 0 of 1,276,320 output words differ. No exclusions.
  Since the speed transforms (images b7161b90...): the single strict kernel and
  the strict phase sequence, each with the column arena filled with 0xFF
  first (`replay.py --poison`, so no private value can pass by being zero),
  0 of 1,276,320 on both cards.
- Fast numerics differ from native in 134,694 of 1,276,320 words, as fast
  numerics do (FMA contraction, CUDA elementary functions); information only.

Option scope: the default 4.8.0 identity only (closure 2.6, cloudpdf 2,
mixlength 1, edmf 1, edmf_mom 1, edmf_tke 0, edmf_dd 0, ess 1, cloudmix 1,
mixqt 0, mixscalars 0, mixaerosols 1 with no aerosol species, mixnumcon 0,
output 0, tke_budget 0). `gpuwm.config` refuses other settings under
`wrf_480` until each has its own native replay.

Upstream defect carried as shipped: 8f8e8b59 still forms the theta tendency
from pre-mixing condensate (fixed upstream in MYNN v6.1.0, e447691). The
separate `v61fix` option owns that fix.
