# UW PBL (bl_pbl_physics = 9) against WRF v4.6.1, column oracle

Grades WOOF's CAMUWPBL port (`woof/core/uwpbl.py`, `woof/core/kernels/uwpbl*.cu*`)
against stock WRF v4.6.1's Fortran, every output word, bitwise.  Builds on the
v4.7.1 oracle in `../uwpbl_wrf471_oracle` (its `run_camuwpbl.F90` driver,
`uwio.F90`, `build.sh` variants and case generators).

## Why the v4.7.1 oracle applies to v4.6.1

`SOURCES-v4.6.1.sha256` hashes the same 20 paths at tag v4.6.1.  The 15 files the
oracle compiles (CAMUWPBL driver, eddy_diff, diffusion_solver, wv_saturation and
their support modules, module_model_constants) are byte-identical between the two
releases; `tests/test_uwpbl_wrf461_sources.py` pins that.  The five driver-side
files differ only outside this scheme (MYNN wrapper rename, KEPS argument names,
RCON/UDM microphysics, IAU tendencies, a soil-type check).  The camuwpbl call in
`module_pbl_driver.F` and camuwpblinit in `module_physics_init.F` are unchanged.

## Files

| file | what it is |
|---|---|
| `SOURCES-v4.6.1.sha256` | pins of the v4.6.1 files; `build.sh` checks them before compiling (`UWPBL_SOURCES_PINS`) |
| `make_verify_cases.py OUT` | 240 fresh columns: nine regime families x 8 columns on 35/44/61-level grids, plus 24 edge columns, four steps each |
| `run_oracle.sh WRF461_SRC OUT` | generates every case set, runs build.sh's four variants (pristine -O0 reference, o2, snan, spy), then the crlibm variant and the cos/acos census |
| `crlibm_shim.c` | crlibm variant: cos and acos resolve to correctly rounded CORE-MATH functions, every other libm call unchanged |
| `crlibm_control.c` | positive control proving the interposition is active and observable |
| `libm_census.c` | census variant: host-libm results, every cos/acos call also compared with the correctly rounded value and counted |
| `compare.py` | grades the product launcher (`--engine gpu`) or the CPU reference (`--engine cpu`) on every step of every fixture; per-field mismatches, max ULP, max abs; negative control |
| `diff_fixtures.py A B` | step-1 word difference between two oracle builds (used for -O2 against -O0) |

## Rerun (box W1, CPU for the oracle, one card through the mutex for the GPU)

    T=/work/pverify/<lane>/src; source /work/pverify/base/env.sh
    PYTHON=$(which python) bash $T/tools/uwpbl_wrf461_verify/run_oracle.sh \
        /work/pverify/wrf-build/src/WRF-4.6.1 /work/pverify/<lane>/oracle461
    # strict build (the 0 ULP standard) and default arithmetic, through the mutex:
    card-lease <lane> --cards 1 --est 900 --wait 3600 bash -c '
      T=/work/pverify/<lane>/src; source /work/pverify/base/env.sh; cd $T
      GPUWM_WRF_EXACT=1 python tools/uwpbl_wrf461_verify/compare.py --engine gpu \
          /work/pverify/<lane>/oracle461/fixtures --json strict.json
      python tools/uwpbl_wrf461_verify/compare.py --engine gpu \
          /work/pverify/<lane>/oracle461/fixtures --json default.json'
    CUDA_VISIBLE_DEVICES= python tools/uwpbl_wrf461_verify/compare.py --engine cpu \
        /work/pverify/<lane>/oracle461/fixtures --workers 32 --json cpu.json

The oracle run takes about two minutes of CPU; each GPU grading pass about one
minute after the kernel compiles.  Case files and fixtures are deterministic:
two runs, and the desktop generator, wrote identical bytes.
