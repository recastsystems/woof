# WRF 4.8.0 dycore confirm

Checks that the WOOF-ArWen dycore, pinned to WRF 4.6.1 numerics, is also bitwise to the native
WRF 4.8.0 strict referee. No engine code changes: WRF's dycore numerics did not change
between 4.6.1, 4.7.1 and 4.8.0.

## Source check

`module_small_step_em.F`, `module_advect_em.F` and `module_diffusion_em.F` are
byte-identical in 4.6.1, 4.7.1 and 4.8.0. The other changed `dyn_em` files carry only:

- the GF to GFL rename (`gfscheme` to `gflscheme`) in big_step_utilities, module_em,
  convtrans_prep;
- IAU (new in 4.7.0): `calculate_phy_tend` and `phy_iau_ten` code, entered only when
  `iau > 0` (default 0), plus argument plumbing in first_rk_step_part2 and addtendc;
- `#ifdef CORDEXDIAG` / `CDXWRF` / `WRF_CFBM` blocks the default build does not compile;
- `curr_mins2` passed to first_rk_step_part1 for the FDDA nudging routines (FDDA off here);
- Noah-MP, lightning, Thompson `XICE`/`cloudnc` arguments (owned by their scheme ports);
- `#if (RWORDSIZE == 4)` became `#ifndef DOUBLE_PRECISION` (same branch in single precision).

## Run check

Case: real 151x151x50, 3 km, dt 12 s, 2024-05-25 18Z (wrfinput/wrfbdy from real.exe),
hybrid_opt 2, diff_opt 2 km_opt 4, 5/3 advection, damp_opt 3, w_damping 1. Physics: Kessler
(`module_mp_kessler.F` identical in all three tags), everything else off, so the step is
the dycore plus moist transport. The non-Kessler moisture variables are dropped from the
wrfinput (`ncks -x -v QGRAUP,QICE,QNICE,QNRAIN,QSNOW`), which native Kessler never reads and
the WOOF-ArWen door refuses by name.

1. Native WRF 4.8.0 and 4.7.1 strict serial builds (`~/wrf480-port/build/*`, recipe in
   the port's native-reference folder), 10 steps, history every step,
   `namelist.input` here. Compare with `compare_wrfout.py`.
2. WOOF-ArWen strict (`GPUWM_WRF_EXACT=1`) through the verify-exact door and integrator:
   `replay_dycore.py ENGINE HANDOFF WORK`. HANDOFF holds the two input files and this
   namelist with the producer's `&physics` block (the door checks the wrfinput's own
   MP_PHYSICS/SF_SURFACE_PHYSICS); the script then sets Kessler and physics off key by key.
   Needs the tree's own `static-fields` and `netcdf-writer` bridges built
   (`cd tools/rustwx && cargo build --release --locked --offline -p static-fields -p netcdf-writer`).
3. `compare_woof_native.py NATIVE_WRFOUT WORK/out/wrfout OUT.json 12`.
