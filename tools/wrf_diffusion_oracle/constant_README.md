# Constant-K direct WRF comparison

These six numerical routines are byte-unmodified source slices of WRF v4.7.1
`module_big_step_utilities_em.F`. The pinned source, constants, slice hashes,
and compiler flags are in `constant-wrf471.json`. The standalone driver calls
the real argument lists with the physical domain, staggering, and three-cell
periodic halos. WRF's own constants module supplies gravity.

WRF `module_em.F:800-874` establishes the diff_opt=1 production calls:
horizontal_diffusion for u/v/w, horizontal_diffusion_3dmp for theta,
vertical_diffusion_u/v for horizontal momentum, vertical_diffusion for w,
and vertical_diffusion_3dmp for theta. They are evaluated only on RK stage 1.
The scalar call uses three times the momentum diffusivity. The direct operator
cases here match the effective per-field K. They do not compare the same
namelist K or the different RK schedules as if those contracts were equal.

WOOF-ArWen explicitly defines this utility as the physical-space constant
Laplacian in `woof/core/diffusion.py:1-26` and `kernels/diffusion.cu:5-15`.
It uses physical base-state heights, periodic coordinate-surface stencils,
and zero-flux top and bottom cell boundaries. It multiplies the primitive
tendency by the field's own dry mass. The WRF routines instead use face-coupled
horizontal fluxes and eta-coordinate vertical fluxes with different boundary
closures. This utility is not a bit-exact WRF operator, and these tests declare
the difference rather than hiding it with tolerances.

There are 56 cases and 106666 stored tendency words. The field values come from
a Rust raw decode of a real WRF initial state at 2024-05-25 18Z. Cases use flat
uniform or stretched heights, a constructed consistent hydrostatic eta metric,
a realistic shear probe, variable mass, zero, near zero, and an exact Cartesian
checkerboard control. Every reference word, every product word, and every
unequal-word mask are retained. Product pins are a drift check on the documented
operator and do not turn its differences into WRF parity.

All six routines are DIFFERENT on the complete case set. The exact Cartesian
meridional checkerboard is a positive control: u, w, and theta are bit-identical
to WRF over every output word. The v control diagnoses the WRF source anomaly
at `module_big_step_utilities_em.F:2854-2855`: its meridional flux lacks the dry
mass factor present in the other branches. Multiplying that specific reference
by the supplied constant dry mass exactly reproduces the product words. The
product does not imitate that source anomaly.

The vertical differences are also directly localized. WRF u/v copy the first
interior flux into the bottom flux and do not update the last mass level
(`:3459-3472`, `:3560-3573`). WRF theta copies the bottom flux as well
(`:3349-3364`). WRF w sets its last interior flux to zero (`:3135-3144`).
WOOF-ArWen applies its documented physical zero-flux cell boundaries. WRF's vertical
v routine also leaves the redundant high-side periodic row untouched, whereas
the product writes a duplicate of row zero, as its storage convention declares.

Rebuild the reference after preparing the same Rust raw dump as the sixth-order
oracle:

```sh
nice -n 10 python tools/wrf_diffusion_oracle/constant_build.py SOURCE BUILD
nice -n 10 python tools/wrf_diffusion_oracle/constant_cases.py dump BUILD tests/data/wrf471_diffusion
```

Under the GPU ownership protocol, retain a complete measurement and word pin:

```sh
PYTHONPATH=. python tools/wrf_diffusion_oracle/constant_compare.py tests/data/wrf471_diffusion tests/data/wrf471_diffusion/constant-product-comparison.json
PYTHONPATH=. python -m pytest -q -s tests/test_constant_diffusion_wrf471_parity.py tests/test_diffusion.py
```

The full output words and differences were identical on RTX 4090 and RTX 5090.
Both cards passed all 18 focused tests. The GPU receipts preserve the assembled
CUDA source hash, all case measurements, and exact first differing words.
