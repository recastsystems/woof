# P3 strict native reference

Strict MP52 and MP53 use the pinned WRF4.7.1 GNU13/glibc2.39 REAL32 build's gamma, log10, rain initialization constants and integer-power order. Default fast numerics keep their existing math. MP50/51 keep their existing strict math.

The source-bound 900-word rain cache is computed by unchanged public-domain WRF4.7.1 p3_init (source SHA256 716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6), GNU13.3, REAL32, -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize, glibc2.39. Each native 300x10 table has ten identical columns; Rust expands the retained first column. Cache SHA256 f3d68ff6c2eb620b35d3f1dc74f9d39d77e00f1b9d60c9074ab18e6c8fb14302. The source and cache must change together when the pinned native recipe changes. The public computed constants have WRF's public-domain status.

The gamma routine is reused from project commit c0ffc53b5d751b2a2fc35c747b57b8554795da82^, under Apache-2.0 as recorded in docs/gf_gamma_known_delta.md. Its FDLIBM helpers retain the Sun permission notice in licenses/LICENSE-FDLIBM-SunPro.txt; Arm helpers retain licenses/LICENSE-Arm-optimized-routines-MIT.txt. No Grell-Freitas code path changes.

GNU13 -O2 lowers x**6 to ((x*x)*x)**2, whereas the historical O0 oracle uses (x*x)**2*(x*x). This difference affected seven real reflectivity words after all prognostic fields matched. The strict P3 branch follows the pinned O2 order. Its log10 and gamma intentionally reproduce the pinned C library's rounded results, including its rounding errors; this is a native replay contract.

The oracle tool's --native-gnu239 flag selects this reference for retained real captures. Historical correctly-rounded analytic fixtures retain their explicitly selected oracle math. Forecasts select the pinned math automatically in strict MP52/53.
