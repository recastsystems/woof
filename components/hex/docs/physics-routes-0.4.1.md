# Physics routes in 0.4.1

Every bare hex forecast uses WOOF physics, including global and limited-area
rows. The default suite is `arwen_mpas_column` and its backend is
`wsm6_column`. This route does not claim bitwise identity to native MPAS
physics. The measured H60 mean step was 0.467 s for WOOF and 0.751 s for
MPAS physics on the same PRO 6000 at the 60 GB cap, about 1.6 times faster.

MPAS suites remain explicit named selections through `woof hex physics list`
and `--physics-suite`. `mpasphys_v842` names the MPAS-A v8.4.2 port;
`mpasphys_rrfs` names the separately pinned RRFS fork. Their suite table
records source and slot-specific scope. Select `--physics-mode exact` with
`--numerics strict` for native bitwise comparisons. Correct mode applies
documented native defect corrections and does not claim bitwise identity.

The qualified v8.4.2 convection-permitting proof covers the public x1.2562
mesh, 24 hours, 124 streams and 650,457,072 written data words with zero
differences. It excludes only the global file_id attribute and requires
identical engine math-header, CUDA, driver and numerics pins in both arms.
This is not a blanket claim for every named suite, fork, mesh or environment.
Other named suites retain their own recorded comparison scope.

Forecasts use the Rust/CUDA port and do not require a Fortran runtime. Native
Fortran references are test inputs, not a forecast dependency. The UCAR
Noah-MP v5 port is excluded from public exports; selecting it names its
licence hold. The UGWP ports ship (US Government code, NOTICE section 8). The permissive WOOF route remains the
default.
