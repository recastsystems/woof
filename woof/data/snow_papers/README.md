# Noah-MP snow and glacier optics written from papers

`&noah_mp opt_alb=3 /` selects this implementation through `woof go`.
**Written from papers, not bitwise to WRF.** Its original source and tables
were never porting inputs. Native WRF runs serve only as comparison runs.
Options 1 and 2 retain their existing default kernels and carrier paths.

The runtime solves three resolved Noah-MP snow layers over soil, water, or
four glacier ice layers. Each column has one 64-thread GPU block; threads
integrate 480 spectral bins (0.2–5 um, 0.01 um spacing). Rust validates,
decodes and hashes the generated table once; CUDA advances all forecast
microphysics and radiative transfer. Python orchestrates launches only.
Transfer coefficients and interface fluxes use double precision. Broadband
integration uses float32, with at most eight bins per thread and a balanced
block reduction. Bins with exactly zero direct and diffuse reference weight
are skipped. Clear and overcast transfer reuse the same solution when no
internally mixed dust is present; internal dust retains both distinct solves.

## Published equations implemented

The newly written Lorenz–Mie series supplies extinction, absorption and
scattering-weighted asymmetry. Snow uses a lognormal distribution with
sigma_g=1.5, 24 midpoint samples over ±4 log-space standard deviations,
and 32 effective radii from 30 to 1500 um. This numerical quadrature and
the fixed atmospheric spectrum are declared choices, not original tables.

`snicar_snowshape_opt` selects 1 sphere, 2 spheroid, 3 hexagonal plate
(default), or 4 Koch snowflake. The shape-factor/aspect-ratio corrections
follow doi:10.1175/JCLI-D-17-0300.1 and doi:10.5194/acp-18-11507-2018;
the hexagonal-crystal base coefficients follow doi:10.1175/2007JAS2289.1.
The six published fit intervals are interpolated as described in the
shape module. Default shape factors are 0.929, 0.788 and 0.712; aspect
ratios are 0.5, 2.5 and 2.5. Extinction and single scattering albedo use
the equivalent-volume/surface-area Mie solution, as in the parameterization.

External black carbon and dust add absorption/extinction and scattering
moments by mass. Internal BC modifies snow coalbedo using the published
2017 fit; internal mineral dust uses the clear/overcast fits in
[doi:10.1029/2019MS001737](https://doi.org/10.1029/2019MS001737).
Clear and overcast fits feed direct and diffuse transfer separately.
`snicar_snowbc_intmix` defaults true; `snicar_snowdust_intmix` defaults false.
The fits describe one inclusion species at a time: simultaneous internal
BC and internal dust is refused rather than inventing an unvalidated
cross term. Both external species may coexist with either internal species
in the GPU layer/state API. This is an explicit limit of the published
single-species fit, and differs from native configurations that enable
both internal switches together.
The recommendation against combining these fits is also explicit in
[doi:10.1029/2023MS003861](https://doi.org/10.1029/2023MS003861), section 2.2.8.

`snicar_bc_ppb` and `snicar_dust_ppm` set initial/new-snow mass fractions
(default zero). Their supported fit domains are 0–10000 ppb and 0–1000 ppm.
The table uses the 0.1 um monodisperse, 1700 kg/m3 BC reference inclusion
with refractive index 1.95+0.79i; it does not apply the separate 2018 BC
size-distribution correction. Mineral dust uses the public 2024 coarse-mode
mean refractive index, lognormal Re=1 um, sigma_g=1.5, rho=2500 kg/m3.
These choices differ from the five-bin Colorado dust/Picard-ice defaults
reported for the newer Noah-MP integration. Layer-specific external/internal
mass carriers permit mixtures and vertical contrasts; no deposition flux,
aerosol chemistry or meltwater scavenging is inferred from unavailable
forcing. The option does not claim those additional aerosol-cycle processes.

The delta-Eddington adding/doubling solver follows
[doi:10.5194/tc-13-2325-2019](https://doi.org/10.5194/tc-13-2325-2019)
and Appendix A1–A20 of
[doi:10.5194/gmd-14-7673-2021](https://doi.org/10.5194/gmd-14-7673-2021),
which documents the method of the 2007 sea-ice technical report
(NCAR/TN-472+STR). It keeps direct beam and diffuse transmission separate,
uses eight Gaussian angles for diffuse coefficients, adds asymmetric
interfaces and layers, and calculates net flux at every interface. The
Gamma*mu=1 removable singularity uses its analytic derivative limit.
Layer absorption is coupled to Noah-MP HRT/TSNOSOI as solar source terms;
the penetrating portion is removed from the existing surface heat input,
so the total column energy is preserved. The existing canopy solver
supplies irradiance incident on the snow.

Glacier ice follows [doi:10.5194/tc-16-1197-2022](https://doi.org/10.5194/tc-16-1197-2022):
air bubbles scatter within an ice host; ice absorbs through 4*pi*k/lambda;
a Fresnel interface uses the absorbing-medium adjusted real index and
10000-angle diffuse quadrature. Four ice thicknesses come from Noah-MP's
soil/ice coordinates. `snicar_ice_density` defaults 900 kg/m3 and
`snicar_bubble_radius_um` defaults 1000 um; supported ranges are 650–916.999
kg/m3 and 30–20000 um. The GPU API accepts independent values per layer.
Impurities in ice act as external scatterers/absorbers. This implements
bubbly ice and refraction, not the paper's optional algal/brown-carbon
constituents or water-film roughness extensions.

## Grain evolution and persistent state

Dry metamorphism is regenerated from
[doi:10.1029/2005JD006834](https://doi.org/10.1029/2005JD006834), equations
5–7, 14–16: 200 radius bins ×40 particle-pore distances, hourly integration
for 14 days, initial SSA=60 m2/kg, sigma_g=2.3 and packing irregularity=5.
Particle-pore positions remain fixed as grains grow. Eight workers generate
448 temperature/gradient/density fits. Kelvin ensemble moments are evaluated
with a converged exponential series; the runtime interpolates these fits.
The fitting norm is log SSA, and the gas transport/saturation formulae are
explicit in the generator. The generated parameters are not the original
aging lookup table. The validation example compares 16 Table 4 cases and
100×20 versus 200×40 resolution; resulting differences are retained in the
lane evidence rather than tuned away using an original table.

Fresh radius rises from 54.526 to 204.526 um between air temperatures −30
and 0 C. Shallow snow without a resolved layer keeps fresh radius. Wet
metamorphism uses dr^3/dt=4.22e-13*f_liquid^3 m3/s. Exact phase-change
refreezing mixes 1000 um grains with pre-existing ice by mass. The coupling
choices follow [doi:10.1175/JHM-D-24-0082.1](https://doi.org/10.1175/JHM-D-24-0082.1),
section 2b2. Layer overlap remapping conserves retained carriers and uses
mass-weighted radii for mergers; splits inherit the parent radius. Radii
are constrained to 30–1500 um. Net snow-mass addition is the fresh-snow
increment; this does not reconstruct unresolved simultaneous accumulation
and ablation inside one Noah-MP time step.

`snicar_statexy` contains 21 f32 planes: three layers of mass, radius,
liquid mass, external BC, internal BC, external dust and internal dust.
Impurity/liquid masses are kg/m2. `TAUSSXY` holds negative column-equivalent
radius for compatibility with the option's earlier carrier; the separate
layer carrier is authoritative. Algorithm/table/state identities are sealed
in restarts. Parent and child must agree on option 3; carrier feedback and
independent ensemble ownership follow the existing resident field paths.

## Generate and inspect

```sh
cargo run --offline --release --manifest-path tools/rustwx/Cargo.toml \
  -p static-fields --example snow-optics --jobs 8 -- \
  tools/snicar_optics/data/ice.yml tools/snicar_optics/data/dust-ri.dat \
  tools/snicar_optics/data/solar.csv gpuwm/data/snow_papers/optics.bin
cargo run --offline --release --manifest-path tools/rustwx/Cargo.toml \
  -p static-fields --example snow-aging-check --jobs 4
```

The asset has 283703 f32 words plus an 8-byte version marker (1134820 bytes).
Its exact digest and all raw-source hashes are in
`tools/snicar_optics/data/source-pin.json`. Rust rejects a changed digest,
invalid dimensions and nonphysical entries before modifying caller output.

Ice indices are the 2008 compilation (doi:10.1029/2007JD009744), redistributed
by the pinned CC0 database. Mineral indices and the derived dust data retain
CC-BY-4.0 attribution. The G173 solar spectrum carries the complete DOE/NLR/
ALLIANCE data notice; direct and diffuse weights are separately integrated
on its reference tilted plane. It is a fixed reference spectrum, not an
atmospheric radiative-transfer calculation at each column. New code is
Apache-2.0. `NOTICE`, `licenses/` and the source pin distinguish these terms.

The lane report records the complete forecast, original native comparison,
observed station snow depths, Rust maps, speed, option-off bytes and lifecycle
checks. Those measurements qualify this paper reconstruction; they do not
establish bitwise or universal physical equivalence to original SNICAR.
