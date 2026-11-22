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

## Ice refractive-index and dust optical sets

Re-implemented from published descriptions, not bitwise to upstream. The
three ice sets and three named dust compositions are the ones described in
[doi:10.5194/gmd-14-7673-2021](https://doi.org/10.5194/gmd-14-7673-2021),
sections 3.1 and 3.3.3; every number comes from the original paper or data.

`snicar_ice_ri_opt` (default 2):

| Value | Set | Data |
| --- | --- | --- |
| 1 | Warren 1984 ([doi:10.1364/AO.23.001206](https://doi.org/10.1364/AO.23.001206)) | CC0 database copy, unchanged |
| 2 | Warren and Brandt 2008 | the existing table, unchanged bytes |
| 3 | Picard et al. 2016 ([doi:10.5194/tc-10-2655-2016](https://doi.org/10.5194/tc-10-2655-2016)) on Warren and Brandt 2008 | Supplement section 2 absorption table, CC BY 3.0 |

Set 3 converts each tabulated absorption coefficient with
k = alpha*lambda/(4*pi) at its 51 wavelengths (350-600 nm, 5 nm) and keeps
the Warren and Brandt real part there. Below 350 nm k is held at the 350 nm
value, the constancy rule described for the shortest Picard wavelength;
above 600 nm the Warren and Brandt values stand. Each set is a whole table
(habits, bubbles, Fresnel, centre-band albedos for the shape fits) generated
by the same Rust Mie code; the kernel is unchanged and picks the table.

`snicar_dust_optics_opt` (default 0) chooses the mass optics of every dust
carrier (external, and the host for internal-mixing fits):

| Value | Set | Index | Density |
| --- | --- | --- | --- |
| 0 | Zhang et al. 2024 coarse-mode mean | dataset mean | 2500 kg/m3 |
| 1 | Saharan: Balkanski et al. 2007 Table 2, S1 central hematite (31.5 % illite, 24 % kaolinite, 23 % montmorillonite, 14 % quartz, 6 % calcite, 1.5 % hematite by volume) | Bruggeman mixture of the Zhang 2024 mineral group indices | 2645 kg/m3 |
| 2 | Colorado: Skiles et al. 2017 ([doi:10.1017/jog.2016.126](https://doi.org/10.1017/jog.2016.126)) Table 1, n=1.525 | published table, linear in wavelength, constant outside 0.35-1.5 um | 2600 kg/m3 |
| 3 | Greenland: Polashenski et al. 2015 central scenario (31 % illite, 15.2 % kaolinite, 7.3 % montmorillonite, 8.3 % quartz, 35.4 % calcite, 2.8 % hematite by volume) | Bruggeman mixture of the same mineral indices | 2747 kg/m3 |

The Bruggeman root solves sum_i V_i (e_i - e)/(e_i + 2e) = 0 by Newton
iteration from the volume mean (doi:10.5194/gmd-14-7673-2021 eq. 20). All
sets use the in-table lognormal (Re=1 um, sigma_g=1.5); size bins belong to
the deposition option. Mass absorption at 555 nm: 0.041, 0.053, 0.014 and
0.068 m2/g for sets 0-3, so Colorado is the least and Greenland the most
absorbing, the ordering the description reports. Declared differences from
the description: the mineral indices are the CC-BY-4.0 Zhang 2024 groups,
not the Scanza 2015 compilation; Colorado uses Skiles' published Table 1,
not the later revised inversion; Warren 1984 does not apply the Perovich
and Govoni 1991 UV replacement; the Greenland volume fractions are taken as
reported in the 2021 description. The dust rows live in `dust-rows.bin`
and replace only the in-table dust row at device upload.

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

Fresh radius rises linearly between air temperatures −30 and 0 C, from
54.526 to 204.526 um with `snicar_fresh_grain_opt=1` (default) or from 33.0
to 91.0 um with `snicar_fresh_grain_opt=2`, the MODSCAG-constrained values
of Lin et al. 2025 (public RAL talk, 3 June 2024, "Optimization of snow
grain size"). Option 2 is re-implemented from published descriptions, not
bitwise to upstream; with it, dry aging follows the same regenerated
power-law trajectory below the fit's 54.5 um start instead of lifting new
grains to it. Shallow snow without a resolved layer keeps fresh radius. Wet
metamorphism follows the CLM5 technical note Eq. 2.3.75,
dr/dt=1e18*C1*f_liquid^3/(4*pi*r^2) um/s with C1=4.22e-13 m3/s
(f_liquid = liquid/(liquid+ice)); before 2026-11-09 the 3/(4*pi) volume
factor was missing and wet grains grew 4.19x too fast. Exact phase-change
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
cd tools/rustwx
cargo run --offline --release --locked \
  -p static-fields --example snow-optics --jobs 8 -- \
  ../snicar_optics/data/ice.yml ../snicar_optics/data/dust-ri.dat \
  ../snicar_optics/data/solar.csv ../../woof/data/snow_papers/optics.bin
cargo run --offline --release --locked \
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

## Incident spectra and band count

**Re-implemented from published descriptions, not bitwise to upstream.**
Two `&noah_mp` controls, read only by `opt_alb=3`:

| Control | Values | Default |
| --- | --- | --- |
| `snicar_solarspec_opt` | 0 packaged ASTM G173 (this port's existing path); 1 mid-latitude winter; 2 mid-latitude summer; 3 sub-Arctic winter; 4 sub-Arctic summer; 5 Summit Greenland summer; 6 high mountain summer | 0 |
| `snicar_bandnumber_opt` | 1 five bands; 2 480 bands | 2 |

The option names and the six choices follow the published Noah-MP SNICAR
namelist (Lin, RAL Noah-MP workshop talk, 3 June 2024, namelist slide;
Lin et al. 2025, JHM 26(2), doi:10.1175/JHM-D-24-0082.1). Upstream's own
defaults there are 1 (mid-latitude winter) and 5 bands. This port keeps 0
and 480 as defaults so a default run's bytes do not move; the defaults are
dropped from every configuration and checkpoint identity.

How the six spectra are made (generator `tools/snicar_optics/solar_spectra.py`,
asset `woof/data/snow_papers/solar_spectra.bin`):

1. **Atmospheres.** AFGL-TR-86-0110 Table 1 (Anderson, Chetwynd, Clough,
   Shettle and Kneizys 1986; US government report, public domain; DTIC
   ADA175173 via archive.org). Pressure, temperature, H2O, O3, N2O, CO and
   CH4, 0-75 km. The text layer is OCR; `tools/snicar_optics/afgl/` parses it
   structurally, checks every row by the ideal-gas law against the report's
   own number density, hydrostatic scale height and gas smoothness, and
   lists each hand repair with its reason (`afgl_repairs.json`). The
   rebuilt columns reproduce the report's published column amounts:
   mid-latitude summer 2.93 cm / 334 DU, mid-latitude winter 0.86 cm /
   378 DU, sub-Arctic summer 2.08 cm / 347 DU, sub-Arctic winter 0.42 cm /
   375 DU. CO2 415 ppmv, O2 0.2095, N2 0.7808 (present-day values, a
   declared choice).
2. **Truncated columns.** Summit summer is sub-Arctic summer cut at
   796 hPa and high mountain summer is mid-latitude summer cut at 556 hPa,
   as Flanner et al. 2021 (GMD 14, 7673, doi:10.5194/gmd-14-7673-2021,
   CC BY 4.0) section 2.4 describe; levels are interpolated in log pressure.
3. **Clear and cloudy skies** (same section). The direct-beam spectrum is
   clear sky; the diffuse spectrum is cloudy sky: liquid cloud of optical
   depth 10 at 500 nm and 10 um effective radius in the layer holding
   800 hPa (bottom layer when the surface is above 800 hPa), plus aerosol
   optical depth 0.05 at 0.55 um. The aerosol is sulfate-like by our own
   declared choice (Angstrom exponent 1.5, single-scattering albedo 0.99,
   asymmetry 0.7, lowest ~2 km). The surface is snow of 100 um effective
   radius, computed with this port's own Mie table and adding-doubling
   solver. Flanner et al. used another solver (SWNB2); ours is our BSD
   RRTMGP.
4. **Band fluxes.** RRTMGP shortwave (14 bands, 224 g-points) gives the
   surface direct and diffuse flux in each band for 12 values of cos(SZA)
   (0.05-1.0).
5. **In-band shape.** Inside each RRTMGP band, the 0.01 um shape is the
   G173 extraterrestrial spectrum times G173's own spectral transmittance
   raised to a band exponent, fitted so the band integral equals RRTMGP's
   (a Beer-law rescaling of the optical path; the generator asserts each
   band integral matches). Diffuse light takes the direct exponent at
   cos(SZA)=0.6 for its in-band shape and is scaled to RRTMGP's cloudy band
   total. This step is our own construction, declared here.

At run time the kernel interpolates both beams linearly in cos(SZA)
between the nodes. With the option at 0 the kernel reads the packaged G173
weights exactly as before.

**Five bands.** `snicar_bandnumber_opt=1` uses the five SNICAR bands
0.3(0.2)-0.7, 0.7-1.0, 1.0-1.2, 1.2-1.5 and 1.5-5.0 um (Flanner et al. 2007,
JGR, doi:10.1029/2006JD008003, section 2). Band optics are flux-weighted
means of the 480-bin optics: extinction, scattering and scattering-weighted
asymmetry, the Fresnel terms, absorption as 4*pi*k/lambda, and the
internal-mixing factors. Direct-beam optics are weighted by the chosen
direct spectrum and diffuse-beam optics by the diffuse one, both at
cos(SZA)=0.5. These are built on the GPU once per run; the solver then
runs twice over five bins (direct, diffuse) and merges on the device.

**Hand-off check.** The direct/diffuse and VIS/NIR hand-off to the
radiation was read end to end (the fault class upstream fixed in its
v5.1.1 WRF coupling, per the public HRLDAS release notes):
- `out[0..3]` is direct VIS, direct NIR, diffuse VIS, diffuse NIR, with
  VIS the bins below 0.7 um;
- `noahmp_radiation_options.cuh` stores them as ALBSND(1:2) and
  ALBSNI(1:2);
- the layer heating multiplies direct absorption by SOLAD*FTDD and diffuse
  absorption by SOLAD*FTID+SOLAI*FTII;
- the packet carries SOLAD(VIS,NIR) then SOLAI(VIS,NIR) in slots 52-55
  (`RADIATION_IN` solad, solai).

No swap was found. `test_incident_spectrum_weights_match_hand_computation`
pins the beam-to-spectrum mapping for all six spectra.

## Solver and band choices (clean-snicar-layers)

Re-implemented from published descriptions, not bitwise to upstream.
`snicar_solver_opt=2` selects the Toon et al. 1989 multi-layer two-stream
(delta-Eddington in the visible, delta-hemispheric-mean in the near-IR, as
Dang et al. 2019 describe for SNICAR); 1, the default, keeps adding-doubling.
`snicar_bandnumber_opt=5` selects the five-band mode (0.3-0.7, 0.7-1, 1-1.2,
1.2-1.5, 1.5-5 um) with Chandrasekhar-mean band optics (Dang et al. 2019 eq 15);
480, the default, keeps the 480 bins. Both feed the same layer-resolved
absorption into the snow/soil temperature solve. Glacier-ice columns keep the
480-bin adding-doubling solve, the only one carrying the Fresnel ice
interface. Defaults launch the unchanged kernel image. Equation-level
provenance: `docs/PROVENANCE-clean-snicar-layers.md`.
