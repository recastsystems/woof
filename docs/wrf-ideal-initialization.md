# WRF ideal initialization inspection

`woof ideal-plan namelist.input --case em_heldsuarez --json` describes the WRF 4.6.1 source recipe,
original namelist options, mass dimensions, exact rational timestep, required
input names and every declared boundary side. `woof ideal-plan --list` lists
the registered global case. This command uses CPU only and does not decode a sounding,
build a model state or advance a forecast.

The global initializer is selected at WRF build time. Pass `--case` with the
original test directory identifier, or run inside that exact directory.
Renamed files need an explicit case. Numeric selectors are never guessed.

`woof/data/wrf_ideal/cases.json` holds the source mapping. Each row records
grid and base state, sounding format, terrain, perturbation, surface,
Coriolis, required files, source evidence and missing runtime binding. Add
source recipes through this table. There are no individual case adapters.

A recognized recipe alone is not an import. The stock dry hydrostatic
`em_heldsuarez` recipe has a measured runtime binding: admitted plans say
`imports=true`, `runs=true`, `status=runnable` and exit 0. A requested
operator without a global binding is named in a refusal and exits 2.
Inspection never starts a forecast. Listing recipes exits 0.
The original option arrays remain in each JSON receipt, including unknown
sections and inactive domain values. This is an initialization inspection,
not an audit of all physics options.

The global case uses separate x and y map factors, latitude-dependent
rotation and curvature, native hydrostatic RK3/acoustic integration,
polar closures and stage-specific batched cuFFT filtering. These are
resident CUDA operators with private per-domain workspaces and graphs.
The common restart owner carries the native state and held thermal rates,
with directional-geometry and algorithm identities checked on restore.

Run `woof import-namelist namelist.input` for the resolved JSON receipt,
then `woof go DIRECTORY` where DIRECTORY holds that stock namelist.
Rust reads an optional native `wrfinput_d01`. Otherwise CUDA builds the
analytic atmosphere, with explicit uniform perturbation samples; native
Fortran does not specify a portable RNG sequence. The unchanged stock
400-frame history grouping is supported. Rust maps render automatically.
The supplied dry single-domain hydrostatic settings are bound; extra global
physics, diffusion or spherical parent/child exchange fail by setting name
until those operators are supplied. Regional option-31 nests retain both
feedback modes through their existing engine paths.

The existing `woof fire-ideal --namelist` door remains available under its
own native initialization contract. It is based on WRF 4.7.1 and is not
evidence that a WRF 4.6.1 stock namelist imports unchanged.

See `docs/public/PHYSICS.md` and the selector-31 oracle README for source pins,
the witnessed native-undefined surface residue, strict real-input module
replays, whole forecast comparisons and the lifecycle/performance evidence.
