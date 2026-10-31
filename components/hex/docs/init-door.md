# The init front door: `woof hex init`

Build MPAS initial conditions without the native Fortran `init_atmosphere_model`.
The numerics are the Rust engine `rw_mpas_init` (woof `tools/rustwx/crates/rw-mpas`,
case-7 port); this door finds the engine, refuses bad inputs by name before an
engine run, and writes a provenance receipt beside the init file.

## Invocation (native-free, the normal mode)

Since the native-free bootstrap merged (136e31c), a normal invocation needs no
native MPAS file at all: the vertical contract is constructed from
`--grid + --static + --vertical-spec` (see
`native-free-init-admission.md`), written to a durable artifact, and handed to
the engine through its existing capsule ABI.

Derived mesh geometry runs in `rw_mpas_geometry`, including the per-cell RBF
solve. Its float32 and float64 arithmetic keeps the scalar reference order;
independent cells run in parallel. No released woof carries a version that
speaks `hex-vertical-v1`, so the prebuilt wheel carries one in `woof/hex/libexec`,
pinned by SHA-256 ([engine bridges](engine-bridges.md)). The binary follows the
same discovery ladder as the init engine, and `WOOF_HEX_RW_MPAS_GEOMETRY` or
`RW_MPAS_GEOMETRY` still wins over the bundled copy. An install built from the
sdist without the bundled bridges can build one from a woof source at the pinned
commit or newer with `cargo build --release --locked --offline -p rw-mpas --bin
rw_mpas_geometry`. A missing or stale binary refuses before geometry is computed.

```sh
export RW_MPAS_INIT=/path/to/rw_mpas_init        # or pass --engine

python -m woof.hex.cli init \
  --met    WORK/MET:2025-03-14_12 \              # WPS intermediate file, or a dir holding exactly one
  --static assets/MESH.static.nc \
  --grid   assets/MESH.grid.nc \                 # REQUIRED in native-free mode
  --vertical-spec verification/vertical-specs/tc55-v1.json \
  --out    init.nc \
  --start-time 2025-03-14_12:00:00 \
  --nfglevels 38 --nfgsoillevels 4 \
  --extrap-airtemp lapse-rate --use-spechumd no \
  --theta-adv-order 3 --coef-3rd-order 0.25 \
  --virtual-factor reproduce-fortran --deep-soil-moisture reproduce-fortran \
  --landuse-table MODIFIED_IGBP_MODIS_NOAH \
  --frac-seaice yes --tsk-seaice-threshold 100.0 \
  --oned-underflow preserve
```

The constructed artifact (default `<out>.vertical.nc`, own receipt beside it)
is a complete init-class capsule: the static's mesh/statics carriers, the
constructed vertical contract (float32, the native ABI), the met-state
landing sites the engine writes its values into, and the derived mesh
geometry (`edgeNormalVectors`, `localVerticalUnitVectors`,
`cellTangentPlane`, `coeffs_reconstruct`) that native MPAS fills during
initialization and static files carry as zeros.  `--theta-adv-order` and
`--coef-3rd-order` must agree with the vertical spec's declaration.

## Invocation (compatibility mode, native capsule)

A native init-class file is opened only when BOTH `--capsule` and
`--reference` are passed; mixing them with `--vertical-spec` refuses by name.

```sh
python -m woof.hex.cli init \
  --met    WORK/MET:2025-03-14_12 \
  --static assets/MESH.static.nc \
  --grid   assets/MESH.grid.nc \                 # optional here; cross-checked against --static
  --capsule   NATIVE_INIT.nc \                   # vertical metrics + smoothed terrain source
  --reference NATIVE_INIT.nc \                   # zgrid checked for exact array equality against this
  --out    init.nc \
  --start-time 2025-03-14_12:00:00 \
  --nfglevels 38 --nfgsoillevels 4 \
  --extrap-airtemp lapse-rate --use-spechumd no \
  --theta-adv-order 3 --coef-3rd-order 0.25 \
  --virtual-factor reproduce-fortran --deep-soil-moisture reproduce-fortran \
  --landuse-table MODIFIED_IGBP_MODIS_NOAH \
  --frac-seaice yes --tsk-seaice-threshold 100.0 \
  --oned-underflow preserve
```

On success: `init.nc`, plus `init.nc.provenance.json` (sha256 of every input,
the engine binary, the argv, the engine's own receipt, and the output), and a
JSON summary on stdout.  Exit 0.

Every physics switch is **required and has no default**, mirroring the engine:
each one changes the numbers in a file that opens cleanly and reads plausibly
either way.  The native namelist key is printed in each refusal so a captured
`namelist.init_atmosphere` transcribes without guessing (`--nfglevels` =
`config_nfglevels`, `--extrap-airtemp` = `config_extrap_airtemp`,
`--landuse-table` = `config_landuse_data`, …).  `--virtual-factor`,
`--deep-soil-moisture` and `--oned-underflow` are properties of the reference
*build*, not namelist keys; `reproduce-fortran` / `reproduce-fortran` /
`preserve` is the defined-behaviour set, and `--oned-underflow
reproduce-ifx-ftz` reproduces an `-O3` ifx reference exactly where the two
modes diverge.

## ECMWF IFS open data

IFS already has the engine registry row `ecmwf-open-data` (aliases `ecmwf`
and `ifs`). It enters this door through exactly the same WPS intermediate
reader as GFS. No source name selects initialization code.

The intermediate writer takes `--map-source ecmwf` and the packaged table
`woof/hex/data/vtables/Vtable.ECMWF-OD.rw`. For the measured 2026-10-03 12Z publication,
declare `--nfglevels 15 --nfgsoillevels 4 --use-spechumd no`: fourteen
pressure levels plus the surface, and four soil layers. Earlier publications
can have a different ladder; use the inventory of the actual input.

The table now selects ECMWF snow water equivalent (`sd`, centre 98, local
table 1, GRIB2 0-1-254). The writer's existing `SNOW_EC` rule converts metres
of water to kg m-2. A sea-ice area-fraction row selects 10-2-0 when present;
it does not select sea-ice thickness 10-2-1. The public indexes carry
thickness but no area fraction at any lead (0.25 and 0.1 degree, checked
2026-10-28), so the IFS rows declare a `surface_input`: the same cycle's GFS
analysis ICEC and LAND records (about 130 kB) are written to a second WPS
intermediate and passed to the init as `--sfc-met` (MPAS `config_input_sst`).
The init interpolates that SEAICE over water with the GFS file's OWN land mask,
and MPAS's own rule then marks water whose skin is below 271 K as ice.

`SOILGEO` is a scratch input and `SOILHGT` is a declared derived output.
The existing writer computes `SOILHGT = SOILGEO / 9.81`; the output row is
required to deliver it. Omitting that row silently leaves the init's source
terrain at zero and changes its terrain correction of skin and soil temperature.

An additional same-time, fraction-only donor is possible through the same
writer only if its coordinate layout matches the primary land mask exactly.
The init engine shares that mask by array index for soil and ice, so a donor
with a different longitude origin is unsafe even when it has the same grid
size. That is why the sea-ice companion travels as the separate surface
input above, never as an extra input to the primary writer.

`verification/ifs-open-data/source-contract.json` records the level, soil and
gap contract. Hydrometeors start at zero on this cold-start path for both
sources. Soil uses the MPAS engine's existing depth interpolation onto Noah
midpoints. The WRF mapped composition's conservative layer remap is a
different path and is not claimed for hex.

The measured +3-hour IFS publication omits surface geopotential. Its source
contract supplies one terrain-only record from the same cycle's step-zero file
as a cycle-invariant supplement. Pass the forecast GRIB and that single-record
donor to the existing multi-file intermediate writer, with the forecast's valid
time in `--date`. The donor must share the primary land mask's coordinate layout.
Do not supply a full step-zero file: later files overwrite matching fields, so
that would replace the forecast atmosphere too. The writer uses the caller's
date for its intermediate headers; the caller owns this invariant-field contract.

## Capsule contract

`rw_mpas_init` does not build the vertical grid.  It reads `zgrid, zz, fzm,
fzp, dzu, rdzw, zb, zb3` and the smoothed terrain from an init-class capsule
and checks the capsule's `zgrid` for exact array equality against `--reference` before
trusting a single level.  In native-free mode the constructed artifact IS the
capsule and is passed as both inputs, which keeps that identity check; in
compatibility mode the capsule is a **native-minted init-class file**.  The
produced init is drop-in only for a consumer pinning the same grid/static
bytes the capsule was built from; the forecast driver refuses otherwise (it
reports changed reconstruction-coefficient bytes: that message means
provenance mismatch, not corruption).

## Refusals (exit 2, each names the breakage and the remedy)

- a physics switch not given (no default; names the namelist key)
- `--met/--static/--capsule/--reference/--out` not given
- no engine configured / engine path missing / not executable
- output directory does not exist (refused before the engine runs)
- met source missing; directory with zero or several WPS intermediates
- file not readable as a WPS intermediate (truncated / non-ungrib bytes)
- `LANDSEA` absent from the intermediate file
- `SOILHGT` absent: source terrain would be assumed zero and skin/soil terrain corrections would be wrong
- no level tagged `200100.0` (the surface tag)
- more distinct first-guess levels than `--nfglevels` declares
- no soil layers (`ST*/SM*/SOILT*/SOILM*`) in the intermediate file
- `--start-time` not among the met file's valid times
- static file missing engine-read dimensions/variables
- `--grid` vs `--static` nCells mismatch (two different meshes)
- capsule missing the vertical-contract variables, or built for another mesh
- reference without `zgrid`, or `zgrid` shape differing from the capsule's

Engine-side refusals (rc 1) pass through verbatim; the door adds one line
naming that the engine refused and that nothing was written.

## Proof of record

The recorded initial-array comparison and the acid test are in the provenance receipts of the
first door run: the door-built init for the ERA5 2025-03-14 12Z factory case
was compared field-by-field against the surviving native golden under
`init-tolerances-r3.json`, and the same file started the CUDA v8.4.1 port for a
5-composite-step forecast smoke, rc 0.
