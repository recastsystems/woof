# WRF-Chem v4.7.1 Wesely gas column oracle

Build only in the authorized CPU scratch directory:

```sh
cd ~/work/aq-cams-wesely
nice -n 15 bash harness/build.sh source build
```

The public entry point is `build.sh WRF_SOURCE_ROOT BUILD_DIR`. The source root
needs `chem/module_dep_simple.F`. The script fetches the two public v4.7.1
support modules into the build directory and checks all three compiled WRF
files against `SOURCES.sha256` before compiling. The deposition module stays
byte-unmodified. All object files linked into the reference use `-O0
-fno-fast-math` and the seven WRF defines recorded in fixture PROVENANCE.md.
Every linked object is inspected for forbidden `_ZGV*` references. The copied
Noah positive control must emit a `_ZGV*` reference at `-Ofast` before running
any fixture. The compiler is `/usr/bin/gfortran`.

No `-DNETCDF` is used. Only the MOZART seasonal-PFT path requires it, at
chem/module_dep_simple.F:1668-1684; this oracle takes the non-MOZART path.
The real model constants and error module are compiled. Stubs supply state
indices, package constants copied from Registry/registry.chem with individual
line citations, configuration queries and services. No stub computes physics.
The unused MOZART HL lookup has an empty service table and an aborting string
service. `wrf_message` is supplied by the compiled error module.

`run_wesely.F90:8` has nine gas slots: absent index 1, six potential real
rows at 2:7, and two trailing dummy gases at 8:9. Thus the non-MOZART zeroing
at chem/module_dep_simple.F:243-244 never lands on a real row. NH3 is absent
(index 1) except cases 07 and 08. SO2 is present in all cases. High/low NH3
input columns alternate in the extra cases. The driver directly calls
`dep_init` and `wesely_driver` with GOCART_SIMPLE=300. The normal WRF
`dry_dep_driver.F:259-282` call site excludes 300, so this oracle is a direct
column exercise of the requested parameter branch, not that normal call site.

Each fixture is 24 by 18 columns. Fortran raw arrays retain their logical
shape in `manifest.txt` and column-major bytes; Python reshapes with order F
and transposes into C-order `(nrow,ny,nx)`. `oracle_io.F90` supports REAL(4)
and INTEGER(4), ranks 0 through 3, with little-endian stream writes.

| fixture | map | julday | rows | purpose |
| --- | --- | ---: | --- | --- |
| case_01 | USGS | 90 | four v1 gases | summer boundary |
| case_02 | USGS | 89 | four v1 gases | winter boundary |
| case_03 | USGS | 270 | four v1 gases | summer end |
| case_04 | USGS | 271 | four v1 gases | winter start |
| case_05 | MODIFIED_IGBP_MODIS_NOAH | 180 | four v1 gases | all 20 mapped classes |
| case_06 | MODIFIED_IGBP_MODIS_NOAH | 89 | four v1 gases | winter map |
| case_07 | USGS | 180 | six including extras | NH3/HNO3 and both highnh3 states |
| case_08 | USGS | 89 | six including extras | winter NH3 arms |
| case_09 | USGS | 180 | four v1 gases | state A |
| case_10 | USGS | 180 | four v1 gases | changed shortwave, state B |
| case_11 | USGS | 180 | four v1 gases | A/B/A calls; saved B and final A |

All USGS cases exercise classes 1:24, including water 16 and ice 24.
Inputs span gsw 0/50/800, dry/RH 80/RH 85/wet, qr and raincv rain flags,
stable/neutral/unstable rmol and the near-zero snap, low/high ust including
ust below the aerodynamic-resistance floor, low/high znt, and tc
-10/-5/-3/-1/0/1/2/3/45 degrees C. Case 10 shifts gsw by 100. The A/B/A
reference proves no memory and is compared to independent A and B fixtures.
There are 4752 fixture columns, with two additional calls in case 11.

## Port contract and transcription differences

- `woof/core/chem_drydep_gas.py:84` packs four float32 row numbers and an
  integer arm. No process branch tests a species name or species index.
  Names are used only for error labels, generic state lookup and diagnostics.
- Arm 0 general, 1 ozone, 2 sulfur_dioxide, 3 ammonia. Source arms are
  chem/module_dep_simple.F:965-1038,1044-1057,1081-1150,1154-1287.
  The tables and both land maps come from JSON, with source citations.
- The CUDA kernel fuses the gas subtree into one column thread. Packed table
  offsets and Fortran-to-C array transposes are storage changes only.
  `deppart` is transcribed, but its particle and fog results cannot affect a
  gas row. The p_sulf overwrite and particle vgp are excluded from gas output.
  MOZART, chemical-package aliases and seasons 3/4/5 are excluded because
  they are outside the requested non-MOZART gas path.
- WRF mutates very small rmol to zero. The port uses the snapped value locally
  for the same calculations and leaves the caller's met array unchanged.
- Every transcendental uses the shared glibc float32 header. Integer squares
  use multiplication, as gfortran's depvel object does at -O0.
- `WFloat` stores exactly one float32 word and every operation is a scalar
  round-to-nearest PTX instruction without an FTZ modifier, because CuPy
  appends `-ftz=true` after user options; an inline-asm operation cannot be
  contracted into an FMA, so no loader option is needed. KARMAN = 0.4 comes
  from share/module_model_constants.F:82.
- Each row carries the six dep_init numbers, including `dratio` and `scpr23`
  as the float32 values the Fortran derived at chem/module_dep_simple.F:
  3478-3489 (a rounded dratio is not an invertible encoding of dvj, so the
  row carries scpr23 itself). The parity test asserts every row number equals
  the Fortran dump bit for bit.
- rc's highnh3 test reads the rows on the ammonia and sulfur_dioxide arms.
- The process (`woof/core/chem_drydep_gas.py`) follows
  `woof/core/chem_context.py`: it writes each Wesely row's plane of
  `ctx.ddvel` from the third step on (chem/chem_driver.F:1047) and publishes
  `wesely_aer_res_def`/`wesely_aer_res_zcen` for the aerosol deposition.

## Validation

* CPU: `python -m pytest tests/test_chem_wesely_wrf471_parity.py -m "not gpu"`
  (tables, land maps and every row number equal the Fortran dump; packing and
  refusals).
* GPU: `python -m pytest tests/test_chem_wesely_wrf471_parity.py -m gpu`:
  0 ULP for every row's ddvel and both aerodynamic resistances on all 11 cases,
  measured on an RTX 5090 (sm_120) and an RTX 4090 (sm_89).
* Mutation: changing one constant of rc in the kernel (rluo1's 3000 to 3001)
  fails all 11 GPU cases; changing one JSON table entry (ri(2,1) 60 to 120)
  fails all 11 table cases. Both were reverted.

## Refusals (woof/core/chem_drydep_gas.py)

- Unsupported MMINLU: "WRF-Chem's dep_init leaves luse2usgs unset for <name>,
  so every land class would index garbage".
- sf_surface_physics = 0: "WRF-Chem requires a soil model: iland would be zero
  in deppart and index outside every table".
- A land-use dataset the surface scheme's tables do not name: the Wesely
  classes cannot be mapped.
- A row naming drydep.wesely without the wesely block, an unknown arm, a
  missing or nonfinite number, a nonpositive dratio or scpr23, or two rows on
  the ammonia or sulfur_dioxide arm.
- Wrong dtype, shape or contiguity of a launch field, and ivgtyp outside the
  land map.
