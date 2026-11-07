# MYNN generation gsd_41

`bl_mynn_version = "gsd_41"` selects the GSD MYNN v4.1 boundary layer of the
NOAA-EMC WRF 3.9 branch (NOAA-EMC/HRRR tag v4.1.21,
`sorc/hrrr_wrfarw.fd/WRFV3.9/phys/module_bl_mynn.F`, 6168 lines).  The
default, `wrf_461`, is WRF v4.6.1 `module_bl_mynn.F`.

## How it is built

The rows live in `woof/core/kernels/mynn_pbl.cu` behind
`#if defined(MYNN_GSD41)` and `#if !defined(MYNN_GSD41)`.  `gsd_41` compiles
the same file through the integer-define loader
(`woof.core.mynn_pbl_gpu.mynn_pbl_kernel`); the default build never sees a
gsd_41 line and compiles to the same binary as before the selector existed.
`tests/test_mynn_dmp_sibling.py` evaluates the conditionals as the default
build does before it compares the file with its DMP sibling.  The radiation
merge is `rla_mynn_gsd41` in `woof/core/kernels/rrtmg_legacy_adapter.cu`.

## Rows ported (audit rows of `HRRR-FORK-DIFF-AUDIT-2026-10-03/mynn-pbl-edmf.md`)

| Row | What | Check |
|---|---|---|
| P3 | A downward surface vapour flux is applied to the lowest layer (v4.1.21 :4408, :3114); v4.6.1 :4512-4516 deletes it. | Column test: the response relative to zero surface flux is symmetric for upward and downward forcing. P21 converts with updated vapour and the actual original mixing ratios. |
| P1 | Mixing length option 2, v4.1.21 :940-1085. | 12 columns against the branch's own Fortran: qkw exact, el within 2 ULP (libm). |
| P2, P12, P14 | Stratus subgrid cloud, its buoyancy functions and the 273.16..253 K phase blend, v4.1.21 :2293-2737, :6064-6162. In-cloud QC_BL, no QI_BL. | 8 columns, 240 levels against the branch's own Fortran: bit for bit. |
| P7 | Shallow-cumulus cloud for radiation, v4.1.21 :5767-5870. | Included in the full fork DMP oracle, with in-cloud water and the stratus hand-off. |
| P9 | Decay memory of the subgrid fraction, v4.1.21 :4690-4713. | Unit test of the hold, the clear and the water floor. |
| P8, A1 | Radiation merge, `module_radiation_driver.F` :1256-1303 of the branch. | Unit test of the phase split, the resolved-cloud gate and the fraction rule. |
| P15 | Surface TKE source: the surface layer's 1/L as handed in (not recomputed, not written back), z/L unclipped, Kansas-type forms, v4.1.21 :4412-4419. | Unit test against the formulas (at z/L = 10, pmz 41 where v4.6.1 goes negative). |
| P16 | PBL height reads theta-v of the liquid-water theta, the carried subgrid cloud standing in where no resolved cloud is present (v4.1.21 :4084-4096, :4236-4264); KPBL blends the two heights' rounded levels (:4948, :4980, :5006). The other consumers keep theta-v. | Unit tests of the input and the blend. |
| P17 | No sh floor, clip or sm cap; cloud floor 0.03 cldavg (v4.1.21 :1599-1635, :1820-1821). | mym_turbulence at level 2.5 with mixing length 2 against the branch's Fortran (5 columns, 70 interfaces): el, dfm, dfh, dfq, pdk, pdt, pdq, pdc, sh within 3 ULP. |
| P21 | Output water tendencies use the updated vapour denominator and original mixing ratios, v4.1.21 :3404, :3417, :3449. | All 144 source outputs are checked as FP32 bits under the explicit source cloud form. |
| P13, P22 | Explicit bl_mynn_cloud_tendency_form=gsd_41 takes the source pre-mixing condensate heat and tendency-only negative-condensate clip together. The wrf_461 default creates no water at the clip and uses mixed condensate; it does not make the P11 transport mass-conserving (see "Water budget" below). | Four liquid and ice columns, including negative forcing, match the unmodified source. This defect form is never selected by a recipe or importer default. |
| P20 | TKE has a zero-gradient top row, floor 1e-4 and no upper cap; initialization production uses qkemin 1e-12, v4.1.21 :203, :546, :2083-2094. | Eight columns and 96 TKE values match the unmodified predictor as FP32 bits, including each binding edge case. |
| P19 | Dissipative heating 0.5 q**3/(b1 l cp), held to 0..2e-5 K/s, no pressure taper, v4.1.21 :4600-4604. | Unit test against the formula, cap binding. |
| P14, P24 | Plume condensation uses the same 273.16..253 K saturation blend and the 2e-5 iteration stop, v4.1.21 :5933-5981. | 24 cold and warm plume states against the unmodified source's condensation_edmf: QC and THV bit for bit. |
| P4, P5, P6 | Ten 100 m plume classes, surface excess, resolved-motion taper, trigger, entrainment, overshoot and height damping, v4.1.21 :5107-5562. | 17 columns against unmodified DMP_mf, including every plume stopping at its first interface. |
| P10 | Scale awareness uses dx, not 2.5 dx, v4.1.21 :6011-6049. | Source check and coupled forecast cuts. |
| P23 | Closure level 2.5 (the fork's levflag 2): only TKE is predicted; tsq, qsq and cov are diagnosed from their production with the start-of-call qkw, v4.1.21 :2232-2253 (v4.6.1's closure 2.6 predicts qsq). mym_turbulence already ran level 2.5 under gsd_41. | Eight 12-level columns with nonzero variance production against the unmodified predictor: qke, tsq, qsq and cov as FP32 bits (`variance-gsd41.csv`). |
| P11 | Density-free mass flux, local transport and surface forcing; no EDMF K floors, v4.1.21 :2044-2114, :2835-3114, :5633-5729. | Full plume oracle and eight clear transport columns. Ten-class workspace sizes are priced and allocated only for gsd_41. |

## Exactness (development branch)

`tools/mynn_pbl_gsd41_oracle/run_driver_families_gsd41.F90` runs the fork's
own `mynn_bl_driver` with HRRR's namelist identity over six column families
for twelve steps (`driver-families-gsd41{,-asis}.csv.gz`, gfortran 13.3 -O0,
glibc 2.39).  With `bl_mynn_cloud_tendency_form = "gsd_41"` the CUDA driver
is that driver's word on every output of every step, free-running from the
cold start, for the file as written and for the squared `:995`
(`tests/test_mynn_gsd41_driver_exact_gpu.py`).  Four corrections got it
there: the fork's `mym_level2` and level-2.5 algebra divide by `a2den`
where v4.6.1 multiplies by `a2fac`; its warm-step `thetav` is
`th*(1+0.608*sqv)`; its `cpm` reads the mixing ratio handed in; and the
level-2 pair kernel now runs the selected generation.  The default
`wrf_461` tendency form changes RTHBLTEN and RQVBLTEN values for the
P13/P22 fix and can change the sign of a zero RQCBLTEN. Its nonzero
RQCBLTEN values remain identical in this family oracle. The raw-word
gate checks this specific zero-sign scope rather than treating all
signed zeros as equal. The fork leaf CSVs `mixlength2-gsd41*.csv`,
`turbulence2-gsd41-sq.csv` and `condensation-gsd41.csv` were rebuilt on
glibc 2.39, the library of every WRF v4.6.1 oracle (2.43's atanf and tanhf
are correctly rounded); the leaf budgets are now 0.

## Selection

The HRRR and RAP rows of `woof/data/physics_sources/request-defaults.v1.toml`
declare `bl_mynn_version = "gsd_41"` under `[request.load_generations]`, so a
configuration naming those sources and omitting the key runs it, on trees
every domain of which admits it (`physics_source_defaults.gsd41_admitted`).

## Review coverage

Both admitted mixing lengths run the fork's own arms. Option 1 uses the
fork's harmonic lengths, minimum BouLac displacement and interface-theta
buoyancy coefficient. It no longer substitutes WRF v4.6.1 option 1.
The regression fixture `driver-families-gsd41-ml1.csv.gz` records twelve
steps from the unmodified fork. The optional third argument to the Fortran
family harness names a `review` namelist for mixing length, cycling,
carried QKE and cloud, and forcing scales. Omission keeps the original
fixture inputs unchanged.

Cycling compares the lowest-level maximum with the fork's float32 `0.0002`,
including equality. The comparison is decided once before column pieces.
WRF v4.6.1 zeroes QKE, QC_BL and CLDFRA_BL after its cycled keep decision; WOOF carries them by default with `cycling = true`.
Both generations reset sh/sm/el/tsq/qsq/cov on a cycled start. The stock
carry referee is built by `tools/mynn_pbl_wrf461_oracle/build_cycled.sh`;
the family driver's fourth argument selects empty, below, equal, above,
high or mixed carry. Omission preserves its cold-start inputs.
The exact-driver checkpoint identity is
`mynn-edmf-pbl-wrf-v4.6.1-v3-exact-driver`; the former v2 identity cannot
resume into these changed tendencies.

## The one defect-shaped line

v4.1.21 :995 converts the interface q to TKE as `0.5*qkw` where option 1 of
the same file and every later generation write `0.5*qkw**2`.  The default
takes the squared form; `bl_mynn_gsd41_unsquared_qtke = true` takes the line
as written.  Both are checked against the Fortran (the squared form against
the one line patched by `tools/mynn_pbl_gsd41_oracle/build.sh`).

## Oracle

`tools/mynn_pbl_gsd41_oracle/build.sh FORK_module_bl_mynn.F BUILD_DIR` builds
the unmodified branch source with GNU Fortran at -O0 and writes the CSVs in
`woof/data/mynn/oracle/` (`mixlength2-gsd41*.csv`, `condensation-gsd41.csv`,
`plume-condensation-gsd41.csv`;
hashes in `mixlength2-gsd41-provenance.txt`).

The mass-flux continuation adds `dmp-mf-gsd41.csv` and
`transport-gsd41.csv`, generated from the unmodified fork. The latter
compares the specific-humidity solve before the P21
conversion. The public HRRR products do not expose internal plume
occurrence; a diagnostic from published profiles is an estimate and is
labelled separately in the lane report.

## Configuration doors

The named hrrr_wrf.nl importer and the legacy MYNN budget key select gsd_41 and,
when radiation is not explicitly overridden, its required legacy RRTMG
merge. The native-spacing HRRR recipe selects the named composition
`thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1`. The full-grid recipe
`configs/recipes/hrrr_v4_gsd41.toml` sets both MYNN keys explicitly.
Generic defaults and all existing composition switch sets are unchanged.
The full raw namelist still needs support for its unrelated sections and
time-control keys; this lane does not silently discard them.

The TKE continuation adds `predict-gsd41.csv`, built from the unmodified
fork. The cloud transport fixture checks the paired P13/P22 defect form and
P21 output conversion. The source-defect form lacks the heat matching
mixed condensate and creates water when condensate is clipped; it requires
explicit bl_mynn_cloud_tendency_form=gsd_41, with wrf_461 the default.

The output conversion receives the actual original qv, qc and qi arrays
from the runtime. Direct gsd_41 leaf callers must supply those mixing
ratios as well as the specific arrays. At exactly unit updated specific
vapour the source denominator is zero; the defined fallback returns
zero water tendencies and keeps the original mixing ratios.
