# Provenance: Noah-MP scale-aware ground snow cover fraction (`opt_scf = 2`)

Label: **re-implemented from published descriptions, not bitwise to upstream.**

The Noah-MP clean re-implementation, catalogue item C3 (2026-11-09).

## What was read

Only the paper, from the NOAA Institutional Repository copy (CC BY 4.0):

- Abolafia-Rosenzweig, R., C. He, T.-S. Lin, M. Barlage and K. Rittger, 2025:
  Improved cross-scale snow cover simulations by developing a scale-aware ground
  snow cover fraction parameterization in the Noah-MP land surface model.
  *J. Adv. Model. Earth Syst.*, 17(6), e2024MS004704. doi:10.1029/2024MS004704.
  https://repository.library.noaa.gov/view/noaa/71110 (licence CC BY 4.0).

No file from NCAR/noahmp, the WRF 4.8.0 noahmp submodule or any UCAR-licensed
copy was opened, cloned or read. Our own WRF-lineage port (WRF v4.6.1 /
v4.7.1 Noah-MP, distributed by WRF under its public-domain notice) is the code
changed.

## Routine by routine

| Our routine | What changed | Paper source |
| --- | --- | --- |
| `woof/core/noahmp_scf.py: scale_aware_parameters` | new: SCfactor and m from grid size DX (km), capped at DX = 36 km | Section 2.3, Eq. 4 (SCfactor = 0.0062 sinh(0.0555 DX) + 0.0555) and Eq. 5 (m = 0.9713 + tanh(0.7436 DX)); the 36 km upper constraint and its printed values 0.0779 / 1.9713 in the paragraph after Eq. 5 |
| `woof/core/noahmp_scf.py: ground_snow_cover_fraction` | new: float64 host reference for tests | Section 2.2, Eq. 1-3 (SCF = tanh(h/(SCfactor f_melt)), f_melt = (rho_snow/rho_new)^m, rho_new = 100 kg m-3, rho_snow = SWE/h) with Eq. 4-5 |
| `woof/core/noahmp_energy_slab.py: energy_seg1` | with `opt_scf = 2`, SCFFAC and MFSNO (land-type MPTABLE values) are replaced by the Eq. 4-5 values; the Eq. 1-3 statements (our existing float32 `powf`/`tanhf` transcriptions) are unchanged | Section 2.3: the parameters become functions of grid size for every land cover type, replacing the per-land-type tuning |
| `woof/core/noahmp_column_slab.py: evaluate_sflx_slab` | passes `opt_scf` and the per-column spacing (`scf_dx` if supplied, else the driver's `dx`) | DX is "the model grid size [km]" (after Eq. 5) |
| `woof/core/noahmp_runtime.py` | `opt_scf` in the land options; a per-cell spacing field `noahmp_cell_dx` (metres) is gathered when a caller carries one (hex variable-resolution meshes) | same |
| `woof/config.py`, `physics.py`, `noahmp_surface_device.py`, `checkpoint_identity.py`, `ingest/prepared_cache.py`, `tools/build_registry.py`, `physics_registry_v2.json` | the selector `opt_scf` (1 default, 2 opt-in), its admission, its registry row and its absence from default identities | none (plumbing) |

Implementation choices that are ours, not the paper's:

- The cap is applied as `min(DX, 36 km)` before evaluating Eq. 4-5. Both
  functions increase with DX, so this is the paper's "solutions of Equations 4
  and 5 when DX = 36-km" exactly, not their four-digit rounding.
- SCfactor and m are evaluated in float64 and rounded once to float32.
- Glacier columns are unchanged: our glacier path sets FSNO = 1 whenever there
  is snow, which the paper does not touch.

## Data

None at run time. Validation sources named by the paper: STC-MODSCAG ground SCF
(NASA JPL; access terms not checked, not used here) and MODIS albedo (NASA,
public). The forecast check uses ASOS station observations (public).
