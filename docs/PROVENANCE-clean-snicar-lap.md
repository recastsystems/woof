# Provenance: clean-snicar-lap (light-absorbing particles in snow)

Option `snicar_lap_opt` (Noah-MP `opt_alb=3` only), labelled **re-implemented from published
descriptions, not bitwise to upstream**. Default 0 = off; with it off no array, kernel module,
checkpoint identity or prepared-cache key changes.

Sources read: papers, public documentation, public data descriptions, and this tree. No file
from NCAR/noahmp, the WRF 4.8.0 noahmp submodule, CLM/CESM source or any UCAR-licensed copy was
opened, cloned, read or quoted.

## What upstream added (from the published description only)

Lin et al. 2025 (JHM 26(2), doi:10.1175/JHM-D-24-0082.1) and the public talk "Improving snow albedo
modeling in Noah-MP v5 via coupling with a snow radiative transfer model" (Lin, 3 June 2024) describe
LAP deposition fluxes for hydrophilic and hydrophobic BC, hydrophilic and hydrophobic OC and dust in
5 size bins (0.1-1, 1-2.5, 2.5-5, 5-10, 10-100 um), tracked layer by layer with mass conserved, from
deposition fluxes, with a table option for the fluxes. This option builds that capability from the
physics papers below.

## Routine by routine

| Routine | File | Source (paper, section, equation) |
| --- | --- | --- |
| Species list and dust bins | `woof/data/snow_papers/lap_species.json` | Lin et al. 2025 sect. 2b (species and bin edges as published; bins taken as particle diameter) |
| Meltwater flushing `lap_flush` | `woof/core/kernels/snow_lap.cu` | Flanner et al. 2007 JGR 112 D11202 sect. 2.4: particles leave a layer with percolating meltwater at scav x layer mixing ratio x water flux. Scavenging factors (BC/OC hydrophilic 0.20, hydrophobic 0.03, dust 0.02, 0.02, 0.01, 0.01) as published there and in the public CLM5 Technical Description (Lawrence et al. 2018) snow-aerosol section. The 10-100 um bin takes the coarsest published factor, 0.01 (declared assumption). Noah-MP v4 exposes only the column-bottom outflow QSNBOT to this stage, so every live layer is taken to pass that flux (declared approximation); particles leaving the bottom layer go to the soil. |
| Layer remap `lap_remap` | `woof/core/kernels/snow_lap.cu` | Own construction, mass-coordinate overlap: new material A = max(0, T-(O-Q)) on top, the old column compressed in order into [A,T). Conserves each species exactly (remainder assigned to the last overlapped layer). Needed because Noah-MP combines/divides layers each step. |
| Deposition into the top live layer | `snow_lap_albedo` thread 0 | Flanner et al. 2007 sect. 2.4 (dry and wet deposition enter the surface layer). Deposition onto bare ground is not counted. |
| Spin-up initial load | `LapDeposition` in `woof/core/snow_papers.py` | Own construction: deposition in the series before the forecast start, summed and mixed through the pack by mass at the first step (the pack's history is not known; declared approximation). |
| BC optics | packaged `optics.bin` BC row | Bond & Bergstrom 2006 (unchanged, lic-snicar). Hydrophilic BC is the internally mixed BC when `snicar_snowbc_intmix` (He et al. 2017 fit, unchanged); hydrophobic BC is always external. |
| OC optics | `woof/data/snow_papers/lap_optics.bin` row 0 | Brown et al. 2018 ACP 18:17745 eqs 1-3, their statement of the Saleh et al. 2014 Nat. Geosci. fit: n = 1.7, k = k550 (550/lambda)^w, k550 = 0.016 log10(R) + 0.04, w = 0.21/(R + 0.7) **as printed in Brown 2018** (open check: Saleh 2014's own constant not seen; the paywalled original was not read), R = 0.08 (Brown 2018's global mean biomass-burning BC-to-OA). Lognormal, effective radius 0.1 um (assumption, as the packaged BC reference radius), sigma_g 1.5, density 1000 kg m-3 (Brown 2018 sect. 2.1, MAM4 POM density). 550 nm MAC 0.71 m2/g. |
| Dust optics | `lap_optics.bin` rows 1-5 | Zhang et al. 2024 mineral refractive index (CC-BY-4.0, doi:10.5281/zenodo.14171144, already packaged by lic-snicar), density 2500 kg m-3, mass log-uniform in diameter inside each bin, 24 sub-bins, self-written Lorenz-Mie (Bohren & Huffman 1983 ch. 4). 550 nm MAC 0.033, 0.045, 0.034, 0.027, 0.014 m2/g by bin. |
| Dust internal mixing | `snow_lap_albedo` | Unchanged He et al. 2017 fit on the total dust concentration; external dust rows are then not added (as the existing option). |
| Snow solver | `snow_lap_albedo` | The lic-snicar adding-doubling solver, reused verbatim (device functions of `snow_papers.cu`; the per-layer impurity sum is the only change). With zero particles its output is bitwise the existing `snow_papers_albedo` (test). |
| Deposition producer | `tools/rustwx/crates/rw-netcdf/src/snow_lap_deposition.rs`, door `woof/snow_lap_deposition.py` | GOCART deposition diagnostics of GEOS-FP and MERRA-2 (`*_2d_adg_Nx`). Bin 001 hydrophobic, 002 hydrophilic (GOCART convention, Chin et al. 2002; Colarco et al. 2010 JGR 115 D14207). GOCART dust bins are RADIUS 0.1-1, 1-1.8, 1.8-3, 3-6, 6-10 um (Colarco 2010 table 1), redistributed onto the diameter bins by log-uniform mass overlap. Dry = DP (+ SD sedimentation for dust); wet = WT + SV, GEOS-FP's single SV total shared among bins in proportion to WT. Bilinear on the source grid with longitude wrap; one file per worker on all cores. MERRA-2 row variable names are from the MERRA-2 file specification (GMAO Office Note 9) and are untested (no Earthdata login on our machines). |

## Data licences

- GEOS-FP (NASA GMAO), public, no use restrictions, no login: used for the forecast check here.
- MERRA-2 (NASA GMAO), no use restrictions; GES DISC needs a free Earthdata login (a credential we do not hold). The door refuses that row by name and accepts already-downloaded files with `--files`.
- Zhang et al. 2024 dust refractive index: CC-BY-4.0, notice already shipped (lic-snicar).
- Brown et al. 2018: CC-BY-4.0 paper; only three published equations used.
