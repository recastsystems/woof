# Provenance: Noah-MP wetland option (`noahmp_wetland = 1`)

Label on every listing: **re-implemented from published descriptions, not bitwise to upstream.** With the option off (the default), nothing in the v4 port changes; the proof is in the port's own results.

## What was read

- Zhang, Z., Chen, F., Barlage, M., Bortolotti, L. E., Famiglietti, J., Li, Z., Ma, X. and Li, Y. (2022). Cooling effects revealed by modeling of wetlands and land-atmosphere interactions. *Water Resources Research* 58(3), e2021WR030573. doi:10.1029/2021WR030573.
- The publisher's PDF and the UCAR OpenSky copy were both behind a bot check. The text actually read was the authors' own preprint of the same paper on ESS Open Archive: doi:10.1002/essoar.10507347, version 3 (posted 1 Dec 2022). Sections 2.3-2.5, 3.1-3.3, 4 and 5 were read, and the equation page was rendered and checked by eye. **Equation numbers below are the preprint's.** The published version may number them differently.
- No file from NCAR/noahmp, from the WRF 4.8.0 noahmp submodule, or from any UCAR-licensed copy was opened, cloned, read or quoted. Our own WRF v4.6.1-lineage port (public domain, already in the tree) is the code that was modified.

## Routine by routine

| Routine | What it does | Source |
| --- | --- | --- |
| `wetland_fsat` (`woof/core/kernels/noahmp_water_options.cuh`) | `FSAT = FSATMX * (SH2O_1 - SMCWLT_1) / (SMCREF_1 - SMCWLT_1)`, replacing the water-table FSAT | Eq. 7, Sec. 2.4. The clamp to `[0, FSATMX]` is ours: the paper states no bound, and a fraction needs one. |
| `d_soilwater<INF,RUN,WET>` | FSAT partitions the surface input: the FSAT share runs off, the rest infiltrates. This is unchanged v4 code with the new FSAT. | Eqs. 4-5 (TOPMODEL partition, Sec. 2.3), which our v4 port already has as `opt_run` 1, 2 and 5 |
| `d_soilwater` wetland outputs | Store inflow = surface input x FSAT, in mm/s | Eq. 9 (with Eq. 8, input = rain + snowmelt; our port's `QINSUR`) |
| `d_soilwater` open-water weighting | While open water covers FSAT, soil evaporation and transpiration extraction are weighted by `1 - FSAT` | Eq. 11 weights the land tiles' fluxes by `1 - FSAT`. Applying the same weight to the soil's water extraction is ours, so the column water budget closes. |
| `noahmp_wetland.advance_store` (GPU) | Penman open-water evaporation `E = (Delta Rn + rho cp VPD ga) / (lambda (Delta + gamma))` | Eq. 10. Tetens saturation vapour pressure, Noah-MP's `HVAP` and `CPAIR`, `Rn = FSA - FIRA`, and `ga =` the bare-ground exchange coefficient `CHB` are our choices of inputs. The paper names the Penman form, not its inputs. |
| same | Store `W += inflow - FSAT*E`, water-limited; spill `max(W - WCAP, 0)` to surface runoff | Eqs. 12-13 and the text of Sec. 2.5 ("water exceeding the maximum capacity will become the outflow"). The printed Eq. 12 compares a rate with a depth; we read it as the store spilling what exceeds `WCAP`. |
| same | `LH = (1-FSAT)*LH_land + FSAT*lambda*E`, and `SH` is the residual | Eq. 11, plus "sensible heat flux is calculated as the residual term from the energy balance equation" (Sec. 2.5). Canopy interception evaporation is left unweighted because the canopy store is grid-mean (ours). |
| `noahmp_wetland.open_water` | Open water exists only where the store holds water, there is no snow, the ground is unfrozen, and the column is not urban | Ours. The paper describes the warm-season open-water period (Sec. 4). |
| `noahmp_wetland.cold_start_store` | First store = `min(WCAP, 200 mm)` x Eq. 7 relative saturation | Ours. The paper spins up for one month (Sec. 3.3), which a forecast cannot do. The 200 mm ceiling is the summer storage the paper reports for the wettest part of the region (Sec. 3.2, Fig. 6a); Eq. 14 capacities reach metres on 3 km cells with relief. |
| `static-fields/src/wetland.rs` `depression_capacity_mm` | `WCAP = mean(max(Hbar - H_i, 0))` over the cell's 90 m DEM pixels, in mm | Eq. 14, Sec. 3.2. The printed sum has no `1/n` and a sign that is never positive, while the paper's capacities are 5-500 mm (Sec. 3.1, Fig. 6). We take the cell-mean depth below the cell mean. |
| `static-fields/src/wetland.rs` FSATMX | Mean GLWD v2 combined wetland percent over the cell footprint / 100 | The paper uses the GIEMS-2 maximum inundation fraction (Sec. 3.2). GIEMS has no open licence, so we use GLWD v2. |

## Runoff pairing (read from the paper, not guessed)

- Single-point and offline runs replace "the TOPMODEL-based runoff generation" saturated fraction (Eqs. 4-6; Niu and Yang 2005; Niu et al. 2011). Sec. 2.3 and 3.1.
- Coupled WRF runs: "The second simulation (WS simulation) incorporates the wetland scheme upon the shallow groundwater scheme" of Miguez-Macho et al. (2007) (Sec. 3.3). The Discussion repeats it: "adding the wetland scheme on top of the MMF groundwater scheme".
- In Noah-MP v4 those are `opt_run` 1 and 2 (TOPMODEL with groundwater, and with an equilibrium water table) and 5 (MMF). The option refuses every other `opt_run`, naming why.

## Data

| Data | Use | Licence (checked at source) |
| --- | --- | --- |
| GLWD v2.0, `GLWD_v2_0_area_pct` (Lehner et al. 2025, ESSD 17, 2277; figshare doi:10.6084/m9.figshare.28519994) | FSATMX | CC BY 4.0. Checked 2026-11-09 on the figshare record and in TechDoc Sec. 4.1, which asks users not to redistribute the whole dataset in its original format. We download it at prep and never ship it. |
| Copernicus DEM GLO-90 (AWS `copernicus-dem-90m`) | WCAP | Copernicus WorldDEM-90 licence: free, worldwide, with attribution. Already verified in `tools/hydro_world/data/sources.v1.json`. |
| MERIT DEM | not used | The paper's DEM. It is CC BY-NC and never ships. |
| GIEMS-2 | not used | The paper's FSATMX source. It has no open licence. |

Attribution carried in every static receipt: "GLWD v2: Lehner, B. et al. (2025), CC BY 4.0. Produced using Copernicus WorldDEM-90 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."

## Default-readiness additions (2026-11-09)

Same label: **re-implemented from published descriptions, not bitwise to upstream.** No UCAR-licensed file was opened.

| Routine | What it does | Source |
| --- | --- | --- |
| `d_soilwater<INF,3,1>` (`wetland_split3`) | With the Schaake runoff (`opt_run=3`, the Noah-MP default), the cell is split into the wetland tile FSAT (Eq. 7, FSATMX from the static map) and the land tile `1 - FSAT`. The FSAT share of rain + snowmelt goes to the store (Eqs. 8-9), carried inside RUNSRF as on the TOPMODEL path. The land tile gets the same input per unit area through the unchanged Schaake partition, and its runoff and infiltration are area-weighted by `1 - FSAT`: `RUNSRF = (1-FSAT) R_schaake + FSAT QINSUR`, infiltration `(1-FSAT) I_schaake`. | Ours, derived from the paper's tile structure (Eq. 11 splits the cell into these two tiles and weights the land fluxes by `1 - FSAT`; Eqs. 4-5 send the saturated share to runoff). The paper itself ran TOPMODEL and MMF only. With FSATMX = 0 this is the v4 `opt_run=3` code, byte for byte. |
| `noahmp_nest.feedback_selected_noahmp` | Feeds back `wetstorexy`, `wetfsatxy`, `wetlhxy` from a child (copied word, as `copy_fcnm`); refuses a pair whose `noahmp_wetland` differs. FSATMX and WCAP stay each domain's own. | Ours (WRF feedback convention). |
| `relocation_init.surface_continuation_fields` | A moving nest carries the store on the overlap; newborn columns cold-start from their own soil and capacity. | Ours. |
| `static/wetland.py` `glwd_area_pct_window` | Reads the figshare zip directory with range requests, streams only the `GLWD_v2_0_area_pct.tif` member (252 MB of the 925 MB archive), checks its CRC-32, and writes the domain window (8x8-tile blocks, LZW tiles copied byte for byte) as a small cached GeoTIFF. FSATMX from the window equals FSATMX from the full raster in every word. | Data handling only; GLWD v2 TechDoc 4.1 asks users not to redistribute the dataset, and nothing is redistributed. |

**Default status: still opt-in.** Scored against ASOS with sheets looked at (the wetland-default check), all with `opt_run=3`:

| Case | T2 bias / RMSE, off -> on | Td2 bias / RMSE, off -> on |
| --- | --- | --- |
| Prairie Pothole, 2022-07-20 12Z, 12 h (76 stations) | +1.05 / 2.07 -> +0.82 / 1.99 K | -4.47 / 7.21 -> -4.15 / 7.08 K |
| Florida, 2023-10-25 12Z, 12 h (70 stations) | +0.68 / 1.63 -> +0.54 / 1.56 K | -1.73 / 2.51 -> -1.59 / 2.39 K |
| Gulf coast (Louisiana), 2024-01-29 12Z, 12 h (83 stations) | +0.45 / 5.43 -> +0.30 / 5.40 K | +1.60 / 5.59 -> +1.97 / 5.91 K |

The two warm-season cases improve; the cold-season Gulf case, where the model is already too moist, gets moister (coastal marsh Td2 +2 K, LH +50-150 W/m2 at 18Z), so the default was not flipped.
