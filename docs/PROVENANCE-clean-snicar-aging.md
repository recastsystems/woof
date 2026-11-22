# Provenance: clean-snicar-aging (Noah-MP SNICAR grain aging, 2026-11-09)

Re-implemented from published descriptions, not bitwise to upstream.
No file from NCAR/noahmp, from the WRF 4.8.0 noahmp submodule, from the
Noah-MP SNICAR coupling repository named in the talk below, or from any
UCAR-licensed copy was opened, cloned or read. The CESM aging table
(`snicar_drdt_bst_fit_60`) was not read; the dry-aging fits in
`woof/data/snow_papers/optics.bin` were regenerated earlier from the
Flanner & Zender 2006 model (see docs/SNOW-PAPERS.md) and are unchanged.

| Routine (woof/core/kernels/snow_papers.cu) | Change | Source |
| --- | --- | --- |
| `snow_papers_albedo`, fresh-snow radius `r0` | New `snicar_fresh_grain_opt`: 1 = 54.526-204.526 um (unchanged default), 2 = 33.0-91.0 um; both a linear ramp in air temperature between -30 and 0 C | Values: Lin, T.-S., "Improving snow albedo modeling in Noah-MP v5 via coupling with a snow radiative transfer model", public NCAR RAL talk, 3 June 2024, slide "Optimization of snow grain size" (Original 54.526 / 204.526, Optimized 33.0 / 91.0, constrained by MODSCAG, Painter et al. 2009, Rittger et al. 2020). Paper: Lin et al. 2025, J. Hydrometeor. 26(2), doi:10.1175/JHM-D-24-0082.1. Ramp shape: CLM5.0 Technical Note, Surface Albedos, section "Snow Aging" (fresh radius linear between -30 C and 0 C). |
| `age`, dry metamorphism below the fit start | Option 2 only: the regenerated SSA-decay trajectory r^kappa = r0^kappa (1 + t/tau) (Flanner & Zender 2006, doi:10.1029/2005JD006834, Eq. 16 fit form; r0 from SSA = 60 m2/kg) is followed for r < r0 (negative equivalent age) instead of clamping r to r0, so 33 um grains are not lifted to 54.5 um on their first step | Flanner & Zender 2006; the consequence of Lin's smaller fresh radii, our reading |
| `age`, wet metamorphism | Defect fix, default for every opt_alb=3 run: d(r^3)/dt = (3/(4 pi)) * 4.22e5 * f_liq^3 um3/s, i.e. dr/dt = 1e18 C1 f_liq^3 / (4 pi r^2), C1 = 4.22e-13 m3/s, f_liq = liquid/(liquid+ice). The previous code omitted 3/(4 pi) and grew wet grains 4.19x too fast | CLM5.0 Technical Note Eq. 2.3.75 (Brun 1989, Ann. Glaciol. 13; as used by Flanner et al. 2007, doi:10.1029/2006JD008003) |
| refreezing, fresh-snow addition, vertical grain profile | Unchanged (already in rel/2.9.13): refrozen mass enters at 1000 um, fresh mass at r0, three-layer mass-weighted radius carrier `snicar_statexy` | CLM5.0 Technical Note Eq. 2.3.72 and the refreezing radius 1000 um; Lin et al. 2025 section 2b2 |

Data: none new. The dr/dt fits are the ones regenerated earlier
from the published Flanner-Zender microphysical model.

Labels: the option, config description, registry entry and identity say
"re-implemented from published descriptions, not bitwise to upstream".
