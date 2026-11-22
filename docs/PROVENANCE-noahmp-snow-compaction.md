# Provenance: Noah-MP snow compaction option (`opt_snow_compact = 2`)

Label: **re-implemented from published descriptions, not bitwise to upstream.**

Option 1 (the default) is WRF v4's fixed overburden parameter C_bd = 0.021 m3/kg
and keeps today's bytes: the default CUDA unit (`noahmp_water.cu`, `k_water`) is
not edited, and option 1 never reaches the new code.

## Source

Abolafia-Rosenzweig, R., He, C., Chen, F., and Barlage, M. (2024). Evaluating and
enhancing snow compaction process in the Noah-MP land surface model. *Journal of
Advances in Modeling Earth Systems*, 16, e2023MS003869.
doi:10.1029/2023MS003869. Open access, CC BY 4.0 (copy read: NOAA Institutional
Repository, noaa/65638).

No file from NCAR/noahmp, the WRF 4.8.0 noahmp submodule, the authors' HRLDAS
code repositories or any other UCAR-licensed copy was opened, cloned or read.
Only the paper's text, equations and Figure 2 were used.

## Routine by routine

| Routine | What it implements | Paper |
| --- | --- | --- |
| `woof/core/noahmp_snow.py::compaction_cbd` | C_bd = -0.00069503 T_air + 0.20606699 (T_air in K) | Section 2.4, Eq. 11 |
| same | pressure-binned lower constraint: 0.019 for psfc < 80,000 Pa, 0.018 for 80,000 <= psfc < 85,000 Pa, 0.017 for psfc >= 85,000 Pa | Section 2.4, Eq. 12, Figure 2 |
| same | upper constraint C_bd <= 0.0315 | Section 2.4 text after Eq. 12 (Figure 4: little temperature control below 251 K) |
| `noahmp_snow.py::compact(cbd=...)` | the unchanged Anderson (1976) overburden term with C_bd swapped in | Eq. 5 (eta0 = 1.33e6 kg s m-2, as already in the port, He et al. 2021) |
| `noahmp_snow.py::snowwater(compaction=, psfc=)` | selects the option; T_air is SNOWWATER's SFCTMP forcing | Section 2.4 ("surface air temperature and pressure") |
| `kernels/noahmp_water_options.cuh::ar24_cbd` | Eq. 11, Eq. 12 floors, cap; float32 one rounding per op, constants as bit patterns | as above |
| `kernels/noahmp_water_options.cuh::compact_cbd`, `snowwater_cbd` | the imported default COMPACT and SNOWWATER with one edit each (C2 -> cbd; compact -> compact_cbd), checked by test | Eq. 5 |
| `noahmp_water_slab.py`, `noahmp_column_slab.py`, `noahmp_runtime.py`, `physics.py`, `config.py` | plumbing: option admission, label warning, surface pressure slot in the option kernel's forcing row | none (our code) |

## The one reading call

Eq. 12 is printed as `min(C_bd, 0.019)` (and 0.018, 0.017). The same section
calls these "variable lower constraints" and names them C_bd,min, and Figure 2
draws each pressure line as Eq. 11 that turns flat at 0.019 / 0.018 / 0.017 on
the warm side (the corners fall at 269.1 K, 270.6 K and 272.0 K). A `min` would
cap cold-snow C_bd at 0.017-0.019 instead, contradicting both. So the bins are
applied as floors, `max(C_bd, bin)`. The paper's own check that C_bd equals
WRF's 0.021 at about 266.3 K holds for Eq. 11 (266.27 K).

## Scope

Land snowpack (SNOWWATER's COMPACT) only. Glacier columns keep their own
COMPACT_GLACIER unchanged; the paper is about the land snowpack and validates
on SNOTEL. The hex WOOF-physics route calls this same port, so it has the
option with no separate copy.

## Validation data

SNOTEL (USDA NRCS, US government, public) for snow depth and SWE; ASOS (NOAA,
public) for T2/Td2.
