"""CuPy-free snow admission shared by MYNN and registry generation."""
from woof.microphysics_schemes import NAMED_SCHEMES as _NAMED_MP_SCHEMES, PORTED_WRF_SNOW_SELECTORS
from woof.core.mp_column_schemes import table_schemes_carrying

MYNN_SNOW_MICROPHYSICS = frozenset((2, 4, 6, 7, 8, 9, 10, 13, 14, 16, 18, 24, 26, 27, 28, 29, 30, 38, 40, 56, 96, 106)) | frozenset(
    _s.mp_id for _s in _NAMED_MP_SCHEMES.values()
    if "qs" in _s.ice_mass_species) | frozenset(table_schemes_carrying("qs"))

MYNN_SNOW_MICROPHYSICS |= frozenset(PORTED_WRF_SNOW_SELECTORS)
