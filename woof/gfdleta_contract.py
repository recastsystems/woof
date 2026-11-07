"""GFDL configuration identity, without importing the device implementation."""

GFDLETA_REAL_FIELDS = tuple('htop hbot htopr hbotr cuppt acfrst acfrcv cfracl cfracm cfrach swdownc rswtoa rlwtoa czmean'.split())
GFDLETA_INTEGER_FIELDS = ('ncfrst', 'ncfrcv')
GFDLETA_FIELDS = GFDLETA_REAL_FIELDS + GFDLETA_INTEGER_FIELDS

def inert_gfdleta_fields(values):
    """Only radiation 99 consumes the CO2 table selection."""
    aggregate = values.get("ra_physics", 0)
    pair = tuple(aggregate if values.get(name, -1) == -1 else values[name]
                 for name in ("ra_lw_physics", "ra_sw_physics"))
    return frozenset() if 99 in pair else frozenset({"co2tf"})
