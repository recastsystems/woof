"""TEMPO's existence test for precipitating species at a tiny mass instead of R1, for fast numerics only.

TEMPO keeps a species on a level when its mixing ratio exceeds R1 = 1e-12 kg/kg (the ``*_check_and_update``
routines set ``l_q*`` and, below R1, zero the mass and its numbers). At 3 km, transport leaves graupel and snow
traces above 1e-12 on every column: on the captured mv3km hrrrv5 inputs (206,726 columns, 2023-03-31 12Z + 40 min),
qg > 1e-12 in 100 % of columns and on 53 % of all level points, but qg > 1e-8 in only 10 % of columns. Every
column therefore runs graupel fall speeds and the hail-aware graupel sedimentation substeps (19-24 % of the TEMPO
slot) for traces that carry no rain and no visible echo. In fast mode the test for the named species uses
``QMIN`` (1e-9 kg/kg): below it the species is treated exactly as TEMPO treats sub-R1 amounts (zeroed with its
number and volume, no process, no sedimentation). Nothing else changes; R1 keeps every other role.

The source owner applies :func:`tiny_mass_source` to the fast column only; strict stays ``_COLUMN`` byte for
byte (TEMPO module replay 0 words; x1.2562 6 h strict 0 of 29,644,818 against native). Fast words change by
design; the fast referee is the forecast (maps, domain rain and echo area).

THE BREAKAGE THIS ENDS: TEMPO at 3 km cost 143 ms per call on one RTX PRO 6000 (Thompson 22 ms), a fifth of it
spent sedimenting graupel traces below 1e-9 kg/kg on every column.

Knob: ``WOOF_HEX_TEMPO_TINY_MASS`` = ``graupel`` (default) | ``precip`` (rain, snow and graupel) | ``off``.
"""
from __future__ import annotations

import os

#: kg/kg. Below this a named species is absent in fast numerics (TEMPO's R1 is 1e-12).
QMIN = "1.0e-9f"

_TESTS = {
    "graupel": "if (qg1d[k] + qgten[k] * C.global_dt > PS(r1)) {",
    "rain": "if (qr1d[k] + qrten[k] * C.global_dt > PS(r1)) {",
    "snow": "if (qs1d[k] + qsten[k] * dt_ > PS(r1)) {",
}
SETS = {"graupel": ("graupel",), "precip": ("rain", "snow", "graupel"), "off": ()}


def species(setting: str | None = None) -> tuple[str, ...]:
    """The species whose existence test uses QMIN (WOOF_HEX_TEMPO_TINY_MASS, default graupel)."""
    value = (setting if setting is not None else os.environ.get("WOOF_HEX_TEMPO_TINY_MASS", "graupel")).strip().lower()
    if value not in SETS:
        raise ValueError(f"WOOF_HEX_TEMPO_TINY_MASS must be one of {sorted(SETS)}, not {value!r}")
    return SETS[value]


def tiny_mass_source(column: str, setting: str | None = None) -> str:
    """The fast column with the named species' existence test at QMIN."""
    for name in species(setting):
        test = _TESTS[name]
        if column.count(test) != 1:
            raise ValueError(f"TEMPO {name} check_and_update test drifted from the tiny-mass edit")
        column = column.replace(test, test.replace("PS(r1)", QMIN))
    return column
