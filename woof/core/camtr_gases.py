"""WRF ``ghg_input = 1``: time-varying greenhouse gases from a CAMtr file.

A port of WRF v4.6.1 phys/module_ra_clWRF_support.F (``read_CAMgases``,
``valid_years``, ``interpolate_CAMgases``), the routine RRTMG_LWRAD and
RRTMG_SWRAD call at every radiation step when ``ghg_input = 1``, the WRF
v4 Registry default (Registry.EM_COMMON:2452).  It reads the run
directory's ``CAMtr_volume_mixing_ratio``, which WRF ships as a link to
``CAMtr_volume_mixing_ratio.SSP245``; that file is packaged here
(woof/data/camtr, sha256 9a427fd1...ca7, byte-identical to WRF v4.6.1
run/).

Mixed precision is WRF's: the day-of-year arithmetic is REAL(4), the gas
table and the interpolated value REAL(8), and the unit factors are REAL(4)
literals (``1.e-06``) widened to REAL(8).  The values returned are the
REAL(8) mole fractions; each radiation wrapper narrows them to its REAL(4)
profile (``co2vmr(ncol,k) = co2``).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

F = np.float32

#: WRF run/CAMtr_volume_mixing_ratio -> CAMtr_volume_mixing_ratio.SSP245.
CAMTR_DEFAULT = (Path(__file__).resolve().parents[1] / "data" / "camtr"
                 / "CAMtr_volume_mixing_ratio.SSP245")
CAMTR_SHA256 = "9a427fd106f8e36b30e0b29266bff1398b025b82af5b878e5a7a8e9dfe268ca7"
#: module_ra_clWRF_support.F ``cyr``: the table capacity the year search
#: walks over (rows past the file's last year read as year 0).
_CYR = 1800
_MONDAY = (0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
GAS_NAMES = ("co2", "n2o", "ch4", "cfc11", "cfc12")


@dataclass(frozen=True)
class CamtrTable:
    years: tuple[int, ...]
    juldata: tuple[np.float32, ...]
    gases: dict  # name -> tuple[float, ...] (REAL(8))
    path: str


def _leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0


def _field(line: str, start: int, stop: int) -> str:
    # A short record pads with blanks (Fortran PAD='YES'); a blank field
    # reads as zero.
    return line[start:stop].strip()


@lru_cache(maxsize=4)
def load_camtr(path: str | None = None) -> CamtrTable:
    """Read the table as WRF does: two header lines, then
    ``(I4, 1x, F8.3, 1x, 4(F10.3, 1x))`` to end of file."""
    source = Path(path) if path else CAMTR_DEFAULT
    from woof.tables import table_path
    source = table_path("camtr", source)
    lines = source.read_text(encoding="ascii").splitlines()[2:]
    years, juldata = [], []
    gases = {name: [] for name in GAS_NAMES}
    spans = ((5, 13), (14, 24), (25, 35), (36, 46), (47, 57))
    for line in lines:
        year = int(_field(line, 0, 4) or 0)
        years.append(year)
        for name, (start, stop) in zip(GAS_NAMES, spans):
            text = _field(line, start, stop)
            gases[name].append(float(text) if text else 0.0)
        mondayi = list(_MONDAY)
        if _leap(year):
            mondayi[2] = 29
        # sum(mondayi(1:6)) + real(mondayi(7))/2. - 0.5, mondata = 6
        juldata.append(F(F(F(sum(mondayi[0:6])) + F(F(mondayi[6]) / F(2.0)))
                         - F(0.5)))
    return CamtrTable(years=tuple(years), juldata=tuple(juldata),
                      gases={k: tuple(v) for k, v in gases.items()},
                      path=str(source))


def _valid_years(year_in: int, gas, tot_years: int) -> tuple[int, int]:
    """``valid_years`` (1-based indices, as in the Fortran)."""
    g = lambda i: gas[i - 1]  # noqa: E731
    yr_loc = year_in
    yr2 = yr_loc
    yr1 = yr_loc - 1
    if not any(g(i) > 0.0 for i in range(1, yr_loc)):
        for idata in range(yr_loc, tot_years):
            if g(idata) > 0.0:
                yr1 = idata
                break
        for idata in range(yr1 + 1, tot_years + 1):
            if g(idata) > 0.0:
                yr2 = idata
                break
        return yr1, yr2
    if g(yr_loc) < 0.0:
        if any(g(i) > 0 for i in range(yr_loc, tot_years + 1)):
            for idata in range(yr_loc + 1, tot_years + 1):
                if g(idata) > 0:
                    yr2 = idata
                    break
        else:
            for idata in range(yr_loc - 1, 1, -1):
                if g(idata) > 0:
                    yr2 = idata
                    break
    yr_loc = min(yr_loc - 1, yr2 - 1)
    if g(yr_loc) < 0.0:
        if any(g(i) > 0 for i in range(1, yr_loc)):
            for idata in range(yr_loc - 1, 0, -1):
                if g(idata) > 0:
                    yr1 = idata
                    break
        else:
            raise ValueError("CAMtr valid_years: no valid year below "
                             f"index {yr_loc} (WRF prints 'this case should "
                             "never happen')")
    else:
        yr1 = yr_loc
    return yr1, yr2


def _interpolate(yr, julian, yeari, juli, yr1, yr2, yearf, julf, gas) -> float:
    """``interpolate_CAMgases``: REAL(4) day arithmetic, REAL(8) result."""
    minyear = min(yr, yeari)
    maxyear = max(yr, yearf)
    fact2 = F(juli)
    fact1 = F(julf)
    x = F(julian)
    for ny in range(minyear, maxyear):
        nday = F(366 if _leap(ny) else 365)
        if ny < yeari:
            fact2 = F(fact2 + nday)
        if ny < yr:
            x = F(x + nday)
        if ny < yearf:
            fact1 = F(fact1 + nday)
    deltat = F(fact1 - fact2)
    fact1 = F(F(fact1 - x) / deltat)
    fact2 = F(F(x - fact2) / deltat)
    value = gas[yr1 - 1] * float(fact1) + gas[yr2 - 1] * float(fact2)
    if value < 0.0:
        value = -99999.0
    return value


def camtr_trace_gases(yr: int, julian, path: str | None = None) -> dict[str, float]:
    """``read_CAMgases(yr, julian, .false., "RRTMG", ...)``: the REAL(8)
    mole fractions of CO2, N2O, CH4, CFC-11 and CFC-12 at ``yr`` and the
    REAL(4) day ``julian`` (0 = 1 January 00 UTC)."""
    table = load_camtr(path)
    yr = int(yr)
    julian = F(julian)
    max_years = len(table.years)
    years = table.years + (0,) * (_CYR - max_years)
    juls = table.juldata + (F(0.0),) * (_CYR - max_years)
    found, iyear, year_in = False, 1, 0
    while not found and iyear <= _CYR:
        if years[iyear - 1] > yr:
            year_in, found = iyear, True
        elif years[iyear - 1] == yr and juls[iyear - 1] > julian:
            year_in, found = iyear, True
        iyear += 1
    if iyear >= max_years:
        year_in, found = max_years - 1, True
    if year_in == 1:
        year_in += 1
    co2r = table.gases["co2"]
    if sum(1 for v in co2r if v > 0) < 2:
        raise ValueError(f"{table.path}: fewer than two valid CO2 years")
    yr1, yr2 = _valid_years(year_in, co2r, max_years)
    nyrm, njulm = years[yr1 - 1], int(juls[yr1 - 1])
    nyrp, njulp = years[yr2 - 1], int(juls[yr2 - 1])
    values = {name: _interpolate(yr, julian, nyrm, njulm, yr1, yr2, nyrp,
                                 njulp, table.gases[name])
              for name in GAS_NAMES}
    if any(v < 0.0 for v in values.values()):
        # WRF falls back to ghg_surfvals' original values here (orig_val);
        # a packaged SSP245 table never reaches it inside 1765-2499.
        raise ValueError(
            f"CAMtr interpolation at year {yr} day {float(julian)} gave a "
            f"missing value {values}; WRF would fall back to its original "
            "fixed gases (orig_val), which is not ported")
    if values["co2"] < 270.0:
        values["co2"] = 270.0
    if values["n2o"] < 270.0:
        values["n2o"] = 270.0
    if values["ch4"] < 700.0:
        values["ch4"] = 700.0
    scale = {"co2": F(1.0e-06), "n2o": F(1.0e-09), "ch4": F(1.0e-09),
             "cfc11": F(1.0e-12), "cfc12": F(1.0e-12)}
    return {name: values[name] * float(scale[name]) for name in GAS_NAMES}


def camtr_identity(path: str | None = None) -> dict:
    """What a receipt records about the gas source."""
    import hashlib
    source = Path(path) if path else CAMTR_DEFAULT
    return {"ghg_input": 1, "file": source.name,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
