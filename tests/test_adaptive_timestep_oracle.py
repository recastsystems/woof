"""Grade :mod:`woof.core.adaptive_timestep` against the WRF ``calc_dt`` oracle.

The oracle is ``tools/calc_dt_wrf_oracle/calc_dt_oracle.csv``, produced by
``run_calc_dt.F90`` -- upstream's ``calc_dt`` compiled verbatim against
WRF's own ESMF time manager (``external/esmf_time_f90``, libesmf_time.a),
swept over 1728 (max_cfl, target_cfl, max_increase_factor, last_dt)
combinations that cross every branch and both quantisation ties.

The comparison is on the EXACT RATIONAL, ``out_S + out_Sn/out_Sd``, not on
the REAL(4) image -- the interval is the controller's state and a float
comparison would hide a one-part-in-100 numerator error.
"""

from __future__ import annotations

import csv
from fractions import Fraction
from pathlib import Path

import pytest

from woof.core.adaptive_timestep import PRECISION, calc_dt, requantise

ORACLE = (Path(__file__).resolve().parents[1]
          / "tools" / "calc_dt_wrf_oracle" / "calc_dt_oracle.csv")


def _rows():
    with ORACLE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            yield {k: v.strip() for k, v in row.items()}


def _interval(whole: str, num: str, den: str) -> Fraction:
    d = int(den)
    if abs(d) < 1:                       # WRF's real_time guard
        return Fraction(int(whole))
    return Fraction(int(whole)) + Fraction(int(num), d)


ROWS = list(_rows())


def test_oracle_is_present_and_populated():
    """A corpus check: this gate is worthless if it matches nothing."""
    assert ORACLE.exists(), f"oracle CSV missing at {ORACLE}"
    assert len(ROWS) == 1728, f"expected 1728 oracle rows, got {len(ROWS)}"


def test_oracle_exercises_every_branch():
    """Each of calc_dt's three arms must actually appear in the corpus.

    Without this the suite could pass while only ever touching the growth
    branch, which is exactly how a gate ends up proving nothing.
    """
    negligible = grow = reduce_ = 0
    for row in ROWS:
        cfl, target = float(row["max_cfl"]), float(row["target_cfl"])
        if cfl < 0.001:
            negligible += 1
        elif cfl > target:
            reduce_ += 1
        else:
            grow += 1
    assert negligible > 0 and grow > 0 and reduce_ > 0, (
        f"branch coverage negligible={negligible} grow={grow} "
        f"reduce={reduce_}")


def test_oracle_exercises_the_min_factor_floor():
    """The 0.1 clamp must be reached, or its port is untested."""
    hit = [r for r in ROWS
           if float(r["max_cfl"]) > 3.0 * float(r["target_cfl"])]
    assert hit, "no row drives the dampened factor below the 0.1 floor"


@pytest.mark.parametrize("row", ROWS, ids=lambda r: (
    f"cfl{r['max_cfl']}_tgt{r['target_cfl']}_mif{r['mif']}"
    f"_last{r['last_S']}+{r['last_Sn']}/{r['last_Sd']}"))
def test_calc_dt_matches_wrf_exactly(row):
    last = _interval(row["last_S"], row["last_Sn"], row["last_Sd"])
    expected = _interval(row["out_S"], row["out_Sn"], row["out_Sd"])
    got = calc_dt(last, float(row["max_cfl"]), float(row["mif"]),
                  float(row["target_cfl"]), int(row["precision"]))
    assert got == expected, (
        f"calc_dt({last}, max_cfl={row['max_cfl']}, "
        f"mif={row['mif']}, target={row['target_cfl']}) = {got} "
        f"({float(got):.9g} s), WRF says {expected} "
        f"({float(expected):.9g} s)")


def test_precision_matches_the_fortran_parameter():
    assert PRECISION == 100


def test_calc_dt_alone_does_NOT_land_on_a_hundredth_tick():
    """``calc_dt`` returns ``last_dt * num/100``, which compounds denominators.

    Found by asserting the opposite: with ``last_dt = 1.5 s`` and a
    growth numerator of 105, ``calc_dt`` gives 63/40 s and ``100*dt`` is
    157.5.  So the integer-tick property does NOT come from ``calc_dt``;
    it comes from the requantisation two steps later, whose upstream
    comment justifies it only as overflow protection.  It is doing more
    than that -- it is the reason an adaptive dt is representable on
    WOOF-ArWen's exact-tick clock at all.  Pinned so that discovery is not
    re-made the hard way.
    """
    offenders = []
    for row in ROWS:
        got = calc_dt(_interval(row["last_S"], row["last_Sn"],
                                row["last_Sd"]),
                      float(row["max_cfl"]), float(row["mif"]),
                      float(row["target_cfl"]), int(row["precision"]))
        if (got * PRECISION).denominator != 1:
            offenders.append((row["last_S"], row["last_Sn"], row["last_Sd"],
                              row["mif"], got))
    assert offenders, (
        "expected calc_dt to produce off-grid intervals for fractional "
        "last_dt; if this now passes cleanly the corpus lost its "
        "fractional last_dt rows and the requantise test below is vacuous")


def test_requantise_puts_every_result_on_a_hundredth_tick():
    """The property that lets an integer-tick clock hold an adaptive dt.

    After ``requantise`` -- upstream's :179-184 -- every interval is an
    exact multiple of 1/100 s, so ``tick_den`` carrying a factor of 100 is
    all WOOF-ArWen's clock needs to hold a varying dt without ever accumulating
    seconds in floating point.
    """
    for row in ROWS:
        got = calc_dt(_interval(row["last_S"], row["last_Sn"],
                                row["last_Sd"]),
                      float(row["max_cfl"]), float(row["mif"]),
                      float(row["target_cfl"]), int(row["precision"]))
        scaled = requantise(got) * PRECISION
        assert scaled.denominator == 1, (
            f"{got} s requantised is still not a whole number of "
            f"1/{PRECISION} s ticks")
