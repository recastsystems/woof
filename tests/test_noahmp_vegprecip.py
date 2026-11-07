"""Acceptance gate for the Noah-MP PHENOLOGY / PRECIP_HEAT FP32 port.

The bar is bitwise identity with the unmodified WRF v4.6.1 module -- max_ulp 0
*and* an identical 32-bit pattern, which is strictly stronger because the ULP
map deliberately treats -0.0 and +0.0 as one point.

The fixtures under ``tests/data/oracles/noahmp/`` were produced by
``tools/noahmp_wrf461_oracle/build_vegprecip.sh`` from

    tree   the pinned stock-WRF v4.6.1 gate checkout
    commit d66e442fccc04111067e29274c9f9eaccc3cef28
    sha256 bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282

with the accessibility lift as the only source change, and were shown to be
identical at ``-O2 -ftree-vectorize -funroll-loops`` (WRF's own FCOPTIM), at
``-O0``, and at ``-ffp-contract=off``.
"""

from __future__ import annotations

import csv
import math
import os
import struct

import pytest

from woof.core import noahmp_vegprecip as vp
from woof.core.fp32_ulp import bitwise_identical, max_ulp

_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "data", "oracles", "noahmp",
)
PHEN_CSV = os.path.join(_DATA, "noahmp-vegprecip-phenology.csv")
PRCP_CSV = os.path.join(_DATA, "noahmp-vegprecip-precip_heat.csv")

PHEN_OUT = ("lai", "sai", "elai", "esai", "igs", "fb")
PRCP_OUT = ("canliq", "canice", "qintr", "qdripr", "qthror", "qints",
            "qdrips", "qthros", "pahv", "pahg", "pahb", "qrain", "qsnow",
            "snowhin", "fwet", "cmc")


def _uh(s: str) -> float:
    return struct.unpack("<f", struct.pack("<I", int(s, 16)))[0]


def _h(x: float) -> str:
    return f"{struct.unpack('<I', struct.pack('<f', x))[0]:08X}"


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _phen_kwargs(row, **override):
    kw = dict(
        dveg=int(row["dveg"]),
        vegtyp=int(row["vegtyp"]),
        croptype=int(row["croptype"]),
        snowh=_uh(row["snowh"]),
        tv=_uh(row["tv"]),
        lat=_uh(row["lat"]),
        yearlen=int(row["yearlen"]),
        julian=_uh(row["julian"]),
        lai=_uh(row["lai_in"]),
        sai=_uh(row["sai_in"]),
        troot=_uh(row["troot"]),
        pgs=int(row["pgs"]),
        iswater=int(row["iswater"]),
        isbarren=int(row["isbarren"]),
        isice=int(row["isice"]),
        urban_flag=bool(int(row["urban_flag"])),
        hvt=_uh(row["hvt"]),
        hvb=_uh(row["hvb"]),
        tmin=_uh(row["tmin"]),
        laim=[_uh(row[f"laim{i:02d}"]) for i in range(1, 13)],
        saim=[_uh(row[f"saim{i:02d}"]) for i in range(1, 13)],
    )
    kw.update(override)
    return kw


def _prcp_kwargs(row, **override):
    kw = dict(
        iloc=int(row["iloc"]),
        jloc=int(row["jloc"]),
        vegtyp=int(row["vegtyp"]),
        ist=int(row["ist"]),
        dt=_uh(row["dt"]),
        uu=_uh(row["uu"]),
        vv=_uh(row["vv"]),
        elai=_uh(row["elai"]),
        esai=_uh(row["esai"]),
        fveg=_uh(row["fveg"]),
        bdfall=_uh(row["bdfall"]),
        rain=_uh(row["rain"]),
        snow=_uh(row["snow"]),
        fp=_uh(row["fp"]),
        canliq=_uh(row["canliq_in"]),
        canice=_uh(row["canice_in"]),
        tv=_uh(row["tv"]),
        sfctmp=_uh(row["sfctmp"]),
        tg=_uh(row["tg"]),
        ch2op=_uh(row["ch2op"]),
    )
    kw.update(override)
    return kw


PHEN_ROWS = _rows(PHEN_CSV)
PRCP_ROWS = _rows(PRCP_CSV)


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------
@pytest.mark.parametrize("row", PHEN_ROWS, ids=[r["case"] for r in PHEN_ROWS])
def test_phenology_is_bit_exact(row):
    got = vp.phenology(**_phen_kwargs(row))
    for name in PHEN_OUT:
        g, w = getattr(got, name), _uh(row[name])
        assert max_ulp([g], [w]) == 0, f"{row['case']}.{name}"
        assert bitwise_identical([g], [w]), (
            f"{row['case']}.{name}: got {_h(g)} want {row[name]}"
        )


@pytest.mark.parametrize("row", PRCP_ROWS, ids=[r["case"] for r in PRCP_ROWS])
def test_precip_heat_is_bit_exact(row):
    got = vp.precip_heat(**_prcp_kwargs(row))
    for name in PRCP_OUT:
        g, w = getattr(got, name), _uh(row[name])
        assert max_ulp([g], [w]) == 0, f"{row['case']}.{name}"
        assert bitwise_identical([g], [w]), (
            f"{row['case']}.{name}: got {_h(g)} want {row[name]}"
        )


# --------------------------------------------------------------------------
# the fixture must actually be a fixture
# --------------------------------------------------------------------------
def test_fixture_pins_the_option_identity():
    """Every row runs at the WRF Registry defaults, not at some other identity."""
    assert PHEN_ROWS, "phenology fixture is empty"
    assert PRCP_ROWS, "precip_heat fixture is empty"
    for row in PHEN_ROWS:
        assert int(row["dveg"]) == 4, row["case"]
        assert int(row["croptype"]) == 0, row["case"]
    for row in PHEN_ROWS + PRCP_ROWS:
        assert row["binds"].strip(), f"{row['case']} does not say what it binds"


def test_dead_branches_are_refused_not_guessed():
    row = PHEN_ROWS[0]
    with pytest.raises(ValueError):
        vp.phenology(**_phen_kwargs(row, dveg=2))
    with pytest.raises(ValueError):
        vp.phenology(**_phen_kwargs(row, croptype=1))


def test_gate_is_not_vacuous_phenology():
    """Small perturbations of live inputs must move an output.

    A gate that passes for any implementation is worthless.  SNOWH is moved by
    a single ULP, which is enough because it enters ``EXP(-SNOWH/0.2)``
    directly.  JULIAN is moved by a quarter day instead: one ULP of
    ``julian = 200.3`` is 1.5e-5, and ``12.0 * DAY`` is evaluated at a
    magnitude whose own ULP is 2.4e-4, so a one-ULP nudge is absorbed by the
    very first multiply.  That is a property of the routine, not slack in the
    gate, and the mutation study is what proves JULIAN is read at all.
    """
    row = next(r for r in PHEN_ROWS if r["case"] == "PH11_EXP_PARTIAL_BURIAL")
    base = vp.phenology(**_phen_kwargs(row))

    snowh1 = _uh(f"{int(row['snowh'], 16) + 1:08X}")
    assert tuple(vp.phenology(**_phen_kwargs(row, snowh=snowh1))) != tuple(base)

    julian1 = _uh(row["julian"]) + 0.25
    assert tuple(vp.phenology(**_phen_kwargs(row, julian=julian1))) != tuple(base)


def test_gate_is_not_vacuous_precip_heat():
    row = next(r for r in PRCP_ROWS if r["case"] == "PR01_CAP_LIMITED")
    base = vp.precip_heat(**_prcp_kwargs(row))
    for field, kwarg in (("rain", "rain"), ("canliq_in", "canliq"),
                         ("tv", "tv")):
        bumped = _uh(f"{int(row[field], 16) + 1:08X}")
        got = vp.precip_heat(**_prcp_kwargs(row, **{kwarg: bumped}))
        assert tuple(got) != tuple(base), f"{field} +1ulp changed nothing"


def test_transcendentals_are_glibc_not_numpy():
    """The port must not fall back to a correctly-rounded exp/pow.

    Over the arguments these leaves form, glibc's expf and the float64-then-
    round-once answer disagree often enough that swapping them would break the
    gate.  This asserts the two really are different functions, so a future
    edit that "simplifies" noahmp_libm away cannot pass silently.
    """
    from woof.core.noahmp_libm import expf, f32, powf

    differ_exp = sum(
        1 for i in range(20000)
        if _h(expf(f32(-0.004 * i))) != _h(f32(math.exp(f32(-0.004 * i))))
    )
    differ_pow = sum(
        1 for i in range(1, 20000)
        if _h(powf(f32(i / 20000.0), f32(0.667)))
        != _h(f32(math.pow(f32(i / 20000.0), f32(0.667))))
    )
    assert differ_exp > 0, "expf transcription is indistinguishable from exp()"
    assert differ_pow > 0, "powf transcription is indistinguishable from pow()"


def test_min_max_tie_breaking_matches_gfortran():
    """gfortran's minss/maxss return the *second* operand on a tie.

    Observable: it decides the sign of a zero, and PR19 puts a signed zero
    through exactly that path.
    """
    assert _h(vp._fmax(-0.0, 0.0)) == "00000000"
    assert _h(vp._fmax(0.0, -0.0)) == "80000000"
    assert _h(vp._fmin(-0.0, 0.0)) == "00000000"
    assert _h(vp._fmin(0.0, -0.0)) == "80000000"
