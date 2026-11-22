"""Arm 2's instrument, pinned against the rows it must reproduce.

`TC-INTENSITY.md:3178` records a table this campaign already acted on --
it is the evidence that Kain-Fritsch buys its 62% pressure recovery by
destroying the outer rainbands, and therefore the reason New Tiedtke was
ported at all.  The table was produced ad hoc; no script in
``tools/tc-intensity-diagnostics/`` emitted it.

``annulus_condensate.py`` is that script, written for Phase 5's arm 2.
Its correctness claim is not "the code looks right" -- it is that the
same code reproduces all three published rows from the three runs that
produced them, and these tests are that claim.

WHY THIS MATTERS MORE THAN A UNIT TEST OF THE ARITHMETIC.  Two things in
the metric were NOT recoverable from the prose and had to be recovered by
validation:

  * which species count as "precipitating condensate" (rain + snow +
    graupel, excluding cloud water and cloud ice); and
  * what "banded area" thresholds against -- an absolute cut-off in kg/m2
    cannot reproduce the rows at all, because the published fractions are
    near-equal across schemes (13.7 / 14.7 / 14.9%) while the means differ
    2.3x.  It is TWICE THE ANNULUS MEAN, a relative measure.

A wrong choice on either would have produced a plausible number for the
New Tiedtke run and no way to notice.  Phase 5's arm 2 fails the phase on
its own, so the instrument had to be graded before the run it grades.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import pathlib

import pytest

TOOL = (pathlib.Path(__file__).resolve().parents[1] / "tools"
        / "tc-intensity-diagnostics" / "annulus_condensate.py")

#: TC-INTENSITY.md:3178, verbatim.  mean / p95 / p99 / CV / banded-area.
PUBLISHED = {
    "GF":  (1.637, 7.717, 17.111, 2.12, 13.7),
    "WRF": (1.163, 4.995, 10.684, 1.95, 14.7),
    "KF":  (0.725, 2.838,  5.467, 1.86, 14.9),
}

#: Tolerances.  The amplitude columns agree to about 0.3%, which is the
#: residual from taking the centre at the nearest track fix rather than
#: interpolating; the fractions agree to a tenth of a point.  Wide enough
#: not to be brittle, tight enough that a changed species set or a changed
#: threshold definition fails immediately -- excluding graupel moves the
#: mean by more than 1%, and any absolute threshold moves banded-area by
#: 10 points or more.
REL_TOL = 0.01
BANDED_TOL = 0.5


def load_tool():
    spec = importlib.util.spec_from_file_location("annulus_condensate", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    if not TOOL.is_file():
        pytest.skip("the arm-2 diagnostic is not present")
    return load_tool()


def measure(tool, which):
    """One published row, recomputed from the run that produced it."""
    when = tool.T0 + dt.timedelta(hours=5)
    if which == "WRF":
        root, run_dir = tool.WRF_TIEDTKE, None
    else:
        run_dir = tool.ARW_ROOT / ("run_myj" if which == "GF" else "run_kf")
        root = run_dir / "wrfout"
    if not root.is_dir():
        pytest.skip(f"{which}: {root} is not on this box")
    path = tool.frame_for(root, "d02", when)
    if path is None:
        pytest.skip(f"{which}: no d02 frame at f005")
    centre = tool.centre_from_track(run_dir, when) if run_dir else None
    return tool.measure(path, centre)


@pytest.mark.parametrize("which", sorted(PUBLISHED))
def test_the_instrument_reproduces_its_published_row(tool, which):
    got = measure(tool, which)
    assert got is not None, f"{which}: the annulus came back empty"
    mean, p95, p99, cv, banded = PUBLISHED[which]

    for name, want, have in (("mean", mean, got["mean"]),
                             ("p95", p95, got["p95"]),
                             ("p99", p99, got["p99"]),
                             ("CV", cv, got["cv"])):
        assert abs(have - want) <= REL_TOL * want, (
            f"{which} {name}: {have:.4f} against the published {want:.4f}. "
            "The instrument no longer reproduces the row it was written to "
            "reproduce, so any New Tiedtke number it produces is "
            "ungradeable.")
    assert abs(got["banded"] - banded) <= BANDED_TOL, (
        f"{which} banded-area: {got['banded']:.1f}% against the published "
        f"{banded:.1f}%. This column is the one that is NOT recoverable "
        "from the prose -- it thresholds at twice the annulus mean, and an "
        "absolute cut-off cannot reproduce all three rows.")


def test_banded_area_is_relative_not_absolute(tool):
    """The negative control on the one recovered parameter.

    Any fixed threshold in kg/m2 fails: the value giving Grell-Freitas its
    13.7% gives 9.5% and 3.9% for the other two, against published figures
    of 14.7% and 14.9%.  So this asserts the definition is relative, which
    is what makes the published near-equality meaningful rather than a
    coincidence.
    """
    assert hasattr(tool, "BANDED_MULTIPLE"), (
        "the banded-area threshold is no longer a multiple of the annulus "
        "mean; an absolute threshold cannot reproduce the published rows")
    assert not hasattr(tool, "BANDED_THRESHOLD"), (
        "an absolute threshold has been reintroduced beside the relative "
        "one -- only one of them can be the published metric")

    got = {w: measure(tool, w) for w in ("GF", "WRF", "KF")}
    means = [got[w]["mean"] for w in ("GF", "WRF", "KF")]
    bands = [got[w]["banded"] for w in ("GF", "WRF", "KF")]
    # The whole point of the metric: amplitudes collapse, texture does not.
    assert max(means) / min(means) > 2.0, (
        "the three runs no longer differ 2.3x in mean annulus condensate; "
        "this fixture is not measuring what the published table measured")
    assert max(bands) - min(bands) < 2.0, (
        "banded-area now spreads across the three runs, which is the "
        "signature of an absolute threshold rather than a relative one")


def test_the_centre_never_comes_from_argmin_psfc(tool):
    """The suite's first named trap, asserted rather than remembered.

    Surface pressure is low over high ground; taking a centre from
    argmin(PSFC) once put a hurricane on Jamaica's Blue Mountains at
    804 mb.  For a WOOF-ArWen run the centre comes from track.csv; the stock
    WRF fallback reduces to sea level and masks land.
    """
    import inspect

    src = inspect.getsource(tool)
    assert "track.csv" in src
    fallback = inspect.getsource(tool.centre_from_slp)
    assert "LANDMASK" in fallback, (
        "the sea-level fallback no longer masks land, which is exactly how "
        "the Blue Mountains hurricane happened")
    assert "exp" in fallback, (
        "the fallback no longer reduces to sea level; it is argmin(PSFC) "
        "again under another name")
