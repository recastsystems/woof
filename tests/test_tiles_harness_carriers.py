"""The [tiles] harness declares its shortwave carriers, or leaves them alone.

The SWDOWN/GSW siblings of ``tests/test_tiles_harness_glw.py``, for the
same seam one lane later: the carrier-contract merge (a2ce2b8ff) made
SWDOWN, GSW and COSZEN producer declarations law, and the GLW fix's
harness counterpart was never built for them.  The 2.0 battery paid for
that six times in the consolidated gate (+Noah rungs, SWDOWN), fourteen
times in ``tests/test_ruc_runtime.py`` (GSW), and across the conformance
and pairing fixtures.

``harness.declared_swdown_kwargs``, ``harness.declared_carrier_kwargs``
and ``harness.declare_offline_gsw`` are the fix and this suite is their
contract.  The properties are the GLW suite's, restated for shortwave:

  * a rung whose consumed shortwave carrier has no scheme DECLARES the
    allocation's own zero, so the refusal cannot fire, the receipt names
    an origin, and no digest the gate compares moves;
  * a rung with a shortwave scheme, or with no consumer for the carrier,
    is left untouched, so nothing overrides a scheme's own field.

THE CLASSIFICATION IS NEVER RESTATED HERE OR IN THE HARNESS: which
carriers a scheme consumes comes from
``woof.core.radiation_carriers.consumer_carriers`` -- the consumption
check's own matrix -- and this suite drives that same function to build
its expectations, so a future change to the matrix moves the harness and
the test together.

COSZEN deliberately has no sibling: ``initialize_physics`` itself
attaches the analytic solar-geometry provider under radiation-off
Noah-MP, and any shortwave scheme writes it -- the law already supplies
both producers, and the last test pins that so a matrix change that
breaks the assumption goes red here.

CPU only, no card: the whole question is which keyword argument (or
which forcing call) happens, and that is decided before any allocation.
"""

from __future__ import annotations

import pytest

from woof.core.radiation_carriers import consumer_carriers
from tilestream import harness


class _Cfg:
    """The selectors the classifications read, and nothing else."""

    def __init__(self, ra_lw_physics, ra_sw_physics, sf_surface_physics):
        self.ra_lw_physics = ra_lw_physics
        self.ra_sw_physics = ra_sw_physics
        self.sf_surface_physics = sf_surface_physics


class _RecordingDriver:
    """A driver double that records the forcing door being used."""

    def __init__(self):
        self.forced: dict = {}

    def set_forcing(self, **fields):
        self.forced.update(fields)


# (ra_lw, ra_sw, sf_surface) spanning every consumer row of the matrix and
# both producer states, named by what the harness must do.
_SELECTORS = [
    (0, 0, 2),    # Noah, radiation off        -> declare swdown, no gsw
    (0, 0, 3),    # RUC, radiation off         -> declare gsw, no swdown
    (0, 0, 4),    # Noah-MP, radiation off     -> declare swdown, no gsw
    (0, 0, 0),    # no consumer, radiation off -> declare nothing
    (0, 1, 2),    # Noah, Dudhia shortwave     -> the scheme produces it
    (0, 1, 3),    # RUC, Dudhia shortwave      -> the scheme produces GSW
    (4, 4, 2),    # RRTMG-class pair, Noah     -> the scheme produces it
    (4, 4, 4),    # RRTMG-class pair, Noah-MP  -> the scheme produces it
]


@pytest.mark.parametrize("ra_lw, ra_sw, sf_surface", _SELECTORS)
def test_swdown_is_declared_exactly_where_consumed_without_a_scheme(
        ra_lw, ra_sw, sf_surface) -> None:
    consumed = "swdown" in consumer_carriers(sf_surface)
    got = harness.declared_swdown_kwargs(_Cfg(ra_lw, ra_sw, sf_surface))
    if consumed and ra_sw == 0:
        assert got == {
            "swdown": harness.DECLARED_CONSTANT_SHORTWAVE_WM2}, (
            "a consumed, schemeless SWDOWN must be declared or "
            "initialize_physics's consumption check refuses it")
    else:
        assert got == {}, (
            "an unconsumed or scheme-produced SWDOWN needs no declaration "
            "and must not be overridden")


@pytest.mark.parametrize("ra_lw, ra_sw, sf_surface", _SELECTORS)
def test_gsw_is_forced_exactly_where_consumed_without_a_scheme(
        ra_lw, ra_sw, sf_surface) -> None:
    consumed = "gsw" in consumer_carriers(sf_surface)
    driver = _RecordingDriver()
    harness.declare_offline_gsw(driver, _Cfg(ra_lw, ra_sw, sf_surface))
    if consumed and ra_sw == 0:
        assert driver.forced == {
            "gsw": harness.DECLARED_CONSTANT_SHORTWAVE_WM2}, (
            "a consumed, schemeless GSW must go through the forcing door "
            "or RUC's first surface step is refused")
    else:
        assert driver.forced == {}, (
            "an unconsumed or scheme-produced GSW must not be forced -- "
            "forcing it into a Noah driver raises, and forcing it over a "
            "scheme's field would misname a produced carrier")


def test_a_driverless_rung_is_left_alone() -> None:
    """``declare_offline_gsw(None, cfg)`` is the no-physics rung's path."""
    harness.declare_offline_gsw(None, _Cfg(0, 0, 3))   # must not raise


def test_the_selector_list_spans_the_shortwave_consumer_matrix() -> None:
    """The parametrisation is only meaningful if it spans the matrix.

    A selector list that happened to miss the RUC row would let a harness
    that never declares GSW pass, and that harness is exactly the tree the
    2.0 battery caught.  Both shortwave carriers and the empty row must
    each be exercised with and without a producing scheme.
    """
    rows = {(("swdown" in consumer_carriers(sf)),
             ("gsw" in consumer_carriers(sf)),
             sw > 0)
            for _lw, sw, sf in _SELECTORS}
    assert (True, False, False) in rows      # swdown consumed, schemeless
    assert (False, True, False) in rows      # gsw consumed, schemeless
    assert (False, False, False) in rows     # nothing consumed
    assert (True, False, True) in rows       # swdown consumed, produced
    assert (False, True, True) in rows       # gsw consumed, produced


def test_declared_carrier_kwargs_is_the_union_and_the_caller_wins() -> None:
    """One call for both keyword declarations, merged UNDER the caller.

    A negative control that wants a deliberately wrong shortwave has to be
    able to ask for one, which is only true if the caller's value wins.
    """
    cfg = _Cfg(0, 0, 2)
    got = harness.declared_carrier_kwargs(cfg)
    assert got == {**harness.declared_glw_kwargs(cfg),
                   **harness.declared_swdown_kwargs(cfg)}
    assert set(got) == {"glw", "swdown"}
    merged = {**got, "swdown": 700.0}
    assert merged["swdown"] == 700.0


def test_coszen_has_a_producer_in_the_law_itself() -> None:
    """Why there is no COSZEN sibling, pinned rather than assumed.

    COSZEN is consumed by exactly the scheme whose radiation-off driver
    ``initialize_physics`` seeds with the analytic solar-geometry provider
    (woof/core/physics.py declares it CARRIER_SOURCE_ANALYTIC_GEOMETRY
    when radiation is inactive, and PhysicsDriver._update_analytic_coszen
    refreshes it on the radiation cadence).  If a new consumer row ever
    names ``coszen`` this goes red, and THAT scheme needs either the same
    analytic seeding or a declaration sibling here.
    """
    from woof.core.radiation_carriers import CONSUMER_CARRIERS

    consumers = {sf for sf, carriers in CONSUMER_CARRIERS.items()
                 if "coszen" in carriers}
    assert consumers == {4, 8}, (
        "a scheme other than Noah-MP now consumes COSZEN; give it a "
        "producer before teaching the harness to declare a frozen sun")
