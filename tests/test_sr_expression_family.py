"""The SR validator's shared-expression roundoff family (1.9.1 D2).

Shipped 1.9.0 granted the proven WRF positive-sum SR roundoff envelope to
``mp_physics=6`` alone, while wdm6.cu:672 forms SR with the IDENTICAL
expression as wsm6.cu:406 -- so a healthy WDM6 real case died on its own
validator at the first frozen-dominated column (accepted_updates=5 on the
reference case), reporting WRF expression-order behaviour as a defect.

The fix keys the envelope on SHARED-EXPRESSION FAMILY membership
(woof/core/physics.py::_SR_EXPRESSION_FAMILY_STEP_SCALINGS and
``_sr_roundoff_envelope``), audited over every kernel that writes SR:
wsm6/wdm6 (identical accumulated expression), milbrandt2 (same quotient
once per call), morrison (order-dominated, provably <= 1.0, excluded),
NSSL (outside the range check by contract, excluded).

This suite pins the family's membership and the 1-ULP-above-1.0 case in
both directions: red under a zero-ULP envelope (the shipped 1.9.0
behaviour), green under the family envelope.  The real-case route proof
is the reference-case twin run, which must survive past
``accepted_updates=5`` where 1.9.0 died.
"""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest


def _cpu_diagnostics_driver(monkeypatch, mp_physics, *, dt=60.0,
                            envelope=None):
    """CPU mirror of the driver's diagnostic handoff for one scheme."""
    import woof.core.physics as physics
    from woof.core.microphysics import MicrophysicsDiagnostics

    monkeypatch.setattr(physics, "cp", np)
    shape = (2, 3)
    zeros = lambda: np.zeros(shape, np.float32)
    driver = object.__new__(physics.PhysicsDriver)
    driver.state = SimpleNamespace(mup=zeros())
    driver.mp_physics = mp_physics
    # The exact assignment PhysicsDriver.__init__ performs.
    (driver._sr_roundoff_upper, driver._sr_roundoff_max_ulps,
     driver._wsm6_minor_loops) = (
        physics._sr_roundoff_envelope(mp_physics, dt)
        if envelope is None else envelope)
    hail = mp_physics in (9, 18, 24, 26)
    driver.microphysics = MicrophysicsDiagnostics(
        rainnc=zeros(), rainncv=zeros(), sr=zeros(),
        snownc=zeros(), snowncv=zeros(),
        graupelnc=zeros(), graupelncv=zeros(),
        hailnc=zeros() if hail else None,
        hailncv=zeros() if hail else None)
    driver.microphysics_updates = 0
    driver._pending_rainbl = zeros()
    driver.ruc_params = None
    driver.noahmp_params = None
    return physics, driver, MicrophysicsDiagnostics, shape


def _accept(driver, diagnostics, shape, sr_value):
    driver.accept_microphysics(diagnostics(
        rainnc=np.ones(shape, np.float32),
        rainncv=np.ones(shape, np.float32),
        sr=np.full(shape, sr_value, np.float32)))


ONE_ULP_ABOVE_ONE = np.nextafter(
    np.float32(1.0), np.float32(np.inf), dtype=np.float32)


def test_family_membership_is_the_kernel_audit():
    """The family is exactly the audited shared-expression set."""
    from woof.core.physics import (_SR_EXPRESSION_FAMILY_STEP_SCALINGS,
                                    _sr_roundoff_envelope)

    # wsm6.cu:519 and wdm6.cu:754: identical expression, identical
    # floor(dt/120+0.5) minor-loop split, and both scale every
    # accumulation step as WRF does, fallsum*delz/denr*dtcld*1000
    # (four roundings per side: wsm6.cu:501-515, wdm6.cu:737-751).
    assert _SR_EXPRESSION_FAMILY_STEP_SCALINGS == {6: 4, 16: 4}
    # milbrandt2.cu:2340: the same positive-sum quotient once per call.
    assert _sr_roundoff_envelope(9, 60.0)[1] > 0
    # Morrison (the WRF v4.6.1 transcription) sums signed fallout
    # components in the raw oracle. Forecast sedimentation is nonnegative; morrison.bound_snow_ratio
    # holds the forecast SR to [0, 1] before this check, and Morrison
    # keeps the tight range.  NSSL sits outside the range check.
    for outsider in (0, 1, 8, 10, 18, 28, 50):
        assert _sr_roundoff_envelope(outsider, 60.0) == (
            np.float32(1.0), 0, 0)


@pytest.mark.parametrize("mp_physics", [16, 9, 24, 26])
def test_one_ulp_above_one_red_at_zero_ulp_green_at_family_envelope(
        monkeypatch, mp_physics):
    """The pinned defect case, in both directions.

    ``SR = 1.0 + 1 ULP`` is WRF expression-order output for every family
    member.  Under the zero-ULP envelope 1.9.0 shipped for these schemes
    the validator kills the run (red half); under the family envelope it
    is accepted unmutated (green half).
    """
    # Red: the shipped 1.9.0 grant (upper=1.0, roundoff_ulps=0).
    physics, driver, diagnostics, shape = _cpu_diagnostics_driver(
        monkeypatch, mp_physics,
        envelope=(np.float32(1.0), 0, 0))
    with pytest.raises(ValueError, match="roundoff_ulps=0"):
        _accept(driver, diagnostics, shape, ONE_ULP_ABOVE_ONE)

    # Green: the family envelope, and the bit pattern is not clipped.
    physics, driver, diagnostics, shape = _cpu_diagnostics_driver(
        monkeypatch, mp_physics)
    assert driver._sr_roundoff_max_ulps >= 1
    _accept(driver, diagnostics, shape, ONE_ULP_ABOVE_ONE)
    assert driver.microphysics.sr.view(np.uint32)[0, 0] == np.asarray(
        ONE_ULP_ABOVE_ONE).view(np.uint32).item()

    # The envelope is still a bound, not an amnesty: one ULP beyond it
    # is refused, and a negative SR is refused.
    upper_bits = np.asarray(
        driver._sr_roundoff_upper, np.float32).view(np.uint32).item()
    beyond = np.asarray(upper_bits + 1, np.uint32).view(np.float32)
    with pytest.raises(ValueError, match="validated range"):
        _accept(driver, diagnostics, shape, beyond)
    with pytest.raises(ValueError, match="validated range"):
        _accept(driver, diagnostics, shape, -np.float32(1.0e-7))


def test_wdm6_envelope_is_the_wsm6_bound_with_step_scalings():
    """WDM6's envelope is the exact analytic widening, not a guess.

    B16 = B6 * ((1+u)/(1-u))^4 for the four post-sum unit scalings per
    step (wdm6.cu:655-667), evaluated by the same exact integer
    floor((B-1)/ULP(1)) selection as the WSM6 bound.
    """
    from woof.core.physics import (_sr_roundoff_envelope,
                                    _wsm6_sr_roundoff_limit)

    scale = 1 << 24
    for dt in (60.0, 240.0, 600.0):
        upper6, ulps6, loops6 = _wsm6_sr_roundoff_limit(dt)
        upper16, ulps16, loops16 = _sr_roundoff_envelope(16, dt)
        assert loops16 == loops6
        assert ulps16 > ulps6
        # WSM6 forms the same scaled accumulation (2.9.0 WRF rewrite).
        assert _sr_roundoff_envelope(6, dt) == (upper16, ulps16, loops16)
        adds = loops6 - 1
        analytic = Fraction(
            (scale + 1) ** 7 * (scale - 3),
            scale ** 2 * (scale - 6) * (scale - 2 * adds)
            * (scale - 1) ** 4)
        encoded = Fraction(scale + 2 * ulps16, scale)
        next_encoded = Fraction(scale + 2 * (ulps16 + 1), scale)
        assert encoded <= analytic < next_encoded
    # The zero-scaling bound (a kernel accumulating pre-scaled returns,
    # WSM6 before 2.9.0) stays available and is strictly tighter.
    assert _wsm6_sr_roundoff_limit(60.0)[1] == 3
    assert _sr_roundoff_envelope(6, 60.0)[1] == 7


def test_wsm6_default_forecast_sr_four_ulps_over_one_is_accepted(
        monkeypatch):
    """The 2.9.0 field case: a default hex forecast (WSM6, dt=90 s, one
    minor loop) produced SR=1.00000048, 4 ULPs over 1, and the 3-ULP
    zero-scaling grant killed it at 4590 s.  The scaled-accumulation
    bound admits it unmutated and still refuses one ULP past itself.
    """
    field_value = np.float32(1.00000048)
    assert np.asarray(field_value).view(np.uint32).item() - 0x3F800000 == 4
    physics, driver, diagnostics, shape = _cpu_diagnostics_driver(
        monkeypatch, 6, dt=90.0)
    assert driver._sr_roundoff_max_ulps == 7
    _accept(driver, diagnostics, shape, field_value)
    assert driver.microphysics.sr.view(np.uint32)[0, 0] == np.asarray(
        field_value).view(np.uint32).item()
    upper_bits = np.asarray(
        driver._sr_roundoff_upper, np.float32).view(np.uint32).item()
    beyond = np.asarray(upper_bits + 1, np.uint32).view(np.float32)
    with pytest.raises(ValueError, match="roundoff_ulps=7"):
        _accept(driver, diagnostics, shape, beyond)


def test_my2_envelope_matches_its_analytic_bound():
    """B9 = (1+u)^5/(1-u)^9, dt-independent (no minor loops)."""
    from woof.core.physics import _sr_roundoff_envelope

    scale = 1 << 24
    upper, ulps, loops = _sr_roundoff_envelope(9, 60.0)
    assert loops == 0
    assert _sr_roundoff_envelope(9, 3600.0) == (upper, ulps, loops)
    analytic = Fraction((scale + 1) ** 5 * scale ** 4, (scale - 1) ** 9)
    encoded = Fraction(scale + 2 * ulps, scale)
    next_encoded = Fraction(scale + 2 * (ulps + 1), scale)
    assert encoded <= analytic < next_encoded


def test_wdm7_hail_envelope_matches_its_expression_bound():
    """Price the extra hail term, including its final numerator addition."""
    from woof.core.physics import (_SR_HAIL_EXPRESSION_FAMILY,
                                    _sr_roundoff_envelope)

    assert _SR_HAIL_EXPRESSION_FAMILY == frozenset((24, 26))
    u = Fraction(1, 1 << 24)
    ulp = 2 * u
    for dt, loops in ((60.0, 1), (240.0, 2), (600.0, 5)):
        gamma_total = 4 * u / (1 - 4 * u)  # Five nonnegative fallout terms.
        gamma_acc = (loops - 1) * u / (1 - (loops - 1) * u)
        # Snow pair, two final frozen-sum additions, final division;
        # four scalings per accumulated term on each side; L-1 adds.
        bound = ((1 + u) ** 4 * ((1 + u) / (1 - u)) ** 4
                 * (1 + gamma_acc) / ((1 - gamma_total) * (1 - gamma_acc)))
        upper, ulps, actual_loops = _sr_roundoff_envelope(26, dt)
        assert actual_loops == loops
        assert 1 + ulps * ulp <= bound < 1 + (ulps + 1) * ulp
        assert np.asarray(upper).view(np.uint32).item() == 0x3f800000 + ulps
        assert _sr_roundoff_envelope(24, dt) == (upper, ulps, actual_loops)
    assert _sr_roundoff_envelope(26, 60.0)[1] == 8
    assert _sr_roundoff_envelope(16, 60.0)[1] == 7


def test_wdm7_hail_envelope_endpoints_are_not_clipped(monkeypatch):
    """Accept the exact bound unchanged and reject the next FP32 word."""
    physics, driver, diagnostics, shape = _cpu_diagnostics_driver(
        monkeypatch, 26)
    upper = driver._sr_roundoff_upper
    _accept(driver, diagnostics, shape, upper)
    assert np.all(driver.microphysics.sr.view(np.uint32) == 0x3f800008)
    beyond = np.nextafter(upper, np.float32(np.inf), dtype=np.float32)
    with pytest.raises(ValueError, match="validated range"):
        _accept(driver, diagnostics, shape, beyond)
