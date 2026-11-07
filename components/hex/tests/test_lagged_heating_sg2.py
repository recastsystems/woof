"""SG2: MPAS-A's lagged microphysics heating on the WOOF column path.

The producer must reproduce native mpas_atmphys_interface.F:909-919 (the
modified-theta change across the microphysics call over dt), not the engine's
dry-theta ``h_diabatic``.  The fixtures are the audit's heating-contract cases
(the 2026-10-13 physics-coupling audit's heating contract), whose native values were computed by compiling the
unchanged Fortran assignments.

Before SG2 the WOOF adapter produced no carry at all (``h_diabatic`` was
"explicitly declined"), and the only rate on offer was the engine's dry one;
``test_engine_dry_rate_is_not_the_native_carry`` is the failure that motivated
the producer, and ``test_producer_matches_native_assignment`` passes only with
it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from woof.hex.lagged_heating_v841 import (  # noqa: E402
    LAGGED_HEATING_FINDING,
    LAGGED_HEATING_SOURCE,
    lagged_theta_m_rate_reference,
)

F32 = np.float32
RVORD = F32(F32(461.6) / F32(287.0))  # woof.hex.cuda_physics_v841.RV_OVER_RD_F32
DT = 18.75

# (theta_before, qv_before, theta_after, qv_after, native carry K/s at RKIND=8)
# from HEATING-CONTRACT.md / evidence/heating-contract-verification.json.
CONTRACT_CASES = [
    (300.0, 0.0100, 298.0, 0.0108, -0.087932462),   # condensing column
    (300.0, 0.0120, 302.0, 0.0112, +0.088001085),   # evaporating column
]


def native_rt_diabatic_tend_f32(theta_before, qv_before, theta_after, qv_after, dt):
    """Native lines 909-919 transcribed at RKIND=4, operation for operation."""

    theta_m_old = F32(F32(theta_before) * F32(F32(1.0) + F32(RVORD * F32(qv_before))))
    theta_m_new = F32(F32(theta_after) * F32(F32(1.0) + F32(RVORD * F32(qv_after))))
    return F32(F32(theta_m_new - theta_m_old) / F32(dt))


def engine_dry_h_diabatic(theta_before, theta_after, dt):
    """gpuwm 2.8.7 microphysics.moist_physics_finish: dry-theta change / dt."""

    return F32(F32(F32(theta_after) - F32(theta_before)) / F32(dt))


@pytest.mark.parametrize("case", CONTRACT_CASES)
def test_producer_matches_native_assignment(case):
    th0, qv0, th1, qv1, native64 = case
    theta_m_before = np.array([[F32(th0) * F32(F32(1.0) + F32(RVORD * F32(qv0)))]], dtype=F32)
    rate = lagged_theta_m_rate_reference(
        theta_m_before, np.array([[th1]], F32), np.array([[qv1]], F32), DT, RVORD)
    expected32 = native_rt_diabatic_tend_f32(th0, qv0, th1, qv1, DT)
    assert rate.dtype == F32
    assert rate[0, 0] == expected32, (rate[0, 0], expected32)
    # and within binary32 resolution of the native double-precision value
    assert abs(float(rate[0, 0]) - native64) < 2e-5


@pytest.mark.parametrize("case", CONTRACT_CASES)
def test_engine_dry_rate_is_not_the_native_carry(case):
    """Why the adapter cannot replay h_diabatic: a 21 % error on the contract."""

    th0, qv0, th1, qv1, native64 = case
    dry = float(engine_dry_h_diabatic(th0, th1, DT))
    assert abs(dry - native64) / abs(native64) > 0.15


def test_vapour_only_change_still_carries():
    """Dry theta unchanged, vapour changed: native carries a nonzero rate."""

    th, qv0, qv1 = 290.0, 0.008, 0.009
    theta_m_before = np.array([[F32(th) * F32(F32(1.0) + F32(RVORD * F32(qv0)))]], dtype=F32)
    rate = lagged_theta_m_rate_reference(theta_m_before, np.array([[th]], F32),
                                         np.array([[qv1]], F32), DT, RVORD)
    assert float(engine_dry_h_diabatic(th, th, DT)) == 0.0
    assert rate[0, 0] == native_rt_diabatic_tend_f32(th, qv0, th, qv1, DT)
    assert rate[0, 0] > 0


def test_negative_vapour_couples_as_the_recovery_does():
    """max(qv,+0) by sign bit, as recover_wsm6_v841_f32 commits theta_m."""

    before = np.array([[300.0]], F32)
    for qv in (F32(-1e-9), F32(-0.0)):
        rate = lagged_theta_m_rate_reference(before, np.array([[300.0]], F32),
                                             np.array([[qv]], F32), DT, RVORD)
        assert rate[0, 0] == F32(0.0)


def test_carry_endpoint_is_neutral_on_a_toy_column():
    """Add rho*R in each RK stage, subtract dt*rho*R at the final recovery
    (mpas_atm_time_integration.F:6937, :4436-4437): with no acoustic response
    the carry leaves the endpoint where the microphysics put it."""

    rho = F32(1.1)
    rate = F32(-0.02)
    rtheta_p_save = F32(5.0)
    for dt in (F32(18.75), F32(150.0)):
        # stage 3 integrates the full dt: rtheta_pp accumulates dt*tend
        rtheta_pp = F32(dt * F32(rho * rate))
        carried = F32(F32(rtheta_p_save + rtheta_pp) - F32(F32(dt * rho) * rate))
        assert carried == rtheta_p_save


def test_kernel_source_names_the_finding_and_has_no_contraction_hazard():
    assert LAGGED_HEATING_FINDING == "SG2"
    assert "lagged_theta_m_rate_v842_f32" in LAGGED_HEATING_SOURCE
    assert "mpas_div" in LAGGED_HEATING_SOURCE and "mpas_mul" in LAGGED_HEATING_SOURCE


def test_held_tendencies_attach_the_backend_carry():
    from woof.hex.cuda_physics_v841 import (
        CudaHeldMpasPhysicsTendenciesV841,
        with_lagged_diabatic_theta_v841,
    )

    held = CudaHeldMpasPhysicsTendenciesV841.__new__(CudaHeldMpasPhysicsTendenciesV841)
    for name in ("rho", "rho_u", "rho_theta", "scalars", "validation_d2h",
                 "execution_provenance"):
        object.__setattr__(held, name, None)
    object.__setattr__(held, "time_seconds", 0.0)
    object.__setattr__(held, "diabatic_theta", None)

    @dataclass
    class NoCarry:
        lagged_diabatic_theta: object = None

    assert with_lagged_diabatic_theta_v841(held, NoCarry()) is held
    assert with_lagged_diabatic_theta_v841(held, object()) is held
    plane = np.zeros((2, 3), F32)
    attached = with_lagged_diabatic_theta_v841(held, NoCarry(plane))
    assert attached is not held and attached.diabatic_theta is plane
    assert held.diabatic_theta is None


def test_lts_recover_wrappers_forward_the_carry():
    """The LTS executors wrap driver._recover_candidate; a wrapper with the old
    six-argument signature raises TypeError at the first physics stage."""

    root = Path(__file__).resolve().parents[1] / ".." / ".." / "woof" / "hex"
    for name in ("cuda_fullstep_lts_active.py", "cuda_fullstep_lts_multi.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "**lagged_heating" in text, name
    text = (root / "cuda_driver_lts.py").read_text(encoding="utf-8")
    assert "**lagged_heating" in text
