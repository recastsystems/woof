"""The shared v841 dycore defers its step validation word only inside a forecast collector.

Breakage it prevents (rrfs review 2026-10-29, conus3km 8 cards): the shared CudaDryDycoreDriver, which serves the
RRFS 44c0849 row and MPAS-A v8.4.2, read the step's validation flag with a blocking copy every step, so every
rank's host waited for its card to drain the step before queuing the next one. Inside an MPAS-physics forecast
loop (a deferred_checks_v842 collector is active) the word now goes through defer(); outside one (replays,
referee helpers, the WOOF route) it is still read at once.
"""

from woof.hex.cuda_driver import _forecast_checks_active
from woof.hex.mpasphys.deferred_checks_v842 import DeferredChecks, active, collecting


def test_the_step_read_defers_only_inside_a_forecast_collector():
    assert active() is None
    assert not _forecast_checks_active()
    checks = DeferredChecks(cp=None)
    with collecting(checks):
        assert active() is checks
        assert _forecast_checks_active()
    assert active() is None
    assert not _forecast_checks_active()
