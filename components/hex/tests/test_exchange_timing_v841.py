"""WOOF_HEX_EXCHANGE_TIMING reaches every same-host transport (no GPU needed).

Breakage it prevents: the variable was read only by the ArWen-route driver, so on the MPAS-physics route (RRFS and
MPAS-A; rank threads and rank processes) gpu_step_timings stayed empty and the exposed halo stall could not be
measured (MULTICARD/MPAS-A-SCALING.md, 2026-10-28; V8X6000 conus3km receipts, 2026-10-29).
"""

import inspect

import pytest

from woof.hex import partition_process_transport_v841 as process_transport
from woof.hex.mpasphys import samehost_v842
from woof.hex.partition_samehost_transport_v841 import exchange_timing_requested


@pytest.mark.parametrize("value, expected", [(None, False), ("0", False), ("1", True)])
def test_exchange_timing_reader(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("WOOF_HEX_EXCHANGE_TIMING", raising=False)
    else:
        monkeypatch.setenv("WOOF_HEX_EXCHANGE_TIMING", value)
    assert exchange_timing_requested() is expected


def test_exchange_timing_reader_refuses_ambiguous_values(monkeypatch):
    monkeypatch.setenv("WOOF_HEX_EXCHANGE_TIMING", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        exchange_timing_requested()


def test_mpas_physics_rank_threads_pass_the_timer_to_their_transport():
    source = inspect.getsource(samehost_v842._run_rank_threads)
    assert "measure_gpu=exchange_timing_requested()" in source


def test_process_transport_reads_the_timer_and_summarises_per_step():
    source = inspect.getsource(process_transport.ProcessHaloTransport.__init__)
    assert "self.measure_gpu = False" not in source
    assert "exchange_timing_requested()" in source
    rows = {"0": {"rounds": 2, "wait_seconds": 0.5, "receive_seconds": 1.0},
            "1": {"rounds": 49, "wait_seconds": 0.25, "receive_seconds": 2.0},
            "2": {"rounds": 49, "wait_seconds": 0.125, "receive_seconds": 3.0}}
    summary = process_transport._timing_summary(rows)
    assert summary["steps"] == 3 and summary["rounds"] == 100
    assert summary["exposed_wait_seconds"] == pytest.approx(0.875)
    assert summary["median_step_exposed_wait_seconds"] == 0.25
    assert summary["median_step_receive_seconds"] == 2.0
