"""Host mirrors follow real tracker/carrier consumption, including fractional dt."""
from woof.hex.mpasphys.moving_v842 import MovingRoute


def test_only_tracker_boundaries_and_final_carrier_need_host_mirrors():
    route = MovingRoute(control=object(), frame_seconds=900.)
    due = [s for s in range(98) if route.host_publication_required(s, steps=97, dt=18.75)]
    assert due == [0, 48, 96, 97]
    route.shock = True
    assert [s for s in range(98) if route.host_publication_required(s, steps=97, dt=18.75)] == due


def test_resumed_clock_skips_duplicate_start_but_keeps_next_frame():
    route = MovingRoute(control=object(), frame_seconds=900., offset_seconds=900., carrier={})
    assert not route.host_publication_required(0, steps=96, dt=18.75)
    assert route.host_publication_required(48, steps=96, dt=18.75)
    assert route.host_publication_required(96, steps=96, dt=18.75)
    route.control = None
    assert not route.host_publication_required(48, steps=96, dt=18.75)
    assert route.host_publication_required(96, steps=96, dt=18.75)
