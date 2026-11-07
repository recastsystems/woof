"""Certified Noah profiles must remain byte-identical to their capture.

RE-PINNED 2026-09-17 (the 2.7.6 verification change) ON ONE CARD.  The fixture was
captured on 2026-07-30 (d63e2e2fc) against the certified v1.1.2 commit as
one change's before/after harness and nothing ran it before a cut; on
test host 1's RTX 4090 (compute capability 8.9, driver 610.57.04) at fc639c51f
its reading is: field_inventory_sha256 MOVED for all four profiles
(77a004176d274621b918e5f083b67a4c1a819d9cba605797d4b419ef099ae9a9 to
217f1bd3fadfab1669ee802cdbe4bd4202bf329014b971e2ccfc16ccf4db568b: the
state the harness hashes gained fields since July), and sha256 MOVED for
all four (wsm6_dudhia f4fafb37 to 590e09e8, thompson_dudhia to e7a7d684,
morrison_rte to 5ce45b7d, nssl2_rte to 1a6e4d09).  The trajectories step
the dycore, and every stepped run's bits moved by construction at the EOS
spelling 6b11e4c99 (2026-09-03, readings in its message) and again at the
Omega column kernel fc639c51f (readings under tests/data/receipts/
omega-column-scan/), and the phase-2 step pin shows the same builders give
different bits on an RTX 5070 Ti and an RTX 4090 (25 of 27 entries,
tests/data/receipts/pin-gates/), so a whole-trajectory digest is a property
of the card as well.  The fixture therefore records the card it describes,
this test skips with the reason on any other card, and a digest that moves
on the recorded card means a change to one of these four compositions
shipped without a reading: the fixer records one and re-pins by running
tools/certified_surface_identity.py on that card.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import requires_gpu

FIXTURE = Path(__file__).with_name("fixtures") / "certified_surface_identity_v112.json"


@requires_gpu
def test_four_certified_noah_profiles_match_the_v112_trajectory_bytes():
    import _card_pins
    from tools.certified_surface_identity import run_profiles

    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    card = expected.pop("card")
    expected.pop("captured_at_commit", None)
    device = _card_pins.device_name()
    if device != card:
        pytest.skip(f"the certified Noah profile digests are per card and were "
                    f"captured on {card!r}; this is {device!r} (compute capability "
                    f"{_card_pins.device_compute_capability()}).  Re-pin on this "
                    "card with tools/certified_surface_identity.py and record the "
                    "reading in this file's docstring.")
    got = run_profiles()
    assert set(got) == set(expected), "a certified profile appeared or disappeared"
    for profile in sorted(expected):
        assert got[profile]["field_inventory_sha256"] == expected[profile]["field_inventory_sha256"], (
            f"the field inventory of the {profile} trajectory changed: a field was "
            "added, dropped or reshaped; record which and re-pin")
    assert got == expected, (
        "a certified Noah profile trajectory moved on the card that captured it, "
        "so a change to that composition is shipping without a reading; record "
        "one and re-pin with tools/certified_surface_identity.py on this card")
