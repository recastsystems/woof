"""The admitted MYNN/MYNN/Noah surface trajectory stays byte-identical.

RE-PINNED 2026-09-17 (the 2.7.6 verification change) ON ONE CARD.  The fixture was
captured on 2026-07-30 (f122ee4ec) as one change's before/after harness and
nothing ran it before a cut; on test host 1's RTX 4090 (compute capability 8.9,
driver 610.57.04) at fc639c51f its reading is: field_inventory_sha256 HELD
(9ef24b8d417eb5b6a3858d9f985572c3a5e931d812ab9eb6cb0cd05903d45303, the same
fields in the same shapes and dtypes), sha256 MOVED (13ed8960c71c8a1957d4e6
7876e2dcd74d2b34eb6b20be3a7e94a447c5f21c8f to cabb0b09944ba28e82351d6f67d7a
35b11c4dbb7d0dba83513f6b0028d4887ef).  The trajectory steps the dycore, and
every stepped run's bits moved by construction at the EOS spelling
6b11e4c99 (2026-09-03, its message carries the readings) and again at the
Omega column kernel fc639c51f (readings under tests/data/receipts/
omega-column-scan/), and the phase-2 step pin shows the same builders give
different bits on an RTX 5070 Ti and an RTX 4090 (25 of 27 entries,
tests/data/receipts/pin-gates/), so a whole-trajectory digest is a property
of the card as well.  The fixture therefore records the card it describes,
this test skips with the reason on any other card, and a digest that moves
on the recorded card means a change to this trajectory shipped without a
reading: the fixer records one and re-pins by running
tools/mynn_noah_surface_identity.py on that card.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import requires_gpu

FIXTURE = Path(__file__).with_name("fixtures") / "mynn_noah_surface_identity_bf45e88a.json"


@requires_gpu
def test_mynn_noah_surface_trajectory_matches_the_pre_pairing_baseline():
    import _card_pins
    from tools.mynn_noah_surface_identity import run_identity

    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    card = expected.pop("card")
    expected.pop("captured_at_commit", None)
    device = _card_pins.device_name()
    if device != card:
        pytest.skip(f"the MYNN/Noah surface digest is per card and was captured on "
                    f"{card!r}; this is {device!r} (compute capability "
                    f"{_card_pins.device_compute_capability()}).  Re-pin on this "
                    "card with tools/mynn_noah_surface_identity.py and record the "
                    "reading in this file's docstring.")
    got = run_identity()
    assert got["field_inventory_sha256"] == expected["field_inventory_sha256"], (
        "the field inventory of the MYNN/Noah trajectory changed: a field was "
        "added, dropped or reshaped; record which and re-pin")
    assert got == expected, (
        "the MYNN/Noah surface trajectory moved on the card that captured it, "
        "so a change to this path is shipping without a reading; record one "
        "and re-pin with tools/mynn_noah_surface_identity.py on this card")
