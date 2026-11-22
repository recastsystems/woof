"""The global-vr door writes standard JSON receipts, uniform globes included.

THE DEFECT THIS PINS (hex 0.4.14, 2026-11-07). A uniform global spec (``"regions": []``) has a steepest gradient of
0 %/cell, so its transition band is ``ln 2 / ln 1 = inf`` cells, and ``json.dumps`` wrote the bare token
``Infinity`` into ``global-vr-receipt.json`` (``sizing.gates_applied_by_hexcore.transition_band.band_cells``).
That is not JSON: JavaScript's ``JSON.parse``, jq and the WOOF site's customer copy of a run's receipt refuse the
whole file, so a uniform 15 km run (run_c20529382a29ebc8) showed its customer "This file is not available".
Real command, published 0.4.14 wheel vs the fix: ``woof hex mesh-plan --global-mesh --spec uniform15.json
--card 96gib-188sm --cards 8 --hours 6 --out R.json``: 0.4.14 writes ``"band_cells": Infinity``; the fix writes
``null`` and nothing else in the receipt changes; a variable-resolution zoom receipt is identical.
"""

from __future__ import annotations

import json
import math

import pytest

from woof.hex import mesh_plan_door
from woof.hex.mesh_spec_gates import transition_band_cells


def _strict(text: str):
    def refuse(token):
        raise ValueError(f"non-standard JSON token {token}")
    return json.loads(text, parse_constant=refuse)


def test_uniform_globe_transition_band_is_written_null_not_infinity():
    band = transition_band_cells(0.0)  # a uniform globe: no refinement, no gradient
    assert math.isinf(band)
    receipt = {"sizing": {"gates_applied_by_hexcore": {"transition_band": {"band_cells": band, "band_cells_floor": 6.0}}}}
    with pytest.raises(ValueError):  # what 0.4.14 wrote
        _strict(json.dumps(receipt, indent=2, sort_keys=True))
    written = mesh_plan_door.receipt_json(receipt, indent=2, sort_keys=True)
    parsed = _strict(written)
    assert parsed["sizing"]["gates_applied_by_hexcore"]["transition_band"] == {"band_cells": None, "band_cells_floor": 6.0}
    assert receipt["sizing"]["gates_applied_by_hexcore"]["transition_band"]["band_cells"] == band  # in memory: kept


@pytest.mark.parametrize("value", (float("inf"), float("-inf"), float("nan")))
def test_every_non_finite_number_is_null_in_lists_and_tuples_too(value):
    written = mesh_plan_door.receipt_json({"a": [1.5, value, (value, 2)], "b": {"c": value}}, sort_keys=True)
    assert _strict(written) == {"a": [1.5, None, [None, 2]], "b": {"c": None}}


def test_a_finite_receipt_is_byte_identical_to_plain_json():
    receipt = {"cells": 348149, "dt_seconds": 18.0, "x": [0.1, 1e-300, 2.5e30], "s": "Infinity is a word here",
               "nested": {"ok": True, "none": None}}
    for kwargs in ({"indent": 2, "sort_keys": True}, {"indent": 2, "sort_keys": True, "default": str}):
        assert mesh_plan_door.receipt_json(receipt, **kwargs) == json.dumps(receipt, **kwargs)
