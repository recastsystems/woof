# SPDX-License-Identifier: Apache-2.0
"""global-receipt.json says the physics mode the forecast stage ran, on both routes. No card, no network.

THE BREAKAGE: a correct-mode `woof hex global` on an MPAS suite wrote `"ran": {"mode": "exact"}` beside
`"planned": {"mode": "correct"}`, while its own forecast receipt said `physics_mode` correct and the terminal printed
the corrections it applied. The door read only the WOOF-ArWen route's key.
"""
from __future__ import annotations

import json

from woof.hex import global_door


def _receipt(tmp_path, record):
    path = tmp_path / "native-physics-receipt.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_an_mpas_suite_record_in_correct_mode_reads_correct(tmp_path):
    path = _receipt(tmp_path, {"schema": "mpas-native-forecast-v1",
                               "physics": {"suite": "convection_permitting", "physics_mode": "correct"}})
    assert global_door._ran_physics(path) == {"suite": "convection_permitting", "mode": "correct"}


def test_an_exact_mpas_suite_record_reads_exact(tmp_path):
    path = _receipt(tmp_path, {"physics": {"suite": "convection_permitting"}})
    assert global_door._ran_physics(path) == {"suite": "convection_permitting", "mode": "exact"}


def test_the_arwen_door_receipt_still_reads_its_own_key(tmp_path):
    path = _receipt(tmp_path, {"configuration": {"physics_mode": {"physics.mode": "correct"}}})
    assert global_door._ran_physics(path) == {"suite": "arwen", "mode": "correct"}
    assert global_door._ran_physics(tmp_path / "missing.json") is None
