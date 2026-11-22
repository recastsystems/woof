# SPDX-License-Identifier: Apache-2.0
"""A 0.6.2 plan's rule kinds are checked against the runner that will run its windows (coverage.RUNNER_RULE_KERNELS)."""
import json

import pytest

from woof.hex import ad_doors
from woof.hex.ad.record import coverage


def test_a_runner_without_the_damping_rules_is_named(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.write_bytes(b"\x7fELF ... rule:accumulate ...")
    new.write_bytes(b"\x7fELF ... win_upper_cam ... win_upper_rayleigh ...")
    kinds = ["cupy_any", "mpas_mcica_seeds_column", "upper_cam_f32", "upper_rayleigh_f32"]
    assert coverage.runner_lacks(kinds, old) == ["upper_cam_f32", "upper_rayleigh_f32"]
    assert coverage.runner_lacks(kinds, new) == []
    assert coverage.runner_lacks(kinds, None) == []
    assert coverage.runner_lacks(["cupy_any"], old) == []


def test_the_door_refuses_the_window_before_the_card(tmp_path):
    plan = tmp_path / "plan"; plan.mkdir()
    old = tmp_path / "ad-binder"; old.write_bytes(b"\x7fELF no damping rules")
    (plan / "derivative-coverage.json").write_text(json.dumps({"rule_kinds": ["upper_cam_f32", "cupy_any"]}))
    with pytest.raises(ad_doors.AdDoorError, match="upper_cam_f32"):
        ad_doors.refuse_runner_without_plan_rules(old, plan)
    (plan / "derivative-coverage.json").write_text(json.dumps({"kinds": 250}))   # sealed before rule_kinds existed
    ad_doors.refuse_runner_without_plan_rules(old, plan)
