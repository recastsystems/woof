"""A plan the door records from --init holds the kernel forms the derivative assets were made for.

THE BREAKAGE: at 120 km (x1.40962) the door's own --init recording sealed and replayed exactly, and then every door
stopped in its window: "ozone_time_pair: no derivative row and no rule" at step 1 (adjoint-test, tangent) and
"add_pair_inplace_f32: no derivative row and no rule" at step 3 (sensitivity). The published 120 km package holds
neither kernel: it has ozone_time (2 launches, on the radiation list) and add_inplace_f32 (196 launches, a kernel-set
row) where the door's plan had ozone_time_pair (1) and add_pair_inplace_f32 (72) beside add_inplace_f32 (52):
52 + 2 x 72 = 196 (evidence/init120/r6/kernel-kinds-door-vs-pk120e4.txt, 2026-11-11). The published package was
recorded from an older forecast source that has neither form; the door records the installed forecast, so the door
asks it for the plain forms.
"""

from __future__ import annotations

import subprocess
import sys
import types

import numpy as np
import pytest

from woof.hex import ad_doors as d
from woof.hex import cuda_driver, pair_add_form
from woof.hex.cuda_backend import regional_admission
from woof.hex.mpasphys import radshared_v842 as rad

PLAIN = {"WOOF_HEX_PAIR_ADD": "plain", "WOOF_HEX_OZONE_MONTHS": "all"}


def test_every_row_records_with_the_plain_forms():
    rows = d.load_table()["rows"]
    assert rows
    for name, row in rows.items():
        env = row.get("record_env") or {}
        for knob, value in PLAIN.items():
            assert env.get(knob) == value, (name, knob, env)


def test_the_door_hands_the_plain_forms_to_the_recorder(tmp_path, monkeypatch):
    """The recorder child process of record_plan sees both switches (and what the caller's environment set is kept)."""
    row = dict(d.load_table()["rows"]["hrrrv5-any"], mesh="x1.40962", dt_s=450)
    for name in ("grid.nc", "oro.nc", "init.nc"):
        (tmp_path / name).write_bytes(b"x")
    seen = {}

    def run(argv, stdout=None, stderr=None, env=None, **kw):
        seen["argv"], seen["env"] = argv, env
        return types.SimpleNamespace(returncode=7)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv(d.RECORDER_ENV, "recorder --init {init} --out {out}")
    args = types.SimpleNamespace(grid=str(tmp_path / "grid.nc"), oro=str(tmp_path / "oro.nc"), steps=None,
                                 out=str(tmp_path / "out"), init=str(tmp_path / "init.nc"), card="0", cards=None,
                                 runner=None)
    with pytest.raises(d.AdDoorError, match="recording from"):
        d.record_plan(args, tmp_path, row)
    for knob, value in PLAIN.items():
        assert seen["env"][knob] == value
    assert seen["env"]["WOOF_HEX_EDGE_HOIST"] == "0"


def test_the_switches_read_fused_and_pair_by_default_and_refuse_other_words():
    assert pair_add_form.pair_add_fused({}) is True
    assert pair_add_form.pair_add_fused({"WOOF_HEX_PAIR_ADD": "plain"}) is False
    assert rad.ozone_month_pair_form({}) is True
    assert rad.ozone_month_pair_form({"WOOF_HEX_OZONE_MONTHS": "all"}) is False
    with pytest.raises(ValueError, match="WOOF_HEX_PAIR_ADD"):
        pair_add_form.pair_add_fused({"WOOF_HEX_PAIR_ADD": "0"})
    with pytest.raises(ValueError, match="WOOF_HEX_OZONE_MONTHS"):
        rad.ozone_month_pair_form({"WOOF_HEX_OZONE_MONTHS": "12"})


def _driver_class():
    owners = [v for v in vars(cuda_driver).values() if isinstance(v, type) and "_add_pair" in vars(v)]
    assert len(owners) == 1, owners
    return owners[0]


class _Launches:
    """Stands for the integrator: it keeps each launch and does the kernel's arithmetic on the host."""

    def __init__(self):
        self.calls = []

    def _launch(self, name, count, args):
        self.calls.append(name)
        if name == "add_inplace_f32":
            _, _, increment, target = args
            target += increment
        elif name == "add_pair_inplace_f32":
            _, _, first, second, target = args
            target += first
            target += second
        else:
            raise AssertionError(name)


def _sum(monkeypatch, form):
    pair_add_form.install()
    cls = _driver_class()
    if form is None:
        monkeypatch.delenv("WOOF_HEX_PAIR_ADD", raising=False)
    else:
        monkeypatch.setenv("WOOF_HEX_PAIR_ADD", form)
    fake = _Launches()
    fake._add = types.MethodType(cls._add, fake)
    rng = np.random.default_rng(72)
    target, first, second = (rng.standard_normal((3, 5)).astype(np.float32) for _ in range(3))
    cls._add_pair(fake, target, first, second)
    return fake.calls, target


def test_the_plain_pair_add_is_two_ordered_adds_with_the_same_words(monkeypatch):
    fused_calls, fused = _sum(monkeypatch, None)
    plain_calls, plain = _sum(monkeypatch, "plain")
    assert fused_calls == ["add_pair_inplace_f32"]
    assert plain_calls == ["add_inplace_f32", "add_inplace_f32"]
    assert plain.view(np.uint32).tolist() == fused.view(np.uint32).tolist()


class _Provider(dict):
    """A resident provider with twelve months on the host; cp is numpy here (no card)."""

    resident = True
    cp = np

    def __init__(self):
        super().__init__(diag_physics={"o3clim": None}, atm_input={"ozmixm": "the twelve-month pool field"})
        months = np.arange(12 * 4 * 6, dtype=np.float32).reshape(12, 4, 6)
        self.native = types.SimpleNamespace(get_field=lambda pool, name, level=1: months)
        self.session = types.SimpleNamespace(
            layout=types.SimpleNamespace(dimensions={"nCellsSolve": 5}),
            arguments=types.SimpleNamespace(calendar={"julian": 89.5, "current_date": "2023-03-31_12:00:00"}))
        self.transfers = {"initial_upload_bytes": 0}


def test_the_ozone_update_binds_two_months_by_default_and_all_twelve_when_asked(monkeypatch):
    monkeypatch.delenv("WOOF_HEX_MEMBER_MEMORY", raising=False)
    monkeypatch.delenv("WOOF_HEX_OZONE_MONTHS", raising=False)
    pair = rad.bind_native_ozone_pools(_Provider(), None)
    assert "ozone_month_pair" in pair and "atm_input" not in pair      # ozone_slot then launches ozone_time_pair
    assert pair["ozone_month_pair"].shape == (2, 4, 5)
    monkeypatch.setenv("WOOF_HEX_OZONE_MONTHS", "all")
    plain = rad.bind_native_ozone_pools(_Provider(), None)
    assert "ozone_month_pair" not in plain                              # ozone_slot then launches ozone_time
    assert plain["atm_input"] == {"ozmixm": "the twelve-month pool field"}



def test_the_switch_leaves_the_regional_kernel_set_as_minted():
    """cuda_driver.py is digested into the regional kernel set; the switch must not move it (pair_add_form.py)."""
    assert "WOOF_HEX_PAIR_ADD" not in open(cuda_driver.__file__, encoding="utf-8").read()
    assert regional_admission.kernel_set_sha256() == regional_admission.MINTED_KERNEL_SET_SHA256


def test_importing_hexcore_installs_the_plain_form_only_when_asked():
    import os
    code = ("import woof.hex, sys; m = sys.modules.get('woof.hex.cuda_driver'); "
            "print(m is not None and getattr(m.CudaDryDycoreDriver._add_pair, '_hex_pair_add_form', False))")
    for form, want in ((None, "False"), ("fused", "False"), ("plain", "True")):
        env = {k: v for k, v in os.environ.items() if k != "WOOF_HEX_PAIR_ADD"}
        if form is not None:
            env["WOOF_HEX_PAIR_ADD"] = form
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        assert out.stdout.strip() == want, (form, out.stdout, out.stderr)
