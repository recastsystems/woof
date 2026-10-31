"""MPAS-physics route: [hex_nests] looks arm and read reflectivity (parity run group 3, capability 11).

Before 2026-10-24 a --nest-triggers run on this route finished rc 0 with the loop never armed and
nothing refused. These tests run on the CPU tier with numpy standing in for CuPy.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys import nest_look_v842 as look_mod


class FakeTrigger:
    def __init__(self, every=2):
        self.every, self.looks, self.polls = every, [], 0

    def due(self, step):
        return step % self.every == 0

    def look(self, step, arrays):
        self.looks.append((step, arrays["refl10cm"]))

    def poll(self, final=False):
        self.polls += 1


def _fields(nz=3, columns=5):
    return {"qv": np.zeros((nz, columns), np.float32)}


def test_look_step_off_diagnostics_forms_into_scratch_not_the_pool():
    trigger = FakeTrigger()
    look = look_mod.NestLook(trigger, np, owned_cells=5)
    pool = {"refl10cm": np.full((3, 5), 7.0, np.float32)}
    calls = []

    def radar(fields, out, *, device_math_source):
        calls.append(out)
        out["refl10cm"][...] = 42.0

    look.begin(2)
    look.form(_fields(), pool, diagnostics_formed=False, launch_radar=radar, device_math_source="m")
    look.after_step(2)
    assert len(calls) == 1 and calls[0] is not pool
    assert np.all(pool["refl10cm"] == 7.0)          # output pool untouched
    assert trigger.looks[0][0] == 2 and np.all(trigger.looks[0][1] == 42.0)


def test_look_on_a_diagnostics_step_reads_the_pool_and_launches_nothing_extra():
    trigger = FakeTrigger()
    look = look_mod.NestLook(trigger, np, owned_cells=5)
    pool = {"refl10cm": np.full((3, 5), 9.0, np.float32)}
    look.begin(4)
    look.form(_fields(), pool, diagnostics_formed=True,
              launch_radar=lambda *a, **k: pytest.fail("no second radar launch"), device_math_source="m")
    look.after_step(4)
    assert trigger.looks[0][1] is pool["refl10cm"]


def test_off_look_step_forms_nothing_and_still_polls():
    trigger = FakeTrigger()
    look = look_mod.NestLook(trigger, np, owned_cells=5)
    look.begin(3)
    look.form(_fields(), {}, diagnostics_formed=False,
              launch_radar=lambda *a, **k: pytest.fail("no launch off a look step"), device_math_source="m")
    look.after_step(3)
    assert trigger.looks == [] and trigger.polls == 1


def test_due_look_with_no_reflectivity_is_refused_not_silent():
    look = look_mod.NestLook(FakeTrigger(), np, owned_cells=5)
    look.begin(2)
    with pytest.raises(RuntimeError, match="formed no reflectivity"):
        look.after_step(2)


def test_route_sources_name_refl_for_the_cuda_thompson_slots_and_the_dycore_always():
    def config(scheme, backend="hex-cuda"):
        return SimpleNamespace(selectors={"microphysics": scheme}, backends={"microphysics": backend},
                               active=lambda slot: True)
    dycore = ["w", "cape", "updraft_helicity_max"]
    assert look_mod.route_sources(config("mp_thompson"), ported_framework=True) == ["refl10cm", *dycore]
    assert look_mod.route_sources(config("mp_thompson_aerosols"), ported_framework=True) == ["refl10cm", *dycore]
    assert look_mod.route_sources(config("mp_wsm6"), ported_framework=True) == dycore
    assert look_mod.route_sources(config("mp_thompson"), ported_framework=False) == dycore


class SourcesTrigger(FakeTrigger):
    def __init__(self, needed, every=2):
        super().__init__(every)
        self.needed = list(needed)

    def needed_sources(self):
        return self.needed

    def look(self, step, arrays):
        self.looks.append((step, dict(arrays)))


class FakeDycoreSources:
    needed = ("w",)

    def __bool__(self):
        return True

    def form(self, driver):
        return {"w": driver}


def test_a_w_only_look_reads_the_dycore_and_never_asks_the_microphysics_for_reflectivity():
    """run_9948508bd7cfdee3 (2026-10-29): 'wmax >= -1000' was refused after the box was rented."""
    trigger = SourcesTrigger(["w"])
    look = look_mod.NestLook(trigger, np, owned_cells=5, dycore_sources=FakeDycoreSources())
    look.begin(2)
    assert look.look_due and not look.due        # the Thompson slot forms no reflectivity for it
    look.form(_fields(), {}, diagnostics_formed=False,
              launch_radar=lambda *a, **k: pytest.fail("no radar for a w look"), device_math_source="m")
    look.after_step(2, "the-dycore")
    assert trigger.looks == [(2, {"w": "the-dycore"})]


def test_a_refl_and_w_look_reads_both():
    trigger = SourcesTrigger(["refl10cm", "w"])
    look = look_mod.NestLook(trigger, np, owned_cells=5, dycore_sources=FakeDycoreSources())
    look.begin(2)
    assert look.due
    look.form(_fields(), {"refl10cm": np.ones((3, 5), np.float32)}, diagnostics_formed=True,
              launch_radar=None, device_math_source="m")
    look.after_step(2, "the-dycore")
    step, arrays = trigger.looks[0]
    assert set(arrays) == {"refl10cm", "w"} and arrays["w"] == "the-dycore"


def test_arm_without_a_table_is_none_and_multi_rank_is_refused(monkeypatch, tmp_path):
    from woof.hex import nest_triggers
    monkeypatch.delenv(nest_triggers.NEST_TRIGGERS_ENV, raising=False)
    args = SimpleNamespace(nest_triggers=None)
    assert look_mod.arm(args, None, mesh=None, owned_cells=1, start="2025-05-19_12:00:00", steps=1, dt=90.0,
                        out=tmp_path, cp=np, ported_framework=True, multi_rank=False) is None
    args = SimpleNamespace(nest_triggers=str(tmp_path / "t.json"))
    with pytest.raises(nest_triggers.NestTriggerRefusal, match="one card"):
        look_mod.arm(args, None, mesh=None, owned_cells=1, start="2025-05-19_12:00:00", steps=1, dt=90.0,
                     out=tmp_path, cp=np, ported_framework=True, multi_rank=True)


def test_start_record_is_one_writer_for_both_routes(tmp_path):
    import json

    from woof.hex import forecast_start
    from woof.hex.drivers import run_cuda_v841_forecast as arwen
    assert arwen.write_start_record is forecast_start.write_start_record
    grid = tmp_path / "g.nc"
    grid.write_bytes(b"")
    forecast_start.write_start_record(tmp_path, paths={"grid": grid, "static": None, "init": grid},
                                      schedule={"start_time": "2025-05-19_12:00:00", "forecast_hours": 6.0,
                                                "history_every_minutes": 60.0})
    record = json.loads((tmp_path / forecast_start.START_RECORD_NAME).read_text())
    assert record["forecast_hours"] == 6.0 and record["history_every_minutes"] == 60.0
