"""CPU coverage of allocator policy, boundary calls, and forecast defaults."""

import ast
from datetime import datetime
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from woof.config import RunConfig
from woof.core import model as model_mod
from woof.core.clock import build_schedule, resolve_clock
from woof.experiment import DomainConfig, ExperimentConfig, VerticalConfig


def _execute_with_policy(monkeypatch, platform, **options):
    monkeypatch.setitem(sys.modules, "woof.core.physics", SimpleNamespace(
        _physics_interval_steps=lambda minutes, dt: 1))
    cfg = RunConfig(nx=16, ny=12, nz=4, dx=500., dy=500., ztop=8000.,
                    dt=1., run_seconds=2., moist=False, mp_physics=0)
    domain = DomainConfig(grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1, history_interval_s=1.,
        run=cfg, time_step=1)
    exp = ExperimentConfig(name="pool-policy", start_time=datetime(2026, 1, 1),
        run_seconds=2., vertical=VerticalConfig(eta_levels=(), p_top=0., hybrid_opt=0, etac=.2),
        projection=None, restart_interval_s=0., domains=(domain,))
    calendar = resolve_clock(exp, lbc_interval_s=1.)
    node = model_mod.DomainNode(domain, None, SimpleNamespace(elapsed_seconds=0.),
                               calendar.clocks()[1], None, [], None)
    model = model_mod.ExperimentState(node, {1: node}, build_schedule(exp, calendar), None, "fixture")
    model._runtime_status = model_mod.ModelRuntimeStatus()
    model._resumed = False
    model._scratch_arena = model._dycore_state_workspace = model._io_manager = None
    model._last_checkpoint = None
    trim = mock.Mock()
    monkeypatch.setitem(sys.modules, "woof.core.dycore", SimpleNamespace(step=lambda *a, **k: None))
    monkeypatch.setattr(model_mod, "_trim_default_pool", trim)
    monkeypatch.setattr(sys, "platform", platform)
    execution = model_mod.execute_experiment(model, validate_state=False, **options)
    assert execution.steps == 2
    return trim, model._pool_trim_policy


@pytest.mark.parametrize("platform, expected", [("linux", 0), ("win32", 4), ("darwin", 0)])
def test_default_pool_policy_at_step_and_period_boundaries(monkeypatch, platform, expected):
    trim, receipt = _execute_with_policy(monkeypatch, platform)
    assert trim.call_count == expected
    assert receipt["platform"] == platform
    assert receipt["release_unused_blocks"] is bool(expected)
    assert receipt["selection"] == "platform-default"
    assert receipt["reason"]
    assert "32%" in receipt["windows_measurement"]


@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
@pytest.mark.parametrize("requested", [False, True])
def test_allocator_override_is_executed_and_recorded(monkeypatch, platform, requested):
    trim, receipt = _execute_with_policy(monkeypatch, platform,
                                         pool_trim_per_period=requested)
    assert trim.call_count == (4 if requested else 0)
    assert receipt["release_unused_blocks"] is requested
    assert receipt["selection"] == "explicit-override"


@pytest.mark.parametrize("relative", [
    "woof/runtime.py", "woof/prepared_domain_tree_forecast.py",
    "woof/prepared_single_domain_forecast.py",
    "tools/hrrr_single_domain_benchmark.py", "tools/hrrr_two_domain_forecast.py",
])
def test_forecast_routes_use_platform_default(relative):
    tree = ast.parse((Path(__file__).parents[1] / relative).read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "execute_experiment"]
    assert calls
    assert all(keyword.arg != "pool_trim_per_period"
               for call in calls for keyword in call.keywords)
