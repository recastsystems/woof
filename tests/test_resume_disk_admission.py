"""Disk admission prices what a resumed run still writes, not the whole run.

Breakage prevented: a 24 hour run resumed from its hour-23 checkpoint was
priced for all 25 history frames and 24 checkpoint sets (3.5 GiB) where it
writes one frame and one set (0.14 GiB), so run-plan refused, before it
started, a resume that fits the disk.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from woof import disk_budget, runplan
from woof.io import restart
from test_restart import _cfg, _fill_setup, _shim_state

# Frame counts and layouts are checked in classic frame bytes; compressed pricing:
# tests/test_compressed_history_pricing.py.
pytestmark = pytest.mark.usefixtures("classic_history_frames")

#: One product drawn on every frame and one drawn as each hour's window
#: closes, both priced from the measured per-product table.
EVERY_FRAME = "10m_wind_speed_and_direction"
HOURLY_WINDOW = "10m_wind_1h_max"


def _case(tmp_path, monkeypatch, elapsed, products="all"):
    """A real checkpoint at ``elapsed`` of a 24 hour hourly run, and its plan."""
    cfg = _cfg(run_seconds=86400.0, output_interval_s=3600.0,
               restart_interval_s=3600.0)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    state.elapsed_seconds = elapsed
    checkpoint = restart.write_restart(tmp_path / "checkpoint.npz", state, cfg)
    exp = SimpleNamespace(
        start_time=datetime(2026, 9, 27), run_seconds=86400.0,
        restart_interval_s=3600.0,
        domains=(SimpleNamespace(grid_id=1, run=cfg, start_time=None,
                                 history_interval_s=3600.0),))
    plan = SimpleNamespace(
        route="experiment", run_dir=tmp_path / "continued", config_intent=None,
        run_options={"restart": str(checkpoint), "keep_checkpoints": 1,
                     "render_products": products})
    return plan, exp


def _project(plan, exp):
    return runplan._disk_projection(plan, exp, raw={}, data=None,
                                    fetch_arguments=None)


@pytest.mark.parametrize("elapsed,frames,sets",
                         [(82800.0, 1, 1), (81000.0, 2, 2), (86400.0, 0, 0)],
                         ids=["hour-23", "hour-22.5", "at-the-stop"])
def test_a_resume_prices_only_the_frames_and_sets_after_its_checkpoint(
        tmp_path, monkeypatch, elapsed, frames, sets):
    plan, exp = _case(tmp_path, monkeypatch, elapsed,
                      products=f"{EVERY_FRAME},{HOURLY_WINDOW}")
    projected = _project(plan, exp)
    run = exp.domains[0].run
    cells = run.nx * run.ny * run.nz
    # The grid is below the table's smallest bracket, so each picture costs
    # its product's small-grid size.  The hourly window closes once for
    # every hour after the checkpoint, as the frames are written.
    rows = disk_budget._picture_table()["products"]
    per_frame = rows[EVERY_FRAME][2] + rows[HOURLY_WINDOW][2]
    # A190 prices the selected writer shapes, rather than full-volume bytes.
    remaining = (disk_budget.history_frame_bytes(run) * frames
                 + int(cells * disk_budget.ROOT_CHECKPOINT_BYTES_PER_CELL * sets)
                 + per_frame * frames
                 + projected["preparation_bytes"])
    assert projected["picture_bytes"] == per_frame * frames
    # A disk that holds exactly what is left admits the resume.
    assert disk_budget.disk_refusal(projected, remaining) is None
    assert projected["domains"][0]["history_frames"] == frames
    assert projected["checkpoint_sets_held"] == sets
    assert projected["total_bytes"] == remaining
    assert projected["resume_seconds"] == elapsed


def test_a_cold_start_still_prices_the_whole_run(tmp_path, monkeypatch):
    plan, exp = _case(tmp_path, monkeypatch, 82800.0)
    plan.run_options.pop("restart")
    projected = _project(plan, exp)
    assert projected["domains"][0]["history_frames"] == 25
    assert projected["checkpoint_sets_held"] == 2
    assert projected.get("resume_seconds") is None


def test_a_checkpoint_at_time_zero_keeps_the_first_frame(tmp_path, monkeypatch):
    plan, exp = _case(tmp_path, monkeypatch, 0.0)
    assert _project(plan, exp)["domains"][0]["history_frames"] == 25


def test_an_unreadable_checkpoint_is_priced_as_a_cold_start(tmp_path, monkeypatch):
    plan, exp = _case(tmp_path, monkeypatch, 82800.0)
    broken = tmp_path / "broken.npz"
    broken.write_bytes(b"not a checkpoint")
    plan.run_options["restart"] = str(broken)
    assert _project(plan, exp)["domains"][0]["history_frames"] == 25


@pytest.mark.parametrize("elapsed,delay", [(180, 0), (300, 480), (480, 480),
                                           (240, 120)])
def test_the_projection_counts_what_the_real_resumed_clock_writes(elapsed, delay):
    """Checked against the executor a resumed tree runs, nest starts included."""
    from woof.core.clock import build_schedule, execute_schedule, resolve_clock
    from test_clock import _chain_experiment, _with_delayed_start

    exp = _with_delayed_start(
        _chain_experiment((1, 3), run_seconds=600, history_s=120), 2, delay)
    exp = replace(exp, restart_interval_s=120)
    clock = resolve_clock(exp, lbc_interval_s=60)
    schedule = build_schedule(exp, clock)
    clocks = clock.clocks()
    for item in clocks.values():
        item.ticks = elapsed * clock.tick_den
    started = {gid for gid, item in clocks.items()
               if item.spec.start_ticks <= item.ticks}
    # The restore marks every domain whose frame is due at the checkpoint
    # as already written (restore_tree_restart).
    committed = {gid for gid, item in clocks.items() if item.history_due()}
    report = execute_schedule(
        schedule, clocks=clocks,
        start_period=elapsed * clock.tick_den // schedule.period_ticks,
        started_grid_ids=started, committed_initial_history_grid_ids=committed)
    projection = disk_budget.projected_run_bytes(
        exp, keep_checkpoints=None, fetch=None, chain=None, render=False,
        resume_seconds=float(elapsed))
    assert ({row["grid_id"]: row["history_frames"]
             for row in projection["domains"]} == dict(report.histories))
    assert projection["checkpoint_sets_held"] == report.restarts


def test_a_prepared_resume_is_not_charged_a_preparation_or_its_frame_stream(
        tmp_path, monkeypatch):
    """A resume runs on the bundle that wrote its checkpoint and prepares nothing.

    The prepared route refuses a restart without that bundle, so its
    download, its preparation and the preparation's decoded frame stream
    (about 40 GiB for this GEM GDPS window on the antimeridian, where the
    ring is kept whole) are never written again, and a resume charged for
    them would be refused on a disk that holds it.
    """
    import tomllib

    from woof.experiment import load_experiment
    from test_compose_scratch_budget import _gdps_config

    config = _gdps_config(tmp_path, on_cut=True)
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    exp = load_experiment(config)
    monkeypatch.setattr(
        runplan, "_existing_prepared_bundle",
        lambda plan: ({"layout": "tree"} if plan.run_options.get("prepared_root")
                      else None))

    def project(options):
        plan = SimpleNamespace(route="prepared", run_dir=tmp_path / "run",
                               config_intent=None,
                               run_options={"render_products": "none", **options})
        return runplan._disk_projection(plan, exp, raw=raw, data=None,
                                        fetch_arguments=None)

    cold = project({})
    assert cold["compose_scratch"]["composes"] and cold["compose_scratch_bytes"] > 0
    assert cold["preparation_bytes"] > 0
    resumed = project({"prepared_root": str(tmp_path / "bundle"),
                       "restart": str(tmp_path / "checkpoint.npz")})
    assert resumed["compose_scratch"]["composes"] is False
    assert resumed["compose_scratch_bytes"] == 0
    assert resumed["preparation_bytes"] == 0 and resumed["download_bytes"] == 0
