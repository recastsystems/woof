"""A downscaled child keeps one checkpoint set and prices its disk first.

The breakage: an 11 hour 250 m child kept every hourly checkpoint it wrote
(11 sets of 2.59 GB, 28.5 GB beside its history) and nothing projected the
disk it would take before it started, so a long child could fill its disk
partway through.  The forecast route has kept one set by default since
run-plan learned ``keep_checkpoints``; the downscale route now keeps one
set the same way, writing the new set before the old one goes, and the
plan review carries the disk projection and refuses a child the free space
cannot hold before the child starts.
"""

from datetime import datetime, timedelta
import json
import os
from pathlib import Path

import numpy as np
import pytest

from woof import disk_budget
from woof.cli import main as cli_main
from woof.downscale import downscale_plan_path
from woof.offline_child import OfflineChildContractError
from woof.offline_child_run import (
    ChildCheckpoints,
    checkpoint_schedule,
    child_cadence,
    child_checkpoint_retention,
    child_disk_projection,
    child_history_frames,
)
from woof.io.restart import _HEADER_KEY as RESTART_HEADER_KEY
from woof.io.restart import RESTART_FORMAT_VERSION, read_restart_header
from woof.resume import KEEP_CHECKPOINTS_ENV, discover_checkpoint_sets
from test_downscale_cli import (  # noqa: F401  (the autouse fixture is imported to apply here)
    _a_box_that_can_draw,
    _add_parent_surface,
    _point_args,
)


# Frame counts and layouts are checked in classic frame bytes; compressed pricing:
# tests/test_compressed_history_pricing.py.
pytestmark = pytest.mark.usefixtures("classic_history_frames")

GIB = 1024 ** 3
START = datetime(2023, 6, 21, 18)


class _StubStepper:
    """The child's stepper with no model behind it: it only advances the clock."""

    def __init__(self, dt: float) -> None:
        self.dt = float(dt)
        self.elapsed_seconds = 0.0

    def __call__(self) -> None:
        self.elapsed_seconds += self.dt


def _drive_child(outdir: Path, *, keep, intervals: int = 4,
                 restart_steps: int = 3, dt: float = 1200.0):
    """``intervals`` restart intervals through the child's own checkpoint path.

    The writer publishes the way ``write_restart`` does: the restart
    header under the key the reader looks for, beside the arrays, into a
    temporary file that is then renamed.  A child's set is its one grid,
    so its header declares no ``domain_ids``, as the child's own
    ``write_restart`` call leaves it; retirement counts a set as whole
    only when that header reads.  The writer counts the child's sets on
    disk the moment each new one is published, before anything older can
    go: that count is the peak the directory reaches.
    """
    steps = intervals * restart_steps
    due = checkpoint_schedule(steps, restart_steps)
    peaks: list[int] = []
    stepper = _StubStepper(dt)

    def write(path: Path) -> Path:
        header = {"format_version": RESTART_FORMAT_VERSION,
                  "elapsed_seconds": stepper.elapsed_seconds,
                  "config": {"grid_id": 2, "dt": stepper.dt}}
        payload = {RESTART_HEADER_KEY: np.frombuffer(
            json.dumps(header, allow_nan=False).encode("utf-8"), dtype=np.uint8),
            "state": np.zeros(8)}
        temporary = path.with_name(path.name + ".partial")
        with open(temporary, "wb") as stream:
            np.savez(stream, **payload)
        os.replace(temporary, path)
        peaks.append(len(list(outdir.glob("gpuwmrst_d02_*.npz"))))
        return path

    checkpoints = ChildCheckpoints(outdir, grid_id=2, keep=keep, write=write)
    for step in range(1, steps + 1):
        stepper()
        if step in due:
            checkpoints.emit(START + timedelta(seconds=stepper.elapsed_seconds))
    return checkpoints, peaks


def test_a_child_keeps_one_checkpoint_set_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(KEEP_CHECKPOINTS_ENV, raising=False)
    keep = child_checkpoint_retention(None)
    assert keep == 1

    checkpoints, peaks = _drive_child(tmp_path, keep=keep)

    assert len(peaks) == 4
    assert max(peaks) <= 2, peaks
    on_disk = sorted(tmp_path.glob("gpuwmrst_d02_*.npz"))
    assert [path.name for path in on_disk] == ["gpuwmrst_d02_2023-06-21_22_00_00.npz"]
    assert checkpoints.on_disk() == [checkpoints.last] == on_disk
    assert len(checkpoints.written) == 4 and len(checkpoints.retired) == 3
    # The set that stays is the one the next downscale discovers, and a
    # restart it can read: the newest clock the child reached.
    sets = discover_checkpoint_sets(tmp_path)
    assert len(sets) == 1 and sets[0].valid_time == START + timedelta(hours=4)
    assert read_restart_header(sets[0].handle)["elapsed_seconds"] == 4 * 3600.0


def test_keeping_every_set_is_still_one_flag_away(tmp_path, monkeypatch):
    monkeypatch.delenv(KEEP_CHECKPOINTS_ENV, raising=False)
    checkpoints, peaks = _drive_child(
        tmp_path, keep=child_checkpoint_retention(0))

    assert peaks == [1, 2, 3, 4]
    assert len(list(tmp_path.glob("gpuwmrst_d02_*.npz"))) == 4
    assert checkpoints.retired == []


def test_the_retention_is_the_flag_then_the_run_plan_knob_then_one(monkeypatch):
    monkeypatch.delenv(KEEP_CHECKPOINTS_ENV, raising=False)
    assert child_checkpoint_retention(None) == 1
    assert child_checkpoint_retention(3) == 3
    assert child_checkpoint_retention(0) is None
    monkeypatch.setenv(KEEP_CHECKPOINTS_ENV, "2")
    assert child_checkpoint_retention(None) == 2
    assert child_checkpoint_retention(5) == 5
    monkeypatch.setenv(KEEP_CHECKPOINTS_ENV, "0")
    assert child_checkpoint_retention(None) is None
    monkeypatch.setenv(KEEP_CHECKPOINTS_ENV, "some")
    with pytest.raises(OfflineChildContractError, match="whole number"):
        child_checkpoint_retention(None)
    with pytest.raises(OfflineChildContractError, match="negative"):
        child_checkpoint_retention(-1)


#: Bytes per history frame of the measured long child under the inventory
#: it was measured with (2.9.1: 1,278,768,100) and under today's default
#: inventory, both from woof.io.history_layout.produced_history_shapes for
#: that child.  The 2.9.1 intake pinned the chain's first inventory
#: (1,807,748,376); b6ea150a040 then made the chain's 3-D tendency rows and
#: strict-only history strict-only, and 2.9.2's default RTE+RRTMGP
#: radiation rows (COSZEN, the 16 top/bottom fluxes and their AC* twins,
#: 33 planes) joined: 1,549,359,360 today.
CHAIN_HISTORY_GROWTH = 1_549_359_360 / 1_278_768_100


def _the_long_child():
    """The measured child: 250 m, 552x552x49, 11 h, hourly history and checkpoints.

    Its own derived child config, physics included (moist Thompson, YSU, Noah,
    RTE+RRTMGP, UH diagnostics): history is priced from the writer's inventory
    for those settings, which a grid size alone does not name.
    """
    from woof.config import RunConfig
    from test_downscale_pricing import _TWO_FIFTY_METRE_CHILD

    return RunConfig(**_TWO_FIFTY_METRE_CHILD)


def test_the_projection_prices_the_long_child_on_its_own_clock(tmp_path):
    cfg = _the_long_child()
    cadence = child_cadence(cfg)
    assert child_history_frames(cadence) == 12
    assert len(cadence.checkpoint_due) == 11
    cells = 552 * 552 * 49

    every = child_disk_projection(cfg, cadence, keep_checkpoints=None,
                                  render_products="all", outdir=tmp_path)
    assert every["checkpoint_sets_written"] == every["checkpoint_sets_held"] == 11
    # The projection bounds what that child actually wrote: 11 sets of
    # 2.59 GB and 12 frames of 1.2 GB.  History is the writer's inventory for
    # the child's own physics (A190), held within 10% above the measurement.
    assert every["checkpoint_bytes"] >= 11 * 2.59e9
    measured = 12 * 1.2e9 * CHAIN_HISTORY_GROWTH
    assert measured <= every["history_bytes"] <= measured * 1.10

    one = child_disk_projection(cfg, cadence, keep_checkpoints=1,
                                render_products="all", outdir=tmp_path)
    assert one["checkpoint_sets_held"] == 2
    assert one["checkpoint_bytes"] == int(cells * disk_budget.CHECKPOINT_BYTES_PER_CELL * 2)
    # Without a local catalog, pricing expands the measured product rows.
    assert one["pictures_per_frame"] is None
    assert one["picture_bytes"] == disk_budget.projected_picture_bytes(
        cfg.nx, cfg.ny, cfg.run_seconds, cfg.output_interval_s, "all")
    assert one["total_bytes"] == (one["history_bytes"] + one["checkpoint_bytes"]
                                  + one["picture_bytes"])
    assert one["keep_checkpoints"] == 1
    # A child downloads nothing and prepares nothing on disk, and says so.
    assert one["download_bytes"] == one["preparation_bytes"] == 0
    assert one["unpriced"] == []

    undrawn = child_disk_projection(cfg, cadence, keep_checkpoints=1,
                                    render_products="none", outdir=tmp_path)
    assert undrawn["picture_bytes"] == 0


def test_a_projection_past_the_free_space_is_a_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(disk_budget, "free_bytes", lambda path: 10 * GIB)
    cfg = _the_long_child()
    disk = child_disk_projection(cfg, child_cadence(cfg), keep_checkpoints=1,
                                 render_products="all", outdir=tmp_path)
    assert disk["fits"] is False and disk["free_bytes"] == 10 * GIB
    assert disk["refusal"].startswith("this child would write about")
    assert "10.0 GiB free" in disk["refusal"] and "--child-size" in disk["refusal"]


#: The renderer's list as ``rw_wrfbatch --list-products`` gives it, with
#: the block of what a local run can draw, each with its kind and the first
#: forecast hour it can exist at.
_LOCAL_CATALOG = {
    "engine": "rust",
    "products": [{"name": name} for name in (
        "composite_reflectivity", "2m_temperature", "10m_wind_speed_and_direction",
        "total_qpf", "sbcape", "qpf_1h", "qpf_24h")],
    "group_keywords": ["all", "direct", "derived", "windowed", "variables"],
    "local_run": {"products": [
        {"name": "composite_reflectivity", "kind": "direct", "minimum_hour": 0},
        {"name": "2m_temperature", "kind": "direct", "minimum_hour": 0},
        {"name": "10m_wind_speed_and_direction", "kind": "direct", "minimum_hour": 0},
        {"name": "total_qpf", "kind": "direct", "minimum_hour": 0},
        {"name": "sbcape", "kind": "derived", "minimum_hour": 0},
        {"name": "qpf_1h", "kind": "windowed", "minimum_hour": 1},
        {"name": "qpf_24h", "kind": "windowed", "minimum_hour": 24}],
        "unavailable": {}},
}

THREE = "composite_reflectivity,2m_temperature,10m_wind_speed_and_direction"


@pytest.fixture
def local_catalog(monkeypatch):
    import woof.runplan as runplan

    monkeypatch.setattr(runplan, "render_catalog", lambda: dict(_LOCAL_CATALOG))


def test_pictures_are_counted_per_product_asked_for(local_catalog):
    count = disk_budget.pictures_per_frame
    assert count(None) == count("none") == count("") == 0
    assert count(THREE) == 3
    # A short name is the product it stands for, not a second picture.
    assert count("refl,composite_reflectivity") == 1
    # `all` is every product a local run can draw; a window that first
    # closes after the run ends is not drawn by it.
    assert count("all", run_hours=11) == 6
    assert count("all") == 7
    assert count("windowed", run_hours=11) == 1
    assert count("direct,composite_reflectivity") == 4
    assert count("all,variables", run_hours=11) == 6 + disk_budget.STORED_VARIABLE_PICTURES
    # A section is one product, its level list included, even where the
    # list's last level carries the term that closes it.
    assert count("xsec:wa=1,2,5@5,composite_reflectivity") == 2
    assert count("xsec:QCLOUD=0.01,0.1/wa,composite_reflectivity") == 2


def test_a_group_this_computer_cannot_expand_is_not_counted(monkeypatch):
    import woof.runplan as runplan

    # No local-run block (a renderer that publishes none, or none at all):
    # named products still count, a group cannot be.
    monkeypatch.setattr(runplan, "render_catalog", lambda: {"products": None})
    assert disk_budget.pictures_per_frame("all") is None
    assert disk_budget.pictures_per_frame(THREE) == 3
    # A group keyword whose members the list does not name by kind.
    monkeypatch.setattr(runplan, "render_catalog", lambda: dict(
        _LOCAL_CATALOG, group_keywords=["all", "severe"]))
    assert disk_budget.pictures_per_frame("severe") is None


def _runs_record(name: str) -> dict:
    return json.loads((Path(__file__).resolve().parents[1] / "tools" / "wiki_seed" / "runs"
                       / name / "sizes.json").read_text(encoding="utf-8"))


def _pictures_per_frame_now(run: dict) -> int:
    """Pictures a frame the projection prices for this run's product list
    today: the renderer's catalog where it is installed, else the measured
    table's groups (the same choice disk_budget._picture_projection makes)."""
    selected = disk_budget._picture_products(run["render_products"], run_hours=run["run_hours"])
    if selected is None:
        selected = disk_budget._picture_products(
            run["render_products"], run_hours=run["run_hours"], measured=True)
    return sum(row[2] for row in selected.values())


def test_the_picture_price_covers_every_frame_it_was_read_from():
    """Every measured frame is inside its projection, and the price is the measurement's, not a guess."""
    record = _runs_record("bytes-per-picture")
    largest_mean = max(run["mean_bytes_per_picture"] for run in record["runs"].values())
    for name, run in record["runs"].items():
        projected = disk_budget.projected_picture_bytes(
            *run["columns"], run["run_hours"] * 3600,
            run["history_interval_s"], run["render_products"])
        ceiling = 1.6 if run["mean_bytes_per_picture"] == largest_mean else 2.5
        # The price is per picture: a product group that has grown since the
        # measurement (8688072afe5 added the nine winter products to the
        # table, so "all" prices 83 pictures a frame where the measured run
        # rendered 74) is priced at today's count against the measured bytes
        # per picture, not at the old frame total.
        counted = _pictures_per_frame_now(run)
        growth = max(1.0, counted / run["projected_pictures_per_frame"])
        assert run["picture_bytes"] <= projected <= ceiling * growth * run["picture_bytes"], name
        count = run["projected_pictures_per_frame"]
        assert run["frames"] == len(run["per_frame"]) > 1, name
        previous = 0
        for index, frame in enumerate(run["per_frame"]):
            # The count is exact on a frame every product can draw on and an
            # upper bound on the first frames, before a window closes.
            assert frame["pictures"] <= count, (name, frame)
            total = disk_budget.projected_picture_bytes(
                *run["columns"], index * run["history_interval_s"],
                run["history_interval_s"], run["render_products"])
            assert frame["bytes"] <= total - previous, (name, frame)
            previous = total
        assert max(frame["pictures"] for frame in run["per_frame"]) == count, name
    # The run the bytes per cell were read from drew the catalog of its day
    # (every stored variable as well); its pictures are inside the price too.
    for grid in _runs_record("bytes-per-cell")["grids"].values():
        price = disk_budget.projected_picture_bytes(*grid["columns"], 0, 3600, "all,variables")
        assert grid["picture_bytes_per_frame"] <= price
    for drawn in record["variables"].values():
        assert drawn["pictures"] <= disk_budget.STORED_VARIABLE_PICTURES
        assert drawn["bytes"] <= disk_budget.projected_picture_bytes(36, 36, 0, 3600, "variables")
    assert disk_budget.PICTURES_MEASURED.endswith("tools/wiki_seed/runs/bytes-per-picture/sizes.json")


def _a_short_child_at_minute_history():
    """120x120x49 over 4 h with a frame every minute: 241 frames, with the
    measured long child's physics."""
    from dataclasses import replace

    return replace(_the_long_child(), nx=120, ny=120, run_seconds=4 * 3600.0,
                   output_interval_s=60.0)


def test_fewer_products_shrink_the_figure_and_a_child_that_fits_is_admitted(
        tmp_path, monkeypatch, local_catalog):
    # The breakage: pictures were a flat 150 MB a frame for any list but
    # `none`, so this three-product child was priced at 33.7 GiB of
    # pictures beside 14.8 GiB of history and refused on a 20 GiB disk it
    # fits on, and trimming the list gave the same figure back.
    # 24 GiB since the exact-chain history rows (CHAIN_HISTORY_GROWTH): the
    # child writes about 16.5 GiB of history, and the flat price would still
    # be refused (33.7 GiB of pictures alone).
    monkeypatch.setattr(disk_budget, "free_bytes", lambda path: 24 * GIB)
    cfg = _a_short_child_at_minute_history()
    cadence = child_cadence(cfg)
    assert child_history_frames(cadence) == 241

    def disk(products):
        return child_disk_projection(cfg, cadence, keep_checkpoints=1,
                                     render_products=products, outdir=tmp_path)

    three, every, none = disk(THREE), disk("all"), disk("none")
    assert three["pictures_per_frame"] == 3 and every["pictures_per_frame"] == 6
    assert three["picture_bytes"] == disk_budget.projected_picture_bytes(
        cfg.nx, cfg.ny, cfg.run_seconds, cfg.output_interval_s, THREE)
    assert none["picture_bytes"] == 0
    assert every["total_bytes"] > three["total_bytes"] > none["total_bytes"]
    # Bounded below by the measured long child's bytes per cell (1.2 GB a
    # frame over 552x552x49): a smaller grid carries relatively more face,
    # surface and header bytes, never fewer per cell.
    measured_per_cell = 1.2e9 * CHAIN_HISTORY_GROWTH / (552 * 552 * 49)
    floor = 241 * 120 * 120 * 49 * measured_per_cell
    assert floor <= three["history_bytes"] <= 1.15 * floor
    assert three["picture_bytes"] < 1 * GIB
    assert three["fits"] is True and three["refusal"] is None


def test_the_refusal_names_only_the_flags_that_shrink_it(tmp_path, monkeypatch):
    import woof.runplan as runplan

    monkeypatch.setattr(disk_budget, "free_bytes", lambda path: 1 * GIB)
    monkeypatch.setattr(runplan, "render_catalog", lambda: dict(_LOCAL_CATALOG))
    cfg = _the_long_child()
    cadence = child_cadence(cfg)

    def refusal(products, keep=1):
        return child_disk_projection(cfg, cadence, keep_checkpoints=keep,
                                     render_products=products,
                                     outdir=tmp_path)["refusal"]

    # Pictures already off: no products to trim, so none are offered.
    undrawn = refusal("none")
    assert "--render-products" not in undrawn and "--child-size" in undrawn
    # Pictures priced per product: fewer products is a way out.
    assert "draw fewer products (--render-products; none draws nothing)" in refusal(THREE)
    # One set kept already: no checkpoint remedy; every set kept: there is.
    assert "--keep-checkpoints" not in refusal(THREE)
    assert "keep one checkpoint set (--keep-checkpoints 1)" in refusal(THREE, keep=None)
    # The measured table still prices fewer products without a local catalog.
    monkeypatch.setattr(runplan, "render_catalog", lambda: {"products": None})
    whole = refusal("all")
    assert "draw fewer products (--render-products; none draws nothing)" in whole


def _fitting_point_args(tmp_path):
    """The deck's --point child, with a parent that can seed its surface."""
    args = _point_args(tmp_path)
    for index in range(3):
        _add_parent_surface(
            tmp_path / f"wrfout_d01_1974-04-03_{12 + index:02d}_00_00",
            ny=18, nx=20, lu_water_column=8)
    return args


def test_the_plan_review_carries_the_disk_projection(
        tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(KEEP_CHECKPOINTS_ENV, raising=False)
    assert cli_main([*_fitting_point_args(tmp_path), "--dry-run"]) == 0
    printed = capsys.readouterr().out
    plan = json.loads(downscale_plan_path(
        tmp_path / "child-run", dry_run=True).read_text(encoding="utf-8"))
    disk = plan["disk"]
    assert disk["keep_checkpoints"] == 1
    assert disk["checkpoint_sets_written"] >= 1
    assert disk["total_bytes"] == (disk["history_bytes"] + disk["checkpoint_bytes"]
                                   + disk["picture_bytes"])
    assert disk["fits"] is True and disk["refusal"] is None
    assert disk["free_bytes"] > 0
    assert disk["download_bytes"] == disk["preparation_bytes"] == 0
    assert "this child will write about" in printed


def test_a_child_its_disk_cannot_hold_is_refused_before_it_starts(
        tmp_path, capsys, monkeypatch):
    import woof.offline_child_run as offline_child_run

    args = _fitting_point_args(tmp_path)
    monkeypatch.setattr(disk_budget, "free_bytes", lambda path: 1024)
    started = []
    monkeypatch.setattr(offline_child_run, "run",
                        lambda namespace: started.append(namespace) or {"result": "PASS"})

    assert cli_main(args) == 2
    captured = capsys.readouterr()
    assert started == []
    assert "This child would write about" in captured.err
    assert "0.0 GiB free" in captured.err and "--child-size" in captured.err
    # --out is handed back, so the corrected retry does not collide.
    assert not (tmp_path / "child-run").exists()


def test_a_child_drawn_into_nothing_is_not_sent_to_draw_fewer_products(
        tmp_path, capsys, monkeypatch):
    # The breakage: a child already run with --render-products none was
    # refused with "draw fewer products (--render-products)" as a way out,
    # which changes nothing.
    import woof.offline_child_run as offline_child_run

    args = [*_fitting_point_args(tmp_path), "--render-products", "none"]
    monkeypatch.setattr(disk_budget, "free_bytes", lambda path: 1024)
    monkeypatch.setattr(offline_child_run, "run",
                        lambda namespace: pytest.fail("the child started"))
    assert cli_main(args) == 2
    err = capsys.readouterr().err
    assert "This child would write about" in err and "of pictures" not in err
    # (The command line the refusal echoes for --explain carries the flag.)
    assert "fewer products" not in err and "(--render-products" not in err
    assert "--child-size" in err


def test_the_review_prices_the_products_it_was_asked_for(
        tmp_path, capsys, monkeypatch, local_catalog):
    args = [*_fitting_point_args(tmp_path), "--render-products", THREE, "--dry-run"]
    assert cli_main(args) == 0
    printed = capsys.readouterr().out
    plan = json.loads(downscale_plan_path(
        tmp_path / "child-run", dry_run=True).read_text(encoding="utf-8"))
    disk = plan["disk"]
    assert disk["pictures_per_frame"] == 3
    frames = disk["domains"][0]["history_frames"]
    grid = plan["child_grid"]
    assert disk["picture_bytes"] == disk_budget.projected_picture_bytes(
        grid["nx"], grid["ny"], grid["run_seconds"], grid["output_interval_s"], THREE)
    assert disk["picture_basis"] == disk_budget._picture_table()["basis"]
    assert "GiB of pictures, 3 pictures a frame)" in printed


def test_a_review_past_the_free_space_says_so_and_still_prints(
        tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(disk_budget, "free_bytes", lambda path: 1024)
    assert cli_main([*_fitting_point_args(tmp_path), "--dry-run"]) == 0
    capsys.readouterr()
    plan = json.loads(downscale_plan_path(
        tmp_path / "child-run", dry_run=True).read_text(encoding="utf-8"))
    assert plan["disk"]["fits"] is False
    assert any("This child would write about" in record.get("action", "")
               for record in plan["warnings"]), plan["warnings"]


def test_the_run_is_handed_the_retention_the_plan_priced(
        tmp_path, capsys, monkeypatch):
    import woof.offline_child_run as offline_child_run

    monkeypatch.delenv(KEEP_CHECKPOINTS_ENV, raising=False)
    handed = []
    monkeypatch.setattr(offline_child_run, "run",
                        lambda namespace: handed.append(namespace) or {"result": "PASS"})
    args = _fitting_point_args(tmp_path)
    assert cli_main(args) == 0
    capsys.readouterr()
    assert handed[0].keep_checkpoints == 1
    plan = json.loads((tmp_path / "child-run" / "downscale-plan.json")
                      .read_text(encoding="utf-8"))
    assert plan["disk"]["keep_checkpoints"] == 1

    again = tmp_path / "again"
    again.mkdir()
    assert cli_main([*_fitting_point_args(again), "--keep-checkpoints", "0"]) == 0
    capsys.readouterr()
    assert handed[1].keep_checkpoints == 0
