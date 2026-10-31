"""Disk admission prices a quick-look stream's subset frames as the subset.

Breakage it prevents (WOOF 2026-10-28, woof 2.9.9): a GFS 12-3-1 km run,
1164 x 714 x 49 on its 1 km grid, 6 h, full frames hourly on every grid and
the quick-look subset every minute on the 1 km grid
(``quicklook = { every_s = 60.0 }``), was projected at 361 full frames on its
1 km grid: 1607.4 GiB of history. ``woof go`` refused it on a disk with
1611.1 GiB free; it writes about 213 GiB.
"""

from pathlib import Path

import pytest

from woof import disk_budget
from woof.experiment import load_experiment


# Frame counts and layouts are checked in classic frame bytes; compressed pricing:
# tests/test_compressed_history_pricing.py.
pytestmark = pytest.mark.usefixtures("classic_history_frames")

GIB = 1024 ** 3

ETA = ("1.0, 0.9978, 0.99519, 0.99212, 0.98849, 0.98422, 0.97918, 0.97325, 0.96627, 0.95808, "
       "0.94846, 0.93719, 0.92402, 0.90866, 0.89079, 0.87006, 0.84612, 0.81857, 0.78706, 0.75124, "
       "0.7108, 0.66556, 0.61547, 0.56067, 0.50519, 0.45474, 0.40886, 0.36713, 0.32918, 0.29466, "
       "0.26328, 0.23473, 0.20877, 0.18516, 0.16369, 0.14417, 0.12641, 0.11026, 0.09557, 0.08222, "
       "0.07007, 0.05902, 0.04898, 0.03984, 0.03153, 0.02398, 0.0171, 0.01085, 0.00517, 0.0")

TOML = f"""
[experiment]
name = "quicklook-disk"
start_time = 2026-10-28T12:00:00
run_seconds = 21600.0
feedback = 0
smooth_option = 0
blend_width = 5
spec_bdy_width = 5
restart_interval_s = 3600.0

[projection]
map_proj = "lambert"
ref_lat = 35.0
ref_lon = -97.0
truelat1 = 30.0
truelat2 = 60.0
stand_lon = -97.0

[shared]
nz = 49
ztop = 20000.0
p_top = 5000.0
eta_levels = [{ETA}]
hybrid_opt = 2
etac = 0.2
base_temp = 290.0
time_step_sound = 4
emdiv = 0.01
hypsometric_opt = 2
h_sca_adv_order = 5
smdiv = 0.1
moist_adv_opt = 1
w_damping = 1
damp_opt = 3
zdamp = 5000.0
dampcoef = 0.2
khdif = 0.0
kvdif = 0.0
spec_zone = 1
relax_zone = 4
bldt = 0.0
nwp_diagnostics = 1
bl_pbl_physics = 1
diff_6th_opt = 2
diff_6th_slopeopt = 1
epssm = 0.5
km_opt = 4
moist = true
moist_cq = true
morr_rimed_ice = 1
mp_physics = 10
num_soil_layers = 4
ra_lw_physics = 4
ra_physics = 0
ra_rrtmg_variant = "rte-rrtmgp"
ra_sw_physics = 4
sf_sfclay_physics = 91
sf_surface_physics = 2
sf_urban_physics = 0
terrain_opt = 1
top_lid = false
wrf_rrtmg_compatibility = "wrf-rrtmg-4-4-to-rte-rrtmgp-v2"
wsm6_hail_opt = 0
map_proj = 1
use_adaptive_time_step = true
max_step_increase_pct = 5

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 122
ny = 86
time_step = 60
dx = 12000.0
specified = true
nested = false
history_interval_s = 3600.0
radt = 12.0
cu_physics = 1
cudt_minutes = 5.0
diff_6th_factor = 0.12

[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 11
j_parent_start = 11
parent_grid_ratio = 4
parent_time_step_ratio = 4
nx = 408
ny = 264
specified = false
nested = true
history_interval_s = 3600.0
radt = 12.0
cu_physics = 0

[[domain]]
grid_id = 3
parent_id = 2
i_parent_start = 11
j_parent_start = 14
parent_grid_ratio = 3
parent_time_step_ratio = 3
nx = 1164
ny = 714
specified = false
nested = true
history_interval_s = 3600.0
epssm = 0.5
radt = 12.0
cu_physics = 0
diff_6th_factor = 0.08
{{quicklook}}
"""


def _experiment(tmp_path: Path, quicklook: str):
    path = tmp_path / "experiment.toml"
    path.write_text(TOML.replace("{quicklook}", quicklook), encoding="utf-8")
    return load_experiment(path)


def _project(exp, render=False):
    return disk_budget.projected_run_bytes(exp, keep_checkpoints=1, fetch=None, chain=None,
                                           render=render)


def test_one_minute_quicklook_prices_354_subset_frames_and_7_full(tmp_path):
    exp = _experiment(tmp_path, "quicklook = { every_s = 60.0 }")
    fine = exp.domains[2]
    assert disk_budget.history_frames(exp, fine) == 361
    assert disk_budget.quicklook_frames(exp, fine) == 354
    row = _project(exp)["domains"][2]
    full = row["history_frame_bytes"]
    quick = row["quicklook_frame_bytes"]
    assert row["history_frames"] == 361 and row["quicklook_frames"] == 354
    # The subset keeps the default variables and the writer's coordinates,
    # and T and PHB only when named (fix/quicklook-static-once 7c60a6f52b):
    # about 250 B a column against about 5,700 for a full frame.
    columns = 1164 * 714
    assert 230 < quick / columns < 280
    assert quick < full / 20
    assert row["history_bytes"] == pytest.approx(7 * full + 354 * quick, abs=full)
    # The whole run's history: about 105 GiB (it was projected at 1607.4, and
    # at about 213 GiB while the subset still carried T and PHB).
    history = _project(exp)["history_bytes"] / GIB
    assert 90 < history < 130


def test_five_minute_quicklook_on_the_same_grids(tmp_path):
    exp = _experiment(tmp_path, "quicklook = { every_s = 300.0 }")
    fine = exp.domains[2]
    assert disk_budget.history_frames(exp, fine) == 73
    assert disk_budget.quicklook_frames(exp, fine) == 66
    # Measured on the production run of this shape: 7 full frames of
    # 4,697,788,164 B and 66 subset frames of 538,560,188 B on the 1 km grid,
    # when every subset frame still carried T (49 levels) and PHB (50) as
    # float32: 1164 x 714 x 99 x 4 = 329,114,016 B. Quick-look frames carry
    # them only when named since fix/quicklook-static-once 7c60a6f52b.
    row = _project(exp)["domains"][2]
    assert row["quicklook_frame_bytes"] == pytest.approx(538_560_188 - 329_114_016, rel=0.05)


def test_a_run_without_quicklook_projects_what_it_did_before(tmp_path):
    exp = _experiment(tmp_path, "")
    fine = exp.domains[2]
    assert disk_budget.quicklook_frames(exp, fine) == 0
    projection = _project(exp, render=True)
    row = projection["domains"][2]
    assert "quicklook_frames" not in row
    assert row["history_frames"] == 7
    assert row["history_bytes"] == 7 * row["history_frame_bytes"]


def test_quicklook_pictures_are_the_quicklook_maps_only(tmp_path):
    """A quick-look frame draws the four quick-look maps, not the run's whole set."""
    with_quick = _project(_experiment(tmp_path, "quicklook = { every_s = 60.0 }"), render=True)
    plain = _project(_experiment(tmp_path, ""), render=True)
    full_pictures = plain["domains"][2]["picture_bytes"]
    quick_pictures = with_quick["domains"][2]["picture_bytes"] - full_pictures
    # 354 subset frames of four maps each, each map under 1.4 MB.
    assert 0 < quick_pictures < 354 * 4 * 1.4e6


def test_a_resumed_run_counts_only_the_subset_frames_after_its_checkpoint(tmp_path):
    exp = _experiment(tmp_path, "quicklook = { every_s = 60.0 }")
    fine = exp.domains[2]
    # Resumed at 3 h: alarms after 10800 s, k = 181 .. 360; full at k = 240, 300, 360.
    assert disk_budget.history_frames(exp, fine, after_seconds=10800.0) == 180
    assert disk_budget.quicklook_frames(exp, fine, after_seconds=10800.0) == 177
