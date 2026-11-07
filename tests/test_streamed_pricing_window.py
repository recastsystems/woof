"""A streamed buffer is priced at the window it holds, not at an ``N x 1`` row.

THE DEFECT (FU-008, a 2.7.2 user report).  A 572x524x49 icon-eu forecast
(Thompson, RTE+RRTMGP, MYNN, Noah, six retained forcing intervals) with a
pinned 250x250 tiling was refused before fetch: the resident forecast
priced at 15.38 GiB, the STREAMED one -- two buffers of the 286x286
compute window -- at 36.92 GiB.  A streamed run costing 2.4x the resident
one is not credible: tiling exists to shrink the device footprint.

REPRODUCED on the candidate tree at 38.96 GiB streamed against 17.43 GiB
resident (reference device profile).  The term was
``PreparedTileMemory.buffer_terms``, which itemized every buffer as an
``N x 1`` domain -- 81,796 x 1 for this window -- on the argument that a
one-row rectangle bounds every rectangle of the same area.  It does, by a
perimeter 143x the window's: the eager ``lbc_forcing_tables`` of that row
came to 9.03 GiB per buffer against 0.06 GiB for the 286x286 window, and
two buffers of it under the allocator headroom were 20.8 GiB of forcing
tables no buffer allocates.

THE FIX prices the buffer at the window's own shape on both roads -- the
pinned tiling's window from ``[tiles]``, the planner's from the tile it
chose -- and, where only a cell count is known (the planner's search),
at the most elongated rectangle a tiling can legally produce instead of a
one-row one.  The refusal prints every term, so a screenshot of it
carries the arithmetic (FU-009 item 1).

MEASURED (2026-09-10, test host 1 RTX 5070 Ti, real icon-eu 2026-09-10T12
data, nvidia-smi per-process at 0.2 s): the user's tiling with two
buffers reaches 11.11 GiB at each radiation call against the corrected
15.54 GiB envelope for that card; one buffer of the same tile reaches
6.18 GiB against 8.89 GiB.  Both bracket the run; neither is the 38.96
GiB the one-row rectangle produced.  The module docstring of
:mod:`woof.core.prepared_tile_memory` carries the full measurement.
"""
from __future__ import annotations

import argparse
import json
import textwrap

import pytest

from woof import go_cli
from woof.core import preflight as pf, streaming as st
from tilestream import autoplan as ap


GIB = 1024 ** 3

#: The user's configuration as reported, minus their output path.  Kept as
#: TOML and loaded through the product's own loader, because the report
#: was about a config a user typed and ``[tiles]`` has to survive the load.
_USER_REPORT_TOML = """\
[experiment]
name = "streamed-pricing"
start_time = 2026-09-10T12:00:00
run_seconds = 21600.0
feedback = 0
smooth_option = 0
blend_width = 5
spec_bdy_width = 5
restart_interval_s = 3600.0

[projection]
map_proj = "lambert"
ref_lat = 48.81801809375925
ref_lon = 14.104343811978481
truelat1 = 38.82
truelat2 = 58.82
stand_lon = 14.104343811978481

[shared]
nz = 49
ztop = 20000.0
p_top = 5000.0
eta_levels = [
    1.0, 0.9978, 0.99519, 0.99212, 0.98849,
    0.98422, 0.97918, 0.97325, 0.96627, 0.95808,
    0.94846, 0.93719, 0.92402, 0.90866, 0.89079,
    0.87006, 0.84612, 0.81857, 0.78706, 0.75124,
    0.7108, 0.66556, 0.61547, 0.56067, 0.50519,
    0.45474, 0.40886, 0.36713, 0.32918, 0.29466,
    0.26328, 0.23473, 0.20877, 0.18516, 0.16369,
    0.14417, 0.12641, 0.11026, 0.09557, 0.08222,
    0.07007, 0.05902, 0.04898, 0.03984, 0.03153,
    0.02398, 0.0171, 0.01085, 0.00517, 0.0,
]
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
moist = true
moist_cq = true
mp_physics = 8
top_lid = false
epssm = 0.5
morr_rimed_ice = 1
wsm6_hail_opt = 0
ra_physics = 0
ra_lw_physics = 4
ra_sw_physics = 4
wrf_rrtmg_compatibility = "wrf-rrtmg-4-4-to-rte-rrtmgp-v2"
ra_rrtmg_variant = "rte-rrtmgp"
sf_sfclay_physics = 5
sf_surface_physics = 2
bl_pbl_physics = 5
num_soil_layers = 4
terrain_opt = 1
km_opt = 4
diff_6th_opt = 2
diff_6th_slopeopt = 1
map_proj = 1

[tiles]
mode = "on"
tile_nx = {tile_nx}
tile_ny = {tile_ny}
nbuffers = {nbuffers}

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 572
ny = 524
time_step = 12
dx = 2400.0
specified = true
nested = false
history_interval_s = 900.0
radt = 2.0
cu_physics = 0
cudt_minutes = 0.0
diff_6th_factor = 0.12

[fetch]
source = "icon-eu"
cycle = "2026-09-10T12"
hours = 6
cadence = 1
"""

#: What the user's gate saw: a 12 GB card with 10.69 GiB free.
_USER_FREE_BYTES = int(10.69 * GIB)
_USER_TOTAL_BYTES = int(12 * GIB)
#: The forcing schedule the config declares: six hourly intervals.
_USER_INTERVALS = 6


def _user_config(tmp_path, *, tile_nx=250, tile_ny=250, nbuffers=2):
    path = tmp_path / "streamed-pricing.toml"
    path.write_text(_USER_REPORT_TOML.format(
        tile_nx=tile_nx, tile_ny=tile_ny, nbuffers=nbuffers), encoding="utf-8")
    return path


def _profile():
    """The measured RTX 3080 profile the prepared-tile probes were taken on."""
    return pf.DeviceLocalMemoryProfile(
        "NVIDIA GeForce RTX 3080", 68, 1536, bare_context_bytes=182452224)


def _phases(exp, *, free_gib, host_gib=64, intervals=_USER_INTERVALS,
            profile=None, source="icon-eu"):
    profile = _profile() if profile is None else profile
    machine = ap.Machine(int(free_gib * GIB), int(host_gib * GIB),
                         device_profile=profile)
    return pf.estimate_phases(
        exp, source=source, profile=profile, machine=machine,
        forcing_intervals=intervals, forcing_interval_seconds=3600.0), machine


def _footprint(exp, machine, intervals=_USER_INTERVALS):
    estimate = pf.estimate_experiment(
        exp, profile=machine.device_profile, forcing_intervals=intervals,
        forcing_interval_seconds=3600.0)
    return st.radiation_footprint(exp.root.run, exp.tiles,
                                  resident_estimate=estimate, machine=machine)


# --------------------------------------------------------------------------
# THE REGRESSION, on the user's exact configuration
# --------------------------------------------------------------------------

def test_the_user_report_buffer_is_priced_at_its_window_not_a_one_row_rectangle(tmp_path):
    exp = pf._load_experiment_any(_user_config(tmp_path))
    phases, machine = _phases(exp, free_gib=_USER_FREE_BYTES / GIB)
    env = phases.streamed
    assert env is not None and (env.window_nx, env.window_ny, env.nbuffers) == (286, 286, 2)
    fp = _footprint(exp, machine)
    pm = fp.prepared_memory
    assert pm is not None, "the user's route is the itemized prepared model"
    cells = 286 * 286 * 49

    # The envelope IS the exact-window price, on the pinned road.
    assert env.vram_bytes == fp.vram_bytes(cells, 2, (286, 286))
    assert env.peak_vram_bytes == phases.forecast_envelope_bytes

    # The phantom term is gone.  What the one-row rectangle priced for the
    # eager forcing tables, against what the window holds: the row's lbc
    # bytes were the whole excess of the 2.7.2 figure.
    window = pm._domain(286, 286)
    one_row = pm._domain(286 * 286, 1)
    assert one_row.category_bytes("lbc") > 30 * window.category_bytes("lbc")
    assert one_row.category_bytes("lbc") > 5 * GIB
    terms = dict(env.terms)
    assert terms["buffer/lbc_bytes"] == window.category_bytes("lbc")
    assert terms["buffer/lbc_bytes"] < GIB // 4

    # The credible band: two buffers of 55% of the domain's columns cost no
    # more than the whole domain resident, and no less than the buffers
    # themselves.
    assert env.peak_vram_bytes <= phases.resident_forecast_envelope_bytes
    assert env.peak_vram_bytes >= 2 * terms["buffer/total_bytes"]
    assert 0.5 * phases.resident_forecast_envelope_bytes < env.peak_vram_bytes


def test_the_user_report_admission_at_the_reported_budget_follows_the_corrected_number(
        tmp_path, monkeypatch):
    """Admitted if and only if the corrected number fits 10.69 GiB free.

    The corrected envelope for this tiling is still above the user's free
    VRAM -- two buffers of a 286x286 window each carry the unfused RTE
    chunk storage and the MYNN column workspace, so at 55% of the domain
    per buffer the pair cost about what the domain costs resident -- and
    the gate keeps refusing.  What changed is the number it refuses on and
    that the refusal now carries the arithmetic.  The assertion is the
    iff, not the outcome, so a later change to a term moves the verdict
    with it rather than against this test.
    """
    def _probe(*_args, **_kwargs):
        return {"free_bytes": _USER_FREE_BYTES, "total_bytes": _USER_TOTAL_BYTES,
                "name": "user card", "local_memory_bytes_per_thread": 0}

    monkeypatch.setattr(pf, "device_memory_probe_subprocess", _probe)
    config = _user_config(tmp_path)
    gate = go_cli.memory_gate({"config": str(config), "source": "icon-eu",
                               "cadence": 1})
    phases = gate["phases"]
    assert phases.streamed_forecast
    assert gate["refuse"] is (phases.peak_envelope_bytes > gate["free_bytes"])
    # Streamed is priced below resident, whatever the verdict.
    assert phases.forecast_envelope_bytes <= phases.resident_forecast_envelope_bytes
    # A streamed forecast no longer prices at more than 1.05x its resident
    # run on this configuration (the report was 2.4x).
    assert phases.forecast_envelope_bytes < 1.05 * phases.resident_forecast_envelope_bytes

    if gate["refuse"]:
        text = go_cli.memory_refusal_text(gate)
        for needle in ("572x524x49", "286x286x49 = 81,796 columns",
                       "buffer/total_bytes", "buffer/radiation_named_storage_bytes",
                       "fixed/local_memory_bytes", "fixed/cuda_context_bytes",
                       "host/pinned_bytes", "remedy, streamed"):
            assert needle in text, needle


def _run_check(argv, *, explain=False):
    """`woof check` through its own registrar; ``--explain`` is the CLI
    parent's flag, so it is set on the parsed namespace the way the parent
    would."""
    parser = argparse.ArgumentParser(prog="woof")
    sub = parser.add_subparsers(dest="command", required=True)
    pf.register_cli(sub)
    args = parser.parse_args(argv)
    args.explain = explain
    return args.func(args)


def test_gpuwm_check_prints_the_terms_under_a_streamed_refusal(tmp_path, capsys):
    config = _user_config(tmp_path)
    rc = _run_check(["check", str(config), "--budget-gib", "10.69"], explain=True)
    out = capsys.readouterr().out
    assert rc != 0
    assert "OVER BUDGET, STREAMED" in out
    for needle in ("domain", "572x524x49", "window", "286x286x49 = 81,796 columns",
                   "buffer/lbc_bytes", "buffer/radiation_named_storage_bytes",
                   "buffers_bytes", "fixed/local_memory_bytes", "host/pinned_bytes",
                   "vram_bytes"):
        assert needle in out, needle
    # And the JSON report carries the same terms for a script.
    rc = _run_check(["check", str(config), "--json", "--budget-gib", "10.69"])
    payload = json.loads(capsys.readouterr().out)
    terms = payload["streamed"]["terms"]
    assert terms["domain"] == "572x524x49"
    assert terms["vram_bytes"] == payload["streamed"]["vram_bytes"]
    assert terms["buffer/lbc_bytes"] < GIB // 4


# --------------------------------------------------------------------------
# THE SIZING SHAPE, where only a cell count is known
# --------------------------------------------------------------------------

def test_the_sizing_rectangle_is_a_legal_window_and_bounds_every_legal_one(tmp_path):
    exp = pf._load_experiment_any(_user_config(tmp_path))
    _phases_, machine = _phases(exp, free_gib=10.69)
    pm = _footprint(exp, machine).prepared_memory
    halo = pm.halo
    lo, hi = 2 * halo + 1, max(572, 524) + 2 * halo
    assert halo == 18

    a, b = pm.sizing_shape(286 * 286)
    assert lo <= a <= b <= hi and a * b >= 286 * 286
    assert (a, b) != (286 * 286, 1)
    # ...and it bounds the exact rectangle on every category, by a perimeter
    # a tiling can actually have, not by a one-row one.
    bound = pm.buffer_terms(286 * 286 * 49)
    exact = pm.buffer_terms(286 * 286 * 49, (286, 286))
    assert exact["resident_bytes"] <= bound["resident_bytes"] <= 1.1 * exact["resident_bytes"]
    assert exact["step_transient_bytes"] <= bound["step_transient_bytes"]

    # Every LEGAL rectangle of a given area is inside the bound.
    for nx, ny in [(37, 37), (37, 608), (100, 250), (286, 286), (536, 152), (608, 50)]:
        assert lo <= min(nx, ny) and max(nx, ny) <= hi
        cells = nx * ny * 49
        exact = pm.buffer_terms(cells, (nx, ny))
        bound = pm.buffer_terms(cells)
        assert exact["resident_bytes"] <= bound["resident_bytes"], (nx, ny)
        assert exact["step_transient_bytes"] <= bound["step_transient_bytes"], (nx, ny)
        assert pm.vram_bytes(cells, 2, (nx, ny)) <= pm.vram_bytes(cells, 2), (nx, ny)


def test_the_shape_free_bound_is_monotone_for_the_planners_inversion(tmp_path):
    exp = pf._load_experiment_any(_user_config(tmp_path))
    _phases_, machine = _phases(exp, free_gib=10.69)
    fp = _footprint(exp, machine)
    previous = 0
    for columns in sorted(set(range(1, 1500, 53)) | set(range(1369, 95000, 397))):
        cost = fp.vram_bytes(columns * 49, 2)
        assert cost >= previous, columns
        previous = cost
    budget = int(9.0 * GIB)
    cells = ap._max_window_cells(fp, 1, budget)
    assert cells > 0 and cells % 49 == 0
    assert fp.vram_bytes(cells, 1) <= budget < fp.vram_bytes(cells + 49, 1)


def test_the_planner_prices_the_tile_it_chose_at_that_tiles_window(tmp_path):
    """The planner-driven road: the plan's price is the exact window's."""
    from dataclasses import replace

    exp = pf._load_experiment_any(_user_config(tmp_path))
    exp = replace(exp, tiles=st.StreamingOptions(mode="auto"))
    profile = _profile()
    machine = ap.Machine(int(12.0 * GIB), 64 * GIB, device_profile=profile)
    estimate = pf.estimate_experiment(exp, profile=profile,
                                      forcing_intervals=_USER_INTERVALS,
                                      forcing_interval_seconds=3600.0)
    decision = st.decide(exp.root.run, exp.tiles, machine=machine,
                         resident_estimate=estimate)
    assert decision.stream, decision.reason
    env = st.streamed_envelope(exp.root.run, exp.tiles, decision=decision,
                               machine=machine, resident_estimate=estimate)
    fp = st.radiation_footprint(exp.root.run, exp.tiles,
                                resident_estimate=estimate, machine=machine)
    cells = env.window_nx * env.window_ny * 49
    assert env.vram_bytes == decision.resident_bytes
    assert env.vram_bytes == fp.vram_bytes(cells, env.nbuffers, (env.window_nx, env.window_ny))
    assert env.peak_vram_bytes <= decision.budget_bytes
    # The exact rectangle never costs more than the bound that admitted it.
    assert env.vram_bytes <= fp.vram_bytes(cells, env.nbuffers)


# --------------------------------------------------------------------------
# THE INVARIANT: streaming does not cost more than the resident run
# --------------------------------------------------------------------------

def _config(tmp_path, *, name, source, physics, nx, ny, tile, nbuffers,
            hours=6, time_step=20, dx=3000.0):
    """One specified single-root experiment on ``source`` with ``physics``."""
    path = tmp_path / f"{name}.toml"
    path.write_text(textwrap.dedent(f"""\
        [experiment]
        name = "{name}"
        start_time = 2026-09-10T12:00:00
        run_seconds = {hours * 3600.0}
        restart_interval_s = 0.0

        [fetch]
        source = "{source}"
        cycle = "2026-09-10T12"
        hours = {hours}

        [shared]
        nz = 49
        ztop = 20000.0
        nwp_diagnostics = 1
        """) + textwrap.dedent(physics) + f"""
[tiles]
mode = "on"
tile_nx = {tile}
tile_ny = {tile}
nbuffers = {nbuffers}

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = {nx}
ny = {ny}
time_step = {time_step}
dx = {dx}
history_interval_s = 3600.0
""", encoding="utf-8")
    return path


_MORRISON_KF_YSU = """
    moist = true
    moist_cq = true
    mp_physics = 10
    ra_lw_physics = 4
    ra_sw_physics = 4
    sf_sfclay_physics = 91
    sf_surface_physics = 2
    bl_pbl_physics = 1
    cu_physics = 1
"""
_THOMPSON_MYNN_NOAH = """
    moist = true
    moist_cq = true
    mp_physics = 8
    ra_lw_physics = 4
    ra_sw_physics = 4
    ra_rrtmg_variant = "rte-rrtmgp"
    sf_sfclay_physics = 5
    sf_surface_physics = 2
    bl_pbl_physics = 5
    cu_physics = 0
    km_opt = 4
"""
_WSM6_LEGACY_RRTMG = """
    moist = true
    moist_cq = true
    mp_physics = 6
    ra_lw_physics = 4
    ra_sw_physics = 4
    ra_rrtmg_variant = "rrtmg_legacy"
    sf_sfclay_physics = 91
    sf_surface_physics = 2
    bl_pbl_physics = 1
    cu_physics = 0
"""
_DRY = """
    mp_physics = 0
    moist = false
    moist_cq = false
"""


@pytest.mark.parametrize("source,physics,nx,ny,tile,nbuffers", [
    # The user's report: icon-eu, Thompson/MYNN/Noah, RTE+RRTMGP, the
    # itemized prepared model.  Two buffers of 286x286 against 572x524.
    ("icon-eu", _THOMPSON_MYNN_NOAH, 572, 524, 250, 2),
    ("icon-eu", _THOMPSON_MYNN_NOAH, 572, 524, 150, 2),
    ("icon-eu", _THOMPSON_MYNN_NOAH, 572, 524, 250, 1),
    # The measured anchor physics of the prepared model, on GFS.
    ("gfs", _MORRISON_KF_YSU, 628, 462, 200, 2),
    # Legacy RRTMG: not itemized, priced on the measured "full" rung with
    # its reserved 2.74 GiB radiation transient.
    ("hrrr", _WSM6_LEGACY_RRTMG, 800, 800, 200, 2),
    ("era5", _WSM6_LEGACY_RRTMG, 1000, 1000, 250, 2),
    # No radiation at all: the "dry" rung, no transient.
    ("gfs", _DRY, 1200, 1200, 250, 2),
])
def test_a_streamed_forecast_never_prices_above_its_resident_run(
        tmp_path, source, physics, nx, ny, tile, nbuffers):
    """For a tiling that shrinks the footprint, streaming shrinks the price.

    The scope is stated in the precondition: the buffers together hold no
    more than 60% of the domain's columns and the domain fits in host RAM
    (a 64 GiB box).  A tiling whose buffers together are most of the
    domain is not the configuration streaming exists for, and its price is
    allowed to say so.
    """
    exp = pf._load_experiment_any(_config(
        tmp_path, name="invariant", source=source, physics=physics,
        nx=nx, ny=ny, tile=tile, nbuffers=nbuffers))
    phases, machine = _phases(exp, free_gib=8.0, source=source)
    env = phases.streamed
    assert env is not None, "the fixture no longer streams"
    assert env.host_budget_bytes is not None and env.host_bytes <= env.host_budget_bytes
    covered = env.nbuffers * env.window_nx * env.window_ny
    assert covered <= 0.6 * nx * ny, "precondition: the tiling shrinks the footprint"
    assert phases.forecast_envelope_bytes == env.peak_vram_bytes
    assert env.peak_vram_bytes <= phases.resident_forecast_envelope_bytes, (
        f"streamed {env.peak_vram_bytes / GIB:.2f} GiB above resident "
        f"{phases.resident_forecast_envelope_bytes / GIB:.2f} GiB")
    # The arithmetic is carried on every road, itemized or measured rung.
    terms = dict(env.terms)
    assert terms["vram_bytes"] == env.vram_bytes
    # host_bytes is the pinned store and arena plus the domain's lateral
    # forcing series, each a named term.
    assert terms["host/pinned_bytes"] == env.pinned_bytes
    assert (terms["host/pinned_bytes"] + terms["host/boundary_table_bytes"]
            == env.host_bytes)
    assert "rung" in terms
