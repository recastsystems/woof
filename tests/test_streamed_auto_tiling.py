"""``[tiles] mode = "auto"`` never streams a domain in tiles too small to run.

THE DEFECT (measured 2026-09-26).  A 206x204x49 3 km GFS
domain on a 10 GiB RTX 3080 with about 4 GiB free missed its resident
admission budget by 0.3 GB.  Auto dropped the planner's redundancy limit
and streamed it anyway, in 1,190 tiles of 6x6 with halo 18 -- 49.95x the
necessary work -- and the only record was a plan warning no surface
printed.  Each 15 s step took 237-547 s; the same forecast with tiles off
ran at 0.6-1.9 s per step on the same busy card and finished in 98
minutes.  The review had promised 0.45-1.2 s per step, because the pace
model priced the domain's columns and the bus, and neither the halo work
nor the per-tile cost of 1,190 tile sweeps.

What is pinned here, CPU only, on the recorded configuration and card:

* auto never returns a streamed plan past
  :data:`tilestream.autoplan.MAX_REDUNDANCY`, at any free memory;
* where the card's measured free memory holds the resident envelope, auto
  runs it resident inside the external margin and says so, which is what
  the tiles-off run that finished did;
* where it does not, auto refuses before the run, in plain words, with the
  numbers and the ways out;
* the pace model quotes the runs its per-tile ends were fitted from back
  to themselves, which checks the arithmetic, and brackets every
  streamed step measured on runs that took no part in either fit, which
  checks the model;
* a streamed plan states its tile size, tile count and redundancy on the
  review, the estimate document and the run log's line.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from woof.core import pace, preflight, streaming
from woof.domain_wizard import experiment_from_text
from tilestream import autoplan

GIB = 1024 ** 3

#: The recorded configuration as ``woof domain`` emitted it, less its
#: comments and its advisory ``[fetch]`` table.
_RECORDED = """\
[experiment]
name = "GFS 3 km"
start_time = 2026-09-26T06:00:00
run_seconds = 86400.0
feedback = 0
smooth_option = 0
blend_width = 5
spec_bdy_width = 5
restart_interval_s = 3600.0

[projection]
map_proj = "lambert"
ref_lat = 35.45
ref_lon = -97.95000000000005
truelat1 = 25.45
truelat2 = 45.45
stand_lon = -97.95000000000005

[shared]
nz = 49
ztop = 20000.0
p_top = 10000.0
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
terrain_opt = 1
top_lid = false
wrf_rrtmg_compatibility = "wrf-rrtmg-4-4-to-rte-rrtmgp-v2"
wsm6_hail_opt = 0
map_proj = 1

[tiles]
mode = "auto"

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 206
ny = 204
time_step = 15
dx = 3000.0
specified = true
nested = false
history_interval_s = 3600.0
radt = 12.0
cu_physics = 0
cudt_minutes = 0.0
diff_6th_factor = 0.12
"""

#: The recorded run's settled step on the streamed road: 237.4 s at step 3
#: (547.2 and 296.0 s at steps 1 and 2), 1,190 tiles of 6x6 + halo 18.
_RECORDED_STREAMED_STEP_S = 237.382983

#: The redundancy auto may not exceed.  Written out rather than read from
#: the planner so that a change to the planner's limit is a change this
#: file has to be told about.
_LIMIT = 4.0


def test_the_limit_is_the_planners_own():
    assert autoplan.MAX_REDUNDANCY == _LIMIT


def _recorded():
    return experiment_from_text(_RECORDED, source="<recorded 206x204 GFS>")


def _card(free_gib: float) -> autoplan.Machine:
    """The recorded card: a 10 GiB RTX 3080 with ``free_gib`` free."""
    return autoplan.Machine(
        vram_bytes=int(free_gib * GIB), host_bytes=96 * 10 ** 9,
        name="NVIDIA GeForce RTX 3080", host_source="explicit",
        device_profile=preflight.DeviceLocalMemoryProfile(
            "NVIDIA GeForce RTX 3080", 68, 1536, 1024))


def _decide(exp, free_gib: float, machine: autoplan.Machine | None = None):
    machine = machine or _card(free_gib)
    options = streaming.options_for_domain(exp.root, exp.tiles)
    estimate = streaming.cold_single_domain_admission(
        exp, machine=machine, options=options)
    return streaming.cold_single_domain_decision(
        exp, machine=machine, cfg=exp.root.run, options=options,
        estimate=estimate), estimate


# ---------------------------------------------------------------------------
# the decision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("free_gib", [round(3.0 + 0.1 * i, 2)
                                      for i in range(17)] + [3.74, 3.99])
def test_auto_never_streams_past_the_redundancy_limit(free_gib):
    """At no free figure does auto hand back the tiling that ran for days.

    3.74 GiB free is the recorded forecast stage's reading (6x6 tiles,
    1,190 of them, 49.95x); 3.99 GiB is the door's (9x9, 529, 25.49x).
    """
    exp = _recorded()
    try:
        decision, _ = _decide(exp, free_gib)
    except autoplan.CannotPlan:
        return                        # a refusal is not a slow plan
    if decision.stream:
        cfg = exp.root.run
        redundancy = autoplan.redundancy(cfg.nx, cfg.ny, decision.tile_nx,
                                         decision.tile_ny, decision.halo)
        ntiles = (-(-cfg.nx // decision.tile_nx)) * (-(-cfg.ny // decision.tile_ny))
        assert redundancy <= _LIMIT, (
            f"auto streamed {ntiles} tiles of {decision.tile_nx}x"
            f"{decision.tile_ny} + halo {decision.halo} at {redundancy:.2f}x "
            f"redundancy with {free_gib} GiB free")


@pytest.mark.parametrize("free_gib", [3.74, 3.99, 4.0, 4.32])
def test_a_near_miss_runs_resident_inside_the_margin(free_gib):
    """The tiles-off run that finished, chosen by auto itself.

    The resident envelope of this domain is about 3.53 GiB.  From 3.74
    GiB free it misses the admission budget (free less the external
    margin) but fits what the card has, which is where ``woof go``
    already runs a tiles-off plan with a warning.  4.32 GiB, the door's
    other reading, admits it outright.
    """
    exp = _recorded()
    decision, estimate = _decide(exp, free_gib)
    assert not decision.stream, decision.explain()
    envelope = int(estimate.peak_envelope_bytes)
    assert decision.resident_bytes == envelope
    budget = int(free_gib * GIB) - preflight.EXTERNAL_MARGIN_BYTES
    if envelope > budget:
        assert "inside" in decision.reason and "free" in decision.reason
        declined = decision.detail["tile_road_declined"]
        assert decision.budget_bytes == int(free_gib * GIB)
        assert decision.detail["inside_external_margin_bytes"] == envelope - budget
        # At 3.74 GiB the only tiling that fit was 6x6, whose seam left
        # east relaxation-zone cells unrelaxed, so no legal tile fitted.
        # Since 06a93aa31 (the startup-speed merge) a tile costs less and
        # the legal 8x6 fits there, so 3.74 is declined on redundancy
        # (884 tiles, 38.87x) like the other readings.
        assert declined["redundancy"] > autoplan.MAX_REDUNDANCY
        # It names what it declined and what that would have cost.
        assert "tiles" in decision.reason and "x the necessary work" in decision.reason
        assert "s per step" in decision.reason or "s against" in decision.reason


@pytest.mark.parametrize("free_gib", [2.85, 3.2, 3.4])
def test_a_card_that_cannot_hold_it_is_refused_in_plain_words(free_gib):
    """1.15 GiB more held on the same card: no road is usable, so say so.

    Before the fix this configuration either streamed in 3x3 tiles at
    about 170x or refused with the planner's arithmetic alone.  The
    refusal now states both roads' numbers and what works.
    """
    exp = _recorded()
    with pytest.raises(autoplan.CannotPlan) as refused:
        _decide(exp, free_gib)
    text = str(refused.value)
    assert "206x204x49" in text
    assert "Resident it needs" in text and f"{free_gib:.2f} GiB free" in text
    assert "free at least" in text
    assert "smaller box" in text and "coarser grid spacing" in text
    assert "mode = 'off'" in text
    assert re.search(r"(?<![\w.])-\d", text) is None, text
    assert refused.value.resource in {"vram", "geometry"}
    assert refused.value.detail["short_bytes"] > 0


def test_an_explicit_max_redundancy_false_still_streams_the_slow_tiling():
    """The one way to get the slow tiling is to ask for it by name.

    ``[tiles] max_redundancy = false`` is the documented knob the refusal
    names; auto honours it, and the plan it returns states its tiling.
    At 3.98 GiB free that is 494 tiles at 24.31x (529 at 25.49x until A163
    measured the forecast margin at 1.13 of the subtotal instead of the
    plan's 1.15, which leaves each buffer more room).

    It never hands back a tiling that runs relaxation-zone cells
    unrelaxed.  At the recorded 3.74 GiB the only tiling that fitted was
    6x6, which put a seam 20 cells from the east edge, so there was no
    slow tiling to ask for.  Since 06a93aa31 (the startup-speed merge) a
    tile costs less and the legal 8x6 fits at 3.74 GiB, so asking for the
    slow tiling there streams it: 884 tiles at 38.87x.
    """
    exp = _recorded()
    exp = replace(exp, tiles=replace(exp.tiles, max_redundancy=False))
    decision, _ = _decide(exp, 3.98)
    assert decision.stream
    assert decision.redundancy > autoplan.MAX_REDUNDANCY
    assert f"{decision.ntiles:,} tiles" in decision.explain()
    assert (decision.ntiles, round(decision.redundancy, 2)) == (494, 24.31)
    decision, _ = _decide(exp, 3.74)
    assert decision.stream
    assert (decision.tile_nx, decision.tile_ny) == (8, 6)
    assert (decision.ntiles, round(decision.redundancy, 2)) == (884, 38.87)


# ---------------------------------------------------------------------------
# the pace
# ---------------------------------------------------------------------------


def test_the_high_end_quotes_back_the_run_it_was_fitted_from():
    """Quoted 0.45-1.2 s per step before the fix; it ran at 237 s.

    :data:`woof.core.pace.TILE_SECONDS_HIGH` is fitted from this run, so
    this is an ARITHMETIC check and cannot judge the model: it fails when
    the column term, the redundancy scale or the tiles the rates already
    carry stop adding up to the step the constant was taken from.  The
    checks that can fail on the model are the independent runs below.
    """
    exp, envelope = _recorded_streamed_plan()
    estimate = pace.estimate_pace(exp, streamed=envelope, machine=_card(3.74))
    measured = _RECORDED_STREAMED_STEP_S
    assert estimate.seconds_per_step_high == pytest.approx(measured, rel=5e-3), (
        f"quoted {estimate.seconds_per_step_low:.3g}-"
        f"{estimate.seconds_per_step_high:.3g} s per step; it ran at "
        f"{measured:.1f} s")
    assert estimate.seconds_per_step_low <= measured


#: The run the per-tile LOW end is fitted from: the recorded configuration
#: on a box two columns wider, 208x204x49, with 3.97 GiB of an idle 16 GiB
#: RTX 5070 Ti under Linux free.  Auto ran it resident and named 8x9 tiles
#: with halo 18 (598 of them, 27.90x) as the only tiling that fit; streamed
#: at that tiling on request, its steps from the second on (102 of them)
#: had a median of 11,123.95 ms, a fastest of 11,082.08 ms and a mean of
#: 11,435.92 ms with the radiation steps.
_LOW_FIT_MEDIAN_S = 11.12395
_LOW_FIT_FASTEST_S = 11.082077
_LOW_FIT_MEAN_S = 11.43592


def _wide():
    return experiment_from_text(_RECORDED.replace("nx = 206", "nx = 208"),
                                source="<recorded 208x204 GFS>")


def _card_5070ti(free_gib: float) -> autoplan.Machine:
    """A 16 GiB RTX 5070 Ti under Linux with ``free_gib`` free."""
    return autoplan.Machine(
        vram_bytes=int(free_gib * GIB), host_bytes=32 * 10 ** 9,
        name="NVIDIA GeForce RTX 5070 Ti", host_source="explicit",
        device_profile=preflight.DeviceLocalMemoryProfile(
            "NVIDIA GeForce RTX 5070 Ti", 70, 1536, 1024))


def test_the_low_end_quotes_back_the_run_it_was_fitted_from():
    """Quoted 17-121 s per step before the refit; it ran at 11.1 s.

    :data:`woof.core.pace.TILE_SECONDS_LOW` is this run's step less the
    fast-end column term, over the tiles past nine, rounded down, so this
    is an ARITHMETIC check: it fails when the constant stops being that
    fit, or the bracket stops holding the run's steps.  The per-tile term
    of the tile sweep (18 ms, fitted beside a smaller column term of its
    own) added to this column term quoted the fast end at 17 s.
    """
    exp = _wide()
    card = _card_5070ti(3.97)
    envelope = streaming.streamed_envelope(
        exp.root.run, streaming.options_for_domain(exp.root, exp.tiles),
        machine=card,
        decision=streaming.StreamingDecision(True, "measured", 8, 9, 1, 18))
    estimate = pace.estimate_pace(exp, streamed=envelope, machine=card)
    assert (estimate.tiles, round(estimate.redundancy, 2)) == (598, 27.90)
    column_low = (estimate.seconds_per_step_low
                  - estimate.tile_seconds_per_step_low)
    fitted = ((_LOW_FIT_MEDIAN_S - column_low)
              / (598 - pace.STREAMED_REFERENCE_TILES))
    assert pace.TILE_SECONDS_LOW <= fitted < 1.15 * pace.TILE_SECONDS_LOW, (
        f"the run fits {fitted * 1e3:.2f} ms a tile against "
        f"{pace.TILE_SECONDS_LOW * 1e3:.2f} ms charged")
    low, high = estimate.seconds_per_step_low, estimate.seconds_per_step_high
    assert low <= _LOW_FIT_FASTEST_S and _LOW_FIT_MEAN_S <= high, (
        f"quoted {low:.3g}-{high:.3g} s per step; it ran at "
        f"{_LOW_FIT_FASTEST_S:.2f}-{_LOW_FIT_MEAN_S:.2f} s")
    # The forecast stage's tiles line prices a tiling with the same model,
    # without the bus floor, which this tiling does not reach.
    assert pace.tiling_step_seconds(
        exp.root.run, tile_nx=8, tile_ny=9, halo=18) == pytest.approx(
            (low, high), rel=1e-9)


def test_the_near_miss_line_quotes_a_step_that_holds_the_measured_one():
    """The sentence the forecast stage printed for the low end's run.

    With 3.89 GiB free auto runs the 208x204 domain resident inside the
    external margin and names the tiling it declined with what a step at
    that tiling costs.  That tiling is the one the low end's run streamed,
    so the step it names has to hold the step that run measured.  (3.97
    GiB on the plan's 1.15 margin; 3.95 after A163 measured it at 1.13 of
    the subtotal.  Since 06a93aa31 a tile costs less, and read at 0.01 GiB
    steps from 3.60 to 3.99 the declined tiling is 8x9 from 3.88 to 3.90,
    6x12 above and 6x9 below.  The 2.9.2 RRTMGP flux rows moved the
    resident envelope, and with it that band, to 3.90 through 3.92.)

    Price the card 320 MiB above the resident envelope, leaving a 192 MiB
    shortfall against the external 512 MiB margin.  This keeps the fixture
    at the measured near miss as the envelope moves without searching for
    a card whose answer already matches the assertion below.
    """
    exp = _wide()
    options = streaming.options_for_domain(exp.root, exp.tiles)
    estimate = streaming.cold_single_domain_admission(
        exp, machine=_card_5070ti(16), options=options)
    free = (estimate.peak_envelope_bytes + 320 * 1024 ** 2) / GIB
    decision, _ = _decide(exp, free, _card_5070ti(free))
    assert not decision.stream, decision.explain()
    declined = decision.detail["tile_road_declined"]
    assert "8x9 tiles (halo 18, 598 of them)" in decision.reason, (
        declined, decision.reason)
    quoted = re.search(r"a step at that tiling costs roughly "
                       r"([\d.,]+)-([\d.,]+) s", decision.reason)
    assert quoted, decision.reason
    low, high = (float(v.replace(",", "")) for v in quoted.groups())
    assert low <= _LOW_FIT_MEDIAN_S <= high, decision.reason


def test_the_tiling_the_rates_were_measured_on_is_quoted_the_rates():
    """No tile is counted twice.

    The streamed rows are whole steps of the HRRR receipt's 3x3 tiling
    (1200x900, tiles 400x300, halo 16, 1.1952x), so its nine tiles' cost
    is inside them.  Priced at that tiling, the model has to give back the
    rows themselves; charging the per-tile term for all nine added
    0.16-1.5 s to a 6.5-22.6 s step.
    """
    cfg = replace(_recorded().root.run, nx=1200, ny=900)
    tiles, redundancy = streaming.tiling_shape(cfg, 400, 300, 16)
    assert (tiles, round(redundancy, 4)) == (
        pace.STREAMED_REFERENCE_TILES, pace.STREAMED_REFERENCE_REDUNDANCY)
    rate = pace.step_rate(autoplan.rung_of(cfg), "streamed")
    quoted = pace.tiling_step_seconds(cfg, tile_nx=400, tile_ny=300, halo=16)
    assert quoted == pytest.approx(rate.seconds_per_step(1200 * 900, 49),
                                   rel=1e-9)
    assert pace.tile_overhead_seconds(9) == (0.0, 0.0)
    assert pace.tile_overhead_seconds(10) == (
        pace.TILE_SECONDS_LOW, pace.TILE_SECONDS_HIGH)


#: Streamed steps of the recorded domain measured on runs that took no part
#: in either per-tile fit, each settled at the tiling it states:
#: (card, its multiprocessors, tile_nx, tile_ny, tiles, redundancy,
#: settled seconds per 15 s step).
#:
#: * RTX 5070 Ti under Linux, ``[tiles] max_redundancy = false`` with 3.98
#:   GiB free (test host 4, 2026-09-27): 598 tiles of 9x8 + halo 18, one
#:   buffer, settled at 20.18 s a step (33.55 s the first).  Auto on the
#:   same card ran the forecast resident at a median 0.053 s a step.
_INDEPENDENT_STEPS = [
    pytest.param("NVIDIA GeForce RTX 5070 Ti", 70, 9, 8, 598, 28.18,
                 20.183459, id="5070ti-linux-598-tiles"),
]


@pytest.mark.parametrize(
    "card_name,multiprocessors,tile_nx,tile_ny,tiles,redundancy,measured",
    _INDEPENDENT_STEPS)
def test_a_streamed_step_no_fit_used_lies_inside_the_bracket(
        card_name, multiprocessors, tile_nx, tile_ny, tiles, redundancy,
        measured):
    """The checks that can fail on the model, not only on its arithmetic.

    None of these runs fed :data:`woof.core.pace.TILE_SECONDS_LOW` or
    :data:`woof.core.pace.TILE_SECONDS_HIGH`, so a measured step outside
    the quoted bracket is the model being wrong, and the quote may not be
    so loose that its fast end is under half the step.
    """
    exp = _recorded()
    card = autoplan.Machine(
        vram_bytes=int(3.98 * GIB), host_bytes=32 * 10 ** 9,
        name=card_name, host_source="explicit",
        device_profile=preflight.DeviceLocalMemoryProfile(
            card_name, multiprocessors, 1536, 1024))
    envelope = streaming.streamed_envelope(
        exp.root.run, streaming.options_for_domain(exp.root, exp.tiles),
        machine=card,
        decision=streaming.StreamingDecision(
            True, "measured", tile_nx, tile_ny, 1, 18))
    estimate = pace.estimate_pace(exp, streamed=envelope, machine=card)
    assert (estimate.tiles, round(estimate.redundancy, 2)) == (tiles,
                                                               redundancy)
    assert (estimate.seconds_per_step_low <= measured
            <= estimate.seconds_per_step_high), (
        f"quoted {estimate.seconds_per_step_low:.3g}-"
        f"{estimate.seconds_per_step_high:.3g} s per step; it ran at "
        f"{measured:.1f} s")
    assert estimate.seconds_per_step_low >= measured / 2.0


def _recorded_streamed_plan():
    """The plan the recorded forecast stage ran: 6x6 + halo 18, one buffer."""
    exp = _recorded()
    decision = streaming.StreamingDecision(True, "recorded", 6, 6, 1, 18)
    envelope = streaming.streamed_envelope(
        exp.root.run, streaming.options_for_domain(exp.root, exp.tiles),
        machine=_card(3.74), decision=decision)
    return exp, envelope


def test_the_recorded_streamed_plan_states_its_tiles_and_their_cost():
    exp, envelope = _recorded_streamed_plan()
    assert (envelope.ntiles, round(envelope.redundancy, 2)) == (1190, 49.95)
    estimate = pace.estimate_pace(exp, streamed=envelope, machine=_card(3.74))
    assert estimate.tiles == 1190
    # The tiles, not the columns, are most of this step: the reason a
    # column rate alone quoted it at a second.
    assert estimate.tile_seconds_per_step_high > estimate.seconds_per_step_high / 2
    assert "1,190 tiles at 49.95x redundancy" in estimate.sentence()
    assert estimate.to_json()["tiles"] == 1190


def test_the_per_tile_term_is_nothing_on_the_resident_road():
    exp = _recorded()
    estimate = pace.estimate_pace(exp, streamed=None, machine=_card(8.0))
    assert estimate.road == "resident"
    assert estimate.tiles is None and estimate.tile_seconds_per_step_high == 0.0


# ---------------------------------------------------------------------------
# every surface states the tiling
# ---------------------------------------------------------------------------


_PINNED = """\
[tiles]
mode = "on"
tile_nx = 48
tile_ny = 40
store = "host"
vram_budget_bytes = 6442450944
host_budget_bytes = 34359738368
"""


def test_the_estimate_document_states_tile_count_and_redundancy(tmp_path):
    import woof.runplan as runplan_module
    from test_runplan_tiles import _config, _plan

    document = json.loads(json.dumps(runplan_module.estimate_plan(
        _plan(tmp_path, _config(tmp_path, tiles=_PINNED)))))
    section = document["vram"]["streamed"]
    assert section["tile_nx"] == 48 and section["tile_ny"] == 40
    assert section["ntiles"] >= 1 and section["redundancy"] > 1.0
    expected = document["expected_pace"]
    assert expected["tiles"] == section["ntiles"]
    assert expected["redundancy"] == pytest.approx(section["redundancy"], rel=1e-3)
    assert f"{section['ntiles']:,} tiles at" in expected["sentence"]


def test_the_review_verdict_and_the_run_log_line_state_the_tiling(tmp_path):
    from woof.experiment import load_experiment
    from test_runplan_tiles import _config

    exp = load_experiment(_config(tmp_path, tiles=_PINNED))
    cfg = exp.root.run
    options = streaming.options_for_domain(exp.root, exp.tiles)
    decision = streaming.decide(cfg, options)
    ntiles, redundancy = streaming.tiling_shape(cfg, 48, 40, decision.halo)
    assert (decision.ntiles, decision.redundancy) == (ntiles, redundancy)
    shape = f"{ntiles:,} tiles at {redundancy:.2f}x redundancy"
    assert shape in decision.explain()
    receipt = streaming.streaming_receipt(exp.tiles, {1: decision})
    assert f"tile 48x40 + halo {decision.halo}, {shape}" in receipt["summary"]
    assert receipt["domains"]["1"]["ntiles"] == ntiles
    phases = preflight.estimate_phases(exp, source=None)
    assert shape in phases.verdict(8 * GIB)
