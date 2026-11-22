"""The memory price carries the legacy RRTMG width the call takes on the card.

Breakage (D2, 2026-11-12): ``woof check --met-em`` priced the 150 x 150 x 49
OKC met_em case (Thompson, YSU, legacy RRTMG) at 6.29 GiB, and the forecast
reached 10.40 GiB on an RTX 5090.  The live call widens its engine chunk to
four waves when the card has room (``RRTMGLegacyRadiation._wide_chunk``); the
estimator priced one wave: 2.00 GiB of radiation workspace against 6.0 GiB
live in the pool.  The estimator now asks the same rule with the card's free
bytes, so the price is the width the run takes.
"""
from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path

from woof.config import RunConfig
from woof.core import preflight as pf
from woof.core import rrtmg_legacy as legacy
from woof.core.streaming import planner_machine
from woof.experiment import experiment_from_run_config

GIB = 1 << 30
ROOT = Path(__file__).resolve().parents[1]


def _okc_shape():
    """The OKC case's shape and physics: 150 x 150 x 49, mp 8, YSU, RRTMG 4/4."""
    cfg = RunConfig(nx=150, ny=150, nz=49, dx=3000.0, dy=3000.0,
                    ztop=20000.0, dt=12.0, run_seconds=21600.0,
                    moist=True, moist_cq=True, mp_physics=8,
                    bl_pbl_physics=1, ra_lw_physics=4, ra_sw_physics=4,
                    ra_rrtmg_variant="rrtmg_legacy")
    return experiment_from_run_config(cfg, datetime(2024, 5, 25, 18))


def _phases(exp, free_gib, alloc_budget_bytes=None):
    profile = pf.card_local_memory_profile(None)
    machine = (None if free_gib is None else planner_machine(
        vram_bytes=int(free_gib * GIB), name="test card", device_profile=profile))
    leg = {} if alloc_budget_bytes is None else {"alloc_budget_bytes": alloc_budget_bytes}
    return pf.estimate_phases(exp, source=None, forcing_interval_seconds=3600.0,
                              profile=profile, machine=machine, **leg)


def _legacy_call(exp, width):
    dc = exp.domains[0]
    profile = pf.card_local_memory_profile(None)
    return legacy.legacy_radiation_vram_bytes(
        ncol=dc.run.nx * dc.run.ny, nz=dc.run.nz, p_top=exp.vertical.p_top,
        column_chunk=width, resident_threads=profile.resident_thread_capacity)


def test_the_okc_price_is_above_the_peak_the_run_measured():
    """The run reached 10.40 GiB (NVML, this process) on a 5090 with 30.9 GiB
    free; the check priced 6.29.  The price must sit above it, a few % up."""
    envelope = _phases(_okc_shape(), 30.9).forecast_envelope_bytes
    assert 10.40 * GIB < envelope < 1.10 * 10.40 * GIB


def test_a_card_with_room_is_priced_at_the_four_wave_width_the_call_takes():
    exp = _okc_shape()
    base = legacy.legacy_saturation_chunk(
        pf.card_local_memory_profile(None).resident_thread_capacity)
    assert base == 2048                      # one wave on the reference card
    wide = _phases(exp, 30.9).forecast
    assert wide.legacy_radiation_widths == (4 * base,)
    assert wide.workspace_bytes == _legacy_call(exp, 4 * base)
    # The one-wave price was 2.0 GiB; the four-wave call holds 6.2 GiB, and
    # the run measured 6.0 GiB of radiation transient in the pool.
    assert round(_legacy_call(exp, None) / GIB, 1) == 2.0
    assert round(wide.workspace_bytes / GIB, 1) == 6.2
    # The whole-process envelope now sits above the measured 10.40 GiB peak.
    assert 10.40 * GIB < wide.peak_envelope_bytes < 11.5 * GIB


def test_the_price_steps_down_to_the_width_that_fits_and_never_past_the_budget():
    exp = _okc_shape()
    two = _phases(exp, 9.0)
    assert two.forecast.legacy_radiation_widths == (4096,)
    assert two.forecast.legacy_radiation_take_widths == ()   # the call takes that too
    one = _phases(exp, 7.0)
    assert one.forecast.legacy_radiation_widths == ()
    assert one.forecast.workspace_bytes == _legacy_call(exp, None)
    # Four waves priced in full are over an 11.4 GiB card's whole-process
    # budget, and two fit: the forecast needs the two-wave price.
    assert _phases(exp, 11.4).forecast.legacy_radiation_widths == (4096,)
    # The width never prices a run over the card it was chosen for when one
    # wave fits it.
    for free_gib in (7.0, 9.0, 11.4, 12.0, 16.0, 30.9):
        phases = _phases(exp, free_gib)
        assert phases.forecast_envelope_bytes <= free_gib * GIB - pf.EXTERNAL_MARGIN_BYTES


def test_where_the_price_steps_down_the_width_the_call_still_takes_is_recorded():
    """The call does not step down with the price.  Measured on an
    RTX 5090 (2026-11-12, a ballast leaving 11.40 GiB free): the OKC run
    took 8,192 columns and the process reached 10.40 GiB, where the
    stepped-down price said 7.59 GiB with 3.31 to spare.  With 10.60 GiB
    free it reached the same 10.40 GiB and left 0.20."""
    exp = _okc_shape()
    tight = _phases(exp, 11.4).forecast
    assert tight.legacy_radiation_widths == (4096,)
    assert tight.legacy_radiation_take_widths == (8192,)
    assert tight.peak_envelope_bytes < 10.40 * GIB          # the need, which fits
    assert 10.40 * GIB < tight.legacy_radiation_take_envelope_bytes < 1.10 * 10.40 * GIB
    # The take's envelope is the price the same call has on a card with room.
    assert (tight.legacy_radiation_take_envelope_bytes
            == _phases(exp, 30.9).forecast_envelope_bytes)
    # One wave priced, two taken.
    small = _phases(exp, 7.8).forecast
    assert small.legacy_radiation_widths == ()
    assert small.legacy_radiation_take_widths == (4096,)
    # No take is recorded where the price already is the take, or where
    # nothing wider than one wave can be taken.
    assert _phases(exp, 30.9).forecast.legacy_radiation_take_widths == ()
    assert _phases(exp, 6.0).forecast.legacy_radiation_take_widths == ()
    note = pf.legacy_take_note(tight)
    assert "d01 8192 columns per engine chunk where 4096 are priced" in note
    assert f"{tight.legacy_radiation_take_envelope_bytes / GIB:.2f} GiB" in note
    assert "d01 4096 columns per engine chunk where one wave is priced" in pf.legacy_take_note(small)
    assert pf.legacy_take_note(_phases(exp, 30.9).forecast) is None


def test_the_free_figure_a_door_read_through_its_own_context_still_finds_the_take():
    """With 10.60 GiB free on the card, `woof check` read 10.11 GiB: its
    own CUDA context held the rest, and the forecast found it free again
    and took four waves (10.40 GiB).  Asked with 10.11 alone, the call's
    rule answers two waves."""
    exp = _okc_shape()
    dc = exp.domains[0]
    profile = pf.card_local_memory_profile(None)
    seen = _phases(exp, 10.11).forecast
    beside = seen.envelope_intercept_bytes + seen.resident_bytes
    terms = dict(ncol=dc.run.nx * dc.run.ny, nz=dc.run.nz, p_top=exp.vertical.p_top,
                 ncol_day=None, base=2048,
                 resident_threads=profile.resident_thread_capacity)
    assert legacy.legacy_wide_chunk_width(
        free_bytes=int(10.11 * GIB) - beside, **terms) == 4096
    assert legacy.legacy_wide_chunk_width(
        free_bytes=int(10.60 * GIB) - beside, **terms) == 8192
    assert seen.legacy_radiation_take_widths == (8192,)
    assert seen.legacy_radiation_take_envelope_bytes > 10.40 * GIB


def test_the_width_never_fails_an_allocation_leg_that_one_wave_passes():
    """`woof check` has a second budget, its allocation leg's.  Stepped down
    on the envelope alone, the full CONUS HRRR recipe on a 96 GB card was
    priced at two waves, 89.76 GiB against that leg's 88.97 GiB, and the
    check refused a forecast `woof go` admits
    (test_check_chained_forcing_slots, which the first D2 change broke)."""
    exp = _okc_shape()
    four = _phases(exp, 30.9).forecast.alloc_estimate_bytes
    two = _phases(exp, 9.0).forecast.alloc_estimate_bytes
    one = _phases(exp, None).forecast.alloc_estimate_bytes
    assert one < two < four
    # The card has room for four waves; the leg's budget decides the price.
    roomy = _phases(exp, 30.9, four).forecast
    assert roomy.legacy_radiation_widths == (8192,)
    assert roomy.legacy_radiation_take_widths == ()
    under_four = _phases(exp, 30.9, four - 1).forecast
    assert under_four.legacy_radiation_widths == (4096,)
    assert under_four.alloc_estimate_bytes == two
    assert under_four.legacy_radiation_take_widths == (8192,)
    under_two = _phases(exp, 30.9, two - 1).forecast
    assert under_two.legacy_radiation_widths == ()
    assert under_two.alloc_estimate_bytes == one
    assert under_two.legacy_radiation_take_widths == (8192,)
    # One wave over the leg's budget too: the one-wave price, refused as
    # before, and nothing said about a take.
    refused = _phases(exp, 30.9, one - 1).forecast
    assert refused.legacy_radiation_widths == ()
    assert refused.alloc_estimate_bytes == one
    assert refused.legacy_radiation_take_widths == ()


def test_the_note_names_the_free_memory_where_the_take_is_over_it():
    """A process cannot hold more than the card has free: the note read "can
    come to hold 95.67 GiB" beside "94.42 GiB free" on the HRRR recipe."""
    tight = _phases(_okc_shape(), 11.4).forecast
    take = tight.legacy_radiation_take_envelope_bytes
    under = pf.legacy_take_note(tight, take + 1)
    assert f"can come to hold {take / GIB:.2f} GiB" in under
    assert under == pf.legacy_take_note(tight)
    over = pf.legacy_take_note(tight, int(10.11 * GIB))
    assert "can come to fill the 10.11 GiB this card has free" in over
    assert "place no other job beside it" in over
    assert "can come to hold" not in over
    assert "d01 8192 columns per engine chunk where 4096 are priced" in over


def test_no_card_keeps_the_one_wave_price():
    exp = _okc_shape()
    plain = _phases(exp, None).forecast
    assert plain.legacy_radiation_widths == ()
    assert plain.workspace_bytes == _legacy_call(exp, None)


def test_the_live_call_and_the_price_ask_one_rule():
    """``_wide_chunk`` delegates to ``legacy_wide_chunk_width``; the loop
    that picked the width is not written twice."""
    tree = ast.parse((ROOT / "woof/core/rrtmg_legacy.py").read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "_wide_chunk")
    called = {node.func.id for node in ast.walk(method)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "legacy_wide_chunk_width" in called
    assert not any(isinstance(node, ast.For) for node in ast.walk(method))
    # The rule itself: widest first, margin kept, None when one wave covers.
    kw = dict(nz=49, p_top=5000.0, ncol_day=None, base=2048)
    need4 = legacy.legacy_radiation_vram_bytes(ncol=22500, nz=49, p_top=5000.0,
                                                column_chunk=8192)
    assert legacy.legacy_wide_chunk_width(
        ncol=22500, free_bytes=need4 + (1 << 30), **kw) == 8192
    assert legacy.legacy_wide_chunk_width(
        ncol=22500, free_bytes=need4 + (1 << 30) - 1, **kw) == 4096
    assert legacy.legacy_wide_chunk_width(ncol=2048, free_bytes=1 << 40, **kw) is None
    assert legacy.legacy_wide_chunk_width(ncol=22500, free_bytes=None, **kw) is None
