"""One price and one ``[tiles]`` decision for the offline child, on both doors.

The failure these pin: a 138x138x49 child the plan review admitted at
2.90 GiB against 7.32 GiB free was refused at run start with "no tile fits
in 3.49 GiB of VRAM", because the runner asked the tile planner AFTER it
had filled the card and with no estimate, so the planner charged the whole
rung's fixed cost against what the process had left.  The fix is one
function both doors call (:mod:`woof.downscale_pricing`), on a machine
captured before any device allocation.
"""

from argparse import Namespace
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from woof import downscale_pricing
from woof.config import RunConfig, load_config, load_streaming_options
from woof.core import streaming
from woof.downscale import (
    _derive_child_run_config, _render_child_toml, build_child_eta_levels)
from woof.offline_child import OfflineChildContractError
from woof.offline_child_run import (
    checkpoint_schedule, child_cadence, child_run_name)
from test_downscale_cli import _PARENT_CONFIG
from test_offline_child import _history

GIB = 1024 ** 3

#: The configuration of the child that was refused: a 4 km, 49-level
#: Morrison child under RTE+RRTMGP with Noah, MYNN surface layer and YSU,
#: derived at ratio 3 from a 12 km parent, [tiles] mode auto.  Only the
#: keys the memory model and the [tiles] decision read are spelled out;
#: everything else is the RunConfig default, as it was in that run.
_REFUSED_CHILD = dict(
    nx=138, ny=138, nz=49, dx=4000.0, dy=4000.0, dt=20.0, ztop=20000.0,
    grid_id=2, specified=True, nested=False, run_seconds=21600.0,
    output_interval_s=3600.0, restart_interval_s=3600.0,
    mp_physics=10, morr_rimed_ice=1, ra_physics=0, ra_lw_physics=4,
    ra_sw_physics=4, ra_rrtmg_variant="rte-rrtmgp", radt=12.0,
    radt_minutes=12.0, o3input=2, bl_pbl_physics=1, sf_sfclay_physics=91,
    sf_surface_physics=2, num_soil_layers=4, cu_physics=1,
    cudt_minutes=5.0, hybrid_opt=2, etac=0.2, hypsometric_opt=2,
    moist=True, terrain_opt=1, map_proj=1, time_step_sound=4,
    spec_bdy_width=5, spec_zone=1, relax_zone=4, damp_opt=3,
    w_damping=1, diff_6th_opt=2, km_opt=4, nwp_diagnostics=1,
    use_mp_re=1, icloud=1,
)


def _refused_child_cfg() -> RunConfig:
    return RunConfig(**_REFUSED_CHILD)


def _fake_machine(free_gib: float):
    from tilestream.autoplan import Machine

    return Machine(vram_bytes=int(free_gib * GIB), host_bytes=64 * GIB,
                   name="fixture card", host_source="explicit")


def test_the_admitted_child_is_resident_on_the_shared_path_and_refused_on_the_old_one(
        monkeypatch):
    """The red/green pair of the whole fix.

    Same child, same 6.2 GiB card.  Through the shared function the
    configured envelope is judged against the whole-process budget and the
    child is resident, without the tile planner ever being consulted.
    Through the old path -- ``decide`` with that machine and no estimate,
    which is what the runner used to do after it had filled the card --
    the tile table's per-process fixed cost is charged and the same child
    is refused.
    """
    from tilestream import autoplan

    cfg = _refused_child_cfg()
    tiles = streaming.StreamingOptions(mode="auto")
    machine = _fake_machine(6.2)

    def never(*args, **kwargs):
        raise AssertionError("the tile planner was consulted for an "
                             "admitted resident child")

    with monkeypatch.context() as patch:
        patch.setattr(autoplan, "plan", never)
        pricing = downscale_pricing.price_child(
            cfg, tiles, machine=machine,
            basis=downscale_pricing.MEASURED_BASIS)
    assert pricing.decision is not None and not pricing.decision.stream
    assert "configured resident envelope fits" in pricing.decision.reason
    assert pricing.mode == "resident"
    assert pricing.peak_envelope_bytes is not None
    assert pricing.peak_envelope_bytes <= pricing.budget_bytes
    assert pricing.machine_free_bytes == machine.vram_bytes
    # The options carry the admission context, so a later decide on THEM
    # answers from the same envelope instead of the tile table.
    assert pricing.options.resident_context is not None
    entry = pricing.plan_entry()
    assert entry["mode"] == "resident" and entry["tile"] is None
    assert entry["basis"] == "measured-local"
    assert entry["budget_bytes"] == pricing.budget_bytes

    # The old order: no estimate, no admission context, the planner alone.
    with pytest.raises(autoplan.CannotPlan) as refused:
        streaming.decide(cfg, tiles, machine=machine)
    assert refused.value.resource == "vram"
    assert "no tile fits" in str(refused.value)


def test_off_and_pinned_options_need_no_card():
    cfg = _refused_child_cfg()
    assert not downscale_pricing.needs_machine(None)
    assert not downscale_pricing.needs_machine(streaming.OFF)
    assert downscale_pricing.needs_machine(streaming.StreamingOptions(mode="auto"))
    assert downscale_pricing.needs_machine(streaming.StreamingOptions(mode="on"))
    assert not downscale_pricing.needs_machine(
        streaming.StreamingOptions(mode="on", tile_nx=64, tile_ny=64))
    pricing = downscale_pricing.price_child(
        cfg, None, machine=None, basis=downscale_pricing.DECLARED_BASIS,
        vram_gib=24.0)
    assert pricing.mode == "resident"
    assert pricing.decision.reason == "[tiles] mode = 'off'"
    assert pricing.peak_envelope_bytes > 0


def test_a_review_without_a_card_leaves_the_decision_to_the_run():
    """No card to plan against is said, not guessed."""
    pricing = downscale_pricing.price_child(
        _refused_child_cfg(), streaming.StreamingOptions(mode="auto"),
        machine=None, basis=downscale_pricing.DECLARED_BASIS, vram_gib=24.0)
    assert pricing.decision is None and pricing.mode is None
    assert "needs a card" in pricing.plan_entry()["why"]


def test_a_vram_refusal_carries_the_measured_figure_and_the_way_out(monkeypatch):
    from tilestream import autoplan

    def refuse(*args, **kwargs):
        raise autoplan.CannotPlan("no tile fits in 1.00 GiB of VRAM", "vram",
                                  {"smallest": 1})

    monkeypatch.setattr(streaming, "decide", refuse)
    with pytest.raises(autoplan.CannotPlan) as refused:
        downscale_pricing.price_child(
            _refused_child_cfg(), streaming.StreamingOptions(mode="auto"),
            machine=_fake_machine(1.5),
            basis=downscale_pricing.MEASURED_BASIS)
    message = str(refused.value)
    assert "measured 1.50 GiB free" in message
    assert "before anything was interpolated or allocated on the device" in message
    assert "--child-size" in message and "--child-levels" in message
    assert refused.value.detail["machine_free_bytes"] == int(1.5 * GIB)
    assert refused.value.detail["basis"] == "measured-local"

    # A DECLARED card was never measured, so "free the card" is not a way
    # out of its refusal; pricing the real card is (--auto-vram measures
    # it, --card / --vram-gib declare a larger one).
    with pytest.raises(autoplan.CannotPlan) as refused:
        downscale_pricing.price_child(
            _refused_child_cfg(), streaming.StreamingOptions(mode="auto"),
            machine=_fake_machine(1.5),
            basis=downscale_pricing.DECLARED_BASIS)
    message = str(refused.value)
    assert "declared card is assumed to present 1.50 GiB free" in message
    assert "--auto-vram" in message and "--card" in message
    assert "--vram-gib" in message
    assert "--child-size" in message and "--child-levels" in message
    assert "freeing the card" not in message
    assert refused.value.detail["basis"] == downscale_pricing.DECLARED_BASIS

    # A geometry refusal names its own way out and is passed through.
    def too_small(*args, **kwargs):
        raise autoplan.CannotPlan("cannot be tiled at all", "geometry")

    monkeypatch.setattr(streaming, "decide", too_small)
    with pytest.raises(autoplan.CannotPlan) as refused:
        downscale_pricing.price_child(
            _refused_child_cfg(), streaming.StreamingOptions(mode="on"),
            machine=_fake_machine(24.0),
            basis=downscale_pricing.DECLARED_BASIS)
    assert str(refused.value) == "cannot be tiled at all"


def test_the_estimator_is_handed_the_capacity_and_the_profile_the_door_holds(
        monkeypatch):
    """What ``_price_child_config`` promised, kept on the shared function.

    The Noah-MP lane retired a refusal by handing the estimator the profile
    the sizing probe read; a route that dropped it would send a Noah-MP
    child on a measured card back into "a declared card that is not in
    this machine".  The capacity rides along on every route too, exactly
    as the fitted sizing passes it, so the review's number is the fit's.
    """
    from woof.core import preflight as pf

    seen = []

    def estimate(exp, **kwargs):
        seen.append(kwargs)
        raise RuntimeError("priced")

    monkeypatch.setattr(pf, "estimate_experiment", estimate)
    measured = object()
    cfg = _refused_child_cfg()
    pricing = downscale_pricing.price_child(
        cfg, None, machine=None, basis=downscale_pricing.MEASURED_BASIS,
        vram_gib=10.0, profile=measured, forcing_intervals=6)
    assert pricing.estimate is None
    assert pricing.pricing_error == "RuntimeError: priced"
    assert pricing.plan_entry()["pricing_error"] == "RuntimeError: priced"
    downscale_pricing.price_child(
        cfg, None, machine=None, basis=downscale_pricing.DECLARED_BASIS,
        vram_gib=16.0)
    assert seen == [
        {"forcing_intervals": 6, "vram_gib": 10.0, "profile": measured},
        {"forcing_intervals": None, "vram_gib": 16.0, "profile": None}]


# ---------------------------------------------------------------------------
# the runner's order
# ---------------------------------------------------------------------------


class _Sentinel(Exception):
    pass


def _stub_cupy(monkeypatch) -> None:
    """A ``cupy`` the runner can import, INSTALLED WITHOUT IMPORTING ONE.

    The ordering test never reaches the device: it stops at the first
    parent-frame read.

    THE IMPORT THAT USED TO GUARD THIS RETIRED THE WHOLE MODULE.  It read
    ``try: import cupy ... except ImportError``, and tests/conftest.py
    marks a module ``gpu`` for any ``import cupy`` its AST can see outside
    a test function -- deliberately, because a helper's device use belongs
    to callers the AST cannot enumerate.  This helper is such a caller, so
    all eleven tests in this file were deselected from every
    ``-m "not gpu"`` leg: the one suite that pins "the plan review and the
    run door price the offline child from one function" ran nowhere on
    CPU.  Nothing here needs a real CuPy, so none is asked for; a real one
    already imported by something else is left alone, because replacing a
    live module under a process that is using it is worse than the stub.
    """
    live = sys.modules.get("cupy")
    if live is not None and getattr(live, "__file__", None) is not None:
        return
    fake = types.ModuleType("cupy")
    fake.__dict__.update({name: value for name, value in vars(np).items()
                          if not name.startswith("__")})
    fake.ndarray = np.ndarray
    fake.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(
            memGetInfo=lambda: (6 * GIB, 10 * GIB),
            deviceSynchronize=lambda: None),
        Device=lambda *args, **kwargs: None,
        Stream=types.SimpleNamespace(null=None))
    fake.get_default_memory_pool = lambda: None
    fake.asnumpy = np.asarray
    monkeypatch.setitem(sys.modules, "cupy", fake)


def _runnable_child(tmp_path: Path, *, tiles_mode: str | None = "auto") -> Path:
    """A 12x10 child on its own four-level ladder over the two-level fixture.

    ``tiles_mode=None`` writes no ``[tiles]`` block at all, which is what a
    run with no ``--tiles`` gets: the TUI and CLI default path.
    """
    parent = {"nx": 20, "ny": 18, "dx": 1000.0, "dy": 1000.0}
    merged = _derive_child_run_config(
        _PARENT_CONFIG, parent=parent, ratio=1, child_nx=12, child_ny=10,
        run_seconds=600.0, output_interval_s=300.0,
        child_eta_levels=build_child_eta_levels(4, stretch=2.5))
    merged["restart_interval_s"] = 300.0
    path = tmp_path / "child.toml"
    path.write_text(_render_child_toml(merged, tiles_mode=tiles_mode),
                    encoding="utf-8", newline="\n")
    return path


def test_the_runner_decides_before_it_reads_a_parent_frame(tmp_path, monkeypatch):
    """The decision is taken cold, before the first byte of preprocessing.

    ``interpolate_parent_initial_state`` is the first thing that spends
    minutes and memory on the parent archive; the decision must already
    have been taken, on a machine captured with nothing allocated, when
    it is called.
    """
    import woof.offline_child_run as child_run

    _stub_cupy(monkeypatch)
    start = datetime(1974, 4, 3, 12)
    for index in range(3):
        _history(tmp_path / f"wrfout_d03_1974-04-03_{12 + index:02d}_00_00",
                 start + timedelta(hours=index), ny=18, nx=20)
    namelist = tmp_path / "namelist.input"
    namelist.write_text("&physics\n mp_physics = 8,\n/\n", encoding="utf-8")
    child = _runnable_child(tmp_path)
    from tilestream import autoplan

    order = []
    measured = []
    touched = []
    # The device runtime, WATCHED.  Where CuPy is the stub (none is
    # installed) every ``cupy.cuda.*`` access before the decision is
    # recorded, so the pin is not only "price_child ran first" but
    # "nothing consulted the card before Machine.detect did".  Where a
    # real CuPy is installed the stub is not in place and that half is
    # not asserted; the ordering half always is.
    fake = sys.modules["cupy"]
    watched = getattr(fake, "__file__", None) is None
    if watched:
        runtime = fake.cuda

        class _WatchedCuda:
            def __getattr__(self, name):
                touched.append(name)
                return getattr(runtime, name)

        fake.cuda = _WatchedCuda()

    def detect(cls, **kwargs):
        # The cold measurement itself, standing in for cudaMemGetInfo:
        # the first and only device touch before the decision.  The
        # real ``cold_card`` runs and reaches this.
        order.append("detect")
        measured.append(kwargs)
        return _fake_machine(6.0)

    real_price = downscale_pricing.price_child

    def price(cfg, options, **kwargs):
        order.append("decide")
        assert kwargs["machine"] is not None, "decided with no machine"
        assert kwargs["machine"].name == "fixture card", (
            "decided on a machine Machine.detect did not measure")
        assert kwargs["basis"] == downscale_pricing.MEASURED_BASIS
        # The default forcing model, the one the review and the fit price
        # with: the child streams its boundary intervals from the host.
        assert kwargs.get("forcing_intervals") is None
        # The machine was asked for the child's own [tiles] options.
        assert options.mode == "auto"
        return real_price(cfg, options, **kwargs)

    def interpolate(*args, **kwargs):
        order.append("interpolate")
        raise _Sentinel()

    monkeypatch.setattr(autoplan.Machine, "detect", classmethod(detect))
    monkeypatch.setattr(downscale_pricing, "price_child", price)
    monkeypatch.setattr(child_run, "interpolate_parent_initial_state",
                        interpolate)
    args = Namespace(
        parent_history=sorted(tmp_path.glob("wrfout_d03_*")),
        parent_restart=None, parent_namelist=namelist, parent_domain_id=3,
        child_config=child, parent_grid_ratio=1, i_parent_start=4,
        j_parent_start=4, max_boundary_interval_seconds=3600.0,
        accepted_parent_cadence=True, child_surface_from=None,
        # no WPS_GEOG tree is staged for the fixture parent
        child_terrain="parent",
        preprocess_backend="cpu", health_interval_seconds=60.0,
        outdir=tmp_path / "child-run")
    with pytest.raises(_Sentinel):
        child_run._run(args, child_run._ChildProgress())
    assert order == ["detect", "decide", "interpolate"]
    # ``cold_card`` handed the child's own host budget to the reader.
    assert measured and "host_bytes" in measured[0]
    if watched:
        assert touched == [], (
            f"the device runtime was consulted before the decision: {touched}")


# ---------------------------------------------------------------------------
# one price on every [tiles] setting
# ---------------------------------------------------------------------------

#: The 250 m child the 2026-09-26 user sweep ran on a 15.47 GiB RTX 5070 Ti:
#: 552x552x49 at ratio 12 under a 3 km parent, Thompson with RTE+RRTMGP, Noah,
#: MYNN surface layer and YSU.  Only the keys that differ from the RunConfig
#: defaults are spelled out; they are the derived child config's own.
_TWO_FIFTY_METRE_CHILD = dict(
    nx=552, ny=552, nz=49, dx=250.0, dy=250.0, ztop=20000.0, dt=1.25,
    run_seconds=39600.0, output_interval_s=3600.0,
    restart_interval_s=3600.0, grid_id=2, specified=True, map_proj=1,
    mp_physics=8, ra_lw_physics=4, ra_sw_physics=4, radt=12.0,
    wrf_rrtmg_compatibility="wrf-rrtmg-4-4-to-rte-rrtmgp-v2",
    bl_pbl_physics=1, sf_sfclay_physics=91, sf_surface_physics=2,
    cudt_minutes=0.0, hybrid_opt=2, hypsometric_opt=2, terrain_opt=1,
    moist=True, moist_cq=True, top_lid=False, epssm=0.5, emdiv=0.01,
    damp_opt=3, w_damping=1, diff_6th_opt=2, diff_6th_slopeopt=1, km_opt=4,
    h_sca_adv_order=5, nwp_diagnostics=1,
)

#: The card's workspace profile, with an explicit fixture memory budget.
#: The budget lies between the profile's price and the reference-card price.
_CARD_PROFILE_FIELDS = ("NVIDIA GeForce RTX 5070 Ti", 70, 1536, 1024)
_CARD_BUDGET_GIB = 15.0
_CARD_FREE_BYTES = int(_CARD_BUDGET_GIB * GIB)


def _drive_the_runner_to_its_decision(run_dir: Path, *, tiles_mode) -> None:
    """Run the child runner on the fixture parent until it has decided.

    Stops at ``interpolate_parent_initial_state``, which the caller has
    replaced with a sentinel raise.
    """
    import woof.offline_child_run as child_run

    run_dir.mkdir()
    start = datetime(1974, 4, 3, 12)
    for index in range(3):
        _history(run_dir / f"wrfout_d03_1974-04-03_{12 + index:02d}_00_00",
                 start + timedelta(hours=index), ny=18, nx=20)
    namelist = run_dir / "namelist.input"
    namelist.write_text("&physics\n mp_physics = 8,\n/\n", encoding="utf-8")
    args = Namespace(
        parent_history=sorted(run_dir.glob("wrfout_d03_*")),
        parent_restart=None, parent_namelist=namelist, parent_domain_id=3,
        child_config=_runnable_child(run_dir, tiles_mode=tiles_mode),
        parent_grid_ratio=1, i_parent_start=4, j_parent_start=4,
        max_boundary_interval_seconds=3600.0, accepted_parent_cadence=True,
        child_surface_from=None, preprocess_backend="cpu",
        # no WPS_GEOG tree is staged for the fixture parent
        child_terrain="parent",
        health_interval_seconds=60.0, outdir=run_dir / "child-run")
    with pytest.raises(_Sentinel):
        child_run._run(args, child_run._ChildProgress())


def test_the_runner_prices_the_child_on_the_card_whatever_the_tiles_setting(
        tmp_path, monkeypatch):
    """One child, one card, one price, with ``[tiles]`` off or auto.

    Without a measured card, the estimator falls back to the 170-SM
    reference profile.  The fixture's explicit 15 GiB budget admits the
    measured profile and refuses the reference profile, so this checks
    admission as well as the price without depending on a host card's
    current capacity.

    Both settings are driven through the runner itself; the price the
    runner took is then applied to the 250 m child, and it must be the
    review's price on both.
    """
    import woof.offline_child_run as child_run
    from woof.core.preflight import DeviceLocalMemoryProfile
    from tilestream import autoplan

    _stub_cupy(monkeypatch)
    profile = DeviceLocalMemoryProfile(*_CARD_PROFILE_FIELDS)
    card = autoplan.Machine(
        vram_bytes=_CARD_FREE_BYTES, host_bytes=64 * GIB, name=profile.name,
        host_source="explicit", device_profile=profile)
    detected = []

    def detect(cls, **kwargs):
        detected.append(kwargs)
        return card

    real_price = downscale_pricing.price_child
    decisions = []

    def price(cfg, options, **kwargs):
        pricing = real_price(cfg, options, **kwargs)
        decisions.append((options, kwargs, pricing))
        return pricing

    def interpolate(*args, **kwargs):
        raise _Sentinel()

    monkeypatch.setattr(autoplan.Machine, "detect", classmethod(detect))
    monkeypatch.setattr(downscale_pricing, "price_child", price)
    monkeypatch.setattr(child_run, "interpolate_parent_initial_state",
                        interpolate)
    for tiles_mode in (None, "auto"):
        _drive_the_runner_to_its_decision(
            tmp_path / str(tiles_mode), tiles_mode=tiles_mode)
    assert [options.mode for options, _, _ in decisions] == ["off", "auto"]

    # The price each setting's runner took, applied to the 250 m child.
    # Before the fix this read 17,033,346,128 B off and 14,922,267,728 B
    # auto.
    child = RunConfig(**_TWO_FIFTY_METRE_CHILD)
    off, auto = (real_price(child, options, **kwargs).peak_envelope_bytes
                 for options, kwargs, _ in decisions)
    assert off == auto
    # The review's call for the same child on the same fixture budget and
    # profile (the probe also reads the compile
    # platform, which a non-Noah-MP child is not priced on).
    review = real_price(
        child, streaming.OFF, machine=None,
        basis=downscale_pricing.MEASURED_BASIS, vram_gib=_CARD_BUDGET_GIB,
        profile=replace(profile, compile_platform=("120", "13.4.92")))
    assert off == review.peak_envelope_bytes
    assert review.peak_envelope_bytes < _CARD_BUDGET_GIB * GIB
    # Full Thompson tables are now counted. The former 14 GiB fixture
    # must refuse this child; 15 GiB still separates the two card prices.
    assert review.peak_envelope_bytes > 14 * GIB

    # The card was read on both settings, before the decision, and the
    # runner's record names it, off included: this block is what
    # report.json carries under "streaming".
    assert len(detected) == 2
    for _, kwargs, pricing in decisions:
        assert kwargs["machine"] is card
        entry = pricing.plan_entry()
        assert entry["machine"] == profile.name
        assert entry["machine_free_bytes"] == _CARD_FREE_BYTES
    # And the old runner's price, on no card, is the one that disagreed.
    unpriced = real_price(child, streaming.OFF, machine=None,
                          basis=downscale_pricing.MEASURED_BASIS)
    # Dropping the measured card must change its non-pool price and cross
    # the fixture's budget, preserving both sides of the admission check.
    assert unpriced.peak_envelope_bytes > review.peak_envelope_bytes
    assert unpriced.peak_envelope_bytes > _CARD_BUDGET_GIB * GIB


def test_a_host_that_cannot_be_read_refuses_only_the_settings_that_decide_on_it(
        monkeypatch):
    """Reading the card on every setting must not add a refusal.

    ``Machine.detect`` raises when no host RAM figure can be read (a
    container with no cgroup limit).  Off and a pinned tiling never consult
    the host, so they still run, priced on the card's own profile; auto and
    an unpinned ``on`` plan a host store and keep refusing.
    """
    from woof.core import preflight
    from woof.core.preflight import DeviceLocalMemoryProfile
    from tilestream import autoplan

    profile = DeviceLocalMemoryProfile(*_CARD_PROFILE_FIELDS)

    def no_host(cls, **kwargs):
        raise autoplan.CannotPlan("no host-memory source", "host")

    monkeypatch.setattr(autoplan.Machine, "detect", classmethod(no_host))
    monkeypatch.setattr(preflight, "live_device_local_memory_profile",
                        lambda: profile)
    for options in (None, streaming.OFF,
                    streaming.StreamingOptions(mode="on", tile_nx=64,
                                               tile_ny=64)):
        card = downscale_pricing.cold_card(options)
        assert card.machine is None and card.profile is profile
    for options in (streaming.StreamingOptions(mode="auto"),
                    streaming.StreamingOptions(mode="on")):
        with pytest.raises(autoplan.CannotPlan) as refused:
            downscale_pricing.cold_card(options)
        assert refused.value.resource == "host"

    # A card that cannot be read is not swallowed on any setting.
    def no_card(cls, **kwargs):
        raise autoplan.CannotPlan("no card", "vram")

    monkeypatch.setattr(autoplan.Machine, "detect", classmethod(no_card))
    with pytest.raises(autoplan.CannotPlan):
        downscale_pricing.cold_card(streaming.OFF)


# ---------------------------------------------------------------------------
# checkpoints on the child's own cadence
# ---------------------------------------------------------------------------


def test_the_checkpoint_schedule_is_every_restart_interval_and_the_end():
    # 1080 steps, a checkpoint every 180: six inside the window, the last
    # of them the end.
    assert sorted(checkpoint_schedule(1080, 180)) == [
        180, 360, 540, 720, 900, 1080]
    # A cadence that does not divide the run still ends on the last step.
    assert sorted(checkpoint_schedule(100, 30)) == [30, 60, 90, 100]
    # No cadence: the final checkpoint only, which is what the child
    # always wrote, now under a discoverable name.
    assert checkpoint_schedule(1080, None) == frozenset({1080})
    assert checkpoint_schedule(1080, 0) == frozenset({1080})


def test_the_checkpoint_names_are_the_ones_discovery_recognises(tmp_path):
    """The name the child writes is the name ``--parent-restart latest`` finds."""
    from woof.io.restart import restart_filename
    from woof.resume import discover_checkpoint_sets

    start = datetime(2026, 9, 11, 12)
    for hour in (1, 2):
        name = restart_filename(start + timedelta(hours=hour), domain="d02")
        (tmp_path / name).write_bytes(b"set")
    sets = discover_checkpoint_sets(tmp_path)
    assert [s.valid_time for s in sets] == [
        datetime(2026, 9, 11, 14), datetime(2026, 9, 11, 13)]
    assert all(list(s.members) == [2] for s in sets)
    # The old name was invisible to the same discovery.
    (tmp_path / "gpuwmrst_d02_final.npz").write_bytes(b"set")
    assert len(discover_checkpoint_sets(tmp_path)) == 2


def test_a_child_config_names_its_restart_cadence(tmp_path):
    cfg = load_config(_runnable_child(tmp_path))
    assert cfg.restart_interval_s == 300.0
    assert load_streaming_options(tmp_path / "child.toml").mode == "auto"


def test_the_run_name_reads_the_spacing_to_three_figures():
    frame = Path("/runs/parent-run/wrfout_d02_2026-09-11_12_00_00")
    assert child_run_name(frame, grid_id=3, ratio=3, dx=4000.0 / 3) == (
        "Downscale of parent-run · d03 ×3 · 1.33 km")
    assert child_run_name(frame, grid_id=2, ratio=3, dx=4000.0) == (
        "Downscale of parent-run · d02 ×3 · 4 km")


def test_the_child_clock_is_checked_once_for_both_doors():
    """``child_cadence`` is the step arithmetic the runner integrates on
    and the plan review refuses with; a clock that is not a whole number
    of steps is a review-time refusal in the runner's own words."""
    cfg = _refused_child_cfg()
    cadence = child_cadence(cfg, health_interval_seconds=60.0)
    assert (cadence.steps, cadence.output_steps, cadence.restart_steps,
            cadence.health_steps) == (1080, 180, 180, 3)
    assert sorted(cadence.checkpoint_due) == [180, 360, 540, 720, 900, 1080]

    # No restart cadence: one checkpoint, at the end.
    quiet = child_cadence(replace(cfg, restart_interval_s=0.0))
    assert quiet.restart_steps is None and quiet.health_steps is None
    assert quiet.checkpoint_due == frozenset({1080})

    for field, value in (("output_interval_s", 3610.0),
                         ("restart_interval_s", 3610.0),
                         ("run_seconds", 21610.0)):
        with pytest.raises(OfflineChildContractError,
                           match=f"{field}/dt must be a positive integer"
                           ) as refused:
            child_cadence(replace(cfg, **{field: value}))
        # The breakage and the way out in one sentence: the child
        # integrates in whole steps of dt, so the multiple to choose is
        # named with dt's own value.
        assert "whole steps of dt = 20 s" in str(refused.value)
        assert f"set {field} to a value that is a whole multiple of dt" in str(
            refused.value)
    with pytest.raises(OfflineChildContractError,
                       match="health_interval_seconds/dt") as refused:
        child_cadence(cfg, health_interval_seconds=50.0)
    assert "whole multiple of dt" in str(refused.value)
