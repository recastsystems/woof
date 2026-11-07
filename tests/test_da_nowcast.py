"""The nowcast front door's argument surface and planning math.

Pure tests: no network, no GPU, no subprocess execution.  Sites in
these tests are synthetic ids -- real station names never enter the
tree's generic code or its fixtures (standing owner rule).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.da_nowcast import (
    DEFAULT_DEALIAS_ENGINE,
    STAGES, VERIFY_SCHEMA, DealiasChoice, FrontDoorError, RadarSelection,
    WindowPlan,
    _stop, _stopped_early, advance_state,
    build_parser, cycle_cmd, fetch_cmd, gallery_rows, geojson_box,
    handoff_state, initial_frames, latest_gfs_cycle,
    merge_gallery_rows, motion_from_centroids, obs_cmd, plan_window,
    prepare_cmd, radar_selection, resolve_latest_window_end,
    site_domain_center, start_verification,
    validate_site, verdict_line, verification_block, verify_obs_name,
    verify_pass, watch_cmd, wizard_cmd, wrfout_name,
    write_partial_receipt, write_verification)
from tools.da_nowcast_render import footer_band_fraction


def utc(*parts) -> datetime:
    return datetime(*parts, tzinfo=timezone.utc)


NOW = utc(2026, 8, 5, 5, 45)


# ---------------------------------------------------------------------------
# site ids are arguments, validated, never defaulted
# ---------------------------------------------------------------------------
class TestSiteValidation:
    def test_uppercases(self):
        assert validate_site("qqqq") == "QQQQ"

    def test_accepts_digit_tail(self) -> None:
        assert validate_site("Q9Z1") == "Q9Z1"

    @pytest.mark.parametrize("bad", ["", "QQ", "QQQQQ", "1QQQ",
                                     "QQ-Q", "QQQ "])
    def test_refuses_non_station_shapes(self, bad):
        with pytest.raises(SystemExit):
            validate_site(bad)

    def test_site_has_no_default(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--window-end", "latest",
                               "--out", "x"])

    def test_no_real_site_tokens_in_the_tools(self):
        """No radar-site names in generic code, defaults, or
        identifiers -- the standing owner rule, checked mechanically
        for the two names the orders used."""

        tools = Path(__file__).resolve().parent.parent / "tools"
        for name in ("da_nowcast.py", "da_nowcast_render.py"):
            source = (tools / name).read_text(encoding="utf-8")
            for token in ("KDMX", "KBMX"):
                assert token not in source, (name, token)
            assert not re.search(r'default\s*=\s*"[A-Z][A-Z0-9]{3}"',
                                 source), name


# ---------------------------------------------------------------------------
# window planning
# ---------------------------------------------------------------------------
class TestWindowPlan:
    def plan(self, **overrides) -> WindowPlan:
        kwargs = dict(cycles=6, cycle_seconds=900, free_legs=6,
                      now=NOW)
        kwargs.update(overrides)
        return plan_window(utc(2026, 8, 5, 5, 30), **kwargs)

    def test_init_lands_on_the_hour(self):
        plan = self.plan()
        assert plan.init == utc(2026, 8, 5, 4, 0)

    def test_cycle_times_span_the_window(self):
        plan = self.plan()
        assert plan.cycle_times[0] == utc(2026, 8, 5, 4, 15)
        assert plan.cycle_times[-1] == plan.window_end

    def test_free_leg_times_continue_the_cadence(self):
        plan = self.plan()
        assert plan.free_leg_times()[0] == utc(2026, 8, 5, 5, 45)
        assert plan.free_leg_times()[-1] == utc(2026, 8, 5, 7, 0)

    def test_run_covers_horizon_with_margin(self):
        plan = self.plan()
        assert plan.horizon_seconds == 12 * 900
        assert plan.run_seconds >= plan.horizon_seconds + 3600

    def test_gfs_cycle_and_start_hour(self):
        plan = self.plan()
        assert plan.gfs_cycle == utc(2026, 8, 5, 0, 0)
        assert plan.forecast_start_hour == 4

    def test_misaligned_init_is_refused(self):
        with pytest.raises(SystemExit):
            plan_window(utc(2026, 8, 5, 5, 40), cycles=6,
                        cycle_seconds=900, free_legs=6, now=NOW)

    def test_zero_cycles_refused(self):
        with pytest.raises(SystemExit):
            self.plan(cycles=0)

    def test_payload_is_versionable(self):
        payload = self.plan().to_payload()
        assert payload["init"] == "2026-08-05T04:00:00Z"
        assert len(payload["cycle_times"]) == 6
        assert len(payload["free_leg_times"]) == 6


class TestGfsCycleSelection:
    def test_prefers_newest_available(self):
        assert latest_gfs_cycle(utc(2026, 8, 5, 4), NOW) \
            == utc(2026, 8, 5, 0)

    def test_respects_availability_lag(self):
        # at 03:30Z the 00Z cycle is 3.5 h old: not trusted yet
        assert latest_gfs_cycle(utc(2026, 8, 5, 3),
                                utc(2026, 8, 5, 3, 30)) \
            == utc(2026, 8, 4, 18)

    def test_never_after_init(self):
        # replaying an old window: newest cycle at/before init wins
        assert latest_gfs_cycle(utc(2026, 8, 1, 7), NOW) \
            == utc(2026, 8, 1, 6)


class TestBackgroundSelection:
    """The window plan rides the woof.da.background registry.

    HRRR is the front door's default background (project ruling,
    2026-08-06); GFS remains for archival reproduction, and its answer
    through the registry is proved identical to ``latest_gfs_cycle`` by
    tests/test_da_background.py.
    """

    def test_hrrr_rides_the_newest_published_hourly_cycle(self):
        plan = plan_window(utc(2026, 8, 5, 5, 30), cycles=2,
                           cycle_seconds=900, free_legs=6,
                           now=utc(2026, 8, 5, 5, 45), source="hrrr")
        # 05Z is only 45 min old -- its window's last lead (f003) is not
        # due yet, so the fail-closed walk lands on 04Z at f001.  HRRR's
        # wait is its route-table row's EXPECTED line (A136 L1
        # follow-ups): f003 is due 53.8 min after the cycle, the slowest
        # of the cycle hours (off-synoptic, 0.8 h + 0.0324 h per lead).
        # L1 had moved these clocks 15 to 30 min later when the wait read
        # the latest posting seen; a cycle later than its expected line
        # is waited for by the fetch instead (fetch_cmd's --wait-for).
        assert plan.background_source == "hrrr"
        assert plan.background_cycle == utc(2026, 8, 5, 4)
        assert plan.forecast_start_hour == 1
        # the compatibility shim answers with the same cycle
        assert plan.gfs_cycle == plan.background_cycle
        # The line itself: 05Z is taken the minute its f003 is due.
        for minute, cycle in ((53, utc(2026, 8, 5, 4)),
                              (54, utc(2026, 8, 5, 5))):
            assert plan_window(utc(2026, 8, 5, 5, 30), cycles=2,
                               cycle_seconds=900, free_legs=6,
                               now=utc(2026, 8, 5, 5, minute),
                               source="hrrr").background_cycle == cycle

    def test_a_shortened_window_lands_init_on_the_cycle_itself(self):
        # window-end HH:30 with 2 cycles of 15 min: init HH:00, and on
        # hourly HRRR cycles that init IS a cycle -- forecast lead 0,
        # the case the GFS-shaped arithmetic never produced.
        plan = plan_window(utc(2026, 8, 5, 5, 30), cycles=2,
                           cycle_seconds=900, free_legs=6,
                           now=utc(2026, 8, 5, 6, 30), source="hrrr")
        assert plan.background_cycle == plan.init
        assert plan.forecast_start_hour == 0

    def test_a_window_the_extended_cycles_still_reach_is_not_refused(self):
        """The boundary, read rather than assumed.

        This case was pinned at 30 hours as "past the horizon" when the
        planner only walked the hourly cycles, whose horizon is f18.  It
        walks the extended 00/06/12/18Z cycles too, and a 30 hour run
        from a 05Z init is leads 5..35 of the 00Z cycle, inside that
        cycle's 48 hour horizon -- so the old literal stopped denoting a
        refusable window and the case passed on nothing.  Measured at
        this commit: 30 and 40 hours are served by the 00Z cycle at lead
        5, and 50 hours is the first of the four whole-hour lengths read
        that is not.
        """

        for run_hours in (30, 40):
            plan = plan_window(utc(2026, 8, 5, 5, 30), cycles=2,
                               cycle_seconds=900, free_legs=6,
                               now=utc(2026, 8, 5, 5, 45), source="hrrr",
                               run_hours=run_hours)
            assert plan.background_source == "hrrr"
            assert plan.background_cycle == utc(2026, 8, 5, 0)
            assert plan.forecast_start_hour == 5

    def test_a_window_past_every_reachable_horizon_is_refused(self):
        """And the refusal names the frame and the horizon it exceeds."""

        with pytest.raises(SystemExit) as refusal:
            plan_window(utc(2026, 8, 5, 5, 30), cycles=2,
                        cycle_seconds=900, free_legs=6,
                        now=utc(2026, 8, 5, 5, 45), source="hrrr",
                        run_hours=50)
        sentence = str(refusal.value)
        assert "2026-08-05T05:00:00" in sentence
        assert "f62" in sentence and "f18" in sentence

    def test_a_source_the_roster_has_gained_is_selected(self):
        """Re-pinned, with the reading.

        ``rrfs`` was an unregistered source when this pinned its
        refusal; it is in the registry the door projects its roster
        from now, so the door plans on it like any other member.  The
        refusal below is the one that was actually being tested: a
        source the roster does not carry.

        Re-pinned for 3.0.0: the rrfs posting row is hourly with a
        declared publication lag of 4676.4 s, so at 05:45 the newest
        posted cycle covering the 05:00 init is 03Z (leads 2..5), not 00Z.
        """
        plan = plan_window(utc(2026, 8, 5, 5, 30), cycles=2,
                           cycle_seconds=900, free_legs=6,
                           now=utc(2026, 8, 5, 5, 45), source="rrfs")
        assert plan.background_source == "rrfs"
        assert plan.background_cycle == utc(2026, 8, 5, 3)
        assert plan.forecast_start_hour == 2

    def test_an_unregistered_source_is_refused_with_the_roster(self):
        """The name has to be one the roster does not carry.

        This case named a source that was unregistered when it was
        written and has since joined the roster, at which point it
        measured a registered source planning normally.  The refusal
        itself never moved: an unknown name is refused and the sentence
        lists what is available, which is what is asserted here.
        """

        with pytest.raises(SystemExit) as refusal:
            plan_window(utc(2026, 8, 5, 5, 30), cycles=2,
                        cycle_seconds=900, free_legs=6,
                        now=NOW, source="no-such-source")
        sentence = str(refusal.value)
        assert "no-such-source" in sentence
        assert "available sources:" in sentence
        assert "hrrr" in sentence and "gfs" in sentence

    def test_payload_carries_both_spellings_of_the_cycle(self):
        payload = plan_window(
            utc(2026, 8, 5, 5, 30), cycles=2, cycle_seconds=900,
            free_legs=6, now=utc(2026, 8, 5, 5, 45),
            source="hrrr").to_payload()
        assert payload["background_source"] == "hrrr"
        assert payload["background_cycle"] == "2026-08-05T04"
        # compatibility duplicate under the pre-HRRR key
        assert payload["gfs_cycle"] == payload["background_cycle"]


class TestLatestWindowEnd:
    def test_round_down_to_aligned_cycle(self):
        stamp = resolve_latest_window_end(
            utc(2026, 8, 5, 5, 36, 30), cycles=6, cycle_seconds=900)
        assert stamp == utc(2026, 8, 5, 5, 30)

    def test_steps_back_until_init_is_whole_hour(self):
        stamp = resolve_latest_window_end(
            utc(2026, 8, 5, 5, 36), cycles=6, cycle_seconds=1200)
        assert stamp == utc(2026, 8, 5, 5, 0)
        assert (stamp.hour * 60 + stamp.minute
                - 6 * 20) % 60 == 0


# ---------------------------------------------------------------------------
# siting
# ---------------------------------------------------------------------------
def stats(gates, east_km, north_km, valid):
    return {"gates": gates, "centroid_east_km": east_km,
            "centroid_north_km": north_km, "valid_time": valid}


class TestMotion:
    def test_pure_eastward(self):
        older = stats(5000, 0.0, 0.0, "2026-08-05T04:45:00Z")
        newer = stats(6000, 27.0, 0.0, "2026-08-05T05:30:00Z")
        motion = motion_from_centroids(older, newer, min_gates=500)
        assert motion is not None
        assert motion["u_ms"] == 10.0
        assert motion["v_ms"] == 0.0
        assert motion["toward_deg"] == 90.0

    def test_weak_echo_refuses_a_vector(self):
        older = stats(10, 0.0, 0.0, "2026-08-05T04:45:00Z")
        newer = stats(6000, 27.0, 0.0, "2026-08-05T05:30:00Z")
        assert motion_from_centroids(older, newer,
                                     min_gates=500) is None

    def test_short_baseline_refuses_a_vector(self):
        older = stats(5000, 0.0, 0.0, "2026-08-05T05:29:00Z")
        newer = stats(6000, 5.0, 0.0, "2026-08-05T05:30:00Z")
        assert motion_from_centroids(older, newer,
                                     min_gates=500) is None


class TestDomainCenter:
    def test_no_motion_centers_on_the_echo(self):
        center = site_domain_center(
            40.0, -95.0, stats(5000, -30.0, 0.0, "x"), None,
            horizon_seconds=10800, downstream_fraction=0.35,
            max_offset_km=60.0)
        assert center["basis"] == "echo centroid"
        assert center["offset_east_km"] == -30.0
        assert center["lon"] < -95.0

    def test_motion_biases_downstream(self):
        motion = {"u_ms": 10.0, "v_ms": 0.0, "speed_ms": 10.0}
        center = site_domain_center(
            40.0, -95.0, stats(5000, 0.0, 0.0, "x"), motion,
            horizon_seconds=10000, downstream_fraction=0.35,
            max_offset_km=60.0)
        assert center["offset_east_km"] == 35.0
        assert "downstream lead" in center["basis"]

    def test_offset_is_clamped(self):
        motion = {"u_ms": 30.0, "v_ms": 0.0, "speed_ms": 30.0}
        center = site_domain_center(
            40.0, -95.0, stats(5000, 100.0, 0.0, "x"), motion,
            horizon_seconds=20000, downstream_fraction=0.5,
            max_offset_km=60.0)
        assert center["offset_east_km"] == 60.0
        assert center["clamped_to_km"] == 60.0

    def test_no_echo_falls_back_to_antenna(self):
        center = site_domain_center(
            40.0, -95.0, stats(0, 0.0, 0.0, "x"), None,
            horizon_seconds=10800, downstream_fraction=0.35,
            max_offset_km=60.0)
        assert center["basis"].startswith("antenna")
        assert center["lat"] == 40.0


class TestGeojsonBox:
    def test_closed_ring_around_the_center(self):
        box = geojson_box(41.73, -93.96, 198.0)
        ring = box["coordinates"][0]
        assert len(ring) == 5
        assert ring[0] == ring[-1]
        lats = [p[1] for p in ring]
        lons = [p[0] for p in ring]
        assert min(lats) < 41.73 < max(lats)
        assert min(lons) < -93.96 < max(lons)


# ---------------------------------------------------------------------------
# stage command composition (never executed here)
# ---------------------------------------------------------------------------
class TestCommands:
    def plan(self) -> WindowPlan:
        return plan_window(utc(2026, 8, 5, 5, 30), cycles=6,
                           cycle_seconds=900, free_legs=6, now=NOW)

    def test_wrfout_name(self):
        assert wrfout_name(utc(2026, 8, 5, 4), 900) \
            == "wrfout_d01_2026-08-05_04_15_00"

    def test_wizard_cmd_carries_the_plan(self):
        argv = wizard_cmd(
            polygon=Path("box.geojson"), out_toml=Path("case.toml"),
            plan=self.plan(), profile="p", name="n", dx_km=3.0,
            source="gfs")
        text = " ".join(argv)
        assert "--cycle 2026-08-05T00" in text
        assert "--forecast-start-hour 4" in text
        assert "--root-dx 3" in text
        assert "woof.cli" in text and "domain" in argv

    def test_fetch_cmd_from_wizard_hints(self):
        hints = {"cycle": "2026-08-05T00", "hours": 4,
                 "area": "1,2,3,4", "cadence": 1,
                 "forecast_start_hour": 4}
        argv = fetch_cmd(hints=hints, data_dir=Path("d"),
                         source="gfs")
        assert "--area=1,2,3,4" in argv
        assert "--cadence" in argv

    def test_fetch_cmd_survives_the_shortened_window_hints(self):
        """Issue #74: a lead-0 window's hints carry no start hour.

        The wizard writes ``forecast_start_hour`` into its [fetch]
        hints only when it is nonzero.  A window ending at HH:30 with
        2 cycles puts init at HH:00 -- on hourly HRRR cycles that is
        f000, so the key is absent, and indexing it was a KeyError for
        BOTH sources.  The absent key must mean lead 0, said out loud.
        """

        plan = plan_window(utc(2026, 8, 5, 5, 30), cycles=2,
                           cycle_seconds=900, free_legs=6,
                           now=utc(2026, 8, 5, 6, 30), source="hrrr")
        assert plan.forecast_start_hour == 0     # the hint the wizard omits
        hints = {"cycle": "2026-08-05T05", "hours": 4, "area": "1,2,3,4"}
        argv = fetch_cmd(hints=hints, data_dir=Path("d"), source="hrrr")
        assert argv[argv.index("--forecast-start-hour") + 1] == "0"
        # HRRR is hourly: `woof fetch --source hrrr` refuses --cadence,
        # and the wizard's hrrr hints never carry one.
        assert "--cadence" not in argv
        # same fix, same meaning, on the archival GFS route
        argv = fetch_cmd(hints={**hints, "cadence": 3}, data_dir=Path("d"),
                         source="gfs")
        assert argv[argv.index("--forecast-start-hour") + 1] == "0"

    def test_fetch_cmd_waits_for_a_background_still_posting(self):
        """A late background is waited for, never refused (A136 L1 follow-ups).

        The plan takes a cycle once its window's last lead is due on the
        row's EXPECTED line, so that cycle can still be posting when the
        fetch starts.  The fetch then waits for each lead, saying which,
        until the last lead is late on its row: 05Z f004 is expected
        55.8 min after the cycle (off-synoptic, 0.8 h + 0.0324 h per
        lead), late 90 min after that (the row's late_after_minutes, at
        least its 57 min measured late spread), so at 06:30 the fetch
        waits up to 56 min.  Past that late time nothing is waited for,
        and the fetch's own publication check refuses a missing lead at
        once.
        """

        hints = {"cycle": "2026-08-05T05", "hours": 4, "area": "1,2,3,4"}
        argv = fetch_cmd(hints=hints, data_dir=Path("d"), source="hrrr",
                         now=utc(2026, 8, 5, 6, 30))
        assert "--wait-for" in argv
        assert argv[argv.index("--wait-timeout-minutes") + 1] == "56"
        late = fetch_cmd(hints=hints, data_dir=Path("d"), source="hrrr",
                         now=utc(2026, 8, 5, 7, 26))
        assert "--wait-for" not in late
        assert "--wait-timeout-minutes" not in late
        # GFS's plan is its shipped fixed wait, not a posting row, and
        # `woof fetch --source gfs` takes no --wait-for.
        gfs = fetch_cmd(hints={**hints, "cycle": "2026-08-05T00",
                               "cadence": 1}, data_dir=Path("d"),
                        source="gfs", now=utc(2026, 8, 5, 4, 30))
        assert "--wait-for" not in gfs

    def test_cycle_cmd_forwards_the_reflectivity_analysis_trio(self):
        """The one-seam wire-through for the Z-DA arm (2026-08-06).

        The ktbw HRRR case showed every trajectory carrying the same
        misplaced complex: velocity-only DA cannot relocate echo.  The
        driver has carried --hydrometeors / --positivity-policy /
        --reflectivity-analysis all along; the front door now forwards
        them, and forwards NOTHING when they are not asked for, so
        every existing invocation builds its existing argv.
        """
        base = dict(
            prepared_root=Path("p"), authority_dir=Path("a"),
            profile="prof", plan=self.plan(), members=8,
            obs_files=[Path("o")], grid_wrfouts=[Path("g")],
            cycle_out=Path("c"), proof_sha="x", manifest_sha="y",
            content_sha="z", seed=1, solve_device="cuda",
            horizontal_loc_m=12000.0, vertical_loc_m=3000.0,
            length_scale_km=50.0, source="hrrr")
        plain = cycle_cmd(**base)
        assert "--hydrometeors" not in plain
        assert "--reflectivity-analysis" not in plain
        assert "--positivity-policy" not in plain
        armed = cycle_cmd(**base, hydrometeors=True,
                          positivity_policy="clip",
                          reflectivity_analysis=True)
        assert armed[armed.index("--positivity-policy") + 1] == "clip"
        assert "--hydrometeors" in armed
        assert "--reflectivity-analysis" in armed

    def test_analysis_flags_refuse_incoherent_combinations_early(self):
        """The driver's refusal chain, met before any stage is paid for."""
        from tools.da_nowcast import validate_analysis_flags

        class Args:
            hydrometeors = False
            positivity_policy = None
            reflectivity_analysis = False

        validate_analysis_flags(Args())     # the wind-only default is fine
        bad = Args()
        bad.reflectivity_analysis = True
        with pytest.raises(SystemExit):
            validate_analysis_flags(bad)
        bad = Args()
        bad.hydrometeors = True
        with pytest.raises(SystemExit):
            validate_analysis_flags(bad)

    def test_prepare_cmd_speaks_each_sources_own_grammar(self):
        gfs = prepare_cmd(
            data_dir=Path("d"), authority_dir=Path("a"),
            bridge=Path("br"), profile="p", geog_root=Path("g"),
            prepared_root=Path("pr"), plan=self.plan(),
            manifest_sha="m", source="gfs")
        text = " ".join(gfs)
        assert "--gfs-series" in text and "--experiment-config" in text
        assert "woof.source_cli" in text

        hrrr_plan = plan_window(utc(2026, 8, 5, 5, 30), cycles=2,
                                cycle_seconds=900, free_legs=6,
                                now=utc(2026, 8, 5, 5, 45),
                                source="hrrr")
        hrrr = prepare_cmd(
            data_dir=Path("d"), authority_dir=Path("a"),
            bridge=None, profile="p", geog_root=Path("g"),
            prepared_root=Path("pr"), plan=hrrr_plan,
            manifest_sha="m", source="hrrr",
            namelist_input=Path("case.namelist.input"),
            domain_spec=Path("case.d01-target.json"),
            history_interval_seconds=900.0)
        text = " ".join(hrrr)
        assert "--source-root" in text
        assert "SHA256SUMS" in text
        assert hrrr[hrrr.index("--valid-time") + 1] \
            == "2026-08-05_04:00:00"
        assert hrrr[hrrr.index("--forecast-start-hour") + 1] == "1"
        assert "--run-seconds" in text
        # the GFS grammar's flags are refusals on the HRRR door
        for flag in ("--gfs-series", "--bridge", "--experiment-config",
                     "--cycle"):
            assert flag not in hrrr

    def test_obs_cmd_takes_the_site_as_argument(self):
        argv = obs_cmd(selection=RadarSelection(anchor="QQQQ"),
                       valid=utc(2026, 8, 5, 4, 15),
                       grid_wrfout=Path("g"), out_nc=Path("o"),
                       work_dir=Path("w"), bucket=None,
                       dealias=DealiasChoice())
        assert "QQQQ" in argv
        assert "--valid-time" in argv
        assert argv[argv.index("--valid-time") + 1] \
            == "2026-08-05T04:15:00Z"

    def test_cycle_cmd_pairs_obs_with_grids(self):
        obs = [Path(f"o{k}") for k in range(6)]
        grids = [Path(f"g{k}") for k in range(6)]
        argv = cycle_cmd(
            prepared_root=Path("p"), authority_dir=Path("a"),
            profile="prof", plan=self.plan(), members=10,
            obs_files=obs, grid_wrfouts=grids, cycle_out=Path("c"),
            proof_sha="x", manifest_sha="y", content_sha="z",
            seed=1, solve_device="cuda", horizontal_loc_m=12000.0,
            vertical_loc_m=3000.0, length_scale_km=50.0,
            source="gfs")
        assert argv.count("--obs") == 6
        assert argv.count("--grid-wrfout") == 6
        assert "--save-composites" in argv
        assert "--free-legs" in argv

    def test_stop_after_orders_stages(self):
        assert STAGES.index("survey") < STAGES.index("obs") \
            < STAGES.index("cycle") < STAGES.index("render")
        assert _stop("survey", "survey")
        assert _stop("obs", "cycle")       # later stage stops too
        assert not _stop("cycle", "obs")   # earlier stage continues
        assert _stop(None, "render") is False


# ---------------------------------------------------------------------------
# the CLI surface itself
# ---------------------------------------------------------------------------
class TestParser:
    def test_run_defaults_are_mechanism_numbers(self):
        args = build_parser().parse_args(
            ["run", "--site", "qqqq", "--window-end", "latest",
             "--out", "case"])
        assert args.site == "QQQQ"
        assert args.cycles == 6
        assert args.cycle_seconds == 900
        assert args.free_legs == 6
        assert args.members == 10
        assert args.solve_device == "cuda"
        assert args.stop_after is None

    def test_hrrr_is_the_default_background(self):
        """Project ruling, 2026-08-06: HRRR default, permanent.

        GFS stays a choice for archival reproduction of pre-HRRR runs,
        and anything else must arrive as a woof.da.background registry
        entry before it can appear here.
        """

        args = build_parser().parse_args(
            ["run", "--site", "qqqq", "--window-end", "latest",
             "--out", "case"])
        assert args.source == "hrrr"
        args = build_parser().parse_args(
            ["run", "--site", "qqqq", "--window-end", "latest",
             "--out", "case", "--source", "gfs"])
        assert args.source == "gfs"
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["run", "--site", "qqqq", "--window-end", "latest",
                 "--out", "case", "--source", "rrfs"])

    def test_verify_requires_case_dir(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["verify"])

    def test_mode_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_verification_is_on_by_default(self):
        args = build_parser().parse_args(
            ["run", "--site", "qqqq", "--window-end", "latest",
             "--out", "case"])
        assert args.no_verify is False

    def test_no_verify_opts_out(self):
        args = build_parser().parse_args(
            ["run", "--site", "qqqq", "--window-end", "latest",
             "--out", "case", "--no-verify"])
        assert args.no_verify is True

    def test_watch_mode_exists_for_an_existing_case(self):
        args = build_parser().parse_args(
            ["watch", "--case-dir", "case", "--poll-seconds", "30"])
        assert args.mode == "watch"
        assert args.case_dir == Path("case")
        assert args.poll_seconds == 30
        assert args.max_minutes > 0


# ---------------------------------------------------------------------------
# the verification state machine: what a GUI polls
# ---------------------------------------------------------------------------
FREE_TIMES = ["2026-08-05T05:45:00Z", "2026-08-05T06:00:00Z",
              "2026-08-05T06:15:00Z"]


def row(valid: str, fss: float = 0.75) -> dict:
    """A per-frame row shaped like the renderer's published ones."""

    return {"leg": 6, "valid": valid,
            "obs_valid_time": valid.replace(":00Z", ":12Z"),
            "obs_cols_gt35": 2817, "fcst_cols_gt35_in_echo": 1899,
            "control_cols_gt35_in_echo": 519, "fss30_fcst": fss,
            "fss30_control": 0.24, "fss_half_width_cells": 4}


class TestFrames:
    def test_initial_frames_are_pending_in_order(self):
        frames = initial_frames(FREE_TIMES)
        assert [f["valid"] for f in frames] == FREE_TIMES
        assert {f["status"] for f in frames} == {"pending"}

    def test_obs_name_carries_the_site_argument(self):
        name = verify_obs_name("QQQQ", utc(2026, 8, 5, 6, 15))
        assert name == "verify-qqqq-202608050615.nc"

    def test_merge_grades_only_frames_with_numbers(self):
        frames = initial_frames(FREE_TIMES)
        merged = merge_gallery_rows(frames, [row(FREE_TIMES[0])])
        assert merged[0]["status"] == "verified"
        assert merged[0]["fss30_fcst"] == 0.75
        assert [f["status"] for f in merged[1:]] == ["pending"] * 2

    def test_merge_clears_a_stale_note_and_keeps_valid_time(self):
        frames = [{"valid": FREE_TIMES[0], "status": "pending",
                   "note": "not covered yet"}]
        merged = merge_gallery_rows(frames, [row(FREE_TIMES[0])])
        assert "note" not in merged[0]
        assert merged[0]["valid"] == FREE_TIMES[0]

    def test_merge_leaves_the_input_untouched(self):
        frames = initial_frames(FREE_TIMES)
        merge_gallery_rows(frames, [row(FREE_TIMES[0])])
        assert frames[0]["status"] == "pending"

    def test_a_built_observation_alone_is_not_a_grade(self):
        frames = [{"valid": FREE_TIMES[0], "status": "pending",
                   "obs_file": "verify-qqqq-202608050545.nc"}]
        assert merge_gallery_rows(frames, [])[0]["status"] == "pending"


class TestStateMachine:
    def test_rolling_while_frames_wait(self):
        frames = merge_gallery_rows(initial_frames(FREE_TIMES),
                                    [row(FREE_TIMES[0])])
        assert advance_state(frames, exhausted=False) == "rolling"

    def test_complete_when_every_frame_carries_numbers(self):
        frames = merge_gallery_rows(initial_frames(FREE_TIMES),
                                    [row(t) for t in FREE_TIMES])
        assert advance_state(frames, exhausted=False) == "complete"

    def test_incomplete_when_the_verifier_gives_up(self):
        frames = initial_frames(FREE_TIMES)
        assert advance_state(frames, exhausted=True) == "incomplete"

    def test_no_free_legs_is_vacuously_complete(self):
        assert advance_state([], exhausted=False) == "complete"

    def test_handoff_defaults_to_pending_not_off(self):
        frames = initial_frames(FREE_TIMES)
        assert handoff_state(no_verify=False, frames=frames) \
            == "pending"

    def test_handoff_respects_the_opt_out(self):
        assert handoff_state(no_verify=True,
                             frames=initial_frames(FREE_TIMES)) \
            == "disabled"

    def test_handoff_with_nothing_to_grade(self):
        assert handoff_state(no_verify=False, frames=[]) == "complete"

    def test_verdicts_count_what_is_graded(self):
        frames = merge_gallery_rows(initial_frames(FREE_TIMES),
                                    [row(FREE_TIMES[0])])
        assert "1/3" in verdict_line(frames, "rolling")
        assert "not started" in verdict_line(frames, "disabled")

    def test_block_is_versioned_and_counted(self):
        frames = merge_gallery_rows(initial_frames(FREE_TIMES),
                                    [row(FREE_TIMES[0])])
        block = verification_block(frames=frames, state="rolling")
        assert block["schema"] == VERIFY_SCHEMA
        assert (block["graded"], block["total"]) == (1, 3)
        assert block["frames"] is frames
        assert "unscored" in block["honesty"]

    def test_block_refuses_an_unknown_state(self):
        with pytest.raises(SystemExit):
            verification_block(frames=[], state="probably-fine")


# ---------------------------------------------------------------------------
# the auto-handoff: a finished run starts its own verifier
# ---------------------------------------------------------------------------
def seed_case(tmp_path: Path, free_times=FREE_TIMES) -> Path:
    """A case directory with just the receipt the verifier reads."""

    (tmp_path / "nowcast-receipt.json").write_text(json.dumps({
        "schema": "gpuwm-da.nowcast.v1", "site": "QQQQ",
        "plan": {"init": "2026-08-05T04:00:00Z",
                 "cycle_seconds": 900,
                 "free_leg_times": list(free_times)}}), encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# the partial receipt a run that stops early owes its verifier (#110)
# ---------------------------------------------------------------------------
class TestThePartialReceiptAStoppedRunLeaves:
    """A two-phase case must reach `verify` without hand-synthesis.

    Every --stop-after door returns before the stage that writes the
    full receipt, so a run whose later phase is driven directly (the
    supported route for tools/da_cycle_prepared) left a case directory
    the verify door refused outright.  What people did instead was write
    the file by hand.
    """

    PLAN = {
        "schema": "gpuwm-da.nowcast.v1", "partial": True, "site": "KICT",
        "radars": {"anchor": "KICT", "route": "discover", "sites": [],
                   "multi_radar": True, "min_coverage_fraction": 0.2,
                   "max_radars": None, "min_radars": 2,
                   "max_radar_time_spread_seconds": None},
        "plan": {"init": "2026-08-05T04:00:00Z", "cycle_seconds": 900,
                 "free_leg_times": list(FREE_TIMES)},
        "domain_center": {"lat": 37.65, "lon": -97.44},
        "seed": 20260807, "case_name": "nowcast_kict_202608050400",
    }

    def _case(self, tmp_path):
        receipts = tmp_path / "receipts"
        receipts.mkdir(parents=True)
        (receipts / "01-plan.json").write_text(
            json.dumps(self.PLAN), encoding="utf-8")
        return receipts

    def test_the_five_fields_the_verifier_reads_are_all_present(
            self, tmp_path):
        """The requirement, stated as the requirement.

        Read through `_case_context` rather than by poking keys, so the
        test tracks what the verifier actually does.
        """
        from tools.da_nowcast import _case_context

        receipts = self._case(tmp_path)
        written = write_partial_receipt(
            tmp_path, receipts=receipts, dealias=DealiasChoice(on=True),
            stopped_after="obs")
        assert written == tmp_path / "nowcast-receipt.json"

        selection, init, frames, dealias = _case_context(tmp_path)
        assert selection.anchor == "KICT"
        assert selection.min_radars == 2
        assert init == datetime(2026, 8, 5, 4, 0, tzinfo=timezone.utc)
        assert [frame["valid"] for frame in frames] == list(FREE_TIMES)
        assert dealias == DealiasChoice(on=True)

    def test_nothing_in_it_is_invented(self, tmp_path):
        """Verbatim from the plan receipt, or from the launch command."""
        receipts = self._case(tmp_path)
        write_partial_receipt(tmp_path, receipts=receipts,
                              dealias=DealiasChoice(), stopped_after="obs")
        receipt = json.loads(
            (tmp_path / "nowcast-receipt.json").read_text(encoding="utf-8"))
        for key in ("site", "radars", "plan", "domain_center", "seed",
                    "case_name"):
            assert receipt[key] == self.PLAN[key]
        assert receipt["obs"] == DealiasChoice().to_payload()
        provenance = receipt["receipt_provenance"]
        assert set(provenance["verbatim_from_receipts_01_plan_json"]) == {
            "site", "radars", "plan", "domain_center", "seed", "case_name"}
        # What a complete run would carry is ABSENT, not guessed.
        for key in ("sizing", "survey", "outputs", "length_scale_km",
                    "members"):
            assert key not in receipt
            assert key in provenance["not_reconstructed"]

    def test_it_says_it_is_partial_and_who_wrote_it(self, tmp_path):
        receipts = self._case(tmp_path)
        write_partial_receipt(tmp_path, receipts=receipts,
                              dealias=DealiasChoice(),
                              stopped_after="forecast")
        receipt = json.loads(
            (tmp_path / "nowcast-receipt.json").read_text(encoding="utf-8"))
        assert receipt["partial"] is True
        assert receipt["schema"] == "gpuwm-da.nowcast.v1"
        provenance = receipt["receipt_provenance"]
        assert provenance["written_by"] == (
            "da_nowcast run --stop-after forecast")
        install = provenance["install"]
        assert "identity_source" in install or "identity_error" in install

    def test_verify_can_write_its_verdict_onto_a_partial_receipt(
            self, tmp_path):
        """The other half of "reaches verify": the receipt is writable.

        `write_verification` re-reads and rewrites the whole file, so a
        partial receipt that could be read but not updated would fail at
        the very end of a verify pass.
        """
        receipts = self._case(tmp_path)
        write_partial_receipt(tmp_path, receipts=receipts,
                              dealias=DealiasChoice(on=True),
                              stopped_after="obs")
        write_verification(tmp_path, verification_block(
            frames=initial_frames(FREE_TIMES), state="pending"))
        receipt = json.loads(
            (tmp_path / "nowcast-receipt.json").read_text(encoding="utf-8"))
        assert receipt["verification"]["state"] == "pending"
        assert receipt["site"] == "KICT"      # nothing else disturbed

    def test_a_complete_receipt_is_never_downgraded(self, tmp_path):
        """A full run of this case outranks a later partial one."""
        receipts = self._case(tmp_path)
        (tmp_path / "nowcast-receipt.json").write_text(json.dumps({
            "schema": "gpuwm-da.nowcast.v1", "site": "KICT",
            "sizing": {"members": 6}}), encoding="utf-8")
        assert write_partial_receipt(
            tmp_path, receipts=receipts, dealias=DealiasChoice(on=True),
            stopped_after="obs") is None
        receipt = json.loads(
            (tmp_path / "nowcast-receipt.json").read_text(encoding="utf-8"))
        assert receipt["sizing"] == {"members": 6}

    def test_an_earlier_partial_receipt_is_refreshed(self, tmp_path):
        receipts = self._case(tmp_path)
        write_partial_receipt(tmp_path, receipts=receipts,
                              dealias=DealiasChoice(),
                              stopped_after="survey")
        write_partial_receipt(tmp_path, receipts=receipts,
                              dealias=DealiasChoice(on=True),
                              stopped_after="obs")
        receipt = json.loads(
            (tmp_path / "nowcast-receipt.json").read_text(encoding="utf-8"))
        assert receipt["obs"]["dealias"] is True
        assert receipt["receipt_provenance"]["written_by"].endswith("obs")

    def test_a_run_that_never_planned_writes_nothing(self, tmp_path):
        """No plan receipt means nothing accurate to copy."""
        (tmp_path / "receipts").mkdir()
        assert write_partial_receipt(
            tmp_path, receipts=tmp_path / "receipts", dealias=DealiasChoice(),
            stopped_after="survey") is None
        assert not (tmp_path / "nowcast-receipt.json").exists()

    def test_a_corrupt_plan_receipt_writes_nothing(self, tmp_path):
        receipts = tmp_path / "receipts"
        receipts.mkdir()
        (receipts / "01-plan.json").write_text("{not json", encoding="utf-8")
        assert write_partial_receipt(
            tmp_path, receipts=receipts, dealias=DealiasChoice(),
            stopped_after="obs") is None

    def test_every_stop_door_goes_through_the_emitter(self, tmp_path,
                                                      capsys):
        """One emitter, so a door added later cannot forget the receipt.

        The defect was seven independent `return 0`s; a test that only
        covered `obs` would let the next one regress.
        """
        receipts = self._case(tmp_path)
        for stage in STAGES:
            (tmp_path / "nowcast-receipt.json").unlink(missing_ok=True)
            args = SimpleNamespace(dealias=False, dealias_engine="vad-region",
                                   dealias_refinement=False)
            assert _stopped_early(tmp_path, receipts, args, stage) == 0
            out = capsys.readouterr().out
            assert f"stopped after {stage}" in out
            assert "partial receipt" in out
            assert (tmp_path / "nowcast-receipt.json").is_file()


class TestWatchCommand:
    def test_watch_cmd_runs_this_module(self):
        argv = watch_cmd(case_dir=Path("case"), poll_seconds=240,
                         max_minutes=180, max_offset_seconds=480.0,
                         bucket=None)
        assert argv[1:4] == ["-m", "tools.da_nowcast", "watch"]
        assert argv[argv.index("--case-dir") + 1] == "case"
        assert "--bucket" not in argv

    def test_watch_cmd_forwards_the_bucket_override(self):
        argv = watch_cmd(case_dir=Path("case"), poll_seconds=30,
                         max_minutes=5, max_offset_seconds=60.0,
                         bucket="some-bucket")
        assert argv[argv.index("--bucket") + 1] == "some-bucket"
        assert argv[argv.index("--poll-seconds") + 1] == "30"

    def test_handoff_spawns_detached_and_records_the_state(
            self, tmp_path, monkeypatch):
        case = seed_case(tmp_path)
        seen = {}

        def fake_spawn(argv, *, cwd, log_path):
            seen["argv"] = argv
            seen["log"] = log_path
            return 4242

        monkeypatch.setattr("tools.da_nowcast.spawn_detached",
                            fake_spawn)
        block = start_verification(
            case, frames=initial_frames(FREE_TIMES), poll_seconds=240,
            max_minutes=180, max_offset_seconds=480.0, bucket=None,
            repo_root=tmp_path)

        assert "watch" in seen["argv"]
        assert seen["log"] == case / "verify-watch.log"
        assert block["state"] == "rolling"
        assert block["watcher"]["pid"] == 4242
        assert block["watcher"]["argv"] == seen["argv"]

        receipt = json.loads(
            (case / "nowcast-receipt.json").read_text(encoding="utf-8"))
        assert receipt["verification"]["state"] == "rolling"
        assert receipt["verification"]["total"] == 3
        assert receipt["site"] == "QQQQ"   # the rest of it survives

    def test_receipt_rewrite_is_atomic_and_key_scoped(self, tmp_path):
        case = seed_case(tmp_path)
        write_verification(case, verification_block(
            frames=initial_frames(FREE_TIMES), state="pending"))
        write_verification(case, verification_block(
            frames=initial_frames(FREE_TIMES), state="rolling"))
        receipt = json.loads(
            (case / "nowcast-receipt.json").read_text(encoding="utf-8"))
        assert receipt["verification"]["state"] == "rolling"
        assert receipt["plan"]["cycle_seconds"] == 900
        assert not list(case.glob("*.tmp"))


class TestFooterBand:
    """The credit line gets reserved geometry, not luck."""

    def test_short_wide_figures_reserve_a_visible_band(self):
        # 02-cycle-numbers is 5.2in tall; its tick labels used to run
        # straight into the footer.
        assert footer_band_fraction(5.2) > 0.05

    def test_tall_figures_reserve_proportionally_less(self):
        assert footer_band_fraction(10.4) < footer_band_fraction(5.2)

    def test_the_band_never_eats_the_figure(self):
        assert footer_band_fraction(0.5) <= 0.12

    def test_a_zero_height_figure_is_refused(self):
        with pytest.raises(ValueError):
            footer_band_fraction(0.0)


class TestVerifyPass:
    def test_gallery_rows_absent_is_not_an_error(self, tmp_path):
        assert gallery_rows(tmp_path) == []

    def test_a_pass_grades_from_the_renderers_published_rows(
            self, tmp_path, monkeypatch):
        case = seed_case(tmp_path)
        (case / "obsverify").mkdir()
        (case / "obsverify" / verify_obs_name(
            "QQQQ", datetime(2026, 8, 5, 5, 45,
                             tzinfo=timezone.utc))).write_bytes(b"x")
        (case / "gallery").mkdir()
        (case / "gallery" / "_verification.json").write_text(
            json.dumps({"frames": [row(FREE_TIMES[0])]}),
            encoding="utf-8")

        asked = []

        def record(*, valid, **kwargs):
            asked.append(valid.strftime("%Y-%m-%dT%H:%M:00Z"))
            return False, "archive does not cover it yet"

        monkeypatch.setattr("tools.da_nowcast.build_verify_frame",
                            record)
        frames, built = verify_pass(
            case_dir=case, selection=RadarSelection(anchor="QQQQ"),
            init=utc(2026, 8, 5, 4), frames=initial_frames(FREE_TIMES),
            bucket=None, max_offset_seconds=480.0, repo_root=case,
            dealias=DealiasChoice())
        assert built == 0
        assert FREE_TIMES[0] not in asked   # already on disk, not refetched
        assert asked == FREE_TIMES[1:]
        assert frames[0]["status"] == "verified"
        assert frames[0]["obs_file"].endswith("202608050545.nc")
        assert frames[1]["note"] == "archive does not cover it yet"

    def test_a_pass_never_asks_for_a_future_valid_time(
            self, tmp_path, monkeypatch):
        soon = datetime.now(timezone.utc) + timedelta(hours=2)
        case = seed_case(tmp_path,
                         [soon.strftime("%Y-%m-%dT%H:%M:00Z")])

        def refuse(**kwargs):
            raise AssertionError("asked the archive for the future")

        monkeypatch.setattr("tools.da_nowcast.build_verify_frame",
                            refuse)
        frames, built = verify_pass(
            case_dir=case, selection=RadarSelection(anchor="QQQQ"),
            init=utc(2026, 8, 5, 4),
            frames=initial_frames([soon.strftime("%Y-%m-%dT%H:%M:00Z")]),
            bucket=None, max_offset_seconds=480.0, repo_root=case,
            dealias=DealiasChoice())
        assert built == 0
        assert frames[0]["status"] == "pending"
        assert "not reached" in frames[0]["note"]


# ---------------------------------------------------------------------------
# the gallery dates its frames from the report, not from a cadence
# ---------------------------------------------------------------------------
class TestGalleryLegTimes:
    """A nowcast that cycles on the radar's rhythm has legs of unequal
    length, so every caption has to come from the leg's own end."""

    def gallery(self, ends, *, cycles=2, notice=None):
        from tools.da_nowcast_render import Gallery

        g = Gallery.__new__(Gallery)
        g.init = utc(2026, 8, 5, 4, 0)
        g.leg_seconds = 900.0
        g.leg_end_s = list(ends)
        g.legs = [{"end_s": e} for e in ends]
        g.cycles = cycles
        g.notice = notice
        return g

    def test_valid_times_come_from_the_report(self):
        g = self.gallery([1260.0, 2505.0, 3405.0])
        assert g.leg_valid(0) == utc(2026, 8, 5, 4, 21)
        assert g.leg_valid(1) == utc(2026, 8, 5, 4, 41, 45)

    def test_a_report_without_leg_ends_falls_back_to_the_cadence(self):
        from tools.da_nowcast_render import Gallery

        g = Gallery.__new__(Gallery)
        g.init = utc(2026, 8, 5, 4, 0)
        g.leg_seconds = 900.0
        g.leg_end_s = []
        assert g.leg_elapsed(0) == 900.0
        assert g.leg_elapsed(3) == 3600.0

    def test_an_observed_frame_matches_the_leg_it_belongs_to(self):
        g = self.gallery([1260.0, 2505.0, 3405.0])
        assert g.leg_at(2505.0) == 1
        assert g.leg_at(2500.0) == 1

    def test_a_frame_between_legs_matches_nothing(self):
        g = self.gallery([600.0, 1200.0, 1800.0])
        assert g.leg_at(900.0) is None

    def test_equal_legs_report_one_cadence(self):
        assert self.gallery([900.0, 1800.0, 2700.0],
                            cycles=3).cadence_label() == "15-min"

    def test_unequal_legs_report_the_range_they_used(self):
        assert self.gallery([300.0, 660.0, 960.0],
                            cycles=3).cadence_label() == "5-6-min"

    def test_no_notice_means_no_reserved_band_and_no_forced_title(self):
        g = self.gallery([900.0])
        assert g.suptitle_y(_FakeFig(6.0)) is None

    def test_a_notice_pushes_the_title_below_its_band(self):
        g = self.gallery([900.0], notice={"level": "warn",
                                          "headline": "late"})
        assert 0.9 < g.suptitle_y(_FakeFig(6.0)) < 1.0


class _FakeFig:
    def __init__(self, height):
        self._h = height

    def get_figheight(self):
        return self._h


# ---------------------------------------------------------------------------
# the georeference cadence the run needs, not the one a forecast wants
# ---------------------------------------------------------------------------
class TestHistoryCadence:
    TOML = (
        "[experiment]\nname = 'x'\n\n"
        "[[domain]]\ngrid_id = 1\nnx = 132\nny = 132\n"
        "history_interval_s = 3600.0\nradt = 1.0\n")

    def test_the_root_cadence_is_rewritten(self):
        from tools.da_nowcast import retime_history

        text, was = retime_history(self.TOML, 300.0)
        assert was == 3600.0
        assert "history_interval_s = 300" in text

    def test_only_that_key_changes(self):
        from tools.da_nowcast import retime_history

        text, _ = retime_history(self.TOML, 300.0)
        before = [ln for ln in self.TOML.splitlines()
                  if "history_interval_s" not in ln]
        after = [ln for ln in text.splitlines()
                 if "history_interval_s" not in ln]
        assert before == after

    def test_only_the_first_domain_is_touched(self):
        from tools.da_nowcast import retime_history

        two = self.TOML + "\n[[domain]]\nhistory_interval_s = 900.0\n"
        text, _ = retime_history(two, 300.0)
        assert text.count("history_interval_s = 900") == 1
        assert text.count("history_interval_s = 300") == 1

    def test_a_wizard_that_stopped_emitting_it_is_a_refusal(self):
        from tools.da_nowcast import FrontDoorError, retime_history

        with pytest.raises(FrontDoorError, match="no history_interval_s"):
            retime_history("[experiment]\nname = 'x'\n", 300.0)

    def test_the_flag_exists_and_defaults_to_the_cycle_cadence(self):
        args = build_parser().parse_args(
            ["run", "--site", "qqqq", "--window-end", "latest",
             "--out", "o"])
        assert args.history_interval_seconds is None
        assert args.cycle_seconds == 900


# ---------------------------------------------------------------------------
# the card a plan was sized against is the card the run is sized against
# ---------------------------------------------------------------------------
class TestCardCarriesThrough:
    def plan(self):
        # window end at hh:05 with one 300 s cycle puts init on the
        # hour, which is the front door's rule
        return plan_window(utc(2026, 8, 5, 6, 5), cycles=1,
                           cycle_seconds=300, free_legs=0, now=NOW,
                           run_hours=4)

    def test_no_card_named_leaves_the_wizard_its_default(self):
        argv = wizard_cmd(polygon=Path("b"), out_toml=Path("t"),
                          plan=self.plan(), profile="p", name="n",
                          dx_km=3.0, source="gfs")
        assert "--vram-gib" not in argv

    def test_a_named_card_reaches_the_wizard(self):
        argv = wizard_cmd(polygon=Path("b"), out_toml=Path("t"),
                          plan=self.plan(), profile="p", name="n",
                          dx_km=3.0, source="gfs", vram_gib=31.84)
        assert argv[argv.index("--vram-gib") + 1] == "31.84"

    def test_the_front_door_takes_one(self):
        args = build_parser().parse_args(
            ["run", "--site", "qqqq", "--window-end", "latest",
             "--out", "o", "--vram-gib", "32"])
        assert args.vram_gib == 32.0


# ---------------------------------------------------------------------------
# a gallery that redraws itself cannot be checked by eye each time
# ---------------------------------------------------------------------------
class TestCreditAndTitleFit:
    FOOT = ("woof radar-DA nowcast demo (N=6), real QQQQ NEXRAD "
            "Level-II (UNSCORED demo, not campaign evidence) basemap: "
            "Natural Earth 10m + US Census counties (vendored assets)")
    SRC = "source: WOOF (model) · QQQQ Level-II (obs)"

    def test_a_wide_figure_keeps_the_two_sided_credit(self):
        from tools.da_nowcast_render import credit_layout

        assert credit_layout(24.0, self.FOOT, self.SRC)["mode"] == "sides"

    def test_a_narrow_figure_stacks_instead_of_overprinting(self):
        from tools.da_nowcast_render import credit_layout

        assert credit_layout(6.0, self.FOOT, self.SRC)["mode"] \
            == "stacked"

    def test_a_side_by_side_credit_actually_fits(self):
        from tools.da_nowcast_render import credit_layout, text_width_in

        for width in (9.6, 12.0, 14.4, 19.2, 24.0):
            layout = credit_layout(width, self.FOOT, self.SRC)
            if layout["mode"] != "sides":
                continue
            used = (text_width_in(self.FOOT, layout["foot_pt"])
                    + text_width_in(self.SRC, layout["src_pt"]))
            assert used <= width, width

    def test_a_title_that_fits_is_left_alone(self):
        from tools.da_nowcast_render import fit_title

        text, points = fit_title("short title", 20.0)
        assert (text, points) == ("short title", 12.0)

    def test_a_long_title_shrinks_before_it_wraps(self):
        from tools.da_nowcast_render import fit_title, text_width_in

        long = "x" * 160
        text, points = fit_title(long, 14.0)
        assert "\n" not in text
        assert points < 12.0
        assert text_width_in(text, points) <= 14.0

    def test_a_title_that_cannot_shrink_enough_wraps(self):
        from tools.da_nowcast_render import fit_title, text_width_in

        text, points = fit_title(" ".join(["word"] * 60), 8.0)
        assert "\n" in text
        assert points >= 8.5
        for line in text.splitlines():
            assert text_width_in(line, points) <= 8.0


# ---------------------------------------------------------------------------
# a length scale the domain can actually carry
# ---------------------------------------------------------------------------
class TestPerturbationScale:
    """A caller who draws a smaller box has not asked for different
    science; they have asked for a smaller box."""

    def test_the_bounds_come_from_the_perturbation_lane_itself(self):
        from tools.da_nowcast import perturbation_scale_bounds

        # pins the import: a rename in woof.da.perturb fails here
        # rather than three minutes into somebody's run
        floor, ceiling = perturbation_scale_bounds(
            nx=132, ny=132, dx_m=3000.0, dy_m=3000.0)
        assert floor == pytest.approx(6.0)
        assert ceiling == pytest.approx(396.0 / (2 * 3.141592653589793))

    def test_a_big_box_keeps_the_default(self):
        from tools.da_nowcast import (DEFAULT_LENGTH_SCALE_KM,
                                      resolvable_length_scale_km)

        scale, note = resolvable_length_scale_km(
            nx=132, ny=132, dx_m=3000.0, dy_m=3000.0)
        assert scale == DEFAULT_LENGTH_SCALE_KM
        assert note is None

    def test_a_small_box_is_capped_and_says_so(self):
        from tools.da_nowcast import resolvable_length_scale_km

        scale, note = resolvable_length_scale_km(
            nx=86, ny=94, dx_m=3000.0, dy_m=3000.0)
        assert scale < 50.0
        assert "span/(2*pi)" in note

    def test_the_capped_value_is_one_the_lane_accepts(self):
        from woof.da.perturb import (FieldPerturbation, _check_resolvable)
        from tools.da_nowcast import resolvable_length_scale_km

        for nx, ny, dx in ((86, 94, 3000.0), (132, 132, 3000.0),
                           (60, 60, 3000.0), (400, 300, 12000.0)):
            scale, _ = resolvable_length_scale_km(
                nx=nx, ny=ny, dx_m=dx, dy_m=dx)
            spec = FieldPerturbation.from_mapping({
                "name": "u", "amplitude": 1.5,
                "length_scale_km": scale})
            # the artifact, not a model of it: the lane's own checker
            _check_resolvable((49, ny, nx), dx / 1000.0, dx / 1000.0,
                              spec)

    def test_an_explicit_request_inside_the_range_is_honoured(self):
        from tools.da_nowcast import resolvable_length_scale_km

        scale, note = resolvable_length_scale_km(
            nx=86, ny=94, dx_m=3000.0, dy_m=3000.0, requested=20.0)
        assert (scale, note) == (20.0, None)

    def test_an_explicit_request_outside_it_is_capped_and_named(self):
        from tools.da_nowcast import resolvable_length_scale_km

        scale, note = resolvable_length_scale_km(
            nx=86, ny=94, dx_m=3000.0, dy_m=3000.0, requested=500.0)
        assert scale < 500.0
        assert "--length-scale-km 500" in note

    def test_a_domain_with_no_usable_scale_is_a_refusal(self):
        from tools.da_nowcast import FrontDoorError, resolvable_length_scale_km

        with pytest.raises(FrontDoorError, match="too small to perturb"):
            resolvable_length_scale_km(nx=8, ny=8, dx_m=3000.0,
                                       dy_m=3000.0)

    def test_the_flag_defaults_to_derived(self):
        args = build_parser().parse_args(
            ["run", "--site", "qqqq", "--window-end", "latest",
             "--out", "o"])
        assert args.length_scale_km is None


# ---------------------------------------------------------------------------
# the front door asks for more than one radar, and says so in the receipt
# ---------------------------------------------------------------------------
def parsed_run(*extra):
    """``run`` args with the required flags and whatever else is asked."""

    return build_parser().parse_args(
        ["run", "--site", "qqqq", "--window-end", "latest", "--out", "o",
         *extra])


def obs_argv(selection):
    return obs_cmd(selection=selection, valid=utc(2026, 8, 5, 4, 15),
                   grid_wrfout=Path("g"), out_nc=Path("o"),
                   work_dir=Path("w"), bucket=None,
                   dealias=DealiasChoice())


def builder_args(argv):
    """Parse a generated argv with the obs builder's REAL parser.

    The argv starts ``<python> -m tools.obs_radar_grid_build``; the
    builder's parser sees only what follows.
    """

    from tools.obs_radar_grid_build import build_parser as obs_parser

    marker = argv.index("tools.obs_radar_grid_build")
    return obs_parser().parse_args(argv[marker + 1:])


class TestRadarSelectionArgv:
    """What the front door emits is what the obs builder accepts."""

    def test_the_default_is_the_single_radar_argv_unchanged(self):
        # The adoption must be a no-op for every run that does not ask
        # for it: this is the argv the front door emitted before
        # multi-radar existed, and a changed byte here changes what
        # every existing case would rebuild as.
        argv = obs_argv(RadarSelection(anchor="QQQQ"))
        marker = argv.index("tools.obs_radar_grid_build")
        assert argv[marker + 1:marker + 3] == ["--site", "QQQQ"]
        assert "--discover-sites" not in argv
        assert "--min-radars" not in argv
        assert "--max-radar-time-spread-seconds" not in argv
        assert "--min-coverage-fraction" not in argv
        assert "--max-radars" not in argv

    def test_named_sites_put_the_anchor_first_and_deduplicate(self):
        argv = obs_argv(RadarSelection(anchor="QQQQ",
                                       sites=("RRRR", "QQQQ", "SSSS")))
        named = [argv[i + 1] for i, a in enumerate(argv) if a == "--site"]
        assert named == ["QQQQ", "RRRR", "SSSS"]

    def test_discovery_names_no_radar_at_all(self):
        argv = obs_argv(RadarSelection(anchor="QQQQ", discover=True,
                                       min_coverage_fraction=0.30,
                                       max_radars=4))
        assert "--site" not in argv
        assert "--discover-sites" in argv
        assert argv[argv.index("--min-coverage-fraction") + 1] == "0.3"
        assert argv[argv.index("--max-radars") + 1] == "4"

    def test_the_delivery_constraints_apply_to_either_route(self):
        # --min-radars and --max-radar-time-spread-seconds are about
        # what the radars DELIVERED, not how they were chosen, so they
        # ride both routes.
        for selection in (
                RadarSelection(anchor="QQQQ", sites=("RRRR",),
                               min_radars=2,
                               max_time_spread_seconds=300.0),
                RadarSelection(anchor="QQQQ", discover=True, min_radars=2,
                               max_time_spread_seconds=300.0)):
            argv = obs_argv(selection)
            assert argv[argv.index("--min-radars") + 1] == "2"
            assert argv[argv.index(
                "--max-radar-time-spread-seconds") + 1] == "300.0"

    def test_discovery_only_flags_do_not_leak_into_the_named_route(self):
        argv = obs_argv(RadarSelection(anchor="QQQQ", sites=("RRRR",),
                                       min_coverage_fraction=0.30,
                                       max_radars=4))
        assert "--min-coverage-fraction" not in argv
        assert "--max-radars" not in argv

    def test_every_emitted_argv_parses_under_the_builders_own_parser(self):
        # Verified against the artifact: the obs builder's real parser,
        # not a copy of its flag list kept here.
        from tools.obs_radar_grid_build import requested_sites

        single = builder_args(obs_argv(RadarSelection(anchor="QQQQ")))
        assert requested_sites(single) == ["QQQQ"]

        named = builder_args(obs_argv(
            RadarSelection(anchor="QQQQ", sites=("RRRR", "SSSS"),
                           min_radars=2, max_time_spread_seconds=300.0)))
        assert requested_sites(named) == ["QQQQ", "RRRR", "SSSS"]
        assert named.min_radars == 2
        assert named.max_radar_time_spread_seconds == 300.0

        found = builder_args(obs_argv(
            RadarSelection(anchor="QQQQ", discover=True,
                           min_coverage_fraction=0.30, max_radars=4,
                           min_radars=2)))
        assert requested_sites(found) == []      # discovery: computed later
        assert found.discover_sites is True
        assert found.min_coverage_fraction == 0.30
        assert found.max_radars == 4

        # The engine selector crosses the same seam, and the builder's own
        # parser is what has to accept it -- this front door BUILDS that
        # tool's argv, so a spelling that only this file agrees with is a
        # run that dies at the first observation stage.
        from woof.obs.dealias import DealiasParams
        from tools.obs_radar_grid_build import dealias_params_from_args

        refined = builder_args(obs_cmd(
            selection=RadarSelection(anchor="QQQQ"),
            valid=utc(2026, 8, 5, 4, 15), grid_wrfout=Path("g"),
            out_nc=Path("o"), work_dir=Path("w"), bucket=None,
            dealias=DealiasChoice(on=True, engine="region-global",
                                  refinement=True)))
        assert refined.dealias is True
        assert refined.dealias_engine == "region-global"
        assert refined.dealias_refinement is True
        params = dealias_params_from_args(refined, DealiasParams,
                                          lambda engine: None)
        assert params.engine == "region-global" and params.refinement is True

        # Both halves of the refinement switch cross it too, and the
        # legacy engine's argv has to survive the round trip now that the
        # default is the other one.
        legacy = builder_args(obs_cmd(
            selection=RadarSelection(anchor="QQQQ"),
            valid=utc(2026, 8, 5, 4, 15), grid_wrfout=Path("g"),
            out_nc=Path("o"), work_dir=Path("w"), bucket=None,
            dealias=DealiasChoice(on=True, engine="vad-region",
                                  refinement=False)))
        assert legacy.dealias_engine == "vad-region"
        assert legacy.dealias_refinement is False
        assert dealias_params_from_args(
            legacy, DealiasParams, lambda engine: None).refinement is False

    def test_the_front_door_and_the_parameter_object_default_alike(self):
        """One default, spelled in two places, bound here.

        ``tools.da_nowcast`` imports no numpy -- it builds argv for a
        subprocess -- so it carries the engine name as a string.  A string
        only one side agrees with is a run that dies at its first
        observation stage, hours in.
        """

        from woof.obs.dealias import DealiasParams
        from tools.obs_radar_grid_build import (build_parser,
                                                dealias_params_from_args)

        shipped = DealiasParams()
        assert DEFAULT_DEALIAS_ENGINE == shipped.engine
        assert DealiasChoice().engine == shipped.engine
        assert DealiasChoice().refinement is shipped.refinement is True

        # And the builder's own parser, asked for nothing but --dealias,
        # produces exactly those parameters.
        args = build_parser().parse_args([
            "--site", "QQQQ", "--valid-time", "2026-08-05T04:00:00Z",
            "--grid-wrfout", "g", "--out", "o", "--work-dir", "w"])
        # Re-anchored: --dealias is gone, because dealiasing is what a bare
        # ingest does.  Asked for NOTHING, the builder's parser produces
        # exactly the shipped parameters.
        assert args.dealias is True
        params = dealias_params_from_args(args, DealiasParams,
                                          lambda engine: None)
        assert params.engine == shipped.engine
        assert params.refinement is True
        # And the off switch is the one that has to be spelled now.
        off = build_parser().parse_args([
            "--site", "QQQQ", "--valid-time", "2026-08-05T04:00:00Z",
            "--grid-wrfout", "g", "--out", "o", "--work-dir", "w",
            "--no-dealias"])
        assert off.dealias is False
        assert dealias_params_from_args(
            off, DealiasParams, lambda engine: None) is None

    def test_refinement_beside_the_engine_without_one_is_refused(self):
        """Named at the door, not as a traceback from the parameters."""

        from woof.obs.dealias import DealiasParams
        from tools.obs_radar_grid_build import (build_parser,
                                                dealias_params_from_args)

        args = build_parser().parse_args([
            "--site", "QQQQ", "--valid-time", "2026-08-05T04:00:00Z",
            "--grid-wrfout", "g", "--out", "o", "--work-dir", "w",
            "--dealias-engine", "vad-region",
            "--dealias-refinement"])
        with pytest.raises(SystemExit) as error:
            dealias_params_from_args(args, DealiasParams,
                                     lambda engine: None)
        assert "no refinement pass" in str(error.value)


class TestRadarSelectionFromTheFrontDoor:
    def test_the_flags_default_to_one_radar(self):
        selection = radar_selection(parsed_run())
        assert selection == RadarSelection(anchor="QQQQ")
        assert selection.multi is False

    def test_sites_are_uppercased_arguments(self):
        selection = radar_selection(
            parsed_run("--sites", "rrrr", "--sites", "ssss"))
        assert selection.sites == ("RRRR", "SSSS")
        assert selection.multi is True

    def test_discovery_and_naming_together_are_refused(self):
        with pytest.raises(FrontDoorError, match="mutually exclusive"):
            radar_selection(parsed_run("--sites", "rrrr",
                                       "--discover-sites"))

    def test_the_tuning_flags_reach_the_selection(self):
        selection = radar_selection(parsed_run(
            "--discover-sites", "--min-coverage-fraction", "0.30",
            "--max-radars", "4", "--min-radars", "2",
            "--max-radar-time-spread-seconds", "300"))
        assert selection.discover is True
        assert selection.min_coverage_fraction == 0.30
        assert selection.max_radars == 4
        assert selection.min_radars == 2
        assert selection.max_time_spread_seconds == 300.0


class TestTheVerifierGradesAgainstWhatItAssimilated:
    """A multi-radar nowcast graded against a single-radar truth field
    would be scored on a different observation than it was given, and
    the difference would look like skill."""

    def test_the_selection_round_trips_through_a_receipt(self):
        selection = RadarSelection(
            anchor="QQQQ", discover=True, min_coverage_fraction=0.30,
            max_radars=4, min_radars=2, max_time_spread_seconds=300.0)
        back = RadarSelection.from_payload(selection.to_payload(),
                                           anchor="QQQQ")
        assert back == selection

    def test_named_sites_round_trip_too(self):
        selection = RadarSelection(anchor="QQQQ", sites=("RRRR", "SSSS"),
                                   min_radars=2)
        back = RadarSelection.from_payload(selection.to_payload(),
                                           anchor="QQQQ")
        assert back == selection
        assert obs_argv(back) == obs_argv(selection)

    def test_a_receipt_without_radars_reads_back_as_one_radar(self):
        # Every case written before this existed was single-radar, and
        # its verifier must keep building exactly what its cycle did.
        assert RadarSelection.from_payload(None, anchor="QQQQ") \
            == RadarSelection(anchor="QQQQ")

    def test_the_case_context_recovers_the_selection(self, tmp_path):
        from tools.da_nowcast import _case_context

        selection = RadarSelection(anchor="QQQQ", sites=("RRRR",),
                                   min_radars=2)
        (tmp_path / "nowcast-receipt.json").write_text(json.dumps({
            "schema": "gpuwm-da.nowcast.v1", "site": "QQQQ",
            "radars": selection.to_payload(),
            "plan": {"init": "2026-08-05T04:00:00Z", "cycle_seconds": 900,
                     "free_leg_times": list(FREE_TIMES)}}),
            encoding="utf-8")
        recovered, init, _, dealias = _case_context(tmp_path)
        assert recovered == selection
        assert init == utc(2026, 8, 5, 4)
        # No "obs" block: this receipt predates --dealias, and every run
        # written before the flag existed masked rather than unfolded.
        assert dealias == DealiasChoice(on=False,
                                        engine=DealiasChoice.LEGACY_ENGINE,
                                        refinement=False)

    def test_the_case_context_recovers_the_dealias_treatment(self, tmp_path):
        """The truth field has to be built the way the analysis was fed.

        Dealiasing changes which velocity gates exist and what they are
        worth.  A run that assimilated unfolded velocities and was graded
        against a masked-only composite would be scored on an observation
        nobody gave it -- and the mismatch would read as skill, which is
        the specific way this class of bug hides.
        """

        from tools.da_nowcast import _case_context

        selection = RadarSelection(anchor="QQQQ", sites=("RRRR",),
                                   min_radars=2)
        (tmp_path / "nowcast-receipt.json").write_text(json.dumps({
            "schema": "gpuwm-da.nowcast.v1", "site": "QQQQ",
            "radars": selection.to_payload(),
            "obs": {"dealias": True},
            "plan": {"init": "2026-08-05T04:00:00Z", "cycle_seconds": 900,
                     "free_leg_times": list(FREE_TIMES)}}),
            encoding="utf-8")
        _, _, _, dealias = _case_context(tmp_path)
        # The receipt predates the engine selector, so it reads back as the
        # engine that was the only one when it was written -- NOT as
        # today's default.  An old receipt describes an old run, and its
        # verification composites have to be built the way its analysis
        # was fed.
        assert dealias == DealiasChoice(on=True, engine="vad-region",
                                        refinement=False)
        assert DealiasChoice.LEGACY_ENGINE == "vad-region"

    def test_the_non_radar_streams_reach_the_cycle_driver(self):
        """The flag that never reached the driver is this project's bug.

        --goes-cwp and --surface-obs existed only on the cycle driver's own
        command line, so no run launched from this front door -- which is
        every WaH run -- could assimilate either.  This pins that the front
        door now emits them, and that leaving them off emits nothing.
        """

        from tools.da_nowcast import cycle_cmd

        base = dict(
            prepared_root=Path("prep"), authority_dir=Path("auth"),
            profile="none",
            plan=plan_window(utc(2026, 8, 5, 5, 30), cycles=2,
                             cycle_seconds=900, free_legs=2,
                             now=utc(2026, 8, 5, 5, 45), source="hrrr"),
            members=4, obs_files=[],
            grid_wrfouts=[], cycle_out=Path("out"), proof_sha="0" * 64,
            manifest_sha="0" * 64, content_sha="0" * 64, seed=1,
            solve_device="host", horizontal_loc_m=12000.0,
            vertical_loc_m=3000.0, length_scale_km=6.0, source="hrrr")

        off = cycle_cmd(**base)
        assert "--goes-cwp" not in off
        assert "--surface-obs" not in off
        assert "--clear-air-analysis" not in off

        on = cycle_cmd(**base, clear_air_analysis=True,
                       surface_obs=Path("sfc.json"),
                       goes_cwp=[Path("g0.nc"), Path("g1.nc")],
                       cwp_vertical_loc_m=20000.0)
        assert "--clear-air-analysis" in on
        assert on[on.index("--surface-obs") + 1] == str(Path("sfc.json"))
        # One --goes-cwp per file, in the order given: the driver matches
        # satellite files to legs by POSITION.
        assert [on[i + 1] for i, tok in enumerate(on)
                if tok == "--goes-cwp"] == [str(Path("g0.nc")),
                                            str(Path("g1.nc"))]
        assert on[on.index("--cwp-vertical-loc-m") + 1] == "20000.0"

    def test_the_dealias_flag_reaches_both_obs_call_sites(self):
        """One flag, two builders, or the run grades itself crooked.

        ``obs_cmd`` serves the assimilated observations and the verifier's
        truth composites both.  Re-anchored now that dealiasing is the
        default: what has to be spelled is OFF.  A receipt from a run that
        did not dealias must re-run its composites without dealiasing, so
        an empty tail on the off path would be the A/B break this object
        exists to prevent.
        """

        from tools.da_nowcast import obs_cmd

        kwargs = dict(selection=RadarSelection(anchor="QQQQ"),
                      valid=utc(2026, 8, 5, 4),
                      grid_wrfout=Path("wrfout"), out_nc=Path("out.nc"),
                      work_dir=Path("vols"), bucket=None)
        on = obs_cmd(**kwargs, dealias=DealiasChoice(on=True))
        assert "--no-dealias" not in on
        # The engine travels with the flag: a truth composite built by the
        # other solver is a differently-built truth.  So does the
        # refinement switch, spelled either way rather than left to
        # whatever the default is on the day the verifier re-runs.
        assert on[on.index("--dealias-engine") + 1] == DEFAULT_DEALIAS_ENGINE
        assert "--dealias-refinement" in on
        legacy = obs_cmd(**kwargs, dealias=DealiasChoice(
            on=True, engine="vad-region", refinement=False))
        assert legacy[legacy.index("--dealias-engine") + 1] == "vad-region"
        assert "--no-dealias-refinement" in legacy
        assert "--dealias-refinement" not in legacy
        off = obs_cmd(**kwargs, dealias=DealiasChoice(on=False))
        assert "--no-dealias" in off
        assert "--dealias-engine" not in off
        # A bare DealiasChoice() now means ON, like every door.
        assert DealiasChoice().on is True
        assert "--no-dealias" not in obs_cmd(
            **kwargs, dealias=DealiasChoice())

    def test_a_verification_frame_is_built_from_every_radar(
            self, tmp_path, monkeypatch):
        # The frame the gallery grades against is built by the same
        # obs_cmd, from the same selection, as the cycle it grades.
        import tools.da_nowcast as front

        seen = {}

        class Done:
            returncode = 0
            stdout = ""
            stderr = ""

        def capture(argv, **kwargs):
            seen["argv"] = argv
            (tmp_path / "obsverify").mkdir(parents=True, exist_ok=True)
            return Done()

        monkeypatch.setattr(front.subprocess, "run", capture)
        selection = RadarSelection(anchor="QQQQ", sites=("RRRR", "SSSS"),
                                   min_radars=2)
        ok, why = front.build_verify_frame(
            case_dir=tmp_path, selection=selection,
            init=utc(2026, 8, 5, 4), valid=utc(2026, 8, 5, 5, 45),
            bucket=None, max_offset_seconds=480.0, repo_root=tmp_path,
            dealias=DealiasChoice())
        assert ok, why
        named = [seen["argv"][i + 1]
                 for i, a in enumerate(seen["argv"]) if a == "--site"]
        assert named == ["QQQQ", "RRRR", "SSSS"]
        assert seen["argv"][seen["argv"].index("--min-radars") + 1] == "2"
        # named for the anchor, so a case's files stay predictable
        assert seen["argv"][seen["argv"].index("--out") + 1].endswith(
            "verify-qqqq-202608050545.nc")


# ---------------------------------------------------------------------------
# an observed composite is named after every radar that built it
# ---------------------------------------------------------------------------
class TestRadarLabelling:
    """A three-radar composite captioned with one radar's id would
    understate the analysis it came from."""

    def char_rows(self, *ids, width=8):
        import numpy as np

        return np.ma.array([[c.encode() for c in name.ljust(width)]
                            for name in ids])

    def test_one_radar_reads_back_as_itself(self):
        from tools.da_nowcast_render import radar_ids, site_label

        ids = radar_ids(self.char_rows("QQQQ"))
        assert ids == ["QQQQ"]
        # single-radar galleries must be unchanged by multi-radar support
        assert site_label(ids) == "QQQQ"

    def test_every_contributing_radar_is_named(self):
        from tools.da_nowcast_render import radar_ids, site_label

        ids = radar_ids(self.char_rows("QQQQ", "RRRR", "SSSS"))
        assert ids == ["QQQQ", "RRRR", "SSSS"]
        assert site_label(ids) == "QQQQ+RRRR+SSSS"

    def test_the_anchor_stays_first(self):
        # the first row sites the domain and is what the map marks
        from tools.da_nowcast_render import radar_ids

        assert radar_ids(self.char_rows("QQQQ", "RRRR"))[0] == "QQQQ"

    def test_padding_is_stripped(self):
        from tools.da_nowcast_render import radar_ids

        assert radar_ids(self.char_rows("QQQQ", width=12)) == ["QQQQ"]


# ---------------------------------------------------------------------------
# the fine nest over the free forecast, asked for at this door
# ---------------------------------------------------------------------------
class TestTheNestReachesTheCycleFromTheFrontDoor:
    """The nest belongs to the cycle driver; this door forwards it.

    The prepared cycle driver has carried ``--nest-*`` all along, and
    the gallery has drawn nest composites all along.  What was missing
    was the wire between them: a run launched from this front door --
    which is every run a user launches -- could not ask for one, and its
    receipt said ``nested_free_forecast: false`` whatever had happened.
    """

    def plan(self) -> WindowPlan:
        return plan_window(utc(2026, 8, 5, 5, 30), cycles=6,
                           cycle_seconds=900, free_legs=6, now=NOW)

    def base(self) -> dict:
        return dict(
            prepared_root=Path("p"), authority_dir=Path("a"),
            profile="prof", plan=self.plan(), members=8,
            obs_files=[Path("o")], grid_wrfouts=[Path("g")],
            cycle_out=Path("c"), proof_sha="x", manifest_sha="y",
            content_sha="z", seed=1, solve_device="cuda",
            horizontal_loc_m=12000.0, vertical_loc_m=3000.0,
            length_scale_km=50.0, source="hrrr")

    def parse(self, *extra: str):
        return build_parser().parse_args(
            ["run", "--site", "QQQQ", "--window-end", "latest",
             "--out", "case", *extra])

    def test_a_run_that_asks_for_no_nest_builds_the_argv_it_always_did(self):
        from tools.da_nowcast import NestChoice

        nest = NestChoice.from_args(self.parse())
        assert not nest.requested
        assert nest.argv_tail() == []
        argv = cycle_cmd(**self.base(), nest=nest)
        assert not [token for token in argv if token.startswith("--nest")]
        assert argv == cycle_cmd(**self.base())

    def test_an_extent_turns_it_on_and_is_forwarded_flag_for_flag(self):
        from tools.da_nowcast import NestChoice

        nest = NestChoice.from_args(self.parse(
            "--nest-half-width-km", "60", "--nest-members", "2",
            "--nest-history-interval-s", "300"))
        assert nest.requested
        argv = cycle_cmd(**self.base(), nest=nest)
        assert argv[argv.index("--nest-half-width-km") + 1] == "60.0"
        assert argv[argv.index("--nest-members") + 1] == "2"
        assert argv[argv.index("--nest-history-interval-s") + 1] == "300.0"
        # the ratio is always spelled out once a nest is asked for, so a
        # receipt replay cannot inherit a different default later
        assert argv[argv.index("--nest-ratio") + 1] == "3"

    def test_cells_are_an_extent_too(self):
        from tools.da_nowcast import NestChoice

        nest = NestChoice.from_args(
            self.parse("--nest-nx", "120", "--nest-ny", "120"))
        assert nest.requested
        argv = cycle_cmd(**self.base(), nest=nest)
        assert argv[argv.index("--nest-nx") + 1] == "120"
        assert argv[argv.index("--nest-ny") + 1] == "120"

    def test_the_ratio_default_is_the_nest_geometrys_own(self):
        from woof.da.nested_forecast import NestGeometry
        from tools.da_nowcast import default_nest_ratio

        assert default_nest_ratio() == NestGeometry.ratio

    def test_an_admissibility_acknowledgement_is_carried(self):
        from tools.da_nowcast import NestChoice

        nest = NestChoice.from_args(self.parse(
            "--nest-half-width-km", "60",
            "--nest-acknowledge", "nested-forecast:sub-gray-zone-pbl"))
        argv = cycle_cmd(**self.base(), nest=nest)
        assert argv[argv.index("--nest-acknowledge") + 1] \
            == "nested-forecast:sub-gray-zone-pbl"

    def test_the_three_cli_refusals_are_met_before_the_card(self):
        from tools.da_nowcast import validate_analysis_flags

        with pytest.raises(SystemExit):
            # the nest runs over the free legs; without them there is
            # nothing for it to refine
            validate_analysis_flags(
                self.parse("--nest-half-width-km", "60",
                           "--free-legs", "0"))
        with pytest.raises(SystemExit):
            validate_analysis_flags(self.parse("--nest-members", "2"))
        with pytest.raises(SystemExit):
            validate_analysis_flags(
                self.parse("--nest-half-width-km", "60",
                           "--members", "2", "--nest-members", "4"))
        # and the asked-for shape passes
        validate_analysis_flags(
            self.parse("--nest-half-width-km", "60", "--members", "4",
                       "--nest-members", "2"))

    def test_the_receipt_names_the_frames_and_the_free_legs(self):
        """The forecast is an output, so the receipt names where it is.

        The frames were always written.  Nothing named them, so a
        reader who wanted the forecast rather than the gallery had to
        find them by listing directories.
        """
        from tools.da_nowcast import forecast_frames_block

        block = forecast_frames_block(Path("case") / "cycle",
                                      cycles=2, free_legs=2)
        assert block["free_legs"] == ["leg02", "leg03"]
        assert block["dir"].endswith("composites")
        assert "wrfout_legNN" in block["wrfout"]
        assert forecast_frames_block(Path("c"), cycles=6,
                                     free_legs=0)["free_legs"] == []
        # The driver's numbering has ONE author.  This door pins the
        # offset to zero; the receipt follows whatever it is given
        # rather than re-deriving a second convention beside the
        # driver's own (tools/da_nowcast_auto.py passes one).
        from tools.da_nowcast import CYCLE_LEG_NUMBER_OFFSET

        assert CYCLE_LEG_NUMBER_OFFSET == 0
        assert block["leg_number_offset"] == CYCLE_LEG_NUMBER_OFFSET
        assert forecast_frames_block(
            Path("c"), cycles=2, free_legs=2,
            leg_number_offset=5)["free_legs"] == ["leg07", "leg08"]
        # the nest writes beside the parent frames, so the block names
        # that pair too or a reader finds only half the run's output
        assert "_dNN" in block["nested"]
        # and the block says WHAT the frame is.  The wrfout is a
        # one-level surface snapshot of the composite; a reader of the
        # receipt alone would otherwise take it for the leg's full
        # state, which is the state this door does not keep.
        assert "one-level" in block["vertical"]
        assert "three-dimensional leg state is not kept" in block["vertical"]
        assert "da_cycle_prepared.py" in block["vertical"]
        # one measured field, and the rest of the file is structure
        assert block["measured_field"].startswith("composite reflectivity")
        assert "surface_wrfout.py" in block["measured_field"]

    def test_the_payload_tells_a_reader_what_was_asked_for(self):
        from tools.da_nowcast import NestChoice

        off = NestChoice.from_args(self.parse()).to_payload()
        assert off == {"requested": False}
        on = NestChoice.from_args(
            self.parse("--nest-half-width-km", "45")).to_payload()
        assert on["requested"] is True
        assert on["half_width_km"] == 45.0
        assert on["ratio"] == 3
