"""``woof render --compare``: what the door decides, and nothing it draws.

Every pixel and every number of a comparison sheet is the Rust engine's
(``rw_compare``), and its own tests hold those.  What this door decides is
which files are one run's frames, which earlier frame an hourly
accumulation needs beside it, and what the engine is told -- so that is
what is held here, without an engine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from woof import cli, render_compare, rustwx_lanes


def _touch(folder: Path, *names: str) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = folder / name
        path.write_bytes(b"")
        paths.append(path)
    return paths


def _args(**overrides) -> argparse.Namespace:
    words = ["render", "--compare", "hrrr", "frame"]
    args = cli.build_parser(render_only=True).parse_args(words)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_a_frame_name_gives_its_domain_and_valid_time_in_both_spellings():
    colon = render_compare.frame_identity(
        Path("wrfout_d01_2026-10-03_06:00:00"))
    underscore = render_compare.frame_identity(
        Path("wrfout_d01_2026-10-03_06_00_00"))
    assert colon == underscore
    assert colon[0] == "d01"
    assert colon[1].isoformat() == "2026-10-03T06:00:00"
    assert render_compare.frame_identity(Path("model_output.nc")) is None
    assert render_compare.frame_identity(
        Path("wrfout_d01_2026-13-40_06_00_00")) is None


def test_a_run_folder_becomes_its_history_frames(tmp_path):
    run = tmp_path / "run"
    frames = _touch(run / "out" / "wrfout",
                    "wrfout_d01_2026-10-03_05_00_00",
                    "wrfout_d01_2026-10-03_06_00_00")
    _touch(run / "out", "report.json")
    assert render_compare.expand_inputs([run]) == sorted(frames)
    # The frames' own folder names the same frames.
    assert render_compare.expand_inputs([run / "out" / "wrfout"]) == sorted(frames)
    # A file is taken as given, whatever it is called.
    assert render_compare.expand_inputs([frames[0]]) == [frames[0]]


def test_a_folder_with_no_frames_is_refused_with_where_it_looked(tmp_path):
    _touch(tmp_path / "empty", "notes.txt")
    with pytest.raises(ValueError) as refusal:
        render_compare.expand_inputs([tmp_path / "empty"])
    text = str(refusal.value)
    assert "out/wrfout" in text and "wrfout_dNN" in text


def test_the_frame_an_hour_earlier_rides_along_as_context(tmp_path):
    folder = tmp_path / "wrfout"
    earlier, kept = _touch(folder, "wrfout_d01_2026-10-03_05_00_00",
                           "wrfout_d01_2026-10-03_06_00_00")
    groups = render_compare.group_frames([kept])
    assert groups == [([kept], [earlier])]
    # Asked for itself, the earlier frame is compared, not context; its own
    # earlier frame does not exist, so nothing rides along.
    groups = render_compare.group_frames([kept, earlier])
    assert groups == [([earlier, kept], [])]


def test_each_nest_of_a_run_goes_to_the_engine_on_its_own(tmp_path):
    folder = tmp_path / "wrfout"
    outer, inner = _touch(folder, "wrfout_d01_2026-10-03_06_00_00",
                          "wrfout_d02_2026-10-03_06_00_00")
    groups = render_compare.group_frames([inner, outer])
    assert [members for members, _ in groups] == [[outer], [inner]]


def test_the_engine_is_told_everything_the_door_was(tmp_path):
    frame = tmp_path / "wrfout_d01_2026-10-03_21_00_00"
    context = tmp_path / "wrfout_d01_2026-10-03_20_00_00"
    args = _args(out=tmp_path / "render", compare="hrrr",
                 products="t2m,refc", compare_difference="off",
                 compare_cycle="2026100300", compare_offline=True,
                 compare_gallery=tmp_path / "gallery",
                 compare_reference_dir=tmp_path / "hrrr",
                 compare_cache=tmp_path / "cache",
                 source_label="WOOF test", compare_label=None,
                 layout="flat")
    command = render_compare.engine_command(
        Path("rw_compare"), args, store=tmp_path / "store", frames=[frame],
        context=[context], size=(1200, 900))

    def after(flag):
        return command[command.index(flag) + 1]

    assert after("--reference") == "hrrr"
    assert after("--products") == "t2m,refc"
    assert after("--difference") == "off"
    assert after("--layout") == "flat"
    assert after("--cycle") == "2026100300"
    assert (after("--width"), after("--height")) == ("1200", "900")
    assert after("--source-label") == "WOOF test"
    assert after("--run-label") == render_compare.RUN_LABEL
    assert after("--flat-dir") == str(tmp_path / "gallery")
    assert after("--reference-dir") == str(tmp_path / "hrrr")
    assert after("--reference-cache") == str(tmp_path / "cache")
    assert "--offline" in command
    assert after("--context") == str(context)
    # The frames to draw are the positionals, last.
    assert command[-1] == str(frame)


def test_reference_lists_follow_the_native_catalogue_and_keep_order():
    catalog = render_compare.reference_catalog(
        "REFERENCE\thrrr\tHRRR\nREFERENCE\trrfs\tRRFS\n"
        "REFERENCE\tmrms\tMRMS\nCAPABILITY\tmrms\trefc\tobservation\n"
        "CAPABILITY\tmrms\tqpf1h\tobservation\n")
    assert catalog["mrms"]["products"] == {"refc": "observation", "qpf1h": "observation"}
    assert render_compare.selected_references(" hrrr, mrms,rrfs,hrrr ", catalog) == ["hrrr", "mrms", "rrfs"]
    with pytest.raises(ValueError, match="native build knows"):
        render_compare.selected_references("hrrr,unknown", catalog)
    with pytest.raises(ValueError):
        render_compare.selected_references("hrrr,", catalog)


def test_multi_reference_command_preserves_the_requested_native_order(tmp_path):
    args = _args(out=tmp_path / "render", compare="rrfs,hrrr,mrms", products="refc,qpf1h")
    command = render_compare.engine_command(Path("rw_compare"), args, store=tmp_path / "store",
        frames=[tmp_path / "frame"], context=[], size=(1200, 900))
    assert command[command.index("--reference") + 1] == "rrfs,hrrr,mrms"
    assert command[command.index("--difference") + 1] == "auto"


def test_comparison_forwards_an_explicit_theme_and_keeps_default_implicit(tmp_path):
    args = _args(out=tmp_path / "render", theme=None)
    command = render_compare.engine_command(Path("rw_compare"), args, store=tmp_path / "store",
        frames=[tmp_path / "frame"], context=[], size=(1200, 900))
    assert "--theme" not in command
    args.theme = "woof-dark"
    command = render_compare.engine_command(Path("rw_compare"), args, store=tmp_path / "store",
        frames=[tmp_path / "frame"], context=[], size=(1200, 900))
    assert command[command.index("--theme") + 1] == "woof-dark"


def test_observation_source_lines_keep_actual_seconds(capsys):
    render_compare._relay("SOURCE\tmrms\tobserved\t2026-10-03T13:00:37+00:00\trefc\tfile.grib2.gz\n", {})
    assert "observed 2026-10-03T13:00:37+00:00 product refc" in capsys.readouterr().out


def test_the_cache_is_kept_between_renders_unless_told_otherwise(tmp_path):
    args = _args(out=tmp_path / "render")
    command = render_compare.engine_command(
        Path("rw_compare"), args, store=tmp_path / "store",
        frames=[tmp_path / "frame"], context=[], size=(1200, 900))
    cache = command[command.index("--reference-cache") + 1]
    assert cache == str(render_compare.default_reference_cache())
    assert "--offline" not in command and "--cycle" not in command


def test_the_fast_and_the_full_parser_agree_on_the_comparison_flags():
    words = ["render", "--compare", "hrrr", "--compare-cycle", "2026100300",
             "--compare-difference", "on", "--compare-gallery", "flat",
             "--compare-offline", "run-folder"]
    fast = vars(cli.build_parser(render_only=True).parse_args(words))
    full = vars(cli.build_parser().parse_args(words))
    full.pop("ingest_preflight_handler", None)
    assert fast == full
    assert fast["compare"] == "hrrr"


def test_an_unbuilt_engine_is_refused_with_the_line_that_builds_it(
        monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rustwx_lanes, "find_compare_bin", lambda: None)
    frame = _touch(tmp_path, "wrfout_d01_2026-10-03_06_00_00")[0]
    args = _args(wrfout=[frame], out=tmp_path / "render")
    assert render_compare.compare_main(args) == 2
    said = capsys.readouterr().err
    assert "rw_compare" in said and "cargo build" in said
    assert not (tmp_path / "render").exists(), "a refusal writes nothing"


def test_a_comparison_does_not_combine_with_pair_sheets(tmp_path, capsys):
    from woof import render

    args = _args(pair=[tmp_path / "a", tmp_path / "b"], wrfout=[])
    assert render.render_main(args) == 2
    assert "--pair" in capsys.readouterr().err


def test_engine_lines_are_relayed_in_the_door_s_own_words(capsys):
    tally = {"rendered": [], "skipped": 0, "failed": 0}
    render_compare._relay("RENDERED\tt2m\tf006\tout/sheet.png\n", tally)
    render_compare._relay(
        "SKIPPED\tqpf1h\t/run/wrfout_d01_2026-10-03_06_00_00\tno earlier "
        "frame\n", tally)
    render_compare._relay(
        "FAILED\t*\t/run/wrfout_d01_2026-10-03_06_00_00\tfetch failed\n",
        tally)
    render_compare._relay("FINISHED rendered=1 skipped=1 failed=1\n", tally)
    assert tally == {"rendered": ["out/sheet.png"], "skipped": 1, "failed": 1}
    said = capsys.readouterr()
    assert "render: out/sheet.png" in said.out
    assert "skipped qpf1h for wrfout_d01_2026-10-03_06_00_00" in said.err
    assert "FAILED every product" in said.err


# ---------------------------------------------------------------------------
# `woof go` finalize: a run is drawn beside the model it started from.
# THE BREAKAGE: every HRRR-sourced run is to be read WOOF-left, HRRR-right
# through the Rust engine, and that was reachable only by an operator typing
# `woof render --compare hrrr` after the run.
# ---------------------------------------------------------------------------

_CATALOG = {
    "hrrr": {"label": "HRRR", "products": {"refc": "forecast", "t2m": "forecast"}},
    "rrfs": {"label": "RRFS", "products": {"refc": "forecast"}},
    "mrms": {"label": "MRMS", "products": {"refc": "observation"}},
}


def test_the_reference_comes_from_the_source_table_not_a_model_branch():
    from woof import go_cli

    # The native route and the mapped pressure-level route both name HRRR
    # as their upstream model in the source table.
    assert go_cli.compare_reference("hrrr", _CATALOG) == "hrrr"
    assert go_cli.compare_reference("hrrr-prs", _CATALOG) == "hrrr"
    # A source whose upstream model the engine cannot draw has no reference.
    assert go_cli.compare_reference("gfs", _CATALOG) is None
    assert go_cli.compare_reference(None, _CATALOG) is None
    assert go_cli.compare_reference("not-a-source", _CATALOG) is None
    # An observation-only entry is never a forecast reference.
    only_observed = {"hrrr": {"label": "HRRR", "products": {"refc": "observation"}}}
    assert go_cli.compare_reference("hrrr", only_observed) is None
    # And a model the engine's catalogue lacks is simply not compared.
    assert go_cli.compare_reference("hrrr", {}) is None


def _fake_engine(tmp_path: Path) -> Path:
    engine = tmp_path / "rw_compare"
    engine.write_text(
        "#!/bin/sh\n"
        "printf 'REFERENCE\thrrr\tHRRR\nCAPABILITY\thrrr\trefc\tforecast\n'\n")
    engine.chmod(0o755)
    return engine


def _go_plan(tmp_path: Path, source: str) -> dict:
    run = tmp_path / "case" / "run"
    _touch(run / "wrfout", "wrfout_d01_2026-10-03_06_00_00")
    return {"source": source, "run": run, "wrfout_dir": run / "wrfout",
            "render": tmp_path / "case" / "png"}


@pytest.mark.skipif(__import__("os").name == "nt", reason="shell-script engine stand-in")
def test_an_hrrr_run_draws_its_comparison_sheets_by_default(tmp_path, monkeypatch):
    from woof import go_cli

    monkeypatch.setattr(rustwx_lanes, "require_compare_bin", lambda: _fake_engine(tmp_path))
    ran = []
    monkeypatch.setattr(go_cli, "_run_stage",
                        lambda label, command, **kw: ran.append((label, command, kw)))
    plan = _go_plan(tmp_path, "hrrr")
    go_cli._reference_compare_stage(plan, explain=False)
    assert len(ran) == 1
    label, command, kw = ran[0]
    assert label == "compare hrrr"
    assert command[1:4] == ["-m", "woof.cli", "render"]
    assert str(plan["wrfout_dir"]) in command
    at = command.index("--compare")
    assert command[at + 1] == "hrrr"
    assert command[command.index("--out") + 1] == str(plan["render"])
    assert command[command.index("--products") + 1] == go_cli.COMPARE_PRODUCTS
    assert command[-2:] == ["--run-stamp", "off"], (
        "the stage draws into the run folder go already claimed")
    # Additive: its failure must never be said as the run stopping.
    assert kw["secondary_failure"]() is True


@pytest.mark.skipif(__import__("os").name == "nt", reason="shell-script engine stand-in")
def test_the_compare_stage_reads_the_runs_own_download_before_fetching(tmp_path, monkeypatch):
    # WOOF 1.1 readiness g1: the stage re-downloaded the HRRR cycle the run
    # already had (724.6 s finalize, 2.7 GB); with the run's own download
    # folder the same sheets took 43 s.  The fetch fallback stays (no
    # --compare-offline), and a run that starts past its cycle keeps fetching
    # because flat file names carry only the cycle hour.
    from woof import go_cli

    monkeypatch.setattr(rustwx_lanes, "require_compare_bin", lambda: _fake_engine(tmp_path))
    ran = []
    monkeypatch.setattr(go_cli, "_run_stage",
                        lambda label, command, **kw: ran.append(command))
    download = tmp_path / "case" / "downloads" / "key"
    download.mkdir(parents=True)
    plan = {**_go_plan(tmp_path, "hrrr"), "data": download, "forecast_start_hour": 0}
    go_cli._reference_compare_stage(plan, explain=False)
    command = ran[-1]
    assert command[command.index("--compare-reference-dir") + 1] == str(download)
    assert "--compare-offline" not in command
    go_cli._reference_compare_stage({**plan, "forecast_start_hour": 24}, explain=False)
    assert "--compare-reference-dir" not in ran[-1]


@pytest.mark.skipif(__import__("os").name == "nt", reason="shell-script engine stand-in")
def test_a_run_with_no_reference_model_draws_nothing_extra(tmp_path, monkeypatch, capsys):
    from woof import go_cli

    monkeypatch.setattr(rustwx_lanes, "require_compare_bin", lambda: _fake_engine(tmp_path))
    ran = []
    monkeypatch.setattr(go_cli, "_run_stage", lambda *a, **kw: ran.append(a))
    go_cli._reference_compare_stage(_go_plan(tmp_path, "gfs"), explain=False)
    assert ran == []
    assert capsys.readouterr().out == ""


def test_a_missing_engine_is_said_once_and_the_run_still_succeeds(tmp_path, monkeypatch, capsys):
    from woof import go_cli

    def refuse():
        raise RuntimeError("comparison sheets come from the rust engine rw_compare, "
                           "and it is not built.")

    monkeypatch.setattr(rustwx_lanes, "require_compare_bin", refuse)
    go_cli._reference_compare_stage(_go_plan(tmp_path, "hrrr"), explain=False)
    said = capsys.readouterr().out
    assert said.count("compare skipped") == 1
    assert "rw_compare" in said


@pytest.mark.skipif(__import__("os").name == "nt", reason="shell-script engine stand-in")
def test_a_failed_comparison_is_a_warning_not_a_failed_run(tmp_path, monkeypatch, capsys):
    from woof import go_cli

    monkeypatch.setattr(rustwx_lanes, "require_compare_bin", lambda: _fake_engine(tmp_path))

    def fail(*a, **kw):
        raise go_cli.GoStageFailed(1, "reference not reachable")

    monkeypatch.setattr(go_cli, "_run_stage", fail)
    go_cli._reference_compare_stage(_go_plan(tmp_path, "hrrr"), explain=False)
    said = capsys.readouterr().err
    assert "hrrr comparison sheets were not drawn" in said
    assert "every picture of the run itself is in place" in said


def test_the_finalize_render_stage_runs_the_comparison_on_every_success_exit():
    import inspect

    from woof import go_cli

    source = inspect.getsource(go_cli._render_stage)
    assert source.count("_reference_compare_stage(") == 2, (
        "both of the render stage's success exits (drawn now, and every frame "
        "already drawn while the forecast ran) must draw the comparison")


def _frames(tmp_path, *domains, hours=6):
    names = [f"wrfout_{domain}_2026-10-03_{hour:02d}_00_00"
             for domain in domains for hour in range(hours)]
    return sorted(_touch(tmp_path / "wrfout", *names))


def test_frames_are_dealt_by_neighbouring_valid_times_every_nest_together(tmp_path):
    # WOOF nyc-500m 2026-10-31: 98 frames drawn one sheet at a time on one
    # core.  A valid time stays on one engine across the nests, so no two
    # engines ask for one reference lead at once.
    frames = _frames(tmp_path, "d01", "d02")
    dealt = render_compare.deal_frames(frames, 3)
    assert sorted(frame for members in dealt for frame in members) == frames
    assert [len(members) for members in dealt] == [4, 4, 4]
    seen = set()
    for members in dealt:
        valids = {render_compare.frame_identity(frame)[1] for frame in members}
        assert not valids & seen
        seen |= valids
        assert {render_compare.frame_identity(frame)[0] for frame in members} == {"d01", "d02"}
        hours = sorted(valid.hour for valid in valids)
        assert hours == list(range(hours[0], hours[-1] + 1))


def test_one_worker_and_a_single_time_keep_the_frames_whole(tmp_path):
    frames = _frames(tmp_path, "d01", "d02")
    assert render_compare.deal_frames(frames, 1) == [frames]
    only = _frames(tmp_path / "one", "d01", "d02", hours=1)
    assert render_compare.deal_frames(only, 8) == [only]
    assert render_compare.deal_frames([], 8) == []


def test_each_dealt_list_finds_the_hour_before_it_as_context(tmp_path):
    frames = _frames(tmp_path, "d01")
    first, second = render_compare.deal_frames(frames, 2)
    (members, context), = render_compare.group_frames(second)
    assert members == second
    assert context == [first[-1]]


def test_the_engine_count_follows_the_cores_and_the_host_memory(tmp_path):
    frames = _frames(tmp_path, "d01", hours=20)
    for frame in frames:
        frame.write_bytes(b"x" * 1000)
    plenty = 10 ** 12
    assert render_compare.compare_workers(frames, cores=16, available_bytes=plenty) == 8
    assert render_compare.compare_workers(frames, cores=192, available_bytes=plenty) == render_compare.COMPARE_WORKERS_MAX
    assert render_compare.compare_workers(frames, cores=4, available_bytes=plenty) == 2
    assert render_compare.compare_workers(frames, cores=1, available_bytes=plenty) == 1
    assert render_compare.compare_workers(frames[:3], cores=16, available_bytes=plenty) == 3
    # Eight bytes of host memory per byte of frame: 8,000 bytes an engine.
    assert render_compare.compare_workers(frames, cores=16, available_bytes=24_000) == 3
    assert render_compare.compare_workers(frames, cores=16, available_bytes=10) == 1


def test_an_engine_threads_share_the_cores_only_when_several_run():
    assert render_compare._engine_environment({"A": "1"}, 1, 16) == {"A": "1"}
    assert render_compare._engine_environment({}, 8, 16)["RAYON_NUM_THREADS"] == "2"
    assert render_compare._engine_environment({"RAYON_NUM_THREADS": "3"}, 8, 16)["RAYON_NUM_THREADS"] == "3"


@pytest.mark.skipif(__import__("os").name == "nt", reason="shell-script engine stand-in")
def test_every_frame_goes_to_exactly_one_engine_and_every_line_is_relayed(tmp_path, monkeypatch, capsys):
    # The door's own run: several engines at once, each told its own frames
    # and the hour before them as context, and every sheet said once.
    log = tmp_path / "calls.txt"
    engine = tmp_path / "rw_compare"
    engine.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = --list-products ]; then\n"
        "  printf 'REFERENCE\thrrr\tHRRR\n'; exit 0\n"
        "fi\n"
        f"echo \"$*\" >> {log}\n"
        "for last; do :; done\n"
        "printf 'RENDERED\tt2m\tx\t%s.png\n' \"$last\"\n")
    engine.chmod(0o755)
    monkeypatch.setattr(rustwx_lanes, "require_compare_bin", lambda: engine)
    from woof import render
    monkeypatch.setattr(render_compare, "compare_workers", lambda frames: 3)
    monkeypatch.setattr(render, "_claim_run_dir", lambda args, frames: None)
    monkeypatch.setattr(render, "_publish_run_dir", lambda args: None)
    frames = _frames(tmp_path, "d01", "d02")
    args = _args(wrfout=frames, out=tmp_path / "sheets", products="t2m")
    assert render_compare.compare_main(args) == 0
    calls = log.read_text().splitlines()
    assert len(calls) == 6, "three engines' lists, one engine per nest"
    drawn, contexts = [], []
    for call in calls:
        words = call.split()
        for at, word in enumerate(words):
            if "wrfout_" in word:
                (contexts if words[at - 1] == "--context" else drawn).append(word)
    assert sorted(drawn) == sorted(str(frame) for frame in frames)
    assert len(contexts) == 4, "two later lists, two nests: the hour before each list's first frame"
    said = capsys.readouterr().out
    assert said.count("render: ") >= 6
