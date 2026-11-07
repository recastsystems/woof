"""One run, one folder -- the level above the render layout.

The defect, reported by the collaborator who runs WOOF inside his own
pipeline: he runs the same configuration repeatedly and every run wrote
into the same directory as the one before it.  A second render dropped
its PNGs among the first's; worse, the run artifacts -- ``wrfout``,
``report.json``, ``progress.jsonl``, the stage receipts -- interleaved
in one tree, and the create-only refusals standing in front of THAT
turned "run it again" into "invent a directory name by hand, every
time".

The remedy is a run-stamped level at the case root, above everything a
run writes, and it EXTENDS the 2026-08-06 project ruling rather than
replacing any of it: ``<domain>/<product>/<valid-day>`` survives
underneath, in the same order, from the same module.

    <out>/<run-stamp>/<domain>/<product>/<valid-day>/<file>.png
    <out>/<run-stamp>/run/wrfout/..., report.json, receipts

These tests pin what a script may rely on:

* the stamp's grammar, exactly, and that it is a pure function of
  (launch instant, initialisation time) with a collision ordinal that
  cannot hand two runs one folder;
* that it is the DEFAULT at all three front doors -- ``go``, ``sim``,
  ``render`` -- with ``--run-stamp off`` as the documented workaround;
* that two back-to-back runs of one configuration land in two folders
  and the first run's files are still exactly where and what they were;
* that pointing a door at an existing run folder -- or at anything
  INSIDE one, which is the shape ``woof go`` hands its own render
  stage -- is honoured verbatim, with each door's own answer to what
  happens next: ``go``/``sim`` refuse (create-only stages), ``render``
  resumes;
* that the render layout underneath is unchanged.
"""

from __future__ import annotations

import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from woof import render_layout, run_stamp
from woof.cli import main as cli_main
from woof.io.wrfout import WrfoutWriter

_UTC = datetime.timezone.utc
_LAUNCH = datetime.datetime(2026, 8, 17, 4, 12, 33, tzinfo=_UTC)
_INIT = datetime.datetime(2026, 5, 17, 18, 0, tzinfo=_UTC)


@pytest.fixture(scope="module")
def wrf_package():
    """The science core, as a FIXTURE rather than a module-level skip.

    This was a bare ``pytest.importorskip("wrf")`` at module scope,
    written under the "`woof render`" banner two fifths of the way down
    the file -- and a module-level skip is not local to the section it
    sits under.  It fires at IMPORT, so on any box without the ``render``
    extra it took the 11 grammar tests defined ABOVE it with it: the
    stamp's spelling, its collision ordinal, the latest pointer, the
    workaround, the default.  Those need no renderer at all, and a
    stage-1 file that reports green while collecting nothing is worse
    than one that is absent.

    A fixture skips exactly its requesters.  Everything that drives the
    real ``woof render`` asks for this (through the ``wrfout`` fixture,
    which every such test already takes); the grammar, ``go``-plan,
    ``sim`` and ``--help`` tests do not, and now run on a bare install.
    """

    return pytest.importorskip(
        "wrf", reason="woof render requires the wrf package (wrf-rust)")


# ---------------------------------------------------------------------------
# 1. The grammar
# ---------------------------------------------------------------------------

def test_the_stamp_is_exactly_what_the_docs_promise():
    """launch, then init, both UTC -- the documented spelling."""

    assert run_stamp.format_stamp(launch=_LAUNCH, init=_INIT) == (
        "run-20260817-041233Z_i202605171800Z")


def test_a_door_that_cannot_read_an_init_time_omits_it_rather_than_guessing():
    """A wrong init time in a folder name is worse than a missing one."""

    assert run_stamp.format_stamp(launch=_LAUNCH) == "run-20260817-041233Z"
    assert run_stamp.format_stamp(launch=_LAUNCH, init=None) == (
        "run-20260817-041233Z")


def test_the_launch_field_leads_so_a_plain_listing_sorts_into_run_order():
    """Three runs, three cycles, listed by name = listed by launch."""

    names = [
        run_stamp.format_stamp(
            launch=_LAUNCH + datetime.timedelta(seconds=step),
            init=_INIT - datetime.timedelta(days=step))
        for step in (2, 0, 1)]
    assert sorted(names) == [names[1], names[2], names[0]]


@pytest.mark.parametrize("spelling,expected", [
    ("2026-05-17T18", _INIT),                     # a wizard config's cycle
    ("2026-05-17T18:00:00Z", _INIT),              # a proof document
    ("2026-05-17_18:00:00", _INIT),               # WRF's own stamp
    ("2026-05-17 18:00:00", _INIT),
    (_INIT, _INIT),
    (datetime.datetime(2026, 5, 17, 18, 0), _INIT),   # naive is READ as UTC
    ("", None),
    (None, None),
    ("not a time", None),
])
def test_every_init_spelling_this_tree_writes_is_read(spelling, expected):
    assert run_stamp.parse_time(spelling) == expected


@pytest.mark.parametrize("init", [_INIT, _INIT.replace(year=1)])
def test_a_stamp_round_trips_through_its_own_parser(init):
    name = run_stamp.format_stamp(launch=_LAUNCH, init=init)
    parsed = run_stamp.parse_stamp(name)
    assert parsed == {"launch": _LAUNCH, "init": init, "ordinal": 1}
    assert run_stamp.is_run_folder(name)
    assert run_stamp.is_run_folder(Path("/anywhere") / name)


@pytest.mark.parametrize("name", [
    "data", "prepared", "png", "run", "run-", "run-20260817",
    "run-20260817-041233", "runs-20260817-041233Z",
])
def test_a_name_that_is_not_a_stamp_is_not_a_run_folder(name):
    assert not run_stamp.is_run_folder(name)
    assert run_stamp.parse_stamp(name) is None


def test_two_runs_launched_in_one_second_get_two_folders(tmp_path,
                                                         monkeypatch):
    """The case the whole thing exists for, at its hardest.

    A check-then-create would hand both runs one directory and two
    runs' receipts in it.  The allocation is a create-exclusive mkdir,
    so the second run takes the next ordinal.
    """

    monkeypatch.setattr(run_stamp, "utcnow", lambda: _LAUNCH)
    first = run_stamp.allocate(tmp_path, init=_INIT)
    second = run_stamp.allocate(tmp_path, init=_INIT)
    assert first != second
    assert first.name == "run-20260817-041233Z_i202605171800Z"
    assert second.name == "run-20260817-041233Z_i202605171800Z-02"
    assert first.is_dir() and second.is_dir()
    assert run_stamp.parse_stamp(second.name)["ordinal"] == 2


def test_the_latest_pointer_names_the_newest_run(tmp_path, monkeypatch):
    """A script finds the run it just started without globbing."""

    monkeypatch.setattr(run_stamp, "utcnow", lambda: _LAUNCH)
    run_stamp.allocate(tmp_path, init=_INIT)
    monkeypatch.setattr(
        run_stamp, "utcnow",
        lambda: _LAUNCH + datetime.timedelta(minutes=5))
    newest = run_stamp.allocate(tmp_path, init=_INIT)
    pointer = tmp_path / run_stamp.LATEST_POINTER
    run_stamp.record_latest(tmp_path, newest)
    assert pointer.read_text(encoding="utf-8").strip() == newest.name
    assert run_stamp.latest(tmp_path) == newest
    # A pointer naming a folder that has been moved away must not lie:
    # the listing answers instead.
    shutil.rmtree(newest)
    assert run_stamp.latest(tmp_path).name == (
        "run-20260817-041233Z_i202605171800Z")


def test_the_workaround_returns_the_root_itself(tmp_path):
    assert run_stamp.resolve(tmp_path, init=_INIT, enabled=False) == tmp_path


def test_a_path_that_is_already_a_run_folder_is_honoured_verbatim(tmp_path):
    """No second stamp level inside a folder the caller named."""

    named = tmp_path / run_stamp.format_stamp(launch=_LAUNCH, init=_INIT)
    assert run_stamp.resolve(named, init=_INIT) == named
    assert list(named.iterdir()) == []


def test_a_path_under_a_run_folder_belongs_to_that_run(tmp_path):
    """The run's identity is established at or ABOVE the path.

    ``woof go`` claims ``<case>/run-.../`` and then hands its render
    stage ``<case>/run-.../png``.  A door that asked only "is my own
    name a stamp?" answered no there and allocated a SECOND stamp
    inside the first -- ``<case>/run-.../png/run-.../`` -- so the early
    render of the first committed frame and the finalize render of the
    same run filed into two different run folders.
    """

    named = tmp_path / run_stamp.format_stamp(launch=_LAUNCH, init=_INIT)
    inside = named / "png"
    assert run_stamp.owning_run(inside) == named
    assert run_stamp.owning_run(named) == named
    assert run_stamp.owning_run(tmp_path / "png") is None
    assert run_stamp.resolve(inside, init=_INIT) == inside
    assert run_stamp.owning_run(named / "png" / "d02-3km") == named


def test_a_stage_inside_a_claimed_run_declines_to_stamp_again():
    """One run, one folder -- said on the stage's own command line."""

    from woof.go_cli import render_command

    command = render_command(
        {"run": Path("case") / "run", "render": Path("case") / "png",
         "render_products": "refl"})
    assert run_stamp.stage_flags() == ["--run-stamp", "off"]
    assert command[command.index("--run-stamp") + 1] == "off"


def test_the_default_is_on(tmp_path):
    """Fixed means default: a bare call stamps."""

    assert run_stamp.DEFAULT_RUN_STAMP is True
    resolved = run_stamp.resolve(tmp_path, init=_INIT)
    assert resolved.parent == tmp_path
    assert run_stamp.is_run_folder(resolved)

    class _Bare:
        pass

    assert run_stamp.run_stamp_enabled(_Bare()) is True


# ---------------------------------------------------------------------------
# 2. `woof render` -- the door the report was written about
# ---------------------------------------------------------------------------

_NZ, _NY, _NX = 4, 12, 16
_STAMPS = ("2026-05-17_18:00:00", "2026-05-17_19:00:00")


def _frame(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    lat = np.tile(np.linspace(38.0, 40.0, _NY)[:, None], (1, _NX))
    lon = np.tile(np.linspace(-98.0, -95.0, _NX)[None, :], (_NY, 1))
    return {
        "T": np.zeros((_NZ, _NY, _NX), np.float32),
        "MU": np.zeros((_NY, _NX), np.float32),
        "REFL_10CM": rng.uniform(-20.0, 65.0,
                                 (_NZ, _NY, _NX)).astype(np.float32),
        "T2": rng.uniform(280.0, 300.0, (_NY, _NX)).astype(np.float32),
        "U10": rng.uniform(-10.0, 10.0, (_NY, _NX)).astype(np.float32),
        "V10": rng.uniform(-10.0, 10.0, (_NY, _NX)).astype(np.float32),
        "RAINC": rng.uniform(0.0, 5.0, (_NY, _NX)).astype(np.float32),
        "RAINNC": rng.uniform(0.0, 30.0, (_NY, _NX)).astype(np.float32),
        "OLR": rng.uniform(90.0, 320.0, (_NY, _NX)).astype(np.float32),
        "XLAT": lat.astype(np.float32),
        "XLONG": lon.astype(np.float32),
        "HGT": np.zeros((_NY, _NX), np.float32),
        "SINALPHA": np.zeros((_NY, _NX), np.float32),
        "COSALPHA": np.ones((_NY, _NX), np.float32),
    }


@pytest.fixture(scope="module")
def wrfout(wrf_package, tmp_path_factory) -> Path:
    """One real wrfout, written by this tree's own writer.

    Takes ``wrf_package`` so every test that renders picks the science
    core up through the file it already asks for, and no renderer test
    can be added that forgets to.
    """

    root = tmp_path_factory.mktemp("run-stamp-wrfout")
    path = root / "wrfout_d02_2026-05-17_18-00-00.nc"
    with WrfoutWriter(
            path, nx=_NX, ny=_NY, nz=_NZ, dx=3000.0, dy=3000.0,
            global_attrs={
                "GRID_ID": 2,
                "SIMULATION_START_DATE": "2026-05-17_18:00:00"}) as writer:
        for index, stamp in enumerate(_STAMPS):
            writer.write_frame(stamp, _frame(seed=11 + index))
    return path


def _pngs(root: Path) -> list[Path]:
    return render_layout.iter_rendered(root)


def _run_folders(root: Path) -> list[Path]:
    return sorted((child for child in root.iterdir()
                   if child.is_dir() and run_stamp.is_run_folder(child)),
                  key=lambda child: child.name)


def _render(wrfout: Path, out: Path, *extra: str) -> int:
    return cli_main(["render", str(wrfout), "--out", str(out),
                     "--products", "refl", *extra])


def test_two_back_to_back_renders_land_in_two_run_folders(wrfout, tmp_path,
                                                          capsys):
    """The report, made into a gate, through the real front door."""

    out = tmp_path / "png"
    assert _render(wrfout, out) == 0
    first = _run_folders(out)
    assert len(first) == 1, "the first render wrote no run-stamped folder"
    before = {path.relative_to(first[0]): path.read_bytes()
              for path in _pngs(first[0])}
    assert before, "the first render drew nothing"

    assert _render(wrfout, out) == 0
    folders = _run_folders(out)
    assert len(folders) == 2, (
        "a second render of the same configuration did not get its own "
        f"folder: {[f.name for f in folders]}")
    # The first run is untouched -- not merely present, byte for byte
    # what it was.
    assert {path.relative_to(first[0]): path.read_bytes()
            for path in _pngs(first[0])} == before
    second = [f for f in folders if f != first[0]][0]
    assert {p.relative_to(second) for p in _pngs(second)} == set(before)


def test_the_render_layout_survives_underneath_unchanged(wrfout, tmp_path):
    """The 2026-08-06 project ruling is extended, never inverted."""

    out = tmp_path / "png"
    assert _render(wrfout, out) == 0
    run_dir = _run_folders(out)[0]
    drawn = _pngs(run_dir)
    assert drawn
    for png in drawn:
        parts = png.relative_to(run_dir).parts
        assert len(parts) == 4, (
            f"{png} is not <domain>/<product>/<valid-day>/<file>.png "
            "under the run folder")
        domain, product, day, _name = parts
        assert domain == "d02-3km"
        assert product
        assert day == "2026-05-17"


def test_the_run_folder_carries_the_models_own_init_time(wrfout, tmp_path):
    """Read from SIMULATION_START_DATE, not from the filename."""

    out = tmp_path / "png"
    assert _render(wrfout, out) == 0
    parsed = run_stamp.parse_stamp(_run_folders(out)[0].name)
    assert parsed["init"] == datetime.datetime(2026, 5, 17, 18, tzinfo=_UTC)


def test_the_run_folder_is_printed_before_anything_is_drawn(wrfout, tmp_path,
                                                            capsys):
    """A script computes the path it will watch instead of globbing."""

    out = tmp_path / "png"
    assert _render(wrfout, out) == 0
    printed = capsys.readouterr().out
    run_dir = _run_folders(out)[0]
    layout_line = next(line for line in printed.splitlines()
                       if line.startswith("render: layout"))
    assert run_dir.name in layout_line, (
        f"the layout line does not name the run folder: {layout_line!r}")


def test_run_stamp_off_is_the_documented_workaround(wrfout, tmp_path):
    """Off restores the unstamped tree, and says nothing else changed."""

    out = tmp_path / "png"
    assert _render(wrfout, out, "--run-stamp", "off") == 0
    assert _run_folders(out) == []
    drawn = _pngs(out)
    assert drawn
    for png in drawn:
        # <domain>/<product>/<valid-day>/<file>.png, straight under
        # --out: the pre-2.5.0 tree, with the layout ruling intact.
        assert len(png.relative_to(out).parts) == 4


def test_pointing_render_at_an_existing_run_folder_resumes_into_it(wrfout,
                                                                   tmp_path):
    """The caller named the run folder, so that IS the run folder."""

    out = tmp_path / "png"
    assert _render(wrfout, out) == 0
    run_dir = _run_folders(out)[0]
    first = {p.relative_to(run_dir) for p in _pngs(run_dir)}

    assert cli_main(["render", str(wrfout), "--out", str(run_dir),
                     "--products", "t2"]) == 0
    assert _run_folders(run_dir) == [], (
        "a second stamp level was inserted inside a folder the caller "
        "named, burying their own path")
    after = {p.relative_to(run_dir) for p in _pngs(run_dir)}
    assert first < after, "the resumed render drew nothing new"


def test_rendering_into_a_run_folders_own_subdirectory_stamps_nothing(
        wrfout, tmp_path):
    """``<case>/run-.../png`` is the run's tree, not a fresh case root.

    This is the shape ``woof go`` hands its render stage.  A second
    stamp level here reads ``run-.../png/run-.../`` on disk, and the two
    renders one run makes -- early on the first committed frame, and
    finalize -- land in two of them.
    """

    run_dir = tmp_path / run_stamp.format_stamp(launch=_LAUNCH, init=_INIT)
    out = run_dir / "png"
    assert _render(wrfout, out) == 0
    assert _run_folders(out) == [], (
        "a render inside an already-claimed run folder stamped again")
    drawn = _pngs(out)
    assert drawn
    for png in drawn:
        assert len(png.relative_to(out).parts) == 4, (
            f"{png} is not <domain>/<product>/<valid-day>/<file>.png")


# ---------------------------------------------------------------------------
# 3. `woof go` -- the whole chain's artifacts, not just its pictures
# ---------------------------------------------------------------------------

PROFILE = "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"


def _config(tmp_path: Path, name: str = "case") -> Path:
    out = tmp_path / f"{name}.toml"
    assert cli_main(["domain", "--point=35.3,-97.5", "--card", "24gb",
                     "--ladder", "12", "--source", "gfs",
                     "--cycle", "2026-07-29T18", "--hours", "6",
                     "--out", str(out),
                     "--physics-profile", PROFILE]) == 0
    return out


def test_go_puts_every_run_artifact_under_one_stamped_folder(tmp_path):
    """authority, prepared, run and png -- all four, one level down."""

    from woof import go_cli

    config = _config(tmp_path)
    case = tmp_path / "out"
    plan = go_cli.plan_from_config(config, outdir=case, claim=True)
    assert run_stamp.is_run_folder(plan["root"]), (
        f"{plan['root']} is not a run-stamped folder")
    assert plan["root"].parent == case
    assert plan["case_root"] == case
    for key in ("authority", "prepared", "run", "render"):
        assert plan[key].parent == plan["root"], (
            f"plan[{key!r}] is not under the run folder")


def test_the_download_is_cached_across_runs_not_stamped(tmp_path):
    """Artifacts are stamped; INPUTS are reused.

    Stamping the download would re-fetch gigabytes of identical GRIB on
    every re-run, which is the opposite of the complaint.
    """

    from woof import go_cli

    config = _config(tmp_path)
    case = tmp_path / "out"
    first = go_cli.plan_from_config(config, outdir=case, claim=True)
    second = go_cli.plan_from_config(config, outdir=case, claim=True)
    assert first["data"] == second["data"]
    assert first["data"].parent == case / "downloads"
    assert not first["data"].is_relative_to(first["root"])
    assert not second["data"].is_relative_to(second["root"])
    assert first["root"] != second["root"]
    # Another request belongs in another cache, even within the same case.
    config.write_text(config.read_text().replace('2026-07-29T18', '2026-07-29T12'))
    changed = go_cli.plan_from_config(config, outdir=case, claim=True)
    assert changed["data"] != first["data"]
    assert changed["data"].parent == first["data"].parent


def test_two_go_runs_never_share_a_tree_even_in_one_second(tmp_path,
                                                           monkeypatch):
    """The claim is create-exclusive, which is what separates them.

    Both chains PREDICT the same stamp at this frozen instant; only the
    allocation failing on an existing directory keeps them apart.
    """

    from woof import go_cli

    monkeypatch.setattr(run_stamp, "utcnow", lambda: _LAUNCH)
    config = _config(tmp_path)
    case = tmp_path / "out"
    first = go_cli.claim_run_root(
        go_cli.plan_from_config(config, outdir=case))
    second = go_cli.claim_run_root(
        go_cli.plan_from_config(config, outdir=case))
    assert first["root"] != second["root"]
    assert first["run"] != second["run"]
    assert first["render"] != second["render"]
    assert first["authority"] != second["authority"]


def test_a_go_plan_names_the_run_folder_without_making_it(tmp_path):
    """The gates refuse before anything is spent, so a plan spends none.

    ``woof go``'s memory and geography gates leave the disk as they
    found it; a plan that created the tree on the way to being refused
    would break that, and a test already stands on it.
    """

    from woof import go_cli

    config = _config(tmp_path)
    case = tmp_path / "out"
    plan = go_cli.plan_from_config(config, outdir=case)
    assert run_stamp.is_run_folder(plan["root"])
    assert not case.exists(), "planning created the case directory"


def test_the_go_stamp_carries_the_configs_own_cycle(tmp_path):
    """The init half comes from [fetch] cycle, not from the clock."""

    from woof import go_cli

    plan = go_cli.plan_from_config(_config(tmp_path), outdir=tmp_path / "out")
    parsed = run_stamp.parse_stamp(plan["root"].name)
    assert parsed["init"] == datetime.datetime(2026, 7, 29, 18, tzinfo=_UTC)


def test_go_run_stamp_off_is_the_pre_2_5_0_tree(tmp_path):
    """The artifact tree is unstamped; input request isolation still applies."""
    from woof import go_cli

    config = _config(tmp_path)
    case = tmp_path / "out"
    plan = go_cli.plan_from_config(config, outdir=case, run_stamp=False)
    assert plan["root"] == case
    assert plan["authority"] == case / "authority"
    assert plan["render"] == case / "png"
    assert plan["data"].parent == case / "downloads"
    stamped = go_cli.plan_from_config(config, outdir=case, run_stamp=True)
    assert plan["data"] == stamped["data"]


def test_go_refuses_to_run_a_second_time_into_one_run_folder(tmp_path,
                                                             capsys,
                                                             monkeypatch):
    """The create-only refusal, at the one path that can still reach it.

    Pointing --outdir at an existing run folder is honoured verbatim,
    so a second chain into it would merge two runs' stages and publish
    receipts describing neither.  That refusal must survive the
    stamping, and it must name the run folder the caller typed.
    """
    from woof import capabilities

    installed = capabilities.is_installed
    monkeypatch.setattr(capabilities, "is_installed",
                        lambda module: module == "cupy" or installed(module))

    config = _config(tmp_path)
    named = tmp_path / run_stamp.format_stamp(launch=_LAUNCH, init=_INIT)
    (named / "authority").mkdir(parents=True)
    assert cli_main(["go", str(config), "--outdir", str(named),
                     "--dry-run"]) == 0     # a dry run touches nothing
    assert cli_main(["go", str(config), "--outdir", str(named)]) == 2
    message = capsys.readouterr().err
    assert str(named / "authority") in message
    assert "create-only" in message


def test_go_dry_run_names_the_folder_it_would_write(tmp_path, capsys):
    config = _config(tmp_path)
    assert cli_main(["go", str(config), "--outdir", str(tmp_path / "out"),
                     "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert run_stamp.RUN_PREFIX in printed, (
        "a dry run does not show the run folder the real run would use")


# ---------------------------------------------------------------------------
# 4. `woof sim` -- the standalone forecast stage
# ---------------------------------------------------------------------------

def _bundle(root: Path, *, source: str = "gfs") -> Path:
    from woof.prepared_single_domain_forecast import _PROOF_SCHEMA

    root.mkdir(parents=True, exist_ok=True)
    (root / "proof.json").write_text(json.dumps({
        "schema": _PROOF_SCHEMA[source],
        "status": "READY",
        "initial_valid_time": "2026-05-17T18:00:00",
        "input_manifest_sha256": "11" * 32,
        "prepared_cache": {"content_sha256": "22" * 32},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def _sim_outdir(printed: str) -> Path:
    """The ``--outdir`` value out of a printed runner command."""

    from host_shell_words import host_shell_words

    fields = host_shell_words(printed.strip().splitlines()[-1])
    return Path(fields[fields.index("--outdir") + 1])


def test_sim_stamps_its_own_outdir_and_prints_the_command_that_proves_it(
        tmp_path, capsys):
    """``--print-command`` is the seam's published boundary.

    It must print the outdir the stage would ACTUALLY use, or a caller
    driving the runner themselves lands somewhere else than a caller
    who did not.
    """

    prepared = _bundle(tmp_path / "prepared")
    config = tmp_path / "experiment.toml"
    config.write_text("[experiment]\nname = 'stamp'\n", encoding="utf-8")
    wps = tmp_path / "namelist.wps"
    wps.write_text("&share\n/\n", encoding="utf-8")
    out = tmp_path / "runs"

    argv = ["sim", str(prepared), "--experiment-config", str(config),
            "--wps-namelist", str(wps), "--outdir", str(out),
            "--print-command"]
    assert cli_main(argv) == 0
    named = _sim_outdir(capsys.readouterr().out)
    assert named.parent == out
    assert run_stamp.is_run_folder(named)
    assert run_stamp.parse_stamp(named.name)["init"] == (
        datetime.datetime(2026, 5, 17, 18, tzinfo=_UTC))
    # Named, not made: asking the question spends nothing, which is
    # this flag's own standing contract.
    assert not out.exists()


def test_two_sim_runs_never_share_an_outdir_even_in_one_second(tmp_path,
                                                               monkeypatch):
    """The claim is create-exclusive at the moment it happens."""

    from woof import stage_cli

    monkeypatch.setattr(run_stamp, "utcnow", lambda: _LAUNCH)
    prepared = _bundle(tmp_path / "prepared")
    bundle = stage_cli.resolve_bundle(prepared)

    class _Args:
        outdir = tmp_path / "runs"
        run_stamp = "on"

    first = stage_cli.claim_run_dir(_Args(), bundle)
    second = stage_cli.claim_run_dir(_Args(), bundle)
    assert first != second
    assert first.is_dir() and second.is_dir()


def test_sim_run_stamp_off_is_the_outdir_verbatim(tmp_path, capsys):
    prepared = _bundle(tmp_path / "prepared")
    config = tmp_path / "experiment.toml"
    config.write_text("[experiment]\nname = 'stamp'\n", encoding="utf-8")
    wps = tmp_path / "namelist.wps"
    wps.write_text("&share\n/\n", encoding="utf-8")
    out = tmp_path / "runs"
    assert cli_main(["sim", str(prepared), "--experiment-config", str(config),
                     "--wps-namelist", str(wps), "--outdir", str(out),
                     "--run-stamp", "off", "--print-command"]) == 0
    assert _sim_outdir(capsys.readouterr().out) == out


def test_sim_refuses_a_second_run_into_one_run_folder(tmp_path, capsys):
    """Create-only, and said at the door rather than four stages deep."""

    prepared = _bundle(tmp_path / "prepared")
    config = tmp_path / "experiment.toml"
    config.write_text("[experiment]\nname = 'stamp'\n", encoding="utf-8")
    wps = tmp_path / "namelist.wps"
    wps.write_text("&share\n/\n", encoding="utf-8")
    named = tmp_path / run_stamp.format_stamp(launch=_LAUNCH, init=_INIT)
    (named / "wrfout").mkdir(parents=True)
    (named / "report.json").write_text("{}", encoding="utf-8")

    assert cli_main(["sim", str(prepared), "--experiment-config", str(config),
                     "--wps-namelist", str(wps), "--outdir", str(named),
                     "--print-command"]) == 2
    message = capsys.readouterr().err
    assert str(named) in message
    assert "report.json" in message


# ---------------------------------------------------------------------------
# 5. The doors declare it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("command", ["go", "sim", "render"])
def test_every_door_that_writes_a_run_declares_run_stamp(command):
    """A capability with no front door is not a feature."""

    result = subprocess.run(
        [sys.executable, "-m", "woof.cli", command, "--help"],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "--run-stamp" in result.stdout, (
        f"`woof {command} --help` does not mention --run-stamp")
