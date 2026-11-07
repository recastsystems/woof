"""Prevent a named native configuration from silently selecting other physics."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from woof.model_presets import launch_blockers, namelist, preset

ROOT = Path(__file__).resolve().parents[1]
NAME = "rrfs-v2-hex-tempo"


def test_native_configuration_keeps_required_physics_and_clock():
    # These are the workflow's settings, not a WRF scheme-number translation.
    row = preset(NAME)
    output = namelist(row, start="2026-10-03_12:00:00", seconds=3600)
    for required in (
        "config_microp_scheme = 'mp_tempo'",
        "config_physics_suite = 'hrrrv5'",
        "config_convection_scheme = 'off'",
        "config_mynn_mixaerosols = 1",
        "config_tempo_ml_for_bl_nc = .true.",
        "config_tempo_aerosolaware = .true.",
        "config_tempo_hailaware = .true.",
        "config_radtlw_interval = '00:15:00'",
        "config_radtsw_interval = '00:15:00'",
        "config_dt = 20",
        "config_number_of_sub_steps = 4",
        "config_number_cam_damping_levels = 8",
        "config_run_duration = '000_01:00:00'",
        "num_soil_layers = 9",
    ):
        assert required in output
    assert "${" not in output
    assert row["vertical"]["layers"] == 59
    assert len(row["vertical"]["interfaces_m"]) == 60
    assert row["vertical"]["top_m"] == 25878.712
    assert isinstance(row["physics_pins"], list)
    assert {pin["path"].split("/")[-1] for pin in row["physics_pins"]} == {
        "MYNN-EDMF", "MYNN-SFC", "RUCLSM", "TEMPO", "UGWP"}


@pytest.mark.parametrize("seconds", [0, -20, 21, True, 20.0])
def test_incomplete_or_noninteger_step_is_not_exported(seconds):
    with pytest.raises(ValueError, match="positive integer multiple"):
        namelist(preset(NAME), start="2026-10-03_12:00:00", seconds=seconds)


def test_multiday_duration_and_calendar_injection():
    output = namelist(preset(NAME), start="2026-10-03_12:00:00", seconds=259220)
    assert "config_run_duration = '003_00:00:20'" in output
    with pytest.raises(ValueError, match="start must"):
        namelist(preset(NAME), start="2026-10-03_12:00:00'\n/\n&physics", seconds=20)


def _fake_hex(root: Path, version: str, *, runners: bool = True) -> Path:
    """An installed-looking woof hex: its metadata, and its two runners or none
    (woof hex 0.3.3 as published ships neither)."""
    root.mkdir(parents=True, exist_ok=True)
    info = root / f"gpuwm_hex-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: woof hex\nVersion: {version}\n", encoding="utf-8")
    package = root / "hexcore"
    (package / "drivers").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "drivers" / "__init__.py").write_text("", encoding="utf-8")
    if runners:
        (package / "preset_prepare.py").write_text("", encoding="utf-8")
        (package / "drivers" / "forecast_receipt.py").write_text("", encoding="utf-8")
    return root


@pytest.fixture
def hex_040(tmp_path_factory, monkeypatch):
    """woof hex 0.4.0 on the import path, for this process and its children."""
    import importlib
    site = _fake_hex(tmp_path_factory.mktemp("site") / "site", "0.4.0")
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [str(site), str(ROOT), os.environ.get("PYTHONPATH", "")]))
    for name in [n for n in sys.modules if n == "hexcore" or n.startswith("hexcore.")]:
        monkeypatch.delitem(sys.modules, name)
    importlib.invalidate_caches()
    yield site
    for name in [n for n in sys.modules if n == "hexcore" or n.startswith("hexcore.")]:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _cli(*args):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1")
    return subprocess.run([sys.executable, "-m", "woof.cli", "preset", *args],
                          cwd=ROOT, env=env, capture_output=True, text=True)


def _absent_hex(monkeypatch):
    from woof import model_presets
    monkeypatch.setattr(model_presets, "_distribution_version", lambda name: None)
    monkeypatch.setattr(model_presets, "_module_error",
                        lambda module: "it is not installed")


def test_check_and_list_count_a_missing_runner_package_as_a_blocker(monkeypatch, capsys):
    # The 2.9.0 door audit: with no woof hex installed, list and check said
    # launchable with 0 blockers and `preset prepare` then stopped with
    # "runner 'woof.hex.preset_prepare' is not installed".
    from woof import model_presets
    from woof.cli import build_parser
    _absent_hex(monkeypatch)
    parser = build_parser(preset_only=True)
    assert model_presets.main(parser.parse_args(["preset", "check", NAME])) == 3
    document = json.loads(capsys.readouterr().out)
    assert document["launchable"] is False
    assert {r["id"] for r in document["blockers"]} == {
        "prepare-runner-importable", "run-runner-importable"}
    assert all(r["remedy"] == 'pip install "recast-woof>=0.4.0,<0.5"'
               for r in document["blockers"])
    assert document["paired_package"]["installed"] is None
    assert model_presets.main(parser.parse_args(["preset", "list"])) == 0
    row = next(r for r in json.loads(capsys.readouterr().out)["presets"] if r["id"] == NAME)
    assert row["launchable"] is False


def test_run_refuses_a_missing_runner_even_when_experimental(tmp_path, monkeypatch, capsys):
    import subprocess as sp
    from woof import model_presets
    from woof.cli import build_parser
    _absent_hex(monkeypatch)
    monkeypatch.setattr(sp, "call", lambda argv: pytest.fail("launched without a runner"))
    common = ["preset", "run", NAME, "--mesh", "conus12km", "--grid", "g",
              "--static", "s", "--init", "i", "--lbc-dir", "l", "--invariant", "v",
              "--out", str(tmp_path / "output"), "--start-time", "2026-10-03_12:00:00",
              "--hours", "6", "--init-source", "public source", "--card", "RTX-PRO-6000",
              "--experimental"]
    assert model_presets.main(build_parser(preset_only=True).parse_args(common)) == 3
    assert "run-runner-importable" in capsys.readouterr().out


def test_an_installed_hex_outside_the_pair_is_a_blocker(tmp_path_factory):
    # woof hex 0.3.3 is the newest release on the index today; it ships neither
    # runner and declares gpuwm>=2.8.0,<2.8.1.
    site = _fake_hex(tmp_path_factory.mktemp("site") / "site", "0.3.3", runners=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1",
               PYTHONPATH=os.pathsep.join([str(site), str(ROOT)]))
    result = subprocess.run([sys.executable, "-m", "woof.cli", "preset", "check", NAME],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 3, result.stdout + result.stderr
    document = json.loads(result.stdout)
    assert document["paired_package"]["installed"] == "0.3.3"
    assert {r["id"] for r in document["blockers"]} == {
        "paired-package-release", "prepare-runner-importable", "run-runner-importable"}
    prepare = subprocess.run(
        [sys.executable, "-m", "woof.cli", "preset", "prepare", NAME, "--cycle",
         "2026101412", "--mesh", "conus12km", "--out", str(site / "out")],
        cwd=ROOT, env=env, capture_output=True, text=True)
    assert prepare.returncode == 2
    assert "woof hex 0.3.3 is installed" in prepare.stderr + prepare.stdout
    assert not (site / "out").exists()


def test_real_cli_check_reports_no_blocker_and_every_declared_divergence(tmp_path, hex_040):
    result = _cli("check", NAME)
    assert result.returncode == 0, result.stdout + result.stderr
    document = json.loads(result.stdout)
    assert document["launchable"] is True
    assert document["blockers"] == []
    assert "pinned-dycore" in {d["id"] for d in document["declared_divergences"]}
    assert {r["id"] for r in document["blockers"]} == {
        r["id"] for r in launch_blockers(preset(NAME))}
    assert "tempo-boundary-layer-cloud" not in {r["id"] for r in document["blockers"]}
    assert "mixed-aerosol-tendencies" not in {r["id"] for r in document["blockers"]}
    assert "matched-state-and-demo" not in {r["id"] for r in document["blockers"]}
    assert all(r["breakage"] and r["site"] for r in document["blockers"])
    assert list(tmp_path.iterdir()) == []


def test_real_cli_export_does_not_overwrite_configuration(tmp_path):
    path = tmp_path / "namelist.atmosphere"
    result = _cli("export", NAME, "--start", "2026-10-03_12:00:00",
                  "--seconds", "3600", "--out", str(path))
    assert result.returncode == 0, result.stdout + result.stderr
    content = path.read_bytes()
    result = _cli("export", NAME, "--start", "2026-10-03_12:00:00",
                  "--seconds", "20", "--out", str(path))
    assert result.returncode != 0
    assert path.read_bytes() == content


def test_real_cli_lists_the_configuration_with_its_divergence_status(hex_040):
    result = _cli("list")
    assert result.returncode == 0, result.stdout + result.stderr
    row = next(r for r in json.loads(result.stdout)["presets"] if r["id"] == NAME)
    assert row["status"] == "rrfs-configuration-with-declared-divergences"
    assert row["launchable"] is True


def test_unknown_identifier_is_not_a_default_fallback():
    with pytest.raises(ValueError, match="unknown model preset"):
        preset("unknown-mesh-suite")


def test_run_launches_the_table_command_without_the_experimental_flag(tmp_path, monkeypatch, hex_040):
    import subprocess as sp
    from woof import model_presets
    from woof.cli import build_parser
    seen = []
    monkeypatch.setattr(sp, "call", lambda argv: seen.append(argv) or 0)
    common = ["preset", "run", NAME, "--mesh", "conus12km", "--grid", "g",
              "--static", "s", "--init", "i", "--lbc-dir", "l", "--invariant", "v",
              "--out", str(tmp_path / "output"), "--start-time", "2026-10-03_12:00:00",
              "--hours", "6", "--init-source", "public source", "--card", "RTX-PRO-6000"]
    parser = build_parser(preset_only=True)
    assert model_presets.main(parser.parse_args(common)) == 0
    assert model_presets.main(parser.parse_args(common + ["--experimental"])) == 0
    assert len(seen) == 2 and seen[0] == seen[1]
    assert seen[0][seen[0].index("--physics-suite") + 1] == "rrfs_v2_tempo_candidate"
    assert not list(tmp_path.iterdir())


def test_candidate_command_uses_table_clock_and_the_existing_forecast_door(tmp_path, hex_040):
    from argparse import Namespace
    from woof.model_presets import launch_command
    args = Namespace(mesh="conus12km", card="RTX-PRO-6000", hours=6.,
        start_time="2026-10-03_12:00:00", init_source="public source",
        history_every_minutes=60, **{k: tmp_path / k for k in
        ("grid", "static", "init", "lbc_dir", "invariant", "out")})
    argv = launch_command(preset(NAME), args)
    assert argv[1:3] == ["-m", "woof.hex.drivers.forecast_receipt"]
    assert not any("tree/tools" in token for token in argv)
    assert argv[argv.index("--dt") + 1] == "60"
    assert argv[argv.index("--convection") + 1] == "ntiedtke"
    assert argv[argv.index("--physics-suite") + 1] == "rrfs_v2_tempo_candidate"
    assert argv[argv.index("--hours") + 1] == "6.0"
    assert "--measurement-mint" in argv
    assert argv[argv.index("--fingerprint-every") + 1] == "60"
    args.fingerprint_every = 3
    argv = launch_command(preset(NAME), args)
    assert argv[argv.index("--fingerprint-every") + 1] == "3"
    args.hours = float("nan")
    with pytest.raises(ValueError, match="positive whole"):
        launch_command(preset(NAME), args)


def test_minimal_parser_imports_no_data_or_numerical_backend():
    # Run in a fresh process so earlier tests cannot hide a parser import.
    code = (
        "import sys; from woof.cli import build_parser; "
        "p=build_parser(preset_only=True); "
        "assert p.parse_args(['preset','list']).preset_action == 'list'; "
        "assert 'netCDF4' not in sys.modules; "
        "assert 'cupy' not in sys.modules; "
        "assert 'woof.downscale' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_physics_without_a_port_is_a_declared_divergence_not_a_blocker(hex_040):
    # 2026-08-20: physics parity is retired and the hex dycore stays pinned to
    # MPAS-A v8.4.1, so each component that runs its own generation is named
    # with its pin, what runs instead and the follow-up instead of refusing.
    result = _cli("check", NAME)
    document = json.loads(result.stdout)
    blockers = {r["id"] for r in document["blockers"]}
    divergences = {d["id"]: d for d in document["declared_divergences"]}
    assert set(divergences) == {"pinned-dycore", "mynn-sfc-revision", "mynn-edmf-revision",
                                "ruc-mpas-code-line", "ugwp-orographic-drag"}
    assert not blockers & set(divergences)
    pins = {pin["sha"] for pin in preset(NAME)["physics_pins"]} | {preset(NAME)["mpas_commit"]}
    for divergence in divergences.values():
        assert any(sha in divergence["pinned"] for sha in pins), divergence["id"]
        assert divergence["runs"] and divergence["reason"] and divergence["follow_up"]


@pytest.mark.parametrize("damage,match", [
    (lambda d: d.pop("follow_up"), "names its component"),
    (lambda d: d.__setitem__("reason", ""), "names its component"),
    (lambda d: d.__setitem__("id", "mixed-aerosol-tendencies"), "collide"),
])
def test_a_malformed_divergence_refuses_the_registry(tmp_path, monkeypatch,
                                                     damage, match):
    import woof.model_presets as presets
    document = json.loads(presets.REGISTRY.read_text(encoding="utf-8"))
    damage(document["presets"][0]["declared_divergences"][0])
    broken = tmp_path / "registry.v1.json"
    broken.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(presets, "REGISTRY", broken)
    with pytest.raises(ValueError, match=match):
        presets.load_registry()
