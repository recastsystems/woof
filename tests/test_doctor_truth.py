"""Three ways ``woof doctor`` told a reader something untrue.

Each test here is anchored to a measured incident, not to a shape:

* A staged ``~/.woof/bridges`` left over from an older release passed
  every probe doctor had -- the launch probe and the ABI marker both
  answer yes to bytes that predate the release -- so doctor printed a
  green estate and the first ``woof prep`` refused, rc 78.  Doctor
  must check what the routes RESOLVE against the bytes this release
  pinned, which is exactly what ``woof fetch-bridges`` checks.
* ``python -m woof.doctor`` exited 0 having printed nothing: a module
  with no ``__main__`` guard is a door that silently answers "no gaps"
  to a reader diagnosing a broken install.
* The report arrived in one block after seconds of silence, with no
  statement of what was being probed while the reader waited.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys

import pytest

import woof.cli as cli
import woof.doctor as doctor
from woof import bridge_assets, bridges


# ---------------------------------------------------------------------------
# A staged estate that does not match this release's pins (#212)
# ---------------------------------------------------------------------------

def _pinned_estate(monkeypatch, tmp_path, *, contents):
    """A wheel-shaped install whose staged dir holds ``contents``.

    ``contents`` maps a bundled artifact's logical name to the bytes to
    stage for it (or ``None`` to leave it absent).  The returned pins
    document pins EVERY named artifact to the bytes in ``contents``, so
    a caller perturbs one file afterwards to make it stale.
    """

    from test_doctor import _wheel_shaped_install

    staged = _wheel_shaped_install(monkeypatch, tmp_path)
    platform = bridge_assets.host_platform() or "linux-x86_64"
    by_name = {artifact.name: artifact
               for artifact in bridge_assets.BUNDLED_ARTIFACTS}
    binaries = []
    for name, payload in contents.items():
        artifact = by_name[name]
        filename = bridge_assets.artifact_filename(artifact, platform)
        if payload is not None:
            (staged / filename).write_bytes(payload)
        binaries.append(bridge_assets.BinaryPin(
            artifact=name, filename=filename,
            bytes=len(payload if payload is not None else b"pinned"),
            sha256=hashlib.sha256(
                payload if payload is not None else b"pinned").hexdigest()))
    bundle = bridge_assets.BundlePin(
        platform=platform, filename=f"gpuwm-bridges-{platform}.zip",
        bytes=1, sha256="0" * 64, binaries=tuple(binaries))
    pins = bridge_assets.BridgePins(release="v9.9.9",
                                    platforms={platform: bundle})
    monkeypatch.setattr(bridge_assets, "load_pins", lambda path=None: pins)
    monkeypatch.setattr(bridge_assets, "host_platform", lambda: platform)
    return staged


def _estate_check(monkeypatch, tmp_path, *, contents):
    staged = _pinned_estate(monkeypatch, tmp_path, contents=contents)
    return staged, doctor._staged_estate_check()


def test_a_stale_staged_estate_is_a_gap_naming_fetch_bridges(monkeypatch,
                                                             tmp_path):
    """THE #212 defect: older bytes, present and launchable, reported ok.

    test host 1 carried an Aug-13 bridge set under a 2.5.0 wheel.  Every
    per-artifact probe passed -- the files exist, they launch, and the
    ABI marker only moves when the contract version bumps, which it had
    not.  ``woof doctor`` said no gaps, rc 0; the first ``woof prep``
    refused rc 78.  A pin mismatch in the directory ``woof
    fetch-bridges`` owns is a gap, and the remedy is that one command.
    """

    staged, check = _estate_check(
        monkeypatch, tmp_path,
        contents={"grib1_bridge": b"release bytes",
                  "gfs_grib2_bridge": b"release bytes"})
    assert check.status == "verified", check.detail

    # Now the estate goes stale exactly as test host 1's had: same filename,
    # older bytes.
    platform = bridge_assets.host_platform()
    by_name = {a.name: a for a in bridge_assets.BUNDLED_ARTIFACTS}
    filename = bridge_assets.artifact_filename(
        by_name["gfs_grib2_bridge"], platform)
    (staged / filename).write_bytes(b"bytes from an older release")

    check = doctor._staged_estate_check()
    assert check.status == "missing", check.detail
    assert check.blocking is True
    assert filename in check.detail
    assert "woof fetch-bridges" in check.remedy
    assert check.action == "woof fetch-bridges"


def test_the_stale_estate_reaches_the_report_and_the_exit_code(monkeypatch,
                                                               tmp_path):
    """The finding is in ``collect_checks`` and blocks, not just in a
    helper nobody calls."""

    staged = _pinned_estate(
        monkeypatch, tmp_path,
        contents={"grib1_bridge": b"release bytes"})
    platform = bridge_assets.host_platform()
    by_name = {a.name: a for a in bridge_assets.BUNDLED_ARTIFACTS}
    filename = bridge_assets.artifact_filename(
        by_name["grib1_bridge"], platform)
    (staged / filename).write_bytes(b"older")

    checks = doctor.collect_checks(sources=())
    estate = [c for c in checks if c.name == doctor._STAGED_ESTATE_NAME]
    assert len(estate) == 1, [c.name for c in checks]
    assert estate[0] in doctor.blocking_gaps(checks)


def test_an_absent_pinned_artifact_is_not_double_blamed(monkeypatch,
                                                        tmp_path):
    """Absent is somebody else's line.

    Every bundled artifact already has a check that names it when it is
    not staged.  This one is about bytes that ARE there and are the
    wrong ones; reporting the absent ones as broken here would print
    the same gap twice and give the reader two counts of one estate.
    """

    _, check = _estate_check(
        monkeypatch, tmp_path,
        contents={"grib1_bridge": None, "gfs_grib2_bridge": None})
    assert check.status != "missing", check.detail
    assert check.blocking is False


def test_an_unpinned_build_says_so_instead_of_alarming(monkeypatch,
                                                       tmp_path):
    """A source checkout carries no pins, and that is not a gap.

    ``woof/data/bridges/bridge-pins.json`` declares no release until a
    cut stamps it, so this check must report that it cannot compare
    rather than inventing a finding on every developer's box.
    """

    from test_doctor import _wheel_shaped_install

    _wheel_shaped_install(monkeypatch, tmp_path)
    monkeypatch.setattr(
        bridge_assets, "load_pins",
        lambda path=None: bridge_assets.BridgePins(release=None,
                                                   platforms={}))
    check = doctor._staged_estate_check()
    assert check.status == "info"
    assert check.blocking is False
    assert "pins" in check.detail


def test_bytes_built_from_source_are_not_called_stale(monkeypatch, tmp_path):
    """A cargo build never matches a release pin, and must not block.

    The remedy for a pin mismatch is ``woof fetch-bridges``, which
    writes into ``~/.woof/bridges`` and nowhere else.  Offering it for
    a binary resolved out of a checkout's ``target/release`` would
    replace nothing and name a breakage that is not there.
    """

    from test_doctor import _wheel_shaped_install

    _wheel_shaped_install(monkeypatch, tmp_path)
    platform = bridge_assets.host_platform() or "linux-x86_64"
    by_name = {a.name: a for a in bridge_assets.BUNDLED_ARTIFACTS}
    artifact = by_name["grib1_bridge"]
    filename = bridge_assets.artifact_filename(artifact, platform)
    built = tmp_path / "checkout-build"
    built.mkdir()
    (built / filename).write_bytes(b"cargo built these")
    monkeypatch.setenv(artifact.env_var, str(built / filename))

    bundle = bridge_assets.BundlePin(
        platform=platform, filename="bundle.zip", bytes=1, sha256="0" * 64,
        binaries=(bridge_assets.BinaryPin(
            artifact="grib1_bridge", filename=filename, bytes=7,
            sha256=hashlib.sha256(b"release").hexdigest()),))
    monkeypatch.setattr(
        bridge_assets, "load_pins",
        lambda path=None: bridge_assets.BridgePins(
            release="v9.9.9", platforms={platform: bundle}))
    monkeypatch.setattr(bridge_assets, "host_platform", lambda: platform)

    check = doctor._staged_estate_check()
    assert check.blocking is False, check.detail
    assert artifact.env_var in check.detail or "outside" in check.detail


# ---------------------------------------------------------------------------
# Map assets: where the renderer draws from, not where fetch-bridges writes
# ---------------------------------------------------------------------------

def _maps_estate(monkeypatch, tmp_path, *, companion):
    """A current staged estate whose pinned map assets were never staged.

    The state every README install was in: the platform wheel's binaries
    match this release, ``fetch-bridges`` never ran, and the renderer in
    ``libexec`` has no map assets above it.  ``companion`` says whether
    the ``recast-woof-data`` companion carries them (it does from 2.8.0 on).
    """

    from woof import rustwx

    staged = _pinned_estate(monkeypatch, tmp_path,
                            contents={"grib1_bridge": b"release bytes"})
    platform = bridge_assets.host_platform()
    pins = bridge_assets.load_pins()
    bundle = pins.platforms[platform]
    asset = bridge_assets.AssetPin(
        path="assets/basemap/natural_earth_10m/ne_10m_coastline.shp",
        bytes=3, sha256=hashlib.sha256(b"shp").hexdigest())
    bundle = bridge_assets.BundlePin(
        platform=bundle.platform, filename=bundle.filename,
        bytes=bundle.bytes, sha256=bundle.sha256,
        binaries=bundle.binaries, assets=(asset,))
    pins = bridge_assets.BridgePins(release=pins.release,
                                    platforms={platform: bundle})
    monkeypatch.setattr(bridge_assets, "load_pins", lambda path=None: pins)
    for name in ("RUSTWX_BASEMAP_DIR", "RUSTWX_ASSETS_DIR"):
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(home)
    renderer = (tmp_path / "site-packages" / "woof" / "libexec" / "bridges"
                / "rw_wrfbatch")
    renderer.parent.mkdir(parents=True)
    renderer.write_bytes(b"renderer")
    monkeypatch.setattr(rustwx, "find_renderer", lambda: renderer)
    monkeypatch.setattr(rustwx, "probe_renderer",
                        lambda path: (True, "probe ok"))
    monkeypatch.setattr(rustwx, "basemap_dir",
                        lambda: tmp_path / "no-checkout-assets")
    monkeypatch.setattr(rustwx, "default_bridge_dir", lambda: staged)
    maps = tmp_path / "woof_data" / "data" / "basemap"
    maps.mkdir(parents=True)
    monkeypatch.setattr(rustwx, "companion_basemap_dir",
                        (lambda: maps) if companion else (lambda: None))
    return maps


def test_unstaged_map_assets_are_no_gap_when_the_companion_draws_them(
        monkeypatch, tmp_path):
    """Every bare pip install printed MISSING for 42 map asset files and
    the remedy `woof fetch-bridges`.  The renderer draws from the
    recast-woof-data companion now, so the copy fetch-bridges would stage is
    not a gap, and saying it was sent readers to a step that changes no
    picture."""

    maps = _maps_estate(monkeypatch, tmp_path, companion=True)
    check = doctor._staged_estate_check()
    assert check.status == "verified", check.detail
    assert "map asset" not in check.detail

    renderer = doctor._rust_renderer_check()
    assert renderer.status == "verified", renderer.detail
    assert str(maps) in renderer.detail


def test_a_renderer_with_no_map_assets_is_a_gap_with_its_fix(monkeypatch,
                                                             tmp_path):
    """Without the companion's copy the staged one is what the renderer
    would read, so its absence is still reported; and the renderer line
    is a gap naming the pip line, never a verified line with a note."""

    _maps_estate(monkeypatch, tmp_path, companion=False)
    check = doctor._staged_estate_check()
    assert check.status == "missing", check.detail
    assert "map asset" in check.detail
    assert check.blocking is False

    renderer = doctor._rust_renderer_check()
    assert renderer.status == "missing", renderer.detail
    assert renderer.blocking is False
    assert "no coastlines, borders or state lines" in renderer.detail
    assert renderer.action.startswith("pip install --force-reinstall "
                                      "recast-woof-data")
    assert renderer.action in renderer.remedy


# ---------------------------------------------------------------------------
# The module door (#185)
# ---------------------------------------------------------------------------

def test_the_module_door_takes_the_console_scripts_arguments(monkeypatch):
    """One registrar, so the two doors cannot take different flags.

    ``--explain`` is the flag this report tells readers to use and it is
    registered by ``woof.cli`` on every subcommand rather than by
    ``doctor.register_cli``; a module door that built its own parser
    from the subcommand registrar alone would reject it.
    """

    seen = {}

    def fake_doctor_main(args):
        seen["args"] = args
        return 1

    monkeypatch.setattr(doctor, "doctor_main", fake_doctor_main)
    assert doctor.main(["--explain", "--json", "--source", "gfs"]) == 1
    args = seen["args"]
    assert args.explain is True
    assert args.json is True
    assert args.source == ["gfs"]

    seen.clear()
    assert doctor.main([]) == 1
    assert seen["args"].explain is False
    assert seen["args"].json is False
    assert not seen["args"].source


def test_the_module_door_does_not_reach_for_the_cli_package():
    """The standalone RW-WPS wheel stages this module and not the CLI.

    ``tools/build_rw_wps_release.py`` refuses a staged tree with an
    import of a module it did not stage, and it reads the AST -- so an
    import nested inside a function counts.  A door that imported
    ``woof.cli`` would either break that wheel's build or, exempted,
    answer a reader on it with an ImportError traceback.
    """

    import ast
    from pathlib import Path

    source = Path(doctor.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    reached = {node.module for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) and node.module}
    reached |= {alias.name for node in ast.walk(tree)
                if isinstance(node, ast.Import) for alias in node.names}
    assert "woof.cli" not in reached
    assert hasattr(cli, "main")     # the console door still exists


def test_the_module_door_is_reachable_as_a_module(tmp_path):
    """The artifact, not the function: run the real interpreter door.

    A ``main`` nobody calls is exactly the defect -- ``woof/doctor.py``
    had one report path and no ``__main__`` guard, so the module door
    imported the module and exited 0 in silence.
    """

    proc = subprocess.run(
        [sys.executable, "-m", "woof.doctor", "--json"],
        capture_output=True, text=True, timeout=600)
    assert proc.stdout.strip(), (
        f"the module door printed nothing (rc {proc.returncode}); "
        f"stderr: {proc.stderr[-2000:]}")
    assert proc.returncode in (0, 1), proc.stderr[-2000:]


# ---------------------------------------------------------------------------
# The silent span (#163)
# ---------------------------------------------------------------------------

def test_collect_checks_announces_the_probes_that_take_seconds():
    """The phases a reader waits through are named as they begin."""

    seen: list[str] = []
    doctor.collect_checks(sources=(), progress=seen.append)
    assert seen, "collect_checks announced nothing"
    joined = " | ".join(seen)
    # The two measured slow spans on a first-contact box: the subprocess
    # import probe per declared package, and the non-repository import
    # probe.  Both are named before they run.
    assert any("package" in line for line in seen), joined
    assert any("provenance" in line or "identity" in line
               for line in seen), joined


def test_progress_goes_to_stderr_and_the_report_is_unchanged(monkeypatch,
                                                             capsys):
    """Progress must not enter the report: ``--explain``'s long form and
    the terse form are both pinned verbatim by golden tests, and a
    machine reading ``--json`` gets JSON and nothing else."""

    sample = [doctor.Check("python", "verified", "3.13.0",
                           brief="3.13.0")]

    def fake_collect(sources=None, progress=None):
        if progress is not None:
            progress("python packages this install declares")
        return list(sample)

    monkeypatch.setattr(doctor, "collect_checks", fake_collect)

    class Args:
        source = None
        json = False
        explain = False

    code = doctor.doctor_main(Args())
    captured = capsys.readouterr()
    assert code == 0
    assert "python packages this install declares" in captured.err
    assert "python packages this install declares" not in captured.out
    assert doctor.format_brief(sample) in captured.out


def test_progress_stays_out_of_the_json_document(monkeypatch, capsys):
    import json

    sample = [doctor.Check("python", "verified", "3.13.0", brief="3.13.0")]

    def fake_collect(sources=None, progress=None):
        if progress is not None:
            progress("python packages this install declares")
        return list(sample)

    monkeypatch.setattr(doctor, "collect_checks", fake_collect)

    class Args:
        source = None
        json = True
        explain = False

    doctor.doctor_main(Args())
    captured = capsys.readouterr()
    assert json.loads(captured.out)[0]["name"] == "python"
    assert "python packages" in captured.err


@pytest.mark.parametrize("phase", ["a", "b"])
def test_progress_lines_carry_the_command_name(monkeypatch, capsys, phase):
    """A stderr line with no subject is noise in a piped log."""

    def fake_collect(sources=None, progress=None):
        progress(phase)
        return []

    monkeypatch.setattr(doctor, "collect_checks", fake_collect)

    class Args:
        source = None
        json = False
        explain = False

    doctor.doctor_main(Args())
    err = capsys.readouterr().err
    assert err.startswith("woof doctor:"), err
    assert phase in err
