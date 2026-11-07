"""2.9.4: five doors no forecast reaches went on demand to keep the Linux wheel
under PyPI's 100,000,000-byte per-file limit (the cut's wheel-size pre-gate
measured 100,878,471 bytes).  Each door's own resolver stages its pinned member
on first use, and doctor reports the on-demand row rather than "not staged"."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from woof import bridge_assets, bridges

NEW_ON_DEMAND = ("rw_hail", "rw_airnow", "rw_scoreboard", "rw_tc_fields", "rw_fire_indices")


@pytest.fixture
def pip_install(tmp_path, monkeypatch):
    """No Rust sources, a bundle pinned for this platform, nothing staged."""
    platform = "linux-x86_64"
    binaries = tuple(
        bridge_assets.BinaryPin(a.name, bridge_assets.artifact_filename(a, platform), 8,
                                hashlib.sha256(a.name.encode()).hexdigest())
        for a in bridge_assets.BUNDLED_ARTIFACTS)
    bundle = bridge_assets.BundlePin(platform=platform, filename="gpuwm-bridges-v0-test.zip",
                                     bytes=96 * 1024 * 1024, sha256="b" * 64, binaries=binaries)
    pins = bridge_assets.BridgePins(release="v0-test", platforms={platform: bundle})
    monkeypatch.setattr(bridge_assets, "load_pins", lambda path=None: pins)
    monkeypatch.setattr(bridge_assets, "host_platform", lambda: platform)
    monkeypatch.setattr(bridges, "_package_parent", lambda: tmp_path / "site-packages")
    monkeypatch.setattr(bridges, "default_bridge_dir", lambda: tmp_path / "staged")
    monkeypatch.setattr(bridges, "_ON_DEMAND_ATTEMPTS", {})
    for name in ("WOOF_RW_SCOREBOARD", "WOOF_RW_TC_FIELDS", "WOOF_FIRE_INDICES",
                 "WOOF_RW_HAIL", "WOOF_RW_AIRNOW"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "staged"


def _fake_fetch(monkeypatch, staged_dir: Path, calls: list):
    import woof.bridge_partial as bridge_partial

    def fake_fetch_partial(pins, bundle, dest, artifacts, *, progress=print, urlopen_fn=None):
        calls.append(list(artifacts))
        staged_dir.mkdir(parents=True, exist_ok=True)
        for name in artifacts:
            (staged_dir / name).write_bytes(b"pinned bytes")
        return [staged_dir / name for name in artifacts]

    monkeypatch.setattr(bridge_partial, "fetch_partial", fake_fetch_partial)


def test_the_five_are_on_demand_and_out_of_the_wheel():
    by_name = {a.name: a for a in bridge_assets.BUNDLED_ARTIFACTS}
    wheel = {a.name for a in bridge_assets.wheel_artifacts()}
    for name in NEW_ON_DEMAND:
        assert by_name[name].on_demand, name
        assert name not in wheel, name
    # The forecast's own tools stay in the wheel.
    for default in ("rw_wrfbatch", "rw_netcdf", "netcdf_writer", "gpuwm_mapped_engine",
                    "rw_fetch", "static_fields"):
        assert default in wheel, default


@pytest.mark.parametrize("instrument", ["hail", "airnow"])
def test_the_obs_front_doors_stage_on_first_use(pip_install, monkeypatch, instrument):
    from woof.obs import frontdoor
    door = frontdoor.FRONT_DOORS[instrument]
    staged = pip_install / door.name
    calls: list = []
    _fake_fetch(monkeypatch, pip_install, calls)
    monkeypatch.setattr(frontdoor.FrontDoor, "find",
                        lambda self: staged if staged.is_file() else None)
    assert door.require() == staged
    assert calls == [[door.name]]


def test_the_scoreboard_door_stages_on_first_use(pip_install, monkeypatch, tmp_path):
    from woof import scoreboard
    staged = pip_install / "rw_scoreboard"
    calls: list = []
    _fake_fetch(monkeypatch, pip_install, calls)

    class Stub:
        name = "rw_scoreboard"
        env_var = "WOOF_RW_SCOREBOARD"

        def find(self):
            return staged if staged.is_file() else None

        def remedy(self):
            return "remedy"

    seen = []
    monkeypatch.setattr(scoreboard, "BINARY", Stub())
    monkeypatch.setattr(scoreboard, "abi_line", lambda path: seen.append(path))
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")
    assert scoreboard.run_request(request) == 3          # the stub speaks no contract
    assert calls == [["rw_scoreboard"]] and seen == [staged]


def test_the_cyclone_field_reader_stages_on_first_use(pip_install, monkeypatch):
    from woof import tc_tracks
    staged = pip_install / "rw_tc_fields"
    calls: list = []
    _fake_fetch(monkeypatch, pip_install, calls)
    monkeypatch.setattr(bridges, "find_artifact",
                        lambda env, filename, **kw: staged if staged.is_file() else None)
    monkeypatch.setattr(tc_tracks.shutil, "which", lambda name: None)
    assert tc_tracks.binary("rw_tc_fields") == str(staged)
    assert calls == [["rw_tc_fields"]]


def test_the_fire_danger_door_stages_on_first_use(pip_install, monkeypatch):
    from woof import fire_danger_state
    staged = pip_install / "rw_fire_indices"
    calls: list = []
    _fake_fetch(monkeypatch, pip_install, calls)
    monkeypatch.setattr(fire_danger_state, "_find", lambda: staged if staged.is_file() else None)
    assert fire_danger_state.binary() == staged
    assert calls == [["rw_fire_indices"]]


def test_a_missing_fire_override_is_never_fetched_around(pip_install, monkeypatch, tmp_path):
    from woof import fire_danger_state
    import woof.bridge_partial as bridge_partial
    monkeypatch.setenv("WOOF_FIRE_INDICES", str(tmp_path / "nope"))
    monkeypatch.setattr(bridge_partial, "fetch_partial",
                        lambda *a, **k: pytest.fail("fetched around an explicit override"))
    with pytest.raises(FileNotFoundError):
        fire_danger_state.binary()


def test_doctor_reports_the_on_demand_row(pip_install, monkeypatch):
    from woof import doctor, scoreboard
    import woof.bridge_partial as bridge_partial
    monkeypatch.setattr(bridge_partial, "fetch_partial",
                        lambda *a, **k: pytest.fail("doctor downloaded"))
    monkeypatch.setattr(bridges, "find_artifact", lambda env, filename, **kw: None)

    class Stub:
        name = "rw_scoreboard"
        env_var = "WOOF_RW_SCOREBOARD"

        def find(self):
            return None

        def remedy(self):
            return "remedy"

    monkeypatch.setattr(scoreboard, "BINARY", Stub())
    for check in (doctor._forecast_scoreboard_check(), doctor._tc_fields_check()):
        assert check.brief == "on demand; staged by its first fetch", check
        assert check.severity == doctor.SEVERITY_OPT_IN and not check.blocking


def test_doctor_reports_the_history_grib2_exporter_on_demand(pip_install, monkeypatch):
    """rw_history_grib2 went on demand in 2.9.12 (SIZE-01); its row read "not staged; its doors
    cannot run" on the Windows wheel smoke (R12) where every other on-demand row says on demand."""
    from woof import doctor
    import woof.bridge_partial as bridge_partial
    monkeypatch.setattr(bridge_partial, "fetch_partial",
                        lambda *a, **k: pytest.fail("doctor downloaded"))
    monkeypatch.setattr(bridges, "find_artifact", lambda env, filename, **kw: None)
    monkeypatch.delenv("WOOF_RW_HISTORY_GRIB2", raising=False)
    check = doctor._history_grib2_check()
    assert check.brief == "on demand; staged by its first fetch", check
    assert check.severity == doctor.SEVERITY_OPT_IN and not check.blocking
