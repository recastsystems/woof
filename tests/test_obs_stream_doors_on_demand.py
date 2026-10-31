"""The observation-stream doors reach an install: on demand, and reported truly.

2.9.0 breakage: `woof obs streams` reported ``decoder_built: true`` for
igra2, goes-dmw, ndbc, gnss-ro, cdaac-ro and wis2 while rw_igra2, rw_amv,
rw_ndbc, rw_gnssro, rw_wis2 and rw_prepbufr were in neither the wheel nor the
bundle, and `woof obs streams fetch --stream ndbc` refused with a cargo build
from a clone.  They are now on-demand bundle artifacts staged by their first
use, and the listing reads the install.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from woof import bridge_assets, bridges
from woof.obs import frontdoor, streams

STREAM_DOORS = ("rw_igra2", "rw_amv", "rw_ndbc", "rw_gnssro", "rw_wis2", "rw_madis", "rw_prepbufr")


@pytest.fixture(autouse=True)
def _fresh_attempts(monkeypatch):
    monkeypatch.setattr(bridges, "_ON_DEMAND_ATTEMPTS", {})
    for name in STREAM_DOORS:
        monkeypatch.delenv(frontdoor.door_by_binary(name).env_var, raising=False)


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
    monkeypatch.setattr(frontdoor.FrontDoor, "find", lambda self: None)
    return pins


def test_the_six_doors_are_bundled_on_demand_and_not_in_the_wheel():
    by_name = {a.name: a for a in bridge_assets.BUNDLED_ARTIFACTS}
    wheel = {a.name for a in bridge_assets.wheel_artifacts()}
    for name in STREAM_DOORS:
        assert by_name[name].on_demand, name
        assert name not in wheel, name
        # The bundle's env var is the one the door resolves.
        assert by_name[name].env_var == frontdoor.door_by_binary(name).env_var, name
    # Nothing a forecast reaches is on demand: the six stream doors and the
    # radar feed listings (2.9.2), which only list URLs for rw_odim.
    assert {a.name for a in bridge_assets.on_demand_artifacts()} == (
        set(STREAM_DOORS) | {"rw_radar_feeds"}
        # 2.9.3 (PyPI per-file limit): five front doors, the sounding renderer, the gauge verifier.
        | {"rw_mrms", "rw_stage4", "rw_asos", "rw_goes", "rw_opera", "rw_soundings", "rw_nextgen_verify"}
        # 2.9.3 improve-1: the hydro door's provider tools, staged by its first preparation.
        | {"rw_hydro_world", "rw_hydro_params", "rw_hydrofabric", "nextgen-reservoir-prepare", "rw_hydro_init"}
        # 2.9.4 capability lanes: ocean preparation, ocean SST verification, ice observation binding.
        | {"rw_ocean", "rw_ocean_verify", "rw_ice_verify", "rw_bathy"}
        # 2.9.4 (PyPI per-file limit): verification and post-run doors no forecast reaches.
        | {"rw_hail", "rw_airnow", "rw_scoreboard", "rw_tc_fields", "rw_fire_indices"}
        # Post-2.9.4: native Rust coupled hydro output.
        | {"rw_nextgen_output"}
        # 2.9.9 (PyPI per-file limit): the fixed-ladder probability renderer.
        | {"rw_probgrid"}
        # 2.9.10 (PyPI per-file limit): the opt-in Icechunk writer and the woof hex RRFS init binaries.
        | {"rw_icechunk", "rw_mpas_init_v831", "rw_mpas_gsl_oro"}
        # 2.9.11 (PyPI per-file limit): opt-in simulated radar, ML export, the wind-farm
        # table decoder and the opt-in hydro door's payload library.
        | {"rw_simradar", "rw_mlexport", "rw-wind-table", "nextgen_hydrofabric"}
        # WOOF 1.1.0 r16 (PyPI per-file limit): European radar, and the three MPAS bridges every hex door
        # reads from the hex helpers first.
        | {"rw_odim", "rw_mpas_geometry", "rw_mpas_convert", "rw_mpas_lbc"})


def test_every_stream_with_a_door_names_a_bundled_artifact():
    bundled = {a.name for a in bridge_assets.BUNDLED_ARTIFACTS}
    for spec in streams.STREAMS.values():
        if spec.door is not None:
            assert spec.door in bundled, spec.name


def test_the_listing_reads_the_install_not_a_declaration(monkeypatch, tmp_path):
    """Nothing resolvable and nothing downloadable: not built, and says how."""
    monkeypatch.setattr(frontdoor.FrontDoor, "find", lambda self: None)
    monkeypatch.setattr(bridges, "on_demand_fetch_offered", lambda artifact, *a: False)
    rows = {row["stream"]: row for row in streams.streams_table()}
    for name in ("igra2", "goes-dmw", "ndbc", "gnss-ro", "cdaac-ro", "wis2", "iem-metar",
                 "madis-acars", "madis-raob", "madis-mesonet"):
        assert rows[name]["decoder_built"] is False, name
        assert rows[name]["door_route"] == "build-from-source", name
    assert rows["nws-upper-air-text"]["door_route"] == "none"


def test_the_listing_says_on_demand_on_a_pip_install(pip_install):
    rows = {row["stream"]: row for row in streams.streams_table()}
    for name in ("igra2", "goes-dmw", "ndbc", "gnss-ro", "cdaac-ro", "wis2",
                 "madis-acars", "madis-raob", "madis-mesonet"):
        assert rows[name]["decoder_built"] is False, name
        assert rows[name]["door_route"] == "on-demand", name
    # rw_asos (METAR) is on demand too since 2.9.3 (PyPI per-file limit).
    assert rows["iem-metar"]["door_route"] == "on-demand"


def test_the_listing_says_installed_when_the_door_resolves(monkeypatch, tmp_path):
    binary = tmp_path / "rw_ndbc"
    binary.write_bytes(b"x")
    monkeypatch.setattr(frontdoor.FrontDoor, "find", lambda self: binary)
    row = {r["stream"]: r for r in streams.streams_table()}["ndbc"]
    assert row["decoder_built"] is True
    assert row["door_route"] == "installed"
    assert row["door_path"] == str(binary)


def test_the_remedy_leads_with_the_one_file_fetch(pip_install):
    remedy = frontdoor.door_by_binary("rw_ndbc").remedy()
    first = next(line.strip() for line in remedy.splitlines()
                 if line.strip() and not line.lstrip().startswith("#"))
    assert first == "woof fetch-bridges --partial --artifact rw_ndbc"
    assert "fetched on demand" in remedy


def test_the_first_use_stages_the_door_and_runs_it(pip_install, monkeypatch, tmp_path):
    """The default command works: require() stages the pinned member itself."""
    staged = tmp_path / "staged" / "rw_ndbc"
    calls = []

    def fake_fetch_partial(pins, bundle, dest, artifacts, *, progress=print, urlopen_fn=None):
        calls.append(list(artifacts))
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(b"pinned bytes")
        return [staged]

    import woof.bridge_partial as bridge_partial
    monkeypatch.setattr(bridge_partial, "fetch_partial", fake_fetch_partial)
    monkeypatch.setattr(frontdoor.FrontDoor, "find",
                        lambda self: staged if staged.is_file() else None)
    assert frontdoor.door_by_binary("rw_ndbc").require() == staged
    assert calls == [["rw_ndbc"]]


def test_inspection_never_downloads(pip_install, monkeypatch):
    import woof.bridge_partial as bridge_partial
    monkeypatch.setattr(bridge_partial, "fetch_partial",
                        lambda *a, **k: pytest.fail("inspection downloaded"))
    with bridges.inspection_only():
        assert bridges.fetch_on_demand("rw_ndbc") is False


def test_a_wheel_carried_artifact_is_never_fetched_on_demand(pip_install, monkeypatch):
    import woof.bridge_partial as bridge_partial
    monkeypatch.setattr(bridge_partial, "fetch_partial",
                        lambda *a, **k: pytest.fail("a wheel artifact was fetched"))
    # rw_asos is on demand since 2.9.3; rw_nexrad is a wheel-carried door.
    assert bridges.fetch_on_demand("rw_nexrad") is False


def test_a_failed_first_use_says_what_happened_once(pip_install, monkeypatch):
    attempts = []

    def offline(*a, **k):
        attempts.append(1)
        raise bridge_assets.BridgeAssetError("no route to the release assets")

    import woof.bridge_partial as bridge_partial
    monkeypatch.setattr(bridge_partial, "fetch_partial", offline)
    door = frontdoor.door_by_binary("rw_igra2")
    for _ in range(2):
        with pytest.raises(RuntimeError) as refusal:
            door.require()
        said = str(refusal.value)
        assert "no route to the release assets" in said
        assert "woof fetch-bridges --partial --artifact rw_igra2" in said
    assert attempts == [1]


def test_doctor_reports_each_stream_door(pip_install):
    from woof import doctor

    checks = doctor._obs_stream_door_checks()
    assert [c.name.split()[3] for c in checks] == list(STREAM_DOORS)
    for check in checks:
        assert check.status == "missing"
        assert check.severity == doctor.SEVERITY_OPT_IN
        assert check.blocking is False
        assert "fetched on demand" in check.detail
    for name in STREAM_DOORS:
        assert name in doctor._CHECKED_ARTIFACTS
