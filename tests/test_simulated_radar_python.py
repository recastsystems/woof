"""Public Python orchestration: admission, ownership and returned inventories."""
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof import rustwx
from woof.simulated_radar import RadarSite, simulate
from woof.simulated_radar_config import SimulatedRadarOptions


@pytest.fixture
def native(monkeypatch):
    calls = []
    monkeypatch.setattr(rustwx, "require_simulated_radar_binary", lambda: Path("rw_simradar"))
    monkeypatch.setattr(rustwx, "estimate_simulated_radar", lambda *a, **kw: {
        "memory_admitted_now": True, "schema": "simulated-radar.resources/v1"})
    def run(paths, **kwargs):
        calls.append((paths, kwargs))
        kwargs["events"].append(dict(event="volume_committed", domain="d01", valid_time="2026-01-01T00:00:00Z"))
        return {"volumes": [dict(domain="d01", site=dict(id="KTLX", latitude_deg=35.33,
                    longitude_deg=-97.28, antenna_height_msl_m=389.5),
                valid_time="2026-01-01T00:00:00Z", source_history=[str(paths[0])],
                files=[{"format": name, "path": f"radar/volume.{name}"}
                       for name in kwargs["config"]["formats"]])]}
    monkeypatch.setattr(rustwx, "simulate_radar", run)
    return calls


def test_paths_are_grouped_and_default_request_matches_the_door(tmp_path, native):
    path = tmp_path / "history.nc"
    path.touch()
    result = simulate(path, site="ktlx", outdir=tmp_path / "output")
    assert result.volume_time == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert result.site.id == "KTLX"
    assert result.paths == {"level2": (tmp_path / "output/radar/volume.level2",)}
    assert native[0][1]["config"] == SimulatedRadarOptions.from_mapping({
        "sites": ["KTLX"], "formats": ["level2"]}).to_mapping()


def test_site_catalog_uses_native_metadata_and_preserves_unknown_bands(monkeypatch):
    from woof import simulated_radar
    monkeypatch.setattr(rustwx,"require_simulated_radar_binary",lambda:Path("rw_simradar"))
    monkeypatch.setattr(rustwx,"renderer_env",lambda:{})
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0,stdout='{"schema":"simulated-radar.site-catalog/v1","reference_sites":[{"band":null}]}')
    monkeypatch.setattr(simulated_radar.subprocess,"run",run)
    assert simulated_radar.site_catalog()["reference_sites"][0]["band"] is None
    assert calls==[["rw_simradar","--site-catalog"]]


def test_missing_binary_keeps_the_named_fetch_remedy(monkeypatch, tmp_path):
    # Exercise an installed wheel with a matching published bundle, even
    # when pytest itself runs from a checkout with source-build remedies.
    monkeypatch.setattr("woof.bridges.sources_present", lambda *a: False)
    monkeypatch.setattr("woof.bridges._package_parent", lambda: tmp_path)
    monkeypatch.setattr("woof.bridges.prebuilt_bundle_offer",
                        lambda artifact=None: ("  woof fetch-bridges",))
    monkeypatch.setattr(rustwx, "simulated_radar_binary", lambda: None)
    # On demand since 2.9.11: the door asks for a first-use fetch; a unit test
    # never downloads, so the fetch is recorded and fails.
    fetched = []
    monkeypatch.setattr("woof.bridges.fetch_on_demand", lambda name: fetched.append(name) or False)
    with pytest.raises(ValueError, match="rw_simradar is not installed") as refusal:
        simulate(tmp_path / "history.nc", site="KTLX", outdir=tmp_path)
    assert fetched == ["rw_simradar"]
    assert "woof fetch-bridges" in str(refusal.value)
    assert "WOOF_RW_SIMRADAR" in str(refusal.value)


@pytest.mark.parametrize("failure", [False, True])
def test_domain_unwraps_runconfig_and_removes_its_owned_transport(monkeypatch, tmp_path, native, failure):
    cfg = SimpleNamespace(nx=5, ny=5, nz=3)
    source = SimpleNamespace(state=object(), cfg=SimpleNamespace(run=cfg), grid=object())
    captured = []
    def snapshot(state, run, grid, instant, directory):
        assert state is source.state and run is cfg and grid is source.grid
        assert instant.utcoffset().total_seconds() == 0
        path = Path(directory) / "history.nc"
        path.touch()
        captured.append(path)
        return path
    monkeypatch.setattr("woof.simulated_radar._snapshot_history", snapshot)
    if failure:
        def fail(*args, **kwargs):
            raise rustwx.SimulatedRadarRefusal("radar_input_missing_columns: W")
        monkeypatch.setattr(rustwx, "simulate_radar", fail)
        with pytest.raises(rustwx.SimulatedRadarRefusal, match="radar_input_missing_columns"):
            simulate(source, site="KTLX", outdir=tmp_path, valid_time="2026-01-01T00:00:00Z")
    else:
        result = simulate(source, site="KTLX", outdir=tmp_path, valid_time="2026-01-01T00:00:00Z")
        assert result.input_transport == "projected_staggered_history"
    assert len(captured) == 1 and not captured[0].parent.exists()


def test_native_columns_uses_the_shared_adapter(monkeypatch, tmp_path, native):
    source = SimpleNamespace(native_columns=tmp_path / "columns.nc")
    converted = []
    def convert(path, *, outdir):
        assert path == source.native_columns
        result = outdir / "canonical.nc"
        result.touch()
        converted.append(result)
        return result
    monkeypatch.setattr(rustwx, "canonical_radar_scene", convert)
    assert simulate(source, site="KTLX", outdir=tmp_path).input_transport == "native_columns"
    assert not converted[0].exists()


def test_custom_beam_width_is_explicit_and_default_site_identity_is_unchanged():
    site = RadarSite("SIM1", 35., -97., 350.)
    assert site.to_mapping() == dict(id="SIM1", lat=35., lon=-97., height_m=350.)
    site = RadarSite("SIM1", 35., -97., 350., beamwidth_deg=1.5)
    parsed = SimulatedRadarOptions.from_mapping({"sites": [site.to_mapping()]})
    assert parsed.to_mapping()["sites"][0]["beamwidth_deg"] == 1.5


def test_unknown_band_cannot_be_mislabeled_as_s_band(tmp_path):
    with pytest.raises(ValueError, match="radar_band_unsupported"):
        simulate("history.nc", site=RadarSite("SIM1", 35., -97., 350., band="Ku"), outdir=tmp_path)


@pytest.mark.parametrize("selection,expected", [({"band": "C"}, "C"),
                                              ({"wavelength_mm": 33.3}, "X")])
def test_band_selection_reaches_the_same_native_door(tmp_path, native, selection, expected):
    path = tmp_path / "history.nc"
    path.touch()
    simulate(path, site="KTLX", outdir=tmp_path, **selection)
    assert native[-1][1]["config"]["band"] == expected


def test_file_time_cannot_be_overridden(tmp_path, native):
    with pytest.raises(ValueError, match="Times is authoritative"):
        simulate("history.nc", site="KTLX", outdir=tmp_path, valid_time="2026-01-01T00:00:00Z")


def test_empty_paths_do_not_publish_an_old_manifest(tmp_path, native):
    with pytest.raises(ValueError, match="at least one history"):
        simulate([], site="KTLX", outdir=tmp_path)
    assert not native


def test_old_output_volumes_are_excluded_by_native_execution_events(monkeypatch, tmp_path, native):
    previous = rustwx.simulate_radar
    def run(paths, **kwargs):
        result = previous(paths, **kwargs)
        old = dict(result["volumes"][0], valid_time="2025-12-31T23:00:00Z")
        result["volumes"].append(old)
        return result
    monkeypatch.setattr(rustwx, "simulate_radar", run)
    path = tmp_path / "history.nc"
    path.touch()
    assert len(simulate(path, site="KTLX", outdir=tmp_path).volumes) == 1
