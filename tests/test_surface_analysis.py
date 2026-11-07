from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from conftest import requires_gpu
from woof.surface_analysis import selected_records, SurfaceAnalysis, SOURCE_TABLE

def row(time="2024-01-01T00:00:00Z", **kw):
    return dict(target="SST", mask="ocean", valid_time=time,
                interval_seconds=86400, filename="0.f32", **kw)

def test_source_clock_never_looks_ahead_and_refuses_stale():
    records = [row(), {**row("2024-01-02T00:00:00Z"), "filename": "1.f32"}]
    t = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
    assert selected_records(records, t+86399)[0]["filename"] == "0.f32"
    assert selected_records(records, t+86400)[0]["filename"] == "1.f32"
    with pytest.raises(ValueError, match="future data"):
        selected_records(records, t-1)
    with pytest.raises(ValueError, match="stale SST"):
        selected_records(records, t+172800)
    with pytest.raises(ValueError, match="duplicate"):
        selected_records([row(), row()], t)

def test_new_source_is_a_table_row():
    table = json.loads(SOURCE_TABLE.read_text())
    assert set(table["sources"]) == {"oisst", "ghrsst-l4", "glsea", "osi-saf", "wrf-lower-boundary"}
    for source in table["sources"].values():
        assert source["interval_seconds"] > 0
        assert all(f["target"] in ("SST", "SEAICE", "VEGFRA", "ALBBCK") for f in source["fields"])

def test_glsea_matches_published_erddap_metadata():
    # NOAA GLSEA_ACSPO_GCS and GLSEA_GCS both declare these CF names.
    # A fixture generated from a mistaken row cannot detect this drift.
    source = json.loads(SOURCE_TABLE.read_text())["sources"]["glsea"]
    assert (source["latitude"], source["longitude"], source["time"]) == (
        "latitude", "longitude", "time")
    assert source["fields"][0]["variable"] == "sst"
    assert "degree_C" in source["fields"][0]["units"]

def fixture_manifest(tmp_path):
    def plane(name, values):
        data = np.asarray(values, dtype="<f4").tobytes()
        (tmp_path/name).write_bytes(data)
        return {"filename": name, "sha256": hashlib.sha256(data).hexdigest()}
    record = {**row(), **plane("0.f32", [280, 281])}
    manifest = dict(schema="gpuwm-surface-analysis-v1", time_policy="hold-latest-no-lookahead",
                    coordinates_sha256=hashlib.sha256(np.asarray([42,42,0,1],dtype="<f4").tobytes()).hexdigest(),
                    shape=[1,2], records=[record], water=plane("water.f32", [1,1]), lake=plane("lake.f32", [0,0]))
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest))
    return path

@pytest.mark.gpu
@requires_gpu
def test_runtime_binds_shape_time_masks_and_content(tmp_path):
    path = fixture_manifest(tmp_path)
    cfg = SimpleNamespace()
    coords = dict(latitude=np.asarray([[42,42]]), longitude=np.asarray([[0,1]]))
    bound = SurfaceAnalysis(path, shape=(1,2), start_time="2024-01-01T00:00:00Z", cfg=cfg, **coords)
    assert bound.water.tolist() == [[1,1]]
    with pytest.raises(ValueError, match="grid shape"):
        SurfaceAnalysis(path, shape=(2,1), start_time="2024-01-01", cfg=cfg, **coords)
    with pytest.raises(ValueError, match="absolute"):
        SurfaceAnalysis(path, shape=(1,2), start_time=None, cfg=cfg, **coords)
    with pytest.raises(ValueError, match="same location"):
        SurfaceAnalysis(path, shape=(1,2), start_time="2024-01-01", cfg=cfg, latitude=[[42,42]], longitude=[[1,2]])
    (tmp_path/"water.f32").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        SurfaceAnalysis(path, shape=(1,2), start_time="2024-01-01", cfg=cfg, **coords)

def test_inactive_arm_does_not_bind_or_read():
    from woof.surface_analysis import bind_surface_analysis
    driver = SimpleNamespace()
    bind_surface_analysis(driver, SimpleNamespace(sst_update=0), None)
    assert not hasattr(driver, "surface_analysis")

def test_sst_update_is_admitted_and_carried():
    from woof.namelist_import import PHYSICS_PINS
    from woof.config import RunConfig, validate_run_config
    from dataclasses import replace
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=1000, dy=1000, ztop=8000, dt=1, run_seconds=10)
    assert not any(r[0] == "sst_update" for r in PHYSICS_PINS)
    assert validate_run_config(replace(cfg, sst_update=1)).sst_update == 1
    with pytest.raises(ValueError, match="integer"):
        validate_run_config(replace(cfg, sst_update=True))


def test_disabled_surface_source_roundtrips_its_omitted_restart_keys():
    from dataclasses import replace
    from woof.config import RunConfig
    from woof.io.restart import configuration_echo, _require_config_match, RestartMismatchError
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=1000., dy=1000., ztop=8000., dt=30., run_seconds=120.)
    header = configuration_echo(cfg)
    assert "sst_update" not in header and "surface_analysis" not in header
    _require_config_match(header, cfg, "checkpoint")
    _require_config_match(header, replace(cfg, surface_analysis="inactive-source.json"), "checkpoint")
    with pytest.raises(RestartMismatchError, match="sst_update"):
        _require_config_match(header, replace(cfg, sst_update=1, surface_analysis="active-source.json"), "checkpoint")


def test_enabled_surface_source_keeps_binding_its_restart_identity():
    from dataclasses import replace
    from woof.config import RunConfig
    from woof.io.restart import configuration_echo, _require_config_match, RestartMismatchError
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=1000., dy=1000., ztop=8000., dt=30., run_seconds=120., sst_update=1, surface_analysis="source.json")
    header = configuration_echo(cfg)
    _require_config_match(header, cfg, "checkpoint")
    with pytest.raises(RestartMismatchError, match="surface_analysis"):
        _require_config_match(header, replace(cfg, surface_analysis="different-source.json"), "checkpoint")
    with pytest.raises(RestartMismatchError, match="sst_update"):
        _require_config_match(header, replace(cfg, sst_update=0), "checkpoint")
