"""Turbine input authority and ordinary user doors, without a GPU."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import math
import tomllib
import pytest

from woof.config import RunConfig
from woof.wind_inputs import prepare_launch
from woof.wind_table import read_table, validate_config


def config(**kw):
    return RunConfig(nx=9, ny=9, nz=40, dx=3000, dy=3000, ztop=20000,
                     dt=15, run_seconds=60, **kw)


def wrf_table(tmp_path):
    path = tmp_path / "windturbines.txt"
    path.write_text("36.2 -97.4 1 farm-a\n-30.0 179.9 1 farm-b\n")
    (tmp_path / "wind-turbine-1.tbl").write_text(
        "3\n100 120 0.05 2000\n3 0.8 0\n10 0.8 1500\n25 0.1 0\n")
    return path


def test_wrf_native_conversion_and_dependency_identity(tmp_path):
    path = wrf_table(tmp_path)
    data, digest = read_table(path)
    assert [t["farm_id"] for t in data["turbines"]] == ["farm-a", "farm-b"]
    assert data["types"]["1"]["curve"][1][2] == pytest.approx(1500000/(.5*1.23*math.pi*60**2*10**3))
    assert data["types"]["1"]["standing_ct"] == .05
    table = tmp_path / "wind-turbine-1.tbl"
    table.write_text(table.read_text().replace("1500", "1400"))
    assert read_table(path)[1] != digest


def test_wrf_native_refuses_missing_curve_and_invalid_latitude(tmp_path):
    path = wrf_table(tmp_path)
    path.write_text("91 0 1\n")
    with pytest.raises(ValueError, match="invalid geographic"):
        read_table(path)
    path.write_text("30 0 2\n")
    with pytest.raises(ValueError, match="WRF turbine input"):
        read_table(path)


def test_python_option_and_off_are_explicit(tmp_path):
    path = wrf_table(tmp_path)
    cfg = config(wind_farms=str(path), bl_pbl_physics=5, bl_mynn_tkeadvect=True)
    assert cfg.wind_farm_scheme == "fitch"
    validate_config(cfg)
    off = replace(cfg, wind_farms="off")
    assert off.wind_farm_scheme == "off" and off.wind_farm_table == ""
    validate_config(off)


@pytest.mark.parametrize("options,reason", [
    ({"bl_pbl_physics": 1}, "other PBL carriers"),
    ({"bldt": 1}, "held PBL tendency"),
])
def test_science_refusal_names_missing_coupling(tmp_path, options, reason):
    cfg = config(wind_farms=str(wrf_table(tmp_path)), bl_pbl_physics=5, bl_mynn_tkeadvect=True)
    with pytest.raises(ValueError, match=reason):
        validate_config(replace(cfg, **options))


def test_go_keeps_original_and_binds_every_domain(tmp_path):
    original = tmp_path / "forecast.toml"
    original.write_text('[experiment]\nname="demo"\n[shared]\nbl_pbl_physics=5\n'
                        '[[domain]]\ngrid_id=1\n[[domain]]\ngrid_id=2\nwind_farms="old.txt"\n'
                        '[fetch]\nout="relative/forcing"\n')
    before = original.read_bytes()
    args = SimpleNamespace(config=original, wind_farms=str(wrf_table(tmp_path)))
    prepare_launch(args)
    raw = tomllib.loads(Path(args.config).read_text())
    assert original.read_bytes() == before
    assert Path(args.config).parent == original.parent
    assert raw["fetch"]["out"] == "relative/forcing"
    assert raw["shared"]["wind_farms"] == str((tmp_path / "windturbines.txt").resolve())
    assert all("wind_farms" not in d for d in raw["domain"])


def test_absent_option_does_not_read_config():
    args = SimpleNamespace(config=Path("absent.toml"))
    prepare_launch(args)
    assert args.config == Path("absent.toml")


def test_retained_local_tke_also_accepts_fitch(tmp_path):
    validate_config(config(wind_farms=str(wrf_table(tmp_path)), bl_pbl_physics=5,
                           bl_mynn_tkeadvect=False))


def test_namelist_fitch_is_per_domain(tmp_path):
    from woof.namelist_import import import_namelists
    from test_namelist_import import INPUT_TEXT, _pair
    text = INPUT_TEXT.replace("&physics", "&physics\n windfarm_opt=0,1,\n bl_mynn_tkeadvect=.true.,")
    import re
    text = re.sub(r"bl_pbl_physics\s*=.*", "bl_pbl_physics = 5,5,", text)
    wrf_table(tmp_path)
    written, _ = import_namelists(*_pair(tmp_path, inp=text), name="wind")
    result = tomllib.loads(written)
    assert "wind_farms" not in result["domain"][0]
    assert result["domain"][1]["wind_farms"].endswith("windturbines.txt")


def test_decoder_artifact_carries_its_source_revision():
    import subprocess
    from woof.wind_inputs import table_binary
    from woof.bridge_assets import embedded_source_revisions
    binary = table_binary()
    probe = subprocess.run([str(binary), "--version"], check=True,
                           capture_output=True, timeout=10)
    assert b"rw-wind-table input-v1" in probe.stdout
    revisions = embedded_source_revisions(binary.read_bytes())
    assert len(revisions) == 1
    assert revisions[0].encode() in probe.stdout
