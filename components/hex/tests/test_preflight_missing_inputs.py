"""A raw unit-sphere grid must not mask missing physical input files."""
import sys

import pytest
from netCDF4 import Dataset

from woof.hex.cli import main


@pytest.mark.parametrize("suite", ["convection_permitting", "arwen_mpas_column"])
@pytest.mark.parametrize("missing", [("static",), ("init",), ("static", "init")])
def test_real_preflight_names_missing_files_before_timestep(tmp_path, monkeypatch, capsys, missing, suite):
    monkeypatch.setattr(sys, "argv", ["woof hex"])
    paths = {name: tmp_path / f"{name}.nc" for name in ("grid", "static", "init")}
    for name, path in paths.items():
        if name in missing:
            continue
        with Dataset(path, "w") as ds:
            ds.createDimension("nEdges", 1)
            ds.createVariable("dcEdge", "f8", ("nEdges",))[:] = 0.0188347
    args = ["forecast", "--physics-suite", suite, "--mesh", "hex-x1.40962-native", "--hours", "24",
            "--history-every-minutes", "60", "--start-time", "2026-10-27_12:00:00",
            "--out", str(tmp_path / "out"), "--preflight"]
    for name, path in paths.items():
        args.extend((f"--{name}", str(path)))
    assert main(args) == 2
    err = capsys.readouterr().err
    for name in missing:
        assert f"--{name} names a missing file: {paths[name]}" in err
        assert ("woof hex init" if name == "init" else "woof hex mesh-plan") in err
    assert "global --stage" not in err
    assert "no timestep divides" not in err
    assert "MPAS suite timestep" not in err
    assert not (tmp_path / "out" / "native-physics-request.json").exists()
