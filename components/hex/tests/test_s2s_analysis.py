"""Area-weighted analysis comparison and source/date provenance."""
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pytest

from woof.hex.s2s_analysis import (read_analysis, same_mesh_coordinates, supplied_references,
                                validate_reference, weighted_metrics)
from woof.hex.s2s_door import S2SError


def test_lagged_forecast_init_fetches_an_analysis_instead_of_reusing_it(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from woof.hex import global_ensemble, s2s_analysis
    value = {"paths": {"work": str(tmp_path)}, "start_time": "2026-10-17_00:00:00", "days": 30,
             "source": "gfs", "members": [{"init": str(tmp_path / "lag6.nc"),
                 "start_time": "2026-10-17_00:00:00", "lead_hours": 6}],
             "commands": {"init": ["init-source", "--source", "gfs", "--cycle", "2026-10-17T00",
                                    "--out-dir", "unused"]}}
    def reject_reuse(*args):
        pytest.fail("a nonzero-lead forecast must not be selected as an analysis")
    calls = []
    def capture(argv, log, env):
        calls.append(argv)
        assert "--lead-hours" not in argv
        raise RuntimeError("analysis preparation reached")
    monkeypatch.setattr(s2s_analysis, "validate_reference", reject_reuse)
    monkeypatch.setattr(global_ensemble, "_cli", capture)
    with pytest.raises(RuntimeError, match="analysis preparation reached"):
        s2s_analysis.verify_analyses(value, {}, SimpleNamespace(analysis_reference=[]), {})
    assert any(argv[argv.index("--cycle") + 1] == "2026-10-17T00" for argv in calls)


def test_native_mesh_accepts_projected_float32_geometry_and_rejects_another_cell():
    grid = np.array([2.7217698849922847e-5, 6.283037970489225], dtype=np.float64)
    # The real init writer retains these float32 words in a float64 variable.
    prepared = grid.astype(np.float32).astype(np.float64)
    assert np.max(np.abs(prepared-grid)) > 1e-7
    assert same_mesh_coordinates(np, prepared, grid)
    different = prepared.copy()
    different[1] = np.nextafter(np.float32(different[1]), np.float32(np.inf))
    assert not same_mesh_coordinates(np, different, grid)
    assert not same_mesh_coordinates(np, prepared[::-1], grid)
    assert not same_mesh_coordinates(np, prepared[:1], grid)


def test_area_weighted_bias_rmse_mask_missing_pairs():
    actual = weighted_metrics(np, np.array([2., 4., np.nan, 99.]),
                              np.array([1., 1., 3., 2.]), np.array([1., 3., 10., 0.]))
    assert actual == {"finite": 2, "bias": 2.5, "rmse": np.sqrt(7.)}
    empty = weighted_metrics(np, np.array([np.nan]), np.array([1.]), np.array([1.]))
    assert empty == {"finite": 0, "bias": None, "rmse": None}


def test_prepared_analysis_requires_source_cycle_and_zero_lead(tmp_path):
    valid = datetime(2026, 10, 17)
    path = tmp_path / "init.nc"
    path.write_bytes(b"analysis")
    receipt = {"source": "gfs", "cycle": "2026-10-17_00:00:00", "lead_hours": 0}
    manifest = tmp_path / "init-source-receipt.json"
    manifest.write_text(json.dumps(receipt))
    assert supplied_references([f"gfs:2026-10-17T00={path}"]) == {("gfs", valid): path}
    validate_reference(path, "gfs", valid)
    for changed in ({"source": "era5"}, {"cycle": "2026-10-16T00"}, {"lead_hours": 24}):
        manifest.write_text(json.dumps({**receipt, **changed}))
        with pytest.raises(S2SError, match="does not match"):
            validate_reference(path, "gfs", valid)
    with pytest.raises(S2SError, match="analysis-reference needs"):
        supplied_references(["invalid"])


def test_analysis_reader_preserves_static_cell_and_level_dimensions(monkeypatch):
    from contextlib import contextmanager
    import sys
    from types import ModuleType, SimpleNamespace
    class Variable:
        def __init__(self, values, dimensions):
            self.values, self.dimensions = values, dimensions
        def __getitem__(self, key):
            return self.values[key]
    static = {"t2m": np.array([270., 280.]), "latCell": np.array([0., 1.]),
              "lonCell": np.array([1., 2.]), "zgrid": np.arange(8).reshape(2, 4)}
    dynamic = {name: np.arange(6).reshape(1, 2, 3) for name in ("rho", "theta", "qv")}
    variables = {name: Variable(values, ("nCells", "nVertLevelsP1") if values.ndim == 2 else ("nCells",))
                 for name, values in static.items()}
    variables.update({name: Variable(values, ("Time", "nCells", "nVertLevels"))
                      for name, values in dynamic.items()})
    # A real init may omit t_init or leave it zero. It is not a reference field.
    @contextmanager
    def opened(path):
        yield SimpleNamespace(variables=variables)
    bridge = ModuleType("woof.netcdf_bridge")
    bridge.open_dataset = opened
    monkeypatch.setitem(sys.modules, "woof", ModuleType("woof"))
    monkeypatch.setitem(sys.modules, "woof.netcdf_bridge", bridge)
    actual = read_analysis(Path("analysis.nc"))
    for name, expected in static.items():
        np.testing.assert_array_equal(actual[name], expected)
    for name, expected in dynamic.items():
        np.testing.assert_array_equal(actual[name], expected[0])


def test_interrupted_owned_analysis_is_rebuilt_not_refused(tmp_path, monkeypatch):
    # fit16-rows s2s2, 2026-11-07: a full disk left init.nc beside a 0-byte receipt; the resume refused the
    # whole verify stage. An owned folder without a finished receipt is rebuilt; a finished one is reused.
    from types import SimpleNamespace
    from woof.hex import global_ensemble, s2s_analysis
    value = {"paths": {"work": str(tmp_path)}, "start_time": "2026-10-17_00:00:00", "days": 0,
             "source": "era5", "members": [],
             "commands": {"init": ["init-source", "--source", "gfs", "--cycle", "2026-10-17T00",
                                    "--out-dir", "unused"]}}
    inits = tmp_path / "inits"
    for source in ("gfs", "era5"):
        folder = inits / f"analysis-{source}-2026101700"
        folder.mkdir(parents=True)
        (folder / "init.nc").write_bytes(b"partial")
    (inits / "analysis-gfs-2026101700" / "init-source-receipt.json").write_text("")
    finished = inits / "analysis-era5-2026101700"
    (finished / "init-source-receipt.json").write_text(json.dumps({
        "source": "era5", "cycle": "2026-10-17_00:00:00", "lead_hours": 0,
        "init": {"path": str(finished / "init.nc")}}))
    calls = []
    def capture(argv, log, env):
        calls.append(argv[argv.index("--source") + 1])
        raise RuntimeError("analysis preparation reached")
    monkeypatch.setattr(global_ensemble, "_cli", capture)
    with pytest.raises(RuntimeError, match="analysis preparation reached"):
        s2s_analysis.verify_analyses(value, {}, SimpleNamespace(analysis_reference=[]), {})
    assert calls == ["gfs"]
