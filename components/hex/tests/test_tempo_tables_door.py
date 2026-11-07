"""The forecast door acquires the pinned TEMPO tables before the card.

THE BREAKAGE: the door only validated the TEMPO table set, and no `woof
fetch-tables` option stages it, so on a clean pip install (golden 0.4.0 R2,
install gate) `--physics config_microp_scheme=mp_tempo` was refused
with "missing Thompson table asset .../tempo-v3/ccn_activate.bin" and no
command that closes it.  The door now verifies-or-acquires through the
engine's own ensure_tempo_tables, as the engine's TEMPO entry does, and
refuses by name only when acquisition itself fails.
"""

from __future__ import annotations

import sys
import types

from woof.hex import forecast_door


class _Row:
    name = "engine_tempo_row"


def _tempo_row(monkeypatch, ensure):
    module = types.ModuleType("woof.tempo_assets")
    module.ensure_tempo_tables = ensure
    monkeypatch.setitem(sys.modules, "woof.tempo_assets", module)
    species = types.SimpleNamespace(engine_scheme="tempo")
    monkeypatch.setattr("woof.hex.species_row.species_row_for_names", lambda names: species)
    row = _Row()
    row.driven_scalar_names = ("qv",)
    row.appended_scalar_names = ()
    return row


def test_the_door_acquires_the_tempo_tables(monkeypatch):
    calls = []
    row = _tempo_row(monkeypatch, lambda *a, **k: calls.append((a, k)))
    assert forecast_door.microphysics_table_problem(row) is None
    assert calls == [((), {})]


def test_a_failed_acquisition_refuses_by_name_with_the_offline_route(monkeypatch):
    def ensure(*_a, **_k):
        raise OSError("network unreachable")

    problem = forecast_door.microphysics_table_problem(_tempo_row(monkeypatch, ensure))
    assert "requires the pinned TEMPO tables" in problem
    assert "network unreachable" in problem
    assert "ensure_tempo_tables(source_dir=" in problem


def _sub(tmp_path):
    sub = tmp_path / "second"
    sub.mkdir()
    return sub


def _init(tmp_path, **fields):
    netCDF4 = __import__("pytest").importorskip("netCDF4")
    import numpy as np

    path = tmp_path / "init.nc"
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("nCells", 4)
        for name, value in fields.items():
            dataset.createVariable(name, "f4", ("nCells",))[:] = np.full(4, value, np.float32)
    return path


def test_tempo_from_an_init_without_aerosol_is_refused_at_the_door(monkeypatch, tmp_path):
    """THE BREAKAGE: the seam refused at step 0, after construction."""

    row = _tempo_row(monkeypatch, lambda *a, **k: None)
    problem = forecast_door.microphysics_init_problem(row, _init(tmp_path, qv=1.0))
    assert "runs TEMPO" in problem and "nwfa or nifa" in problem
    problem = forecast_door.microphysics_init_problem(row, _init(_sub(tmp_path), nwfa=0.0, nifa=1.0e9))
    assert "carries no nwfa" in problem and "nifa" not in problem.split("carries no")[1].split("(")[0]


def test_tempo_from_an_init_with_aerosol_passes_the_door(monkeypatch, tmp_path):
    row = _tempo_row(monkeypatch, lambda *a, **k: None)
    assert forecast_door.microphysics_init_problem(row, _init(tmp_path, nwfa=1.0e9, nifa=1.0e6)) is None


def test_other_schemes_never_open_the_init(monkeypatch, tmp_path):
    species = types.SimpleNamespace(engine_scheme="wsm6")
    monkeypatch.setattr("woof.hex.species_row.species_row_for_names", lambda names: species)
    row = _Row()
    row.driven_scalar_names = ("qv",)
    row.appended_scalar_names = ()
    assert forecast_door.microphysics_init_problem(row, tmp_path / "absent.nc") is None


def test_an_unreadable_init_is_left_to_the_init_reader(monkeypatch, tmp_path):
    row = _tempo_row(monkeypatch, lambda *a, **k: None)
    fake = tmp_path / "init.nc"
    fake.write_bytes(b"not netcdf")
    assert forecast_door.microphysics_init_problem(row, fake) is None


def test_an_init_without_aerosol_is_refused_before_any_table_download(monkeypatch, tmp_path):
    # release review M3: the door downloaded ~930 MB of TEMPO tables into a
    # fresh home and only then refused the init.  The init is checked first.
    calls = []
    row = _tempo_row(monkeypatch, lambda *a, **k: calls.append((a, k)))
    problems = forecast_door.microphysics_problems(row, _init(tmp_path, qv=1.0))
    assert len(problems) == 1 and "carries no nwfa or nifa" in problems[0]
    assert calls == []
    problems = forecast_door.microphysics_problems(row, _init(_sub(tmp_path), nwfa=1.0e9, nifa=1.0e6))
    assert problems == []
    assert calls == [((), {})]
