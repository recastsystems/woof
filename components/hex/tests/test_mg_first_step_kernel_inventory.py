"""Live compile inputs and queue order for first-use MPAS startup."""
from types import SimpleNamespace

from woof.hex import kernel_plan
from woof.hex.cuda_backend import aot_build
from woof.hex.mpasphys import rrtmg_sw_v842


def test_shortwave_inventory_tracks_current_source(monkeypatch):
    source = aot_build.shortwave_source()
    assert source.endswith(rrtmg_sw_v842.CUDA_SOURCE)
    for name in ("mpas_sw_aerosol_columns", "mpas_sw_aerosol_levels", "mpas_sw_aerosol_finish"):
        assert "void " + name + "(" in source
    changed = rrtmg_sw_v842.CUDA_SOURCE + "\n// new source revision\n"
    monkeypatch.setattr(rrtmg_sw_v842, "CUDA_SOURCE", changed)
    assert aot_build.shortwave_source() == source + "\n// new source revision\n"


def test_selected_physics_precedes_unrelated_catalog_without_removing_rows(monkeypatch):
    config = SimpleNamespace(table_key="mpasphys_v842",
                             selectors={"radiation_sw": "rrtmg_sw", "land_surface": "sf_noah"},
                             active=lambda _: True)
    monkeypatch.setattr(kernel_plan, "_state", {})
    kernel_plan.prioritize_physics(config)
    rows = [dict(id="unselected", physics_family="tempo_rrfs", seconds=80),
            dict(id="generic", seconds=10),
            dict(id="selected", physics_family="rrtmg_sw", seconds=2),
            dict(id="selected_suffix", physics_family="noah_shared_tables", seconds=1),
            dict(id="known_plan", last_run=1, seconds=100)]
    result = sorted(rows, key=kernel_plan._compile_priority)
    assert [row["id"] for row in result] == [
        "known_plan", "selected", "selected_suffix", "generic", "unselected"]
    assert len(result) == len(rows) and {id(row) for row in result} == {id(row) for row in rows}
