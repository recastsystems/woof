"""Selected history sizing follows the writer's inventory and storage axes."""
from __future__ import annotations

from dataclasses import replace
import subprocess
import sys
from types import SimpleNamespace

import pytest

from woof.config import RunConfig, soil_layer_count
from woof.io.history_layout import history_frame_bytes, produced_history_shapes
from woof.io.history_selection import HistorySelection


def _cfg(**overrides):
    return replace(RunConfig(nx=32, ny=24, nz=12, dx=3000.0, dy=3000.0,
                             ztop=12000.0, dt=6.0, run_seconds=120.0,
                             moist=True, mp_physics=6), **overrides)


def priced_run(nx, ny, nz=49, **overrides) -> RunConfig:
    """A grid run with the engine's default real-case suite, as the wizard emits it.

    For pricing tests whose subject is the grid, the clock or the pictures:
    history is the writer's inventory for a resolved RunConfig, so a test grid
    carries a real physics suite rather than a bare size.
    """
    from dataclasses import fields
    from woof.domain_wizard import shared_physics
    from woof.physics_registry import MORRISON_TEMPLATE_ID

    names = {row.name for row in fields(RunConfig)}
    settings = {key: value for key, value in shared_physics(MORRISON_TEMPLATE_ID).items()
                if key in names}
    # The wizard's 49-level ladder belongs to its own nz; a test grid may vary nz.
    settings.pop("eta_levels", None)
    settings.update(nx=nx, ny=ny, nz=nz, dx=3000.0, dy=3000.0, dt=18.0,
                    run_seconds=3600.0)
    settings.update(overrides)
    return RunConfig(**settings)


def test_history_sizing_stays_importable_without_gpu_or_writer_runtime():
    script = r'''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == "cupy" or fullname.startswith("cupy.")
                or fullname in {"woof.core.physics", "woof.io.wrfout", "netCDF4"}):
            raise AssertionError("runtime import: " + fullname)
sys.meta_path.insert(0, Block())
from dataclasses import replace
from woof.config import RunConfig
from woof.io.history_layout import history_frame_bytes
for surface, pbl, sfclay in ((0, 0, 0), (2, 1, 1), (3, 5, 5), (4, 5, 5)):
    cfg = RunConfig(nx=32, ny=24, nz=12, dx=3000.0, dy=3000.0,
                    ztop=12000.0, dt=6.0, run_seconds=120.0,
                    moist=True, mp_physics=6, sf_surface_physics=surface,
                    bl_pbl_physics=pbl, sf_sfclay_physics=sfclay,
                    num_soil_layers=9 if surface == 3 else 4)
    assert history_frame_bytes(cfg) > 0
'''
    completed = subprocess.run([sys.executable, "-c", script],
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_shapes_keep_faces_geography_and_vertical_coordinates():
    shapes = produced_history_shapes(_cfg())
    assert shapes["U"] == (12, 24, 33)
    assert shapes["V"] == (12, 25, 32)
    assert shapes["W"] == shapes["PHB"] == (13, 24, 32)
    assert shapes["XLAT_U"] == (24, 33)
    assert shapes["XLONG_V"] == (25, 32)
    assert shapes["ZNU"] == (12,)
    assert shapes["ZNW"] == (13,)


@pytest.mark.parametrize("surface,layers", [(2, 4), (3, 6), (3, 9), (4, 4)])
def test_soil_axes_follow_the_selected_land_scheme(surface, layers):
    cfg = _cfg(sf_surface_physics=surface, num_soil_layers=layers)
    shapes = produced_history_shapes(cfg)
    assert shapes["TSLB"] == shapes["SMOIS"] == (soil_layer_count(cfg), 24, 32)
    if surface == 3:
        assert shapes["SMFR3D"] == shapes["KEEPFR3DFLAG"] == (layers, 24, 32)
    if surface == 4:
        assert shapes["TSNO"] == (3, 24, 32)
        assert shapes["ZSNSO"] == (3 + soil_layer_count(cfg), 24, 32)


def test_soil_sizing_retains_the_wrong_geometry_refusal():
    with pytest.raises(ValueError, match="num_soil_layers must be 6 or 9"):
        produced_history_shapes(_cfg(sf_surface_physics=3, num_soil_layers=4))


def test_mynn_history_prices_its_padded_top_interface():
    shapes = produced_history_shapes(_cfg(bl_pbl_physics=5))
    assert shapes["EL_PBL"] == shapes["EXCH_H"] == shapes["EXCH_M"] == (13, 24, 32)


def test_slope_radiation_output_follows_the_live_attachment_predicate():
    cfg = _cfg(slope_rad=1, ra_physics=1, sf_surface_physics=2)
    assert produced_history_shapes(cfg)["SWNORM"] == (24, 32)
    assert "SWNORM" not in produced_history_shapes(replace(cfg, ra_physics=0))


def test_trimmed_selection_removes_volumes_but_keeps_reader_coordinates():
    cfg = _cfg()
    trimmed = HistorySelection(history_vars=("T2",))
    full_shapes = produced_history_shapes(cfg)
    selected = set(trimmed.select(full_shapes))
    assert {"T", "PHB", "XLAT", "XLONG", "Times"} <= selected
    assert {"U", "QVAPOR", "QGRAUP", "REFL_10CM"}.isdisjoint(selected)
    assert history_frame_bytes(cfg, trimmed) < history_frame_bytes(cfg) / 2


def test_initial_frame_omits_the_output_due_reflectivity_stash():
    cfg = _cfg()
    assert "REFL_10CM" in produced_history_shapes(cfg)
    assert "REFL_10CM" not in produced_history_shapes(cfg, include_reflectivity=False)
    assert "REFL_10CM" not in produced_history_shapes(replace(cfg, mp_physics=0))


def test_a_grid_size_alone_is_refused_rather_than_priced_dry():
    # A190: filled from the dataclass's dry defaults, a moist 552x552x49 child
    # was priced at 0.50 GB a frame where it wrote 1.2 GB.
    dimensions = SimpleNamespace(nx=32, ny=24, nz=12)
    for price in (produced_history_shapes, history_frame_bytes):
        with pytest.raises(TypeError, match="resolved RunConfig"):
            price(dimensions)
    # A domain wrapper is read through its run, as projected_run_bytes passes it.
    assert history_frame_bytes(SimpleNamespace(run=_cfg())) == history_frame_bytes(_cfg())


@pytest.mark.usefixtures("classic_history_frames")
@pytest.mark.parametrize("preset", ["full", "minimal"])
def test_cdf2_file_bytes_stay_below_the_header_allowance(tmp_path, preset):
    import numpy as np
    from woof.io import nc_writer_bridge
    from woof.io.wrfout import WrfoutWriter
    from woof.io.wrf_output_schema import HISTORY_FIELDS_BY_NETCDF_NAME

    reason = nc_writer_bridge.unavailable_reason()
    if reason is not None:
        pytest.skip(reason)
    cfg = _cfg(sf_surface_physics=2, sf_sfclay_physics=1, bl_pbl_physics=1)
    selection = HistorySelection(preset=preset)
    shapes = produced_history_shapes(cfg)
    frame = {}
    for name in selection.select(shapes):
        if name in {"Times", "XTIME", "ITIMESTEP"}:
            continue
        schema = HISTORY_FIELDS_BY_NETCDF_NAME.get(name)
        dtype = "i4" if schema is not None and schema.dtype == "i4" else "f4"
        frame[name] = np.zeros(shapes[name], dtype=dtype)
    path = tmp_path / "wrfout"
    with WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
                      soil_layers=soil_layer_count(cfg), dx=cfg.dx, dy=cfg.dy,
                      global_attrs={"START_DATE": "2020-01-01_00:00:00", "DT": 1.0},
                      field_schema=frame, engine="rust") as writer:
        writer.write_frame("2020-01-01_00:00:00", frame)
    estimate = history_frame_bytes(cfg, selection)
    actual = path.stat().st_size
    assert actual <= estimate
    assert estimate - actual < 65536 + 256 * len(selection.select(shapes))
