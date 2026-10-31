"""Front-door regressions found with actual archived native restart/history."""

from dataclasses import asdict
from datetime import datetime
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from woof.cli import main as cli_main
from woof.config import RunConfig, load_config
from woof.downscale import (
    _derive_child_run_config, _render_child_toml,
    _validate_child_surface_placement, _validate_parent_evidence_grid,
)
from woof.offline_child import OfflineChildContractError, OfflineChildPlacement
from test_downscale_cli import _point_args, _restart_evidence, _surface_child_args
from test_offline_child import _history


def test_complete_native_restart_config_round_trips_default_ladder(tmp_path):
    parent = asdict(RunConfig(nx=60, ny=60, nz=49, dx=12000, dy=12000,
                              dt=60, ztop=20000, run_seconds=180,
                              mp_physics=6, moist=True))
    assert parent["eta_levels"] is None
    derived = _derive_child_run_config(
        parent, parent=parent, ratio=3, child_nx=72, child_ny=72,
        run_seconds=180, output_interval_s=60)
    path = tmp_path / "child.toml"
    path.write_text(_render_child_toml(derived), encoding="utf-8")
    assert asdict(load_config(path)) == asdict(RunConfig(**derived))


def test_renderer_does_not_silently_drop_invalid_null():
    with pytest.raises(ValueError, match="cannot render config value None"):
        _render_child_toml({"nx": 60, "dt": None})


@pytest.mark.parametrize("dry_run", [False, True])
def test_point_refuses_overlong_window_before_writing_config(tmp_path, capsys, dry_run):
    # This request must fail before integration or pictures. Keep the
    # archive-window contract independent of the host's renderer, including
    # on the non-dry-run door; do not skip that assertion on bare checkouts.
    args = _point_args(tmp_path) + ["--render-products", "none"]
    args[args.index("--hours") + 1] = "3"
    assert cli_main(args + (["--dry-run"] if dry_run else [])) == 2
    captured = capsys.readouterr()
    assert "exceeds the archived parent forcing window of 7200 seconds" in captured.err
    assert "downscale_plan" not in captured.out
    assert not (tmp_path / "child-run.child.toml").exists()
    assert not (tmp_path / "child-run").exists()


def test_supplied_config_refuses_overlong_window(tmp_path, capsys):
    args = _surface_child_args(tmp_path)
    path = tmp_path / "child.toml"
    path.write_text(path.read_text().replace("run_seconds = 600.0", "run_seconds = 10800.0"))
    assert cli_main(args + ["--dry-run"]) == 2
    assert "exceeds the archived parent forcing window" in capsys.readouterr().err


@pytest.mark.parametrize("mismatch", ["domain", "nx", "ny", "nz", "dx", "dy"])
def test_restart_evidence_must_describe_the_actual_parent_grid(tmp_path, mismatch):
    frame = tmp_path / "history"
    _history(frame, datetime(2020, 1, 1), nx=20, ny=18)
    with netCDF4.Dataset(frame, "a") as dataset:
        dataset.GRID_ID = 1
    config = dict(nx=20, ny=18, nz=2, dx=1000.0, dy=1000.0, mp_physics=8, grid_id=1)
    binding = SimpleNamespace(domain_id=1)
    if mismatch == "domain":
        binding.domain_id = 2
    else:
        config[mismatch] *= 2
    restart = _restart_evidence(tmp_path / "restart.npz", config)
    with pytest.raises(OfflineChildContractError, match="does not match history"):
        _validate_parent_evidence_grid(frame, binding, restart)


def _surface_geometry_case(tmp_path):
    frame = tmp_path / "parent"
    _history(frame, datetime(2020, 1, 1), nx=20, ny=18)
    surface_path = tmp_path / "surface"
    with netCDF4.Dataset(surface_path, "w") as dataset:
        dataset.DX = 1000.0
        dataset.DY = 1000.0
    surface = SimpleNamespace(path=surface_path, fields={
        "XLAT": np.full((10, 12), 35.0, dtype=np.float32),
        "XLONG": np.full((10, 12), -97.0, dtype=np.float32)})
    placement = OfflineChildPlacement(parent_nx=20, parent_ny=18,
        child_nx=12, child_ny=10, parent_grid_ratio=1,
        i_parent_start=4, j_parent_start=4)
    cfg = SimpleNamespace(dx=1000.0, dy=1000.0)
    return frame, surface, placement, cfg


def test_explicit_surface_accepts_matching_coordinates_with_float32_rounding(tmp_path):
    frame, surface, placement, cfg = _surface_geometry_case(tmp_path)
    surface.fields["XLAT"][3, 4] = np.nextafter(np.float32(35), np.float32(36))
    receipt = _validate_child_surface_placement(surface, frame, placement=placement, cfg=cfg)
    assert 0 < receipt["maximum_separation_m"] < receipt["tolerance_m"]


@pytest.mark.parametrize("mismatch", ["latitude", "longitude", "spacing", "missing-coordinate"])
def test_explicit_surface_refuses_wrong_location_or_unproven_grid(tmp_path, mismatch):
    frame, surface, placement, cfg = _surface_geometry_case(tmp_path)
    if mismatch == "latitude":
        surface.fields["XLAT"] += 0.02
    elif mismatch == "longitude":
        surface.fields["XLONG"] += 0.02
    elif mismatch == "spacing":
        with netCDF4.Dataset(surface.path, "a") as dataset:
            dataset.DX = 2000.0
    else:
        del surface.fields["XLAT"]
    with pytest.raises(OfflineChildContractError, match="placement|spacing"):
        _validate_child_surface_placement(surface, frame, placement=placement, cfg=cfg)


@pytest.mark.parametrize("spacing", [4000.0, 12000.0])
@pytest.mark.parametrize("projection,latitude,longitude,true_latitude", [
    ("lambert", 34.0, -87.0, 30.0),
    ("mercator", 0.0, 179.9, 0.0),
    ("polar", 89.9, 10.0, 60.0),
    ("polar", -89.9, 10.0, -60.0),
])
def test_surface_guard_accepts_independently_projected_child_across_globe(
        tmp_path, spacing, projection, latitude, longitude, true_latitude):
    """Projected child coordinates are independent of SINT reference interpolation."""
    from woof.static.projection import projection_class

    grid = projection_class(projection)(
        ref_lat=latitude, ref_lon=longitude, truelat1=true_latitude,
        truelat2=60.0 if projection == "lambert" else true_latitude,
        stand_lon=longitude, dx=spacing, dy=spacing, e_we=61, e_sn=61)
    child = grid.nest(13, 13, 3, 73, 73)
    parent_coords = grid.ij_to_latlon(
        *np.meshgrid(np.arange(1, 61), np.arange(1, 61)))
    child_coords = child.ij_to_latlon(
        *np.meshgrid(np.arange(1, 73), np.arange(1, 73)))
    frame = tmp_path / "parent"
    with netCDF4.Dataset(frame, "w") as dataset:
        dataset.createDimension("south_north", 60)
        dataset.createDimension("west_east", 60)
        dataset.createDimension("bottom_top", 49)
        dataset.DX = spacing
        dataset.DY = spacing
        for key, value in zip(("XLAT", "XLONG"), parent_coords):
            dataset.createVariable(key, "f4", ("south_north", "west_east"))[:] = value
    surface_path = tmp_path / "surface"
    with netCDF4.Dataset(surface_path, "w") as dataset:
        dataset.DX = spacing / 3
        dataset.DY = spacing / 3
    surface = SimpleNamespace(path=surface_path, fields={
        key: np.asarray(value, dtype=np.float32)
        for key, value in zip(("XLAT", "XLONG"), child_coords)})
    placement = OfflineChildPlacement(parent_nx=60, parent_ny=60,
        child_nx=72, child_ny=72, parent_grid_ratio=3,
        i_parent_start=13, j_parent_start=13)
    cfg = SimpleNamespace(dx=spacing / 3, dy=spacing / 3)
    receipt = _validate_child_surface_placement(surface, frame, placement=placement, cfg=cfg)
    assert receipt["maximum_separation_m"] < receipt["tolerance_m"]

    # Shift by exactly one child cell using the independent projection.
    displaced = child.ij_to_latlon(
        *np.meshgrid(np.arange(2, 74), np.arange(1, 73)))
    surface.fields.update({key: np.asarray(value, dtype=np.float32)
                           for key, value in zip(("XLAT", "XLONG"), displaced)})
    with pytest.raises(OfflineChildContractError, match="do not match the child placement"):
        _validate_child_surface_placement(surface, frame, placement=placement, cfg=cfg)
