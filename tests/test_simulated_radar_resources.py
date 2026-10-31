"""Prove native quote and allocation refusals before any history sampling."""
import json
from pathlib import Path
import subprocess

import pytest

from woof import bridges, rustwx


@pytest.fixture
def binary():
    with bridges.inspection_only():
        result = rustwx.simulated_radar_binary()
    assert result is not None, "native CPU estate requires the built rw_simradar"
    return result


def run(binary, tmp_path, config, *, mode="--estimate", history=()):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"schema": "simulated-radar.request/v1",
        "history_paths": list(history), "outdir": str(tmp_path / "output"),
        "config": config}), encoding="utf-8")
    return subprocess.run([str(binary), mode, str(request)], capture_output=True,
                          text=True, timeout=30)


def test_native_quote_is_read_only_and_has_counts_for_worker_pricing(binary, tmp_path):
    result = run(binary, tmp_path, {"sites": ["KTLX", "KICT", "KAMA"],
        "fields": ["reflectivity", "velocity"]})
    assert result.returncode == 0, result.stderr
    quote = json.loads(result.stdout)
    assert quote["schema"] == "simulated-radar.resources/v1"
    assert quote["geometry"]["rays_per_cut"] == 360
    assert quote["geometry"]["gates_per_ray"] == 920
    assert quote["geometry"]["physical_cuts"] > 14  # Repeated VCP cuts cost work too.
    assert quote["geometry"]["site_count"] == 3
    assert quote["geometry"]["sites_processed_concurrently"] is None
    assert "memory-admitted batches" in quote["geometry"]["site_concurrency_policy"]
    assert quote["geometry"]["quadrature_points_per_bin"] == 9
    assert quote["geometry_only"] and not quote["input_fields_validated"]
    assert not quote["price_or_duration_guarantee"]
    assert quote["measurement_reference"]["elapsed_seconds"] == 20.436
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("field", ["gate_spacing_m", "azimuth_step_deg"])
def test_tiny_positive_spacing_cannot_reach_history_read(binary, tmp_path, field):
    result = run(binary, tmp_path, {field: 1e-300}, mode="--request",
                 history=[str(tmp_path / "must-not-be-opened.nc")])
    assert result.returncode != 0
    assert "must-not-be-opened" not in result.stderr
    assert "whole metre" in result.stderr or "geometry" in result.stderr
    assert not (tmp_path / "output").exists()


def test_format_valid_multiterabyte_scan_is_refused_before_history_hash(binary, tmp_path):
    result = run(binary, tmp_path, {"formats": ["cfradial1"],
        "fields": ["reflectivity", "velocity"], "scan_strategy": "custom",
        "elevations_deg": [i * .25 for i in range(255)],
        "azimuth_step_deg": .006, "gate_spacing_m": 32, "range_km": 460},
        mode="--request", history=[str(tmp_path / "must-not-be-opened.nc")])
    assert result.returncode != 0
    assert "host bytes" in result.stderr, result.stderr
    assert "must-not-be-opened" not in result.stderr
    assert not (tmp_path / "output").exists()


def test_auto_sites_quote_does_not_invent_a_selected_site_count(binary, tmp_path):
    result = run(binary, tmp_path, {})
    assert result.returncode == 0, result.stderr
    quote = json.loads(result.stdout)
    assert quote["geometry"]["site_count"] is None
    assert quote["geometry"]["auto_site_count_requires_domain_coverage"]


def test_display_frame_names_missing_columns_through_the_real_request_cli(binary, tmp_path):
    import netCDF4
    source = tmp_path / "display.nc"
    with netCDF4.Dataset(source, "w", format="NETCDF3_64BIT_OFFSET") as dataset:
        dataset.createDimension("south_north", 2)
        dataset.createDimension("west_east", 2)
        for name in ("XLAT", "XLONG", "HGT", "REFC", "U10", "V10"):
            dataset.createVariable(name, "f4", ("south_north", "west_east"))[:] = 0.
    result = run(binary, tmp_path, {}, mode="--request", history=[str(source)])
    assert result.returncode != 0
    assert "radar_input_missing_columns:" in result.stderr
    for name in ("T", "U", "V", "W", "PH", "PHB", "bottom_top"):
        assert name in result.stderr
    assert not (tmp_path / "output").exists()


def test_history_shapes_enter_memory_quote_without_writing(binary, tmp_path):
    from test_canonical_radar_scene import capsule
    source = tmp_path / "columns.nc"
    canonical = tmp_path / "full-atmosphere.nc"
    capsule(source)
    converted = subprocess.run([str(binary), "--canonical-atmosphere", str(source),
                                "--out", str(canonical)], capture_output=True, text=True, timeout=30)
    assert converted.returncode == 0, converted.stderr
    result = run(binary, tmp_path, {"timing": "scan"}, history=[str(canonical)])
    assert result.returncode == 0, result.stderr
    quote = json.loads(result.stdout)
    assert [quote["histories"][0][axis] for axis in ("nx", "ny", "nz")] == [5, 5, 3]
    assert quote["atmospheres_retained_upper_bound"] == 2
    assert quote["estimated_peak_working_bytes"] > quote["geometry"]["scan_working_bytes"]
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("key,value", [("range_km", 461), ("azimuth_step_deg", 721),
                                       ("volume_duration_s", 3601)])
def test_native_and_python_ceilings_name_the_same_breakage(binary, tmp_path, key, value):
    from woof.simulated_radar_config import GEOMETRY_LIMITS
    result = run(binary, tmp_path, {key: value})
    assert result.returncode != 0
    limit, reason = GEOMETRY_LIMITS[key]
    assert f"{key} must be at most {limit:g}: {reason}" in result.stderr


def test_forecast_shapes_and_output_bound_reach_the_quote(binary, tmp_path):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"schema": "simulated-radar.request/v1", "history_paths": [],
        "outdir": str(tmp_path / "output"), "scene_shapes": [[1799, 1059, 50]],
        "config": {"sites": ["KTLX"]}}), encoding="utf-8")
    quote = json.loads(subprocess.run([str(binary), "--estimate", str(request)],
                                      capture_output=True, text=True, timeout=30,
                                      check=True).stdout)
    assert quote["histories"][0]["atmosphere_working_bytes"] == 1799 * 1059 * 50 * 128
    assert quote["estimated_peak_working_bytes"] == (
        quote["geometry"]["scan_working_bytes"] + 1799 * 1059 * 50 * 128)
    assert quote["output"]["output_bytes_per_site_volume_upper_bound"] > 0
    assert not quote["geometry_only"]
    request.write_text(json.dumps({"schema": "simulated-radar.request/v1",
        "history_paths": [str(tmp_path / "must-not-be-opened.nc")],
        "outdir": str(tmp_path / "output"), "scene_shapes": [[4, 4, 3]],
        "config": {"sites": ["KTLX"]}}), encoding="utf-8")
    refused = subprocess.run([str(binary), "--request", str(request)], capture_output=True,
                             text=True, timeout=30)
    assert refused.returncode != 0 and "scene_shapes" in refused.stderr
    assert "must-not-be-opened" not in refused.stderr
    assert not (tmp_path / "output").exists()


def test_history_without_t_is_quoted_from_its_mass_grid_and_refused_at_request(binary, tmp_path):
    """wrf-core 93bd2ca's reader takes grid dimensions from another mass volume when T is absent.

    The older 9874474 reader refused such a file at open ("variable not found in WRF file:
    Cannot determine grid dimensions: variable 'T' not found"), so --estimate failed. Now the
    quote reads the shape from P, and the run itself is still refused by the input contract.
    """
    import netCDF4
    source = tmp_path / "no-t.nc"
    with netCDF4.Dataset(source, "w", format="NETCDF3_64BIT_OFFSET") as dataset:
        dataset.setncattr("BOTTOM-TOP_GRID_DIMENSION", 4)
        for name, size in (("Time", 1), ("bottom_top", 3), ("south_north", 5), ("west_east", 6)):
            dataset.createDimension(name, size)
        for name in ("XLAT", "XLONG", "HGT"):
            dataset.createVariable(name, "f4", ("Time", "south_north", "west_east"))[:] = 0.
        for name in ("P", "PB", "QVAPOR", "QRAIN"):
            dataset.createVariable(
                name, "f4", ("Time", "bottom_top", "south_north", "west_east"))[:] = 0.
    quoted = run(binary, tmp_path, {"timing": "scan"}, history=[str(source)])
    assert quoted.returncode == 0, quoted.stderr
    assert "Cannot determine grid dimensions" not in quoted.stderr
    history = json.loads(quoted.stdout)["histories"][0]
    assert [history[axis] for axis in ("nx", "ny", "nz")] == [6, 5, 3]
    refused = run(binary, tmp_path, {}, mode="--request", history=[str(source)])
    assert refused.returncode != 0
    assert "radar_input_missing_columns:" in refused.stderr
    assert "T" in refused.stderr.split("lacks variables [", 1)[1].split("]", 1)[0].split(", ")
    assert not (tmp_path / "output").exists()
