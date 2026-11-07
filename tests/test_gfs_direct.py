from __future__ import annotations

import dataclasses
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pytest

from woof.gfs_direct import (
    INPUT_MANIFEST_SCHEMA,
    _PRESSURE_LEVELS_HPA,
    _geometry_contract,
    _load_bridge_snapshots,
    _read_series,
    _source_coverage_receipt,
    _validate_grid_and_vertical_contract,
    _verify_static_receipt,
    _verify_input_manifest,
)
from woof.bridges import decode_failure_message
from woof.experiment import VerticalConfig, load_experiment
from woof.physics_compat import (
    MORRISON_PROFILE_ID,
    NOAHMP_PROFILE_ID,
    WSM6_PROFILE_ID,
    single_domain_runtime_switches,
)
from woof.ingest.grib import Era5Snapshot


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gfs_series_may_begin_at_a_lead_and_keeps_its_cadence_rules(tmp_path):
    """Converted from ``..._requires_f000_...``, and here is why.

    The old name asserted the thing this release removes: that a series
    must begin at f000.  That anchor had no product justification -- a
    GFS f018 record has the same shape as an f000 record and WPS/real
    initializes from forecast leads routinely -- and it is what forced a
    user who wanted the f174..f240 window to decode from f000 to reach
    it.  Every OTHER clause of the old test is kept verbatim below,
    because none of them were about the anchor.
    """

    for hour in (0, 3, 6, 18, 21):
        (tmp_path / f"f{hour:03}.grib2").write_bytes(b"GRIB")
    series = tmp_path / "series.tsv"
    series.write_text(
        "0\tf000.grib2\t81\n"
        "3\tf003.grib2\t96\n"
        "6\tf006.grib2\t96\n",
        encoding="utf-8")
    assert [hour for hour, _ in _read_series(series)] == [0, 3, 6]

    # The change: a series that begins at a forecast lead is a series.
    series.write_text(
        "18\tf018.grib2\t96\n21\tf021.grib2\t96\n", encoding="utf-8")
    assert [hour for hour, _ in _read_series(series)] == [18, 21]

    # ...and one time is still not a series: an initial condition with no
    # boundary time cannot force a run.
    series.write_text("18\tf018.grib2\t96\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least two times"):
        _read_series(series)

    series.write_text("0\tf000.grib2\t96\n3\tf003.grib2\t96\n")
    with pytest.raises(ValueError, match="analysis process ID 81"):
        _read_series(series)
    series.write_text("0\tf000.grib2\t81\n3\tf003.grib2\t82\n")
    with pytest.raises(ValueError, match="uncertified forecast process ID 82"):
        _read_series(series)
    series.write_text("0\tf000.grib2\n3\tf003.grib2\n7\tf006.grib2\n")
    with pytest.raises(ValueError, match="uniform"):
        _read_series(series)

    series.write_text("0\tf000.grib2\n385\tf003.grib2\n")
    with pytest.raises(ValueError, match="f384"):
        _read_series(series)

    series.write_text("0\tf000.grib2\n6\tf003.grib2\n")
    assert [hour for hour, _ in _read_series(series)] == [0, 6]

    # The cadence rules bind at a lead exactly as they do at f000.
    series.write_text("18\tf018.grib2\t96\n21\tf021.grib2\t96\n"
                      "6\tf006.grib2\t96\n", encoding="utf-8")
    with pytest.raises(ValueError, match="uniform"):
        _read_series(series)


def test_gfs_manifest_binds_all_dynamic_grib_roles(tmp_path):
    paths = {}
    for role in (
            "series", "bridge", "static_receipt", "grib-f000", "grib-f003"):
        paths[role] = tmp_path / role
        paths[role].write_text(role, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": INPUT_MANIFEST_SCHEMA,
        "files": {
            role: {"name": path.name, "sha256": _digest(path)}
            for role, path in paths.items()
        },
    }), encoding="utf-8")
    _verify_input_manifest(manifest, _digest(manifest), paths)
    paths["grib-f003"].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="grib-f003"):
        _verify_input_manifest(manifest, _digest(manifest), paths)


def test_bridge_loader_keeps_antimeridian_crop_axis_continuous(tmp_path):
    """Worldwide lane: a crop crossing 180E keeps a continuous,
    uniform ascending axis (extending past 180) instead of the
    retired rotate-and-argsort, whose non-uniform axis the
    horizontal interpolator refuses.  Fields stay unpermuted."""
    root = tmp_path / "decoded"
    root.mkdir()
    gate_text = (
        "status\tPASS\n"
        "schema\tgpuwm-gfs-grib2-bridge-v1\n"
        "cycle\t2026-07-20 00:00:00\n"
        "forecast_hours\t0,3\n"
        "pressure_levels_pa\t" + ",".join(
            str(int(level * 100)) for level in _PRESSURE_LEVELS_HPA) + "\n"
        "nx\t4\nny\t2\nlat1\t20\nlon1\t179.75\ndx\t0.25\ndy\t0.25\n"
        "lat2\t20.25\nlon2\t180.5\nnum_data_points\t8\n"
        "scan_mode\t0x40\n"
        "originating_center\t7\nmaster_table_version\t2\n"
        "local_table_version\t1\nforecast_generating_process_ids\t0:81,3:96\n"
        "land_mask_parameter\t0\n"
        "invariant_fields\tSOURCE_OROGRAPHY,LANDSEA\n"
        "invariant_fingerprint_fnv1a64\tSOURCE_OROGRAPHY:abc,LANDSEA:def\n")
    for hour in (0, 3):
        time_root = root / f"f{hour:03}"
        time_root.mkdir()
        base_2d = np.tile(np.arange(4, dtype="<f4"), (2, 1))
        for name in ("GHT", "T", "RH", "U", "V"):
            np.tile(base_2d, (21, 1, 1)).tofile(time_root / f"{name}.f32le")
        for name in (
            "PSFC", "SOURCE_OROGRAPHY", "SKINTEMP", "SNOW", "SNOWH",
            "T2", "RH2", "U10", "V10", "LANDSEA", "XICE",
            "GFS_ST000010", "GFS_ST010040", "GFS_ST040100",
            "GFS_ST100200", "GFS_SM000010", "GFS_SM010040",
            "GFS_SM040100", "GFS_SM100200",
        ):
            base_2d.tofile(time_root / f"{name}.f32le")
    dummy = tmp_path / "dummy.grib2"
    dummy.write_bytes(b"GRIB")
    records = ((0, dummy), (3, dummy))
    inventory = root / "inventory.tsv"
    inventory.write_text("test inventory\n", encoding="utf-8")
    decoded_manifest = root / "decoded-sha256.tsv"
    decoded_lines = ["hour\tvariable\tbytes\tsha256\tfilename"]
    names = (
        "GHT", "T", "RH", "U", "V",
        "PSFC", "SOURCE_OROGRAPHY", "SKINTEMP", "SNOW", "SNOWH",
        "T2", "RH2", "U10", "V10", "LANDSEA", "XICE",
        "GFS_ST000010", "GFS_ST010040", "GFS_ST040100",
        "GFS_ST100200", "GFS_SM000010", "GFS_SM010040",
        "GFS_SM040100", "GFS_SM100200",
    )
    for hour in (0, 3):
        for name in names:
            relative = f"f{hour:03d}/{name}.f32le"
            path = root / relative
            decoded_lines.append(
                f"{hour}\t{name}\t{path.stat().st_size}\t{_digest(path)}\t{relative}")
    decoded_manifest.write_text("\n".join(decoded_lines) + "\n", encoding="utf-8")
    source_digest = _digest(dummy)
    (root / "gate.tsv").write_text(
        gate_text
        + f"source_sha256\t0:{source_digest},3:{source_digest}\n"
        + f"inventory_sha256\t{_digest(inventory)}\n"
        + f"decoded_manifest_sha256\t{_digest(decoded_manifest)}\n",
        encoding="utf-8")
    snapshots = _load_bridge_snapshots(
        root, datetime(2026, 7, 20), records)
    np.testing.assert_array_equal(
        snapshots[0].longitude, [179.75, 180.0, 180.25, 180.5])
    np.testing.assert_array_equal(
        snapshots[0].fields["PSFC"][0], [0.0, 1.0, 2.0, 3.0])
    assert np.all(np.diff(snapshots[0].longitude) == 0.25)


def _matching_wps() -> str:
    return (
        "&share\n max_dom = 1,\n/\n"
        "&geogrid\n"
        " map_proj = 'lambert',\n"
        " e_we = 251,\n e_sn = 201,\n"
        " dx = 12000.0,\n dy = 12000.0,\n"
        " ref_lat = 39.6848,\n ref_lon = -83.9297,\n"
        " truelat1 = 30.0,\n truelat2 = 60.0,\n"
        " stand_lon = -83.9297,\n/\n"
    )


def test_gfs_geometry_and_static_receipt_fail_closed(tmp_path):
    config = Path(__file__).parents[1] / "configs" / "gfs_wrf_direct_proof.toml"
    exp = load_experiment(config)
    wps = tmp_path / "namelist.wps"
    wps.write_text(_matching_wps(), encoding="utf-8")
    grid = _validate_grid_and_vertical_contract(exp, wps)
    static = tmp_path / "static.npz"
    static.write_bytes(b"domain-specific-static")
    receipt = tmp_path / "static-receipt.json"
    receipt.write_text(json.dumps({
        "schema": "gpuwm-native-static-direct-v1",
        "status": "PASS",
        "geometry": _geometry_contract(grid, exp.root.run),
        "cache": {
            "path": static.name,
            "bytes": static.stat().st_size,
            "sha256": _digest(static),
        },
    }), encoding="utf-8")
    _verify_static_receipt(receipt, static, grid, exp.root.run)

    wps.write_text(_matching_wps().replace(
        "ref_lon = -83.9297", "ref_lon = -83.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="geometry mismatch"):
        _validate_grid_and_vertical_contract(exp, wps)

    static.write_bytes(b"same-shape-wrong-domain")
    with pytest.raises(ValueError, match="does not bind"):
        _verify_static_receipt(receipt, static, grid, exp.root.run)


@pytest.mark.parametrize("nz", [7, 113])
def test_gfs_vertical_contract_accepts_any_source_covered_explicit_eta(
        tmp_path, nz):
    config = Path(__file__).parents[1] / "configs" / "gfs_wrf_direct_proof.toml"
    baseline = load_experiment(config)
    run = dataclasses.replace(baseline.root.run, nz=nz)
    domain = dataclasses.replace(baseline.root, run=run)
    vertical = VerticalConfig(
        eta_levels=tuple(float(value)
                         for value in np.linspace(1.0, 0.0, nz + 1)),
        p_top=10_000.0,
        hybrid_opt=2,
        etac=0.37,
    )
    exp = dataclasses.replace(
        baseline, domains=(domain,), vertical=vertical)
    wps = tmp_path / "namelist.wps"
    wps.write_text(_matching_wps(), encoding="utf-8")
    _validate_grid_and_vertical_contract(exp, wps)


def test_gfs_vertical_contract_refuses_model_top_above_source(tmp_path):
    config = Path(__file__).parents[1] / "configs" / "gfs_wrf_direct_proof.toml"
    baseline = load_experiment(config)
    vertical = dataclasses.replace(baseline.vertical, p_top=5_000.0)
    exp = dataclasses.replace(baseline, vertical=vertical)
    wps = tmp_path / "namelist.wps"
    wps.write_text(_matching_wps(), encoding="utf-8")
    with pytest.raises(ValueError, match="source atmosphere stops"):
        _validate_grid_and_vertical_contract(exp, wps)


class _CoverageGrid:
    def __init__(self, mass, u=None, v=None):
        self._mass = mass
        self._u = mass if u is None else u
        self._v = mass if v is None else v

    def latlon_mass(self):
        return self._mass

    def latlon_u(self):
        return self._u

    def latlon_v(self):
        return self._v


def test_gfs_source_coverage_requires_unclipped_parabolic_and_masked_halos():
    axis = np.arange(30, dtype=np.float64)
    longitude, latitude = np.meshgrid([12.0, 13.0], [12.0, 13.0])
    snapshot = Era5Snapshot(
        valid_time=datetime(2026, 7, 20),
        levels_hpa=np.array([1000.0], dtype=np.float64),
        latitude=axis,
        longitude=axis,
        fields={
            "LANDSEA": np.zeros((30, 30), dtype=np.float64),
            "SKINTEMP": np.full((30, 30), 290.0, dtype=np.float64),
        },
    )
    grid = _CoverageGrid((latitude, longitude))
    lake = np.array([[True, False], [False, False]])
    receipt = _source_coverage_receipt(snapshot, grid, lake)
    # v2 masked coverage: deterministic-stencil box (floor-based [-1, +2]),
    # search reach declared crop-wide, class support reported.
    assert receipt["masked_surface_donor_x"] == [11, 15]
    assert receipt["masked_deterministic_donor_offsets"] == [-1, 2]
    assert "masked_search_radius" not in receipt
    assert receipt["masked_class_support"] == {"land": 0, "water": 900}
    assert receipt["lake_cells"] == 1

    edge_longitude = longitude.copy()
    edge_longitude[:, 0] = 0.25
    edge_grid = _CoverageGrid((latitude, longitude), u=(latitude, edge_longitude))
    with pytest.raises(ValueError, match="parabolic donor halo for u"):
        _source_coverage_receipt(snapshot, edge_grid, lake)

    shallow_longitude = longitude - 11.5
    shallow_grid = _CoverageGrid((latitude, shallow_longitude))
    with pytest.raises(ValueError, match="parabolic donor halo"):
        _source_coverage_receipt(snapshot, shallow_grid, lake)


def test_gfs_lake_coverage_proves_a_global_in_crop_water_donor():
    axis = np.arange(60, dtype=np.float64)
    source_land = np.ones((60, 60), dtype=np.float64)
    source_land[30, 50] = 0.0
    snapshot = Era5Snapshot(
        valid_time=datetime(2026, 7, 20),
        levels_hpa=np.array([1000.0], dtype=np.float64),
        latitude=axis,
        longitude=axis,
        fields={
            "LANDSEA": source_land,
            "SKINTEMP": np.full((60, 60), 290.0, dtype=np.float64),
        },
    )
    point = (np.array([[30.0]]), np.array([[30.0]]))
    receipt = _source_coverage_receipt(
        snapshot, _CoverageGrid(point), np.array([[True]]))
    assert receipt["lake_source_search"] == "global expanding nearest-water (coverage diagnostic only)"
    assert receipt["max_lake_source_search_radius_cells"] == 32.0
    assert receipt["max_lake_source_water_distance_cells"] == 20.0


def test_wizard_suggested_margin_passes_the_lake_donor_proof():
    """Ties the wizard's suggested fetch margin to THIS module's
    donor-coverage function: a crop carrying the documented margin
    shows every lake its nearest source water whenever that lies within
    the GFS_LAKE_DONOR_MARGIN_DEG allowance, and a crop with no water at
    all is counted rather than refused."""
    from woof.fetch import (GFS_SOURCE_RESOLUTION_DEG,
                             gfs_suggested_fetch_margin_deg)

    margin_cells = int(round(gfs_suggested_fetch_margin_deg()
                             / GFS_SOURCE_RESOLUTION_DEG))
    n = 2 * margin_cells + 1
    axis = np.arange(n, dtype=np.float64)
    center = margin_cells
    land = np.ones((n, n), dtype=np.float64)
    # Worst allowed donor: water at exactly the margin allowance -- the
    # crop edge (margin_cells + 1 half-cell steps away) is still
    # provably farther, so the proof passes.
    land[center, n - 1] = 0.0
    snapshot = Era5Snapshot(
        valid_time=datetime(2026, 7, 29),
        levels_hpa=np.array([1000.0], dtype=np.float64),
        latitude=axis,
        longitude=axis,
        fields={
            "LANDSEA": land,
            "SKINTEMP": np.full((n, n), 290.0, dtype=np.float64),
        },
    )
    point = (np.array([[float(center)]]), np.array([[float(center)]]))
    receipt = _source_coverage_receipt(
        snapshot, _CoverageGrid(point), np.array([[True]]))
    assert (receipt["max_lake_source_water_distance_cells"]
            == float(margin_cells))
    assert "lake_cells_nearest_water_past_crop" not in receipt
    assert "lake_cells_without_source_water" not in receipt

    # No water anywhere in the crop: the same function fails closed.
    dry = Era5Snapshot(
        valid_time=datetime(2026, 7, 29),
        levels_hpa=np.array([1000.0], dtype=np.float64),
        latitude=axis,
        longitude=axis,
        fields={
            "LANDSEA": np.ones((n, n), dtype=np.float64),
            "SKINTEMP": np.full((n, n), 290.0, dtype=np.float64),
        },
    )
    receipt = _source_coverage_receipt(
        dry, _CoverageGrid(point), np.array([[True]]))
    assert receipt["lake_cells_without_source_water"] == 1


def test_a_lake_whose_nearer_water_could_lie_past_the_crop_is_counted():
    """Refusing it stopped a preparation over the extent of its download;
    the lake takes the crop's nearest water and the receipt says so."""
    axis = np.arange(20, dtype=np.float64)
    land = np.ones((20, 20), dtype=np.float64)
    land[10, 19] = 0.0              # nine cells east; the west edge is two
    snapshot = Era5Snapshot(
        valid_time=datetime(2026, 9, 27),
        levels_hpa=np.array([1000.0], dtype=np.float64),
        latitude=axis,
        longitude=axis,
        fields={
            "LANDSEA": land,
            "SKINTEMP": np.full((20, 20), 290.0, dtype=np.float64),
        },
    )
    point = (np.array([[10.0]]), np.array([[2.0]]))
    receipt = _source_coverage_receipt(
        snapshot, _CoverageGrid(point), np.array([[True]]))
    assert receipt["lake_cells_nearest_water_past_crop"] == 1
    assert receipt["max_lake_source_water_distance_cells"] == 17.0


def test_a_whole_globe_file_has_no_edge_for_a_lake_to_miss_water_past():
    """A lake a few columns from a whole-globe file's stored cut finds the
    water just across it, and nothing is counted as past the crop."""
    latitude = np.arange(-10.0, 11.0, dtype=np.float64)
    longitude = np.arange(360, dtype=np.float64)
    land = np.ones((21, 360), dtype=np.float64)
    land[10, 358] = 0.0             # 3.5 columns west, across the cut
    land[10, 10] = 0.0              # 8.5 columns east, same side
    snapshot = Era5Snapshot(
        valid_time=datetime(2026, 9, 27),
        levels_hpa=np.array([1000.0], dtype=np.float64),
        latitude=latitude,
        longitude=longitude,
        fields={
            "LANDSEA": land,
            "SKINTEMP": np.full((21, 360), 290.0, dtype=np.float64),
        },
    )
    point = (np.array([[0.0]]), np.array([[1.5]]))
    receipt = _source_coverage_receipt(
        snapshot, _CoverageGrid(point), np.array([[True]]))
    assert receipt["max_lake_source_water_distance_cells"] == 3.5
    assert "lake_cells_nearest_water_past_crop" not in receipt


def test_a_lake_whose_nearer_gfs_water_could_lie_past_the_crop_is_announced(
        capsys):
    from woof import gfs_direct
    from woof.gfs_direct import _announce_lake_source_water

    _announce_lake_source_water(1, lake_cells=3)
    said = capsys.readouterr().err
    assert "of 3 lake cell(s), 1 took the nearest GFS water inside the fetched area" in said
    _announce_lake_source_water(0, lake_cells=3)
    assert capsys.readouterr().err == ""
    # The 2.8.x nearest-water lake override is retired with its workaround:
    # lakes are water targets of the masked chain, whose whole-source search
    # takes that water and whose other_surface rule covers a crop with none.
    assert not hasattr(gfs_direct, "lake_skin_with_source_skin_fallback")
    assert not hasattr(gfs_direct, "interpolate_lake_skin_temperature")


def test_git_identity_degrades_accurately_outside_a_checkout(monkeypatch):
    """An installed wheel is neither a git checkout nor a sealed
    runtime: provenance reports unavailable instead of demanding
    WOOF_NATIVE_DISTRIBUTION_MANIFEST or raising."""
    import subprocess as subprocess_module

    import woof.gfs_direct as gfs_direct

    monkeypatch.delenv("WOOF_NATIVE_DISTRIBUTION_MANIFEST",
                       raising=False)

    def fail(*arguments, **keywords):
        raise subprocess_module.CalledProcessError(128, "git")

    monkeypatch.setattr(gfs_direct.subprocess, "run", fail)
    # An unstamped install: no build-metadata commit either. Without this the
    # test read whatever wheel ran it; the release cut's science stage runs on
    # its own stamped wheel, where the metadata commit is real and available
    # is True (2.9.4 pass 4b, 2026-10-21, gfs-lake-ice-ims 104/105).
    import woof.build_provenance as build_provenance
    monkeypatch.setattr(build_provenance, "package_build_identity", lambda: dict(
        version="unknown", commit="unknown",
        identity_source="installed-package-metadata", receipt_line=""))
    identity = gfs_direct._git_source_identity()
    assert identity["available"] is False
    assert identity["identity_source"] == "unavailable-installed-runtime"


def test_implementation_hashes_no_longer_demand_the_sealed_manifest(
        monkeypatch, tmp_path):
    """Wheel installs miss the repo-only paths (tools/, Rust sources);
    the inventory records them accurately without requiring the sealed
    archive's distribution manifest."""
    import woof.gfs_direct as gfs_direct

    monkeypatch.delenv("WOOF_NATIVE_DISTRIBUTION_MANIFEST",
                       raising=False)
    monkeypatch.setattr(
        gfs_direct, "_IMPLEMENTATION_PATHS",
        ("woof/gfs_direct.py", "tools/absent-in-wheel-installs.rs"))
    hashes = gfs_direct._implementation_sha256()
    assert "woof/gfs_direct.py" in hashes
    assert "distribution/repo_only_paths.json" in hashes
    assert "distribution/manifest.json" not in hashes


def test_output_root_parent_is_created_instead_of_tracebacking(tmp_path):
    """PP-10: an absent --output-root parent was a bare traceback.

    The bridge stages its scratch space in the PARENT of --output-root so
    the result can be renamed into place on one filesystem.  An absent
    parent surfaced as `FileNotFoundError: .../gpuwm-gfs-bridge-4g0c7np7`
    about 40 s into the work -- after --dry-run had passed cleanly -- in
    a product that is otherwise exemplary about naming the remedy.
    """
    from woof.gfs_direct import _prepare_output_root_parent

    output_root = tmp_path / "out" / "miami-init"
    assert not output_root.parent.exists()
    parent = _prepare_output_root_parent(output_root)
    assert parent == output_root.parent
    assert parent.is_dir()
    # The output root itself stays create-only: only its parent is made.
    assert not output_root.exists()
    # Idempotent.
    assert _prepare_output_root_parent(output_root) == parent

    # A parent path that is a FILE is a refusal with a stated remedy,
    # not a traceback from tempfile.
    blocked = tmp_path / "afile" / "init"
    blocked.parent.write_text("not a directory", encoding="utf-8")
    with pytest.raises(NotADirectoryError, match="parent"):
        _prepare_output_root_parent(blocked)


def test_the_front_door_prints_the_forecast_command_it_already_knows(tmp_path):
    """B-2: the runner needed six hashes the user extracted by hand.

    `woof fetch --author-front-door-manifest` prints the complete
    rw-wps line with its digest filled in and is the most-praised thing
    in both pilots.  The front door printed nothing at all -- it dumped
    42 KB of proof.json and stopped -- so every user grepped
    `content_sha256` out of the JSON themselves.  Every value below was
    already in this process.
    """
    import json
    from woof.gfs_direct import prepared_forecast_next_command

    root = tmp_path / "miami-init"
    root.mkdir()
    proof = {
        "input_manifest_sha256": "a" * 64,
        "prepared_cache": {"content_sha256": "b" * 64},
    }
    (root / "proof.json").write_text(json.dumps(proof), encoding="utf-8")
    proof_digest = hashlib.sha256(
        (root / "proof.json").read_bytes()).hexdigest()

    config = _wizard_single_domain_config(tmp_path)
    namelist = tmp_path / "x.namelist.wps"
    namelist.write_text("&share\n/\n", encoding="utf-8")

    lines = prepared_forecast_next_command(
        proof, output_root=root, experiment_config=config,
        wps_namelist=namelist)
    text = "\n".join(lines)
    # The module form, not a script path under tools/: what this line
    # prints has to be runnable by a reader who pip-installed the wheel
    # and has no checkout.  (tools/ still carries a delegating entry
    # point for the spelling older transcripts use.)
    assert "python -m woof.prepared_single_domain_forecast" in text
    assert "tools/prepared_single_domain_forecast.py" not in text
    assert f"--proof-sha256 {proof_digest}" in text
    assert f"--source-manifest-sha256 {'a' * 64}" in text
    assert f"--prepared-content-sha256 {'b' * 64}" in text
    assert "--run-seconds" not in text.split("(")[0]  # no longer required
    assert "default to the hash-bound experiment" in text
    # The two values v1.0.0 left the user to work out are filled in.
    assert f"--physics-profile {WSM6_PROFILE_ID}" in text
    outdir = root.parent / f"{root.name}-forecast"
    # POSIX display form: the certified runtime is Linux/CUDA, and
    # forward slashes are accepted by every path API on Windows too.
    assert f"--outdir {_printed(outdir)}" in text
    _assert_pasteable(lines)
    _assert_runner_accepts_printed_outdir(text, prepared_root=root)

    # A multi-domain hierarchy proof has no prepared-cache identity, so
    # it names the OTHER runner -- and prints that runner's WHOLE
    # command rather than a fragment: it binds the experiment config by
    # digest too, and this process can read it.
    hierarchy = prepared_forecast_next_command(
        {"schema": "gpuwm-gfs-native-hierarchy-proof-v1"},
        output_root=root, experiment_config=config, wps_namelist=namelist)
    hierarchy_text = "\n".join(hierarchy)
    assert "python -m woof.prepared_domain_tree_forecast" in hierarchy_text
    assert f"--preparation-receipt-sha256 {proof_digest}" in hierarchy_text
    assert f"--experiment-config-sha256 {_digest(config)}" in hierarchy_text
    assert f"--outdir {_printed(outdir)}" in hierarchy_text
    _assert_pasteable(hierarchy)
    _assert_runner_accepts_printed_outdir(
        hierarchy_text, prepared_root=root, config=config)


def _printed(value) -> str:
    """A path as the next-command printer renders it."""

    import shlex

    return shlex.quote(str(value).replace("\\", "/"))


def _assert_runner_accepts_printed_outdir(text, *, prepared_root,
                                          config=None):
    """The printed --outdir must survive the runner's own guard.

    ``_assert_pasteable`` is lexical: it rejects placeholders but never
    asks whether the command would be REFUSED.  That is exactly how the
    test host 8 finding shipped green -- the front door suggested
    ``<prepared_root>/forecast`` while both runners declare
    ``--prepared-root`` a protected input and refuse any --outdir that
    overlaps it, so the suggestion produced a traceback.
    """

    from woof import prepared_domain_tree_forecast as module

    printed = [line.split("--outdir ", 1)[1].strip()
               for line in text.splitlines() if "--outdir " in line]
    assert printed, "no --outdir was printed"
    protected = ((Path(prepared_root), Path(config)) if config is not None
                 else (Path(prepared_root),))
    for candidate in printed:
        # The guard itself, not a re-implementation of it.  It creates
        # the directory on success, so release what we just claimed --
        # both printed commands name the same outdir, and the second
        # claim would otherwise fail for the wrong reason.
        claimed = module.claim_output_directory(
            Path(candidate), protected_roots=protected)
        claimed.rmdir()


#: A bare ALL-CAPS token is how v1.0.0 spelled "you work this one out"
#: (`--outdir OUTPUT_DIR`, `--geog-root WPS_GEOG_DIR`).
_PLACEHOLDER_WORD = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")


def _assert_pasteable(lines) -> None:
    """No printed command line may carry an unresolved placeholder.

    Deliberately mechanical: a command line is one that starts a shell
    invocation or continues one, and on such a line an angle bracket or
    a bare ALL-CAPS token means the user must edit before pasting --
    which is the whole failure.  Parenthesised prose is exempt; it is
    not a command and never claimed to be.
    """

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("("):
            continue
        looks_like_command = (
            stripped.startswith(("python ", "rw-wps ", "woof ", "cargo ",
                                 "git ", "bash ", "--"))
            or stripped.endswith("\\"))
        if not looks_like_command:
            continue
        assert "<" not in stripped and ">" not in stripped, (
            f"unresolved placeholder in a printed command: {line!r}")
        for token in stripped.rstrip("\\").split():
            assert not _PLACEHOLDER_WORD.match(token), (
                f"unresolved ALL-CAPS placeholder {token!r} in: {line!r}")


def _wizard_single_domain_config(tmp_path):
    """A config the prepared-forecast runner would actually accept.

    Emitted by the wizard rather than hand-written, because the point of
    resolving the profile is that both sides read the same table.
    """

    from woof.cli import main as cli_main

    out = tmp_path / "wizard" / "case.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    assert cli_main(["domain", "--point=39.7,-96.6", "--card", "24gb",
                     "--ladder", "12", "--source", "gfs",
                     "--physics-profile", WSM6_PROFILE_ID,
                     "--cycle", "2026-07-29T18", "--out", str(out)]) == 0
    return out


def test_a_config_bound_to_no_shipped_profile_prints_a_runnable_command(
        tmp_path):
    """Converted (owner ruling 2026-07-31): there is no gap to apologise
    for -- the runner executes this suite as written, so the printed
    command omits --physics-profile and one sentence states the
    verification status."""

    import json
    from woof.gfs_direct import prepared_forecast_next_command

    root = tmp_path / "init"
    root.mkdir()
    proof = {"input_manifest_sha256": "a" * 64,
             "prepared_cache": {"content_sha256": "b" * 64}}
    (root / "proof.json").write_text(json.dumps(proof), encoding="utf-8")

    # A shipped proof descriptor with exactly one selector taken off
    # the profile it otherwise resolves to: single-domain and valid,
    # matching none of the shipped profiles.  Derived from the shipped
    # file rather than hand-written, so the only difference from a
    # config that DOES resolve is the one line this test is about --
    # and `radt` is the specific line.  The GLW fix moved the shipped
    # descriptor from Dudhia-only at radt 1.0 to legacy RRTMG on both
    # streams at radt 12.0 (it shipped a standing nocturnal bypass and a
    # frozen 300 W m-2 downward longwave), so the perturbation moves off
    # 12.0 now instead of off 1.0.
    descriptor = tmp_path / "off-profile.toml"
    shipped = Path("configs/gfs_wrf_direct_proof.toml").read_text(
        encoding="utf-8")
    assert shipped.count("radt = 12.0") == 1
    descriptor.write_text(shipped.replace("radt = 12.0", "radt = 9.0"),
                          encoding="utf-8")

    lines = prepared_forecast_next_command(
        proof, output_root=root,
        experiment_config=descriptor,
        wps_namelist=Path("configs/gfs_wrf_direct_proof.namelist.wps"))
    text = "\n".join(lines)
    assert "python -m woof.prepared_single_domain_forecast" in text
    assert "--physics-profile" not in text
    assert "supported, not yet WRF-verified" in text
    assert "matches none of the profiles" not in text
    _assert_pasteable(lines)


def _wizard_multi_domain_config(tmp_path, profile=None):
    """The wizard's own DEFAULT emission, with a nest: max_dom = 2.

    Emitted rather than hand-written for the same reason the
    single-domain helper is: the claim under test is that the config the
    product tells a user to make passes the door the product tells them
    to use, and a hand-written stand-in can only test a config nobody
    was told to write.  With no ``--physics-profile`` this is the
    product default suite -- Thompson MP8, Kain-Fritsch on d01 and none
    on d02, RTE+RRTMGP -- which is deliberately NOT in the prepared
    single-domain runner's whitelist.
    """

    from woof.cli import main as cli_main

    name = "default" if profile is None else profile
    out = tmp_path / "wizard-nested" / f"{name}.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    command = ["domain", "--point=39.7,-96.6", "--card", "24gb",
               "--ladder", "12-3", "--source", "gfs",
               "--cycle", "2026-07-29T18", "--out", str(out)]
    if profile is not None:
        command.extend(("--physics-profile", profile))
    assert cli_main(command) == 0
    return out


def test_the_wizard_default_multi_domain_config_passes_its_own_front_door(
        tmp_path):
    """F21, the 1.1.0 regression: max_dom = 2 met a single-domain gate.

    v1.1's any-combo work made `prepare_gfs_wrf` call
    `validate_single_domain_physics_profile` for every configuration it
    was given.  That whitelist belongs to the prepared SINGLE-domain
    forecast runner -- the wizard says so in the note it prints beside
    the file -- and the product's own default suite is deliberately not
    in it, so the config the wizard emits BY DEFAULT could not pass the
    GFS front door at all.  A two-domain config that prepared cleanly on
    1.0.1 died with a raw traceback on 1.1.0.
    """

    from woof.gfs_direct import front_door_physics_selection

    exp = load_experiment(_wizard_multi_domain_config(tmp_path))
    assert len(exp.domains) == 2

    # This is exactly what 1.1.0 did to this config, kept here as the
    # falsifier: the whitelist itself is right, and refusing the default
    # suite is what it is FOR.  The defect was asking it at all.
    with pytest.raises(ValueError, match="selected physics differs"):
        from woof.physics_compat import (
            validate_single_domain_physics_profile,
        )
        validate_single_domain_physics_profile(
            WSM6_PROFILE_ID, config=exp.root.run)

    receipt = front_door_physics_selection(exp)
    assert receipt["profile"] is None
    # Every domain of the tree is recorded, not just the root: a child
    # selects its own cumulus and radiation cadence.
    assert sorted(receipt["domains"]) == ["1", "2"]
    assert receipt["domains"]["1"]["selectors"]["cu_physics"] == 1
    assert receipt["domains"]["2"]["selectors"]["cu_physics"] == 0
    # 2026-08-06: the wizard's default is the certified Morrison profile
    # (nocturnal-radiation directive); the unnamed governance path is
    # unchanged and still records every domain.
    assert (receipt["domains"]["1"]["components"]["microphysics"]
            == "morrison-mp10")
    assert receipt["domains"]["1"]["components"]["cumulus"] == "kain-fritsch"
    assert receipt["domains"]["2"]["components"]["cumulus"] == "off"


def test_the_shipped_two_domain_proof_config_still_passes_the_front_door():
    """The 1.0.1-shape max_dom = 2 config prepares through the gate.

    `configs/gfs_wrf_hierarchy_proof.toml` is the committed two-domain
    descriptor the hierarchy route has always used, and the door admits
    it unnamed, with a per-domain receipt, rather than against a profile
    whitelist.

    Through 1.8.7 this file also carried the sharper half of the case: it
    selected the LEGACY AGGREGATE radiation spelling with radiation OFF
    (`ra_lw_physics`/`ra_sw_physics` at -1, `ra_physics` 0), which the
    registry then had no option for, so its receipt recorded a blocker
    and the run proceeded anyway.  That spelling is gone from the shipped
    file -- radiation off under Noah meant Noah read a fabricated
    300 W m-2 downward longwave for the whole forecast -- and it moved to
    the derived config below.  Since af5332967 the capability door reads
    the two radiation spellings the way the engine does, so the derived
    config resolves to exactly the receipt its split spelling gets, with
    no blocker.  The blocker half is still exercised, on a selector the
    registry has no option for: recording is not permission, the receipt
    names what it can and reports the blocker for what it cannot, and
    the door itself refuses neither.
    """

    import dataclasses

    from woof.gfs_direct import front_door_physics_selection

    config = (Path(__file__).parents[1] / "configs"
              / "gfs_wrf_hierarchy_proof.toml")
    exp = load_experiment(config)
    assert len(exp.domains) == 2
    receipt = front_door_physics_selection(exp)
    assert receipt["schema"] == (
        "gpuwm-front-door-physics-selection-multi-domain-v1")
    assert receipt["domains"]["1"]["selectors"]["mp_physics"] == 6
    # Both radiation streams now resolve to a registry component, so the
    # receipt records components and no blocker.
    assert receipt["domains"]["1"]["components"] is not None
    assert receipt["domains"]["1"]["registry_blocker"] is None
    assert receipt["domains"]["1"]["selectors"]["ra_lw_physics"] == 4

    def _respelled(**selectors):
        return dataclasses.replace(exp, domains=tuple(
            dataclasses.replace(
                domain, run=dataclasses.replace(domain.run, **selectors))
            for domain in exp.domains))

    # The legacy aggregate with radiation off, preserved.  af5332967
    # retired the (-1, -1) sentinel option it used to land on, whose
    # ra_physics = 4 requirement was the blocker recorded here, and
    # resolves the pair through woof.config.radiation_scheme_ids as every
    # run-path consumer does: radiation off, the receipt of the split
    # spelling, no blocker.
    legacy_receipt = front_door_physics_selection(
        _respelled(ra_physics=0, ra_lw_physics=-1, ra_sw_physics=-1))
    split_receipt = front_door_physics_selection(
        _respelled(ra_physics=0, ra_lw_physics=0, ra_sw_physics=0))
    for grid_id in ("1", "2"):
        legacy_domain = legacy_receipt["domains"][grid_id]
        assert legacy_domain["registry_blocker"] is None
        assert legacy_domain["components"]["radiation"] == "off"
        assert legacy_domain["components"] == (
            split_receipt["domains"][grid_id]["components"])
        assert legacy_domain["governance"] == (
            split_receipt["domains"][grid_id]["governance"])

    # A selector the registry has no option for is recorded as a blocker,
    # not raised.  The loader refuses mp_physics = 32 (the full HUJI
    # spectral bin scheme, no woof package; 95 and 7 joined with their
    # ports) before any door sees it (the registry resolves every selector
    # the loader admits), so it is handed to the door directly here.
    blocked = front_door_physics_selection(_respelled(mp_physics=32))
    assert blocked["domains"]["1"]["components"] is None
    assert "mp_physics" in blocked["domains"]["1"]["registry_blocker"]


def test_the_single_domain_default_suite_passes_and_a_named_gate_binds(
        tmp_path):
    """POSITIVE CONTROL, converted from "one domain still meets the
    profile whitelist" (owner ruling 2026-07-31).

    The single-domain half of the whitelist is gone: the wizard's
    default emission -- the exact config the v1.1.0 field regression
    could never pass -- is now admitted unnamed, governed the way the
    tree route has always been governed, with a per-domain receipt.
    What did NOT widen: naming a profile still gates switch for switch,
    and a config bound to one when NAMED still yields the profile
    receipt.
    """

    from woof.gfs_direct import front_door_physics_selection

    default_suite = load_experiment(_wizard_single_domain_config_default(
        tmp_path))
    assert len(default_suite.domains) == 1
    receipt = front_door_physics_selection(default_suite)
    assert receipt["schema"] == (
        "gpuwm-front-door-physics-selection-multi-domain-v1")
    assert receipt["profile"] is None
    assert sorted(receipt["domains"]) == ["1"]
    # 2026-08-06: the default emission is the certified Morrison profile.
    assert receipt["domains"]["1"]["components"]["microphysics"] \
        == "morrison-mp10"
    assert receipt["domains"]["1"]["governance"]["state"] \
        == "registry-reachable"

    # The gate a caller asks for is not dropped: the default suite
    # named against a shipped profile is still refused with the drift.
    with pytest.raises(ValueError, match="selected physics differs"):
        front_door_physics_selection(
            default_suite, physics_profile=WSM6_PROFILE_ID)

    bound = load_experiment(_wizard_single_domain_config(tmp_path))
    assert len(bound.domains) == 1
    named = front_door_physics_selection(
        bound, physics_profile=WSM6_PROFILE_ID)
    assert named["schema"] == "gpuwm-front-door-physics-selection-v1"
    assert named["profile"] == WSM6_PROFILE_ID


def _noahmp_experiment(exp, acknowledgements=()):
    run = dataclasses.replace(
        exp.root.run,
        **single_domain_runtime_switches(NOAHMP_PROFILE_ID),
    )
    root = dataclasses.replace(exp.root, run=run)
    return dataclasses.replace(
        exp, domains=(root,), acknowledgements=tuple(acknowledgements))


@pytest.mark.parametrize(
    ("flag", "toml", "sources"),
    (
        (("noahmp-host-column-throughput-v1",), (), ["--ack"]),
        ((), ("noahmp-host-column-throughput-v1",),
         ["[experiment].acknowledgements"]),
        (("noahmp-host-column-throughput-v1",),
         ("noahmp-host-column-throughput-v1",),
         ["--ack", "[experiment].acknowledgements"]),
    ),
)
def test_ack_delivery_unions_flag_and_toml_with_source_provenance(
        tmp_path, flag, toml, sources):
    from woof.gfs_direct import front_door_physics_selection

    exp = _noahmp_experiment(
        load_experiment(_wizard_single_domain_config(tmp_path)), toml)
    receipt = front_door_physics_selection(
        exp, physics_profile=NOAHMP_PROFILE_ID,
        expert_acknowledgements=flag)

    assert receipt["acknowledgements"] == [
        "noahmp-host-column-throughput-v1"]
    assert receipt["acknowledgement_provenance"] == {
        "noahmp-host-column-throughput-v1": sources}


@pytest.mark.parametrize("delivery", ("flag", "toml"))
def test_unrecognized_ack_id_keeps_expert_advisory_without_blocking_physics(
        tmp_path, monkeypatch, delivery):
    from woof.gfs_direct import front_door_physics_selection
    from woof import physics_compat

    toml = ("wrong-id-v2",) if delivery == "toml" else ()
    flag = ("wrong-id-v2",) if delivery == "flag" else ()
    exp = _noahmp_experiment(
        load_experiment(_wizard_single_domain_config(tmp_path)), toml)
    warnings = []
    monkeypatch.setattr(physics_compat, "warn", lambda message, **kwargs: warnings.append(message))
    receipt = front_door_physics_selection(
        exp, physics_profile=NOAHMP_PROFILE_ID, expert_acknowledgements=flag)
    assert receipt["profile"] == NOAHMP_PROFILE_ID
    assert receipt["governance"]["acknowledged"] is False
    assert receipt["governance"]["required_acknowledgements"] == [
        "noahmp-host-column-throughput-v1"]
    advisory = [message for message in warnings if "evidence advisory unacknowledged" in message]
    other_warnings = [message for message in warnings if message not in advisory]
    assert len(advisory) == 1
    assert "running with its evidence advisory unacknowledged" in advisory[0]
    assert ("--ack noahmp-host-column-throughput-v1 or "
            'acknowledgements = ["noahmp-host-column-throughput-v1"]') in advisory[0]
    warnings.clear()
    acknowledged = front_door_physics_selection(
        exp, physics_profile=NOAHMP_PROFILE_ID,
        expert_acknowledgements=("noahmp-host-column-throughput-v1",))
    # Other measured throughput notices remain true regardless of the ack.
    assert warnings == other_warnings
    assert acknowledged["governance"]["acknowledged"] is True
    assert acknowledged["selectors"] == receipt["selectors"]
    assert acknowledged["resolved"] == receipt["resolved"]


@pytest.mark.parametrize("delivery", ("flag", "toml", "both"))
def test_registry_reachable_tuple_is_unaffected_by_ack_delivery(
        tmp_path, delivery):
    from woof.gfs_direct import front_door_physics_selection

    flag = ("irrelevant-v1",) if delivery in ("flag", "both") else ()
    toml = ("irrelevant-v2",) if delivery in ("toml", "both") else ()
    exp = dataclasses.replace(
        load_experiment(_wizard_single_domain_config(tmp_path)),
        acknowledgements=toml)
    receipt = front_door_physics_selection(
        exp, expert_acknowledgements=flag)
    # Unnamed selection carries no inferred profile any more; the tuple
    # is registry-reachable and irrelevant acknowledgements change
    # nothing about its admission.
    assert receipt["profile"] is None
    assert receipt["domains"]["1"]["governance"]["state"] \
        == "registry-reachable"
    assert receipt["domains"]["1"]["governance"]["acknowledged"] is True


def test_unnamed_prepare_and_export_speak_the_same_selection(tmp_path):
    """The seam the 2026-07-31 field run died on, now watched.

    prepare_gfs_wrf hands export_prepared_wrf the profileless contract
    and then requires the exporter's v3 manifest to carry a physics
    receipt byte-equal to the front door's own selection.  The exporter
    recomputes from the CACHE-IDENTITY spelling of the config (asdict,
    JSON-stable) rather than trusting the caller, so this test pins
    that both spellings of the config produce one receipt -- and that
    one drifted selector breaks the equality the RuntimeError gate
    checks.
    """

    from woof.gfs_direct import front_door_physics_selection
    from woof.ingest.prepared_cache import prepared_domain_config_identity
    from woof.physics_compat import (
        MULTI_DOMAIN_SELECTION_SCHEMA,
        single_domain_physics_selection,
    )

    exp = load_experiment(_wizard_single_domain_config_default(tmp_path))
    front_door = front_door_physics_selection(exp)
    assert front_door["schema"] == MULTI_DOMAIN_SELECTION_SCHEMA
    assert front_door["profile"] is None

    identity_cfg = prepared_domain_config_identity(exp.root)["run"]
    exporter_side = single_domain_physics_selection(
        identity_cfg,
        expert_acknowledgements=tuple(front_door["acknowledgements"]),
        acknowledgement_provenance=front_door[
            "acknowledgement_provenance"])

    assert exporter_side == front_door
    # JSON round trip (the proof is written and re-read as JSON).
    canonical = json.dumps(front_door, sort_keys=True)
    assert json.dumps(exporter_side, sort_keys=True) == canonical
    assert json.loads(canonical) == json.loads(
        json.dumps(exporter_side, sort_keys=True))

    # Negative control: one drifted selector and the equality the
    # prepare gate enforces no longer holds.
    drifted_cfg = dict(identity_cfg)
    drifted_cfg["cu_physics"] = 0 if identity_cfg["cu_physics"] else 1
    assert single_domain_physics_selection(drifted_cfg) != front_door


def test_an_explicit_profile_is_still_enforced_on_a_domain_tree(tmp_path):
    """A gate the caller ASKED for is never dropped by the scoping fix.

    `--physics-profile` on a multi-domain run is how a user says "bind
    this tree to a shipped suite"; test host 8 verified that route works on
    1.1.0 and it must keep working.  It binds the ROOT -- children carry
    their own cumulus and radiation cadence by design, so demanding the
    profile of every domain would refuse the wizard's own nested
    emission of that same profile.
    """

    from woof.gfs_direct import front_door_physics_selection

    exp = load_experiment(_wizard_multi_domain_config(
        tmp_path, profile=MORRISON_PROFILE_ID))
    assert len(exp.domains) == 2
    receipt = front_door_physics_selection(
        exp, physics_profile=MORRISON_PROFILE_ID)
    assert receipt["profile"] == MORRISON_PROFILE_ID

    with pytest.raises(ValueError, match="selected physics differs"):
        front_door_physics_selection(exp, physics_profile=WSM6_PROFILE_ID)


def test_a_front_door_refusal_reaches_the_user_as_a_sentence(tmp_path,
                                                             capsys):
    """Not a traceback.  F21's third consequence, and F19's whole point.

    The gate is entered through `python -m woof.gfs_direct`, whose
    return code `rw-wps` passes straight through, so what the user sees
    is whatever this `main` prints.  On 1.1.0 that was a stack trace.
    """

    from woof.gfs_direct import main as gfs_main

    config = _wizard_single_domain_config_default(tmp_path)
    capsys.readouterr()  # the wizard's own report is not under test
    missing = tmp_path / "series.tsv"
    missing.write_text("0\tf000.grib2\n3\tf003.grib2\n", encoding="utf-8")
    code = gfs_main([
        "--series", str(missing),
        "--cycle", "2026-07-29_18:00:00",
        "--bridge", str(tmp_path / "absent-bridge"),
        "--wps-namelist", str(tmp_path / "absent.namelist.wps"),
        "--experiment-config", str(config),
        "--input-manifest", str(tmp_path / "absent-manifest.json"),
        "--input-manifest-sha256", "0" * 64,
        "--output-root", str(tmp_path / "out"),
    ])
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(lines) == 1, captured.err
    assert lines[0].startswith("rw-wps --source gfs: ")
    assert "Traceback" not in captured.err


def _wizard_single_domain_config_default(tmp_path):
    """One domain, product default suite: no shipped profile matches it."""

    from woof.cli import main as cli_main

    out = tmp_path / "wizard-default" / "case.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    assert cli_main(["domain", "--point=39.7,-96.6", "--card", "24gb",
                     "--ladder", "12", "--source", "gfs",
                     "--cycle", "2026-07-29T18", "--out", str(out)]) == 0
    return out


def test_the_front_door_manifest_line_is_pasteable_too(tmp_path):
    """`woof fetch --author-front-door-manifest` prints a command too.

    It carried `--output-root OUTPUT_DIR` and `--geog-root
    WPS_GEOG_DIR`: the same defect, on the line both pilots praised
    most.
    """

    from woof import fetch

    out = tmp_path / "gfs"
    out.mkdir()
    (out / "gfs-series.tsv").write_text("0\tf000.grib2\n3\tf003.grib2\n", encoding="utf-8")
    (out / "f000.grib2").write_bytes(b"GRIB")
    (out / "f003.grib2").write_bytes(b"GRIB")
    (out / fetch.FETCH_MANIFEST_NAME).write_text(json.dumps({
        "schema": fetch.FETCH_MANIFEST_SCHEMA,
        "source": "gfs", "cycle": "2026-07-29T18:00:00Z",
        "forecast_hours": [0, 3],
        "files": [{"name": "f000.grib2", "role": "gfs-subset",
                   "forecast_hour": 0},
                  {"name": "f003.grib2", "role": "gfs-subset",
                   "forecast_hour": 3}],
    }), encoding="utf-8")
    bridge = tmp_path / "gfs_grib2_bridge"
    bridge.write_bytes(b"stand-in")
    namelist = tmp_path / "x.namelist.wps"
    namelist.write_text("&share\n/\n", encoding="utf-8")
    config = tmp_path / "x.toml"
    config.write_text("[experiment]\n", encoding="utf-8")

    said: list[str] = []
    fetch.author_gfs_front_door_manifest(
        out=out, bridge=bridge, wps_namelist=namelist,
        experiment_config=config, progress=said.append)
    printed = [line for text in said for line in text.splitlines()]
    assert any("rw-wps --source gfs" in line for line in printed)
    _assert_pasteable(printed)


def test_a_bound_refusal_reaches_python_with_the_reason_and_a_remedy():
    """The reported blocker's shape, as a user would now receive it.

    v1.1.1 raised ``GFS Rust bridge failed: <stderr>`` and stopped.  The
    stderr is true and unreadable on its own: a soil-moisture value of
    1.05 means nothing to someone who did not write the bridge, and the
    obvious reading -- that the range is too tight -- is the one thing
    that must not be acted on.
    """

    stderr = (
        "GFS_SM010040 value 1.05 outside [0,1] by 0.05 "
        "(quantization tolerance 0.000001); a bound-kissing value is "
        "clamped, this one is not\n")
    message = decode_failure_message("GFS Rust bridge", stderr)
    lines = message.splitlines()

    # The bridge's own words survive verbatim on the first line.
    assert lines[0] == (
        "GFS Rust bridge failed: " + stderr.strip())
    # And the remedy says what the number means, that bound-kissing is
    # already handled, and what to actually do.
    assert "remedy:" in message
    assert "GFS_SM010040 decoded 1.05" in message
    assert "clamps those" in message
    assert "Re-fetch the cycle and re-run" in message
    # Every remedy line is a comment or a command, so the block survives
    # being pasted whole.
    for line in lines[2:]:
        assert line.strip().startswith("#"), line
    _assert_pasteable(lines)


def test_a_failure_that_is_not_a_bound_refusal_gains_no_quantization_prose():
    """A missing file is not a packing story and must not be told as one."""

    message = decode_failure_message(
        "GFS Rust bridge", "series line 1 references missing \"f000.grb2\"\n")
    assert message == (
        "GFS Rust bridge failed: series line 1 references missing "
        "\"f000.grb2\"")
    assert "quantization" not in message
    assert "remedy" not in message


def test_a_silent_decoder_failure_still_names_itself():
    assert decode_failure_message("GFS Rust bridge", "   ") == (
        "GFS Rust bridge failed")


def test_the_printed_command_survives_a_path_with_a_space_in_it():
    """Not lexical this time: the line is shell-parsed back to argv.

    The existing checks reject angle brackets and ALL-CAPS placeholders,
    which is why a real defect walked past them -- a perfectly valid
    `--outdir` containing a space was printed bare and split into two
    arguments the moment the command was pasted.
    """

    import shlex

    from woof.gfs_direct import _arg

    for awkward in ("plain/path", "a dir/with spaces", "quote's/dir",
                    "dollar$dir/x"):
        rendered = _arg(awkward)
        assert shlex.split(f"--outdir {rendered}") == ["--outdir", awkward]
    # An ordinary path is left alone, so this is invisible except where
    # it matters.
    assert _arg("relative/output") == "relative/output"


@pytest.mark.parametrize("proof", [
    # The single-domain branch (gfs_direct.py ~1233-1242): prints
    # --prepared-root, --experiment-config, --wps-namelist, --outdir.
    {"input_manifest_sha256": "a" * 64,
     "prepared_cache": {"content_sha256": "b" * 64}},
    # The multi-domain hierarchy branch (~1207-1216): a proof with no
    # single prepared-cache identity, printing the OTHER runner's
    # --prepared-root, --experiment-config, --outdir.  Both branches
    # interpolate paths, so both must survive a path with a space.
    {"schema": "gpuwm-gfs-native-hierarchy-proof-v1"},
], ids=["single-domain", "hierarchy"])
def test_every_printed_path_in_the_forecast_command_is_shell_parseable(
        tmp_path, proof):
    import shlex
    import shutil

    from woof.gfs_direct import prepared_forecast_next_command

    root = tmp_path / "a case with spaces" / "prepared"
    root.mkdir(parents=True)
    (root / "proof.json").write_text(json.dumps(proof), encoding="utf-8")
    # The experiment config and namelist live in the spaced directory too,
    # so their printed paths -- not only --outdir -- carry a space and
    # must be quoted to survive the parse.
    config = shutil.copy2(
        _wizard_single_domain_config(tmp_path), root / "case config.toml")
    namelist = root / "namelist with spaces.wps"
    namelist.write_text("&share\n/\n", encoding="utf-8")

    lines = prepared_forecast_next_command(
        proof, output_root=root, experiment_config=config,
        wps_namelist=namelist)
    command = " ".join(
        line.strip().rstrip("\\") for line in lines
        if line.strip().startswith(("python ", "--", "      --")))
    # The line shell-parses at all only because every path with a space
    # in it is quoted; an unquoted one would split here.
    argv = shlex.split(command)
    outdir = argv[argv.index("--outdir") + 1]
    assert " " in outdir, outdir
    assert Path(outdir).name == f"{root.name}-forecast"
    # Every value that IS a path this process wrote to a spaced directory
    # must come back through the parse whole, not split.
    prepared_root = argv[argv.index("--prepared-root") + 1]
    assert prepared_root == str(root).replace("\\", "/"), prepared_root
    config_arg = argv[argv.index("--experiment-config") + 1]
    assert " " in config_arg and Path(config_arg).is_file(), config_arg


# ---------------------------------------------------------------------------
# F2: the export request reaches the adapter from both front doors
# ---------------------------------------------------------------------------

def _gfs_front_door_args(extra: list[str]):
    from woof import source_cli

    return source_cli._parser().parse_args([
        "--source", "gfs",
        "--gfs-series", "series.tsv",
        "--cycle", "2026-07-29_06:00:00",
        "--bridge", "bridge.exe",
        "--wps-namelist", "namelist.wps",
        "--experiment-config", "experiment.toml",
        "--source-manifest", "manifest.json",
        "--source-manifest-sha256", "a" * 64,
        "--output-root", "prep",
        "--geog-root", "geog",
        *extra,
    ])


def test_the_gfs_front_door_forwards_the_export_request_only_when_declined():
    """`rw-wps` passes the flag through; the default prints nothing extra."""

    from woof import source_cli

    default = source_cli._gfs_command(_gfs_front_door_args([]))
    declined = source_cli._gfs_command(
        _gfs_front_door_args(["--no-stock-wrf-export"]))

    assert "--no-stock-wrf-export" not in default
    assert "--no-stock-wrf-export" in declined
    assert source_cli._required_gfs_args(
        _gfs_front_door_args(["--no-stock-wrf-export"])) == []
    # The default has to be FALSY, not True-meaning-"export": every
    # inventory action on this parser refuses to be combined with any
    # namespace entry that is not None/False, so a flag defaulting to True
    # breaks `--validate-hrrr-domain` and its siblings for every caller.
    assert _gfs_front_door_args([]).no_stock_wrf_export is False
    assert source_cli._active_action_arguments(
        _gfs_front_door_args([]),
        allowed=frozenset({"no_stock_wrf_export"})) == \
        source_cli._active_action_arguments(
            _gfs_front_door_args([]), allowed=frozenset())


def test_export_intent_reaches_mapped_preparation_and_stays_route_scoped(monkeypatch):
    from woof import source_cli, mapped_direct

    args = _gfs_front_door_args(["--no-stock-wrf-export"])
    for validator in (source_cli._required_era5_args,
                      source_cli._required_twentycr_args):
        errors = validator(args)
        assert any("--no-stock-wrf-export" in error for error in errors), validator.__name__
    # U-E6: both HRRR preparers take the flag (woof prep relays it), so the HRRR
    # route no longer refuses it.
    assert not any("--no-stock-wrf-export" in error
                   for error in source_cli._required_hrrr_args(args))

    class ReachedPreparation(Exception):
        pass
    observed = []
    def prepare(**kwargs):
        observed.append(kwargs["stock_wrf_export"])
        raise ReachedPreparation
    monkeypatch.setattr(mapped_direct, "load_mapping", lambda path: {"format": "netcdf"})
    monkeypatch.setattr(mapped_direct, "prepare_mapped_wrf", prepare)
    base = ["--source", "mapped", "--source-format", "netcdf",
        "--mapping", "mapping.toml", "--composition", "composition.json",
        "--input", "atmosphere.nc", "--supplement", "terrain=terrain.nc",
        "--provenance", "terrain=terrain.json", "--source-manifest", "manifest.json",
        "--source-manifest-sha256", "a" * 64, "--wps-namelist", "namelist.wps",
        "--experiment-config", "case.toml", "--output-root", "prep", "--geog-root", "geog"]
    for extra, expected in ((["--no-stock-wrf-export"], "off"),
                            (["--stock-wrf-export", "required"], "required")):
        args = source_cli._parser().parse_args(base + extra)
        assert source_cli._required_mapped_args(args) == []
        command = source_cli._mapped_command(args)
        assert command[command.index("--stock-wrf-export") + 1] == expected
        with pytest.raises(ReachedPreparation):
            mapped_direct.main(command[3:])
        assert observed[-1] == expected


def test_the_gfs_adapter_cli_carries_the_export_request_to_the_preparation(
        monkeypatch, tmp_path, capsys):
    from woof import gfs_direct

    observed = {}

    def prepare(**kwargs):
        observed.update(kwargs)
        return {"schema": "test", "wrf_manifest": {"status": "READY"}}

    monkeypatch.setattr(gfs_direct, "prepare_gfs_wrf", prepare)
    monkeypatch.setattr(
        gfs_direct, "prepared_forecast_next_command",
        lambda *_a, **_k: [])
    base = [
        "--series", "series.tsv", "--cycle", "2026-07-29_06:00:00",
        "--bridge", "bridge.exe", "--wps-namelist", "namelist.wps",
        "--experiment-config", "experiment.toml",
        "--input-manifest", "manifest.json",
        "--input-manifest-sha256", "a" * 64,
        "--output-root", str(tmp_path / "prep"),
    ]

    assert gfs_direct.main(base) == 0
    assert observed["stock_wrf_export"] is True

    assert gfs_direct.main(base + ["--no-stock-wrf-export"]) == 0
    assert observed["stock_wrf_export"] is False
    capsys.readouterr()


def test_the_front_door_says_plainly_when_the_bonus_export_did_not_happen():
    """A user who expected wrf-native-input/ is told, in one sentence."""

    from woof.gfs_direct import stock_wrf_export_notice

    assert stock_wrf_export_notice({"wrf_manifest": {"status": "READY"}}) == []
    assert stock_wrf_export_notice({}) == []

    refused = stock_wrf_export_notice({"wrf_manifest": {
        "status": "REFUSED",
        "reason": "unsupported direct-export configuration: "
                  "{'bl_pbl_physics': (5, 1)}"}})
    assert any("bonus stock-WRF export was refused" in line
               for line in refused)
    assert any("'bl_pbl_physics': (5, 1)" in line for line in refused)
    assert any("run the forecast command below" in line for line in refused)

    skipped = stock_wrf_export_notice(
        {"wrf_manifest": {"status": "NOT_REQUESTED"}})
    assert any("no stock-WRF export was requested" in line
               for line in skipped)


def test_a_single_domain_with_no_stock_contract_records_the_refusal(
        tmp_path, monkeypatch):
    """WDM6 has no stock-WRF package contract, and that is about exporting.

    The single-domain route raised the exporter's refusal, so `woof go`
    on the shipped Grell-Freitas suite (WDM6) stopped at prepare with
    "unsupported direct-export microphysics ... mp_physics=16" although
    its forecast restores the prepared cache, not the exported files.  The
    slot now records the refusal the mapped and HRRR routes write, and the
    forecast reader admits a proof carrying it.
    """
    from woof import gfs_direct
    from woof.prepared_single_domain_forecast import (
        _optional_stock_wrf_export)
    from woof.wrf_direct import validate_stock_wrf_export_config

    selection = {"profile": None, "acknowledgements": [],
                 "acknowledgement_provenance": {}}
    wrf_output = tmp_path / "wrf-native-input"

    def refuse(*_args, **_kwargs):
        wrf_output.mkdir()
        (wrf_output / "wrfinput_d01").write_bytes(b"partial")
        validate_stock_wrf_export_config({"mp_physics": 16},
                                         configured_suite=True)

    monkeypatch.setattr(gfs_direct, "export_prepared_wrf", refuse)
    receipt, refusal = gfs_direct._single_domain_stock_export(
        "cache", "static", "geometry", wrf_output, valid_time=None,
        boundary_interval_seconds=3600, physics_selection=selection)
    assert refusal is not None
    assert receipt["status"] == "REFUSED"
    assert receipt["schema"] == gfs_direct.SINGLE_DOMAIN_EXPORT_SCHEMA
    assert "mp_physics=16" in receipt["reason"]
    assert receipt["unsupported"] == {"mp_physics": [16, None]}
    assert not wrf_output.exists()
    proof = {"stock_wrf_export": "optional", "export": receipt}
    assert _optional_stock_wrf_export(proof, receipt) is True
    notice = gfs_direct.stock_wrf_export_notice(proof)
    assert any("bonus stock-WRF export was refused" in line
               for line in notice)
    assert any("run the forecast command below" in line for line in notice)

    # An export that runs keeps its READY receipt, and one whose physics
    # differs from the selection is still refused.
    ready = {"schema": gfs_direct.SINGLE_DOMAIN_EXPORT_SCHEMA,
             "status": "READY", "physics": selection}
    monkeypatch.setattr(gfs_direct, "export_prepared_wrf",
                        lambda *a, **k: dict(ready))
    receipt, refusal = gfs_direct._single_domain_stock_export(
        "cache", "static", "geometry", wrf_output, valid_time=None,
        boundary_interval_seconds=3600, physics_selection=selection)
    assert (receipt, refusal) == (ready, None)
    monkeypatch.setattr(gfs_direct, "export_prepared_wrf",
                        lambda *a, **k: {**ready, "physics": {}})
    with pytest.raises(RuntimeError, match="physics provenance differs"):
        gfs_direct._single_domain_stock_export(
            "cache", "static", "geometry", wrf_output, valid_time=None,
            boundary_interval_seconds=3600, physics_selection=selection)


def test_a_single_domain_export_not_requested_is_not_attempted(
        tmp_path, monkeypatch):
    """Declined, the single-domain slot says so and the exporter never runs."""
    from woof import gfs_direct
    from woof.prepared_single_domain_forecast import (
        _optional_stock_wrf_export)

    def must_not_run(*_args, **_kwargs):
        pytest.fail("a declined stock-WRF export was attempted")

    monkeypatch.setattr(gfs_direct, "export_prepared_wrf", must_not_run)
    wrf_output = tmp_path / "wrf-native-input"
    receipt, refusal = gfs_direct._single_domain_stock_export(
        "cache", "static", "geometry", wrf_output, valid_time=None,
        boundary_interval_seconds=3600,
        physics_selection={"profile": None, "acknowledgements": [],
                           "acknowledgement_provenance": {}},
        stock_wrf_export=False)
    assert refusal is None
    assert receipt == {
        "schema": gfs_direct.SINGLE_DOMAIN_EXPORT_SCHEMA,
        "status": "NOT_REQUESTED",
        "reason": "the caller did not request a stock-WRF export"}
    assert not wrf_output.exists()
    proof = {"stock_wrf_export": "off", "export": receipt}
    assert _optional_stock_wrf_export(proof, receipt) is True
    assert any("no stock-WRF export was requested" in line
               for line in gfs_direct.stock_wrf_export_notice(proof))


def test_a_bridge_decode_refusal_reaches_the_reader_as_a_sentence(
        monkeypatch, tmp_path, capsys):
    """E-06: the detection was already right; only the delivery was wrong.

    The Rust bridge diagnoses a GRIB whose valid time is not the one
    requested by CONTENT, and decode_failure_message turns its stderr
    into a remedy.  That careful diagnosis was then raised as a bare
    RuntimeError, which ``main`` does not catch -- it catches
    (ValueError, OSError) -- so the reader got a 42-line traceback at
    rc 1 where every neighbouring refusal prints one line at rc 2.
    """
    from woof import gfs_direct

    assert issubclass(gfs_direct.GfsRouteError, ValueError)

    def prepare(**_kwargs):
        raise gfs_direct.GfsRouteError(decode_failure_message(
            "GFS Rust bridge",
            "f006 valid 2026-07-29 12:00 does not match requested "
            "2026-07-29 06:00\n"))

    monkeypatch.setattr(gfs_direct, "prepare_gfs_wrf", prepare)
    rc = gfs_direct.main([
        "--series", "series.tsv", "--cycle", "2026-07-29_06:00:00",
        "--bridge", "bridge.exe", "--wps-namelist", "namelist.wps",
        "--experiment-config", "experiment.toml",
        "--input-manifest", "manifest.json",
        "--input-manifest-sha256", "a" * 64,
        "--output-root", str(tmp_path / "prep"),
    ])
    err = capsys.readouterr().err
    assert rc == 2
    assert "Traceback" not in err
    assert err.startswith("rw-wps --source gfs:")
    # the bridge's own words survive the delivery
    assert "does not match requested" in err


def test_an_internal_invariant_keeps_its_traceback(monkeypatch, tmp_path):
    """The negative control for the commit above: widening the handler to
    swallow RuntimeError would have reported OUR defect as the reader's
    mistake.  A violated internal invariant is still uncaught."""
    from woof import gfs_direct

    def prepare(**_kwargs):
        raise RuntimeError("direct-WRF export physics provenance differs")

    monkeypatch.setattr(gfs_direct, "prepare_gfs_wrf", prepare)
    with pytest.raises(RuntimeError, match="provenance differs"):
        gfs_direct.main([
            "--series", "series.tsv", "--cycle", "2026-07-29_06:00:00",
            "--bridge", "bridge.exe", "--wps-namelist", "namelist.wps",
            "--experiment-config", "experiment.toml",
            "--input-manifest", "manifest.json",
            "--input-manifest-sha256", "a" * 64,
            "--output-root", str(tmp_path / "prep"),
        ])


# ---------------------------------------------------------------------------
# The statics corridor request reaches the adapter from both front doors
# ---------------------------------------------------------------------------

def test_the_gfs_front_door_forwards_the_corridor_request_only_when_named():
    from woof import source_cli

    default = source_cli._gfs_command(_gfs_front_door_args([]))
    bare = source_cli._gfs_command(
        _gfs_front_door_args(["--statics-corridor"]))
    selected = source_cli._gfs_command(
        _gfs_front_door_args(["--statics-corridor", "2,3"]))

    assert "--statics-corridor" not in default
    assert "--statics-corridor" in bare
    assert selected[selected.index("--statics-corridor") + 1] == "2,3"
    assert source_cli._required_gfs_args(
        _gfs_front_door_args(["--statics-corridor"])) == []
    assert _gfs_front_door_args([]).statics_corridor is None

    # Malformed grid lists and a corridor without a GEOG source are
    # named at the door, not one process later.
    errors = source_cli._required_gfs_args(
        _gfs_front_door_args(["--statics-corridor", "2,potato"]))
    assert any("comma-separated" in error for error in errors)
    args = _gfs_front_door_args(["--statics-corridor"])
    args.geog_root = None
    args.static_input = "static.npz"
    args.static_receipt = "static.json"
    errors = source_cli._required_gfs_args(args)
    assert any("requires --geog-root" in error for error in errors)


def test_the_corridor_request_is_refused_by_the_routes_that_cannot_seal_one():
    """`--statics-corridor` is no longer GFS-only.

    A moving nest needs terrain and land use for everywhere it can reach,
    so the mapped, ERA5 and native 20CRv3 routes seal corridors too and
    accept the flag. The routes that cannot build one
    still refuse it by name, which is the half worth pinning: a flag
    that is silently ignored is worse than one that is refused.
    """
    from woof import source_cli

    args = _gfs_front_door_args(["--statics-corridor"])
    for validator in (source_cli._required_hrrr_args,):
        errors = validator(args)
        assert any("--statics-corridor" in error for error in errors),             validator.__name__
    for validator in (source_cli._required_twentycr_args,
                      source_cli._required_era5_args,
                      source_cli._required_mapped_args):
        errors = validator(args)
        assert not any("--statics-corridor" in error for error in errors),             validator.__name__


def test_the_gfs_adapter_cli_carries_the_corridor_request_to_the_preparation(
        monkeypatch, tmp_path, capsys):
    from woof import gfs_direct

    observed = {}

    def prepare(**kwargs):
        observed.update(kwargs)
        return {"schema": "test", "wrf_manifest": {"status": "READY"}}

    monkeypatch.setattr(gfs_direct, "prepare_gfs_wrf", prepare)
    monkeypatch.setattr(
        gfs_direct, "prepared_forecast_next_command",
        lambda *_a, **_k: [])
    base = [
        "--series", "series.tsv", "--cycle", "2026-07-29_06:00:00",
        "--bridge", "bridge.exe", "--wps-namelist", "namelist.wps",
        "--experiment-config", "experiment.toml",
        "--input-manifest", "manifest.json",
        "--input-manifest-sha256", "a" * 64,
        "--output-root", str(tmp_path / "prep"),
    ]

    assert gfs_direct.main(base) == 0
    assert observed["statics_corridor"] is None

    assert gfs_direct.main(base + ["--statics-corridor"]) == 0
    assert observed["statics_corridor"] == "all"

    assert gfs_direct.main(base + ["--statics-corridor", "2,4"]) == 0
    assert observed["statics_corridor"] == (2, 4)

    assert gfs_direct.main(base + ["--statics-corridor", "2,x"]) == 2
    capsys.readouterr()


def test_the_adapter_prints_corridor_size_accuracy_lines(
        monkeypatch, tmp_path, capsys):
    from woof import gfs_direct

    proof = {
        "schema": "test",
        "wrf_manifest": {"status": "READY"},
        "statics_corridor": {"domains": {"d02": {
            "parent_id": 1, "corridor_nx": 900, "corridor_ny": 900,
            "cache": {"bytes": 629_000_000}, "host_bytes": 628_000_000,
        }}},
    }
    monkeypatch.setattr(gfs_direct, "prepare_gfs_wrf",
                        lambda **_kwargs: proof)
    monkeypatch.setattr(gfs_direct, "prepared_forecast_next_command",
                        lambda *_a, **_k: [])
    assert gfs_direct.main([
        "--series", "series.tsv", "--cycle", "2026-07-29_06:00:00",
        "--bridge", "bridge.exe", "--wps-namelist", "namelist.wps",
        "--experiment-config", "experiment.toml",
        "--input-manifest", "manifest.json",
        "--input-manifest-sha256", "a" * 64,
        "--output-root", str(tmp_path / "prep"),
        "--statics-corridor",
    ]) == 0
    err = capsys.readouterr().err
    assert "statics corridor d02: 900x900 child cells" in err
    assert "629.0 MB on disk" in err
    assert "no GPU residency" in err


# ---------------------------------------------------------------------------
# UX finding R2 (2026-08-18 walk C, step 5): the eta-ladder refusal on the
# GFS/native direct routes speaks, in the mapped door's own voice.
# ---------------------------------------------------------------------------

def test_a_config_with_no_eta_ladder_is_refused_with_both_doors_named(
        tmp_path):
    """An import-namelist config declares a level count and no explicit
    ladder -- the shape every stock WRF namelist has.  The mapped door
    answers that in sentences (UX finding N6); this route answered
    ``explicit eta_levels has shape (0,)`` with no remedy at all."""

    from woof.ingest.source_coverage import VerticalLadderRefusal

    config = Path(__file__).parents[1] / "configs" / "gfs_wrf_direct_proof.toml"
    baseline = load_experiment(config)
    vertical = VerticalConfig(
        eta_levels=(), p_top=0.0, hybrid_opt=2, etac=0.37)
    exp = dataclasses.replace(baseline, vertical=vertical)
    wps = tmp_path / "namelist.wps"
    wps.write_text(_matching_wps(), encoding="utf-8")
    with pytest.raises(VerticalLadderRefusal) as caught:
        _validate_grid_and_vertical_contract(exp, wps)
    nz = int(baseline.root.run.nz)
    message = str(caught.value)
    assert f"nz={nz}" in message
    assert "no explicit eta_levels ladder" in message
    remedy = caught.value.remedy
    assert "eta_levels" in remedy
    assert "[shared]" in remedy
    assert "woof domain" in remedy


def test_a_malformed_eta_ladder_keeps_the_shape_error_and_gains_a_remedy(
        tmp_path):
    from woof.ingest.source_coverage import VerticalLadderRefusal

    config = Path(__file__).parents[1] / "configs" / "gfs_wrf_direct_proof.toml"
    baseline = load_experiment(config)
    nz = int(baseline.root.run.nz)
    short = VerticalConfig(
        eta_levels=tuple(float(value)
                         for value in np.linspace(1.0, 0.0, nz)),
        p_top=10_000.0, hybrid_opt=2, etac=0.37)
    exp = dataclasses.replace(baseline, vertical=short)
    wps = tmp_path / "namelist.wps"
    wps.write_text(_matching_wps(), encoding="utf-8")
    with pytest.raises(VerticalLadderRefusal) as caught:
        _validate_grid_and_vertical_contract(exp, wps)
    assert "shape" in str(caught.value)
    assert "eta_levels" in caught.value.remedy


def test_the_door_prints_the_ladder_refusal_with_its_remedy(
        tmp_path, capsys, monkeypatch):
    """Through ``python -m woof.gfs_direct``: the refusal family owns
    two lines -- the message and ITS remedy -- rather than being
    flattened into the one-sentence handler with the remedy dropped."""

    import woof.gfs_direct as gfs_direct_module
    from woof.gfs_direct import main as gfs_main
    from woof.ingest.source_coverage import (
        PREPARATION_REFUSAL_EXIT_CODE, VerticalLadderRefusal)

    def refuse(**kwargs):
        raise VerticalLadderRefusal(
            "GFS direct adapter d01: the experiment config declares nz=49 "
            "mass levels (WRF e_vert=50) and no explicit eta_levels ladder",
            remedy="remedy: add an explicit eta_levels ladder")

    monkeypatch.setattr(gfs_direct_module, "prepare_gfs_wrf", refuse)
    code = gfs_main([
        "--series", str(tmp_path / "series.tsv"),
        "--cycle", "2026-08-18_18:00:00",
        "--bridge", str(tmp_path / "bridge"),
        "--wps-namelist", str(tmp_path / "namelist.wps"),
        "--experiment-config", str(tmp_path / "case.toml"),
        "--input-manifest", str(tmp_path / "manifest.json"),
        "--input-manifest-sha256", "0" * 64,
        "--output-root", str(tmp_path / "out"),
    ])
    captured = capsys.readouterr()
    assert code == PREPARATION_REFUSAL_EXIT_CODE
    assert captured.out == ""
    assert "no explicit eta_levels ladder" in captured.err
    assert "remedy: add an explicit eta_levels ladder" in captured.err
    assert "Traceback" not in captured.err


# ---------------------------------------------------------------------------
# The unchanged-WRF companion files never stop a GFS forecast's preparation
# ---------------------------------------------------------------------------


def _single_domain_prepared(tmp_path, monkeypatch, profile, *, stock_wrf_export):
    """Prepare one GFS domain with PROFILE through the real orchestration.

    Decoding, interpolation and array numerics are the CPU fixtures of
    test_gfs_initial_perturbation; the cache writer is a stand-in that
    records the identity it was given.  The export stand-in runs the
    exporter's own configuration admission on that identity, exactly as
    ``export_prepared_wrf`` does before it writes a byte, so the refusal
    is the real one.
    """

    from woof import gfs_direct, wrf_direct
    from woof.cli import main as cli_main
    from test_gfs_initial_perturbation import _cpu_preparation, _inputs

    config = tmp_path / "experiment.toml"
    # Kessler's suite runs no longwave, and this window is at night there.
    night = (["--ack", "asymmetric-radiation-nocturnal-window-v1"]
             if profile.startswith("kessler-") else [])
    assert cli_main([
        "domain", "--point=35.3,-97.5", "--root-dx", "12", "--vram-gib", "8",
        "--source", "gfs", "--hours", "3", "--cycle", "2026-09-05T00",
        "--physics-profile", profile, *night, "--out", str(config)]) == 0
    exp = load_experiment(config)
    assert len(exp.domains) == 1
    exported = []
    _cpu_preparation(monkeypatch, exp)
    monkeypatch.setattr(gfs_direct, "_canonical_surface", lambda soil: {})

    import woof.ingest.prepared_cache as prepared_cache_module

    class RecordingCacheStream:
        """The prepared-cache writer, recorded: head, segments, seal.

        The seal writes a header naming the identity, which is what the
        export double below reads.
        """

        def __init__(self, directory, *, identity, **_kwargs):
            self.directory = Path(directory)
            self.identity = identity

        def move(self, directory):
            self.directory = Path(directory)

        def write_head(self, **kwargs):
            self.directory.mkdir(parents=True)
            return {"identity": {}, "metadata": {}, "arrays": {},
                    "payload_bytes": 0, "lbc": kwargs["lbc"],
                    "setup_core_fingerprint": "0" * 64}

        def write_segment(self, index, interval):
            return {"index": index,
                    "start_seconds": float(interval.start_seconds),
                    "end_seconds": float(interval.end_seconds),
                    "fields": sorted(interval.fields), "arrays": {},
                    "payload_bytes": 0, "prefix": {}}

        def seal(self):
            (self.directory / "header.json").write_text(
                json.dumps({"identity": self.identity}, sort_keys=True,
                           default=str),
                encoding="utf-8")
            return {"schema": "gpuwm-prepared-cache-v1", "status": "BUILT",
                    "content_sha256": "c" * 64, "array_count": 1,
                    "payload_bytes": 1}

    def export(prepared_cache, static_cache, geometry_receipt, output_dir,
               **_kwargs):
        header = json.loads(
            (Path(prepared_cache) / "header.json").read_text(encoding="utf-8"))
        exported.append(Path(output_dir))
        wrf_direct.validate_stock_wrf_export_config(
            header["identity"]["domain_config"]["run"], configured_suite=True)
        pytest.fail("the export admitted a scheme the unchanged-WRF files "
                    "have no package for")

    monkeypatch.setattr(
        prepared_cache_module, "PreparedCacheStream", RecordingCacheStream)
    monkeypatch.setattr(gfs_direct, "export_prepared_wrf", export)
    arguments = _inputs(tmp_path, config, "inputs")
    proof = gfs_direct.prepare_gfs_wrf(
        **arguments, stock_wrf_export=stock_wrf_export)
    return exp, arguments["output_root"], proof, exported


@pytest.mark.parametrize("profile", [
    "kessler-mp1-ysu-mm5-noah-dudhia-v1",
    "milbrandt2mom-mp9-ysu-mm5-noah-ntiedtke-rrtmg-legacy-v1",
    "wdm6-mp16-ysu-mm5-noah-grell-freitas-rte-rrtmgp-v1",
])
def test_a_scheme_the_wrf_files_cannot_carry_still_prepares_its_forecast(
        tmp_path, monkeypatch, capsys, profile):
    """`woof go` from GFS stopped in preparation for Kessler, Milbrandt-Yau
    and WDM6 with "unsupported direct-export microphysics: stock-WRF
    wrfinput export for mp_physics=1 is not available on this route",
    although WOOF runs all three and the forecast reads only the prepared
    cache.  The unchanged-WRF files are now refused by name in the proof
    and the preparation completes, as a domain tree's always has."""

    from woof import prepared_single_domain_forecast as runner
    from woof.gfs_direct import stock_wrf_export_notice

    exp, output_root, proof, exported = _single_domain_prepared(
        tmp_path, monkeypatch, profile, stock_wrf_export=True)
    capsys.readouterr()
    mp = int(exp.root.run.mp_physics)
    assert exported, "the export was not attempted"
    assert proof["stock_wrf_export"] == "optional"
    slot = proof["export"]
    assert slot["status"] == "REFUSED"
    assert slot["schema"] == "gpuwm-native-direct-wrf-export-v3"
    assert slot["reason"].startswith(
        f"unsupported direct-export microphysics: stock-WRF wrfinput export "
        f"for mp_physics={mp} is not available")
    assert slot["unsupported"] == {"mp_physics": [mp, None]}
    assert proof["initialization_artifacts"]["wrf_files"] == {}
    assert not (output_root / "wrf-native-input").exists()
    assert json.loads((output_root / "proof.json").read_text()) == proof
    # The forecast recognizes the refused slot as the requested mode.
    assert runner._optional_stock_wrf_export(proof, slot) is True
    notice = stock_wrf_export_notice(proof)
    assert any("bonus stock-WRF export was refused" in line for line in notice)
    assert any("run the forecast command below" in line for line in notice)


def test_a_declined_export_is_not_attempted_on_a_single_domain(
        tmp_path, monkeypatch, capsys):
    """`--no-stock-wrf-export` declined the files only for a domain tree."""

    from woof import prepared_single_domain_forecast as runner
    from woof.gfs_direct import stock_wrf_export_notice

    _exp, output_root, proof, exported = _single_domain_prepared(
        tmp_path, monkeypatch, "kessler-mp1-ysu-mm5-noah-dudhia-v1",
        stock_wrf_export=False)
    capsys.readouterr()
    assert exported == []
    assert proof["stock_wrf_export"] == "off"
    assert proof["export"] == {
        "schema": "gpuwm-native-direct-wrf-export-v3",
        "status": "NOT_REQUESTED",
        "reason": "the caller did not request a stock-WRF export"}
    assert not (output_root / "wrf-native-input").exists()
    assert runner._optional_stock_wrf_export(proof, proof["export"]) is True
    assert any("no stock-WRF export was requested" in line
               for line in stock_wrf_export_notice(proof))


def _deep_tree_root(tmp_path):
    """A 92-character output name in a 125-character folder: 277 deep."""
    if len(str(tmp_path)) >= 124:
        pytest.skip("temporary root already exceeds the 125-character parent")
    parent = tmp_path / ("p" * (125 - len(str(tmp_path)) - 1))
    return parent, parent / ("gfs-tree-domain-z80-" + "x" * 72)


def test_a_gfs_domain_tree_too_deep_for_windows_is_refused_before_decode(
        tmp_path, monkeypatch):
    """The GFS door publishes the same domain tree the HRRR stage does,
    and prepared it to the end under a root whose header the forecast
    could not open.  It is refused before the manifest verification
    hashes a GRIB, and so before the bridge decodes a field."""
    from woof import fetch_guard, gfs_direct
    from test_gfs_initial_perturbation import (
        _config, _cpu_preparation, _inputs)

    config = _config(tmp_path, domains=2)
    _cpu_preparation(monkeypatch, load_experiment(config))
    decoded = []
    hashed = []
    original_sha256 = gfs_direct._sha256

    def bridge(command, **_kwargs):
        decoded.append(command)
        pytest.fail("the bridge decoded under a root the forecast cannot read")

    def sha256_spy(path):
        hashed.append(Path(path))
        return original_sha256(path)

    monkeypatch.setattr(gfs_direct.subprocess, "run", bridge)
    monkeypatch.setattr(gfs_direct, "_sha256", sha256_spy)
    arguments = _inputs(tmp_path, config, "inputs")
    parent, arguments["output_root"] = _deep_tree_root(tmp_path)
    monkeypatch.setattr(fetch_guard, "windows_path_limit", lambda: 259)

    with pytest.raises(ValueError) as caught:
        gfs_direct.prepare_gfs_wrf(**arguments)

    message = str(caught.value)
    assert message.startswith("refusing output root ")
    assert "277 characters" in message
    assert "at least 18 characters shorter" in message
    assert not hashed
    assert not decoded
    assert not arguments["output_root"].exists()
    assert not list(parent.glob(".d-*"))


def test_decode_extends_old_full_file_manifest_above_top():
    from woof.gfs_direct import _decode_levels_above_model_top
    levels = _decode_levels_above_model_top({"source": {}}, 10000.0)
    assert min(levels) == 70.0
    assert 100.0 in levels


def test_snow_policy_defaults_to_wps_and_raw_requires_a_declared_knob(tmp_path):
    from woof.case_data import optional_case_data_from_config, gfs_preparation_policy
    assert gfs_preparation_policy(None)["gfs_snow_convention"] == "wps"
    from test_case_data import make_case_toml, _CASE_DATA_TOML
    p = make_case_toml(tmp_path, case_data=_CASE_DATA_TOML + 'gfs_snow_convention = "raw"\n')
    assert gfs_preparation_policy(optional_case_data_from_config(p))["gfs_snow_convention"] == "raw"
    p.write_text(p.read_text().replace('gfs_snow_convention = "raw"', 'gfs_snow_convention = "mistyped"'))
    with pytest.raises(ValueError, match="gfs_snow_convention"):
        optional_case_data_from_config(p)


def test_lake_skin_policy_defaults_to_wps_and_air_temperature_is_a_declared_knob(tmp_path):
    from woof.case_data import optional_case_data_from_config, gfs_preparation_policy
    assert gfs_preparation_policy(None)["gfs_lake_skin"] == "wps"
    from test_case_data import make_case_toml, _CASE_DATA_TOML
    p = make_case_toml(tmp_path, case_data=_CASE_DATA_TOML + 'gfs_lake_skin = "air_temperature"\n')
    assert gfs_preparation_policy(optional_case_data_from_config(p))["gfs_lake_skin"] == "air_temperature"
    p.write_text(p.read_text().replace('"air_temperature"', '"nearest"'))
    with pytest.raises(ValueError, match="gfs_lake_skin"):
        optional_case_data_from_config(p)


def test_gfs_only_policy_stays_out_of_the_shared_preparation_identity():
    """ERA5 and HRRR bind preparation_case_policy into their source identity;
    a GFS-only key there would move every one of their prepared caches."""
    from woof.case_data import gfs_preparation_policy, preparation_case_policy
    shared = preparation_case_policy(None)
    assert not any(key.startswith("gfs_") for key in shared)
    assert set(gfs_preparation_policy(None)) == {
        "gfs_snow_convention", "gfs_lake_skin", "gfs_lake_ice"}


def test_lake_ice_defaults_to_observed_and_wps_is_a_declared_knob(tmp_path):
    from woof.case_data import optional_case_data_from_config, gfs_preparation_policy
    assert gfs_preparation_policy(None)["gfs_lake_ice"] == "observed"
    from test_case_data import make_case_toml, _CASE_DATA_TOML
    p = make_case_toml(tmp_path, case_data=_CASE_DATA_TOML + 'gfs_lake_ice = "wps"\n')
    assert gfs_preparation_policy(optional_case_data_from_config(p))["gfs_lake_ice"] == "wps"
    p.write_text(p.read_text().replace('gfs_lake_ice = "wps"', 'gfs_lake_ice = "ims"'))
    with pytest.raises(ValueError, match="gfs_lake_ice"):
        optional_case_data_from_config(p)


def test_the_gfs_route_freezes_observed_lakes_on_the_root_and_every_child():
    """The root's soil call reads the provider's fields, and both child
    catalogs (one-shot and chained tree) carry the root's provider."""
    import inspect
    from woof import gfs_direct
    source = inspect.getsource(gfs_direct.prepare_gfs_wrf)
    assert 'case_policy["gfs_lake_ice"], times[0]' in source
    assert "soil = preprocess_land_surface_soil(\n            soil_fields," in source
    assert source.count("lake_ice_provider=lake_ice_provider,") == 2
    chained = inspect.getsource(gfs_direct._prepare_chained_gfs_tree)
    assert "lake_ice_provider=c.lake_ice_provider," in chained


def test_the_gfs_route_keeps_its_lakes_on_the_wps_search_by_default():
    """The route hands its lake mask to the mapping only under "wps", and
    both child catalogs (one-shot and chained tree) carry the root's
    conventions, so a child is mapped as its root is."""
    import inspect
    from woof import gfs_direct
    source = inspect.getsource(gfs_direct.prepare_gfs_wrf)
    assert "inland_water_search=lake_skin_search" in source
    assert 'if case_policy["gfs_lake_skin"] == "wps" else None' in source
    assert source.count('relative_humidity_convention="gfs_mixed"') == 2
    assert source.count('case_policy["gfs_lake_skin"] == "wps"),') == 1
    chained = inspect.getsource(gfs_direct._prepare_chained_gfs_tree)
    assert chained.count('relative_humidity_convention="gfs_mixed"') == 1
    assert 'c.case_policy["gfs_lake_skin"] == "wps"),' in chained


def test_git_disappearing_after_root_probe_falls_back_to_package_metadata(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    from woof import build_provenance, gfs_direct
    root = tmp_path / 'source'
    (root / '.git').mkdir(parents=True)
    monkeypatch.setattr(gfs_direct, '__file__', str(root / 'woof' / 'gfs_direct.py'))
    monkeypatch.delenv('WOOF_NATIVE_DISTRIBUTION_MANIFEST', raising=False)
    def query(command, **kwargs):
        if command[-1] == '--show-toplevel':
            return SimpleNamespace(stdout=str(root))
        raise FileNotFoundError('git disappeared')
    monkeypatch.setattr(subprocess, 'run', query)
    monkeypatch.setattr(build_provenance, 'owning_distribution', lambda: None)
    identity = gfs_direct._git_source_identity()
    assert identity['commit'] == 'unknown'
    assert identity['tracked_worktree_clean'] is None
    assert 'build_hash=unknown' in identity['package_provenance']['receipt_line']
