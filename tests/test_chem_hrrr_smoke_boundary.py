"""HRRR-Smoke's own 3-D smoke as WOOF's start state and edge values (CPU).

The hrrr-native-smoke row is a data-store boundary source: its wrfnat
MASSDEN/PRES/TMP records are fetched by byte range from the public bucket
(``noaa_index`` acquisition), decoded on HRRR's Lambert grid (GRIB2 template
3.30), converted from kg m-3 to ug kg-1 with each level's own density, placed
in the vertical by the levels' own pressure, and filled into every forcing
frame by woof.chem_source_init, so the start state and the lateral boundary
carry it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from woof.chem_table import catalog

ROW = catalog().sources["hrrr-native-smoke"]


def _index(names):
    lines, offset = [], 0
    for n, name in enumerate(names, start=1):
        lines.append(f"{n}:{offset}:d=2025010812:{name}:anl:")
        offset += 100
    return "\n".join(lines) + "\n"


def test_index_selection_takes_each_wanted_record_once():
    from woof.data_store_fetch import index_selection
    text = _index(["TMP:1 hybrid level", "MASSDEN:1 hybrid level", "UGRD:1 hybrid level",
                   "MASSDEN:2 hybrid level"])
    chosen = index_selection(text, {"MASSDEN:1 hybrid level", "MASSDEN:2 hybrid level"})
    assert chosen == [("MASSDEN:1 hybrid level", 100, 199), ("MASSDEN:2 hybrid level", 300, None)]
    with pytest.raises(ValueError, match="missing"):
        index_selection(text, {"MASSDEN:3 hybrid level"})
    with pytest.raises(ValueError, match="repeated"):
        index_selection(text + "5:400:d=2025010812:TMP:1 hybrid level:anl:\n", {"TMP:1 hybrid level"})


def test_the_plan_asks_for_smoke_with_its_own_pressure_and_temperature():
    from woof.data_store_fetch import resolve_acquisition
    plan = resolve_acquisition(ROW, area="31.5,-122,36,-115.8",
                               start="2025-01-08T12:00:00+00:00",
                               end="2025-01-09T00:00:00+00:00")
    groups = {r.group: dict(r.request) for r in plan.requests}
    assert set(groups) == {"hybrid_level", "surface"}
    assert groups["hybrid_level"]["variable"] == ("MASSDEN", "PRES", "TMP")
    assert len(groups["hybrid_level"]["hybrid_level"]) == 50
    assert groups["surface"]["variable"] == ("PRES",)
    assert groups["hybrid_level"]["time"] == ("12:00",)
    assert groups["hybrid_level"]["leadtime_hour"][0] == "0"
    assert groups["hybrid_level"]["leadtime_hour"][-1] == "12"


def test_index_range_download_concatenates_every_lead_in_inventory_order(tmp_path):
    from woof.data_store_fetch import index_range_download, resolve_acquisition
    plan = resolve_acquisition(ROW, area="31.5,-122,36,-115.8",
                               start="2025-01-08T12:00:00+00:00",
                               end="2025-01-08T13:00:00+00:00")
    request = next(r for r in plan.requests if r.group == "surface")
    names = ["UGRD:surface", "PRES:surface", "TMP:surface"]
    calls = []

    def opener(url, headers=None):
        calls.append((url, dict(headers or {})))
        if url.endswith(".idx"):
            return _index(names).encode()
        first = int(headers["Range"].split("=")[1].split("-")[0])
        return b"GRIB" + url[-8:-6].encode() + str(first).encode()

    part = tmp_path / "out.grib2"
    index_range_download(request, part, progress=lambda *a: None, opener=opener)
    urls = [u for u, _h in calls if not u.endswith(".idx")]
    assert urls == [
        "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20250108/conus/hrrr.t12z.wrfnatf00.grib2",
        "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20250108/conus/hrrr.t12z.wrfnatf01.grib2"]
    assert [h["Range"] for u, h in calls if "Range" in h] == ["bytes=100-199", "bytes=100-199"]
    assert part.read_bytes() == b"GRIB00100GRIB01100"


def test_a_lambert_stack_reads_as_index_axes_with_its_projection(tmp_path):
    import hashlib
    import json
    from woof.grib2_stack import read_stack
    data = np.arange(2 * 3 * 4, dtype="<f4").tobytes()
    (tmp_path / "x.f32le").write_bytes(data)
    grid = {"template": 30, "nx": 4, "ny": 3, "lat1": 21.138123, "lat2": 0, "lon1": 237.280472,
            "lon2": 0, "dlat": 3000.0, "dlon": 3000.0, "scan_mode": 64, "latin1": 38.5,
            "latin2": 38.5, "lov": 262.5}
    row = {"key": "x", "file": "x.f32le", "sha256": hashlib.sha256(data).hexdigest(),
           "levels": [1, 2], "pdt": 0, "valid_time": "2025-01-08T12:00:00+00:00", "grid": grid}
    (tmp_path / "stack.json").write_text(json.dumps(
        {"schema": "gpuwm-grib2-stack-v1", "outputs": [row], "coordinate_values": {"x": []}}))
    (field,) = read_stack(tmp_path)
    assert field.array.shape == (2, 3, 4)
    assert field.projection["lov"] == 262.5 and field.projection["nx"] == 4
    assert list(field.latitude) == [0, 1, 2] and list(field.longitude) == [0, 1, 2, 3]
    grid["scan_mode"] = 0
    (tmp_path / "stack.json").write_text(json.dumps(
        {"schema": "gpuwm-grib2-stack-v1", "outputs": [row], "coordinate_values": {"x": []}}))
    with pytest.raises(ValueError, match="wrong order"):
        read_stack(tmp_path)


def _hrrr_projection(nx=1799, ny=1059):
    return {"template": 30, "nx": nx, "ny": ny, "lat1": 21.138123, "lon1": 237.280472,
            "dlat": 3000.0, "dlon": 3000.0, "latin1": 38.5, "latin2": 38.5, "lov": 262.5,
            "scan_mode": 64}


def test_targets_land_on_hrrrs_own_grid_points():
    from woof.chem_boundary_remap import TargetGrid, _horizontal_plan
    from woof.ingest.hrrr import hrrr_source_grid
    grid = hrrr_source_grid()
    # Two HRRR mass points (0-based i, j) as geographic targets.
    lat, lon = grid.ij_to_latlon(np.array([[401.0, 1201.0]]), np.array([[301.0, 801.0]]))
    seen = {}
    backend = SimpleNamespace(indexed_plan=lambda shape, y, x: seen.update(shape=shape, y=y, x=x) or "plan")
    field = SimpleNamespace(projection=_hrrr_projection())
    target = TargetGrid(lat, lon, np.ones((2, 1, 2), np.float32))
    assert _horizontal_plan(backend, ROW, [field], None, None, target) == "plan"
    assert seen["shape"] == (1059, 1799)
    assert np.allclose(seen["x"], [[400.0, 1200.0]], atol=1e-6)
    assert np.allclose(seen["y"], [[300.0, 800.0]], atol=1e-6)
    far = TargetGrid(np.array([[60.0]]), np.array([[-150.0]]), np.ones((2, 1, 1), np.float32))
    with pytest.raises(ValueError, match="outside"):
        _horizontal_plan(backend, ROW, [field], None, None, far)


def test_hrrrs_own_outer_rows_land_on_its_edge_points():
    """The recipe runs HRRR's own 1799 x 1059 grid, so its outer rows ARE the
    source's outer rows.  Projection round-off put them a hair past the edge
    and the start state was refused ("the domain reaches outside its 1799 x
    1059 grid"); they now land exactly on the edge, and a point a real cell
    outside is still refused."""
    from woof.chem_boundary_remap import TargetGrid, _horizontal_plan
    from woof.static.lambert import LambertGrid
    from woof.ingest.hrrr import hrrr_source_grid
    grid = hrrr_source_grid()
    i = np.array([[1.0, 1799.0, 1.0, 1799.0, 900.0]])
    j = np.array([[1.0, 1.0, 1059.0, 1059.0, 530.0]])
    lat, lon = grid.ij_to_latlon(i, j)   # 1-based, as the grid declares (known_x = 1)
    # As the static file stores them: float32.
    lat, lon = (np.asarray(v, np.float32).astype(np.float64) for v in (lat, lon))
    seen = {}
    backend = SimpleNamespace(indexed_plan=lambda shape, y, x: seen.update(shape=shape, y=y, x=x) or "plan")
    field = SimpleNamespace(projection=_hrrr_projection())
    target = TargetGrid(lat, lon, np.ones((2, 1, 5), np.float32))
    assert _horizontal_plan(backend, ROW, [field], None, None, target) == "plan"
    assert seen["x"].min() >= 0.0 and seen["x"].max() <= 1798.0
    assert seen["y"].min() >= 0.0 and seen["y"].max() <= 1058.0
    assert np.allclose(seen["x"], i - 1.0, atol=1e-2)
    assert np.allclose(seen["y"], j - 1.0, atol=1e-2)
    lat, lon = grid.ij_to_latlon(np.array([[-1.0]]), np.array([[500.0]]))
    with pytest.raises(ValueError, match="outside"):
        _horizontal_plan(backend, ROW, [field], None, None,
                         TargetGrid(lat, lon, np.ones((2, 1, 1), np.float32)))


def test_the_full_hrrr_grid_copies_its_own_cells_index_for_index():
    """The recipe's 1799 x 1059 grid as WPS builds it (dx 3000 m on the
    6370 km sphere, the static file's float32 latitude/longitude) against
    HRRR's GRIB header (3000 m on 6371229 m): the two descriptions drift to
    0.35 cell at the north-east corner, and the start state was refused
    ("target x 0..1798.35, y 0..1058.2").  It now takes the identity route:
    every point its own cell, exactly."""
    from woof.chem_boundary_remap import TargetGrid, _horizontal_plan
    from woof.static.lambert import LambertGrid
    wps = LambertGrid(ref_lat=21.138123, ref_lon=-122.719528, truelat1=38.5,
                      truelat2=38.5, stand_lon=-97.5, dx=3000.0, dy=3000.0,
                      e_we=1800, e_sn=1060, known_x=1.0, known_y=1.0)
    j, i = np.meshgrid(np.arange(1, 1060, dtype=np.float64),
                       np.arange(1, 1800, dtype=np.float64), indexing="ij")
    lat, lon = wps.ij_to_latlon(i, j)
    lat, lon = (np.asarray(v, np.float32).astype(np.float64) for v in (lat, lon))
    seen = {}
    backend = SimpleNamespace(indexed_plan=lambda shape, y, x: seen.update(shape=shape, y=y, x=x) or "plan")
    field = SimpleNamespace(projection=_hrrr_projection())
    target = TargetGrid(lat, lon, np.ones((2, 1059, 1799), np.float32))
    assert _horizontal_plan(backend, ROW, [field], None, None, target) == "plan"
    assert np.array_equal(seen["x"], i - 1.0) and np.array_equal(seen["y"], j - 1.0)
    # An index-aligned window of it (columns 349.., every row) is its own
    # cells too.
    window = TargetGrid(lat[:, 349:1450], lon[:, 349:1450],
                        np.ones((2, 1059, 1101), np.float32))
    assert _horizontal_plan(backend, ROW, [field], None, None, window) == "plan"
    assert np.array_equal(seen["x"], i[:, 349:1450] - 1.0)
    assert np.array_equal(seen["y"], j[:, 349:1450] - 1.0)


def _smoke_column(*, cycle, unit_factor=1.0, row=ROW):
    """A synthetic HRRR column end to end on the real Rust operators: smoke
    density rho*q with rho = P/(R T) on four hybrid levels numbered from the
    ground, onto target pressures inside the source column.  ``unit_factor``
    is what the producer multiplied the kg m-3 density by (1e9 for the
    ug m-3 bytes NOAA wrote before 2021-12-21 14Z)."""
    from woof.chem_boundary_remap import SourceFrames, TargetGrid, boundary_fields_for_grid
    from woof.chem_table import load_sets
    try:
        from woof.ingest.cpu_backend import CpuPreprocessBackend
        backend = CpuPreprocessBackend()
        backend.weighted_combination
    except Exception as error:   # pragma: no cover - a box without the bridge
        pytest.skip(f"CPU preprocessing bridge not loadable: {error}")
    nx, ny, nz = 6, 5, 4
    proj = _hrrr_projection(nx, ny)
    p_levels = np.array([95000.0, 85000.0, 70000.0, 50000.0], np.float32)   # level 1 = ground
    pres = np.broadcast_to(p_levels[:, None, None], (nz, ny, nx)).astype(np.float32)
    tmp = np.full((nz, ny, nx), 280.0, np.float32)
    q = np.float32(20.0)                                    # ug/kg
    rho = pres / (np.float32(287.04) * tmp)
    massden = (q * 1e-9 * rho * unit_factor).astype(np.float32)
    psfc = np.full((ny, nx), 100000.0, np.float32)

    def fld(values, levels):
        return SimpleNamespace(values=values, levels=levels, latitude=np.arange(ny, dtype=float),
                               longitude=np.arange(nx, dtype=float), coordinate_values=(),
                               projection=proj)
    when = cycle or datetime(2025, 1, 8, 12, tzinfo=timezone.utc)
    levels = tuple(float(k) for k in range(1, nz + 1))
    frame = {"MASSDEN": fld(massden, levels), "PRES": fld(pres, levels), "TMP": fld(tmp, levels),
             "PSFC": fld(psfc[None], (0.0,))}
    frames = SourceFrames("hrrr-native-smoke", {when: frame})
    from woof.ingest.hrrr import hrrr_source_grid
    lat, lon = hrrr_source_grid().ij_to_latlon(np.array([[2.5, 3.5]]), np.array([[2.0, 2.5]]))
    target = TargetGrid(lat, lon, np.broadcast_to(
        np.array([90000.0, 80000.0, 60000.0], np.float32)[:, None, None], (3, 1, 2)).copy())
    table = load_sets(["smoke"], ["rave-3km", "hrrr-native-smoke"])
    return boundary_fields_for_grid(table, row, frames, target, when, backend=backend,
                                    available=("rave-3km", "hrrr-native-smoke"), cycle=cycle)


def test_mass_density_becomes_mixing_ratio_on_the_models_levels():
    out, receipt = _smoke_column(cycle=datetime(2025, 1, 8, 12, tzinfo=timezone.utc))
    assert set(out) == {"smoke"}
    assert np.allclose(out["smoke"], 20.0, rtol=1e-4)
    assert "kg_m3_to_ug_kg_dry" in receipt["operators"]["convert_and_time"]
    assert "Lambert" in receipt["operators"]["horizontal"]


def test_massden_before_the_units_epoch_is_read_as_ug_m3():
    """NOAA HRRR wrfnat and RAP awp130bgrb wrote MASSDEN in ug m-3 through
    the 2021-12-21 13Z cycle (same kg m-3 label), kg m-3 from 14Z.  A
    2021-07-28 start read as kg m-3 began with 8e8 ug/kg smoke and failed
    its initial health check (nwfa ~6e17)."""
    out, _ = _smoke_column(cycle=datetime(2021, 7, 28, 12, tzinfo=timezone.utc), unit_factor=1e9)
    assert np.allclose(out["smoke"], 20.0, rtol=1e-4)
    last_old, _ = _smoke_column(cycle=datetime(2021, 12, 21, 13, tzinfo=timezone.utc), unit_factor=1e9)
    first_new, _ = _smoke_column(cycle=datetime(2021, 12, 21, 14, tzinfo=timezone.utc))
    assert np.allclose(last_old["smoke"], 20.0, rtol=1e-4)
    assert np.allclose(first_new["smoke"], 20.0, rtol=1e-4)


def test_a_cycle_after_the_epoch_is_read_bitwise_as_before():
    """The epoch scale is 1.0 after 2021-12-21 14Z: the smoke field is
    bitwise the field of the same row without any units_epochs row."""
    import dataclasses
    when = datetime(2023, 6, 7, 18, tzinfo=timezone.utc)
    variables = dict(ROW.variables)
    variables["MASSDEN"] = {k: v for k, v in variables["MASSDEN"].items() if k != "units_epochs"}
    plain = dataclasses.replace(ROW, variables=variables)
    with_epochs, _ = _smoke_column(cycle=when)
    without, _ = _smoke_column(cycle=None, row=plain)
    assert with_epochs["smoke"].tobytes() == without["smoke"].tobytes()


def test_a_read_without_a_cycle_is_refused_by_name():
    from woof.chem_boundary_remap import BoundaryRemapError
    with pytest.raises(BoundaryRemapError, match=r"MASSDEN: its units changed at cycle 2021-12-21T14:00:00Z.*carries no file cycle"):
        _smoke_column(cycle=None)


def test_the_files_cycle_comes_from_the_fetch_receipts(tmp_path):
    import json
    from woof.chem_source_init import ChemSourceUnavailable, files_cycle
    files = {}
    for stem, cycle in (("a", "12:00"), ("b", "12:00")):
        (tmp_path / f"{stem}.json").write_text(json.dumps({"dataset": "hrrr-wrfnat", "request": {
            "date": "2021-07-28/2021-07-28", "time": [cycle], "leadtime_hour": ["0"]}}))
        files[stem] = tmp_path / f"{stem}.grib2"
    assert files_cycle(ROW, files) == datetime(2021, 7, 28, 12, tzinfo=timezone.utc)
    (tmp_path / "b.json").write_text(json.dumps({"dataset": "hrrr-wrfnat", "request": {
        "date": "2021-07-28/2021-07-28", "time": ["13:00"], "leadtime_hour": ["0"]}}))
    with pytest.raises(ChemSourceUnavailable, match="cycles"):
        files_cycle(ROW, files)
    assert files_cycle(ROW, {"x": tmp_path / "none.grib2"}) is None


def test_the_rap_row_carries_the_same_epoch():
    from woof.chem_conversions import units_epoch_scale
    rap = catalog().sources["rap-native-smoke"].variables["MASSDEN"]
    for row in (ROW.variables["MASSDEN"], rap):
        assert units_epoch_scale(row, datetime(2021, 12, 21, 13, tzinfo=timezone.utc)) == 1e-9
        assert units_epoch_scale(row, "2021-12-21T14:00:00Z") == 1.0


def test_a_run_started_from_hrrr_takes_hrrr_smoke_without_naming_it():
    from woof.config import apply_route_chem_sources
    raw = {"shared": {"chem_sets": "smoke", "chem_sources": "rave-3km"}}
    apply_route_chem_sources(raw, {"source": "hrrr-prs"})
    assert raw["shared"]["chem_sources"] == "rave-3km,hrrr-native-smoke"
    # Stated already: unchanged.  Another start (GFS) or chem off: unchanged.
    raw = {"shared": {"chem_sets": ["smoke"], "chem_sources": ["hrrr-native-smoke", "rave-3km"]}}
    apply_route_chem_sources(raw, {"source": "hrrr"})
    assert raw["shared"]["chem_sources"] == ["hrrr-native-smoke", "rave-3km"]
    for raw, hints in (({"shared": {"chem_sets": "smoke", "chem_sources": "rave-3km"}}, {"source": "gfs"}),
                       ({"shared": {"chem_sets": "smoke", "chem_sources": "rave-3km"}}, None),
                       ({"shared": {"chem_sources": "rave-3km"}}, {"source": "hrrr"}),
                       ({"shared": {"chem_sets": "gocart_primary"}}, {"source": "hrrr"})):
        before = dict(raw["shared"])
        apply_route_chem_sources(raw, hints)
        assert raw["shared"] == before


def test_the_hrrr_smoke_row_is_a_data_store_boundary_source_with_its_own_levels():
    from woof.chem_source_init import chem_boundary_fields
    from woof.chem_table import load_sets
    from woof.config import RunConfig, validate_chem_config
    assert ROW.acquisition["kind"] == "noaa_index"
    assert ROW.vertical["kind"] == "pressure_field"
    assert set(ROW.default_for_fetch_sources) == {"hrrr", "hrrr-prs"}
    cfg = RunConfig(nx=12, ny=10, nz=6, dx=3000.0, dy=3000.0, ztop=12000.0, dt=18.0,
                    run_seconds=60.0, bl_pbl_physics=1, sf_sfclay_physics=1,
                    chem_sets="smoke", chem_sources="rave-3km,hrrr-native-smoke")
    validate_chem_config(cfg)          # the old "no forcing-frame fill" refusal is retired
    table = load_sets(["smoke"], ["rave-3km", "hrrr-native-smoke"])
    assert chem_boundary_fields(table, cfg) == ("chem_smoke",)


def test_the_surface_group_never_takes_the_hybrid_level_file(tmp_path):
    """Both receipts of one HRRR fetch name PRES; only the surface one holds
    PRES at the surface.  The hybrid-level file's content-addressed name
    sorted first and was handed to the surface group, so PSFC stacked from it
    matched zero records (box D 2026-10-21 06Z and 2024-04-27 18Z)."""
    import json
    from woof.chem_source_init import covering_files
    area = [53.001, -135.0, 20.979, -59.994]
    common = {"date": "2026-10-21/2026-10-21", "time": ["06:00"],
              "leadtime_hour": ["0", "1"], "product": "wrfnat", "area": area}
    receipts = {
        "0hybrid": dict(common, variable=["MASSDEN", "PRES", "TMP"],
                        hybrid_level=[str(k) for k in range(1, 51)]),
        "1surface": dict(common, variable=["PRES"]),
    }
    for stem, request in receipts.items():
        (tmp_path / f"{stem}.json").write_text(json.dumps(
            {"dataset": "hrrr-wrfnat", "request": request}))
        (tmp_path / f"{stem}.grib2").write_bytes(b"GRIB")
    files = covering_files(ROW, ("MASSDEN", "PRES", "TMP", "PSFC"),
                           datetime(2026, 10, 21, 6, tzinfo=timezone.utc),
                           cache_root=tmp_path)
    assert files == {"hybrid_level": tmp_path / "0hybrid.grib2",
                     "surface": tmp_path / "1surface.grib2"}


def test_initial_frame_prefers_smallest_covering_receipt(tmp_path):
    import json
    from woof.chem_source_init import covering_files
    common = {"date": "2026-10-21/2026-10-21", "time": ["06:00"],
              "product": "wrfnat", "area": [53.001, -135.0, 20.979, -59.994]}
    for stem, leads in (("0full", [str(k) for k in range(25)]), ("1initial", ["0"])):
        for group, extra in (("hybrid", {"variable": ["MASSDEN", "PRES", "TMP"],
                                        "hybrid_level": [str(k) for k in range(1, 51)]}),
                             ("surface", {"variable": ["PRES"]})):
            path = tmp_path / f"{stem}-{group}"
            path.with_suffix(".json").write_text(json.dumps({"dataset": "hrrr-wrfnat",
                "request": dict(common, leadtime_hour=leads, **extra)}))
            path.with_suffix(".grib2").write_bytes(b"GRIB")

    def select(hour):
        return covering_files(ROW, ("MASSDEN", "PRES", "TMP", "PSFC"),
            datetime(2026, 10, 21, hour, tzinfo=timezone.utc), cache_root=tmp_path)

    assert all(path.name.startswith("1initial") for path in select(6).values())
    assert all(path.name.startswith("0full") for path in select(7).values())
