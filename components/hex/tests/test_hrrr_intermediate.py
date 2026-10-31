"""The projected-source intermediate: the writer, the target, the window, the table.

Engine-free where it can be: the writer is proved against this tree's own
reader; the decoder and the interpolation operator are the engine's and are
exercised on hardware by the lane's receipt, not here.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from woof.hex import hrrr_intermediate as hi  # noqa: E402
from woof.hex.cli import build_parser  # noqa: E402
from woof.hex.hrrr_intermediate import IntermediateRefusal, LatLonTarget  # noqa: E402
from woof.hex.wps_intermediate import WpsIntermediateReader, inventory  # noqa: E402
from _layout import PACKAGE_DIR


# ---------------------------------------------------------------------------
# the writer is the inverse of the tree's reader
# ---------------------------------------------------------------------------
def test_a_written_record_reads_back_through_the_frozen_reader(tmp_path: Path) -> None:
    target = LatLonTarget(south=37.5, west=-96.0, dlat=0.025, dlon=0.025, nx=5, ny=3)
    values = np.arange(15, dtype=np.float32).reshape(3, 5) + 280.0
    path = tmp_path / "MET:2026-09-13_15"
    path.write_bytes(hi.wps_record_bytes(
        hdate="2026-09-13_15:00:00", xfcst=0.0, map_source="NCEP HRRR", field_name="TT",
        units="K", description="Temperature", level=200100.0, target=target, values=values,
    ))
    with WpsIntermediateReader(path) as reader:
        fields = list(reader.iter_fields())
    assert len(fields) == 1
    got = fields[0]
    assert got.version == 5 and got.field == "TT" and got.level == 200100.0
    assert got.valid_time == "2026-09-13_15:00:00" and got.map_source == "NCEP HRRR"
    assert (got.nx, got.ny) == (5, 3)
    assert got.projection.code == 0 and got.projection.name == "latlon"
    assert got.projection.start_location == "SWCORNER"
    assert got.projection.start_latitude == pytest.approx(37.5)
    assert got.projection.start_longitude == pytest.approx(-96.0)
    assert got.projection.delta_latitude == pytest.approx(0.025)
    assert got.projection.delta_longitude == pytest.approx(0.025)
    assert got.projection.earth_radius_km == pytest.approx(6371.229)
    assert got.is_wind_grid_relative is False
    # The reader keeps Fortran (nx, ny); our slab was (ny, nx): the same numbers.
    assert np.array_equal(got.values, values.T)
    # Big-endian Fortran sequential markers, as ungrib writes them.
    assert struct.unpack(">i", path.read_bytes()[:4])[0] == 4


def test_write_intermediate_refuses_a_slab_of_the_wrong_shape_or_with_a_nan(tmp_path: Path) -> None:
    target = LatLonTarget(south=0.0, west=0.0, dlat=0.1, dlon=0.1, nx=2, ny=2)
    with pytest.raises(IntermediateRefusal):
        hi.wps_record_bytes(hdate="x", xfcst=0, map_source="s", field_name="TT", units="K",
                            description="d", level=1.0, target=target, values=np.zeros((3, 2)))
    with pytest.raises(IntermediateRefusal) as refusal:
        hi.wps_record_bytes(hdate="x", xfcst=0, map_source="s", field_name="TT", units="K",
                            description="d", level=1.0, target=target,
                            values=np.array([[1.0, np.nan], [0.0, 0.0]]))
    assert "non-finite" in str(refusal.value)


def test_write_intermediate_inventories_every_record_before_signing(tmp_path: Path) -> None:
    target = LatLonTarget(south=30.0, west=-100.0, dlat=0.5, dlon=0.5, nx=4, ny=3)
    records = [
        {"field": "PRESSURE", "units": "Pa", "description": "P", "level": 1.0,
         "values": np.full((3, 4), 90000.0, dtype=np.float32)},
        {"field": "TT", "units": "K", "description": "T", "level": 200100.0,
         "values": np.full((3, 4), 290.0, dtype=np.float32)},
    ]
    written = hi.write_intermediate(
        tmp_path / "MET:2026-09-13_16", records, valid_time=datetime(2026, 9, 13, 16),
        forecast_hour=1.0, map_source="test", target=target,
    )
    assert written["field_records"] == 2 and written["fields"] == {"PRESSURE": 1, "TT": 1}
    assert written["valid_time"] == "2026-09-13_16:00:00" and written["record_endian"] == "big"
    seen = inventory(tmp_path / "MET:2026-09-13_16", include_statistics=True)
    assert seen["records"][0]["data"]["min"] == 90000.0


# ---------------------------------------------------------------------------
# the target box and the helpers
# ---------------------------------------------------------------------------
def test_the_target_covers_the_cap_plus_the_margin_on_the_stated_spacing() -> None:
    target = hi.target_for_cap((39.1, -94.58), 135.0, margin_km=30.0, spacing_deg=0.025)
    assert target.dlat == 0.025 and target.dlon == 0.025
    assert target.south < 39.1 - 165.0 / hi.KM_PER_DEG <= target.south + target.dlat
    assert target.north > 39.1 + 165.0 / hi.KM_PER_DEG - target.dlat
    assert target.west < -94.58 - 165.0 / (hi.KM_PER_DEG * np.cos(np.radians(39.1)))
    lat, lon = target.mesh()
    assert lat.shape == (target.ny, target.nx) == lon.shape
    assert lat[0, 0] == pytest.approx(target.south) and lon[0, 0] == pytest.approx(target.west)
    assert lat[1, 0] - lat[0, 0] == pytest.approx(0.025) and lon[0, 1] - lon[0, 0] == pytest.approx(0.025)
    with pytest.raises(IntermediateRefusal):
        hi.target_for_cap((89.0, 0.0), 500.0)


def test_the_target_from_a_plan_reads_the_cull_region_and_its_halo(tmp_path: Path) -> None:
    plan = {
        "schema": "gpuwm-hex.point-generate/v1",
        "plan": {"cull_region": {"kind": "cap", "center_deg": [39.1, -94.58], "radius_km": 135.0},
                 "cull": {"halo_km": 20.0}},
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    target, basis = hi.target_from_plan(path, margin_km=30.0, spacing_deg=0.025)
    assert basis == {"centre_deg": [39.1, -94.58], "cut_radius_km": 135.0, "halo_km": 20.0}
    reach = (135.0 + 20.0 + 30.0) / hi.KM_PER_DEG
    assert target.north - target.south >= 2.0 * reach - 0.05
    path.write_text('{"plan": {}}', encoding="utf-8")
    with pytest.raises(IntermediateRefusal):
        hi.target_from_plan(path, margin_km=30.0, spacing_deg=0.025)


def test_a_drawn_cut_takes_the_cap_that_encloses_its_polygon(tmp_path: Path) -> None:
    # regional --shape / --ellipse write the grown outline as the cull_region; the window must cover all of it.
    ring = [[34.0, -101.5], [33.5, -95.0], [37.0, -93.5], [40.0, -96.0], [39.5, -101.0]]
    plan = {"schema": "gpuwm-hex.point-generate/v1",
            "plan": {"cull_region": {"kind": "polygon", "vertices_deg": ring}, "cull": {"halo_km": 60.0}}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    target, basis = hi.target_from_plan(path, margin_km=30.0, spacing_deg=0.025)
    assert basis["cut_kind"] == "polygon" and basis["halo_km"] == 60.0
    for lat, lon in ring:
        assert target.south < lat < target.north and target.west < lon < target.east
    assert 380.0 < basis["cut_radius_km"] < 520.0


def test_cycle_and_hour_parsing_refuse_gaps_and_nonsense() -> None:
    assert hi.parse_cycle("2026-09-13T15") == datetime(2026, 9, 13, 15)
    assert hi.parse_cycle("2026-09-13_15:00:00") == datetime(2026, 9, 13, 15)
    assert hi.parse_hours("0-3") == (0, 1, 2, 3)
    assert hi.parse_hours("2,0,1") == (0, 1, 2)
    with pytest.raises(IntermediateRefusal):
        hi.parse_hours("0,2")
    with pytest.raises(IntermediateRefusal):
        hi.parse_cycle("yesterday")


def test_the_source_table_is_the_only_place_a_source_lives() -> None:
    row = hi.source_row("HRRR")
    assert row.decoder_key == "hrrr" and row.levels == 50
    assert row.grib_names[0].format(cycle=datetime(2026, 9, 13, 15), hour=2) == "hrrr.t15z.wrfnatf02.grib2"
    assert row.grib_names[1].format(cycle=datetime(2026, 9, 13, 15), hour=0) == "hrrr.t15z.soilf00.grib2"
    assert row.soil_layer_names[0] == ("ST000010", "SM000010")
    # HRRR's own state: hydrometeors, its numbers, 2 m RH, RUC's nine nodes under the RAP/HRRR Vtable names.
    wps = {spec[0] for spec in row.atmosphere_3d.values()}
    assert {"QC", "QR", "QI", "QS", "QG", "QNC", "QNI", "QNR"} <= wps
    assert row.surface["RH2"][0] == "RH" and set(row.decoder_extras) == {"QNC", "QNI", "QNR", "RH2"}
    assert row.analyzed_aerosol and row.map_source == "NCEP HRRR Model"
    assert row.soil_node_names[0] == ("SOILT000", "SOILM000") and row.soil_node_names[-1] == ("SOILT300", "SOILM300")
    assert len(row.soil_node_names) == 9
    # The map source lands in a 32-character WPS field; a longer one refused
    # the first real regrid on 2026-09-13 after the decode had already run.
    for each in hi.SOURCE_ROWS.values():
        assert len(each.map_source.encode("ascii")) <= 32, each.map_source
    with pytest.raises(IntermediateRefusal) as refusal:
        hi.source_row("rap")
    assert "SOURCE_ROWS" in str(refusal.value)
    # Nothing outside the table names the source.
    text = (PACKAGE_DIR / "hrrr_intermediate.py").read_text(encoding="utf-8")
    body = text.split("SOURCE_ROWS: Mapping[str, SourceRow] = {", 1)[1].split("\ndef source_row", 1)[1]
    assert '== "hrrr"' not in body and "if row.name" not in body


def test_the_doors_are_subcommands_of_the_console_script() -> None:
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices  # noqa: SLF001
    assert "intermediate" in commands and "lbc" in commands
    arguments = parser.parse_args([
        "intermediate", "--grib-dir", "d", "--cycle", "2026-09-13T15", "--out-dir", "o",
        "--point", "39.1,-94.58", "--radius-km", "155",
    ])
    assert arguments.hours == "0-3" and arguments.spacing_deg == hi.DEFAULT_SPACING_DEG
    with pytest.raises(IntermediateRefusal) as refusal:
        hi.request_from_arguments(parser.parse_args([
            "intermediate", "--grib-dir", "d", "--cycle", "2026-09-13T15", "--out-dir", "o",
        ]))
    assert "none of --grid, --from-plan or --point" in str(refusal.value)


def test_intermediate_valid_times_read_the_header_not_the_name(tmp_path: Path) -> None:
    target = LatLonTarget(south=30.0, west=-100.0, dlat=0.5, dlon=0.5, nx=2, ny=2)
    for hour in (1, 0):
        hi.write_intermediate(
            tmp_path / f"anything-{hour}", [{"field": "TT", "units": "K", "description": "T",
                                             "level": 200100.0, "values": np.zeros((2, 2), np.float32)}],
            valid_time=datetime(2026, 9, 13, 15 + hour), forecast_hour=float(hour),
            map_source="t", target=target,
        )
    (tmp_path / "notes.txt").write_text("not a met file", encoding="utf-8")
    times = hi.intermediate_valid_times(tmp_path)
    assert [t for t, _ in times] == [datetime(2026, 9, 13, 15), datetime(2026, 9, 13, 16)]
    assert times[0][1].name == "anything-0"


# ---------------------------------------------------------------------------
# the boundary producer resolves through the one ladder
# ---------------------------------------------------------------------------
def _no_lbc_anywhere(monkeypatch) -> None:
    from woof.hex import engines

    for name in ("WOOF_HEX_RW_MPAS_LBC", "RW_MPAS_LBC", "WOOF_RW_MPAS_LBC"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", "")
    # Rung 3 too: an installed wheel bundles rw_mpas_lbc in woof/hex/libexec, and a
    # checkout with a staged libexec carries it there, so without this both tests
    # resolved the bundled binary instead of the rung they exercise.
    monkeypatch.setattr(engines, "bundled_candidate",
                        lambda spec: Path(__file__).parent / "no-bundled-libexec" / engines.executable_name(spec.name))
    monkeypatch.setattr(engines, "gpuwm_candidates", lambda spec: ())


def test_a_staged_lbc_engine_resolves_with_no_environment_variable(monkeypatch, tmp_path: Path) -> None:
    """``woof fetch-bridges`` alone opens the boundary leg.

    The chain's own resolver read one variable and PATH, so a box where the
    init leg had just resolved ``rw_mpas_init`` out of ``~/.woof/bridges``
    refused the lbc leg of the same chain for a file sitting beside it.
    """

    from woof.hex import engines
    from woof.hex.cycle.chain import resolve_lbc_engine

    staged = tmp_path / engines.executable_name("rw_mpas_lbc")
    staged.write_bytes(b"")
    staged.chmod(0o755)
    _no_lbc_anywhere(monkeypatch)
    monkeypatch.setattr(engines, "gpuwm_candidates", lambda spec: (staged,))

    assert resolve_lbc_engine(None) == staged.resolve()


def test_lbc_engine_absent_refusal_names_the_ladder_and_the_breakage(monkeypatch) -> None:
    from woof.hex.cycle.chain import CycleRefusal, resolve_lbc_engine

    _no_lbc_anywhere(monkeypatch)
    with pytest.raises(CycleRefusal) as caught:
        resolve_lbc_engine(None)
    message = str(caught.value)
    assert "rw_mpas_lbc not found" in message
    assert "--lbc-exe" in message
    assert "RW_MPAS_LBC" in message
    assert "woof fetch-bridges" in message
    assert "seven boundary rings" in message


def test_lbc_engine_named_but_missing_is_refused_not_skipped(tmp_path: Path) -> None:
    from woof.hex.cycle.chain import CycleRefusal, resolve_lbc_engine

    with pytest.raises(CycleRefusal) as caught:
        resolve_lbc_engine(tmp_path / "no-such-exe")
    assert "which is not a file" in str(caught.value)


def test_the_lbc_engine_is_a_row_of_the_ladder_the_doctor_reads() -> None:
    from woof.hex import engines

    assert engines.LBC in engines.ENGINES
    assert engines.LBC.name == "rw_mpas_lbc"


# ---------------------------------------------------------------------------
# the HRRR start: box over every mesh cell, east longitudes, soil sets, requests
# ---------------------------------------------------------------------------
def _mesh(path: Path, lat_deg, lon_deg) -> Path:
    from netCDF4 import Dataset

    with Dataset(path, "w") as dataset:
        dataset.createDimension("nCells", len(lat_deg))
        dataset.createVariable("latCell", "f8", ("nCells",))[:] = np.radians(np.asarray(lat_deg, dtype=float))
        dataset.createVariable("lonCell", "f8", ("nCells",))[:] = np.radians(np.asarray(lon_deg, dtype=float) % 360.0)
    return path


def test_longitude_conventions() -> None:
    assert hi.east_longitude(-80.98) == pytest.approx(279.02) and hi.east_longitude(360.0) == 0.0
    assert hi.west_longitude(279.02) == pytest.approx(-80.98) and hi.west_longitude(-80.98) == pytest.approx(-80.98)


def test_the_mesh_box_covers_every_cell_and_trims_only_where_the_source_ends(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("netCDF4")
    # A rectangle of cells near a fake source edge at 294.2 E (65.8 W); the box gives up its margin
    # there and keeps it on the other three sides.
    lat = [38.31, 47.78, 43.0, 40.0]
    lon = [-80.91, -66.10, -73.0, -70.0]
    grid = _mesh(tmp_path / "cut.grid.nc", lat, lon)

    class Grid:  # a plate carree "projection": native x, y in 0.03 degree steps from (35 N, 270 E)
        pass

    def limits(row):
        return Grid(), 1.0, (294.2 - 270.0) / 0.03, 1.0, (55.0 - 35.0) / 0.03

    monkeypatch.setattr(hi, "_source_window_limits", limits)
    monkeypatch.setattr(hi, "_native_positions",
                        lambda grid, la, lo: (((lo % 360.0) - 270.0) / 0.03, (la - 35.0) / 0.03))
    target, basis = hi.target_from_mesh(grid, hi.source_row("hrrr"), margin_km=30.0, spacing_deg=0.025)
    assert hi.cells_outside(target, np.array(lat), np.array(lon) % 360.0) == 0
    assert -180.0 <= target.west < 180.0 and target.west < -80.91 - 30.0 / 111.0
    assert target.south < 38.31 - 30.0 / 111.195 and target.north > 47.78 + 30.0 / 111.195
    assert target.east <= 294.2 - 360.0 and basis["trimmed_to_source"]["east"] > 0
    assert basis["trimmed_to_source"]["west"] == basis["trimmed_to_source"]["south"] == 0
    # A cell past the source edge is a refusal by count, never a silent cut.
    beyond = _mesh(tmp_path / "beyond.grid.nc", lat + [44.0], lon + [-65.4])
    with pytest.raises(IntermediateRefusal, match="1 of the 5 cells"):
        hi.target_from_mesh(beyond, hi.source_row("hrrr"), margin_km=30.0, spacing_deg=0.025)


def test_a_first_guess_short_of_the_mesh_is_refused_by_count(tmp_path: Path) -> None:
    pytest.importorskip("netCDF4")
    target = LatLonTarget(south=38.3, west=-80.8, dlat=0.025, dlon=0.025, nx=585, ny=381)
    met = tmp_path / "MET:2023-07-10_09"
    hi.write_intermediate(met, [{"field": "TT", "units": "K", "description": "T", "level": 200100.0,
                                 "values": np.full((target.ny, target.nx), 290.0, dtype=np.float32)}],
                          valid_time=datetime(2023, 7, 10, 9), forecast_hour=0.0, map_source="NCEP HRRR Model",
                          target=target)
    inside = _mesh(tmp_path / "inside.nc", [40.0, 45.0], [-75.0, -70.0])
    short = _mesh(tmp_path / "short.nc", [40.0, 45.0, 38.31], [-75.0, -70.0, -80.91])
    assert hi.uncovered_cells_refusal(met, inside) is None
    message = hi.uncovered_cells_refusal(met, short)
    assert message and message.startswith("1 of the 3 cells") and "--grid" in message


def test_the_soil_set_is_one_set_and_an_rrfs_suite_reads_the_nodes() -> None:
    row = hi.source_row("hrrr")
    pieces = {"soil_nodes_m": np.array([0.0, 0.01, 0.04, 0.10, 0.30, 0.60, 1.0, 1.6, 3.0]),
              "noah_midpoints_m": np.array([0.05, 0.25, 0.70, 1.50])}
    nodes = hi.soil_spec(row, pieces, "nodes")
    assert nodes["names"] == [] and nodes["midpoints"] == [] and nodes["preserve_nodes"]
    assert nodes["node_names"][1] == ["SOILT001", "SOILM001"]
    layers = hi.soil_spec(row, pieces, "layers")
    assert "node_names" not in layers and len(layers["names"]) == 4
    assert hi.first_guess_levels(row, "nodes") == (51, 9) and hi.first_guess_levels(row, "layers") == (51, 4)
    assert hi.soil_set_for_suite(row, "hrrrv5") == "nodes"
    assert hi.soil_set_for_suite(row, None) == "layers"
    assert hi.soil_set_for_suite(row, "convection_permitting") == "layers"


def test_mixing_ratios_and_numbers_are_held_nonnegative() -> None:
    row = hi.source_row("hrrr")
    assert hi._field_spec("QR", "QR", "kg kg-1", "Rain", "parabolic", row.nonnegative)["transform"] == "nonnegative"
    assert hi._field_spec("QNR", "QNR", "kg-1", "Rain n", "parabolic", row.nonnegative)["transform"] == "nonnegative"
    assert hi._field_spec("TT", "TT", "K", "T", "parabolic", row.nonnegative)["transform"] == "none"


def test_an_engine_that_cannot_ask_for_the_start_fields_is_refused_by_name() -> None:
    row = hi.source_row("hrrr")

    def old_producer(*, decoder, series, output, signals, cycle, window, workers, log, admissions=None,
                     analyzed_aerosol=False):
        return None

    def new_producer(*, decoder, series, output, signals, cycle, window, workers, log, admissions=None,
                     analyzed_aerosol=False, extras=()):
        return None

    with pytest.raises(IntermediateRefusal, match="extras"):
        hi._producer_requests(old_producer, row)
    assert hi._producer_requests(new_producer, row) == {"analyzed_aerosol": True,
                                                        "extras": ("QNC", "QNI", "QNR", "RH2")}
