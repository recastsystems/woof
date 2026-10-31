"""The preparation's decoded frame stream is priced before the download.

A chain that composes through the mapped engine stages every decoded valid
time in a scratch folder while it prepares, sized by the SOURCE grid.  The
disk check before the download priced the target grid only, so a GEM GDPS
48 hour window, whose stream is about 82 GB, was admitted onto a disk that
could not hold it; the engine then refused it itself, but only after the
whole download and the first valid time's decode.

These cells hold the table rows to the mapping documents they price and to
the engine's own figures, and hold ``run-plan``/``go`` to refusing, before
anything is fetched, a run whose frame stream does not fit, naming the
scratch folder and WOOF_COMPOSE_SCRATCH.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof import disk_budget, download_budget

ROOT = Path(__file__).resolve().parents[1]
AUTHORITIES = ROOT / "woof" / "authorities"
FIXTURES = Path(__file__).parent / "fixtures" / "download_budget"
GIB = 1024 ** 3

#: The engine's frame stream per valid time, read off frames.json after a
#: full compose of a real two-time fetch (gpuwm_mapped_engine built from
#: this tree, no atmospheric window, 2026-09-28).  GEM GDPS is the
#: figure the scratch-disk refusal measured on a 48 hour window: 210 layers
#: on 2400 x 1201 points.  The regional sources' figures are their whole
#: grids, the stream a window that cannot crop leaves.  hrrr-prs is the
#: one figure that is not the engine's own any more: it measured
#: 6,980,436,624 (458 layers) and 6a69b356f then added one surface plane,
#: vegetation_fraction, so it is that measurement plus one plane of the
#: grid, 1,905,141 x 8 bytes, until the compose is run again.
ENGINE_BYTES_PER_VALID_TIME = {
    "gem-gdps": 4_842_432_000,
    # 2026-09-28 stream plus the sea-ice layer bound from the GFS analysis since fix/ifs-open-data-sea-ice.
    "ecmwf-open-data": 838_897_920 + 1_038_240 * 8,
    # 2026-09-28 stream plus the sea-ice layer bound from the GFS analysis since fix/ecmwf-seaice-aifs-ens.
    "aifs": 755_838_720 + 1_038_240 * 8,
    "aigfs": 789_062_400,
    "aigefs": 789_062_400,
    "gefs": 422_110_080,
    "hrrr-prs": 6_980_436_624 + 1_905_141 * 8,
    "rrfs": 4_557_097_272,
    "rap": 233_561_968,
    "icon-eu": 1_483_689_960,
}

#: The canonical atmospheric fields the window crops, the engine's own list
#: (woof.ingest.atmospheric_window.CANONICAL_ATMOSPHERIC_FIELDS).
from woof.ingest.atmospheric_window import CANONICAL_ATMOSPHERIC_FIELDS  # noqa: E402


def _section() -> dict:
    return download_budget.table()["compose_scratch"]


def _mapping_layers(mapping: dict) -> tuple[int, int]:
    """(layers each valid time publishes above the soil, of those the window may crop).

    Every mapped field on the frame's grid that is not a soil field, each
    vertical field once per level of the mapping's ladder: what the engine
    writes (mapped-engine frames.rs).  A field marked ``dependency_only``
    is read by a derivation and not written, so it takes no layer.  The
    soil column is the source's own and a mapping does not state it
    (ICON-EU publishes 9 temperature and 8 water layers), so it is the
    row's ``soil_layers``.
    """

    levels = len(mapping["coordinates"]["vertical"]["levels"])
    total = windowed = 0
    for name, field in mapping["fields"].items():
        if "soil" in field["target_axes"] or field.get("dependency_only") is True:
            continue
        if "half_level" in field["target_axes"]:
            count = len(mapping["coordinates"]["vertical"]["interface_levels"])
        else:
            count = levels if "vertical" in field["target_axes"] else 1
        total += count
        if name in CANONICAL_ATMOSPHERIC_FIELDS and field["target_axes"] == ["vertical", "y", "x"]:
            windowed += count
    return total, windowed


def test_mapping_layers_counts_each_declared_interface():
    """Pricing an interface field as one plane admits a stream that cannot fit."""
    mapping = {
        "coordinates": {"vertical": {
            "levels": [3, 7, 11], "interface_levels": [2, 5, 9, 13],
        }},
        "fields": {
            "air_temperature": {"target_axes": ["vertical", "y", "x"]},
            "interface_height": {"target_axes": ["half_level", "y", "x"]},
            "surface_state": {"target_axes": ["y", "x"]},
            "soil_state": {"target_axes": ["soil", "y", "x"]},
            "raw_fraction": {"target_axes": ["vertical", "y", "x"], "dependency_only": True},
        },
    }
    assert _mapping_layers(mapping) == (8, 3)


def test_every_row_publishes_the_layers_its_mapping_declares():
    """A mapping that grows a field or a level cannot leave its row behind.

    Named breakage: a row that undercounts the stream admits a run the
    engine then refuses after the whole download.
    """

    section = _section()
    assert section["chains"] == ["prepared:staged"]
    for source, row in section["sources"].items():
        mapping = json.loads((AUTHORITIES / row["mapping"]).read_text(encoding="utf-8"))
        layers, windowed = _mapping_layers(mapping)
        assert row["layers_per_valid_time"] == layers + row["soil_layers"], source
        assert row["bytes_per_value"] == 8, source
        assert row["windowed_layers"] in (0, windowed), source


def test_the_rows_give_the_engines_own_figure_on_the_routes_it_was_run_on():
    rows = _section()["sources"]
    for source, measured in ENGINE_BYTES_PER_VALID_TIME.items():
        row = rows[source]
        assert (row["grid_points"] * row["layers_per_valid_time"] * row["bytes_per_value"]
                == measured), source
    estimate = download_budget.compose_scratch_estimate(
        None, chain="prepared:staged", source="gem-gdps", forcing_times=17)
    # The 48 hour window the refusal was measured on: 17 three-hourly times.
    # With no target to place, the whole stream is the estimate.  Only its
    # 12 surface and soil layers are certain whatever the target: since
    # A135 the atmospheric window crops a global source too, wherever the
    # target is clear of the ring's stored cut (this pin was bytes ==
    # min_bytes while a global source's window never cropped).
    assert estimate["bytes"] == estimate["max_bytes"] == 17 * 4_842_432_000
    assert estimate["min_bytes"] == 17 * 2_882_400 * 12 * 8
    assert round(estimate["bytes"] / 1e9) == 82


def test_a_regional_row_prices_the_grid_its_coverage_window_declares():
    """A regional row's window is its coverage window; a global row's is its ring's own axes."""
    from woof.source_adapters import source_coverage_window

    for source, row in _section()["sources"].items():
        window = source_coverage_window(source)
        if row.get("normalization"):
            # Composed on the window its normalization sizes around the target.
            assert "grid_points" not in row and row["windowed_layers"] == 0, source
            document = json.loads((AUTHORITIES / row["normalization"]).read_text(encoding="utf-8"))
            assert document["source_id"] == source, source
        elif row.get("global_axes"):
            # A whole ring: no coverage window, and the axes are the grid.
            axes = row["global_axes"]
            assert window is None and row["windowed_layers"] > 0, source
            assert axes["nx"] * axes["ny"] == row["grid_points"], source
            assert axes["nx"] * axes["step_degrees"] == pytest.approx(360.0), source
            assert (axes["ny"] - 1) * axes["step_degrees"] == pytest.approx(180.0), source
            assert (axes["longitude_first"], axes["latitude_first"]) == (-180.0, -90.0), source
        elif row["windowed_layers"]:
            assert window is not None and window.nx * window.ny == row["grid_points"], source
        else:
            assert window is None, source


def test_a_normalized_source_is_priced_on_the_window_its_normalization_computes():
    """ICON-D2's mesh is remapped onto a 0.02 degree window around the target before it is composed."""
    from woof.domain_wizard import _root_grid
    from woof.source_normalization import load_document, target_from_points

    row = _section()["sources"]["icon-d2"]
    layout = _regional_layout(150, 150, 2000.0, 51.0, 10.0)
    estimate = download_budget.compose_scratch_estimate(
        layout, chain="prepared:staged", source="icon-d2", forcing_times=4)
    projection = vars(layout.projection)
    latitude, longitude = _root_grid(
        {key: projection[key] for key in ("map_proj", "ref_lat", "ref_lon", "truelat1",
                                          "truelat2", "stand_lon")},
        150, 150, 2000.0).latlon_c()
    window = target_from_points(load_document(AUTHORITIES / row["normalization"]),
                                latitude.ravel().tolist(), longitude.ravel().tolist())
    per_time = window.nx * window.ny * row["layers_per_valid_time"] * 8
    assert estimate["bytes"] == estimate["min_bytes"] == 4 * per_time
    # A 300 km root at 51 N: about 4.2 by 2.7 degrees plus the halo each side.
    assert 230 <= window.nx <= 260 and 150 <= window.ny <= 175


def test_a_chain_that_composes_nothing_stages_no_stream_and_an_unknown_source_is_named():
    none = download_budget.compose_scratch_estimate(
        None, chain="prepared:hrrr", source="hrrr", forcing_times=19)
    assert none["bytes"] == 0 and none["min_bytes"] == 0 and none["composes"] is False
    unknown = download_budget.compose_scratch_estimate(
        None, chain="prepared:staged", source="not-a-source", forcing_times=3)
    assert unknown["bytes"] is None and "not-a-source" in unknown["basis"]
    assert unknown["composes"] is True


def _regional_layout(nx: int, ny: int, dx: float, lat: float, lon: float):
    projection = SimpleNamespace(map_proj="lambert", ref_lat=lat, ref_lon=lon,
                                 truelat1=38.5, truelat2=38.5, stand_lon=lon)
    from test_history_disk_layout import priced_run

    run = priced_run(nx, ny, 49, dx=dx, dy=dx, spec_bdy_width=5)
    domain = SimpleNamespace(grid_id=1, history_interval_s=3600.0, run=run)
    return SimpleNamespace(run_seconds=6 * 3600.0, restart_interval_s=0.0, domains=(domain,),
                           projection=projection, root=domain)


def test_a_regional_stream_is_priced_over_the_targets_footprint():
    """A 600 km root on the 3 km CONUS grid keeps about 200 x 200 of its 1799 x 1059 points."""
    row = _section()["sources"]["hrrr-prs"]
    layout = _regional_layout(200, 200, 3000.0, 38.5, -97.5)
    estimate = download_budget.compose_scratch_estimate(
        layout, chain="prepared:staged", source="hrrr-prs", forcing_times=7)
    fixed = row["grid_points"] * (row["layers_per_valid_time"] - row["windowed_layers"]) * 8
    assert estimate["min_bytes"] == 7 * fixed
    assert estimate["max_bytes"] == 7 * row["grid_points"] * row["layers_per_valid_time"] * 8
    kept = (estimate["per_valid_time"] - fixed) // (row["windowed_layers"] * 8)
    assert 200 * 200 <= kept <= 215 * 215
    assert estimate["min_bytes"] < estimate["bytes"] < estimate["max_bytes"]


def test_a_global_stream_is_priced_over_the_footprint_clear_of_its_stored_cut():
    """A135: the window crops a global source wherever the target is clear of the ring's cut.

    Named breakage: every global row priced its whole ring, so run-plan
    refused a GDPS 48 hour run on a disk that held its stream about 17
    times over.  The price is held to the window the preparation itself
    takes (atmospheric_window_for_grids on the ring's canonical axes).
    """
    import numpy as np

    from woof.domain_wizard import _root_grid
    from woof.ingest.atmospheric_window import atmospheric_window_for_grids
    from woof.ingest.grib import Era5Snapshot
    from woof.ingest.source_metadata import SourceSnapshotMetadata

    row = _section()["sources"]["gem-gdps"]
    axes = row["global_axes"]
    whole = row["grid_points"] * row["layers_per_valid_time"] * 8
    fixed = row["grid_points"] * (row["layers_per_valid_time"] - row["windowed_layers"]) * 8
    layout = _regional_layout(200, 200, 3000.0, 35.5, -97.5)
    clear = download_budget.compose_scratch_estimate(
        layout, chain="prepared:staged", source="gem-gdps", forcing_times=7)
    assert clear["min_bytes"] == 7 * fixed and clear["max_bytes"] == 7 * whole
    assert clear["min_bytes"] < clear["bytes"] < clear["max_bytes"] // 10
    kept = (clear["per_valid_time"] - fixed) // (row["windowed_layers"] * 8)
    step = axes["step_degrees"]
    metadata = SourceSnapshotMetadata(
        Era5Snapshot, axes["latitude_first"] + step * np.arange(axes["ny"]),
        axes["longitude_first"] + step * np.arange(axes["nx"]))
    projection = vars(layout.projection)
    grid = _root_grid({key: projection[key] for key in ("map_proj", "ref_lat", "ref_lon", "truelat1",
                                                        "truelat2", "stand_lon")}, 200, 200, 3000.0)
    window = atmospheric_window_for_grids(metadata, (grid,))
    assert window is not None
    taken = window.shape[0] * window.shape[1]
    # The price never counts fewer points than the window keeps, and
    # stays within its three-cell margin of it.
    assert taken <= kept <= (window.shape[0] + 6) * (window.shape[1] + 6)
    # The same root on the antimeridian reaches the stored cut: the ring
    # is kept whole there, and that is certain.
    on_cut = download_budget.compose_scratch_estimate(
        _regional_layout(200, 200, 3000.0, 52.0, 180.0), chain="prepared:staged",
        source="gem-gdps", forcing_times=7)
    assert on_cut["bytes"] == on_cut["min_bytes"] == on_cut["max_bytes"] == 7 * whole
    assert "stored longitude cut" in on_cut["basis"]
    grid = _root_grid({**{key: projection[key] for key in ("map_proj", "truelat1", "truelat2")},
                       "ref_lat": 52.0, "ref_lon": 180.0, "stand_lon": 180.0}, 200, 200, 3000.0)
    assert atmospheric_window_for_grids(metadata, (grid,)) is None


def _gdps_config(tmp_path: Path, *, on_cut: bool = False) -> Path:
    """The GFS fixture's 206 x 204 root, forced from GEM GDPS over its 24 hours.

    ``on_cut`` moves the root onto the antimeridian, where it reaches the
    GDPS ring's stored longitude cut: the preparation keeps that ring
    whole, so its whole stream (about 40 GiB) is staged.
    """

    text = (FIXTURES / "gfs-3km.toml").read_text(encoding="utf-8")
    text = text.replace("start_time = 2026-09-26T06:00:00", "start_time = 2026-09-26T00:00:00")
    text = text.replace('source = "gfs"', 'source = "gem-gdps"')
    text = text.replace('cycle = "2026-09-26T06"', 'cycle = "2026-09-26T00"')
    text = text.replace('area = "17.61,-116.54,53.24,-79.36"\n', "")
    assert 'source = "gem-gdps"' in text and "area =" not in text
    if on_cut:
        for name, value in (("ref_lat", "35.45"), ("ref_lon", "-97.95000000000005"),
                            ("stand_lon", "-97.95000000000005")):
            assert f"{name} = {value}\n" in text
            text = text.replace(f"{name} = {value}\n",
                                f"{name} = {'52.0' if name == 'ref_lat' else '180.0'}\n")
    config = tmp_path / "gdps-3km.toml"
    config.write_text(text, encoding="utf-8")
    return config


def _plan(tmp_path: Path, *, on_cut: bool = False) -> Path:
    from woof.runplan import PLAN_SCHEMA

    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"schema": PLAN_SCHEMA, "name": "gdps", "route": "prepared",
                                "config": {"path": str(_gdps_config(tmp_path, on_cut=on_cut))},
                                "output_root": str(tmp_path / "run")}), encoding="utf-8")
    return path


def test_the_review_prices_the_frame_stream_the_preparation_stages(tmp_path, capsys, monkeypatch):
    from woof.cli import build_parser
    from woof.runplan import run_plan_main

    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    monkeypatch.delenv("WOOF_COMPOSE_SCRATCH", raising=False)
    row = _section()["sources"]["gem-gdps"]
    fixed = row["grid_points"] * (row["layers_per_valid_time"] - row["windowed_layers"]) * 8
    disks = {}
    for on_cut in (False, True):
        folder = tmp_path / f"cut-{on_cut}"
        folder.mkdir()
        plan = _plan(folder, on_cut=on_cut)
        assert run_plan_main(build_parser().parse_args(["run-plan", "--estimate", str(plan)])) == 0
        disk = disks[on_cut] = json.loads(capsys.readouterr().out)["disk"]
        assert disk["bytes"] >= (disk["download_bytes"] + disk["preparation_bytes"]
                                 + disk["compose_scratch_bytes"])
        assert "compose scratch" not in disk["unpriced"]
    # Over Oklahoma the window keeps the root's footprint of the 198
    # atmospheric layers (a 618 km root: fewer than 60 x 60 of the ring's
    # 0.15 degree points), beside the 12 surface and soil layers whole.
    windowed = disks[False]["compose_scratch_bytes"] - 9 * fixed
    assert 0 < windowed < 9 * row["windowed_layers"] * 8 * 60 * 60
    # On the antimeridian the ring is kept whole: the stream the
    # scratch-disk refusal measured.
    assert disks[True]["compose_scratch_bytes"] == 9 * 4_842_432_000


def _execute(plan_path: Path, monkeypatch, free) -> tuple[list, list]:
    from woof import capabilities
    import woof.runplan as runplan
    from woof.runplan import EVENTS_FILENAME, EventStream, execute_plan, load_plan, read_events

    monkeypatch.setattr(capabilities, "require", lambda *args, **kwargs: None)
    fetched: list = []
    monkeypatch.setattr(runplan, "_run_fetch", lambda *args, **kwargs: fetched.append(args))
    monkeypatch.setattr(runplan, "_staged_chain", lambda *args, **kwargs: fetched.append("chain"))
    monkeypatch.setattr(disk_budget, "free_bytes", free)
    plan = load_plan(plan_path)
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    with EventStream(plan.run_dir / EVENTS_FILENAME, mirror=None) as events:
        execute_plan(plan, events=events)
    return read_events(plan.run_dir / EVENTS_FILENAME), fetched


def test_go_refuses_before_the_download_a_stream_its_disk_cannot_hold(tmp_path, capsys, monkeypatch):
    """Room for the download, the preparation and the forecast, not for the stream: refused, nothing fetched.

    The root sits on the antimeridian, where the GDPS ring is kept whole.
    """
    from woof.cli import build_parser
    from woof.runplan import run_plan_main

    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    monkeypatch.delenv("WOOF_COMPOSE_SCRATCH", raising=False)
    plan = _plan(tmp_path, on_cut=True)
    assert run_plan_main(build_parser().parse_args(["run-plan", "--estimate", str(plan)])) == 0
    disk = json.loads(capsys.readouterr().out)["disk"]
    forecast = disk["history_bytes"] + disk["checkpoint_bytes"] + disk["picture_bytes"]
    assert disk["compose_scratch_bytes"] > forecast
    room = disk["download_bytes"] + disk["preparation_bytes"] + forecast + GIB
    events, fetched = _execute(plan, monkeypatch, lambda path: room)
    failed = events[-1]
    assert failed["event"] == "failed" and fetched == []
    message = failed["message"]
    assert "Refused before the download" in message
    assert "decoded frame stream, 40.6 GiB (9 valid times of 4.5 GiB)" in message
    assert str((tmp_path / "run" / "chain").resolve()) in message
    assert "WOOF_COMPOSE_SCRATCH" in message


def test_a_scratch_folder_on_another_disk_is_measured_on_that_disk(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    scratch = tmp_path / "elsewhere"
    scratch.mkdir()
    monkeypatch.setenv("WOOF_COMPOSE_SCRATCH", str(scratch))
    monkeypatch.setattr(disk_budget, "same_disk",
                        lambda first, second: scratch not in (Path(first), Path(second)))
    plan = _plan(tmp_path, on_cut=True)

    def free(path):
        return 10 * GIB if Path(path) == scratch else 10 ** 15

    events, fetched = _execute(plan, monkeypatch, free)
    failed = events[-1]
    assert failed["event"] == "failed" and fetched == []
    assert f"in a scratch folder in {scratch}" in failed["message"]
    assert "has 10.0 GiB free, so the preparation would stop" in failed["message"]
    # With room there, the same run is not refused for its stream.
    again = tmp_path / "again"
    again.mkdir()
    events, fetched = _execute(_plan(again, on_cut=True), monkeypatch, lambda path: 10 ** 15)
    assert fetched == ["chain"]
    assert not any("frame stream" in str(event.get("message")) for event in events)


def test_a_scratch_variable_naming_no_folder_is_refused_before_the_download(tmp_path, monkeypatch):
    """The preparation refuses it by name only as it starts composing, after the whole download.

    Named breakage: free space was measured on the nearest existing parent
    of the named folder, the run was admitted, everything was fetched, and
    only then did the preparation stop on a folder that was never there.
    """
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    missing = tmp_path / "not-made"
    monkeypatch.setenv("WOOF_COMPOSE_SCRATCH", str(missing))
    events, fetched = _execute(_plan(tmp_path), monkeypatch, lambda path: 10 ** 15)
    failed = events[-1]
    assert failed["event"] == "failed" and fetched == []
    assert f"WOOF_COMPOSE_SCRATCH={missing} does not name an existing folder" in failed["message"]
    assert "Refused before any download or preparation, so nothing was spent" in failed["message"]
    assert failed["folders"] == [str(missing)]
    # Once the folder is there, the same run is measured there and admitted.
    missing.mkdir()
    again = tmp_path / "again"
    again.mkdir()
    events, fetched = _execute(_plan(again), monkeypatch, lambda path: 10 ** 15)
    assert fetched == ["chain"]
    assert not any("does not name an existing folder" in str(event.get("message")) for event in events)


def test_the_plan_measures_the_folder_the_preparation_stages_in(tmp_path, monkeypatch):
    """The check and the engine must look at one folder, or the check measures the wrong disk."""
    from woof.ingest.source_coverage import compose_scratch_folder
    from woof.mapped_composition import _compose_scratch_base
    from woof.runplan import _staged_prep_root

    prep_root = _staged_prep_root(tmp_path / "run")
    monkeypatch.delenv("WOOF_COMPOSE_SCRATCH", raising=False)
    assert compose_scratch_folder(prep_root) == _compose_scratch_base(prep_root)
    monkeypatch.setenv("WOOF_COMPOSE_SCRATCH", str(tmp_path))
    assert compose_scratch_folder(prep_root) == _compose_scratch_base(prep_root)


def test_a_regional_stream_that_may_not_fit_is_a_warning_not_a_refusal():
    row = download_budget.table()["compose_scratch"]["sources"]["hrrr-prs"]
    layout = _regional_layout(200, 200, 3000.0, 38.5, -97.5)
    p = disk_budget.projected_run_bytes(
        layout, keep_checkpoints=None,
        fetch={"source": "hrrr-prs", "cycle": "2026-09-27T00", "hours": 6},
        chain="prepared:staged", render=False)
    assert 0 < p["compose_scratch_min_bytes"] < p["compose_scratch_bytes"]
    base = p["download_bytes"] + p["preparation_bytes"]
    between = base + (p["compose_scratch_min_bytes"] + p["compose_scratch_bytes"]) // 2
    folder = Path("/scratch")
    assert disk_budget.disk_refusal(p, between, scratch_folder=folder) is None
    words = disk_budget.disk_warning(p, between, scratch_folder=folder)
    assert words.startswith("may not fit: ") and "WOOF_COMPOSE_SCRATCH" in words
    assert disk_budget.disk_warning(p, base + p["compose_scratch_bytes"], scratch_folder=folder) is None
    below = base + p["compose_scratch_min_bytes"] - 1
    assert "stages its decoded frame stream" in disk_budget.disk_refusal(
        p, below, scratch_folder=folder)
    assert row["windowed_layers"] > 0


def test_run_plan_says_a_stream_that_may_not_fit_before_the_download_and_goes_on(
        tmp_path, capsys, monkeypatch):
    """The one disk admission hands run-plan its warning: an event before the chain, not a refusal.

    The GDPS stream of a root on the antimeridian is certain whole (the
    ring is not windowed there); it is priced here as a windowed one would
    be, a tenth of it certain and the rest over the target's footprint, so
    the disk can hold the certain part and not the estimate.
    """
    from woof.cli import build_parser
    from woof.runplan import run_plan_main

    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    monkeypatch.delenv("WOOF_COMPOSE_SCRATCH", raising=False)
    priced = download_budget.compose_scratch_estimate

    def regional(*args, **kwargs):
        estimate = dict(priced(*args, **kwargs))
        if estimate["bytes"]:
            estimate["min_bytes"] = estimate["bytes"] // 10
        return estimate

    monkeypatch.setattr(download_budget, "compose_scratch_estimate", regional)
    plan = _plan(tmp_path, on_cut=True)
    assert run_plan_main(build_parser().parse_args(["run-plan", "--estimate", str(plan)])) == 0
    disk = json.loads(capsys.readouterr().out)["disk"]
    stream = disk["compose_scratch_bytes"]
    forecast = disk["history_bytes"] + disk["checkpoint_bytes"] + disk["picture_bytes"]
    floor = max(forecast, stream // 10)
    assert stream > floor
    room = disk["download_bytes"] + disk["preparation_bytes"] + floor + (stream - floor) // 2
    events, fetched = _execute(plan, monkeypatch, lambda path: room)
    assert fetched == ["chain"], "a stream that may not fit is not a refusal"
    warned = [event for event in events if event.get("code") == "compose_scratch_may_not_fit"]
    assert len(warned) == 1
    assert warned[0]["message"].startswith("May not fit: this run's preparation stages")
    assert warned[0]["folder"] == str((tmp_path / "run" / "chain").resolve())
    assert warned[0]["detail"].startswith("Frame stream: ")
    # Admitted: nothing refused it before the chain.  The chain is a double
    # here, so what the run does after it is not this test's.
    assert not any("Refused before" in str(event.get("message")) for event in events)
    assert events.index(warned[0]) < min(
        (index for index, event in enumerate(events) if event["event"] == "failed"),
        default=len(events))
