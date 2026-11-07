"""Candidate domain removal, reference integrity and WPS survivor identity."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import tomllib

import pytest

from woof import companion_domains as editor
from woof.hrrr_prepared_bundle import render_wps_namelist
from woof.namelist_import import parse_namelist_text
from woof.static import rust_bridge
from woof.toml_document import emit_experiment_toml

#: The tests below that cross the seam through the NATIVE editor refuse with
#: "Domain editing requires the native static-fields bridge" when that
#: library is not built.  The marker is per test rather than a module-level
#: pytestmark on purpose: the pure-geometry half of this deck answers the
#: same questions with no bridge at all, and a whole-module gate would retire
#: it on every box that has not built tools/rustwx.
_STATIC_BRIDGE_UNAVAILABLE = rust_bridge.unavailable_reason()

needs_static_bridge = pytest.mark.skipif(
    _STATIC_BRIDGE_UNAVAILABLE is not None,
    reason=("the native static-fields bridge is not built, so native domain "
            f"editing cannot run here: {_STATIC_BRIDGE_UNAVAILABLE}"))


def topology(parents=(0, 1, 2)):
    return {"domain": [{"grid_id": index + 1, "parent_id": parent}
                       for index, parent in enumerate(parents)]}


def configured_case(tmp_path, parents=(0, 1, 2)):
    raw = topology(parents)
    raw.update(experiment={"name": "domain-removal-proof", "start_time": datetime(2013, 5, 31, 18),
                           "run_seconds": 3600.0, "restart_interval_s": 0.0},
               shared={"nz": 8, "ztop": 16000.0, "p_top": 10000.0, "hybrid_opt": 2,
                       "eta_levels": [1.0, .9, .75, .6, .45, .3, .2, .1, 0.0]},
               projection={"map_proj": "lambert", "ref_lat": 35.0, "ref_lon": -98.0,
                           "truelat1": 30.0, "truelat2": 60.0, "stand_lon": -98.0},
               case_data={"forcing": ["forcing.grib"], "vtable": "Vtable.ERA5",
                          "wps_namelist": "namelist.wps", "geog_root": "GEOG",
                          "forcing_interval_s": 21600.0, "output_domain": 1,
                          "sfcp_to_sfcp": True, "output_title": "Domain removal metadata proof"})
    for row in raw["domain"]:
        root = row["parent_id"] == 0
        row.update(nx=180 if root else 72, ny=160 if root else 72,
                   i_parent_start=1 if root else 20, j_parent_start=1 if root else 20,
                   parent_grid_ratio=1 if root else 3, parent_time_step_ratio=1 if root else 3,
                   specified=root, nested=not root, history_interval_s=60.0)
        if root:
            row.update(dx=12000.0, time_step=60)
    source = tmp_path / "source.toml"
    source.write_text(emit_experiment_toml(raw), encoding="utf-8")
    exp = editor._build(raw, source)
    wps = render_wps_namelist(exp).replace("geog_data_res     = 'default',", "geog_data_res = " +
        ", ".join(repr(value) for value in ("5m", "default", "modis_lai+default", "5m+modis_lai")[:len(parents)]) + ",")
    (tmp_path / "namelist.wps").write_text(wps, encoding="utf-8")
    return source, raw


def request_for(source, grid_id, include_children=False):
    return {"schema": editor.REQUEST_SCHEMA, "config_path": str(source),
            "expected_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output_path": str(source.parent / "candidate.toml"),
            "action": {"kind": "remove_nest", "grid_id": grid_id, "include_children": include_children}}


def edit_action(source, action):
    request = request_for(source, action["grid_id"])
    request["action"] = action
    result = editor.edit_configuration(request)
    candidate = tomllib.loads(Path(result["config_path"]).read_text())
    return result, candidate


def test_move_root_translates_native_tree_and_preserves_domain_settings(tmp_path):
    source, original = configured_case(tmp_path)
    original_bytes = source.read_bytes()
    old = editor.native_domain_outlines(editor._build(original, source))
    result, candidate = edit_action(source, {"kind": "move_domain", "grid_id": 1,
                                            "latitude": 38.0, "longitude": -94.0})
    assert candidate["domain"] == original["domain"]
    assert candidate["shared"] == original["shared"]
    assert candidate["projection"] == dict(original["projection"], ref_lat=38.0, ref_lon=-94.0)
    outlines = result["configuration"]["domains"]
    assert outlines[0]["center_latlon"] == pytest.approx([38.0, -94.0], abs=1e-8)
    assert all(before["center_latlon"] != after["center_latlon"] for before, after in zip(old, outlines))
    assert source.read_bytes() == original_bytes
    assert result["forecast_started"] is False


@pytest.mark.parametrize("grid_id", [1, 2, 3])
def test_existing_domain_output_edit_round_trips_query_and_preserves_other_settings(tmp_path, grid_id):
    from woof.companion_query import inspect_configuration
    source, original = configured_case(tmp_path)
    before = source.read_bytes()
    result, candidate = edit_action(source, {"kind": "set_output", "grid_id": grid_id,
        "history_interval_s": 300., "restart_interval_s": None})
    expected = deepcopy(original)
    expected["domain"][grid_id - 1]["history_interval_s"] = 300.
    assert candidate["domain"] == expected["domain"]
    assert candidate["shared"] == original["shared"]
    assert candidate["experiment"] == original["experiment"]
    assert candidate["projection"] == original["projection"]
    queried = inspect_configuration(Path(result["config_path"]))
    expected_intervals = [row["history_interval_s"] for row in expected["domain"]]
    assert [row["history_interval_s"] for row in queried["domains"]] == expected_intervals
    assert [row["history_interval_s"] for row in result["configuration"]["domains"]] == expected_intervals
    assert source.read_bytes() == before
    assert result["forecast_started"] is False


@pytest.mark.parametrize("checkpoint", [0., 600.])
def test_output_edit_exposes_global_restart_policy_without_changing_other_domains(tmp_path, checkpoint):
    source, original = configured_case(tmp_path)
    result, candidate = edit_action(source, {"kind": "set_output", "grid_id": 2,
        "history_interval_s": 300., "restart_interval_s": checkpoint})
    assert candidate["experiment"] == dict(original["experiment"], restart_interval_s=checkpoint)
    assert result["configuration"]["experiment"]["restart_interval_s"] == checkpoint
    assert all(row["run"]["restart_interval_s"] == checkpoint for row in result["domains"])
    assert candidate["domain"][0] == original["domain"][0]
    assert candidate["domain"][2] == original["domain"][2]


@pytest.mark.parametrize("grid_id,history,restart,match", [
    (1, 0., None, "history_interval_s"),
    (1, 61., None, "whole number"),
    (2, 21., None, "whole number"),
    (1, 300., 61., "whole number"),
    (1, 300., -1., "non-negative"),
    (1, True, None, "finite number"),
])
def test_invalid_output_cadence_uses_engine_admission_and_publishes_nothing(tmp_path, grid_id, history, restart, match):
    source, _ = configured_case(tmp_path)
    before = source.read_bytes()
    with pytest.raises(ValueError, match=match):
        edit_action(source, {"kind": "set_output", "grid_id": grid_id,
                            "history_interval_s": history, "restart_interval_s": restart})
    assert source.read_bytes() == before
    assert not (source.parent / "candidate.toml").exists()
    assert not (source.parent / "candidate.namelist.wps").exists()


@pytest.mark.parametrize("field", ["history_interval_s", "restart_interval_s"])
def test_output_edit_rejects_subsecond_file_aliasing_even_on_a_legal_model_step(tmp_path, field):
    source, raw = configured_case(tmp_path)
    raw["domain"][0].update(time_step=0, time_step_fract_num=1, time_step_fract_den=2)
    source.write_text(emit_experiment_toml(raw))
    action = {"kind": "set_output", "grid_id": 1, "history_interval_s": 60., field: .5}
    with pytest.raises(ValueError, match="whole number of seconds"):
        edit_action(source, action)
    assert not (source.parent / "candidate.toml").exists()


def test_output_edit_preserves_existing_weather_tracker_cadence_admission(tmp_path, monkeypatch):
    original = Path(__file__).resolve().parents[1] / "configs/moving_nest_20110427_follow_2km.toml"
    source = tmp_path / "following.toml"
    monkeypatch.setenv("WOOF_DEMO_20110427_ROOT", str(tmp_path))
    source.write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="whole multiple of history_interval_s"):
        edit_action(source, {"kind": "set_output", "grid_id": 1, "history_interval_s": 1800.})
    assert not (source.parent / "candidate.toml").exists()


@pytest.mark.parametrize("mode", ["off", "auto", "on"])
def test_global_tile_mode_preserves_explicit_tile_controls_and_reports_saved_mode(tmp_path, mode):
    from woof.companion_query import inspect_configuration
    source, raw = configured_case(tmp_path)
    raw["tiles"] = {"mode": "on", "vram_budget_bytes": 1024**3, "host_budget_bytes": 4*1024**3,
                    "store": "host", "write_mode": "ring", "pipeline": "prefetch"}
    if mode == "on":
        raw["tiles"].update(tile_nx=32, tile_ny=32, nbuffers=2)
    source.write_text(emit_experiment_toml(raw))
    before = source.read_bytes()
    request = request_for(source, 1)
    request["action"] = {"kind": "set_tiles", "mode": mode}
    result = editor.edit_configuration(request)
    candidate = tomllib.loads(Path(result["config_path"]).read_text())
    assert candidate["tiles"] == dict(raw["tiles"], mode=mode)
    assert candidate["domain"] == raw["domain"] and candidate["shared"] == raw["shared"]
    assert result["configuration"]["tiles"]["mode"] == mode
    assert inspect_configuration(Path(result["config_path"]))["tiles"]["mode"] == mode
    assert result["validation"]["forecast_or_memory_admission"] == "not_run"
    assert source.read_bytes() == before


def test_tile_query_exposes_engine_off_default_and_invalid_mode_is_not_published(tmp_path):
    from woof.companion_query import inspect_configuration
    source, _ = configured_case(tmp_path)
    assert inspect_configuration(source)["tiles"]["mode"] == "off"
    request = request_for(source, 1)
    request["action"] = {"kind": "set_tiles", "mode": "automatic"}
    with pytest.raises(ValueError, match="off, auto or on"):
        editor.edit_configuration(request)
    assert not (source.parent / "candidate.toml").exists()


@pytest.mark.parametrize("mode", ["off", "auto"])
def test_changing_explicit_tile_pins_keeps_engine_refusal_and_original_settings(tmp_path, mode):
    source, raw = configured_case(tmp_path)
    raw["tiles"] = {"mode": "on", "tile_nx": 32, "tile_ny": 32}
    source.write_text(emit_experiment_toml(raw))
    before = source.read_bytes()
    request = request_for(source, 1)
    request["action"] = {"kind": "set_tiles", "mode": mode}
    with pytest.raises(ValueError, match="surface that is off must be empty|while mode = 'auto'"):
        editor.edit_configuration(request)
    assert source.read_bytes() == before
    assert not (source.parent / "candidate.toml").exists()


def test_move_nested_grid_clamps_to_parent_and_keeps_descendants(tmp_path):
    source, original = configured_case(tmp_path)
    result, candidate = edit_action(source, {"kind": "move_domain", "grid_id": 2,
                                            "latitude": 48.0, "longitude": -70.0})
    exp = editor._build(candidate, Path(result["config_path"]))
    child, parent = exp.domain(2), exp.domain(1)
    assert child.i_parent_start == parent.run.nx - child.run.nx // 3 - 10 + 1
    assert child.j_parent_start == parent.run.ny - child.run.ny // 3 - 10 + 1
    assert candidate["domain"][0] == original["domain"][0]
    assert candidate["domain"][2] == original["domain"][2]
    assert candidate["shared"] == original["shared"]


@pytest.mark.parametrize("ratio,nx,ny,expected", [
    (3, 164, 218, (165, 219)),  # The saved southern-hemisphere Add nest request.
    (3, 163, 217, (162, 216)), (3, 165, 219, (165, 219)),
    (2, 165, 219, (166, 220)), (4, 166, 218, (168, 220)),
])
def test_add_drawn_nest_saves_ratio_aligned_mass_cells_and_native_placement(
        tmp_path, ratio, nx, ny, expected):
    source, raw = configured_case(tmp_path, (0,))
    raw["domain"][0].update(nx=802, ny=537)
    raw["projection"] = dict(map_proj="lambert", ref_lat=-46.719424874125096,
        ref_lon=-9.895900721668568, truelat1=-30.6, truelat2=-50.6,
        stand_lon=-37.20916030534397)
    source.write_text(emit_experiment_toml(raw))
    before = source.read_bytes()
    target = (-40.5992279334933, -37.20916030534397)
    request = request_for(source, 1)
    request["action"] = {"kind": "add_nest", "parent_id": 1, "nx": nx, "ny": ny,
        "parent_grid_ratio": ratio, "parent_time_step_ratio": 3,
        "history_interval_s": 300.0,
        "placement": {"kind": "center", "latitude": target[0], "longitude": target[1]}}
    result = editor.edit_configuration(request)
    candidate = tomllib.loads(Path(result["config_path"]).read_text())
    child = candidate["domain"][1]
    assert (child["nx"], child["ny"]) == expected
    assert child["parent_grid_ratio"] == ratio and child["parent_time_step_ratio"] == 3
    assert child["history_interval_s"] == 300.0
    assert candidate["domain"][0] == raw["domain"][0]
    assert candidate["shared"] == raw["shared"]
    assert candidate["projection"] == raw["projection"]
    assert candidate["experiment"] == raw["experiment"]
    exp = editor._build(candidate, Path(result["config_path"]))
    assert exp.domain(2).run.dx == 12000.0 / ratio
    assert exp.domain(2).run.dt == 20.0
    bridge, grids = editor._native_grids(exp)
    actual_center = result["configuration"]["domains"][1]["center_latlon"]
    actual, requested = editor._transform(bridge, grids[1], 1, [actual_center, target])
    assert abs(actual[0] - requested[0]) <= .5 + 1e-8
    assert abs(actual[1] - requested[1]) <= .5 + 1e-8
    wps = parse_namelist_text(Path(result["wps_path"]).read_text())["geogrid"]
    assert wps["e_we"] == [803, expected[0] + 1]
    assert wps["e_sn"] == [538, expected[1] + 1]
    assert source.read_bytes() == before
    assert result["forecast_started"] is False


@pytest.mark.parametrize("overrides,match", [
    ({"nx": 0}, "integer"), ({"nx": True}, "integer"),
    ({"nx": 164.5}, "integer"), ({"parent_grid_ratio": 0}, "integer"),
    ({"parent_grid_ratio": 1}, "must be >= 2"),
    ({"parent_time_step_ratio": 1}, "must be >= 2"),
    ({"nx": 3000}, "parent-row clearance"),
    ({"history_interval_s": 301.0}, "whole number"),
])
def test_add_nest_snapping_preserves_engine_refusals_and_publishes_nothing(tmp_path, overrides, match):
    source, _ = configured_case(tmp_path, (0,))
    before = source.read_bytes()
    request = request_for(source, 1)
    request["action"] = {"kind": "add_nest", "parent_id": 1, "nx": 164, "ny": 218,
        "parent_grid_ratio": 3, "parent_time_step_ratio": 3,
        "history_interval_s": 300.0,
        "placement": {"kind": "parent_cells", "i_parent_start": 20, "j_parent_start": 20},
        **overrides}
    with pytest.raises(ValueError, match=match):
        editor.edit_configuration(request)
    assert source.read_bytes() == before
    assert not (source.parent / "candidate.toml").exists()
    assert not (source.parent / "candidate.namelist.wps").exists()


@pytest.mark.parametrize("ratio,boundary,relax,expected", [
    (3, 5, 4, 12), (2, 5, 4, 12), (5, 5, 4, 15),
    (3, 8, 4, 18), (3, 9, 8, 21),
])
def test_tiny_add_nest_uses_the_same_boundary_and_stencil_minimum_as_resize(
        tmp_path, ratio, boundary, relax, expected):
    source, raw = configured_case(tmp_path, (0,))
    raw["experiment"]["spec_bdy_width"] = boundary
    raw["shared"]["relax_zone"] = relax
    source.write_text(emit_experiment_toml(raw))
    before = source.read_bytes()
    request = request_for(source, 1)
    request["action"] = {"kind": "add_nest", "parent_id": 1, "nx": 1, "ny": 2,
        "parent_grid_ratio": ratio, "parent_time_step_ratio": 3,
        "history_interval_s": 300.0,
        "placement": {"kind": "parent_cells", "i_parent_start": 20, "j_parent_start": 20}}
    result = editor.edit_configuration(request)
    candidate = tomllib.loads(Path(result["config_path"]).read_text())
    child = candidate["domain"][1]
    assert (child["nx"], child["ny"]) == (expected, expected)
    assert (child["i_parent_start"], child["j_parent_start"]) == (20, 20)
    assert candidate["domain"][0] == raw["domain"][0]
    assert candidate["shared"] == raw["shared"]
    assert candidate["projection"] == raw["projection"]
    assert candidate["experiment"] == raw["experiment"]
    assert source.read_bytes() == before
    exp = editor._build(candidate, Path(result["config_path"]))
    editor._apply(candidate, {"kind": "resize_domain_cells", "grid_id": 2,
        "handle": "ne", "nx": 1, "ny": 2}, Path(result["config_path"]))
    assert (candidate["domain"][1]["nx"], candidate["domain"][1]["ny"]) == (expected, expected)
    assert exp.domain(2).run.dx == 12000.0 / ratio


@pytest.mark.parametrize("kind", ["resize_domain_cells", "resize_domain_edges", "resize_domain"])
@pytest.mark.parametrize("grid_id", [1, 2])
def test_resizing_wps_dimension_aliases_matches_equivalent_mass_counts(tmp_path, kind, grid_id):
    outputs = []
    for representation in ("mass", "wps"):
        folder = tmp_path / representation
        folder.mkdir()
        source, raw = configured_case(folder, (0, 1))
        if representation == "wps":
            for row in raw["domain"]:
                row["e_we"] = row.pop("nx") + 1
                row["e_sn"] = row.pop("ny") + 1
            source.write_text(emit_experiment_toml(raw))
        before = source.read_bytes()
        if kind == "resize_domain_cells":
            domain = editor._build(raw, source).domain(grid_id)
            action = {"kind": kind, "grid_id": grid_id, "handle": "e",
                      "nx": domain.run.nx + 12, "ny": domain.run.ny}
        elif kind == "resize_domain_edges":
            action = native_edge_action(raw, source, grid_id, "ne", di=12, dj=9)
        else:
            action = {"kind": kind, "grid_id": grid_id,
                      "bounds": {"south": 34., "west": -99., "north": 36., "east": -97.}}
        result, candidate = edit_action(source, action)
        exp = editor._build(candidate, Path(result["config_path"]))
        changed = candidate["domain"][grid_id-1]
        if representation == "wps":
            assert changed["e_we"] == changed["nx"] + 1
            assert changed["e_sn"] == changed["ny"] + 1
        assert candidate["shared"] == raw["shared"]
        assert candidate["experiment"] == raw["experiment"]
        assert source.read_bytes() == before
        wps = parse_namelist_text(Path(result["wps_path"]).read_text())["geogrid"]
        outputs.append((candidate["projection"],
            [(d.run.nx, d.run.ny, d.i_parent_start, d.j_parent_start) for d in exp.domains],
            wps["e_we"], wps["e_sn"], result["configuration"]["domains"]))
    assert outputs[0] == outputs[1]


def test_resize_root_keeps_child_sizes_and_clamps_their_placement(tmp_path):
    source, original = configured_case(tmp_path)
    result, candidate = edit_action(source, {"kind": "resize_domain", "grid_id": 1,
        "bounds": {"south": 34.99, "west": -98.01, "north": 35.01, "east": -97.99}})
    exp = editor._build(candidate, Path(result["config_path"]))
    assert (exp.domain(1).run.nx, exp.domain(1).run.ny) == (44, 44)
    assert (exp.domain(2).i_parent_start, exp.domain(2).j_parent_start) == (11, 11)
    for before, after in zip(original["domain"], candidate["domain"]):
        assert {k: v for k, v in before.items() if k not in ("nx", "ny", "i_parent_start", "j_parent_start")} == {
            k: v for k, v in after.items() if k not in ("nx", "ny", "i_parent_start", "j_parent_start")}
    assert [(row["nx"], row["ny"]) for row in candidate["domain"][1:]] == [(72, 72), (72, 72)]
    assert candidate["shared"] == original["shared"]
    wps = parse_namelist_text(Path(result["wps_path"]).read_text())
    assert wps["geogrid"]["e_we"] == [45, 73, 73]
    assert wps["geogrid"]["geog_data_res"] == ["5m", "default", "modis_lai+default"]


def test_resize_nested_grid_snaps_ratio_and_stops_at_parent_boundary(tmp_path):
    source, original = configured_case(tmp_path)
    result, candidate = edit_action(source, {"kind": "resize_domain", "grid_id": 2,
        "bounds": {"south": 20.0, "west": -120.0, "north": 50.0, "east": -70.0}})
    exp = editor._build(candidate, Path(result["config_path"]))
    assert (exp.domain(2).run.nx, exp.domain(2).run.ny) == (480, 420)
    assert (exp.domain(2).i_parent_start, exp.domain(2).j_parent_start) == (11, 11)
    assert candidate["domain"][0] == original["domain"][0]
    assert candidate["domain"][2] == original["domain"][2]
    assert result["configuration"]["geometry_backend"] == "rust-static-fields"


def native_edge_action(raw, path, grid_id, handle, di=0, dj=0):
    bridge, grids = editor._native_grids(editor._build(raw, path))
    grid = grids[grid_id]
    start = (grid.e_we / 2, grid.e_sn / 2)
    points = editor._transform(bridge, grid, 0, [start, (start[0] + di, start[1] + dj)])
    return {"kind": "resize_domain_edges", "grid_id": grid_id, "handle": handle,
            "start": dict(zip(("latitude", "longitude"), points[0])),
            "end": dict(zip(("latitude", "longitude"), points[1]))}


@pytest.mark.parametrize("grid_id", [1, 2, 3])
@pytest.mark.parametrize("handle", ["nw", "n", "ne", "e", "se", "s", "sw", "w"])
def test_zero_pointer_resize_preserves_exact_native_configuration(tmp_path, grid_id, handle):
    source, raw = configured_case(tmp_path)
    original = deepcopy(raw)
    editor._apply(raw, native_edge_action(raw, source, grid_id, handle), source)
    assert raw == original


@pytest.mark.parametrize("grid_id,handle,di,expected_nx,expected_start", [
    (1, "e", 12., 192, 1),
    (2, "e", 8., 81, 20),
    (2, "w", 8., 63, 23),
    (2, "e", 1000., 453, 20),
    (2, "w", -1000., 99, 11),
])
def test_native_edge_resize_snaps_and_clamps_without_moving_the_opposite_anchor(tmp_path, grid_id, handle, di, expected_nx, expected_start):
    source, original = configured_case(tmp_path)
    action = native_edge_action(original, source, grid_id, handle, di=di, dj=7.)
    result, candidate = edit_action(source, action)
    before = editor._build(original, source)
    after = editor._build(candidate, Path(result["config_path"]))
    domain = after.domain(grid_id)
    assert (domain.run.nx, domain.run.ny, domain.i_parent_start) == (expected_nx, before.domain(grid_id).run.ny, expected_start)
    old_bridge, old_grids = editor._native_grids(before)
    new_bridge, new_grids = editor._native_grids(after)
    old_grid, new_grid = old_grids[grid_id], new_grids[grid_id]
    old_x = .5 if handle == "e" else before.domain(grid_id).run.nx + .5
    new_x = .5 if handle == "e" else domain.run.nx + .5
    fixed_before = editor._transform(old_bridge, old_grid, 0, [(old_x, .5), (old_x, domain.run.ny + .5)])
    fixed_after = editor._transform(new_bridge, new_grid, 0, [(new_x, .5), (new_x, domain.run.ny + .5)])
    for old, new in zip(fixed_before, fixed_after):
        assert new == pytest.approx(old, abs=1e-8)
    assert candidate["shared"] == original["shared"]
    for old, new in zip(original["domain"], candidate["domain"]):
        assert {k: v for k, v in old.items() if k not in ("nx", "ny", "i_parent_start", "j_parent_start")} == {
            k: v for k, v in new.items() if k not in ("nx", "ny", "i_parent_start", "j_parent_start")}


def test_preview_cell_resize_grows_both_axes_for_the_saved_southern_ne_drag(tmp_path):
    """The exact SH projection/gesture collapsed 464x372 into1121x11."""
    source, raw = configured_case(tmp_path, (0,))
    raw["domain"][0].update(nx=464, ny=372)
    raw["projection"] = dict(map_proj="lambert", ref_lat=-40.5992279334933,
        ref_lon=-37.20916030534397, truelat1=-30.6, truelat2=-50.6,
        stand_lon=-37.20916030534397)
    source.write_text(emit_experiment_toml(raw))
    old = deepcopy(raw)
    exp = editor._build(raw, source)
    bridge, grids = editor._native_grids(exp)
    grid = grids[1]
    start = {"latitude": -15.882002819497671, "longitude": 5.50403489663114}
    end = {"latitude": 4.520443921526862, "longitude": 87.41286804820913}
    native = editor._transform(bridge, grid, 1,
        [(start["latitude"], start["longitude"]), (end["latitude"], end["longitude"])])
    assert native[1][0] - native[0][0] == pytest.approx(657.0959317655961)
    assert native[1][1] - native[0][1] == pytest.approx(-584.8133361646708)
    # Preserve the old action's public meaning; it really is a native-point
    # delta, which is why the map's rectangular preview must use another door.
    legacy = deepcopy(raw)
    editor._apply(legacy, {"kind": "resize_domain_edges", "grid_id": 1,
                          "handle": "ne", "start": start, "end": end}, source)
    assert (legacy["domain"][0]["nx"], legacy["domain"][0]["ny"]) == (1121, 11)
    # Actual flat MapLibre preview ratios for this same saved gesture and
    # native perimeter:895x502, independently pinned by the Companion test.
    result, candidate = edit_action(source, {"kind": "resize_domain_cells", "grid_id": 1,
                                            "handle": "ne", "nx": 895, "ny": 502})
    assert (candidate["domain"][0]["nx"], candidate["domain"][0]["ny"]) == (895, 502)
    after = editor._build(candidate, Path(result["config_path"]))
    after_bridge, after_grids = editor._native_grids(after)
    old_anchor = editor._transform(bridge, grid, 0, [(.5, .5)])[0]
    new_anchor = editor._transform(after_bridge, after_grids[1], 0, [(.5, .5)])[0]
    assert new_anchor == pytest.approx(old_anchor, abs=1e-8)
    assert candidate["shared"] == old["shared"]
    assert {k:v for k,v in candidate["domain"][0].items() if k not in ("nx", "ny")} == {
        k:v for k,v in old["domain"][0].items() if k not in ("nx", "ny")}
    assert "resize_domain_cells" in editor.capabilities()["actions"]


@pytest.mark.parametrize("grid_id,handle,requested_nx,expected_nx,expected_start", [
    (1, "e", 192, 192, 1), (2, "e", 80, 81, 20),
    (2, "w", 64, 63, 23), (2, "e", 2000, 453, 20),
    (2, "w", 2000, 99, 11),
])
def test_preview_cell_resize_uses_existing_native_quantization_and_opposite_anchor(
        tmp_path, grid_id, handle, requested_nx, expected_nx, expected_start):
    source, original = configured_case(tmp_path)
    ny = original["domain"][grid_id-1]["ny"]
    result, candidate = edit_action(source, {"kind": "resize_domain_cells", "grid_id": grid_id,
        "handle": handle, "nx": requested_nx, "ny": ny})
    before = editor._build(original, source)
    after = editor._build(candidate, Path(result["config_path"]))
    domain = after.domain(grid_id)
    assert (domain.run.nx, domain.run.ny, domain.i_parent_start) == (expected_nx, ny, expected_start)
    old_bridge, old_grids = editor._native_grids(before)
    new_bridge, new_grids = editor._native_grids(after)
    old_x = .5 if handle == "e" else before.domain(grid_id).run.nx + .5
    new_x = .5 if handle == "e" else domain.run.nx + .5
    for old_point, new_point in zip(
        editor._transform(old_bridge, old_grids[grid_id], 0, [(old_x, .5), (old_x, ny+.5)]),
        editor._transform(new_bridge, new_grids[grid_id], 0, [(new_x, .5), (new_x, ny+.5)])):
        assert new_point == pytest.approx(old_point, abs=1e-8)
    assert candidate["shared"] == original["shared"]


@pytest.mark.parametrize("patch", [dict(nx=True), dict(nx=0), dict(ny=1.5),
                                   dict(handle="north"), dict(handle="e", ny=161)])
def test_invalid_preview_dimensions_publish_nothing(tmp_path, patch):
    source, _ = configured_case(tmp_path)
    before = source.read_bytes()
    action = {"kind": "resize_domain_cells", "grid_id": 1, "handle": "ne", "nx": 200, "ny": 180}
    action.update(patch)
    with pytest.raises(ValueError):
        edit_action(source, action)
    assert source.read_bytes() == before
    assert not (source.parent / "candidate.toml").exists()


@pytest.mark.parametrize("action", [
    {"kind": "resize_domain", "grid_id": 1, "bounds": {"south": 40, "north": 30, "west": -100, "east": -90}},
    {"kind": "resize_domain", "grid_id": 1, "bounds": {"south": 30, "north": 40, "west": -100, "east": 100}},
    {"kind": "move_domain", "grid_id": 1, "latitude": 90, "longitude": -100},
])
def test_invalid_drag_leaves_source_and_candidate_untouched(tmp_path, action):
    source, _ = configured_case(tmp_path)
    before = source.read_bytes()
    with pytest.raises(ValueError):
        edit_action(source, action)
    assert source.read_bytes() == before
    assert not (source.parent / "candidate.toml").exists()
    assert not (source.parent / "candidate.namelist.wps").exists()


def test_leaf_removal_publishes_native_candidate_without_changing_original(tmp_path):
    source, original = configured_case(tmp_path)
    before_config = source.read_bytes()
    before_wps = (tmp_path / "namelist.wps").read_bytes()
    result = editor.edit_configuration(request_for(source, 3))
    assert result["created"] is True and result["forecast_started"] is False
    assert [row["grid_id"] for row in result["domains"]] == [1, 2]
    assert [row["parent_id"] for row in result["configuration"]["domains"]] == [0, 1]
    assert all(row["perimeter_latlon"] for row in result["configuration"]["domains"])
    candidate = tomllib.loads(Path(result["config_path"]).read_text())
    assert candidate["domain"] == original["domain"][:2]
    assert candidate["case_data"]["wps_namelist"] == result["wps_path"]
    wps = parse_namelist_text(Path(result["wps_path"]).read_text())
    assert wps["share"]["max_dom"] == [2]
    assert wps["geogrid"]["geog_data_res"] == ["5m", "default"]
    assert source.read_bytes() == before_config
    assert (tmp_path / "namelist.wps").read_bytes() == before_wps
    assert json.loads(Path(result["receipt_path"]).read_text())["config_sha256"] == hashlib.sha256(Path(result["config_path"]).read_bytes()).hexdigest()


def test_subtree_requires_explicit_inclusion_and_then_preserves_root(tmp_path):
    source, _ = configured_case(tmp_path)
    with pytest.raises(ValueError, match="explicitly include its children"):
        editor.edit_configuration(request_for(source, 2))
    assert not (tmp_path / "candidate.toml").exists()
    assert not (tmp_path / "candidate.namelist.wps").exists()
    result = editor.edit_configuration(request_for(source, 2, True))
    assert [row["grid_id"] for row in result["domains"]] == [1]
    assert parse_namelist_text(Path(result["wps_path"]).read_text())["geogrid"]["geog_data_res"] == ["5m"]


@pytest.mark.parametrize("grid_id,include_children,match", [(1, True, "root domain"), (9, False, "no domain"), (3, 1, "boolean")])
def test_invalid_removal_preserves_the_tree(grid_id, include_children, match):
    raw = topology()
    before = deepcopy(raw)
    with pytest.raises(ValueError, match=match):
        editor._remove_nest(raw, grid_id, include_children)
    assert raw == before


@pytest.mark.parametrize("reference", ["domain-follow", "relocation-follow", "relocation-containment", "output-domain"])
def test_surviving_references_refuse_with_an_action(reference):
    raw = topology()
    if reference == "domain-follow":
        raw["domain"][1]["follow"] = {"refine_grid_id": 3}
    elif reference == "output-domain":
        raw["case_data"] = {"output_domain": 3}
    else:
        table, key = ("follow", "refine_grid_id") if reference.endswith("follow") else ("containment", "grid_id")
        raw["relocation"] = {"grid_id": 2, table: {key: 3}}
    before = deepcopy(raw)
    with pytest.raises(ValueError, match="Settings first"):
        editor._remove_nest(raw, 3, False)
    assert raw == before


def test_removing_its_owner_removes_relocation_without_touching_survivors():
    raw = topology()
    raw["relocation"] = {"grid_id": 2, "follow": {"refine_grid_id": 3}, "track": {"output": "track.csv"}}
    editor._remove_nest(raw, 2, True)
    assert raw == {"domain": [{"grid_id": 1, "parent_id": 0}]}


def test_middle_removal_preserves_stable_survivor_ids():
    raw = topology((0, 1, 1, 3))
    before = deepcopy(raw)
    editor._remove_nest(raw, 2, False)
    assert raw["domain"] == [row for row in before["domain"] if row["grid_id"] != 2]


def test_middle_removal_round_trips_native_geometry_geography_and_later_edit(tmp_path):
    from woof.native_wrf_contract import validate_native_lambert_contracts
    from woof.static.build import GeogSelection
    from woof.source_hierarchy import _validated_static_one_way_topology
    from woof.wps_domain_ids import domain_ids_from_wps_text
    from types import SimpleNamespace

    source, original = configured_case(tmp_path, (0, 1, 1, 3))
    original_outlines = editor.native_domain_outlines(editor._build(original, source))
    result = editor.edit_configuration(request_for(source, 2))
    candidate_path = Path(result["config_path"])
    candidate = tomllib.loads(candidate_path.read_text())
    exp = editor._build(candidate, candidate_path)
    assert candidate["domain"] == [row for row in original["domain"] if row["grid_id"] != 2]
    assert result["configuration"]["domains"] == [row for row in original_outlines if row["grid_id"] != 2]
    wps_path = Path(result["wps_path"])
    text = wps_path.read_text()
    assert domain_ids_from_wps_text(text, 3) == (1, 3, 4)
    assert parse_namelist_text(text)["geogrid"]["parent_id"] == [1, 1, 2]
    grids = validate_native_lambert_contracts(exp, wps_path, source_name="ERA5")
    topology_receipt = _validated_static_one_way_topology(exp, grids)
    assert [row["grid_id"] for row in topology_receipt["domains"]] == [1, 3, 4]
    data = SimpleNamespace(geog_root=tmp_path / "GEOG", wps_namelist=wps_path)
    assert [GeogSelection.from_case_data(data, domain_id=grid_id).resolution_tokens
            for grid_id in (1, 3, 4)] == [("5m",), ("modis_lai", "default"), ("5m", "modis_lai")]
    with pytest.raises(ValueError, match="absent"):
        GeogSelection.from_case_data(data, domain_id=2)
    # A later edit must interpret the candidate's compact slots through its
    # declared stable IDs, including a custom selection on the final survivor.
    next_request = request_for(candidate_path, 4)
    next_request["output_path"] = str(tmp_path / "next.toml")
    next_result = editor.edit_configuration(next_request)
    next_text = Path(next_result["wps_path"]).read_text()
    assert domain_ids_from_wps_text(next_text, 2) == (1, 3)
    assert parse_namelist_text(next_text)["geogrid"]["geog_data_res"] == ["5m", "modis_lai+default"]


def test_edit_refuses_ambiguous_original_wps_identity_without_publishing(tmp_path):
    source, _ = configured_case(tmp_path, (0, 1, 1, 3))
    path = tmp_path / "namelist.wps"
    path.write_text("! GPUWM_DOMAIN_IDS_V1 = 1,3,2,4\n" + path.read_text())
    with pytest.raises(ValueError, match="original WPS domain identity differs"):
        editor.edit_configuration(request_for(source, 2))
    assert not (tmp_path / "candidate.toml").exists()


@pytest.mark.parametrize("identity", ["", "! GPUWM_DOMAIN_IDS_V1 = 1,4,3\n"])
def test_native_admission_rejects_missing_or_reordered_stable_identity(tmp_path, identity):
    from woof.native_wrf_contract import validate_native_lambert_contracts
    source, _ = configured_case(tmp_path, (0, 1, 1, 3))
    result = editor.edit_configuration(request_for(source, 2))
    path = Path(result["wps_path"])
    text = path.read_text().split("\n", 1)[1]
    path.write_text(identity + text)
    config = Path(result["config_path"])
    exp = editor._build(tomllib.loads(config.read_text()), config)
    with pytest.raises(ValueError, match="domain identity mismatch"):
        validate_native_lambert_contracts(exp, path, source_name="ERA5")


def test_parent_identity_cannot_hide_behind_identical_sibling_geometry(tmp_path):
    from woof.native_wrf_contract import validate_native_lambert_contracts
    source, raw = configured_case(tmp_path, (0, 1, 1, 3))
    exp = editor._build(raw, source)
    path = tmp_path / "namelist.wps"
    original = path.read_text()
    assert "parent_id         = 1, 1, 1, 3," in original
    # d02 and d03 share this fixture's geometry, so moving d04's declared
    # parent between them leaves every numeric grid coordinate identical.
    path.write_text(original.replace("parent_id         = 1, 1, 1, 3,", "parent_id         = 1, 1, 1, 2,"))
    with pytest.raises(ValueError, match="parent-slot identity mismatch"):
        validate_native_lambert_contracts(exp, path, source_name="ERA5")


def test_wps_geography_selection_follows_surviving_ids_not_array_truncation(tmp_path):
    source, raw = configured_case(tmp_path, (0, 1, 1, 3))
    survivors = deepcopy(raw)
    survivors["domain"] = [row for row in survivors["domain"] if row["grid_id"] != 2]
    exp = editor._build(survivors, source)
    wps = editor._wps_text(exp, tmp_path / "namelist.wps", tmp_path / "edited.wps", survivors, 4,
                           original_domain_ids=[1, 2, 3, 4])
    assert [row.grid_id for row in exp.domains] == [1, 3, 4]
    assert parse_namelist_text(wps)["geogrid"]["geog_data_res"] == ["5m", "modis_lai+default", "5m+modis_lai"]


def test_unassigned_wps_geography_uses_fortran_default(tmp_path):
    source, raw = configured_case(tmp_path)
    (tmp_path / "namelist.wps").write_text(render_wps_namelist(editor._build(raw, source)).replace("'default'", "'modis_lai'"))
    wps = editor._wps_text(editor._build(raw, source), tmp_path / "namelist.wps", tmp_path / "edited.wps", raw, 3)
    assert parse_namelist_text(wps)["geogrid"]["geog_data_res"] == ["modis_lai", "default", "default"]


def test_physics_edit_changes_only_selected_domain_and_keeps_native_geometry(tmp_path):
    source, original = configured_case(tmp_path)
    before = source.read_bytes()
    request = request_for(source, 2)
    request["action"] = {"kind": "set_physics", "grid_id": 2, "settings": {"radt_minutes": 15.0}}
    result = editor.edit_configuration(request)
    candidate = tomllib.loads(Path(result["config_path"]).read_text())
    assert candidate["domain"][1]["radt_minutes"] == 15.0
    assert candidate["domain"][0] == original["domain"][0]
    assert candidate["domain"][2] == original["domain"][2]
    assert result["configuration"]["domains"] == editor.native_domain_outlines(editor._build(original, source))
    assert source.read_bytes() == before
    assert result["forecast_started"] is False


def test_all_domain_physics_edit_and_invalid_setting_do_not_publish_partial_change(tmp_path):
    source, _ = configured_case(tmp_path)
    request = request_for(source, 1)
    request["action"] = {"kind": "set_physics", "grid_id": 0, "settings": {"radt_minutes": 15.0}}
    result = editor.edit_configuration(request)
    assert all(row["run"]["radt_minutes"] == 15.0 for row in result["domains"])
    request["output_path"] = str(tmp_path / "invalid.toml")
    request["action"]["settings"] = {"radt_minutes": -1.0}
    with pytest.raises(ValueError):
        editor.edit_configuration(request)
    assert not (tmp_path / "invalid.toml").exists()


def test_physics_menu_uses_registered_implemented_choices():
    from woof.physics_registry import physics_registry
    registry = physics_registry()
    components = editor.physics_components()
    assert components
    for component in components:
        for option in component["options"]:
            native = registry["components"][component["id"]]["options"][option["registry_option_id"]]
            assert native["implemented"] is True
            assert all(option["selectors"][key] == value for key,value in native["selectors"].items())
            if "ra_rrtmg_variant" in option["selectors"]:
                assert option["settings"]["ra_rrtmg_variant"] in registry["parameters"]["ra_rrtmg_variant"]["enum"]
                assert option["settings"]["wrf_rrtmg_compatibility"] == "none"
            else:
                assert option["label"] == native["label"]


def test_every_option_carries_the_registrys_own_couplings():
    """The pairing laws reach the panel as the table states them.

    They were dropped from this payload, so the only place a reader met
    "MYJ needs the Eta surface layer" was the refusal after choosing.
    Forwarded verbatim: no second vocabulary, and a pairing added to the
    registry needs no code change here to reach a front end.
    """
    from woof.physics_registry import _conditional_refusals, physics_registry

    registry = physics_registry()
    couplings = 0
    for component in editor.physics_components():
        for option in component["options"]:
            native = registry["components"][component["id"]]["options"][
                option["registry_option_id"]]
            constraints = native.get("constraints", {})
            assert option["requires_components"] == constraints.get(
                "requires_components", {})
            # The validator's own filter, so a malformed rule is dropped in
            # one place rather than reaching a caller that renders it.
            assert option["refused_when"] == _conditional_refusals(constraints)
            couplings += bool(option["requires_components"]) + bool(
                option["refused_when"])
    assert couplings, "no coupling reached the payload; the forward is dead"

    # The pairing this began with, both directions, as data.
    options = {option["id"]: option for component in editor.physics_components()
               for option in component["options"]}
    assert options["myj"]["requires_components"] == {
        "surface_layer": ["eta-similarity"]}
    # EEPS (bl 16) and BouLac (bl 8) joined MYJ when their ports landed:
    # both consume the Eta layer without the fm/fh denominators.
    assert options["eta-similarity"]["requires_components"] == {
        "pbl": ["myj", "eeps", "boulac"]}
    # The conditional refusal this began with, Milbrandt-Yau against
    # RTE+RRTMGP, retired with the defect it described (the adapter carries
    # the scheme's own cloud-optics row).  The carriers today, named by
    # (component, option) because "off" is an id in every component:
    # microphysics off on a native HRRR source, and every turbulence
    # closure, which carries the run door's diffusion-selector refusal
    # (diff_opt = 1 on the three closures that supply no coordinate
    # coefficients; mix_full_fields = false under diff_opt = 2 carries
    # none since the run door admits it)
    # and, on the 1.5-order TKE closure, the PBL pairing under diff_opt
    # = 2 that development branch wrote when diff_opt = 1
    # admitted it.  The forward is held equal to the table above, so a
    # rule a later pass writes reaches the payload unchanged.
    carriers = {
        (component["id"], option["id"]): option["refused_when"]
        for component in editor.physics_components()
        for option in component["options"] if option["refused_when"]}
    turbulence = {"closure-supplied", "constant-k", "smagorinsky-2d",
                  "smagorinsky-3d", "tke-1.5-order"}
    # FAST_SBM carries the registry's RTE+RRTMGP refusal (its remedy is the
    # legacy RRTMG pair) until its bin cloud optics are qualified; Lin
    # (mp2) carries it because it produces no effective radii.
    # The 3.0 ports brought their own conditional refusals: SBU-YLin and
    # NTU (no radii for RTE+RRTMGP, like Lin), SSiB (its spectral surface
    # carriers on RTE+RRTMGP), BMJ (radiation adapters that do not merge its
    # convective cloud) and SMS-3DTKE (scale-aware KF, coordinate diffusion).
    assert set(carriers) == {("microphysics", "off"),
                             ("microphysics", "fast-sbm-mp30"),
                             ("microphysics", "lin-mp2"),
                             ("microphysics", "sbu-ylin-mp13"),
                             ("microphysics", "ntu-mp56"),
                             # old Eta Ferrier (3.0 port) carries its own
                             # conditional refusal as well.
                             ("microphysics", "etanew-mp95"),
                             ("land_surface", "ssib"),
                             ("cumulus", "betts-miller-janjic"),
                             ("turbulence", "sms-3dtke")} | {
        ("turbulence", option) for option in turbulence}
    for option in turbulence:
        rules = carriers[("turbulence", option)]
        # Terrain admission adds more rules after the diffusion rules.
        # The metric mixing refusal retired with the run door's; no rule
        # may carry it back (identified by its condition, not list order).
        assert not [rule for rule in rules
                    if rule.get("settings") == {"diff_opt": [2],
                                                "mix_full_fields": [False]}]
        assert any(rule["settings"] == {"diff_opt": [1]}
                   for rule in rules) is (
            option in {"closure-supplied", "constant-k", "smagorinsky-3d"})
        for name, values in (("topo_wind", [1, 2]), ("gwd_opt", [1, 3])):
            terrain = [rule for rule in rules
                       if rule.get("settings") == {name: values}]
            assert len(terrain) == 1
            assert terrain[0]["remedy_settings"] == {name: 0}
    tke_rule = carriers[("turbulence", "tke-1.5-order")][0]
    assert tke_rule["settings"] == {"diff_opt": [2]}
    assert tke_rule["remedy_settings"] == {"bl_pbl_physics": 0}
    # The ra_rrtmg_variant fan-out gives one registry option several ids;
    # every one of them carries that registry option's couplings.
    variants = [option for option in options.values()
                if option["registry_option_id"] == "rte-rrtmgp"]
    assert len(variants) > 1
    assert all(option["requires_components"] == variants[0]["requires_components"]
               and option["refused_when"] == variants[0]["refused_when"]
               for option in variants)


def test_shared_physics_controls_do_not_claim_per_domain_support(tmp_path):
    source, original = configured_case(tmp_path)
    request = request_for(source, 1)
    request["action"] = {"kind":"set_physics", "grid_id":1, "settings":{"icloud":0}}
    with pytest.raises(ValueError, match="Select All domains"):
        editor.edit_configuration(request)
    request["action"]["grid_id"] = 0
    result = editor.edit_configuration(request)
    candidate = tomllib.loads(Path(result["config_path"]).read_text())
    assert candidate["shared"]["icloud"] == 0
    assert candidate["domain"] == original["domain"]


# Antimeridian export contract. A domain that crosses the seam must not export
# an edge that a Cartesian consumer reads as the 340-degree complement of a
# 20-degree domain (RFC 7946 3.1.9), and a domain that does not cross it must
# export exactly what it exported before seam cutting existed. Inputs here are
# synthetic rectangles and one native Mercator grid. Ring winding and repeated
# vertices are deliberately NOT asserted: both are legal GeoJSON, no producer
# in this tree emits them, and no consumer in this tree rejects them.
def _dateline_outline(ring, *, grid_id=1):
    return {"grid_id": grid_id, "parent_id": 0 if grid_id == 1 else 1,
            "nx": 180, "ny": 160, "nz": 8, "dx_m": 12000., "dy_m": 12000.,
            "width_km": 2160., "height_km": 1920., "history_interval_s": 60.,
            "center_latlon": (0., 180.),
            "perimeter_latlon": [(lat, lon) for lon, lat in ring]}


def _dateline_rectangle(west, south, east, north):
    return [[west, south], [east, south], [east, north], [west, north]]


def _dateline_polygons(geometry):
    assert geometry["type"] in ("Polygon", "MultiPolygon")
    return ([geometry["coordinates"]] if geometry["type"] == "Polygon"
            else geometry["coordinates"])


def _dateline_area(ring):
    # Translate first to avoid cancellation at a large longitude offset.
    x0, y0 = ring[0]
    return sum((a[0] - x0) * (b[1] - y0) - (b[0] - x0) * (a[1] - y0)
               for a, b in zip(ring, ring[1:])) / 2


def _dateline_lifted_area(perimeter):
    """The requested ring's area, computed from the input, not from the export.

    Each step follows the short arc, the same rule the perimeter sampler used
    to walk the native edge, so this is the area the export has to preserve.
    """
    lifted, previous = [], None
    for latitude, longitude in perimeter:
        if previous is not None:
            longitude = previous + (longitude - previous + 180.) % 360. - 180.
        lifted.append([longitude, latitude])
        previous = longitude
    return abs(_dateline_area(lifted + [lifted[0]]))


def _dateline_check_geometry(geometry, expected_area):
    polygons = _dateline_polygons(geometry)
    assert polygons
    area = 0.
    for polygon in polygons:
        # Native model perimeters are one shell, not a requested-area mask.
        # No hole can be introduced into these rectangular synthetic cases.
        assert len(polygon) == 1
        ring = polygon[0]
        assert len(ring) >= 4 and ring[0] == ring[-1]
        # A cut part must be a polygon, not the sliver a clipper leaves when it
        # grazes a strip edge; such a part draws as a hairline map artefact.
        assert len({tuple(p) for p in ring[:-1]}) >= 3
        assert all(-180. <= lon <= 180. and -90. <= lat <= 90. for lon, lat in ring)
        for a, b in zip(ring, ring[1:]):
            assert a != b, "the exporter emitted a zero-length edge"
            assert abs(b[0] - a[0]) <= 180., "world-spanning signed-longitude edge"
        area += abs(_dateline_area(ring))
    assert area == pytest.approx(expected_area, rel=1e-10, abs=1e-10)


_DATELINE_CASES = [
    pytest.param(170., 10., -170., 20., id="170E-to-170W"),
    pytest.param(170., -10., -170., 10., id="dateline-and-equator"),
    pytest.param(-20., -10., 20., 10., id="equator-only"),
    pytest.param(120., -40., 140., -20., id="southern-hemisphere"),
    pytest.param(-110., 30., -90., 40., id="ordinary"),
    pytest.param(170., -10., 180., 10., id="exact-east-seam"),
    pytest.param(-180., -10., -170., 10., id="exact-west-seam"),
]


@pytest.mark.parametrize("west,south,east,north", _DATELINE_CASES)
def test_dateline_export_canonical_topology(west, south, east, north):
    ring = _dateline_rectangle(west, south, east, north)
    domain = _dateline_outline(ring)
    before = deepcopy(domain)
    result = editor.domain_geojson([domain])
    assert domain == before
    assert len(result["features"]) == 1
    feature = result["features"][0]
    assert feature["id"] == "domain-1-initial"
    assert feature["properties"]["nx"] == domain["nx"]
    assert feature["properties"]["parent_id"] == domain["parent_id"]
    _dateline_check_geometry(feature["geometry"], ((east - west) % 360) * (north - south))


@pytest.mark.parametrize("west,south,east,north", _DATELINE_CASES)
def test_dateline_export_leaves_an_uncrossed_domain_byte_identical(west, south, east, north):
    """Seam cutting may not change the export of a domain that never reaches it.

    Every ordinary domain in the tree goes through this one path, so the change
    has to be inert away from the seam or it silently re-authors the published
    geometry of every existing case.
    """
    ring = _dateline_rectangle(west, south, east, north)
    geometry = editor.domain_geojson([_dateline_outline(ring)])["features"][0]["geometry"]
    if max(point[0] for point in ring) <= 180. and ((east - west) % 360.) < 180. and west <= east:
        assert geometry == {"type": "Polygon", "coordinates": [ring + [ring[0]]]}
    else:
        assert geometry["type"] == "MultiPolygon"


def test_dateline_export_cuts_the_seam_into_the_canonical_two_parts():
    """The exported document itself, so a consumer-contract change is visible."""
    ring = _dateline_rectangle(170., -10., -170., 10.)
    result = editor.domain_geojson([_dateline_outline(ring)])
    assert result["features"][0]["geometry"] == {"type": "MultiPolygon", "coordinates": [
        [[[170., -10.], [180., -10.], [180., 10.], [170., 10.], [170., -10.]]],
        [[[-180., -10.], [-170., -10.], [-170., 10.], [-180., 10.], [-180., -10.]]]]}
    assert result["features"][0]["properties"]["label"] == "d01"
    assert result["features"][0]["id"] == "domain-1-initial"


@pytest.mark.parametrize("reverse", [False, True])
def test_dateline_export_mixed_exact_seam_vertices(reverse):
    # Explicit +180 and -180 denote the same point, not an edge around Earth.
    ring = [[170., -10.], [180., -10.], [-180., -10.], [-170., -10.],
            [-170., 10.], [-180., 10.], [180., 10.], [170., 10.]]
    if reverse:
        ring.reverse()
    result = editor.domain_geojson([_dateline_outline(ring)])
    _dateline_check_geometry(result["features"][0]["geometry"], 400.)


def test_dateline_export_refuses_an_antipodal_perimeter_step():
    """Two equally short arcs leave the exported domain no defined interior."""
    ring = [[0., -10.], [180., -10.], [180., 10.], [0., 10.]]
    with pytest.raises(ValueError, match="exactly 180 degrees"):
        editor.domain_geojson([_dateline_outline(ring)])


def test_dateline_export_refuses_a_perimeter_that_re_enters_the_seam():
    """A shell crossing the seam twice would clip to one false, filled part."""
    ring = [[170., -20.], [-170., -20.], [-170., -10.], [170., -10.],
            [170., 10.], [-170., 10.], [-170., 20.], [170., 20.]]
    with pytest.raises(ValueError, match="re-enters the antimeridian"):
        editor.domain_geojson([_dateline_outline(ring)])


def test_dateline_export_refuses_a_perimeter_with_no_area():
    """An empty or collapsed perimeter used to export as an empty filled ring."""
    with pytest.raises(ValueError, match="three distinct corners"):
        editor.domain_geojson([_dateline_outline([])])
    with pytest.raises(ValueError, match="three distinct corners"):
        editor.domain_geojson([_dateline_outline([[170., -10.], [170., -10.]])])


def test_dateline_export_refuses_a_perimeter_that_encircles_a_pole():
    """A ring around a pole has no seam crossing and no canonical strip form.

    Cutting it at the strip edges leaves parts that still carry a world-spanning
    edge, which is the class this exporter exists to remove, so it is refused by
    name instead of emitted.
    """
    ring = [[-180. + 30. * step, 80.] for step in range(12)]
    with pytest.raises(ValueError, match="encircles a pole"):
        editor.domain_geojson([_dateline_outline(ring)])
    # Southern, and starting away from the seam, refuse the same way.
    ring = [[150. - 45. * step, -75.] for step in range(8)]
    with pytest.raises(ValueError, match="encircles a pole") as refusal:
        editor.domain_geojson([_dateline_outline(ring)])
    # The refusal points at the tree's own reason such a domain cannot be run --
    # the pipeline is not pole-capable -- rather than inventing a map-shaped
    # rationale, and it carries that wall's own way through.
    message = str(refusal.value)
    assert "not pole-capable" in message
    assert "not a domain this pipeline can run" in message
    # It claims nothing about a warning the caller never received: nothing hands a
    # footprint to the wizard, and a hand-authored polar root reaches this export
    # unwarned.
    assert "warn" not in message and "domain wizard" not in message
    assert "away from the pole" in message and "shrink the domain" in message


def test_dateline_export_keeps_a_pole_free_polar_ring_whole():
    """The refusal is about enclosing the pole, not about being near it."""
    ring = _dateline_rectangle(-40., 65., -20., 75.)
    geometry = editor.domain_geojson([_dateline_outline(ring)])["features"][0]["geometry"]
    assert geometry == {"type": "Polygon", "coordinates": [ring + [ring[0]]]}


def test_dateline_export_target_track_has_no_world_spanning_edges():
    outline = _dateline_outline(_dateline_rectangle(170., -10., -170., 10.), grid_id=2)
    targets = [{"time": datetime(2020, 1, 1, hour), "outline": outline,
                "requested_latlon": (lat, lon), "i_parent_start": 20,
                "j_parent_start": 20}
               for hour, (lat, lon) in enumerate([(0., 175.), (2., -175.)])]
    before = deepcopy(targets)
    exported = editor.domain_geojson([], targets)
    assert targets == before
    track = next(f for f in exported["features"] if f["properties"]["kind"] == "target_track")
    assert track["id"] == "scheduled-target-track"
    geometry = track["geometry"]
    assert geometry["type"] in ("LineString", "MultiLineString")
    lines = ([geometry["coordinates"]] if geometry["type"] == "LineString"
             else geometry["coordinates"])
    assert lines
    for line in lines:
        assert len(line) >= 2
        assert all(-180. <= lon <= 180. for lon, _ in line)
        assert all(abs(b[0] - a[0]) <= 180. for a, b in zip(line, line[1:]))
    assert lines[0][0] == [175., 0.]
    assert lines[-1][-1] == [-175., 2.]
    points = [f for f in exported["features"] if f["properties"]["kind"] == "target_point"]
    assert [f["geometry"]["coordinates"] for f in points] == [[175., 0.], [-175., 2.]]
    assert [f["properties"]["target_time_utc"] for f in points] == [
        "2020-01-01T00:00:00Z", "2020-01-01T01:00:00Z"]


def test_dateline_export_track_away_from_the_seam_is_one_unchanged_linestring():
    outline = _dateline_outline(_dateline_rectangle(-110., 30., -90., 40.), grid_id=2)
    targets = [{"time": datetime(2020, 1, 1, hour), "outline": outline,
                "requested_latlon": (lat, lon), "i_parent_start": 20, "j_parent_start": 20}
               for hour, (lat, lon) in enumerate([(30., -105.), (32., -95.)])]
    exported = editor.domain_geojson([], targets)
    track = next(f for f in exported["features"] if f["properties"]["kind"] == "target_track")
    assert track["geometry"] == {"type": "LineString",
                                 "coordinates": [[-105., 30.], [-95., 32.]]}


def test_dateline_export_track_that_does_not_move_keeps_two_positions():
    """Two scheduled times at one place must not export a one-position LineString.

    RFC 7946 3.1.4 needs two positions; a stationary target is an ordinary
    schedule, so the repeated position is kept rather than collapsed.
    """
    outline = _dateline_outline(_dateline_rectangle(170., -10., -170., 10.), grid_id=2)
    # The third column is the exported document itself, so a consumer-contract
    # change is visible: away from the seam it is what this case always exported.
    for lat, lon, expected in [(10., 175., [[175., 10.], [175., 10.]]),
                               (10., 180., [[-180., 10.], [-180., 10.]]),
                               (-5., -30., [[-30., -5.], [-30., -5.]])]:
        targets = [{"time": datetime(2020, 1, 1, hour), "outline": outline,
                    "requested_latlon": (lat, lon), "i_parent_start": 20,
                    "j_parent_start": 20} for hour in range(2)]
        exported = editor.domain_geojson([], targets)
        track = next(f for f in exported["features"] if f["properties"]["kind"] == "target_track")
        assert track["geometry"] == {"type": "LineString", "coordinates": expected}


def _dateline_native_case(tmp_path, center_lat=0.):
    source, raw = configured_case(tmp_path)
    raw["projection"] = {"map_proj": "mercator", "ref_lat": center_lat,
                         "ref_lon": 180., "truelat1": 0., "truelat2": 0., "stand_lon": 180.}
    return source, raw


@pytest.mark.parametrize("center_lat", [-35., 0., 35.])
@needs_static_bridge
def test_dateline_native_resize_keeps_directed_arc_and_tree(tmp_path, center_lat):
    from woof.experiment import validate_spawn_placement
    source, raw = _dateline_native_case(tmp_path, center_lat)
    results = []
    for west, east in [(170., -170.), (170., 190.), (-190., -170.)]:
        candidate = deepcopy(raw)
        action = {"kind": "resize_domain", "grid_id": 1, "bounds": {
            "south": center_lat - 5., "north": center_lat + 5., "west": west, "east": east}}
        editor._resize_domain(candidate, action, source)
        exp = editor._build(candidate, source)
        for child in exp.domains[1:]:
            validate_spawn_placement(exp, child.grid_id, child.i_parent_start, child.j_parent_start)
        # Exporting has no authority to modify projection, geometry or nesting.
        before = deepcopy(candidate)
        outlines = editor.native_domain_outlines(exp)
        original_outlines = deepcopy(outlines)
        exported = editor.domain_geojson(outlines)
        assert outlines == original_outlines and candidate == before
        assert editor.native_domain_outlines(exp) == original_outlines
        assert candidate["shared"] == raw["shared"]
        assert candidate["domain"][1:] == raw["domain"][1:]
        # The artifact itself: this native root perimeter is where the
        # 359.89-degree exported longitude jump was measured. Every domain of
        # the tree is checked, not only the crossing root.
        assert len(exported["features"]) == len(outlines)
        for feature, outline in zip(exported["features"], outlines):
            _dateline_check_geometry(feature["geometry"],
                                     _dateline_lifted_area(outline["perimeter_latlon"]))
        results.append(candidate)
    assert results[0]["domain"] == results[1]["domain"] == results[2]["domain"]
    for result in results[1:]:
        assert result["projection"] == pytest.approx(results[0]["projection"])


@needs_static_bridge
def test_dateline_native_polar_root_on_the_pole_refuses_by_name(tmp_path):
    """The front door itself: editor._build -> outlines -> export, on the pole.

    A polar root centred on the pole encircles it, so its perimeter spans a full
    turn. Before the refusal this exported parts carrying a 228-degree edge, the
    very class the seam cut removes.
    """
    source, raw = configured_case(tmp_path, parents=(0,))
    for ref_lat, dx in [(90., 12000.), (90., 30000.), (85., 12000.),
                        (-90., 12000.), (-90., 30000.), (-85., 12000.)]:
        truelat = math.copysign(60., ref_lat)  # A polar grid stands on its own hemisphere.
        candidate = deepcopy(raw)
        candidate["projection"] = {"map_proj": "polar", "ref_lat": ref_lat, "ref_lon": 0.,
                                   "truelat1": truelat, "truelat2": truelat, "stand_lon": 0.}
        candidate["domain"][0]["dx"] = dx
        exp = editor._build(candidate, source)
        outlines = editor.native_domain_outlines(exp)
        perimeter = outlines[0]["perimeter_latlon"]
        assert max(lon for _, lon in perimeter) - min(lon for _, lon in perimeter) > 300.
        with pytest.raises(ValueError, match="encircles a pole"):
            editor.domain_geojson(outlines)


#: Arctic roots on the antimeridian that enclose a pole, and one at the same
#: projection and resolution that does not.  Measured, not assumed: the
#: footprint geometry this exporter refuses on is the same one the doors and
#: plan review refuse on (woof.static.projection.footprint_contains_pole), and
#: tools/pole_blast_radius.py sweeps it over lambert/polar/mercator x six
#: latitudes x three resolutions -- 16 of 54 configurations enclose the pole.
#: The five rows below are this exporter's own outcomes and agree with that
#: measurement row for row.
#: None of the four below is centred on a pole, which is the point of the test:
#: the refusal is about a footprint that ENCLOSES a pole, and ordinary wide
#: Arctic domains reach it.
_DATELINE_ARCTIC_ROOTS = [
    ("lambert", 75., 30000., 300, 240, 30., 60., "refuse"),
    ("lambert", 75., 30000., 200, 180, 30., 60., "refuse"),
    ("polar", 65., 30000., 200, 180, 60., 60., "refuse"),
    ("polar", 75., 30000., 200, 180, 60., 60., "refuse"),
    ("lambert", 65., 30000., 300, 240, 30., 60., "export"),
]


@pytest.mark.parametrize("map_proj,ref_lat,dx,nx,ny,truelat1,truelat2,outcome",
                         _DATELINE_ARCTIC_ROOTS)
@needs_static_bridge
def test_dateline_native_arctic_root_refuses_only_when_it_encloses_the_pole(
        tmp_path, map_proj, ref_lat, dx, nx, ny, truelat1, truelat2, outcome):
    """The refusal is not a ref_lat 90 special case, and not a latitude band either.

    The last row is the control: the same projection, resolution and cell count
    as the first, four hundred kilometres further south, so its footprint clears
    the pole. It exports, and its export passes the same geometry check every
    other seam-crossing case in this file passes.
    """
    source, raw = configured_case(tmp_path, parents=(0,))
    raw["projection"] = {"map_proj": map_proj, "ref_lat": ref_lat, "ref_lon": 180.,
                         "truelat1": truelat1, "truelat2": truelat2, "stand_lon": 180.}
    raw["domain"][0].update(dx=dx, nx=nx, ny=ny)
    outlines = editor.native_domain_outlines(editor._build(raw, source))
    perimeter = outlines[0]["perimeter_latlon"]
    # Every row here is on the antimeridian, so every row's raw perimeter spans
    # the world before the lift; that is what makes the exported form the whole
    # question rather than a detail.
    assert max(lon for _, lon in perimeter) - min(lon for _, lon in perimeter) > 300.
    if outcome == "refuse":
        with pytest.raises(ValueError, match="encircles a pole"):
            editor.domain_geojson(outlines)
        return
    exported = editor.domain_geojson(outlines)
    for feature in exported["features"]:
        _dateline_check_geometry(feature["geometry"],
                                 _dateline_lifted_area(perimeter))


@needs_static_bridge
def test_dateline_native_polar_root_off_the_pole_still_exports(tmp_path):
    """The same projection away from the pole keeps its ordinary single ring."""
    source, raw = configured_case(tmp_path, parents=(0,))
    raw["projection"] = {"map_proj": "polar", "ref_lat": 70., "ref_lon": -40.,
                         "truelat1": 70., "truelat2": 70., "stand_lon": -40.}
    outlines = editor.native_domain_outlines(editor._build(raw, source))
    geometry = editor.domain_geojson(outlines)["features"][0]["geometry"]
    assert geometry["type"] == "Polygon"
    _dateline_check_geometry(geometry, _dateline_lifted_area(outlines[0]["perimeter_latlon"]))


def test_dateline_native_resize_rejects_sorted_complement(tmp_path):
    source, raw = _dateline_native_case(tmp_path)
    before = deepcopy(raw)
    with pytest.raises(ValueError, match="less than 180"):
        editor._resize_domain(raw, {"kind": "resize_domain", "grid_id": 1,
            "bounds": {"south": -5., "west": -170., "north": 5., "east": 170.}}, source)
    assert raw == before


def test_dateline_wizard_polygon_and_multipolygon_keep_same_arc(tmp_path):
    from woof import domain_wizard as dw
    # Reader already supports both types. This is not a desktop rendering check.
    full = _dateline_rectangle(170., -10., -170., 10.)
    left = _dateline_rectangle(170., -10., 180., 10.)
    right = _dateline_rectangle(-180., -10., -170., 10.)
    docs = [{"type": "Polygon", "coordinates": [full + [full[0]]]},
            {"type": "MultiPolygon", "coordinates": [[left + [left[0]]], [right + [right[0]]]]}]
    spans = []
    for index, document in enumerate(docs):
        path = tmp_path / f"requested-{index}.geojson"
        path.write_text(json.dumps(document), encoding="utf-8")
        footprint = dw.load_polygon_footprint(path)
        spans.append((footprint.south, footprint.north, footprint.longitude_span))
    assert spans == [(-10., 10., 20.), (-10., 10., 20.)]


def test_dateline_exported_geometry_round_trips_through_the_wizard_reader(tmp_path):
    """The seam-cut export reads back as the one area it was cut from.

    Without this, the two parts could be read as two requested areas and fit a
    root spanning the whole world instead of the 20 degrees asked for.
    """
    from woof import domain_wizard as dw
    exported = editor.domain_geojson([_dateline_outline(
        _dateline_rectangle(170., -10., -170., 10.))])
    path = tmp_path / "requested-area.geojson"
    path.write_text(json.dumps(exported["features"][0]["geometry"]), encoding="utf-8")
    footprint = dw.load_polygon_footprint(path)
    assert (footprint.south, footprint.north) == (-10., 10.)
    assert footprint.longitude_span == pytest.approx(20.)


@needs_static_bridge
def test_dateline_forcing_crop_is_not_the_native_outline(tmp_path):
    from woof import domain_wizard as dw
    source, raw = _dateline_native_case(tmp_path)
    exp = editor._build(raw, source)
    before = editor.native_domain_outlines(exp)
    projection = deepcopy(raw["projection"])
    crop = dw._fetch_area(projection, exp.domains[0].run.nx, exp.domains[0].run.ny,
                          margin_deg=5., root_dx_m=12000.)
    outline = before[0]["perimeter_latlon"]
    assert crop[0] < min(lat for lat, _ in outline)
    assert crop[2] > max(lat for lat, _ in outline)
    assert crop[1] > crop[3]  # Crossing signed bounds, not their sorted complement.
    assert raw["projection"] == projection
    assert editor.native_domain_outlines(exp) == before
