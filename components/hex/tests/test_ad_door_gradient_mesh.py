# SPDX-License-Identifier: Apache-2.0
"""--gradient-mesh: the gradient of a regional window taken on a coarser companion mesh of the same region and
carried back, as table rows (gradient_meshes), with no card.

What these hold: native runs the mesh itself and says so in the receipt; auto reads the mesh's default in the table
(native today), so changing it is one entry; a companion name runs the same command on the companion mesh and the
carried answer names the mesh asked for, the mesh it was made on, the native mesh, the method, and that the answer
check was made on the coarse operator; refusals by name (a door with no coarse route, a mesh with no companion, a
companion name the mesh lacks, a target the route was not shown on, another start, a wrong map), each before any
download or card.
"""
import argparse
import hashlib
import json
import struct
import types

import numpy as np
import pytest

from woof.hex import ad_doors as d
from woof.hex.ad import assets, gradmesh, rows

STORM = "q3.96.38278.n40.60w91.30"
SHOWN = ["--metric", "theta_m", "--circle", "40.6,-91.3,150", "--steps", "59"]


def _parse(argv):
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


@pytest.fixture
def no_fetch(monkeypatch, tmp_path):
    """A card that fits, a host that fits, a runner, an empty row cache, and an asset root that must not be resolved."""
    for name in d.NO_LOCAL_GPU_ENV:
        monkeypatch.delenv(name, raising=False)
    fetched = []

    def asset_root(cache=None):
        fetched.append("asset_root")
        raise AssertionError("the asset root was resolved (a first use downloads ~940 MB)")

    monkeypatch.setattr(assets, "asset_root", asset_root)
    monkeypatch.setattr(assets, "_fetch", lambda *a, **k: fetched.append("_fetch"))
    for name in ("WOOF_HEX_AD_ASSETS", rows.IMAGE_BUILDER_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(rows.ROWS_ENV, str(tmp_path / "rows"))
    monkeypatch.setattr(d, "resolve_runner", lambda explicit: tmp_path / "runner")
    monkeypatch.setattr(d, "compute_capability", lambda card: "12.0")
    monkeypatch.setattr(d, "total_mib", lambda card: 97887)
    monkeypatch.setattr(d, "free_mib", lambda card: 97000)
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 500000)
    monkeypatch.setattr(d, "refuse_plan_libraries_missing", lambda *a, **k: None)
    return fetched


@pytest.fixture
def storm(tmp_path):
    """The 3 km storm box named on the command line, its grid an empty file (nothing reads it before the card)."""
    grid = tmp_path / "storm.grid.nc"
    grid.write_bytes(b"")
    return ["--mesh", STORM, "--grid", str(grid)]


def write_csr(path, n_rows, n_cols, entries):
    """An HXCSR1 map: entries is one list of (column, weight) per row."""
    ptr, col, w = [0], [], []
    for row in entries:
        for c, v in row:
            col.append(c)
            w.append(v)
        ptr.append(len(col))
    body = (b"HXCSR1\0\0" + struct.pack("<3Q", n_rows, n_cols, len(col)) + struct.pack(f"<{len(ptr)}Q", *ptr)
            + struct.pack(f"<{len(col)}I", *col) + struct.pack(f"<{len(w)}d", *w))
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def write_plan(path, start="2024-05-21_15:00:00", dt=30.0):
    path.mkdir(parents=True, exist_ok=True)
    (path / "plan.json").write_text(json.dumps({"physics": {"state_xtime": start, "dt_seconds": dt}}))
    return path


@pytest.fixture
def small(monkeypatch, tmp_path):
    """The shipped table with the storm box's entry cut to a 4-cell native mesh and a 2-cell companion, and an asset
    root that holds the companion's map, grid and plan."""
    loaded = d.load_table
    root = tmp_path / "root"
    comp_dir = root / "gradient-meshes" / STORM / "6km"
    comp_dir.mkdir(parents=True)
    # native cells 0..3; cells 0 and 3 lie in the boundary zone in the tests that read it
    sha = write_csr(comp_dir / "pc.csr", 4, 2, [[(0, 1.0)], [(0, 0.5), (1, 0.5)], [(1, 1.0)], [(1, 1.0)]])
    (comp_dir / "q6.96.10014.n40.60w91.30.grid.nc").write_bytes(b"")
    write_plan(comp_dir / "plan-hrrr-20240521T15-s30")

    def table(path=None):
        t = loaded(path)
        entry = t["gradient_meshes"]["by_mesh"][STORM]
        entry["native"]["cells"] = 4
        comp = entry["companions"]["6km"]
        comp["cells"] = 2
        comp["map"].update(sha256=sha, rows=4, cols=2)
        return t

    monkeypatch.setattr(d, "load_table", table)
    return {"root": root, "comp_dir": comp_dir, "sha": sha}


# ------------------------------------------------------------------------------------------------- the table
def test_the_table_names_the_storm_box_companion_with_its_default_native_and_one_shown_target():
    section = d.load_table()["gradient_meshes"]
    assert section["doors_carried"] == ["sensitivity"]
    assert set(section["doors_refused"]) == {"tangent", "adjoint-test", "param-grad"}   # the other three, by name
    entry = section["by_mesh"][STORM]
    assert entry["default"] == "native"   # until the speed lead's tests read: then one entry changes
    comp = entry["companions"]["6km"]
    assert comp["map"]["sha256"] == "5f3bb748fb68e757becffae96255f9adaeb9a479b53d1e4311add662c63f4143"
    assert (comp["map"]["rows"], comp["map"]["cols"], comp["cells"], entry["native"]["cells"]) == (38278, 10014,
                                                                                                    10014, 38278)
    [shown] = comp["shown"]
    assert shown["metrics"] == ["theta_m"] and shown["region"]["circle"] == [40.6, -91.3, 150.0]
    assert shown["window_s"] == [885.0, 885.0] and shown["starts"] == ["2024-05-21T15:00:00"]
    assert "not shown" in comp["scope"].lower() or "Not shown" in comp["scope"]


@pytest.mark.parametrize("door", d.DOORS)
def test_every_door_takes_the_word_and_its_default_is_auto(door):
    args = _parse([door, "--out", "o"])
    assert args.gradient_mesh == "auto" and args.gradient_plan is None


# -------------------------------------------------------------------------------------------- native and auto
@pytest.mark.parametrize("door", d.DOORS)
def test_native_is_the_mesh_itself_on_every_door(door):
    route = d.gradient_route(_parse([door, "--gradient-mesh", "native", "--out", "o"]), door)
    assert route["carried"] is False and "itself" in route["why"]


def test_auto_on_a_mesh_with_no_companion_is_the_mesh_itself():
    route = d.gradient_route(_parse(["sensitivity", "--out", "o"]), "sensitivity")   # the 120 km default row
    assert route["carried"] is False and route["why"] == "auto: the table names no companion mesh for this mesh"


def test_auto_on_the_storm_box_reads_the_table_default_native(storm):
    route = d.gradient_route(_parse(["sensitivity", *storm, "--out", "o"]), "sensitivity")
    assert route["carried"] is False and route["why"].startswith(f"auto: the table's default for mesh {STORM} is "
                                                                 "native (native until")


@pytest.mark.parametrize("door", ["tangent", "adjoint-test", "param-grad"])
def test_auto_on_a_door_with_no_coarse_route_is_the_mesh_itself_even_when_the_default_is_coarse(door, storm,
                                                                                                monkeypatch):
    loaded = d.load_table

    def coarse_default(path=None):
        t = loaded(path)
        t["gradient_meshes"]["by_mesh"][STORM]["default"] = "6km"
        return t

    monkeypatch.setattr(d, "load_table", coarse_default)
    route = d.gradient_route(_parse([door, *storm, "--out", "o"]), door)
    assert route["carried"] is False and "has no coarse route" in route["why"]


def test_changing_the_default_is_one_table_entry(storm, small, tmp_path, monkeypatch):
    loaded = d.load_table

    def coarse_default(path=None):
        t = loaded(path)
        t["gradient_meshes"]["by_mesh"][STORM]["default"] = "6km"
        return t

    monkeypatch.setattr(d, "load_table", coarse_default)
    route = d.gradient_route(_parse(["sensitivity", *storm, *SHOWN, "--assets", str(small["root"]),
                                     "--out", str(tmp_path / "o")]), "sensitivity")
    assert route["carried"] is True and route["how"] == f"auto: the table's default for mesh {STORM} is 6km"


# ------------------------------------------------------------------------------------------- refusals by name
@pytest.mark.parametrize("door,said", [("tangent", "a tangent is a forecast change"),
                                       ("adjoint-test", "it checks the operator it runs"),
                                       ("param-grad", "dJ/dln p on a companion mesh")])
def test_the_three_doors_with_no_coarse_route_refuse_a_companion_by_name(door, said, storm, no_fetch, tmp_path):
    args = _parse([door, *storm, "--gradient-mesh", "6km", "--out", str(tmp_path / "o")])
    with pytest.raises(d.AdDoorError, match=rf"--gradient-mesh 6km: the {door} door is not carried between meshes: "
                                            rf"{said}.*--gradient-mesh native runs it on the mesh itself\. Nothing "
                                            r"was downloaded"):
        args.handler.__wrapped__(args)
    assert no_fetch == [] and not (tmp_path / "o").exists()


def test_a_mesh_with_no_companion_row_is_refused_by_name(no_fetch, tmp_path):
    with pytest.raises(d.AdDoorError, match=r"--gradient-mesh 6km: no companion row for mesh x1\.40962 \(the table "
                                            rf"names companion meshes for: {STORM}\); --gradient-mesh native"):
        d.gradient_route(_parse(["sensitivity", "--gradient-mesh", "6km", "--out", str(tmp_path / "o")]),
                         "sensitivity")
    assert no_fetch == []


def test_a_companion_name_the_mesh_lacks_is_refused_with_its_companions(storm, no_fetch, tmp_path):
    with pytest.raises(d.AdDoorError, match=rf"mesh {STORM} has no companion of that name \(its companions: 6km; or "
                                            r"native\)"):
        d.gradient_route(_parse(["sensitivity", *storm, "--gradient-mesh", "12km", "--out", "o"]), "sensitivity")
    assert no_fetch == []


@pytest.mark.parametrize("change,said", [
    (["--metric", "qr"], r"metric qr \(shown: theta_m\)"),
    (["--circle", "40.6,-91.3,30"], r"radius 30 km \(shown: 150 to 150 km\)"),
    (["--circle", "41.6,-91.3,150"], r"a circle centred 111 km from the shown centre 40\.6,-91\.3"),
    (["--box", "39,42,-93,-89"], r"a --box region \(shown on a circle\)"),
    (["--levels", "1-10"], r"levels 1-10 \(shown on every level\)"),
    (["--steps", "3"], r"a 45 s window \(shown: 885 to 885 s\)"),
])
def test_a_target_the_route_was_not_shown_on_is_refused_naming_what_was_shown(change, said, storm, no_fetch,
                                                                              tmp_path):
    argv = list(SHOWN)
    for k in range(0, len(change), 2):
        if change[k] in argv:
            argv[argv.index(change[k]) + 1] = change[k + 1]
        else:
            argv += change[k:k + 2]
    if "--box" in argv:
        i = argv.index("--circle")
        del argv[i:i + 2]
    with pytest.raises(d.AdDoorError, match=rf"the coarse route was not shown on this target: .*{said}.*It was shown "
                                            r"on: mean modified potential temperature \(theta_m\) over the 150 km "
                                            r"circle at 40\.6N 91\.3W"):
        d.gradient_route(_parse(["sensitivity", *storm, *argv, "--gradient-mesh", "6km", "--out", "o"]),
                         "sensitivity")
    assert no_fetch == []


def test_the_shown_target_s_companion_files_are_in_the_pinned_downloads(storm, no_fetch):
    """The 6 km plan, mesh files and map are a pinned download (kind ad-gradient-mesh): the route passes the pin check
    and only then goes to the asset root (which this test forbids)."""
    pins = assets.load_pins()
    [pin] = [r for r in pins["assets"].values() if r["kind"] == "ad-gradient-mesh"]
    comp = d.load_table()["gradient_meshes"]["by_mesh"][STORM]["companions"]["6km"]
    for rel in (comp["grid"], comp["oro"], comp["plan"], comp["map"]["file"]):
        assert any(rel.startswith(h) for h in pin["holds"]), rel
    with pytest.raises(AssertionError, match="the asset root was resolved"):
        d.gradient_route(_parse(["sensitivity", *storm, *SHOWN, "--gradient-mesh", "6km", "--out", "o"]),
                         "sensitivity")
    assert no_fetch == ["asset_root"]


def test_init_is_refused_on_the_coarse_route(storm, no_fetch, tmp_path):
    with pytest.raises(d.AdDoorError, match="--init records a plan on the native mesh"):
        d.gradient_route(_parse(["sensitivity", *storm, *SHOWN, "--init", str(tmp_path / "i.nc"),
                                 "--gradient-mesh", "6km", "--out", "o"]), "sensitivity")


def test_another_start_is_refused(storm, small, tmp_path):
    native = write_plan(tmp_path / "plan3", start="2024-05-22_15:00:00", dt=15.0)
    with pytest.raises(d.AdDoorError, match=r"the native plan starts at 2024-05-22T15:00:00 and the companion's plan "
                                            r"at 2024-05-21T15:00:00"):
        d.gradient_route(_parse(["sensitivity", *storm, *SHOWN, "--plan", str(native), "--assets", str(small["root"]),
                                 "--gradient-mesh", "6km", "--out", "o"]), "sensitivity")
    other = write_plan(tmp_path / "plan6", start="2024-05-22_15:00:00")   # both on another day: not shown
    native = write_plan(tmp_path / "plan3b", start="2024-05-22_15:00:00", dt=15.0)
    with pytest.raises(d.AdDoorError, match=r"start 2024-05-22T15:00:00 \(shown: 2024-05-21T15:00:00\)"):
        d.gradient_route(_parse(["sensitivity", *storm, *SHOWN, "--plan", str(native), "--gradient-plan", str(other),
                                 "--assets", str(small["root"]), "--gradient-mesh", "6km", "--out", "o"]),
                         "sensitivity")


def test_a_map_that_is_not_the_tables_is_refused(storm, small, tmp_path):
    write_csr(small["comp_dir"] / "pc.csr", 4, 2, [[(0, 1.0)], [(0, 1.0)], [(1, 1.0)], [(1, 1.0)]])
    with pytest.raises(d.AdDoorError, match=r"has sha256 [0-9a-f]{64}; the table pins [0-9a-f]{64}: another map's "
                                            "carrying is not this companion's"):
        d.gradient_route(_parse(["sensitivity", *storm, *SHOWN, "--assets", str(small["root"]),
                                 "--gradient-mesh", "6km", "--out", "o"]), "sensitivity")


def test_a_map_of_other_meshes_is_refused(tmp_path):
    sha = write_csr(tmp_path / "pc.csr", 3, 2, [[(0, 1.0)], [(1, 1.0)], [(1, 1.0)]])
    with pytest.raises(gradmesh.GradientMeshError, match=r"takes 2 coarse cells to 3 native cells; the meshes have "
                                                         r"2 and 4"):
        gradmesh.load_map(tmp_path / "pc.csr", rows=4, cols=2, sha256=sha)


# ------------------------------------------------------------------------------------------- the coarse route
def test_the_coarse_route_runs_the_same_command_on_the_companion_mesh(storm, small, tmp_path):
    native = write_plan(tmp_path / "plan3", dt=15.0)
    route = d.gradient_route(_parse(["sensitivity", *storm, *SHOWN, "--plan", str(native),
                                     "--assets", str(small["root"]), "--gradient-mesh", "6km",
                                     "--out", str(tmp_path / "o")]), "sensitivity")
    c = route["coarse_args"]
    assert route["carried"] is True and route["how"] == "asked for (--gradient-mesh 6km)"
    assert (c.mesh, c.row, c.init, c.steps) == ("q6.96.10014.n40.60w91.30", None, None, 30)   # 885 s in 30 s steps
    assert c.plan == small["comp_dir"] / "plan-hrrr-20240521T15-s30"
    assert c.grid == small["comp_dir"] / "q6.96.10014.n40.60w91.30.grid.nc"
    assert c.out == tmp_path / "o" / "made-on-6km" and c.no_map is True and c.gradient_mesh == "native"
    assert (c.metric, c.circle, c.levels) == ("theta_m", "40.6,-91.3,150", None)   # the same target
    assert route["native"]["window_s"] == 885.0 and route["coarse"]["window_s"] == 900.0
    assert route["native"]["plan_start"] == route["coarse"]["plan_start"] == "2024-05-21T15:00:00"


def test_carrying_keeps_a_constant_density_and_zeroes_the_boundary_zone(tmp_path):
    sha = write_csr(tmp_path / "pc.csr", 4, 2, [[(0, 1.0)], [(0, 0.5), (1, 0.5)], [(1, 1.0)], [(1, 1.0)]])
    op = gradmesh.load_map(tmp_path / "pc.csr", rows=4, cols=2, sha256=sha)
    area_c, area_n = np.array([4.0, 2.0]), np.array([1.0, 1.5, 0.5, 2.0])
    g_c = np.array([[8.0, 4.0], [-4.0, -2.0]])   # density 2 and -1 per unit area on two levels
    carried = gradmesh.carry_gradient(op, g_c, area_c, area_n)
    assert np.allclose(carried, [[2.0, 3.0, 1.0, 4.0], [-1.0, -1.5, -0.5, -2.0]])   # same density, native areas
    inner = np.array([False, True, True, False])
    assert np.allclose(gradmesh.carry_gradient(op, g_c, area_c, area_n, inner), [[0, 3.0, 1.0, 0], [0, -1.5, -0.5, 0]])
    with pytest.raises(gradmesh.GradientMeshError, match="different units"):
        gradmesh.carry_gradient(op, g_c, area_c * 1e-12, area_n)


def test_a_carried_gradient_never_reads_as_a_native_one(storm, small, tmp_path, monkeypatch):
    """The whole carried door with the window faked: the answer, its file and its receipt say carried; the window on
    the companion mesh keeps its own folder and receipt; an edge field is not carried and is named."""
    native = write_plan(tmp_path / "plan3", dt=15.0)
    args = _parse(["sensitivity", *storm, *SHOWN, "--plan", str(native), "--assets", str(small["root"]),
                   "--gradient-mesh", "6km", "--no-map", "--out", str(tmp_path / "o")])
    route = d.gradient_route(args, "sensitivity")
    made = {}

    class Ctx:
        def __init__(self, a, door):
            made["args"] = a
            self.args, self.door, self.row_name, self.steps = a, door, "hrrrv5-q6.96.10014.n40.60w91.30", a.steps
            self.row = {"mesh": "q6.96.10014.n40.60w91.30", "dt_s": 30.0}
            self.out = a.out.resolve()
            self.out.mkdir(parents=True)
            self.geometry = {"areaCell": np.array([4.0, 2.0])}
            self.fmap = types.SimpleNamespace(doc={"plan_sha256": "ab" * 32})

    def window(ctx, a):
        made["block"] = ctx.gradient_block_override
        g = np.array([[8.0, 4.0]] * 2)
        rec = {"door": "sensitivity", "row": ctx.row_name, "mesh": ctx.row["mesh"], "valid": True, "passed": True,
               "stages": [], "strict_check": {"how": "ran", "pass": True},
               "result": {"sensitivity_nc": str(ctx.out / "sensitivity.nc")}}
        return 0, {"out_fields": {"dJ_dtheta": (g, "K K-1", "dJ/dtheta", "cell"),
                                  "dJ_dqv": (g, "K (kg/kg)-1", "dJ/dqv", "cell"),
                                  "dJ_du_stored": (np.ones((2, 5)), "K per m s-1", "dJ/du", "edge")},
                   "objective": {"name": "J", "metric": "theta_m"}, "rec": rec, "units": "K"}

    written = {}
    monkeypatch.setattr(d, "Context", Ctx)
    monkeypatch.setattr(d, "_sensitivity", window)
    monkeypatch.setattr(d, "mesh_geometry", lambda grid, cells, edges: {
        "areaCell": np.array([1.0, 1.5, 0.5, 2.0]), "latCell": np.array([40.0, 40.1, 40.2, 40.3]),
        "lonCell": np.array([-91.0, -91.1, -91.2, -91.3]), "latEdge": np.zeros(9), "lonEdge": np.zeros(9)})
    monkeypatch.setattr(gradmesh, "boundary_inner", lambda grid, cells: np.array([False, True, True, True]))
    monkeypatch.setattr(d, "write_mesh_fields", lambda path, fields, attrs, coords=None: written.update(
        path=path, fields=fields, attrs=attrs, coords=coords))
    assert d._sensitivity_carried(args, route) == 0
    assert made["args"].mesh == "q6.96.10014.n40.60w91.30" and made["block"]["note"].startswith(
        f"the window on the 6km companion mesh of mesh {STORM}")
    assert written["path"] == (tmp_path / "o" / "sensitivity.nc").resolve()
    assert sorted(written["fields"]) == ["dJ_dqv", "dJ_dtheta"]   # the edge field is not carried
    # the carried file places its cells: the NATIVE mesh's cell centres, and no edge coordinates (no edge field)
    assert sorted(written["coords"]) == ["latCell", "lonCell"]
    assert np.allclose(written["coords"]["latCell"][0], [40.0, 40.1, 40.2, 40.3])
    assert written["coords"]["lonCell"][1] == "degrees_east"
    assert np.allclose(written["fields"]["dJ_dtheta"][0], [[0, 3.0, 1.0, 4.0]] * 2)
    assert written["fields"]["dJ_dtheta"][2].startswith("CARRIED from the 6km companion mesh")
    attrs = written["attrs"]
    assert attrs["gradient_made_on_mesh"] == "q6.96.10014.n40.60w91.30" and attrs["gradient_carried_to_mesh"] == STORM
    assert attrs["gradient_carried"].startswith("yes") and "CARRIED" in attrs["title"]
    assert attrs["gradient_fields_not_carried"] == "dJ_du_stored"
    top = json.loads((tmp_path / "o" / "receipt.json").read_text())
    b = top["gradient_mesh"]
    assert b["asked"] == "6km" and b["carried"] is True
    assert b["made_on"]["mesh"] == "q6.96.10014.n40.60w91.30" and b["made_on"]["steps"] == 30
    assert b["carried_to"]["mesh"] == STORM and b["carried_to"]["window_s"] == 885.0
    assert b["method"].startswith("the gradient is a covector") and b["map"]["sha256"] == small["sha"]
    assert b["answer_check"].startswith("made on the coarse operator (mesh q6.96.10014.n40.60w91.30")
    assert b["fields_not_carried"] == {"dJ_du_stored": "an edge field: carrying it between the meshes was not shown"}
    assert top["result"]["carried"] is True and top["result"]["sensitivity_nc"].endswith("o/sensitivity.nc")
    assert top["result"]["made_on_sensitivity_nc"].endswith("made-on-6km/sensitivity.nc")
    assert "the carried field has no check of its own" in top["valid_rule"]


def test_a_native_receipt_and_file_say_the_mesh_itself():
    ctx = types.SimpleNamespace(row={"mesh": STORM}, row_name=f"hrrrv5-{STORM}",
                                args=types.SimpleNamespace(gradient_mesh="auto"),
                                gradient_route={"asked": "auto", "why": "auto: the table's default ... is native"})
    block = d.Context.gradient_block(ctx)
    assert block["carried"] is False and block["made_on"]["mesh"] == STORM and block["carried_to"] is None
    attrs = d.Context.gradient_nc_attrs(ctx)
    assert attrs["gradient_carried"] == "no: taken on the mesh itself" and attrs["gradient_mesh_asked"] == "auto"


def test_the_storm_example_on_the_6km_route_is_refused_as_another_forecast(small, tmp_path, monkeypatch):
    """sensitivity --row <storm row> --gradient-mesh 6km with no plan: the native window would be the storm example
    (HRRR 2023-03-31 18Z) and the shipped 6 km plan starts 2024-05-21 15Z, so the gradient would be of another
    forecast. Refused in one sentence naming both starts, before anything is fetched."""
    fetched = []
    monkeypatch.setattr(d, "resolve_assets", lambda *a, **k: fetched.append(a) or small["root"])
    with pytest.raises(d.AdDoorError) as err:
        d.gradient_route(_parse(["sensitivity", "--row", "hrrrv5-q3.96.38278.n40.60w91.30", *SHOWN,
                                 "--assets", str(small["root"]), "--gradient-mesh", "6km",
                                 "--out", str(tmp_path / "o")]), "sensitivity")
    words = str(err.value)
    assert "2024-05-21 15:00:00" in words and "2023-03-31 18:00:00" in words and "another forecast" in words
    assert "Nothing was downloaded" in words and fetched == []


def test_the_storm_row_s_6km_route_with_a_plan_of_its_start_reads_the_downloaded_3km_grid(small, tmp_path):
    """The 6 km route on the storm row with --plan, a 3 km recording of the 6 km plan's own start: the 3 km grid is
    the row's downloaded mesh file (no --grid), and the native plan's start is the companion's."""
    table = d.load_table()
    row = table["rows"]["hrrrv5-q3.96.38278.n40.60w91.30"]
    grid = small["root"] / row["assets"]["grid"]
    grid.parent.mkdir(parents=True, exist_ok=True)
    grid.write_bytes(b"")
    plan = tmp_path / "plan-3km-20240521T15"
    plan.mkdir()
    (plan / "plan.json").write_text(json.dumps({"physics": {"state_xtime": "2024-05-21_15:00:00",
                                                            "dt_seconds": 15}}))
    route = d.gradient_route(_parse(["sensitivity", "--row", "hrrrv5-q3.96.38278.n40.60w91.30", *SHOWN,
                                     "--assets", str(small["root"]), "--gradient-mesh", "6km", "--plan", str(plan),
                                     "--out", str(tmp_path / "o")]), "sensitivity")
    assert route["carried"] is True and route["native"]["grid"] == str(grid)
    assert route["native"]["plan_start"] == route["coarse"]["plan_start"] == "2024-05-21T15:00:00"
    assert route["native"]["start_note"] is None


def test_a_user_of_either_example_does_not_download_the_6km_gradient_mesh():
    pins = assets.load_pins()
    [name] = [n for n, r in pins["assets"].items() if r["kind"] == "ad-gradient-mesh"]
    table = d.load_table()
    for row in ("hrrrv5-x1.40962", "hrrrv5-q3.96.38278.n40.60w91.30"):
        r = table["rows"][row]
        assert name not in assets.wanted(pins, [r["assets"]["grid"], r["assets"]["oro"], r["example_plan"]])
    comp = table["gradient_meshes"]["by_mesh"][STORM]["companions"]["6km"]
    assert name in assets.wanted(pins, [comp["grid"], comp["plan"], comp["map"]["file"]])
