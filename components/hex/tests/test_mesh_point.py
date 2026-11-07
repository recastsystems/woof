"""``woof hex mesh-plan --point``: the ladder, the pricing, the refusals.

Card-free and generator-free by construction: what needs ``rw_mpas_mesh`` is
faked with a tiny executable that prints the receipt shape the real one
prints, and what needs a card is priced on the admission surface's own rows.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from woof.hex import mesh_point  # noqa: E402
from woof.hex.cli import build_parser, main  # noqa: E402
from woof.hex.mesh_point import PointPlanRefusal  # noqa: E402


@pytest.fixture(autouse=True)
def _stub_grids_take_the_default_30s_sampling(monkeypatch, request):
    """The door's tests fake the generator with files that are not NetCDF; the
    30" sampling factor is read from a real grid and is tested on one below."""
    if "thirty_second" not in request.node.name:
        monkeypatch.setattr(mesh_point, "thirty_second_supersample", lambda grid: 1)


def _unit_sphere_grid(path: Path, dc_edge_m: list[float]) -> Path:
    import numpy as np
    from netCDF4 import Dataset

    with Dataset(str(path), "w") as ds:
        ds.createDimension("nCells", 4)
        ds.createDimension("nEdges", len(dc_edge_m))
        for name, value in (("xCell", 1.0), ("yCell", 0.0), ("zCell", 0.0)):
            ds.createVariable(name, "f8", ("nCells",))[:] = np.full(4, value)
        dc = np.asarray(dc_edge_m, dtype=np.float64) / mesh_point.MPAS_EARTH_RADIUS_M
        ds.createVariable("dcEdge", "f8", ("nEdges",))[:] = dc
        ds.createVariable("dvEdge", "f8", ("nEdges",))[:] = dc / np.sqrt(3.0)
    return path


@pytest.mark.parametrize(("shortest_m", "factor"), [
    (697.594, 2),    # a 750 m core: 4,258 non-finite terrain cells at factor 1
    (450.0, 3),
    (1309.0, 2),
    (1311.0, 1),     # every mesh at 1.31 km and coarser keeps the builder's default
    (2699.884, 1),   # a 3 km core
])
def test_thirty_second_sampling_gives_every_cell_a_pixel(tmp_path, shortest_m, factor) -> None:
    grid = _unit_sphere_grid(tmp_path / "g.grid.nc", [shortest_m, 2.0 * shortest_m, 0.0])
    assert mesh_point.thirty_second_supersample(grid) == factor
    # The guarantee itself: a factor-N sample grid lies within s/sqrt(2) of any
    # point, inside the dcEdge/2 disc every Voronoi cell holds.
    spacing = mesh_point.THIRTY_SECOND_PIXEL_M / factor
    if factor > 1:
        assert spacing / 2 ** 0.5 < shortest_m / 2.0


def test_place_stage_pairs_events_and_preserves_work_error(monkeypatch):
    from types import ModuleType

    events = []
    ledger = ModuleType("woof.timeline.ledger")
    ledger.emit = lambda stage, phase, **context: events.append((stage, phase, context))
    monkeypatch.setitem(sys.modules, "woof.timeline.ledger", ledger)
    with pytest.raises(PointPlanRefusal, match="build failed"):
        with mesh_point._place_stage("prep.place.mesh", attrs={"arm": "COLD"}):
            raise PointPlanRefusal("build failed")
    assert [event[:2] for event in events] == [
        ("prep.place.mesh", "begin"), ("prep.place.mesh", "end")]
    assert all(event[2] == {"product": "hex", "attrs": {"arm": "COLD"}} for event in events)


def test_place_stage_never_raises_from_timing(monkeypatch):
    from types import ModuleType

    ledger = ModuleType("woof.timeline.ledger")

    def failed_append(*args, **kwargs):
        raise OSError("ledger full")

    ledger.emit = failed_append
    monkeypatch.setitem(sys.modules, "woof.timeline.ledger", ledger)
    with mesh_point._place_stage("prep.place.static"):
        result = 17
    assert result == 17
    with pytest.raises(PointPlanRefusal, match="static failed"):
        with mesh_point._place_stage("prep.place.static"):
            raise PointPlanRefusal("static failed")

REGISTERED_SPEC = ROOT.parent.parent.parent / "woof" / "evidence" / "fine-mesh-20260829" / "meshes" / "v0.9.120.110533.spec.json"


# ---------------------------------------------------------------------------
# the ladder is the registered recipe, moved
# ---------------------------------------------------------------------------
def test_the_ladder_at_the_registered_point_is_the_registered_spec() -> None:
    spec = mesh_point.ladder_spec((35.0, -97.0), fine_dx_m=937.5, radius_km=100.0)
    rows = [(r["shape"]["radius_km"], r["spacing_km"], r["transition_km"]) for r in spec["regions"]]
    assert spec["background_km"] == 120.0
    assert rows == [
        (3289.375, 60.0, 1080.0), (1669.375, 30.0, 540.0), (859.375, 15.0, 270.0),
        (454.375, 7.5, 135.0), (251.875, 3.75, 67.5), (150.625, 1.875, 33.75),
        (100.0, 0.9375, 16.875),
    ]
    assert all(r["shape"]["center_deg"] == [35.0, -97.0] for r in spec["regions"])


@pytest.mark.skipif(not REGISTERED_SPEC.is_file(), reason="the registered spec of record is not on this machine")
def test_the_ladder_reproduces_the_spec_of_record_to_the_digit() -> None:
    reference = json.loads(REGISTERED_SPEC.read_text(encoding="utf-8"))
    spec = mesh_point.ladder_spec((35.0, -97.0), fine_dx_m=937.5, radius_km=100.0)
    for ours, theirs in zip(spec["regions"], reference["regions"]):
        assert ours["shape"]["radius_km"] == pytest.approx(theirs["shape"]["radius_km"])
        assert ours["spacing_km"] == theirs["spacing_km"]
        assert ours["transition_km"] == theirs["transition_km"]
    assert len(spec["regions"]) == len(reference["regions"])


def test_moving_the_point_moves_every_cap_and_nothing_else() -> None:
    here = mesh_point.ladder_spec((39.1, -94.58), fine_dx_m=937.5, radius_km=100.0)
    there = mesh_point.ladder_spec((35.0, -97.0), fine_dx_m=937.5, radius_km=100.0)
    for a, b in zip(here["regions"], there["regions"]):
        assert a["shape"]["center_deg"] == [39.1, -94.58]
        assert a["shape"]["radius_km"] == b["shape"]["radius_km"]
        assert a["spacing_km"] == b["spacing_km"]


def test_a_spacing_off_the_ladder_is_refused_naming_both_rungs() -> None:
    with pytest.raises(PointPlanRefusal) as refusal:
        mesh_point.ladder_rungs(120.0, 900.0)
    text = str(refusal.value)
    assert "937.5 m" in text and "468.75 m" in text
    assert "ALWAYS FINER" in text


def test_the_ladder_rungs_halve_from_the_background() -> None:
    assert mesh_point.ladder_rungs(120.0, 937.5) == [60.0, 30.0, 15.0, 7.5, 3.75, 1.875, 0.9375]
    assert mesh_point.ladder_rungs(120.0, 1875.0) == [60.0, 30.0, 15.0, 7.5, 3.75, 1.875]
    with pytest.raises(PointPlanRefusal):
        mesh_point.ladder_rungs(120.0, 120_000.0)


def test_point_parsing_refuses_what_is_not_a_point() -> None:
    assert mesh_point.parse_point("39.1,-94.58") == (39.1, -94.58)
    assert mesh_point.parse_point("39.1, 265.42") == (39.1, pytest.approx(-94.58))
    for bad in ("39.1", "x,y", "95,0"):
        with pytest.raises(PointPlanRefusal):
            mesh_point.parse_point(bad)


# ---------------------------------------------------------------------------
# pricing
# ---------------------------------------------------------------------------
def test_the_cull_prediction_bounds_the_registered_cull_from_above_within_three_percent() -> None:
    """r0.9.120.40520: a 125 km cap (pad 1.25) of the 100 km core -> 40,520 cells."""

    spec = mesh_point.ladder_spec((35.0, -97.0), fine_dx_m=937.5, radius_km=100.0)
    cull = mesh_point.predicted_cull_cells(
        spec=spec, radius_km=100.0, pad_scale=1.25, attained_fine_km=0.94699,
    )
    assert cull["basis"] == "area_integral"
    assert 40_520 <= cull["predicted_cells"] <= 40_520 * 1.03
    assert cull["fine_flat_radius_km"] == pytest.approx(83.125)


def test_the_spacing_profile_is_flat_inside_the_ramp_and_ramps_to_the_next_rung() -> None:
    spec = mesh_point.ladder_spec((35.0, -97.0), fine_dx_m=937.5, radius_km=100.0)
    profile = mesh_point.spacing_profile(spec)
    assert profile[0] == (100.0, 0.9375, 16.875)
    assert mesh_point.spacing_at(profile, 50.0, 120.0) == 0.9375
    assert mesh_point.spacing_at(profile, 100.0, 120.0) == pytest.approx(1.875)
    assert mesh_point.spacing_at(profile, 110.0, 120.0) == pytest.approx(1.875)
    assert mesh_point.spacing_at(profile, 150.625, 120.0) == pytest.approx(3.75)
    assert mesh_point.spacing_at(profile, 5000.0, 120.0) == 120.0


def test_the_card_table_prices_on_the_limited_area_row_with_the_models_own_margin() -> None:
    card = mesh_point.resolve_card("32gb")
    assert card["admission_card"] == "32gib-170sm" and card["generator_card"] == "rtx-5090"
    verdict = mesh_point.device_verdict(42_000, card)
    assert verdict["row"] == "limited-area/170sm" and verdict["row_measured"]
    assert verdict["fits"] and verdict["short_by_mib"] == 0.0
    assert verdict["required_free_mib"] == pytest.approx(
        verdict["predicted_mib"] + verdict["margin_mib"], abs=0.2)
    small = mesh_point.device_verdict(42_000, card, budget_mib=4_000.0)
    assert not small["fits"] and small["short_by_mib"] > 0.0
    assert small["budget_basis"] == "--vram-gib"


def test_an_unmeasured_card_is_refused_by_name() -> None:
    with pytest.raises(PointPlanRefusal) as refusal:
        mesh_point.resolve_card("h100")
    assert "not a card this plan can price" in str(refusal.value)
    assert mesh_point.resolve_card("10gb")["generator_card"] is None


def test_the_timestep_is_the_largest_anchored_one_under_the_courant_limit() -> None:
    """RETIRED 2026-10-10 as an anchor choice: MPAS-A's 6 s per km, stepped
    down to the first whole-second divisor of the radiation interval the
    physical stability check admits; anchors are recorded, not required."""

    chosen = mesh_point.choose_timestep(869.25, fine_dx_m=937.5)
    assert chosen["dt_seconds"] == 5.0 and chosen["cumulus_scheme"] is None
    assert chosen["courant_limit_seconds"] == pytest.approx(8.0665, abs=1e-3)
    assert "anchor" in chosen and chosen["anchor"]["matched"] is False
    coarse = mesh_point.choose_timestep(20_000.0, fine_dx_m=15_000.0)
    assert coarse["dt_seconds"] == 90.0 and coarse["cumulus_scheme"] == "gf"
    # 300 m: MPAS-A's rule gives 1.8 s; the largest whole-second divisor of
    # 1800 s at or under it is 1 s, which the check admits.
    fine = mesh_point.choose_timestep(300.0, fine_dx_m=300.0)
    assert fine["dt_seconds"] == 1.0
    with pytest.raises(PointPlanRefusal) as refusal:
        mesh_point.choose_timestep(10.0, fine_dx_m=300.0)
    assert "is stable on this mesh" in str(refusal.value)


def test_the_cull_region_is_a_cap_at_the_pad_and_a_pad_under_one_is_refused() -> None:
    region = mesh_point.cull_region((39.1, -94.58), 100.0, 1.35)
    assert region == {"kind": "cap", "center_deg": [39.1, -94.58], "radius_km": 135.0}
    with pytest.raises(PointPlanRefusal):
        mesh_point.cull_region((39.1, -94.58), 100.0, 0.9)


# ---------------------------------------------------------------------------
# the door, with a faked generator
# ---------------------------------------------------------------------------
FAKE_RECEIPT = {
    "predicted_cells": 110_437.7,
    "footprint_mib": 14_515.9,
    "card": "rtx-5090",
    "steepest_requested_gradient_percent_per_cell": 10.97,
    "ladder_snap": {"moved": False, "regions": []},
    "region_attainment": [{"attained_spacing_km": 0.9470}],
}


@pytest.fixture
def fake_mesh_exe(tmp_path: Path) -> Path:
    """An ``rw_mpas_mesh`` that prints one receipt for any ``--dry-run``."""

    script = tmp_path / "rw_mpas_mesh.py"
    script.write_text(
        "import json, sys\n"
        "args = sys.argv[1:]\n"
        "assert '--dry-run' in args, args\n"
        f"print({json.dumps(FAKE_RECEIPT)!r})\n",
        encoding="utf-8",
    )
    if sys.platform == "win32":
        launcher = tmp_path / "rw_mpas_mesh.cmd"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        launcher = tmp_path / "rw_mpas_mesh"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        launcher.chmod(0o755)
    return launcher


def test_the_plan_prices_the_parent_the_cull_and_the_card(fake_mesh_exe: Path, monkeypatch) -> None:
    from woof.hex import mesh_spec_gates

    monkeypatch.setattr(
        mesh_spec_gates, "gates_from_receipt",
        lambda spec, receipt, measure=None: {
            "transition_band": {"steepest_gradient_percent_per_cell": 10.97, "band_cells": 6.5,
                                "band_cells_floor": 6, "gradient_percent_per_cell_ceiling": 12.25},
            "short_dual_edge_floor": {"verdict": "not decidable", "limit_m": 200, "gate": "storage"},
        },
    )
    parser = build_parser()
    arguments = parser.parse_args([
        "mesh-plan", "--point", "39.1,-94.58", "--fine-dx-m", "937.5", "--radius-km", "100",
        "--card", "32gb", "--mesh-exe", str(fake_mesh_exe),
    ])
    request = mesh_point.request_from_arguments(arguments)
    plan = mesh_point.plan_point(request)
    assert plan["schema"] == "gpuwm-hex.point-plan/v1"
    assert plan["parent"]["predicted_cells"] == pytest.approx(110_437.7)
    assert plan["cull"]["attained_fine_spacing_km"] == pytest.approx(0.947)
    assert 40_000 < plan["cull"]["predicted_cells"] < 46_000
    assert plan["device"]["fits"] and plan["device"]["row"] == "limited-area/170sm"
    assert plan["device"]["max_core_radius_km_on_this_card"] > 100.0
    assert plan["cull_region"]["radius_km"] == pytest.approx(135.0)
    report = "\n".join(mesh_point._report(plan))
    assert "FITS" in report and "limited-area/170sm" in report


def test_the_front_door_takes_point_or_spec_and_refuses_both_or_neither(capsys) -> None:
    assert main(["mesh-plan"]) == 2
    assert "neither --spec nor --point" in capsys.readouterr().err
    assert main(["mesh-plan", "--spec", "x.json", "--point", "1,2"]) == 2
    assert "both given" in capsys.readouterr().err


def test_generate_without_an_out_dir_refuses_before_pricing_anything(fake_mesh_exe: Path, monkeypatch) -> None:
    from woof.hex import mesh_spec_gates

    monkeypatch.setattr(mesh_spec_gates, "gates_from_receipt", lambda *a, **k: {})
    parser = build_parser()
    arguments = parser.parse_args([
        "mesh-plan", "--point", "39.1,-94.58", "--generate", "--mesh-exe", str(fake_mesh_exe),
    ])
    with pytest.raises(PointPlanRefusal) as refusal:
        mesh_point.run_point(arguments)
    assert "--generate needs --out-dir" in str(refusal.value)


def test_a_cull_that_does_not_fit_the_card_is_refused_before_anything_is_built(
    fake_mesh_exe: Path, monkeypatch, tmp_path: Path
) -> None:
    from woof.hex import mesh_spec_gates

    monkeypatch.setattr(mesh_spec_gates, "gates_from_receipt", lambda *a, **k: {})
    parser = build_parser()
    arguments = parser.parse_args([
        "mesh-plan", "--point", "39.1,-94.58", "--card", "32gb", "--vram-gib", "3",
        "--generate", "--out-dir", str(tmp_path / "out"), "--mesh-exe", str(fake_mesh_exe),
    ])
    with pytest.raises(PointPlanRefusal) as refusal:
        mesh_point.run_point(arguments)
    text = str(refusal.value)
    assert "refused before anything is built" in text and "reduce --radius-km" in text
    assert not (tmp_path / "out").exists()


def test_the_row_names_carry_the_spacing_the_background_the_count_and_the_point() -> None:
    assert mesh_point.parent_row_name((39.1, -94.58), 937.5, 120.0, 110_600) == "p0.9375.120.110600.n39.10w94.58"
    assert mesh_point.cull_row_name((39.1, -94.58), 937.5, 120.0, 42_000) == "q0.9375.120.42000.n39.10w94.58"


# ---------------------------------------------------------------------------
# the halo route
# ---------------------------------------------------------------------------
def test_the_halo_reaches_the_cut_plus_the_halo_and_a_flat_halo_is_refused() -> None:
    region = mesh_point.halo_region((37.68, -122.27), 100.0, 1.35, 1600.0)
    assert region == {"kind": "cap", "center_deg": [37.68, -122.27], "radius_km": 1735.0}
    with pytest.raises(PointPlanRefusal) as refusal:
        mesh_point.halo_region((37.68, -122.27), 100.0, 1.35, 0.0)
    assert "--halo-km" in str(refusal.value)


def test_the_door_defaults_to_the_halo_and_takes_the_parent_static_route_by_name() -> None:
    parser = build_parser()
    plain = parser.parse_args(["mesh-plan", "--point", "39.1,-94.58"])
    assert plain.parent_static is False and plain.halo_km == mesh_point.DEFAULT_HALO_KM
    chosen = parser.parse_args(["mesh-plan", "--point", "39.1,-94.58", "--parent-static",
                                "--halo-km", "900"])
    assert chosen.parent_static is True and chosen.halo_km == 900.0


def test_a_parent_is_admitted_on_the_lengths_its_static_would_store(tmp_path: Path) -> None:
    import numpy as np

    netCDF4 = pytest.importorskip("netCDF4")
    rng = np.random.default_rng(3)
    path = tmp_path / "unit.grid.nc"
    with netCDF4.Dataset(str(path), "w", format="NETCDF3_64BIT_OFFSET") as ds:
        ds.createDimension("nCells", 70)
        ds.createDimension("nEdges", 200)
        xyz = rng.normal(size=(3, 70))
        xyz /= np.linalg.norm(xyz, axis=0)
        for axis, values in zip("xyz", xyz):
            ds.createVariable(f"{axis}Cell", "f8", ("nCells",))[:] = values
        dv = rng.random(200) * 1e-4
        dc = rng.random(200) * 1e-4 + 1e-5
        ds.createVariable("dvEdge", "f8", ("nEdges",))[:] = dv
        ds.createVariable("dcEdge", "f8", ("nEdges",))[:] = dc
    got_dv, got_dc = mesh_point.static_equivalent_lengths(path)
    # rw_mpas_static: scale the unit-sphere float64 by the earth radius, store float32.
    assert got_dv.dtype == np.float32 and got_dc.dtype == np.float32
    assert np.array_equal(got_dv, (dv * 6_371_229.0).astype(np.float32))
    assert np.array_equal(got_dc, (dc * 6_371_229.0).astype(np.float32))


def _fake_culler(tmp_path: Path, body: str) -> Path:
    """An ``rw_mpas_mesh --cull-parent`` whose behaviour ``body`` decides."""

    script = tmp_path / "fake_cull.py"
    script.write_text(
        "import sys\n"
        "args = sys.argv[1:]\n"
        "out = args[args.index('--out') + 1]\n"
        "graph = args[args.index('--graph') + 1]\n" + body,
        encoding="utf-8",
    )
    if sys.platform == "win32":
        launcher = tmp_path / "fake_cull.cmd"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        launcher = tmp_path / "fake_cull"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        launcher.chmod(0o755)
    return launcher


@pytest.mark.parametrize("body,expect", [
    ("open(out, 'wb').write(b'CUT'); open(graph, 'wb').write(b'GRAPH')\n", None),
    ("open(out, 'wb').write(b'CUX'); open(graph, 'wb').write(b'GRAPH')\n", "different bytes"),
    ("sys.stderr.write('parent cellsOnCell[cell=40,slot=1]=0 is outside 1..=9 in a VALID slot\n'); sys.exit(1)\n",
     "cannot cut a region out of a culled file"),
], ids=["agrees", "differs", "old-engine"])
def test_the_halo_route_is_taken_only_when_the_engine_cuts_the_halo_as_it_cuts_the_parent(
    tmp_path: Path, body: str, expect: str | None
) -> None:
    direct, graph = tmp_path / "cull.grid.nc", tmp_path / "cull.graph.info"
    direct.write_bytes(b"CUT")
    graph.write_bytes(b"GRAPH")
    note = mesh_point.prove_regional_cut(
        mesh_exe=_fake_culler(tmp_path, body), work_grid=tmp_path / "halo.grid.nc",
        region=tmp_path / "region.json", direct_cut=direct, direct_graph=graph,
        scratch_stem=tmp_path / "probe",
    )
    if expect is None:
        assert note is None
    else:
        assert expect in note
    assert not list(tmp_path.glob("probe*")), "the probe left its files behind"


def _halo_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, narrow_edit=None,
                dtype="float64", signed_zero=False) -> dict:
    """``check_halo_vertical`` on a halo and parent sharing a 4-cell, 3-edge cut.

    The engines and the vertical construction are stood in for; the comparison
    the door makes on their output is the real one.
    """

    import numpy as np
    from types import SimpleNamespace

    from woof.hex import cull_door, vertical_spec
    from woof.hex.mesh import Mesh

    rng = np.random.default_rng(11)
    wide_grid = tmp_path / "halo.grid.nc"
    narrow_grid = tmp_path / "parent.grid.nc"
    index = {  # where the cut's cells and edges sit in each halo
        ("Cell", wide_grid.name): np.array([2, 3, 5, 7]), ("Cell", narrow_grid.name): np.array([0, 1, 3, 5]),
        ("Edge", wide_grid.name): np.array([1, 4, 6]), ("Edge", narrow_grid.name): np.array([0, 2, 3]),
    }

    def fields(n_cells: int, n_edges: int, cells: np.ndarray, edges: np.ndarray, source: dict) -> tuple:
        out = {}
        for name, kind in mesh_point._HALO_CHECKED:
            shape = {"zb": (5, 2), "zb3": (5, 2)}.get(name, (5,))
            full = rng.normal(size=(*shape, n_cells if kind == "Cell" else n_edges)).astype(dtype)
            if source:  # the narrow halo carries the wide halo's values on the cut
                full[..., cells if kind == "Cell" else edges] = source[name]
            out[name] = full
        return out

    wide = fields(10, 8, index[("Cell", wide_grid.name)], index[("Edge", wide_grid.name)], {})
    if signed_zero:
        wide["hx"][0, index[("Cell", wide_grid.name)][1]] = 0.0
    on_cut = {name: np.take(wide[name], index[(kind, wide_grid.name)], axis=-1)
              for name, kind in mesh_point._HALO_CHECKED}
    narrow = fields(8, 5, index[("Cell", narrow_grid.name)], index[("Edge", narrow_grid.name)], on_cut)
    if narrow_edit is not None:
        narrow_edit(narrow, index[("Cell", narrow_grid.name)], index[("Edge", narrow_grid.name)])

    def built(values: dict) -> tuple:
        return (SimpleNamespace(**{k: values[k] for k in ("hx", "zgrid", "zz", "dss", "zxu")}),
                SimpleNamespace(zb=values["zb"], zb3=values["zb3"]))

    monkeypatch.setattr(cull_door, "cull_one", lambda *a, **k: {})
    monkeypatch.setattr(Mesh, "from_netcdf", classmethod(lambda cls, grid, static=None: Path(grid).name))
    monkeypatch.setattr(vertical_spec.VerticalSpec, "from_file", classmethod(lambda cls, path: None))
    monkeypatch.setattr(vertical_spec, "build_vertical_from_spec",
                        lambda mesh, spec, regional_halo=False: built(wide if mesh == wide_grid.name else narrow))
    monkeypatch.setattr(mesh_point, "_element_index", lambda sub, sup, kind: index[(kind, Path(sup).name)])
    return mesh_point.check_halo_vertical(
        work_grid=wide_grid, work_static=tmp_path / "halo.static.nc",
        parent_grid=narrow_grid, parent_static=tmp_path / "parent-terrain.static.nc",
        vertical_spec=tmp_path / "spec.json", cut_grid=tmp_path / "cut.grid.nc",
        log=lambda *a: None,
    )


def test_the_halo_check_agrees_when_every_cut_value_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    check = _halo_check(tmp_path, monkeypatch)
    assert check["agreed"] is True and check["differing"] == {}
    # 5 values a cell for 5 cell fields on 4 cells, 5 for zxu and 10 for zb, zb3 on 3 edges
    assert check["values_compared"] == 4 * 5 * 4 + 3 * 5 + 2 * 3 * 10


@pytest.mark.parametrize("field,kind", mesh_point._HALO_CHECKED)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_halo_check_refuses_on_one_moved_value_and_counts_values_not_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field, kind, dtype
) -> None:
    def move_one(narrow: dict, cells, edges) -> None:
        selected = cells if kind == "Cell" else edges
        key = (0,) * (narrow[field].ndim - 1) + (selected[1],)
        narrow[field][key] = -7.25 if narrow[field][key] != -7.25 else 3.5

    check = _halo_check(tmp_path, monkeypatch, narrow_edit=move_one, dtype=dtype)
    assert check["agreed"] is False
    assert check["differing"] == {field: 1}, "one value moved; a byte count reports several"


def test_the_parent_check_detects_signed_zero_bits(tmp_path, monkeypatch) -> None:
    def move_one(parent, cells, edges):
        parent["hx"][0, cells[1]] = -0.0
    check = _halo_check(tmp_path, monkeypatch, narrow_edit=move_one, signed_zero=True)
    assert not check["agreed"] and check["differing"] == {"hx": 1}


# ---------------------------------------------------------------------------
# the halo route's decision: every check before anything is registered or cut
# ---------------------------------------------------------------------------
def _generate_with(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, static_checks=(),
                   vertical_checks=(), static_refuses=(), terrain_refuses=False,
                   missing_flag=None, levels=55, static_cache=None,
                   mislabelled_statics=()) -> tuple[dict, dict]:
    """``generate_point`` with every engine and measurement stood in for.

    Each stand-in writes bytes that name what they were made from (a cut
    names its region's radius and its source's bytes, a static the bytes of
    the grid it was built on), so the test reads which static the cut's
    static was cut from, and which one the receipt its row names describes.
    The route decision and the order of the work are the real ones.
    """

    from types import SimpleNamespace

    from woof.hex import cull_door, engines, mesh_rows, vertical_spec

    seen: dict = {"statics": [], "terrains": [], "vertical": None, "rows": [], "lineage": [],
                  "parent_authorities": []}
    static_answers, vertical_answers = iter(static_checks), iter(vertical_checks)

    def fake_run(argv, *, what, log=None):
        argv = [str(item) for item in argv]
        out = Path(argv[argv.index("--out") + 1])
        if what == "rw_mpas_mesh":
            out.write_bytes(b"PARENT")
        else:
            grid = Path(argv[argv.index("--grid") + 1])
            if terrain_refuses and "--terrain-only" in argv:
                raise PointPlanRefusal("rw_mpas_static exited 1: unknown argument --terrain-only")
            seen["terrains" if "--terrain-only" in argv else "statics"].append(grid.read_bytes().decode())
            if grid.read_bytes().decode() in static_refuses:
                raise PointPlanRefusal("rw_mpas_static exited 1.  Last line: no neighbour has a value to carry")
            out.write_bytes(b"STATIC on " + grid.read_bytes())
            Path(argv[argv.index("--receipt") + 1]).write_text(
                json.dumps({"grid_bytes": grid.read_bytes().decode()}), encoding="utf-8")
        return SimpleNamespace(elapsed=0.0)

    def fake_cull(engine, parent, region, out, *, graph=None, clobber=False, provenance_grid=None):
        radius = json.loads(Path(region).read_text(encoding="utf-8"))["radius_km"]
        Path(out).write_bytes(f"CUT {radius:g} OF ".encode() + Path(parent).read_bytes())
        if graph is not None:
            Path(graph).write_bytes(b"GRAPH")
        return {"receipt": {"region_cells": 10}}

    def fake_static_check(**kwargs):
        agreed = next(static_answers)
        return {"agreed": agreed, **({} if agreed else {"why": "albedo12m was carried 400 rings"})}

    def fake_vertical_check(**kwargs):
        seen["parent_authorities"].append(kwargs["parent_built"])
        agreed = next(vertical_answers)
        return {"agreed": agreed, "values_compared": 1, "differing": {} if agreed else {"hx": 5},
                "seconds": 0.0, "built": ("BUILT ON", Path(kwargs["work_static"]).read_bytes()),
                "parent_built": "PARENT AUTHORITY"}

    admitted = {"regional": False, "n_cells": 100, "timestep_choice": {"dt_seconds": 5.0}}

    def describe_generated(**kwargs):
        return SimpleNamespace(name="p", n_cells=100, n_edges=300, dt_seconds=5.0,
                               n_levels=kwargs["n_levels"], n_interfaces=kwargs["n_levels"] + 1,
                               static=kwargs["static"], static_receipt=kwargs["static_receipt"])

    def describe_cull(**kwargs):
        return SimpleNamespace(name="q", n_cells=10, n_edges=30, boundary_zone_width=7,
                               n_levels=kwargs["parent"].n_levels, n_interfaces=kwargs["parent"].n_interfaces,
                               bdy_mask_sha256="0" * 64, static_receipt=kwargs.get("static_receipt"),
                               static=kwargs["static"])

    def materialize(*, grid, static, spec_path, output, receipt_path, regional_halo, prebuilt):
        seen["vertical"] = {"grid": Path(grid).read_bytes(), "static": Path(static).read_bytes(),
                            "regional_halo": regional_halo, "prebuilt": prebuilt}
        Path(output).write_bytes(b"VERTICAL on " + Path(static).read_bytes())
        return {"invariants": {"n_vert_levels": levels}}

    monkeypatch.setattr(engines, "resolve", lambda spec, explicit=None: tmp_path / "engine")
    if static_cache is None:
        # the in-place route (no hex cache configured), whatever the host's environment says
        monkeypatch.setenv("WOOF_HEX_STATIC_CACHE", "off")
        monkeypatch.setattr(mesh_point, "resolve_geog", lambda explicit: tmp_path)
    else:
        # woof.hex.static_cache keys by the builder's and every geography file's bytes
        (tmp_path / "engine").write_bytes(b"static builder v1")
        geog = tmp_path / "geog"
        geog.mkdir(exist_ok=True)
        (geog / "tile").write_bytes(b"geography v1")
        monkeypatch.setenv("WOOF_HEX_STATIC_CACHE", str(static_cache))
        monkeypatch.setattr(mesh_point, "resolve_geog", lambda explicit: geog)
    monkeypatch.setattr(mesh_point, "_supports_flag", lambda engine, flag: flag != missing_flag)
    monkeypatch.setattr(mesh_point, "_run", fake_run)
    # The stand-in statics are bytes, not netCDF: the coordinate check answers
    # from the grid each one names (tests/test_static_coordinate_precision.py
    # grades the real check on real files).
    monkeypatch.setattr(mesh_point, "static_coordinate_errors", lambda grid, static: (
        [f"{Path(static).name} declares binary64_earth_centred, but its positions are off the grid"]
        if Path(grid).read_bytes().decode() in mislabelled_statics else []))
    monkeypatch.setattr(cull_door, "cull_one", fake_cull)
    # carry_lineage stamps the file it is handed, so the probe must see the cut
    # before its lineage is carried (the cut as the engine wrote it).
    monkeypatch.setattr(cull_door, "carry_lineage",
                        lambda parent, out, **k: seen["lineage"].append(Path(out).name) or {})
    monkeypatch.setattr(mesh_point, "prove_regional_cut", lambda **k: (
        "the cut was stamped before the probe" if Path(k["direct_cut"]).name in seen["lineage"] else None))
    monkeypatch.setattr(mesh_point, "admit_parent_grid", lambda grid, **k: dict(admitted))
    monkeypatch.setattr(mesh_point, "admit_pair",
                        lambda grid, static, **k: {**admitted, "regional": "cull" in Path(grid).name})
    monkeypatch.setattr(mesh_point, "check_halo_static", fake_static_check)
    monkeypatch.setattr(mesh_point, "check_halo_vertical", fake_vertical_check)
    monkeypatch.setattr(mesh_rows, "describe_generated", describe_generated)
    monkeypatch.setattr(mesh_rows, "describe_cull", describe_cull)
    monkeypatch.setattr(mesh_rows, "write_rows", lambda path, rows: seen["rows"].extend(rows))
    monkeypatch.setattr(mesh_rows, "append_row", lambda path, row: seen["rows"].append(row))
    monkeypatch.setattr(vertical_spec, "materialize_vertical_artifact", materialize)

    request = mesh_point.PointRequest(
        point=(37.68, -122.27), fine_dx_m=937.5, radius_km=100.0, background_km=120.0,
        transition_factor=18.0, ring_factor=3.0, cull_pad_scale=1.35, card=None, vram_gib=None,
        mesh_exe=None, name="t",
    )
    plan = {"spec": {}, "cull_region": mesh_point.cull_region((37.68, -122.27), 100.0, 1.35),
            "cull": {"predicted_cells": 10}}
    (tmp_path / "spec.json").write_text(json.dumps({"n_vert_levels": levels}))
    receipt = mesh_point.generate_point(
        request, plan, out_dir=tmp_path / "out", geog=None, vertical_spec=tmp_path / "spec.json",
        clobber=False, log=lambda *a: None,
    )
    return receipt, seen


@pytest.mark.parametrize("levels", [41, 55, 59])
def test_generated_rows_follow_declared_vertical_height(tmp_path, monkeypatch, levels):
    receipt, seen = _generate_with(tmp_path, monkeypatch, static_checks=[True],
                                   vertical_checks=[True], levels=levels)
    assert [row.n_levels for row in seen["rows"]] == [levels, levels]
    assert [row.n_interfaces for row in seen["rows"]] == [levels + 1, levels + 1]
    assert receipt["vertical"]["n_vert_levels"] == levels


def test_a_halo_that_holds_gives_the_cut_its_static_its_receipt_and_its_vertical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, seen = _generate_with(tmp_path, monkeypatch, static_checks=[True], vertical_checks=[True])
    out = tmp_path / "out"
    assert receipt["route"] == "halo"
    assert seen["statics"] == ["CUT 1735 OF PARENT"], "only the halo's geography is built"
    assert (out / "t-cull.static.nc").read_bytes() == b"CUT 135 OF STATIC on CUT 1735 OF PARENT"
    parent, cull = seen["rows"]
    assert parent.static is None and cull.static_receipt == out / "t-halo.static.receipt.json"
    assert json.loads(cull.static_receipt.read_text())["grid_bytes"] == "CUT 1735 OF PARENT"
    assert seen["vertical"]["regional_halo"] is True
    assert seen["vertical"]["prebuilt"] == ("BUILT ON", b"STATIC on CUT 1735 OF PARENT")
    assert not (out / "t-halo.grid.nc").exists() and not (out / "t-halo.static.nc").exists()


def test_a_static_that_does_not_hold_its_declaration_stops_the_door_at_the_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2026-10-15: a static builder that reads its grid
    through binary32 (gpuwm 2.9.0's) wrote a static declaring binary64
    coordinates it did not hold, and the door refused later, twice, blaming
    the mesh.  It now refuses right after the first static, naming the
    builder, and does not fall back to a wider halo or the parent's geography,
    which the same builder would only mislabel again."""

    with pytest.raises(mesh_point.StaticBuilderRefusal) as refusal:
        _generate_with(tmp_path, monkeypatch, static_checks=[True], vertical_checks=[True],
                       mislabelled_statics=("CUT 1735 OF PARENT",))
    message = str(refusal.value)
    assert "t-halo.static.nc declares binary64_earth_centred" in message
    assert "The builder was" in message and "46c8a71513e" in message
    assert not isinstance(refusal.value, mesh_point.PointPlanRefusal)
    out = tmp_path / "out"
    assert not (out / "t-cull.static.nc").exists() and not (out / "mesh-rows.json").exists()


def test_a_doubled_halo_is_the_one_the_cut_and_its_row_are_taken_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Before 2026-10-01 the cut's static was cut from the first halo and its
    # row written first; doubling then replaced the static receipt that row
    # named with the wider halo's.
    receipt, seen = _generate_with(tmp_path, monkeypatch, static_checks=[True, True],
                                   vertical_checks=[False, True])
    out = tmp_path / "out"
    assert receipt["route"] == "halo"
    assert seen["statics"] == ["CUT 1735 OF PARENT", "CUT 3335 OF PARENT"]
    assert (out / "t-cull.static.nc").read_bytes() == b"CUT 135 OF STATIC on CUT 3335 OF PARENT"
    cull = seen["rows"][1]
    assert json.loads(cull.static_receipt.read_text())["grid_bytes"] == "CUT 3335 OF PARENT"
    assert seen["vertical"]["prebuilt"] == ("BUILT ON", b"STATIC on CUT 3335 OF PARENT")
    assert [c["halo_km"] for c in receipt["halo"]["checks"]] == [1600.0, 3200.0]
    assert seen["terrains"] == ["PARENT"]
    assert seen["parent_authorities"] == [None, "PARENT AUTHORITY"]


@pytest.mark.parametrize("static_checks,vertical_checks,static_refuses", [
    ([True, True], [False, False], ()),
    ([False, False], [], ()),
    ([], [], ("CUT 1735 OF PARENT", "CUT 3335 OF PARENT")),
], ids=["vertical-never-settles", "static-not-shown", "static-builder-refuses-the-halo"])
def test_a_halo_that_never_holds_takes_the_parent_route_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, static_checks, vertical_checks, static_refuses
) -> None:
    # Before 2026-10-01 the vertical fall back built the parent's geography
    # for the vertical grid only and kept the cut's static cut from the halo,
    # and a static the halo could not hold had no check at all.
    receipt, seen = _generate_with(tmp_path, monkeypatch, static_checks=static_checks,
                                   vertical_checks=vertical_checks, static_refuses=static_refuses)
    out = tmp_path / "out"
    assert receipt["route"] == "parent-static"
    assert seen["statics"][-1] == "PARENT"
    assert (out / "t-cull.static.nc").read_bytes() == b"CUT 135 OF STATIC on PARENT"
    parent, cull = seen["rows"]
    assert parent.static == out / "t.static.nc", "the parent row is runnable on this route"
    assert cull.static_receipt is None, "the cut names the static receipt of the parent"
    assert seen["vertical"] == {"grid": b"PARENT", "static": b"STATIC on PARENT",
                                "regional_halo": False, "prebuilt": None}
    assert receipt["halo"]["not_used"]


def test_an_engine_without_the_parent_terrain_authority_keeps_the_parent_route(tmp_path, monkeypatch) -> None:
    receipt, seen = _generate_with(tmp_path, monkeypatch, static_checks=[True], terrain_refuses=True)
    assert receipt["route"] == "parent-static"
    assert seen["statics"][-1] == "PARENT"
    assert not seen["vertical"]["regional_halo"]
    assert "cannot emit the parent's terrain authority" in receipt["halo"]["not_used"]


def test_with_a_hex_cache_each_static_is_built_once_per_host_through_the_same_run(tmp_path, monkeypatch) -> None:
    """WOOF_HEX_STATIC_CACHE (or WOOF_HEX_CACHE_ROOT) set: the statics come from woof.hex.static_cache.

    THE BREAKAGE THIS PREVENTS: every mesh-point run and every ensemble preparation on one host rebuilt the same
    static.  The cached build still goes through ``_run`` (the per-mesh store and the timeline stage), the outputs
    link to the one cached artifact, and a second run of the same request builds nothing.
    """

    cache = tmp_path / "static-cache"
    receipt, seen = _generate_with(tmp_path, monkeypatch, static_checks=[True], vertical_checks=[True],
                                   static_cache=cache)
    out = tmp_path / "out"
    assert receipt["route"] == "halo"
    assert seen["statics"] == ["CUT 1735 OF PARENT"] and seen["terrains"] == ["PARENT"]
    assert (out / "t-cull.static.nc").read_bytes() == b"CUT 135 OF STATIC on CUT 1735 OF PARENT"
    cached = [leg for leg in receipt["legs"] if "static_cache" in leg]
    assert [leg["leg"] for leg in cached] == ["static", "parent-terrain"]
    assert not any(leg["static_cache"]["hit"] for leg in cached)
    assert all(Path(leg["static_cache"]["cache"]).parent == cache for leg in cached)
    assert len(list(cache.glob("*/complete.json"))) == 2

    # the same request again, into a fresh output folder: both statics are cache hits, nothing is built
    import shutil
    shutil.rmtree(out)
    receipt, seen = _generate_with(tmp_path, monkeypatch, static_checks=[True], vertical_checks=[True],
                                   static_cache=cache)
    assert seen["statics"] == [] and seen["terrains"] == []
    assert all(leg["static_cache"]["hit"] for leg in receipt["legs"] if "static_cache" in leg)
    assert (out / "t-cull.static.nc").read_bytes() == b"CUT 135 OF STATIC on CUT 1735 OF PARENT"


def test_the_door_uses_the_static_cache_only_when_a_hex_cache_is_configured(tmp_path) -> None:
    from woof.hex import static_cache

    assert static_cache.door_cache_dir({}) is None, "a bare install builds its statics in place"
    assert static_cache.door_cache_dir({"WOOF_HEX_CACHE_ROOT": str(tmp_path)}) == tmp_path / "static"
    assert static_cache.door_cache_dir({"WOOF_HEX_STATIC_CACHE": str(tmp_path / "s")}) == (tmp_path / "s").absolute()
    assert static_cache.door_cache_dir({"WOOF_HEX_STATIC_CACHE": "off", "WOOF_HEX_CACHE_ROOT": str(tmp_path)}) is None


@pytest.mark.parametrize("flag", ["--terrain-only", "--cull-provenance-grid"])
def test_missing_bridge_capabilities_choose_the_parent_before_building_halos(tmp_path, monkeypatch, flag) -> None:
    receipt, seen = _generate_with(tmp_path, monkeypatch, missing_flag=flag)
    assert receipt["route"] == "parent-static" and receipt["halo"] is None
    assert seen["statics"] == ["PARENT"] and seen["terrains"] == []
    assert flag in receipt["route_note"]


# ---------------------------------------------------------------------------
# the static guard, on real (tiny) files
# ---------------------------------------------------------------------------
def _strip_grid(path: Path, n_cells: int, rim: set[int], dc=None) -> None:
    """A line of cells, each touching the one before and after; the cells in
    ``rim`` also have one neighbour slot naming no cell (a cut-away one)."""

    import numpy as np

    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(str(path), "w", format="NETCDF3_64BIT_OFFSET") as ds:
        ds.createDimension("nCells", n_cells)
        ds.createDimension("maxEdges", 3)
        ds.createDimension("nEdges", n_cells)
        coc = np.zeros((n_cells, 3), dtype=np.int32)
        noc = np.zeros(n_cells, dtype=np.int32)
        for c in range(n_cells):
            slots = [k + 1 for k in (c - 1, c + 1) if 0 <= k < n_cells]
            if c in rim:
                slots.append(0)
            coc[c, :len(slots)] = slots
            noc[c] = len(slots)
        ds.createVariable("cellsOnCell", "i4", ("nCells", "maxEdges"))[:] = coc
        ds.createVariable("nEdgesOnCell", "i4", ("nCells",))[:] = noc
        for axis in "xyz":
            ds.createVariable(f"{axis}Cell", "f8", ("nCells",))[:] = np.arange(n_cells) + {"x": 0.0, "y": 0.5, "z": 0.25}[axis]
        ds.createVariable("dcEdge", "f8", ("nEdges",))[:] = dc if dc is not None else np.full(n_cells, 2.0)


def _static_receipt(path: Path, rounds: dict) -> Path:
    from woof.hex.mesh_rows import sha256_file
    all_rounds = {f: 0 for f in ("ter", "landuse", "soilcat_top", "greenfrac", "albedo12m",
                                "snoalb", "soiltemp", "soilcomp", "soilcl1", "soilcl2", "soilcl3", "soilcl4")}
    all_rounds.update(rounds)
    path.write_text(json.dumps({"stamped_provenance": {"grid_sha256": sha256_file(path.parent / "halo.grid.nc")},
                               "datasets": [{"field": f, "spread_rounds": r} if r is not None else {"field": f}
                                             for f, r in all_rounds.items()]}), encoding="utf-8")
    return path


def test_the_cut_is_measured_in_rings_from_the_halos_rim(tmp_path: Path) -> None:
    import numpy as np

    halo, cut = tmp_path / "halo.grid.nc", tmp_path / "cut.grid.nc"
    _strip_grid(halo, 30, rim={0, 29})
    _strip_grid(cut, 4, rim=set())
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(str(cut), "a") as ds:  # the cut is cells 13..16 of the halo
        for axis in "xyz":
            ds.variables[f"{axis}Cell"][:] = np.arange(13, 17) + {"x": 0.0, "y": 0.5, "z": 0.25}[axis]
    assert mesh_point.cut_to_rim_hops(halo, cut) == 13


@pytest.mark.parametrize("rounds,agreed,why", [
    ({"ter": 0, "albedo12m": 2}, True, None),
    ({"ter": 0, "albedo12m": 10}, False, "albedo12m was carried 10 rings"),
    ({"ter": 0, "albedo12m": None}, False, "older than its spread_rounds receipt"),
], ids=["near", "carried-past-the-rim", "old-engine"])
def test_the_static_holds_only_when_nothing_was_carried_from_past_the_rim(
    tmp_path: Path, rounds, agreed, why
) -> None:
    import numpy as np

    halo, cut, parent = tmp_path / "halo.grid.nc", tmp_path / "cut.grid.nc", tmp_path / "parent.grid.nc"
    _strip_grid(halo, 30, rim={0, 29})
    _strip_grid(parent, 40, rim=set())
    _strip_grid(cut, 4, rim=set())
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(str(cut), "a") as ds:
        for axis in "xyz":
            ds.variables[f"{axis}Cell"][:] = np.arange(13, 17) + {"x": 0.0, "y": 0.5, "z": 0.25}[axis]
    check = mesh_point.check_halo_static(
        halo_grid=halo, parent_grid=parent, static_receipt=_static_receipt(tmp_path / "r.json", rounds),
        cut_grid=cut, log=lambda *a: None,
    )
    assert check["agreed"] is agreed
    if why:
        assert why in check["why"]
    else:
        assert check["cut_to_rim_rings"] == 13 and check["reach_rings"] == 2 + mesh_point.STATIC_STENCIL_RINGS


def test_the_static_is_not_shown_when_the_halo_lost_the_parents_shortest_spacing(tmp_path: Path) -> None:
    import numpy as np

    halo, cut, parent = tmp_path / "halo.grid.nc", tmp_path / "cut.grid.nc", tmp_path / "parent.grid.nc"
    _strip_grid(halo, 30, rim={0, 29})
    shorter = np.full(40, 2.0)
    shorter[35] = 1.5  # the parent's shortest edge lies outside the halo
    _strip_grid(parent, 40, rim=set(), dc=shorter)
    _strip_grid(cut, 4, rim=set())
    check = mesh_point.check_halo_static(
        halo_grid=halo, parent_grid=parent, static_receipt=_static_receipt(tmp_path / "r.json", {"ter": 0}),
        cut_grid=cut, log=lambda *a: None,
    )
    assert check["agreed"] is False and "shortest dcEdge" in check["why"]


def test_a_cut_the_halo_does_not_hold_is_not_shown_and_does_not_stop_the_build(tmp_path: Path) -> None:
    import numpy as np

    halo, cut, parent = tmp_path / "halo.grid.nc", tmp_path / "cut.grid.nc", tmp_path / "parent.grid.nc"
    _strip_grid(halo, 30, rim={0, 29})
    _strip_grid(parent, 40, rim=set())
    _strip_grid(cut, 4, rim=set())
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(str(cut), "a") as ds:  # cells 100..103: not cells of the halo
        for axis in "xyz":
            ds.variables[f"{axis}Cell"][:] = np.arange(100, 104) + {"x": 0.0, "y": 0.5, "z": 0.25}[axis]
    check = mesh_point.check_halo_static(
        halo_grid=halo, parent_grid=parent, static_receipt=_static_receipt(tmp_path / "r.json", {"ter": 0}),
        cut_grid=cut, log=lambda *a: None,
    )
    assert check["agreed"] is False and "not in" in check["why"]


@pytest.mark.parametrize("damage", ["missing-field", "negative-rounds", "boolean-rounds", "wrong-grid"])
def test_a_static_proof_requires_complete_valid_receipts(tmp_path, damage) -> None:
    halo, cut, parent = (tmp_path / name for name in ("halo.grid.nc", "cut.grid.nc", "parent.grid.nc"))
    _strip_grid(halo, 30, rim={0, 29})
    _strip_grid(parent, 40, rim=set())
    _strip_grid(cut, 4, rim=set())
    receipt = _static_receipt(tmp_path / "r.json", {"ter": 0})
    payload = json.loads(receipt.read_text())
    if damage == "missing-field":
        payload["datasets"] = payload["datasets"][:-1]
    elif damage == "wrong-grid":
        payload["stamped_provenance"]["grid_sha256"] = "0" * 64
    else:
        payload["datasets"][0]["spread_rounds"] = -1 if damage == "negative-rounds" else True
    receipt.write_text(json.dumps(payload))
    check = mesh_point.check_halo_static(halo_grid=halo, parent_grid=parent,
                                        static_receipt=receipt, cut_grid=cut, log=lambda *args: None)
    assert not check["agreed"]
    assert check["why"]
