"""The moving ensemble must not reuse identity, clocks, or mesh covariances."""
import json
from pathlib import Path

import pytest

from woof.hex.cli import build_parser
from woof.hex.moving_members import (
    ROSTER_SCHEMA, MovingCaseError, moving_roster, moving_ensemble_products, stage_case,
)

CASES = Path(__file__).parents[1] / "verification/moving48/cases"


@pytest.fixture
def prepared(tmp_path):
    case = CASES / "tc-wp-2025110812.json"
    stage = tmp_path / "prepared"
    stage.mkdir()
    (stage / "stage.json").write_text(json.dumps(dict(source="gfs", cycle="2025-11-08T12")))
    member = dict(name="start", dt_seconds=18.75, center_deg=[13.3, 128.3])
    for key in ("grid", "static", "init", "mesh_rows"):
        path = stage / key
        path.write_bytes(key.encode())
        member[key] = str(path)
    (stage / "family.json").write_text(json.dumps([member]))
    return case, stage


def manifest(tmp_path, rows, **extra):
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(dict(schema=ROSTER_SCHEMA, members=rows, **extra)))
    return path


def test_front_door_defaults_are_eight_members_without_changing_forecast_defaults():
    args = build_parser().parse_args(["moving-ensemble", "--case", "c", "--casedir", "d",
                                     "--out", "o", "--hours", "6"])
    assert args.count == 8 and args.seed == 53
    assert args.handler.__name__ == "run_moving_ensemble"


def test_default_roster_pins_source_and_preserves_stochastic_identity(prepared):
    case, stage = prepared
    rows = moving_roster(case, stage)
    assert len(rows) == 8
    assert len({r["stochastic"]["member"] for r in rows}) == 8
    assert all(r["stochastic"]["sppt"] and r["stochastic"]["skeb"] for r in rows)
    assert all(r["weight"] == .125 for r in rows)
    assert {r["physics"] for r in rows} == {"wsm6", "thompson"}
    assert len({r["provenance"]["init_sha256"] for r in rows}) == 1
    assert all(r["provenance"]["cycle"] == "2025-11-08T12" for r in rows)


def test_source_member_and_enabled_forcing_are_separate_provenance(prepared, tmp_path):
    case, stage = prepared
    (stage / "stage.json").write_text(json.dumps(dict(
        source="gefs", source_member=3, cycle="2025-11-08T12")))
    rows = moving_roster(case, stage, count=2, manifest=manifest(tmp_path, [
        dict(id="a", forecast_flags=["--physics", "config_microp_scheme=mp_wsm6"]),
        dict(id="b", forecast_flags=["--physics", "config_microp_scheme=mp_thompson"])],
        stochastic=dict(sppt=False, skeb=False)))
    assert all(r["provenance"]["source_member"] == 3 for r in rows)
    assert all(r["provenance"]["perturbation"] == dict(sppt=False, skeb=False) for r in rows)


def source_case(tmp_path, member=3, source="gefs"):
    body = json.loads((CASES / "tc-wp-2025110812.json").read_text())
    body.update(init_source=source, init_member=member)
    path = tmp_path / "source-case.json"
    path.write_text(json.dumps(body))
    return path


def test_stage_selects_and_caches_the_table_declared_source_member(tmp_path, monkeypatch):
    from woof.hex import init_sources
    case = source_case(tmp_path)
    acquired = []

    def acquire(plans, *args, **kwargs):
        acquired.extend(plans)
        return [dict(file=str(tmp_path / p.filename)) for p in plans]

    met = tmp_path / "met"
    met.write_text("decoded fixture")
    monkeypatch.setattr(init_sources, "acquire", acquire)
    monkeypatch.setattr(init_sources, "resolve_writer", lambda _: "unused")
    monkeypatch.setattr(init_sources, "resolve_vtable", lambda _: "unused")
    monkeypatch.setattr(init_sources, "decode", lambda *args: dict(met=str(met)))
    stage = stage_case(case, tmp_path / "prepared", log=lambda _: None)
    assert acquired and all("gep03" in p.url for p in acquired)
    assert stage["source_member"] == 3
    acquired.clear()
    assert stage_case(case, tmp_path / "prepared", log=lambda _: None) == stage
    assert acquired == []


@pytest.mark.parametrize("lead", [0, 6])
def test_gfs_stage_from_a_forecast_hour_or_without_its_donor_keeps_cold_soil(tmp_path, monkeypatch, lead):
    # A GFS moving case from a later hour (its donor is analysis-only), or one whose archive lacks the donor, stages
    # without it and initializes with --soil-remap off instead of refusing (WOOF 0.4.10 candidate canary, 2026-10-31).
    from woof.hex import init_sources
    body = json.loads((CASES / "tc-wp-2025110812.json").read_text())
    body.update(init_source="gfs", init_lead_hours=lead)
    case = tmp_path / "gfs-case.json"
    case.write_text(json.dumps(body))
    roles = []

    def acquire(plans, *args, **kwargs):
        roles.extend(p.role for p in plans)
        if any(p.role == "cold-soil-donor" for p in plans):
            init_sources._refuse("gfs.t12z.sfcanl.nc: no address has it (HTTP 404).")
        return [dict(role=p.role, file=str(tmp_path / p.filename), sha256="0") for p in plans]

    met = tmp_path / "met"
    met.write_text("decoded fixture")
    monkeypatch.setattr(init_sources, "acquire", acquire)
    monkeypatch.setattr(init_sources, "resolve_writer", lambda _: "unused")
    monkeypatch.setattr(init_sources, "resolve_vtable", lambda _: "unused")
    monkeypatch.setattr(init_sources, "decode", lambda *args: dict(met=str(met)))
    stage = stage_case(case, tmp_path / "prepared", log=lambda _: None)
    assert ("cold-soil-donor" in roles) == (lead == 0)
    flags = init_sources.cold_soil_init_flags(stage.get("products", []), init_sources.resolve_source(
        init_sources.load_table(), "gfs").get("cold_soil"))
    assert flags == ["--soil-remap", "off"]


@pytest.mark.parametrize("previous", [
    dict(source="gefs", source_member=2, cycle="2025-11-08_12:00:00"),
    dict(source="gfs", cycle="2025-11-08_12:00:00"),
    dict(source="gefs", source_member=3, cycle="2025-11-08_06:00:00"),
])
def test_restage_cannot_relabel_prepared_initial_conditions(tmp_path, previous):
    case = source_case(tmp_path)
    folder = tmp_path / "prepared"
    folder.mkdir()
    path = folder / "stage.json"
    original = json.dumps(previous)
    path.write_text(original)
    (folder / "family.json").write_text("[]")
    with pytest.raises(MovingCaseError, match="initial conditions cannot be mislabeled"):
        stage_case(case, folder)
    assert path.read_text() == original


@pytest.mark.parametrize("member", [True, -1, 3.0, "3"])
def test_stage_rejects_noninteger_member_before_fetch(tmp_path, member):
    with pytest.raises(MovingCaseError, match="nonnegative integer"):
        stage_case(source_case(tmp_path, member), tmp_path / "prepared")


@pytest.mark.parametrize("source,member,match", [
    ("gefs", None, "is an ensemble"),
    ("gefs", 31, "outside.*declared members"),
    ("gfs", 3, "declares no members"),
])
def test_source_table_admission_runs_before_cached_stage_reuse(tmp_path, source, member, match):
    from woof.hex.init_sources import InitSourceRefusal
    case = source_case(tmp_path, member, source)
    folder = tmp_path / "prepared"
    folder.mkdir()
    met = folder / "met"
    met.write_text("cached fixture")
    (folder / "stage.json").write_text(json.dumps(dict(
        source=source, source_member=member, cycle="2025-11-08_12:00:00", met=str(met))))
    with pytest.raises(InitSourceRefusal, match=match):
        stage_case(case, folder)


def test_default_roster_admits_shared_member_forcing(prepared):
    from woof.hex.drivers.fullstep_lts_forecast_v841 import require_stochastic_support
    case, stage = prepared
    for row in moving_roster(case, stage):
        require_stochastic_support(row['stochastic'])


def test_weighted_member_selection_is_explicit_and_order_independent(prepared, tmp_path):
    case, stage = prepared
    a = moving_roster(case, stage, manifest=manifest(tmp_path, [
        dict(id="a", weight=16), dict(id="b", weight=6)]), count=2)
    b = moving_roster(case, stage, manifest=manifest(tmp_path, [
        dict(id="b", weight=6), dict(id="a", weight=16)]), count=2)
    assert {r["id"]: r["stochastic"] for r in a} == {r["id"]: r["stochastic"] for r in b}
    assert a[0]["weight"] == pytest.approx(16/22)


@pytest.mark.parametrize("change,match", [
    (dict(id="../escape"), "unsafe"),
    (dict(weight=0), "weight"),
    (dict(weight=float("nan")), "weight"),
    (dict(forecast_flags=["--out=other"]), "controller"),
    (dict(forecast_flags=["--stochastic", "{}"]), "controller"),
    (dict(forecast_flags=["--dt", "1"]), "controller"),
    (dict(forecast_flags=["--receipt", "other.json"]), "controller"),
    (dict(forecast_flags=["--no-fullstep-local-timestep"]), "controller"),
    (dict(forecast_flags=["--hour", "1"]), "invalid forecast"),
    (dict(forecast_flags="--physics x"), "argv"),
    (dict(stochastic=dict(member=1)), "identity"),
])
def test_bad_rosters_fail_before_forecast_or_output_creation(prepared, tmp_path, change, match):
    case, stage = prepared
    with pytest.raises(MovingCaseError, match=match):
        moving_roster(case, stage, manifest=manifest(tmp_path, [dict({"id": "a"}, **change), dict(id="b")]), count=2)


def test_wrong_source_clock_is_rejected(prepared):
    case, stage = prepared
    (stage / "stage.json").write_text(json.dumps(dict(source="gfs", cycle="2025-11-08T06")))
    with pytest.raises(MovingCaseError, match="cycle"):
        moving_roster(case, stage)


def test_dateline_mean_missing_fix_and_weighted_intensity_do_not_hide_members(tmp_path):
    rows = []
    for name, lon, pressure, weight, good in [
        ("a", 179., 980., .6, True), ("b", -179., 1000., .3, True),
        ("c", 0., 0., .1, False),
    ]:
        root = tmp_path / name
        root.mkdir()
        (root / "controller.json").write_text(json.dumps(dict(frames=[
            dict(model_seconds=3600, fix=dict(ok=good, lat=0., lon=lon,
                mslp_min_hpa=pressure, vmax10_ms=None))])))
        rows.append(dict(id=name, weight=weight, out=str(root)))
    product = moving_ensemble_products(rows, tmp_path)
    lead = product["leads"][0]
    assert abs(lead["lon"]) > 179
    assert lead["fixes"] == 2 and lead["members_expected"] == 3
    assert lead["included_weight"] == pytest.approx(.9)
    assert 100 < lead["track_rms_km"] < 110
    assert lead["mslp_min_hpa"]["p50"] == 980
    assert lead["vmax10_ms"]["count"] == 0
    assert lead["vmax10_ms"]["p50"] is None


def test_identical_disabled_perturbation_branches_are_not_counted_twice(prepared, tmp_path):
    case, stage = prepared
    with pytest.raises(MovingCaseError, match="duplicate a forecast"):
        moving_roster(case, stage, manifest=manifest(tmp_path, [dict(id="a"), dict(id="b")],
            stochastic=dict(sppt=False, skeb=False)), count=2)


def test_missing_entire_member_remains_in_expected_count(tmp_path):
    root = tmp_path / "a"
    root.mkdir()
    (root / "controller.json").write_text(json.dumps(dict(frames=[
        dict(model_seconds=3600, fix=dict(ok=True, lat=10., lon=20.,
             mslp_min_hpa=990., vmax10_ms=20.))])))
    product = moving_ensemble_products([dict(id="a", weight=.5, out=str(root)),
        dict(id="b", weight=.5, out=str(tmp_path / "absent"))], tmp_path)
    assert product["leads"][0]["members_expected"] == 2
    assert product["leads"][0]["fixes"] == 1
    assert product["leads"][0]["included_weight"] == .5


def test_large_finite_weights_preserve_positive_normalized_mass(prepared, tmp_path):
    case, stage = prepared
    rows = moving_roster(case, stage, manifest=manifest(tmp_path, [
        dict(id="a", weight=1e308), dict(id="b", weight=1e308)]), count=2)
    assert [r["weight"] for r in rows] == [.5, .5]


@pytest.mark.parametrize("cycle", ["2025-11-08_12:00:00", "2025-11-08T12", "2025110812"])
def test_native_stage_cycle_and_case_clock_compare_as_instants(prepared, cycle):
    case, stage = prepared
    (stage / "stage.json").write_text(json.dumps(dict(source="gfs", cycle=cycle)))
    rows = moving_roster(case, stage)
    assert all(r["provenance"]["cycle"] == cycle for r in rows)


def test_native_track_plume_retains_missing_member_fixes_and_valid_clock(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace as NS
    from woof.hex import moving_members, render_door

    renderer = tmp_path / "rw_wrfbatch"
    renderer.write_bytes(b"test-only native invocation substitute")
    monkeypatch.setitem(sys.modules, "woof", NS(rustwx=NS(renderer_env=lambda: {})))
    monkeypatch.setattr(render_door, "resolve_renderer_exe", lambda: renderer)
    monkeypatch.setattr(render_door, "resolve_basemap_dir", lambda _: tmp_path)
    monkeypatch.setattr(render_door, "renderer_environment", lambda _: {})
    def execute(command, **kwargs):
        Path(command[-1]).write_bytes(b"test-only picture substitute")
        return NS(returncode=0, stdout="RENDERED", stderr="")
    monkeypatch.setattr(moving_members.subprocess, "run", execute)
    product = dict(members=[dict(id="a", weight=.25), dict(id="b", weight=.75)], tracks=[
        dict(id="a", seconds=0, ok=True, lat=10., lon=20., mslp_min_hpa=980., vmax10_ms=25.),
        dict(id="a", seconds=1800, ok=False)])
    moving_members.render_moving_tracks(product, tmp_path, "2025-11-08_12:00:00")
    native = json.loads((tmp_path / "native-tracks.json").read_text())
    assert native["tracks"][0]["fixes"][0] == dict(valid="2025-11-08T12:00:00+00:00",
        lat=10., lon=20., mslp_pa=98000., max_wind10_ms=25.)
    assert native["tracks"][0]["fixes"][1] is None
    assert native["tracks"][1]["fixes"] == [None, None]
    assert native["tracks"][1]["weight"] == .75
    receipt = json.loads((tmp_path / "track-render-receipt.json").read_text())
    assert receipt["command"][1] == "--tc-track-plume"
    assert receipt["wrapper"] == "woof.rustwx"



def test_member_raw_waits_for_a_slow_product_stage(tmp_path):
    """A passed member's grid stays while its detached product stage still draws."""
    import subprocess
    import sys
    from woof.hex.moving_members import retire_member_raw
    run = tmp_path / "member-005" / "run"
    segments = []
    for index in range(2):
        segment = run / f"segment-00{index}"
        (segment / "fc").mkdir(parents=True)
        segments.append(dict(path=str(segment), rc=0))
    (run / "controller.json").write_text(json.dumps(dict(segments=segments)))
    grid = run / "members" / "mn02550e27700.grid.nc"
    grid.parent.mkdir()
    grid.write_bytes(b"0" * 2048)
    first, last = (Path(s["path"]) / "fc" for s in segments)
    (first / "products.log").write_text("PRODUCTS done maps=319 point_pictures=2 failed_views=none\n")
    (last / "products.log").write_text("PRODUCTS view=global window=global\n")
    slow = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                             "products", "--forecast", str(last)])
    try:
        assert retire_member_raw(run, min_bytes=1) is None
        assert grid.is_file()
    finally:
        slow.kill()
        slow.wait()
    (last / "products.log").write_text("PRODUCTS done maps=43 point_pictures=1 failed_views=none\n")
    deleted = retire_member_raw(run, min_bytes=1)
    assert [d["path"] for d in deleted] == [str(grid)] and not grid.exists()


def test_retire_member_raw_is_a_front_door_option():
    args = build_parser().parse_args(["moving-ensemble", "--case", "c", "--casedir", "d",
                                     "--out", "o", "--hours", "6", "--retire-member-raw"])
    assert args.retire_member_raw
