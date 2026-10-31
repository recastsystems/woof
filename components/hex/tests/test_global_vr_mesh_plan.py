"""Regressions for a global VR request being priced and bound as global."""
from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from woof.hex import card_fit, device_admission, mesh_plan_door, mesh_point, mesh_rows
from woof.hex.cli import build_parser, main
from woof.hex.errors import MpasPortError


def test_pro6000_aliases_use_the_actual_part():
    for key in ("96gb", "96gib", "rtx-pro-6000", "96gib-188sm"):
        card = mesh_point.resolve_card(key)
        assert card["admission_card"] == "96gib-188sm"
        assert card["nameplate_mib"] == 97887
        assert card["generator_card"] is None  # the old affine generator table has no measured row


def test_prediction_uses_global_admission_and_keeps_the_model_provenance():
    fit = card_fit.predict(250000, card=mesh_point.resolve_card("96gb"), levels=55,
                           dt_seconds=90, hours=12, observations=[])
    model = device_admission.model_for_card(device_admission.KNOWN_CARDS["96gib-188sm"], "global")
    assert fit["required_free_mib"] == round(model.required_bytes(250000) / 1048576, 3)
    assert fit["predicted_peak_mib"] == round(model.predict_bytes(250000) / 1048576, 3)
    assert fit["model"]["configuration"] == "global"
    assert fit["model"]["provenance"]
    # No whole run of the default (MPAS) route given: no rate, never the ArWen route's table row.
    assert fit["runtime"]["seconds"] is None
    assert fit["runtime"]["basis"].startswith("no measured rate for this route/card")
    once = card_fit.predict(250000, card=mesh_point.resolve_card("96gb"), levels=55,
                            dt_seconds=90, hours=12)
    twice = card_fit.predict(250000, card=mesh_point.resolve_card("96gb"), levels=55,
                             dt_seconds=45, hours=12)
    assert "EXTRAPOLATED" in once["runtime"]["basis"]
    assert twice["runtime"]["stepping_seconds"] == pytest.approx(2 * once["runtime"]["stepping_seconds"], abs=.01)
    assert twice["runtime"]["startup_seconds"] == once["runtime"]["startup_seconds"]


def test_runtime_never_borrows_a_different_card_measurement():
    fit = card_fit.predict(100000, card=mesh_point.resolve_card("10gb"), levels=55,
                           dt_seconds=90, hours=12)
    assert fit["runtime"]["seconds"] is None


def test_runtime_is_the_measured_rate_of_this_route_card_and_card_count():
    """hex 0.4.10 review: the r700 zoom globe on 2 RTX PRO 6000s (MPAS route, convection_permitting +
    hrrr-like-aerosol, 1 h) was priced 626.66 s from a one-card ArWen-route x1.163842 row against 173.399 s measured
    (+261 %, receipt r700-2c-global-vr-receipt.json), and 2 and 4 cards
    printed the same 7,517 s for 12 h."""
    pro = mesh_point.resolve_card("rtx-pro-6000")
    aerosol = dict(card=pro, levels=55, dt_seconds=18, suite="convection_permitting", profile="hrrr-like-aerosol")
    fit = card_fit.predict(286530, hours=1, cards=2, **aerosol)
    assert fit["runtime"]["reference"]["route"] == "mpas" and fit["runtime"]["reference"]["cards"] == 2
    error = card_fit.comparison(fit, {"wall_seconds": 173.399, "peak_card_vram_mib": 30813.0})["runtime_error_percent"]
    assert abs(error) <= 20
    by_cards = {cards: card_fit.predict(286428, hours=12, cards=cards, **aerosol)["runtime"]["seconds"]
                for cards in (1, 2, 4)}
    assert by_cards[1] > by_cards[2] > by_cards[4]
    # RTX 5090 (VRFS, one process per card): startup measured apart from the steady hour, so 12 h is not 6 x 2 h.
    rtx5090 = mesh_point.resolve_card("rtx-5090")
    two = card_fit.predict(286530, card=rtx5090, levels=55, dt_seconds=18, hours=2, cards=2)["runtime"]
    twelve = card_fit.predict(286530, card=rtx5090, levels=55, dt_seconds=18, hours=12, cards=2)["runtime"]
    assert two["seconds"] == pytest.approx(288.0) and twelve["seconds"] == pytest.approx(78.0 + 12 * 105.0)


def test_runtime_never_borrows_another_route_or_card_count():
    # The 5070 Ti has an ArWen-route x1.40962 rate and no MPAS-route run: the MPAS default gets no rate.
    card = mesh_point.resolve_card("rtx-5070-ti")
    mpas = card_fit.predict(40962, card=card, levels=55, dt_seconds=600, hours=24)
    assert mpas["physics"]["route"] == "mpas" and mpas["runtime"]["seconds"] is None
    assert "no measured rate for this route/card" in mpas["runtime"]["basis"]
    arwen = card_fit.predict(40962, card=card, levels=55, dt_seconds=300, hours=24, route="arwen",
                             suite="arwen_mpas_column")
    assert arwen["runtime"]["seconds"] == pytest.approx(5.5 * 24)
    # No 3-card run on a 5090: not the 2- or 4-card rate.
    three = card_fit.predict(286530, card=mesh_point.resolve_card("rtx-5090"), levels=55, dt_seconds=18, hours=12,
                             cards=3)
    assert three["runtime"]["seconds"] is None


@pytest.mark.parametrize("key,value", [("cells", float("nan")), ("hours", 0), ("dt_seconds", -1),
                                       ("budget_mib", float("inf"))])
def test_invalid_cost_inputs_fail_before_compute(key, value):
    kwargs = dict(cells=100000, card=mesh_point.resolve_card("96gb"), levels=55,
                  dt_seconds=90, hours=12)
    kwargs[key] = value
    with pytest.raises(MpasPortError, match="finite"):
        card_fit.predict(**kwargs)


def test_process_memory_missing_is_not_reported_as_zero():
    fit = card_fit.predict(100000, card=mesh_point.resolve_card("96gb"), levels=55,
                           dt_seconds=90, hours=12)
    result = card_fit.comparison(fit, {"wall_seconds": 10, "peak_process_vram_mib": None})
    assert result["memory_error_percent"] is None
    assert result["measured_peak_process_vram_mib"] is None


def args(tmp_path, *extra):
    return build_parser().parse_args(["mesh-plan", "--global-mesh", "--point", "40,-98",
        "--background-km", "60", "--fine-dx-m", "15000", "--radius-km", "1800",
        "--card", "96gb", "--out-dir", str(tmp_path), *extra])


def test_global_plan_prices_parent_not_regional_cull(tmp_path, monkeypatch):
    from woof.hex.swath import sizing
    monkeypatch.setattr(mesh_plan_door, "_load", lambda p: {})
    monkeypatch.setattr("woof.hex.engines.resolve", lambda *a: Path("/engine"))
    monkeypatch.setattr(sizing, "dry_run", lambda *a, **k: {"predicted_cells": 250000})
    request = args(tmp_path, "--out", str(tmp_path / "plan.json"))
    assert mesh_plan_door.run_global_mesh(request) == 0
    receipt = json.loads((tmp_path / "plan.json").read_text())
    assert receipt["prediction_before_generation"]["cells"] == 250000
    assert receipt["prediction_before_generation"]["model"]["configuration"] == "global"
    assert not (tmp_path / "mesh.grid.nc").exists()


def test_nonfitting_global_mesh_refused_before_generation(tmp_path, monkeypatch):
    from woof.hex.swath import sizing
    monkeypatch.setattr("woof.hex.engines.resolve", lambda *a: Path("/engine"))
    monkeypatch.setattr(sizing, "dry_run", lambda *a, **k: {"predicted_cells": 2621442})
    monkeypatch.setattr(card_fit, "run_stage", lambda *a, **k: pytest.fail("must not start"))
    with pytest.raises(MpasPortError, match="whole global mesh requires"):
        mesh_plan_door.run_global_mesh(args(tmp_path, "--generate"))
    assert not (tmp_path / "global-vr-receipt.json").exists()


def test_run_without_global_mode_is_refused():
    with pytest.raises(MpasPortError, match="requires --global-mesh"):
        mesh_plan_door.run_mesh_plan(build_parser().parse_args(["mesh-plan", "--run"]))


def test_stage_records_failure_and_preserves_log(tmp_path, monkeypatch):
    monkeypatch.setattr(card_fit, "_gpu_memory_by_card", lambda pids: None)
    result = card_fit.run_stage([sys.executable, "-c", "print('specific failure'); raise SystemExit(7)"],
                                name="failure", directory=tmp_path)
    assert result["returncode"] == 7
    assert "specific failure" in Path(result["log"]).read_text()
    assert result["peak_process_vram_mib"] is None
    assert result["peak_card_vram_mib"] is None


def test_a_multicard_stage_is_compared_per_card(tmp_path, monkeypatch):
    """THE BREAKAGE (V8X6000 2026-10-29): a 2-card run's receipt compared its 28,722 MiB per-card prediction with
    both cards' sum, 60,722 MiB, and said the plan was 52.7 % under; the largest card held 30,813."""
    monkeypatch.setattr(card_fit, "_gpu_memory_by_card", lambda pids: {"GPU-a": 29924.0, "GPU-b": 30798.0})
    result = card_fit.run_stage([sys.executable, "-c", "pass"], name="two-cards", directory=tmp_path)
    assert result["peak_process_vram_mib"] == 60722.0
    assert result["peak_card_vram_mib"] == 30798.0
    assert result["peak_vram_mib_by_card"] == {"GPU-a": 29924.0, "GPU-b": 30798.0}
    fit = card_fit.predict(286530, card=mesh_point.resolve_card("96gb"), levels=55, dt_seconds=18, hours=1,
                           suite="convection_permitting", cards=2)
    compared = card_fit.comparison(fit, result)
    assert compared["measured_peak_card_vram_mib"] == 30798.0
    assert compared["memory_error_percent"] == round(100 * (fit["predicted_peak_mib"] / 30798.0 - 1), 3)
    assert result["wall_seconds"] > 0
    assert Path(result["samples"]).is_file()


@pytest.mark.parametrize("route", ["mpas", "arwen", "native", "exact"])
def test_whole_binding_keeps_generated_row_and_source_arguments(tmp_path, monkeypatch, route):
    from types import SimpleNamespace
    from woof.hex import init_sources
    from woof.hex.swath import sizing
    # MPAS physics is the hex default (0.4.6); the ArWen route is named.
    named = {"mpas": [], "arwen": ["--forecast-arg=--physics-suite=arwen_mpas_column"],
             "native": ["--forecast-arg=--physics-suite=convection_permitting"],
             "exact": ["--forecast-arg=--physics-mode=exact"]}[route]
    request = args(tmp_path, "--generate", "--run", "--source", "gfs", "--cycle", "2023-03-31T12", *named)
    binary = tmp_path / "engine"
    binary.write_bytes(b"binary")
    monkeypatch.setattr("woof.hex.engines.resolve", lambda *a: binary)
    monkeypatch.setattr(init_sources, "resolve_writer", lambda *a: binary)
    monkeypatch.setattr(init_sources, "resolve_vtable", lambda *a: binary)
    monkeypatch.setattr(sizing, "dry_run", lambda *a, **k: {"predicted_cells": 216000})
    monkeypatch.setattr(mesh_point, "resolve_geog", lambda *a: tmp_path)
    monkeypatch.setattr(mesh_point, "static_coordinate_errors", lambda *a: [])
    monkeypatch.setattr(mesh_point, "admit_pair", lambda *a, **k:
                        {"n_cells": 215000, "timestep_choice": {"dt_seconds": 90}, "regional": False})
    row = SimpleNamespace(name="generated-row", n_cells=215000, as_dict=lambda: {"name": "generated-row"})
    def describe(**kwargs):
        assert kwargs["core_radius_km"] == 1800
        return row
    monkeypatch.setattr(mesh_rows, "describe_resolution_spec", describe)
    monkeypatch.setattr(mesh_rows, "write_rows", lambda path, rows: path.write_text("runtime rows"))
    calls = []

    def run(argv, *, name, directory, env):
        calls.append((name, argv, dict(env)))
        if name == "mesh-template":
            Path(argv[argv.index("--out") + 1]).write_bytes(b"template grid")
        if name == "init-source":
            out = tmp_path / "init"
            out.mkdir()
            (out / "init-source-receipt.json").write_text(json.dumps({"source": "gfs", "cycle": "2023-03-31_12:00:00"}))
        return {"returncode": 0, "wall_seconds": 10, "peak_process_vram_mib": 16000}

    monkeypatch.setattr(card_fit, "run_stage", run)
    monkeypatch.setattr(card_fit, "start_stage", fake_background(calls))
    monkeypatch.setattr("woof.hex.moving_mesh.rotate_grid", fake_rotation)
    assert mesh_plan_door.run_global_mesh(request) == 0
    assert [call[0] for call in calls] == ["init-source-decode", "mesh-template", "static", "init-source-decode.finish",
                                           "init-source", "forecast", "products"]
    forecast = calls[5]
    assert forecast[1][forecast[1].index("--mesh") + 1] == "generated-row"
    if route == "arwen":
        assert forecast[1][forecast[1].index("--dt") + 1] == "90"
        assert "--physics-suite=arwen_mpas_column" in forecast[1]
    else:  # the forecast door's own step for the suite, as on the global door
        assert "--dt" not in forecast[1] and "--physics-suite=convection_permitting" in forecast[1]
    if route == "mpas":
        assert "--physics-mode=correct" in forecast[1]
        assert "--physics-profile=hrrr-like-aerosol" in forecast[1]
        init_argv = calls[4][1]
        assert init_argv[init_argv.index("--physics-profile") + 1] == "hrrr-like-aerosol"
    if route in ("native", "exact"):
        assert not any("--physics-profile" in flag for flag in forecast[1])
        assert "--physics-profile" not in calls[4][1]
    if route == "exact":
        assert "--physics-mode=exact" in forecast[1]
    assert str(tmp_path / "mesh-rows.json") in forecast[2]["WOOF_HEX_MESH_ROWS"]
    receipt = json.loads((tmp_path / "global-vr-receipt.json").read_text())
    assert receipt["physics"]["suite"] == ("convection_permitting" if route != "arwen" else "arwen_mpas_column")
    if route == "mpas":
        assert receipt["physics"]["profile"] == "hrrr-like-aerosol"
    # The mode the forecast door runs: a named suite with no mode runs correct, not exact (the shapes table was
    # read as exact-mode numbers from this label, 2026-10-29).
    assert receipt["physics"]["mode"] == ("exact" if route == "exact" else "correct")
    assert receipt["status"] == "complete"
    assert receipt["prediction_on_generated_mesh"]["cells"] == 215000
    assert receipt["source_binding"]["source"] == "gfs"
    assert receipt["predicted_vs_measured"]["measured_peak_process_vram_mib"] == 16000


def fake_background(calls, returncode=0):
    """card_fit.start_stage stand-in: records the start and the join in CALLS."""
    def start(argv, *, name, directory, env):
        calls.append((name, [str(a) for a in argv], dict(env)))

        class Handle:
            started_utc = "2026-10-27T00:00:00+00:00"

            def __init__(self):
                self.argv = [str(a) for a in argv]
                self.cancelled = False

            def finish(self):
                calls.append((f"{name}.finish", self.argv, {}))
                return {"stage": name, "returncode": returncode, "wall_seconds": 13.4, "log": "decode.log",
                        "background": True}

            def cancel(self):
                calls.append((f"{name}.cancel", self.argv, {}))
        return Handle()
    return start


def fake_rotation(source, output, *, start, end, file_id=None):
    Path(output).write_bytes(Path(source).read_bytes() + repr((end, file_id)).encode())
    return {"sha256": "rotated", "wall_seconds": 0.5, "start": list(start), "end": list(end)}


def test_zoom_globe_rotates_the_cap_family_template_and_decodes_beside_the_mesh(tmp_path, monkeypatch):
    """U-H6 (production run_fde6921eaf18b793): a zoom globe generated a fresh mesh for its point (16.6 s) and
    fetched and decoded its source only after the static. The door now asks for the point's cap family about
    the template centre (a prep-store key with no point in it), rotates it to the point, and starts the source
    fetch and decode before the mesh; the init is built --from-decoded."""
    from types import SimpleNamespace
    from woof.hex import init_sources
    from woof.hex.moving_members import TEMPLATE_CENTER
    from woof.hex.swath import sizing
    request = args(tmp_path, "--generate", "--run", "--source", "ifs", "--cycle", "2026-10-26T00")
    binary = tmp_path / "engine"
    binary.write_bytes(b"binary")
    monkeypatch.setattr("woof.hex.engines.resolve", lambda *a: binary)
    monkeypatch.setattr(init_sources, "resolve_writer", lambda *a: binary)
    monkeypatch.setattr(init_sources, "resolve_vtable", lambda *a: binary)
    monkeypatch.setattr(sizing, "dry_run", lambda *a, **k: {"predicted_cells": 216000})
    monkeypatch.setattr(mesh_point, "resolve_geog", lambda *a: tmp_path)
    monkeypatch.setattr(mesh_point, "static_coordinate_errors", lambda *a: [])
    monkeypatch.setattr(mesh_point, "admit_pair", lambda *a, **k:
                        {"n_cells": 215000, "timestep_choice": {"dt_seconds": 90}, "regional": False})
    row = SimpleNamespace(name="generated-row", n_cells=215000, as_dict=lambda: {"name": "generated-row"})
    monkeypatch.setattr(mesh_rows, "describe_resolution_spec", lambda **k: row)
    monkeypatch.setattr(mesh_rows, "write_rows", lambda path, rows: path.write_text("runtime rows"))
    calls, rotations = [], []

    def run(argv, *, name, directory, env):
        calls.append((name, [str(a) for a in argv], dict(env)))
        if name == "mesh-template":
            Path(argv[argv.index("--out") + 1]).write_bytes(b"template grid")
        if name == "init-source":
            (tmp_path / "init").mkdir()
            (tmp_path / "init" / "init-source-receipt.json").write_text(json.dumps({"source": "ifs"}))
        return {"returncode": 0, "wall_seconds": 10, "peak_process_vram_mib": 9000}

    def rotate(source, output, *, start, end, file_id=None):
        rotations.append((start, end, file_id))
        return fake_rotation(source, output, start=start, end=end, file_id=file_id)

    monkeypatch.setattr(card_fit, "run_stage", run)
    monkeypatch.setattr(card_fit, "start_stage", fake_background(calls))
    monkeypatch.setattr("woof.hex.moving_mesh.rotate_grid", rotate)
    assert mesh_plan_door.run_global_mesh(request) == 0
    names = [call[0] for call in calls]
    # The source fetch and decode start first and are joined only after the static.
    assert names[:3] == ["init-source-decode", "mesh-template", "static"]
    assert names.index("init-source-decode.finish") > names.index("static")
    decode = calls[0][1]
    assert decode[decode.index("--stop-after") + 1] == "decode" and "--grid" not in decode
    init = calls[names.index("init-source")][1]
    assert "--from-decoded" in init and init[init.index("--static") + 1] == str(tmp_path / "mesh.static.nc")
    assert "rw_mpas_mesh" not in names and "mesh" not in names  # no fresh generation for the point
    template = json.loads(Path(calls[1][1][calls[1][1].index("--spec") + 1]).read_text())
    assert template["name"] == mesh_plan_door.ROTATION_TEMPLATE_NAME
    assert {tuple(r["shape"]["center_deg"]) for r in template["regions"]} == {TEMPLATE_CENTER}
    spec = json.loads((tmp_path / "resolution-spec.json").read_text())
    assert [r["radius_km"] for r in (x["shape"] for x in template["regions"])] == \
        [r["radius_km"] for r in (x["shape"] for x in spec["regions"])]
    assert rotations[0][0] == TEMPLATE_CENTER and rotations[0][1] == (40.0, -98.0) and rotations[0][2]
    receipt = json.loads((tmp_path / "global-vr-receipt.json").read_text())
    mesh = next(stage for stage in receipt["stages"] if stage.get("name") == "mesh")
    assert mesh["source"] == "rotated-template" and mesh["end_deg"] == [40.0, -98.0]
    assert receipt["status"] == "complete"
    rotation = json.loads((tmp_path / "mesh-generator.json").read_text())
    assert rotation["schema"] == "gpuwm-hex.zoom-globe-rotation/v1"
    assert not (tmp_path / "template" / "rotation-template.grid.nc").exists()


def test_rotation_template_key_is_the_same_for_every_point():
    """Two zoom globes of one family at different points ask the prep store for the same template."""
    a = mesh_point.ladder_spec((40.0, -97.0), fine_dx_m=15000, radius_km=2000, background_km=120)
    b = mesh_point.ladder_spec((-33.9, 151.2), fine_dx_m=15000, radius_km=2000, background_km=120)
    assert a != b
    assert mesh_plan_door.rotation_template_spec(a) == mesh_plan_door.rotation_template_spec(b)
    assert mesh_plan_door.concentric_cap_center(a) == (40.0, -97.0)
    two = dict(a, regions=[*a["regions"], {"shape": {"kind": "cap", "center_deg": [10.0, 10.0], "radius_km": 500.0},
                                            "spacing_km": 30.0, "transition_km": 60.0}])
    assert mesh_plan_door.concentric_cap_center(two) is None          # several centres: generated
    assert mesh_plan_door.concentric_cap_center({"background_km": 60, "regions": []}) is None  # uniform: generated


def test_template_only_builds_the_family_template_for_a_bake(tmp_path, monkeypatch):
    """The bake stages a zoom family's template (no static, no run) so boxes rotate it instead of generating."""
    from woof.hex.moving_members import TEMPLATE_CENTER
    from woof.hex.swath import sizing
    binary = tmp_path / "engine"
    binary.write_bytes(b"binary")
    monkeypatch.setattr("woof.hex.engines.resolve", lambda *a: binary)
    monkeypatch.setattr(sizing, "dry_run", lambda *a, **k: {"predicted_cells": 216000})
    monkeypatch.setattr(mesh_point, "resolve_geog", lambda *a: tmp_path)
    calls = []

    def run(argv, *, name, directory, env):
        calls.append((name, [str(a) for a in argv]))
        Path(argv[argv.index("--out") + 1]).write_bytes(b"template grid")
        return {"returncode": 0, "wall_seconds": 9}

    monkeypatch.setattr(card_fit, "run_stage", run)
    assert mesh_plan_door.run_global_mesh(args(tmp_path, "--generate", "--template-only")) == 0
    assert [c[0] for c in calls] == ["mesh-template"]
    receipt = json.loads((tmp_path / "global-vr-receipt.json").read_text())
    assert receipt["status"] == "template"
    assert {tuple(r["shape"]["center_deg"]) for r in receipt["template"]["spec"]["regions"]} == {TEMPLATE_CENTER}
    with pytest.raises(MpasPortError, match="no --run"):
        mesh_plan_door.run_global_mesh(args(tmp_path / "b", "--generate", "--template-only", "--run",
                                            "--source", "ifs", "--cycle", "2026-10-26T00"))


def test_a_failed_door_stops_the_decode_beside_it(tmp_path, monkeypatch):
    from woof.hex import init_sources
    from woof.hex.swath import sizing
    request = args(tmp_path, "--generate", "--run", "--source", "ifs", "--cycle", "2026-10-26T00")
    binary = tmp_path / "engine"
    binary.write_bytes(b"binary")
    monkeypatch.setattr("woof.hex.engines.resolve", lambda *a: binary)
    monkeypatch.setattr(init_sources, "resolve_writer", lambda *a: binary)
    monkeypatch.setattr(init_sources, "resolve_vtable", lambda *a: binary)
    monkeypatch.setattr(sizing, "dry_run", lambda *a, **k: {"predicted_cells": 216000})
    monkeypatch.setattr(mesh_point, "resolve_geog", lambda *a: tmp_path)
    calls = []
    monkeypatch.setattr(card_fit, "run_stage", lambda argv, *, name, directory, env:
                        calls.append((name, argv, env)) or {"returncode": 3 if name == "mesh-template" else 0,
                                                            "log": "x"})
    monkeypatch.setattr(card_fit, "start_stage", fake_background(calls))
    with pytest.raises(MpasPortError, match="mesh-template exited 3"):
        mesh_plan_door.run_global_mesh(request)
    assert [c[0] for c in calls] == ["init-source-decode", "mesh-template", "init-source-decode.cancel"]


def test_background_stage_reports_its_own_wall_and_cpu(tmp_path):
    handle = card_fit.start_stage([sys.executable, "-c", "import time; sum(range(10**6)); time.sleep(0.3)"],
                                  name="beside", directory=tmp_path)
    import time
    time.sleep(1.0)  # the door is busy with the foreground stage
    row = handle.finish()
    assert row["returncode"] == 0 and row["background"]
    assert 0.3 <= row["wall_seconds"] < 0.95  # its own exit, not the join
    assert row["cpu_seconds"] > 0
    failing = card_fit.start_stage([sys.executable, "-c", "raise SystemExit(5)"], name="bad", directory=tmp_path)
    assert failing.finish()["returncode"] == 5
    slow = card_fit.start_stage([sys.executable, "-c", "import time; time.sleep(60)"], name="slow", directory=tmp_path)
    slow.cancel()
    assert slow.finish()["returncode"] != 0


def test_init_source_from_decoded_refuses_another_requests_decode(tmp_path):
    from woof.hex import init_sources
    met = tmp_path / "MET:2026-10-26_00"
    met.write_bytes(b"met")
    base = {"schema": init_sources.RECEIPT_SCHEMA, "source": "ifs", "member": None,
            "source_cycle": "2026-10-26_00:00:00", "lead_hours": 0, "valid_time": "2026-10-26_00:00:00",
            "table": {"sha256": "t"}}
    prior = dict(base, decode={"met": str(met), "met_sha256": init_sources._sha256_file(met)},
                 stage_seconds={"acquire": 9.0, "decode": 4.4})
    (tmp_path / "init-source-receipt.json").write_text(json.dumps(prior))
    assert init_sources._decoded_receipt(tmp_path, dict(base))["decode"]["met"] == str(met)
    with pytest.raises(MpasPortError, match="lead_hours"):
        init_sources._decoded_receipt(tmp_path, dict(base, lead_hours=6))
    met.write_bytes(b"changed")
    with pytest.raises(MpasPortError, match="SHA-256"):
        init_sources._decoded_receipt(tmp_path, dict(base))
    with pytest.raises(MpasPortError, match="does not exist"):
        init_sources._decoded_receipt(tmp_path / "elsewhere", dict(base))


def test_observed_whole_run_floor_prevents_repeating_a_false_fit():
    anchor = {"status": "complete", "card": "96gib-188sm", "levels": 55,
              "cells": 216189, "peak_process_vram_mib": 25180, "receipt_sha256": "evidence"}
    arwen = {"suite": "arwen_mpas_column"}  # a row that names no route is an ArWen-route run (before 0.4.6)
    fit = card_fit.predict(216189, card=mesh_point.resolve_card("96gb"), levels=55,
                           dt_seconds=90, hours=18, budget_mib=24000, observations=[anchor], **arwen)
    margin = 1 + card_fit.WHOLE_RUN_MARGIN
    assert fit["base_predicted_peak_mib"] < 16000
    assert fit["predicted_peak_mib"] == pytest.approx(25180 * margin, abs=.001)
    assert fit["required_free_mib"] > 25180
    assert not fit["fits"]
    assert fit["whole_run_floor"]["observation"]["receipt_sha256"] == "evidence"
    larger = card_fit.predict(432378, card=mesh_point.resolve_card("96gb"), levels=55,
                              dt_seconds=90, hours=18, observations=[anchor], **arwen)
    assert larger["predicted_peak_mib"] >= 50360 * margin - .01
    # Below its one run the floor falls with device_admission's own slope; it is
    # never held at the run's peak (128,942 cells priced at the 216k run's 25,272).
    smaller = card_fit.predict(108094, card=mesh_point.resolve_card("96gb"), levels=55,
                               dt_seconds=90, hours=18, observations=[anchor], **arwen)
    assert smaller["predicted_peak_mib"] < 25180
    other = card_fit.predict(216189, card=mesh_point.resolve_card("32gb"), levels=55,
                             dt_seconds=90, hours=18, observations=[anchor], **arwen)
    assert other["whole_run_floor"] is None
    unfinished = dict(anchor, status="running")
    assert card_fit.predict(216189, card=mesh_point.resolve_card("96gb"), levels=55,
                            dt_seconds=90, hours=18, observations=[unfinished], **arwen)["whole_run_floor"] is None
    # The same ArWen run never prices the MPAS route (the hex default): measured 2026-10-28, see below.
    assert card_fit.predict(216189, card=mesh_point.resolve_card("96gb"), levels=55,
                            dt_seconds=90, hours=18, observations=[anchor])["whole_run_floor"] is None


def test_shipped_whole_run_observation_rejects_the_measured_false_fit():
    fit = card_fit.predict(216189, card=mesh_point.resolve_card("96gb"), levels=55,
                           dt_seconds=90, hours=18, budget_mib=21000, suite="arwen_mpas_column")
    assert fit["base_predicted_peak_mib"] < 19816
    assert fit["predicted_peak_mib"] >= 19816
    assert fit["required_free_mib"] > fit["predicted_peak_mib"]
    assert not fit["fits"]
    assert fit["whole_run_floor"]["observation"]["cells"] == 216187
    assert len(fit["whole_run_floor"]["observation"]["receipt_sha256"]) == 64


#: Every ArWen-route whole run on disk (card, cells, sampled forecast peak MiB); hex 0.4.3 / 0.4.4 bare runs.
MEASURED_WHOLE_RUNS = [
    ("96gb", 52618, 7472.0),     # 120 -> 30 km, 24 h, RTX PRO 6000
    ("96gb", 216187, 19816.0),   # 60 -> 15 km, 15 h, RTX PRO 6000
    ("96gb", 581895, 44450.0),   # 80 -> 5 km, 1,800 km core, 3 h, RTX PRO 6000
    ("32gb", 128942, 13036.0),   # 80 -> 5 km, 400 km core, 3 h, RTX 5090
    ("96gb", 128942, 13036.0),   # the same run as the site prices it (96 GB card); was 25,272
]


@pytest.mark.parametrize("card,cells,peak", MEASURED_WHOLE_RUNS)
def test_shipped_curve_lands_10_to_15_percent_over_every_measured_run(card, cells, peak):
    fit = card_fit.predict(cells, card=mesh_point.resolve_card(card), levels=55, dt_seconds=30, hours=3,
                           suite="arwen_mpas_column")
    over = fit["predicted_peak_mib"] / peak - 1
    assert 0.10 - 1e-9 <= over <= 0.15, (card, cells, fit["predicted_peak_mib"], over)


@pytest.mark.parametrize("suite", ["arwen_mpas_column", "convection_permitting"])
@pytest.mark.parametrize("key", ["96gb", "32gb"])
def test_shipped_curve_rises_with_cells(suite, key):
    card = mesh_point.resolve_card(key)
    for cards in (1, 2, 8):
        peaks = [card_fit.predict(cells, card=card, levels=55, dt_seconds=30, hours=3, suite=suite,
                                  cards=cards)["predicted_peak_mib"]
                 for cells in range(20000, 1400001, 20000)]
        assert peaks == sorted(peaks), (suite, key, cards)


#: The runs measured on 8 x RTX 5090 (box VRFS, woof hex 0.4.9, 2026-10-28) and the RRFS hrrrv5 runs on one RTX 5090
#: (rrfs/mynn-speed c2efafcf): (cells, cards, levels, suite, per-card MiB, outcome).  "ran" is the sampled per-card
#: peak of a run that finished; "oom" is a run that stopped out of memory at its first physics step, and the MiB is a
#: floor on its need (what the card held plus the request that failed; x1.163842: more than the card).
MEASURED_2026_10_28 = [
    (119691, 1, 55, "convection_permitting", 19187.0, "ran"),   # r300 zoom globe, strict, 1 h
    (119691, 8, 55, "convection_permitting", 6901.0, "ran"),    # r300 on 8 cards, strict
    (286530, 1, 55, "convection_permitting", 33580.0, "oom"),   # r700 on one card
    (286530, 2, 55, "convection_permitting", 26111.0, "ran"),
    (286530, 4, 55, "convection_permitting", 17449.0, "ran"),
    (286530, 8, 55, "convection_permitting", 11325.0, "ran"),
    (417810, 2, 55, "convection_permitting", 32151.0, "oom"),   # r900 on 2 cards
    (10242, 1, 59, "hrrrv5", 6770.0, "ran"),                    # RRFS x1.10242
    (40962, 1, 59, "hrrrv5", 12024.0, "ran"),                   # RRFS x1.40962
    (163842, 1, 59, "hrrrv5", 32607.0, "oom"),                  # RRFS x1.163842
]


@pytest.mark.parametrize("cells,cards,levels,suite,measured,outcome", MEASURED_2026_10_28)
def test_plan_says_fits_exactly_where_the_card_held_the_run(cells, cards, levels, suite, measured, outcome):
    """THE BREAKAGE (VRFS, 2026-10-28): the plan priced the MPAS-physics zoom globes on ArWen-route runs (r300
    13,821 MiB against a measured 19,187; r700 31,865 against a measured need of at least 33,580) and had no card
    count, so priced on two cards of memory the r900 globe fit and ran both cards out of memory."""
    fit = card_fit.predict(cells, card=mesh_point.resolve_card("rtx-5090"), levels=levels, dt_seconds=18, hours=1,
                           suite=suite, cards=cards)
    assert fit["physics"] == {"route": "mpas", "suite": suite}
    assert fit["cards"] == cards and fit["columns_per_card"] == -(-cells // cards)
    assert fit["predicted_peak_mib"] >= measured * (1 + card_fit.WHOLE_RUN_MARGIN) - 0.01
    if outcome == "ran":
        assert fit["fits"]
        assert fit["predicted_peak_mib"] <= measured * 1.15
    else:
        assert not fit["fits"]


def test_fewest_cards_follow_the_measured_runs():
    card = mesh_point.resolve_card("rtx-5090")
    r700 = card_fit.by_card_count(286530, card=card, levels=55, dt_seconds=18, hours=1)
    assert r700["fewest_cards_that_fit"] == 2
    assert [row["fits"] for row in r700["by_cards"]][:4] == [False, True, True, True]
    r900 = card_fit.by_card_count(417810, card=card, levels=55, dt_seconds=18, hours=1)
    assert r900["by_cards"][1]["fits"] is False  # ran out of memory on 2 cards
    assert r900["fewest_cards_that_fit"] == 3
    r300 = card_fit.by_card_count(119691, card=card, levels=55, dt_seconds=18, hours=1)
    assert r300["fewest_cards_that_fit"] == 1


def test_a_suite_with_no_multicard_run_is_priced_as_a_rank_not_as_one_cards_loop():
    card = mesh_point.resolve_card("rtx-5090")
    two = card_fit.predict(163842, card=card, levels=59, dt_seconds=300, hours=1, suite="hrrrv5", cards=2)
    one = card_fit.predict(81921, card=card, levels=59, dt_seconds=300, hours=1, suite="hrrrv5")
    excess = two["whole_run_floor"]["rank_excess"]
    assert excess["ratio"] > 1.2  # the 2-card r700 rank against the one-card curve at its columns
    assert two["predicted_peak_mib"] == pytest.approx(one["predicted_peak_mib"] * excess["ratio"], rel=1e-4)
    # hrrrv5 is never priced from the lighter convection_permitting rows while it has runs of its own.
    assert two["whole_run_floor"]["physics"]["suite"] == "hrrrv5"


def test_the_arwen_route_keeps_its_own_runs():
    card = mesh_point.resolve_card("rtx-5090")
    arwen = card_fit.predict(128942, card=card, levels=55, dt_seconds=30, hours=3, suite="arwen_mpas_column")
    assert arwen["physics"]["route"] == "arwen"
    assert arwen["predicted_peak_mib"] == pytest.approx(13036.0 * (1 + card_fit.WHOLE_RUN_MARGIN), abs=.01)
    bare = card_fit.predict(128942, card=card, levels=55, dt_seconds=30, hours=3)
    assert bare["physics"] == {"route": "mpas", "suite": "convection_permitting"}  # the hex default
    assert bare["predicted_peak_mib"] > arwen["predicted_peak_mib"]


def zoom_plan(tmp_path, monkeypatch, cells, *extra):
    from woof.hex.swath import sizing
    monkeypatch.setattr("woof.hex.engines.resolve", lambda *a: Path("/engine"))
    monkeypatch.setattr(sizing, "dry_run", lambda *a, **k: {"predicted_cells": cells})
    monkeypatch.setattr(card_fit, "run_stage", lambda *a, **k: pytest.fail("must not start"))
    out = tmp_path / "plan.json"
    request = build_parser().parse_args(["mesh-plan", "--global-mesh", "--point", "37.5,-91",
        "--background-km", "96", "--fine-dx-m", "3000", "--radius-km", "700", "--card", "rtx-5090",
        "--out-dir", str(tmp_path / "run"), "--out", str(out), *extra])
    return request, out


def test_door_prices_the_zoom_globe_per_card_on_the_physics_it_runs(tmp_path, monkeypatch, capsys):
    request, out = zoom_plan(tmp_path, monkeypatch, 286530)
    assert mesh_plan_door.run_global_mesh(request) == 0
    plan = json.loads(out.read_text())["prediction_before_generation"]
    assert plan["physics"] == {"route": "mpas", "suite": "convection_permitting"}
    assert plan["cards"] == 1 and not plan["fits"] and plan["fewest_cards_that_fit"] == 2
    assert "fewest cards that fit: 2" in capsys.readouterr().out
    request, out = zoom_plan(tmp_path, monkeypatch, 286530, "--cards", "2")
    assert mesh_plan_door.run_global_mesh(request) == 0
    assert json.loads(out.read_text())["prediction_before_generation"]["fits"]
    request, out = zoom_plan(tmp_path, monkeypatch, 286530, "--forecast-arg=--samehost-devices=0,1,2,3")
    assert mesh_plan_door.run_global_mesh(request) == 0
    assert json.loads(out.read_text())["prediction_before_generation"]["cards"] == 4
    request, _ = zoom_plan(tmp_path, monkeypatch, 286530, "--generate")
    with pytest.raises(MpasPortError, match="pass --cards 2"):
        mesh_plan_door.run_global_mesh(request)
    request, _ = zoom_plan(tmp_path, monkeypatch, 286530, "--cards", "2", "--forecast-arg=--samehost-devices=0,1,2")
    with pytest.raises(MpasPortError, match="disagree"):
        mesh_plan_door.run_global_mesh(request)


@pytest.mark.parametrize("route", ["mpas", "arwen"])
def test_the_run_goes_on_the_cards_the_plan_priced(tmp_path, monkeypatch, route):
    """--cards 2 with --run: both predictions are per card on 2 cards and the forecast gets --samehost-devices 0,1."""
    plain = args
    monkeypatch.setattr(sys.modules[__name__], "args", lambda path, *extra: plain(path, *extra, "--cards", "2"))
    test_whole_binding_keeps_generated_row_and_source_arguments(tmp_path, monkeypatch, route)
    receipt = json.loads((tmp_path / "global-vr-receipt.json").read_text())
    assert "--samehost-devices=0,1" in receipt["started_stages"]["forecast"]["argv"]
    assert receipt["prediction_before_generation"]["cards"] == 2
    assert receipt["prediction_on_generated_mesh"]["cards"] == 2


#: hex 0.4.10's regional GFS cold-start default, convection_permitting + hrrr-like-aerosol (physics-default
#: recipe_rules), measured on the r700 96 -> 3 km zoom globe (286,530 cells, 55 levels) on RTX PRO 6000s, box V8X6000
#: 2026-10-29, the HB9 tip ed2dd96b, one process per card: (cards, the largest card's peak MiB).
MEASURED_AEROSOL_2026_10_29 = [(1, 44989.0), (2, 30813.0), (4, 19929.0)]
AEROSOL = {"route": "mpas", "suite": "convection_permitting", "profile": "hrrr-like-aerosol"}
GFS_ZOOM_GLOBE = {"geometry": "zoom-globe", "source": "gfs", "start": "cold"}


@pytest.mark.parametrize("cards,measured", MEASURED_AEROSOL_2026_10_29)
def test_the_aerosol_default_is_priced_from_its_own_runs(cards, measured):
    """THE BREAKAGE (hex 0.4.10 review): the plan priced the aerosol default as plain convection_permitting (2 cards:
    28,715 MiB per card against a measured 30,813), so "fits" could come back for a shape that runs out of memory."""
    fit = card_fit.predict(286530, card=mesh_point.resolve_card("rtx-pro-6000"), levels=55, dt_seconds=18, hours=1,
                           suite="convection_permitting", profile="hrrr-like-aerosol", cards=cards)
    assert fit["physics"] == AEROSOL
    assert measured * (1 + card_fit.WHOLE_RUN_MARGIN) - 0.01 <= fit["predicted_peak_mib"] <= measured * 1.15
    assert fit["fits"]


def test_the_bare_default_is_priced_on_the_recipe_its_context_picks():
    card = mesh_point.resolve_card("rtx-pro-6000")
    regional = card_fit.predict(286530, card=card, levels=55, dt_seconds=18, hours=1, cards=2, context=GFS_ZOOM_GLOBE)
    named = card_fit.predict(286530, card=card, levels=55, dt_seconds=18, hours=1, cards=2,
                             suite="convection_permitting", profile="hrrr-like-aerosol")
    assert regional["physics"] == AEROSOL and regional["predicted_peak_mib"] == named["predicted_peak_mib"]
    # No recipe row for a global run or a warm cycle: the plain suite, as the forecast takes it.
    for context in (None, {"geometry": "global", "source": "gfs", "start": "cold"}, dict(GFS_ZOOM_GLOBE, start="warm")):
        plain = card_fit.predict(286530, card=card, levels=55, dt_seconds=18, hours=1, cards=2, context=context)
        assert plain["physics"] == {"route": "mpas", "suite": "convection_permitting"}


@pytest.mark.parametrize("key", ["rtx-5090", "rtx-pro-6000"])
def test_a_profile_is_never_priced_under_its_plain_suite(key):
    card = mesh_point.resolve_card(key)
    for cards in (1, 2, 3, 4, 8):
        for cells in (40962, 119691, 200000, 286530, 417810, 655362):
            plain = card_fit.predict(cells, card=card, levels=55, dt_seconds=18, hours=1, cards=cards,
                                     suite="convection_permitting")
            aerosol = card_fit.predict(cells, card=card, levels=55, dt_seconds=18, hours=1, cards=cards,
                                       suite="convection_permitting", profile="hrrr-like-aerosol")
            assert aerosol["predicted_peak_mib"] >= plain["predicted_peak_mib"], (key, cards, cells)
            assert aerosol["physics"] == AEROSOL


def test_a_profile_with_no_run_in_a_regime_is_priced_over_its_plain_suite_by_its_measured_excess():
    rows = [row for row in card_fit.shipped_observations()
            if not (row.get("profile") == "hrrr-like-aerosol" and int(row.get("cards", 1)) > 1)]
    card = mesh_point.resolve_card("rtx-pro-6000")
    fit = card_fit.predict(286530, card=card, levels=55, dt_seconds=18, hours=1, cards=4, observations=rows,
                           suite="convection_permitting", profile="hrrr-like-aerosol")
    # Its one-card curve times the measured rank excess, never under the plain suite's 4-card runs.
    plain = card_fit.predict(286530, card=card, levels=55, dt_seconds=18, hours=1, cards=4, observations=rows,
                             suite="convection_permitting")
    assert fit["physics"] == AEROSOL and fit["predicted_peak_mib"] >= plain["predicted_peak_mib"]
    never = [row for row in rows if row.get("profile") is None]
    unmeasured = card_fit.predict(286530, card=card, levels=55, dt_seconds=18, hours=1, cards=2, observations=never,
                                  suite="convection_permitting", profile="hrrr-like-aerosol")
    assert "has no whole run" in unmeasured["whole_run_floor"]["basis"]


def test_door_prices_the_gfs_zoom_globe_on_the_aerosol_profile_it_runs(tmp_path, monkeypatch, capsys):
    request, out = zoom_plan(tmp_path, monkeypatch, 286530, "--cards", "2", "--source", "gfs")
    assert mesh_plan_door.run_global_mesh(request) == 0
    plan = json.loads(out.read_text())["prediction_before_generation"]
    assert plan["physics"] == AEROSOL
    # zoom_plan prices 2 x RTX 5090: the run held to a 5090's 32,607 MiB per card (27,998 MiB), not the 96 GB card's
    # 30,813 MiB, which refused this shape on 2 x 5090 (V8X6000 2026-10-29).
    assert plan["predicted_peak_mib"] == pytest.approx(27998.0 * (1 + card_fit.WHOLE_RUN_MARGIN), abs=.01)
    assert plan["fits"]
    assert "convection_permitting + hrrr-like-aerosol (mpas route) on 2 cards" in capsys.readouterr().out
    # The same globe with no source (a plan alone) or a named suite keeps the plain suite's price.
    request, out = zoom_plan(tmp_path, monkeypatch, 286530, "--cards", "2",
                             "--forecast-arg=--physics-suite=convection_permitting")
    assert mesh_plan_door.run_global_mesh(request) == 0
    assert json.loads(out.read_text())["prediction_before_generation"]["physics"] == {
        "route": "mpas", "suite": "convection_permitting"}


def test_a_plan_takes_the_runs_measured_with_its_own_card_memory():
    """THE BREAKAGE (hex 0.4.10 review, V8X6000 2026-10-29): the 2-card aerosol row was a 96 GB card's peak (30,813
    MiB: idle pool sets kept while the card had room, free-sized tiles), so 2 x RTX 5090 was priced 33,894 MiB per card
    and refused, a shape the same run held to a 5090's 32,607 MiB completes.  At each size a plan takes the runs
    measured with the card memory nearest its own card's from above, else the most below."""
    base = {"route": "mpas", "suite": "convection_permitting", "profile": "hrrr-like-aerosol", "levels": 55,
            "dt_seconds": 18, "status": "complete", "cells": 286530, "cards": 2, "hours": 1, "numerics": "fast"}
    roomy = dict(base, id="roomy", card="96gib-188sm", peak_process_vram_mib=30000.0)
    tight = dict(base, id="tight", card="96gib-188sm", card_capacity_mib=32607, peak_process_vram_mib=26000.0)
    assert card_fit.row_capacity_mib(roomy) == 97887.0 and card_fit.row_capacity_mib(tight) == 32607.0
    for key, expected in (("rtx-5090", 26000.0), ("rtx-pro-6000", 30000.0), ("rtx-5070-ti", 26000.0)):
        fit = card_fit.predict(286530, card=mesh_point.resolve_card(key), levels=55, dt_seconds=18, hours=1, cards=2,
                               observations=[roomy, tight], suite="convection_permitting", profile="hrrr-like-aerosol")
        assert fit["whole_run_floor"]["curve_mib"] == expected, key
    # Out-of-memory floors hold on every card.
    stopped = dict(base, id="stopped", card="32gib-170sm", status="out_of_memory", at_least_mib=29000.0,
                   peak_process_vram_mib=28000.0)
    fit = card_fit.predict(286530, card=mesh_point.resolve_card("rtx-pro-6000"), levels=55, dt_seconds=18, hours=1,
                           cards=2, observations=[tight, stopped], suite="convection_permitting",
                           profile="hrrr-like-aerosol")
    assert fit["whole_run_floor"]["curve_mib"] == 29000.0


def test_two_5090s_take_the_r700_aerosol_default_the_run_held_to_a_5090_completed():
    """THE BREAKAGE (hex 0.4.10 review): 2 x RTX 5090 was refused for the 0.4.10 regional default r700 zoom globe
    ('fewest 3') from a 96 GB card's peak; held to 32,607 MiB per card the run completed at 27,998 MiB per card."""
    fit = card_fit.predict(286530, card=mesh_point.resolve_card("rtx-5090"), levels=55, dt_seconds=18, hours=1,
                           cards=2, suite="convection_permitting", profile="hrrr-like-aerosol")
    assert fit["fits"] and fit["whole_run_floor"]["curve_mib"] == 27998.0
    assert 32607.0 in fit["whole_run_floor"]["capacities_mib"]  # (the 4-card size has only a 96 GB run)
    assert card_fit.by_card_count(286530, card=mesh_point.resolve_card("rtx-5090"), levels=55, dt_seconds=18,
                                  hours=1, suite="convection_permitting",
                                  profile="hrrr-like-aerosol")["fewest_cards_that_fit"] == 2
    # The 96 GB card keeps its own (roomier) runs.
    roomy = card_fit.predict(286530, card=mesh_point.resolve_card("rtx-pro-6000"), levels=55, dt_seconds=18, hours=1,
                             cards=2, suite="convection_permitting", profile="hrrr-like-aerosol")
    assert roomy["whole_run_floor"]["curve_mib"] == 30813.0
