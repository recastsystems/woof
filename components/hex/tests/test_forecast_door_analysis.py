"""``woof hex forecast --analysis`` (woof.hex.analysis_door) on a CPU box: how a bundle becomes the
door's own inputs, every refusal, and what reaches the driver and the receipt."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from woof.hex import analysis_bundle as ab
from woof.hex import analysis_door
from woof.hex import forecast_door as door

from test_analysis_bundle import VALID, write_grid, write_state
from test_forecast_door import (  # noqa: F401  (autouse fixtures: no card, no seam check)
    _proven_architecture, _reference_card, _registry, _satisfied_seam_pin)


def _bundle(tmp_path: Path, *, increment: bool = True) -> tuple[ab.AnalysisBundle, Path, Path]:
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    grid = write_grid(inputs / "x1.grid.nc")
    static = inputs / "x1.static.nc"
    static.write_bytes(b"static bytes")
    analysis = write_state(inputs / "an.nc", seed=1)
    background = write_state(inputs / "bg.nc", seed=2)
    bundle = ab.make_bundle(
        tmp_path / ("bundle-iau" if increment else "bundle-di"), analysis=analysis, grid=grid, static=static,
        mesh_id="x1.40962", static_builder="test", physics_suite="arwen_mpas_column", physics_mode="correct",
        kind="mean", members=36, recipe_id="recipe.v1", recipe_sha256="c" * 64, cycles=5, window=900.0,
        producer="tests", background=background if increment else None,
        iau_window_s=600.0 if increment else None)
    return bundle, grid, static


def _arguments(tmp_path: Path, bundle: ab.AnalysisBundle, grid: Path, static: Path, *extra: str) -> argparse.Namespace:
    checkout = tmp_path / "woof"
    checkout.mkdir(exist_ok=True)
    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    return parser.parse_args([
        "--mesh", "x1.40962", "--grid", str(grid), "--static", str(static),
        "--analysis", str(bundle.root), "--hours", "1.0", "--history-every-minutes", "30",
        "--out", str(tmp_path / "out"), "--gpuwm-checkout", str(checkout), *extra])


def test_a_bundle_without_increment_is_inserted_directly(tmp_path):
    bundle, grid, static = _bundle(tmp_path, increment=False)
    args = _arguments(tmp_path, bundle, grid, static)
    receipt = analysis_door.resolve_analysis(args, registry=_registry())
    assert Path(args.init) == bundle.analysis
    assert args.start_time == VALID
    assert args.init_source.startswith("analysis bundle ")
    assert receipt["start"].startswith("direct insertion")
    assert receipt["run_iau"]["on"] is False
    args.analysis_receipt = receipt
    request = door.resolve_request(args, registry=_registry())
    argv = door.build_driver_argv(request)
    assert argv[argv.index("--init") + 1] == str(bundle.analysis)
    assert "--iau-window-seconds" not in argv
    door_receipt = door.build_receipt(request=request, admission=None, bind_receipt=None, driver_receipt=None,
                                      history=[], driver_argv=argv, seconds=0, status="test")
    assert door_receipt["analysis_bundle"]["manifest_sha256"] == bundle.manifest_sha256


@pytest.mark.parametrize("mode,route", [(None, "hex"), ("correct", "hex"), ("exact", "native")])
def test_a_bundle_with_increment_starts_an_iau_window_from_its_background(tmp_path, mode, route):
    bundle, grid, static = _bundle(tmp_path)
    extra = () if mode is None else ("--physics-mode", mode)
    args = _arguments(tmp_path, bundle, grid, static, *extra)
    receipt = analysis_door.resolve_analysis(args, registry=_registry())
    work = tmp_path / "out.analysis-work"
    assert Path(args.init) == work / "background.nc" and Path(args.init).is_file()
    assert Path(args.iau_analysis) == bundle.analysis
    assert Path(args.iau_increment) == bundle.increment
    assert args.iau_window_seconds == 600.0
    assert receipt["run_iau"]["window_source"] == "manifest iau.window_s"
    args.analysis_receipt = receipt
    request = door.resolve_request(args, registry=_registry())
    assert request.iau_route == route
    argv = door.build_driver_argv(request)
    assert argv[argv.index("--iau-route") + 1] == route
    assert argv[argv.index("--iau-increment") + 1] == str(bundle.increment)
    assert argv[argv.index("--iau-window-seconds") + 1] == "600.0"


def test_iau_off_and_an_explicit_route_and_window_are_knobs(tmp_path):
    bundle, grid, static = _bundle(tmp_path)
    args = _arguments(tmp_path, bundle, grid, static, "--iau", "off")
    analysis_door.resolve_analysis(args, registry=_registry())
    assert Path(args.init) == bundle.analysis and args.iau_analysis is None
    (tmp_path / "k").mkdir()
    bundle2, grid2, static2 = _bundle(tmp_path / "k")
    args = _arguments(tmp_path / "k", bundle2, grid2, static2, "--iau-route", "native", "--iau-window-seconds", "300")
    receipt = analysis_door.resolve_analysis(args, registry=_registry())
    assert args.iau_window_seconds == 300.0 and receipt["run_iau"]["window_source"] == "--iau-window-seconds"
    args.analysis_receipt = receipt
    assert door.resolve_request(args, registry=_registry()).iau_route == "native"


@pytest.mark.parametrize("extra,setup,match", [
    (("--init", "INIT"), None, "both name the initial state"),
    (("--iau", "on"), "no-increment", "no increment.nc"),
    (("--restart-from", "CKPT", "--init", "INIT", "--iau", "on", "--iau-route", "hex"), None,
     "not a restarted run's background"),
    (("--restart-from", "CKPT"), None, "needs --init"),
    (("--remap", "x1.40962"), None, "--remap-grid"),
    (("--iau-increment", "INC", "--iau-window-seconds", "60"), None, "second analysis"),
    ((), "wrong-grid", "the analysis is on mesh x1.40962"),
    ((), "out-exists", "exists"),
    ((), "work-exists", "fresh directory"),
    (("--start-time", "2022-06-13_23:30:00"), None, "valid time"),
])
def test_refusals_name_their_breakage(tmp_path, extra, setup, match):
    bundle, grid, static = _bundle(tmp_path, increment=setup != "no-increment")
    placeholder = tmp_path / "placeholder.nc"
    placeholder.write_bytes(b"x")
    extra = tuple(str(placeholder) if item in ("INIT", "CKPT", "INC", "ROSTER") else item for item in extra)
    if setup == "wrong-grid":
        grid = write_grid(tmp_path / "other.grid.nc", cells=7)
    args = _arguments(tmp_path, bundle, grid, static, *extra)
    if setup == "out-exists":
        (tmp_path / "out").mkdir()
    if setup == "work-exists":
        (tmp_path / "out.analysis-work").mkdir()
    with pytest.raises(analysis_door.AnalysisDoorRefusal, match=match):
        analysis_door.resolve_analysis(args, registry=_registry())
    if setup not in ("work-exists",):
        assert not (tmp_path / "out.analysis-work").exists(), "a refused bundle leaves no derived files"


def test_bundle_flags_without_a_bundle_are_refused(tmp_path):
    from test_forecast_door import _namespace

    args = _namespace(tmp_path, iau="on")
    with pytest.raises(analysis_door.AnalysisDoorRefusal, match="no --analysis"):
        analysis_door.resolve_analysis(args)
    (tmp_path / "two").mkdir()
    args = _namespace(tmp_path / "two", iau_route="hex")
    with pytest.raises(door.ForecastDoorRefusal, match="never used"):
        door.resolve_request(args, registry=_registry())


def test_preflight_checks_the_bundle_and_writes_nothing(tmp_path):
    bundle, grid, static = _bundle(tmp_path)
    args = _arguments(tmp_path, bundle, grid, static, "--preflight")
    receipt = analysis_door.resolve_analysis(args, registry=_registry())
    assert receipt["start"].startswith("preflight")
    assert not (tmp_path / "out.analysis-work").exists()


def test_a_da_cycling_restart_takes_the_analysis_as_its_da_analysis(tmp_path):
    bundle, grid, static = _bundle(tmp_path, increment=False)
    checkpoint = tmp_path / "checkpoint.pkl"
    checkpoint.write_bytes(b"pickle")
    init = write_state(tmp_path / "original-init.nc", seed=9, valid="2022-06-13_23:00:00")
    args = _arguments(tmp_path, bundle, grid, static, "--restart-from", str(checkpoint), "--init", str(init))
    receipt = analysis_door.resolve_analysis(args, registry=_registry())
    assert Path(args.da_analysis) == bundle.analysis and Path(args.init) == init
    assert receipt["checks"]["vertical_sources_checked"] == [str(init)]
    assert receipt["start"].startswith("DA-cycling restart")


def test_raw_increment_runs_the_native_route_and_hex_needs_an_analysis(tmp_path):
    from test_forecast_door import _namespace

    increment = tmp_path / "inc.nc"
    increment.write_bytes(b"x")
    args = _namespace(tmp_path, iau_increment=increment, iau_window_seconds=600.0, physics_mode="exact")
    request = door.resolve_request(args, registry=_registry())
    assert request.iau_route == "native"
    argv = door.build_driver_argv(request)
    assert "--iau-analysis" not in argv and argv[argv.index("--iau-route") + 1] == "native"
    (tmp_path / "two").mkdir()
    args = _namespace(tmp_path / "two", iau_increment=increment, iau_window_seconds=600.0, iau_route="hex")
    with pytest.raises(door.ForecastDoorRefusal, match="needs --iau-analysis"):
        door.resolve_request(args, registry=_registry())


def test_restart_with_iau_admits_the_bundle_increment_on_the_native_route(tmp_path):
    # Retired refusal (named gap W1-G1): checkpoints carry their IAU window, so a DA-cycling restart with
    # --iau on starts from the checkpoint (the background) and admits increment.nc over a new window.
    bundle, grid, static = _bundle(tmp_path)
    placeholder = tmp_path / "placeholder.nc"
    placeholder.write_bytes(b"x")
    args = _arguments(tmp_path, bundle, grid, static, "--restart-from", str(placeholder), "--init",
                      str(bundle.analysis), "--iau", "on")
    receipt = analysis_door.resolve_analysis(args, registry=_registry())
    assert Path(args.iau_increment) == Path(bundle.increment) and args.iau_route == "native"
    assert getattr(args, "da_analysis", None) is None and receipt["run_iau"]["route"] == "native"


def test_iau_runs_on_the_fixed_step_executor_now(tmp_path):
    # Retired refusal: IAU needed --fullstep-lts-mode active; the fixed-step executor
    # carries the same held-tendency hook now.
    from test_forecast_door import _namespace

    analysis = tmp_path / "an.nc"
    analysis.write_bytes(b"x")
    args = _namespace(tmp_path, iau_analysis=analysis, iau_window_seconds=600.0, balance_series=True)
    request = door.resolve_request(args, registry=_registry())
    argv = door.build_driver_argv(request)
    assert "--fullstep-local-timestep" not in argv and "--balance-series" in argv
    assert argv[argv.index("--iau-route") + 1] == "hex"


def _carried(tmp_path: Path, *, row: bool = True) -> ab.AnalysisBundle:
    from test_analysis_bundle import _rows_document

    inputs = tmp_path / "carried-inputs"
    inputs.mkdir(exist_ok=True)
    files = {"grid": write_grid(inputs / "m.grid.nc"), "static": inputs / "m.static.nc",
             "analysis": write_state(inputs / "an.nc", seed=1), "background": write_state(inputs / "bg.nc", seed=2)}
    files["static"].write_bytes(b"static bytes")
    rows = _rows_document(inputs, files, name="carried-mesh") if row else None
    return ab.make_bundle(
        tmp_path / "bundle-carried", analysis=files["analysis"], grid=files["grid"], static=files["static"],
        mesh_id="carried-mesh", static_builder="test", physics_suite="arwen_mpas_column", physics_mode="correct",
        kind="mean", members=36, recipe_id="recipe.v1", recipe_sha256="c" * 64, cycles=5, window=900.0,
        producer="tests", background=files["background"], iau_window_s=600.0, carry_mesh=True, mesh_rows=rows)


def _bare_arguments(tmp_path: Path, bundle: ab.AnalysisBundle, *extra: str) -> argparse.Namespace:
    checkout = tmp_path / "woof"
    checkout.mkdir(exist_ok=True)
    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    return parser.parse_args([
        "--analysis", str(bundle.root), "--hours", "1.0", "--history-every-minutes", "60",
        "--out", str(tmp_path / "out"), "--gpuwm-checkout", str(checkout), *extra])


def test_a_bundle_that_carries_its_mesh_needs_no_mesh_grid_or_static(tmp_path, monkeypatch):
    import os

    from woof.hex.mesh_rows import MESH_ROWS_ENVIRONMENT, load_rows

    monkeypatch.delenv(MESH_ROWS_ENVIRONMENT, raising=False)
    bundle = _carried(tmp_path)
    args = _bare_arguments(tmp_path, bundle, "--iau", "on", "--physics-mode", "correct")
    seen = {}

    def loader():
        seen["rows"] = [row.name for row in load_rows()]
        return dict(_registry(), **{"carried-mesh": _registry()["x1.40962"]})

    receipt = analysis_door.resolve_analysis(args, registry_loader=loader)
    assert seen["rows"] == ["carried-mesh"], "the registry is read after the carried row is bound"
    assert args.mesh == "carried-mesh"
    assert Path(args.grid) == bundle.carried_grid and Path(args.static) == bundle.carried_static
    assert receipt["mesh_binding"]["bound"] == ["--grid", "--static", "--mesh"]
    assert os.environ[MESH_ROWS_ENVIRONMENT].split(os.pathsep)[0] == receipt["mesh_binding"]["mesh_rows"]


def test_carried_files_without_a_row_need_mesh(tmp_path, monkeypatch):
    from woof.hex.mesh_rows import MESH_ROWS_ENVIRONMENT

    monkeypatch.delenv(MESH_ROWS_ENVIRONMENT, raising=False)
    bundle = _carried(tmp_path, row=False)
    args = _bare_arguments(tmp_path, bundle)
    with pytest.raises(analysis_door.AnalysisDoorRefusal, match="no mesh-row.json"):
        analysis_door.resolve_analysis(args, registry=_registry())
    args = _bare_arguments(tmp_path, bundle, "--mesh", "x1.40962")
    analysis_door.resolve_analysis(args, registry=_registry())
    assert args.mesh == "x1.40962" and Path(args.grid) == bundle.carried_grid


def test_a_bundle_that_carries_no_mesh_needs_grid_and_static(tmp_path):
    bundle, _grid, _static = _bundle(tmp_path)
    args = _bare_arguments(tmp_path, bundle, "--mesh", "x1.40962")
    with pytest.raises(analysis_door.AnalysisDoorRefusal, match="--grid must name the bundle's mesh file"):
        analysis_door.resolve_analysis(args, registry=_registry())


def test_the_native_route_names_the_increments_mpas_iau_does_not_admit(tmp_path, capsys):
    bundle, grid, static = _bundle(tmp_path)
    args = _arguments(tmp_path, bundle, grid, static, "--physics-mode", "exact")
    receipt = analysis_door.resolve_analysis(args, registry=_registry())
    assert receipt["run_iau"]["native_route_not_admitted"] == ["w_amb"]
    assert "w_amb start at the background" in capsys.readouterr().err


def test_a_carried_row_already_registered_with_the_same_bytes_binds_once(tmp_path, monkeypatch):
    import os

    from woof.hex.mesh_rows import MESH_ROWS_ENVIRONMENT, load_rows

    monkeypatch.delenv(MESH_ROWS_ENVIRONMENT, raising=False)
    bundle = _carried(tmp_path)
    first = _bare_arguments(tmp_path, bundle)
    analysis_door.resolve_analysis(first, registry=_registry())
    # the same process, a second forecast from the same mesh: the row is registered already
    (tmp_path / "two").mkdir()
    second = _bare_arguments(tmp_path / "two", bundle)
    receipt = analysis_door.resolve_analysis(second, registry=_registry())
    assert receipt["mesh_binding"]["existing_registration"] is True
    assert [row.name for row in load_rows()] == ["carried-mesh"], "registered once"
    assert len(os.environ[MESH_ROWS_ENVIRONMENT].split(os.pathsep)) == 1


def test_a_registered_row_of_the_same_name_with_other_bytes_is_refused(tmp_path, monkeypatch):
    from woof.hex.mesh_rows import MESH_ROWS_ENVIRONMENT
    from test_analysis_bundle import _rows_document

    other = tmp_path / "other"
    other.mkdir()
    files = {"grid": write_grid(other / "g.nc", cells=4), "static": other / "s.nc"}
    files["static"].write_bytes(b"other static bytes")
    rows = _rows_document(other, files, name="carried-mesh")
    monkeypatch.setenv(MESH_ROWS_ENVIRONMENT, str(rows))
    bundle = _carried(tmp_path)
    with pytest.raises(analysis_door.AnalysisDoorRefusal, match="already registers mesh 'carried-mesh' with other bytes"):
        analysis_door.resolve_analysis(_bare_arguments(tmp_path, bundle), registry=_registry())


def test_the_producers_forecast_configuration_applies_where_the_run_names_none(tmp_path, capsys):
    inputs = tmp_path / "fc-inputs"
    inputs.mkdir()
    grid = write_grid(inputs / "g.nc")
    static = inputs / "s.nc"
    static.write_bytes(b"static bytes")
    bundle = ab.make_bundle(
        tmp_path / "bundle-fc", analysis=write_state(inputs / "an.nc", seed=1), grid=grid, static=static,
        mesh_id="x1.40962", static_builder="test", physics_suite="arwen_mpas_column", physics_mode="correct",
        kind="mean", members=36, recipe_id="recipe.v1", recipe_sha256="c" * 64, cycles=5, window=900.0,
        producer="tests", forecast_args=["--dt", "18.75", "--fullstep-local-timestep", "--fullstep-lts-mode",
                                         "active", "--fullstep-lts-max-rate", "8", "--physics-mode", "exact"])
    args = _arguments(tmp_path, bundle, grid, static, "--physics-mode", "correct")
    receipt = analysis_door.resolve_analysis(args, registry=_registry())
    assert args.dt == "18.75" and args.fullstep_local_timestep is True
    assert args.fullstep_lts_mode == "active" and args.fullstep_lts_max_rate == 8
    assert args.physics_mode == "correct", "a flag the run names is the run's"
    assert receipt["forecast_defaults"]["run_named"] == ["physics_mode"]
    assert "the producer's forecast configuration" in capsys.readouterr().err


def test_a_preflight_binds_the_carried_mesh_and_writes_nothing_beside_out(tmp_path, monkeypatch):
    from woof.hex.mesh_rows import MESH_ROWS_ENVIRONMENT, load_rows

    monkeypatch.delenv(MESH_ROWS_ENVIRONMENT, raising=False)
    bundle = _carried(tmp_path)
    args = _bare_arguments(tmp_path, bundle, "--preflight")
    receipt = analysis_door.resolve_analysis(args, registry=_registry())
    assert args.mesh == "carried-mesh" and [row.name for row in load_rows()] == ["carried-mesh"]
    assert receipt["start"].startswith("preflight")
    assert not (tmp_path / "out.analysis-work").exists() and not (tmp_path / "out").exists()
