"""No default route runs upstream code that cannot be sold; selecting it is an opt-in that names its terms.

THE BREAKAGES THESE PREVENT
---------------------------
1. A default run executes code that cannot be sold (owner ruling 2026-10-14: the package stays Apache-2.0
   and no default routes into code that cannot be sold or is limited).  The MPAS suites,
   their exact mode, the RRFS fork's suites and the WOOF route must resolve without one gated selection.
2. A run selects Noah-MP v5.0 (UCAR terms: no sale) and nothing says so: the door must refuse it unless the run
   names the terms, and say so loudly and in the receipt when it does.
2b. The owner ruling of 2026-10-14 treats UGWP and the RRFS fork's NOAA GSL physics as US Government code.  A gate
   left on them would refuse the fork-exact RRFS stage (hrrrv5) and UGWP by default (the stale UGWP hold did
   exactly that, and sent hrrrv5's drag to the native library); a recipe that still names a retired opt-in
   (noaa-gsl-unlicensed, ugwp-held) must run, not be refused as unknown.
3. The public distribution, which withholds the Noah-MP v5.0 port, silently swaps in another land surface or dies
   on an import: the door must refuse a hex-cuda selection of a withheld port by naming the licence.
4. An RRFS or MPAS unit lands a hex-cuda port of a gated upstream and nobody adds it to tools/licence_holds.json,
   so the public export ships it.  Every gated selection's hex-cuda module that exists in the tree must be held.

CPU only: the door runs in preflight with the Rust pool reader and the stream reader stood in.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.hex import forecast_door as door
from woof.hex import licence_gate
from woof.hex.mpasphys import forecast_v842
from woof.hex.mpasphys.driver_v842 import resolve_namelist
from woof.hex.mpasphys.rrfs import driver_rrfs
from woof.hex.physics_suites import load_table as suite_table

from test_forecast_door import (  # noqa: F401  (autouse fixtures)
    _namespace,
    _proven_architecture,
    _reference_card,
    _registry,
    _satisfied_seam_pin,
)

ROOT = Path(__file__).resolve().parents[1]
NOAHMP_CLAUSE = ("You shall not sell, license or transfer for a fee the Software, or any work that in any "
                 "manner contains the Software.")


def _config(suite: str, *, rrfs: bool = False, physics: str = "", backends: str = ""):
    text = f"&physics\nconfig_physics_suite='{suite}'\n{physics}/\n"
    if backends:
        text += f"&hex_mpasphys\n{backends}/\n"
    return (driver_rrfs.resolve_namelist if rrfs else resolve_namelist)(text)


def _arguments(*opt_in: str) -> SimpleNamespace:
    return SimpleNamespace(licence_opt_in=list(opt_in) or None)


def test_rrfs_nonstationary_drag_is_not_gated():
    config = _config("hrrrv5", rrfs=True, physics="config_ngw_scheme=.true.\n")
    assert config.selectors["ngw"] == "on"
    assert licence_gate.hits(config) == []
    assert licence_gate.admit(config, _arguments(), stream=io.StringIO(), environ={}) is config


# ---------------------------------------------------------------- 1. defaults
@pytest.mark.parametrize("suite", ["mesoscale_reference", "convection_permitting", "none"])
def test_no_mpas_suite_default_selects_gated_code(suite):
    assert licence_gate.hits(_config(suite)) == []


@pytest.mark.parametrize("suite", ["hrrrv5", "rrfs_mesoscale_reference", "rrfs_convection_permitting", "rrfs_none"])
def test_the_rrfs_forks_suites_select_no_gated_code(suite):
    assert licence_gate.hits(_config(suite, rrfs=True)) == []


def test_a_run_without_gated_code_is_untouched_and_silent():
    config = _config("convection_permitting")
    stream = io.StringIO()
    assert licence_gate.admit(config, _arguments(), stream=stream, environ={}) is config
    assert stream.getvalue() == ""
    assert "licence_opt_in" not in config.receipt()


def test_the_arwen_default_route_never_reaches_the_gate():
    config, text = forecast_v842.native_route(SimpleNamespace(physics_namelist=None, physics_suite=None,
                                                              physics=None))
    assert config is None and text == ""


# ---------------------------------------------------------------- 2. Noah-MP v5.0
def test_noahmp_on_an_mpas_suite_is_refused_by_its_terms_without_an_opt_in():
    config = _config("convection_permitting", physics="config_lsm_scheme='sf_noahmp'\n")
    with pytest.raises(licence_gate.LicenceRefusal) as refused:
        licence_gate.admit(config, _arguments(), environ={})
    text = str(refused.value)
    assert "land_surface=sf_noahmp" in text and "UCAR Noah-MP terms" in text and NOAHMP_CLAUSE in text
    assert "Noah-MP: WRF-lineage v4.x (permissive) runs on the WOOF suite; not MPAS-bitwise; the MPAS v5.0 " \
           "port is withheld for licence" in text
    assert "--licence-opt-in ucar-noahmp" in text and "config_lsm_scheme='sf_noah'" in text


def test_the_native_noahmp_opt_in_is_loud_and_recorded():
    config = _config("mesoscale_reference", physics="config_lsm_scheme='sf_noahmp'\n")
    stream = io.StringIO()
    admitted = licence_gate.admit(config, _arguments("ucar-noahmp"), stream=stream, environ={})
    line = stream.getvalue()
    assert line.startswith("LICENCE OPT-IN ucar-noahmp (--licence-opt-in): land_surface=sf_noahmp")
    assert "your own MPAS build" in line and "NOT Apache-2.0" in line and NOAHMP_CLAUSE in line
    record = admitted.receipt()["licence_opt_in"]
    assert record == [{"slot": "land_surface", "scheme": "sf_noahmp", "backend": "native-cpu",
                       "component": "noahmp-v5-mpas", "terms": "ucar-noahmp", "named_by": "--licence-opt-in",
                       "upstream": record[0]["upstream"]}]
    # The admitted configuration is the same run: same selections, backends and options.
    assert admitted.selectors == config.selectors and admitted.backends == config.backends
    assert {k: v for k, v in admitted.receipt().items() if k != "licence_opt_in"} == config.receipt()


def test_the_environment_opt_in_is_recorded_as_such():
    config = _config("mesoscale_reference", physics="config_lsm_scheme='sf_noahmp'\n")
    admitted = licence_gate.admit(config, _arguments(), stream=io.StringIO(),
                                  environ={licence_gate.ENVIRONMENT: "ucar-noahmp"})
    assert admitted.receipt()["licence_opt_in"][0]["named_by"] == licence_gate.ENVIRONMENT


def test_a_withheld_hex_cuda_port_is_refused_by_its_licence(monkeypatch):
    config = _config("mesoscale_reference", physics="config_lsm_scheme='sf_noahmp'\n",
                     backends="config_lsm_backend='hex-cuda'\n")
    monkeypatch.setattr(licence_gate, "port_present", lambda module: False)
    with pytest.raises(licence_gate.LicenceRefusal) as refused:
        licence_gate.admit(config, _arguments("ucar-noahmp"), environ={})
    text = str(refused.value)
    assert "woof.hex.mpasphys.noahmp_v842" in text and "hold ucar-noahmp-v5" in text
    assert "UCAR Noah-MP terms" in text and "native-cpu" in text


def test_the_hex_cuda_port_runs_by_opt_in_where_the_build_carries_it():
    if not licence_gate.port_present("noahmp_v842"):
        pytest.skip("this distribution withholds the Noah-MP v5.0 port (tools/licence_holds.json ucar-noahmp-v5)")
    config = _config("mesoscale_reference", physics="config_lsm_scheme='sf_noahmp'\n",
                     backends="config_lsm_backend='hex-cuda'\n")
    stream = io.StringIO()
    admitted = licence_gate.admit(config, _arguments("ucar-noahmp"), stream=stream, environ={})
    assert "the hex-cuda port in this build" in stream.getvalue()
    assert admitted.receipt()["licence_opt_in"][0]["backend"] == "hex-cuda"


def test_an_unknown_opt_in_is_refused_by_name():
    with pytest.raises(licence_gate.LicenceRefusal, match="unknown opt-in ucar-noahmp-v4"):
        licence_gate.admit(_config("mesoscale_reference"), _arguments("ucar-noahmp-v4"), environ={})


# ---------------------------------------------------------------- 2b. UGWP and the RRFS stage are not gated
@pytest.mark.parametrize("physics", ["config_gwdo_scheme='bl_ugwp_gwdo'\n",
                                    "config_gwdo_scheme='bl_ugwp_gwdo'\nconfig_ngw_scheme=.true.\n"])
def test_ugwp_on_an_mpas_suite_runs_with_no_opt_in_and_no_line(physics):
    config = _config("mesoscale_reference", physics=physics)
    stream = io.StringIO()
    assert licence_gate.hits(config) == []
    assert licence_gate.admit(config, _arguments(), stream=stream, environ={}) is config
    assert stream.getvalue() == "" and "licence_opt_in" not in config.receipt()


def test_the_fork_exact_rrfs_stage_runs_with_no_opt_in():
    config = _config("hrrrv5", rrfs=True)
    assert {config.selectors[slot] for slot in ("pbl", "surface_layer", "land_surface", "gwdo")} == {
        "bl_mynnedmf", "sf_mynnsfclay", "sf_ruc", "bl_ugwp_gwdo"}
    stream = io.StringIO()
    assert licence_gate.admit(config, _arguments(), stream=stream, environ={}) is config
    assert stream.getvalue() == ""


def test_no_gate_row_names_ugwp_or_the_noaa_gsl_physics():
    table = licence_gate.load_table()
    assert set(table["terms"]) == {"ucar-noahmp"}
    assert {row["component"] for row in table["selections"]} == {"noahmp-v5-mpas", "noahmp-v5-rrfs"}
    assert {"ugwp", "noaa-gsl-rrfs-physics"} <= set(table["not_gated"])
    assert {row["scheme"] for row in table["selections"]} == {"sf_noahmp"}
    holds = {hold["id"] for hold in json.loads((ROOT / "tools" / "licence_holds.json").read_text())["holds"]}
    assert holds == {"ucar-noahmp-v5"}


@pytest.mark.parametrize("name", ["noaa-gsl-unlicensed", "ugwp-held"])
def test_a_retired_opt_in_runs_the_same_run_and_says_so(name):
    for arguments, environ in ((_arguments(name), {}),
                               (_arguments(), {licence_gate.ENVIRONMENT: name})):
        config = _config("hrrrv5", rrfs=True)
        stream = io.StringIO()
        assert licence_gate.admit(config, arguments, stream=stream, environ=environ) is config
        assert stream.getvalue().startswith(f"LICENCE: opt-in {name} is retired and gates nothing")
    # with a live opt-in beside it, the live one still admits Noah-MP and is recorded alone
    config = _config("mesoscale_reference", physics="config_lsm_scheme='sf_noahmp'\n")
    admitted = licence_gate.admit(config, _arguments(f"{name},ucar-noahmp"), stream=io.StringIO(),
                                  environ={})
    assert [row["terms"] for row in admitted.receipt()["licence_opt_in"]] == ["ucar-noahmp"]


def test_noahmp_on_the_rrfs_stage_stays_gated_and_names_its_hold(monkeypatch):
    config = _config("hrrrv5", rrfs=True, physics="config_lsm_scheme='sf_noahmp'\n")
    with pytest.raises(licence_gate.LicenceRefusal, match="--licence-opt-in ucar-noahmp"):
        licence_gate.admit(config, _arguments(), environ={})
    config = _config("hrrrv5", rrfs=True, physics="config_lsm_scheme='sf_noahmp'\n",
                     backends="config_lsm_backend='hex-cuda'\n")
    monkeypatch.setattr(licence_gate, "port_present", lambda module: False)
    with pytest.raises(licence_gate.LicenceRefusal, match="hold ucar-noahmp-v5"):
        licence_gate.admit(config, _arguments("ucar-noahmp"), environ={})


# ---------------------------------------------------------------- 4. holds cover every port of gated code
def test_every_hex_cuda_port_of_gated_code_in_this_tree_is_licence_held():
    holds = json.loads((ROOT / "tools" / "licence_holds.json").read_text(encoding="utf-8"))["holds"]
    held = [path for hold in holds for path in hold["paths"]]
    table = licence_gate.load_table()
    unheld = []
    for row in table["selections"]:
        module = licence_gate.cuda_module(row["row"], row["slot"], row["scheme"])
        if module is None:
            continue
        relative = "src/hexcore/mpasphys/" + module.replace(".", "/") + ".py"
        if (ROOT / relative).is_file() and not any(
                relative == path or (path.endswith("/") and relative.startswith(path)) for path in held):
            unheld.append(f"{row['row']} {row['slot']}={row['scheme']}: {relative}")
    assert unheld == [], ("hex-cuda ports of restricted or unlicensed upstream code that the public export would "
                          f"ship; add each to tools/licence_holds.json: {unheld}")


def test_every_component_with_a_hold_names_a_hold_that_exists():
    holds = {hold["id"] for hold in json.loads((ROOT / "tools" / "licence_holds.json").read_text())["holds"]}
    for name, component in licence_gate.load_table()["components"].items():
        if component.get("hex_port_hold"):
            assert component["hex_port_hold"] in holds, f"{name} names an unknown hold"


def test_the_gate_table_names_only_selections_the_suite_table_has():
    tables = suite_table()
    for row in licence_gate.load_table()["selections"]:
        assert row["scheme"] in tables[row["row"]]["slots"][row["slot"]]["schemes"]


def test_an_opted_in_native_noahmp_run_stages_its_table(tmp_path):
    """Measured 2026-10-14: the opted-in native run stopped reading NoahmpTable.TBL; every other run is unchanged."""
    from woof.hex.mpasphys.runtime_v842 import native_table_roots
    (tmp_path / "src/core_atmosphere/physics/physics_wrf/files").mkdir(parents=True)
    parameters = tmp_path / "src/core_atmosphere/physics/physics_noahmp/parameters"
    parameters.mkdir(parents=True)
    noahmp = _config("mesoscale_reference", physics="config_lsm_scheme='sf_noahmp'\n")
    noah = _config("mesoscale_reference")
    assert native_table_roots(tmp_path, noahmp)[-1] == parameters
    assert parameters not in native_table_roots(tmp_path, noah)
    assert native_table_roots(tmp_path, noahmp, [parameters]).count(parameters) == 1


# ---------------------------------------------------------------- 2/3 at the real door
def _door_arguments(tmp_path: Path, *extra: str) -> argparse.Namespace:
    # The streams deck is parsed as XML before the run (rrfs/match deck_stream_inputs), so its placeholder is an empty deck.
    for name in ("library.so", "reader", "source", "init.nc", "streams.atmosphere"):
        (tmp_path / name).write_text("<streams/>" if name == "streams.atmosphere" else "placeholder")
    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    return parser.parse_args([
        "--mesh", "x1.2562", "--init", str(tmp_path / "init.nc"), "--hours", "1",
        "--streams", str(tmp_path / "streams.atmosphere"), "--out", str(tmp_path / "out"),
        "--native-physics-library", str(tmp_path / "library.so"),
        "--native-pool-reader", str(tmp_path / "reader"), "--mpas-source", str(tmp_path / "source"),
        "--preflight", *extra])


@pytest.fixture
def native_preflight(monkeypatch):
    # These door tests fake the forecast itself; the ported-route suite refusal
    # (resident_v842.ported_route_refusal, tested in test_mpasphys_resident_v842) is not their subject.
    monkeypatch.setattr("woof.hex.mpasphys.resident_v842.ported_route_refusal", lambda config: None)
    layout = SimpleNamespace(configs={"config_start_time": "2023-03-31_12:00:00", "config_dt": 360.0,
                                      "config_run_duration": "0_01:00:00", "config_do_restart": False})
    monkeypatch.setattr(forecast_v842, "decode_registry_pools", lambda *a, **k: layout)
    monkeypatch.setattr(forecast_v842, "read_streams", lambda *a, **k: ())
    # The placeholder init.nc is not NetCDF: stand in for the spacing the suite-route step reads from the
    # mesh files (forecast_door._mesh_file_spacing_m, routeb-door b4477dcd) with x1.2562's 480 km.
    monkeypatch.setattr(door, "_mesh_file_spacing_m", lambda arguments: 480_000.0)
    # ... and for the init xtime the suite route takes its start from (forecast_v842.bind_init_start,
    # routeb-door 1146680c), the same 2023-03-31 12Z the stand-in layout names.
    from datetime import datetime
    from woof.hex import lbc
    monkeypatch.setattr(lbc, "_open_lbc", lambda path: None)
    monkeypatch.setattr(lbc, "_read_xtime", lambda dataset, path: datetime(2023, 3, 31, 12))
    monkeypatch.delenv(licence_gate.ENVIRONMENT, raising=False)


def test_the_door_refuses_noahmp_on_an_mpas_suite_by_its_terms(tmp_path, native_preflight):
    arguments = _door_arguments(tmp_path, "--physics-suite", "convection_permitting",
                                "--physics", "config_lsm_scheme='sf_noahmp'")
    with pytest.raises(door.ForecastDoorRefusal) as refused:
        door._run_forecast(arguments)
    assert "UCAR Noah-MP terms" in str(refused.value) and "--licence-opt-in ucar-noahmp" in str(refused.value)
    assert not (Path(arguments.out) / "native-physics-request.json").exists()


def test_the_door_runs_an_opted_in_selection_and_records_it(tmp_path, native_preflight, capsys):
    arguments = _door_arguments(tmp_path, "--physics-suite", "mesoscale_reference",
                                "--physics", "config_lsm_scheme='sf_noahmp'", "--physics-mode", "exact",
                                "--licence-opt-in", "ucar-noahmp")
    assert door._run_forecast(arguments) == 0
    physics = json.loads((Path(arguments.out) / "native-physics-request.json").read_text())["physics"]
    assert [row["terms"] for row in physics["licence_opt_in"]] == ["ucar-noahmp"]
    assert "LICENCE OPT-IN ucar-noahmp (--licence-opt-in): land_surface=sf_noahmp" in capsys.readouterr().err


def _stubbed_library(tmp_path: Path) -> None:
    """The receipt the default `physics-resources build` writes beside its library: Noah-MP v5.0 stubbed."""
    (tmp_path / "build-receipt.json").write_text(json.dumps({"licence": {
        "stubbed": {"noahmp": {"terms": "ucar-noahmp"}}, "compiled": {}}}))


def test_the_door_refuses_an_opted_in_selection_on_a_stubbed_native_library(tmp_path, native_preflight):
    arguments = _door_arguments(tmp_path, "--physics-suite", "mesoscale_reference",
                                "--physics", "config_lsm_scheme='sf_noahmp'", "--physics-mode", "exact",
                                "--licence-opt-in", "ucar-noahmp")
    _stubbed_library(tmp_path)
    with pytest.raises(door.ForecastDoorRefusal) as refused:
        door._run_forecast(arguments)
    assert "woof hex physics-resources build --licence-opt-in ucar-noahmp" in str(refused.value)
    assert not (Path(arguments.out) / "native-physics-request.json").exists()


def test_a_default_run_on_the_stubbed_native_library_is_unchanged(tmp_path, native_preflight, capsys):
    arguments = _door_arguments(tmp_path, "--physics-suite", "mesoscale_reference", "--physics-mode", "exact")
    _stubbed_library(tmp_path)
    assert door._run_forecast(arguments) == 0
    physics = json.loads((Path(arguments.out) / "native-physics-request.json").read_text())["physics"]
    assert set(physics) == {"schema", "suite", "selectors", "options", "backends"}
    assert "LICENCE" not in capsys.readouterr().err


def test_the_door_leaves_a_default_suite_receipt_unchanged(tmp_path, native_preflight, capsys):
    arguments = _door_arguments(tmp_path, "--physics-suite", "mesoscale_reference", "--physics-mode", "exact")
    assert door._run_forecast(arguments) == 0
    physics = json.loads((Path(arguments.out) / "native-physics-request.json").read_text())["physics"]
    assert set(physics) == {"schema", "suite", "selectors", "options", "backends"}
    assert "LICENCE" not in capsys.readouterr().err


def test_the_door_runs_the_rrfs_stage_with_no_flag(tmp_path, native_preflight, capsys, monkeypatch):
    """The RRFS door with no flag resolves the fork's hrrrv5 and hands it to the native route unrefused."""
    taken = {}
    monkeypatch.setattr(forecast_v842, "run_forecast",
                        lambda arguments, config, text: taken.update(config=config) or 0)
    arguments = _door_arguments(tmp_path, "--physics-suite", "hrrrv5", "--physics-mode", "exact")
    assert door._run_forecast(arguments) == 0
    config = taken["config"]
    assert getattr(config, "native_suite", config.suite) == "hrrrv5" and config.table_key == "mpasphys_rrfs_44c0849"
    assert config.selectors["gwdo"] == "bl_ugwp_gwdo" and "licence_opt_in" not in config.receipt()
    assert str(config.backends["gwdo"]) == "hex-cuda"
    assert "LICENCE" not in capsys.readouterr().err
