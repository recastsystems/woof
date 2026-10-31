"""Single-domain HRRR quick looks must survive the native prepare relay."""
import copy
from types import SimpleNamespace
import tomllib

import pytest

from woof.experiment import build_experiment, load_experiment
from woof.experiment_document import publish_experiment_document
from woof.hrrr_route_inputs import render_namelist_input
from tools.prepare_hrrr_wrf import _configured_defaults
from test_hrrr_configured_physics import _native_tables


RADAR = "REFL_10CM,T2,U10,V10,RAINNC,RAINC,UP_HELI_MAX,Q2,PSFC,T,PH,PHB,U,V,W".split(",")


def _case(variables=None, quicklook=True):
    raw, _ = _native_tables()
    raw["domain"][0]["history_interval_s"] = 3600.0
    if quicklook:
        raw["domain"][0]["quicklook"] = {"every_s": 300.0}
        if variables:
            raw["domain"][0]["quicklook"]["history_vars"] = variables
    return raw, build_experiment(copy.deepcopy(raw), source="quicklook regression")


@pytest.mark.parametrize("variables", [None, RADAR])
def test_published_quicklook_roundtrips_cadence_and_variables(tmp_path, variables):
    raw, exp = _case(variables)
    path = publish_experiment_document(tmp_path / "published.toml", raw, exp)
    assert tomllib.loads(path.read_text())["domain"] == raw["domain"]
    restored = load_experiment(path)
    assert restored.root.history_interval_s == 300
    assert restored.root.full_history_interval_s == 3600
    assert restored.root.quicklook_output.history_vars == exp.root.quicklook_output.history_vars


def test_quicklook_identity_ignores_diagnostic_path_but_binds_selection():
    from woof.ingest.prepared_cache import prepared_domain_config_identity
    raw, first = _case()
    second = build_experiment(copy.deepcopy(raw), source="another/location.toml")
    assert prepared_domain_config_identity(first.root) == prepared_domain_config_identity(second.root)
    raw["domain"][0]["quicklook"]["history_vars"] = RADAR
    changed = build_experiment(raw, source="another/location.toml")
    assert prepared_domain_config_identity(first.root) != prepared_domain_config_identity(changed.root)


@pytest.mark.parametrize("quicklook", [False, True])
def test_omitted_prepare_flag_defaults_to_full_cadence(monkeypatch, quicklook):
    _, exp = _case(quicklook=quicklook)
    monkeypatch.setattr("woof.hrrr_configuration.native_configuration_defaults",
                        lambda **kwargs: exp)
    args = SimpleNamespace(run_seconds=None, history_interval_seconds=None,
                           experiment_config="case.toml")
    _configured_defaults(args)
    assert args.history_interval_seconds == 3600
    assert args.run_seconds == exp.run_seconds


@pytest.mark.parametrize("quicklook", [False, True])
def test_route_namelist_uses_full_history_cadence(quicklook):
    _, exp = _case(quicklook=quicklook)
    line = next(line for line in render_namelist_input(exp).splitlines()
                if line.strip().startswith("history_interval_s "))
    assert line.split("=")[1].strip().rstrip(",") == "3600"


@pytest.mark.parametrize("quicklook", [False, True])
def test_prepare_receipt_validates_the_alarm_not_the_full_frame_interval(quicklook):
    from tools.prepare_hrrr_wrf import _validate_prepared_history_cadence
    _, exp = _case(quicklook=quicklook)
    _validate_prepared_history_cadence(
        {"history_interval_seconds": exp.root.history_interval_s}, exp)
    for invalid in (None, True, float("nan"), 600, 3600 if quicklook else 300):
        with pytest.raises(RuntimeError, match="history cadence differs"):
            _validate_prepared_history_cadence({"history_interval_seconds": invalid}, exp)


def test_chain_passes_full_cadence_to_native_preparation(tmp_path, monkeypatch):
    import woof.runplan as runplan
    from test_runplan_hrrr_tree import _Observer, _plan
    from woof.toml_document import emit_experiment_toml

    plan = _plan(tmp_path, ladder=None, root_dx_km=3, hours=1, history_interval_s=3600)
    raw, _ = _case()
    raw["fetch"] = {"source": "hrrr", "cycle": "2024-05-03T12", "hours": 1,
                    "as_posted": False}
    config = tmp_path / "quicklook.toml"
    config.write_text(emit_experiment_toml(raw))
    exp = load_experiment(config)
    captured = []

    def fetch(args, *unused, **kwargs):
        from pathlib import Path
        out = Path(args[args.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "SHA256SUMS").write_text("0" * 64 + "  hrrr.t12z.wrfprsf00.grib2\n")
        return {}

    class Captured(Exception):
        pass

    def prepare(*args, arguments, **kwargs):
        captured.extend(arguments)
        raise Captured

    monkeypatch.setattr(runplan, "_run_fetch", fetch)
    monkeypatch.setattr(runplan, "_prepare_stage", prepare)
    with pytest.raises(Captured):
        runplan._hrrr_chain(plan, config_path=config, exp=exp,
                           observer=_Observer(), run_dir=plan.run_dir, prepare_only=True)
    assert captured[captured.index("--history-interval-seconds") + 1] == "3600"
