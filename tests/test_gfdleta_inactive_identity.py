from dataclasses import fields, replace
import inspect
import pytest
from woof.config import RunConfig
from woof.experiment import build_experiment, domain_config_document, experiment_config_document
from woof.core.model import restart_identity_payload
from woof.io.restart import configuration_echo, _configuration_fingerprint

@pytest.mark.parametrize("value", [0, 1])
def test_inactive_co2_selection_preserves_documents_and_checkpoint(value):
    from test_radiation_driver_identity import _experiment
    exp = _experiment()
    base = exp.root.run
    selected = replace(exp, domains=tuple(replace(d, run=replace(d.run, co2tf=value)) for d in exp.domains))
    assert "co2tf" not in domain_config_document(selected.root)["run"]
    assert experiment_config_document(selected) == experiment_config_document(exp)
    assert restart_identity_payload(selected) == restart_identity_payload(exp)
    assert "co2tf" not in configuration_echo(selected.root.run)
    assert _configuration_fingerprint(selected.root.run) == _configuration_fingerprint(base)

@pytest.mark.parametrize("pair", [(99,99), (99,0), (0,99)])
def test_selected_gfdl_co2_setting_remains_trajectory_bound(pair):
    cfg = RunConfig(nx=24, ny=24, nz=16, dx=1000, dy=1000, ztop=8000, dt=1, run_seconds=10, ra_physics=0, ra_lw_physics=pair[0], ra_sw_physics=pair[1])
    assert configuration_echo(cfg)["co2tf"] == 1
    assert configuration_echo(replace(cfg,co2tf=0))["co2tf"] == 0
    assert _configuration_fingerprint(cfg) != _configuration_fingerprint(replace(cfg,co2tf=0))

def test_co2_setting_does_not_shift_positional_prefix():
    names = [f.name for f in fields(RunConfig)]
    assert names[names.index("co2tf") + 1] == "ifsnow"
    assert inspect.signature(RunConfig).parameters["co2tf"].kind == inspect.Parameter.KEYWORD_ONLY

@pytest.mark.parametrize("value", [0, 1])
def test_inactive_checkpoint_reader_accepts_omitted_co2_selection(value):
    from woof.io.restart import _require_config_match
    from test_radiation_driver_identity import _experiment
    cfg = replace(_experiment().root.run, co2tf=value)
    _require_config_match(configuration_echo(cfg), cfg, "inactive.npz")

@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_active_checkpoint_reader_refuses_unknown_or_changed_co2_selection(damage):
    from woof.io.restart import _require_config_match, RestartMismatchError
    from test_radiation_driver_identity import _experiment
    cfg = replace(_experiment().root.run, ra_physics=0,ra_lw_physics=99,ra_sw_physics=99)
    header = configuration_echo(cfg)
    if damage == "missing": header.pop("co2tf")
    else: header["co2tf"] = 0
    with pytest.raises(RestartMismatchError, match="co2tf"):
        _require_config_match(header, cfg, "selected.npz")
