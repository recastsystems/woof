"""A standalone prep manifest must survive a launch using the same fetch folder."""
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof import fetch_routes, mapped_authoring, runplan, source_cli, stage_cli
from woof.config import RunConfig
from woof.experiment import experiment_from_run_config
from test_audit_area1_acquisition import _member_inputs
from test_ux_cli_polish import _fake_bridge_identity


@pytest.mark.parametrize("source,member", [("aigefs", "mem007"), ("gefs", "p05")])
def test_printed_member_prep_then_go_keeps_both_manifests(
        tmp_path, monkeypatch, source, member):
    data = tmp_path / "shared data"
    data.mkdir()
    handoff, _, _ = _member_inputs(data, monkeypatch, source, member)
    donor = tmp_path / "donor.grib2"
    donor.write_bytes(b"fixture donor")
    request = fetch_routes.resolve_request(
        source, cycle=datetime(2026, 7, 29), hours=6, member=member)
    fetch_routes.write_handoff(request, data,
                               donor_files={d.role: donor for d in request.donors})
    handoff = json.loads((data / fetch_routes.PREP_ARGUMENTS_NAME).read_text())
    tools = []
    for role in ("grib2-inventory", "grib2-dump"):
        path = tmp_path / role
        path.write_bytes(role.encode())
        tools += ["--" + role, str(path)]
    monkeypatch.setattr(mapped_authoring, "bridge_identity", _fake_bridge_identity)
    monkeypatch.setattr(source_cli, "_distribution_decoder", lambda path, *a: path)

    # The printed command reads the upstream list. Run its actual authoring
    # door, stopping before field decoding and numerical preparation.
    assert source_cli.main([*handoff["argv"], *tools, "--author-only"]) == 0
    standalone = data / "inputs.json"
    original = standalone.read_bytes()
    config = tmp_path / "config.toml"
    config.write_text(f'[fetch]\nsource="{source}"\nmember="{member}"\n'
                      'cycle="2026-07-29T00"\nhours=6\ncadence=6\n')
    config.with_suffix(".namelist.wps").write_text("&share /\n")
    plan = SimpleNamespace(run_options={"data_dir": str(data),
                                        "geog_root": str(tmp_path)}, config_intent={})
    observer = SimpleNamespace(enter_stage=lambda *a, **k: None,
                               finish_stage=lambda **k: None,
                               arm_first_products=lambda *a, **k: None)
    manifests = []
    forecasts = []
    run_prep = runplan._run_prep

    def prepare(arguments):
        run_prep([*arguments, *tools, "--author-only"])
        path = Path(arguments[arguments.index("--author-input-manifest") + 1])
        manifests.append(path)
        Path(arguments[arguments.index("--output-root") + 1]).mkdir(parents=True)

    monkeypatch.setattr(runplan, "_run_prep", prepare)
    monkeypatch.setattr(runplan, "_run_fetch", lambda *a, **k: {})
    monkeypatch.setattr(stage_cli, "resolve_bundle", lambda *a: {"layout": "single", "source": source})
    monkeypatch.setattr(stage_cli, "sim_command", lambda *a, **k: ["python", "-m", "fixture"])
    monkeypatch.setattr(runplan, "_chain_render_plan", lambda *a, **k: {})
    monkeypatch.setattr(runplan, "_clear_forecast_output", lambda path, **k: path)
    monkeypatch.setattr(runplan, "_staged_forecast", lambda *a, **k: forecasts.append(True))
    monkeypatch.setattr(runplan, "_chain_render", lambda *a, **k: {})
    monkeypatch.setattr(runplan, "_asserted_profile", lambda *a, **k: None)
    run = tmp_path / "run"
    # A launch has a root even when preparation and forecast are stubbed.
    # The empty-domain fake broke when chemistry began reading that root.
    exp = experiment_from_run_config(
        RunConfig(nx=8, ny=8, nz=4, dx=3000.0, dy=3000.0, dt=10.0,
                  ztop=10000.0, run_seconds=21600.0), datetime(2026, 7, 29))
    runplan._staged_chain(plan, config_path=config,
                         exp=exp,
                         observer=observer, run_dir=run)
    assert forecasts == [True]
    assert standalone.read_bytes() == original
    assert manifests == [run / "chain" / "inputs.json"]
    authored = json.loads(manifests[0].read_text())
    verified = json.loads((run / "chain/prep/forcing-member.json").read_text())
    assert verified["member"] == member
    actual = [(manifests[0].parent / row["path"]).resolve()
              for row in authored["primary_files"]]
    assert actual == [Path(row["path"]) for row in verified["files"]]
