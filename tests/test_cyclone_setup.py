from datetime import datetime
import argparse
import json
from types import SimpleNamespace
import tomllib

import pytest

from woof import cyclone_setup as tc
from woof import domain_wizard as dw


def test_latest_map_resolves_once_and_selects_only_gfs_f000_pressure_winds(monkeypatch):
    calls = []
    def resolve(raw, **kwargs):
        calls.append((raw, kwargs))
        return datetime(2026, 9, 9, 0)
    monkeypatch.setattr(dw, "_resolve_cycle", resolve)
    result = tc.latest_map()
    assert calls == [("latest", {"source": "gfs", "hours": 0})]
    assert result["cycle"] == "2026090900"
    assert result["map_request"] == {"source": "gfs", "date": "2026-09-09",
        "hour": 0, "forecast_hour": 0, "member": 0, "product": "mslp_10m_winds",
        "bounds": [-85., -180., 85., 180.]}
    assert result["forecast_started"] is False


@pytest.mark.parametrize("point", [(18., -65.), (35., -75.), (-18., 65.)])
def test_config_has_immediate_centered_12_to_3km_vortex_nest_and_full_physics(point):
    from woof.companion_domains import VORTEX_PRESET
    text, exp = tc.configuration_text(cycle="2026090900", point=point)
    raw = tomllib.loads(text)
    parent, child = exp.domains
    assert (parent.run.dx, child.run.dx) == (12000., 3000.)
    assert (parent.run.nx, parent.run.ny) == tc.ROOT_DIMS
    assert (child.run.nx, child.run.ny) == tc.CHILD_DIMS
    assert child.parent_grid_ratio == child.parent_time_step_ratio == 4
    assert raw["projection"]["ref_lat"] == point[0]
    assert raw["projection"]["ref_lon"] == point[1]
    assert {k: v for k, v in raw["domain"][1]["follow"].items() if k != "track"} == {**VORTEX_PRESET, "level_hpa": 0., "threshold": 1100.,
            "pressure_center": "minimum", "fallback_level_hpa": 850., "fallback_deficit_m": 25.}
    assert child.follow.track.path == "storm-track.d02.csv"
    assert "spawn" not in raw["domain"][1] and "retire" not in raw["domain"][1]
    assert child.i_parent_start == 1 + (parent.run.nx - child.run.nx // 4) // 2
    assert child.j_parent_start == 1 + (parent.run.ny - child.run.ny // 4) // 2
    assert exp.run_seconds == 21600
    assert raw["fetch"]["cycle"] == "2026-09-09T00"
    assert raw["fetch"]["hours"] == 6
    for domain in exp.domains:
        assert domain.run.mp_physics == 10 and domain.run.bl_pbl_physics == 1
        assert domain.run.sf_surface_physics == 2 and domain.run.sf_sfclay_physics == 91
        assert domain.run.ra_lw_physics == domain.run.ra_sw_physics == 4
        assert domain.run.no_mp_heating == 0 and domain.run.moist
    assert parent.run.cu_physics == 1 and child.run.cu_physics == 0
    assert child.run.dt == parent.run.dt / 4


def test_creation_never_reresolves_latest_or_accepts_bad_selection():
    with pytest.raises(ValueError, match="exact cycle"):
        tc.configuration_text(cycle="latest", point=(18., -65.))
    with pytest.raises(ValueError, match="00/06/12/18"):
        tc.configuration_text(cycle="2026090901", point=(18., -65.))
    with pytest.raises(ValueError, match="finite center"):
        tc.configuration_text(cycle="2026090900", point=(float("nan"), -65.))


def test_inner_core_controls_the_carried_nest_placement():
    text, exp = tc.configuration_text(cycle="2026090900", point=(20.,-80.),
                                      inner_nest=(3,120.))
    raw = tomllib.loads(text)
    assert len(exp.domains) == 3
    assert exp.domains[-1].run.dx == 1000.
    assert raw['domain'][1]['follow']['refine_grid_id'] == exp.domains[-1].grid_id
    assert exp.domains[1].follow.tracker.refine_grid_id == 3


def test_target_machine_reaches_pricing_and_memory_refusal_keeps_config_unpublished(monkeypatch):
    seen = []
    machine = object()
    sizing = dw.SizingBudget(32., 30 * dw.GIB, None, "target", measured=True)
    phases = SimpleNamespace(peak_envelope_bytes=100, binding_phase="forecast",
                             verdict=lambda budget: "fixture exceeds target")
    def price(exp, **kwargs):
        seen.append(kwargs)
        return phases
    monkeypatch.setattr(dw, "_sizing_phases", price)
    monkeypatch.setattr(dw, "sizing_budget_bytes", lambda *args, **kwargs: 200)
    result = tc.plan_cyclone(cycle="2026090900", point=(18., -65.), sizing=sizing,
                             target_machine=machine)
    assert seen[0]["machine"] is machine and seen[0]["free_bytes"] == sizing.free_bytes
    assert result["forecast_started"] is False and result["memory"]["sizing_basis"] == "measured-available"
    assert [domain["following"] for domain in result["domains"]] == [False, True]
    monkeypatch.setattr(dw, "sizing_budget_bytes", lambda *args, **kwargs: 50)
    with pytest.raises(Exception, match="selected computer"):
        tc.plan_cyclone(cycle="2026090900", point=(18., -65.), sizing=sizing,
                        target_machine=machine)


def test_cli_creation_publishes_valid_wps_once_without_running_forecast(tmp_path, monkeypatch, capsys):
    from woof import companion_query
    from woof.namelist_import import parse_namelist_text
    parser = argparse.ArgumentParser()
    tc.register_cli(parser.add_subparsers(dest="command", required=True))
    out = tmp_path / "cyclone.toml"
    args = parser.parse_args(["cyclone-setup", "--cycle", "2026090900", "--point=18,-65",
                             "--vram-gib", "32", "--tiles", "off", "--out", str(out), "--json"])
    budget = dw.SizingBudget(32., 30 * dw.GIB, None, "fixture", measured=False)
    monkeypatch.setattr(dw, "_domain_target_hardware", lambda args: (budget, None, False))
    monkeypatch.setattr(dw, "_sizing_phases", lambda *args, **kwargs:
        SimpleNamespace(peak_envelope_bytes=100, binding_phase="forecast"))
    monkeypatch.setattr(dw, "sizing_budget_bytes", lambda *args, **kwargs: 200)
    monkeypatch.setattr(companion_query, "inspect_configuration", lambda path:
        {"schema": "arwen.companion-configuration.v1", "config_path": str(path)})
    assert tc.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["configuration"]["config_path"] == str(out)
    assert result["forecast_started"] is False
    wps = parse_namelist_text(out.with_suffix(".namelist.wps").read_text())
    assert wps["share"]["interval_seconds"] == [10800]
    assert wps["share"]["max_dom"] == [2]
    assert wps["geogrid"]["parent_grid_ratio"] == [1, 4]
    assert len(list(tmp_path.iterdir())) == 3
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert tc.main(args) == 1
    assert "never overwrites" in json.loads(capsys.readouterr().out)["error"]
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_unresolved_cycle_cli_refuses_before_any_target_probe(tmp_path, monkeypatch, capsys):
    args = SimpleNamespace(latest_map=False, cycle="latest", point="18,-65")
    def forbidden(*args, **kwargs):
        raise AssertionError("unresolved selected cycle must not probe hardware")
    monkeypatch.setattr(dw, "_domain_target_hardware", forbidden)
    assert tc.main(args) == 1
    assert "exact cycle" in json.loads(capsys.readouterr().out)["error"]
