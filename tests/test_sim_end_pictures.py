"""`woof sim` ends a run that drew pictures with `woof go`'s render stage over the same folder.

THE BREAKAGE THIS PREVENTS (WOOF canary run_6048ab38808a0625, 2026-11-10): a Hex with Regional 1 km nest runs through
`woof sim`, which drew each frame as it landed and never ran the end-of-run windowed pass, so `uh_2to5km_run_max`
stopped at F002 and `uh_2to5km_3h_max` was never drawn; `woof go` of the same request drew both at every hour.
"""
from __future__ import annotations

from woof import go_cli, stage_cli


def test_the_end_of_run_stage_draws_into_the_runs_own_folders(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(go_cli, "_render_stage", lambda plan, **kw: seen.append((plan, kw)) or True)
    stage_cli._finish_sim_pictures(tmp_path / "run", "uh_2to5km_run_max,uh_2to5km_3h_max", None, explain=False)
    plan, kw = seen[0]
    assert plan["run"] == tmp_path / "run" and plan["render"] == tmp_path / "run" / "png"
    assert plan["render_products"] == "uh_2to5km_run_max,uh_2to5km_3h_max"
    assert kw == {"explain": False, "door": "sim"}
    seen.clear()
    stage_cli._finish_sim_pictures(tmp_path / "run", "all", tmp_path / "pics", explain=True)
    assert seen[0][0]["render"] == tmp_path / "pics"


def test_a_failed_end_of_run_stage_leaves_the_forecast_result(tmp_path, monkeypatch, capsys):
    def boom(plan, **kw):
        raise OSError("renderer missing")

    monkeypatch.setattr(go_cli, "_render_stage", boom)
    stage_cli._finish_sim_pictures(tmp_path, "uh_2to5km_run_max", None, explain=False)
    assert "end-of-run pictures were not drawn: OSError: renderer missing" in capsys.readouterr().err
