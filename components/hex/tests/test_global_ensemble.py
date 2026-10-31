"""The global ensemble: member inits, resident groups on the card, splits on refusal.

CPU only: the door commands are answered by a stand-in."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pytest

from woof.hex import device_admission, global_door, global_ensemble
from woof.hex.global_ensemble import EnsembleDoorError, parse_members


def test_member_specs():
    assert parse_members("1-3") == [1, 2, 3]
    assert parse_members("0,5,7-8") == [0, 5, 7, 8]
    for bad in ("3-1", "1,1", ""):
        with pytest.raises(EnsembleDoorError):
            parse_members(bad)


def test_mutex_uuid_visibility_retains_four_member_admission(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(global_door.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout="0, GPU-card-zero, 97887\n1, GPU-card-one, 97887\n"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-card-zero")
    assert global_door._visible_cards() == [97887 * 1024**2]
    value = {"mesh": {"cells": 40962}}
    assert global_ensemble.group_limit(value, SimpleNamespace(members_per_card="auto")) == 4
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-card")
    assert global_door._visible_cards() == []  # Ambiguous UUID prefixes are not a device.


def test_numeric_and_empty_visibility_masks(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(global_door.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout="0, GPU-zero, 97887\n1, GPU-one, 32768\n"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert global_door._visible_cards() == [32768 * 1024**2, 97887 * 1024**2]
    for mask in ("", "-1"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
        assert global_door._visible_cards() == []


def test_global_meshes_have_co_tenancy_rows_and_the_floor_sum_still_binds():
    gib = 1024**3
    got = device_admission.admit_co_tenancy(mesh_kind="global", cells=40_962, levels=55, members=2,
                                            card_bytes=int(31.8 * gib), free_bytes=30 * gib,
                                            solo_required_bytes=12 * gib)
    # Since fix-admission-render (bbaca17f) the reported maximum is also
    # limited by measured free memory: 30 GiB // 12 GiB = 2. The table row
    # still allows 3, and three members still refuse on the floor sum.
    assert got["maximum_members"] == 2
    assert got["table_maximum_members"] == 3
    assert got["memory_limited_members"] == 2
    with pytest.raises(MemoryError, match="co-tenancy refuses 3"):
        device_admission.admit_co_tenancy(mesh_kind="global", cells=40_962, levels=55, members=3,
                                          card_bytes=int(31.8 * gib), free_bytes=30 * gib,
                                          solo_required_bytes=12 * gib)
    with pytest.raises(MemoryError, match="table limit 1"):
        device_admission.admit_co_tenancy(mesh_kind="global", cells=2_621_442, levels=55, members=2,
                                          card_bytes=96 * gib, free_bytes=90 * gib, solo_required_bytes=gib)


def _plan(tmp_path, **extra):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    global_door.add_global_parser(commands)
    argv = ["global", "--row", "global-120km", "--cycle", "2026-10-16T00", "--work-dir", str(tmp_path)]
    for key, value in extra.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return global_door.plan(parser.parse_args(argv)), parser.parse_args(argv)


def test_each_member_gets_its_own_init(tmp_path):
    value, _ = _plan(tmp_path, source="gefs", members="1-3")
    assert [m["run_id"] for m in value["members"]] == ["mem001", "mem002", "mem003"]
    for row in value["members"]:
        argv = row["init_argv"]
        assert argv[argv.index("--member") + 1] == str(row["member"])
        assert argv[argv.index("--out-dir") + 1] == row["init_dir"]
        assert row["init_dir"].startswith(str(tmp_path / "members"))


def test_one_member_and_a_one_state_source(tmp_path):
    value, _ = _plan(tmp_path, source="gefs", member=7)
    assert value["commands"]["init"][-2:] == ["--member", "7"]
    assert "member 7" in " ".join(value["commands"]["forecast"])
    generated, _ = _plan(tmp_path, source="era5", members="1-3")
    assert [m["run_id"] for m in generated["members"]] == ["mem001", "mem002", "mem003"]
    assert all(m["member"] is None and "--member" not in m["init_argv"] for m in generated["members"])
    assert len({m["stochastic"]["member"] for m in generated["members"]}) == 3
    assert all(m["stochastic"]["sppt"] and m["stochastic"]["skeb"] for m in generated["members"])


def test_generated_members_refuse_repeated_unperturbed_forecasts(tmp_path):
    with pytest.raises(global_door.GlobalDoorError, match="repeated source input"):
        _plan(tmp_path, source="era5", members="1-3", stochastic='{"sppt":false,"skeb":false}')


def test_generated_members_share_acquisition_with_independent_forecast_binding(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test-allocation")
    value, args = _plan(tmp_path, source="era5", members="1-3", lagged="0,6", devices="0", members_per_card=2)
    assert len(value["members"]) == 6
    assert len({m["init"] for m in value["members"]}) == 2
    calls, launched = [], []
    def command(argv, log, env):
        if "--members" in argv:
            launched.extend(json.loads(Path(argv[argv.index("--members") + 1]).read_text()))
        else:
            calls.append(argv)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("complete")
        return 0
    monkeypatch.setattr(global_ensemble, "_cli", command)
    result = global_ensemble.run_inits(value, value["members"], args, _Status(), {})
    assert result["built"] == result["source_inputs"] == len(calls) == 2
    assert result["bound_members"] == 6
    assert all(m["init_returncode"] == 0 for m in value["members"])
    assert global_ensemble.run_forecasts(value, value["members"], args, _Status(), {})["failed"] == []
    assert len({m["stochastic"]["member"] for m in launched}) == 6
    assert {json.loads(m["init_source"])["generated_member"] for m in launched} == {1, 2, 3}
    assert {json.loads(m["init_source"])["lag_hours"] for m in launched} == {0, 6}


def test_analysis_lags_reach_forecast_with_private_clocks(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test-allocation")
    from datetime import datetime, timedelta
    value, args = _plan(tmp_path, source="era5", lagged="0,6,12,18", members_per_card=4, devices="0")
    captured = []
    def launch(argv, log, env):
        captured.extend(json.loads(Path(argv[argv.index("--members") + 1]).read_text()))
        log.write_text("complete")
        return 0
    monkeypatch.setattr(global_ensemble, "_cli", launch)
    assert global_ensemble.run_forecasts(value, value["members"], args, _Status(), {})["failed"] == []
    assert len(captured) == 4
    for record, lag in zip(captured, (0, 6, 12, 18)):
        p = json.loads(record["init_source"])
        assert p["lead_hours"] == 0 and p["spinup_hours"] == lag
        assert record["start_time"] == p["source_cycle"] == p["valid_time"]
        start = datetime.strptime(record["start_time"], "%Y-%m-%d_%H:%M:%S")
        assert start + timedelta(minutes=record["history_from_minutes"]) == datetime(2026, 10, 16)
        assert start + timedelta(hours=record["hours"]) == datetime(2026, 10, 17)


def test_postprocessing_refuses_different_analysis_lags(tmp_path):
    original, _ = _plan(tmp_path, source="era5", lagged="0,6")
    saved = []
    for m in original["members"]:
        forecast = tmp_path / "ensemble/g01" / m["run_id"]
        forecast.mkdir(parents=True)
        saved.append({**m, "forecast": str(forecast)})
    (tmp_path / "ensemble.json").write_text(json.dumps({"source": "era5", "cycle": original["cycle"],
        "row": original["row"]["id"], "members": saved}))
    changed, _ = _plan(tmp_path, source="era5", lagged="0,12")
    with pytest.raises(EnsembleDoorError, match="no matching forecast folder"):
        global_ensemble.restore_forecasts(changed, changed["members"])


def test_stochastic_member_defaults_are_decided_before_card_grouping(tmp_path):
    small, _ = _plan(tmp_path, source='gefs', members='1-3', stochastic_seed=71)
    large, _ = _plan(tmp_path, source='gefs', members='1-10')
    assert all(m['stochastic']['sppt'] and m['stochastic']['skeb'] for m in small['members'])
    assert [m['stochastic']['member'] for m in small['members']] == [1,2,3]
    assert all(m['stochastic']['seed'] == 71 for m in small['members'])
    assert all(not m['stochastic']['sppt'] for m in large['members'])
    single, _ = _plan(tmp_path, source='gefs', member=1, stochastic='{"sppt":true,"seed":71}')
    command = single['commands']['forecast']
    assert json.loads(command[command.index('--stochastic')+1]) == {'sppt':True,'seed':71}


def test_a_group_runs_resident_through_the_roster_on_one_card(tmp_path):
    value, _ = _plan(tmp_path, source="gefs", members="1-2", devices="0")
    roster = tmp_path / "g01.roster.json"
    argv = global_ensemble.forecast_argv(value, roster, tmp_path / "out", tmp_path / "scr", "/x/init.nc")
    assert argv[-2:] == ["--members", str(roster)] and "--samehost-devices" not in argv
    assert argv[argv.index("--init") + 1] == "/x/init.nc"


class _Status:
    def __init__(self):
        self.notes = []

    def stage(self, *a, **k):
        pass

    def step(self, *a, note=None, **k):
        self.notes.append(note)

    def note(self, text):
        self.notes.append(text)


def test_a_group_the_card_refuses_is_split_and_every_member_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-pool-allocation")
    value, args = _plan(tmp_path, source="gefs", members="1-3", members_per_card=3, devices="0")
    members = value["members"]
    calls = []

    def fake(argv, log, env):
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-pool-allocation"
        roster = json.loads(Path(argv[argv.index("--members") + 1]).read_text())
        calls.append([m["run_id"] for m in roster])
        log.parent.mkdir(parents=True, exist_ok=True)
        if len(roster) == 3:
            log.write_text("MemoryError: co-tenancy refuses 3 members: table limit 3 ...")
            return 1
        log.write_text("ok")
        return 0

    monkeypatch.setattr(global_ensemble, "_cli", fake)
    status = _Status()
    result = global_ensemble.run_forecasts(value, members, args, status, {})
    assert calls[0] == ["mem001", "mem002", "mem003"] and sorted(sum(calls[1:], [])) == ["mem001", "mem002", "mem003"]
    assert result["failed"] == [] and any(g.get("split") for g in result["groups"])
    assert all(m["forecast"].endswith(m["run_id"]) for m in members)


@pytest.mark.parametrize("mask,requested,expected", [
    ("GPU-pool-allocation", "0", ["GPU-pool-allocation"]),
    ("2,5", "1,0", ["5", "2"]),
    ("GPU-a,GPU-b", None, ["GPU-a", "GPU-b"]),
    ("GPU-a,GPU-b", "GPU-b", ["GPU-b"]),
    ("MIG-pool-allocation", "0", ["MIG-pool-allocation"]),
    (None, "2,0", ["2", "0"]),
])
def test_device_selection_preserves_inherited_allocation(mask, requested, expected):
    environment = {} if mask is None else {"CUDA_VISIBLE_DEVICES": mask}
    assert global_ensemble.forecast_devices(requested, environment) == expected


@pytest.mark.parametrize("mask,requested", [
    ("GPU-a", "1"), ("2,5", "5"), ("GPU-a", "GPU-other"),
    ("GPU-a", "0,GPU-a"), ("", "0"), ("-1", "0"), ("GPU-a,GPU-a", None),
])
def test_device_selection_refuses_allocation_escape_or_duplicates(mask, requested):
    with pytest.raises(EnsembleDoorError, match="allocation"):
        global_ensemble.forecast_devices(requested, {"CUDA_VISIBLE_DEVICES": mask})


def test_init_jobs_auto_is_bounded_by_cores_and_host_ram(monkeypatch):
    """--init-jobs auto (unified image, 2026-10-16: a fixed 4 left members waiting on a 24-core box)."""
    value = {"mesh": {"cells": 40_962}}
    monkeypatch.setattr(global_ensemble.os, "sched_getaffinity", lambda pid: set(range(24)))
    monkeypatch.setattr(global_ensemble, "_available_host_bytes", lambda: 100 * 1024**3)
    assert global_ensemble.init_jobs("auto", value, 6) == 6
    assert global_ensemble.init_jobs("auto", value, 30) == 6
    assert global_ensemble.init_jobs("2", value, 6) == 2
    per_init = 40_962 * global_ensemble.INIT_BYTES_PER_CELL
    monkeypatch.setattr(global_ensemble, "_available_host_bytes", lambda: 2 * per_init + 1)
    assert global_ensemble.init_jobs("auto", value, 6) == 2
    monkeypatch.setattr(global_ensemble, "_available_host_bytes", lambda: 1)
    assert global_ensemble.init_jobs("auto", value, 6) == 1
    with pytest.raises(EnsembleDoorError):
        global_ensemble.init_jobs("0", value, 6)


def test_resumed_member_products_use_saved_forecast_outputs(tmp_path, monkeypatch):
    value, args = _plan(tmp_path, source='gefs', members='1-2')
    saved = []
    for m in value['members']:
        forecast = tmp_path / 'ensemble/g03' / m['run_id']
        forecast.mkdir(parents=True)
        saved.append({**m, 'forecast': str(forecast), 'group': 'g03', 'card': '0', 'forecast_returncode': 0})
    (tmp_path / 'ensemble.json').write_text(json.dumps({'source': 'gefs', 'cycle': value['cycle'],
        'row': value['row']['id'], 'members': saved}))
    calls = []
    monkeypatch.setattr(global_door, '_run_door', lambda argv, env: calls.append(argv) or 0)
    record = global_ensemble.run_ensemble(value, args, ['products', 'export'], _Status(), {})
    assert [a[1] for a in calls] == [m['forecast'] for m in saved] * 2
    assert all('/ensemble/g03/' in a[1] for a in calls)
    assert all(m['products_returncode'] == 0 for m in record['members'])


def test_member_postprocessing_refuses_foreign_manifest(tmp_path):
    value, args = _plan(tmp_path, source='gefs', members='1-2')
    (tmp_path / 'ensemble.json').write_text(json.dumps({'source': 'other', 'cycle': value['cycle'],
        'row': value['row']['id'], 'members': []}))
    with pytest.raises(EnsembleDoorError, match='different row, source or cycle'):
        global_ensemble.run_ensemble(value, args, ['products'], _Status(), {})


@pytest.mark.parametrize('scorer_rc,verified,expected_rc', [(3, True, 0), (3, False, 3), (2, True, 2)])
def test_member_scores_only_complete_on_verified_measurements(tmp_path, monkeypatch, scorer_rc, verified, expected_rc):
    from woof.hex import observation_scorecard

    value, args = _plan(tmp_path, source='gefs', members='1-2')
    saved = []
    for member in value['members']:
        forecast = tmp_path / 'ensemble/g01' / member['run_id']
        forecast.mkdir(parents=True)
        Path(member['init_dir']).mkdir(parents=True)
        saved.append({**member, 'forecast': str(forecast), 'forecast_returncode': 0})
    (tmp_path / 'ensemble.json').write_text(json.dumps({'source': value['source'], 'cycle': value['cycle'],
        'row': value['row']['id'], 'members': saved}))
    checked = []

    def completed(forecast, config):
        checked.append((forecast, json.loads(config.read_text())))
        return {'status': 'PARTIALLY_MEASURED', 'measured_metrics': 7, 'note': 'measured partial coverage'} if verified else None

    monkeypatch.setattr(global_door, '_run_door', lambda argv, env: scorer_rc)
    monkeypatch.setattr(observation_scorecard, 'completed_scorecard', completed)
    if expected_rc:
        # A score that does not complete is that member's missing score (graceful recovery): retried once, recorded
        # with its reason, and the ensemble goes on.
        record = global_ensemble.run_ensemble(value, args, ['score'], _Status(), {})
        assert all(m['score_returncode'] == expected_rc for m in record['members'])
        assert [(x['run_id'], x['stage']) for x in record['missing']] == [('mem001', 'score'), ('mem002', 'score')]
    else:
        record = global_ensemble.run_ensemble(value, args, ['score'], _Status(), {})
        assert all(m['score_returncode'] == 0 and m['scorer_returncode'] == 3 for m in record['members'])
        assert all(m['measured_metrics'] == 7 for m in record['members'])
    # an unverified score is checked on each of its two tries
    assert len(checked) == (2 if expected_rc == 0 else 4 if scorer_rc == 3 else 0)
    by_forecast = {m['forecast']: m for m in saved}
    for forecast, config in checked:
        assert config['init'] == by_forecast[forecast]['init']


def test_planner_prices_members_like_the_roster_admission(monkeypatch):
    """2026-10-20: 60 km GEFS members 1-2 on an RTX 5090 planned 3 per group, the roster refused 2 and split.

    The planner now charges the roster's own sum (solo floor + stochastic arrays) and plans 1; the refusal it
    used to hit reports the true memory limit, 1, not the solo-floor-only 2."""
    from types import SimpleNamespace
    from woof.hex import forecast_door
    from woof.hex.stochastic import member_options

    monkeypatch.setattr(global_door, "_visible_cards", lambda: [32607 * 1024**2])
    card = {"name": "NVIDIA GeForce RTX 5090", "free": 33_138_933_760, "total": 32607 * 1024**2,
            "sms": 170, "threads": 1536}
    monkeypatch.setattr(global_ensemble, "_probe_cards", lambda devices: [card])
    cells = 163_842
    members = [{"stochastic": member_options(n, source_members=2, ensemble_size=2)} for n in (1, 2)]
    value = {"mesh": {"cells": cells}, "members": members,
             "commands": {"forecast": ["forecast", "--mesh", "x1.163842"]}}
    assert global_ensemble.group_limit(value, SimpleNamespace(members_per_card="auto", devices=None)) == 1
    model = device_admission.model_for_card(device_admission.card_profile_from_attributes(
        card["name"], {"MultiProcessorCount": 170, "MaxThreadsPerMultiProcessor": 1536}), "global")
    solo = forecast_door.required_free_bytes(cells, model)
    extra = tuple(forecast_door.stochastic_array_bytes(cells, m["stochastic"]) for m in members)
    with pytest.raises(MemoryError, match="memory limit 1 "):
        device_admission.admit_co_tenancy(mesh_kind="global", cells=cells, levels=55, members=2,
                                          card_bytes=card["total"], free_bytes=card["free"],
                                          solo_required_bytes=solo, member_additional_bytes=extra)


def test_each_member_draws_its_own_hours_while_its_group_integrates(tmp_path, monkeypatch):
    """A hex global-120km ensemble (2 members, GFS 2026-10-24 00Z, 6 h): no member map existed
    until the forecast and every member's full products pass had ended, and all 610 maps reached the site at the end."""
    from unittest import mock
    from woof.hex import first_look

    value, args = _plan(tmp_path, source="gefs", members="1-2", members_per_card=2, devices="0")
    members = value["members"]
    runs, popens = [], []

    def fake(argv, log, env):
        out = Path(argv[argv.index("--out") + 1])
        for m in json.loads(Path(argv[argv.index("--members") + 1]).read_text()):
            folder = out / m["run_id"]
            folder.mkdir(parents=True)
            for h in range(2):
                frame = folder / f"cuda-history.2026-10-16_{h:02d}.00.00.nc"
                frame.write_bytes(b"frame")
                with (folder / first_look.READY).open("a") as stream:
                    stream.write(json.dumps({"step": h, "path": str(frame)}) + "\n")
        deadline = time.monotonic() + 5  # the forecast is still running while the looks draw
        while len(popens) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        log.write_text("ok")
        return 0

    def popen(argv, **kw):
        popens.append(argv)
        return mock.Mock(wait=lambda: 0, poll=lambda: 0)

    monkeypatch.setattr(global_ensemble, "_cli", fake)
    # The stand-in members need no allocation; a CPU host that hides its cards
    # (CUDA_VISIBLE_DEVICES="") must not refuse them.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with mock.patch.object(first_look.subprocess, "run", side_effect=lambda a, **k: runs.append(a) or mock.Mock(returncode=0)), \
            mock.patch.object(first_look.subprocess, "Popen", side_effect=popen):
        result = global_ensemble.run_forecasts(value, members, args, _Status(), {}, live=True)
    assert result["failed"] == []
    for m in members:
        own = tmp_path / "members" / m["run_id"]
        mine = [a for a in runs + popens if a[a.index("--out") + 1] == str(own / "products")]
        assert len(mine) == 2                                       # f000, then f001 while the forecast ran
        assert all(a[a.index("--history") + 1].startswith(m["forecast"]) for a in mine)
        assert all(a[a.index("--case") + 1] == global_ensemble.member_case(value, m) for a in mine)
        assert all(a[a.index("--source-label") + 1] == f"Hex forecast, member {m['run_id']} of 2" for a in mine)
        assert m["live_frames"] == str(own / first_look.LIVE_RECEIPT)
        assert json.loads(Path(m["live_frames"]).read_text())["drawn"] == [f"cuda-history.2026-10-16_{h:02d}.00.00.nc" for h in range(2)]

    calls = []
    monkeypatch.setattr(global_door, "_run_door", lambda argv, env: calls.append(argv) or 0)
    # A products-only run resumes from ensemble.json, live receipts included.
    (tmp_path / "ensemble.json").write_text(json.dumps({"source": "gefs", "cycle": value["cycle"],
                                                        "row": value["row"]["id"], "members": members}))
    fresh, _ = _plan(tmp_path, source="gefs", members="1-2", members_per_card=2, devices="0")
    global_ensemble.run_ensemble(fresh, args, ["products"], _Status(), {})
    for argv, m in zip(calls, members):
        assert argv[argv.index("--live-frames") + 1] == m["live_frames"]   # the full pass keeps the hours drawn live
        assert argv[argv.index("--case") + 1] == global_ensemble.member_case(value, m)


_PRO_6000 = {"name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "free": 97_300 * 1024**2,
             "total": 97_887 * 1024**2, "sms": 188, "threads": 1536}


@pytest.mark.parametrize("physics, expected", [((), 1), (("--physics-suite=convection_permitting",), 1),
                                                (("--physics-suite=arwen_mpas_column",), 2)])
def test_auto_members_per_card_prices_the_mpas_route_30km(monkeypatch, physics, expected):
    """U-H4 (global ensemble cycle, 2026-10-27): ``global --row global-30km --source gefs --members-per-card auto`` on
    RTX PRO 6000 cards planned 2 MPAS-physics members per card at the ArWen-route floor (43,080 MiB each); one member
    alone peaks 75,069 MiB, so all four ran out of memory ~3 min in.  A run that names no physics (the hex default)
    or names an MPAS suite is priced on the MPAS route and planned 1 per card; the WOOF route keeps its price."""
    from types import SimpleNamespace

    monkeypatch.setattr(global_door, "_visible_cards", lambda: [_PRO_6000["total"]])
    monkeypatch.setattr(global_ensemble, "_probe_cards", lambda devices: [dict(_PRO_6000)])
    value = {"mesh": {"cells": 655_362}, "members": [{"stochastic": None}, {"stochastic": None}],
             "commands": {"forecast": ["forecast", "--mesh", "x1.655362", *physics]}}
    assert global_ensemble.group_limit(value, SimpleNamespace(members_per_card="auto", devices=None)) == expected


def test_mpas_suite_roster_refuses_a_group_the_card_cannot_hold(tmp_path, monkeypatch):
    """U-H4, the runtime half: the MPAS-suite roster started every member at once with no memory decision.  Two 30 km
    members on one RTX PRO 6000 are now refused before any member starts, with the "co-tenancy refuses" text the
    ensemble door splits on; one member is left to the one-member door."""
    from types import SimpleNamespace
    from woof.hex import mesh_rows, multi_member

    monkeypatch.setattr(global_ensemble, "_probe_cards", lambda devices: [dict(_PRO_6000)])
    monkeypatch.setattr(mesh_rows, "_dimensions", lambda path: {"nCells": 655_362})
    member = SimpleNamespace(grid=tmp_path / "x1.655362.grid.nc", init=None, stochastic=None, headroom_mib=None,
                             lbc_dir=None, stop_on_refusal=False)
    assert multi_member.admit_suite_group([("mem001", member)], None) is None
    with pytest.raises(MemoryError, match=global_ensemble.REFUSED_GROUP + " 2 members"):
        multi_member.admit_suite_group([("mem001", member), ("mem002", member)], None)
    admitted = device_admission.route_required_free_bytes(
        655_362, device_admission.model_for_card(device_admission.card_profile_from_attributes(
            _PRO_6000["name"], {"MultiProcessorCount": 188, "MaxThreadsPerMultiProcessor": 1536}), "global"),
        route="mpas")
    # Never below the measured single-member peak, plus the curve's 10 %; and one member still fits the card.
    assert 75_069 * 1024**2 * 1.10 < admitted < _PRO_6000["free"]


_RTX_5090 = {"name": "NVIDIA GeForce RTX 5090", "free": 32_100 * 1024**2, "total": 32_607 * 1024**2,
             "sms": 170, "threads": 1536}


def test_a_30km_mpas_member_runs_over_several_32gb_cards(tmp_path, monkeypatch):
    """8x RTX 5090 fit (2026-10-27): a 30 km MPAS-route member peaks 75,069 MiB on one card and the ensemble door put
    every member on ONE card, so a 30 km global ensemble (the synthetic-radar DA cycle's members) could not run on a
    32 GB card at all.  Auto now spreads one member over the fewest cards whose per-rank price fits, as the forecast
    door's same-host ranks, with the member's own init, start, label and stochastic identity; a member that fits one
    card keeps one card and its roster."""
    value, args = _plan(tmp_path, source="gefs", members="1-3", devices="0,1,2,3,4,5,6,7")
    value["mesh"]["cells"] = 655_362
    monkeypatch.setattr(global_ensemble, "_probe_cards", lambda devices: [dict(_RTX_5090) for _ in range(8)])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    ranks = global_ensemble.cards_per_member(value, args)
    assert 2 <= ranks <= 4
    calls = []

    def fake(argv, log, env):
        calls.append((argv, env["CUDA_VISIBLE_DEVICES"]))
        log.write_text("ok")
        return 0

    monkeypatch.setattr(global_ensemble, "_cli", fake)
    result = global_ensemble.run_forecasts(value, value["members"], args, _Status(), {})
    assert result["failed"] == [] and result["cards_per_member"] == ranks and result["members_per_group"] == 1
    assert len(calls) == 3 and all("--members" not in argv for argv, _ in calls)
    for argv, cards in calls:
        assert len(cards.split(",")) == ranks
        assert argv[argv.index("--samehost-devices") + 1] == ",".join(str(i) for i in range(ranks))
        member = next(m for m in value["members"] if argv[argv.index("--init") + 1] == m["init"])
        assert argv[argv.index("--out") + 1] == member["forecast"]
        assert json.loads(argv[argv.index("--stochastic") + 1]) == json.loads(json.dumps(member["stochastic"]))
        assert json.loads(argv[argv.index("--init-source") + 1]) == member["provenance"]
    # 120 km members still fit one card each: one card, the roster.
    small, small_args = _plan(tmp_path / "small", source="gefs", members="1-2", devices="0,1")
    monkeypatch.setattr(global_ensemble, "_probe_cards", lambda devices: [dict(_RTX_5090) for _ in range(2)])
    assert global_ensemble.cards_per_member(small, small_args) == 1
