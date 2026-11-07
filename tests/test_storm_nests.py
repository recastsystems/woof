"""[storm_nests]: the authoring door for storm-following nests.

CPU only.  The table expands into dormant [[domain]] slots the existing
lifecycle machinery runs; these tests pin the expansion on the packaged HRRR
v4 recipe (the headline shape), the loader's acceptance of what it writes,
the planner's per-slot price, the merge retirement and the spawn cadence.
"""

from __future__ import annotations

import tomllib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof import storm_nests

RECIPE = Path(__file__).resolve().parents[1] / "woof" / "data" / "recipes" / "hrrr_v4_gsd41.toml"


def _recipe(extra: str = "", run_seconds: float = 21600.0) -> dict:
    text = RECIPE.read_text(encoding="utf-8")
    raw = tomllib.loads(text + "\n" + extra)
    raw["experiment"]["start_time"] = datetime(2024, 4, 27, 18)
    raw["experiment"]["run_seconds"] = run_seconds
    return raw


def test_hrrr_recipe_expands_to_one_km_dormant_slots_that_load():
    raw = _recipe('[storm_nests]\nslots = 3\npoints = [{lat = 41.25, lon = -96.0, at = "2024-04-27T19:00:00"}]\n')
    doc, receipt = storm_nests.expand(raw)
    assert "storm_nests" not in doc
    slots = doc["domain"][1:]
    assert [d["grid_id"] for d in slots] == [2, 3, 4, 5]
    assert receipt["shape"]["ratio"] == 3 and receipt["shape"]["child_n"] == 240
    # The on-command slot fires at its time, the others watch 2-5 km UH.
    assert slots[0]["spawn"] == {"trigger": "time", "at_s": 3600.0}
    for slot in slots[1:]:
        assert slot["spawn"]["trigger"] == "uh"
        assert slot["spawn"]["threshold"] == 75.0
        assert slot["spawn"]["cadence_s"] == 900.0
        assert slot["rearm"]["max_firings"] == 4
    for slot in slots:
        assert slot["follow"]["field"] == "uh"
        assert slot["retire"]["merge_overlap"] == 0.5
        assert slot["sf_lake_physics"] == 0 and slot["gwd_opt"] == 0
    # The follower's echo fallback needs the parent's history at its cadence.
    history = float(doc["domain"][0]["history_interval_s"])
    assert 900.0 % history == 0.0
    changed = receipt["parent_history_interval"]
    assert changed is None or changed["to_s"] == 900.0

    from woof.domain_wizard import _render_table, experiment_from_text
    from woof.experiment import dormant_domain_ids
    from woof.runtime import _spawn_leg_seconds
    from woof.static.corridor import planned_corridor

    text = storm_nests._emit(doc, _render_table)
    exp = experiment_from_text(text, source="storm-nests-test.toml")
    assert dormant_domain_ids(exp) == (2, 3, 4, 5)
    assert _spawn_leg_seconds(exp) == 900.0
    plan = planned_corridor(exp, exp.domains[1])
    assert plan.reach["whole_frame"] is True
    root = doc["domain"][0]
    assert plan.window == (0, 0, int(root["nx"]) * 3, int(root["ny"]) * 3)


def test_fixed_nests_survive_and_slots_beside_them_are_refused():
    # A config with its own fixed nests lost them whenever the expansion ran
    # (a 3-domain tree re-stated as 1 domain).
    raw = _recipe()
    root = raw["domain"][0] if isinstance(raw["domain"], list) else raw["domain"]
    nest = {**{k: v for k, v in root.items()
               if k not in ("grid_id", "parent_id", "nx", "ny", "dx", "dy",
                            "i_parent_start", "j_parent_start",
                            "parent_grid_ratio")},
            "grid_id": 2, "parent_id": 1, "parent_grid_ratio": 3,
            "i_parent_start": 100, "j_parent_start": 100, "nx": 240, "ny": 240}
    raw["domain"] = [root, nest]
    doc, receipt = storm_nests.expand(raw, slots_override=0)
    assert [d["grid_id"] for d in doc["domain"]] == [root["grid_id"], 2]
    assert receipt["slot_grid_ids"] == []
    with pytest.raises(storm_nests.StormNestsRefusal, match=r"fixed nest\(s\) \(grid 2\)"):
        storm_nests.expand(raw, slots_override=2)


def test_point_lands_on_its_parent_cell():
    raw = _recipe('[storm_nests]\nslots = 0\npoints = [{lat = 38.5, lon = -97.5, at_s = 1200}]\n')
    doc, _ = storm_nests.expand(raw)
    slot = doc["domain"][1]
    # The HRRR grid is centred on 38.5N 97.5W: the nest is centred there too.
    centre_i = slot["i_parent_start"] + 40
    centre_j = slot["j_parent_start"] + 40
    root = doc["domain"][0]
    assert abs(centre_i - int(root["nx"]) / 2) <= 2
    assert abs(centre_j - int(root["ny"]) / 2) <= 2
    assert slot["spawn"] == {"trigger": "time", "at_s": 1200.0}


def test_unknown_keys_and_bad_values_refuse_by_name():
    with pytest.raises(storm_nests.StormNestsRefusal, match="nest_size"):
        storm_nests.expand(_recipe("[storm_nests]\nnest_size = 3\n"))
    with pytest.raises(storm_nests.StormNestsRefusal, match="dx_km"):
        storm_nests.expand(_recipe("[storm_nests]\ndx_km = 2.0\n"))
    with pytest.raises(storm_nests.StormNestsRefusal, match="outside"):
        storm_nests.expand(_recipe('[storm_nests]\npoints = [{lat = 40, lon = -95, at_s = 99999}]\n'))
    assert storm_nests.parse_point("41.2,-96.0@2024-04-27T19:00") == {
        "lat": 41.2, "lon": -96.0, "at": "2024-04-27T19:00"}
    assert storm_nests.parse_point("41.2,-96.0@600") == {"lat": 41.2, "lon": -96.0, "at_s": 600.0}


def test_auto_slots_are_priced_from_the_card():
    raw = _recipe('[storm_nests]\nslots = "auto"\nmax_slots = 6\n')
    _doc, receipt = storm_nests.expand(raw, free_bytes=96 * 2**30)
    pricing = receipt["pricing"]
    assert pricing["slot_bytes"] > 0
    assert receipt["automatic_slots"] == pricing["slots"]
    assert 1 <= pricing["slots"] <= 6
    with pytest.raises(storm_nests.StormNestsRefusal, match="free memory"):
        storm_nests.expand(raw)


def _child(gid, i, j, n=240, r=3, parent=1):
    return SimpleNamespace(grid_id=gid, parent_id=parent, i_parent_start=i,
                           j_parent_start=j, parent_grid_ratio=r,
                           run=SimpleNamespace(nx=n, ny=n), nx=n, ny=n)


def test_merged_storms_retire_the_younger_nest():
    from woof.core.nest_lifecycle import RetireConfig, merged_retirements

    retire = RetireConfig(trigger="uh", threshold=25.0, merge_overlap=0.5)
    live = {2: (_child(2, 100, 100), retire),
            3: (_child(3, 120, 110), retire),
            4: (_child(4, 400, 400), retire)}
    rows = merged_retirements(live, birth_times={2: 900.0, 3: 1800.0, 4: 900.0}, t=3600.0)
    assert [(r["grid_id"], r["merged_into"], r["stop_reason"]) for r in rows] == [(3, 2, "merged")]
    assert rows[0]["overlap_fraction"] == pytest.approx((60 * 70) / (80 * 80))
    # Off unless asked: a slot without merge_overlap never merges.
    plain = RetireConfig(trigger="uh", threshold=25.0)
    assert merged_retirements({2: (_child(2, 100, 100), plain), 3: (_child(3, 101, 101), plain)},
                              birth_times={}, t=0.0) == []
    # The spawn runner hands its RetirementWatch objects (the 6 h CONUS run died on
    # "'RetirementWatch' object has no attribute 'merge_overlap'" at the first merge check).
    from woof.core.nest_lifecycle import RetirementWatch
    watched = {g: (cfg, RetirementWatch(r)) for g, (cfg, r) in live.items()}
    assert merged_retirements(watched, birth_times={2: 900.0, 3: 1800.0, 4: 900.0},
                              t=3600.0) == rows
    with pytest.raises(ValueError, match="merge_overlap"):
        RetireConfig(trigger="uh", threshold=25.0, merge_overlap=1.5)


def test_spawn_cadence_sets_the_leg_and_is_refused_on_a_time_trigger():
    from woof.core.nest_spawn import SpawnConfig

    cfg = SpawnConfig(trigger="uh", threshold=75.0, earliest_s=0.0,
                      latest_s=3600.0, cadence_s=600.0)
    assert cfg.to_json()["cadence_s"] == 600.0
    assert "cadence_s" not in SpawnConfig(trigger="uh", threshold=75.0, earliest_s=0.0,
                                          latest_s=3600.0).to_json()
    with pytest.raises(ValueError, match="cadence_s"):
        SpawnConfig(trigger="time", at_s=60.0, cadence_s=60.0)


def test_go_front_door_re_states_the_config(tmp_path):
    from woof.go_cli import _at_flag_storm_nests

    text = RECIPE.read_text(encoding="utf-8").replace(
        "start_time = 2026-10-02T21:00:00", "start_time = 2024-04-27T18:00:00")
    config = tmp_path / "hrrr.toml"
    config.write_text(text, encoding="utf-8")
    args = SimpleNamespace(config=config, storm_nests="2", nest_at=["41.2,-96.0@1200"],
                           outdir=tmp_path / "out")
    _at_flag_storm_nests(args)
    raw = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    assert [d["grid_id"] for d in raw["domain"]] == [1, 2, 3, 4]
    assert (Path(args.config).parent / "storm-nests.json").is_file()
    # Without the table or a flag the config is untouched.
    untouched = SimpleNamespace(config=config, storm_nests=None, nest_at=None, outdir=None)
    _at_flag_storm_nests(untouched)
    assert untouched.config == config


def test_inland_daily_mean_takes_each_time_on_its_own_grid():
    """An analysis initial state on its own grid ahead of a boundary model's
    times (HRRR f00 + RAP boundaries): the initial time's inland-water air
    temperature is its own T2, the boundary times' the mean of theirs."""
    import numpy as np

    from woof.ingest import horiz

    field = horiz._INLAND_AIR_FIELD
    t0 = datetime(2024, 4, 27, 18)
    from datetime import timedelta

    series = [SimpleNamespace(valid_time=t0 + timedelta(hours=h),
                              latitude=np.arange(3), longitude=np.arange(4), projection=None,
                              fields={field: np.full((3, 4), 280.0 + h),
                                      "SKINTEMP": np.zeros((3, 4))})
              for h in range(3)]
    provider = horiz.DailyMeanAirTemperature(series)
    same, receipt = provider(series[0])
    assert np.allclose(same, 281.0) and "3 forcing time" in receipt["window"]
    analysis = SimpleNamespace(valid_time=t0, latitude=np.arange(5),
                               longitude=np.arange(6), projection=None, fields={
        field: np.full((5, 6), 290.0), "SKINTEMP": np.zeros((5, 6))})
    own, receipt = provider(analysis)
    assert own.shape == (5, 6) and np.allclose(own, 290.0)
    assert "own 2 m air temperature" in receipt["window"]


def test_a_spawned_nest_keeps_the_declared_tick():
    """THE DEFECT the box C runs hit (2026-10-21): the startup tree (root
    only) resolved its clock to 1 tick per second, the tree with a ratio-3
    nest needs 3, and the leg walk carried integer ticks across, so a nest
    born at 1800 s read model second 600 (fixed clock: CarrierContractError,
    GLW stale) or never started (adaptive: tick-exact sync violated).  The
    startup clock now takes the declared tree's tick."""
    from woof.core.clock import resolve_clock
    from woof.domain_wizard import experiment_from_text
    from woof.experiment import pre_spawn_experiment

    doc, _ = storm_nests.expand(_recipe("[storm_nests]\nslots = 1\n"))
    exp = experiment_from_text(storm_nests._emit(doc), source="tick.toml")
    startup = pre_spawn_experiment(exp)
    declared = resolve_clock(exp, lbc_interval_s=3600).tick_den
    assert resolve_clock(startup, lbc_interval_s=3600).tick_den != declared
    assert resolve_clock(startup, lbc_interval_s=3600,
                         tick_den_floor=declared).tick_den == declared
    import inspect

    from woof.core import model
    from woof import prepared_domain_tree_forecast as tree
    from woof import runtime
    assert "tick_den_floor=" in inspect.getsource(model.build_experiment)
    assert "tick_den_floor=" in inspect.getsource(tree.run_prepared_tree)
    assert "tick_den_floor=previous_den" in inspect.getsource(runtime._retarget_tree_schedule)


def test_a_nest_takes_its_parents_adaptive_step_over_its_ratio():
    """THE DEFECT (box D, 2026-10-21): HRRR's [shared] pins the adaptive
    clock's starting/min/max steps at 20 s, those are per-domain keys, and a
    1 km nest stepped at the parent's 20 s and went non-finite within eight
    steps.  The loader now gives a child that sets no bound its parent's over
    parent_time_step_ratio; a child that sets its own keeps it."""
    from woof.domain_wizard import experiment_from_text

    doc, _ = storm_nests.expand(_recipe("[storm_nests]\nslots = 1\n"))
    exp = experiment_from_text(storm_nests._emit(doc), source="adaptive.toml")
    child = exp.domains[1].run
    for name in ("starting_time_step", "max_time_step", "min_time_step"):
        assert (getattr(child, name), getattr(child, f"{name}_den")) == (20, 3)
    doc["domain"][1]["max_time_step"] = 10
    doc["domain"][1]["max_time_step_den"] = 1
    exp = experiment_from_text(storm_nests._emit(doc), source="adaptive.toml")
    assert (exp.domains[1].run.max_time_step, exp.domains[1].run.max_time_step_den) == (10, 1)
    assert (exp.domains[1].run.min_time_step, exp.domains[1].run.min_time_step_den) == (20, 3)



def test_slots_sharing_whole_frame_ground_link_one_corridor_cache(tmp_path):
    """Every dormant slot of a CONUS 1 km whole frame wrote and hashed its own
    10 GB corridor on one core (box D, 2026-10-21). A slot whose corridor is
    another's ground links that cache: same bytes, same digest, one inode."""
    import os

    import numpy as np

    from woof.static.corridor import CorridorBuild, write_statics_corridor_set

    fields = {"HGT_M": np.arange(12.0).reshape(3, 4)}
    builds = [CorridorBuild(grid_id=2, fields=fields, entry={"corridor_nx": 4}),
              CorridorBuild(grid_id=3, fields=fields,
                            entry={"corridor_nx": 4, "shared_ground_with": "d02"})]
    receipt = write_statics_corridor_set(tmp_path / "set", builds)
    d02, d03 = (receipt["domains"][k]["cache"] for k in ("d02", "d03"))
    assert d03 == {**d02, "path": "d03.npz"}
    assert os.stat(tmp_path / "set" / "d02.npz").st_ino == os.stat(tmp_path / "set" / "d03.npz").st_ino
    assert np.load(tmp_path / "set" / "d03.npz")["HGT_M"].tolist() == fields["HGT_M"].tolist()


def test_nest_lives_reads_the_relocation_runners_moves_and_the_run_end(tmp_path):
    """The 1 h door run (box D, 2026-10-21) said "0 moves, stopped 2700 s" for
    two nests that moved and lived to 3600 s: the summary looked for keys the
    relocation runner does not write and stamped the last spawn boundary."""
    import json

    from woof.nest_lives import nest_lives

    (tmp_path / "spawn_receipts.jsonl").write_text("\n".join(json.dumps(row) for row in (
        {"elapsed_seconds": 900.0, "born": [{"grid_id": 2, "episode": 1, "placement": [742, 362]}]},
        {"elapsed_seconds": 2700.0},
    )) + "\n", encoding="utf-8")
    (tmp_path / "relocation_receipts.d02.json").write_text(json.dumps({"receipts": [
        {"event": "relocated", "elapsed_seconds": 1800.0, "grid_id": 2,
         "placement_from": {"i_parent_start": 742, "j_parent_start": 362},
         "placement_to": {"i_parent_start": 734, "j_parent_start": 370}},
        {"event": "held", "elapsed_seconds": 2700.0, "grid_id": 2},
    ]}), encoding="utf-8")
    (life,) = nest_lives(tmp_path, run_end_s=3600.0)
    assert [(row["event"], row["placement"]) for row in life["track"]] == [
        ("spawn", [742, 362]), ("move", [734, 370])]
    assert (life["stop_s"], life["stop_reason"]) == (3600.0, "run-ended")


def test_a_rearmed_slot_keeps_its_earlier_lifes_move_receipts(tmp_path):
    """The 6 h storm-nest run (box D, 2026-10-21) lost d03's first life of
    moves: its second life's fresh runner rewrote relocation_receipts.d03.json
    with only its own rows."""
    import json
    from types import SimpleNamespace

    from woof.core.relocation_runner import RelocationRunner, RelocationRunnerCollection

    path = tmp_path / "relocation_receipts.d03.json"

    def runner(rows):
        return SimpleNamespace(
            config=SimpleNamespace(grid_id=3, receipt=lambda: {"grid_id": 3}),
            receipts=list(rows), receipts_path=path, provider=None)

    first = runner([{"event": "relocated", "elapsed_seconds": 2700.0}])
    collection = RelocationRunnerCollection()
    collection.runners[3] = first
    collection.drop_absent(SimpleNamespace(nodes_by_grid_id={}))
    second = runner([])
    collection.merge_from(second)
    assert collection.runners[3] is second
    RelocationRunner._record(second, SimpleNamespace(_relocation_runner=None),
                             {"event": "relocated", "elapsed_seconds": 9900.0})
    rows = json.loads(path.read_text(encoding="utf-8"))["receipts"]
    assert [row["elapsed_seconds"] for row in rows] == [2700.0, 9900.0]
