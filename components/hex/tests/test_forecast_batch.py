"""The member batch's roster law, card pairing and coordinator/worker protocol (CPU only)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from woof.hex import forecast_batch
from woof.hex.forecast_batch import BatchRefusal, granted_cards, load_roster


def _member(tmp_path, name, **extra):
    argv = ["--mesh", "nowhere", "--grid", str(tmp_path / "absent.grid.nc"),
            "--static", str(tmp_path / "absent.static.nc"), "--init", str(tmp_path / f"{name}.init.nc"),
            "--out", str(tmp_path / name / "fc"), "--scratch", str(tmp_path / name / "scratch")]
    record = {"run_id": name, "argv": argv, "log": str(tmp_path / name / "forecast.log"),
              "result": str(tmp_path / name / "result.json")}
    record.update(extra)
    return record


def _roster(tmp_path, members, **extra):
    path = tmp_path / "roster.json"
    path.write_text(json.dumps({"schema": forecast_batch.SCHEMA, "members": members, **extra}))
    return path


def test_roster_refuses_shared_destinations_and_door_modes(tmp_path):
    one = _member(tmp_path, "mem001")
    two = _member(tmp_path, "mem002")
    two["argv"][two["argv"].index("--out") + 1] = one["argv"][one["argv"].index("--out") + 1]
    with pytest.raises(BatchRefusal, match="share"):
        load_roster(_roster(tmp_path, [one, two]))
    with pytest.raises(BatchRefusal, match="no --init"):
        bare = _member(tmp_path, "mem003")
        index = bare["argv"].index("--init")
        del bare["argv"][index:index + 2]
        load_roster(_roster(tmp_path, [bare]))
    with pytest.raises(BatchRefusal, match="--preflight"):
        load_roster(_roster(tmp_path, [_member(tmp_path, "mem004", argv=_member(tmp_path, "mem004")["argv"] + ["--preflight"])]))
    with pytest.raises(BatchRefusal, match="subcommand"):
        load_roster(_roster(tmp_path, [_member(tmp_path, "mem005", argv=["forecast"] + _member(tmp_path, "mem005")["argv"])]))
    with pytest.raises(BatchRefusal, match="twice"):
        load_roster(_roster(tmp_path, [_member(tmp_path, "mem006"), dict(_member(tmp_path, "mem007"), run_id="mem006")]))


def test_cards_pair_the_grant_with_the_visible_devices():
    assert granted_cards({"GPU_MUTEX_CARDS": "2,3", "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"}) == [("2", "GPU-a"), ("3", "GPU-b")]
    assert granted_cards({"CUDA_VISIBLE_DEVICES": "1"}) == [("1", "1")]
    assert granted_cards({}) == [("0", "0")]
    with pytest.raises(BatchRefusal):
        granted_cards({"GPU_MUTEX_CARDS": "2,3", "CUDA_VISIBLE_DEVICES": "GPU-a"})


def test_every_member_runs_once_and_reports_its_own_result(tmp_path, monkeypatch):
    """Two cards x two workers over five members whose door refuses at once (no mesh files):
    each member is run exactly once, writes its own log and result, and the receipt says failed."""

    members = [_member(tmp_path, f"mem{index:03d}") for index in range(1, 6)]
    roster = _roster(tmp_path, members, workers_per_card=2)
    receipt_path = tmp_path / "batch-receipt.json"
    src = Path(forecast_batch.__file__).resolve().parents[1]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(src), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("GPU_MUTEX_CARDS", "5,6")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-x,GPU-y")
    rc = forecast_batch.run_batch(roster, receipt_path)
    receipt = json.loads(receipt_path.read_text())
    assert rc == 1 and receipt["status"] == "failed"
    assert sorted(row["run_id"] for row in receipt["members"]) == [m["run_id"] for m in members]
    assert sorted(name for worker in receipt["workers"] for name in worker["members"]) == [m["run_id"] for m in members]
    assert {worker["card"] for worker in receipt["workers"]} == {"5", "6"}
    for member, row in zip(members, receipt["members"]):
        assert row["returncode"] != 0
        assert json.loads(Path(member["result"]).read_text())["returncode"] == row["returncode"]
        assert Path(member["log"]).read_text(), "the member's door output lands in its own log"


def test_memory_refused_member_moves_its_destinations_aside(tmp_path):
    member = _member(tmp_path, "mem009")
    out = Path(member["argv"][member["argv"].index("--out") + 1])
    scratch = Path(member["argv"][member["argv"].index("--scratch") + 1])
    out.mkdir(parents=True)
    (out / "timeline.jsonl").write_text("{}\n")
    moved = forecast_batch._move_destinations_aside(member, 1)
    assert moved == [str(out) + ".memory-refused-1"]
    assert not out.exists() and not scratch.exists()
    assert (Path(str(out) + ".memory-refused-1") / "timeline.jsonl").exists()


# ---------------------------------------------------------------- tail dispatch and card release
# Measured on a rented 4-card host, 2026-10-14, a free ensemble batch (4 cards x 3 slots, 31 members, the 5 h member
# first): when each slot asked for its next member, in seconds after the batch began, from its batch receipt (round 1 by the 20 s stagger,
# then the done_unix of each slot's previous member).  The 5 h member on slot 0.0 ended at 1502.11 s.
_MEASURED_ASKS = [
    (0.32, "0.0"), (0.33, "1.0"), (0.33, "2.0"), (0.34, "3.0"), (20.31, "0.1"), (20.32, "1.1"), (20.33, "2.1"),
    (20.33, "3.1"), (40.31, "0.2"), (40.31, "1.2"), (40.31, "2.2"), (40.32, "3.2"),
    (652.83, "1.0"), (660.83, "2.0"), (670.29, "0.1"), (675.23, "2.1"), (677.94, "3.0"), (679.57, "3.1"),
    (683.03, "1.1"), (688.76, "3.2"), (691.45, "1.2"), (693.46, "2.2"), (694.31, "0.2"),
    (1317.02, "2.0"), (1324.08, "3.0"), (1327.82, "0.1"), (1342.14, "2.1"), (1344.84, "1.1"), (1345.88, "1.0"),
    (1355.79, "2.2"), (1358.79, "0.2"), (1368.90, "1.2"), (1374.51, "3.1"), (1380.30, "3.2"), (1502.11, "0.0")]


def _replay(asks, members, cards, per_card, wait_seconds=None):
    """Drive _Dispatch through a timeline of slot requests on a fake clock.  A slot's request also ends its
    previous member; the requesting slot polls first, then every deferring slot polls again (as the real slots do
    on their 1 s re-check).  Returns ([(t, card, member index)] of every start, the dispatch)."""

    now = [0.0]
    dispatch = forecast_batch._Dispatch(members, cards, per_card, wait_seconds=wait_seconds, clock=lambda: now[0])
    running, waiting, starts = {}, {}, []
    for t, slot in asks:
        now[0] = t
        if slot in running:
            card, token, began = running.pop(slot)
            dispatch.finished(card, token, t - began)
        for name in [slot] + [other for other in waiting if other != slot]:
            card = name.split(".")[0]
            with dispatch.cond:
                verdict, index, token = dispatch.poll(card, t - waiting[name] if name in waiting else None)
            if verdict == "start":
                running[name] = (card, token, t)
                starts.append((t, card, index))
                waiting.pop(name, None)
            elif verdict == "wait":
                waiting.setdefault(name, t)
            else:
                waiting.pop(name, None)
    return starts, dispatch


def _per_card_since(starts, since):
    counts = {}
    for t, card, _ in starts:
        if t >= since:
            counts[card] = counts.get(card, 0) + 1
    return counts


def test_tail_spreads_the_last_members_over_the_cards_measured_replay():
    starts, dispatch = _replay(_MEASURED_ASKS, 31, ["0", "1", "2", "3"], 3)
    assert sorted(index for _, _, index in starts) == list(range(31)), "every member starts exactly once"
    assert _per_card_since(starts, 1317.0) == {"0": 2, "1": 2, "2": 2, "3": 2}
    summary = dispatch.summary()
    assert summary["tail_members"] == 11 and summary["deferred_starts"] == len(summary["deferred_seconds"]) >= 1
    assert 0 < max(summary["deferred_seconds"]) < summary["tail_wait_seconds"]


def test_tail_wait_zero_is_the_old_order_and_reproduces_the_measured_pileup():
    """The knob that turns spreading off gives exactly what the measured batch did: card 2 took three of the last
    eight, card 3 one."""

    starts, _ = _replay(_MEASURED_ASKS, 31, ["0", "1", "2", "3"], 3, wait_seconds=0)
    assert _per_card_since(starts, 1317.0) == {"0": 2, "1": 2, "2": 3, "3": 1}


def test_bulk_never_defers():
    """While at least as many members are queued as live slots, every slot takes one at once, however uneven."""

    dispatch = forecast_batch._Dispatch(20, ["0", "1"], 2, clock=lambda: 0.0)
    with dispatch.cond:
        verdicts = [dispatch.poll("0")[0] for _ in range(2)] + [dispatch.poll("1")[0]]
    assert verdicts == ["start", "start", "start"] and dispatch.tail_began is None


def test_tail_deferral_is_bounded():
    """A free slot defers to a lighter card that will have a slot soon, but only up to the wait limit: then it
    starts the member anyway, so a card whose members run past the estimate cannot stall the tail."""

    now = [0.0]
    dispatch = forecast_batch._Dispatch(5, ["0", "1"], 2, wait_seconds=30.0, clock=lambda: now[0])
    with dispatch.cond:
        _, _, a = dispatch.poll("0")          # bulk
        _, _, b = dispatch.poll("1")          # bulk
        dispatch.poll("1")                    # 3 queued for 4 slots: the tail; card 0 is not lighter, so no wait
    now[0] = 100.0
    dispatch.finished("0", a, 100.0)          # the median member wall is now 100 s
    with dispatch.cond:
        assert dispatch.poll("0")[0] == "start"       # card 1's two members have run 100 s: leaving, not load
        now[0] = 120.0
        dispatch.finished("1", b)                     # card 1 now has a free slot and one member about to leave
        assert dispatch.poll("0")[0] == "wait"        # card 0 carries one fresh member, card 1 none
        assert dispatch.poll("0", 29.0)[0] == "wait"
        assert dispatch.poll("0", 30.0)[0] == "start"
    assert dispatch.summary()["deferred_seconds"] == [30.0]


def _fake_worker(tmp_path):
    path = tmp_path / "fake_worker.py"
    path.write_text(
        "import json, os, sys, time\n"
        "for line in sys.stdin:\n"
        "    job = json.loads(line)\n"
        "    if job.get('exit'):\n"
        "        break\n"
        "    began = time.time()\n"
        "    time.sleep(job.get('sleep', 0.05))\n"
        "    if job.get('until'):\n"
        "        path, lines = job['until']\n"
        "        deadline = time.time() + 60\n"
        "        while time.time() < deadline and not (os.path.exists(path) and\n"
        "                                               len(open(path).read().splitlines()) >= lines):\n"
        "            time.sleep(0.02)\n"
        "    if job.get('die'):\n"
        "        os._exit(3)\n"
        "    os.makedirs(os.path.dirname(job['result']), exist_ok=True)\n"
        "    open(job['result'], 'w').write(json.dumps({'returncode': 0, 'batch': True}))\n"
        "    print(json.dumps({'run_id': job['run_id'], 'returncode': 0, 'began_unix': began,\n"
        "                      'wall_seconds': time.time() - began, 'card_env': os.environ['GPU_MUTEX_CARDS']}),\n"
        "          flush=True)\n")
    return path


def _fake_release(tmp_path, returncode):
    log = tmp_path / "release.log"
    path = tmp_path / "fake_release.py"
    path.write_text(
        "import sys, time\n"
        f"open({str(log)!r}, 'a').write(sys.argv[1] + ' ' + repr(time.time()) + '\\n')\n"
        f"print(('ok: card ' if {returncode} == 0 else 'refused: card ') + sys.argv[1])\n"
        f"sys.exit({returncode})\n")
    return path, log


def _run_fake_batch(tmp_path, monkeypatch, members, cards, per_card, release=0, **roster):
    """run_batch with stand-in slot processes and a stand-in queue release (None: no GPU_MUTEX_RELEASE)."""

    worker = _fake_worker(tmp_path)
    monkeypatch.setattr(forecast_batch, "_worker_command", lambda: [sys.executable, str(worker)])
    log = None
    if release is None:
        monkeypatch.setattr(forecast_batch, "_release_argv", lambda card: None)
    else:
        script, log = _fake_release(tmp_path, release)
        monkeypatch.setattr(forecast_batch, "_release_argv", lambda card: [sys.executable, str(script), str(card)])
    monkeypatch.setenv("GPU_MUTEX_CARDS", ",".join(cards))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(f"GPU-{card}" for card in cards))
    receipt_path = tmp_path / "batch-receipt.json"
    rc = forecast_batch.run_batch(_roster(tmp_path, members, workers_per_card=per_card, **roster), receipt_path)
    receipt = json.loads(receipt_path.read_text())
    calls = [line.split() for line in log.read_text().splitlines()] if log is not None and log.exists() else []
    return rc, receipt, calls


def test_idle_cards_go_back_to_the_queue_while_another_card_still_works(tmp_path, monkeypatch):
    # The long member runs until both other cards have been handed back (their release log lines), not for a fixed
    # 2.5 s: at host load ~55 (hex 0.5.0 sealing pass) the short members' process starts outlasted that sleep.
    until = [str(tmp_path / "release.log"), 2]
    members = ([_member(tmp_path, "long", sleep=0.2, until=until)] +
               [_member(tmp_path, f"m{i:02d}", sleep=0.2) for i in range(7)])
    rc, receipt, calls = _run_fake_batch(tmp_path, monkeypatch, members, ["4", "5", "6"], 2)
    rows = {row["run_id"]: row for row in receipt["members"]}
    assert rc == 0 and receipt["status"] == "passed" and receipt["card_release"] == "on"
    assert all(row["card_env"] == row["card"] for row in rows.values()), "each slot runs on its own card"
    long_card, long_done = rows["long"]["card"], rows["long"]["done_unix"]
    others = sorted({"4", "5", "6"} - {long_card})
    assert sorted(card for card, _ in calls) == others, "every card but the long member's, once each"
    assert all(float(unix) < long_done for _, unix in calls), "handed back while the long member still ran"
    assert sorted(row["card"] for row in receipt["released_cards"]) == others
    assert all(row["released"] and row["message"].startswith("ok") for row in receipt["released_cards"])


def test_without_the_queue_helper_nothing_is_released(tmp_path, monkeypatch):
    members = [_member(tmp_path, "long", sleep=1.0)] + [_member(tmp_path, f"m{i}", sleep=0.1) for i in range(3)]
    rc, receipt, calls = _run_fake_batch(tmp_path, monkeypatch, members, ["0", "1"], 1, release=None)
    assert rc == 0 and calls == [] and receipt["released_cards"] == []
    assert receipt["card_release"].startswith("off: GPU_MUTEX_RELEASE is not set")


def test_the_roster_can_keep_every_card_to_the_end(tmp_path, monkeypatch):
    members = [_member(tmp_path, "long", sleep=1.0), _member(tmp_path, "short", sleep=0.1)]
    rc, receipt, calls = _run_fake_batch(tmp_path, monkeypatch, members, ["0", "1"], 1, release_idle_cards=False)
    assert rc == 0 and calls == [] and receipt["card_release"].startswith("off: the roster")


def test_a_refused_release_is_recorded_and_the_batch_still_passes(tmp_path, monkeypatch):
    members = [_member(tmp_path, "long", sleep=1.5), _member(tmp_path, "short", sleep=0.1)]
    rc, receipt, calls = _run_fake_batch(tmp_path, monkeypatch, members, ["0", "1"], 1, release=1)
    assert rc == 0 and receipt["status"] == "passed" and len(calls) == 1
    (row,) = receipt["released_cards"]
    assert row["released"] is False and row["message"].startswith("refused")


def test_a_card_with_a_dead_slot_keeps_its_grant(tmp_path, monkeypatch):
    """A slot that died may leave a member process on its card, so that card is never handed back early."""

    members = [_member(tmp_path, "long", sleep=2.0), _member(tmp_path, "dies", sleep=0.1, die=True),
               _member(tmp_path, "short", sleep=0.1)]
    rc, receipt, calls = _run_fake_batch(tmp_path, monkeypatch, members, ["0", "1", "2"], 1)
    rows = {row["run_id"]: row for row in receipt["members"]}
    assert rc == 1 and rows["dies"]["returncode"] == 1
    assert [card for card, _ in calls] == [rows["short"]["card"]]
