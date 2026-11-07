"""One scorer per run, and the shared-input rules of the parallel scorer (hex global on 1-2 cards,
2026-10-16: the forecast hook and `woof hex global --stage score` each scored the same run, on one
process, over 30 min for a 24 h run)."""
import json
import os

from woof.hex import observation_scorecard as scorecard


def _queued_run(tmp_path, monkeypatch, config_text):
    run = tmp_path / "forecast"
    run.mkdir()
    (run / "cuda-history.2026-10-10_00.00.00.nc").write_bytes(b"history inventory fixture")
    init = tmp_path / "init.nc"
    init.write_bytes(b"init fixture")
    config = tmp_path / "observation-scorecard.config.json"
    config.write_text(config_text)
    monkeypatch.setenv("WOOF_HEX_OBSERVATION_CONFIG", str(config))
    queued = []
    monkeypatch.setattr(scorecard, "queue_retry", lambda run, config: queued.append(config))
    scorecard.forecast_hook(run, init=init, grid=tmp_path / "grid.nc", start="2026-10-10_00:00:00")
    assert len(queued) == 1 and queued[0]["region"]["west"] == -180.0  # the door's config, not CONUS
    root = run / "observation-scorecard"
    (root / "automatic-retry.json").write_text(json.dumps(queued[0]))
    return run, config, root


def test_score_stage_joins_the_forecast_scorer_instead_of_scoring_again(tmp_path, monkeypatch):
    run, config, root = _queued_run(tmp_path, monkeypatch, json.dumps({"region": {"west": -180.0, "south": -90.0,
                                                                                  "east": 180.0, "north": 90.0}}))
    (root / "automatic-retry-receipt.json").write_text(json.dumps({"status": "PARTIALLY_MEASURED"}))
    joined = scorecard.join_automatic_scorer(run, config)
    assert joined is not None and joined["joined"] == "automatic"
    # A still-running scorer whose last lead's observations are in: reported at once, never waited for, so
    # the door does not hold its card through CPU scoring (2026-10-24: ~15 min on the lease).
    (root / "automatic-retry-receipt.json").write_text(json.dumps({"status": "QUEUED", "pid": os.getpid()}))
    calls = []
    monkeypatch.setattr(scorecard.time, "sleep", lambda seconds: calls.append(seconds))
    joined = scorecard.join_automatic_scorer(run, config)
    assert calls == [] and joined["status"] == "QUEUED" and joined["pid"] == os.getpid()
    assert "no card is held" in joined["note"]


def test_score_stage_scores_itself_for_a_new_config_or_a_dead_scorer(tmp_path, monkeypatch):
    run, config, root = _queued_run(tmp_path, monkeypatch, json.dumps({"region": {"west": -180.0, "south": -90.0,
                                                                                  "east": 180.0, "north": 90.0}}))
    (root / "automatic-retry-receipt.json").write_text(json.dumps({"status": "QUEUED", "pid": 2 ** 22 + 12345}))
    assert scorecard.join_automatic_scorer(run, config) is None  # its waiter is gone
    config.write_text(json.dumps({"region": {"west": -10.0, "south": 30.0, "east": 30.0, "north": 60.0}}))
    (root / "automatic-retry-receipt.json").write_text(json.dumps({"status": "MEASURED"}))
    assert scorecard.join_automatic_scorer(run, config) is None  # an explicit rescore with other settings


def test_shared_inputs_are_fetched_once_across_threads(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    commands = scorecard.Commands(tmp_path, 10, 60)
    calls = []

    def fetch():
        calls.append(1)
        scorecard.time.sleep(0.05)
        return "object"
    with ThreadPoolExecutor(8) as pool:
        got = list(pool.map(lambda _: commands.once(("reference", "same"), fetch), range(8)))
    assert got == ["object"] * 8 and calls == [1]


def test_door_without_a_forecast_hook_queues_cpu_scoring(tmp_path, monkeypatch):
    run = tmp_path / 'forecast'
    config = tmp_path / 'config.json'
    config.write_text('{}')
    monkeypatch.setattr(scorecard, 'join_automatic_scorer', lambda *a: None)
    monkeypatch.setattr(scorecard, 'completed_scorecard', lambda *a: None)
    monkeypatch.setattr(scorecard, 'discover', lambda *a: ({'init': 'init', 'grid': 'grid', 'start': 'clock'}, []))
    calls = []
    def hook(output, **kwargs):
        calls.append((output, kwargs))
        return {'status': 'QUEUED', 'path': 'scorecard.json'}
    monkeypatch.setattr(scorecard, 'forecast_hook', hook)
    assert scorecard.door_scorecard(run, config)['status'] == 'QUEUED'
    assert calls == [(run, {'init': 'init', 'grid': 'grid', 'start': 'clock', 'config_path': config})]


def test_future_card_without_a_retry_plan_is_returned_at_once(tmp_path, monkeypatch):
    run, config, root = _queued_run(tmp_path, monkeypatch, json.dumps({'region': {'west': -180.0}}))
    (root / 'automatic-retry.json').unlink()
    (root / 'automatic-retry-receipt.json').write_text(json.dumps({'status': 'NOT_YET_OBSERVABLE'}))
    (root / 'scorecard.json').write_text(json.dumps({'status': 'NOT_YET_OBSERVABLE'}))
    assert scorecard.join_automatic_scorer(run, config)['status'] == 'NOT_YET_OBSERVABLE'
