"""first_look.FirstLook: the f000 maps are drawn once the forecast lists its first frame, never before, and a
forecast that ends without a frame ends the wait; every later frame is drawn the moment it is listed, several draws at
once, and finish() draws what is left before it returns."""
import json
import threading
import time
from pathlib import Path
from unittest import mock

from woof.hex import first_look


class _Draws:
    """A stand-in for subprocess.Popen: records each draw's argv; a draw ends when the test says so (or at once)."""

    def __init__(self, hold: bool = False):
        self.argvs: list[list[str]] = []
        self.hold = hold
        self.ended: set[int] = set()
        self.lock = threading.Lock()

    def __call__(self, argv, **kw):
        with self.lock:
            self.argvs.append(list(argv))
            index = len(self.argvs) - 1
        return mock.Mock(poll=lambda: (0 if (not self.hold or index in self.ended) else None), wait=lambda: 0,
                         terminate=lambda: None)

    def end(self, index):
        with self.lock:
            self.ended.add(index)

    def running(self):
        with self.lock:
            return [i for i in range(len(self.argvs)) if i not in self.ended]


def _histories(argv):
    """The frames a draw's argv names after --history (up to the next flag)."""
    out = []
    for item in argv[argv.index("--history") + 1:]:
        if item.startswith("--"):
            break
        out.append(item)
    return out


def _wait(predicate, seconds=5.0):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    return predicate()


def test_draws_the_first_listed_frame_into_the_full_pass_case(tmp_path):
    forecast = tmp_path / "forecast"
    forecast.mkdir()
    draws = _Draws()
    with mock.patch.object(first_look.subprocess, "Popen", side_effect=draws):
        look = first_look.FirstLook(forecast, grid="g.nc", init="i.nc", start_time="2026-10-23_12:00:00",
                                    out=tmp_path / "products", case="hex-x1.40962-native-2026102312",
                                    extra=["--region", "event=region:48,10,2500,2500"], poll_seconds=0.01)
        look.start()
        time.sleep(0.05)
        assert draws.argvs == []                     # nothing listed yet: nothing drawn
        frame = forecast / "cuda-history.2026-10-23_12.00.00.nc"
        (forecast / first_look.READY).write_text(json.dumps({"step": 0, "path": str(frame)}) + "\n")
        result = look.finish(timeout=5)
    assert result["status"] == "drawn"
    argv = draws.argvs[0]
    assert argv[argv.index("--history") + 1] == str(frame)
    assert argv[argv.index("--case") + 1] == "hex-x1.40962-native-2026102312"
    assert argv[argv.index("--simulation-start") + 1] == "2026-10-23_12:00:00"
    assert "--region" in argv and "--no-points" not in argv   # the first look draws its soundings


def test_a_forecast_without_frames_ends_the_wait(tmp_path):
    forecast = tmp_path / "forecast"
    forecast.mkdir()
    with mock.patch.object(first_look.subprocess, "run") as run, \
            mock.patch.object(first_look.subprocess, "Popen") as popen:
        look = first_look.FirstLook(forecast, grid="g", init="i", start_time="s", out=tmp_path, case="c", poll_seconds=0.01)
        look.start()
        result = look.finish(timeout=5)
    assert result["status"] == "no-frame"
    assert not run.called and not popen.called


def test_a_frame_with_its_durable_sidecar_counts_without_a_ready_list(tmp_path):
    forecast = tmp_path / "forecast"
    forecast.mkdir()
    frame = forecast / "cuda-history.2026-10-23_12.00.00.nc"
    frame.write_bytes(b"x" * 10)
    (forecast / (frame.name + ".durable.json")).write_text(json.dumps({"bytes": 9}))   # not yet whole
    draws = _Draws()
    with mock.patch.object(first_look.subprocess, "Popen", side_effect=draws):
        look = first_look.FirstLook(forecast, grid="g", init="i", start_time="s", out=tmp_path, case="c", poll_seconds=0.01)
        look.start()
        time.sleep(0.05)
        assert draws.argvs == []
        (forecast / (frame.name + ".durable.json")).write_text(json.dumps({"bytes": 10}))
        result = look.finish(timeout=5)
    assert result["status"] == "drawn"
    assert draws.argvs[0][draws.argvs[0].index("--history") + 1] == str(frame)


def test_every_later_frame_is_drawn_while_the_forecast_runs(tmp_path):
    """Production canary 2026-10-24 (hex global 60 km 24 h): f000 came 62.5 s in, f001-f024 only after the forecast."""
    forecast = tmp_path / "forecast"
    forecast.mkdir()
    frames = [forecast / f"cuda-history.2026-10-23_{h:02d}.00.00.nc" for h in range(12, 16)]
    for frame in frames:
        frame.write_bytes(b"frame")
    ready = forecast / first_look.READY
    ready.write_text(json.dumps({"step": 0, "path": str(frames[0])}) + "\n")
    draws = _Draws()
    with mock.patch.object(first_look.subprocess, "Popen", side_effect=draws):
        look = first_look.FirstLook(forecast, grid="g.nc", init="i.nc", start_time="2026-10-23_12:00:00",
                                    out=tmp_path / "products", case="c", poll_seconds=0.01)
        look.start()
        assert _wait(lambda: len(draws.argvs) >= 1)
        with ready.open("a") as stream:
            for step, frame in enumerate(frames[1:3], 1):
                stream.write(json.dumps({"step": step, "path": str(frame)}) + "\n")
        assert _wait(lambda: len(draws.argvs) >= 3)
        with ready.open("a") as stream:
            stream.write(json.dumps({"step": 3, "path": str(frames[3])}) + "\n")
        result = look.finish(timeout=5)
    first, later = draws.argvs[0], draws.argvs[1:]
    assert first[first.index("--history") + 1] == str(frames[0]) and "--no-points" not in first   # f000, soundings
    drawn = [argv[argv.index("--history") + 1:] for argv in later]
    assert drawn == [[str(f)] for f in frames[1:]]                                   # each later frame once, alone
    assert all("--no-points" in argv for argv in later)
    scratches = [argv[argv.index("--scratch") + 1] for argv in draws.argvs]
    assert len(set(scratches)) == len(scratches)                                     # no two draws share a scratch
    assert result["live"]["frames"] == 4 and result["live"]["drawn"] == 4
    assert result["live"]["receipt"] == str(tmp_path / first_look.LIVE_RECEIPT)
    receipt = json.loads((tmp_path / first_look.LIVE_RECEIPT).read_text())
    # f000 (the first look) and every later frame, each with the identity the full pass checks before skipping it
    assert sorted(receipt["frames"]) == sorted(receipt["drawn"]) == sorted(f.name for f in frames)
    assert receipt["stat"][frames[2].name] == [frames[2].stat().st_size, frames[2].stat().st_mtime_ns]
    assert receipt["out"] == str(tmp_path / "products") and receipt["case"] == "c"


def test_frames_are_drawn_side_by_side_up_to_the_slots_and_finish_draws_the_rest(tmp_path, monkeypatch):
    """An Arizona zoom globe on 8 cards (woof hex 0.4.10, 2026-10-30): a frame every 22 s, one draw at a time,
    9 of 37 frames drawn when the forecast ended. Frames now draw several at once, and finish() leaves none behind."""
    monkeypatch.setattr(first_look, "LIVE_SLOTS", "3")
    forecast = tmp_path / "forecast"
    forecast.mkdir()
    frames = [forecast / f"history.2026-10-30_{h:02d}.00.00.nc" for h in range(7)]
    for frame in frames:
        frame.write_bytes(b"frame")
    ready = forecast / first_look.READY
    ready.write_text("".join(json.dumps({"step": s, "path": str(f)}) + "\n" for s, f in enumerate(frames)))
    draws = _Draws(hold=True)
    with mock.patch.object(first_look.subprocess, "Popen", side_effect=draws):
        look = first_look.FirstLook(forecast, grid="g", init="i", start_time="2026-10-30_00:00:00",
                                    out=tmp_path / "products", case="c", poll_seconds=0.01)
        look.start()
        assert _wait(lambda: len(draws.argvs) == 3)
        time.sleep(0.05)
        assert len(draws.argvs) == 3 and len(draws.running()) == 3          # three at once, no more
        draws.end(1)
        assert _wait(lambda: len(draws.argvs) == 4)                          # a slot frees: the next frame starts
        finisher = threading.Thread(target=lambda: look.finish(timeout=10))
        finisher.start()
        while _wait(lambda: draws.running(), 0.2) and finisher.is_alive():
            for index in draws.running():
                draws.end(index)
        finisher.join(10)
    drawn = [_histories(argv) for argv in draws.argvs]
    assert sorted(sum(drawn, [])) == sorted(str(f) for f in frames)      # every frame, each once
    assert look.live["slots"] == 3 and len(look.live["drawn"]) == len(frames)


def test_live_slots_follow_the_hosts_memory_and_cores():
    gib = 1024 ** 3
    frame = 1.5 * gib
    per_draw = first_look.LIVE_DRAW_MEMORY_FACTOR * frame
    # 2 TiB and 192 cores (an 8-card g7e): cores bind
    assert first_look.live_slots(int(frame), available=2048 * gib, cpus=192, setting="auto") == \
        min(int(0.5 * 2048 * gib // per_draw), 192 // first_look.LIVE_DRAW_CORES)
    # little memory: memory binds, never below one
    assert first_look.live_slots(int(frame), available=4 * gib, cpus=192, setting="auto") == 1
    assert first_look.live_slots(0, available=2048 * gib, cpus=192, setting="auto") == 1   # no frame size: one
    assert first_look.live_slots(int(frame), available=2048 * gib, cpus=192, setting="5") == 5


def test_live_frames_off_draws_the_first_frame_only(tmp_path, monkeypatch):
    monkeypatch.setattr(first_look, "LIVE_ON", False)
    forecast = tmp_path / "forecast"
    forecast.mkdir()
    lines = [json.dumps({"step": h, "path": str(forecast / f"cuda-history.2026-10-23_{12 + h:02d}.00.00.nc")}) for h in range(3)]
    (forecast / first_look.READY).write_text("\n".join(lines) + "\n")
    (tmp_path / first_look.LIVE_RECEIPT).write_text("{}")  # an earlier run's: the full pass must not skip by it
    with mock.patch.object(first_look.subprocess, "run", return_value=mock.Mock(returncode=0)) as run, \
            mock.patch.object(first_look.subprocess, "Popen") as popen:
        look = first_look.FirstLook(forecast, grid="g", init="i", start_time="s", out=tmp_path / "p", case="c", poll_seconds=0.01)
        look.start()
        result = look.finish(timeout=5)
    assert result["status"] == "drawn" and run.call_count == 1 and not popen.called
    assert not (tmp_path / first_look.LIVE_RECEIPT).exists()
