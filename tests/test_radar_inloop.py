"""CPU-only lifecycle checks. Run directly; no CUDA import or model execution."""
import ast
import json
import os
import subprocess
import time
from fractions import Fraction
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from woof import simulated_radar as radar


class FakeWriter:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.paths = []
        self.selections = []

    def submit(self, path, valid_time, state, **kwargs):
        value = state.value if state is not None else kwargs["frame"]["value"]
        Path(path).write_text(str(value))
        self.paths.append(Path(path))
        self.selections.append(kwargs["history_selection"]["history_vars"])
        self.landing_observer(domain=self.grid_id, valid_time=valid_time, path=path)

    def drain(self):
        pass

    def close(self):
        pass

    def update_global_attrs(self, attrs):
        self.global_attrs = attrs


class InloopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.when = datetime(2026, 10, 29)
        self.node = SimpleNamespace(cfg=SimpleNamespace(grid_id=1, quicklook_output=object()), state=SimpleNamespace(value=1))
        self.template = SimpleNamespace(nx=3, ny=2, nz=4, dx=3000, dy=3000, title="test", soil_layers=4,
                                        global_attrs={"SIMULATION_START_DATE": "2026-10-29_00:00:00"}, _device=None)
        selection = SimpleNamespace(from_mapping=lambda value: value)
        modules = {
            "woof.io.wrfout": SimpleNamespace(AsyncDomainWrfoutWriter=FakeWriter,
                wrfout_filename=lambda time, grid: f"wrfout_d{grid:02d}_{time:%Y-%m-%d_%H_%M_%S}"),
            "woof.io.history_selection": SimpleNamespace(HISTORY_VOCABULARY=set(radar.RADAR_SNAPSHOT_FIELDS), HistorySelection=selection),
        }
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict(sys.modules, modules).start()
        mock.patch.object(radar, "history_missing_columns", return_value=()).start()

    def live(self, native, timing="history"):
        result = radar.LiveSimulatedRadar(radar.SimulatedRadarOptions.from_mapping({"timing": timing}), self.root, runner=native)
        result._render_snapshot = lambda path: self.assertTrue(Path(path).is_file())
        return result

    def test_snapshots_are_private_frozen_and_removed_after_scan(self):
        values = []
        with self.live(lambda paths, **kw: values.extend(Path(p).read_text() for p in paths)) as live:
            live.capture(self.node, self.when, self.template, {})
            self.node.state.value = 2
            live.capture(self.node, self.when + timedelta(minutes=5), self.template, {})
            live.drain()
        self.assertEqual(values, ["1", "2"])
        self.assertEqual(list(self.root.glob("wrfout*")), [])
        self.assertEqual(list((self.root / ".radar-scratch").rglob("wrfout*")), [])
        self.assertEqual(len(live._scratch_deleted), 2)

    def test_quicklook_keeps_only_radar_columns_and_two_moments(self):
        configs = []
        with self.live(lambda paths, **kw: configs.append(kw["config"])) as live:
            live.capture(self.node, self.when, self.template, {}, quicklook=True)
            live.drain()
        fields = live._snapshot_writers[1].selections[0]
        self.assertEqual(set(fields), {"REFL_10CM", "T", "PH", "PHB", "U", "V", "W"})
        self.assertEqual(configs[0]["fields"], ["reflectivity", "velocity"])
        # VCP 212's three lowest elevations: the drawn 0.5/0.9 deg PPIs are unchanged, the scan is 3 cuts, not 17.
        self.assertEqual(configs[0]["scan_strategy"], "low_tilts")
        self.assertEqual(list(configs[0]["elevations_deg"]), [0.5, 0.9, 1.3])
        radar.SimulatedRadarOptions.from_mapping(configs[0])  # the native request still validates

    def test_quicklook_keeps_a_custom_ladder_or_scan_timing_as_requested(self):
        for request in ({"scan_strategy": "custom", "elevations_deg": [0.5, 2.0, 4.0]}, {"timing": "scan"}):
            config = radar.SimulatedRadarOptions.from_mapping(request).to_mapping()
            self.assertEqual(radar.quicklook_scan(config), {})

    def test_full_frame_keeps_hydrometeors_and_original_request(self):
        configs = []
        with self.live(lambda paths, **kw: configs.append(kw["config"])) as live:
            live.capture(self.node, self.when, self.template, {}, quicklook=False)
            live.drain()
        self.assertTrue({"QRAIN", "QCLOUD", "QICE", "QSNOW", "QGRAUP", "QVAPOR"} <=
                        set(live._snapshot_writers[1].selections[0]))
        self.assertEqual(configs[0]["fields"], "auto")
        self.assertEqual(configs[0]["scan_strategy"], "vcp212")

    def test_streamed_snapshot_uses_current_store(self):
        values = []
        self.node.state._streamed_domain = SimpleNamespace(history_fields=lambda: {"value": 9})
        with self.live(lambda paths, **kw: values.append(Path(paths[0]).read_text())) as live:
            live.capture(self.node, self.when, self.template, {})
        self.assertEqual(values, ["9"])

    def test_three_slots_backpressure_includes_queued_and_active_snapshots(self):
        entered, release = threading.Event(), threading.Event()
        def native(paths, **kw):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test scan timed out")
        live = self.live(native)
        finished = threading.Event()
        failures = []
        try:
            for index in range(3):
                live.capture(self.node, self.when + timedelta(minutes=5 * index), self.template, {})
            self.assertTrue(entered.wait(2))
            def fourth():
                try:
                    live.capture(self.node, self.when + timedelta(minutes=15), self.template, {})
                except BaseException as error:
                    failures.append(error)
                finally:
                    finished.set()
            thread = threading.Thread(target=fourth)
            thread.start()
            self.assertFalse(finished.wait(.05))
            self.assertEqual(len(live._owned_snapshots), 3)
            release.set()
            thread.join(3)
            self.assertTrue(finished.is_set())
            self.assertEqual(failures, [])
        finally:
            release.set()
            live.close()

    def test_duplicate_capture_never_removes_an_earlier_inflight_snapshot(self):
        entered, release = threading.Event(), threading.Event()
        def native(paths, **kw):
            entered.set()
            self.assertTrue(release.wait(3))
            self.assertEqual(Path(paths[0]).read_text(), "1")
        live = self.live(native)
        try:
            live.capture(self.node, self.when, self.template, {})
            self.assertTrue(entered.wait(2))
            with self.assertRaisesRegex(RuntimeError, "duplicate radar snapshot"):
                live.capture(self.node, self.when, self.template, {})
            self.assertEqual(len(live._owned_snapshots), 1)
        finally:
            release.set()
            live.close()

    def test_scan_pair_keeps_successor_until_next_scan_and_flushes_it_at_close(self):
        calls = []
        def native(paths, **kw):
            self.assertTrue(all(Path(p).exists() for p in paths))
            calls.append(len(paths))
        with self.live(native, "scan") as live:
            for index in range(3):
                live.capture(self.node, self.when + timedelta(minutes=5 * index), self.template, {})
            live.drain()
            self.assertEqual(len(live._owned_snapshots), 1)
        self.assertEqual(calls, [2, 2, 1])
        self.assertEqual(len(live._scratch_deleted), 3)

    def test_failure_releases_all_private_files(self):
        def native(paths, **kw):
            raise RuntimeError("native scan failure")
        live = self.live(native)
        live.capture(self.node, self.when, self.template, {})
        with self.assertRaisesRegex(RuntimeError, "native scan failure"):
            live.close()
        self.assertEqual(live._owned_snapshots, {})

    def test_writer_hook_uses_private_capture_only_when_selected(self):
        source = (Path(__file__).resolve().parents[1] / "woof/io/wrfout.py").read_text()
        tree = ast.parse(source)
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PerDomainWrfoutWriters")
        function = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_capture_radar")
        scope = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "writer-hook", "exec"), scope)
        worker = SimpleNamespace(_simulated_radar=SimpleNamespace(capture=mock.Mock()), _writers={1: self.template}, _metadata_by_grid_id={1: {}}, _write_beat=lambda *args: nullcontext())
        with mock.patch.dict("os.environ", {"WOOF_RADAR_INLOOP": "1"}):
            scope["_capture_radar"](worker, self.node, self.when, None)
        worker._simulated_radar.capture.assert_called_once()

        with mock.patch.dict("os.environ", {"WOOF_RADAR_INLOOP": "0"}):
            scope["_capture_radar"](worker, self.node, self.when, None)
        worker._simulated_radar.capture.assert_called_once()

    def test_submit_classifies_quicklook_from_the_actual_history_clock(self):
        source = (Path(__file__).resolve().parents[1] / "woof/io/wrfout.py").read_text()
        tree = ast.parse(source)
        selector = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "quicklook_selection")
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PerDomainWrfoutWriters")
        submit = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_submit")
        prefix = []
        for statement in submit.body:
            prefix.append(statement)
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) and getattr(statement.value.func, "attr", "") == "_capture_radar":
                break
        submit.body = prefix
        scope = {"Fraction": Fraction, "timedelta": timedelta}
        exec(compile(ast.Module(body=[selector, submit], type_ignores=[]), "history-clock", "exec"), scope)
        worker = SimpleNamespace(_simulated_radar=None, start_time=self.when, _capture_radar=mock.Mock())
        self.node.cfg.full_history_interval_s = 3600
        self.node.clock = SimpleNamespace(tick_den=100, spec=SimpleNamespace(start_ticks=0, history_begin_ticks=0))
        for seconds, quick in ((0, False), (300, True), (3600, False), (3900, True)):
            scope["_submit"](worker, self.node, seconds * 100)
            self.assertEqual(worker._capture_radar.call_args.kwargs["quicklook"], quick)

    def test_slow_radar_outlasts_real_forecast_stall_bound_without_losing_frames(self):
        from woof.forecast_supervisor import ForecastWatchdog
        from woof.supervisor import RuntimeHeartbeat
        # Virtual seconds make each 0.4 s fake scan exceed the real 120 s step bound.
        started = time.monotonic()
        watch = ForecastWatchdog([sys.executable, "-m", "unused", "--outdir", str(self.root)],
                                 clock=lambda: (time.monotonic() - started) * 500)
        heartbeat = RuntimeHeartbeat(watch.path, run_id=watch.run_id, config_sha256=watch.digest,
                                     started_at_utc=watch.started_at)
        heartbeat(model_elapsed_seconds=300, outer_step=2, last_durable_wrfout=None, last_checkpoint=None)
        errors, scanned, stop = [], [], threading.Event()
        def supervise():
            while not stop.wait(.005):
                error = watch.check(os.getpid())
                if error:
                    errors.append(error)
        thread = threading.Thread(target=supervise)
        def native(paths, **kw):
            time.sleep(.4)
            scanned.extend(paths)
        live = radar.LiveSimulatedRadar(radar.SimulatedRadarOptions.from_mapping({}), self.root,
            runner=native, idle_timeout_seconds=1, progress_interval_seconds=.01, isolate_failures=True)
        live._render_snapshot = lambda path: None
        thread.start()
        try:
            for index in range(5):
                live.capture(self.node, self.when + timedelta(minutes=index * 5), self.template, {}, quicklook=True)
                heartbeat(model_elapsed_seconds=(index + 2) * 300, outer_step=index + 3,
                          last_durable_wrfout=None, last_checkpoint=None)
            live.close()
        finally:
            stop.set()
            thread.join(1)
            live.close()
        self.assertEqual(errors, [])
        self.assertEqual(len(scanned), 5)
        self.assertEqual(live._completed_frames, 5)
        self.assertIsNone(live._isolated)
        self.assertFalse((self.root / "radar-failure.json").exists())

    def test_hung_native_process_is_killed_and_only_the_radar_is_isolated(self):
        processes = []
        def native(paths, **kw):
            process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            processes.append(process)
            kw["started"](process)
            if process.wait():
                raise RuntimeError("native process stopped")
        live = radar.LiveSimulatedRadar(radar.SimulatedRadarOptions.from_mapping({}), self.root,
            runner=native, idle_timeout_seconds=.15, progress_interval_seconds=.01, isolate_failures=True)
        live._render_snapshot = lambda path: None
        forecast_steps = []
        try:
            for index in range(6):
                live.capture(self.node, self.when + timedelta(minutes=index * 5), self.template, {}, quicklook=True)
                forecast_steps.append(index)
            live.close()
            self.assertEqual(forecast_steps, list(range(6)))
            self.assertTrue(all(p.poll() is not None for p in processes))
            receipt = json.loads((self.root / "radar-failure.json").read_text())
            self.assertIn("completed no volume", receipt["message"])
            self.assertFalse(live.check())
            self.assertEqual(live._owned_snapshots, {})
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            live.close()

    def test_completed_volumes_extend_the_deadline_inside_a_long_scan(self):
        def native(paths, **kw):
            folder = self.root / "radar"
            folder.mkdir(exist_ok=True)
            volumes = []
            for index in range(6):
                time.sleep(.04)
                volumes.append({"domain": "d01", "site": {"id": str(index)}, "valid_time": "t",
                                "files": [{"sha256": str(index)}]})
                tmp = folder / "manifest.tmp"
                tmp.write_text(json.dumps({"volumes": volumes}))
                tmp.replace(folder / "manifest.json")
        live = radar.LiveSimulatedRadar(radar.SimulatedRadarOptions.from_mapping({}), self.root,
            runner=native, idle_timeout_seconds=.12, progress_interval_seconds=.01, isolate_failures=True)
        live._render_snapshot = lambda path: None
        with live:
            live.capture(self.node, self.when, self.template, {})
        self.assertIsNone(live._isolated)
        self.assertEqual(live._completed_frames, 1)

    def test_maps_use_both_native_products_before_release(self):
        live = radar.LiveSimulatedRadar.__new__(radar.LiveSimulatedRadar)
        live.outdir = self.root / "run"
        path = self.root / "snapshot"
        path.write_text("columns")
        live._owned_snapshots = {path: {"domain": 1, "valid_time": self.when.isoformat()}}
        live._radar_map_frames = []
        live._idle_timeout = 600
        plans = []
        def command(plan, paths):
            plans.append(plan)
            self.assertTrue(paths[0].exists())
            return ["rw-wrfbatch-via-render"]
        with mock.patch.dict(sys.modules, {"woof.go_cli": SimpleNamespace(render_command=command)}), \
                mock.patch.object(radar.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
            live._render_snapshot(path)
        self.assertEqual(plans[0]["render_products"], "composite_reflectivity,1km_reflectivity")
        self.assertTrue((self.root / "radar-png/radar-maps.json").exists())
        self.assertTrue(path.exists())

    def test_native_config_keeps_lossless_cfradial2_and_requested_moments(self):
        options = radar.SimulatedRadarOptions.from_mapping({"formats": ["level2", "cfradial2"],
                                                          "fields": ["reflectivity", "velocity"]})
        self.assertEqual(options.to_mapping()["formats"], ["level2", "cfradial2"])
        self.assertEqual(options.to_mapping()["fields"], ["reflectivity", "velocity"])

    def test_explicit_full_history_cadence_skips_even_full_column_display_frames(self):
        quick = self.root / "quicklook/frame"
        quick.parent.mkdir()
        quick.write_text("user-selected columns")
        full = self.root / "history"
        full.write_text("ordinary history")
        calls = []
        with mock.patch.dict("os.environ", {"WOOF_RADAR_INLOOP": "0"}):
            with self.live(lambda paths, **kw: calls.extend(paths)) as live:
                live.output_committed(domain=1, valid_time=self.when, path=quick)
                live.output_committed(domain=1, valid_time=self.when, path=full)
        self.assertEqual(calls, [str(full.resolve())])
        self.assertTrue(quick.exists())

    def test_ordinary_history_is_never_deleted(self):
        path = self.root / "history"
        path.write_text("keep")
        with self.live(lambda paths, **kw: None) as live:
            live.output_committed(domain=1, valid_time=self.when, path=path)
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
