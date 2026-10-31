"""CPU-only protocol tests for the two physics routes' disposable radar columns."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from woof.hex import forecast_start as radar


class RadarStreamTests(unittest.TestCase):
    def test_slow_slot_wait_is_inactivity_and_dead_consumer_releases(self):
        import json
        import time
        for progressing in (True, False):
            with self.subTest(progressing=progressing), tempfile.TemporaryDirectory() as d:
                out = Path(d)
                scratch = out/'.radar-scratch'
                scratch.mkdir()
                for i in range(2):
                    (scratch/f'radar{i}.nc').write_bytes(b'columns')
                clock = [0]
                def tick(_):
                    clock[0] += 100
                    if progressing:
                        (out/'radar-progress.json').write_text(json.dumps({'sequence': clock[0]}))
                        if clock[0] >= 2400:
                            (scratch/'radar0.nc').unlink()
                with patch.object(time, 'monotonic', side_effect=lambda: clock[0]), patch.object(time, 'sleep', side_effect=tick):
                    self.assertEqual(radar.radar_slot(out), progressing)
                if progressing:
                    self.assertEqual(clock[0], 2400)  # Healthy work may exceed the former 20-minute cap.
                    self.assertFalse((out/'radar-error.json').exists())
                else:
                    self.assertIn('no progress', (out/'radar-error.json').read_text())

    def test_isolated_encoder_wait_does_not_hold_capture_or_close(self):
        from concurrent.futures import Future
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {radar.RADAR_MINUTES_ENV: '5'}):
            out = Path(d)
            stream = radar.RadarSnapshots(out, {'start_time': '2026-10-30_00:00:00'}, 120)
            stream.pending = Future()  # Encoder never completes.
            child = Mock()
            child.poll.return_value = None
            stream.writer = Mock(_child=child)
            radar.radar_error(out, 'radar consumer made no progress')
            source = Mock()
            stream.capture_from(3, source, Mock(), {})
            source.assert_not_called()
            self.assertTrue(stream.failed)
            child.kill.assert_called_once()
            child.poll.return_value = 0
            stream.close()
            stream.writer.close.assert_called_once()

    def test_pending_encoder_observes_product_failure_instead_of_retrying_inline(self):
        from concurrent.futures import Future
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {radar.RADAR_MINUTES_ENV: '5'}):
            out = Path(d)
            future = Future()
            writer = Mock()
            writer.submit.return_value = future
            child = Mock()
            child.poll.return_value = None
            writer._child = child
            proof = Mock()
            with patch('woof.hex.history_writer.open_writer', return_value=writer):
                stream = radar.RadarSnapshots(out, {'start_time': '2026-10-30_00:00:00'}, 120)
                stream.capture_from(3, lambda: {}, proof, {})
                radar.radar_error(out, 'radar consumer made no progress')
                stream.close()
            writer.result.assert_not_called()
            proof.write_snapshot_netcdf.assert_not_called()
            child.kill.assert_called_once()
            self.assertTrue(stream.failed)

    def test_cadence_and_member_selection(self):
        with patch.dict(os.environ, {radar.RADAR_MINUTES_ENV: '5', 'WOOF_HEX_RADAR_MEMBER': 'mem000'}):
            self.assertEqual(radar.radar_minutes(Path('/run/ensemble/g00/mem000')), 5)
            self.assertIsNone(radar.radar_minutes(Path('/run/ensemble/g00/mem001')))
            self.assertEqual(radar.radar_minutes(Path('/run/forecast')), 5)

    def test_mpas_stream_only_required_columns_and_history_unchanged(self):
        names = ('xtime', 'theta', 'pressure', 'qv', 'qr', 'qc', 'qi', 'qs', 'qg',
                 'w', 'uReconstructZonal', 'uReconstructMeridional', 'smois', 'swdown')
        layout = SimpleNamespace(fields=[{'name': n} for n in names])
        def stream(*args, **kwargs):
            return SimpleNamespace(name=args[0], fields=args[5], interval=args[4], filename=args[2])
        original = (SimpleNamespace(name='output'),)
        with patch.dict(os.environ, {radar.RADAR_MINUTES_ENV: '10'}):
            result = radar.radar_streams(original, layout, Path('/run/forecast'), stream)
        self.assertIs(result[0], original[0])
        self.assertEqual(result[1].interval, '0_00:10:00')
        self.assertNotIn('smois', result[1].fields)
        self.assertNotIn('swdown', result[1].fields)
        self.assertTrue({'theta', 'pressure', 'qv', 'qr', 'w', 'uReconstructZonal', 'uReconstructMeridional'} <= set(result[1].fields))
        self.assertTrue(result[1].filename.startswith('.radar-scratch/'))

    def test_history_time_reuses_history_and_resets_radar_alarm(self):
        selected = [SimpleNamespace(name='output'), SimpleNamespace(name='radar')]
        alarms = Mock()
        self.assertEqual([s.name for s in radar.radar_selected(selected, alarms, 'time')], ['output'])
        alarms.reset.assert_called_once_with('radar', 'time')
        self.assertEqual(radar.radar_selected(selected[1:], Mock(), 'time'), selected[1:])

    def test_marks_on_60s_steps_and_first_committed_120s_step(self):
        # No timestep selection is changed. Each row is the actual state time.
        for dt, expected in [(60, [(5, [300]), (10, [600])]),
                             (120, [(3, [300]), (5, [600])])]:
            hits = [(step, radar.radar_marks(step, dt, 5)) for step in range(1, int(600/dt)+1)
                    if radar.radar_marks(step, dt, 5)]
            self.assertEqual(hits, expected)
            with tempfile.TemporaryDirectory() as d:
                for step, _ in hits:
                    radar.radar_clock(Path(d), step, dt, '2026-10-30_00:00:00', 5)
                rows = [__import__('json').loads(line) for line in (Path(d)/'radar-clock.jsonl').read_text().splitlines()]
                self.assertEqual(rows[0]['requestedMarks'], ['2026-10-30T00:05:00Z'])
                self.assertEqual(rows[0]['validTime'], '2026-10-30T00:05:00Z' if dt == 60 else '2026-10-30T00:06:00Z')
                self.assertEqual(rows[1]['validTime'], '2026-10-30T00:10:00Z')

    def test_native_alarm_uses_same_late_step_times_without_drift(self):
        from datetime import datetime, timedelta
        from woof.hex.streams_v842 import Stream, StreamAlarms
        start = datetime(2026, 10, 30)
        stream = Stream('radar', {'type': 'output', 'output_interval': '0_00:05:00'}, ())
        for dt, expected in [(60, [0, 300, 600]), (120, [0, 360, 600])]:
            alarms = StreamAlarms({'radar': stream}, start, start, start+timedelta(minutes=10))
            actual = []
            for seconds in range(0, 601, dt):
                when = start+timedelta(seconds=seconds)
                if alarms.due('radar', when):
                    actual.append(seconds)
                    alarms.reset('radar', when)
            self.assertEqual(actual, expected)

    def test_real_mpas_registry_has_computed_radar_columns(self):
        from woof.hex.mpasphys import resolve_namelist
        from woof.hex.mpasphys.forecast_v842 import Stream
        from woof.hex.mpasphys.framework_backend_v842 import registry_source
        from woof.hex.mpasphys.pool_v842 import RegistryLayout
        from woof.hex.mpasphys.resident_output_diagnostics_v842 import PRODUCED
        from woof.hex.mpasphys.streams_output_v842 import output_inventory
        text = '&physics\nconfig_physics_suite="convection_permitting"\n/'
        layout = RegistryLayout(registry_source(), resolve_namelist(text), text,
            dict(nCells=2, nEdges=3, nVertices=2, maxEdges=6, maxEdges2=12, vertexDegree=3, nVertLevels=55))
        with patch.dict(os.environ, {radar.RADAR_MINUTES_ENV: '5'}):
            streams = radar.radar_streams((), layout, Path('/run'), Stream)
        self.assertEqual(len(streams), 1)
        self.assertEqual(output_inventory(streams, layout, produced=PRODUCED)['radar'], [])

    def test_scratch_does_not_advance_history_file_ids(self):
        from woof.hex.mpasphys.streams_output_v842 import PhysicsStreams
        from woof.hex.mpasphys.stream_metadata_v842 import NativeFileIds
        writer = object.__new__(PhysicsStreams)
        writer._file_attributes = {}
        writer.attrs = {}
        writer.contract = SimpleNamespace(stream_file_attributes=lambda stream: {})
        writer.file_ids = NativeFileIds(seed=42)
        reference = NativeFileIds(seed=42)
        writer._bind_file_attributes([(SimpleNamespace(name='output'), Path('h0'), 0),
                                     (SimpleNamespace(name='radar'), Path('r1'), 0),
                                     (SimpleNamespace(name='output'), Path('h2'), 0)])
        self.assertEqual(writer._file_attributes[Path('h0')]['file_id'], reference.next())
        self.assertEqual(writer._file_attributes[Path('h2')]['file_id'], reference.next())

    def test_arwen_capture_only_columns_uses_committed_fields_and_background_writer(self):
        array = np.ones((2, 3), dtype=np.float32)
        prepare = SimpleNamespace(prepare_mpas_to_phys_cuda_v841=Mock(return_value=SimpleNamespace(
            u_p=array*2, v_p=array*3, th_p=array*300, pres_p=array*90000)))
        stack = {'driver': SimpleNamespace(atmosphere=SimpleNamespace(state=SimpleNamespace(scalars=[array, array*0.01]),
            saved=SimpleNamespace(vertical_velocity=array*4)), cache=None), 'backend': SimpleNamespace(take_history_refl10cm=lambda: array*35),
            'prep_geometry': None}
        proof = SimpleNamespace(write_snapshot_netcdf=Mock())
        writer = Mock()
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {radar.RADAR_MINUTES_ENV: '5'}), \
             patch.dict('sys.modules', {'woof.hex.cuda_physics_prep_v841': prepare}), \
             patch('woof.hex.history_writer.open_writer', return_value=writer), patch.object(radar, 'radar_ready') as ready:
            stream = radar.RadarSnapshots(Path(d), {'start_time': '2026-10-30_00:00:00'}, 60)
            self.assertTrue(stream.due(5))
            self.assertFalse(stream.due(4))
            stream.capture(5, stack, {'scalar_names': ['qv', 'qr']}, proof, {})
            stream.close()
            snapshot = writer.submit.call_args.args[1]
            self.assertEqual(set(snapshot['arrays']), {'qv', 'qr', 'u_zonal', 'v_meridional', 'theta', 'pressure', 'w', 'refl10cm'})
            self.assertEqual(snapshot['valid_time'], '2026-10-30_00:05:00')
            np.testing.assert_array_equal(snapshot['arrays']['u_zonal'], array*2)
            ready.assert_called_once()

    def test_missing_columns_and_capture_failure_name_radar_only(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {radar.RADAR_MINUTES_ENV: '5'}):
            out = Path(d)
            self.assertEqual(radar.radar_streams((), SimpleNamespace(fields=[]), out, Mock()), ())
            self.assertIn('MPAS radar columns missing', (out/'radar-error.json').read_text())
            stream = radar.RadarSnapshots(out, {'start_time': '2026-10-30_00:00:00'}, 60)
            stream.capture(5, {}, {}, Mock(), {})
            stream.close()
            self.assertTrue(stream.failed)
            self.assertEqual(__import__('json').loads((out/'radar-error.json').read_text())['product'], 'simulated-radar')


if __name__ == '__main__':
    unittest.main()
