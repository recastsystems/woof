"""The forecast's start record, ``forecast-start.json``, written before the first step.

Which mesh and init a forecast runs on, its start, length and history cadence,
so a reader drawing each history hour as it lands (a site publishing a forecast
while it runs, ``woof hex add-nest`` joining a live parent) needs nothing but
the output directory, even when the door derived the init (an analysis bundle's
background, a remap onto a nest).

One writer for both physics routes: the ArWen-physics driver
(``drivers/run_cuda_v841_forecast``) and the MPAS-physics route
(``mpasphys/forecast_v842``).  THE BREAKAGE A SECOND WRITER WOULD BRING BACK:
until 2026-10-24 only the WOOF driver wrote it, so on the MPAS route
``add-nest`` had no live parent record and fell back to a receipt argv that a
bare ``woof hex forecast`` run does not have.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

START_RECORD_NAME = "forecast-start.json"
START_RECORD_SCHEMA = "gpuwm-hex.forecast-start/1"


def write_start_record(output_root: Path, *, paths: Mapping[str, Any], schedule: Mapping[str, Any]) -> Path:
    record = {"schema": START_RECORD_SCHEMA,
              "grid": str(Path(paths["grid"]).resolve()),
              "static": None if paths.get("static") is None else str(Path(paths["static"]).resolve()),
              "init": str(Path(paths["init"]).resolve()),
              "start_time": schedule.get("start_time"),
              # The run's length and history cadence are written at the START, so a
              # door that joins a live run (woof hex add-nest) reads them here
              # instead of from a receipt that is complete only after the run ends.
              "forecast_hours": schedule.get("forecast_hours"),
              "history_every_minutes": schedule.get("history_every_minutes"),
              "history_from_minutes": schedule.get("history_from_minutes")}
    target = Path(output_root) / START_RECORD_NAME
    partial = target.with_name(target.name + ".partial")
    partial.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(target)
    return target


# WOOF's radar consumer never changes the history cadence. Both physics routes
# expose disposable, column-only frames between histories through this protocol.
RADAR_STREAM_VERSION = 1
RADAR_MINUTES_ENV = "WOOF_HEX_RADAR_MINUTES"


def radar_minutes(output: Path) -> float | None:
    import os
    import re
    value = os.environ.get(RADAR_MINUTES_ENV)
    if not value:
        return None
    member = os.environ.get("WOOF_HEX_RADAR_MEMBER")
    identities = [p for p in Path(output).parts if re.fullmatch(r"(?:mem\d+|member-\d+)", p)]
    if member and identities and member not in identities:
        return None
    minutes = float(value)
    if minutes not in (5, 10):
        raise ValueError("simulated radar cadence must be 5 or 10 minutes")
    return minutes


def radar_streams(streams, layout, output, stream_type):
    """MPAS pools: select only the columns used by rw_mpas_convert full.

    zgrid and mesh coordinates come from --init and --mesh. Wind reconstruction
    stays earth relative. No land, radiation, restart or display fields are copied.
    """
    minutes = radar_minutes(output)
    if minutes is None:
        return streams
    active = {r['name'] for r in layout.fields}
    active.update(c for r in layout.fields for c in r.get('components', ()))
    fields = ('xtime', 'theta', 'pressure', 'qv', 'qr', 'qc', 'qi', 'qs', 'qg',
              'w', 'uReconstructZonal', 'uReconstructMeridional')
    required = {'theta', 'pressure', 'qv', 'qr', 'w', 'uReconstructZonal', 'uReconstructMeridional'}
    if not required <= active:
        radar_error(output, 'MPAS radar columns missing: ' + ', '.join(sorted(required - active)))
        return streams
    interval = f"0_00:{int(minutes):02d}:00"
    return (*streams, stream_type('radar', ('output',),
            '.radar-scratch/radar.$Y-$M-$D_$h.$m.$s.nc', 'none', interval,
            tuple(f for f in fields if f in active), filename_interval='output_interval'))


def radar_error(output, message):
    path = Path(output) / 'radar-error.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'product': 'simulated-radar', 'error': str(message)}) + '\n')


def radar_wait(output, ready):
    """A bounded wait for radar work, released by isolated product failure.

    The worker advances radar-progress.json only on changed output bytes or
    completed uploads. Slow work can outlast 20 minutes; a missing consumer
    still times out after 20 minutes without progress. No heartbeat or PID
    keeps this wait alive. Both scratch slots and WOOF encoding use it.
    """
    import time
    output = Path(output)
    began = time.monotonic()
    previous = None
    while not ready():
        if (output / 'radar-error.json').exists():
            return False
        try:
            progress = (output / 'radar-progress.json').read_text()
        except OSError:
            progress = None
        if progress != previous:
            previous, began = progress, time.monotonic()
        if time.monotonic() - began > 1200:
            radar_error(output, 'radar consumer made no progress for 20 minutes')
            return False
        time.sleep(0.1)
    return not (output / 'radar-error.json').exists()


def radar_slot(output):
    """At most two column frames await consumption; the forecast pays the wait."""
    return radar_wait(output, lambda: len(list((Path(output) / '.radar-scratch').glob('*.nc'))) < 2)


def radar_ready(output, path, step):
    from .history_writer import mark_closed_frame_durable
    mark_closed_frame_durable(path)
    with (Path(output) / 'radar-ready.jsonl').open('a', encoding='utf-8') as ready:
        ready.write(json.dumps({'step': int(step), 'path': str(path)}) + '\n')


def radar_selected(selected, alarms, when):
    """A history at the same time is the radar scene; avoid a duplicate scratch."""
    if any(s.name == 'output' for s in selected):
        for s in selected:
            if s.name == 'radar':
                alarms.reset(s.name, when)
        return [s for s in selected if s.name != 'radar']
    return selected


class RadarSnapshots:
    """ArWen-physics column capture, with encoding on a separate CPU thread.

    Uses the same prepared pressure, dry theta and reconstructed winds as history.
    Only moisture, winds, temperature, pressure and model reflectivity cross to
    host. The normal history writer and its capture schedule remain unchanged.
    """
    def __init__(self, output, schedule, dt, stride=1):
        from concurrent.futures import ThreadPoolExecutor
        self.output, self.schedule, self.dt = Path(output), schedule, float(dt)
        self.minutes = radar_minutes(output)
        self.stride = stride
        self.pool = ThreadPoolExecutor(max_workers=1) if self.minutes else None
        self.pending = None
        self.writer = None
        self.failed = False

    def due(self, step):
        if not self.minutes or self.failed:
            return False
        return bool(radar_marks(step, self.dt, self.minutes, self.stride))

    def note_history(self, step):
        if self.due(step):
            radar_clock(self.output, step, self.dt, self.schedule['start_time'], self.minutes,
                        stride=self.stride, history=True)

    @staticmethod
    def columns(stack, host):
        from .cuda_physics_prep_v841 import prepare_mpas_to_phys_cuda_v841
        driver, backend = stack['driver'], stack['backend']
        prepared = prepare_mpas_to_phys_cuda_v841(driver.atmosphere,
            scalar_names=tuple(host['scalar_names']), geometry=stack['prep_geometry'],
            kernel_cache=driver.cache, post_rk_wsm6=False)
        def download(a):
            import numpy as np
            return a.get() if hasattr(a, 'get') else np.array(a, copy=True)
        arrays = {name: download(driver.atmosphere.state.scalars[i])
                  for i, name in enumerate(host['scalar_names'])
                  if name in ('qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'qh')}
        arrays.update({name: download(value) for name, value in {
            'u_zonal': prepared.u_p, 'v_meridional': prepared.v_p,
            'theta': prepared.th_p, 'pressure': prepared.pres_p,
            'w': driver.atmosphere.saved.vertical_velocity}.items()})
        refl = backend.take_history_refl10cm()
        if refl is not None:
            arrays['refl10cm'] = download(refl)
        return arrays

    def capture(self, step, stack, host, proof, static):
        self.capture_from(step, lambda: self.columns(stack, host), proof, static)

    def capture_from(self, step, source, proof, static):
        try:
            if self.pending is not None:
                if not radar_wait(self.output, self.pending.done):
                    self.failed = True
                    self.stop_writer()
                    return
                self.pending.result()
            if not radar_slot(self.output):
                self.failed = True
                return
            arrays = source()
            from datetime import datetime, timedelta
            valid = (datetime.strptime(self.schedule['start_time'], '%Y-%m-%d_%H:%M:%S')
                     + timedelta(seconds=step * self.dt)).strftime('%Y-%m-%d_%H:%M:%S')
            target = self.output / '.radar-scratch' / ('radar.' + valid.replace(':', '.') + '.nc')
            target.parent.mkdir(parents=True, exist_ok=True)
            snapshot = {'arrays': arrays, 'valid_time': valid, 'receipt': {'step': step}}
            from .history_writer import open_writer
            if self.writer is None:
                self.writer = open_writer(proof.write_snapshot_netcdf, max_pending=1)
            pending = self.writer.submit(target, snapshot, static, defer_fallback=True) if self.writer is not None else None
            def write():
                try:
                    if self.writer is None:
                        proof.write_snapshot_netcdf(target, snapshot, static)
                    else:
                        if not radar_wait(self.output, pending.done):
                            self.failed = True
                            return
                        # A broken radar encoder is isolated, never retried on
                        # a model or waiter thread through history's fallback.
                        from .history_writer import _FALLBACK
                        if pending.result() is _FALLBACK:
                            raise RuntimeError('radar background writer did not complete its frame')
                        self.writer.result(pending)
                    radar_ready(self.output, target, step)
                    radar_clock(self.output, step, self.dt, self.schedule['start_time'], self.minutes,
                                stride=self.stride)
                except Exception as error:
                    self.failed = True
                    radar_error(self.output, error)
            self.pending = self.pool.submit(write)
        except Exception as error:
            self.failed = True
            radar_error(self.output, error)

    def stop_writer(self):
        # This writer belongs exclusively to disposable radar columns. Ending
        # it releases its pump without touching the forecast history writer.
        child = getattr(self.writer, '_child', None)
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()

    def close(self):
        if self.pending is not None and not radar_wait(self.output, self.pending.done):
            self.failed = True
        if self.failed or (self.output / 'radar-error.json').exists():
            self.stop_writer()
        if self.pool is not None:
            self.pool.shutdown(wait=True)
        if self.writer is not None:
            self.writer.close()



def radar_marks(step, dt, minutes, stride=1):
    """Marks crossed by this committed step, without changing the model clock.

    At dt=120 s the 5-min mark is sampled at 06:00, and the 10-min mark
    at 10:00. An LTS executor calls at shared macro boundaries (stride=top).
    """
    import math
    if minutes is None:
        return []
    every = float(minutes) * 60
    seconds = step * float(dt)
    if step == 0:
        return [0.0]
    first = max(1, math.floor((step - stride) * float(dt) / every + 1e-10) + 1)
    last = math.floor(seconds / every + 1e-10)
    return [index * every for index in range(first, last + 1)]


def radar_clock(output, step, dt, start, minutes, *, stride=1, history=False):
    from datetime import datetime, timedelta
    marks = radar_marks(step, dt, minutes, stride)
    if not marks:
        return
    base = datetime.strptime(start, '%Y-%m-%d_%H:%M:%S')
    valid = (base + timedelta(seconds=step * float(dt))).strftime('%Y-%m-%dT%H:%M:%SZ')
    row = {'step': int(step), 'modelSeconds': step * float(dt), 'validTime': valid,
           'requestedMarks': [(base + timedelta(seconds=m)).strftime('%Y-%m-%dT%H:%M:%SZ') for m in marks],
           'source': 'history' if history else 'radar scratch'}
    with (Path(output) / 'radar-clock.jsonl').open('a', encoding='utf-8') as receipt:
        receipt.write(json.dumps(row) + '\n')
