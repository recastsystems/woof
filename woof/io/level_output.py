"""Asynchronous auxhist22/23 output through the Rust product-tape writer."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
import json
import math
import numpy as np

from woof.io.wrfout import WrfoutWriter, _NETCDF4_IO_LOCK

_UNITS = dict(U='m s-1', V='m s-1', T='K', RH='%', GHT='m', S='m s-1',
              TD='K', Q='kg/kg', P='Pa', Z='m')
_DESCRIPTIONS = dict(U='U wind', V='V wind', T='Temperature', RH='Relative humidity',
                     GHT='Geopotential Height', S='Speed', TD='Dew point temperature',
                     Q='Mixing ratio', P='Air Pressure', Z='Height')


class LevelTape(WrfoutWriter):
    """Only level fields acquire these dimensions; ordinary history is inert."""
    def _dims_for(self, name, shape):
        if name.endswith(('_PL', '_ZL')):
            dim = 'num_press_levels_stag' if name.endswith('_PL') else 'num_z_levels_stag'
            if len(shape) not in (1, 3) or (len(shape) == 3 and tuple(shape[1:]) != (self.ny, self.nx)):
                raise ValueError('native level output has invalid shape '+repr((name, shape)))
            self._ensure_dimension(dim, shape[0])
            return ('Time', dim, *(('south_north', 'west_east') if len(shape) == 3 else ()))
        return super()._dims_for(name, shape)

    def _create_variable(self, name, dims):
        var = super()._create_variable(name, dims)
        if name.endswith(('_PL', '_ZL')):
            field, suffix = name.rsplit('_', 1)
            var.stagger = 'Z'
            var.units = _UNITS[field]
            label = 'Pressure' if suffix == 'PL' else 'Height'
            desc = 'Pressure' if name == 'P_PL' else _DESCRIPTIONS[field]
            var.description = label+' level data, '+desc
        return var


class _Stream:
    def __init__(self, owner, node, prefix, stream):
        self.owner, self.node, self.prefix, self.stream = owner, node, prefix, stream
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'gpuwm-aux{stream}-d{node.cfg.grid_id}')
        self.pending = []
        self.tape = None
        self.count = 0
        self.paths = []
        self._last_tick = None

    def submit(self, ticks):
        import cupy as cp
        from woof.core.level_output import device_fields
        node, cfg = self.node, self.node.cfg.run
        seconds = ticks / node.clock.tick_den
        epoch = float(getattr(node.state, 'domain_start_offset', 0.) or 0.)
        elapsed = seconds-epoch
        stem = f'auxhist{self.stream}'
        interval = float(getattr(cfg, stem+'_interval_s'))
        begin, end = getattr(cfg, stem+'_begin_s'), getattr(cfg, stem+'_end_s')
        # Native alarms are tested after each step: a nonaligned alarm rings
        # on the first committed state at or beyond it, once per step.
        previous = elapsed if self._last_tick is None else self._last_tick
        self._last_tick = elapsed
        if elapsed < begin or (end and elapsed > end):
            return
        first = previous == elapsed
        due = (first and math.isclose(elapsed, begin)) or (
            math.floor((elapsed-begin)/interval) > math.floor((previous-begin)/interval))
        if not due:
            return
        for future in self.pending:
            if future.done():
                future.result()
        self.pending = [f for f in self.pending if not f.done()]
        if len(self.pending) >= 4:
            # Bounded output backpressure, only at an output alarm.
            self.pending.pop(0).result()
        device = cp.cuda.Device()
        host = {}; held = []
        for name, array in device_fields(node.state, cfg, self.prefix).items():
            if isinstance(array, np.ndarray):
                # The host store is advanced by the next GPU sweep. Own
                # this frame's bytes before letting that sweep reuse it.
                host[name] = array.copy()
                continue
            memory = cp.cuda.alloc_pinned_memory(array.nbytes)
            value = np.frombuffer(memory, dtype=array.dtype, count=array.size).reshape(array.shape)
            array.get(out=value, blocking=False)
            host[name] = value
            held.extend((memory, array))
        # The copies are enqueued before subsequent mutation on the producer
        # stream. Only this writer's worker waits for their completion.
        ready = cp.cuda.Event(); ready.record()
        metadata = self.owner.parent._metadata_by_grid_id[node.cfg.grid_id]
        for name in ('XLAT', 'XLONG'):
            if name in metadata:
                host[name] = np.asarray(metadata[name], np.float32).copy()
        valid = self.owner.parent.start_time + timedelta(seconds=seconds)
        self.pending.append(self.executor.submit(self._write, valid, host, ready, held, device))

    def _write(self, valid, fields, ready, held, device):
        with device:
            ready.synchronize()
        node, cfg = self.node, self.node.cfg.run
        frames = getattr(cfg, f'frames_per_auxhist{self.stream}')
        valid_string = valid.strftime('%Y-%m-%d_%H:%M:%S')
        with _NETCDF4_IO_LOCK:
            if self.tape is None:
                template = getattr(cfg, f'auxhist{self.stream}_outname')
                name = template.replace('<domain>', f'{node.cfg.grid_id:02d}').replace('<date>', valid_string)
                path = Path(name)
                if not path.is_absolute():
                    path = self.owner.parent.output_dir.parent / path
                attrs = self.owner.parent._writers[node.cfg.grid_id].global_attrs
                self.tape = LevelTape(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
                    dx=cfg.dx, dy=cfg.dy, title=self.owner.parent.title,
                    global_attrs=attrs, field_schema=fields)
                self.paths.append(path)
            self.tape.write_frame(valid_string, fields)
            self.count += 1
            if self.count % frames == 0:
                self.tape.close(); self.tape = None

    def close(self):
        try:
            for future in self.pending:
                future.result()
            if self.tape is not None:
                with _NETCDF4_IO_LOCK:
                    self.tape.close(); self.tape = None
        finally:
            self.executor.shutdown(wait=True)

    def drain(self):
        for future in self.pending:
            future.result()
        self.pending.clear()


class LevelOutputWriters:
    def __init__(self, parent, *, restored_checkpoint=False):
        self.parent, self.streams, self.archived = parent, {}, []
        for node in parent.model.walk_parent_first():
            if not (getattr(node.cfg.run, 'p_lev_diags', 0) or
                    getattr(node.cfg.run, 'z_lev_diags', 0)):
                continue
            self.add_domain(node.cfg.grid_id, restored_checkpoint=restored_checkpoint)

    def add_domain(self, grid_id, *, restored_checkpoint=False):
        from woof.core.level_output import initial_diagnostics
        node = self.parent.model.node(grid_id)
        cfg = node.cfg.run
        if not (cfg.p_lev_diags or cfg.z_lev_diags):
            return
        if not restored_checkpoint:
            initial_diagnostics(node.state, cfg, self.parent._metadata_by_grid_id[grid_id])
        for prefix, number in (('p', 23), ('z', 22)):
            if getattr(cfg, prefix+'_lev_diags'):
                stream = _Stream(self, node, prefix, number)
                self.streams[grid_id, number] = stream
                if restored_checkpoint:
                    stream._last_tick = float(node.clock.elapsed_seconds)-float(getattr(node.state, 'domain_start_offset', 0.) or 0.)
                else:
                    stream.submit(node.clock.ticks)

    def drain(self, grid_id=None):
        for (gid, _), stream in self.streams.items():
            if grid_id is None or grid_id == gid:
                stream.drain()

    def remove_domain(self, grid_id):
        for number in (22, 23):
            stream = self.streams.pop((grid_id, number), None)
            if stream is not None:
                stream.close()
                self.archived.append(stream)

    def submit(self, node, ticks):
        for number in (22, 23):
            stream = self.streams.get((node.cfg.grid_id, number))
            if stream is not None:
                stream.submit(ticks)

    def close(self):
        error = None
        for stream in self.streams.values():
            try:
                stream.close()
            except BaseException as exc:
                if error is None:
                    error = exc
        if self.streams or self.archived:
            path = self.parent.output_dir.parent / 'auxhist-level-output.json'
            path.write_text(json.dumps({'streams': [
                {'grid_id': s.node.cfg.grid_id, 'stream': s.stream,
                 'frames': s.count, 'paths': [str(p) for p in s.paths]}
                for s in (*self.archived, *self.streams.values())]}, indent=2)+'\n')
        if error is not None:
            raise error
