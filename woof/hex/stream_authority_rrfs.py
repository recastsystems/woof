"""Complete RRFS Registry/native-pool stream boundary, without field arithmetic.

MPAS-Model-rrfs/src/framework/mpas_io_streams.F:925-1023,4140-4207;
MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:137-166,782-816.
The owner synchronizes CUDA into these pools before publication. Rust owns
record decoding, primitive conversion and file-axis permutation.
"""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .output import HistoryField
from .restart_rrfs import read_stream
from .streams_rrfs import field_definition


class NativePoolStreamAuthority:
    """MPAS-Model-rrfs/src/framework/mpas_io_streams.F:925-1023.

    Bind the owner's allocated Registry layout, including distinct in-code
    names, constituent arrays, time levels and native dummy storage. Missing
    selected fields are errors; no diagnostic whitelist is used.
    """

    def __init__(self, native, layout, *, active_packages, streams, global_attrs,
                 input_directory, restart_timestamp_path=None, physics=True,
                 field_attributes=None):
        """MPAS-Model-rrfs/src/framework/mpas_io_streams.F:925-1023,4140-4207."""
        self.native, self.layout = native, layout
        self.active_packages, self.streams = frozenset(active_packages), streams
        self.global_attrs = dict(global_attrs)
        self.input_directory = Path(input_directory)
        self.restart_timestamp_path = restart_timestamp_path
        self.physics = physics
        self.field_attributes = dict(field_attributes or {})
        self.zero_based_connectivity = ()  # Native pools retain MPAS's one-based IDs.
        self.mesh = SimpleNamespace(dimensions=dict(layout.dimensions), arrays={}, attrs={})
        self.bindings = {}
        for row in layout.fields:
            members = tuple(row['components']) or (row['name'],)
            for index, name in enumerate(members):
                if name in self.bindings:
                    raise ValueError(f'ambiguous native Registry stream field {name}')
                self.bindings[name] = (row, index if row['components'] else None)

    def _binding(self, name):
        """MPAS-Model-rrfs/src/framework/mpas_io_streams.F:925-1023."""
        try:
            return self.bindings[name]
        except KeyError:
            raise ValueError(f'allocated native Registry lacks stream field {name}') from None

    def capture_stream(self, stream, *, level=1):
        """MPAS-Model-rrfs/src/framework/mpas_io_streams.F:925-1023.

        Views only remove native dummy slots and select constituent storage.
        Rust performs the output dimension permutation and primitive encoding.
        """
        fields, buffers = {}, {}
        for name in stream.fields(self.active_packages, physics=self.physics, streams=self.streams):
            if name in ('xtime', 'initial_time'):
                continue  # StreamSession supplies its authoritative calendar strings.
            row, component = self._binding(name)
            key = (row['pool'], row['pool_name'])
            if row['kind'] == 'character':
                value = self.native.get_text_field_storage(*key, level=level)[:64].view('S1')
                dims = ('StrLen',)
            else:
                if key not in buffers:
                    buffers[key] = self.native.get_field(*key, level=level)
                value, dims = buffers[key], tuple(row['dimensions'])
                if component is not None:
                    value, dims = value[component], dims[1:]
                value = value[tuple(slice(0, self.layout.dimensions[d]) for d in dims)]
            attrs = self.field_attributes.get(name, {})
            fields[name] = HistoryField(value, dims, attrs)
        return fields

    def restore_stream(self, state, *, level=1):
        """MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:137-166.

        Require Rust-decoded Registry axes. Merge constituent and dummy views
        into existing native storage, then commit each full buffer once. The
        returned inventory includes consumed clock strings even when the
        owner has already initialized its native calendar from those strings.
        """
        pending, text = {}, []
        for name, var in state.variables.items():
            if name in ('xtime', 'initial_time'):
                if name in self.bindings:
                    row, _ = self.bindings[name]
                    text.append((row, var.text[0]))
                continue
            row, component = self._binding(name)
            key = (row['pool'], row['pool_name'])
            if row['kind'] == 'character':
                if len(var.text) != 1:
                    raise ValueError(f'{name}: scalar native character field required')
                text.append((row, var.text[0]))
                continue
            dims = tuple(row['dimensions'][1:] if component is not None else row['dimensions'])
            if var.dimensions != dims or var.values.shape != tuple(self.layout.dimensions[d] for d in dims):
                raise ValueError(f'{name}: Rust ingress differs from allocated Registry axes')
            if key not in pending:
                pending[key] = (row, self.native.get_field(*key, level=level))
            buffer = pending[key][1]
            target = buffer if component is None else buffer[component]
            if var.values.dtype != buffer.dtype:
                raise TypeError(f'{name}: Rust ingress primitive differs from native storage')
            target[tuple(slice(0, self.layout.dimensions[d]) for d in dims)] = var.values
        for key, (row, buffer) in pending.items():
            self.native.put_field(*key, buffer, level=level, nlevels=row['nlevels'])
        for row, value in text:
            self.native.put_text_field(row['pool'], row['pool_name'], value,
                                       level=level, nlevels=row['nlevels'])
        return tuple(state.variables)

    def read_file(self, path, names, *, record, scratch, level=1):
        """MPAS-Model-rrfs/src/framework/mpas_io_streams.F:925-1023."""
        layouts = {}
        for name in names:
            definition = field_definition(name, physics=self.physics)
            if definition['type'] == 'text':
                continue
            layouts[name] = dict(dtype='f4' if definition['type'] == 'real' else 'i4',
                dimensions=[d for d in definition['dimensions'] if d != 'Time'],
                expected_dimensions=list(reversed(definition['dimensions'])))
        with read_stream(path, scratch=scratch, select=names, record=record, layouts=layouts) as state:
            restored = self.restore_stream(state, level=level)
            if frozenset(restored) != frozenset(names):
                raise ValueError('native stream read omitted selected Registry fields')
            return restored

    def read_stream_latest_before(self, stream, when, session, *, scratch):
        """MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:782-816."""
        path = self.input_directory / stream.filename(when, session.alarms.initial_time)
        with read_stream(path, scratch=scratch, select=('xtime',)) as clock:
            stamps = clock.variables['xtime'].text
        eligible = [(datetime.strptime(stamp, '%Y-%m-%d_%H:%M:%S'), i)
                    for i, stamp in enumerate(stamps)]
        eligible = [(stamp, i) for stamp, i in eligible if stamp <= when]
        if not eligible:
            raise ValueError(f'{stream.name}: no native record at or before {when}')
        record = max(eligible)[1]
        names = stream.fields(self.active_packages, physics=self.physics, streams=self.streams)
        return self.read_file(path, names, record=record, scratch=scratch)

    def record_indices(self, session, when, *, scratch):
        """MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:3140-3527."""
        indices = {}
        for name, stream in self.streams.items():
            if not session.alarms.due(name, when):
                continue
            path = session.output_directory / stream.filename(when, session.alarms.initial_time)
            if session._open_paths.get(name) == path:
                with read_stream(path, scratch=scratch, select=('xtime',)) as clock:
                    indices[name] = len(clock.variables['xtime'].text)
        return indices
