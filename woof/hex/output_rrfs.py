"""RRFS stream session for driver-selected native output and DA/restart ingress.

Fork 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:120-166,662-675.
The suite/pin table supplies this module to the driver; no suite name is tested.
All numeric transport goes through o1's rw_mpas_stream artifact.
"""
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Any
import time

from .streams_rrfs import Stream, StreamAlarms, parse_streams, _interval
from .restart_rrfs import RestartControl, read_restart, write_stream, write_restart, global_attributes_enabled


def native_file_id(seed: int, ordinal: int) -> str:
    """RRFS MPAS-Model-rrfs/src/framework/random_id.c:20-46, original libc sequence (Python)."""
    from .mpasphys.rrfs.stream_xml_rrfs import native_file_id as file_id
    return file_id(seed & 0xffffffff, ordinal)


def stream_file_attributes(stream: Stream) -> dict:
    """RRFS MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:4379-4444: what a new file adds after file_id.

    The stream interval attribute is an integer minute count of the filename interval, unrelated to
    the forecast hour or the output_timelevels interval. With months/years the native getter returns
    before setting it, so such a stream adds nothing.
    """
    raw = stream.options.get('filename_interval', 'none')
    if raw[:4] != 'none' and raw != 'output':
        interval = _interval(raw)
        if not interval.months:
            return {'output_interval': int(interval.seconds // 60)}
    return {}


@dataclass
class StreamSession:
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:120-166,662-675."""
    streams: Mapping[str, Stream]
    alarms: StreamAlarms
    control: RestartControl
    output_directory: Path
    random_seed: int = field(default_factory=lambda: int(time.time()))
    _random_ordinal: int = field(default=0, init=False, repr=False)
    _open_paths: dict[str, Path] = field(default_factory=dict, init=False, repr=False)

    def build_attributes(self, stream: Stream, inherited=None) -> dict:
        """RRFS MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:4379-4444.

        Native seeds its manager once and consumes ten libc rand words for each
        output stream build. The stream interval attribute is an integer minute
        count, unrelated to the forecast hour or the output_timelevels interval.
        """
        attrs = dict(inherited or {})
        if not global_attributes_enabled(stream):
            return {}
        attrs['file_id'] = native_file_id(self.random_seed, self._random_ordinal)
        self._random_ordinal += 1
        # With months/years the native getter returns before setting M; any
        # authority-supplied word is kept (stream_file_attributes).
        attrs.update(stream_file_attributes(stream))
        return attrs

    @classmethod
    def from_namelist(cls, namelist: str, stream_path: str | Path, *,
                      start_time: datetime, initial_time: datetime, final_time: datetime,
                      output_directory: str | Path, active_packages=(), physics=True):
        """RRFS MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:120-166,662-675."""
        control = RestartControl.from_namelist(namelist)
        streams = parse_streams(stream_path, physics=physics)
        alarms = StreamAlarms(streams, start_time, initial_time, final_time,
                              do_restart=control.config_do_restart,
                              do_DAcycling=control.config_do_DAcycling,
                              active_packages=frozenset(active_packages))
        return cls(streams, alarms, control, Path(output_directory))

    def read_initial_state(self, *, input_directory: str | Path, scratch: str | Path,
                           physics=True, authority_layout=True, zero_based_connectivity=()):
        """RRFS MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:137-166.

        Caller restores every returned active field before initializing diagnostics.
        Invariants and IAU are separate driver reads, as in the native core.
        """
        stream = self.streams[self.control.initial_stream]
        filename = stream.filename(self.alarms.start_time, self.alarms.initial_time)
        return read_restart(Path(input_directory)/filename,
                            valid_time=self.alarms.start_time, scratch=scratch,
                            active_packages=self.alarms.active_packages, physics=physics,
                            stream=stream, streams=self.streams,
                            authority_layout=authority_layout,
                            zero_based_connectivity=zero_based_connectivity)

    def publish_due(self, when: datetime, mesh: Any,
                    fields: Callable[[Stream], Mapping[str, Any]], *, physics=True,
                    global_attrs=None, scratch=None, record_indices=None,
                    restart_timestamp_path=None) -> tuple[Path, ...]:
        """RRFS MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:3140-3527.

        The provider captures actual driver authority for each selected stream.
        A failed capture, Rust write or marker write leaves its alarm ringing.
        """
        written = []
        for name, stream in self.streams.items():
            if not self.alarms.due(name, when): continue
            target = self.output_directory/stream.filename(when, self.alarms.initial_time)
            build_stream = self._open_paths.get(name) != target
            kwargs = dict(initial_time=self.alarms.initial_time,
                          active_packages=self.alarms.active_packages, physics=physics,
                          streams=self.streams,
                          global_attrs=self.build_attributes(stream, global_attrs) if build_stream else global_attrs,
                          update_global_attrs=build_stream, scratch=scratch,
                          record_index=(record_indices or {}).get(name))
            captured = fields(stream)
            # Native clobber policy governs file creation, then records continue
            # in the open file. A transactional Rust write reopens that same
            # file, so preserve its records rather than truncating it again.
            effective = stream if build_stream else replace(stream, options=dict(stream.options,
                clobber_mode='overwrite' if stream.options.get('clobber_mode') == 'overwrite' else 'append'))
            if name == 'restart':
                path = write_restart(target, mesh, captured, when, stream=effective,
                                     timestamp_path=restart_timestamp_path, **kwargs)
            else:
                path = write_stream(target, effective, mesh, captured, when, **kwargs)
            self.alarms.reset(name, when)
            self._open_paths[name] = target
            written.append(path)
        return tuple(written)
