"""MPAS-A v8.4.2 stream queries used by in-model diagnostics.

The stream service supplies current views on each query: output alarms may
change between timesteps. No output schedule is inferred here.
"""
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Iterable, Mapping


class Direction(IntEnum):
    INPUT = 1
    OUTPUT = 2
    INPUT_OUTPUT = 3


@dataclass(frozen=True)
class StreamView:
    name: str
    direction: Direction
    active: bool
    output_alarm: bool
    fields: Mapping[str, bool]
    input_interval: str = "none"
    output_interval: str = "none"


class DiagnosticsUtils:
    def init(self, streams: Callable[[], Iterable[StreamView]]) -> None:
        """MPAS v8.4.2 mpas_atm_diagnostics_utils.F:36-46."""
        self.streams = streams

    def finalize(self) -> None:
        """MPAS v8.4.2 mpas_atm_diagnostics_utils.F:60-66: no-op."""

    def field_will_be_written(self, field_name: str) -> bool:
        """MPAS v8.4.2 mpas_atm_diagnostics_utils.F:85-129."""
        streams = self.streams()
        if isinstance(streams, tuple):
            # A provider that hands back the same views tuple for one output event
            # (ResidentOutputDiagnostics.views) is answered from one set of the names
            # those views write: the same predicate as the loop below, built once.
            if streams is not getattr(self, '_written_views', None):
                self._written = frozenset(
                    name.rstrip(" ") for stream in streams
                    if stream.active and stream.direction in (Direction.OUTPUT, Direction.INPUT_OUTPUT)
                    and stream.output_alarm
                    for name, active in stream.fields.items() if active)
                self._written_views = streams
            return field_name.rstrip(" ") in self._written
        for stream in streams:
            if (stream.active and stream.direction in
                    (Direction.OUTPUT, Direction.INPUT_OUTPUT) and stream.output_alarm):
                if any(active and name.rstrip(" ") == field_name.rstrip(" ")
                       for name, active in stream.fields.items()):
                    return True
        return False

    def stream_inclusion_count(self, field_name: str,
                               direction: Direction = Direction.INPUT_OUTPUT,
                               include_inactive: bool = False) -> int:
        """MPAS v8.4.2 mpas_atm_diagnostics_utils.F:148-232."""
        count = 0
        for stream in self.streams():
            active = include_inactive or (stream.active and (
                (direction == Direction.INPUT and stream.input_interval.rstrip(" ") != "none") or
                (direction == Direction.OUTPUT and stream.output_interval.rstrip(" ") != "none") or
                (direction == Direction.INPUT_OUTPUT and (
                    stream.input_interval.rstrip(" ") != "none" or
                    stream.output_interval.rstrip(" ") != "none"))))
            matches = (direction == Direction.INPUT_OUTPUT or
                       stream.direction == Direction.INPUT_OUTPUT or
                       direction == stream.direction)
            if active and matches and any(
                    field_active and name.rstrip(" ") == field_name.rstrip(" ")
                    for name, field_active in stream.fields.items()):
                count += 1
        return count
