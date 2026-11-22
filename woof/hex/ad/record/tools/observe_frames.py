#!/usr/bin/env python3
"""Observe host clock and physics-manager words in an ordinary forecast.

This is a partial observer, not a sealed binder. It reads no weather buffers.
For native-library clocks it does not snapshot hidden Fortran storage.
"""
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import dataclasses
import functools
import gzip
import json
import sys
from pathlib import Path


def alarm_state(manager):
    return {name: {"period": str(value.period), "next_ring": str(value.next_ring)}
            for name, value in sorted(manager.alarms.items())}


def clock_state(native):
    clock = getattr(native, "clock_state", None)
    if clock is None:
        return None
    current = clock.current_time
    return {"stamp": current.stamp(), "calendar": current.calendar,
            "year": current.year, "month": current.month, "day": current.day,
            "seconds": str(current.seconds), "elapsed": str(clock.elapsed), "dt": str(clock.dt)}


def install(path):
    from woof.hex.mpasphys.driver_v842 import NativeCalendarState, PhysicsManager
    from woof.hex.mpasphys.framework_backend_v842 import FrameworkPhysics
    import numpy as np

    path = Path(path)
    output = (gzip.open(path, "wt", encoding="utf-8", compresslevel=1)
              if path.suffix == ".gz" else path.open("w", encoding="utf-8"))

    def emit(row):
        row["complete"] = False
        output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        output.flush()

    manager_advance = PhysicsManager.advance

    @functools.wraps(manager_advance)
    def advance(manager, seconds):
        before = alarm_state(manager)
        active = {name: bool(manager.config.active(name)) for name in
                  ("radiation_lw", "radiation_sw", "convection")}
        result = manager_advance(manager, seconds)
        emit({"event": "manager", "time_seconds": float(seconds), "dt": str(manager.dt),
              "alarms_before": before, "alarms_after": alarm_state(manager),
              "scheme_dt": {name: str(value) for name, value in sorted(manager.scheme_dt.items())},
              "active": active, "flags": dataclasses.asdict(result),
              "pbl_due": bool(manager.pbl_due), "correct_rp6": bool(manager.config.correct("RP6"))})
        return result

    PhysicsManager.advance = advance
    calendar = NativeCalendarState.__call__

    @functools.wraps(calendar)
    def calendar_call(state, call):
        before = clock_state(state.native)
        result = calendar(state, call)
        julian = np.asarray(result["julian"], dtype=np.float32)
        julian_in = np.asarray(result["julian_in"], dtype=np.float32)
        emit({"event": "calendar", "step": int(call.itimestep),
              "time_seconds": float(call.time_seconds), "flags": dataclasses.asdict(call.flags),
              "framework_only": bool(getattr(state.native, "framework_only", False)),
              "clock_before": before, "clock_after": clock_state(state.native),
              "result": {"current_date": result["current_date"],
                         "julian_f32_le": julian.tobytes().hex(),
                         "julian_in_f32_le": julian_in.tobytes().hex(),
                         "leapyear": bool(result["leapyear"])}})
        return result

    NativeCalendarState.__call__ = calendar_call
    clock_advance = FrameworkPhysics.advance_clock

    @functools.wraps(clock_advance)
    def advance_clock(native):
        before = clock_state(native)
        result = clock_advance(native)
        emit({"event": "clock_advance", "before": before, "after": clock_state(native)})
        return result

    FrameworkPhysics.advance_clock = advance_clock

    update_state_time = FrameworkPhysics.update_state_time

    @functools.wraps(update_state_time)
    def state_time(native, dt, *, pool='state', clock_advanced=False, source_level=1, target_level=2):
        # A scalar observation for later equality checking; this does not
        # provide data to the native replay.
        source = np.asarray(native.get_field(pool, 'Time', level=source_level), dtype=np.float32)
        if source.size != 1:
            raise RuntimeError('state Time is not a scalar')
        # A source and target may share storage. Snapshot before the call so
        # the observation cannot silently become the updated value.
        source_word = source.tobytes().hex()
        result = update_state_time(native, dt, pool=pool, clock_advanced=clock_advanced,
                                   source_level=source_level, target_level=target_level)
        target = np.asarray(native.get_field(pool, 'Time', level=target_level), dtype=np.float32)
        if target.size != 1:
            raise RuntimeError('updated state Time is not a scalar')
        emit({'event': 'state_time', 'pool': pool, 'clock_advanced': clock_advanced,
              'source_level': source_level, 'target_level': target_level,
              'clock': clock_state(native),
              'target_xtime': native.get_text_field(pool, 'xtime', level=target_level),
              'dt_f32_le': np.asarray(dt, dtype=np.float32).tobytes().hex(),
              'source_f32_le': source_word, 'target_f32_le': target.tobytes().hex()})
        return result

    FrameworkPhysics.update_state_time = state_time
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("forecast", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.forecast or args.forecast[0] != "--":
        parser.error("put -- before the ordinary hex CLI arguments")
    output = install(args.out)
    try:
        from woof.hex.cli import main as forecast
        return forecast(args.forecast[1:])
    finally:
        output.close()


if __name__ == "__main__":
    sys.exit(main())
