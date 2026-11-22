#!/usr/bin/env python3
"""Compare the live framework clock and alarms with the Rust frame rule.

The bytes are a logical checkpoint allocation, not a complete physics census.
This module reads the ordinary forecast only for the reference comparison.
"""
# SPDX-License-Identifier: Apache-2.0
from fractions import Fraction
import struct


def integral(value):
    value = Fraction(value)
    if value.denominator != 1 or value < 0 or value.numerator >= 1 << 64:
        raise ValueError('frame requires a nonnegative integral-second value')
    return value.numerator


def encode(clock, elapsed, next_step, dt, last_time, active, correct_rp6,
           pbl_due, alarms, time_bits, xtime):
    if clock.get('calendar', 'gregorian') != 'gregorian':
        raise ValueError('unsupported frame calendar')
    if len(active) != 3 or not isinstance(xtime, str) or not xtime:
        raise ValueError('invalid frame state')
    data = bytearray(b'ADFRAME1')
    data += struct.pack('<iIII', int(clock['year']), int(clock['month']),
                        int(clock['day']), integral(clock['seconds']))
    data += struct.pack('<QQQ', integral(elapsed), integral(next_step), integral(dt))
    data += struct.pack('<IQ', int(last_time is not None),
                        0 if last_time is None else integral(last_time))
    active_bits = sum(int(bool(value)) << i for i, value in enumerate(active))
    data += struct.pack('<IIII', active_bits, int(bool(correct_rp6)),
                        int(bool(pbl_due)), len(alarms))
    for name, alarm in sorted(alarms.items()):
        encoded = name.encode('ascii')
        if not encoded:
            raise ValueError('empty alarm name')
        data += struct.pack('<I', len(encoded)) + encoded
        data += struct.pack('<QQ', integral(alarm['period']), integral(alarm['next_ring']))
    encoded = xtime.encode('ascii')
    data += struct.pack('<II', int(time_bits), len(encoded)) + encoded
    return bytes(data)


def live(session, completed_step, physics_step_offset=0):
    """Read ordinary live objects while the forecast is paused at a boundary."""
    import numpy as np

    native = session.native
    manager = session.driver.manager
    clock = native.clock_state
    if clock is None:
        raise ValueError('framework clock is absent')
    current = clock.current_time
    time = np.asarray(native.get_field('state', 'Time', level=1))
    if time.size != 1 or time.dtype.kind != 'f' or time.dtype.itemsize != 4:
        raise ValueError('framework state Time is not one float32 word')
    bits = int.from_bytes(time.astype('<f4', copy=False).tobytes(), 'little')
    alarms = {name: {'period': alarm.period, 'next_ring': alarm.next_ring}
              for name, alarm in manager.alarms.items()}
    active = [manager.config.active(name) for name in
              ('radiation_lw', 'radiation_sw', 'convection')]
    return encode({'calendar': current.calendar, 'year': current.year,
                   'month': current.month, 'day': current.day, 'seconds': current.seconds},
                  clock.elapsed, integral(completed_step) + integral(physics_step_offset) + 1,
                  manager.dt, manager.last_time, active, manager.config.correct('RP6'),
                  manager.pbl_due, alarms, bits, native.get_text_field('state', 'xtime', level=1))
