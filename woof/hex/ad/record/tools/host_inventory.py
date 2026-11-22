#!/usr/bin/env python3
"""Observe candidates for host-owned forecast state at outer boundaries.

Only metadata and SHA-256 are emitted, not recorded bytes for replay. The
snapshot is an inventory of explicitly visited framework storage, not proof
that every host owner, live field, clock, or random counter was visited.
"""
# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json


def _array(name, value, lower=None):
    import numpy as np
    result = {'name': name, 'kind': 'array', 'complete': False}
    if not isinstance(value, np.ndarray):
        result.update({'kind': 'unsupported', 'type': type(value).__qualname__})
        return result
    result.update({'dtype': value.dtype.str, 'shape': list(value.shape),
                   'strides': list(value.strides), 'bytes': int(value.nbytes),
                   'address': int(value.__array_interface__['data'][0]),
                   'lower': list(lower) if lower is not None else None})
    if not (value.flags.c_contiguous or value.flags.f_contiguous) or value.dtype.hasobject:
        result['kind'] = 'unsupported-layout'
        return result
    flat = value.ravel(order='K')
    if value.size and not np.shares_memory(flat, value):
        result['kind'] = 'unsupported-copy'
        return result
    result['sha256'] = hashlib.sha256(memoryview(flat).cast('B')).hexdigest()
    return result


def _text(name, value):
    if not isinstance(value, bytes):
        return {'name': name, 'kind': 'unsupported', 'type': type(value).__qualname__, 'complete': False}
    return {'name': name, 'kind': 'text', 'bytes': len(value),
            'sha256': hashlib.sha256(value).hexdigest(), 'complete': False}


def _safe_key(key):
    if not isinstance(key, tuple) or not all(isinstance(x, str) for x in key):
        raise TypeError('unexpected framework pool key')
    return '.'.join(key)


class HostInventory:
    def __init__(self, workers=4):
        if not 1 <= workers <= 4:
            raise ValueError('invalid worker count')
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='binder-host-hash')

    def close(self):
        self.executor.shutdown(wait=True)

    def sample(self, frame, step):
        session = frame.f_locals['session']
        native = session.native
        if not getattr(native, 'framework_only', False):
            raise ValueError('native-library host storage requires its own census')
        jobs = []
        for key, levels in sorted(native._storage.items()):
            name = 'framework.field.' + _safe_key(key)
            for index in range(len(levels)):
                # Do not trigger _DeferredLevels.__getitem__ and materialize a
                # value that the ordinary forward has not needed.
                value = levels.peek(index) if hasattr(levels, 'peek') else levels[index]
                label = name + '.' + str(index + 1)
                if value is None:
                    jobs.append(self.executor.submit(lambda label=label:
                        {'name': label, 'kind': 'deferred', 'complete': False}))
                else:
                    jobs.append(self.executor.submit(_array, label, value))
        for name, (value, lower) in sorted(native._variables.items()):
            jobs.append(self.executor.submit(_array, 'framework.variable.' + name, value, lower))
        for key, levels in sorted(native._text.items()):
            for index, value in enumerate(levels):
                jobs.append(self.executor.submit(_text, 'framework.text.' + _safe_key(key) + '.' + str(index + 1), value))
        manager = session.driver.manager
        clock = native.clock_state
        stochastic = getattr(frame.f_locals.get('coupling'), 'stochastic', None)
        state = {
            'clock': {'current': clock.current_time.stamp(), 'elapsed': str(clock.elapsed), 'dt': str(clock.dt)},
            'manager': {'dt': str(manager.dt), 'last_time': None if manager.last_time is None else str(manager.last_time),
                        'alarms': {name: {'period': str(a.period), 'next_ring': str(a.next_ring)}
                                   for name, a in sorted(manager.alarms.items())},
                        'scheme_dt': {name: str(value) for name, value in sorted(manager.scheme_dt.items())},
                        'pbl_due': bool(manager.pbl_due)},
            'driver_pending': getattr(session.driver, '_pending', None) is not None,
            'tendencies_ready': bool(getattr(session.driver, '_tendencies_ready', False)),
            'outer_synchronized_after': frame.f_locals.get('synchronized_after'),
            'stochastic_present': stochastic is not None,
            'stochastic_state': 'unresolved' if stochastic is not None else 'absent',
        }
        return {'schema': 'ad.binder.phys-host-observation/v1', 'step': int(step),
                'boundary': 'outer_after_step', 'complete': False, 'state': state,
                'items': [job.result() for job in jobs],
                'limitations': ['This covers framework storage and selected scalar owners only.',
                                'Resident device authority, aliases, and lifetimes are unresolved.',
                                'Host scheme objects and RNG state need separate discovery.',
                                'Hashes check observations and are never replay inputs.']}
