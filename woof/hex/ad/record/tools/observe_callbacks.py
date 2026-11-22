#!/usr/bin/env python3
"""Observe callback source locations in the separate ordinary referee.

This records routing metadata only. A Python callback can perform host work,
call a native routine, launch GPU work, or do all three. A source location is
neither an implementation of that work nor proof that the host census is
complete. The native replay never imports this observer.
"""
# SPDX-License-Identifier: Apache-2.0
from functools import partial, wraps
import gzip
import inspect
import json
import sys
import threading
import time

import observe_cupti_host as host
import observe_cupti_outer as base


def locations(callback, max_depth=4):
    """Follow callable defaults, closures and partials, never their data."""
    result, seen = [], set()

    def visit(value, depth, relation):
        ident = id(value)
        if ident in seen or depth > max_depth:
            return
        seen.add(ident)
        if isinstance(value, partial):
            visit(value.func, depth + 1, relation + '.partial')
            return
        function = value.__func__ if inspect.ismethod(value) else value
        if not inspect.isfunction(function):
            implementation = getattr(type(value), '__call__', None)
            if inspect.isfunction(implementation):
                visit(implementation, depth + 1, relation + '.__call__')
            else:
                result.append({'relation': relation, 'kind': 'unresolved-native-or-builtin',
                               'type': type(value).__qualname__})
            return
        code = function.__code__
        result.append({'relation': relation, 'kind': 'python', 'path': code.co_filename,
                       'line': code.co_firstlineno, 'function': function.__qualname__})
        values = list(function.__defaults__ or ())
        values.extend((function.__kwdefaults__ or {}).values())
        for cell in function.__closure__ or ():
            try:
                values.append(cell.cell_contents)
            except ValueError:
                pass
        wrapped = getattr(function, '__wrapped__', None)
        if wrapped is not None:
            values.append(wrapped)
        for inner in values:
            if callable(inner) and not isinstance(inner, type):
                visit(inner, depth + 1, relation + '.captured')

    visit(callback, 0, 'selected')
    return result


def install(observer):
    from woof.hex.mpasphys.driver_v842 import NativeDriverHooks, SlotRegistry

    def emit(record):
        record.update({'clock_ns': time.monotonic_ns(), 'thread_id': threading.get_native_id(),
                       'complete': False})
        with observer.lock:
            observer.callback_output.write(json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n')

    original_invoke = NativeDriverHooks.invoke

    @wraps(original_invoke)
    def invoke(hooks, operation, call):
        external = hooks.external.get(operation)
        if external is not None:
            route, callbacks = 'external', {'selected': locations(external)}
        else:
            route = 'native-entry-or-unresolved-resident'
            callbacks = {'arguments': locations(hooks.arguments), 'native_call': locations(hooks.native.call),
                         'after_call': locations(hooks.after_call)}
        emit({'event': 'host_glue', 'operation': operation, 'step': int(call.itimestep),
              'backend': str(call.backend), 'route': route, 'callbacks': callbacks})
        return original_invoke(hooks, operation, call)

    NativeDriverHooks.invoke = invoke
    original_run = SlotRegistry.run

    @wraps(original_run)
    def run(registry, call):
        callback = registry._callbacks[call.slot, call.scheme, call.backend]
        emit({'event': 'scheme_slot', 'operation': call.slot, 'scheme': call.scheme,
              'step': int(call.itimestep), 'backend': str(call.backend),
              'callbacks': {'selected': locations(callback)}})
        return original_run(registry, call)

    SlotRegistry.run = run


class CallbackObserver(host.HostObserver):
    def __init__(self, root):
        super().__init__(root)
        self.callback_output = gzip.open(root / 'host-callbacks.jsonl.gz', 'wt',
                                         encoding='utf-8', compresslevel=1)

    def install(self):
        super().install()
        install(self)

    def outer_boundary(self, frame, step, cp):
        super().outer_boundary(frame, step, cp)
        self.callback_output.flush()

    def close(self):
        try:
            self.callback_output.close()
        finally:
            super().close()


if __name__ == '__main__':
    base.Observer = CallbackObserver
    sys.exit(base.main())
