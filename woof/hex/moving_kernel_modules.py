"""Bounded resident compiled modules, without per-class hooks or field arrays."""
from collections import OrderedDict
from copy import deepcopy
import threading

_LOCK = threading.RLock()
_MODULES = OrderedDict()
MAX_MODULES = 64
_HITS = _MISSES = 0


def get(key):
    global _HITS, _MISSES
    with _LOCK:
        value = _MODULES.get(key)
        if value is not None:
            _HITS += 1
            _MODULES.move_to_end(key)
            module, effective, route = value
            return module, deepcopy(effective), route
        _MISSES += 1
        return None


def put(key, module, effective, route):
    with _LOCK:
        _MODULES[key] = (module, deepcopy(effective), route)
        _MODULES.move_to_end(key)
        while len(_MODULES) > MAX_MODULES:
            _MODULES.popitem(last=False)


def clear():
    global _HITS, _MISSES
    with _LOCK:
        _MODULES.clear()
        _HITS = _MISSES = 0


def stats():
    with _LOCK:
        return dict(modules=len(_MODULES), hits=_HITS, misses=_MISSES)
