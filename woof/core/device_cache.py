"""Lazy cache ownership for context-bound handles and stream-owned scratch."""
from collections import OrderedDict
from functools import lru_cache, wraps
from threading import RLock


@lru_cache(maxsize=1)
def _install_prebuilt_images():
    """Core callers use the shipped images even without a forecast supervisor.

    Thompson's large unit still leaves a serial tail after parallel static
    compilation. The source/option/compiler/SM-bound pack avoids that compile
    entirely. Installing at the CUDA handle boundary also covers library
    callers; it starts no process pool and never re-imports their main script.
    """
    import cupy as cp
    # Host parity backends provide CUDA-shaped stand-ins, not a compiler.
    if not hasattr(cp.cuda, "compiler"):
        return
    from woof.kernel_aot import install
    install(cp)


def cuda_cache(*, maxsize=None, stream=False, ready=False):
    """Add the current device, and optionally stream, to a factory's key.

    Factories execute under a lock so first-use callers cannot publish two
    owners. The lock never covers kernel launches by consumers. CuPy is
    imported at use time, allowing CPU-only source and inventory checks.
    Use stream=True for writable scratch. Use ready=True for immutable
    uploaded tables: consumers wait for the upload event without duplicating
    tables or attempting another host upload during graph capture.
    """
    def decorate(factory):
        lock = RLock()
        ages = OrderedDict()
        devices = set()
        bounded_uploads = ready and maxsize is not None and maxsize > 0

        @lru_cache(maxsize=maxsize)
        def cached(owner, args, kwargs):
            value = factory(*args, **dict(kwargs))
            if ready:
                import cupy as cp
                event = cp.cuda.Event(disable_timing=True)
                event.record()
                return value, event, int(cp.cuda.get_current_stream().ptr)
            return value

        @wraps(factory)
        def call(*args, **kwargs):
            import cupy as cp
            _install_prebuilt_images()
            owner = (int(cp.cuda.Device().id),
                     int(cp.cuda.get_current_stream().ptr) if stream else None)
            keywords = tuple(kwargs.items())
            key = (owner, args, keywords)
            with lock:
                if bounded_uploads and key not in ages and len(ages) >= maxsize:
                    oldest = next(iter(ages))
                    # Upload ordering only protects first use. An evicted
                    # table can still be read on any stream of its owner.
                    with cp.cuda.Device(oldest[0][0]):
                        cp.cuda.Device().synchronize()
                value = cached(owner, args, keywords)
                if ready:
                    devices.add(owner[0])
                if bounded_uploads:
                    if key in ages:
                        ages.move_to_end(key)
                    else:
                        if len(ages) >= maxsize:
                            ages.popitem(last=False)
                        ages[key] = None
            if ready:
                value, event, source = value
                current = cp.cuda.get_current_stream()
                if int(current.ptr) != source and not event.done:
                    current.wait_event(event)
            return value

        def clear():
            with lock:
                if ready:
                    import cupy as cp
                    for device in sorted(devices):
                        with cp.cuda.Device(device):
                            cp.cuda.Device().synchronize()
                cached.cache_clear()
                ages.clear()
                devices.clear()

        call.cache_clear = clear
        call.cache_info = cached.cache_info
        return call
    return decorate


_READY_LOCK = RLock()


def cached_ready(cp, cache, key, factory):
    """Publish immutable values only after recording their upload event."""
    current = cp.cuda.get_current_stream()
    with _READY_LOCK:
        held = cache.get(key)
        if held is None:
            value = factory()
            event = cp.cuda.Event(disable_timing=True)
            event.record()
            held = (value, event, int(current.ptr))
            cache[key] = held
    value, event, source = held
    if int(current.ptr) != source and not event.done:
        current.wait_event(event)
    return value
