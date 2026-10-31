"""Device-memory census for one hex run: every CuPy pool allocation by call site, snapshot at the peak.

Activated by a sitecustomize shim when HEX_MEMCENSUS=<out.json> is set. Installs a CuPy MemoryHook in the main
thread and in every thread started afterwards (CuPy hooks are thread-local), so rank threads and executor
workers are counted too. No device syncs are added; the hook runs on the host at malloc/free time only.

Output JSON: per device, the live-bytes peak, and at that peak the live bytes grouped by
(innermost hexcore file:line function, field name when the allocating frame names one) plus the
pool's cached total and cudaMemGetInfo used.
"""
from __future__ import annotations
import atexit, collections, json, os, sys, threading, time

_SKIP = ('/cupy/', '/cupyx/', 'hex_memcensus', '/numpy/', '<frozen')
_NAME_KEYS = ('name', 'field', 'key', 'pool_name', 'var', 'array_name', 'n')


def _site(frame):
    chain = []
    detail = None
    f = frame
    while f is not None and len(chain) < 3:
        fn = f.f_code.co_filename
        i = fn.find('/hexcore/')
        if i >= 0:
            chain.append(f"{fn[i + 9:]}:{f.f_lineno}:{f.f_code.co_name}")
            if detail is None:
                try:
                    loc = f.f_locals
                    for k in _NAME_KEYS:
                        v = loc.get(k)
                        if isinstance(v, str) and 0 < len(v) <= 48:
                            detail = f"{k}={v}"
                            break
                except Exception:
                    pass
        f = f.f_back
    if not chain:
        f = frame
        while f is not None:
            fn = f.f_code.co_filename
            if not any(s in fn for s in _SKIP):
                chain.append(f"{os.path.basename(fn)}:{f.f_lineno}:{f.f_code.co_name}")
                break
            f = f.f_back
    return (' < '.join(chain) or '?', detail or '')


class _Dev:
    __slots__ = ('live', 'site', 'total', 'peak', 'snap', 'snap_at', 'snap_t', 'snap_pool', 'snap_used', 'nmalloc',
                 'modpeak', 'modsnap', 'sitemax', 'timeline', 'tl_last')

    def __init__(self):
        self.live = {}
        self.site = collections.Counter()
        self.total = 0
        self.peak = 0
        self.snap = {}
        self.snap_at = 0
        self.snap_t = 0.0
        self.snap_pool = 0
        self.snap_used = 0
        self.nmalloc = 0
        self.modpeak = {}
        self.modsnap = {}
        self.sitemax = {}
        self.timeline = []
        self.tl_last = 0


class Census:
    def __init__(self, out, step_mib=16):
        self.out = out
        self.lock = threading.Lock()
        self.dev = collections.defaultdict(_Dev)
        self.step = step_mib << 20
        self.t0 = time.time()
        self.hook = None
        self.last_write = 0.0
        self.owner = {}
        self.by_thread = os.environ.get('HEX_MEMCENSUS_BY_THREAD') == '1'

    def _snapshot(self, d, device_id):
        d.snap = dict(d.site)
        d.snap_at = d.total
        d.snap_t = time.time() - self.t0
        try:
            import cupy as cp
            d.snap_pool = int(cp.get_default_memory_pool().total_bytes())
            free, tot = cp.cuda.runtime.memGetInfo()
            d.snap_used = int(tot - free)
        except Exception:
            pass

    def _key(self, device_id):
        # HEX_MEMCENSUS_BY_THREAD=1: one census per (device, allocating thread), so several same-host ranks
        # sharing one card are told apart (rank threads are named by the launcher's executor).
        if not self.by_thread:
            return device_id
        return f"{device_id}:{threading.current_thread().name}"

    def malloc(self, device_id, mem_size, mem_ptr, frame):
        site = _site(frame)
        key = self._key(device_id)
        with self.lock:
            self.owner[(device_id, mem_ptr)] = key
            d = self.dev[key]
            d.nmalloc += 1
            d.live[mem_ptr] = (mem_size, site)
            d.site[site] += mem_size
            d.total += mem_size
            v = d.site[site]
            if v > d.sitemax.get(site, 0):
                d.sitemax[site] = v
            mod = site[0].split(' < ')[0].split(':')[0]
            if d.total >= d.modpeak.get(mod, 0) + self.step:
                d.modpeak[mod] = d.total
                d.modsnap[mod] = (d.total, time.time() - self.t0, dict(d.site))
            if d.total > d.peak:
                d.peak = d.total
                if d.total >= d.snap_at + self.step or not d.snap:
                    self._snapshot(d, device_id)
            if abs(d.total - d.tl_last) >= (32 << 20):
                d.tl_last = d.total
                d.timeline.append((round(time.time() - self.t0, 3), d.total, mod, site[1]))
        now = time.time()
        if now - self.last_write > 20.0:
            self.last_write = now
            self.write()

    def free(self, device_id, mem_ptr):
        with self.lock:
            d = self.dev.get(self.owner.pop((device_id, mem_ptr), device_id))
            if d is None:
                return
            rec = d.live.pop(mem_ptr, None)
            if rec is None:
                return
            size, site = rec
            d.total -= size
            c = d.site[site] - size
            if c:
                d.site[site] = c
            else:
                del d.site[site]
            if abs(d.total - d.tl_last) >= (32 << 20):
                d.tl_last = d.total
                d.timeline.append((round(time.time() - self.t0, 3), d.total, 'free:' + site[0].split(' < ')[0].split(':')[0], site[1]))

    def write(self):
        with self.lock:
            res = {'pid': os.getpid(), 'argv': sys.argv[:6], 'elapsed_s': round(time.time() - self.t0, 1), 'devices': {}}
            for dev, d in self.dev.items():
                rows = sorted(d.snap.items(), key=lambda kv: -kv[1])
                res['devices'][str(dev)] = {
                    'peak_live_bytes': d.peak, 'snapshot_live_bytes': d.snap_at, 'snapshot_t_s': round(d.snap_t, 1),
                    'snapshot_pool_total_bytes': d.snap_pool, 'snapshot_device_used_bytes': d.snap_used,
                    'mallocs': d.nmalloc, 'live_now_bytes': d.total,
                    'sites': [{'site': s[0], 'detail': s[1], 'bytes': b} for (s, b) in rows if b >= 1 << 16],
                    'sites_small_bytes': sum(b for (s, b) in rows if b < 1 << 16),
                    'site_max': [{'site': s[0], 'detail': s[1], 'bytes': b}
                                 for (s, b) in sorted(d.sitemax.items(), key=lambda kv: -kv[1]) if b >= 1 << 20],
                    'module_peaks': {m: {'live_bytes': tot, 't_s': round(t, 2),
                                         'sites': [{'site': s[0], 'detail': s[1], 'bytes': b}
                                                   for (s, b) in sorted(sn.items(), key=lambda kv: -kv[1]) if b >= 1 << 20]}
                                     for m, (tot, t, sn) in d.modsnap.items()},
                    'timeline': d.timeline,
                    'live_now_sites': [{'site': s[0], 'detail': s[1], 'bytes': b}
                                       for (s, b) in sorted(d.site.items(), key=lambda kv: -kv[1]) if b >= 1 << 20],
                }
        tmp = self.out + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(res, fh, indent=1)
        os.replace(tmp, self.out)


def install(out):
    import cupy  # noqa: F401
    from cupy.cuda import memory_hook
    census = Census(out.replace('{pid}', str(os.getpid())))

    class Hook(memory_hook.MemoryHook):
        name = 'HexMemCensus'

        def malloc_postprocess(self, **kw):
            census.malloc(kw.get('device_id'), kw.get('mem_size'), kw.get('mem_ptr'), sys._getframe(1))

        def free_postprocess(self, **kw):
            census.free(kw.get('device_id'), kw.get('mem_ptr'))

    hook = Hook()
    census.hook = hook
    hook.__enter__()
    orig_run = threading.Thread.run

    def run(self, *a, **k):
        try:
            hook.__enter__()
        except Exception:
            pass
        try:
            return orig_run(self, *a, **k)
        finally:
            try:
                hook.__exit__(None, None, None)
            except Exception:
                pass

    threading.Thread.run = run
    atexit.register(census.write)
    return census
