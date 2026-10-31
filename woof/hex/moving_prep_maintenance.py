"""Coalesce moving-run preparation-cache housekeeping outside the move pause."""
from pathlib import Path
import threading
import time

_LOCK = threading.Lock()
_PENDING = {}
_WORKER = None


def request(root, protect):
    global _WORKER
    with _LOCK:
        _PENDING.setdefault(Path(root), set()).update(protect)
        if _WORKER is None:
            _WORKER = threading.Thread(target=_run, name='hex-moving-cache-maintenance', daemon=True)
            _WORKER.start()


def _run():
    global _WORKER
    from . import prep_cache
    while True:
        # Batch the entries published together by the class/preparation workers.
        threading.Event().wait(.2)
        with _LOCK:
            todo = list(_PENDING.items())
            _PENDING.clear()
        for root, protected in todo:
            class LiveProtect:
                def __contains__(self, key):
                    # A class may publish another entry while this sweep scans.
                    with prep_cache._lock:
                        return key in protected or key in prep_cache._memory
            try:
                prep_cache.evict(root, LiveProtect())
            except OSError as error:
                prep_cache._note(dict(event='moving-maintenance-failed', reason=str(error)))
        with _LOCK:
            if not _PENDING:
                _WORKER = None
                return


def drain(timeout=5.):
    """Finish outstanding housekeeping before tests or owned-scratch cleanup."""
    end = time.monotonic()+timeout
    while True:
        with _LOCK:
            worker = _WORKER
        if worker is None:
            return True
        left = end-time.monotonic()
        if left <= 0:
            return False
        worker.join(left)
