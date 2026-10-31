"""One-use transfer cache for an unstepped predicted moving class.

The admitted class gets a new driver, context and validation. Only its input
allocations are reused, with exact integer-word updates for changed entries.
Never attach this cache to a driver that has advanced a forecast step.
"""
from contextvars import ContextVar
import numpy as np

ACTIVE = ContextVar('hex_moving_upload_cache', default=None)


class UploadCache:
    def __init__(self):
        self.arrays = {}
        self.claimed = False
        self.reused_bytes = self.changed_bytes = self.fresh_bytes = 0

    def claim(self):
        if self.claimed:
            return None
        self.claimed = True
        return self

    def upload(self, cp, namespace, name, source):
        if namespace in ('DevicePrognosticState.from_host','DeviceSavedDiagnostics.from_host'):
            # These fields change with every carried forecast. Comparing and
            # snapshotting them only duplicates their obligatory full upload.
            return cp.asarray(source, order='K')
        source = np.asarray(source)
        if not (source.flags.c_contiguous or source.flags.f_contiguous):
            from .host_parallel import contiguous_copy
            source = contiguous_copy(source,dtype=source.dtype)
        key = namespace, name
        previous = self.arrays.get(key)
        if previous is None or previous[0].shape != source.shape or previous[0].dtype != source.dtype:
            device = cp.ascontiguousarray(cp.asarray(source, order='K'))
            self.fresh_bytes += source.nbytes
        else:
            old, device = previous
            # Float comparisons would equate signed zeros and reject identical
            # NaNs. Integer words preserve every payload, including denormals.
            unsigned = np.dtype('u'+str(source.dtype.itemsize))
            before, after = old.view(unsigned), source.view(unsigned)
            mask = before != after
            count = int(np.count_nonzero(mask))
            self.reused_bytes += source.nbytes
            self.changed_bytes += count * source.dtype.itemsize
            if count:
                if count * 8 < source.size:
                    coordinates = np.nonzero(mask)
                    changed = np.ravel_multi_index(coordinates, source.shape)
                    words = device.view(unsigned).reshape(-1)
                    words[cp.asarray(changed)] = cp.asarray(after[coordinates])
                else:
                    # Preserve a Fortran source's transfer order and perform
                    # the layout conversion on the GPU. A host C transpose
                    # here made the fine-class cache slower than a fresh build.
                    uploaded = cp.asarray(source, order='K')
                    cp.copyto(device.view(unsigned),cp.ascontiguousarray(uploaded.view(unsigned)))
        # Each namespace belongs to one upload worker. No array is modified
        # after return until that stream synchronizes and admission claims it.
        if self.claimed:
            # Admission consumes this cache once. Its next owner is the live
            # driver, so another host snapshot can never be reused.
            self.arrays.pop(key, None)
        else:
            self.arrays[key] = source.copy(order='K'), device
        return device

    def receipt(self):
        return dict(reused_bytes=self.reused_bytes, changed_bytes=self.changed_bytes,
                    fresh_bytes=self.fresh_bytes)


def upload(cp, source, name):
    active = ACTIVE.get()
    if active is None:
        return cp.asarray(source, order='K')
    cache, namespace = active
    return cache.upload(cp, namespace, name, source)
