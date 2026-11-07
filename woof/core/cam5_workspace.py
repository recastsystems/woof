"""Per-domain CAM column buffers; never shared between ensemble members."""
from __future__ import annotations

import cupy as cp
import numpy as np


class CamWorkspace:
    """Own stable device allocations and pointer tables for one domain.

    Allocate/warm outside graph capture. Reusing one owner on another card is
    refused: a pointer-table address belongs to its CUDA device. The caller
    owns the stream ordering, as for the rest of a DomainState.
    """
    def __init__(self):
        self.device = cp.cuda.Device().id
        self.arrays = {}
        self.tables = {}

    def array(self, name, shape, dtype, *, clear=False):
        if cp.cuda.Device().id != self.device:
            raise ValueError("CAM workspace belongs to another CUDA device")
        shape = tuple(shape) if not isinstance(shape, int) else (shape,)
        dtype = np.dtype(dtype)
        array = self.arrays.get(name)
        if array is None:
            array = self.arrays[name] = cp.empty(shape, dtype=dtype)
        elif array.shape != shape or array.dtype != dtype:
            raise ValueError(f"CAM workspace {name} changed its allocation contract")
        if clear:
            array.fill(0)
        return array

    def pointers(self, name, values):
        # Keep owners alive with their table. No H2D transfer when stable.
        values = tuple(values)
        key = tuple(v.data.ptr if v is not None else 0 for v in values)
        cached = self.tables.get(name)
        if cached is None or cached[0] != key:
            cached = self.tables[name] = (key, cp.asarray(key, dtype=cp.uint64), values)
        return cached[1]
