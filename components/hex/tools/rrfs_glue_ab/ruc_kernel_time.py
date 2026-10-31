"""Measurement launcher (not product code): CUDA-event times of the resident RUC boundary kernels.

Each boundary launch of resident_ruc_rrfs._Boundary is bracketed by two recorded CUDA events (no synchronisation on
the step path; the events are read at exit). RUC_BLOCK sets the launch block size (32 = the product launch); every
RUC kernel is one independent thread per cell (no shared memory, no barriers, no warp intrinsics), so the block size
cannot change a word. At exit: per entry the median/mean kernel ms over calls after the first, plus each kernel's
registers and local (spill/stack) bytes per thread, as JSON lines in RUC_TIME_OUT.
"""
import atexit
import json
import os
import statistics
import sys

OUT = os.environ["RUC_TIME_OUT"]
BLOCK = int(os.environ.get("RUC_BLOCK", "32"))
PENDING = []
ATTRS = {}


def install():
    import numpy as np
    import cupy as cp
    from cupy.cuda import driver
    from woof.hex.mpasphys.rrfs import resident_ruc_rrfs as rr

    def call(self):
        n = self.n
        if n == 0:
            return
        if self.preflight is not None:
            self.preflight((1,), (1,), (self.ptrs, self.tile_stop, np.int32(n)))
        if self.entry not in ATTRS:
            ATTRS[self.entry] = {"registers": driver.funcGetAttribute(4, self.kernel.ptr),
                                 "local_bytes": driver.funcGetAttribute(3, self.kernel.ptr),
                                 "max_threads_per_block": driver.funcGetAttribute(0, self.kernel.ptr)}
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        self.kernel(((n + BLOCK - 1) // BLOCK,), (BLOCK,),
                    (self.ptrs, self.tables, self.workspace, self.errors, np.int32(n)))
        stop.record()
        PENDING.append((self.entry, start, stop))

    rr._Boundary.__call__ = call

    def dump():
        times = {}
        for entry, start, stop in PENDING:
            stop.synchronize()
            times.setdefault(entry, []).append(cp.cuda.get_elapsed_time(start, stop))
        with open(OUT, "w") as stream:
            for entry, values in sorted(times.items()):
                rest = values[1:] or values
                stream.write(json.dumps({"entry": entry, "block": BLOCK, "calls": len(values),
                                         "median_ms_after_first": round(statistics.median(rest), 4),
                                         "mean_ms_after_first": round(statistics.mean(rest), 4),
                                         **ATTRS.get(entry, {})}) + "\n")
    atexit.register(dump)


if __name__ == "__main__":
    install()
    from woof.hex import cli
    a = cli.build_parser().parse_args(sys.argv[1:])
    cli._select_numerics(a)
    sys.exit(int(a.handler(a)))
