"""Measurement launcher (not product code): the hex CLI with host-wall timers around every physics driver op.

perf_counter before and after each PhysicsDriver._call (RRFS and v8.4.2 drivers); no device synchronisation is added,
so each number is the host time the op took to issue its work (plus any blocking read inside it). At exit one JSON
line per op (calls, median and mean ms) goes to OP_TIMER_OUT.
"""
import atexit
import functools
import json
import os
import statistics
import sys
import time

OUT = os.environ["OP_TIMER_OUT"]
TIMES = {}


def install():
    from woof.hex.mpasphys.rrfs import driver_rrfs as d
    from woof.hex.mpasphys import driver_v842 as v
    for mod in (d, v):
        original = mod.PhysicsDriver._call

        def wrap(original):
            @functools.wraps(original)
            def call(self, *a, **k):
                op = str(k.get("slot") or a[0])
                t0 = time.perf_counter()
                try:
                    return original(self, *a, **k)
                finally:
                    TIMES.setdefault(op, []).append(time.perf_counter() - t0)
            return call
        mod.PhysicsDriver._call = wrap(original)

    def dump():
        with open(OUT, "w") as stream:
            for op, values in sorted(TIMES.items()):
                rest = values[1:] or values
                stream.write(json.dumps({"op": op, "calls": len(values),
                                         "median_ms_after_first": round(1000 * statistics.median(rest), 3),
                                         "mean_ms_after_first": round(1000 * statistics.mean(rest), 3),
                                         "first_ms": round(1000 * values[0], 3)}) + "\n")
    atexit.register(dump)


if __name__ == "__main__":
    install()
    from woof.hex import cli
    a = cli.build_parser().parse_args(sys.argv[1:])
    cli._select_numerics(a)
    sys.exit(int(a.handler(a)))
