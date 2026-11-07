"""Compare fresh Fortran streams with every committed fixture word."""
import hashlib
import json
import sys
from pathlib import Path
from woof.verify.osas_oracle import read_run, read_stages

def main(cases, fixtures, out):
    import numpy as np
    cases, fixtures = Path(cases), Path(fixtures)
    report = dict(schema="osas-wrf461-fixture-receipt-v1", grids={})
    for grid in ("k44", "k29", "k44-default", "k29-default"):
        raw = read_run(cases / f"out-{grid}.bin")
        stages = read_stages(cases / f"stage-{grid}.bin", raw)
        checked = 0
        with np.load(fixtures / f"osas-{grid}.npz") as fx:
            assert len(raw.calls) == int(fx["ncall"])
            for n, (call, stage) in enumerate(zip(raw.calls, stages)):
                for kind, values in (("in", call.inputs), ("out", call.outputs),
                                     ("stage", stage)):
                    for name, value in values.items():
                        want = fx[f"c{n}_{kind}_{name}"]
                        assert value.dtype == want.dtype
                        assert value.shape == want.shape
                        assert value.tobytes() == want.tobytes(), (grid, n, kind, name)
                        checked += value.size
        report["grids"][grid] = dict(columns=raw.ncol, levels=raw.kx,
            calls=len(raw.calls), checked_words=checked,
            fixture_sha256=hashlib.sha256(
                (fixtures / f"osas-{grid}.npz").read_bytes()).hexdigest())
    tiled, single = read_run(cases / "out-k44.bin"), read_run(cases / "out-k44-single.bin")
    for a, b in zip(tiled.calls, single.calls):
        for name in a.outputs:
            assert a.outputs[name].tobytes() == b.outputs[name].tobytes(), name
    report["layout_check"] = "12x8 equals 1x96, every output word"
    report["table_sha256"] = hashlib.sha256((cases / "tables.bin").read_bytes()).hexdigest()
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
if __name__ == "__main__":
    main(*sys.argv[1:])
