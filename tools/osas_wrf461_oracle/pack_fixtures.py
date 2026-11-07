"""Pack the CU_OSAS oracle's outputs into the test fixtures.

    python pack_fixtures.py CASES_DIR OUT_DIR

Reads ``out-k44.bin``, ``out-k29.bin``, their stage dumps and ``tables.bin``
from CASES_DIR (run_osas.F90's streams) and writes ``osas-k44.npz`` and
``osas-k29.npz`` (every call's inputs, outputs and stage records,
compressed), their ``-default`` twins (XKT2 from the default arm's draw) plus ``tables.json`` (the table digest the launcher pins, and
the index coefficients).  It also states the layout check: the 12 x 8 tile
and the 1 x 96 single-column layout of the k44 cases must agree word for
word, or nothing is packed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from woof.verify.osas_oracle import (TABLE_COEFFICIENTS, TABLE_SHAPES,
                                      read_run, read_stages, read_tables)


def _words(tables: dict) -> np.ndarray:
    parts = [np.array([tables[name] for name in TABLE_COEFFICIENTS], dtype="<f8")]
    parts += [np.asarray(tables[name], dtype="<f8") for name, _ in TABLE_SHAPES]
    return np.concatenate(parts)


def main(cases: str, out: str) -> None:
    cases_dir, out_dir = Path(cases), Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiled = read_run(cases_dir / "out-k44.bin")
    single = read_run(cases_dir / "out-k44-single.bin")
    for a, b in zip(tiled.calls, single.calls):
        for name in a.outputs:
            if a.outputs[name].tobytes() != b.outputs[name].tobytes():
                raise SystemExit(f"layout check failed on {name}")
    for grid in ("k44", "k29", "k44-default", "k29-default"):
        run = read_run(cases_dir / f"out-{grid}.bin")
        stages = read_stages(cases_dir / f"stage-{grid}.bin", run)
        arrays = {"im": run.im, "jm": run.jm, "kx": run.kx,
                  "ncall": len(run.calls)}
        for n, (call, stage) in enumerate(zip(run.calls, stages)):
            arrays[f"c{n}_scalars"] = np.array(
                [call.dt, call.stepcu, call.itimestep, call.chain], dtype=np.float64)
            for name, value in call.inputs.items():
                arrays[f"c{n}_in_{name}"] = value
            for name, value in call.outputs.items():
                arrays[f"c{n}_out_{name}"] = value
            for name, value in stage.items():
                arrays[f"c{n}_stage_{name}"] = value
        np.savez_compressed(out_dir / f"osas-{grid}.npz", **arrays)
    tables = read_tables(cases_dir / "tables.bin")
    words = _words(tables)
    receipt = {
        "sha256": hashlib.sha256(words.tobytes()).hexdigest(),
        "words": int(words.size),
        "coefficients": {name: tables[name].hex() for name in TABLE_COEFFICIENTS},
    }
    (out_dir / "tables.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps(receipt["sha256"]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
