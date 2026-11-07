"""Replay the lane's instrumented native WRF pre/post module records.

Usage: python replay_native.py CAPTURE_DIRECTORY MANIFEST_JSON RECEIPT_JSON
The instrumenter records little-endian float32/int32 words without record
markers. This tool never changes captured inputs or masks differing outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def decode(path, spec):
    raw = path.read_bytes()
    offset = 0

    def take(count, dtype="<f4"):
        nonlocal offset
        out = np.frombuffer(raw, dtype=dtype, count=count, offset=offset).copy()
        offset += count * 4
        return out

    n, nz = (int(x) for x in take(2, "<i4"))
    ny = int(take(1, "<i4")[0]) if spec.get("full_tile") else 1
    nx = n
    n *= ny
    scalars = {name: take(1, "<i4" if name in
                           ("ITIMESTEP", "STEPBL", "KRR") else "<f4")[0]
               for name in spec["scalar"]}
    states = []
    for _ in range(2):
        fields = {name.lower(): take(n, "<i4" if name == "KPBL" else "<f4")
                  .reshape(ny, nx) for name in spec["s2"]}
        fields.update({name.lower(): np.ascontiguousarray(take(n * nz).reshape(ny, nz, nx).transpose(1, 0, 2))
                       for name in spec["s3"]})
        fields["pint"] = np.ascontiguousarray(take(n * (nz + 1)).reshape(ny, nz + 1, nx).transpose(1, 0, 2))
        states.append(fields)
    if offset != len(raw):
        raise ValueError(f"{path}: decoded {offset} bytes, file has {len(raw)}")
    return n, nz, scalars, *states


def replay(kind, n, nz, scalars, before):
    import cupy as cp
    from woof.core import myjpbl as pbl, myjsfc as sfc
    from woof.core.kernels import get_kernel

    dev = {name: cp.asarray(value) for name, value in before.items()}
    aliases = {"ust": "ustar", "p": "pmid", "qc": "cwm", "lh": "flx_lh",
               "t2": "t02", "th2": "th02", "q2": "q02"}

    def field(name):
        return dev[aliases.get(name, name)]

    if kind == "sfc":
        cols = {"dz": dev["dz"], "tke": dev["q2"]}
        surface = {name: dev[source][0] for name, source in
                   (("u1", "u"), ("v1", "v"), ("t1", "t"),
                    ("th1", "th"), ("qv1", "qv"), ("qc1", "qc"),
                    ("p1", "pmid"), ("psfc", "pint"))}
        surface.update({name: dev[name] for name in
                        ("tsk", "xland", "mavail", "z0base", "ht")})
        state = {name: field(name) for name in sfc.MYJ_SFCLAY_INOUT}
        outputs = {name: field(name) for name in sfc.MYJ_SFCLAY_OUTPUTS}
        sfc.launch_myj_sfclay(cols, surface, state, outputs,
                             itimestep=int(scalars["ITIMESTEP"]),
                             sf_sfclay_physics=4)
        selected = {aliases.get(name, name): array for name, array in
                    {**state, **outputs}.items()}
    else:
        dt = np.float32(scalars["DT"]) * np.float32(scalars["STEPBL"])
        if kind == "pbl":
            surface = {name: field(name) for name in pbl.QNSE_SURFACE_INPUTS
                       if name != "psfc"}
            surface["psfc"] = dev["pint"][0]
            args = (tuple(field(name) for name in pbl.QNSE_COLUMN_INPUTS)
                    + (dev["tke"],)
                    + tuple(surface[name] for name in pbl.QNSE_SURFACE_INPUTS)
                    + tuple(field(name) for name in pbl.QNSE_PBL_INOUT)
                    + tuple(field(name) for name in pbl.QNSE_PBL_COLUMN_OUTPUTS)
                    + tuple(field(name) for name in pbl.QNSE_PBL_SURFACE_OUTPUTS)
                    + (dt, np.int32(nz), np.int32(n)))
            selected = {name: field(name) for name in
                        (*pbl.QNSE_PBL_INOUT, *pbl.QNSE_PBL_COLUMN_OUTPUTS,
                         *pbl.QNSE_PBL_SURFACE_OUTPUTS, "tke")}
            entry = "qnsepbl_column"
        else:
            # KRR=2 is the EM driver contract implemented by this module.
            if int(scalars["KRR"]) != 2:
                raise ValueError(f"native MFSHCONVPBL KRR={scalars['KRR']}")
            args = (tuple(field(name) for name in pbl.MFSHCONV_COLUMN_INPUTS)
                    + (dev["tke"],)
                    + tuple(field(name) for name in pbl.MFSHCONV_SURFACE_INPUTS)
                    + (dev["rc_mf"],)
                    + tuple(field(name) for name in pbl._QNSE_TENDENCIES)
                    + tuple(field(name) for name in pbl.MFSHCONV_OUTPUTS_3D)
                    + (dev["wthv"], dev["plm_bl89"])
                    + (dt, np.int32(1), np.int32(nz), np.int32(n)))
            selected = {name: dev[name] for name in
                        (*pbl._QNSE_TENDENCIES, *pbl.MFSHCONV_OUTPUTS_3D,
                         "rc_mf", "wthv", "plm_bl89")}
            entry = "mfshconv_column"
        kernel = get_kernel("qnsepbl", entry)
        if kind == "mf":
            tile_top = cp.zeros(1, dtype=cp.int32)
            kernel(((n + 127) // 128,), (128,), args + (tile_top, np.int32(1)))
            kernel(((n + 127) // 128,), (128,), args + (tile_top, np.int32(2)))
        else:
            kernel(((n + 127) // 128,), (128,), args)
    return {name: cp.asnumpy(value) for name, value in selected.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    import cupy as cp
    from woof.core import kernels
    from woof.verify.qnsesfc_oracle import _stats
    from woof.wrf_exact import ENABLED, referee_correctly_rounded_trig
    report = {"strict": ENABLED,
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "source_sha256": {name: hashlib.sha256(kernels.module_source(name).encode()).hexdigest()
                                for name in ("qnsepbl", "qnsesfc")},
              "correctly_rounded_trig": referee_correctly_rounded_trig(),
              "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
              "module_options": {name: kernels.module_options(name) for name in
                                 ("qnsepbl", "qnsesfc")}, "rows": {}}
    for kind, spec in json.loads(args.manifest.read_text()).items():
        path = args.captures / f"capture-{kind}.bin"
        n, nz, scalars, before, after = decode(path, spec)
        got = replay(kind, n, nz, scalars, before)
        fields = {name: _stats(value, after[name]) for name, value in got.items()}
        report["rows"][kind] = {
            "capture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "columns": n, "levels": nz,
            "scalars": {name: value.item() for name, value in scalars.items()},
            "fields": fields,
            "words": sum(row["count"] for row in fields.values()),
            "different_words": sum(row["mismatch"] for row in fields.values())}
    args.receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: {k: row[k] for k in ("words", "different_words")}
                      for key, row in report["rows"].items()}, indent=2))
    if not ENABLED or any(row["different_words"] for row in report["rows"].values()):
        raise SystemExit("native module proof requires strict numerics and zero differing words")


if __name__ == "__main__":
    main()
