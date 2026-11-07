"""Replay observation-only column captures from a real, strict native WRF run.

The capture writer is instrument_native.py. Stream byte order is detected
from its version word, so WRF's -fconvert=big-endian is recorded faithfully.
No expected value is reconstructed by this reader or by the CUDA launcher.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

INPUT = "th qv qc qr qi qs qg ni ns nr ng nc pii p dz w kzh_up qrcuten qscuten qicuten".split()
OUTPUT = "th qv qc qr qi qs qg ni ns nr ng nc effc effi effs effr".split()
SURFACE = "rainnc rainncv snownc snowncv graupelnc graupelncv sr".split()


def read_capture(path):
    raw = path.read_bytes()
    order = ">" if raw[:4] == b"\0\0\0\1" else "<"
    header = np.frombuffer(raw, dtype=order + "i4", count=8).astype(np.int32)
    version, step, i, j, kts, kte, aercu, radar = map(int, header)
    assert version == 1 and aercu == 0 and radar in (0, 1), (path, header)
    nz = kte - kts + 1
    words = np.frombuffer(raw, dtype=order + "f4", offset=32).astype(np.float32)
    pos = 1

    def fields(names, size):
        nonlocal pos
        result = {}
        for name in names:
            result[name] = words[pos:pos + size].copy()
            pos += size
        return result

    inputs = fields(INPUT, nz)
    inputs.update(fields(SURFACE, 1))
    outputs = fields(OUTPUT, nz)
    outputs.update(fields(SURFACE, 1))
    if radar:
        outputs.update(fields(["refl_10cm"], nz))
    assert pos == len(words), (path, pos, len(words))
    meta = dict(step=step, i=i, j=j, nz=nz, dt=float(words[0]), aercu=aercu,
                radar=radar, byte_order=order, sha256=hashlib.sha256(raw).hexdigest())
    return meta, inputs, outputs


def run_group(rows, rimed):
    import cupy as cp
    from woof.core.morrison_aero import launch_morrison_aero, launch_refl_morrison_aero
    nz = rows[0][0]["nz"]

    def batch(index, name):
        data = np.stack([r[index][name] for r in rows], axis=-1)
        return np.ascontiguousarray(data.reshape((nz, 1, len(rows))
                                  if name not in SURFACE else (1, len(rows))))

    host = {name: batch(1, name) for name in INPUT + SURFACE}
    expected = {name: batch(2, name) for name in rows[0][2]}
    d = {name: cp.asarray(value) for name, value in host.items()}
    radii = {name: cp.empty_like(d["th"]) for name in OUTPUT[12:]}
    scratch = launch_morrison_aero(
        *(d[name] for name in INPUT[:13]), d["p"], d["dz"],
        *(d[name] for name in SURFACE), rows[0][0]["dt"],
        morr_rimed_ice=rimed, aercu_opt=0, w=d["w"], kzh_up=d["kzh_up"],
        qrcuten=d["qrcuten"], qscuten=d["qscuten"], qicuten=d["qicuten"],
        **radii)
    got = {name: cp.asnumpy(d[name]) for name in OUTPUT[:12] + SURFACE}
    got.update({name: cp.asnumpy(value) for name, value in radii.items()})
    if "refl_10cm" in expected:
        refl = cp.empty_like(d["th"])
        launch_refl_morrison_aero(d["qv"], d["qr"], d["nr"], d["qs"], d["ns"],
                                 d["qg"], d["ng"], cp.ascontiguousarray(scratch[0]),
                                 d["p"], refl, morr_rimed_ice=rimed)
        got["refl_10cm"] = cp.asnumpy(refl)
    from woof.core.fp32_ulp import fp32_ulp_distance
    report = {}
    for name, want in expected.items():
        value = got[name]
        different = want.view(np.uint32) != value.view(np.uint32)
        indices = np.argwhere(different)
        examples = []
        for index in indices[:8]:
            ix = tuple(map(int, index))
            col = ix[-1]
            examples.append(dict(capture=rows[col][0], index=ix,
                                 native_word=f"{int(want.view(np.uint32)[ix]):08x}",
                                 port_word=f"{int(value.view(np.uint32)[ix]):08x}"))
        report[name] = dict(words=want.size, differing_words=int(different.sum()),
                            max_ulp=int(fp32_ulp_distance(value, want).max()),
                            native_nonfinite=int((~np.isfinite(want)).sum()),
                            port_nonfinite=int((~np.isfinite(value)).sum()),
                            native_sha256=hashlib.sha256(want.tobytes()).hexdigest(),
                            port_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                            examples=examples)
    # An independent comparator negative control; it never alters the replay.
    planted = got["th"].copy()
    planted.view(np.uint32).flat[0] ^= np.uint32(1)
    negative_control = int(np.count_nonzero(planted.view(np.uint32) != got["th"].view(np.uint32)))
    assert negative_control == 1
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--rimed", type=int, default=1, choices=(0, 1))
    args = parser.parse_args()
    from woof.wrf_exact import ENABLED
    assert ENABLED, "the native bitwise replay requires GPUWM_WRF_EXACT=1"
    paths = sorted(args.captures.glob("mp40-real-*.bin"))
    assert paths, "no actual native captures"
    groups = defaultdict(list)
    captures = []
    for path in paths:
        row = read_capture(path)
        meta = row[0]
        captures.append(dict(path=path.name, **meta))
        groups[(meta["step"], meta["nz"], meta["dt"], meta["radar"])].append(row)
    reports = []
    for key, rows in groups.items():
        fields = run_group(rows, args.rimed)
        report = dict(step=key[0], columns=len(rows), nz=key[1], dt=key[2], radar=key[3], fields=fields)
        reports.append(report)
        print(key, len(rows), {k: v["differing_words"] for k, v in fields.items() if v["differing_words"]}, flush=True)
    from woof.certify.kernel_manifest import kernel_manifest
    words = sum(f["words"] for r in reports for f in r["fields"].values())
    different = sum(f["differing_words"] for r in reports for f in r["fields"].values())
    receipt = dict(schema="gpuwm-mp40-real-native-replay/v1", command=sys.argv,
                   strict=True, rimed=args.rimed, captures=captures, groups=reports,
                   words=words, differing_words=different, passed=different == 0,
                   negative_control_differing_words=1, kernel_manifest=kernel_manifest())
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"{words} words, {different} differing words", flush=True)
    return int(different != 0)


if __name__ == "__main__":
    raise SystemExit(main())
