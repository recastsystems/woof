#!/usr/bin/env python3
"""Analytic branch checks of native PSD hail. This is not forecast proof."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import cupy as cp
    from woof import wrf_exact
    from woof.core import nwp_hail
    from woof.core.kernels import module_source
    if not wrf_exact.ENABLED:
        raise ValueError("start with GPUWM_WRF_EXACT=1")
    nz, ny, nx = 6, 3, 5
    rng = np.random.default_rng(4713)
    shape = (nz, ny, nx)
    data = {
        "qg": np.geomspace(1.e-9, .015, np.prod(shape)).astype("f4").reshape(shape),
        "rho": rng.uniform(.1, 1.3, shape).astype("f4"),
        "qh": rng.uniform(1.e-5, .01, shape).astype("f4"),
        "ng": rng.uniform(.1, 2.e4, shape).astype("f4"),
        "nh": rng.uniform(.1, 1.e4, shape).astype("f4"),
        "qr": rng.uniform(0., .01, shape).astype("f4"),
        "nr": rng.uniform(1., 1.e6, shape).astype("f4"),
    }
    # Includes native GH density rejection and clipping at both density limits.
    data["volume"] = (data["qg"] / np.linspace(10., 1000., np.prod(shape)).reshape(shape)).astype("f4")
    packed = np.zeros((nx, nz+1, ny, 8), dtype="f4", order="F")
    for i, key in enumerate(("qg", "rho", "qh", "ng", "nh", "volume", "qr", "nr")):
        packed[:, :nz, :, i] = data[key].transpose(2, 0, 1)
    device = {k: cp.asarray(v) for k, v in data.items()}
    lib = ctypes.CDLL(str(args.library.resolve()))
    lib.nwp_hail_ref.restype = None
    fptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    iptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    cases = [
        ("no-graupel", 1, {}), ("Lin", 2, {}), ("WSM6", 6, {}),
        ("WDM6-hail", 16, {"mpuse_hail": 1}),
        ("GSFC-graupel", 97, {}), ("GSFC-hail", 97, {"gsfcgce_hail": 1}),
        ("GSFC-no-graupel", 97, {"gsfcgce_2ice": 1}),
        ("Thompson-single", 8, {"qr": device["qr"], "nr": device["nr"]}),
        ("Thompson-number", 8, {"ng": device["ng"]}),
        ("Thompson-aero", 28, {"ng": device["ng"], "nh": device["nh"]}),
        ("Thompson-GH", 38, {"ng": device["ng"], "volume": device["volume"]}),
        ("Morrison", 10, {"ng": device["ng"]}),
        ("Morrison-aero-hail", 40, {"ng": device["ng"], "mpuse_hail": 1}),
        ("Milbrandt-hail", 9, {"qh": device["qh"], "nh": device["nh"]}),
        ("NSSL-multimoment-native-inactive", 18, {}),
        ("NSSL-single", 18, {"nssl_single_moment": nwp_hail.NsslSingleMoment(4.e5, 4.e4, 500., 900., 0., 1.)}),
        ("NSSL-single-hail", 18, {"nssl_single_moment": nwp_hail.NsslSingleMoment(4.e5, 4.e4, 500., 900., 0., 1., 1)}),
    ]
    results = []
    for name, mp, kw in cases:
        flags = np.array([key in kw for key in ("qh", "ng", "nh", "volume")], dtype="i4")
        nssl = kw.get("nssl_single_moment")
        options = np.array([kw.get("mpuse_hail", 0), kw.get("gsfcgce_hail", 0),
                            kw.get("gsfcgce_2ice", 0), nssl is None, nssl.hail if nssl else 0], dtype="i4")
        constants = np.array([getattr(nssl, key) if nssl else 0. for key in
                             ("cnoh", "cnohl", "rho_qh", "rho_qhl", "alphah", "alphahl")], dtype="f4")
        native = np.zeros((nx, ny, 2), dtype="f4", order="F")
        lib.nwp_hail_ref(ctypes.c_int(nx), ctypes.c_int(ny), ctypes.c_int(nz), fptr(packed),
                         fptr(native), ctypes.c_int(mp), iptr(flags), iptr(options), fptr(constants))
        out1, out2 = cp.zeros((ny, nx), dtype="f4"), cp.zeros((ny, nx), dtype="f4")
        nwp_hail.update(device["qg"], device["rho"], out1, out2, mp_physics=mp, **kw)
        observed = np.stack([cp.asnumpy(out1), cp.asnumpy(out2)])
        expected = np.ascontiguousarray(native.transpose(2, 1, 0))
        diff = expected.view("u4") != observed.view("u4")
        row = dict(case=name, mp_physics=mp, optional_species=flags.tolist(),
                   options=options.tolist(), constants=constants.tolist(),
                   words=int(expected.size), differing_words=int(diff.sum()))
        if diff.any():
            ix = tuple(map(int, np.argwhere(diff)[0]))
            row["first_difference"] = dict(index=ix, native=float(expected[ix]), cuda=float(observed[ix]),
                                            native_word=int(expected.view("u4")[ix]), cuda_word=int(observed.view("u4")[ix]))
        results.append(row)
    receipt = dict(schema="wrf471-nwp-hail-development-v1", qualification="none; analytic inputs",
                   native_build=json.loads((args.library.parent/"nwp-build.json").read_text()), results=results,
                   module_source_sha256=hashlib.sha256(module_source("nwp_hail").encode()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2)+"\n")
    differing = sum(r["differing_words"] for r in results)
    print(json.dumps(dict(words=sum(r["words"] for r in results), differing_words=differing,
                          receipt=str(args.output))))
    if differing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
