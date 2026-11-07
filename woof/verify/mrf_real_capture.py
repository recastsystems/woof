"""Replay MRF's real native-WRF center-column capture through its CUDA path.

The fixture records float32 words before and after the unmodified physics
body. Its capture hook, native build flags and source digest are documented
in tools/mrf_wrf461_oracle/README.md. No weather arrays leave the device
inside the physics launcher; this offline proof reads the final outputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from woof import data_assets

import numpy as np

CAPTURE = data_assets.verify_oracle_path("mrf/oracle/native-real-capture.json")


def float_words(words):
    return np.asarray([int(word, 16) for word in words], dtype=np.uint32).view(np.float32)


def replay():
    import cupy as cp
    from woof.core.mrf import MRF_3D_INPUTS, MRF_3D_OUTPUTS, launch_mrf
    from woof.core.fp32_ulp import fp32_ulp_distance
    from woof.numerics import mode

    capture = json.loads(data_assets.require_verify_oracle(CAPTURE).read_text())
    records = []
    for rec in capture["records"]:
        nz = rec["nz"]
        inputs = {key: cp.asarray(float_words(value).reshape(
            (nz, 1, 1) if len(value) == nz else (1, 1)))
                  for key, value in rec["inputs"].items()}
        outputs = launch_mrf(*(inputs[key] for key in MRF_3D_INPUTS),
            **{key: value for key, value in inputs.items() if key not in MRF_3D_INPUTS},
            dt=float(float_words(rec["dt"])[0]), flag_qi=rec["flag_qi"])
        outputs["ust"] = inputs["ust"]
        fields = {}
        for name in MRF_3D_OUTPUTS + ("pblh", "hol", "ust"):
            if name == "rqiblten" and not rec["flag_qi"]:
                continue  # native MRF leaves this output undefined
            got = cp.asnumpy(outputs[name]).ravel()
            expected = float_words(rec["outputs"][name])
            fields[name] = {"words": int(got.size), "differing_words": int(
                np.count_nonzero(got.view(np.uint32) != expected.view(np.uint32))),
                "max_ulp": int(fp32_ulp_distance(got, expected).max())}
        native_kpbl = int(rec["outputs"]["kpbl_native_top_down"][0], 16)
        gpu_native_kpbl = int(cp.asnumpy(outputs["kpbl_native"])[0, 0])
        kpbl = int(cp.asnumpy(outputs["kpbl"])[0, 0])
        fields["kpbl"] = {"words": 1, "differing_words": int(gpu_native_kpbl != native_kpbl),
                          "native_top_down": native_kpbl, "gpu_native_top_down": gpu_native_kpbl,
                          "gpu_bottom_up": kpbl}
        assert kpbl == nz + 1 - gpu_native_kpbl
        records.append({"call": rec["call"], "fields": fields})
    return {"schema": "mrf-native-real-replay-v1", "numerics": mode(),
            "capture_sha256": capture["capture_sha256"], "records": records,
            "compared_words": sum(v["words"] for r in records for v in r["fields"].values()),
            "differing_words": sum(v["differing_words"] for r in records for v in r["fields"].values())}


if __name__ == "__main__":
    import sys
    result = replay()
    text = json.dumps(result, indent=2) + "\n"
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text)
    print(text, end="")
    raise SystemExit(bool(result["differing_words"]))
