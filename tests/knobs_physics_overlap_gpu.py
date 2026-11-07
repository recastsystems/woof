"""On-card marginal cloud fraction and overlap endpoints, not weather skill."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--value", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from woof.knobs import physics
    physics.activate({"physics.rrtmg.overlap_break_fraction": args.value})
    import cupy as cp
    ncol, nlay, ngpt = 2048, 5, 140
    # The same independent draws for every endpoint and both process runs.
    cdf = cp.asarray(np.random.default_rng(83047).random(
        (ngpt, nlay, ncol), dtype=np.float32))
    fraction = cp.full((ncol, nlay), np.float32(.5))
    mask = cp.empty((ngpt, ncol, nlay), dtype=cp.uint8)
    from woof.core.rrtmg_mcica import _mcica_gpu_module
    module = _mcica_gpu_module()
    module.get_function("rmcw_walk_mask")(
        ((ngpt*ncol+127)//128,), (128,), (np.int32(0), np.int32(ncol),
        np.int32(nlay), np.int32(ngpt), cdf, fraction, mask))
    host = cp.asnumpy(mask)
    marginal = host.mean(axis=(0, 1))
    joint = np.logical_and(host[:, :, :-1], host[:, :, 1:]).mean(axis=(0, 1))
    expected = .5 - .25*args.value
    assert np.max(abs(marginal-.5)) < .005, marginal
    assert np.max(abs(joint-expected)) < .005, (joint, expected)
    from woof.core.kernels import _KDIR
    source = physics.kernel_source("rrtmg_mcica_wrf",
        (_KDIR / "rrtmg_mcica_wrf.cu").read_text(encoding="utf-8"))
    args.out.write_text(json.dumps({"value": args.value, "marginal": marginal.tolist(),
        "joint": joint.tolist(), "expected_joint": expected,
        "mask_sha256": hashlib.sha256(host.tobytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "ngpt": ngpt, "ncol": ncol, "nlay": nlay}, indent=2)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
