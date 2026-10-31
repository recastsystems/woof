"""One GPU, 64 x 64 x 40 warm bubble; first run: woof fetch-tables.

Run: python examples/domain_api_warm_bubble.py --seconds 900
The example uses Thompson (8), with no substitution when tables are missing.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from woof.domain import Domain, WarmBubble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=900.0)
    args = parser.parse_args()
    z = np.linspace(0.0, 20000.0, 401)
    # A conditionally unstable lower troposphere and stable stratosphere.
    theta = np.where(z <= 12000, 300 + 43 * (z / 12000) ** 1.25,
                     343 * np.exp(9.81 * (z - 12000) / (1004.5 * 213)))
    qv = np.minimum(0.014, 0.018 * np.exp(-z / 2500))
    domain = Domain.from_sounding(
        z=z, theta=theta, qv=qv, nx=64, ny=64, nz=40,
        dx=1000, dy=1000, ztop=20000, dt=3, microphysics=8,
        boundary="periodic", bubble=WarmBubble(amplitude=3),
        overrides={"km_opt": 4, "damp_opt": 3, "zdamp": 5000,
                   "dampcoef": 0.2, "diff_6th_opt": 2})
    import cupy as cp
    cp.cuda.get_current_stream().synchronize()
    initial = float(cp.max(domain.get("w")))
    count, peak = 0, initial
    started = time.perf_counter()
    while domain.time < args.seconds:
        domain.dt = min(3.0, args.seconds - domain.time)
        domain.step()
        count += 1
        if count % 20 == 0 or domain.time >= args.seconds:
            host = domain.to_host(domain.prognostic_names())
            if not all(np.isfinite(a).all() for a in host.values()):
                raise RuntimeError(f"nonfinite field at t={domain.time}")
            wmax = float(host["w"].max())
            peak = max(peak, wmax)
            print(f"time={domain.time:.1f}s max_w={wmax:.6f}m/s "
                  f"seconds_per_step={(time.perf_counter() - started) / count:.6f}", flush=True)
    cp.cuda.get_current_stream().synchronize()
    if peak <= initial:
        raise RuntimeError("warm bubble produced no updraft growth")
    print(f"finite=True steps={count} initial_max_w={initial:.6f} "
          f"peak_max_w={peak:.6f} final_time={domain.time:.1f} "
          f"seconds_per_step={(time.perf_counter() - started) / count:.6f}")


if __name__ == "__main__":
    main()
