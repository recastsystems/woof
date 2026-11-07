"""Capture one cycle analysis's inputs, then replay it under two settings.

(Was tools/da_analysis_capture.py on development branch; renamed on the
combined branch because the host-parallel lane's solve-seam capture holds
that path.)

The cycle driver's forecast legs are not reproduced bit for bit across
processes today, so two separate driver runs hand the analysis two slightly
different backgrounds and their increments cannot judge an analysis change.
This tool takes the background out of the comparison:

``capture``  runs ``tools.da_cycle_prepared`` in-process with
             :func:`woof.da.radar_assimilation.assimilate_radar_grid`
             wrapped so that each call's complete inputs are saved under
             ``--out`` (member checkpoints, every member's reflectivity
             H(x), the extra observation batches, the radar observation
             document path, the grid and the configuration) before the
             analysis runs as usual.

``replay``   reruns a captured call with ``--neighbor-search`` forward or
             index (and anything else that stays the same) and writes every
             member's increments and the call's provenance.

``compare``  reports, per field over all members, whether two replays are
             byte-identical, and the max and RMS of their difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def _capture(out: Path, argv):
    from woof.da import radar_assimilation as ra

    original = ra.assimilate_radar_grid
    calls = {"n": 0}

    def wrapped(checkpoints, observations, grid, cfg, **kwargs):
        slot = out / f"call{calls['n']:02d}"
        calls["n"] += 1
        (slot / "checkpoints").mkdir(parents=True)
        saved = {}
        for index, path in checkpoints.items():
            target = slot / "checkpoints" / f"member_{int(index):03d}.npz"
            shutil.copyfile(path, target)
            saved[int(index)] = str(target)
        provider = kwargs.get("reflectivity_provider")
        if provider is not None:
            dbz = {}
            for index, path in saved.items():
                state = ra.read_checkpoint_state(path)
                dbz[f"m{index:03d}"] = np.asarray(provider(index, state))
            np.savez(slot / "reflectivity.npz", **dbz)
        payload = {
            "checkpoints": saved,
            "observations": (str(observations)
                             if isinstance(observations, (str, Path))
                             else observations),
            "grid": grid, "cfg": cfg,
            "extra_obs": kwargs.get("extra_obs"),
            "extra_obs_provenance": kwargs.get("extra_obs_provenance"),
            "has_reflectivity": provider is not None,
            "other": sorted(k for k, v in kwargs.items() if v is not None
                            and k not in ("reflectivity_provider", "extra_obs",
                                          "extra_obs_provenance")),
        }
        with open(slot / "inputs.pkl", "wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"analysis capture: {slot}", flush=True)
        return original(checkpoints, observations, grid, cfg, **kwargs)

    ra.assimilate_radar_grid = wrapped
    sys.argv = ["tools.da_cycle_prepared", *argv]
    import runpy
    runpy.run_module("tools.da_cycle_prepared", run_name="__main__")


def _replay(slot: Path, out: Path, mode: str):
    from dataclasses import replace

    from woof.da import radar_assimilation as ra

    with open(slot / "inputs.pkl", "rb") as stream:
        payload = pickle.load(stream)
    if payload["other"]:
        raise SystemExit(f"capture carries inputs replay cannot rebuild: "
                         f"{payload['other']}")
    provider = None
    if payload["has_reflectivity"]:
        dbz = np.load(slot / "reflectivity.npz")

        def provider(index, state, _dbz=dbz):
            return _dbz[f"m{int(index):03d}"]

    real = ra._letkf_config

    def letkf_config(cfg, budget):
        return replace(real(cfg, budget), neighbor_search=mode)

    ra._letkf_config = letkf_config
    started = time.perf_counter()
    increments, prov = ra.assimilate_radar_grid(
        {int(k): v for k, v in payload["checkpoints"].items()},
        payload["observations"], payload["grid"], payload["cfg"],
        reflectivity_provider=provider, extra_obs=payload["extra_obs"],
        extra_obs_provenance=payload["extra_obs_provenance"])
    wall = time.perf_counter() - started
    out.mkdir(parents=True, exist_ok=True)
    fields = sorted(next(iter(increments.values())))
    np.savez(out / "increments.npz", **{
        f"{name}": np.stack([increments[i][name] for i in sorted(increments)])
        for name in fields})
    digest = hashlib.sha256()
    for name in fields:
        for i in sorted(increments):
            digest.update(np.ascontiguousarray(increments[i][name]).tobytes())
    try:
        import resource
        peak = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                     / 2**20, 2)
    except ImportError:          # no getrusage on Windows
        peak = None
    receipt = {"mode": mode, "wall_seconds": round(wall, 2),
               "peak_rss_gib": peak,
               "digest": digest.hexdigest(),
               "stage_wall_seconds": prov.get("stage_wall_seconds"),
               "filter": {k: prov["filter"].get(k) for k in (
                   "setup_seconds", "solve_seconds", "weights_seconds",
                   "transform_seconds", "finish_seconds", "active_points",
                   "max_local_obs", "neighbor_search", "neighbor_index")},
               "storage": prov.get("analysis_storage"),
               "innovations": prov.get("innovations")}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, default=str))
    print(json.dumps({k: receipt[k] for k in ("mode", "wall_seconds",
                                               "peak_rss_gib", "digest",
                                               "storage")}), flush=True)


def _compare(a: Path, b: Path):
    left, right = np.load(a / "increments.npz"), np.load(b / "increments.npz")
    rows = {}
    for name in sorted(left.files):
        x, y = left[name], right[name]
        d = x.astype(np.float64) - y.astype(np.float64)
        rows[name] = {"bitwise": bool(np.array_equal(x, y)),
                      "max_abs_diff": float(np.abs(d).max()),
                      "rms_diff": float(np.sqrt(np.mean(d * d))),
                      "increment_rms": float(np.sqrt(np.mean(
                          x.astype(np.float64) ** 2)))}
    print(json.dumps(rows, indent=1))
    return rows


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "capture":
        out = Path(argv[1])
        out.mkdir(parents=True, exist_ok=True)
        return _capture(out, argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("replay")
    rp.add_argument("slot", type=Path)
    rp.add_argument("out", type=Path)
    rp.add_argument("--neighbor-search", choices=("index", "forward"),
                    required=True)
    cp = sub.add_parser("compare")
    cp.add_argument("a", type=Path)
    cp.add_argument("b", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "replay":
        _replay(args.slot, args.out, args.neighbor_search)
    else:
        _compare(args.a, args.b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
