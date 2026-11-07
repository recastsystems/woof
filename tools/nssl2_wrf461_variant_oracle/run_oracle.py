"""Run the official WRF v4.6.1 NSSL driver on every variant case.

Writes, per mode, ``<out>/<mode>.npz`` holding the float32 inputs (state and
environment, shaped ``(nx, nz)``), the run settings, and the exact float32
outputs (decoded from the oracle's hexadecimal images).

Usage:
    python run_oracle.py ORACLE_EXE OUT_DIR [--modes a,b] [--nsteps 3]
        [--dt 6] [--itimestep0 2]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases  # noqa: E402

OUTPUT_FIELDS = (
    "theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
    "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
    "qvolg", "qvolh", "qzr", "qzg", "qzh", "refl_10cm",
    "effc_m", "effi_m", "effs_m",
)
SURFACE_FIELDS = (
    "rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
    "graupelncv", "hailnc", "hailncv", "sr",
)


def _hex32(text: str) -> np.float32:
    return np.array([int(text, 16)], dtype=np.uint32).view(np.float32)[0]


def read_oracle_csv(path: Path, nx: int, nz: int, nsteps: int) -> dict:
    out = {name: np.zeros((nsteps, nx, nz), np.float32) for name in OUTPUT_FIELDS}
    out.update({name: np.zeros((nsteps, nx), np.float32) for name in SURFACE_FIELDS})
    seen = set()
    with path.open(newline="", encoding="ascii") as stream:
        for row in csv.DictReader(stream):
            step = int(row["step"]) - 1
            ix, k = int(row["ix"]) - 1, int(row["k"]) - 1
            key = (step, ix, k)
            if key in seen or not (0 <= step < nsteps and 0 <= ix < nx and 0 <= k < nz):
                raise ValueError(f"duplicate or invalid oracle row {key}")
            seen.add(key)
            for name in OUTPUT_FIELDS:
                out[name][step, ix, k] = _hex32(row[name].strip())
            if k == 0:
                for name in SURFACE_FIELDS:
                    out[name][step, ix] = _hex32(row[name].strip())
    if len(seen) != nsteps * nx * nz:
        raise ValueError(f"incomplete oracle output: {len(seen)} rows")
    return out


def run_mode(exe: Path, out_dir: Path, mode, nsteps: int, dt: float,
             itimestep0: int) -> Path:
    # The native process runs in its namelist directory. Resolve these
    # before changing cwd so relative CLI paths still name the same files.
    exe, out_dir = Path(exe).resolve(), Path(out_dir).resolve()
    state, env = cases.build_case(mode)
    nx, nz = state["theta"].shape
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    text_in = work / f"{mode.label}.txt"
    text_out = work / f"{mode.label}.csv"
    cases.write_oracle_input(text_in, mode, state, env, nsteps, itimestep0, dt)
    # The module opens ./namelist.input (empty here) for nssl_mp_params.
    (work / "namelist.input").touch()
    (work / "namelist.output").touch()
    proc = subprocess.run([str(exe), str(text_in), str(text_out)], cwd=work,
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0 or "NSSL2_VARIANT_WRF_COMPLETE" not in proc.stdout:
        raise RuntimeError(
            f"oracle failed for {mode.label}: rc={proc.returncode}\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    (work / f"{mode.label}.log").write_text(proc.stdout + proc.stderr)
    outputs = read_oracle_csv(text_out, nx, nz, nsteps)
    for name in cases.STATE_FIELDS:
        previous = np.concatenate((state[name][None, ...], outputs[name][:-1]), axis=0)
        outputs[f"tendency_{name}"] = (outputs[name] - previous) / np.float32(dt)
    target = out_dir / f"{mode.label}.npz"
    payload = {f"in_{k}": v for k, v in state.items()}
    payload.update({f"env_{k}": v for k, v in env.items()})
    payload.update({f"wrf_{k}": v for k, v in outputs.items()})
    payload["settings"] = np.array([nsteps, itimestep0], dtype=np.int64)
    payload["dt"] = np.float32(dt)
    payload["mode_label"] = np.array(mode.label)
    np.savez(target, **payload)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exe")
    parser.add_argument("out_dir")
    parser.add_argument("--modes", default="")
    parser.add_argument("--nsteps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=6.0)
    parser.add_argument("--itimestep0", type=int, default=2)
    args = parser.parse_args()
    labels = [m for m in args.modes.split(",") if m] or [
        m.label for m in cases.MODES]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for label in labels:
        target = run_mode(Path(args.exe), out_dir, cases.MODE_BY_LABEL[label],
                          args.nsteps, args.dt, args.itimestep0)
        print("ORACLE", label, target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
