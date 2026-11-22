"""Grade the YSU port on real columns recorded inside a native WRF run.

    python -m tools.ysu_generations_oracle.real_capture.replay_real \
        --capture RUN_DIR --referee-oracle REF/oracle \
        --oracles BUILD (build.sh output) --version wrf_480 --out OUT

Per captured step (ysucap-stepN-in.bin / -out.bin, written by
instrument_pbl_driver.py in a copy of the referee build):
  1. driver fidelity: the referee-object oracle on the recorded inputs must
     write the recorded in-run words (every field the in-run capture keeps;
     ttnp and qmixtnp are not recorded in-run);
  2. build.sh's scalar oracle (same source, its own compile) must write the
     referee-object oracle's words, every word;
  3. the traced build gives the top-down diagnostics (active, capped,
     below cap), checked word for word against the untraced build;
  4. the GPU port is graded against the IN-RUN words
     (tools.ysu_generations_oracle.gate.grade), in this process's
     arithmetic mode.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from woof.verify.ysu_wide_oracle import load_wide_oracle, run_port_wide
from tools.ysu_generations_oracle.gate import grade

TAG = {"wrf_480": "v4.8.0", "wrf_461": "v4.6.1"}


def record_slices(raw_in, raw_out):
    """(output offset, nz) of each column record."""
    pos = opos = 0
    out = []
    while pos < len(raw_in):
        nz = int(np.frombuffer(raw_in, "<i4", 1, pos)[0])
        pos += 16 + 18 * 4 + 11 * 4 * nz + 4 * (nz + 1)
        out.append((opos, nz))
        opos += 24 + 40 * nz
    assert opos == len(raw_out), (opos, len(raw_out))
    return out


def recorded_mask(raw_in, raw_out):
    """True on the output bytes the in-run capture records (not ttnp/qmixtnp)."""
    mask = np.ones(len(raw_out), bool)
    for opos, nz in record_slices(raw_in, raw_out):
        lv = opos + 24
        mask[lv + 2 * 4 * nz: lv + 3 * 4 * nz] = False   # ttnp
        mask[lv + 9 * 4 * nz: lv + 10 * 4 * nz] = False  # qmixtnp
    return mask


def word_diff(a, b, mask=None):
    wa = np.frombuffer(a, "<u4")
    wb = np.frombuffer(b, "<u4")
    assert wa.size == wb.size, (wa.size, wb.size)
    if mask is not None:
        m = mask.reshape(-1, 4).all(axis=1)
        wa, wb = wa[m], wb[m]
    return {"words": int(wa.size), "different": int((wa != wb).sum())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--referee-oracle", type=Path, required=True)
    p.add_argument("--oracles", type=Path, required=True)
    p.add_argument("--version", default="wrf_480", choices=tuple(TAG))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)
    from woof.wrf_exact import ENABLED
    tag = TAG[args.version]
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"version": args.version, "strict_arithmetic": bool(ENABLED), "steps": [], "pass": True}
    captures = sorted(args.capture.glob("ysucap-step*-in.bin"),
                      key=lambda q: int(q.name.split("step")[1].split("-")[0]))
    assert captures, f"no ysucap-step*-in.bin in {args.capture}"
    for inp in captures:
        step = int(inp.name.split("step")[1].split("-")[0])
        work = args.out / f"step{step}"
        work.mkdir(exist_ok=True)
        raw_in = inp.read_bytes()
        out_file = inp.with_name(f"ysucap-step{step}-out.bin")
        inrun = out_file.read_bytes()
        shutil.copyfile(inp, work / "ysu-wide-inputs.bin")
        row = {"step": step, "inputs_sha256": hashlib.sha256(raw_in).hexdigest(),
               "inrun_sha256": hashlib.sha256(inrun).hexdigest()}
        subprocess.run([str(args.referee_oracle), str(work / "ysu-wide-inputs.bin"), str(work / "referee.bin")],
                       check=True, stdout=subprocess.DEVNULL)
        ref = (work / "referee.bin").read_bytes()
        row["driver_vs_inrun"] = word_diff(ref, inrun, recorded_mask(raw_in, inrun))
        subprocess.run([str(args.oracles / tag / "scalar/oracle"), str(work / "ysu-wide-inputs.bin"),
                        str(work / "scalar.bin")], check=True, stdout=subprocess.DEVNULL)
        row["scalar_vs_referee"] = word_diff((work / "scalar.bin").read_bytes(), ref)
        trace = subprocess.check_output([str(args.oracles / tag / "diagnostics/oracle"),
                                         str(work / "ysu-wide-inputs.bin"), str(work / "trace.bin")], text=True)
        row["traced_vs_referee"] = word_diff((work / "trace.bin").read_bytes(), ref)
        active = capped = below = 0
        for line in trace.splitlines():
            if line.startswith("YSUTD "):
                active += 1
                tmp1 = np.asarray([int(line.split()[1], 16)], np.uint32).view(np.float32)[0]
                v = np.float32(np.float32(.2) * np.float32(8.) * tmp1)
                capped += int(v >= np.float32(.4))
                below += int(v < np.float32(.4))
        row["topdown"] = {"active_columns": active, "capped": capped, "below_cap": below}
        shutil.copyfile(out_file, work / "ysu-wide-wrf.bin")
        cols = load_wide_oracle(work)
        row["columns"] = len(cols)
        row["t_phy_not_theta_times_pi_words"] = sum(
            int((np.float32(c.levels["theta"] * c.levels["pi"]).view(np.uint32)
                 != c.levels["tx"].view(np.uint32)).sum()) for c in cols)
        row["ctopo_not_one"] = sum(int(c.scalars["ctopo"] != 1 or c.scalars["ctopo2"] != 1) for c in cols)
        row["uoce_voce_nonzero"] = sum(int(c.scalars["uox"] != 0 or c.scalars["vox"] != 0) for c in cols)
        if args.port:
            port = run_port_wide(cols, ysu_version=args.version)
            fields, states, cells = grade(cols, port, step=step, allow_declarations=False)
            row["port_fields"] = fields
            row["port_states"] = states
            row["port_words"] = sum(v["words"] for v in fields.values())
            row["port_different"] = sum(v["different"] for v in fields.values())
            (work / "cells.json").write_text(json.dumps(cells[:2000]))
        ok = (row["driver_vs_inrun"]["different"] == 0 and row["scalar_vs_referee"]["different"] == 0
              and row["traced_vs_referee"]["different"] == 0
              and row["t_phy_not_theta_times_pi_words"] == 0 and row["ctopo_not_one"] == 0
              and row["uoce_voce_nonzero"] == 0
              and (not args.port or row["port_different"] == 0))
        row["pass"] = ok
        report["pass"] &= ok
        report["steps"].append(row)
        for name in ("referee.bin", "scalar.bin", "trace.bin", "ysu-wide-inputs.bin", "ysu-wide-wrf.bin"):
            (work / name).unlink()
        print(json.dumps({k: v for k, v in row.items() if k not in ("port_fields", "port_states")}), flush=True)
    (args.out / "summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"pass": report["pass"], "strict_arithmetic": report["strict_arithmetic"],
                      "columns": sum(s["columns"] for s in report["steps"]),
                      "port_words": sum(s.get("port_words", 0) for s in report["steps"]),
                      "port_different": sum(s.get("port_different", 0) for s in report["steps"])}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
