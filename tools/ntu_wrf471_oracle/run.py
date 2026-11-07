#!/usr/bin/env python3
"""Record the native initialization diagnostic, including an expected trap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import signal
import subprocess

from tools.ntu_wrf471_oracle.build import digest


def no_core() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    out = args.build_dir.resolve()
    build = json.loads((out / "build-receipt.json").read_text())
    executable = out / "init_witness"
    if digest(executable) != build["executable_sha256"]:
        raise ValueError("diagnostic executable differs from build receipt")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w") as log:
        result = subprocess.run([str(executable)], cwd=out, stdout=log,
                                stderr=subprocess.STDOUT, timeout=30,
                                preexec_fn=no_core)
    output = args.log.read_text()
    witnessed = (result.returncode == -signal.SIGFPE
                 and "init_aerosol" in output.lower()
                 and ":1618" in output)
    receipt = build | {
        "command": [str(executable)],
        "exit_code": result.returncode,
        "log": str(args.log.resolve()),
        "native_undefined_read_witnessed": witnessed,
        "result": ("SIGFPE on native unfilled RHO top level" if witnessed
                   else "expected native witness was not observed"),
        "replay_word_count": 0,
        "forecast_run": False,
        "gpu_run": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    # Success here means the diagnostic observed the expected native fault.
    # It never means that the physics passed a module replay.
    return 0 if witnessed else 1


if __name__ == "__main__":
    raise SystemExit(main())
