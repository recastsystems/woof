#!/usr/bin/env python3
"""Compare every pre-existing CUDA module's composed source and options."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="4123f1916e3c95984862cd9b21f2fc1e8dd5f850")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from woof.core import kernels
    from woof import wrf_exact

    original = subprocess.check_output(["git", "show", f"{args.base}:woof/core/kernels/__init__.py"], cwd=ROOT, text=True)
    module = types.ModuleType("woof.core.kernels._lane_baseline")
    module.__file__ = kernels.__file__
    exec(compile(original, kernels.__file__, "exec"), module.__dict__)
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", args.base,
                                     "woof/core/kernels"], cwd=ROOT, text=True).splitlines()
    rows = []
    for name in (Path(p).stem for p in paths if p.endswith(".cu")):
        before, after = module.module_source(name), kernels.module_source(name)
        opt_before, opt_after = module.module_options(name), kernels.module_options(name)
        rows.append({"module": name, "bytes": len(after.encode()),
                     "source_sha256": hashlib.sha256(after.encode()).hexdigest(),
                     "equal": before == after and opt_before == opt_after})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"base": args.base, "strict": wrf_exact.ENABLED,
                                     "modules": rows}, indent=2) + "\n")
    print(json.dumps({"modules": len(rows), "different": sum(not r["equal"] for r in rows),
                      "receipt": str(args.output)}))
    if not all(r["equal"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
