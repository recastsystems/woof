"""CPU-only comparison with an unchanged base checkout, no CUDA context.

Run once per GPUWM_WRF_EXACT value. This proves source and option inertia;
it does not claim whole-forecast output identity.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

from woof.core import kernels
from woof.wrf_exact import ENABLED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    folder = args.base / "woof/core/kernels"
    spec = importlib.util.spec_from_file_location("tslist_base_kernel_loader", folder / "__init__.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    rows = []
    for path in sorted(folder.glob("*.cu")):
        name = path.stem
        before = base.module_source(name, kernel_dir=folder)
        after = kernels.module_source(name)
        old_options = base.module_options(name)
        new_options = kernels.module_options(name)
        if before != after or old_options != new_options:
            raise AssertionError(f"preexisting CUDA module changed: {name}")
        rows.append({"module": name, "source_sha256": hashlib.sha256(before.encode()).hexdigest(),
                     "options": old_options})
    receipt = {"schema": "arwen.tslist.loader-inertia.v1", "strict": ENABLED,
        "base_tip": subprocess.check_output(["git", "-C", str(args.base), "rev-parse", "HEAD"], text=True).strip(),
        "modules_compared": len(rows), "different_sources": 0, "different_options": 0,
        "whole_forecast_output_identity": False, "modules": rows}
    args.receipt.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "modules"}, indent=2))


if __name__ == "__main__":
    main()
