"""Build source-bound MSKF images in parallel before forecasts, without a GPU.

Direct-NVRTC requests contain the assembled engine headers and no CuPy include
root. The image therefore binds its actual source, options and compiler without
traversing unrelated toolkit headers such as nvToolsExt.h. Both aerosol tiers
and both arithmetic modes use exactly the runtime recipe.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def catalog(output):
    from woof.core.kernels.no_ftz import module_recipe_no_ftz
    from woof.kernel_aot import compiler_identity, effective_options, DIRECT_ROUTE
    from woof.kernel_pack import write
    rows = []
    for defines in ((), (("GPUWM_MSKF_NO_AEROSOL", 1),)):
        source, options = module_recipe_no_ftz("mskf", defines)
        rows.append(dict(route=DIRECT_ROUTE, source=source,
                         options=list(effective_options(options)), filename="mskf.cu"))
    write(output, compiler_identity(), rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args()
    if args.workers < 2:
        parser.error("use at least two independent compiler workers")
    if args.catalog:
        catalog(args.catalog)
        return
    args.work.mkdir(parents=True, exist_ok=True)
    catalogs = []
    for exact in (False, True):
        path = args.work / ("strict.json.xz" if exact else "fast.json.xz")
        env = dict(os.environ, GPUWM_WRF_EXACT=str(int(exact)),
                   GPUWM_NUMERICS="strict" if exact else "fast")
        subprocess.run([sys.executable, __file__, "--work", str(args.work),
                        "--catalog", str(path)], env=env, check=True)
        catalogs.append(path)
    try:
        from tools.build_gfdleta_pack import build
    except ImportError:
        from build_gfdleta_pack import build
    from woof.kernel_aot import compiler_identity, mskf_pack_path
    output = mskf_pack_path(compiler_identity())
    receipt = build(output, catalogs, args.work, args.workers)
    receipt.update(output=str(output), compiler=compiler_identity(), command=sys.argv)
    (args.work / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
