"""Option-off proof for ``mynn_fixes_v61``: a step on a shipped config, word for word.

THE BREAKAGE IT PREVENTS: an opt-in option that moves the default route.
``mynn_fixes_v61`` adds a kernel unit, a config field and a branch in the
MYNN surface-layer first-step block; with the option off a forecast must
write the same words as the tree before it.

Usage (one process per tree, then compare)::

    python tools/mynn_v61/option_off_compare.py dump OUT.npz \
        [--config configs/recipes/conus_hrrr_configuration.toml] [--steps 1]
        [--fixes] [--wrf461] [--cycling]
    python tools/mynn_v61/option_off_compare.py compare A.npz B.npz

``dump`` builds the step-1 smoke row of the config (tools/battery/step1_smoke:
the config's own physics on a 32x32 open flat grid, the WK82 builder), steps
it and writes every floating and integer array of the state.  ``--fixes``
selects the option; ``--wrf461`` first moves the row onto the WRF 4.6.1 MYNN
generations the option is admitted on (the HRRR recipe runs gsd_41 and the
gsl_wrf39 surface layer); ``--cycling`` marks the start cycled.
``compare`` counts differing words (raw bit patterns) per array.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def dump(args) -> int:
    import cupy as cp

    from tools.battery import step1_smoke
    from tilestream import harness, physics_inventory

    domains = step1_smoke._load_domains(ROOT / args.config)
    label, run = domains[0]
    cfg = dataclasses.replace(run, nx=args.nxy, ny=args.nxy,
                              **step1_smoke.OPEN_FLAT)
    changes = {}
    if args.wrf461:
        changes.update(bl_mynn_version="wrf_461", mynn_sfclay_variant="wrf_461")
    if args.fixes:
        changes["mynn_fixes_v61"] = True
    if args.cycling:
        changes["cycling"] = True
    if changes:
        cfg = dataclasses.replace(cfg, **changes)
        from woof.config import validate_run_config
        validate_run_config(cfg)
    state, driver = physics_inventory.default_builder(cfg, args.seed)
    step1_smoke._seed_static_leaf_area(driver)
    for _ in range(args.steps):
        harness.run_steps(state, cfg, 1)
    cp.cuda.Device().synchronize()
    arrays = {}
    for name, array in harness.state_arrays(state).items():
        kind = getattr(getattr(array, "dtype", None), "kind", None)
        if kind in ("f", "i", "u", "b"):
            arrays[name] = cp.asnumpy(array)
    print(f"{label}: bl_mynn_version={cfg.bl_mynn_version} "
          f"mynn_sfclay_variant={cfg.mynn_sfclay_variant} "
          f"mynn_fixes_v61={getattr(cfg, 'mynn_fixes_v61', None)} cycling={cfg.cycling} "
          f"arrays={len(arrays)} words={sum(a.size for a in arrays.values())}")
    np.savez(args.out, **arrays)
    return 0


def _words(array: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(array)
    if array.dtype.itemsize == 8:
        return array.view(np.uint64)
    if array.dtype.itemsize == 4:
        return array.view(np.uint32)
    if array.dtype.itemsize == 2:
        return array.view(np.uint16)
    return array.view(np.uint8)


def compare(args) -> int:
    a, b = np.load(args.a), np.load(args.b)
    names = sorted(set(a.files) | set(b.files))
    total = diff = 0
    missing = [n for n in names if n not in a.files or n not in b.files]
    changed = []
    for name in names:
        if name in missing:
            continue
        x, y = a[name], b[name]
        if x.shape != y.shape or x.dtype != y.dtype:
            changed.append((name, "shape/dtype", x.size))
            diff += x.size
            total += x.size
            continue
        n = int(np.count_nonzero(_words(x) != _words(y)))
        total += x.size
        diff += n
        if n:
            changed.append((name, n, x.size))
    print(f"arrays {len(names) - len(missing)} compared, missing {missing}")
    print(f"words {total}, different {diff}")
    for row in changed[:40]:
        print("  ", *row)
    return 0 if diff == 0 and not missing else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="option_off_compare")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dump")
    d.add_argument("out")
    d.add_argument("--config", default="configs/recipes/conus_hrrr_configuration.toml")
    d.add_argument("--steps", type=int, default=1)
    d.add_argument("--nxy", type=int, default=32)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--fixes", action="store_true")
    d.add_argument("--wrf461", action="store_true")
    d.add_argument("--cycling", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = parser.parse_args(argv)
    return dump(args) if args.command == "dump" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
