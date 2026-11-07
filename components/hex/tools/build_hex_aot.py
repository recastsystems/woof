#!/usr/bin/env python3
"""Build compiler-bound kernel packs that ship as release assets."""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="Release asset output (default: release-assets/kernel-packs/kernel-pack-cuda<N>.json)")
    parser.add_argument("--base-pack", type=Path,
                        help="Preserve this compiler-bound pack and build only missing selections")
    parser.add_argument("--prune", action="store_true",
                        help="Keep only the base-pack images this tree's inventory selects")
    parser.add_argument("--job-dir", type=Path,
                        help="Compile one process per image, admitted by measured memory, and keep each "
                             "finished image here so an interrupted build resumes")
    parser.add_argument("--reserve-gib", type=float, default=8.0,
                        help="With --job-dir: memory left free for the rest of the machine (default 8)")
    parser.add_argument("--physics-family", action="append",
                        help="Build only these direct physics families, under both numerics modes")
    parser.add_argument("--workers", type=int, help="Maximum concurrent compiler processes")
    args = parser.parse_args()
    from woof.hex.cuda_backend.aot_build import build, physics_inventory
    if args.output is None:
        from woof.hex.cuda_backend.aot import compiler_identity
        major = int(compiler_identity()["nvrtc"][0])
        args.output = (Path(__file__).resolve().parents[1] / "release-assets" / "kernel-packs"
                       / f"kernel-pack-cuda{major}.json")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    if args.prune and args.base_pack is None:
        parser.error("--prune needs --base-pack")
    entries = None
    if args.physics_family:
        if args.base_pack is None:
            parser.error("--physics-family requires --base-pack to preserve the shared pack")
        if args.prune:
            parser.error("--physics-family builds part of a shared pack; --prune would drop the rest")
        families = set(args.physics_family)
        available = [row for mode in ("strict", "fast") for row in physics_inventory(mode)]
        unknown = families - {row["physics_family"] for row in available}
        if unknown:
            parser.error("unknown physics families: " + ", ".join(sorted(unknown)))
        entries = [row for row in available if row["physics_family"] in families]
    print(json.dumps(build(args.output, entries=entries, base_pack=args.base_pack,
                           workers=args.workers, prune=args.prune, job_dir=args.job_dir,
                           reserve_bytes=int(args.reserve_gib * 2**30)), sort_keys=True))


if __name__ == "__main__":
    main()
