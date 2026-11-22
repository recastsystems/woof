#!/usr/bin/env python3
"""Build compiler-family packs from real WOOF-ArWen compiler request catalogs."""
import argparse
import json
from pathlib import Path

from woof.kernel_aot_build import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, action="extend", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-pack", type=Path)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    print(json.dumps(build(args.output, args.recipes, workers=args.workers,
                           base_pack=args.base_pack), sort_keys=True))


if __name__ == "__main__":
    main()
