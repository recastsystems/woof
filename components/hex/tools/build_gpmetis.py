#!/usr/bin/env python3
"""Build hex_gpmetis, the METIS 5.1.0 partitioner the multi-card door runs.

THE BREAKAGE THIS PREVENTS (hex global on a rented 2x RTX PRO 6000 box,
2026-10-16, RESULTS.md defect 4): every automatic multi-card run needs a
contiguous METIS partition, and the door looked for ``gpmetis`` on PATH only.
A pip install carries none, so a two-card 30 km run refused until a JEDI spack
view was put on PATH by hand.  The wheel now carries this build (setup.py puts
it in ``woof/hex/libexec``) and the door runs it before any PATH copy.

What is compiled: ``third_party/metis-5.1.0`` (Apache-2.0; see its
HEX-VENDORING.txt), with the flags METIS's own CMake build uses on Linux
(``GKlib/GKlibSystem.cmake``: -O3 -std=c99 -fno-strict-aliasing -DLINUX
-D_FILE_OFFSET_BITS=64 -DNDEBUG -DNDEBUG2) and the 64-bit idx_t/real_t a spack
``metis@5.1.0+int64+real64`` build uses.  Its partition files are byte-identical
to that gpmetis's (HEX-VENDORING.txt names the measurement).

Stdlib only, so setup.py and the manylinux build image can run it.

    python tools/build_gpmetis.py --out DIR      # writes DIR/hex_gpmetis
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

SOURCE = Path(__file__).resolve().parents[1] / "third_party" / "metis-5.1.0"
NAME = "hex_gpmetis" + (".exe" if sys.platform == "win32" else "")
#: METIS 5.1.0's CMake flags for a GNU-compatible compiler, release (no asserts).
CFLAGS = ("-O3", "-std=c99", "-fno-strict-aliasing", "-DLINUX", "-D_FILE_OFFSET_BITS=64",
          "-DNDEBUG", "-DNDEBUG2", '-DSVNINFO=""', "-w")


def sources(root: Path = SOURCE) -> list[Path]:
    return sorted([*(root / "GKlib").glob("*.c"), *(root / "libmetis").glob("*.c"),
                   *(root / "programs").glob("*.c")])


def build(out_dir: Path, root: Path = SOURCE, cc: str | None = None) -> Path:
    """Compile every translation unit in parallel and link ``out_dir/hex_gpmetis``."""

    if sys.platform == "win32":
        raise RuntimeError("hex_gpmetis is built for Linux and macOS only (METIS needs GKlib's "
                           "LGPL regex on Windows, which this tree does not carry)")
    compiler = shlex.split(cc or os.environ.get("CC") or "cc")
    if shutil.which(compiler[0]) is None:
        raise RuntimeError(f"no C compiler ({compiler[0]}) on PATH")
    includes = [f"-I{root / part}" for part in ("include", "GKlib", "libmetis", "programs")]
    env = dict(os.environ)
    # METIS prints __DATE__/__TIME__ in its banner. A fresh build must not
    # acquire different binary bytes just because the wheel was built later.
    env.setdefault("SOURCE_DATE_EPOCH", "0")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hex-gpmetis-") as scratch:
        objects = []

        def compile_one(index_source):
            index, source = index_source
            target = Path(scratch) / f"{index:03d}-{source.parent.name}-{source.stem}.o"
            done = subprocess.run([*compiler, *CFLAGS, *includes, "-c", str(source), "-o", str(target)],
                                  capture_output=True, text=True, env=env)
            if done.returncode != 0:
                raise RuntimeError(f"{source}: {done.stderr.strip()[-2000:]}")
            return target

        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            objects = list(pool.map(compile_one, enumerate(sources(root))))
        binary = out_dir / NAME
        done = subprocess.run([*compiler, "-o", str(binary), *map(str, objects), "-lm"],
                              capture_output=True, text=True, env=env)
        if done.returncode != 0 or not binary.is_file():
            raise RuntimeError(f"linking {binary} failed: {done.stderr.strip()[-2000:]}")
    return binary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cc", default=None)
    args = parser.parse_args(argv)
    print(build(args.out, cc=args.cc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
