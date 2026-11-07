"""Turn dump_constants output into the [morrison_aero.*] tables of
woof/data/morrison/constants.toml.

usage: python constants_to_toml.py OUT_DIR > block.toml

OUT_DIR is build.sh's output directory.  The words come from the t0fix
build (D1 in kernels/morrison_aero.cu): WOOF ships the corrected melting
point in the activation thermodynamics.  The pinned build's words are
recorded beside them for every constant D1 changes, so the table states
both values and which one the kernel reads.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path


def read_dump(path: Path) -> dict[str, str]:
    words: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        name, word = line.split()
        words[name] = word.upper()
    return words


def decimal(word: str) -> str:
    value = struct.unpack(">f", bytes.fromhex(word))[0]
    return repr(value)


def main(out_dir: str) -> None:
    root = Path(out_dir)
    lines = [
        "",
        "# mp_physics=40 (Morrison two-moment with CESM-NCSU prescribed aerosol).",
        "# Every value is the binary32 word MORR_TWO_MOMENT_INIT_AERO holds after",
        "# the call, printed by tools/morrison_aero_wrf461_oracle/dump_constants",
        "# from phys/module_mp_morr_two_moment_aero.F at WRF commit",
        "# d66e442fccc04111067e29274c9f9eaccc3cef28, built -O0 by gfortran 13.3.",
        "# Values are the t0fix build: the activation reads the melting point",
        "# SVPT0 where WRF reads T0 = 300 K (declared divergence D1); entries",
        "# whose pinned WRF word differs carry it as <NAME>_PINNED.  The kernel",
        "# woof/core/kernels/morrison_aero.cu reads these words through",
        "# woof/core/morrison_aero.py; the decimal comment is informational.",
    ]
    for table, rimed in (("hail", 1), ("graupel", 0)):
        fixed = read_dump(root / f"constants-t0fix-rimed{rimed}.txt")
        pinned = read_dump(root / f"constants-pinned-rimed{rimed}.txt")
        if set(fixed) != set(pinned):
            raise SystemExit("pinned and t0fix dumps name different constants")
        lines.append("")
        lines.append(f"[morrison_aero.{table}]")
        lines.append(f"morr_rimed_ice = {rimed}")
        for name, word in fixed.items():
            lines.append(f'{name} = "0x{word}"  # {decimal(word)}')
            if pinned[name] != word:
                lines.append(f'{name}_PINNED = "0x{pinned[name]}"  '
                             f'# {decimal(pinned[name])}')
    for table, rimed in (("hail",1),("graupel",0)):
        lines.append(f"\n[morrison_aero.{table}.radar]")
        for line in (root / f"radar-rimed{rimed}.txt").read_text(encoding="ascii").splitlines():
            name,word=line.split()
            lines.append(f'{name} = "0x{word}"')
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: constants_to_toml.py OUT_DIR")
    main(sys.argv[1])
