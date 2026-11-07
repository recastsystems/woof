"""Bitwise comparison of WOOF's WDM7 against the WRF v4.6.1 oracle.

usage: python3 compare.py FIXTURE_DIR REFERENCE_DIR WOOF_DIR [--json OUT]
                          [--columns-only REGEX] [--fortran-pair] [--skip F]

The WDM6 oracle's comparator (tools/wdm6_wrf461_oracle/compare.py), run on
the WDM7 layout: this module registers WDM7's oracle_io under the name the
WDM6 comparator imports, then loads and runs that comparator unchanged.
Every output field of every step (qh and the hail accumulators included) is
compared word for word; FP32 ULP distances use the repository's one
certified definition, and WRF's effective radii are converted to WOOF's
micron word by the one float32 multiply WRF's RRTMG applies.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import oracle_io  # noqa: E402,F401  (WDM7 layout, bound as 'oracle_io')


def _wdm6_comparator():
    path = _HERE.parent / "wdm6_wrf461_oracle" / "compare.py"
    spec = importlib.util.spec_from_file_location("wdm6_oracle_compare", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.OUTPUT_LEVEL_FIELDS != oracle_io.OUTPUT_LEVEL_FIELDS:
        raise RuntimeError("the WDM6 comparator did not bind the WDM7 layout")
    return module


compare_config = None


def main(argv=None) -> int:
    return _wdm6_comparator().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
