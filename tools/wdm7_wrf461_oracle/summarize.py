"""Summarise compare.py JSON results: per configuration, the columns that
differ and each field's largest absolute difference.

usage: python3 summarize.py RESULT.json [RESULT.json ...]
"""

from __future__ import annotations

import json
import sys

FIELDS = ("th", "qv", "qc", "qr", "qi", "qs", "qg", "qh", "nn", "nc", "nr",
          "re_cloud", "re_ice", "re_snow", "refl_10cm", "rain", "rainncv",
          "snow", "snowncv", "graupel", "graupelncv", "hail", "hailncv", "sr")


def main(argv):
    for path in argv[1:]:
        print(f"== {path}")
        for c in json.load(open(path, encoding="utf-8")):
            f = c["fields"]
            parts = [f"{k}={f[k]['max_abs']:.3g}" for k in FIELDS
                     if k in f and f[k]["differ"]]
            print(f"  {c['config']:16s} columns={len(c['columns_differing']):3d}"
                  f"/{c['ncol']} differ={c['differ']}/{c['values']} "
                  f"max_ulp={c['max_ulp']} " + " ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
