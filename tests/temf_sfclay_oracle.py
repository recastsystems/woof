"""Reader for the TEMF surface-layer oracle fixtures of record.

``tools/temf_sfclay_wrf461_oracle/build.sh`` writes one input record and one
output record per arm, one row per (step, column), every value an 8-digit
hex binary32 word.  Two arms are kept:

* ``wrf``          -- the WRF v4.6.1 scalar build's own object
  (phys/module_sf_temfsfclay.o), linked against the C library;
* ``wrf-crlog10``  -- the same object with the C library's log10f replaced
  by a correctly rounded one (the arm WOOF's log10 is graded against).

The carried HFX and WM_TEMF columns of an arm's input record are that arm's
own previous outputs; every other input column is the same in both arms.
"""

from __future__ import annotations

import csv
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "temf_sfclay_wrf461"
ARMS = ("wrf", "wrf-crlog10")

#: Input record columns that are the step's EXTERNAL feed (atmosphere,
#: surface, and what the TEMF PBL hands the surface layer).
EXTERNAL_INPUTS = ("u", "v", "th", "qv", "p", "pi", "rho", "z", "ht", "psfc",
                   "znt", "mavail", "xland", "tsk", "fcor", "te", "hd")
#: Input record columns the surface layer carries itself.
CARRIED_INPUTS = ("hfx", "wm")
#: Output record columns, in file order.
OUTPUTS = ("ust", "hfx", "qfx", "lh", "flhc", "flqc", "chs", "chs2", "cqs2",
           "cpm", "qgh", "qsfc", "u10", "v10", "th2", "t2", "q2", "exch", "wm",
           "te", "znt")

#: Oracle catalogue (run_temf_sfclay.F90): 13 regimes x 6 variants.
COLUMNS_PER_REGIME = 6
REGIMES = ("convective", "stable", "neutral-windy", "calm", "sea", "rough",
           "very-stable", "dry-mavail0", "te-floor", "near-calm",
           "cold-or-equatorial", "hot-desert-low-level", "log10-words")


def regime_columns(regime: str) -> np.ndarray:
    """0-based column indices of one regime of the catalogue."""
    start = REGIMES.index(regime) * COLUMNS_PER_REGIME
    return np.arange(start, start + COLUMNS_PER_REGIME)


def _words(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    index = np.array([[int(r[0]), int(r[1])] for r in body], dtype=np.int64)
    words = np.array([[int(w, 16) for w in r[2:]] for r in body],
                     dtype=np.uint32)
    return header[2:], index, words


@lru_cache(maxsize=None)
def load(arm: str) -> dict[str, np.ndarray]:
    """``{"in:<name>"|"out:<name>": float32 (nstep, ncol)}`` for one arm."""
    if arm not in ARMS:
        raise ValueError(f"unknown oracle arm {arm!r}")
    result: dict[str, np.ndarray] = {}
    for kind, name in (("in", f"temf-inputs-{arm}.csv"),
                       ("out", f"temf-outputs-{arm}.csv")):
        header, index, words = _words(FIXTURE_DIR / name)
        nstep = int(index[:, 0].max())
        ncol = int(index[:, 1].max())
        if not np.array_equal(
                index, np.array([[s, c] for s in range(1, nstep + 1)
                                 for c in range(1, ncol + 1)])):
            raise ValueError(f"{name} is not one row per (step, column)")
        for j, field in enumerate(header):
            result[f"{kind}:{field}"] = (
                words[:, j].reshape(nstep, ncol).view(np.float32))
    return result


def dims(arm: str = "wrf") -> tuple[int, int]:
    """(steps, columns) of the fixture."""
    return load(arm)["in:u"].shape


def recorded_sha256() -> dict[str, str]:
    """The oracle's own sha256 receipt, by file name."""
    table = {}
    for line in (FIXTURE_DIR / "oracle-sha256sums.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        table[Path(name).name] = digest
    return table


def file_sha256(name: str) -> str:
    return hashlib.sha256((FIXTURE_DIR / name).read_bytes()).hexdigest()
