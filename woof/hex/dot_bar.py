"""The window check's verdict: one definition, used by every door that judges the whole-state dot.

The window check compares two dot products of the same float32 window, <y, M x> (tangent) and <M^T y, x>
(adjoint). The model and its derivative run in single precision, so the two sides cannot agree to better than the
single-precision rounding of the words they are made of. The verdict is therefore stated in roundings of the dot:

    one rounding  rho = 2^-23 (float32 epsilon) x max(|<y, M x>|, |<M^T y, x>|)
    reading       r   = (<M^T y, x> - <y, M x>) / rho          (signed)

Both sums are binary64 in the runner (each float32 word is widened before its product; window.rs win_dot2_f32), so
the float32 rounding is in the words, not in the sums. For this definition |r| is exactly rel / 2^-23, where rel is
the runner's rel_aligned of the self form (|a - b| / max(|a|, |b|)).

The bar (the table's ``thresholds.self_dot_bar``, setting ``roundings``, the default):
  (a) every direction within ``each_within_roundings`` (4) roundings, about 4.8e-07 relative;
  (b) over at least ``drift.min_directions`` (8) directions, no one-signed drift: the mean reading lies within
      ``drift.standard_errors`` (2) standard errors of zero.
The setting ``strict`` keeps the earlier bar, 1e-07 relative (0.84 of one rounding), for anyone who wants it.

THE BREAKAGE THIS PREVENTS: a bar below one rounding measured luck, not the derivative. The released, unchanged
build scored on the model's fields reads 0.3 to 0.6 roundings rms, passes 12 of 12 directions at 12 km, yet missed
one 3 km direction in five at 1.18 roundings (1.406e-07), and the README command passed on one install (8.575e-08)
and failed on another (2.046e-07, 1.72 roundings). A real wrong term still fails: the cumulus port's window reads
128 to 705 roundings (fails a), the 30-step 12 km window reads seven of eight directions positive (fails b).
The gradient test against the real model is a separate check with its own 1 per cent bar; nothing here changes it.
"""
from __future__ import annotations

import math
import os

#: float32 epsilon, 2^-23: the size of one single-precision rounding relative to the dot.
EPS32 = 2.0 ** -23
#: the environment word that names the bar for the window runner (the door sets it on every window it starts)
RUNNER_ENV = "AD_DOT_BAR"
#: the environment word a user may set instead of --dot-bar
USER_ENV = "WOOF_HEX_DOT_BAR"
SETTINGS = ("roundings", "strict")


def roundings(y_mx: float | None, x_mty: float | None) -> float | None:
    """The signed reading in float32 roundings of the dot; None when either side is missing or not finite."""
    if y_mx is None or x_mty is None:
        return None
    a, b = float(y_mx), float(x_mty)
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    if a == b:
        return 0.0
    size = max(abs(a), abs(b))
    return (b - a) / (EPS32 * size)


def roundings_of_rel(rel: float | None) -> float | None:
    """|r| from a relative reading alone (rel_aligned): the same number without its sign."""
    if rel is None or not math.isfinite(float(rel)):
        return None
    return abs(float(rel)) / EPS32


def bar_of(table: dict, setting: str | None = None) -> dict:
    """The bar in force: the table's self_dot_bar block with the setting chosen (--dot-bar, else WOOF_HEX_DOT_BAR,
    else the table's default), and its relative size ``rel`` (what one direction's rel_aligned may reach)."""
    spec = table["thresholds"]["self_dot_bar"]
    chosen = setting or os.environ.get(USER_ENV) or spec["default"]
    if chosen not in SETTINGS:
        raise ValueError(f"the whole-state dot's bar {chosen!r} is not one of {', '.join(SETTINGS)}")
    if chosen == "strict":
        rel = float(spec["strict"]["relative"])
        return {"setting": "strict", "rel": rel, "each_within_roundings": rel / EPS32, "drift": None,
                "words": f"strict: the two sides agree within {rel:.0e} relative ({rel / EPS32:.2f} of one "
                         "single-precision rounding of the dot), on every direction"}
    k = float(spec["each_within_roundings"])
    drift = dict(spec["drift"])
    return {"setting": "roundings", "rel": k * EPS32, "each_within_roundings": k, "drift": drift,
            "words": bar_sentence(k, drift)}


def bar_sentence(k: float, drift: dict | None) -> str:
    """The bar in one plain sentence (printed by every door beside the reading)."""
    s = (f"the two sides of the dot must agree within {k:g} single-precision roundings of the dot "
         f"(one rounding is 2^-23 of its size, so {k:g} is {k * EPS32:.2e} relative), because the model and its "
         "derivative run in single precision")
    if drift:
        s += (f"; over {drift['min_directions']} or more directions the mean reading must also lie within "
              f"{drift['standard_errors']:g} standard errors of zero (no one-signed drift)")
    return s


def runner_env(bar: dict) -> dict:
    """The words the window runner reads for its own valid_derivative (the same bar, so both say the same)."""
    if bar["setting"] == "strict":
        return {RUNNER_ENV: f"relative:{bar['rel']!r}"}
    return {RUNNER_ENV: f"roundings:{bar['each_within_roundings']:g}"}


def judge_one(y_mx: float | None, x_mty: float | None, bar: dict, rel_aligned: float | None = None) -> dict:
    """One direction: part (a) only. Without both sides, the runner's rel_aligned gives the same size unsigned."""
    r = roundings(y_mx, x_mty)
    if r is None and rel_aligned is not None:
        r = roundings_of_rel(rel_aligned)
    rel = None if r is None else abs(r) * EPS32
    ok = r is not None and abs(r) <= bar["each_within_roundings"] * (1.0 + 1e-12)
    return {"roundings": r, "rel": rel, "pass": ok}


def judge_many(readings: list[float | None], bar: dict) -> dict:
    """Several directions: (a) each within the bar; (b) with at least drift.min_directions readings, the mean within
    drift.standard_errors standard errors of zero. With fewer readings (b) is not judged and says so."""
    k = bar["each_within_roundings"]
    rs = [None if r is None else float(r) for r in readings]
    each = [r is not None and abs(r) <= k * (1.0 + 1e-12) for r in rs]
    part_a = bool(rs) and all(each)
    good = [r for r in rs if r is not None]
    n = len(good)
    mean = sum(good) / n if n else None
    sd = math.sqrt(sum((r - mean) ** 2 for r in good) / (n - 1)) if n >= 2 else None
    se = sd / math.sqrt(n) if sd is not None else None
    drift = bar.get("drift")
    if drift is None:
        part_b, why_b = None, "the strict setting judges each direction only"
    elif n < int(drift["min_directions"]):
        part_b, why_b = None, (f"not judged: {n} directions read, the drift test needs "
                               f"{drift['min_directions']}")
    else:
        lim = float(drift["standard_errors"]) * (se or 0.0)
        part_b = abs(mean) <= lim if se else mean == 0.0
        why_b = (f"mean {mean:+.2f} roundings, standard error {se:.2f}: "
                 + ("within" if part_b else "NOT within") + f" {drift['standard_errors']:g} standard errors of zero")
    ok = part_a and part_b is not False and n == len(rs)
    return {"readings": rs, "each_pass": each, "within_each": part_a, "n": n, "mean": mean, "sd": sd, "se": se,
            "positive": sum(1 for r in good if r > 0), "no_drift": part_b, "no_drift_why": why_b, "pass": ok}
