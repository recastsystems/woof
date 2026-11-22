# woof/verify/draw_spread.py
"""Realisation spread of receipt statistics across independent draws.

The engine side of the difference-sigma protocol.  Two runs of one case at
two seeds are two draws from one distribution and nothing else; this
aggregates their committed receipts and reports, per metric: n, mean,
sample standard deviation (ddof = 1), min, max, and coefficient of
variation.

**It computes no band, no threshold and no verdict.**  It is arithmetic
over committed numbers.  Bands are cut by the registered method in
``docs/internal/receipts/les/P1-MOIST-COMPARISON-EXPECTATION.md`` §4
from the spreads this produces, in a step that cites that document's
registration sha; nothing here can gate anything.

``sd`` is ``None`` for n < 2 rather than 0.0, because one draw has no
spread and reporting one is the same error class as an absent denominator
reading 1.000 -- the failure the dry lane's ``resolved_fraction_ml``
carrier guard exists to prevent.

------------------------------------------------------------------------
WHY THIS EXISTS RATHER THAN A PATCH TO THE ORACLE'S AGGREGATOR
------------------------------------------------------------------------
``tools/wrf_em_les_oracle/moist_spread.py`` aggregates the WRF side and
indexes ``wrfout_sha256`` unconditionally, which no engine receipt has or
should have.  Editing the oracle's instrument so it can also score the
engine it exists to check is precisely what
``tools/wrf_em_les_oracle/INSTRUMENT-HISTORY.md`` forbids, and it would
put both sides' numbers behind one mutable artifact.  So the oracle's file
is untouched and this is a second, independent reduction.

Two independent reductions can drift apart, which would be worse than one
shared one -- so they are pinned together:
``tests/test_les_draw_spread.py`` runs THIS module over the oracle's own
committed multi-draw receipts, with the oracle's own metric and identity
lists, and requires it to reproduce the oracle's published aggregate JSON
exactly.  The equivalence is measured on real committed output, not
asserted from the source.

------------------------------------------------------------------------
DOTTED PATHS
------------------------------------------------------------------------
Metric, identity and flag names are dotted paths into the receipt
(``config.km_opt``, ``saturated_branch.engaged_somewhere``), so one
mechanism covers both sides' receipt shapes: the oracle's are mostly flat
with one nested engagement block, the engine's nest their configuration.
A missing path yields ``None``, which the statistics drop -- an absent
metric reports ``n = 0`` and is never silently counted as a zero.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from woof.verify.cases.convective_boundary_layer import ENVIRONMENTAL_FIELDS

__all__ = [
    "stats", "aggregate", "aggregate_paths", "dig",
    "MOIST_METRICS", "MOIST_IDENTITY", "MOIST_FLAGS", "NOTE", "main",
]

#: Printed and stored verbatim, the same sentence the oracle's aggregator
#: carries, so neither output can be read as carrying a verdict.
NOTE = "no band, threshold or verdict is computed here"

#: The moist headline set on an engine receipt.  Every name is a field
#: ``cloud_topped_boundary_layer`` writes, and every one of them comes out
#: of the SHARED reducer, so a spread taken here is a spread of the same
#: quantity the oracle's spread is taken of.
MOIST_METRICS = (
    "zi_thetav_load_m",
    "cloud_base_m", "cloud_top_m",
    "cloud_fraction_max", "sat_fraction_max", "n2_moist_fraction_max",
    "wthv_res_max", "wthv_total_min", "wthv_res_max_over_qs",
    "wqv_res_max", "wqv_total_max",
    "resolved_fraction_wthv", "resolved_fraction_wqv",
    "qv_surface", "qc_profile_max", "qr_profile_max",
    "qc_max_pointwise_run", "qr_max_pointwise_run",
    "first_cloud_seconds",
    "lwp_kg_m2", "rwp_kg_m2", "rainnc_mm_end",
    # AC-CAP.1 v2: this family is non-stationary, so the LWP trend is a
    # first-class quantity that travels with every windowed statistic
    # above rather than a footnote to them.
    "lwp_trend_pct_per_h",
    # validity observables -- they vary between draws and their spread is
    # part of the picture
    "w_max", "cfl_max", "mass_drift_rel",
)

#: A spread over draws is only a spread if the draws are the same
#: configuration.  Anything that differs is REPORTED, not assumed away.
MOIST_IDENTITY = (
    "config.km_opt", "config.mp_physics",
    "config.nx", "config.ny", "config.nz",
    "config.dx", "config.dt", "config.ztop", "config.run_seconds",
    "config.c_s", "config.c_k", "config.tke_heat_flux",
    "config.isfflx", "config.bl_pbl_physics",
    "window_minutes", "window_frames",
    "sounding.sha256",
    # Which BUILD produced the draw.  A mutation-control arm runs a
    # deliberately broken kernel; averaging one into a scored draw set
    # would corrupt the sigma every band is cut from, and quietly.  It is
    # an identity key so that mixing the arms is REPORTED at the top of the
    # output instead of disappearing into a mean.
    "mutation_control",
)

#: ``(output name, dotted path, "all" | "any")``.  The clamp-coverage
#: discipline as a draw-level statement: every draw must engage the
#: saturated branch somewhere, and no draw may engage it everywhere.  The
#: output names are the ORACLE's names, so the two aggregates read the
#: same way side by side.
MOIST_FLAGS = (
    ("saturated_branch_all_engaged_somewhere",
     "n2_engaged_somewhere", "all"),
    ("saturated_branch_any_engaged_everywhere",
     "n2_engaged_everywhere", "any"),
)


def dig(obj, path: str):
    """Follow a dotted path into nested dicts; ``None`` if it is not there."""
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def stats(values) -> dict:
    """n / mean / sample sd / min / max / cv over the present values.

    Transcribed from ``moist_spread.stats`` and held equal to it by
    ``tests/test_les_draw_spread.py``.  The arithmetic is deliberately
    plain Python rather than numpy: the oracle's is, and reproducing its
    published aggregate to the bit is the test that keeps the two
    reductions accurate.

    ``None`` and NaN are dropped rather than propagated -- a metric that is
    undefined in one draw (the dry anchor's vapour resolved fraction, whose
    denominator is exactly zero) should reduce the sample size, not poison
    the statistic.
    """
    vals = [v for v in values if v is not None
            and not (isinstance(v, float) and math.isnan(v))]
    n = len(vals)
    if n == 0:
        return dict(n=0, mean=None, sd=None, min=None, max=None, cv=None)
    mean = sum(vals) / n
    if n < 2:
        return dict(n=n, mean=mean, sd=None, min=min(vals), max=max(vals),
                    cv=None)
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    return dict(n=n, mean=mean, sd=sd, min=min(vals), max=max(vals),
                cv=(sd / abs(mean) if mean else None))


def aggregate(receipts, *, label: str, source_names=None,
              metrics=MOIST_METRICS, identity=MOIST_IDENTITY,
              flags=MOIST_FLAGS, registration: str | None = None,
              extra: dict | None = None) -> dict:
    """Aggregate a list of already-loaded receipt dicts.

    ``registration`` is the commit sha of the expectation document a scored
    comparison must cite (§0 of
    ``P1-MOIST-COMPARISON-EXPECTATION.md``).  It is recorded, never
    checked: this module publishes spread and gates nothing, and a tool
    that refused to compute arithmetic without a sha would be a gate this
    lane did not register.  An output with ``registration: null`` is a
    measurement rather than a scored comparison, and says so on its face.
    """
    receipts = list(receipts)
    rows = {name: stats([dig(r, name) for r in receipts]) for name in metrics}

    identity_out = {}
    for key in identity:
        seen = sorted({json.dumps(dig(r, key), sort_keys=True)
                       for r in receipts})
        identity_out[key] = (json.loads(seen[0]) if len(seen) == 1
                             else {"DIFFERS": [json.loads(v) for v in seen]})
    differing = [k for k, v in identity_out.items()
                 if isinstance(v, dict) and "DIFFERS" in v]

    out = dict(
        label=label,
        n_draws=len(receipts),
        source_receipts=list(source_names) if source_names is not None
        else [None] * len(receipts),
        registration=registration,
        configuration=identity_out,
        configuration_differs_on=differing,
        metrics=rows,
        note=NOTE,
    )
    for out_name, path, mode in flags:
        vals = [dig(r, path) for r in receipts]
        out[out_name] = (all(vals) if mode == "all" else any(vals))
    if extra:
        out.update(extra)
    return out


def aggregate_paths(paths, *, label: str, **kwargs) -> dict:
    """:func:`aggregate` over receipt JSON files, keyed by basename."""
    paths = [Path(p) for p in paths]
    receipts = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    return aggregate(receipts, label=label,
                     source_names=[p.name for p in paths], **kwargs)


def _print_table(out: dict) -> None:
    print("=== %s : %d draws ===" % (out["label"], out["n_draws"]))
    if out["registration"] is None:
        print("registration: none cited -- this is a measurement, not a "
              "scored comparison")
    else:
        print("registration: %s" % out["registration"])
    if out["configuration_differs_on"]:
        print("!! configuration DIFFERS across draws on: %s"
              % out["configuration_differs_on"])
    print("%-34s %3s %14s %12s %12s %10s"
          % ("metric", "n", "mean", "sd", "cv", "max-min"))
    for name, s in out["metrics"].items():
        if s["n"] == 0:
            print("%-34s %3d %14s" % (name, 0, "absent in all draws"))
            continue
        print("%-34s %3d %14.6g %12s %12s %10.4g"
              % (name, s["n"], s["mean"],
                 "-" if s["sd"] is None else "%.4g" % s["sd"],
                 "-" if s["cv"] is None else "%.4f" % s["cv"],
                 s["max"] - s["min"]))
    print("%-34s %s" % ("all draws engaged somewhere",
                        out["saturated_branch_all_engaged_somewhere"]))
    print("%-34s %s" % ("any draw engaged everywhere",
                        out["saturated_branch_any_engaged_everywhere"]))
    print(NOTE)


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Realisation spread of engine receipts across draws")
    parser.add_argument("label")
    parser.add_argument("out", type=Path)
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--registration", default=None,
                        help="commit sha of the registered expectation a "
                             "scored comparison must cite; recorded, never "
                             "checked (this tool gates nothing)")
    args = parser.parse_args(argv)

    out = aggregate_paths(args.receipts, label=args.label,
                          registration=args.registration)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="\n", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    _print_table(out)
    return 0


#: Receipt fields that must never appear in a spread: they measure the
#: MACHINE at the moment of measurement rather than the run, so their
#: draw-to-draw scatter is not a physical spread and pooling it into a
#: sigma would inflate every band cut from that sigma.  Single-sourced
#: from the case module's determinism-screen partition so the two
#: classifications cannot disagree.
EXCLUDED_FROM_SPREAD = ENVIRONMENTAL_FIELDS


if __name__ == "__main__":
    import sys
    sys.exit(main())
