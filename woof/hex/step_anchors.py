"""Dycore step certificates: evidence a run records, not a gate it passes.

The key includes RK order, acoustic count and dynamics splits. Reusing an
RK3 certificate at order 2 would conceal different first-stage timesteps,
coefficient reuse and scalar density weights, so a row names exactly the
schedule its native schedule bits and device contract receipts measured.

RETIRED AS A GATE (2026-10-10).  ``require_step_anchor`` refused any schedule without a row; this
module now only answers whether the run's schedule matches one, and the run
writes that answer into its receipt.  Native MPAS dry-step agreement is measured with explicit tolerances by
verify-dycore. A schedule certificate alone does not establish agreement.
"""
from __future__ import annotations

from importlib import resources
import json
from typing import Any


def _table() -> dict[str, Any]:
    return json.loads(resources.files("woof.hex").joinpath(
        "data", "dycore-step-anchors.v1.json"
    ).read_text(encoding="utf-8"))


def step_anchor_record(config: Any) -> dict[str, Any]:
    """The certificate row matching this schedule, or a record that none does."""

    key = (
        float(config.config_dt), int(config.config_time_integration_order),
        int(config.config_number_of_sub_steps), int(config.config_dynamics_split_steps),
    )
    for row in _table()["anchors"]:
        if key == (float(row["dt_seconds"]), int(row["order"]),
                   int(row["acoustic_substeps"]), int(row["dynamics_splits"])):
            return {"matched": True, "scope": "dycore", **row}
    return {
        "matched": False,
        "scope": "dycore",
        "schedule": {
            "dt_seconds": key[0], "order": key[1],
            "acoustic_substeps": key[2], "dynamics_splits": key[3],
        },
        "note": (
            "no dycore step certificate holds this schedule; the run is not "
            "refused for it (retired 2026-10-10) and this field is the "
            "evidence record"
        ),
    }
