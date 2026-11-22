"""The radiation cloud-sample seed rule (``config_radt_mcica_seed``) on the adjoint doors.

How the derivative treats the McICA sub-column draw, today and with this option:

* The linear model never differentiates the draw. ``mpas_lw_mcica_seeds`` and ``mpas_sw_mcica_seeds`` write integer
  KISS words and have identity rules; the sub-column kernels take the cloudy/clear decision as a frozen mask (their
  derivative rows say so: "These are the saved McICA samples"). So the tangent and the adjoint are the derivative of
  radiation *with the base forecast's cloud sample held*, under either rule.
* The real forecasts a window runs beside them (the perturbed forwards of ``adjoint-test``) are what differ. Under
  MPAS's ``pressure`` rule a perturbed forward re-draws every column whose four lowest pressures moved by one bit,
  which no tangent predicts; under the ``cell`` rule the recorded plan carries ``mpas_mcica_seeds_column``, which
  writes the same words in every forward, so the real forecasts and the linear model hold the same sample.

The choice is the plan's: it is made when the forecast is recorded (the forecast word
``--physics config_radt_mcica_seed=cell`` among the recording's forecast words, which the record receipt keeps, and
the column-seed launch among the plan's kernels). A door asked for one rule on a plan recorded under the other refuses
by name: the plan's kernels decide which sample its forwards draw, and the door cannot change them.
"""
from __future__ import annotations

import json
from pathlib import Path

from .mpasphys import mcica_seed_rule as rule

#: The door option's values; None (the default) takes the plan's own rule.
CHOICES = rule.MODES


def forecast_seed(words) -> str | None:
    """The seed rule a list of forecast words names (``--physics config_radt_mcica_seed=X``, or ``--physics=...``;
    the last one wins, as the forecast door's parser reads them), or None when no word names one."""
    found = None
    words = [str(w) for w in (words or [])]
    for i, word in enumerate(words):
        value = None
        if word == "--physics" and i + 1 < len(words):
            value = words[i + 1]
        elif word.startswith("--physics="):
            value = word.split("=", 1)[1]
        if value is None:
            continue
        key, sep, setting = value.partition("=")
        if sep and key.strip().casefold() == rule.OPTION:
            found = setting.strip().strip("'\"").casefold()
    return found


def plan_seed(plan: Path) -> dict:
    """What a plan package says about its cloud-sample seed rule: ``{"rule": "pressure" | "cell" | None, "from": ...}``.

    The plan's kernels decide (a plan with ``mpas_mcica_seeds_column`` was recorded under ``cell``; one with the
    pressure seed launches and not it, under ``pressure``; one with no McICA launch calls no radiation in its window,
    so ``None``: either rule replays it). The record receipt's forecast words must agree when they name a rule."""
    said = None
    try:
        receipt = json.loads((Path(plan) / "record-receipt.json").read_text())
        said = forecast_seed(receipt.get("forecast_args")) if isinstance(receipt, dict) else None
    except (OSError, ValueError):
        receipt = None
    kernels = None
    try:
        names = (rule.KERNEL.encode(), b"mpas_lw_mcica_seeds", b"mpas_sw_mcica_seeds")
        seen, tail = set(), b""
        with open(Path(plan) / "plan.json", "rb") as stream:   # in 16 MiB pieces: a 3 km plan is gigabytes
            while piece := stream.read(1 << 24):
                text = tail + piece
                seen.update(name for name in names if name in text)
                tail = text[-64:]
        kernels = rule.CELL if names[0] in seen else rule.PRESSURE if seen else None
    except OSError:
        pass
    if said is not None and kernels is not None and said != kernels:
        from .ad_doors import AdDoorError
        raise AdDoorError(f"plan {plan}: its record receipt's forecast words name {rule.OPTION} = {said!r} and its "
                          f"kernels are those of the {kernels!r} rule ({'with' if kernels == rule.CELL else 'without'} "
                          f"{rule.KERNEL}): the package is not one recording; record it again")
    if kernels is not None:
        return {"rule": kernels, "from": "the plan's kernels" + (" and its record receipt" if said else "")}
    if said is not None:
        return {"rule": said, "from": "its record receipt's forecast words"}
    return {"rule": None, "from": "no radiation call in the recorded window"}


def refuse_plan_seed(plan: Path, asked: str | None, door: str) -> str | None:
    """The plan's seed rule; a door asked for the other rule refuses by name, before any card."""
    found = plan_seed(plan)
    recorded = found["rule"]
    if asked is not None and recorded is not None and asked != recorded:
        from .ad_doors import AdDoorError
        raise AdDoorError(
            f"{door}: --radiation-seed {asked}: plan {plan} was recorded under {rule.OPTION} = {recorded!r} "
            f"({found['from']}), and a plan's forwards draw the cloud sample its recording held: replaying it under "
            f"{asked!r} would run another forecast than the one the linear model is of. Record a plan under "
            f"{asked!r} (--init with --radiation-seed {asked}, or the recorder with "
            f"-- --physics {rule.OPTION}={asked}), or leave --radiation-seed out")
    if recorded == rule.CELL:
        print(f"woof hex {door}: the plan draws radiation's cloud sample by the cell rule ({rule.OPTION} = cell: "
              "the cell's global index and the radiation call's time step), so every real forecast of this window "
              "draws the sub-columns the linear model holds; this is not MPAS's rule", flush=True)
    return recorded


def record_words(args) -> list[str]:
    """The forecast words a recording from --init takes for ``--radiation-seed`` (none for MPAS's rule)."""
    asked = getattr(args, "radiation_seed", None)
    if asked in (None, rule.DEFAULT):
        return []
    return ["--", "--physics", f"{rule.OPTION}={asked}"]


HELP = ("the radiation cloud-sample seed rule (the forecast's config_radt_mcica_seed): pressure, MPAS's rule, "
        "re-draws a column's sub-columns whenever its lowest pressures move by one bit; cell draws them from the "
        "cell's global index and the radiation call's time step, so every real forecast of a window draws the "
        "sample the linear model holds (derivative checks and twin experiments; not MPAS's rule). "
        "With --init the recording is made under it; with --plan it must be the plan's own rule, or the door "
        "refuses. Default: the plan's rule (pressure for a recording from --init)")
