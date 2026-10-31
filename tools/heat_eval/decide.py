"""The decision on the nowcast heating arm, from the score tables and the eye.

The rule is the one in the evaluation's
cases-and-scoring record ("Decision rule"), with the arm letters of DESIGN.md:
A is the control, C the nowcast-heated arm, E the oracle.  C is promoted
only if every rule holds:

R1  By eye, on the MRMS | A | B | C | E | HRRR sheets, C places the new
    storms of f01 to f03 better than A in at least 3 of the 4 decision
    cases.  (A person judges; ``eye.json`` records it.  Without it the
    verdict is pending, never a pass.)
R2  Mean FSS at 35 dBZ in the 27 km box over f01 to f03 beats A in at
    least 3 of the 4 decision cases, and no decision case loses more than
    0.03.
R3  The 35 dBZ area ratio (model echo area over MRMS echo area) at each of
    f01 to f03 moves toward 1 or stays inside [0.5, 2], in every decision
    case.  Prevents: heating that wins FSS by painting too much echo.
    "Toward 1" is measured as |log(ratio)|, so 0.5 and 2 are equally far.
R4  Station T2 and Td2 RMSE at f01 to f06 no more than 0.05 K worse than A
    at any lead in any decision case.  Prevents: the right echo with
    spurious cold pools and outflow.
R5  At f12 and f18: FSS35 at 27 km no more than 0.03 below A, and T2 and
    Td2 RMSE no more than 0.05 K above A, in every decision case.
    Prevents: a short-range gain paid for with a long-range drift.
R6  The anchor case (case 5) is not worse than A on R2 to R4, read case by
    case: its f01-f03 mean FSS35 loses no more than 0.03, its area ratios
    pass R3, and its station RMSE passes R4.

E minus C (mean FSS35 at 27 km over f01 to f03) is reported per case as
the nowcast's cost.  The same rules are run on E against A; if E also
fails, the heating method is the problem, not the nowcast.

An arm whose heating read any oracle window can fill only an oracle column
(E, E3D); a table that puts one anywhere else is refused, because future
observations would be scored as forecast skill.

    python -m tools.heat_eval.decide --root /work/nh [--eye eye.json] [--out decision.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

from woof.verify.obs.registration import NOWCAST_HEAT_DECISION as RULE
from tools.heat_eval import plan as planmod

DECISION_SCHEMA = "gpuwm.heat-eval-decision.v1"
TABLE_SCHEMA = "gpuwm.heat-eval-scores.v1"

FSS_KEY = "fss35_27km"
AREA_KEY = "area_ratio35"
STATION_KEYS = ("t2_rmse", "td2_rmse")
# The numbers are the registration's (nowcast-heat-v1), never retyped here.
EARLY_LEADS = tuple(int(v) for v in RULE["early_leads"])
SURFACE_LEADS = tuple(int(v) for v in RULE["surface_leads"])
LATE_LEADS = tuple(int(v) for v in RULE["late_leads"])
FSS_MARGIN = float(RULE["fss_margin"])
RMSE_MARGIN_K = float(RULE["rmse_margin_k"])
AREA_BAND = tuple(float(v) for v in RULE["area_ratio_band"])
NEEDED_WINS = int(RULE["needed_wins"])
DECISION_CASES = int(RULE["decision_cases"])

CONTROL, TEST, ORACLE = "A", "C", "E"


class DecisionRefused(ValueError):
    """A score table the decision cannot be taken on."""


def key_by_valid(t0: str, rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    """Score rows keyed by their valid time's hours after the case's t0.

    Every arm is filed by valid time, never by its own lead: arm B starts an
    hour before t0, so its own f02 is everyone else's f01.  Rows that do not
    fall on a whole hour after t0 are refused rather than rounded.
    """
    start = planmod.utc(t0)
    out: dict[str, dict[str, object]] = {}
    for row in rows:
        hours = (planmod.utc(str(row["valid"])) - start).total_seconds() / 3600.0
        if hours != int(hours) or hours < 0:
            raise DecisionRefused(f"valid {row['valid']} is not a whole hour after t0 {t0}")
        key = str(int(hours))
        if key in out:
            raise DecisionRefused(f"two rows for valid {row['valid']}")
        out[key] = dict(row)
    return out


def check_columns(table: Mapping[str, object], oracle_columns: Sequence[str]) -> None:
    """Refuse an oracle-heated run anywhere but an oracle column."""
    for arm_id, arm in dict(table.get("arms", {})).items():
        classes = set(arm.get("lead_classes", []) or [])
        column = str(arm.get("column", arm_id))
        if "oracle" in classes and column not in oracle_columns:
            raise DecisionRefused(
                f"{table.get('case_id')}: arm {arm_id} read oracle windows and sits in the "
                f"forecast column {column!r}; future observations would be scored as skill")


def _value(table: Mapping[str, object], arm: str, lead: int, key: str) -> float | None:
    row = (((table.get("arms") or {}).get(arm) or {}).get("by_lead") or {}).get(str(lead))
    if not row:
        return None
    value = row.get(key)
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def _mean(values: Sequence[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return sum(values) / len(values)  # type: ignore[arg-type]


def _early_fss(table: Mapping[str, object], arm: str) -> float | None:
    return _mean([_value(table, arm, lead, FSS_KEY) for lead in EARLY_LEADS])


def _area_ok(test: float | None, control: float | None) -> bool | None:
    if test is None or control is None or test <= 0.0 or control <= 0.0:
        return None
    toward = abs(math.log(test)) <= abs(math.log(control))
    return toward or AREA_BAND[0] <= test <= AREA_BAND[1]


def case_checks(table: Mapping[str, object], test: str = TEST,
                control: str = CONTROL) -> dict[str, object]:
    """R2-R5 inputs for one case: per-rule pass/fail with the numbers."""
    fss_test, fss_control = _early_fss(table, test), _early_fss(table, control)
    fss = {"test": fss_test, "control": fss_control,
           "difference": None if None in (fss_test, fss_control) else fss_test - fss_control}
    area = []
    for lead in EARLY_LEADS:
        t, c = _value(table, test, lead, AREA_KEY), _value(table, control, lead, AREA_KEY)
        area.append({"lead": lead, "test": t, "control": c, "pass": _area_ok(t, c)})
    surface = []
    for lead in SURFACE_LEADS:
        for key in STATION_KEYS:
            t, c = _value(table, test, lead, key), _value(table, control, lead, key)
            surface.append({"lead": lead, "variable": key, "test": t, "control": c,
                            "pass": None if None in (t, c) else t - c <= RMSE_MARGIN_K})
    late = []
    for lead in LATE_LEADS:
        t, c = _value(table, test, lead, FSS_KEY), _value(table, control, lead, FSS_KEY)
        late.append({"lead": lead, "variable": FSS_KEY, "test": t, "control": c,
                     "pass": None if None in (t, c) else c - t <= FSS_MARGIN})
        for key in STATION_KEYS:
            t, c = _value(table, test, lead, key), _value(table, control, lead, key)
            late.append({"lead": lead, "variable": key, "test": t, "control": c,
                         "pass": None if None in (t, c) else t - c <= RMSE_MARGIN_K})
    return {"fss_early": fss, "area": area, "surface": surface, "late": late}


def _all(rows: Sequence[Mapping[str, object]]) -> bool | None:
    states = [row["pass"] for row in rows]
    if any(state is False for state in states):
        return False
    if any(state is None for state in states):
        return None
    return True


def evaluate(tables: Sequence[Mapping[str, object]], eye: Mapping[str, object] | None,
             *, test: str = TEST, control: str = CONTROL) -> dict[str, object]:
    """Every rule for ``test`` against ``control``; True, False or None (missing)."""
    decision = [t for t in tables if t.get("role") == "decision"]
    anchors = [t for t in tables if t.get("role") == "anchor"]
    checks = {str(t["case_id"]): case_checks(t, test, control) for t in tables}
    rules: list[dict[str, object]] = []

    # R1: the eye.
    verdicts = dict((eye or {}).get("cases", {}))
    seen = [verdicts.get(str(t["case_id"]), {}).get(f"{test}_better_than_{control}")
            for t in decision]
    wins = sum(1 for v in seen if v is True)
    r1 = (None if len(decision) < DECISION_CASES or any(v is None for v in seen)
          else wins >= NEEDED_WINS)
    rules.append({"rule": "R1", "name": "by eye, new storms f01-f03 placed better in 3 of 4",
                  "pass": r1, "wins": wins, "judged": sum(v is not None for v in seen)})

    # R2: FSS35 at 27 km, f01-f03 mean.
    diffs = [checks[str(t["case_id"])]["fss_early"]["difference"] for t in decision]
    if len(decision) < DECISION_CASES or any(d is None for d in diffs):
        r2 = None
    else:
        r2 = sum(1 for d in diffs if d > 0.0) >= NEEDED_WINS and min(diffs) >= -FSS_MARGIN
    rules.append({"rule": "R2", "name": "mean FSS35 27 km f01-f03 beats A in 3 of 4, no loss > 0.03",
                  "pass": r2, "differences": dict(zip([str(t["case_id"]) for t in decision], diffs))})

    # R3-R5: every decision case.
    for rule, name, part in (
            ("R3", "35 dBZ area ratio f01-f03 toward 1 or inside [0.5, 2]", "area"),
            ("R4", "T2 and Td2 RMSE f01-f06 within 0.05 K of A", "surface"),
            ("R5", "f12 and f18 FSS35 within 0.03 and station RMSE within 0.05 K of A", "late")):
        per_case = {str(t["case_id"]): _all(checks[str(t["case_id"])][part]) for t in decision}
        states = list(per_case.values())
        state = (None if len(decision) < DECISION_CASES or None in states
                 else all(states))
        if False in states:
            state = False
        failing = [{"case": case, **row} for case, value in checks.items()
                   if case in per_case for row in value[part] if row["pass"] is False]
        rules.append({"rule": rule, "name": name, "pass": state, "by_case": per_case,
                      "failing": failing})

    # R6: the anchor.
    if not anchors:
        r6, detail = None, {}
    else:
        anchor = checks[str(anchors[0]["case_id"])]
        diff = anchor["fss_early"]["difference"]
        parts = {"fss": None if diff is None else diff >= -FSS_MARGIN,
                 "area": _all(anchor["area"]), "surface": _all(anchor["surface"])}
        values = list(parts.values())
        r6 = False if False in values else (None if None in values else True)
        detail = {"case": str(anchors[0]["case_id"]), **parts}
    rules.append({"rule": "R6", "name": "the anchor case is not worse than A on R2-R4",
                  "pass": r6, "detail": detail})

    states = [rule["pass"] for rule in rules]
    verdict = ("do-not-promote" if False in states
               else "pending" if None in states else "promote")
    return {"test": test, "control": control, "verdict": verdict, "rules": rules,
            "checks": checks}


def admit_tables(tables: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """The tables the rule reads, with a reserve standing in for a missing decision case.

    The box chain runs the reserve only when a decision case fails gate G3,
    and that case is then never scored, so a reserve table fills the empty
    decision slot (marked ``stands_in``).  Without this the rule would count
    three decision cases and stay pending for good.  Reserves beyond the
    empty slots are left out.
    """
    out = [dict(t) for t in tables if t.get("role") in ("decision", "anchor")]
    open_slots = DECISION_CASES - sum(1 for t in out if t.get("role") == "decision")
    for table in tables:
        if table.get("role") == "reserve" and open_slots > 0:
            out.append(dict(table, role="decision", stands_in=True))
            open_slots -= 1
    return out


def decide(tables: Sequence[Mapping[str, object]], eye: Mapping[str, object] | None,
           *, oracle_columns: Sequence[str] = ("E", "E3D"),
           rule_status: str = "proposed-unratified") -> dict[str, object]:
    for table in tables:
        if table.get("schema") != TABLE_SCHEMA:
            raise DecisionRefused(f"{table.get('case_id')}: schema {table.get('schema')!r}")
        check_columns(table, oracle_columns)
    hashes = {str(t.get("registration_sha256")) for t in tables}
    if len(hashes) > 1:
        raise DecisionRefused(
            f"the score tables were made under {len(hashes)} different registrations; one "
            f"rule scores every case or the cross-case counts mean nothing")
    nowcast = evaluate(tables, eye, test=TEST, control=CONTROL)
    oracle = evaluate(tables, eye, test=ORACLE, control=CONTROL)
    cost = {}
    for table in tables:
        c, e = _early_fss(table, TEST), _early_fss(table, ORACLE)
        cost[str(table["case_id"])] = None if None in (c, e) else e - c
    if nowcast["verdict"] == "do-not-promote" and oracle["verdict"] == "do-not-promote":
        reading = ("the oracle fails too: the heating method is the problem, not the nowcast")
    elif nowcast["verdict"] == "do-not-promote" and oracle["verdict"] == "promote":
        reading = "the oracle passes and the nowcast does not: the nowcast's error is the problem"
    else:
        reading = ""
    return {
        "schema": DECISION_SCHEMA,
        "registration_sha256": hashes.pop() if hashes else None,
        "rule_status": rule_status,
        "verdict": nowcast["verdict"],
        "nowcast": nowcast,
        "oracle": oracle,
        "nowcast_cost_fss35_27km_f01_f03": cost,
        "reading": reading,
        "cases": [{"case": t["case_id"], "role": t.get("role"),
                   "stands_in": bool(t.get("stands_in"))} for t in tables],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.heat_eval.decide")
    parser.add_argument("--root", type=Path, default=planmod.DEFAULT_ROOT)
    parser.add_argument("--eye", type=Path, help="eye.json (R1); pending without it")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    tables = []
    for case in planmod.load_cases():
        path = planmod.case_dir(args.root, case) / "scores" / "table.json"
        if path.is_file():
            tables.append(json.loads(path.read_text(encoding="utf-8")))
    tables = admit_tables(tables)
    eye = json.loads(args.eye.read_text(encoding="utf-8")) if args.eye and args.eye.is_file() else None
    try:
        status = tables[0].get("rule_status", "proposed-unratified") if tables else "proposed-unratified"
        result = decide(tables, eye, oracle_columns=planmod.load_arms().oracle_columns,
                        rule_status=str(status))
    except DecisionRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    text = json.dumps(result, indent=2) + "\n"
    out = args.out or (args.root / "decision.json")
    out.write_text(text, encoding="utf-8")
    print(f"verdict {result['verdict']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
