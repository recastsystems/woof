"""Replay the init door's RRFS job builder against the fork's native init on real inputs.

For each named row of the engine's committed RRFS module plan (r10's
tools/rustwx/crates/rw-mpas/tests/rrfs_native/module-file-replay-plan-v1.json), this
builds the prepare and case7 jobs with woof.hex.mpasphys.rrfs.cold_start_rrfs from the
producer table alone (static, met, start time and first-guess level counts are the only
inputs), checks that every recipe value equals the plan's job (paths and provenance text
aside), runs them through the given rw_mpas_init_v831 and compares every stored value of
the init with the native reference (rw_mpas_static --compare). No tolerance, no exclusion.

usage: python rrfs_init_door_replay.py --plan PLAN --references DIR --worker BIN \
           --comparer BIN --work DIR --receipt JSON [--row ID ...]
DIR holds <row id>/init.nc (the fork's strict native init for that row).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from woof.hex.mpasphys.rrfs import cold_start_rrfs as cs

PATH_KEYS = {"static_path", "output_path", "met_path", "gocart_path", "capsule_path", "reference_path",
             "out_path", "specified_levels_path", "provenance", "lineage"}
DEFAULT_ROWS = ("x1.2562-2023-03-31-workflow-specified-canonical-off",
                "x1.2562-2022-12-22-workflow-specified-canonical-off",
                "x1.10242-2023-03-31-workflow-specified-canonical-off",
                "x1.10242-2022-12-22-workflow-specified-canonical-off",
                "canonical-hourly-cut-2023-03-31-case7",
                "canonical-hourly-cut-2022-12-22-case7")


def strip(value):
    if isinstance(value, dict):
        out = {k: strip(v) for k, v in value.items() if k not in PATH_KEYS}
        if "boundary" in out and isinstance(out["boundary"], dict):
            out["boundary"] = {k: v for k, v in out["boundary"].items() if k != "met_path"}
        return out
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in ("plan", "references", "worker", "comparer", "work", "receipt"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--row", action="append", default=None)
    p.add_argument("--physics-pin-set", default=None, help="select the same producer route as the init door")
    p.add_argument("--lbc-row", action="append", default=None, help="a case-9 plan row to replay through case9_jobs")
    p.add_argument("--lbc-init-row", default="canonical-hourly-cut-2023-03-31-case7",
                   help="the case-7 row whose native init.nc (under --references) is the boundary carrier's template")
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    rows = {r["id"]: r for r in plan["rows"]}
    key, producer = cs.producer_for_suite("hrrrv5", pin_set=a.physics_pin_set)
    assert producer["source_pin"] == plan["native_source_pin"]
    a.work.mkdir(parents=True, exist_ok=True)
    record = {"schema": "gpuwm-hex.rrfs-init-door-replay/v1", "producer": key,
              "native_source_pin": producer["source_pin"], "rows": [], "deleted_files": []}
    for rid in a.row if a.row is not None else (() if a.lbc_row else DEFAULT_ROWS):
        row = rows[rid]
        planned = {j["stage"]: j for j in row["jobs"]}
        p7 = planned["case7"]["config"]
        work = a.work / rid
        work.mkdir(parents=True, exist_ok=True)
        levels = work / "ZETA_LEVELS.txt"
        levels.write_text(cs.specified_levels_text(producer))
        regional = bool(planned["prepare"]["config"]["vertical"]["regional_halo"])
        jobs = cs.case7_jobs(producer, static=Path(p7["static_path"]), met=Path(p7["met_path"]),
                             start_time=p7["start_time"], nfglevels=p7["n_fg_levels"],
                             nfgsoillevels=p7["n_fg_soil_levels"], artifact=work / "artifact.nc",
                             out=work / "init.nc", levels_file=levels, aerosols=Path(p7["gocart_path"]),
                             regional=regional, version="door-replay")
        built = {j["stage"]: j for j in jobs}
        recipe_equal = {stage: strip(built[stage]) == strip(planned[stage]) for stage in ("prepare", "case7")}
        started = time.monotonic()
        receipts = cs.run_jobs(a.worker, jobs, work / "logs")
        seconds = time.monotonic() - started
        report = work / "compare.json"
        reference = a.references / rid / "init.nc"
        rc = subprocess.run([str(a.comparer), "--compare", str(reference), str(work / "init.nc"),
                             "--report", str(report)], capture_output=True).returncode
        comparison = json.loads(report.read_text())
        fields = comparison["fields"]
        result = {"id": rid, "recipe_equal_to_plan": recipe_equal, "regional": regional,
                  "producer_seconds": round(seconds, 3), "stage_seconds": {r["stage"]: r["seconds"] for r in receipts},
                  "dimensions_agree": comparison.get("dimensions_agree"),
                  "word_identical": comparison.get("word_identical"),
                  "compare_returncode": rc, "fields": len(fields), "values": sum(f["n"] for f in fields),
                  "differing": sum(f["n"] - f["exact"] for f in fields),
                  "only_in_reference": comparison.get("only_in_reference"),
                  "only_in_candidate": comparison.get("only_in_candidate"),
                  "shape_mismatch": comparison.get("shape_mismatch"), "type_mismatch": comparison.get("type_mismatch")}
        record["rows"].append(result)
        print(json.dumps(result), flush=True)
        for path in (work / "init.nc", work / "artifact.nc"):
            if path.exists():
                record["deleted_files"].append({"path": str(path), "bytes": path.stat().st_size})
                a.receipt.write_text(json.dumps(record, indent=2) + "\n")
                path.unlink()
    for rid in a.lbc_row or ():
        # Boundaries: the door's carrier is made from the case-7 init (the native init of the
        # same date and mesh); the plan's from the fork's invariant file. Same fields, same values.
        row = rows[rid]
        planned = {j["stage"]: j for j in row["jobs"]}
        p9 = planned["case9"]["config"]
        work = a.work / rid
        work.mkdir(parents=True, exist_ok=True)
        init = a.references / a.lbc_init_row / "init.nc"
        intervals = [(i["valid_time"], Path(i["met_path"])) for i in p9["intervals"]]
        jobs = cs.case9_jobs(producer, init=init, start_time=p9["start_time"], intervals=intervals,
                             nfglevels=p9["n_fg_levels"], out_dir=work / "lbc", carrier=work / "carrier.nc",
                             aerosols=Path(p9["gocart_path"]))
        jobs[1]["config"].update(hydrometeors_rrfs=p9["hydrometeors_rrfs"], hydrometeors_gfs=p9["hydrometeors_gfs"])
        built = {j["stage"]: j for j in jobs}
        recipe_equal = {"case9": strip({k: v for k, v in built["case9"]["config"].items()
                                        if k not in ("grid_path", "out_dir", "intervals")})
                        == strip({k: v for k, v in p9.items() if k not in ("grid_path", "out_dir", "intervals")})}
        (work / "lbc").mkdir(exist_ok=True)
        started = time.monotonic()
        receipts = cs.run_jobs(a.worker, jobs, work / "logs")
        seconds = time.monotonic() - started
        for index, pair in enumerate(row["comparisons"]):
            reference = Path(pair["reference"]["path"])
            candidate = work / "lbc" / Path(pair["candidate"]).name
            report = work / f"compare-{index}.json"
            rc = subprocess.run([str(a.comparer), "--compare", str(reference), str(candidate), "--report", str(report)],
                                capture_output=True).returncode
            comparison = json.loads(report.read_text())
            fields = comparison["fields"]
            result = {"id": rid, "kind": "lbc", "file": candidate.name, "recipe_equal_to_plan": recipe_equal, "producer_seconds": round(seconds, 3),
                      "stage_seconds": {r["stage"]: r["seconds"] for r in receipts},
                      "dimensions_agree": comparison.get("dimensions_agree"),
                  "word_identical": comparison.get("word_identical"),
                  "compare_returncode": rc, "fields": len(fields), "values": sum(f["n"] for f in fields),
                      "differing": sum(f["n"] - f["exact"] for f in fields),
                      "differing_fields": [f["name"] for f in fields if f["n"] != f["exact"]],
                      "only_in_reference": comparison.get("only_in_reference"),
                      "only_in_candidate": comparison.get("only_in_candidate"),
                      "shape_mismatch": comparison.get("shape_mismatch"), "type_mismatch": comparison.get("type_mismatch")}
            record["rows"].append(result)
            print(json.dumps(result), flush=True)
        for path in [work / "carrier.nc", *sorted((work / "lbc").glob("*.nc"))]:
            if path.exists():
                record["deleted_files"].append({"path": str(path), "bytes": path.stat().st_size})
                a.receipt.write_text(json.dumps(record, indent=2) + "\n")
                path.unlink()
    record["values"] = sum(r["values"] for r in record["rows"])
    record["differing"] = sum(r["differing"] for r in record["rows"])
    record["passed"] = bool(record["rows"]) and all(r["dimensions_agree"] and r["word_identical"] and r["differing"] == 0 and r["compare_returncode"] == 0 and all(r["recipe_equal_to_plan"].values())
                           and not r["only_in_reference"] and not r["only_in_candidate"]
                           and not r["shape_mismatch"] and not r["type_mismatch"] for r in record["rows"])
    a.receipt.write_text(json.dumps(record, indent=2) + "\n")
    print("RRFS_INIT_DOOR_REPLAY values", record["values"], "differing", record["differing"], "passed", record["passed"])
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
