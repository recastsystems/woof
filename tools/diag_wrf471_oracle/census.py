#!/usr/bin/env python3
"""WRF 4.7.1 diagnostic controls, importer behavior, and implementation evidence.

This is a static census plus CPU namelist probes, not a physics qualification.
The importer accepting a control does not establish that its outputs exist.
Run from the repository root with --output pointing at a receipt directory.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def selected(row):
    key = row["key"]
    return (row["section"] in ("diags", "afwa")
            or re.search(r"diag|refl|radar_ref|bucket|prec_acc|acc_phy|hailcast|lightning|iccg", key)
            or key in ("noahmp_output", "noahmp_acc_dt", "flashrate_factor",
                       "cellcount_method", "cldtop_adjustment", "reset_interval1", "haildt"))


def fixture():
    path = ROOT / "tests/test_namelist_import.py"
    constants = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("WPS_TEXT", "INPUT_TEXT"):
                    constants[target.id] = ast.literal_eval(node.value)
    return constants, hashlib.sha256(path.read_bytes()).hexdigest()


def source_hits(key):
    hits = {}
    for rel in ("woof/config.py", "woof/namelist_import.py", "woof/core/dycore.py",
                "woof/core/state.py", "woof/core/gsd_diagnostics.py",
                "woof/core/hail.py", "woof/core/output_diagnostics.py",
                "woof/io/history_layout.py", "woof/afwa_config.py",
                "woof/core/afwa_diagnostics.py", "woof/core/misc_buckets.py"):
        lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
        found = [i for i, line in enumerate(lines, 1)
                 if re.search(r"\b" + re.escape(key) + r"\b", line)]
        if found:
            hits[rel] = found
    return hits


def probe_value(row):
    key = row["key"]
    if row["type"] == "character":
        return ["diag_output_d<domain>_<date>"]
    if key.startswith("io_form_auxhist"):
        return [2]
    if key == "press_levels":
        return [85000., 50000.]
    if key == "z_levels":
        return [1000., 3000.]
    if key in ("num_press_levels", "num_z_levels"):
        return [2]
    if key in ("p_lev_missing", "z_lev_missing"):
        return [-888.]
    if key in ("bucket_mm", "bucket_j"):
        return [1.]
    if key in ("use_tot_or_hyd_p", "extrap_below_grnd"):
        return [3 - int(row["default"])]
    if row["type"] == "real":
        return [float(row["default"]) + 1.]
    if row["type"] == "logical":
        return [True]
    return [0 if int(row["default"]) != 0 else 1]


def generated_io_rows():
    """Expand the pinned Registry generator, absent from bundled JSON.

    Source line numbers and literal generator statements accompany every
    expansion so the census does not silently omit the two output streams.
    """
    source = Path(__file__).parent / "upstream/tools/gen_streams.c"
    rows = []
    pattern = re.compile(r'fprintf\(fp,"rconfig (character|integer) (\S+) (.*?)\\n"')
    for line_no, line in enumerate(source.read_text().splitlines(), 1):
        match = pattern.search(line)
        if not match:
            continue
        kind, key, rest = match.groups()
        if "%s%d" not in key:
            continue
        for stream in (22, 23):
            expanded = key.replace("%s", "hist").replace("%d", str(stream))
            default = f'"auxhist{stream}_d<domain>_<date>"' if kind == "character" else rest.split()[-1]
            rows.append(dict(key=expanded, section="time_control", type=kind, default=default,
                             generated_source=f"{source.relative_to(ROOT)}:{line_no}",
                             generator_statement=line.strip()))
    if len(rows) != 44:
        raise ValueError(f"pinned auxhist22/23 generator shape changed: {len(rows)} rows")
    return rows


def native_references(key):
    """The exact public source declarations, never a competitor inventory."""
    hits = {}
    for path in sorted((Path(__file__).parent / "upstream").rglob("*")):
        if path.is_file():
            lines = [n for n, line in enumerate(path.read_text().splitlines(), 1)
                     if re.search(r"\b" + re.escape(key) + r"\b", line, re.I)]
            if lines:
                hits[str(path.relative_to(ROOT))] = lines
    return hits


def field_inventory():
    """Record native package outputs and candidate name references separately.

    A textual reference or a history name is not evidence of matching native
    arithmetic, cadence, restart or nesting. The per-control assessment owns
    that distinction. Refusals stay in place for the unimplemented packages.
    """
    directory = Path(__file__).parent / "upstream/Registry"
    packages = {}
    wanted = {"nwp_output", "acc_phys_tend", "bucketropt", "bucketfopt",
              "press_diags", "z_diags", "do_trad_fields"}
    for path in sorted(directory.iterdir()):
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            words = line.split()
            if len(words) < 5 or words[0] != "package":
                continue
            if words[1] not in wanted and not words[1].startswith("afwa_"):
                continue
            state = next((v[6:] for v in words[4:] if v.startswith("state:")), "")
            packages[words[1]] = {"selector": words[2], "fields": [s.upper() for s in state.split(",") if s],
                                  "source": f"{path.relative_to(ROOT)}:{line_no}"}
    files = [p for p in (ROOT / "woof").rglob("*") if p.suffix in (".py", ".cu")]
    names = {name for row in packages.values() for name in row["fields"]}
    hits = {name: {} for name in names}
    word = re.compile(r"\b[A-Za-z_][A-Za-z_0-9]*\b")
    for path in files:
        rel = str(path.relative_to(ROOT))
        for n, line in enumerate(path.read_text().splitlines(), 1):
            for name in names.intersection(v.upper() for v in word.findall(line)):
                hits[name].setdefault(rel, []).append(n)
    return {"packages": packages, "candidate_name_references": hits,
            "qualification": "name references only; they do not establish a native producer or a written field"}


def candidate_inventory(baseline, output):
    from dataclasses import replace
    from woof.experiment import load_experiment
    from woof.io.history_layout import produced_history_shapes
    path = output / "baseline.toml"
    path.write_text(baseline)
    cfg = load_experiment(path).root.run
    probes = {"baseline": cfg,
              "nwp_diagnostics=1": replace(cfg, nwp_diagnostics=1),
              "gsd_diagnostics=1": replace(cfg, gsd_diagnostics=1),
              "cu_diag=1,cu_physics=3": replace(cfg, cu_diag=1, cu_physics=3),
              "hailcast_opt=1": replace(cfg, hailcast_opt=1),
              "do_radar_ref=1": replace(cfg, do_radar_ref=1),
              "afwa_diag_opt=1": replace(cfg,afwa_diag_opt=1),
              "afwa_therm_opt=1": replace(cfg,afwa_diag_opt=1,afwa_therm_opt=1),
              "afwa_six_packages": replace(cfg,afwa_diag_opt=1,afwa_therm_opt=1,
                   afwa_ptype_opt=1,afwa_radar_opt=1,afwa_vil_opt=1,afwa_vis_opt=1)}
    inventories = {key: {name: list(shape) for name, shape in produced_history_shapes(value).items()}
                   for key, value in probes.items()}
    baseline_names = set(inventories["baseline"])
    return {"qualification": "writer inventory from produced_history_shapes, not native numeric equivalence",
            "source": "woof/io/history_layout.py:743",
            "config": str(path),
            "inventories": inventories,
            "deltas_from_baseline": {key: {"added": sorted(set(value)-baseline_names),
                                            "removed": sorted(baseline_names-set(value))}
                                     for key, value in inventories.items()}}


def coverage(key):
    if re.search(r"auxhist(22|23)(_|$)", key):
        return "Native PLD/ZLD stream control; active streams use the asynchronous Rust auxhist22/23 writer with per-domain native alarms. Inactive stream controls remain inert."
    if key.startswith("afwa_"):
        if key in ('afwa_diag_opt','afwa_therm_opt','afwa_ptype_opt','afwa_vil_opt','afwa_radar_opt','afwa_vis_opt','afwa_ptype_ccn_tmp','afwa_ptype_tot_melt','afwa_bad_data_check'):
            return "Native AFWA CUDA producers, held atmospheric time levels, history alarms and persistent/rebuilt carriers are wired. Numerical, forecast, lifecycle and timing qualification is external to this census."
        return "This AFWA producer/check remains refused by name; no substituted product is selected. See the lane report for remaining native algorithms and undefined operands."
    if key == "nwp_diagnostics":
        return ("Native NWP GPU driver supplies all nine history fields, first-RK helicity and native interval clocks. "
                "GSD uses its separately pinned fork; HAILCAST is a separate product. Numeric qualification is external to this census.")
    if key == "diag_nwp2":
        return "Traditional-fields diagnostics is a distinct native routine; no stock diag_nwp2 driver is ported here."
    if key in ("p_lev_diags", "p_lev_diags_dfi", "num_press_levels", "press_levels", "p_lev_missing",
               "z_lev_diags", "z_lev_diags_dfi", "num_z_levels", "z_levels", "z_lev_missing",
               "use_tot_or_hyd_p", "extrap_below_grnd"):
        return "Native PLD/ZLD CUDA driver and Rust auxhist23/22 output are wired from stock &diags. DFI companions are admitted, while the existing full-DFI gate still names its missing integration. Numeric and lifecycle qualification is external to this census."
    if key == "cu_diag":
        return "Prior cu_diag implementation already present; qualified prior evidence must be retained, not re-claimed by this census."
    if key == "hailcast_opt":
        return "Existing HAILCAST state, kernels, history and restart; not proof of stock NWP PSD hail."
    if key == "prec_acc_dt":
        return "Native WRF 4.7.1 precipitation amounts, exact restart-relative reset clock and conditional snow clamp use CUDA; restart, history, nests, ordinary ensembles and streamed domains carry the amounts and clock. Qualification receipts are external to this census."
    if key in ("bucket_mm", "bucket_j"):
        return "Native hourly single-subtraction CUDA buckets, all eighteen int32 carry counters, the nested water/energy gate and restart-relative clock are wired through history, restart, nests, ensembles and streamed domains. Energy bucketing requires the existing RRTMG boundary-flux carriers (or inactive radiation); other active radiation producers remain refused by name. Qualification is external to this census."
    if key in ("acc_phy_tend", "bucket_j", "bucket_mm", "diag_print", "noahmp_output", "noahmp_acc_dt"):
        return "Native module_diag_misc control; importer result and source references below distinguish inert, refused and implemented paths."
    if key == "do_radar_ref":
        return "Scheme-specific reflectivity paths exist. module_diag_refl itself is NMM-only; ARW uses microphysics reflectivity."
    return "Adjacent diagnostic control inventoried for completeness; outside the six-module port row. No qualification inferred."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from woof.fortran_namelist import parse_namelist_text
    from woof.namelist_import import import_parsed_namelists

    constants, fixture_sha = fixture()
    wps = parse_namelist_text(constants["WPS_TEXT"])
    inp = parse_namelist_text(constants["INPUT_TEXT"])
    options = dict(wps_path="census-namelist.wps", input_path="census-namelist.input", name="diag-census")
    baseline, _ = import_parsed_namelists(wps, inp, source_text=constants["INPUT_TEXT"], **options)
    registry_path = ROOT / "woof/data/wrf_namelist/registry_v4.7.1.json"
    registry = json.loads(registry_path.read_text())
    rows = []
    for row in [r for r in registry["keys"] if selected(r)] + generated_io_rows():
        work = copy.deepcopy(inp)
        value = probe_value(row)
        work.setdefault(row["section"], {})[row["key"]] = value
        result = {**row, "probe_value": value, "implementation": coverage(row["key"]),
                  "candidate_references": source_hits(row["key"]),
                  "native_references": native_references(row["key"])}
        if "generated_source" in row:
            result["native_references"][row["generated_source"]] = [row["generator_statement"]]
        try:
            text, report = import_parsed_namelists(wps, work, source_text=None, **options)
            result.update(importer="accepted", changes_toml=text != baseline,
                          report=report.format())
        except ValueError as exc:
            result.update(importer="refused", error=str(exc))
        rows.append(result)
    paired = []
    for label, changes in (
            ("cu_diag with Grell-Freitas", {"physics": {"cu_physics": [3, 3], "cu_diag": [1, 1]},
                                           "domains": {"e_vert": [33, 33],
                                                       "eta_levels": [1. - k/32. for k in range(33)]}}),
            ("radar independent of lightning", {"physics": {"do_radar_ref": [1], "lightning_option": [0]}}),
            ("pressure levels complete request", {"diags": {"p_lev_diags": [1], "num_press_levels": [2], "press_levels": [85000., 50000.]},
                                                 "time_control": {"io_form_auxhist23": [2], "auxhist23_interval": [60], "frames_per_auxhist23": [1]}}),
            ("height levels complete request", {"diags": {"z_lev_diags": [1], "num_z_levels": [2], "z_levels": [1000., 3000.]},
                                               "time_control": {"io_form_auxhist22": [2], "auxhist22_interval": [60], "frames_per_auxhist22": [1]}}),
            ("AFWA complete base request", {"afwa": {"afwa_diag_opt": [1, 1]}}),
            ("AFWA six packages", {"afwa": {k:[1,1] for k in ('afwa_diag_opt','afwa_therm_opt','afwa_ptype_opt','afwa_vil_opt','afwa_radar_opt','afwa_vis_opt')}})):
        work = copy.deepcopy(inp)
        for section, values in changes.items():
            work.setdefault(section, {}).update(values)
        result = {"label": label, "changes": changes}
        try:
            text, report = import_parsed_namelists(wps, work, source_text=None, **options)
            result.update(importer="accepted", changes_toml=text != baseline, report=report.format())
        except ValueError as exc:
            result.update(importer="refused", error=str(exc))
        paired.append(result)
    receipt = {"schema": "wrf471-diagnostic-census-v1", "wrf": registry["wrf"],
               "candidate": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
               "fixture_sha256": fixture_sha,
               "command": "python3 tools/diag_wrf471_oracle/census.py --output <receipt-dir>",
               "qualification": "none; importer probes are not forecast or output validation", "controls": rows,
               "paired_probes": paired,
               "candidate_file_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                         for p in ("woof/namelist_import.py", "woof/config.py")}}
    (args.output / "census.json").write_text(json.dumps(receipt, indent=2) + "\n")
    md = ["# WRF 4.7.1 diagnostic census", "", f"Candidate `{receipt['candidate']}`. {len(rows)} controls.",
          "Importer acceptance establishes parsing only. Exact probe errors and line references are in census.json.", "",
          "| Section | Control | Default | Nondefault import | Candidate implementation |", "|---|---|---|---|---|"]
    for r in rows:
        status = r["importer"] + ("; TOML unchanged" if r.get("changes_toml") is False else "")
        md.append(f"| {r['section']} | `{r['key']}` | `{r['default']}` | {status} | {r['implementation']} |")
    (args.output / "census.md").write_text("\n".join(md) + "\n")
    (args.output / "field-inventory.json").write_text(json.dumps(field_inventory(), indent=2) + "\n")
    (args.output / "candidate-written-fields.json").write_text(
        json.dumps(candidate_inventory(baseline, args.output), indent=2) + "\n")
    print(json.dumps({"controls": len(rows), "accepted": sum(r["importer"] == "accepted" for r in rows),
                      "refused": sum(r["importer"] == "refused" for r in rows), "receipt": str(args.output / "census.json")}))


if __name__ == "__main__":
    main()
