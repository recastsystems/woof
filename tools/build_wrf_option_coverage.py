"""Generate docs/public/WRF-OPTIONS.md by asking the importer about every WRF option.

Every value WRF v4.7.1 declares for its physics selectors
(woof/data/wrf_namelist/physics_options_v4.7.1.json, extracted from WRF's
Registry and README.namelist by tools/wrf_physics_options/extract.py) is put
through ``woof.namelist_import.import_namelists`` -- the door
``woof import-namelist`` runs -- on a one-domain namelist pair.  The page
lists what imports and, for what does not, the importer's own reason.
Nothing on the page is written by hand: a release's "not in this release"
list is this table, not prose.

The probe pairs each selector value with the partner WRF itself requires
(:data:`PARTNERS`: MYJ with its Eta surface layer, Noah-MP with four soil
layers, and so on), so a row says whether the OPTION imports, not whether
an arbitrary neighbour fits it.  A refusal that does not name the probed
key is a probe defect, and the build fails on it rather than publishing it.

Usage:
  python tools/build_wrf_option_coverage.py            # write the page
  python tools/build_wrf_option_coverage.py --check    # exit 1 if stale
  python tools/build_wrf_option_coverage.py --check --audit-json audit.json

The audit probes every ARW Registry key at its default in the same fixture.
It records consumed-but-dropped keys and unrelated fixture failures separately.
It is an admission check only, never a substitute for a real forecast.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import argparse
import hashlib
import subprocess
import tomllib
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OPTIONS = ROOT / "woof" / "data" / "wrf_namelist" / "physics_options_v4.7.1.json"
PAGE = ROOT / "docs" / "public" / "WRF-OPTIONS.md"

#: The probe's base selection: a plain one-domain WRF CONUS-style suite that
#: imports as it is (Thompson, RRTMG, revised MM5 surface layer, Noah, YSU,
#: no cumulus).
BASE = {
    "mp_physics": 8, "ra_lw_physics": 4, "ra_sw_physics": 4,
    "sf_sfclay_physics": 1, "sf_surface_physics": 2, "bl_pbl_physics": 1,
    "cu_physics": 0, "num_soil_layers": 4, "sf_urban_physics": 0,
    "radt": 10,
}

#: The partner settings WRF itself requires beside a probed value.  Sources:
#: WRF v4.7.1 phys/module_physics_init.F and share/module_check_a_mundo.F
#: (PBL/surface-layer pairings, soil layer counts), README.namelist.
PARTNERS: dict[tuple[str, object], dict[str, object]] = {
    ("bl_pbl_physics", 0): {"sf_sfclay_physics": 1},
    ("bl_pbl_physics", 2): {"sf_sfclay_physics": 2},
    ("bl_pbl_physics", 3): {"sf_sfclay_physics": 3},
    ("bl_pbl_physics", 4): {"sf_sfclay_physics": 4},
    ("bl_pbl_physics", 5): {"sf_sfclay_physics": 5},
    ("bl_pbl_physics", 7): {"sf_sfclay_physics": 7, "sf_surface_physics": 7,
                            "num_soil_layers": 2},
    ("bl_pbl_physics", 10): {"sf_sfclay_physics": 10},
    # No surface layer: WRF's land models read its exchange coefficients.
    ("sf_sfclay_physics", 0): {"bl_pbl_physics": 0, "sf_surface_physics": 0,
                               "num_soil_layers": 5},
    ("sf_sfclay_physics", 2): {"bl_pbl_physics": 2},
    ("sf_sfclay_physics", 3): {"bl_pbl_physics": 3},
    ("sf_sfclay_physics", 4): {"bl_pbl_physics": 4},
    ("sf_sfclay_physics", 5): {"bl_pbl_physics": 5},
    ("sf_sfclay_physics", 7): {"bl_pbl_physics": 7, "sf_surface_physics": 7,
                               "num_soil_layers": 2},
    ("sf_sfclay_physics", 10): {"bl_pbl_physics": 10},
    ("sf_surface_physics", 0): {"num_soil_layers": 5},
    ("sf_surface_physics", 1): {"num_soil_layers": 5},
    ("sf_surface_physics", 3): {"num_soil_layers": 9},
    ("sf_surface_physics", 5): {"num_soil_layers": 10},
    ("sf_surface_physics", 7): {"num_soil_layers": 2, "sf_sfclay_physics": 7,
                                "bl_pbl_physics": 7},
    ("sf_surface_physics", 8): {"num_soil_layers": 3, "fractional_seaice": 1},
    # KIAPS SAS shallow convection runs only beside KSAS (README.namelist).
    ("shcu_physics", 4): {"cu_physics": 14},
    # Deng works with MYJ (README.namelist).
    ("shcu_physics", 5): {"bl_pbl_physics": 2, "sf_sfclay_physics": 2,
                           "icloud_bl": 0},
    ("sf_ocean_physics", 2): {},
    # WRF's cumulus-to-radiation feedback is defined for GF, G3, GD and KF.
    ("cu_rad_feedback", True): {"cu_physics": 3},
    # The candidate's Fitch coupling injects TKE through MYNN only.
    # WRF module_physics_init.F: windfarm initialization accepts MYNN;
    # namelist_import.py enforces this narrower runtime contract.
    ("windfarm_opt", 1): {"bl_pbl_physics": 5, "sf_sfclay_physics": 5},
    ("windfarm_opt", 2): {"bl_pbl_physics": 5, "sf_sfclay_physics": 5,
                          "windfarm_ij": 2},
    # With no radiative forcing the land-energy solver has no GLW input.
    ("ra_lw_physics", 0): {"sf_surface_physics": 0, "num_soil_layers": 5},
    ("ra_sw_physics", 0): {"sf_surface_physics": 0, "num_soil_layers": 5},
    ("aer_opt", 3): {"mp_physics": 28},
    ("lightning_option", 1): {"do_radar_ref": 1},
    ("lightning_option", 2): {"do_radar_ref": 1},
    ("lightning_option", 11): {"cu_physics": 5},
    # Clear-sky optics are admitted on the RRTM/Dudhia pair, not RRTMGP.
    ("icloud", 0): {"ra_lw_physics": 1, "ra_sw_physics": 1},
}

#: Selectors probed as a longwave/shortwave pair when WRF declares the value
#: for both; a value declared for one side only is probed beside RRTMG.
_RADIATION = ("ra_lw_physics", "ra_sw_physics")

# A synthetic, explicitly labelled input for schema admission. Native WRF
# turbine text additionally needs rw-wind-table and is a separate IO check.
# The production reader accepts JSON by content, including at this filename.
WIND_INPUT = {
    "schema": "gpuwm-wind-farm.v1",
    "types": {"probe": {"curve": [[0, 0, 0], [10, 0.8, 0.4], [30, 0, 0]]}},
    "turbines": [{"id": "probe", "type": "probe", "hub_height_m": 100,
                  "rotor_diameter_m": 100, "latitude": 35, "longitude": -97}],
}

# Synthetic MAV admission input in WRF's native geographic and curve formats.
# This exercises the real Rust decoder; it is not forecast or native parity proof.
MAV_INPUT = "35 -97 1 1\n"
MAV_CURVE = "3\n100 100 0.1 2\n4 0.8 50\n10 0.8 1000\n25 0.05 2000\n"

# WRF 4.7.1 share/module_check_a_mundo.F:3401,3410. These are source
# expectations, independent of the candidate's 4.6.1 compatibility table.
NATIVE_NSSL_CCN = {17: 1, 22: 1}


def _fmt(value) -> str:
    if value is True:
        return ".true."
    if value is False:
        return ".false."
    return str(value)


def _pair(tmp: Path, physics: dict) -> tuple[Path, Path]:
    if physics.get("windfarm_opt", 0) == 2:
        (tmp / "windturbines-ll.txt").write_text(MAV_INPUT, encoding="utf-8")
        (tmp / "wind-turbine-1.tbl").write_text(MAV_CURVE, encoding="utf-8")
    elif physics.get("windfarm_opt", 0):
        (tmp / "windturbines.txt").write_text(json.dumps(WIND_INPUT), encoding="utf-8")
    levels = 40
    eta = ", ".join(f"{1.0 - i / levels:.17g}" for i in range(levels + 1))
    wps = """&share
 wrf_core = 'ARW', max_dom = 1,
 start_date = '2020-05-01_00:00:00', end_date = '2020-05-01_06:00:00',
 interval_seconds = 3600,
/
&geogrid
 parent_id = 1, parent_grid_ratio = 1, i_parent_start = 1, j_parent_start = 1,
 e_we = 61, e_sn = 51, geog_data_res = 'default', dx = 12000, dy = 12000,
 map_proj = 'lambert', ref_lat = 35.0, ref_lon = -97.0, truelat1 = 30.0,
 truelat2 = 60.0, stand_lon = -97.0, geog_data_path = '/geog',
/
&ungrib
 out_format = 'WPS', prefix = 'SOURCE',
/
&metgrid
 fg_name = 'SOURCE',
/
"""
    lines = "\n".join(f" {key} = {_fmt(value)}," for key, value in physics.items())
    inp = f"""&time_control
 run_hours = 6, start_year = 2020, start_month = 5, start_day = 1, start_hour = 0,
 end_year = 2020, end_month = 5, end_day = 1, end_hour = 6,
 input_from_file = .true., history_interval = 60,
/
&domains
 time_step = 60, max_dom = 1, e_we = 61, e_sn = 51, e_vert = {levels + 1},
 eta_levels = {eta}, p_top_requested = 5000, dx = 12000.0, dy = 12000.0,
 grid_id = 1, parent_id = 0, i_parent_start = 1, j_parent_start = 1,
 parent_grid_ratio = 1, parent_time_step_ratio = 1, feedback = 0,
 smooth_option = 0,
/
&physics
{lines}
/
&dynamics
 hybrid_opt = 2, etac = 0.2, use_theta_m = 0, diff_opt = 2, km_opt = 4,
 mix_full_fields = .true.,
/
&bdy_control
 spec_bdy_width = 5, specified = .true., nested = .false.,
/
"""
    wps_path, input_path = tmp / "namelist.wps", tmp / "namelist.input"
    wps_path.write_text(wps, encoding="utf-8")
    input_path.write_text(inp, encoding="utf-8")
    return wps_path, input_path


def _reason(message: str, keys: tuple[str, ...]) -> str | None:
    """The refusal sentence that names one of ``keys``, or None."""
    text = " ".join(message.split())
    for key in keys:
        # Numbered multi-refusal ("1. &physics key = ...") or one sentence.
        for part in re.split(r"(?:^|\s)\d+\.\s(?=&|[a-z_]+=)", text):
            if re.search(rf"\b{re.escape(key)}\b", part):
                return part.strip()
    return None


def probe(key: str, value) -> dict:
    """Import a one-domain pair with ``key = value``; say what happened."""
    from woof.namelist_import import import_namelists

    physics = dict(BASE)
    probe_keys: tuple[str, ...] = (key,)
    physics[key] = value
    if key in _RADIATION:
        table = json.loads(OPTIONS.read_text(encoding="utf-8"))["selectors"]
        other = _RADIATION[1 - _RADIATION.index(key)]
        if any(row["value"] == value for row in table[other]):
            physics[other] = value
            probe_keys = (key, other)
    physics.update(PARTNERS.get((key, value), {}))
    first = None
    for options in ({}, {"rrtmg_variant": "rrtmg_legacy"}):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                text, report = import_namelists(*_pair(Path(tmp), physics), **options)
            except (ValueError, NotImplementedError) as error:
                message = str(error)
                result = {"imports": False, "message": message,
                          "reason": _reason(message, probe_keys),
                          "physics": physics, "options": options}
                first = first or result
                # WRF's RRTMG 4/4 imports as RTE+RRTMGP by default; a
                # scheme that has no RRTMGP cloud-optics coupling is
                # refused there and imports with WRF's own RRTMG
                # (--rrtmg-variant rrtmg_legacy).  Ask that door too.
                if "rrtmg_legacy" in message and not options:
                    continue
                return first if first["reason"] else result
        # A probed option the importer turns into a different scheme (WRF's
        # RRTMG 4/4 runs as RTE+RRTMGP by default) says so on its row.
        swapped = [f"{sub.wrf_name} -> {sub.gpuwm_name}"
                   for sub in getattr(report, "substitutions", ())
                   if set(str(sub.key).split("/")) & set(probe_keys)]
        resolved = tomllib.loads(text)
        differences = []
        if key == "mp_physics" and value in NATIVE_NSSL_CCN:
            actual = resolved.get("shared", {}).get("nssl_ccn_on")
            if actual != NATIVE_NSSL_CCN[value]:
                differences.append({"key": "nssl_ccn_on", "candidate": actual,
                                    "native_wrf471": NATIVE_NSSL_CCN[value],
                                    "source": "share/module_check_a_mundo.F:3397-3416"})
        return {"imports": True, "physics": physics, "options": options,
                "substituted": swapped,
                "native_differences": differences,
                "resolved": {"shared": resolved.get("shared", {}),
                             "domains": resolved.get("domain", [])},
                "report": {kind: [asdict(entry) for entry in getattr(report, kind)
                                  if set(str(entry.key).split("/")) & set(probe_keys)]
                           for kind in ("substitutions", "fixed", "dropped",
                                        "defaults_applied", "translated")}}
    return first


#: Words added after WRF's own label where that label misleads.  WRF's
#: README.namelist calls 91 the "Old MM5 scheme (previously option 1)", and a
#: WRF user read 91 as today's option 1: it is the pre-v3.6 scheme, and
#: option 1 is now the revised MM5 (Jimenez et al. 2012).  woof runs each
#: as its own scheme and imports each as written.
LABEL_NOTES: dict[tuple[str, object], str] = {
    ("sf_sfclay_physics", 1): "WRF's option 1 since v3.6; not the same scheme as 91",
    ("sf_sfclay_physics", 91): ("the pre-v3.6 MM5 scheme, a different scheme from "
                                "today's option 1 (revised MM5)"),
}


def _clean(reason: str) -> str:
    """The importer's reason, without the probe's temporary file paths, cut
    before the sentence that lists every implemented value (the table's own
    "imports" rows say that)."""
    reason = re.sub(r"\S*namelist\.(input|wps)", r"namelist.\1", reason)
    reason = re.sub(r"^woof import-namelist:\s*", "", reason)
    reason = re.split(r"\s(?:mp_physics must be 0 \(off\)|Next:)", reason)[0]
    return reason.replace("|", "/").strip()


def build(records: list | None = None) -> tuple[str, list[str]]:
    table = json.loads(OPTIONS.read_text(encoding="utf-8"))
    problems: list[str] = []
    out = [
        "# WRF physics options in woof",
        "",
        "Generated by `tools/build_wrf_option_coverage.py`; do not edit by hand.",
        "",
        "Every value of WRF v4.7.1's physics scheme selectors and the on/off",
        "switches below was put through `woof import-namelist` on a",
        "one-domain namelist, beside the partner settings WRF itself requires",
        "(for example MYJ with its Eta surface layer). **Imports** means the",
        "importer writes a configuration and validates its schema. It does not",
        "prove GPU execution, numerical agreement, a forecast, restart or speed.",
        "**Refused** means",
        "the importer stops and names the option; the reason is the importer's own",
        "message. A refused option never runs as a different scheme; an option",
        "that imports as a different scheme says so on its row. Other WRF",
        "namelist options (gravity-wave drag and the",
        "rest) are inventoried separately by `--audit-json` at their Registry",
        "defaults. A default probe is not a claim about every possible value.",
        "The JSON receipt retains the exact companion settings, resolved",
        "selectors, substitutions, and refusal messages. Forecast qualification",
        "and per-stage speed require separate receipts from the scheme lanes.",
        "",
        f"WRF source: {table['wrf']['url']} at {table['wrf']['tag']} "
        f"({table['wrf']['commit'][:12]}).",
        "",
    ]
    summary: list[str] = []
    for kind in ("selectors", "switches"):
        for key, rows in table[kind].items():
            ok, refused = [], []
            body = ["", f"## `{key}`", "", "| value | WRF option | woof |",
                    "|---|---|---|"]
            for row in rows:
                value, label = row["value"], row["label"]
                if (key, value) in LABEL_NOTES:
                    label = f"{label}: {LABEL_NOTES[(key, value)]}"
                result = probe(key, value)
                if records is not None:
                    records.append({"key": key, "value": value, "label": label,
                                    **result})
                if result["imports"]:
                    status = ("imports with `--rrtmg-variant rrtmg_legacy` "
                              "(WRF's own RRTMG)" if result["options"]
                              else "imports")
                    if result.get("substituted"):
                        status += (" as " + "; ".join(result["substituted"])
                                   .replace("|", "/"))
                        if key in _RADIATION and not result["options"]:
                            status += ("; `--rrtmg-variant rrtmg_legacy` "
                                       "runs WRF's own RRTMG")
                    if key == "mp_physics" and value in (17, 19, 21, 22):
                        status += "; legacy NSSL spelling, rewritten to 18 plus variant flags"
                    for difference in result["native_differences"]:
                        status += (f"; **differs from WRF 4.7.1**: "
                                   f"`{difference['key']}={difference['candidate']}` "
                                   f"(native: `{difference['native_wrf471']}`)")
                    if key == "windfarm_opt" and value == 1:
                        status += ("; synthetic JSON turbine input fixture; native WRF "
                                   "turbine text needs `rw-wind-table` (not checked here)")
                    if key == "windfarm_opt" and value == 2:
                        status += ("; synthetic native MAV turbine input decoded by "
                                   "`rw-wind-table`; no forecast proof")
                    if key == "aer_opt" and value:
                        variant = result["resolved"]["shared"]["ra_rrtmg_variant"]
                        status += f"; resolves `ra_rrtmg_variant={variant}`"
                    partners = PARTNERS.get((key, value), {})
                    if partners:
                        status += "; companions: " + ", ".join(
                            f"`{k}={_fmt(v)}`" for k, v in partners.items())
                    ok.append(value)
                elif result["reason"] is None:
                    problems.append(
                        f"{key} = {value!r}: refused without naming {key}: "
                        f"{result['message'][:400]}")
                    status = "probe defect"
                else:
                    status = "refused: " + _clean(result["reason"])
                    refused.append(value)
                body.append(f"| `{_fmt(value)}` | {label} | {status} |")
            out.extend(body)
            summary.append(f"| `{key}` | {', '.join(map(_fmt, ok)) or '-'} | "
                           f"{', '.join(map(_fmt, refused)) or '-'} |")
    head = ["", "## Summary", "", "| selector | imports | refused |", "|---|---|---|",
            *summary]
    split = next(i for i, line in enumerate(out) if line.startswith("WRF source:")) + 2
    return "\n".join(out[:split] + head + out[split:]) + "\n", problems


def registry_audit() -> list[dict]:
    """One default probe per Registry row, through the real translation.

    No inferred support for untested values or pairings. In particular a key
    consumed and dropped is not reported as an executable feature.
    """
    from copy import deepcopy
    from woof.fortran_namelist import parse_namelist
    from woof.namelist_contract import _registry_column
    from woof.namelist_import import import_parsed_namelists
    from woof.wrf_namelist_registry import REGISTRY_TABLE, wrf_namelist_table

    source_lines = {}
    section = None
    for number, line in enumerate(REGISTRY_TABLE.read_text().splitlines(), 1):
        if '"section":' in line:
            section = json.loads(line.split(":", 1)[1].strip().rstrip(","))
        if '"key":' in line:
            key = json.loads(line.split(":", 1)[1].strip().rstrip(","))
            source_lines[(section, key)] = number
    with tempfile.TemporaryDirectory() as tmp:
        paths = _pair(Path(tmp), BASE)
        base = tuple(parse_namelist(p) for p in paths)
    # A broken fixture must stop the audit, not poison 1,115 support rows.
    path_options = {"wps_path": Path("namelist.wps"),
                    "input_path": Path("namelist.input")}
    import_parsed_namelists(*base, **path_options)
    out = []
    for declaration in wrf_namelist_table()["keys"]:
        group, key = declaration["section"], declaration["key"]
        row = {**declaration, "source": str(REGISTRY_TABLE.relative_to(ROOT)),
               "line": source_lines[(group, key)]}
        try:
            column = _registry_column(declaration, 1)
        except (ValueError, OverflowError) as error:
            row.update(status="unparseable-default", message=str(error))
            out.append(row)
            continue
        wps, inp = deepcopy(base)
        inp.setdefault(group, {})[key] = column
        row["probed"] = column
        if group == "time_control" and key.startswith("start_"):
            # WPS and WRF must describe the same six-hour case. Moving a
            # Registry start default alone tests a contradictory fixture,
            # rather than whether the importer consumes that option.
            from datetime import datetime, timedelta
            parts = ("year", "month", "day", "hour", "minute", "second")
            clock = inp[group]
            start = datetime(*(int(clock.get(f"start_{part}", [0])[0])
                               for part in parts))
            end = start + timedelta(hours=6)
            companions = {}
            for part in parts:
                clock[f"end_{part}"] = [getattr(end, part)]
                companions[f"end_{part}"] = clock[f"end_{part}"]
            for name, value in (("start_date", start), ("end_date", end)):
                wps["share"][name] = [value.strftime("%Y-%m-%d_%H:%M:%S")]
                companions[f"share/{name}"] = wps["share"][name]
            row["fixture_companions"] = companions
        try:
            _, report = import_parsed_namelists(wps, inp, **path_options)
        except (ValueError, NotImplementedError) as error:
            message = str(error)
            # Structured refusal identity takes precedence over mere mention.
            keys = getattr(error, "keys", ())
            named = ((group, key) in keys if keys else
                     re.search(rf"\b{re.escape(key)}\b", message) is not None)
            section_refused = ("unknown section(s)" in message and
                               repr(group) in message.split("; known:", 1)[0])
            status = ("section-refused" if section_refused else
                      "default-refused" if named else "fixture-blocked")
            row.update(status=status,
                       message=_clean(message))
        else:
            classifications = [kind for kind in ("translated", "fixed", "dropped")
                               if any(entry.section == group and entry.key == key
                                      for entry in getattr(report, kind))]
            row.update(status="/".join(classifications) or "unclassified",
                       report={kind: [asdict(entry) for entry in getattr(report, kind)
                                      if entry.section == group and entry.key == key]
                               for kind in classifications})
        out.append(row)
    return out


def gate_conflicts() -> list[str]:
    """Find named unported guards that contradict implemented Registry rows.

    Pairing refusals are retained in the probes, rather than guessed to be
    stale. A conflict is a handoff to the owning lane, not permission to
    remove a gate without its replay and forecast.
    """
    from woof.config import unported_scheme_reasons
    registry = json.loads((ROOT / "woof/physics_registry_v2.json").read_text())
    conflicts = []
    for component, spec in registry["components"].items():
        for option, declaration in spec["options"].items():
            if not declaration.get("implemented"):
                continue
            for key, value in declaration.get("selectors", {}).items():
                reason = unported_scheme_reasons(key).get(value)
                if reason:
                    conflicts.append(f"{component}/{option}: implemented registry row "
                                     f"still has {key}={value} unported guard: {reason}")
    return conflicts


def audit_document(records: list[dict], problems: list[str]) -> dict:
    paths = ("woof/namelist_import.py", "woof/config.py",
             "woof/physics_registry_v2.json",
             "woof/data/wrf_namelist/registry_v4.7.1.json",
             "woof/data/wrf_namelist/physics_options_v4.7.1.json",
             "tools/build_wrf_option_coverage.py")
    rows = registry_audit()
    conflicts = gate_conflicts()
    return {
        "schema": "gpuwm.wrf-option-admission-audit.v1",
        "wrf": json.loads(OPTIONS.read_text())["wrf"],
        "candidate": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                          for p in paths},
        "scope": "Import and emitted-config validation only. No GPU execution, "
                 "module replay, forecast, restart, nest, ensemble or timing proof.",
        "base_physics": BASE,
        "wind_input_fixture": {
            "fitch": {"format": "synthetic woof JSON, not native WRF text",
                      "data": WIND_INPUT},
            "mav": {"format": "synthetic native WRF MAV text through rw-wind-table",
                    "windturbines-ll.txt": MAV_INPUT,
                    "wind-turbine-1.tbl": MAV_CURVE}},
        "selector_counts": dict(Counter("imports" if r["imports"] else "refused"
                                        for r in records)),
        "registry_counts": dict(Counter(r["status"] for r in rows)),
        "selectors": records, "registry_defaults": rows,
        "probe_problems": problems, "gate_conflicts": conflicts,
        "native_contract_differences": [
            {"key": r["key"], "value": r["value"],
             "differences": r["native_differences"]}
            for r in records if r.get("native_differences")],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--require-native", action="store_true",
                        help="Fail on observed selector rewrites that differ from WRF 4.7.1")
    parser.add_argument("--baseline-json", type=Path,
                        help="Prior audit; list newly importing rows for forecast follow-up")
    parser.add_argument("--speed-csv", type=Path,
                        help="Write admitted rows with timing explicitly unmeasured")
    args = parser.parse_args(argv[1:])
    if args.baseline_json and not args.audit_json:
        parser.error("--baseline-json requires --audit-json")
    records: list[dict] = []
    text, problems = build(records)
    if args.require_native:
        problems.extend(f"{r['key']}={r['value']}: {r['native_differences']}"
                        for r in records if r.get("native_differences"))
    if args.audit_json:
        document = audit_document(records, problems)
        if args.baseline_json:
            previous = json.loads(args.baseline_json.read_text())
            old = {(r["key"], _fmt(r["value"])) for r in previous["selectors"]
                   if r["imports"]}
            document["newly_importing"] = [
                {"key": r["key"], "value": r["value"],
                 "forecast": "pending scheme receipt"}
                for r in records if r["imports"] and (r["key"], _fmt(r["value"])) not in old]
        args.audit_json.write_text(json.dumps(document, indent=2, default=str) + "\n")
        problems.extend(document["gate_conflicts"])
    if args.speed_csv:
        import csv
        with args.speed_csv.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("selector", "value", "companions", "import_options",
                             "qualification", "stage_ms", "nearest_scheme", "ratio"))
            for row in records:
                if row["imports"]:
                    writer.writerow((row["key"], _fmt(row["value"]),
                                     json.dumps(row["physics"], sort_keys=True),
                                     json.dumps(row["options"], sort_keys=True),
                                     "not measured in admission audit", "", "", ""))
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2
    if args.check:
        current = PAGE.read_text(encoding="utf-8") if PAGE.is_file() else ""
        if current != text:
            print(f"{PAGE.relative_to(ROOT)} is stale: run "
                  "python tools/build_wrf_option_coverage.py", file=sys.stderr)
            return 1
        return 0
    PAGE.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {PAGE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
