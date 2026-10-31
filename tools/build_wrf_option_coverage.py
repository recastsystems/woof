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
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
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
    ("sf_surface_physics", 8): {"num_soil_layers": 3},
    # KIAPS SAS shallow convection runs only beside KSAS (README.namelist).
    ("shcu_physics", 4): {"cu_physics": 14},
    # Deng works with MYJ (README.namelist).
    ("shcu_physics", 5): {"bl_pbl_physics": 2, "sf_sfclay_physics": 2},
    ("sf_ocean_physics", 2): {},
    # WRF's cumulus-to-radiation feedback is defined for GF, G3, GD and KF.
    ("cu_rad_feedback", True): {"cu_physics": 3},
}

#: Selectors probed as a longwave/shortwave pair when WRF declares the value
#: for both; a value declared for one side only is probed beside RRTMG.
_RADIATION = ("ra_lw_physics", "ra_sw_physics")


def _fmt(value) -> str:
    if value is True:
        return ".true."
    if value is False:
        return ".false."
    return str(value)


def _pair(tmp: Path, physics: dict) -> tuple[Path, Path]:
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
                _, report = import_namelists(*_pair(Path(tmp), physics), **options)
            except Exception as error:  # every refusal the door can raise
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
        return {"imports": True, "physics": physics, "options": options,
                "substituted": swapped}
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


def build() -> tuple[str, list[str]]:
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
        "importer writes a configuration that runs that option. **Refused** means",
        "the importer stops and names the option; the reason is the importer's own",
        "message. A refused option never runs as a different scheme; an option",
        "that imports as a different scheme says so on its row. Other WRF",
        "namelist options (gravity-wave drag, lightning, aerosol and the",
        "rest) are not on this page.",
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


def main(argv: list[str]) -> int:
    text, problems = build()
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2
    if "--check" in argv:
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
