"""Write woof/data/wrf_namelist/physics_options_v4.7.1.json from a WRF tree.

The option VALUES come from WRF's own Registry ``package`` lines (every
``Registry/*`` file, so registry.sbm's spectral-bin schemes are included);
the LABELS come from WRF's ``run/README.namelist`` ``= N, label`` rows.  A
value with a package and no README label is labelled with its package name.
README-only values are included too: WRF rewrites the legacy NSSL selectors
17/19/21/22 before package selection, and radiation-off has no package.

Usage: python tools/wrf_physics_options/extract.py WRF_SOURCE_DIR OUT_JSON

The published table (docs/public/WRF-OPTIONS.md) is generated from this JSON
by tools/build_wrf_option_coverage.py, which asks the importer about every
row; nothing here decides what woof runs.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

#: The scheme selectors whose values WRF declares as Registry packages.
SCHEME_SELECTORS = (
    "mp_physics", "cu_physics", "shcu_physics", "bl_pbl_physics",
    "sf_sfclay_physics", "sf_surface_physics", "sf_urban_physics",
    "sf_ocean_physics", "ra_lw_physics", "ra_sw_physics",
)

#: On/off and enumerated &physics switches with no Registry package.  Each
#: value is WRF's own (Registry.EM_COMMON rconfig description or
#: README.namelist). The older on/off rows list active values only; aerosol,
#: cloud and lightning rows include their documented off/default values too.
SWITCHES = {
    "sst_update": [(1, "time-varying sea-surface temperature, sea ice, albedo and vegetation from wrflowinp")],
    "sst_skin": [(1, "skin sea-surface temperature (diurnal warm layer and cool skin)")],
    "tmn_update": [(1, "deep-soil temperature updated from the skin temperature (lagday)")],
    "sf_lake_physics": [(1, "CLM lake model")],
    "windfarm_opt": [(1, "Fitch wind farm"), (2, "Ma et al. (2022) wind farm with subgrid wakes")],
    "cu_rad_feedback": [(True, "cumulus sub-grid cloud fed to radiation")],
    "icloud": [(0, "no cloud optical effect"),
               (1, "Xu-Randall cloud fraction"),
               (2, "threshold cloud fraction"),
               (3, "Sundqvist cloud fraction (cal_cldfra3)")],
    "aer_opt": [(0, "no aerosol optical effect"),
                (1, "Tegen six-type aerosol climatology"),
                (2, "user aerosol optical properties"),
                (3, "Thompson water/ice friendly aerosol")],
    "lightning_option": [(0, "off"), (1, "PR92 maximum vertical velocity"),
                         (2, "PR92 20 dBZ echo top"),
                         (3, "lightning potential index"),
                         (11, "PR92 convective equilibrium level")],
}

#: Labels for package values WRF's README.namelist does not list.
LABEL_FALLBACK = {
    "omlscheme": "1-D ocean mixed layer",
    "pwp3dscheme": "3-D Price-Weller-Pinkel ocean",
}

_PACKAGE = re.compile(r"^package\s+(\S+)\s+([a-z_0-9]+)==(-?\d+)\b")


def _packages(root: Path) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {key: {} for key in SCHEME_SELECTORS}
    for path in sorted((root / "Registry").iterdir()):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _PACKAGE.match(line)
            if match and match.group(2) in out:
                out[match.group(2)].setdefault(int(match.group(3)), match.group(1))
    return out


def _labels(readme: str, key: str) -> dict[int, str]:
    lines = readme.splitlines()
    head = re.compile(rf"^ {re.escape(key)}\b")
    row = re.compile(r"^\s+=\s*(-?\d+(?:\s*,\s*-?\d+)*)\s*[,:]\s*(.*\S)\s*$")
    out: dict[int, str] = {}
    for index, line in enumerate(lines):
        if not head.match(line):
            continue
        first = re.search(r"=\s*(-?\d+)\s*[,:]\s*(.*\S)\s*$", line)
        if first:
            out.setdefault(int(first.group(1)), first.group(2))
        for follow in lines[index + 1:]:
            if follow.strip() and not follow.startswith(" " * 20):
                break
            match = row.match(follow)
            if match:
                for value in match.group(1).split(","):
                    out.setdefault(int(value), match.group(2))
        if out:
            break
    return out


def selector_rows(packages: dict, readme: str) -> dict:
    """Union of executable packages and documented compatibility selectors."""
    selectors = {}
    for key in SCHEME_SELECTORS:
        labels = _labels(readme, key)
        # The ocean README declares its off value inline, not on a '= N,' row.
        if key == "sf_ocean_physics":
            labels.setdefault(0, "no ocean model")
        if key == "sf_urban_physics" and 0 in labels:
            labels[0] = "no urban canopy model"
        selectors[key] = [
            {"value": value, "package": packages[key].get(value),
             "label": labels.get(value, LABEL_FALLBACK.get(
                 packages[key].get(value), packages[key].get(value)))}
            for value in sorted(set(packages[key]) | set(labels)) if value >= 0]
    return selectors


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    root, out = Path(argv[1]), Path(argv[2])
    readme_path = root / "run" / "README.namelist"
    readme = readme_path.read_text(encoding="utf-8", errors="replace")
    packages = _packages(root)
    selectors = selector_rows(packages, readme)
    switches = {key: [{"value": value, "label": label} for value, label in rows]
                for key, rows in SWITCHES.items()}
    sources = [{"file": str(p.relative_to(root)).replace("\\", "/"), "sha256": _sha(p)}
               for p in [readme_path, *sorted((root / "Registry").iterdir())]
               if p.is_file()]
    document = {"schema": "gpuwm-wrf-physics-options-v1",
                "wrf": {"tag": "v4.7.1",
                        "commit": "f52c197ed39d12e087d02c50f412d90d418f6186",
                        "url": "https://github.com/wrf-model/WRF"},
                "read_from": sources,
                "selectors": selectors, "switches": switches}
    out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
