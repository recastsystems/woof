"""Source-backed WRF ideal initialization plans, without allocating a device.

This is an inspection door. A recognized recipe is not an imported RunConfig
or an initialized atmosphere. Keep those outcomes separate in callers and
censuses until each source-order state builder has a runtime binding.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

from woof.fortran_namelist import parse_namelist


CATALOG = Path(__file__).parent / "data" / "wrf_ideal" / "cases.json"


def case_catalog():
    """Read fresh records so callers cannot mutate another plan's metadata."""
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def _one(document, section, name, default=None):
    entries = document.get(section, {}).get(name)
    return default if entries is None else entries[0]


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or (minimum is not None and value < minimum):
        bound = "" if minimum is None else f" >= {minimum}"
        raise ValueError(f"{name} must be an integer{bound}; otherwise the requested grid or clock cannot be constructed")
    return value


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive; otherwise the grid spacing or model height is invalid")
    return value


def _flag(document, name, domain):
    entries = document.get("bdy_control", {}).get(name, [False])
    # WRF initializes unsupplied per-domain array elements from the Registry,
    # rather than broadcasting a root switch onto its children.
    value = entries[domain] if domain < len(entries) else False
    if not isinstance(value, bool):
        raise ValueError(f"&bdy_control {name} must be logical; an integer would select an ambiguous boundary topology")
    return value


def boundary_topology(document, domain=0):
    """Retain both sides, including conflicts, instead of replacing them."""
    result = {}
    for axis in ("x", "y"):
        periodic = _flag(document, f"periodic_{axis}", domain)
        sides = {}
        for side in ("s", "e"):
            active = [kind for kind in ("open", "symmetric")
                      if _flag(document, f"{kind}_{axis}{side}", domain)]
            sides[side] = active
        topology = "periodic" if periodic else "unselected"
        if any(sides.values()):
            topology = "/".join("+".join(sides[side]) or "unselected" for side in ("s", "e"))
        conflict = any(len(v) > 1 for v in sides.values()) or (periodic and any(sides.values()))
        result[axis] = dict(periodic=periodic, sides=sides, topology=topology, conflict=conflict)
    result.update({name: _flag(document, name, domain) for name in ("specified", "nested", "polar")})
    return result


def select_case(document, path, case=None):
    """Use the Registry selector, an explicit choice, or an exact test directory.

    Dedicated fire, column and global initializers are selected at WRF build
    time. Their absence of &ideal is not evidence for selector zero or for a
    benchmark inferred from dimensions, physics or a renamed filename.
    """
    catalog = case_catalog()
    rows = catalog["cases"]
    by_name = {row["name"]: row for row in rows}
    supplied = _one(document, "ideal", "ideal_case")
    if supplied is not None:
        _integer(supplied, "&ideal ideal_case", minimum=0)
    selected = case
    if selected is None and supplied not in (None, 0):
        selected = next((row["name"] for row in rows if row["selector"] == supplied), None)
        if selected is None:
            raise ValueError(f"ideal_case={supplied} has no WRF 4.6.1 recipe in the table; guessing a thermal or terrain recipe would initialize a different atmosphere")
    if selected is None and Path(path).parent.name in by_name:
        selected = Path(path).parent.name
    if selected not in by_name:
        raise ValueError("WRF build-time ideal case is unresolved; pass --case with the original test directory name. Guessing from a sounding filename or grid would initialize a different atmosphere")
    row = by_name[selected]
    if supplied not in (None, 0) and row["selector"] != supplied:
        raise ValueError(f"--case {selected} conflicts with ideal_case={supplied}; they select different WRF initial atmospheres")
    if supplied == 0 and row["selector"] is not None:
        raise ValueError("ideal_case=0 selects real-data initialization; the selected generic ideal routine requires its nonzero Registry selector")
    return deepcopy(row), catalog


def inspect_namelist(path, *, case=None, input_directory=None):
    """Describe the requested state and all known missing runtime bindings.

    Original sections and every array entry remain in the receipt. Nothing is
    translated to real-data WPS, nor are unknown options silently consumed.
    Input files are inventoried only. Sounding/jet decoding belongs to Rust.
    """
    path = Path(path).resolve()
    document = parse_namelist(path)
    row, catalog = select_case(document, path, case)
    directory = path.parent if input_directory is None else Path(input_directory).resolve()
    domains = document.get("domains", {})
    n = _integer(_one(document, "domains", "max_dom", 1), "max_dom")
    grid = {}
    for source, target, start in (("e_we", "nx", "s_we"), ("e_sn", "ny", "s_sn"), ("e_vert", "nz", "s_vert")):
        end = _integer(_one(document, "domains", source), source)
        begin = _integer(_one(document, "domains", start, 1), start)
        if end <= begin:
            raise ValueError(f"{source} must exceed {start}; otherwise the mass grid is empty")
        grid[target] = end - begin
    for name in ("dx", "dy", "ztop"):
        grid[name] = _positive(_one(document, "domains", name), name)
    whole = _integer(_one(document, "domains", "time_step", 0), "time_step", minimum=None)
    num = _integer(_one(document, "domains", "time_step_fract_num", 0), "time_step_fract_num", minimum=None)
    den = _integer(_one(document, "domains", "time_step_fract_den", 1), "time_step_fract_den")
    dt = Fraction(whole) + Fraction(num, den)
    if dt == 0:
        raise ValueError("the requested fixed timestep is zero; it cannot advance an ideal atmosphere")
    grid["dt_rational"] = dict(numerator=dt.numerator, denominator=dt.denominator)
    topology = [boundary_topology(document, domain) for domain in range(n)]
    refusals = []
    if not row.get("runtime_binding"):
        refusals.append(dict(code="initialization_binding",breakage=row["missing_binding"],
                             wrf_evidence=row["evidence"],engine_evidence=row["engine_evidence"]))
    else:
        from woof.namelist_import import import_global_ideal_namelist
        try:
            import_global_ideal_namelist(path)
        except (ValueError,TypeError) as error:
            refusals.append(dict(code="global_admission",breakage=str(error)))
    if dt < 0:
        refusals.append(dict(code="reverse_clock", breakage="The source selects a negative timestep for backward integration. The engine's forward alarm clock cannot reproduce reversed forecast and forcing times; taking its absolute value would run a different trajectory."))
    if n > 1:
        refusals.append(dict(code="nested_initialization", breakage="The ideal plan has no multi-domain initialization and parent exchange binding. Using the root recipe independently would omit the requested parent interpolation and feedback state."))
    for domain, boundaries in enumerate(topology, 1):
        for axis in ("x", "y"):
            detail = boundaries[axis]
            if detail["conflict"]:
                refusals.append(dict(code="boundary_conflict", breakage=f"Domain {domain} {axis} simultaneously selects incompatible periodic, open or symmetry boundaries; no single edge operation can reproduce all of them."))
            if any("symmetric" in v for v in detail["sides"].values()):
                refusals.append(dict(code="symmetry_boundary", breakage=f"Domain {domain} {axis} requests a reflecting symmetry edge. RunConfig carries paired open edges or periodic wrap, so replacing symmetry with either would change normal flow at the boundary.", engine_evidence="woof/config.py:130"))
            if detail["sides"]["s"] != detail["sides"]["e"]:
                refusals.append(dict(code="asymmetric_boundary", breakage=f"Domain {domain} {axis} selects different edge operations. The engine's paired open switch would replace one requested edge operation.", engine_evidence="woof/config.py:130"))
        if boundaries["specified"]:
            refusals.append(dict(code="specified_forcing", breakage=f"Domain {domain} needs external wrfbdy forcing. A sounding recipe alone cannot supply that time-dependent boundary state."))
        if boundaries["polar"] and not row.get("runtime_binding"):
            refusals.append(dict(code="polar_topology", breakage=f"Domain {domain} needs global pole closure and polar filtering. Cartesian periodic or open boundaries would connect the wrong neighboring cells."))
    damp = _one(document, "dynamics", "damp_opt", 0)
    if damp in (1, 2):
        operation = "diffusive upper damping" if damp == 1 else "Rayleigh relaxation to the base state"
        refusals.append(dict(code="upper_damping", breakage=f"damp_opt={damp} requests {operation}. This engine admits only 0 or 3, so replacing it would remove or change the requested damping tendency.", engine_evidence="woof/config.py:4004"))
    files = []
    for name in row["required_files"]:
        target = directory / name
        files.append(dict(name=name, path=str(target), exists=target.is_file()))
        if not target.is_file():
            refusals.append(dict(code="missing_input", breakage=f"{name} is absent from {directory}; the required {row['sounding']['format']} input cannot be initialized."))
    return dict(schema="gpuwm-wrf-ideal-plan-v1", source_version=catalog["wrf_version"],
                source_commit=catalog["wrf_commit"], path=str(path),
                namelist_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                case=row["name"], recipe=row, root_grid=grid, max_domains=n,
                boundaries=topology, files=files, namelist=document,
                metadata_recognized=True, imports=not refusals, runs=not refusals,
                initialization_identity="native-capture bitwise module replay" if not refusals else "not applicable", forecast_comparison="see docs/public/PHYSICS.md",
                runtime=row.get("runtime_binding"), command=f"woof go {directory}",
                status="runnable" if not refusals else "refused", refusals=refusals,
                scope="Initialization planning only. This plan is not a complete physics or namelist option admission audit.")


def _arguments(parser):
    parser.add_argument("namelist", type=Path, nargs="?", help="stock ideal namelist.input")
    parser.add_argument("--case", choices=[r["name"] for r in case_catalog()["cases"]],
                        help="WRF build-time test directory, required when no selector or directory identifies it")
    parser.add_argument("--input-directory", type=Path)
    parser.add_argument("--list", action="store_true", help="list source recipes and their missing runtime bindings")
    parser.add_argument("--json", action="store_true", help="emit the complete structured inspection receipt")


def _run(args):
    if args.list:
        if args.namelist is not None:
            raise ValueError("choose a recipe listing or a namelist inspection so the inspected artifact is unambiguous")
        rows = case_catalog()["cases"]
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for row in rows:
                print(f"{row['name']}: {row.get('runtime_binding','initialization binding missing')}; {row['missing_binding']}")
        return 0
    if args.namelist is None:
        raise ValueError("ideal-plan needs namelist.input or --list to identify the initialization being inspected")
    result = inspect_namelist(args.namelist, case=args.case, input_directory=args.input_directory)
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print(f"{result['case']}: {result['status']}; inspection only, no state initialized")
        for refusal in result["refusals"]:
            print(f"  {refusal['code']}: {refusal['breakage']}")
    # Exit zero means a runtime binding passed admission. This inspection
    # itself never allocates state or runs the forecast.
    return 0 if result["runs"] else 2


def register_cli(subparsers):
    parser = subparsers.add_parser("ideal-plan", help="inspect WRF ideal initialization recipes and named missing bindings (CPU only)")
    _arguments(parser)
    parser.set_defaults(func=_run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    _arguments(parser)
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (ValueError, OSError) as error:
        print(f"ideal-plan: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
