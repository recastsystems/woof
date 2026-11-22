"""Emit the HRRR native-route input set for a tree the product path refuses.

``woof.hrrr_route_inputs.render_namelist_input`` admits only
``bl_pbl_physics`` in ``[1, 11]`` -- each value a registered HRRR
preparation profile pins -- so no product path can emit namelists for a
tree carrying a PBL-off LES child.  The shipped 250 m demonstration hit
that wall with one such domain and solved it by emitting a wizard set and
extending the per-domain columns by hand
(``docs/internal/receipts/les/make_d03.py``,
``INFLOW-FETCH-D90-2026-08-03.md:68-75``).

This is that idiom generalized: it derives the whole set from the run
authority -- the experiment TOML -- instead of patching columns onto a
neighbouring tree's namelists, so the four artefacts cannot describe a
different tree than the one that will actually run.  It handles any
number of PBL-off domains rather than exactly one.

Emits, using the names ``woof.hrrr_route_inputs.route_input_paths``
fixes (``hrrr_route_inputs.py:719-732``):

    <stem>.namelist.wps            WPS geogrid geometry
    <stem>.d01-target.json         the root target-domain document
    <stem>.namelist.input          the native woof half
    <stem>.stock.namelist.input    the stock-WRF half

The native/stock pair differs by exactly THREE things, which the hierarchy
route verifies key for key before it prepares anything: ``ra_lw_physics``
0 -> 1 (per domain, wherever the native entry is 0), stock-only
``&physics/ghg_input = 0``, and stock-only ``&physics/do_radar_ref = 1``.
The last two are MANDATORY on the stock arm and FORBIDDEN on the native
one.  ``use_theta_m`` is 0 on both halves: the stock-WRF export holds dry
theta (``woof.wrf_physics_inventory.EXPORT_USE_THETA_M``).

``--verify`` runs TWO checks, and the order matters:

1. the hierarchy route's own raw-namelist gate
   (``_require_raw_stock_delta`` + ``_require_raw_wps_contract``), called
   directly rather than re-implemented, because that is the check that
   actually stops the route;
2. a round trip back through ``woof.namelist_import``, compared against
   the TOML the set was generated from, domain by domain.

They answer different questions and passing one says nothing about the
other.  MEASURED 2026-08-06: a set that round-tripped perfectly was
refused by the hierarchy gate after a 9.6 GB fetch and a root
preparation, for two keys in the wrong section and one never emitted at
all.  Check 1 exists because of that.

Generic: no case, campaign or configuration name appears here.

Usage
-----
    python tools/emit_les_route_inputs.py --config configs/<x>.toml \\
        --outdir <dir> --stem route --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from woof.experiment import load_experiment                # noqa: E402
from woof.wrf_physics_inventory import EXPORT_USE_THETA_M  # noqa: E402

#: HRRR native forcing: 50 hybrid levels + the surface record, and the nine
#: RUC soil depths.  Both are properties of the SOURCE, not of the tree, and
#: the shipped route set carries the same two numbers.
NUM_METGRID_LEVELS = 51
NUM_METGRID_SOIL_LEVELS = 9
#: MODIS, which is what woof's static build produces and the importer pins.
NUM_LAND_CAT = 21
#: The wizard's cumulus interval for a nest with no cumulus scheme.  Dead
#: namelist state either way (cu_physics = 0 everywhere on an LES tree); the
#: shipped set spells it this way and the importer normalises it.
NEST_CUDT_MINUTES = 5.0


def _col(values, fmt="{}") -> str:
    return ", ".join(fmt.format(v) for v in values) + ","


def _logical(values) -> str:
    return ", ".join(".true." if v else ".false." for v in values) + ","


def _stock_only_physics(stock: bool) -> str:
    """The two stock-only ``&physics`` keys, or nothing.

    Both live in ``&physics`` and both are MANDATORY on the stock arm and
    FORBIDDEN on the native one -- ``woof.hrrr_hierarchy_direct
    ._require_raw_stock_delta`` reads them at
    ``stock["physics"][...]`` and refuses the pair if the native file
    carries either.

    * ``ghg_input = 0`` pins the substituted RRTM's gas table. WRF's
      default 1 reads the time-varying CAM table, which is not a valid
      implicit substitute for the fixed-gas configuration the acceptance
      gate uses.
    * ``do_radar_ref = 1`` is answered in CODE on the native arm -- woof
      evaluates REFL_10CM at output time unconditionally -- while WRF
      allocates the array only under ``package radar_refl
      compute_radar_ref==1`` (Registry.EM_COMMON:3059, fed via
      module_check_a_mundo.F:3477). Left at the Registry default the
      stock arm writes history frames with no REFL_10CM at all and every
      reflectivity score on that arm is unanswerable. A SCALAR:
      Registry.EM_COMMON:2447 declares it nentries 1.
    """

    if not stock:
        return ""
    return (" ghg_input                           = 0,\n"
            " do_radar_ref                        = 1,\n")


def _eta_block(eta, per_line: int = 5, pad: int = 14) -> str:
    cells = [f"{v:.5f}" for v in eta]
    lines = []
    for start in range(0, len(cells), per_line):
        chunk = ", ".join(cells[start:start + per_line])
        prefix = " eta_levels = " if start == 0 else " " * pad
        lines.append(prefix + chunk + ",")
    return "\n".join(lines)


def build(exp, *, stock: bool) -> str:
    """One half of the namelist pair, from the experiment authority."""

    domains = exp.domains
    n = len(domains)
    runs = [d.run for d in domains]
    root = runs[0]
    start = exp.start_time
    end = start + timedelta(seconds=exp.run_seconds)
    hours, remainder = divmod(int(exp.run_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    days, hours = divmod(hours, 24)

    def each(attr, fmt="{}"):
        return _col([getattr(r, attr) for r in runs], fmt)

    banner = ("stock-WRF" if stock else "native woof")
    head = (
        f"! Generated by `tools/emit_les_route_inputs.py` from "
        f"{exp.name}.  This is the {banner} half of the pair;\n"
        "! the two differ only by ra_lw_physics 0->1, "
        "stock-only ghg_input=0, which the HRRR hierarchy\n"
        "! route verifies key for key before it prepares anything.\n"
        "! The per-domain turbulence columns are ACCURATE: this tree carries "
        f"{sum(1 for r in runs if r.bl_pbl_physics == 0)} PBL-off LES "
        "domain(s), which\n"
        "! is why no product path can emit this set and this generator "
        "exists.  The TOML is the run authority.\n")

    time_control = f"""&time_control
 run_days                            = {days},
 run_hours                           = {hours},
 run_minutes                         = {minutes},
 run_seconds                         = {seconds},
 start_year                         = {_col([start.year] * n)}
 start_month                        = {_col([f'{start.month:02d}'] * n)}
 start_day                          = {_col([f'{start.day:02d}'] * n)}
 start_hour                         = {_col([f'{start.hour:02d}'] * n)}
 start_minute                       = {_col([f'{start.minute:02d}'] * n)}
 start_second                       = {_col([f'{start.second:02d}'] * n)}
 end_year                         = {_col([end.year] * n)}
 end_month                        = {_col([f'{end.month:02d}'] * n)}
 end_day                          = {_col([f'{end.day:02d}'] * n)}
 end_hour                         = {_col([f'{end.hour:02d}'] * n)}
 end_minute                       = {_col([f'{end.minute:02d}'] * n)}
 end_second                       = {_col([f'{end.second:02d}'] * n)}
 interval_seconds                    = 3600,
 input_from_file                     = {_logical([True] * n)}
 history_interval_s                  = {_col([int(d.history_interval_s) for d in domains])}
 frames_per_outfile                  = {_col([1] * n)}
 restart                             = .false.,
 restart_interval                    = {int(exp.restart_interval_s // 60)},
 io_form_history                     = 2,
 io_form_restart                     = 2,
 io_form_input                       = 2,
 io_form_boundary                    = 2,
/
"""

    domains_block = f"""
&domains
 time_step                           = {int(domains[0].time_step)},
 max_dom                             = {n},
 e_we                                = {_col([r.nx + 1 for r in runs])}
 e_sn                                = {_col([r.ny + 1 for r in runs])}
 e_vert                              = {_col([root.nz + 1] * n)}
{_eta_block(exp.vertical.eta_levels)}
 p_top_requested                     = {exp.vertical.p_top:.1f},
 dx                                  = {_col([r.dx for r in runs], '{:.1f}')}
 dy                                  = {_col([r.dy for r in runs], '{:.1f}')}
 grid_id                             = {_col([d.grid_id for d in domains])}
 parent_id                           = {_col([d.parent_id for d in domains])}
 i_parent_start                      = {_col([d.i_parent_start for d in domains])}
 j_parent_start                      = {_col([d.j_parent_start for d in domains])}
 parent_grid_ratio                   = {_col([d.parent_grid_ratio for d in domains])}
 parent_time_step_ratio              = {_col([d.parent_time_step_ratio for d in domains])}
 feedback                            = {exp.feedback},
 smooth_option                       = {exp.smooth_option},
 hypsometric_opt                     = {root.hypsometric_opt},
 num_metgrid_levels                  = {NUM_METGRID_LEVELS},
 num_metgrid_soil_levels             = {NUM_METGRID_SOIL_LEVELS},
 sfcp_to_sfcp                        = .true.,
/
"""

    cudt = [runs[0].cudt_minutes] + [NEST_CUDT_MINUTES] * (n - 1)
    physics = f"""
&physics
 mp_physics                          = {each('mp_physics')}
 ra_lw_physics                       = {_col([1 if stock else r.ra_lw_physics for r in runs])}
 ra_sw_physics                       = {each('ra_sw_physics')}
 radt                                = {each('radt', '{:.1f}')}
 icloud                              = 1,
 swrad_scat                          = 1.0,
 sf_sfclay_physics                   = {each('sf_sfclay_physics')}
 sf_surface_physics                  = {each('sf_surface_physics')}
 bl_pbl_physics                      = {each('bl_pbl_physics')}
 bldt                                = {each('bldt', '{:.1f}')}
 cu_physics                          = {each('cu_physics')}
 cudt                                = {_col(cudt, '{:.1f}')}
 isfflx                              = {runs[0].isfflx},
 ifsnow                              = 1,
 surface_input_source                = 1,
 num_soil_layers                     = {root.num_soil_layers},
 num_land_cat                        = {NUM_LAND_CAT},
 sf_urban_physics                    = {_col([0] * n)}
 sst_update                          = 0,
{_stock_only_physics(stock)}/

&fdda
/
"""

    dynamics = f"""
&dynamics
 use_theta_m                         = {EXPORT_USE_THETA_M if stock else 0},
 hybrid_opt                          = {exp.vertical.hybrid_opt},
 etac                                = {exp.vertical.etac},
 top_lid                             = {_logical([r.top_lid for r in runs])}
 w_damping                           = {root.w_damping},
 epssm                               = {each('epssm', '{:.2f}')}
 diff_opt                            = {_col([2] * n)}
 km_opt                              = {each('km_opt')}
 mix_full_fields                     = {_logical([True] * n)}
 diff_6th_opt                        = {each('diff_6th_opt')}
 diff_6th_factor                     = {each('diff_6th_factor')}
 diff_6th_slopeopt                   = {each('diff_6th_slopeopt')}
 c_s                                 = {each('c_s', '{:.2f}')}
 mix_isotropic                       = {each('mix_isotropic')}
 mix_upper_bound                     = {each('mix_upper_bound')}
 diff_6th_thresh                     = {_col([0.1] * n)}
 base_temp                           = {root.base_temp:.1f},
 damp_opt                            = {root.damp_opt},
 zdamp                               = {each('zdamp', '{:.1f}')}
 dampcoef                            = {each('dampcoef')}
 khdif                               = {each('khdif', '{:.1f}')}
 kvdif                               = {each('kvdif', '{:.1f}')}
 non_hydrostatic                     = {_logical([True] * n)}
 moist_adv_opt                       = {each('moist_adv_opt')}
 scalar_adv_opt                      = {_col([1] * n)}
 time_step_sound                     = {each('time_step_sound')}
 smdiv                               = {each('smdiv')}
 emdiv                               = {each('emdiv')}
 h_sca_adv_order                     = {each('h_sca_adv_order')}
/
"""

    bdy = f"""
&bdy_control
 spec_bdy_width                      = {exp.spec_bdy_width},
 spec_zone                           = {root.spec_zone},
 relax_zone                          = {root.relax_zone},
 spec_exp                            = 0.0,
 specified                           = {_logical([d.grid_id == 1 for d in domains])}
 nested                              = {_logical([d.grid_id != 1 for d in domains])}
/

&grib2
/

&namelist_quilt
 nio_tasks_per_group                 = 0,
 nio_groups                          = 1,
/
"""
    return head + time_control + domains_block + physics + dynamics + bdy


def build_wps(exp) -> str:
    domains = exp.domains
    runs = [d.run for d in domains]
    proj = exp.projection
    return f"""&share
 wrf_core = 'ARW',
 max_dom = {len(domains)},
 interval_seconds = 3600,
 io_form_geogrid = 2,
/

&geogrid
 parent_id         = {_col([max(d.parent_id, 1) for d in domains])}
 parent_grid_ratio = {_col([d.parent_grid_ratio for d in domains])}
 i_parent_start    = {_col([d.i_parent_start for d in domains])}
 j_parent_start    = {_col([d.j_parent_start for d in domains])}
 e_we              = {_col([r.nx + 1 for r in runs])}
 e_sn              = {_col([r.ny + 1 for r in runs])}
 geog_data_res     = {_col(["'default'"] * len(domains))}
 dx = {int(runs[0].dx)},
 dy = {int(runs[0].dy)},
 map_proj = '{proj.map_proj}',
 ref_lat   = {proj.ref_lat},
 ref_lon   = {proj.ref_lon},
 truelat1  = {proj.truelat1},
 truelat2  = {proj.truelat2},
 stand_lon = {proj.stand_lon},
/
"""


def build_target(exp, name: str) -> dict:
    root = exp.domains[0]
    proj = exp.projection
    return {
        "dx_m": root.run.dx, "dy_m": root.run.dy,
        "map_proj": proj.map_proj, "name": name,
        "nx": root.run.nx, "ny": root.run.ny, "nz": root.run.nz,
        "ref_lat": proj.ref_lat, "ref_lon": proj.ref_lon,
        "relax_zone": root.run.relax_zone,
        "schema": "gpuwm-hrrr-target-domain-v1",
        "spec_bdy_width": exp.spec_bdy_width,
        "spec_zone": root.run.spec_zone,
        "stand_lon": proj.stand_lon,
        "time_step_seconds": int(root.time_step),
        "truelat1": proj.truelat1, "truelat2": proj.truelat2,
    }


#: What a round trip must preserve.  Geometry, the vertical grid, and every
#: per-domain switch the LES row moves -- the columns a hand-edit gets wrong.
_VERIFIED = ("nx", "ny", "nz", "dx", "dy", "dt", "km_opt", "bl_pbl_physics",
             "c_s", "mix_isotropic", "mix_upper_bound", "isfflx",
             "mp_physics", "sf_sfclay_physics", "sf_surface_physics",
             "cu_physics", "epssm", "diff_6th_opt", "diff_6th_factor",
             "ra_lw_physics", "ra_sw_physics", "top_lid", "w_damping",
             "damp_opt", "zdamp", "dampcoef", "khdif", "kvdif",
             "moist_adv_opt", "time_step_sound", "smdiv", "emdiv",
             "hypsometric_opt", "h_sca_adv_order", "spec_zone", "relax_zone")


def verify_hierarchy_gate(wps_path: Path, input_path: Path,
                          stock_path: Path) -> list[str]:
    """Run the REAL gate: the hierarchy route's own raw-namelist checks.

    This is the check whose absence cost a stage-3 failure after a 9.6 GB
    fetch and a root preparation. The importer round trip below proves
    the pair describes the right TREE; it says nothing about whether the
    hierarchy route will accept the pair, because that route applies a
    separate raw contract -- certified pins it requires present and
    exact, keys it requires ABSENT from the native file, and an
    exhaustive native-to-stock delta set.

    So the emitter now calls the route's own private validators rather
    than a re-implementation of them. Importing private names is
    deliberate: a copy of the contract here would be a second place for
    it to drift, and the whole point is to fail in the generator instead
    of after the expensive stages.
    """

    from woof.hrrr_hierarchy_direct import (
        _require_raw_stock_delta, _require_raw_wps_contract)

    bad: list[str] = []
    try:
        delta = _require_raw_stock_delta(input_path, stock_path)
    except ValueError as exc:
        return [f"hierarchy stock-delta gate REFUSES: {exc}"]
    try:
        _require_raw_wps_contract(wps_path, delta["max_dom"])
    except ValueError as exc:
        bad.append(f"hierarchy WPS gate REFUSES: {exc}")
    return bad


def verify(exp, wps_path: Path, input_path: Path) -> list[str]:
    """Import the emitted pair and compare it against its own source."""

    import tempfile

    from woof.namelist_import import import_namelists

    # import_namelists returns (TOML text, substitution report); the text is
    # already validated through build_experiment, so loading it back is what
    # turns it into the comparable object.  The source config's governance
    # declarations ([experiment].acknowledgements) have no namelist
    # spelling, so the round trip inherits them from the experiment it is
    # verifying against rather than tripping the load guard the source
    # config already satisfies.
    toml_text, report = import_namelists(
        str(wps_path), str(input_path),
        acknowledgements=tuple(getattr(exp, "acknowledgements", ())))
    with tempfile.TemporaryDirectory() as tmp:
        echo = Path(tmp) / "roundtrip.toml"
        echo.write_text(toml_text, encoding="utf-8", newline="\n")
        reimported = load_experiment(str(echo))

    bad: list[str] = []
    for substitution in getattr(report, "substitutions", ()) or ():
        bad.append(f"the importer SUBSTITUTED something: {substitution}")

    if len(reimported.domains) != len(exp.domains):
        return [f"round trip gives {len(reimported.domains)} domains, "
                f"source has {len(exp.domains)}"]
    if reimported.start_time != exp.start_time:
        bad.append(f"start_time {reimported.start_time} != {exp.start_time}")
    if float(reimported.run_seconds) != float(exp.run_seconds):
        bad.append(f"run_seconds {reimported.run_seconds} "
                   f"!= {exp.run_seconds}")
    if float(reimported.restart_interval_s) != float(exp.restart_interval_s):
        bad.append(f"restart_interval_s {reimported.restart_interval_s} "
                   f"!= {exp.restart_interval_s}")
    source_eta = tuple(round(v, 5) for v in exp.vertical.eta_levels)
    round_eta = tuple(round(v, 5) for v in reimported.vertical.eta_levels)
    if source_eta != round_eta:
        bad.append("eta_levels differ across the round trip")

    for want, got in zip(exp.domains, reimported.domains):
        for field in _VERIFIED:
            a = getattr(want.run, field, None)
            b = getattr(got.run, field, None)
            if isinstance(a, float) and isinstance(b, float):
                if abs(a - b) > 1e-9:
                    bad.append(f"d{want.grid_id:02d}.{field}: {b} != {a}")
            elif a != b:
                bad.append(f"d{want.grid_id:02d}.{field}: {b!r} != {a!r}")
        for field in ("i_parent_start", "j_parent_start",
                      "parent_grid_ratio", "parent_time_step_ratio",
                      "history_interval_s"):
            a, b = getattr(want, field), getattr(got, field)
            if a != b:
                bad.append(f"d{want.grid_id:02d}.{field}: {b!r} != {a!r}")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--stem", default="route")
    ap.add_argument("--verify", action="store_true",
                    help="import the emitted pair back and compare it "
                         "against --config; refuse on any difference")
    args = ap.parse_args(argv)

    exp = load_experiment(str(args.config))
    args.outdir.mkdir(parents=True, exist_ok=True)

    wps_path = args.outdir / f"{args.stem}.namelist.wps"
    input_path = args.outdir / f"{args.stem}.namelist.input"
    stock_path = args.outdir / f"{args.stem}.stock.namelist.input"
    target_path = args.outdir / f"{args.stem}.d01-target.json"

    wps_path.write_text(build_wps(exp), encoding="utf-8", newline="\n")
    input_path.write_text(build(exp, stock=False), encoding="utf-8",
                          newline="\n")
    stock_path.write_text(build(exp, stock=True), encoding="utf-8",
                          newline="\n")
    target_path.write_text(
        json.dumps(build_target(exp, args.stem), indent=2, sort_keys=True)
        + "\n", encoding="utf-8", newline="\n")

    for path in (wps_path, input_path, stock_path, target_path):
        print(f"  wrote {path}")

    pbl_off = [d.grid_id for d in exp.domains if d.run.bl_pbl_physics == 0]
    print(f"  PBL-off domains carried: {pbl_off or 'none'}")

    if args.verify:
        # The hierarchy gate FIRST: it is the one that actually stops the
        # route, and it is cheaper to read than a round-trip diff.
        gate = verify_hierarchy_gate(wps_path, input_path, stock_path)
        if gate:
            print("\nHIERARCHY GATE FAILED -- the route would refuse this "
                  "set after the fetch and the root preparation:")
            for line in gate:
                print(f"  {line}")
            return 1
        print("  hierarchy gate: the route's own raw-namelist contract "
              "accepts the pair")

        bad = verify(exp, wps_path, input_path)
        if bad:
            print("\nROUND TRIP FAILED -- the emitted set does not import "
                  "back to its own source config:")
            for line in bad:
                print(f"  {line}")
            return 1
        print("  round trip: the emitted pair imports back to "
              f"{args.config.name} exactly")
    return 0


__all__ = ["build", "build_wps", "build_target", "verify",
           "verify_hierarchy_gate", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
