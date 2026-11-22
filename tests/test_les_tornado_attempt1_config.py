"""The tornado-LES attempt #1 config must keep matching what was ruled.

The ratifications are prose in
``docs/internal/specs/P6-LES-DECISIONS-RATIFIED-2026-08-05.md``, the
geometry is numbers in the TOML, and nothing else connects them. This
file is the connection: it runs the case module's own audit, and adds
the mutation controls that prove the audit can fail.

Two cases are covered. ``les_tornado_mayfield_20211210`` is the live one
that attempt #1 runs; ``les_tornado_dodgecity_20160524`` is PARKED behind
an unadjudicated ingest question and is held to the same structural
standard so a future attempt starts from a config that still means what
it meant when it was ratified.

Everything here is CPU-only.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from woof.verify.cases import les_tornado_dodgecity_20160524 as parked
from woof.verify.cases import les_tornado_mayfield_20211210 as case

ROOT = Path(__file__).resolve().parents[1]


def test_the_shipped_config_matches_every_ratified_pin() -> None:
    assert case.audit() == []


def test_the_parked_case_still_matches_its_own_ruling() -> None:
    """Parked is not abandoned. A config nobody can run today is still a
    config somebody will pick up, and it should not have rotted by then."""

    assert parked.audit() == []


@pytest.mark.parametrize("module", (case, parked),
                         ids=("mayfield", "dodgecity"))
def test_the_case_agrees_with_its_own_ratification(module) -> None:
    """A case may not ratify a pairing the model refuses to build.

    Both modules name
    ``docs/internal/specs/P6-LES-DECISIONS-RATIFIED-2026-08-05.md`` as
    their ratification, and its G4 rules ``inflow_perturbation`` ON at
    the first LES domain under a PARAMETERIZED-turbulence parent and OFF
    below it: the generator's vertical extent is the parent's diagnosed
    PBLH, so a ``bl_pbl_physics = 0`` parent has nothing to define it
    with and ``build_inflow_perturbation`` refuses.

    Held against the case's own ratified CHAIN rather than against the
    prose, because CHAIN is the thing ``audit()`` enforces on the TOML.
    Dodge City was PARKED (d531fb521) before G4's sweep and shipped
    ``INFLOW_DOMAINS = (3, 4)`` -- so its gate returned ``[]`` for a
    configuration the model kills twelve seconds after admission, and
    would have gone RED the moment the config was corrected.  That is
    the 2026-08-25 guard-retirement law's own failure mode: G4 swept
    mayfield and left the parked twin ratifying the defect.
    """

    assert module.RATIFICATION == ("docs/internal/specs/"
                                   "P6-LES-DECISIONS-RATIFIED-2026-08-05.md")
    # The ratified chain is linear (d(n) nests in d(n-1)) and listed
    # parent-before-child, so the preceding row IS the parent.  Asserted,
    # not assumed: a future chain that branches must not read as passing.
    ids = [row[0] for row in module.CHAIN]
    assert ids == list(range(1, len(ids) + 1))
    pbl_of = {row[0]: row[-1] for row in module.CHAIN}
    seeded_under_les = [
        grid_id for grid_id in module.INFLOW_DOMAINS
        if grid_id > 1 and pbl_of[grid_id - 1] == 0]
    assert not seeded_under_les, (
        f"{module.MODULE} ratifies inflow seeding on "
        f"d{seeded_under_les[0]:02d}, whose ratified parent "
        f"d{seeded_under_les[0] - 1:02d} runs bl_pbl_physics = 0; G4 rules "
        "that OFF and the model refuses to build it")


def test_the_case_module_runs_as_a_script() -> None:
    """It registers as a ``script`` capability, so it must be one."""

    proc = subprocess.run(
        [sys.executable, "-m",
         "woof.verify.cases.les_tornado_mayfield_20211210"],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "matches every ratified pin" in proc.stdout


def test_the_case_is_discoverable_without_importing_it() -> None:
    from woof.verify.cases import manifest

    row = next(entry for entry in manifest()
               if entry["name"] == "les_tornado_mayfield_20211210")
    assert row["capabilities"] == ["script"], (
        "the case declares no GATES and no run(), so it must register as a "
        "script and must not offer itself to `woof verify`, which would "
        "promise gates that do not exist until a run does")


def test_the_chain_uses_integer_time_step_ratios_and_the_dt_law() -> None:
    """``parent_time_step_ratio`` is an integer, which is why 250 m -> 100 m
    is not on the ladder and 500 m -> 100 m is. dt follows 5 * dx(km)."""

    from woof.experiment import load_experiment

    exp = load_experiment(str(case.CONFIG))
    for domain in exp.domains:
        run = domain.run
        assert run.dt == pytest.approx(5.0 * run.dx / 1000.0), (
            f"d{domain.grid_id:02d} dt={run.dt} breaks dt = 5*dx(km)")
        assert int(domain.parent_time_step_ratio) == \
            domain.parent_time_step_ratio
        if domain.parent_id:
            parent = exp.domains[domain.parent_id - 1]
            assert parent.run.dt / run.dt == pytest.approx(
                domain.parent_time_step_ratio)
            assert parent.run.dx / run.dx == pytest.approx(
                domain.parent_grid_ratio)
            assert run.nx % domain.parent_grid_ratio == 0, (
                f"d{domain.grid_id:02d} nx={run.nx} is not a whole number of "
                f"parent cells (WPS e_we = n*ratio + 1)")


def test_every_nest_is_concentric_on_the_ruled_box_centre() -> None:
    """The 100 m box is placed by centring, not by a hand-typed offset, so
    the placement is checkable arithmetic rather than a claim."""

    from woof.experiment import load_experiment

    exp = load_experiment(str(case.CONFIG))
    for domain in exp.domains[1:]:
        parent = exp.domains[domain.parent_id - 1]
        span = domain.run.nx // domain.parent_grid_ratio
        assert domain.i_parent_start == (parent.run.nx - span) // 2 + 1
        span_j = domain.run.ny // domain.parent_grid_ratio
        assert domain.j_parent_start == (parent.run.ny - span_j) // 2 + 1


def test_the_placement_contract_makes_this_a_transit_box() -> None:
    """The whole placement argument, checked rather than asserted.

    A 20 km box on a 250 km track is only defensible if it holds the
    tornadic transit and lets the parents own genesis. That is a claim
    about which landmark lands in which domain, so it is checked as one.
    """

    from woof.experiment import load_experiment

    exp = load_experiment(str(case.CONFIG))
    grids = case.domain_grids(exp)

    for label, lat, lon, inside, outside in case.PLACEMENT:
        for gid in inside:
            assert case.contains(grids, gid, lat, lon), (
                f"{label} must be inside d{gid:02d}")
        for gid in outside:
            assert not case.contains(grids, gid, lat, lon), (
                f"{label} must be outside d{gid:02d}")

    # The two halves of "transit box", stated as the assertions they are.
    assert case.contains(grids, 4, case.MAYFIELD_LAT, case.MAYFIELD_LON)
    assert not case.contains(grids, 3, case.INITIATION_LAT,
                             case.INITIATION_LON), (
        "convective initiation must fall OUTSIDE the LES domains: the "
        "parents resolve genesis and the children resolve the transit")


def test_domain_grids_honours_offcentre_placement() -> None:
    """``domain_grids`` must place children by the real nest registration.

    Its previous form re-derived every domain centred on the projection
    reference and never read ``i_parent_start`` -- harmless for attempt
    #1's genuinely concentric family, wrong for any off-centre placement
    (recorded in docs/les/ATTEMPT2-EXPECTATIONS.md section 6, deferred
    that night to avoid perturbing attempt #1's passing audit).  The
    attempt-2 config in this same tree moves d03/d04 off-centre, and its
    committed placement receipt is the known answer: the fixed
    instrument must reproduce it, and under the old code it cannot
    (the old d04 stayed on the reference and contained Mayfield).
    """

    from woof.experiment import load_experiment

    exp = load_experiment(str(
        case.CONFIG.parent
        / "les_tornado_100m_mayfield_20211210_attempt2.toml"))
    grids = case.domain_grids(exp)
    # docs/les/ATTEMPT2-EXPECTATIONS.md section 2, cross-checked against
    # ProjectedGrid.nest by the placement receipt.
    receipt = {3: (36.4999, -89.1545), 4: (36.4497, -89.1485)}
    for gid, (want_lat, want_lon) in receipt.items():
        grid, nx, ny = grids[gid]
        lat, lon = grid.ij_to_latlon((nx + 1) / 2.0, (ny + 1) / 2.0)
        assert float(lat) == pytest.approx(want_lat, abs=5e-4)
        assert float(lon) == pytest.approx(want_lon, abs=5e-4)
    # The behavioural consequences the receipt records: Cayce enters
    # d04, Mayfield leaves it but stays in d03.
    assert case.contains(grids, 4, case.CAYCE_LAT, case.CAYCE_LON)
    assert not case.contains(grids, 4, case.MAYFIELD_LAT,
                             case.MAYFIELD_LON)
    assert case.contains(grids, 3, case.MAYFIELD_LAT, case.MAYFIELD_LON)


def test_the_d04_box_is_twenty_km_and_offset_upstream_of_mayfield() -> None:
    """Centred southwest of the town so the storm enters mature."""

    from woof.experiment import load_experiment

    exp = load_experiment(str(case.CONFIG))
    d04 = exp.domains[3].run
    assert d04.nx * d04.dx == pytest.approx(20000.0)
    assert d04.ny * d04.dx == pytest.approx(20000.0)

    # Southwest: the centre is south of and west of Mayfield, and the
    # offset is a few km rather than most of the box.
    assert case.BOX_CENTER_LAT < case.MAYFIELD_LAT
    assert case.BOX_CENTER_LON < case.MAYFIELD_LON
    dlat_km = (case.MAYFIELD_LAT - case.BOX_CENTER_LAT) * 111.32
    dlon_km = ((case.MAYFIELD_LON - case.BOX_CENTER_LON) * 111.32
               * math.cos(math.radians(case.BOX_CENTER_LAT)))
    offset_km = math.hypot(dlat_km, dlon_km)
    assert 2.0 < offset_km < 8.0, (
        f"the offset is {offset_km:.1f} km; too small and the storm does "
        "not enter mature, too large and Mayfield leaves the box")


def test_the_window_covers_the_tornadic_transit() -> None:
    from woof.experiment import load_experiment
    from datetime import timedelta

    exp = load_experiment(str(case.CONFIG))
    end = exp.start_time + timedelta(seconds=exp.run_seconds)
    assert exp.start_time < case.MAYFIELD_STRUCK_UTC < end
    assert exp.start_time <= case.INITIATION_UTC < end, (
        "initiation must be inside the window even though it is outside "
        "the LES domains -- d01/d02 have to produce the storm")
    # Enough lead for the LES children to spin up before the storm arrives.
    lead = (case.MAYFIELD_STRUCK_UTC - exp.start_time).total_seconds()
    assert lead >= 3.0 * 3600.0, (
        f"only {lead / 3600:.1f} h of spin-up before the transit")


def test_the_ratified_vertical_grid_beats_the_shipped_ladder() -> None:
    """G2's whole point: 18 half levels below 1.7 km was the binding
    constraint on the demonstration, and the ladder must actually fix it."""

    sys.path.insert(0, str(ROOT / "tools"))
    try:
        from build_stretched_eta_ladder import score_ladder
    finally:
        sys.path.pop(0)
    from woof.experiment import load_experiment
    from woof.native_wrf_contract import CERTIFIED_ETA_LEVELS

    exp = load_experiment(str(case.CONFIG))
    vertical = exp.vertical
    scored = score_ladder(list(vertical.eta_levels), p_top=vertical.p_top,
                          hybrid_opt=2, etac=0.2, bl_top=1700.0)
    shipped = score_ladder(list(CERTIFIED_ETA_LEVELS), p_top=10000.0,
                           hybrid_opt=2, etac=0.2, bl_top=1700.0)

    # The control that makes the instrument an instrument: it reproduces the
    # published count for the shipped ladder (docs/public/LES.md:265-266).
    assert shipped["levels_below_bl_top"] == 18
    assert shipped["first_half_level_m"] == pytest.approx(8.40, abs=0.01)

    assert scored["levels_below_bl_top"] >= 30
    assert scored["levels_below_bl_top"] > shipped["levels_below_bl_top"]
    assert scored["mean_dz_below_bl_top_m"] < \
        shipped["mean_dz_below_bl_top_m"]


@pytest.mark.parametrize("mutation", [
    {"restart_interval_s": 0.0},
    {"run_seconds": 3600.0},
])
def test_the_audit_can_fail(mutation: dict, tmp_path: Path) -> None:
    """A check that cannot fail is not a check.

    ``restart_interval_s = 0.0`` is the demonstration's own named defect,
    so it is the mutation that matters most: the audit exists partly to
    stop it coming back.
    """

    import tomllib

    with case.CONFIG.open("rb") as handle:
        raw = case.CONFIG.read_text(encoding="utf-8")
        tomllib.load(handle)
    for key, value in mutation.items():
        import re
        raw = re.sub(rf"(?m)^{key} = .*$", f"{key} = {value}", raw)
    mutated = tmp_path / case.CONFIG.name
    mutated.write_text(raw, encoding="utf-8")

    # Hand the mutated copy to `audit` rather than patching the module
    # global.  `audit(config=...)` is the seam the case module offers, and
    # patching `case.CONFIG` stopped reaching the audit when the config
    # resolution became `_repo_config.locate(CONFIG_NAME) or CONFIG`: the
    # locate wins whenever the real config is on disk, which is every run
    # from a checkout, so the audit read the SHIPPED config, found it
    # correct, and returned [] no matter what this test wrote.  A control
    # that cannot fail is exactly what this test exists to forbid, so it
    # must not be one itself.
    assert case.audit(config=mutated) != [], f"{mutation} was not caught"


# --- the hand-built route input set -------------------------------------
#
# woof.hrrr_route_inputs.render_namelist_input admits bl_pbl_physics in
# [1, 11] only, so no product path can emit namelists for a tree with a
# PBL-off LES child, and this tree has two.  The set is therefore
# generated by tools/emit_les_route_inputs.py, and the thing that makes
# that safe rather than merely necessary is that it round-trips.


def _emitter():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import emit_les_route_inputs
    finally:
        sys.path.pop(0)
    return emit_les_route_inputs


def test_the_emitted_route_set_imports_back_to_its_own_config(tmp_path):
    """The check the hand-edited route sets never had.

    A namelist that does not import back to the config it was generated
    from describes a different tree than the one that will run, and the
    hierarchy route would prepare it happily.
    """

    emitter = _emitter()
    from woof.experiment import load_experiment

    exp = load_experiment(str(case.CONFIG))
    assert emitter.main([
        "--config", str(case.CONFIG), "--outdir", str(tmp_path),
        "--stem", "route", "--verify"]) == 0

    for name in ("route.namelist.wps", "route.namelist.input",
                 "route.stock.namelist.input", "route.d01-target.json"):
        assert (tmp_path / name).is_file(), name

    assert emitter.verify(exp, tmp_path / "route.namelist.wps",
                          tmp_path / "route.namelist.input") == []


def test_the_emitted_set_passes_the_hierarchy_routes_own_gate(tmp_path):
    """The check whose absence cost a stage-3 failure after a 9.6 GB fetch.

    The round trip above proves the pair describes the right TREE. It
    says nothing about whether the hierarchy route will ACCEPT the pair,
    because that route applies a separate raw contract: certified pins it
    requires present and exact, keys it requires absent from the native
    file, and an exhaustive native-to-stock delta set. A set that
    round-tripped perfectly was refused by it for two keys in the wrong
    section and one never emitted at all.

    So this calls the route's own validators rather than a
    re-implementation, and it is the reason `--verify` runs them first.
    """

    emitter = _emitter()
    assert emitter.main([
        "--config", str(case.CONFIG), "--outdir", str(tmp_path),
        "--stem", "route"]) == 0
    assert emitter.verify_hierarchy_gate(
        tmp_path / "route.namelist.wps",
        tmp_path / "route.namelist.input",
        tmp_path / "route.stock.namelist.input") == []


def test_the_emitted_pair_differs_by_exactly_the_three_stock_deltas(tmp_path):
    """ra_lw_physics 0->1 and the two stock-only &physics keys
    ghg_input=0 and do_radar_ref=1.  use_theta_m is 0 on both halves:
    the stock arm runs the direct export, which holds dry theta.

    The hierarchy route verifies the pair key for key, so an extra delta
    is a refusal after the expensive preparation rather than here. Both
    stock-only keys must land in &physics: the validator reads them at
    ``stock["physics"][...]`` and `ghg_input` was originally emitted into
    &dynamics, where it parsed fine and was invisible to the gate.
    """

    emitter = _emitter()
    emitter.main(["--config", str(case.CONFIG), "--outdir", str(tmp_path),
                  "--stem", "route"])
    native = (tmp_path / "route.namelist.input").read_text().splitlines()
    stock = (tmp_path / "route.stock.namelist.input").read_text().splitlines()

    only_native = [ln.strip() for ln in native
                   if ln not in stock and not ln.strip().startswith("!")]
    only_stock = [ln.strip() for ln in stock
                  if ln not in native and not ln.strip().startswith("!")]

    assert [ln.split("=")[0].strip() for ln in only_native] == [
        "ra_lw_physics"]
    assert [ln.split("=")[0].strip() for ln in only_stock] == [
        "ra_lw_physics", "ghg_input", "do_radar_ref"]

    # Section placement, not just presence.
    from woof.namelist_import import parse_namelist

    parsed_stock = parse_namelist(tmp_path / "route.stock.namelist.input")
    parsed_native = parse_namelist(tmp_path / "route.namelist.input")
    assert parsed_stock["dynamics"]["use_theta_m"] == [0]
    assert parsed_native["dynamics"]["use_theta_m"] == [0]
    assert parsed_stock["physics"]["ghg_input"] == [0]
    assert parsed_stock["physics"]["do_radar_ref"] == [1]
    assert "ghg_input" not in parsed_native["physics"]
    assert "do_radar_ref" not in parsed_native["physics"]
    # And the native file must stay silent on nwp_diagnostics too.
    assert "nwp_diagnostics" not in parsed_native["time_control"]


def test_the_emitter_carries_every_pbl_off_domain(tmp_path):
    """The demonstration's generator handled exactly one; this tree has two,
    and a set that quietly dropped one would prepare a tree with a PBL the
    run does not have."""

    emitter = _emitter()
    emitter.main(["--config", str(case.CONFIG), "--outdir", str(tmp_path),
                  "--stem", "route"])
    text = (tmp_path / "route.namelist.input").read_text()
    line = next(ln for ln in text.splitlines()
                if ln.strip().startswith("bl_pbl_physics"))
    assert line.split("=")[1].strip().rstrip(",").replace(" ", "") == \
        "1,1,0,0"
    km = next(ln for ln in text.splitlines()
              if ln.strip().startswith("km_opt"))
    assert km.split("=")[1].strip().rstrip(",").replace(" ", "") == "4,4,3,3"
