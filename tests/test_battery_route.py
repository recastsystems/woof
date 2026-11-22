"""The observation battery's run routes, asked the same way the campaign asks.

Two tools are under test and one committed receipt:

* ``tools/battery_route_preflight.py`` -- every static admission gate on the
  HRRR run route, answered for one experiment config before anything is
  fetched.  Its value is that it calls the route's OWN gate functions, so
  these tests pin the answers rather than a restatement of the rules; the
  day a gate changes, the receipt these tests read changes with it and the
  assertions about WHICH gate answered still hold.
* ``tools/battery_wrf_node_plan.py`` -- the WRF node bundle, generated from
  the same config so the two model arms cannot drift apart in physics.
* ``docs/public/receipts/obsbattery/B4-SPEED-ANCHORS.json`` -- the speed
  anchors every compute projection is arithmetic over.  The committed
  anchor entry is checked against the run report it cites, which is what
  keeps the receipt from becoming folklore.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools import battery_route_preflight as preflight  # noqa: E402
from tools import battery_wrf_node_plan as node_plan  # noqa: E402
from tools.release_exclusions import matches, read_exclusions  # noqa: E402

CONFIG_DIR = REPOSITORY_ROOT / "configs" / "battery"
SIZING_SMOKE = CONFIG_DIR / "shape_smoke_3km_thompson_dudhia.toml"
FAITHFUL_ARM = CONFIG_DIR / "shape_3km_thompson_rrtmg_legacy.toml"
SPEED_ANCHORS = (
    REPOSITORY_ROOT / "docs" / "public" / "receipts" / "obsbattery"
    / "B4-SPEED-ANCHORS.json")

#: The battery's registered domain shape (spec section 1.3).
BATTERY_CELLS = 480 * 400 * 49


def _receipt(config: Path) -> dict:
    return preflight.evaluate(config, repository_root=REPOSITORY_ROOT)


def _gate(receipt: dict, gate_id: str) -> dict:
    for gate in receipt["gates"]:
        if gate["gate"] == gate_id:
            return gate
    raise AssertionError(
        f"{gate_id} is not among {[g['gate'] for g in receipt['gates']]}")


def _fidelity_arm(tmp_path: Path, patches, *, base: Path = FAITHFUL_ARM,
                  name: str = "arm") -> Path:
    """One physics-fidelity arm of a battery config, as an arm is composed.

    The rule is :func:`woof.physics_mode.governed_keys`'s own: strip
    exactly the keys the axis authors under this selection, then add the
    ``[experiment]`` overlay that selects the mode.  The list is QUERIED
    from the axis rather than written down, so this helper keeps
    stripping the right set the day the divergence ledger grows.

    ``patches=None`` is the whole registered patch-set (the P-all arm);
    a list names the subset (the single-entry ablation arms).
    """

    from woof import physics_mode

    governed = physics_mode.governed_keys(
        physics_mode.PHYSICS_MODE_ARWEN_PATCHED, "v1",
        list(patches) if patches is not None else None)
    kept = []
    for line in base.read_text(encoding="utf-8").splitlines(keepends=True):
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in governed:
            continue
        kept.append(line)
    overlay = [
        f'physics_mode = "{physics_mode.PHYSICS_MODE_ARWEN_PATCHED}"',
        'patchset = "v1"',
    ]
    if patches is not None:
        overlay.append(
            "patches = [" + ", ".join(f'"{p}"' for p in patches) + "]")
    text = "".join(kept).replace(
        "[experiment]", "[experiment]\n" + "\n".join(overlay), 1)
    config = tmp_path / f"{name}.toml"
    config.write_text(text, encoding="utf-8")
    return config


#: The smoke config's own declaration line, matched literally so this
#: helper fails loudly the day the shipped config stops carrying it
#: rather than silently writing an arm that declares nothing.
_ACK_BLOCK = re.compile(
    r"^# JUSTIFY .*?^acknowledgements = \[[^]]*]\n",
    re.DOTALL | re.MULTILINE)


def _radiation_fully_off(tmp_path: Path, *, declared: bool,
                         name: str = "radiation_fully_off.toml") -> Path:
    """The smoke config with both radiation streams explicitly disabled.

    The shared land-surface declaration check still applies. Once declared,
    a configuration uses the common runtime checks without a source pair list.
    Tokens are read from their owning guards.
    """

    from woof.physics_compat import (CONSTANT_DOWNWARD_LONGWAVE_ACK,
                                      RADIATION_OFF_LAND_SURFACE_ACK)

    text = SIZING_SMOKE.read_text(encoding="utf-8")
    assert "ra_sw_physics = 1" in text
    assert _ACK_BLOCK.search(text), (
        f"{SIZING_SMOKE.name} no longer carries a justified acknowledgement "
        "block; this helper edits that block and would otherwise write an "
        "arm whose declarations are not the ones it means")
    text = text.replace("ra_sw_physics = 1", "ra_sw_physics = 0")
    if declared:
        replacement = (
            f"# JUSTIFY {CONSTANT_DOWNWARD_LONGWAVE_ACK}: a test arm, never "
            f"a forecast.\n"
            f"# JUSTIFY {RADIATION_OFF_LAND_SURFACE_ACK}: this arm exists to "
            f"exercise the radiation-off runtime selection. Both "
            f"streams off under Noah is two claims and takes two tokens.\n"
            f'acknowledgements = ["{CONSTANT_DOWNWARD_LONGWAVE_ACK}", '
            f'"{RADIATION_OFF_LAND_SURFACE_ACK}"]\n')
    else:
        replacement = ""
    config = tmp_path / name
    config.write_text(_ACK_BLOCK.sub(replacement, text, count=1),
                      encoding="utf-8")
    return config


def test_both_battery_configs_carry_the_registered_shape():
    """Geometry is the one thing both arms must share exactly."""

    for config in (SIZING_SMOKE, FAITHFUL_ARM):
        receipt = _receipt(config)
        assert len(receipt["domains"]) == 1, config.name
        d01 = receipt["domains"][0]
        assert (d01["nx"], d01["ny"], d01["nz"]) == (480, 400, 49), config.name
        assert d01["dx_m"] == 3000.0 and d01["dt_s"] == 15.0, config.name
        assert d01["history_interval_s"] == 3600.0, config.name
        assert receipt["sizing"]["cells"] == BATTERY_CELLS, config.name
        # Convection-permitting: cumulus off, Thompson, YSU, MM5, Noah.
        assert d01["mp_physics"] == 8, config.name
        assert d01["cu_physics"] == 0, config.name
        assert d01["bl_pbl_physics"] == 1, config.name
        assert d01["sf_sfclay_physics"] == 91, config.name
        assert d01["sf_surface_physics"] == 2, config.name


def test_sizing_smoke_is_admitted_by_every_static_gate():
    """The smoke that can run tonight, and the profile it must name."""

    receipt = _receipt(SIZING_SMOKE)
    assert receipt["status"] == "ADMITTED", receipt["refusing_gates"]
    assert receipt["refusing_gates"] == []
    profile = _gate(receipt, "hrrr.root_preparation.profile")
    assert profile["status"] == "ADMITS"
    assert "thompson-mp8-ysu-mm5-noah-dudhia-daytime-v1" in profile["detail"]
    assert _gate(receipt, "hrrr.hierarchy.slice")["status"] == "ADMITS"
    assert _gate(receipt, "hrrr.coverage")["status"] == "ADMITS"
    # One forecast hour needs f00 and f01, which is what makes this smoke
    # affordable to fetch for.
    assert "f00..f01" in _gate(receipt, "hrrr.forcing_inventory")["detail"]


def test_faithful_arm_is_admitted_by_every_static_gate():
    """The whole B4 four-item route motion, landed and visible here.

    History of this pin: it first recorded three refusing gates (the
    route pinned radiation to (0, 1)); items 1-3 of the costed motion
    reduced that to exactly one (no registered profile); the lead's
    2026-08-04 ruling landed item 4 -- the
    thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1 template/profile -- and
    the battery's section 1.3 composition is now ADMITTED end to end,
    with the profile gate naming the profile the root preparer takes.
    """

    receipt = _receipt(FAITHFUL_ARM)
    assert receipt["status"] == "ADMITTED", receipt["refusing_gates"]
    assert receipt["refusing_gates"] == []
    d01 = receipt["domains"][0]
    assert (d01["ra_lw_physics"], d01["ra_sw_physics"]) == (4, 4)
    assert d01["ra_rrtmg_variant"] == "rrtmg_legacy"
    profile = _gate(receipt, "hrrr.root_preparation.profile")
    assert profile["status"] == "ADMITS"
    assert "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1" in profile["detail"]
    assert _gate(receipt, "hrrr.route_inputs.physics")["status"] == "ADMITS"
    assert _gate(receipt, "hrrr.hierarchy.slice")["status"] == "ADMITS"
    assert _gate(receipt, "run_config.validate.d01")["status"] == "ADMITS"
    # The machine's own preconditions are asked HERE, before a byte is
    # fetched, and not by the configuration battery beside them: an
    # install-state question that lived in validate_run_config refused the
    # namelist importer, which translates a file and runs nothing.  This
    # tool is the surface that keeps asking it early.
    assert _gate(receipt, "run_preparation.d01")["status"] == "ADMITS"
    assert _gate(receipt, "hrrr.coverage")["status"] == "ADMITS"


def test_gray_zone_arms_are_admitted_by_every_static_gate(tmp_path):
    """The Shin-Hong route-admission motion, landed and visible here.

    Measured before it landed, on three separate cases and identical
    every time: an arm resolving divergence-ledger entry L3
    (``bl_pbl_physics`` 1 -> 11) was REFUSED by three independent gates
    -- the emission physics gate, the root-preparation profile gate and
    the certified hierarchy slice -- for one reason: no registered HRRR
    preparation profile carried a Shin-Hong PBL, so the scheme with the
    strongest scheme-level evidence in the tree had no run route at all.

    Registering the composition is what fixes it, not widening anything:
    the profile gate now names the profile the root preparer takes, and
    both other gates admit exactly the value that profile pins.  The
    arm that also carries L4 (``moist_mix6_off``) admits the same way --
    that entry moves a RunConfig key no route gate reads.
    """

    for name, patches in (("arm_L3", ["L3"]), ("arm_all", None)):
        config = _fidelity_arm(tmp_path, patches, name=name)
        receipt = _receipt(config)
        assert receipt["status"] == "ADMITTED", (name,
                                                 receipt["refusing_gates"])
        assert receipt["refusing_gates"] == [], name
        d01 = receipt["domains"][0]
        assert d01["bl_pbl_physics"] == 11, name
        # The rest of the composition is its sibling's, untouched.
        assert d01["mp_physics"] == 8, name
        assert (d01["ra_lw_physics"], d01["ra_sw_physics"]) == (4, 4), name
        assert d01["ra_rrtmg_variant"] == "rrtmg_legacy", name
        profile = _gate(receipt, "hrrr.root_preparation.profile")
        assert profile["status"] == "ADMITS", name
        assert ("thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1"
                in profile["detail"]), name
        assert _gate(receipt, "hrrr.route_inputs.physics")["status"] == (
            "ADMITS"), name
        assert _gate(receipt, "hrrr.hierarchy.slice")["status"] == (
            "ADMITS"), name


def test_the_l4_arm_resolves_the_ysu_profile_unchanged(tmp_path):
    """L4 composes without a route gate, so the pair stays controlled.

    The two runnable-before arms (faithful, and L4 alone) resolve the
    YSU composition and the L3 arms resolve the Shin-Hong one, so one
    root preparation per PBL serves its arms and the two profiles differ
    in the closure and nothing else.
    """

    config = _fidelity_arm(tmp_path, ["L4"], name="arm_L4")
    receipt = _receipt(config)
    assert receipt["status"] == "ADMITTED", receipt["refusing_gates"]
    assert receipt["domains"][0]["bl_pbl_physics"] == 1
    assert "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1" in _gate(
        receipt, "hrrr.root_preparation.profile")["detail"]


def test_route_accepts_implemented_pbl_without_profile_membership(tmp_path):
    """Implemented MYNN settings do not require a matching source preset."""

    from woof.experiment import load_experiment
    from woof.hrrr_route_inputs import (HrrrRouteInputError,
                                         validate_route_physics)

    text = FAITHFUL_ARM.read_text(encoding="utf-8")
    assert "bl_pbl_physics = 1" in text
    config = tmp_path / "mynn_pbl.toml"
    config.write_text(text.replace("bl_pbl_physics = 1",
                                   "bl_pbl_physics = 5"),
                      encoding="utf-8")

    validate_route_physics(load_experiment(config))
    receipt = _receipt(config)
    assert _gate(receipt, "hrrr.route_inputs.physics")["status"] == "ADMITS"
    assert _gate(receipt, "hrrr.root_preparation.profile")["status"] == "ADMITS"
    assert _gate(receipt, "hrrr.hierarchy.slice")["status"] == "ADMITS"


def test_native_source_does_not_gate_pbl_on_registered_profile_membership():
    from woof.hrrr_route_inputs import route_physics_problems
    from woof.physics_compat import WSM6_PROFILE_ID, single_domain_runtime_switches
    switches = single_domain_runtime_switches(WSM6_PROFILE_ID)
    for pbl in (0, 1, 5, 11):
        assert route_physics_problems(dict(switches, bl_pbl_physics=pbl)) == []


def test_the_registered_gray_zone_pair_differs_in_one_switch():
    """A paired run isolates the closure, so the rows must differ once."""

    from woof.physics_compat import single_domain_runtime_switches
    from woof.physics_registry import physics_registry

    ysu = single_domain_runtime_switches(
        "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1")
    shinhong = single_domain_runtime_switches(
        "thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1")
    assert {key for key in ysu if ysu[key] != shinhong[key]} == {
        "bl_pbl_physics"}
    assert shinhong["bl_pbl_physics"] == 11

    templates = physics_registry()["templates"]
    ysu_components = templates[
        "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1"]["components"]
    shinhong_template = templates[
        "thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1"]
    assert {key for key in ysu_components
            if ysu_components[key]
            != shinhong_template["components"][key]} == {"pbl"}
    assert shinhong_template["components"]["pbl"] == "shinhong"
    assert shinhong_template["maturity"] == "wrf-matched-run-candidate"
    assert any("stock-WRF-paired" in warning
               for warning in shinhong_template["warnings"]), (
        "the candidate label must name the receipt that upgrades it")


def test_route_accepts_declared_radiation_off(tmp_path):
    """The native source accepts an explicitly declared radiation-off selection."""

    from woof.experiment import load_experiment
    from woof.hrrr_route_inputs import (HrrrRouteInputError,
                                         validate_route_physics)

    config = _radiation_fully_off(tmp_path, declared=True)
    exp = load_experiment(config)
    validate_route_physics(exp)


def test_preflight_names_the_single_domain_runner_before_a_node_is_booked():
    """The gate the rented-node smoke was missing, 2026-08-04.

    Every static gate admitted the single-domain battery shape, the
    fetch, the preparation and the tree build all passed, and the run
    step was then refused by woof.prepared_domain_tree_forecast's
    designed two-domain floor.  The floor is division of labor and is
    NOT widened; the preflight now names that refusal, and the right
    runner, in the receipt itself.
    """

    for config in (SIZING_SMOKE, FAITHFUL_ARM):
        gate = _gate(_receipt(config), "run.runner_selection")
        assert gate["status"] == "INFO", config.name
        assert "requires at least two domains" in gate["detail"], config.name
        assert "not widened" in gate["detail"], config.name
        assert "tools/hrrr_single_domain_benchmark.py" in gate["detail"], (
            config.name)
        assert "dual-run" in gate["detail"], config.name


def test_preflight_prices_the_fd_budget_with_a_named_remedy():
    """The EMFILE class from the 2026-08-04 case run, priced up front.

    INFO, not a verdict: the limit belongs to the launching shell, and
    the fail-closed export step is the enforcing guard.  Whatever this
    platform reports, the gate must state the conservative need and --
    when measurable and short -- the exact ulimit remedy.
    """

    gate = _gate(_receipt(FAITHFUL_ARM), "run.fd_budget")
    assert gate["status"] == "INFO"
    assert "25 forcing hours x 8 workers" in gate["detail"]
    try:
        import resource
    except ImportError:
        assert "not measurable on this platform" in gate["detail"]
        assert "ulimit -n" in gate["detail"]
    else:
        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < 25 * 8 * 8 + 128:
            assert "ulimit -n" in gate["detail"]
        else:
            assert "covers the conservative need" in gate["detail"]


def test_preflight_names_the_tree_runner_for_a_real_tree():
    from types import SimpleNamespace

    gate = preflight._runner_selection(
        SimpleNamespace(domains=(object(), object())))
    assert gate.status == "INFO"
    assert "woof.prepared_domain_tree_forecast" in gate.detail
    assert "hrrr_single_domain_benchmark" not in gate.detail


def test_preflight_cli_exit_status_follows_the_verdict(tmp_path):
    # Valid configurations earn route receipts; missing required physical
    # declarations remain configuration errors before a receipt is formed.
    admitted = tmp_path / "admitted.json"
    faithful = tmp_path / "faithful.json"
    refused = tmp_path / "refused.json"
    refused_config = _radiation_fully_off(tmp_path, declared=False)
    assert preflight.main([
        "--experiment-config", str(SIZING_SMOKE),
        "--receipt", str(admitted), "--quiet"]) == 0
    assert preflight.main([
        "--experiment-config", str(FAITHFUL_ARM),
        "--receipt", str(faithful), "--quiet"]) == 0
    assert preflight.main([
        "--experiment-config", str(refused_config),
        "--receipt", str(refused), "--quiet"]) == 2
    assert json.loads(admitted.read_text())["status"] == "ADMITTED"
    assert json.loads(faithful.read_text())["status"] == "ADMITTED"
    assert not refused.exists()  # malformed/undeclared config never earns a route receipt


def test_a_token_less_radiation_off_config_is_refused_at_load(tmp_path):
    """Missing physical-experiment declarations fail configuration loading.

    No route verdict or receipt exists when the shared loader refuses the file.
    """

    from woof.experiment import load_experiment
    from woof.physics_compat import RADIATION_OFF_LAND_SURFACE_ACK

    config = _radiation_fully_off(tmp_path, declared=False)
    assert RADIATION_OFF_LAND_SURFACE_ACK not in config.read_text(
        encoding="utf-8")

    # Layer one: the config does not become an Experiment at all, so no
    # route logic runs.  The refusal names the token that would let it.
    with pytest.raises(ValueError,
                       match=RADIATION_OFF_LAND_SURFACE_ACK):
        load_experiment(config)

    # And the CLI reports that as a config error, not as a verdict.
    receipt = tmp_path / "never_written.json"
    assert preflight.main([
        "--experiment-config", str(config),
        "--receipt", str(receipt), "--quiet"]) == 2
    assert not receipt.exists(), (
        "the preflight wrote a receipt for a config it could not load; a "
        "receipt is a verdict about a run route and there was no experiment "
        "to have one")


def test_node_plan_generates_the_mirrored_namelist_from_the_config(tmp_path):
    """The WRF arm's physics is the battery config's, not a second copy."""

    outdir = tmp_path / "node"
    plan = node_plan.build(SIZING_SMOKE, outdir, ranks=24,
                           repository_root=REPOSITORY_ROOT)
    assert plan["schema"] == node_plan.PLAN_SCHEMA
    assert plan["ranks"] == 24
    for name in ("namelist.input", "namelist.wps", "d01-target.json",
                 "plan.json", "launch_arm.sh"):
        assert (outdir / name).is_file(), name

    namelist = (outdir / "namelist.input").read_text(encoding="ascii")
    # Geometry and clock mirror the config exactly.
    assert " e_we                                = 481," in namelist
    assert " e_sn                                = 401," in namelist
    assert " e_vert                              = 50," in namelist
    assert " time_step                           = 15," in namelist
    assert " mp_physics                          = 8," in namelist
    assert " cu_physics                          = 0," in namelist
    assert " bl_pbl_physics                      = 1," in namelist
    # The stock twin's documented deltas from the native namelist, and
    # the one that is no longer one: the WRF arm integrates dry theta,
    # as the export it runs from declares (EXPORT_USE_THETA_M).  It said
    # 1 while that export's T_B* rows were dry-coupled, which a
    # moist-theta wrf.exe read as a boundary colder than the analysis.
    assert " ra_lw_physics                       = 1," in namelist
    assert " use_theta_m                         = 0," in namelist
    assert " use_theta_m                         = 1," not in namelist
    assert " ghg_input                           = 0," in namelist

    # WRF declares hypsometric_opt in &domains (Registry.EM_COMMON:2283,
    # `namelist,domains`, nentries 1).  Emitted into &dynamics -- as it
    # was, until the 2026-08-04 node run -- the Fortran read of that
    # group fails and wrf.exe stops before its first timestep.  Section
    # scoped over BOTH files the nodes are handed, because a whole-file
    # substring passes on a namelist that still says &dynamics.
    from woof.namelist_import import parse_namelist

    for name in ("namelist.input", "namelist.native.input"):
        sections = parse_namelist(outdir / name)
        assert sections["domains"]["hypsometric_opt"] == [2], name
        assert "hypsometric_opt" not in sections["dynamics"], name

    target = json.loads((outdir / "d01-target.json").read_text())
    assert (target["nx"], target["ny"], target["nz"]) == (480, 400, 49)
    assert target["time_step_seconds"] == 15


def test_the_wrf_arm_is_told_to_produce_the_reflectivity_it_is_scored_on(
        tmp_path):
    """What the first WRF-arm scoring attempt measured.

    Every history frame came back missing REFL_10CM with all 200 other
    variables intact, so both B-04 WRF arms were unscorable on the
    registration's reflectivity family.  WRF gates that array on
    ``do_radar_ref`` (Registry.EM_COMMON:2447, ``namelist,physics``,
    nentries 1, default 0) -> derived ``compute_radar_ref``
    (module_check_a_mundo.F:3477) -> ``package radar_refl
    compute_radar_ref==1 - state:refl_10cm,refd_max``
    (Registry.EM_COMMON:3059).  At the default, under a scheme that is
    not Milbrandt/NSSL, the array is never even allocated.

    It is STOCK-ONLY, like ghg_input.  The WRF arm runs
    ``namelist.input`` -- launch_arm.sh copies exactly that file -- and
    the native namelist must stay silent, because woof evaluates
    REFL_10CM at output time whatever a namelist says and the certified
    raw-runtime contract refuses a native file that claims otherwise.
    Section scoped, and asserted on BOTH halves of that asymmetry.
    """

    from woof.hrrr_hierarchy_direct import _require_raw_stock_delta
    from woof.namelist_import import parse_namelist

    for config in (SIZING_SMOKE, FAITHFUL_ARM):
        outdir = tmp_path / f"node_{config.stem}"
        node_plan.build(config, outdir, ranks=24,
                        repository_root=REPOSITORY_ROOT)

        # The file the arm actually runs.
        stock = parse_namelist(outdir / "namelist.input")
        assert stock["physics"]["do_radar_ref"] == [1], config.name
        assert not [s for s, e in stock.items()
                    if s != "physics" and "do_radar_ref" in e], config.name

        # The file woof reads, which must not claim to control it.
        native = parse_namelist(outdir / "namelist.native.input")
        assert not [s for s, e in native.items()
                    if "do_radar_ref" in e], config.name

        # And the certified contract, over the emitted bytes: the key is
        # a registered stock-only delta now, not an unexplained fourth
        # difference between the pair.
        receipt = _require_raw_stock_delta(
            outdir / "namelist.native.input", outdir / "namelist.input")
        assert receipt["status"] == "PASS", config.name
        assert receipt["allowed_deltas"]["physics.do_radar_ref"] == {
            "native": None, "stock": [1]}, config.name


def test_mirrored_namelist_round_trips_moist_mix6_off(tmp_path):
    """Spec section 6.3: the mirrored namelist is generated from the
    battery config, so the L4 candidate switch must survive the full
    circle -- config -> rendered namelist -> re-import -> same
    experiment.  An omitted emitter line here would silently hand the
    WRF arm the Registry default while the WOOF-ArWen arm read the TOML."""

    from woof.experiment import load_experiment
    from woof.hrrr_route_inputs import verify_round_trip

    text = SIZING_SMOKE.read_text(encoding="utf-8")
    assert "moist_mix6_off" not in text
    config = tmp_path / "smoke_mix6.toml"
    config.write_text(
        text.replace("[shared]", "[shared]\nmoist_mix6_off = true", 1),
        encoding="utf-8")
    outdir = tmp_path / "node"
    node_plan.build(config, outdir, ranks=24,
                    repository_root=REPOSITORY_ROOT)
    native = (outdir / "namelist.native.input").read_text(encoding="ascii")
    stock = (outdir / "namelist.input").read_text(encoding="ascii")
    line = " moist_mix6_off                      = .true.,"
    assert line in native
    assert line in stock, "the WRF arm must carry the same moist filter"
    exp = load_experiment(config)
    assert exp.root.run.moist_mix6_off is True
    # The route's own round-trip verifier: re-imports the real bytes and
    # demands the same per-domain effective identity.
    verify_round_trip(exp, outdir / "namelist.wps",
                      outdir / "namelist.native.input")


def test_node_plan_mirrors_the_faithful_arms_radiation(tmp_path):
    """The mirrored WRF arm carries the battery's 4/4 RRTMG selection.

    Under the (4, 4) pair the documented longwave delta collapses --
    both arms run 4 -- while stock-only ghg_input=0 remains, mirroring
    what woof/core/rrtmg_legacy.py pins on the native side.  Both arms
    declare dry theta.
    """

    outdir = tmp_path / "node"
    node_plan.build(FAITHFUL_ARM, outdir, ranks=24,
                    repository_root=REPOSITORY_ROOT)
    native = (outdir / "namelist.native.input").read_text(encoding="ascii")
    stock = (outdir / "namelist.input").read_text(encoding="ascii")
    for text in (native, stock):
        assert " ra_lw_physics                       = 4," in text
        assert " ra_sw_physics                       = 4," in text
    assert " use_theta_m                         = 0," in native
    assert " use_theta_m                         = 0," in stock
    assert " ghg_input                           = 0," not in native
    assert " ghg_input                           = 0," in stock


def test_node_plan_emits_the_resolved_fidelity_axis(tmp_path,
                                                     pinned_thompson_tables):
    """What the first Shin-Hong root preparation on node 18 found.

    The axis resolved ``bl_pbl_physics`` to 11 and the loader wrote it
    onto every domain -- and the plan emitter spelled the key as a
    literal 1, so the mirrored namelist described the arm that was NOT
    being run.  The root preparer's profile validator refused the drift,
    correctly, after the arm had been composed and a node booked.

    Asserted against the axis's OWN resolution rather than a scheme
    number written here: a test that pins 11 by hand passes on an
    emitter that has simply swapped one literal for another.
    """

    from woof.experiment import load_experiment
    from woof.namelist_import import parse_namelist
    from woof.physics_compat import (
        THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID)
    from tools.hrrr_single_domain_benchmark import (
        _validate_native_hrrr_physics_profile)

    for name, patches in (("arm_L3", ["L3"]), ("arm_all", None)):
        config = _fidelity_arm(tmp_path, patches, name=name)
        exp = load_experiment(config)
        resolved = exp.physics_mode.settings
        assert resolved, name  # the axis governs this arm

        outdir = tmp_path / f"node_{name}"
        node_plan.build(config, outdir, ranks=24,
                        repository_root=REPOSITORY_ROOT)

        # Every governed key, in both emitted flavors, at the value the
        # axis resolved -- and nowhere else in the file.
        for flavor in ("namelist.input", "namelist.native.input"):
            sections = parse_namelist(outdir / flavor)
            for key, value in resolved.items():
                carrying = [entries[key] for entries in sections.values()
                            if key in entries]
                assert len(carrying) == 1, (name, flavor, key)
                assert carrying[0] == [value], (name, flavor, key)

        # And the gate that refused: the emitted native namelist now
        # binds the profile the arm's own physics resolves to.
        binding = _validate_native_hrrr_physics_profile(
            outdir / "namelist.native.input",
            THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID)
        assert binding["profile"] == (
            THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID), name


def test_an_ungoverned_configs_emission_is_byte_inert(tmp_path):
    """The axis promises absent == wrf-faithful, byte for byte.

    The whole tree predates the fidelity axis, so naming the axis at its
    faithful value must produce the same namelists as not naming it at
    all.  Byte comparison, because "the physics matches" is what the
    emitter was already believed to guarantee.
    """

    plain = tmp_path / "plain"
    node_plan.build(FAITHFUL_ARM, plain, ranks=24,
                    repository_root=REPOSITORY_ROOT)

    from woof import physics_mode

    declared_config = _fidelity_arm(
        tmp_path, None, name="faithful_declared")
    declared_config.write_text(
        declared_config.read_text(encoding="utf-8").replace(
            f'"{physics_mode.PHYSICS_MODE_ARWEN_PATCHED}"',
            f'"{physics_mode.PHYSICS_MODE_WRF_FAITHFUL}"', 1),
        encoding="utf-8")
    declared = tmp_path / "declared"
    node_plan.build(declared_config, declared, ranks=24,
                    repository_root=REPOSITORY_ROOT)

    for flavor in ("namelist.input", "namelist.native.input"):
        assert (plain / flavor).read_bytes() == (
            declared / flavor).read_bytes(), flavor


def test_emission_refuses_to_contradict_the_axis_it_rendered_from(tmp_path):
    """The guard, fired -- the generic half of this fix.

    Rendering every governed key from the resolved config is the repair;
    this is what stops the NEXT one being spelled as a literal, and it
    names no ledger entry and no scheme number so it grows with the
    ledger.  Two firings: a governed key emitted at a value the axis did
    not resolve, and an APPLIED patch whose key never reaches the
    namelist at all (a WRF arm that cannot express the patch is not a
    mirror of the woof arm).
    """

    import dataclasses

    from woof.experiment import load_experiment
    from woof.hrrr_route_inputs import (HrrrRouteInputError,
                                         render_namelist_input,
                                         verify_axis_authored_keys)
    from woof import physics_mode

    faithful = load_experiment(FAITHFUL_ARM)
    text = render_namelist_input(faithful)

    def resolution(*patches):
        return physics_mode.PhysicsModeResolution(
            mode=physics_mode.PHYSICS_MODE_ARWEN_PATCHED, patchset="v1",
            requested_patches=tuple(p.entry_id for p in patches),
            resolved=patches, governed=True)

    # The measured defect's exact shape: the axis says one thing, the
    # emitted bytes say another.  The entry and value are taken from the
    # ledger, so this stays true when the ledger's numbers change.
    l3 = physics_mode.LEDGER_BY_ID["L3"]
    drifted = dataclasses.replace(
        faithful,
        physics_mode=resolution(physics_mode.ResolvedPatch(
            entry_id=l3.entry_id, key=l3.key, value=l3.patched,
            patched=True, summary=l3.summary)))
    with pytest.raises(HrrrRouteInputError, match=l3.key):
        verify_axis_authored_keys(drifted, text)

    # An applied patch on a key no WRF namelist carries.  moist_cq is a
    # real woof key with no WRF counterpart (WRF derives calc_cq from
    # the species that exist), so this is the accurate shape of a future
    # ledger entry the mirrored arm could not express.
    absent = dataclasses.replace(
        faithful,
        physics_mode=resolution(physics_mode.ResolvedPatch(
            entry_id="LX", key="moist_cq", value=True, patched=True,
            summary="a ledger entry no WRF namelist can carry")))
    with pytest.raises(HrrrRouteInputError, match="carries no such key"):
        verify_axis_authored_keys(absent, text)

    # Resolved to its FAITHFUL side, the same absent key is silent: the
    # faithful arm is what the emitter has always written.
    verify_axis_authored_keys(
        dataclasses.replace(
            faithful,
            physics_mode=resolution(physics_mode.ResolvedPatch(
                entry_id="LX", key="moist_cq", value=False, patched=False,
                summary="a ledger entry no WRF namelist can carry"))),
        text)


def test_a_governed_key_in_the_config_body_is_still_refused(tmp_path):
    """The axis's own two-authors refusal, preserved by this fix.

    The emitter now authors governed keys from the resolved config, and
    that must not become a second way to write them: a case body that
    pins a governed key while the overlay governs it is still refused by
    the loader, by name, before any emission happens.
    """

    from woof.experiment import load_experiment
    from woof import physics_mode

    config = _fidelity_arm(tmp_path, ["L3"], name="two_authors")
    governed = physics_mode.governed_keys(
        physics_mode.PHYSICS_MODE_ARWEN_PATCHED, "v1", ["L3"])
    key = governed[0]
    body = config.read_text(encoding="utf-8").replace(
        "[shared]", f"[shared]\n{key} = 1", 1)
    clashing = tmp_path / "two_authors_clash.toml"
    clashing.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=key):
        load_experiment(clashing)
    with pytest.raises(ValueError, match=key):
        node_plan.build(clashing, tmp_path / "never", ranks=24,
                        repository_root=REPOSITORY_ROOT)


def test_launch_script_survives_a_pre_existing_tmux_server(tmp_path):
    """Both measured launch failures from the 2026-08-04 node runs.

    (a) The campaign nodes already run a tmux server for unrelated
    long-lived sessions, so a session created under it inherits nothing
    from the launching shell: ``export WRF_BUILD`` reached no one, mpirun
    tried to exec ``/main/wrf.exe`` and all 24 ranks died.  The value now
    goes to the SERVER, which is what a new session inherits from.

    (b) mpirun is on neither the default nor the login PATH of those
    nodes; oneAPI's setvars.sh is what puts it there, and the arm sources
    it -- guarded, since a node without the file must say so rather than
    fail on a missing script.
    """

    outdir = tmp_path / "node"
    node_plan.build(SIZING_SMOKE, outdir, ranks=24,
                    repository_root=REPOSITORY_ROOT)
    script = (outdir / "launch_arm.sh").read_text(encoding="ascii")

    # (a) The build path reaches the session through the tmux server
    # itself, not through an exported variable the server never sees.
    publish = script.index('tmux setenv -g WRF_BUILD "$WRF_BUILD"')
    # start-server is what makes setenv work on a node with no server
    # yet, and a no-op on one that has had a session open for weeks.
    assert script.index("\ntmux start-server\n") < publish
    assert publish < script.index('tmux new-session -d -s "$SESSION"')
    # The arm refuses to launch on an empty value rather than handing
    # mpirun the bare "/main/wrf.exe" every rank died on.
    assert '[ -n "${WRF_BUILD:-}" ] || fail' in script
    assert "/main/wrf.exe" in script

    # (b) sourced when present, and a launch that still cannot find
    # mpirun says exactly that instead of dying inside MPI.
    assert "if [ -f /opt/intel/oneapi/setvars.sh ]; then" in script
    assert ". /opt/intel/oneapi/setvars.sh" in script
    assert "command -v mpirun > /dev/null 2>&1 || fail" in script
    assert "mpirun is not on PATH" in script

    # Both arms still leave the launch's own verdict on disk.
    assert "echo $? > exit.code" in script
    assert "echo 127 > exit.code" in script


def test_node_plan_accepts_declared_radiation_off(tmp_path):
    """Node planning accepts the same declared configuration as the runtime."""

    from woof.hrrr_route_inputs import HrrrRouteInputError

    config = _radiation_fully_off(tmp_path, declared=True)
    outdir = tmp_path / "node"
    node_plan.build(config, outdir, ranks=24, repository_root=REPOSITORY_ROOT)
    assert node_plan.main([
        "--experiment-config", str(config),
        "--outdir", str(tmp_path / "cli")]) == 0


def test_node_plan_prices_the_arms_from_the_committed_speed_anchor(tmp_path):
    plan = node_plan.build(SIZING_SMOKE, tmp_path / "node", ranks=24,
                           repository_root=REPOSITORY_ROOT)
    projection = plan["projection"]
    anchors = json.loads(SPEED_ANCHORS.read_text(encoding="utf-8"))
    entry = anchors["anchors"]["wrf_v461_nodes_24rank"]
    expected = (entry["seconds_per_sim_minute"]
                * BATTERY_CELLS / entry["cells"])
    assert projection["projected_seconds_per_sim_minute"] == pytest.approx(
        expected, rel=1e-4)
    # The anchor's accuracy travels with every number derived from it.
    assert projection["anchor_status"] == "unreceipted-controller-measurement"
    arms = {arm["arm"] for arm in plan["arms"]}
    assert arms == {"W", "W-twin"}
    twin = next(arm for arm in plan["arms"] if arm["arm"] == "W-twin")
    assert "tools/n5s/perturb_ulp.py" in twin["command"]
    assert "--member-ordinal 0" in twin["command"]


def test_twin_field_and_member_ordinal_are_in_the_perturbers_domain():
    """The plan's twin constants stay in rung 1's documented domain.

    The full binding against the perturber's own signature lives in
    ``tests/test_n5s_toolchain.py``
    (``test_battery_node_plan_twin_command_matches_the_perturber``):
    the perturber is release-excluded campaign harness, so a shipped
    test may not import it -- the snapshot gate
    (test_release_snapshot_machine_paths.py) enforces exactly that.
    """

    assert node_plan.TWIN_FIELD == "T"
    # ``--member-ordinal`` is the perturber's own documented member index
    # and it refuses anything outside 0..3, so the plan may not invent one.
    assert node_plan.TWIN_MEMBER_ORDINAL in range(4)


def _release_drops(rel: str) -> str | None:
    """The RELEASE-EXCLUDE.txt rule that keeps ``rel`` out of the public tree."""
    return matches(rel, read_exclusions(REPOSITORY_ROOT))


def test_committed_speed_anchor_matches_the_run_report_it_cites():
    """The committed anchor is a receipt on the public tree too.

    The speed-anchor file ships, and it called this anchor a committed run
    receipt while citing a report under ``evidence/**``, which the public
    export drops: a public reader could not open the receipt the status
    promised.  The report ships beside the anchor now, its bytes pinned by
    ``evidence_sha256`` (which is also what lets the release's machine-path
    scan forgive the run's own output paths inside it).
    """
    anchors = json.loads(SPEED_ANCHORS.read_text(encoding="utf-8"))
    entry = anchors["anchors"]["arwen_3km_conus"]
    assert entry["status"] == "committed-run-receipt"
    rule = _release_drops(entry["evidence"])
    assert rule is None, (
        f"the anchor is labelled a committed run receipt but cites "
        f"{entry['evidence']}, which the public export drops "
        f"(RELEASE-EXCLUDE.txt: {rule})")
    raw = (REPOSITORY_ROOT / entry["evidence"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry["evidence_sha256"]
    report = json.loads(raw.decode("utf-8"))
    assert report["domain"]["nx"] == entry["grid"]["nx"]
    assert report["domain"]["ny"] == entry["grid"]["ny"]
    assert report["domain"]["nz"] == entry["grid"]["nz"]
    assert report["wall_seconds"] == entry["wall_seconds"]
    assert report["run_seconds"] == entry["run_seconds"]
    assert (report["memory"]["gpu_peak_used_bytes_observed"]
            == entry["gpu_peak_used_bytes_observed"])
    assert (report["gridded_output"]["total_bytes"]
            == entry["gridded_output_total_bytes"])
    rate = entry["wall_seconds"] / (entry["run_seconds"] / 60.0)
    assert entry["seconds_per_sim_minute"] == pytest.approx(rate, abs=5e-5)


def test_the_route_projection_scales_from_the_anchor_the_public_tree_carries():
    """The preflight prices from the same shipped report the anchor cites.

    Read from ``evidence/``, the wall-clock projection was absent from every
    public run of the tool, reported only as "not found".
    """
    anchors = json.loads(SPEED_ANCHORS.read_text(encoding="utf-8"))
    cited = anchors["anchors"]["arwen_3km_conus"]["evidence"]
    assert preflight.ANCHOR_RECEIPT.as_posix() == cited
    assert _release_drops(cited) is None
    receipt = _receipt(SIZING_SMOKE)
    assert receipt["projection"]["available"] is True
    assert receipt["projection"]["anchor"] == cited
    assert _gate(receipt, "arwen.wall_clock_projection")["authority"] == cited
    for name in ("B4-ARWEN-HRRR-ROUTE.json", "B4-ARWEN-HRRR-SIZING-SMOKE.json"):
        committed = json.loads(
            (SPEED_ANCHORS.parent / name).read_text(encoding="utf-8"))
        assert committed["projection"]["anchor"] == cited, name
        assert _gate(committed, "arwen.wall_clock_projection")[
            "authority"] == cited, name


def test_the_memory_fit_ships_the_evidence_it_cites():
    """The fit is a committed measurement on the public tree too.

    The anchors file labelled the fit "committed-measurement-fit" while
    citing notes under ``evidence/**``, and the preflight wrote the same
    path as ``fit_source`` into every receipt, both committed ones
    included; the public export drops ``evidence/**``, so a public reader
    was sent to a file their install does not have.  The fit and the 4 Hz
    samples it rests on ship beside the anchor now, their bytes pinned.
    """
    anchors = json.loads(SPEED_ANCHORS.read_text(encoding="utf-8"))
    fit = anchors["device_memory_fit"]
    assert fit["status"] == "committed-measurement-fit"
    for key in ("evidence", "samples"):
        rel = fit[key]
        rule = _release_drops(rel)
        assert rule is None, (
            f"the fit is labelled a committed measurement but its {key} "
            f"{rel} is dropped by the public export (RELEASE-EXCLUDE.txt: "
            f"{rule})")
        raw = (REPOSITORY_ROOT / rel).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == fit[f"{key}_sha256"], rel
    cited = fit["evidence"]
    assert preflight.FIT_SOURCE.as_posix() == cited
    assert _receipt(SIZING_SMOKE)["sizing"]["fit_source"] == cited
    for name in ("B4-ARWEN-HRRR-ROUTE.json", "B4-ARWEN-HRRR-SIZING-SMOKE.json"):
        committed = json.loads(
            (SPEED_ANCHORS.parent / name).read_text(encoding="utf-8"))
        assert committed["sizing"]["fit_source"] == cited, name


def _fit_rungs(text: str) -> list[tuple[str, int, float]]:
    """The eight RTX 4080 rungs the shipped fit prints, as (name, cells, GiB)."""
    rows = re.findall(
        r"^(\S+)\s+(\d{6,})\s+(\d+\.\d+)\s+\d+\.\d+\s+[+-]\d+\.\d+$",
        text, re.MULTILINE)
    return [(name, int(cells), float(gib)) for name, cells, gib in rows]


def test_the_memory_fit_is_arithmetic_over_what_ships():
    """Every number of the fit comes back out of the shipped files.

    The 5090 point is re-read from the samples, the 4080 line is refitted
    from the rungs the document prints, and the anchor's slope and the
    preflight's constants are recomputed from the two.
    """
    anchors = json.loads(SPEED_ANCHORS.read_text(encoding="utf-8"))
    fit = anchors["device_memory_fit"]
    point = fit["anchor_point"]
    assert point["cells"] == anchors["anchors"]["arwen_3km_conus"][
        "grid"]["cells"]

    device_peak = 0
    process_peak: dict[str, int] = {}
    lines = (REPOSITORY_ROOT / fit["samples"]).read_text(
        encoding="utf-8").splitlines()
    assert lines[0].split("\t") == [
        "epoch", "dev_used_mib", "dev_total_mib", "util", "procs"]
    for line in lines[1:]:
        _, used, _, _, procs = line.split("\t")
        device_peak = max(device_peak, int(used))
        for entry in filter(None, procs.split(";")):
            pid, mib = entry.split(":")
            process_peak[pid] = max(process_peak.get(pid, 0), int(mib))
    assert device_peak == point["device_peak_mib"]
    assert max(process_peak.values()) == point["forecast_process_peak_mib"]

    text = (REPOSITORY_ROOT / fit["evidence"]).read_text(encoding="utf-8")
    rungs = _fit_rungs(text)
    assert len(rungs) == 8, rungs
    printed = re.search(
        r"slope (\d+\.\d+) KiB/cell\s+intercept (\d+\.\d+) GiB", text)
    assert printed, "the shipped fit no longer prints its slope and intercept"
    xs =[cells for _, cells, _ in rungs]
    ys = [gib for _, _, gib in rungs]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = (sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
             / sum((x - mean_x) ** 2 for x in xs))
    intercept = mean_y - slope * mean_x
    assert slope * 1024.0 ** 2 == pytest.approx(
        float(printed.group(1)), abs=5e-5)
    assert intercept == pytest.approx(float(printed.group(2)), abs=5e-4)

    def slope_at(intercept_gib: float) -> float:
        peak_kib = point["forecast_process_peak_mib"] * 1024.0
        return (peak_kib - intercept_gib * 1024.0 ** 2) / point["cells"]

    kib = fit["slope_kib_per_cell"]
    # The best value assumes the suite's itemized non-pool 2.91 GiB, the
    # high end the 2.00 GiB intercept row, the low end the 4080 line.
    assert round(slope_at(2.91), 2) == kib["best"]
    assert round(slope_at(2.00), 2) == kib["high"]
    assert kib["low"] == pytest.approx(slope * 1024.0 ** 2, abs=0.01)
    gib = fit["intercept_gib"]
    assert gib["low"] == pytest.approx(intercept, abs=0.05)
    assert gib["low"] < gib["high"]
    assert preflight.FIT_SLOPE_KIB_PER_CELL == (
        kib["low"], kib["best"], kib["high"])
    assert preflight.FIT_INTERCEPT_GIB == (gib["low"], gib["high"])


#: What a public receipt says beside a path the public export drops.
_DEVELOPMENT_ONLY = "development checkouts only"
_CITED_PATH = re.compile(
    r"(?<![\w./-])((?:docs|evidence|tools|configs|woof|tests|work)/"
    r"[\w./-]*\w)")


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def test_the_speed_anchors_cite_nothing_a_public_reader_cannot_open_unsaid():
    """A path the export drops is named as development-only where it is cited.

    The anchors file ships; without this, the next citation into
    ``evidence/**`` or ``docs/internal/**`` reads as a receipt a public
    reader can open, as the memory fit's did.
    """
    anchors = json.loads(SPEED_ANCHORS.read_text(encoding="utf-8"))
    cited = 0
    offenders = []
    for text in _strings(anchors):
        for rel in _CITED_PATH.findall(text):
            cited += 1
            rule = _release_drops(rel)
            if rule is not None and _DEVELOPMENT_ONLY not in text:
                offenders.append(f"{rel} (RELEASE-EXCLUDE.txt: {rule})")
    assert cited >= 5, "the scan found almost no cited path"
    assert offenders == [], offenders


def test_unreceipted_anchors_say_so_and_name_their_upgrade():
    anchors = json.loads(SPEED_ANCHORS.read_text(encoding="utf-8"))
    for name in ("wrf_v461_nodes_24rank", "arwen_5090_same_domain"):
        entry = anchors["anchors"][name]
        assert entry["status"] == "unreceipted-controller-measurement", name
        assert entry["upgrade_path"], name
        assert "NO run receipt" in entry["evidence"], name


def test_battery_shape_projections_are_arithmetic_over_the_anchors():
    anchors = json.loads(SPEED_ANCHORS.read_text(encoding="utf-8"))
    shape = anchors["battery_shape"]
    assert shape["cells"] == BATTERY_CELLS
    projections = shape["projections"]

    arwen = anchors["anchors"]["arwen_3km_conus"]
    rate = arwen["wall_seconds"] / (arwen["run_seconds"] / 60.0)
    scale = shape["cells"] / arwen["grid"]["cells"]
    assert projections["arwen_from_3km_anchor"][
        "seconds_per_sim_minute"] == pytest.approx(rate * scale, abs=5e-5)
    assert projections["arwen_from_3km_anchor"][
        "wall_hours_24h"] == pytest.approx(
            rate * scale * 1440.0 / 3600.0, abs=5e-4)

    wrf = anchors["anchors"]["wrf_v461_nodes_24rank"]
    wrf_rate = wrf["seconds_per_sim_minute"] * shape["cells"] / wrf["cells"]
    entry = projections["wrf_from_controller_pair"]
    assert entry["seconds_per_sim_minute"] == pytest.approx(
        wrf_rate, abs=5e-5)
    assert entry["wall_hours_24h"] == pytest.approx(
        wrf_rate * 1440.0 / 3600.0, abs=5e-4)
    assert entry["wall_hours_per_case_control_plus_twin"] == pytest.approx(
        2.0 * wrf_rate * 1440.0 / 3600.0, abs=1e-3)


def test_the_stock_namelist_carries_each_domains_diffusion_selectors():
    """``diff_opt`` and ``mix_full_fields`` are written per domain from config.

    The breakage this prevents: the writer spelled ``diff_opt = 2`` and
    ``mix_full_fields = .true.`` as literals, so a ``diff_opt = 1`` run, or
    one with ``mix_full_fields = false``, was refused by its own
    config-versus-namelist round trip with a mismatch that named neither
    key's real cause, and a mirrored WRF arm would have run the other
    operator.
    """
    import dataclasses

    from woof.experiment import load_experiment
    from woof.hrrr_route_inputs import render_namelist_input

    def column(text, key):
        match = re.search(rf"^ {key}\s*=\s*(.*)$", text, flags=re.M)
        assert match, key
        return [v.strip() for v in match.group(1).split(",") if v.strip()]

    exp = load_experiment(FAITHFUL_ARM)
    count = len(exp.domains)
    text = render_namelist_input(exp)
    assert column(text, "diff_opt") == ["2"] * count
    assert column(text, "mix_full_fields") == [".true."] * count

    coordinate = dataclasses.replace(exp, domains=type(exp.domains)(
        dataclasses.replace(domain, run=dataclasses.replace(
            domain.run, diff_opt=1, mix_full_fields=False))
        for domain in exp.domains))
    text = render_namelist_input(coordinate)
    assert column(text, "diff_opt") == ["1"] * count
    assert column(text, "mix_full_fields") == [".false."] * count
