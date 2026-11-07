"""The physics-fidelity axis, its refusals, and the ledger it resolves.

Three things are pinned here and each one is a different failure the axis
exists to prevent:

1. ABSENCE IS INERT.  A configuration that does not name the axis resolves
   exactly as it did before the axis existed.  This is the pin that lets the
   axis land in a tree whose whole point is bitwise reproducibility.
2. THE MODE IS THE AUTHOR.  When the axis is named it writes its keys and
   refuses a second author, in either mode.  A merge here would produce a
   run whose receipt names an arm it did not integrate.
3. THE VECTOR IS FROZEN BY VERSION.  ``patchset = "v1"`` resolves the same
   entries tomorrow as today, and an entry the ledger holds back (SASE's
   entry gate, the dormant class-C rows) cannot be smuggled into a vector.
"""

from __future__ import annotations

import dataclasses
import tomllib

import pytest

from woof import physics_mode as pm
from woof.config import RunConfig
from woof.experiment import build_experiment


# This probe is a REAL case (it has a [projection]) that runs Noah with no
# radiation selector at all, which is precisely the class
# woof.physics_compat.radiation_off_land_surface_refusal refuses at load:
# nothing computes the downward longwave Noah reads every surface step.  The
# axis under test here is the physics-fidelity one and its 60 seconds never
# read a surface budget, so the probe DECLARES the experiment rather than
# muting the guard -- the same answer the shipped 60-second VRAM/step-cost
# probes give, and it keeps every arm below loading through the real front
# door instead of a fixture-only bypass.
#
# BOTH tokens, for the same reason the shipped probes carry both: radiation
# entirely off under a land surface is the one class the two 1.8.8 guards
# overlap on, and they ask different questions -- "nothing computes my sky"
# and "the number my land surface integrates is one I typed".  Dropping
# either one puts this fixture back behind a refusal.
BASE_TOML = """
[experiment]
name = "axis-probe"
start_time = 2021-06-01T12:00:00
run_seconds = 60.0
restart_interval_s = 0.0
acknowledgements = ["radiation-off-land-surface-v1",
                    "constant-downward-longwave-v1"]
{extra}

[projection]
map_proj = "lambert"
ref_lat = 38.0
ref_lon = -97.0
truelat1 = 30.0
truelat2 = 45.0
stand_lon = -97.0

[shared]
nz = 8
ztop = 20000.0
map_proj = 1
moist = true
mp_physics = 8
diff_6th_opt = 2
diff_6th_factor = 0.12
sf_sfclay_physics = 1
sf_surface_physics = 2
{shared_extra}

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
history_interval_s = 60.0
e_we = 41
e_sn = 41
dx = 3000.0
dy = 3000.0
time_step = 15
specified = true
{domain_extra}
"""


def _build(extra: str = "", shared_extra: str = "", domain_extra: str = ""):
    text = BASE_TOML.format(
        extra=extra, shared_extra=shared_extra, domain_extra=domain_extra)
    return build_experiment(tomllib.loads(text), source="axis-probe.toml")


# --------------------------------------------------------------- 1. inert

def test_an_absent_axis_authors_nothing_and_changes_no_value():
    """Absence is the state of every configuration written before the axis.

    It must not merely resolve -- it must resolve to the SAME RunConfig,
    field for field, as the identical file does with the axis code removed.
    The proxy for "code removed" is the configuration's own explicit values:
    the axis's two keys keep whatever the file said (or their RunConfig
    defaults), which is what "authors nothing" means.
    """

    exp = _build(shared_extra="bl_pbl_physics = 5\n")
    run = exp.domains[0].run
    assert exp.physics_mode is pm.UNGOVERNED
    assert exp.physics_mode.governed is False
    # Reported mode is still the faithful one, because that is true of it.
    assert exp.physics_mode.mode == pm.PHYSICS_MODE_WRF_FAITHFUL
    assert exp.physics_mode.resolved == ()
    # Neither ledger key was touched: MYNN survives, and moist_mix6_off is
    # RunConfig's own WRF default.
    assert run.bl_pbl_physics == 5
    assert run.moist_mix6_off is False
    assert RunConfig.__dataclass_fields__["moist_mix6_off"].default is False


def test_an_absent_axis_still_reports_itself_in_a_receipt_shape():
    receipt = pm.UNGOVERNED.receipt()
    assert receipt["physics_mode"] == pm.PHYSICS_MODE_WRF_FAITHFUL
    assert receipt["governed"] is False
    assert receipt["resolved_patch_vector"] == []
    assert receipt["patchset"] is None


# ------------------------------------------------------ 2. mode as author

def test_faithful_mode_writes_the_wrf_side_of_every_ledger_edge():
    exp = _build(extra='physics_mode = "wrf-faithful"')
    run = exp.domains[0].run
    assert run.bl_pbl_physics == 1          # L3 faithful: YSU
    assert run.moist_mix6_off is False      # L4 faithful: WRF's default
    assert exp.physics_mode.governed is True
    assert [p.patched for p in exp.physics_mode.resolved] == [False, False]


def test_patched_mode_writes_the_whole_registered_vector():
    exp = _build(extra='physics_mode = "arwen-patched"')
    run = exp.domains[0].run
    assert run.bl_pbl_physics == 11         # L3 patched: Shin-Hong
    assert run.moist_mix6_off is True       # L4 patched
    assert exp.physics_mode.requested_patches == ("L3", "L4")
    assert exp.physics_mode.patchset == "v1"


@pytest.mark.parametrize(
    "patches, pbl, mix6",
    [('patches = ["L3"]', 11, False),
     ('patches = ["L4"]', 1, True)])
def test_a_single_patch_arm_moves_exactly_one_edge(patches, pbl, mix6):
    """The ablation arms P-L3 and P-L4: one edge each, the rest faithful.

    Single-patch arms are the promotion evidence, so an arm that quietly
    carried a second patch would attribute one patch's effect to another.
    """

    exp = _build(extra=f'physics_mode = "arwen-patched"\n{patches}')
    run = exp.domains[0].run
    assert (run.bl_pbl_physics, run.moist_mix6_off) == (pbl, mix6)


def test_the_axis_writes_every_domain_of_a_tree():
    """One experiment is one arm: no domain may sit on the other side."""

    child = """
[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 12
j_parent_start = 12
parent_grid_ratio = 3
parent_time_step_ratio = 3
history_interval_s = 60.0
e_we = 31
e_sn = 31
nested = true
"""
    exp = build_experiment(
        tomllib.loads(
            BASE_TOML.format(
                extra='physics_mode = "arwen-patched"',
                shared_extra="", domain_extra="") + child),
        source="axis-probe.toml")
    assert len(exp.domains) == 2
    assert all(d.run.moist_mix6_off is True for d in exp.domains)
    assert all(d.run.bl_pbl_physics == 11 for d in exp.domains)


def test_one_case_file_serves_every_arm_through_the_real_loader(tmp_path):
    """The composition claim, exercised on the entry point the CLI uses.

    The arms differ by the [experiment] overlay and by nothing else: the
    same [shared] and [[domain]] bytes serve all of them, which is the whole
    reason the axis is experiment-scope sugar instead of a second case file
    per arm that can drift from the first.
    """

    from woof.experiment import load_experiment

    body = BASE_TOML.format(extra="{extra}", shared_extra="", domain_extra="")
    patched = 'physics_mode = "arwen-patched"'
    arms = {
        "F": 'physics_mode = "wrf-faithful"',
        "P-L3": patched + '\npatches = ["L3"]',
        "P-L4": patched + '\npatches = ["L4"]',
        "P-all": patched,
    }
    resolved = {}
    for arm, overlay in arms.items():
        path = tmp_path / f"arm-{arm}.toml"
        path.write_text(body.format(extra=overlay))
        exp = load_experiment(path)
        run = exp.domains[0].run
        resolved[arm] = (run.bl_pbl_physics, run.moist_mix6_off)

    assert resolved == {
        "F": (1, False),
        "P-L3": (11, False),
        "P-L4": (1, True),
        "P-all": (11, True),
    }
    # Four distinct physics identities from one case body.
    assert len(set(resolved.values())) == 4


@pytest.mark.parametrize("mode", ["wrf-faithful", "arwen-patched"])
@pytest.mark.parametrize(
    "where, extra",
    [("shared", "bl_pbl_physics = 1"),
     ("domain", "moist_mix6_off = false")])
def test_a_second_author_for_a_governed_key_is_refused_in_either_mode(
        mode, where, extra):
    """Refused, not merged -- and refused even when the values AGREE.

    Agreement is a property of today's value, not of the configuration: the
    ledger's faithful/patched edge can move under a new patch-set version
    while the hand-written value stays put, and the run would then integrate
    a value nobody chose while the receipt named an arm.
    """

    kwargs = {"extra": f'physics_mode = "{mode}"'}
    kwargs["shared_extra" if where == "shared" else "domain_extra"] = extra
    with pytest.raises(ValueError, match="already the author"):
        _build(**kwargs)


def test_the_refusal_names_the_ledger_entry_and_the_governed_key_list():
    with pytest.raises(ValueError) as excinfo:
        _build(extra='physics_mode = "arwen-patched"',
               shared_extra="bl_pbl_physics = 1")
    message = str(excinfo.value)
    assert "'bl_pbl_physics'" in message
    assert "divergence-ledger L3" in message
    assert "governed_keys" in message


def test_governed_keys_is_the_list_a_composer_must_strip():
    """The public list, checked against what the loader actually refuses.

    A composer that strips a hand-maintained copy of this list is a composer
    that stops stripping one the day the ledger grows, so the list is
    exported and this is the pin that it is the same list.
    """

    for mode in pm.PHYSICS_MODES:
        keys = pm.governed_keys(mode)
        assert set(keys) == {"bl_pbl_physics", "moist_mix6_off"}
        for key in keys:
            with pytest.raises(ValueError, match="already the author"):
                _build(extra=f'physics_mode = "{mode}"',
                       shared_extra=f"{key} = "
                                    f"{'1' if key == 'bl_pbl_physics' else 'true'}")


def test_a_qualifier_without_the_axis_is_refused_rather_than_ignored():
    """``patchset``/``patches`` do not select the axis and must not imply it.

    Silently implying ``arwen-patched`` from a qualifier would make the most
    consequential line in a battery config the one that is not written.
    """

    with pytest.raises(ValueError, match="without physics_mode"):
        _build(extra='patchset = "v1"')
    with pytest.raises(ValueError, match="without physics_mode"):
        _build(extra='patches = ["L4"]')


def test_patches_under_the_faithful_mode_is_refused():
    with pytest.raises(ValueError, match="applies no patch"):
        _build(extra='physics_mode = "wrf-faithful"\npatches = ["L4"]')


def test_an_unregistered_mode_is_refused_by_name():
    with pytest.raises(ValueError, match="not a registered fidelity mode"):
        _build(extra='physics_mode = "arwen-improved"')


# ------------------------------------------------- 3. the frozen ledger

def test_patchset_v1_is_exactly_the_two_toggleable_entries():
    assert pm.PATCHSETS["v1"].entries == ("L3", "L4")
    for entry in pm.PATCHSETS["v1"].ledger_entries():
        assert entry.toggleable is True
        assert entry.key


def test_an_unregistered_patchset_version_is_refused():
    with pytest.raises(ValueError, match="not a registered patch-set"):
        pm.resolve("arwen-patched", "v2", None, source="probe.toml")


def test_sase_cannot_enter_a_vector_until_its_entry_gate_passes():
    """L5's gate is registered BEFORE any run exists, which is the point.

    A conditional arm that can be selected before its gate is not a gate.
    """

    with pytest.raises(ValueError) as excinfo:
        pm.resolve("arwen-patched", "v1", ["L5"], source="probe.toml")
    message = str(excinfo.value)
    assert "not a toggleable patch" in message
    assert "entry gate" in message.lower()
    assert pm.LEDGER_BY_ID["L5"].toggleable is False
    assert "L5" not in pm.PATCHSETS["v1"].entries


@pytest.mark.parametrize("entry_id", ["L1", "L2", "L6", "L7"])
def test_every_non_toggleable_ledger_entry_states_why(entry_id):
    entry = pm.LEDGER_BY_ID[entry_id]
    assert entry.toggleable is False
    assert entry.gate.strip(), entry_id
    with pytest.raises(ValueError, match="not a toggleable patch"):
        pm.resolve("arwen-patched", "v1", [entry_id], source="probe.toml")


def test_a_ledger_entry_outside_the_named_version_is_refused(monkeypatch):
    """A frozen version is widened by registering a new one, never in place.

    No toggleable entry currently sits outside v1, so the refusal is probed
    against a narrower registered version rather than left unexercised until
    the day a second version makes it reachable in production.
    """

    monkeypatch.setitem(
        pm.PATCHSETS, "probe",
        pm.PatchSet(version="probe", entries=("L3",)))
    assert pm.resolve(
        "arwen-patched", "probe", ["L3"], source="probe.toml").settings == {
            "bl_pbl_physics": 11}
    with pytest.raises(ValueError, match="not a member of patch-set"):
        pm.resolve("arwen-patched", "probe", ["L4"], source="probe.toml")


def test_patches_refuses_repeats_and_unknown_ids():
    with pytest.raises(ValueError, match="twice"):
        pm.resolve("arwen-patched", "v1", ["L4", "L4"], source="probe.toml")
    with pytest.raises(ValueError, match="not a divergence-ledger entry"):
        pm.resolve("arwen-patched", "v1", ["L99"], source="probe.toml")


def test_every_ledger_key_is_a_real_runconfig_field():
    """A ledger entry naming a key nothing reads is a patch that does nothing."""

    names = {f.name for f in dataclasses.fields(RunConfig)}
    for entry in pm.LEDGER:
        if entry.key:
            assert entry.key in names, entry.entry_id


def test_the_resolved_vector_records_both_ends_of_every_edge():
    """A receipt that records only the value taken cannot be audited later.

    The summary carries the faithful -> patched edge in words and the entry
    id carries it in the register, so a scoreboard row is readable without
    the tree it was produced from.
    """

    resolution = pm.resolve(
        "arwen-patched", "v1", ["L4"], source="probe.toml")
    receipt = resolution.receipt()
    vector = {row["entry"]: row for row in receipt["resolved_patch_vector"]}
    assert set(vector) == {"L3", "L4"}
    assert vector["L3"]["patched"] is False and vector["L3"]["value"] == 1
    assert vector["L4"]["patched"] is True and vector["L4"]["value"] is True
    assert receipt["requested_patches"] == ["L4"]
    assert all(row["summary"] for row in receipt["resolved_patch_vector"])


# ------------------------------- 4. what L4's key actually turns off

def test_moist_mix6_off_exempts_the_wrf_moist_array_and_nothing_else():
    """The scope of L4, pinned against WRF's own per-array switch set.

    WRF has FIVE independent 6th-order filter switches, one per Registry
    array (moist/chem/tracer/scalar/tke, Registry.EM_COMMON:2889-2893).  A
    gate that took woof's whole transported set for "moisture" would turn
    off a filter WRF leaves on -- silently, on the number and volume
    tracers, in the one arm whose purpose is to measure the moist filter.
    This is a pure function, so the scope is checked without a device; the
    trajectory it produces is
    tests/test_diff6.py::test_step_moist_mix6_off_spares_moisture_and_keeps_damping_theta.
    """

    from woof.core.dycore import diff6_exempt_slots

    base = dict(nx=8, ny=8, nz=6, dx=1000.0, dy=1000.0, ztop=6000.0,
                dt=0.05, run_seconds=0.0)
    assert diff6_exempt_slots(RunConfig(**base)) == frozenset()

    exempt = diff6_exempt_slots(RunConfig(**base, moist_mix6_off=True))
    # Every WRF moist-array species woof can carry, and only those.
    # qi2/qi3 are P3's second and third ice-category masses, declared in
    # WRF's moist array like qi (woof.core.moist.WRF_MOIST_ARRAY_SPECIES,
    # d765178a427).
    assert exempt == {"smag_rqv", "smag_rqc", "smag_rqr",
                      "smag_rqi", "smag_rqs", "smag_rqg", "smag_rqh",
                      "smag_rqi2", "smag_rqi3"}
    # Theta, the three momentum rows and TKE keep their filter: theta has
    # no switch at all, and TKE has tke_mix6_off.
    assert not exempt & {"smag_rth", "smag_ru", "smag_rv", "smag_rw",
                         "smag_rtke"}
    # WRF scalar-package tracers keep theirs (scalar_mix6_off, :2892).
    from woof.core.moist import (NSSL_SCALAR_SPECIES,
                                  THOMPSON_AERO_NUMBER_SPECIES,
                                  TRANSPORTED_NUMBER_SPECIES)
    for name in (set(TRANSPORTED_NUMBER_SPECIES)
                 | set(THOMPSON_AERO_NUMBER_SPECIES)
                 | set(NSSL_SCALAR_SPECIES)):
        assert "smag_r" + name not in exempt, name


def test_both_diff6_row_builders_consult_the_same_exemption():
    """Production and the verification helper must exempt the same rows.

    ``apply_diff6`` is the helper the kernel-normalisation tests measure
    with.  If it kept filtering a row production skips it would keep
    certifying an operator the model no longer applies, which is the exact
    class of drift the helper exists to catch in the other direction.
    """

    import inspect

    from woof.core import dycore

    for func in (dycore.prepare_fixed_tendencies, dycore.apply_diff6):
        assert "diff6_exempt_slots" in inspect.getsource(func), func.__name__
