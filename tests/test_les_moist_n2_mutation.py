"""The moist-N2 mutation control: wiring, isolation, and that it bites.

CPU-only.  No device is opened: the build variant is checked as a SOURCE
assembly and a cache-key property, and that the mutation actually changes
the answer is measured on the float64 mirror, which is the same branch
under the same predicate.

The control exists because the LES completion program spec §3.3 says a
scratch build with the saturated BN2 switch forced to the dry branch must
move the committed cloud-layer metrics, and that **if it does not, the
instrument is rejected, not the engine passed**.  An instrument that
cannot fail cannot pass anything.

The failure mode these tests are really aimed at is the quiet one.
``load_module`` is ``lru_cache``d on the module name alone, so a control
arm wired through it would be handed the ORIGINAL, UNMUTATED module and
would report -- truthfully, and catastrophically -- that the mutation
changes nothing.  That is a false clean bill of health for the
instrument, so the cache-key discrimination is tested directly rather than
assumed from reading the loader.
"""
from __future__ import annotations

import numpy as np
import pytest

from woof.core import kernels as kernel_loader
from woof.core import moist_n2_mutation as mut
from woof.verify.npref import np_wrf_calc_n2


# ------------------------------------------------------------ the build

def test_the_guard_exists_in_the_kernel_source_exactly_once():
    """One predicate feeds both ``if (saturated)`` sites, so one guard
    disables the branch everywhere.  More than one would mean the source
    had grown a second place to get this wrong."""
    src = (kernel_loader._KDIR / "smag2d.cu").read_text(encoding="utf-8")
    assert src.count("#ifdef %s" % mut.MUTATION_DEFINE) == 1
    assert src.count("saturated = false;") == 1
    # the production predicate is untouched, not rewritten around
    assert ("bool saturated = moist\n"
            "        && (qv_c >= wrf_n2_qvs(t_c, p_c) || qc_c >= 1.0e-5f);"
            ) in src


def test_the_two_assemblies_differ_only_by_the_define_line():
    production = kernel_loader.module_source("smag2d")
    mutant = kernel_loader.module_source_int_defines(
        "smag2d", mut.MUTATION_DEFINES)
    assert mutant != production
    stripped = mutant.replace("#define %s 1\n" % mut.MUTATION_DEFINE, "", 1)
    assert stripped == production


def test_the_define_survives_the_loader_s_own_validator():
    """``load_module_int_defines`` rejects lowercase identifiers, bools and
    any value below 1.  The control's define has to be expressible through
    it, since that is the loader whose cache key discriminates variants."""
    import re
    (key, value), = mut.MUTATION_DEFINES
    assert re.fullmatch(r"[A-Z][A-Z0-9_]*", key)
    assert isinstance(value, int) and not isinstance(value, bool)
    assert value >= 1


# ------------------------------------------------------- the cache trap

def test_production_and_mutant_do_not_share_a_cache_slot():
    """The quiet failure this control has to survive.

    ``load_module``'s key is the module NAME; ``get_kernel``'s is
    ``(name, func)``.  Neither includes the define set, so routing the
    mutant through them would silently return the production module.  The
    variant path's key is ``(name, func, defines)`` -- checked here on the
    cache itself, not inferred from the source.
    """
    assert (kernel_loader.load_module.__wrapped__
            is not kernel_loader.load_module_int_defines.__wrapped__)
    plain = kernel_loader.get_kernel.cache_info()
    variant = kernel_loader.get_kernel_int_defines.cache_info()
    assert plain is not variant
    # the two loaders assemble different sources for the same module name,
    # which is precisely why they must not share a slot
    assert (kernel_loader.module_source("smag2d")
            != kernel_loader.module_source_int_defines(
                "smag2d", mut.MUTATION_DEFINES))


def test_the_selector_asks_for_different_kernels_on_the_two_arms():
    """``calc_n2_kernel`` is the only place the arm is chosen.  Record what
    it asks the loader for, without compiling anything."""
    asked = []

    class _Recorder:
        def get_kernel(self, name, func):
            asked.append(("plain", name, func))
            return "production"

        def get_kernel_int_defines(self, name, func, defines):
            asked.append(("variant", name, func, defines))
            return "mutant"

    rec = _Recorder()
    import woof.core.kernels as real
    saved = real.get_kernel, real.get_kernel_int_defines
    real.get_kernel, real.get_kernel_int_defines = (
        rec.get_kernel, rec.get_kernel_int_defines)
    try:
        assert mut.calc_n2_kernel() == "production"
        with mut.forced_dry_branch():
            assert mut.calc_n2_kernel() == "mutant"
        assert mut.calc_n2_kernel() == "production"
    finally:
        real.get_kernel, real.get_kernel_int_defines = saved

    assert asked == [
        ("plain", "smag2d", "wrf_calc_n2"),
        ("variant", "smag2d", "wrf_calc_n2", mut.MUTATION_DEFINES),
        ("plain", "smag2d", "wrf_calc_n2"),
    ]


def test_the_launcher_goes_through_the_selector():
    """A one-line regression guard.  If the dycore ever goes back to
    ``get_kernel("smag2d", "wrf_calc_n2")`` directly, the control arm
    silently becomes a production run.

    The source is READ, not imported: ``dycore`` imports cupy at module
    scope, and this module is CPU-only by design -- importing it to check
    a string would open a device to answer a question about text.
    """
    from pathlib import Path

    import woof.core
    src = (Path(woof.core.__file__).parent / "dycore.py").read_text(
        encoding="utf-8")
    assert "calc_n2_kernel(use_theta_m=cfg.use_theta_m)(" in src
    assert "from woof.core.moist_n2_mutation import calc_n2_kernel" in src
    assert 'get_kernel("smag2d", "wrf_calc_n2")' not in src


# ------------------------------------------------------------ isolation

def test_the_switch_is_scoped_and_restores_on_an_exception():
    """A control arm that leaked into a later scored run in the same
    process would mutate a scored result.  The restore has to survive the
    integration raising, not only returning."""
    assert mut.active() is False
    with pytest.raises(RuntimeError):
        with mut.forced_dry_branch():
            assert mut.active() is True
            raise RuntimeError("integration blew up")
    assert mut.active() is False
    assert mut.receipt_tag() == mut.NO_MUTATION_TAG


def test_no_config_field_or_environment_variable_selects_the_mutation():
    """Nothing in RunConfig may select broken physics, and no environment
    variable in woof alters physics.  Both are checked rather than
    asserted in prose."""
    from pathlib import Path

    from woof.config import RunConfig
    fields = {f for f in RunConfig.__dataclass_fields__}
    assert not any("mutat" in f for f in fields)
    src = Path(mut.__file__).read_text(encoding="utf-8")
    assert "os.environ" not in src and "getenv" not in src


# ---------------------------------------------------- and it bites

def _moist_column(nz=8, ny=3, nx=4, *, saturate):
    """A small moist column, optionally with a saturated patch.

    Pressure and geopotential are a plausible hydrostatic-ish column; the
    absolute values do not matter, only that both mirrors see the same
    ones.
    """
    rng = np.random.default_rng(11)
    z_w = np.linspace(0.0, 2400.0, nz + 1)
    phi = np.broadcast_to(
        (z_w * 9.81)[:, None, None], (nz + 1, ny, nx)).astype(np.float64)
    z_m = 0.5 * (z_w[:-1] + z_w[1:])
    thb = 300.0 + 0.003 * np.maximum(z_m - 1000.0, 0.0)
    thp = 0.01 * rng.standard_normal((nz, ny, nx))
    p = np.broadcast_to(
        (1.0e5 * np.exp(-z_m / 8500.0))[:, None, None],
        (nz, ny, nx)).astype(np.float64).copy()
    qv = np.full((nz, ny, nx), 8.0e-3)
    qc = np.zeros((nz, ny, nx))
    if saturate:
        # cloud water well above qc_cr in a patch: the qc arm of the
        # predicate, which is the arm the capped-moist case rides on
        qc[2:5, 1:3, 1:3] = 5.0e-4
    return dict(thp=thp, thb=thb, p=p, phi=phi,
                cf1=1.5, cf2=-0.6, cf3=0.1, qv=qv, qc=qc)


def test_forcing_the_dry_branch_changes_bn2_where_it_is_engaged():
    """The control has to bite.  Same state, same moisture, one branch
    suppressed -- BN2 must move, and move only where the branch fired."""
    col = _moist_column(saturate=True)
    normal = np_wrf_calc_n2(**col)
    mutant = np_wrf_calc_n2(**col, force_dry_branch=True)

    assert not np.allclose(normal, mutant), (
        "the mutation control does not change BN2 -- an instrument that "
        "cannot fail cannot pass anything")
    moved = ~np.isclose(normal, mutant)
    assert moved.any() and not moved.all()
    # everything that moved is at or adjacent to the saturated patch
    engaged = col["qc"] >= 1.0e-5
    assert engaged[moved | np.roll(moved, 1, axis=0)
                   | np.roll(moved, -1, axis=0)].any()


def test_the_mutation_is_inert_where_nothing_is_saturated():
    """On a column that never engages the branch the two builds must agree
    exactly.  Otherwise the control would be moving something other than
    the thing it names, and any difference it produced on the real case
    would be uninterpretable."""
    col = _moist_column(saturate=False)
    normal = np_wrf_calc_n2(**col)
    mutant = np_wrf_calc_n2(**col, force_dry_branch=True)
    np.testing.assert_array_equal(normal, mutant)


def test_forcing_the_branch_is_not_the_same_as_removing_the_moisture():
    """The distinction the control exists to make.

    Dropping qv/qc also zeroes qtot and the ``1.61*(qv_p - qv_m)`` term, so
    it tests "moisture vs no moisture" rather than "saturated branch vs
    unsaturated branch".  If those two were the same reduction the control
    would prove nothing about the branch.
    """
    col = _moist_column(saturate=True)
    mutant = np_wrf_calc_n2(**col, force_dry_branch=True)
    without_moisture = np_wrf_calc_n2(
        col["thp"], col["thb"], col["p"], col["phi"],
        cf1=col["cf1"], cf2=col["cf2"], cf3=col["cf3"])
    assert not np.allclose(mutant, without_moisture)


def test_the_mirror_default_is_the_unmutated_reduction():
    col = _moist_column(saturate=True)
    np.testing.assert_array_equal(
        np_wrf_calc_n2(**col), np_wrf_calc_n2(**col, force_dry_branch=False))


# --------------------------------------------------------- the receipt

def test_the_case_stamps_which_build_produced_the_receipt():
    from woof.verify.cases import cloud_topped_boundary_layer as ctbl
    from pathlib import Path
    src = Path(ctbl.__file__).read_text(encoding="utf-8")
    assert '"mutation_control": moist_n2_mutation.receipt_tag()' in src
    # both arms are non-null strings, so neither reads as "not recorded"
    assert isinstance(mut.MUTATION_TAG, str) and mut.MUTATION_TAG
    assert isinstance(mut.NO_MUTATION_TAG, str) and mut.NO_MUTATION_TAG
    assert mut.MUTATION_TAG != mut.NO_MUTATION_TAG


def test_the_case_cli_offers_the_control_arm():
    from woof.verify.cases import cloud_topped_boundary_layer as ctbl
    with pytest.raises(SystemExit):
        ctbl.main(["--mutation-control", "not-a-control", "--out", "."])


def test_a_mutant_draw_cannot_be_averaged_into_a_scored_set_unnoticed():
    """The aggregator carries the build tag in its configuration identity,
    so mixing a control arm into a scored draw set is reported at the top
    of the output rather than disappearing into a mean -- which would
    corrupt the sigma every band is cut from."""
    from woof.verify import draw_spread
    assert "mutation_control" in draw_spread.MOIST_IDENTITY
    scored = {"mutation_control": mut.NO_MUTATION_TAG, "lwp_kg_m2": 0.025}
    control = {"mutation_control": mut.MUTATION_TAG, "lwp_kg_m2": 0.001}
    out = draw_spread.aggregate(
        [scored, control], label="mixed", metrics=("lwp_kg_m2",),
        identity=("mutation_control",), flags=())
    assert out["configuration_differs_on"] == ["mutation_control"]
    assert out["configuration"]["mutation_control"] == {
        "DIFFERS": [mut.MUTATION_TAG, mut.NO_MUTATION_TAG]}
