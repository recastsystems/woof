"""The dry-CBL case: registry contract, and both closures configurable.

CPU-only.  Integrating the case is a GPU job covered by the closure's own
suites and by the run receipts; what is worth a fast test is the part
that silently rots -- the case-registry contract, the gate convention,
and whether a second closure can still be selected on it at all.
"""
from __future__ import annotations

import pytest

from woof.config import SASE_PBL_SCHEME, validate_run_config
from woof.verify.cases import cbl_dry


def test_the_case_registry_discovers_it_with_both_capabilities():
    from woof.verify import cases

    manifest = {entry["name"]: entry for entry in cases.manifest()}
    assert "cbl_dry" in manifest, sorted(manifest)
    capabilities = set(manifest["cbl_dry"]["capabilities"])
    assert {"verify", "script"} <= capabilities, capabilities


def test_nan_is_reported_but_never_a_gate_row():
    """The driver checks ``nan`` first and by itself.

    Listing it in GATES too would gate a zero-valued metric with a zero
    bound, and the driver's intervals are OPEN -- so the case would fail
    precisely when nothing was wrong.  That is not hypothetical: it is
    what this case did before the convention was matched.
    """
    assert "nan" not in cbl_dry.GATES
    for name, (lo, hi) in cbl_dry.GATES.items():
        assert lo is None or hi is None or lo < hi, name


@pytest.mark.parametrize("selector", [1, 5, 11, SASE_PBL_SCHEME])
def test_every_pbl_closure_this_engine_has_can_drive_the_case(selector):
    """The case's whole purpose is comparison, so a closure that cannot
    be configured on it is a case defect, not a closure defect."""
    cfg = cbl_dry.default_config(selector)
    assert validate_run_config(cfg) is cfg
    assert cfg.bl_pbl_physics == selector


def test_the_experimental_closure_gets_km_opt_zero_and_the_others_do_not():
    """The one configuration difference between the arms, and its reason.

    SASE supplies the mixing the km_opt operator would otherwise apply.
    If this case handed it km_opt=4 as well, the comparison would be
    between one closure and another closure PLUS a Smagorinsky operator
    -- and the difference would be attributed to the closure.
    """
    assert cbl_dry.default_config(SASE_PBL_SCHEME).km_opt == 0
    assert cbl_dry.default_config(1).km_opt == 4
    assert cbl_dry.default_config(5).km_opt == 4
    # Shin-Hong keeps km_opt=4 like the other WRF-transcribed closures:
    # PBL vertical transport + horizontal Smagorinsky is the
    # configuration class Shin & Hong (2015) ran (the registered
    # gray-zone expectation note), so the SASE-only predicate holding
    # for 11 is a declared reading, not a fall-through accident.
    assert cbl_dry.default_config(11).km_opt == 4


def test_the_gates_admit_a_range_of_closures_not_one_profile():
    """A validity case must not encode a preferred answer.

    Each bound is checked to be loose enough that two closures differing
    by a plausible margin both sit inside it -- the mixed-layer depth
    band spans better than a factor of three, and the warming band better
    than two orders of magnitude.  A gate tight enough to separate two
    reasonable closures would make this case an unratified referee.
    """
    lo, hi = cbl_dry.GATES["mixed_layer_depth"]
    assert hi / lo >= 3.0
    lo, hi = cbl_dry.GATES["theta_surface_warming"]
    assert hi / lo >= 100.0
    lo, hi = cbl_dry.GATES["surface_heat_consistency"]
    assert lo < 1.0 < hi, "order unity must sit strictly inside the band"


# ---------------------------------------------------------------------------
# D1: the gray-zone partition sweep instrument (campaign spec
# docs/internal/specs/2026-08-02-sase-grayzone-campaign.md, I1).
# CPU-only: what is tested is the scoring arithmetic and the declared
# constants -- integrating the sweep is the campaign's own GPU job and
# its receipts are the evidence for that half.
# ---------------------------------------------------------------------------


def test_sweep_ladder_and_cadence_divide_into_whole_steps():
    """Every rung's dt divides the 300 s scoring cadence, the 3 h
    spin-up and the 4 h run exactly -- a fractional chunk would silently
    shift the scored window, which is the kind of rot a receipt cannot
    show."""
    assert tuple(sorted(cbl_dry.SWEEP_DT)) == tuple(sorted(cbl_dry.SWEEP_DX))
    spin = cbl_dry.SWEEP_SECONDS - cbl_dry.SWEEP_SCORE_SECONDS
    for dx, dt in cbl_dry.SWEEP_DT.items():
        for interval in (cbl_dry.SWEEP_SAMPLE_SECONDS, spin,
                         cbl_dry.SWEEP_SECONDS):
            steps = interval / dt
            assert steps == round(steps), (dx, dt, interval)
        # never looser than the case's own registered operating point
        # (3 s at 500 m with time_step_sound = 4)
        assert dt / dx <= 3.0 / 500.0 + 1e-12, (dx, dt)


def test_sweep_seeds_are_six_distinct_and_anchor_the_registered_case():
    assert len(cbl_dry.SWEEP_SEEDS) == 6
    assert len(set(cbl_dry.SWEEP_SEEDS)) == 6
    assert cbl_dry.SWEEP_SEEDS[0] == 20260801  # the case's own seed


def test_scored_windows_are_the_published_validity_ranges():
    """The mixed-layer window is eq. (9)'s published validity range and
    the advisory entrainment window is eq. (10)'s; they must meet at
    0.85 z/h with no gap and no overlap."""
    assert cbl_dry.MIXED_LAYER_WINDOW == (0.05, 0.85)
    assert cbl_dry.ENTRAINMENT_WINDOW == (0.85, 1.10)


def test_window_mean_scores_the_declared_levels_and_nothing_else():
    """40 levels at 50 m spacing, h = 875 m: the mixed-layer window
    [0.05, 0.85] z/h is [43.75, 743.75] m, catching centers 75..725 --
    levels 1 through 14 inclusive -- and the hand mean of those levels
    is what comes back."""
    import numpy as np

    z = (np.arange(40) + 0.5) * 50.0
    v = np.arange(40, dtype=np.float64)
    got = cbl_dry.window_mean(v, z, 875.0, cbl_dry.MIXED_LAYER_WINDOW)
    assert got == np.mean(np.arange(1, 15, dtype=np.float64))
    # a window catching no level scores NaN, never a number
    assert np.isnan(cbl_dry.window_mean(v, z, 10.0,
                                        cbl_dry.MIXED_LAYER_WINDOW))


def test_band_rule_is_the_declared_arithmetic():
    """[env_lo - 2*sigma/sqrt(n), env_hi + 2*sigma/sqrt(n)], verbatim."""
    import numpy as np

    lo, hi = cbl_dry.sweep_band((0.9161, 0.9846), 0.006, n=6)
    half = 2.0 * 0.006 / np.sqrt(6.0)
    assert lo == 0.9161 - half and hi == 0.9846 + half
    # sigma 0 collapses the band to the published envelope itself
    assert cbl_dry.sweep_band((0.2, 0.4), 0.0) == (0.2, 0.4)


def test_sweep_config_is_the_case_at_sweep_width():
    """Same closure rule, same sounding contract, 64 x 64, dx = dy, and
    only declared rungs are accepted."""
    from woof.config import SASE_PBL_SCHEME

    for dx in cbl_dry.SWEEP_DX:
        cfg = cbl_dry.sweep_config(dx, SASE_PBL_SCHEME)
        assert cfg.nx == cfg.ny == cbl_dry.SWEEP_COLUMNS
        assert cfg.dx == cfg.dy == dx
        assert cfg.dt == cbl_dry.SWEEP_DT[dx]
        assert cfg.run_seconds == cbl_dry.SWEEP_SECONDS
        assert cfg.km_opt == 0          # SASE supplies its own mixing
        assert cfg.case == "cbl_dry"
    assert cbl_dry.sweep_config(400.0, 1).km_opt == 4
    # The registered Shin-Hong reading: sweep_config(dx, 11) yields
    # km_opt=4 (vertical transport from the scheme, horizontal
    # Smagorinsky -- the SH15 configuration class).
    assert cbl_dry.sweep_config(400.0, 11).km_opt == 4
    with pytest.raises(ValueError):
        cbl_dry.sweep_config(555.0, SASE_PBL_SCHEME)


def test_sweep_targets_reproduce_the_spec_planning_table():
    """The spec's I1 planning row: eq. (9) at the declared planning
    abscissae (the HEAD baseline's own per-rung x, h ~ 875 m).  Pins the
    instrument to the registered campaign numbers so a drifted
    transcription cannot re-baseline silently."""
    import numpy as np

    from woof.verify import gray_zone as GZ

    for x, target in ((3.55, 0.9815), (1.83, 0.9503), (0.91, 0.8647),
                      (0.46, 0.6881), (0.23, 0.4449), (0.114, 0.2496)):
        np.testing.assert_allclose(
            float(GZ.subgrid_tke_fraction(x)), target, atol=5e-4)
