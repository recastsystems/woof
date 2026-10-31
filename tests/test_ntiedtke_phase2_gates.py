"""Gates that guard New Tiedtke's INTEGRATION, not its arithmetic.

The parity files grade the transcription. This one holds the checks that
have to be true before `cu_physics = 16` may be reachable at all, and each
exists because the alternative failure is silent.

It will grow: the cumulus-calendar refusal and the `cudt_minutes` law
(docs/ntiedtke/PORT-RECORD.md §10) belong here too when they land.
"""
from __future__ import annotations

from types import SimpleNamespace

import re
import pytest

from woof.core.preflight import (
    column_workspace_bytes, ntiedtke_column_workspace_bytes)


def _exp(*cu_physics):
    """The smallest thing the workspace terms actually read."""
    return SimpleNamespace(domains=[
        SimpleNamespace(run=SimpleNamespace(
            cu_physics=c, bl_pbl_physics=0, nx=372, ny=284, nz=49))
        for c in cu_physics])


def test_new_tiedtke_is_priced_rather_than_refused():
    """INVERTED 2026-08-29. The refusal did its job and is retired.

    It stood while the distinct-array count was underivable, and refusing
    was right then: a silent zero in this sum lets a run proceed
    under-budgeted with nothing to show for it. The count is measured now
    -- 89 level arrays, 37 surface, 11 scratch slabs -- so the term is a
    number, and this asserts it is a real one rather than the zero the
    refusal existed to prevent.
    """
    priced = ntiedtke_column_workspace_bytes(_exp(16))
    assert priced > 0, "cu_physics = 16 is priced at zero again"
    assert column_workspace_bytes(_exp(16)) >= priced, (
        "the New Tiedtke term does not reach the sum")


def test_the_term_reaches_the_sum_from_any_domain():
    """A nested tree must price an inner domain as readily as a root one."""
    alone = ntiedtke_column_workspace_bytes(_exp(16))
    for tree in (_exp(3, 16), _exp(16, 1)):
        assert ntiedtke_column_workspace_bytes(tree) == alone, (
            "a 16 domain beside another scheme prices differently from a "
            "16 domain alone; the term takes the worst domain, not the "
            "first")
        assert column_workspace_bytes(tree) >= alone


def test_the_price_comes_from_the_workspace_itself_not_a_second_formula():
    """One census, and it lives with the thing it counts.

    A formula apart from its allocator is what put 75 arrays where 89
    belonged (docs/ntiedtke/PORT-RECORD.md section 33), so preflight holds neither the
    cap nor the byte count -- both are imported.
    """
    import inspect
    src = inspect.getsource(ntiedtke_column_workspace_bytes)
    assert "nt_workspace_bytes" in src and "nt_tile_columns" in src, (
        "preflight now computes the New Tiedtke workspace itself; that is "
        "a second copy of a census")

    from woof.core.ntiedtke import (NtWorkspace, nt_tile_columns,
                                     nt_workspace_bytes)
    import pytest as _pytest
    _pytest.importorskip("cupy")
    columns = nt_tile_columns(4096, 70)
    assert (nt_workspace_bytes(20, columns)
            == NtWorkspace(ncol=columns, nz=20).storage_census()["bytes"]), (
        "the formula and the allocator disagree")


@pytest.mark.parametrize("cu", (0, 1, 3))
def test_every_currently_selectable_scheme_is_unaffected(cu):
    """The refusal must cost exactly nothing today.

    `CU_SCHEMES` is (0, 1, 3); none of them may newly raise, and the New
    Tiedtke term must contribute zero rather than perturbing a budget
    the owner's live campaign depends on.
    """
    assert ntiedtke_column_workspace_bytes(_exp(cu)) == 0
    column_workspace_bytes(_exp(cu))


def test_the_term_is_actually_summed_in():
    """A term nobody calls is a comment.

    `column_workspace_bytes` must reach the New Tiedtke arm, or the
    refusal above passes while the real gate stays open.
    """
    import inspect
    src = inspect.getsource(column_workspace_bytes)
    assert "ntiedtke_column_workspace_bytes" in src


def test_cu_schemes_admits_16_and_every_co_requisite_landed():
    """INVERTED. 16 is selectable, and the six things that had to land
    with it are each asserted here rather than remembered.

    The old test said: "when 16 IS added, this test fails -- and that
    failure is the reminder that the workspace term, the cumulus calendar
    and the cudt law all have to land in the same edit". It fired, and
    this is that list, checked.
    """
    pytest.importorskip("cupy")
    import inspect
    from woof.config import CU_SCHEMES, CUMULUS_ADVECTIVE_FORCING_SCHEMES
    from woof.core import clock
    from woof.core.preflight import _CUMULUS_KERNEL_MODULES

    assert 16 in CU_SCHEMES
    # 1. a real workspace term, not the refusal
    assert ntiedtke_column_workspace_bytes(_exp(16)) > 0
    # 2. the cumulus calendar, so the F14 three-way check runs at all
    assert 16 in clock._CU_SCHEMES
    assert "run.cu_physics in _CU_SCHEMES" in inspect.getsource(clock)
    # 3. the cudt law
    from woof import config as cfgmod
    assert "cu_physics == 16" in inspect.getsource(cfgmod)
    # 4. the advective forcing pair
    assert 16 in CUMULUS_ADVECTIVE_FORCING_SCHEMES
    # 5. the kernel-module table, which is fail-closed
    assert _CUMULUS_KERNEL_MODULES.get(16) == ("ntiedtke",)
    # 6. the optional tendency components
    from woof.core import physics
    # WRF returns ice tendency only when the microphysics carries QI.
    # Check the output contract after G3 shares the early-return branch.
    assert physics._cumulus_optional_tendency_components(
        SimpleNamespace(cu_physics=16, mp_physics=1)) == ()
    assert physics._cumulus_optional_tendency_components(
        SimpleNamespace(cu_physics=16, mp_physics=6)) == ("rqi",)


def test_the_momentum_gate_is_the_corrected_one():
    """The Phase 3 gate must demand a PRE-EXTENSION baseline comparison.

    The original wording -- two runs of the same build, byte-identical,
    cmp all 33 -- proves DETERMINISM, not INERTNESS, and a deterministic
    shift in GF's answer passes it perfectly. That is the one outcome the
    constraint exists to prevent.

    This is a gate on the document because the document is what a future
    session follows, and the port has now twice had a fact sit unread in a
    file that was not checked.
    """
    from pathlib import Path
    rules = (Path(__file__).resolve().parents[1]
             / "docs/ntiedtke/STANDING-RULES.md").read_text(encoding="utf-8")
    for phrase in ("INERTNESS", "PRE-EXTENSION", "GF *and* KF"):
        assert phrase in rules, (
            f"the corrected momentum gate lost {phrase!r}. Two runs of one "
            "build prove determinism; inertness needs a pre-extension "
            "baseline.")


# ---------------------------------------------------------------------------
# The FIFTH mechanical Phase 2 gap: the advective forcing pair and the fold
# ---------------------------------------------------------------------------
# Found by review (review) off the section 17 measurement, and confirmed
# against woof/config.py:831-853 -- which names NTiedtke explicitly, and was
# written before anyone was porting it.
#
# At cu_physics = 16 outside CUMULUS_ADVECTIVE_FORCING_SCHEMES, physics.py
# feeds ZEROS for the theta/qv forcing pair.  Section 17 measured ptent/ptenq
# non-zero on 4,428 of 5,292 rows, and the closure and cuadjtqn read them --
# so the convection itself would change.  Finite, plausible, wrong.


def test_new_tiedtke_consumes_the_advective_forcing_pair():
    """INVERTED, and the fold's location is the assertion that matters.

    WRF's cumulus driver pre-folds RTHRATEN + RTHBLTEN into RTHFTEN at
    module_cumulus_driver.F:867 for G3SCHEME and NTIEDTKESCHEME ONLY. GF
    is not in that list and sums the three lanes itself, so WOOF's dycore
    exports PURE ADVECTION. The fold belongs in the New Tiedtke ADAPTER --
    moving the export double-counts radiative and PBL heating for GF and
    moves run_myj, the baseline every intensity number is graded against.
    """
    from woof.config import CUMULUS_ADVECTIVE_FORCING_SCHEMES as S
    assert 16 in S
    assert 3 in S, "Grell-Freitas fell out while New Tiedtke was added"


def test_the_fold_trap_note_still_stands_beside_the_table():
    """The note is a receipt, and it is the strongest a receipt can be.

    It sits beside the table it constrains, names the two schemes it
    applies to, states the wrong repair and why it is wrong, and it found
    this port rather than the other way round. It still cannot fail -- so
    this test is the minimum that can: the note may not be deleted by
    whoever adds the entry, which is exactly when it matters most.
    """
    import inspect
    import woof.config as cfg
    src = inspect.getsource(cfg)
    i = src.index("CUMULUS_ADVECTIVE_FORCING_SCHEMES = ")
    raw = src[max(0, i - 2000):i]
    # Strip the "#:" sphinx comment markers before collapsing, or a
    # phrase that wraps across two comment lines is never found -- and
    # the test then passes for the wrong reason on the short phrases.
    note = " ".join(raw.replace("#:", " ").split())
    for phrase in ("THE FOLD TRAP", "NTIEDTKESCHEME ONLY",
                   "exports PURE ADVECTION",
                   "rather than moving the export"):
        assert phrase in note, f"the fold-trap note lost {phrase!r}"


def test_grell_freitas_is_still_the_only_member():
    """GF must not be removed from the set while adding New Tiedtke.

    The set is read by the dycore export, the state allocation, the VRAM
    projection, the restart inventory and the serialization contract. A
    change that swapped rather than added would silently stop allocating
    GF's pair.
    """
    from woof.config import CUMULUS_ADVECTIVE_FORCING_SCHEMES as S
    assert 3 in S, "Grell-Freitas fell out of the advective forcing set"


def test_phase_one_is_defined_by_the_assembly_not_the_kernels():
    """"Thirteen of thirteen kernels graded" is not Phase 1 done.

    Every routine grading green against captures the orchestration did
    not produce is the same circularity the capture architecture was
    built to retire, one level up. The definition is cheap to state now
    and awkward to argue later, so it is pinned in the contract and
    checked here.
    """
    from pathlib import Path
    rules = (Path(__file__).resolve().parents[1]
             / "docs/ntiedtke/STANDING-RULES.md").read_text(encoding="utf-8")
    for phrase in ("PHASE 1 ENDS WHEN THE ASSEMBLED PIPELINE",
                   "EVERY capture boundary", "bracket the unowned"):
        assert phrase in rules, (
            f"the Phase 1 definition lost {phrase!r}. Grading every kernel "
            "is not the same claim as grading the pipeline that runs them.")


def test_the_cumulus_kernel_module_table_carries_16():
    """The fail-closed table: a selector with no entry RAISES.

    That is why this was the one item of the seven that could not be
    forgotten -- it announces itself. Asserted anyway, because "it would
    have raised" is not the same as "it is right".
    """
    from woof.core.preflight import _CUMULUS_KERNEL_MODULES
    assert _CUMULUS_KERNEL_MODULES[16] == ("ntiedtke",)
    assert _CUMULUS_KERNEL_MODULES[3] == ("gf",), "GF's entry moved"


def test_phase_one_remaining_work_is_stated_in_full():
    """Three times the remaining-work count measured a smaller thing.

    "Thirteen of thirteen kernels graded" omitted the assembler. The
    assembler omitted cu_ntiedtke_post_run. Each count was taken over the
    artifact in front of me rather than over the artifact the end
    condition names -- and the end condition names ``nt-levels.csv``,
    whose contents ``cu_ntiedtke_run`` does not produce.

    SUPERSEDED IN PART by test_ntiedtke_output_provenance.py, which takes
    the count over the artifact the end condition NAMES -- nt-levels.csv's
    header -- rather than over a hand-maintained list of categories. A
    hand-maintained list is exactly what failed three times, so this test
    is kept only as the record's index and the header-driven one is the
    gate (review).
    """
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1]
           / "docs/ntiedtke/PORT-RECORD.md").read_text(encoding="utf-8")
    for item in ("cu_ntiedtke_post_run", "the allocator", ":566",
                 "reduction"):
        assert item in doc, (
            f"Phase 1's remaining-work list lost {item!r}. It is not done "
            "until the assembled pipeline reproduces nt-levels.csv bitwise.")


def test_the_llo3_reduction_scope_is_decided_not_left_open():
    """32 columns and 17,920 are different reductions, not an impl detail.

    llo3 is monotone, so it flips earlier the wider the population is. A
    32-column block of clear ocean air keeps it FALSE and puts the whole
    level-loop body -- including the departure-level reset that runs for
    every column regardless of loflag -- into the branch the fixture
    never exercises. The fixture only ever runs llo3 true (48 of 48
    triggering columns).
    """
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1]
           / "docs/ntiedtke/PORT-RECORD.md").read_text(encoding="utf-8")
    assert "DECIDED: chunk-wide" in doc, (
        "the llo3 reduction scope is no longer recorded as decided. It is "
        "a decision, not an implementation choice -- see section 27.")


# ---------------------------------------------------------------------------
# The forcing function forces NOTICING, not COMPLETENESS
# ---------------------------------------------------------------------------
# test_cu_schemes_still_refuses_16 fails the instant someone adds 16, so the
# tuple cannot land silently. But trace what happens next: they add 16, they
# update that test, and NOTHING then requires the cumulus calendar to exist.
# clock.py leaves stepcu None, PhysicsDriver computes its own from a
# cudt_minutes that defaults to 5.0, and the scheme runs on an unvalidated
# five-minute hold. That is the original finding arriving intact through the
# gate built to prevent it (review).
#
# So each remaining item asserts its CURRENT ABSENT STATE, the same shape as
# test_the_cumulus_kernel_module_table_still_lacks_16. Each fails the moment
# the edit begins, and together they make "one edit" mean one edit.
#
# Writing them while the answer is absent is also easier than afterwards:
# the assertion is just "the current state", and it does not require knowing
# yet what the correct new state looks like.


def test_the_cumulus_calendar_includes_16():
    """Outside the tuple the scheme still runs -- on a cadence nothing
    validated. Inside it, the F14 three-way check runs for New Tiedtke.
    """
    import inspect
    from woof.core import clock
    assert 16 in clock._CU_SCHEMES
    assert "run.cu_physics in _CU_SCHEMES" in inspect.getsource(clock)


def test_the_cudt_law_for_16_exists_and_refuses_a_hold():
    """cudt_minutes DEFAULTS TO 5.0, and that is the whole point.

    A config that merely selects 16 and says nothing else would hold the
    scheme's rates for five minutes -- for a scheme whose RAINCV is
    rn/stepcu, a per-call rate with no persistence. The held rate would be
    reapplied every step and the precipitation would be five times what
    the scheme produced.

    THE NEGATIVE CONTROL IS THE HALF THAT MATTERS: a law that accepted
    everything would pass any test that only builds a valid config.
    """
    import pytest as _pytest
    from woof.config import RunConfig, validate_run_config

    def cfg(**kw):
        # The law lives in validate_run_config, not __post_init__, so a
        # test that only CONSTRUCTS a config asserts nothing -- which is
        # how the first version of this passed the valid case and failed
        # the negative one.
        base = dict(nx=8, ny=8, nz=10, dx=4500.0, dy=4500.0, ztop=20000.0,
                    dt=20.0, run_seconds=0.0, time_step_sound=4, moist=True,
                    mp_physics=10, ra_physics=4, cu_physics=16,
                    cudt_minutes=0.0, sf_sfclay_physics=1,
                    sf_surface_physics=2, bl_pbl_physics=1)
        base.update(kw)
        return validate_run_config(RunConfig(**base))

    cfg()                                     # the valid shape is accepted
    with _pytest.raises(ValueError, match="cudt_minutes=0"):
        cfg(cudt_minutes=5.0)
    # NEW TIEDTKE WITH THE PBL SLOT OFF IS ADMITTED.  This asserted a
    # refusal cloned from Grell-Freitas, whose reason is GF's own read of
    # fields["kpbl"] as a one-based index; New Tiedtke reads no kpbl
    # anywhere, its hfx/qfx come from the surface stack (which runs
    # independently of bl_pbl_physics) and its advective-forcing lanes are
    # the zero planes WRF's cumulus driver folds in when no PBL tendency
    # exists.  The registry agrees: new-tiedtke declares no pbl
    # requirement, and tests/test_authority_agreement.py holds the two
    # authorities together over the whole cross-product.
    cfg(bl_pbl_physics=0)


def test_the_optional_tendency_components_branch_for_16():
    """New Tiedtke produces no separate rain or snow category.

    cu_ntiedtke_post_run forms exactly six tendency fields
    (module_cu_ntiedtke.F:514-524): theta, qv, qc, qi, u and v. Applying
    Kain-Fritsch's phase logic at 16 would ask it for rqr, and on a
    separate-ice-snow package for rqs as well.
    """
    pytest.importorskip("cupy")
    from woof.config import RunConfig
    from woof.core.physics import _cumulus_optional_tendency_components

    def cfg(mp, cu=16):
        return RunConfig(
            nx=8, ny=8, nz=10, dx=4500.0, dy=4500.0, ztop=20000.0, dt=20.0,
            run_seconds=0.0, time_step_sound=4, moist=True, mp_physics=mp,
            ra_physics=4, cu_physics=cu,
            cudt_minutes=0.0 if cu in (3, 16) else 5.0,
            sf_sfclay_physics=1, sf_surface_physics=2, bl_pbl_physics=1)

    got = _cumulus_optional_tendency_components(cfg(10))
    assert "rqr" not in got and "rqs" not in got, got
    assert got in ((), ("rqi",)), got
    # And Kain-Fritsch is untouched -- it still gets its rain category.
    assert "rqr" in _cumulus_optional_tendency_components(cfg(10, cu=1))


def test_the_prepared_cache_identity_needs_no_scheme_entry():
    """Full domain serialization binds the scheme and every other run field."""
    from dataclasses import asdict, replace
    from datetime import datetime, timedelta
    from woof.config import CHEM_RUN_FIELDS, RunConfig
    from woof.experiment import DomainConfig
    from woof.ingest import prepared_cache

    run = RunConfig(nx=8, ny=8, nz=20, dx=4500., dy=4500., ztop=20000.,
        dt=20., run_seconds=0., moist=True, mp_physics=10, cu_physics=3,
        cudt_minutes=0., sf_sfclay_physics=1, sf_surface_physics=2,
        bl_pbl_physics=1)
    start = datetime(2026, 8, 29, 1)
    domain = DomainConfig(2, 1, 10, 10, 3, 3, 300., run, start_time=start)
    identity = prepared_cache.prepared_domain_config_identity
    compare = prepared_cache.compare_prepared_domain_config
    cached = identity(domain)
    expected_run = asdict(run)
    from woof.config import drop_inert_g3_fields
    drop_inert_g3_fields(expected_run)
    from woof.checkpoint_identity import drop_default_cumulus_selectors
    drop_default_cumulus_selectors(expected_run)
    assert run.chem_sets == ""
    for name in CHEM_RUN_FIELDS:
        expected_run.pop(name)
    # The fire block is omitted from an inactive domain's identity
    # (lane/ec-sfire), by the same rule the document applies.
    from woof.config import inert_fire_fields
    for name in inert_fire_fields(run):
        expected_run.pop(name)
    # The lightning and aer_opt = 2 controls (golden/ports-rad) leave an
    # identity that reads neither, by the rule every run document applies.
    from woof.config import drop_inactive_radiation_port_keys
    drop_inactive_radiation_port_keys(expected_run)
    # The 2.9.4 capabilities campaign (lane/cap-*) leaves the identity by
    # the checkpoint echo's rule (drop_capability_campaign_keys): the
    # HAILCAST selectors, the ocean door and the lake-state analysis at
    # their defaults.
    from woof.checkpoint_identity import drop_capability_campaign_keys
    drop_capability_campaign_keys(expected_run)
    # lane/hrrr-v4-output's controls leave it off, by the document's rule
    # (drop_default_output_controls).
    from woof.checkpoint_identity import drop_default_output_controls
    drop_default_output_controls(expected_run)
    # Fitch wind farms off (labs/windfarm, 2.9.11) leave it by the same rule
    # (woof.wind_table.drop_inert_config).
    from woof.wind_table import drop_inert_config
    drop_inert_config(expected_run)
    neutral = {"surface_energy_diag": False, "cycling": False,
               "ruc_irrigation": "wrf_461", "ruc_qvg_cold_start": "wrf",
               "ruc_2m_diagnostic": "flux", "ruc_snow": "wrf_461",
               "swint_opt": 0, "aer_opt": 0, "alb_sol": 0,
               "thompson_version": "wrf_461", "thompson_table_dir": None,
               "thompson_fork_snow_fall": "blend",
               "rrtmg_cloud_optics_form": "wrf_461", "rrtmg_smoke_manifest": "",
               "rrtmg_aer3_type_source": "rural", "bl_mynn_mixaerosols": 0,
               "bl_mynn_mixnumcon": 0, "tempo_ml_for_bl_nc": False,
               "tempo_surface_emission_before": False,
               # golden/wrf-fixes: the MYNN transport, Noah-MP snow-age and
               # urban generation/canopy controls leave the identity at
               # their defaults like the generations above.
               "scalar_adv_opt": 1, "noahmp_snow_age": "wrf_470",
               "urban_water_balance": "wrf_480",
               "urban_ucm_generation": "wrf_471", "urban_treeoption": -1,
               "urban_ttscheme": -1, "urban_fvg": -1.0,
               "urban_table_dir": "",
               # 2.9.1: the ocean mixed layer off leaves the identity.
               "sf_ocean_physics": 0, "oml_hml0": 50.0, "oml_gamma": 0.14,
               "oml_relaxation_time": 0.0,
               # 2.9.4: lane/cap-eclipse at its default-on shadow and
               # lane/cap-slab-ocean's PWP controls with the ocean off.
               "ra_sw_eclipse": 1, "ocean_levels": 30, "ocean_analysis": "",
               "ocean_analysis_source": "rtofs",
               "ocean_mld_climatology": "woa18", "omdt": 1.0,
               "ocean_z": (), "ocean_t": (), "ocean_s": ()}
    for name, default in neutral.items():
        assert expected_run.pop(name) == default
    # 2.9.7 tc/nest: the fixed-frame archive is read only on a nested domain;
    # this root leaves it out by the document's rule.
    assert expected_run.pop("nest_accumulation_archive") is True
    # 2.9.0 correct mode (woof/correct_mode.py): a key leaves the document
    # where no selected scheme reads it (and WB1's output-only key always),
    # as in the checkpoint echo; this domain runs Morrison and Grell-Freitas.
    from woof import correct_mode
    for name in correct_mode.CORRECTION_KEYS:
        expected_run.pop(name)
    # 2.9.7 HN1 (hydrometeor_handover_numbers) is read under Morrison, so
    # this domain's identity binds it while it is on.
    if correct_mode.CORRECT_KEYS_DEFAULT:
        expected_run["hydrometeor_handover_numbers"] = True
    assert cached["run"] == expected_run
    assert cached["start_time"] == start.isoformat()
    assert compare(cached, identity(domain)) == ([], [])

    new_scheme = identity(replace(domain, run=replace(run, cu_physics=16)))
    # Under New Tiedtke its WB4 key is read, so the default (on) is bound.
    wb4 = ({"ntiedtke_dry_mass_coupling": True}
           if correct_mode.CORRECT_KEYS_DEFAULT else {})
    assert new_scheme["run"] == {**expected_run, "cu_physics": 16, **wb4}
    assert compare(cached, new_scheme) == (
        [], ["run.cu_physics"] + (["run.ntiedtke_dry_mass_coupling"] if wb4 else []))
    # A selected-field or per-scheme serializer cannot drop unrelated
    # trajectory controls or the newly serialized delayed-domain date.
    changed = identity(replace(domain, run=replace(run, dt=10.),
        start_time=start + timedelta(hours=1)))
    assert compare(cached, changed) == ([], ["run.dt", "start_time"])


def test_adding_16_did_not_reparent_the_grell_key_check():
    """THE REGRESSION THE BASELINE RE-RUN CAUGHT AND NO UNIT TEST DID.

    ``validate_run_config`` had::

        if cfg.cu_physics == 3:
            ...
        elif cfg.clos_choice != 0 or cfg.ishallow != 0:
            raise ...

    and the first version of the New Tiedtke law was inserted BETWEEN
    them. That silently re-parented the ``elif`` onto the 16 test, so
    every ``cu_physics = 3`` config carrying ``ishallow = 1`` -- which is
    most of the campaign's -- was refused with a message about Grell-family
    keys being read only where cu_physics=3, on a config that selects
    exactly that.

    Every unit test passed. None exercised a GF config with a non-default
    Grell key through the validator, so the whole suite was blind to it
    and the forecast refused instead.

    The general form: inserting a branch into an ``if/elif`` chain changes
    what the later arms attach to, and nothing about the inserted branch
    looks wrong in isolation.
    """
    from woof.config import RunConfig, validate_run_config

    def cfg(**kw):
        base = dict(nx=8, ny=8, nz=20, dx=4500.0, dy=4500.0, ztop=20000.0,
                    dt=20.0, run_seconds=0.0, time_step_sound=4, moist=True,
                    mp_physics=10, ra_physics=4, sf_sfclay_physics=1,
                    sf_surface_physics=2, bl_pbl_physics=1)
        base.update(kw)
        return validate_run_config(RunConfig(**base))

    # The shape that was refused: Grell-Freitas with its shallow arm on.
    cfg(cu_physics=3, cudt_minutes=0.0, ishallow=1)
    cfg(cu_physics=3, cudt_minutes=0.0, ishallow=0)
    # And the check still fires where it should -- a Grell key on a
    # non-Grell scheme.
    with pytest.raises(ValueError, match="Grell-family keys"):
        cfg(cu_physics=1, cudt_minutes=5.0, ishallow=1)


# ---------------------------------------------------------------------------
# The tenth site, found by running rather than by reading
# ---------------------------------------------------------------------------


def test_no_scheme_whitelist_restates_the_scheme_set():
    """TWO whitelists, and only one of them was in the group of nine.

    ``validate_run_config`` gates the config; ``initialize_physics`` gates
    the driver, and it restated ``(0, 1, 3)`` as a literal rather than
    reading :data:`CU_SCHEMES`.  So a ``cu_physics = 16`` config was valid,
    priced, scheduled, compiled, frame-recorded -- and then refused at
    driver construction, after the whole preflight had passed.

    THIS IS THE SAME FAILURE MODE AS THE NINE, one layer deeper: a scheme
    set written down twice.  Nothing found it by reading, because the
    search that found the others looked for ``cu_physics == 3`` dispatch
    sites and this is a membership test.  Running the model found it in two
    seconds.

    The guard is structural rather than another list: no module may compare
    ``cu_physics`` against a literal tuple.  A negative control keeps it
    accurate -- the pattern must actually match the shape it forbids.
    """
    import pathlib
    import re

    forbidden = re.compile(r"cu_physics\s+(?:not\s+)?in\s+\((?![^)]*\b16\b)")
    # NEGATIVE CONTROL: the pattern must match the shape it exists to
    # forbid, or this test passes by matching nothing.
    assert forbidden.search("if cfg.cu_physics not in (0, 1, 3):"), (
        "the pattern does not match the very line this test was written "
        "for; it would pass against any tree")
    assert not forbidden.search("if run.cu_physics in (1, 3, 16):"), (
        "the pattern flags the cumulus calendar, which names 16 and is "
        "correct")

    # Not scheme whitelists: the strict WRF 4.6.1 chain's two cumulus-forcing
    # sites (paper/conus-suite-exact, 2.9.11), which name the schemes whose
    # stock driver consumes the advective qv forcing and the WW ring in the
    # measured CONUS chain (Grell-Freitas 3, Tiedtke 6); they run only with
    # GPUWM_WRF_EXACT, and admit or refuse no scheme.  Exact lines, so an
    # edit to either is read again.
    deliberate = {
        ("core/dycore.py", "if cfg.cu_physics in (3, 6):"),
        ("core/moist.py", 'if name == "qv" and WRF_EXACT and cfg.cu_physics in (3, 6):'),
        ("core/moist.py", 'if name == "qv" and (export_advective_forcing or (WRF_EXACT and cfg.cu_physics in (3, 6))):'),
    }
    root = pathlib.Path(__file__).resolve().parents[1] / "woof"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        for n, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            if forbidden.search(line):
                if (path.relative_to(root).as_posix(), line.strip()) in deliberate:
                    continue
                offenders.append(f"{path.relative_to(root)}:{n}: {line.strip()}")
    assert not offenders, (
        "a scheme set is restated as a literal that omits 16:\n  "
        + "\n  ".join(offenders))


def test_initialize_physics_admits_16_at_the_driver():
    """The site itself, asserted at the boundary that refused.

    Reading the source is not enough here: what broke was construction, so
    this constructs.  ``initialize_physics`` is heavy, so the assertion is
    that the cu_physics gate specifically is passed -- a refusal naming
    cu_physics is the failure this guards.
    """
    pytest.importorskip("cupy")
    import inspect

    from woof.config import CU_SCHEMES
    from woof.core.physics import initialize_physics

    src = inspect.getsource(initialize_physics)
    assert "cu_physics not in CU_SCHEMES" in src, (
        "initialize_physics no longer reads the config module's tuple")
    assert 16 in CU_SCHEMES


# ---------------------------------------------------------------------------
# The twelfth site, found by a 14-hour run and nothing shorter
# ---------------------------------------------------------------------------


def test_every_cumulus_scheme_has_a_restart_identity():
    """A checkpoint cannot be written for a scheme with no identity string.

    ``cu_physics = 16`` ran for six forecast hours and then died writing
    its first restart: ``cannot identify unsupported cumulus scheme 16``.
    Nothing shorter than a run that reaches the checkpoint cadence could
    have found it -- the 30-minute, 2-hour and 4-hour probes all completed
    without ever writing one.

    Two tables were missing it, and they fail differently.
    ``CUMULUS_ALGORITHM_IDENTITIES`` raises with the scheme id in the
    message.  The ``expected`` class map does not: an unlisted scheme
    leaves ``expected_class`` None, which routes a perfectly stock adapter
    down the custom-callable path and complains that it lacks a
    ``restart_identity`` attribute -- a true statement that points at the
    wrong thing.

    Both are checked here against ``CU_SCHEMES`` rather than against a
    list, so the next scheme fails this test instead of a long run.
    """
    import inspect

    from woof.config import CU_SCHEMES
    from woof.io.restart import (CUMULUS_ALGORITHM_IDENTITIES,
                                  physics_setup_identity)

    missing = [s for s in CU_SCHEMES if s not in CUMULUS_ALGORITHM_IDENTITIES]
    assert not missing, (
        f"cumulus schemes with no restart algorithm identity: {missing}. "
        "A run selecting one writes wrfout happily and then dies at its "
        "first checkpoint.")

    # Identities must be DISTINCT: two schemes sharing a string would let a
    # checkpoint resume under the wrong algorithm, which is the single
    # thing this table exists to prevent.
    strings = [CUMULUS_ALGORITHM_IDENTITIES[s] for s in CU_SCHEMES]
    assert len(set(strings)) == len(strings), (
        f"two cumulus schemes share a restart identity: {strings}")

    # And the second table: the stock adapter class per scheme, which used
    # to be a literal dict inside physics_setup_identity (read out of its
    # source here) and is now the registry's consumers.stock_callable_class
    # row, resolved through woof.physics_registry.stock_callable_class.  A
    # scheme with no row is refused there by name, not routed down the
    # custom-callable path.
    from woof.physics_registry import stock_callable_class
    assert "woof.core.kf.KainFritsch" not in inspect.getsource(
        physics_setup_identity), "the literal class dict is back"
    for scheme, cls in ((1, "woof.core.kf.KainFritsch"),
                        (3, "woof.core.gf.GrellFreitas"),
                        (16, "woof.core.ntiedtke.NewTiedtke")):
        assert stock_callable_class("cumulus", {"cu_physics": scheme}) == cls
    assert stock_callable_class("cumulus", {"cu_physics": 0}) is None
    with pytest.raises(KeyError):
        stock_callable_class("cumulus", {"cu_physics": 99})


def test_the_new_tiedtke_restart_identity_names_this_port():
    """The string binds the implementation, not just the scheme.

    Grell-Freitas' identity carries ``corrected-k22`` because that variant
    must not resume under a WRF-faithful build.  New Tiedtke's equivalent
    is the pratec divergence (docs/ntiedtke/PORT-RECORD.md section 38): every
    checkpoint under this identity has ``cu_pratec == 0`` by construction,
    and an implementation that delivered it would give the slot a
    different meaning.
    """
    from woof.io.restart import CUMULUS_ALGORITHM_IDENTITIES

    identity = CUMULUS_ALGORITHM_IDENTITIES[16]
    assert "tiedtke" in identity.lower()
    assert "wrf461" in identity, (
        "the identity does not name the reference version it was ported "
        "from")
    assert re.search(r"-v[0-9]+$", identity), (
        "the identity carries no revision suffix, so a change to the "
        "driver seam has no way to invalidate old checkpoints")


def test_a_reused_pipeline_runs_on_the_step_it_is_handed(monkeypatch):
    """The pipeline cache is keyed by chunk shape, and the step is not in it.

    An adaptive clock moves dt every root step while the chunk shape stays
    put, so a reused pipeline has to take each call's step, and take it
    exactly as a pipeline built on that step would.
    """
    import numpy as np

    from woof.core import ntiedtke

    class _Inert:
        def __init__(self, *args, **kwargs):
            pass

    for name in ("NtWorkspace", "NtLaunchGeometry", "NtStages"):
        monkeypatch.setattr(ntiedtke, name, _Inert)

    def cfg(dt):
        return SimpleNamespace(dt=dt, clock_dt=0.0,
                               ntiedtke_tiedtke_closure=False)

    scheme = ntiedtke.NewTiedtke()
    built = scheme._for(64, 10, cfg(12.0))
    for dt in (15.5, 9.25, 12.0):
        reused = scheme._for(64, 10, cfg(dt))
        assert reused is built
        fresh = ntiedtke.NtPipeline(ncol=64, nz=10, dt=dt, stepcu=1,
                                    itimestep=2)
        assert reused.scalars.keys() == fresh.scalars.keys()
        for name, value in fresh.scalars.items():
            assert type(reused.scalars[name]) is type(value), name
            assert reused.scalars[name] == value, name
        assert reused.scalars["dt"] == np.float32(dt)
        assert reused.scalars["delt"] == np.float32(dt)
