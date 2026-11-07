"""CPU-hermetic contract tests for the MPAS column-batch physics seam.

WHAT THIS FILE GATES
--------------------
The parts of ``woof/core/mpas_column_batch.py`` that are provable
without a device: the published API name and its identity with the
implementation, the leaf-wrapper refusal (the seam's phase entries ARE
the ARW orchestration function objects), the cadence law the seam owns,
and the whole constructor refusal surface -- every refusal here fires
BEFORE the first device allocation, by design.

The behavioural half (read-only inputs, held radiation, buckets,
restart bit-identity, in-place WSM6) lives in
``tests/test_mpas_column_batch_gpu.py`` and runs on a rented card.

REVERT-CHECK NOTES.  The cadence tests below pin the exact WRF due
calendars against the resolved step counts: reverting the seam to a
private scheduler (or WRF's predicates to anything else) goes red.  The
identity tests pin the orchestration objects: reverting ``run_phase1``
to thin per-kernel wrappers cannot keep
``_PHASE1_ORCHESTRATION is PhysicsDriver.compute`` true while changing
the call path, because the GPU suite asserts the driver's own counters
and held state advance through that exact entry.
"""

import datetime

import numpy as np
import pytest

from woof.core import mpas_column_batch as mcb

#: The MPAS counterparty's pinned configuration (2026-08-10): model step
#: 120 s, radiation 600 s, surface/PBL 120 s, cumulus 120 s.
PINNED = dict(dt=120.0, radiation_seconds=600.0,
              surface_pbl_seconds=120.0, cumulus_seconds=120.0)


def _constructor_kwargs(**overrides):
    """A fully valid constructor argument set (validation-only tests)."""
    kwargs = dict(
        n_levels=20, n_columns=8, dt=120.0,
        radiation_seconds=600.0, surface_pbl_seconds=120.0,
        cumulus_seconds=120.0, cumulus_scheme="kf",
        start_time=datetime.datetime(2021, 6, 1, 18, 0),
        latitude_deg=np.full(8, 35.0), longitude_deg=np.full(8, -97.0),
        terrain_height_m=np.zeros(8),
        z_interface_nominal_m=np.linspace(0.0, 16000.0, 21),
        p_top_pa=5000.0, dx_m=15000.0)
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# API identity and the leaf-wrapper refusal.
# ---------------------------------------------------------------------------

def test_the_published_api_name_is_the_implementation():
    import woof.core.physics as physics
    assert physics.run_mpas_column_batch is mcb.run_mpas_column_batch
    assert "run_mpas_column_batch" in physics.__all__


def test_phase1_is_the_arw_orchestration_not_a_wrapper():
    """Leaf-wrapper refusal: the phase-1 call site IS PhysicsDriver.compute.

    ``run_phase1`` invokes ``type(self)._PHASE1_ORCHESTRATION``; pinning
    the class attribute to the exact function object the ARW step calls
    (woof/core/dycore.py:2325) means a rewrite of the seam into
    per-kernel wrappers must either change this attribute (red here) or
    keep routing through the driver's own orchestration (not a wrapper).
    """
    from woof.core.physics import PhysicsDriver
    assert mcb.MpasColumnBatchPhysics._PHASE1_ORCHESTRATION \
        is PhysicsDriver.compute


def test_phase2_is_the_arw_microphysics_dispatch():
    from woof.core import microphysics
    assert mcb.MpasColumnBatchPhysics._PHASE2_MICROPHYSICS \
        is microphysics.apply


def test_phase2_carries_wrf_diagflag_for_refl_10cm():
    # THE BREAKAGE THIS PREVENTS: the MPAS port's history stream carried no
    # refl10cm, which left the obs referee's three MRMS reflectivity metrics
    # unscorable.  The seam must accept WRF's history-step diagflag so the
    # due step's OWN microphysics call computes REFL_10CM (prepared pressure
    # + post-call temperature); dropping the kwarg silently re-severs the
    # model's reflectivity from every downstream referee.
    import inspect
    parameters = inspect.signature(
        mcb.MpasColumnBatchPhysics.run_phase2).parameters
    assert "refl_10cm_due" in parameters
    assert parameters["refl_10cm_due"].default is False


def test_the_prepare_atmosphere_supplied_seam_exists():
    """The grid-adaptation seam is a read of the state attribute.

    ``_prepare_atmosphere`` must consult ``prepared_physics_atmosphere``
    before any ARW-grid arithmetic; reverting that hook strands the
    column batch on the C-grid destagger and this goes red.
    """
    import inspect

    from woof.core import physics
    source = inspect.getsource(physics._prepare_atmosphere)
    assert "prepared_physics_atmosphere" in source


# ---------------------------------------------------------------------------
# Cadence law (the seam owns the sub-cadences).
# ---------------------------------------------------------------------------

def test_the_pinned_counterparty_cadences_resolve_exactly():
    resolved = mcb.resolve_column_batch_cadences(
        cumulus_scheme="gf", **PINNED)
    assert resolved["stepra"] == 5
    assert resolved["stepbl"] == 1
    assert resolved["stepcu"] == 1
    assert resolved["radt_minutes"] == 10.0
    assert resolved["bldt_minutes"] == 2.0
    # GF carries WRF's pinned cudt=0 spelling (woof/config.py law).
    assert resolved["cudt_minutes"] == 0.0
    assert resolved["cu_physics"] == 3


def test_radiation_follows_wrf_stepra_calendar():
    """Due steps 1, 6, 11, ... for the pinned 600 s / 120 s pair.

    Uses WRF's own predicate from woof.core.physics, so this goes red
    if either the resolution or the calendar drifts.
    """
    from woof.core.physics import _radiation_step_due
    resolved = mcb.resolve_column_batch_cadences(
        cumulus_scheme="gf", **PINNED)
    due = [_radiation_step_due(step, resolved["stepra"],
                               resolved["radt_minutes"])
           for step in range(1, 11)]
    assert due == [True, False, False, False, False,
                   True, False, False, False, False]


def test_surface_pbl_and_cumulus_run_every_step_at_dt_cadence():
    from woof.core.physics import (_cumulus_step_due,
                                    _surface_pbl_step_due)
    resolved = mcb.resolve_column_batch_cadences(
        cumulus_scheme="gf", **PINNED)
    for step in range(1, 11):
        assert _surface_pbl_step_due(step, resolved["stepbl"],
                                     resolved["bldt_minutes"])
        assert _cumulus_step_due(step, resolved["stepcu"],
                                 resolved["cudt_minutes"])


def test_kf_takes_a_slower_commensurate_cumulus_cadence():
    from woof.core.physics import _cumulus_step_due
    resolved = mcb.resolve_column_batch_cadences(
        dt=120.0, radiation_seconds=600.0, surface_pbl_seconds=120.0,
        cumulus_seconds=600.0, cumulus_scheme="kf")
    assert resolved["stepcu"] == 5
    assert resolved["cudt_minutes"] == 10.0
    due = [_cumulus_step_due(step, resolved["stepcu"],
                             resolved["cudt_minutes"])
           for step in range(1, 11)]
    # Mandatory step-1 call, then MOD(ITIMESTEP, STEPCU) == 0.
    assert due == [True, False, False, False, True,
                   False, False, False, False, True]


def test_positive_bldt_keeps_the_raw_ysu_retention_on():
    """The seam's du/dv come from the driver's retained raw YSU output.

    ``physics_retains_ysu_output`` is bldt-gated; the seam always maps a
    positive surface/PBL cadence onto a positive bldt, so retention can
    never silently turn off.  Red if either side of that pact moves.
    """
    from woof.config import RunConfig
    from woof.core.physics import physics_retains_ysu_output
    resolved = mcb.resolve_column_batch_cadences(
        cumulus_scheme="gf", **PINNED)
    assert resolved["bldt_minutes"] > 0.0
    cfg = RunConfig(nx=8, ny=1, nz=20, dx=15000.0, dy=15000.0,
                    ztop=16000.0, dt=120.0, run_seconds=0.0,
                    bl_pbl_physics=1, bldt=resolved["bldt_minutes"])
    assert physics_retains_ysu_output(cfg)


@pytest.mark.parametrize('seconds', [130.0])
def test_incommensurate_or_nonpositive_cadences_refuse(seconds):
    with pytest.raises(ValueError):
        mcb.resolve_column_batch_cadences(
            dt=120.0, radiation_seconds=seconds,
            surface_pbl_seconds=120.0, cumulus_seconds=None,
            cumulus_scheme=None)


def test_gf_refuses_any_cadence_but_the_model_step():
    with pytest.raises(ValueError, match="cudt=0"):
        mcb.resolve_column_batch_cadences(
            dt=120.0, radiation_seconds=600.0,
            surface_pbl_seconds=120.0, cumulus_seconds=240.0,
            cumulus_scheme="gf")


def test_cumulus_seconds_without_a_scheme_refuses():
    with pytest.raises(ValueError, match="cumulus_scheme"):
        mcb.resolve_column_batch_cadences(
            dt=120.0, radiation_seconds=600.0,
            surface_pbl_seconds=120.0, cumulus_seconds=120.0,
            cumulus_scheme=None)


def test_unknown_cumulus_scheme_refuses():
    with pytest.raises(ValueError, match="'grell'"):
        mcb.resolve_column_batch_cadences(
            dt=120.0, radiation_seconds=600.0,
            surface_pbl_seconds=120.0, cumulus_seconds=120.0,
            cumulus_scheme="grell")


# ---------------------------------------------------------------------------
# Constructor refusal surface (all pre-device).
# ---------------------------------------------------------------------------

def test_a_misspelled_constructor_keyword_refuses_with_its_name():
    with pytest.raises(TypeError, match="n_levles"):
        mcb.run_mpas_column_batch(
            **{**_constructor_kwargs(), "n_levles": 40})


@pytest.mark.parametrize('overrides, match', [
    pytest.param(dict(n_levels=2), 'n_levels', id='overrides0-n_levels'),
    pytest.param(dict(n_columns=0), 'n_columns', id='overrides1-n_columns'),
    pytest.param(dict(dx_m=0.0), 'dx_m', id='overrides4-dx_m'),
    pytest.param(dict(wsm6_hail_opt=3), 'wsm6_hail_opt', id='overrides5-wsm6_hail_opt'),
    pytest.param(dict(latitude_deg=np.zeros(3)), 'latitude_deg', id='overrides8-latitude_deg'),
    pytest.param(dict(terrain_height_m=np.zeros(3)), 'terrain_height_m', id='overrides9-terrain_height_m'),
    pytest.param(dict(z_interface_nominal_m=np.linspace(0.0, 1.0, 11)), 'z_interface_nominal_m', id='overrides10-z_interface_nominal_m'),
])
def test_invalid_constructor_values_refuse_before_device_work(
        overrides, match):
    with pytest.raises((ValueError, TypeError), match=match):
        mcb.run_mpas_column_batch(**_constructor_kwargs(**overrides))


def test_descending_nominal_interfaces_refuse():
    with pytest.raises(ValueError, match="increase strictly upward"):
        mcb.run_mpas_column_batch(**_constructor_kwargs(
            z_interface_nominal_m=np.linspace(16000.0, 0.0, 21)))


def test_start_time_must_be_a_datetime():
    with pytest.raises(TypeError, match="datetime"):
        mcb.run_mpas_column_batch(**_constructor_kwargs(
            start_time="2021-06-01T18:00"))


# ---------------------------------------------------------------------------
# Setup arithmetic and restart schema.
# ---------------------------------------------------------------------------

def test_uniform_nominal_spacing_gives_midpoint_interface_weights():
    fnm, fnp = mcb._interface_weights(np.linspace(0.0, 2000.0, 11))
    assert fnm.dtype == np.float32 and fnp.dtype == np.float32
    assert fnm[0] == 0.0 and fnp[0] == 0.0
    np.testing.assert_array_equal(fnm[1:], np.float32(0.5))
    np.testing.assert_array_equal(fnp[1:], np.float32(0.5))


def test_interface_weights_sum_to_one_on_stretched_meshes():
    z = np.concatenate([[0.0], np.cumsum(50.0 * 1.12 ** np.arange(30))])
    fnm, fnp = mcb._interface_weights(z)
    np.testing.assert_allclose(
        (fnm + fnp)[1:], 1.0, rtol=0.0, atol=2.0e-7)


def test_restart_schema_and_lazy_keys_are_pinned():
    """The export payload layout is a cross-program contract.

    The MPAS side stores these payloads; a silent schema or key change
    would restore garbage.  Any deliberate change bumps RESTART_SCHEMA.
    """
    # v2: scalars["carriers"] (radiation CarrierContract provenance)
    # joined the payload -- the RESTART_SCHEMA comment carries the story.
    assert mcb.RESTART_SCHEMA == "mpas-column-batch-v2"
    assert mcb._LAZY_RESTART_KEYS == ("cumulus/w0avg",
                                      "radiation/o33d_grid")
    assert mcb._OUTPUT_BUFFERS == ("du", "dv", "dtheta", "dqv", "dqc",
                                   "dqr", "dqi", "dqs", "dqg",
                                   "h_diabatic")


def test_the_export_payload_carries_the_carrier_provenance():
    """Schema v2 coverage: a payload without CarrierContract provenance
    restores a seam that refuses at its first radiation-not-due step
    ("GLW has no producer", check_before_consumption) -- the red of
    tests/test_mpas_column_batch_gpu.py::
    test_restart_round_trip_continues_bit_identically, latent in shipped
    2.6.0 because no release list ran that file.  CPU restatement by
    source inspection, so the coverage cannot silently regress without a
    card noticing."""
    import inspect
    source = inspect.getsource(mcb.MpasColumnBatchPhysics.export_state)
    assert "carriers.state()" in source
    restore = inspect.getsource(mcb.MpasColumnBatchPhysics.restore_state)
    assert "carriers.restore" in restore


# ---------------------------------------------------------------------------
# Microphysics scheme rows (P3, 2026-08-31).
# ---------------------------------------------------------------------------

def test_the_species_rows_are_pinned_per_scheme():
    """The transported species sets are cross-program contracts.

    The WSM6 row is the pinned 2026-08-10 counterparty set, unchanged to
    the byte by the P3 landing (the off-path gate).  The P3 row is WRF's
    own mp=50 call shape (module_microphysics_driver.F:1569-1602): four
    moist masses, both number moments, the rime mass/volume pair --
    EIGHT scalars, no qs, no qg.  A drifted row is a state substitution
    on whichever side trusted the old one.
    """
    assert mcb._SPECIES_BY_SCHEME == {
        "wsm6": ("qv", "qc", "qr", "qi", "qs", "qg"),
        "p3": ("qv", "qc", "qr", "qi", "ni", "nr", "qir", "qib"),
        # mp=51 (P3 with predicted droplet number): the mp=50 call shape
        # plus QNCLOUD (package p3_1category_nc, Registry.EM_COMMON:3039).
        "p3_nc": ("qv", "qc", "qr", "qi", "nc", "ni", "nr", "qir", "qib"),
        "thompson_aero": ("qv", "qc", "qr", "qi", "qs", "qg",
                          "ni", "nr", "nc", "nwfa", "nifa"),
        # TEMPO (NCAR/TEMPO 9adf9ef): Thompson-aero's set plus graupel
        # number and volume, thirteen transported scalars.
        "tempo": ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc",
                  "nwfa", "nifa", "ng", "volg"),
    }
    assert mcb._SPECIES == mcb._SPECIES_BY_SCHEME["wsm6"]
    assert mcb._MP_PHYSICS_BY_SCHEME == {
        "wsm6": 6, "p3": 50, "p3_nc": 51, "thompson_aero": 28, "tempo": 900}
    assert set(mcb._MP_PHYSICS_BY_SCHEME) == set(mcb._SPECIES_BY_SCHEME)


def test_the_mp28_row_matches_the_hex_scalar_requirement_row():
    """The mp=28 species row is a CROSS-PROGRAM contract.

    hexcore's ``_required_scalar_names`` requires exactly
    ``qv/qc/qr/qi/qs/qg`` + ``nr/ni`` + ``nc/nwfa/nifa`` for
    mp_physics=28 (MPAS-A allocates and round-trips the same set,
    mpas_atmphys_interface.F:653-702).  If the two sides disagree the
    hex route either cannot construct the seam or constructs it on a
    species set the scheme does not write.  DomainState's own mp=28
    tuple is the third copy and is checked here too: the seam must not
    transport a field the engine state has no slot for.
    """
    from woof.core.moist import THOMPSON_AERO_NUMBER_SPECIES
    row = set(mcb._SPECIES_BY_SCHEME["thompson_aero"])
    assert row == {"qv", "qc", "qr", "qi", "qs", "qg",
                   "nr", "ni", "nc", "nwfa", "nifa"}
    assert set(THOMPSON_AERO_NUMBER_SPECIES) < row
    # No black carbon anywhere: WRF's qnbca arrives only with
    # wif_input_opt=2 and no ArWen state allocates it.
    assert "nbca" not in row and "qnbca" not in row


def test_the_restart_state_rows_are_pinned_per_scheme():
    """P3 restart state adds th_old/qv_old and drops effs.

    th_old/qv_old are P3's cross-step supersaturation carriers
    (module_mp_p3.F:5018-5021); nothing between a resume and the next
    microphysics call can refill them, so a row that lost them would
    resume with a repeated first-step transient and no error anywhere.
    effs is absent because P3 has no snow species or snow radius (WRF's
    has_reqs=0 override).  The WSM6 row is unchanged.
    """
    assert mcb._STATE_RESTART_FIELDS_BY_SCHEME == {
        "wsm6": ("effc", "effi", "effs", "h_diabatic"),
        "p3": ("effc", "effi", "h_diabatic", "th_old", "qv_old"),
        # mp=51 shares P3's radii and supersaturation carriers; its droplet
        # number rides the transported species row, not this one.
        "p3_nc": ("effc", "effi", "h_diabatic", "th_old", "qv_old"),
        "thompson_aero": ("effc", "effi", "effs", "h_diabatic",
                          "nwfa2d", "nifa2d"),
        "tempo": ("effc", "effi", "effs", "h_diabatic", "nwfa2d", "nifa2d"),
    }


def test_the_mp28_restart_row_carries_the_surface_emission_pair():
    """nwfa2d/nifa2d are the mp=28 analogue of P3's th_old/qv_old.

    Both are INTENT(IN) to every microphysics call (module_mp_thompson.F
    :1310-1327): thompson_init derives nwfa2d once at construction and
    nothing in mp_gt_driver ever refills either.  A row that lost them
    would resume with the surface aerosol source silently at zero and
    no error anywhere -- the aerosol physics would be inert in exactly
    the way microphysics.microphysics_init's docstring names as the
    scheme's most dangerous failure mode.
    """
    row = mcb._STATE_RESTART_FIELDS_BY_SCHEME["thompson_aero"]
    assert ("nwfa2d", "nifa2d") == row[-2:]
    # mp=28 keeps WSM6's radius TRIPLE: mp_gt_driver seeds and clamps
    # re_qc/re_qi/re_qs from the same RE_*_BG parameters.
    assert "effs" in row
    assert "effs" not in mcb._STATE_RESTART_FIELDS_BY_SCHEME["p3"]


def test_the_graupel_bearing_rows_are_pinned():
    """Which rows report graupelncv/GRAUPELNC, and which cannot.

    mp=28 has a graupel category and its driver arm binds
    GRAUPELNC/GRAUPELNCV; P3 has one ice category and binds neither
    (woof/core/physics_inventory.py's mp=50 surface-slot row), so a
    permanent zero key would let a consumer claim a field P3 never has.
    """
    assert mcb._GRAUPEL_SCHEMES == frozenset({"wsm6", "thompson_aero", "tempo"})
    assert set(mcb._RHO_DRY_REFUSAL_BY_SCHEME) == (
        set(mcb._SPECIES_BY_SCHEME) - {"wsm6"}), (
            "every non-WSM6 row must name why it refuses rho_dry")


def test_an_unknown_microphysics_scheme_refuses_with_the_roster():
    with pytest.raises(ValueError, match="microphysics_scheme"):
        mcb.run_mpas_column_batch(**_constructor_kwargs(
            microphysics_scheme="thompson"))


def test_thompson_aero_refuses_the_wsm6_hail_knob():
    """The knob is WSM6's; on an mp=28 seam it would be ignored."""
    with pytest.raises(ValueError, match="wsm6_hail_opt"):
        mcb.run_mpas_column_batch(**_constructor_kwargs(
            microphysics_scheme="thompson_aero", wsm6_hail_opt=1))


def test_p3_refuses_the_wsm6_hail_knob():
    """A WSM6 knob on a P3 seam would be silently ignored -- accepted
    configuration the run does not have, which is the refusal law's
    exact target."""
    with pytest.raises(ValueError, match="wsm6_hail_opt"):
        mcb.run_mpas_column_batch(**_constructor_kwargs(
            microphysics_scheme="p3", wsm6_hail_opt=1))


def test_phase_signatures_carry_the_p3_species_keywords():
    """Both phase entries accept the P3 four as keywords (scheme-gated
    at runtime); phase 2's rho_dry is optional because P3 refuses it.
    Dropping a keyword silently re-welds the seam to WSM6."""
    import inspect
    for method in (mcb.MpasColumnBatchPhysics.run_phase1,
                   mcb.MpasColumnBatchPhysics.run_phase2):
        parameters = inspect.signature(method).parameters
        for name in ("ni", "nr", "qir", "qib", "nc", "nwfa", "nifa"):
            assert name in parameters
            assert parameters[name].default is None
    phase2 = inspect.signature(
        mcb.MpasColumnBatchPhysics.run_phase2).parameters
    assert phase2["rho_dry"].default is None


def test_mp28_first_contact_lives_in_phase2_and_rides_the_payload():
    """thompson_init on the caller's arrays can only run where the seam
    holds them writable: phase 2.  Phase 1 is read-only by law, so the
    first-contact hook must not appear there; and export/restore must
    carry the receipt, or a restored seam would re-run thompson_init's
    presence tests on a checkpointed aerosol field.
    """
    import inspect
    phase1 = inspect.getsource(mcb.MpasColumnBatchPhysics.run_phase1)
    phase2 = inspect.getsource(mcb.MpasColumnBatchPhysics.run_phase2)
    assert "_thompson_first_contact" not in phase1
    assert "_thompson_first_contact" in phase2
    export = inspect.getsource(mcb.MpasColumnBatchPhysics.export_state)
    restore = inspect.getsource(mcb.MpasColumnBatchPhysics.restore_state)
    assert "aerosol_init" in export and "aerosol_init" in restore
    assert isinstance(
        inspect.getattr_static(mcb.MpasColumnBatchPhysics,
                               "aerosol_init_receipt"), property)
