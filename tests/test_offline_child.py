from dataclasses import replace
from datetime import datetime, timedelta
import json

import netCDF4
import numpy as np
import pytest

from conftest import requires_gpu, requires_netcdf_bridge
from woof.offline_child import (
    OFFLINE_CHILD_MP_PHYSICS,
    PARENT_SCHEME_CONTRACT,
    _resolve_source_physics,
    bind_parent_physics_from_wrf_namelist,
    build_offline_child_domain_state,
    OfflineChildContractError,
    OfflineChildPlacement,
    build_offline_lateral_boundaries,
    inspect_parent_history_frame,
    interpolate_parent_boundary_snapshot,
    interpolate_parent_initial_state,
    offline_cross_scheme_refusal,
    read_parent_microphysics,
    validate_parent_history,
)
from woof.config import RunConfig
from woof.offline_child_run import (
    _ChildProgress,
    _create_output_root,
    _file_receipt,
    main as offline_child_main,
    _verify_file_receipts,
)

# Consumer expectations come from the independent generated declarations.
from woof.physics_registry import implemented_selector_values as _intake_selectors
from woof.physics_registry import consumer_rows_by_selector as _intake_rows
_INTAKE_MP = tuple(sorted(_intake_selectors("microphysics")))
_INTAKE_REFL = tuple(mp for mp in _INTAKE_MP if mp)
_INTAKE_EDGES = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "nest_transition").items()
    if (row or {}).get("mixed_edge_ported")))
_INTAKE_PRODUCERS = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "reflectivity_input_species").items()
    if row is not None))
_INTAKE_NATIVE = tuple(sorted(set(_INTAKE_REFL) - set(_INTAKE_PRODUCERS)))
_INTAKE_SNOW = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "ring_guard").items()
    if "qs" in (row or {}).get("state_fields", ())))
_INTAKE_SIX = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "ring_guard").items()
    if "qg" in (row or {}).get("state_fields", ())))


def _variable(dataset, name, dims, value):
    variable = dataset.createVariable(name, "f4", dims)
    variable[:] = np.asarray(value, dtype=np.float32)


def test_offline_child_output_root_is_create_only(tmp_path):
    requested = tmp_path / "new" / "child-run"
    assert _create_output_root(requested) == requested.resolve()
    marker = requested / "prior-evidence.txt"
    marker.write_text("preserve", encoding="utf-8")
    # The refusal is a sentence, not a Windows error number: the reader
    # who typed the same output directory twice is told what the
    # directory holds and the two ways out, and their evidence survives.
    with pytest.raises(OfflineChildContractError) as caught:
        _create_output_root(requested)
    assert "prior-evidence.txt" in str(caught.value)
    assert "--outdir" in str(caught.value)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_offline_child_parent_receipts_detect_midrun_input_change(tmp_path):
    parent = tmp_path / "parent.nc"
    parent.write_bytes(b"first")
    receipts = [_file_receipt(parent)]
    _verify_file_receipts(receipts, label="parent history input")
    parent.write_bytes(b"later")
    with pytest.raises(OfflineChildContractError, match="changed while"):
        _verify_file_receipts(receipts, label="parent history input")


def test_offline_child_capabilities_are_warning_only_and_exact(capsys):
    assert offline_child_main(["--show-capabilities"]) == 0
    capability = json.loads(capsys.readouterr().out)
    assert capability["schema"] == "gpuwm-offline-child-capabilities-v1"
    assert capability["explicit_expert_consent_required"] is False
    # Every ported scheme is read same-scheme; 16 (WDM6) joined last, when
    # the field map learned its scheme-qualified QNCCN row.  A child of a
    # different scheme is converted by the online nest edge's own contract
    # and kernel on the parent archive, for every ordered pair of the
    # schemes that edge ports; the one edge without a contract is mp=0.
    # 2.8.9 pinned [0, 1, 6, 8, 9, 10, 16, 18, 28, 50]; the ports add 2, 3,
    # 4, 24 and 106.  Ferrier-Aligo (5), FAST_SBM (30) and TEMPO (900) are
    # refused by name in their registry rows (rime factor, bin spectra,
    # graupel volume), so none of them is a same-scheme offline parent.
    assert capability["same_scheme_mp_physics"] == [
        0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 13, 14, 15, 16, 18, 24, 26, 28, 40, 50, 51, 53, 55, 56, 97, 106]
    assert capability["cross_scheme_transitions"]["mp_physics"] == list(_INTAKE_EDGES)
    assert capability["cross_scheme_transitions"]["order"] == (
        "diagnose-parent-then-spatially-interpolate")
    # Was pinned to False while the runner refused any child whose nz
    # differed from its parent's.  A child may now carry its OWN eta ladder
    # when it declares one, through the conservative host-side remap in
    # woof/vertical_remap.py, so the declaration says what it does instead
    # of denying it exists.  A child that declares no ladder still inherits
    # its parent's, bitwise.
    assert capability["vertical_remapping"] == "conservative-offline-prepare"
    assert capability["output_ownership"] == "create-only"


def _physics_binding(tmp_path, *, mp=8, morr_rimed_ice=1):
    path = tmp_path / f"namelist-mp{mp}.input"
    rimed = (f" morr_rimed_ice = {morr_rimed_ice},\n"
             if mp == 10 else "")
    path.write_text(
        f"&physics\n mp_physics = {mp},\n{rimed}/\n",
        encoding="utf-8")
    return bind_parent_physics_from_wrf_namelist(path)


def _history(path, valid_time, *, mp=8, hgt_offset=0.0, signal=0.0,
             ny=3, nx=4, omit_aerosol_surface_emission=False,
             qnrain=None, qir=None, producer="woof"):
    nz = 2
    with netCDF4.Dataset(path, "w") as dataset:
        for name, size in (
                ("Time", 1), ("DateStrLen", 19), ("west_east", nx),
                ("south_north", ny), ("bottom_top", nz),
                ("west_east_stag", nx + 1),
                ("south_north_stag", ny + 1),
                ("bottom_top_stag", nz + 1)):
            dataset.createDimension(name, size)
        dataset.TITLE = ("woof offline-child fixture" if producer == "woof"
                         else " OUTPUT FROM WRF V4.6.1 MODEL")
        dataset.DX = 1000.0
        dataset.DY = 1000.0
        dataset.MAP_PROJ = 1
        dataset.TRUELAT1 = 30.0
        dataset.TRUELAT2 = 60.0
        dataset.STAND_LON = -97.0
        dataset.CEN_LAT = 35.0
        dataset.CEN_LON = -97.0
        dataset.HYBRID_OPT = 2
        dataset.ETAC = 0.2
        if producer == "woof":
            dataset.GPUWM_WRITE_COMPLETE = 1
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0] = np.frombuffer(
            valid_time.strftime("%Y-%m-%d_%H:%M:%S").encode(), dtype="S1")
        mass3 = ("Time", "bottom_top", "south_north", "west_east")
        u3 = ("Time", "bottom_top", "south_north", "west_east_stag")
        v3 = ("Time", "bottom_top", "south_north_stag", "west_east")
        w3 = ("Time", "bottom_top_stag", "south_north", "west_east")
        mass2 = ("Time", "south_north", "west_east")
        # P3 (mp=50) declares NO qs and NO qg (Registry.EM_COMMON:3038:
        # ``moist:qv,qc,qr,qi``), so a faithful P3 archive omits both, and
        # its QICE is written nonzero below so the rime pair has ice to
        # describe.  Every other fixture keeps the six-species zeros.
        # mp 0 and 1 carry the warm-rain trio and no frozen species: WRF's
        # Kessler declares moist:qv,qc,qr (Registry.EM_COMMON:3015) and
        # woof's own mp=0 advects the same three
        # (woof/offline_child.py::_transported_source_fields), which is
        # exactly why the inventory cannot separate them on a woof tape.
        zero_masses = (("P", "QVAPOR", "QCLOUD", "QRAIN")
                       if mp in {0, 1, 50} else
                       ("P", "QVAPOR", "QCLOUD", "QRAIN",
                        "QICE", "QSNOW", "QGRAUP"))
        for name in zero_masses:
            _variable(dataset, name, mass3, np.zeros((1, nz, ny, nx)))
        pb = np.broadcast_to(
            np.asarray([65000.0, 30000.0], dtype=np.float32)[None, :, None, None],
            (1, nz, ny, nx))
        _variable(dataset, "PB", mass3, pb)
        def uniform(name, dimensions, value):
            shape = tuple(len(dataset.dimensions[d]) for d in dimensions)
            _variable(dataset, name, dimensions, np.full(shape, value))

        for name, dimensions, value in (
            ("T", mass3, signal), ("U", u3, 5.0 + signal),
            ("V", v3, -2.0 + signal), ("W", w3, 0.0), ("PH", w3, 0.0),
            ("MU", mass2, signal), ("MUB", mass2, 80000.0),
            ("MAPFAC_M", mass2, 1.0),
            ("MAPFAC_U", ("Time", "south_north", "west_east_stag"), 1.0),
            ("MAPFAC_V", ("Time", "south_north_stag", "west_east"), 1.0),
            ("HGT", mass2, hgt_offset), ("PSFC", mass2, 90000.0),
            ("F", mass2, 8.0e-5), ("E", mass2, 1.0e-4),
            ("SINALPHA", mass2, 0.0), ("COSALPHA", mass2, 1.0),
            ("XLAT", mass2, 35.0), ("XLONG", mass2, -97.0),
        ):
            uniform(name, dimensions, value)
        phb = np.broadcast_to(
            np.asarray([0.0, 40000.0, 90000.0], dtype=np.float32)
            [None, :, None, None], (1, nz + 1, ny, nx))
        _variable(dataset, "PHB", w3, phb)
        _variable(dataset, "P_TOP", ("Time",), [10000.0])
        _variable(dataset, "ZNU", ("Time", "bottom_top"), [[0.75, 0.25]])
        _variable(dataset, "ZNW", ("Time", "bottom_top_stag"), [[1.0, 0.5, 0.0]])
        if mp in {8, 10, 50}:
            uniform("QNRAIN", mass3, 0.0 if qnrain is None else qnrain)
            uniform("QNICE", mass3, 0.0)
        if mp == 10:
            for name in ("QNSNOW", "QNGRAUPEL"):
                uniform(name, mass3, 0.0)
        if mp == 28:
            # Distinct aerosol values expose crossed interpolation fields.
            for name, value in (("QNRAIN", 0.0), ("QNICE", 0.0),
                                ("QNCLOUD", 1.0e8), ("QNWFA", 1.5e8),
                                ("QNIFA", 2.5e5)):
                uniform(name, mass3, value)
            if not omit_aerosol_surface_emission:
                uniform("QNWFA2D", mass2, 4321.0)
                uniform("QNIFA2D", mass2, 0.0)
        if mp == 16:
            # WDM6 has no ice number, unlike the other number inventories.
            for name, value in (("QNCCN", 1.0e8), ("QNCLOUD", 1.0e8),
                                ("QNRAIN", 0.0)):
                uniform(name, mass3, value)
        if mp == 50:
            # P3 rime density is 500 kg/m3, inside its [50, 900] band.
            uniform("QICE", mass3, 3.0e-4)
            uniform("QIR", mass3, 1.0e-4 if qir is None else qir)
            uniform("QIB", mass3, 2.0e-7)


def test_parent_history_contract_checks_geometry_cadence_and_scheme(tmp_path):
    start = datetime(1974, 4, 3, 12)
    paths = [tmp_path / f"parent-{index}.nc" for index in range(2)]
    for index, path in enumerate(paths):
        _history(path, start + timedelta(minutes=5 * index))
    contract = validate_parent_history(
        paths, max_boundary_interval_seconds=900, source_mp_physics=8)
    assert contract.interval_seconds == 300.0
    assert contract.source_kind == "woof"
    assert contract.source_mp_physics == 8
    assert contract.start_time == start
    assert contract.end_time == start + timedelta(minutes=5)
    with pytest.raises(OfflineChildContractError, match="exceeds"):
        validate_parent_history(paths, max_boundary_interval_seconds=299)


def test_offline_child_refuses_feedback_modified_parent_provenance(tmp_path):
    path = tmp_path / "feedback-parent.nc"
    _history(path, datetime(1974, 4, 3, 12))
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.GPUWM_FEEDBACK = "experimental"
        dataset.GPUWM_FEEDBACK_VALUE = 1
    with pytest.raises(
            OfflineChildContractError,
            match="two-way feedback provenance.*one-way parent"):
        inspect_parent_history_frame(path)


def test_wrf_namelist_binding_is_authoritative_and_digested(tmp_path):
    binding = _physics_binding(tmp_path, mp=10, morr_rimed_ice=0)
    assert binding.mp_physics == 10
    assert binding.morr_rimed_ice == 0
    assert binding.domain_id == 1
    assert binding.evidence_kind == "wrf-namelist"
    assert len(binding.evidence_sha256) == 64


def test_parent_history_rejects_changed_static_geometry(tmp_path):
    start = datetime(1974, 4, 3, 12)
    first = tmp_path / "first.nc"
    second = tmp_path / "second.nc"
    _history(first, start)
    _history(second, start + timedelta(minutes=5), hgt_offset=1.0)
    with pytest.raises(OfflineChildContractError, match="geometry/static"):
        validate_parent_history(
            (first, second), max_boundary_interval_seconds=900)


def test_the_cross_scheme_contract_is_the_online_ported_set():
    """The offline conversion IS the online one, so its admitted set is
    derived from PORTED_MP_PHYSICS rather than re-spelled; mp=0 is the one
    scheme the lane reads that has no mixed-edge contract."""
    from woof.core import microphysics_transition as mt

    assert PARENT_SCHEME_CONTRACT == (
        OFFLINE_CHILD_MP_PHYSICS & frozenset(mt.PORTED_MP_PHYSICS))
    assert PARENT_SCHEME_CONTRACT == frozenset(_INTAKE_EDGES)
    for source in sorted(PARENT_SCHEME_CONTRACT):
        for target in sorted(PARENT_SCHEME_CONTRACT):
            assert offline_cross_scheme_refusal(source, target) is None
    # Same-scheme is never refused, mp=0 included.
    assert offline_cross_scheme_refusal(0, 0) is None
    # The ONE refusal: a microphysics-off end, named at either end.
    for source, target in ((0, 18), (10, 0), (0, 16)):
        message = offline_cross_scheme_refusal(source, target)
        assert message is not None and "REFUSED" in message
        assert "mp_physics=0" in message
        assert f"mp_physics={source} -> {target}" in message
        assert "no microphysics-transition contract" in message


def test_a_microphysics_off_mixed_edge_is_refused_at_both_sites(tmp_path):
    frame = tmp_path / "parent-mp0.nc"
    _history(frame, datetime(1974, 4, 3, 12), mp=0, ny=18, nx=20)
    placement = _mp28_placement()
    with pytest.raises(OfflineChildContractError, match="mp_physics=0"):
        interpolate_parent_initial_state(
            frame, placement, source_mp_physics=0, target_mp_physics=18,
            backend="cpu")
    with pytest.raises(OfflineChildContractError, match="mp_physics=0"):
        interpolate_parent_boundary_snapshot(
            frame, placement, source_mp_physics=0, target_mp_physics=18,
            backend="cpu")


def test_the_offline_edge_contract_is_the_online_contract():
    """resolve_microphysics_transition, with the parent's bound switches."""
    from woof.core import microphysics_transition as mt
    from woof.offline_child import _offline_transition_contract

    contract = _offline_transition_contract(
        10, 18, morr_rimed_ice=1, hail_opt=None, child_cfg=None)
    assert contract.mixed and contract.policy_id == mt.EDGE_MATRIX_POLICY
    assert contract.source_rimed_category == "hail"
    assert contract.mass_source("qh") == "qg"
    # The ratified pair keeps its own policy id.
    assert (_offline_transition_contract(
        8, 18, morr_rimed_ice=None, hail_opt=None).policy_id
        == mt.MP8_TO_MP18_POLICY)
    # WSM6's hail switch reaches the contract as the parent's category.
    assert _offline_transition_contract(
        6, 10, morr_rimed_ice=None, hail_opt=1).source_rimed_category == "hail"
    assert _offline_transition_contract(
        6, 10, morr_rimed_ice=None, hail_opt=None
    ).source_rimed_category == "graupel"
    # The child's own config supplies its switches: the WDM6 seed.
    child = RunConfig(
        nx=12, ny=10, nz=2, dx=1000.0, dy=1000.0, ztop=9000.0, dt=5.0,
        run_seconds=300.0, hybrid_opt=2, etac=0.2, moist=True,
        moist_cq=True, mp_physics=16, wdm6_ccn_conc=2.5e8, specified=True,
        nested=False, terrain_opt=1, map_proj=1, hypsometric_opt=2)
    contract = _offline_transition_contract(
        10, 16, morr_rimed_ice=0, hail_opt=None, child_cfg=child)
    assert contract.target_wdm6_ccn_conc == 2.5e8
    # The online contract's own gate reaches the offline caller in words:
    # a child config without the moist/CQ contract is refused by name.
    dry = replace(child, moist_cq=False)
    with pytest.raises(OfflineChildContractError, match="child.moist_cq=true"):
        _offline_transition_contract(
            10, 16, morr_rimed_ice=0, hail_opt=None, child_cfg=dry)
    with pytest.raises(OfflineChildContractError, match="mp_physics=16"):
        _offline_transition_contract(
            10, 18, morr_rimed_ice=0, hail_opt=None, child_cfg=child)


def test_wdm6_parent_is_read_through_its_own_ccn_row(tmp_path):
    """mp=16 stood at every gate on one named breakage: QNCCN publishes
    both WDM6's ``nn`` and NSSL's ``qnn`` and the field map had no
    scheme-qualified row.  The row exists now, so the binding, the frame
    inspection and the reader all admit a WDM6 parent and the reservoir
    is read rather than zero-filled."""
    namelist = tmp_path / "namelist-mp16.input"
    namelist.write_text(
        "&physics\n mp_physics = 16,\n hail_opt = 1,\n/\n",
        encoding="utf-8")
    binding = bind_parent_physics_from_wrf_namelist(namelist)
    assert binding.mp_physics == 16 and binding.hail_opt == 1
    assert binding.receipt()["hail_opt"] == 1

    frame = tmp_path / "parent-mp16.nc"
    _history(frame, datetime(1974, 4, 3, 12), mp=16)
    info = inspect_parent_history_frame(frame, source_mp_physics=16)
    assert info.source_mp_physics == 16 and info.inferred_mp_physics == 16
    moisture = read_parent_microphysics(frame, source_mp_physics=16)
    assert set(moisture) == {
        "qv", "qc", "qr", "qi", "qs", "qg", "nn", "nc", "nr"}
    assert np.all(moisture["nn"] == np.float32(1.0e8))
    assert _resolve_source_physics(16, None, None) == (16, None)
    assert 16 in OFFLINE_CHILD_MP_PHYSICS and 16 in PARENT_SCHEME_CONTRACT


def test_conservative_parent_snapshot_and_streamed_lateral_intervals(tmp_path):
    start = datetime(1974, 4, 3, 12)
    paths = (tmp_path / "parent-0.nc", tmp_path / "parent-1.nc")
    _history(paths[0], start, ny=18, nx=20, signal=0.0)
    _history(paths[1], start + timedelta(minutes=5),
             ny=18, nx=20, signal=1.0)
    placement = _mp28_placement()
    snapshot = interpolate_parent_boundary_snapshot(
        paths[0], placement, source_mp_physics=8, backend="cpu")
    assert snapshot.fields["u"].shape == (2, 10, 13)
    assert snapshot.fields["v"].shape == (2, 11, 12)
    assert snapshot.fields["w"].shape == (3, 10, 12)
    assert snapshot.fields["mu"].shape == (1, 10, 12)
    assert set(snapshot.fields) == {
        "u", "v", "w", "theta", "phi", "mu",
        "qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni"}

    binding = _physics_binding(tmp_path, mp=8)
    contract = validate_parent_history(
        paths, max_boundary_interval_seconds=900, physics_binding=binding)
    result = build_offline_lateral_boundaries(
        contract, placement, backend="cpu")
    assert len(result.boundaries.intervals) == 1
    interval = result.boundaries.intervals[0]
    assert interval.start_seconds == 0.0
    assert interval.end_seconds == 300.0
    assert set(interval.fields) == set(snapshot.fields)
    assert np.isfinite(interval.fields["theta"].west.tendency).all()
    assert np.any(interval.fields["theta"].west.tendency != 0.0)


@pytest.mark.gpu
@requires_gpu
def test_thompson_parent_boundary_can_change_to_nssl_inventory(tmp_path):
    path = tmp_path / "parent.nc"
    _history(path, datetime(1974, 4, 3, 12), ny=18, nx=20)
    placement = _mp28_placement()
    snapshot = interpolate_parent_boundary_snapshot(
        path, placement, source_mp_physics=8, target_mp_physics=18)
    assert set(snapshot.fields) == {
        "u", "v", "w", "theta", "phi", "mu", "qv", "qc", "qr",
        "qi", "qs", "qg", "qh", "qndrop", "qnr", "qni", "qns",
        "qng", "qnh", "qnn", "qvolg", "qvolh"}
    assert np.all(snapshot.fields["qh"] == 0.0)
    conversion = snapshot.receipt["conversion"]
    assert conversion["target_mp_physics"] == 18
    assert conversion["source"]["mp_physics"] == 8
    assert conversion["target"]["mp_physics"] == 18
    assert conversion["policy_id"] == "mp8-to-mp18-mass-diagnosed-v1"
    assert conversion["translation_order"] == (
        "diagnose-parent-then-spatially-interpolate")
    # The default preprocess backend is the host, so the conversion went
    # through the device in bands and the receipt says so.
    assert conversion["host_chunked"] is True and conversion["chunks"] >= 1


def test_parent_initial_state_builds_standalone_numpy_domain(tmp_path):
    path = tmp_path / "parent.nc"
    valid_time = datetime(1974, 4, 3, 12)
    _history(path, valid_time, ny=18, nx=20, signal=0.5)
    placement = _mp28_placement()
    initial = interpolate_parent_initial_state(
        path, placement, physics_binding=_physics_binding(tmp_path, mp=8),
        backend="cpu")
    cfg = _mp28_child_config(mp=8)
    state = build_offline_child_domain_state(
        initial, cfg, array_module=np)

    assert initial.valid_time == valid_time
    assert initial.receipt["terrain_policy"] == "sint-parent-inherited"
    assert "held physics tendencies" in initial.receipt["spinup_policy"]
    assert initial.receipt["source_physics_binding"]["evidence_kind"] == \
        "wrf-namelist"
    assert state.u.shape == (2, 10, 13)
    assert state.v.shape == (2, 11, 12)
    assert state.thb.shape == (2, 10, 12)
    assert np.all(state.u == np.float32(5.5))
    assert np.all(state.qv == 0.0)
    assert np.array_equal(state.u0, state.u)
    assert np.array_equal(state.nr0, state.nr)
    assert state.rotational
    for field in (state.p, state.al, state.alt, state.thp, state.phb):
        assert np.isfinite(field).all()

    # WRF history stores ETAC in FP32; the corresponding round-trip must not
    # reject the decimal RunConfig value as a different vertical coordinate.
    fp32_receipt = dict(initial.receipt)
    fp32_receipt["etac"] = float(np.float32(0.2))
    fp32_initial = replace(initial, receipt=fp32_receipt)
    build_offline_child_domain_state(fp32_initial, cfg, array_module=np)

    incompatible = replace(cfg, hybrid_opt=1)
    with pytest.raises(OfflineChildContractError, match="hybrid_opt"):
        build_offline_child_domain_state(initial, incompatible, array_module=np)


# ---------------------------------------------------------------------------
# mp_physics=28 (Thompson aerosol-aware) -- the offline-child lane decision.
#
# The lane admits 28 for the SAME-SCHEME case (a 28 parent forcing a 28
# child) and refuses every CROSS-scheme edge touching it by name.  Both
# halves are asserted here, because "half-supported" is exactly the state
# this work package existed to eliminate.
# ---------------------------------------------------------------------------

def _mp28_placement():
    return OfflineChildPlacement(
        parent_nx=20, parent_ny=18, child_nx=12, child_ny=10,
        parent_grid_ratio=1, i_parent_start=4, j_parent_start=4)


def _mp28_child_config(*, mp=28):
    return RunConfig(
        nx=12, ny=10, nz=2, dx=1000.0, dy=1000.0,
        ztop=9000.0, dt=5.0, run_seconds=300.0,
        hybrid_opt=2, etac=0.2, moist=True, mp_physics=mp,
        specified=True, nested=False, terrain_opt=1, map_proj=1,
        hypsometric_opt=2)


def test_mp28_parent_history_is_read_and_inferred_ahead_of_classic_thompson(
        tmp_path):
    """An mp=28 stream must not be advertised as classic Thompson.

    mp=28's wrfout inventory is a strict SUPERSET of mp=8's -- it carries
    QNRAIN/QNICE too -- so the advisory inference has to test the aerosol
    pair first or every aerosol-aware archive reports 8 in its receipt.
    """
    path = tmp_path / "parent-mp28.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=28, ny=18, nx=20)
    info = inspect_parent_history_frame(path, source_mp_physics=28)
    assert info.source_mp_physics == 28
    assert info.inferred_mp_physics == 28

    classic = tmp_path / "parent-mp8.nc"
    _history(classic, datetime(1974, 4, 3, 12), mp=8, ny=18, nx=20)
    assert inspect_parent_history_frame(classic).inferred_mp_physics == 8


def test_mp28_transported_inventory_reaches_the_child_state(tmp_path):
    """The five mp=28 scalars plus the two surface constants land on state."""
    path = tmp_path / "parent-mp28.nc"
    valid_time = datetime(1974, 4, 3, 12)
    _history(path, valid_time, mp=28, ny=18, nx=20, signal=0.5)
    initial = interpolate_parent_initial_state(
        path, _mp28_placement(),
        physics_binding=_physics_binding(tmp_path, mp=28), backend="cpu")

    # Registry.EM_COMMON:3036's scalar list, minus qnbca (wif_input_opt=2).
    assert set(initial.microphysics) == {
        "qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni",
        "nc", "nwfa", "nifa"}

    state = build_offline_child_domain_state(
        initial, _mp28_child_config(), array_module=np)
    assert np.all(state.nc == np.float32(1.0e8))
    assert np.all(state.nwfa == np.float32(1.5e8))
    assert np.all(state.nifa == np.float32(2.5e5))
    # The RK time-t copies mp=28 alone owns.
    assert np.array_equal(state.nc0, state.nc)
    assert np.array_equal(state.nwfa0, state.nwfa)
    assert np.array_equal(state.nifa0, state.nifa)
    # And the per-domain surface emission constants, which nothing in the
    # child can re-derive: thompson_init's nwfa2d fill
    # (module_mp_thompson.F:510) runs only on the "no initial CCN" branch,
    # and this child HAS initial CCN.
    assert np.all(state.nwfa2d == np.float32(4321.0))
    assert np.all(state.nifa2d == np.float32(0.0))
    assert state.nwfa2d.shape == (10, 12)


def test_mp28_child_refuses_a_parent_without_surface_aerosol_emission(
        tmp_path):
    """A parent history missing QNWFA2D must fail loud, not default to zero.

    Zero surface emission is a legal, finite, bounded mp=28 forecast that
    nothing anywhere would flag -- WRF's terminal clamps
    (module_mp_thompson.F:3976-3982) hold the aerosol at its floors.  That
    is precisely why the absence has to raise here.
    """
    path = tmp_path / "parent-mp28-noemit.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=28, ny=18, nx=20,
             omit_aerosol_surface_emission=True)
    with pytest.raises(OfflineChildContractError, match="QNWFA2D"):
        interpolate_parent_initial_state(
            path, _mp28_placement(),
            physics_binding=_physics_binding(tmp_path, mp=28),
            backend="cpu")


def test_mp28_boundary_snapshot_carries_the_aerosol_scalars(tmp_path):
    path = tmp_path / "parent-mp28.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=28, ny=18, nx=20)
    snapshot = interpolate_parent_boundary_snapshot(
        path, _mp28_placement(), source_mp_physics=28, backend="cpu")
    assert set(snapshot.fields) == {
        "u", "v", "w", "theta", "phi", "mu",
        "qv", "qc", "qr", "qi", "qs", "qg",
        "nr", "ni", "nc", "nwfa", "nifa"}


def test_mp28_streamed_lateral_intervals_include_the_aerosol_tracers(
        tmp_path):
    start = datetime(1974, 4, 3, 12)
    paths = (tmp_path / "p0.nc", tmp_path / "p1.nc")
    _history(paths[0], start, mp=28, ny=18, nx=20, signal=0.0)
    _history(paths[1], start + timedelta(minutes=5), mp=28,
             ny=18, nx=20, signal=1.0)
    binding = _physics_binding(tmp_path, mp=28)
    contract = validate_parent_history(
        paths, max_boundary_interval_seconds=900, physics_binding=binding)
    assert contract.source_mp_physics == 28
    result = build_offline_lateral_boundaries(
        contract, _mp28_placement(), backend="cpu")
    interval = result.boundaries.intervals[0]
    assert {"nc", "nwfa", "nifa"} <= set(interval.fields)
    assert np.isfinite(interval.fields["nwfa"].west.value).all()


#: Reading a parent history file goes through the Rust NetCDF decoder
#: (woof.netcdf_bridge.NetcdfBridgeMissing otherwise).  Per test rather than
#: module-wide: the rest of this deck writes its fixtures with netCDF4 and
#: asks woof to decode none of them.  The gate is the CAPABILITY probe in
#: conftest, not `find_netcdf_bin() is None`: that call raises on a
#: WOOF_RW_NETCDF override naming a missing file, and evaluated here at
#: import it took the whole collection down with it.
needs_netcdf_bridge = requires_netcdf_bridge


@needs_netcdf_bridge
@pytest.mark.gpu
@requires_gpu
def test_mp28_offline_converts_to_nssl_and_drops_only_its_aerosols(tmp_path):
    """mp=28 -> 18 runs the online edge on the archive.

    mp=28 carries classic Thompson's six masses and adds nc/nwfa/nifa; the
    contract's species actions record the three as dropped and every NSSL
    moment as diagnosed, exactly as the live nest edge receipts it.
    """
    aero = tmp_path / "parent-mp28.nc"
    _history(aero, datetime(1974, 4, 3, 12), mp=28, ny=18, nx=20)
    placement = _mp28_placement()

    assert 28 in PARENT_SCHEME_CONTRACT
    state = interpolate_parent_initial_state(
        aero, placement, source_mp_physics=28, target_mp_physics=18,
        backend="cpu")
    assert state is not None
    snapshot = interpolate_parent_boundary_snapshot(
        aero, placement, source_mp_physics=28, target_mp_physics=18,
        backend="cpu")
    assert snapshot is not None
    actions = state.receipt["conversion"]["species_actions"]
    dropped = {row["source_field"] for row in actions
               if row["action"] == "dropped"}
    assert {"nc", "nwfa", "nifa"} <= dropped
    assert set(state.microphysics) == {
        "qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop", "qnr", "qni",
        "qns", "qng", "qnh", "qnn", "qvolg", "qvolh"}


def test_every_admitted_parent_scheme_has_a_transport_mapping():
    # The contract set and the transport function are two spellings of the
    # same promise.  WSM6 (mp=6) was admitted by OFFLINE_CHILD_MP_PHYSICS
    # and then refused by _transported_source_fields, which never
    # terminated its chain for a single-moment parent -- a stock WSM6
    # archive hit "unsupported" from a path whose contract said supported
    # (reported by a user against 2.5.2).
    from woof.offline_child import (
        OFFLINE_CHILD_MP_PHYSICS,
        _transported_source_fields,
    )

    for mp in sorted(OFFLINE_CHILD_MP_PHYSICS):
        fields = _transported_source_fields(mp)
        assert fields, f"mp_physics={mp} transports no fields"


def test_wsm6_transports_the_single_moment_set():
    from woof.offline_child import _transported_source_fields

    assert _transported_source_fields(6) == (
        "qv", "qc", "qr", "qi", "qs", "qg")


# The concrete breakage, measured 2026-08-29: a ratio-3 downscale of an
# mp=10 parent aborts at its first radiation call with
#   nr must be finite and non-negative: first_index=(161115, 18),
#   first_value=-3.7252903e-09, negative_count=626, nonfinite_count=0
# -- 626 cells of float32 SINT rounding out of 22.5 million, in an interior
# column.  The fix-up for exactly that artefact had existed and been
# measured since the online nest lane needed it, but reached only children
# built through a DomainState; the offline downscale initializer holds its
# SINT output in a dict and silently went without.  These two tests pin
# both halves: the artefact is cleared, and a real defect still is not.
_SINT_ROUNDING_UNDERSHOOT = -3.7252903e-09


_mp10_placement = _mp28_placement


def test_offline_initial_state_clamps_sint_rounding_undershoot(tmp_path):
    path = tmp_path / "parent.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=10, ny=18, nx=20,
             qnrain=_SINT_ROUNDING_UNDERSHOOT)
    initial = interpolate_parent_initial_state(
        path, _mp10_placement(),
        physics_binding=_physics_binding(tmp_path, mp=10), backend="cpu")

    nr = initial.microphysics["nr"]
    assert nr.min() == 0.0
    account = initial.receipt["positive_definite_clamp"]["nr"]
    assert account["cells"] == nr.size
    assert account["most_negative"] == pytest.approx(
        _SINT_ROUNDING_UNDERSHOOT, rel=1e-6)


def test_offline_initial_state_leaves_a_real_negative_for_the_gate(tmp_path):
    # A thousand per kilogram is not rounding.  The clamp has a ceiling so
    # that the engine's own refusal -- the one that caught the real case --
    # still has something to catch.
    path = tmp_path / "parent.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=10, ny=18, nx=20,
             qnrain=-1.0e3)
    initial = interpolate_parent_initial_state(
        path, _mp10_placement(),
        physics_binding=_physics_binding(tmp_path, mp=10), backend="cpu")

    assert initial.microphysics["nr"].min() < 0.0
    assert initial.receipt["positive_definite_clamp"] == {}


def test_offline_boundary_snapshot_clamps_coupled_undershoot(tmp_path):
    # The boundary lane SINTs COUPLED moments, so the absolute floor has to
    # travel with the coupling or it clamps nothing.  Same parent field as
    # the initial-state case; here it arrives multiplied by chm ~ 8e4.
    path = tmp_path / "parent.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=10, ny=18, nx=20,
             qnrain=_SINT_ROUNDING_UNDERSHOOT)
    snapshot = interpolate_parent_boundary_snapshot(
        path, _mp10_placement(),
        physics_binding=_physics_binding(tmp_path, mp=10), backend="cpu")

    assert snapshot.fields["nr"].min() == 0.0
    assert snapshot.receipt["positive_definite_clamp"]["nr"]["cells"] > 0


def test_offline_boundary_snapshot_leaves_a_real_negative_for_the_gate(tmp_path):
    path = tmp_path / "parent.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=10, ny=18, nx=20,
             qnrain=-1.0e3)
    snapshot = interpolate_parent_boundary_snapshot(
        path, _mp10_placement(),
        physics_binding=_physics_binding(tmp_path, mp=10), backend="cpu")

    assert snapshot.fields["nr"].min() < 0.0
    assert snapshot.receipt["positive_definite_clamp"] == {}


# ---------------------------------------------------------------------------
# mp_physics=50 (P3, one-category ice with prognostic riming) -- the
# offline-child lane decision, on mp=28's exact terms: admitted for the
# SAME-SCHEME case (a 50 parent forcing a 50 child), every cross-scheme
# edge refused by name through the DERIVED mirror of the online lane's
# UNVALIDATED_MIXED_EDGE_SELECTORS.  Transported set per
# Registry.EM_COMMON:3038: moist qv,qc,qr,qi (no qs, no qg) plus scalar
# qni,qnr,qir,qib.
# ---------------------------------------------------------------------------

_p3_placement = _mp28_placement


def _p3_child_config():
    return _mp28_child_config(mp=50)


def test_p3_parent_history_is_inferred_ahead_of_classic_thompson(tmp_path):
    """A P3 stream must not be advertised as classic Thompson.

    P3's wrfout inventory carries QNRAIN/QNICE beside its rime pair, so
    the advisory inference has to test QIR/QIB first -- the mp=28-vs-mp=8
    superset problem again, with a different discriminant: QIR/QIB are
    declared by no other scheme.
    """
    path = tmp_path / "parent-mp50.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=50, ny=18, nx=20)
    info = inspect_parent_history_frame(path, source_mp_physics=50)
    assert info.source_mp_physics == 50
    assert info.inferred_mp_physics == 50


def test_p3_transported_inventory_reaches_the_child_state(tmp_path):
    """qv,qc,qr,qi + ni/nr + the rime pair land on state, and NOTHING else.

    A P3 state allocates no qs/qg at all (one ice category), so the
    right assertion is two-sided: the eight transported fields arrive,
    and the six-species leftovers do not exist to be zero-filled.
    """
    path = tmp_path / "parent-mp50.nc"
    valid_time = datetime(1974, 4, 3, 12)
    _history(path, valid_time, mp=50, ny=18, nx=20, signal=0.5)
    initial = interpolate_parent_initial_state(
        path, _p3_placement(),
        physics_binding=_physics_binding(tmp_path, mp=50), backend="cpu")

    assert set(initial.microphysics) == {
        "qv", "qc", "qr", "qi", "ni", "nr", "qir", "qib"}

    state = build_offline_child_domain_state(
        initial, _p3_child_config(), array_module=np)
    assert np.all(state.qi == np.float32(3.0e-4))
    assert np.all(state.qir == np.float32(1.0e-4))
    assert np.all(state.qib == np.float32(2.0e-7))
    # One ice category: P3 allocates neither qs nor qg
    # (woof/core/state.py mp==50 branch), and this route must not have
    # fabricated them.
    assert getattr(state, "qs", None) is None
    assert getattr(state, "qg", None) is None
    # The RK time-t copies, seeded through the online lane's own table
    # (woof/ingest/nest_init.py::RK_TIME_T_SEED_PAIRS) so the rime pair's
    # qir0/qib0 cannot go missing on this birth path the way they once did
    # on the online one.
    assert np.array_equal(state.qir0, state.qir)
    assert np.array_equal(state.qib0, state.qib)
    assert np.array_equal(state.ni0, state.ni)
    assert np.array_equal(state.nr0, state.nr)


def test_p3_parent_reader_is_scheme_aware(tmp_path):
    """The bound form reads P3's own inventory; the blind form still
    holds the six-species closed world it has always promised."""
    path = tmp_path / "parent-mp50.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=50)
    moisture = read_parent_microphysics(path, source_mp_physics=50)
    assert set(moisture) == {
        "qv", "qc", "qr", "qi", "ni", "nr", "qir", "qib"}
    assert np.all(moisture["qir"] == np.float32(1.0e-4))
    # No scheme evidence means no smaller inventory is accurately complete:
    # a P3 archive has no QSNOW/QGRAUP, and the blind contract says so.
    with pytest.raises(OfflineChildContractError, match="mass fields"):
        read_parent_microphysics(path)


def test_p3_boundary_snapshot_carries_the_rime_pair_and_receipts_it(tmp_path):
    path = tmp_path / "parent-mp50.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=50, ny=18, nx=20)
    snapshot = interpolate_parent_boundary_snapshot(
        path, _p3_placement(), source_mp_physics=50, backend="cpu")
    assert set(snapshot.fields) == {
        "u", "v", "w", "theta", "phi", "mu",
        "qv", "qc", "qr", "qi", "ni", "nr", "qir", "qib"}
    # The receipt names the scheme's field set the way it names every
    # other scheme's: the full sorted inventory, rime pair included.
    assert snapshot.receipt["field_inventory"] == tuple(
        sorted(snapshot.fields))
    assert {"qib", "qir"} <= set(snapshot.receipt["field_inventory"])


def test_p3_streamed_lateral_intervals_include_the_rime_pair(tmp_path):
    start = datetime(1974, 4, 3, 12)
    paths = (tmp_path / "p0.nc", tmp_path / "p1.nc")
    _history(paths[0], start, mp=50, ny=18, nx=20, signal=0.0)
    _history(paths[1], start + timedelta(minutes=5), mp=50,
             ny=18, nx=20, signal=1.0)
    binding = _physics_binding(tmp_path, mp=50)
    contract = validate_parent_history(
        paths, max_boundary_interval_seconds=900, physics_binding=binding)
    assert contract.source_mp_physics == 50
    result = build_offline_lateral_boundaries(
        contract, _p3_placement(), backend="cpu")
    interval = result.boundaries.intervals[0]
    assert {"qi", "ni", "nr", "qir", "qib"} <= set(interval.fields)
    assert np.isfinite(interval.fields["qir"].west.value).all()
    assert np.all(interval.fields["qir"].west.value != 0.0)


@pytest.mark.gpu
@requires_gpu
def test_p3_cross_scheme_offline_edges_run_the_ratified_closure(tmp_path):
    """Both directions, both the initial-state and forcing paths.

    The online lane ratified 50's rime-pair closure; the offline lane now
    runs that closure on the archive, so a P3 parent leaves through the
    exit split and a classic parent enters through the merge, with the
    P3 constants in the receipt.
    """
    p3 = tmp_path / "parent-mp50.nc"
    classic = tmp_path / "parent-mp8.nc"
    _history(p3, datetime(1974, 4, 3, 12), mp=50, ny=18, nx=20)
    _history(classic, datetime(1974, 4, 3, 12), mp=8, ny=18, nx=20)
    placement = _p3_placement()

    # 50 -> 18: the archived ice (3e-4, rime 1e-4 at 500 kg/m3) is split
    # by rime state; total frozen mass is conserved.
    leaving = interpolate_parent_initial_state(
        p3, placement, source_mp_physics=50, target_mp_physics=18,
        backend="cpu")
    frozen = sum(leaving.microphysics[name]
                 for name in ("qi", "qs", "qg", "qh"))
    assert np.allclose(frozen, 3.0e-4, rtol=1e-5)
    assert np.all(leaving.microphysics["qi"] > 0.0)
    assert leaving.receipt["conversion"]["p3_edge"]["direction"] == "leave"
    snapshot = interpolate_parent_boundary_snapshot(
        p3, placement, source_mp_physics=50, target_mp_physics=18,
        backend="cpu")
    assert {"qh", "qvolh", "qnn"} <= set(snapshot.fields)

    # 8 -> 50: a child the lane could not seed before carries the rime
    # pair the entry diagnosis defines.
    entering = interpolate_parent_initial_state(
        classic, placement, source_mp_physics=8, target_mp_physics=50,
        backend="cpu")
    assert set(entering.microphysics) == {
        "qv", "qc", "qr", "qi", "nr", "ni", "qir", "qib"}
    assert entering.receipt["conversion"]["p3_edge"]["direction"] == "enter"
    snapshot = interpolate_parent_boundary_snapshot(
        classic, placement, source_mp_physics=8, target_mp_physics=50,
        backend="cpu")
    assert {"qir", "qib", "ni", "nr"} <= set(snapshot.fields)
    assert "qs" not in snapshot.fields and "qg" not in snapshot.fields


def test_p3_clamp_membership_matches_the_online_lane(tmp_path):
    """ni/nr are clamped members; the rime pair is DELIBERATELY not.

    The membership is the online nest lane's
    (woof/ingest/nest_init.py::POSITIVE_DEFINITE_MOMENTS, imported, not
    re-spelled): number moments carry 1e3..1e9 per kilogram, so a float32
    SINT can round one across zero, while qir/qib are O(1e-4) mixing
    ratios whose absolute rounding error sits decades lower and has never
    been observed to cross -- clamping them would be a trajectory change
    with no evidence behind it.  This test pins that the two lanes keep
    agreeing: the same fabricated undershoot is cleaned off nr and left
    exactly where it is on qir.
    """
    path = tmp_path / "parent-mp50.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=50, ny=18, nx=20,
             qnrain=_SINT_ROUNDING_UNDERSHOOT,
             qir=_SINT_ROUNDING_UNDERSHOOT)
    initial = interpolate_parent_initial_state(
        path, _p3_placement(),
        physics_binding=_physics_binding(tmp_path, mp=50), backend="cpu")

    assert initial.microphysics["nr"].min() == 0.0
    assert "nr" in initial.receipt["positive_definite_clamp"]
    assert initial.microphysics["qir"].min() == pytest.approx(
        _SINT_ROUNDING_UNDERSHOOT, rel=1e-6)
    assert "qir" not in initial.receipt["positive_definite_clamp"]
    assert "qib" not in initial.receipt["positive_definite_clamp"]


def test_offline_transport_inventory_derives_from_the_online_forcing_table():
    """Every admitted scheme's offline inventory IS the online one.

    ``_transported_source_fields`` and ``nest_field_kinds`` are two
    spellings of the same forcing promise; a species present in one and
    absent from the other is exactly how a scheme becomes half-supported.
    Pinned for the WHOLE admitted set so the next scheme's landing cannot
    drift the two lanes apart.
    """
    from types import SimpleNamespace

    from woof.core.preflight import nest_field_kinds
    from woof.offline_child import (
        OFFLINE_CHILD_MP_PHYSICS,
        _transported_source_fields,
    )

    dynamics = {"u", "v", "w", "t", "ph", "mu"}
    for mp in sorted(OFFLINE_CHILD_MP_PHYSICS):
        online = set(nest_field_kinds(
            SimpleNamespace(moist=True, mp_physics=mp))) - dynamics
        assert set(_transported_source_fields(mp)) == online, (
            f"mp_physics={mp}: offline transport inventory drifted from "
            "the online forcing table")


# ---------------------------------------------------------------------
# The child's own progress receipts.  A downscale used to publish
# report.json at the end and nothing a reader could bind to the process
# writing it, so no run browser could show one in flight.
# ---------------------------------------------------------------------

_CHILD_TOML = """[grid]
nx = 12
ny = 10
nz = 3

[run]
run_seconds = 900.0
output_interval_s = 900.0
grid_id = 1
"""


def test_a_child_is_named_after_its_parent_run_not_a_layout_folder():
    """"Downscale of wrfout" names nobody's forecast.

    The prepared routes write ``<stamp>/run/wrfout/wrfout_d01_*``, so the
    frame's own folder is a layout name and so is the one above it; the
    stamped run folder is the first name that says which run this was.
    A parent whose frames sit in its own directory keeps that directory.
    """
    from pathlib import PurePosixPath

    from woof.offline_child_run import _parent_label

    frame = PurePosixPath(
        "/out/chain/run-20260910-214921Z_i202609091200Z/run/wrfout"
        "/wrfout_d01_2026-09-09_12_00_00")
    assert _parent_label(frame) == "run-20260910-214921Z_i202609091200Z"
    assert _parent_label(PurePosixPath(
        "/cases/parent-run/wrfout_d01_2026-09-09_12_00_00")) == "parent-run"


def _frame_under(root, *parts):
    """An empty history frame at ``root/parts.../wrfout_d01_*``."""
    folder = root.joinpath(*parts)
    folder.mkdir(parents=True, exist_ok=True)
    frame = folder / "wrfout_d01_2026-09-26_12_00_00"
    frame.write_bytes(b"")
    return frame


def _write_run_manifest(folder, name, **extra):
    from woof import runplan

    folder.mkdir(parents=True, exist_ok=True)
    (folder / runplan.MANIFEST_FILENAME).write_text(json.dumps(
        {"schema": runplan.MANIFEST_SCHEMA, "name": name, **extra}),
        encoding="utf-8")


def test_a_downscale_of_a_run_plan_forecast_is_named_after_that_forecast(
        tmp_path):
    """A run-plan forecast writes its frames to ``<run>/chain/run/wrfout``.

    'chain' is a layout folder like 'run' and 'wrfout', so a child of that
    forecast was named "Downscale of chain · d02 ×12 · 0.25 km" in every
    run browser.  The parent is named by its own manifest, which is the
    name a run browser shows for it, and by its run folder where there is
    no manifest.
    """
    from woof.offline_child_run import child_run_name

    frame = _frame_under(tmp_path, "run-parent3km", "chain", "run", "wrfout")
    assert child_run_name(frame, grid_id=2, ratio=12, dx=250.0) == (
        "Downscale of run-parent3km · d02 ×12 · 0.25 km")

    _write_run_manifest(tmp_path / "run-parent3km", "Front Range 3 km",
                        route="prepared")
    assert child_run_name(frame, grid_id=2, ratio=12, dx=250.0) == (
        "Downscale of Front Range 3 km · d02 ×12 · 0.25 km")


def test_a_parent_given_by_a_relative_path_is_named_as_an_absolute_one(
        tmp_path, monkeypatch):
    """``woof downscale chain/run/wrfout`` run from inside the parent's
    run folder hands the frames on as typed.  Walking the typed text ran
    out of folders at ``.`` before the run folder, so the manifest was
    never read and the child was named "Downscale of wrfout"; a
    downscale of a downscale given as ``.`` had no folder name at all."""
    from pathlib import Path

    from woof.offline_child_run import child_run_name

    run = tmp_path / "run-parent3km"
    _frame_under(tmp_path, "run-parent3km", "chain", "run", "wrfout")
    monkeypatch.chdir(run)
    frame = Path("chain/run/wrfout/wrfout_d01_2026-09-26_12_00_00")
    assert child_run_name(frame, grid_id=2, ratio=12, dx=250.0) == (
        "Downscale of run-parent3km · d02 ×12 · 0.25 km")
    _write_run_manifest(run, "Front Range 3 km", route="prepared")
    assert child_run_name(frame, grid_id=2, ratio=12, dx=250.0) == (
        "Downscale of Front Range 3 km · d02 ×12 · 0.25 km")

    # The woof go run the forecast bound, reached the same way.
    stamp = "run-20260927-091721Z_i202609270000Z"
    _frame_under(run, "chain", stamp, "run", "wrfout")
    _write_run_manifest(run / "chain" / stamp, "config", route="prepared")
    _write_run_manifest(
        run, "Front Range 3 km", route="prepared",
        native_run={"run_dir": f"/elsewhere/run-parent3km/chain/{stamp}"})
    assert child_run_name(
        Path(f"chain/{stamp}/run/wrfout/wrfout_d01_2026-09-26_12_00_00"),
        grid_id=2, ratio=12, dx=250.0) == (
        "Downscale of Front Range 3 km · d02 ×12 · 0.25 km")

    # A grandchild whose parent downscale is the current folder.
    child = tmp_path / "child-1km"
    _frame_under(tmp_path, "child-1km")
    _write_run_manifest(child, "Downscale of Front Range 3 km · d02 ×3 · 1 km",
                        route="downscale")
    monkeypatch.chdir(child)
    assert child_run_name(
        Path("wrfout_d01_2026-09-26_12_00_00"),
        grid_id=3, ratio=3, dx=1000.0 / 3) == (
        "Downscale of Front Range 3 km · d02 ×3 · 1 km · d03 ×3 · 0.333 km")


def test_a_run_plan_chain_that_claims_a_stamped_folder_names_the_forecast(
        tmp_path):
    """``<run>/chain/run-<stamp>/run/wrfout``: the stamped folder the chain
    claimed is not the run a person started; the run-plan forecast above
    it is."""
    from woof.offline_child_run import child_run_name

    frame = _frame_under(tmp_path, "gfs-parent", "chain",
                         "run-20260927-091721Z_i202609270000Z", "run",
                         "wrfout")
    _write_run_manifest(tmp_path / "gfs-parent", "GFS parent",
                        route="prepared")
    assert child_run_name(frame, grid_id=2, ratio=3, dx=1000.0) == (
        "Downscale of GFS parent · d02 ×3 · 1 km")


def test_a_native_run_gives_way_to_the_forecast_that_bound_it(tmp_path):
    """The ``woof go`` run a run-plan forecast drives writes its own
    manifest, named after its config file.  The forecast records it as
    its ``native_run``; the child is named after the forecast, including
    in a copy of the folder whose manifests still hold the paths they
    were written with."""
    from woof.offline_child_run import child_run_name

    stamp = "run-20260927-091721Z_i202609270000Z"
    frame = _frame_under(tmp_path, "run-parent3km", "chain", stamp, "run",
                         "wrfout")
    _write_run_manifest(tmp_path / "run-parent3km" / "chain" / stamp,
                        "config", route="prepared")
    _write_run_manifest(
        tmp_path / "run-parent3km", "Front Range 3 km", route="prepared",
        native_run={"run_dir": f"/elsewhere/runs/run-parent3km/chain/{stamp}"})
    assert child_run_name(frame, grid_id=2, ratio=12, dx=250.0) == (
        "Downscale of Front Range 3 km · d02 ×12 · 0.25 km")

    # A standalone `woof go` run is its own run: nothing above binds it.
    alone = _frame_under(tmp_path / "case", stamp, "run", "wrfout")
    _write_run_manifest(tmp_path / "case" / stamp, "front-range",
                        route="prepared")
    assert child_run_name(alone, grid_id=2, ratio=3, dx=1000.0) == (
        "Downscale of front-range · d02 ×3 · 1 km")


def test_a_grandchild_extends_its_parent_downscale_name(tmp_path):
    """A downscale's frames sit in its own folder beside its manifest,
    whose name already says "Downscale of"; its child adds its own grid
    to that name instead of wrapping it in a second "Downscale of"."""
    from woof.offline_child_run import child_run_name

    frame = _frame_under(tmp_path, "child-1km")
    _write_run_manifest(tmp_path / "child-1km",
                        "Downscale of Front Range 3 km · d02 ×3 · 1 km",
                        route="downscale")
    assert child_run_name(frame, grid_id=3, ratio=3, dx=1000.0 / 3) == (
        "Downscale of Front Range 3 km · d02 ×3 · 1 km · d03 ×3 · 0.333 km")


def test_a_manifest_that_is_not_the_parents_does_not_name_it(tmp_path):
    """Frames copied into an ordinary folder inside some run keep that
    folder's name: the search for a manifest stops at the first folder
    that is neither a layout folder nor a stamped run folder.  A file
    that is not a named run manifest is not read as one."""
    from woof import runplan
    from woof.offline_child_run import child_run_name

    frame = _frame_under(tmp_path, "some-run", "imports", "wrf-parent")
    _write_run_manifest(tmp_path / "some-run", "Some other run",
                        route="prepared")
    assert child_run_name(frame, grid_id=2, ratio=3, dx=1000.0) == (
        "Downscale of wrf-parent · d02 ×3 · 1 km")

    frame = _frame_under(tmp_path, "run-parent", "chain", "run", "wrfout")
    manifest = tmp_path / "run-parent" / runplan.MANIFEST_FILENAME
    for text in ("{not json", json.dumps({"schema": "other", "name": "x"}),
                 json.dumps({"schema": runplan.MANIFEST_SCHEMA, "name": " "})):
        manifest.write_text(text, encoding="utf-8")
        assert child_run_name(frame, grid_id=2, ratio=3, dx=1000.0) == (
            "Downscale of run-parent · d02 ×3 · 1 km")


def _start_child_progress(progress, tmp_path):
    """Publish one child's manifest and stream, as ``run`` does."""
    outdir = tmp_path / "child-run"
    outdir.mkdir(exist_ok=True)
    config = tmp_path / "child.toml"
    config.write_text(_CHILD_TOML, encoding="utf-8")
    progress.start(
        outdir=outdir, child_config=config, ratio=3,
        start_time=datetime(1974, 4, 3, 12),
        parent={"run_dir": str(tmp_path), "restart": None, "frames": 3,
                "cadence_seconds": 900.0},
        name="Downscale of parent-run")
    return outdir, config


def test_child_publishes_the_manifest_every_other_run_publishes(tmp_path):
    """One manifest shape for every route, so every reader works unchanged."""
    import os

    from woof import runplan
    from woof.offline_child_run import _sha256

    progress = _ChildProgress()
    outdir, config = _start_child_progress(progress, tmp_path)
    try:
        manifest = json.loads(
            (outdir / runplan.MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert manifest["schema"] == runplan.MANIFEST_SCHEMA
        assert manifest["route"] == "downscale"
        assert manifest["pid"] == os.getpid()
        # ...with the identity a page checks before it calls the run alive
        # (a bare pid can be reused by another program after a crash).
        from woof import proc_identity
        assert proc_identity.alive(manifest["process"], os.getpid())
        assert manifest["run_id"]
        # The binding a reader checks: run_dir == outputs_dir == the run's
        # own directory, and a plan source naming the exact config it ran
        # with that file's hash beside it.
        assert manifest["run_dir"] == manifest["outputs_dir"] == str(outdir)
        assert manifest["plan_source"] == f"woof downscale {config.resolve()}"
        assert manifest["plan_sha256"] == _sha256(config.resolve())
        assert manifest["events_path"] == str(outdir / "events.jsonl")
        # No supervisor heartbeat on this route: null, never a path to a
        # file nothing writes.
        assert manifest["progress_path"] is None
        assert manifest["start_time"] == "1974-04-03T12:00:00Z"
        assert manifest["parent"]["frames"] == 3
        assert manifest["name"] == "Downscale of parent-run"

        progress.emit("stage_started", stage="forecast", phase="integrate")
        progress.emit("output_committed", domain=1,
                      valid_time="1974-04-03T12:00:00Z",
                      path=str(outdir / "wrfout_d01_1974-04-03_12_00_00"),
                      bytes=64)
        progress.emit("model_progress", domain=1, model_seconds=900.0,
                      run_seconds=900.0, outer_step=300, total_steps=300,
                      wall_seconds=1.5)
        progress.emit("completed", stage="forecast", result="PASS",
                      outputs=1)
    finally:
        progress.close()
    events = runplan.read_events(outdir / "events.jsonl")
    assert [event["event"] for event in events] == [
        "resolved_plan", "stage_started", "output_committed",
        "model_progress", "completed"]
    assert [event["sequence"] for event in events] == [1, 2, 3, 4, 5]
    assert all(event["schema_version"] == runplan.EVENT_SCHEMA
               for event in events)
    assert events[0]["config_source"] == str(config.resolve())
    assert events[0]["config_sha256"] == events[0]["config_sha256"]


def test_a_child_that_dies_says_so_in_its_own_stream(tmp_path, monkeypatch):
    """The last event a reader sees is always terminal.

    A run that raised used to leave a stream whose final record was an
    ordinary progress line, so a reader tailing it could not tell a dead
    child from a slow one.
    """
    from woof import runplan
    import woof.offline_child_run as child_run

    def explode(args, progress):
        _start_child_progress(progress, tmp_path)
        raise RuntimeError("offline child became non-finite at step 7")

    monkeypatch.setattr(child_run, "_run", explode)
    with pytest.raises(RuntimeError):
        child_run.run(object())
    events = runplan.read_events(tmp_path / "child-run" / "events.jsonl")
    assert events[-1]["event"] == "failed"
    assert "non-finite at step 7" in events[-1]["message"]
    assert events[-1]["message"].startswith("RuntimeError:")
def test_a_gpuwm_warm_rain_frame_is_not_claimed_as_kessler(tmp_path):
    """The R-018 residual: the writer, not the scheme, sets this inventory.

    Stock WRF's passiveqv (mp=0) transports qv alone, so a stock frame
    carrying QVAPOR/QCLOUD/QRAIN is Kessler and nothing else.  woof's own
    mp=0 allocates and advects the warm-rain pair beside qv
    (_transported_source_fields says so in the same module) and
    woof/io/wrfout.py writes all three whenever the state is moist, so on
    a woof tape the same three names are mp=0 OR mp=1.  Naming one of them
    is the mislabel class this ladder was rewritten to end.
    """
    ours = tmp_path / "gpuwm-warm.nc"
    theirs = tmp_path / "wrf-warm.nc"
    _history(ours, datetime(1974, 4, 3, 12), mp=1)
    _history(theirs, datetime(1974, 4, 3, 12), mp=1, producer="wrf")

    mine = inspect_parent_history_frame(ours)
    assert mine.source_kind == "woof"
    assert mine.inferred_mp_physics is None

    stock = inspect_parent_history_frame(theirs)
    assert stock.source_kind == "wrf"
    assert stock.inferred_mp_physics == 1

    # Both are still READABLE: the ambiguity is advisory, and declaring the
    # parent's scheme is the evidence-bearing route that always worked.
    for path in (ours, theirs):
        frame = inspect_parent_history_frame(path, source_mp_physics=1)
        assert frame.source_mp_physics == 1


def test_an_undeclared_wdm6_parent_is_inferred_and_admitted(tmp_path):
    """A WDM6 archive used to satisfy the blind six-species contract and
    was refused on inference for want of a QNCCN row.  The row exists, so
    the inference names 16 and the frame is admitted with or without a
    declaration, like every other ported scheme."""
    frame = tmp_path / "wdm6.nc"
    _history(frame, datetime(1974, 4, 3, 12), mp=16)
    info = inspect_parent_history_frame(frame)
    assert info.inferred_mp_physics == 16
    assert info.source_mp_physics is None
    declared = inspect_parent_history_frame(frame, source_mp_physics=16)
    assert declared.source_mp_physics == 16


def test_an_admitted_scheme_inferred_from_a_frame_still_needs_no_declaration(
        tmp_path):
    """The refusal above must not have closed the undeclared route itself."""
    frame = tmp_path / "morrison.nc"
    _history(frame, datetime(1974, 4, 3, 12), mp=10)
    info = inspect_parent_history_frame(frame)
    assert info.inferred_mp_physics == 10
    assert info.source_mp_physics is None


# --- a child that goes non-finite reports it in words ------------------


def test_a_non_finite_child_logs_its_health_instead_of_raising(capsys):
    """The record this log exists for is the one it used to choke on.

    ``decode_stability_record`` computes no CFL from fields that are not
    finite, so the record's ``cfl`` is ``None`` exactly when ``nan`` is
    true -- and the step log coerced it with ``float()``, three lines
    above the refusal that names the breakage.  A blown-up child
    therefore ended as a ``TypeError`` traceback at exit 1 rather than as
    the refusal, which reads this record back over the last several
    checks and prints that ``None`` as "not computed"
    (tests/test_child_nonfinite_capsule.py).

    AND THE LINE IT WRITES IS JSON.  ``w_max`` is the reading that is not
    finite on exactly this record, and written as a float it reached the
    event stream as the token ``NaN``, which RFC 8259 has no spelling
    for: Python reads it, ``JSON.parse``, ``serde_json``,
    ``encoding/json`` and ``jq`` do not.  So the one ``child_step`` line
    that reports a blown-up child was the one line a reader outside
    Python could not open.
    """
    import json

    import numpy as np

    from woof.core.dycore import decode_stability_record
    from woof.offline_child_run import _log, child_health_log_fields

    blown = decode_stability_record(
        np.array([np.inf, np.nan, np.nan, 0.0, 0.0, np.nan, 0.0, 0.0],
                 dtype=np.float64),
        cfg=None)
    assert blown["nan"] is True and blown["cfl"] is None

    fields = child_health_log_fields(blown)
    assert fields["nan"] is True
    # THE CARRYING SHAPE: null beside a state word, and the two
    # non-numbers stay distinguishable.
    assert fields["cfl"] is None
    assert fields["cfl_state"] == "not computed"
    assert fields["w_max"] is None
    assert fields["w_max_state"] == "non-finite"

    def refuse(token):
        raise ValueError(f"invalid JSON token: {token}")

    _log("child_step", step=6624, total_steps=69120,
         elapsed_seconds=2760.0, **fields)
    line = capsys.readouterr().out.strip()
    assert "NaN" not in line
    # Parsed as a STRICT reader parses it: Python's default accepts NaN,
    # and accepting it here is how the defect stayed invisible.
    record = json.loads(line, parse_constant=refuse)
    assert record["w_max"] is None
    assert record["w_max_state"] == "non-finite"
    assert record["cfl"] is None
    assert record["cfl_state"] == "not computed"

    # A healthy record still travels as the numbers it is, each one said
    # to be a measurement rather than one of the two non-numbers.
    healthy = {"nan": False, "cfl": 0.42, "w_max": 3.5}
    assert child_health_log_fields(healthy) == {
        "nan": False, "cfl": 0.42, "cfl_state": "measured",
        "w_max": 3.5, "w_max_state": "measured"}
