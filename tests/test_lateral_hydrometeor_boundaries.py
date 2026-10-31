"""The root's specified boundary carries the analysed hydrometeors.

WRF v4.7.1 carries the moist species beyond water vapour on a specified
domain only under have_bcs_moist (dyn_em/solve_em.F:2265-2267, :2346,
:4701-4703), off by default, and real.exe writes water vapour only.  A
flow-dependent boundary has zero inflow, so an analysed cloud or snow
field drained out of the domain: a HRRR-forced two-hour winter case kept
1.0 million t of snow over its inner domain where HRRR's own analysis
held 3.1.  These tests hold the table fact (which sources publish the
masses on every frame), the ring's values (the source frame's
hydrometeors, coupled), the number moments seeded beside them, the
end-of-step force-back, the cache admission and the price.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from woof.boundary_fields import (
    BOUNDARY_HYDROMETEOR_MASSES, COLD_START_SEEDED_NUMBERS,
    admissible_boundary_inventory, boundary_hydrometeor_fields,
    external_scalar_fields, mapping_boundary_species,
    potential_external_scalar_fields, sealed_boundary_species,
    source_boundary_species)
from woof.config import RunConfig

FIVE = ("qc", "qr", "qi", "qs", "qg")


def _cfg(**changes):
    values = dict(nx=13, ny=12, nz=4, dx=12000., dy=12000., ztop=10000.,
                  dt=1., run_seconds=120., moist=True, mp_physics=8,
                  specified=True)
    return RunConfig(**(values | changes))


# ---------------------------------------------------------------- table fact

@pytest.mark.parametrize("source", ["hrrr", "hrrr-prs", "icon-d2", "gdas"])
def test_rows_that_publish_hydrometeors_declare_all_five(source):
    """GDAS/GFS pgrb2.0p25 writes the five masses on 22 of its 33 isobaric
    levels; its packaged mapping declares them on those levels
    (``published_levels``), so its boundary carries them."""
    assert BOUNDARY_HYDROMETEOR_MASSES == FIVE
    assert source_boundary_species(source) == FIVE


@pytest.mark.parametrize("source", [
    "gfs", "era5", "ecmwf", "icon-eu", "icon", "gem", "rap",
    "rrfs", "aifs", "gefs", "not-a-source", "", None])
def test_rows_that_publish_none_keep_water_vapour_only(source):
    assert source_boundary_species(source) == ()


def test_a_mapping_declares_its_own_boundary_hydrometeors():
    """A user's mapping is the same table: declared fields are published."""
    assert mapping_boundary_species({"fields": {}}) == ()
    assert mapping_boundary_species({"fields": {
        "snow_mixing_ratio": {}, "cloud_water_mixing_ratio": {},
        "air_temperature": {}}}) == ("qc", "qs")


def test_a_sealed_inventory_names_the_masses_its_tables_hold():
    """A prepared cache's own interval rows answer for its boundary.

    Numbers and dynamics are not masses; a row set that names no fields
    is unreadable (``None``), which is not the same answer as a boundary
    that carries none (``()``).  The masses it names price like a
    source's own row.
    """
    carried = [{"fields": ["mu", "nr", "phi", "qc", "qr", "qs", "qv",
                           "theta", "u", "v"]}] * 2
    assert sealed_boundary_species(carried) == ("qc", "qr", "qs")
    vapour = [{"fields": ["mu", "phi", "qv", "theta", "u", "v"]}]
    assert sealed_boundary_species(vapour) == ()
    for unreadable in (None, [], [{}], [{"fields": None}], "qc"):
        assert sealed_boundary_species(unreadable) is None
    assert source_boundary_species(("qs", "qc")) == ("qc", "qs")
    assert source_boundary_species(()) == ()


def test_the_cold_start_closure_and_the_ring_seed_the_same_numbers():
    assert COLD_START_SEEDED_NUMBERS == {
        8: (("qr", "nr"), ("qi", "ni")),
        28: (("qc", "nc"), ("qr", "nr"), ("qi", "ni"))}


@pytest.mark.parametrize("mp, expected", [
    (1, ("qc", "qr")),
    (6, FIVE),
    (8, (*FIVE, "nr", "ni")),
    # Morrison: the handover closure's numbers ride with the masses
    # (woof/ingest/handover_numbers.py); a flow-dependent Morrison ring
    # brought the mass in at N = 0, sized at the slope limit.
    (10, (*FIVE, "nr", "ni", "ns", "ng")),
    (28, (*FIVE, "nc", "nr", "ni")),
    (50, ("qc", "qr", "qi")),
])
def test_the_ring_carries_what_the_scheme_holds_and_the_numbers_it_seeds(
        mp, expected):
    cfg = _cfg(mp_physics=mp, mp28_aerosol_source="synthetic")
    assert boundary_hydrometeor_fields(cfg, FIVE) == expected
    assert external_scalar_fields(cfg, boundary_species=FIVE) == (
        "qv", *expected)
    # A source publishing none, or a nest (its parent supplies every
    # moist species through its own tables), keeps water vapour only.
    assert external_scalar_fields(cfg) == ("qv",)
    assert boundary_hydrometeor_fields(
        replace(cfg, specified=False, nested=True), FIVE) == ()


# ------------------------------------------------------------- ring values

def _frame(mp, *, cloud_water=None):
    from test_real_init import _analyzed_hrrr_real_init

    result, cfg = _analyzed_hrrr_real_init(
        mp, shape=(12, 13), cloud_water=cloud_water, specified=True,
        mp28_aerosol_source="synthetic",
        init_kwargs={"boundary_species": source_boundary_species("hrrr-prs")})
    return result, cfg


def _coupled(state, name):
    """The frame's hydrometeor in WRF's coupled boundary units (mu-only
    coupling, module_bc.F:2081), in the state's own float32 arithmetic."""
    mu = np.asarray(state.total_mu())
    chm = (np.asarray(state.c1h)[:, None, None] * mu[None]
           + np.asarray(state.c2h)[:, None, None])
    return np.asarray(chm * np.asarray(getattr(state, name)),
                      dtype=np.float64)


@pytest.mark.parametrize("mp", [8, 28])
def test_the_ring_carries_the_source_frames_hydrometeors(mp):
    from woof.ingest.lateral_bc import StateBoundaryFrames

    first, cfg = _frame(mp)
    second, _ = _frame(mp, cloud_water=[2e-4, 1e-4, 5e-5, 3e-5, 1e-5, 0.])
    carried = boundary_hydrometeor_fields(cfg, FIVE)
    assert first.state._external_scalar_boundary_fields == ("qv", *carried)
    assert first.hydrometeor_initialization["lateral_boundary_species"] == (
        list(carried))
    frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    frames.add_state(first.state)
    frames.add_state(second.state)
    interval = frames.build([0.0, 3600.0]).intervals[0]
    assert {*FIVE, *carried} <= set(interval.fields)
    for name in carried:
        before = _coupled(first.state, name)
        after = _coupled(second.state, name)
        side = interval.fields[name]
        np.testing.assert_array_equal(side.west.value, before[:, :, :5])
        np.testing.assert_array_equal(
            side.south.value, before[:, :5, :])
        np.testing.assert_array_equal(
            side.west.tendency, (after[:, :, :5] - before[:, :, :5]) / 3600.0)
    # The analysed masses are really on the ring, not zeros standing in.
    assert np.count_nonzero(interval.fields["qs"].west.value) > 0
    assert np.count_nonzero(interval.fields["nr"].east.value) > 0
    # Mass and number from the SAME frame: every cell carrying rain mass
    # above the scheme's gate carries the number the cold start gave it.
    from woof.core.thompson_entry import R1
    qr = np.asarray(first.state.qr)
    nr = np.asarray(first.state.nr)
    assert (nr[qr > R1] > 0).all()


def test_a_source_that_publishes_none_is_byte_identical():
    from test_real_init import _analyzed_hrrr_real_init
    from woof.ingest.lateral_bc import domain_boundary_snapshot

    plain, _ = _analyzed_hrrr_real_init(8, shape=(12, 13), specified=True)
    named, _ = _analyzed_hrrr_real_init(
        8, shape=(12, 13), specified=True,
        init_kwargs={"boundary_species": ()})
    assert plain.state._external_scalar_boundary_fields == ("qv",)
    assert "lateral_boundary_species" not in plain.hydrometeor_initialization
    first, second = (domain_boundary_snapshot(r.state) for r in (plain, named))
    assert set(first) == set(second) == {"u", "v", "theta", "phi", "mu", "qv"}
    for name in first:
        assert first[name].tobytes() == second[name].tobytes()


def test_an_unknown_boundary_species_is_refused():
    from test_real_init import _analyzed_hrrr_real_init

    with pytest.raises(ValueError, match="boundary_species"):
        _analyzed_hrrr_real_init(8, init_kwargs={"boundary_species": ("qh",)})


# ------------------------------------------------------- end-of-step forcing

def test_spec_bdy_final_forces_the_masses_back_with_their_numbers():
    """Masses are WRF moist-array species (forced back, solve_em.F:4703).
    The seeded numbers are scalar-array species, which WRF forces back only
    on a nest; WOOF forces a number back WITH its mass on a specified ring
    too, because left to its tendency it reached 0 on the outermost row
    under a restored mass (woof/ingest/handover_numbers.py).  A strict WRF
    process keeps WRF's ring (test below)."""
    from woof.ingest import lateral_bc

    fields = dict.fromkeys(
        ("u", "v", "theta", "phi", "mu", "qv", *FIVE, "nr", "ni"), 0)
    forced = []
    for specified in (True, False):
        cfg = SimpleNamespace(specified=specified, nested=not specified,
                              spec_zone=1)
        state = SimpleNamespace(
            qv=object(), lateral_boundaries=object(),
            _lateral_boundary_device=SimpleNamespace(clock=object()))
        calls = []
        interval = SimpleNamespace(fields=fields)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(lateral_bc, "_active_device_interval",
                       lambda *a: (interval, 0., None, None))
            mp.setattr(lateral_bc, "_launch_mu_boundary_values",
                       lambda *a: None)
            mp.setattr(lateral_bc, "_launch_finalize_field",
                       lambda s, name, *a: calls.append(name))
            mp.setattr(lateral_bc, "_ring_at_record_end", lambda *a: False)
            lateral_bc.apply_state_boundary_values(state, cfg)
        forced.append(calls)
    assert forced[0] == ["u", "v", "theta", "phi", "qv", *FIVE, "nr", "ni"]
    assert forced[1] == ["u", "v", "theta", "phi", "qv", *FIVE, "nr", "ni"]


def test_strict_specified_ring_leaves_the_numbers_to_their_tendency(monkeypatch):
    from woof import correct_mode
    from woof.ingest import lateral_bc

    monkeypatch.setattr(correct_mode, "STRICT", True)
    fields = dict.fromkeys(
        ("u", "v", "theta", "phi", "mu", "qv", *FIVE, "nr", "ni"), 0)
    cfg = SimpleNamespace(specified=True, nested=False, spec_zone=1)
    state = SimpleNamespace(
        qv=object(), lateral_boundaries=object(),
        _lateral_boundary_device=SimpleNamespace(clock=object()))
    calls = []
    interval = SimpleNamespace(fields=fields)
    monkeypatch.setattr(lateral_bc, "_active_device_interval",
                        lambda *a: (interval, 0., None, None))
    monkeypatch.setattr(lateral_bc, "_launch_mu_boundary_values", lambda *a: None)
    monkeypatch.setattr(lateral_bc, "_launch_finalize_field",
                        lambda s, name, *a: calls.append(name))
    lateral_bc.apply_state_boundary_values(state, cfg)
    assert calls == ["u", "v", "theta", "phi", "qv", *FIVE]


def _ring_state(nz=4, ny=6, nx=6):
    zeros = lambda: np.zeros((nz, ny, nx), np.float32)
    state = SimpleNamespace(qr=zeros(), nr=zeros(), qs=zeros(), ns=zeros(),
                            alt=np.full((nz, ny, nx), 1.0 / 1.1, np.float32),
                            p=np.full((nz, ny, nx), 8.5e4, np.float32),
                            thp=zeros(), thb=np.full(nz, 300.0, np.float32))
    return state


def test_a_record_end_ring_number_cancelled_to_zero_is_closed_from_its_mass():
    """The step ending on a new forcing record serves the old interval at its
    full length; value + tendency*dtbc cancelled a number falling 2.4e4 ->
    6.6e-5 per kg to 0 beside 1.8e-9 kg/kg of rain on the 1 km downscale
    ring (GFS 2024-05-06 case).  The ring pass closes it from the mass; the
    interior is not the ring's to touch."""
    from woof.ingest import lateral_bc

    state = _ring_state()
    state.qr[2, 0, 3] = 1.8e-9          # south ring row, number cancelled
    state.qr[2, 3, 5] = 4.0e-7          # east ring column
    state.qr[2, 3, 3] = 5.0e-7          # interior
    state.qs[1, 5, 2] = 2.0e-6          # north ring row, snow
    state.nr[2, 1, 1] = 7.0             # an interior number is kept
    cfg = SimpleNamespace(spec_zone=1, mp_physics=10, morr_rimed_ice=1)
    written = lateral_bc.close_ring_numbers(state, cfg, (("qr", "nr"), ("qs", "ns")))
    assert written == 3
    assert state.nr[2, 0, 3] > 0 and state.nr[2, 3, 5] > 0 and state.ns[1, 5, 2] > 0
    assert np.isfinite(state.nr).all() and np.isfinite(state.ns).all()
    assert state.nr[2, 3, 3] == 0.0, "the interior is left to the scheme"
    assert state.nr[2, 1, 1] == 7.0
    assert np.count_nonzero(state.nr) == 3 and np.count_nonzero(state.ns) == 1


def test_the_ring_pass_runs_only_on_the_step_that_ends_on_a_record():
    from woof.ingest import lateral_bc

    interval = SimpleNamespace(start_seconds=900.0, end_seconds=1800.0)
    cfg = SimpleNamespace(nested=False, dt=5.0)
    def state_at(dtbc, clock=True):
        resident = SimpleNamespace(clock=SimpleNamespace(
            dtbc_launch_fp32=np.float32(dtbc), elapsed_seconds=900.0 + dtbc) if clock else None)
        return SimpleNamespace(_lateral_boundary_device=resident,
                               lateral_boundaries=SimpleNamespace(interval_at=lambda t: interval))
    assert lateral_bc._ring_at_record_end(state_at(900.0), cfg)
    assert not lateral_bc._ring_at_record_end(state_at(895.0), cfg)
    assert not lateral_bc._ring_at_record_end(state_at(450.0), cfg)
    assert not lateral_bc._ring_at_record_end(state_at(900.0, clock=False), cfg)


# ----------------------------------------------------------- cache admission

@pytest.mark.parametrize("scalars, admitted", [
    (("qv",), True),
    (("qv", *FIVE, "nr", "ni"), True),
    (("ni", "nr", "qc", "qg", "qi", "qr", "qs", "qv"), True),
    (("qv", "qc", "qs"), True),
    (("qv", "qr"), False),                 # rain without its seeded number
    (("qv", *FIVE), False),
    (("qv", *FIVE, "nr", "ni", "nc"), False),  # mp=8 has no droplet number
    (("qv", "nwfa", "nifa"), False),       # aerosol off mp=28
    (("qv", "qh"), False),
    ((*FIVE, "nr", "ni"), False),          # no water vapour
])
def test_a_sealed_inventory_is_admitted_by_its_structure(scalars, admitted):
    assert admissible_boundary_inventory(_cfg(), scalars) is admitted


def test_mp28_admits_the_aerosol_pair_beside_the_hydrometeors():
    cfg = _cfg(mp_physics=28)
    assert admissible_boundary_inventory(
        cfg, ("qv", *FIVE, "nc", "nr", "ni", "nwfa", "nifa"))
    assert not admissible_boundary_inventory(cfg, ("qv", "nwfa"))


def _refusal(published, fields, *, mp=28, source="hrrr", analysed=FIVE):
    from woof import prepared_single_domain_forecast as runner

    cfg = _cfg(mp_physics=mp, mp28_aerosol_source="synthetic")
    metadata = {
        "lbc": {"intervals": [{"fields": fields}] * 2},
        "hydrometeor_initialization": {"initialized_state_species": {
            name: {"nonzero_count": 4 if name in analysed else 0}
            for name in FIVE}}}
    try:
        runner._refuse_stale_boundary(
            cfg, metadata, published=published, source=source,
            prepared_root=None)
    except ValueError as error:
        return str(error)
    return None


DYNAMICS = ["mu", "phi", "qv", "theta", "u", "v"]


def test_a_cache_sealed_before_the_hydrometeor_boundary_is_refused():
    """The warning it ran with is retired: a stale boundary is refused.

    Run, a water-vapour-only boundary drains the analysed cloud and snow
    out of the edges, so the refusal names that and the re-preparation.
    """
    message = _refusal(FIVE, DYNAMICS)
    assert message is not None
    assert "sealed before its lateral boundary carried the hydrometeors "         "hrrr publishes" in message
    assert "qc, qr, qi, qs, qg" in message
    assert "drain out of the domain edges" in message
    assert "woof prep --source hrrr" in message and "--output-root" in message
    # One mass short is still stale, and is named alone.
    held = sorted([*DYNAMICS, "qc", "qr", "qi", "qs", "nc", "nr", "ni"])
    assert "(qg)" in _refusal(FIVE, held)
    # Only what the start state holds can drain: two analysed masses.
    assert "(qc, qs)" in _refusal(FIVE, DYNAMICS, analysed=("qc", "qs"))


def test_a_cache_a_fresh_preparation_would_write_is_admitted():
    """A source that publishes none, a cache that carries what it
    publishes, a scheme that carries none of it, and a start state with
    nothing analysed to drain: no refusal and no warning.  The GFS and
    ERA5 caches whose boundaries did not change are not touched."""
    assert _refusal((), DYNAMICS, source="gfs") is None
    assert _refusal(FIVE, sorted(
        [*DYNAMICS, *FIVE, "nc", "nr", "ni"])) is None
    # Kessler carries qc and qr only; a cache with exactly those is fresh.
    assert _refusal(FIVE, sorted([*DYNAMICS, "qc", "qr"]), mp=1) is None
    assert "(qc, qr)" in _refusal(FIVE, DYNAMICS, mp=1)
    assert _refusal(FIVE, DYNAMICS, analysed=()) is None
    # A cache with no initialization receipt names nothing analysed.
    from woof import prepared_single_domain_forecast as runner
    runner._refuse_stale_boundary(
        _cfg(mp_physics=28, mp28_aerosol_source="synthetic"),
        {"lbc": {"intervals": [{"fields": DYNAMICS}]}},
        published=FIVE, source="hrrr", prepared_root=None)


def _gdas_documents():
    import json

    from woof.source_authorities import (packaged_authorities,
                                          packaged_authority_sha256)

    pins = packaged_authority_sha256("gdas-pgrb2-0p25-grib2-v1")
    current = json.loads(packaged_authorities(
        "gdas-pgrb2-0p25-grib2-v1")["mapping"].read_text(encoding="utf-8"))
    earlier = json.loads(json.dumps(current))
    names = ("cloud_water_mixing_ratio", "rain_water_mixing_ratio",
             "cloud_ice_mixing_ratio", "snow_mixing_ratio",
             "graupel_or_hail_mixing_ratio")
    for name in names:
        del earlier["fields"][name]
        earlier["target"]["initialization_policies"][name] = (
            "explicit_zero_with_adapter_validation")
    earlier["target"]["required_fields"] = [
        row for row in earlier["target"]["required_fields"]
        if row["name"] not in names]
    return pins, current, earlier


def test_a_gdas_preparation_from_before_its_mapping_declared_them_is_named(
        tmp_path):
    """The packaged GDAS profile's door names the stale boundary.

    The bound mapping of an earlier preparation no longer matches the
    pin; the door says why that matters rather than only that the
    digests differ, under the profile's own name and when the stage door
    hands the same root in as a caller's ``mapped`` preparation.
    """
    import json

    from woof import prepared_single_domain_forecast as runner

    pins, current, earlier = _gdas_documents()
    stale = json.dumps(earlier).encode()
    for source, profile in (("gdas", "gdas-pgrb2-0p25-grib2-v1"),
                            ("mapped", None)):
        message = runner._stale_profile_mapping_refusal(
            source=source, profile_id=profile, mapping_bytes=stale,
            composition_sha256=pins["composition"], prepared_root=tmp_path)
        assert message is not None, source
        assert "gdas publishes" in message
        assert "qc, qr, qi, qs, qg" in message
        assert "woof prep --source gdas" in message
    fresh = json.dumps(current).encode()
    assert runner._stale_profile_mapping_refusal(
        source="gdas", profile_id="gdas-pgrb2-0p25-grib2-v1",
        mapping_bytes=fresh, composition_sha256=pins["composition"],
        prepared_root=tmp_path) is None
    # A caller's own mapping is its own: another name, or another
    # composition, is not the profile's stale preparation.
    renamed = dict(earlier, name="my-own-gfs-mapping")
    assert runner._stale_profile_mapping_refusal(
        source="mapped", profile_id=None,
        mapping_bytes=json.dumps(renamed).encode(),
        composition_sha256=pins["composition"], prepared_root=tmp_path) is None
    assert runner._stale_profile_mapping_refusal(
        source="mapped", profile_id=None, mapping_bytes=stale,
        composition_sha256="0" * 64, prepared_root=tmp_path) is None


def test_a_mapped_root_publishes_what_its_bound_mapping_declares(tmp_path):
    import json

    from woof import prepared_single_domain_forecast as runner

    _pins, current, earlier = _gdas_documents()
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(current), encoding="utf-8")
    assert runner._boundary_publication(
        "mapped", {"mapped_mapping": path}) == FIVE
    path.write_text(json.dumps(earlier), encoding="utf-8")
    assert runner._boundary_publication(
        "mapped", {"mapped_mapping": path}) == ()
    assert runner._boundary_publication("hrrr", {}) == FIVE
    assert runner._boundary_publication("gfs", {}) == ()


# -------------------------------------------------------------------- price

def test_the_price_carries_the_added_tables_and_no_held_slot():
    from woof.core.preflight import (
        estimate_domain, lbc_interval_values, scratch_slot_registry)
    cfg = _cfg(nz=6)
    carried = boundary_hydrometeor_fields(cfg, FIVE)
    per_field = 2 * (2 * cfg.nz * cfg.ny * cfg.spec_bdy_width
                     + 2 * cfg.nz * cfg.spec_bdy_width * cfg.nx)
    assert (lbc_interval_values(cfg, boundary_species=FIVE)
            - lbc_interval_values(cfg)) == len(carried) * per_field
    assert potential_external_scalar_fields(cfg) == ("qv",)
    slots = scratch_slot_registry(cfg, n_lbc_intervals=3)
    assert not any(f"lbc_{name}_held" in slots for name in carried)
    dc = SimpleNamespace(run=cfg, grid_id=1, parent_id=0)
    plain = estimate_domain(dc, n_lbc_intervals=3)
    carrying = estimate_domain(dc, n_lbc_intervals=3, boundary_species=FIVE)
    table = {item.name: item.nbytes for item in carrying.items}
    base = {item.name: item.nbytes for item in plain.items}
    assert (table["lbc_forcing_tables"] - base["lbc_forcing_tables"]
            == 4 * 3 * len(carried) * per_field)


def test_the_alloc_measurement_builds_the_tables_it_prices():
    """``woof check --alloc`` uploads the tables the run holds, and prices them.

    Its synthetic root boundary carries the masses and seeded numbers the
    source puts on the run's boundary, so the eager upload it measures is
    the same size as the ``lbc_forcing_tables`` its estimate and the
    report's observed envelope price.  Red before the A92 follow-up: the
    synthetic set held water vapour alone while the report's peak
    envelope priced the hydrometeor tables, so ``--alloc`` printed two
    forecast envelopes for one run.
    """
    from woof.core.preflight import (
        _synthetic_root_boundaries, lbc_interval_values)
    from woof.ingest.lateral_bc import boundary_storage_shapes

    cfg = _cfg(nz=6)
    carried = boundary_hydrometeor_fields(cfg, FIVE)
    assert carried
    for species, fields in ((FIVE, carried), ((), ())):
        built = _synthetic_root_boundaries(cfg, 3, boundary_species=species)
        assert len(built.intervals) == 3
        for interval in built.intervals:
            assert set(fields) <= set(interval.fields)
            assert not (set(BOUNDARY_HYDROMETEOR_MASSES) - set(fields)) & set(
                interval.fields)
        assert boundary_storage_shapes(built)["lbc_forcing_tables"] == (
            3 * lbc_interval_values(cfg, boundary_species=species),)


def _hrrr_demo():
    """The shipped native-HRRR 3 km demo: one mp=8 specified root."""
    from pathlib import Path

    from woof.experiment import load_experiment

    exp = load_experiment(Path(__file__).resolve().parents[1] / "configs"
                          / "hrrr_native_3km_demo.toml")
    assert len(exp.domains) == 1 and exp.root.run.specified
    return exp


def _forcing_tables(estimate):
    (tables,) = [item.nbytes for domain in estimate.domains
                 for item in domain.items
                 if item.name == "lbc_forcing_tables"]
    return tables


def _added_values(run):
    from woof.core.preflight import lbc_interval_values

    added = (lbc_interval_values(run, boundary_species=FIVE)
             - lbc_interval_values(run))
    assert added == len(boundary_hydrometeor_fields(run, FIVE)) * (
        4 * run.nz * run.spec_bdy_width * (run.nx + run.ny))
    return added


@pytest.mark.parametrize("source", ["hrrr", "hrrr-prs", "mapping"])
def test_the_admission_prices_the_tables_the_source_publishes(source):
    """Every run door's admission carries the source's hydrometeor tables.

    The tiles admission (cold_tree_streaming_decision,
    cold_single_domain_admission), the resident admission
    (admit_resident_road) and the chained and supervised reservations all
    price from ``admission_estimate``; left without the source it priced a
    HRRR-forced root on water vapour tables alone while the review
    (``estimate_phases``) carried the hydrometeors, so the door weighed a
    smaller forecast than the run holds.
    """
    from woof.core import preflight as pf
    from woof.core import streaming as st
    from woof.ingest import boundary_stream
    from woof.mapped_source import HYDROMETEOR_LEGACY_NAMES
    from woof.supervisor import priced_reservation_bytes

    if source == "mapping":
        # A user's own mapping that declares all five: the document is
        # the table the mapped route hands in.
        source = {"fields": {name: {} for name in HYDROMETEOR_LEGACY_NAMES}}
    exp = _hrrr_demo()
    run = exp.root.run
    intervals = pf.lbc_intervals(exp.run_seconds,
                                 pf.DEFAULT_FORCING_INTERVAL_SECONDS)
    added = 4 * intervals * _added_values(run)
    vapour = pf.admission_estimate(exp)
    carrying = pf.admission_estimate(exp, source=source)
    assert _forcing_tables(carrying) - _forcing_tables(vapour) == added
    assert carrying.resident_bytes - vapour.resident_bytes == added
    assert carrying.peak_envelope_bytes > vapour.peak_envelope_bytes
    # A source that publishes none is priced as it was, byte for byte.
    for plain in (None, "gfs", "era5", "not-a-source", {"fields": {}}):
        assert (pf.admission_estimate(exp, source=plain).resident_bytes
                == vapour.resident_bytes)
    # The review and the doors weigh the same tables.
    review = pf.estimate_phases(exp, source=source if isinstance(
        source, str) else "hrrr").forecast
    assert _forcing_tables(review) == _forcing_tables(carrying)
    single = st.cold_single_domain_admission(exp, source=source)
    assert single.peak_envelope_bytes == carrying.peak_envelope_bytes
    assert (priced_reservation_bytes(exp, source=source)
            == carrying.peak_envelope_bytes)
    chained = boundary_stream.chained_admission(
        experiment=exp, backend="cuda", device_bytes=0,
        card=(1 << 40, 0), source=source)
    assert chained["forecast_bytes"] == carrying.peak_envelope_bytes


def test_the_resident_door_refuses_what_the_tables_push_over_the_card(
        monkeypatch):
    """``admit_resident_road`` on a card between the two envelopes."""
    from woof.core import preflight as pf
    from woof.core import streaming as st
    from woof.core.resident_admission import (
        MEMORY_GATE_OVERRIDE_ENV, resident_forecast_terms)
    from woof.ingest.memory_refusal import ResidentMemoryRefused

    monkeypatch.delenv(MEMORY_GATE_OVERRIDE_ENV, raising=False)
    # The tables at the width the run walks: narrowing the MYNN width to
    # fit (A3c, test_rank_pool_residue) is a different question.
    monkeypatch.setattr(pf, "mynn_width_adjustable", lambda exp: False)
    exp = _hrrr_demo()
    vapour = pf.admission_estimate(exp)
    carrying = pf.admission_estimate(exp, source="hrrr")

    def need(estimate):
        return sum(int(value) for value in
                   dict(resident_forecast_terms(estimate)).values())

    assert need(carrying) - need(vapour) > 0
    card = SimpleNamespace(vram_bytes=(need(vapour) + need(carrying)) // 2,
                           device_profile=None)
    assert st.admit_resident_road(exp, None, machine=card)["fits"] is True
    with pytest.raises(ResidentMemoryRefused) as refused:
        st.admit_resident_road(exp, None, machine=card, source="hrrr")
    assert refused.value.need_bytes == need(carrying)
    # At a prepared cache's retained interval count too.
    retained = 4
    added = 4 * retained * _added_values(exp.root.run)
    at_count = pf.estimate_experiment(
        exp, column_chunk=exp.column_chunk, forcing_intervals=retained,
        boundary_species=FIVE)
    plain = pf.estimate_experiment(
        exp, column_chunk=exp.column_chunk, forcing_intervals=retained)
    assert _forcing_tables(at_count) - _forcing_tables(plain) == added


@pytest.mark.parametrize("source", ["hrrr", "hrrr-prs"])
def test_the_host_series_carries_the_tables_the_source_publishes(source):
    """The streamed host claim holds the float64 series with the masses.

    A streamed root keeps its whole lateral forcing series on the host
    and cuts each tile's edge from it; priced on water vapour alone, a
    HRRR-forced streamed run's host admission was short by the
    hydrometeor tables (``lbc_host_series_bytes``,
    ``streaming._boundary_series_host_bytes``, the streamed envelope's
    ``host/boundary_table_bytes`` and the tree road's).
    """
    from dataclasses import replace as _replace

    from woof.core import preflight as pf
    from woof.core import streaming as st

    exp = _hrrr_demo()
    run = exp.root.run
    retained = 6
    added = 8 * retained * _added_values(run)
    assert (pf.lbc_host_series_bytes(run, retained, source=source)
            - pf.lbc_host_series_bytes(run, retained)) == added
    assert pf.lbc_host_series_bytes(run, retained, source="gfs") == (
        pf.lbc_host_series_bytes(run, retained))
    assert (st._boundary_series_host_bytes(run, 3600.0, retained,
                                           source=source)
            - st._boundary_series_host_bytes(run, 3600.0, retained)) == added
    # A nest's forcing is its parent's rolling frame: no host series.
    nest = _replace(run, specified=False, nested=True)
    assert pf.lbc_host_series_bytes(nest, retained, source=source) == 0

    options = _replace(st.options_for_domain(exp.domains[0], exp.tiles),
                       mode="on", tile_nx=64, tile_ny=64, nbuffers=2)
    vapour = st.streamed_envelope(run, options, forcing_intervals=retained)
    carrying = st.streamed_envelope(run, options, forcing_intervals=retained,
                                    source=source)
    assert carrying.boundary_table_bytes - vapour.boundary_table_bytes == added
    assert carrying.host_bytes - vapour.host_bytes == added
    assert carrying.vram_bytes == vapour.vram_bytes
    tiled = _replace(exp, tiles=options)
    envelope = pf.streamed_forecast_envelope(
        tiled, forcing_intervals=retained, source=source)
    assert envelope.boundary_table_bytes == carrying.boundary_table_bytes


# -------------------------------------------------------------- the forecast

@pytest.mark.gpu
@pytest.mark.parametrize("final", [False, True])
def test_the_forecast_ring_takes_the_supplied_hydrometeors(final):
    """Inflow on the west edge: the specified ring serves the table's mass
    and number (not the zero a flow-dependent inflow gives it), held at the
    smaller of its two frames -- here the start, since the end frame holds
    1.25 times more (lateral_bc.hold_hydrometeor_ring_envelope) -- and an
    unsupplied number keeps the flow-dependent zero."""
    cp = pytest.importorskip("cupy")
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import advance_scalars_stage
    from woof.core.state import init_at_rest
    from woof.ingest.lateral_bc import (
        attach_lateral_boundaries, build_lateral_boundaries,
        domain_boundary_snapshot)

    cfg = _cfg(nz=6)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.),
                           cfg.p_surf, cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    state._external_scalar_boundary_fields = external_scalar_fields(
        cfg, boundary_species=("qs", "qr"))
    assert state._external_scalar_boundary_fields == ("qv", "qr", "qs", "nr")
    settled = (("qv", .01), ("qr", 2e-4), ("qs", 5e-4), ("nr", 3e3),
               ("ni", 7e4))
    for name, value in settled:
        getattr(state, name)[:] = value
        getattr(state, name + "0")[:] = value
    state.mup0[:] = state.mup
    first = domain_boundary_snapshot(state)
    for name in ("qr", "qs", "nr"):
        getattr(state, name)[:] *= 1.25
    second = domain_boundary_snapshot(state)
    for name in ("qr", "qs", "nr"):
        getattr(state, name)[:] = getattr(state, name + "0")
    attach_lateral_boundaries(
        state, build_lateral_boundaries([first, second], [0., 60.]))
    ru = cp.full_like(state.u, 1.)
    rv = cp.zeros_like(state.v)
    ww = cp.zeros_like(state.w)
    advance_scalars_stage(state, cfg, ru, rv, ww, dt_eff=1., final=final,
                          apply_relax=True)
    for name, value in (("qr", 2e-4), ("qs", 5e-4), ("nr", 3e3)):
        np.testing.assert_allclose(
            cp.asnumpy(getattr(state, name))[:, :, 0], value, rtol=3e-6)
    np.testing.assert_array_equal(cp.asnumpy(state.ni)[:, :, 0], 0.)


# ------------------------------------------- what the ring serves of a frame

def _ring_tables(start, end, *, numbers=None, duration=3600.0):
    """West-side tables (others empty) of one linear interval, as served."""
    nz, ny, width = start["qs"].shape
    nx = ny

    def side(v0, v1):
        return SimpleNamespace(value=np.array(v0, np.float32),
                               tendency=np.array((v1 - v0) / duration, np.float32),
                               time_law=None)
    fields = {}
    for name in start:
        sides = {"west": side(start[name], end[name]),
                 "east": side(np.zeros((nz, ny, width)), np.zeros((nz, ny, width)))}
        for other in ("south", "north"):
            sides[other] = side(np.zeros((nz, width, nx)), np.zeros((nz, width, nx)))
        fields[name] = SimpleNamespace(**sides)
    return fields


def test_a_cell_in_one_frame_is_not_poured_in_for_the_whole_hour():
    """The 29 Oct 12Z Toronto streaks (woof review): a HRRR cell at a west
    ring point in one hourly frame and gone in the other was served as a
    fixed source ramping over the whole hour, and its snow streamed east as
    a 40-46 dBZ band from the edge.  A mass in one frame only, or present
    in both but narrower along the edge than a convective cell, is not
    served; one that persists over the hour and spans more than a cell is,
    at the smaller frame, with zero tendency; a number keeps the mean
    particle mass of the frame its mass was held from."""
    from woof.ingest.lateral_bc import (hold_hydrometeor_ring_envelope,
                                         ring_cell_radius)
    nz, ny, width = 3, 60, 5
    zeros = np.zeros((nz, ny, width))
    qs0, qs1 = zeros.copy(), zeros.copy()
    qs0[:, 2:30], qs1[:, 2:30] = 4e-4, 5e-4      # stratiform: wide, both frames
    qs1[:, 40:43] = 2.2e-3                        # a cell in the end frame only
    qs0[:, 50:53], qs1[:, 51:54] = 1.5e-3, 2e-3  # cells overlapping in both
    qr0, qr1 = qs0 * 0.5, qs1 * 0.5
    nr0, nr1 = qr0 * 1e7, qr1 * 3e7
    fields = _ring_tables({"qs": qs0, "qr": qr0, "nr": nr0},
                          {"qs": qs1, "qr": qr1, "nr": nr1})
    radius = ring_cell_radius(SimpleNamespace(dx=2250.0))
    assert radius == 6
    held = hold_hydrometeor_ring_envelope(
        lambda name, side, item: getattr(getattr(fields[name], side), item),
        fields, 3600.0, radius, np)
    assert held == 2
    qs = fields["qs"].west.value[0, :, 0]
    np.testing.assert_allclose(qs[2 + radius:30 - radius], 4e-4, rtol=1e-6)
    assert qs[30:].max() == 0.0 and qs[:2].max() == 0.0
    for name in ("qs", "qr", "nr"):
        assert not getattr(fields[name], "west").tendency.any()
    qr = fields["qr"].west.value
    nr = fields["nr"].west.value
    inside = qr > 0
    np.testing.assert_allclose(nr[inside] / qr[inside], 1e7, rtol=1e-5)
    assert not nr[~inside].any()


def test_the_ring_cell_scale_is_physical():
    """The window is a cell's half-width in columns of the domain's own
    grid: one column at 13 km, six at 2.25 km, thirteen at 1 km."""
    from woof.ingest.lateral_bc import ring_cell_radius
    assert [ring_cell_radius(SimpleNamespace(dx=dx))
            for dx in (13000., 2250., 1000.)] == [1, 6, 13]
    with pytest.raises(ValueError, match="grid spacing"):
        ring_cell_radius(SimpleNamespace(dx=0.0))


def test_a_tile_serves_its_slice_of_the_whole_edge_held(monkeypatch):
    """A streamed tile or a split rank windows the domain's edge
    (streaming.window_interval); its ring takes its slice of the edge held
    WHOLE, so it serves the bytes the resident run serves.  Held on its own
    window, a wide snow band cut by the window would be eroded as narrow
    (here: to nothing) and the tiled run would stop matching the resident
    one."""
    from woof.ingest import lateral_bc
    nz, ny, width = 2, 60, 5
    q0 = np.zeros((nz, ny, width))
    q0[:, 20:40] = 6e-4                           # 45 km of snow at 2.25 km
    domain = SimpleNamespace(start_seconds=0.0, end_seconds=3600.0,
                             fields=_ring_tables({"qs": q0}, {"qs": q0}))
    cfg = SimpleNamespace(dx=2250.0)
    whole = lateral_bc._held_host_sides(domain, lateral_bc.ring_cell_radius(cfg))
    served_whole = whole[("qs", "west", "value")][0, :, 0]
    assert served_whole[26:34].min() > 0 and served_whole[:26].max() == 0
    rows = (slice(None), slice(30, 60), slice(None))
    tile = SimpleNamespace(start_seconds=0.0, end_seconds=3600.0,
                           ring_window=(domain, {"west": rows}))
    device = {name: SimpleNamespace(**{
        side: SimpleNamespace(
            value=np.array(getattr(field, side).value[rows if side == "west" else ...], np.float32),
            tendency=np.array(getattr(field, side).tendency[rows if side == "west" else ...], np.float32),
            time_law=None)
        for side in ("west", "east", "south", "north")})
        for name, field in domain.fields.items()}
    resident = SimpleNamespace(ring_held=set(), external_reload_count=0,
                               streaming_external=True)
    state = SimpleNamespace(_lateral_boundary_device=resident, _host_setup_state=True)
    monkeypatch.setattr(lateral_bc, "_resident_interval",
                        lambda s, iv: SimpleNamespace(fields=device))
    lateral_bc._held_resident_interval(state, cfg, tile)
    np.testing.assert_array_equal(device["qs"].west.value,
                                  whole[("qs", "west", "value")][rows])
    assert device["qs"].west.value[0, :4, 0].min() > 0   # the band's held rows 30-33
    assert not device["qs"].west.tendency.any()
