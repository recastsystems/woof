"""The real-radar assimilate(): checkpoints in, seam-shaped increments out.

CPU only.  The observation file is written by the radar lane's own writer
(never a dict pretending to be a file), the checkpoints carry the restart
layout (``state/`` arrays, ARW staggering), and the end-to-end test hands
the callable exactly what :func:`woof.ensemble.cycle.run_cycles` hands it.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from woof.da import obsop
from woof.da.letkf import Localization
from woof.da.radar_assimilation import (
    FALL_SPEED_POLICIES, RadarAssimilationConfig, RadarAssimilationError,
    analysis_sources, assimilate_radar_grid, grid_rotation, innovation_summary,
    make_assimilate, mass_to_u_faces, mass_to_v_faces, mass_to_w_faces,
    member_background_checkpoint, read_checkpoint_state,
    scheme_reflectivity_provider)

NZ, NY, NX = 6, 20, 20
DX_M = 3000.0
TOP_M = 12000.0
MEMBERS = 8
SEED = 20260731
#: Rows/columns of unobserved margin; wide enough that the far corner is
#: outside every observation's localisation lens.
MARGIN = 7
H_LOC_M = 15000.0
V_LOC_M = 4000.0
OBS_ERR_MS = 1.0
#: The analysed set once moisture and hydrometeors are in it.  Spelled
#: here rather than at each call site so a test that adds a species adds
#: it everywhere the invariants are checked.
HYDRO_FIELDS = ("u", "v", "thp", "qv", "qr")


# ---------------------------------------------------------------------------
# world building
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grid():
    from woof.obs.target_grid import TargetGrid
    from woof.static.lambert import LambertGrid

    projection = LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.0, dx=DX_M, dy=DX_M, e_we=NX + 1, e_sn=NY + 1)
    return TargetGrid.from_projection(
        projection, z_w=np.linspace(0.0, TOP_M, NZ + 1),
        name="radar-assim-test")


def smooth(field):
    """Two-pass three-point average: a cheap correlated field."""
    for _ in range(2):
        for axis in (1, 2):
            field = (np.roll(field, 1, axis) + field
                     + np.roll(field, -1, axis)) / 3.0
    return field


def correlated_noise(rng, sigma, shape):
    """Spatially correlated noise with the requested pointwise sigma.

    An ensemble perturbed with WHITE noise defeats an 8-member filter --
    sampled covariances between a gridpoint and its neighbours are pure
    sampling error -- and real IC perturbations are correlated by
    construction (woof.da.perturb draws from a spectrum).  This is the
    cheap stand-in for that property.
    """
    field = smooth(rng.normal(0.0, 1.0, shape))
    return sigma * field / field.std()


def _mass_truth(rng):
    """Smooth-ish mass-point truth winds and moisture."""
    truth = {
        "u": 8.0 + smooth(rng.normal(0.0, 3.0, (NZ, NY, NX))),
        "v": -2.0 + smooth(rng.normal(0.0, 3.0, (NZ, NY, NX))),
        "w": smooth(rng.normal(0.0, 0.5, (NZ, NY, NX))),
        "thp": smooth(rng.normal(0.0, 1.0, (NZ, NY, NX))),
        "qv": np.full((NZ, NY, NX), 8.0e-3),
        "qr": np.maximum(smooth(rng.normal(5.0e-4, 5.0e-4, (NZ, NY, NX))),
                         0.0),
    }
    return truth


def _stagger(name, mass):
    if name == "u":
        return mass_to_u_faces(mass)
    if name == "v":
        return mass_to_v_faces(mass)
    if name == "w":
        return mass_to_w_faces(mass)
    return mass


def _write_checkpoint(path: Path, fields: dict) -> None:
    payload = {f"state/{name}": np.asarray(values, np.float32)
               for name, values in fields.items()}
    np.savez(path, **payload)


def _member_fields(center, rng):
    """One member: an independent draw around the ensemble centre, in
    checkpoint (staggered) layout, with full pressure.

    Truth is its OWN draw around the same centre (the twin-experiment
    construction :mod:`woof.da.synthetic_cycle` uses), so the prior
    mean error and the ensemble spread match by construction -- a
    calibrated toy, which is the setting in which "the analysis must
    beat the prior" is a fair demand of an 8-member filter.
    """
    fields = {}
    for name, sigma in (("u", 1.5), ("v", 1.5), ("w", 0.3)):
        mass = center[name] + correlated_noise(rng, sigma, (NZ, NY, NX))
        fields[name] = _stagger(name, mass)
    fields["thp"] = center["thp"] + correlated_noise(rng, 0.5, (NZ, NY, NX))
    fields["qv"] = center["qv"]
    fields["qr"] = np.maximum(
        center["qr"] + rng.normal(0.0, 2.0e-4, (NZ, NY, NX)), 0.0)
    column = np.linspace(9.0e4, 4.0e4, NZ)
    fields["p"] = np.broadcast_to(column[:, None, None],
                                  (NZ, NY, NX)).copy()
    return fields


@pytest.fixture(scope="module")
def world(grid, tmp_path_factory):
    """Truth, an ensemble of member directories, and a real obs file."""
    from woof.obs.radar_grid import write_radar_grid
    from woof.obs.superob import GriddedObservations, SuperobParams

    root = tmp_path_factory.mktemp("radar-assim")
    rng = np.random.default_rng(SEED)
    center = _mass_truth(rng)
    # Truth: its own draw around the centre, not one of the R members.
    truth = {name: values.copy() for name, values in center.items()}
    for name, sigma in (("u", 1.5), ("v", 1.5), ("w", 0.3),
                        ("thp", 0.5)):
        truth[name] = center[name] + correlated_noise(
            rng, sigma, (NZ, NY, NX))

    # -- members: real checkpoint layout in real member directories --------
    leg_root = root / "cycle_000"
    member_states = {}
    for index in range(MEMBERS):
        member_dir = leg_root / f"member_{index:03d}"
        member_dir.mkdir(parents=True)
        _write_checkpoint(member_dir / "gpuwmrst_d01_000600.npz",
                          _member_fields(center, rng))
        member_states[index] = {"member_dir": str(member_dir)}

    # -- observations through the radar lane's own writer ------------------
    sites = [
        obsop.RadarSite(latitude_deg=float(grid.lat[NY // 2, 1]),
                        longitude_deg=float(grid.lon[NY // 2, 1]),
                        altitude_m=350.0, name="AAAA"),
        obsop.RadarSite(latitude_deg=float(grid.lat[1, NX // 2]),
                        longitude_deg=float(grid.lon[1, NX // 2]),
                        altitude_m=380.0, name="BBBB"),
    ]
    geometry = obsop.GridGeometry.from_target_grid(grid)
    shape = (NZ, NY, NX)
    interior = np.zeros(shape, bool)
    interior[:, MARGIN:NY - MARGIN, MARGIN:NX - MARGIN] = True

    sina, cosa = grid_rotation(grid)
    u_e, v_n = obsop.earth_relative_winds(truth["u"], truth["v"],
                                          sina, cosa)

    east = np.zeros((2,) + shape)
    north = np.zeros((2,) + shape)
    up = np.zeros((2,) + shape)
    vr_obs = np.zeros((2,) + shape)
    for slot, site in enumerate(sites):
        beam = obsop.beam_geometry(geometry, site)
        e, n, u = (np.broadcast_to(np.asarray(c, np.float64), shape).copy()
                   for c in beam.unit_vector_enu())
        east[slot], north[slot], up[slot] = e, n, u
        clean = u_e * e + v_n * n + truth["w"] * u
        vr_obs[slot] = clean + rng.normal(0.0, OBS_ERR_MS, shape)

    vr_mask = np.broadcast_to(interior, (2,) + shape).astype(np.int8)
    zeros = np.zeros(shape, np.float32)
    observations = GriddedObservations(
        z_obs=zeros, z_mask=np.zeros(shape, np.int8), z_err=zeros,
        z_max=zeros, z_mean=zeros, z_count=np.zeros(shape, np.int32),
        vr_obs=vr_obs.astype(np.float32), vr_mask=vr_mask,
        vr_err=np.where(vr_mask.astype(bool), OBS_ERR_MS,
                        0.0).astype(np.float32),
        vr_count=vr_mask.astype(np.int32),
        vr_rejected=np.zeros((2,) + shape, np.int32),
        vr_beam_east=east.astype(np.float32),
        vr_beam_north=north.astype(np.float32),
        vr_beam_up=up.astype(np.float32),
        radars=[{"id": site.name, "lat_deg": site.latitude_deg,
                 "lon_deg": site.longitude_deg, "alt_m": site.altitude_m,
                 "valid_time": "2026-01-01T00:00:00Z"} for site in sites],
        counts=[], provenance=[])
    obs_path = root / "obs-radar-grid.nc"
    write_radar_grid(obs_path, observations, grid,
                     valid_time="2026-01-01T00:00:00Z",
                     params=SuperobParams(), overwrite=True)

    return types.SimpleNamespace(
        root=root, truth=truth, member_states=member_states,
        obs_path=obs_path, interior=interior)


def _config(**overrides):
    # solve_device is PINNED here rather than left to the shipped default,
    # which is "auto".  This module's first line says CPU only, and a
    # default that reads the box would make that claim true on some
    # machines and false on others -- including by opening a CUDA context
    # on a card another run owns.
    kwargs = dict(solve_device="host",
                  localization=Localization(horizontal_m=H_LOC_M,
                                            vertical_m=V_LOC_M),
                  rtps_alpha=0.9, analysis_fields=("u", "v"))
    kwargs.update(overrides)
    return RadarAssimilationConfig(**kwargs)


# ---------------------------------------------------------------------------
# staggering round trips
# ---------------------------------------------------------------------------


def test_restagger_shapes_and_constant_preservation():
    mass = np.full((NZ, NY, NX), 3.25)
    u = mass_to_u_faces(mass)
    v = mass_to_v_faces(mass)
    w = mass_to_w_faces(mass)
    assert u.shape == (NZ, NY, NX + 1)
    assert v.shape == (NZ, NY + 1, NX)
    assert w.shape == (NZ + 1, NY, NX)
    for faces in (u, v, w):
        assert np.all(faces == 3.25)
    # Constant fields survive the full round trip bit-exactly.
    assert np.array_equal(obsop.destagger_u(u), mass)
    assert np.array_equal(obsop.destagger_v(v), mass)
    assert np.array_equal(obsop.destagger_w(w), mass)


def test_restagger_interior_is_adjacent_mean():
    rng = np.random.default_rng(7)
    mass = rng.normal(size=(NZ, NY, NX))
    u = mass_to_u_faces(mass)
    assert np.allclose(u[:, :, 1:NX],
                       0.5 * (mass[:, :, :-1] + mass[:, :, 1:]))
    assert np.array_equal(u[:, :, 0], mass[:, :, 0])
    assert np.array_equal(u[:, :, NX], mass[:, :, -1])


def test_restagger_refuses_wrong_rank():
    with pytest.raises(RadarAssimilationError):
        mass_to_u_faces(np.zeros((NY, NX)))


# ---------------------------------------------------------------------------
# checkpoint discovery binds to the driver's own rule
# ---------------------------------------------------------------------------


def test_member_background_checkpoint_matches_cycle_driver(tmp_path):
    from woof.ensemble.cycle import _member_background_checkpoint

    member = tmp_path / "member_000"
    member.mkdir()
    for name in ("gpuwmrst_d01_000300.npz", "gpuwmrst_d01_000600.npz",
                 "gpuwmrst_d01_000150.npz"):
        np.savez(member / name, **{"state/u": np.zeros(1, np.float32)})
    ours = member_background_checkpoint(member)
    theirs = _member_background_checkpoint(member)
    assert ours == theirs
    assert ours.name == "gpuwmrst_d01_000600.npz"


def test_member_background_checkpoint_fails_closed(tmp_path):
    empty = tmp_path / "member_001"
    empty.mkdir()
    with pytest.raises(RadarAssimilationError, match="no gpuwmrst"):
        member_background_checkpoint(empty)


def test_read_checkpoint_state_refuses_missing_field(tmp_path):
    path = tmp_path / "gpuwmrst_d01_000600.npz"
    np.savez(path, **{"state/u": np.zeros((2, 2, 3), np.float32)})
    with pytest.raises(RadarAssimilationError, match="does not carry"):
        read_checkpoint_state(path, fields=["u", "v"])


# ---------------------------------------------------------------------------
# configuration refusals
# ---------------------------------------------------------------------------


def test_config_refuses_assimilating_nothing():
    # extra_observations=False is the caller STATING that no attributed
    # extra batch will arrive, which is the only thing construction can
    # know.  Left unset the batch list decides, and the analysis call
    # carries the refusal instead; see
    # test_analysis_refuses_when_no_source_and_no_extra_batch_arrives.
    with pytest.raises(RadarAssimilationError, match="assimilate nothing"):
        _config(velocity=False, reflectivity=False, extra_observations=False)


def test_analysis_refuses_when_no_source_and_no_extra_batch_arrives():
    cfg = _config(velocity=False, reflectivity=False)
    with pytest.raises(RadarAssimilationError,
                       match="no extra observation batch arrived"):
        assimilate_radar_grid({0: "unread"}, None, object(), cfg)


def test_analysis_admits_attributed_extra_batches_without_a_declaration(
        grid, tmp_path):
    # The point-only analysis a declaration-gated config refused: no radar
    # source, one attributed extra batch, nothing declared in advance.
    from woof.da.letkf import GriddedObs

    members = 3
    rng = np.random.default_rng(SEED)
    center = _mass_truth(rng)
    paths = {}
    for index in range(members):
        member_dir = tmp_path / f"member_{index:03d}"
        member_dir.mkdir()
        path = member_dir / "gpuwmrst_d01_000600.npz"
        _write_checkpoint(path, _member_fields(center, rng))
        paths[index] = path

    mask = np.zeros((NZ, NY, NX), bool)
    mask[1, NY // 2, NX // 2] = True
    simulated = np.zeros((members, NZ, NY, NX))
    simulated[:, 1, NY // 2, NX // 2] = np.arange(members, dtype=float)
    batch = GriddedObs(
        name="point:temperature", values=np.full((NZ, NY, NX), 1.5),
        errors=np.full((NZ, NY, NX), 1.0), simulated=simulated, mask=mask,
        localization=Localization(horizontal_m=H_LOC_M, vertical_m=V_LOC_M))
    cfg = _config(velocity=False, reflectivity=False,
                  analysis_fields=("thp",))
    increments, provenance = assimilate_radar_grid(
        paths, None, grid, cfg, extra_obs=[batch],
        extra_obs_provenance=[{"source": "fixture"}])
    assert provenance["analysis_sources"] == ["extra_observations"]
    assert provenance["extra_observation_batches"] == 1
    assert set(increments) == set(paths)


def test_config_with_no_radar_source_waits_for_the_batch_list():
    cfg = _config(velocity=False, reflectivity=False)
    assert analysis_sources(cfg, extra_batches=None) == ("extra_observations",)
    assert analysis_sources(cfg, extra_batches=2) == ("extra_observations",)
    assert analysis_sources(cfg, extra_batches=0) == ()


def test_config_refuses_unknown_z_source():
    with pytest.raises(RadarAssimilationError, match="z_source"):
        _config(z_source="z_median")


def test_reflectivity_for_a_scheme_with_no_hx_route_is_refused_at_config_time():
    """The DA door's own plan review: the registry's radar_da row is read
    when the cycle is configured, so a scheme the operator cannot simulate
    is refused here and never inside the first analysis."""

    from woof.physics_registry import consumer_rows_by_selector

    rows = consumer_rows_by_selector("microphysics", "radar_da")
    routes = {mp: row["reflectivity_route"] for mp, row in rows.items()}
    assert set(routes.values()) <= {
        "operator", "scheme-diagnostic", "native-not-separable",
        "native-not-routed", "none", "unrouted"}, routes
    fused = [mp for mp, route in routes.items()
             if route == "native-not-separable"]
    # WSM7 has a native diagnostic but no radar DA H(x) operator yet.
    # Its route remains refused explicitly; earlier routes stay admitted.
    # mp=9 was unrouted, then recorded as fused when audit R-050 read its
    # kernel; its Z block has since been lifted into its own kernel
    # (woof/core/kernels/milbrandt2_zet.cu), which updates nothing, so it
    # is routed as a scheme diagnostic.  The gate that refuses an unrouted
    # scheme is exercised below on a synthesised row, because a positive
    # control that can only run while the defect exists is not a control.
    #
    # FAST_SBM (30) is the one unrouted scheme: it writes no REFL_10CM (its
    # polar-scattering diagnostic is not ported), so there is no H(x) over
    # its bin spectra.  The gate refuses it here, at configuration, which is
    # the shipped positive control the synthesised row below stood in for.
    # CAMMGMP retains an initialized zero field and supplies no radar H(x).
    # 3.0.0 port folds: UDM (27) and RCON (29) keep their own native field,
    # Eta new (95) and the empty MAD-WRF arm (96) have none; none has a radar
    # H(x), so each is refused at configuration (woof.da.obsop cites them).
    unrouted = [11, 24, 27, 29, 30, 95, 96]
    assert sorted(mp for mp, route in routes.items() if route == "unrouted") == unrouted
    for mp in unrouted:
        with pytest.raises(RadarAssimilationError, match=r"no H\(x\) route"):
            _config(reflectivity=True, analysis_fields=HYDRO_FIELDS,
                    positivity_policy="clip", mp_physics=mp)
    # NOR is any scheme fused any more: mp=50's Z was lifted into
    # woof/core/kernels/p3_zdiag.cu (audit S14) and is a scheme
    # diagnostic.  The fused arm of the gate is driven on a probe row
    # (_fused_probe) by the tests below.
    # The named schemes are the fused ones now: TEMPO (900) computes its Z
    # inside its own step (obsop.NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP), so a
    # reflectivity cycle on it is refused by name below, the positive
    # control this arm lacked.  Ferrier-Aligo (5) computes its DBZ column
    # inside EGCP01COLUMN, so it is fused too. Goddard 4-ice (7) writes
    # hail-inclusive reflectivity inside saticel_s.
    assert set(fused) == {5, 7, 15, 55, 900}, routes
    assert routes[50] == "scheme-diagnostic", routes
    for mp in fused:
        with pytest.raises(RadarAssimilationError,
                           match="Turn off the arm"):
            _config(reflectivity=True, analysis_fields=HYDRO_FIELDS,
                    positivity_policy="clip", mp_physics=mp)
    with pytest.raises(RadarAssimilationError, match="not an implemented"):
        _config(reflectivity=True, analysis_fields=HYDRO_FIELDS,
                positivity_policy="clip", mp_physics=99)
    for mp, route in routes.items():
        if route in ("operator", "scheme-diagnostic"):
            cfg = _config(reflectivity=True, analysis_fields=HYDRO_FIELDS,
                          positivity_policy="clip", mp_physics=mp)
            assert cfg.mp_physics == mp
    # Velocity-only cycles never consult the route: the way out works.
    assert _config(mp_physics=50).reflectivity is False


def test_an_unrouted_scheme_would_be_refused_at_config_time(monkeypatch):
    """The unrouted arm of the same gate, on a synthesised registry row.

    See the note in the test above: the shipped registry has no unrouted
    scheme left, so the arm is driven by making one option's row say so.
    """

    from woof import physics_registry
    from woof.physics_registry import consumer_row_for_selector

    def unrouted_row(component_id, key, selector):
        row = consumer_row_for_selector(component_id, key, selector)
        if (component_id, key, int(selector)) == ("microphysics",
                                                  "radar_da", 6):
            row = dict(row)
            row["reflectivity_route"] = "unrouted"
            row["reflectivity_route_reason"] = "synthesised for this test"
        return row

    monkeypatch.setattr(physics_registry, "consumer_row_for_selector",
                        unrouted_row)
    with pytest.raises(RadarAssimilationError, match=r"no H\(x\) route"):
        _config(reflectivity=True, analysis_fields=HYDRO_FIELDS,
                positivity_policy="clip", mp_physics=6)
def test_the_unrouted_class_is_still_refused_by_name(monkeypatch):
    """Positive control for the class no shipped scheme is in.

    "unrouted" is what a scheme added ahead of its operator row would be.
    Nothing ships in that state, so the refusal is exercised against a
    perturbed registry rather than left as a comment -- and the sentence
    it produces has to be about THIS scheme, not the one a nonexistent
    selector gets.
    """

    from copy import deepcopy

    import woof.physics_registry as pr
    from woof.da.obsop import _refuse_unrouted_reflectivity

    perturbed = deepcopy(pr.registry_view())
    option = perturbed["components"]["microphysics"]["options"]["kessler-mp1"]
    option["consumers"]["radar_da"]["reflectivity_route"] = "unrouted"
    option["consumers"]["radar_da"]["reflectivity_route_reason"] = (
        "a probe row, not a shipped state")
    monkeypatch.setattr(pr, "_REGISTRY", perturbed)

    with pytest.raises(NotImplementedError, match="a probe row"):
        _refuse_unrouted_reflectivity(1)
    with pytest.raises(RadarAssimilationError, match=r"no H\(x\) route"):
        _config(reflectivity=True, analysis_fields=HYDRO_FIELDS,
                positivity_policy="clip", mp_physics=1)
    # Every other scheme is untouched by the probe.
    assert _config(reflectivity=True, analysis_fields=HYDRO_FIELDS,
                   positivity_policy="clip", mp_physics=10).mp_physics == 10


def _fused_probe(monkeypatch, mp=1, option="kessler-mp1"):
    """Make one scheme's registry row say its Z is fused with its step.

    No shipped scheme is in that class any more (P3 left it with audit
    S14), so the refusal is driven on a probe row, as the unrouted class
    is above: the row, the reason the refusal prints, and no clear-air
    floor, exactly the state P3 was in.
    """
    from copy import deepcopy

    import woof.physics_registry as pr
    from woof.da import obsop

    perturbed = deepcopy(pr.registry_view())
    row = perturbed["components"]["microphysics"]["options"][option][
        "consumers"]["radar_da"]
    row["reflectivity_route"] = "native-not-separable"
    row["reflectivity_route_reason"] = "a probe row, not a shipped state"
    monkeypatch.setattr(pr, "_REGISTRY", perturbed)
    monkeypatch.setitem(obsop.NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP, mp,
                        "a probe scheme whose Z is fused with its step.")
    monkeypatch.delitem(obsop.CLEAR_AIR_FLOOR_DBZ, mp, raising=False)
    return mp


def test_the_way_out_of_a_reflectivity_refusal_names_the_arm_that_is_ON(
        monkeypatch):
    """A refusal whose way out leaves the run refused is not a refusal.

    The third arm (``fall_speed="reflectivity"``, which reads the simulated
    dBZ whether or not reflectivity is assimilated) was gated with a
    sentence written for the other two: it told a cycle that already had
    ``reflectivity=False, clear_air=False`` to set exactly those, and never
    named ``fall_speed="none"``.  Following it verbatim left the
    configuration refused.  The way out is now composed from the arms that
    are actually on, and every branch of it is followed here to a config
    that constructs.
    """
    fused = _fused_probe(monkeypatch)
    with pytest.raises(RadarAssimilationError) as third_arm:
        _config(velocity=True, reflectivity=False, clear_air=False,
                fall_speed="reflectivity", mp_physics=fused)
    message = str(third_arm.value)
    assert "fall_speed='none'" in message, message
    assert "reflectivity=False" not in message, message
    # ... and the sentence, followed verbatim, constructs.
    kept = _config(velocity=True, reflectivity=False, clear_air=False,
                   fall_speed="none", mp_physics=fused)
    assert kept.fall_speed == "none" and kept.velocity

    with pytest.raises(RadarAssimilationError) as obs_arm:
        _config(reflectivity=True, analysis_fields=HYDRO_FIELDS,
                positivity_policy="clip", mp_physics=fused)
    obs_message = str(obs_arm.value)
    assert "reflectivity=False" in obs_message, obs_message
    assert "fall_speed='none'" not in obs_message, obs_message
    assert _config(reflectivity=False, analysis_fields=HYDRO_FIELDS,
                   positivity_policy="clip",
                   mp_physics=fused).reflectivity is False


def test_a_reflectivity_cycle_that_states_no_scheme_is_still_accepted():
    """``mp_physics=None`` is not a refusal on any arm.

    A first pass at the R-051 gate refused every reflectivity-evaluating
    cycle that stated no scheme.  Those cycles run: this module never
    derives the operator from this field (``assimilate_radar_grid`` takes a
    caller-supplied ``reflectivity_provider``, built off the RUN config),
    and the field's own documented meaning is "detect the moment structure
    from the checkpoint".  The clear-air arm is the exception and is
    refused by its own older guard, which needs the floor rather than the
    operator.
    """
    assert _config(reflectivity=True, analysis_fields=HYDRO_FIELDS,
                   positivity_policy="clip",
                   mp_physics=None).mp_physics is None
    assert _config(fall_speed="reflectivity",
                   mp_physics=None).mp_physics is None
    with pytest.raises(RadarAssimilationError, match="clear-air floor"):
        _config(clear_air=True, reflectivity=True,
                analysis_fields=HYDRO_FIELDS, positivity_policy="clip",
                mp_physics=None)


def test_the_unimplemented_selectors_two_ways_out_are_both_reachable():
    """Both sentences the refusal offers lead to a config that constructs.

    The unimplemented-selector refusal names two: state the ACTIVE scheme,
    or leave it ``None``.  A sibling gate that refused ``None`` made the
    second one instruct the opposite of what it promised, from one
    configuration -- so both are followed here.
    """
    with pytest.raises(RadarAssimilationError) as exc:
        _config(reflectivity=True, analysis_fields=HYDRO_FIELDS,
                positivity_policy="clip", mp_physics=99)
    message = str(exc.value)
    assert "leave it None" in message, message
    assert _config(reflectivity=True, analysis_fields=HYDRO_FIELDS,
                   positivity_policy="clip",
                   mp_physics=None).mp_physics is None
    routed = _routed_reflectivity_schemes()
    assert routed, "no scheme carries a routed H(x)"
    assert _config(reflectivity=True, analysis_fields=HYDRO_FIELDS,
                   positivity_policy="clip",
                   mp_physics=routed[0]).mp_physics == routed[0]
    assert str(routed) in message, message


def _routed_reflectivity_schemes():
    from woof.physics_registry import consumer_rows_by_selector

    rows = consumer_rows_by_selector("microphysics", "radar_da")
    return sorted(
        mp for mp, row in rows.items()
        if isinstance(row, dict)
        and row.get("reflectivity_route") in ("operator", "scheme-diagnostic"))


def test_the_clear_air_floor_refusal_names_its_own_way_out(monkeypatch):
    """A sibling refusal that names no way out is still a dead end.

    The clear-air arm is refused for a scheme whose floor nobody has read,
    which is right -- the floor is not guessable and a wrong one
    manufactures a 35 dB innovation over a genuinely clear sky.  What was
    missing is the half that makes it a refusal rather than a wall: the
    two things the caller can do.  Both are followed here to a
    configuration that constructs.  Every routed scheme's floor has been
    read on this tree, so the row is taken out of the table for the test,
    which is what a scheme added ahead of its floor reading presents.
    """
    from woof.da.obsop import CLEAR_AIR_FLOOR_DBZ

    routed = _routed_reflectivity_schemes()
    mp = 16 if 16 in routed else routed[0]
    monkeypatch.delitem(CLEAR_AIR_FLOOR_DBZ, mp, raising=False)
    with pytest.raises(RadarAssimilationError) as refusal:
        _config(clear_air=True, analysis_fields=HYDRO_FIELDS,
                positivity_policy="clip", mp_physics=mp)
    message = str(refusal.value)
    assert "clear_air_value_dbz" in message, message
    assert "clear_air=False" in message, message
    # Way out 1: state the floor.
    stated = _config(clear_air=True, clear_air_value_dbz=-35.0,
                     analysis_fields=HYDRO_FIELDS,
                     positivity_policy="clip", mp_physics=mp)
    assert stated.clear_air_value_dbz == -35.0
    # Way out 2: turn the arm off, keeping the rest of the cycle.
    without = _config(clear_air=False, analysis_fields=HYDRO_FIELDS,
                      positivity_policy="clip", mp_physics=mp)
    assert without.velocity and without.clear_air is False


def test_a_scheme_with_no_operator_is_refused_before_the_floor_question(
        monkeypatch):
    """Order matters because the floor's way out assumes an operator.

    For a scheme the radar operator cannot simulate at all, "state
    clear_air_value_dbz" reaches a sibling refusal rather than a running
    configuration -- the exact shape this lane already retired once.  The
    route question is therefore asked first, so such a cycle is answered
    with the arm-composed way out, and the floor refusal is only ever
    reached by a scheme that has an H(x).
    """
    from woof.da.obsop import CLEAR_AIR_FLOOR_DBZ

    _fused_probe(monkeypatch)
    routes = _all_reflectivity_routes()
    fused = [mp for mp, route in routes.items()
             if route == "native-not-separable"
             and mp not in CLEAR_AIR_FLOOR_DBZ]
    assert fused, (routes, sorted(CLEAR_AIR_FLOOR_DBZ))
    # No stated floor and no recorded one: BOTH refusals are live, and the
    # one that answers first decides whether the caller is sent somewhere
    # that works.
    with pytest.raises(RadarAssimilationError) as refusal:
        _config(clear_air=True, analysis_fields=HYDRO_FIELDS,
                positivity_policy="clip", mp_physics=fused[0])
    message = str(refusal.value)
    assert "clear_air=False" in message, message
    assert "no clear-air reflectivity floor is recorded" not in message, (
        "the floor refusal answered first, and its 'state "
        "clear_air_value_dbz' way out reaches the route refusal rather "
        f"than a running configuration: {message}")
    # And the sentence it names constructs.
    assert _config(clear_air=False, analysis_fields=HYDRO_FIELDS,
                   positivity_policy="clip",
                   mp_physics=fused[0]).clear_air is False


def _all_reflectivity_routes():
    from woof.physics_registry import consumer_rows_by_selector

    rows = consumer_rows_by_selector("microphysics", "radar_da")
    return {mp: row.get("reflectivity_route")
            for mp, row in rows.items() if isinstance(row, dict)}


def test_config_refuses_unknown_fall_speed():
    with pytest.raises(RadarAssimilationError, match="fall_speed"):
        _config(fall_speed="always")
    assert set(FALL_SPEED_POLICIES) == {"none", "reflectivity"}


def test_reflectivity_without_provider_is_a_refusal(world, grid):
    checkpoints = {
        index: member_background_checkpoint(info["member_dir"])
        for index, info in world.member_states.items()}
    with pytest.raises(RadarAssimilationError,
                       match="reflectivity_provider"):
        assimilate_radar_grid(checkpoints, world.obs_path, grid,
                              _config(reflectivity=True,
                                      analysis_fields=HYDRO_FIELDS,
                                      positivity_policy="clip",
                                      mp_physics=10))
    with pytest.raises(RadarAssimilationError,
                       match="reflectivity_provider"):
        assimilate_radar_grid(checkpoints, world.obs_path, grid,
                              _config(fall_speed="reflectivity",
                                      mp_physics=10))


def test_reflectivity_against_a_wind_only_state_vector_is_refused():
    """The ablation's own finding, turned into a refusal.

    Reflectivity constrains condensate.  With ``u``/``v`` alone in the
    state vector every dBZ increment would come from wind-hydrometeor
    sampling covariance in a rank-``R-1`` ensemble, which is noise, and
    the run would report a reflectivity analysis that analysed nothing.
    """
    with pytest.raises(RadarAssimilationError,
                       match="no analysed field is a thermodynamic"):
        _config(reflectivity=True, analysis_fields=("u", "v"),
                mp_physics=10)


def test_constrained_field_without_a_positivity_policy_is_refused():
    with pytest.raises(RadarAssimilationError,
                       match="states no positivity_policy"):
        _config(analysis_fields=("u", "v", "qr"))
    # ... and an unconstrained field set does not need one.
    assert _config(analysis_fields=("u", "v")).positivity_policy is None


def test_config_refuses_unknown_relaxation_and_moment_policy():
    with pytest.raises(RadarAssimilationError, match="relaxation must be"):
        _config(relaxation="rtpq")
    with pytest.raises(RadarAssimilationError, match="moment_policy must be"):
        _config(moment_policy="whatever")


def test_mapping_bound_observations_refuse_unknown_cycle(world, grid):
    assimilate = make_assimilate({0: world.obs_path}, grid, _config())
    with pytest.raises(RadarAssimilationError, match="cycle 3"):
        assimilate(3, world.member_states)


# ---------------------------------------------------------------------------
# the analysis, end to end on the driver's own seam shapes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analysis(world, grid):
    assimilate = make_assimilate(world.obs_path, grid, _config())
    increments, provenance = assimilate(0, world.member_states)
    return increments, provenance


def test_increments_are_seam_shaped(analysis, world):
    increments, _ = analysis
    assert set(increments) == set(world.member_states)
    for index, member in increments.items():
        assert set(member) == {"u", "v"}
        assert member["u"].shape == (NZ, NY, NX + 1)
        assert member["v"].shape == (NZ, NY + 1, NX)
        for name in ("u", "v"):
            assert np.all(np.isfinite(member[name]))


def test_increments_zero_outside_localization(analysis):
    """The far corner sits outside every observation's lens; face points
    there average mass increments that are exactly zero, so they are
    exactly zero too."""
    increments, _ = analysis
    corner = 2
    saw_signal = 0.0
    for member in increments.values():
        for name in ("u", "v"):
            block = member[name][:, :corner, :corner]
            assert np.all(block == 0.0), (
                f"{name} increments in the unobserved corner are not "
                f"exactly zero (max {np.abs(block).max():g})")
            saw_signal = max(saw_signal, float(np.abs(member[name]).max()))
    assert saw_signal > 0.0, "every increment is zero; the check is vacuous"


def test_analysis_beats_prior_against_truth(analysis, world):
    """Ensemble-mean mass-point wind RMSE must drop where observed."""
    increments, _ = analysis
    region = world.interior
    for name in ("u", "v"):
        prior_members, post_members = [], []
        for index, info in world.member_states.items():
            state = read_checkpoint_state(
                member_background_checkpoint(info["member_dir"]))
            faces = np.asarray(state[name], np.float64)
            mass = (obsop.destagger_u(faces) if name == "u"
                    else obsop.destagger_v(faces))
            inc_faces = np.asarray(increments[index][name], np.float64)
            inc_mass = (obsop.destagger_u(inc_faces) if name == "u"
                        else obsop.destagger_v(inc_faces))
            prior_members.append(mass)
            post_members.append(mass + inc_mass)
        prior_error = np.mean(prior_members, axis=0) - world.truth[name]
        post_error = np.mean(post_members, axis=0) - world.truth[name]
        prior_rmse = float(np.sqrt(np.mean(prior_error[region] ** 2)))
        post_rmse = float(np.sqrt(np.mean(post_error[region] ** 2)))
        assert post_rmse < prior_rmse, (
            f"{name}: analysis RMSE {post_rmse:.3f} did not improve on "
            f"prior {prior_rmse:.3f}")


def test_provenance_names_what_happened(analysis):
    _, provenance = analysis
    assert provenance["schema"] == "gpuwm-da.radar-assimilation.v1"
    assert provenance["stability"] == "experimental"
    assert provenance["cycle"] == 0
    assert provenance["members"] == MEMBERS
    assert provenance["wind_fields_restaggered"] == ["u", "v"]
    assert provenance["fall_speed"] == "none"
    names = {entry["name"] for entry in provenance["innovations"]}
    assert names == {"vr:AAAA", "vr:BBBB"}
    for entry in provenance["innovations"]:
        assert entry["observations"] > 0
        assert entry["innovation_rms"] > 0.0
        assert np.isfinite(entry["innovation_mean"])
        assert entry["obs_error_mean"] == pytest.approx(OBS_ERR_MS)
    assert provenance["filter"]["active_points"] > 0
    # Where the call spent its wall clock, stage by stage in run order: the
    # receipt that tells an observation-side cost from the filter's.
    stages = provenance["stage_wall_seconds"]
    assert list(stages)[:3] == ["observations_read_and_thinned",
                                "checkpoints_read", "prior_stacked"]
    assert {"radar_batches", "filter", "restaggered"} <= set(stages)
    assert all(value >= 0.0 for value in stages.values())
    assert provenance["filter"]["neighbor_search"] in ("index", "forward", "dense")
    # JSON-serialisable, because it lands in the cycle manifest.
    import json
    json.dumps(provenance)


def test_increments_apply_to_the_real_checkpoint(analysis, world, tmp_path):
    from woof.ensemble.increments import apply_increments_to_checkpoint

    increments, _ = analysis
    index = sorted(world.member_states)[0]
    background = member_background_checkpoint(
        world.member_states[index]["member_dir"])
    receipt = apply_increments_to_checkpoint(
        background, increments[index], tmp_path / "analysis.npz")
    assert receipt["field_count"] == 2
    assert receipt["state_sha256_before"] != receipt["state_sha256_after"]
    analysed = read_checkpoint_state(tmp_path / "analysis.npz")
    original = read_checkpoint_state(background)
    for name in ("u", "v"):
        expected = (original[name]
                    + increments[index][name].astype(np.float32))
        assert np.allclose(analysed[name], expected, atol=1e-6)


def test_checkpoints_from_a_different_domain_are_refused(world, grid,
                                                         tmp_path):
    member = tmp_path / "member_000"
    member.mkdir()
    _write_checkpoint(member / "gpuwmrst_d01_000600.npz", {
        "u": np.zeros((NZ, NY, NX + 3), np.float32),
        "v": np.zeros((NZ, NY + 3, NX + 2), np.float32),
        "w": np.zeros((NZ + 1, NY + 2, NX + 2), np.float32),
    })
    checkpoints = dict.fromkeys(
        range(2), member / "gpuwmrst_d01_000600.npz")
    with pytest.raises(RadarAssimilationError, match="not from this"):
        assimilate_radar_grid(checkpoints, world.obs_path, grid, _config())


# ---------------------------------------------------------------------------
# scheme reflectivity provider and the fall-speed closure
# ---------------------------------------------------------------------------


def test_scheme_reflectivity_provider_kessler(world):
    from woof.config import RunConfig

    run = RunConfig(nx=NX, ny=NY, nz=NZ, dx=DX_M, dy=DX_M, dt=10.0,
                    run_seconds=600.0, ztop=TOP_M, mp_physics=1)
    provider = scheme_reflectivity_provider(
        run, base_theta=np.linspace(300.0, 330.0, NZ))
    info = world.member_states[0]
    state = read_checkpoint_state(
        member_background_checkpoint(info["member_dir"]))
    dbz = provider(0, state)
    assert dbz.shape == (NZ, NY, NX)
    assert np.all(np.isfinite(dbz))
    # Rain is present, so somewhere must read above the clear-air floor.
    assert float(dbz.max()) > -35.0


@pytest.mark.parametrize("mp_physics, version", [(8, "wrf_461"),
                                                  (28, "wrf_39_noaa"),
                                                  (28, "wrf_461")])
def test_scheme_reflectivity_provider_thompson_host_checkpoint(mp_physics, version):
    """A host checkpoint of a Thompson scheme gets the scheme's own H(x).

    The host fallback mirrors only Kessler, WSM6 and Morrison; the regional
    door hands the provider host arrays, so before the device route every
    Thompson analysis (the HRRR suite's mp 28 included) ran with no
    reflectivity or clear-air batch at all.  The provider answers on the
    device copy now, and the answer is the CUDA path's own value.
    """
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from woof.da.obsop import HOST_REFLECTIVITY_MIRRORS

    assert mp_physics not in HOST_REFLECTIVITY_MIRRORS
    rng = np.random.default_rng(7)
    shape = (NZ, NY, NX)
    p = np.linspace(95000.0, 30000.0, NZ)[:, None, None] * np.ones(shape)
    theta = np.linspace(300.0, 330.0, NZ)
    state = {"p": p, "thp": np.zeros(shape), "qv": np.full(shape, 8e-3),
             "qc": np.zeros(shape), "qi": np.zeros(shape),
             "qr": np.where(rng.random(shape) > 0.5, 2e-3, 0.0),
             "nr": np.full(shape, 5e3), "qs": np.zeros(shape),
             "qg": np.zeros(shape)}
    run = types.SimpleNamespace(mp_physics=mp_physics, thompson_version=version,
                                thompson_fork_snow_fall=(
                                    "wrf_39_noaa" if version == "wrf_39_noaa"
                                    else "blend"))
    dbz = scheme_reflectivity_provider(run, base_theta=theta)(0, state)
    assert dbz.shape == shape and dbz.dtype == np.float64
    assert np.all(np.isfinite(dbz))
    rain = state["qr"] > 0.0
    assert float(dbz[rain].min()) > 20.0       # 2 g/kg of rain is an echo
    assert float(dbz[~rain].max()) <= -30.0    # an empty cell reads the floor
    # The same state, evaluated by the CUDA path directly.
    view = types.SimpleNamespace(
        **{k: cp.asarray(v, dtype=cp.float32) for k, v in state.items()},
        thb=cp.asarray(theta, dtype=cp.float32), qh=None, ns=None, ng=None,
        nc=None, ni=None)
    buffers = {}
    view.scratch = lambda shp, slot, dtype=None: buffers.setdefault(
        slot, cp.zeros(tuple(shp), dtype=cp.float32 if dtype is None else dtype))
    direct = cp.asnumpy(obsop.simulated_reflectivity(view, run))
    np.testing.assert_array_equal(dbz, direct.astype(np.float64))
    # The checkpoint itself is not written by the operator.
    assert float(state["qg"].max()) == 0.0


def test_scheme_provider_refuses_bad_base_theta():
    from woof.config import RunConfig

    run = RunConfig(nx=NX, ny=NY, nz=NZ, dx=DX_M, dy=DX_M, dt=10.0,
                    run_seconds=600.0, ztop=TOP_M, mp_physics=1)
    with pytest.raises(RadarAssimilationError, match="base_theta"):
        scheme_reflectivity_provider(run, base_theta=np.zeros((2, 2)))


def test_full_operator_with_fall_speed(world, grid):
    """Vr with the Sun & Crook closure wired through the scheme's dBZ."""
    from woof.config import RunConfig

    run = RunConfig(nx=NX, ny=NY, nz=NZ, dx=DX_M, dy=DX_M, dt=10.0,
                    run_seconds=600.0, ztop=TOP_M, mp_physics=1)
    provider = scheme_reflectivity_provider(
        run, base_theta=np.linspace(300.0, 330.0, NZ))
    checkpoints = {
        index: member_background_checkpoint(info["member_dir"])
        for index, info in world.member_states.items()}
    increments, provenance = assimilate_radar_grid(
        checkpoints, world.obs_path, grid,
        _config(fall_speed="reflectivity", mp_physics=1),
        reflectivity_provider=provider)
    assert provenance["fall_speed"] == "reflectivity"
    for member in increments.values():
        for name in ("u", "v"):
            assert np.all(np.isfinite(member[name]))


# ---------------------------------------------------------------------------
# observation thinning and error inflation
# ---------------------------------------------------------------------------


def test_thin_mask_picks_most_gates_then_smallest_error():
    from woof.da.radar_assimilation import thin_mask

    mask = np.ones((1, 4, 4), bool)
    counts = np.zeros((1, 4, 4))
    errors = np.ones((1, 4, 4))
    # Block (0,0): counts decide.
    counts[0, 1, 1] = 9.0
    # Block (0,1): counts tie at 0, errors decide.
    errors[0, 0, 3] = 0.25
    # Block (1,0): all equal -> first cell in block order wins.
    # Block (1,1): no observations at all.
    mask[0, 2:, 2:] = False
    kept = thin_mask(mask, counts, errors, 2)
    assert kept.sum() == 3
    assert kept[0, 1, 1]          # most gates
    assert kept[0, 0, 3]          # smallest error
    assert kept[0, 2, 0]          # first cell of an all-tied block
    assert not kept[0, 2:, 2:].any()   # empty block keeps none
    # Survivors are always original observations.
    assert not (kept & ~mask).any()


def test_thin_mask_identity_and_refusals():
    from woof.da.radar_assimilation import thin_mask

    mask = np.zeros((2, 3, 5), bool)
    mask[0, 1, 2] = True
    kept = thin_mask(mask, np.ones(mask.shape), np.ones(mask.shape), 1)
    assert np.array_equal(kept, mask)
    # Non-divisible extents are handled by padding, not refused.
    kept3 = thin_mask(mask, np.ones(mask.shape), np.ones(mask.shape), 3)
    assert kept3.sum() == 1 and kept3[0, 1, 2]
    with pytest.raises(RadarAssimilationError):
        thin_mask(mask, np.ones(mask.shape), np.ones(mask.shape), 0)
    with pytest.raises(RadarAssimilationError):
        thin_mask(mask[0], np.ones((3, 5)), np.ones((3, 5)), 2)


def test_config_refuses_bad_thinning_inflation_device():
    with pytest.raises(RadarAssimilationError, match="thinning"):
        _config(velocity_thinning_cells=0)
    with pytest.raises(RadarAssimilationError, match="inflation"):
        _config(velocity_error_inflation=0.5)
    with pytest.raises(RadarAssimilationError, match="solve_device"):
        _config(solve_device="gpu")


def test_the_shipped_solve_device_default_is_auto():
    """The default is the fast path where a card exists, not the slow one.

    Pinned as a fact about the SHIPPED config rather than about any CLI,
    because the config is what every caller that does not type a flag
    gets, and the flag's default is only one of them.
    """

    from woof.da.radar_assimilation import (RadarAssimilationConfig,
                                             SOLVE_DEVICES)

    cfg = RadarAssimilationConfig(
        localization=Localization(horizontal_m=H_LOC_M,
                                  vertical_m=V_LOC_M),
        rtps_alpha=0.9, analysis_fields=("u", "v"))
    assert cfg.solve_device == "auto"
    assert SOLVE_DEVICES == ("auto", "host", "cuda")


def test_auto_resolves_and_explicit_devices_are_honoured(monkeypatch):
    """auto answers the box; host and cuda are never second-guessed.

    The explicit half matters as much as the automatic one.  A run pinned
    to host to reproduce a banked receipt must stay on host beside a
    working card, and a run that asked for cuda must meet a missing card
    as an error where it happens rather than as a silent demotion to a
    slower arm with different last bits.
    """

    from woof.da.radar_assimilation import resolve_solve_device

    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    for named in ("host", "cuda"):
        device, reason = resolve_solve_device(named)
        assert device == named
        assert "explicitly" in reason

    # The project-wide keep-off-the-card switch outranks the probe, and
    # says so in the reason a receipt will carry.
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    device, reason = resolve_solve_device("auto")
    assert device == "host"
    assert "GPUWM_NO_LOCAL_GPU" in reason
    # ... and it does not silence an explicit request.
    assert resolve_solve_device("cuda")[0] == "cuda"

    # A box with no reachable device resolves to host and records why,
    # rather than raising: "auto" is a request to work, not to insist.
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    import builtins

    real_import = builtins.__import__

    def no_cupy(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("no cupy on this box")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cupy)
    device, reason = resolve_solve_device("auto")
    assert device == "host"
    assert "ImportError" in reason

    with pytest.raises(RadarAssimilationError, match="solve_device"):
        resolve_solve_device("gpu")


def test_thinned_inflated_analysis_end_to_end(world, grid):
    """Thinning caps the local observation count; inflation reaches the
    batches the filter consumed; the structural zeros survive."""
    checkpoints = {
        index: member_background_checkpoint(info["member_dir"])
        for index, info in world.member_states.items()}
    base_cfg = _config()
    _, base_prov = assimilate_radar_grid(
        checkpoints, world.obs_path, grid, base_cfg)
    thinned_cfg = _config(velocity_thinning_cells=2,
                          velocity_error_inflation=2.0)
    increments, provenance = assimilate_radar_grid(
        checkpoints, world.obs_path, grid, thinned_cfg)

    receipt = provenance["velocity_thinning"]
    assert receipt["cells"] == 2
    assert receipt["error_inflation"] == 2.0
    for radar in receipt["radars"]:
        # One survivor per non-empty 2x2 block: at least a quarter of the
        # observations survive, and strictly fewer than all of them.
        assert (radar["points_before"] / 4.0
                <= radar["points_after"] < radar["points_before"])
    for entry in provenance["innovations"]:
        assert entry["obs_error_mean"] == pytest.approx(2.0 * OBS_ERR_MS)
    assert (provenance["filter"]["max_local_obs"]
            < base_prov["filter"]["max_local_obs"])
    corner = 2
    for member in increments.values():
        for name in ("u", "v"):
            assert np.all(np.isfinite(member[name]))
            assert np.all(member[name][:, :corner, :corner] == 0.0)


# ---------------------------------------------------------------------------
# innovation summary arithmetic
# ---------------------------------------------------------------------------


def test_innovation_summary_hand_computed():
    from woof.da.letkf import GriddedObs

    values = np.zeros((1, 1, 2))
    values[0, 0, 0] = 2.0
    mask = np.zeros((1, 1, 2), bool)
    mask[0, 0, 0] = True
    simulated = np.zeros((2, 1, 1, 2))
    simulated[0, 0, 0, 0] = 1.0
    simulated[1, 0, 0, 0] = 3.0     # mean H(x) = 2.0 -> innovation 0
    errors = np.full((1, 1, 2), 0.5)
    batch = GriddedObs(name="vr:TEST", values=values, errors=errors,
                       simulated=simulated, mask=mask)
    (entry,) = innovation_summary([batch])
    assert entry["observations"] == 1
    assert entry["obs_mean"] == 2.0
    assert entry["hx_mean"] == 2.0
    assert entry["innovation_mean"] == 0.0
    assert entry["innovation_rms"] == 0.0
    assert entry["ensemble_spread_mean"] == pytest.approx(np.sqrt(2.0))
    assert entry["obs_error_mean"] == 0.5


def test_innovation_summary_empty_batch_reports_count_only():
    from woof.da.letkf import GriddedObs

    shape = (1, 2, 2)
    batch = GriddedObs(name="vr:NONE", values=np.zeros(shape),
                       errors=np.ones(shape),
                       simulated=np.zeros((2,) + shape),
                       mask=np.zeros(shape, bool))
    (entry,) = innovation_summary([batch])
    assert entry == {"name": "vr:NONE", "observations": 0}


# ---------------------------------------------------------------------------
# the moisture / hydrometeor analysis
#
# The same twin-experiment gate the wind analysis is held to, extended to
# the variables reflectivity actually constrains.  The ensemble is built
# by the REAL producer -- woof.da.perturb, not a hand-rolled draw in this
# file -- because the whole point of the milestone is that a perturbation
# module and a filter have to agree about what a two-moment state is, and
# a fixture that perturbs its own way could agree with neither.
# ---------------------------------------------------------------------------

HZ, HY, HX = 6, 24, 24
H_DX_M = 3000.0
H_MEMBERS = 12
#: Resolvable on this domain: >= 2 grid spacings and <= span/(2*pi).
H_SCALE_KM = 8.0
H_RIM = 3
H_MARGIN = 8
Z_ERR_DBZ = 5.0


def _morrison_run_config():
    from woof.config import RunConfig

    return RunConfig(nx=HX, ny=HY, nz=HZ, dx=H_DX_M, dy=H_DX_M, dt=10.0,
                     run_seconds=600.0, ztop=TOP_M, mp_physics=10)


def _hydro_center():
    """A background with one storm: condensate in a blob, clear outside.

    Deliberately zero outside the blob.  A multiplicative perturbation
    leaves clear air exactly clear, so this is also the fixture that
    proves the filter cannot create echo where no member has any -- the
    accurate limitation the milestone has to state.
    """
    base_theta = np.linspace(300.0, 340.0, HZ)
    column = np.linspace(9.2e4, 3.0e4, HZ)
    zz, yy, xx = np.meshgrid(np.arange(HZ), np.arange(HY), np.arange(HX),
                             indexing="ij")
    radius = np.sqrt((yy - 12.0) ** 2 + (xx - 12.0) ** 2)
    blob = np.exp(-(radius / 4.0) ** 2) * np.exp(-((zz - 2.5) / 2.0) ** 2)
    state = {
        "u": np.full((HZ, HY, HX + 1), 6.0),
        "v": np.full((HZ, HY + 1, HX), -3.0),
        "w": np.zeros((HZ + 1, HY, HX)),
        "thp": np.zeros((HZ, HY, HX)),
        "qv": np.full((HZ, HY, HX), 6.0e-3),
        "p": np.broadcast_to(column[:, None, None], (HZ, HY, HX)).copy(),
        "qr": 1.2e-3 * blob,
        "nr": 8.0e3 * blob,
        "qs": 4.0e-4 * blob,
        "ns": 4.0e3 * blob,
        "qg": 6.0e-4 * blob,
        "ng": 2.0e3 * blob,
    }
    # Below the scheme's own activity gate the species is absent, and a
    # background that trails off to 1e-30 is not a storm edge, it is
    # noise the guard would have to reason about.
    for name, number in (("qr", "nr"), ("qs", "ns"), ("qg", "ng")):
        off = state[name] < 1.0e-8
        state[name] = np.where(off, 0.0, state[name])
        state[number] = np.where(off, 0.0, state[number])
    return state, base_theta


def _perturbed_member(center, base_theta, seed):
    """One member, perturbed by woof.da.perturb's own contract."""
    from woof.da import perturb

    state = types.SimpleNamespace(**{name: values.copy()
                                     for name, values in center.items()})
    state.thb = base_theta
    for absent in ("qc", "nc", "qi", "ni", "qh", "php", "mup", "al", "alt",
                   "h_diabatic", "qndrop", "qvolg", "qvolh"):
        setattr(state, absent, None)
    cfg = perturb.PerturbationConfig(
        dx_km=H_DX_M / 1000.0, dy_km=H_DX_M / 1000.0, rim_width=H_RIM,
        # this namespace carries no vertical coordinate (php, mup are
        # None), so the default hydrostatic mass balance refuses it by
        # name; the twin here is about the species draw, not the column
        mass_balance="none",
        fields=(
            perturb.FieldPerturbation("u", 1.5, H_SCALE_KM),
            perturb.FieldPerturbation("v", 1.5, H_SCALE_KM),
            perturb.FieldPerturbation("theta", 0.5, H_SCALE_KM),
            perturb.FieldPerturbation("qv", 0.05, H_SCALE_KM,
                                      mode="lognormal", clip_sigmas=2.5),
        ),
        species=(
            perturb.SpeciesPerturbation("qr", 0.7, H_SCALE_KM,
                                        clip_sigmas=2.5),
            perturb.SpeciesPerturbation("qs", 0.7, H_SCALE_KM,
                                        clip_sigmas=2.5),
            perturb.SpeciesPerturbation("qg", 0.7, H_SCALE_KM,
                                        clip_sigmas=2.5),
        ))
    provenance = perturb.apply_perturbations(state, seed, cfg)
    fields = {name: getattr(state, name)
              for name in ("u", "v", "w", "thp", "qv", "p",
                           "qr", "nr", "qs", "ns", "qg", "ng")}
    return fields, provenance


@pytest.fixture(scope="module")
def hydro_grid():
    from woof.obs.target_grid import TargetGrid
    from woof.static.lambert import LambertGrid

    projection = LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.0, dx=H_DX_M, dy=H_DX_M, e_we=HX + 1, e_sn=HY + 1)
    return TargetGrid.from_projection(
        projection, z_w=np.linspace(0.0, TOP_M, HZ + 1),
        name="radar-assim-hydro")


@pytest.fixture(scope="module")
def hydro_world(hydro_grid, tmp_path_factory):
    """Twin experiment: truth is one more draw from the same centre."""
    from woof.obs.radar_grid import write_radar_grid
    from woof.obs.superob import GriddedObservations, SuperobParams

    root = tmp_path_factory.mktemp("radar-assim-hydro")
    center, base_theta = _hydro_center()
    run = _morrison_run_config()
    provider = scheme_reflectivity_provider(run, base_theta=base_theta)

    truth, _ = _perturbed_member(center, base_theta, 9000001)

    leg_root = root / "cycle_000"
    member_states = {}
    provenances = []
    for index in range(H_MEMBERS):
        member_dir = leg_root / f"member_{index:03d}"
        member_dir.mkdir(parents=True)
        fields, provenance = _perturbed_member(center, base_theta,
                                               7000000 + index)
        _write_checkpoint(member_dir / "gpuwmrst_d01_000600.npz", fields)
        member_states[index] = {"member_dir": str(member_dir)}
        provenances.append(provenance)

    shape = (HZ, HY, HX)
    interior = np.zeros(shape, bool)
    interior[:, H_MARGIN:HY - H_MARGIN, H_MARGIN:HX - H_MARGIN] = True

    # -- reflectivity observations, from the truth through the SAME H(x)
    rng = np.random.default_rng(4242)
    z_truth = provider(0, truth)
    z_mask = interior & (z_truth > -10.0)
    z_obs = np.where(z_mask, z_truth + rng.normal(0.0, Z_ERR_DBZ, shape),
                     0.0)

    # -- radial velocity, exactly as the wind fixture does it ------------
    sites = [obsop.RadarSite(latitude_deg=float(hydro_grid.lat[HY // 2, 1]),
                             longitude_deg=float(hydro_grid.lon[HY // 2, 1]),
                             altitude_m=350.0, name="AAAA")]
    geometry = obsop.GridGeometry.from_target_grid(hydro_grid)
    sina, cosa = grid_rotation(hydro_grid)
    u_e, v_n = obsop.earth_relative_winds(
        obsop.destagger_u(truth["u"]), obsop.destagger_v(truth["v"]),
        sina, cosa)
    w_mass = obsop.destagger_w(truth["w"])
    beam = obsop.beam_geometry(geometry, sites[0])
    east, north, up = (np.broadcast_to(np.asarray(c, np.float64),
                                       shape).copy()
                       for c in beam.unit_vector_enu())
    vr_obs = (u_e * east + v_n * north + w_mass * up
              + rng.normal(0.0, OBS_ERR_MS, shape))[None]
    vr_mask = interior[None].astype(np.int8)

    observations = GriddedObservations(
        z_obs=z_obs.astype(np.float32), z_mask=z_mask.astype(np.int8),
        z_err=np.where(z_mask, Z_ERR_DBZ, 0.0).astype(np.float32),
        z_max=z_obs.astype(np.float32), z_mean=z_obs.astype(np.float32),
        z_count=z_mask.astype(np.int32),
        vr_obs=vr_obs.astype(np.float32), vr_mask=vr_mask,
        vr_err=np.where(vr_mask.astype(bool), OBS_ERR_MS,
                        0.0).astype(np.float32),
        vr_count=vr_mask.astype(np.int32),
        vr_rejected=np.zeros((1,) + shape, np.int32),
        vr_beam_east=east[None].astype(np.float32),
        vr_beam_north=north[None].astype(np.float32),
        vr_beam_up=up[None].astype(np.float32),
        radars=[{"id": sites[0].name, "lat_deg": sites[0].latitude_deg,
                 "lon_deg": sites[0].longitude_deg,
                 "alt_m": sites[0].altitude_m,
                 "valid_time": "2026-01-01T00:00:00Z"}],
        counts=[], provenance=[])
    obs_path = root / "obs-radar-grid.nc"
    write_radar_grid(obs_path, observations, hydro_grid,
                     valid_time="2026-01-01T00:00:00Z",
                     params=SuperobParams(), overwrite=True)

    return types.SimpleNamespace(
        root=root, center=center, base_theta=base_theta, truth=truth,
        member_states=member_states, obs_path=obs_path, interior=interior,
        z_mask=z_mask, provider=provider, run=run,
        perturbation=provenances)


def test_perturbation_gives_every_species_spread_and_breaks_no_pair(
        hydro_world):
    """Criterion (a), at the source: the ensemble carries the spread."""
    from woof.da.moments import moment_consistency_report

    members = [read_checkpoint_state(
        member_background_checkpoint(info["member_dir"]))
        for info in hydro_world.member_states.values()]
    for name in ("qv", "qr", "qs", "qg", "nr", "ns", "ng", "thp"):
        stack = np.stack([m[name] for m in members]).astype(np.float64)
        spread = stack.std(axis=0, ddof=1)
        assert float(spread.max()) > 0.0, f"{name} has no ensemble spread"
    # ... and every member is still a state the scheme can evaluate.
    for member in members:
        report = moment_consistency_report(member, mp_physics=10)
        assert report["consistent"], report
        assert report["offending_cells_total"] == 0
        assert report["nonfinite_cells_total"] == 0
    # The species records say so themselves, per member.
    for provenance in hydro_world.perturbation:
        assert provenance["species"], "no species were perturbed"
        for record in provenance["species"]:
            assert record["depleted_pairs_created"] == 0
            assert record["negative_points"] == 0
            assert record["active_points"] > 0
            assert record["factor_min"] > 0.0


def _positivity_parts(receipt):
    """The positivity receipt as the list of passes that ran.

    Since 69505529f (DA lane 3's field rules) a card analysis bounds the
    joint solve on the device and then bounds again the fields the rules
    replaced, and the receipt is ``{"device": ..., "after_field_rules":
    ...}``; the host route still writes one flat receipt.  Every pass
    carries its own ``policy`` and mass accounting, so each is checked.
    """
    if receipt is None:
        return []
    if "policy" in receipt:
        return [receipt]
    assert set(receipt) == {"device", "after_field_rules"}, sorted(receipt)
    return [receipt["device"], receipt["after_field_rules"]]


def _hydro_config(**overrides):
    from woof.da.moments import analysis_fields

    carried = ("u", "v", "w", "thp", "qv", "p",
               "qr", "nr", "qs", "ns", "qg", "ng")
    fields = tuple(name for name in analysis_fields(10) if name in carried)
    kwargs = dict(
        localization=Localization(horizontal_m=H_LOC_M, vertical_m=V_LOC_M),
        rtps_alpha=0.9, analysis_fields=fields, velocity=True,
        reflectivity=True, positivity_policy="clip", mp_physics=10)
    kwargs.update(overrides)
    return RadarAssimilationConfig(**kwargs)


@pytest.fixture(scope="module")
def hydro_analysis(hydro_world, hydro_grid):
    checkpoints = {
        index: member_background_checkpoint(info["member_dir"])
        for index, info in hydro_world.member_states.items()}
    return assimilate_radar_grid(
        checkpoints, hydro_world.obs_path, hydro_grid, _hydro_config(),
        reflectivity_provider=hydro_world.provider)


def test_hydro_analysis_updates_the_scheme_whole_moment_set(hydro_analysis):
    increments, provenance = hydro_analysis
    analysed = set(provenance["analysis_fields"])
    assert {"thp", "qv", "u", "v", "qr", "nr", "qs", "ns", "qg",
            "ng"} <= analysed
    receipt = provenance["moment_policy"]
    assert receipt["policy"] == "full-moment"
    assert receipt["repair_required"] is False
    assert set(receipt["pairs_updated"]) == {"qr", "qs", "qg"}
    assert receipt["mass_only_species"] == []


def test_hydro_analysis_beats_the_prior_against_truth(hydro_world,
                                                      hydro_analysis):
    """The twin-experiment skill gate, on the new variables.

    Scored inside the observed region only: outside it the filter is
    contractually the identity, so including it would dilute the very
    quantity being measured with points nobody claimed to improve.
    """
    increments, _ = hydro_analysis
    members = {index: read_checkpoint_state(
        member_background_checkpoint(info["member_dir"]))
        for index, info in hydro_world.member_states.items()}
    inside = hydro_world.interior
    improved = {}
    for name in ("qr", "qs", "qg", "qv"):
        truth = hydro_world.truth[name]
        prior = np.stack([members[i][name] for i in sorted(members)])
        analysis = prior + np.stack([increments[i][name]
                                     for i in sorted(increments)])
        prior_err = float(np.sqrt(np.mean(
            (prior.mean(axis=0) - truth)[inside] ** 2)))
        post_err = float(np.sqrt(np.mean(
            (analysis.mean(axis=0) - truth)[inside] ** 2)))
        improved[name] = (prior_err, post_err)
    # Rain is what the reflectivity operator is most sensitive to and is
    # the species the gate is asserted on; the rest are reported so a
    # regression shows which one moved.
    prior_err, post_err = improved["qr"]
    assert post_err < prior_err, improved


def test_hydro_increments_are_exactly_zero_outside_localization(
        hydro_analysis):
    """The guarantee that has survived every run so far, on 10 fields.

    Not "small" -- the literal ``0.0``.  With prior_inflation = 1 the
    inactive-point transform is the exact identity, so a corner outside
    every observation's lens must come back bitwise unchanged for every
    analysed field, hydrometeors included.
    """
    increments, provenance = hydro_analysis
    for member in increments.values():
        for name, values in member.items():
            corner = values[:, :2, :2]
            assert np.array_equal(corner, np.zeros_like(corner)), name
    # And every mass-shaped field reached exactly the same gridpoints.
    mass = [n for n in provenance["analysis_fields"] if n not in ("u", "v")]
    support = None
    for name in mass:
        stack = np.stack([increments[i][name] for i in sorted(increments)])
        touched = np.any(stack != 0.0, axis=0)
        if support is None:
            support = touched
        else:
            assert np.array_equal(support, touched), name
    assert int(support.sum()) < support.size


def test_hydro_analysis_is_non_negative_and_moment_consistent(
        hydro_world, hydro_analysis):
    """Criterion (d): the analysed state is one the scheme can evaluate."""
    from woof.da.moments import moment_consistency_report

    increments, provenance = hydro_analysis
    parts = _positivity_parts(provenance["positivity"])
    assert parts
    for positivity in parts:
        assert positivity["policy"] == "clip"
        assert positivity["mass_left_negative"] == 0.0
    for index, member in increments.items():
        background = read_checkpoint_state(member_background_checkpoint(
            hydro_world.member_states[index]["member_dir"]))
        analysis = dict(background)
        for name, values in member.items():
            analysis[name] = np.asarray(background[name],
                                        np.float64) + values
        for name in ("qv", "qr", "qs", "qg", "nr", "ns", "ng"):
            assert float(np.min(analysis[name])) >= 0.0, (index, name)
            assert np.all(np.isfinite(analysis[name])), (index, name)
        report = moment_consistency_report(analysis, mp_physics=10)
        assert report["consistent"], (index, report)


def test_hydro_analysis_applies_through_the_shipped_applier(
        hydro_world, hydro_analysis, tmp_path):
    """The applier's own moment guard has nothing to repair."""
    from woof.ensemble.increments import apply_increments_to_checkpoint

    increments, _ = hydro_analysis
    source = member_background_checkpoint(
        hydro_world.member_states[0]["member_dir"])
    receipt = apply_increments_to_checkpoint(
        source, increments[0], tmp_path / "analysis.npz", mp_physics=10)
    assert receipt["moments"]["consistent"] is True
    assert receipt["moments"]["offending_cells_total"] == 0
    assert receipt["moments"]["nonfinite_cells_total"] == 0
    assert receipt["moments"]["repaired_cells_total"] == 0


def test_hydro_prior_and_posterior_spread_are_both_reported(hydro_analysis):
    """The instrument the spread diagnosis needs, present and paired."""
    _, provenance = hydro_analysis
    filt = provenance["filter"]
    assert set(filt["prior_spread"]) == set(filt["posterior_spread"])
    assert set(filt["prior_spread"]) == set(provenance["analysis_fields"])
    for name, prior in filt["prior_spread"].items():
        assert prior > 0.0, name
    assert filt["relaxation"] == "rtps"
    assert filt["rtps_alpha"] == 0.9


def test_the_receipt_names_the_eigensolver_that_produced_it(analysis):
    """A cycle report has to say which solver factored its matrices.

    The bundled Jacobi kernel is the default, so a device run stopped
    calling cuSOLVER the moment that landed.  The two agree to ~1e-11
    relative and NOT bitwise, which means every receipt banked before the
    change reproduces only under ``eigensolver='library'``.  A receipt that
    does not name its solver cannot be reproduced on purpose.

    This fixture solves on the host, where the kernel does not apply, so
    the accurate answer here is the library solver and no sweeps.
    """
    _, provenance = analysis
    filt = provenance["filter"]
    assert filt["eigensolver"] == "library"
    assert filt["max_jacobi_sweeps"] == 0


def test_rtpp_is_available_and_differs_from_rtps(
        hydro_world, hydro_grid, hydro_analysis):
    """Both relaxations run; at the same alpha they are not the same.

    The ensemble MEAN increment is relaxation-independent -- neither
    scheme touches ``wbar`` -- while the individual members are not.
    That pair of facts is what makes the knob a spread knob rather than
    a second analysis.
    """
    checkpoints = {
        index: member_background_checkpoint(info["member_dir"])
        for index, info in hydro_world.member_states.items()}
    rtpp, prov = assimilate_radar_grid(
        checkpoints, hydro_world.obs_path, hydro_grid,
        _hydro_config(relaxation="rtpp"),
        reflectivity_provider=hydro_world.provider)
    assert prov["relaxation"] == "rtpp"
    assert prov["filter"]["relaxation"] == "rtpp"
    rtps, _ = hydro_analysis
    mean_rtpp = np.mean([rtpp[i]["thp"] for i in sorted(rtpp)], axis=0)
    mean_rtps = np.mean([rtps[i]["thp"] for i in sorted(rtps)], axis=0)
    assert np.allclose(mean_rtpp, mean_rtps, atol=1e-10)
    assert not np.allclose(rtpp[0]["thp"], rtps[0]["thp"], atol=1e-10)


def test_reflectivity_thinning_reduces_the_assimilated_count(hydro_world,
                                                             hydro_grid):
    checkpoints = {
        index: member_background_checkpoint(info["member_dir"])
        for index, info in hydro_world.member_states.items()}
    _, provenance = assimilate_radar_grid(
        checkpoints, hydro_world.obs_path, hydro_grid,
        # the horizontal thinning alone: no classes, every level, no top
        # (woof.da.radar_classes has its own counts in test_da_radar_classes)
        _hydro_config(reflectivity_thinning_cells=2,
                      reflectivity_error_inflation=3.0,
                      reflectivity_clear_floor_dbz=None,
                      reflectivity_level_stride=1, radar_top_pa=None),
        reflectivity_provider=hydro_world.provider)
    receipt = provenance["reflectivity_thinning"]
    assert receipt["cells"] == 2
    assert receipt["error_inflation"] == 3.0
    assert 0 < receipt["points_after"] < receipt["points_before"]
    (z_batch,) = [entry for entry in provenance["innovations"]
                  if entry["name"] == "z"]
    assert z_batch["observations"] == receipt["points_after"]
    assert z_batch["obs_error_mean"] == pytest.approx(Z_ERR_DBZ * 3.0)


# ---------------------------------------------------------------------------
# the guards, watched FIRING
#
# The tests above watch the moment and positivity guards stay quiet on a
# healthy analysis, which is the outcome the design is for and is also
# the outcome a guard that had been accidentally disabled would produce.
# These four make each one fire on purpose and check the number it
# reports, so "clean" upstream means "clean" and not "asleep".
# ---------------------------------------------------------------------------


def test_positivity_clip_fires_and_is_counted(hydro_world, hydro_grid):
    """Clip at zero ADDS mass; the count and the mass are the diagnostic.

    Forced by shrinking the observation error until the filter proposes
    increments larger than the background it is correcting, which is
    exactly the regime a badly calibrated sigma_o puts a real run in.
    The point is not that the analysis is good -- it is not -- but that
    the policy fires, is counted, and leaves nothing negative behind.
    """
    from woof.da.positivity import apply_positivity, verify_non_negative

    checkpoints = {
        index: member_background_checkpoint(info["member_dir"])
        for index, info in hydro_world.member_states.items()}
    increments, provenance = assimilate_radar_grid(
        checkpoints, hydro_world.obs_path, hydro_grid,
        _hydro_config(reflectivity_error_inflation=1.0,
                      positivity_policy="none"),
        reflectivity_provider=hydro_world.provider)
    # With no policy, negatives are counted and LEFT -- a stated choice.
    for receipt in _positivity_parts(provenance["positivity"]):
        assert receipt["policy"] == "none"

    prior = {}
    for name in ("qr", "qs", "qg", "qv"):
        prior[name] = np.stack([
            read_checkpoint_state(member_background_checkpoint(
                hydro_world.member_states[i]["member_dir"]))[name]
            for i in sorted(hydro_world.member_states)]).astype(np.float64)
    stacked = {name: np.stack([increments[i][name]
                               for i in sorted(increments)])
               for name in prior}
    negatives = sum(int(np.count_nonzero(prior[n] + stacked[n] < 0.0))
                    for n in prior)
    assert negatives > 0, (
        "this fixture no longer drives any analysis negative, so it can no "
        "longer prove the clip fires; tighten sigma_o further")

    clipped, clip_receipt = apply_positivity(prior, stacked, policy="clip")
    assert clip_receipt["negative_points"] == negatives
    assert clip_receipt["mass_added_by_clip"] > 0.0
    assert clip_receipt["mass_left_negative"] == 0.0
    verify_non_negative(prior, clipped)
    for name in prior:
        assert float((prior[name] + clipped[name]).min()) >= 0.0
    # ... and the wetward bias is real: clipping only ever adds.
    for entry in clip_receipt["per_field"]:
        assert entry["mass_added_by_clip"] >= 0.0


def test_moment_guard_refuses_a_truncated_field_set_before_the_solve(
        hydro_world, hydro_grid):
    """A mass field without its number moment, refused at configuration.

    This is the test host 8 defect: nine fields, no number concentrations,
    reflectivity assimilation creating rain in cells the background left
    clear, and Morrison's slope closure evaluating to NaN.  Under
    full-moment it is now refused before a single H(x) is computed,
    naming every species.
    """
    from woof.da.moments import MomentPolicyError

    checkpoints = {
        index: member_background_checkpoint(info["member_dir"])
        for index, info in hydro_world.member_states.items()}
    truncated = ("u", "v", "thp", "qv", "qr", "qs", "qg")   # no nr/ns/ng
    with pytest.raises(MomentPolicyError, match="qr without nr"):
        assimilate_radar_grid(
            checkpoints, hydro_world.obs_path, hydro_grid,
            _hydro_config(analysis_fields=truncated),
            reflectivity_provider=hydro_world.provider)


def test_a_multiplicative_ensemble_cannot_break_a_pair_even_mass_only(
        hydro_world, hydro_grid, tmp_path):
    """The truncation is legal once declared -- and here it is harmless.

    Under ``single-moment-with-repair`` the same field set the previous
    test refuses is allowed, and on a hand-built ensemble it is exactly
    how ``q > 0`` lands beside ``N = 0``.  On THIS ensemble it cannot,
    and the reason is a property of the perturbation rather than luck:
    the filter's increment lives in the span of the prior perturbations,
    a multiplicative perturbation is exactly zero where the species is
    absent, so every member agrees the cell is clear and the analysis
    increment there is bitwise zero.  A mass-only update can therefore
    only rescale condensate that already has a number moment.

    That is worth pinning, because it is the reason the run's
    ``moment_policy.repair_required`` is expected to stay False and not
    a claim that the guard is unnecessary -- the guard fires on a
    hand-built offender in ``tests/test_da_moments.py``, which is where
    the scheme's limiter is exercised directly.
    """
    from woof.da.moments import moment_consistency_report
    from woof.ensemble.increments import apply_increments_to_checkpoint

    checkpoints = {
        index: member_background_checkpoint(info["member_dir"])
        for index, info in hydro_world.member_states.items()}
    truncated = ("u", "v", "thp", "qv", "qr", "qs", "qg")
    increments, provenance = assimilate_radar_grid(
        checkpoints, hydro_world.obs_path, hydro_grid,
        _hydro_config(analysis_fields=truncated,
                      moment_policy="single-moment-with-repair",
                      reflectivity_error_inflation=1.0),
        reflectivity_provider=hydro_world.provider)
    receipt = provenance["moment_policy"]
    assert receipt["policy"] == "single-moment-with-repair"
    assert receipt["repair_required"] is True
    assert set(receipt["mass_only_species"]) == {"qr", "qs", "qg"}

    for index in sorted(increments):
        background = read_checkpoint_state(member_background_checkpoint(
            hydro_world.member_states[index]["member_dir"]))
        for mass, number in (("qr", "nr"), ("qs", "ns"), ("qg", "ng")):
            # Where every member is clear the increment is the literal
            # 0.0, so the mass-only update cannot promote a clear cell.
            clear = background[number] <= 0.0
            assert np.all(increments[index][mass][clear] == 0.0), (
                index, mass)
        applied = apply_increments_to_checkpoint(
            source_of := member_background_checkpoint(
                hydro_world.member_states[index]["member_dir"]),
            increments[index], tmp_path / f"a{index}.npz",
            moment_policy="single-moment-with-repair", mp_physics=10)
        assert applied["moments"]["offending_cells_total"] == 0, index
        assert applied["moments"]["nonfinite_cells_total"] == 0, index
        assert applied["moments"]["repaired_cells_total"] == 0, index
        assert str(source_of).endswith(".npz")

    # And the same field set with the repair switched off is therefore
    # NOT a refusal here: there is nothing broken to refuse.
    applied = apply_increments_to_checkpoint(
        member_background_checkpoint(
            hydro_world.member_states[0]["member_dir"]),
        increments[0], tmp_path / "no-repair.npz",
        moment_policy="single-moment-with-repair", moment_repair=False,
        mp_physics=10)
    assert applied["moments"]["consistent"] is True
    state = read_checkpoint_state(tmp_path / "no-repair.npz")
    assert moment_consistency_report(state, mp_physics=10)["consistent"]


@pytest.mark.parametrize("bounded", [True, False])
def test_default_cuda_residency_and_custom_callback_contract(world, grid, monkeypatch, bounded):
    import sys
    from woof.da import radar_assimilation as owner
    calls = []
    class Namespace:
        @staticmethod
        def asarray(values):
            array = np.asarray(values)
            calls.append(array.shape)
            return array
        asnumpy = staticmethod(np.asarray)
    namespace = Namespace()
    monkeypatch.setitem(sys.modules, "cupy", namespace)
    monkeypatch.setattr(owner, "resolve_solve_device", lambda mode: ("cuda", "selected"))
    def solver(prior, batches, geometry, config, diagnostics, **options):
        assert all(isinstance(v, np.ndarray) for v in prior.values())
        if bounded:
            assert 'solve_namespace' not in options
            assert callable(options['progress'])
            assert any(len(shape) == 4 for shape in calls)
        else:
            assert not options
            assert any(len(shape) == 4 for shape in calls)
        return {name: np.zeros_like(v) for name, v in prior.items()}
    solver.supports_host_staging = bounded
    if bounded:
        monkeypatch.setattr(owner, "analyze", solver)
    paths = {member: member_background_checkpoint(state['member_dir'])
             for member, state in world.member_states.items()}
    owner.assimilate_radar_grid(paths, world.obs_path, grid, _config(),
                               analysis_runner=None if bounded else solver)


def test_the_analysis_price_counts_what_the_resident_solve_receives(
        world, grid, monkeypatch):
    """The admission's analysis price and the solve's inputs agree.

    ``analysis_device_price`` is what the DA cycle admits the analysis
    with before its first upload.  Here the real ``assimilate_radar_grid``
    runs the resident route with the upload stood in by numpy, and the
    solver records the prior and batches it is handed: their bytes are
    the input share of the price, its stencil slots are the ones the
    solve's own stencils count, and its chunk is the solve's own sizing.
    """
    import math
    import sys

    from woof.da import radar_assimilation as owner
    from woof.da.letkf import (chunk_points_for_budget,
                                solve_bytes_per_point, stencil_slots)
    from woof.da.obs_radar import letkf_grid_geometry, read_document

    class Namespace:
        asarray = staticmethod(np.asarray)
        asnumpy = staticmethod(np.asarray)

    monkeypatch.setitem(sys.modules, "cupy", Namespace())
    monkeypatch.setattr(owner, "resolve_solve_device",
                        lambda mode: ("cuda", "selected"))
    seen = {}

    def solver(prior, batches, geometry, config, diagnostics, **options):
        seen.update(prior=prior, batches=batches, config=config)
        return {name: np.zeros_like(v) for name, v in prior.items()}

    cfg = _config(solve_device="cuda", memory_budget_mib=1.0)
    paths = {member: member_background_checkpoint(state["member_dir"])
             for member, state in world.member_states.items()}
    owner.assimilate_radar_grid(paths, world.obs_path, grid, cfg,
                                analysis_runner=solver)
    price = owner.analysis_device_price(
        cfg, members=MEMBERS, grid=grid,
        document=read_document(world.obs_path, expected_grid=grid))

    prior_bytes = sum(v.nbytes for v in seen["prior"].values())
    batch_bytes = sum(b.values.nbytes + b.errors.nbytes + b.simulated.nbytes
                      + b.mask.nbytes for b in seen["batches"])
    assert len(seen["batches"]) == 2
    npts = NZ * NY * NX
    fields = len(cfg.analysis_fields)
    observed = sum(b.mask.size for b in seen["batches"])
    derived = (observed * ((2 + MEMBERS) * 8 + 1) + observed * 8
               + fields * npts * 8 * (2 * MEMBERS + 1)
               + 8 * npts + 2 * 8 * NY * NX)
    assert price.setup_bytes == prior_bytes + batch_bytes + derived
    geometry = letkf_grid_geometry(grid)
    slots = 2 * stencil_slots(cfg.localization, geometry, NX, NY)
    assert price.stencil_slots == slots
    budget = int(seen["config"].memory_budget_mib * (1 << 20))
    assert price.budget_bytes == budget
    assert price.chunk_points == chunk_points_for_budget(
        slots, MEMBERS, 8, budget, npts)
    per_point = solve_bytes_per_point(slots, MEMBERS, 8)
    assert price.solve_bytes_per_point == per_point
    # The budget binds on this grid, so the solve fills it.
    assert 0 < price.chunk_points < npts
    assert price.scratch_bytes == budget
    assert price.resident_bytes == price.setup_bytes + max(
        price.finish_bytes, math.ceil(budget / 0.8))
    assert price.reduced_bytes == price.setup_bytes + max(
        price.finish_bytes, math.ceil(per_point / 0.8))


def test_a_host_solve_puts_no_analysis_on_the_card(world, grid):
    from woof.da.obs_radar import read_document
    from woof.da.radar_assimilation import analysis_device_price

    assert analysis_device_price(
        _config(solve_device="host"), members=MEMBERS, grid=grid,
        document=read_document(world.obs_path, expected_grid=grid)) is None


# ---------------------------------------------------------------------------
# member backgrounds without a host round trip (DA analysis data movement)
# ---------------------------------------------------------------------------


def test_views_and_in_memory_states_give_byte_identical_increments(
        hydro_world, hydro_grid, hydro_analysis):
    """The cycle driver no longer saves a whole-state mirror of every
    member to a second npz and reads it back with ``fields=None``: it hands
    the analysis a lazy view of each member's own restart, or the live
    mirror itself.  Neither may move a single bit of the analysis."""
    from woof.da.radar_assimilation import CheckpointStateView

    paths = {index: member_background_checkpoint(info["member_dir"])
             for index, info in hydro_world.member_states.items()}
    views = {index: CheckpointStateView(path)
             for index, path in paths.items()}
    in_memory = {index: read_checkpoint_state(path)
                 for index, path in paths.items()}
    reference, reference_prov = hydro_analysis
    try:
        for states in (views, in_memory):
            increments, provenance = assimilate_radar_grid(
                states, hydro_world.obs_path, hydro_grid, _hydro_config(),
                reflectivity_provider=hydro_world.provider)
            assert set(increments) == set(reference)
            for index, member in reference.items():
                assert set(increments[index]) == set(member)
                for name, values in member.items():
                    assert increments[index][name].dtype == values.dtype
                    assert increments[index][name].tobytes() == \
                        values.tobytes(), (index, name)
            assert provenance["innovations"] == reference_prov["innovations"]
        # Every field is read once and then served from the view's cache.
        assert views[0]["qv"] is views[0]["qv"]
        assert reference_prov["checkpoints"][0] == paths[0].name
    finally:
        for view in views.values():
            view.close()
    with pytest.raises(RadarAssimilationError, match="closed"):
        views[0]["thp"]


def test_checkpoint_view_is_a_read_only_mapping_of_state_arrays(tmp_path):
    from woof.da.radar_assimilation import CheckpointStateView

    path = tmp_path / "gpuwmrst_d01_000600.npz"
    u = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    np.savez(path, **{"state/u": u, "state/qv": u * 2,
                      "driver/rainnc": u[0], "header": np.zeros(3, np.uint8)})
    view = CheckpointStateView(path)
    assert sorted(view) == ["qv", "u"] and len(view) == 2
    assert "u" in view and "rainnc" not in view
    assert view["u"].tobytes() == u.tobytes()
    assert view.cached_bytes == u.nbytes
    with pytest.raises(KeyError):
        view["rainnc"]
    assert view.name == path.name
    # Stored members are mapped read-only from the file, not copied.
    assert not view["qv"].flags.writeable
    with pytest.raises(ValueError):
        view["qv"][0, 0, 0] = 1.0
    view.close()
    assert view.cached_bytes == 0
    with pytest.raises(RadarAssimilationError, match="no checkpoint"):
        CheckpointStateView(tmp_path / "missing.npz")


def test_checkpoint_view_maps_every_layout_and_falls_back_when_compressed(
        tmp_path):
    from woof.da.radar_assimilation import CheckpointStateView

    rng = np.random.default_rng(9)
    arrays = {"state/c": rng.normal(size=(3, 4, 5)).astype(np.float32),
              "state/f": np.asfortranarray(rng.normal(size=(2, 3))),
              "state/i": np.arange(7, dtype=np.int16),
              "state/s": np.float64(2.5),
              "state/e": np.zeros((0, 3), np.float32)}
    for name, writer in (("stored.npz", np.savez),
                         ("deflated.npz", np.savez_compressed)):
        path = tmp_path / name
        writer(path, **arrays)
        view = CheckpointStateView(path)
        for key, value in arrays.items():
            got = view[key[len("state/"):]]
            want = np.asarray(value)
            assert got.dtype == want.dtype and got.shape == want.shape
            assert np.array_equal(got, want)
            assert got.tobytes(order="A") == want.tobytes(order="A")
        view.close()


def test_in_memory_member_states_are_the_same_analysis_as_checkpoint_files(
        world, grid):
    """The cycle driver hands its held snapshots straight to the analysis
    instead of a savez/load round trip; the answer must not move a bit."""
    checkpoints = {
        index: member_background_checkpoint(info["member_dir"])
        for index, info in world.member_states.items()}
    states = {index: read_checkpoint_state(path)
              for index, path in checkpoints.items()}
    before = {index: {name: value.copy() for name, value in state.items()}
              for index, state in states.items()}
    from_files, prov_files = assimilate_radar_grid(
        checkpoints, world.obs_path, grid, _config())
    from_memory, prov_memory = assimilate_radar_grid(
        states, world.obs_path, grid, _config())
    assert sorted(from_files) == sorted(from_memory)
    for index in from_files:
        assert sorted(from_files[index]) == sorted(from_memory[index])
        for name in from_files[index]:
            a, b = from_files[index][name], from_memory[index][name]
            assert a.dtype == b.dtype and a.shape == b.shape
            assert a.tobytes() == b.tobytes(), (index, name)
    # Read, never written: the caller's arrays come back untouched.
    for index, state in states.items():
        for name, value in state.items():
            assert value.tobytes() == before[index][name].tobytes()
    assert all(label.startswith("in-memory member")
               for label in prov_memory["checkpoints"].values())
    def untimed(record):
        if not isinstance(record, dict):
            return record
        return {key: untimed(value) for key, value in record.items()
                if not key.endswith("_seconds")}
    assert untimed(prov_files["filter"]) == untimed(prov_memory["filter"])
