# tests/test_uh_lifecycle.py
"""UP_HELI_MAX lifecycle: allocation, emission, reset, restart, inertness.

WRF semantics under test (authorities cited in woof/core/uh_diag.py):
the accumulator exists exactly when nwp_diagnostics = 1, is folded every
step, rides in every history frame, is zeroed after each frame ("reset at
history interval"), is carried through restarts (Registry IO "rh02"), and
-- the essential property -- is provably trajectory-inert: a run with
the knob on is bitwise identical to the same run with it off in every
model field, differing only by the UP_HELI_MAX plane itself.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import RunConfig
from woof.core.uh_diag import reset_up_heli_max, update_up_heli_max
from woof.io import restart

from test_restart import (_cfg, _fill_serialized, _fill_setup,
                          _identity_bound_physics_state, _shim_state)

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# CPU tier
# ---------------------------------------------------------------------------

def test_state_allocates_the_accumulator_eagerly_under_the_knob(monkeypatch):
    on = _shim_state(_cfg(nwp_diagnostics=1), monkeypatch)
    assert on.existing_scratch("up_heli_max") is not None
    assert on.existing_scratch("up_heli_max").shape == (4, 6)
    off = _shim_state(_cfg(), monkeypatch)
    assert off.existing_scratch("up_heli_max") is None


def test_reset_zeroes_and_tolerates_an_absent_slot(monkeypatch):
    state = _shim_state(_cfg(nwp_diagnostics=1), monkeypatch)
    state.existing_scratch("up_heli_max")[...] = 123.0
    reset_up_heli_max(state)
    assert (state.existing_scratch("up_heli_max") == 0.0).all()
    # No slot (knob off) is a clean no-op, not an error.
    reset_up_heli_max(_shim_state(_cfg(), monkeypatch))


def test_wrfout_meta_and_shared_field_builder_carry_up_heli_max(monkeypatch):
    from woof.io.wrfout import _VAR_META, _live_state_history_fields

    # WRF Registry.EM_COMMON:2083 attributes, verbatim.
    assert _VAR_META["UP_HELI_MAX"] == ("MAX UPDRAFT HELICITY", "m2 s-2")
    state = _shim_state(_cfg(nwp_diagnostics=1), monkeypatch)
    buf = state.existing_scratch("up_heli_max")
    buf[...] = 7.5
    fields = _live_state_history_fields(state)
    assert fields["UP_HELI_MAX"] is buf     # live view, snapshot at write
    off = _shim_state(_cfg(), monkeypatch)
    assert "UP_HELI_MAX" not in _live_state_history_fields(off)


def test_tree_history_handler_snapshots_then_resets(monkeypatch):
    """Reset-on-write: the production tree handler zeroes the accumulator
    AFTER the frame submit observed the accumulated values."""
    from woof.runtime import _submit_tree_history_frame

    state = _shim_state(_cfg(nwp_diagnostics=1), monkeypatch)
    state.existing_scratch("up_heli_max")[...] = 42.0
    seen = {}

    class _Writers:
        def submit(self, node, ticks, *, refl_field=None):
            seen["at_submit"] = node.state.existing_scratch(
                "up_heli_max").copy()

    node = SimpleNamespace(state=state)
    _submit_tree_history_frame(_Writers(), node, 0)
    assert (seen["at_submit"] == 42.0).all()
    assert (state.existing_scratch("up_heli_max") == 0.0).all()


def test_restart_roundtrip_carries_the_running_max(monkeypatch, tmp_path):
    cfg = _cfg(nwp_diagnostics=1)
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    pattern = np.arange(24, dtype=np.float32).reshape(4, 6) * np.float32(1.5)
    source.existing_scratch("up_heli_max")[...] = pattern
    path = restart.write_restart(tmp_path / "uh.npz", source, cfg)

    with np.load(path, allow_pickle=False) as archive:
        assert "scratch/up_heli_max" in archive.files

    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    restart.restore_restart(path, fresh, cfg)
    np.testing.assert_array_equal(
        fresh.existing_scratch("up_heli_max"), pattern)


def test_old_checkpoint_without_the_slot_restores_zeroed_with_a_note(
        monkeypatch, tmp_path, capsys):
    """A checkpoint predating the diagnostic restores under
    nwp_diagnostics=1 with the accumulator zeroed -- a note, NOT a
    refusal.  The header written by the old build also lacked the config
    key entirely, which the diagnostic-field echo exemption covers."""
    import json

    old_cfg = _cfg()
    source = _shim_state(old_cfg, monkeypatch)
    _fill_setup(source)
    path = restart.write_restart(tmp_path / "old.npz", source, old_cfg)

    # Age the header: strip the config key a pre-knob build never wrote.
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
        assert "scratch/up_heli_max" not in archive.files
    header = json.loads(bytes(bytearray(
        payload[restart._HEADER_KEY])).decode("utf-8"))
    del header["config"]["nwp_diagnostics"]
    payload[restart._HEADER_KEY] = np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)
    aged = tmp_path / "pre-knob.npz"
    with aged.open("wb") as stream:
        np.savez(stream, **payload)

    live_cfg = _cfg(nwp_diagnostics=1)
    live = _shim_state(live_cfg, monkeypatch)
    _fill_setup(live)
    live.existing_scratch("up_heli_max")[...] = 99.0   # must be wiped
    restart.restore_restart(aged, live, live_cfg)
    assert (live.existing_scratch("up_heli_max") == 0.0).all()
    out = capsys.readouterr().out
    assert "note:" in out and "up_heli_max" in out
    assert "zero-initialized" in out


def test_diagnostics_off_resume_drops_the_stored_accumulator_with_a_note(
        monkeypatch, tmp_path, capsys):
    on_cfg = _cfg(nwp_diagnostics=1)
    source = _shim_state(on_cfg, monkeypatch)
    _fill_setup(source)
    source.existing_scratch("up_heli_max")[...] = 3.25
    path = restart.write_restart(tmp_path / "on.npz", source, on_cfg)

    off_cfg = _cfg()
    live = _shim_state(off_cfg, monkeypatch)
    _fill_setup(live)
    restart.restore_restart(path, live, off_cfg)
    assert live.existing_scratch("up_heli_max") is None
    out = capsys.readouterr().out
    assert "note:" in out and "dropped" in out


@pytest.mark.parametrize("written,resumed", [(0, 1), (1, 0)],
                         ids=["switched-on", "switched-off"])
def test_physics_resume_flipping_nwp_diagnostics_keeps_every_other_plane(
        monkeypatch, tmp_path, capsys, written, resumed):
    """V2W WoFS, 2026-10-29: a forecast branched from an analysis checkpoint
    with nwp_diagnostics flipped refused "restart surface-field inventory
    ... extra ['refd_max']".  REFD_MAX is a zero-held driver plane exactly
    when the NWP family is off; a flip drops it (on) or initialises it (off)
    with a note, and every other driver plane restores word for word."""
    physics = dict(moist=True, mp_physics=10, morr_rimed_ice=1,
                   sf_sfclay_physics=1, sf_surface_physics=2,
                   bl_pbl_physics=1, ra_physics=4, cu_physics=1)
    source_cfg = _cfg(nwp_diagnostics=written, **physics)
    source, source_driver = _identity_bound_physics_state(
        source_cfg, monkeypatch)
    _fill_serialized(source, seed=20261029)
    for index, (name, plane) in enumerate(sorted(source_driver.fields.items())):
        plane[...] = np.float32(index + 1)
    assert ("refd_max" in source_driver.fields) == (written == 0)
    path = restart.write_restart(tmp_path / "flip.npz", source, source_cfg)

    live_cfg = _cfg(nwp_diagnostics=resumed, **physics)
    live, live_driver = _identity_bound_physics_state(live_cfg, monkeypatch)
    if "refd_max" in live_driver.fields:
        live_driver.fields["refd_max"][...] = 7.0      # must be zeroed
    restart.restore_restart(path, live, live_cfg)

    assert ("refd_max" in live_driver.fields) == (resumed == 0)
    if resumed == 0:
        assert (live_driver.fields["refd_max"] == 0.0).all()
    shared = set(source_driver.fields) - {"refd_max"}
    assert shared == set(live_driver.fields) - {"refd_max"}
    for name in shared:
        np.testing.assert_array_equal(
            live_driver.fields[name], source_driver.fields[name], err_msg=name)
    out = capsys.readouterr().out
    assert "fields/refd_max" in out
    assert ("dropped" if written == 0 else "initialised to zero") in out
    assert (live.existing_scratch("up_heli_max") is not None) == (resumed == 1)


def test_physics_resume_refuses_a_refd_max_plane_its_header_disowns(
        monkeypatch, tmp_path):
    """The tolerance is the switch's, not a blanket one: a checkpoint whose
    REFD_MAX presence contradicts the nwp_diagnostics its own header records
    still refuses on the exact inventory check."""
    import json

    physics = dict(moist=True, mp_physics=10, morr_rimed_ice=1,
                   sf_sfclay_physics=1, sf_surface_physics=2,
                   bl_pbl_physics=1, ra_physics=4, cu_physics=1)
    cfg = _cfg(nwp_diagnostics=0, **physics)
    source, _ = _identity_bound_physics_state(cfg, monkeypatch)
    path = restart.write_restart(tmp_path / "off.npz", source, cfg)
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    header = json.loads(bytes(bytearray(
        payload[restart._HEADER_KEY])).decode("utf-8"))
    header["config"]["nwp_diagnostics"] = 1
    payload[restart._HEADER_KEY] = np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)
    forged = tmp_path / "forged.npz"
    with forged.open("wb") as stream:
        np.savez(stream, **payload)

    live_cfg = _cfg(nwp_diagnostics=1, **physics)
    live, _ = _identity_bound_physics_state(live_cfg, monkeypatch)
    with pytest.raises(restart.RestartMismatchError,
                       match=r"surface-field inventory.*extra \['refd_max'\]"):
        restart.restore_restart(forged, live, live_cfg)


def test_launcher_refuses_unsupported_geometry_and_tiny_grids(monkeypatch):
    # Periodic idealized geometry: WRF's periodic cal_helicity branch is
    # not transcribed; refusing beats guessing.
    state = _shim_state(_cfg(nwp_diagnostics=1), monkeypatch)
    with pytest.raises(NotImplementedError, match="specified/nested"):
        update_up_heli_max(state, _cfg(nwp_diagnostics=1))
    # Stencil floor.
    tiny = _cfg(nwp_diagnostics=1, specified=True, nx=3, ny=8, nz=8)
    with pytest.raises(ValueError, match="nx, ny >= 4"):
        update_up_heli_max(_shim_state(tiny, monkeypatch), tiny)


def _mesocyclone_winds(ny, nx, nz):
    """Coherent cyclonic rotation around (4.5, 3.5) plus westerly shear."""
    xs = np.arange(nx + 1, dtype=np.float32) - 0.5
    ys = np.arange(ny, dtype=np.float32)
    r2u = (xs[None, :] - 4.5)**2 + (ys[:, None] - 3.5)**2
    u = (8.0 - 12.0 * np.exp(-r2u / 6.0) * (ys[:, None] - 3.5)
         )[None].repeat(nz, axis=0).astype(np.float32)
    xs = np.arange(nx, dtype=np.float32)
    ys = np.arange(ny + 1, dtype=np.float32) - 0.5
    r2v = (xs[None, :] - 4.5)**2 + (ys[:, None] - 3.5)**2
    v = (2.0 + 12.0 * np.exp(-r2v / 6.0) * (xs[None, :] - 4.5)
         )[None].repeat(nz, axis=0).astype(np.float32)
    return u, v


def _loaded_numpy_state(monkeypatch, **overrides):
    from woof.core.grid import make_base_state, make_vertical_coord

    cfg = _cfg(nx=9, ny=8, nz=24, ztop=12000.0, specified=True,
               nwp_diagnostics=1, **overrides)
    state = _shim_state(cfg, monkeypatch)
    vc = make_vertical_coord(cfg.nz)
    base = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state.load_base(vc, base)
    return state, cfg


def test_numpy_state_path_accumulates_and_matches_the_mirror(monkeypatch):
    """update_up_heli_max on a NumPy state == the parity-gated mirror, and
    the running max holds when a later step weakens."""
    from woof.core.uh_diag import mirror_up_heli_max_step_np

    state, cfg = _loaded_numpy_state(monkeypatch)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    z_w = (np.asarray(state.phb, dtype=np.float32)
           / np.float32(9.81))[:, None, None]
    blob = np.exp(-(((np.arange(nx) - 4.5)[None, :])**2
                    + ((np.arange(ny) - 3.5)[:, None])**2) / 4.0)
    wz = np.exp(-(((z_w - 3500.0) / 2000.0)**2))
    strong = (18.0 * blob[None, :, :] * wz).astype(np.float32)
    state.u[...], state.v[...] = _mesocyclone_winds(ny, nx, nz)

    mirror_max = np.zeros((ny, nx), dtype=np.float32)

    def mirror_step():
        mirror_up_heli_max_step_np(
            state.u, state.v, state.w, state.php, state.phb,
            state.msfu, state.msfv, state.ht,
            state.dn, state.dnw, state.fnm, state.fnp,
            state.cf1, state.cf2, state.cf3,
            np.float32(1.0) / np.float32(cfg.dx),
            np.float32(1.0) / np.float32(cfg.dy), mirror_max)

    state.w[...] = strong
    update_up_heli_max(state, cfg)
    mirror_step()
    first = state.existing_scratch("up_heli_max").copy()
    np.testing.assert_array_equal(first, mirror_max)
    assert first.max() > 10.0, "fixture updraft produced no UH"

    state.w[...] = np.float32(0.4) * strong     # weaker step: max holds
    update_up_heli_max(state, cfg)
    mirror_step()
    second = state.existing_scratch("up_heli_max")
    np.testing.assert_array_equal(second, mirror_max)
    np.testing.assert_array_equal(second, first)

    # Reset starts a new window: the weak step alone now sets a smaller max.
    reset_up_heli_max(state)
    update_up_heli_max(state, cfg)
    third = state.existing_scratch("up_heli_max")
    assert third.max() < first.max()
    assert third.max() > 0.0


def _updraft(state, cfg):
    """The same Gaussian updraft column the mirror-parity test uses.

    UH is w * vertical vorticity, so rotating winds alone produce a plane
    of exact zeros -- which is what the "must actually rotate" control in
    the tests below is there to catch.
    """
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    z_w = (np.asarray(state.phb, dtype=np.float32)
           / np.float32(9.81))[:, None, None]
    blob = np.exp(-(((np.arange(nx) - 4.5)[None, :])**2
                    + ((np.arange(ny) - 3.5)[:, None])**2) / 4.0)
    wz = np.exp(-(((z_w - 3500.0) / 2000.0)**2))
    state.u[...], state.v[...] = _mesocyclone_winds(ny, nx, nz)
    return (18.0 * blob[None, :, :] * wz).astype(np.float32)


def _step_uh(state, cfg, scale):
    """One diagnostic step at a given updraft strength; folds every window."""
    state.w[...] = (np.float32(scale) * _updraft(state, cfg)).astype(np.float32)
    update_up_heli_max(state, cfg)


def test_the_history_writer_reset_does_not_touch_the_tracker_windows(
        monkeypatch):
    """The decoupling, at the exact line where the coupling lived.

    UP_HELI_MAX is zeroed by the history writer.  Both consumer windows
    are folded from the SAME columns in the same pass and must survive
    that reset untouched, or an output knob is steering the model again.
    """
    state, cfg = _loaded_numpy_state(monkeypatch)
    _step_uh(state, cfg, 1.0)

    diagnostic = state.existing_scratch("up_heli_max")
    follow = state.existing_scratch("uh_follow_window")
    spawn = state.existing_scratch("uh_spawn_window")
    assert diagnostic.max() > 0.0, "the fixture must actually rotate"
    # One pass, three identical maxima.
    assert np.array_equal(follow, diagnostic)
    assert np.array_equal(spawn, diagnostic)

    carried = follow.copy()
    reset_up_heli_max(state)
    assert diagnostic.max() == 0.0, "the history writer zeroes its own"
    assert np.array_equal(follow, carried), \
        "the history writer must not zero the relocation window"
    assert np.array_equal(spawn, carried), \
        "the history writer must not zero the spawn window"


def test_each_consumer_reset_leaves_the_other_windows_alone(monkeypatch):
    """Two windows, no cross-blinding: the whole reason there are two.

    Relocation evaluates on cadence boundaries and spawn on leg
    boundaries, so at any boundary they share, a single window would let
    whichever read first blind the other.
    """
    from woof.core.uh_diag import reset_tracker_window

    state, cfg = _loaded_numpy_state(monkeypatch)
    _step_uh(state, cfg, 1.0)
    carried = state.existing_scratch("up_heli_max").copy()

    reset_tracker_window(state, "uh_follow_window")
    assert state.existing_scratch("uh_follow_window").max() == 0.0
    assert np.array_equal(state.existing_scratch("uh_spawn_window"), carried)
    assert np.array_equal(state.existing_scratch("up_heli_max"), carried)

    reset_tracker_window(state, "uh_spawn_window")
    assert state.existing_scratch("uh_spawn_window").max() == 0.0
    assert np.array_equal(state.existing_scratch("up_heli_max"), carried)

    # The diagnostic is NOT a tracker window and cannot be reset through
    # this door; the history writer owns it.
    with pytest.raises(ValueError, match="not a tracker window"):
        reset_tracker_window(state, "up_heli_max")


@pytest.mark.parametrize("history_every", [2, 3, 5])
def test_the_tracker_window_is_invariant_to_the_history_interval(
        monkeypatch, history_every):
    """THE RULING'S TEST: same evaluation cadence, different output cadence,
    identical signal.

    Twelve steps with a pulsing mesocyclone, evaluated every 6 steps.  The
    history writer fires every ``history_every`` steps, which is what used
    to decide the tracker's answer.  The window the tracker reads at each
    evaluation must be bit-identical across all three output cadences.

    The control that makes this mean something is in the assertion below:
    UP_HELI_MAX itself is NOT invariant here, and the test says so.  If
    both were invariant the fixture would simply not be exercising the
    reset.
    """
    evaluate_every = 6
    # A pulse: the peak lands mid-window and is GONE by the boundary, so a
    # max-over-interval sees it and an instantaneous sample does not.
    pulses = [0.4, 0.7, 1.0, 0.6, 0.3, 0.2] * 2

    def run(history_every):
        from woof.core.uh_diag import reset_tracker_window

        state, cfg = _loaded_numpy_state(monkeypatch)
        seen = []
        for step, scale in enumerate(pulses, start=1):
            _step_uh(state, cfg, scale)
            # Evaluation before the history write, deliberately: the
            # writer snapshots THEN zeroes (runtime._submit_tree_history_
            # frame), so a consumer reading at a boundary sees the
            # pre-reset value.  Resetting first would zero UP_HELI_MAX in
            # BOTH arms whenever history_every divides evaluate_every,
            # and the control below would pass by accident on a fixture
            # that never exercised the reset at all.
            if step % evaluate_every == 0:
                seen.append((
                    state.existing_scratch("uh_follow_window").copy(),
                    state.existing_scratch("up_heli_max").copy()))
                reset_tracker_window(state, "uh_follow_window")
            if step % history_every == 0:
                reset_up_heli_max(state)
        return seen

    reference = run(evaluate_every)      # history aligned to evaluation
    measured = run(history_every)

    assert len(measured) == len(reference) == 2
    for index, ((win, _), (ref_win, _)) in enumerate(
            zip(measured, reference)):
        assert np.array_equal(win, ref_win), (
            f"evaluation {index}: the tracker window moved when only the "
            f"history interval changed ({history_every} vs "
            f"{evaluate_every} steps)")

    if history_every != evaluate_every:
        # THE INSTRUMENT'S OWN CONTROL: prove the fixture exercises the
        # reset at all, by showing the quantity that IS cadence-bound did
        # move.  Without this, an invariance claim is unfalsifiable.
        assert any(not np.array_equal(diag, ref_diag)
                   for (_, diag), (_, ref_diag) in zip(measured, reference)), (
            "UP_HELI_MAX should differ between these two output cadences; "
            "if it does not, this fixture never exercised the reset and "
            "the invariance above proves nothing")


def test_up_heli_max_has_no_trajectory_or_restart_reader():
    """The accumulator can feed output only, never the model.

    Grep pin: the exact repository files allowed to mention the slot (or
    the update/reset entry points whose names embed it), plus an AST pin
    that no module outside the owner reads the slot through the scratch
    APIs except the sanctioned allocation/emission/restore sites.
    """
    woof = REPO / "woof"
    hits = {
        path.relative_to(REPO).as_posix()
        for path in woof.rglob("*.py")
        if "up_heli_max" in path.read_text(encoding="utf-8")
    }
    assert hits == {
        "woof/core/uh_diag.py",      # owner: update/reset/mirror
        "woof/core/gsd_diagnostics.py",  # independent GSD history window
        "woof/ensemble/mass_launcher.py",  # diagnostic download only
        "woof/ensemble/batch_product_output.py",  # output field assembly only
        # The WRF warm start's closed-world carrier census names the slot as
        # REBUILT, never restored: WRF's UP_HELI_MAX running maximum starts a
        # new window at the restart instant, as after every history write.
        # A reason-table string (SCRATCH_REBUILT); nothing is read into the
        # trajectory or the slot.
        "woof/ingest/wrfrst.py",
        "woof/core/state.py",        # eager allocation under the knob
        "woof/core/dycore.py",       # calls update_up_heli_max only
        "woof/core/preflight.py",    # pricing registry + lifetime audit
        "woof/runtime.py",           # calls reset_up_heli_max only
        "woof/offline_child_run.py",  # downscale child reset-on-write
        # Each ensemble member's completed history frame resets its window.
        "woof/ensemble/native_forecast.py",
        # `woof go`'s runner: reset-on-write, same as the others.  This
        # entry is an ADDITION and the reason is worth the line: for as
        # long as it was absent, this assertion was the thing certifying
        # that the one production runner which never reset the window was
        # behaving correctly.  A list of who may MENTION a slot cannot
        # notice who forgot to USE it, which is what
        # test_every_history_publisher_resets_the_window is for.
        "woof/prepared_single_domain_forecast.py",
        # Emission.  The frame mapping moved out of woof/io/wrfout.py
        # into history_layout.py (A190), so the writer and the disk
        # planner walk one inventory; wrfout.py no longer names the slot.
        "woof/io/history_layout.py",
        "woof/io/restart.py",        # serialization + tolerant restore
        # The [tiles] seam.  It never names the slot in code -- it takes
        # one -- but `live_scratch`'s docstring names the resets it exists
        # for, and it is the accessor every one of them now goes through.
        # ADDED because the roster's own question ("who may touch this
        # accumulator") acquired a new correct answer: under a host store the
        # domain's arrays are in the store, so a reset written to the state
        # zeroed a copy and the window silently became "max since the run
        # began".  It reads nothing into the trajectory; it hands the arrays
        # to whoever is about to zero them.
        # Out-of-core streaming names the slot as a CARRIER KEY
        # ("scratch/up_heli_max"), never through the scratch API.  A
        # streamed domain's arrays live in a store and its state is a
        # projection of that store, so the accumulator has to be listed
        # among the planes the executor projects before a whole-domain
        # consumer reduces one -- otherwise the tracker, and the wrfout
        # frame after it, read the plane the state was attached with,
        # which is all zeros for the life of the run.  This is the
        # OPPOSITE of a trajectory reader: nothing here feeds the model,
        # and the AST pins below still hold because streaming touches no
        # scratch API at all.
        "woof/core/streaming.py",
        # Packed member history resets the borrowed view after its
        # synchronous consumer through reset_up_heli_max only. The strict
        # scratch-access and no-trajectory AST guards below still apply.
        "woof/ensemble/native_forecast.py",
    }

    native_tree = ast.parse((REPO / "woof/ensemble/native_forecast.py")
                           .read_text(encoding="utf-8"))
    uh_imports = [node for node in ast.walk(native_tree)
                 if isinstance(node, ast.ImportFrom)
                 and node.module == "woof.core.uh_diag"]
    assert len(uh_imports) == 1
    assert [(alias.name, alias.asname) for alias in uh_imports[0].names] \
        == [("reset_up_heli_max", None)]
    uh_calls = [node for node in ast.walk(native_tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and "up_heli_max" in node.func.id]
    assert len(uh_calls) == 1
    assert uh_calls[0].func.id == "reset_up_heli_max"
    assert len(uh_calls[0].args) == 1
    assert isinstance(uh_calls[0].args[0], ast.Name)
    assert uh_calls[0].args[0].id == "view"

    # Outside the owner and the sanctioned sites, no scratch-API access to
    # the slot at all; in dycore/runtime specifically, only the two entry
    # points may appear (no direct slot access).
    sanctioned_scratch = {"woof/core/uh_diag.py", "woof/core/state.py",
                          "woof/core/gsd_diagnostics.py",
                          "woof/ensemble/mass_launcher.py",
                          "woof/ensemble/batch_product_output.py",
                          "woof/io/history_layout.py", "woof/io/restart.py",
                          # Sanctioned for the indirect-acquisition rule
                          # below: live_scratch/domain_scratch ARE the
                          # scratch site for a streamed domain, and
                          # storm_tracking.py stopped being one in the same
                          # change -- it now asks streaming.domain_scratch
                          # instead of binding existing_scratch itself.
                          "woof/core/streaming.py"}
    for path in woof.rglob("*.py"):
        rel = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("scratch", "existing_scratch")
                    and any(isinstance(arg, ast.Constant)
                            and arg.value == "up_heli_max"
                            for arg in node.args)):
                assert rel in sanctioned_scratch, \
                    f"unsanctioned slot access in {rel}"

    # THE ALIAS HOLE, closed 2026-08-07.  The pin above matches only a
    # DIRECT attribute call carrying a CONSTANT slot name, and a reader
    # that binds the API by name first defeats both halves at once::
    #
    #     getter = getattr(state, "existing_scratch", None)
    #     buf = getter(slot)          # slot is a Name, getter is a Name
    #
    # `node.func` is then ast.Name, not ast.Attribute, and the argument is
    # ast.Name, not ast.Constant -- so the access is invisible.  That is
    # not hypothetical: it is exactly how woof/core/storm_tracking.py's
    # signal_plane read the accumulator into the moving nest's placement
    # decision, and only the coarse grep roster above caught it.  A
    # governance pin that a legitimate duck-typing idiom walks straight
    # through is not a pin.
    #
    # The rule that actually holds: a module allowed to MENTION the slot
    # at all (it is in `hits`, checked above) and which acquires the
    # scratch API indirectly must ALSO be a sanctioned scratch site.
    # Modules that never mention the slot are unaffected, so the ordinary
    # `getattr(x, "scratch", ...)` duck-typing elsewhere in the tree stays
    # legal -- measured at the time of writing: 4 indirect acquisitions
    # repository-wide, of which uh_diag/wrfout are sanctioned owners and
    # ingest/lateral_bc.py never names this slot.
    for rel in sorted(hits):
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and any(isinstance(arg, ast.Constant)
                            and arg.value in ("scratch", "existing_scratch")
                            for arg in node.args)):
                assert rel in sanctioned_scratch, (
                    f"{rel} names the slot AND binds the scratch API "
                    f"indirectly at line {node.lineno}; that combination "
                    "reads the accumulator without tripping the direct "
                    "pin above, so it must be a sanctioned site")


#: The one delegate a runner may reset THROUGH rather than directly.
#: It is `woof/runtime.py::_submit_tree_history_frame`, which submits
#: the frame and zeroes the window in the same two lines.
_RESET_DELEGATES = ("_submit_tree_history_frame",)


def _history_publishing_modules() -> dict[str, str]:
    """Every woof module that constructs a `PerDomainWrfoutWriters`.

    DERIVED, not listed.  A hand-maintained roster of runners is the
    same instrument as the allowlist above and fails the same way: a new
    runner is simply not on it, so its absence reads as compliance.  The
    question "who publishes durable history frames" already has a
    mechanical answer in the tree -- who builds the writer that publishes
    them -- so the test asks the tree instead of a list.
    """

    found = {}
    for path in (REPO / "woof").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.id if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute)
                    else None)
            if name == "PerDomainWrfoutWriters":
                found[path.relative_to(REPO).as_posix()] = text
                break
    return found


def test_every_history_publisher_resets_the_window():
    """A runner that writes history frames must restart the UH window.

    WRF zeroes the nwp_diagnostics running maxima every history interval
    (module_diag_nwp.F:246-269).  A runner that publishes frames and
    never resets does not merely lose the reset: every frame after the
    first reports the maximum since MODEL START, so UP_HELI_MAX is a
    published, headline severe-weather field carrying a wrong number
    that grows monotonically worse with run length.

    `woof.prepared_single_domain_forecast` -- the runner behind `woof
    go`, i.e. the one an ordinary run uses -- did exactly that, while
    the tree runner beside it and the two case integrators did it right.
    The gap survived because the only structural test in this file asked
    which files may MENTION the slot, and a file that never mentions it
    passes that question by doing nothing.
    """

    modules = _history_publishing_modules()
    # Non-vacuity: the derivation has to actually find the runners.
    assert "woof/prepared_single_domain_forecast.py" in modules
    assert "woof/prepared_domain_tree_forecast.py" in modules
    assert len(modules) >= 2

    delinquent = [
        name for name, text in modules.items()
        if "reset_up_heli_max" not in text
        and not any(delegate in text for delegate in _RESET_DELEGATES)]
    assert delinquent == [], (
        "history-publishing runner(s) never restart the UP_HELI_MAX "
        f"window: {delinquent}.  Call "
        "woof.core.uh_diag.reset_up_heli_max on the domain state right "
        "after the frame is submitted, as every other publisher does.")


# ---------------------------------------------------------------------------
# GPU tier
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_trajectory_inertness_bitwise(tmp_path):
    """THE essential gate: knob on vs off over real dycore steps leaves
    every model field bitwise identical, THROUGH to actual wrfout files --
    every variable common to both files is byte-identical and the on-file
    differs exactly by carrying UP_HELI_MAX."""
    import cupy as cp
    from netCDF4 import Dataset

    from woof.core.dycore import step
    from woof.io.wrfout import (WrfoutWriter, _live_state_history_fields,
                                 state_frame)

    from test_mp_spec_zone_ring import _square_moist_state

    s_off, cfg_off = _square_moist_state(mp=1, open_x=True, open_y=True)
    s_on, cfg_on = _square_moist_state(mp=1, open_x=True, open_y=True,
                                       nwp_diagnostics=1)
    for _ in range(6):
        step(s_off, cfg_off)
        step(s_on, cfg_on)

    mutated = ("u", "v", "w", "thp", "php", "mup", "p", "al", "alt",
               "qv", "qc", "qr", "h_diabatic")
    for name in mutated:
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(s_on, name)), cp.asnumpy(getattr(s_off, name)),
            err_msg=f"nwp_diagnostics=1 moved model field {name}")
    # Non-vacuous: the trajectories actually evolved.
    assert float(cp.abs(s_off.w).max()) > 0.0

    on_fields = _live_state_history_fields(s_on)
    off_fields = _live_state_history_fields(s_off)
    assert set(on_fields) - set(off_fields) == {"UP_HELI_MAX"}
    # Precipitation accumulators (scratch) are equally untouched.
    np.testing.assert_array_equal(
        cp.asnumpy(s_on.scratch((cfg_on.ny, cfg_on.nx), "mp_rainnc")),
        cp.asnumpy(s_off.scratch((cfg_off.ny, cfg_off.nx), "mp_rainnc")))

    # File-level proof, the task's own wording: write real wrfouts from
    # both states and byte-compare every common variable.
    paths = {}
    for label, state, cfg in (("off", s_off, cfg_off), ("on", s_on, cfg_on)):
        frame = state_frame(state, include_diagnostic_pressure=True)
        path = tmp_path / f"wrfout_{label}.nc"
        with WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
                          dx=cfg.dx, dy=cfg.dy, title="uh-inertness",
                          field_schema=frame) as writer:
            writer.write_frame("1974-01-01_00:00:00", frame)
        paths[label] = path
    with Dataset(paths["off"]) as off_nc, Dataset(paths["on"]) as on_nc:
        off_vars = set(off_nc.variables)
        on_vars = set(on_nc.variables)
        assert on_vars - off_vars == {"UP_HELI_MAX"}
        assert off_vars - on_vars == set()
        for name in sorted(off_vars):
            if name == "Times":
                continue
            np.testing.assert_array_equal(
                on_nc.variables[name][...], off_nc.variables[name][...],
                err_msg=f"wrfout variable {name} differs across the knob")


@pytest.mark.gpu
@requires_gpu
def test_gpu_device_path_matches_the_numpy_state_path():
    """One update on a device state == the same update on the NumPy state
    (the NumPy path is itself pinned to the oracle by the parity gate)."""
    import cupy as cp
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import DomainState

    cfg = _cfg(nx=9, ny=8, nz=24, ztop=12000.0, specified=True,
               nwp_diagnostics=1)
    host_state = DomainState(cfg, array_module=np)
    device_state = DomainState(cfg)
    vc = make_vertical_coord(cfg.nz)
    base = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    host_state.load_base(vc, base)
    device_state.load_base(vc, base)

    u, v = _mesocyclone_winds(cfg.ny, cfg.nx, cfg.nz)
    z_w = (np.asarray(host_state.phb, dtype=np.float32)
           / np.float32(9.81))[:, None, None]
    w = (16.0 * np.exp(-(((z_w - 3500.0) / 2000.0)**2))
         * np.ones((1, cfg.ny, cfg.nx))).astype(np.float32)
    for state, xp in ((host_state, np), (device_state, cp)):
        state.u[...] = xp.asarray(u)
        state.v[...] = xp.asarray(v)
        state.w[...] = xp.asarray(w)
        update_up_heli_max(state, cfg)
    got = cp.asnumpy(device_state.existing_scratch("up_heli_max"))
    want = host_state.existing_scratch("up_heli_max")
    assert want.max() > 0.0
    np.testing.assert_array_equal(got, want)


# ---------------------------------------------------------------------------
# the streamed halo the diagnostic needs
# ---------------------------------------------------------------------------

def test_the_streamed_halo_widens_for_the_diagnostics_own_reach():
    """F4 at the 2.2.0 cut: one ULP at one cell, at exactly a tile seam.

    ``UP_HELI_MAX`` was the only field of 75 that a streamed run did not
    reproduce bit-identically -- one cell, ``(j=331, i=200)``, 1.16e-10,
    on a 438x350 domain tiled 200x200.  ``i=200`` is the first interior
    column of the second tile, which is what named the mechanism: the
    diagnostic runs in the dycore epilogue from a stencil two cells wide,
    at the one instant when a tile's halo has already been invalidated
    from the outside in by that step's own dependency cone.

    ``halo_radius`` prescribes 16 against a measured carrier minimum of
    14, so the margin is exactly 2 and the diagnostic consumes all of it.
    Re-running that configuration at halo 18 gave bit-identity in all 75
    fields at every frame, which is the measurement this pins.

    Asserted as ``bare + UH_DIAGNOSTIC_HALO_CELLS`` rather than as the
    literal 18, so a change to the sound-step formula moves both together
    and this test keeps testing the diagnostic's reach rather than a
    number that happened to be right once.
    """
    from woof.core.uh_diag import UH_DIAGNOSTIC_HALO_CELLS
    from tilestream import harness

    for ns in (4, 6, 8):
        off = harness.make_config(64, 64, 8, time_step_sound=ns,
                                  nwp_diagnostics=0)
        on = harness.make_config(64, 64, 8, time_step_sound=ns,
                                 nwp_diagnostics=1)
        assert harness.halo_radius(off) == 10 + 3 * ns // 2
        assert (harness.halo_radius(on)
                == harness.halo_radius(off) + UH_DIAGNOSTIC_HALO_CELLS)

    # The default-on half: a run that emits the diagnostic gets the wider
    # halo with no flag set, which is the whole difference between a fix
    # and a workaround.
    default_on = harness.make_config(64, 64, 8, time_step_sound=4,
                                     nwp_diagnostics=1)
    assert harness.halo_radius(default_on) == 18


def test_the_diagnostic_reach_matches_the_stencil_it_describes():
    """The constant is not a magic number: it is 1 + 1, and both are here.

    Vorticity and the 8-point ``w`` average read ``(j-1..j, i-1..i)``;
    the 9-point smoother then reads column values at ``(j+-1, i+-1)``.
    A change to either stencil that did not move this constant would
    silently re-open the defect, so the arithmetic is asserted rather
    than described.
    """
    from woof.core.uh_diag import UH_DIAGNOSTIC_HALO_CELLS

    column_stencil_radius = 1        # rvort / wavg, backward-staggered
    smoother_radius = 1              # the 9-point smooth, both directions
    assert UH_DIAGNOSTIC_HALO_CELLS == column_stencil_radius + smoother_radius
