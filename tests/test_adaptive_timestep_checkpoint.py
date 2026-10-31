"""The adaptive-timestep block moves the checkpoint bytes by EXACTLY itself.

Twelve fields joining ``RunConfig`` necessarily changes the checkpoint,
because the header echoes the config.  What must be proved is that it
changes it by those twelve keys and by nothing else -- the same
construction, and the same argument, ``_digest_without_the_wif_config_keys``
makes for the pair that landed with the mp=28 aerosol work.

The restart IDENTITY is a separate question and is scoped in
``woof.core.model.restart_identity_payload``: with the controller off the
whole block drops out, so an existing checkpoint still resumes.  That is
covered in ``test_adaptive_timestep_surface.py``; this file covers the
bytes.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from datetime import timedelta

import numpy as np
import pytest

import woof.io.restart as restart
from woof.core.model import ADAPTIVE_TIMESTEP_RUN_FIELDS
from test_restart import (_HISTORICAL_FORMAT_VERSION, _unwind_hailcast_members,
                                _PINNED_UNDER_MEASURED_CLOCK,
                                _PRE_REAL_CLAMP_ALGORITHMS,
                                _VOLATILE_CHECKPOINT_HEADER,
                                _canonical_member_digest,
                                _sealed_tree_fixture,
                                _unwind_v7_inverse_binding)

#: ``_canonical_member_digest`` on the tree immediately BEFORE the
#: adaptive-timestep block joined RunConfig, harvested from that tree.
_PRE_ADAPTIVE_ROOT_DIGEST = \
    "c85c921d720d92be2e138d889eae3db2ce978b5c3fc63f867340a86cf26faac1"
_PRE_ADAPTIVE_CHILD_DIGEST = \
    "fb448bfd57196090e0aa3ac025f5ec7234764e1f80f628aded9e5b991a75312d"


def _digest_without_the_adaptive_config_keys(
        path, *, restore_historical_moist_cq=False) -> str:
    """The canonical member digest as it would read WITHOUT the block.

    Same construction as :func:`_canonical_member_digest` with exactly
    three substitutions: the twelve keys are dropped from the config echo,
    and the two hashes the writer derives from that echo are recomputed
    from the trimmed one using the writer's own helpers.  Every array
    member is hashed unchanged.

    The named historical arm also restores the moist_cq value recorded
    before e13fa45c0 / 59f7e280f. It changes no array or non-config header.
    The anchor also predates 2.8.8's Kessler identity move
    (gfix/288-mp-clamp), so every arm names the v1 identity again and
    recomputes the fingerprint from it.
    """
    with np.load(path, allow_pickle=False) as data:
        header = json.loads(bytes(bytearray(
            data[restart._HEADER_KEY])).decode("utf-8"))
        for name in _VOLATILE_CHECKPOINT_HEADER:
            header.pop(name, None)
        # The v6 stamp is the declared 2.7.0 break, not a config key.
        header["format_version"] = _HISTORICAL_FORMAT_VERSION
        # Format 7 (paper/conus-suite-exact, 2.9.11) binds MF_VX_INV into the
        # setup fingerprints; unwind exactly that binding, as test_restart's
        # canonical digest does.
        _unwind_v7_inverse_binding(header)
        # eta_levels was appended later (80a3009c2/06c29b747), the
        # relax_timescale_s / relax_w pair after it, and the three urban
        # canopy keys (lane/urban-infra) after those.  Unwind those separate
        # config additions to reach the historical pre-adaptive tree;
        # leave every array and non-config header bound.
        for key in ADAPTIVE_TIMESTEP_RUN_FIELDS + (
                "eta_levels", "relax_timescale_s", "relax_w",
                "sf_urban_physics", "use_wudapt_lcz", "num_urban_hi",
                # tc/nest's fixed-frame archive (2.9.7), echoed on a nested domain.
                "nest_accumulation_archive"):
            header["config"].pop(key, None)
        # The historical anchor predates the e13fa45c0 / 59f7e280f
        # moist_cq default flip. Retain the key and unwind only its value.
        assert header["config"]["moist_cq"] is True
        if restore_historical_moist_cq:
            header["config"]["moist_cq"] = False
        values = restart._configuration_digest_values(header["config"])
        setup = copy.deepcopy(header["physics_setup"])
        setup["configuration_sha256"] = restart._json_sha256(
            restart._json_value(values, "RunConfig"))
        for component, identity in _PRE_REAL_CLAMP_ALGORITHMS.items():
            assert component in setup["algorithms"], component
            setup["algorithms"][component] = identity
        header["physics_setup"] = setup
        header["physics_setup_fingerprint"] = restart._json_sha256(setup)
        # The historical anchor also predates lane/cap-hail's eleven
        # persistent HAILCAST members (tests/test_restart.py
        # _LIFECYCLE_FREE_ROOT_DIGEST): unwind exactly those.
        names = _unwind_hailcast_members(
            header, [name for name in sorted(data.files)
                     if name != restart._HEADER_KEY])
        digest = hashlib.sha256()
        digest.update(json.dumps(header, sort_keys=True).encode("utf-8"))
        for name in names:
            host = data[name]
            digest.update(name.encode("utf-8"))
            digest.update(str(host.dtype).encode("utf-8"))
            digest.update(str(host.shape).encode("utf-8"))
            digest.update(host.tobytes(order="C"))
    return digest.hexdigest()


def _write(monkeypatch, tmp_path):
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31,
        run_overrides=_PINNED_UNDER_MEASURED_CLOCK)
    root = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600))
    child = next(p for p in tmp_path.glob("gpuwmrst_d02_*.npz"))
    return root, child


def test_removing_the_block_restores_the_pre_adaptive_digest(
        monkeypatch, tmp_path):
    root, child = _write(monkeypatch, tmp_path)
    assert (_digest_without_the_adaptive_config_keys(
        root, restore_historical_moist_cq=True)
            == _PRE_ADAPTIVE_ROOT_DIGEST)
    assert (_digest_without_the_adaptive_config_keys(
        child, restore_historical_moist_cq=True)
            == _PRE_ADAPTIVE_CHILD_DIGEST)


def test_the_pre_adaptive_anchor_moved_for_the_moist_cq_default_and_nothing_else(
        monkeypatch, tmp_path):
    """e13fa45c0 / 59f7e280f moved one default, not checkpoint members."""
    root, child = _write(monkeypatch, tmp_path)
    for path, historical_digest in (
            (root, _PRE_ADAPTIVE_ROOT_DIGEST),
            (child, _PRE_ADAPTIVE_CHILD_DIGEST)):
        current = _digest_without_the_adaptive_config_keys(path)
        restored = _digest_without_the_adaptive_config_keys(
            path, restore_historical_moist_cq=True)
        assert current != restored
        assert restored == historical_digest


def test_the_keys_really_are_in_the_echo(monkeypatch, tmp_path):
    """So the reconstruction removes something rather than succeeding vacuously.

    Without this the test above would pass just as happily if the block
    had never reached the checkpoint at all -- which is the shape of gate
    that proves nothing.
    """
    root, _ = _write(monkeypatch, tmp_path)
    with np.load(root, allow_pickle=False) as data:
        echo = json.loads(bytes(bytearray(
            data[restart._HEADER_KEY])).decode("utf-8"))["config"]
    for key in ADAPTIVE_TIMESTEP_RUN_FIELDS:
        assert key in echo, key


def test_the_block_really_does_move_the_digest(monkeypatch, tmp_path):
    """The counterpart: the trimmed digest must NOT equal the plain one."""
    root, _ = _write(monkeypatch, tmp_path)
    assert (_canonical_member_digest(root)
            != _digest_without_the_adaptive_config_keys(root))


# ------------------------------------- the acoustic substep count moves

def _run_cfg(**over):
    from woof.config import RunConfig

    base = dict(nx=12, ny=12, nz=8, dx=1.0e4, dy=1.0e4, ztop=1.5e4,
                dt=30.0, run_seconds=300.0)
    base.update(over)
    return RunConfig(**base)


def test_time_step_sound_is_state_not_identity_under_an_adaptive_clock():
    """``_apply`` rewrites it from the live dt, so it moves with the dt.

    Upstream zeroes ``time_step_sound`` whenever the adaptive clock is on
    (``start_em.F:966``) precisely so ``solve_em`` derives the acoustic
    substep count from the live dt, and ``adaptive_clock._apply`` does the
    same here through ``wrf_num_sound_steps``.  Bound as identity, a
    checkpoint written after dt grew past the four-substep floor -- the
    doc's 10 km case reaches dt ~ 76 s, where the count is 6 against a
    configured 4 -- could be resumed only by a run that had adapted to the
    identical step, i.e. by nothing.
    """
    live = _run_cfg(use_adaptive_time_step=True, dt=30.0, time_step_sound=4)
    grown = _run_cfg(use_adaptive_time_step=True, dt=76.0, time_step_sound=6)
    restart._require_config_match(
        dataclasses.asdict(grown), live, "checkpoint")
    assert (restart._configuration_fingerprint(live)
            == restart._configuration_fingerprint(grown))


def test_a_fixed_clock_still_binds_the_acoustic_substep_count():
    """Where it really is a setting, it is still compared to the bit."""
    live = _run_cfg(time_step_sound=4)
    grown = _run_cfg(time_step_sound=6)
    with pytest.raises(restart.RestartMismatchError,
                       match="time_step_sound"):
        restart._require_config_match(
            dataclasses.asdict(grown), live, "checkpoint")
    assert (restart._configuration_fingerprint(live)
            != restart._configuration_fingerprint(grown))


# ------------------------------ what a fixed-dt checkpoint does not need

def _stored_echo(cfg, *, drop=()):
    echo = dataclasses.asdict(cfg)
    for key in drop:
        echo.pop(key, None)
    return echo


def test_a_fixed_dt_header_without_the_adaptive_echo_matches_the_defaults():
    """The twelve adaptive fields are optional-with-default when the clock is fixed.

    With the controller off nothing reads them, so a header written without
    them and a live config holding their defaults describe one clock.  This
    is the forward-compatibility rule for the v6 line; a 2.6.5 (v5) file is
    refused by the version gate before this walk runs.
    """
    live = _run_cfg()
    assert not live.use_adaptive_time_step
    restart._require_config_match(
        _stored_echo(live, drop=ADAPTIVE_TIMESTEP_RUN_FIELDS), live, "checkpoint")


def test_the_adaptive_echo_is_still_required_when_the_clock_is_on_or_retuned():
    live_on = _run_cfg(use_adaptive_time_step=True)
    with pytest.raises(restart.RestartMismatchError,
                       match="use_adaptive_time_step: absent from the restart file"):
        restart._require_config_match(
            _stored_echo(live_on, drop=ADAPTIVE_TIMESTEP_RUN_FIELDS), live_on,
            "checkpoint")
    retuned = _run_cfg(target_cfl=1.0)
    with pytest.raises(restart.RestartMismatchError, match="target_cfl"):
        restart._require_config_match(
            _stored_echo(retuned, drop=("target_cfl",)), retuned, "checkpoint")


def test_eta_levels_absent_matches_only_the_none_default():
    live = _run_cfg()
    assert live.eta_levels is None
    restart._require_config_match(
        _stored_echo(live, drop=("eta_levels",)), live, "checkpoint")
    ladder = _run_cfg(eta_levels=tuple(np.linspace(1.0, 0.0, 9)))
    with pytest.raises(restart.RestartMismatchError, match="eta_levels"):
        restart._require_config_match(
            _stored_echo(ladder, drop=("eta_levels",)), ladder, "checkpoint")


def test_a_format_5_header_is_refused_by_name_before_the_identity_walk():
    with pytest.raises(restart.RestartMismatchError) as refused:
        restart.require_readable_format_version(5, "old.npz")
    text = str(refused.value)
    assert "2.6.5 checkpoint format 5" in text
    assert "complete it on 2.6.5" in text
    restart.require_readable_format_version(restart.RESTART_FORMAT_VERSION, "new.npz")
    with pytest.raises(restart.RestartMismatchError, match="this build reads"):
        restart.require_readable_format_version(99, "future.npz")
