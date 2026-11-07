"""New Tiedtke's closure flag moves the checkpoint bytes by EXACTLY itself.

``ntiedtke_tiedtke_closure`` joining ``RunConfig`` (49a3f357, "Tiedtke's
deep closure behind a flag, default off") necessarily changes the
checkpoint, because the header echoes the config.  What must be proved is
that it changes it by that ONE key and by nothing else.

WHY THIS FILE EXISTS RATHER THAN JUST A RE-PIN.  The flag landed on this
branch and reached the release baseline at the 2.6.1 merge (585c73ae)
without the pins in ``tests/test_restart.py`` moving with it, so the three
tests that read them ran red from that merge until 2026-09-02.  A red pin
is re-pinned by pasting in the new number, and that is precisely the move
these pins exist to make unsafe: "a config field was added" explains a
moved digest, so a checkpoint-FORMAT change riding in behind one -- a
moved header key, a renamed array member, a widened dtype -- would be
absorbed by the same paste.  Here it cannot be.  The reconstruction below
removes the single key and requires the digest the pins carried BEFORE the
flag existed, byte for byte; anything else that moved survives into the
result and this fails.

THE ANCHOR IS MEASURED, NOT INHERITED.  ``_PRE_NTIEDTKE_*`` was re-derived
by checking out ArWen 2.6.0 (91d92f8d) and 8b9f89f1 (2.6.1) and running
the fixture there.  Both reproduce the pair below exactly, which is what
makes it the release value rather than a number copied out of a diff.

Same construction, and the same argument,
``_digest_without_the_wif_config_keys`` makes for the pair that landed
with the mp=28 aerosol work and ``tests/test_adaptive_timestep_checkpoint``
makes for the twelve adaptive-timestep fields.

SCOPE.  This file covers the BYTES.  Whether a checkpoint written before
the flag still RESUMES is a separate question: ``configuration_sha256``
hashes the whole RunConfig, so any field addition invalidates every
checkpoint written before it, and that is this tree's existing behaviour
for every config-surface addition rather than anything this flag did.  The
prepared-cache identity is separately tolerant of the field -- see
``DEFAULT_TOLERANT_IDENTITY_FIELDS`` in ``woof/ingest/prepared_cache.py``,
where it is the first ``run.``-nested member -- which is why the tolerance
question and the byte question have to be asked apart.
"""

from __future__ import annotations

import json
from datetime import timedelta

import numpy as np

import woof.io.restart as restart
from woof.core.model import ADAPTIVE_TIMESTEP_RUN_FIELDS
from test_restart import (_canonical_member_digest,
                                _digest_without_config_keys,
                                _PINNED_UNDER_MEASURED_CLOCK,
                                _PRE_REAL_CLAMP_ALGORITHMS,
                                _sealed_tree_fixture)

#: Every anchor here predates 2.8.8's Kessler identity move
#: (gfix/288-mp-clamp), so each reconstruction names v1 again.
_HISTORICAL = dict(algorithm_overrides=_PRE_REAL_CLAMP_ALGORITHMS)

#: The one RunConfig field the New Tiedtke closure lane appended.
#:
#: Not in ``woof.core.model.SCHEME_SCOPED_RUN_FIELDS``, and correctly so:
#: that table is keyed by ``mp_physics`` and this flag belongs to
#: ``cu_physics = 16``.  It is therefore an unconditional RunConfig field,
#: which is exactly why it reaches the header config echo and moved the
#: digest.
NTIEDTKE_CLOSURE_RUN_FIELDS: tuple[str, ...] = ("ntiedtke_tiedtke_closure",)

#: ``_canonical_member_digest`` on the tree immediately BEFORE the closure
#: flag joined RunConfig -- harvested at ArWen 2.6.0 (91d92f8d) and
#: confirmed unchanged at 2.6.1 (8b9f89f1).  These are the values
#: ``_LIFECYCLE_FREE_ROOT_DIGEST`` / ``_LIFECYCLE_FREE_CHILD_DIGEST``
#: carried before the 2026-09-02 re-pin.
_PRE_NTIEDTKE_ROOT_DIGEST = \
    "5431237076a41c391a12fee84526ff9981d3303939c2967b399e6bdec10e6f34"
_PRE_NTIEDTKE_CHILD_DIGEST = \
    "0b9e45449cba685b3c7a167bc648d64a2022aaf4b9edf99adad386c5f41ac08c"

#: The digest immediately AFTER the flag landed and before anything else
#: did.  Identical to ``_PRE_ADAPTIVE_*`` in
#: ``tests/test_adaptive_timestep_checkpoint.py`` -- deliberately, because
#: they are one tree state named from the two sides that meet at it, and
#: the next link in the chain anchors here.
_POST_NTIEDTKE_ROOT_DIGEST = \
    "c85c921d720d92be2e138d889eae3db2ce978b5c3fc63f867340a86cf26faac1"
_POST_NTIEDTKE_CHILD_DIGEST = \
    "fb448bfd57196090e0aa3ac025f5ec7234764e1f80f628aded9e5b991a75312d"

#: Every RunConfig key appended AFTER this one.  A per-change attribution
#: anchors on a HISTORICAL tree state, so it has to unwind everything
#: added since that state; otherwise the reconstruction falls short by the
#: later lanes and the anchor is unreachable.  That is not hypothetical --
#: the adaptive-timestep surface (785f1463) landed between this file being
#: written and being merged, which is exactly the race this shape absorbs.
#:
#: This is the growth point.  A lane that appends config keys imports its
#: own tuple here (as the adaptive block does) and the two-link chain below
#: keeps working.  A lane that changes the checkpoint FORMAT cannot be
#: absorbed here at all, which is the whole point of the file.
# eta_levels was appended by the later offline-child ladder change
# (80a3009c2, moved to the dataclass end in 06c29b747), and the downscaled
# child's relax_timescale_s / relax_w pair after it. Field additions to
# unwind, not reasons to move either historical digest.
_KEYS_APPENDED_SINCE: tuple[str, ...] = (
    ADAPTIVE_TIMESTEP_RUN_FIELDS
    + ("eta_levels", "relax_timescale_s", "relax_w",
       # the urban canopy keys (development branch)
       "sf_urban_physics", "use_wudapt_lcz", "num_urban_hi",
       # tc/nest's fixed-frame archive, echoed on a nested domain (2.9.7)
       "nest_accumulation_archive"))


def _write(monkeypatch, tmp_path):
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31,
        # These retained anchors precede the declared local_face default.
        run_overrides=_PINNED_UNDER_MEASURED_CLOCK)
    root = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600))
    child = next(p for p in tmp_path.glob("gpuwmrst_d02_*.npz"))
    return root, child


def _echo(path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(bytes(bytearray(
            data[restart._HEADER_KEY])).decode("utf-8"))["config"]


def test_unwinding_the_later_lanes_reaches_the_post_ntiedtke_digest(
        monkeypatch, tmp_path):
    """Link one: strip everything appended after the flag.

    Establishes the state the flag itself produced, so the step below is
    measured against the tree as it stood and not against today's tree
    minus a guess.
    """
    root, child = _write(monkeypatch, tmp_path)
    # e13fa45c0 / 59f7e280f changed the later moist_cq default. Keep
    # every historical anchor and reconstruct its disabled value explicitly.
    assert (_digest_without_config_keys(
        root, _KEYS_APPENDED_SINCE, config_overrides={"moist_cq": False},
        **_HISTORICAL)
            == _POST_NTIEDTKE_ROOT_DIGEST)
    assert (_digest_without_config_keys(
        child, _KEYS_APPENDED_SINCE, config_overrides={"moist_cq": False},
        **_HISTORICAL)
            == _POST_NTIEDTKE_CHILD_DIGEST)


def test_removing_the_flag_restores_the_pre_ntiedtke_digest(
        monkeypatch, tmp_path):
    """Link two, and the attribution proper: the flag and nothing else.

    Together with the test above this brackets the change: strip the later
    lanes and you are at the post-flag state; strip the flag as well and
    you are at the pre-flag state.  The ONLY difference between the two
    reconstructions is this one key, so that key is what moved the digest
    between those two pins -- measured, not asserted.

    Fails if ANY other part of the checkpoint moved -- a header key, an
    array member's name, dtype, shape or bytes -- because only the config
    echo is reconstructed and everything else is hashed as written.
    """
    root, child = _write(monkeypatch, tmp_path)
    keys = NTIEDTKE_CLOSURE_RUN_FIELDS + tuple(_KEYS_APPENDED_SINCE)
    assert (_digest_without_config_keys(
        root, keys, config_overrides={"moist_cq": False}, **_HISTORICAL)
            == _PRE_NTIEDTKE_ROOT_DIGEST)
    assert (_digest_without_config_keys(
        child, keys, config_overrides={"moist_cq": False}, **_HISTORICAL)
            == _PRE_NTIEDTKE_CHILD_DIGEST)


def test_the_ntiedtke_anchors_moved_for_the_moist_cq_default_and_nothing_else(
        monkeypatch, tmp_path):
    """The e13fa45c0 / 59f7e280f flip is the only new reconstruction."""
    root, child = _write(monkeypatch, tmp_path)
    for path, post_digest, pre_digest in (
            (root, _POST_NTIEDTKE_ROOT_DIGEST, _PRE_NTIEDTKE_ROOT_DIGEST),
            (child, _POST_NTIEDTKE_CHILD_DIGEST, _PRE_NTIEDTKE_CHILD_DIGEST)):
        assert _echo(path)["moist_cq"] is True
        for keys, historical_digest in (
                (_KEYS_APPENDED_SINCE, post_digest),
                (NTIEDTKE_CLOSURE_RUN_FIELDS + _KEYS_APPENDED_SINCE,
                 pre_digest)):
            current = _digest_without_config_keys(path, keys, **_HISTORICAL)
            restored = _digest_without_config_keys(
                path, keys, config_overrides={"moist_cq": False},
                **_HISTORICAL)
            assert current != restored
            assert restored == historical_digest


def test_the_flag_really_is_in_the_echo(monkeypatch, tmp_path):
    """So the reconstruction removes something, not vacuously nothing.

    Without this the test above would pass just as happily if the flag had
    never reached the checkpoint at all -- which is the shape of gate that
    proves nothing.
    """
    root, child = _write(monkeypatch, tmp_path)
    for path in (root, child):
        echo = _echo(path)
        for key in NTIEDTKE_CLOSURE_RUN_FIELDS:
            assert key in echo, key


def test_the_flag_really_does_move_the_digest(monkeypatch, tmp_path):
    """The counterpart: the trimmed digest must NOT equal the plain one."""
    root, child = _write(monkeypatch, tmp_path)
    for path in (root, child):
        assert (_canonical_member_digest(path)
                != _digest_without_config_keys(
                    path, NTIEDTKE_CLOSURE_RUN_FIELDS))


def test_the_echo_grew_by_exactly_this_key(monkeypatch, tmp_path):
    """The negative arm stated as a count rather than as a hash.

    A hash comparison says "something else moved" without saying what.
    This says it in the one currency a reader can act on: the echo carries
    one more key than the release it forked from, and removing this key
    leaves a set of exactly that size.
    """
    root, child = _write(monkeypatch, tmp_path)
    for path in (root, child):
        echo = _echo(path)
        post = {k: v for k, v in echo.items()
                if k not in _KEYS_APPENDED_SINCE}
        pre = {k: v for k, v in post.items()
               if k not in NTIEDTKE_CLOSURE_RUN_FIELDS}
        assert len(post) - len(pre) == len(NTIEDTKE_CLOSURE_RUN_FIELDS)
        assert len(pre) == 158, (
            "the pre-flag config echo carried 158 keys at 2.6.0 and 2.6.1; "
            "a different count means a SECOND config field rode in beside "
            "this one and the per-change attribution above is no longer "
            "exact.  If a later lane appended keys, the fix is to list "
            "them in _KEYS_APPENDED_SINCE, not to move this number")


def test_the_flag_defaults_off_so_the_echo_records_the_pre_flag_state(
        monkeypatch, tmp_path):
    """``False`` is what every checkpoint written before the flag describes.

    The reconstruction only removes the key; it does not assert a value.
    That is sound for the digest, but it would let a default flip to True
    pass silently, and a True default would mean the flag changed what
    every existing run computes rather than only what it echoes.
    """
    root, child = _write(monkeypatch, tmp_path)
    for path in (root, child):
        assert _echo(path)["ntiedtke_tiedtke_closure"] is False
