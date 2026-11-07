"""mp=50's lookup table fails closed, and its parse matches running Fortran.

WHAT THIS MODULE CLAIMS, EXACTLY
--------------------------------
1. The three ways a P3 table can be wrong -- ABSENT, CORRUPT, WRONG VERSION
   -- each produce a distinct refusal that names the breakage it prevents,
   and none of them can be reached by a fallback, a substituted table or a
   zero fill.
2. The arrays :func:`woof.core.p3_tables.load_lookup_table_1` builds from
   the packaged bytes are BYTE-IDENTICAL to the arrays WRF's own
   ``p3_init`` builds from the same bytes.

Claim 2 is an ORACLE claim and it is why this module is separate from
``tests/test_p3_port.py``, whose docstring correctly says it makes none.
The oracle: WRF's ``phys/module_mp_p3.F`` (P3 v4.5.2, sha-256
``716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6``,
byte-identical across WRF v4.6.1/4.7.1/4.8.0), compiled with gfortran
15.2.0 unmodified except for an APPENDED dump routine, run as
``p3_init(nCat=1, trplMomI=.false., model='WRF')``, with ``itab`` and
``itabcoll`` written straight out of the module variables.  The digests
that anchor claim 2 live beside the table pin in
:mod:`woof.core.p3_tables` and were measured 2026-08-28.  They were
identical at -O0 and -O2, and identical whether p3_init read the packaged
CRLF file or WRF's own LF ``run/`` copy.

:func:`test_the_fortran_digest_pin_moves_when_one_value_moves` is the
mutation control: it perturbs a single parsed float by one ULP and
requires the digest to move, so a green claim 2 cannot mean "the digest
was computed over something that ignores the table".
"""

from __future__ import annotations

import hashlib
import shutil

import numpy as np
import pytest

from woof.core.p3_tables import (
    FORTRAN_ITAB_SHA256,
    FORTRAN_ITABCOLL_SHA256,
    P3TableVersionError,
    TABLE_1_2MOM_ASSET,
    TABLE_1_2MOM_VERSION,
    UPSTREAM_WRF_TABLE_1_2MOM_SHA256,
    UPSTREAM_WRF_TABLE_1_2MOM_SIZE,
    load_lookup_table_1,
    p3_table_root,
    packaged_p3_table_root,
)

#: A version token that is well formed and simply not the one P3 v4.5.2
#: intends.  Same length as the intended one on purpose: the wrong-version
#: file then also passes the SIZE gate, which is what makes the ordering
#: test below a real test rather than a size check in disguise.
OTHER_VERSION = b"5.3_2momI"


def _staged(tmp_path):
    """A private copy of the packaged table, ready to be damaged."""
    src = packaged_p3_table_root() / TABLE_1_2MOM_ASSET.filename
    dst = tmp_path / TABLE_1_2MOM_ASSET.filename
    shutil.copyfile(src, dst)
    return dst


# ---------------------------------------------------------------------------
# Refusal 1 of 3: the table is ABSENT.
# ---------------------------------------------------------------------------

def test_an_absent_table_is_refused_and_names_the_scheme_that_cannot_run(
        tmp_path):
    """BREAKAGE PREVENTED: an mp=50 run with no ice process rates at all.

    ``itab`` IS the scheme's ice physics -- fallspeed, collection, melting,
    effective radius, the lambda limiter.  The two failure modes a refusal
    here forecloses are (a) a bare ``FileNotFoundError`` surfacing mid
    forecast, after fetch and preprocessing have been paid for, with no
    statement of what is missing or where it should be, and (b) the far
    worse repair somebody reaches for next: zero-filling the array so init
    succeeds, which leaves ice that never falls, never collects and never
    melts, in a run that stays finite and produces no health trip.
    """
    with pytest.raises(FileNotFoundError) as caught:
        load_lookup_table_1(str(tmp_path))
    message = str(caught.value)
    assert "mp_physics=50/51/52 cannot run" in message
    assert TABLE_1_2MOM_ASSET.filename in message
    # It must say where the canonical copy is, or the operator is stuck.
    assert "ships inside the woof package" in message


# ---------------------------------------------------------------------------
# Refusal 2 of 3: the table is CORRUPT.  Three shapes, one class of refusal.
# ---------------------------------------------------------------------------

def test_a_truncated_table_is_refused_on_size(tmp_path):
    """BREAKAGE PREVENTED: a half-written table parsed into a valid shape.

    A truncated file -- interrupted copy, full disk, aborted download -- is
    still perfectly well-formed ASCII for as far as it goes, and its header
    is intact, so nothing before the size gate can see anything wrong with
    it.  The size gate reports the byte count the operator can act on.
    """
    dst = _staged(tmp_path)
    blob = dst.read_bytes()
    dst.write_bytes(blob[: len(blob) // 2])

    with pytest.raises(ValueError) as caught:
        load_lookup_table_1(str(tmp_path))
    message = str(caught.value)
    assert "has size" in message and str(TABLE_1_2MOM_ASSET.size) in message
    assert "Refusing to parse a modified table" in message
    assert not isinstance(caught.value, P3TableVersionError)


def test_a_same_size_byte_edit_is_refused_on_digest(tmp_path):
    """BREAKAGE PREVENTED: one poisoned interpolation node, silently.

    Every value in ``itab`` is a node of a trilinear interpolation over
    (normalized q, rime fraction, rime density).  A single wrong digit does
    not make the file unparseable and does not make the run unstable: it
    biases one neighbourhood of the ice state space forever, in a forecast
    that looks completely normal.  The gate is a SHA-256 over the whole
    file, so a same-length edit -- which the size gate cannot see -- still
    fails closed.
    """
    dst = _staged(tmp_path)
    blob = bytearray(dst.read_bytes())
    for index in range(len(blob) - 1, 0, -1):
        if blob[index : index + 1].isdigit():
            blob[index] = ord("9") if blob[index] != ord("9") else ord("1")
            break
    dst.write_bytes(bytes(blob))
    assert len(blob) == TABLE_1_2MOM_ASSET.size  # the size gate alone passes

    with pytest.raises(ValueError) as caught:
        load_lookup_table_1(str(tmp_path))
    message = str(caught.value)
    assert "SHA-256" in message
    assert "Refusing to parse a modified table" in message
    assert not isinstance(caught.value, P3TableVersionError)


def test_a_file_that_is_not_a_p3_table_is_refused_by_its_header(tmp_path):
    """BREAKAGE PREVENTED: a decode failure 167,002 tokens into the wrong file.

    ``p3_init`` reads the version token out of the first record before it
    reads anything else (module_mp_p3.F:351).  Transcribing that boundary
    means a file staged under the right name but holding something else --
    the gzipped table, an HTML error page a proxy returned, the wrong asset
    entirely -- is named as "not a P3 lookup table" instead of surfacing as
    a bare UnicodeDecodeError with no statement of what was expected.
    """
    dst = _staged(tmp_path)
    dst.write_bytes(bytes([0x1F, 0x8B, 0x08, 0x00]) + bytes(64))

    with pytest.raises(P3TableVersionError) as caught:
        load_lookup_table_1(str(tmp_path))
    assert "not a P3 lookup table" in str(caught.value)


# ---------------------------------------------------------------------------
# Refusal 3 of 3: the table is a DIFFERENT VERSION.
# ---------------------------------------------------------------------------

def test_a_wrong_version_table_is_refused_as_a_version_mismatch(tmp_path):
    """BREAKAGE PREVENTED: P3 v4.5.2 integrating another version's physics.

    Upstream's severity is the bar and it is not a warning: ``p3_init``
    compares the header against ``version_intended_table_1_2mom``
    (module_mp_p3.F:138) and under ``model='WRF'`` an unequal header is an
    unconditional ``stop`` (:362-365).  The tables ARE the scheme's process
    rates, so another version is different physics wearing the right
    filename, and the 3-moment variant is a different SHAPE (15 quantities
    on an 11-deep mu_i axis) that would reshape into garbage.

    REGRESSION THIS PINS: before 2026-08-28 the digest gate ran first, so
    this file -- a legitimately different table, same length, same token
    count -- was refused with "Refusing to parse a modified table", which
    sends the operator hunting for corruption that is not there.  The
    version refusal was unreachable for every input except the pinned
    bytes, for which it can never fire.
    """
    dst = _staged(tmp_path)
    blob = dst.read_bytes()
    other = blob.replace(TABLE_1_2MOM_VERSION.encode(), OTHER_VERSION, 1)
    assert len(other) == len(blob)          # same size: the size gate passes
    assert other != blob
    dst.write_bytes(other)

    with pytest.raises(P3TableVersionError) as caught:
        load_lookup_table_1(str(tmp_path))
    message = str(caught.value)
    assert OTHER_VERSION.decode() in message
    assert TABLE_1_2MOM_VERSION in message
    assert "unconditional stop" in message
    # The regression: it must NOT be reported as byte corruption.
    assert "Refusing to parse a modified table" not in message
    assert "SHA-256" not in message


def test_the_version_refusal_outranks_the_digest_refusal(tmp_path):
    """Ordering IS the fix, so the ordering is what gets pinned.

    A wrong-version file necessarily also fails the digest.  Only the order
    decides which of the two an operator is told about, and only the
    version answer tells them what to do about it.
    """
    dst = _staged(tmp_path)
    blob = dst.read_bytes()
    damaged = bytearray(
        blob.replace(TABLE_1_2MOM_VERSION.encode(), OTHER_VERSION, 1))
    damaged[-2] = ord("7") if damaged[-2] != ord("7") else ord("2")
    dst.write_bytes(bytes(damaged))

    with pytest.raises(P3TableVersionError):
        load_lookup_table_1(str(tmp_path))


def test_a_wrong_version_table_refuses_the_restart_identity_too(
        tmp_path, monkeypatch):
    """The checkpoint must not record a table identity it could not load.

    ``woof/io/restart.py::_p3_setup_identity`` re-validates the resolved
    table before writing the mp=50 setup identity, precisely so a resume
    binds the bytes the trajectory actually used.  The version refusal has
    to reach it, which is why :class:`P3TableVersionError` subclasses
    ``ValueError`` -- the exception class that path already refuses on.
    """
    from woof.core.p3_tables import P3_TABLE_ROOT_ENV
    from woof.io.restart import RestartManifestError, _p3_setup_identity

    dst = _staged(tmp_path)
    blob = dst.read_bytes()
    dst.write_bytes(
        blob.replace(TABLE_1_2MOM_VERSION.encode(), OTHER_VERSION, 1))
    monkeypatch.setenv(P3_TABLE_ROOT_ENV, str(tmp_path))

    with pytest.raises(RestartManifestError) as caught:
        _p3_setup_identity()
    assert "P3 table identity is invalid" in str(caught.value)
    assert isinstance(caught.value.__cause__, P3TableVersionError)


# ---------------------------------------------------------------------------
# Provenance: what the packaged table IS, proved rather than asserted.
# ---------------------------------------------------------------------------

def test_stripping_the_crs_reproduces_wrfs_own_table_byte_for_byte():
    """The provenance claim, proved from the shipped bytes alone.

    ``woof/data/p3/PROVENANCE.md`` used to say the packaged copy was
    byte-identical to WRF's ``run/p3_lookupTable_1.dat-v5.4_2momI``.  It is
    not: it is that file with one CR added per record, 31,002 of them.
    This test is the corrected claim in executable form -- delete every
    CRLF's CR and the SHA-256 is WRF's, exactly -- so the relationship is
    pinned here rather than resting on a sentence in a markdown file.
    """
    blob = (packaged_p3_table_root() / TABLE_1_2MOM_ASSET.filename
            ).read_bytes()
    upstream = blob.replace(b"\r\n", b"\n")
    assert len(upstream) == UPSTREAM_WRF_TABLE_1_2MOM_SIZE
    assert len(blob) - len(upstream) == 31002          # one CR per record
    assert (hashlib.sha256(upstream).hexdigest()
            == UPSTREAM_WRF_TABLE_1_2MOM_SHA256)


def test_wrfs_own_run_copy_is_refused_and_is_told_apart_from_damage(tmp_path):
    """BREAKAGE PREVENTED: "your table is corrupt" aimed at WRF's own file.

    ``WOOF_P3_TABLE_ROOT`` invites an operator to point at a mirror, and
    on any machine with WRF on it the obvious mirror is WRF's ``run/``
    directory -- whose bytes this pin does not accept.  The two files
    differ in SIZE, so a size-first order would answer "wrong size", which
    reads as a damaged download.  The recognition therefore runs before
    the size gate and says what is actually true: same table, different
    line endings, refused for a restart-identity reason.
    """
    dst = _staged(tmp_path)
    blob = dst.read_bytes()
    dst.write_bytes(blob.replace(b"\r\n", b"\n"))

    with pytest.raises(ValueError) as caught:
        load_lookup_table_1(str(tmp_path))
    message = str(caught.value)
    assert "WRF's own run/ copy" in message
    assert "SAME table" in message
    assert "PROVENANCE.md" in message
    assert "has size" not in message      # not reported as a damaged file
    assert not isinstance(caught.value, P3TableVersionError)


# ---------------------------------------------------------------------------
# The oracle claim: our parse IS the Fortran's arrays.
# ---------------------------------------------------------------------------

def _parsed_digests():
    itab, itabcoll = load_lookup_table_1(p3_table_root())
    return itab, itabcoll, (
        hashlib.sha256(
            np.ascontiguousarray(itab, dtype=np.float32).tobytes()
        ).hexdigest(),
        hashlib.sha256(
            np.ascontiguousarray(itabcoll, dtype=np.float32).tobytes()
        ).hexdigest(),
    )


def test_the_parsed_arrays_are_byte_identical_to_wrfs_own_p3_init():
    """The parser reproduces running Fortran exactly, not approximately.

    Compared as BYTES (the digest is over ``tobytes()``), never with
    ``allclose``: byte-identity is the claim, so an ULP tolerance would be
    a different and weaker claim wearing this test's name.
    """
    itab, itabcoll, (itab_digest, coll_digest) = _parsed_digests()
    assert itab.shape == (5, 4, 50, 14) and itab.dtype == np.float32
    assert itabcoll.shape == (5, 4, 50, 30, 2)
    assert itabcoll.dtype == np.float32
    assert itab_digest == FORTRAN_ITAB_SHA256
    assert coll_digest == FORTRAN_ITABCOLL_SHA256


def test_the_fortran_digest_pin_moves_when_one_value_moves():
    """MUTATION CONTROL for the test above.

    One float32 nudged by one ULP -- 1 word in 74,000, invisible to any
    plot and under any tolerance a spot check would choose -- must move the
    digest.  Without this, a green oracle test could mean the digest was
    taken over something that does not depend on the table at all.
    """
    itab, _itabcoll, (itab_digest, _coll) = _parsed_digests()
    assert itab_digest == FORTRAN_ITAB_SHA256
    mutated = np.array(itab, dtype=np.float32, copy=True)
    mutated.view(np.uint32)[0, 0, 0, 0] += np.uint32(1)
    moved = hashlib.sha256(
        np.ascontiguousarray(mutated, dtype=np.float32).tobytes()).hexdigest()
    assert moved != FORTRAN_ITAB_SHA256
    assert float(mutated[0, 0, 0, 0]) != float(itab[0, 0, 0, 0])
