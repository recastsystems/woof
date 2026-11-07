"""CPU-only gates for the aerosol-aware Thompson mp=28 table contract.

Every pinned number in this file was produced by compiling WRF v4.6.1's own
code with GNU Fortran 13.3.0 and printing the result at full precision:

* ``_CCN_*`` -- a standalone program reproducing ``table_ccnAct``
  (module_mp_thompson.F:5110-5166), i.e. one sequential-unformatted ``READ``
  of ``CCN_ACTIVATE.BIN`` into ``REAL(4) tnccn_act(7,9,7,5,4)`` with
  ``GFORTRAN_CONVERT_UNIT='big_endian:20'``.
* ``_WGAMMA_AUTHORITY`` / ``_GAMMA_RATIO_AUTHORITY`` -- a verbatim
  transcription of ``GAMMLN``/``WGAMMA`` (5325-5377) and the ``cce``/``ccg``
  loop (671-685), printed with ``ES24.16E3``.
* the ``table_dropEvap`` comparison uses no pin at all: it checks woof's
  reimplementation against the already-SHA-256-pinned classic auxiliary
  asset, which is WRF's own output.

``CCN_ACTIVATE.BIN`` ships with this repository (see
``woof/data/thompson/PROVENANCE.md``), so on a clean checkout the byte-level
tests below run.  The skip guard is kept as defence for a tree where the file
has been deleted or an override points elsewhere; everything that can be
checked without the bytes -- the fail-closed error, the override precedence,
the derived gamma moments, the bin grids, and the ``tnc_wev`` wiring -- runs
unconditionally either way.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import struct
from types import MappingProxyType

import numpy as np
import pytest

from woof.core.thompson_aerosol_contract import (
    ACTIV_KAPPA_INDEX,
    ACTIV_RADIUS_INDEX,
    AEROSOL_ASSET_REDISTRIBUTED,
    AEROSOL_ASSET_SOURCE,
    AEROSOL_CEILING,
    AEROSOL_TABLE_ASSETS,
    AEROSOL_TABLE_FILE,
    AEROSOL_TABLE_PATH_ENV,
    AEROSOL_TABLE_ROOT_ENV,
    AEROSOL_TABLE_SET_ID,
    CCG2,
    CCN_ACTIVATION_FILE_BYTES,
    CCN_ACTIVATION_PAYLOAD_BYTES,
    CCN_ACTIVATION_SHAPE,
    CCN_ACTIVATION_VALUES,
    CLASSIC_DROPLET_BIN,
    DERIVED_CONSTANT_SHA256,
    DROP_EVAP_READ_BY_MP28,
    DROP_EVAP_SHAPE,
    DROP_EVAP_TABLE_NAMES,
    G_RATIO,
    G_RATIO_1BASED,
    NA_CCN0,
    NA_CCN1,
    NA_IN0,
    NA_IN1,
    NIC1,
    NIFA_FLOOR,
    NT_C,
    NT_C_MAX,
    NWFA_FLOOR,
    OCG1,
    TA_KA,
    TA_NA,
    TA_RA,
    TA_TK,
    TA_WW,
    TNC_WEV_RECORD_ORDINAL,
    TNC_WEV_SOURCE_FILE,
    TNC_WEV_TABLE_NAME,
    T_NC_MIN,
    MissingAerosolTableAsset,
    cloud_shape_parameter,
    derived_constant_sha256,
    droplet_number_bin,
    load_validated_aerosol_tables,
    packaged_aerosol_table_root,
    read_ccn_activation_table,
    resolve_aerosol_table_root,
    resolve_ccn_activation_path,
    validate_ccn_activation_asset,
    validate_ccn_activation_table,
    validate_drop_evaporation_tables,
    wgamma_fp32,
)
from woof.core.thompson_aerosol_runtime import (
    AEROSOL_DEVICE_TABLE_NAMES,
    CCN_ACTIVATION_TABLE_NAME,
    DeviceAerosolTableSet,
    _upload_aerosol_table_set,
    device_drop_evaporation_number_table,
)
from woof.core.thompson_contract import (
    AUXILIARY_TABLE_FILE,
    AUXILIARY_TABLE_RECORDS,
    CLASSIC_TABLE_ASSETS,
    TABLE_SET_ID as CLASSIC_TABLE_SET_ID,
    read_sequential_records,
)
from woof.physics_compat import packaged_thompson_table_root


#: Resolved rather than joined: the classic-table directory ships in the
#: recast-woof-data companion distribution since 2.5.0.  ``oracle-aero`` below
#: did NOT move and is still joined onto woof/data.
_TABLES = packaged_thompson_table_root()
_CCN_PATH = _TABLES / AEROSOL_TABLE_FILE
_ORACLE_AERO = (Path(__file__).parents[1] / "woof" / "data" / "thompson"
                / "oracle-aero")
#: Fortran-side readback of tnccn_act written by the WP-03 aerosol oracle
#: harness: one NATIVE-endian sequential record of 8820 REAL(4), i.e. the
#: array as WRF's own READ produced it in memory.
_ORACLE_TNCCN = _ORACLE_AERO / "tnccn_act_native.bin"

# CCN_ACTIVATE.BIN is committed; this guard therefore does not fire on a clean
# checkout.  It is kept so a tree missing the file says so by name instead of
# failing obscurely.  See PROVENANCE.md.
_needs_ccn = pytest.mark.skipif(
    not _CCN_PATH.is_file(),
    reason=f"{AEROSOL_TABLE_FILE} is missing from this tree; restore it "
           f"from {AEROSOL_ASSET_SOURCE} at {_CCN_PATH} or point "
           f"{AEROSOL_TABLE_PATH_ENV} at a copy")
_needs_oracle = pytest.mark.skipif(
    not _ORACLE_TNCCN.is_file(),
    reason="aerosol oracle fixture tnccn_act_native.bin is absent")

# SHA-256 of the widened float64 Fortran-ordered array, i.e. the exact bytes
# this module hands to the device uploader.
_CCN_WIDENED_SHA256 = (
    "bbe00f293eb3569766ab03e5d7797af336c6cbe3c2bf4da56b142a8e89a2983f")
_CCN_MIN = 3.2883699168451130e-04
_CCN_MAX = 9.9930197000503540e-01
# Zero-based indexes; the Fortran dumper printed one-based.
_CCN_SPOT = {
    (0, 0, 0, 0, 0): 4.7921698540449142e-02,
    (0, 0, 3, 2, 1): 2.8176099061965942e-01,
    (6, 8, 6, 4, 3): 9.9168795347213745e-01,
    (3, 4, 3, 2, 1): 7.6158797740936279e-01,
    (0, 8, 3, 2, 1): 9.9930197000503540e-01,
    (6, 0, 3, 2, 1): 2.2361800074577332e-02,
    (2, 5, 1, 0, 2): 7.1762698888778687e-01,
    (4, 2, 5, 3, 0): 1.4617399871349335e-01,
}

# ccg(1..5,n) and (ocg1(n), ocg2(n)) from gfortran's REAL(4) WGAMMA.
_WGAMMA_AUTHORITY = {
    1: ((1.0, 24.0, 5040.001953125, 6.0, 720.0000610351562),
        (1.0, 0.0416666679084301)),
    2: ((2.0, 120.00000762939453, 40319.99609375, 24.0, 5040.001953125),
        (0.5, 0.00833333283662796)),
    3: ((6.0, 720.0000610351562, 362879.96875, 120.00000762939453,
         40319.99609375),
        (0.1666666716337204, 0.00138888880610466)),
    4: ((24.0, 5040.001953125, 3628801.75, 720.0000610351562, 362879.96875),
        (0.0416666679084301, 0.0001984126283787191)),
    5: ((120.00000762939453, 40319.99609375, 39916800.0, 5040.001953125,
         3628801.75),
        (0.00833333283662796, 2.4801589461276308e-05)),
    6: ((720.0000610351562, 362879.96875, 479001856.0, 40319.99609375,
         39916800.0),
        (0.00138888880610466, 2.7557321118365508e-06)),
    7: ((5040.001953125, 3628801.75, 6227022336.0, 362879.96875,
         479001856.0),
        (0.0001984126283787191, 2.7557305770642415e-07)),
    8: ((40319.99609375, 39916800.0, 87178297344.0, 3628801.75,
         6227022336.0),
        (2.4801589461276308e-05, 2.5052107943679403e-08)),
    9: ((362879.96875, 479001856.0, 1307673886720.0, 39916800.0,
         87178297344.0),
        (2.7557321118365508e-06, 2.0876744777353906e-09)),
    10: ((3628801.75, 6227022336.0, 20922782187520.0, 479001856.0,
          1307673886720.0),
         (2.7557305770642415e-07, 1.605904020873794e-10)),
    11: ((39916800.0, 87178297344.0, 355687448182784.0, 6227022336.0,
          20922782187520.0),
         (2.5052107943679403e-08, 1.1470744493147222e-11)),
    12: ((479001856.0, 1307673886720.0, 6402383730966528.0, 87178297344.0,
          355687448182784.0),
         (2.0876744777353906e-09, 7.647166320318144e-13)),
    13: ((6227022336.0, 20922782187520.0, 1.21645284982784e+17,
          1307673886720.0, 6402383730966528.0),
         (1.605904020873794e-10, 4.77947895019884e-14)),
    14: ((87178297344.0, 355687448182784.0, 2.4329033975532093e+18,
          20922782187520.0, 1.21645284982784e+17),
         (1.1470744493147222e-11, 2.8114571472081335e-15)),
    15: ((1307673886720.0, 6402383730966528.0, 5.109091444889066e+19,
          355687448182784.0, 2.4329033975532093e+18),
         (7.647166320318144e-13, 1.5619182991739872e-16)),
}

# ccg(2,n)*ocg1(n) evaluated in REAL(4), the runtime gamma ratio.  Compare
# with G_RATIO_1BASED, the exact integers calc_effectRad uses instead.
_GAMMA_RATIO_AUTHORITY = (
    24.0, 60.000003814697266, 120.00001525878906, 210.00009155273438,
    335.99993896484375, 503.99993896484375, 720.0001220703125,
    990.0000610351562, 1320.0008544921875, 1715.9996337890625, 2184.0,
    2729.997314453125, 3359.998046875, 4079.999755859375, 4896.00927734375,
)


class _CopyBackend:
    float64 = np.float64

    @staticmethod
    def asarray(value, dtype, order):
        return np.array(value, dtype=dtype, order=order, copy=True)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# The external asset, redistributed verbatim from WRF.
# ---------------------------------------------------------------------------

@_needs_ccn
def test_supplied_ccn_activate_matches_the_pinned_wrf_asset():
    assert len(AEROSOL_TABLE_ASSETS) == 1
    asset = AEROSOL_TABLE_ASSETS[0]
    assert asset.filename == "CCN_ACTIVATE.BIN"
    assert asset.bytes == CCN_ACTIVATION_FILE_BYTES == 35288
    assert CCN_ACTIVATION_PAYLOAD_BYTES == 35280
    assert CCN_ACTIVATION_VALUES == 8820
    assert _CCN_PATH.stat().st_size == asset.bytes
    assert _sha256_file(_CCN_PATH) == asset.sha256
    assert validate_ccn_activation_asset(_CCN_PATH) == asset


def test_ccn_activate_is_vendored_but_stays_out_of_the_classic_contract():
    # The blob is third-party parcel-model output, not a WRF product and not a
    # thompson_init product.  Until 2026-08-01 this port did not redistribute
    # it; it now does, as WRF v4.6.1's own run/ file under WRF's public-domain
    # dedication.  What did NOT change is the part that keeps mp=8 free of it,
    # and the pin that stops "shipped" from becoming "any 35 KB file will do".
    assert AEROSOL_ASSET_REDISTRIBUTED is True
    manifest = (_TABLES / "MANIFEST.sha256").read_text(encoding="ascii")
    assert f"*{AEROSOL_TABLE_FILE}" in manifest
    assert len(manifest.strip().splitlines()) == 5
    assert f"{AEROSOL_TABLE_ASSETS[0].sha256} *{AEROSOL_TABLE_FILE}" in manifest
    # Being in tables/MANIFEST.sha256 is NOT membership of the classic set:
    # nothing reads that file at run time, CLASSIC_TABLE_ASSETS is the tuple
    # every mp=8 launch validates, and it must stay four assets long.
    assert AEROSOL_TABLE_FILE not in {
        asset.filename for asset in CLASSIC_TABLE_ASSETS}
    ignore = (Path(__file__).parents[1] / ".gitignore").read_text(
        encoding="utf-8")
    assert ("\nrecast-woof-data/woof_data/data/thompson/tables/CCN_ACTIVATE.BIN"
            not in ignore), (
        "the asset is committed now; a .gitignore entry would make a clean "
        "checkout silently unable to validate mp=28")
    # The pin itself must survive, or "redistributed" quietly becomes
    # "any 35 KB file will do".
    assert AEROSOL_TABLE_ASSETS[0].sha256 == (
        "f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd")
    assert AEROSOL_ASSET_SOURCE.startswith("WRF v4.6.1")
    assert "run/CCN_ACTIVATE.BIN" in AEROSOL_ASSET_SOURCE


def test_missing_ccn_activate_fails_closed_with_an_actionable_error(tmp_path):
    # No synthetic fallback: table_ccnAct prefills tnccn_act with 1.0, so a
    # quiet default would activate every available CCN everywhere and still
    # produce a bounded, NaN-free, entirely wrong forecast.
    with pytest.raises(MissingAerosolTableAsset) as raised:
        load_validated_aerosol_tables(tmp_path, require_classic_assets=False)
    message = str(raised.value)
    assert AEROSOL_TABLE_FILE in message
    assert "mp_physics=28" in message
    assert "WRF v4.6.1" in message and "run/CCN_ACTIVATE.BIN" in message
    assert "d66e442fccc04111067e29274c9f9eaccc3cef28" in message
    assert "35288 bytes" in message
    assert AEROSOL_TABLE_ASSETS[0].sha256 in message
    assert AEROSOL_TABLE_PATH_ENV in message
    assert AEROSOL_TABLE_ROOT_ENV in message
    assert str(tmp_path / AEROSOL_TABLE_FILE) in message
    # It is a FileNotFoundError so ordinary handling still applies, and a
    # distinct type so "operator has not supplied WRF's table" is separable
    # from "the packaged coefficient install is broken".
    assert isinstance(raised.value, FileNotFoundError)


def test_ccn_activation_path_resolution_precedence(tmp_path, monkeypatch):
    from woof import physics_compat

    monkeypatch.delenv(AEROSOL_TABLE_PATH_ENV, raising=False)
    monkeypatch.delenv(AEROSOL_TABLE_ROOT_ENV, raising=False)
    packaged = tmp_path / "packaged"
    fetched = tmp_path / "fetched"
    for directory in (packaged, fetched):
        directory.mkdir()
        for asset in (*CLASSIC_TABLE_ASSETS, *AEROSOL_TABLE_ASSETS):
            (directory / asset.filename).touch()
    monkeypatch.setattr(physics_compat, "packaged_thompson_table_root", lambda: packaged)
    monkeypatch.setattr(physics_compat, "user_thompson_table_root", lambda: fetched)
    assert packaged_aerosol_table_root() == packaged
    assert Path(physics_compat.thompson_table_root(env={})) == packaged
    assert resolve_aerosol_table_root(env={}) == packaged
    assert resolve_ccn_activation_path(env={}) == packaged / AEROSOL_TABLE_FILE

    # Both schemes prefer the complete fetched set over an incomplete package,
    # even if that package still contains the activation file itself.
    missing = next(asset for asset in CLASSIC_TABLE_ASSETS
                   if asset.filename != AEROSOL_TABLE_FILE)
    (packaged / missing.filename).unlink()
    assert Path(physics_compat.thompson_table_root(env={})) == fetched
    assert resolve_aerosol_table_root(env={}) == fetched
    assert resolve_ccn_activation_path(env={}) == fetched / AEROSOL_TABLE_FILE

    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    loose = tmp_path / "elsewhere" / "CCN_ACTIVATE.BIN"
    for target in (root_a / AEROSOL_TABLE_FILE, root_b / AEROSOL_TABLE_FILE,
                   loose):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * 35288)

    env = {AEROSOL_TABLE_ROOT_ENV: str(root_a),
           AEROSOL_TABLE_PATH_ENV: str(loose)}
    # Explicit file, file override, explicit root, root override, shared fallback.
    assert resolve_ccn_activation_path(env=env) == loose
    assert resolve_ccn_activation_path(loose, root_b, env={}) == loose
    assert resolve_ccn_activation_path(
        None, root_b, env={AEROSOL_TABLE_ROOT_ENV: str(root_a)}) == (
            root_b / AEROSOL_TABLE_FILE)
    assert resolve_ccn_activation_path(
        env={AEROSOL_TABLE_ROOT_ENV: str(root_a)}) == (
            root_a / AEROSOL_TABLE_FILE)

    # An explicit override that does not exist is fatal even when the packaged
    # copy would have worked.  Silently ignoring an operator's override is how
    # a run ends up on coefficients nobody chose.
    absent = tmp_path / "nope" / AEROSOL_TABLE_FILE
    with pytest.raises(MissingAerosolTableAsset):
        resolve_ccn_activation_path(absent, env={})
    with pytest.raises(MissingAerosolTableAsset):
        resolve_ccn_activation_path(env={AEROSOL_TABLE_PATH_ENV: str(absent)})


def test_aerosol_table_root_env_name_matches_the_repository_convention():
    # The name is duplicated rather than imported so this module stays out of
    # the physics-capability import graph.  Duplication needs a drift guard.
    from woof import physics_compat

    assert AEROSOL_TABLE_ROOT_ENV == physics_compat.THOMPSON_TABLE_ROOT_ENV
    assert packaged_aerosol_table_root() == \
        physics_compat.packaged_thompson_table_root()


@_needs_ccn
def test_ccn_activate_is_loadable_from_an_arbitrary_override_path(tmp_path):
    relocated = tmp_path / "wrf-run" / "CCN_ACTIVATE.BIN"
    relocated.parent.mkdir(parents=True)
    relocated.write_bytes(_CCN_PATH.read_bytes())
    table_set = load_validated_aerosol_tables(
        tmp_path / "no-such-root", ccn_path=relocated,
        require_classic_assets=False)
    assert table_set.ccn_path == relocated.resolve()
    np.testing.assert_array_equal(
        table_set.arrays[CCN_ACTIVATION_TABLE_NAME],
        read_ccn_activation_table(_CCN_PATH))
    # The path is machine-local; the identity is content addressed, so two
    # byte-identical copies must compare equal across hosts.
    assert "ccn_path" not in table_set.identity
    assert str(relocated) not in repr(table_set.identity)


def test_aerosol_asset_set_is_separate_from_the_classic_mp8_contract():
    # CCN_ACTIVATE.BIN must never join CLASSIC_TABLE_ASSETS: every existing
    # mp=8 launch would then fail closed on a file it never needed.
    classic_names = {asset.filename for asset in CLASSIC_TABLE_ASSETS}
    assert "CCN_ACTIVATE.BIN" not in classic_names
    assert len(CLASSIC_TABLE_ASSETS) == 4
    assert AEROSOL_TABLE_SET_ID != CLASSIC_TABLE_SET_ID
    assert AEROSOL_TABLE_SET_ID == "wrf-v4.6.1-aerosol-thompson-mp28-v1"


@_needs_ccn
def test_ccn_activation_reader_reproduces_the_fortran_authority():
    table = read_ccn_activation_table(_CCN_PATH)
    assert table.shape == CCN_ACTIVATION_SHAPE == (7, 9, 7, 5, 4)
    assert table.dtype == np.dtype(np.float64)
    assert table.flags.f_contiguous
    assert not table.flags.writeable
    assert float(table.min()) == _CCN_MIN
    assert float(table.max()) == _CCN_MAX
    for index, expected in _CCN_SPOT.items():
        assert float(table[index]) == expected, index
    assert hashlib.sha256(
        table.tobytes(order="F")).hexdigest() == _CCN_WIDENED_SHA256
    # Widening float32 -> float64 is exact.
    assert np.array_equal(table.astype(np.float32).astype(np.float64), table)


@_needs_ccn
def test_ccn_activation_table_axes_and_hardcoded_slab_indexes():
    assert TA_NA == (10.0, 31.6, 100.0, 316.0, 1000.0, 3160.0, 10000.0)
    assert TA_WW == (0.01, 0.0316, 0.1, 0.316, 1.0, 3.16, 10.0, 31.6, 100.0)
    assert TA_TK == (243.15, 253.15, 263.15, 273.15, 283.15, 293.15, 303.15)
    assert TA_RA == (0.01, 0.02, 0.04, 0.08, 0.16)
    assert TA_KA == (0.2, 0.4, 0.6, 0.8)
    assert (len(TA_NA), len(TA_WW), len(TA_TK), len(TA_RA), len(TA_KA)) \
        == CCN_ACTIVATION_SHAPE
    # activ_ncloud:5229-5230 hardcodes l=3, m=2 (one-based).
    assert (ACTIV_RADIUS_INDEX, ACTIV_KAPPA_INDEX) == (3, 2)

    table = read_ccn_activation_table(_CCN_PATH)
    slab = table[:, :, :, ACTIV_RADIUS_INDEX - 1, ACTIV_KAPPA_INDEX - 1]
    assert slab.shape == (7, 9, 7)
    # Activated fraction rises with updraft and falls with aerosol loading.
    for k in range(7):
        assert np.all(np.diff(slab[0, :, k]) >= 0.0)
        assert np.all(np.diff(slab[:, 0, k]) <= 0.0)


@_needs_ccn
def test_ccn_activation_reader_rejects_wrong_byte_order_and_layout(tmp_path):
    payload = _CCN_PATH.read_bytes()[4:-4]

    swapped = tmp_path / "swapped.bin"
    marker = struct.pack("<i", CCN_ACTIVATION_PAYLOAD_BYTES)
    swapped.write_bytes(marker + payload + marker)
    with pytest.raises(ValueError, match="marker declares"):
        read_ccn_activation_table(swapped)

    truncated = tmp_path / "short.bin"
    big = struct.pack(">i", CCN_ACTIVATION_PAYLOAD_BYTES)
    truncated.write_bytes(big + payload[:-16] + big)
    with pytest.raises(ValueError, match="truncated|trailing"):
        read_ccn_activation_table(truncated)

    # A C-order reshape leaves the file SHA-256 and the value histogram
    # untouched, so only a structural check can catch it.
    values = np.frombuffer(payload, dtype=">f4", count=CCN_ACTIVATION_VALUES)
    wrong_order = np.array(values.reshape(CCN_ACTIVATION_SHAPE, order="C"),
                           dtype=np.float64, order="F")
    with pytest.raises(ValueError, match="order flip|not strictly increasing"):
        validate_ccn_activation_table(wrong_order)


def test_ccn_activation_validator_rejects_the_thompson_init_prefill():
    # table_ccnAct prefills tnccn_act with 1.0 at :993-1002 and only replaces
    # it inside the one-time micro_init block.  A silent read failure leaves
    # activ_ncloud returning fraction 1.0 -- total activation, no error.
    prefill = np.ones(CCN_ACTIVATION_SHAPE, dtype=np.float64, order="F")
    with pytest.raises(ValueError, match="uniformly 1.0"):
        validate_ccn_activation_table(prefill)

    out_of_range = np.full(CCN_ACTIVATION_SHAPE, 1.5, order="F")
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        validate_ccn_activation_table(out_of_range)


# ---------------------------------------------------------------------------
# Derived gamma moments and bin grids.
# ---------------------------------------------------------------------------

def test_wgamma_transcription_is_bitwise_equal_to_gfortran_real4():
    from woof.core.thompson_aerosol_contract import (
        CCG1, CCG3, CCG4, CCG5, OCG2)

    rows = (CCG1, CCG2, CCG3, CCG4, CCG5)
    for n, (ccg, ocg) in _WGAMMA_AUTHORITY.items():
        for row, expected in zip(rows, ccg):
            assert float(row[n]) == expected, (n, expected)
        assert float(OCG1[n]) == ocg[0], n
        assert float(OCG2[n]) == ocg[1], n
    # Index 0 is an unused zero so device code can be written one-based.
    for row in rows + (OCG1, OCG2):
        assert row.shape == (16,)
        assert float(row[0]) == 0.0
        assert not row.flags.writeable


def test_wgamma_disagrees_with_a_library_gamma_where_it_matters():
    import math

    # WRF's REAL(4) Lanczos series is not the true Gamma function, and the
    # difference is exactly the value already embedded in ArWen's mp=8
    # kernels.  A library gamma would silently move every mp=28 coefficient.
    assert float(wgamma_fp32(16.0)) == 1307673886720.0
    assert float(np.float32(math.gamma(16.0))) == 1307674411008.0
    assert float(wgamma_fp32(16.0)) != float(np.float32(math.gamma(16.0)))
    # ~4e-7 relative -- three orders of magnitude above float32 rounding.
    assert abs(wgamma_fp32(16.0) / math.gamma(16.0) - 1.0) > 1.0e-7


def test_runtime_gamma_ratio_differs_from_the_exact_integer_g_ratio():
    # calc_effectRad:5611-5613 uses the exact integers; every other consumer
    # forms ccg(2,n)*ocg1(n) at runtime.  The two are not interchangeable,
    # which is why mp=8's literal 2730.0f is right at thompson.cu:343 and a
    # 1e-6 deviation at thompson.cu:882/999/4005/4128/4680.
    assert G_RATIO_1BASED == (24, 60, 120, 210, 336, 504, 720, 990,
                              1320, 1716, 2184, 2730, 3360, 4080, 4896)
    assert [int(n) for n in G_RATIO[1:]] == list(G_RATIO_1BASED)
    assert float(G_RATIO[0]) == 0.0
    for n, expected in enumerate(_GAMMA_RATIO_AUTHORITY, start=1):
        product = float(np.float32(np.float32(CCG2[n]) * np.float32(OCG1[n])))
        assert product == expected, n
        assert (n + 1) * (n + 2) * (n + 3) == G_RATIO_1BASED[n - 1]
    assert _GAMMA_RATIO_AUTHORITY[11] == 2729.997314453125
    assert G_RATIO_1BASED[11] == 2730


def test_droplet_number_bin_reproduces_the_mp8_hardcoded_index():
    # nic1 is declared INTEGER (:246) and assigned a DOUBLE (:896), so
    # Fortran truncates 7.926303891973744 to 7.
    assert NIC1 == 7
    assert T_NC_MIN == 1.0408439118849806e+06
    # The exact number ArWen's mp=8 kernels embed as cloud_number_bin.
    assert droplet_number_bin(NT_C) == CLASSIC_DROPLET_BIN == 65
    assert cloud_shape_parameter(NT_C) == 12
    assert droplet_number_bin(1.0) == 0
    assert droplet_number_bin(1.0e12) == 99
    # nu_c is an integer in 2..15 across the physical droplet-number range.
    for nc in (2.0, 30.0e6, 100.0e6, 300.0e6, 1000.0e6, 1800.0e6, NT_C_MAX):
        assert 2 <= cloud_shape_parameter(nc) <= 15
    assert cloud_shape_parameter(30.0e6) == 15
    assert cloud_shape_parameter(300.0e6) == 5
    assert cloud_shape_parameter(1000.0e6) == 3
    # nu_c = 2 needs NINT(1000e6/nc) == 0, i.e. nc > 2e9, which is ABOVE
    # Nt_c_max.  The floor is therefore unreachable from prognostic nc and is
    # only visited through the t_Nc bin ladder (which runs to 2.88e9) and
    # calc_effectRad's separate nc > 1e10 branch.
    assert cloud_shape_parameter(NT_C_MAX) == 3
    assert cloud_shape_parameter(2.1e9) == 2


def test_derived_constants_are_sha256_pinned():
    assert derived_constant_sha256() == DERIVED_CONSTANT_SHA256
    assert len(DERIVED_CONSTANT_SHA256) == 64


def test_pinned_aerosol_scalars_match_module_mp_thompson():
    assert (NA_IN0, NA_IN1, NA_CCN0, NA_CCN1) == (1.5e6, 0.5e6, 300.0e6,
                                                  50.0e6)
    assert (NT_C, NT_C_MAX) == (100.0e6, 1999.0e6)
    # The floors are naCCN1*0.222 and naIN1*0.01 written out (:1805-1806).
    assert NWFA_FLOOR == pytest.approx(NA_CCN1 * 0.222, rel=1e-12)
    assert NIFA_FLOOR == NA_IN1 * 0.01
    assert AEROSOL_CEILING == 9999.0e6


# ---------------------------------------------------------------------------
# table_dropEvap: the classic auxiliary asset already IS the mp=28 table.
# ---------------------------------------------------------------------------

def test_packaged_auxiliary_asset_carries_the_aerosol_drop_evaporation_table():
    # table_dropEvap is called unconditionally at thompson_init:1025, outside
    # the is_aerosol_aware guard, and its droplet-number axis already sweeps
    # nu_c from 15 down to 2.  mp=28 therefore ships NO new asset for it and
    # must not re-pin thompson_aux_tables.dat.
    arrays = read_sequential_records(_TABLES / AUXILIARY_TABLE_FILE,
                                     AUXILIARY_TABLE_RECORDS)
    for name in DROP_EVAP_TABLE_NAMES:
        assert arrays[name].shape == DROP_EVAP_SHAPE == (100, 37, 100)
    measured = validate_drop_evaporation_tables(arrays)
    assert set(measured) == set(DROP_EVAP_TABLE_NAMES)
    for name, relative in measured.items():
        assert relative < 1.0e-13, (name, relative)
    # tpc_wev's only use is commented out in v4.6.1 (:3465-3466); it stays
    # pinned and uploaded so the classic asset SHA is unchanged, and unread.
    assert DROP_EVAP_READ_BY_MP28 == ("tnc_wev",)

    # The table is genuinely nc-dependent: a fixed-Nt_c table would be
    # constant along its third axis.
    tnc = arrays["tnc_wev"]
    assert not np.allclose(tnc[:, :, 0], tnc[:, :, 99])


def test_drop_evaporation_validator_rejects_a_transposed_upload():
    arrays = read_sequential_records(_TABLES / AUXILIARY_TABLE_FILE,
                                     AUXILIARY_TABLE_RECORDS)
    # tnc_wev is (nbc, ntb_c, nbc): axes 0 and 2 are both length 100, so a
    # Fortran/C order confusion is shape-compatible and otherwise invisible.
    swapped = dict(arrays)
    swapped["tnc_wev"] = np.asfortranarray(
        np.transpose(arrays["tnc_wev"], (2, 1, 0)))
    with pytest.raises(ValueError, match="deviates from WRF's table_dropEvap"):
        validate_drop_evaporation_tables(swapped)


def test_tnc_wev_needs_no_new_asset_and_is_already_on_the_device():
    # The wiring claim, checked in this repository rather than asserted: the
    # array mp=28's aerosol-only droplet evaporation reads is record 7 of
    # thompson_aux_tables.dat, is already in the classic record contract, and
    # is already fed to load_classic_device_tables.  mp=8 has been uploading
    # it, unread, since the classic port.
    from woof.core import thompson_runtime

    assert TNC_WEV_TABLE_NAME == "tnc_wev"
    assert TNC_WEV_SOURCE_FILE == AUXILIARY_TABLE_FILE
    assert AUXILIARY_TABLE_RECORDS[TNC_WEV_RECORD_ORDINAL - 1].name == \
        TNC_WEV_TABLE_NAME
    assert AUXILIARY_TABLE_RECORDS[TNC_WEV_RECORD_ORDINAL - 1].shape == \
        DROP_EVAP_SHAPE
    classic_names = [record.name
                     for record in thompson_runtime._classic_records()]
    assert TNC_WEV_TABLE_NAME in classic_names
    assert AEROSOL_TABLE_FILE not in str(AUXILIARY_TABLE_RECORDS)
    # ...and it is NOT re-uploaded by the aerosol owner.
    assert TNC_WEV_TABLE_NAME not in AEROSOL_DEVICE_TABLE_NAMES

    sentinel = object()

    class _Classic:
        arrays = MappingProxyType({TNC_WEV_TABLE_NAME: sentinel})

    assert device_drop_evaporation_number_table(_Classic) is sentinel

    class _Missing:
        arrays = MappingProxyType({})

    with pytest.raises(KeyError, match="record 7"):
        device_drop_evaporation_number_table(_Missing)


# ---------------------------------------------------------------------------
# Loader and device ownership.
# ---------------------------------------------------------------------------

@_needs_ccn
def test_load_validated_aerosol_tables_freezes_and_identifies_the_set():
    table_set = load_validated_aerosol_tables(
        _TABLES, require_classic_assets=False)
    assert set(table_set.arrays) == set(AEROSOL_DEVICE_TABLE_NAMES)
    for name, array in table_set.arrays.items():
        assert array.dtype == np.dtype(np.float64), name
        assert array.flags.f_contiguous, name
        assert not array.flags.writeable, name
    identity = table_set.identity
    assert identity["table_set"] == AEROSOL_TABLE_SET_ID
    assert identity["classic_table_set"] == CLASSIC_TABLE_SET_ID
    assert identity["mp_physics"] == 28
    assert identity["wrf_commit"] == (
        "d66e442fccc04111067e29274c9f9eaccc3cef28")
    assert identity["derived_constants_sha256"] == DERIVED_CONSTANT_SHA256
    assert identity["assets"] == [
        {"filename": "CCN_ACTIVATE.BIN", "bytes": 35288,
         "sha256": AEROSOL_TABLE_ASSETS[0].sha256}]


@_needs_ccn
def test_load_validated_aerosol_tables_fails_closed_on_substituted_bytes(
        tmp_path):
    corrupt = bytearray(_CCN_PATH.read_bytes())
    corrupt[100] ^= 0x01
    (tmp_path / AEROSOL_TABLE_FILE).write_bytes(bytes(corrupt))
    with pytest.raises(ValueError, match="SHA-256"):
        load_validated_aerosol_tables(tmp_path, require_classic_assets=False)

    with pytest.raises(MissingAerosolTableAsset, match="was not found"):
        load_validated_aerosol_tables(tmp_path / "empty",
                                      require_classic_assets=False)


@_needs_ccn
def test_aerosol_device_upload_round_trips_every_array():
    table_set = load_validated_aerosol_tables(
        _TABLES, require_classic_assets=False)
    device = _upload_aerosol_table_set(
        table_set, _CopyBackend, device_id=None, verify_roundtrip=True)
    assert set(device.arrays) == set(AEROSOL_DEVICE_TABLE_NAMES)
    assert device.roundtrip_verified
    assert device.payload_bytes == sum(
        array.nbytes for array in table_set.arrays.values())
    assert len(device.identity_sha256) == 64
    ccn = device.ccn_activation_table
    assert ccn.shape == CCN_ACTIVATION_SHAPE
    assert ccn.flags.f_contiguous
    np.testing.assert_array_equal(
        ccn, table_set.arrays[CCN_ACTIVATION_TABLE_NAME])
    assert len(device.gamma_moment_tables) == 12


@_needs_ccn
def test_aerosol_device_upload_rejects_corruption_and_bad_inventory(tmp_path):
    table_set = load_validated_aerosol_tables(
        _TABLES, require_classic_assets=False)

    class CorruptBackend:
        float64 = np.float64

        @staticmethod
        def asarray(source, dtype, order):
            result = np.array(source, dtype=dtype, order=order, copy=True)
            result.ravel(order="F")[0] += 1.0
            return result

    with pytest.raises(RuntimeError, match="round trip changed"):
        _upload_aerosol_table_set(
            table_set, CorruptBackend, device_id=None, verify_roundtrip=True)

    thinned = type(table_set)(
        root=table_set.root,
        arrays=MappingProxyType(
            {k: v for k, v in table_set.arrays.items() if k != "ocg2"}),
        assets=table_set.assets)
    with pytest.raises(ValueError, match="inventory mismatch"):
        _upload_aerosol_table_set(
            thinned, _CopyBackend, device_id=None, verify_roundtrip=False)


def test_device_aerosol_table_set_exposes_arrays_as_attributes(tmp_path):
    sentinels = {name: object() for name in AEROSOL_DEVICE_TABLE_NAMES}
    owner = DeviceAerosolTableSet(
        root=tmp_path,
        arrays=MappingProxyType(sentinels),
        payload_bytes=0,
        array_sha256=MappingProxyType({}),
        identity_json="{}",
        identity_sha256="a" * 64,
        device_id=None,
        upload_seconds=0.0,
        verification_seconds=0.0,
        roundtrip_verified=True,
    )
    assert owner.tnccn_act is sentinels["tnccn_act"]
    assert owner.ccg2 is sentinels["ccg2"]
    assert owner.ccn_activation_table is sentinels["tnccn_act"]
    assert owner.ccn_path is None
    with pytest.raises(AttributeError):
        owner.not_a_table


# ---------------------------------------------------------------------------
# Cross-check against the WRF-side readback, and against a real device.
# ---------------------------------------------------------------------------

@_needs_ccn
@_needs_oracle
def test_parsed_tnccn_act_equals_the_fortran_readback_dump():
    """The Python parse must equal what WRF's own READ produced in memory.

    ``tnccn_act_native.bin`` is the WP-03 aerosol harness dumping ``tnccn_act``
    after ``table_ccnAct`` ran -- one NATIVE-endian sequential record, i.e.
    little-endian on this host, against the big-endian shipped asset.  That
    endianness flip is exactly the step the Python reader has to get right,
    and comparing against WRF's post-read array is the only check that closes
    it end to end.  Equality must be EXACT: both sides are the same float32
    values, and float32 -> float64 widening is lossless.
    """
    raw = _ORACLE_TNCCN.read_bytes()
    assert len(raw) == CCN_ACTIVATION_FILE_BYTES
    head = struct.unpack("<i", raw[:4])[0]
    tail = struct.unpack("<i", raw[-4:])[0]
    assert head == tail == CCN_ACTIVATION_PAYLOAD_BYTES
    # Native, NOT byte-swapped: the shipped asset is big-endian only because
    # WRF compiles with BYTESWAPIO, and the in-memory array is not.
    reference = np.array(
        np.frombuffer(raw[4:-4], dtype="<f4",
                      count=CCN_ACTIVATION_VALUES).reshape(
                          CCN_ACTIVATION_SHAPE, order="F"),
        dtype=np.float64, order="F")

    table = read_ccn_activation_table(_CCN_PATH)
    assert np.array_equal(table, reference)
    assert float(np.abs(table - reference).max()) == 0.0
    # Guard the guard: the dump is not trivially symmetric, so a byte-order or
    # order='C' mistake on either side would have shown up.
    assert not np.array_equal(reference, reference.astype(">f4").view("<f4")
                              .astype(np.float64))
    assert not np.array_equal(
        reference,
        np.asfortranarray(reference.ravel(order="F").reshape(
            CCN_ACTIVATION_SHAPE, order="C")))


def _reference_activ_ncloud(table, Tt, Ww, NCCN):
    """Host float32 transcription of ``activ_ncloud``, 5178-5253.

    This lives here as a TABLE-ORIENTATION gate, not as a physics helper:
    WP-02 owns the CUDA transcription and its own probe comparison.  What it
    proves for this work package is that ``read_ccn_activation_table`` produced
    an array whose five axes, Fortran order and hardcoded ``l=3, m=2`` slab are
    the ones WRF indexes -- a claim no value-range or spot check can settle.
    """
    f32 = np.float32
    n_local = f32(f32(NCCN) * f32(1.0e-6))
    if n_local >= f32(TA_NA[-1]):
        n_local = f32(f32(TA_NA[-1]) - f32(1.0))
    elif n_local <= f32(TA_NA[0]):
        n_local = f32(f32(TA_NA[0]) + f32(1.0))
    w_local = f32(Ww)
    # ASYMMETRIC: -1.0/+1.0 for N but -1.0/+0.001 for w.  Transcribed, not
    # tidied; +1.0 on the 0.01 floor would land outside ta_Ww's first bracket.
    if w_local >= f32(TA_WW[-1]):
        w_local = f32(f32(TA_WW[-1]) - f32(1.0))
    elif w_local <= f32(TA_WW[0]):
        w_local = f32(f32(TA_WW[0]) + f32(0.001))

    def _bracket(axis, value):
        # Fortran DO/GOTO: on normal termination the loop variable is
        # ubound+1, NOT ubound.  The clamps above guarantee the GOTO fires, so
        # this is unreachable; it is written faithfully anyway so an
        # out-of-range input degrades the way WRF's does.
        for candidate in range(2, len(axis) + 1):
            if f32(axis[candidate - 2]) <= value < f32(axis[candidate - 1]):
                return candidate
        return len(axis) + 1

    i = _bracket(TA_NA, n_local)
    j = _bracket(TA_WW, w_local)
    x1 = f32(np.log(f32(TA_NA[i - 2])))
    x2 = f32(np.log(f32(TA_NA[i - 1])))
    y1 = f32(np.log(f32(TA_WW[j - 2])))
    y2 = f32(np.log(f32(TA_WW[j - 1])))
    # NEAREST-NEIGHBOUR in temperature -- no interpolation on the T axis.
    arg = f32(f32(f32(Tt) - f32(TA_TK[0])) * f32(0.1))
    k = max(1, min(int(np.floor(float(arg) + 0.5)) + 1, len(TA_TK)))
    l, m = ACTIV_RADIUS_INDEX, ACTIV_KAPPA_INDEX

    slab = (k - 1, l - 1, m - 1)
    A = f32(table[(i - 2, j - 2) + slab])
    B = f32(table[(i - 1, j - 2) + slab])
    C = f32(table[(i - 1, j - 1) + slab])
    D = f32(table[(i - 2, j - 1) + slab])
    t = f32(f32(f32(np.log(n_local)) - x1) / f32(x2 - x1))
    u = f32(f32(f32(np.log(w_local)) - y1) / f32(y2 - y1))
    one = f32(1.0)
    fraction = f32(
        f32(f32(f32(one - t) * f32(one - u)) * A)
        + f32(f32(t * f32(one - u)) * B)
        + f32(f32(t * u) * C)
        + f32(f32(f32(one - t) * u) * D))
    return f32(f32(NCCN) * fraction)


@_needs_ccn
@_needs_oracle
def test_parsed_table_reproduces_the_fortran_activ_ncloud_probe():
    probe = _ORACLE_AERO / "probe-activncloud.csv"
    if not probe.is_file():
        pytest.skip("probe-activncloud.csv is absent")
    rows = np.genfromtxt(probe, delimiter=",", names=True)
    assert len(rows) >= 1000
    table = read_ccn_activation_table(_CCN_PATH)

    mismatched = []
    for row in rows:
        got = float(_reference_activ_ncloud(
            table, row["temp_k"], row["w_m_s"], row["nccn_per_m3"]))
        expected = float(row["activated_per_m3"])
        if got != expected:
            mismatched.append((row["temp_k"], row["w_m_s"],
                               row["nccn_per_m3"], got, expected))
    # Bit-exact, not approximate: the sweep is float32 on both sides, the
    # temperature axis is nearest-neighbour, and both bracket searches are
    # integer.  Any tolerance here would hide exactly the failure this gate
    # exists to catch -- a transposed or mis-sliced tnccn_act.
    assert not mismatched, mismatched[:5]

    # The gate has teeth.  Each mutation below is shape-preserving, leaves the
    # value histogram and the file SHA-256 untouched, and must still break the
    # sweep: moving off the hardcoded radius slab, moving off the hardcoded
    # kappa slab, reversing the updraft axis, and reversing the temperature
    # axis.  A single row is not enough -- the first probe row sits at both
    # clamp corners, where several of these are coincidentally invariant.
    def _mismatch_count(candidate):
        return sum(
            1 for row in rows
            if float(_reference_activ_ncloud(
                candidate, row["temp_k"], row["w_m_s"],
                row["nccn_per_m3"])) != float(row["activated_per_m3"]))

    assert _mismatch_count(table) == 0
    # ta_Ra has an odd length and l=3 is its midpoint, so a flip of that axis
    # is a no-op there; roll the two slab axes and flip the two swept ones.
    mutations = (
        ("mean radius", np.roll(table, 1, axis=3)),
        ("kappa", np.roll(table, 1, axis=4)),
        ("aerosol loading", np.flip(table, axis=0)),
        ("updraft", np.flip(table, axis=1)),
        ("temperature", np.flip(table, axis=2)),
    )
    for label, mutated in mutations:
        assert _mismatch_count(np.asfortranarray(mutated)) > 0, label


@_needs_ccn
def test_aerosol_tables_upload_and_read_back_on_a_real_cuda_device():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device")

    from woof.core import thompson_aerosol_runtime

    thompson_aerosol_runtime._DEVICE_CACHE.clear()
    device = thompson_aerosol_runtime.load_aerosol_device_tables(
        _TABLES, backend=cupy, verify_roundtrip=True, cache=False,
        require_classic_assets=False)
    assert device.roundtrip_verified
    assert device.device_id is not None
    assert device.ccn_path == _CCN_PATH.resolve()
    assert set(device.arrays) == set(AEROSOL_DEVICE_TABLE_NAMES)

    host = load_validated_aerosol_tables(
        _TABLES, require_classic_assets=False)
    for name in AEROSOL_DEVICE_TABLE_NAMES:
        resident = device.arrays[name]
        assert isinstance(resident, cupy.ndarray), name
        assert resident.dtype == cupy.float64, name
        assert resident.flags.f_contiguous, name
        returned = cupy.asnumpy(resident)
        np.testing.assert_array_equal(returned, host.arrays[name])

    ccn = cupy.asnumpy(device.ccn_activation_table)
    assert ccn.shape == CCN_ACTIVATION_SHAPE
    assert float(ccn[0, 0, 3, 2, 1]) == _CCN_SPOT[(0, 0, 3, 2, 1)]
    assert float(ccn.min()) == _CCN_MIN and float(ccn.max()) == _CCN_MAX
    assert hashlib.sha256(
        np.asfortranarray(ccn).tobytes(order="F")).hexdigest() == \
        _CCN_WIDENED_SHA256

    # A device-side reduction, so the bytes are proven readable by a kernel
    # and not merely by a DtoH copy of a buffer that was never touched.
    on_device = device.ccn_activation_table
    assert float(cupy.asnumpy(on_device.min())) == _CCN_MIN
    assert float(cupy.asnumpy(on_device.max())) == _CCN_MAX
    assert int(cupy.asnumpy(cupy.count_nonzero(on_device > 1.0))) == 0
    thompson_aerosol_runtime._DEVICE_CACHE.clear()


@_needs_ccn
def test_device_cache_is_keyed_by_the_resolved_ccn_path(tmp_path):
    from woof.core import thompson_aerosol_runtime

    relocated = tmp_path / "wrf-run" / AEROSOL_TABLE_FILE
    relocated.parent.mkdir(parents=True)
    relocated.write_bytes(_CCN_PATH.read_bytes())

    thompson_aerosol_runtime._DEVICE_CACHE.clear()
    try:
        first = thompson_aerosol_runtime.load_aerosol_device_tables(
            _TABLES, backend=_CopyBackend, require_classic_assets=False)
        again = thompson_aerosol_runtime.load_aerosol_device_tables(
            _TABLES, backend=_CopyBackend, require_classic_assets=False)
        assert again is first
        other = thompson_aerosol_runtime.load_aerosol_device_tables(
            _TABLES, ccn_path=relocated, backend=_CopyBackend,
            require_classic_assets=False)
        assert other is not first
        assert other.ccn_path == relocated.resolve()
        assert len(thompson_aerosol_runtime._DEVICE_CACHE) == 2
        # A now-absent override must fail closed rather than be answered from
        # the cache populated above.
        relocated.unlink()
        with pytest.raises(MissingAerosolTableAsset):
            thompson_aerosol_runtime.load_aerosol_device_tables(
                _TABLES, ccn_path=relocated, backend=_CopyBackend,
                require_classic_assets=False)
    finally:
        thompson_aerosol_runtime._DEVICE_CACHE.clear()


def test_both_thompson_schemes_select_the_complete_fetched_cache(tmp_path, monkeypatch):
    from woof import physics_compat
    from woof.core.thompson_contract import CLASSIC_TABLE_ASSETS

    packaged = tmp_path / "package"
    staged = tmp_path / "fetched"
    packaged.mkdir()
    staged.mkdir()
    for asset in CLASSIC_TABLE_ASSETS:
        (staged / asset.filename).touch()
    monkeypatch.setattr(physics_compat, "packaged_thompson_table_root", lambda: packaged)
    monkeypatch.setattr(physics_compat, "user_thompson_table_root", lambda: staged)
    assert Path(physics_compat.thompson_table_root(env={})) == staged
    assert resolve_aerosol_table_root(env={}) == staged
    explicit = tmp_path / "explicit"
    assert resolve_aerosol_table_root(explicit, env={}) == explicit
    assert resolve_aerosol_table_root(env={AEROSOL_TABLE_ROOT_ENV: str(explicit)}) == explicit
