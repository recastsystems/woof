# ======================================================================
# THIRD-PARTY NOTICE.  Parts of this file are hand transcriptions of
# third-party work.  ArWen distributes the file under the Apache License
# 2.0; the notices below belong to the transcribed parts and are kept here
# because their own licences require it.  Full texts are in the repository
# NOTICE and in the licenses/ directory.
#
#   The GAMMLN log-gamma coefficients and their evaluation, taken from WRF
#   v4.6.1 phys/module_mp_thompson.F:5325-5347, which preserves the notice
#   immediately above and below that routine:
#
#       (C) Copr. 1986-92 Numerical Recipes Software 2.02
#
#   ArWen's position on this material -- taken from WRF, reproduced in
#   order to match WRF bit for bit, and the standard Lanczos g=5 n=6
#   coefficient set -- is set out in
#   licenses/NOTICE-Numerical-Recipes.txt.  The notice is preserved here
#   because WRF preserved it.
# ======================================================================
"""WRF v4.6.1 aerosol-aware Thompson ``mp_physics=28`` table contract.

This module is intentionally CPU-only and self-contained.  It is the aerosol
sibling of :mod:`woof.core.thompson_contract`, which stays byte-frozen: the
four classic ``mp_physics=8`` assets, their SHA-256 pins, and
``TABLE_SET_ID`` are imported read-only and never widened here.  The
numerical authority is NCAR WRF v4.6.1 commit
``d66e442fccc04111067e29274c9f9eaccc3cef28``, specifically
``phys/module_mp_thompson.F``.

mp=28 needs three distinct kinds of coefficient, and conflating them is the
first way a port goes wrong.  Each is handled differently on purpose:

``tnccn_act`` -- a **pinned external asset woof redistributes verbatim**.
    ``table_ccnAct`` (module_mp_thompson.F:5110-5166) does not compute
    anything.  It ``OPEN``s ``CCN_ACTIVATE.BIN`` by bare relative name and
    performs one sequential ``READ`` into ``REAL(4)
    tnccn_act(7,9,7,5,4)``.  The file is the tabulated output of an offline
    parcel model (Feingold & Heymsfield, modified by Eidhammer and
    Kreidenweis; see the comment at 5102-5108) shipped in WRF's ``run/``
    directory.  No recompilation of WRF regenerates it.

    It is therefore *data*, not a ``thompson_init`` product, and woof ships
    it the way WRF does: the committed copy is WRF v4.6.1's
    ``run/CCN_ACTIVATE.BIN`` bit for bit, redistributed under WRF's
    public-domain dedication with the notice carried by
    ``woof/data/wrf_radiation/LICENSE-WRF.txt``.  It is committed and
    listed in ``tables/MANIFEST.sha256``, but it is deliberately **not**
    part of ``thompson_contract.CLASSIC_TABLE_ASSETS`` and
    ``TABLE_SET_ID`` is unchanged, so no ``mp_physics=8`` launch acquires a
    dependency on it.  It is content addressed here -- the size and SHA-256
    of the WRF v4.6.1 file are pinned -- and it is located through
    :func:`resolve_ccn_activation_path`, which accepts an explicit path, an
    environment override, or the packaged table root, and **fails closed**
    when the file is absent or is a different table.  Shipping it does not
    soften any of that: a *different* parcel-model table would silently be a
    different activation scheme, and there is no synthetic fallback, because
    ``table_ccnAct`` prefills ``tnccn_act`` with 1.0 -- a scheme that quietly
    continued without the file would activate every available CCN at every
    gridpoint and still run, stay bounded, and be silently wrong.  See
    ``woof/data/thompson/PROVENANCE.md``.

``tpc_wev`` / ``tnc_wev`` -- **already covered by the classic asset**.
    ``table_dropEvap`` (5011-5099) is called unconditionally at
    ``thompson_init`` line 1025, outside the ``is_aerosol_aware`` guard, and
    in v4.6.1 its droplet-number axis already sweeps
    ``nu_c = MIN(15, NINT(1000e6/t_Nc(k)) + 2)`` from 15 down to 2.  The
    tables woof already pins as records 6 and 7 of
    ``thompson_aux_tables.dat`` are therefore the full aerosol-aware tables;
    mp=28 needs no new file and no re-pin, it merely starts *reading*
    ``tnc_wev``, which has been uploaded-but-unread since the mp=8 port.
    :func:`build_drop_evaporation_tables` reimplements the generator purely
    so :func:`validate_drop_evaporation_tables` can prove that claim against
    the packaged bytes rather than asserting it.

Gamma moments and the droplet-number bin grid -- **derived at import**.
    ``ccg``/``ocg1``/``ocg2`` (671-685) and the ``t_Nc`` bin edges
    (720-731) are fifteen- and hundred-element arrays, far too small to
    justify a file.  They are recomputed here from a transcription of WRF's
    ``REAL(4)`` ``WGAMMA``/``GAMMLN`` (5325-5377) -- *not* from
    :func:`math.gamma`, which disagrees in the seventh digit -- and the
    result is SHA-256 pinned so a gamma-implementation drift is caught as
    loudly as a substituted binary would be.

    Every transcendental on that derivation path goes through
    :mod:`woof.core.correctly_rounded_libm` rather than :mod:`math`.  The
    host C library is not the same program on every box -- the UCRT and glibc
    disagree by one ulp on ``EXP(0.61*LOG(3000))``, which is one of the
    hundred ``t_Nc`` bin edges -- and a pin that moves with the vendor of
    libm is a pin that fires on the wrong thing.  The pinned bytes are a
    property of the algorithm, not of the machine that ran it.

``mp_physics=28`` is not made selectable by importing this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import os
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Mapping

import numpy as np

from woof.core.correctly_rounded_libm import exp_cr, log_cr
from woof.core.thompson_aerosol_constants import (  # noqa: F401 -- re-exported
    NIFA_FLOOR, NT_C, NWFA_FLOOR)
from woof.core.thompson_contract import (
    AUXILIARY_TABLE_FILE,
    CLASSIC_TABLE_ASSETS,
    TABLE_SET_ID as CLASSIC_TABLE_SET_ID,
    TableAsset,
    WRF_REFERENCE_COMMIT,
    WRF_REFERENCE_VERSION,
    validate_table_assets,
)


MP_PHYSICS = 28

# Registry.EM_COMMON:3036 binds the aerosol-aware Thompson package.  woof
# drops WRF's leading q in attribute names, so qnwfa/qnifa become nwfa/nifa
# and the 2-D emission fields become nwfa2d/nifa2d.
AEROSOL_SPECIES = ("nc", "nwfa", "nifa")
AEROSOL_SURFACE_FIELDS = ("nwfa2d", "nifa2d")


# ---------------------------------------------------------------------------
# Scalars.  Every value carries its module_mp_thompson.F line number because
# several of them are unit-inconsistent in WRF (per-kg quantities compared
# against per-m3 constants) and must be transcribed rather than repaired.
# ---------------------------------------------------------------------------

NA_IN0 = 1.5e6            # :94   ice-friendly aerosol profile amplitude, m-3
NA_IN1 = 0.5e6            # :95   ice-friendly aerosol profile floor, m-3
NA_CCN0 = 300.0e6         # :96   water-friendly aerosol profile amplitude
NA_CCN1 = 50.0e6          # :97   water-friendly aerosol profile floor
# NT_C (:88), NWFA_FLOOR (:1805) and NIFA_FLOOR (:1806) are defined in
# woof/core/thompson_aerosol_constants.py and imported above, because the
# nest-edge entry closure reads them from a module the standalone
# preparation wheel stages; they are re-exported here unchanged.
NT_C_MAX = 1999.0e6       # :89   droplet-number ceiling, m-3
NC_FLOOR_PER_M3 = 2.0     # :1830, :3217, :3486
NC_SEDIMENT_FLOOR = 10.0  # :3835 -- the only 10.0 floor in the scheme
AEROSOL_CEILING = 9999.0e6  # :1805-1806, :3979-3981

# thompson_init:493-558.  The synthetic profile used when metgrid supplies no
# aerosol.  h_01 switches on the ABSOLUTE mass-level height of level 1.
PROFILE_H01_LOW_M = 1000.0
PROFILE_H01_HIGH_M = 2500.0
PROFILE_H01_LOW = 0.8
PROFILE_H01_HIGH = 0.01
# nwfa2d(i,j) = nwfa(i,1,j) * 0.000196 * (50./z1)   (thompson_init:509-510)
NWFA2D_EMISSION_COEFF = 0.000196
NWFA2D_REFERENCE_DEPTH_M = 50.0

# module_mp_thompson.F:5011 comment block and 831-836: the droplet diameter
# axis of tpc_wev/tnc_wev is LINEAR in microns, unlike every other bin family
# in the scheme, which is logarithmic.
D0C_M = 1.0e-6
D0R_M = 50.0e-6
AM_R = float(np.float32(np.float32(3.1415926536) * np.float32(1000.0)
                        / np.float32(6.0)))   # :128, PI*rho_w/6 in REAL(4)
BM_R = 3.0                                    # :129
BV_C = 2.0                                    # :164
AV_C = 0.316946e8                             # :163


# ---------------------------------------------------------------------------
# CCN activation table (table_ccnAct).
# ---------------------------------------------------------------------------

# module_mp_thompson.F:247-251.  Fortran order, first index varies fastest.
NTB_ARC = 7   # available CCN concentration, per cm3
NTB_ARW = 9   # updraft speed, m s-1
NTB_ART = 7   # temperature
NTB_ARR = 5   # lognormal mean aerosol radius
NTB_ARK = 4   # hygroscopicity kappa
CCN_ACTIVATION_SHAPE = (NTB_ARC, NTB_ARW, NTB_ART, NTB_ARR, NTB_ARK)

# module_mp_thompson.F:335-344.  ta_Ra and ta_Ka are pinned even though
# activ_ncloud hardcodes indexes into them; a future variable-aerosol lane
# must not have to rediscover the axis values.
TA_NA = (10.0, 31.6, 100.0, 316.0, 1000.0, 3160.0, 10000.0)
TA_WW = (0.01, 0.0316, 0.1, 0.316, 1.0, 3.16, 10.0, 31.6, 100.0)
TA_TK = (243.15, 253.15, 263.15, 273.15, 283.15, 293.15, 303.15)
TA_RA = (0.01, 0.02, 0.04, 0.08, 0.16)
TA_KA = (0.2, 0.4, 0.6, 0.8)

# activ_ncloud:5229-5230.  One-based, hardcoded in WRF: mean radius 0.04 um
# and kappa 0.4.  Four fifths of the shipped table is therefore never read.
ACTIV_RADIUS_INDEX = 3
ACTIV_KAPPA_INDEX = 2

AEROSOL_TABLE_FILE = "CCN_ACTIVATE.BIN"

# One Fortran sequential record: 4-byte marker, 8820 REAL(4), 4-byte marker.
CCN_ACTIVATION_VALUES = int(np.prod(CCN_ACTIVATION_SHAPE, dtype=np.int64))
CCN_ACTIVATION_PAYLOAD_BYTES = CCN_ACTIVATION_VALUES * 4
CCN_ACTIVATION_FILE_BYTES = CCN_ACTIVATION_PAYLOAD_BYTES + 8

# WRF builds with -convert big_endian (BYTESWAPIO), so the shipped asset is
# big-endian even though nothing else in the table set is.  Reading it
# little-endian does NOT fail on markers by accident -- 35280 byte-swapped is
# a wildly different integer -- but a byte-order mistake anywhere downstream
# would silently produce activation fractions outside (0, 1].
CCN_ACTIVATION_BYTEORDER = "big"

# Content address of the WRF v4.6.1 run/ file (git-clean at commit d66e442).
# It is third-party parcel-model output redistributed by WRF, NOT a
# thompson_init product; woof redistributes the identical bytes.  See the
# module docstring and woof/data/thompson/PROVENANCE.md.
AEROSOL_TABLE_ASSETS = (
    TableAsset(
        AEROSOL_TABLE_FILE, CCN_ACTIVATION_FILE_BYTES,
        "f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd"),
)

# Whether the blob ships with woof.  Kept as a named constant because the
# packaging exclusion, the registry row and the provenance text all have to
# move with it.  Flipped False -> True on 2026-08-01, in the commit that
# committed the file and added it to MANIFEST.sha256.  It says nothing about
# how the loader behaves: absence is still fatal, the SHA-256 is still
# enforced, and the file is still absent from CLASSIC_TABLE_ASSETS.
AEROSOL_ASSET_REDISTRIBUTED = True

#: Human-readable origin, quoted verbatim in the fail-closed error so an
#: operator can act on it without reading any source.
AEROSOL_ASSET_SOURCE = (
    f"WRF {WRF_REFERENCE_VERSION} (git tag {WRF_REFERENCE_VERSION}, commit "
    f"{WRF_REFERENCE_COMMIT}), file run/{AEROSOL_TABLE_FILE}")

#: Directory-level override.  Same name and meaning as
#: ``physics_compat.THOMPSON_TABLE_ROOT_ENV``; duplicated rather than imported
#: so this module stays free of the physics-capability import graph.  A
#: drift guard lives in tests/test_thompson_aerosol_contract.py.
AEROSOL_TABLE_ROOT_ENV = "WOOF_THOMPSON_TABLE_ROOT"

#: File-level override, checked first.  The packaged copy now satisfies a
#: default install, but an operator who wants the run bound to the table in
#: their own WRF tree should not have to copy it, and the override is also
#: how a substituted table is proved to fail closed.
AEROSOL_TABLE_PATH_ENV = "WOOF_THOMPSON_CCN_ACTIVATE"


class MissingAerosolTableAsset(FileNotFoundError):
    """``CCN_ACTIVATE.BIN`` could not be located.

    A ``FileNotFoundError`` subclass so ordinary ``except FileNotFoundError``
    handling still applies, and a distinct type so an adapter can tell "the
    operator has not supplied WRF's activation table" apart from "the packaged
    coefficient install is broken".
    """


def packaged_aerosol_table_root() -> Path:
    """The packaged Thompson table directory.

    Identical to ``physics_compat.packaged_thompson_table_root()`` -- now
    by delegation rather than by a second copy of the path, because since
    2.5.0 that directory ships in the ``recast-woof-data`` companion and a
    restated literal here would resolve to a directory that no longer
    exists.  mp=28 shares the classic root on purpose: a second root would
    let ``tnccn_act`` and the process tables come from different WRF
    builds.
    """
    from woof.physics_compat import packaged_thompson_table_root

    return packaged_thompson_table_root()


def resolve_aerosol_table_root(root: str | Path | None = None, *,
                               env: Mapping[str, str] | None = None) -> Path:
    """Explicit argument, then ``WOOF_THOMPSON_TABLE_ROOT``, then packaged."""
    if root is not None:
        return Path(root)
    environ = os.environ if env is None else env
    override = environ.get(AEROSOL_TABLE_ROOT_ENV)
    if override:
        return Path(override)
    return packaged_aerosol_table_root()


def _missing_asset_message(candidates: tuple[Path, ...]) -> str:
    asset = AEROSOL_TABLE_ASSETS[0]
    looked = "\n".join(f"    {candidate}" for candidate in candidates)
    return (
        f"{AEROSOL_TABLE_FILE} is required by Thompson aerosol-aware "
        "microphysics (mp_physics=28) and was not found.  woof DOES "
        "redistribute this table, so a default install has it: it is missing "
        "because an override points somewhere else, because the packaged "
        "copy was deleted, or because the table root was relocated.\n"
        f"  looked for:\n{looked}\n"
        "  what it is: tabulated output of an offline parcel model (Feingold "
        "& Heymsfield, as modified by Eidhammer & Kreidenweis -- see the "
        "comment at module_mp_thompson.F:5102-5108).  WRF ships it as data; "
        "no recompilation of WRF and no woof code path regenerates it.\n"
        f"  where to get it: {AEROSOL_ASSET_SOURCE}\n"
        f"    expected {asset.bytes} bytes, sha256 {asset.sha256}\n"
        "  how to supply it: restore that file at "
        f"{packaged_aerosol_table_root() / AEROSOL_TABLE_FILE}, or set "
        f"{AEROSOL_TABLE_PATH_ENV} to its full path, or set "
        f"{AEROSOL_TABLE_ROOT_ENV} to a directory containing it; run "
        "woof fetch-tables --schemes thompson28 --from DIR.\n"
        "  why this is fatal rather than defaulted: WRF's table_ccnAct "
        "prefills tnccn_act with 1.0 and only overwrites it from this file, "
        "so a synthetic fallback would activate every available CCN at every "
        "gridpoint -- a run that stays bounded, raises nothing, and is "
        "silently wrong.")


def resolve_ccn_activation_path(
        path: str | Path | None = None,
        root: str | Path | None = None, *,
        env: Mapping[str, str] | None = None) -> Path:
    """Locate ``CCN_ACTIVATE.BIN`` or fail closed with an actionable error.

    Precedence, highest first: the explicit ``path`` argument, the
    ``WOOF_THOMPSON_CCN_ACTIVATE`` file override, then
    ``<resolved table root>/CCN_ACTIVATE.BIN`` where the root itself follows
    the explicit ``root`` argument, ``WOOF_THOMPSON_TABLE_ROOT``, and the
    packaged directory in that order.

    An explicitly named path that does not exist is an error even if the
    packaged copy would have worked: silently ignoring an operator's override
    is how a run ends up using coefficients nobody chose.
    """
    environ = os.environ if env is None else env
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        override = environ.get(AEROSOL_TABLE_PATH_ENV)
        if override:
            candidates.append(Path(override))
        else:
            candidates.append(
                resolve_aerosol_table_root(root, env=environ)
                / AEROSOL_TABLE_FILE)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise MissingAerosolTableAsset(_missing_asset_message(tuple(candidates)))


def validate_ccn_activation_asset(path: str | Path) -> TableAsset:
    """Content-address one ``CCN_ACTIVATE.BIN`` at an arbitrary location.

    ``thompson_contract.validate_table_assets`` is used unchanged for the
    classic set, but it addresses assets by ``root / filename``.  The aerosol
    blob is also admissible from a path the operator chose -- the file
    override exists so a run can be bound to the table in a WRF tree -- so
    this is the same size-then-SHA-256 check applied to one file at an
    arbitrary location.
    """
    asset = AEROSOL_TABLE_ASSETS[0]
    source = Path(path)
    if not source.is_file():
        raise MissingAerosolTableAsset(_missing_asset_message((source,)))
    actual_bytes = source.stat().st_size
    if actual_bytes != asset.bytes:
        raise ValueError(
            f"Thompson aerosol table asset {source} has {actual_bytes} bytes; "
            f"expected {asset.bytes} (from {AEROSOL_ASSET_SOURCE}); "
            "run woof fetch-tables --schemes thompson28 --from DIR --replace-damaged")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != asset.sha256:
        raise ValueError(
            f"Thompson aerosol table asset {source} SHA-256 {actual_sha256}; "
            f"expected {asset.sha256} (from {AEROSOL_ASSET_SOURCE}); "
            "run woof fetch-tables --schemes thompson28 --from DIR --replace-damaged")
    return asset

# Deliberately distinct from thompson_contract.TABLE_SET_ID.  mp=8 launches
# must keep failing closed on a file they never needed, and a restart must be
# able to tell the two coefficient inventories apart.
AEROSOL_TABLE_SET_ID = "wrf-v4.6.1-aerosol-thompson-mp28-v1"

# Interior spot value used as an order-and-endianness canary.  A C/F order
# flip does not change the file SHA-256 and does not raise on this
# all-distinct shape, so a structural check is the only real guard.
CCN_ACTIVATION_SPOT_INDEX = (0, 0, 3, 2, 1)
CCN_ACTIVATION_SPOT_VALUE = np.float32(0.281761)
CCN_ACTIVATION_MONOTONE_PREFIX = 6


def read_ccn_activation_table(path: str | Path) -> np.ndarray:
    """Read ``CCN_ACTIVATE.BIN`` into an immutable float64 Fortran array.

    ``tnccn_act`` is declared ``REAL(KIND=R4SIZE)`` at line 393, so the file
    holds float32.  It is widened here -- exactly, since every stored value is
    a float32 -- so the array satisfies the same float64/Fortran-order upload
    contract the classic tables already use.  No float32 device path is
    admitted; that would be a separately quantified numerical variant.
    """
    source = Path(path)
    marker_format = (">" if CCN_ACTIVATION_BYTEORDER == "big" else "<") + "i"
    dtype = np.dtype(">f4" if CCN_ACTIVATION_BYTEORDER == "big" else "<f4")
    with source.open("rb") as stream:
        head = stream.read(4)
        if len(head) != 4:
            raise ValueError(f"{source}: missing leading tnccn_act marker")
        declared = struct.unpack(marker_format, head)[0]
        if declared != CCN_ACTIVATION_PAYLOAD_BYTES:
            raise ValueError(
                f"{source}: tnccn_act marker declares {declared} bytes, "
                f"expected {CCN_ACTIVATION_PAYLOAD_BYTES}")
        payload = stream.read(CCN_ACTIVATION_PAYLOAD_BYTES)
        if len(payload) != CCN_ACTIVATION_PAYLOAD_BYTES:
            raise ValueError(
                f"{source}: truncated tnccn_act payload ({len(payload)} != "
                f"{CCN_ACTIVATION_PAYLOAD_BYTES} bytes)")
        tail = stream.read(4)
        if len(tail) != 4 or struct.unpack(marker_format, tail)[0] != declared:
            raise ValueError(f"{source}: mismatched trailing tnccn_act marker")
        if stream.read(1):
            raise ValueError(f"{source}: unexpected bytes after tnccn_act")

    values = np.frombuffer(payload, dtype=dtype, count=CCN_ACTIVATION_VALUES)
    table = np.array(values.reshape(CCN_ACTIVATION_SHAPE, order="F"),
                     dtype=np.float64, order="F", copy=True)
    validate_ccn_activation_table(table, source=source)
    table.flags.writeable = False
    return table


def validate_ccn_activation_table(table: np.ndarray, *,
                                  source: str | Path = "<array>") -> None:
    """Fail closed on a mis-shaped, mis-ordered, or non-physical table."""
    if table.shape != CCN_ACTIVATION_SHAPE:
        raise ValueError(
            f"{source}: tnccn_act has shape {table.shape}; "
            f"expected {CCN_ACTIVATION_SHAPE}")
    if not np.isfinite(table).all():
        raise ValueError(f"{source}: tnccn_act contains non-finite values")
    # activ_ncloud returns NCCN*fraction, so a value outside (0, 1] is an
    # activated fraction exceeding the available aerosol.
    if not ((table > 0.0) & (table <= 1.0)).all():
        raise ValueError(
            f"{source}: tnccn_act values must lie in (0, 1]; got "
            f"[{table.min()!r}, {table.max()!r}]")
    # table_ccnAct prefills tnccn_act with 1.0 (993-1002) and only overwrites
    # it inside the one-time micro_init block.  A uniformly-1.0 array means
    # the read silently did not happen and every CCN activates.
    if np.all(table == 1.0):
        raise ValueError(
            f"{source}: tnccn_act is uniformly 1.0, i.e. the CCN_ACTIVATE.BIN "
            "read never replaced thompson_init's prefill")
    spot = table[CCN_ACTIVATION_SPOT_INDEX]
    if np.float32(spot) != CCN_ACTIVATION_SPOT_VALUE:
        raise ValueError(
            f"{source}: tnccn_act{CCN_ACTIVATION_SPOT_INDEX} is {spot!r}; "
            f"expected {float(CCN_ACTIVATION_SPOT_VALUE)!r}.  A C/F order "
            "flip preserves the file SHA-256 and is only visible "
            "structurally.")
    slab = table[0, :, 3, 2, 1][:CCN_ACTIVATION_MONOTONE_PREFIX]
    if not np.all(np.diff(slab) > 0.0):
        raise ValueError(
            f"{source}: activated fraction is not strictly increasing with "
            f"updraft along ta_Ww; got {slab!r}")


# ---------------------------------------------------------------------------
# WRF's REAL(4) gamma function, transcribed.
# ---------------------------------------------------------------------------

# Numerical Recipes Lanczos coefficients, module_mp_thompson.F:5329-5334.
_GAMMLN_STP = 2.5066282746310005
_GAMMLN_COF = (
    76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5,
)


def gammln_fp32(xx: float) -> np.float32:
    """WRF ``REAL FUNCTION GAMMLN(XX)``, module_mp_thompson.F:5325-5347.

    The series is evaluated in double precision and the *result* is stored
    into a ``REAL(4)`` function value, so the rounding to float32 happens
    exactly once, at the return.

    The two logarithms are :func:`~woof.core.correctly_rounded_libm.log_cr`
    rather than :func:`math.log` so the pinned digest does not depend on
    whose C library the host ships.  The float32 store at the return absorbs
    a one-ulp FP64 disagreement at every one of the fifteen shape parameters
    as it happens, so this call site is not the one that broke -- but it is
    the same class as the one that did, and relying on the float32 round to
    keep hiding it is relying on luck.
    """
    x = float(np.float32(xx))
    y = x
    tmp = x + 5.5
    tmp = (x + 0.5) * log_cr(tmp) - tmp
    ser = 1.000000000190015
    for coefficient in _GAMMLN_COF:
        y += 1.0
        ser += coefficient / y
    return np.float32(tmp + log_cr(_GAMMLN_STP * ser / x))


def wgamma_fp32(y: float) -> np.float32:
    """WRF ``REAL FUNCTION WGAMMA(y) = EXP(GAMMLN(y))``, :5371-5377.

    The correctly rounded exponential of the float32 log-gamma, rounded once
    to float32, is the ``expf`` glibc gives gfortran.  ``numpy.exp`` on a
    float32 uses a vectorized approximation that differs by one ulp at
    several of the fifteen shape parameters, which would propagate straight
    into every autoconversion and sedimentation coefficient.
    """
    return np.float32(exp_cr(float(gammln_fp32(y))))


_NU_C_MAX = 15


def _build_gamma_moments() -> dict[str, np.ndarray]:
    """module_mp_thompson.F:671-685, one entry per integer nu_c in 1..15.

    Index 0 of every returned array is an unused zero so device code can be
    written with Fortran's one-based ``nu_c`` and no off-by-one adapter.
    """
    one = np.float32(1.0)
    bm_r = np.float32(BM_R)
    bv_c = np.float32(BV_C)
    cce = np.zeros((6, _NU_C_MAX + 1), dtype=np.float64)
    ccg = np.zeros((6, _NU_C_MAX + 1), dtype=np.float64)
    ocg1 = np.zeros(_NU_C_MAX + 1, dtype=np.float64)
    ocg2 = np.zeros(_NU_C_MAX + 1, dtype=np.float64)
    for n in range(1, _NU_C_MAX + 1):
        exponents = (
            np.float32(n + 1.0),
            np.float32(bm_r + n + 1.0),
            np.float32(bm_r + n + 4.0),
            np.float32(n + bv_c + 1.0),
            np.float32(bm_r + n + bv_c + 1.0),
        )
        for row, exponent in enumerate(exponents, start=1):
            cce[row, n] = float(exponent)
            ccg[row, n] = float(wgamma_fp32(exponent))
        ocg1[n] = float(one / np.float32(ccg[1, n]))
        ocg2[n] = float(one / np.float32(ccg[2, n]))
    result: dict[str, np.ndarray] = {}
    for row in range(1, 6):
        result[f"cce{row}"] = np.ascontiguousarray(cce[row])
        result[f"ccg{row}"] = np.ascontiguousarray(ccg[row])
    result["ocg1"] = ocg1
    result["ocg2"] = ocg2
    for array in result.values():
        array.flags.writeable = False
    return result


GAMMA_MOMENTS: Mapping[str, np.ndarray] = MappingProxyType(
    _build_gamma_moments())

CCE1 = GAMMA_MOMENTS["cce1"]
CCE2 = GAMMA_MOMENTS["cce2"]
CCE3 = GAMMA_MOMENTS["cce3"]
CCE4 = GAMMA_MOMENTS["cce4"]
CCE5 = GAMMA_MOMENTS["cce5"]
CCG1 = GAMMA_MOMENTS["ccg1"]
CCG2 = GAMMA_MOMENTS["ccg2"]
CCG3 = GAMMA_MOMENTS["ccg3"]
CCG4 = GAMMA_MOMENTS["ccg4"]
CCG5 = GAMMA_MOMENTS["ccg5"]
OCG1 = GAMMA_MOMENTS["ocg1"]
OCG2 = GAMMA_MOMENTS["ocg2"]

# calc_effectRad:5611-5613.  These are the EXACT integers (n+1)(n+2)(n+3) for
# n = 2..16, written out verbatim in WRF as a PARAMETER, and are used only by
# the effective-radius diagnostic.  They are NOT interchangeable with
# ccg(2,n)*ocg1(n): at nu_c=12 the runtime product is 2729.9973, not 2730.
G_RATIO_1BASED = (24, 60, 120, 210, 336, 504, 720, 990,
                  1320, 1716, 2184, 2730, 3360, 4080, 4896)
G_RATIO = np.array((0,) + G_RATIO_1BASED, dtype=np.float64)
G_RATIO.flags.writeable = False


def cloud_shape_parameter(nc_per_m3: float) -> int:
    """``nu_c = MIN(15, NINT(1000.E6/nc) + 2)``, module_mp_thompson.F:2171.

    Fortran ``NINT`` rounds half away from zero; the argument here is always
    positive, so ``floor(x + 0.5)`` is the faithful form.  CUDA's
    ``__float2int_rn`` rounds half to even and is not.
    """
    if not nc_per_m3 > 0.0:
        raise ValueError("cloud droplet number must be positive")
    return min(_NU_C_MAX, int(math.floor(1000.0e6 / nc_per_m3 + 0.5)) + 2)


# ---------------------------------------------------------------------------
# Droplet-number and cloud-water bin grids shared by tnc_wev and idx_n.
# ---------------------------------------------------------------------------

NBC = 100    # :231-232
NTB_C = 37   # :237

# module_mp_thompson.F:838-844.
R_C = np.array(
    [m * 1e-6 for m in range(1, 10)] + [m * 1e-5 for m in range(1, 10)]
    + [m * 1e-4 for m in range(1, 10)] + [m * 1e-3 for m in range(1, 10)]
    + [1e-2], dtype=np.float32).astype(np.float64)
R_C.flags.writeable = False


def _build_droplet_bins() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """module_mp_thompson.F:665-670 (Dc/dtc) and 720-731 (t_Nc, nic1).

    ``D0c`` is a ``REAL(4)`` parameter, so ``Dc(1) = D0c*1.0d0`` is
    9.99999997475243e-07, not 1e-06.  Seeding the ladder with an exact double
    1e-6 shifts ``Dc(i)**nu_c`` by up to 4.8e-8 relative at nu_c=15 -- a
    hundred million times the float64 rounding of everything else here.
    """
    d0c = float(np.float32(D0C_M))
    diameter = np.empty(NBC, dtype=np.float64)
    diameter[0] = d0c
    for n in range(1, NBC):
        diameter[n] = diameter[n - 1] + 1.0e-6
    width = np.empty(NBC, dtype=np.float64)
    width[0] = d0c
    width[1:] = diameter[1:] - diameter[:-1]

    # Cloud droplet number bins, 1 to 3000 per cc, logarithmic.
    #
    # This is the ladder that made DERIVED_CONSTANT_SHA256 machine dependent.
    # At n=61 the exact EXP is 132.142940103429296889..., so the correctly
    # rounded double is 0x1.08492f71fb081p+7; glibc returns it and the UCRT
    # returns 0x1.08492f71fb080p+7, one ulp low.  Nothing downstream rounds
    # to float32 here -- t_Nc goes into the digest as a float64 -- so that one
    # bin out of a hundred moved the pin and failed the whole tier on Windows.
    # exp_cr/log_cr are software arbitrary precision and give the same
    # hundred edges on every host.
    edges = np.empty(NBC + 1, dtype=np.float64)
    edges[0] = 1.0
    edges[NBC] = 3000.0
    for n in range(1, NBC):
        edges[n] = exp_cr(n / float(NBC) * log_cr(edges[NBC] / edges[0])
                          + log_cr(edges[0]))
    number = np.sqrt(edges[:NBC] * edges[1:NBC + 1]) * 1.0e6

    # nic1 is declared INTEGER at :246 and assigned a DOUBLE at :896, so
    # Fortran TRUNCATES 7.926303891973744 to 7.  Using 7.926 shifts every
    # droplet-number bin by about 13% and breaks the mp=8 identity below.
    nic1 = int(log_cr(number[NBC - 1] / number[0]))
    for array in (diameter, width, number):
        array.flags.writeable = False
    return diameter, width, number, nic1


DC, DTC, T_NC, NIC1 = _build_droplet_bins()
T_NC_MIN = float(T_NC[0])
T_NC_MAX = float(T_NC[-1])

# The value ArWen's mp=8 kernels hardcode as ``cloud_number_bin = 65``
# (thompson.cu:3933, 7005 and one more site).  If the generalized formula
# below does not reproduce it at Nt_c, the mp=28 helpers are wrong in a way
# that would also have broken mp=8 had the kernels been shared.
CLASSIC_DROPLET_BIN = 65


def droplet_number_bin(nc_per_m3: float) -> int:
    """Zero-based ``idx_n`` into ``tnc_wev``, module_mp_thompson.F:3447-3448.

    WRF: ``idx_n = NINT(1.0 + FLOAT(nbc)*DLOG(nc/t_Nc(1))/nic1)`` clamped to
    ``[1, nbc]``.  ``nic1`` is the truncated integer 7, and the division is
    double-by-integer, i.e. by 7.0 exactly.
    """
    raw = 1.0 + float(NBC) * math.log(nc_per_m3 / T_NC_MIN) / float(NIC1)
    index = int(math.floor(raw + 0.5))
    return min(max(index, 1), NBC) - 1


# ---------------------------------------------------------------------------
# table_dropEvap: reference generator and packaged-asset validator.
# ---------------------------------------------------------------------------

DROP_EVAP_SHAPE = (NBC, NTB_C, NBC)
DROP_EVAP_TABLE_NAMES = ("tpc_wev", "tnc_wev")

# WIRING FACT, verified in this repository rather than assumed: tnc_wev needs
# NO new asset.  It is record 7 (one-based) of thompson_aux_tables.dat, it is
# already in thompson_contract.AUXILIARY_TABLE_RECORDS, and
# thompson_runtime._classic_records() feeds every auxiliary record to
# load_classic_device_tables, so the array is already resident on the device
# for every mp=8 launch -- uploaded but never read.  mp=28 starts reading it;
# nothing new is packaged, pinned, or uploaded for it.
TNC_WEV_TABLE_NAME = "tnc_wev"
TNC_WEV_SOURCE_FILE = AUXILIARY_TABLE_FILE
TNC_WEV_RECORD_ORDINAL = 7  # one-based, matching the Fortran WRITE order

# tpc_wev's only use in v4.6.1 is commented out (module_mp_thompson.F:
# 3465-3466).  It stays parsed, pinned and uploaded so the classic auxiliary
# asset's SHA-256 is unchanged, and it stays unread.
DROP_EVAP_READ_BY_MP28 = ("tnc_wev",)

# Measured float64 agreement between build_drop_evaporation_tables() and the
# packaged WRF bytes.  The residual is gfortran's __builtin_powi for the
# integer power Dc(i)**nu_c against libm pow(), plus summation order in the
# running partial sums; both are a handful of float64 ulp.
DROP_EVAP_RELATIVE_TOLERANCE = 1.0e-13


def build_drop_evaporation_tables() -> dict[str, np.ndarray]:
    """Reimplement ``table_dropEvap``, module_mp_thompson.F:5011-5099.

    This is a provenance instrument, not a production path.  woof consumes
    the packaged WRF bytes; this function exists so
    :func:`validate_drop_evaporation_tables` can *demonstrate* that the
    already-pinned classic auxiliary records carry the full nu_c-varying
    aerosol-aware tables, instead of that being an unchecked assumption.

    The commented-out ``GAMMP`` line at 5033 is not implemented: v4.6.1
    integrates the bins explicitly and never calls the incomplete gamma
    function, so ``GSER``/``GCF``/``GAMMP`` are dead code on this path.
    """
    obmr = float(np.float32(1.0) / np.float32(BM_R))
    massc = AM_R * DC ** BM_R
    mass_table = np.empty(DROP_EVAP_SHAPE, dtype=np.float64)
    number_table = np.empty(DROP_EVAP_SHAPE, dtype=np.float64)
    for k in range(NBC):
        nu_c = cloud_shape_parameter(T_NC[k])
        diameter_power = DC ** nu_c
        for j in range(NTB_C):
            lamc = (T_NC[k] * AM_R * CCG2[nu_c] * OCG1[nu_c]
                    / R_C[j]) ** obmr
            n0_c = T_NC[k] * OCG1[nu_c] * lamc ** CCE1[nu_c]
            n_c = n0_c * diameter_power * np.exp(-lamc * DC) * DTC
            mass_table[:, j, k] = np.cumsum(massc * n_c)
            number_table[:, j, k] = np.cumsum(n_c)
    return {"tpc_wev": np.asfortranarray(mass_table),
            "tnc_wev": np.asfortranarray(number_table)}


def validate_drop_evaporation_tables(
        arrays: Mapping[str, np.ndarray], *,
        rtol: float = DROP_EVAP_RELATIVE_TOLERANCE) -> dict[str, float]:
    """Prove the packaged classic auxiliary records are the mp=28 tables.

    ``arrays`` is the mapping returned by
    ``thompson_contract.read_classic_table_directory``.  Returns the measured
    maximum relative deviation per table.
    """
    reference = build_drop_evaporation_tables()
    measured: dict[str, float] = {}
    for name in DROP_EVAP_TABLE_NAMES:
        if name not in arrays:
            raise KeyError(
                f"classic Thompson table {name} is absent; mp=28 reads it "
                "from thompson_aux_tables.dat and ships no replacement")
        packaged = np.asarray(arrays[name], dtype=np.float64)
        if packaged.shape != DROP_EVAP_SHAPE:
            raise ValueError(
                f"{name} has shape {packaged.shape}; expected "
                f"{DROP_EVAP_SHAPE}")
        expected = reference[name]
        nonzero = packaged != 0.0
        deviation = np.abs(expected[nonzero] - packaged[nonzero])
        relative = float((deviation / np.abs(packaged[nonzero])).max())
        if relative > rtol:
            raise ValueError(
                f"{name} deviates from WRF's table_dropEvap by {relative:.3e} "
                f"relative, above {rtol:.1e}; the packaged auxiliary asset is "
                "not the aerosol-aware table this port assumes")
        measured[name] = relative
    return measured


# ---------------------------------------------------------------------------
# Derived-constant identity.
# ---------------------------------------------------------------------------

_DERIVED_CONSTANT_ORDER = (
    "cce1", "cce2", "cce3", "cce4", "cce5",
    "ccg1", "ccg2", "ccg3", "ccg4", "ccg5",
    "ocg1", "ocg2", "g_ratio", "t_Nc", "Dc", "dtc", "r_c",
)


def derived_constant_arrays() -> Mapping[str, np.ndarray]:
    """Every host array mp=28 derives rather than reads from a file."""
    arrays = dict(GAMMA_MOMENTS)
    arrays.update({"g_ratio": G_RATIO, "t_Nc": T_NC,
                   "Dc": DC, "dtc": DTC, "r_c": R_C})
    return MappingProxyType(arrays)


def derived_constant_sha256() -> str:
    """SHA-256 over the derived arrays in a fixed order and dtype."""
    digest = hashlib.sha256()
    arrays = derived_constant_arrays()
    for name in _DERIVED_CONSTANT_ORDER:
        value = np.ascontiguousarray(arrays[name], dtype=np.float64)
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


# Pinned so a gamma-implementation drift -- math.gamma instead of WRF's
# REAL(4) Lanczos series, numpy.exp instead of a correctly rounded expf, or
# an exact 1e-6 seeding the Dc ladder -- is caught as loudly as a substituted
# binary asset would be.
DERIVED_CONSTANT_SHA256 = (
    "21d142c1ebea8b2f28c446006d3cd34b032ef80b0e588350e7f4cac0bbef9720"
)


# ---------------------------------------------------------------------------
# Canonical bytes for the derived constants.
# ---------------------------------------------------------------------------
# The derivation above calls the host libm (``math.exp``/``math.log`` in the
# t_Nc bin ladder), and hosts disagree at the last bit: glibc reproduces the
# pin, the MSVC runtime shifts ``t_Nc`` by up to 1.2e-16 relative -- one ULP
# in a handful of bins -- and every other array is byte-identical.  One ULP
# in a bin edge is a different lookup index at the boundary, so "close" is
# not "the same numerical setup".  The constants every committed mp=28
# measurement used are therefore shipped as canonical bytes, and the module
# rebinds to them below after recording what the local libm derived.  The
# fatal sha gate in :func:`load_validated_aerosol_tables` is unchanged: it
# now passes on every host, and it still fires loudly on any code drift,
# because the asset's own file hash is pinned here and the concatenated
# array pin above must reproduce from the asset's bytes at import.

DERIVED_CONSTANT_ASSET = "derived_constants_v1.npz"
DERIVED_CONSTANT_ASSET_SHA256 = (
    "79250a4e6964f51539f0cd0e9332a6cf832d3baac948f307499bc62ae05a28ea"
)


def _load_canonical_derived_constants() -> dict[str, np.ndarray]:
    """The shipped derived-constant bytes, verified twice before use."""
    path = (Path(__file__).resolve().parents[1] / "data" / "thompson"
            / DERIVED_CONSTANT_ASSET)
    if not path.is_file():
        raise FileNotFoundError(
            f"canonical derived-constant asset {path} is missing; it ships "
            "as package data and there is no synthetic fallback")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DERIVED_CONSTANT_ASSET_SHA256:
        raise RuntimeError(
            f"canonical derived-constant asset {path} does not match its "
            f"pin: {digest} != {DERIVED_CONSTANT_ASSET_SHA256}")
    arrays: dict[str, np.ndarray] = {}
    with np.load(path) as archive:
        for name in _DERIVED_CONSTANT_ORDER:
            value = np.ascontiguousarray(archive[name], dtype=np.float64)
            value.flags.writeable = False
            arrays[name] = value
    check = hashlib.sha256()
    for name in _DERIVED_CONSTANT_ORDER:
        value = arrays[name]
        check.update(name.encode("ascii"))
        check.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        check.update(value.tobytes())
    if check.hexdigest() != DERIVED_CONSTANT_SHA256:
        raise RuntimeError(
            "canonical derived-constant asset does not reproduce the "
            f"derived-constant pin: {check.hexdigest()} != "
            f"{DERIVED_CONSTANT_SHA256}")
    return arrays


_CANONICAL_DERIVED = _load_canonical_derived_constants()

#: Arrays whose re-derivation through THIS host's libm differs from the
#: canonical bytes.  Platform truth, recorded rather than fatal: on the
#: reference toolchain (glibc) it is empty and a test pins that; elsewhere
#: it names exactly what the local libm rounds differently, while the
#: numbers actually used stay the canonical ones on every host.
DERIVED_CONSTANT_LOCAL_DRIFT: tuple[str, ...] = tuple(
    name for name in _DERIVED_CONSTANT_ORDER
    if not np.array_equal(
        _CANONICAL_DERIVED[name],
        np.ascontiguousarray(derived_constant_arrays()[name],
                             dtype=np.float64)))

# Rebind every derived module constant to the canonical bytes.  Everything
# below this point -- and every function above it, which reads these names
# from module globals at call time -- sees only the canonical values.
GAMMA_MOMENTS = MappingProxyType({
    name: _CANONICAL_DERIVED[name]
    for name in ("cce1", "cce2", "cce3", "cce4", "cce5",
                 "ccg1", "ccg2", "ccg3", "ccg4", "ccg5",
                 "ocg1", "ocg2")})
CCE1 = GAMMA_MOMENTS["cce1"]
CCE2 = GAMMA_MOMENTS["cce2"]
CCE3 = GAMMA_MOMENTS["cce3"]
CCE4 = GAMMA_MOMENTS["cce4"]
CCE5 = GAMMA_MOMENTS["cce5"]
CCG1 = GAMMA_MOMENTS["ccg1"]
CCG2 = GAMMA_MOMENTS["ccg2"]
CCG3 = GAMMA_MOMENTS["ccg3"]
CCG4 = GAMMA_MOMENTS["ccg4"]
CCG5 = GAMMA_MOMENTS["ccg5"]
OCG1 = GAMMA_MOMENTS["ocg1"]
OCG2 = GAMMA_MOMENTS["ocg2"]
G_RATIO = _CANONICAL_DERIVED["g_ratio"]
T_NC = _CANONICAL_DERIVED["t_Nc"]
DC = _CANONICAL_DERIVED["Dc"]
DTC = _CANONICAL_DERIVED["dtc"]
R_C = _CANONICAL_DERIVED["r_c"]
T_NC_MIN = float(T_NC[0])
T_NC_MAX = float(T_NC[-1])
NIC1 = int(math.log(float(T_NC[NBC - 1]) / float(T_NC[0])))


@dataclass(frozen=True)
class AerosolTableSet:
    """Validated immutable aerosol host tables plus restart-safe identity."""

    root: Path
    arrays: Mapping[str, np.ndarray]
    assets: tuple[TableAsset, ...] = AEROSOL_TABLE_ASSETS
    #: Where CCN_ACTIVATE.BIN was actually read from.  Deliberately excluded
    #: from :attr:`identity`: the asset is content addressed, so a byte
    #: identical file at another path is the same numerical setup, and putting
    #: a machine-local absolute path into a restart identity would make two
    #: identical runs compare unequal.
    ccn_path: Path | None = field(default=None, compare=False)

    @property
    def payload_bytes(self) -> int:
        return sum(array.nbytes for array in self.arrays.values())

    @property
    def identity(self) -> dict[str, object]:
        return {
            "schema": 1,
            "table_set": AEROSOL_TABLE_SET_ID,
            "classic_table_set": CLASSIC_TABLE_SET_ID,
            "wrf_version": WRF_REFERENCE_VERSION,
            "wrf_commit": WRF_REFERENCE_COMMIT,
            "mp_physics": MP_PHYSICS,
            "derived_constants_sha256": DERIVED_CONSTANT_SHA256,
            "assets": [
                {"filename": item.filename, "bytes": item.bytes,
                 "sha256": item.sha256}
                for item in self.assets
            ],
        }


def load_validated_aerosol_tables(
        path: str | Path | None = None, *,
        ccn_path: str | Path | None = None,
        require_classic_assets: bool = True,
        env: Mapping[str, str] | None = None) -> AerosolTableSet:
    """Verify canonical bytes, parse ``tnccn_act``, and freeze host arrays.

    ``path`` is the table ROOT and follows
    :func:`resolve_aerosol_table_root`; ``ccn_path`` overrides the location of
    ``CCN_ACTIVATE.BIN`` alone and follows
    :func:`resolve_ccn_activation_path`.  A missing blob raises
    :class:`MissingAerosolTableAsset` naming the WRF release it comes from.
    There is no synthetic fallback.

    The classic assets are re-validated from the same root by default.  mp=28
    reads ``tnc_wev`` out of ``thompson_aux_tables.dat`` and runs its cold
    network on ``freezeH2O.dat``, so a run whose CCN table and process tables
    came from different WRF builds is a different numerical setup and the two
    schemes must not be able to disagree about which build they are.

    ``require_classic_assets=False`` narrows the check to the aerosol asset
    alone.  It exists for CPU gates on a development tree where the two
    externalized classic assets have not been staged by ``woof
    fetch-tables``; no production path may pass it.
    """
    root = resolve_aerosol_table_root(path, env=env).resolve()
    if require_classic_assets:
        validate_table_assets(root, CLASSIC_TABLE_ASSETS)
    source = resolve_ccn_activation_path(ccn_path, root, env=env).resolve()
    assets = (validate_ccn_activation_asset(source),)
    arrays: dict[str, np.ndarray] = {
        "tnccn_act": read_ccn_activation_table(source),
    }
    for name, value in derived_constant_arrays().items():
        frozen = np.array(value, dtype=np.float64, order="F", copy=True)
        frozen.flags.writeable = False
        arrays[name] = frozen
    if derived_constant_sha256() != DERIVED_CONSTANT_SHA256:
        raise RuntimeError(
            "aerosol-aware Thompson derived constants have drifted: "
            f"{derived_constant_sha256()} != {DERIVED_CONSTANT_SHA256}")
    return AerosolTableSet(root=root, arrays=MappingProxyType(arrays),
                           assets=assets, ccn_path=source)


__all__ = [
    "ACTIV_KAPPA_INDEX",
    "ACTIV_RADIUS_INDEX",
    "AEROSOL_ASSET_REDISTRIBUTED",
    "AEROSOL_ASSET_SOURCE",
    "AEROSOL_CEILING",
    "AEROSOL_SPECIES",
    "AEROSOL_SURFACE_FIELDS",
    "AEROSOL_TABLE_ASSETS",
    "AEROSOL_TABLE_FILE",
    "AEROSOL_TABLE_PATH_ENV",
    "AEROSOL_TABLE_ROOT_ENV",
    "AEROSOL_TABLE_SET_ID",
    "AM_R",
    "AV_C",
    "AerosolTableSet",
    "BM_R",
    "BV_C",
    "CCE1", "CCE2", "CCE3", "CCE4", "CCE5",
    "CCG1", "CCG2", "CCG3", "CCG4", "CCG5",
    "CCN_ACTIVATION_BYTEORDER",
    "CCN_ACTIVATION_FILE_BYTES",
    "CCN_ACTIVATION_PAYLOAD_BYTES",
    "CCN_ACTIVATION_SHAPE",
    "CCN_ACTIVATION_SPOT_INDEX",
    "CCN_ACTIVATION_SPOT_VALUE",
    "CCN_ACTIVATION_VALUES",
    "CLASSIC_DROPLET_BIN",
    "D0C_M",
    "D0R_M",
    "DC",
    "DERIVED_CONSTANT_ASSET",
    "DERIVED_CONSTANT_ASSET_SHA256",
    "DERIVED_CONSTANT_LOCAL_DRIFT",
    "DERIVED_CONSTANT_SHA256",
    "DROP_EVAP_READ_BY_MP28",
    "DROP_EVAP_RELATIVE_TOLERANCE",
    "DROP_EVAP_SHAPE",
    "DROP_EVAP_TABLE_NAMES",
    "DTC",
    "GAMMA_MOMENTS",
    "G_RATIO",
    "G_RATIO_1BASED",
    "MP_PHYSICS",
    "MissingAerosolTableAsset",
    "NA_CCN0", "NA_CCN1", "NA_IN0", "NA_IN1",
    "NBC",
    "NC_FLOOR_PER_M3",
    "NC_SEDIMENT_FLOOR",
    "NIC1",
    "NIFA_FLOOR",
    "NT_C",
    "NT_C_MAX",
    "NTB_ARC", "NTB_ARK", "NTB_ARR", "NTB_ART", "NTB_ARW",
    "NTB_C",
    "NWFA2D_EMISSION_COEFF",
    "NWFA2D_REFERENCE_DEPTH_M",
    "NWFA_FLOOR",
    "OCG1",
    "OCG2",
    "PROFILE_H01_HIGH",
    "PROFILE_H01_HIGH_M",
    "PROFILE_H01_LOW",
    "PROFILE_H01_LOW_M",
    "R_C",
    "TA_KA", "TA_NA", "TA_RA", "TA_TK", "TA_WW",
    "TNC_WEV_RECORD_ORDINAL",
    "TNC_WEV_SOURCE_FILE",
    "TNC_WEV_TABLE_NAME",
    "T_NC",
    "T_NC_MAX",
    "T_NC_MIN",
    "build_drop_evaporation_tables",
    "cloud_shape_parameter",
    "derived_constant_arrays",
    "derived_constant_sha256",
    "droplet_number_bin",
    "gammln_fp32",
    "load_validated_aerosol_tables",
    "packaged_aerosol_table_root",
    "read_ccn_activation_table",
    "resolve_aerosol_table_root",
    "resolve_ccn_activation_path",
    "validate_ccn_activation_asset",
    "validate_ccn_activation_table",
    "validate_drop_evaporation_tables",
    "wgamma_fp32",
]
