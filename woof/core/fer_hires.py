"""Ferrier-Aligo microphysics (WRF ``mp_physics=5``, package fer_mp_hires).

The CUDA translation unit ``kernels/fer_hires.cu`` carries WRF v4.6.1
``phys/module_mp_fer_hires.F`` statement by statement: FER_HIRES, its
EGCP01DRV/EGCP01COLUMN column physics with CONDENSE and DEPOSIT, and the
fer_hires_init/GPVS tables.  This module owns the three things around it:

* the scheme's lookup table ``ETAMPNEW_DATA.expanded_rain``, packaged
  byte-identical to WRF's ``run/`` copy under ``woof/data/ferrier/tables``
  and SHA-256-validated before it is parsed (the P3/Thompson table path;
  ``WOOF_FERRIER_TABLE_ROOT`` names another directory holding the same
  bytes);
* the per-domain table buffer the init kernels fill once from the domain's
  time step, as WRF's microphysics_init calls fer_hires_init once per domain
  (module_physics_init.F:4499-4505) -- CBFR is taken from that step and
  never recomputed (module_mp_fer_hires.F:3013), while CIACW, CIACR, CRACW,
  BRAUT and the Miller-Young growth table follow every call's DT
  (:323-343);
* the forecast adapter :func:`apply`, which builds WRF's
  moist_physics_prep_em inputs, runs the column kernel and finishes the
  retained heating.

State.  WRF registers mp=5 as moist qv,qc,qr,qi, scalar qt and state
f_ice_phy,f_rain_phy,f_rimef_phy (Registry.EM_COMMON:3019).  ``qi`` is the
total ice (cloud ice and precipitating ice together).  ``qt`` is rebuilt
from qc+qr+qi on every entry (:388), so it is the scheme's output only and
woof keeps it as a carried state field rather than transporting it.  The
three ``f_*`` fields are WRF state (never advected): F_ICE_PHY and
F_RAIN_PHY are overwritten on entry but read by radiation after the call,
and F_RIMEF_PHY carries the rime factor of every ice-bearing cell from one
step to the next (:389-395).  fer_hires_init sets them to 0, 0 and 1 on a
non-restart start (:2912-2923), which is what the allocator does here.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

#: The table WRF reads at module_mp_fer_hires.F:2969.
TABLE_NAME = "ETAMPNEW_DATA.expanded_rain"
#: SHA-256 of WRF v4.6.1 ``run/ETAMPNEW_DATA.expanded_rain`` (d66e442f).
TABLE_SHA256 = (
    "28e60e6c83eb55889c4ab65eaa33e9fc3b557e5ab45c3dde7ee7153c2ef95a13")
TABLE_ROOT_ENV = "WOOF_FERRIER_TABLE_ROOT"

#: Bounds the module declares (:143-158): MDImin:MDImax and MDRmin:MDRmax are
#: both 50:1000, VEL_RF is (2:9, 0:Nrime) with Nrime = 40, and the
#: saturation tables have NX = 7501 entries.
MDMIN, MDMAX = 50, 1000
NTAB = MDMAX - MDMIN + 1
NRIME = 40
NX = 7501
#: The twelve records fer_hires_init READs, in order (:2972-2983).  The
#: file carries a thirteenth, a 35-word Miller-Young table whose READ is
#: commented out at :2984 (it holds 180 s values only); it is validated and
#: ignored, as WRF ignores it.
RECORDS = (("VENTR1", NTAB), ("VENTR2", NTAB), ("ACCRR", NTAB),
           ("MASSR", NTAB), ("VRAIN", NTAB), ("RRATE", NTAB),
           ("VENTI1", NTAB), ("VENTI2", NTAB), ("ACCRI", NTAB),
           ("MASSI", NTAB), ("VSNOWI", NTAB), ("VEL_RF", 8 * (NRIME + 1)))
UNREAD_TRAILING_RECORD_WORDS = 35

# Offsets into the device buffer; kernels/fer_hires.cu spells the same
# numbers as FH_OFF_* and FH_S_* (tests/test_fer_hires_port.py holds the two
# in step).
OFF_TBPVS = 11 * NTAB + 8 * (NRIME + 1)
OFF_TBPVS0 = OFF_TBPVS + NX
OFF_SCALARS = OFF_TBPVS0 + NX
SCALARS = ("C1XPVS", "C2XPVS", "C1XPVS0", "C2XPVS0", "PI_E", "ABFR", "CBFR",
           "RQR_DRmin", "RQR_DRmax", "CN0r0", "ESW0", "RFmax", "RFmx1",
           "BETA6", "ARAUT", "QAUT0", "ARcw", "RH_NgC", "RQhail", "BVhail",
           "AVhail", "RH_NgT", "DTPH", "Crain")
TABLE_WORDS = OFF_SCALARS + len(SCALARS)

#: The level window, spelled once in woof/physics_vertical_contract.py.
from woof.physics_vertical_contract import (  # noqa: E402
    FERRIER_VERTICAL_LEVEL_BOUNDS as VERTICAL_LEVEL_BOUNDS)

_TPB = 64


class FerrierTableError(ValueError):
    """The Ferrier table is missing, modified, or not the shape WRF reads."""


def packaged_table_root() -> Path:
    """The in-package canonical table directory."""
    return Path(__file__).resolve().parent.parent / "data" / "ferrier" / "tables"


def table_root() -> Path:
    """``WOOF_FERRIER_TABLE_ROOT`` when set, else the packaged directory."""
    override = os.environ.get(TABLE_ROOT_ENV)
    return Path(override) if override else packaged_table_root()


def parse_table(blob: bytes) -> dict[str, np.ndarray]:
    """WRF's big-endian Fortran unformatted records -> float32 arrays.

    fer_hires_init opens the file unformatted under WRF's big-endian
    runtime conversion and READs whole records (:2969-2983).  Each record is
    a 4-byte big-endian length, that many bytes, and the length again; a
    record of another length is the wrong table, refused rather than
    truncated or padded.
    """
    out: dict[str, np.ndarray] = {}
    pos = 0

    def record(name: str, words: int) -> np.ndarray:
        nonlocal pos
        if pos + 4 > len(blob):
            raise FerrierTableError(f"{TABLE_NAME} ends before record {name}")
        head = int.from_bytes(blob[pos:pos + 4], "big")
        if head != 4 * words:
            raise FerrierTableError(
                f"{TABLE_NAME} record {name} holds {head} bytes; "
                f"fer_hires_init reads {4 * words} (module_mp_fer_hires.F)")
        body = blob[pos + 4:pos + 4 + head]
        tail = int.from_bytes(blob[pos + 4 + head:pos + 8 + head], "big")
        if len(body) != head or tail != head:
            raise FerrierTableError(
                f"{TABLE_NAME} record {name} is truncated or its markers "
                "disagree")
        pos += 8 + head
        return np.frombuffer(body, dtype=">f4").astype(np.float32)

    for name, words in RECORDS:
        out[name] = record(name, words)
    record("MY_600 (unread by WRF)", UNREAD_TRAILING_RECORD_WORDS)
    if pos != len(blob):
        raise FerrierTableError(
            f"{TABLE_NAME} has {len(blob) - pos} bytes after its last record")
    return out


def load_table(root: Path | None = None) -> dict[str, np.ndarray]:
    """Read, SHA-256-validate and parse the table under ``root``."""
    root = table_root() if root is None else Path(root)
    path = root / TABLE_NAME
    from woof.tables import table_path
    from woof.tables import CorruptTableAssets, MissingTableAssets
    try:
        path = table_path("ferrier", path)
    except (CorruptTableAssets, MissingTableAssets) as error:
        raise FerrierTableError(str(error)) from error
    try:
        blob = path.read_bytes()
    except FileNotFoundError:
        raise FerrierTableError(
            f"mp_physics=5 needs {path}: WRF's fer_hires_init reads this "
            "table at model start (module_mp_fer_hires.F:2969) and every "
            "rain and ice rate is a lookup into it.  Reinstall woof, or "
            f"point {TABLE_ROOT_ENV} at a directory holding WRF v4.6.1's "
            f"run/{TABLE_NAME}.") from None
    digest = hashlib.sha256(blob).hexdigest()
    if digest != TABLE_SHA256:
        raise FerrierTableError(
            f"{path} has SHA-256 {digest}, not WRF v4.6.1's {TABLE_SHA256}. "
            "Refusing to integrate Ferrier-Aligo against a modified table.")
    return parse_table(blob)


def host_table_words(tables: dict[str, np.ndarray]) -> np.ndarray:
    """The integral tables laid out as the kernel's buffer prefix."""
    words = np.zeros(TABLE_WORDS, dtype=np.float32)
    pos = 0
    for name, n in RECORDS:
        words[pos:pos + n] = tables[name]
        pos += n
    assert pos == OFF_TBPVS
    return words


_DEVICE_TABLES: dict[tuple, object] = {}


def device_tables(dtph: float):
    """The kernel's table buffer for a domain whose init step is ``dtph``.

    fer_hires_init (:2793-3150): GPVS and the SAVE scalars, CBFR from DTPH.
    Cached per device and per binary32 DTPH, the two things that make two
    domains' buffers differ.
    """
    import cupy as cp
    from woof.core.kernels import get_kernel

    key = (cp.cuda.runtime.getDevice(),
           int(np.float32(dtph).view(np.uint32)), str(table_root()))
    tab = _DEVICE_TABLES.get(key)
    if tab is None:
        if not np.isfinite(dtph) or dtph <= 0.0:
            raise ValueError(f"Ferrier init step must be positive, got {dtph}")
        tab = cp.asarray(host_table_words(load_table()))
        get_kernel("fer_hires", "fer_hires_init_tables")(
            ((NX + 255) // 256,), (256,), (tab,))
        get_kernel("fer_hires", "fer_hires_init_scalars")(
            (1,), (1,), (tab, np.float32(dtph)))
        _DEVICE_TABLES[key] = tab
    return tab


def launch_fer_hires(th, qv, qc, qr, qi, qt, f_ice, f_rain, f_rimef,
                     p, pii, rho, dz8w, rainnc, rainncv, sr, tab,
                     dt: float, gid: int, refl=None, *, analysis_only=False) -> None:
    """One FER_HIRES call over contiguous float32 ``(nz, ny, nx)`` fields.

    th/qv/qc/qr/qi/f_ice/f_rain/f_rimef are updated and qt written in
    place; p (Pa), pii, rho (kg m-3) and dz8w (m) are read; rainnc is
    accumulated and rainncv/sr written on ``(ny, nx)``.  With ``refl`` the
    scheme's own reflectivity column (EGCP01COLUMN DBZ_col, dBZ) is
    written there too.
    """
    from woof.core.kernels import get_kernel
    from woof.core.state import DTYPE

    shape = th.shape
    if len(shape) != 3:
        raise ValueError(f"Ferrier fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz < VERTICAL_LEVEL_BOUNDS[0]:
        raise ValueError(f"Ferrier needs at least {VERTICAL_LEVEL_BOUNDS[0]} "
                         f"levels, got {nz}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    for name, a in (("th", th), ("qv", qv), ("qc", qc), ("qr", qr),
                    ("qi", qi), ("qt", qt), ("f_ice", f_ice),
                    ("f_rain", f_rain), ("f_rimef", f_rimef), ("p", p),
                    ("pii", pii), ("rho", rho), ("dz8w", dz8w)):
        if a.shape != shape or a.dtype != DTYPE or not a.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous float32 {shape}, "
                             f"got {a.dtype} {a.shape}")
    for name, a in (("rainnc", rainnc), ("rainncv", rainncv), ("sr", sr)):
        if a.shape != (ny, nx) or a.dtype != DTYPE or not a.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous float32 "
                             f"{(ny, nx)}, got {a.dtype} {a.shape}")
    if tab.shape != (TABLE_WORDS,) or tab.dtype != DTYPE:
        raise ValueError("tab is not the buffer device_tables() builds")
    if refl is not None and (refl.shape != shape or refl.dtype != DTYPE
                             or not refl.flags.c_contiguous):
        raise ValueError(f"refl must be C-contiguous float32 {shape}")
    blocks = (ny * nx + _TPB - 1) // _TPB
    get_kernel("fer_hires", "fer_hires_column")(
        (blocks,), (_TPB,),
        (th, qv, qc, qr, qi, qt, f_ice, f_rain, f_rimef, p, pii, rho, dz8w,
         rainnc, rainncv, sr, th if refl is None else refl, tab, DTYPE(dt),
         np.int32(gid), np.int32(0 if refl is None else 1), np.int32(nz),
         np.int32(ny), np.int32(nx), np.int32(analysis_only)))


#: The adapter's state fields beyond qv/qc/qr: the moist and scalar species
#: and the three WRF state arrays (Registry.EM_COMMON:3019).
STATE_FIELDS = ("qi", "qt", "f_ice", "f_rain", "f_rimef")


def apply(state, cfg, dt: float, *, refl_10cm_due: bool = False):
    """WRF's microphysics_driver FER_MP_HIRES arm for one domain step."""
    import cupy as cp

    from woof.core import constants as c
    from woof.core.microphysics import (MicrophysicsDiagnostics,
                                         moist_physics_finish,
                                         save_pre_mp_theta)
    from woof.core.state import DTYPE

    missing = [n for n in STATE_FIELDS if getattr(state, n, None) is None]
    if missing:
        raise ValueError("mp_physics=5 state lacks Ferrier fields: "
                         + ", ".join(missing))
    nz, ny, nx = state.p.shape
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    th = state.scratch((nz, ny, nx), "mp_th")
    rho = state.scratch((nz, ny, nx), "mp_rho")
    pii = state.scratch((nz, ny, nx), "mp_pii")
    dz8w = state.scratch((nz, ny, nx), "mp_dz8w")
    z8w = state.scratch((nz + 1, ny, nx), "mp_z8w")
    th[...] = thb + state.thp
    rho[...] = 1.0 / state.alt                   # dry density 1/(al+alb)
    pii[...] = cp.power(state.p / DTYPE(c.P0), DTYPE(c.RCP))
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    dz8w[...] = z8w[1:] - z8w[:nz]

    surface = (ny, nx)
    rainnc = state.scratch(surface, "mp_rainnc")
    rainncv = state.scratch(surface, "mp_rainncv")
    sr = state.scratch(surface, "mp_sr")

    # fer_hires_init runs once per domain with the domain's model step
    # (WRF microphysics_init passes grid%dt); the configured step is that.
    tab = device_tables(float(cfg.dt))
    refl = state.scratch((nz, ny, nx), "refl_10cm") if refl_10cm_due else None
    save_pre_mp_theta(state)                     # moist_physics_prep_em
    launch_fer_hires(th, state.qv, state.qc, state.qr, state.qi, state.qt,
                     state.f_ice, state.f_rain, state.f_rimef, state.p, pii,
                     rho, dz8w, rainnc, rainncv, sr, tab, dt,
                     int(getattr(cfg, "grid_id", 1)), refl=refl)
    if refl is not None:
        # The scheme's own DBZ_col (kernels/fer_hires.cu, REFLECTIVITY),
        # parked for the history writer the way P3 and Milbrandt-Yau park
        # theirs.
        from woof.core.refl import stash_refl_10cm
        stash_refl_10cm(state, refl)
    moist_physics_finish(state, cfg, th, dt)
    return MicrophysicsDiagnostics(rainnc=rainnc, rainncv=rainncv, sr=sr)


__all__ = ["apply", "device_tables", "launch_fer_hires", "load_table",
           "parse_table", "STATE_FIELDS", "TABLE_NAME", "TABLE_SHA256"]


def reflectivity(state, cfg):
    """Native current-state radar moments, without sources or sedimentation."""
    import cupy as cp
    from woof.core import constants as c
    from woof.core.state import DTYPE
    shape = tuple(state.p.shape)
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    th = cp.ascontiguousarray(thb + state.thp)
    pii = cp.power(state.p / DTYPE(c.P0), DTYPE(c.RCP))
    rho = 1 / state.alt
    # Thickness cancels in the native PSD setup when there is no fallout.
    dz = cp.ones(shape, dtype=DTYPE)
    surface = cp.zeros(shape[1:], dtype=DTYPE)
    unused = cp.empty(shape, dtype=DTYPE)
    refl = state.scratch(shape, "refl_10cm")
    launch_fer_hires(th, state.qv, state.qc, state.qr, state.qi,
                     unused, unused, unused, state.f_rimef, state.p, pii,
                     rho, dz, surface, surface, surface,
                     device_tables(float(cfg.dt)), float(cfg.dt),
                     int(getattr(cfg, "grid_id", 1)), refl=refl, analysis_only=True)
    return refl
