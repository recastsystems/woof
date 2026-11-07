"""GRIMS shallow convection (WRF ``shcu_physics = 3``) on the GPU.

The CUDA column source was recovered from the October port. WRF 4.7.1
ships the same native module bytes as WRF 4.6.1. The real-input
4.7.1 replay is tools/shallow_wrf471_replay.py; its capture and
native diagnostic witness recipes are documented in
docs/shallow-wrf471.md.

* the five ``grimsinit`` tables come from the packaged
  ``woof/data/grims/grims_tables.npz`` (``tools/generate_grims_tables.py``,
  pinned by SHA-256 in :data:`GRIMS_TABLE_SHA256`), loaded once per process
  and refused if their bytes moved;
* ``levshc`` -- one level past the highest half level with eta > 0.6
  (``module_shcu_grims.F:153-159``) -- is a property of the vertical grid,
  so it is computed once from ``znu`` and sizes the kernel's dynamic
  shared memory (the normalised upper diagonal of the column's
  tridiagonal solve, ``levshc`` words per thread; the kernel holds no
  per-thread local frame);
* the inputs are exactly what ``module_shallowcu_driver.F:558-568`` hands
  ``grims``: current ``qv`` and temperature, the hydrostatic interface and
  layer pressures, the Exner function of the full pressure, the interface
  heights, and the PBL scheme's ``WSTAR_YSU``/``DELTA_YSU``/``PBLH``/
  ``KPBL`` with the cumulus scheme's ``RAINCV``.

Selection, pairing and dispatch are registry rows and table entries
(``shallow_cumulus`` in the physics registry, ``shcu_physics`` in
:data:`woof.core.physics_inventory.PHYSICS_SLOT_DISPATCH`); nothing here
decides whether GRIMS runs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from woof.core import constants as c

F32 = np.float32

#: The scheme's own selector value (WRF ``GRIMSSHCUSCHEME``).
GRIMS_SHCU_SCHEME = 3

#: ``tools/generate_grims_tables.py`` output, by SHA-256.  The bytes are
#: checked against WRF's own ``grimsinit`` tables by
#: ``tests/test_grims_tables.py``; this pin is what a run checks.
GRIMS_TABLE_RELATIVE = "grims/grims_tables.npz"
GRIMS_TABLE_SHA256 = "7761926078fef985d4060c0ca179b98e2d81d47eeb4e38ac9cbc55087d627e78"

#: Interpolation-constant order inside the npz's ``constants`` member and
#: the kernel's ``cst`` argument (``grims.cu`` GR_C* indices).
CONSTANT_NAMES = ("c1xma", "c2xma", "c1yma", "c2yma", "c1xpvs", "c2xpvs",
                  "c1xtdp", "c2xtdp", "c1xthe", "c2xthe", "c1ythe", "c2ythe")

TABLE_SHAPES = {
    "tbtdp": (5001,), "tbthe": (151, 241), "tbtma": (121, 151),
    "tbqma": (121, 151), "tbpvs": (7501,), "constants": (12,),
}

#: The WRF module_model_constants words the shallow-cumulus driver passes
#: (module_shallowcu_driver.F:563-565): G, XLV, R_D, R_V, RCP, P1000MB.
WRF_G = F32(c.G)
WRF_XLV = F32(c.XLV)
WRF_RD = F32(c.RD)
WRF_RV = F32(c.RV)
WRF_RCP = F32(F32(c.RD) / F32(F32(F32(7.0) * F32(c.RD)) / F32(2.0)))
WRF_P1000MB = F32(c.P0)

#: Dynamic shared memory a block may take without an opt-in attribute.
_SHARED_LIMIT_BYTES = 48 * 1024


def launch_block(levshc: int) -> int:
    """Threads per block: the widest of 128/64/32 whose shared upper
    diagonal (``levshc`` float32 words per thread) fits the default 48 KiB."""
    for block in (128, 64, 32):
        if levshc * block * 4 <= _SHARED_LIMIT_BYTES:
            return block
    raise ValueError(
        f"levshc={levshc}: GRIMS's shared upper diagonal needs "
        f"{levshc * 32 * 4} B per 32-thread block, over the "
        f"{_SHARED_LIMIT_BYTES} B a launch may take")


def table_path() -> Path:
    from woof import data_assets
    return data_assets.data_path(GRIMS_TABLE_RELATIVE)


class GrimsTableError(RuntimeError):
    """The packaged GRIMS tables are missing or are not the pinned bytes."""


@lru_cache(maxsize=1)
def host_tables() -> dict[str, np.ndarray]:
    """The pinned tables as float32 host arrays (read once per process).

    Refuses bytes that are not the pinned ones: every column's trigger,
    cloud top and diffusivity interpolate these tables, so a table that
    moved would change where and how hard shallow convection mixes with
    nothing else to show for it.
    """
    path = table_path()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise GrimsTableError(
            f"GRIMS tables {path} are unreadable ({exc}); shcu_physics=3 "
            "interpolates them in every column. Reinstall the package or "
            "rebuild them with tools/generate_grims_tables.py") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != GRIMS_TABLE_SHA256:
        raise GrimsTableError(
            f"GRIMS tables {path} hash to {digest}, not the pinned "
            f"{GRIMS_TABLE_SHA256}: the run would interpolate tables that "
            "were not checked against WRF's grimsinit")
    import io
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        tables = {name: np.ascontiguousarray(archive[name], dtype=F32)
                  for name in TABLE_SHAPES}
    for name, shape in TABLE_SHAPES.items():
        if tables[name].shape != shape:
            raise GrimsTableError(
                f"GRIMS table {name} has shape {tables[name].shape}, "
                f"expected {shape}")
    return tables


def levshc_for(znu) -> int:
    """``module_shcu_grims.F:153-159``: one past the last level with eta > 0.6."""
    if hasattr(znu, "get"):          # a device array (DomainState.znu)
        znu = znu.get()
    levshc = 0
    for k, value in enumerate(np.asarray(znu, dtype=F32), start=1):
        if value > F32(0.6):
            levshc = k + 1
    return levshc


@dataclass(frozen=True)
class GrimsResult:
    rthshten: object
    rqvshten: object
    kbot: object | None = None
    ktop: object | None = None


class GrimsShallowCumulus:
    """WRF's GRIMS shallow cumulus for one domain's vertical grid."""

    def __init__(self, znu, nz: int):
        levshc = levshc_for(znu)
        if levshc > nz:
            # module_shcu_grims.F:437 hands phys_moist_adiabat_pa
            # klev = levshc-1 levels to write into arrays sized kte-1.
            raise ValueError(
                f"shcu_physics=3 on a grid whose model top lies below "
                f"eta 0.6 (levshc={levshc} > nz={nz}): WRF's GRIMS writes "
                "its moist-adiabat scratch past the end of its arrays there "
                "and the result is undefined")
        if levshc < 3:
            raise ValueError(
                f"shcu_physics=3 needs at least two half levels with eta > "
                f"0.6 (levshc={levshc}): GRIMS mixes between kbot >= 2 and "
                "ktop >= kbot+2, which such a grid cannot hold, so the "
                "scheme could never act")
        self.levshc = levshc
        self.nz = int(nz)
        self.block = launch_block(levshc)
        self._device = None
        self._zero_raincv = None

    def zero_raincv(self, shape):
        """RAINCV for a domain with no cumulus scheme: WRF's zero state."""
        import cupy as cp
        if self._zero_raincv is None or self._zero_raincv.shape != shape:
            self._zero_raincv = cp.zeros(shape, dtype=F32)
        return self._zero_raincv

    def _device_tables(self):
        if self._device is None:
            import cupy as cp
            host = host_tables()
            self._device = {name: cp.asarray(value)
                            for name, value in host.items()}
        return self._device

    def kernel(self):
        from woof.core.kernels import get_kernel
        return get_kernel("grims", "grims_column")

    def __call__(self, *, qv, temperature, pressure, exner, p_interface,
                 z_interface, wstar, pblh, delta, raincv, kpbl, dt,
                 diagnostics: bool = False) -> GrimsResult:
        import cupy as cp
        nz, ny, nx = temperature.shape
        if nz != self.nz:
            raise ValueError(f"GRIMS was built for nz={self.nz}, got {nz}")
        shape3, shapew, shape2 = (nz, ny, nx), (nz + 1, ny, nx), (ny, nx)
        for name, arr, shape, dtype in (
                ("qv", qv, shape3, F32), ("temperature", temperature, shape3, F32),
                ("pressure", pressure, shape3, F32), ("exner", exner, shape3, F32),
                ("p_interface", p_interface, shapew, F32),
                ("z_interface", z_interface, shapew, F32),
                ("wstar", wstar, shape2, F32), ("pblh", pblh, shape2, F32),
                ("delta", delta, shape2, F32), ("raincv", raincv, shape2, F32),
                ("kpbl", kpbl, shape2, np.int32)):
            if (not isinstance(arr, cp.ndarray) or arr.shape != shape
                    or arr.dtype != dtype or not arr.flags.c_contiguous):
                raise ValueError(
                    f"GRIMS {name} must be a C-contiguous {np.dtype(dtype)} "
                    f"CuPy array of shape {shape}")
        tables = self._device_tables()
        rth = cp.empty(shape3, dtype=F32)
        rqv = cp.empty(shape3, dtype=F32)
        kbot = cp.empty(shape2, dtype=np.int32) if diagnostics else None
        ktop = cp.empty(shape2, dtype=np.int32) if diagnostics else None
        ncol = ny * nx
        null = np.uint64(0)
        block = self.block
        self.kernel()(
            ((ncol + block - 1) // block,), (block,),
            (qv, temperature, pressure, exner, p_interface, z_interface,
             wstar, pblh, delta, raincv, kpbl,
             tables["tbtdp"], tables["tbthe"], tables["tbtma"],
             tables["tbqma"], tables["tbpvs"], tables["constants"],
             rth, rqv,
             kbot if kbot is not None else null,
             ktop if ktop is not None else null,
             F32(dt), WRF_G, WRF_XLV, WRF_RD, WRF_RV, WRF_RCP, WRF_P1000MB,
             np.int32(self.levshc), np.int32(nz), np.int32(ncol)),
            shared_mem=self.levshc * block * 4)
        return GrimsResult(rth, rqv, kbot, ktop)
