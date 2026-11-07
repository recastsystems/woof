"""A real wrfout history file, written from the streamed case's host store.

WHY THIS EXISTS AND WHAT IT IS NOT
----------------------------------
``woof render --engine rust`` -- the production renderer, and the tier the
campaign material comes out of -- reads **wrfout NetCDF**.  The streamed case
run has no resident ``DomainState`` to hand ``woof.io.wrfout.state_frame``:
the domain lives in pinned host RAM and only a halo tile is ever on the card.
So the frame is assembled on the host, out of the store, and handed to the
PRODUCTION ``WrfoutWriter`` -- the same class, the same schema table, the same
Registry types, the same create/validate/publish sequence a resident run uses.
Nothing about the file format is reimplemented here; what is written here is
only the *mapping* from streamed carrier keys to WRF history names.

WHICH FIELDS, AND WHY THOSE -- MEASURED AGAINST THE RENDERER
------------------------------------------------------------
A COMPLETE ``_device_state_frame`` at this rung is 122 fields, 31 of them
three-dimensional.  At 1200x900x49 that is **6.6 GB per frame** and a
15-minute history over ten forecast hours is forty-one of them: 270 GB, more
disk than the box has.  So the frame is a subset -- and the subset was
CHOSEN BY MEASUREMENT, by writing variants at 96x80 and asking
``woof render --list-products --engine rust`` how many products each one
actually supports:

===============================================  =========  ============
3-D rows written                                 file size  renderable
===============================================  =========  ============
14 (+ QCLOUD/QRAIN/QICE/QSNOW/QGRAUP)              25.1 MB      **164**
 9 (U V W PH PHB T P PB QVAPOR)                    17.5 MB      **164**
 8 (the same, without W)                           16.0 MB          163
===============================================  =========  ============

The five hydrometeor mixing ratios buy **nothing**: the catalog's radar
products read the stored ``REFL_10CM``, not a recomputed one, so carrying
them would cost 0.9 GB a frame and 37 GB over the run for zero extra
products.  Dropping ``W`` costs one.  So nine 3-D rows it is -- 2.5 GB a
frame at 1200x900x49, 164 of the catalog's products renderable, including
the whole CAPE/CIN/SRH/shear/STP severe suite, the 200-850 mb isobaric
families, MSLP, 2 m T/Td, QPF and composite reflectivity.

Also written and NOT part of the streamed carrier set: ``UP_HELI_MAX``,
without which ``composite_reflectivity_uh`` and ``uh_2to5km`` -- the two
products a severe-weather forecaster actually reaches for -- are refused.
**ITS TIME SEMANTICS DIFFER FROM WRF's AND THE FILE SAYS SO.**  WRF's
``UP_HELI_MAX`` is a running maximum between history writes; what this run
can compute is the INSTANTANEOUS 2-5 km updraft helicity at the frame's own
valid time (``woof.core.uh_diag``'s host mirror of ``cal_helicity``,
evaluated on the whole domain at the sweep seam, because the device lane
refuses a tiled periodic geometry by design).  Every file carries
``GPUWM_UP_HELI_MAX_SEMANTICS`` saying exactly that, and any figure made
from it must repeat it.

Not carried: the physics schemes' own 3-D diagnostics (MYNN's
QKE/TSQ/QSQ/COV, the eddy viscosities, Noah-MP's 3-D soil).  No product in
the catalog reads them; a reader who wants them wants a restart.

WHAT IS DERIVED AND HOW
-----------------------
``T``, ``P``, ``PB`` and ``PHB`` are not carriers -- the store holds the
*perturbations* -- so they are formed on the host in the DEVICE's own
arithmetic order.  ``T = (thb + thp) - 300``, not ``thp + (thb - 300)``:
float addition is not associative and the reassociated form changes a
quarter of T's bytes on an un-stepped state (measured on branch
``tilestream-output``, §12.1).  The base state is captured ONCE from the
time-zero domain state and reused, because it is input and nothing in a
forecast writes it.

WHEN A FRAME MAY BE TAKEN
-------------------------
At a SWEEP BOUNDARY only.  ``run_tiled``'s ring mode keeps one store and
every tile reads and writes it inside the same sweep, so a frame taken
mid-sweep mixes generations.  ``run_case_hrrr`` calls this from ``on_sweep``,
which is the seam, and after the lateral-boundary frame has been imposed --
so what is written is exactly the state the next sweep will step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


__all__ = [
    "KIND_FULL",
    "KIND_LIGHT",
    "CaseFrameWriter",
    "FULL_ONLY_3D",
    "LIGHT_2D",
]

KIND_LIGHT = "light"
KIND_FULL = "full"


#: ``WRF name -> carrier key`` for the 2-D rows both frame kinds carry.
#: Every one of these is in the streamed inventory, so each is a zero-copy
#: view of a pinned page.
LIGHT_2D: dict[str, str] = {
    "MU": "state/mup",
    "T2": "fields/t2",
    "Q2": "fields/q2",
    "TH2": "fields/th2",
    "PSFC": "fields/psfc",
    "U10": "fields/u10",
    "V10": "fields/v10",
    "TSK": "fields/tsk",
    "PBLH": "fields/pblh",
    "HFX": "fields/hfx",
    "LH": "fields/lh",
    "QFX": "fields/qfx",
    "GRDFLX": "fields/grdflx",
    "SWDOWN": "fields/swdown",
    "GLW": "fields/glw",
    "UST": "fields/ust",
    "ZNT": "fields/znt",
    "SNOW": "fields/snow",
    "SNOWH": "fields/snowh",
    "SNOWC": "fields/snowc",
    "CANWAT": "fields/canwat",
    "SST": None,
    "RAINNC": "scratch/mp_rainnc",
    "SNOWNC": "scratch/mp_snownc",
    "GRAUPELNC": "scratch/mp_graupelnc",
    "SR": "scratch/mp_sr",
    "XLAND": "fields/xland",
    "VEGFRA": "fields/vegfra",
    "LAI": "fields/lai",
    "ALBEDO": "fields/albedo",
    "EMISS": "fields/emiss",
    "TMN": "fields/tmn",
    "XICE": "fields/xice",
}

#: 2-D rows WRF always writes and this configuration does not produce.  Exact
#: zeros, because that is what WRF writes when the producing scheme is off:
#: ``cu_physics = 0`` (no convective rain), no shallow cumulus, and Morrison
#: carries no hail category.  A precipitation recipe that reads
#: ``RAINC + RAINNC`` gets the right answer instead of a KeyError.
ZERO_2D: tuple[str, ...] = ("RAINC", "RAINSH", "HAILNC")

#: Soil rows, on the land-surface scheme's own 4-layer axis.
SOIL_3D: dict[str, str] = {
    "TSLB": "fields/tslb",
    "SMOIS": "fields/smois",
    "SH2O": "fields/sh2o",
}

#: The 3-D prognostic rows only the FULL frame carries, and their carrier
#: keys.  ``None`` means "derived here"; see :meth:`CaseFrameWriter._fields`.
FULL_ONLY_3D: dict[str, str | None] = {
    "U": "state/u",
    "V": "state/v",
    "W": "state/w",
    "PH": "state/php",
    "PHB": None,
    "T": None,
    "P": None,
    "PB": None,
    "QVAPOR": "state/qv",
}

#: Recorded so the choice above is reproducible rather than folklore: the
#: five Morrison mixing ratios were written, measured to add nothing, and
#: removed.  Put them back by extending :data:`FULL_ONLY_3D` with this.
DROPPED_HYDROMETEORS: dict[str, str] = {
    "QCLOUD": "state/qc",
    "QRAIN": "state/qr",
    "QICE": "state/qi",
    "QSNOW": "state/qs",
    "QGRAUP": "state/qg",
}

#: Simulated reflectivity.  BOTH frame kinds carry it: it is the field this
#: whole lane exists to show, and a light frame without it would be a
#: surface-only file with nothing to animate.
REFL_NAME = "REFL_10CM"
#: ONE spelling for the stash key, re-exported from this package's carrier
#: inventory, which itself re-exports :data:`woof.core.streaming
#: .REFL_STORE_KEY` -- the module that owns the whole ``refl_10cm`` handoff
#: for every streamed route.  Audit R-052 found the string typed out in
#: three modules and one lane's store built without the slot at all.
from tilestream.physics_inventory import REFL_KEY          # noqa: E402,F401


def _host(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.ascontiguousarray(np.asarray(value))


def _arr(value) -> np.ndarray:
    """A carrier as a host array, zero-copy when the store is already host.

    The streamed run's store IS pinned host memory, so this is a view and
    costs nothing.  The monolithic control's "store" is a dict of device
    arrays, and there the copy is unavoidable and explicit.
    """
    if hasattr(value, "get"):
        return np.ascontiguousarray(value.get())
    return np.asarray(value)


@dataclass
class CaseFrameWriter:
    """Assemble and publish wrfout frames from a streamed case's store.

    Constructed from the time-zero domain state, BEFORE it is released --
    that state is the only place the base state and the grid metadata exist,
    and re-deriving them later would mean a second full-domain ingest.
    """

    cfg: Any
    nx: int
    ny: int
    nz: int
    #: base state, host, at domain shape
    thb: np.ndarray
    pb3: np.ndarray
    phb3: np.ndarray
    #: setup 2-D/1-D rows
    statics: dict[str, np.ndarray]
    #: XLAT/XLONG/MAPFAC/... , the grid's own metadata frame
    metadata: dict[str, np.ndarray]
    global_attrs: dict[str, Any]
    title: str = "WOOF"
    _dst: dict[str, np.ndarray] = None          # type: ignore[assignment]

    # ------------------------------------------------------------- build
    @classmethod
    def capture(cls, state, cfg, grid, static, *, landuse_attrs, start_time,
                title: str = "WOOF") -> "CaseFrameWriter":
        from woof.io.wrfout import wrf_global_attrs
        from woof.runtime import _metadata_frame

        nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
        thb = _host(state.thb)
        pb = _host(state.pb)
        phb = _host(state.phb)
        pb3 = (pb if pb.ndim == 3
               else np.ascontiguousarray(
                   np.broadcast_to(pb[:, None, None], (nz, ny, nx))))
        phb3 = (phb if phb.ndim == 3
                else np.ascontiguousarray(
                    np.broadcast_to(phb[:, None, None], (nz + 1, ny, nx))))
        if thb.ndim != 3:
            thb = np.ascontiguousarray(
                np.broadcast_to(thb[:, None, None], (nz, ny, nx)))
        statics = {
            "MUB": _host(state.mub2d),
            "HGT": _host(state.ht),
            "ZNU": _host(state.znu),
            "ZNW": _host(state.znw),
            # WRF's P_TOP is a scalar row (dims ``Time`` only); a shape-(1,)
            # array would be routed onto no axis at all and raise.
            "P_TOP": np.asarray(float(np.reshape(_host(state.p_top), -1)[0]),
                                dtype=np.float32),
        }
        metadata = {k: np.ascontiguousarray(np.asarray(v, dtype=np.float32))
                    for k, v in _metadata_frame(grid, static).items()}
        # LU_INDEX is WRF's land-use category and is declared REAL in the
        # Registry, so it is written as one -- but it must not be rounded
        # here: geogrid's dominant category is already integral.
        attrs = wrf_global_attrs(
            grid, start_time, landuse_attrs=landuse_attrs,
            grid_id=1, parent_id=1, i_parent_start=1, j_parent_start=1,
            parent_grid_ratio=1, dt=float(cfg.dt),
            hybrid_opt=int(cfg.hybrid_opt), etac=float(cfg.etac), run=cfg)
        # The one place this file's UP_HELI_MAX differs from WRF's, written
        # INTO the file so it travels with the data rather than living in a
        # log somebody else will not read.
        attrs["GPUWM_UP_HELI_MAX_SEMANTICS"] = (
            "INSTANTANEOUS 2-5 km updraft helicity at this frame's valid "
            "time, NOT WRF's running maximum between history writes; "
            "computed on the host by woof.core.uh_diag's mirror of "
            "cal_helicity over the whole domain at the sweep seam")
        return cls(cfg=cfg, nx=nx, ny=ny, nz=nz, thb=thb, pb3=pb3, phb3=phb3,
                   statics=statics, metadata=metadata, global_attrs=attrs,
                   title=title, _dst={})

    @property
    def bytes_held(self) -> int:
        return (self.thb.nbytes + self.pb3.nbytes + self.phb3.nbytes
                + sum(a.nbytes for a in self.statics.values())
                + sum(a.nbytes for a in self.metadata.values())
                + sum(a.nbytes for a in (self._dst or {}).values()))

    # ------------------------------------------------------------- frame
    def _scratch(self, name, shape, dtype=np.float32) -> np.ndarray:
        got = self._dst.get(name)
        if got is None or got.shape != tuple(shape):
            got = np.zeros(tuple(shape), dtype=dtype)
            self._dst[name] = got
        return got

    def _fields(self, store: Mapping[str, np.ndarray], kind: str,
                extra: Mapping[str, np.ndarray] | None = None
                ) -> dict[str, np.ndarray]:
        """The frame dict, in a fixed order, as host arrays.

        Order is fixed because the dict doubles as the writer's schema and
        HDF5 lays its name heap out in creation order; a wandering order
        gives files that differ on disk while every variable compares equal.
        """
        out: dict[str, np.ndarray] = {}
        # -- coordinates and grid metadata first, as WRF does
        for name in ("XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V",
                     "XLONG_V", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V",
                     "F", "E", "SINALPHA", "COSALPHA", "LANDMASK",
                     "LU_INDEX"):
            value = self.metadata.get(name)
            if value is not None:
                out[name] = value
        out["HGT"] = self.statics["HGT"]
        out["MUB"] = self.statics["MUB"]
        out["ZNU"] = self.statics["ZNU"]
        out["ZNW"] = self.statics["ZNW"]
        out["P_TOP"] = self.statics["P_TOP"]

        if kind == KIND_FULL:
            for name, key in FULL_ONLY_3D.items():
                if key is not None:
                    out[name] = _arr(store[key])
                    continue
                if name == "PB":
                    out[name] = self.pb3
                elif name == "PHB":
                    out[name] = self.phb3
                elif name == "T":
                    dst = self._scratch("T", (self.nz, self.ny, self.nx))
                    if getattr(self.cfg, "use_theta_m", 0) and "state/qv" in store:
                        from tilestream.output import derive_dry_t
                        out[name] = derive_dry_t(self.thb, _arr(store["state/thp"]),
                                                _arr(store["state/qv"]), dst)
                        continue
                    # The device computes (thb + thp) - 300.  Keep that order.
                    np.add(self.thb, _arr(store["state/thp"]), out=dst)
                    np.subtract(dst, np.float32(300.0), out=dst)
                    out[name] = dst
                elif name == "P":
                    dst = self._scratch("P", (self.nz, self.ny, self.nx))
                    np.subtract(_arr(store["state/p"]), self.pb3,
                                out=dst)
                    out[name] = dst

        refl = store.get(REFL_KEY)
        if refl is not None:
            out[REFL_NAME] = _arr(refl)

        for name, key in LIGHT_2D.items():
            if key is None:
                continue
            value = store.get(key)
            if value is not None:
                out[name] = _arr(value)
        for name in ZERO_2D:
            out[name] = self._scratch(name, (self.ny, self.nx))
        for name, key in SOIL_3D.items():
            value = store.get(key)
            if value is not None:
                out[name] = _arr(value)
        for name, value in (extra or {}).items():
            out[name] = np.ascontiguousarray(np.asarray(value))
        return out

    # ------------------------------------------------------------- write
    def write(self, path, store: Mapping[str, np.ndarray], valid_time, *,
              kind: str = KIND_LIGHT, extra=None) -> Path:
        """One complete, validated, atomically published wrfout frame."""
        from woof.config import soil_layer_count
        from woof.io.wrfout import WrfoutWriter

        fields = self._fields(store, kind, extra)
        path = Path(path)
        writer = WrfoutWriter(
            path, nx=self.nx, ny=self.ny, nz=self.nz,
            dx=float(self.cfg.dx), dy=float(self.cfg.dy), title=self.title,
            global_attrs=dict(self.global_attrs), field_schema=fields,
            soil_layers=soil_layer_count(self.cfg))
        try:
            writer.write_frame(
                valid_time.strftime("%Y-%m-%d_%H:%M:%S"), fields)
        except BaseException:
            writer.abort()
            raise
        writer.close()
        return path
