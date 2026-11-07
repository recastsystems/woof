"""GFS SAS deep convection with the ARW shallow branch, ``cu_physics = 94``.

WRF v4.6.1 ``module_cu_sas.F`` (``CU_SAS`` -> ``SASCNVN`` -> ``OLD_ARW_SHALCV``),
ported to :mod:`woof.core.kernels` ``gfs_sas.cu`` and graded word for word
against the built WRF tree (``tools/sas_wrf461_oracle``,
:mod:`woof.verify.sas_oracle`).  This module only launches the kernels:

* :func:`tables` builds module_gfs_funcphys's five lookup tables on the device
  once per device (WOOF's own binary64 exp/pow);
* :class:`GfsSas` is the cumulus callable the physics driver resolves from
  the registry row for ``cu_physics = 94`` (``consumers.stock_callable_class``);
  it tiles the grid's columns through one workspace and returns the WRF
  outputs as a :class:`woof.core.physics.CumulusResult`.

Cadence.  SAS runs every ``STEPCU`` model steps (``cudt``) like WRF: the
driver's due predicate decides the call, ``DELT = DT*STEPCU`` inside, and
the result carries PRATEC with a zero hold so the driver keeps the rates
and accumulates ``RAINC += PRATEC*DT`` every step (WRF ``advance_ppt``,
module_physics_addtendc.F:2141) until the next call replaces them.
"""
from __future__ import annotations

import numpy as np

#: WRF's own number for SASSCHEME (module_cumulus_driver.F CASE (SASSCHEME)).
SAS_CU_PHYSICS = 94

#: ymin/ymax of the gthe and gtma tables (module_gfs_funcphys.F :1713-1714,
#: :1953-1954): 0.04**rocp, 1.10**rocp, 0.01**rocp, 1.10**rocp.  gfortran
#: folds these constant expressions; the words are measured against the
#: oracle's c1y/c2y coefficients (the correctly rounded value and the
#: run-time pow agree on all four).
TABLE_BOUNDS = (float.fromhex("0x1.982fc3357ea6ep-2"),
                float.fromhex("0x1.07114cea74d14p+0"),
                float.fromhex("0x1.12ae5ac77057bp-2"),
                float.fromhex("0x1.07114cea74d14p+0"))

# Table layout, the same as sas.cu's TB_* and tables.bin.
NXPVS, NXTDP = 7501, 5001
NXTHE, NYTHE = 241, 151
NXMA, NYMA = 151, 121
NXTLCL, NYTLCL = 151, 61
TB_PVS = 0
TB_TDP = TB_PVS + 2 + NXPVS
TB_THE = TB_TDP + 2 + NXTDP
TB_MA = TB_THE + 4 + NXTHE * NYTHE
TB_TLCL = TB_MA + 4 + 2 * NXMA * NYMA
TB_TOTAL = TB_TLCL + 4 + NXTLCL * NYTLCL
TABLE_SECTIONS = {
    "pvs": (TB_PVS, 2 + NXPVS),
    "tdp": (TB_TDP, 2 + NXTDP),
    "the": (TB_THE, 4 + NXTHE * NYTHE),
    "ma": (TB_MA, 4 + 2 * NXMA * NYMA),
    "tlcl": (TB_TLCL, 4 + NXTLCL * NYTLCL),
}

#: Workspace slots per column (sas.cu enum S_COUNT).
SAS_SLOTS = 54
#: Columns per launch at most; the workspace is SAS_SLOTS*(nz+1)*8 bytes a
#: column (about 22 KiB at 50 levels, so 360 MiB at the cap).
SAS_TILE_COLUMNS = 16384
_THREADS = 128

_TABLES: dict = {}


def module():
    from woof.core.kernels import load_module
    return load_module("gfs_sas")


def tables():
    """The funcphys tables on the current device, built once."""
    import cupy as cp
    dev = cp.cuda.Device().id
    tb = _TABLES.get(dev)
    if tb is None:
        tb = cp.zeros(TB_TOTAL, dtype=cp.float64)
        nthreads = NXPVS + 1 + NXTHE * NYTHE + NYMA + NXTLCL * NYTLCL
        kern = module().get_function("sas_tables")
        kern(((nthreads + _THREADS - 1) // _THREADS,), (_THREADS,),
             (tb, *(np.float64(b) for b in TABLE_BOUNDS)))
        _TABLES[dev] = tb
    return tb


def gfs_workspace_bytes(nz: int, columns: int) -> int:
    """Device bytes of the column workspace for ``columns`` at ``nz`` levels."""
    return int(SAS_SLOTS) * (int(nz) + 1) * int(columns) * 8


def tile_columns(ncol: int) -> int:
    return max(1, min(int(ncol), SAS_TILE_COLUMNS))


class SasWorkspace:
    """One binary64 workspace reused by every tile of every call."""

    def __init__(self, nz: int, columns: int):
        import cupy as cp
        self.kw = int(nz) + 1
        self.columns = int(columns)
        self.array = cp.empty(SAS_SLOTS * self.kw * self.columns, dtype=cp.float64)

    def release(self) -> int:
        freed = int(self.array.nbytes)
        self.array = None
        return freed


def run_columns(*, t3d, qv, qc, qi, u, v, w, pcps, pi, rho, dz8w, p8w, xland,
                dt, stepcu, pgcon, has_qc=True, has_qi=True, shalconv=1,
                kmcol=None, dtcol=None, stepcucol=None, pgconcol=None,
                workspace=None, diagnostics=False, mod=None):
    """CU_SAS over every column of level-major (nz, ncol) float32 inputs.

    ``w`` and ``p8w`` have nz+1 rows.  Per-column ``kmcol``/``dtcol``/
    ``stepcucol``/``pgconcol`` serve the oracle's mixed case set; a forecast
    passes scalars.  Returns a dict of float32 outputs (and int32/float64
    diagnostics when asked).  ``mod`` replaces the compiled module (the
    verifier's planted-mutation check compiles a mutated copy).
    """
    import cupy as cp
    nz, ncol = int(t3d.shape[0]), int(t3d.shape[1])
    f32 = lambda a: cp.ascontiguousarray(cp.asarray(a, dtype=cp.float32))
    ins = [f32(a) for a in (t3d, qv, qc, qi, u, v, w, pcps, pi, rho, dz8w, p8w)]
    xland = f32(xland)
    out = {}
    for name, enabled in (("rthcuten", True), ("rqvcuten", True),
                          ("rqccuten", has_qc), ("rqicuten", has_qi)):
        allocate = cp.empty if kmcol is None and enabled else cp.zeros
        out[name] = allocate((nz, ncol), dtype=cp.float32)
    for name in ("raincv", "pratec", "hbot", "htop"):
        out[name] = cp.empty(ncol, dtype=cp.float32)
    diag = None
    if diagnostics:
        diag = {"kbot": cp.zeros(ncol, dtype=cp.int32),
                "ktop": cp.zeros(ncol, dtype=cp.int32),
                "kcnv": cp.zeros(ncol, dtype=cp.int32),
                "cldwrk": cp.zeros(ncol, dtype=cp.float64)}
    width = tile_columns(ncol)
    if workspace is None or workspace.kw != nz + 1 or workspace.columns < width:
        workspace = SasWorkspace(nz, width)
    opt = lambda a, dtype: (None if a is None
                            else cp.ascontiguousarray(cp.asarray(a, dtype=dtype)))
    kmcol, dtcol = opt(kmcol, cp.int32), opt(dtcol, cp.float32)
    stepcucol, pgconcol = opt(stepcucol, cp.int32), opt(pgconcol, cp.float32)
    compiled = mod or module()
    kernels = [compiled.get_function(name) for name in
               ("sas_prepare_fields", "sas_prepare_columns", "sas_deep_prepare",
                "sas_column", "sas_finish")]
    tb = tables()
    for lo in range(0, ncol, width):
        n = min(width, ncol - lo)
        args = (tb, workspace.array, np.int32(workspace.kw), np.int32(n),
              np.int32(lo), np.int64(ncol), np.int32(nz), kmcol,
              *ins, xland, dtcol, stepcucol, pgconcol,
              np.float32(dt), np.int32(stepcu), np.float32(pgcon),
              np.int32(shalconv), np.int32(bool(has_qc)), np.int32(bool(has_qi)),
              out["rthcuten"], out["rqvcuten"], out["rqccuten"], out["rqicuten"],
              out["raincv"], out["pratec"], out["hbot"], out["htop"],
              None if diag is None else diag["kbot"],
              None if diag is None else diag["ktop"],
              None if diag is None else diag["kcnv"],
              None if diag is None else diag["cldwrk"])
        columns_grid = ((n + _THREADS - 1) // _THREADS,)
        levels_grid = (*columns_grid, nz)
        for index, kernel in enumerate(kernels):
            kernel(levels_grid if index in (0, 2, 4) else columns_grid,
                   (_THREADS,), args)
    if diag is not None:
        out.update(diag)
    return out, workspace


class GfsSas:
    """The ``cu_physics = 94`` cumulus callable (see the module docstring)."""

    def __init__(self) -> None:
        self._driver = None
        self._sas_workspace = None

    def bind_driver(self, driver) -> None:
        """Driver hook: SAS reads the driver's STEPCU (its due cadence)."""
        self._driver = driver

    def release(self) -> int:
        freed = 0
        if self._sas_workspace is not None:
            freed = self._sas_workspace.release()
        self._sas_workspace = None
        self._driver = None
        return int(freed)

    def _stepcu(self, cfg) -> int:
        stepcu = getattr(self._driver, "stepcu", None)
        if stepcu is None:
            from woof.core.physics import _physics_interval_steps
            stepcu = _physics_interval_steps(cfg.cudt_minutes, cfg.dt)
        return max(1, int(stepcu))

    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp
        from woof.core.physics_inventory import PBL_RQI_MICROPHYSICS
        from woof.core.physics import CumulusResult, _model_clock_dt

        nz, ny, nx = state.p.shape
        ncol = ny * nx

        def lev(a, rows=nz):
            return cp.ascontiguousarray(a, dtype=cp.float32).reshape(rows, ncol)

        zeros = None
        qc = atmosphere.get("qc")
        qi = atmosphere.get("qi")
        has_qc = qc is not None
        has_qi = qi is not None and cfg.mp_physics in PBL_RQI_MICROPHYSICS
        if not has_qc or not has_qi:
            zeros = cp.zeros((nz, ncol), dtype=cp.float32)
        dt = float(_model_clock_dt(cfg))
        stepcu = self._stepcu(cfg)
        out, self._sas_workspace = run_columns(
            t3d=lev(atmosphere["temperature"]), qv=lev(atmosphere["qv"]),
            qc=lev(qc) if has_qc else zeros, qi=lev(qi) if has_qi else zeros,
            u=lev(atmosphere["u"]), v=lev(atmosphere["v"]),
            w=lev(state.w, nz + 1), pcps=lev(atmosphere["pressure"]),
            pi=lev(atmosphere["exner"]), rho=lev(atmosphere["rho"]),
            dz8w=lev(atmosphere["dz"]), p8w=lev(atmosphere["p_interface"], nz + 1),
            xland=cp.ascontiguousarray(fields["xland"], dtype=cp.float32).reshape(ncol),
            dt=dt, stepcu=stepcu, pgcon=float(cfg.sas_pgcon),
            has_qc=has_qc, has_qi=has_qi, shalconv=int(cfg.shalconv),
            workspace=self._sas_workspace)

        def grid(a):
            return a.reshape(nz, ny, nx)

        pratec = out["pratec"].reshape(ny, nx)
        cp.copyto(fields["sas_hbot"], out["hbot"].reshape(ny, nx))
        cp.copyto(fields["sas_htop"], out["htop"].reshape(ny, nx))
        return CumulusResult(
            rthcuten=grid(out["rthcuten"]), rqvcuten=grid(out["rqvcuten"]),
            rqccuten=grid(out["rqccuten"]) if has_qc else None,
            rqicuten=grid(out["rqicuten"]) if has_qi else None,
            # WRF holds the rates for STEPCU steps and accumulates
            # RAINC += PRATEC*DT every step (advance_ppt); a zero hold makes
            # every column eligible at each due call, which is SAS's own
            # cadence (every column recomputed every STEPCU steps).
            pratec=pratec, nca_seconds=cp.zeros((ny, nx), dtype=cp.float32),
            rainc=out["raincv"].reshape(ny, nx))


# KIAPS SAS retains its independently graded native column interface.
import hashlib
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from woof.core.state import DTYPE

#: WRF v4.6.1 share/module_model_constants.F, the values the cumulus driver
#: hands CU_KSAS, formed exactly as gfortran folds the PARAMETERs: every
#: operation a float32 IEEE operation on float32 operands.
_F = np.float32
_R_D = _F(287.0)
_R_V = _F(461.6)
WRF_MODEL_CONSTANTS = MappingProxyType({
    "cp": _F(_F(7.0) * _R_D) / _F(2.0),
    "cliq": _F(4190.0),
    "cpv": _F(4.0) * _R_V,
    "g": _F(9.81),
    "xlv": _F(2.5e6),
    "r_d": _R_D,
    "r_v": _R_V,
    "ep_1": _F(_R_V / _R_D) - _F(1.0),
    "ep_2": _R_D / _R_V,
    "cice": _F(2106.0),
    "xls": _F(2.85e6),
    "psat": _F(610.78),
})
#: The order the kernel takes them in.
CONSTANT_ORDER = ("cp", "cliq", "cpv", "g", "xlv", "r_d", "r_v", "ep_1",
                  "ep_2", "cice", "xls", "psat")


def constants_sha256(constants=WRF_MODEL_CONSTANTS) -> str:
    """Identity of the constant table the kernel was launched with."""
    raw = b"".join(np.float32(constants[name]).tobytes()
                   for name in CONSTANT_ORDER)
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class SasVariant:
    """One SAS-family scheme: everything the seam needs that differs."""

    cu_physics: int
    label: str
    kernel: str
    workspace_slots: int
    wrf_source: str
    wrf_driver_case: str


#: The family.  KSAS's 49 workspace slots are sas.cu's SAS_KSAS_SLOTS.
SAS_VARIANTS = MappingProxyType({
    14: SasVariant(
        cu_physics=14, label="KIAPS SAS", kernel="sas_ksas_column",
        workspace_slots=49,
        wrf_source="phys/module_cu_ksas.F (WRF v4.6.1)",
        wrf_driver_case="phys/module_cumulus_driver.F:1444 CASE (KSASSCHEME)"),
})

#: Columns per launch.  The workspace is SLOTS x (nz + 1) x tile floats,
#: 49 x 51 x 32768 x 4 B = 327.5 MiB at 50 levels; a domain larger than
#: the tile is walked in tiles (the kernel is per column, so tiling cannot
#: change an answer).
KSAS_TILE_COLUMNS = 32768
SAS_BLOCK = 128


def ksas_ncloud(mp_physics: int) -> int:
    """module_cu_ksas.F:145-151: the detrained-condensate category count."""
    if mp_physics == 0:
        return 0
    if mp_physics in (1, 3):
        return 1
    return 2


def ksas_workspace_bytes(variant: SasVariant, nz: int, ncol: int) -> int:
    tile = min(int(ncol), KSAS_TILE_COLUMNS)
    return variant.workspace_slots * (nz + 1) * tile * 4


def _module():
    from woof.core.kernels import load_module
    return load_module("sas")


def launch_column(variant: SasVariant, *, inputs, znu, dt, dx, stepcu,
                  mp_physics, f_qc, f_qi, constants=WRF_MODEL_CONSTANTS,
                  outputs=None, workspace=None, function=None):
    """Run one variant over every column of ``inputs``.

    ``inputs`` maps the driver names to device float32 arrays: level fields
    ``t3d qv3d qc3d qi3d p3d pi3d rho3d dz8w u3d v3d`` shaped ``(nz, ncol)``,
    ``w`` shaped ``(nz + 1, ncol)`` and ``xland hpbl hpbl_hold`` shaped
    ``(ncol,)``, all C-contiguous.  Returns the output dict (the six
    tendencies ``(nz, ncol)`` and ``raincv pratec hbot htop`` ``(ncol,)``).
    ``function`` substitutes another compiled entry point (the planted-bit
    test uses it); ``workspace`` reuses a caller's buffer.
    """
    import cupy as cp

    nz, ncol = inputs["t3d"].shape
    for name in ("t3d", "qv3d", "qc3d", "qi3d", "p3d", "pi3d", "rho3d",
                 "dz8w", "u3d", "v3d"):
        a = inputs[name]
        if a.shape != (nz, ncol) or a.dtype != DTYPE or not a.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous float32 {(nz, ncol)}")
    if inputs["w"].shape != (nz + 1, ncol):
        raise ValueError(f"w must be {(nz + 1, ncol)}, got {inputs['w'].shape}")
    for name in ("xland", "hpbl", "hpbl_hold"):
        if inputs[name].shape != (ncol,):
            raise ValueError(f"{name} must be ({ncol},)")
    if znu.shape != (nz,):
        raise ValueError(f"znu must be ({nz},)")
    if outputs is None:
        outputs = {n: cp.zeros((nz, ncol), dtype=DTYPE) for n in
                   ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                    "rucuten", "rvcuten")}
        outputs.update({n: cp.zeros(ncol, dtype=DTYPE) for n in
                        ("raincv", "pratec", "hbot", "htop")})
    tile = min(ncol, KSAS_TILE_COLUMNS)
    need = variant.workspace_slots * (nz + 1) * tile
    if workspace is None or workspace.size < need:
        workspace = cp.empty(need, dtype=DTYPE)
    fn = function or _module().get_function(variant.kernel)
    consts = tuple(np.float32(constants[name]) for name in CONSTANT_ORDER)
    for col0 in range(0, ncol, tile):
        ncw = min(tile, ncol - col0)
        fn(((ncw + SAS_BLOCK - 1) // SAS_BLOCK,), (SAS_BLOCK,), (
            inputs["t3d"], inputs["qv3d"], inputs["qc3d"], inputs["qi3d"],
            inputs["p3d"], inputs["pi3d"], inputs["rho3d"], inputs["dz8w"],
            inputs["u3d"], inputs["v3d"], inputs["w"], znu,
            inputs["xland"], inputs["hpbl"], inputs["hpbl_hold"],
            outputs["rthcuten"], outputs["rqvcuten"], outputs["rqccuten"],
            outputs["rqicuten"], outputs["rucuten"], outputs["rvcuten"],
            outputs["raincv"], outputs["pratec"], outputs["hbot"],
            outputs["htop"], workspace,
            np.float32(dt), np.float32(dx), np.int32(stepcu),
            np.int32(ksas_ncloud(int(mp_physics))), np.int32(bool(f_qc)),
            np.int32(bool(f_qi)), np.int32(nz), np.int32(col0),
            np.int32(ncw), np.int32(ncol)) + consts)
    return outputs


class SasCumulus:
    """The SAS-family ``cumulus_callable`` for :class:`PhysicsDriver`.

    One class for every family member: the variant is chosen by the run's
    ``cu_physics`` at the first call.  It carries no NCA hold.  Its result
    supplies PRATEC as the held rate WRF accumulates every model step and no
    per-call RAINC increment (see the module docstring).
    """

    def __init__(self) -> None:
        self._driver = None
        self._hold = None
        self._hold_live = False
        self._workspace = None
        self.raincv = None
        self.hbot = None
        self.htop = None

    # -- driver hooks ----------------------------------------------------
    def bind_driver(self, driver) -> None:
        self._driver = driver

    def reset_pbl_height_hold(self) -> None:
        """A hold is valid only for a PBL call in the current step."""
        self._hold_live = False

    def hold_pbl_height(self, pblh) -> None:
        """module_pbl_driver.F:1025, HPBL_HOLD(i,j) = PBLH(i,j), taken by the
        driver immediately before the PBL scheme runs."""
        import cupy as cp
        if self._hold is None or self._hold.shape != pblh.shape:
            self._hold = cp.empty_like(pblh)
        self._hold[...] = pblh
        self._hold_live = True

    def release(self) -> int:
        freed = 0 if self._workspace is None else int(self._workspace.nbytes)
        self._workspace = None
        self._hold = None
        self._driver = None
        return freed

    # -- the call ----------------------------------------------------------
    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp

        from woof.core.physics import CumulusResult, _model_clock_dt

        variant = SAS_VARIANTS[int(cfg.cu_physics)]
        nz, ny, nx = state.p.shape
        ncol = ny * nx

        def lev(a):
            return cp.ascontiguousarray(a, dtype=DTYPE).reshape(nz, ncol)

        pblh = fields["pblh"]
        live = self._hold is not None and self._hold_live
        hold = self._hold if live else pblh
        self._hold_live = False
        has_qi = getattr(state, "qi", None) is not None
        has_qc = getattr(state, "qc", None) is not None
        inputs = {
            "t3d": lev(atmosphere["temperature"]), "qv3d": lev(atmosphere["qv"]),
            "qc3d": lev(atmosphere["qc"]), "qi3d": lev(atmosphere["qi"]),
            "p3d": lev(atmosphere["pressure"]), "pi3d": lev(atmosphere["exner"]),
            "rho3d": lev(atmosphere["rho"]), "dz8w": lev(atmosphere["dz"]),
            "u3d": lev(atmosphere["u"]), "v3d": lev(atmosphere["v"]),
            "w": cp.ascontiguousarray(state.w, dtype=DTYPE).reshape(nz + 1, ncol),
            "xland": cp.ascontiguousarray(fields["xland"], dtype=DTYPE).reshape(ncol),
            "hpbl": cp.ascontiguousarray(pblh, dtype=DTYPE).reshape(ncol),
            "hpbl_hold": cp.ascontiguousarray(hold, dtype=DTYPE).reshape(ncol),
        }
        znu = cp.ascontiguousarray(state.znu, dtype=DTYPE).reshape(nz)
        stepcu = int(getattr(self._driver, "stepcu", 1) or 1)
        tile = min(ncol, KSAS_TILE_COLUMNS)
        need = variant.workspace_slots * (nz + 1) * tile
        if self._workspace is None or self._workspace.size < need:
            self._workspace = None
            self._workspace = cp.empty(need, dtype=DTYPE)
        out = launch_column(
            variant, inputs=inputs, znu=znu, dt=_model_clock_dt(cfg),
            dx=float(cfg.dx), stepcu=stepcu, mp_physics=int(cfg.mp_physics),
            f_qc=has_qc, f_qi=has_qi, workspace=self._workspace)

        def grid(a):
            return a.reshape(nz, ny, nx)

        self.raincv = out["raincv"].reshape(ny, nx)
        for name in ("hbot", "htop"):
            fields[name][...] = out[name].reshape(ny, nx)
        self.hbot = fields["hbot"]
        self.htop = fields["htop"]
        for name in ("rucuten", "rvcuten"):
            fields["ksas_" + name][...] = grid(out[name])
        return CumulusResult(
            rthcuten=grid(out["rthcuten"]), rqvcuten=grid(out["rqvcuten"]),
            rqccuten=grid(out["rqccuten"]) if has_qc else None,
            rqicuten=grid(out["rqicuten"]) if has_qi else None,
            rucuten=grid(out["rucuten"]), rvcuten=grid(out["rvcuten"]),
            pratec=out["pratec"].reshape(ny, nx), persistent_pratec=True,
            raincv=self.raincv)


def workspace_bytes(variant_or_nz, nz_or_columns, columns=None):
    """Price either native SAS workspace through its declared interface."""
    if columns is None:
        return gfs_workspace_bytes(variant_or_nz, nz_or_columns)
    return ksas_workspace_bytes(variant_or_nz, nz_or_columns, columns)
