"""Grell 3D ensemble cumulus: the ``cu_physics = 5`` engine callable.

The scheme is ``woof/core/kernels/g3.cu``, WRF v4.6.1's
``phys/module_cu_g3.F`` transcribed in WRF's operation order: G3DRV and
CUP_enss_3d per column (``g3_column``) and ``conv_grell_spread3d``
(``g3_spread``).  It is held word for word against a column oracle built
from the byte-unmodified module (``tools/g3_wrf461_oracle``,
``tests/test_g3_cu5_parity.py``) under the strict arm and within a stated
tolerance under the default arm; both arms compile the same source.

This module is the seam.  :func:`run_g3` takes WRF-named domain fields in
the engine's (nz, ny, nx) layout and returns everything G3DRV and
conv_grell_spread3d write; :class:`Grell3D` packs the engine's state into
it and hands the driver its :class:`~woof.core.physics.CumulusResult`.

THE FORCING FOLD IS DONE HERE.  WRF's cumulus driver folds the radiative
and boundary-layer theta tendencies into RTHFTEN, times Exner, for
G3SCHEME and NTIEDTKESCHEME only (``module_cumulus_driver.F:867-886``)::

    RTHFTEN = (RTHFTEN + RTHRATEN + RTHBLTEN) * pi
    RQVFTEN =  RQVFTEN + RQVBLTEN

The engine's dycore exports pure advection (Grell-Freitas sums the lanes
itself and must not see them twice), so the fold happens in the adapter,
exactly as New Tiedtke's does.

HORIZONTAL COUPLING, STATED.  Grell 3D is not column-local.  Its forcing
ensemble reads w, RTHFTEN and RQVFTEN on the 3 x 3 neighbourhood
(``module_cu_g3.F:404-426``; 5 x 5 under ``cugd_avedx > 1`` through the
ave_f average), and conv_grell_spread3d spreads and smooths the result over
a 3 x 3 stencil.  WRF writes only the interior band ids+4..ide-5 (both
axes, non-periodic): columns within four points of a lateral boundary get
no convection, by WRF's own design.  The engine holds the whole domain on
one device, so the neighbours are read directly; that is the same data
WRF's HALO_CUP_G3_IN/OUT exchanges deliver.

RAIN.  RAINCV is the per-call amount ``pret * dt``; with ``cudt = 0``
(pinned by ``validate_run_config``) the driver adds it once per step,
which is WRF's ``RAINC += PRATEC * DT`` (module_physics_addtendc.F:2141)
word for word.

DIAGNOSTICS HELD ON THE ADAPTER.  The closure-ensemble outputs WRF keeps as
restart-only state (Registry io "r": APR_GR/W/MC/ST/AS/CAPMA/CAPME/CAPMI,
XF_ENS, PR_ENS, the shallow and deep level diagnostics, GD_CLOUD and
GD_CLOUD2) are computed by the kernel every call and held on the adapter
as :attr:`Grell3D.diagnostics` and canonical driver.fields carriers.
The normal restart inventory carries them. cu_diag=1 adds WRF
convtrans_prep accumulators and time-averaged cloud/rain outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DTYPE = np.float32

#: WRF's model constants as G3DRV receives them (share/module_model_constants.F).
XLV = np.float32(2.5e6)
CP = np.float32(np.float32(7.0) * np.float32(287.0) / np.float32(2.0))
R_V = np.float32(461.6)
G = np.float32(9.81)

#: The Grell ensemble dimensions G3 is defined at (Registry defaults).
MAXIENS, MAXENS, MAXENS2, MAXENS3 = 1, 3, 3, 16
ENSDIM = MAXIENS * MAXENS * MAXENS2 * MAXENS3

LEV_IN = ("u", "v", "t", "q", "p", "pi", "rho", "rthften", "rqvften",
          "rthblten", "rqvblten")
SFC_IN = ("psfc", "ht", "xland", "gsw")
LEV_OUT = ("cugd_tten", "cugd_qvten", "cugd_qcten", "cugd_ttens",
           "cugd_qvtens", "rqccuten_g3", "rqicuten_g3", "gdc", "gdc2")
SFC_OUT = ("raincv_g3", "pratec_g3", "htop", "hbot", "mass_flux", "edt_out",
           "xmb_shallow")
ISFC_OUT = ("k22_shallow", "kbcon_shallow", "ktop_shallow", "ktop_deep")
APR = ("apr_gr", "apr_w", "apr_mc", "apr_st", "apr_as", "apr_capma",
       "apr_capme", "apr_capmi")

#: Launch block.  The workspace is indexed by the thread id within a tile.
G3_BLOCK = 64

#: The kernel's workspace slot count, G3_SLOT_COUNT in kernels/g3.cu: one
#: (nz+1)-word level array per slot per column in flight.  Stated here so
#: preflight can price the workspace without a device;
#: tests/test_g3_cu5_workspace.py re-derives it from the .cu source and
#: from the compiled kernel and fails if either side moves alone.
G3_SLOT_COUNT = 203

#: Upper bound on the column workspace one call allocates.  The workspace
#: is G3_SLOT_COUNT level arrays per column in flight, so the bound sets
#: the tile; the kernel walks the band in tiles of that many columns.
G3_WORKSPACE_BYTES = 384 * 1024 * 1024


def g3_band(nx: int, ny: int, periodic_x: bool = False,
            periodic_y: bool = False) -> tuple[int, int, int, int]:
    """G3DRV's written band, 0-based inclusive (module_cu_g3.F:213-234)."""
    ib, ie = (0, nx - 1) if periodic_x else (4, nx - 5)
    jb, je = (0, ny - 1) if periodic_y else (4, ny - 5)
    return ib, ie, jb, je


def spread_band(nx: int, ny: int) -> tuple[int, int, int, int]:
    """conv_grell_spread3d's written band: ids+4..ide-5 always (:5211-5229)."""
    return 4, nx - 5, 4, ny - 5


def _module(module=None, dimensions=(1, 3, 3, 16, 144)):
    if module is not None:
        return module
    from woof.core.kernels import load_module

    if dimensions == (1, 3, 3, 16, 144):
        return load_module("g3")
    from woof.core.kernels import load_module_int_defines
    return load_module_int_defines("g3", tuple(zip(
        ("G3_MAXENS", "G3_MAXENS2", "G3_MAXENS3", "G3_ENSDIM"), dimensions[1:])))


_SLOT_COUNT: dict[int, int] = {}


def g3_slot_count(module=None) -> int:
    """The kernel's own workspace slot count (``G3_SLOT_COUNT``)."""
    import cupy as cp

    mod = _module(module)
    key = id(mod)
    if key not in _SLOT_COUNT:
        out = cp.zeros(1, dtype=cp.int32)
        mod.get_function("g3_slot_count")((1,), (1,), (out,))
        _SLOT_COUNT[key] = int(out.get()[0])
    return _SLOT_COUNT[key]


def g3_tile_columns(nz: int, nband: int, slots: int,
                    budget: int = G3_WORKSPACE_BYTES) -> int:
    per_col = slots * (nz + 1) * 4
    tile = max(G3_BLOCK, (budget // per_col) // G3_BLOCK * G3_BLOCK)
    return int(max(1, min(nband, tile)))


def g3_workspace_bytes(nz: int, ny: int, nx: int,
                       slots: int = G3_SLOT_COUNT) -> int:
    """Column workspace priced for preflight: the tile actually allocated."""
    ib, ie, jb, je = g3_band(nx, ny)
    nband = max(0, ie - ib + 1) * max(0, je - jb + 1)
    if nband == 0:
        return 0
    tile = g3_tile_columns(nz, nband, slots)
    return int(tile * slots * (nz + 1) * 4)


@dataclass
class G3Step:
    """Everything one G3DRV + conv_grell_spread3d call writes."""

    lev: object            # (9, nz, ny, nx): LEV_OUT
    sfc: object            # (7, ny, nx): SFC_OUT
    isfc: object           # (4, ny, nx): ISFC_OUT
    xf_ens: object         # (ENSDIM, ny, nx)
    pr_ens: object         # (ENSDIM, ny, nx)
    apr: object            # (8, ny, nx), accumulated in place
    rthcuten: object       # (nz, ny, nx)
    rqvcuten: object
    rqccuten: object
    rqicuten: object
    raincv: object         # (ny, nx), after the spread limiters
    pratec: object
    cu_act_flag: object    # G3DRV assigns true for every domain column
    ws: object = None      # the workspace, for the caller to reuse


def run_g3(*, lev_in, w, sfc_in, kpbl, apr, dt, dx, ishallow, avedx,
           ichoice, fqi, periodic_x=False, periodic_y=False, xf_ens=None,
           pr_ens=None, module=None, workspace_bytes=G3_WORKSPACE_BYTES,
           ws=None, maxiens=1, maxens=3, maxens2=3, maxens3=16, ensdim=144) -> G3Step:
    """One cumulus call over a whole domain.

    ``lev_in`` is (len(LEV_IN), nz, ny, nx) float32 with ``rthften`` the
    FOLDED temperature forcing G3DRV receives; ``w`` is (nz+1, ny, nx);
    ``sfc_in`` (len(SFC_IN), ny, nx) with ``psfc`` = p8w(k=1) in Pa;
    ``kpbl`` (ny, nx) int32, WRF's one-based KPBL; ``apr`` (8, ny, nx) is
    accumulated in place.  ``xf_ens``/``pr_ens`` carry their previous
    values outside the band (WRF never writes them there).
    """
    import cupy as cp

    from woof.config import validate_g3_dimensions
    validate_g3_dimensions(maxiens, maxens, maxens2, maxens3, ensdim, avedx)
    dimensions = (maxiens, maxens, maxens2, maxens3, ensdim)
    mod = _module(module, dimensions)
    _, nz, ny, nx = lev_in.shape
    if lev_in.shape[0] != len(LEV_IN) or lev_in.dtype != DTYPE:
        raise ValueError("lev_in must be float32 (11, nz, ny, nx)")
    lev_in = cp.ascontiguousarray(lev_in)
    w = cp.ascontiguousarray(w, dtype=DTYPE)
    sfc_in = cp.ascontiguousarray(sfc_in, dtype=DTYPE)
    kpbl = cp.ascontiguousarray(kpbl, dtype=cp.int32)
    lev = cp.zeros((len(LEV_OUT), nz, ny, nx), dtype=DTYPE)
    sfc = cp.zeros((len(SFC_OUT), ny, nx), dtype=DTYPE)
    # G3DRV's per-call initialisation for every column (:291-315)
    sfc[SFC_OUT.index("htop")] = DTYPE(1.0)        # REAL(KTS)
    sfc[SFC_OUT.index("hbot")] = DTYPE(nz)         # REAL(KTE)
    isfc = cp.zeros((len(ISFC_OUT), ny, nx), dtype=cp.int32)
    if xf_ens is None:
        xf_ens = cp.zeros((ensdim, ny, nx), dtype=DTYPE)
    if pr_ens is None:
        pr_ens = cp.zeros((ensdim, ny, nx), dtype=DTYPE)
    ib, ie, jb, je = g3_band(nx, ny, periodic_x, periodic_y)
    nband = max(0, ie - ib + 1) * max(0, je - jb + 1)
    slots = g3_slot_count(mod)
    rthcuten = cp.zeros((nz, ny, nx), dtype=DTYPE)
    rqvcuten = cp.zeros((nz, ny, nx), dtype=DTYPE)
    if nband > 0:
        tile = g3_tile_columns(nz, nband, slots, workspace_bytes)
        need = slots * (nz + 1) * tile
        if ws is None or ws.size < need:
            ws = cp.empty(need, dtype=DTYPE)
        fn = mod.get_function("g3_column")
        for col0 in range(0, nband, tile):
            nt = min(tile, nband - col0)
            fn(((nt + G3_BLOCK - 1) // G3_BLOCK,), (G3_BLOCK,),
               (lev_in, w, sfc_in, kpbl, apr, lev, sfc, isfc, xf_ens, pr_ens,
                ws, np.int32(nt), np.int32(col0), np.int32(nx), np.int32(ny),
                np.int32(nz), np.int32(ib), np.int32(ie), np.int32(jb),
                np.int32(je), np.int32(ishallow), np.int32(avedx),
                np.int32(ichoice), np.int32(fqi), DTYPE(dt), DTYPE(dx), XLV,
                CP, G, R_V))
    rqccuten = cp.array(lev[LEV_OUT.index("rqccuten_g3")])
    rqicuten = cp.array(lev[LEV_OUT.index("rqicuten_g3")])
    raincv = cp.array(sfc[SFC_OUT.index("raincv_g3")])
    pratec = cp.array(sfc[SFC_OUT.index("pratec_g3")])
    sib, sie, sjb, sje = spread_band(nx, ny)
    nsp = max(0, sie - sib + 1) * max(0, sje - sjb + 1)
    if nsp > 0:
        tile = min(nsp, max(G3_BLOCK, workspace_bytes // (2 * (nz + 1) * 4)))
        need = 2 * (nz + 1) * tile
        if ws is None or ws.size < need:
            ws = cp.empty(need, dtype=DTYPE)
        fn = mod.get_function("g3_spread")
        pi = lev_in[LEV_IN.index("pi")]
        qv = lev_in[LEV_IN.index("q")]
        for col0 in range(0, nsp, tile):
            nt = min(tile, nsp - col0)
            fn(((nt + G3_BLOCK - 1) // G3_BLOCK,), (G3_BLOCK,),
               (lev, pi, qv, rthcuten, rqvcuten, rqccuten, rqicuten, raincv,
                pratec, ws, np.int32(nt), np.int32(col0), np.int32(nx),
                np.int32(ny), np.int32(nz), np.int32(sib), np.int32(sie),
                np.int32(sjb), np.int32(sje), np.int32(avedx), np.int32(fqi),
                DTYPE(dt)))
    return G3Step(lev=lev, sfc=sfc, isfc=isfc, xf_ens=xf_ens, pr_ens=pr_ens,
                  apr=apr, rthcuten=rthcuten, rqvcuten=rqvcuten,
                  rqccuten=rqccuten, rqicuten=rqicuten, raincv=raincv,
                  pratec=pratec, cu_act_flag=cp.ones((ny,nx), dtype=cp.int32), ws=ws)


class Grell3D:
    """``cu_physics = 5`` callable for :class:`PhysicsDriver`.

    Task-1 contract (no ``nca_seconds``): every due call returns fresh rates
    and the per-call RAINCV.  ``cudt_minutes`` is pinned to 0 for G3 by
    ``validate_run_config``: WRF accumulates the HELD PRATEC every step
    between calls, and this contract consumes RAINCV once per call, so a
    cadence above one step would drop rain.
    """

    def __init__(self):
        self._driver = None
        self._ws = None

    @property
    def diagnostics(self):
        """Views of canonical driver fields, never a second state owner."""
        fields = {} if self._driver is None else self._driver.fields
        out = {name: fields[key] for name, key in
               (("apr", "g3_apr"), ("xf_ens", "g3_xf_ens"), ("pr_ens", "g3_pr_ens"))
               if key in fields}
        for name, key, index in (("gdc", "g3_lev", 7), ("gdc2", "g3_lev", 8)):
            if key in fields: out[name] = fields[key][index]
        for key, names in (("g3_sfc", SFC_OUT), ("g3_isfc", ISFC_OUT)):
            if key in fields: out.update({name: fields[key][k] for k, name in enumerate(names)})
        out.update({key: fields[key] for key in
                    ('gd_cloud_a','qc_cu','gd_cloud2_a','qi_cu','gd_cldfr','raincv_a','raincv_b')
                    if key in fields})
        return out

    def bind_driver(self, driver) -> None:
        self._driver = driver

    def release(self) -> int:
        freed = 0 if self._ws is None else int(self._ws.nbytes)
        self._ws = None
        self._driver = None
        return freed

    def _lane(self, name, shape):
        import cupy as cp

        lane = None if self._driver is None else getattr(self._driver, name, None)
        if lane is None:
            return cp.zeros(shape, dtype=DTYPE)
        if tuple(lane.shape) != tuple(shape):
            raise ValueError(f"driver {name} must be {shape}, got {lane.shape}")
        return lane

    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp

        from woof.core.physics import CumulusResult, _model_clock_dt

        nz, ny, nx = state.p.shape
        shape = (nz, ny, nx)
        exner = atmosphere["exner"]
        # THE FOLD (module_cumulus_driver.F:879-881), including the * pi.
        rthraten = self._lane("rthratenlw", shape) + self._lane("rthratensw", shape)
        rthblten = self._lane("gf_rthblten", shape)
        rqvblten = self._lane("gf_rqvblten", shape)
        theta_advective = self._lane("gf_rthdynten", shape)
        vapour_advective = self._lane("gf_rqvdynten", shape)
        if getattr(state, "use_theta_m", 0):
            from woof.core.theta_m_physics import dry_cumulus_theta_tendency
            theta_advective = theta_advective.copy()
            dry_cumulus_theta_tendency(state, theta_advective, vapour_advective)
        thften = (theta_advective + rthraten + rthblten) * exner
        qvften = vapour_advective + rqvblten
        lev_in = cp.empty((len(LEV_IN), nz, ny, nx), dtype=DTYPE)
        lev_in[0] = atmosphere["u"]
        lev_in[1] = atmosphere["v"]
        lev_in[2] = atmosphere["temperature"]
        lev_in[3] = atmosphere["qv"]
        lev_in[4] = atmosphere["pressure"]
        lev_in[5] = exner
        lev_in[6] = atmosphere["rho"]
        lev_in[7] = thften
        lev_in[8] = qvften
        lev_in[9] = rthblten
        lev_in[10] = rqvblten
        sfc_in = cp.empty((len(SFC_IN), ny, nx), dtype=DTYPE)
        sfc_in[0] = atmosphere["p_interface"][0]
        sfc_in[1] = state.ht
        sfc_in[2] = fields["xland"]
        gsw = fields.get("gsw") if hasattr(fields, "get") else None
        sfc_in[3] = 0.0 if gsw is None else gsw
        kpbl = fields["kpbl"] if "kpbl" in fields else cp.zeros((ny, nx), cp.int32)
        diag = self.diagnostics
        # These are driver.fields carriers, so the normal restart exchange
        # restores their arrays in place before the next cumulus call.
        from woof.core.physics_inventory import cumulus_state_layout
        for key, (field_shape, dtype) in cumulus_state_layout(cfg).items():
            if key not in fields:
                fields[key] = cp.zeros(field_shape, dtype=dtype)
        for key, field in (("apr","g3_apr"),("xf_ens","g3_xf_ens"),("pr_ens","g3_pr_ens")):
            diag[key] = fields[field]
        fqi = int(getattr(state, "qi", None) is not None)
        dimensions = dict(zip(("maxiens","maxens","maxens2","maxens3","ensdim"),
                          (cfg.maxiens,cfg.maxens,cfg.maxens2,cfg.maxens3,cfg.ensdim)))
        step = run_g3(
            lev_in=lev_in, w=state.w[:nz], sfc_in=sfc_in,
            kpbl=cp.asarray(kpbl, dtype=cp.int32), apr=diag["apr"],
            dt=float(_model_clock_dt(cfg)), dx=float(cfg.dx),
            ishallow=int(cfg.ishallow), avedx=int(cfg.cugd_avedx),
            ichoice=int(cfg.clos_choice), fqi=fqi,
            periodic_x=not (cfg.open_x or cfg.specified or cfg.nested),
            periodic_y=not (cfg.open_y or cfg.specified or cfg.nested),
            xf_ens=diag["xf_ens"], pr_ens=diag["pr_ens"], ws=self._ws, **dimensions)
        self._ws = step.ws
        if cfg.lightning_option == 11:
            cloud_top = state.scratch((ny, nx), "ltng_ktop_deep", dtype=np.int32)
            cloud_top[...] = step.isfc[ISFC_OUT.index("ktop_deep")]
        for key, value in (("g3_lev",step.lev),("g3_sfc",step.sfc),("g3_isfc",step.isfc)):
            fields[key][...] = value
        for name in ("gdc", "gdc2"):
            diag[name] = fields["g3_lev"][LEV_OUT.index(name)]
        for k, name in enumerate(SFC_OUT):
            diag[name] = fields["g3_sfc"][k]
        for k, name in enumerate(ISFC_OUT):
            diag[name] = fields["g3_isfc"][k]
        if cfg.cu_diag or cfg.cu_rad_feedback:
            averages = {key:fields[key] for key in
                ('gd_cloud_a','qc_cu','gd_cloud2_a','qi_cu','gd_cldfr','raincv_a','raincv_b')}
            averages['clock'] = fields['g3_diag_clock']
            run_g3_diag(averages, cloud=diag['gdc'], ice=diag['gdc2'],
                rain=step.raincv, temperature=atmosphere['temperature'],
                pressure=atmosphere['pressure'], qv=atmosphere['qv'],
                qc=atmosphere['qc'], qi=atmosphere['qi'] if 'qi' in atmosphere else cp.zeros(shape,dtype=DTYPE),
                fqi=fqi, dt=float(_model_clock_dt(cfg)), curr_secs=float(state.elapsed_seconds),
                avg_minutes=float(cfg.convtrans_avglen_m), adaptive=cfg.use_adaptive_time_step,
                module=_module(dimensions=tuple(dimensions.values())))
            diag.update(averages)
        rqi = step.rqicuten if fqi else None
        rqc = step.rqccuten
        return CumulusResult(
            rthcuten=step.rthcuten, rqvcuten=step.rqvcuten, rqccuten=rqc,
            rqicuten=rqi, rainc=step.raincv,
            # G3DRV's native bounds include its .001 level marker.
            cloud_base=(step.sfc[SFC_OUT.index('hbot')]
                        if 99 in (cfg.ra_lw_physics, cfg.ra_sw_physics) else None),
            cloud_top=(step.sfc[SFC_OUT.index('htop')]
                       if 99 in (cfg.ra_lw_physics, cfg.ra_sw_physics) else None))


__all__ = ["Grell3D", "run_g3", "run_g3_diag", "G3Step", "g3_band", "spread_band"]


def run_g3_diag(diag, *, cloud, ice, rain, temperature, pressure, qv, qc, qi,
                fqi, dt, curr_secs, avg_minutes=30., adaptive=False, module=None):
    """WRF convtrans_prep after G3/spreading, including the first window."""
    import cupy as cp
    if not np.isfinite(avg_minutes) or np.float32(avg_minutes) * np.float32(60.) < np.float32(dt):
        raise ValueError("the averaging period must span one model step: MOD(ktau,0) would be undefined")
    for key in ('gd_cloud_a', 'qc_cu', 'gd_cloud2_a', 'qi_cu', 'gd_cldfr'):
        if key not in diag: diag[key] = cp.zeros_like(cloud)
    for key in ('raincv_a', 'raincv_b'):
        if key not in diag: diag[key] = cp.zeros_like(rain)
    if 'clock' not in diag: diag['clock'] = cp.zeros(4, dtype=cp.int32)
    mod = _module(module)
    mod.get_function('g3_diag_clock')((1,), (1,),
        (diag['clock'], DTYPE(dt), DTYPE(avg_minutes), DTYPE(curr_secs), np.int32(adaptive)))
    nvol, ncol = int(cloud.size), int(rain.size)
    mod.get_function('g3_diag')(((nvol+255)//256,), (256,),
        (cloud, ice, rain, temperature, pressure, qv, qc, qi,
         *(diag[k] for k in ('gd_cloud_a','qc_cu','gd_cloud2_a','qi_cu','gd_cldfr','raincv_a','raincv_b')),
         diag['clock'], np.int32(fqi), np.int64(nvol), np.int64(ncol)))
    mod.get_function('g3_diag_finish')((1,), (1,), (diag['clock'],))
