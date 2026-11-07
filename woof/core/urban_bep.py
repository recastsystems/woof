"""BEP, the multi-layer urban canopy model (``sf_urban_physics = 2``).

WRF v4.7.1 ``phys/module_sf_bep.F`` as one float32 CUDA column
(``kernels/urban_bep.cu``, transcribed by the BEP port and graded
here), word-identical to the gfortran/glibc build of the byte-unmodified
module on every fixture column (``tests/test_urban_bep_wrf471_parity.py``):
the street-canyon radiation (shadowing, view factors, multiple reflections),
the wall/roof/road conduction, the surface fluxes, and the momentum, heat
and TKE source-term profiles the PBL receives.

Table words are the post-``urban_param_init`` module variables of
module_sf_urban (unit conversions included), keyed by their Fortran names.

Model-lane entry points for ``woof.core.urban_driver`` (DESIGN 3.4):
:data:`STATE_SPEC`, :data:`DIMENSIONS`, :func:`init_state`,
:func:`after_lsm` (zeroing, BEP, then the shared coupling block in
:mod:`woof.core.urban_bep_couple`) and :func:`after_surface_diagnostics`.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from hashlib import sha256
import numpy as np

from woof.core.urban_state import BEP_SPEC

NDM, NZ_UM, NG_U, NWR_U, NBUI, NURBMAX, DZ_U = 2, 18, 10, 10, 1, 11, 5.0
#: WRF v4.7.1 module_sf_bep.F header (bep_ndm() ..), in the keys
#: woof.core.urban_state.WRF_URBAN_DIMENSIONS checks them against.
DIMENSIONS = {"ndm": NDM, "nz_um": NZ_UM, "ng_u": NG_U,
              "nwr_u": NWR_U, "nbui_max": NBUI}
urban_map_zrd = NZ_UM * NWR_U * NDM
urban_map_zwd = NZ_UM * NWR_U * NDM
urban_map_gd = NG_U * NDM
urban_map_zd = NZ_UM * NDM
urban_map_zdf = NZ_UM * NDM
num_urban_ndm = NDM
#: The eight layer-state arrays in kernel argument order.
_STATE_ORDER = ("trb_urb4d", "tw1_urb4d", "tw2_urb4d", "tgb_urb4d",
                "sfw1_urb3d", "sfw2_urb3d", "sfr_urb3d", "sfg_urb3d")

#: Registry.EM_COMMON rows BEP carries (options 2 and 3), in
#: woof.core.urban_state's (dtype, layers) form: a string names an
#: urban_map_* size (check_a_mundo.F:3274-3302), 0 is a 2-D field.
STATE_SPEC = dict(BEP_SPEC)

# Static ABI offsets mirrored by CUDA static_asserts in urban_bep.cu.
# (field name, numpy scalar type, element count, byte offset)
TABLE_LAYOUT = (
    ('icate', '<i4', 1, 0),
    ('capb_tbl', '<f4', 11, 4),
    ('capr_tbl', '<f4', 11, 48),
    ('capg_tbl', '<f4', 11, 92),
    ('aksb_tbl', '<f4', 11, 136),
    ('aksr_tbl', '<f4', 11, 180),
    ('aksg_tbl', '<f4', 11, 224),
    ('tblend_tbl', '<f4', 11, 268),
    ('trlend_tbl', '<f4', 11, 312),
    ('tglend_tbl', '<f4', 11, 356),
    ('albb_tbl', '<f4', 11, 400),
    ('albr_tbl', '<f4', 11, 444),
    ('albg_tbl', '<f4', 11, 488),
    ('epsb_tbl', '<f4', 11, 532),
    ('epsr_tbl', '<f4', 11, 576),
    ('epsg_tbl', '<f4', 11, 620),
    ('z0r_tbl', '<f4', 11, 664),
    ('z0g_tbl', '<f4', 11, 708),
    ('numdir_tbl', '<i4', 11, 752),
    ('street_direction_tbl', '<f4', 33, 796),
    ('street_width_tbl', '<f4', 33, 928),
    ('building_width_tbl', '<f4', 33, 1060),
    ('numhgt_tbl', '<i4', 11, 1192),
    ('height_bin_tbl', '<f4', 550, 1236),
    ('hpercent_bin_tbl', '<f4', 550, 3436),
)
TABLE_BYTES = 5636
CLASS_BYTES = 4432

TABLE_DTYPE = np.dtype({
    'names': [row[0] for row in TABLE_LAYOUT],
    'formats': [(row[1], (row[2],)) if row[2] > 1 else row[1] for row in TABLE_LAYOUT],
    'offsets': [row[3] for row in TABLE_LAYOUT], 'itemsize': TABLE_BYTES,
})


def pack_bep_table(params):
    """Return one structured numpy record, usable through .tobytes().

    Keys are exact uppercase module variable names, not URBPARM input labels.
    Rank-two inputs have Fortran shape (MAXDIRS or MAXHGTS, ICATE).
    ICATE can be 1..11. Unused classes and unused array tails are zero padded.
    Smaller leading extents are accepted if they cover every active entry.
    """
    icate = int(params['ICATE'])
    if not 1 <= icate <= NURBMAX:
        raise ValueError('ICATE must be in 1..11')
    record = np.zeros(1, dtype=TABLE_DTYPE)
    record['icate'][0] = icate
    for name, scalar, count, _ in TABLE_LAYOUT[1:]:
        source = np.asarray(params[name.upper()])
        if np.dtype(scalar).kind == 'i' and (
                not np.all(np.isfinite(source)) or not np.all(source == np.trunc(source))):
            raise ValueError(f'{name.upper()} requires int32 values')
        source = np.asarray(source, dtype=scalar)
        leading = 3 if name.startswith(('street_', 'building_width')) else (
            50 if name.startswith(('height_bin', 'hpercent_bin')) else 1)
        if leading == 1:
            if source.ndim != 1 or source.size not in (icate, NURBMAX):
                raise ValueError(f'{name.upper()} needs ({icate},) or (11,)')
            record[name][0, :source.size] = source
        else:
            if (source.ndim != 2 or source.shape[0] > leading or
                    source.shape[1] not in (icate, NURBMAX)):
                raise ValueError(f'{name.upper()} needs (<= {leading}, {icate} or 11)')
            active_count = np.asarray(params['NUMDIR_TBL' if leading == 3 else 'NUMHGT_TBL'])
            if source.shape[0] < int(np.max(active_count[:icate])):
                raise ValueError(f'{name.upper()} omits active entries')
            dest = record[name][0].reshape((leading, NURBMAX), order='F')
            dest[:source.shape[0], :source.shape[1]] = source
    return record


# Device and byte identity are both part of this cache key. Events make a
# class first initialized on one stream safe to consume on another stream.
_CLASS_CACHE = OrderedDict()
#: Default column-scratch budget per BEP call.  A BEP column's scratch is
#: ~36 KB at nz = 30 (bep_workspace_slots), so this holds ~7,000 city
#: columns per tile.
BEP_WORKSPACE_BYTES = 256 * 1024 * 1024
_TPB = 32
_THREE_D = ('a_u', 'a_v', 'a_t', 'a_e', 'b_u', 'b_v', 'b_t', 'b_e',
            'b_q', 'dlg', 'dl_u', 'vl')
_TWO_D = ('rl_up', 'rs_abs', 'emiss', 'grdflx_urb')


def bep_workspace_slots(nz):
    """Conservative high-water bound for the nested global arena, per lane.

    8192 fixed slots plus 24*(nz+1); the actual audited maximum is smaller.
    Each slot is four bytes. Blocks interleave exactly 32 lanes per element.
    """
    if nz < 1:
        raise ValueError('nz must be positive')
    return 8192 + 24 * (nz + 1)


def _class_for_table(cp, params, *, with_views=False):
    from woof.core.kernels import get_kernel
    packed = pack_bep_table(params).tobytes()
    key = (cp.cuda.Device().id, sha256(packed).digest())
    stream = cp.cuda.get_current_stream()
    if key in _CLASS_CACHE:
        table, classes, views, ready = _CLASS_CACHE[key]
        stream.wait_event(ready)
        _CLASS_CACHE.move_to_end(key)
        return (classes, views) if with_views else classes
    table = cp.asarray(np.frombuffer(packed, dtype=np.uint8))
    classes = cp.zeros(CLASS_BYTES, dtype=cp.uint8)
    get_kernel('urban_bep', 'urban_bep_class_init')((1,), (1,), (table, classes))
    views = cp.zeros((NURBMAX, 795), dtype=cp.float32)
    scratch = cp.empty((NURBMAX, 8192, _TPB), dtype=cp.float32)
    get_kernel('urban_bep', 'urban_bep_view_init')(
        (NURBMAX,), (1,), (classes, views, scratch))
    ready = cp.cuda.Event()
    ready.record(stream)
    _CLASS_CACHE[key] = (table, classes, views, ready)
    if len(_CLASS_CACHE) > 16:
        _CLASS_CACHE.popitem(last=False)
    return (classes, views) if with_views else classes


def launch_bep_columns(*, params, frc_urb2d, utype_urb2d, dz8w,
                       u_phy, v_phy, th_phy, rho, p_phy,
                       swdown, glw, cosz_urb2d, omg_urb2d, declin_urb, dt,
                       lp_urb2d, lb_urb2d, hgt_urb2d, hi_urb2d,
                       trb_urb4d, tw1_urb4d, tw2_urb4d, tgb_urb4d,
                       sfw1_urb3d, sfw2_urb3d, sfr_urb3d, sfg_urb3d,
                       nz=None, num_urban_hi=None, tile_columns=None,
                       workspace=None, outputs=None, columns=None,
                       workspace_bytes=None, cache_view_factors=True, **output_arrays):
    """Run BEP and return output arrays, including int32 error_flags.

    Existing outputs can be supplied in outputs or as named keyword arrays.
    Nonurban columns are untouched; newly allocated outputs start at zero.
    error_flags holds a Fortran source line on STOP/fatal or bound failure.
    State arrays update in place. tsk_rural_bep belongs to the LSM handoff.

    ``columns`` is the int32 flat index list of the FRC_URB2D > 0 columns
    (computed here, one host read of its length, when not given; a caller
    whose urban fraction is fixed for the run passes it once and reuses it).
    Tiles then cover only city columns.  ``workspace_bytes`` bounds the
    column scratch (default :data:`BEP_WORKSPACE_BYTES`).
    """
    import cupy as cp
    from woof.core.kernels import get_kernel

    surface_shape = frc_urb2d.shape
    if len(surface_shape) != 2 or len(u_phy.shape) != 3:
        raise ValueError('surface shape must be (ny,nx); atmosphere (nz,ny,nx)')
    nz = u_phy.shape[0] if nz is None else int(nz)
    ncol = int(np.prod(surface_shape))
    num_urban_hi = hi_urb2d.shape[0] if num_urban_hi is None else int(num_urban_hi)
    if not 0 <= num_urban_hi < NZ_UM:
        raise ValueError('num_urban_hi must be less than NZ_UM=18')
    if nz < 1 or ncol > np.iinfo(np.int32).max // (nz + 1):
        raise ValueError('invalid nz or column count exceeds int32 addressing')

    def check(name, value, shape, dtype=cp.float32):
        if not isinstance(value, cp.ndarray) or value.dtype != dtype:
            raise TypeError(f'{name} must be a CuPy {dtype} array')
        if value.shape != shape or not value.flags.c_contiguous:
            raise ValueError(f'{name} must be C contiguous with shape {shape}')
        if value.device.id != cp.cuda.Device().id:
            raise ValueError(f'{name} must be on the current CUDA device')

    check('utype_urb2d', utype_urb2d, surface_shape, cp.int32)
    for name, value in [('frc_urb2d',frc_urb2d),('swdown',swdown),('glw',glw),
                        ('cosz_urb2d',cosz_urb2d),('omg_urb2d',omg_urb2d),
                        ('lp_urb2d',lp_urb2d),('lb_urb2d',lb_urb2d),('hgt_urb2d',hgt_urb2d)]:
        check(name, value, surface_shape)
    for name, value in [('dz8w',dz8w),('u_phy',u_phy),('v_phy',v_phy),
                        ('th_phy',th_phy),('rho',rho),('p_phy',p_phy)]:
        check(name, value, (nz, *surface_shape))
    check('hi_urb2d', hi_urb2d, (num_urban_hi, *surface_shape))
    states = (trb_urb4d,tw1_urb4d,tw2_urb4d,tgb_urb4d,
              sfw1_urb3d,sfw2_urb3d,sfr_urb3d,sfg_urb3d)
    layers = {"zrd": urban_map_zrd, "zwd": urban_map_zwd, "gd": urban_map_gd,
              "zd": urban_map_zd, "zdf": urban_map_zdf, "ndm": num_urban_ndm}
    for name, value in zip(_STATE_ORDER, states):
        check(name, value, (layers[STATE_SPEC[name][1]], *surface_shape))
    result = dict(outputs or {})
    result.update(output_arrays)
    shapes = {name:(nz, *surface_shape) for name in _THREE_D}
    shapes.update({name:surface_shape for name in _TWO_D})
    shapes['sf'] = (nz+1, *surface_shape)
    shapes['error_flags'] = surface_shape
    if set(result) - set(shapes):
        raise TypeError(f'unknown BEP outputs: {sorted(set(result)-set(shapes))}')
    for name, shape in shapes.items():
        dtype = cp.int32 if name == 'error_flags' else cp.float32
        if name not in result:
            result[name] = cp.zeros(shape, dtype=dtype)
        check(name, result[name], shape, dtype)
    if ncol == 0:
        return result
    if columns is None:
        columns = cp.flatnonzero(frc_urb2d.ravel() > 0).astype(cp.int32)
    elif (not isinstance(columns, cp.ndarray) or columns.dtype != cp.int32
          or columns.ndim != 1):
        raise ValueError('columns must be a 1-D int32 CuPy index array')
    nrun = int(columns.size)
    if nrun == 0:
        return result
    slots = bep_workspace_slots(nz)
    fn = get_kernel('urban_bep','urban_bep_column')
    budget = BEP_WORKSPACE_BYTES if workspace_bytes is None else int(workspace_bytes)
    # One tile = as many city columns as the scratch budget holds, in whole
    # 32-lane blocks.  Each column is a long serial thread (view factors,
    # two radiation solves, four conduction solves), so throughput is
    # columns in flight: MEASURED on test host A's RTX 4090 at nz=30,
    # 38,889 city columns: 64 MiB (1,856 per tile) 428 ms per call.
    cap = max(_TPB, (budget // (4*slots*_TPB))*_TPB)
    tile = min(nrun, cap)
    if tile_columns is not None:
        if int(tile_columns) < 1:
            raise ValueError('tile_columns must be positive')
        tile = min(tile,int(tile_columns))
    blocks = (tile+_TPB-1)//_TPB
    if workspace is None:
        workspace = cp.empty((blocks, slots, _TPB),dtype=cp.float32)
    else:
        check('workspace',workspace,(blocks,slots,_TPB))
    classes, views = _class_for_table(cp,params,with_views=True)
    prefix = (frc_urb2d,utype_urb2d,dz8w,u_phy,v_phy,th_phy,rho,p_phy,
              swdown,glw,cosz_urb2d,omg_urb2d,np.float32(declin_urb),np.float32(dt),
              lp_urb2d,lb_urb2d,hgt_urb2d,hi_urb2d,classes,*states)
    ordered = tuple(result[name] for name in (*_THREE_D,'sf',*_TWO_D,'error_flags'))
    for offset in range(0,nrun,tile):
        count = min(tile,nrun-offset)
        fn(((count+_TPB-1)//_TPB,),(_TPB,),
           (*prefix,*ordered,workspace,np.int32(slots),np.int32(nz),np.int32(ncol),
            np.int32(num_urban_hi),np.int32(offset),np.int32(count),
            columns,np.int32(1),views,np.int32(cache_view_factors)))
    return result


# ---------------------------------------------------------------------------
# DESIGN 3.4 model-lane entry points
# ---------------------------------------------------------------------------

#: The module_sf_urban words BEP's init_para reads (module_sf_bep.F:3057-3098),
#: as urban_param_init leaves them.
BEP_TABLE_WORDS = ("ICATE",) + tuple(row[0].upper() for row in TABLE_LAYOUT[1:])

#: UrbanState.pbl_terms key -> launch_bep_columns output name.  a_q_bep is
#: not a BEP output (the couple zeroes it).
PBL_OUTPUT_NAMES = {f"{name}_bep": name for name in (*_THREE_D, "sf")}


def bep_table_words(params) -> dict:
    """The :data:`BEP_TABLE_WORDS` out of an ``UrbanParams`` (mapping or
    attribute access, WRF's upper-case names; ``woof.core.urban_tables``
    keeps ICATE as the ``icate`` attribute)."""
    words = {}
    for name in BEP_TABLE_WORDS:
        try:
            words[name] = (params[name] if isinstance(params, Mapping)
                           else getattr(params, name))
        except (KeyError, AttributeError):
            if name != "ICATE":
                raise
            words[name] = int(getattr(params, "icate"))
    return words


def init_state(state, params, *, tlayer0, tsurface0=None, tdeep0=None,
               smois=None, restart: bool = False) -> None:
    """Nothing left to do: urban_var_init's BEP arm (module_sf_urban.F:
    2912-2930 wall/roof/road layers, :2974-2990 the PBL terms) is already
    applied by ``woof.core.urban_state.urban_var_init_host``, graded bit
    for bit by tests/test_urban_init_wrf471_parity.py.  Kept as the DESIGN
    3.4 hook so a BEP-only cold-start step has one obvious home."""
    del state, params, tlayer0, tsurface0, tdeep0, smois, restart


def _bep_out(state):
    """BEP's per-call column outputs (noahdrv.F locals EMISS_URB, RL_UP_URB,
    RS_ABS_URB, GRDFLX_URB), held on the state so they are allocated once."""
    import cupy as cp

    out = getattr(state, "bep_out", None)
    if out is None:
        shape = state.frc_urb2d.shape
        out = {n: cp.zeros(shape, dtype=cp.float32)
               for n in ("rl_up_urb", "rs_abs_urb", "emiss_urb",
                         "grdflx_urb")}
        out["error_flags"] = cp.zeros(shape, dtype=cp.int32)
        try:
            state.bep_out = out
        except AttributeError:          # a frozen/slotted state: per call
            pass
    return out


def run_bep(state, params, *, atmosphere: Mapping, fields: Mapping,
            dt: float, solar) -> dict:
    """noahdrv.F:1603-1631 (== noahmpdrv.F:3608-3642): zero, then BEP.

    ``solar`` carries WRF's ``declin_urb`` (scalar, radians), ``cosz_urb2d``
    and ``omg_urb2d`` (the radiation driver's coszen and hour angle), as
    ``declin``/``coszen``/``hrang``.
    """
    from woof.core import health_ledger

    out = _bep_out(state)
    for n in ("rl_up_urb", "rs_abs_urb", "emiss_urb", "grdflx_urb"):
        out[n].fill(0.0)                                   # :1607-1610
    terms = state.pbl_terms
    nz = atmosphere["dz"].shape[0]
    terms["b_q_bep"][:nz] = 0.0                            # :1611
    outputs = {PBL_OUTPUT_NAMES[k]: v for k, v in terms.items()
               if k in PBL_OUTPUT_NAMES}
    outputs.update(rl_up=out["rl_up_urb"], rs_abs=out["rs_abs_urb"],
                   emiss=out["emiss_urb"], grdflx_urb=out["grdflx_urb"],
                   error_flags=out["error_flags"])
    launch_bep_columns(
        params=bep_table_words(params), frc_urb2d=state.frc_urb2d,
        utype_urb2d=state.utype_urb2d, dz8w=atmosphere["dz"],
        u_phy=atmosphere["u"], v_phy=atmosphere["v"],
        th_phy=atmosphere["theta"], rho=atmosphere["rho"],
        p_phy=atmosphere["pressure"], swdown=fields["swdown"],
        glw=fields["glw"], cosz_urb2d=solar.coszen, omg_urb2d=solar.hrang,
        declin_urb=float(solar.declin), dt=float(dt),
        lp_urb2d=state.lp_urb2d, lb_urb2d=state.lb_urb2d,
        hgt_urb2d=state.hgt_urb2d, hi_urb2d=state.hi_urb2d,
        trb_urb4d=state.trb_urb4d, tw1_urb4d=state.tw1_urb4d,
        tw2_urb4d=state.tw2_urb4d, tgb_urb4d=state.tgb_urb4d,
        sfw1_urb3d=state.sfw1_urb3d, sfw2_urb3d=state.sfw2_urb3d,
        sfr_urb3d=state.sfr_urb3d, sfg_urb3d=state.sfg_urb3d,
        outputs=outputs)
    # The health ledger's slots are uint32 status words, and every other
    # site reports one.  error_flags is int32 (it carries the Fortran source
    # line, always >= 0), and cupy will not OR an int32 word into a uint32
    # slot (TypeError, same_kind), so since ac988f1fd deferred every
    # single-device forecast step's health reads, option 2 crashed on its
    # first step.  The cast keeps every line number (0 .. 2**31-1).
    status = out["error_flags"].max().reshape(1).astype(np.uint32)

    def refuse(line: int) -> None:
        raise RuntimeError(
            f"BEP stopped where WRF stops (module_sf_bep.F:{line}): a wall, "
            "roof or road layer fell below 100 K (upward_rad, :3240-3273) "
            "or the gridded building heights need more than nz_um=18 urban "
            "levels (icBEPHI_XY, :3474-3477); continuing would integrate a "
            "state WRF itself refuses to")

    if health_ledger.read_status(status, site="urban_bep", describe=refuse):
        refuse(int(status[0].item()))
    return out


def after_lsm(state, params, *, lsm: int, fields: Mapping,
              atmosphere: Mapping, dt: float, itimestep: int, solar,
              cfg=None) -> None:
    """Option 2 after the LSM: BEP, then the shared coupling block."""
    from woof.core.urban_bep_couple import couple

    del itimestep, cfg
    out = run_bep(state, params, atmosphere=atmosphere, fields=fields,
                  dt=dt, solar=solar)
    couple(state, lsm=lsm, fields=fields, atmosphere=atmosphere, dt=dt,
           bep_out=out)


def after_surface_diagnostics(state, *, lsm: int, fields: Mapping,
                              atmosphere: Mapping, cfg=None) -> None:
    """module_surface_driver.F:3022-3035 / :3408-3421 (options 2 and 3)."""
    from woof.core import urban_bep_couple

    urban_bep_couple.after_surface_diagnostics(
        state, lsm=lsm, fields=fields, atmosphere=atmosphere, cfg=cfg)
