"""Native WRF ideal-fire setup, executed on the selected GPU."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import math
import numpy as np

MODULE_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false")
MODULE_KEY = "woof.core.sfire_ideal:sfire_ideal"
VERTICAL_FIELDS = ("znw", "znu", "dnw", "rdnw", "dn", "rdn", "fnp", "fnm",
                   "c1f", "c2f", "c3f", "c4f", "c1h", "c2h", "c3h", "c4h")
VERTICAL_SCALARS = ("cf1", "cf2", "cf3", "cfn", "cfn1", "rdx", "rdy")


def module_source(kernel_dir=None):
    from woof.core.kernels import module_source as compose
    root = Path(__file__).parent / "kernels" if kernel_dir is None else Path(kernel_dir)
    return compose("sfire_ideal", kernel_dir=root)


@lru_cache(maxsize=None)
def _module(device):
    import cupy as cp
    from cupy.cuda import compiler
    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile
    source = module_source()
    with cp.cuda.Device(device), observe_module_compile(MODULE_KEY):
        ptx, _ = compiler.compile_using_nvrtc(source, MODULE_OPTIONS, None, "sfire_ideal.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS, module=None)
    return module


def _launch(name, count, args):
    import cupy as cp
    _module(cp.cuda.Device().id).get_function(name)(((count + 127) // 128,), (128,), args)


def _positive(value, name):
    result = np.float32(value)
    if not math.isfinite(float(result)) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def build_vertical(nz, *, p_top, dx, dy, stretch_grd=False, stretch_hyp=False,
                   z_grd_scale=0.4, eta_levels=None, hybrid_opt=0, etac=0.2):
    """Generate source eta, differencing and hybrid coefficients in native order."""
    import cupy as cp
    if isinstance(nz, bool) or int(nz) != nz or nz < 3:
        raise ValueError("native fire initialization needs at least three mass levels for cf1..cf3")
    if hybrid_opt not in (0, 1, 2, 3):
        raise ValueError("native fire hybrid_opt must be 0, 1, 2 or 3")
    scale = _positive(z_grd_scale, "z_grd_scale") if stretch_grd and eta_levels is None else np.float32(z_grd_scale)
    if hybrid_opt == 2 and (not math.isfinite(etac) or not 0 <= etac < 1):
        raise ValueError("hybrid_opt=2 needs 0 <= etac < 1 for its polynomial denominator")
    if eta_levels is None:
        eta = cp.zeros(nz + 1, cp.float32)
    else:
        values = tuple(np.float32(value) for value in eta_levels)
        if len(values) != nz + 1 or values[0] != 1 or values[-1] != 0:
            raise ValueError("native eta_levels must contain nz+1 interfaces beginning at 1 and ending at 0")
        if any(not math.isfinite(float(v)) for v in values) or any(a <= b for a, b in zip(values, values[1:])):
            raise ValueError("native eta_levels must decrease strictly to keep layer denominators finite")
        eta = cp.asarray(values, cp.float32)
    top = cp.ascontiguousarray(cp.asarray(p_top, cp.float32).reshape(1))
    out = cp.zeros((len(VERTICAL_FIELDS), nz + 1), cp.float32)
    scalar = cp.zeros(len(VERTICAL_SCALARS), cp.float32)
    _launch("sfire_ideal_vertical", 1, (out, scalar, eta, top, np.int32(nz),
        np.int32(stretch_grd), np.int32(stretch_hyp), np.int32(eta_levels is not None),
        np.int32(hybrid_opt), scale, np.float32(etac), _positive(dx, "dx"), _positive(dy, "dy")))
    result = {name: out[k, :nz + (name.endswith("f") or name == "znw")]
              for k, name in enumerate(VERTICAL_FIELDS)}
    result.update({name: scalar[k].reshape(()) for k, name in enumerate(VERTICAL_SCALARS)})
    result["p_top"] = top[0].reshape(())
    return result


def mountain(shape, dx, dy, *, kind, height, start_x, start_y, end_x, end_y):
    """Prepare the source hill or ridge on an atmosphere or refined fire grid."""
    import cupy as cp
    ny, nx = map(int, shape)
    if min(nx, ny) < 1 or kind not in (0, 1, 2, 3):
        raise ValueError("fire_mountain_type must be 0, 1, 2 or 3 on a nonempty grid")
    values = tuple(np.float32(v) for v in (height, start_x, start_y, end_x, end_y))
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("native fire mountain parameters must be finite")
    if (kind in (1,3) and values[3] == values[1]) or (kind in (1,2) and values[4] == values[2]):
        raise ValueError("native fire mountain endpoints must differ to keep coordinate denominators finite")
    result = cp.empty((ny, nx), cp.float32)
    _launch("sfire_ideal_mountain", nx * ny, (result, np.int32(nx), np.int32(ny),
        _positive(dx, "dx"), _positive(dy, "dy"), np.int32(kind), *values))
    return result


def terrain_gradient(height, dx, dy):
    """Original central differences after the source boundary continuation."""
    import cupy as cp
    from woof.core.sfire_core import continue_at_boundary
    height = cp.ascontiguousarray(cp.asarray(height, cp.float32))
    if height.ndim != 2 or min(height.shape) < 2:
        raise ValueError("native fire terrain gradient needs at least two cells on each axis")
    ny, nx = height.shape
    padded = cp.zeros((ny + 2, nx + 2), cp.float32)
    padded[1:-1, 1:-1] = height
    continue_at_boundary(padded, bias=0.0)
    gx, gy = cp.empty_like(height), cp.empty_like(height)
    _launch("sfire_ideal_gradient", nx * ny, (padded, gx, gy, np.int32(nx), np.int32(ny),
        _positive(dx, "dx"), _positive(dy, "dy")))
    return gx, gy


def soil(tsk, tmn, *, scheme, layers):
    """Source process_soil_ideal depths and initialized soil columns."""
    import cupy as cp
    tsk, tmn = (cp.ascontiguousarray(cp.asarray(v, cp.float32)) for v in (tsk, tmn))
    if tsk.ndim != 2 or tsk.shape != tmn.shape:
        raise ValueError("native soil TSK and TMN must be matching two-dimensional fields")
    if isinstance(layers, bool) or int(layers) != layers or layers < 1:
        raise ValueError("native soil layer count must be a positive integer")
    if layers > 1 and ((scheme == 1 and layers != 5) or
                       (scheme in (2, 4) and layers != 4) or
                       (scheme == 3 and layers not in (6, 9))):
        raise ValueError("native ideal soil depth tables need SLAB=5, Noah/Noah-MP=4 or RUC=6/9 layers")
    zs, dzs = cp.zeros(layers, cp.float32), cp.zeros(layers, cp.float32)
    tslb, smois = (cp.zeros((layers, *tsk.shape), cp.float32) for _ in range(2))
    if layers > 1 and scheme in (1, 2, 3, 4):
        _launch("sfire_ideal_soil_depth", 1, (zs, dzs, np.int32(layers), np.int32(scheme)))
        _launch("sfire_ideal_soil", tsk.size, (tsk, tmn, zs, tslb, smois,
            np.int32(tsk.size), np.int32(layers), np.int32(scheme)))
    return dict(zs=zs, dzs=dzs, tslb=tslb, smois=smois)


def coordinates(shape, dx, dy):
    import cupy as cp
    ny, nx = map(int, shape)
    x, y = (cp.empty((ny, nx), cp.float32) for _ in range(2))
    _launch("sfire_ideal_coordinates", nx * ny, (x, y, np.int32(nx), np.int32(ny),
        _positive(dx, "dx"), _positive(dy, "dy")))
    return x, y


def initialize_native_landuse(lu_index, table, *, julday, cen_lat, iswater, isice,
        snowc=0., xice=0., snoalb=0., initial_albbck=0., usemonalb=False,
        fractional_seaice=0, nodata_category=0):
    """Original landuse_init lookup and sea-ice/snow branches on the device."""
    import cupy as cp
    lu = cp.ascontiguousarray(cp.asarray(lu_index, cp.float32))
    values = cp.ascontiguousarray(cp.asarray(table, cp.float32))
    if lu.ndim != 2 or values.ndim != 3 or values.shape[2] != 7:
        raise ValueError("native land use requires a 2D category field and a (seasons,categories,7) table")
    seas, cats, _ = values.shape
    if not 1 <= cats <= 100 or not 1 <= seas <= 12:
        raise ValueError("native LANDUSE.TBL exceeds landuse_init's 100-category/12-season table")
    if fractional_seaice not in (0,1):
        raise ValueError("native fractional_seaice must be 0 or 1")
    season = 2 if julday < 105 or julday > 288 else 1
    if cen_lat < 0:
        season = 3-season
    if seas == 1:
        season = 1
    if season > seas:
        raise ValueError("native land-use season is absent from LANDUSE.TBL")
    def field(value):
        return cp.ascontiguousarray(cp.broadcast_to(cp.asarray(value,cp.float32),lu.shape))
    out=cp.empty((11,*lu.shape),cp.float32)
    ivg=cp.empty(lu.shape,cp.int32);status=cp.empty_like(ivg)
    _launch("sfire_ideal_landuse",lu.size,(lu,field(snowc),field(xice),field(snoalb),field(initial_albbck),values,
        out,ivg,status,*map(np.int32,(lu.size,cats,seas,season,iswater,isice,fractional_seaice,usemonalb,nodata_category))))
    if bool(cp.any(status)):
        raise ValueError("native LANDUSE.TBL category is outside its table; zero with iswater=0 needs an explicit no-data category")
    names=("albedo","albbck","mavail","emiss","embck","znt","z0","thc","xland","xicem","landmask")
    result={name:out[k] for k,name in enumerate(names)}
    result.update(ivgtyp=ivg,season=season)
    return result


def diagnose_moisture_surface(atmosphere, cf1, cf2, cf3, surface):
    """Provide current near-ground diagnostics when no surface model produces them.

    The native three-level ground-temperature extrapolation is applied per
    column. Vapor uses the lowest mass level and pressure uses the ground
    interface. No exchange coefficient, flux or atmospheric field is changed.
    """
    import cupy as cp
    temperature, theta, qv = (cp.ascontiguousarray(cp.asarray(atmosphere[name],cp.float32))
                              for name in ("temperature","theta","qv"))
    if temperature.ndim!=3 or temperature.shape[0]<3 or theta.shape!=temperature.shape or qv.shape!=temperature.shape:
        raise ValueError("fire moisture diagnostics need matching temperature/theta/vapor columns with three mass levels")
    shape=temperature.shape[1:]
    pressure=cp.ascontiguousarray(cp.asarray(atmosphere["p_interface"][0],cp.float32))
    if pressure.shape!=shape:
        raise ValueError("fire moisture surface pressure differs from the atmospheric column shape")
    for name in ("t2","th2","q2","psfc"):
        if surface[name].shape!=shape or surface[name].dtype!=cp.float32:
            raise ValueError(f"fire moisture diagnostic {name} must be a float32 surface field")
    coeff=cp.concatenate([cp.asarray(value,cp.float32).reshape(1) for value in (cf1,cf2,cf3)])
    _launch("sfire_moisture_surface_diagnostics",temperature.shape[1]*temperature.shape[2],
        (temperature,theta,qv,pressure,coeff,surface["t2"],surface["th2"],surface["q2"],surface["psfc"],np.int32(pressure.size)))
    return surface


def build_state(cfg, raw_sounding, input_fields=None):
    """Build a native ideal atmosphere and the surface/refined-grid input payload.

    The caller attaches PhysicsDriver after consuming the returned surface and
    fire fields. The fourth result contains static geography only and can be
    added to subsequent history frames without replacing evolved fields.
    """
    import cupy as cp
    from woof.core.sfire_ideal_atmos import prepare_sounding, interpolate_sounding, initialize_atmosphere
    from woof.core.sfire_coupler import _pad
    from woof.core.sfire_atm import interpolate_2d
    from woof.core.sfire_phys import FuelTable
    from woof.core.state import DomainState
    from woof.sfire_config import validate_fuel_subcell_geometry
    validate_fuel_subcell_geometry(cfg)
    if cfg.ifire != 2:
        raise ValueError("native ideal fire preparation requires ifire=2")
    if cfg.terrain_opt != 1:
        raise ValueError("native ideal fire preparation requires terrain_opt=1 for its column-dependent base state")
    if cfg.fire_read_atm_grad and cfg.fire_mountain_type == 0:
        raise ValueError("WRF ideal fire cannot read atmospheric terrain gradients: fire_read_atm_grad is unsupported in its initializer")
    ny, nx = cfg.ny, cfg.nx
    fy, fx = ny * cfg.sr_y, nx * cfg.sr_x
    fields = {str(key).upper().removeprefix("INPUT_"): value for key, value in (input_fields or {}).items()}
    native_shape = (ny + 1, nx + 1)
    fine_shape = (fy + cfg.sr_y, fx + cfg.sr_x)

    def supplied(name, shape):
        value = fields.get(name)
        if value is None:
            raise ValueError(f"native ideal fire {name} file is enabled but its Rust input is absent")
        value = cp.ascontiguousarray(cp.asarray(value, cp.float32))
        if value.shape != shape:
            raise ValueError(f"native ideal fire {name} file shape {value.shape} differs from {shape}")
        return value

    # Spacing is a device division even when used later as a launch scalar.
    fine_spacing = cp.empty(2, cp.float32)
    _launch("sfire_ideal_spacing", 1, (fine_spacing, np.float32(cfg.dx), np.float32(cfg.dy),
        np.int32(cfg.sr_x), np.int32(cfg.sr_y)))
    fdx, fdy = map(np.float32, cp.asnumpy(fine_spacing))
    mountain_args = dict(kind=cfg.fire_mountain_type, height=cfg.fire_mountain_height,
        start_x=cfg.fire_mountain_start_x, start_y=cfg.fire_mountain_start_y,
        end_x=cfg.fire_mountain_end_x, end_y=cfg.fire_mountain_end_y)
    terrain = mountain(native_shape, cfg.dx, cfg.dy, **mountain_args)
    if cfg.fire_mountain_type:
        zsf_native = mountain(fine_shape, fdx, fdy, **mountain_args)
    else:
        if cfg.fire_read_atm_ht:
            terrain = supplied("HT", native_shape)
        if cfg.fire_read_fire_ht:
            zsf_native = supplied("ZSF", fine_shape)
        else:
            # Source interpolate_2d leaves half-cell edge strips unwritten.
            # Continue the coarse grid before the original interpolation.
            coarse = _pad(terrain, native_shape, linear=True)
            zsf_native = interpolate_2d(coarse, fine_shape, cfg.sr_x, cfg.sr_y,
                coarse_origin=(1., 1.), fine_origin=((cfg.sr_x-1)*0.5, (cfg.sr_y-1)*0.5))
    if cfg.fire_mountain_type == 0 and cfg.fire_read_fire_grad:
        gx = supplied("DZDXF", fine_shape)[:fy, :fx].copy()
        gy = supplied("DZDYF", fine_shape)[:fy, :fx].copy()
    else:
        gx_native, gy_native = terrain_gradient(zsf_native, fdx, fdy)
        gx, gy = gx_native[:fy, :fx].copy(), gy_native[:fy, :fx].copy()
    zsf = zsf_native[:fy, :fx].copy()
    dry = prepare_sounding(raw_sounding, dry=True)
    moist = prepare_sounding(raw_sounding, dry=False)
    p_top = interpolate_sounding(dry["p_moist"], dry["height"], cfg.ztop)
    vertical = build_vertical(cfg.nz, p_top=p_top, dx=cfg.dx, dy=cfg.dy,
        stretch_grd=cfg.stretch_grd, stretch_hyp=cfg.stretch_hyp, z_grd_scale=cfg.z_grd_scale,
        eta_levels=cfg.eta_levels, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac)
    atmosphere = initialize_atmosphere(cfg, terrain[:ny, :nx].copy(), vertical, {"dry": dry, "moist": moist})
    state = DomainState(cfg)
    for name, key in (("u", "U"), ("v", "V"), ("w", "W"), ("php", "PH"),
                      ("mup", "MU"), ("pb", "PB"), ("alb", "ALB"), ("al", "AL"), ("alt", "ALT")):
        getattr(state, name)[...] = atmosphere[key]
    state.phb[...] = atmosphere["PHB"]
    state.mub = None
    state.mub2d[...] = atmosphere["MUB"]
    state.ht[...] = terrain[:ny, :nx]
    if state.qv is not None:
        state.qv[...] = atmosphere["QVAPOR"]
    elif cfg.moist:
        raise ValueError("native moist ideal state did not allocate its vapor carrier")
    thermal = atmosphere["T_DRY"]
    if getattr(state, "use_theta_m", 0):
        from woof.core.constants import RVOVRD
        thermal = thermal * (np.float32(1) + np.float32(RVOVRD) * state.qv)
    _launch("sfire_ideal_state_fields", state.p.size, (atmosphere["T_INIT"], thermal,
        atmosphere["PB"], atmosphere["P"], state.thb, state.thp, state.p, np.int32(state.p.size)))
    for name in VERTICAL_FIELDS:
        getattr(state, name)[...] = vertical[name]
    for name in VERTICAL_SCALARS[:5]:
        setattr(state, name, np.float32(vertical[name].item()))
    state.p_top = np.float32(p_top.item())
    _launch("sfire_ideal_state_coordinate_drops", cfg.nz,
        (state.c3f, state.c4f, state.dc3f, state.dc4f, np.int32(cfg.nz)))
    # The native base is stored as binary32; its residual is exactly zero.
    state.dphb_resid.fill(0)
    state._phb_host = cp.asnumpy(state.phb).astype(np.float64)
    spacing = cp.empty((ny, nx), cp.float32)
    _launch("sfire_ideal_column_spacing", nx * ny, (state.phb, spacing, np.int32(nx*ny), np.int32(cfg.nz)))
    state._dz_min = float(cp.min(spacing).item())
    table = FuelTable.from_namelist(cfg.fire_fuel_namelist) if cfg.fire_fuel_namelist else FuelTable()
    fuel = supplied("FC", fine_shape)[:fy, :fx].copy() if cfg.fire_fuel_read == 2 else cp.zeros((fy, fx), cp.float32)
    if cfg.fire_fmc_read == 2:
        fmc = supplied("FMC_G", fine_shape)[:fy, :fx].copy()
    else:
        fmc = cp.full((fy, fx), table.scalars["fuelmc_g"], cp.float32)
    xfine, yfine = coordinates((fy, fx), fdx, fdy)
    fire = dict(NFUEL_CAT=fuel, FMC_G=fmc, ZSF=zsf, DZDXF=gx, DZDYF=gy,
                FXLONG=xfine, FXLAT=yfine, FIRE_COORDINATE_MODE="metric", MAP_PROJ=0)
    if cfg.sfc_full_init:
        tsk = supplied("TSK", native_shape)[:ny, :nx].copy() if cfg.fire_read_tsk else cp.full((ny,nx), cfg.sfc_tsk, cp.float32)
        tmn = supplied("TMN", native_shape)[:ny, :nx].copy() if cfg.fire_read_tmn else cp.full((ny,nx), cfg.sfc_tmn, cp.float32)
        lu = supplied("LU", native_shape)[:ny, :nx].copy() if cfg.fire_read_lu else cp.full((ny,nx), cfg.sfc_lu_index, cp.float32)
        soil_fields = soil(tsk, tmn, scheme=cfg.sf_surface_physics, layers=cfg.num_soil_layers)
    else:
        tsk, tmn = atmosphere["TSK"], atmosphere["TMN"]
        lu = cp.zeros((ny,nx),cp.float32)
        soil_fields = dict(tslb=cp.zeros((cfg.num_soil_layers,ny,nx),cp.float32),smois=cp.zeros((cfg.num_soil_layers,ny,nx),cp.float32))
    surface = dict(tsk=tsk, tmn=tmn, soil_temperature=soil_fields["tslb"], soil_moisture=soil_fields["smois"],
        glw=0., swdown=0.,
        landmask=1., xland=1., landuse_dataset="USGS", ivgtyp=cfg.sfc_ivgtyp if cfg.sfc_full_init and cfg.sf_surface_physics==2 else 0,
        isltyp=cfg.sfc_isltyp if cfg.sfc_full_init and cfg.sf_surface_physics==2 else 0,
        vegfra=cfg.sfc_vegfra if cfg.sfc_full_init and cfg.sf_surface_physics==2 else 0.,
        canwat=cfg.sfc_canwat if cfg.sfc_full_init and cfg.sf_surface_physics==2 else 0., xice=0., snow=0.)
    if cfg.sfc_full_init:
        from woof.core.landuse import LanduseInitialization
        table_values = fields.get("_LANDUSE_TABLE")
        table_metadata = fields.get("_LANDUSE_METADATA", {})
        if table_values is None:
            from woof.static.sfire import read_landuse_table
            from woof.core.noah import TBL_DIR
            table_values, table_metadata = read_landuse_table(TBL_DIR / "LANDUSE.TBL", "USGS")
        values = initialize_native_landuse(lu, table_values, julday=int(fields.get("_IDEAL_JULDAY",1)),
            cen_lat=40.,
            iswater=0, isice=24, nodata_category=cfg.sfc_lu_index)
        zeros = cp.zeros((ny,nx),cp.float32)
        soil_categories = cp.full((ny,nx), surface["isltyp"], cp.int32)
        surface["landuse"] = LanduseInitialization(season=values["season"],landmask=values["landmask"],
            xland=values["xland"],lakemask=zeros,ivgtyp=values["ivgtyp"],isltyp=soil_categories,
            snowc=zeros,pblh=zeros,ust=cp.full((ny,nx),1.e-4,cp.float32),mavail=values["mavail"],
            z0=values["z0"],znt=values["znt"],albbck=values["albbck"],albedo=values["albedo"],
            embck=values["embck"],emiss=values["emiss"])
        surface["xland"] = values["xland"]
    xatm, yatm = coordinates((ny,nx),cfg.dx,cfg.dy)
    static_history = dict(XLONG=xatm, XLAT=yatm, LU_INDEX=lu)
    receipt = dict(source="WRF v4.7.1 dyn_em/module_initialize_fire.F", coordinate_mode="ideal_metric",
        surface_initialized=bool(cfg.sfc_full_init), mountain_type=cfg.fire_mountain_type,
        hybrid_opt=cfg.hybrid_opt, terrain_interpolation="continued coarse edge strips",
        native_initial_surface_coordinates=dict(latitude=cfg.fire_lat_init, longitude=cfg.fire_lon_init),
        native_initial_vegetation_category=cfg.sfc_ivgtyp,
        native_overwrites=["single-domain atmospheric coordinates become metric cell centers",
                           "physics initialization derives vegetation category from LU_INDEX"],
        global_attrs=dict(MAP_PROJ=0,CEN_LAT=40.,CEN_LON=-105.,MOAD_CEN_LAT=0.,
                          STAND_LON=0.,TRUELAT1=0.,TRUELAT2=0.,POLE_LAT=90.,POLE_LON=0.),
        initial_sky=dict(glw_wm2=0.,swdown_wm2=0.,source="native allocated fields and physics_init GLW=0 cold start"),
        landuse_table=table_metadata if cfg.sfc_full_init else None,
        corrections=["column-specific skin temperature", "nonoverlapping RUC soil thickness",
                     "initialized terrain interpolation edges", "zero ideal land-use category uses the configured fallback"])
    return state, fire, surface, static_history, receipt
