"""One contract for MPAS Noah history, DA carry and warm initialization.

Arrays here are cell-major, without Time. Admission refuses bad state; it never
repairs or clips it. Porosity follows the run's land model: for Noah (4 soil
levels) the STAS MAXSMC used by the pinned Noah driver, including its urban,
glacial and water-class branches; for RUC (sf_ruc, 9 soil levels) the STAS-RUC
MAXSMC bound of RUC's soil-fraction mix, with the same water/ice branches.
"""
from dataclasses import dataclass
import json
import numpy as np


@dataclass(frozen=True)
class LandField:
    name: str
    aliases: tuple[str, ...] = ()
    lower: float | None = None
    upper: float | None = None
    like: str = 'skintemp'
    units: str = '1'
    required: bool = False
    role: str = "surface_state"
    signed_diagnostic_soil_levels: tuple[int, ...] = ()

    @property
    def sources(self):
        return (self.name,) + self.aliases


# Core Noah memory is mandatory for a soil-bearing warm background, including
# slots absent from old cold-init streams. Other rows are carried when published;
# a slot already in the template may never fall back to its init-hour value.
LAND_FIELDS = (
    LandField('smois', lower=0, upper=1, like='tslb', units='m3 m-3', required=True),
    LandField('sh2o', lower=0, upper=1, like='tslb', units='m3 m-3', required=True),
    LandField('tslb', lower=200, upper=340, like='tslb', units='K', required=True),
    LandField('skintemp', ('tsk',), 150, 360, units='K', required=True),
    LandField('canwat', lower=0, units='kg m-2', required=True),
    LandField('snow', lower=0, units='kg m-2', required=True),
    LandField('snowh', lower=0, units='m', required=True),
    LandField('snowc', lower=0, upper=1, required=True),
    LandField('snotime', lower=0, units='s', required=True),
    LandField('t2m', ('t2',), 150, 360, units='K'),
    # Native atmphys_sfc_diagnostics.F:81-85 overwrites MYNN q2 with
    # qsfc - qfx/(rho*cqs2), which can be negative for valid flux inputs.
    # Preserve this diagnostic verbatim for DA, with a finite-value check;
    # it is not a Noah water reservoir and cannot use reservoir bounds.
    LandField('q2', units='kg kg-1', role='diagnostic'),
    LandField('tmn', lower=200, upper=340, units='K'),
    LandField('snoalb', lower=0, upper=1),
    LandField('sfc_albedo', ('albedo',), lower=0, upper=1),
    LandField('sfc_albbck', ('albbck',), lower=0, upper=1),
    LandField('sfc_emiss', lower=0, upper=1),
    LandField('sfc_emibck', lower=0, upper=1),
    LandField('znt', lower=0, units='m'), LandField('z0', lower=0, units='m'),
    # Noah diagnoses Q1=Q2+ETA_KINEMATIC*CP/RCH (signed dew flux),
    # then QSFC=Q1/(1-Q1); native noahlsm:805 and noahdrv:1283.
    LandField('qsfc', units='kg kg-1', role='diagnostic'),
    LandField('sfcrunoff', lower=0, units='mm'), LandField('udrunoff', lower=0, units='mm'),
    # Native RUC land_snowtemp L4334-4344 evaluates
    # snwepr/delt - (snwepr/(epot*ras*delt))*epot*ras, then GOTO 88
    # skips MAX(0,smelt). L3249 accumulates that signed cancellation
    # residual in SNOM/ACSNOM. This is a derived diagnostic, not a snow
    # reservoir. Preserve its finite native words on nine-level RUC;
    # other producers retain their nonnegative accumulator contract.
    LandField('acsnom', lower=0, units='mm', role='diagnostic',
              signed_diagnostic_soil_levels=(9,)),
    LandField('acsnow', lower=0, units='mm'),
    LandField('snopcx', units='kg m-2'), LandField('potevp', units='m'),
    LandField('hfx', units='W m-2'), LandField('qfx', units='kg m-2 s-1'),
    LandField('lh', units='W m-2'), LandField('grdflx', units='W m-2'),
    LandField('smstav'), LandField('smstot', units='mm'),
    LandField('chs'), LandField('chs2'), LandField('cqs2'), LandField('chklowq'),
    LandField('lai', lower=0), LandField('smcrel', like='tslb'), LandField('noahres'),
)
# Preserve the ordinary forecast's default history surface. The extended state
# inventory below is published only when the native background is requested.
DEFAULT_HISTORY_FIELDS = tuple(f.name for f in LAND_FIELDS)
LAND_FIELDS += (
    # Native RUC warm memory. Cycling skips the cold first-call assignments
    # (module_ruc_land.F90:587-650), so allocation zeros are not a warm state:
    # in particular soilt1 feeds the surface balance's temperature solve.
    LandField('qcg', units='kg kg-1'), LandField('qsg', units='kg kg-1'),
    LandField('qvg', units='kg kg-1'), LandField('soilt1', lower=170, upper=400, units='K'),
    LandField('tsnav', units='degC'), LandField('dew', units='m'),
    LandField('mavail'), LandField('flqc'), LandField('flhc'),
    LandField('smfr3d', like='tslb'), LandField('keepfr3d', like='tslb'),
    LandField('acrunoff', units='mm'), LandField('sfcexc'), LandField('sfcevp'),
    LandField('rhosnf', units='kg m-3'), LandField('precipfr'),
    LandField('snowfallac', units='m'),
)
LAND_BY_NAME = {f.name: f for f in LAND_FIELDS}
LAND_SOURCES = tuple((f.name, f.sources) for f in LAND_FIELDS)
LAND_ATTRIBUTE = 'hex_land_state_fields_v1'
# SOILPARM.TBL STAS, MAXSMC column, rows 1..19. float32 matches Noah.
# Source and public-domain notice: NOAH-TABLE-NOTICE.txt.
STAS_POROSITY = np.array([np.nan, .339, .421, .434, .476, .484, .439, .404,
                         .464, .465, .406, .468, .468, .439, 1., .20, .421,
                         .468, .20, .339], dtype=np.float32)
# SOILPARM.TBL STAS-RUC, MAXSMC column, rows 1..19 (same source; identical in the MPAS-A v8.4.2 / RRFS
# 44c0849 physics table, sha256 1e2275a3...). RUC runs 9 soil levels; Noah and Noah-MP run 4.
STAS_RUC_POROSITY = np.array([np.nan, .395, .410, .435, .485, .485, .451, .420,
                             .477, .451, .426, .492, .482, .451, 1., .200, .435,
                             .468, .200, .339], dtype=np.float32)
RUC_SOIL_LEVELS = 9
# Noah-MP's glacier arm explicitly publishes XLAIXY=PLAI=undefined_value
# (module_sf_noahmpdrv.F; woof/core/noahmp_runtime.py _glacier_lsm_step).
# This is live model memory, not a missing land reservoir or a value to clip.
NOAHMP_GLACIER_LAI_UNDEFINED = np.float32(-1e36)
RUC_WARM_MEMORY = frozenset(('qcg', 'qsg', 'qvg', 'soilt1', 'tsnav', 'dew',
                             'mavail', 'flqc', 'flhc', 'smfr3d'))
# RUC (module_sf_ruclsm.F soilprop, mosaic_soil=1) saturates a cell at the soil-fraction-weighted MAXSMC over
# its land classes (water, class 14, excluded), so no per-class row bounds it: hrrrv5 histories carried smois
# 0.4387 on sandy loam (row .435), 0.4196 on loamy sand (row .410) and 0.451 on urban cells (Noah cap .45).
# The bound is the largest land-class row (.492), widened 1e-6 for the float32 sums (0.43500003 measured on a
# pure .435 cell). Breakage it prevents: the Noah-only bound refused every RUC background (hrrrv5 state analysis arms,
# eight-card system 2026-10-29: smois 0.3398 > .339 on class 1), so no RRFS v2 state analysis cycle could compose a background.
RUC_SOIL_POROSITY = np.float32(np.max(STAS_RUC_POROSITY[1:][STAS_RUC_POROSITY[1:] < 1]) * np.float32(1 + 1e-6))


def read(var):
    return np.asarray(var[0] if var.dimensions[:1] == ('Time',) else var[...])


def require_land(template_vars, history_vars, where='history'):
    soil = any(n in template_vars for n in ('smois', 'sh2o', 'tslb'))
    variable = template_vars.get('tslb') if hasattr(template_vars, 'get') else None
    ruc = (getattr(variable, 'dimensions', ())[-1:] == ('nSoilLevels',)
           and getattr(variable, 'shape', ())[-1:] == (RUC_SOIL_LEVELS,))
    for f in LAND_FIELDS:
        if (f.name in template_vars or (soil and f.required) or (ruc and f.name in RUC_WARM_MEMORY)) and not any(n in history_vars for n in f.sources):
            raise ValueError(f'{where} lacks {f.name} (aliases {f.sources}); refusing init-hour land fallback')


def context(template):
    names = ('isltyp', 'ivgtyp', 'xland', 'xice', 'isice', 'mminlu')
    result = {n: read(template.variables[n]) for n in names if n in template.variables}
    if 'isice_lu' in template.variables:
        result['isice'] = read(template.variables['isice_lu'])
    return result


def porosity(ctx, ncells, nsoil=None):
    """Per-cell smois upper bound; nsoil (soil levels of the state) selects the land model: 9 is RUC, else Noah."""
    required = ('isltyp', 'ivgtyp', 'xland', 'xice', 'isice', 'mminlu')
    missing = set(required) - set(ctx)
    if missing:
        raise ValueError(f'land bounds need soil/land classification: missing {sorted(missing)}')
    vectors = {}
    for name in required[:4]:
        a = np.asarray(ctx[name])
        if a.shape != (ncells,) or not np.isfinite(a).all():
            raise ValueError(f'{name}: invalid cell classification shape/value')
        vectors[name] = a
    soil, veg, land, ice = (vectors[n] for n in required[:4])
    if np.any(soil != np.floor(soil)) or np.any((soil < 1) | (soil >= len(STAS_POROSITY))):
        raise ValueError('isltyp: expected STAS soil classes 1..19')
    if np.any((land != 1) & (land != 2)) or np.any((ice < 0) | (ice > 1)):
        raise ValueError('invalid xland/xice classification')
    text = np.asarray(ctx['mminlu']).tobytes().decode('ascii').strip(' \x00')
    from woof.hex.mpasphys.noah_v842 import LANDUSE_URBAN_INDEX
    if text not in LANDUSE_URBAN_INDEX:
        raise ValueError(f'unknown Noah land classification {text!r}')
    isice = np.asarray(ctx['isice'])
    if isice.size != 1:
        raise ValueError('isice: expected scalar land-ice class')
    if nsoil == RUC_SOIL_LEVELS:
        cap = np.full(ncells, RUC_SOIL_POROSITY, dtype=np.float32)
        # A cell whose soil fractions hold no land class falls back to its dominant row, MAXSMC(14) = 1 (soilprop).
        cap[soil == 14] = 1
    else:
        effective = soil.astype(int).copy()
        effective[(effective == 14) & (ice == 0)] = 7  # module_sf_noahdrv.F:919-924
        cap = STAS_POROSITY[effective].copy()
        urban = (veg == LANDUSE_URBAN_INDEX[text])
        if np.any(veg >= 31):
            raise ValueError('LCZ land classes require an explicit porosity contract')
        cap[urban] = np.float32(.45)  # module_sf_noahlsm.F:440
    # real.exe merges the lake class into water before IVGTYP is set (module_initialize_real.F:2859-2872;
    # woof.core.landuse._derive_categories), so Noah-MP and RUC integrate a lake cell as water and writes the water
    # carrier SMOIS=1 there, while the hex init keeps ivgtyp=ISLAKE and xland=1. Breakage: every ArWen
    # Noah-MP background refused "soil bounds cell=288 layer=0: sh2o=0.189 smois=1.0 porosity=0.476"
    # (a Kansas lake cell, 2026-11-03; under RUC the same cell, porosity 0.492). ISLAKE from WRF LANDUSE.TBL.
    islake = {"MODIFIED_IGBP_MODIS_NOAH": 21, "USGS": 28}.get(text)
    if islake is not None:
        cap[veg == islake] = 1
    # The glacier/sea-ice branch writes SMC=SH2O=1; water is not a soil column.
    cap[(land == 2) | (veg == isice.item()) | (ice >= float(ctx.get('xice_threshold', .02)))] = 1
    return cap


def validate_land(fields, ctx, *, require_core=False):
    if require_core:
        missing = {f.name for f in LAND_FIELDS if f.required} - set(fields)
        if missing:
            raise ValueError(f'incomplete warm land state: {sorted(missing)}')
    soil_temperature = np.asarray(fields.get('tslb', ()))
    nsoil = soil_temperature.shape[1] if soil_temperature.ndim == 2 else None
    for name, arr in fields.items():
        if name not in LAND_BY_NAME:
            continue
        a = np.asarray(arr)
        f = LAND_BY_NAME[name]
        lower = None if nsoil in f.signed_diagnostic_soil_levels else f.lower
        bad = ~np.isfinite(a) | (np.abs(a) >= 1e30)
        if lower is not None:   # a rounded zero (acsnom -3.9e-17 after a cycle) is not a bad land state
            bad |= a < lower - 1e-9
        if f.upper is not None:
            bad |= a > f.upper
        if name == 'lai' and np.any(a == NOAHMP_GLACIER_LAI_UNDEFINED):
            if a.ndim != 1:
                raise ValueError('lai: expected cell-major surface state')
            # Validate the complete classification before granting the one
            # native sentinel exception. Never infer a glacier from LAI.
            porosity(ctx, a.size)
            isice = np.asarray(ctx['isice']).item()
            threshold = float(ctx.get('xice_threshold', .02))
            if (not np.isfinite(isice) or isice != np.floor(isice) or isice < 1
                    or not np.isfinite(threshold) or not 0 <= threshold <= 1):
                raise ValueError('lai: invalid glacier classification')
            glacier = ((np.asarray(ctx['ivgtyp']) == isice)
                       & (np.asarray(ctx['xland']) == 1)
                       & (np.asarray(ctx['xice']) < threshold))
            bad &= ~((a == NOAHMP_GLACIER_LAI_UNDEFINED) & glacier)
        if np.any(bad):
            idx = tuple(int(i) for i in np.argwhere(bad)[0])
            raise ValueError(f'{name}{idx}={a[idx]} outside finite bounds [{lower}, {f.upper}]')
    if 'smois' in fields or 'sh2o' in fields:
        if not {'smois', 'sh2o', 'tslb'} <= set(fields):
            raise ValueError('soil bounds require smois, sh2o and tslb together')
        sm, sw, st = (np.asarray(fields[n]) for n in ('smois', 'sh2o', 'tslb'))
        if sm.ndim != 2 or sm.shape != sw.shape or sm.shape != st.shape:
            raise ValueError('soil fields need identical (nCells, nSoilLevels) shape')
        cap = porosity(ctx, sm.shape[0], sm.shape[1])[:, None]
        liquid_exceeds_total = sw > sm
        if sm.shape[1] == 4:
            # Noah-MP skips soil physics for XICE >= XICE_THRESHOLD and
            # explicitly writes SH2O=1 there every step (:715-725). SMC is
            # set to one only for full ice at ITIMESTEP=1 (:689-711), so
            # partial ice can retain the cold MAXSMC. These are non-soil
            # carriers, not liquid fractions of that retained SMC. Admit
            # only the exact prescribed SH2O, never arbitrary bad water.
            threshold = float(ctx.get('xice_threshold', .02))
            if not np.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError('soil bounds: invalid sea-ice threshold')
            prescribed_ice = ((np.asarray(ctx['xice']) >= threshold)[:, None]
                              & (sw == np.float32(1)))
            liquid_exceeds_total &= ~prescribed_ice
        bad = liquid_exceeds_total | (sm > cap)
        if np.any(bad):
            i, j = (int(v) for v in np.argwhere(bad)[0])
            raise ValueError(f'soil bounds cell={i} layer={j}: sh2o={sw[i,j]} smois={sm[i,j]} porosity={cap[i,0]}')


def collect_land(history, template, cell_map=None):
    """Preflight every field before any writes, including aliases and dimensions."""
    require_land(template.variables, history.variables)
    fields, pairs, dtypes = {}, [], {}
    for f in LAND_FIELDS:
        source = next((n for n in f.sources if n in history.variables), None)
        if source is None:
            continue
        if f.name not in template.variables and f.like not in template.variables:
            continue  # atmosphere-only template
        v = history[source]
        dims = v.dimensions[1:] if v.dimensions[:1] == ('Time',) else v.dimensions
        expected = ('nCells', 'nSoilLevels') if f.like == 'tslb' else ('nCells',)
        if tuple(dims) != expected:
            raise ValueError(f'{source}: land layout {dims} differs from {expected}')
        values = read(v)
        if cell_map is not None:
            values = values[cell_map]
        target = template[f.name] if f.name in template.variables else template[f.like]
        shape = target.shape[1:] if target.dimensions[:1] == ('Time',) else target.shape
        if values.shape != shape:
            raise ValueError(f'{source}: land shape {values.shape} differs from {shape}')
        fields[f.name] = values
        dtypes[f.name] = target.dtype
        pairs.append((f.name, source))
    validate_land(fields, context(template), require_core='smois' in template.variables)
    fields = {n: a.astype(dtypes[n], copy=False) for n, a in fields.items()}
    validate_land(fields, context(template), require_core='smois' in template.variables)
    return fields, pairs


def marker(fields):
    return json.dumps(sorted(set(fields) & set(LAND_BY_NAME)))


def define_land(path, names):
    """Add old-init missing surface slots in one classic-NetCDF define pass."""
    import ctypes as ct
    import netCDF4
    from netCDF4 import Dataset
    with Dataset(path) as ds:
        missing = [(n, LAND_BY_NAME[n].like) for n in names if n not in ds.variables]
    if not missing:
        return
    lib = ct.CDLL(netCDF4._netCDF4.__file__)
    lib.nc_strerror.restype = ct.c_char_p
    def check(code):
        if code:
            raise RuntimeError(lib.nc_strerror(code).decode())
    ncid = ct.c_int()
    check(lib.nc_open(str(path).encode(), 1, ct.byref(ncid)))
    try:
        check(lib.nc_redef(ncid))
        for name, like in missing:
            vid, xtype, ndims = ct.c_int(), ct.c_int(), ct.c_int()
            check(lib.nc_inq_varid(ncid, like.encode(), ct.byref(vid)))
            check(lib.nc_inq_vartype(ncid, vid, ct.byref(xtype)))
            check(lib.nc_inq_varndims(ncid, vid, ct.byref(ndims)))
            dims = (ct.c_int * ndims.value)()
            check(lib.nc_inq_vardimid(ncid, vid, dims))
            new = ct.c_int()
            check(lib.nc_def_var(ncid, name.encode(), xtype, ndims, dims, ct.byref(new)))
            units = LAND_BY_NAME[name].units.encode()
            check(lib.nc_put_att_text(ncid, new, b'units', ct.c_size_t(len(units)), units))
        check(lib.nc_enddef(ncid))
    finally:
        check(lib.nc_close(ncid))


def write_land(path, fields):
    from netCDF4 import Dataset
    define_land(path, fields)
    with Dataset(path, 'r+') as ds:
        ds.set_auto_maskandscale(False)
        for name, values in fields.items():
            v = ds[name]
            if v.dimensions[:1] == ('Time',):
                v[0] = values
            else:
                v[...] = values
        if fields:
            ds.setncattr(LAND_ATTRIBUTE, marker(fields))


def warm_land_fields(sources, *, xice_threshold=.02):
    """Validate marked warm inputs before the Rust pool reader can fill defaults."""
    from netCDF4 import Dataset
    marked = None
    for path in sources:
        # Native physics tables can also occur in sources. Inspect only NetCDF
        # files, using their magic bytes rather than assuming a suffix.
        with open(path, 'rb') as stream:
            magic = stream.read(8)
        if not (magic.startswith(b'CDF') or magic == b'\x89HDF\r\n\x1a\n'):
            continue
        with Dataset(path) as ds:
            if LAND_ATTRIBUTE not in ds.ncattrs():
                continue
            ds.set_auto_maskandscale(False)
            names = json.loads(ds.getncattr(LAND_ATTRIBUTE))
            if not isinstance(names, list) or len(set(names)) != len(names) or set(names) - set(LAND_BY_NAME):
                raise ValueError(f'{path}: invalid warm land field manifest')
            if set(names) - set(ds.variables):
                raise ValueError(f'{path}: warm land manifest names absent fields')
            fields = {n: read(ds[n]) for n in names}
            if 'tslb' in fields and fields['tslb'].shape[-1:] == (RUC_SOIL_LEVELS,):
                missing = RUC_WARM_MEMORY - set(fields)
                if missing:
                    raise ValueError(f'{path}: incomplete native RUC warm memory: {sorted(missing)}')
            validate_land(fields, dict(context(ds), xice_threshold=xice_threshold), require_core='smois' in ds.variables)
            if marked is not None:
                raise ValueError('multiple marked warm land sources: ambiguous member state')
            marked = tuple(names)
    return marked or ()


def initialize_preserving_land(session):
    """Warm carry survives Noah/landuse initialization; cold init is unchanged."""
    names = set(getattr(session.layout, 'land_state_fields', ()))
    rows = {f['name']: f for f in session.layout.fields if f['name'] in names}
    if set(rows) != names:
        raise ValueError(f'warm land fields unavailable in physics layout: {sorted(names-set(rows))}')
    saved = {(r['pool'], r.get('pool_name', n)): session.field(r['pool'], r.get('pool_name', n)).copy()
             for n, r in rows.items()}
    session.driver.initialize(session)
    for (pool, name), value in saved.items():
        session.put(pool, name, value)


def land_receipt(fields):
    """Small durable receipt; call only after admission succeeded."""
    return {'schema': 'hex-land-carry/v1', 'bounds': 'passed; no clipping',
            'fields': {n: {'shape': list(a.shape), 'min': float(np.min(a)), 'max': float(np.max(a))}
                       for n, a in fields.items() if n in LAND_BY_NAME}}


# These are disabled in Registry defaults. Their private running sums and timed
# input clocks are outside Noah's public fields, so never claim a warm restart.
UNSUPPORTED_WARM_SURFACE = {
    'config_sst_update': 'timed SST/ice input state',
    'config_sstdiurn_update': 'diurnal SST accumulators',
    'config_deepsoiltemp_update': 'deep-soil annual/daily accumulators',
}


def admit_warm_configuration(configs):
    active = {n: reason for n, reason in UNSUPPORTED_WARM_SURFACE.items() if configs.get(n, False)}
    if active:
        raise ValueError(f'warm land carry lacks optional surface-service state: {active}')
