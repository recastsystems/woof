"""Pure CuPy launchers for pinned WRF-Chem GOCART volume optics.

Every launcher covers k=0..nz-1. No state or radiation adapter is owned here.
Species fields are float32 contiguous (nz,ny,nx). Tables and REAL parameters
are float32. WRF REAL*8 bin edges stay float64 in JSON provenance; their
first-call section fractions were computed by the Fortran oracle.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import numpy as np

#: ``woof/data``-relative directory of the WRF-Chem Mie tables.  It ships in
#: the recast-woof-data companion since 2.8.7 (woof.data_assets.COMPANION_TREES):
#: the woof manylinux wheel crossed PyPI's 100,000,000-byte limit, and these
#: tables are pure data.  Resolved at first table read, not at import.
DATA_RELATIVE = 'chem/optics'


def data_dir() -> Path:
    """Where the optics tables resolve (the companion, via data_path)."""
    from woof import data_assets
    return data_assets.data_path(DATA_RELATIVE)

#: MANIFEST.txt kinds of the Fortran-written tables (little-endian words).
_KINDS = {'f4': np.dtype('<f4'), 'f8': np.dtype('<f8'), 'i4': np.dtype('<i4')}


@lru_cache(maxsize=1)
def _read_tables():
    """The shipped WRF Mie tables, digest-checked, in Fortran index order
    (MANIFEST.txt: name, kind, rank, extents).  Read here rather than through
    woof.verify, which a preprocessing install does not carry."""
    tables=data_dir()
    for line in (tables/'table-sha256sums.txt').read_text().splitlines():
        digest,name=line.split(maxsplit=1)
        if hashlib.sha256((tables/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'optics table digest differs: {name}')
    out={}
    for line in (tables/'MANIFEST.txt').read_text().splitlines():
        parts=line.split()
        if not parts:continue
        name,kind,rank=parts[0],parts[1],int(parts[2])
        shape=tuple(int(v) for v in parts[3:3+rank])
        dtype=_KINDS[kind]
        raw=np.fromfile(tables/f'{name}.bin',dtype=dtype)
        if raw.size!=(int(np.prod(shape)) if shape else 1):
            raise ValueError(f'optics table {name}: {raw.size} values, manifest says {shape}')
        native=dtype.newbyteorder('=')
        out[name]=raw.reshape(shape,order='F').astype(native) if shape else raw.astype(native)[0]
    return out

@lru_cache(maxsize=None)
def _device_tables(device):
    import cupy as cp
    t=_read_tables()
    metadata=json.loads((data_dir()/'wrfchem_components.json').read_text())
    c=metadata['components']
    def val(key,field):return c[key][field]['value']
    packed=np.zeros((20,11,2450),dtype=np.float32)
    for b in range(20):
        if b<4:
            names=['extpsw','ascatpsw','asmpsw','sbackpsw']+[f'pmom{i}psw' for i in range(2,8)]
            for j,name in enumerate(names):packed[b,j]=t[name][:,:,:,b].ravel(order='F')
        else:packed[b,0]=t['absplw'][:,:,:,b-4].ravel(order='F')
    grids=np.empty((20,2,7),dtype=np.float32)
    for b in range(20):
        wave='sw' if b<4 else 'lw';j=b if b<4 else b-4
        grids[b,0]=t['refrtab'+wave][:,j];grids[b,1]=t['refitab'+wave][:,j]
    # Complex volume sums at 4053:4063 and 4115:4125, not BC's table-grid index.
    keys=['sulfate','ammonium','ammonium','other_inorganic','dust','organic_carbon',
          'black_carbon','sodium','chloride','msa','water']
    refr=np.empty((20,11,3),dtype=np.float32)
    for b in range(20):
        wave='sw' if b<4 else 'lw';j=b if b<4 else b-4
        for i,key in enumerate(keys):
            component=c[key]
            if 'refractive_index' in component:
                re,im=val(key,'refractive_index')
            else:
                suffix={'sulfate':'sulf','dust':'dust','organic_carbon':'oc','sodium':'seas','chloride':'seas','water':''}[key]
                rn=f'refrw{wave}' if key=='water' else f'refr{wave}_{suffix}'
                inn=f'refiw{wave}' if key=='water' else f'refi{wave}_{suffix}'
                re,im=val(key,rn)[j],val(key,inn)[j]
            refr[b,i]=re,im,val(key,'density_g_cm3')
    densities=[val(key,'density_g_cm3') for key in ['sulfate','ammonium','other_inorganic',
               'organic_carbon','black_carbon','sodium','chloride','dust','msa','water']]
    kappas=[val(key,'kappa') for key in ['sulfate','organic_carbon','ammonium','chloride',
            'sodium','msa','other_inorganic','black_carbon','dust']]
    modal=metadata['mode_constants']
    conversion=[modal['sulfate_conversion_molecular_mass']['value'],modal['dry_air_conversion_molecular_mass']['value'],
                val('sea_salt','sodium_mass_fraction')[0],val('sea_salt','chloride_mass_fraction')[0],
                val('sea_salt','sodium_mass_fraction')[1]]
    constants=np.array([t[name] for name in ['rmmin','rmmax','xrmin','xrmax','pie']],np.float32)
    bounds=np.array([3.077,2.500,2.150,1.942,1.626,1.299,1.242,.778,.625,.442,.345,.263,.200,3.846,
                     3.846,3.077,2.500,2.150,1.942,1.626,1.299,1.242,.778,.625,.442,.345,.263,12.195],np.float32)
    with cp.cuda.Device(device):
        return {**{name:cp.asarray(t[name]) for name in ['mass_i','mass_j','mass_c','diam_cm']},
                'tables':cp.asarray(packed),'grids':cp.asarray(grids),'constants':cp.asarray(constants),
                'p':cp.asarray(np.array(densities+kappas+[modal['rh_cap']['value']]+conversion,np.float32)),
                'ri':cp.asarray(refr),'bounds':cp.asarray(bounds)}

def load_mie_tables(device=None):
    """Verify the shipped data, upload once and cache on a CuPy device."""
    import cupy as cp
    return _device_tables(cp.cuda.Device().id if device is None else int(device))

def pack_rows(rows, fields):
    """Pack plain row dicts and matching device fields without testing names.

    optics=null means the species contributes nothing to WRF optics. Rows sort
    by optics.wrf_order, the registry order at registry.chem:4022. A modal row
    may emit a secondary component after the same conversion operations.
    """
    import cupy as cp
    if len(rows)!=len(fields):raise ValueError('one field is required per row')
    t=_read_tables()
    items=sorted([(r['optics'],f) for r,f in zip(rows,fields) if r.get('optics') is not None],
                 key=lambda v:v[0]['wrf_order'])
    orders=[o['wrf_order'] for o,f in items]
    if len(set(orders))!=len(orders):raise ValueError('optics.wrf_order must be unique')
    pointers=[];ip=[];rp=[];frac=[];owners=[]
    for o,f in items:
        if f.dtype!=cp.float32 or not f.flags.c_contiguous or f.ndim!=3:
            raise ValueError('species fields must be contiguous float32 (nz,ny,nx)')
        owners.append(f)
        treatment=o['size_treatment']
        if treatment=='modal':
            if o['conversion'] not in ('ppmv_sulfate','ug_kg'):
                raise ValueError('unsupported optics mixing-ratio conversion')
            targets=[(o['target'],o['mass_multiplier'])]
            if o.get('secondary_target') is not None:
                targets.append((o['secondary_target'],o['secondary_multiplier']))
            for target,mult in targets:
                if not isinstance(target,int) or not 0<=target<8:
                    raise ValueError('modal component target must be an integer in 0..7')
                pointers.append(f.data.ptr);ip.append([target,0,int(o['conversion']=='ppmv_sulfate')])
                rp.append([o['aitken_fraction'],mult]);frac.append(np.zeros(9,np.float32))
        elif treatment=='fixed_bin':
            # Distribution identity is a data key, not a species-name test.
            mode={'seasfrc_goc9bin':1,'dustfrc_goc9bin':2}[o['distribution_table']]
            if not isinstance(o['distribution_bin'],int) or not 0<=o['distribution_bin']<t[o['distribution_table']].shape[0]:
                raise ValueError('section distribution bin is out of range')
            pointers.append(f.data.ptr);ip.append([0,mode,0]);rp.append([0.,1.])
            frac.append(t[o['distribution_table']][o['distribution_bin'],:])
        else:raise ValueError(f'unsupported optics size treatment: {treatment}')
    return {'pointers':cp.asarray(np.asarray(pointers,dtype=np.uint64)),
            'ip':cp.asarray(np.asarray(ip,dtype=np.int32).reshape(-1,3)),
            'rp':cp.asarray(np.asarray(rp,dtype=np.float32).reshape(-1,2)),
            'frac':cp.asarray(np.asarray(frac,dtype=np.float32).reshape(-1,9)),
            'owners':owners,'nrow':len(pointers)}

def _field(a,shape=None):
    import cupy as cp
    if a.dtype!=cp.float32 or a.ndim!=3 or not a.flags.c_contiguous:
        raise ValueError('input must be contiguous float32 (nz,ny,nx)')
    if shape is not None and a.shape!=shape:raise ValueError('field shapes differ')
    if a.device.id!=cp.cuda.Device().id:raise ValueError('field must be on the active CuPy device')

#: Every optical output the kernel can store, with its band count.  The SW
#: quantities are the four wavelengths 300/400/600/999 nm, the LW ones the
#: sixteen RRTMG LW bands.
OUTPUTS = {'tauaer': 4, 'extaer': 4, 'waer': 4, 'gaer': 4, 'bscoef': 4,
           'tauaerlw': 16, 'extaerlw': 16}


def gocart_optics(rows,fields,alt,relhum,dz8w,*,outputs=None,out=None,
                  moments=True,opt_out_fields=True,diagnostics=False,
                  check_refindex=True):
    """All levels. No global workspace apart from the output fields.

    ``outputs`` names the optical fields to store (default every one in
    :data:`OUTPUTS`); an output not named is computed and discarded, never
    stored, so a caller that needs only tauaer holds four fields, not 77.
    ``out`` may hand preallocated contiguous float32 ``(bands,nz,ny,nx)``
    arrays for any of them.  Returns band-major SW/LW fields
    (band,nz,ny,nx) and, when ``moments``, the moments (6,4,nz,ny,nx).
    ``opt_out_fields`` adds EXTCOF55 and AOD5502D (it needs tauaer).
    With diagnostics=True, also return radius/number (9,nz,ny,nx), and
    refractive indices (20,9,2,nz,ny,nx). These add 378 float32 fields.
    WRF's fatal refractive-index diagnostics raise by default. An asynchronous
    caller may defer this check and inspect the returned invalid_refindex mask.
    """
    from woof.core.kernels import get_kernel
    import cupy as cp
    _field(alt);_field(relhum,alt.shape);_field(dz8w,alt.shape)
    for f in fields:_field(f,alt.shape)
    shape=alt.shape;nz,ny,nx=shape;ncol=ny*nx
    outputs=tuple(OUTPUTS) if outputs is None else tuple(outputs)
    unknown=set(outputs)-set(OUTPUTS)
    if unknown:raise ValueError(f'unknown optics outputs {sorted(unknown)}')
    if opt_out_fields and 'tauaer' not in outputs:
        raise ValueError('EXTCOF55 and AOD5502D are computed from tauaer')
    out=dict(out or {})
    stored={}
    for name in outputs:
        a=out.get(name)
        if a is None:a=cp.empty((OUTPUTS[name],*shape),cp.float32)
        elif (a.dtype!=cp.float32 or a.shape!=(OUTPUTS[name],*shape)
              or not a.flags.c_contiguous):
            raise ValueError(f'{name} must be contiguous float32 {(OUTPUTS[name],*shape)}')
        stored[name]=a
    r=pack_rows(rows,fields);t=_device_tables(cp.cuda.Device().id)
    mom=cp.empty((6,4,*shape),cp.float32) if moments else None
    rad=cp.empty((9,*shape),cp.float32) if diagnostics else None
    num=cp.empty_like(rad) if diagnostics else None
    ri=cp.empty((20,9,2,*shape),cp.float32) if diagnostics else None
    invalid=cp.empty((ny,nx),cp.int32)
    def ptr(a):return np.uint64(0) if a is None else a
    args=(r['pointers'],np.int32(r['nrow']),r['ip'],r['rp'],r['frac'],t['mass_i'],t['mass_j'],t['mass_c'],
          t['diam_cm'],t['p'],t['ri'],alt,relhum,dz8w,t['tables'],t['grids'],t['constants'],
          *(ptr(stored.get(n)) for n in OUTPUTS),ptr(mom),ptr(rad),ptr(num),ptr(ri),invalid,
          np.int32(nz),np.int32(ncol))
    get_kernel('chem_optics','chem_optics_gocart')(((ncol+31)//32,),(32,),args)
    if check_refindex and np.any(invalid.get()):
        raise ValueError('invalid aerosol refractive index in WRF Mie diagnostics')
    result=dict(stored)
    result['invalid_refindex']=invalid
    if moments:
        result['moments']=mom
        result.update({f'l{i+2}aer':mom[i] for i in range(6)})
    if diagnostics:result.update(radius=rad,number=num,refindex=ri)
    if opt_out_fields:
        result.update(opt_out(stored['tauaer'],dz8w,
                              ext=out.get('EXTCOF55'),aod=out.get('AOD5502D')))
    # Retain pointer owners and parameter arrays until the stream finishes.
    result['_rows']=r
    return result

def opt_out(tauaer,dz8w,*,ext=None,aod=None):
    """EXTCOF55 and its ordered all-level WOOF-ArWen column integral.

    ``ext`` (nz,ny,nx) and ``aod`` (ny,nx) may be preallocated float32."""
    from woof.core.kernels import get_kernel
    import cupy as cp
    _field(dz8w);nz,ny,nx=dz8w.shape;ncol=ny*nx
    if tauaer.shape!=(4,nz,ny,nx) or tauaer.dtype!=cp.float32 or not tauaer.flags.c_contiguous:
        raise ValueError('tauaer must be contiguous float32 (4,nz,ny,nx)')
    if ext is None:ext=cp.empty_like(dz8w)
    else:_field(ext,dz8w.shape)
    if aod is None:aod=cp.empty((ny,nx),cp.float32)
    elif aod.dtype!=cp.float32 or aod.shape!=(ny,nx) or not aod.flags.c_contiguous:
        raise ValueError('AOD5502D must be contiguous float32 (ny,nx)')
    get_kernel('chem_optics','chem_optics_opt_out')(((ncol+31)//32,),(32,),
        (tauaer,dz8w,ext,aod,np.int32(nz),np.int32(ncol)))
    return {'EXTCOF55':ext,'AOD5502D':aod}

def rrtmg_sw_bands(tauaer,waer,gaer,*,check_negative=True):
    """All levels; return tau/ssa/asm (nz,14,ny,nx). Defaults match WRF.

    Negative column tau is fatal in WRF. The default reads its device band
    mask and raises ValueError. An asynchronous adapter may use
    check_negative=False and check the returned negative_tau mask itself.
    """
    from woof.core.kernels import get_kernel
    import cupy as cp
    if tauaer.ndim!=4 or tauaer.shape[0]!=4:raise ValueError('expected four SW bands')
    for f in (tauaer,waer,gaer):
        if f.shape!=tauaer.shape or f.dtype!=cp.float32 or not f.flags.c_contiguous:
            raise ValueError('band fields must be contiguous float32 with equal shapes')
    _,nz,ny,nx=tauaer.shape;ncol=ny*nx;t=_device_tables(cp.cuda.Device().id)
    out={name:cp.empty((nz,14,ny,nx),cp.float32) for name in ['tau','ssa','asm']}
    out['negative_tau']=cp.empty((ny,nx),cp.int32)
    get_kernel('chem_optics','chem_optics_rrtmg_sw')(((ncol+31)//32,),(32,),
        (tauaer,waer,gaer,t['bounds'],out['tau'],out['ssa'],out['asm'],out['negative_tau'],np.int32(nz),np.int32(ncol)))
    if check_negative and np.any(out['negative_tau'].get()):
        raise ValueError('negative total optical depth in RRTMG SW conversion')
    return out


# ---------------------------------------------------------------------------
# The process (``optics.gocart``): the chem driver's view.
# ---------------------------------------------------------------------------

from woof.core.chem_context import ChemAllocation  # noqa: E402

KEY = "optics.gocart"
#: Optics reads the species and writes optical properties; it moves no mass,
#: so the bucket the driver measures around it closes at zero.
LEDGER = "coupling"
#: The kernel translation units step() launches (priced by the preflight).
KERNEL_MODULES = ("chem_optics",)
#: WRF's optical_driver takes rri (= alt), rh and dz8w from chem_prep
#: (chem/chem_driver.F:926-928).
REQUIRES = ("alt", "rh", "dz8w")
ALLOCATES = (
    ChemAllocation("extcof55", "3d", output_name="EXTCOF55", units="km-1",
                   description="aerosol extinction at 550 nm, WRF-Chem "
                               "aer_opt_out's EXTCOF55 "
                               "(chem/module_aer_opt_out.F:54-57)"),
    ChemAllocation("aod550", "2d", output_name="AOD5502D", units="1",
                   description="aerosol optical depth at 550 nm: the ordered "
                               "column integral of EXTCOF55 (WOOF-ArWen; WRF-Chem "
                               "writes no such field)"),
)


def rows(table):
    return table.rows_for(KEY)


def _stepphot(cfg, dt) -> int:
    """WRF's stepphot, ``max(nint(photdt*60/dt), 1)`` (chem/chemics_init.F:
    643-648), with photdt the run's radiation interval: GOCART-lite has no
    photolysis, and the optical properties exist for radiation and for the
    550 nm products, so they are computed on radiation's cadence."""
    from woof.config import effective_radt_minutes

    steps = float(effective_radt_minutes(cfg)) * 60.0 / float(dt)
    return max(int(np.floor(steps + 0.5)), 1)


def photolysis_step(cfg, ktau, dt) -> bool:
    """WRF's do_photstep for a fixed step (chem/chem_driver.F:349-365)."""
    step = _stepphot(cfg, dt)
    return ktau == 1 or step == 1 or ktau % step == 0


def _row_dicts(table):
    return [{"optics": dict(r.optics) if r.optics else None}
            for r in rows(table)]


def init(ctx):
    missing = [name for name in REQUIRES if not ctx.has(name)]
    if missing:
        raise ValueError(f"GOCART optics reads {missing}, which this run's "
                         "chem preparation does not produce")
    for r in rows(ctx.table):
        if r.optics and "wrf_order" not in r.optics:
            raise ValueError(f"row {r.name!r} names {KEY} with an optics "
                             "object that has no WRF volume-optics contract "
                             "(wrf_order, size_treatment, ...)")
    load_mie_tables()


def step(ctx, dt, ktau):
    if not photolysis_step(ctx.cfg, ktau, dt):
        return
    if getattr(ctx.cfg, "aer_ra_feedback", 0):
        # Radiation owns the optics clock when it consumes the result.
        return
    update_diagnostics(ctx)


def prepare_history(state):
    """Evaluate live optics at every frame, including the untouched IC.

    History can precede the first chemistry/radiation call, or fall between
    radiation calls. A cached radiation result therefore cannot supply it.
    The diagnostic needs meteorology and mass only, with no process clock.
    """
    from woof.core.chem_context import ChemContext
    from woof.core.chem_prep import ChemPrep
    from woof.core.chem_state import process_attr

    chem = state.chem
    prep = getattr(chem, "prep", None)
    if prep is None:
        prep = chem.prep = ChemPrep(state)
    else:
        prep.refresh(state)
    diag = {alloc.name: getattr(state, process_attr(alloc))
            for alloc in ALLOCATES}
    update_diagnostics(ChemContext(state, None, chem.table,
                                   prep=prep, diag=diag))


def update_diagnostics(ctx):
    """Live 550 nm products, also called before the first radiation solve."""
    import cupy as cp

    rs = rows(ctx.table)
    alt = ctx.met("alt")
    chem = ctx.state.chem
    work = getattr(chem, "gocart_optics_tauaer", None)
    if work is None or work.shape[1:] != alt.shape:
        # The four SW wavelengths' tau, the only stored optical output EXTCOF55
        # needs; held once per domain (16 B per cell).
        work = chem.gocart_optics_tauaer = cp.empty((4, *alt.shape),
                                                    cp.float32)
    gocart_optics(_row_dicts(ctx.table), [ctx.field(r) for r in rs], alt,
                  ctx.met("rh"), ctx.met("dz8w"), outputs=("tauaer",),
                  out={"tauaer": work, "EXTCOF55": ctx.diag["extcof55"],
                       "AOD5502D": ctx.diag["aod550"]},
                  moments=False)
