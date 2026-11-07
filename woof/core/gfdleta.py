"""WRF 4.6.1 Eta/GFDL radiation, with all numerical work on the device."""
from pathlib import Path
import json
import math
import numpy as np
from woof.core.kernels import get_kernel
from woof.config import DEFAULT_COLUMN_CHUNK

_ROOT=Path(__file__).parent
_LAYOUT=json.loads((_ROOT/'gfdleta_column_layout.json').read_text())
_INIT_LAYOUT=json.loads((_ROOT/'gfdleta_init_layout.json').read_text())
_OFFSETS={}
_table_count=0
for _name,(_kind,_shape) in _LAYOUT['tables'].items():
    _count=math.prod(int(v) for v in _shape.split(',')) if _shape else 1
    _OFFSETS[_name]=(_table_count,_count)
    _table_count+=_count

class GFDLTables:
    """Run-level tables built for the actual vertical grid, as GFDLETAINIT."""
    def __init__(self,sigma_full,sigma_half,p_top,*,co2tf=1,table_root=None,transmission=None,start_time=None):
        import cupy as cp
        from woof.table_assets import physics_table_root
        root=physics_table_root('gfdleta',table_root)
        n=int(sigma_half.size)
        if n<1 or n>81 or int(sigma_full.size)!=n+1:
            raise ValueError('GFDL ozone tables cover 1..81 layers; full sigma needs n+1 interfaces')
        if co2tf not in (0,1):
            raise ValueError('co2tf must be 0 (CONRAD) or 1 (CO2O3)')
        self.n=n
        self.words=cp.zeros(_table_count,cp.uint32)
        def read(name):
            # These NPY assets were decoded and rounded by the Rust table tool.
            return cp.asarray(np.load(root/name,allow_pickle=False),dtype=cp.float32)
        ozone=read('ozone-coefficients.npy')
        sf=cp.ascontiguousarray(cp.asarray(sigma_full,dtype=cp.float32)).copy()
        sh=cp.ascontiguousarray(cp.asarray(sigma_half,dtype=cp.float32)).copy()
        get_kernel('gfdleta_co2_init','gd_initialize_climatology')((30,),(256,),(
            self.words,ozone,sh,np.float32(p_top/1000.),np.int32(n)))
        if co2tf==1:
            raw=read('co2-raw.npy')
            work=cp.empty((8,_INIT_LAYOUT['scratch_bytes']),cp.uint8)
            state=cp.empty((8,_INIT_LAYOUT['state_bytes']),cp.uint8)
            private_tables=cp.empty((8,_table_count),cp.uint32)
            status=cp.empty(8,cp.int32)
            get_kernel('gfdleta_co2_init','gd_prepare_co2_profiles')(((8*106929+255)//256,),(256,),(raw,state))
            get_kernel('gfdleta_co2_init','gd_initialize_co2_profiles')((8,),(32,),(
                private_tables,sf,sh,np.float32(p_top/1000.),np.int32(n),work,state,status))
            get_kernel('gfdleta_co2_init','gd_merge_co2_profiles')((((n+1)*(n+1)+255)//256,),(256,),(self.words,work,np.int32(n)))
            # Initialization validates the asset/grid once, before stepping.
            if np.any(cp.asnumpy(status)):
                raise ValueError('GFDL CO2 interpolation produced a nonpositive logarithm argument')
        if co2tf==0:
            if transmission is None:
                file_levels=json.loads((root/'MANIFEST.json').read_text())['binary_levels']
                if type(file_levels) is not int or not 1<=file_levels<=81:
                    raise ValueError('CONRAD table metadata must declare an integer layer count within 1..81')
                if n!=file_levels:
                    raise ValueError(f'CONRAD co2_trans has {file_levels} layers, run has {n}; select co2tf=1 to compute transmission for this grid')
                transmission=np.load(root/f'co2-trans-{file_levels}.npy',allow_pickle=False)
            transmission=cp.ascontiguousarray(cp.asarray(transmission,dtype=cp.float32))
            expected=14*(n+1)+6*n+6*(n+1)*(n+1)
            if transmission.size!=expected:
                raise ValueError(f'CONRAD transmission shape is wrong for {n} layers: expected {expected} values')
            get_kernel('gfdleta_co2_init','gd_conrad')((27,),(256,),(self.words,transmission,np.int32(n)))
        # TABLE uses its published coefficients, not precomputed oracle outputs.
        coeff=read('table-coefficients.npy');aux=cp.empty(4,cp.float32)
        get_kernel('gfdleta_table_runtime','gd_table_rows_runtime')((1,),(32,),(
            coeff,self.view('EM1'),self.view('TABLE1'),self.view('EM3'),self.view('EM1WDE'),self.view('SOURCE'),aux))
        get_kernel('gfdleta_tables','gd_table_derivatives')((40,),(128,),(
            self.view('TABLE1'),self.view('SOURCE'),self.view('TABLE2'),self.view('TABLE3'),self.view('DSRCE')))
        for k,name in enumerate(['AB15WD','SKC1R','SKO3R','SKO2D']):self.view(name)[:]=aux[k:k+1]
        for name,source in [('T1','TABLE1'),('T2','TABLE2'),('T4','TABLE3'),('EM1V','EM1'),('EM1VW','EM1WDE'),('EM3V','EM3')]:self.view(name)[:]=self.view(source)
        x=cp.arange(1,311,dtype=cp.float32)*np.float32(.01)
        get_kernel('gfdleta_init','gd_gaussian')((3,),(128,),(x,self.view('AXSD'),np.int32(310)))
        self.solar_distance=None
        if start_time is not None:
            hour=start_time.hour+start_time.minute/60.+start_time.second/3600.
            rounded=int(np.floor(hour+.5))
            date=cp.asarray([start_time.year,start_time.month,start_time.day,rounded],dtype=cp.int32)
            self.solar_distance=cp.empty(1,cp.float32)
            get_kernel('gfdleta_init','gd_solar')((1,),(1,),(date,self.solar_distance,np.int32(1)))


    def view(self,name):
        off,count=_OFFSETS[name]
        return self.words[off:off+count].view(np.float32)

class GFDLEtaRadiation:
    """ETARA driver using WRF stepra/nphs, cloud state and table initialization."""
    publishes_cldfra=True
    above_atmosphere_policy='wrf-v4.6.1-gfdl-model-column'
    def __init__(self,start_time,latitude_deg,longitude_deg,*,p_top=None,co2tf=1,longwave=True,shortwave=True,column_chunk=None):
        import cupy as cp
        self.start_time=start_time
        self.latitude_deg=cp.ascontiguousarray(cp.asarray(latitude_deg,dtype=cp.float32))
        self.longitude_deg=cp.ascontiguousarray(cp.asarray(longitude_deg,dtype=cp.float32))
        self.p_top=None if p_top is None else float(p_top)
        self.co2tf=int(co2tf)
        self.longwave=longwave;self.shortwave=shortwave
        self.publishes_olr=longwave
        # Device workers reuse bounded scratch across the column grid.
        self.column_chunk=DEFAULT_COLUMN_CHUNK if column_chunk is None else int(column_chunk)
        if self.column_chunk<1:raise ValueError('GFDL column_chunk must be positive')
        self.tables=None
        self.update_count=0

    @staticmethod
    def retained_fields(nz,ny,nx):
        import cupy as cp
        from woof.gfdleta_contract import GFDLETA_REAL_FIELDS, GFDLETA_INTEGER_FIELDS
        fields={k:cp.zeros((ny,nx),cp.float32) for k in GFDLETA_REAL_FIELDS}
        fields['hbot'].fill(nz+1)
        fields.update({k:cp.zeros((ny,nx),cp.int32) for k in GFDLETA_INTEGER_FIELDS})
        return fields

    def __call__(self,*,atmosphere,fields,state,cfg):
        import cupy as cp
        from datetime import timedelta
        from woof.core.physics import RadiationResult,_physics_interval_steps
        from woof.core.mynn_radiation import wrf_itimestep
        from woof.core.swint import calendar_scalars
        nz,ny,nx=atmosphere['temperature'].shape
        if self.latitude_deg.shape!=(ny,nx):raise ValueError('GFDL latitude/longitude must match the model grid')
        if self.tables is None:
            self.tables=GFDLTables(state.znw,state.znu,float(state.p_top),co2tf=self.co2tf,start_time=self.start_time)
        ncol=ny*nx
        f=cp.zeros((ncol,_LAYOUT['float_count']),cp.float32)
        ints=cp.zeros((ncol,_LAYOUT['int_count']),cp.int32)
        descriptors={x['name']:x for x in _LAYOUT['fields']}
        def put(name,value):
            d=descriptors[name];a=f if d['kind']=='real' else ints
            target=a[:,d['offset']:d['offset']+d['count']]
            if np.isscalar(value):target[:]=value;return
            value=cp.asarray(value)
            if value.ndim==3:
                # ETARA owns WRF's vertical flip. Input remains bottom to top.
                cols=value.reshape(value.shape[0],ncol).T
                target[:,:cols.shape[1]]=cols
            else:target[:,0]=value.reshape(-1)
        for name,key in [('pi3d','exner'),('p8w','p_interface'),('dz8w','dz'),('rho_phy','rho'),('p_phy','pressure'),('t','temperature'),('qv','qv'),('qw','qc'),('qi','qi'),('qs','qs')]:put(name,atmosphere[key])
        for name,key,default in [('xland','xland',1.),('tsk2d','tsk',288.),('albedo','albedo',.2),('vegfra','shdfac',0.),('snow','snow',0.)]:put(name,fields.get(key,default))
        for name in 'htop hbot htopr hbotr cuppt acfrst acfrcv ncfrst ncfrcv'.split():put(name,fields[name])
        put('glw',fields['glw']);put('rswin',fields['swdown'])
        put('glat',self.latitude_deg*np.float32(.01745329238474369))
        put('glon',self.longitude_deg*np.float32(.01745329238474369))
        stepra=_physics_interval_steps(cfg.radt if cfg.radt>0 else cfg.radt_minutes,cfg.dt)
        nphs=_physics_interval_steps(cfg.bldt,cfg.dt)
        julian,xtime,gmt=calendar_scalars(self.start_time,state.elapsed_seconds)
        valid=self.start_time+timedelta(seconds=float(state.elapsed_seconds))
        scalar={'dt':float(cfg.dt),'g':np.float32(9.8100004196166992),'gmt':gmt,'xtime':xtime,'julian':julian,'nstepra':stepra,'nphs':nphs,'itimestep':wrf_itimestep(state.elapsed_seconds,cfg.dt),'julyr':valid.year,'julday':valid.timetuple().tm_yday}
        scalar.update({k:1 for k in 'ids jds kds ims jms kms its jts kts ime jme ite jte'.split()})
        scalar.update({'ide':2,'jde':2,'kde':nz+1,'kme':nz+1,'kte':nz})
        for name,value in scalar.items():put(name,value)
        chunk=min(ncol,self.column_chunk)
        work=cp.empty((chunk,_LAYOUT['scratch_bytes']),cp.uint8)
        kernel=get_kernel('gfdleta_column','gd_radiation_columns_pooled')
        kernel(((chunk+31)//32,),(32,),
               (self.tables.words,f,ints,work,np.int32(ncol),np.int32(chunk),np.int32(self.longwave),np.int32(self.shortwave)))
        def take(name):
            d=descriptors[name];a=f if d['kind']=='real' else ints
            if d['count']>1:return cp.ascontiguousarray(a[:,d['offset']:d['offset']+nz].T.reshape(nz,ny,nx))
            return cp.ascontiguousarray(a[:,d['offset']].reshape(ny,nx))
        for name in 'htop hbot htopr hbotr cuppt acfrst acfrcv ncfrst ncfrcv cfracl cfracm cfrach rswtoa rlwtoa czmean'.split():fields[name][...]=take(name)
        fields['swdownc'][...]=take('rswinc')
        self.update_count+=1
        zero=cp.zeros_like(atmosphere['temperature'])
        return RadiationResult(rthratenlw=take('thratenlw') if self.longwave else zero,rthratensw=take('thratensw') if self.shortwave else zero,swdown=take('rswin') if self.shortwave else fields['swdown'],glw=take('glw') if self.longwave else fields['glw'],gsw=take('gsw'),coszen=take('czmean'),olr=take('rlwtoa') if self.longwave else None,cldfra=take('cldfra'))

    @property
    def restart_identity(self):
        from woof.table_assets import physics_table_root
        manifest=json.loads((physics_table_root('gfdleta')/'MANIFEST.json').read_text())
        return {'algorithm':'wrf-v4.6.1-gfdleta','source':'module_ra_gfdleta.F:ETARA','co2tf':self.co2tf,'longwave':self.longwave,'shortwave':self.shortwave,'above_atmosphere_policy':self.above_atmosphere_policy,'table_assets':manifest['assets']}
