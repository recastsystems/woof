import ctypes as C,json,hashlib,sys,argparse
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser(description="Strict MPAS UGWP synthetic column replay (debugging, not forecast acceptance)")
parser.add_argument('--work',type=Path,required=True)
parser.add_argument('--receipt',type=Path,required=True)
parser.add_argument('--gpu',action='store_true')
parser.add_argument('--defect-probes',action='store_true',
                    help='also replay native zero time denominator and dead launch scratch read')
args=parser.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
nl=55; nc=24
F=np.float32
lat=np.array([-89,-60,-45,-41,-31,-30,-15.3,-3,0,3,15,15.3,16,30,31,41,45,60,75,89,10,-10,25,-25],dtype='f4')*F(np.pi/180)
zi=np.broadcast_to(np.linspace(0,44000,nl+1,dtype='f4')[:,None],(nl+1,nc)).copy()
z=((zi[1:]+zi[:-1])*F(.5)).astype('f4')
p=(F(101325)*np.exp(-z/F(7000))).astype('f4');pi=(F(101325)*np.exp(-zi/F(7000))).astype('f4')
t=np.maximum(F(190),F(288)-z*F(.004)).astype('f4')
k=np.arange(nl,dtype='f4')[:,None]
u=(F(10)+F(15)*np.sin(k*F(.2)+np.arange(nc,dtype='f4')[None,:])).astype('f4');v=(F(-5)+F(10)*np.cos(k*F(.23)+np.arange(nc,dtype='f4')[None,:])).astype('f4')
q=(F(.01)*np.exp(-z/F(2000))).astype('f4');ex=(p/F(1e5))**F(2/7)
rc=np.resize(np.array([0,.001,.01,.03,.09,1],dtype='f4'),nc);rn=rc*F(.5)
rdzw=np.full(nl,F(nl/44000),dtype='f4');dzu=np.full(nl,F(44000/nl),dtype='f4')
taulat=np.array([-90,-60,-30,0,30,60,90],dtype='f4');days=np.array([0,60,120,180,240,300,367],dtype='f4')
tau=(np.arange(49,dtype='f4').reshape(7,7)+F(1))*F(.0001)
inputs=[rdzw,dzu,lat,taulat,days,tau,u,v,t,q,p,ex,pi,z,zi,rc,rn]
def ptr(a):return a.ctypes.data_as(C.c_void_p)
lib=C.CDLL(str((args.work/'native/liboracle.so').resolve()));host=C.CDLL(str((args.work/'libhost.so').resolve()))
native=lib.oracle;native.restype=None;native.argtypes=[C.c_int,C.c_int,C.c_float,C.c_float,C.c_float]+[C.c_void_p]*6+[C.c_int,C.c_int]+[C.c_void_p]*24
native_modes=lib.oracle_modes;native_modes.restype=None;native_modes.argtypes=native.argtypes
# 11 dynamic + 3 tendency + 3 diag + dk + zh + state + j1 + j2 + w1 + w2 =25
export_native=lib.oracle_writeback;export_native.restype=None;export_native.argtypes=[C.c_int]*3+[C.c_void_p]*16
cpu=host.host_run;cpu.restype=None;cpu.argtypes=[C.c_int]*3+[C.c_float]*3+[C.c_void_p]*29
# 17 inputs + 8 outputs + state/index/weight/status =29
names=['rublten','rvblten','rthblten','dudt_ngw','dvdt_ngw','dtdt_ngw','dked','zngw','initial_state','latitude_indices','latitude_weights']
def arrays():
    out=[np.full((nl,nc),F(.00013*(i+1)),dtype='f4') for i in range(3)]+[np.zeros((nl,nc),dtype='f4') for _ in range(4)]+[np.zeros(nc,dtype='f4'),np.zeros(4*58+2*(nl+1),dtype='f4'),np.zeros((2,nc),dtype='i4'),np.zeros((2,nc),dtype='f4')]
    return out
records=[]
if args.gpu:
    import cupy as cp
    from woof.hex.mpasphys.ugwp_ngw_v842 import run,to_mpas,bind_native_pools,launch_slot,PREPARED_ALIASES
    from types import SimpleNamespace
    class NativeDriverFixture(dict):
        def prepared(self,name):
            return self.profiles[name]
    from dataclasses import replace
scenarios=[(120.,.003,80.,44000.,15.),(1800.,.02,140.,44000.,15.),
           (30.,.0001,310.,44000.,15.),(120.,.2,80.,130000.,150.),
           (7200.,.1,140.,25000.,300.),(10.,.00001,30.,35000.,0.),(120.,.003,120.,130000.,0.),
           (120.,1000.,180.,44000.,50000.),(7200.,.1,80.,10000.,15.),
           (120.,.003,80.,2000.,15.),(720000.,.003,80.,130000.,0.),
           (120.,.003,80.,44000.,15.),(.0001,1000.,80.,44000.,15.),
           (1.e-25,.00025,80.,44000.,0.)]
if args.defect_probes:
    scenarios += [(120.,.003,-20.,44000.,15.),(120.,.003,80.,44000.,15.)]
original_lat=lat.copy()
for case,(dtp,amp,day,top,shear) in enumerate(scenarios):
    lat=np.zeros_like(original_lat) if case == 13 else original_lat
    zi=np.broadcast_to(np.linspace(0,top,nl+1,dtype='f4')[:,None],(nl+1,nc)).copy()
    if case == 12:
        zi[13:15]=np.array([9601.,9602.],dtype='f4')[:,None]
    if case == 13:
        # Near-transparent first propagation layer tests the aggregate flux cap;
        # tiny dt keeps native explicit diffusion stable without huge substeps.
        zi[:13]=np.linspace(0,3.e-9,13,dtype='f4')[:,None]
    z=((zi[1:]+zi[:-1])*F(.5)).astype('f4')
    p=(F(101325)*np.exp(-z/F(7000))).astype('f4');pi=(F(101325)*np.exp(-zi/F(7000))).astype('f4')
    if case == 15:
        # Native source launch clamps to three and reads the dead azmeti(0).
        p *= F(.1);pi *= F(.1)
    t=np.maximum(F(190),F(288)-z*F(.004)).astype('f4')
    if case == 13:
        t[:14]=F(280);p[:12]=F(30000);p[12]=F(25000);pi[:13]=F(30000)
    u=(F(10)+F(shear)*np.sin(k*F(.2)+np.arange(nc,dtype='f4')[None,:])).astype('f4')
    v=(F(-5)+F(shear*.66)*np.cos(k*F(.23)+np.arange(nc,dtype='f4')[None,:])).astype('f4')
    if case == 13:
        u.fill(0);v.fill(0)
        rc=np.zeros_like(rc);rn=np.zeros_like(rn)
    q=(F(.01)*np.exp(-z/F(2000))).astype('f4');ex=(p/F(1e5))**F(2/7)
    if case == 13:
        q[:14]=0
    dzu=np.diff(zi[:,0]).copy();rdzw=F(1)/dzu
    taulat=np.linspace(-80 if case==11 else -90,80 if case==11 else 90,7,dtype='f4')
    inputs=[rdzw,dzu,lat,taulat,days,tau,u,v,t,q,p,ex,pi,z,zi,rc,rn]
    n=arrays();h=arrays();status=np.zeros(nc,dtype='i4')
    native_entry = native_modes if case % 2 else native
    native_entry(nl,nc,dtp,amp,day,*[ptr(a) for a in inputs[:6]],7,7,*[ptr(a) for a in inputs[6:]],*[ptr(a) for a in n[:9]],ptr(n[9][0]),ptr(n[9][1]),ptr(n[10][0]),ptr(n[10][1]))
    cpu(nc,7,7,dtp,amp,day,*[ptr(a) for a in inputs],*[ptr(a) for a in h],ptr(status))
    if args.gpu:
        columns=dict(zip(('um','vm','tm','qm','prsl','prslk','prsi','zmet','zmeti','raincv','rainncv'),map(cp.asarray,inputs[6:])))
        columns['lat_r']=cp.asarray(lat)
        for name,array in zip(names[:3],arrays()[:3]): columns[name]=cp.asarray(array)
        climate=dict(zip(('rdzw','dzu','LATS','DAYS','ABSMF'),map(cp.asarray,(rdzw,dzu,taulat,days,tau))))
        output=run(columns,climate,dtp=dtp,amplitude=amp,day=day)
        h=[cp.asnumpy(getattr(output,name if name!='initial_state' else 'initial_state')) for name in names]
        status=cp.asnumpy(output.status)
    checked_names=list(names)
    if args.gpu:
        # Native writeback slice, including subnormal rates and disabled diag sentinels.
        inputs_export=[a.copy() for a in n[:6]]
        before=[a.copy() for a in arrays()[:2]]
        if case==0:
            for array in inputs_export[:2]: array[0,:4]=np.array([1.e-40,-1.e-40,0.,-0.],dtype='f4')
            for array in before: array[0,:4]=np.array([0.,0.,-1.e-40,1.e-40],dtype='f4')
        expected=[np.full((nl,nc),F(.00099),dtype='f4') for _ in range(8)]
        enabled=case%2==0
        export_native(nc,nl,int(enabled),*[ptr(a) for a in inputs_export],*[ptr(a) for a in before],*[ptr(a) for a in expected])
        target_names=('rublten','rvblten','rthblten','rubldiff','rvbldiff','dudt_ngw','dvdt_ngw','dtdt_ngw')
        target=dict(zip(target_names,map(cp.asarray,[np.full_like(a,F(.00099)) for a in expected])))
        pool={key:cp.empty(nc,cp.int32 if key.startswith('jindx') else cp.float32) for key in ('jindx1_tau','jindx2_tau','ddy_j1tau','ddy_j2tau')}
        export_result=replace(output,**dict(zip(names[:6],map(cp.asarray,inputs_export))))
        # Native rates and drag differences live in distinct Registry pools.
        tendencies={name:target[name] for name in target_names[:3]}
        diagnostics={name:target[name] for name in target_names[3:]}
        to_mpas(export_result,diagnostics,tendencies,gwdo_before=dict(zip(names[:2],map(cp.asarray,before))),ngw_input=pool,ugwp_diags=enabled)
        n.extend(expected+[n[9][0].copy(),n[9][1].copy(),n[10][0].copy(),n[10][1].copy()])
        h.extend([cp.asnumpy(target[name]) for name in target_names]+[cp.asnumpy(pool[name]) for name in pool])
        checked_names.extend(['writeback_'+name for name in target_names]+list(pool))
        # Exercise the real table binder/callback with the host snapshots
        # actually produced by session_v842.snapshot_gwdo, padded native mass
        # storage, independent Registry pools and native climate axis order.
        bound_expected=[np.full((nl,nc),F(.00099),dtype='f4') for _ in range(8)]
        export_native(nc,nl,int(enabled),*[ptr(a) for a in n[:6]],*[ptr(a) for a in before],*[ptr(a) for a in bound_expected])
        bound_targets=dict(zip(target_names,map(cp.asarray,[np.full_like(a,F(.00099)) for a in bound_expected])))
        for name,initial in zip(names[:3],arrays()[:3]):bound_targets[name][...]=cp.asarray(initial)
        bound_diag={name:bound_targets[name] for name in target_names[3:]}
        bound_diag.update(raincv=columns['raincv'],rainncv=columns['rainncv'])
        bound_pool={key:cp.empty_like(value) for key,value in pool.items()}
        bound_pool.update(LATS=climate['LATS'],DAYS=climate['DAYS'],ABSMF=cp.ascontiguousarray(climate['ABSMF'].T))
        provider=NativeDriverFixture(mesh={'nVertLevels':nl,'latCell':columns['lat_r'],'rdzw':climate['rdzw'],'dzu':climate['dzu']},
            diag_physics=bound_diag,tend_physics={name:bound_targets[name] for name in names[:3]},ngw_input=bound_pool)
        provider.cp=cp
        provider.profiles={alias:cp.concatenate((columns[name],cp.full((1,nc),F(-999),dtype='f4')),axis=0)
            if name not in ('prsi','zmeti') else columns[name] for name,alias in PREPARED_ALIASES.items()}
        provider.session=SimpleNamespace(layout=SimpleNamespace(dimensions={'nCellsSolve':nc}),
            arguments=SimpleNamespace(calendar={'julian':day}),
            gwdo_before={name:np.asfortranarray(np.vstack((value,np.full((1,nc),F(-999),dtype='f4')))) for name,value in zip(names[:2],before)})
        call=SimpleNamespace(slot='ngw',scheme='ugwp_ngw',backend='hex-cuda',dt=dtp,
            options={'config_convection_scheme':'cu_kf','config_microp_scheme':'mp_wsm6','config_knob_ugwp_tauamp':amp,'config_ugwp_diags':enabled})
        call.pools=bind_native_pools(provider,call)
        bound_result=launch_slot(call)
        n.extend(bound_expected+[n[9][0].copy(),n[9][1].copy(),n[10][0].copy(),n[10][1].copy()])
        h.extend([cp.asnumpy(bound_targets[name]) for name in target_names]+[cp.asnumpy(bound_pool[name]) for name in pool])
        checked_names.extend(['bound_slot_'+name for name in target_names]+['bound_slot_'+name for name in pool])
        n.extend([a.copy() for a in n[:11]])
        h.extend([cp.asnumpy(getattr(bound_result,name)) for name in names])
        checked_names.extend(['bound_result_'+name for name in names])
        assert all(code==0 for code in cp.asnumpy(bound_result.status))
    result={}
    for name,a,b in zip(checked_names,n,h):
        diff=a.view('u4')!=b.view('u4');idx=np.argwhere(diff)
        result[name]={'words':int(a.size),'differing_words':int(diff.sum()),'native_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'candidate_sha256':hashlib.sha256(b.tobytes()).hexdigest()}
        if idx.size:
            print(case,name,int(diff.sum()),'sample',[(tuple(i),a[tuple(i)].item(),b[tuple(i)].item(),hex(int(a.view('u4')[tuple(i)])),hex(int(b.view('u4')[tuple(i)]))) for i in idx[:3]])
    records.append({'case':case,'native_driver_diags':case%2==0,'native_kdt':1 if case%2 else 2,'dtp':dtp,'amp':amp,'day':day,'top':top,'wind_amplitude':shear,'status':status.tolist(),'arrays':result})
from woof.hex.mpasphys import ugwp_ngw_v842 as port
provenance={'port_sha256':hashlib.sha256(Path(port.__file__).read_bytes()).hexdigest(),
            'native_sources':port.SOURCE_PINS,
            'native_binary_sha256':hashlib.sha256((args.work/'native/liboracle.so').read_bytes()).hexdigest(),
            'host_witness_binary_sha256':hashlib.sha256((args.work/'libhost.so').read_bytes()).hexdigest(),
            'replay_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if args.gpu:
    provenance.update(shared_math_sha256=hashlib.sha256(port.shared_math_header().encode()).hexdigest(),
                      cupy_version=cp.__version__,runtime_version=cp.cuda.runtime.runtimeGetVersion(),
                      driver_version=cp.cuda.runtime.driverGetVersion(),
                      compiler_options=['--std=c++17','--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true'])
args.receipt.write_text(json.dumps({'schema':'p15.ugwp-ngw-debug/v2','backend':'hex-cuda shared math' if args.gpu else 'host-glibc C++ witness of CUDA arithmetic','acceptance':False,'provenance':provenance,'cases':records},indent=2)+'\n')
print('total differing words',sum(v['differing_words'] for r in records for v in r['arrays'].values()))

assert all(v['differing_words']==0 for r in records for v in r['arrays'].values())
assert all(code==0 for r in records for code in r['status'])
