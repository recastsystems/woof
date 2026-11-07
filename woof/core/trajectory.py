"""Native EM particle integration and asynchronous Rust point output.

This module owns configuration, pointer binding and output scheduling only.
Weather interpolation and particle movement execute in trajectory.cu.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta
from pathlib import Path
import hashlib
import numpy as np
from woof.core.trajectory_contract import active,parse_points,COORDINATES,scratch_shapes
from woof.core.kernels import get_kernel


def _launch(name,n,args):
    if n:get_kernel('trajectory',name)(((n+127)//128,),(128,),args)


def feedback(parent,child):
    if not hasattr(parent,'traj_coordinates') or not hasattr(child,'traj_coordinates'):return
    if parent.traj_coordinates.shape!=child.traj_coordinates.shape:
        raise ValueError('native num_traj is global across the nest tree')
    _launch('trajectory_feedback',parent.traj_coordinates.shape[1],
            (parent.traj_coordinates,child.traj_coordinates,np.int32(parent.traj_coordinates.shape[1])))


class Trajectories:
    def __init__(self,state,cfg,restored,*,start_time,end_time,domain_id,output_directory):
        import cupy as cp
        self.cp,self.state,self.cfg=cp,state,cfg
        self.start_time=start_time.replace(tzinfo=None)
        self.end_time=end_time.replace(tzinfo=None)
        self.capacity=cfg.num_traj
        self.points=list(parse_points(cfg.trajectory_input))
        self.domain_id=domain_id
        self.path=Path(output_directory)/f'wrfout_traj_d{domain_id:02d}'
        self._shapes=scratch_shapes(cfg)
        self.proj=self._array('projection',(14,),cp.float32)
        attrs=restored.global_attributes
        self.code=int(attrs['MAP_PROJ'])
        if self.code not in (1,2,3):raise ValueError('native trajectory projection needs an admitted Lambert, polar or Mercator grid')
        get_kernel('trajectory','trajectory_projection')((1,),(1,),
            (self.proj,np.int32(self.code),np.float32(cfg.dx),np.float32(cfg.dy),
             np.float32(restored.raw['XLAT'][0,0]),np.float32(restored.raw['XLONG'][0,0]),
             np.float32(attrs['TRUELAT1']),np.float32(attrs['TRUELAT2']),np.float32(attrs['STAND_LON'])))
        self._pointer_identity=None
        self._executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix=f'trajectory-d{domain_id:02d}')
        self._pending=None;self._slot=0;self._bank=0;self.records=0;self.writer=None
        self._times=[[],[]]
        self._prepared=False

    def _array(self,name,shape,dtype):
        dtype=np.dtype(dtype);size=int(np.prod(shape))
        slot='trajectory_'+name
        full=self.state.scratch(self._shapes[slot],slot)
        return full.view(dtype)[:size].reshape(shape)

    def cold_start(self,*,roster_only=False):
        cp=self.cp;s=self.state;c=self.cfg;n=len(self.points)
        # Native restart re-scans the original point locations to rebuild
        # its compact roster, while preserving the stored moving positions.
        coordinates=(self._array('initial_coordinates',(5,self.capacity),cp.float32)
                     if roster_only else s.traj_coordinates)
        if not n or not any(self._binding(pkg,name) is not None for p in self.points for pkg in ('hyd','trc','dyn','msc') for name in p[pkg]):
            s.traj_status[:]=0
            return
        source=self._array('input',(n,3),cp.float32)
        source.set(np.asarray([[p['lon'],p['lat'],p['lev']] for p in self.points],np.float32).reshape(n,3))
        agl=self._array('agl',(n,),cp.int32)
        agl.set(np.asarray([p['z_is_agl'] for p in self.points],np.int32))
        _launch('trajectory_initialize',self.capacity,
            (coordinates,np.int32(self.capacity),np.int32(n),source,agl,np.int32(self.code),self.proj,
             s.php,s.phb,np.int32(s.phb.ndim==1),np.int32(c.nx),np.int32(c.ny),np.int32(c.nz)))
        valid=cp.asnumpy(coordinates[0,:n])!=-9999
        count=int(np.count_nonzero(valid))
        if count!=n:
            # Native's conditional increment of m is deliberate here. It is
            # observable when the first point is valid and a later one is not.
            order=list(range(n));m=0
            for j in range(n):
                if valid[j] and j!=m:order[m]=j;m+=1
            order=order[:count]
            if count:coordinates[:,:count]=coordinates[:,cp.asarray(order)]
            coordinates[:,count:]=-9999
            self.points=[self.points[i] for i in order]
        if not count:coordinates.fill(0)  # native returns before assigning missing_val
        s.traj_status[0]=np.float32(count)
        s.traj_status[1]=np.float32(0)

    def attach_output(self,*,checkpoint=None,warm=False):
        cp=self.cp;s=self.state;c=self.cfg
        if checkpoint:
            self.points=checkpoint['points']
        elif warm:
            self.cold_start(roster_only=True)
        self.count=int(cp.asnumpy(s.traj_status)[0])
        if self.count<0:self.cold_start();self.count=int(cp.asnumpy(s.traj_status)[0])
        if self.count==0:return
        self.points=self.points[:self.count]
        if not checkpoint:
            has_vars=[any(self._binding(pkg,name) is not None for pkg in ('hyd','trc','dyn','msc') for name in p[pkg]) for p in self.points]
            kept=sum(has_vars)
            if kept==0:
                s.traj_status[0]=np.float32(0);self.count=0;return
            if kept!=self.count:
                # This branch reads in_dom before cold startup initializes
                # it (native lines 677 and 765). Do not invent a compaction.
                raise ValueError('native module_trajectory no-variable compaction reads undefined in_dom; each retained point must request a native variable')
        self.variables=[];self._bindings=[];requested=[]
        for package in ('hyd','trc','dyn','msc'):
            for p in self.points:
                for name in p[package]:
                    if (package,name) in self.variables:continue
                    binding=self._binding(package,name)
                    if binding is None:continue # native scan_vars discards unknown names
                    self.variables.append((package,name));self._bindings.append(binding)
                    requested.append([int(name in q[package]) for q in self.points])
        self.requested=self._array('requested',(len(requested),self.count),cp.int32)
        self.requested.set(np.asarray(requested,np.int32).reshape(len(requested),self.count))
        intervals=[]
        for p in self.points:
            a=datetime.strptime(p['start_time'],'%Y-%m-%d_%H:%M:%S') if p['start_time'] else self.start_time
            b=datetime.strptime(p['stop_time'],'%Y-%m-%d_%H:%M:%S') if p['stop_time'] else self.end_time
            if b<a:raise ValueError('trajectory stop_time precedes start_time')
            intervals.append(((a-self.start_time).total_seconds(),(b-self.start_time).total_seconds(),not p['is_stationary']))
        self.intervals=self._array('intervals',(self.count,3),cp.float64)
        self.intervals.set(np.asarray(intervals,np.float64))
        self.moving=self._array('moving',(self.count,),cp.int32)
        self.enabled=self._array('enabled',(self.count,),cp.int32)
        self.ptrs=self._array('pointers',(3*len(self.variables),),cp.uint64)
        self.desc=self._array('descriptors',(len(self.variables),4),cp.int32)
        shape=(1000,5+len(self.variables),self.count)
        self.buffers=[self._array('output'+str(j),shape,cp.float32) for j in range(2)]
        from woof.core.preflight import host_available_bytes
        pinned_bytes=2*int(np.prod(shape))*4
        available=host_available_bytes()
        if available is not None and pinned_bytes>available:
            raise MemoryError(f'trajectory asynchronous output needs {pinned_bytes} pinned host bytes; {available} available')
        self.pinned=[cp.cuda.alloc_pinned_memory(int(np.prod(shape))*4) for _ in range(2)]
        self.host=[np.ndarray(shape,np.float32,buffer=b) for b in self.pinned]
        self.events=[cp.cuda.Event(),cp.cuda.Event()]
        self._refresh_bindings()
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if checkpoint:
            if self.path.exists():raise FileExistsError(self.path)
            self.path.write_bytes(checkpoint['prefix'].tobytes())
            self.records=int(checkpoint['records'])
            self.writer=self._schema().resume(self.path,self.records)
        else:
            if self.path.exists():raise FileExistsError(self.path)
            self.writer=self._schema().create(self.path)
        self._prepared=True
        # WRF calls trajectory_driver after solve_em. Its first point
        # record is at the end of the first step, never at initialization.

    def _binding(self,package,name):
        s=self.state
        if package=='dyn':
            if name in ('rainprod','evapprod'):
                raise ValueError(f'native non-Chem module_trajectory does not initialize dyn_var={name}; no invented values are substituted')
            carried=getattr(s,'p_perturbation',None) is not None
            key={'p':('p_perturbation','pb',None,4) if carried else ('p',None,None,0),
                 'T':('thp','p_perturbation','pb',5) if carried else ('thp',None,'p',8),
                 'z':('php','phb',None,6),
                 'u':('u',None,None,1),'v':('v',None,None,2),'w':('w',None,None,3)}.get(name)
            return key
        from woof.ingest.wrfinput import active_moisture_map
        moisture=active_moisture_map(self.cfg)
        if package=='hyd':
            attr=moisture.get(name)
            return (attr,None,None,0) if attr and getattr(s,attr,None) is not None else None
        if package=='trc':return (name,None,None,0) if name.startswith('tr17_') and getattr(s,name,None) is not None else None
        # reg_scan accepts only native XZY 3-D state, never T2 or another 2-D field.
        core={'U':('u',None,None,1),'V':('v',None,None,2),'W':('w',None,None,3),'PH':('php',None,None,3),
              'PHB':('phb',None,None,3),'T':('thp',None,None,0),'THM':('thp',None,None,0),
              'AL':('al',None,None,0),'ALT':('alt',None,None,0),'ALB':('alb',None,None,0),
              'P':('p_perturbation',None,None,0),'PB':('pb',None,None,0)}
        if name=='P' and getattr(s,'p_perturbation',None) is None:return ('p','pb',None,9)
        if name in core:
            result=core[name]
            if getattr(s,result[0],None) is None:raise ValueError(f'trajectory Registry sampler has no live carrier for {name}')
            return result
        if name in moisture:return self._binding('hyd',name)
        if name.startswith('tr17_'):return self._binding('trc',name)
        from woof.io.history_layout import live_state_history_fields
        fields=live_state_history_fields(s)
        a=fields.get(name)
        if a is None:
            driver=getattr(s,'physics',None)
            if driver is not None:
                from woof.io.history_layout import physics_history_fields
                a=physics_history_fields(driver).get(name)
                if a is not None and a.ndim==3:
                    for key,value in driver.fields.items():
                        if value is a:
                            if a.shape[0] not in (self.cfg.nz,self.cfg.nz+1):return None
                            kind=1 if a.shape[2]==self.cfg.nx+1 else 2 if a.shape[1]==self.cfg.ny+1 else 3 if a.shape[0]==self.cfg.nz+1 else 0
                            return ('physics.fields:'+key,None,None,kind)
        if a is None or a.ndim!=3:return None
        # Bind a field by its live state attribute; metadata never owns a copy.
        for attr,value in vars(s).items():
            if value is a:
                kind=1 if a.shape[2]==self.cfg.nx+1 else 2 if a.shape[1]==self.cfg.ny+1 else 3 if a.shape[0]==self.cfg.nz+1 else 0
                return (attr,None,None,kind)
        raise ValueError(f'trajectory Registry field {name} has no stable producer binding')

    def _carrier(self,name):
        return self.state.physics.fields[name.split(':',1)[1]] if name.startswith('physics.fields:') else getattr(self.state,name)

    def _refresh_bindings(self):
        ptr=[];desc=[];s=self.state
        for a,b,p,kind in self._bindings:
            first=self._carrier(a);second=self._carrier(b) if b else first;pressure=self._carrier(p) if p else first
            if first.ndim!=3:raise ValueError(f'trajectory sampler requires a native 3-D {a} carrier')
            ptr.extend((first.data.ptr,second.data.ptr,pressure.data.ptr))
            desc.append((kind,first.shape[2],first.shape[1],int(second.ndim==1)))
        identity=tuple(ptr)
        if identity!=self._pointer_identity:
            self.ptrs.set(np.asarray(ptr,np.uint64));self.desc.set(np.asarray(desc,np.int32).reshape(-1,4));self._pointer_identity=identity

    def _schema(self):
        from woof.io.nc_writer_bridge import ClassicSchema
        schema=ClassicSchema();time=schema.def_dim('time',0,unlimited=True)
        traj=schema.def_dim('traj',self.count);chars=schema.def_dim('DateStrLen',19)
        self.time_var=schema.def_var('Times','S1',(time,chars))
        self.varids=[]
        for name in COORDINATES:self.varids.append(schema.def_var(name,np.float32,(time,traj)))
        for package,name in self.variables:
            varid=schema.def_var(name+'_traj',np.float32,(time,traj));self.varids.append(varid)
            if package=='hyd':description,units=name+'_traj mixing ratio','ug/kg-dryair'
            elif package=='trc':description,units=name+'_traj mixing ratio',''
            elif package=='dyn':
                description,units={'p':('pressure','hPa'),'T':('temperature','K'),'z':('height','m'),
                    'u':('x wind component','m/s'),'v':('y wind component','m/s'),'w':('z wind component','m/s')}[name]
            else:
                from woof.io.wrf_output_schema import REGISTRY_VAR_META
                description,units=REGISTRY_VAR_META.get(name,(name,''))[:2]
            schema.put_var_attr(varid,'description',description);schema.put_var_attr(varid,'units',units)
        return schema

    def advance(self,ru,rv,ww,mut,muu,muv,dt):
        if not self._prepared or not self.count:return
        s=self.state;c=self.cfg
        next_seconds=float(s.elapsed_seconds)+int(dt)
        _launch('trajectory_clock',self.count,(self.intervals,np.float64(next_seconds),self.moving,self.enabled,np.int32(self.count)))
        _launch('trajectory_advance',self.count,
            (s.traj_coordinates,np.int32(self.capacity),np.int32(self.count),self.moving,np.int32(self.code),self.proj,
             np.float32(dt),np.float32(1/c.dx),np.float32(1/c.dy),ru,rv,ww,mut,muu,muv,s.msft,s.msfu,s.msfv,
             s.c1h,s.c2h,s.c1f,s.c2f,s.znw,np.int32(c.nx),np.int32(c.ny),np.int32(c.nz)))

    def sample(self,seconds,*,initial=False):
        if not self._prepared or not self.count:return
        s=self.state;c=self.cfg
        _launch('trajectory_clock',self.count,
            (self.intervals,np.float64(float(seconds)+(int(c.dt) if initial else 0)),self.moving,self.enabled,np.int32(self.count)))
        self._refresh_bindings()
        _launch('trajectory_sample',self.count*(5+len(self.variables)),
            (s.traj_coordinates,np.int32(self.capacity),np.int32(self.count),self.ptrs,self.desc,self.requested,self.enabled,
             self.buffers[self._bank],np.int32(self._slot),np.int32(len(self.variables)),np.int32(c.nx),np.int32(c.ny),np.int32(c.nz)))
        self._times[self._bank].append((self.start_time+timedelta(seconds=float(seconds))).strftime('%Y-%m-%d_%H:%M:%S'))
        self._slot+=1
        s.traj_status[1]+=np.float32(1)
        if self._slot==1000:self.flush(wait=False)

    def flush(self,*,wait=True):
        if not self._prepared:return
        if self._pending is not None:self._pending.result();self._pending=None
        count=self._slot
        if count:
            bank=self._bank;times=self._times[bank];self.events[bank].record()
            self._bank=1-bank;self._slot=0;self._times[bank]=[]
            def write():
                cp=self.cp
                with cp.cuda.Stream(non_blocking=True) as stream:
                    stream.wait_event(self.events[bank])
                    self.buffers[bank][:count].get(out=self.host[bank][:count],stream=stream,blocking=False)
                    stream.synchronize()
                if not np.isfinite(self.host[bank][:count]).all():raise FloatingPointError('trajectory sampled a native undefined halo or nonfinite input')
                for j,stamp in enumerate(times):
                    self.writer.write_record(self.records,self.time_var,np.frombuffer(stamp.encode(),dtype='S1'))
                    for k,varid in enumerate(self.varids):self.writer.write_record(self.records,varid,self.host[bank][j,k])
                    self.records+=1
            self._pending=self._executor.submit(write)
        if wait and self._pending is not None:self._pending.result();self._pending=None

    def checkpoint_metadata(self):
        if not self._prepared:return None
        self.flush();self.writer.finish();self.writer=None
        # The archive owns this prefix. A later step may append to the live
        # output, and the archive must remain independently resumable.
        prefix=np.frombuffer(self.path.read_bytes(),dtype=np.uint8)
        row=dict(sha256=hashlib.sha256(prefix).hexdigest(),records=self.records,points=self.points)
        self.writer=self._schema().resume(self.path,self.records)
        return row,prefix

    def close(self):
        try:
            self.flush()
            if self.writer is not None:self.writer.finish();self.writer=None
        finally:self._executor.shutdown(wait=True)
