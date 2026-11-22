"""MPAS-A v8.4.2 sounding lifecycle and CUDA numerical calculations.

Rust owns list-directed location parsing and exact formatted file output. The
clock adapter owns MPAS time intervals/alarms. Device transcendental functions
must come from the shared GNU libm header; there is no CUDA fast-math fallback.
"""
from __future__ import annotations

from .cloud_v842 import _array, _qv_array, _compile_cuda


def shared_device_math():
    """Use the admitted engine's common GNU libm transcriptions verbatim."""
    from pathlib import Path
    from woof.core import kernels
    directory=Path(kernels.__file__).parent
    return ("typedef float real;\n"+
        (directory/"glibc_flt32.cuh").read_text(encoding="utf-8")+"\n"+
        (directory/"glibc_trig_flt32.cuh").read_text(encoding="utf-8")+"\n"+
        "#define mpas_logf gfk_log\n#define mpas_sinf glibc_sinf\n"
        "#define mpas_cosf glibc_cosf\n#define mpas_asinf glibc_asinf\n"
        "#define mpas_acosf glibc_acosf\n")

CUDA_SOURCE = r'''
#define ADD(a,b) __fadd_rn(a,b)
#define SUB(a,b) __fsub_rn(a,b)
#define MUL(a,b) __fmul_rn(a,b)
#define DIV(a,b) __fdiv_rn(a,b)
// mpas_soundings.F:431-490; native ordered Horner polynomials.
__device__ float rslf(float p,float t) {
    float x=fmaxf(-80.0f,SUB(t,273.16f));
    float e=ADD(0.379534310E-11f,MUL(x,-0.321582393E-13f));
    e=ADD(0.702620698E-8f,MUL(x,e));
    e=ADD(0.203154182E-5f,MUL(x,e));
    e=ADD(0.299291081E-3f,MUL(x,e));
    e=ADD(0.264224321E-1f,MUL(x,e));
    e=ADD(0.143177157E1f,MUL(x,e));
    e=ADD(0.444606896E2f,MUL(x,e));
    e=ADD(0.611583699E3f,MUL(x,e));
    return DIV(MUL(0.622f,e),SUB(p,e));
}
__device__ float rsif(float p,float t) {
    float x=fmaxf(-80.0f,SUB(t,273.16f));
    float e=ADD(0.105785160E-9f,MUL(x,0.161444444E-12f));
    e=ADD(0.307839583E-7f,MUL(x,e));
    e=ADD(0.521693933E-5f,MUL(x,e));
    e=ADD(0.565392987E-3f,MUL(x,e));
    e=ADD(0.402737184E-1f,MUL(x,e));
    e=ADD(0.184672631E1f,MUL(x,e));
    e=ADD(0.499320233E2f,MUL(x,e));
    e=ADD(0.609868993E3f,MUL(x,e));
    return DIV(MUL(0.622f,e),SUB(p,e));
}
extern "C" __global__ void saturation(int n,const float* p,const float* t,float* liquid,float* ice) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<n) {liquid[j]=rslf(p[j],t[j]);ice[j]=rsif(p[j],t[j]);}
}
#ifndef MPAS_SOUNDINGS_SATURATION_ONLY
// mpas_soundings.F:408-416
__device__ float sphere_distance(float lat1,float lon1,float lat2,float lon2,float radius) {
    float a=mpas_sinf(MUL(0.5f,SUB(lat2,lat1)));
    float b=mpas_sinf(MUL(0.5f,SUB(lon2,lon1)));
    float sum=ADD(MUL(a,a),MUL(MUL(mpas_cosf(lat1),mpas_cosf(lat2)),MUL(b,b)));
    return MUL(MUL(2.0f,radius),mpas_asinf(__fsqrt_rn(sum)));
}
// mpas_soundings.F:363-400; strict improvement and first-neighbor tie ordering.
__device__ int nearest(float lat,float lon,int start,int nc,int me,const int* nec,
    const int* coc,const float* latCell,const float* lonCell) {
    int next=start,current=-1;
    while(next!=current) {
        current=next;
        float best=sphere_distance(latCell[current],lonCell[current],lat,lon,1.0f);
        next=current;
        for(int i=0;i<nec[current];++i) {
            int c=coc[i+me*current]-1;
            if(c<nc) {
                float d=sphere_distance(latCell[c],lonCell[c],lat,lon,1.0f);
                if(d<best) {next=c;best=d;}
            }
        }
    }
    return next;
}
// mpas_soundings.F:186-203; each location starts from previous station's cell.
extern "C" __global__ void locate(int ns,int nc,int solve,int me,const int* nec,const int* coc,
    const float* latCell,const float* lonCell,const float* lat,const float* lon,float pi,
    int* cells,int* owned) {
    if(blockIdx.x||threadIdx.x) return;
    int cell=nc-1;
    for(int s=0;s<ns;++s) {
        cell=nearest(DIV(MUL(lat[s],pi),180.0f),DIV(MUL(lon[s],pi),180.0f),cell,nc,me,nec,coc,latCell,lonCell);
        owned[s]=cell<solve;
        cells[s]=owned[s]?cell+1:0;
    }
}
// mpas_soundings.F:292-324; output shape (6,levels,stations), Fortran order.
extern "C" __global__ void sounding(int ns,int nz,const int* cells,const int* owned,
    const float* tm,const float* q,const float* exner,const float* pb,const float* pp,
    const float* u,const float* v,const float* z,float rvord,float pi,float* rows) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=ns*nz) return;
    int s=j/nz,k=j%nz,c=cells[s]-1;
    if(!owned[s]) return;
    int f=k+nz*c;
    float t=MUL(DIV(tm[f],ADD(1.0f,MUL(rvord,q[f]))),exner[f]);
    float pres=ADD(pb[f],pp[f]);
    float rh=fmaxf(1.0e-8f,fminf(1.0f,DIV(q[f],rslf(pres,t))));
    float logr=mpas_logf(rh);
    t=SUB(t,273.15f);pres=MUL(pres,0.01f);
    float term=DIV(MUL(17.625f,t),ADD(243.04f,t));
    float td=DIV(MUL(243.04f,ADD(logr,term)),SUB(SUB(17.625f,logr),term));
    float speed=__fsqrt_rn(ADD(MUL(u[f],u[f]),MUL(v[f],v[f]))), dir=0.0f;
    if(speed!=0.0f) {
        dir=mpas_acosf(DIV(-v[f],speed));
        if(u[f]>0.0f) dir=SUB(MUL(2.0f,pi),dir);
        dir=DIV(MUL(dir,180.0f),pi);
    }
    int out=6*j;
    rows[out]=pres;
    rows[out+1]=MUL(0.5f,ADD(z[k+(nz+1)*c],z[k+1+(nz+1)*c]));
    rows[out+2]=t;rows[out+3]=td;rows[out+4]=dir;rows[out+5]=speed;
}
#endif
'''


class Soundings:
    """Lifecycle from mpas_soundings.F:51-220,237-340,353-366.

    io.read_locations returns None when absent or a Rust-decoded object with
    names, float32 device latitudes/longitudes. io.write_soundings writes the
    native headers, names and F-width rows, without numerical processing in
    Python. Clock methods implement MPAS alarms and time interval strings.
    """

    def __init__(self, mesh=None, state=None, diag=None, clock=None, io=None, *, device_math=None, rvord=None):
        self.mesh,self.state,self.diag,self.clock=mesh,state,diag,clock
        self.io=io if io is not None else RustSoundingIO()
        self.device_math=device_math
        self.locations=None
        self.cells=self.owned=None
        self._module=None
        import numpy as np
        self.rvord=np.float32(rvord) if rvord is not None else np.float32(461.6)/np.float32(287.0)
        # Native initialized expression 2*asin(1) folds to this RKIND word.
        self.pi=np.float32(float.fromhex("0x1.921fb6p+1"))

    def _kernel(self, name):
        if self._module is None:
            import cupy as cp
            if self.device_math is None:
                self.device_math=shared_device_math()
            self._module=_compile_cuda(self.device_math+CUDA_SOURCE)
        return self._module.get_function(name)

    def setup(self, config_sounding_interval, pools=None, clock=None, dminfo=None):
        """Location list and initial alarm: mpas_soundings.F:51-220."""
        if pools is not None:
            configs = config_sounding_interval
            config_sounding_interval = configs.get("config_sounding_interval")
            self.mesh, self.state, self.diag = (pools[name] for name in ("mesh", "state", "diag"))
            self.clock = clock
        if config_sounding_interval is None:
            return
        config_sounding_interval = config_sounding_interval.rstrip()
        if config_sounding_interval=="none":
            return
        native_probe = hasattr(self.io, "exists")
        if native_probe:
            if not self.io.exists("sounding_locations.txt", dminfo=dminfo):
                return
            self.clock.add_alarm("soundingAlarm",self.clock.now(),config_sounding_interval)
        self.locations=(self.io.read_locations("sounding_locations.txt") if dminfo is None
                        else self.io.read_locations("sounding_locations.txt", dminfo=dminfo))
        if self.locations is None:
            return
        if not native_probe:
            self.clock.add_alarm("soundingAlarm",self.clock.now(),config_sounding_interval)
        ns=len(self.locations.names)
        if ns==0:
            return
        import cupy as cp
        self.cells=cp.empty(ns,dtype=cp.int32)
        self.owned=cp.empty(ns,dtype=cp.int32)
        m=self.mesh
        self._kernel("locate")((1,),(1,),(
            cp.int32(ns),cp.int32(m["nCells"]),cp.int32(m["nCellsSolve"]),cp.int32(m["maxEdges"]),
            _array(m["nEdgesOnCell"],"int32",1),_array(m["cellsOnCell"],"int32",2),
            _array(m["latCell"],"float32",1),_array(m["lonCell"],"float32",1),
            _array(self.locations.latitudes,"float32",1),_array(self.locations.longitudes,"float32",1),
            self.pi,self.cells,self.owned))

    def compute(self):
        """Alarm, six-column profile and Rust writer: mpas_soundings.F:237-340."""
        if not self.clock.is_ringing("soundingAlarm"):
            return
        ns=0 if self.locations is None else len(self.locations.names)
        if ns:
            import cupy as cp
            nz=int(self.mesh["nVertLevels"])
            rows=cp.empty((6,nz,ns),dtype=cp.float32,order="F")
            self._kernel("sounding")(((ns*nz+127)//128,),(128,),(
                cp.int32(ns),cp.int32(nz),self.cells,self.owned,
                _array(self.state["theta_m"],"float32",2),_qv_array(self.state),
                *(_array(self.diag[name],"float32",2) for name in (
                    "exner","pressure_base","pressure_p","uReconstructZonal","uReconstructMeridional")),
                _array(self.mesh["zgrid"],"float32",2),self.rvord,self.pi,rows))
            # Transfer bytes only; Rust owns every filename and formatting decision.
            self.io.write_soundings(self.clock.now(),self.locations,self.owned.get(),rows.get())
        self.clock.reset_alarm("soundingAlarm")

    def cleanup(self):
        """Release station allocations: mpas_soundings.F:344-358."""
        self.locations=None
        self.cells=self.owned=None


class RustSoundingIO:
    """Rust text I/O and native IO_NODE broadcast: mpas_soundings.F:112-220,261-338.

    ``broadcast_bytes(payload, root=0)`` is the driver's collective; non-root
    ranks pass None. Python transports the raw coordinate/profile buffers.
    """

    def __init__(self, executable=None, directory=".", *, broadcast_bytes=None, rank=0):
        import os
        import shutil
        from pathlib import Path
        self.executable=executable or os.environ.get("WOOF_MPAS_SOUNDINGS_BRIDGE") or shutil.which("rw_mpas_soundings")
        self.directory=Path(directory)
        self.broadcast_bytes=broadcast_bytes
        self.rank=rank

    def _run(self, *args, payload=None):
        import subprocess
        if self.executable is None:
            raise RuntimeError("rw_mpas_soundings is missing: build the rw-mpas sounding bridge")
        return subprocess.run([str(self.executable),*map(str,args)],input=payload,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout

    def _broadcast(self, payload, dminfo):
        callback=self.broadcast_bytes if dminfo is None else dminfo.bcast_bytes
        if callback is None:
            if self.rank != 0:
                raise ValueError("distributed sounding setup requires the driver's byte broadcast")
            return payload
        return callback(payload,root=0)

    def _rank(self, dminfo):
        return self.rank if dminfo is None else dminfo.my_proc_id

    def exists(self, path, *, dminfo=None):
        """INQUIRE and broadcast precede alarm creation: mpas_soundings.F:112-130."""
        packet=self._run("probe",self.directory/path) if self._rank(dminfo)==0 else None
        return self._broadcast(packet,dminfo).strip()==b"true"

    def read_locations(self, path, *, dminfo=None):
        """Count/rewind/internal READ on IO_NODE: mpas_soundings.F:132-220."""
        import json
        from types import SimpleNamespace
        packet=self._run("read",self.directory/path) if self._rank(dminfo)==0 else None
        packet=self._broadcast(packet,dminfo)
        size=int.from_bytes(packet[:4],"little")
        metadata=json.loads(packet[4:4+size])
        if not metadata["exists"] or not metadata["opened"]:
            return None
        import numpy as np
        import cupy as cp
        ns=len(metadata["names"])
        coordinates=packet[4+size:]
        if len(coordinates)!=8*ns:
            raise ValueError("Rust sounding coordinates have inconsistent byte extents")
        # Binary views/host-device transfer only; Rust performed the decoding.
        return SimpleNamespace(names=metadata["names"],names_hex=metadata.get("names_hex",[]),coordinates=coordinates,
            latitudes=cp.asarray(np.frombuffer(coordinates[:4*ns],dtype="<f4")),
            longitudes=cp.asarray(np.frombuffer(coordinates[4*ns:],dtype="<f4")))

    def write_soundings(self, time, locations, owned, rows):
        """Transfer native words to Rust's replacement writer: mpas_soundings.F:261-338."""
        import json
        metadata=json.dumps(dict(exists=True,opened=True,names=locations.names,names_hex=getattr(locations,"names_hex",[]),levels=rows.shape[1])).encode()
        payload=(len(metadata).to_bytes(4,"little")+metadata+locations.coordinates+
            owned.tobytes(order="F")+rows.tobytes(order="F"))
        self._run("write",time,self.directory,payload=payload)


class RustSoundingClock:
    """Alarm adapter backed by MPAS Gregorian arithmetic in rw-mpas.

    mpas_soundings.F:125-126,251-257,340-340. The model supplies now(); Rust
    handles all date/interval processing and mpas_timekeeping.F:490-566,876-985.
    """

    def __init__(self, now, io=None):
        self._now = now
        self.io = io if io is not None else RustSoundingIO()
        self.alarms = {}

    def now(self):
        return self._now()

    def _query(self, now, interval, previous=None, reset=False):
        import json
        request = dict(now=now, interval=interval, previous=previous, reset=reset)
        return json.loads(self.io._run("clock", payload=json.dumps(request).encode()))

    def add_alarm(self, name, start, interval):
        if name in self.alarms:
            return
        result = self._query(start, interval)
        self.alarms[name] = dict(interval=interval, previous=result["previous"])

    def is_ringing(self, name):
        alarm = self.alarms.get(name)
        if alarm is None:
            return False
        return self._query(self.now(), **alarm)["ringing"]

    def reset_alarm(self, name):
        alarm = self.alarms.get(name)
        if alarm is not None:
            result = self._query(self.now(), reset=True, **alarm)
            alarm["previous"] = result["previous"]
