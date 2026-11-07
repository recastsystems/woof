"""MPAS-A v8.4.2 UPP cloud fractions; all numerical work stays on device.

Arrays use MPAS Fortran shapes (levels, cells), float32, Fortran contiguous.
Only owned cells are written, and any requested UPP field computes all four.
"""
from __future__ import annotations

from woof.backend import compilation_target as _compilation_target
from woof.hex import numerics as _hex_numerics

from functools import lru_cache
from collections.abc import Mapping

FIELDS = ("cldfrac_low_UPP", "cldfrac_mid_UPP", "cldfrac_high_UPP", "cldfrac_tot_UPP")

CUDA_SOURCE = r'''
// MPAS scalar selection; mpas_pv_diagnostics.F:1281-1289 and mpas_soundings.F:262-294.
extern "C" __global__ void select_qv(int count,int ntracer,int index,const float* scalars,float* qv) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<count) qv[j]=scalars[index+ntracer*j];
}
// mpas_cloud_diagnostics.F:75-141
extern "C" __global__ void cloud_upp(
    int nc, int nz, const float* pp, const float* pb, const float* cf,
    float* low, float* mid, float* high, float* total) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    float l=0.0f, m=0.0f, h=0.0f, t=0.0f;
    for (int k=0;k<nz;++k) {
        int j=k+nz*c;
        float p=__fadd_rn(pp[j],pb[j]), q=cf[j];
        // GNU MAX selects its second argument on ties and unordered compares;
        // preserve native negative zero and NaN cloud words without arithmetic.
        if (p>=64200.0f) l=(l>q ? l:q);
        else if (p<64200.0f && p>=35000.0f) m=(m>q ? m:q);
        else if (p<35000.0f && p>=15000.0f) h=(h>q ? h:q);
        t=(t>q ? t:q);
    }
    low[c]=l; mid[c]=m; high[c]=h; total[c]=t;
}
'''



def _compile_cuda(source):
    """Load direct NVRTC output, preserving native subnormal arithmetic.

    NATIVE-DEFECT-POLICY.md: CuPy RawModule's cache path appends FTZ=true.
    """
    import cupy as cp
    return _compile_cuda_device(source, cp.cuda.Device().id)


@lru_cache(maxsize=None)
def _compile_cuda_device(source, device_id):
    import cupy as cp
    from cupy.cuda import compiler
    with cp.cuda.Device(device_id):
        binary, _ = compiler.compile_using_nvrtc(source,
            options=_hex_numerics.options(("--std=c++17", "--fmad=false", "--ftz=false",
                     "--prec-div=true", "--prec-sqrt=true")),
            arch=_compilation_target(device_id, xp=cp))
        module = cp.cuda.function.Module()
        module.load(binary)
        return module


def _array(array, dtype, ndim):
    """Reject layouts which would reinterpret MPAS levels as cells."""
    if array.dtype.name != dtype or array.ndim != ndim or not array.flags.f_contiguous:
        raise ValueError(f"MPAS diagnostic requires Fortran-contiguous {dtype} rank {ndim}")
    return array


def _qv_array(state):
    """Select native scalars(index_qv,:,:) on device, retaining RKIND words."""
    if "qv" in state:
        return _array(state["qv"], "float32", 2)
    import cupy as cp
    scalars=_array(state["scalars"], "float32", 3)
    ntracer,nz,nc=scalars.shape
    index=int(state["index_qv"])-1
    if not 0 <= index < ntracer:
        raise ValueError("MPAS index_qv lies outside scalar dimension")
    qv=cp.empty((nz,nc),dtype=cp.float32,order="F")
    kernel=_compile_cuda(CUDA_SOURCE).get_function("select_qv")
    if nz*nc:
        kernel(((nz*nc+127)//128,),(128,),
            (cp.int32(nz*nc),cp.int32(ntracer),cp.int32(index),scalars,qv))
    return qv

class CloudDiagnostics:
    """Pool bindings from mpas_cloud_diagnostics.F:38-55."""

    def __init__(self, mesh=None, diag=None, diag_physics=None):
        self.mesh, self.diag, self.diag_physics = mesh, diag, diag_physics
        self.utils = None
        self._module = None

    def setup(self, pools, clock, utils):
        """Bind native pools: mpas_cloud_diagnostics.F:38-55."""
        self.mesh, self.diag, self.diag_physics = (
            pools[name] for name in ("mesh", "diag", "diag_physics"))
        self.clock, self.utils = clock, utils

    def compute(self, requested=None):
        """Output demand and owned-cell computation: mpas_cloud_diagnostics.F:75-141."""
        if requested is None:
            if self.utils is None:
                raise ValueError("cloud diagnostics require stream queries or explicit requested fields")
            requested = [name for name in FIELDS if self.utils.field_will_be_written(name)]
        if not set(requested).intersection(FIELDS):
            return
        import cupy as cp
        pp = _array(self.diag["pressure_p"], "float32", 2)
        pb = _array(self.diag["pressure_base"], "float32", 2)
        cf = _array(self.diag_physics["cldfrac"], "float32", 2)
        if pp.shape != pb.shape or pp.shape != cf.shape:
            raise ValueError("cloud pressure and cloud-fraction shapes differ")
        nc, nz = int(self.mesh["nCellsSolve"]), int(self.mesh["nVertLevels"])
        if pp.shape[0] != nz or not 0 <= nc <= pp.shape[1]:
            raise ValueError("cloud pool dimensions differ from array extents")
        outputs = tuple(_array(self.diag[name], "float32", 1) for name in FIELDS)
        if any(a.size < nc for a in outputs):
            raise ValueError("cloud output omits owned cells")
        if self._module is None:
            self._module = _compile_cuda(CUDA_SOURCE)
        if nc:
            self._module.get_function("cloud_upp")(((nc+127)//128,), (128,),
                (cp.int32(nc), cp.int32(nz), pp, pb, cf, *outputs))


class _NativeDiagnosticPool(Mapping):
    """Native pool/device transport; mpas_cloud_diagnostics.F:38-55."""

    def __init__(self, owner, pool, metadata=()):
        """Retain native pool names; mpas_cloud_diagnostics.F:38-55."""
        self.owner, self.pool = owner, pool
        self.metadata = dict(metadata)
        self.arrays = {}

    def __iter__(self):
        """Enumerate bound fields; mpas_cloud_diagnostics.F:38-55."""
        return iter(dict.fromkeys((*self.metadata,
            *(name for pool, name in self.owner.native._fields if pool == self.pool))))

    def __len__(self):
        """Count bound fields; mpas_cloud_diagnostics.F:38-55."""
        return sum(1 for _ in self)

    def __getitem__(self, name):
        """Copy ABI words without arithmetic; mpas_cloud_diagnostics.F:38-55."""
        if name in self.metadata:
            return self.metadata[name]
        if name not in self.arrays:
            self.arrays[name] = self.owner.cp.asarray(
                self.owner.native.get_field(self.pool, name, level=1), order="F")
        return self.arrays[name]

    def publish(self, names):
        """Return changed pool storage; mpas_pv_diagnostics.F:1575-1648."""
        for name in names:
            if name in self.arrays:
                self.owner.session.put(self.pool, name, self.arrays[name], level=1)


class NativeCloudPVSoundingsOutput:
    """Thin native-forecast adapter for all three CUDA diagnostic modules.

    mpas_atm_diagnostics_manager.F:31-63,104-126. The forecast owner computes
    solve/output diagnostics and retains the native per-scheme tendencies
    before compute(). This adapter copies their words; it derives no substitute
    geometry, velocities or tendencies on the host.
    """

    def __init__(self, session, streams, directory, device, *, clock=None,
                 dminfo=None, max_int=None, distributed=False, sounding_io=None):
        """Bind complete native pools; mpas_atm_diagnostics_manager.F:31-63."""
        from .pv_v842 import PVDiagnostics, PV_FIELDS, BUDGET_FIELDS
        from .soundings_v842 import Soundings, RustSoundingClock, RustSoundingIO
        self.session, self.native, self.layout = session, session.native, session.layout
        self.streams, self.cp = tuple(streams), device
        self.pv_fields, self.budget_fields = PV_FIELDS, BUDGET_FIELDS
        self.fields = FIELDS + PV_FIELDS + BUDGET_FIELDS
        self.pools = {name: _NativeDiagnosticPool(self, name) for name in
                      ("mesh", "state", "diag", "diag_physics", "tend", "tend_physics")}
        self.pools["mesh"].metadata.update(self.layout.dimensions)
        self.pools["state"].metadata.update({name: value for (pool, name), value in
                                            self.layout.indices.items() if pool == "state"})
        self.cloud = CloudDiagnostics()
        self.pv = PVDiagnostics(exchange=self.exchange, max_int=max_int,
                                distributed=distributed)
        io = sounding_io if sounding_io is not None else RustSoundingIO(directory=directory)
        self.clock = clock if clock is not None else RustSoundingClock(
            lambda: self.native.get_text_field("state", "xtime", level=1), io)
        self.soundings = Soundings(io=io)
        self.dminfo = dminfo
        self.cloud.setup(self.pools, self.clock, self)
        self.pv.setup(self.pools, self.clock, self)
        self.initialized = False
        self.calls = []

    def field_will_be_written(self, name):
        """Query live stream alarms; mpas_cloud_diagnostics.F:99-102."""
        return any(name in stream.fields and "output" in stream.directions
                   and stream.output_interval != "none"
                   and self.native.alarm_ringing(stream.name + ":out")
                   for stream in self.streams)

    def attach(self, missing):
        """Attach only produced Registry fields; mpas_atm_core.F:675-705."""
        for stream in self.streams:
            for name in tuple(missing[stream.name]):
                if name in self.fields:
                    self.native.stream_field(stream.name, name)
                    missing[stream.name].remove(name)

    def exchange(self, name, array):
        """Use the native field halo collective; mpas_pv_diagnostics.F:1253-1324."""
        if array is None:
            return
        pool = next(pool for pool, views in self.pools.items()
                    if views.arrays.get(name) is array)
        registry_name = next(registry_name for registry_name, key in
                             self.native._registered.items() if key == (pool, name))
        key = name
        self.session.put(pool, key, array, level=1)
        self.native.exchange_halo(registry_name, level=1)
        self.native.get_field(pool, key, level=1, out=array)

    def compute(self, step):
        """Compute before stream writes; mpas_atm_diagnostics_manager.F:104-126."""
        for pool in self.pools.values():
            pool.arrays.clear()
        if not self.initialized:
            self.soundings.setup(self.layout.configs, self.pools, self.clock, self.dminfo)
            self.initialized = True
        requested = tuple(name for name in self.fields if self.field_will_be_written(name))
        self.cloud.compute(requested)
        self.pv.compute(requested)
        self.soundings.compute()
        self.pools["diag"].publish((*self.fields, "theta", "rho", "dtheta_dt_mix", "tend_u_phys"))
        # The native budget intentionally uncouples w_euler in place.
        if set(requested).intersection(self.budget_fields):
            self.pools["tend"].publish(("w_euler",))
        self.calls.append(dict(step=step, fields=requested,
                               soundings_initialized=self.soundings.locations is not None))

    def cleanup(self):
        """Release station/device storage; mpas_soundings.F:344-358."""
        self.soundings.cleanup()
        for pool in self.pools.values():
            pool.arrays.clear()
