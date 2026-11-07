"""Device-resident WRF SFDDAGD operator, for composition into a CUDA graph.

This module deliberately does not admit a namelist selector. The forecast
driver must supply native cadence, ramp factors, PBL weights and halo data,
then consume the returned tendencies at its physics stage. FASDAS additionally
needs the native Noah evaporation/soil coupling; these buffers alone do not
implement grid_sfdda=2. There is no CPU execution backend.
"""
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SurfaceCoefficients:
    dx: float
    rinblw: float
    guv_sfc: float
    gt_sfc: float
    gq_sfc: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(value):
                raise ValueError(f"surface FDDA {name} must be finite")
        if self.dx <= 0 or self.rinblw < 0.001:
            raise ValueError("surface FDDA requires dx > 0 m and rinblw >= 0.001 km")
        if min(self.guv_sfc,self.gt_sfc,self.gq_sfc) < 0:
            raise ValueError("surface FDDA coefficients must be nonnegative")


_MODEL = ("u", "v", "theta", "temperature", "qv", "pressure", "z")
_MASS = ("ht", "regime", "znt", "odis_old", "odis_new")


def _strict(requested):
    from woof.numerics import mode
    active = mode() == "strict"
    if requested is not None and (type(requested) is not bool or requested != active):
        raise ValueError("surface FDDA numerics must match GPUWM_NUMERICS, set before CUDA compilation")
    return active


def _module(strict):
    import cupy as cp
    source = Path(__file__).with_name("surface.cu").read_text()
    if strict:
        header = Path(__file__).parents[1] / "core/kernels/glibc_flt32.cuh"
        source = "#define SURFACE_FDDA_STRICT\n" + header.read_text() + source
    options = (("--std=c++17","--fmad=false","--ftz=false",
                "--prec-div=true","--prec-sqrt=true") if strict else
               ("--std=c++17",))
    module = cp.RawModule(code=source, options=options)
    from woof.certify.kernel_manifest import record_module
    record_module("woof.spectral_nudging.surface:" + ("strict" if strict else "fast"),
                  source=source, options=options, module=module)
    return module


class SurfaceOperator:
    """Own only scratch storage; model/forcing/control buffers stay live.

    Native common-grid storage includes the high x/y boundary. Models and
    weights use (level,y,x); endpoints use (U10,V10,TH2,Q2,y,x) collapsed
    to (4,y,x). U10/V10 have already been placed at WRF's staggered points.
    wpbl is (4,level,y,x), wzfac is (4,level), and time_factors is a device
    vector containing (interpolation coefficient, ramp factor). Every buffer
    is contiguous float32 on the current device. No buffer is copied here.

    Call prepare_density() when observation-distance endpoints change, at
    a forcing seam. Call record() while recording the caller's physics graph.
    No Python callback or host synchronization is needed to replay that graph.
    Boundary tendency words retain their incoming values, as in SFDDAGD.
    The driver owns initialization and the held-tendency cadence semantics.
    """

    def __init__(self, coefficients, fields, *, fasdas=False, strict=None):
        import cupy as cp

        if not isinstance(coefficients, SurfaceCoefficients):
            raise TypeError("coefficients must be SurfaceCoefficients")
        if type(fasdas) is not bool:
            raise TypeError("fasdas must be bool")
        strict = _strict(strict)
        if "u" not in fields or getattr(fields["u"], "ndim", 0) != 3:
            raise ValueError("surface FDDA needs a three-dimensional common grid")
        self.nz,self.ny,self.nx = fields["u"].shape
        if min(self.nx,self.ny) < 3 or self.nz < 1:
            raise ValueError("surface FDDA needs at least 3 x 3 common-grid points")
        shape = (self.nz,self.ny,self.nx)
        surface = (self.ny,self.nx)
        shapes = {**{name:shape for name in _MODEL},
                  **{name:surface for name in _MASS},
                  "old_analysis":(4,*surface), "new_analysis":(4,*surface),
                  "wpbl":(4,*shape), "wzfac":(4,self.nz), "time_factors":(2,),
                  "tendencies":(4,*shape), "sda_hfx":surface, "sda_qfx":surface}
        device = cp.cuda.runtime.getDevice()
        for name,expected in shapes.items():
            value = fields.get(name)
            if (not isinstance(value,cp.ndarray) or value.dtype != cp.float32
                    or value.shape != expected or not value.flags.c_contiguous
                    or value.device.id != device):
                raise ValueError(f"surface FDDA {name} must be contiguous float32 {expected} on device {device}")
        outputs = ("tendencies","sda_hfx","sda_qfx")
        for name in outputs:
            for other in shapes:
                if name != other and cp.may_share_memory(fields[name],fields[other]):
                    raise ValueError(f"surface FDDA outputs must not alias: {name}, {other}")
        self.fields = {name:fields[name] for name in shapes}
        self.coefficients = coefficients
        self.fasdas = fasdas
        self.module = _module(strict)
        self._density,self._smooth,self._tendencies = (self.module.get_function(name)
            for name in ("sfc_density","sfc_smooth","sfc_tendencies"))
        self.raw_density = cp.empty((2,*surface),dtype=cp.float32)
        self.density = cp.empty_like(self.raw_density)
        self.grid,self.block = ((self.nx*self.ny+127)//128,),(128,)
        self._ints = tuple(np.int32(x) for x in (self.nx,self.ny,self.nz,fasdas))
        self._coeffs = tuple(np.float32(getattr(coefficients,name)) for name in
                             ("guv_sfc","gt_sfc","gq_sfc"))
        self._density_ready = False

    def prepare_density(self):
        """Record/launch the two endpoint preprocessing kernels on this stream."""
        f,c = self.fields,self.coefficients
        self._density(self.grid,self.block,(f["odis_old"],f["odis_new"],self.raw_density,
            *self._ints[:2],np.float32(c.dx),np.float32(c.rinblw)))
        self._smooth(self.grid,self.block,(self.raw_density,self.density,*self._ints[:2]))
        self._density_ready = True

    def record(self):
        """Emit one column kernel into the caller's stream or graph capture."""
        if not self._density_ready:
            raise RuntimeError("call prepare_density on this stream before recording surface FDDA")
        f = self.fields
        args = tuple(f[name] for name in (*_MODEL,"ht","regime","znt",
                                          "old_analysis","new_analysis"))
        self._tendencies(self.grid,self.block,(*args,self.density,f["wpbl"],f["wzfac"],
            f["tendencies"],f["sda_hfx"],f["sda_qfx"],*self._ints,f["time_factors"],*self._coeffs))


class PxObservationTargets:
    """PXLSM's device target calculation, independent of grid_sfdda.

    Inputs are T2/Q2/SN old and new endpoints and live model surface pressure
    (Pa). SN is snow water equivalent (mm). The scalar interpolation weight
    lives in time_factors[0], allowing the caller's device clock to update it
    inside the graph. The land model must consume all four returned targets.
    """

    def __init__(self, fields, *, strict=None):
        import cupy as cp
        strict = _strict(strict)
        self.names = ("t_old","t_new","q_old","q_new","sn_old","sn_new","psfc",
                      "t2obs","q2obs","rh2obs","snobs")
        shape = getattr(fields.get("psfc"),"shape",())
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError("PX observation targets need a nonempty mass-grid surface")
        device = cp.cuda.runtime.getDevice()
        for name in (*self.names,"time_factors"):
            value = fields.get(name)
            expected = (2,) if name == "time_factors" else shape
            if (not isinstance(value,cp.ndarray) or value.dtype != cp.float32
                    or value.shape != expected or not value.flags.c_contiguous
                    or value.device.id != device):
                raise ValueError(f"PX target {name} must be contiguous float32 {expected} on device {device}")
        for name in self.names[7:]:
            for other in (*self.names,"time_factors"):
                if name != other and cp.may_share_memory(fields[name],fields[other]):
                    raise ValueError(f"PX outputs must not alias: {name}, {other}")
        self.fields = {name:fields[name] for name in (*self.names,"time_factors")}
        self.module = _module(strict)
        self.kernel = self.module.get_function("sfc_px_targets")
        self.size = np.int32(shape[0]*shape[1])
        self.grid,self.block = ((int(self.size)+127)//128,),(128,)

    def record(self):
        """Emit PX's target kernel into the caller's stream or graph capture."""
        self.kernel(self.grid,self.block,(*(self.fields[name] for name in self.names),
                    self.size,self.fields["time_factors"]))
