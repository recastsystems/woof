"""Resident batched binary64 polar filter from WRF's public equations.

Plans and workspaces are owned by a domain, so separate ensemble streams do
not replace shared pointers or synchronize through host reads. Calls enqueue
pack, FFT, the native float32 spectral weights, inverse FFT and writeback.
"""
import numpy as np


class PoleFaces:
    """The two native v-face prescriptions; no meridional periodic wrap.

    WRF zeroes the polar faces before RK preparation and copies the adjacent
    face after physical boundary processing. These are distinct placements;
    this operator alone does not provide the polar flux boundary equations.
    """
    def __init__(self, nx, ny, nz):
        from woof.core.kernels import load_module_no_ftz
        if any(type(v) is not int or v < 1 for v in (nx, ny, nz)) or ny < 2:
            raise ValueError("pole faces require nx/nz>=1 and at least two mass rows")
        self.shape = (nz, ny+1, nx)
        self.args = tuple(np.int32(v) for v in (nx, ny+1, nz))
        self.grid = ((nx*nz+255)//256,)
        self.kernel = load_module_no_ftz("global_polar", no_fmad=True).get_function("polar_meridional_faces")

    def apply(self, field, *, copy_neighbor=False):
        import cupy as cp
        if (not isinstance(field, cp.ndarray) or field.dtype != cp.float32
                or field.shape != self.shape or not field.flags.c_contiguous):
            raise ValueError(f"pole field must be packed resident float32 {self.shape}")
        if type(copy_neighbor) is not bool:
            raise ValueError("copy_neighbor must be a logical")
        self.kernel(self.grid, (256,), (*self.args, np.int32(copy_neighbor), field))


class PolarFFT:
    def __init__(self, nx, ny, nz, *, latitude, filter_lat=45.0, dvlat=0.0,
                 last_row=None, staggered_x=False, allocator=None):
        import cupy as cp
        from cupy.cuda import cufft
        from woof.core.kernels import load_module_no_ftz
        if any(type(v) is not int or v < 1 for v in (nx, ny, nz)) or nx < 2:
            raise ValueError("polar FFT requires nx>=2 and positive ny/nz")
        if (not isinstance(latitude, cp.ndarray) or latitude.dtype != cp.float32
                or latitude.shape != (ny, nx) or not latitude.flags.c_contiguous):
            raise ValueError("latitude must be packed resident float32 (ny,nx)")
        if not 0 <= filter_lat <= 90:
            raise ValueError("fft_filter_lat must lie in [0,90]")
        last = ny-1 if last_row is None else last_row
        if type(last) is not int or not -1 <= last < ny:
            raise ValueError("last filtered row lies outside the domain")
        self.shape = (nz, ny, nx+int(staggered_x))
        self.dims = tuple(np.int32(v) for v in (nx, ny, nz, self.shape[2]))
        self.module = load_module_no_ftz("global_polar", no_fmad=True)
        allocate = (lambda shape, name, dtype: cp.empty(shape, dtype)) if allocator is None else allocator
        self.weights = allocate((ny, nx//2+1), 'weights', cp.float32)
        self.active = allocate((ny,), 'active', cp.int32)
        self.work = allocate((ny*nz, nx), 'work', cp.float64)
        self.spectrum = allocate((ny*nz, nx//2+1), 'spectrum', cp.complex128)
        self.forward = cufft.Plan1d(nx, cufft.CUFFT_D2Z, ny*nz)
        self.backward = cufft.Plan1d(nx, cufft.CUFFT_Z2D, ny*nz)
        self.module.get_function("polar_weights")(
            ((self.weights.size+255)//256,), (256,),
            (np.int32(nx), np.int32(ny), np.int32(last), np.float32(filter_lat),
             np.float32(dvlat), latitude, self.weights, self.active))

    def apply(self, field):
        import cupy as cp
        from cupy.cuda import cufft
        if (not isinstance(field, cp.ndarray) or field.dtype != cp.float32
                or field.shape != self.shape or not field.flags.c_contiguous):
            raise ValueError(f"polar field must be packed resident float32 {self.shape}")
        self.module.get_function("polar_pack")(
            ((self.work.size+255)//256,), (256,), (*self.dims, field, self.work))
        self.forward.fft(self.work, self.spectrum, cufft.CUFFT_FORWARD)
        self.module.get_function("polar_scale")(
            ((self.spectrum.size+255)//256,), (256,),
            (*self.dims[:3], self.weights, self.spectrum))
        self.backward.fft(self.spectrum, self.work, cufft.CUFFT_INVERSE)
        self.module.get_function("polar_unpack")(
            ((self.work.size+255)//256,), (256,),
            (*self.dims, self.active, self.work, field))
