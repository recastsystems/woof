"""Incremental ocean ABI driver. Field arithmetic runs in the CUDA library."""
from __future__ import annotations

import ctypes as C
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess


class _Parameters(C.Structure):
    _fields_ = [(n, C.c_int) for n in ("nx", "ny", "boundary_mask")] + [
        (n, C.c_double) for n in ("dx", "dy")] + [
        (n, C.c_float) for n in ("dry", "manning", "initial_level", "boundary_level",
                                "reference_pressure", "cfl")]


def _library():
    source = Path(__file__).parent / "kernels" / "device.cu"
    from woof.hip_backend import active,compile_shared
    if active():
        library=compile_shared(source.read_text())
    else:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            raise FileNotFoundError("nvcc is required for the surge device library")
        identity = hashlib.sha256(source.read_bytes() + source.with_name("surge.cu").read_bytes()
                                  + subprocess.check_output([nvcc, "--version"])).hexdigest()[:20]
        cache = Path(os.environ.get("WOOF_SURGE_DEVICE_CACHE", os.environ.get(
            "CUPY_CACHE_DIR", str(Path.home() / ".woof" / "cache" / "surge"))))
        cache.mkdir(parents=True, exist_ok=True)
        # A process lock prevents another ensemble member loading a partial library.
        import fcntl
        with (cache / (identity + ".lock")).open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            library = cache / ("surge-device-" + identity + ".so")
            if not library.exists():
                temporary = library.with_suffix(f".{os.getpid()}.so")
                with (cache / (identity + ".log")).open("w") as log:
                    subprocess.run([nvcc, "-O3", "-std=c++17", "--shared", "--cudart=shared", "-Xcompiler=-fPIC",
                                    "-gencode=arch=compute_80,code=compute_80", str(source),
                                    "-o", str(temporary)], check=True, timeout=180,
                                   stdout=log, stderr=subprocess.STDOUT)
                temporary.replace(library)
    native = C.CDLL(str(library))
    args = [C.POINTER(_Parameters), C.POINTER(C.c_void_p), C.c_void_p]
    native.surge_device_init.argtypes = args
    native.surge_device_step.argtypes = args + [C.c_float, C.c_int]
    native.surge_device_init.restype = native.surge_device_step.restype = C.c_int
    return native


class DeviceSurgeSolver:
    """Persistent finite-volume state on the caller's current device/stream.

    Initialization may synchronize to bound bathymetric wave speeds. Setters
    and step neither transfer fields nor synchronize. A configured velocity/
    surface-rise envelope chooses conservative fixed substeps; device counters
    report violations of CFL/positivity/finiteness at output cadence.
    """

    def __init__(self, grid, bathymetry, config):
        import cupy as cp
        from woof.ocean.interface import SurgeOutputs, validate_fields
        self.cp = cp
        grid.validate()
        validate_fields(bathymetry, grid)
        self.grid, self.bathymetry = grid, bathymetry
        self.device = cp.cuda.runtime.getDevice()
        self._inputs = {}
        for key, array in (("bed", bathymetry.elevation), ("wet", bathymetry.wet),
                           ("lat", grid.latitude), ("ca", grid.cosalpha), ("sa", grid.sinalpha)):
            self._inputs[key] = self._plane(array, key)
        allowed = {"dry", "manning", "initial_level", "boundary_level", "boundary_mask",
                   "reference_pressure", "cfl", "max_velocity", "max_surface_rise", "max_dt"}
        if set(config) - allowed:
            raise ValueError(f"unknown surge options: {sorted(set(config) - allowed)}")
        values = dict(dry=.001, manning=.025, initial_level=0., boundary_level=0.,
                      boundary_mask=15, reference_pressure=101325., cfl=.4,
                      max_velocity=100., max_surface_rise=100., max_dt=30.)
        values.update(config)
        if 'boundary_level' not in config:
            values['boundary_level'] = values['initial_level']
        if not all(math.isfinite(float(v)) for v in values.values()):
            raise ValueError("surge options must be finite scalars")
        if (values['dry'] <= 0 or values['manning'] < 0 or not 0 < values['cfl'] <= .45
                or values['max_velocity'] <= 0 or values['max_surface_rise'] < 0
                or values['max_dt'] <= 0 or int(values['boundary_mask']) != values['boundary_mask']
                or not 0 <= values['boundary_mask'] <= 15):
            raise ValueError("invalid surge wet/dry, friction, CFL, boundary or substep bounds")
        self.parameters = _Parameters(grid.nx, grid.ny, int(values['boundary_mask']), grid.dx, grid.dy,
                                     *(values[n] for n in ('dry', 'manning', 'initial_level',
                                        'boundary_level', 'reference_pressure', 'cfl')))
        # Only the constructor reads this reduction. No bathymetric/forcing
        # inspection or reduction to the host occurs in set_* or step.
        if not bool(cp.isfinite(self._inputs['bed']).all().get()):
            raise ValueError("bathymetry contains nonfinite elevations")
        depth = float(cp.maximum(-self._inputs['bed'], 0).max().get())
        speed = values['max_velocity'] + math.sqrt(9.80665 * (
            depth + abs(values['initial_level']) + values['max_surface_rise']))
        self.internal_dt = min(values['max_dt'], values['cfl'] / (speed*(1/grid.dx+1/grid.dy)))
        n = grid.nx*grid.ny
        self._state = [cp.empty(n*16, cp.uint8) for _ in range(3)]
        self._flux = [cp.empty((grid.nx+1)*grid.ny*40, cp.uint8),
                      cp.empty(grid.nx*(grid.ny+1)*40, cp.uint8)]
        self._clock = cp.empty(80, cp.uint8)
        self._theta = cp.empty(grid.shape, cp.float32)
        self._stress = cp.empty((3, *grid.shape), cp.float32)
        self._zero = cp.zeros(grid.shape, cp.float32)
        self.diagnostics = cp.zeros(4, cp.int32)  # nonfinite, negative depth, CFL, invalid forcing
        self._outputs = SurgeOutputs(cp.empty(grid.shape, cp.float32), cp.empty(grid.shape, cp.float32))
        self.forcing = self.waves = None
        self.native = _library()
        self._call(self.native.surge_device_init)

    def _plane(self, array, name):
        from woof.ocean.interface import validate_plane
        validate_plane(array, self.grid.shape, name)
        pointer = array.__cuda_array_interface__['data'][0]
        attributes = self.cp.cuda.runtime.pointerGetAttributes(pointer)
        device = attributes.device if hasattr(attributes, 'device') else attributes['device']
        if device != self.device:
            raise ValueError(f"{name} belongs to another CUDA device")
        # Binding by pointer avoids CUDA-array-interface stream synchronization
        # performed by generic array conversion. Caller owns stream ordering.
        memory = self.cp.cuda.UnownedMemory(pointer, self.grid.nx*self.grid.ny*4, array)
        return self.cp.ndarray(self.grid.shape, self.cp.float32,
                               memptr=self.cp.cuda.MemoryPointer(memory, 0))

    @property
    def outputs(self):
        return self._outputs

    def set_forcing(self, forcing):
        from woof.ocean.interface import validate_fields
        validate_fields(forcing, self.grid)
        bound = {k: self._plane(getattr(forcing, k), k) for k in ('psfc', 'tau_x', 'tau_y')}
        self._inputs.update(bound)
        self.forcing = forcing

    def set_waves(self, waves):
        from woof.ocean.interface import validate_fields
        validate_fields(waves, self.grid)
        bound = {k: self._plane(getattr(waves, k), k) for k in ('hs', 'tp', 'direction')}
        self._inputs.update(bound)
        self.waves = waves

    def _call(self, function, *args):
        if self.cp.cuda.runtime.getDevice() != self.device:
            raise ValueError("surge solver must be used on its creation device")
        names = ('bed', 'wet', 'lat', 'ca', 'sa', 'psfc', 'tau_x', 'tau_y', 'hs', 'tp', 'direction')
        arrays = [self._inputs.get(k, self._zero) for k in names] + self._state + self._flux + [
            self._clock, self._theta, self._stress, self.outputs.eta, self.outputs.inundation,
            self.diagnostics]
        pointers = (C.c_void_p*len(arrays))(*(a.data.ptr for a in arrays))
        rc = function(C.byref(self.parameters), pointers,
                      C.c_void_p(self.cp.cuda.get_current_stream().ptr), *args)
        if rc:
            raise RuntimeError(f"surge device kernel launch failed (CUDA error {rc})")

    def step(self, dt):
        if self.forcing is None:
            raise RuntimeError("set_forcing must precede surge stepping")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("surge dt must be finite and positive")
        pieces = math.ceil(dt/self.internal_dt)
        if pieces > 2147483647:
            raise ValueError("surge interval exceeds native substep counter capacity")
        self._call(self.native.surge_device_step, C.c_float(dt/pieces), pieces)


def create_solver(grid, bathymetry, *, config=None):
    """Construct gpuwm.ocean.device/v1 surge state on the current CUDA device."""
    return DeviceSurgeSolver(grid, bathymetry, config or {})
