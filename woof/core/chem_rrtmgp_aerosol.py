"""Pinned MERRA aerosol optics, float32 like the engine's radiation kernels.

All nz mass levels are processed. Inputs use (nz, ny, nx), x fastest.
Outputs use (ncolumn, nz, nband), matching RRTMGP. No state or driver hook.
The tables are Rust decoded and cached on host and separately per CUDA device.
Workspace per call: four output cubes and one int32 validation field.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import hashlib
import numpy as np
from woof.netcdf_bridge import Dataset

TABLE_PINS = {'lw': '2c39dace5a730d1ee0a6007b4c972043b16168f58cb184def9bc949853d7154b', 'sw': '7885b7058f1f536163a1f8e5ca97d13783a353310a44974364788deb7abf6e11'}

@dataclass
class AerosolTables:
    kind: str
    packed: np.ndarray
    limits: np.ndarray
    rh: np.ndarray
    bands: np.ndarray
    _devices: dict = field(default_factory=dict, repr=False)

    def to_device(self):
        import cupy as cp
        key = cp.cuda.runtime.getDevice()
        if key not in self._devices:
            self._devices[key] = tuple(cp.asarray(x) for x in
                                      (self.packed, self.limits, self.rh))
        return self._devices[key]

@lru_cache(maxsize=2)
def load_tables(kind, root=None):
    if kind not in ('lw', 'sw'):
        raise ValueError('kind must be lw or sw')
    if root is None:
        from woof import data_assets
        root = data_assets.rrtmgp_data_dir()
    path = Path(root) / f'rrtmgp-aerosols-merra-{kind}.nc'
    from woof.tables import table_path
    path = table_path("rrtmgp-aerosol", path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != TABLE_PINS[kind]:
        raise ValueError(f'aerosol table hash mismatch: {path}')
    with Dataset(path) as nc:
        def read(name, dims):
            v = nc.variables[name]
            if set(v.dimensions) != set(dims):
                raise ValueError(f'{name}: unexpected dimensions {v.dimensions}')
            return np.ascontiguousarray(np.transpose(v[:],
                         tuple(v.dimensions.index(d) for d in dims)), dtype=np.float32)
        limits = read('merra_aero_bin_lims', ('nbin', 'pair'))
        rh = read('aero_rh', ('nrh',))
        bands = read('bnd_limits_wavenumber', ('nband', 'pair'))
        packed = np.empty((7, len(bands), 5, len(rh), 3), np.float32)
        for t, name in enumerate(('dust', 'salt', 'sulf', 'bcar_rh',
                                  'bcar', 'ocar_rh', 'ocar')):
            dims = ['nband']
            if t in (0, 1): dims += ['nbin']
            if t in (1, 2, 3, 5): dims += ['nrh']
            dims += ['nval']
            a = read(f'aero_{name}_tbl', tuple(dims))
            if t not in (0, 1): a = np.expand_dims(a, 1)
            if t not in (1, 2, 3, 5): a = np.expand_dims(a, 2)
            packed[t] = a
    if limits.shape != (5, 2) or len(rh) < 2 or np.any(np.diff(rh) <= 0):
        raise ValueError('invalid MERRA size or RH grid')
    return AerosolTables(kind, packed, limits, rh, bands)

@dataclass
class RowParameters:
    indices: tuple
    types: np.ndarray
    sizes: np.ndarray
    scales: np.ndarray

def pack_rows(rows, tables):
    """Row constants are FP32, the selected RTE wp. No species-name tests.

    merra_bin is one based. For a prescribed bin the interior midpoint avoids
    the source's shared-edge rule, where the last matching bin wins.
    merra_radius_um overrides this for an explicitly chosen representative radius.
    mass_scale is kg aerosol / kg dry air per unit row mixing ratio. It must
    document molecular or carbon-to-matter conversions in the owning row.
    """
    indices, types, sizes, scales = [], [], [], []
    for i, row in enumerate(rows):
        optics = row.get('optics') or {}
        if optics.get('merra_type') is None: continue
        t = optics['merra_type']; b = optics.get('merra_bin', 1)
        if not isinstance(t, int) or not 1 <= t <= 7:
            raise ValueError('merra_type must be an integer in 1..7')
        if not isinstance(b, int) or not 1 <= b <= 5:
            raise ValueError('merra_bin must be an integer in 1..5')
        size = optics.get('merra_radius_um', float(np.mean(tables.limits[b-1])))
        scale = optics.get('merra_mass_scale')
        if scale is None:
            if row.get('units') != 'ug kg-1':
                raise ValueError('optical row needs merra_mass_scale for its units')
            scale = 1.e-9
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('merra_mass_scale must be finite and positive')
        if not np.isfinite(size) or not tables.limits[0,0] <= size <= tables.limits[-1,1]:
            raise ValueError('merra_radius_um is outside the MERRA size range')
        indices.append(i); types.append(t); sizes.append(size); scales.append(scale)
    return RowParameters(tuple(indices), np.asarray(types, np.int32),
                         np.asarray(sizes, np.float32), np.asarray(scales, np.float32))

def _input(a, dtype, shape=None):
    import cupy as cp
    if not isinstance(a, cp.ndarray) or a.dtype != dtype or not a.flags.c_contiguous:
        raise TypeError('input must be a contiguous CuPy array of the required kind')
    if a.ndim != 3 or (shape is not None and a.shape != shape):
        raise ValueError('inputs must share (nz, ny, nx) shape')
    return a

def _outputs(shape, nb):
    import cupy as cp
    nz, ny, nx = shape
    outputs = tuple(cp.empty((ny*nx, nz, nb), cp.float32) for _ in range(4))
    return outputs, cp.empty(shape, cp.int32)

def optics(tables, types, sizes, mass, rh):
    """Single-component Fortran interface. mass is kg/m2, radius is microns.

    Returns tau, ssa, g, absorption_tau. No mass sign check is added to the
    source routine. Nonfinite inputs are refused before its normal range checks.
    """
    from woof.core.kernels import get_kernel
    import cupy as cp
    _input(types, cp.int32)
    for a in (sizes, mass, rh):
        _input(a, cp.float32, types.shape)
        if not bool(cp.all(cp.isfinite(a))): raise ValueError('nonfinite aerosol input')
    outputs, status = _outputs(types.shape, len(tables.bands))
    nz, ny, nx = types.shape; nc = ny*nx
    get_kernel('chem_rrtmgp_aerosol', 'merra_optics')(
        ((nc+127)//128,), (128,), (types, sizes, mass, rh, *tables.to_device(),
         np.int32(len(tables.rh)), np.int32(len(tables.bands)), np.int32(nz),
         np.int32(nc), *outputs, status))
    if bool(cp.any(status)):
        raise ValueError('aerosol validity failure: type=1 size=2 RH=4')
    return outputs

def launch(rows, fields, dry_pressure_thickness, rh, tables, *, gravity=9.81):
    """Mix row optical properties in row order using RTE increment semantics.

    Column mass = (mixing ratio * merra_mass_scale) * (dry_dp / gravity).
    dry_dp is the engine's DRY pressure thickness in Pa. Do not pass total-air
    dp without first removing moisture mass. gravity uses the caller's engine
    value. Each field is an independent FP32 pointer; absent optics are skipped.
    Returns tau, ssa, g, absorption_tau. The lead applies SW delta scaling.
    """
    from woof.core.kernels import get_kernel
    import cupy as cp
    _input(dry_pressure_thickness, cp.float32)
    _input(rh, cp.float32, dry_pressure_thickness.shape)
    if len(rows) != len(fields): raise ValueError('one field is required per row')
    if not np.isfinite(gravity) or gravity <= 0: raise ValueError('invalid gravity')
    pars = pack_rows(rows, tables)
    selected = [_input(fields[i], cp.float32, rh.shape) for i in pars.indices]
    pointers = cp.asarray([a.data.ptr for a in selected], dtype=cp.uint64)
    params = tuple(cp.asarray(a) for a in (pars.types, pars.sizes, pars.scales))
    outputs, status = _outputs(rh.shape, len(tables.bands))
    nz, ny, nx = rh.shape; nc = ny*nx
    get_kernel('chem_rrtmgp_aerosol', 'merra_rows')(
        ((nc+127)//128,), (128,), (pointers, *params, np.int32(len(selected)),
         dry_pressure_thickness, rh, np.float32(gravity), *tables.to_device(),
         np.int32(len(tables.rh)), np.int32(len(tables.bands)), np.int32(nz),
         np.int32(nc), *outputs, status))
    if bool(cp.any(status)): raise ValueError('invalid dry dp, RH or optical row mass')
    return outputs
