"""RRFS-MPAS pools, clock and module storage with no native library.

The GSL v8.3.1 fork RRFSx/MPAS-Model 1d6e370e (pins in data/mpasphys-native/mpas-source-rrfs.json) on the
ported framework of the plain MPAS route: the same storage, halo, clock and allocation code
(framework_backend_v842.FrameworkPhysics), driven by the fork's own declarations, generated from the pinned
fork source by tools/framework_metadata_rrfs.py into data/mpasphys-framework-rrfs/ (656 module variables,
12 allocation sets, 61 constant words, the fork Registry expanded with Registry_tempo.xml and
registry.chemistry.xml). Nothing here loads libmpasphys_native_rrfs.so or a Fortran runtime.

MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_vars.F (module storage),
mpas_atmphys_driver_*.F allocate_* routines (allocation sets), mpas_atmphys_manager.F:137-252 (calendar).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..framework_backend_v842 import FrameworkPhysics

FORK_PIN = '1d6e370e7b43a332074809f170532a216c2212fb'
DATA = Path(__file__).resolve().parents[2] / 'data' / 'mpasphys-framework-rrfs'


def registry_source(*, data=None):
    """Installed fork Registry, rather than a downloaded fork source tree."""
    return DATA if data is None else Path(__file__).resolve().parents[2] / data


def table_source(*, pin_set='rrfs', resource='mpas-source-rrfs'):
    """The fork tree whose physics_wrf/files holds the lookup files the RRFS route opens by name.

    The wheel carries the fork Registry (``registry_source``) but not its tables: about 1 GB (the TEMPO v3 lookup
    files, ugwp_limb_tau.nc, ccn_activate.bin, MPAS-Data), fetched by SHA-256 on first use from NOAA's published
    RRFS fix files and MPAS-Data (resources_build.assemble_rrfs_source, mpas-source-rrfs.json). This is the same
    tree FrameworkInitialization.tempo_lookup_path reads.
    """
    from ..resources_build import ensure_route_resources
    return ensure_route_resources((resource,), pin_set=pin_set)[resource]


class FrameworkPhysicsRRFS(FrameworkPhysics):
    """Owned Python storage for the RRFS route's MPAS framework boundary.

    ``build`` carries the fork pin the RRFS driver and scheme modules check
    (driver_rrfs.NativeDriverHooks.configure, tempo_v303). Scheme tables a
    module reads by name (``get_table``) are the ones the ported scheme
    initialization put here; an absent one reads None, as native reports an
    unallocated table, and the reading module refuses by name.
    """
    pin_set = 'rrfs'
    source_pin = FORK_PIN
    data = DATA

    def __init__(self, *, layout=None, resources=None, stream_factory=None, tables=None,
                 data=None, source_pin=FORK_PIN, pin_set='rrfs', resource='mpas-source-rrfs'):
        self.data = registry_source(data=data)
        self.source_pin, self.pin_set, self.resource = source_pin, pin_set, resource
        super().__init__(layout=layout, resources=resources, stream_factory=stream_factory)
        # The fork's lookup files (TEMPO v3, ccn_activate.bin); None resolves the fetched mpas-source-rrfs DATA.
        self.tables = tables
        if self.abi['source_pin'].get('rrfs-model') != self.source_pin:
            raise ValueError('RRFS framework metadata is not from the pinned fork ' + self.source_pin)
        self.scheme_state = {}
        self.build = dict(source_pin=self.source_pin, pin_set=self.pin_set, framework='ported',
                          pin_set_sha256=None, metadata=str(self.data / 'metadata.json'))
        self.path = None

    def get_table(self, name):
        self._check()
        return self._tables.get(name)

    def put_table(self, name, value, *, lower=None, copy=True):
        """Module table words owned by a ported scheme initializer (native kind and memory order).

        ``copy=False`` keeps a read-only Fortran-ordered array as is (the TEMPO lookup tables are ~1 GB).
        """
        self._check()
        a = np.asarray(value.get() if hasattr(value, 'get') else value)
        lower = tuple(lower) if lower is not None else (1,) * a.ndim
        # A module scalar can read back as one word with no bounds, as the native table getter returns it.
        if len(lower) != a.ndim and not (a.shape == (1,) and lower == ()):
            raise ValueError('MPAS table lower bound rank differs')
        keep = not copy and (a.ndim == 0 or a.flags.f_contiguous)
        self._tables[name] = (a if keep else np.array(a, order='F', copy=True)), lower

    def _allocation_local(self, name, value):
        # mpas_atmphys_driver_smoke.F:64 (allocate_smoke): nblocks = 24 / bb_input_prevh, a local.
        if name == 'allocate_smoke' and value == 'nblocks':
            hours = int(self._config_values['configs', 'bb_input_prevh'])
            if hours <= 0:
                raise ValueError('bb_input_prevh must be positive: allocate_smoke divides 24 by it')
            return 24 // hours
        return super()._allocation_local(name, value)

    def metadata_sha256(self):
        import hashlib
        return hashlib.sha256((self.data / 'metadata.json').read_bytes()).hexdigest()

    def describe(self):
        return json.dumps(dict(pin_set=self.pin_set, source_pin=self.source_pin,
                               variables=len(self.abi['variables']),
                               allocations=len(self.abi['allocations']),
                               constants=len(self.abi['constants'])))


# The names the RRFS row's framework entry binds (route_table: woof.hex.mpasphys.rrfs.framework_rrfs:FrameworkPhysics).
FrameworkPhysics = FrameworkPhysicsRRFS
