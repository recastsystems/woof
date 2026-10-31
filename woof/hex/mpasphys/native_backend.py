"""Explicit test-only native MPAS driver boundary with pinned library selection.

MPAS source: src/core_atmosphere/physics/mpas_atmphys_driver.F:179-384.
Registry arrays retain MPAS/Fortran dimension order and include local halos.
The library owns copied pool storage; Python only orchestrates transfers and calls.
Native drivers retain their original from/to-MPAS conversions and module state.
"""
from __future__ import annotations

import ctypes as ct
from functools import partial
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Mapping

import numpy as np

_LOCK = threading.RLock()
_ACTIVE: NativePhysics | None = None
_PIN_SETS = {
    'v8.4.2': ('libmpasphys_native.so', 'f34984b2c89353cf31c235b08d3b4acd96b37bb3'),
    'rrfs': ('libmpasphys_native_rrfs.so', '1d6e370e7b43a332074809f170532a216c2212fb'),
    'rrfs-44c0849': ('libmpasphys_native_rrfs.so', '44c084944e17f7098668bef6d1e353d12e3c2d1a'),
}
_RRFS_PIN_SET_SHA256 = 'd0aa837b898350a86dd01ddbe15b564aaa3db2280180ec7a5f33937863559d09'
_PIN_SET_DIGESTS = {'rrfs': _RRFS_PIN_SET_SHA256,
                    'rrfs-44c0849': '12e8a370924f5743901e3d149b67d4a785ed17ac09f784033223c32cb889bd03'}
_KINDS = {np.dtype('float32'): 'real', np.dtype('int32'): 'integer'}


class _RealArray2(ct.Structure):
    _fields_ = [('data', ct.c_void_p), ('shape', ct.c_int32 * 2)]


_CTYPES = {'handle': ct.c_void_p, 'int32': ct.c_int32, 'float32': ct.c_float,
           'string': ct.c_char_p, 'bool': ct.c_bool, 'real_array2': _RealArray2}


# Native binding/orchestration authority: src/framework/mpas_kind_types.F:32.
def _name(value: str) -> bytes:
    encoded = value.encode('ascii')
    if not encoded or b'\x00' in encoded or len(encoded) >= 512:
        raise ValueError('MPAS names must contain 1..511 ASCII bytes without NUL')
    return encoded


# Native binding/orchestration authority: src/framework/mpas_pool_routines.F:2108-2370.
def _host(value: Any) -> np.ndarray:
    # CuPy's get() is the seam transfer; importing this module never opens a GPU.
    result = value.get() if hasattr(value, 'get') else value
    return np.asarray(result)


class NativePhysics:
    """One serialized native physics instance, with one entry for each native driver.

    The build directory contains libmpasphys_native.so, abi.json and its build
    receipt. A directory selects the library for the explicit pin_set; a file
    path still verifies the requested source pin. Construction requires an explicit test_only=True. Scientific
    acceptance is recorded by f1/f2 receipts, never inferred from successful load.
    Every registered field is an MPAS rank-local array, including its halos.
    ``level`` is one-based; state time levels are separate native fields.
    """

    # Native binding/orchestration authority: src/framework/mpas_dmpar.F:272-343; src/framework/mpas_log.F:99-273.
    def __init__(self, library: str | Path, *, test_only: bool = False, pin_set: str = 'v8.4.2'):
        if not test_only:
            raise ValueError('native-cpu physics is test-only; pass test_only=True')
        if pin_set not in _PIN_SETS:
            raise ValueError(f'unknown native physics pin set: {pin_set!r}')
        filename, source_pin = _PIN_SETS[pin_set]
        global _ACTIVE
        with _LOCK:
            if _ACTIVE is not None:
                raise RuntimeError('native MPAS module state already has a live instance')
            self.pin_set = pin_set
            self.path = Path(library).resolve()
            if self.path.is_dir():
                self.path /= filename
            self.abi = json.loads(self.path.with_name('abi.json').read_text())
            self.build = json.loads(self.path.with_name('build-receipt.json').read_text())
            # A fork cannot silently execute the other model's driver signature.
            if self.build.get('source_pin') != source_pin:
                raise ValueError('native physics build does not match the requested pin set')
            if pin_set in _PIN_SET_DIGESTS:
                declared_pins = {'source_pin': self.build['source_pin'],
                                 'external_pins': self.build.get('external_pins', {})}
                pins_digest = hashlib.sha256(json.dumps(declared_pins, sort_keys=True).encode()).hexdigest()
                if pins_digest != _PIN_SET_DIGESTS[pin_set] or self.build.get('pin_set_sha256') != pins_digest:
                    raise ValueError('native physics submodules do not match the requested pin set')
            if self.abi.get('pin_set', pin_set) != pin_set:
                raise ValueError('native ABI does not match the requested pin set')
            if self.build.get('abi_sha256') != hashlib.sha256(self.path.with_name('abi.json').read_bytes()).hexdigest():
                raise ValueError('native ABI manifest does not match its build receipt')
            actual = hashlib.sha256(self.path.read_bytes()).hexdigest()
            if self.build.get('library_sha256') != actual:
                raise ValueError('native physics library does not match its build receipt')
            if self.abi['version'] != 2:
                raise ValueError('unsupported native physics ABI')
            self.lib = ct.CDLL(str(self.path), mode=ct.RTLD_LOCAL)
            self._bind('native_abi_version', [], ct.c_int)
            if self.lib.native_abi_version() != 2:
                raise ValueError('library ABI does not match the generated manifest')
            self._bind('native_pool_create', [], ct.c_void_p)
            self._bind('native_pool_destroy', [ct.c_void_p])
            self._bind('native_variables_release', [])
            self._bind('native_field_character_0', [ct.c_void_p, ct.c_char_p, ct.c_void_p, ct.c_int])
            self._bind('native_table_shape', [ct.c_int, ct.c_void_p, ct.c_void_p,
                                               ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)])
            self._bind('native_runtime_create', [], ct.c_void_p)
            self._bind('native_runtime_destroy', [ct.c_void_p])
            self._bind('native_runtime_block', [ct.c_void_p] * 6 + [ct.POINTER(ct.c_int)])
            self._bind('native_runtime_decomposition', [ct.c_void_p, ct.c_int, ct.c_int,
                                                         ct.c_int, ct.c_void_p, ct.POINTER(ct.c_int)])
            self._bind('native_runtime_group', [ct.c_void_p, ct.c_char_p, ct.POINTER(ct.c_int)])
            self._bind('native_runtime_group_field', [ct.c_void_p, ct.c_char_p, ct.c_char_p,
                                                     ct.c_int, ct.POINTER(ct.c_int)])
            self._bind('native_runtime_stream_field', [ct.c_void_p, ct.c_char_p, ct.c_char_p,
                                                        ct.POINTER(ct.c_int)])
            self._bind('native_runtime_stream_transfer', [ct.c_void_p, ct.c_char_p, ct.c_int,
                                                           ct.c_bool, ct.POINTER(ct.c_int)])
            self._bind('native_runtime_clock', [ct.c_void_p, ct.c_char_p, ct.c_char_p,
                                               ct.c_char_p, ct.c_float, ct.POINTER(ct.c_int)])
            self._bind('native_runtime_streams', [ct.c_void_p, ct.POINTER(ct.c_int)])
            self._bind('native_runtime_stream', [ct.c_void_p, ct.c_char_p, ct.c_char_p, ct.c_int,
                                                ct.c_char_p, ct.c_char_p, ct.POINTER(ct.c_int)])
            if hasattr(self.lib, 'native_runtime_stream_filename_interval'):
                self._bind('native_runtime_stream_filename_interval',
                           [ct.c_void_p, ct.c_char_p, ct.c_char_p, ct.POINTER(ct.c_int)])
            self._bind('native_runtime_advance', [ct.c_void_p, ct.POINTER(ct.c_int)])
            self._bind('native_runtime_alarm_ringing', [ct.c_void_p, ct.c_char_p,
                                                       ct.POINTER(ct.c_int)], ct.c_bool)
            self._bind('native_runtime_alarm_reset', [ct.c_void_p, ct.c_char_p, ct.POINTER(ct.c_int)])
            self._bind('native_runtime_object', [ct.c_void_p, ct.c_int], ct.c_void_p)
            self._bind('native_dimension', [ct.c_void_p, ct.c_char_p, ct.c_int, ct.c_void_p, ct.c_int])
            for kind, typ in [('real', ct.c_float), ('integer', ct.c_int),
                              ('logical', ct.c_bool), ('character', ct.c_char_p)]:
                self._bind('native_config_' + kind, [ct.c_void_p, ct.c_char_p, typ, ct.c_int])
            for name, e in self.abi['entries'].items():
                self._bind('native_' + name, [_CTYPES[x] for x in e['abi']])
            self._pools: dict[str, int] = {}
            self._fields: dict[tuple[str, str], tuple[np.dtype, tuple[int, ...], int]] = {}
            self._configs: dict[tuple[str, str], str] = {}
            self._text_fields: dict[tuple[str, str], int] = {}
            self._dimensions: dict[tuple[str, str], int] = {}
            self._dimension_values: dict[tuple[str, str], int] = {}
            self._core_dimensions = None
            self._groups: set[str] = set()
            self._registered: dict[str, tuple[str, str]] = {}
            self._decompositions: set[str] = set()
            self.closed = False
            self._runtime = self.lib.native_runtime_create()
            self._initialized = False
            _ACTIVE = self

    # Native binding/orchestration authority: src/core_atmosphere/physics/mpas_atmphys_driver.F:209-384.
    def _bind(self, name: str, args: list, result=None):
        function = getattr(self.lib, name)
        function.argtypes = args
        function.restype = result
        return function

    # Native binding/orchestration authority: src/framework/mpas_pool_routines.F:169-190,203-361.
    def _check(self):
        if self.closed:
            raise RuntimeError('native physics instance is closed')

    # Native binding/orchestration authority: src/framework/mpas_pool_routines.F:169-190.
    def pool(self, name: str) -> int:
        with _LOCK:
            self._check()
            _name(name)
            if name not in self._pools:
                handle = self.lib.native_pool_create()
                if not handle:
                    raise MemoryError('MPAS could not create pool')
                self._pools[name] = handle
            return self._pools[name]

    # Native binding/orchestration authority: src/framework/mpas_pool_routines.F:4937-5103.
    def dimension(self, pool: str, name: str, value: Any):
        with _LOCK:
            self._check()
            a = np.asarray(value, dtype=np.int32)
            if a.ndim > 1:
                raise ValueError('MPAS dimensions are scalars or vectors')
            count = a.size if a.ndim else 0
            if a.ndim and count == 0:
                raise ValueError('dimension vectors cannot be empty')
            key = (pool, name)
            if key in self._dimensions and self._dimensions[key] != count:
                raise ValueError('dimension rank or size cannot change in a native pool')
            a = np.ascontiguousarray(a)
            self.lib.native_dimension(self.pool(pool), _name(name), count,
                                      a.ctypes.data, int(key not in self._dimensions))
            self._dimensions[key] = count
            if count == 0:
                self._dimension_values[key] = int(a[0])

    # Native binding/orchestration authority: src/framework/mpas_pool_routines.F:4540-4726,4761-4933.
    def config(self, name: str, value: bool | int | float | str, *, pool: str = 'configs'):
        with _LOCK:
            self._check()
            if isinstance(value, (bool, np.bool_)):
                kind = 'logical'
            elif isinstance(value, (int, np.integer)):
                kind = 'integer'
            elif isinstance(value, (float, np.floating)):
                kind = 'real'
            elif isinstance(value, str):
                kind = 'character'
                value = value.encode('ascii')
                if len(value) > 512 or b'\x00' in value:
                    raise ValueError('MPAS config strings must fit StrKIND and contain no NUL')
            else:
                raise TypeError(f'unsupported config value {type(value)}')
            key = (pool, name)
            if key in self._configs and self._configs[key] != kind:
                raise TypeError('MPAS config type cannot change')
            getattr(self.lib, 'native_config_' + kind)(self.pool(pool), _name(name), value,
                                                      int(key not in self._configs))
            self._configs[key] = kind

    # Native binding/orchestration authority: src/framework/mpas_pool_routines.F:4761-4791.
    def get_real_config(self, name: str, *, pool: str = 'configs'):
        """Read the current native RKIND config, including native pointer updates."""
        with _LOCK:
            self._check()
            key = (pool, name)
            if key not in self._configs:
                raise KeyError(key)
            if self._configs[key] != 'real':
                raise TypeError('native config is not declared real')
            function = self._bind('native_config_get_real',
                                 [ct.c_void_p, ct.c_char_p, ct.POINTER(ct.c_float)])
            value = ct.c_float()
            function(self.pool(pool), _name(name), ct.byref(value))
            return np.float32(value.value)

    # Native binding/orchestration authority: src/framework/mpas_pool_routines.F:2108-3211,3436-4240.
    def _field_fn(self, dtype, rank):
        if dtype not in _KINDS or rank > (5 if dtype == np.dtype('float32') else 3):
            raise TypeError('MPAS fields require float32 rank 0..5 or int32 rank 0..3')
        return self._bind(f'native_field_{_KINDS[dtype]}_{rank}',
                          [ct.c_void_p, ct.c_char_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p, ct.c_int])

    # Native binding/orchestration authority: src/framework/mpas_pool_routines.F:2108-3211,3436-4240.
    def put_field(self, pool: str, name: str, value: Any, *, level: int = 1, nlevels: int = 1):
        with _LOCK:
            self._check()
            a = _host(value)
            dtype, shape = a.dtype, a.shape
            fn = self._field_fn(dtype, a.ndim)
            # Associated native Registry fields may have zero extents (for
            # example, disabled constituents); keep their rank and time levels.
            if nlevels < 1 or not 1 <= level <= nlevels or any(n < 0 for n in shape):
                raise ValueError('invalid native field shape or time level')
            key = (pool, name)
            if key in self._text_fields:
                raise TypeError('native field type cannot change')
            metadata = (dtype, shape, nlevels)
            if key in self._fields and self._fields[key] != metadata:
                raise ValueError('native field dtype, shape and time levels cannot change')
            # The native operation copies into Fortran-owned storage before
            # returning; it never retains or writes this input pointer. Reuse
            # aligned Fortran-order host storage instead of copying it twice.
            # Noncontiguous/unaligned inputs still get a suitable staging copy.
            buffer = np.require(a, requirements=['F', 'A'])
            dims = np.asarray(shape or (1,), dtype=np.int32)
            fn(self.pool(pool), _name(name), dims.ctypes.data, nlevels, level,
               buffer.ctypes.data, int(key in self._fields))
            self._fields[key] = metadata

    # Native binding/orchestration authority: src/framework/mpas_pool_routines.F:2548-2590.
    def put_text_field(self, pool: str, name: str, value: str, *, level=1, nlevels=1):
        """Registry scalar character field, e.g. land-use category mminlu."""
        with _LOCK:
            self._check()
            key = (pool, name)
            if key in self._fields:
                raise TypeError('native field type cannot change')
            if nlevels < 1 or not 1 <= level <= nlevels:
                raise ValueError('invalid native character field time level')
            if key in self._text_fields and self._text_fields[key] != nlevels:
                raise ValueError('native character field time levels cannot change')
            encoded = value.encode('ascii')
            if len(encoded) > 512 or b'\x00' in encoded:
                raise ValueError('native character fields must fit StrKIND without NUL')
            buffer = ct.create_string_buffer(encoded, 513)
            if nlevels == 1:
                self.lib.native_field_character_0(self.pool(pool), _name(name), buffer,
                                                   int(key in self._text_fields))
            else:
                function = self._bind('native_field_character_levels_0',
                                      [ct.c_void_p, ct.c_char_p, ct.c_int, ct.c_int, ct.c_void_p, ct.c_int])
                function(self.pool(pool), _name(name), nlevels, level, buffer, int(key in self._text_fields))
            self._text_fields[key] = nlevels

    # Native binding/orchestration authority: src/framework/mpas_pool_routines.F:4056-4116.
    def get_text_field(self, pool: str, name: str, *, level=1) -> str:
        return self.get_text_field_storage(pool, name, level=level).tobytes().decode('ascii').rstrip(' ')

    # Native binding/orchestration authority: src/framework/mpas_pool_routines.F:4056-4116.
    def get_text_field_storage(self, pool: str, name: str, *, level=1):
        """Copy all StrKIND bytes, preserving padding and native character storage."""
        with _LOCK:
            self._check()
            if (pool, name) not in self._text_fields:
                raise KeyError((pool, name))
            nlevels = self._text_fields[pool, name]
            if not 1 <= level <= nlevels:
                raise ValueError('invalid native character field time level')
            buffer = ct.create_string_buffer(513)
            if nlevels == 1:
                self.lib.native_field_character_0(self.pool(pool), _name(name), buffer, 2)
            else:
                function = self._bind('native_field_character_levels_0',
                                      [ct.c_void_p, ct.c_char_p, ct.c_int, ct.c_int, ct.c_void_p, ct.c_int])
                function(self.pool(pool), _name(name), nlevels, level, buffer, 2)
            return np.ctypeslib.as_array((ct.c_ubyte * 512).from_buffer(buffer)).copy()

    # Native binding/orchestration authority: src/framework/mpas_pool_routines.F:3436-4240.
    def get_field(self, pool: str, name: str, *, level: int = 1, out: Any = None):
        with _LOCK:
            self._check()
            dtype, shape, nlevels = self._fields[(pool, name)]
            if not 1 <= level <= nlevels:
                raise ValueError('invalid native field time level')
            result = np.empty(shape, dtype=dtype, order='F')
            dims = np.asarray(shape or (1,), dtype=np.int32)
            self._field_fn(dtype, len(shape))(self.pool(pool), _name(name), dims.ctypes.data,
                                             nlevels, level, result.ctypes.data, 2)
            if out is not None:
                if out.shape != shape or out.dtype != dtype:
                    raise ValueError('output shape/dtype differs from native field')
                if hasattr(out, 'set'):
                    out.set(result)
                else:
                    out[...] = result
                return out
            return result

    # Native binding/orchestration authority: src/framework/mpas_pool_routines.F:2108-3211.
    def put_fields(self, pools: Mapping[str, Mapping[str, Any]]):
        for pool, fields in pools.items():
            for name, value in fields.items():
                self.put_field(pool, name, value)

    # Native binding/orchestration authority: src/core_atmosphere/physics/mpas_atmphys_driver.F:209-384; src/core_atmosphere/physics/mpas_atmphys_todynamics.F:98-498.
    def call(self, name: str, **arguments):
        """Call the exact native routine; no implicit preparation or extra scheme call."""
        with _LOCK:
            self._check()
            entry = self.abi['entries'][name]
            expected = set(entry['arguments'])
            if set(arguments) != expected:
                raise TypeError(f'{name} requires {entry["arguments"]}; got {list(arguments)}')
            if name == 'physics_get_tend':
                if self._core_dimensions is None:
                    raise ValueError('physics_get_tend requires mpas_atm_set_dims; native explicit extents otherwise are unset')
                mesh = arguments['mesh']
                if not isinstance(mesh, str):
                    mesh = next((p for p, h in self._pools.items() if h == mesh), None)
                nz = self._core_dimensions[0]
                for arg, dimension in [('tend_ru_physics', 'nEdges'),
                                       ('tend_rtheta_physics', 'nCells'), ('tend_rho_physics', 'nCells')]:
                    extent = self._dimension_values.get((mesh, dimension))
                    if extent is None or arguments[arg].shape != (nz, extent + 1):
                        raise ValueError(f'{arg} must match native nVertLevels and {dimension}+1; incorrect extents corrupt native storage')
            values = []
            buffers = []
            for arg, abi in zip(entry['arguments'], entry['abi']):
                value = arguments[arg]
                if abi == 'handle' and isinstance(value, str):
                    if value not in self._pools:
                        raise ValueError(f'pool {value!r} has not been populated')
                    value = self._pools[value]
                if abi == 'handle' and not value:
                    raise ValueError(f'{name}.{arg} requires a native object handle')
                if abi == 'string':
                    value = _name(value)
                if abi == 'real_array2':
                    host = _host(value)
                    if host.dtype != np.dtype('float32') or host.ndim != 2:
                        raise ValueError(f'{name}.{arg} requires a float32 rank-two array')
                    if isinstance(value, np.ndarray) and not value.flags.writeable:
                        raise ValueError(f'{name}.{arg} requires writable output storage')
                    buffer = np.array(host, dtype=np.float32, order='F', copy=True)
                    buffers.append((value, buffer))
                    value = _RealArray2(buffer.ctypes.data, (ct.c_int32 * 2)(*buffer.shape))
                values.append(value)
            getattr(self.lib, 'native_' + name)(*values)
            for original, buffer in buffers:
                if hasattr(original, 'set'):
                    original.set(buffer)
                else:
                    original[...] = buffer
            if name == 'mpas_atm_set_dims':
                self._core_dimensions = tuple(arguments[n] for n in entry['arguments'])
            if name == 'physics_init':
                self._initialized = True
                self._init_configs = arguments['configs']
            elif name == 'atmphys_finalize':
                self._initialized = False

    # Native binding/orchestration authority: src/framework/mpas_derived_types.F:75-81.
    def framework_object(self, name: str) -> int:
        """Native dm_info, clock or stream manager handle for physics_init."""
        with _LOCK:
            self._check()
            which = {'dminfo': 1, 'clock': 2, 'stream_manager': 3, 'domain': 4, 'block': 5}[name]
            handle = self.lib.native_runtime_object(self._runtime, which)
            if not handle:
                raise ValueError(f'native {name} has not been initialized')
            return handle

    # Native binding/orchestration authority: src/framework/mpas_block_types.inc:19-49; src/framework/mpas_pool_routines.F:5105-5137.
    def block(self, *, configs='configs', mesh='mesh', diag_physics='diag_physics',
              sfc_input='sfc_input', atm_input='atm_input'):
        """Attach one real native block to existing owned pools for the time manager."""
        with _LOCK:
            self._check()
            status = ct.c_int()
            handles = [self.pool(name) for name in [configs, mesh, diag_physics, sfc_input, atm_input]]
            self.lib.native_runtime_block(self._runtime, *handles, ct.byref(status))
            if status.value:
                raise RuntimeError(f'native block attachment failed: status={status.value}')

    # Native binding/orchestration authority: src/framework/mpas_timekeeping.F:87-130,176-221,1143-1298,1347-1590.
    def attach_pool(self, name: str, *, pool: str | None = None):
        """Attach another populated pool to the native block (e.g. emissions).

        MPAS source: src/framework/mpas_pool_routines.F:5105-5137.
        The fork's corresponding source range is recorded in the lane report.
        The block borrows storage until runtime teardown, before pool destruction.
        """
        with _LOCK:
            self._check()
            status = ct.c_int()
            function = self._bind('native_runtime_attach_pool',
                                  [ct.c_void_p, ct.c_char_p, ct.c_void_p, ct.POINTER(ct.c_int)])
            function(self._runtime, _name(name), self.pool(pool or name), ct.byref(status))
            if status.value:
                raise RuntimeError(f'native pool attachment failed: status={status.value}')

    def clock(self, *, start: str, stop: str, dt: float, calendar: str = 'gregorian'):
        with _LOCK:
            self._check()
            if dt <= 0:
                raise ValueError('native clock requires a positive time step')
            status = ct.c_int()
            self.lib.native_runtime_clock(self._runtime, _name(calendar), _name(start),
                                          _name(stop), dt, ct.byref(status))
            if status.value:
                raise RuntimeError(f'native clock initialization failed: status={status.value}')

    # Native binding/orchestration authority: src/framework/mpas_dmpar.F:1600-2160,6456-6503.
    def decomposition(self, element: str, *, global_ids: Any, owned: int, halo_ends=None):
        """Build native halo lists for one entity using owned-then-halo global IDs.

        Collective across the native MPI communicator. Every rank must participate,
        including ranks with no halos. ID zero is not a physical entity; exclude
        trailing MPAS dummy storage from this ID list.
        """
        with _LOCK:
            self._check()
            which = {'nCells': 1, 'nEdges': 2, 'nVertices': 3}[element]
            ids = _host(global_ids)
            if ids.dtype != np.dtype('int32') or ids.ndim != 1:
                raise ValueError('native decomposition requires an int32 ID vector')
            if not 0 <= owned <= ids.size or np.any(ids <= 0):
                raise ValueError('native decomposition requires positive IDs and valid owned count')
            ids = np.ascontiguousarray(ids)
            status = ct.c_int()
            if halo_ends is None:
                self.lib.native_runtime_decomposition(self._runtime, which, owned, ids.size,
                                                       ids.ctypes.data, ct.byref(status))
            else:
                ends = np.asarray(halo_ends, dtype=np.int32)
                if (ends.ndim != 1 or not ends.size or ends[-1] != ids.size or
                        np.any(ends < owned) or np.any(ends > ids.size) or np.any(np.diff(ends) < 0)):
                    raise ValueError('halo ends must be increasing cumulative offsets ending at the ID count')
                ends = np.ascontiguousarray(ends)
                function = self._bind('native_runtime_decomposition_layers',
                                      [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_void_p,
                                       ct.c_int, ct.c_void_p, ct.POINTER(ct.c_int)])
                function(self._runtime, which, owned, ids.size, ids.ctypes.data,
                         ends.size, ends.ctypes.data, ct.byref(status))
            if status.value:
                raise RuntimeError(f'native decomposition initialization failed: status={status.value}')
            self._decompositions.add(element)

    # Native binding/orchestration authority: src/framework/mpas_pool_routines.F:2110-3143; src/framework/mpas_io_streams.F:976.
    def register_field(self, pool: str, name: str, *, dimensions=(), decomposed=False,
                       registry_name=None, components=(), has_time=None):
        """Attach owned field storage and native metadata to the block's allFields.

        The outer dimension of a decomposed field must have a native decomposition.
        This preserves every time level and gives native I/O its Registry names.
        """
        with _LOCK:
            self._check()
            key = (pool, name)
            registry_name = name if registry_name is None else registry_name
            components = tuple(components)
            if registry_name in self._registered or key in self._registered.values():
                raise ValueError(f'native block field {registry_name!r} or pool storage is already registered')
            if key in self._text_fields:
                kind, rank = 'character', 0
                has_time = self._text_fields[key] > 1 if has_time is None else has_time
            else:
                dtype, shape, levels = self._fields[key]
                kind, rank = _KINDS[dtype], len(shape)
                has_time = levels > 1 if has_time is None else has_time
            if len(dimensions) != rank:
                raise ValueError('Registry dimension names must match field rank')
            if components and (not rank or len(components) != shape[0]):
                raise ValueError('Registry constituents must match the first array extent')
            if len(set(components)) != len(components):
                raise ValueError('Registry constituent names must be distinct')
            if decomposed and (not rank or dimensions[-1] not in self._decompositions):
                raise ValueError('decomposed field requires native halo lists for its outer dimension')
            names = b''.join(_name(value).ljust(512, b'\0') for value in dimensions)
            buffer = ct.create_string_buffer(names)
            constituents = ct.create_string_buffer(b''.join(_name(v).ljust(512, b'\0') for v in components))
            status = ct.c_int()
            function = self._bind(f'native_runtime_field_{kind}_{rank}',
                                  [ct.c_void_p, ct.c_void_p, ct.c_char_p, ct.c_char_p, ct.c_void_p,
                                   ct.c_bool, ct.c_int, ct.c_void_p, ct.c_bool, ct.POINTER(ct.c_int)])
            function(self._runtime, self.pool(pool), _name(name), _name(registry_name), buffer,
                     decomposed, len(components), constituents, has_time, ct.byref(status))
            if status.value:
                raise RuntimeError(f'native field registration failed: status={status.value}')
            self._registered[registry_name] = key

    # Native binding/orchestration authority: src/framework/mpas_dmpar.F:6853-6903,7009-7147.
    def exchange_group(self, group: str, fields, *, level: int = 1, halo_layers=None):
        """Register the named MPAS halo groups used by native RK tendencies."""
        with _LOCK:
            self._check()
            if group in self._groups or level < 1:
                raise ValueError('native exchange group already exists or time level is invalid')
            fields = tuple(fields)
            layers = None
            if halo_layers is not None:
                layers = np.asarray(halo_layers, dtype=np.int32)
                if layers.ndim != 1 or not layers.size or np.any(layers < 1):
                    raise ValueError('native exchange group requires positive halo layers')
                layers = np.ascontiguousarray(layers)
            for name in fields:
                pool, key = self._registered[name]
                if level > self._fields[pool, key][2]:
                    raise ValueError('native exchange group time level exceeds field storage')
            status = ct.c_int()
            self.lib.native_runtime_group(self._runtime, _name(group), ct.byref(status))
            if status.value:
                raise RuntimeError(f'native exchange group creation failed: status={status.value}')
            for name in fields:
                if layers is None:
                    self.lib.native_runtime_group_field(self._runtime, _name(group), _name(name), level, ct.byref(status))
                else:
                    function = self._bind('native_runtime_group_field_layers',
                                          [ct.c_void_p, ct.c_char_p, ct.c_char_p, ct.c_int,
                                           ct.c_int, ct.c_void_p, ct.POINTER(ct.c_int)])
                    function(self._runtime, _name(group), _name(name), level,
                             layers.size, layers.ctypes.data, ct.byref(status))
                if status.value:
                    raise RuntimeError(f'native exchange group field registration failed: status={status.value}')
            self._groups.add(group)

    # Native binding/orchestration authority: src/framework/mpas_dmpar.F:4788-6257.
    def exchange_halo_group(self, group: str):
        """Execute the pristine native callback on a previously registered group."""
        with _LOCK:
            self._check()
            if group not in self._groups:
                raise ValueError('native exchange group must be registered first')
            function = self._bind('native_runtime_group_exchange',
                                  [ct.c_void_p, ct.c_char_p, ct.POINTER(ct.c_int)])
            status = ct.c_int()
            function(self._runtime, _name(group), ct.byref(status))
            if status.value:
                raise RuntimeError(f'native group exchange failed: status={status.value}')

    def exchange_halo(self, name: str, *, level: int = 1):
        """Collectively exchange registered field halos using pristine MPAS MPI."""
        with _LOCK:
            self._check()
            pool, key = self._registered[name]
            dtype, shape, nlevels = self._fields[pool, key]
            if not shape or not 1 <= level <= nlevels:
                raise ValueError('native halo exchange requires an array and valid time level')
            function = self._bind(f'native_runtime_halo_{_KINDS[dtype]}_{len(shape)}',
                                  [ct.c_void_p, ct.c_char_p, ct.c_int, ct.POINTER(ct.c_int)])
            status = ct.c_int()
            function(self._runtime, _name(name), level, ct.byref(status))
            if status.value:
                raise RuntimeError(f'native halo exchange failed: status={status.value}')

    # Native binding/orchestration authority: src/framework/mpas_stream_manager.F:834-930.
    def stream_field(self, stream: str, name: str):
        """Add a registered Registry field to an existing native stream."""
        with _LOCK:
            self._check()
            if name not in self._registered:
                raise ValueError(f'native stream field {name!r} has not been registered')
            status = ct.c_int()
            self.lib.native_runtime_stream_field(self._runtime, _name(stream), _name(name), ct.byref(status))
            if status.value:
                raise RuntimeError(f'native stream field attachment failed: status={status.value}')

    # Native binding/orchestration authority: src/framework/mpas_stream_manager.F:3467-3586,2764-2860.
    def transfer_stream(self, stream: str, *, read: bool, level: int = 1):
        """Invoke pristine MPAS stream input/output, including native input halos."""
        with _LOCK:
            self._check()
            if level < 1:
                raise ValueError('native stream time levels are one based')
            status = ct.c_int()
            self.lib.native_runtime_stream_transfer(self._runtime, _name(stream), level, read, ct.byref(status))
            if status.value:
                raise RuntimeError(f'native stream transfer failed: status={status.value}')

    # Native binding/orchestration authority: src/framework/mpas_io.F:147-238; src/framework/mpas_stream_manager.F:120-195.
    def streams(self):
        """Initialize the real native I/O context and an empty stream manager.

        Stream field registration and decomposition must be supplied before invoking
        configurations whose initialization reads atmospheric or UGWP streams.
        """
        with _LOCK:
            self._check()
            if self.build['io'] != 'SMIOL':
                raise RuntimeError('native stream input requires the SMIOL build')
            status = ct.c_int()
            self.lib.native_runtime_streams(self._runtime, ct.byref(status))
            if status.value:
                raise RuntimeError(f'native stream initialization failed: status={status.value}')

    # Native binding/orchestration authority: src/core_atmosphere/physics/mpas_atmphys_driver.F:209-384.
    def __getattr__(self, name: str):
        abi = self.__dict__.get('abi', {})
        if name in abi.get('entries', {}):
            return partial(self.call, name)
        raise AttributeError(name)

    # Native binding/orchestration authority: src/framework/mpas_stream_manager.F:316-435,1833-1905.
    def stream(self, name: str, *, filename: str, direction: str, interval: str = 'none',
               input_interval=None, output_interval=None, filename_interval: str = 'none'):
        """Create a native stream; field registration is a separate operation."""
        with _LOCK:
            self._check()
            which = {'input': 1, 'output': 2, 'input;output': 3, 'output;input': 3}[direction]
            incoming = interval if input_interval is None else input_interval
            outgoing = interval if output_interval is None else output_interval
            if filename_interval != 'none' and not hasattr(self.lib, 'native_runtime_stream_filename_interval'):
                raise RuntimeError('native library lacks filename_interval support required for stream file rotation')
            status = ct.c_int()
            self.lib.native_runtime_stream(self._runtime, _name(name), _name(filename),
                                            which, _name(incoming), _name(outgoing), ct.byref(status))
            if status.value:
                raise RuntimeError(f'native stream creation failed: {name}, status={status.value}')
            if filename_interval != 'none':
                self.lib.native_runtime_stream_filename_interval(
                    self._runtime, _name(name), _name(filename_interval), ct.byref(status))
                if status.value:
                    raise RuntimeError(f'native stream filename interval failed: {name}, status={status.value}')

    # Native binding/orchestration authority: src/framework/mpas_timekeeping.F:397-426.
    def advance_clock(self):
        with _LOCK:
            self._check()
            status = ct.c_int()
            self.lib.native_runtime_advance(self._runtime, ct.byref(status))
            if status.value:
                raise RuntimeError(f'native clock advance failed: status={status.value}')

    # Native binding/orchestration authority: src/framework/mpas_timekeeping.F:814-842.
    def alarm_ringing(self, name: str) -> bool:
        with _LOCK:
            self._check()
            status = ct.c_int()
            ringing = self.lib.native_runtime_alarm_ringing(self._runtime, _name(name), ct.byref(status))
            if status.value:
                raise RuntimeError(f'native alarm query failed: status={status.value}')
            return ringing

    # Native binding/orchestration authority: src/framework/mpas_timekeeping.F:925-985.
    def reset_alarm(self, name: str):
        with _LOCK:
            self._check()
            status = ct.c_int()
            self.lib.native_runtime_alarm_reset(self._runtime, _name(name), ct.byref(status))
            if status.value:
                raise RuntimeError(f'native alarm reset failed: status={status.value}')

    # Native binding/orchestration authority: src/core_atmosphere/physics/mpas_atmphys_constants.F:8-103; src/framework/mpas_constants.F:43-56.
    def get_constant(self, name: str):
        """Return the original compiled scalar, preserving its native precision."""
        with _LOCK:
            self._check()
            entry = self.abi['constants'][name.lower()]
            function = self._bind('native_constant_copy',
                                  [ct.c_int, ct.c_void_p, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)])
            width, status = ct.c_int(), ct.c_int()
            function(entry['index'], None, ct.byref(width), ct.byref(status))
            if status.value:
                raise RuntimeError(f'native constant lookup failed: status={status.value}')
            value = np.empty((), dtype=np.dtype(f'f{width.value}'))
            function(entry['index'], value.ctypes.data, ct.byref(width), ct.byref(status))
            return value[()]

    # Native binding/orchestration authority: src/core_atmosphere/mpas_atm_core.F:264-277.
    def initialize_state_time(self, *, pool='state'):
        with _LOCK:
            self._check()
            for name in ('xtime', 'initial_time'):
                if self._text_fields.get((pool, name)) != 2:
                    raise ValueError(f'native {pool}.{name} requires two character time levels')
            if self._fields.get((pool, 'Time')) != (np.dtype('float32'), (), 2):
                raise ValueError('native CF Time requires scalar RKIND storage and two time levels')
            function = self._bind('native_runtime_initialize_state_time',
                                  [ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_int)])
            status = ct.c_int()
            function(self._runtime, self.pool(pool), ct.byref(status))
            if status.value:
                raise RuntimeError(f'native time initialization failed: status={status.value}')

    # Native binding/orchestration authority: src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1835-1852.
    def update_state_time(self, dt: float, *, pool='state', clock_advanced=False,
                          source_level=1, target_level=2):
        """Update the output level using native RKIND addition and native calendar."""
        with _LOCK:
            self._check()
            if self._text_fields.get((pool, 'xtime'), 0) < target_level:
                raise ValueError('native xtime output time level is missing')
            if self._fields.get((pool, 'Time')) != (np.dtype('float32'), (), 2):
                raise ValueError('native CF Time requires scalar RKIND storage and two time levels')
            if source_level not in (1, 2) or target_level not in (1, 2):
                raise ValueError('native time level must be 1 or 2')
            function = self._bind('native_runtime_update_state_time',
                                  [ct.c_void_p, ct.c_void_p, ct.c_float, ct.c_bool,
                                   ct.c_int, ct.c_int, ct.POINTER(ct.c_int)])
            status = ct.c_int()
            function(self._runtime, self.pool(pool), dt, clock_advanced,
                     source_level, target_level, ct.byref(status))
            if status.value:
                raise RuntimeError(f'native state time update failed: status={status.value}')

    # Native binding/orchestration authority: src/core_atmosphere/physics/mpas_atmphys_vars.F:68-955.
    def _variable_fn(self, metadata):
        return self._bind(f'native_vars_{metadata["kind"]}_{metadata["rank"]}',
                          [ct.c_char_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int], ct.c_int)

    # Native binding/orchestration authority: src/core_atmosphere/physics/mpas_atmphys_vars.F:68-955.
    def put_variable(self, name: str, value: Any, *, lower: tuple[int, ...] | None = None):
        with _LOCK:
            self._check()
            metadata = self.abi['variables'][name]
            dtype = np.float32 if metadata['kind'] == 'real' else np.int32
            a = _host(value)
            if a.ndim != metadata['rank']:
                raise ValueError('native module variable rank differs from its declaration')
            if metadata['kind'] != 'logical' and a.dtype != np.dtype(dtype):
                raise TypeError('native module variable dtype differs from its declaration')
            lower = tuple(lower) if lower is not None else (1,) * a.ndim
            if len(lower) != a.ndim or any(n < 0 for n in a.shape):
                raise ValueError('invalid lower bounds or shape')
            dims = np.asarray(a.shape or (1,), dtype=np.int32)
            bounds = np.asarray(lower or (1,), dtype=np.int32)
            # The generated ABI copies into native-owned module storage. Reuse
            # aligned host input and an existing allocation with identical
            # extents/bounds. Query native state each time: drivers may allocate
            # or release these globals between calls; a Python cache is stale.
            buffer = np.require(a, dtype=dtype, requirements=['F', 'A'])
            fn = self._variable_fn(metadata)
            operation = 0
            if a.ndim:
                current_dims = np.ones(a.ndim, dtype=np.int32)
                current_bounds = np.ones(a.ndim, dtype=np.int32)
                query = fn(_name(name), current_dims.ctypes.data,
                           current_bounds.ctypes.data, None, 3)
                if query == 0 and np.array_equal(current_dims, dims) and np.array_equal(current_bounds, bounds):
                    operation = 1
                elif query not in (0, 1):
                    raise RuntimeError(f'native variable query failed: {name}, status={query}')
            status = fn(_name(name), dims.ctypes.data, bounds.ctypes.data,
                        buffer.ctypes.data, operation)
            if status:
                raise RuntimeError(f'native variable write failed: {name}, status={status}')

    # Native binding/orchestration authority: src/core_atmosphere/physics/mpas_atmphys_vars.F:68-955.
    def get_variable(self, name: str):
        with _LOCK:
            self._check()
            metadata = self.abi['variables'][name]
            fn = self._variable_fn(metadata)
            dims = np.ones(max(1, metadata['rank']), dtype=np.int32)
            bounds = dims.copy()
            status = fn(_name(name), dims.ctypes.data, bounds.ctypes.data, None, 3)
            if status == 1:
                return None
            if status:
                raise RuntimeError(f'native variable inspection failed: {name}, status={status}')
            shape = tuple(int(n) for n in dims[:metadata['rank']])
            dtype = np.float32 if metadata['kind'] == 'real' else np.int32
            result = np.empty(shape, dtype=dtype, order='F')
            status = fn(_name(name), dims.ctypes.data, bounds.ctypes.data, result.ctypes.data, 2)
            if status:
                raise RuntimeError(f'native variable read failed: {name}, status={status}')
            return result, tuple(int(n) for n in bounds[:metadata['rank']])

    # Native binding/orchestration authority: src/core_atmosphere/physics/mpas_atmphys_rrtmg_lwinit.F:40-58; src/core_atmosphere/physics/mpas_atmphys_rrtmg_swinit.F:45-62.
    def get_table(self, name: str):
        """Read a generated public native table in its original kind and bounds.

        The ABI manifest states coverage. A snapshot of these tables is not a
        complete native checkpoint: private/local SAVE and derived state differ.
        """
        with _LOCK:
            self._check()
            e = self.abi['tables'][name]
            dims = np.ones(5, dtype=np.int32)
            lower = dims.copy()
            status = ct.c_int()
            width = ct.c_int()
            self.lib.native_table_shape(e['index'], dims.ctypes.data, lower.ctypes.data,
                                        ct.byref(width), ct.byref(status))
            if status.value == 1:
                return None
            if status.value:
                raise RuntimeError(f'native table metadata failed: {name}, status={status.value}')
            dtype = {('real', 4): np.float32, ('real', 8): np.float64,
                     ('integer', 4): np.int32, ('integer', 8): np.int64}[e['kind'], width.value]
            value = np.empty(tuple(int(n) for n in dims[:e['rank']]), dtype=dtype, order='F')
            self._bind(f'native_table_copy_{e["index"]}', [ct.c_void_p])(value.ctypes.data)
            return value, tuple(int(n) for n in lower[:e['rank']])

    # MPAS source: src/framework/mpas_log.F:1074-1116.
    # The fork's corresponding source range is recorded in the lane report.
    def log_write(self, message: str):
        """Write a literal output message through the existing native C logger."""
        with _LOCK:
            self._check()
            self._bind('mpas_log_write_c', [ct.c_char_p, ct.c_char_p])(
                _name(message), b'MPAS_LOG_OUT')

    # MPAS-A v8.4.2: src/core_atmosphere/physics/mpas_atmphys_vars.F:907.
    # MPAS-A v8.4.2: src/core_atmosphere/physics/physics_noahmp/drivers/mpas/NoahmpIOVarType.F90:18.
    def get_noahmp_io(self, name: str):
        """Copy actual initialized derived IO, preserving native type and bounds."""
        with _LOCK:
            self._check()
            field = self.abi['noahmp_io'][name]
            function = self._bind('native_noahmp_io_copy',
                [ct.c_int, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int),
                 ct.POINTER(ct.c_int), ct.c_void_p, ct.POINTER(ct.c_int)])
            dims, lower = (ct.c_int * 5)(), (ct.c_int * 5)()
            width, status = ct.c_int(), ct.c_int()
            function(field['index'], dims, lower, ct.byref(width), None, ct.byref(status))
            if status.value == 1 and field['rank']:
                return None
            if status.value:
                raise RuntimeError(f'native Noah-MP IO metadata failed: {name}, status={status.value}')
            if field['kind'] == 'character':
                expected = field['character_length']
                dtype = np.dtype(f'S{expected}')
            else:
                expected = 4
                dtype = np.dtype('float32' if field['kind'] == 'real' else 'int32')
            if width.value != expected:
                raise ValueError(f'native Noah-MP IO storage width differs: {name}')
            shape = tuple(dims[:field['rank']])
            bounds = tuple(lower[:field['rank']])
            value = np.empty(shape, dtype=dtype, order='F')
            function(field['index'], dims, lower, ct.byref(width), value.ctypes.data, ct.byref(status))
            if status.value or tuple(dims[:field['rank']]) != shape or tuple(lower[:field['rank']]) != bounds or width.value != expected:
                raise RuntimeError(f'native Noah-MP IO changed during copy: {name}')
            return value, bounds

    # Native binding/orchestration authority: src/core_atmosphere/physics/mpas_atmphys_finalize.F:37-69; src/framework/mpas_pool_routines.F:203-361.
    def close(self):
        global _ACTIVE
        with _LOCK:
            if self.closed:
                return
            if self._initialized:
                self.atmphys_finalize(configs=self._init_configs)
            self.lib.native_variables_release()
            # Streams and block aliases borrow the pool fields and must release
            # references while the owning pools still exist.
            self.lib.native_runtime_destroy(self._runtime)
            for handle in reversed(list(self._pools.values())):
                self.lib.native_pool_destroy(handle)
            self._pools.clear()
            self.closed = True
            _ACTIVE = None

    # Native binding/orchestration authority: src/framework/mpas_pool_routines.F:169-190.
    def __enter__(self):
        return self

    # Native binding/orchestration authority: src/core_atmosphere/physics/mpas_atmphys_finalize.F:37-69; src/framework/mpas_pool_routines.F:203-361.
    def __exit__(self, *exc):
        self.close()
