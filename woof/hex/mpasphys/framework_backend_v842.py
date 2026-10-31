"""Single-process MPAS pools and clock, with no native framework library.

Storage follows mpas_pool_routines.F and backend/runtime.f90. Numeric fields
retain Registry dimension order, dummy entities and independent time levels.
"""
from __future__ import annotations

from functools import partial
import json
import os
import threading
from pathlib import Path

import numpy as np

from .framework_calendar_v842 import PhysicsClock, interval_from_rkind
from ..host_parallel import copied, copy_into


DATA = Path(__file__).resolve().parents[1] / 'data' / 'mpasphys-framework'


def registry_source():
    """Installed Registry metadata, rather than a downloaded engine tree."""
    return DATA


def _name(value):
    encoded = value.encode('ascii')
    if not encoded or len(encoded) >= 512 or b'\0' in encoded:
        raise ValueError('MPAS names require 1..511 ASCII bytes without NUL')
    return value


def _zero_storage(shape, dtype):
    """Fresh zero pages for repeated moving-class allocations on Linux.

    calloc can clear a recycled multi-MB heap block at each move. A private
    anonymous mapping always starts with kernel-owned zero pages and owns
    the same writable F-order words when a physics field first uses them.
    """
    count = int(np.prod(shape, dtype=np.int64))
    if (os.name == 'posix' and os.environ.get('WOOF_HEX_MOVING_RESIDENT') == '1'
            and os.environ.get('WOOF_HEX_MOVING_ZERO_MMAP', '1') != '0'
            and count*np.dtype(dtype).itemsize >= 64*1024):
        import mmap
        pages = mmap.mmap(-1, count*np.dtype(dtype).itemsize,
                          flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                          prot=mmap.PROT_READ | mmap.PROT_WRITE)
        return np.ndarray(shape, dtype=dtype, order='F', buffer=pages)
    return np.zeros(shape, dtype=dtype, order='F')


def _host(value):
    return np.asarray(value.get() if hasattr(value, 'get') else value)


class _DeferredLevels(list):
    """Private cold words on first access, replaceable without reading them."""
    def __init__(self, levels, loader):
        super().__init__([None]*levels)
        self.loader = loader
        self.lock = threading.RLock()

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        with self.lock:
            value = super().__getitem__(index)
            if value is None:
                value = self.loader()
                self[index] = value
            return value

    def __setitem__(self, index, value):
        with self.lock:
            super().__setitem__(index, value)
            if all(v is not None for v in super().__iter__()):
                self.loader = None

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def peek(self, index):
        return super().__getitem__(index)


class FrameworkPhysics:
    """Owned Python storage for the GPU route's MPAS framework boundary.

    Unsupported physics operations fail explicitly. They never fall back to a
    Fortran library. Device scheme implementations own their own table state.
    """
    framework_only = True
    lib = None
    pin_set = 'v8.4.2'
    # Installed declarations of the pinned source (a pin set's subclass names its own).
    data = DATA

    def __init__(self, *, layout=None, resources=None, stream_factory=None):
        self.layout, self.resources = layout, resources
        self.abi = json.loads((self.data / 'metadata.json').read_text())
        self._pools, self._fields, self._configs, self._text_fields = {}, {}, {}, {}
        self._dimensions, self._dimension_values = {}, {}
        self._registered, self._registration, self._decompositions = {}, {}, {}
        self._groups, self._storage, self._text, self._config_values = {}, {}, {}, {}
        self._variables, self._tables, self._attachments = {}, {}, {}
        self._core_dimensions = None
        self._initialized = False
        self.closed = False
        self.clock_state = None
        self.stream_manager = stream_factory(self) if stream_factory else None
        for name, row in self.abi['variables'].items():
            if row['rank'] == 0:
                dtype = np.float32 if row['kind'] == 'real' else np.int32
                self._variables[name] = (np.array(row.get('initial', 0), dtype=dtype), ())

    def _check(self):
        if self.closed:
            raise RuntimeError('MPAS framework is closed')

    def streams(self):
        """Keep session initialization compatible with the native referee.

        Runtime attaches the Python output writer after session.initialize().
        The owned framework needs no native stream manager or output alarms.
        """
        self._check()

    def pool(self, name):
        self._check()
        self._pools.setdefault(_name(name), name)
        return name

    def dimension(self, pool, name, value):
        self.pool(pool)
        _name(name)
        a = np.asarray(value, dtype=np.int32)
        if a.ndim > 1 or (a.ndim and not a.size):
            raise ValueError('MPAS dimensions require a scalar or nonempty vector')
        key, count = (pool, name), a.size if a.ndim else 0
        if key in self._dimensions and self._dimensions[key] != count:
            raise ValueError('MPAS dimension rank or size cannot change')
        self._dimensions[key] = count
        self._dimension_values[key] = int(a) if not count else a.copy()

    def config(self, name, value, *, pool='configs'):
        self.pool(pool)
        _name(name)
        if isinstance(value, (bool, np.bool_)):
            kind, stored = 'logical', bool(value)
        elif isinstance(value, (int, np.integer)):
            kind, stored = 'integer', np.int32(value)
        elif isinstance(value, (float, np.floating)):
            kind, stored = 'real', np.float32(value)
        elif isinstance(value, str):
            b = value.encode('ascii')
            if len(b) > 512 or b'\0' in b:
                raise ValueError('MPAS config string exceeds StrKIND or contains NUL')
            kind, stored = 'character', value
        else:
            raise TypeError('unsupported MPAS config type')
        key = (pool, name)
        if key in self._configs and self._configs[key] != kind:
            raise TypeError('MPAS config type cannot change')
        self._configs[key], self._config_values[key] = kind, stored

    def get_real_config(self, name, *, pool='configs'):
        self._check()
        if self._configs[pool, name] != 'real':
            raise TypeError('MPAS config is not real')
        return self._config_values[pool, name]

    def put_field(self, pool, name, value, *, level=1, nlevels=1):
        self.pool(pool)
        _name(name)
        a = _host(value)
        if not ((a.dtype == np.float32 and a.ndim <= 5) or
                (a.dtype == np.int32 and a.ndim <= 3)):
            raise TypeError('MPAS field requires RKIND rank <= 5 or integer rank <= 3')
        if nlevels < 1 or not 1 <= level <= nlevels:
            raise ValueError('invalid MPAS field time level')
        key, metadata = (pool, name), (a.dtype, a.shape, nlevels)
        if key in self._text_fields:
            raise TypeError('MPAS field type cannot change')
        if key in self._fields and self._fields[key] != metadata:
            raise ValueError('MPAS field dtype, shape and time levels cannot change')
        if key not in self._fields:
            self._storage[key] = [_zero_storage(a.shape, a.dtype) for _ in range(nlevels)]
        self._fields[key] = metadata
        if isinstance(self._storage[key], _DeferredLevels) and self._storage[key].peek(level-1) is None:
            self._storage[key][level-1] = copied(a, order='F')
            return
        target = self._storage[key][level-1]
        if a is not target:
            # The device views write the live storage in place (field_storage); handing it back is a no-op.
            # Large fields copy in blocks on host threads (host_parallel.copy_into: the same words).
            copy_into(target, a)

    def adopt_field(self, pool, name, value, *, nlevels=1):
        """``put_field`` of every time level of a new field, keeping ``value`` itself as level 1's storage.

        For words the caller hands over and never touches again: the physics session's hydration passes each
        field's freshly read payload buffer (pool_v842.NativePoolHydration.populate).  Level 1 holds those words
        and every other level a copy of them, the words ``put_field`` stores at each level.  THE COST THIS
        REMOVES (RRFS hrrrv5 mv3km, V6000, 2026-10-28): hydration allocated a zeroed copy of every buffer it had
        just read and copied the words in on one core, ~4 s for 206,726 cells and about 11x that on conus3km.
        Anything ``put_field`` would treat differently (a field already held, a non-Fortran or read-only buffer)
        goes through ``put_field``.
        """
        a = value
        key = (pool, name)
        if (key in self._fields or key in self._text_fields or not isinstance(a, np.ndarray)
                or not a.flags.f_contiguous or not a.flags.writeable or not a.flags.aligned
                or not a.dtype.isnative or nlevels < 1):
            for level in range(1, nlevels + 1):
                self.put_field(pool, name, a, level=level, nlevels=nlevels)
            return
        self.pool(pool)
        _name(name)
        if not ((a.dtype == np.float32 and a.ndim <= 5) or
                (a.dtype == np.int32 and a.ndim <= 3)):
            raise TypeError('MPAS field requires RKIND rank <= 5 or integer rank <= 3')
        self._storage[key] = [a] + [copied(a, order='F') for _ in range(nlevels - 1)]
        self._fields[key] = (a.dtype, a.shape, nlevels)

    def adopt_restored_field(self, pool, name, value, *, level=1, nlevels=1):
        """Consume private projected words only when no live cold view exists.

        The moving class relinquishes this buffer after restoration. A field
        whose storage was already observed keeps its original allocation and
        all outstanding views, through the ordinary put_field copy.
        """
        self._check()
        key = pool,name
        storage = self._storage.get(key)
        if (isinstance(value,np.ndarray) and value.flags.f_contiguous
                and value.flags.writeable and value.flags.aligned and value.dtype.isnative
                and self._fields.get(key) == (value.dtype,value.shape,nlevels)
                and 1 <= level <= nlevels and isinstance(storage,_DeferredLevels)
                and storage.peek(level-1) is None):
            storage[level-1] = value
            return
        self.put_field(pool,name,value,level=level,nlevels=nlevels)

    def put_zero_field(self, pool, name, shape, dtype, *, nlevels=1):
        """Every time level of a field whose words are the Registry initial zero, allocated and not written.

        The same words ``put_field`` stores from a zero buffer, without touching a page: the allocation is
        calloc'd, so a field no slot or stream ever writes costs no host memory.  THE COST THIS REMOVES
        (MPAS-route host RAM, 2026-10-27): 6.83 of the 8.16 GB of Registry storage on a 99,898-cell mesh are
        such fields (abstot, ozmixm, the dycore's diag/tend arrays the GPU dycore keeps on the device), and
        hydration wrote each word, once per rate class on the LTS route.
        """
        self.pool(pool)
        _name(name)
        dtype = np.dtype(dtype)
        shape = tuple(int(v) for v in shape)
        if not ((dtype == np.float32 and len(shape) <= 5) or (dtype == np.int32 and len(shape) <= 3)):
            raise TypeError('MPAS field requires RKIND rank <= 5 or integer rank <= 3')
        if nlevels < 1:
            raise ValueError('invalid MPAS field time level')
        key, metadata = (pool, name), (dtype, shape, nlevels)
        if key in self._text_fields:
            raise TypeError('MPAS field type cannot change')
        if key in self._fields and self._fields[key] != metadata:
            raise ValueError('MPAS field dtype, shape and time levels cannot change')
        if key in self._fields:
            storage = self._storage[key]
            for index in range(nlevels):
                value = storage.peek(index) if isinstance(storage, _DeferredLevels) else storage[index]
                if value is None:
                    storage[index] = _zero_storage(shape, dtype)
                else:
                    value.fill(0)
        else:
            self._storage[key] = [_zero_storage(shape, dtype) for _ in range(nlevels)]
        self._fields[key] = metadata

    def put_deferred_field(self, pool, name, shape, dtype, loader, *, nlevels=1):
        """A validated resident cold projection, evaluated only if a reader needs it.

        A moved class overwrites its carried pools before physics init. Lazy
        storage avoids projecting those discarded cold words. Reads and live
        storage still materialize independent native time levels as before.
        """
        self.pool(pool)
        _name(name)
        dtype, shape = np.dtype(dtype), tuple(int(v) for v in shape)
        if not ((dtype == np.float32 and len(shape) <= 5) or (dtype == np.int32 and len(shape) <= 3)):
            raise TypeError('MPAS field requires RKIND rank <= 5 or integer rank <= 3')
        if nlevels < 1:
            raise ValueError('invalid MPAS field time level')
        key = pool, name
        if key in self._fields or key in self._text_fields:
            raise ValueError('deferred MPAS field must be new')
        def read():
            value = loader()
            if (not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype
                    or not value.flags.f_contiguous or not value.flags.writeable):
                raise ValueError('deferred MPAS field shape, dtype or storage differs')
            return value
        self._storage[key] = _DeferredLevels(nlevels, read)
        self._fields[key] = dtype, shape, nlevels

    def field_storage(self, pool, name, *, level=1):
        """The live storage of one time level (no copy), for the device pool views.

        THE COST THIS REMOVES (MPAS-route host RAM, 2026-10-27): ``get_field`` returns a copy, and the device
        views kept that copy for the whole run beside the storage it came from (3.2 GB of resident copies plus
        1.0 GB of transaction copies on a 99,898-cell mesh, once per LTS rate class).  The views now read and
        publish this array in place.
        """
        self._check()
        dtype, shape, levels = self._fields[pool, name]
        if not 1 <= level <= levels:
            raise ValueError('invalid MPAS field time level')
        return self._storage[pool, name][level-1]

    def get_field(self, pool, name, *, level=1, out=None):
        self._check()
        dtype, shape, levels = self._fields[pool, name]
        if not 1 <= level <= levels:
            raise ValueError('invalid MPAS field time level')
        result = copied(self._storage[pool, name][level-1], order='F')
        if out is None:
            return result
        if out.shape != shape or out.dtype != dtype:
            raise ValueError('MPAS output shape/dtype differs')
        if hasattr(out, 'set'):
            out.set(result)
        else:
            out[...] = result
        return out

    def put_fields(self, pools):
        for pool, fields in pools.items():
            for name, value in fields.items():
                self.put_field(pool, name, value)

    def put_text_field(self, pool, name, value, *, level=1, nlevels=1):
        self.pool(pool)
        _name(name)
        encoded = value.encode('ascii')
        if len(encoded) > 512 or b'\0' in encoded:
            raise ValueError('MPAS text field exceeds StrKIND or contains NUL')
        key = (pool, name)
        if key in self._fields:
            raise TypeError('MPAS field type cannot change')
        if nlevels < 1 or not 1 <= level <= nlevels:
            raise ValueError('invalid MPAS text time level')
        if key in self._text_fields and self._text_fields[key] != nlevels:
            raise ValueError('MPAS text time levels cannot change')
        if key not in self._text:
            self._text[key] = [b' ' * 512 for _ in range(nlevels)]
        self._text_fields[key] = nlevels
        self._text[key][level-1] = encoded.ljust(512, b' ')

    def get_text_field_storage(self, pool, name, *, level=1):
        self._check()
        if not 1 <= level <= self._text_fields[pool, name]:
            raise ValueError('invalid MPAS text time level')
        return np.frombuffer(self._text[pool, name][level-1], dtype=np.uint8).copy()

    def get_text_field(self, pool, name, *, level=1):
        return self.get_text_field_storage(pool, name, level=level).tobytes().decode('ascii').rstrip(' ')

    def block(self, **pools):
        self._check()
        for name in ('configs', 'mesh', 'diag_physics', 'sfc_input', 'atm_input'):
            self.attach_pool(name, pool=pools.get(name, name))

    def attach_pool(self, name, *, pool=None):
        self._attachments[_name(name)] = self.pool(pool or name)

    def framework_object(self, name):
        self._check()
        return {'clock': self.clock_state, 'block': self._attachments,
                'domain': self, 'dminfo': self, 'stream_manager': self.stream_manager}[name]

    def decomposition(self, element, *, global_ids, owned, halo_ends=None):
        self._check()
        if element not in ('nCells', 'nEdges', 'nVertices'):
            raise ValueError('unknown MPAS decomposition entity')
        ids = _host(global_ids)
        if ids.dtype != np.int32 or ids.ndim != 1 or np.any(ids <= 0) or not 0 <= owned <= ids.size:
            raise ValueError('MPAS decomposition requires positive int32 IDs and valid owned count')
        if np.unique(ids[:owned]).size != owned:
            raise ValueError('owned MPAS IDs must be unique')
        ends = np.asarray([ids.size] if halo_ends is None else halo_ends, dtype=np.int32)
        if (ends.ndim != 1 or not ends.size or ends[-1] != ids.size or
                np.any(ends < owned) or np.any(ends > ids.size) or np.any(np.diff(ends) < 0)):
            raise ValueError('halo ends must increase from owned count to total count')
        local = {int(v): i for i, v in enumerate(ids[:owned])}
        if getattr(self.layout, 'samehost_partition', False):
            # A same-host rank's halo IDs are owned by other ranks; its rank
            # owner (samehost_v842.RankHalo and the dycore exchanger) fills
            # them. No in-process source exists, so exchange_halo refuses.
            sources = None
        else:
            try:
                sources = np.array([local[int(v)] for v in ids[owned:]], dtype=np.intp)
            except KeyError as error:
                raise ValueError('one-process halo ID has no owned source: ' + str(error)) from error
        self._decompositions[element] = (owned, ids.copy(), ends.copy(), sources)

    def register_field(self, pool, name, *, dimensions=(), decomposed=False,
                       registry_name=None, components=(), has_time=None):
        self._check()
        key, registry_name = (pool, name), name if registry_name is None else registry_name
        _name(registry_name)
        if registry_name in self._registered or key in self._registered.values():
            raise ValueError('MPAS Registry name or field storage already registered')
        shape = () if key in self._text_fields else self._fields[key][1]
        if len(dimensions) != len(shape):
            raise ValueError('Registry dimensions must match field rank')
        if components and (not shape or len(components) != shape[0] or len(set(components)) != len(components)):
            raise ValueError('Registry constituents must be distinct and match first extent')
        if decomposed and (not shape or dimensions[-1] not in self._decompositions):
            raise ValueError('decomposed Registry field needs halo lists')
        levels = self._text_fields[key] if key in self._text_fields else self._fields[key][2]
        self._registered[registry_name] = key
        self._registration[registry_name] = dict(dimensions=tuple(dimensions), decomposed=decomposed,
            components=tuple(components), has_time=levels > 1 if has_time is None else bool(has_time))

    def exchange_group(self, group, fields, *, level=1, halo_layers=None):
        self._check()
        if group in self._groups or level < 1:
            raise ValueError('duplicate exchange group or invalid time level')
        fields = tuple(fields)
        layers = None if halo_layers is None else tuple(halo_layers)
        if layers is not None and (not layers or any(v < 1 for v in layers)):
            raise ValueError('halo layers must be positive')
        for name in fields:
            if level > self._fields[self._registered[name]][2]:
                raise ValueError('exchange time level exceeds storage')
        self._groups[group] = fields, level, layers

    def exchange_halo(self, name, *, level=1, _layers=None):
        self._check()
        key = self._registered[name]
        row = self._registration[name]
        if not row['decomposed']:
            return
        if not 1 <= level <= self._fields[key][2]:
            raise ValueError('invalid halo time level')
        owned, ids, ends, sources = self._decompositions[row['dimensions'][-1]]
        if sources is None:
            # Breakage prevented: a framework exchange on a same-host rank would
            # leave halo words stale (their owners live on other ranks).
            raise RuntimeError(f'{name}: the halo of a same-host rank is exchanged by its rank owner '
                               '(samehost_v842), not by the one-process framework')
        a = self._storage[key][level-1]
        if _layers is None:
            a[..., owned:ids.size] = a[..., sources]
        else:
            for layer in _layers:
                if layer > len(ends):
                    raise ValueError('halo layer exceeds decomposition')
                start = owned if layer == 1 else int(ends[layer-2])
                end = int(ends[layer-1])
                a[..., start:end] = a[..., sources[start-owned:end-owned]]

    def exchange_halo_group(self, group):
        fields, level, layers = self._groups[group]
        for name in fields:
            self.exchange_halo(name, level=level, _layers=layers)

    def clock(self, **kwargs):
        self._check()
        self.clock_state = PhysicsClock(**kwargs)

    def advance_clock(self):
        self._check()
        self.clock_state.advance()

    def initialize_state_time(self, *, pool='state'):
        initial = self.get_text_field(pool, 'initial_time', level=1)
        self.put_text_field(pool, 'initial_time', initial, level=2, nlevels=2)
        for level in (1, 2):
            self.put_text_field(pool, 'xtime', self.clock_state.start_time.stamp(), level=level, nlevels=2)
        self.time_units = 'seconds since ' + initial.replace('_', ' ')

    def update_state_time(self, dt, *, pool='state', clock_advanced=False, source_level=1, target_level=2):
        time = self.clock_state.current_time
        if not clock_advanced:
            time = time.add(interval_from_rkind(dt))
        self.put_text_field(pool, 'xtime', time.stamp(), level=target_level, nlevels=2)
        value = np.float32(self.get_field(pool, 'Time', level=source_level) + np.float32(dt))
        self.put_field(pool, 'Time', np.array(value), level=target_level, nlevels=2)

    def get_constant(self, name):
        self._check()
        return np.array(self.abi['constants'][name.lower()]['word'], np.uint32).view(np.float32)[()]

    def put_variable(self, name, value, *, lower=None):
        self._check()
        row, a = self.abi['variables'][name], _host(value)
        dtype = np.float32 if row['kind'] == 'real' else np.int32
        if a.ndim != row['rank'] or (row['kind'] != 'logical' and a.dtype != dtype):
            raise TypeError('MPAS module variable rank/type differs')
        lower = tuple(lower) if lower is not None else (1,) * a.ndim
        if len(lower) != a.ndim:
            raise ValueError('MPAS variable lower bound rank differs')
        current = self._variables.get(name)
        if current is not None and current[0] is a and a.dtype == dtype and tuple(current[1]) == lower:
            # The device views wrote the live array in place (variable_storage); handing it back is a no-op.
            return
        self._variables[name] = np.array(a, dtype=dtype, order='F', copy=True), lower

    def get_variable(self, name):
        self._check()
        self.abi['variables'][name]
        value = self._variables.get(name)
        return None if value is None else (value[0].copy(order='F'), value[1])

    def variable_storage(self, name):
        """The live module array and its lower bounds (no copy), for the device pool views; None if unallocated.

        THE COST THIS REMOVES: as ``field_storage``; the prepared module arrays are allocated untouched
        (``_allocate``), and the views' copies of them were the touched, run-long host words.
        """
        self._check()
        self.abi['variables'][name]
        return self._variables.get(name)

    def get_table(self, name):
        self._check()
        return self._tables[name]

    def calendar_state(self, flags):
        words = self.clock_state.calendar_words()
        for name, flag in (('l_radtlw', 'radiation_lw'), ('l_radtsw', 'radiation_sw'),
                           ('l_conv', 'convection'), ('doabsems', 'cam_abs'),
                           ('l_camlw', 'cam_restart'),
                           ('l_acrain', 'rain_bucket'), ('l_acradt', 'radiation_bucket'),
                           ('l_diags', 'diagnostics')):
            self.put_variable(name, np.array(bool(getattr(flags, flag)), np.int32))
        self._tables['mpas_atmphys_manager.curr_julday'] = np.array(words['julian']), ()
        time = self.clock_state.current_time
        for name, value in (('year', time.year), ('month', time.month),
                            ('day', time.day), ('julday', time.day_of_year)):
            self._tables['mpas_atmphys_manager.' + name] = np.array(value, np.int32), ()
        return words

    def call(self, name, **arguments):
        self._check()
        if name == 'mpas_atm_set_dims':
            self._core_dimensions = tuple(arguments[key] for key in
                ('nvertlevels_val', 'maxedges_val', 'maxedges2_val', 'num_scalars_val'))
        elif name.startswith('allocate_'):
            self._allocate(name)
        elif name.startswith('deallocate_'):
            for variable in self.abi['allocations'].get(name.replace('deallocate_', 'allocate_', 1), {}):
                self._variables.pop(variable, None)
        elif name == 'atmphys_finalize':
            self._initialized = False
        else:
            raise RuntimeError('MPAS framework operation has no Python implementation: ' + name)

    def _allocation_local(self, name, value):
        """A bound that is a local of the allocate routine, not a module integer (None: not one)."""
        return 14 if value == 'nbndsw' else None

    def _allocate(self, name):
        for variable, bounds in self.abi['allocations'][name].items():
            if variable in self._variables:
                continue
            lower, shape = [], []
            for lo, hi in bounds:
                def bound(value):
                    local = self._allocation_local(name, value)
                    if local is not None:
                        return local
                    if value.isdigit():
                        return int(value)
                    # Generated expressions contain only module integer names and +/- integers.
                    import re
                    match = re.fullmatch(r'(\w+)([+-]\d+)?', value)
                    if match is None:
                        raise ValueError('unsupported module allocation bound: ' + value)
                    return int(self._variables[match[1]][0]) + int(match[2] or 0)
                l, h = bound(lo), bound(hi)
                lower.append(l)
                shape.append(max(0, h-l+1))
            row = self.abi['variables'][variable]
            dtype = np.float32 if row['kind'] == 'real' else np.int32
            self._variables[variable] = np.zeros(shape, dtype=dtype, order='F'), tuple(lower)

    def __getattr__(self, name):
        if name in self.__dict__.get('abi', {}).get('entries', {}):
            return partial(self.call, name)
        raise AttributeError(name)

    def close(self):
        if self.closed:
            return
        if self.stream_manager is not None:
            self.stream_manager.close()
        self._storage.clear()
        self._text.clear()
        self._variables.clear()
        self.closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *exc):
        self.close()
