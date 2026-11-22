import os as _hex_os
_hex_os.environ["WOOF_HEX_NUMERICS"] = "strict"  # a bitwise proof against native MPAS runs strict numerics (fast is the default, strict the opt-in)

_COMPILED_MODULES = {}

_ARGUMENT_DTYPE = np.dtype([
    ('data', np.uint64), ('lo', np.int32, (4,)), ('hi', np.int32, (4,)),
    ('ivalue', np.int32), ('fvalue', np.float32), ('present', np.int32),
], align=True)

def shared_math_source(kernel_directory: Path) -> str:
    """Read the pinned engine's common device-math implementations.

    The caller must add the shared binary32 erf entrypoint as
    ``smoke_shared_erf(float)``. FENGSHA reaches it at
    MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_fengsha_mod.F90:355-356.
    No replacement erf implementation is supplied by this scheme.
    """
    names = ('glibc_flt32.cuh', 'flt32_expf_fma.cuh', 'glibc_flt64.cuh',
             'glibc_trig_flt32.cuh', 'thompson_aerosol_libm.cuh')
    return '\n'.join((kernel_directory / name).read_text() for name in names)

class SmokeState:
    """Native tile state; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:34-638 @1d6e370e."""

    def __init__(self, *, math_source: str, workspace_words: int = 1_000_000,
                 persistent_words: int = 50_000, storage_transport_source: str = ''):
        """Initialize SAVE state; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:648-814 @1d6e370e."""
        import cupy as cp
        from cupy.cuda import compiler, function
        prefix = '#define SDKERNEL __global__\n#define SDGUARD if(blockIdx.x || threadIdx.x) return\n'
        # RawKernel injects --ftz=true. Native GNU preserves subnormals; use
        # the low-level compiler with an explicit --ftz=false instead.
        import hashlib
        source = math_source + '\n' + prefix + CUDA_SOURCE + '\n' + storage_transport_source
        digest = hashlib.sha256(source.encode()).hexdigest()
        architecture = cp.cuda.Device().compute_capability
        key = (cp.cuda.runtime.getDevice(), architecture, digest)
        if key not in _COMPILED_MODULES:
            image, _ = compiler.compile_using_nvrtc(
                source, options=('--std=c++17', '--fmad=false', '--ftz=false',
                                 '--prec-div=true', '--prec-sqrt=true'),
                arch=architecture, cache_in_memory=True,
            )
            module = function.Module()
            module.load(image)
            _COMPILED_MODULES[key] = module
        self._module = _COMPILED_MODULES[key]
        self.compile_metadata = dict(source_sha256=digest, ftz=False, fmad=False,
                                     architecture=architecture)
        self._cp = cp
        self._functions = {}
        # The whole native call graph passes many Fortran descriptors through
        # nested calls. CUDA's default per-thread stack is only 1 KiB.
        limit = cp.cuda.runtime.cudaLimitStackSize
        current_stack = cp.cuda.runtime.deviceGetLimit(limit)
        cp.cuda.runtime.deviceSetLimit(limit, max(current_stack, 16_384))
        self.compile_metadata['stack_bytes'] = cp.cuda.runtime.deviceGetLimit(limit)
        sizes = cp.empty(3, dtype=cp.int32)
        self._function('sd_sizes')((1,), (1,), (sizes,))
        state_size, plume_size, arg_size = map(int, sizes.get())
        if arg_size != _ARGUMENT_DTYPE.itemsize:
            raise RuntimeError(f'SMOKE descriptor ABI differs: {arg_size}')
        self._state = cp.empty(state_size, dtype=cp.uint8)
        self._plume = cp.empty(plume_size, dtype=cp.uint8)
        self._persistent = cp.empty(persistent_words, dtype=cp.float32)
        self._workspace = cp.empty(workspace_words, dtype=cp.float32)
        self._function('sd_initialize')((1,), (1,), (
            self._state, self._plume, self._persistent, np.int32(persistent_words),
        ))

    def _function(self, name: str):
        """Resolve graph entry; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:34-638 @1d6e370e."""
        if name not in self._functions:
            self._functions[name] = self._module.get_function(name)
        return self._functions[name]

    def call(self, routine: str, arguments: dict, *, lower_bounds: dict | None = None):
        """Launch one complete routine graph using its original driver arguments.

        ``None`` denotes a missing OPTIONAL argument. Scalars and array
        descriptors are packed on the host; weather arrays remain on device.
        The returned descriptor buffer carries scalar INOUT/OUT values.
        MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:34-638 @1d6e370e.
        """
        spec = ROUTINES[routine]
        cp = self._cp
        lower_bounds = lower_bounds or {}
        # The wrapper's automatic arrays scale with the native tile, including
        # its full chemistry tensor (wrapper:210-240). Reserve from descriptor
        # sizes; the initial workspace is a floor, not a supported tile limit.
        required_words = 50_000 + 2 * sum(
            value.size for value in arguments.values() if isinstance(value, cp.ndarray)
        )
        if self._workspace.size < required_words:
            self._workspace = cp.empty(required_words, dtype=cp.float32)
        descriptor = np.zeros(len(spec['args']), dtype=_ARGUMENT_DTYPE)
        keep = []
        for index, name in enumerate(spec['args']):
            symbol = spec['symbols'][name]
            if name not in arguments and not symbol['optional']:
                raise ValueError(f'{routine} requires {name}')
            value = arguments.get(name)
            if value is None:
                if not symbol['optional']:
                    raise ValueError(f'{routine} requires {name}')
                continue
            descriptor['present'][index] = 1
            if symbol['dim']:
                dtype = cp.float32 if symbol['type'] == 'float' else cp.int32
                if not isinstance(value, cp.ndarray) or value.dtype != dtype:
                    raise TypeError(f'{name} must be a device {dtype} array')
                if value.ndim != len(symbol['dim']) or not value.flags.f_contiguous:
                    raise ValueError(f'{name} requires rank {len(symbol["dim"])} in Fortran order')
                lo = lower_bounds.get(name, (1,) * value.ndim)
                if len(lo) != value.ndim:
                    raise ValueError(f'{name} lower bounds have the wrong rank')
                descriptor['data'][index] = value.data.ptr
                descriptor['lo'][index, :value.ndim] = lo
                descriptor['hi'][index, :value.ndim] = tuple(l + n - 1 for l, n in zip(lo, value.shape))
                keep.append(value)
            elif symbol['type'] == 'FStr':
                if not isinstance(value, cp.ndarray) or value.dtype != cp.uint8:
                    raise TypeError(f'{name} must be a device character buffer')
                descriptor['data'][index] = value.data.ptr
                keep.append(value)
            elif symbol['type'] == 'float':
                descriptor['fvalue'][index] = value
            else:
                descriptor['ivalue'][index] = value
        device_descriptor = cp.asarray(descriptor)
        self._function('sd_run_' + routine)((1,), (1,), (
            self._state, self._workspace, np.int32(self._workspace.size), device_descriptor,
        ))
        # Device work owns references until completion, including arrays
        # passed by an ephemeral mapping.
        cp.cuda.get_current_stream().synchronize()
        return device_descriptor

    def advance(self, arguments: dict, *, lower_bounds: dict | None = None):
        """RRFS driver; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:34-638 @1d6e370e."""
        return self.call('mpas_smoke_driver', arguments, lower_bounds=lower_bounds)


def _schema_nodes(node):
    """Registry traversal; MPAS-Model-rrfs/src/core_atmosphere/physics/registry.chemistry.xml:1-548."""
    yield node
    for child in node['children']:
        yield from _schema_nodes(child)


def chemistry_defaults():
    """Native metadata defaults; MPAS-Model-rrfs/src/core_atmosphere/physics/registry.chemistry.xml:56-271."""
    values = {}
    for node in _schema_nodes(CHEMISTRY_SCHEMAS['registry']['atmosphere']['tree']):
        if node['tag'] != 'nml_option':
            continue
        attr = node['attributes']; raw = attr['default_value']; kind = attr['type']
        values[attr['name']] = (int(raw) if kind == 'integer' else float(raw) if kind == 'real'
                               else raw.lower() == 'true' if kind == 'logical' else raw)
    return values


def chemistry_packages(options):
    """Native package predicates; MPAS-Model-rrfs/src/core_atmosphere/physics/registry.chemistry.xml:31-51."""
    o = {**chemistry_defaults(), **options}
    smoke = o['config_smoke_scheme'].rstrip(' ') != 'off'
    dust = o['config_dust_scheme'].rstrip(' ') != 'off'
    anthro = o['config_anthro_scheme'].rstrip(' ')
    rwc = o['config_rwc_scheme'].rstrip(' ') != 'off'
    return dict(mpas_smoke_fine=smoke or rwc, mpas_smoke_in=smoke, mpas_dust_in=dust,
                mpas_anthro_in=anthro != 'off' or rwc,
                mpas_anthro_aero_in=anthro in ('simple_aero', 'simple_gasaero', 'acomp_gasaero'),
                mpas_rwc_in=rwc)


def chemistry_scalar_registry(options):
    """Native scalar order/units; MPAS-Model-rrfs/src/core_atmosphere/physics/registry.chemistry.xml:309-328."""
    packages = chemistry_packages(options)
    for node in _schema_nodes(CHEMISTRY_SCHEMAS['registry']['atmosphere']['tree']):
        if node['tag'] == 'var_array' and node['attributes']['name'] == 'scalars':
            return tuple(dict(c['attributes']) for c in node['children']
                         if any(packages.get(p, False) for p in c['attributes']['packages'].split(';')))
    raise RuntimeError('pinned chemistry scalar registry is absent')


class SmokeSlot:
    """Complete driver callback with bindings supplied by the pool adapter.

    ``bindings(call)`` yields tiles with ``key``, ``from_mpas``, ``wrapper``,
    ``diagnostics`` and ``to_mpas`` mappings, plus optional ``lower_bounds``.
    The pool adapter binds fields/characters and native dimensions; emission
    decode and interpolation stay in Rust. State persists under each tile
    key, matching native module SAVE/THREADPRIVATE ownership.
    MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_smoke.F:1025-1299 @1d6e370e.
    """
    def __init__(self, bindings, *, math_source, workspace_words=1_000_000):
        """Tile ownership; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_smoke.F:35-192,1025-1299 @1d6e370e."""
        self._bindings = bindings
        self._math_source = math_source
        self._workspace_words = workspace_words
        self._states = {}

    def __call__(self, call):
        """Native driver order; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_smoke.F:1079-1299 @1d6e370e."""
        for tile in self._bindings(call):
            key = tile['key']
            if key not in self._states:
                self._states[key] = SmokeState(math_source=self._math_source,
                                               workspace_words=self._workspace_words)
            state = self._states[key]
            lo = tile.get('lower_bounds', {})
            state.call('smoke_from_mpas_fields', tile['from_mpas'], lower_bounds=lo)
            state.advance(tile['wrapper'], lower_bounds=lo)
            state.call('smoke_driver_diagnostics', tile['diagnostics'], lower_bounds=lo)
            state.call('smoke_to_mpas_fields', tile['to_mpas'], lower_bounds=lo)


def register_slot(registry, bindings, *, math_source, workspace_words=1_000_000):
    """Register the fork slot; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver.F:334-351 @1d6e370e."""
    callback = SmokeSlot(bindings, math_source=math_source, workspace_words=workspace_words)
    registry.register('smoke', 'on', 'hex-cuda', callback)
    return callback
