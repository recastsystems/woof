"""RRFS-SD framework binding without a native physics library.

Science is the pinned SMOKE CUDA graph of the route's own pin, chosen from the suite table
(``submodule_pins.SMOKE``): smoke_5d2c71e2 for the shipping RRFSx 44c0849 row, smoke_9652bbf for the 1d6e370e row
(``--physics-pin-set rrfs``). This module supplies only driver locals, shared device math and the ported
framework's log service. Authority: mpas_atmphys_driver_smoke.F:998-1299 @1d6e370e and :1050-1335 @44c0849.
"""
from importlib import import_module
from importlib.resources import files
import logging


def pinned(config):
    """The smoke graph module of the route's SMOKE pin (table work: a new pin is a new smoke_<pin> module)."""
    import re
    from importlib.resources import files as package_files
    from ..route_table import route_row
    pin = route_row(config)['submodule_pins']['SMOKE']
    names = [path.name[6:-3] for path in package_files(__package__).iterdir()
             if re.fullmatch(r'smoke_[0-9a-f]{7,40}\.py', path.name) and pin.startswith(path.name[6:-3])]
    if len(names) != 1:
        raise ValueError(f'no ported RRFS-SD graph for SMOKE pin {pin}: smoke_<pin>.py modules {names}')
    return import_module(f'{__package__}.smoke_{names[0]}')


def _entry(native, call):
    """Driver locals at driver_smoke.F:382-384,1133-1144.

    nblocks is read only with ebb_dcycle=2. Other selections leave the native
    local undefined; zero here describes an unused extent, not emissions.
    Captured native replays retain their original local words via their own
    runtime and do not use this framework binding.
    """
    cycle = int(call.options['ebb_dcycle'])
    hours = int(call.options['bb_input_prevh'])
    if cycle == 2 and hours <= 0:
        raise ValueError('bb_input_prevh must be positive: smoke divides 24 by it')
    blocks = 24 // hours if cycle == 2 else 0
    return {'from_mpas': {'nblocks': blocks}, 'wrapper': {'nblocks': blocks}}


def math_source(smoke=None):
    """Common device math, also available to CPU-only kernel preparation."""
    smoke = smoke or import_module(f'{__package__}.smoke_9652bbf')
    kernels = files('woof.core.kernels')
    math = smoke.shared_math_source(kernels)
    math += '\n' + files('woof.hex').joinpath('glibc_erff.cuh').read_text()
    math += '\n__device__ float smoke_shared_erf(float x) { return hex_glibc_erff(x); }\n'
    return math


def new_state(smoke):
    """Declared SAVE initialization plus the common strict device math, column launches."""
    return smoke.SmokeState(math_source=math_source(smoke), columns=True)


def _flush_logs(state, native):
    """Preserve the wrapper's literal messages in the ported framework log."""
    logger = logging.getLogger(__name__)
    while state._pending_logs:
        logger.info('%s', state._pending_logs.popleft())


def install(provider, smoke=None):
    """Install framework services for the route's pin; no native science or native handles."""
    smoke = smoke or pinned(provider.session.config)
    provider._rrfs_smoke_module = smoke
    provider._rrfs_smoke_runtime = {
        'entry': _entry, 'new_state': lambda native, call: new_state(smoke),
        'timer': lambda native, name, start: None,
        'flush_logs': _flush_logs,
    }


def bind_native_pools(provider, call):
    """Use the existing full-allocation binding on resident device authority."""
    if not getattr(provider, '_rrfs_smoke_runtime', None):
        if not getattr(provider.native, 'framework_only', False):
            raise ValueError('native SMOKE replay requires its captured entry-state runtime')
        install(provider)
    return _module(provider).bind_native_pools(provider, call)


def _module(provider):
    smoke = getattr(provider, '_rrfs_smoke_module', None)
    if smoke is None:
        smoke = provider._rrfs_smoke_module = pinned(provider.session.config)
    return smoke


def run_slot(call):
    """The pinned graph's driver slot (its own tables and launcher)."""
    return _module(call.pools['smoke_native']['provider']).run_slot(call)
