"""RRFS Noah sea ice: shared thermal solver, fork partial-ice driver mask.

MPAS-Model-rrfs 1d6e370e driver_seaice.F:436,570-601 excludes full ice
from reconstruction and performs flux reconstruction only for fractional ice.
The Noah sea-ice scheme and its thermal solver are unchanged from v8.4.2.
"""
from functools import lru_cache
from importlib.resources import files
from .. import seaice_v842 as shared
from ... import numerics


def cuda_source(nsoil):
    assert shared._CORE_SOURCE.count('F(xice)<=1.0f') == 2, 'RRFS Noah sea-ice source drift'
    source = shared._CORE_SOURCE.replace('F(xice)<=1.0f', 'F(xice)<1.0f')
    return (f'#define NSOLD {nsoil}\ntypedef float real;\n'
            + files('woof.core.kernels').joinpath('mynn_libm.cuh').read_text() + source)


@lru_cache(None)
def driver_module(device, nsoil):
    import cupy as cp
    from ...cuda_backend.aot import direct_image
    source = cuda_source(nsoil)
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(direct_image(source, numerics.options(('--std=c++11', '--fmad=false',
                    '--ftz=false', '--prec-div=true', '--prec-sqrt=true'), keep_ftz=True),
                    cp.cuda.Device(device).compute_capability, filename='rrfs_noah_seaice.cu'))
    return module


def launch_slot(call):
    p = call.pools
    frac = call.options['config_frac_seaice']
    options = shared.options_from_MPAS(call.dt, config_frac_seaice=frac)
    diag, surface, prepared = p['diag_physics'], p['sfc_input'], p['seaice_shared']
    module = driver_module(surface['tslb'].device.id, surface['tslb'].shape[0])
    work = shared.seaice_from_MPAS(diag, surface, prepared, options,
        lsm_scheme=call.scheme, microp_scheme=call.options['config_microp_scheme'],
        convection_scheme=call.options['config_convection_scheme'], driver_module=module)
    shared.launch_seaice_noah(work, options, module_state=p.get('seaice_module_state'))
    shared.seaice_to_MPAS(work, diag, surface, options, lsm_scheme=call.scheme,
        shared=prepared, driver_module=module, blend=frac)
    return work
