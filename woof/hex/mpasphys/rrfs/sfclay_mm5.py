"""RRFS revised MM5 surface layer, shared MMM scheme and fork ice handoff.

MPAS-Model-rrfs 1d6e370e driver_sfclayer.F:606-615,899-972:
strict partial-ice mask, ice-side land flag, and deferred FLHC/FLQC blend.
The MMM a4baf7f scheme is identical to MPAS-A v8.4.2.
"""
from functools import lru_cache
from .. import sfclay_mm5_v842 as shared
from ... import numerics


def cuda_source():
    source = shared.CUDA_SOURCE
    assert source.count('xice[i]<=1.0f') == 1, "RRFS MM5 source drift"
    source = source.replace('xice[i]<=1.0f', 'xice[i]<1.0f')
    assert source.count('ice<=1.f') == 1, "RRFS MM5 source drift"
    source = source.replace('ice<=1.f', 'ice<1.f')
    assert source.count('sea[44*n+i]=2.f;') == 1, "RRFS MM5 source drift"
    source = source.replace('sea[44*n+i]=2.f;', 'a[44*n+i]=1.f;\nsea[44*n+i]=2.f;')
    for field in (22, 23):
        line = f'a[{field}*n+i]=FADD(FMUL(a[{field}*n+i],ice),FMUL(ocean,sea[{field}*n+i]));'
        assert source.count(line) == 1
        source = source.replace(line, '')
    return shared._shared_math_source() + source


@lru_cache(None)
def driver_module(device):
    import cupy as cp
    from ...cuda_backend.aot import direct_image
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(direct_image(cuda_source(), numerics.options(shared.COMPILE_OPTIONS),
                    cp.cuda.Device(device).compute_capability, filename='rrfs_mm5.cu'))
    return module


def bind_native_pools(provider, call):
    packet = shared.bind_native_pools(provider, call)
    if packet['sea_exchange'] is not None:
        for name in ('flhc', 'flqc'):
            packet['sea_exchange'][name] = provider.prepared(name + '_sea')
    return packet


def run_slot(call):
    p = call.pools
    frac = call.options['config_frac_seaice']
    module = driver_module(p['constants'].device.id)
    main, sea, held = shared.sfclayer_from_mpas(
        p['atmosphere'], p['diag_physics'], p['sfc_input'], p['mesh_density'],
        config_len_disp=p['config_len_disp'], config_frac_seaice=frac,
        xice_threshold=p['xice_threshold'], driver_module=module)
    shared.launch_columns(main, p['constants'], scheme=call.scheme)
    if frac:
        shared.launch_columns(sea, p['constants'], scheme=call.scheme)
        for name, destination in p['sea_exchange'].items():
            destination[...] = sea[shared.FIELDS.index(name)]
    shared.sfclayer_to_mpas(main, sea, held, p['diag_physics'], p['sfc_input'],
        config_frac_seaice=frac, xice_threshold=p['xice_threshold'], driver_module=module)


run_native_diagnostics = shared.run_native_diagnostics
