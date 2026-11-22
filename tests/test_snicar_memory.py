"""The added SNICAR caches must be priced only when selected."""
from dataclasses import replace
from math import prod
from woof.config import RunConfig
from woof.core.snicar_memory import option_cache_shapes, option_transient_shapes


def test_default_has_no_added_price():
    cfg = RunConfig(nx=6, ny=4, nz=40, dx=3000., dy=3000., ztop=16000., dt=12., run_seconds=0., sf_surface_physics=4, opt_alb=3)
    assert option_cache_shapes(cfg) == {}
    assert option_transient_shapes(cfg) == {}
    inactive = replace(cfg, opt_alb=1, snicar_bandnumber_opt=1, snicar_lap_opt=1)
    assert option_cache_shapes(inactive) == {}
    assert option_transient_shapes(inactive) == {}


def test_selected_cache_matches_device_allocations():
    import cupy as cp
    from woof.core.snow_papers import _device_spectrum, _lap_device, _lap_five_band, _host_spectra
    cfg = RunConfig(nx=6, ny=4, nz=40, dx=3000., dy=3000., ztop=16000., dt=12., run_seconds=0., sf_surface_physics=4, opt_alb=3, snicar_solarspec_opt=3,
                    snicar_bandnumber_opt=1, snicar_lap_opt=1)
    assert len(_host_spectra()[0]) == 12
    arrays = (*_device_spectrum(cp.cuda.Device().id, 3, 1),
              *_lap_device(cp.cuda.Device().id), *_lap_five_band(cp.cuda.Device().id, 3))
    assert sum(a.nbytes for a in arrays) == sum(prod(s) for s in option_cache_shapes(cfg).values())


def test_lap_frame_is_priced():
    from woof.core.noahmp_selected_frames import selected_frames, LAP_FRAME_CEILING
    frames, basis = selected_frames(snow_papers=True, snow_lap=True)
    assert 'snow_lap' not in basis
    assert frames['snow_papers'] >= LAP_FRAME_CEILING
