"""WRF solar generations. The corrected generic generation is WRF 4.8."""
import numpy as np

SOLAR_GENERATIONS = {
    'wrf_461': {'phase_days': 1.0, 'sin_two_coefficient': 0.04089,
                'source': 'WRF v4.6.1 calc_coszen',
                'algorithm': 'wrf-v4.6.1-solar-eot-v1'},
    'wrf_480': {'phase_days': 0.5, 'sin_two_coefficient': 0.040849,
                'source': 'WRF v4.8.0 calc_coszen',
                'algorithm': 'wrf-v4.8.0-solar-eot-v1'},
}
DEFAULT_SOLAR_GENERATION = 'wrf_480'


def solar_generation(value=DEFAULT_SOLAR_GENERATION):
    if value not in SOLAR_GENERATIONS:
        raise ValueError(f'solar_generation={value!r} must be wrf_480 or wrf_461; '
                         'an unknown phase would change COSZEN and surface shortwave')
    return SOLAR_GENERATIONS[value]


def radiation_calendar(start_time, elapsed_seconds, *,
                       generation=DEFAULT_SOLAR_GENERATION):
    """``(julian, xtime, gmt)`` float32 at ``start_time + elapsed_seconds``.

    ``wrf_480`` forms them as WRF's frame does
    (woof.core.wrf_radiation_clock.wrf_calendar_scalars:
    ``REAL(dayOfYear_r8) - 1.0``, ``REAL(days)*24.*60. + REAL(seconds)/60.``,
    and GMT from the start hour), the words the radiation driver receives.
    The earlier statements rounded the day fraction once, as
    ``REAL(day)``; on days whose JULIAN and JULIAN+1 sit in different
    binades (day 0, 1, 3, 7, 15, 31, 63, 127 and 255) that word differs
    from WRF's in about half of the calls, and DECLIN, the equation of
    time, COSZEN and HRANG follow it (tools/wrf480_solar_replay).
    ``wrf_461`` keeps those statements, so the earlier generation stays
    byte-identical.
    """
    solar_generation(generation)
    if generation == 'wrf_480':
        from woof.core.wrf_radiation_clock import wrf_calendar_scalars
        return wrf_calendar_scalars(start_time, elapsed_seconds)
    from datetime import timedelta
    valid = start_time + timedelta(seconds=float(elapsed_seconds))
    hour = (valid.hour + valid.minute / 60.0 + valid.second / 3600.0
            + valid.microsecond / 3.6e9)
    julian = np.float32((int(valid.timetuple().tm_yday) - 1) + hour / 24.0)
    gmt = np.float32(start_time.hour + start_time.minute / 60.0
                     + start_time.second / 3600.0 + start_time.microsecond / 3.6e9)
    return julian, np.float32(float(elapsed_seconds) / 60.0), gmt


def solar_period_seconds(cfg, rounded_seconds):
    """The WRF solar midpoint reads requested RADT, separately from STEPRA."""
    if getattr(cfg, 'solar_generation', DEFAULT_SOLAR_GENERATION) == 'wrf_461':
        return float(rounded_seconds)
    from woof.config import effective_radt_minutes
    return float(effective_radt_minutes(cfg)) * 60.0


def solar_geometry_device(julian, xtime, gmt, latitude, longitude, *,
                          generation=DEFAULT_SOLAR_GENERATION, declination=None):
    """DECLIN/SOLCON, COSZEN and HRANG with WRF REAL(4) operation order."""
    import cupy as cp
    from woof.core.kernels import load_module
    from woof.core.rrtmg_legacy import _DEGRAD, _DPD
    solar_generation(generation)
    lat = cp.ascontiguousarray(cp.asarray(latitude, dtype=cp.float32))
    lon = cp.ascontiguousarray(cp.asarray(longitude, dtype=cp.float32))
    if lat.shape != lon.shape:
        raise ValueError('radiation latitude/longitude shapes must match')
    scalars = np.asarray([julian, xtime, gmt], np.float32)
    if not np.isfinite(scalars).all():
        raise ValueError('solar calendar scalars must be finite')
    module = load_module('rrtmg_legacy_solar')
    ds = cp.empty(2, cp.float32)
    coszen = cp.empty_like(lat)
    hrang = cp.empty_like(lat)
    from woof.wrf_exact import ENABLED as exact
    if declination is None and exact:
        # Strict: WRF's scalar ASIN takes the qualified host C library's
        # binary32 rounding (woof.static.rust_bridge.scalar_reference_asin_f32),
        # as the wrf_461 route (rrtmg_legacy.solar_geometry_device) always
        # has.  Named breakage: the device transcription is glibc 2.43's
        # correctly rounded asinf, and a referee wrf.exe on glibc 2.39 rounds
        # DECLIN one ulp apart on some calls (A071 on WRF 4.8.0, 2024-05-21
        # 21Z, the step-169 radiation call; WRF-480 assembly gate).
        from woof.static.rust_bridge import scalar_reference_asin_f32
        argument = cp.empty(1, cp.float32)
        module.get_function('rls_radconst_with_argument')((1,), (1,),
            (scalars[0], _DEGRAD, _DPD, ds, argument))
        ds[0] = np.float32(scalar_reference_asin_f32(cp.asnumpy(argument))[0])
    else:
        module.get_function('rls_radconst')((1,), (1,),
            (scalars[0], _DEGRAD, _DPD, ds))
    if declination is not None:
        ds[0] = np.float32(declination)
    if lat.size:
        module.get_function('rls_calc_coszen_hrang')(((lat.size+255)//256,), (256,),
            (np.int64(lat.size), lat, lon, *scalars, _DEGRAD,
             np.int32(generation == 'wrf_480'), ds, coszen, hrang))
    return ds, coszen, hrang


def geometry_at(valid_time, latitude, longitude, *, hour_offset_seconds=0.,
                generation=DEFAULT_SOLAR_GENERATION, start_time=None,
                elapsed_seconds=0.):
    """Use the delivered call calendar; shift only XTIME for the midpoint."""
    if not np.isfinite(hour_offset_seconds):
        raise ValueError('hour_offset_seconds must be finite')
    if start_time is None:
        start_time = valid_time
        elapsed_seconds = 0.
    if generation == 'wrf_480':
        # WRF's frame words: JULIAN at the call time, XTIME since the start,
        # GMT the start hour (radiation_calendar); then XTIME + RADT*0.5.
        julian = radiation_calendar(valid_time, 0.)[0]
        _, xtime, gmt = radiation_calendar(start_time, elapsed_seconds)
        xtime = np.float32(xtime + np.float32(hour_offset_seconds/60.))
        return solar_geometry_device(julian, xtime, gmt, latitude, longitude,
                                     generation=generation)
    hour = (valid_time.hour + valid_time.minute/60. + valid_time.second/3600.
            + valid_time.microsecond/3.6e9)
    gmt = (start_time.hour + start_time.minute/60. + start_time.second/3600.
           + start_time.microsecond/3.6e9)
    # WRF adds separately rounded XTIME and RADT*0.5 in minute units.
    xtime = np.float32(np.float32(elapsed_seconds/60.) + np.float32(hour_offset_seconds/60.))
    return solar_geometry_device(valid_time.timetuple().tm_yday-1+hour/24.,
        xtime, gmt, latitude, longitude,
        generation=generation)
