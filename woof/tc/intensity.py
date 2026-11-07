"""CUDA driver for the field-derived pressure and 150 km wind diagnostics."""
def surface_pressure_hpa(state):
    import cupy as cp
    fields = state.physics.fields
    tv = fields['t2'] * (1.0 + 0.608 * cp.maximum(fields['q2'], 0.0))
    pressure = fields['psfc'] * 0.01 * cp.exp(9.80665 * state.ht / (287.0 * (tv + 0.00325 * state.ht)))
    return cp.where((fields['xland'] > 1.5) & cp.isfinite(pressure), pressure, cp.nan)


def wind_within_radius(state, latitude, longitude, index):
    import cupy as cp
    fields = state.physics.fields
    lat = cp.deg2rad(latitude)
    lon = cp.deg2rad(longitude)
    centre_lat, centre_lon = lat.ravel()[index], lon.ravel()[index]
    h = cp.sin((lat-centre_lat)*.5)**2 + cp.cos(lat)*cp.cos(centre_lat)*cp.sin((lon-centre_lon)*.5)**2
    radius_km = 12742.0 * cp.arcsin(cp.sqrt(cp.clip(h, 0., 1.)))
    speed = cp.hypot(fields['u10'], fields['v10'])
    return cp.max(cp.where((radius_km <= 150.0) & cp.isfinite(speed), speed, -cp.inf))
