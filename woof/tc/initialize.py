"""Member-local common-pressure TC transform, before native initialization.

Physics: independently implemented from DOI 10.3390/atmos11090968 and
10.1175/1520-0493(1993)121<2030:AISOHM>2.0.CO;2. Native pressure-coordinate
interpolation, source provenance and ensemble target selection are caller
contracts. This module does not acquire observations or run assimilation.
"""
import bisect
import math
from . import vortex


def pressure_patch(fields, coarse_fields, *, grid, coarse_grid, pressure_levels,
                   old_centre, new_centre, source_radius_m, target_radius_m,
                   map_coefficients, maximum_wind_m_s, boundary_layer_pressure_pa,
                   minimum_pressure_pa=None, profile_points=257, carried_fields=(), maximum_wind_gain=None):
    """Return a separate initialized pressure patch and diagnostic declarations.

    All fields are resident CUDA FP64, with regular ascending geographic axes.
    Pressure levels ascend top to bottom. ``coarse_fields`` contains the same
    fields at one-degree spacing, including the published filter's full halo.
    The supplied boundary-layer pressure must be diagnosed for this member
    (the reference formulation uses sigma=0.85); it is never a 10-m wind.

    The resolved Vmax constrains the wind gain. An optional minimum MSLP
    limits that SAME gain before wind, pressure and temperature are changed.
    If pressure limits the gain, Vmax can remain below its target; the receipt
    reports this explicitly. No independent pressure clipping is applied.
    RMW/R34 target
    selection owns the supplied monotone map. No native state is mutated.
    """
    import cupy as cp
    required=("air_temperature","specific_humidity","eastward_wind","northward_wind",
              "geopotential_height","surface_pressure","air_pressure_at_mean_sea_level",
              "terrain_height","air_temperature_2m","specific_humidity_2m",
              "eastward_wind_10m","northward_wind_10m")
    if any(n not in fields or n not in coarse_fields for n in required):
        raise ValueError("complete thermodynamic, MSLP and 10-m source fields are required")
    levels=tuple(float(v) for v in pressure_levels);K=len(levels)
    if K<2 or not all(math.isfinite(v) and v>0. for v in levels) or any(a>=b for a,b in zip(levels,levels[1:])):
        raise ValueError("pressure metadata must ascend from model top")
    if not math.isfinite(boundary_layer_pressure_pa) or not levels[0]<=boundary_layer_pressure_pa<=levels[-1]:
        raise ValueError("diagnosed boundary-layer pressure is outside the source levels")
    if not isinstance(profile_points,int) or not 3<=profile_points<=1024:
        raise ValueError("profile resolution must be 3..1024 points")
    for collection,g in ((fields,grid),(coarse_fields,coarse_grid)):
        for name in required:
            value=collection[name]
            shape=(K,g.ny,g.nx) if name in required[:5] else (g.ny,g.nx)
            if not isinstance(value,cp.ndarray) or value.shape!=shape or value.dtype!=cp.float64 or not value.flags.c_contiguous:
                raise ValueError(f"invalid pressure-patch storage for {name}")
            if not bool(cp.isfinite(value).all()):
                raise ValueError(f"missing pressure-patch values in {name}")
    for collection in (fields,coarse_fields):
        if not bool(((collection['specific_humidity']>=0.)&(collection['specific_humidity']<1.)).all()) or not bool(((collection['specific_humidity_2m']>=0.)&(collection['specific_humidity_2m']<1.)).all()):
            raise ValueError("source specific humidity is outside [0,1)")
    geometry=dict(old_centre=old_centre,new_centre=new_centre,
                  source_radius_m=source_radius_m,target_radius_m=target_radius_m,
                  map_coefficients=map_coefficients)
    radii=cp.linspace(0.,source_radius_m,profile_points,dtype=cp.float64)
    a,b=map_coefficients;target_radii=cp.ascontiguousarray(a*radii+.5*b*radii*radii)
    def stack(collection,names):
        return cp.ascontiguousarray(cp.concatenate([collection[n].reshape(-1,collection[n].shape[-2],collection[n].shape[-1]) for n in names]))
    def transform(names,vector=False,fine=None,coarse_source=None):
        original=stack(fields if fine is None else fine,names)
        coarse=stack(coarse_fields if coarse_source is None else coarse_source,names)
        basic=vortex.basic_field(coarse,coarse_grid)
        moved=(vortex.relocate_vectors if vector else vortex.relocate_scalars)(original,grid,basic,coarse_grid,**geometry)
        profiles=vortex.axisymmetric_profiles(original,grid,basic,coarse_grid,radii,
            centre=old_centre,source_radius_m=source_radius_m,vector=vector)
        return moved,profiles
    def projected(profiles,**kwargs):
        return vortex.profile_field(radii,cp.ascontiguousarray(profiles),grid,centre=new_centre,
            target_radius_m=target_radius_m,map_coefficients=map_coefficients,**kwargs)
    wind10,profile10=transform(('eastward_wind_10m','northward_wind_10m'),True)
    # Intensity correction changes the tangential component only. The radial
    # flow is retained, avoiding an independent imposed divergence change.
    profile10[1:]=0.
    axis10=projected(profile10,vector=True)
    # The outside sentinel identifies the complete destination disk without
    # inferring it from a nonzero wind or temperature anomaly.
    disk=cp.ascontiguousarray(projected(cp.ones((1,profile_points),cp.float64))[0]!=0.,dtype=cp.uint8)
    gain=vortex.intensity_gain(wind10,axis10,disk,target_maximum_m_s=maximum_wind_m_s)
    requested_gain=float(gain)
    if maximum_wind_gain is not None:
        # Per-cycle bound on the intensity correction (the wind-gain analogue of the 15 % size bound). Breakage it
        # prevents (hex the storm GEFS members, 2026-10-27): a broad 0.5-degree member vortex (PRMSL ~1000 hPa, RMW
        # 60 km) taken to the 77 m/s vitals needed gains of 2.7-3.9; the thermal balance scales the warm-core anomaly
        # with the gradient-wind ratio, which gave +47 K at 300 hPa and 60 g/kg vapour at 350 hPa, and every hex
        # forecast from it went non-finite after 46 steps. The analysis supplies the rest of the core.
        if not (math.isfinite(maximum_wind_gain) and maximum_wind_gain >= 1.):
            raise ValueError("maximum wind gain must be finite and at least 1")
        gain=cp.minimum(gain,maximum_wind_gain)
    wind,wind_profiles=transform(('eastward_wind','northward_wind'),True)
    wind_profiles[K:]=0.
    # Diagnose the signed tangential profile at the member's boundary-layer
    # pressure; vertical interpolation is linear in log pressure.
    hi=min(max(bisect.bisect_right(levels,boundary_layer_pressure_pa),1),K-1);lo=hi-1
    w=math.log(boundary_layer_pressure_pa/levels[lo])/math.log(levels[hi]/levels[lo])
    old_bl=cp.ascontiguousarray((1.-w)*wind_profiles[lo]+w*wind_profiles[hi])
    coriolis=dict(source_coriolis=2*7.292115e-5*math.sin(math.radians(old_centre[0])),
                  target_coriolis=2*7.292115e-5*math.sin(math.radians(new_centre[0])))
    mslp,mslp_profiles=transform(('air_pressure_at_mean_sea_level',))
    axis_pressure=projected(mslp_profiles)[0]
    wind_gain=gain
    if minimum_pressure_pa is not None:
        # Equation 48's streamfunction is quadratic in the common wind gain:
        # the centrifugal term is g^2, the Coriolis term g. Evaluate the two
        # coefficients, then intersect all resolved pressure inequalities.
        # This is an observation constraint on the published balance relation,
        # not an independent MSLP/temperature correction or a forecast sweep.
        g1=vortex.gradient_wind_ratio(radii,old_bl,target_radii,old_bl,**coriolis)
        g2=vortex.gradient_wind_ratio(radii,old_bl,target_radii,cp.ascontiguousarray(2.*old_bl),**coriolis)
        quadratic=.5*(g2-2.*g1);linear=g1-quadratic
        gain=cp.minimum(gain,vortex.pressure_gain_limit(mslp[0],axis_pressure,
            projected(quadratic[None])[0],projected(linear[None])[0],disk,
            minimum_pressure_pa=minimum_pressure_pa))
    corrected_wind=wind+(gain-1.)*projected(wind_profiles,vector=True)
    corrected_wind10=wind10+(gain-1.)*axis10
    resolved_wind=cp.where(disk!=0,cp.hypot(*corrected_wind10),0.).max()
    if not bool(resolved_wind<=maximum_wind_m_s+1.e-8):
        raise ValueError('pressure-constrained gain cannot retain the requested wind ceiling and asymmetry')
    new_bl=cp.ascontiguousarray(old_bl*gain)
    gamma=vortex.gradient_wind_ratio(radii,old_bl,target_radii,new_bl,**coriolis)
    gamma_field=projected(gamma[None],outside=1.)[0]
    thermal,thermal_profiles=transform(('air_temperature','air_temperature_2m'))
    # Vapour is relocated as ln q: the separated environment and the moved anomaly are multiplicative, so the result
    # is positive by construction (exp) instead of an additive anomaly landing on a drier environment. Breakage it
    # prevents (hex the storm GEFS members, 2026-10-27): the 0.5-degree GEFS pressure levels carry packed q = 0 cells,
    # and the additive relocation of gep02-gep04 drove q to -3.6e-5 kg/kg at 300-600 hPa, so every member but gep01
    # refused here and no ensemble could be vortex-initialized. Zero source vapour enters as Q_FLOOR (1e-8 kg/kg,
    # below the source packing precision); outside the two disks the input bits are kept (the affected mask below).
    Q_FLOOR=1.e-8
    q_names=('specific_humidity','specific_humidity_2m')
    def log_q(collection):
        return {n:cp.ascontiguousarray(cp.log(cp.maximum(collection[n],Q_FLOOR))) for n in q_names}
    log_moisture,_=transform(q_names,fine=log_q(fields),coarse_source=log_q(coarse_fields))
    moisture=cp.ascontiguousarray(cp.exp(log_moisture))
    if not bool(((moisture>0.)&(moisture<1.)).all()):
        raise ValueError("vortex separation creates invalid humidity; no clipping applied")
    mixing=cp.ascontiguousarray(moisture/(1.-moisture))
    temperature,mixing=vortex.thermal_balance(thermal,mixing,projected(thermal_profiles),cp.ascontiguousarray(gamma_field))
    specific=mixing/(1.+mixing)
    mslp=cp.ascontiguousarray(mslp[0]+(gamma_field-1.)*axis_pressure)
    resolved_pressure=cp.where(disk!=0,mslp,cp.inf).min()
    if minimum_pressure_pa is not None and not bool(resolved_pressure>=minimum_pressure_pa-1.e-6):
        raise ValueError('balanced pressure constraint failed at the resolved source grid')
    old_q=fields['specific_humidity']/(1.-fields['specific_humidity'])
    height,surface_pressure=vortex.pressure_anchor(cp.asarray(levels,cp.float64),fields['air_temperature'],
        cp.ascontiguousarray(old_q),cp.ascontiguousarray(temperature[:K]),cp.ascontiguousarray(mixing[:K]),
        fields['geopotential_height'],old_mslp=fields['air_pressure_at_mean_sea_level'],new_mslp=mslp,
        old_surface_pressure=fields['surface_pressure'],terrain_height=fields['terrain_height'],
        old_surface_virtual_temperature=cp.ascontiguousarray(fields['air_temperature_2m']*(1.+(1./.622-1.)*fields['specific_humidity_2m'])),
        new_surface_virtual_temperature=cp.ascontiguousarray(temperature[K]*(1.+(1./.622-1.)*specific[K])))
    output=dict(fields)
    output.update(air_temperature=cp.ascontiguousarray(temperature[:K]),specific_humidity=cp.ascontiguousarray(specific[:K]),
        eastward_wind=cp.ascontiguousarray(corrected_wind[:K]),northward_wind=cp.ascontiguousarray(corrected_wind[K:]),
        air_temperature_2m=cp.ascontiguousarray(temperature[K]),specific_humidity_2m=cp.ascontiguousarray(specific[K]),
        eastward_wind_10m=cp.ascontiguousarray(corrected_wind10[0]),northward_wind_10m=cp.ascontiguousarray(corrected_wind10[1]),
        air_pressure_at_mean_sea_level=mslp,geopotential_height=height,surface_pressure=surface_pressure)
    # Carried condensate/moment pairs are selected together by the model
    # inventory, never silently left at the removed vortex's old location.
    for name in carried_fields:
        if name in required or name not in fields or name not in coarse_fields:
            raise ValueError("carried fields must be extra paired model constituents")
        for collection,g in ((fields,grid),(coarse_fields,coarse_grid)):
            value=collection[name]
            if not isinstance(value,cp.ndarray) or value.shape!=(K,g.ny,g.nx) or value.dtype!=cp.float64 or not value.flags.c_contiguous:
                raise ValueError(f"invalid carried constituent {name}")
            if not bool((cp.isfinite(value) & (value >= 0.)).all()):
                raise ValueError(f"invalid source constituent {name}: finite nonnegative values required")
        moved=vortex.relocate_constituents(fields[name],grid,**geometry)
        if not bool(cp.isfinite(moved).all()) or not bool((moved>=0.).all()):
            raise ValueError(f"bounded constituent remap creates invalid {name}")
        output[name]=moved
    old_disk=vortex.profile_field(radii,cp.ones((1,profile_points),cp.float64),grid,
        centre=old_centre,target_radius_m=source_radius_m)[0]!=0.
    affected=old_disk|(disk!=0)
    for name,value in tuple(output.items()):
        if value is not fields[name]:
            output[name]=cp.ascontiguousarray(cp.where(affected,value,fields[name]))
    return output,{'wind_gain':float(gain),'requested_wind_gain':requested_gain,
        'maximum_wind_gain':maximum_wind_gain,'boundary_layer_pressure_pa':boundary_layer_pressure_pa,
        'unconstrained_wind_gain':float(wind_gain),'pressure_limited':bool(gain<wind_gain),
        'minimum_pressure_constraint_pa':minimum_pressure_pa,
        'resolved_maximum_wind_m_s':float(resolved_wind),
        'resolved_minimum_pressure_pa':float(resolved_pressure),
        'profile_points':profile_points,'source_centre_deg':list(old_centre),'target_centre_deg':list(new_centre),
        'constituent_remap':'full-field convex donor/boundary blend; shared weights for mass and number; nonnegative, not area-conservative under resizing',
        'pressure_constraint':'gradient-wind ratio; native MSLP and spin-up checks required',
        'height_anchor':'fixed terrain and sea-level pressure; prior discrete hydrostatic residual retained',
        'qualification':'pressure-patch candidate; enclosing native initialization required'}
