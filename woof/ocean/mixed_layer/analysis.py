"""Public analysis decode in Rust, interpolation on the GPU."""
import numpy as np

SOURCES = {
    'rtofs': {'temperature':'temperature','salinity':'salinity','u':'u','v':'v',
              'latitude':'Latitude','longitude':'Longitude','depth':'Depth',
              'temperature_units':'degC','time':'MT','max_age_hours':24,
              'terms':'https://www.noaa.gov/disclaimer'},
    'cf': {'temperature':'thetao','salinity':'so','u':'uo','v':'vo',
           'latitude':'lat','longitude':'lon','depth':'depth',
           'temperature_units':'degC','time':'time','max_age_hours':24,
           'terms':'caller must retain source licence'},
    'hycom': {'temperature':'water_temp','salinity':'salinity','u':'water_u','v':'water_v',
              'latitude':'lat','longitude':'lon','depth':'depth',
              'temperature_units':'degC','time':'time','max_age_hours':24,
              'terms':'https://www.hycom.org/dataserver/access-methods/http-https'},
}


def nearest_indices(latitude, longitude, land, source_latitude, source_longitude, valid, levels):
    """Exact spherical wet-neighbor mapping, with a one-degree search cap."""
    import cupy as cp
    from . import kernels
    slat,slon=(cp.ascontiguousarray(a,dtype=cp.float32) for a in (source_latitude,source_longitude))
    ns=slat.size;nt=land.size
    surface=cp.ascontiguousarray(cp.where(valid,cp.float32(300),cp.float32('nan')))
    module=kernels(levels);indices=cp.empty(land.shape,dtype=cp.int32)
    counts=cp.zeros(180*360,dtype=cp.int32);offsets=cp.zeros(180*360+1,dtype=cp.int32)
    cursor=cp.zeros_like(counts);members=cp.empty(ns,dtype=cp.int32)
    module.get_function('ocean_bin_count')(((ns+255)//256,),(256,),
        (slat,slon,surface,counts,np.int32(ns)))
    cp.cumsum(counts,dtype=cp.int32,out=offsets[1:])
    module.get_function('ocean_bin_scatter')(((ns+255)//256,),(256,),
        (slat,slon,surface,offsets,cursor,members,np.int32(ns)))
    module.get_function('ocean_nearest')((nt,),(256,),
        (slat,slon,offsets,members,
         cp.ascontiguousarray(cp.asarray(latitude,dtype=cp.float32)),
         cp.ascontiguousarray(cp.asarray(longitude,dtype=cp.float32)),land,indices,
         np.int32(nt)))
    return indices


def load(path, *, latitude, longitude, land, levels, source='rtofs', names=None, rotation=None, expected_time=None, layer_m=5.0, uncovered="refuse"):
    """Read one regional analysis and remap to the model columns on CUDA.

    A wet source must lie within one degree. Profiles are piecewise-linearly integrated into layer means, with
    layer thickness given by layer_m (5 m by default). WRF's flat-bottom ocean requires complete profiles to
    that depth. Coastal points use the nearest complete offshore profile
    within one degree. Uncovered water is refused by default; the explicit
    prescribed policy returns an inactive mask and keeps it outside PWP. This does not supply
    bathymetry. The configured uniform MLD fallback is labelled separately.
    """
    import cupy as cp
    import math
    if uncovered not in ('refuse', 'prescribed'):
        raise ValueError('uncovered ocean policy must be refuse or prescribed')
    if not math.isfinite(layer_m) or layer_m <= 0:
        raise ValueError("ocean layer thickness must be positive and finite")
    from woof import netcdf_bridge
    from . import kernels
    from datetime import datetime, timezone
    if expected_time is None:
        raise ValueError('ocean analysis needs the forecast UTC start time to reject stale or future analyses')
    if isinstance(expected_time,str):
        expected_time=datetime.fromisoformat(expected_time.replace('Z','+00:00'))
    if expected_time.tzinfo is not None:
        expected_time=expected_time.astimezone(timezone.utc).replace(tzinfo=None)
    row=dict(SOURCES[source]);row.update(names or {})
    latitude,longitude=(cp.ascontiguousarray(cp.asarray(a,dtype=cp.float32)) for a in (latitude,longitude))
    if latitude.shape!=land.shape or longitude.shape!=land.shape or bool(cp.any((land>1.5)&(~cp.isfinite(latitude)|~cp.isfinite(longitude)|(cp.abs(latitude)>90)))):
        raise ValueError('ocean remapping requires finite forecast latitude and longitude planes on its ocean grid')
    keys=('temperature','salinity','u','v','latitude','longitude','depth')
    with netcdf_bridge.open_dataset(path) as ds:
        required=[row[key] for key in (*keys,'time')]
        missing=set(required)-set(ds.variables)
        if missing:
            raise ValueError(f'ocean analysis {path} is missing variables {sorted(missing)}')
        ds.prefetch(required)
        times=ds.variables[row['time']].times()
        if len(times)!=1 or not 0 <= (expected_time-times[0]).total_seconds() <= row['max_age_hours']*3600:
            raise ValueError(f'ocean analysis needs one time within the preceding {row["max_age_hours"]} hours of {expected_time}; got {times}')
        values={key:cp.asarray(ds.variables[row[key]][...],dtype=cp.float32) for key in keys}
        units=str(getattr(ds.variables[row['temperature']],'units',row['temperature_units']))
        allowed={'salinity':('psu','1e-3','practical_salinity_unit'),
                 'u':('m/s','m s-1','m.s-1','meter second-1'),
                 'v':('m/s','m s-1','m.s-1','meter second-1'),
                 'depth':('m','meter','meters','metre','metres'),
                 'latitude':('degrees_north','degree_north'),
                 'longitude':('degrees_east','degree_east')}
        for key,choices in allowed.items():
            unit=str(getattr(ds.variables[row[key]],'units','')).strip().lower()
            if unit not in choices:
                raise ValueError(f'ocean analysis {key} units {unit!r} must be one of {choices}')
    lat,lon=values['latitude'].squeeze(),values['longitude'].squeeze()
    if lat.ndim==lon.ndim==1:
        lon,lat=cp.meshgrid(lon,lat)
    if lat.shape != lon.shape or lat.ndim!=2:
        raise ValueError('ocean analysis needs matching 2D grids or 1D coordinate axes')
    depth=values['depth'].ravel();nk=depth.size;ns=lat.size;nt=land.size
    if nk<2 or bool(cp.any(~cp.isfinite(depth))) or bool(cp.any(depth[1:]<=depth[:-1])):
        raise ValueError('ocean analysis depths must be finite and strictly increasing')
    if float(depth[-1].item())<layer_m*levels:
        raise ValueError(f'ocean analysis does not reach the deepest configured layer at {layer_m*levels} m')
    if units.lower() not in ('degc','degrees_c','degree_celsius','degrees_celsius','celsius','k','kelvin'):
        raise ValueError(f'ocean analysis temperature units {units!r} are not Celsius or Kelvin')
    celsius=units.lower() not in ('k','kelvin')
    arrays={}
    for key in ('temperature','salinity','u','v'):
        a=values[key]
        if a.ndim==4 and a.shape[0]==1:a=a[0]
        if a.shape!=(nk,*lat.shape):
            raise ValueError(f'ocean analysis {key} must have one analysis time and shape {(nk,*lat.shape)}, got {a.shape}')
        arrays[key]=cp.ascontiguousarray(a)
    surface=arrays['temperature'][0]
    if celsius:surface=surface+cp.float32(273.15)
    # Search complete profiles rather than selecting a shallow/masked source
    # and discovering its missing deep layers after the horizontal mapping.
    upper=int(cp.searchsorted(depth,float(layer_m*levels)).item())+1
    valid=cp.ones(lat.shape,dtype=cp.bool_)
    for key,a in arrays.items():
        v=a[:upper]
        low,high=((-13.15,56.85) if celsius else (260,330)) if key=='temperature' else ((0,50) if key=='salinity' else (-20,20))
        valid &= cp.all(cp.isfinite(v)&(v>=low)&(v<=high),axis=0)
    surface=cp.ascontiguousarray(cp.where(valid,surface,cp.float32('nan')))
    module=kernels(levels)
    indices=nearest_indices(latitude,longitude,land,lat,lon,valid,levels)
    missing = int(cp.count_nonzero(indices == -1).item())
    if missing and uncovered == 'refuse':
        raise ValueError(f'ocean analysis lacks a complete wet source profile within one degree for {missing} forecast water cells')
    if missing:
        import logging
        logging.getLogger(__name__).warning('Ocean analysis coverage: %d water cells stay prescribed; no complete profile within one degree', missing)
    active = (indices >= 0).astype(cp.float32)
    indices = cp.where(indices < 0, cp.int32(-2), indices)
    result={'om_active': active};z=cp.empty((levels,*land.shape),dtype=cp.float32)
    for key,name in (('temperature','om_tmp'),('salinity','om_s'),('u','om_u'),('v','om_v')):
        out=cp.empty_like(z)
        module.get_function('ocean_profiles')(((out.size+127)//128,),(128,),
            (arrays[key],depth,indices,out,z,np.int32(ns),np.int32(nt),np.int32(nk),np.int32(celsius and key=='temperature'),np.float32(layer_m)))
        result[name]=out
    result['om_depth']=z
    if rotation is None:
        raise ValueError('ocean analysis earth-relative currents require the forecast grid rotation (SINALPHA, COSALPHA)')
    sina,cosa=(cp.asarray(a,dtype=cp.float32) for a in rotation)
    if sina.shape!=land.shape or cosa.shape!=land.shape or bool(cp.any(~cp.isfinite(sina)|~cp.isfinite(cosa)|(cp.abs(sina*sina+cosa*cosa-1)>1.e-4))):
        raise ValueError('ocean current rotation must contain finite unit sine/cosine planes on the forecast grid')
    u,v=result['om_u'],result['om_v']
    result['om_u']=cp.ascontiguousarray(u*cosa+v*sina)
    result['om_v']=cp.ascontiguousarray(v*cosa-u*sina)
    import logging
    logging.getLogger(__name__).info('Ocean initialized: source=%s analysis=%s forecast=%s file=%s grid=%s levels=%d terms=%s',
        source,times[0].isoformat(),expected_time.isoformat(),path,land.shape,levels,row['terms'])
    return result
