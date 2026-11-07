"""CUDA TC transform of one sealed common-pressure initial frame.

This produces canonical input for either engine's native interpolation. It
never changes later boundary frames and never applies a native index shift.
"""
import hashlib
import json
import math
import time
from pathlib import Path

from . import frames, vortex
from .initialize import pressure_patch

REQUIRED=('air_temperature','specific_humidity','eastward_wind','northward_wind',
          'geopotential_height','surface_pressure','air_pressure_at_mean_sea_level',
          'terrain_height','air_temperature_2m','specific_humidity_2m',
          'eastward_wind_10m','northward_wind_10m')


MSLP='air_pressure_at_mean_sea_level'


def derived_mslp(surface_pressure,terrain_height,temperature_2m,specific_humidity_2m):
    """Sea-level pressure for a source mapping that carries none (the packaged GEFS member profile).

    Hypsometric reduction through a column at the 2 m virtual temperature plus half the standard 6.5 K/km lapse
    over the terrain height; exactly the surface pressure over water at sea level, where TC centres are fixed.
    It feeds only the transform's pressure bound and anchor; it is never published (the source has no MSLP)."""
    try:
        import cupy
        xp=cupy.get_array_module(surface_pressure)
    except ImportError:
        import numpy as xp
    tv=temperature_2m*(1.+.6078*specific_humidity_2m)
    mean=tv+.5*.0065*terrain_height
    return surface_pressure*xp.exp(9.80665*terrain_height/(287.04*mean))


def _axis(frame,name):
    # Coordinate arrays are small metadata, not meteorological computation.
    import struct
    row=frame[name];values=tuple(float(x) for x in row['values'])
    if len(values)!=row['count'] or not all(math.isfinite(v) for v in values):
        raise ValueError(f'invalid {name} coordinate metadata')
    if hashlib.sha256(struct.pack('<'+'d'*len(values),*values)).hexdigest()!=row['sha256']:
        raise ValueError(f'{name} coordinate hash differs')
    return values


def _indices(axis,start,stop,step,*,cyclic=False):
    spacing=axis[1]-axis[0]
    if spacing<=0 or any(not math.isclose(b-a,spacing,abs_tol=1e-9,rel_tol=1e-10) for a,b in zip(axis,axis[1:])):
        raise ValueError('canonical TC lattice must be regular and ascending')
    count=round((stop-start)/step)+1
    selected=[]
    for i in range(count):
        v=start+i*step;f=(v-axis[0])/spacing;n=round(f)
        if not math.isclose(f,n,abs_tol=1e-7):
            raise ValueError('source lattice must contain the one-degree filter nodes')
        if cyclic:n%=len(axis)
        if not 0<=n<len(axis):raise ValueError('initial frame lacks vortex/filter coverage')
        selected.append(n)
    return selected


def relocate_frame(*,library,input_directory,output_directory,valid_time,
                   old_centre,new_centre,source_rmw_m,target_rmw_m,
                   source_radius_m,target_radius_m,maximum_wind_m_s,
                   boundary_layer_pressure_pa,minimum_pressure_pa=None,carried_fields=(),provenance,
                   relative_humidity_convention,size_change_limit=.15,maximum_wind_gain=None):
    """Create a full canonical candidate; diagnose native H(x) before use.

    Geometry/intensity parameters are member-local diagnosed/selected inputs,
    not inferred from a storm name. Provenance binds the observation and target
    selection. Extra atmospheric constituents must be explicitly carried.
    """
    import cupy as cp
    started=time.perf_counter()
    stages={}
    if not math.isfinite(size_change_limit) or not 0.<=size_change_limit<1.:
        raise ValueError('size-change limit must be in [0,1)')
    requested_rmw=target_rmw_m;requested_radius=target_radius_m
    for value in (source_rmw_m,target_rmw_m,source_radius_m,target_radius_m):
        if not math.isfinite(value) or value<=0.:
            raise ValueError('source and requested radii must be positive')
    # DOI 10.3390/atmos11090968, section 3.2: large one-cycle contractions
    # cause model adjustment. Preserve the requested and applied sizes.
    target_rmw_m=min(max(target_rmw_m,(1.-size_change_limit)*source_rmw_m),
                     (1.+size_change_limit)*source_rmw_m)
    target_radius_m=min(max(target_radius_m,(1.-size_change_limit)*source_radius_m),
                        (1.+size_change_limit)*source_radius_m)
    if relative_humidity_convention not in ('water','gfs_mixed'):
        raise ValueError('declare RH convention as water or gfs_mixed')
    original_manifest=(Path(input_directory)/'frames.json').read_bytes()
    doc=json.loads(original_manifest)
    chosen=[f for f in doc['frames'] if f['valid_time']==valid_time]
    if len(chosen)!=1:raise ValueError('select exactly one initial time/member')
    descriptor=chosen[0]
    units=dict(zip(REQUIRED,('K','kg kg-1','m s-1','m s-1','m','Pa','Pa',
                            'm','K','kg kg-1','m s-1','m s-1')))
    # TC-VI1: a mapping without sea-level pressure (packaged GEFS members) gets it derived (derived_mslp), not read
    derive_mslp=not any(f['name']==MSLP for f in descriptor['fields'])
    if derive_mslp:
        units.pop(MSLP)
    for name,unit in units.items():
        rows=[f for f in descriptor['fields'] if f['name']==name]
        if len(rows)!=1 or rows[0]['units']!=unit:
            raise ValueError(f'canonical TC field requires {name} in {unit}')
    if descriptor['vertical_kind']!='pressure' or descriptor['vertical_units']!='Pa':
        raise ValueError('vortex transform requires pressure coordinates in Pa')
    lat=_axis(descriptor,'latitude');lon=_axis(descriptor,'longitude')
    levels=_axis(descriptor,'vertical_values')
    if len(lat)<2 or len(lon)<2:raise ValueError('source horizontal lattice is empty')
    # Unknown 3-D prognostics cannot remain behind when the vortex moves.
    diagnostic={'relative_humidity','air_pressure'}
    extra={f['name'] for f in descriptor['fields'] if f['axes']==['vertical','y','x']}-set(REQUIRED)-diagnostic
    if extra!=set(carried_fields):
        raise ValueError(f'explicit carried constituent inventory must equal {sorted(extra)}')
    if not isinstance(provenance,dict) or not provenance:
        raise ValueError('initialization requires observation/target provenance')
    dl=lat[1]-lat[0];dx=lon[1]-lon[0]
    if not math.isclose(1/dl,round(1/dl),abs_tol=1e-8) or not math.isclose(1/dx,round(1/dx),abs_tol=1e-8):
        raise ValueError('source resolution must divide the one-degree background lattice')
    bounds=[]
    for centre,radius in ((old_centre,source_radius_m),(new_centre,target_radius_m)):
        if len(centre)!=2 or not all(math.isfinite(v) for v in (*centre,radius)) or not 0<radius<=2000000:
            raise ValueError('invalid member vortex geometry')
        d=math.degrees(radius/6371000.)
        if abs(centre[0])+d>=89:raise ValueError('vortex disk approaches a pole')
        e=math.degrees(math.asin(math.sin(math.radians(d))/math.cos(math.radians(centre[0]))))
        x=old_centre[1]+math.remainder(centre[1]-old_centre[1],360.)
        bounds.append((centre[0]-d,centre[0]+d,x-e,x+e))
    south=math.floor(min(b[0] for b in bounds))-12
    north=math.ceil(max(b[1] for b in bounds))+12
    west=math.floor(min(b[2] for b in bounds))-12
    east=math.ceil(max(b[3] for b in bounds))+12
    grid=vortex.Patch(south,west,dl,dx,round((east-west)/dx)+1,round((north-south)/dl)+1)
    coarse_grid=vortex.Patch(south,west,1.,1.,east-west+1,north-south+1)
    grid.validate();coarse_grid.validate()
    cyclic=math.isclose(len(lon)*dx,360.,abs_tol=1e-8)
    iy=_indices(lat,south,north,dl);ix=_indices(lon,west,east,dx,cyclic=cyclic)
    cy=_indices(lat,south,north,1.);cx=_indices(lon,west,east,1.,cyclic=cyclic)
    available={f['name'] for f in descriptor['fields']}
    humidity_diagnostics=tuple(n for n in ('relative_humidity','relative_humidity_2m') if n in available)
    read_fields=tuple(n for n in REQUIRED if not (derive_mslp and n==MSLP))
    arrays,frame,digest=frames.read(library=library,input_directory=input_directory,
        valid_time=valid_time,fields=read_fields+tuple(carried_fields)+humidity_diagnostics)
    if derive_mslp:
        arrays[MSLP]=cp.ascontiguousarray(derived_mslp(arrays['surface_pressure'],arrays['terrain_height'],
            arrays['air_temperature_2m'],arrays['specific_humidity_2m']))
    stages['admission_read_upload']=time.perf_counter()-started
    tick=time.perf_counter()
    if digest!=hashlib.sha256(original_manifest).hexdigest():
        raise ValueError('canonical TC manifest changed during transform admission')
    iy,ix,cy,cx=(cp.asarray(i,dtype=cp.int64) for i in (iy,ix,cy,cx))
    def extract(y,x):
        return {n:cp.ascontiguousarray(v[...,y[:,None],x[None,:]]) for n,v in arrays.items()}
    fine=extract(iy,ix);coarse=extract(cy,cx)
    coefficients=vortex.radial_map(source_rmw=source_rmw_m,target_rmw=target_rmw_m,
        source_outer=source_radius_m,target_outer=target_radius_m)
    patched,receipt=pressure_patch(fine,coarse,grid=grid,coarse_grid=coarse_grid,
        pressure_levels=levels,old_centre=old_centre,new_centre=new_centre,
        source_radius_m=source_radius_m,target_radius_m=target_radius_m,
        map_coefficients=coefficients,maximum_wind_m_s=maximum_wind_m_s,
        boundary_layer_pressure_pa=boundary_layer_pressure_pa,
        minimum_pressure_pa=minimum_pressure_pa,carried_fields=carried_fields,maximum_wind_gain=maximum_wind_gain)
    stages['patch_transform_dispatch']=time.perf_counter()-tick
    tick=time.perf_counter()
    # Keep redundant RH diagnostics consistent with the transformed T/q/p.
    # Invert the canonical WRF-real Bolton conversion on CUDA. Outside changed
    # columns retain the input bits, including original RH below the q floor.
    for name in humidity_diagnostics:
        surface=name.endswith('_2m')
        tn='air_temperature_2m' if surface else 'air_temperature'
        qn='specific_humidity_2m' if surface else 'specific_humidity'
        t=patched[tn];q=patched[qn]
        p=patched['surface_pressure'] if surface else cp.asarray(levels,cp.float64)[:,None,None]
        vapour=p*q/(.622+(1.-.622)*q)
        rh=100.*vapour/(611.2*cp.exp(17.67*(t-273.15)/(t-29.65)))
        if relative_humidity_convention=='gfs_mixed':
            from woof.ingest.horiz import _era5_rh_to_water_gpu
            ratio=_era5_rh_to_water_gpu(cp.full_like(t,100.),t).astype(cp.float64)*.01
            rh/=ratio
        changed=(t!=fine[tn])|(q!=fine[qn])
        if surface:changed|=p!=fine['surface_pressure']
        patched[name]=cp.ascontiguousarray(cp.where(changed,rh,fine[name]))
    updates={}
    for name,value in patched.items():
        if value is fine[name]:continue
        if derive_mslp and name==MSLP:continue
        destination=arrays[name]
        destination[...,iy[:,None],ix[None,:]]=value
        updates[name]=destination
    declaration={'schema':'gpuwm-tc-initial-transform/1','mode':'relocate',
        'valid_time':valid_time,'background_manifest_sha256':digest,
        'old_centre':old_centre,'new_centre':new_centre,'source_rmw_m':source_rmw_m,
        'target_rmw_m':target_rmw_m,'source_radius_m':source_radius_m,
        'requested_target_rmw_m':requested_rmw,'requested_target_radius_m':requested_radius,
        'size_change_limit':size_change_limit,
        **({'maximum_wind_gain':maximum_wind_gain} if maximum_wind_gain is not None else {}),
        'target_radius_m':target_radius_m,'maximum_wind_m_s':maximum_wind_m_s,
        'boundary_layer_pressure_pa':boundary_layer_pressure_pa,
        'minimum_pressure_pa':minimum_pressure_pa,
        'relative_humidity_convention':relative_humidity_convention,
        'carried_fields':list(carried_fields),'provenance':provenance,
        **({'sea_level_pressure':'derived from surface pressure, terrain and 2 m virtual temperature '
            '(the source mapping carries none); bounds and anchors the transform, not published'}
           if derive_mslp else {})}
    encoded=json.dumps(declaration,sort_keys=True,allow_nan=False).encode()
    stages['diagnostic_consistency_dispatch']=time.perf_counter()-tick
    tick=time.perf_counter()
    manifest=frames.publish(library=library,input_directory=input_directory,
        output_directory=output_directory,valid_time=valid_time,
        input_manifest_sha256=digest,transform_sha256=hashlib.sha256(encoded).hexdigest(),fields=updates)
    stages['download_verify_publish']=time.perf_counter()-tick
    receipt.update(declaration=declaration,updated_fields=sorted(updates),
        output_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        stage_wall_seconds=stages,total_wall_seconds=time.perf_counter()-started,
        timing='host wall; asynchronous dispatch stages finish at required publication download; no timing-only synchronization')
    (Path(output_directory)/'tc-transform.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return manifest,receipt
