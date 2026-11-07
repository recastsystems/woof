"""CUDA vortex separation on a common isobaric regular latitude/longitude patch.

The native engine adapter supplies interpolated pressure-level fields and the
coarse background patch. No eta-coordinate or mesh index displacement is valid
input. This module does not perform data I/O or assimilation.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math


@dataclass(frozen=True)
class Patch:
    latitude0: float
    longitude0: float
    dlatitude: float
    dlongitude: float
    nx: int
    ny: int

    def validate(self):
        if self.nx < 2 or self.ny < 2 or self.dlatitude <= 0 or self.dlongitude <= 0:
            raise ValueError("vortex patch requires ascending regular axes with at least two points")
        if not all(math.isfinite(v) for v in (self.latitude0, self.longitude0, self.dlatitude, self.dlongitude)):
            raise ValueError("nonfinite vortex patch geometry")
        if self.latitude0 <= -89 or self.latitude0+(self.ny-1)*self.dlatitude >= 89:
            raise ValueError("vortex patch must exclude the polar coordinate singularity")
        if (self.nx-1)*self.dlongitude >= 180:
            raise ValueError("regional vortex patch must span less than 180 degrees")


@lru_cache(maxsize=1)
def _module():
    import cupy as cp
    from woof.certify.kernel_manifest import record_module
    path = Path(__file__).parents[1] / "core/kernels/vortex_init.cu"
    source = path.read_text()
    options = ("--std=c++17",)
    module = cp.RawModule(code=source, options=options)
    record_module("tc_vortex_init", source=source, options=options, module=module)
    return module


def _field(value, grid):
    import cupy as cp
    grid.validate()
    if not isinstance(value, cp.ndarray) or value.dtype != cp.float64 or value.ndim != 3 or value.shape[1:] != (grid.ny, grid.nx) or not value.flags.c_contiguous:
        raise ValueError("vortex fields must be contiguous CUDA float64[plane,latitude,longitude]")
    if value.shape[0] < 1:
        raise ValueError("vortex patch has no field planes")


def basic_field(coarse, grid):
    """Published 11-pass filter, zonal then meridional, on ONE-DEGREE fields.

    Its unchanged edges require a full 11-degree halo around the storm filter
    disk. A fine-grid one-cell smoother is not this published filter.
    """
    import cupy as cp
    import numpy as np
    _field(coarse, grid)
    if grid.dlatitude != 1. or grid.dlongitude != 1.:
        raise ValueError("background filter stencil spacing must be one geographic degree")
    work = coarse.copy(); scratch = cp.empty_like(work)
    kernel = _module().get_function("tc_filter")
    for axis in (0, 1):
        for wavelength in (2,3,4,2,5,6,7,2,8,9,2):
            coefficient = .5/(1.-math.cos(2.*math.pi/wavelength))
            kernel(((work.size+255)//256,), (256,),
                   (work,scratch,np.int32(grid.nx),np.int32(grid.ny),np.int32(work.shape[0]),np.int32(axis),np.float64(coefficient)))
            work,scratch = scratch,work
    return work


def radial_map(*, source_rmw, target_rmw, source_outer, target_outer):
    """Monotone quadratic radius map through RMW and the vortex outer radius.

    Scalars are independently diagnosed per member; the driver preserves prior
    ensemble anomalies when constructing target radii. No case-specific limits.
    """
    values = (source_rmw,target_rmw,source_outer,target_outer)
    if not all(math.isfinite(v) and v>0 for v in values) or source_rmw>=source_outer or target_rmw>=target_outer:
        raise ValueError("positive RMW smaller than outer radius required")
    rm,ro,Rm,Ro = source_rmw,target_rmw,source_outer,target_outer
    a=(ro*Rm*Rm-rm*rm*Ro)/(Rm*rm*(Rm-rm))
    b=2.*(Ro*rm-Rm*ro)/(Rm*rm*(Rm-rm))
    if a<=0 or a+b*Rm<=0:
        raise ValueError("requested size correction folds the radius map")
    return a,b


def _disk_inside(grid, latitude, longitude, radius_m, *, halo_deg=0.):
    delta = radius_m/6371000.*180./math.pi
    if abs(latitude)+delta>=89:
        raise ValueError("vortex filter disk approaches a geographic pole")
    # Conservative spherical longitude bound; both poles excluded above.
    lon_delta=math.degrees(math.asin(min(1.,math.sin(math.radians(delta))/math.cos(math.radians(latitude)))))
    offset=math.remainder(longitude-grid.longitude0,360.)
    if (latitude-delta < grid.latitude0+halo_deg or
        latitude+delta > grid.latitude0+(grid.ny-1)*grid.dlatitude-halo_deg or
        offset-lon_delta < halo_deg or offset+lon_delta > (grid.nx-1)*grid.dlongitude-halo_deg):
        raise ValueError("vortex disk and published filter halo do not fit the supplied patch")


def _relocate(fields, grid, basic, coarse_grid, *, old_centre, new_centre,
                     source_radius_m, target_radius_m=None, map_coefficients=(1.,0.), angles=24, vector=False):
    """Remove the diagnosed vortex, then insert its translated/resized anomaly.

    The input planes must be scalar isobaric fields. Wind vectors need the
    vector adapter, and intensity/balance adjustments are separate operations.
    Returns a separate device array; native states are not mutated.
    """
    import cupy as cp
    import numpy as np
    _field(fields,grid);_field(basic,coarse_grid)
    if basic.shape[0]!=fields.shape[0] or coarse_grid.dlatitude!=1. or coarse_grid.dlongitude!=1.:
        raise ValueError("basic field must match every plane on the one-degree grid")
    if not isinstance(angles,int) or not 8<=angles<=256:
        raise ValueError("boundary quadrature requires 8..256 azimuths")
    target_radius_m=source_radius_m if target_radius_m is None else target_radius_m
    a,b=map_coefficients
    if not all(math.isfinite(v) for v in (*old_centre,*new_centre,source_radius_m,target_radius_m,a,b)) or not 0<source_radius_m<=2000000 or not 0<target_radius_m<=2000000:
        raise ValueError("invalid relocation geometry")
    if a<=0 or a+b*source_radius_m<=0 or not math.isclose(a*source_radius_m+.5*b*source_radius_m**2,target_radius_m,rel_tol=1.e-10,abs_tol=1.e-6):
        raise ValueError("radius map must be monotone and reach the stated outer radius")
    _disk_inside(grid,*old_centre,source_radius_m)
    _disk_inside(grid,*new_centre,target_radius_m)
    _disk_inside(coarse_grid,*old_centre,source_radius_m,halo_deg=11.)
    if vector and fields.shape[0] % 2:
        raise ValueError("vector planes must contain equal east and north level counts")
    if old_centre == new_centre and a == 1. and b == 0.:
        return fields.copy()
    P=fields.shape[0];ring=cp.empty((P,angles),cp.float64);mean=cp.empty(P,cp.float64)
    geometry=(np.int32(grid.nx),np.int32(grid.ny),np.float64(grid.latitude0),np.float64(grid.longitude0),
              np.float64(grid.dlatitude),np.float64(grid.dlongitude),np.int32(coarse_grid.nx),np.int32(coarse_grid.ny),
              np.float64(coarse_grid.latitude0),np.float64(coarse_grid.longitude0),np.int32(P),np.int32(angles))
    _module().get_function("tc_ring")((P,), (256,),
       (fields,basic,ring,mean,*geometry,*map(np.float64,(*old_centre,source_radius_m))),shared_mem=256*8)
    out=cp.empty_like(fields)
    count=fields.size//2 if vector else fields.size
    _module().get_function("tc_move_vector" if vector else "tc_move")(((count+255)//256,), (256,),
       (fields,basic,ring,mean,out,*geometry,*map(np.float64,(*old_centre,*new_centre,source_radius_m,a,b,target_radius_m))))
    return out


def relocate_scalars(fields, grid, basic, coarse_grid, **geometry):
    """Relocate scalar isobaric planes; return separate storage."""
    return _relocate(fields, grid, basic, coarse_grid, **geometry)


def relocate_vectors(fields, grid, basic, coarse_grid, **geometry):
    """Relocate Earth-relative wind planes [east levels, north levels].

    The anomaly retains its radial and tangential components in the source and
    destination great-circle bases. This operation does not adjust intensity or
    rebalance pressure and temperature; a native adapter must supply those steps.
    """
    return _relocate(fields, grid, basic, coarse_grid, vector=True, **geometry)


def relocate_constituents(fields, grid, *, old_centre, new_centre,
                         source_radius_m, target_radius_m, map_coefficients, angles=24):
    """Bounded full-field remap with shared weights for all constituent moments.

    Unlike signed dynamical anomalies, a dry environment cannot supply a
    negative condensate. The old disk is filled from its boundary; the new
    disk takes the inverse-mapped donor with a smooth outer blend. This
    initialization remap preserves nonnegativity and proportional moments,
    but does not claim area-integrated mass conservation under resizing.
    """
    import cupy as cp
    import numpy as np
    _field(fields, grid)
    a, b = map_coefficients
    _disk_inside(grid, *old_centre, source_radius_m)
    _disk_inside(grid, *new_centre, target_radius_m)
    if old_centre == new_centre and a == 1. and b == 0.:
        return fields.copy()
    out = cp.empty_like(fields)
    _module().get_function('tc_move_constituents')(
        ((fields.size + 255)//256,), (256,),
        (fields, out, np.int32(grid.nx), np.int32(grid.ny),
         *map(np.float64, (grid.latitude0, grid.longitude0, grid.dlatitude, grid.dlongitude)),
         np.int32(fields.shape[0]), np.int32(angles),
         *map(np.float64, (*old_centre, *new_centre, source_radius_m, a, b, target_radius_m))))
    return out


def gradient_wind_ratio(source_radii, source_tangential, target_radii, target_tangential,
                        *, source_coriolis, target_coriolis):
    """Pressure/thermal ratio from corresponding boundary-layer-top profiles.

    Radial profiles start at zero with zero tangential wind, increase strictly,
    and end where the separated vortex vanishes. Both signed winds must be
    cyclonic in their hemisphere. Arrays are CUDA FP64, length 2..1024.
    An absent original vortex cannot support an amplified balanced vortex.
    The native adapter must diagnose these profiles, not substitute 10-m winds.
    """
    import cupy as cp
    import numpy as np
    arrays=(source_radii,source_tangential,target_radii,target_tangential)
    n=source_radii.size
    if not 2<=n<=1024 or any(not isinstance(v,cp.ndarray) or v.dtype!=cp.float64 or v.shape!=(n,) or not v.flags.c_contiguous for v in arrays):
        raise ValueError("balance profiles require contiguous CUDA float64[2..1024]")
    if not all(math.isfinite(f) and f!=0. for f in (source_coriolis,target_coriolis)) or source_coriolis*target_coriolis<=0:
        raise ValueError("gradient-wind balance requires nonzero Coriolis parameters in the same hemisphere")
    gamma=cp.empty_like(source_radii)
    _module().get_function("tc_balance_ratio")((1,),(256,),
        (*arrays,np.float64(source_coriolis),np.float64(target_coriolis),np.int32(n),gamma),shared_mem=4*n*8)
    if not bool(cp.isfinite(gamma).all()):
        raise ValueError("invalid radial profile or no original vortex to balance")
    return gamma


def thermal_balance(temperature, mixing_ratio, axisymmetric_temperature, gamma):
    """Apply the published horizontal pressure ratio to the thermal anomaly.

    All fields are CUDA FP64; volume shape [level,y,x], gamma [y,x]. The caller
    must also adjust the sea-level pressure anomaly and reconstruct compatible
    geopotential/native column mass before committing an initialized state.
    """
    import cupy as cp
    import numpy as np
    if not isinstance(temperature,cp.ndarray) or temperature.ndim!=3:
        raise ValueError("thermal balance needs CUDA [level,y,x] fields")
    shape=temperature.shape
    for value in (temperature,mixing_ratio,axisymmetric_temperature):
        if not isinstance(value,cp.ndarray) or value.shape!=shape or value.dtype!=cp.float64 or not value.flags.c_contiguous:
            raise ValueError("thermal balance needs matching contiguous CUDA float64 fields")
    if not isinstance(gamma,cp.ndarray) or gamma.shape!=shape[1:] or gamma.dtype!=cp.float64 or not gamma.flags.c_contiguous:
        raise ValueError("thermal ratio must be a contiguous CUDA float64 horizontal field")
    t=cp.empty_like(temperature);q=cp.empty_like(mixing_ratio)
    _module().get_function("tc_thermal_balance")(((t.size+255)//256,),(256,),
        (temperature,mixing_ratio,axisymmetric_temperature,gamma,t,q,np.int32(shape[1]*shape[2]),np.int32(shape[0])))
    if not bool(cp.isfinite(t).all()) or not bool(cp.isfinite(q).all()):
        raise ValueError("thermal balance produced a nonfinite or invalid state")
    return t,q


def axisymmetric_profiles(fields, grid, basic, coarse_grid, radii, *,
                          centre, source_radius_m, angles=128, vector=False):
    """Diagnose the separated vortex's radial profiles on the device.

    Scalars return [plane,radius]. Earth-relative [east,north] wind planes
    return [counterclockwise tangential,outward radial] level groups. The
    radial coordinate starts at zero and ends at the separation boundary.
    Temperature and boundary-layer-top wind profiles must be diagnosed from
    the same member and common pressure levels used in relocation.
    """
    import cupy as cp
    import numpy as np
    _field(fields,grid);_field(basic,coarse_grid)
    if basic.shape[0]!=fields.shape[0] or coarse_grid.dlatitude!=1. or coarse_grid.dlongitude!=1.:
        raise ValueError("profile basic field must match on the one-degree grid")
    if not isinstance(angles,int) or not 8<=angles<=256 or (vector and fields.shape[0]%2):
        raise ValueError("invalid quadrature or wind plane count")
    if not math.isfinite(source_radius_m) or not 0<source_radius_m<=2000000:
        raise ValueError("invalid separation radius")
    if not isinstance(radii,cp.ndarray) or radii.dtype!=cp.float64 or radii.ndim!=1 or not radii.flags.c_contiguous or not 2<=radii.size<=1024:
        raise ValueError("radii must be contiguous CUDA float64[2..1024]")
    if not bool(cp.isfinite(radii).all()) or float(radii[0])!=0. or float(radii[-1])!=source_radius_m or not bool((radii[1:]>radii[:-1]).all()):
        raise ValueError("radii must strictly increase from zero to the separation radius")
    _disk_inside(grid,*centre,source_radius_m)
    _disk_inside(coarse_grid,*centre,source_radius_m,halo_deg=11.)
    P=fields.shape[0];ring=cp.empty((P,angles),cp.float64);mean=cp.empty(P,cp.float64)
    geometry=(np.int32(grid.nx),np.int32(grid.ny),np.float64(grid.latitude0),np.float64(grid.longitude0),
              np.float64(grid.dlatitude),np.float64(grid.dlongitude),np.int32(coarse_grid.nx),np.int32(coarse_grid.ny),
              np.float64(coarse_grid.latitude0),np.float64(coarse_grid.longitude0),np.int32(P),np.int32(angles))
    _module().get_function("tc_ring")((P,),(256,),
        (fields,basic,ring,mean,*geometry,*map(np.float64,(*centre,source_radius_m))),shared_mem=256*8)
    output=cp.empty((P,radii.size),cp.float64)
    _module().get_function("tc_profiles")((radii.size,P//2 if vector else P),(256,),
        (fields,basic,ring,mean,radii,output,*geometry,*map(np.float64,(*centre,source_radius_m)),
         np.int32(radii.size),np.int32(vector)),shared_mem=256*8*(2 if vector else 1))
    if not bool(cp.isfinite(output).all()):
        raise ValueError("vortex profile samples include a missing/nonfinite field")
    return output


def profile_field(radii, profiles, grid, *, centre, target_radius_m,
                  map_coefficients=(1.,0.), outside=0., vector=False):
    """Map separated source profiles onto the destination patch on CUDA.

    Profiles are [plane,radius], sampled at the source radii. Vector groups
    contain counterclockwise tangential and outward radial components. The
    destination output uses Earth-relative east/north components. An anomaly
    uses outside=0; a balance ratio uses outside=1. No extrapolation is allowed.
    """
    import cupy as cp
    import numpy as np
    grid.validate()
    if not isinstance(radii,cp.ndarray) or radii.dtype!=cp.float64 or radii.ndim!=1 or not radii.flags.c_contiguous or not 2<=radii.size<=1024:
        raise ValueError("profile radii need contiguous CUDA FP64[2..1024]")
    if not isinstance(profiles,cp.ndarray) or profiles.dtype!=cp.float64 or profiles.ndim!=2 or not profiles.flags.c_contiguous or profiles.shape[1]!=radii.size or profiles.shape[0]<1:
        raise ValueError("profiles need contiguous CUDA FP64[plane,radius]")
    if vector and (profiles.shape[0]%2 or outside!=0.):
        raise ValueError("wind profiles need two equal groups and zero exterior")
    a,b=map_coefficients
    if not all(math.isfinite(v) for v in (*centre,target_radius_m,a,b,outside)):
        raise ValueError("nonfinite profile geometry")
    if not bool(cp.isfinite(radii).all()) or float(radii[0])!=0. or not bool((radii[1:]>radii[:-1]).all()) or not bool(cp.isfinite(profiles).all()):
        raise ValueError("profile coordinates/values are invalid")
    source=float(radii[-1])
    if a<=0. or a+b*source<=0. or not math.isclose(a*source+.5*b*source**2,target_radius_m,rel_tol=1.e-10,abs_tol=1.e-6):
        raise ValueError("profile map must be monotone and reach the target boundary")
    _disk_inside(grid,*centre,target_radius_m)
    out=cp.empty((profiles.shape[0],grid.ny,grid.nx),cp.float64)
    count=out.size//2 if vector else out.size
    _module().get_function("tc_profile_field")(((count+255)//256,),(256,),
        (radii,profiles,out,np.int32(radii.size),np.int32(profiles.shape[0]),
         np.int32(grid.nx),np.int32(grid.ny),*map(np.float64,
         (grid.latitude0,grid.longitude0,grid.dlatitude,grid.dlongitude,*centre,a,b,target_radius_m,outside)),np.int32(vector)))
    return out


def pressure_gain_limit(moved_pressure, axis_pressure, quadratic, linear, storm_mask,
                        *, minimum_pressure_pa):
    """Largest common gain with balanced MSLP at or above the supplied floor.

    P(g) = moved - axis + axis * (quadratic*g*g + linear*g).
    The feasible interval must include zero. An incompatible retained
    environment/asymmetry is refused rather than altered or clipped.
    """
    import cupy as cp
    import numpy as np
    shape=moved_pressure.shape
    arrays=(moved_pressure,axis_pressure,quadratic,linear)
    if len(shape)!=2 or any(not isinstance(v,cp.ndarray) or v.dtype!=cp.float64 or v.shape!=shape or not v.flags.c_contiguous for v in arrays):
        raise ValueError('pressure gain needs matching contiguous CUDA FP64 maps')
    if not isinstance(storm_mask,cp.ndarray) or storm_mask.shape!=shape or storm_mask.dtype!=cp.uint8 or not storm_mask.flags.c_contiguous:
        raise ValueError('pressure constraint needs a contiguous CUDA uint8 mask')
    if not math.isfinite(minimum_pressure_pa) or minimum_pressure_pa<=0. or not bool(storm_mask.any()):
        raise ValueError('pressure constraint must be positive with a nonempty storm disk')
    upper=cp.empty(shape,cp.float64);C=upper.size
    _module().get_function('tc_pressure_gain_bounds')(((C+255)//256,),(256,),
        (*arrays,storm_mask,upper,np.int32(C),np.float64(minimum_pressure_pa)))
    limit=upper.min()
    if not bool((limit>0.)&~cp.isnan(limit)):
        raise ValueError('pressure target is incompatible with retained environment/asymmetry')
    return limit


def hydrostatic_increment(pressure_levels, old_temperature, old_mixing_ratio,
                          new_temperature, new_mixing_ratio, old_height):
    """Fixed-pressure height increment anchored at the unchanged top height.

    This preserves the original column's hydrostatic discretization residual.
    It does not independently set surface pressure: the enclosing initializer
    must reconcile its pressure anchor before accepting a balanced state.
    """
    import cupy as cp
    import numpy as np
    shape=old_temperature.shape
    if len(shape)!=3 or shape[0]<2 or not isinstance(pressure_levels,cp.ndarray) or pressure_levels.shape!=(shape[0],) or pressure_levels.dtype!=cp.float64 or not pressure_levels.flags.c_contiguous:
        raise ValueError("hydrostatic increment requires pressure levels and CUDA volumes")
    fields=(old_temperature,old_mixing_ratio,new_temperature,new_mixing_ratio,old_height)
    if any(not isinstance(v,cp.ndarray) or v.shape!=shape or v.dtype!=cp.float64 or not v.flags.c_contiguous for v in fields):
        raise ValueError("hydrostatic fields need matching contiguous CUDA FP64 volumes")
    C=shape[1]*shape[2];out=cp.empty_like(old_height)
    _module().get_function("tc_hydrostatic_increment")(((C+255)//256,),(256,),
        (pressure_levels,*fields,out,np.int32(C),np.int32(shape[0])))
    if not bool(cp.isfinite(out).all()):
        raise ValueError("hydrostatic increment received invalid pressure/thermodynamic fields")
    return out


def intensity_gain(wind_10m, axisymmetric_10m, storm_mask, *, target_maximum_m_s):
    """Fit a positive axisymmetric gain to the resolved 10-m maximum.

    Full winds and axisymmetric components are Earth-relative [2,y,x]. The
    environment and asymmetric residual stay unchanged. The selected disk
    must cover the complete storm; it must not contain a second cyclone.
    Refuses a target incompatible with the retained vector residual. This
    returns a CUDA scalar for subsequent profile and balance operations.
    """
    import cupy as cp
    import numpy as np
    if not isinstance(wind_10m,cp.ndarray) or wind_10m.ndim!=3 or wind_10m.shape[0]!=2:
        raise ValueError("intensity fit needs Earth-relative surface winds")
    shape=wind_10m.shape
    if any(not isinstance(v,cp.ndarray) or v.shape!=shape or v.dtype!=cp.float64 or not v.flags.c_contiguous for v in (wind_10m,axisymmetric_10m)):
        raise ValueError("intensity wind arrays must be contiguous CUDA FP64")
    if not isinstance(storm_mask,cp.ndarray) or storm_mask.shape!=shape[1:] or storm_mask.dtype!=cp.uint8 or not storm_mask.flags.c_contiguous:
        raise ValueError("intensity mask must be contiguous CUDA uint8")
    if not math.isfinite(target_maximum_m_s) or target_maximum_m_s<=0. or not bool(storm_mask.any()):
        raise ValueError("intensity target and selected disk must be nonempty")
    C=storm_mask.size;lower=cp.empty(C,cp.float64);upper=cp.empty_like(lower)
    _module().get_function("tc_gain_bounds")(((C+255)//256,),(256,),
        (wind_10m,axisymmetric_10m,storm_mask,lower,upper,np.int32(C),np.float64(target_maximum_m_s)))
    lo=lower.max();hi=upper.min()
    if not bool(cp.isfinite(hi)&(hi>0.)&(hi>=lo)):
        raise ValueError("target wind is incompatible with the retained environment/asymmetry")
    return hi


def pressure_anchor(pressure_levels, old_temperature, old_mixing_ratio,
                    new_temperature, new_mixing_ratio, old_height, *,
                    old_mslp, new_mslp, old_surface_pressure, terrain_height,
                    old_surface_virtual_temperature, new_surface_virtual_temperature):
    """Reconcile hydrostatic increments to a changed sea-level pressure.

    The lowest pressure-level virtual temperature supplies the sea-level
    extrapolation. Original extrapolation/discrete hydrostatic residuals are
    retained; near-surface virtual temperatures adjust the pressure reduction
    at FIXED terrain. The driver must check this approximation against native
    rediagnosed MSLP and spin-up. This does not assert gradient-wind balance.
    """
    import cupy as cp
    import numpy as np
    top=hydrostatic_increment(pressure_levels,old_temperature,old_mixing_ratio,
                              new_temperature,new_mixing_ratio,old_height)
    shape=old_temperature.shape;C=shape[1]*shape[2]
    surfaces=(old_mslp,new_mslp,old_surface_pressure,terrain_height,
              old_surface_virtual_temperature,new_surface_virtual_temperature)
    if any(not isinstance(v,cp.ndarray) or v.shape!=shape[1:] or v.dtype!=cp.float64 or not v.flags.c_contiguous for v in surfaces):
        raise ValueError("pressure anchors require contiguous CUDA FP64 horizontal fields")
    height=cp.empty_like(top);pressure=cp.empty_like(old_surface_pressure)
    _module().get_function("tc_pressure_anchor")(((C+255)//256,),(256,),
        (pressure_levels,old_temperature,old_mixing_ratio,new_temperature,new_mixing_ratio,
         old_height,top,*surfaces,height,pressure,np.int32(C),np.int32(shape[0])))
    if not bool(cp.isfinite(height).all()) or not bool(cp.isfinite(pressure).all()) or not bool(((pressure>10000.)&(pressure<120000.)).all()):
        raise ValueError("pressure anchor produces invalid native initialization fields")
    return height,pressure
