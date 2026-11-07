"""Landfall decay and resolved TC boundary-layer vortex candidates.

Wu, Liu and Li (2019), doi:10.5194/acp-19-2477-2019, section 3.1.
Use the engine's existing storm-diagnostics w and Earth-relative zeta.
Threshold candidates are not an automatic tornado forecast. The paper's
1--2 km circulation diameter requires examination of the wind field.
"""
import math

VORTEX = dict(max_grid_m=100., min_w_ms=20., min_cyclonic_zeta_s=.2,
              max_height_m=3000., collocation_m=1000.)


def vortex_candidates(w, zeta, *, height_m, dx_m, dy_m, latitude, xp):
    """Return a device mask. Production uses CuPy and cupyx neighbourhoods.

    The two extrema need not occupy the same grid point. Search a circular
    1 km neighbourhood as in the paper's object association criterion.
    Each connected core still needs its circulation diameter reviewed.
    """
    if not all(math.isfinite(x) and x>0 for x in (dx_m,dy_m)):
        raise ValueError("grid spacing must be finite and positive")
    if max(dx_m,dy_m)>VORTEX["max_grid_m"]:
        raise ValueError("unresolved: TC vortex detection requires grid spacing <=100 m")
    if not 0<=height_m<=VORTEX["max_height_m"]:
        raise ValueError("TC boundary-layer vortex diagnostics require 0--3000 m AGL")
    if not math.isfinite(latitude) or latitude==0:
        raise ValueError("a nonzero latitude is required to define cyclonic vorticity")
    if w.shape!=zeta.shape or w.ndim!=2:
        raise ValueError("w and zeta must be matching named-height planes")
    if xp.__name__=="cupy":
        from cupyx.scipy.ndimage import maximum_filter
    else:
        from scipy.ndimage import maximum_filter
    rx=math.ceil(VORTEX["collocation_m"]/dx_m)
    ry=math.ceil(VORTEX["collocation_m"]/dy_m)
    yy,xx=xp.ogrid[-ry:ry+1,-rx:rx+1]
    footprint=(xx*dx_m)**2+(yy*dy_m)**2<=VORTEX["collocation_m"]**2
    near_w=maximum_filter(w,footprint=footprint,mode="constant",cval=-math.inf)
    sign=1 if latitude>0 else -1
    return (xp.isfinite(w)&xp.isfinite(zeta)&
            (near_w>VORTEX["min_w_ms"])&
            (sign*zeta>VORTEX["min_cyclonic_zeta_s"]))


def inland_decay(rows):
    """Observed/model wind ratios after each ocean-to-land centre crossing.

    Rows are time_hours, max_wind_ms and centre_land_fraction, in increasing
    time. This is a diagnostic of actual wind decay, not an imposed decay
    law. A return to sea ends the current inland episode.
    """
    out=[];crossing=None;last=None
    for row in rows:
        t,u,land=(float(row[k]) for k in
                  ("time_hours","max_wind_ms","centre_land_fraction"))
        if not all(math.isfinite(x) for x in (t,u,land)) or u<0 or not 0<=land<=1:
            raise ValueError("invalid landfall diagnostic row")
        if last is not None and t<=last[0]:
            raise ValueError("landfall rows must have increasing times")
        if land<.5:
            crossing=None
        elif crossing is None and last is not None and last[2]<.5:
            # Crossing time is bracketed by frames. Never claim the later
            # frame is the exact physical landfall time.
            crossing=(last[0],t,last[1])
        if crossing is not None:
            lo,hi,initial=crossing
            out.append(dict(**row,landfall_time_bracket_hours=[lo,hi],
                inland_hours_min=t-hi,inland_hours_max=t-lo,
                wind_ratio=u/initial if initial>0 else None))
        last=(t,u,land)
    return out
