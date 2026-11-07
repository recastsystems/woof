"""Live parent PWP columns for newly exposed moving-nest strips.

Soil interpolation permits an opposite land-class fallback. Ocean columns
must instead have active, non-lake water donors; absent support keeps the
prescribed surface and an inactive column. Full-water bilinear arithmetic
and mixed-corner averaging retain the existing interpolation convention.
"""
import numpy as np
from . import STATE_FIELDS


def parent_columns(fields, child_dc, child_landmask):
    from woof.core.physics_continuation import _array_module
    from woof.core.nest_interp import mask_donor_index
    xp = _array_module(fields['om_tmp'])
    ny, nx = child_landmask.shape
    py, px = fields['xland'].shape
    ratio = int(child_dc.parent_grid_ratio)
    ci, dx = mask_donor_index(nx, ratio, child_dc.i_parent_start)
    cj, dy = mask_donor_index(ny, ratio, child_dc.j_parent_start)
    if ci.min() < 0 or cj.min() < 0 or ci.max()+1 >= px or cj.max()+1 >= py:
        raise ValueError('PWP child interpolation lies outside the parent mass grid')
    ii, jj = xp.asarray(ci)[None, :], xp.asarray(cj)[:, None]
    wx, wy = xp.asarray(dx, dtype=xp.float32)[None, :], xp.asarray(dy, dtype=xp.float32)[:, None]
    wet = ((fields['xland'] > 1.5) & (fields.get('lakemask', 0) < .5)
           & (fields['om_active'] >= 1-1.e-6))
    def corners(a):
        return (a[..., jj, ii], a[..., jj, ii+1], a[..., jj+1, ii], a[..., jj+1, ii+1])
    masks = corners(wet)
    count = sum(m.astype(xp.float32) for m in masks)
    supported = (count > 0) & (xp.asarray(child_landmask) < .5)
    result = {}
    for name in STATE_FIELDS:
        if name == 'om_active':
            result[name] = supported.astype(xp.float32)
            continue
        a, b, c, d = corners(fields[name])
        bilinear = (1-wx)*((1-wy)*a+wy*c)+wx*((1-wy)*b+wy*d)
        total = sum(xp.where(m, v, xp.float32(0)) for m, v in zip(masks, (a,b,c,d)))
        mean = total/xp.maximum(count, xp.float32(1))
        result[name] = xp.ascontiguousarray(xp.where(supported, xp.where(count == 4, bilinear, mean), xp.float32(0)))
    return result
