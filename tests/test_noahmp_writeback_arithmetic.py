"""The pinned write-back kernel against the preceding CuPy implementation."""

import numpy as np
import pytest
from conftest import requires_gpu
from woof.core.noahmp_runtime import (NSNOW, _WRITE_BACK_DIRECT,
                                       acsnom_accumulate)

def _reference_write_back(fields, j, i, r, *, nsoil, dt) -> None:
    """``module_sf_noahmpdrv.F:1223-1400`` on the device, for one slab chunk.

    The identical statements as :func:`_write_back_batch` -- same operator
    order, same ``where(b > a, b, a)`` spelling of gfortran's ``MAX``, same
    layer-order energy accumulation -- reading the orchestration's output
    bundle instead of per-column ``SflxResult`` objects, writing the CuPy
    carriers instead of host copies.  The one structural difference is the
    dead-layer guard in the energy integrals: the host loop skips a layer no
    staged column has reached, this evaluates it masked to an exact ``+0.0``,
    and the docstring of :func:`_write_back_batch` already argues why that
    contribution is the identity.
    """
    import cupy as cp
    from woof.core.noahmp_slab_libm import scatter_slab_fields

    at = (j, i)
    zero = np.float32(0.0)
    one = np.float32(1.0)

    scatter_slab_fields(j, i, [(fields[carrier], r[name])
                               for carrier, name in _WRITE_BACK_DIRECT])

    fields["qfx"][at] = (r["ecan"] + r["edir"]) + r["etran"]           # :1206
    fields["lh"][at] = (r["fcev"] + r["fgev"]) + r["fctr"]             # :1207
    fields["smstav"][at] = zero                                        # :1227
    fields["smstot"][at] = zero
    fields["sfcrunoff"][at] = fields["sfcrunoff"][at] + r["runsrf"]
    fields["udrunoff"][at] = fields["udrunoff"][at] + r["runsub"]
    lit = r["albedo"] > np.float32(-999.0)                             # :1232
    fields["albedo"][at] = cp.where(lit, r["albedo"], fields["albedo"][at])

    fields["canwat"][at] = r["canliq"] + r["canice"]                   # :1241
    fields["acsnow"][at] = (fields["acsnow"][at]
                            + fields["rainbl"][at] * r["fpice"])
    ponding = (r["ponding"] + r["ponding1"]) + r["ponding2"]
    # :1243, left to right: ((((ACSNOM + QMELT*DT) + PONDING) + PONDING1)
    # + PONDING2).  Not ACSNOM + (QMELT*DT + PONDINGXY): the WRF column
    # oracle measured that grouping 1 ULP off on snowmelt columns.
    fields["acsnom"][at] = acsnom_accumulate(
        fields["acsnom"][at], r["qmelt"], dt, r["ponding"], r["ponding1"],
        r["ponding2"])
    fields["pondingxy"][at] = ponding

    # :1285-1286, specific humidity back to mixing ratio.
    fields["q2mvxy"][at] = r["q2v"] / (one - r["q2v"])
    fields["q2mbxy"][at] = r["q2b"] / (one - r["q2b"])

    # ---- the column itself -------------------------------------------------
    scatter_slab_fields(j, i, [
        (fields["smois"], r["smc"]), (fields["sh2o"], r["sh2o"]),
        (fields["tslb"], r["stc"][:, NSNOW:]),
        (fields["tsnoxy"], r["stc"][:, :NSNOW]),
        (fields["zsnsoxy"], r["zsnso"]), (fields["snicexy"], r["snice"]),
        (fields["snliqxy"], r["snliq"]), (fields["snow"], r["sneqv"]),
        (fields["snowh"], r["snowh"]), (fields["isnowxy"], r["isnow"]),
    ])

    # ---- :1305-1314, the canopy conductance inverse --------------------------
    laisun = cp.where(zero > r["laisun"], zero, r["laisun"])
    laisha = cp.where(zero > r["laisha"], zero, r["laisha"])
    rb = cp.where(zero > r["rb"], zero, r["rb"])
    closed = ((r["rssun"] <= zero) | (r["rssha"] <= zero)
              | (laisun == zero) | (laisha == zero))
    inverse = ((one / (r["rssun"] + rb)) * laisun
               + (one / (r["rssha"] + rb)) * laisha)
    fields["rs"][at] = cp.where(closed, zero, one / inverse)

    # ---- :1381-1394, the two column-energy integrals -------------------------
    stc = r["stc"]
    zsnso = r["zsnso"]
    hcpct = r["hcpct"]
    isnow = r["isnow"].astype(cp.int32)
    count = int(j.size)
    soil_energy = cp.zeros(count, dtype=cp.float32)
    snow_energy = cp.zeros(count, dtype=cp.float32)
    for k in range(-NSNOW + 1, nsoil + 1):
        slot = k + NSNOW - 1
        live = k >= isnow + 1
        top = k == isnow + 1
        above = zsnso[:, slot - 1] if slot > 0 else zero
        thickness = cp.where(top, -zsnso[:, slot], above - zsnso[:, slot])
        term = ((thickness * hcpct[:, slot])
                * (stc[:, slot] - np.float32(273.16))) * np.float32(0.001)
        contribution = cp.where(live, term, zero)
        if k >= 1:
            soil_energy = soil_energy + contribution
        else:
            snow_energy = snow_energy + contribution
    fields["soilenergy"][at] = soil_energy
    fields["snowenergy"][at] = snow_energy

@requires_gpu
@pytest.mark.parametrize("snow_mm,depth", [(0.0, 0.0), (60.0, 0.30)])
def test_pinned_writeback_matches_previous_ufunc_order(snow_mm, depth, monkeypatch):
    from test_noahmp_runtime import _host_layout_route
    _host_layout_route(monkeypatch)  # the host-classified write-back is what it pins
    import cupy as cp
    from woof.core import noahmp_runtime as runtime
    from woof.core.dycore import step
    from test_noahmp_runtime import _build

    saved = []
    def capture(fields, j, i, r, *, nsoil, dt):
        saved.append((fields, j.copy(), i.copy(),
                      {name: value.copy() for name, value in r.items()}, nsoil, dt))
    monkeypatch.setattr(runtime, "_write_back_slab", capture)
    state, cfg, driver = _build(nx=6, ny=4, snow_mm=snow_mm, snow_depth_m=depth)
    step(state, cfg)
    fields, j, i, values, nsoil, dt = saved[0]
    first = {name: value.copy() for name, value in fields.items()}
    second = {name: value.copy() for name, value in fields.items()}
    _reference_write_back(first, j, i, values, nsoil=nsoil, dt=dt)
    # Invoke the production entry point from the module before monkeypatching.
    from woof.core.noahmp_slab_libm import write_slab_arithmetic
    _reference_write_back(second, j, i, values, nsoil=nsoil, dt=dt)
    # Reset all accumulation carriers before applying the fused operations.
    for name in ("sfcrunoff", "udrunoff", "acsnow", "acsnom", "albedo"):
        second[name][...] = fields[name]
    assert write_slab_arithmetic(second, j, i, values, nsoil=nsoil, dt=dt)
    for name in first:
        a, b = cp.asnumpy(first[name]), cp.asnumpy(second[name])
        np.testing.assert_array_equal(a.reshape(-1).view(np.uint8), b.reshape(-1).view(np.uint8), err_msg=name)
