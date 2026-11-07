"""grav_settling = 1, 2: the device port against WRF v4.6.1's own Fortran.

The fixture is ``tools/fogdes_wrf461_oracle/build.sh``: WRF's
``phys/module_sf_fogdes.F`` and ``phys/module_bl_fogdes.F``, byte-unmodified
and pinned by sha256, run in the drivers' order (sf_fogdes after the surface
layer, bl_fogdes after the PBL scheme) for six consecutive steps on 62
columns: every FogDES vegetation class and the OTHERS arm under USGS 24/28
and MODIS 20/21, dense radiation fog, thin fog, no cloud water, the
qcgmin = 1e-8 edges, an elevated deck, cloud at the model top, LAI = 0 and a
wind strong enough to reach the half-layer cap.  The build produced the
same bytes three ways: the sources at -O0, the sources at the scalar
referee's flags, and the referee wrf.exe's own objects (libm-report.txt).

Graded per field, per step, in ULP:

* VDFG, DFGDP, FGDP: 0 ULP, every column, every step, FGDP carried on the
  device across the steps.
* RQCBLTEN: 0 ULP at every mass level, including WRF's unchanged top.

A planted one-bit change in a kernel constant must be caught.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import requires_gpu

ORACLE = (Path(__file__).resolve().parents[1] / "woof" / "data" / "fogdes"
          / "oracle")
CASES = ("modis20_gs1", "modis21_gs2", "usgs24_gs1", "usgs24_gs2",
         "usgs28_gs2")
QCGMIN = np.float32(1.0e-8)


def _case(name: str) -> dict[str, np.ndarray]:
    from woof.verify.urban_oracle import load_case
    return load_case(ORACLE / name)


def _ulp(a, b) -> np.ndarray:
    from woof.core.fp32_ulp import fp32_ulp_distance
    return np.asarray(fp32_ulp_distance(np.asarray(a, np.float32),
                                        np.asarray(b, np.float32)))


def run_port(case: dict[str, np.ndarray], *, module=None) -> dict[str, np.ndarray]:
    """The port over every recorded step of one oracle case, on the device.

    Each step starts from the oracle's recorded cloud water and recorded
    PBL tendency (WRF's own arithmetic advanced the column between steps);
    VDFG, DFGDP and FGDP are the port's own, carried on the device from the
    first step.  Returns the oracle's layouts: (ncol, nsteps) and
    (ncol, kte, nsteps).
    """
    import cupy as cp

    from woof.core import fog_settling as fog

    gs = int(case["grav_settling"])
    kte = int(case["kte"])
    dtbl = np.float32(case["dtbl"])
    categories = fog.categories_for_num_land_cat(int(case["nlcat"]))
    ncol, nsteps = case["vdfg"].shape
    f32 = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.float32))
    ivgtyp = cp.asarray(np.ascontiguousarray(case["ivgtyp"], dtype=np.int32))
    lai, wspd = f32(case["lai"]), f32(case["wspd"])
    rho1 = f32(case["rho"][:, 0])
    dz = f32(case["dz8w"][:, :kte].T)
    vdfg = cp.zeros(ncol, dtype=cp.float32)
    dfgdp = cp.zeros(ncol, dtype=cp.float32)
    fgdp = f32(case["fgdp_in"][:, 0])
    out = {"vdfg": np.empty((ncol, nsteps), np.float32),
           "dfgdp": np.empty((ncol, nsteps), np.float32),
           "fgdp": np.empty((ncol, nsteps), np.float32),
           "rqcblten": np.empty((ncol, kte, nsteps), np.float32)}
    for step in range(nsteps):
        qc = f32(case["qc_in"][:, :kte, step].T)
        dqc = f32(case["dqc_in"][:, :kte, step].T)
        dfgdp[...] = 0.0
        fog.surface_deposition(
            grav_settling=gs, qc1=qc[0], rho1=rho1, dz1=dz[0],
            ivgtyp=ivgtyp, lai=lai, wspd=wspd, vdfg=vdfg, fgdp=fgdp,
            dfgdp=dfgdp, dtbl=dtbl, categories=categories, module=module)
        fog.settling_tendency(grav_settling=gs, qc=qc, dz=dz, vdfg=vdfg,
                              dqc=dqc, module=module)
        out["vdfg"][:, step] = cp.asnumpy(vdfg)
        out["dfgdp"][:, step] = cp.asnumpy(dfgdp)
        out["fgdp"][:, step] = cp.asnumpy(fgdp)
        out["rqcblten"][:, :, step] = cp.asnumpy(dqc).T
    return out


def top_interface_flux_active(case) -> np.ndarray:
    """(ncol, nsteps): where WRF's kte-1/2 settling flux is nonzero."""
    kte = int(case["kte"])
    qc = case["qc_in"]
    mean = np.float32(0.5) * (qc[:, kte - 1, :] + qc[:, kte - 2, :])
    return mean > QCGMIN


@requires_gpu
@pytest.mark.parametrize("name", CASES)
def test_surface_fields_match_wrf_bit_for_bit(name):
    case = _case(name)
    port = run_port(case)
    for field in ("vdfg", "dfgdp", "fgdp"):
        ulp = _ulp(port[field], case[field])
        assert int(ulp.max()) == 0, (name, field, int(ulp.max()),
                                     np.argwhere(ulp > 0)[:5].tolist())


@requires_gpu
@pytest.mark.parametrize("name", CASES)
def test_rqcblten_matches_wrf_at_every_mass_level(name):
    case = _case(name)
    kte = int(case["kte"])
    port = run_port(case)["rqcblten"]
    wrf = case["rqcblten"][:, :kte, :]
    ulp = _ulp(port, wrf)
    assert int(ulp.max()) == 0, (name, int(ulp.max()),
                                 np.argwhere(ulp > 0)[:5].tolist())


@requires_gpu
@pytest.mark.parametrize("constant", ["17.03f", "0.0164f", "1.6666666f"])
def test_a_planted_one_bit_change_is_caught(constant):
    """Flip the last bit of one kernel constant; the oracle must see it."""
    import struct

    from woof.core import fog_settling as fog

    source = fog.module_source()
    assert source.count(constant) >= 1, constant
    value = np.float32(float(constant.rstrip("f")))
    bits = struct.unpack("<I", struct.pack("<f", value))[0] ^ 1
    planted = struct.unpack("<f", struct.pack("<I", bits))[0]
    mutated = source.replace(constant, f"{float(planted).hex()}f")
    module = fog.compile_module(mutated)
    caught = 0
    for name in CASES:
        case = _case(name)
        kte = int(case["kte"])
        port = run_port(case, module=module)
        for field in ("vdfg", "dfgdp", "fgdp"):
            caught += int((_ulp(port[field], case[field]) > 0).sum())
        caught += int((_ulp(port["rqcblten"][:, :kte - 1, :],
                            case["rqcblten"][:, :kte - 1, :]) > 0).sum())
    assert caught > 0, constant


@requires_gpu
def test_production_frames():
    import cupy as cp
    from woof.core import fog_settling as fog
    from woof.core.preflight import CHAINED_TRANSLATION_UNIT_FRAMES
    module = fog._module(cp.cuda.Device().id)
    frames = {name: int(cp.cuda.driver.funcGetAttribute(
        cp.cuda.driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        module.get_function(name).ptr))
        for name in ("fogdes_surface", "fogdes_column")}
    assert max(frames.values()) <= CHAINED_TRANSLATION_UNIT_FRAMES[
        "fogdes_composed"].max_local_size_bytes, frames


@requires_gpu
def test_real_physics_driver_runs_both_calls_on_bldt_and_carries_state():
    import cupy as cp
    from test_physics_driver import _full_state
    from woof.core.physics import _prepare_atmosphere
    from woof.io.wrfout import state_frame
    from woof.core import fog_settling as fog

    state, cfg, driver = _full_state(grav_settling=1, mp_physics=6,
                                     dt=30.0, bldt=2.0,
                                     landuse_dataset='MODIFIED_IGBP_MODIS_NOAH')
    state.qc[0:3] = cp.float32(5.0e-4)
    before = _prepare_atmosphere(state)
    driver.compute(state, cfg)
    assert driver.fog_settling is not None
    assert int(cp.count_nonzero(driver.fields['fgdp']).item()) > 0
    expected = cp.zeros_like(state.qc)
    fog.settling_tendency(grav_settling=1, qc=before['qc'], dz=before['dz'],
                          vdfg=driver.fields['vdfg'], dqc=expected)
    assert int(cp.count_nonzero(expected).item()) > 0
    held = driver.fields['fgdp'].copy()
    count = driver.call_counts['sfclay']
    state.elapsed_seconds = 30.0  # ITIMESTEP=2, between the bldt calls.
    driver.compute(state, cfg)
    assert driver.call_counts['sfclay'] == count
    assert cp.asnumpy(held).tobytes() == cp.asnumpy(driver.fields['fgdp']).tobytes()
    state.elapsed_seconds = 90.0  # ITIMESTEP=4, the next fixed-dt bldt call.
    driver.compute(state, cfg)
    assert driver.call_counts['sfclay'] == count + 1
    assert bool(cp.all(driver.fields['fgdp'] > held))
    output = state_frame(state)
    # WRF 4.7.1 Registry.EM_COMMON:1150-1152 gives these internal fields
    # no history flag. Their live state is retained by the driver/restart;
    # the history writer must not invent additional native history rows.
    assert not {'VDFG', 'FGDP', 'DFGDP'} & set(output)
