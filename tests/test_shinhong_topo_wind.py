"""topo_wind under Shin-Hong (bl_pbl_physics = 11), as WRF v4.6.1 runs it.

module_pbl_driver.F hands Shin-Hong CTOPO/CTOPO2 (:1318) exactly as it hands
YSU, and module_bl_shinhong.F reads both: CTOPO scales the surface-drag
diagonal (:1387) and CTOPO2 blends the 10 m wind toward the first-level
wind (:1472-1473).  Until development branch WOOF refused this
combination on the claim that only YSU reads the coefficients.

These are the CPU wiring checks: the run door admits it, and the physics
driver hands the domain's coefficients to the kernel's arm B and writes the
blended 10 m wind back, only when topo_wind is on.  The kernel's words are
graded against WRF's own Shin-Hong on the topo_wind chain oracle
(tests/test_topo_wind_wrf461_chain_parity.py, Shin-Hong consumer).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import TOPO_WIND_PBL_SCHEMES, terrain_drag_refusal


class _Launched(Exception):
    pass


def test_wrf_schemes_that_read_ctopo_are_the_admitted_ones():
    assert TOPO_WIND_PBL_SCHEMES == (1, 11)
    for option in (1, 2):
        assert terrain_drag_refusal(topo_wind=option, gwd_opt=0,
                                    bl_pbl_physics=11) is None
    why = terrain_drag_refusal(topo_wind=1, gwd_opt=0, bl_pbl_physics=5)
    assert "Shin-Hong" in why and "module_bl_shinhong.F:1387" in why


def test_registry_rule_names_both_schemes():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    reg = json.loads((root / "woof" / "physics_registry_v2.json").read_text(
        encoding="utf-8"))
    rules = [rule for option in
             reg["components"]["turbulence"]["options"].values()
             for rule in option.get("constraints", {}).get("refused_when", [])
             if "topo_wind" in rule.get("settings", {})]
    assert rules
    pbl_ids = {opt_id: opt.get("selectors", {}).get("bl_pbl_physics")
               for opt_id, opt in reg["components"]["pbl"]["options"].items()}
    for rule in rules:
        refused = set(rule["components"]["pbl"])
        assert "shinhong" not in refused and "ysu" not in refused
        assert all(pbl_ids[r] not in TOPO_WIND_PBL_SCHEMES for r in refused)
        assert "Shin-Hong" in rule["reason"]


def _driver(monkeypatch, topo_wind):
    import woof.core.physics as physics

    monkeypatch.setattr(physics, "cp", np)
    ny, nx, nz = 2, 3, 4
    shape2 = (ny, nx)
    f = {name: np.full(shape2, value, np.float32) for name, value in {
        "znt": 0.1, "ust": 0.3, "hfx": 10.0, "qfx": 1e-5, "wspd": 5.0,
        "br": 0.1, "fm": 6.0, "fh": 8.0, "xland": 1.0, "u10": 3.0,
        "v10": -1.0}.items()}
    driver = object.__new__(physics.PhysicsDriver)
    driver.fields = f
    driver.state = SimpleNamespace(e_sgs=np.full((nz, ny, nx), 0.1,
                                                 np.float32),
                                   f=np.full(shape2, 1e-4, np.float32),
                                   scratch=lambda shape, name: np.zeros(
                                       shape, np.uint32))
    driver._shinhong_entry_advisory = False
    driver.bldt_seconds = 30.0
    driver.terrain_drag = None if topo_wind is None else SimpleNamespace(
        topo_wind=topo_wind,
        ctopo=np.full(shape2, 2.0, np.float32),
        ctopo2=np.full(shape2, 0.25, np.float32))
    atmosphere = {name: np.ones((nz, ny, nx), np.float32) for name in
                  ("u", "v", "theta", "qv", "qc", "qi", "pressure",
                   "exner", "dz")}
    atmosphere["p_interface"] = np.ones((nz + 1, ny, nx), np.float32)
    cfg = SimpleNamespace(dx=3000.0, dy=3000.0)
    return physics, driver, atmosphere, cfg


@pytest.mark.parametrize("topo_wind", [1, 2])
def test_the_driver_hands_shinhong_the_domain_coefficients(monkeypatch,
                                                           topo_wind):
    physics, driver, atmosphere, cfg = _driver(monkeypatch, topo_wind)
    seen = {}

    def launch(*args, **kwargs):
        seen.update(kwargs)
        kwargs["u10_out"][...] = 7.0
        kwargs["v10_out"][...] = -7.0
        return {}

    def stop(*args, **kwargs):
        raise _Launched

    monkeypatch.setattr(physics, "launch_shinhong", launch)
    monkeypatch.setattr(physics, "invalid_shinhong_outputs", stop)
    with pytest.raises(_Launched):
        driver._run_shinhong(atmosphere, cfg)
    ctopo, ctopo2 = seen["topo"]
    assert ctopo is driver.terrain_drag.ctopo
    assert ctopo2 is driver.terrain_drag.ctopo2
    # WRF's u10/v10 are intent(inout): the blend replaces the surface
    # layer's 10 m wind, which the kernel read and did not write.
    assert seen["u10"] is driver.fields["u10"]
    assert (driver.fields["u10"] == 7.0).all()
    assert (driver.fields["v10"] == -7.0).all()


@pytest.mark.parametrize("drag", [None, 0])
def test_without_topo_wind_the_call_is_arm_a(monkeypatch, drag):
    physics, driver, atmosphere, cfg = _driver(monkeypatch, drag)
    seen = {}

    def launch(*args, **kwargs):
        seen.update(kwargs)
        return {}

    def stop(*args, **kwargs):
        raise _Launched

    monkeypatch.setattr(physics, "launch_shinhong", launch)
    monkeypatch.setattr(physics, "invalid_shinhong_outputs", stop)
    with pytest.raises(_Launched):
        driver._run_shinhong(atmosphere, cfg)
    assert "topo" not in seen and "u10_out" not in seen
    assert (driver.fields["u10"] == 3.0).all()


@pytest.mark.gpu
@requires_gpu
def test_shinhong_tanhf_is_the_reference_platforms_tanhf():
    """csfac's tanh (module_bl_shinhong.F:847), against stored words of the
    reference platform's tanhf (glibc 2.39, test host), the YSU wide oracle's
    sample: every argument of a W1 sweep where that tanhf is not the
    correctly rounded tanh, which this unit computed before
    development branch, plus a spread.  tanhf is odd, so the negated
    arguments take the negated words.  tools/topo_wind_wrf461_oracle/
    sh_tanhf_sweep.py checks every float in [-10, 10] on W1."""
    import cupy as cp

    from pathlib import Path

    from woof.core.kernels import get_kernel

    blob = (Path(__file__).resolve().parent / "data" / "oracles" / "ysu_wide"
            / "tanhf-ref.bin").read_bytes()
    nbad, nspread = np.frombuffer(blob, "<i4", 2)
    n = int(nbad + nspread)
    x = np.frombuffer(blob, "<f4", n, 8).astype(np.float32)
    want = np.frombuffer(blob, "<f4", n, 8 + 4 * n).astype(np.float32)
    assert nbad >= 4000
    x = np.concatenate([x, -x])
    want = np.concatenate([want, -want])
    fn = get_kernel("shinhong", "shinhong_tanhf_probe")
    d = cp.asarray(x)
    o = cp.empty_like(d)
    fn(((x.size + 255) // 256,), (256,), (d, o, np.int64(x.size)))
    np.testing.assert_array_equal(cp.asnumpy(o).view(np.uint32),
                                  want.view(np.uint32))
