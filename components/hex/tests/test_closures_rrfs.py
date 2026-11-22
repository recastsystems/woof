# SPDX-License-Identifier: Apache-2.0
"""The surface and boundary-layer closures on the hrrrv5 route (woof.hex.mpasphys.rrfs.closures): the switches, the
files' sums, the conservation properties and the refusals.  No card needed; the torch tests skip without torch."""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pytest

from woof.hex.mpasphys.rrfs import closures
from woof.hex.mpasphys.rrfs.closures import autophysics_pbl as ap

NO_WORDS: dict[str, str] = {}


@dataclass(frozen=True)
class _Config:
    selectors: Mapping[str, str] = field(default_factory=lambda: MappingProxyType(
        {"land_surface": "sf_ruc", "pbl": "bl_mynnedmf", "surface_layer": "sf_mynnsfclay"}))
    backends: Mapping[str, str] = field(default_factory=lambda: MappingProxyType(
        {"land_surface": "hex-cuda", "pbl": "hex-cuda"}))
    closure: Mapping[str, Any] | None = None
    table_key: str = "mpasphys_rrfs"

    def active(self, slot):
        return self.selectors.get(slot, "off") != "off"


# ------------------------------------------------------------------ switches
def test_every_option_off_is_the_plain_config_and_nothing_installs():
    config = _Config()
    assert closures.configure(config, {}, NO_WORDS) is config
    assert closures.configure(config, {"config_neural_pbl": "off", "config_autophysics_pbl": "off"}, NO_WORDS) is config
    assert closures.install(config) is None and closures.ACTIVE is None


@pytest.mark.parametrize("explicit,words,option,name", [
    ({"config_surface_flux_net": "r4_a50"}, {}, "surface_flux_net", "r4_a50"),
    ({"config_surface_flux_net": "on"}, {}, "surface_flux_net", "r4_a50"),
    ({}, {"GPUWM_HEX_SURFACE_FLUX_NET": "r4_a50"}, "surface_flux_net", "r4_a50"),
    ({"config_neural_pbl": "v1"}, {}, "neural_pbl", "v1"),
    ({"config_neural_pbl": "v2"}, {}, "neural_pbl", "v2"),
    ({"config_neural_pbl": "v1", "config_neural_pbl_dry_gate": "true"}, {}, "neural_pbl", "v2"),
    ({}, {"GPUWM_HEX_NEURAL_PBL": "v1", "GPUWM_HEX_NEURAL_PBL_DRY_GATE": "1"}, "neural_pbl", "v2"),
    ({"config_neural_pbl": "v1", "config_neural_pbl_dry_gate": "false"}, {}, "neural_pbl", "v1"),
    ({"config_autophysics_pbl": "v2"}, {}, "autophysics_pbl", "v2"),
    ({"config_autophysics_pbl": "'v3_bowen'"}, {}, "autophysics_pbl", "v3_bowen"),
    ({}, {"GPUWM_HEX_AUTOPHYSICS_PBL": "v2_warm_night"}, "autophysics_pbl", "v2_warm_night"),
    ({"config_autophysics_pbl": "v2"}, {"GPUWM_HEX_AUTOPHYSICS_PBL": "v2"}, "autophysics_pbl", "v2"),
])
def test_each_switch_selects_its_option(explicit, words, option, name, monkeypatch):
    monkeypatch.setattr(closures, "find_spec", lambda name: object())
    config = closures.configure(_Config(), explicit, words)
    assert config.closure["option"] == option and config.closure["set"] == name
    row = closures.load_table()["options"][option]
    assert config.closure["slots"] == row["slots"]
    for fname, sha in config.closure["files"].items():
        assert closures.load_table()["files"][fname] == sha


def test_sets_carry_the_fitted_numbers():
    table = closures.load_table()
    sfn = table["options"]["surface_flux_net"]["sets"]["r4_a50"]
    assert (sfn["scale"], sfn["stride"], sfn["tsk_couple"]) == (0.5, 5, "diag")
    npbl = table["options"]["neural_pbl"]["sets"]
    assert (npbl["v1"]["dry_gate"], npbl["v2"]["dry_gate"], npbl["v1"]["weights"]) == (False, True, npbl["v2"]["weights"])
    path = closures.check_file("autophysics_pbl_sets.v1.json", table["files"]["autophysics_pbl_sets.v1.json"])
    assert ap.load_parameters(path, "v2") == {"params": (0.18, 0.52, 0.10, 0.40, 0.09, -16.0, -0.22, -0.45),
                                              "night_gate": "both"}
    assert ap.load_parameters(path, "v3_bowen")["params"] == (0.42, 0.45, 0.22, 0.35, 0.10, 0.0, -0.25, 0.0)
    warm = ap.load_parameters(path, "v2_warm_night")
    assert warm["night_gate"] == "warm" and warm["params"] == ap.load_parameters(path, "v2")["params"]


# ------------------------------------------------------------------ sums
def test_every_shipped_file_matches_its_sum():
    table = closures.load_table()
    for name, sha in table["files"].items():
        assert closures.check_file(name, sha).is_file()


def test_a_changed_file_is_refused_by_name(tmp_path):
    table = closures.load_table()
    for name in table["files"]:
        shutil.copy(closures._data_dir() / name, tmp_path / name)
    with open(tmp_path / "sfn_r4.pt", "r+b") as handle:
        handle.seek(4000)
        byte = handle.read(1)
        handle.seek(4000)
        handle.write(bytes([byte[0] ^ 1]))
    with pytest.raises(closures.ClosureRefusal, match="sfn_r4.pt has SHA-256"):
        closures.check_file("sfn_r4.pt", table["files"]["sfn_r4.pt"], tmp_path)
    (tmp_path / "neural_pbl_v1.pt").unlink()
    with pytest.raises(closures.ClosureRefusal, match="not in"):
        closures.check_file("neural_pbl_v1.pt", table["files"]["neural_pbl_v1.pt"], tmp_path)


# ------------------------------------------------------------------ refusals
@pytest.mark.parametrize("explicit,words,match", [
    ({"config_surface_flux_net": "r4_a50", "config_neural_pbl": "v1"}, {}, "cannot run together"),
    ({"config_surface_flux_net": "r4_a50"}, {"GPUWM_HEX_AUTOPHYSICS_PBL": "v2"}, "cannot run together"),
    ({"config_neural_pbl": "v2", "config_autophysics_pbl": "v3_bowen"}, {}, "cannot run together"),
    ({"config_neural_pbl_dry_gate": "true"}, {}, "gates the neural boundary layer"),
    ({"config_neural_pbl": "v2", "config_neural_pbl_dry_gate": "false"}, {}, "disagree"),
    ({"config_autophysics_pbl": "v4"}, {}, "not off or one of"),
    ({"config_neural_pbl_dry_gate": "maybe", "config_neural_pbl": "v1"}, {}, "true or false"),
    ({"config_autophysics_pbl": "v2"}, {"GPUWM_HEX_AUTOPHYSICS_PBL": "v3_bowen"}, "two selections"),
    ({"config_unknown_net": "1"}, {}, "unknown closure options"),
])
def test_refusals_name_the_breakage(explicit, words, match):
    with pytest.raises(closures.ClosureRefusal, match=match):
        closures.configure(_Config(), explicit, words)


def test_route_refusals():
    with pytest.raises(closures.ClosureRefusal, match="no MPAS suite route"):
        closures.configure(None, {}, {"GPUWM_HEX_AUTOPHYSICS_PBL": "v2"})
    with pytest.raises(closures.ClosureRefusal, match="this run's route is mpasphys_v842"):
        closures.configure(_Config(table_key="mpasphys_v842"), {"config_neural_pbl": "v1"}, NO_WORDS)
    noah =_Config(selectors=MappingProxyType({"land_surface": "sf_noah", "pbl": "bl_mynnedmf"}))
    with pytest.raises(closures.ClosureRefusal, match="not 'sf_ruc'"):
        closures.configure(noah, {"config_autophysics_pbl": "v2"}, NO_WORDS)
    ysu = _Config(selectors=MappingProxyType({"land_surface": "sf_ruc", "pbl": "bl_ysu"}))
    assert closures.configure(ysu, {"config_autophysics_pbl": "v2"}, NO_WORDS).closure["set"] == "v2"
    with pytest.raises(closures.ClosureRefusal, match="not 'bl_mynnedmf'"):
        closures.configure(ysu, {"config_neural_pbl": "v1"}, NO_WORDS)
    native = _Config(backends=MappingProxyType({"land_surface": "native-cpu", "pbl": "hex-cuda"}))
    with pytest.raises(closures.ClosureRefusal, match="backend is native-cpu"):
        closures.configure(native, {"config_autophysics_pbl": "v2"}, NO_WORDS)


def test_a_net_without_torch_is_refused_and_the_closure_needs_none(monkeypatch):
    monkeypatch.setattr(closures, "find_spec", lambda name: None)
    with pytest.raises(closures.ClosureRefusal, match="no torch"):
        closures.configure(_Config(), {"config_surface_flux_net": "r4_a50"}, NO_WORDS)
    assert closures.configure(_Config(), {"config_autophysics_pbl": "v2"}, NO_WORDS).closure["torch"] is False


def test_a_closure_off_the_resident_slots_is_refused():
    config = closures.configure(_Config(), {"config_autophysics_pbl": "v2"}, NO_WORDS)
    try:
        closures.install(config)
        closures.require_resident("pbl", None)     # the closure has no MYNN hook
        with pytest.raises(closures.ClosureRefusal, match="native library route"):
            closures.require_resident("land", None)
    finally:
        closures.install(None)
    closures.require_resident("land", None)


def test_the_symbolic_closure_never_imports_torch():
    code = ("import sys; import woof.hex.mpasphys.rrfs.closures.autophysics_pbl as m; "
            "assert 'torch' not in sys.modules, 'torch imported'; "
            "assert 'torch' not in open(m.__file__).read().replace('no torch', '').replace('no PyTorch', '')")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_off_imports_neither_torch_nor_cupy():
    code = ("import sys; from woof.hex.mpasphys.rrfs import closures; closures.install(None); "
            "assert closures.configure(None, {}, {}) is None; "
            "assert 'torch' not in sys.modules and 'cupy' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


# ------------------------------------------------------------------ conservation: the symbolic closure
def _columns(n=4000, seed=7, day=True):
    rng = np.random.default_rng(seed)
    f = lambda lo, hi: rng.uniform(lo, hi, n).astype(np.float32)
    return dict(land_mask=(rng.uniform(0, 1, n) > 0.2).astype(np.float32),
                gsw=f(130, 900) if day else f(0, 0.0), hfx=f(-50, 400), lh=f(-20, 500), sat0=f(0.05, 1.0),
                sat_sub=f(0.05, 1.0), t0=f(255, 310), qv0=f(0.001, 0.02), pres0=f(80000, 102000),
                tsk=f(255, 320), u10=f(-12, 12), v10=f(-12, 12))


@pytest.mark.parametrize("name", ["v2", "v3_bowen", "v2_warm_night"])
def test_the_daytime_bowen_law_conserves_the_flux_sum(name):
    table = closures.load_table()
    path = closures.check_file("autophysics_pbl_sets.v1.json", table["files"]["autophysics_pbl_sets.v1.json"])
    p = ap.load_parameters(path, name)
    d_hfx, d_lh, d_qfx, d_tsk = ap.deltas(np, **_columns(), params=p["params"], gain=1.0, night_gate=p["night_gate"])
    assert np.abs(d_hfx).max() > 1.0                 # it acts
    assert np.array_equal(d_hfx + d_lh, np.zeros_like(d_hfx))   # dH + dLH == 0, word for word
    assert np.array_equal(d_tsk, np.zeros_like(d_tsk))           # no nocturnal skin shift in full sun
    assert np.allclose(d_qfx * np.float32(2.501e6), d_lh, rtol=1e-6)


def test_the_warm_night_gate_drops_only_cold_night_cooling():
    table = closures.load_table()
    path = closures.check_file("autophysics_pbl_sets.v1.json", table["files"]["autophysics_pbl_sets.v1.json"])
    v2 = ap.load_parameters(path, "v2")["params"]
    cols = _columns(day=False)
    both = ap.deltas(np, **cols, params=v2, gain=1.0, night_gate="both")
    warm = ap.deltas(np, **cols, params=v2, gain=1.0, night_gate="warm")
    cold = cols["t0"] < 270.0
    hot = cols["t0"] > 303.0
    land = cols["land_mask"] > 0
    assert np.abs(both[0][cold & land]).mean() > 1.0                  # v2 cools freezing nights ...
    assert np.abs(warm[0][cold & land]).max() < 0.05 * np.abs(both[0][cold & land]).max()   # ... the gate does not
    assert np.allclose(warm[0][hot & land], both[0][hot & land], rtol=2e-3, atol=1e-3)     # warm nights unchanged
    assert np.array_equal(warm[1], both[1])                            # the moisture law is not gated


def test_gain_zero_is_the_plain_model():
    d = ap.deltas(np, **_columns(), params=(0.18, 0.52, 0.10, 0.40, 0.09, -16.0, -0.22, -0.45), gain=0.0)
    assert all(not np.any(x) for x in d)


class _Layout:
    def __init__(self, n):
        self.dimensions = {"nCellsSolve": n}
        self.configs = {"config_start_time": "2024-05-21_15:00:00"}


class _Session:
    def __init__(self, n):
        self.layout = _Layout(n)


class _NumpyProvider:
    """The two calls the hook makes on a provider (field, prepared), on host arrays (numpy stands in for cupy)."""

    cp = np

    def __init__(self, n=600, seed=3, day=True):
        rng = np.random.default_rng(seed)
        f = lambda lo, hi, *shape: rng.uniform(lo, hi, shape or (n,)).astype(np.float32)
        self.session = _Session(n)
        self.pools = {
            ("sfc_input", "xland"): np.where(rng.uniform(0, 1, n) > 0.25, 1.0, 2.0).astype(np.float32),
            ("sfc_input", "isltyp"): rng.integers(1, 17, n).astype(np.int32),
            ("sfc_input", "xice"): np.zeros(n, np.float32), ("mesh", "bdyMaskCell"): np.zeros(n, np.int32),
            ("sfc_input", "smois"): f(0.05, 0.45, 9, n), ("sfc_input", "skintemp"): f(270, 315),
            ("diag_physics", "hfx"): f(-40, 350), ("diag_physics", "lh"): f(-10, 450), ("diag_physics", "qfx"): f(0, 2e-4),
            ("diag_physics", "qsfc"): f(0.004, 0.018), ("diag_physics", "gsw"): f(200, 900) if day else f(0, 0.0),
            ("diag_physics", "u10"): f(-8, 8), ("diag_physics", "v10"): f(-8, 8), ("diag_physics", "chs"): f(0.005, 0.05),
            ("diag_physics", "cqs"): f(0.005, 0.05), ("diag", "surface_pressure"): f(85000, 101500)}
        self.prep = {"t_p": f(275, 305, 50, 1, n), "qv_p": f(0.003, 0.016, 50, 1, n),
                     "pres2_hyd_p": f(84000, 101000, 50, 1, n), "rho_p": f(0.9, 1.25, 50, 1, n)}

    def field(self, pool, name):
        return self.pools[pool, name]

    def prepared(self, name):
        return self.prep[name]


def test_the_closure_hook_conserves_by_day_and_never_reaches_the_soil():
    provider = _NumpyProvider(day=True)
    hook = ap.AutoPhysicsPblHook(params=(0.42, 0.45, 0.22, 0.35, 0.10, 0.0, -0.25, 0.0), stride=5)
    tsk0, qsfc0 = (provider.pools["sfc_input", "skintemp"].copy(), provider.pools["diag_physics", "qsfc"].copy())
    hfx0, lh0 = provider.pools["diag_physics", "hfx"].copy(), provider.pools["diag_physics", "lh"].copy()
    for step in range(7):
        hook.before_land_step(provider)
        assert np.array_equal(provider.pools["sfc_input", "skintemp"], tsk0)   # RUC's own skin, every step
        assert np.array_equal(provider.pools["diag_physics", "qsfc"], qsfc0)
        hook.after_land_step(provider, None)
        d_h = provider.pools["diag_physics", "hfx"].astype(np.float64) - hfx0
        d_l = provider.pools["diag_physics", "lh"].astype(np.float64) - lh0
        assert np.abs(d_h).max() > 1.0 and np.abs(d_h + d_l).max() < 1e-3      # float32 rounding of the two adds
        provider.pools["diag_physics", "hfx"][:] = hfx0
        provider.pools["diag_physics", "lh"][:] = lh0
    assert (hook.calls, hook.evals) == (7, 2)                                 # stride 5: steps 1 and 6
    land = (provider.pools["sfc_input", "xland"] < 1.5)
    assert not np.any(provider.pools["sfc_input", "skintemp"][~land] != tsk0[~land])   # water untouched


# ------------------------------------------------------------------ conservation: the neural boundary layer
def test_the_boundary_layer_redistribution_conserves_every_column():
    torch = pytest.importorskip("torch")
    from woof.hex.mpasphys.rrfs.closures.neural_pbl import conservative_tendencies
    g = torch.Generator().manual_seed(11)
    n = 5000
    u = lambda lo, hi, *s: lo + (hi - lo) * torch.rand(*s, generator=g)
    theta = 290.0 + torch.cumsum(u(-0.5, 2.0, n, 10), dim=1)
    qv = u(0.002, 0.018, n, 10)
    uu, vv = u(-15, 15, n, 10), u(-15, 15, n, 10)
    rho, dz = u(0.9, 1.25, n, 10), u(20.0, 250.0, n, 10)
    kzh, kzm = u(0.0, 60.0, n, 10), u(0.0, 60.0, n, 10)
    dk = [u(-0.7, 0.7, n, 9) for _ in range(3)]
    d_th, d_qv, d_u, d_v = conservative_tendencies(*dk, theta, qv, uu, vv, rho, dz, kzh, kzm, 60.0)
    mass = torch.clamp(rho * dz, min=1.0)
    for tend in (d_th, d_qv, d_u, d_v):
        assert tend.abs().max() > 0
        col = (mass * tend).sum(dim=1).abs()
        scale = (mass * tend).abs().sum(dim=1) + 1e-30
        assert float((col / scale).max()) < 1e-5            # zero to float32 rounding, every column
    zero = torch.zeros(n, 9)
    assert all(not t.any() for t in conservative_tendencies(zero, zero, zero, theta, qv, uu, vv, rho, dz, kzh, kzm, 60.0))


def test_the_shipped_nets_load_on_the_host_with_weights_only():
    pytest.importorskip("torch")
    from woof.hex.mpasphys.rrfs.closures import neural_pbl, surface_flux_net
    table = closures.load_table()
    model, ckpt = surface_flux_net.load_checkpoint(closures.check_file("sfn_r4.pt", table["files"]["sfn_r4.pt"]),
                                                   device="cpu")
    assert ckpt["variant"] == "bowen_nete" and not model.training
    model, ckpt = neural_pbl.load_checkpoint(closures.check_file("neural_pbl_v1.pt", table["files"]["neural_pbl_v1.pt"]),
                                             device="cpu")
    assert sum(p.numel() for p in model.parameters()) == 45070 and not model.training
