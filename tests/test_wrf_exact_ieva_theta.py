"""Strict WRF arithmetic on the implicit-explicit vertical advection path.

Under ``GPUWM_WRF_EXACT=1`` the acoustic step transports WRF's
``t_2 = theta - t0`` (``wrf_advance_mu_theta`` reads ``thp`` with
``thb = t0``), and the non-split big step advects ``thp`` alone.  The split
(IEVA, ``zadvect_implicit = 1``) big step still added the t0 constant's
flux divergence with the full mass flux, the share only the default
full-theta acoustic step balances.  On the stock-WRF door case (g400 crop,
2024-05-21 18Z, 20 s step) that put theta 0.080 K and MU 16 Pa off WRF
after one step and stopped the run at step 9 ("non-finite vertical Courant
number"); with the share dropped the first step reads theta 2.0e-4 K, the
size of WRF's own one-ULP twin (2.1e-4 K).

The breakage this guards: the strict split path transporting a theta its
acoustic step does not, i.e. any flux launch on the full Omega or on a
field other than ``thp``.
"""
from types import SimpleNamespace

import numpy as np
import pytest


def _run_split_path(monkeypatch, strict):
    from woof.core import dycore
    if dycore.cp is None:
        # fix/ds-cpu (2.9.11): without a loadable CuPy the dycore binds no
        # CUDA path (its flux launchers and IEVA are imported beside it).
        pytest.skip("the dycore's CUDA path needs CuPy")
    from woof.config import RunConfig

    seen = []
    wwE = np.zeros((9, 8, 8), np.float32)
    ww = np.zeros((9, 8, 8), np.float32)
    thp = np.zeros((8, 8, 8), np.float32)

    def spy(name):
        def launch(field, ru, rv, rw, tend, coord, dx, dy, **kw):
            seen.append((name,
                         "explicit" if rw is wwE else "full" if rw is ww else "other",
                         "thp" if field is thp else "other"))
        return launch

    for name in ("scalar", "u", "v", "w"):
        monkeypatch.setattr(dycore, f"launch_flux_div_{name}", spy(name))
    for name in ("solve_u", "solve_v", "solve_theta", "solve_ph", "solve_w"):
        monkeypatch.setattr(dycore.ieva, name, lambda *a, **k: None)
    monkeypatch.setattr(dycore.ieva, "theta_minus_t0", lambda state: state.thp)
    for name in ("_launch_slow_pgf", "_launch_slow_buoyancy",
                 "_validate_geopotential_config", "_launch_slow_geopotential"):
        monkeypatch.setattr(dycore, name, lambda *a, **k: None)
    monkeypatch.setattr(dycore, "WRF_EXACT", strict)
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=1000.0, dy=1000.0, ztop=8000.0, dt=5.0,
                    run_seconds=0.0, zadvect_implicit=1)
    state = SimpleNamespace(
        thp=thp, u=np.zeros((8, 8, 9), np.float32),
        v=np.zeros((8, 9, 8), np.float32), w=np.zeros((9, 8, 8), np.float32),
        rth_t=None, ru_t=None, rv_t=None, rw_t=None, msft=None, msfu=None,
        msfv=None, has_msf=False, rotational=False,
        p=np.zeros((8, 8, 8), np.float32))
    dycore._add_slow_tendencies_ieva(state, cfg, None, None, ww,
                                     SimpleNamespace(wwE=wwE, wwI=None),
                                     cq=object())
    return [row for row in seen if row[0] == "scalar"]


def test_strict_split_path_transports_perturbation_theta_only(monkeypatch):
    assert _run_split_path(monkeypatch, strict=True) == [
        ("scalar", "explicit", "thp")]


def test_default_split_path_keeps_the_t0_share_its_acoustic_step_balances(
        monkeypatch):
    rows = _run_split_path(monkeypatch, strict=False)
    assert rows[0] == ("scalar", "explicit", "thp")
    assert [row[1] for row in rows[1:]] == ["full"]
