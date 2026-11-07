"""WRF's vertical advection orders: the vert_order 5 ladder and its wiring.

The kernels (woof/core/kernels/advection.cu zface_half and advect_w's
vertical faces, pd_advection.cu pd_zface_half) carry WRF's ``vert_order``
3 and 5 ladders selected by the trailing ``vorder`` launch argument.  The
CPU half of this file checks the configuration surface and the float64
mirror against a literal transcription of the Fortran loops; the GPU half
checks every kernel against the mirror at order 5 on short and long
columns, that order 3 is the ladder every earlier run took, and that the
configured orders reach the launchers from a RunConfig.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from conftest import requires_gpu

#: The 51 eta levels (50 layers) of the operational 3 km configuration this
#: order closes (NOAA-EMC/HRRR parm/conus/hrrr_wrf.nl ``eta_levels``, with
#: ``hybrid_opt = 2`` and ``etac = 0.2``): strongly stretched, 0.002 thick
#: at the surface and 0.045 to 0.05 thick in the middle troposphere.
OPERATIONAL_ETA_LEVELS = (
    1.0, 0.998, 0.994, 0.987, 0.975, 0.959, 0.939, 0.916, 0.892, 0.865,
    0.835, 0.802, 0.766, 0.727, 0.685, 0.64, 0.592, 0.542, 0.497, 0.4565,
    0.4205, 0.3877, 0.3582, 0.3317, 0.3078, 0.2863, 0.267, 0.2496, 0.2329,
    0.2188, 0.2047, 0.1906, 0.1765, 0.1624, 0.1483, 0.1342, 0.1201, 0.106,
    0.0919, 0.0778, 0.0657, 0.0568, 0.0486, 0.0409, 0.0337, 0.0271, 0.0209,
    0.0151, 0.0097, 0.0047, 0.0)


# ---------------------------------------------------------------------------
# A literal transcription of the Fortran loops (HRRR fork module_advect_em.F,
# NOAA-EMC/HRRR 40ee6058c WRFV3.9; the same text as WRF 4.7.1), kept
# independent of woof.verify.npref so the mirror is checked, not assumed.
# 1-based k as in the Fortran; kts = 1, ktf = nz for a half-level field.
# ---------------------------------------------------------------------------
def _f_flux3(qm2, qm1, q0, qp1, ua):
    return (7.0 * (q0 + qm1) - (qp1 + qm2)) / 12.0 + np.sign(ua) * ((qp1 - qm2) - 3.0 * (q0 - qm1)) / 12.0


def _f_flux5(qm3, qm2, qm1, q0, qp1, qp2, ua):
    return ((37.0 * (q0 + qm1) - 8.0 * (qp1 + qm2) + (qp2 + qm3)) / 60.0
            - np.sign(ua) * ((qp2 - qm3) - 5.0 * (qp1 - qm2) + 10.0 * (q0 - qm1)) / 60.0)


def _fortran_half_level_fluxes(field, rom, fzm, fzp, vert_order):
    """advect_scalar's vertical flux block (F:5106-5147 for 5, F:5176-5204
    for 3): vflux(k) at k = kts+1 .. ktf, 1-based, in the Fortran's own
    assignment order (interior loop first, then the boundary faces)."""
    nz = field.shape[0]
    kts, ktf = 1, nz
    f = lambda k: field[k - 1]                      # field(i,k,j), k 1-based
    vflux = {}
    if vert_order == 5:
        for k in range(kts + 3, ktf - 2 + 1):
            vel = rom[k - 1]
            vflux[k] = vel * _f_flux5(f(k - 3), f(k - 2), f(k - 1), f(k), f(k + 1), f(k + 2), -vel)
        k = kts + 1
        vflux[k] = rom[k - 1] * (fzm[k - 1] * f(k) + fzp[k - 1] * f(k - 1))
        for k in (kts + 2, ktf - 1):
            vel = rom[k - 1]
            vflux[k] = vel * _f_flux3(f(k - 2), f(k - 1), f(k), f(k + 1), -vel)
        k = ktf
        vflux[k] = rom[k - 1] * (fzm[k - 1] * f(k) + fzp[k - 1] * f(k - 1))
    elif vert_order == 3:
        for k in range(kts + 2, ktf - 1 + 1):
            vel = rom[k - 1]
            vflux[k] = vel * _f_flux3(f(k - 2), f(k - 1), f(k), f(k + 1), -vel)
        for k in (kts + 1, ktf):
            vflux[k] = rom[k - 1] * (fzm[k - 1] * f(k) + fzp[k - 1] * f(k - 1))
    else:
        raise ValueError(vert_order)
    out = np.zeros((nz + 1,) + field.shape[1:])
    for k, value in vflux.items():                  # vflux(k) sits on 0-based face k-1
        out[k - 1] = value
    return out


def _fortran_w_faces(w, rom, vert_order):
    """advect_w's vertical flux block (F:6782-6830 for 5, F:6866-6902 for
    3): vflux(k) for k = kts+1 .. ktf+1 on 1-based w levels (w(i,k,j), k =
    1 is the surface), with the 0.25*(rom(k)+rom(k-1))*(w(k)+w(k-1)) faces."""
    nz = w.shape[0] - 1
    kts, ktf = 1, nz
    wf = lambda k: w[k - 1]
    rf = lambda k: rom[k - 1]
    vflux = {}
    if vert_order == 5:
        for k in range(kts + 3, ktf - 1 + 1):
            vel = 0.5 * (rf(k) + rf(k - 1))
            vflux[k] = vel * _f_flux5(wf(k - 3), wf(k - 2), wf(k - 1), wf(k), wf(k + 1), wf(k + 2), -vel)
        k = kts + 1
        vflux[k] = 0.25 * (rf(k) + rf(k - 1)) * (wf(k) + wf(k - 1))
        for k in (kts + 2, ktf):
            vel = 0.5 * (rf(k) + rf(k - 1))
            vflux[k] = vel * _f_flux3(wf(k - 2), wf(k - 1), wf(k), wf(k + 1), -vel)
        k = ktf + 1
        vflux[k] = 0.25 * (rf(k) + rf(k - 1)) * (wf(k) + wf(k - 1))
    else:
        for k in range(kts + 2, ktf + 1):
            vel = 0.5 * (rf(k) + rf(k - 1))
            vflux[k] = vel * _f_flux3(wf(k - 2), wf(k - 1), wf(k), wf(k + 1), -vel)
        for k in (kts + 1, ktf + 1):
            vflux[k] = 0.25 * (rf(k) + rf(k - 1)) * (wf(k) + wf(k - 1))
    out = np.zeros((nz,) + w.shape[1:])
    for k, value in vflux.items():                  # vflux(k) sits on mass face k-2
        out[k - 2] = value
    return out


# ---------------------------------------------------------------------------
# CPU: configuration surface
# ---------------------------------------------------------------------------
def test_fields_default_to_the_ladder_every_earlier_run_took():
    from woof.config import RunConfig
    names = [f.name for f in dataclasses.fields(RunConfig)]
    start = names.index("v_sca_adv_order")
    assert names[start - 1:start + 4] == [
        "upper_wind_limiter_form", "v_sca_adv_order", "v_mom_adv_order",
        "h_mom_adv_order", "ruc_irrigation"]
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=1000.0, dy=1000.0, ztop=8000.0, dt=5.0, run_seconds=0.0)
    assert (cfg.v_sca_adv_order, cfg.v_mom_adv_order, cfg.h_mom_adv_order) == (3, 3, 5)


# The 3.0 transport port carries vertical orders 2-6 and horizontal momentum
# orders 0 and 2-6 (2582c065c99); what is still refused is a value WRF has no
# operator for.
@pytest.mark.parametrize("field,value,needle", [
    ("v_sca_adv_order", 1, "v_sca_adv_order = 1"),
    ("v_sca_adv_order", 7, "integer orders 2 through 6"),
    ("v_mom_adv_order", 7, "v_mom_adv_order = 7"),
    ("h_mom_adv_order", 1, "orders 0, 2, 3, 4, 5 and 6 only"),
    ("v_sca_adv_order", True, "v_sca_adv_order = True"),
])
def test_unported_orders_are_refused_by_name(field, value, needle):
    from woof.config import RunConfig, validate_run_config
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=1000.0, dy=1000.0, ztop=8000.0, dt=5.0, run_seconds=0.0,
                    **{field: value})
    with pytest.raises(ValueError, match=needle):
        validate_run_config(cfg)


def test_order_five_validates_on_every_order_combination():
    from woof.config import RunConfig, validate_run_config
    for vsca, vmom in ((3, 5), (5, 3), (5, 5)):
        cfg = RunConfig(nx=8, ny=8, nz=8, dx=1000.0, dy=1000.0, ztop=8000.0, dt=5.0,
                        run_seconds=0.0, v_sca_adv_order=vsca, v_mom_adv_order=vmom)
        assert validate_run_config(cfg) is cfg


def test_run_profile_selects_the_fork_ladder_without_a_runtime_flag():
    from pathlib import Path
    from woof.experiment import load_experiment
    exp = load_experiment(Path(__file__).resolve().parents[1] / "configs/hrrr_v4_vertical_order5.toml")
    cfg = exp.domains[0].run
    assert (cfg.v_sca_adv_order, cfg.v_mom_adv_order, cfg.h_mom_adv_order) == (5, 5, 5)
    assert cfg.moist_adv_opt == 1 and cfg.zadvect_implicit == 1
    assert cfg.zadvect_implicit_variant == "wrf_legacy"


def test_vertical_orders_helper_reads_the_config_and_tolerates_a_namespace():
    from types import SimpleNamespace
    from woof.core.advection import vertical_orders
    from woof.config import RunConfig
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=1000.0, dy=1000.0, ztop=8000.0, dt=5.0,
                    run_seconds=0.0, v_sca_adv_order=5, v_mom_adv_order=3)
    assert vertical_orders(cfg) == (5, 3)
    assert vertical_orders(SimpleNamespace(dx=1.0)) == (3, 3)


def test_experiment_admits_the_orders_per_domain(tmp_path):
    """The operational HRRR shape: 5 on the root, 3 on a declared nest."""
    from woof.experiment import _DOMAIN_RUN_OVERRIDES
    for key in ("v_sca_adv_order", "v_mom_adv_order", "h_mom_adv_order"):
        assert key in _DOMAIN_RUN_OVERRIDES


def test_restart_echo_drops_the_orders_at_their_defaults_and_keeps_a_moved_one():
    from woof.config import RunConfig
    from woof.io import restart
    base = dict(nx=8, ny=8, nz=8, dx=1000.0, dy=1000.0, ztop=8000.0, dt=5.0, run_seconds=0.0)
    echo = restart.configuration_echo(RunConfig(**base))
    assert not {"v_sca_adv_order", "v_mom_adv_order", "h_mom_adv_order"} & set(echo)
    moved = restart.configuration_echo(RunConfig(**base, v_sca_adv_order=5))
    assert moved["v_sca_adv_order"] == 5 and "v_mom_adv_order" not in moved


def test_prepared_bundles_are_inert_to_the_orders():
    from woof.ingest.prepared_cache import PREPARATION_INERT_RUN_FIELDS
    for key in ("run.v_sca_adv_order", "run.v_mom_adv_order", "run.h_mom_adv_order"):
        assert key in PREPARATION_INERT_RUN_FIELDS


def test_hrrr_route_namelist_carries_a_moved_order_and_keeps_default_bytes():
    """The nested HRRR route runs from the namelist it writes, and its
    round trip compares the prepared identity the orders do not enter.
    The breakage this prevents: a configured order 5 left out of the file
    runs at 3 with nothing said.  At the defaults the rows stay absent,
    so every earlier emission keeps its bytes."""
    import re
    from pathlib import Path
    from woof.experiment import load_experiment
    from woof.hrrr_route_inputs import render_namelist_input

    arm = (Path(__file__).resolve().parents[1] / "configs" / "battery"
           / "shape_3km_thompson_rrtmg_legacy.toml")
    exp = load_experiment(arm)
    text = render_namelist_input(exp)
    assert "v_sca_adv_order" not in text and "v_mom_adv_order" not in text

    def column(text, key):
        match = re.search(rf"^ {key}\s*=\s*(.*)$", text, flags=re.M)
        assert match, key
        return [v.strip() for v in match.group(1).split(",") if v.strip()]

    domains = list(exp.domains)
    moved = dataclasses.replace(exp, domains=type(exp.domains)(
        dataclasses.replace(domain, run=dataclasses.replace(
            domain.run, v_sca_adv_order=5 if at == 0 else 3,
            v_mom_adv_order=5 if at == 0 else 3))
        for at, domain in enumerate(domains)))
    text = render_namelist_input(moved)
    expected = ["5"] + ["3"] * (len(domains) - 1)
    assert column(text, "v_sca_adv_order") == expected
    assert column(text, "v_mom_adv_order") == expected
    dynamics = text.split("&dynamics", 1)[1].split("\n/", 1)[0]
    assert "v_sca_adv_order" in dynamics


# ---------------------------------------------------------------------------
# CPU: the float64 mirror against the Fortran loop transcription
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nz", (4, 5, 6, 7, 8, 12, 49, 50))
@pytest.mark.parametrize("vorder", (3, 5))
def test_mirror_half_level_ladder_matches_the_fortran_loops(nz, vorder):
    from woof.verify.npref import _fz_half_levels
    rng = np.random.default_rng(nz * 10 + vorder)
    q = rng.normal(300.0, 5.0, (nz, 3, 4))
    rom = rng.normal(0.0, 1.0, (nz + 1, 3, 4))
    rom[0] = rom[-1] = 0.0
    fzm = rng.uniform(0.3, 0.7, nz)
    fzp = 1.0 - fzm
    got = _fz_half_levels(q, rom, fzm, fzp, vorder)
    want = _fortran_half_level_fluxes(q, rom, fzm, fzp, vorder)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    # The ladder itself, face by face, on a column long enough for every rung.
    if nz >= 6 and vorder == 5:
        for kf in range(3, nz - 2):
            assert not np.allclose(got[kf], _fortran_half_level_fluxes(q, rom, fzm, fzp, 3)[kf])


@pytest.mark.parametrize("vorder", (3, 5))
def test_mirror_matches_the_fortran_loops_on_the_operational_ladder(vorder):
    """The 2nd-order faces weight by the ladder's own fzm/fzp (WRF fnm/fnp
    of the 51 operational eta levels, far from 0.5/0.5 near the surface),
    and the rungs above them are WRF's uniform-grid stencils on the
    stretched levels, as the Fortran applies them."""
    from woof.core.grid import make_vertical_coord
    from woof.verify.npref import _fz_half_levels
    nz = len(OPERATIONAL_ETA_LEVELS) - 1
    vc = make_vertical_coord(nz, hybrid_opt=2, etac=0.2,
                             eta_levels=np.asarray(OPERATIONAL_ETA_LEVELS))
    fzm = np.asarray(vc.fnm, dtype=np.float64)
    fzp = np.asarray(vc.fnp, dtype=np.float64)
    assert abs(fzm[1] - 0.5) > 0.1                  # the stretched weights, not 0.5/0.5
    rng = np.random.default_rng(510 + vorder)
    q = rng.normal(300.0, 5.0, (nz, 3, 4))
    rom = rng.normal(0.0, 1.0, (nz + 1, 3, 4))
    rom[0] = rom[-1] = 0.0
    got = _fz_half_levels(q, rom, fzm, fzp, vorder)
    want = _fortran_half_level_fluxes(q, rom, fzm, fzp, vorder)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("nz", (3, 4, 5, 6, 7, 8, 12, 49, 50))
@pytest.mark.parametrize("vorder", (3, 5))
def test_mirror_w_faces_match_the_fortran_loops(nz, vorder):
    from woof.verify.npref import _flux3, _flux5v
    rng = np.random.default_rng(nz * 7 + vorder)
    w = rng.normal(0.0, 2.0, (nz + 1, 3, 4))
    rom = rng.normal(0.0, 1.0, (nz + 1, 3, 4))
    rom[0] = rom[-1] = 0.0
    want = _fortran_w_faces(w, rom, vorder)
    # The mirror's face assembly, as np_flux_div_w forms it.
    velz = 0.5 * (rom[:nz] + rom[1:])
    got = np.zeros((nz, 3, 4))
    got[0] = 0.5 * velz[0] * (w[1] + w[0])
    got[nz - 1] = 0.5 * velz[nz - 1] * (w[nz] + w[nz - 1])
    got[1:nz - 1] = _flux3(w[0:nz - 2], w[1:nz - 1], w[2:nz], w[3:nz + 1], velz[1:nz - 1])
    if vorder == 5 and nz >= 5:
        got[2:nz - 2] = _flux5v(w[0:nz - 4], w[1:nz - 3], w[2:nz - 2], w[3:nz - 1], w[4:nz],
                                w[5:nz + 1], velz[2:nz - 2])
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# GPU: kernels against the mirror at order 5, order 3 unchanged, wiring
# ---------------------------------------------------------------------------
def _fields(seed, nz, ny, nx, periodic=False):
    """Random C-grid inputs.  ``periodic`` makes the duplicate u column
    and v row copies of column / row 0 (u, ru, msfu; v, rv, msfv), the
    state a periodic run always holds: the mirror returns column 0 there,
    and an independent random value would test an input no periodic run
    can carry."""
    rng = np.random.default_rng(seed)
    q = rng.normal(300.0, 5.0, (nz, ny, nx)).astype(np.float32)
    u = rng.normal(0.0, 8.0, (nz, ny, nx + 1)).astype(np.float32)
    v = rng.normal(0.0, 8.0, (nz, ny + 1, nx)).astype(np.float32)
    w = rng.normal(0.0, 3.0, (nz + 1, ny, nx)).astype(np.float32)
    ru = rng.normal(0.0, 10.0, (nz, ny, nx + 1)).astype(np.float32)
    rv = rng.normal(0.0, 10.0, (nz, ny + 1, nx)).astype(np.float32)
    rw = rng.normal(0.0, 1.0, (nz + 1, ny, nx)).astype(np.float32)
    rw[0] = rw[-1] = 0.0
    msft = rng.uniform(0.9, 1.1, (ny, nx)).astype(np.float32)
    msfu = rng.uniform(0.9, 1.1, (ny, nx + 1)).astype(np.float32)
    msfv = rng.uniform(0.9, 1.1, (ny + 1, nx)).astype(np.float32)
    if periodic:
        for a in (u, ru):
            a[..., -1] = a[..., 0]
        for a in (v, rv):
            a[:, -1] = a[:, 0]
        msfu[:, -1] = msfu[:, 0]
        msfv[-1] = msfv[0]
    return q, u, v, w, ru, rv, rw, msft, msfu, msfv


def _launch_all(cp, vc, fields, dx, dy, vorder, **bc):
    """Every kernel through its launcher.  ``cp`` is the caller's cupy:
    a helper that imported it itself would mark this whole file ``gpu``
    (tests/conftest.py), and the CPU half would leave the CPU legs."""
    from woof.core.advection import (launch_flux_div_scalar, launch_flux_div_u,
                                      launch_flux_div_v, launch_flux_div_w)
    q, u, v, w, ru, rv, rw, msft, msfu, msfv = (cp.asarray(a) for a in fields)
    out = {}
    for name, launch, field, msf in (("scalar", launch_flux_div_scalar, q, msft),
                                     ("u", launch_flux_div_u, u, msfu),
                                     ("v", launch_flux_div_v, v, msfv),
                                     ("w", launch_flux_div_w, w, msft)):
        tend = cp.zeros_like(field)
        launch(field, ru, rv, rw, tend, vc, dx, dy, msf=msf, vorder=vorder, **bc)
        out[name] = cp.asnumpy(tend)
    return out


def _mirror_all(vc, fields, dx, dy, vorder, **bc):
    from woof.verify.npref import (np_flux_div_scalar, np_flux_div_u,
                                    np_flux_div_v, np_flux_div_w)
    q, u, v, w, ru, rv, rw, msft, msfu, msfv = (a.astype(np.float64) for a in fields)
    return {"scalar": np_flux_div_scalar(q, ru, rv, rw, vc, dx, dy, msf=msft, vorder=vorder, **bc),
            "u": np_flux_div_u(u, ru, rv, rw, vc, dx, dy, msf=msfu, vorder=vorder, **bc),
            "v": np_flux_div_v(v, ru, rv, rw, vc, dx, dy, msf=msfv, vorder=vorder, **bc),
            "w": np_flux_div_w(w, ru, rv, rw, vc, dx, dy, msf=msft, vorder=vorder, **bc)}


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("nz", (4, 5, 6, 7, 8, 12, 49))
@pytest.mark.parametrize("stretch", (None, 1.5))
def test_kernels_match_the_mirror_at_order_five_periodic(nz, stretch):
    import cupy as cp
    from woof.core.grid import make_vertical_coord
    vc = make_vertical_coord(nz, stretch=stretch)
    fields = _fields(nz, nz, 4, 24, periodic=True)
    got = _launch_all(cp, vc, fields, 100.0, 100.0, 5)
    want = _mirror_all(vc, fields, 100.0, 100.0, 5)
    for name in ("scalar", "u", "v", "w"):
        np.testing.assert_allclose(got[name], want[name], rtol=5e-4, atol=5e-4, err_msg=name)
    if nz >= 6:
        three = _mirror_all(vc, fields, 100.0, 100.0, 3)
        for name in ("scalar", "u", "v", "w"):
            assert not np.allclose(got[name], three[name], rtol=5e-4, atol=5e-4), name


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("spec", (False, True))
def test_kernels_match_the_mirror_at_order_five_with_lateral_boundaries(spec):
    """Open and specified x/y take the kernels' boundary paths; the vertical
    ladder is the same on every column (WRF's vertical branches do not
    depend on the lateral boundary type)."""
    import cupy as cp
    from woof.core.grid import make_vertical_coord
    nz = 12
    vc = make_vertical_coord(nz, stretch=1.2)
    fields = _fields(77, nz, 10, 12)
    bc = dict(open_x=True, open_y=True, spec=spec)
    got = _launch_all(cp, vc, fields, 100.0, 100.0, 5, **bc)
    want = _mirror_all(vc, fields, 100.0, 100.0, 5, **bc)
    for name in ("scalar", "u", "v", "w"):
        np.testing.assert_allclose(got[name], want[name], rtol=5e-4, atol=5e-4, err_msg=name)


@pytest.mark.gpu
@requires_gpu
def test_order_three_is_the_default_bitwise():
    """vorder 3 given explicitly is the launch with the argument omitted,
    word for word, on every kernel: the argument is inert at 3."""
    import cupy as cp
    from woof.core.grid import make_vertical_coord
    vc = make_vertical_coord(12, stretch=1.5)
    fields = _fields(5, 12, 4, 24)
    explicit = _launch_all(cp, vc, fields, 100.0, 100.0, 3)
    from woof.core.advection import (launch_flux_div_scalar, launch_flux_div_u,
                                      launch_flux_div_v, launch_flux_div_w)
    q, u, v, w, ru, rv, rw, msft, msfu, msfv = (cp.asarray(a) for a in fields)
    for name, launch, field, msf in (("scalar", launch_flux_div_scalar, q, msft),
                                     ("u", launch_flux_div_u, u, msfu),
                                     ("v", launch_flux_div_v, v, msfv),
                                     ("w", launch_flux_div_w, w, msft)):
        tend = cp.zeros_like(field)
        launch(field, ru, rv, rw, tend, vc, 100.0, 100.0, msf=msf)
        np.testing.assert_array_equal(cp.asnumpy(tend).view(np.uint32),
                                      explicit[name].view(np.uint32), err_msg=name)


@pytest.mark.gpu
@requires_gpu
def test_short_columns_run_the_second_order_faces_as_order_three_does():
    """nz < 4 is where WRF's vert_order 5 loops would read below their own
    memory bounds; the kernels' 2nd-order test comes first, so those
    columns run exactly the order 3 ladder, word for word."""
    import cupy as cp
    from woof.core.grid import make_vertical_coord
    for nz in (2, 3):
        vc = make_vertical_coord(nz)
        fields = _fields(nz, nz, 4, 16)
        five = _launch_all(cp, vc, fields, 100.0, 100.0, 5)
        three = _launch_all(cp, vc, fields, 100.0, 100.0, 3)
        for name in ("scalar", "u", "v", "w"):
            np.testing.assert_array_equal(five[name].view(np.uint32),
                                          three[name].view(np.uint32), err_msg=f"{name} nz={nz}")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("vorder", (3, 5))
def test_a_cloud_water_layer_keeps_its_mass_on_a_periodic_domain(vorder):
    """No boundary, no microphysics, no source or sink: cloud water is a
    passive positive-definite scalar here, so its dry-mass-weighted total
    may move by rounding only.  A sharp layer straddling a rising bubble has
    empty air above and below it, so the final stage meets empty upstream
    cells on both faces every step.  At vertical order 5 the low-order eta
    flux once took the downstream cell at Courant numbers below 1, drained
    those empty cells, and the final clamp turned the drained mass into new
    water (on a 12 h 3 km forecast, 181,000 t of cloud ice and 52.5 t of
    smoke made in transport)."""
    import cupy as cp
    from woof.config import RunConfig
    from woof.core.dycore import run_steps
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced

    nx, ny, nz = 16, 12, 16
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=500.0, dy=500.0, ztop=8000.0,
                    dt=2.0, run_seconds=0.0, moist=True,
                    v_sca_adv_order=vorder)
    vc = make_vertical_coord(nz)
    b = make_base_state(vc, lambda z: 300.0 * np.exp(
        1e-4 * np.asarray(z, float) / 9.81), p_surf=cfg.p_surf, ztop=cfg.ztop)

    def bubble(x, z):
        zz = z[:, None, None] if np.ndim(z) == 1 else z
        L = np.sqrt((x[None, None, :] / 2000.0) ** 2
                    + ((zz - 2000.0) / 1500.0) ** 2)
        return (np.where(L < 1.0, 5.0 * np.cos(np.pi * L / 2) ** 2, 0.0)
                * np.ones((nz, ny, nx)))

    state = init_moist_balanced(
        cfg, vc, b, lambda z: 0.012 * np.exp(-np.asarray(z, float) / 2500.0),
        thp_func=bubble)
    ic = (np.arange(nx) + 0.5) / nx
    jc = (np.arange(ny) + 0.5) / ny

    def fxy(xi, yj):
        return (1.0 + 0.04 * np.sin(2 * np.pi * xi)[None, :]
                * np.cos(2 * np.pi * yj)[:, None])

    state.set_map_coriolis(msft=fxy(ic, jc), msfu=fxy(np.arange(nx + 1) / nx, jc),
                           msfv=fxy(ic, np.arange(ny + 1) / ny))
    layer = np.zeros((nz, ny, nx), dtype=np.float32)
    layer[3:6] = 1.0e-3
    state.qc[...] = cp.asarray(layer)

    def mass():
        c1 = cp.asarray(state.c1h, cp.float64)[:, None, None]
        c2 = cp.asarray(state.c2h, cp.float64)[:, None, None]
        mu = (state.mub2d + state.mup).astype(cp.float64)[None]
        dnw = -cp.asarray(state.dnw, cp.float64)[:, None, None]
        m2 = state.msft.astype(cp.float64)[None] ** 2
        return float(((c1 * mu + c2) * dnw / m2
                      * state.qc.astype(cp.float64)).sum())

    before = mass()
    run_steps(state, cfg, 60)
    after = mass()
    print("periodic cloud water layer, order", vorder, before, after,
          (after - before) / before)
    assert abs(after - before) <= 2e-6 * before, (vorder, before, after)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("vorder", (3, 5))
@pytest.mark.parametrize("sink", (1.0, -1.0), ids=("downward", "upward"))
def test_pd_final_stage_never_drains_an_empty_upstream_cell(vorder, sink):
    """A tracer layer with empty air above and below, in a column of uniform
    eta-mass flux at face Courant numbers below 1.  The cell upstream of the
    layer's leading face is empty, so no flux may leave it: the positive-
    definite final stage must leave every coupled value non-negative, with
    nothing for the final clamp to add.  The order-five low-order eta flux
    once took the downstream (full) cell there, drained the empty one, and
    the clamp manufactured the drained mass (a 12 h 3 km smoke forecast
    gained 52.5 t of 722 t emitted in transport alone)."""
    import cupy as cp
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import launch_pd_fluxes, launch_pd_renorm_apply
    nz, ny, nx = 16, 8, 8
    vc = make_vertical_coord(nz, stretch=1.5, hybrid_opt=2, etac=0.2)
    b = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                        p_surf=1.0e5, ztop=12000.0)
    mub = float(b.mub)
    mut = np.full((ny, nx), mub, np.float32)
    q0 = np.zeros((nz, ny, nx), np.float32)
    q0[6:9] = 40.0
    ru = np.zeros((nz, ny, nx + 1), np.float32)
    rv = np.zeros((nz, ny + 1, nx), np.float32)
    dt = 20.0
    rdnw = np.asarray(vc.rdnw, np.float64)
    c1h = np.asarray(vc.c1h, np.float64)
    c2h = np.asarray(vc.c2h, np.float64)
    rw = np.zeros((nz + 1, ny, nx), np.float32)
    for k in range(1, nz):
        dz = 2.0 / (rdnw[k] + rdnw[k - 1])          # < 0
        mu = c1h[k] * mub + c2h[k]
        rw[k] = sink * 0.3 * abs(dz) * mu / dt      # |Courant| = 0.3
    shapes = ((nz, ny, nx + 1),) * 2 + ((nz, ny + 1, nx),) * 2 + ((nz + 1, ny, nx),) * 2
    bufs = [cp.zeros(sh, cp.float32) for sh in shapes]
    args = [cp.asarray(a) for a in (q0, q0, ru, rv, rw, mut)]
    launch_pd_fluxes(*args, vc, 3000.0, 3000.0, dt, *bufs, vorder=vorder)
    tend = cp.zeros((nz, ny, nx), cp.float32)
    launch_pd_renorm_apply(cp.asarray(q0), cp.asarray(mut), *bufs, tend, vc,
                           3000.0, 3000.0, dt)
    chm = (c1h[:, None, None] * mub + c2h[:, None, None])
    coupled = chm * q0 + dt * cp.asnumpy(tend).astype(np.float64)
    upstream = 9 if sink > 0 else 5                  # empty, feeds the layer
    assert coupled[upstream].min() >= -1e-6 * (chm[upstream].max() * 40.0),         (vorder, sink, float(coupled[upstream].min()))
    assert coupled.min() >= -1e-6 * chm.max() * 40.0, (vorder, sink, float(coupled.min()))
    # Flux form: the column sum moves by rounding only.
    dnw = -1.0 / rdnw
    before = float((chm * q0 * dnw[:, None, None]).sum())
    after = float((coupled * dnw[:, None, None]).sum())
    assert abs(after - before) < 1e-5 * before


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("vorder", (3, 5))
def test_pd_fluxes_match_the_mirror(vorder):
    import cupy as cp
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import launch_pd_fluxes
    from woof.verify.npref import np_pd_fluxes
    nz, ny, nx = 12, 4, 24
    vc = make_vertical_coord(nz, stretch=1.5, hybrid_opt=2, etac=0.2)
    b = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                        p_surf=1.0e5, ztop=12000.0)
    rng = np.random.default_rng(vorder)
    q = rng.normal(0.01, 0.004, (nz, ny, nx)).astype(np.float32)
    q0 = np.clip(rng.normal(0.008, 0.008, (nz, ny, nx)), 0.0, None).astype(np.float32)
    mub = float(b.mub)
    mut = (mub * (1.0 + 0.02 * rng.standard_normal((ny, nx)))).astype(np.float32)
    ru = (mub * rng.normal(0, 6.0, (nz, ny, nx + 1))).astype(np.float32)
    rv = (mub * rng.normal(0, 6.0, (nz, ny + 1, nx))).astype(np.float32)
    rw = (mub * rng.normal(0, 3e-3, (nz + 1, ny, nx))).astype(np.float32)
    rw[0] = rw[-1] = 0.0
    dx = dy = 500.0
    dt = 20.0
    shapes = ((nz, ny, nx + 1),) * 2 + ((nz, ny + 1, nx),) * 2 + ((nz + 1, ny, nx),) * 2
    bufs = [cp.zeros(s, cp.float32) for s in shapes]
    launch_pd_fluxes(cp.asarray(q), cp.asarray(q0), cp.asarray(ru), cp.asarray(rv), cp.asarray(rw),
                     cp.asarray(mut), vc, dx, dy, dt, *bufs, vorder=vorder)
    refs = np_pd_fluxes(q.astype(np.float64), q0.astype(np.float64), ru.astype(np.float64),
                        rv.astype(np.float64), rw.astype(np.float64), mut.astype(np.float64),
                        vc, dx, dy, dt, vorder=vorder)
    scale = float(np.abs(refs[5]).max()) + 1e-30
    for got, ref, name in zip(bufs, refs, ("fxl", "fxc", "fyl", "fyc", "fzl", "fzc")):
        np.testing.assert_allclose(cp.asnumpy(got), ref, rtol=5e-4, atol=5e-4 * scale, err_msg=name)
    if vorder == 5:
        three = np_pd_fluxes(q.astype(np.float64), q0.astype(np.float64), ru.astype(np.float64),
                             rv.astype(np.float64), rw.astype(np.float64), mut.astype(np.float64),
                             vc, dx, dy, dt, vorder=3)
        assert not np.allclose(cp.asnumpy(bufs[5]), three[5], rtol=5e-4, atol=5e-4 * scale)
        # The low-order eta flux keeps order 3's upwind words wherever the
        # face Courant number is at most 1 (the fork's downstream cell there
        # drained empty cells and the final clamp turned that into mass),
        # and takes the semi-Lagrangian upstream sum only above 1.
        low3 = [cp.zeros(sh, cp.float32) for sh in shapes]
        launch_pd_fluxes(cp.asarray(q), cp.asarray(q0), cp.asarray(ru), cp.asarray(rv),
                         cp.asarray(rw), cp.asarray(mut), vc, dx, dy, dt, *low3, vorder=3)
        five_w = cp.asnumpy(bufs[4]).view(np.uint32)
        three_w = cp.asnumpy(low3[4]).view(np.uint32)
        rdnw = np.asarray(vc.rdnw, np.float64)
        kf = np.arange(1, nz)
        dz = (2.0 / (rdnw[kf] + rdnw[kf - 1]))[:, None, None]
        mu = np.asarray(vc.c1h, np.float64)[kf][:, None, None] * mut[None]             + np.asarray(vc.c2h, np.float64)[kf][:, None, None]
        cr = np.zeros((nz + 1, ny, nx))
        cr[1:nz] = rw[1:nz].astype(np.float64) * dt / dz / mu
        multi = np.zeros(cr.shape, bool)
        multi[2:nz - 1] = np.abs(cr[2:nz - 1]) > 1.0
        clear = ~multi & (np.abs(np.abs(cr) - 1.0) > 1e-4)
        assert multi.any()
        np.testing.assert_array_equal(five_w[clear], three_w[clear])
        assert (five_w[multi] != three_w[multi]).any()


@pytest.mark.gpu
@requires_gpu
def test_launchers_refuse_an_unported_order_by_name():
    import cupy as cp
    from woof.core.advection import launch_flux_div_scalar
    from woof.core.grid import make_vertical_coord
    vc = make_vertical_coord(6)
    q, u, v, w, ru, rv, rw, msft, msfu, msfv = (cp.asarray(a) for a in _fields(1, 6, 4, 8))
    with pytest.raises(ValueError, match="vorder must be one of"):
        # Orders 2 through 6 are ported (26548e9345); 7 is no WRF
        # vert_order branch.
        launch_flux_div_scalar(q, ru, rv, rw, cp.zeros_like(q), vc, 100.0, 100.0, vorder=7)


@pytest.mark.gpu
@requires_gpu
def test_dycore_hands_each_kernel_its_configured_order(monkeypatch):
    """Scalars and w take v_sca_adv_order, u and v take v_mom_adv_order
    (WRF advect_w keys on the scalar order), on the explicit path."""
    import cupy as cp
    from woof.core import dycore
    from woof.core import advection as advection_module
    seen = {}

    def spy(name):
        def launch(field, ru, rv, rw, tend, coord, dx, dy, **kw):
            seen[name] = kw.get("vorder", "absent")
        return launch
    for name in ("scalar", "u", "v", "w"):
        monkeypatch.setattr(dycore, f"launch_flux_div_{name}", spy(name))
    monkeypatch.setattr(dycore, "_launch_slow_pgf", lambda *a, **k: None)
    monkeypatch.setattr(dycore, "_launch_slow_buoyancy", lambda *a, **k: None)
    monkeypatch.setattr(dycore, "_validate_geopotential_config", lambda *a, **k: None)
    monkeypatch.setattr(dycore, "_launch_slow_geopotential", lambda *a, **k: None)
    monkeypatch.setattr(dycore, "prepare_moist_cq", lambda *a, **k: object())
    from types import SimpleNamespace
    from woof.config import RunConfig
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=1000.0, dy=1000.0, ztop=8000.0, dt=5.0,
                    run_seconds=0.0, v_sca_adv_order=5, v_mom_adv_order=3)
    state = SimpleNamespace(
        thp=cp.zeros((8, 8, 8), cp.float32), u=cp.zeros((8, 8, 9), cp.float32),
        v=cp.zeros((8, 9, 8), cp.float32), w=cp.zeros((9, 8, 8), cp.float32),
        rth_t=None, ru_t=None, rv_t=None, rw_t=None, msft=None, msfu=None, msfv=None,
        has_msf=False, rotational=False, p=cp.zeros((8, 8, 8), cp.float32),
        total_theta=lambda: cp.zeros((8, 8, 8), cp.float32))
    dycore._add_slow_tendencies(state, cfg, None, None, None, cq=object())
    assert seen == {"scalar": 5, "u": 3, "v": 3, "w": 5}


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("bc", ("periodic", "specified", "open"))
def test_kernels_match_the_mirror_at_order_five_on_the_operational_ladder(bc):
    """Order 5 on the 50 layers the operational configuration runs (hybrid
    coordinate, etac 0.2): every kernel against the float64 mirror, with
    periodic, specified and open lateral boundaries, and not the order 3
    answer."""
    import cupy as cp
    from woof.core.grid import make_vertical_coord
    nz = len(OPERATIONAL_ETA_LEVELS) - 1
    vc = make_vertical_coord(nz, hybrid_opt=2, etac=0.2,
                             eta_levels=np.asarray(OPERATIONAL_ETA_LEVELS))
    if bc == "periodic":
        fields, kw = _fields(501, nz, 4, 24, periodic=True), {}
    else:
        fields = _fields(502, nz, 10, 12)
        kw = dict(open_x=True, open_y=True, spec=(bc == "specified"))
    got = _launch_all(cp, vc, fields, 3000.0, 3000.0, 5, **kw)
    want = _mirror_all(vc, fields, 3000.0, 3000.0, 5, **kw)
    three = _mirror_all(vc, fields, 3000.0, 3000.0, 3, **kw)
    for name in ("scalar", "u", "v", "w"):
        scale = float(np.abs(want[name]).max())
        np.testing.assert_allclose(got[name], want[name], rtol=5e-4, atol=5e-4 * scale, err_msg=name)
        assert not np.allclose(got[name], three[name], rtol=5e-4, atol=5e-4 * scale), name


@pytest.mark.gpu
@requires_gpu
def test_split_path_advects_the_explicit_share_at_the_configured_orders(monkeypatch):
    """Under the implicit-explicit split the four explicit launches take
    the EXPLICIT share of Omega (ctx.wwE) at the configured orders, scalars
    and w at v_sca_adv_order, u and v at v_mom_adv_order; the t0 offset
    flux rejoins on the full Omega at the scalar order (default arithmetic
    only: strict arithmetic balances t0 in its acoustic kernels).  The
    breakage this prevents: the split path left on order 3, or advecting
    the full flux, while the explicit path runs 5."""
    import cupy as cp
    from types import SimpleNamespace
    from woof.core import dycore
    from woof.config import RunConfig
    seen = []

    def spy(name):
        def launch(field, ru, rv, rw, tend, coord, dx, dy, **kw):
            seen.append((name, "explicit" if rw is wwE else "full" if rw is ww else "other",
                         kw.get("vorder", "absent")))
        return launch
    for name in ("scalar", "u", "v", "w"):
        monkeypatch.setattr(dycore, f"launch_flux_div_{name}", spy(name))
    for name in ("solve_u", "solve_v", "solve_theta", "solve_ph", "solve_w"):
        monkeypatch.setattr(dycore.ieva, name, lambda *a, **k: None)
    monkeypatch.setattr(dycore.ieva, "theta_minus_t0", lambda state: state.thp)
    monkeypatch.setattr(dycore.ieva, "add_theta_offset_flux",
                        lambda state, cfg, theta_t, ru, rv, w_, flux: flux(theta_t, ru, rv, w_, None))
    monkeypatch.setattr(dycore, "_launch_slow_pgf", lambda *a, **k: None)
    monkeypatch.setattr(dycore, "_launch_slow_buoyancy", lambda *a, **k: None)
    monkeypatch.setattr(dycore, "_validate_geopotential_config", lambda *a, **k: None)
    monkeypatch.setattr(dycore, "_launch_slow_geopotential", lambda *a, **k: None)
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=1000.0, dy=1000.0, ztop=8000.0, dt=5.0,
                    run_seconds=0.0, v_sca_adv_order=5, v_mom_adv_order=3,
                    zadvect_implicit=1, zadvect_implicit_variant="wrf_legacy")
    wwE = cp.zeros((9, 8, 8), cp.float32)
    ww = cp.zeros((9, 8, 8), cp.float32)
    state = SimpleNamespace(
        thp=cp.zeros((8, 8, 8), cp.float32), u=cp.zeros((8, 8, 9), cp.float32),
        v=cp.zeros((8, 9, 8), cp.float32), w=cp.zeros((9, 8, 8), cp.float32),
        rth_t=None, ru_t=None, rv_t=None, rw_t=None, msft=None, msfu=None, msfv=None,
        has_msf=False, rotational=False, p=cp.zeros((8, 8, 8), cp.float32))
    dycore._add_slow_tendencies_ieva(state, cfg, None, None, ww,
                                     SimpleNamespace(wwE=wwE, wwI=None), cq=object())
    offset = [] if dycore.WRF_EXACT else [("scalar", "full", 5)]
    assert seen == [("scalar", "explicit", 5), ("u", "explicit", 3), ("v", "explicit", 3),
                    ("w", "explicit", 5)] + offset, seen


def test_every_forecast_launch_passes_its_wrf_order():
    """Every flux-divergence and PD launch on the forecast paths names its
    order: scalars, theta, TKE, w and the PD limiter take v_sca_adv_order,
    u and v take v_mom_adv_order (WRF advect_scalar/advect_w/
    advect_scalar_pd read v_sca_adv_order, advect_u/advect_v
    v_mom_adv_order).  That covers the explicit path, the
    implicit-explicit split's explicit share (dycore launches on wwE,
    moist launches on ww_explicit) and the TKE stage.  The breakage this
    prevents: a launch left on the launcher's default 3 runs a field at
    the wrong order under an HRRR configuration with nothing said."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "woof" / "core"
    scalar = {"vsca", "vertical_orders(cfg)[0]"}
    momentum = {"vmom", "vertical_orders(cfg)[1]"}
    expected = {"launch_flux_div_scalar": scalar, "launch_flux_div_w": scalar,
                "launch_pd_fluxes": scalar, "launch_flux_div_u": momentum,
                "launch_flux_div_v": momentum}
    seen = {}
    for name in ("advection.py", "dycore.py", "moist.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if func not in expected:
                continue
            keywords = {k.arg: k.value for k in node.keywords}
            where = f"woof/core/{name}:{node.lineno} {func}"
            assert "vorder" in keywords, f"{where} runs the launcher default"
            assert ast.unparse(keywords["vorder"]) in expected[func], (
                where, ast.unparse(keywords["vorder"]))
            seen[name] = seen.get(name, 0) + 1
    # advection.py's Phase-1 path (4), dycore's explicit and split paths
    # (8), moist's scalar, PD and TKE launches including the selected
    # expanded aerosol/species carry paths (8). Every site is checked above.
    assert seen == {"advection.py": 4, "dycore.py": 8, "moist.py": 8}, seen
