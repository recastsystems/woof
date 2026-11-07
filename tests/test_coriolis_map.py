# tests/test_coriolis_map.py  (Phase 3 Task 3: Coriolis, curvature, and map
# factors in the dycore: WRF v4.6.1 module_big_step_utilities_em.F
# coriolis/curvature + the map-scale-factor flux forms of ARW tech note
# sec. 2.3-2.4)
import dataclasses
import textwrap
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_gpu

from woof.config import RunConfig, load_config

_DATA = Path(__file__).parent / "data"


# ---- config: map_proj -------------------------------------------------------

def _write_toml(tmp_path, extra=""):
    body = textwrap.dedent(f"""
        [grid]
        nx = 8
        ny = 8
        nz = 8
        dx = 1000.0
        dy = 1000.0
        ztop = 8000.0
        [dynamics]
        {extra}
        [run]
        dt = 2.0
        run_seconds = 0.0
    """)
    p = tmp_path / "cfg.toml"
    p.write_text(body)
    return p


def test_map_proj_default_zero():
    assert RunConfig(nx=8, ny=8, nz=8, dx=1e3, dy=1e3, ztop=8e3, dt=1.0,
                     run_seconds=0.0).map_proj == 0


def test_map_proj_config_validation(tmp_path):
    # 0 (idealized/none) plus the WRF convention 1=lambert, 2=polar
    # stereographic, 3=mercator are the wired projections.
    for code in (1, 2, 3):
        assert load_config(
            _write_toml(tmp_path, f"map_proj = {code}")).map_proj == code
    with pytest.raises(ValueError, match="map_proj"):
        load_config(_write_toml(tmp_path, "map_proj = 4"))


# ---- state: map-factor / Coriolis fields ------------------------------------

@requires_gpu
@pytest.mark.gpu
def test_state_map_coriolis_fields_and_defaults():
    import cupy as cp
    from woof.verify.npref import random_acoustic_state
    s, cfg = random_acoustic_state(seed=1)
    ny, nx = cfg.ny, cfg.nx
    assert s.msft.shape == (ny, nx) and bool((s.msft == 1.0).all())
    assert s.msfu.shape == (ny, nx + 1) and bool((s.msfu == 1.0).all())
    assert s.msfv.shape == (ny + 1, nx) and bool((s.msfv == 1.0).all())
    assert s.f.shape == (ny, nx) and bool((s.f == 0.0).all())
    assert s.e.shape == (ny, nx) and bool((s.e == 0.0).all())
    # WRF's unrotated identity (sina = 0, cosa = 1) is the default frame.
    assert s.sina.shape == (ny, nx) and bool((s.sina == 0.0).all())
    assert s.cosa.shape == (ny, nx) and bool((s.cosa == 1.0).all())
    assert s.rotational is False and s.has_msf is False

    # f-plane: rotation on, but the msf==1 flux forms stay on the
    # bitwise-Phase-2 path.
    s.set_map_coriolis(f=np.full((ny, nx), 1e-4))
    assert s.rotational is True and s.has_msf is False
    assert float(cp.asnumpy(s.f).max()) == np.float32(1e-4)

    # non-uniform map factors flip has_msf as well
    s2, _ = random_acoustic_state(seed=1)
    msft = 1.0 + 0.05 * np.arange(ny * nx).reshape(ny, nx) / (ny * nx)
    s2.set_map_coriolis(msft=msft)
    assert s2.has_msf is True and s2.rotational is True

    # identity values must leave both flags off (bitwise pin depends on it)
    s3, _ = random_acoustic_state(seed=1)
    s3.set_map_coriolis(msft=np.ones((ny, nx)), msfu=np.ones((ny, nx + 1)),
                        msfv=np.ones((ny + 1, nx)), f=np.zeros((ny, nx)),
                        e=np.zeros((ny, nx)), sina=np.zeros((ny, nx)),
                        cosa=np.ones((ny, nx)))
    assert s3.rotational is False and s3.has_msf is False

    # rotated frame: sina/cosa land on device via the sanctioned setter
    # (they scale the e-coriolis terms only, so the flags key off f/e/msf)
    s4, _ = random_acoustic_state(seed=1)
    alpha = 0.2 * np.arange(ny * nx).reshape(ny, nx) / (ny * nx)
    s4.set_map_coriolis(f=np.full((ny, nx), 1e-4),
                        e=np.full((ny, nx), 1.4e-4),
                        sina=np.sin(alpha), cosa=np.cos(alpha))
    assert s4.rotational is True and s4.has_msf is False
    np.testing.assert_allclose(cp.asnumpy(s4.sina),
                               np.sin(alpha).astype(np.float32))
    np.testing.assert_allclose(cp.asnumpy(s4.cosa),
                               np.cos(alpha).astype(np.float32))


# ---- msf==1 / f==0 bitwise regression against the Phase 2 pin ---------------

@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("case", ["dry_flat", "moist", "terrain", "open"])
def test_msf_one_f_zero_bitwise_phase2(case):
    # Captured from the pre-Task-3 tree (commit add26dc) by the one-shot
    # generator over tests/_phase2_pin.py builders: with default map factors
    # (1) and Coriolis parameters (0), N full dycore steps must be BITWISE
    # identical to Phase 2 across the dry/moist/terrain/open paths.
    # The terrain/* entries were recaptured after the PGF term-4 fix (the
    # coefficient is WRF's FULL half-level geopotential, calc_php at
    # module_big_step_utilities_em.F:1261 consumed at :2315-2316/2390-2391
    # and module_small_step_em.F:861-862/935-936); the flat-base entries
    # were verified bitwise unchanged by that fix at recapture time.
    #
    # The moist/* entries were recaptured after the WRF sumflux parity fix
    # (scalars advected with the acoustic time-averaged mass fluxes
    # ru_m/rv_m/ww_m, solve_em.F:2210-2212 / module_small_step_em.F:1473),
    # which legitimately changes moist scalar transport; dry/terrain/open
    # were verified bitwise-unchanged by that fix before saving.
    #
    # Merge note (p4w-pgf x p4w-transport): the shipped npz is the verified
    # union: terrain/* from the PGF-lane capture (sumflux is a no-op for
    # the dry terrain case), moist/* from the transport-lane capture (the
    # PGF fix is bitwise-inert on flat bases), dry_flat/open bitwise-equal
    # on both sides at merge time.  The pin's msf==1 guarantee is
    # unchanged: subsequent msf work must keep this capture bitwise.
    #
    # The moist/* entries were recaptured once more after the W6
    # h_diabatic merge (e72db7d): the microphysics finish now writes theta
    # through the h_diabatic increment ((th_new-t_save) re-added), an
    # FP32 op-order change that legitimately moves the moist case by ULPs
    # even with zero net heating: verified by both W6 reviewers and
    # adjudicated on GPU (u/v/w/thp/php/mup/qv moved; qc/qr and all
    # dry/terrain/open entries stayed bitwise, 38/39 pre-recapture).
    #
    # RECAPTURED TWICE on 2026-09-03 (development branch, commits
    # 962c068da then 6b11e4c99), 25 of 27 entries moving each time -- the
    # two that held are moist/qc and moist/qr, which are identically zero.
    #
    #   962c068da recaptured at the UNPATCHED tip, with the working tree
    #   byte-identical to d9e48213d, because [dry_flat] and [terrain] were
    #   ALREADY RED there: the shipped file differed from a fresh run of
    #   the same builders in 25 of 27 entries (largest, dry_flat/mup,
    #   2.293e-02 on an rms of 1.081e+01).  That drift is NOT root-caused.
    #   The recapture states what the builders produce there.  It is not
    #   a claim that those numbers are right.
    #
    #   6b11e4c99 recaptured after the EOS geopotential fix, which moves
    #   every run's bits by construction (calc_p_alpha stopped forming the
    #   base layer thickness by cancelling two totals).
    #
    # RECAPTURED on 2026-09-16 (the 2.7.6 column verification) by
    # tools/recapture_phase2_pin.py, the generator now kept in the tree, in
    # two readings because the move has two causes:
    #
    #   (a) [dry_flat] and [terrain] were ALREADY RED at the base
    #   c36f4c1f1, the 2.7.5 release tip: with the shipped file restored, a
    #   run there fails u on 2870 of 2880 words (max abs 1.569e-04) and
    #   terrain/u on 2661 of 2720 (1.675e-04), and a fresh capture at that
    #   base differs from the shipped file in 25 of 27 entries (largest
    #   open/v 2.033e-02 on an rms of 1.762e-02, open/mup 1.882e+00 on
    #   9.396e+00, dry_flat/mup 1.669e-02 on 1.081e+01, terrain/mup
    #   3.055e-02 on 5.539e+00).  The dry dynamics moved after the
    #   2026-09-03 recapture without this file following; the release
    #   contract set does not run this GPU test, so 2.7.5 shipped it red.
    #   That drift is NOT root-caused here either, and this capture
    #   states what a run produces rather than that it is right.
    #
    #   (b) The Omega column kernel (calc_ww_cp one column per thread in
    #   WRF's own operation order, replacing CuPy's sum/cumsum) moves every
    #   stepped run's bits by construction.  Its own reading, base capture
    #   against lane capture on the same card, same builders: 25 of 27
    #   entries, largest moist/v 3.313e-04 on an rms of 3.286e-02 (1.0e-2),
    #   open/v 1.420e-04 on 1.758e-02 (8.1e-3), dry_flat/mup 2.494e-02 on
    #   1.081e+01 (2.3e-3), terrain/mup 5.758e-03 on 5.539e+00 (1.0e-3);
    #   moist/qc and moist/qr, identically zero, held.  Sixty steps of the
    #   seeded 64x64x32 benchmark state put u, v, w, thp, php, mup within
    #   1.7e-4 to 7.2e-4 of each field's maximum (docs/manual/
    #   04-gpu-numerics.md section 4.3).  Both readings are committed under
    #   tests/data/receipts/omega-column-scan/ as
    #   phase2-pin-shipped-vs-base.json and phase2-pin-base-vs-lane.json.
    #
    # THE CONSEQUENCE, stated plainly because the sentence below no longer
    # says it: this file describes THIS TIP, not Phase 2.  The dry
    # dynamics have moved under it many times since add26dc.  Its live
    # value is detecting the NEXT change to the msf==1 path; it is no
    # longer evidence of equivalence with the pre-Task-3 code, and it
    # cannot be used to attribute the drift above.
    #
    # NAMED FOLLOW-UP, and the reason this ledger matters: the acoustic
    # solver still spells the cancellation the EOS just shed.
    # woof/core/kernels/acoustic.cu:831-843 and :1064-1076 (advance_w_phi
    # and its msf twin) build the wdwn layer thickness as
    # (phb[k+1]+php[k+1]) - (phb[k]+php[k]) in float32, once per ACOUSTIC
    # SUBSTEP rather than once per RK stage.  Measured 2026-09-03 against
    # a float64 truth on make_base_state columns, max relative error in
    # the thickness:
    #
    #     nz=16  ztop=6400   7.67e-07  ->  6.10e-08  with dphb_resid
    #     nz=64  ztop=2400   1.24e-04  ->  5.15e-06
    #     nz=160 ztop=2400   5.62e-05  ->  1.68e-06
    #     nz=160 ztop=20000  1.46e-05  ->  1.56e-07
    #
    # i.e. 12.6x-93.4x, growing 73x from nz=16 to nz=160 -- the direction
    # every LES configuration moves.  The fix needs NO new memory:
    # dphb_resid is already resident and already carries the (bstr, boff)
    # stride convention advance_w uses.  It was NOT taken here because it
    # moves this pin a THIRD time in one day, and a third recapture on top
    # of an un-root-caused drift would leave nothing able to attribute any
    # of the three.  The pin has an independent grader that does not need
    # recapturing -- tests/test_acoustic.py::test_w_phi_solve_matches_
    # reference, against the genuine float64 np_acoustic_substep mirror
    # (npref.py:2837 differences the geopotential in float64) -- and that
    # is the instrument the follow-up should use.
    #
    # 2026-09-17 (the 2.7.6 verification change): THE CARD, AND THE ATTRIBUTION OF
    # (a) ABOVE.  test host 1's card became an RTX 4090 (compute capability 8.9,
    # driver 610.57.04); every 2026-09-16 reading above was taken on the
    # RTX 5070 Ti (12.0) it replaced, and the two 2026-09-03 recaptures say
    # RTX 3080 (8.6) in their messages.  Measured on the 4090 in one
    # session, with tools/recapture_phase2_pin.py's readings and a scratch
    # copy of its capture checked out at each commit (receipts under
    # tests/data/receipts/pin-gates/):
    #
    #   The pin is CARD-DEPENDENT.  The committed file (5070 Ti bytes) run
    #   at fc639c51f on the 4090: [dry_flat] and [terrain] red, 25 of 27
    #   entries differ, dry_flat/mup 2.274e-02 on an rms of 1.081e+01,
    #   terrain/mup 2.693e-02 on 5.539, dry_flat/u 1.502e-04 on 3.732,
    #   open/v 2.031e-02 on 1.758e-02, open/mup 1.888 on 9.390; moist/qc
    #   and moist/qr held (phase2-pin-4090-vs-committed-fc639c51f.json).
    #   The same builders twice at fc639c51f on the 4090: 0 of 27 differ
    #   (phase2-pin-4090-run1-vs-run2-at-fc639c51f.json).
    #
    #   The 3080 and the 4090 AGREE: the file committed at 6b11e4c99,
    #   captured on the 3080, against a 4090 capture at 6b11e4c99: 0 of 27
    #   entries differ (phase2-pin-3080-vs-4090-at-6b11e4c99.json).
    #
    #   NO COMMIT moved dry_flat or terrain between 6b11e4c99 and the 2.7.5
    #   tip c36f4c1f1 on one card: 4090 captures at 6b11e4c99, a355a7673
    #   (rk_addtend_dry's 1/msf) and fd44111c3 (tiles, derived base-state
    #   arrays) are bit-identical in all 27 entries; 877dc8e97 (scalar
    #   diff6 takes dt/3) moves exactly the seven moist entries (u, v, w,
    #   thp, php, mup, qv; largest moist/mup 2.371e-02 on 1.628e+01) and
    #   nothing dry, which is what its message declares ("the dry rows are
    #   unchanged"), a closed matter; fc20bfc5a and c36f4c1f1 carry
    #   877dc8e97's bytes exactly.  So the drift in (a), "already red at
    #   c36f4c1f1", was the 2026-09-03 file, captured on a 3080, graded on
    #   a 5070 Ti: card, not code.  The Omega kernel's own move on the
    #   4090, c36f4c1f1 against fc639c51f, is 25 of 27 entries, in line
    #   with (b) (phase2-pin-4090-c36f4c1f1-vs-fc639c51f.json).
    #
    #   CONSEQUENCE: this file describes one tip ON ONE CARD.  The 5070 Ti
    #   file is kept as committed, a 4090 file is added beside it with the
    #   first reading above as its reading, tests/_phase2_pin.py PIN_FILES
    #   maps the device name to the file, and a card with no file SKIPS
    #   here with the reason, because a gate that fails for the card and
    #   not the code prevents no breakage.  tools/release/precut_gpu_gate.py
    #   runs this test on the release node's card before a cut and records
    #   the card in its receipt, which is what 2.7.4 and 2.7.5 lacked.
    #
    # RECAPTURED 2026-09-30 (development branch, A146) on the RTX
    # 5070 Ti (12.0, driver 595.91.07, NVRTC 13.4.92), the card the 5070 Ti
    # file describes.  NVRTC had compiled every float division by a
    # compile-time constant as a multiply by the rounded reciprocal on
    # Blackwell, the advection fluxes' / 60 and / 12 among them, and the
    # kernels now spell those divisions __fdiv_rn, the IEEE quotient.  Two
    # readings, both by tools/recapture_phase2_pin.py against the committed
    # file, under tests/data/receipts/a146/: integrate/2.8 10977552d on this
    # card, 0 of 27 entries moved (the file was green there); the fix
    # (lane commit 4370d5b87, exported), 25 of 27, largest dry_flat/mup
    # 2.079e-02 on an rms of 1.081e+01, dry_flat/u 1.743e-04 on 3.732,
    # terrain/mup 2.688e-02 on 5.539, open/v 2.031e-02 on 1.758e-02; moist/qc
    # and moist/qr, identically zero, held.  The 4090 file is unchanged:
    # sm_89 compiles the same division before and after the fix.
    import cupy as cp
    import _phase2_pin as pin
    from woof.config import validate_run_config
    from woof.core.dycore import run_steps
    s, cfg = pin.CASES[case]()
    if cfg.km_opt == 4 and cfg.bl_pbl_physics == 0:
        # These historical captures contain the retired horizontal-only
        # answer.  Admission now succeeds because vertical_diffusion_2 is
        # present.  Exercise both moist-periodic and dry-open full-step
        # paths here; pointwise vertical authority lives in test_smag2d.
        assert validate_run_config(cfg).bl_pbl_physics == 0
        before = {
            f: cp.asnumpy(getattr(s, f)).copy()
            for f in pin.PIN_FIELDS
        }
        run_steps(s, cfg, n=pin.PIN_STEPS)
        for f in pin.PIN_FIELDS:
            got = cp.asnumpy(getattr(s, f))
            assert np.isfinite(got).all(), f
        assert any(
            not np.array_equal(cp.asnumpy(getattr(s, f)), before[f])
            for f in pin.PIN_FIELDS
        )
        return
    reason = pin.pin_skip_reason()
    if reason is not None:
        pytest.skip(reason)
    ref = np.load(pin.pin_path())
    if case == "dry_flat":
        # exercising the setter with identity values must stay bitwise too
        s.set_map_coriolis(msft=np.ones((cfg.ny, cfg.nx)),
                           f=np.zeros((cfg.ny, cfg.nx)))
    run_steps(s, cfg, n=pin.PIN_STEPS)
    fields = pin.PIN_FIELDS + (("qv", "qc", "qr") if s.qv is not None else ())
    for f in fields:
        got = cp.asnumpy(getattr(s, f))
        np.testing.assert_array_equal(got, ref[f"{case}/{f}"], err_msg=f)


# ---- Coriolis + curvature kernel vs float64 mirror --------------------------

def _random_rotational_inputs(seed, nz=8, ny=6, nx=10, rotated=False):
    rng = np.random.default_rng(seed)
    d = {
        "u": rng.normal(0, 10, (nz, ny, nx + 1)),
        "v": rng.normal(0, 10, (nz, ny + 1, nx)),
        "w": rng.normal(0, 1, (nz + 1, ny, nx)),
        "ru": rng.normal(0, 1e5, (nz, ny, nx + 1)),
        "rv": rng.normal(0, 1e5, (nz, ny + 1, nx)),
        "mut": 1e5 + rng.normal(0, 1e3, (ny, nx)),
        "msft": 1.0 + 0.1 * rng.random((ny, nx)),
        "msfu": 1.0 + 0.1 * rng.random((ny, nx + 1)),
        "msfv": 1.0 + 0.1 * rng.random((ny + 1, nx)),
        "f": 1e-4 * (1.0 + 0.2 * rng.random((ny, nx))),
        "e": 1e-4 * (1.0 + 0.2 * rng.random((ny, nx))),
    }
    d["u"][..., -1] = d["u"][..., 0]
    d["v"][:, -1, :] = d["v"][:, 0, :]
    d["w"][0] = d["w"][-1] = 0.0
    if rotated:
        # Local map-rotation angle field (geo_em SINALPHA/COSALPHA
        # conventions); |alpha| up to 0.28 rad matches the real74 d01
        # Lambert domain (sina range +/-0.264).
        alpha = rng.uniform(-0.28, 0.28, (ny, nx))
        d["sina"] = np.sin(alpha)
        d["cosa"] = np.cos(alpha)
    return d


def _wrf_coriolis_curvature_fp64(d, coord, dx, dy):
    """Independent FP64 transcription of WRF v4.6.1
    module_big_step_utilities_em.F SUBROUTINE coriolis (:3726-3729 u,
    :3800-3803 v, :3839-3844 w, including every sina/cosa rotation term)
    plus SUBROUTINE curvature 'normal code' (:4282-4283 vxgm, :4384-4387 u,
    :4437-4440 v, :4459-4463 w), written as explicit Fortran-shaped loops
    so it shares no code with the kernel or the npref mirror.  Isotropic
    map factors (msfux==msfuy etc.), periodic wrap on the duplicate faces.
    """
    from woof.core import constants as c
    nz, ny, nxp1 = d["ru"].shape
    nx = nxp1 - 1
    rdx, rdy = 1.0 / dx, 1.0 / dy
    ru, rv, u, v, w = (np.asarray(d[n], dtype=np.float64)
                       for n in ("ru", "rv", "u", "v", "w"))
    mut, msft, msfu, msfv, f, e = (np.asarray(d[n], dtype=np.float64)
                                   for n in ("mut", "msft", "msfu", "msfv",
                                             "f", "e"))
    sina = np.asarray(d.get("sina", np.zeros((ny, nx))), dtype=np.float64)
    cosa = np.asarray(d.get("cosa", np.ones((ny, nx))), dtype=np.float64)
    c1f = np.asarray(coord.c1f, dtype=np.float64)
    c2f = np.asarray(coord.c2f, dtype=np.float64)
    fnm = np.asarray(coord.fnm, dtype=np.float64)
    fnp = np.asarray(coord.fnp, dtype=np.float64)

    # couple_momentum's rw = (c1f*mut + c2f)*w/msfty.
    rw = (c1f[:, None, None] * mut[None] + c2f[:, None, None]) * w / msft[None]
    # curvature :4282-4283 vxgm at mass points.
    vxgm = np.empty((nz, ny, nx))
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                vxgm[k, j, i] = (
                    0.5 * (u[k, j, i] + u[k, j, i + 1])
                    * (msfv[j + 1, i] - msfv[j, i]) * rdy
                    - 0.5 * (v[k, j, i] + v[k, j + 1, i])
                    * (msfu[j, i + 1] - msfu[j, i]) * rdx)

    dru = np.zeros((nz, ny, nx + 1))
    drv = np.zeros((nz, ny + 1, nx))
    drw = np.zeros((nz + 1, ny, nx))
    for k in range(nz):
        for j in range(ny):
            for i in range(nx + 1):        # u faces; periodic wrap
                ic, im = i % nx, (i - 1) % nx
                rv4 = 0.25 * (rv[k, j, im] + rv[k, j, ic]
                              + rv[k, j + 1, im] + rv[k, j + 1, ic])
                rw4 = 0.25 * (rw[k, j, im] + rw[k + 1, j, im]
                              + rw[k, j, ic] + rw[k + 1, j, ic])
                # coriolis :3726-3729
                dru[k, j, i] = (0.5 * (f[j, ic] + f[j, im]) * rv4
                                - 0.5 * (e[j, ic] + e[j, im])
                                * 0.5 * (cosa[j, ic] + cosa[j, im]) * rw4)
                # curvature :4384-4387
                dru[k, j, i] += (0.5 * (vxgm[k, j, ic] + vxgm[k, j, im]) * rv4
                                 - u[k, j, i] * c.RERADIUS * rw4)
        for j in range(ny + 1):            # v rows; periodic wrap
            jc, jm = j % ny, (j - 1) % ny
            for i in range(nx):
                ru4 = 0.25 * (ru[k, jc, i] + ru[k, jc, i + 1]
                              + ru[k, jm, i] + ru[k, jm, i + 1])
                rw4 = 0.25 * (rw[k, jm, i] + rw[k + 1, jm, i]
                              + rw[k, jc, i] + rw[k + 1, jc, i])
                # coriolis :3800-3803 (msfvy/msfvx == 1)
                drv[k, j, i] = (-0.5 * (f[jc, i] + f[jm, i]) * ru4
                                + 0.5 * (e[jc, i] + e[jm, i])
                                * 0.5 * (sina[jc, i] + sina[jm, i]) * rw4)
                # curvature :4437-4440
                drv[k, j, i] += (-0.5 * (vxgm[k, jc, i] + vxgm[k, jm, i]) * ru4
                                 - v[k, j, i] * c.RERADIUS * rw4)
    for k in range(1, nz):                 # interior w levels
        for j in range(ny):
            for i in range(nx):
                ruf = 0.5 * (fnm[k] * (ru[k, j, i] + ru[k, j, i + 1])
                             + fnp[k] * (ru[k - 1, j, i] + ru[k - 1, j, i + 1]))
                uf = 0.5 * (fnm[k] * (u[k, j, i] + u[k, j, i + 1])
                            + fnp[k] * (u[k - 1, j, i] + u[k - 1, j, i + 1]))
                rvf = 0.5 * (fnm[k] * (rv[k, j, i] + rv[k, j + 1, i])
                             + fnp[k] * (rv[k - 1, j, i] + rv[k - 1, j + 1, i]))
                vf = 0.5 * (fnm[k] * (v[k, j, i] + v[k, j + 1, i])
                            + fnp[k] * (v[k - 1, j, i] + v[k - 1, j + 1, i]))
                # coriolis :3839-3844 (msftx/msfty == 1)
                drw[k, j, i] = e[j, i] * (cosa[j, i] * ruf
                                          - sina[j, i] * rvf)
                # curvature :4459-4463
                drw[k, j, i] += c.RERADIUS * (ruf * uf + rvf * vf)
    return dru, drv, drw


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("rotated", [False, True])
def test_coriolis_curvature_kernel_vs_mirror(rotated):
    import cupy as cp
    from woof.core.dycore import launch_coriolis_curvature
    from woof.core.grid import make_vertical_coord
    from woof.verify.npref import np_coriolis_curvature
    d = _random_rotational_inputs(seed=42, rotated=rotated)
    nz, ny, nxp1 = d["u"].shape
    nx = nxp1 - 1
    coord = make_vertical_coord(nz, stretch=2.0)

    ru_t = cp.zeros((nz, ny, nx + 1), cp.float32)
    rv_t = cp.zeros((nz, ny + 1, nx), cp.float32)
    rw_t = cp.zeros((nz + 1, ny, nx), cp.float32)
    dev = {k: cp.asarray(v, cp.float32) for k, v in d.items()}
    rot = ({"sina": dev["sina"], "cosa": dev["cosa"]} if rotated else {})
    launch_coriolis_curvature(
        dev["ru"], dev["rv"], dev["u"], dev["v"], dev["w"], dev["mut"],
        dev["msft"], dev["msfu"], dev["msfv"], dev["f"], dev["e"],
        cp.asarray(coord.c1f, cp.float32), cp.asarray(coord.c2f, cp.float32),
        cp.asarray(coord.fnm, cp.float32), cp.asarray(coord.fnp, cp.float32),
        200.0, 250.0, ru_t, rv_t, rw_t, **rot)
    rot = ({"sina": d["sina"], "cosa": d["cosa"]} if rotated else {})
    dru, drv, drw = np_coriolis_curvature(
        d["ru"], d["rv"], d["u"], d["v"], d["w"], d["mut"],
        d["msft"], d["msfu"], d["msfv"], d["f"], d["e"], coord, 200.0, 250.0,
        **rot)
    np.testing.assert_allclose(cp.asnumpy(ru_t), dru, rtol=2e-4, atol=2e-3)
    np.testing.assert_allclose(cp.asnumpy(rv_t), drv, rtol=2e-4, atol=2e-3)
    np.testing.assert_allclose(cp.asnumpy(rw_t), drw, rtol=2e-4, atol=2e-3)
    # periodic duplicates stay consistent
    np.testing.assert_array_equal(cp.asnumpy(ru_t)[:, :, -1],
                                  cp.asnumpy(ru_t)[:, :, 0])
    np.testing.assert_array_equal(cp.asnumpy(rv_t)[:, -1, :],
                                  cp.asnumpy(rv_t)[:, 0, :])


@requires_gpu
@pytest.mark.gpu
def test_coriolis_rotated_grid_wrf_fortran_loops():
    # Rotated Lambert grid (nonzero sina): kernel AND npref mirror against
    # the independent FP64 Fortran-loop transcription of WRF coriolis
    # (module_big_step_utilities_em.F:3726-3729, :3800-3803, :3839-3844)
    # + curvature (:4282-4283, :4384-4387, :4437-4440, :4459-4463).  The
    # mirror comparison is the shared-reduction blindness guard: both sides
    # are checked against loops transcribed straight from the Fortran.
    # FAILED before the sina/cosa fix (v-equation missing +e*sina*<rw>,
    # w-equation missing e*(cosa-1)*<ru> - e*sina*<rv>, u-equation e-term
    # missing the cosa factor).
    import cupy as cp
    from woof.core.dycore import launch_coriolis_curvature
    from woof.core.grid import make_vertical_coord
    from woof.verify.npref import np_coriolis_curvature
    d = _random_rotational_inputs(seed=1974, rotated=True)
    nz, ny, nxp1 = d["u"].shape
    nx = nxp1 - 1
    coord = make_vertical_coord(nz, stretch=2.0)
    dru, drv, drw = _wrf_coriolis_curvature_fp64(d, coord, 200.0, 250.0)

    ru_t = cp.zeros((nz, ny, nx + 1), cp.float32)
    rv_t = cp.zeros((nz, ny + 1, nx), cp.float32)
    rw_t = cp.zeros((nz + 1, ny, nx), cp.float32)
    dev = {k: cp.asarray(v, cp.float32) for k, v in d.items()}
    launch_coriolis_curvature(
        dev["ru"], dev["rv"], dev["u"], dev["v"], dev["w"], dev["mut"],
        dev["msft"], dev["msfu"], dev["msfv"], dev["f"], dev["e"],
        cp.asarray(coord.c1f, cp.float32), cp.asarray(coord.c2f, cp.float32),
        cp.asarray(coord.fnm, cp.float32), cp.asarray(coord.fnp, cp.float32),
        200.0, 250.0, ru_t, rv_t, rw_t,
        sina=dev["sina"], cosa=dev["cosa"])
    np.testing.assert_allclose(cp.asnumpy(ru_t), dru, rtol=2e-4, atol=2e-3)
    np.testing.assert_allclose(cp.asnumpy(rv_t), drv, rtol=2e-4, atol=2e-3)
    np.testing.assert_allclose(cp.asnumpy(rw_t), drw, rtol=2e-4, atol=2e-3)

    mru, mrv, mrw = np_coriolis_curvature(
        d["ru"], d["rv"], d["u"], d["v"], d["w"], d["mut"],
        d["msft"], d["msfu"], d["msfv"], d["f"], d["e"], coord, 200.0, 250.0,
        sina=d["sina"], cosa=d["cosa"])
    np.testing.assert_allclose(mru, dru, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mrv, drv, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mrw, drw, rtol=1e-12, atol=1e-12)


@requires_gpu
@pytest.mark.gpu
def test_coriolis_rotated_uniform_closed_form():
    # Uniform winds/coupled momenta on a constant-(f, e, sina, cosa) rotated
    # frame, msf == 1: every stencil average is exact, so the WRF coriolis
    # forms (module_big_step_utilities_em.F:3726-3729, :3800-3803,
    # :3839-3844) reduce to hand-computable FP64 closed forms per level:
    #   dRU/dt = f*RV - e*cosa*<rw>          - u*<rw>/a
    #   dRV/dt = -f*RU + e*sina*<rw>         - v*<rw>/a
    #   dRW/dt(interior) = e*(cosa*RU - sina*RV) + (RU*u + RV*v)/a
    # FAILED before the sina/cosa fix on all three equations.
    import cupy as cp
    from woof.core import constants as c
    from woof.core.dycore import launch_coriolis_curvature
    from woof.core.grid import make_vertical_coord
    nz, ny, nx = 6, 5, 7
    f0, e0, mu0 = 1e-4, 1.4e-4, 1.0e5
    u0, v0, w0 = 10.0, 4.0, 0.3
    sina0 = -0.26                      # real74 d01 west-edge magnitude
    cosa0 = np.sqrt(1.0 - sina0 * sina0)
    ru0, rv0 = mu0 * u0, mu0 * v0
    coord = make_vertical_coord(nz)
    ones_t = cp.ones((ny, nx), cp.float32)
    d = {
        "ru": cp.full((nz, ny, nx + 1), ru0, cp.float32),
        "rv": cp.full((nz, ny + 1, nx), rv0, cp.float32),
        "u": cp.full((nz, ny, nx + 1), u0, cp.float32),
        "v": cp.full((nz, ny + 1, nx), v0, cp.float32),
        "w": cp.full((nz + 1, ny, nx), w0, cp.float32),
        "mut": cp.full((ny, nx), mu0, cp.float32),
    }
    d["w"][0] = d["w"][-1] = 0.0
    ru_t = cp.zeros((nz, ny, nx + 1), cp.float32)
    rv_t = cp.zeros((nz, ny + 1, nx), cp.float32)
    rw_t = cp.zeros((nz + 1, ny, nx), cp.float32)
    launch_coriolis_curvature(
        d["ru"], d["rv"], d["u"], d["v"], d["w"], d["mut"],
        ones_t, cp.ones((ny, nx + 1), cp.float32),
        cp.ones((ny + 1, nx), cp.float32),
        f0 * ones_t, e0 * ones_t,
        cp.asarray(coord.c1f, cp.float32), cp.asarray(coord.c2f, cp.float32),
        cp.asarray(coord.fnm, cp.float32), cp.asarray(coord.fnp, cp.float32),
        1000.0, 1000.0, ru_t, rv_t, rw_t,
        sina=cp.full((ny, nx), sina0, cp.float32),
        cosa=cp.full((ny, nx), cosa0, cp.float32))

    # FP64 hand computation: rw = (c1f*mu0 + c2f)*w (msf == 1), so the
    # half-level 4-point average is exact and per-level.
    c1f = np.asarray(coord.c1f, dtype=np.float64)
    c2f = np.asarray(coord.c2f, dtype=np.float64)
    wf = np.full(nz + 1, w0)
    wf[0] = wf[-1] = 0.0
    rwf = (c1f * mu0 + c2f) * wf
    rw4 = 0.5 * (rwf[:nz] + rwf[1:])                       # (nz,)
    exp_ru = f0 * rv0 - e0 * cosa0 * rw4 - u0 * c.RERADIUS * rw4
    exp_rv = -f0 * ru0 + e0 * sina0 * rw4 - v0 * c.RERADIUS * rw4
    exp_rw = e0 * (cosa0 * ru0 - sina0 * rv0) \
        + c.RERADIUS * (ru0 * u0 + rv0 * v0)
    np.testing.assert_allclose(cp.asnumpy(ru_t),
                               np.broadcast_to(exp_ru[:, None, None],
                                               (nz, ny, nx + 1)), rtol=1e-5)
    np.testing.assert_allclose(cp.asnumpy(rv_t),
                               np.broadcast_to(exp_rv[:, None, None],
                                               (nz, ny + 1, nx)), rtol=1e-5)
    got_w = cp.asnumpy(rw_t)
    np.testing.assert_allclose(got_w[1:nz], exp_rw, rtol=1e-5)
    assert bool((got_w[0] == 0).all()) and bool((got_w[nz] == 0).all())


@requires_gpu
@pytest.mark.gpu
def test_coriolis_sina_zero_bitwise_old_path():
    # Rotation-free grids are unchanged: with sina == 0 / cosa == 1 (passed
    # explicitly AND via the defaults) the kernel must reproduce the
    # pre-sina/cosa kernel BITWISE (pin captured from the pre-fix tree on
    # this exact fixture).  WRF's identity setting for unrotated projections
    # (module_big_step_utilities_em.F:3703-3704 "generally sina=0, cosa=1").
    #
    # PER CARD since 2026-09-17 (tests/_card_pins.py).  The original
    # capture (2026-07-19, card unrecorded) against the same launch on
    # test host 1's RTX 4090 (compute capability 8.9, driver 610.57.04): 105 of
    # 560 words differ, max 3.05e-05, max relative 1.07e-05, one launch of
    # launch_coriolis_curvature, ULP-scale: the card, as for the phase-2
    # pin above.  The 4090 file (tests/data/coriolis_map_sina0_pin.rtx4090
    # .npz) was written by tools/recapture_card_pins.py --pin
    # coriolis_map_sina0_pin with its reading beside it in
    # tests/data/receipts/pin-gates/.  A card with no file skips.
    import cupy as cp
    import _card_pins
    from woof.core.dycore import launch_coriolis_curvature
    from woof.core.grid import make_vertical_coord
    reason = _card_pins.skip_reason("coriolis_map_sina0_pin")
    if reason is not None:
        pytest.skip(reason)
    ref = np.load(_card_pins.path("coriolis_map_sina0_pin"))
    d = _random_rotational_inputs(seed=1974)
    nz, ny, nxp1 = d["u"].shape
    nx = nxp1 - 1
    coord = make_vertical_coord(nz, stretch=2.0)
    dev = {k: cp.asarray(v, cp.float32) for k, v in d.items()}
    identity = {"sina": cp.zeros((ny, nx), cp.float32),
                "cosa": cp.ones((ny, nx), cp.float32)}
    for rot in ({}, identity):
        ru_t = cp.zeros((nz, ny, nx + 1), cp.float32)
        rv_t = cp.zeros((nz, ny + 1, nx), cp.float32)
        rw_t = cp.zeros((nz + 1, ny, nx), cp.float32)
        launch_coriolis_curvature(
            dev["ru"], dev["rv"], dev["u"], dev["v"], dev["w"], dev["mut"],
            dev["msft"], dev["msfu"], dev["msfv"], dev["f"], dev["e"],
            cp.asarray(coord.c1f, cp.float32),
            cp.asarray(coord.c2f, cp.float32),
            cp.asarray(coord.fnm, cp.float32),
            cp.asarray(coord.fnp, cp.float32),
            200.0, 250.0, ru_t, rv_t, rw_t, **rot)
        np.testing.assert_array_equal(cp.asnumpy(ru_t), ref["ru_t"])
        np.testing.assert_array_equal(cp.asnumpy(rv_t), ref["rv_t"])
        np.testing.assert_array_equal(cp.asnumpy(rw_t), ref["rw_t"])


@requires_gpu
@pytest.mark.gpu
def test_coriolis_signs_uniform_flow():
    # Uniform (u0, v0) on an f-plane, msf == 1, w == 0: the WRF coriolis
    # forms reduce to dU/dt = +f*V, dV/dt = -f*U, dW/dt(interior) = +e*U
    # exactly (all averages of uniform fields are exact).
    import cupy as cp
    from woof.core.dycore import launch_coriolis_curvature
    from woof.core.grid import make_vertical_coord
    nz, ny, nx = 6, 5, 7
    f0, e0, mu0, u0, v0 = 1e-4, 5e-5, 1.0e5, 10.0, 4.0
    coord = make_vertical_coord(nz)
    ones_t = cp.ones((ny, nx), cp.float32)
    d = {
        "ru": cp.full((nz, ny, nx + 1), mu0 * u0, cp.float32),
        "rv": cp.full((nz, ny + 1, nx), mu0 * v0, cp.float32),
        "u": cp.full((nz, ny, nx + 1), u0, cp.float32),
        "v": cp.full((nz, ny + 1, nx), v0, cp.float32),
        "w": cp.zeros((nz + 1, ny, nx), cp.float32),
        "mut": cp.full((ny, nx), mu0, cp.float32),
    }
    ru_t = cp.zeros((nz, ny, nx + 1), cp.float32)
    rv_t = cp.zeros((nz, ny + 1, nx), cp.float32)
    rw_t = cp.zeros((nz + 1, ny, nx), cp.float32)
    launch_coriolis_curvature(
        d["ru"], d["rv"], d["u"], d["v"], d["w"], d["mut"],
        ones_t, cp.ones((ny, nx + 1), cp.float32),
        cp.ones((ny + 1, nx), cp.float32),
        f0 * ones_t, e0 * ones_t,
        cp.asarray(coord.c1f, cp.float32), cp.asarray(coord.c2f, cp.float32),
        cp.asarray(coord.fnm, cp.float32), cp.asarray(coord.fnp, cp.float32),
        1000.0, 1000.0, ru_t, rv_t, rw_t)
    np.testing.assert_allclose(cp.asnumpy(ru_t), f0 * mu0 * v0, rtol=1e-5)
    np.testing.assert_allclose(cp.asnumpy(rv_t), -f0 * mu0 * u0, rtol=1e-5)
    # w: interior levels only get e*<ru> + curvature (u*ru + v*rv)/a
    from woof.core import constants as c
    expect_w = e0 * mu0 * u0 + c.RERADIUS * mu0 * (u0 * u0 + v0 * v0)
    got_w = cp.asnumpy(rw_t)
    np.testing.assert_allclose(got_w[1:nz], expect_w, rtol=1e-5)
    assert bool((got_w[0] == 0).all()) and bool((got_w[nz] == 0).all())


# ---- msf-weighted advection kernels vs mirrors ------------------------------

@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("which", ["scalar", "u", "v", "w"])
def test_flux_div_msf_matches_mirror(which):
    import cupy as cp
    from woof.core import advection as adv
    from woof.core.grid import make_vertical_coord
    from woof.verify import npref
    rng = np.random.default_rng(7)
    nz, ny, nx = 10, 9, 12
    coord = make_vertical_coord(nz, stretch=2.0)
    f = {
        "q": rng.normal(300, 5, (nz, ny, nx)),
        "u": rng.normal(0, 5, (nz, ny, nx + 1)),
        "v": rng.normal(0, 5, (nz, ny + 1, nx)),
        "w": rng.normal(0, 1, (nz + 1, ny, nx)),
        "ru": rng.normal(0, 10, (nz, ny, nx + 1)),
        "rv": rng.normal(0, 10, (nz, ny + 1, nx)),
        "rw": rng.normal(0, 1, (nz + 1, ny, nx)),
    }
    f["w"][0] = f["w"][-1] = 0.0
    f["rw"][0] = f["rw"][-1] = 0.0
    msf = {
        "scalar": 1.0 + 0.1 * rng.random((ny, nx)),
        "u": 1.0 + 0.1 * rng.random((ny, nx + 1)),
        "v": 1.0 + 0.1 * rng.random((ny + 1, nx)),
    }
    msf["u"][:, -1] = msf["u"][:, 0]        # periodic duplicates
    msf["v"][-1, :] = msf["v"][0, :]
    msf["w"] = msf["scalar"]
    f32 = {k: cp.asarray(v, cp.float32) for k, v in f.items()}
    launchers = {"scalar": adv.launch_flux_div_scalar,
                 "u": adv.launch_flux_div_u,
                 "v": adv.launch_flux_div_v,
                 "w": adv.launch_flux_div_w}
    mirrors = {"scalar": npref.np_flux_div_scalar,
               "u": npref.np_flux_div_u,
               "v": npref.np_flux_div_v,
               "w": npref.np_flux_div_w}
    field = {"scalar": "q", "u": "u", "v": "v", "w": "w"}[which]
    tend = cp.zeros(f[field].shape, cp.float32)
    launchers[which](f32[field], f32["ru"], f32["rv"], f32["rw"], tend,
                     coord, 100.0, 120.0,
                     msf=cp.asarray(msf[which], cp.float32))
    ref = mirrors[which](f[field], f["ru"], f["rv"], f["rw"], coord,
                         100.0, 120.0, msf=msf[which])
    np.testing.assert_allclose(cp.asnumpy(tend), ref, rtol=2e-4, atol=2e-4)


# ---- msf-weighted acoustic substep vs mirrors -------------------------------

@requires_gpu
@pytest.mark.gpu
def test_acoustic_substep_msf_matches_mirror():
    import cupy as cp
    from woof.core.acoustic import acoustic_substep
    from woof.verify.npref import (np_acoustic_substep,
                                    random_acoustic_state, snapshot)
    s, cfg = random_acoustic_state(seed=17, msf_amp=0.1)
    assert s.has_msf
    before = snapshot(s)
    acoustic_substep(s, cfg, dtau=0.5, first=False)
    ref = np_acoustic_substep(before, cfg, dtau=0.5)
    for name in ("u_pp", "v_pp", "w_pp", "ph_pp", "mu_pp", "th_pp", "p_pp"):
        np.testing.assert_allclose(cp.asnumpy(getattr(s, name)), ref[name],
                                   rtol=3e-4, atol=2e-6, err_msg=name)


@requires_gpu
@pytest.mark.gpu
def test_emdiv_msf_matches_mirror():
    import cupy as cp
    from woof.core.dycore import apply_emdiv_filter
    from woof.verify.npref import np_emdiv_uv, random_acoustic_state
    s, cfg = random_acoustic_state(seed=23, msf_amp=0.1)
    cfg = dataclasses.replace(cfg, emdiv=0.01)
    rng = np.random.default_rng(5)
    mudf = rng.normal(0, 1.0, (cfg.ny, cfg.nx)).astype(np.float32)
    before = {n: cp.asnumpy(getattr(s, n)).astype(np.float64)
              for n in ("u_pp", "v_pp", "msfu", "msfv")}
    coord = type("C", (), {"c1h": cp.asnumpy(s.c1h).astype(np.float64)})()
    apply_emdiv_filter(s, cfg, cp.asarray(mudf))
    u_ref, v_ref = np_emdiv_uv(before["u_pp"], before["v_pp"],
                               mudf.astype(np.float64), coord, cfg,
                               msfu=before["msfu"], msfv=before["msfv"])
    np.testing.assert_allclose(cp.asnumpy(s.u_pp), u_ref, rtol=2e-4, atol=1e-5)
    np.testing.assert_allclose(cp.asnumpy(s.v_pp), v_ref, rtol=2e-4, atol=1e-5)


@requires_gpu
@pytest.mark.gpu
def test_pd_fluxes_msf_matches_mirror():
    import cupy as cp
    from woof.core.grid import make_vertical_coord
    from woof.core.moist import launch_pd_fluxes, launch_pd_renorm_apply
    from woof.verify.npref import np_pd_fluxes, np_pd_renorm_apply
    rng = np.random.default_rng(11)
    nz, ny, nx = 8, 6, 10
    coord = make_vertical_coord(nz)
    q = np.clip(rng.normal(5e-3, 3e-3, (nz, ny, nx)), 0, None)
    q0 = np.clip(q + rng.normal(0, 1e-3, q.shape), 0, None)
    ru = rng.normal(0, 5e5, (nz, ny, nx + 1))
    rv = rng.normal(0, 5e5, (nz, ny + 1, nx))
    rw = rng.normal(0, 5e4, (nz + 1, ny, nx))
    rw[0] = rw[-1] = 0.0
    mut = np.full((ny, nx), 1e5) + rng.normal(0, 1e3, (ny, nx))
    msft = 1.0 + 0.1 * rng.random((ny, nx))
    dx, dy, dt = 500.0, 500.0, 2.0

    dev = {k: cp.asarray(v, cp.float32) for k, v in
           dict(q=q, q0=q0, ru=ru, rv=rv, rw=rw, mut=mut, msft=msft).items()}
    bufs = (cp.zeros((nz, ny, nx + 1), cp.float32),
            cp.zeros((nz, ny, nx + 1), cp.float32),
            cp.zeros((nz, ny + 1, nx), cp.float32),
            cp.zeros((nz, ny + 1, nx), cp.float32),
            cp.zeros((nz + 1, ny, nx), cp.float32),
            cp.zeros((nz + 1, ny, nx), cp.float32))
    launch_pd_fluxes(dev["q"], dev["q0"], dev["ru"], dev["rv"], dev["rw"],
                     dev["mut"], coord, dx, dy, dt, *bufs,
                     msft=dev["msft"])
    ref = np_pd_fluxes(q, q0, ru, rv, rw, mut, coord, dx, dy, dt, msft=msft)
    for got, want, name in zip(bufs, ref,
                               ("fxl", "fxc", "fyl", "fyc", "fzl", "fzc")):
        np.testing.assert_allclose(cp.asnumpy(got), want, rtol=3e-4,
                                   atol=3e-1, err_msg=name)

    tend = cp.zeros((nz, ny, nx), cp.float32)
    launch_pd_renorm_apply(dev["q0"], dev["mut"], *bufs, tend=tend,
                           coord=coord, dx=dx, dy=dy, dt=dt,
                           msft=dev["msft"])
    ref_t = np_pd_renorm_apply(q0, mut, *ref, coord=coord, dx=dx, dy=dy,
                               dt=dt, msft=msft)
    np.testing.assert_allclose(cp.asnumpy(tend), ref_t, rtol=3e-4, atol=3.0)


# ---- guards ------------------------------------------------------------------

@requires_gpu
@pytest.mark.gpu
def test_open_bc_with_rotation_is_wired():
    import cupy as cp
    from woof.core.dycore import step
    from woof.verify.npref import random_acoustic_state
    s, cfg = random_acoustic_state(seed=2, ny=8)
    cfg = dataclasses.replace(cfg, open_x=True)
    s.set_map_coriolis(f=np.full((cfg.ny, cfg.nx), 1e-4))
    step(s, cfg)
    assert bool(cp.isfinite(s.u).all())
    assert bool(cp.isfinite(s.v).all())


# ---- physics: inertial oscillation -------------------------------------------

def _flat_base(cfg, N2=1e-4):
    from woof.core.grid import make_base_state, make_vertical_coord
    vc = make_vertical_coord(cfg.nz)
    th = lambda z: 300.0 * np.exp(N2 * np.asarray(z, float) / 9.81)
    b = make_base_state(vc, th, p_surf=cfg.p_surf, ztop=cfg.ztop)
    return vc, b


@requires_gpu
@pytest.mark.gpu
def test_inertial_oscillation_period():
    # f-plane (msf == 1, e == 0), uniform u0 = 10 m/s, no pressure gradient:
    # the wind vector must rotate anticyclonically with period 2*pi/f to
    # within 0.5% (plan gate).
    import cupy as cp
    from woof.core.dycore import step
    from woof.core.state import init_at_rest
    f0 = 1e-4
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=10e3, dy=10e3, ztop=8e3,
                    dt=60.0, run_seconds=0.0)
    vc, b = _flat_base(cfg)
    s = init_at_rest(cfg, vc, b)
    s.u[...] = 10.0
    s.set_map_coriolis(f=np.full((cfg.ny, cfg.nx), f0))

    period = 2 * np.pi / f0
    nsteps = int(round(1.05 * period / cfg.dt))          # ~1.05 periods
    kz, jy, ix = cfg.nz // 2, cfg.ny // 2, cfg.nx // 2
    us, vs = [], []
    for _ in range(nsteps):
        step(s, cfg)
        us.append(float(s.u[kz, jy, ix]))
        vs.append(float(s.v[kz, jy, ix]))
    us, vs = np.asarray(us), np.asarray(vs)
    # amplitude must be conserved (no spurious damping/growth)
    amp = np.hypot(us, vs)
    assert abs(amp[-1] - 10.0) < 0.05 * 10.0
    # phase-slope fit of the anticyclonic rotation angle
    theta = np.unwrap(np.arctan2(-vs, us))
    t = cfg.dt * (1.0 + np.arange(nsteps))
    omega = np.polyfit(t, theta, 1)[0]
    T_meas = 2 * np.pi / omega
    assert abs(T_meas - period) / period < 0.005, (T_meas, period)


# ---- physics: geostrophic balance hold ---------------------------------------

@requires_gpu
@pytest.mark.gpu
def test_geostrophic_jet_holds_6h():
    # Balanced barotropic-ish jet on an f-plane: build a mu'(y) ridge,
    # rebalance phi' hydrostatically, MEASURE the discrete PGF, invert the
    # discrete Coriolis average for u, then integrate 6 h.  The jet must
    # stay balanced: u drift and v excursion < 2% of the jet max (plan gate).
    import cupy as cp
    from woof.core import constants as c
    from woof.core import dycore
    from woof.core.diagnostics import update_diagnostics
    from woof.core.state import DTYPE, init_at_rest, mu_at_u_faces
    f0 = 1e-4
    cfg = RunConfig(nx=8, ny=64, nz=20, dx=20e3, dy=20e3, ztop=16e3,
                    dt=60.0, run_seconds=0.0)
    vc, b = _flat_base(cfg)
    s = init_at_rest(cfg, vc, b)

    # mu'(y) ridge (float64), phi' rebalanced with unchanged theta via the
    # same discrete recurrence as make_base_state.
    y = (np.arange(cfg.ny) + 0.5) * cfg.dy - 0.5 * cfg.ny * cfg.dy
    L = 150e3
    mup = 210.0 * np.exp(-((y / L) ** 2))                  # (ny,)
    mu_tot = float(b.mub) + mup
    p = (vc.c3h[:, None] * mu_tot[None] + vc.c4h[:, None] + b.p_top)
    alpha = c.RD * np.asarray(b.thb)[:, None] * (p / c.P0) ** c.RCP / p
    ph = np.zeros((cfg.nz + 1, cfg.ny))
    for k in range(cfg.nz):
        ph[k + 1] = ph[k] - vc.dnw[k] * (vc.c1h[k] * mu_tot
                                         + vc.c2h[k]) * alpha[k]
    php = ph - np.asarray(b.phb)[:, None]
    s.mup[...] = cp.asarray(np.broadcast_to(mup[None, :, None],
                                            (1, cfg.ny, cfg.nx))[0], DTYPE)
    s.php[...] = cp.asarray(np.broadcast_to(php[:, :, None],
                            (cfg.nz + 1, cfg.ny, cfg.nx)), DTYPE)

    # measure the discrete PGF with zero winds (rotation not yet enabled)
    update_diagnostics(s)
    for name in dycore._TENDENCIES:
        getattr(s, name)[...] = 0
    ru, rv, ww = dycore.stage_fluxes(s, cfg)
    dycore._add_slow_tendencies(s, cfg, ru, rv, ww)
    pgf = cp.asnumpy(s.rv_t).astype(np.float64)[:, :cfg.ny, 0]  # (nz, ny)

    # invert 0.5*(RU(j) + RU(j-1)) = pgf/f along periodic y (FFT; the
    # Nyquist null mode of the two-point average is projected out)
    G = pgf / f0
    m = np.fft.rfftfreq(cfg.ny) * cfg.ny                    # 0..ny/2
    H = 0.5 * (1.0 + np.exp(-2j * np.pi * m / cfg.ny))
    Gh = np.fft.rfft(G, axis=1)
    Gh[:, np.abs(H) < 1e-12] = 0.0
    Hs = np.where(np.abs(H) < 1e-12, 1.0, H)
    RU = np.fft.irfft(Gh / Hs[None], n=cfg.ny, axis=1)      # (nz, ny)

    mux = cp.asnumpy(mu_at_u_faces(s.total_mu())).astype(np.float64)
    chm = (vc.c1h[:, None] * mux[None, :, 0] + vc.c2h[:, None])  # x-uniform
    u = RU / chm                                            # (nz, ny)
    u3 = np.broadcast_to(u[:, :, None], (cfg.nz, cfg.ny, cfg.nx + 1))
    s.u[...] = cp.asarray(u3, DTYPE)
    u0max = float(np.abs(u).max())
    assert 5.0 < u0max < 20.0                               # sane jet scale

    s.set_map_coriolis(f=np.full((cfg.ny, cfg.nx), f0))
    dycore.run_steps(s, cfg, n=360)                         # 6 h
    du = cp.asnumpy(s.u).astype(np.float64) - u3
    vmax = float(cp.abs(s.v).max())
    assert float(np.abs(du).max()) < 0.02 * u0max, float(np.abs(du).max())
    assert vmax < 0.02 * u0max, vmax
    assert not dycore.stability_report(s, cfg)["nan"]
