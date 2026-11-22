# tests/test_small_step_lateral_wrap.py
"""The small-step uv kernels must obey the configured lateral boundary.

What was wrong
--------------
``small_step_init_uv`` and ``small_step_finish_uv`` build WRF's ``muu``/
``muus`` and ``muv``/``muvs`` inline -- the column mass carried to a u/v
face as the mean of the two mass cells that share it, which
``small_step_prep`` multiplies into the acoustic perturbation
(module_small_step_em.F:225-228) and ``small_step_finish`` divides back out
(:96-98).  WRF builds that pair in ``calc_mu_uv``
(module_big_step_utilities_em.F:59-115 for x, :118-174 for y) and picks the
off-domain neighbour from ``config_flags%periodic_x``/``periodic_y``::

    west face i = ids:   im = its   ; periodic_x -> its-1
    east face i = ide:   im = ite-1 ; periodic_x -> ite

The periodic branch reads the halo cell the period fill has loaded from the
opposite edge; the non-periodic branch names the boundary cell twice, so the
boundary face carries that cell's own mass.  WOOF-ArWen wrapped unconditionally
(``i % nx`` / ``(i-1+nx) % nx``, and the j equivalent) regardless of
``open_x``/``open_y``/``specified``/``nested``, so on every non-periodic
domain the west boundary face took its mass from the EAST-most mass column
and the east face from the west-most -- a physical coupling of the two
boundaries, present at both ends of every acoustic rung of every RK stage.
These were the last two sites where a wrapped index SURVIVED on a
non-periodic domain.  Every other boundary-sensitive kernel already takes
the flag (``slow_pgf`` skips the boundary face, ``advance_mu_th`` and
``stage_fluxes`` clamp, the advection family and ``smag2d`` narrow); the one
remaining ``i % nx`` -- ``advance_uv``'s -- is dead at the boundary either
way, because its spec_zone guard never reaches the boundary face under
specified/nested and ``acoustic.py`` overwrites both boundary face sets from
their saved pre-substep values under open boundaries
(``openbc_upp_faces``/``openbc_vpp_faces``).

What this file pins
-------------------
* :func:`test_uv_kernels_match_wrf_calc_mu_uv` -- the ORACLE.  An FP32 NumPy
  mirror of the two kernels' arithmetic (every ``rn_*`` op is one
  round-to-nearest FP32 binary operation, which is exactly what NumPy does
  on float32 arrays, and neither side contracts) is compared bit-for-bit for
  a periodic, a specified and an open config, with and without map factors.
  The periodic rows are the wrap; the non-periodic rows are the clamp.
* :func:`test_periodic_uv_is_exactly_the_wrap` -- the bit-identity gate for
  every existing periodic run: the kernel under a periodic config must equal
  the same kernel with the flags forced to the wrap, AND must differ from
  the same kernel with the flags forced to the clamp.  A test whose two
  branches cannot be told apart proves nothing, so the second half is what
  makes the first half mean something.
* :func:`test_one_step_does_not_couple_the_opposite_edge` -- the NEGATIVE
  CONTROL, and the one that fails on the unpatched kernel.  One mass column
  at the west edge of a 96-cell-wide domain is bumped by 25 Pa and the
  domain is stepped ONCE; nothing in the eastern 8 columns may move.  A step
  reaches 16 cells (``tilestream.harness.halo_radius``), so 88 cells of
  separation is five influence cones: only a wrap can carry the signal.
  Measured at 684bcd5 (the unpatched kernel), 96x64x24, dt=3 s, over all
  76 device arrays the state owns -- far-edge arrays that moved (and cells
  within them), x axis / y axis::

      periodic    33 (261174) / 33 (392525)   <- MUST couple
      specified    1 (21)     /  1 (20)       <- must not
      open        34 (143108) / 34 (210667)   <- must not

  With the fix the two lower rows are 0 arrays and the periodic row is
  unchanged.  The specified row is small because the spec-zone epilogue
  overwrites the boundary-face u the wrap corrupted; what survives is the
  stage mass flux ``rk_ru``/``rk_rv``, one float32 ULP in each of those 21
  and 20 cells.  The PERIODIC row is the positive control -- there the far
  edge MUST move, which is what proves the probe can see coupling at all.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu

#: 96 x 64 mass cells at dx = 500 m.  The width is not decorative: the
#: opposite-edge probe needs the far slab to sit outside the one-step
#: influence cone (16 cells at time_step_sound=4), and 88 cells of
#: separation is the margin that makes "it moved" mean "it wrapped".
NX, NY, NZ = 96, 64, 24
BUMP_PA = 25.0
FAR = 8


def _cfg(regime, **overrides):
    from tilestream import harness

    if regime == "periodic":
        return harness.make_config(NX, NY, NZ, **overrides)
    if regime == "open":
        return harness.make_config(NX, NY, NZ, periodic=False, open_x=True,
                                   open_y=True, **overrides)
    if regime == "specified":
        return harness.make_config(NX, NY, NZ, periodic=False,
                                   specified=True, **overrides)
    raise AssertionError(regime)


def _seeded_state(cfg, *, msf_amp=0.0, seed=20260731):
    """A state with mup0 != mup and u0 != u, so the kernels are not trivial.

    ``harness.make_state`` seeds ``mup``/``u``/``v`` but leaves the time-t
    copies at rest.  With ``mup0 == mup`` the init kernel's ct and cs are
    equal and its output is exactly zero whatever the face neighbour is --
    an oracle comparison against zero would pass on the wrap too.
    """
    import cupy as cp
    from tilestream import harness

    state = harness.make_state(cfg, seed, msf_amp=msf_amp,
                               f_amp=(1e-4 if msf_amp else 0.0))
    rng = np.random.default_rng(seed + 17)
    for name, amp in (("mup0", 20.0), ("u0", 0.05), ("v0", 0.05),
                      ("mu_pp", 0.5), ("u_pp", 0.2), ("v_pp", 0.2)):
        arr = getattr(state, name)
        arr[...] = cp.asarray(amp * rng.standard_normal(arr.shape),
                              dtype=arr.dtype)
    return state


# --------------------------------------------------------------------------
# the FP32 oracle: WRF calc_mu_uv's two index rules, spelled out
# --------------------------------------------------------------------------

def _face_neighbours(n, boundary):
    """``(ia, ib)`` for the ``n + 1`` faces of an ``n``-cell axis.

    ``boundary`` false is WRF's periodic_x branch (the halo cell, which the
    period fill holds from the opposite edge); true is the plain branch,
    which names the boundary cell twice.
    """
    i = np.arange(n + 1)
    if boundary:
        return np.minimum(i, n - 1), np.maximum(i - 1, 0)
    return i % n, (i - 1 + n) % n


def _gather(field2d, index, axis):
    """``field2d`` sampled along ``axis`` at the face-neighbour indices."""
    return field2d[:, index] if axis == "x" else field2d[index, :]


def _host(state, *names):
    """The named state arrays on the host, still float32."""
    import cupy as cp

    return [cp.asnumpy(getattr(state, name)) for name in names]


def _map_factors(state):
    """``(msfu, msfv)`` on the host, or ``(None, None)`` without a map."""
    if not state.has_msf:
        return None, None
    return tuple(_host(state, "msfu", "msfv"))


def _mirror_init_uv(state, boundary_x, boundary_y):
    """FP32 mirror of ``small_step_init_uv``: mu*u'' = C(muus)*u_t - C(muu)*u.

    Every ``rn_*`` in the kernel is one round-to-nearest FP32 binary op, and
    NumPy rounds each float32 operation the same way without contracting, so
    this reproduces the kernel exactly -- not to a tolerance.
    """
    mub2d, mup0, mup, u0, u, v0, v = _host(
        state, "mub2d", "mup0", "mup", "u0", "u", "v0", "v")
    c1h, c2h = (arr[:, None, None] for arr in _host(state, "c1h", "c2h"))
    half = np.float32(0.5)
    msfu, msfv = _map_factors(state)

    def couple(ia, ib, axis, field0, field, msf):
        mt = half * (_gather(mub2d + mup0, ia, axis)
                     + _gather(mub2d + mup0, ib, axis))
        ms = half * (_gather(mub2d + mup, ia, axis)
                     + _gather(mub2d + mup, ib, axis))
        value = ((c1h * mt[None] + c2h) * field0
                 - (c1h * ms[None] + c2h) * field)
        return value if msf is None else value / msf[None]

    ia, ib = _face_neighbours(NX, boundary_x)
    ja, jb = _face_neighbours(NY, boundary_y)
    return (couple(ia, ib, "x", u0, u, msfu),
            couple(ja, jb, "y", v0, v, msfv))


def _mirror_finish_uv(state, boundary_x, boundary_y):
    """FP32 mirror of ``small_step_finish_uv``, the same face mass undone."""
    mub2d, mup, mu_pp, u, v, u_pp, v_pp = _host(
        state, "mub2d", "mup", "mu_pp", "u", "v", "u_pp", "v_pp")
    c1h, c2h = (arr[:, None, None] for arr in _host(state, "c1h", "c2h"))
    half = np.float32(0.5)
    msfu, msfv = _map_factors(state)

    def uncouple(ia, ib, axis, field, pert, msf):
        msa = _gather(mub2d + mup, ia, axis)
        msb = _gather(mub2d + mup, ib, axis)
        mna = msa + _gather(mu_pp, ia, axis)
        mnb = msb + _gather(mu_pp, ib, axis)
        cs = c1h * (half * (msa + msb))[None] + c2h
        cn = c1h * (half * (mna + mnb))[None] + c2h
        pp = pert if msf is None else pert * msf[None]
        return (cs * field + pp) / cn

    ia, ib = _face_neighbours(NX, boundary_x)
    ja, jb = _face_neighbours(NY, boundary_y)
    return (uncouple(ia, ib, "x", u, u_pp, msfu),
            uncouple(ja, jb, "y", v, v_pp, msfv))


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("regime", ("periodic", "specified", "open"))
@pytest.mark.parametrize("msf_amp", (0.0, 0.2))
def test_uv_kernels_match_wrf_calc_mu_uv(regime, msf_amp):
    """Both kernels reproduce WRF calc_mu_uv's face mass, bit for bit.

    RED at 684bcd5 for every non-periodic row: the kernel wraps where the
    mirror clamps, so the two boundary-normal face columns disagree.
    """
    import cupy as cp
    from woof.core import dycore

    cfg = _cfg(regime)
    boundary = regime != "periodic"
    state = _seeded_state(cfg, msf_amp=msf_amp)

    want_u_pp, want_v_pp = _mirror_init_uv(state, boundary, boundary)
    dycore._prepare_small_step_init_launch(state, cfg)()
    cp.cuda.runtime.deviceSynchronize()
    np.testing.assert_array_equal(cp.asnumpy(state.u_pp), want_u_pp)
    np.testing.assert_array_equal(cp.asnumpy(state.v_pp), want_v_pp)

    want_u, want_v = _mirror_finish_uv(state, boundary, boundary)
    dycore._prepare_small_step_finish_launch(state, cfg)()
    cp.cuda.runtime.deviceSynchronize()
    np.testing.assert_array_equal(cp.asnumpy(state.u), want_u)
    np.testing.assert_array_equal(cp.asnumpy(state.v), want_v)


@requires_gpu
@pytest.mark.gpu
def test_periodic_uv_is_exactly_the_wrap():
    """A periodic config keeps the pre-fix arithmetic, and the flag is live.

    Half one: the periodic launch equals the mirror's wrap branch exactly --
    that is the bit-identity every existing periodic run rests on.  Half
    two: forcing the clamp changes the answer, and only in the two
    boundary-normal face columns/rows.  Without half two the first half
    would also pass on a kernel that ignored the flag entirely.
    """
    import cupy as cp
    from woof.core import dycore

    cfg = _cfg("periodic")
    state = _seeded_state(cfg, msf_amp=0.2)
    wrap_u_pp, wrap_v_pp = _mirror_init_uv(state, False, False)
    clamp_u_pp, clamp_v_pp = _mirror_init_uv(state, True, True)

    dycore._prepare_small_step_init_launch(state, cfg)()
    cp.cuda.runtime.deviceSynchronize()
    got_u_pp = cp.asnumpy(state.u_pp)
    got_v_pp = cp.asnumpy(state.v_pp)
    np.testing.assert_array_equal(got_u_pp, wrap_u_pp)
    np.testing.assert_array_equal(got_v_pp, wrap_v_pp)

    assert not np.array_equal(clamp_u_pp, wrap_u_pp)
    assert not np.array_equal(clamp_v_pp, wrap_v_pp)
    # and the disagreement is confined to the boundary-normal faces
    moved_i = np.unique(np.argwhere(clamp_u_pp != wrap_u_pp)[:, 2])
    moved_j = np.unique(np.argwhere(clamp_v_pp != wrap_v_pp)[:, 1])
    assert set(moved_i.tolist()) <= {0, NX}
    assert set(moved_j.tolist()) <= {0, NY}


@requires_gpu
@pytest.mark.gpu
def test_prepared_launches_take_the_flags_from_the_config(monkeypatch):
    """The decision is the config's, not a module global or a measurement.

    Also the only row that covers a SINGLE non-periodic axis: an ``open_x``
    channel keeps the y wrap, and the two flags must not be collapsed into
    one.
    """
    from woof.core import dycore

    captured = []

    def recording_kernel(_module, function):
        def launch(_grid, _block, args):
            if function.endswith("_uv"):
                captured.append((function, int(args[-5]), int(args[-4])))
        return launch

    state = _seeded_state(_cfg("periodic"))
    monkeypatch.setattr(dycore, "get_kernel", recording_kernel)
    for regime, want in (("periodic", (0, 0)), ("specified", (1, 1)),
                         ("open", (1, 1))):
        cfg = _cfg(regime)
        dycore._prepare_small_step_init_launch(state, cfg)()
        dycore._prepare_small_step_finish_launch(state, cfg)()
        assert [row[1:] for row in captured] == [want, want], regime
        captured.clear()
    channel = SimpleNamespace(open_x=True, open_y=False, specified=False,
                              nested=False)
    dycore._prepare_small_step_init_launch(state, channel)()
    assert captured == [("small_step_init_uv", 1, 0)]


# --------------------------------------------------------------------------
# the negative control
# --------------------------------------------------------------------------

def _step_once_with_bumped_edge(regime, axis):
    """Return ``(clean, bumped)`` device-array maps after ONE step.

    Both runs share one boundary-forcing table set, built from the
    UNPERTURBED state, so the only difference between them is the 25 Pa on
    one mass column of the perturbed edge.
    """
    import cupy as cp
    from tilestream import harness
    from woof.core.diagnostics import update_diagnostics

    cfg = _cfg(regime)
    tables = None
    if regime == "specified":
        from woof.ingest.lateral_bc import build_state_lateral_boundaries
        reference = harness.make_state(cfg)
        tables = build_state_lateral_boundaries(
            [reference, reference], [0.0, 3600.0])   # time-constant frames
    out = []
    for bump in (False, True):
        state = harness.make_state(cfg)
        if bump:
            if axis == "x":
                state.mup[:, 0] += cp.float32(BUMP_PA)
            else:
                state.mup[0, :] += cp.float32(BUMP_PA)
            update_diagnostics(state, cfg.hypsometric_opt)  # keep p/al/alt
        if tables is not None:                              # consistent
            from woof.ingest.lateral_bc import attach_lateral_boundaries
            attach_lateral_boundaries(state, tables)
        harness.run_steps(state, cfg, 1)
        arrays = {"state/" + n: v for n, v in vars(state).items()
                  if isinstance(v, cp.ndarray)}
        arrays.update({"scratch/" + str(slot): v
                       for slot, v in state._scratch.items()
                       if isinstance(v, cp.ndarray)})
        out.append({name: cp.asnumpy(v) for name, v in arrays.items()})
    return out


def _far_slabs(clean, bumped, axis):
    """``(far, near)``: names whose far / near ``FAR``-wide slab differs.

    Only arrays whose relevant axis is a genuine horizontal domain axis are
    eligible: the open-BC scratch keeps its two boundary face sets in a
    length-2 trailing axis (``openbc_upp_faces`` is ``(nz, ny, 2)``), and
    slicing that as if it were x would score the WEST face as an east-edge
    move.
    """
    span = {"x": (NX, NX + 1), "y": (NY, NY + 1)}[axis]
    low, high = slice(None, FAR), slice(-FAR, None)
    tail = () if axis == "x" else (slice(None),)
    near_slab, far_slab = (..., low) + tail, (..., high) + tail
    far, near = [], []
    for name in sorted(clean):
        a, b = clean[name], bumped[name]
        if a.ndim < 2 or a.shape != b.shape:
            continue
        if (a.shape[-1] if axis == "x" else a.shape[-2]) not in span:
            continue
        if not np.array_equal(a[far_slab], b[far_slab]):
            far.append(name)
        if not np.array_equal(a[near_slab], b[near_slab]):
            near.append(name)
    return far, near


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("axis", ("x", "y"))
def test_periodic_step_DOES_couple_the_opposite_edge(axis):
    """The probe's positive control: a period MUST reach around.

    If this fails the negative controls below are vacuous -- they would be
    reporting that a perturbation which changed nothing changed nothing.
    """
    clean, bumped = _step_once_with_bumped_edge("periodic", axis)
    far, near = _far_slabs(clean, bumped, axis)
    assert near, "the 25 Pa bump did nothing even at its own edge"
    assert len(far) >= 10, far


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("regime", ("specified", "open"))
@pytest.mark.parametrize("axis", ("x", "y"))
def test_one_step_does_not_couple_the_opposite_edge(regime, axis):
    """RED at 684bcd5: the wrap carries the west bump to the east edge.

    Unpatched, ``specified`` moves 21 (x) / 20 (y) cells of the stage mass
    flux by one ULP and ``open`` moves 34 arrays on either axis; patched,
    both move nothing.  The near-edge assertion is what stops a no-op
    perturbation from passing this by doing nothing at all.
    """
    clean, bumped = _step_once_with_bumped_edge(regime, axis)
    far, near = _far_slabs(clean, bumped, axis)
    assert near, "the 25 Pa bump did nothing even at its own edge"
    assert far == [], (
        f"{regime} domain coupled its {axis} boundaries in one step: {far}")


if __name__ == "__main__":  # standalone: python -m tests.test_...
    import sys

    sys.exit(pytest.main([__file__, "-x", "-q"]))
