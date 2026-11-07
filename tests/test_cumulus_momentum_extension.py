"""``CumulusResult``'s momentum pair: inert when absent, applied when present.

WHY THE FIRST HALF IS THE ESSENTIAL ONE. The extension exists for New
Tiedtke, which is the first cumulus scheme in this tree with ``lmfdudv``
(``cu_ntiedtke.F90:55``, a PARAMETER, so its momentum update is not a
runtime option). Every scheme the owner runs today -- Grell-Freitas and
Kain-Fritsch -- supplies neither field, and the whole point of capturing
four baselines before this landed is that their answers must not move by
one bit.

Inertness here is **by construction, not by care**: with ``ru`` absent,
``couple_column_tendencies`` returns the same zero stacks it always did.
This file asserts that, and the forecast-level gate
(``docs/ntiedtke/PHASE2-BASELINES.md``) proves it on real runs.

AND THE SECOND HALF IS THE NEGATIVE CONTROL. An extension that is inert
because it never does anything would pass the first test perfectly. So the
supplied path is checked to actually move the momentum stacks, and to go
through the SAME face interpolation YSU's momentum takes rather than a
second copy of it.
"""
from __future__ import annotations

import pytest


def _state_and_cfg():
    """A small domain built by the PRODUCTION constructor.

    Same shape as test_health.py's inventory config: a real DomainState
    rather than a synthetic stand-in, so the map factors, c1h/c2h and
    total_mu the coupling reads are the ones the driver would hand it.
    """
    from woof.config import RunConfig
    from woof.core.state import DomainState

    import cupy as cp

    cfg = RunConfig(
        nx=6, ny=5, nz=8, dx=2000.0, dy=2000.0, ztop=8000.0,
        dt=10.0, run_seconds=0.0, time_step_sound=4, moist=True,
        mp_physics=10, ra_physics=4, cu_physics=1,
        sf_sfclay_physics=1, sf_surface_physics=2, bl_pbl_physics=1)
    state = DomainState(cfg)
    # A FRESH DomainState HAS NO MASS: mub, c1h and c2h are all zero, so
    # the dry-mass coupling factor chm is zero and EVERY tendency this
    # function returns is zero whatever it is given. The inertness tests
    # would pass on that state and so would a function that did nothing,
    # which is the degeneracy this file's negative control exists to
    # refuse -- and it caught it on the first run.
    state.mub2d[...] = cp.float32(1.0e5)
    state.c1h[...] = cp.float32(1.0)
    state.c2h[...] = cp.float32(0.0)
    assert float(cp.abs(state.total_mu()).max()) > 0.0
    return state, cfg


@pytest.fixture(scope="module")
def bits():
    cp = pytest.importorskip("cupy")
    return cp


def test_the_pair_is_all_or_nothing():
    """Half a momentum update is not a physical state.

    WRF guards both with one ``lmfdudv``; a result carrying one and not
    the other would be applied half-way with nothing to signal it.
    """
    from woof.core.physics import CumulusResult

    for kwargs in ({"rucuten": object()}, {"rvcuten": object()}):
        with pytest.raises(ValueError, match="one momentum tendency"):
            CumulusResult(rthcuten=None, rqvcuten=None, **kwargs)


def test_absent_momentum_leaves_the_stacks_exactly_as_before(bits):
    """INERTNESS, at the coupling boundary.

    The zeros returned when ``ru`` is omitted must be the zeros the
    function returned before the extension existed -- same shape, same
    dtype, and identically zero. Anything else and Grell-Freitas moves.
    """
    cp = bits
    state, cfg = _state_and_cfg()
    from woof.core.physics import couple_column_tendencies

    nz, ny, nx = state.p.shape
    rtheta = cp.ones((nz, ny, nx), dtype=cp.float32)
    out = couple_column_tendencies(state, cfg, rtheta=rtheta,
                                   rqv=rtheta, rqc=rtheta)
    assert out.ru.shape == (nz, ny, nx + 1)
    assert out.rv.shape == (nz, ny + 1, nx)
    assert not bool(cp.any(out.ru)), "ru is non-zero with no momentum given"
    assert not bool(cp.any(out.rv)), "rv is non-zero with no momentum given"


def test_absent_momentum_is_BITWISE_what_the_scalar_path_produces(bits):
    """And the scalars must not move either.

    The extension inserted a branch above the return; this asserts the
    branch changed nothing about theta/qv/qc, which is the half a shape
    check would not notice.
    """
    cp = bits
    state, cfg = _state_and_cfg()
    from woof.core.physics import couple_column_tendencies

    nz, ny, nx = state.p.shape
    r = cp.arange(nz * ny * nx, dtype=cp.float32).reshape(nz, ny, nx)
    a = couple_column_tendencies(state, cfg, rtheta=r, rqv=r, rqc=r)
    b = couple_column_tendencies(state, cfg, rtheta=r, rqv=r, rqc=r,
                                 ru=None, rv=None)
    for name in ("rtheta", "rqv", "rqc", "ru", "rv"):
        x, y = getattr(a, name), getattr(b, name)
        assert bool(cp.all(x.view(cp.uint32) == y.view(cp.uint32))), name


def test_supplied_momentum_actually_reaches_the_stacks(bits):
    """THE NEGATIVE CONTROL.

    An extension that never applies anything passes every inertness test
    ever written. This one must move the faces.
    """
    cp = bits
    state, cfg = _state_and_cfg()
    from woof.core.physics import couple_column_tendencies

    nz, ny, nx = state.p.shape
    zero = cp.zeros((nz, ny, nx), dtype=cp.float32)
    du = cp.ones((nz, ny, nx), dtype=cp.float32)
    out = couple_column_tendencies(state, cfg, rtheta=zero, rqv=zero,
                                   rqc=zero, ru=du, rv=du)
    assert bool(cp.any(out.ru)), "supplied ru did not reach the stack"
    assert bool(cp.any(out.rv)), "supplied rv did not reach the stack"


def test_cumulus_momentum_takes_YSU_s_OWN_face_path(bits):
    """The ``chm`` pre-multiplication contract, and ONLY that.

    WHAT THIS DOES NOT PROVE, corrected after review pointed it out.
    Since YSU now delegates to the same helper, this compares the
    delegating path against the delegate -- the same code reached two
    ways. It cannot show the extracted block still matches what was there
    before the extraction, which was the actual risk in moving a
    correction that was MEASURED (``pbl_tendencies/rv``, the last of 158
    carriers still differing between 1 GPU and 4).

    What it does pin is real and worth having: the caller applies the dry
    mass factor BEFORE the helper, and the helper applies the map factor
    after. Swap that order and this fails.

    The extraction's own inertness is a RUN-LEVEL result, because no unit
    test in this file can reach it: ``nt_ysu_pre`` against ``nt_ysu_post``,
    20 of 20 wrfout files byte-identical across the refactor
    (docs/ntiedtke/PHASE2-BASELINES.md).
    """
    cp = bits
    state, cfg = _state_and_cfg()
    from woof.core.physics import (_couple_momentum_to_faces,
                                    couple_column_tendencies)

    nz, ny, nx = state.p.shape
    zero = cp.zeros((nz, ny, nx), dtype=cp.float32)
    du = cp.arange(nz * ny * nx, dtype=cp.float32).reshape(nz, ny, nx)
    dv = du[::-1].copy()

    out = couple_column_tendencies(state, cfg, rtheta=zero, rqv=zero,
                                   rqc=zero, ru=du, rv=dv)
    chm = (state.c1h[:, None, None] * state.total_mu()[None]
           + state.c2h[:, None, None])
    ru, rv = _couple_momentum_to_faces(state, cfg, chm * du, chm * dv)
    assert bool(cp.all(out.ru.view(cp.uint32)
                       == cp.ascontiguousarray(ru).view(cp.uint32)))
    assert bool(cp.all(out.rv.view(cp.uint32)
                       == cp.ascontiguousarray(rv).view(cp.uint32)))


def test_ysu_still_routes_through_the_shared_helper():
    """The refactor's own gate: YSU must not have grown a private copy."""
    import inspect

    from woof.core import physics

    src = inspect.getsource(physics.couple_ysu_tendencies)
    assert "_couple_momentum_to_faces" in src, (
        "couple_ysu_tendencies no longer delegates; if the momentum block "
        "was inlined again there are two copies of a measured fix")
    assert "cp.roll(mass_u" not in src, (
        "the face interpolation is back inside couple_ysu_tendencies")


# ---------------------------------------------------------------------------
# The arm the original gate did not have
# ---------------------------------------------------------------------------


def test_the_no_hold_branch_passes_momentum_to_the_coupling():
    """The last hop, which was missing until 2026-08-30.

    `CumulusResult` carried `rucuten`/`rvcuten`, `__post_init__` validated
    them as a pair, and the kernel graded them at ``max_ulp == 0``, and
    the no-hold branch then called ``couple_column_tendencies`` without
    ``ru=``/``rv=``, so every value was dropped at the coupling.

    **NONE of those three checks could see it**, because each inspects a
    value that is produced.  What found it was an ABLATION: a build with
    the momentum tendencies withheld produced output bit-identical to one
    with them present, 141 of 141 d02 frames and ``track.csv``.  A term
    that changes nothing when removed was never being applied.

    THE ORIGINAL GATE WAS VALID AND MEANINGLESS.  It asserted that adding
    the momentum slots left GF, KF and YSU byte-identical, 33 of 33
    files, which could not have failed with nothing connected.  "Changes
    nothing" passes trivially when the thing is never invoked, and that
    is this tree's own vacuity lesson committed in code rather than in a
    regex.

    So this asserts the call passes them, structurally, at the site.
    """
    import inspect
    import re

    from woof.core.physics import PhysicsDriver

    src = inspect.getsource(PhysicsDriver._run_cumulus)
    # The no-hold branch is the one guarded by `nca_seconds is None`.
    marker = "if result.nca_seconds is None:"
    assert marker in src, "the no-hold branch has moved or been renamed"
    branch = src[src.index(marker):]
    call = re.search(r"couple_column_tendencies\(", branch)
    assert call is not None, (
        "the no-hold branch no longer calls couple_column_tendencies")
    # The whole argument list, to its matching parenthesis: an argument may
    # itself be parenthesised (rqc=(None if ... else rqc), b9ae790fe8), so
    # the first ")" is not the end of the call.
    depth, end = 1, call.end()
    while depth:
        depth += {"(": 1, ")": -1}.get(branch[end], 0)
        end += 1
    args = branch[call.end():end - 1]
    assert "ru=" in args and "rv=" in args, (
        "the no-hold branch calls couple_column_tendencies WITHOUT ru/rv "
        "again. New Tiedtke's convective momentum is computed, validated "
        "and then dropped at the coupling, and no test that inspects the "
        "value can see it -- only an ablation can.")


def test_the_held_branch_refuses_momentum_rather_than_dropping_it():
    """Fail closed where storage does not exist; persist it where it does.

    The held (NCA) path rebuilds from ``self.cu_rates``.  Since the MSKF
    admission (854ee8c14d) a held scheme that carries momentum declares
    ``rucuten``/``rvcuten`` storage there, and the held coupling passes
    those slots as ``ru=``/``rv=``.  A held result carrying momentum
    WITHOUT declared storage would lose it exactly the way the no-hold
    branch once did, so that case still refuses.

    The negative control is that the refusal names the reason: a bare
    ``raise`` would send the next reader looking at the scheme rather than
    at ``cu_rates``.
    """
    import inspect

    from woof.core.physics import PhysicsDriver

    src = inspect.getsource(PhysicsDriver._run_cumulus)
    assert "held cumulus momentum needs declared persistent rate storage" in src, (
        "the held path no longer refuses momentum it cannot persist")
    held = src[src.index("if result.pratec is None:"):]
    assert 'ru=self.cu_rates.get("rucuten")' in held and \
        'rv=self.cu_rates.get("rvcuten")' in held, (
            "the held coupling no longer passes the persisted momentum slots")


def test_momentum_reaches_the_state_not_merely_the_tendency_object():
    """`add_to_slow` is where the value becomes a forecast difference.

    Grading `cududvn` proves the kernel; passing `ru=`/`rv=` proves the
    coupling; neither proves the coupled tendency is added to the state.
    This walks the last step so all three links are asserted rather than
    two of three.
    """
    import inspect

    from woof.core.physics import PhysicsTendencies

    src = inspect.getsource(PhysicsTendencies.add_to_slow)
    assert "state.ru_t" in src and "state.rv_t" in src, (
        "add_to_slow no longer adds the momentum tendencies to the state, "
        "so a correctly coupled value would still never reach the forecast")
