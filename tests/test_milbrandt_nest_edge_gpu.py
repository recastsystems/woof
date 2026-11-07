"""GPU gate: the Milbrandt-Yau (mp_physics=9) mixed nest edge.

The claim that admitted mp=9 to ``PORTED_MP_PHYSICS`` (audit R-003) is
that its entry closure is not an invention: MY2 runs its own
mass-to-number consistency block at the top of every call
(module_mp_milbrandt2mom.F:1459-1528), so the numbers a mixed edge has to
diagnose are the numbers the scheme itself would build from the mapped
masses on its first step.  The edge kernel's arm transcribes that block,
and this file MEASURES the claim -- it drives the real edge launcher and
the scheme's own ``milbrandt2_prelim`` over the same masses and compares
the diagnosed numbers.

The two are not bitwise equal for one stated reason and no other: the
scheme forms its density from ``pres = ps * (p/ps)`` (:1216), an FP32
round trip through the surface pressure, while the edge forms it from the
state's ``p`` directly.  Everything else -- the intercepts, the exponents,
Cooper's N(T), Thompson's Nos(T), the ck constants -- is the same
arithmetic on the same words, which is what the tolerance below is sized
for.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.gpu

from woof.core import constants as c                     # noqa: E402
from woof.core import microphysics_transition as mt      # noqa: E402


class _DuckState:
    def __init__(self):
        self._slots = {}

    def scratch(self, shape, slot, dtype=None):
        import cupy as cp
        dtype = cp.float32 if dtype is None else dtype
        buf = self._slots.get(slot)
        if buf is None or buf.shape != tuple(shape) or buf.dtype != dtype:
            buf = cp.zeros(shape, dtype=dtype)
            self._slots[slot] = buf
        return buf


_SHAPE = (8, 4, 3)


def _parent_state(seed=20260910):
    """A Morrison parent: six masses, no hail, ordinary tropospheric air."""
    import cupy as cp

    nz, ny, nx = _SHAPE
    rng = np.random.default_rng(seed)

    def dev(array):
        return cp.asarray(np.ascontiguousarray(array, dtype=np.float32))

    state = _DuckState()
    state.p = dev(np.linspace(9.8e4, 2.5e4, nz)[:, None, None]
                  * np.ones(_SHAPE))
    state.thb = cp.asarray(np.linspace(288.0, 345.0, nz).astype(np.float32))
    state.thp = dev(rng.uniform(-1.5, 1.5, _SHAPE))
    state.alt = dev(np.full(_SHAPE, 1.0))
    state.mub2d = dev(np.full((ny, nx), 9.0e4))
    state.mup = dev(np.zeros((ny, nx)))
    state.c1h = cp.asarray(np.ones((nz,), dtype=np.float32))
    state.c2h = cp.asarray(np.zeros((nz,), dtype=np.float32))
    state.qv = dev(np.full(_SHAPE, 6.0e-3))
    for name in ("qc", "qr", "qi", "qs", "qg"):
        setattr(state, name, dev(rng.uniform(0.0, 2.0e-3, _SHAPE)))
    return state


def _contract():
    parent = SimpleNamespace(mp_physics=10, moist=True, moist_cq=True,
                             morr_rimed_ice=1)
    child = SimpleNamespace(
        mp_physics=9, moist=True, moist_cq=True,
        nest_microphysics_transition=mt.EDGE_MATRIX_POLICY)
    return mt.resolve_microphysics_transition(parent, child)


def _scheme_closure(state, contract):
    """What MY2's own consistency block builds from these masses, in #/kg.

    The masses are the ones the edge MAPS into the child, resolved through
    the contract itself -- a Morrison parent with ``morr_rimed_ice=1``
    calls its single rimed category hail, so the child's qh is the
    parent's qg and the child's qg is zero.  Running the reference on the
    parent's own spelling instead would compare a graupel number against a
    hail number and prove nothing.
    """
    import cupy as cp

    from woof.core.kernels import get_kernel
    from woof.core.milbrandt2 import _constants_device, _surface_pressure

    nz, ny, nx = _SHAPE
    shape = _SHAPE
    theta = state.thb[:, None, None] + state.thp
    if getattr(state, "use_theta_m", 0):
        theta = theta / (np.float32(1.0) + np.float32(c.RVOVRD) * state.qv)
    temperature = cp.ascontiguousarray(
        theta * cp.power(state.p / np.float32(c.P0), np.float32(c.RCP)))
    z8w = cp.asarray(
        np.linspace(0.0, 16.0e3, nz + 1)[:, None, None]
        * np.ones((nz + 1, ny, nx)), dtype=cp.float32)
    zh = cp.ascontiguousarray(0.5 * (z8w[:nz] + z8w[1:]))
    psfc = cp.zeros((ny, nx), dtype=cp.float32)
    _surface_pressure(state.p, zh, z8w[0], psfc)

    fields = {}
    for target in ("qv", "qc", "qr", "qi", "qs", "qg", "qh"):
        source = contract.mass_source(target)
        fields[target] = (
            cp.zeros(shape, dtype=cp.float32) if source is None
            else cp.ascontiguousarray(getattr(state, source).copy()))
    numbers = {name: cp.zeros(shape, dtype=cp.float32)
               for name in ("nc", "nr", "ni", "ns", "ng", "nh")}
    scratch = {name: cp.zeros(shape, dtype=cp.float32)
               for name in ("pres", "de", "ide", "gamfact", "qsw", "qsi",
                            "qc_in", "qr_in", "nc_in", "nr_in")}
    ncell = nz * ny * nx
    threads = 64
    get_kernel("milbrandt2", "milbrandt2_prelim")(
        (((ncell + threads - 1) // threads),), (threads,), (
            cp.ascontiguousarray(temperature),
            fields["qv"], fields["qc"], fields["qr"], fields["qi"],
            fields["qs"], fields["qg"], fields["qh"],
            numbers["nc"], numbers["nr"], numbers["ni"],
            numbers["ns"], numbers["ng"], numbers["nh"],
            cp.ascontiguousarray(state.p), psfc,
            scratch["pres"], scratch["de"], scratch["ide"],
            scratch["gamfact"], scratch["qsw"], scratch["qsi"],
            scratch["qc_in"], scratch["qr_in"],
            scratch["nc_in"], scratch["nr_in"],
            _constants_device(),
            np.int32(nz), np.int32(ny), np.int32(nx)))
    cp.cuda.Stream.null.synchronize()
    # prelim leaves the numbers per unit VOLUME; the state carries per mass.
    return {name: value * scratch["ide"] for name, value in numbers.items()}


@pytest.mark.parametrize("use_theta_m", [0, 1])
def test_the_edge_diagnoses_the_schemes_own_numbers(use_theta_m):
    import cupy as cp

    state = _parent_state()
    state.use_theta_m = use_theta_m
    if use_theta_m:
        state.thp = ((state.thb[:, None, None] + state.thp)
                     * (np.float32(1.0) + np.float32(c.RVOVRD) * state.qv)
                     - state.thb[:, None, None])
    contract = _contract()
    assert contract.mixed is True
    reference = _scheme_closure(state, contract)
    for name in ("nc", "nr", "ni", "ns", "ng", "nh"):
        out = cp.zeros(_SHAPE, dtype=cp.float32)
        mt.launch_microphysics_edge_parent_field(
            contract, state, name, out=out, coupled=False)
        cp.cuda.Stream.null.synchronize()
        got = cp.asnumpy(out)
        want = cp.asnumpy(reference[name])
        assert np.all(np.isfinite(got)), name
        np.testing.assert_allclose(got, want, rtol=2.0e-5, atol=0.0,
                                   err_msg=name)


def test_a_parent_without_hail_hands_the_child_no_hail_mass():
    """Morrison with morr_rimed_ice=1 calls its single rimed category HAIL.

    So it maps to the child's qh and the child's qg is defaulted -- and the
    hail NUMBER follows the mass it was diagnosed from.
    """
    import cupy as cp

    state = _parent_state()
    contract = _contract()
    assert contract.mass_source("qh") == "qg"
    assert contract.mass_source("qg") is None
    out = cp.zeros(_SHAPE, dtype=cp.float32)
    mt.launch_microphysics_edge_parent_field(
        contract, state, "qg", out=out, coupled=False)
    cp.cuda.Stream.null.synchronize()
    assert float(cp.asnumpy(out).max()) == 0.0
    nh = cp.zeros(_SHAPE, dtype=cp.float32)
    mt.launch_microphysics_edge_parent_field(
        contract, state, "nh", out=nh, coupled=False)
    cp.cuda.Stream.null.synchronize()
    graupel = cp.asnumpy(state.qg)
    hail_number = cp.asnumpy(nh)
    assert np.all(hail_number[graupel > 1.0e-14] > 0.0)
    assert np.all(hail_number[graupel <= 1.0e-14] == 0.0)


def test_the_state_the_edge_reads_is_not_modified():
    import cupy as cp

    state = _parent_state()
    contract = _contract()
    before = {name: getattr(state, name).copy()
              for name in ("qv", "qc", "qr", "qi", "qs", "qg", "p", "thp")}
    out = cp.zeros(_SHAPE, dtype=cp.float32)
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh",
                 "nc", "nr", "ni", "ns", "ng", "nh"):
        mt.launch_microphysics_edge_parent_field(
            contract, state, name, out=out, coupled=False)
    cp.cuda.Stream.null.synchronize()
    for name, original in before.items():
        cp.testing.assert_array_equal(getattr(state, name), original)


_DONOR_WINDOW = (slice(1, 4), slice(0, 2))


def test_the_windowed_donor_drives_the_edge_exactly_as_the_resident_parent():
    """The tile-streamed nest route, which is the one that broke.

    ``parent_only_init(window=...)`` -- driven slab by slab from
    ``woof/ingest/reconstruction_store.py`` for a tile-streamed child --
    does not hand the edge launcher a ``DomainState``.  It hands it the
    bounded namespace ``transition_parent_window`` builds, whose planes are
    window-shaped copies and which owns no scratch arena.  Every ported
    target before mp=9 read only the masses and the two mass-coupling
    planes that namespace carried, so the omission was invisible; MY2's arm
    reads the state's theta pair and pressure as well, and a windowed mp=9
    edge died with ``AttributeError: 'types.SimpleNamespace' object has no
    attribute 'thb'`` AFTER plan review had admitted the edge.

    The measurement is equality with the resident parent on the same cells:
    a windowed edge that merely runs proves nothing, because the arm could
    be reading a mis-registered column.
    """
    import cupy as cp

    state = _parent_state()
    contract = _contract()
    donor = mt.transition_parent_window(state, _DONOR_WINDOW)
    jj, ii = _DONOR_WINDOW
    window_shape = (_SHAPE[0], jj.stop - jj.start, ii.stop - ii.start)
    assert donor.thp.shape == window_shape
    assert donor.p.shape == window_shape
    assert donor.thb.shape == (_SHAPE[0],)      # a base profile is borrowed

    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh",
                 "nc", "nr", "ni", "ns", "ng", "nh"):
        resident = cp.zeros(_SHAPE, dtype=cp.float32)
        mt.launch_microphysics_edge_parent_field(
            contract, state, name, out=resident, coupled=False)
        windowed = cp.zeros(window_shape, dtype=cp.float32)
        mt.launch_microphysics_edge_parent_field(
            contract, donor, name, out=windowed, coupled=False)
        cp.cuda.Stream.null.synchronize()
        cp.testing.assert_array_equal(
            windowed, resident[:, jj, ii], err_msg=name)


def test_a_columnar_and_a_full_field_base_theta_agree():
    """``thb`` is (nz,) on most states and mass-shaped on some.

    The kernel selects the index; this holds the two forms equal so the
    selector cannot silently read a column-constant field per cell.
    """
    import cupy as cp

    columnar = _parent_state()
    expanded = _parent_state()
    expanded.thb = cp.ascontiguousarray(
        cp.broadcast_to(columnar.thb[:, None, None], _SHAPE).astype(cp.float32))
    contract = _contract()
    for name in ("nc", "nr", "ni", "ns", "ng", "nh"):
        a = cp.zeros(_SHAPE, dtype=cp.float32)
        b = cp.zeros(_SHAPE, dtype=cp.float32)
        mt.launch_microphysics_edge_parent_field(
            contract, columnar, name, out=a, coupled=False)
        mt.launch_microphysics_edge_parent_field(
            contract, expanded, name, out=b, coupled=False)
        cp.cuda.Stream.null.synchronize()
        cp.testing.assert_array_equal(a, b, err_msg=name)


@pytest.mark.parametrize("missing", ["thb", "thp", "p"])
def test_a_parent_without_the_scheme_temperature_planes_is_refused_by_name(
        missing):
    """Not an AttributeError: the plane is named and so is the way out."""
    import cupy as cp

    state = _parent_state()
    setattr(state, missing, None)
    out = cp.zeros(_SHAPE, dtype=cp.float32)
    with pytest.raises(ValueError, match=f"parent's {missing}"):
        mt.launch_microphysics_edge_parent_field(
            _contract(), state, "nc", out=out, coupled=False)
