"""A frame and a final digest for a domain that was never resident -- CPU only.

Run with ``pytest tilestream/test_store_direct_frame.py`` or directly with
``python tilestream/test_store_direct_frame.py``.  No GPU: everything here is
numpy, dicts and hashlib, which is the point -- the two seams under test are
exactly the two places a store-direct forecast used to need a domain-shaped
``DomainState`` it cannot have.

WHAT THE SLAB CAN AND CANNOT ANSWER
-----------------------------------
``woof.ingest.prepared_store`` fills the store one ROW SLAB at a time and
keeps the last slab as a template, so the tallest state a store-direct run
ever builds is ``rows`` deep.  ``tilestream.output.frame_plan`` measured
against that slab gets the CLASSIFICATION right (which scheme published which
field, and whether it is a carrier, a setup array, a host derive or an
output-only diagnostic -- none of which varies with height) and the SHAPES
wrong (it records ``np.shape`` of the arrays it matched, and those are the
slab's).  :func:`negative_slab_plan_is_not_a_domain_plan` is that failure,
measured: the frame constructs without complaint and then dies on the first
derive, and its structural-zero rows would have gone to disk as ``(rows, nx)``
planes.

:func:`tilestream.output.domain_frame_plan` is the fix and
:func:`test_domain_frame_plan_reproduces_the_domains_own_plan` is the gate on
it: the plan re-sized off a slab must equal, field for field, the plan a
domain-shaped state would have produced, and the two frames built from them
must be byte-identical.  That is the CPU half of the real-data parity gate,
which compares whole wrfout files for a domain small enough to run both ways.

AND THE DIGEST
--------------
``woof.state_digest.canonical_state_digest`` walks a resident state.  Pointed
at the state a streamed run holds it does not fail -- it returns a complete,
self-consistent digest OF THE ANALYSIS, because that state stopped moving when
the store was filled.  ``canonical_store_digest`` is the same document taken
off the store, and :func:`test_store_digest_equals_the_resident_digest` is the
equality: same sha256, same inventory sha256, same field order, over a
manifest assembled two different ways from the same arrays.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woof.state_serialization_contract import (  # noqa: E402
    STATE_SETUP_ARRAYS, STATE_SETUP_SCALARS,
)
from woof.state_digest import (  # noqa: E402
    canonical_state_digest, canonical_store_digest,
)
from tilestream import output as tsout  # noqa: E402
from tilestream.checkpoint import DomainSetup  # noqa: E402


NZ, NY, NX = 4, 12, 9
#: Rows in the last slab the store was built through.  Deliberately not a
#: divisor of NY and deliberately smaller than NX, so a plan that got the two
#: horizontal axes the wrong way round cannot pass by coincidence.
ROWS = 5


# ---------------------------------------------------------------------------
# a domain, its store, and the plan a domain-shaped state would have produced
# ---------------------------------------------------------------------------

class _Cfg:
    """The three extents ``StoreFrame`` and ``domain_frame_plan`` read."""

    def __init__(self, nz=NZ, ny=NY, nx=NX):
        self.nz, self.ny, self.nx = int(nz), int(ny), int(nx)
        self.mp_physics = 6  # WSM6 carrier inventory used by these fixtures.


def _ramp(shape, seed):
    """A distinct, exactly-representable value in every cell.

    Whole numbers scaled by 1/8: every one is exact in float32, so a value
    that comes back changed came back through a different code path rather
    than through rounding.
    """
    size = int(np.prod(shape)) if shape else 1
    values = ((np.arange(size, dtype=np.float32) + float(seed))
              / np.float32(8.0))
    return np.ascontiguousarray(values.reshape(shape))


def domain_store() -> dict:
    """The full-domain carrier store, keyed as the restart archive keys it."""
    return {
        "state/thp": _ramp((NZ, NY, NX), 1),
        "state/p": _ramp((NZ, NY, NX), 2),
        "state/php": _ramp((NZ + 1, NY, NX), 3),
        "state/u": _ramp((NZ, NY, NX + 1), 4),
        "state/v": _ramp((NZ, NY + 1, NX), 5),
        "state/w": _ramp((NZ + 1, NY, NX), 6),
        "state/mup": _ramp((NY, NX), 7),
        "state/qv": _ramp((NZ, NY, NX), 8),
    }


def domain_setup(cfg) -> DomainSetup:
    """The production ``checkpoint.DomainSetup``, at DOMAIN shape.

    Built through the real constructor rather than a stand-in so the frame's
    setup rows come out of the shipped ``output_statics`` -- the ``PB``/``PHB``
    column broadcasts included, which is where a base state that was captured
    off a slab would show up as one band of terrain repeated over the map.
    """
    arrays = {name: _ramp((cfg.nz,), 20) for name in STATE_SETUP_ARRAYS}
    arrays["thb"] = _ramp((cfg.nz,), 300 * 8)          # ~300 K, exactly
    arrays["pb"] = _ramp((cfg.nz,), 30)
    arrays["phb"] = _ramp((cfg.nz + 1,), 40)
    arrays["mub2d"] = _ramp((cfg.ny, cfg.nx), 50)
    arrays["ht"] = _ramp((cfg.ny, cfg.nx), 60)
    arrays["znu"] = _ramp((cfg.nz,), 70)
    arrays["znw"] = _ramp((cfg.nz + 1,), 80)
    scalars = {name: 0.0 for name in STATE_SETUP_SCALARS}
    scalars["p_top"] = np.float32(5000.0)
    return DomainSetup(cfg, arrays, scalars, None, {})


#: One frame's worth of rows, with the class and the origin a live
#: ``frame_plan`` measures and the shape a DOMAIN-shaped state would record.
#: Every class ``StoreFrame`` treats differently is represented, including the
#: three whose shape it allocates from (derived, structural-zero, and the
#: ``overlap=True`` carrier snapshots) and the two whose shape it never reads
#: (setup rows and carriers).
_ROWS = (
    ("T", tsout.SOURCE_DERIVED, "(thb + thp) - 300", (NZ, NY, NX)),
    ("U", tsout.SOURCE_CARRIER, "state/u", (NZ, NY, NX + 1)),
    ("V", tsout.SOURCE_CARRIER, "state/v", (NZ, NY + 1, NX)),
    ("W", tsout.SOURCE_CARRIER, "state/w", (NZ + 1, NY, NX)),
    ("PH", tsout.SOURCE_CARRIER, "state/php", (NZ + 1, NY, NX)),
    ("MU", tsout.SOURCE_CARRIER, "state/mup", (NY, NX)),
    ("PHB", tsout.SOURCE_DERIVED, "broadcast(phb)", (NZ + 1, NY, NX)),
    ("MUB", tsout.SOURCE_SETUP, "mub2d", (NY, NX)),
    ("HGT", tsout.SOURCE_SETUP, "ht", (NY, NX)),
    ("P", tsout.SOURCE_DERIVED, "p - pb", (NZ, NY, NX)),
    ("PB", tsout.SOURCE_DERIVED, "broadcast(pb)", (NZ, NY, NX)),
    ("PSFC", tsout.SOURCE_DERIVED,
     "z-extrapolated p (physics-free states only)", (NY, NX)),
    ("QVAPOR", tsout.SOURCE_CARRIER, "state/qv", (NZ, NY, NX)),
    ("RAINNC", tsout.SOURCE_ZERO, "zeros", (NY, NX)),
    ("RAINC", tsout.SOURCE_ZERO, "zeros", (NY, NX)),
    ("ZNU", tsout.SOURCE_SETUP, "znu", (NZ,)),
    ("ZNW", tsout.SOURCE_SETUP, "znw", (NZ + 1,)),
    ("P_TOP", tsout.SOURCE_DERIVED, "setup scalar", ()),
    ("OLR", tsout.SOURCE_DRIVER_DIAGNOSTIC, "driver.olr", (NY, NX)),
)


def _plan(shapes) -> tsout.FramePlan:
    return tsout.FramePlan(
        order=tuple(name for name, _, _, _ in _ROWS),
        source={name: kind for name, kind, _, _ in _ROWS},
        origin={name: origin for name, _, origin, _ in _ROWS},
        shape=dict(shapes),
        dtype={name: np.dtype(np.float32) for name, _, _, _ in _ROWS})


def domain_plan() -> tsout.FramePlan:
    """What ``frame_plan`` returns when the state IS the domain."""
    return _plan({name: shape for name, _, _, shape in _ROWS})


#: The store a route that PUBLISHES must hand ``StoreFrame``: the carriers
#: plus the driver diagnostics the run arranged to scatter.  ``_ROWS`` keeps
#: ``OLR`` in the unavailable driver-diagnostic class on purpose -- that is
#: the class the plan-level tests below exercise -- but a frame built over a
#: plan with an unavailable row is a SHORT frame, and ``StoreFrame`` refuses
#: one by default (``require_complete``; tilestream/output.py, the fix that
#: measured 73 of 74 variables published at the 2.2.0 cut with no error
#: anywhere).  These two helpers apply the exact transformation
#: ``store_frame_plan(extra_available=...)`` applies when the run does carry
#: the diagnostic: the row becomes a CARRIER served by its store key.  The
#: frame-level tests use them, so what they compare is a frame a route is
#: allowed to write rather than one the law would stop.
def carried_store() -> dict:
    store = domain_store()
    store["diag/olr"] = _ramp((NY, NX), 9)
    return store


def carried(plan: tsout.FramePlan) -> tsout.FramePlan:
    source = dict(plan.source)
    origin = dict(plan.origin)
    for name in plan.names(tsout.SOURCE_DRIVER_DIAGNOSTIC):
        source[name], origin[name] = tsout.SOURCE_CARRIER, "diag/olr"
    return tsout.FramePlan(order=plan.order, source=source, origin=origin,
                           shape=dict(plan.shape), dtype=dict(plan.dtype))


def slab_plan(rows=ROWS) -> tsout.FramePlan:
    """What ``frame_plan`` returns on the slab the store was built through.

    Same order, same classes, same origins, same dtypes -- and every
    horizontal extent replaced by the slab's, mass rows and the closing
    staggered face alike.  That is exactly what ``frame_plan`` records,
    because it takes ``np.shape`` of the arrays it matched.
    """
    shrunk = {}
    for name, _, _, shape in _ROWS:
        axes = list(shape)
        if len(axes) >= 2:
            axes[-2] = rows + (axes[-2] - NY)
        shrunk[name] = tuple(axes)
    return _plan(shrunk)


# ---------------------------------------------------------------------------
# (A) the plan
# ---------------------------------------------------------------------------

def test_domain_frame_plan_reproduces_the_domains_own_plan():
    """The gate: a slab-measured plan re-sized == the domain's own plan."""
    cfg = _Cfg()
    got = tsout.domain_frame_plan(slab_plan(), domain_store(), cfg,
                                  template_shape=(ROWS, NX))
    want = domain_plan()
    assert got.order == want.order
    assert got.source == want.source
    assert got.origin == want.origin
    assert got.dtype == want.dtype
    assert got.shape == want.shape


def test_a_domain_shaped_template_is_a_fixed_point():
    """The single-slab case: nothing to re-size, nothing changed."""
    cfg = _Cfg()
    want = domain_plan()
    got = tsout.domain_frame_plan(want, domain_store(), cfg,
                                  template_shape=(NY, NX))
    assert got.shape == want.shape


def test_carrier_shapes_come_from_the_store_not_from_the_rule():
    """A carrier is read, never mapped -- so an odd layout still publishes.

    The mapping rule ("the last two axes are horizontal") is an inference and
    the store is a fact.  A carrier whose horizontal extents the rule could
    not name must still come back as the array the frame will actually hand
    the writer, because that array is what goes on disk.
    """
    cfg = _Cfg()
    store = domain_store()
    store["state/mup"] = _ramp((NY, NX, 3), 9)     # a trailing band axis
    plan = tsout.domain_frame_plan(slab_plan(), store, cfg,
                                   template_shape=(ROWS, NX))
    assert plan.shape["MU"] == (NY, NX, 3)


def test_an_unmappable_materialised_row_is_refused():
    """A derived/zero/setup row of unrecognisable horizontal shape must fail.

    Those are the three classes ``StoreFrame`` ALLOCATES from, so a guess
    there is a wrongly-sized variable in the file rather than an exception --
    which is the class of failure this whole seam exists to remove.
    """
    cfg = _Cfg()
    bad = slab_plan()
    bad.shape["RAINNC"] = (ROWS + 4, NX)
    with pytest.raises(ValueError, match="RAINNC"):
        tsout.domain_frame_plan(bad, domain_store(), cfg,
                                template_shape=(ROWS, NX))


def test_an_unmappable_diagnostic_row_keeps_its_shape_instead_of_refusing():
    """A driver diagnostic is never built, so it cannot be mis-sized.

    ``fields()`` skips the class and the constructor allocates nothing for
    it; the row exists in the plan so ``FramePlan.unavailable`` can name it.
    Refusing a whole forecast's history over the recorded extent of a field
    that was never going to be published would be a refusal with no failure
    behind it.
    """
    cfg = _Cfg()
    odd = slab_plan()
    odd.shape["OLR"] = (ROWS + 4, NX)
    plan = tsout.domain_frame_plan(odd, domain_store(), cfg,
                                   template_shape=(ROWS, NX))
    assert plan.shape["OLR"] == (ROWS + 4, NX)
    assert plan.unavailable == ("OLR",)


def test_the_two_horizontal_axes_are_never_crossed_over():
    """Each axis resolves against its OWN template extent.

    A square template over a non-square domain, and the ``rows + 1 == nx``
    coincidence, are the two ways a single shared rule would swap y and x.
    """
    assert tsout._domain_row_shape((NZ, 6, 6), (6, 6), (12, 9)) == (NZ, 12, 9)
    assert tsout._domain_row_shape((NZ, 7, 6), (6, 6), (12, 9)) == (NZ, 13, 9)
    assert tsout._domain_row_shape((NZ, 6, 7), (6, 6), (12, 9)) == (NZ, 12, 10)
    # rows + 1 == nx on the template: a mass row and a staggered column have
    # the same extent, and they must still land on different domain answers.
    assert tsout._domain_row_shape((5, 6), (5, 6), (12, 9)) == (12, 9)
    assert tsout._domain_row_shape((6, 6), (5, 6), (12, 9)) == (13, 9)
    # Columns and scalars pass through untouched.
    assert tsout._domain_row_shape((NZ,), (ROWS, NX), (NY, NX)) == (NZ,)
    assert tsout._domain_row_shape((), (ROWS, NX), (NY, NX)) == ()


def test_store_frame_plan_measures_the_template_and_re_sizes_it(monkeypatch):
    """The glue: the template's extents come off ``template.mup``.

    Which is not a convention invented here -- ``wrfout._device_state_frame``
    opens with ``ny, nx = state.mup.shape`` and sizes every field it builds
    from those two numbers, so they are the extents the recorded shapes are
    in.  ``frame_plan`` itself needs a live device state, so it is the one
    thing stubbed; everything downstream of it is the shipped code.
    """
    cfg = _Cfg()
    seen = {}

    def fake_frame_plan(state, *, include_diagnostic_pressure=True,
                        extra_available=()):
        seen["state"] = state
        seen["extra"] = tuple(extra_available)
        return slab_plan()

    monkeypatch.setattr(tsout, "frame_plan", fake_frame_plan)

    class _Template:
        mup = np.zeros((ROWS, NX), np.float32)

    template = _Template()
    store = domain_store()
    plan = tsout.store_frame_plan(template, store, cfg,
                                  extra_available=store.keys())
    assert seen["state"] is template
    assert seen["extra"] == tuple(store.keys())
    assert plan.shape == domain_plan().shape


# ---------------------------------------------------------------------------
# (A) the frame built from it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("overlap", [False, True])
def test_the_re_sized_plan_publishes_a_byte_identical_frame(overlap):
    """The whole point: same rows, same order, same bytes.

    Compared row by row AND as a sequence, because the frame dict doubles as
    the writer's schema and HDF5 lays its name heap out in variable-creation
    order -- ``test_io.negative_field_order`` measured a file 189 bytes larger
    in which every variable compared equal.
    """
    cfg = _Cfg()
    store = carried_store()
    setup = domain_setup(cfg)
    resident = tsout.StoreFrame(carried(domain_plan()), store, setup, cfg,
                                overlap=overlap).fields()
    direct = tsout.StoreFrame(
        carried(tsout.domain_frame_plan(slab_plan(), store, cfg,
                                        template_shape=(ROWS, NX))),
        store, setup, cfg, overlap=overlap).fields()
    assert list(direct) == list(resident)
    for name in resident:
        want, got = np.asarray(resident[name]), np.asarray(direct[name])
        assert got.shape == want.shape, name
        assert got.dtype == want.dtype, name
        assert got.tobytes(order="C") == want.tobytes(order="C"), name


def test_the_frame_is_at_domain_extents_and_the_carriers_are_views():
    """Sanity on the frame itself, so the equality above is not two wrongs."""
    cfg = _Cfg()
    store = carried_store()
    frame = tsout.StoreFrame(
        carried(tsout.domain_frame_plan(slab_plan(), store, cfg,
                                        template_shape=(ROWS, NX))),
        store, domain_setup(cfg), cfg).fields()
    assert frame["T"].shape == (NZ, NY, NX)
    assert frame["RAINNC"].shape == (NY, NX)
    assert frame["MUB"].shape == (NY, NX)
    assert frame["PHB"].shape == (NZ + 1, NY, NX)
    # zero-copy: the carrier row IS the store's pinned page, which is what
    # makes an out-of-core frame cheaper than the resident one.
    assert frame["QVAPOR"] is store["state/qv"]


def test_moist_theta_history_downloads_vapour_and_keeps_thm_prognostic():
    """The real two-rank run wrote THM words into T despite exact dynamics."""
    cfg = _Cfg()
    cfg.use_theta_m = 1
    store = carried_store()
    store["state/qv"].fill(np.float32(0.01))
    plan = carried(domain_plan())
    # A thermal-only output must still download qv for its dry conversion.
    plan.order = ("T", "THM")
    plan.source = {name: tsout.SOURCE_DERIVED for name in plan.order}
    plan.origin = {name: "thermal" for name in plan.order}
    plan.shape = {name: (NZ, NY, NX) for name in plan.order}
    plan.dtype = {name: np.dtype("float32") for name in plan.order}
    setup = domain_setup(cfg)
    writer = tsout.StoreFrame(plan, store, setup, cfg)
    assert writer.store_keys() == ("state/thp", "state/qv")
    frame = writer.fields()
    moist = setup.thb[:, None, None] + store["state/thp"]
    np.testing.assert_array_equal(frame["THM"], moist - np.float32(300))
    # Recover the prognostic from the emitted dry field within its FP32
    # rounding; T cannot be the same array/value as THM at nonzero vapour.
    from woof.core.constants import RVOVRD
    reconstructed = (frame["T"] + np.float32(300)) * (
        np.float32(1) + np.float32(RVOVRD) * store["state/qv"])
    np.testing.assert_allclose(reconstructed, moist, rtol=2e-7, atol=0)
    assert np.all(frame["T"] < frame["THM"])


def test_the_store_direct_road_may_not_publish_a_short_frame():
    """The two laws, held together, on the road this lane added.

    The store-direct road builds its frame from a plan measured on a SLAB
    and re-sized to the domain.  Re-sizing preserves the source classes, so
    a run that did not arrange to scatter its driver diagnostics arrives at
    ``StoreFrame`` with an unavailable row -- and the frame it would publish
    is missing a variable the resident run of the same configuration writes,
    which is exactly how OLR went absent from every streamed run up to
    2.2.0 with forecast validity PASS and health status_bits 0.  The road
    must meet that refusal rather than route around it; the remedy is one
    call (carry the diagnostic), which is what ``carried`` does above and
    what every frame-level test here goes through.
    """
    cfg = _Cfg()
    store = carried_store()
    plan = tsout.domain_frame_plan(slab_plan(), store, cfg,
                                   template_shape=(ROWS, NX))
    assert plan.unavailable == ("OLR",)
    with pytest.raises(tsout.ShortFrameRefused) as refusal:
        tsout.StoreFrame(plan, store, domain_setup(cfg), cfg)
    text = str(refusal.value)
    assert "OLR" in text and "driver.olr" in text
    # And the remedy is reachable from the message, not just describable.
    assert "inventory_fn" in text
    assert carried(plan).unavailable == ()
    tsout.StoreFrame(carried(plan), store, domain_setup(cfg), cfg).fields()


def negative_slab_plan_is_not_a_domain_plan() -> dict:
    """CONTROL: the slab-height plan used as-is.  MUST fail.

    Returned as a record rather than asserted inline so the failure is
    reported as data, in the register the rest of this lane's controls use.
    """
    cfg = _Cfg()
    store = carried_store()
    frame = tsout.StoreFrame(carried(slab_plan()), store, domain_setup(cfg),
                             cfg)
    record = {"zero_row_shape": tuple(frame._dst["RAINNC"].shape),
              "domain_zero_row_shape": (NY, NX),
              "derive_raised": None}
    try:
        frame.fields()
    except ValueError as exc:
        record["derive_raised"] = type(exc).__name__
    record["fires"] = (record["derive_raised"] == "ValueError"
                       and record["zero_row_shape"] != (NY, NX))
    return record


def test_negative_slab_plan_is_not_a_domain_plan():
    record = negative_slab_plan_is_not_a_domain_plan()
    assert record["fires"], record


# ---------------------------------------------------------------------------
# (B) the digest
# ---------------------------------------------------------------------------

class _Clock:
    dtbc_fp32 = 12.0


class _StandInState:
    """A ``DomainState`` as far as the resident digest walk is concerned.

    ``state_manifest`` walks ``vars(state)`` and classifies EVERY instance
    attribute, so the serialized carriers go on the instance and the three
    infrastructure attributes stay on the class where ``vars()`` cannot see
    them (``restart.classify_state_attr`` files all three ``infra``, so they
    would be legal either way; keeping them off the instance is what keeps
    this stand-in a manifest of exactly the arrays under test).
    """

    physics = None
    elapsed_seconds = 3600.0
    # The streamed driver walk sizes its owners from the pressure field's
    # shape (carrier_manifest, 86c86ab7875); on the class, so vars() stays
    # exactly the arrays under test.
    p = np.empty((NZ, NY, NX), np.float32)

    def __init__(self, arrays, scratch):
        self._scratch = dict(scratch)
        for name, value in arrays.items():
            setattr(self, name, value)


def _digest_pair():
    """The same domain, once as a resident state and once as a store.

    The scratch pool spans all three classes on purpose: ``mp_rainnc`` and
    ``up_heli_max`` SERIALIZE (so the resident ``_scratch_manifest`` takes
    them), ``refl_10cm`` is REBUILT but is one of the audited lazy classes (so
    ``_canonical_extra_manifest`` takes it), and ``uh_spawn_window`` is the
    CARRY class -- cross-step state a sweep must move and a checkpoint must
    not hold, present in the store and absent from both digests.
    """
    arrays = {"u": _ramp((NZ, NY, NX + 1), 4),
              "v": _ramp((NZ, NY + 1, NX), 5),
              "thp": _ramp((NZ, NY, NX), 1),
              "mup": _ramp((NY, NX), 7),
              "qv": _ramp((NZ, NY, NX), 8)}
    scratch = {"mp_rainnc": _ramp((NY, NX), 11),
               "up_heli_max": _ramp((NY, NX), 12),
               "refl_10cm": _ramp((NZ, NY, NX), 13),
               "uh_spawn_window": _ramp((NY, NX), 14)}
    state = _StandInState(arrays, scratch)
    store = {f"state/{name}": value for name, value in arrays.items()}
    store.update({f"scratch/{slot}": value
                  for slot, value in scratch.items()})
    scalars = {"elapsed_seconds": state.elapsed_seconds}
    return state, store, scalars


@pytest.mark.parametrize("scope", ["trajectory", "full"])
def test_store_digest_equals_the_resident_digest(scope):
    state, store, scalars = _digest_pair()
    resident = canonical_state_digest(state, _Clock(), scope=scope)
    direct = canonical_store_digest(store, scalars, _Clock(), scope=scope)
    assert direct == resident
    assert direct["sha256"] == resident["sha256"]
    assert direct["field_order"] == resident["field_order"]


class _StandInDriver:
    """The three counters ``canonical_state_digest`` reads off the driver.

    Its ARRAYS come from ``restart._driver_manifest``, which is the one
    walk both roads share by name and neither road can be handed here (a
    real ``PhysicsDriver`` needs a device); the tests below substitute
    that call and leave everything else on the production path.
    """

    call_counts = {"radiation": 3, "microphysics": 7}
    ysu_nan_guard_fires = 0
    microphysics_updates = 7


def _driver_scalars(state):
    """``carrier_scalars``' shape for a domain that is running physics."""
    return {"elapsed_seconds": state.elapsed_seconds,
            "call_counts": dict(_StandInDriver.call_counts),
            "ysu_nan_guard_fires": _StandInDriver.ysu_nan_guard_fires,
            "microphysics_updates": _StandInDriver.microphysics_updates}


def test_the_restart_only_driver_slot_is_in_neither_digest(monkeypatch):
    """A member the store road CANNOT produce must not move the document.

    ``restart.RESTART_ONLY_DRIVER_SLOTS`` -- today legacy RRTMG's
    ``radiation/o33d_grid`` -- is emitted by ``restart._driver_manifest``,
    which the resident road calls directly, and is POPPED by
    ``physics_inventory.carrier_manifest``, which is the only way the store
    road can be built.  So for ``ra_rrtmg_variant = "rrtmg_legacy"`` -- what
    several shipped physics profiles select, with ``o3input`` defaulting to
    a value that fills the field on every radiation call -- the resident
    digest carried one member the store digest structurally could not, and
    that moves ``array_count``, ``field_order``, ``inventory_sha256`` and
    ``sha256``: every field of the document.  Two runs of the same case in
    the two modes published different ``final_state_digest`` values for
    bit-identical weather, under a module whose headline says the two "must
    agree BIT FOR BIT".

    The fixture above carries no legacy-RRTMG adapter, which is exactly why
    the equality test above did not see this.

    RED before the fix: ``legacy`` differs from both ``plain`` and
    ``direct`` on every field.
    """
    from woof.io import restart as restart_io

    slot, = sorted(restart_io.RESTART_ONLY_DRIVER_SLOTS)
    state, store, _ = _digest_pair()
    state.physics = _StandInDriver()
    scalars = _driver_scalars(state)
    assert slot not in store, (
        "carrier_manifest pops this slot, so no store can hold it -- if one "
        "can, the asymmetry is on the other side and this test is wrong")

    monkeypatch.setattr(restart_io, "_driver_manifest", lambda driver: {})
    plain = canonical_state_digest(state, _Clock())

    # The same domain, same weather, with the legacy adapter's o33d grid.
    monkeypatch.setattr(restart_io, "_driver_manifest",
                        lambda driver: {slot: _ramp((NZ, NY, NX), 21)})
    legacy = canonical_state_digest(state, _Clock())

    assert slot not in legacy["field_order"]
    assert legacy == plain
    assert legacy == canonical_store_digest(store, scalars, _Clock())


def test_active_cam_owner_is_hashed_once_on_both_roads(monkeypatch):
    from woof.io import restart as restart_io
    from tilestream.physics_inventory import streaming_inventory

    state, _, _ = _digest_pair()
    # This control is before output staging; the helper's synthetic frame
    # stash is tested separately by the frame-inventory cases above.
    state._scratch.pop("refl_10cm")
    state.physics = _StandInDriver()
    state.physics.o3rad = _ramp((NZ, NY, NX), 21)
    # The driver walker owns this canonical key; carrier_manifest decides
    # whether this explicit owner belongs in the actual store inventory.
    monkeypatch.setattr(restart_io, "_driver_manifest",
                        lambda driver, **_: {"radiation/o33d_grid": driver.o3rad})
    store = {key: value.copy() for key, value in streaming_inventory(state).items()}
    scalars = _driver_scalars(state)
    resident = canonical_state_digest(state, _Clock())
    assert resident == canonical_store_digest(store, scalars, _Clock())
    assert resident["field_order"].count("radiation/o33d_grid") == 1
    state.physics.o3rad[0, 0, 0] += np.float32(1.)
    changed = canonical_state_digest(state, _Clock())
    assert changed["sha256"] != resident["sha256"]
    assert changed != canonical_store_digest(store, scalars, _Clock())
    store["radiation/o33d_grid"][0, 0, 0] += np.float32(1.)
    assert changed == canonical_store_digest(store, scalars, _Clock())


def test_the_carry_class_is_in_the_store_and_in_neither_digest():
    """A tracker window moves between steps and hashes in no digest.

    The store must carry it -- a tile buffer that lost it between sweeps
    would fold its running max over whatever tiles it happened to serve --
    and the digest must not, because ``_scratch_manifest`` keeps only the
    serialize class and the store road has to make the same cut or the two
    digests differ by two ``(ny, nx)`` planes.
    """
    state, store, scalars = _digest_pair()
    assert "scratch/uh_spawn_window" in store
    direct = canonical_store_digest(store, scalars, _Clock())
    assert "scratch/uh_spawn_window" not in direct["field_order"]
    assert "scratch/refl_10cm" in direct["field_order"]
    assert "scratch/up_heli_max" in direct["field_order"]

    moved = dict(store)
    moved["scratch/uh_spawn_window"] = _ramp((NY, NX), 99)
    assert canonical_store_digest(
        moved, scalars, _Clock())["sha256"] == direct["sha256"]


def test_the_digest_moves_when_the_weather_does():
    """The control on the control: an included member must change the hash."""
    state, store, scalars = _digest_pair()
    base = canonical_store_digest(store, scalars, _Clock())
    for key in ("state/thp", "scratch/mp_rainnc", "scratch/refl_10cm"):
        moved = dict(store)
        moved[key] = np.asarray(store[key]) + np.float32(1.0)
        assert canonical_store_digest(
            moved, scalars, _Clock())["sha256"] != base["sha256"], key


def test_the_driver_scalar_block_matches_the_resident_one(monkeypatch):
    """The three counters, read off the store's scalars instead of a driver.

    On the resident road they come off ``state.physics``; on a store-direct
    one that object is a TILE BUFFER's driver, which has stepped once per
    TILE rather than once per domain step, so the numbers have to come from
    the scalar carriers the sweep advances.  ``_driver_manifest`` needs a real
    ``PhysicsDriver`` to walk, so it is stubbed to the members a driver would
    have contributed and the store is given the same arrays under the same
    keys.
    """
    from woof.io import restart as restart_io

    state, store, scalars = _digest_pair()
    driver_members = {"fields/ust": _ramp((NY, NX), 21),
                      "driver/rthratenlw": _ramp((NZ, NY, NX), 22),
                      "cumulus/w0avg": _ramp((NZ, NY, NX), 23)}
    monkeypatch.setattr(restart_io, "_driver_manifest",
                        lambda driver: dict(driver_members))

    class _Driver:
        call_counts = {"radiation": 3, "cumulus": 11}
        ysu_nan_guard_fires = 0
        microphysics_updates = 240

    state.physics = _Driver()
    store.update(driver_members)
    scalars.update(call_counts=dict(_Driver.call_counts),
                   ysu_nan_guard_fires=_Driver.ysu_nan_guard_fires,
                   microphysics_updates=_Driver.microphysics_updates)

    resident = canonical_state_digest(state, _Clock())
    direct = canonical_store_digest(store, scalars, _Clock())
    assert direct == resident
    assert direct["scalars"]["driver"]["microphysics_updates"] == 240


def test_a_physics_free_store_reports_no_driver():
    """``carrier_scalars`` adds the three counters together or not at all.

    Same test ``write_streamed_restart`` makes for the header's ``driver``
    block, so a digest and a checkpoint of the same instant cannot disagree
    about whether physics is running.
    """
    _state, store, scalars = _digest_pair()
    assert canonical_store_digest(
        store, scalars, _Clock())["scalars"]["driver"] is None


def test_a_byte_swapped_carrier_is_refused():
    """The one way a host array differs from a ``.get()`` and compares equal.

    A swapped dtype moves the member descriptor (``'>f4'`` rather than
    ``'float32'``) AND every payload byte, so the store road would mint a
    digest the resident road can never reproduce while holding identical
    numbers.
    """
    _state, store, scalars = _digest_pair()
    swapped = dict(store)
    swapped["state/thp"] = np.asarray(store["state/thp"]).astype(">f4")
    with pytest.raises(ValueError, match="byte order"):
        canonical_store_digest(swapped, scalars, _Clock())


def test_an_unrecognised_lazy_prefix_member_is_refused():
    """Mirrors the resident walk's refusal, on the store's keys.

    A member under an audited lazy prefix that is not a concrete registered
    one is how a new member class arrives silently -- and the resident
    function has refused it since the sixth hand-copied table on this route.
    """
    _state, store, scalars = _digest_pair()
    store = dict(store)
    store["scratch/nest_not_a_real_slot"] = _ramp((NY, NX), 31)
    with pytest.raises(RuntimeError, match="audited lazy prefix"):
        canonical_store_digest(store, scalars, _Clock())


def test_a_store_digest_without_a_clock_scalar_is_refused():
    _state, store, _scalars = _digest_pair()
    with pytest.raises(ValueError, match="elapsed_seconds"):
        canonical_store_digest(store, {}, _Clock())


def test_unknown_scope_is_refused():
    _state, store, scalars = _digest_pair()
    with pytest.raises(ValueError, match="scope"):
        canonical_store_digest(store, scalars, _Clock(), scope="bogus")


# ---------------------------------------------------------------------------
# the seam on StreamedDomain
# ---------------------------------------------------------------------------

class _RunStub:
    """The three attributes ``StreamedDomain`` reads off its ``TiledRun``."""

    def __init__(self, store, cfg):
        self.store = store
        self.cfg = cfg
        from types import SimpleNamespace
        self.tiles = (SimpleNamespace(existing_scratch=lambda *_: None),)

    def drain(self):
        pass


def _streamed_domain(*, store, cfg, scalars=None, state=None, template=None):
    from woof.core import streaming

    decision = streaming.StreamingDecision(True, "test", 4, 4, 2, 4)
    return streaming.StreamedDomain(_RunStub(store, cfg), decision,
                                    state=state, scalars=scalars,
                                    host_store=True, template=template)


def test_a_domain_with_neither_state_nor_template_refuses_a_frame():
    """And says which of the two it wants, because a store cannot supply it.

    Which fields a wrfout frame carries is MEASURED off a live state so a
    scheme publishing a new diagnostic cannot drift a transcribed table; a
    store on its own says only which carriers exist.
    """
    from woof.core import streaming

    domain = _streamed_domain(store=domain_store(), cfg=_Cfg())
    with pytest.raises(streaming.StreamingRefused, match="template"):
        domain.history_fields()


def test_the_domain_digests_its_store():
    _state, store, scalars = _digest_pair()
    domain = _streamed_domain(store=store, cfg=_Cfg(), scalars=scalars)
    assert domain.canonical_digest(_Clock()) == canonical_store_digest(
        store, scalars, _Clock())
    assert domain.canonical_digest(_Clock(), scope="full")["sha256"] == \
        canonical_store_digest(store, scalars, _Clock(),
                               scope="full")["sha256"]


def test_a_carry_nothing_domain_refuses_to_be_digested():
    """``attach(scalars=None)`` is the gate's control, not a forecast.

    Its digest would be stamped with model second zero and no physics call
    counts, which is a receipt claiming the run never happened.
    """
    from woof.core import streaming

    domain = _streamed_domain(store=domain_store(), cfg=_Cfg(), scalars=None)
    with pytest.raises(streaming.StreamingRefused, match="scalars"):
        domain.canonical_digest(_Clock())


def main() -> int:
    return pytest.main([__file__, "-q"])


if __name__ == "__main__":
    raise SystemExit(main())
