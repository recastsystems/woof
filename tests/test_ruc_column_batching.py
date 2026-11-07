"""``LSMRUC``'s ``DO i`` is a batching, and batching it must not move a bit.

``woof.core.ruc.ruc_land_surface_step`` was transcribed the way WRF is
written: a Python ``for i in range(ncolumn)`` calling
:func:`~woof.core.ruc.ruc_surface_temperature_step` once per column with
one-element arrays.  ``sfctmp`` and every leaf under it are already written
over a column axis, so that loop paid ``sfctmp``'s fixed per-call cost once
per column -- 360,000 times for a production nest.  Measured on the reference
box before the conversion: 1.687 ms/column at 48 columns and 1.724 at 24,576.

The prologue, the water and sea-ice arms and the epilogue now run over the
whole column axis, and ``SFCTMP`` is called ONCE for all land columns.  This
file is what says that did not change the answer.

The claims are deliberately configuration-independent, because a recorded
digest is a property of one fixture on one box:

* **partition invariance** -- one call over N columns equals the
  concatenation of calls over any disjoint split of the same columns.  This
  is the strongest available statement that the batch is not coupling
  columns, and it needs no reference implementation to compare against.
* **order independence** -- permuting the columns permutes the answer and
  changes nothing else.
* **the two batchings agree** -- ``ilnb_chain=True`` still dispatches one
  column at a time (the chain is a genuine sequential dependence), so on a
  grid where no column's ``ilnb`` is ever read the chained and batched paths
  must be bitwise equal.  That is a paired host authority, in-process,
  against the same tree.

Each carries a falsification that is shown to fail.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from woof.core.ruc import (RUC_DRIVER_COLUMN_STATE, RUC_DRIVER_PROFILE_STATE,
                            RucLandSurfaceStep, ruc_land_surface_step)
from woof.core.ruc_runtime import (C1SN, C2SN, DEFINED_ILNB, ISNCOVR_OPT,
                                    RucRuntimeParameters)

_PARAMS = RucRuntimeParameters()
_NSOIL = 9
_GRASSLAND = 10
_LOAM = 6

#: Production width, the four-domain case's finest nest: 600x600.
_PRODUCTION_D04_COLUMNS = 600 * 600


def _columns(n: int, *, snow: bool, water: int = 0, ice: int = 0, seed: int = 0,
             nzs: int = _NSOIL):
    """``n`` land columns, each perturbed, so no two share an answer.

    Identical columns would make partition invariance and order independence
    pass for a batch that silently broadcast one column's answer over the
    field, which is the exact defect those two gates exist to catch.

    ``nzs`` defaults to :data:`_NSOIL` and is drawn FIRST, before any scalar,
    so at nine levels every caller here gets the identical draw sequence --
    and therefore the identical columns -- it got before the argument
    existed.  ``tests/test_ruc_nzs_device.py`` imports this to build the
    six-level host/device comparison out of the same generator the
    nine-level one uses, rather than a second fixture that could differ.
    """
    rng = np.random.default_rng(seed)

    def jitter(base, spread):
        return (base + spread * rng.standard_normal(n)).astype(np.float32)

    base = 266.0 if snow else 297.0
    values = {
        "soilmois": np.clip(
            np.stack([jitter(0.28, 0.03) for _ in range(nzs)]), 0.06, 0.44),
        "sh2o": np.clip(
            np.stack([jitter(0.26, 0.03) for _ in range(nzs)]), 0.06, 0.44),
        "tso": np.stack(
            [jitter(base - k, 0.6) for k in range(nzs)]),
        "smfr3d": np.zeros((nzs, n), np.float32),
        "keepfr3dflag": np.zeros((nzs, n), np.float32),
    }
    scalars = {
        "snow": 15.0 if snow else 0.0, "snowh": 0.08 if snow else 0.0,
        "snowc": 1.0 if snow else 0.0, "canwat": 0.4,
        "snoalb": 0.7, "alb": 0.2, "emiss": 0.98, "lai": 2.0,
        "mavail": 0.5, "sfcexc": 0.02, "z0": 0.05, "znt": 0.05,
        "soilt": 265.0 if snow else 303.0, "hfx": 0.0, "qfx": 0.0, "lh": 0.0,
        "sfcevp": 0.0, "sfcrunoff": 0.0, "udrunoff": 0.0, "acrunoff": 0.0,
        "grdflx": 0.0, "acsnow": 0.0, "snom": 0.0, "qvg": 0.011, "qcg": 0.0,
        "dew": 0.0, "qsfc": 0.011, "qsg": 0.011, "chklowq": 1.0,
        "soilt1": 265.0 if snow else 303.0, "tsnav": -6.0,
        "smavail": 0.0, "smmax": 0.0, "rhosnf": 200.0, "precipfr": 0.0,
        "snowfallac": 0.0,
        # z3d is HALF the lowest model layer inside LSMRUC and it is a
        # per-column quantity on a terrain-following coordinate.  Varied on
        # purpose: it used to reach a scalar sfctmp argument, so a batch that
        # took one column's value for the field would pass every gate here
        # with a constant z3d.
        "z3d": 40.0, "p8w": 95000.0, "t3d": 268.0 if snow else 300.0,
        "qv3d": 0.004 if snow else 0.011, "qc3d": 0.0, "rho3d": 1.2,
        "rainbl": 0.6, "frzfrac": 1.0 if snow else 0.0,
        "glw": 280.0 if snow else 340.0, "gsw": 260.0 if snow else 560.0,
        "chs": 0.02, "flqc": 0.02, "flhc": 20.0, "albbck": 0.2,
        "xland": 1.0, "xice": 0.0, "tbot": 288.0,
        "shdmin": 10.0, "shdmax": 80.0, "vegfra": 60.0,
    }
    spread = {"soilt": 1.5, "soilt1": 1.5, "t3d": 1.0, "gsw": 20.0,
              "glw": 5.0, "rainbl": 0.2, "vegfra": 8.0, "mavail": 0.05,
              "canwat": 0.1, "flhc": 2.0, "z3d": 6.0, "p8w": 900.0,
              "snow": 3.0 if snow else 0.0, "snowh": 0.01 if snow else 0.0}
    for name, value in scalars.items():
        values[name] = jitter(value, spread.get(name, 0.0))
    for name in ("snow", "snowh", "canwat", "rainbl", "mavail", "z3d"):
        values[name] = np.maximum(values[name], np.float32(0.0))
    values["snowc"] = np.full(n, scalars["snowc"], np.float32)
    values["xland"] = np.ones(n, np.float32)
    values["xice"] = np.zeros(n, np.float32)
    if water:
        values["xland"][:water] = 2.0
    if ice:
        values["xice"][water:water + ice] = 1.0
    return (values, np.full(n, _GRASSLAND, np.int32),
            np.full(n, _LOAM, np.int32))


def _run(values, ivgtyp, isltyp, *, ktau=2, chain=False, dt=12.0):
    return ruc_land_surface_step(
        values, dt=dt, ktau=ktau, zs=_PARAMS.zs, ivgtyp=ivgtyp,
        isltyp=isltyp, em_core=0, ilnb=DEFINED_ILNB, ilnb_chain=chain, c1sn=C1SN,
        c2sn=C2SN, isncovr_opt=ISNCOVR_OPT,
        mminlu=_PARAMS.dataset_identifier, parameters=_PARAMS.bundle)


_FIELDS = tuple(sorted(RucLandSurfaceStep.__dataclass_fields__))
_CARRIED = tuple(RUC_DRIVER_COLUMN_STATE) + tuple(RUC_DRIVER_PROFILE_STATE)


def _take(result, columns):
    """Every returned field restricted to ``columns``, as raw bytes."""
    out = {}
    for name in _FIELDS:
        array = np.asarray(getattr(result, name))
        if array.ndim == 0:            # the carried ilnb
            continue
        out[name] = np.ascontiguousarray(
            array[..., columns]).tobytes()
    return out


def _differing(left, right) -> list[str]:
    return [name for name in left if left[name] != right[name]]


def _advance(values, result):
    out = dict(values)
    for name in _CARRIED:
        out[name] = np.array(
            getattr(result, name), dtype=np.float32, copy=True)
    return out


# ---------------------------------------------------------------------------
# 1.  the batch does not couple columns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("snow", [False, True], ids=["warm", "snow"])
@pytest.mark.parametrize("water,ice", [(0, 0), (5, 4)],
                         ids=["all-land", "water+ice"])
def test_one_call_equals_any_disjoint_split_of_the_same_columns(
        snow, water, ice) -> None:
    """Partition invariance, which needs no reference implementation.

    ``LSMRUC`` has no horizontal coupling, so the answer for a column cannot
    depend on which other columns share its call.  A batch that reduced,
    broadcast, or indexed across the column axis by mistake would break this
    and could still reproduce a recorded digest at one fixed width.
    """
    n = 24
    values, ivgtyp, isltyp = _columns(n, snow=snow, water=water, ice=ice)
    whole = _run(values, ivgtyp, isltyp)

    for split in ([np.arange(n)],
                  [np.arange(0, 7), np.arange(7, n)],
                  [np.arange(0, 1), np.arange(1, 5), np.arange(5, n)],
                  [np.arange(0, n, 2), np.arange(1, n, 2)]):
        for part in split:
            piece = _run(
                {name: np.asarray(value)[..., part]
                 for name, value in values.items()},
                ivgtyp[part], isltyp[part])
            moved = _differing(_take(piece, slice(None)), _take(whole, part))
            assert moved == [], (
                f"columns {part.tolist()} answered differently in a call of "
                f"{part.size} than in a call of {n}: {moved}.  The batched "
                "driver is coupling columns that LSMRUC does not couple")


@pytest.mark.parametrize("snow", [False, True], ids=["warm", "snow"])
def test_permuting_the_columns_permutes_the_answer_and_nothing_else(
        snow) -> None:
    """Order independence, which the chained ``ilnb`` deliberately lacks."""
    n = 20
    values, ivgtyp, isltyp = _columns(n, snow=snow, water=3, ice=3)
    forward = _run(values, ivgtyp, isltyp)

    order = np.arange(n)[::-1].copy()
    reversed_result = _run(
        {name: np.asarray(value)[..., order] for name, value in values.items()},
        ivgtyp[order], isltyp[order])
    moved = _differing(
        _take(reversed_result, slice(None)), _take(forward, order))
    assert moved == [], (
        f"reversing the column order changed {moved}; the defined "
        "(ilnb_chain=False) path is supposed to be order-independent, which "
        "is the whole reason woof does not reproduce WRF's chained ilnb")


def test_the_partition_gate_sees_a_misrouted_column() -> None:
    """The falsification.

    If the two gates above cannot tell one column's answer from another's,
    they are passing on a fixture rather than on a property.  Route a
    column's inputs to the wrong slot and the comparison must reject it.
    """
    n = 8
    values, ivgtyp, isltyp = _columns(n, snow=False)
    whole = _run(values, ivgtyp, isltyp)

    part = np.arange(0, 4)
    piece = _run(
        {name: np.asarray(value)[..., part] for name, value in values.items()},
        ivgtyp[part], isltyp[part])
    assert _differing(_take(piece, slice(None)), _take(whole, part)) == []

    # ...now compare the same piece against the WRONG four columns.
    wrong = np.arange(4, 8)
    moved = _differing(_take(piece, slice(None)), _take(whole, wrong))
    assert moved, (
        "the comparison could not tell columns 0-3 from columns 4-7, so the "
        "partition and order gates above are not measuring anything")


# ---------------------------------------------------------------------------
# 2.  the two batchings of the same dispatch agree
# ---------------------------------------------------------------------------

def test_the_chained_and_batched_dispatches_agree_where_ilnb_is_never_read():
    """A paired host authority, in-process, over six steps.

    ``ilnb_chain=True`` still runs one column at a time, because column
    ``i``'s seed is column ``i-1``'s answer.  ``ilnb`` is read only at
    ``:4410``/``:5716``, under ``if(snhei.gt.0.)``, and assigned only inside
    ``if(snhei.ge.snth)`` -- so on a grid with no snow at all it is never
    read, both paths must agree bitwise, and the difference between them is
    purely how many columns each ``sfctmp`` call covers.

    Six steps rather than one, feeding the driver's own output back in, so
    the accumulators and the carried soil state are compared too.
    """
    n = 16
    values, ivgtyp, isltyp = _columns(n, snow=False, water=4, ice=3)
    chained, batched = dict(values), dict(values)
    assert np.all(np.asarray(values["snow"]) == 0.0), (
        "this grid must be snow-free for ilnb to be unread; a snowy column "
        "would make the two paths legitimately differ and the comparison "
        "below would be asserting the wrong thing")

    for ktau in range(1, 7):
        left = _run(chained, ivgtyp, isltyp, ktau=ktau, chain=True)
        right = _run(batched, ivgtyp, isltyp, ktau=ktau, chain=False)
        moved = _differing(
            _take(left, slice(None)), _take(right, slice(None)))
        assert moved == [], f"step {ktau}: {moved}"
        chained, batched = _advance(chained, left), _advance(batched, right)


def _nudge(values, field, ulps, toward):
    out = dict(values)
    array = np.array(values[field], dtype=np.float32, copy=True)
    view = array if array.ndim == 1 else array[0]
    for _ in range(ulps):
        view[0] = np.nextafter(view[0], np.float32(toward))
    out[field] = array
    return out


def test_the_paired_gate_sees_the_smallest_thing_each_input_can_move():
    """The falsification for the pairing above, at its measured floor.

    Two paths that both ran the same code would also compare equal, so the
    comparison has to be shown rejecting a difference it should reject.  The
    first attempt used one ULP of the entry ``soilt`` and **failed to fail**:
    ``VILKA``'s solve converges the seed away, exactly as the Noah-MP lane
    measured for ``BARE_FLUX``'s ``TGB``.  Swept per input on a warm
    12-column grid, the floor at which a perturbation first reaches ANY
    returned field is:

    ======== ==== =====================================================
    input    ULP  fields reached
    ======== ==== =====================================================
    soilmois    1 8, led by ``qvg``/``qsfc``/``sh2o``
    t3d         1 1 -- ``hfx`` alone
    soilt       2 10
    z3d         2 6
    tso        64 2 -- ``grdflx`` and ``tso``
    gsw         - absorbed below about 1e-3 W m-2
    ======== ==== =====================================================

    So the gate leads with ``soilmois``, whose floor is one ULP and which
    reaches eight fields, and adds ``z3d`` because that is the input this
    conversion changed the *shape* of: ``0.5*dz8w`` used to reach a SCALAR
    ``sfctmp`` argument, so a batch that took one column's depth for the
    whole field would be invisible to every other check here.

    ``soilt`` at one ULP is asserted to be absorbed, so the ``2`` above
    cannot be quietly tidied back to a ``1`` and the table cannot go stale
    without this failing.
    """
    n = 12
    values, ivgtyp, isltyp = _columns(n, snow=False)
    reference = _take(_run(values, ivgtyp, isltyp), slice(None))

    for field, ulps, toward, least in (("soilmois", 1, 1.0, 4),
                                       ("z3d", 2, 100.0, 3)):
        moved = _differing(
            _take(_run(_nudge(values, field, ulps, toward), ivgtyp, isltyp),
                  slice(None)),
            reference)
        assert len(moved) >= least, (
            f"{ulps} ULP of the entry {field} reached {len(moved)} returned "
            f"fields ({moved}); the measured floor is {least} or more, so "
            "either the comparison in this file cannot see a real difference "
            "or the coupling changed and the table above needs re-deriving")

    absorbed = _differing(
        _take(_run(_nudge(values, "soilt", 1, 400.0), ivgtyp, isltyp),
              slice(None)),
        reference)
    assert absorbed == [], (
        "one ULP of the entry soilt is now observable, so the floor recorded "
        f"in this docstring is wrong: {absorbed}.  Re-derive the sweep rather "
        "than adjusting the number")


def test_a_trailing_water_column_leaves_the_returned_ilnb_at_the_seed():
    """The one branch the trajectories above cannot reach.

    ``LSMRUC``'s loop resets the seed at the TOP of every iteration when the
    chain is off, and a water column then ``CYCLE``s before assigning
    anything -- so a grid whose LAST column is water returns the seed, not
    the last land column's answer, and a grid that is entirely water returns
    the seed having called ``sfctmp`` not at all.  With the chain ON, a water
    column does not reset it, so the value survives from the last land
    column.

    Both were verified bitwise against the pre-conversion scalar loop over 24
    layout/flag/ktau combinations.  Pinned here because every other gate in
    this file puts its water columns first, where the distinction is
    invisible, and because the batched path reconstructs this by hand rather
    than getting it for free from a loop.
    """
    n = 10
    values, ivgtyp, isltyp = _columns(n, snow=False)
    values["xland"] = np.ones(n, np.float32)
    values["xland"][-3:] = 2.0

    for chain in (False, True):
        result = _run(values, ivgtyp, isltyp, chain=chain)
        assert int(np.asarray(result.ilnb)) == int(DEFINED_ILNB)

    # ...and a grid with no land at all never calls sfctmp, so nothing but
    # the seed can come back.
    values["xland"] = np.full(n, 2.0, np.float32)
    for chain in (False, True):
        result = _run(values, ivgtyp, isltyp, chain=chain)
        assert int(np.asarray(result.ilnb)) == int(DEFINED_ILNB)
        # The water arm still ran: it forces the whole soil column to the
        # skin temperature and unit moisture (``:824-847``).
        np.testing.assert_array_equal(
            np.asarray(result.soilmois), np.ones((_NSOIL, n), np.float32))
        for level in range(_NSOIL):
            np.testing.assert_array_equal(
                np.asarray(result.tso)[level], np.asarray(result.soilt))


# ---------------------------------------------------------------------------
# 3.  what the conversion bought, and what it did not
# ---------------------------------------------------------------------------

_COST_COUNTS = (48, 96, 192, 384, 768)


def _elapsed(work, n: int, *, invocations: int = 1) -> float:
    start = time.perf_counter()
    for _ in range(invocations):
        work(n)
    return time.perf_counter() - start


def _ms_per_column(work, counts=_COST_COUNTS, repeats: int = 5) -> list[float]:
    """Best-of-``repeats`` cost after equal requested column work.

    Every timed sample processes ``max(counts)`` requested columns, and the
    five default rounds rotate the counts so each occupies every timing
    position once.  A single 48-column call is short enough for scheduler
    noise to dominate depending on which RUC timing test ran first; equal work
    and balanced order give every point comparable exposure while preserving
    both scientific bounds below.
    """
    counts = tuple(counts)
    sample_columns = max(counts)
    if any(sample_columns % n for n in counts):
        raise ValueError(
            "RUC cost counts must divide the equal-work sample exactly")

    for n in counts:
        work(n)

    samples = {n: [] for n in counts}
    for repeat in range(repeats):
        offset = repeat % len(counts)
        order = counts[offset:] + counts[:offset]
        for n in order:
            invocations = sample_columns // n
            processed_columns = n * invocations
            elapsed = _elapsed(work, n, invocations=invocations)
            samples[n].append(elapsed / processed_columns * 1e3)
    return [min(samples[n]) for n in counts]


def _width_trend(costs, counts=_COST_COUNTS) -> float:
    """Fitted per-column cost ratio across ``counts`` (>= 1, either direction).

    Least-squares slope of log(cost) against log(n), carried across the whole
    width range: 1.0 is flat, 16.0 is a fixed per-call cost amortized over
    48 -> 768 columns.  This is the quantity the projection to production
    width depends on.  ``max/min`` over the points is not: one point delayed
    by the scheduler moves it by the whole delay (2.9.12 pass 19 R5, a development machine at
    load ~150: 48: 0.845, 96: 0.785, 192: 1.037, 384: 0.663, 768: 0.653 read
    1.59 by max/min with no trend -- the middle point was the slow one; the
    fitted ratio of those numbers is 1.32 and the same test passed 3/3 at
    ~1.0x load), while the fit moves by a fraction of it.
    """
    x = np.log(np.asarray(counts, dtype=float))
    y = np.log(np.asarray(costs, dtype=float))
    slope = float(np.polyfit(x, y, 1)[0])
    return float(np.exp(abs(slope) * (x[-1] - x[0])))


def _land_surface_call(n: int) -> None:
    values, ivgtyp, isltyp = _columns(n, snow=False)
    _run(values, ivgtyp, isltyp)


def test_the_host_column_cost_is_still_flat_after_the_conversion() -> None:
    """Batching removed a constant factor.  It did not remove the flatness.

    Measured on the reference box over a 512x range and published in
    ``docs/wrf_ruc_runtime_admission.md``: 0.539 ms/column at 48 columns and
    0.479 at 24,576 warm; 1.322 and 1.254 snow-covered.  Against 1.711 and
    3.261 before, that is 3.6x warm and 2.6x over snow -- and a spread of
    1.13x and 1.06x, so the cost is **still flat**.

    Flatness surviving the conversion is the finding, not a disappointment:
    it says the residue is genuine per-element work rather than per-call
    overhead, and it is the per-element Python loops inside
    ``ruc_soil_properties``, ``ruc_soil_moisture_step``,
    ``_ruc_phase_partition``, ``ruc_soil_temperature_step`` and
    ``_ruc_vilka`` -- every one of which has a CUDA leaf already.

    The bound is loose because the absolute number belongs to the host.  What
    makes flatness a measurement rather than an assertion is the control.
    """
    costs = _ms_per_column(_land_surface_call)
    # Scheduler noise on a loaded host only ever ADDS time, so a further
    # best-of round can only lower a point toward its true cost; a real loss
    # of flatness survives every round.  (2.9.4 passes e3aa36d0 and 35a19528:
    # 40-slot Stage-1 hosts read 1.60-1.65 here while the unloaded file read
    # flat.)  The statistic is the fitted trend across the widths, not
    # max/min over the points: see ``_width_trend``.
    for _round in range(3):
        if _width_trend(costs) < 1.5:
            break
        costs = [min(a, b) for a, b in zip(costs, _ms_per_column(_land_surface_call))]
    spread = _width_trend(costs)
    report = ", ".join(f"{n}: {c:.3f}" for n, c in zip(_COST_COUNTS, costs))
    assert spread < 1.5, (
        f"RUC's per-column cost is no longer flat over {_COST_COUNTS} "
        f"({report}; fitted ratio {spread:.2f}x); the projection to "
        "production width assumes it is")

    # The control.  Hold the work FIXED at the smallest count and divide by n
    # anyway: that is what a call whose cost is NOT per-column looks like
    # through this same statistic, and it must be rejected.
    control = _ms_per_column(lambda _n: _land_surface_call(_COST_COUNTS[0]))
    control_spread = _width_trend(control)
    assert control_spread > 8.0, (
        "the flatness statistic did not reject constant work "
        f"(spread {control_spread:.1f}x), so it cannot tell per-column work "
        "from amortized work and the assertion above means nothing")

    # And the consequence, in the unit a caller cares about.  d04 of the
    # production four-domain case is 360,000 columns at dt = 1.667 s with
    # bldt = 0, so this is ONE of 25,920 land-surface calls per 12 h.
    projected = float(np.mean(costs)) * _PRODUCTION_D04_COLUMNS / 1e3
    assert projected > 60.0, (
        f"one 360,000-column land-surface call projects to {projected:.0f} s "
        "on this host, which is fast enough that the flat-cost blocker "
        "recorded in the registry and in the admission document should be "
        "re-measured rather than trusted")
