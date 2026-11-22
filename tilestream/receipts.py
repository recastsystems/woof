"""Domain-wide FP64 receipts for a domain that is streamed, not resident.

WOOF-ArWen's conservation receipts are whole-domain SUMS, and that one word is
what separates them from every other observer this transport has had to
carry.  A max or an OR folds over tiles for free -- ``max`` is associative
AND idempotent, so a tile overlap costs nothing and the tile order does not
matter.  A SUM is associative in exact arithmetic and NOT in floating point,
and it double-counts whatever the tiles overlap in.  Both halves of that bite
here, and they bite differently.

TWO DEFECTS, AND THE SECOND SURVIVES FIXING THE FIRST
-----------------------------------------------------
*Stale state.*  A streamed domain's carriers live in the store; the
``DomainState`` the model still holds is the one the store was FILLED FROM
and it is never stepped again.  So ``dycore.domain_mass_measure(state)`` on a
streamed run returns the t=0 mass, forever, with no error and no NaN.
MEASURED, 256x192x49, 24 steps, periodic: the resident run ends at
3205297165.2070312 and the state-read receipt returns 3205297165.3867188 --
the t=0 value exactly, 376832 ulp away (1223182 ulp on the mapped domain).  A
receipt like that does not look broken.  It looks like a beautifully
conserved domain.

*Reduction order.*  Fix the first defect the obvious way -- fold each tile's
contribution as the sweep goes past -- and the number comes back differing in
the last few ulp from the resident run's, because 16 partial sums rounded and
then added is not the same rounding as one sum of 49152 terms.  Every other
streaming result in this project is stated as BIT-EXACT.  A conservation
receipt that disagrees in the last ulp looks exactly like a transport bug,
and chasing one costs a day.

So this module computes the receipts from the STORE, in the MONOLITHIC
TRAVERSAL ORDER -- one reduction over the whole (ny, nx) plane, the same
reduction the resident run does -- and does not fold per tile.  That is a
deliberate policy, not an implementation accident, and it is stated here
because the number is produced here.

WHY THE OUT-OF-CORE PREMISE DOES NOT FORBID IT
----------------------------------------------
"The domain does not fit on the card" is a statement about the 3-D carriers.
The dry-mass measure is a 2-D integral: ``sum(mu / msft**2)`` over (ny, nx).
The largest domain the planner will build -- 5476^2 x 49 against a 44 GiB
pinned host ceiling -- puts 30.0 M columns, 240 MB of FP64, on the GPU for
that reduction.  2-D always fits.  A receipt that IS 3-D does not get this
escape, which is why ``cfg.tke_budget`` is refused below rather than
reimplemented.

WHAT THE FOLD ACTUALLY COSTS, MEASURED
--------------------------------------
Both legs on the same values, 256x192x49, 24 steps, 4x4 tiles that divide the
domain exactly, interiors partitioning it with no overlap.  Every number is
one association of the SAME 49152 terms against the monolithic reduction, and
the survey row folds the same values over six different partitions -- 2x2,
4x4, 8x8, 16x16, per-row, per-column -- because one draw invites the reader
to conclude "0 ulp, folding is fine":

====================================  ====  ==================  ==============
configuration                            N  the 4x4 fold        6-fold survey
====================================  ====  ==================  ==============
periodic, ``map_proj=0``, flat          24  0 ulp               all six 0
periodic, real Lambert + real terrain   24  +1 ulp, 4.77e-07    +1 +1 -1 +2
                                            abs, 1.52e-16 rel   -2 -1 (max 2)
periodic, real Lambert + real terrain    8  0 ulp               0 0 -2 -1 0
                                                                -4 (max 4)
====================================  ====  ==================  ==============

The first row is a TRAP and is recorded so nobody re-derives it as
reassurance.  Under the milestone-one settings ``msft == 1`` exactly, so
``cell_area_weight`` is exactly 1.0; every term is then an FP32 column mass,
a multiple of 2**-7 near 65000, and the whole 49152-term sum needs 39 bits.
It is EXACT in FP64, so no association can change it and the zero says
nothing about associativity.  The second row is the real answer: with
``map_proj=1`` the weight ``1/msft**2`` carries a full FP64 mantissa, the sum
rounds, and folding moves it -- by up to 4 ulp, in a direction and by an
amount that depend on the partition AND on the step count.  That wandering is
itself the finding: the number a folded receipt returns is not a property of
the domain, it is a property of the tiling, and the SAME tiling gives a
different answer at a different N.  Nothing about the transport moves.

4 ulp is 2e-6 of a domain mass whose 24-step CHANGE is -0.583, so the fold is
physically irrelevant and diagnostically fine.  It is the bit-exactness claim
it destroys, and that claim is the only reason anyone trusts this mode.

THREE RECEIPTS THIS MODULE REFUSES INSTEAD OF PRODUCING
-------------------------------------------------------
Refusal is a design choice here, and the alternative is not "produce nothing"
-- it is "produce a plausible wrong number", which is what the code does
today.  Each refusal names what a caller would have to build.

``cfg.tke_budget`` (``woof.core.tke_budget``)
    A 3-D VOLUME integral accumulated by a device kernel INSIDE
    ``dycore.step``, into ``state._scratch['tke_budget_acc']``.  Under
    streaming that step runs on the TILE, so the accumulation lands on the
    tile buffer; the slot is classified REBUILT (``woof/io/restart.py``
    ``REBUILT_SCRATCH_PREFIXES``), so it is not a carrier and is neither
    gathered nor scattered.  MEASURED, 24 steps, 16 tiles, 2 buffers: the
    domain state's ``drain`` returns ``None`` (it never stepped), each buffer
    holds 192 accumulated steps rather than 24, and summing the two buffers
    gives 0.311x the resident volume integral on every physical term --
    because a buffer's reduction covers its GATHERED window, halo included,
    at a redundancy of 2.5, divided by 8x the step count.  ``transport`` is
    worse than scaled: -1.9e+06 times the resident value, sign flipped,
    because a tile's window edges carry an advective flux divergence that
    telescopes away only over the whole domain.  Making this right needs an
    interior mask inside the accumulate kernel plus a domain-level
    accumulator the tiles reduce into -- a kernel change, not a wrapper.

``mass_flux_observer`` / ``mass_flux_accumulator``
    The telescoped lateral flux, sampled per acoustic substep from the
    state's OUTERMOST FACES.  It cannot be taken from the store at all: it
    reads ``u_pp``/``v_pp``, which are within-step acoustic scratch and do
    not exist at any point where the store is coherent.  Taken per tile it is
    wrong twice over -- ``_boundary_x(cfg)`` is a CONFIG property, so every
    tile believes all four of its window sides are domain edges, and under
    ``periodic=False`` neighbouring tiles' clamped windows overlap along the
    same domain edge by ``2*halo`` rows, so even the true edges double-count.
    :meth:`tilestream.driver.TiledRun.sweep` refuses both keywords for that
    reason.

``hmix_k_diag`` / ``sase_flux_diag``
    Not reductions at all -- (nz, ny, nx) and (nz+1, ny, nx) output fields on
    the ``PhysicsDriver``.  ``woof/io/restart.py`` classifies both
    output-only and ``tilestream.physics_inventory.OUTPUT_ONLY_DRIVER_ATTRS``
    already records the streaming consequence: they are neither gathered nor
    scattered, so under tiling each holds whatever the last tile through that
    buffer wrote.  Trajectory-inert, diagnostically wrong, and out of this
    module's scope because there is nothing to sum.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: The carrier key for perturbation column mass under both store
#: conventions: ``tilestream.gather.inventory`` names arrays by bare
#: attribute, ``tilestream.physics_inventory.carrier_inventory`` by the
#: restart member name.  A store built either way answers here.
MUP_KEYS = ("state/mup", "mup")


class ReceiptRefused(RuntimeError):
    """A domain receipt that cannot be produced accurately under streaming."""


class StoreDomainView:
    """The two methods ``dycore.domain_mass_measure`` actually calls.

    A view rather than a reimplementation, and that is the point: the
    reduction that produces the receipt stays the one line in
    ``woof.core.dycore``, so the store leg and the resident leg cannot drift
    apart because somebody edited one of them.  ``domain_mass_measure`` reads
    ``total_mu()`` and ``cell_area_weight()`` and nothing else, so a duck with
    those two methods is a domain as far as it is concerned.

    ``mup`` comes from the STORE (it moves every step).  ``mub2d`` and
    ``msft`` come from the domain's geography, which is INPUT -- gathered per
    buffer, never scattered, never changed -- so the prepared state still
    holds the domain's own copy and it is still correct at step N.
    """

    def __init__(self, mup, mub2d, msft):
        import cupy as cp

        self._mup = cp.asarray(mup)
        self._mub2d = cp.asarray(mub2d)
        self._msft = cp.asarray(msft)
        if self._mup.shape != self._msft.shape:
            raise ReceiptRefused(
                f"the store's column mass is {self._mup.shape} and the "
                f"domain map factor is {self._msft.shape}; the receipt "
                "would weight the wrong cells")

    # -- the DomainState surface domain_mass_measure consumes --------------

    def total_mu(self):
        return self._mub2d + self._mup

    def cell_area_weight(self):
        import cupy as cp

        msft2 = self._msft * self._msft
        return 1.0 / msft2.astype(cp.float64)


def store_arrays(source) -> Mapping[str, Any]:
    """``{name: array}`` for a StreamedDomain, a host store or a mapping."""
    for attr in ("store", "arrays"):
        inner = getattr(source, attr, None)
        if isinstance(inner, Mapping):
            return inner
        if inner is not None and isinstance(getattr(inner, "arrays", None),
                                            Mapping):
            return inner.arrays
    if isinstance(source, Mapping):
        return source
    raise ReceiptRefused(
        f"{source!r} is not a streamed domain, a host store or a "
        "{name: array} mapping")


def column_mass(source):
    """The store's perturbation column mass, under either key convention."""
    arrays = store_arrays(source)
    for key in MUP_KEYS:
        if key in arrays:
            return arrays[key]
    raise ReceiptRefused(
        f"the store holds none of {list(MUP_KEYS)}; without the domain's "
        "column mass there is no dry-mass receipt to take")


def domain_geography(state) -> tuple[Any, Any]:
    """``(mub2d, msft)`` of the DOMAIN, from the prepared state.

    Geography is INPUT: the tiled transport gathers it into a buffer when
    that buffer changes tile and never writes it back, so the domain state's
    copy is as valid at step N as it was at step 0.  Read it from the state
    rather than from a tile, which holds one window of it.
    """
    for name in ("mub2d", "msft"):
        if getattr(state, name, None) is None:
            raise ReceiptRefused(
                f"the domain state carries no {name!r}; the dry-mass "
                "receipt is area-weighted and cannot be taken without the "
                "domain's map factor")
    return state.mub2d, state.msft


def domain_mass_measure(streamed, state=None) -> float:
    """The FP64 dry-mass receipt of a STREAMED domain, in monolithic order.

    ONE reduction over the whole (ny, nx) plane -- the same call, on the same
    values, in the same traversal order as
    :func:`woof.core.dycore.domain_mass_measure` on a resident domain.  It
    is therefore BIT-IDENTICAL to the resident run's, which is what lets the
    receipt sit beside every other bit-exactness claim this mode makes rather
    than undermining them.  MEASURED at 0 ulp on both a flat identity-map
    periodic domain and a real Lambert + real terrain one; the tile-folded
    alternative costs 1 ulp on the second (see the module docstring).

    ``streamed`` is a :class:`woof.core.streaming.StreamedDomain`, a
    :class:`tilestream.hoststore.HostDomainStore` or a plain store mapping.
    ``state`` is the prepared domain state the geography is read from; a
    ``StreamedDomain`` carries its own and does not need it.
    """
    from woof.core.dycore import domain_mass_measure as _measure

    if state is None:
        state = getattr(streamed, "state", None)
    if state is None:
        raise ReceiptRefused(
            "the dry-mass receipt is area-weighted by the DOMAIN's map "
            "factor; pass the prepared domain state the store was filled "
            "from, or a StreamedDomain that still holds it")
    mub2d, msft = domain_geography(state)
    return _measure(StoreDomainView(column_mass(streamed), mub2d, msft))


def mass_budget_terms(streamed, cfg) -> dict[str, float]:
    """The three separately-measured terms, for the domains that have them.

    A fully periodic domain has no lateral boundary, so every one of
    :data:`woof.verify.conservation_closure.MASS_BUDGET_TERMS` is exactly
    zero and the residual IS the conservation observable -- which is why the
    receipt gate runs periodic.  Anything else raises: the flux term is not
    obtainable from a store (see the module docstring), and defaulting it to
    zero on a forced domain is precisely the false receipt
    ``conservation_closure.mass_budget_residual`` refuses to write.
    """
    from woof.core.dycore import _boundary_x, _boundary_y

    if _boundary_x(cfg) or _boundary_y(cfg):
        raise ReceiptRefused(
            "a streamed domain with lateral boundaries cannot report its "
            "telescoped lateral flux: the term is sampled per acoustic "
            "substep from u_pp/v_pp, which are within-step scratch and do "
            "not exist where the store is coherent, and per tile it is "
            "wrong twice over (every tile believes its window edges are "
            "domain edges; clamped windows overlap along a shared domain "
            "edge by 2*halo).  Reporting zero here would be a flux-only "
            "closure on a forced domain, which is the exact false receipt "
            "woof.verify.conservation_closure refuses to write.")
    return {"lateral_flux_integral": 0.0,
            "lbc_mass_forcing_integral": 0.0,
            "specified_zone_mass_reset": 0.0}


def tke_budget(streamed, cfg):
    """Refuse the km_opt=2 volume budget, with the reason and the fix.

    Today's failure mode is worse than an exception and that is why this
    exists: ``tke_budget.drain(state, cfg)`` on a streamed domain returns
    ``None``, which is the same answer it gives when the budget is switched
    off and when no step has been taken yet.  A caller cannot tell "this
    configuration does not accumulate" from "this configuration accumulated
    somewhere you are not looking".
    """
    from woof.core import tke_budget as _tke

    if not _tke.enabled(cfg):
        return None
    raise ReceiptRefused(
        "cfg.tke_budget=1 is a 3-D volume integral accumulated inside "
        "dycore.step, so under streaming it accumulates on the TILE BUFFER "
        "and its scratch slots are classified REBUILT -- neither gathered "
        "nor scattered, never in the store.  MEASURED at 256x192x49, 24 "
        "steps, 16 tiles, 2 buffers: the domain state drains None, each "
        "buffer holds 192 steps instead of 24, and summing the buffers "
        "gives 0.311x the resident volume integral on every physical term "
        "(gathered windows at redundancy 2.5 over 8x the step count) with "
        "'transport' off by -1.9e+06x and sign-flipped.  Making it right "
        "needs an interior mask inside the tke_budget_accumulate kernel and "
        "a domain-level accumulator the tiles reduce into, in a fixed order "
        "-- a kernel change, not a wrapper, and not attempted here.")


def receipt_provenance(mode: str) -> dict[str, str]:
    """What a receipt should carry beside the number, so a reader can bind it.

    Two runs of the same forecast can now produce the same receipt by two
    different routes, and a reader who cannot tell which route produced a
    number cannot tell whether an ulp of disagreement is a transport bug or a
    reduction order.  So the route travels with the value.
    """
    if mode not in ("resident", "streamed-store"):
        raise ReceiptRefused(f"unknown receipt route {mode!r}")
    return {
        "receipt_source": ("woof.core.dycore.domain_mass_measure(state)"
                           if mode == "resident"
                           else "tilestream.receipts.domain_mass_measure"
                                "(store)"),
        "reduction_order": "monolithic",
        "reduction_order_note": (
            "one FP64 reduction over the whole (ny, nx) plane; NOT folded "
            "per tile.  Folding is 1 ulp away on a real projection and "
            "0 ulp on an identity-map domain where the sum is exact -- see "
            "tilestream/receipts.py"),
    }


__all__ = [
    "MUP_KEYS", "ReceiptRefused", "StoreDomainView", "column_mass",
    "domain_geography", "domain_mass_measure", "mass_budget_terms",
    "receipt_provenance", "store_arrays", "tke_budget",
]
