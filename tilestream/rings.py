"""Halo rings instead of a full shadow: one domain store plus ~6% of a second.

:mod:`tilestream.driver`'s ``write_mode="shadow"`` is correct and costs a
whole second copy of the domain.  That doubling is what sets the largest
runnable domain, because the host store is the binding constraint (see
:mod:`tilestream.hoststore`): at 32.26 B/cell dry and a 44.14 GiB pinned
ceiling, a single store holds 5476^2 x 49 and a shadow pair only 3872^2.

This module removes the doubling.  It is pure index arithmetic plus a thin
block-copy executor; it never decides WHEN anything runs -- ``run_tiled``
does that -- but it does decide WHAT has to be saved, and that is the entire
correctness argument, so the argument is written out here in full.


THE READ-AT-TIME-t RULE, AND WHY A RING IS ENOUGH
-------------------------------------------------
Every tile must see its neighbours as they were at the START of the sweep.
With one store, a tile that scatters its interior in place destroys exactly
that.  But it does not destroy all of it: a tile's interior splits into

* cells some OTHER tile will read this sweep -- always within the tile's
  outer band, because a neighbour's compute window only reaches ``halo``
  cells in (and the exact reach is computed here, not assumed); and
* a CORE that no other tile ever reads.

The core can be overwritten the instant the tile is stepped.  The band -- the
RING -- has to survive until every tile that reads it has read it.  So::

    gather tile j from the store        (may contain already-written cells)
    save   tile j's own ring            (still time-t: nobody has written it)
    patch  tile j's window from the rings of the tiles already written
    step   tile j
    scatter tile j's whole interior back into the store, in place

After gather+patch the tile's window holds time-t values EVERYWHERE, which is
the invariant :func:`assert_ring_covers_reads` proves for a plan and the gate
proves numerically against a monolithic run.

Note what the patch buys beyond memory: it makes the gather's result
independent of whether the racing scatter has landed yet.  A cell the gather
may have read at ``t + dt`` is exactly a cell in some already-written tile's
ring, and the patch overwrites it with the saved ``t`` value either way.


THE FOUR WAYS TO GET THIS SUBTLY WRONG
--------------------------------------
Each of these is silent: the run completes, the fields look like weather, and
only a bit-exact comparison against a monolithic run says otherwise.

1. **A ring of exactly ``halo`` cells is too narrow for the staggered
   variants at the periodic seam.**  A tile whose window ends at mass cell
   ``j1 + halo`` reads ``v`` faces up to and INCLUDING ``j1 + halo`` -- one
   more point than the mass window, because the closing face belongs to the
   next cell.  Wrapped around the seam that lands on face ``halo`` of the
   first tile row, which an ``halo``-wide band does not contain.  MEASURED:
   ``margin_mode="halo"`` leaves 192 such cells unsaved on a 192^2 3x3 plan
   and 9,344 on a ragged 200x192 one.

   AND IT IS BIT-EXACT ANYWAY, at halo 14, 15 and 16 alike -- the dropped
   cells sit at the OUTER edge of the reading tile's window, outside the
   influence cone that decides that tile's interior.  So this failure is
   invisible to the hash gate at every halo that produces a correct answer at
   all, and :func:`assert_ring_covers_reads` is the only instrument that sees
   it.  That is the reason the margins come from the plan's OWN gather
   rectangles, per variant, and never from the halo width: the alternative is
   sound only by an accident nothing tests.

2. **A non-periodic edge tile reaches ``2*halo`` into its neighbour.**
   :mod:`tilestream.spec` clamps an edge window by SLIDING it inward rather
   than truncating it, so tile 0's window is ``[0, tile + 2*halo)`` and
   overlaps its right neighbour by ``2*halo``, not ``halo``.  Same cure: the
   reach is measured from the plan.

3. **A compute window wider than the domain reads every tile's core.**
   ``cnx > nx`` (a plan with one tile in x, or a tile+halo bigger than the
   domain) wraps the window onto itself, so "only the outer band is read" is
   simply false and the whole interior has to be saved.  The greedy margin
   search arrives at that on its own because it is driven by the actual
   intersections; the erosion formula in the old
   ``driver.ring_bytes_vs_shadow`` estimate would not have.

4. **The two cross-tile hazards are on DIFFERENT CUDA streams.**  Ordering
   inside a stream is free; between streams there is none, and both of the
   orderings this scheme needs are between streams:

   * WAR on the store -- tile j's gather (j < k) READS cells that tile k's
     scatter WRITES.  There is no patch for this direction (tile k's ring is
     not saved until tile k is gathered), so the read must happen first.
   * RAW on the ring arena -- tile k's save WRITES a band that tile j's patch
     (j > k) READS.

   It is tempting to argue that this card satisfies both for free:
   ``asyncEngineCount == 1`` means one DMA queue drained in submission order,
   and the loop submits in tile order.  THAT ARGUMENT IS FALSE, and it was
   measured to be false rather than reasoned away.  Running the gate's 3x3
   plan with the events removed (``run_tiled(ring_ordering="submission")``)
   is BIT-EXACT on an idle card and WRONG BY 3.7e+02 while another process
   shares the GPU -- same code, same machine, hours apart.  The submission
   order is a property of an uncontended card, so a run that relied on it
   would corrupt intermittently, as a function of what else the machine was
   doing.  ``run_tiled`` therefore records one event per tile and waits on
   the few neighbours that matter: :attr:`RingPlan.war_deps` and
   :attr:`RingPlan.patch_deps` are those neighbour lists, and they are the
   reason this scheme is safe rather than lucky.

A fifth, which the loop rather than this module has to honour: the ring holds
time-t values for ONE sweep.  Nothing may be prefetched across a sweep
boundary, and the arena is refilled every sweep.


WHAT IT COSTS
-------------
Only the bands read by a LATER tile need saving; a band read by an earlier
tile is protected by the WAR ordering instead.  In row-major order that is
the top and right bands of a typical tile (plus the bottom of row 0 and the
left of column 0, which the periodic wrap makes "later"), i.e. about half of
the full frame the naive estimate assumed.  MEASURED as arena bytes over store
bytes at halo 16: 4.97% at 1950^2 tile 650, 5.84% at 3276^2 tile 546, 2.39% at
5412^2 tile 1353 -- against 100% for the shadow, and against the ~10-13% a
four-sided ``interior - erosion(halo)`` estimate predicts for the same plans.

A tile size that does NOT divide the domain is far dearer, because a ragged
trailing tile is narrower than its neighbours' windows and so is read right
through: 3276^2 at tile 512 leaves 13 ragged tiles and costs 24.46%, against
5.84% at tile 546, which divides it exactly.  A 6.6% change in tile size, a
4.2x change in the arena.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field as _dc_field
from typing import Any, Iterable, Mapping, Sequence

from tilestream import gather as _gather


__all__ = [
    "Band",
    "Rect",
    "RingArena",
    "RingError",
    "RingPlan",
    "arena_bytes",
    "arena_bytes_for_manifest",
    "assert_ring_covers_reads",
    "build_ring_plan",
    "field_geometry",
    "geometry_kind",
    "max_square_domain",
    "ring_report",
]


class RingError(RuntimeError):
    """A ring plan could not be built or does not cover what it must."""


#: Half-open rectangle in FULL-ARRAY coordinates for one variant.  ``y``/``x``
#: are the array's own trailing axes, so for ``u`` the ``x`` range may include
#: the alias slot ``nx`` and for ``v`` the ``y`` range may include ``ny``.
Rect = namedtuple("Rect", "y0 y1 x0 x1")

#: The three horizontal geometries.  ``w``/``phi`` are mass points
#: horizontally (:mod:`tilestream.spec` gives them stagger ``(0, 0)``), so
#: they share the mass rectangles exactly and are never planned separately.
GEOMETRY_KINDS: tuple[str, ...] = ("mass", "u", "v")


def geometry_kind(kind: str) -> str:
    """The horizontal geometry a :func:`tilestream.gather.classify` kind uses."""
    return "mass" if kind in ("w", "phi") else kind


def _area(r: Rect) -> int:
    return max(0, r.y1 - r.y0) * max(0, r.x1 - r.x0)


def _intersect(a: Rect, b: Rect) -> Rect | None:
    y0, y1 = max(a.y0, b.y0), min(a.y1, b.y1)
    if y0 >= y1:
        return None
    x0, x1 = max(a.x0, b.x0), min(a.x1, b.x1)
    if x0 >= x1:
        return None
    return Rect(y0, y1, x0, x1)


def _transfer_rects(spec, direction: str, kind: str) -> list[tuple[Rect, Rect]]:
    """``[(full_rect, tile_rect)]`` for one variant, from the spec itself.

    Both sides come from :mod:`tilestream.spec`'s own ``Transfer`` objects, so
    the periodic wrap, the ragged trailing tile, the alias slot and the
    staggered extra face are all whatever the transport actually does -- this
    module never re-derives them.  The two rectangles of a pair always have
    the same extent, which is what makes ``tile = full + offset`` valid inside
    it.
    """
    fn = getattr(spec, direction, None)
    if not callable(fn):
        raise RingError(
            f"ring planning needs {type(spec).__name__}.{direction}(kind) to "
            "return tilestream.spec.Transfer objects; this spec does not "
            "expose it, so the read/write geometry cannot be established")
    out: list[tuple[Rect, Rect]] = []
    for t in fn(kind):
        full = Rect(t.full_y.start, t.full_y.stop, t.full_x.start, t.full_x.stop)
        tile = Rect(t.tile_y.start, t.tile_y.stop, t.tile_x.start, t.tile_x.stop)
        if (full.y1 - full.y0) != (tile.y1 - tile.y0) or \
                (full.x1 - full.x0) != (tile.x1 - tile.x0):
            raise RingError(
                f"{direction} transfer for kind {kind!r} has mismatched "
                f"extents {full} vs {tile}")
        out.append((full, tile))
    return out


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------

def _covered(w: Rect, margins: tuple[int, int, int, int], r: Rect) -> bool:
    """True if ``r`` lies entirely in the frame of ``w`` with ``margins``."""
    mb, mt, ml, mr = margins
    core = Rect(w.y0 + mb, w.y1 - mt, w.x0 + ml, w.x1 - mr)
    if core.y0 >= core.y1 or core.x0 >= core.x1:
        return True                     # the frame is the whole rectangle
    return _intersect(core, r) is None


def _greedy_margins(w: Rect, rects: Sequence[Rect]) -> tuple[int, int, int, int]:
    """Smallest frame (bottom, top, left, right margins) covering ``rects``.

    Greedy, and it does not have to be optimal: every candidate is a value
    that COVERS the rectangle being considered, margins only ever grow, and
    :func:`assert_ring_covers_reads` re-checks the result against the same
    intersections independently.  The cheapest-first choice is what turns the
    typical "read from the north and from the east" pattern into two bands
    instead of a full frame.

    Any rectangle can always be covered on any of the four sides (e.g. a
    bottom margin of ``r.y1 - w.y0`` puts every row of ``r`` inside the
    bottom band), so this terminates with a valid frame even for a window
    that wraps onto itself and reads the whole interior.
    """
    margins = (0, 0, 0, 0)
    height = w.y1 - w.y0
    width = w.x1 - w.x0
    for r in rects:
        if _covered(w, margins, r):
            continue
        mb, mt, ml, mr = margins
        cands = (
            ((r.y1 - w.y0 - mb) * width, 0, r.y1 - w.y0),
            ((w.y1 - r.y0 - mt) * width, 1, w.y1 - r.y0),
            ((r.x1 - w.x0 - ml) * height, 2, r.x1 - w.x0),
            ((w.x1 - r.x0 - mr) * height, 3, w.x1 - r.x0),
        )
        _cost, which, value = min(cands)
        margins = tuple(value if i == which else m
                        for i, m in enumerate(margins))       # type: ignore[assignment]
    return margins                                            # type: ignore[return-value]


def _frame_bands(w: Rect, margins: tuple[int, int, int, int]) -> list[Rect]:
    """The frame as up to four DISJOINT rectangles, in a stable order.

    Bottom and top take the full width; left and right take only the rows
    between them.  Either pair collapses to the whole rectangle when the
    margins meet, which is what a fully-read interior needs.
    """
    mb, mt, ml, mr = margins
    height, width = w.y1 - w.y0, w.x1 - w.x0
    if mb <= 0 and mt <= 0 and ml <= 0 and mr <= 0:
        return []
    if mb + mt >= height:
        return [w]
    bands: list[Rect] = []
    if mb > 0:
        bands.append(Rect(w.y0, w.y0 + mb, w.x0, w.x1))
    if mt > 0:
        bands.append(Rect(w.y1 - mt, w.y1, w.x0, w.x1))
    my0, my1 = w.y0 + mb, w.y1 - mt
    if ml + mr >= width:
        if my0 < my1:
            bands.append(Rect(my0, my1, w.x0, w.x1))
        return bands
    if ml > 0:
        bands.append(Rect(my0, my1, w.x0, w.x0 + ml))
    if mr > 0:
        bands.append(Rect(my0, my1, w.x1 - mr, w.x1))
    return bands


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Band:
    """One saved rectangle of one tile's write set, for one geometry kind.

    ``rect`` is in full-array coordinates; ``tile_y0``/``tile_x0`` is where it
    sits inside the OWNING tile's array (so the save is a plain block copy);
    ``plane_offset`` is where its plane lives in that kind's arena, in cells.
    """

    index: int
    tile: int
    kind: str
    rect: Rect
    tile_y0: int
    tile_x0: int
    plane_offset: int

    @property
    def height(self) -> int:
        return self.rect.y1 - self.rect.y0

    @property
    def width(self) -> int:
        return self.rect.x1 - self.rect.x0

    @property
    def cells(self) -> int:
        return self.height * self.width


#: One patch block: take ``(h, w)`` from band ``band`` at ``(by0, bx0)`` and
#: put it in the tile array at ``(ty0, tx0)``.
Patch = namedtuple("Patch", "band by0 bx0 ty0 tx0 h w")

#: One save block: take ``(h, w)`` from the tile array at ``(ty0, tx0)`` and
#: put it at the origin of band ``band``.
Save = namedtuple("Save", "band ty0 tx0 h w")


@dataclass
class RingPlan:
    """Which cells have to survive a tile's own scatter, and where they go.

    Built once per (plan geometry, kind set).  Holds no arrays: the storage
    and the copies are :class:`RingArena`'s job.
    """

    specs: list
    kinds: tuple[str, ...]
    bands: list[Band]
    saves: list[list[Save]]
    patches: list[list[Patch]]
    war_deps: list[tuple[int, ...]]
    patch_deps: list[tuple[int, ...]]
    plane_cells: dict[str, int]
    margin_mode: str
    needed: dict[tuple[int, str], list[Rect]] = _dc_field(default_factory=dict)

    @property
    def ntiles(self) -> int:
        return len(self.specs)

    def cells_by_kind(self) -> dict[str, int]:
        out = {k: 0 for k in self.kinds}
        for b in self.bands:
            out[b.kind] += b.cells
        return out


def build_ring_plan(specs: Sequence, kinds: Iterable[str] = GEOMETRY_KINDS, *,
                    margin_mode: str = "exact") -> RingPlan:
    """Work out the ring for a plan: what to save, what to patch, what to order.

    ``specs`` is a :func:`tilestream.spec.plan_tiles` list; tiles are
    processed in exactly that order and the answer depends on the order, so
    the two must not be permuted independently.

    ``margin_mode`` is ``"exact"`` for a forecast.  The other two are broken
    on purpose, and what they do to the ANSWER is not what one would guess:

    ``"exact"``
        Every margin comes from the intersections of the plan's own gather
        and scatter rectangles.  Provably sufficient, and SMALLER than the
        naive frame (44,862 saved cells against 67,200 on a 192^2 3x3 plan),
        because a band only a preceding tile reads is not saved at all.

    ``"halo"``
        A band exactly ``spec.halo`` wide on all four sides -- the plausible
        mistake.  It genuinely violates the invariant: MEASURED 192 cells on
        a 192^2 3x3 plan, 9,280 on a ragged 200x192 one, which
        :func:`assert_ring_covers_reads` reports and refuses.  And yet it is
        BIT-EXACT against a monolithic run at halo 14, 15 and 16 alike.  The
        cells it misses are the staggered closing face and the deep reach of
        a ragged tile, both of which sit at the OUTER edge of the reading
        tile's window, outside the influence cone that decides that tile's
        interior.  So this mode is the demonstration that the hash gate
        CANNOT see this class of error and the geometric invariant is not
        redundant with it.  Do not conclude that the naive ring is fine: it
        is unsound at any halo, and its soundness margin is invisible.

    ``"x_only"``
        Margins in x only, i.e. the mistake of thinking about the x seam and
        forgetting the y one.  This one does change the answer, in any plan
        with a y seam, which is what makes it the negative control the gate
        runs.
    """
    if margin_mode not in ("exact", "halo", "x_only"):
        raise ValueError(f"margin_mode must be 'exact', 'halo' or 'x_only', "
                         f"got {margin_mode!r}")
    specs = list(specs)
    if not specs:
        raise RingError("empty plan")
    kinds = tuple(dict.fromkeys(geometry_kind(k) for k in kinds))
    for kind in kinds:
        if kind not in GEOMETRY_KINDS:
            raise RingError(f"unknown geometry kind {kind!r}")

    ntiles = len(specs)
    reads: dict[tuple[int, str], list[tuple[Rect, Rect]]] = {}
    writes: dict[tuple[int, str], list[tuple[Rect, Rect]]] = {}
    for i, spec in enumerate(specs):
        for kind in kinds:
            reads[(i, kind)] = _transfer_rects(spec, "gather", kind)
            writes[(i, kind)] = _transfer_rects(spec, "scatter", kind)

    bands: list[Band] = []
    plane_cells = {k: 0 for k in kinds}
    needed: dict[tuple[int, str], list[Rect]] = {}
    war: list[set[int]] = [set() for _ in range(ntiles)]

    for k in range(ntiles):
        for kind in kinds:
            for w_full, w_tile in writes[(k, kind)]:
                later: list[Rect] = []
                for j in range(ntiles):
                    if j == k:
                        continue
                    hit = False
                    for r_full, _r_tile in reads[(j, kind)]:
                        overlap = _intersect(r_full, w_full)
                        if overlap is None:
                            continue
                        hit = True
                        if j > k:
                            later.append(overlap)
                    if hit and j < k:
                        # tile j read this rectangle BEFORE tile k wrote it;
                        # nothing is saved for that direction, so the loop
                        # must order the read ahead of the write.
                        war[k].add(j)
                needed.setdefault((k, kind), []).extend(later)
                if not later:
                    continue
                if margin_mode == "halo":
                    h = int(getattr(specs[k], "halo", 0))
                    margins = (h, h, h, h)
                else:
                    margins = _greedy_margins(w_full, later)
                    if margin_mode == "x_only":
                        margins = (0, 0, margins[2], margins[3])
                for rect in _frame_bands(w_full, margins):
                    bands.append(Band(
                        index=len(bands), tile=k, kind=kind, rect=rect,
                        tile_y0=w_tile.y0 + (rect.y0 - w_full.y0),
                        tile_x0=w_tile.x0 + (rect.x0 - w_full.x0),
                        plane_offset=plane_cells[kind]))
                    plane_cells[kind] += bands[-1].cells

    by_kind: dict[str, list[Band]] = {k: [] for k in kinds}
    for band in bands:
        by_kind[band.kind].append(band)

    saves: list[list[Save]] = [[] for _ in range(ntiles)]
    for band in bands:
        saves[band.tile].append(
            Save(band=band, ty0=band.tile_y0, tx0=band.tile_x0,
                 h=band.height, w=band.width))

    patches: list[list[Patch]] = [[] for _ in range(ntiles)]
    patch_deps: list[set[int]] = [set() for _ in range(ntiles)]
    for j in range(ntiles):
        for kind in kinds:
            for r_full, r_tile in reads[(j, kind)]:
                for band in by_kind[kind]:
                    if band.tile >= j:
                        continue
                    overlap = _intersect(r_full, band.rect)
                    if overlap is None:
                        continue
                    patches[j].append(Patch(
                        band=band,
                        by0=overlap.y0 - band.rect.y0,
                        bx0=overlap.x0 - band.rect.x0,
                        ty0=r_tile.y0 + (overlap.y0 - r_full.y0),
                        tx0=r_tile.x0 + (overlap.x0 - r_full.x0),
                        h=overlap.y1 - overlap.y0,
                        w=overlap.x1 - overlap.x0))
                    patch_deps[j].add(band.tile)

    plan = RingPlan(
        specs=specs, kinds=kinds, bands=bands, saves=saves, patches=patches,
        war_deps=[tuple(sorted(s)) for s in war],
        patch_deps=[tuple(sorted(s)) for s in patch_deps],
        plane_cells=plane_cells, margin_mode=margin_mode, needed=needed)
    if margin_mode == "exact":
        assert_ring_covers_reads(plan)
    return plan


def assert_ring_covers_reads(plan: RingPlan) -> None:
    """Raise unless every cell a LATER tile reads is in the saved bands.

    The independent re-check of :func:`_greedy_margins`: it walks the same
    intersections the plan was built from and tests each against the bands
    that were actually emitted, so a margin search that undercounted, a band
    decomposition that dropped a corner, or a plan whose specs were reordered
    after the fact all fail here rather than in the forecast.

    Cheap -- rectangle containment, no arrays.  The exhaustive per-cell
    version lives in ``test_rings`` and agrees with it.
    """
    by_tile_kind: dict[tuple[int, str], list[Rect]] = {}
    for band in plan.bands:
        by_tile_kind.setdefault((band.tile, band.kind), []).append(band.rect)
    for (tile, kind), rects in plan.needed.items():
        saved = by_tile_kind.get((tile, kind), [])
        for r in rects:
            remaining = _area(r)
            for s in saved:
                overlap = _intersect(r, s)
                if overlap is not None:
                    remaining -= _area(overlap)
            if remaining != 0:
                raise RingError(
                    f"tile {tile} kind {kind!r}: {remaining} of {_area(r)} "
                    f"cells of {r}, which a later tile reads, are not in the "
                    f"saved bands {saved}. A later tile would read this "
                    "tile's UPDATED values there -- the read-at-time-t bug, "
                    "silent and plausible-looking.")


def field_geometry(home: Mapping[str, Any], *, nz: int | None = None
                   ) -> list[tuple[str, str, int, int]]:
    """``[(name, geometry kind, leading depth, itemsize)]`` for an inventory.

    The same classification :func:`tilestream.gather.build_plan` does, kept in
    one place so the arena and the transport can never disagree about which
    variant a field is.
    """
    layers_ok = nz is not None
    dnz, dny, dnx = _gather.domain_extents(home, nz=nz)
    out = []
    for name, arr in home.items():
        kind = geometry_kind(_gather.classify(arr.shape, dnz, dny, dnx,
                                              layers_ok=layers_ok))
        depth = int(arr.shape[0]) if arr.ndim >= 3 else 1
        out.append((name, kind, depth, int(arr.dtype.itemsize)))
    return out


def arena_bytes(plan: RingPlan, home: Mapping[str, Any], *,
                nz: int | None = None) -> int:
    """Bytes the ring arena needs for ``home``, without allocating anything.

    This is the number that replaces the shadow's ``domain_bytes``, so it is
    what a capacity report must use: host footprint is
    ``store + arena``, not ``2 * store``.
    """
    total = 0
    for _name, kind, depth, itemsize in field_geometry(home, nz=nz):
        total += plan.plane_cells.get(kind, 0) * depth * itemsize
    return int(total)


def arena_bytes_for_manifest(plan: RingPlan, manifest, nz: int, ny: int,
                             nx: int) -> int:
    """Arena bytes from a :mod:`tilestream.hoststore` manifest, not from arrays.

    The capacity form of :func:`arena_bytes`: it answers "how much host RAM
    would the ring need for a domain this size" without allocating the domain
    to find out.
    """
    total = 0
    for spec in manifest:
        shape = spec.shape(nz, ny, nx)
        kind = geometry_kind(_gather.classify(shape, nz, ny, nx,
                                              layers_ok=True))
        depth = int(shape[0]) if len(shape) >= 3 else 1
        total += plan.plane_cells.get(kind, 0) * depth * spec.dtype.itemsize
    return int(total)


def max_square_domain(budget_bytes: int, manifest, nz: int = 49, *,
                      tile: int = 512, halo: int = 16,
                      mode: str = "ring") -> dict:
    """Largest ``n x n x nz`` domain whose HOST FOOTPRINT fits the budget.

    The footprint is the thing that decides the answer, and it is not the
    store:

    ``mode="ring"``
        store + ring arena.  The arena depends on the TILING, so this
        re-plans at every candidate ``n`` rather than scaling a fraction --
        a trailing ragged tile is read right through by its neighbours and
        has to be saved whole, which a fraction would miss.
    ``mode="shadow"``
        two stores.  What :func:`tilestream.driver.run_tiled` needed before
        the ring existed.
    ``mode="store"``
        one store and nothing else: the unreachable bound both are measured
        against.

    Bisected on the exact byte count, like
    :func:`tilestream.hoststore.max_square_domain`, so what it returns always
    fits.  It is NOT necessarily the largest that fits, and the reason is
    worth stating because it is the whole tile-choice story: the footprint is
    not monotone in ``n``.  A domain the tile size divides exactly has no
    ragged tiles and a small ring; one cell more adds a trailing strip that
    its neighbours' windows read right through, so the whole strip has to be
    saved and the arena jumps.  MEASURED at ``nz=49`` dry, tile 1024:
    ``n = 5120`` (5 x 5 tiles, none ragged) needs an arena of 3.14% of the
    store, while nearby ragged sizes need 20-40%.  So this bisects and then
    scans upward for a larger size that happens to fit; and a caller who can
    choose both should choose ``n`` and ``tile`` together, with ``tile``
    dividing ``n``.
    """
    from tilestream import hoststore as _hoststore
    from tilestream import spec as _spec

    manifest = tuple(manifest)
    budget_bytes = int(budget_bytes)
    if mode not in ("ring", "shadow", "store"):
        raise ValueError(f"mode must be 'ring', 'shadow' or 'store', "
                         f"got {mode!r}")

    def footprint(n: int) -> tuple[int, int, float]:
        store = _hoststore.domain_bytes(manifest, nz, n, n)
        if mode == "store":
            return store, 0, 0.0
        if mode == "shadow":
            return store, store, 1.0
        specs = _spec.plan_tiles(n, n, min(tile, n), min(tile, n), halo, True)
        plan = build_ring_plan(specs)
        arena = arena_bytes_for_manifest(plan, manifest, nz, n, n)
        return store, arena, arena / store

    lo, hi = 1, 2
    while sum(footprint(hi)[:2]) <= budget_bytes:
        lo, hi = hi, hi * 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if sum(footprint(mid)[:2]) <= budget_bytes:
            lo = mid
        else:
            hi = mid
    if mode == "ring":
        # The footprint is not monotone in n (see the docstring), so bisection
        # can stop below a size that fits.  Scan up to one tile further; the
        # arena only re-shrinks when a whole new tile row lands.
        ceiling = _hoststore.max_square_domain(budget_bytes, manifest, nz)["nx"]
        candidates = sorted({n for n in range(lo + 1, ceiling + 1)
                             if n % int(tile) == 0} | {ceiling})
        for n in candidates:
            if n > lo and sum(footprint(n)[:2]) <= budget_bytes:
                lo = n
    store, arena, ratio = footprint(lo)
    return {"nx": lo, "ny": lo, "nz": nz, "mode": mode, "tile": min(tile, lo),
            "store_bytes": store, "arena_bytes": arena,
            "total_bytes": store + arena, "arena_over_store": ratio,
            "cells": nz * lo * lo, "budget_bytes": budget_bytes,
            "bytes_per_cell": (store + arena) / float(nz * lo * lo)}


def ring_report(plan: RingPlan, *, bytes_per_cell: float = 1.0) -> dict:
    """Ring size against the shadow it replaces, for one plan.

    ``ring_fraction`` is of ONE mass-grid domain, so ``1 + ring_fraction`` is
    the host store multiplier the scheme needs where the shadow needs 2.0.
    Staggered variants carry their extra row/column, so the fraction is over
    the whole plan's saved cells and can exceed a naive mass-only estimate by
    a little; that is the accurate number.
    """
    s0 = plan.specs[0]
    domain = float(s0.nx * s0.ny)
    cells = plan.cells_by_kind()
    total = float(sum(cells.values()))
    per_kind_domain = float(len(cells)) * domain
    return {
        "tiles": float(plan.ntiles),
        "domain_cells": domain,
        "ring_cells": total,
        "ring_cells_by_kind": {k: float(v) for k, v in cells.items()},
        "bands": float(len(plan.bands)),
        "patches": float(sum(len(p) for p in plan.patches)),
        "ring_fraction": total / per_kind_domain if per_kind_domain else 0.0,
        "shadow_bytes": domain * bytes_per_cell,
        "ring_bytes": total / max(1, len(cells)) * bytes_per_cell,
    }


# --------------------------------------------------------------------------
# storage and transport
# --------------------------------------------------------------------------

class _Blk:
    """One 3-D block copy between a TILE array and an ARENA band.

    The arena side is a permanent view this module owns, so it is held
    directly.  The tile side is held by NAME and resolved at issue time --
    see :class:`_BlockList`.
    """

    __slots__ = ("view", "memkind",
                 "tile_pitch", "tile_xsize", "tile_ysize",
                 "band_pitch", "band_xsize", "band_ysize",
                 "tile_x", "tile_y", "band_x", "band_y",
                 "w_bytes", "height", "depth", "nbytes")

    def __init__(self, view, tile_shape, itemsize, memkind,
                 tile_y0, tile_x0, band_y0, band_x0, height, width, depth):
        self.view = view
        self.memkind = memkind
        self.tile_pitch = int(tile_shape[-1]) * itemsize
        self.tile_xsize = int(tile_shape[-1])
        self.tile_ysize = int(tile_shape[-2])
        self.band_pitch = int(view.shape[-1]) * itemsize
        self.band_xsize = int(view.shape[-1])
        self.band_ysize = int(view.shape[-2])
        self.tile_x = int(tile_x0) * itemsize
        self.tile_y = int(tile_y0)
        self.band_x = int(band_x0) * itemsize
        self.band_y = int(band_y0)
        self.w_bytes = int(width) * itemsize
        self.height = int(height)
        self.depth = int(depth)
        self.nbytes = self.w_bytes * self.height * self.depth


class _BlockList:
    """The copies for one (tile, direction), grouped by field.

    WHY THE TILE POINTER IS RESOLVED AT ISSUE TIME, WHICH IS NOT OPTIONAL
    --------------------------------------------------------------------
    CARRIER ARRAYS ARE NOT IDENTITY-STABLE ACROSS ``dycore.step``.
    ``PhysicsDriver`` REPLACES whole tendency bundles when the scheme that
    owns them runs: MEASURED on a 64x48x49 state, the device pointer behind
    ``driver/pbl_tendencies/*`` changes after EVERY step at ``bldt=0``, and
    ``driver/radiation_tendencies/*``, ``driver/rthraten{lw,sw}`` and
    ``driver/cumulus_tendencies/*`` change on every step at which their
    scheme is due (15 of 153 carriers in the fast-cadence configuration; 0
    of 138 dry or at ``mp10``).

    :func:`tilestream.gather.gather_tile` is immune because it re-takes the
    inventory on every call.  A ring arena that resolved the tile pointers
    once at construction is NOT: it then saves from, and patches into, memory
    the model has stopped using.  That was a real bug in this module, and its
    signature is worth remembering -- the whole window/interior overlap wrong
    (62,720 cells = exactly 16x80x49) in precisely the reallocated carriers,
    at one physics rung out of fourteen, on the second step and not the
    first.

    So the arena side is cached (this module owns those views and never
    reallocates them) and the tile side is looked up by name every time, with
    its shape re-checked.
    """

    __slots__ = ("groups", "direction", "nbytes", "_parms", "_addr")

    def __init__(self, direction: str, groups):
        import ctypes

        self.direction = direction
        self.groups = groups
        self.nbytes = sum(b.nbytes for _n, _s, blocks in groups for b in blocks)
        self._parms = _gather._Memcpy3DParms()
        self._addr = ctypes.addressof(self._parms)

    def __len__(self) -> int:
        return sum(len(blocks) for _n, _s, blocks in self.groups)

    def execute(self, tile_arrays: Mapping[str, Any], stream) -> int:
        import ctypes

        if not self.groups:
            return 0
        rt = _gather._runtime()
        parms = self._parms
        addr = self._addr
        from woof.hip_runtime import memcpy3d_function
        memcpy3d = memcpy3d_function(rt)
        stream_ptr = int(getattr(stream, "ptr", stream))
        pointer = _gather.array_pointer
        to_arena = self.direction == "save"
        for name, shape, blocks in self.groups:
            arr = tile_arrays.get(name)
            if arr is None:
                raise RingError(
                    f"ring {self.direction}: the tile state no longer holds "
                    f"carrier {name!r}")
            if tuple(arr.shape) != shape:
                raise RingError(
                    f"ring {self.direction}: carrier {name!r} is now "
                    f"{tuple(arr.shape)}, was {shape} when the arena was "
                    "built; the ring geometry is no longer valid for it")
            tile_ptr = ctypes.c_void_p(pointer(arr))
            for b in blocks:
                band_ptr = ctypes.c_void_p(pointer(b.view))
                if to_arena:
                    src_ptr, dst_ptr = tile_ptr, band_ptr
                    parms.srcPtr.pitch, parms.srcPtr.xsize, parms.srcPtr.ysize = \
                        b.tile_pitch, b.tile_xsize, b.tile_ysize
                    parms.dstPtr.pitch, parms.dstPtr.xsize, parms.dstPtr.ysize = \
                        b.band_pitch, b.band_xsize, b.band_ysize
                    parms.srcPos.x, parms.srcPos.y = b.tile_x, b.tile_y
                    parms.dstPos.x, parms.dstPos.y = b.band_x, b.band_y
                else:
                    src_ptr, dst_ptr = band_ptr, tile_ptr
                    parms.srcPtr.pitch, parms.srcPtr.xsize, parms.srcPtr.ysize = \
                        b.band_pitch, b.band_xsize, b.band_ysize
                    parms.dstPtr.pitch, parms.dstPtr.xsize, parms.dstPtr.ysize = \
                        b.tile_pitch, b.tile_xsize, b.tile_ysize
                    parms.srcPos.x, parms.srcPos.y = b.band_x, b.band_y
                    parms.dstPos.x, parms.dstPos.y = b.tile_x, b.tile_y
                parms.srcPtr.ptr = src_ptr
                parms.dstPtr.ptr = dst_ptr
                parms.srcPos.z = 0
                parms.dstPos.z = 0
                parms.extent.width = b.w_bytes
                parms.extent.height = b.height
                parms.extent.depth = b.depth
                parms.kind = b.memkind
                memcpy3d(addr, stream_ptr)
        return self.nbytes


class RingArena:
    """Storage for a :class:`RingPlan`, plus the save and patch block lists.

    One flat buffer per field (pinned host or device, matching the store),
    carved into per-band views.  Views are C-contiguous ``(depth, h, w)``
    slices of that buffer, which is what lets the save and the patch be plain
    ``cudaMemcpy3DAsync`` blocks with no repacking.

    The block lists are resolved once per TILE at construction -- not per
    (tile, buffer).  The arena side is a permanent view this module owns; the
    tile side is looked up by name at issue time, because carrier arrays move
    (see :class:`_BlockList`), which also makes one list valid for every
    buffer.
    """

    def __init__(self, plan: RingPlan, home: Mapping[str, Any],
                 tiles: Sequence[Any], *, nz: int | None = None,
                 names=None, inventory_fn=None,
                 allow_pageable: bool = False):
        self.plan = plan
        self.nbuffers = len(tiles)
        take = inventory_fn or _gather.inventory
        tile_inv = [take(t, names) for t in tiles]

        self.fields: list[tuple[str, str, int]] = []      # (name, kind, depth)
        self.buffers: dict[str, Any] = {}
        self.views: dict[str, list[Any]] = {}
        for name, kind, depth, _itemsize in field_geometry(home, nz=nz):
            if kind not in plan.plane_cells:
                raise RingError(
                    f"field {name!r} has geometry {kind!r}, which the ring "
                    f"plan was not built for ({sorted(plan.plane_cells)})")
            self.fields.append((name, kind, depth))

        self._alloc(home, allow_pageable)

        # One block list per tile, not per (tile, buffer): the tile-side
        # pointer is resolved at issue time, so the SAME list serves every
        # buffer -- and, more to the point, serves a buffer whose carriers
        # the driver has reallocated.
        ref = tile_inv[0]
        self.saves = [self._save_blocks(itile, ref)
                      for itile in range(plan.ntiles)]
        self.patches = [self._patch_blocks(itile, ref)
                        for itile in range(plan.ntiles)]
        self.save_bytes = sum(bl.nbytes for bl in self.saves)
        self.patch_bytes = sum(bl.nbytes for bl in self.patches)

    # -- storage --------------------------------------------------------

    def _alloc(self, home, allow_pageable: bool) -> None:
        import cupy as cp

        plan = self.plan
        first = next(iter(home.values()))
        self.arena_on_device = bool(_gather.is_device_array(first))
        for name, kind, depth in self.fields:
            src = home[name]
            if bool(_gather.is_device_array(src)) != self.arena_on_device:
                raise RingError(
                    f"store field {name!r} is on the "
                    f"{'host' if self.arena_on_device else 'device'} while "
                    "the rest of the store is not; the ring arena mirrors "
                    "the store's memory class and cannot be mixed")
            cells = plan.plane_cells[kind] * depth
            if cells == 0:
                self.buffers[name] = None
                self.views[name] = [None] * len(plan.bands)
                continue
            if _gather.is_device_array(src):
                flat = cp.empty(cells, dtype=src.dtype)
            else:
                from tilestream import hoststore as _hoststore

                if not _gather.is_pinned(src) and not allow_pageable:
                    raise _gather.PageableHostMemoryError(
                        f"store array {name!r} is pageable host memory; the "
                        "ring arena mirrors the store's memory class and "
                        "pageable transfers are neither fast nor async")
                flat = _hoststore.alloc_pinned_array((cells,), src.dtype)
            self.buffers[name] = flat
            views: list[Any] = [None] * len(plan.bands)
            for band in plan.bands:
                if band.kind != kind:
                    continue
                start = band.plane_offset * depth
                stop = start + band.cells * depth
                views[band.index] = flat[start:stop].reshape(
                    depth, band.height, band.width)
            self.views[name] = views

    @property
    def nbytes(self) -> int:
        return int(sum(0 if b is None else b.nbytes
                       for b in self.buffers.values()))

    # -- block lists ----------------------------------------------------

    def _group(self, direction: str, ops, tile_arrays):
        # The tile buffer is always device memory; the arena mirrors the
        # store's class (device store -> D2D, pinned host store -> D2H/H2D).
        memkind = (_gather._memcpy_kind(True, self.arena_on_device)
                   if direction == "save"
                   else _gather._memcpy_kind(self.arena_on_device, True))
        groups = []
        for name, kind, depth in self.fields:
            tile_arr = tile_arrays[name]
            shape = tuple(int(s) for s in tile_arr.shape)
            itemsize = int(tile_arr.dtype.itemsize)
            if not _gather.is_device_array(tile_arr):
                raise RingError(
                    f"tile carrier {name!r} is not on the device; the ring "
                    "arena assumes tile buffers are device arrays")
            blocks: list[_Blk] = []
            for op in ops:
                if op.band.kind != kind:
                    continue
                view = self.views[name][op.band.index]
                if direction == "save":
                    blocks.append(_Blk(view, shape, itemsize, memkind,
                                       op.ty0, op.tx0, 0, 0, op.h, op.w, depth))
                else:
                    blocks.append(_Blk(view, shape, itemsize, memkind,
                                       op.ty0, op.tx0, op.by0, op.bx0,
                                       op.h, op.w, depth))
            if blocks:
                groups.append((name, shape, blocks))
        return _BlockList(direction, groups)

    def _save_blocks(self, itile: int, tile_arrays) -> _BlockList:
        return self._group("save", self.plan.saves[itile], tile_arrays)

    def _patch_blocks(self, itile: int, tile_arrays) -> _BlockList:
        return self._group("patch", self.plan.patches[itile], tile_arrays)

    # -- the two operations ---------------------------------------------

    def save(self, itile: int, tile_arrays, stream) -> int:
        """Copy tile ``itile``'s own ring out of the buffer holding it.

        Must be issued after that tile's gather and BEFORE its step: the ring
        has to leave the buffer holding time-t values, and the step overwrites
        them.  It does not need the patch to have run -- a patch only ever
        writes cells belonging to OTHER tiles' interiors, which are disjoint
        from this tile's own.

        ``tile_arrays`` must be a CURRENT inventory of the buffer, taken
        after the previous step it served; see :class:`_BlockList`.
        """
        return self.saves[itile].execute(tile_arrays, stream)

    def patch(self, itile: int, tile_arrays, stream) -> int:
        """Restore time-t values into the parts of the window already rewritten."""
        return self.patches[itile].execute(tile_arrays, stream)

    def free(self) -> None:
        self.buffers.clear()
        self.views.clear()
        self.saves.clear()
        self.patches.clear()
