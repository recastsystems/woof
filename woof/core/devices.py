"""``[devices]``: one forecast split across several cards.

THE SURFACE
-----------
::

    [devices]
    count = 2              # slabs (ranks) the domain is split into
    # grid = "1x2"         # GYxGX; default: the split with the shortest seams
    # ids = [0, 1]         # the card each rank runs on; default 0 .. count-1
    # transport = "auto"   # how halos cross between cards
    # domains = [1, 2]     # a nested tree: which grids are split; default all

``domains`` (a nested tree only) names the grids that run split, each with
the same ``count``/``grid``/``ids``; a grid left out runs resident on the
first card of ``ids``.  Absent, every grid of the tree is split.  The
parent's forcing reaches a split nest through the nest coupler's rolling
tables, windowed per slab, and a split parent serves its nest's FORCE and
takes its feedback through the parent's host store.

``count = 1`` (the default, and what an experiment that never mentions the
table carries) is the one-card run, unchanged: no builder, no wrapper, no
import of the rank machinery.  ``count > 1`` runs the domain as ``count``
permanently resident slabs, each on its own card, stepping at the same time
and swapping a wide halo once per model step (``tilestream.ranks``).

``ids`` may repeat a card.  ``count = 2, ids = [0, 0]`` puts both slabs on
card 0.  That buys no speed and is not meant to: it is the configuration the
bit-exactness proof runs on a one-card machine, because it exercises every
line of the split (the plan, the per-slab build, the seam exchange, the
shared clock, the joined output) except the physical link between two cards.

``transport``
    ``"auto"``   peer-to-peer copies between two cards when the driver says
                 both directions can reach each other
                 (``cudaDeviceCanAccessPeer``), CUDA's own host staging of
                 the same peer copy otherwise.  Decided at run time, per
                 ordered pair of cards, and written in the receipt.
    ``"peer"``   the peer copy, and the run is refused when some pair cannot
                 reach each other directly (a benchmark that must measure the
                 direct path).
    ``"staged"`` the peer-copy call with peer access left off, so CUDA stages
                 it through host memory (the GeForce path, measured faster
                 than hand-rolled staging on 2x RTX 4090).
    ``"host"``   explicit pinned-host staging, device to host then host to
                 device, in two copies.
    Two slabs on the SAME card always use a plain device-to-device copy.

WHAT IT CONTRIBUTES TO THE RESTART IDENTITY: NOTHING
----------------------------------------------------
Same law as ``[tiles]`` (:func:`woof.core.streaming.identity_payload_entry`):
the split is an execution choice whose whole claim is that it changes no
byte, so a checkpoint written on two cards must resume on one and back again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


#: The keys a ``[devices]`` table may carry.
DEVICE_KEYS = frozenset({"count", "grid", "ids", "transport", "domains"})

#: The transports ``transport`` may name; see the module docstring.
DEVICE_TRANSPORTS = ("auto", "peer", "staged", "host")

def parse_grid(value) -> tuple[int, int] | None:
    """``"GYxGX"`` (or a two-item list) as ``(gy, gx)``; ``None`` passes."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = value.lower().replace(" ", "").split("x")
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise ValueError(
            f"[devices] grid = {value!r} must be a string 'GYxGX' such as "
            "'1x2' (two slabs side by side in x) or '2x1' (stacked in y)")
    if len(parts) != 2:
        raise ValueError(
            f"[devices] grid = {value!r} must name exactly two extents, "
            "'GYxGX'")
    if any(isinstance(part, bool) or isinstance(part, float) for part in parts):
        raise ValueError(f"[devices] grid = {value!r} must contain whole extents; "
                         "truncation would choose a different split")
    try:
        gy, gx = int(parts[0]), int(parts[1])
    except (TypeError, ValueError):
        raise ValueError(
            f"[devices] grid = {value!r} is not two whole numbers") from None
    if gy < 1 or gx < 1:
        raise ValueError(f"[devices] grid = {value!r} must be positive")
    return gy, gx


@dataclass(frozen=True)
class DeviceOptions:
    """One validated ``[devices]`` table."""

    count: int = 1
    grid: tuple[int, int] | None = None
    ids: tuple[int, ...] | None = None
    transport: str = "auto"
    domains: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise ValueError(
                f"[devices] count = {self.count!r} must be a whole number")
        if self.count < 1:
            raise ValueError(
                f"[devices] count = {self.count} must be at least 1")
        grid = parse_grid(self.grid)
        object.__setattr__(self, "grid", grid)
        if self.ids is not None:
            if (not isinstance(self.ids, (list, tuple))
                    or any(isinstance(v, bool) or not isinstance(v, int)
                           for v in self.ids)):
                raise ValueError("[devices] ids must be whole card numbers; "
                                 "fractional ids would select the wrong card")
            ids = tuple(self.ids)
            if any(v < 0 for v in ids):
                raise ValueError(
                    f"[devices] ids = {list(ids)} names a negative card")
            object.__setattr__(self, "ids", ids)
        if self.transport not in DEVICE_TRANSPORTS:
            raise ValueError(
                f"[devices] transport = {self.transport!r} is not one of "
                f"{list(DEVICE_TRANSPORTS)}")
        if self.domains is not None:
            if (not isinstance(self.domains, (list, tuple)) or not self.domains
                    or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                           for v in self.domains)):
                raise ValueError(
                    f"[devices] domains = {self.domains!r} must be a non-empty "
                    "list of grid ids; anything else would split a grid the "
                    "list does not name")
            if len(set(self.domains)) != len(self.domains):
                raise ValueError(f"[devices] domains = {list(self.domains)} "
                                 "names a grid twice")
            object.__setattr__(self, "domains", tuple(int(v) for v in self.domains))
        if self.count == 1:
            # A SURFACE THAT IS OFF MUST BE EMPTY, the [tiles] and
            # [relocation] discipline.  The breakage it prevents: a table
            # inherited with grid/ids/transport set and count flipped to 1
            # somewhere else would read as a split that is not running.
            stray = [k for k in ("grid", "ids", "domains")
                     if getattr(self, k) is not None]
            if self.transport != "auto":
                stray.append("transport")
            if stray:
                raise ValueError(
                    f"[devices] sets {', '.join(stray)} while count = 1.  "
                    "A one-card run has no split to describe; set count "
                    "above 1 or delete the key(s).")
            return
        if grid is not None and grid[0] * grid[1] != self.count:
            raise ValueError(
                f"[devices] grid = '{grid[0]}x{grid[1]}' makes "
                f"{grid[0] * grid[1]} slabs but count = {self.count}")
        if self.ids is not None and len(self.ids) != self.count:
            raise ValueError(
                f"[devices] ids = {list(self.ids)} names {len(self.ids)} "
                f"card(s) for count = {self.count} slabs; give one card per "
                "slab (a card may repeat)")

    @property
    def enabled(self) -> bool:
        """``count > 1``.  A one-card run is not a split."""
        return self.count > 1

    def device_ids(self) -> tuple[int, ...]:
        """The card each rank runs on, rank order (row-major over the grid)."""
        return self.ids if self.ids is not None else tuple(range(self.count))

    def split_grid_ids(self, grid_ids) -> tuple[int, ...]:
        """Which of a tree's ``grid_ids`` run split, in the tree's order.

        Every grid when ``domains`` is absent; a ``domains`` entry the tree
        does not have is refused, because a typo would otherwise run that
        grid resident with nothing said.
        """
        grid_ids = tuple(int(g) for g in grid_ids)
        if not self.enabled:
            return ()
        if self.domains is None:
            return grid_ids
        unknown = sorted(set(self.domains) - set(grid_ids))
        if unknown:
            raise ValueError(
                f"[devices] domains = {list(self.domains)} names grid(s) "
                f"{unknown} this experiment does not have (grids "
                f"{list(grid_ids)}); refused rather than running them resident")
        return tuple(g for g in grid_ids if g in self.domains)

    def resolved_grid(self, nx: int, ny: int) -> tuple[int, int]:
        """``(gy, gx)`` for an ``nx x ny`` domain.

        The configured grid when there is one.  Otherwise the factor pair of
        ``count`` whose seams are shortest in total, ``(gx - 1) * ny +
        (gy - 1) * nx`` mass columns, which is the halo traffic and the halo
        recompute both; a tie goes to the one-dimensional split, which has one
        exchange phase instead of two.  On HRRR's 1799 x 1059 grid two slabs
        split x (a 1059-row seam, not a 1799-column one).
        """
        if self.grid is not None:
            return self.grid
        best = None
        for gy in range(1, self.count + 1):
            if self.count % gy:
                continue
            gx = self.count // gy
            seam = (gx - 1) * int(ny) + (gy - 1) * int(nx)
            phases = int(gx > 1) + int(gy > 1)
            key = (seam, phases)
            if best is None or key < best[0]:
                best = (key, (gy, gx))
        return best[1]

    @classmethod
    def from_mapping(cls, table: Any, *, source: str = "<config>"
                     ) -> "DeviceOptions":
        """Validate one ``[devices]`` table.  ``None`` gives :data:`DEVICES_OFF`."""
        if table is None:
            return DEVICES_OFF
        if not isinstance(table, dict):
            raise ValueError(
                f"[devices] of {source} must be a table, got {table!r}")
        unknown = sorted(set(table) - DEVICE_KEYS)
        if unknown:
            raise ValueError(
                f"unknown key(s) {unknown} in [devices] of {source}; known "
                f"keys: {sorted(DEVICE_KEYS)}")
        return cls(**table)

    def to_mapping(self) -> dict[str, object]:
        """The TOML form: only what is set, ``grid`` as ``"GYxGX"``."""
        out: dict[str, object] = {"count": int(self.count)}
        if self.grid is not None:
            out["grid"] = f"{self.grid[0]}x{self.grid[1]}"
        if self.ids is not None:
            out["ids"] = list(self.ids)
        if self.transport != "auto":
            out["transport"] = self.transport
        if self.domains is not None:
            out["domains"] = list(self.domains)
        return out

    def to_json(self) -> dict[str, object]:
        """The receipt form."""
        out = {"count": int(self.count),
               "grid": None if self.grid is None
               else f"{self.grid[0]}x{self.grid[1]}",
               "ids": list(self.device_ids()),
               "transport": self.transport}
        if self.domains is not None:
            out["domains"] = list(self.domains)
        return out


#: The one-card contract, as one shared object.  An experiment that never
#: mentions ``[devices]`` carries THIS.
DEVICES_OFF = DeviceOptions()


def identity_payload_entry(options: "DeviceOptions | None") -> dict:
    """What ``[devices]`` contributes to the restart identity: NOTHING.

    See the module docstring.  A checkpoint written on two cards resumes on
    one and back again, because the split changes no byte.
    """
    return {}


__all__ = ["DEVICE_KEYS", "DEVICE_TRANSPORTS", "DEVICES_OFF", "DeviceOptions",
           "identity_payload_entry", "parse_grid"]


class DevicesRefused(ValueError):
    """An execution request that this route cannot safely honor."""


def validate_ranked_physics(cfg):
    """Refuse physics whose full-domain state the rank factory cannot build."""
    grid = int(getattr(cfg, "grid_id", 0))
    chem_sets = getattr(cfg, "chem_sets", "")
    # Only profiles with a tested domain ledger owner can run on ranks.
    from tilestream.sfire_smoke import ranked_accepts
    fire_smoke_only = bool(chem_sets) and ranked_accepts(cfg)
    if chem_sets and not fire_smoke_only:
        raise DevicesRefused(
            f"[devices] grid_id = {grid} enables chem_sets: chemistry carries "
            "per-domain (species,) ledger vectors and reduces mass over the "
            "complete domain. These chemistry profiles have no qualified rank ledger "
            "owner, so halo copies would duplicate mass accounting and the "
            "gather would slice those vectors as grid fields. Run this grid "
            "resident by leaving it out of [devices] domains.")
    if int(getattr(cfg, "slope_rad", 0) or 0) == 1:
        raise DevicesRefused(
            f"[devices] grid_id = {grid} sets slope_rad = 1: the shadow "
            "search reads full-domain terrain up to shadlen and holds "
            "radiation-time state that resident slabs do not carry. "
            "Run this grid resident (leave it out of [devices] domains), "
            "or set slope_rad = 0.")
    if int(getattr(cfg, "sf_surface_mosaic", 0) or 0) == 1:
        raise DevicesRefused(
            f"[devices] grid_id = {grid} sets sf_surface_mosaic = 1: the "
            "rank factory does not build Noah's land-use tiles from LANDUSEF, "
            "so its first Noah step would have no mosaic state. Run this "
            "grid resident (leave it out of [devices] domains), or set "
            "sf_surface_mosaic = 0.")


def validate_device_road(options, tiles=None, domains=()):
    if not options.enabled:
        return
    if (getattr(tiles, "enabled", False)
            or any(getattr(getattr(dc, "tiles", None), "enabled", False)
                   for dc in domains)):
        raise DevicesRefused(
            "[devices] count > 1 beside enabled [tiles] is refused: one domain "
            "has one road; two stores and two planners would each price a "
            "domain the other runs")
    if domains:
        try:
            split = options.split_grid_ids(dc.grid_id for dc in domains)
        except ValueError as error:
            raise DevicesRefused(str(error)) from error
        if not split:
            raise DevicesRefused(
                f"[devices] domains = {list(options.domains)} splits no grid of "
                "this experiment while count > 1 says the run is split")
        for dc in domains:
            if int(dc.grid_id) in split:
                validate_ranked_physics(getattr(dc, "run", dc))


def validate_tree_devices(exp):
    """The grids a split TREE runs split, after its by-name refusals.

    Wired and proven (byte-identical to the unsplit tree on a two-domain
    template, one card split in slabs): a split parent forcing a split or
    resident nest each parent step, and two-way feedback into a split
    parent.  Refused here, before anything restores, each for the breakage
    it would otherwise produce:

    * ``[relocation]`` (a moving or following nest): the move re-initializes
      the nest from a full SINT of the live parent and rebuilds its domain;
      nothing rebuilds a split nest's slabs, so they would keep stepping the
      old position.
    * a split grid that starts after the run does: its birth restore rebuilds
      a ``[tiles]`` store domain (``_restore_streamed_child_at_start``), not
      slabs, so the grid would come up unsplit on one card.
    * a split parent with a resident nest: the nest coupler serves a resident
      nest from its parent's resident device arrays, and a split parent has
      none (its arrays are host store arrays).  MEASURED on the SF tree with
      domains = [1]: the first FORCE stopped on a numpy array handed to a
      kernel.  Split the nest too, or leave the parent whole.
    """
    options = getattr(exp, "devices", DEVICES_OFF)
    if not options.enabled:
        return ()
    domains = tuple(exp.domains)
    split = options.split_grid_ids(dc.grid_id for dc in domains)
    for dc in domains:
        if int(dc.grid_id) in split:
            validate_ranked_physics(dc.run)
    if len(domains) < 2:
        return split
    if getattr(getattr(exp, "relocation", None), "enabled", False):
        raise DevicesRefused(
            "[devices] on a tree with [relocation] is refused: a moving nest is "
            "re-initialized from a full SINT of its live parent and its domain "
            "rebuilt at each move, and nothing rebuilds a split grid's slabs, so "
            "they would keep stepping the old position")
    resident_under_split = [int(dc.grid_id) for dc in domains
                            if int(dc.parent_id) in split
                            and int(dc.grid_id) not in split]
    if resident_under_split:
        raise DevicesRefused(
            f"[devices] splits the parent of resident grid(s) {resident_under_split}: "
            "the nest coupler serves a resident nest from its parent's resident "
            "device arrays, and a split parent has only host store arrays, so the "
            "first FORCE would stop; split the nest too (add it to domains) or "
            "leave the parent whole")
    start = getattr(exp, "start_time", None)
    timing = getattr(exp, "domain_start_time", None)
    if timing is not None:
        late = [gid for gid in split if timing(gid) != start]
        if late:
            raise DevicesRefused(
                f"[devices] splits grid(s) {late} that start after the run does: "
                "a late grid's birth restore rebuilds a [tiles] store domain, not "
                "slabs, so it would come up unsplit on one card; leave it out of "
                "[devices] domains to run it resident")
    return split


def describe_split(exp, options=None, *, halo=None):
    """One sentence for what a split does: the doors and the check print it.

    A single domain reads as it always has (count, cards, grid, transport).
    A tree names each grid that runs split with its own slab grid, after
    the tree's by-name refusals (:func:`validate_tree_devices`), and says
    which grids stay resident on the first card.
    """
    options = getattr(exp, "devices", DEVICES_OFF) if options is None else options
    head = f"[devices] {options.count} slabs on cards {list(options.device_ids())}"
    domains = tuple(exp.domains)
    if len(domains) == 1:
        gy, gx = options.resolved_grid(exp.root.run.nx, exp.root.run.ny)
        tail = "" if halo is None else f", halo {halo}"
        return f"{head}, grid {gy}x{gx}{tail}, transport {options.transport}"
    from dataclasses import replace
    split = validate_tree_devices(replace(exp, devices=options))
    parts = []
    for dc in domains:
        gid = int(dc.grid_id)
        if gid in split:
            gy, gx = options.resolved_grid(dc.run.nx, dc.run.ny)
            width = (f", halo {halo[gid]}" if isinstance(halo, dict) and gid in halo
                     else "")
            parts.append(f"d{gid:02d} split {gy}x{gx}{width}")
        else:
            parts.append(f"d{gid:02d} resident on card {options.device_ids()[0]}")
    return f"{head}, transport {options.transport}; " + ", ".join(parts)


def refuse_unrouted_devices(exp, route):
    options = getattr(exp, "devices", DEVICES_OFF)
    if options.enabled:
        raise DevicesRefused(
            f"{route} does not read [devices] count = {options.count}: a nested "
            "tree or this route is not wired into the ranked road. Refused to "
            "prevent a silent one-card run dying at the allocation the split "
            "exists to avoid")


def validate_device_count(options, visible_count):
    ids = options.device_ids()
    if options.enabled and any(dev >= visible_count for dev in ids):
        raise DevicesRefused(
            f"[devices] ids = {list(ids)} but visible card count = {visible_count}; "
            "refused before allocation on a nonexistent card")


def override_device_count(options, count, *, log=print):
    if count is None:
        return options
    from dataclasses import replace
    try:
        updated = replace(options, count=count)
    except ValueError as error:
        raise DevicesRefused(
            f"--devices {count} contradicts [devices] {options.to_mapping()}: "
            f"{error}; refused to prevent a split running on a different plan") from error
    log(f"--devices {count} replaces [devices] count = {options.count}; "
        "the flag is the later statement")
    return updated


_DEVICE_PROBE = """import json
import cupy as cp
count = cp.cuda.runtime.getDeviceCount()
rows = {}
for dev in range(count):
    with cp.cuda.Device(dev):
        free, total = cp.cuda.runtime.memGetInfo()
        props = cp.cuda.runtime.getDeviceProperties(dev)
        name = props["name"]
        rows[dev] = {"free_bytes": int(free), "total_bytes": int(total),
                     "profile": {"name": name.decode() if isinstance(name, bytes) else str(name),
                         "multiprocessor_count": int(props["multiProcessorCount"]),
                         "max_threads_per_multiprocessor": int(props["maxThreadsPerMultiProcessor"]),
                         "default_stack_limit_bytes": (0 if cp.cuda.runtime.is_hip else int(cp.cuda.runtime.deviceGetLimit(0)))}}
print(json.dumps({"visible_count": count, "cards": rows}))
"""


def probe_devices():
    """Read visible count and each card's free memory in a short-lived process."""
    import json
    import subprocess
    import sys
    from woof.local_gpu import no_local_gpu
    if no_local_gpu():
        return None
    completed = subprocess.run([sys.executable, "-c", _DEVICE_PROBE],
                               capture_output=True, text=True, timeout=60)
    if completed.returncode:
        raise DevicesRefused(
            "[devices] card probe failed before download/allocation: "
            + completed.stderr.strip())
    return json.loads(completed.stdout)

__all__ += ["DevicesRefused", "validate_device_road", "refuse_unrouted_devices",
            "validate_device_count", "override_device_count", "probe_devices",
            "validate_tree_devices"]
