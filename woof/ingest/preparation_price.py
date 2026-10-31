"""The device price of a CUDA preparation, known before its first allocation.

A CUDA preparation builds, for each forcing time, a horizontally
interpolated analysis, the vertical-interpolation plans and outputs, and
then one complete :class:`woof.core.state.DomainState`.  Nothing asked
whether that fitted the card before allocating it, so a 1792x1024x55 GFS
preparation on a 24 GB card decoded its inputs, built its statics and then
stopped about two minutes later with a raw CuPy out-of-memory inside
``DomainState.__init__``.  This module prices that preparation from the
same inventories the allocations are made from, per route, so every door
can decide before the first device allocation. The mapped route stages CUDA
batches and retains completed arrays on the host when the whole build would
take more than its 1 GiB staging pool, leaving room for a concurrent forecast.
Other routes choose the CPU under ``auto`` when their price does not
fit, and refuse an explicit ``cuda`` request by name.

No GPU imports: the price is arithmetic over shapes.

THE TERMS, per domain being built (bytes, float32):

* ``model_state``: the exact
  :func:`woof.core.device_inventory.state_array_shapes` inventory
  (83 arrays at mp=8 with terrain).
* ``forcing_analysis``: the horizontally interpolated source fields on the
  target grid, ``level_fields x source_levels`` level planes plus the
  single-level planes, from the decoded inventory where the door has it.
* ``vertical_setup``: the mass and U/V vertical plans,
  ``(source_levels + nz) x (C + U + V)``, and the vertical outputs,
  ``nz x (mass_outputs x C + U + V)``, with ``mass_outputs`` the source's
  mass-point level fields (temperature, humidity, pressure and every
  hydrometeor the source carries), never fewer than three.
* ``real_columns``: source doubles, column thermodynamics and staggered
  pressure workspaces for the specific-humidity device dispatch. Allocation
  hooks measured 12,371,798,528 bytes live and 13,114,399,232 reserved on
  sm_120 at 800 x 600 x 50; this term conservatively covers that peak.
* ``setup_residual``: :data:`SETUP_RESIDUAL` minus one, times the analysis
  and vertical setup, for the temporaries the itemization does not name.
* ``pool_headroom``: :data:`PREPARATION_POOL_HEADROOM` minus one, times
  everything live: what the CuPy pool reserves beyond what is referenced.
* ``cuda_context``: the card's CUDA context, outside the pool.
* ``boundary_tables``: the root's float32 lateral-boundary tables, on the
  routes that attach them on the card before the children are built.
* ``worker_slots``: native HRRR's spawned boundary workers, each with its
  own context and one side strip being built.
* ``source_transform``: the FP64 humidity conversion ERA5 runs on the whole
  SOURCE grid, which no target-shaped term can see.

The ROUTE decides which of those coexist (:data:`PREPARATION_ROUTES`, a
table: a new route is a row, not a code path).  The price is the largest
phase plus the context.

CALIBRATION (:data:`MEASURED_PREPARATION_PEAKS`): four CUDA preparations on
an otherwise empty H100 80 GB, HRRR pressure-level input, 7 forcing times,
pool high-water read after every allocation.  Live non-state memory at the
peak was 1.005 to 1.006 times the itemized analysis and vertical setup on
all three builds (1792x1024x55, 896x512x59 and the 480x480x59 child), and
the pool reserved 1.10 to 1.18 times what was live.  The price carries
1.10 on the first and 1.20 on the second.  THE MARGIN, and why: with the
Linux context the price sits 4.7% over the measured card peak on the
tightest case (the 3 km root, where the pool's 1.18 is closest to 1.20)
and 8% to 11% over the others (the nests carry every child's setup at
once, an upper bound for a child built one at a time).  The residual's
10% is kept for sources this run did not measure: the 24 GB failure (GFS,
22.92 GiB reserved with 1.95 GiB of the state still to allocate) held
1.17 times its itemization reserved, inside 1.10 x 1.20, and is priced at
30.2 GiB.  Tests hold the price at or above every measured peak and
within a quarter of it.

THE OTHER ROUTES (:data:`MEASURED_ROUTE_PEAKS`): the routes whose terms
the mapped runs never exercised were each run once on a 16 GB card at a
reference shape, and tests hold each route's price at or above its own
peak.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType

GIB = 1024 ** 3

#: Live bytes the setup of one build holds beyond its itemized analysis and
#: vertical plans/outputs, as a multiple of them.  MEASURED 1.005 to 1.006
#: (see the module docstring); 1.10 leaves the margin for sources whose
#: interpolation temporaries differ from the measured one (the GFS failure
#: held 1.17 times its itemization RESERVED, which this term times
#: :data:`PREPARATION_POOL_HEADROOM` covers at 1.32).
#: This margin also holds the preparation-owned regular plans: four FP32
#: coordinate planes per staggering, plus a receipt's integer magnitude,
#: masks and packed bits. The focused inventory test bounds them in this term.
SETUP_RESIDUAL = 1.10

#: CuPy pool bytes reserved per byte live at a preparation's peak.
#: MEASURED 1.183 (1792x1024x55), 1.153 (896x512x59) and 1.096 (the nested
#: child build).  Deliberately not the forecast's ``ALLOCATOR_HEADROOM``
#: (1.15), which the 3 km preparation exceeded.
PREPARATION_POOL_HEADROOM = 1.20

#: Bytes per SOURCE point per source level the ERA5 humidity conversion
#: holds on the card: eight FP64 working arrays plus the FP32 input and
#: output (``woof/ingest/horiz.py`` era5_rh_to_water).
FP64_HUMIDITY_BYTES_PER_SOURCE_POINT_LEVEL = 8 * 8 + 2 * 4

#: What the price was calibrated on, printed beside it in every receipt.
PREPARATION_PRICE_BASIS = (
    "itemized from the model-state, analysis and vertical-setup inventories; "
    f"setup residual x{SETUP_RESIDUAL:g} and pool headroom "
    f"x{PREPARATION_POOL_HEADROOM:g} fitted to four measured CUDA "
    "preparations (1792x1024x55, 896x512x59, a 3:1 nest and a tiled nest, "
    "H100 80 GB, 2026-09-28)")


@dataclass(frozen=True)
class SourceInventory:
    """What one forcing time of a source puts on the card before the state.

    ``levels`` is the source's vertical level count, ``level_fields`` the
    number of fields on those levels (winds included), ``surface_planes``
    every single-level plane (a stacked soil field counts one plane per
    layer).  ``source_points`` is the SOURCE grid's horizontal size, known
    only once decoded; 0 is unknown, and prices no source-grid transform.
    """

    levels: int
    level_fields: int
    surface_planes: int
    source_points: int = 0
    fp64_humidity_transform: bool = False
    hydrometeor_replays: bool | None = None
    device_real_columns: bool = False

    def __post_init__(self):
        for name in ("levels", "level_fields", "surface_planes",
                     "source_points"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) \
                    or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.levels < 1:
            raise ValueError("a source has at least one level")
        if self.hydrometeor_replays is None:
            # Shape-only nominal inventories above the six atmospheric
            # profiles conservatively include possible analyzed mass.
            object.__setattr__(self, "hydrometeor_replays", self.level_fields > 6)

    @property
    def mass_outputs(self) -> int:
        """Mass-point fields the vertical interpolation writes at nz.

        Every level field except the two winds; temperature, moisture and
        pressure are always written.  The height field counted here is not
        written (it makes the plan), so this is an upper bound by one field,
        which the calibration carries.
        """
        return max(3, self.level_fields - 2)

    @classmethod
    def from_shapes(cls, shapes: Mapping[str, Sequence[int]], *,
                    source_points: int = 0,
                    fp64_humidity_transform: bool = False
                    ) -> "SourceInventory":
        """Count an inventory of named 2-D/3-D shapes.

        The deepest 3-D stack sets the level count; every 3-D field of that
        depth is a level field, and every other plane (surface fields,
        stacked soil layers) a surface plane.
        """
        dims = [tuple(int(n) for n in shape) for shape in shapes.values()
                if len(tuple(shape)) in (2, 3)]
        if not dims:
            raise ValueError("an inventory needs at least one 2-D or 3-D field")
        levels = max((shape[0] for shape in dims if len(shape) == 3),
                     default=1)
        level_fields = sum(1 for shape in dims
                           if len(shape) == 3 and shape[0] == levels
                           and levels > 1)
        planes = sum((shape[0] if len(shape) == 3 else 1) for shape in dims
                     if not (len(shape) == 3 and shape[0] == levels
                             and levels > 1))
        from woof.mapped_source import HYDROMETEOR_LEGACY_NAMES
        mass_names = {name.upper() for name in HYDROMETEOR_LEGACY_NAMES}
        mass_names.update(HYDROMETEOR_LEGACY_NAMES.values())
        mass_names.add("QH")
        hydrometeor_replays = any(name.upper() in mass_names and len(shape) == 3
                                 for name, shape in shapes.items())
        return cls(levels=levels, level_fields=level_fields,
                   surface_planes=planes, source_points=int(source_points),
                   fp64_humidity_transform=bool(fp64_humidity_transform),
                   hydrometeor_replays=hydrometeor_replays,
                   device_real_columns=(
                       {"PRES", "SPFH", "Q2"}.issubset(shapes)
                       or {"air_pressure", "specific_humidity",
                           "specific_humidity_2m"}.issubset(shapes)))

    @classmethod
    def from_snapshot(cls, snapshot, *, fp64_humidity_transform: bool = False
                      ) -> "SourceInventory":
        """The inventory of one decoded forcing snapshot (``fields`` map)."""
        shapes = {name: tuple(getattr(value, "shape", ()))
                  for name, value in snapshot.fields.items()}
        horizontal = [shape[-2:] for shape in shapes.values()
                      if len(shape) in (2, 3)]
        points = max((int(a) * int(b) for a, b in horizontal), default=0)
        return cls.from_shapes(shapes, source_points=points,
                               fp64_humidity_transform=fp64_humidity_transform)


#: Nominal inventories for the doors that price before a decode exists.
#: Native HRRR: the 50 hybrid levels and the bridge's atmosphere tuple
#: (``woof/ingest/hrrr.py`` ``_ATMOSPHERE_3D``, 11 fields), eleven surface
#: planes plus nine soil nodes of temperature and moisture.
NOMINAL_SOURCE_INVENTORIES = MappingProxyType({
    "hrrr-native": SourceInventory(levels=50, level_fields=11,
                                   surface_planes=11 + 2 * 9),
})


@dataclass(frozen=True)
class PreparationRoute:
    """One preparation route's residency, as a row.

    ``held_analyses``: whole forcing-time analyses held beside the build
    (the experiment route keeps the last time's for its caller).
    ``state_on_card``: False where the state is built in host memory and
    only the transforms run on the card.  ``children``: ``"together"``
    when the root's start time, its analysis and its boundary tables stay
    on the card while every child is built and held for one export,
    ``"one_at_a_time"`` when each domain is released before the next,
    ``None`` for a single-domain route.  ``boundary_workers``: native
    HRRR's spawned per-hour boundary processes.  ``tables_on_root``: the
    root's boundary tables are attached on the card while its own build
    is still held (met_em and the experiment route).  ``stage``: where
    the preparation would stop if started anyway, for the refusal.
    ``pool_headroom``: the CuPy pool's reserve per itemized live byte on
    a route whose card measurement (:data:`MEASURED_ROUTE_PEAKS`) reserved
    more than :data:`PREPARATION_POOL_HEADROOM` allows; ``None`` takes it.
    """

    name: str
    stage: str
    held_analyses: int = 0
    state_on_card: bool = True
    children: str | None = "together"
    boundary_workers: bool = False
    tables_on_root: bool = False
    door: str = ""
    pool_headroom: float | None = None

    @property
    def headroom(self) -> float:
        return (PREPARATION_POOL_HEADROOM if self.pool_headroom is None
                else float(self.pool_headroom))


#: THE ROUTE TABLE.  A route prepares on the card only through a row here.
PREPARATION_ROUTES = MappingProxyType({row.name: row for row in (
    PreparationRoute(
        "gfs", "while building the root forcing states",
        door="woof prep --source gfs (woof/gfs_direct.py)"),
    PreparationRoute(
        "era5", "while building the root forcing states",
        door="woof prep --source era5/era5-arco (woof/era5_direct.py)"),
    PreparationRoute(
        "mapped", "while building the root forcing states",
        door="woof prep for mapped sources (woof/mapped_direct.py)"),
    PreparationRoute(
        "met_em", "while building a domain from its met_em files",
        children="one_at_a_time", tables_on_root=True,
        door="woof met_em preparation (woof/metem_forecast.py)"),
    PreparationRoute(
        "hrrr-native", "while building the f00 state or its boundary strips",
        children=None, boundary_workers=True,
        door="native HRRR (tools/hrrr_single_domain_benchmark.py)"),
    # Pool headroom 1.25: at 12 km 500x400x49 from ERA5 the pool reserved
    # 3.907 GB against 3.211 GB itemized (1.217), 1.4% past what 1.20
    # priced (MEASURED_ROUTE_PEAKS).
    PreparationRoute(
        "experiment", "while building the case's forcing states",
        held_analyses=1, children=None, tables_on_root=True,
        door="woof run with [case_data] (woof/runtime.py)",
        pool_headroom=1.25),
    PreparationRoute(
        "experiment-host-store",
        "while interpolating the case's forcing times",
        state_on_card=False, children=None, tables_on_root=False,
        door="woof run with [case_data] and a host store (woof/runtime.py)"),
    PreparationRoute(
        "downscale-child",
        "while interpolating the parent frames onto the child",
        children=None,
        door="woof downscale (woof/offline_child_run.py)"),
    PreparationRoute(
        "nest-activation",
        "while re-initializing the nest from the analysis at its start",
        children=None,
        door="woof run activating a delayed nest (woof/core/model.py)"),
)})


def route_pool_headroom(route: str | None) -> float:
    """The pool headroom a preparation on ``route`` is priced with.

    ``None`` is the calibrated :data:`PREPARATION_POOL_HEADROOM`; a named
    route reads its row.  Generic callers ask here instead of importing the
    route table, whose row names are source names.
    """
    if route is None:
        return PREPARATION_POOL_HEADROOM
    row = PREPARATION_ROUTES.get(route)
    if row is None:
        raise ValueError(
            f"no preparation route {route!r}; the routes are "
            f"{', '.join(sorted(PREPARATION_ROUTES))}")
    return row.headroom


@dataclass(frozen=True)
class PreparationDevicePrice:
    """What a CUDA preparation needs on the card, itemized."""

    route: str
    need_bytes: int
    terms: Mapping[str, int]
    phase: str
    basis: str = PREPARATION_PRICE_BASIS
    phases: Mapping[str, int] = field(default_factory=dict)
    #: Minimum staging capacity for a full column row plus source-plane donors.
    chunk_minimum_bytes: int = 0
    chunk_source_bytes: int = 0

    @property
    def stage(self) -> str:
        row = PREPARATION_ROUTES.get(self.route)
        return "while preparing" if row is None else row.stage

    def summary(self) -> str:
        """``model state 19.7, forcing analysis 0.9, ...`` in GiB."""
        labels = (("model_state", "model state"),
                  ("forcing_analysis", "forcing analysis"),
                  ("vertical_setup", "vertical setup"),
                  ("real_columns", "device REAL columns"),
                  ("parent_fields", "parent fields"),
                  ("child_fields", "child fields"),
                  ("setup_residual", "setup temporaries"),
                  ("physics", "physics"),
                  ("boundary_tables", "boundary tables"),
                  ("source_transform", "source-grid transform"),
                  ("worker_slots", "boundary workers"),
                  ("pool_headroom", "allocator headroom"),
                  ("cuda_context", "CUDA context"))
        return ", ".join(f"{text} {self.terms[key] / GIB:.1f}"
                         for key, text in labels if self.terms.get(key))

    def record(self) -> dict:
        return {"route": self.route, "need_bytes": int(self.need_bytes),
                "phase": self.phase,
                "terms": {key: int(value) for key, value in self.terms.items()},
                "phases": {key: int(value) for key, value in self.phases.items()},
                "basis": self.basis}


def _columns(cfg) -> tuple[int, int, int]:
    ny, nx = int(cfg.ny), int(cfg.nx)
    return ny * nx, ny * (nx + 1), (ny + 1) * nx


def state_bytes(cfg) -> int:
    """The exact ``DomainState`` inventory of ``cfg``, in bytes."""
    from woof.core.device_inventory import state_array_shapes

    return sum(4 * math.prod(shape)
               for shape in state_array_shapes(cfg).values())


def analysis_bytes(cfg, inventory: SourceInventory) -> int:
    """One forcing time's analysis on ``cfg``'s grid (winds on the widest stagger)."""
    mass, u, v = _columns(cfg)
    widest = max(mass, u, v)
    return 4 * (inventory.level_fields * inventory.levels * widest
                + inventory.surface_planes * mass)


def vertical_setup_bytes(cfg, inventory: SourceInventory) -> int:
    """Vertical plans plus vertical outputs of one build."""
    mass, u, v = _columns(cfg)
    nz = int(cfg.nz)
    plans = 4 * (inventory.levels + nz) * (mass + u + v)
    outputs = 4 * nz * (inventory.mass_outputs * mass + u + v)
    return plans + outputs


def boundary_table_bytes(cfg, intervals: int, *, boundary_species=()) -> int:
    """The root's float32 value+tendency tables for ``intervals`` intervals.

    ``boundary_species`` is the source's published hydrometeor inventory,
    whose masses and seeded numbers the tables then carry too.
    """
    if intervals <= 0 or not bool(getattr(cfg, "specified", False)):
        return 0
    from woof.core.device_inventory import lbc_interval_values

    return 4 * lbc_interval_values(
        cfg, boundary_species=boundary_species) * int(intervals)


def context_bytes(*, profile=None, vram_gib: float | None = None,
                  platform: str | None = None) -> int:
    """The card's CUDA context, per process that holds one.

    The profile's context (0.75 GiB) covers the 0.66 GB every measured
    preparation held outside the pool.  The forecast projection's Windows
    device-overhead constant is NOT added: it was fitted to a forecast
    process on one Windows card, and charged once per native HRRR worker it
    priced a 2-worker preparation of a small test domain at 6.4 GiB, which
    a desktop card with 5.5 GiB free prepares.  A refusal must not fire on
    a preparation that completes.  ``vram_gib`` and ``platform`` are kept
    for callers that price a named card.
    """
    from woof.core.device_inventory import MEASURED_LOCAL_MEMORY_PROFILE

    del vram_gib, platform
    chosen = MEASURED_LOCAL_MEMORY_PROFILE if profile is None else profile
    return int(chosen.cuda_context_bytes)


def _build(cfg, inventory: SourceInventory, *, state_on_card=True):
    device_columns = inventory.device_real_columns and int(cfg.mp_physics) != 28
    state = state_bytes(cfg) if state_on_card else 0
    analysis = analysis_bytes(cfg, inventory)
    setup = vertical_setup_bytes(cfg, inventory)
    residual = math.ceil((SETUP_RESIDUAL - 1.0) * (analysis + setup))
    mass, u, v = _columns(cfg)
    # Cached regular plans hold x/y and supported x/y on each staggering.
    # The receipt temporarily holds a uint32 magnitude plus four masks
    # and packed bits. Existing padding covers the product grids; shallow
    # inputs still price these allocations explicitly before admission.
    receipt_cells = int(cfg.nz) * mass
    # Named source inventories identify the mass fields that run a receipt.
    receipt_masks = (8 * receipt_cells + (receipt_cells + 7) // 8
                     if inventory.hydrometeor_replays else 0)
    retained_setup = 16 * (mass + u + v) + receipt_masks
    residual = max(residual, retained_setup)
    # Candidate number arrays span the grid; all other closure scratch is
    # bounded by closure_device.COLD_START_CHUNK_CELLS. Reuse setup's budget.
    if state_on_card and int(cfg.mp_physics) in (8, 28):
        cells = int(cfg.nz) * mass
        numbers = 3 if int(cfg.mp_physics) == 28 else 2
        # Supplied theta/pressure may still be host fields. Their lazy
        # f64 uploads cost 16 bytes per cell; on the device REAL route they
        # are already resident (real_columns) and are reused.
        uploads = 0 if device_columns else 16
        closure = (4 * numbers + uploads) * cells + 112 * min(cells, 1048576)
        residual = max(residual, closure)
    terms = {"model_state": state, "forcing_analysis": analysis,
             "vertical_setup": setup, "setup_residual": residual}
    if state_on_card and device_columns:
        # Twelve source planes cover the four widened inputs, q/pd/RH
        # and their contiguous temporaries. Eighteen target planes cover
        # the thermodynamics, base, condensate and split temporaries.
        # Twenty surface planes and both staggered ladders coexist.
        # Source inventory cannot see all initialization options, so this
        # also reserves an upper bound for option-selected host fallbacks.
        terms["real_columns"] = (
            8 * mass * (12 * inventory.levels + 18 * int(cfg.nz) + 20)
            + 8 * (u + v) * (inventory.levels + int(cfg.nz)))
    return terms


def _add(total: dict, part: Mapping[str, int]) -> dict:
    for key, value in part.items():
        total[key] = total.get(key, 0) + int(value)
    return total


def _strip_config(cfg, width: int):
    """The largest of the four boundary side strips of ``cfg`` as a domain."""
    return replace(cfg, nx=max(int(cfg.nx), int(cfg.ny)), ny=int(width))


def price_preparation(route: str, domains: Sequence, inventory: SourceInventory,
                      *, boundary_intervals: int = 0,
                      boundary_workers: int = 0,
                      profile=None, vram_gib: float | None = None,
                      platform: str | None = None,
                      boundary_species=()
                      ) -> PreparationDevicePrice:
    """Price ``route`` preparing ``domains`` (root first) from ``inventory``.

    ``domains`` are the RunConfigs the preparation builds, root first;
    ``boundary_intervals`` the root's forcing intervals (boundary tables
    attached on the card before the children); ``boundary_workers`` the
    native HRRR worker processes that run beside the kept f00 state.
    """
    try:
        row = PREPARATION_ROUTES[route]
    except KeyError:
        raise ValueError(
            f"no preparation route {route!r}; known: "
            f"{sorted(PREPARATION_ROUTES)}") from None
    domains = tuple(domains)
    if not domains:
        raise ValueError("a preparation builds at least one domain")
    root, children = domains[0], domains[1:]
    context = context_bytes(profile=profile, vram_gib=vram_gib,
                            platform=platform)
    phases: dict[str, dict] = {}

    # Every domain's own build.  The experiment route keeps one more
    # analysis beside it; met_em builds each domain alone.
    for index, cfg in enumerate(domains):
        if index and row.children != "one_at_a_time":
            continue
        build = _build(cfg, inventory, state_on_card=row.state_on_card)
        if row.held_analyses:
            build["forcing_analysis"] += (row.held_analyses
                                          * analysis_bytes(cfg, inventory))
        if index == 0 and row.tables_on_root and row.state_on_card:
            build["boundary_tables"] = boundary_table_bytes(
                cfg, boundary_intervals, boundary_species=boundary_species)
        phases[f"build d{index + 1:02d}"] = build

    if children and row.children == "together":
        residue = {"model_state": state_bytes(root),
                   "forcing_analysis": analysis_bytes(root, inventory),
                   "boundary_tables": boundary_table_bytes(
                       root, boundary_intervals,
                       boundary_species=boundary_species)}
        for cfg in children:
            _add(residue, _build(cfg, inventory))
        phases["children"] = residue

    if row.boundary_workers and int(boundary_workers) > 0:
        slots = int(boundary_workers)
        width = int(getattr(root, "spec_bdy_width", 5) or 5)
        strip = _strip_config(root, width)
        slot = _build(strip, inventory)
        slot_pool = math.ceil(row.headroom * sum(slot.values()))
        phases["boundary workers"] = {
            "model_state": state_bytes(root),
            "forcing_analysis": 2 * analysis_bytes(root, inventory),
            "worker_slots": slots * (context + slot_pool),
        }

    if inventory.fp64_humidity_transform and inventory.source_points:
        phases["source transform"] = {
            "forcing_analysis": analysis_bytes(root, inventory),
            "source_transform": (FP64_HUMIDITY_BYTES_PER_SOURCE_POINT_LEVEL
                                 * inventory.levels
                                 * inventory.source_points),
        }

    totals = {}
    for name, terms in phases.items():
        pooled = sum(value for key, value in terms.items()
                     if key != "worker_slots")
        headroom = math.ceil((row.headroom - 1.0) * pooled)
        terms = dict(terms, pool_headroom=headroom, cuda_context=context)
        phases[name] = terms
        totals[name] = sum(terms.values())
    binding = max(totals, key=totals.get)
    return PreparationDevicePrice(
        route=route, need_bytes=int(totals[binding]),
        terms=MappingProxyType(dict(phases[binding])), phase=binding,
        basis=(_route_basis(row) + (
            "; device REAL column workspace priced from its live arrays"
            if inventory.device_real_columns and row.state_on_card else "")),
        phases=MappingProxyType(dict(totals)),
        chunk_source_bytes=(16 * inventory.source_points if route == "mapped" else 0),
        chunk_minimum_bytes=((
            16 * inventory.source_points +
            max((max(inventory.levels, int(cfg.nz) + 1) * (int(cfg.nx) + 1)
                 * 1024 for cfg in domains), default=0))
            if route == "mapped" else 0))


def _route_basis(row: PreparationRoute) -> str:
    """The calibration basis, naming a route's own pool headroom."""
    if row.pool_headroom is None:
        return PREPARATION_PRICE_BASIS
    return (f"{PREPARATION_PRICE_BASIS}; this route's pool headroom "
            f"x{row.headroom:g}, from its own measured CUDA preparation "
            "(RTX 5070 Ti 16 GB, 2026-09-29)")


def price_nest_activation(cfg, snapshot, *, physics_bytes: int = 0
                          ) -> PreparationDevicePrice:
    """A delayed nest's re-initialization at its start, from its DECODED analysis.

    ``woof run`` builds a delayed nest again when it starts, from the
    catalog's analysis at that time (:func:`woof.ingest.nest_init
    .initialize_child`), after its startup build is released: the new state
    and physics driver, that analysis on the nest's grid and the vertical
    setup beside them, with the same residual and allocator headroom every
    preparation carries.  ``snapshot`` is the decoded analysis the rebuild
    reads; ``physics_bytes`` the nest's physics arrays.  No CUDA context is
    charged: the forecast process already holds it, and the free figure
    this is weighed against is read in that process.

    Priced here and only for this door.  The shared admission estimate
    every review and the prepared route read carries no re-ingest term,
    because the prepared route restores a delayed nest from its prepared
    cache instead.
    """
    inventory = SourceInventory.from_snapshot(snapshot)
    terms = _build(cfg, inventory)
    if int(physics_bytes):
        terms["physics"] = int(physics_bytes)
    terms["pool_headroom"] = math.ceil(
        (PREPARATION_POOL_HEADROOM - 1.0) * sum(terms.values()))
    need = sum(terms.values())
    return PreparationDevicePrice(
        route="nest-activation", need_bytes=int(need),
        terms=MappingProxyType(dict(terms)), phase="activation",
        phases=MappingProxyType({"activation": int(need)}))


def price_forcing_preparation(route: str, exp, snapshots, *,
                              fp64_humidity_transform: bool = False,
                              profile=None, vram_gib: float | None = None,
                              boundary_species=()
                              ) -> PreparationDevicePrice:
    """Price a source door's preparation from its DECODED forcing.

    ``exp`` is the experiment after its vertical adaptation (the ``nz`` the
    states are built at), ``snapshots`` the decoded forcing times the door
    is about to interpolate: their inventory is the analysis, and their
    count sets the root's boundary tables.

    Only the first snapshot is read, by index, and the count by ``len``.
    Named breakage: the mapped route's forcing is a lazy sequence that
    packs one valid time at a time (``_RegularSnapshots``); turning it into
    a tuple packed all seven forcing times at once, which added 17 GB of host
    memory and 87 s to a 1792x1024x55 preparation before its first device
    allocation, and took a 64 GB host out of memory at 896x512x59 beside
    a second preparation.
    """
    if not (hasattr(snapshots, "__len__")
            and hasattr(snapshots, "__getitem__")):
        snapshots = tuple(snapshots)       # a one-shot iterable: no sequence
    count = len(snapshots)
    if not count:
        raise ValueError("a forcing preparation has at least one time")
    inventory = SourceInventory.from_snapshot(
        snapshots[0], fp64_humidity_transform=fp64_humidity_transform)
    return price_preparation(
        route, [domain.run for domain in exp.domains], inventory,
        boundary_intervals=count - 1, profile=profile,
        vram_gib=vram_gib, boundary_species=boundary_species)


#: The smallest inventory any source has: one level, no level field and no
#: surface plane.  Every term :func:`price_preparation` itemizes grows with
#: the inventory or does not read it, so a price built on this one is at or
#: below the decoded price of the same domains on the same route.
FLOOR_SOURCE_INVENTORY = SourceInventory(levels=1, level_fields=0,
                                         surface_planes=0)

#: The basis a floor price records in place of the calibration's.
PREPARATION_FLOOR_BASIS = (
    "a lower bound known before the decode: the model state, the smallest "
    "vertical setup, their temporaries and allocator headroom and the CUDA "
    "context, with no forcing analysis and no boundary tables; the price "
    "of the decoded forcing is the binding check")


def price_preparation_floor(route: str, exp, *, profile=None,
                            vram_gib: float | None = None
                            ) -> PreparationDevicePrice:
    """A lower bound on ``route``'s price for ``exp``, before any decode.

    The door calls this right after it loads the experiment, so an
    explicit ``cuda`` whose domains alone cannot fit the card is refused
    within seconds instead of after the host decode and the statics
    (501 s at 1792x1024x55 on a 24 GB card), and ``auto`` moves to the
    CPU at the same point.  Only the vertical coordinate's ``etac`` moves
    between this load and the decode, never a grid shape, so the domains
    priced here are the ones built.  Being a lower bound, it never refuses
    a preparation the decoded price would admit on the same card.
    """
    price = price_preparation(
        route, [domain.run for domain in exp.domains],
        FLOOR_SOURCE_INVENTORY, boundary_intervals=0, profile=profile,
        vram_gib=vram_gib)
    return replace(price, basis=PREPARATION_FLOOR_BASIS)


def price_downscale_interpolation(*, parent_nx: int, parent_ny: int,
                                  parent_nz: int, parent_fields: int,
                                  child_cfg, profile=None,
                                  vram_gib: float | None = None,
                                  platform: str | None = None
                                  ) -> PreparationDevicePrice:
    """Price the offline child's parent-to-child interpolation on the card.

    The boundary interpolation uploads every raw parent field on the
    PARENT's extent and levels together, couples them (a second copy),
    interpolates them onto the child's horizontal grid still at the
    parent's levels, and only then remaps to the child's levels
    (``woof/offline_child.py`` build_offline_lateral_boundaries).  A price
    sized on the child alone cannot see the first two, which is where a
    small child from a large parent spends its memory (A65, F08).  The
    child's own model state is the forecast's and is priced there.
    Itemized, with the mapped routes' residual and the ``downscale-child``
    route row's pool headroom; the route was run once on a card
    (:data:`MEASURED_ROUTE_PEAKS`) and peaked under half this price.
    """
    row = PREPARATION_ROUTES["downscale-child"]
    parent_columns = int(parent_nx) * int(parent_ny)
    child_mass, _u, _v = _columns(child_cfg)
    fields = int(parent_fields)
    parent = 4 * 2 * fields * int(parent_nz) * parent_columns
    child = 4 * fields * child_mass * (int(parent_nz) + int(child_cfg.nz))
    residual = math.ceil((SETUP_RESIDUAL - 1.0) * (parent + child))
    live = parent + child + residual
    terms = {"parent_fields": parent, "child_fields": child,
             "setup_residual": residual,
             "pool_headroom": math.ceil((row.headroom - 1.0) * live),
             "cuda_context": context_bytes(profile=profile, vram_gib=vram_gib,
                                           platform=platform)}
    need = sum(terms.values())
    return PreparationDevicePrice(
        route="downscale-child", need_bytes=int(need),
        terms=MappingProxyType(terms), phase="parent interpolation",
        basis=_route_basis(row),
        phases=MappingProxyType({"parent interpolation": int(need)}))


#: The measured peaks the price must never undercount (tests hold it).
#: Bytes are 1e9-byte GB as measured; ``card`` is nvidia-smi's whole-card
#: peak on an otherwise empty card, ``reserved`` the CuPy pool high-water.
#: Source: docs/dev/a65-preparation-peaks.md (integrate/2.8 at 7645b5125,
#: H100 80 GB, HRRR 2026-09-27 21Z f00-f06,
#: `woof prep --source hrrr-prs --preprocess-backend cuda`).
MEASURED_PREPARATION_PEAKS = (
    {"case": "3 km CONUS", "domains": ((1792, 1024, 55),),
     "card_gb": 37.46, "reserved_gb": 36.80, "live_gb": 31.10},
    {"case": "6 km CONUS", "domains": ((896, 512, 59),),
     "card_gb": 10.18, "reserved_gb": 9.52, "live_gb": 8.26},
    {"case": "nest", "domains": ((896, 512, 59), (480, 480, 59)),
     "card_gb": 12.58, "reserved_gb": 11.91, "live_gb": 10.87},
    {"case": "tiled nest", "domains": ((896, 512, 59), (480, 480, 59)),
     "card_gb": 12.56, "reserved_gb": 11.89, "live_gb": 10.87},
)

#: The measured source: HRRR pressure-level files, 39 levels, eleven level
#: fields (temperature, humidity, five hydrometeors, the winds, height and
#: the coordinate pressure), eleven surface planes and nine soil nodes of
#: two fields.
MEASURED_SOURCE_INVENTORY = SourceInventory(
    levels=39, level_fields=11, surface_planes=11 + 2 * 9)

#: Forcing intervals of the measured runs (f00 to f06).
MEASURED_BOUNDARY_INTERVALS = 6

#: The routes the calibration above reaches without a card measurement of
#: their own, each measured once on a card at a reference shape (tests hold
#: every price at or above its peak).  One CUDA preparation per route on an
#: RTX 5070 Ti 16 GB (Linux), integrate/2.8 at cc3cb0ad6, 2026-09-29.
#: ``predicted_bytes`` is the price the door decided on (the receipt's
#: ``selection.device_fit.need_bytes``); ``priced_bytes`` is the price now,
#: with the route's own term where the run needed one.  ``card_gb`` is the
#: preparation's own card memory, with every process it spawned summed
#: into it, from nvidia-smi compute-apps every 0.2 s, up to the end of the
#: preparation (the downscale child's forecast and the run route's physics
#: attach come after it and are priced by the forecast).  ``reserved_gb``
#: is the CuPy pool's reserved bytes at that point, where an instrument
#: read it.  GB are 1e9 bytes.  Method and runs:
#: docs/dev/a65-preparation-peaks.md.
MEASURED_ROUTE_PEAKS = (
    {"route": "downscale-child",
     "case": "3 km parent 408x420x49, 16 fields, to a 1 km child 450x450x49",
     "parent": (408, 420, 49), "parent_fields": 16,
     "child": (450, 450, 49), "child_dx": 1000.0,
     "predicted_bytes": 3_898_148_968, "priced_bytes": 3_898_148_968,
     "card_gb": 1.791, "reserved_gb": 1.520},
    {"route": "experiment",
     "case": "ERA5 (ARCO), 12 km 500x400x49, mp 10, two forcing times",
     "domains": ((500, 400, 49),), "dx": 12000.0, "mp_physics": 10,
     "inventory": (37, 6, 24), "boundary_intervals": 1,
     # 2.9.11 (paper/conus-suite-exact): the stored MF_VX_INV plane msfv_inv,
     # 4 x 401 x 500 = 802,000 B, 1,002,500 B at the 1.25 headroom.
     "predicted_bytes": 4_656_160_979, "priced_bytes": 4_817_714_059,
     "card_gb": 4.261, "reserved_gb": 3.907},
    {"route": "hrrr-native",
     "case": "3 km 556x444x49, f00 to f02, two boundary workers",
     "domains": ((556, 444, 49),), "boundary_workers": 2,
     # gp-closure: bounded scratch and temperature uploads add 40,312,580 bytes.
     # The measured historical prediction and card peak stay unchanged.
     # 2.9.11 (paper/conus-suite-exact): msfv_inv, 4 x 445 x 556 = 989,680 B,
     # 1,219,642 B with the route's margins.
     "predicted_bytes": 6_946_725_087, "priced_bytes": 6_988_257_309,
     "card_gb": 4.496, "reserved_gb": None},
)


__all__ = [
    "FP64_HUMIDITY_BYTES_PER_SOURCE_POINT_LEVEL",
    "MEASURED_BOUNDARY_INTERVALS", "MEASURED_PREPARATION_PEAKS",
    "MEASURED_ROUTE_PEAKS", "MEASURED_SOURCE_INVENTORY", "NOMINAL_SOURCE_INVENTORIES",
    "PREPARATION_POOL_HEADROOM", "PREPARATION_PRICE_BASIS",
    "PREPARATION_ROUTES", "PreparationDevicePrice", "PreparationRoute",
    "SETUP_RESIDUAL", "SourceInventory", "analysis_bytes",
    "boundary_table_bytes", "context_bytes", "price_downscale_interpolation",
    "FLOOR_SOURCE_INVENTORY", "PREPARATION_FLOOR_BASIS",
    "price_forcing_preparation",
    "price_preparation", "price_preparation_floor",
    "state_bytes", "vertical_setup_bytes",
]
