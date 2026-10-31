"""Itemize independent prepared radiation tile buffers without CUDA.

Both itemized radiation models reach this: RTE+RRTMGP through
:func:`standalone_rte_storage_bytes`, the legacy RRTMG port through its
own :func:`woof.core.rrtmg_legacy.legacy_radiation_vram_bytes`, selected
by ``physics_compat.rrtmg_variant`` in
:meth:`PreparedTileMemory.radiation_storage_bytes` and named in
:meth:`PreparedTileMemory.terms` under ``basis``.  The paragraphs below
describe the RTE+RRTMGP arm, which is the one the probes were taken on.

The prepared factory executes RRTMGPRadiation without chunk_workspace.
Its finalized optics and Planck arrays therefore cannot be priced from the
fused shared-workspace inventory.  Every buffer below pays its own complete
LW+SW allocation set: CuPy keeps separate free arenas for separate compute
streams.  This deliberately forgoes both phase aliasing and cross-buffer
pool reuse.  The column-bounded loader and its retained last slab are also
inside the envelope.  No physics implementation or user setting is changed.

THE MODEL PRICES THE ROUTE, NOT ONE CONFIGURATION.  Every term below is
read off the experiment it is handed -- ``nz``, ``p_top``, ``column_chunk``
from the vertical and chunking, the resident and transient bytes from
:func:`woof.core.preflight.estimate_domain` for whatever physics is
selected -- so it applies to any prepared single-root host-store run on the
RTE+RRTMGP solver.  It was first shipped behind a twelve-value config
fingerprint (Morrison, KF, YSU, nz 49, chunk 3125, ...), and every other
configuration on the same route silently kept the fused-inventory price,
measured 23 % low against this itemization at 980,000 cells with two
buffers (9.304 GiB itemized against 7.580 GiB fused, ENG-013).  The
fingerprint survives only as :func:`measured_anchor`, the configuration the
itemization was MEASURED against, for reports and tests -- never as a gate.

THE BUFFER IS PRICED AT THE WINDOW'S OWN SHAPE, NOT AT ``N x 1``.  Until
2.7.3 every buffer was itemized as an ``N x 1`` domain, ``N`` its column
count, on the argument that a one-row rectangle maximizes every horizontal
face and edge product and so bounds any rectangle of the same area.  It
does -- and for the terms that scale with the PERIMETER rather than the
area it bounds them by the perimeter of a rectangle no tiling can produce.
REPRODUCED on a user's 572x524x49 icon-eu forecast (Thompson, RTE+RRTMGP,
MYNN, Noah, six retained forcing intervals) with a pinned 250x250 tiling:
the 286x286 compute window is 81,796 columns, so the sizing rectangle was
81,796 x 1 with a perimeter 143x the window's, and its eager
``lbc_forcing_tables`` came to 9.03 GiB PER BUFFER against 0.06 GiB for the
window itself.  Two buffers of that, under the allocator headroom, put
20.8 GiB of forcing tables that no buffer allocates into the streamed
envelope, which read 38.96 GiB against a 17.43 GiB resident run -- a
streamed forecast priced at 2.2x the resident one, refused before fetch.

So a caller that knows the window (the pinned road, and the planner once
it has chosen a tile) passes ``shape=(window_nx, window_ny)`` and the
buffer is itemized at exactly that rectangle.  A caller that has only a
cell count (the planner's binary search over window sizes) gets
:meth:`PreparedTileMemory.sizing_shape`: the most elongated rectangle of
that area a tiling can LEGALLY produce -- no side narrower than the
smallest compute window, ``2 * halo + 1``, and none longer than the domain
plus its halo -- which still bounds every legal rectangle of that area on
every term, and does so by a perimeter a tiling can actually have.

MEASURED against the fix, 2026-09-10, test host A (RTX 5070 Ti, 15.51 GiB,
cupy 14.2.0 / CUDA 13), the same 572x524x49 icon-eu forecast on real
2026-09-10T12 data, per-process device memory read by nvidia-smi
(``compute-apps used_memory``: the process's whole device allocation,
CUDA context included) at 0.2 s:

* 250x250 tiles, nbuffers = 1: this model 8.89 GiB; measured peak 6.18 GiB
  (6,326 MiB, at the radiation call), steady 5.97 GiB, over 40 steps and
  four radiation calls.
* 250x250 tiles, nbuffers = 2 (the user's tiling): this model 15.54 GiB;
  measured peak 11.11 GiB (11,374 MiB, recurring at each radiation call),
  steady 7.83 GiB between calls, over 122 steps, no allocation failure.

So the corrected envelope brackets the run on both roads with 28-31 %
to spare, and the per-buffer radiation storage it carries is a real
transient (about 1.6 GiB per buffer measured against the 2.01 GiB priced
per 3,125-column chunk here).  The run the user asked for does reach
11.1 GiB on a card of this class, above the 10.69 GiB they had free; one
buffer of the same tile measured 6.2 GiB and is the remedy.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

GIB = 1024 ** 3


def domain_of(exp, cfg):
    """The :class:`DomainConfig` in ``exp`` whose ``run`` IS ``cfg``.

    THE DOMAIN BEING PRICED, not the root.  A tree walk asks this module
    about every domain in turn (``woof.core.streaming`` builds one
    footprint per node), and answering about ``exp.root`` priced a nest's
    buffers at the root's geometry, its boundary tables and its nesting.
    ``None`` when the config is not a domain of this experiment, which is
    the one case the itemization has nothing to say about.
    """
    for dc in getattr(exp, "domains", ()) or ():
        if dc.run is cfg or dc.run == cfg:
            return dc
    return None


def supported(exp, cfg, options) -> bool:
    """The route this module itemizes: a prepared streamed domain's buffers.

    The ROUTE condition, and nothing narrower (ENG-013): a domain of this
    experiment whose ``[tiles]`` is on, running a radiation model this
    module can itemize.  Both spectra are itemized --
    :func:`standalone_rte_storage_bytes` for RTE+RRTMGP and
    :func:`woof.core.rrtmg_legacy.legacy_radiation_vram_bytes` for the
    legacy port, selected in :meth:`PreparedTileMemory.radiation_storage_bytes`
    -- and :meth:`PreparedTileMemory.terms` states which one priced the
    run under ``basis``.

    NOT A GATE ON ANYTHING UNMEASURED.  Four conditions this used to carry
    beyond the route -- ``cfg.specified``, a host store, CAM ozone, a
    follower window -- priced nothing and only sent the run back to the
    fused rung measured 23 % low, which is spent as window size and then
    fails in the allocator at the first radiation call.  Each is priced
    instead: the ozone and follower carriers through ``estimate_domain``'s
    own keywords, the device carrier store through
    :meth:`PreparedTileMemory.device_store_bytes`.

    ONE LIMIT REMAINS, and it names what it cannot price: a DOMAIN TREE.
    Every term of the fixed floor below is the PREPARED STORE's -- the
    column-bounded loader's initialization peak, its retained template
    slab -- and a nest inside a live tree has neither, so charging each
    nest a prepared loader floor would be an invented basis rather than a
    conservative one; the measurements this module reports were all taken
    on a single prepared root.  It also moves shipped tree admission
    decisions (MEASURED: with the tree arm on, the two-domain case
    tests/test_auto_admission.py:129 pins moves its ROOT from resident to
    streamed), and a tree's per-domain planner search re-runs this
    itemization per candidate window, which took plan review from seconds
    to minutes on the 4-to-16 domain searches.  Retire by measuring a
    streamed nest's actual per-buffer floor and splitting the prepared
    loader terms out of :meth:`PreparedTileMemory.fixed_terms`; the
    per-domain plumbing is already here (:func:`domain_of`, the model's
    ``domain``/``options`` fields), so the gate is this one clause.

    The conditions themselves live in :func:`route_decline`, which states
    each one in words instead of implying it, so a caller that only learns
    the itemization declined can still print what it declined on.
    """
    return route_decline(exp, cfg, options) is None


def route_decline(exp, cfg, options) -> str | None:
    """The ROUTE condition of :func:`supported`, stated rather than implied.

    ``None`` when this run is on the route.  Otherwise the one phrase that
    says why it is not, so a caller that only knows the itemization
    declined can print what it declined on.  :func:`supported` is exactly
    this answer as a bool and :func:`decline_basis` is this answer plus the
    estimate term, so the three can never disagree about one configuration.
    """
    from woof.config import radiation_scheme_ids
    if exp is None or domain_of(exp, cfg) is None:
        return ("this configuration is not a domain of the experiment being "
                "priced, which is the one case the itemization has nothing "
                "to say about")
    if len(exp.domains) != 1:
        return ("a domain tree keeps the empirical rung: every term of this "
                "model's fixed floor is the prepared store loader's, which a "
                "nest inside a live tree does not have, so charging one "
                "would be an invented basis rather than a conservative one")
    # ON THE ROUTE: [tiles] configured for this domain (mode on/auto).
    # A resident run has no prepared tile buffers to price and keeps
    # the resident estimate untouched.
    if not options.enabled:
        return ("[tiles] is off for this domain, so there are no prepared "
                "tile buffers to itemize and the resident estimate stands")
    # The implementation selector retains its default even when no
    # spectrum requests RRTMG. Only an active RRTMG spectrum allocates
    # the unfused radiation storage itemized below; other schemes keep
    # their own price. A single active spectrum is conservatively
    # charged both until the itemization is split by spectrum.
    if 4 not in radiation_scheme_ids(cfg):
        return ("no active RRTMG spectrum, so this domain allocates none of "
                "the unfused radiation storage this model itemizes and keeps "
                "its own scheme's price")
    return None


def measured_anchor(exp, cfg, options) -> bool:
    """Is this the configuration the itemization was MEASURED against?

    The Sep08 prepared-store probes on the RTX 3080 (allocator peak,
    loader template, unfused LW+SW arrays -- see
    tests/test_prepared_tile_memory.py) were taken on exactly this
    configuration.  A report may say so; a test may pin it.  It is NOT a
    gate on :func:`supported`: gating the model on it left every other
    configuration on the fused-inventory price measured 23 % low (ENG-013).
    """
    from woof.config import radiation_scheme_ids
    from woof.physics_compat import RRTMG_VARIANT_RTE_RRTMGP, rrtmg_variant
    return bool(
        supported(exp, cfg, options)
        # The probes were taken on the single prepared RTE+RRTMGP root with
        # a host store; the itemization now reaches further than they do,
        # so the anchor keeps the shape it was MEASURED at.
        and cfg.specified and not cfg.nested and options.store == "host"
        and rrtmg_variant(cfg) == RRTMG_VARIANT_RTE_RRTMGP
        and options.follower_context is None
        and not getattr(options.radiation_context, "cam_ozone", False)
        and cfg.nz == 49 and 0.0 < exp.vertical.p_top <= 10000.0
        and exp.column_chunk == 3125
        and cfg.mp_physics == 10 and cfg.cu_physics == 1
        and cfg.bl_pbl_physics == 1 and cfg.sf_sfclay_physics == 91
        and cfg.sf_surface_physics == 2 and cfg.num_soil_layers == 4
        and radiation_scheme_ids(cfg) == (4, 4)
        and not cfg.use_adaptive_time_step)


def standalone_rte_storage_bytes(nz: int, columns: int, column_chunk: int,
                                 p_top: float) -> dict[str, int]:
    """Named storage of the actual workspace=None branch, without aliasing.

    Columns/upper profiles/metadata are already in estimate_domain. This
    prices the solver allocations beside them, keeping both spectra and an
    old Planck result simultaneously even where the actual code frees them.
    The old sources variable survives into SW and the next chunk's RHS.
    Partial-flux storage takes the largest non-folded tile (16 g-points),
    independent of the attached device; folded launches allocate none.
    This bound stays monotone across chunk tails and SM coverage thresholds.
    """
    from woof.core.rrtmgp import (
        gas_table_shape, rrtmgp_above_model_layer_counts)

    c = min(int(columns), int(column_chunk))
    if c < 1:
        return {}
    ul, us = rrtmgp_above_model_layer_counts(p_top)
    nl, ns = nz + ul, nz + us
    lw, sw = gas_table_shape("lw"), gas_table_shape("sw")
    planck = c * lw.ngpt * (nl + nl + 1 + 1) * 4
    return {
        "lw/gas_tau": c * nl * lw.ngpt * 4,
        "lw/vmr": c * nl * (lw.ngas + 1) * 4,
        "lw/cloud_tau_ssa_g": 3 * c * nl * lw.nband * 4,
        "lw/col_dry": c * nl * 4,
        "lw/mcica_mask": c * nl * lw.ngpt,
        "lw/finalized_tau": c * nl * lw.ngpt * 4,
        "lw/planck_current": planck,
        # The previous chunk's sources are still bound while the new result
        # is constructed. Always charge both, including a one-chunk tile.
        "lw/planck_previous": planck,
        "lw/emiss_incident": 2 * c * lw.ngpt * 4,
        "lw/flux_up_dn": 2 * c * (nl + 1) * 4,
        "lw/partial_flux": 2 * 16 * c * (nl + 1) * 4,
        "lw/ozone_interp_log": c * nl * (8 + 4),
        "sw/gas_tau_ssa": 2 * c * ns * sw.ngpt * 4,
        "sw/vmr": c * ns * (sw.ngas + 1) * 4,
        "sw/cloud_tau_ssa_g": 3 * c * ns * sw.nband * 4,
        "sw/col_dry": c * ns * 4,
        "sw/mcica_mask": c * ns * sw.ngpt,
        "sw/finalized_tau_ssa_g": 3 * c * ns * sw.ngpt * 4,
        "sw/albedo_incident": 2 * c * sw.ngpt * 4,
        "sw/mu0_layers": c * ns * 4,
        "sw/flux_up_dn_dir": 3 * c * (ns + 1) * 4,
        "sw/partial_flux": 3 * 16 * c * (ns + 1) * 4,
        "sw/ozone_interp_log": c * ns * (8 + 4),
    }


@dataclass(frozen=True)
class PreparedTileMemory:
    """Monotone current-code envelope, suitable for planner inversion."""
    experiment: object = field(repr=False, compare=False)
    profile: object = field(repr=False, compare=False)
    forcing_intervals: int = 0
    _cache: dict = field(default_factory=dict, repr=False, compare=False)
    #: The domain being priced.  ``None`` keeps the historical meaning --
    #: the root -- for the direct constructions that predate the tree walk.
    domain: object = field(default=None, repr=False, compare=False)
    #: That domain's own ``[tiles]``.  It carries the radiation and
    #: follower contexts whose arrays this model now PRICES rather than
    #: declining to price, and the store road the carriers take.
    options: object = field(default=None, repr=False, compare=False)
    loader_budget_bytes: int | None = field(default=None, repr=False, compare=False)

    @property
    def domain_config(self):
        return self.experiment.root if self.domain is None else self.domain

    @property
    def cfg(self):
        return self.domain_config.run

    @property
    def nz(self):
        return self.cfg.nz

    @property
    def pool_headroom(self) -> float:
        """The forecast's measured pool margin (A163), the same number the
        resident envelope carries for this experiment."""
        from woof.core import preflight as pf
        return pf.forecast_pool_headroom(
            dc.run for dc in self.experiment.domains)

    @property
    def cam_ozone(self) -> bool:
        """Does this domain carry the CAM ozone climatology's arrays?

        ``estimate_domain`` already derives the increment generically
        (``physics_array_shapes(..., cam_ozone=...)``), so declining to
        price it was one unpassed keyword rather than a missing model.
        """
        context = getattr(self.options, "radiation_context", None)
        return bool(getattr(context, "cam_ozone", False))

    @property
    def follower_slots(self) -> tuple:
        """The declared follower window carriers, priced as scratch planes."""
        context = getattr(self.options, "follower_context", None)
        return tuple(getattr(context, "slots", ()) or ())

    def _domain(self, nx, ny, *, tile_buffer=False):
        """``estimate_domain`` of this domain at ``nx x ny``.

        ``tile_buffer=True`` for a tile buffer: its MYNN workspace is its
        own, at the width a buffer walks
        (:func:`woof.core.mynn_pbl_scratch.resolve_mynn_tile_column_chunk`,
        which :func:`woof.core.streaming.prepared_tile_state_factory`
        gives every buffer it builds).  The loader template and slab are
        not buffers and keep the run's width.
        """
        from woof.core import preflight as pf
        exp = self.experiment
        dc = self.domain_config
        dc = replace(dc, run=replace(dc.run, nx=nx, ny=ny))
        return pf.estimate_domain(
            dc, spec_bdy_width=exp.spec_bdy_width,
            n_lbc_intervals=self.forcing_intervals,
            p_top=exp.vertical.p_top, column_chunk=exp.column_chunk,
            cam_ozone=self.cam_ozone, follower_slots=self.follower_slots,
            tile_buffer=tile_buffer)

    def fixed_terms(self):
        if "fixed" not in self._cache:
            from woof.core import preflight as pf
            from woof.ingest.prepared_store import default_slab_rows, maximum_slab_rows
            exp = self.experiment
            nx, ny = self.cfg.nx, self.cfg.ny
            fixed = {
                "k_tables_bytes": pf.k_distribution_bytes(),
                "physics_tables_bytes": pf.thompson_generation_coefficient_bytes(self.cfg),
                "cuda_context_bytes": self.profile.cuda_context_bytes,
                "local_memory_bytes": pf.kernel_local_memory_bytes(exp, profile=self.profile),
                "unmodelled_bytes": pf.ENVELOPE_UNMODELLED_BYTES,
            }
            rows = default_slab_rows(nx, ny)
            template_rows = 1
            if self.loader_budget_bytes is not None:
                rows = maximum_slab_rows(nx, ny)
                # The last one-row slab retains the same height-invariant
                # template regardless of a lower runtime memory allowance.
                template_rows = 1
                tables = fixed["k_tables_bytes"] + fixed["physics_tables_bytes"]
                overhead = (fixed["cuda_context_bytes"] + fixed["local_memory_bytes"]
                            + fixed["unmodelled_bytes"] + self.device_store_bytes())

                def loader_price(height):
                    slab = self._domain(nx, height)
                    pool = slab.resident_bytes + slab.transient_bytes + tables
                    return math.ceil(self.pool_headroom * pool) + overhead

                if loader_price(rows) > int(self.loader_budget_bytes):
                    lo, hi = 1, rows
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if loader_price(mid) <= int(self.loader_budget_bytes):
                            lo = mid
                        else:
                            hi = mid - 1
                    rows = lo
                    # If one row exceeds the allowance, keep its real price.
                    # The ordinary planner refusal then names that floor.
            template, slab = self._domain(nx, template_rows), self._domain(nx, rows)
            fixed.update(loader_rows=rows, template_rows=template_rows,
                         template_resident_bytes=template.resident_bytes,
                         loader_pool_peak_bytes=slab.resident_bytes + slab.transient_bytes)
            self._cache["fixed"] = fixed
        return self._cache["fixed"]

    def _columns(self, window_cells):
        columns, rem = divmod(int(window_cells), self.nz)
        if rem or columns < 1:
            raise ValueError("a prepared tile must contain whole vertical columns")
        return columns

    @property
    def halo(self) -> int:
        """The domain's own per-step dependency radius, in mass cells."""
        if "halo" not in self._cache:
            from tilestream.harness import halo_radius
            self._cache["halo"] = int(halo_radius(self.cfg))
        return self._cache["halo"]

    def sizing_shape(self, columns) -> tuple[int, int]:
        """The rectangle a buffer of ``columns`` is priced at when its shape
        is not known: the most elongated one a tiling can LEGALLY produce.

        Every compute window is a tile plus a halo on both sides of both
        axes, so no side is narrower than ``2 * halo + 1``; and a window is
        clamped inside a non-periodic domain or wraps a periodic one, so no
        side is longer than the domain's longer axis plus two halos.  Among
        rectangles of area ``N`` inside those limits the one with the
        longest perimeter -- the one that bounds every perimeter-scaled
        term for all of them -- puts one side at the lower limit, or, when
        that would make the other side too long, one side at the upper.
        Its area is at least ``N``, never less, so the per-cell terms are
        bounded too.  Monotone in ``N`` on every side, which is what the
        planner's binary inversion needs.

        A one-row ``N x 1`` rectangle was the sizing shape before this: see
        the module docstring for what its 143x perimeter did to a real
        user's envelope.
        """
        columns = int(columns)
        run = self.cfg
        lo = 2 * self.halo + 1
        hi = max(int(run.nx), int(run.ny)) + 2 * self.halo
        a = max(1, min(lo, math.isqrt(columns)))
        b = -(-columns // a)
        if b > hi:
            b = hi
            a = -(-columns // hi)
        return a, b

    def _shape_for(self, window_cells, shape):
        columns = self._columns(window_cells)
        if shape is None:
            return self.sizing_shape(columns)
        nx, ny = int(shape[0]), int(shape[1])
        if nx < 1 or ny < 1 or nx * ny != columns:
            raise ValueError(
                f"window shape {nx}x{ny} does not hold {columns} columns")
        return nx, ny

    def buffer_terms(self, window_cells, shape=None):
        """Itemized bytes of ONE tile buffer holding ``window_cells``.

        ``shape`` is the compute window ``(window_nx, window_ny)`` when the
        caller knows it, and the buffer is itemized at exactly that
        rectangle; without it the buffer is itemized at
        :meth:`sizing_shape`, which bounds every legal rectangle of that
        area.  Either way this is a sizing shape only, never an emitted or
        executed grid.
        """
        from woof.core import preflight as pf
        nx, ny = self._shape_for(window_cells, shape)
        # A sizing bound (no shape) and an exact window of the same rectangle
        # are different prices, so they never share a cache entry.
        key = (nx, ny, shape is None)
        if key not in self._cache:
            exp = self.experiment
            itemized = self._domain(nx, ny, tile_buffer=True)
            resident, transient = itemized.resident_bytes, itemized.transient_bytes
            if shape is None and nx != ny:
                # The sizing shape bounds a rectangle of either orientation.
                # Some state planes carry one more row than column (msfv and,
                # from 2.9.11, the stored MF_VX_INV plane msfv_inv, both
                # (ny + 1, nx)), so a window long in x outgrew the bound
                # itemized long in y (test_prepared_tile_memory, 256 x 44:
                # 364 B over).  Itemize the transpose too and keep each
                # term's larger value.
                turned = self._domain(ny, nx, tile_buffer=True)
                resident = max(resident, turned.resident_bytes)
                transient = max(transient, turned.transient_bytes)
            run = replace(self.cfg, nx=nx, ny=ny)
            work_exp = replace(exp, domains=(replace(self.domain_config, run=run),))
            self._cache[key] = {
                "resident_bytes": resident,
                "step_transient_bytes": transient,
                "radiation_named_storage_bytes": self.radiation_storage_bytes(nx * ny),
                "column_workspace_bytes": pf.column_workspace_bytes(work_exp, profile=self.profile),
            }
        return self._cache[key]

    def radiation_storage_bytes(self, columns) -> int:
        """Per-buffer radiation storage of the model THIS domain selects.

        A two-row selection over pricing functions that already exist, not
        a per-model code path: ``rrtmg_variant`` picks the row.

        * RTE+RRTMGP -- :func:`standalone_rte_storage_bytes`, the solver's
          own unfused workspace=None storage.
        * legacy RRTMG -- :func:`woof.core.rrtmg_legacy.
          legacy_radiation_vram_bytes`, the legacy port's own "peak
          transient device bytes of ONE adapter call", composed from the
          engines' pricing functions.  Called on the same basis
          :func:`woof.core.preflight.estimate_experiment` already prices
          it on -- the engines' own column chunk, this profile's resident
          thread capacity -- so the two surfaces cannot disagree about one
          domain.

        Charging one solver's arrays as a bound on the other's would be an
        invented basis: nothing measures them against each other.
        """
        from woof.config import radiation_scheme_ids
        from woof.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant
        exp, cfg = self.experiment, self.cfg
        longwave, shortwave = radiation_scheme_ids(cfg)
        if rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY:
            from woof.core.rrtmg_legacy import legacy_radiation_vram_bytes
            return int(legacy_radiation_vram_bytes(
                ncol=int(columns), nz=int(cfg.nz), p_top=exp.vertical.p_top,
                column_chunk=None, longwave=longwave == 4, shortwave=shortwave == 4,
                resident_threads=(0 if self.profile is None
                                  else self.profile.resident_thread_capacity),
                aer_opt=int(getattr(cfg, "aer_opt", 0))))
        return int(sum(standalone_rte_storage_bytes(
            self.nz, columns, exp.column_chunk, exp.vertical.p_top).values()))

    @property
    def radiation_basis(self) -> str:
        """Which radiation model priced the buffer, for the refusal to print."""
        from woof.config import radiation_scheme_ids
        from woof.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant
        cfg = self.cfg
        longwave, shortwave = radiation_scheme_ids(cfg)
        name = ("legacy RRTMG, priced from its own adapter-call function"
                if rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY else
                "RTE+RRTMGP, priced from its own unfused solver storage")
        if 1 in (longwave, shortwave):
            # A mixed selection (classic RRTM on one spectrum, RRTMG on the
            # other) keeps BOTH: the classic columns are priced inside
            # buffer/step_transient_bytes through
            # preflight.classic_rrtm_column_shapes, not dropped.
            name += ("; the classic RRTM spectrum's columns are inside "
                     "buffer/step_transient_bytes")
        return name

    def device_store_bytes(self) -> int:
        """The carrier store, when ``[tiles] store = 'device'`` keeps it here.

        A device store is the SAME tile buffers plus a full-domain carrier
        store that lives on the card instead of in pinned host RAM.  The
        per-cell carrier rate has one owner, ``tilestream.autoplan.
        Footprint.store_bytes``, and this reads it rather than restating
        it.  Zero on the host road, where the store is host RAM and
        :func:`woof.core.streaming.streamed_envelope` already prices it.
        """
        if getattr(self.options, "store", "host") != "device":
            return 0
        from tilestream import autoplan

        cfg = self.cfg
        extra = {}
        if self.follower_slots:
            extra["follower_slots"] = self.follower_slots
        context = getattr(self.options, "radiation_context", None)
        if context is not None:
            extra["radiation_context"] = context
        fp = autoplan.footprint_for(cfg, **extra)
        return int(fp.store_bytes(int(cfg.nx) * int(cfg.ny) * int(cfg.nz)))

    def buffer_bytes(self, window_cells, shape=None):
        return sum(self.buffer_terms(window_cells, shape).values())

    def vram_bytes(self, window_cells, nbuffers, shape=None):
        from woof.core import preflight as pf
        fixed = self.fixed_terms()
        tables = fixed["k_tables_bytes"] + fixed["physics_tables_bytes"]
        pool = (int(nbuffers) * self.buffer_bytes(window_cells, shape)
                + fixed["template_resident_bytes"] + tables)
        pool = max(pool, fixed["loader_pool_peak_bytes"] + tables)
        return (math.ceil(self.pool_headroom * pool)
                + fixed["cuda_context_bytes"] + fixed["local_memory_bytes"]
                + fixed["unmodelled_bytes"] + self.device_store_bytes())

    def terms(self, window_cells, nbuffers, shape=None):
        """Every term of :meth:`vram_bytes`, named, in the order they add up.

        The arithmetic a refusal prints: a reader holding a screenshot of
        it can check the total with a calculator and see which term the
        tile can move (the buffers) and which it cannot (the floors).
        Byte-valued entries end in ``_bytes``; the rest are labels.
        """
        from woof.core import preflight as pf
        nx, ny = self._shape_for(window_cells, shape)
        columns = nx * ny
        exp = self.experiment
        run = self.cfg
        fixed = self.fixed_terms()
        buffer = self.buffer_terms(window_cells, shape)
        per_buffer = sum(buffer.values())
        nbuffers = int(nbuffers)
        buffers = nbuffers * per_buffer
        tables = fixed["k_tables_bytes"] + fixed["physics_tables_bytes"]
        pool = buffers + fixed["template_resident_bytes"] + tables
        loader_peak = fixed["loader_pool_peak_bytes"] + tables
        pool_priced = max(pool, loader_peak)
        pool_with_headroom = math.ceil(self.pool_headroom * pool_priced)
        itemized = self._domain(nx, ny, tile_buffer=True)
        return {
            "domain": f"{int(run.nx)}x{int(run.ny)}x{int(run.nz)}",
            # WHICH domain: a tree walk prices one footprint per node and
            # every one of them reports a "domain" shape.
            "grid": f"d{int(getattr(self.domain_config, 'grid_id', 1)):02d}",
            "basis": self.radiation_basis,
            "window": (f"{nx}x{ny}x{int(run.nz)} = {columns:,} columns"
                       + ("" if shape is not None else
                          " (sizing rectangle; the window's shape was not given)")),
            "buffer/state_bytes": itemized.category_bytes("state"),
            "buffer/physics_bytes": itemized.category_bytes("physics"),
            "buffer/scratch_bytes": itemized.category_bytes("scratch"),
            "buffer/lbc_bytes": (itemized.category_bytes("lbc")
                                 + itemized.category_bytes("nest")),
            "buffer/diagnostic_bytes": itemized.category_bytes("diagnostic"),
            "buffer/step_transient_bytes": buffer["step_transient_bytes"],
            "buffer/radiation_named_storage_bytes": buffer["radiation_named_storage_bytes"],
            "buffer/column_workspace_bytes": buffer["column_workspace_bytes"],
            "buffer/total_bytes": per_buffer,
            "buffer/per_column_bytes": per_buffer // columns,
            "buffers": f"{nbuffers} x {per_buffer / GIB:.3f} GiB",
            "buffers_bytes": buffers,
            "fixed/template_resident_bytes": fixed["template_resident_bytes"],
            "fixed/k_tables_bytes": fixed["k_tables_bytes"],
            "fixed/physics_tables_bytes": fixed["physics_tables_bytes"],
            "fixed/loader_pool_peak_bytes": fixed["loader_pool_peak_bytes"],
            "pool_bytes": pool_priced,
            "pool_basis": ("the buffers plus the retained template and the k-tables"
                           if pool >= loader_peak else
                           f"the {fixed['loader_rows']}-row loader's initialization "
                           "peak, which is larger than the buffers"),
            "pool_headroom": f"x {self.pool_headroom:.2f}",
            "pool_with_headroom_bytes": pool_with_headroom,
            "fixed/cuda_context_bytes": fixed["cuda_context_bytes"],
            "fixed/local_memory_bytes": fixed["local_memory_bytes"],
            "fixed/unmodelled_bytes": fixed["unmodelled_bytes"],
            "store/device_carrier_bytes": self.device_store_bytes(),
            "radiation_chunk_columns": min(columns, int(exp.column_chunk)),
            "radiation_transient_bytes": 0,
            "vram_bytes": int(self.vram_bytes(window_cells, nbuffers, shape)),
        }

    @property
    def process_overhead_bytes(self):
        from woof.core import preflight as pf
        f = self.fixed_terms()
        return (math.ceil(self.pool_headroom * (
                    f["template_resident_bytes"] + f["k_tables_bytes"]
                    + f["physics_tables_bytes"]))
                + f["cuda_context_bytes"] + f["local_memory_bytes"]
                + f["unmodelled_bytes"])

    def max_window_cells(self, nbuffers, budget):
        # Binary search over the shape-free BOUND (sizing_shape), which is
        # monotone in the column count; the planner then prices the tile it
        # chose at that tile's own window, and the exact rectangle never
        # costs more than the bound that admitted it.
        lo, hi = 0, 1
        while self.vram_bytes(hi * self.nz, nbuffers) <= budget:
            lo, hi = hi, hi * 2
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.vram_bytes(mid * self.nz, nbuffers) <= budget:
                lo = mid
            else:
                hi = mid
        return lo * self.nz


def decline_basis(cfg, options, *, estimate=None) -> str | None:
    """Why the itemization declined to price this run, in one phrase.

    ``None`` when it did NOT decline, so a caller reads this exactly the
    way it reads :func:`for_options`: a string here and ``None`` there are
    the same answer said two ways, and there is no configuration for which
    :func:`for_options` returns ``None`` and this returns ``None`` too.

    EVERY decline is stated here, including the one that used to be
    silent.  ``estimate is None`` means no resolved forcing schedule was
    supplied, and the retained boundary count is a term of the fixed floor
    -- assuming a GFS cadence for an HRRR/RAP run or a differently
    retained archive would be an invented basis, so the run keeps the
    fused rung price.  It was previously a bare ``return None`` with the
    reason only in a comment, which left the plan review's own basis line
    unable to say which model priced the run or why.
    """
    context = getattr(options, "resident_context", None)
    exp = None if context is None else context.experiment
    off_route = route_decline(exp, cfg, options)
    if off_route is not None:
        return off_route
    if estimate is None:
        return ("no resident estimate was supplied, so the resolved forcing "
                "schedule this model prices its retained boundary tables "
                "from is unknown; assuming a cadence would be an invented "
                "basis, so the fused rung price stands")
    return None


def for_options(cfg, options, *, profile=None, estimate=None, machine=None):
    """Return an inventoried-contract model, otherwise retain old pricing.

    Every ``None`` return states its basis through :func:`decline_basis`;
    none of them is silent.
    """
    context = getattr(options, "resident_context", None)
    exp = None if context is None else context.experiment
    if decline_basis(cfg, options, estimate=estimate) is not None:
        return None
    from woof.core import preflight as pf
    # THE DOMAIN BEING PRICED and ITS OWN [tiles], so a tree walk itemizes
    # each streamed domain at its own geometry, boundary tables, radiation
    # context and follower carriers instead of the root's.
    budget = getattr(options, "vram_budget_bytes", None)
    if budget is None and machine is not None:
        budget = max(0, int(machine.vram_bytes) - pf.EXTERNAL_MARGIN_BYTES)
    return PreparedTileMemory(exp, profile or pf.MEASURED_LOCAL_MEMORY_PROFILE,
                              estimate.retained_forcing_intervals,
                              domain=domain_of(exp, cfg), options=options,
                              loader_budget_bytes=budget)
