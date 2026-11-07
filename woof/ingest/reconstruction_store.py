"""Reserve device scratch, reconstruct windows, and publish a canonical host store.

This is a reconstruction operation, not executor admission. The route supplies
the existing initializer and physics/continuation preparer. No forecast step,
source selection, or physics selector is chosen here.
"""
from dataclasses import replace
import gc
from types import MappingProxyType

import numpy as np


class ReconstructionRefused(RuntimeError):
    pass


class ReconstructionReservation:
    """One private device reservation reused through reconstruction phases."""
    def __init__(self, budget_bytes):
        import cupy as cp
        if (isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int)
                or budget_bytes <= 0 or budget_bytes % 512):
            raise ValueError("device reconstruction budget must be positive and 512-byte aligned")
        self.budget_bytes = budget_bytes
        self.pool = cp.cuda.MemoryPool()
        self.pool.set_limit(size=budget_bytes)
        with cp.cuda.Stream.null:
            seed = self.pool.malloc(budget_bytes)
        del seed

    def malloc(self, size):
        import cupy as cp
        # CuPy keeps independent free lists for each allocation stream. Route
        # every suballocation through the seeded arena. A returned block may
        # last have been used by any worker, so drain before allowing reuse.
        # This conservative path trades overlap for a strict fixed reservation.
        cp.cuda.runtime.deviceSynchronize()
        with cp.cuda.Stream.null:
            return self.pool.malloc(size)

    def activate(self):
        import cupy as cp
        return cp.cuda.using_allocator(self.malloc)


def store_from_reconstruction(initializer, domain, parent, *, rows_per_slab,
                              device_budget_bytes, host_budget_bytes,
                              prepare=None, inventory_fn=None, reservation=None):
    """Build the ordinary streaming store without a full child device state.

    Reserve the entire device working budget before calling the initializer.
    CuPy default-allocator array construction uses this private bounded pool.
    Explicit scheme-owned pools and preallocated shared workspaces are outside
    this reservation and remain the route planner's separate obligation.
    Exceeding this reservation fails before a store is returned; no live node
    or old canonical store is mutated here.

    ``prepare(initialized, slab_domain, window)`` must attach the route's real
    physics and continuation data before harvesting. It receives the original
    domain-coordinate window; it must not derive global decisions (e.g. season)
    from a slab center. A caller preparing only atmospheric state may omit it,
    but that does not make the result a physics-ready forecast.
    """
    import cupy as cp
    from tilestream import hoststore
    from tilestream.driver import geography_inventory
    from tilestream.physics_inventory import carrier_scalars
    from woof.core.grid import BaseState
    from woof.core.streaming import prime_lazy_carriers, streamed_store_inventory
    from woof.ingest.prepared_store import (
        PreparedStore, _plan_slabs, _scatter_rows, _seam_disagreement)

    cfg = domain.run
    reservation = (ReconstructionReservation(device_budget_bytes)
                   if reservation is None else reservation)
    if not isinstance(reservation, ReconstructionReservation) or reservation.budget_bytes != device_budget_bytes:
        raise ValueError("reconstruction reservation must match the declared device budget")
    slabs = _plan_slabs(int(cfg.ny), int(rows_per_slab))
    pool = reservation.pool
    inventory_fn = inventory_fn or streamed_store_inventory()
    stores = ({}, {})
    manifest = None
    scalars = None
    template = None
    last = None
    peak_live = 0
    try:
        with reservation.activate():
            for index, (j0, rows) in enumerate(slabs):
                window = (slice(j0, j0+rows), slice(0, int(cfg.nx)))
                initialized = initializer(domain, parent, window=window)
                slab_domain = replace(domain, run=replace(cfg, ny=rows))
                if prepare is not None:
                    prepare(initialized, slab_domain, window)
                state = initialized.state
                prime_lazy_carriers(state, slab_domain.run)
                current_scalars = carrier_scalars(state)
                if scalars is None:
                    scalars = current_scalars
                elif current_scalars != scalars:
                    raise ReconstructionRefused("reconstruction scalar carriers differ between slabs")
                inventories = (inventory_fn(state, None), geography_inventory(state))
                if manifest is None:
                    manifest = []
                    planned = 0
                    for inv in inventories:
                        entries = hoststore.manifest_from_arrays(inv, cfg.nz, rows, cfg.nx, square_slab=True)
                        manifest.append(entries)
                        planned += sum(spec.nbytes(cfg.nz, cfg.ny, cfg.nx) for spec in entries)
                    hoststore.check_allocatable(planned, budget_bytes=host_budget_bytes)
                    for store, entries in zip(stores, manifest):
                        for spec in entries:
                            store[spec.name] = hoststore.alloc_pinned_array(
                                spec.shape(cfg.nz, cfg.ny, cfg.nx), spec.dtype)
                for store, inv in zip(stores, inventories):
                    if set(store) != set(inv):
                        raise ReconstructionRefused("reconstruction carrier manifest changed between slabs")
                    for name, value in inv.items():
                        host = cp.asnumpy(value) if isinstance(value, cp.ndarray) else np.asarray(value)
                        if _seam_disagreement(store[name], host, j0, rows):
                            raise ReconstructionRefused(f"reconstruction shared-face mismatch: {name}")
                        _scatter_rows(store[name], host, j0)
                cp.cuda.runtime.deviceSynchronize()
                peak_live = max(peak_live, int(pool.used_bytes()))
                if index == len(slabs)-1:
                    template, last = state, initialized
                del inventories, inv, state, initialized, value
                gc.collect()
        geo = stores[1]
        def setup(name):
            value = geo.get("setup/"+name)
            if value is not None:
                return value
            value = getattr(template, name)
            return cp.asnumpy(value) if isinstance(value, cp.ndarray) else value
        base = BaseState(mub=setup("mub2d") if cfg.terrain_opt else float(template.mub),
                         p_top=float(template.p_top), pb=setup("pb"), alb=setup("alb"),
                         thb=setup("thb"), phb=setup("phb"),
                         terrain_z=setup("ht") if cfg.terrain_opt else None)
        return PreparedStore(
            store=stores[0], geography=geo, scalars=scalars, template=template,
            coord=last.coord, base=base, boundaries=None, missing=(),
            receipt=MappingProxyType({
                "schema": "gpuwm-reconstruction-store-v1", "slabs": len(slabs),
                "rows_per_slab": int(rows_per_slab),
                "device_reservation_bytes": device_budget_bytes,
                "retained_pool_bytes": int(pool.total_bytes()),
                "observed_live_bytes_at_slab_boundaries": peak_live,
                "store_bytes": sum(a.nbytes for a in stores[0].values()),
                "geography_bytes": sum(a.nbytes for a in geo.values()),
                "physics_preparer_supplied": prepare is not None,
                "reservation_scope": "default-allocator reconstruction; shared and independent scheme pools excluded",
            }))
    except BaseException as error:
        try:
            cp.cuda.runtime.deviceSynchronize()
        except BaseException as cleanup_error:
            error.add_note(f"reconstruction drain also failed: {cleanup_error}")
        raise


def rederive_reconstructed_store(bundle, cfg, *, reservation, tile_nx, tile_ny):
    """Apply the existing post-transplant EOS/RK/held-rate operations in tiles.

    A one-cell halo serves the ordinary A-grid-to-face rate coupling. This
    operation advances no forecast clock and invokes no physics producer.
    Prognostics and raw held rates are unchanged, so completed diagnostic
    writes cannot contaminate the next tile's reconstruction inputs.
    """
    import cupy as cp
    from woof.core.diagnostics import update_diagnostics
    from woof.core.streaming import prepared_tile_state_factory, streamed_store_inventory
    from woof.ingest.nest_init import seed_rk_time_t_copies
    from tilestream import gather, harness, spec
    from tilestream.driver import geography_inventory, geography_scalars, _pin_scheme_geography
    from tilestream.physics_inventory import set_carrier_scalars

    if not isinstance(reservation, ReconstructionReservation):
        raise TypeError("post-transplant reconstruction requires its device reservation")
    plans = spec.plan_tiles(nx=cfg.nx, ny=cfg.ny, tile_nx=tile_nx,
                           tile_ny=tile_ny, halo=1, periodic=False)
    take = streamed_store_inventory()
    factory = prepared_tile_state_factory(bundle.template, cfg, warmup=0)
    tile_cfg = harness.tile_config(cfg, plans[0].cnx, plans[0].cny)
    peak = 0
    try:
        with reservation.activate():
            tile = factory(tile_cfg)
            _pin_scheme_geography(tile)
            for name, value in geography_scalars(bundle.geography).items():
                setattr(tile, name, value)
            for window in plans:
                gather.gather_tile(bundle.store, tile, window, inventory_fn=take, nz=cfg.nz)
                gather.gather_tile(bundle.geography, tile, window,
                                   inventory_fn=geography_inventory, nz=cfg.nz)
                set_carrier_scalars(tile, bundle.scalars)
                update_diagnostics(tile, cfg.hypsometric_opt)
                seeded = seed_rk_time_t_copies(tile)
                driver = getattr(tile, "physics", None)
                if driver is not None:
                    driver.recouple_after_relocation(tile, tile_cfg)
                if set(take(tile)) != set(bundle.store):
                    raise ReconstructionRefused("post-transplant recoupling changed the canonical carrier set")
                gather.scatter_tile(tile, bundle.store, window, inventory_fn=take, nz=cfg.nz)
                cp.cuda.runtime.deviceSynchronize()
                peak = max(peak, int(reservation.pool.used_bytes()))
            del tile
            gc.collect()
        return {"operation": "bounded existing EOS, RK reseed, and held-physics recoupling",
                "tiles": len(plans), "halo": 1, "rk_seeded": list(seeded),
                "forecast_steps": 0, "device_reservation_bytes": reservation.budget_bytes,
                "observed_live_bytes_at_tile_boundaries": peak}
    except BaseException as error:
        try:
            cp.cuda.runtime.deviceSynchronize()
        except BaseException as cleanup_error:
            error.add_note(f"post-transplant reconstruction drain also failed: {cleanup_error}")
        raise
