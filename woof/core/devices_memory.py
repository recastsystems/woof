"""CPU admission for permanently resident ranks, grouped by physical card."""
from __future__ import annotations

from dataclasses import replace
from math import prod
from types import SimpleNamespace

from woof.core.devices import DEVICES_OFF, DevicesRefused, refuse_unrouted_devices

GIB = 2**30
# Optional per-rank history storage. A rank whose complete conservative
# carrier census exceeds the cap always uses the producer-fenced direct
# copy, including when one requested field subset would fit by itself.
FRAME_SNAPSHOT_LIMIT_BYTES = 4 * GIB


def _span_bytes(spans):
    total = end = 0
    for left, right in sorted(spans):
        total += max(0, right - max(left, end))
        end = max(end, right)
    return total


def priced_host_live_bytes(inventory, geography, boundaries=None):
    """Unique already allocated host bytes of the supplied prepared bundle.

    Store/geography arrays and materialized forcing intervals are counted;
    no lazy interval is loaded to earn admission credit. File-backed arrays
    are excluded because their reclaimable pages can already be included in
    the OS available-memory reading. Aliases across both terms count once.
    The caller caps each credit at the matching estimate term.
    """
    import numpy as np
    from woof.ingest.prepared_mmap import is_file_backed_array

    def spans(arrays):
        result = []
        for array in arrays:
            if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
                continue
            if is_file_backed_array(array):
                continue
            start = int(array.__array_interface__["data"][0])
            if array.nbytes:
                result.append((start, start + int(array.nbytes)))
        return result

    store = spans(list((inventory or {}).values()) + list((geography or {}).values()))
    fields = []
    intervals = getattr(boundaries, "intervals", ())
    if isinstance(intervals, (tuple, list)):
        for interval in intervals:
            for boundary in interval.fields.values():
                for side in (boundary.west, boundary.east,
                             boundary.south, boundary.north):
                    fields.extend(array for _, array in side.array_items())
    boundary = spans(fields)
    store_bytes = _span_bytes(store)
    return {"host_store_bytes": store_bytes,
            "host_boundary_bytes": _span_bytes(store + boundary) - store_bytes}


def inventory_shapes(cfg):
    """Conservative carrier census from the resident allocation inventories.

    Physics persistents include rebuilt operands, so this is an upper bound
    until a prepared bundle supplies its actual carrier inventory.
    """
    from woof.core import preflight as pf
    from woof.state_serialization_contract import serialized_state_attrs
    from woof.io.restart import SERIALIZED_SCRATCH_SLOTS
    shapes = {"state/" + key: shape
              for key, shape in pf.state_array_shapes(cfg).items()
              if key in serialized_state_attrs(cfg)}
    if getattr(cfg, "chem_sets", ""):
        from woof.core.device_inventory import chem_state_array_shapes
        from woof.core.chem_history import history_schema
        from woof.chem_table import load
        from tilestream.sfire_smoke import OLD_MU_KEY, time_row_key
        chem = chem_state_array_shapes(cfg)
        shapes.update({"state/" + key: shape for key, shape in chem.items()
                       if key.startswith(("chem_", "chemdiag_"))})
        shapes[OLD_MU_KEY] = (cfg.ny, cfg.nx)
        shapes.update({time_row_key(row): (cfg.nz, cfg.ny, cfg.nx)
                       for row in load(cfg).transported})
        # Diagnostics can alias carriers; counting them separately bounds
        # snapshots and host stores even when a source adds an output row.
        shapes.update({"diag/chem/" + name: spec[0]
                       for name, spec in history_schema(cfg).items()})
    # Terrain drag is read-only geography, never a carrier exchanged at
    # every step or retained in an output snapshot.  Price it in geography
    # below, and in each rank's resident physics allocation.
    shapes.update({"driver/" + key: shape
                   for key, shape in pf.physics_array_shapes(cfg).items()
                   if not key.startswith("terrain_drag/")})
    shapes.update({"scratch/" + key: shape
                   for key, shape in pf.scratch_slot_registry(cfg).items()
                   if key in SERIALIZED_SCRATCH_SLOTS or key == "refl_10cm"})
    return {key: shape for key, shape in shapes.items()
            if len(shape) >= 2 and shape[-2] in (cfg.ny, cfg.ny + 1)
            and shape[-1] in (cfg.nx, cfg.nx + 1)}


def _arrays(cfg):
    import numpy as np
    return {name: SimpleNamespace(shape=shape, ndim=len(shape),
                                 dtype=np.dtype("float32"),
                                 flags=SimpleNamespace(c_contiguous=True))
            for name, shape in inventory_shapes(cfg).items()}


def priced_template_live_bytes(template, cfg, *, device):
    """A lower bound on the template's already allocated, priced bytes.

    Only names in the resident inventories are eligible. Count overlapping
    contiguous device spans once, so aliases cannot buy room twice. Kernel
    tables, context memory, unpriced objects and other live allocations earn
    no credit. The admission also caps this at its template price.
    """
    from woof.core import preflight as pf

    arrays = [getattr(template, name, None) for name in pf.state_array_shapes(cfg)]
    scratch = getattr(template, "_scratch", {})
    arrays.extend(scratch.get(name) for name in pf.scratch_slot_registry(cfg)
                  if not name.startswith("lbc_"))
    driver = getattr(template, "physics", None)
    for path in pf.physics_array_shapes(cfg):
        value = driver
        for name in path.split("/"):
            value = value.get(name) if isinstance(value, dict) else getattr(value, name, None)
            if value is None:
                break
        arrays.append(value)
    spans = []
    for array in arrays:
        if (getattr(getattr(array, "device", None), "id", None) != device
                or not getattr(getattr(array, "flags", None), "c_contiguous", False)):
            continue
        start = getattr(getattr(array, "data", None), "ptr", None)
        size = getattr(array, "nbytes", 0)
        if start is not None and size > 0:
            spans.append((int(start), int(start) + int(size)))
    return _span_bytes(spans)


def frame_snapshot_budget(cfg) -> int:
    """Optional history-copy capacity shared by admission and the runtime.

    A small requested frame does not prove that later frames will remain
    small. Eligibility is fixed from the complete conservative carrier
    inventory, so a large rank never builds a growing partial snapshot
    cache. Its direct-copy fallback fences the next producer until DMA
    completes. Eligible ranks keep the existing overlap capacity.
    """
    full_inventory_bytes = sum(prod(shape) * 4
                               for shape in inventory_shapes(cfg).values())
    return (FRAME_SNAPSHOT_LIMIT_BYTES
            if full_inventory_bytes <= FRAME_SNAPSHOT_LIMIT_BYTES else 0)


def rank_lake_column_bounds(specs, lake_mask):
    """Per rank, the lake-eligible columns inside its compute window.

    THE BREAKAGE THIS PREVENTS (open-defects ledger A13, 2026-10-05): a
    rank's sparse CLM lake work arrays (``lake/columns``, ``lake/static``,
    ``lake/gather_work``, forcing, outputs, the arena) are allocated at the
    rank's own lake column count (``woof.core.lake.LakeModel``), but
    admission priced them at every column of the rank: 0.75 GiB of price
    per 2x2 rank of the HRRR grid, part of the refusal of full HRRR on four
    32 GB cards.  ``lake_mask`` is any 2-D array that is nonzero wherever
    the run's ``lakemask`` can be 1 (an upper bound is enough; a smaller
    count is never charged).  Windows wrap on periodic axes like a gather.
    """
    import numpy as np
    mask = np.asarray(lake_mask) != 0
    ny, nx = mask.shape
    counts = []
    for spec in specs:
        rows = np.arange(spec.cj0, spec.cj0 + spec.cny) % ny
        cols = np.arange(spec.ci0, spec.ci0 + spec.cnx) % nx
        counts.append(int(mask[np.ix_(rows, cols)].sum()))
    return counts


def estimate_devices(exp, *, options=None, max_map_factor=1.0,
                     forcing_intervals=None, forcing_interval_seconds=3600.0,
                     vram_gib=None, profile=None, inventory=None,
                     geography=None, source=None, profiles=None, budgets=None,
                     streaming_boundaries=False, lake_mask=None):
    """Price rank resident envelopes, packed bands, template, and pinned host.

    The lazy rank APIs own halo and decomposition validation. No device is
    opened here. Repeated ids sum all resident ranks on that card. Optional
    frame snapshots use only the space left by the required allocations in
    ``budgets``; the returned rank limits also bound their runtime buffers.
    A preparation head uses one reusable device interval while retaining
    the complete host series as its segments arrive.  ``lake_mask`` (see
    :func:`rank_lake_column_bounds`) prices each rank's sparse lake arrays
    at its own lake columns instead of every column.
    """
    from woof.core import preflight as pf
    from woof.core.streaming import ranked_halo, ranked_specs
    from woof.core.mynn_pbl_scratch import (
        mynn_pricing_rank_chunk, mynn_rank_chunk_candidates)
    from woof.ingest.prepared_store import default_slab_rows
    from tilestream.harness import tile_config
    from tilestream.multigpu import seam_plan, _bands
    if len(exp.domains) != 1:
        # A tree is priced grid by grid through estimate_devices_tree,
        # which calls this once per split grid as a one-grid experiment.
        refuse_unrouted_devices(exp, "rank memory admission for a nested tree "
                                "(estimate_devices_tree prices a tree)")
    options = exp.devices if options is None else options
    cfg = exp.root.run
    # The byte bound each device rank's pool margin carries (A13), or
    # None when the measured ratio must stand: this suite matches an
    # itemization-gap row, or has no rank-scale receipt
    # (pf.rank_pool_residue_cap_bytes, pf.RANK_POOL_RESIDUE_SUITES).
    rank_cap = pf.rank_pool_residue_cap_bytes([cfg])
    rank_basis = pf.rank_pool_residue_basis([cfg])
    if options.enabled:
        from woof.core.devices import validate_ranked_physics
        validate_ranked_physics(cfg)
    try:
        halo = ranked_halo(cfg, max_map_factor=max_map_factor)
        specs = ranked_specs(cfg, options, halo=halo)
    except (ValueError, RuntimeError) as error:
        raise DevicesRefused(f"[devices] split refused before allocation: {error}") from error
    active = max(int(cfg.spec_zone), int(cfg.relax_zone)) if cfg.specified else 0
    for rank, spec in enumerate(specs):
        if min(spec.interior_nx, spec.interior_ny) < max(halo + 1, 2 * active + 1):
            raise DevicesRefused(
                f"[devices] rank {rank} interior {spec.interior_ny}x{spec.interior_nx} "
                f"is narrower than halo + 1 = {halo + 1} or twice boundary "
                f"frame {active}; refused to prevent a seam reading a neighbor's "
                "halo or a boundary frame without a unique interior")
    ids = options.device_ids()
    cards = {dev: {"card": dev, "ranks": [], "resident_bytes": 0,
                   "seam_bytes": 0, "template_bytes": 0,
                   "frame_snapshot_bytes": 0} for dev in ids}
    lake_counts = (None if lake_mask is None
                   or int(getattr(cfg, "sf_lake_physics", 0) or 0) != 1
                   else rank_lake_column_bounds(specs, lake_mask))

    def resident(run, dev=None, *, tile_buffer=False, loader=False,
                 mynn_chunk=None, lake_columns=None):
        local = replace(exp, devices=DEVICES_OFF,
                        domains=(replace(exp.root, run=run),))
        from woof.boundary_fields import source_boundary_species
        with mynn_pricing_rank_chunk(mynn_chunk):
            estimate = pf.estimate_experiment(
                local, forcing_intervals=(1 if (tile_buffer or streaming_boundaries)
                                   else forcing_intervals),
                forcing_interval_seconds=forcing_interval_seconds,
                vram_gib=vram_gib, profile=(profile if profiles is None else profiles.get(dev)),
                boundary_species=source_boundary_species(source),
                tile_buffer=tile_buffer,
                lake_columns=(None if lake_columns is None
                              else {int(exp.root.grid_id): int(lake_columns)}),
                # A device rank's pool margin is bounded in bytes (A13):
                # the ratio was measured on 3.5-11 GiB single-card
                # subtotals, and on 25-28 GiB ranks every pool receipt
                # measured the residue it prices at 0.40-0.72 GiB.  The
                # loader slab and the retained template are small
                # resident domains and keep the ratio; so does a suite
                # an itemization-gap row raised the headroom for
                # (rank_cap is None: pf.rank_pool_residue_cap_bytes).
                pool_residue_cap_bytes=(None if loader else rank_cap))
        if loader:
            # The store loader never attaches device lateral boundaries:
            # its slabs only restore and initialize the cached analysis.
            # The host series is attached to the ranks, not this template.
            # Keep the other scratch as an upper bound for initialization.
            estimate = replace(estimate, domains=tuple(
                replace(domain, items=tuple(item for item in domain.items
                                            if item.category != "lbc"))
                for domain in estimate.domains))
        return estimate
    arrays = []
    ranks = []
    rank_runs = []
    host_physics = 0
    mynn_candidates = (mynn_rank_chunk_candidates(int(cfg.nz))
                       if int(cfg.bl_pbl_physics) == 5 else ())
    initial_mynn_chunk = mynn_candidates[0] if mynn_candidates else None
    for rank, (dev, spec) in enumerate(zip(ids, specs)):
        run = tile_config(cfg, spec.cnx, spec.cny)
        rank_runs.append(run)
        if int(run.sf_surface_physics) == 3:
            from woof.core.ruc_memory import ruc_pinned_host_bytes
            host_physics += ruc_pinned_host_bytes(
                int(run.nx) * int(run.ny), pf.soil_layer_count(run))
        # Start with the minimum rank workspace, or an exact operator
        # request. A wider resident-rank width is fitted below against all
        # required allocations on its physical card. Loader slabs retain
        # their existing resident policy.
        estimate = resident(run, dev, tile_buffer=True,
                            mynn_chunk=initial_mynn_chunk,
                            lake_columns=(None if lake_counts is None
                                          else lake_counts[rank]))
        cards[dev]["ranks"].append(rank)
        cards[dev]["resident_bytes"] += estimate.peak_envelope_bytes
        if inventory is None:
            arrays.append(_arrays(run))
        else:
            import numpy as np
            arrays.append({key: SimpleNamespace(
                shape=tuple(array.shape[:-2]) +
                      (spec.cny + array.shape[-2] - cfg.ny,
                       spec.cnx + array.shape[-1] - cfg.nx),
                ndim=array.ndim, dtype=array.dtype,
                flags=SimpleNamespace(c_contiguous=True))
                for key, array in inventory.items() if array.ndim >= 2})
        ranks.append({"rank": rank, "card": dev, "compute_shape": [spec.cny, spec.cnx],
                      **({} if lake_counts is None
                         else {"lake_columns_bound": lake_counts[rank]}),
                      "interior_shape": [spec.interior_ny, spec.interior_nx],
                      "resident_bytes": estimate.peak_envelope_bytes})
        if initial_mynn_chunk is not None:
            ranks[-1]["mynn_column_chunk"] = min(
                initial_mynn_chunk, spec.cnx * spec.cny)
        # Runtime eligibility uses this same full-rank census, including
        # halos and staggered faces. Once eligible, the prepared inventory
        # bounds every member a frame can ask for. An ineligible rank uses
        # direct copies for all subsets and allocates no snapshots.
        snapshot_budget = frame_snapshot_budget(run)
        snapshot = min(snapshot_budget,
                       sum(prod(array.shape) * array.dtype.itemsize
                           for array in arrays[-1].values()))
        cards[dev]["frame_snapshot_bytes"] += snapshot
        ranks[-1]["frame_snapshot_bytes"] = snapshot
        ranks[-1]["frame_snapshot_budget_bytes"] = snapshot_budget
    staging = 0
    for seam in seam_plan(specs, halo, nx=cfg.nx, ny=cfg.ny):
        _, size = _bands(seam, arrays[seam.src_gpu], arrays[seam.dst_gpu], cfg.nz,
                         specs[seam.src_gpu], specs[seam.dst_gpu],
                         sorted(arrays[seam.src_gpu]))
        # A channel owns one source send buffer and one destination receive buffer.
        cards[ids[seam.src_gpu]]["seam_bytes"] += size
        cards[ids[seam.dst_gpu]]["seam_bytes"] += size
        # Explicit host transport allocates one pinned band per channel.
        # CUDA's staged peer-copy path owns its staging internally; reserve
        # the same full-band upper bound for it, including auto before peer
        # topology is known. A peer-only request or a same-card seam needs
        # no host staging. Omitting auto hid staged transfers from host RAM
        # admission on cards without peer access.
        if (options.transport in ("host", "staged", "auto")
                and ids[seam.src_gpu] != ids[seam.dst_gpu]):
            staging += size
    rows = default_slab_rows(cfg.nx, cfg.ny)
    last = cfg.ny % rows or rows
    template = resident(tile_config(cfg, cfg.nx, last), ids[0], loader=True)
    # The retained template keeps resident arrays, not a second step envelope.
    cards[ids[0]]["template_bytes"] = template.domains[0].resident_bytes
    loader = resident(tile_config(cfg, cfg.nx, rows), ids[0], loader=True).peak_envelope_bytes
    for row in cards.values():
        # WHAT THE CARD NEEDS, kept apart from what it is given (open-defects
        # ledger A3c, 2026-10-05).  Below, the MYNN workspace is widened into
        # whatever the budget leaves, which is a speed choice that never
        # changes a byte; the admitted total therefore tracks the card's size,
        # not the forecast's need.  Read as a need, it made the 1 km runs look
        # like 105 KB per column on 8 x RTX 5090 against 75.5 on 3 x RTX PRO
        # 6000, and that figure went into the CONUS box count.  The required
        # price is the narrowest MYNN width every run may fall back to, which
        # is also the only price that refuses a forecast.
        row["required_resident_bytes"] = row["resident_bytes"]
        required = sum(row[key] for key in
                       ("resident_bytes", "seam_bytes", "template_bytes"))
        row["required_bytes"] = (max(required, loader) if row["card"] == ids[0]
                                 else required)
        budget = None if budgets is None else budgets.get(row["card"])
        if budget is not None and len(mynn_candidates) > 1:
            # Rank threads spend the same Python dispatch and validation
            # work for every column chunk. Wider chunks reduce those gaps,
            # but every rank on a repeated device ID retains its workspace
            # concurrently. Fit their combined complete envelopes, including
            # allocator headroom, before giving optional frames any room.
            for candidate in reversed(mynn_candidates[1:]):
                proposed = {rank: resident(
                    rank_runs[rank], row["card"], tile_buffer=True,
                    mynn_chunk=candidate,
                    lake_columns=(None if lake_counts is None
                                  else lake_counts[rank])).peak_envelope_bytes
                    for rank in row["ranks"]}
                candidate_resident = sum(proposed.values())
                candidate_required = required - row["resident_bytes"] + candidate_resident
                if max(candidate_required,
                       loader if row["card"] == ids[0] else 0) > int(budget):
                    continue
                row["resident_bytes"] = candidate_resident
                required = candidate_required
                for rank, size in proposed.items():
                    ranks[rank]["resident_bytes"] = size
                    ranks[rank]["mynn_column_chunk"] = min(
                        candidate, rank_runs[rank].nx * rank_runs[rank].ny)
                break
        if budget is not None:
            # A large optional snapshot must not refuse a forecast whose
            # state fits. The existing direct copy fences its producer, so
            # reducing this allowance changes overlap, never frame bytes.
            remaining = max(0, int(budget) - required)
            for offset, rank in enumerate(row["ranks"]):
                fair_share = remaining // (len(row["ranks"]) - offset)
                limit = min(ranks[rank]["frame_snapshot_bytes"], fair_share)
                ranks[rank]["frame_snapshot_bytes"] = limit
                remaining -= limit
            row["frame_snapshot_bytes"] = sum(
                ranks[rank]["frame_snapshot_bytes"] for rank in row["ranks"])
        row["mynn_widening_bytes"] = (row["resident_bytes"]
                                      - row["required_resident_bytes"])
        row["total_bytes"] = sum(row[key] for key in
                                 ("resident_bytes", "seam_bytes", "template_bytes",
                                  "frame_snapshot_bytes"))
        if row["card"] == ids[0]:
            row["total_bytes"] = max(row["total_bytes"], loader)
    columns = int(cfg.nx) * int(cfg.ny)
    footprint = {
        "columns": columns,
        "required_bytes": sum(row["required_bytes"] for row in cards.values()),
        "admitted_bytes": sum(row["total_bytes"] for row in cards.values()),
        "mynn_widening_bytes": sum(row["mynn_widening_bytes"]
                                   for row in cards.values()),
        "basis": ("required = every card at the narrowest MYNN width "
                  "(what refuses a forecast and what capacity planning may "
                  "use); admitted = with the MYNN workspace widened into "
                  "free memory and frame snapshots, which change speed and "
                  "overlap, never bytes"),
    }
    footprint["required_kb_per_column"] = round(
        footprint["required_bytes"] / columns / 1000, 2)
    footprint["admitted_kb_per_column"] = round(
        footprint["admitted_bytes"] / columns / 1000, 2)
    # Same shape products used by prepared_store's allocation guard. With a
    # bundle this is its own exact manifest; before preparation use the census.
    if inventory is None:
        # Use the host store's FieldSpec products, also used by prepared_store.
        from tilestream.hoststore import FieldSpec, domain_bytes
        import numpy as np
        def manifest(shapes):
            return tuple(FieldSpec(
                key, np.dtype("float32"), len(shape) == 3 and shape[0] in (cfg.nz, cfg.nz + 1),
                len(shape) == 3 and shape[0] == cfg.nz + 1,
                shape[-2] == cfg.ny + 1, shape[-1] == cfg.nx + 1,
                layers=shape[0] if len(shape) == 3 and shape[0] not in (cfg.nz, cfg.nz + 1) else None)
                for key, shape in shapes.items())
        # Exact products include every leading chemistry species/process axis.
        store = sum(prod(shape) * 4 for shape in inventory_shapes(cfg).values())
        from woof.state_serialization_contract import STATE_SETUP_ARRAYS, STATE_DERIVED_SETUP_ARRAYS
        setup = {key: shape for key, shape in pf.state_array_shapes(cfg).items()
                 if key in STATE_SETUP_ARRAYS + STATE_DERIVED_SETUP_ARRAYS and len(shape) >= 2}
        from woof.core.physics_inventory import terrain_drag_array_shapes
        # These arrays are resident physics inputs on each card and also
        # live in the full-domain host geography gathered by every rank.
        setup.update(terrain_drag_array_shapes(cfg))
        geo = domain_bytes(manifest(setup), cfg.nz, cfg.ny, cfg.nx)
        # The geography owners are driver._SCHEME_GEOGRAPHY, including
        # independently selected radiation components. Scheme coordinates
        # can be float64; bound those operands at eight bytes, not four.
        from woof.config import radiation_scheme_ids
        lw, sw = radiation_scheme_ids(cfg)
        radiation_owners = int(bool(lw or sw)) + (2 if lw != sw and lw and sw else 0)
        coordinate_owners = (radiation_owners
                             + int(cfg.sf_surface_physics == 4)
                             + int(cfg.o3input == 2)
                             + int(int(getattr(cfg, "swint_opt", 0) or 0) in (1, 2)))
        geo += coordinate_owners * 2 * cfg.nx * cfg.ny * 8
        basis = "hoststore FieldSpec products over resident carrier/setup census upper bound"
    else:
        store = sum(int(array.nbytes) for array in inventory.values())
        geo = sum(int(array.nbytes) for array in (geography or {}).values()
                  if hasattr(array, "nbytes"))
        basis = "prepared store and geography allocation inventory"
    intervals = pf.lbc_intervals(exp.run_seconds, forcing_interval_seconds,
                                retained_intervals=forcing_intervals)
    host_boundary = pf.lbc_host_series_bytes(cfg, intervals, source=source)
    return {"options": options.to_json(), "grid": list(options.resolved_grid(cfg.nx, cfg.ny)),
            "halo": halo, "rank_shapes": ranks, "cards": list(cards.values()),
            "rank_pool_residue_bound_bytes": rank_cap,
            "rank_pool_headroom": pf.forecast_pool_headroom([cfg]),
            "rank_pool_basis": rank_basis,
            "host_store_bytes": store + geo, "host_staging_bytes": staging,
            "host_staging_basis": (
                "one pinned band per explicit host channel; full-band upper "
                "bound for CUDA staged or unresolved auto channels"),
            "host_boundary_bytes": host_boundary,
            "host_physics_bytes": host_physics,
            "host_bytes": store + geo + staging + host_boundary + host_physics,
            "host_basis": basis,
            "footprint": footprint}


def tree_grid_as_root(dc):
    """One grid of a tree priced as a single-domain root (memory only).

    A nest's own forcing is its coupler's rolling tables, not a tabulated
    series; pricing it as a specified root prices a boundary series in their
    place, an upper bound of the same order.
    """
    run = replace(dc.run, nested=False, specified=True)
    step = dc.time_step if dc.time_step is not None else max(1, int(dc.run.dt))
    return replace(dc, parent_id=0, i_parent_start=1, j_parent_start=1,
                   parent_grid_ratio=1, parent_time_step_ratio=1, run=run,
                   time_step=step, time_step_fract_num=0,
                   time_step_fract_den=1, start_time=None)


def estimate_devices_tree(exp, *, split_ids=None, forcing_intervals=None,
                          forcing_interval_seconds=3600.0, source=None,
                          vram_gib=None, profile=None, profiles=None,
                          streaming_boundaries=False):
    """Price a split TREE per card: the one pricing the door and the runner share.

    Each split grid is priced the way a split single domain is
    (:func:`estimate_devices`: every slab's resident envelope, its packed
    seam bands and the store-building template), each resident grid as a
    resident domain on the first card of ``ids``, and the pinned host
    stores of the split grids together.  ``woof check --devices``, the
    ``woof go`` gate before the download and the tree runner before its
    first restore all read these numbers, so the three cannot disagree
    about one card.  Every card row carries ``grids``: the bytes each grid
    puts on it.  ``split_ids`` defaults to the tree's own by-name
    validation (:func:`woof.core.devices.validate_tree_devices`), so a
    tree it refuses is refused here too, before anything is priced.
    """
    from woof.boundary_fields import source_boundary_species
    from woof.core import preflight as pf
    from woof.core.devices import validate_tree_devices
    if split_ids is None:
        split_ids = validate_tree_devices(exp)
    split = tuple(int(gid) for gid in split_ids)
    options = exp.devices
    ids = list(dict.fromkeys(options.device_ids()))
    first = ids[0]
    cards = {dev: {"card": dev, "ranks": [], "resident_bytes": 0,
                   "seam_bytes": 0, "template_bytes": 0, "total_bytes": 0,
                   "frame_snapshot_bytes": 0,
                   "grids": {}} for dev in ids}
    host = {"host_store_bytes": 0, "host_staging_bytes": 0,
            "host_boundary_bytes": 0, "host_physics_bytes": 0}
    # Each grid is priced as a one-grid experiment, so the split it carries
    # names no domains list: a list naming another grid of the tree would
    # be refused there as a grid that experiment does not have.
    split_options = replace(options, domains=None)
    grids, halos = [], {}
    for dc in exp.domains:
        gid = int(dc.grid_id)
        local = replace(exp, domains=(tree_grid_as_root(dc),),
                        devices=split_options if gid in split else DEVICES_OFF)
        if gid in split:
            estimate = estimate_devices(
                local, forcing_intervals=forcing_intervals,
                forcing_interval_seconds=forcing_interval_seconds,
                source=source, vram_gib=vram_gib, profile=profile,
                profiles=profiles, streaming_boundaries=streaming_boundaries)
            for row in estimate["cards"]:
                target = cards[row["card"]]
                for key in ("resident_bytes", "seam_bytes", "template_bytes",
                            "frame_snapshot_bytes", "total_bytes"):
                    target[key] += int(row[key])
                target["ranks"].extend(f"d{gid:02d}:{rank}" for rank in row["ranks"])
                target["grids"][gid] = target["grids"].get(gid, 0) + int(row["total_bytes"])
            for key in host:
                host[key] += int(estimate[key])
            halos[gid] = estimate["halo"]
            grids.append({"grid_id": gid, "road": "split", "halo": estimate["halo"],
                          "grid": estimate["grid"],
                          "cards": [{"card": r["card"], "total_bytes": int(r["total_bytes"])}
                                    for r in estimate["cards"]],
                          "host_bytes": int(estimate["host_bytes"])})
        else:
            resident = pf.estimate_experiment(
                local, forcing_intervals=(1 if streaming_boundaries else forcing_intervals),
                forcing_interval_seconds=forcing_interval_seconds,
                vram_gib=vram_gib,
                profile=(profile if profiles is None else profiles.get(first)),
                boundary_species=source_boundary_species(source))
            size = int(resident.peak_envelope_bytes)
            cards[first]["resident_bytes"] += size
            cards[first]["total_bytes"] += size
            cards[first]["grids"][gid] = cards[first]["grids"].get(gid, 0) + size
            if int(dc.run.sf_surface_physics) == 3:
                from woof.core.ruc_memory import ruc_pinned_host_bytes
                host["host_physics_bytes"] += ruc_pinned_host_bytes(
                    int(dc.run.nx) * int(dc.run.ny), pf.soil_layer_count(dc.run))
            grids.append({"grid_id": gid, "road": "resident", "card": first,
                          "total_bytes": size})
    return {"options": options.to_json(), "split_grid_ids": list(split),
            "grids": grids, "halo": halos, "cards": list(cards.values()),
            **host, "host_bytes": sum(host.values()),
            "host_basis": "per split grid: " + ("hoststore FieldSpec products over "
                                                "resident carrier/setup census upper bound")}


def devices_gate(estimate, *, budgets=None, host_budget=None):
    from woof.core import preflight as pf
    lines = []
    refused = False
    for row in estimate["cards"]:
        budget = None if budgets is None else budgets.get(row["card"])
        over = budget is not None and row["total_bytes"] > budget
        refused |= over
        verdict = "UNMEASURED" if budget is None else "REFUSED" if over else "ADMITTED"
        if "grids" in row:
            # A split tree: what each grid puts on this card, then the sum.
            terms = " + ".join(f"d{gid:02d} {size/GIB:.2f} GiB"
                               for gid, size in sorted(row["grids"].items()))
            lines.append(
                f"card {row['card']}: {verdict}: {terms or 'nothing'} "
                f"= {row['total_bytes']/GIB:.2f} GiB; budget "
                + ("unknown" if budget is None else f"{budget/GIB:.2f} GiB"))
            continue
        widened = row.get("mynn_widening_bytes", 0)
        lines.append(
            f"card {row['card']}: {verdict}: resident {row['resident_bytes']/GIB:.2f} GiB "
            + (f"(of which {widened/GIB:.2f} GiB is MYNN workspace widened "
               "into free memory) " if widened else "")
            + f"+ seams {row['seam_bytes']/GIB:.2f} GiB "
            f"+ template {row['template_bytes']/GIB:.2f} GiB "
            f"+ frame snapshots {row['frame_snapshot_bytes']/GIB:.2f} GiB "
            f"= {row['total_bytes']/GIB:.2f} GiB; budget "
            + ("unknown" if budget is None else f"{budget/GIB:.2f} GiB"))
    if "rank_pool_residue_bound_bytes" in estimate:
        # The price above is traceable to its measurement: each rank's
        # pool margin is the measured ratio or the byte bound, whichever
        # is smaller, and the line names the receipts it rests on; a
        # suite an itemization-gap row raised the headroom for carries
        # no bound and the line says why.
        headroom = float(estimate.get("rank_pool_headroom",
                                      pf.FORECAST_POOL_HEADROOM))
        bound = estimate["rank_pool_residue_bound_bytes"]
        lines.append(
            (f"ranks: pool margin per rank is min(x{headroom:.2f} "
             f"of the itemized pool, {bound/GIB:.2f} GiB): "
             if bound is not None else
             f"ranks: pool margin per rank is x{headroom:.2f} of the "
             "itemized pool, no byte bound: ")
            + f"{estimate.get('rank_pool_basis', '')}")
    footprint = estimate.get("footprint")
    if footprint:
        # The capacity figure a reader may scale up, beside the admitted one
        # that fills each card (open-defects ledger A3c).
        lines.append(
            f"devices: {footprint['required_kb_per_column']:.1f} KB per column "
            f"required ({footprint['required_bytes']/GIB:.2f} GiB over "
            f"{footprint['columns']} columns), "
            f"{footprint['admitted_kb_per_column']:.1f} KB admitted; the "
            "difference is MYNN workspace and frame snapshots fitted to free "
            "memory, which change speed, never bytes")
    host_over = host_budget is not None and estimate["host_bytes"] > host_budget
    refused |= host_over
    lines.append(f"host: {'REFUSED' if host_over else 'PRICED'}: pinned store "
                 f"{estimate['host_store_bytes']/GIB:.2f} GiB + staging "
                 f"{estimate['host_staging_bytes']/GIB:.2f} GiB + boundary "
                 f"{estimate['host_boundary_bytes']/GIB:.2f} GiB + physics mirrors "
                 f"{estimate.get('host_physics_bytes', 0)/GIB:.2f} GiB")
    return {"verdict": "\n".join(lines), "refuse": bool(refused), "warn": False,
            "devices": estimate}


def include_preparation(gate, exp, *, source, budgets=None, host_budget=None,
                        forcing_intervals=None, forcing_interval_seconds=3600.0,
                        ingest_forcing_interval_seconds=None, vram_gib=None,
                        profile=None):
    """Keep the existing preparation gate beside the per-card forecast gate."""
    from woof.core import preflight as pf
    kwargs = dict(source=source, forcing_intervals=forcing_intervals,
                  forcing_interval_seconds=forcing_interval_seconds,
                  ingest_forcing_interval_seconds=ingest_forcing_interval_seconds,
                  vram_gib=vram_gib, profile=profile)
    phases = pf.estimate_phases(exp, **kwargs)
    first_budget = None if budgets is None else budgets.get(exp.devices.device_ids()[0])
    moved_to_cpu = False
    if (phases.ingest_priced and first_budget is not None
            and phases.ingest_envelope_bytes > first_budget):
        if getattr(phases, "preprocess_backend", "cuda") == "auto":
            phases = pf.estimate_phases(exp, preprocess_backend="cpu", **kwargs)
            moved_to_cpu = True
        else:
            gate["refuse"] = True
            gate["verdict"] += ("\npreparation: REFUSED: "
                                f"{phases.ingest_envelope_bytes/GIB:.2f} GiB exceeds "
                                f"first card budget {first_budget/GIB:.2f} GiB")
    host_refusal = phases.host_preparation_refusal(host_budget)
    if host_refusal:
        gate["refuse"] = True
        gate["verdict"] += "\npreparation: " + host_refusal
    gate["preparation"] = {"priced": phases.ingest_priced,
                           "backend": getattr(phases, "preprocess_backend", None),
                           "may_move_to_cpu": moved_to_cpu,
                           "device_bytes": phases.ingest_envelope_bytes,
                           "host_refusal": host_refusal}
    return gate
