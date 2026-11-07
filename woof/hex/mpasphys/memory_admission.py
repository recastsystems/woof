"""Price the MPAS route before warm-up, pool decode or device upload."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
from types import SimpleNamespace

from dataclasses import dataclass

from ..device_admission import (MIB, ROUTE_RUN_MARGIN, model_for_card, mpas_rank_card_bytes,
                                route_required_free_bytes)


@dataclass(frozen=True)
class CoTenantRankRow:
    """Two whole runs of one mesh on one card, same tree: the mesh as one rank, and as ``ranks`` co-tenant ranks.

    Solved for the two terms of a co-tenant rank's price: a fixed cost per rank (its CUDA context, kernel images,
    pool arena and per-rank scratch) and a cost per local column (halos included).
    """

    route: str
    executor: str
    numerics: str
    levels: int
    ranks: int
    card_cells: int        # sum of the ranks' local columns, halos included
    card_peak_mib: float   # the whole card's peak over every rank process of the run (nvidia-smi rows)
    single_cells: int
    single_peak_mib: float
    receipt: str

    def terms_mib(self):
        """(fixed MiB per rank, MiB per local column)."""
        per_column = ((self.ranks * self.single_peak_mib - self.card_peak_mib)
                      / (self.ranks * self.single_cells - self.card_cells))
        return self.single_peak_mib - per_column * self.single_cells, per_column


#: Co-tenant ranks of the MPAS route: several same-host ranks on ONE card.
#:
#: THE BREAKAGE IT PREVENTS (INTAKE 2026-11-06T00:10Z, hex 0.4.13): rank_requirements charged every rank on a card
#: the whole one-member curve (route_required_free_bytes at its own ~5,900 columns, extrapolated below the curve's
#: smallest measured run, 40,962 columns, plus its margin), so 8 ranks of x1.40962 (59 levels) on one 32 GB card were
#: refused before start as needing 105,905,390,889 bytes; the measured run below peaks at a fraction of that.  A card
#: with two or more ranks is priced from these measured terms instead (never below one member of the card's combined
#: columns); one rank per card keeps the one-member price.  The rank-process row (the default executor: a CUDA
#: context and pool per rank) also prices the thread executor, which holds one per process: conservative there.
MPAS_COTENANT_RANK_ROWS: tuple[CoTenantRankRow, ...] = (
    CoTenantRankRow(
        route="mpas", executor="processes", numerics="fast", levels=59, ranks=8,
        card_cells=47_491, card_peak_mib=27_958.0,
        single_cells=40_962, single_peak_mib=13_228.0,
        receipt=("x1.40962 120 km, 59 levels, RRFS hrrrv5 (44c0849 pin), dt 720 s, 12 steps with radiation at steps "
                 "0, 5 and 10, fast numerics, one RTX 5090 (170 SM): 8 same-host rank processes (seed-841 part.8) "
                 "against the same mesh as one rank; whole-card peak over every process of each run, nvidia-smi rows "
                 "every 0.5 s; hex 0.4.14 release measurement, "
                 "2026-11-06")),
)


def cotenant_card_bytes(local_cells, levels, model):
    """(bytes, basis) for two or more ranks sharing one card, from :data:`MPAS_COTENANT_RANK_ROWS`."""
    row = min(MPAS_COTENANT_RANK_ROWS, key=lambda r: abs(r.levels - int(levels)))
    fixed, per_column = row.terms_mib()
    # More levels than the measured row cost proportionally more, as on the one-member curve.
    scale = max(1.0, int(levels) / row.levels)
    ranks, columns = len(local_cells), int(sum(local_cells))
    priced = int(round((ranks * fixed + per_column * columns) * scale * (1.0 + ROUTE_RUN_MARGIN) * MIB))
    # Never above the ranks priced as that many whole one-member runs of their own columns: the row was measured
    # on the RRFS hrrrv5 suite, and on a lighter suite (convection_permitting, re-measured curve) two ranks of a
    # whole 120 km mesh priced from it came out above two whole members.
    priced = min(priced, sum(int(route_required_free_bytes(int(cells), model, route="mpas")) for cells in local_cells))
    floor = int(route_required_free_bytes(columns, model, route="mpas"))
    basis = dict(method="co-tenant ranks from a measured row", ranks=ranks, local_cells=columns,
                 fixed_mib_per_rank=round(fixed, 1), mib_per_local_column=round(per_column, 5),
                 level_scale=round(scale, 4), margin=ROUTE_RUN_MARGIN, priced_bytes=priced,
                 one_member_floor_bytes=floor, row=row.receipt)
    return max(priced, floor), basis


def rank_requirements(local_cells, devices, profiles, levels, *, basis=None):
    """What every card needs for the ranks assigned to it, halos included.

    One rank on a card is one member (route_required_free_bytes).  Two or more ranks on one card are priced as
    co-tenants from a measured row (:func:`cotenant_card_bytes`), not as that many whole members.  ``basis``, a
    dict, receives each card's pricing.
    """
    if len(local_cells) != len(devices):
        raise ValueError("one local cell count is required per rank device")
    by_card = defaultdict(list)
    for cells, device in zip(local_cells, devices):
        by_card[device].append(int(cells))
    required = {}
    for device, cells in by_card.items():
        model = model_for_card(profiles[device], "global", levels=int(levels))
        if len(cells) == 1 and len(by_card) == 1:
            required[device] = route_required_free_bytes(cells[0], model, route="mpas")
            card_basis = dict(method="one member", ranks=1, local_cells=cells[0])
        elif len(cells) == 1:
            # One rank alone on its card, of a member split over several cards: the measured rank-per-card row
            # (device_admission.MPAS_RANK_CARD_ROWS; both-routes RED 4), never above the one-member price.
            required[device] = mpas_rank_card_bytes(cells[0], int(levels), model)
            card_basis = dict(method="one rank per card from a measured row", ranks=1, local_cells=cells[0])
        else:
            required[device], card_basis = cotenant_card_bytes(cells, levels, model)
        if basis is not None:
            basis[device] = card_basis
    return required


def amd_capacity_trial(local_cells, devices, *, global_cells, input_payload_bytes,
                       reserve_bytes=8 * 1024**3):
    """An experimental allocation trial, never a measured AMD capacity fit.

    CUDA local-frame and sampled whole-run fits do not describe ROCm. Until
    an AMD whole-run row exists, charge each context its halo-inclusive share
    of the actual input payload plus a declared 8 GiB startup reserve. This
    is a startup budget, NOT an upper bound on forecast memory. Normal HIP
    allocations still enforce physical free memory; no managed spill or
    allocator substitution is enabled. Whole-run telemetry measures the fit.
    """
    if global_cells <= 0 or input_payload_bytes < 0 or reserve_bytes < 0:
        raise ValueError('invalid AMD capacity-trial dimensions or reserve')
    if len(local_cells) != len(devices):
        raise ValueError('one local cell count is required per rank device')
    required = defaultdict(int)
    for cells, device in zip(local_cells, devices):
        if cells <= 0:
            raise ValueError('rank cell counts must be positive')
        required[device] += math.ceil(input_payload_bytes * int(cells) / global_cells) + reserve_bytes
    return dict(required)


def input_payload_bytes(path):
    """Metadata only: count the pinned init's numeric/S1 words, without decode."""
    from netCDF4 import Dataset
    import numpy as np
    with Dataset(path) as dataset:
        return sum(int(value.size) * np.dtype(value.dtype).itemsize
                   for value in dataset.variables.values()
                   if value.dtype is not str)


def admit(arguments):
    from ..mesh_rows import _dimensions
    from ..drivers.samehost_forecast_v841 import device_list, request_options
    from ..forecast_door import measure_device_memory, read_card_profile

    shape = _dimensions(Path(arguments.init))
    cells, levels = shape["nCells"], shape["nVertLevels"]
    named = getattr(arguments, "samehost_devices", None)
    devices = device_list(named) if named else (0,)
    if named:
        request = SimpleNamespace(grid=Path(getattr(arguments, "grid", None) or arguments.init),
                                  scratch=Path(arguments.out), cells=cells)
        options = request_options(arguments, request)
        local_cells = [int(layout.n_local_cells) for layout in options["layouts"]]
        # Keep the exact proven layout for the run; do not build it twice.
        arguments.samehost_layouts = options["layouts_path"]
        arguments.samehost_part = options["part_path"]
    else:
        local_cells = [cells]
    import cupy as cp
    profiles, memory = {}, {}
    for device in sorted(set(devices)):
        with cp.cuda.Device(device):
            # These probes read the current device; no model arrays or kernels
            # have been allocated when a request is refused here.
            profiles[device] = read_card_profile(device)
            memory[device] = measure_device_memory()
    physical_amd = bool(cp.cuda.runtime.is_hip)
    payload = input_payload_bytes(arguments.init) if physical_amd else None
    pricing = {}
    required = (amd_capacity_trial(local_cells, devices, global_cells=cells,
                                  input_payload_bytes=payload)
                if physical_amd else rank_requirements(local_cells, devices, profiles, levels, basis=pricing))
    record = {"route": "mpas", "levels": levels, "local_cells": local_cells,
              "devices": list(devices), "required_bytes": required,
              "free_bytes": {device: value[0] for device, value in memory.items()},
              "card_pricing": pricing}
    record.update(backend='amd' if physical_amd else 'cuda',
                  qualified_capacity=not physical_amd,
                  capacity_trial=physical_amd,
                  basis=('experimental startup budget: halo-inclusive init payload share + 8 GiB/context; '
                         'forecast peak is unmeasured; normal physical HIP allocator'
                         if physical_amd else 'existing CUDA route footprint model'),
                  input_payload_bytes=payload)
    Path(arguments.out).mkdir(parents=True, exist_ok=True)
    import json
    (Path(arguments.out) / 'device-memory-admission.json').write_text(json.dumps(record, indent=2)+'\n')
    arguments.native_memory_admission = record
    for device, need in required.items():
        free = memory[device][0]
        if need > free:
            raise ValueError(f"MPAS device admission: card {device} needs {need} bytes for its ranks "
                             f"(halos included){_pricing_words(pricing.get(device))}, but has {free} free; "
                             f"use more cards or a smaller mesh")
    return record


def _pricing_words(how):
    """The co-tenant price in words for a refusal ('' for one rank on the card)."""
    if not how or how.get("ranks", 1) < 2:
        return ""
    return (f" as {how['ranks']} co-tenant ranks on one card, {how['local_cells']:,} local columns: "
            f"{how['fixed_mib_per_rank']:,.0f} MiB per rank + {how['mib_per_local_column']:.4f} MiB per column"
            f" (x{how['level_scale']:.3f} for levels), +{how['margin']:.0%}, from a measured "
            f"{MPAS_COTENANT_RANK_ROWS[0].ranks}-rank run")
