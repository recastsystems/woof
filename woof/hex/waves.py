"""Shared device wave service for native hex cells and both physics routes.

Every atmospheric column binding reads forcing only. Feedback creates an
explicit surface-layer variant; feedback-off never installs that variant.
"""
from __future__ import annotations

import bisect
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from importlib.resources import files
import json
import math
from pathlib import Path
import threading
import time

import numpy as np

from .wave_options import configuration

PRODUCTS = ("hs", "tp", "tm01", "tm02", "dp", "dir", "dspr", "ws_hs", "ws_tp", "ws_dp",
            "sw1_hs", "sw1_tp", "sw1_dp", "sw2_hs", "sw2_tp", "sw2_dp", "sw3_hs", "sw3_tp",
            "sw3_dp", "ustar", "charnock", "z0", "direction")
_SERVICES = {}
_LOCK = threading.RLock()


def _on_host(value):
    """A host copy of a CuPy array (completing its card's stream), or the host array itself."""
    get = getattr(value, "get", None)
    return np.asarray(get() if callable(get) and not isinstance(value, np.ndarray) else value)


def _utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("_", "T").replace("Z", "+00:00"))
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def for_mesh(mesh, *, output_root, grid_path, start_time, seconds, initial_seconds=0., shared_cards=False):
    """The run's wave service for this output root, built once.

    ``shared_cards``: bindings on several cards of this process share ONE service (on the card that builds it;
    ``HexWaveService.on_card``).  THE BREAKAGE THIS FIXES (hex 0.4.14): the key was (root, card), so the MPAS
    route's local time stepping on several cards (one block of every rate class per card, one process) built one
    whole-mesh service per card, each fed only its own card's cells, and every one of them wrote ``waves/``.
    """
    options = configuration()
    if not options["enabled"]:
        return None
    import cupy as cp
    from .wave_prepared import ensure
    start = _utc(start_time)
    key = ((str(Path(output_root).resolve()), "shared-cards") if shared_cards else
           (str(Path(output_root).resolve()), cp.cuda.Device().id))
    with _LOCK:
        service = _SERVICES.get(key)
        if service is None:
            prepared_grid = getattr(mesh, "static_path", None) or grid_path or getattr(mesh, "grid_path", None)
            if prepared_grid is None:
                raise ValueError("wave preparation needs the native mesh/static file identity")
            prepared = ensure(mesh=mesh, grid_path=prepared_grid,
                start_time=start+timedelta(seconds=initial_seconds),
                end_time=start+timedelta(seconds=seconds), output_root=output_root, options=options)
            service = HexWaveService(mesh, prepared, options, output_root=output_root,
                start_epoch=start.timestamp(), end_seconds=seconds, initial_seconds=initial_seconds)
            if shared_cards:
                # Bindings on other cards start from a finished construction (their threads use other streams).
                cp.cuda.Device().synchronize()
            _SERVICES[key] = service
        elif service.n != int(mesh.dimensions["nCells"]) or service.start_epoch != start.timestamp():
            raise ValueError("a shared wave service was rebound to another mesh or time origin")
        return service


def _arwen_columns(cp, mesh, n, layout, nlocal):
    """A WOOF owner pack's (local owned, owned global, all local global) cell lists; the regional garbage
    column repeats its donor."""
    if layout is None:
        ids = cp.arange(n, dtype=cp.int32)
        local = ids
    else:
        ids = cp.asarray(layout.cell_l2g, dtype=cp.int32)
        local = cp.arange(layout.n_owned_cells, dtype=cp.int32)
    owners = ids[local]
    if nlocal == ids.size+1:
        donor = int(mesh.attrs.get("regional_garbage_donor_cell", n-1))
        ids = cp.concatenate((ids, cp.asarray([donor], dtype=cp.int32)))
    if nlocal != ids.size:
        raise ValueError("wave owner mapping differs from WOOF column extent")
    return local, owners, ids


def attach_arwen(backend, host, schedule, output_root, *, dt, layout=None, ranks=None):
    """Bind a serial/rank/rate-class owner pack to one whole-mesh wave state.

    ``ranks`` is ``(rank, layouts)`` for a same-host rank group of more than one rank (threads or processes):
    rank 0 alone builds the run's service, with one binding per rank, and the coordinator hands it every other
    rank's owned forcing after each step (``couple_rank_group``). THE BREAKAGE THIS PREVENTS: each rank
    process built its own whole-mesh wave model on its own cells' winds (and same-card rank threads one model
    with no writer after the initial frame); every rank wrote the same waves/ files.
    """
    if not configuration()["enabled"]:
        if getattr(backend, "_wave_restart", None) is not None:
            raise ValueError("a wave checkpoint cannot resume with waves off")
        return None
    cp = __import__("cupy")
    mesh = host["prepared"].mesh
    if ranks is not None and len(ranks[1]) > 1 and int(ranks[0]) != 0:
        nlocal = backend._constructor.n_columns
        local, owners, ids = _arwen_columns(cp, mesh, int(mesh.dimensions["nCells"]), layout, nlocal)
        binding = backend.wave_binding = ArwenRankBinding(int(ranks[0]), int(owners.size), int(ids.size))
        if binding.service.options["roughness_feedback"]:
            driver = backend._seam._driver
            driver.ocean = ArwenFeedback(binding, driver.fields["psfc"].shape, getattr(driver, "ocean", None))
        return binding.service.receipt()
    moving = getattr(backend, "_wave_move_restart", None)
    saved = getattr(backend, "_wave_restart", None)
    carried = moving[0] if moving is not None else saved
    origin = (schedule["start_time"] if carried is None else
              datetime.fromtimestamp(carried["start_epoch"], timezone.utc))
    initial = 0. if moving is None else float(carried["clock"])
    end = float(schedule["steps"])*dt
    if moving is not None:
        end += initial
    service = for_mesh(mesh, output_root=output_root, grid_path=host.get("grid_path"),
        start_time=origin, seconds=end, initial_seconds=initial)
    nlocal = backend._constructor.n_columns
    local, owners, ids = _arwen_columns(cp, mesh, service.n, layout, nlocal)
    binding = backend.wave_binding = service.bind(local=local, global_cells=owners,
        all_global_cells=ids, label="arwen")
    if ranks is not None and len(ranks[1]) > 1:
        # Rank 0: one binding per other rank, in rank order; their forcing arrives through the coordinator.
        pad = nlocal - int(layout.n_local_cells)   # the regional garbage column, the same on every rank
        binding.peers = []
        for other in list(ranks[1])[1:]:
            _, other_owners, other_ids = _arwen_columns(cp, mesh, service.n, other, int(other.n_local_cells) + pad)
            binding.peers.append(service.bind(local=cp.arange(other_owners.size, dtype=cp.int32),
                                              global_cells=other_owners, all_global_cells=other_ids, label="arwen"))
    if moving is not None:
        service.restore_moving(*moving)
    elif saved is not None:
        service.restore(saved)
    if service.options["roughness_feedback"]:
        driver = backend._seam._driver
        driver.ocean = ArwenFeedback(binding, driver.fields["psfc"].shape, getattr(driver, "ocean", None))
    service.output(service.clock)
    return service.receipt()


def attach_mpas(session, mesh, arguments, *, start, dt, steps, out, execution=None,
                carried=None, weights=None, source_mesh=None, clock_offset=0., source_carrier=None):
    if not configuration()["enabled"]:
        return None
    # LTS releases the consumed atmosphere mesh after restore. Waves off must
    # never read it; only a carried wave remap needs the source geometry.
    if carried is not None and weights is not None and source_carrier is not None:
        source_mesh = source_carrier["mesh"]
    cp = __import__("cupy")
    # Class and rank executors publish their global mesh and owner maps. A
    # serial run uses the identity map, including a regional solve extent.
    binding_input = getattr(execution, "wave_input", None)
    if binding_input is not None:
        mesh, ids, owned = binding_input()
    else:
        ids = cp.arange(mesh.dimensions["nCells"], dtype=cp.int32)
        owned = int(session.layout.dimensions["nCellsSolve"])
        if owned != ids.size:
            raise ValueError("wave owner mapping is missing from a partitioned MPAS executor")
    origin = start if carried is None else datetime.fromtimestamp(carried["start_epoch"], timezone.utc)
    rank_comm = getattr(execution, "wave_rank_comm", None)
    rank_comm = None if rank_comm is None else rank_comm()
    if rank_comm is not None and rank_comm.Get_size() > 1:
        return _attach_mpas_ranks(rank_comm, execution, session, mesh, arguments, origin=origin,
            seconds=clock_offset+dt*steps, initial=0. if weights is None else clock_offset, out=out,
            carried=carried, weights=weights, source_mesh=source_mesh, owned=owned)
    card = getattr(execution, "wave_service_card", None)
    if card is None:
        service = for_mesh(mesh, output_root=out, grid_path=getattr(arguments, "grid", None),
            start_time=origin, seconds=clock_offset+dt*steps,
            initial_seconds=0. if weights is None else clock_offset)
    else:
        # Local time stepping on several cards (lts_v842): one service per run, built on the first block's card.
        with cp.cuda.Device(int(card)):
            service = for_mesh(mesh, output_root=out, grid_path=getattr(arguments, "grid", None),
                start_time=origin, seconds=clock_offset+dt*steps,
                initial_seconds=0. if weights is None else clock_offset, shared_cards=True)
    ids = np.asarray(ids.get() if hasattr(ids, "get") else ids)
    binding = session.wave_binding = service.bind(local=np.arange(owned, dtype=np.int32),
        global_cells=ids[:owned], all_global_cells=ids[:owned], label="mpas")
    if carried is not None:
        with service.on_card():
            if weights is None:
                service.restore(carried)
            else:
                service.restore_moving(carried, source_mesh, weights)
    service.output(service.clock)
    return binding


#: The forcing a binding hands to the wave service, in the order a rank ships it to rank 0.
FORCING = ("u10", "v10", "psfc", "ust", "rho_air")


def _attach_mpas_ranks(comm, execution, session, mesh, arguments, *, origin, seconds, initial, out,
                       carried, weights, source_mesh, owned):
    """One wave service for an N-rank MPAS run: rank 0 owns it, every rank couples through it.

    THE BREAKAGE THIS PREVENTS: the service registry (``for_mesh``) is per process and per card, so same-host
    rank processes (the Linux default), multi-card ranks and ``--hosts`` ranks each built a whole-mesh wave model
    that saw winds only on its own cells, and every one of them wrote the same ``waves/`` files (2 rank
    processes on x1.40962, 2026-11-06: rank 1 lost its ``points.nc`` rename to rank 0, then the run stopped).
    Now only rank 0 prepares, builds, restores, advances and writes the service. It holds one binding per rank
    (that rank's owned cells, in rank order), and every rank's owned forcing reaches it once per step over the
    rank communicator (``RankWaveBinding``). The same words land in the same whole-mesh buffers as on one card,
    and the service advances at the same clocks, because it advances only when every binding's clock is ready.
    """
    cp = __import__("cupy")
    rank, size = int(comm.Get_rank()), int(comm.Get_size())
    feedback = bool(configuration()["roughness_feedback"])
    if rank != 0:
        coupled = RankWaveBinding(comm, rank, owned, feedback=feedback)
    else:
        service = for_mesh(mesh, output_root=out, grid_path=getattr(arguments, "grid", None),
            start_time=origin, seconds=seconds, initial_seconds=initial)
        partitions = execution.partitions
        if len(partitions) != size:
            raise ValueError("wave rank coupling needs one partition per rank")
        bindings = []
        for partition in partitions:
            ids = cp.asarray(np.asarray(partition.cell_l2g)[:int(partition.n_owned_cells)], dtype=cp.int32)
            bindings.append(service.bind(local=cp.arange(ids.size, dtype=cp.int32), global_cells=ids,
                                         all_global_cells=ids, label="mpas"))
        if int(bindings[0].global_cells.size) != int(owned):
            raise ValueError("rank 0's wave binding differs from its owned cells")
        if carried is not None:
            from .mpasphys.moving_mpi_v842 import is_wave_marker
            if is_wave_marker(carried):
                raise ValueError("rank 0 holds only the marker of the carried wave state, not the state; "
                                 "prevents a resumed wave service that silently starts again")
            if weights is None:
                service.restore(carried)
            else:
                service.restore_moving(carried, source_mesh, weights)
        coupled = RankWaveBinding(comm, 0, owned, feedback=feedback, service=service, bindings=bindings)
        service.output(service.clock)
    coupled.share_roughness()
    session.wave_binding = coupled
    return coupled


class RemoteWaveService:
    """What a rank other than 0 sees of the run's wave service: rank 0 advances, writes and checkpoints it."""

    def __init__(self, rank):
        self.rank = int(rank)
        self.options = configuration()

    def output(self, endpoint):
        return None

    def snapshot(self):
        # Rank 0's checkpoint carries the spectrum; this rank's carries only this marker.
        return dict(schema="gpuwm-hex.waves/remote-rank", owner_rank=0)

    def restore(self, state):
        return None

    def restore_moving(self, *args):
        return None

    def receipt(self):
        return dict(enabled=True, rank=self.rank, owner_rank=0,
                    coupling="owned forcing sent to rank 0's wave service every step; rank 0 writes waves/")


class RankWaveBinding:
    """One rank's side of an N-rank wave coupling (``_attach_mpas_ranks``).

    Every rank calls ``after_mpas_step`` at the same point of every step (the step thread, after the clock is
    committed), so the gather and scatter below line up on all ranks like the other step collectives.
    """

    def __init__(self, comm, rank, owned, *, feedback, service=None, bindings=None):
        self.comm, self.rank, self.owned = comm, int(rank), int(owned)
        self.feedback = bool(feedback)
        self.bindings = bindings
        self.service = service if self.rank == 0 else RemoteWaveService(self.rank)
        self.z0 = None

    def after_mpas_step(self, session, endpoint):
        cp = __import__("cupy")
        fields = _mpas_forcing(session)
        block = cp.stack([cp.asarray(fields[name][:self.owned], dtype=cp.float32) for name in FORCING])
        gathered = self.comm.gather((float(endpoint), None if self.rank == 0 else cp.asnumpy(block)), root=0)
        if self.rank == 0:
            for rank, binding in enumerate(self.bindings):
                if rank:
                    clock, host = gathered[rank]
                    if abs(float(clock) - float(endpoint)) > 1.e-7:
                        raise ValueError(f"wave rank {rank} handed its forcing at {clock} s, rank 0 at {endpoint} s")
                    source = cp.asarray(host)
                else:
                    source = block
                self.service.handoff(binding, {name: cp.ascontiguousarray(source[index])
                                               for index, name in enumerate(FORCING)}, endpoint)
        self.share_roughness()

    def share_roughness(self):
        """Each rank's owned surface roughness from the service's current state (feedback on only)."""
        if not self.feedback:
            return
        cp = __import__("cupy")
        values = None
        if self.rank == 0:
            values = [None] + [cp.asnumpy(binding.roughness()) for binding in self.bindings[1:]]
        mine = self.comm.scatter(values, root=0)
        if self.rank != 0:
            self.z0 = cp.ascontiguousarray(cp.asarray(mine), dtype=cp.float32)

    def roughness(self):
        if not self.feedback:
            return None
        return self.bindings[0].roughness() if self.rank == 0 else self.z0


class ArwenRankBinding:
    """An ArWen-route rank other than 0: it keeps its owned forcing for the coordinator to hand to rank 0."""

    def __init__(self, rank, owned, columns):
        self.rank, self.owned, self.columns = int(rank), int(owned), int(columns)
        self.service = RemoteWaveService(self.rank)
        self.pending = None
        self.z0 = None

    def after_arwen_step(self, backend):
        cp = __import__("cupy")
        source = _arwen_forcing(backend)
        self.pending = (float(backend._seam.elapsed_seconds),
                        cp.asnumpy(cp.stack([cp.asarray(source[name][:self.owned], dtype=cp.float32)
                                             for name in FORCING])))

    def take(self):
        pending, self.pending = self.pending, None
        return pending

    def set_roughness(self, values):
        cp = __import__("cupy")
        self.z0 = None if values is None else cp.ascontiguousarray(cp.asarray(values), dtype=cp.float32)

    def roughness(self):
        return self.z0 if self.service.options["roughness_feedback"] else None


def couple_rank_group(binding, pending):
    """Rank 0 (the coordinator's thread, rank 0's card): hand every other rank's step forcing to the service.

    ``pending`` is the other ranks' ``ArwenRankBinding.take()`` values in rank order. Returns each other rank's
    surface roughness for its local columns (None with feedback off).
    """
    cp = __import__("cupy")
    service = binding.service
    for peer, item in zip(binding.peers, pending):
        if item is None:
            raise ValueError("a wave rank handed no forcing for this step")
        endpoint, host = item
        source = cp.asarray(host)
        service.handoff(peer, {name: cp.ascontiguousarray(source[index]) for index, name in enumerate(FORCING)},
                        float(endpoint))
    if not service.options["roughness_feedback"]:
        return [None] * len(binding.peers)
    return [cp.asnumpy(peer.roughness()) for peer in binding.peers]


def peer_roughness(binding):
    """Rank 0: every other rank's current surface roughness (its first values, before step 1)."""
    if not binding.service.options["roughness_feedback"]:
        return [None] * len(binding.peers)
    cp = __import__("cupy")
    return [cp.asnumpy(peer.roughness()) for peer in binding.peers]


def _arwen_forcing(backend):
    """The WOOF route's wave forcing fields, local column order."""
    cp = __import__("cupy")
    fields = backend._seam._driver.fields
    # The backend keeps the step's lowest-level moist air density for this (its preparation's dry density times
    # 1 + qv, set where the LES winds are kept).
    source = {k: fields[k].reshape(-1) for k in ("u10", "v10", "psfc", "ust")}
    source["rho_air"] = cp.ascontiguousarray(backend._les_columns.wave_rho_air)
    return source


def _mpas_forcing(session):
    """The MPAS route's wave forcing fields, local column order."""
    cp = __import__("cupy")
    return {"u10": session.field("diag_physics", "u10"),
            "v10": session.field("diag_physics", "v10"),
            "psfc": session.field("diag", "surface_pressure"),
            "ust": session.field("diag_physics", "ust"),
            "rho_air": cp.ascontiguousarray(session.field("diag", "rho")[0] *
                (np.float32(1)+session.field("state", "scalars")[session.layout.indices["state", "index_qv"]-1, 0]))}


class HexWaveService:
    def __init__(self, mesh, prepared, options, *, output_root, start_epoch, end_seconds, initial_seconds=0.):
        import cupy as cp
        from woof.ocean.interface import Bathymetry
        from woof.ocean.waves import create_solver
        from woof.ocean.waves.backends.voronoi import VoronoiBackend
        self.cp, self.options, self.mesh = cp, dict(options), mesh
        self.device = int(cp.cuda.Device().id)
        self.output_root = Path(output_root)
        self.start_epoch, self.end_seconds = float(start_epoch), float(end_seconds)
        self.initial_seconds = float(initial_seconds)
        self.prepared = prepared
        manifest, arrays = prepared.manifest, prepared.arrays
        self.n = int(mesh.dimensions["nCells"])
        if manifest["grid"]["ncell"] != self.n or manifest["initial_valid_time"] != self.start_epoch+self.initial_seconds:
            raise ValueError("wave preparation cell extent or initial time differs from the forecast")
        order = mesh.arrays.get("cell_order_original")
        self.order = cp.arange(self.n, dtype=cp.int32) if order is None else cp.asarray(order, dtype=cp.int32)
        self.inverse_order = cp.empty_like(self.order)
        self.inverse_order[self.order] = cp.arange(self.n, dtype=cp.int32)
        geometry = {key: cp.asarray(mesh.arrays[key]) for key in (
            "latCell", "lonCell", "latEdge", "lonEdge", "areaCell", "cellsOnEdge",
            "edgesOnCell", "nEdgesOnCell", "dvEdge", "dcEdge", "angleEdge")}
        # Mesh canonicalizes MPAS connectivity to zero-based, missing=-1.
        # The shared backend consumes native MPAS one-based, missing=0.
        for key in ("cellsOnEdge", "edgesOnCell"):
            geometry[key] = cp.ascontiguousarray(geometry[key]+1, dtype=cp.int32)
        from .wave_prepared import coordinate_difference
        for name, key in (("lat", "latCell"), ("lon", "lonCell")):
            difference, tolerance = coordinate_difference(geometry[key], arrays[name][self.order],
                longitude=name == "lon", xp=cp)
            if not bool(cp.all(cp.isfinite(difference) & (difference <= tolerance))):
                raise ValueError(f"prepared wave {name} points do not match this native mesh in its cell order: "
                    f"maximum difference={float(cp.max(difference)):.9g} deg, "
                    f"maximum encoding allowance={float(cp.max(tolerance)):.9g} deg")
        depth = cp.ascontiguousarray(arrays["depth"][self.order], dtype=cp.float32)
        wet = cp.ascontiguousarray(arrays["wet"][self.order], dtype=cp.float32)
        self.bathy = Bathymetry(-depth, depth, wet, cp.zeros_like(wet),
            manifest.get("bathymetry_datum", "EGM2008"), str(prepared.path / "manifest.json"))
        backend = self.backend = VoronoiBackend(geometry, self.bathy,
            dict(options, sphere_radius_m=float(mesh.attrs.get("sphere_radius", 6371220.0))))
        self.solver = create_solver(geometry, self.bathy, config={
            "backend": "voronoi", "grid_backend": backend,
            "spectral": options["spectral"], "source_pack": options["source_pack"]})
        # Fixed-order sums (the DIA exchange, basin and partition sums), before the first diagnostics: the same
        # forcing gives the same wave words on every run and executor (woof.hex.wave_ordered).
        from .wave_ordered import install as order_reductions
        self.reductions = order_reductions(self.solver)
        self.shape_table = json.loads(files("woof.ocean.waves").joinpath("partition_shapes.v1.json").read_text())
        self.solver.initialize_partitions(cp.ascontiguousarray(arrays["initial_partitions"][self.order]),
            valid_time=self.start_epoch+self.initial_seconds, shape_table=self.shape_table)
        full_boundary = cp.asarray(arrays["boundary_cells"], dtype=cp.int32)
        slots = cp.full(self.n, -1, cp.int32)
        slots[full_boundary] = cp.arange(full_boundary.size, dtype=cp.int32)
        needed = self.order[backend.wet_to_full[backend.boundary_cells]]
        self.boundary_slots = slots[needed]
        if bool(cp.any(self.boundary_slots < 0)):
            raise ValueError("prepared incident spectra omit a native open boundary cell")
        self.boundary = cp.ascontiguousarray(arrays["boundary_partitions"][:, self.boundary_slots])
        self.boundary_times = tuple(float(t)-self.start_epoch for t in manifest["boundary_times"])
        if (backend.boundary_cells.size and (len(self.boundary_times) < 2 or
                self.boundary_times[0] > self.initial_seconds or self.boundary_times[-1] < self.end_seconds)):
            raise ValueError("incident wave spectra do not cover the forecast window")
        stations = manifest["stations"]
        self.station_indices = self.inverse_order[cp.asarray([s["cell"] for s in stations], dtype=cp.int32)]
        self.solver.set_stations(self.station_indices)
        self.buffers = {k: cp.zeros(self.n, dtype=cp.float32) for k in
                       ("u10", "v10", "psfc", "ustar", "tau_x", "tau_y", "rho_air")}
        self.buffers["u10"][...] = arrays["u10_east"][self.order]
        self.buffers["v10"][...] = arrays["v10_north"][self.order]
        self.buffers["rho_air"].fill(1.225)
        from woof.ocean.interface import AtmosphereForcing
        self.solver.set_forcing(AtmosphereForcing(**self.buffers,
            time_seconds=self.start_epoch+self.initial_seconds))
        # The first atmosphere step needs stress from the prepared spectrum
        # and wind, before the first coupling interval has elapsed.
        self.solver.refresh_outputs(refresh_stress=True)
        self.z0, self.alpha = cp.zeros(self.n, cp.float32), cp.zeros(self.n, cp.float32)
        source = files("woof.hex").joinpath("kernels/waves.cu").read_text()
        self.module = cp.RawModule(code=source, options=("--std=c++17", "--use_fast_math"))
        self.forcing_kernel = self.module.get_function("hex_wave_forcing")
        self.remap_kernel = self.module.get_function("hex_wave_remap")
        from woof.ocean.roughness import CUDA as roughness_source
        self.roughness_module = cp.RawModule(code=roughness_source, options=("--std=c++17",))
        self.roughness_kernel = self.roughness_module.get_function("spectral_roughness")
        from woof.certify.kernel_manifest import record_module
        record_module("woof.hex.waves:exchange", source=source,
                      options=("--std=c++17", "--use_fast_math"), module=self.module)
        record_module("woof.hex.waves:roughness", source=roughness_source,
                      options=("--std=c++17",), module=self.roughness_module)
        self.clock = self.initial_seconds
        self.watermarks = {}
        self.bindings = []
        self._boundary_index = None
        self.calls = 0
        self.wall = dict(forcing=0., solver=0., roughness=0.)
        self.events = []
        self.device_ms = 0.
        self._checkpoint_cache = None
        self._last_output = None
        self.point_frames = []
        self.refresh_roughness()
        self.capture_stations()

    @contextmanager
    def on_card(self):
        """Service work, serialised; from another card's thread it runs on the service's card, fully ordered.

        A thread on another card (LTS blocks on several cards share one service, ``for_mesh(shared_cards=True)``)
        issues on that card's default stream, which is not ordered with the service card's own thread: the card is
        synchronised before and after, so neither thread's service work overlaps the other's.  A thread on the
        service's card keeps its stream order and pays nothing.
        """
        cp = self.cp
        with _LOCK:
            if int(cp.cuda.Device().id) == self.device:
                yield
                return
            with cp.cuda.Device(self.device):
                cp.cuda.Device(self.device).synchronize()
                try:
                    yield
                finally:
                    cp.cuda.Device(self.device).synchronize()

    def bind(self, *, local, global_cells, all_global_cells, label):
        cp = self.cp
        device = int(cp.cuda.Device().id)
        with self.on_card():
            label = f"{label}:{len(self.bindings)}"
            binding = WaveBinding(self, cp.asarray(_on_host(local)), cp.asarray(_on_host(global_cells)),
                                  cp.asarray(_on_host(all_global_cells)), label)
            self.bindings.append(binding)
            self.watermarks[label] = self.clock
        binding.device = device
        if device != self.device:
            binding.z0 = cp.empty(binding.all_global_cells.shape, cp.float32)   # on the binding's own card
        return binding

    def handoff(self, binding, fields, endpoint):
        if getattr(binding, "device", self.device) != self.device:
            # Another card's words cross through the host (each .get() completes its own card's stream).
            fields = {name: _on_host(value) for name, value in fields.items()}
            with self.on_card():
                fields = {name: self.cp.ascontiguousarray(self.cp.asarray(value, dtype=self.cp.float32))
                          for name, value in fields.items()}
                self._handoff(binding, fields, endpoint)
            return
        with self.on_card():
            self._handoff(binding, fields, endpoint)

    def _handoff(self, binding, fields, endpoint):
        if not math.isfinite(endpoint) or endpoint < self.watermarks[binding.label]-1.e-7:
            raise ValueError("wave atmospheric owner clock moved backwards")
        began = time.perf_counter()
        self._checkpoint_cache = None
        source = []
        for name in ("u10", "v10", "psfc", "ust", "rho_air"):
            value = fields[name]
            if not isinstance(value, self.cp.ndarray) or value.dtype != self.cp.float32 or not value.flags.c_contiguous:
                raise ValueError(f"wave forcing {name} must be contiguous CUDA float32")
            source.append(value)
        n = binding.local.size
        self.forcing_kernel(((n+127)//128,), (128,),
            (binding.local, binding.global_cells, *source,
             *(self.buffers[k] for k in ("u10", "v10", "psfc", "ustar", "tau_x", "tau_y", "rho_air")),
             np.int32(n)))
        self.wall["forcing"] += time.perf_counter()-began
        self.watermarks[binding.label] = float(endpoint)
        ready = min(self.watermarks.values())
        interval = self.options["coupling_interval_seconds"]
        while ready-self.clock >= interval-1.e-8:
            self.advance_to(self.clock+interval)
        if ready >= self.end_seconds-1.e-8 and ready > self.clock+1.e-8:
            self.advance_to(self.end_seconds)

    def advance_to(self, endpoint):
        from woof.ocean.interface import AtmosphereForcing
        if endpoint > min(self.watermarks.values())+1.e-7:
            raise ValueError("cannot advance waves ahead of atmospheric owners")
        self._checkpoint_cache = None
        cp = self.cp
        while endpoint > self.clock+1.e-8:
            end = endpoint
            if self.backend.boundary_cells.size:
                i = bisect.bisect_right(self.boundary_times, self.clock+1.e-8)-1
                if i < 0 or i+1 >= len(self.boundary_times):
                    raise ValueError("wave boundary interval is unavailable at the coupled clock")
                if i != self._boundary_index:
                    self.solver.set_boundary_partitions(self.boundary[i], self.boundary[i+1],
                        time0=self.start_epoch+self.boundary_times[i],
                        time1=self.start_epoch+self.boundary_times[i+1], shape_table=self.shape_table)
                    self._boundary_index = i
                end = min(end, self.boundary_times[i+1])
            self.solver.set_forcing(AtmosphereForcing(**self.buffers, time_seconds=self.start_epoch+self.clock))
            first, last = cp.cuda.Event(), cp.cuda.Event()
            began = time.perf_counter()
            first.record()
            self.solver.step(end-self.clock)
            last.record()
            self.events.append((first, last))
            self.wall["solver"] += time.perf_counter()-began
            self.clock = end
            self.calls += 1
            if abs(self.clock/3600-round(self.clock/3600)) < 1.e-8:
                self.capture_stations()
        self.refresh_roughness()

    def capture_stations(self):
        epoch = self.start_epoch+self.clock
        if self.point_frames and abs(self.point_frames[-1]["epoch"]-epoch)<1.e-7:
            return
        self.point_frames.append(dict(epoch=epoch,
            fields={k:v.copy() for k,v in self.solver.station_outputs().items() if k in PRODUCTS},
            efth=self.solver.station_spectra().copy()))

    def refresh_roughness(self):
        if not self.options["roughness_feedback"]:
            return
        began = time.perf_counter()
        o = self.solver.outputs
        self.roughness_kernel(((self.n+127)//128,), (128,),
            (o.z0, o.dspr, self.buffers["u10"], self.buffers["v10"], self.buffers["ustar"],
             self.bathy.wet, self.alpha, self.z0, np.int32(self.n)))
        self.wall["roughness"] += time.perf_counter()-began

    def flush(self, endpoint):
        """Output/checkpoint cadence only; clocks must stand at one boundary."""
        if any(abs(t-endpoint) > 1.e-7 for t in self.watermarks.values()):
            raise ValueError("wave output requires all atmospheric owner clocks at its endpoint")
        if endpoint > self.clock+1.e-8:
            self.advance_to(endpoint)

    def collect_timing(self):
        """Called after the existing output transfer has completed, never per step."""
        for first, last in self.events:
            if not last.done:
                raise RuntimeError("collect wave timings after completing the output transfer")
            self.device_ms += self.cp.cuda.get_elapsed_time(first, last)
        self.events.clear()

    def output(self, endpoint):
        with self.on_card():
            return self._output(endpoint)

    def _output(self, endpoint):
        # Class writers arrive separately at a common macro boundary. The
        # first completed set writes once; no class creates a partial map.
        if any(abs(t-endpoint)>1.e-7 for t in self.watermarks.values()):
            return None
        if self._last_output is not None and abs(self._last_output-endpoint)<1.e-7:
            return None
        self.flush(endpoint)
        self.capture_stations()
        from .wave_products import write_frame
        path = write_frame(self)
        self._last_output = endpoint
        return path

    def snapshot(self):
        with self.on_card():
            return self._snapshot()

    def _snapshot(self):
        """One shared host snapshot per clock; repeated class references alias it."""
        if self._checkpoint_cache is None:
            payload = self.solver.restart_state()
            self._checkpoint_cache = dict(schema="gpuwm-hex.waves/v1", clock=self.clock,
                options=self.options, start_epoch=self.start_epoch,
                watermarks=dict(self.watermarks),
                prepared_sha256=self.prepared.sha256,
                prepared_content_sha256=getattr(self.prepared, "content_sha256", None),
                station_mapping_sha256=self.prepared.manifest.get("station_mapping_sha256"),
                solver=dict(metadata=payload["metadata"], arrays={k: self.cp.asnumpy(v) for k,v in payload["arrays"].items()}),
                outputs={k:self.cp.asnumpy(v) for k,v in vars(self.solver.outputs).items()},
                forcing={k: self.cp.asnumpy(v) for k,v in self.buffers.items()},
                wet_to_full=self.cp.asnumpy(self.backend.wet_to_full),
                full_to_wet=self.cp.asnumpy(self.backend.full_to_wet),
                point_stations=[s["id"] for s in self.prepared.manifest["stations"]],
                point_frames=[dict(epoch=f["epoch"],fields={k:self.cp.asnumpy(v) for k,v in f["fields"].items()},
                                   efth=self.cp.asnumpy(f["efth"])) for f in self.point_frames])
        return self._checkpoint_cache

    def restore(self, state):
        if state["schema"] != "gpuwm-hex.waves/v1" or state["options"] != self.options:
            raise ValueError("hex wave restart configuration differs")
        # The prepared arrays must be the same words; their manifest's run-only bytes (the output directory, wall
        # times) may differ, because a resume prepares under its own output root (wave_prepared.content_sha256).
        carried = state.get("prepared_content_sha256")
        same = (carried == getattr(self.prepared, "content_sha256", None) if carried else
                state["prepared_sha256"] == self.prepared.sha256)
        if not same or state["start_epoch"] != self.start_epoch:
            raise ValueError("hex wave restart prepared inputs or clock origin differs")
        if state.get("station_mapping_sha256") != self.prepared.manifest.get("station_mapping_sha256"):
            raise ValueError("hex wave restart station sampling differs")
        payload = state["solver"]
        self.solver.restore_state(dict(metadata=payload["metadata"],
            arrays={k:self.cp.asarray(v) for k,v in payload["arrays"].items()}))
        # The core rebuilds geometry and diagnostics during restore. Its
        # moment evaluation also writes the checkpointed surface iteration.
        # Reinstate the exact warm state and its cached diagnostic outputs,
        # so feedback before the next wave interval sees the saved state.
        for key, target in self.solver.checkpoint_arrays().items():
            self.cp.copyto(target, self.cp.asarray(payload["arrays"][key]))
        outputs = state.get("outputs")
        if outputs is not None:
            targets = vars(self.solver.outputs)
            if set(outputs) != set(targets):
                raise ValueError("hex wave restart output inventory differs")
            for key, target in targets.items():
                value = outputs[key]
                if value.shape != target.shape or value.dtype != target.dtype:
                    raise ValueError("hex wave restart output layout differs: " + key)
                self.cp.copyto(target, self.cp.asarray(value))
        for key, target in self.buffers.items():
            value = state["forcing"][key]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError("hex wave restart forcing layout differs")
            target[...] = self.cp.asarray(value)
        self.clock = float(state["clock"])
        from woof.ocean.interface import AtmosphereForcing
        self.solver.set_forcing(AtmosphereForcing(**self.buffers, time_seconds=self.start_epoch+self.clock))
        self.watermarks = {key:float(state.get("watermarks", {}).get(key, self.clock)) for key in self.watermarks}
        self._boundary_index = None
        self.restore_point_frames(state)
        self.refresh_roughness()
        self._checkpoint_cache = None

    def restore_point_frames(self, state):
        if state.get("point_stations") != [s["id"] for s in self.prepared.manifest["stations"]]:
            # A changed regional station roster writes a separate segment
            # points.nc; old segment observations remain in its own tree.
            return
        self.point_frames = [dict(epoch=f["epoch"],fields={k:self.cp.asarray(v) for k,v in f["fields"].items()},
                                  efth=self.cp.asarray(f["efth"])) for f in state.get("point_frames", [])]

    def restore_moving(self, state, source_mesh, weights):
        """Carry spectra with the atmosphere's validated polygon overlaps."""
        cp = self.cp
        physical = ("spectral", "source_pack", "roughness_feedback", "coupling_interval_seconds",
                    "geographic_cfl", "gse_spreading_seconds")
        if any(state["options"].get(k) != self.options.get(k) for k in physical):
            raise ValueError("moving wave spectra require unchanged physical/numerical options")
        old_time = float(state["start_epoch"])+float(state["clock"])
        if abs(old_time-(self.start_epoch+self.initial_seconds)) > 1.e-6:
            raise ValueError("moving wave carrier time differs from target initialization")
        target = self.solver.restart_state()
        known = {"action", "depth", "water_level", "depth_previous", "surface_iteration",
                 "shelter_iteration", "wind_input_iteration", "clock", "steps"}
        if set(target["arrays"]) - known:
            raise ValueError("moving wave remap needs a declared rule for new core prognostic auxiliary arrays")
        for name in ("source_parameter_sha256", "partition_shapes_sha256"):
            if state["solver"]["metadata"][name] != target["metadata"][name]:
                raise ValueError("moving wave physical parameter digest differs: " + name)
        old = cp.asarray(state["solver"]["arrays"]["action"])
        seed = target["arrays"]["action"]
        if old.shape[1:] != seed.shape[1:]:
            raise ValueError("moving wave spectral bins differ")
        matrix = weights.overlap
        ptr = cp.asarray(matrix.indptr, dtype=cp.int32)
        col = cp.asarray(matrix.indices, dtype=cp.int32)
        overlap = cp.asarray(matrix.data, dtype=cp.float64)
        area = cp.asarray(weights.target_area, dtype=cp.float64)
        out = cp.empty_like(seed)
        nf, nd = seed.shape[1:]
        geometry = tuple(cp.ascontiguousarray(m.arrays[k], dtype=cp.float64)
                         for m in (source_mesh,self.mesh) for k in ("latCell", "lonCell"))
        self.remap_kernel(((seed.size+127)//128,), (128,),
            (old, seed, out, ptr, col, overlap, area, cp.asarray(state["full_to_wet"], dtype=cp.int32),
             self.backend.wet_to_full, *geometry, np.int32(self.backend.ncell), np.int32(nf), np.int32(nd)))
        # The scalar forcing buffers use the same area operator. Geometry and
        # wet mask stay those of the newly prepared member.
        from cupyx.scipy.sparse import csr_matrix
        operator = csr_matrix((overlap, col, ptr), shape=matrix.shape)
        old_wet = cp.asarray(state["wet_to_full"], dtype=cp.int32)
        new_wet = self.backend.wet_to_full

        def compact_scalar(value):
            value = cp.asarray(value)
            trailing = value.shape[1:]
            full = cp.zeros((matrix.shape[1], math.prod(trailing)), cp.float32)
            full[old_wet] = value.reshape(value.shape[0], -1)
            result = (operator @ full)/area[:, None]
            return cp.ascontiguousarray(result[new_wet].reshape((new_wet.size,*trailing)), dtype=cp.float32)

        old_arrays = state["solver"]["arrays"]
        arrays = dict(target["arrays"], action=out)
        arrays["water_level"] = compact_scalar(old_arrays["water_level"])
        arrays["depth"] = cp.maximum(self.bathy.depth[new_wet]+arrays["water_level"], cp.float32(0))
        # A static bathymetric change caused by remeshing is not a temporal
        # water-level change and must not create a shoaling impulse.
        arrays["depth_previous"] = arrays["depth"].copy()
        arrays["surface_iteration"] = compact_scalar(old_arrays["surface_iteration"])
        arrays["steps"] = cp.asarray(old_arrays["steps"])
        for name, value in state["forcing"].items():
            if name not in ("u10", "v10", "tau_x", "tau_y"):
                self.buffers[name][...] = (operator @ cp.asarray(value)) / area
        slat, slon, tlat, tlon = geometry
        se = cp.stack((-cp.sin(slon), cp.cos(slon), cp.zeros_like(slon)))
        sn = cp.stack((-cp.sin(slat)*cp.cos(slon), -cp.sin(slat)*cp.sin(slon), cp.cos(slat)))
        te = cp.stack((-cp.sin(tlon), cp.cos(tlon), cp.zeros_like(tlon)))
        tn = cp.stack((-cp.sin(tlat)*cp.cos(tlon), -cp.sin(tlat)*cp.sin(tlon), cp.cos(tlat)))

        def compact_vector(values):
            values = cp.asarray(values)
            shape = values.shape
            u = cp.zeros((matrix.shape[1], math.prod(shape[1:-1])), cp.float32)
            v = cp.zeros_like(u)
            u[old_wet] = values[...,0].reshape(shape[0],-1)
            v[old_wet] = values[...,1].reshape(shape[0],-1)
            xyz = cp.stack([(operator @ (u*se[k,:,None]+v*sn[k,:,None]))/area[:,None] for k in range(3)])
            mapped = cp.stack((cp.sum(xyz*te[:,:,None],axis=0),cp.sum(xyz*tn[:,:,None],axis=0)),axis=-1)
            return cp.ascontiguousarray(mapped[new_wet].reshape((new_wet.size,*shape[1:])),dtype=cp.float32)

        arrays["shelter_iteration"] = compact_vector(old_arrays["shelter_iteration"])
        arrays["surface_iteration"][:,10:12] = compact_vector(cp.asarray(old_arrays["surface_iteration"])[:,10:12])
        self.solver.restore_state(dict(metadata=target["metadata"], arrays=arrays))
        for east, north in (("u10", "v10"), ("tau_x", "tau_y")):
            u, v = cp.asarray(state["forcing"][east]), cp.asarray(state["forcing"][north])
            xyz = cp.stack([(operator @ (u*se[k]+v*sn[k]))/area for k in range(3)])
            self.buffers[east][...] = cp.sum(xyz*te, axis=0)
            self.buffers[north][...] = cp.sum(xyz*tn, axis=0)
        # wind_input_iteration is a packed uint32 mask, not a scalar field.
        # Keep the target initialization as its warm seed, then rebuild it
        # from the carried action and mapped winds. Restore clears forcing;
        # install the mapped buffers before diagnosing stress or feedback.
        from woof.ocean.interface import AtmosphereForcing
        self.solver.set_forcing(AtmosphereForcing(**self.buffers, time_seconds=old_time))
        self.solver.refresh_outputs(refresh_stress=True)
        self.refresh_roughness()
        self.restore_point_frames(state)
        self._checkpoint_cache = None
        self.move_receipt = dict(operator="positive polygon overlap with spherical direction parallel transport",
            source_wet_cells=int(old.shape[0]), target_wet_cells=self.backend.ncell,
            source_epoch=old_time, target_epoch=self.start_epoch+self.initial_seconds,
            wind_input_mask="rebuilt from remapped spectrum and forcing; packed words are not interpolated",
            uncovered="current prepared incident partitions; no calm-start reset")

    def receipt(self):
        return dict(enabled=True, clock=self.clock, calls=self.calls,
            roughness_feedback=self.options["roughness_feedback"], options=self.options,
            backend=self.backend.receipt(), prepared_sha256=self.prepared.sha256,
            stage_host_launch_seconds=dict(self.wall), solver_device_ms=self.device_ms,
            reductions=dict(order="fixed", supplied_by=getattr(self, "reductions", None)),
            timing_collection="CUDA event spans collected at completed output transfers; no per-step synchronization",
            coupling="one wet-cell spectrum; owner winds gathered on device; advancement waits for all owner clocks")


@dataclass
class WaveBinding:
    service: HexWaveService
    local: object
    global_cells: object
    all_global_cells: object
    label: str

    def __post_init__(self):
        cp = self.service.cp
        self.local = cp.ascontiguousarray(self.local, dtype=cp.int32)
        self.global_cells = cp.ascontiguousarray(self.global_cells, dtype=cp.int32)
        self.all_global_cells = cp.ascontiguousarray(self.all_global_cells, dtype=cp.int32)
        if self.local.shape != self.global_cells.shape or not self.local.size:
            raise ValueError("wave binding needs matching nonempty owner index lists")
        self.z0 = cp.empty(self.all_global_cells.shape, cp.float32)

    def roughness(self):
        if not self.service.options["roughness_feedback"]:
            return None
        service = self.service
        if getattr(self, "device", service.device) != service.device:
            with service.on_card():
                words = _on_host(service.cp.take(service.z0, self.all_global_cells))
            self.z0[...] = service.cp.asarray(words)   # this binding's card (HexWaveService.bind)
            return self.z0
        service.cp.take(service.z0, self.all_global_cells, out=self.z0)
        return self.z0

    def after_arwen_step(self, backend):
        self.service.handoff(self, _arwen_forcing(backend), float(backend._seam.elapsed_seconds))

    def after_mpas_step(self, session, endpoint):
        self.service.handoff(self, _mpas_forcing(session), endpoint)


class ArwenFeedback:
    """Surface-only adapter; the committed hex clock owns wave advancement."""
    def __init__(self, binding, shape, previous=None):
        self.binding, self.shape, self.previous = binding, shape, previous

    @property
    def feedback_z0(self):
        z = self.binding.roughness()
        return None if z is None else z.reshape(self.shape)

    @property
    def z0(self):
        return self.feedback_z0

    @property
    def checkpoint_target(self):
        """The WOOF coupler this adapter wraps, for woof's restart writer (None: nothing to carry there).

        The spectrum is checkpointed by the hex wave service (snapshot/restore, gpuwm-hex.waves/v1).
        """
        return self.previous

    def surface_roughness(self, znt, ustar, *, ice=None):
        if self.previous is not None:
            self.previous.surface_roughness(znt, ustar, ice=ice)
        z = self.feedback_z0
        if z is not None:
            mask = z > 0
            if ice is not None:
                mask &= ice <= .01
            self.binding.service.cp.copyto(znt, z, where=mask)

    def atmosphere_step(self, *args, **kwargs):
        if self.previous is not None:
            self.previous.atmosphere_step(*args, **kwargs)

    def history_fields(self):
        return {} if self.previous is None else self.previous.history_fields()
