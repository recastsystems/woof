"""Resident ranked domain with one host store and one domain clock."""
from __future__ import annotations

from dataclasses import fields, replace
import os
import time
from queue import Queue
import sys
from threading import Thread
from time import perf_counter

from tilestream import driver, gather, harness, multigpu, spec
from tilestream import physics_inventory as physics


class RankedRunError(driver.TiledRunError):
    pass


def spans_numa_nodes(nodes):
    """True when the known NUMA nodes of a run's cards are not all one node."""
    return len({int(n) for n in (nodes or {}).values()
                if n is not None and int(n) >= 0}) > 1


def choose_transports(devices, requested, peers, nodes=None, links=None):
    """Resolve ordered card pairs without allocating or importing CUDA.

    ``auto`` takes peer copies when both directions can reach the other card,
    EXCEPT when the run's cards sit on more than one NUMA node: then every
    cross-card pair is staged.  THE BREAKAGE THIS PREVENTS, measured
    2026-10-03 on a two-socket 4 x RTX PRO 6000 box (cards 0-2 on node 0,
    card 3 on node 2): the HRRR 2x2 exchange's eight concurrent 140 MiB peer
    copies took 245 ms, against 14.8 ms staged; peer copies inside one node
    or only across the link were each fast alone (17 and 8 ms), and it is
    the mix that collapses.  The halo exchange sits between every two model
    steps, so four cards ran slower than two.  ``nodes`` maps card to its
    sysfs NUMA node (None or -1 when unknown, which never counts as a node).
    """
    if requested not in ("auto", "peer", "staged", "host"):
        raise RankedRunError(f"unknown ranked transport {requested!r}")
    cross_node = requested == "auto" and spans_numa_nodes(nodes)
    paths = {}
    for src in devices:
        for dst in devices:
            pair = (int(src), int(dst))
            both = peers.get(pair, False) and peers.get(pair[::-1], False)
            if src == dst:
                paths[pair] = "local"
            elif requested == "peer" and not both:
                raise RankedRunError(
                    f"peer transport refused for cards {src} and {dst}: both "
                    "directions must support peer access to prevent a staged "
                    "copy being reported as direct peer transport")
            else:
                direct_link = links is not None and links.get(pair, {}).get("type") == "xgmi"
                paths[pair] = ("peer" if both and (not cross_node or direct_link) else "staged") \
                    if requested == "auto" else requested
                if links is not None and paths[pair] == "staged":
                    # HIP need not stage memcpyPeerAsync when peer access is
                    # unavailable. The explicit pinned two-leg route does.
                    paths[pair] = "host"
    return paths


def _gil_enabled_now():
    from woof.free_threading import gil_enabled
    return gil_enabled()


def rank_placements(devices):
    """Per rank: the host CPUs its thread runs on, from the card's own locality.

    THE BREAKAGE THIS PREVENTS: on a two-socket box the rank threads and the
    pinned buffers they fill floated across sockets (measured on a 4 x RTX
    PRO 6000 box: rank threads seen on NUMA node 2 while cards 0-2 sit on
    node 0), so every launch, readback and pinned copy for a card crossed the
    socket link.  Each rank's thread is bound to the CPUs sysfs lists as local
    to its card, intersected with what this process may use.

    Nothing is bound on a one-node box or when the firmware reports no
    locality (``numa_node`` -1 is "unknown", never node 0); the receipt says
    which.  Binding changes where host code runs, never what it computes.
    """
    rows = [dict(card=int(dev), numa_node=None, cpus=None, applied=False,
                 reason=None) for dev in devices]
    if not sys.platform.startswith("linux") or not hasattr(os, "sched_setaffinity"):
        for row in rows:
            row["reason"] = "thread CPU affinity is unavailable on this platform"
        return rows
    from tilestream import node_probe
    try:
        local = {int(item["device"]): item for item in node_probe.gpu_affinity()}
        allowed = set(os.sched_getaffinity(0))
    except Exception as exc:  # locality is advice; an unreadable probe binds nothing
        for row in rows:
            row["reason"] = f"card locality could not be read: {exc}"
        return rows
    for row in rows:
        item = local.get(row["card"])
        if item is None:
            row["reason"] = "card absent from the locality probe; left unbound"
            continue
        row["numa_node"] = item["numa_node"]
        if not item["binding_required"]:
            row["reason"] = "one NUMA node, or no locality reported; left unbound"
            continue
        cpus = sorted(allowed.intersection(item["local_cpus"] or ()))
        if not cpus:
            row["reason"] = "no card-local CPU is allowed to this process; left unbound"
            continue
        row["cpus"] = cpus
    return rows


_GIL_NOTICE_GIVEN = False


def _gil_notice(ranks, report):
    """Say once per process that rank threads are taking turns."""
    global _GIL_NOTICE_GIVEN
    if ranks < 2 or not report["gil_enabled"] or _GIL_NOTICE_GIVEN:
        return
    _GIL_NOTICE_GIVEN = True
    if report["free_threaded_build"]:
        why = ("this free-threaded Python re-enabled its interpreter lock "
               "(PYTHON_GIL=%s); start it with PYTHON_GIL=0"
               % (report["python_gil_env"] or "unset, and an extension import"))
    else:
        why = ("Python %s is a GIL build; run under a free-threaded "
               "python3.14t to step the cards at once (install.sh and "
               "install.ps1 make .venv on it by default; woof doctor "
               "prints the steps)" % report["python"])
    print(f"[devices] {ranks} rank threads share one interpreter lock, so the "
          f"cards take turns on the host: {why}", file=sys.stderr, flush=True)


class RankedRun(multigpu.MultiGPUDomain):
    """A rank stays resident; host transfers occur only at an exposed seam."""
    ranked = True

    def __init__(self, store, cfg, *, options, scalars, geography, template,
                 clock=None, seam="zeros", boundaries=None,
                 check_geography=True, step_mode="threads", halo=None,
                 nest_hook=None, snapshot_limits=None, mynn_column_chunks=None,
                 _unsafe_short_halo=False, global_keys=()):
        from woof.core.devices import validate_ranked_physics
        validate_ranked_physics(cfg)
        import cupy as cp
        from woof.core import streaming
        from woof.core.adaptive_clock import maximum_map_factor

        if step_mode not in ("threads", "sequential"):
            raise RankedRunError(f"unknown ranked step mode {step_mode!r}")
        self.cfg = cfg
        self.step_mode = step_mode
        self.nz, self.ny, self.nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
        self.devices = list(options.device_ids())
        self.ngpu = self.nbuffers = len(self.devices)
        from woof.free_threading import host_threads_report
        self.host_threads = host_threads_report()
        _gil_notice(self.ngpu, self.host_threads)
        self.placements = rank_placements(self.devices)
        from woof.core.devices_memory import FRAME_SNAPSHOT_LIMIT_BYTES
        self._snapshot_limit = FRAME_SNAPSHOT_LIMIT_BYTES
        if snapshot_limits is None:
            snapshot_limits = [self._snapshot_limit] * self.ngpu
        if (len(snapshot_limits) != self.ngpu
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value < 0 for value in snapshot_limits)):
            raise RankedRunError(
                "frame snapshot limits must give one non-negative byte count "
                "per rank; otherwise output allocation could exceed admission")
        self._snapshot_limits = tuple(snapshot_limits)
        if mynn_column_chunks is None:
            mynn_column_chunks = (None,) * self.ngpu
        if (len(mynn_column_chunks) != self.ngpu
                or any(value is not None and (type(value) is not int or value < 1)
                       for value in mynn_column_chunks)):
            raise RankedRunError(
                "MYNN widths must give one positive column count or None per "
                "rank; otherwise scratch allocation could differ from admission")
        self._mynn_column_chunks = list(mynn_column_chunks)
        self.grid = options.resolved_grid(self.nx, self.ny)
        self.periodic_x, self.periodic_y = streaming._periodic_axes(cfg)
        self.periodic = self.periodic_x and self.periodic_y
        need = streaming.ranked_halo(
            cfg, max_map_factor=maximum_map_factor(geography=geography))
        self.halo = need if halo is None else int(halo)
        if self.halo < need and not _unsafe_short_halo:
            raise RankedRunError(
                f"ranked halo {self.halo} is below required halo {need}; "
                "seam fiction would reach owned forecast cells")
        self.specs = streaming.ranked_specs(cfg, options, halo=self.halo)
        spec.validate_plan(self.specs, self.ny, self.nx)
        multigpu.validate_forced_plan(cfg, self.specs, self.halo, boundaries,
                                      enforce_halo=not _unsafe_short_halo,
                                      nest_forcing=nest_hook is not None)
        self.seams = multigpu.seam_plan(self.specs, self.halo)
        self.boundaries = boundaries
        self._boundary_clock = clock
        # The caller owns source waits and clock changes. Resolve the first
        # interval before a rank constructor can wrap those exceptions.
        self._require_boundary_interval(float(scalars.get("elapsed_seconds", 0.0)))
        self.transport = options.transport
        available = cp.cuda.runtime.getDeviceCount()
        if any(d < 0 or d >= available for d in self.devices):
            raise RankedRunError(f"rank cards {self.devices} exceed {available} available cards")
        peers = multigpu.peer_access_matrix(self.devices)
        self._peers = peers
        self._card_nodes = {row["card"]: row["numa_node"] for row in self.placements}
        links = None
        if cp.cuda.runtime.is_hip:
            from woof.hip_runtime import peer_links
            links = peer_links(self.devices)
        self._paths = choose_transports(self.devices, self.transport, peers,
                                        nodes=self._card_nodes, links=links)
        self._links = links
        original_inventory = streaming.streamed_store_inventory()
        self.global_keys = tuple(global_keys)
        if not self.global_keys:
            # The 0-d domain-owned carriers when the caller names none
            # (see tilestream.driver.TiledRun).
            from tilestream.global_inventory import global_keys as _owned
            self.global_keys = tuple(key for key in _owned(store)
                                     if getattr(store[key], "ndim", 1) == 0)
        if self.global_keys:
            from tilestream.global_inventory import global_keys as domain_global_keys
            if set(self.global_keys) != set(domain_global_keys(store)):
                raise RankedRunError("ranked global carriers must match their single domain owners")
        def rank_inventory(state, names=None):
            return {key: value for key, value in original_inventory(state, names).items()
                    if key not in self.global_keys}
        self.inventory_fn = rank_inventory
        self.volatile_inventory = True
        self._home = store
        self.scalars = scalars
        self._clock = dict(scalars)
        self.geography_fields = list(geography)
        self.sub_cfgs = [harness.tile_config(cfg, s.cnx, s.cny) for s in self.specs]
        self.tile_cfg = self.sub_cfgs[0]
        self.tiles = self.states = []
        self.compute_streams, self.copy_streams, self.unpack_streams = [], [], []
        if nest_hook is not None and boundaries is not None:
            raise RankedRunError("a nested ranked domain takes its forcing from its "
                                 "parent each parent step; it cannot also carry "
                                 "tabulated lateral boundaries")
        tables = None if boundaries is None else streaming.tile_boundary_tables(
            boundaries, self.specs, seam=seam)
        hook = None if tables is None else streaming.make_tile_hook(
            tables, domain_state=None, external_clock=clock)
        # A NEST's slab forcing is windowed out of the rolling tables its
        # coupler's FORCE attaches to the nest's own state, and the first
        # FORCE comes after this constructor (the executor guarantees FORCE
        # before the nest's first STEP).  So the hook runs at each slab's
        # first step instead of here (``_step_rank``); after that the
        # launch-time reload keeps the slab current, as on the [tiles] road.
        self._nest_hook = nest_hook
        self._nest_attached = [False] * len(self.devices)
        flags = driver.geography_scalars(geography)
        # Where the time before step 1 goes, per rank: the buffer build and
        # the two gathers.  The startup profile had 84-92 s of
        # idle cards here on HRRR's lattice and could not say in what.
        self.build_report = []
        for rank, dev in enumerate(self.devices):
            try:
                with cp.cuda.Device(dev):
                    t_build = time.perf_counter()
                    factory = streaming.prepared_tile_state_factory(
                        template, cfg, tables0=None if tables is None else tables[rank])
                    tile = factory(self.sub_cfgs[rank])
                    self.tiles.append(tile)
                    t_build = time.perf_counter() - t_build
                    if int(cfg.bl_pbl_physics) == 5:
                        from woof.core.mynn_pbl_scratch import (
                            bind_mynn_rank_chunk, resolve_mynn_tile_column_chunk)
                        chunk = self._mynn_column_chunks[rank]
                        if chunk is None:
                            chunk = resolve_mynn_tile_column_chunk(int(cfg.nz))
                        self._mynn_column_chunks[rank] = bind_mynn_rank_chunk(
                            tile, self.sub_cfgs[rank], chunk)
                    compute = cp.cuda.Stream(non_blocking=True)
                    self.compute_streams.append(compute)
                    self.copy_streams.append(cp.cuda.Stream(non_blocking=True))
                    self.unpack_streams.append(cp.cuda.Stream(non_blocking=True))
                    # Constructor setup uploads on the default stream.
                    cp.cuda.runtime.deviceSynchronize()
                    t_gather = time.perf_counter()
                    driver._pin_scheme_geography(tile)
                    with compute:
                        gather.gather_tile(geography, tile, self.specs[rank], compute,
                                           inventory_fn=driver.geography_inventory, nz=self.nz)
                        for key, value in flags.items():
                            setattr(tile, key, value)
                        if check_geography and rank == 0:
                            driver.assert_geography_gathered(tile, keys=self.geography_fields)
                        if hook is not None:
                            hook(tile, self.specs[rank], rank, compute)
                        gather.gather_tile(store, tile, self.specs[rank], compute,
                                           inventory_fn=self.inventory_fn, nz=self.nz)
                        from tilestream.sfire import configure_tile
                        configure_tile(tile, self.specs[rank])
                        physics.set_carrier_scalars(tile, self._clock)
                    compute.synchronize()
                    t_gather = time.perf_counter() - t_gather
                    self.build_report.append(dict(
                        rank=rank, card=int(dev), buffer_seconds=round(t_build, 2),
                        gather_seconds=round(t_gather, 2)))
                    print(f"[devices] rank {rank} card {dev}: buffer built in "
                          f"{t_build:.1f} s, gathered in {t_gather:.1f} s")
            except BaseException as exc:
                self._drain_after_error()
                raise RankedRunError(
                    f"rank {rank} card {dev} build failed: {exc}") from exc
        # Each rank's inventory is taken on that rank's card: an inventory
        # member may launch work (chem history diagnostics refresh their
        # buffers), and card 0 is current here.  Without this, fire smoke
        # on two cards with no peer path failed to build ("Peer access is
        # unavailable", MULTICARD-CHECK 2026-10-08).
        self.arrays = []
        for dev, tile in zip(self.devices, self.tiles):
            with cp.cuda.Device(dev):
                self.arrays.append(self.inventory_fn(tile))
        self.names = list(self.arrays[0])
        for rank, arrays in enumerate(self.arrays):
            expected_names = set(store) - set(self.global_keys)
            if set(arrays) != expected_names:
                raise RankedRunError(f"rank {rank} inventory differs from host store: "
                                     f"{sorted(set(arrays) ^ expected_names)}")
            if any(a.device.id != self.devices[rank] for a in arrays.values()):
                raise RankedRunError(f"rank {rank} allocated carriers on the wrong card")
        # Template vertical setup may use CuPy's cross-device copy path,
        # which can enable peer access. Resolve the measured seam transport
        # only after those construction copies have drained.
        for (src, dst), path in self._paths.items():
            if path == "staged":
                with cp.cuda.Device(src):
                    try:
                        cp.cuda.runtime.deviceDisablePeerAccess(dst)
                    except cp.cuda.runtime.CUDARuntimeError as exc:
                        if exc.status != 705:  # cudaErrorPeerAccessNotEnabled
                            raise
            if path == "peer":
                with cp.cuda.Device(src):
                    try:
                        cp.cuda.runtime.deviceEnablePeerAccess(dst)
                    except cp.cuda.runtime.CUDARuntimeError as exc:
                        if exc.status != 704:  # cudaErrorPeerAccessAlreadyEnabled
                            raise
        self.smoke_ledger = None
        from tilestream.sfire_smoke import accepts, ranked_accepts, RankedSmokeLedger
        if cfg.chem_sets and not accepts(cfg) and ranked_accepts(cfg):
            self.smoke_ledger = RankedSmokeLedger(self)
        self.fire_names = tuple(name for name in self.names
                               if name.startswith(("fire/", "driver/fire_tendencies/")))
        # diag/chem/... are output-only buffers (aliases of state carriers,
        # or per-column products a frame evaluates on each rank over its whole
        # window and reads interior-only), so no seam carries them.
        self.exchange_names = [name for name in self.names if name not in self.fire_names
                               and not name.startswith("diag/chem/")]
        self._build_channels()
        self._streams = self.compute_streams
        self.observer = None
        self.post_step_hook = streaming.refl_handoff_hook()
        self._closed = self._ahead = self._exposed = False
        # The output road (see download()): which store keys already hold
        # this generation, the in-flight frame download, the per-rank
        # streams it runs on, the callables that release borrowed views of
        # the store before anything writes it again, and the receipt.
        self._fresh = set()
        # Members a frame download copied this generation.  The history
        # writer may still borrow their store views; nothing else can, once
        # this generation's first download has run the store guards.
        self._downloaded = set()
        self._downloads = []
        self._generation = 0
        self._guarded_generation = -1
        self._frame_streams = None
        self._frame_snapshots = [{} for _ in self.devices]
        self._snapshot_done = [None for _ in self.devices]
        from woof.core.devices_memory import (FRAME_SNAPSHOT_LIMIT_BYTES,
                                               frame_snapshot_budget)
        self._snapshot_limit = FRAME_SNAPSHOT_LIMIT_BYTES
        # Admission and allocation use the same whole-rank policy.  An
        # oversized carrier inventory takes direct DMA even when a frame
        # requests a small subset, so admission may safely price no snapshot.
        self._snapshot_budgets = [frame_snapshot_budget(cfg)
                                  for cfg in self.sub_cfgs]
        self._store_guards = []
        self.output_report = dict(full_drains=0, full_drain_seconds=0.0,
                                  window_reads=0, window_read_bytes=0,
                                  window_read_seconds=0.0,
                                  full_gathers=0, frame_downloads=0,
                                  frame_download_bytes=0, frame_issue_seconds=[],
                                  frame_wait_seconds=[], scratch_zeroes=0,
                                  frame_snapshot_bytes=0, fallback_downloads=0)
        self._caller_events = {}
        self._jobs = [Queue() for _ in self.tiles]
        self._done = Queue()
        self._workers = [Thread(target=self._worker, args=(rank,),
                                name=f"rank-{rank}-card-{dev}", daemon=True)
                         for rank, dev in enumerate(self.devices)]
        try:
            for rank, worker in enumerate(self._workers):
                worker.start()
        except BaseException as exc:
            self._drain_after_error()
            for jobs, worker in zip(self._jobs, self._workers):
                if worker.is_alive():
                    jobs.put(None)
            for worker in self._workers:
                if worker.ident is not None:
                    worker.join()
            self._closed = True
            self._release_owners()
            raise RankedRunError(
                f"rank {rank} card {self.devices[rank]} worker startup failed: {exc}") from exc

    def _build_channels(self):
        self.channels = []
        for seam in self.seams:
            src, dst = seam.src_gpu, seam.dst_gpu
            bands, size = multigpu._bands(seam, self.arrays[src], self.arrays[dst],
                self.nz, self.specs[src], self.specs[dst], self.exchange_names)
            self.channels.append(multigpu._SeamChannel(seam, self.devices[src],
                self.devices[dst], bands, size, self._paths[(self.devices[src], self.devices[dst])]))
        self.channel_phases = [[(k, ch) for k, ch in enumerate(self.channels)
                                if ch.seam.phase == phase]
                               for phase in sorted({c.seam.phase for c in self.channels})]
        self._events = None
        self._seam_bytes = sum(ch.nbytes for ch in self.channels)
        self._transfers_per_exchange = sum(ch.n_transfers_packed for ch in self.channels)

    def transport_report(self):
        across = self.transport == "auto" and spans_numa_nodes(self._card_nodes)
        nodes = sorted({int(n) for n in self._card_nodes.values()
                        if n is not None and int(n) >= 0})
        return [dict(src_dev=src, dst_dev=dst, requested=self.transport,
                     can_access_peer=bool(self._peers.get((src, dst), False)),
                     actual=path, transfer_legs=2 if path == "host" else 1,
                     host_legs=(2 if path == "host" else 1 if path == "staged" else 0),
                     **({"hip_link": self._links.get((src, dst))} if self._links is not None else {}),
                     **({"auto_staged_reason": f"cards span NUMA nodes {nodes}"}
                        if across and src != dst and path in ("staged", "host") else {}))
                for (src, dst), path in sorted(self._paths.items())]

    def devices_report(self):
        return dict(ranks=self.ngpu, grid=list(self.grid), devices=list(self.devices),
                    halo=self.halo, rank_shapes=[[s.cny, s.cnx] for s in self.specs],
                    build=[dict(row) for row in getattr(self, "build_report", ())],
                    mynn_column_chunks=list(self._mynn_column_chunks),
                    host_threads=dict(self.host_threads,
                                      gil_enabled_now=_gil_enabled_now()),
                    rank_placements=[dict(row) for row in self.placements],
                    seam_bytes_per_exchange=self._seam_bytes,
                    transfers_per_exchange=self._transfers_per_exchange,
                    transport=self.transport_report())

    def _worker(self, rank):
        placement = self.placements[rank]
        if placement["cpus"]:
            try:
                os.sched_setaffinity(0, placement["cpus"])
                placement["applied"] = True
            except OSError as exc:
                placement["reason"] = f"binding refused by the kernel: {exc}"
        while True:
            job = self._jobs[rank].get()
            if job is None:
                return
            try:
                self._step_rank(rank, *job)
            except BaseException as exc:
                self._done.put((rank, exc))
            else:
                self._done.put((rank, None))

    def _require_boundary_interval(self, elapsed):
        """Seal the current forcing interval before dispatching any rank.

        Later intervals stay lazy. A source failure or a clock change must
        reach the forecast controller unchanged, before any rank steps, so
        it can checkpoint or retry the sealed preparation as appropriate.
        """
        boundaries = self.boundaries
        if (boundaries is not None
                and getattr(boundaries.intervals, "bounds", None) is not None):
            clock = self._boundary_clock
            boundaries.interval_at(float(elapsed if clock is None
                                         else clock.elapsed_seconds))

    def _step_rank(self, rank, kwargs, control, step_function=None):
        import cupy as cp
        from woof.core import dycore
        with cp.cuda.Device(self.devices[rank]), self.compute_streams[rank]:
            tile = self.tiles[rank]
            physics.set_carrier_scalars(tile, self._clock)
            if self._nest_hook is not None and not self._nest_attached[rank]:
                self._nest_hook(tile, self.specs[rank], rank, self.compute_streams[rank])
                self._nest_attached[rank] = True
            if control is not None:
                control.apply(tile)
            stochastic = getattr(self, "_ensemble_stochastic_lease", None)
            if stochastic is not None:
                stochastic.bind_window(tile, self.sub_cfgs[rank], self.specs[rank], rank,
                                       stream=self.compute_streams[rank])
            dycore.set_wrf_cfl_tile_window(self.cfg.grid_id, self.specs[rank])
            function = dycore.step if step_function is None else step_function
            function(tile, self.sub_cfgs[rank], **kwargs)
            dycore.finish_wrf_cfl_tile(self.cfg.grid_id)
            if self.observer is not None:
                self.observer(tile, self.specs[rank], rank, self.compute_streams[rank])
            self.post_step_hook(tile, self.specs[rank], rank, self.compute_streams[rank])

    def _require_open(self):
        if self._closed:
            raise RankedRunError("ranked run is closed; resident ranks were released")

    @property
    def closed(self):
        return self._closed

    @property
    def store(self):
        self._require_open()
        self.drain()
        return self._home

    def checkpoint_store(self):
        """This generation's store, for a checkpoint that only READS it.

        :attr:`store` drains, and a drain first waits for every store guard,
        because it then hands the store out for WRITING.  At an hourly
        checkpoint the guard it waits for is the history frame of the same
        hour, still being written from borrowed store views: about 8 to 12 s
        of a 1 km run's checkpoint stall, measured on the 2026-10-04 CONUS
        run (8 x RTX 5090).  A checkpoint does not write the store, so when
        this generation's first frame download has already run the guards
        (the only views still borrowed are of members downloaded this
        generation) it needs no wait:

        * members already fresh this generation are in the store as is;
        * every other member is copied slab to store by the plan
          :meth:`drain` and :meth:`download` use, so the bytes are a drain's;
        * a downloaded member the slabs changed after its download (the
          history-interval ``UP_HELI_MAX`` reset) is borrowed by the frame
          and stale, so it is copied into a pinned array of its own and the
          store's copy is left to the frame.

        Returns ``{member: host array}``.  The slabs are not touched and the
        store is not handed out for writing, so the next sweep does not
        gather it back.  Every other case is :attr:`store`, unchanged.
        """
        import cupy as cp
        self._require_open()
        if (self._exposed or not self._ahead
                or self._guarded_generation != self._generation):
            return self.store
        self._await_download()
        self.sync_all()
        self.refresh_arrays()
        stale = [k for k in self._home if k in self._downloaded
                 and k not in self._fresh]
        rest = [k for k in self._home if k not in self._fresh
                and k not in self._downloaded]
        side = {k: gather.pinned_empty_like(self._home[k]) for k in stale}
        start = perf_counter()
        for rank, dev in enumerate(self.devices):
            with cp.cuda.Device(dev):
                for names, dst_store in ((rest, self._home), (stale, side)):
                    if not names:
                        continue
                    src = self.inventory_fn(self.tiles[rank], names)
                    dst = {name: dst_store[name] for name in names}
                    plan = gather.make_plan(src, dst, self.specs[rank],
                                            "scatter", nz=self.nz)
                    plan.execute(src, dst, self.compute_streams[rank])
        self.sync_compute()
        self._fresh.update(rest)
        if not stale:
            self._ahead = False
        report = self.output_report
        report["checkpoint_snapshots"] = report.get("checkpoint_snapshots", 0) + 1
        report["checkpoint_snapshot_seconds"] = (
            report.get("checkpoint_snapshot_seconds", 0.0)
            + perf_counter() - start)
        return {**self._home, **side}

    def store_keys(self):
        """The host store's member names, WITHOUT draining it.

        ``key in run.store`` is a full drain on this road (every carrier of
        every slab copied to the host, and the whole store copied back before
        the next step), and the stepper asked it before every reflectivity
        step.  A membership question needs neither.
        """
        self._require_open()
        return self._home.keys()

    @property
    def raw_store(self):
        """The pinned host store as an object, for planning only.

        Nothing may read a member's VALUES through this: they are as fresh as
        the last :meth:`drain` or :meth:`download` that named them and no
        fresher.  It exists for what only needs keys, shapes and dtypes (the
        frame plan) and for the frame road, which reads the members it
        downloaded after waiting for that download.
        """
        self._require_open()
        return self._home

    def add_store_guard(self, guard):
        """Register a callable that must return before the store is written.

        The history writer borrows views of the store for the frame it is
        writing.  On this road the next sweep never writes the store, so the
        writer can keep them while the model steps on; the next write into
        the store (the next frame's download, a full drain) calls every guard
        first, which waits until the writer has released them.
        """
        if not any(g is guard or g == guard for g in self._store_guards):
            self._store_guards.append(guard)

    def _guard_store(self):
        for guard in list(self._store_guards):
            guard()

    def pending_downloads(self):
        """The frame downloads in flight now, as a value a reader can keep."""
        return tuple(self._downloads)

    def wait_downloads(self, downloads):
        """Block until ``downloads`` (from :meth:`pending_downloads`) land.

        Thread-safe: it touches only the events it was handed.  The frame's
        reader calls it on the writer thread, for the downloads its frame
        was issued behind and no others.
        """
        import cupy as cp
        start = perf_counter()
        for download in downloads:
            for dev, event in download["events"]:
                with cp.cuda.Device(dev):
                    event.synchronize()
        waited = perf_counter() - start
        self.output_report["frame_wait_seconds"].append(round(waited, 6))
        return waited

    def _await_download(self):
        """Wait for every frame download in flight (stepping thread only)."""
        pending, self._downloads = self._downloads, []
        return self.wait_downloads(pending) if pending else 0.0

    def download(self, names):
        """Copy the named store members of every slab to the host, async.

        THE OUTPUT ROAD.  A history frame needs a few dozen members of a
        store that holds every carrier, and the store is only a mirror on
        this road: the slabs stay resident and the next step never reads it.
        So a frame does not :meth:`drain` (which copies every carrier of
        every slab to the host and marks the whole store to be copied back
        before the next step); it downloads the members it names:

        * each slab's owned interior of each member, by the same scatter
          plan :meth:`drain` uses, so the bytes that land are the bytes a
          drain would land;
        * into reusable device snapshots on the compute stream, after the
          step and halo exchange, so later kernels cannot change a frame;
        * from those snapshots on a frame stream per rank, while the next
          model step runs. Snapshot reuse waits for its previous host copy.

        Nothing here blocks the host.  :meth:`_await_download` (called by the
        frame's reader, on the writer thread, and before any other host use
        of the store) is the only wait.  Members already fresh this
        generation (an earlier download or a drain) are not copied again.
        Returns the list of members copied.
        """
        import cupy as cp
        self._require_open()
        start = perf_counter()
        names = [n for n in dict.fromkeys(names)
                 if n in self._home and n not in self.global_keys]
        if self._exposed or not self._ahead:
            # The store already holds this generation (a drain landed it, or
            # nothing stepped since), and an exposed store may carry a
            # consumer's write that the next sweep gathers: copying the
            # slabs over it would discard that write.
            return []
        names = [n for n in names if n not in self._fresh]
        if not names:
            return []
        if self._guarded_generation != self._generation:
            # The first download of a generation: any frame the writer still
            # holds views of must be written before these bytes land.  Later
            # downloads of the same generation write other members only.
            self._guard_store()
            self._guarded_generation = self._generation
        # Event queries need the recording card current just like waits.
        # A caller can still be on card 0 while every rank uses other cards.
        pending = []
        for download in self._downloads:
            complete = True
            for dev, event in download["events"]:
                with cp.cuda.Device(dev):
                    if not event.done:
                        complete = False
                        break
            if not complete:
                pending.append(download)
        self._downloads = pending
        if self._frame_streams is None:
            self._frame_streams = []
            for dev in self.devices:
                with cp.cuda.Device(dev):
                    self._frame_streams.append(cp.cuda.Stream(non_blocking=True))
        events = []
        moved = 0
        captures = []
        # Queue every slab's capture before any host DMA. On one card,
        # pitched copies share a copy engine; a host transfer issued first
        # can otherwise delay a later slab's device capture and its step.
        for rank, dev in enumerate(self.devices):
            with cp.cuda.Device(dev):
                compute = self.compute_streams[rank]
                snapshot = self._frame_snapshots[rank]
                # A subsequent frame may reuse these device buffers only
                # after its previous DMA has stopped reading them. This is
                # distinct from the host store's writer-view guard.
                previous = self._snapshot_done[rank]
                if previous is not None:
                    compute.wait_event(previous)
                source = self.inventory_fn(self.tiles[rank], names)
                if any(not array.flags.c_contiguous for array in source.values()):
                    raise RankedRunError(
                        f"rank {rank} output source is not C-contiguous; a raw "
                        "snapshot copy would read different cells than the frame")
                needed = sum(array.nbytes for array in snapshot.values()) + sum(
                    array.nbytes for name, array in source.items() if name not in snapshot)
                limit = min(self._snapshot_limit, self._snapshot_limits[rank],
                            self._snapshot_budgets[rank])
                buffered = 0 < needed <= limit
                if buffered:
                    with compute:
                        for name, array in source.items():
                            if name not in snapshot:
                                snapshot[name] = cp.empty_like(array)
                            cp.cuda.runtime.memcpyAsync(
                                snapshot[name].data.ptr, array.data.ptr,
                                array.nbytes, cp.cuda.runtime.memcpyDeviceToDevice,
                                compute.ptr)
                ready = cp.cuda.Event(disable_timing=True)
                ready.record(compute)
                captures.append((source, buffered, ready))
        for rank, dev in enumerate(self.devices):
            with cp.cuda.Device(dev):
                source, buffered, ready = captures[rank]
                snapshot = self._frame_snapshots[rank]
                compute = self.compute_streams[rank]
                stream = self._frame_streams[rank]
                stream.wait_event(ready)
                src = {name: snapshot[name] for name in names} if buffered else source
                dst = {name: self._home[name] for name in names}
                plan = gather.make_plan(src, dst, self.specs[rank], "scatter", nz=self.nz)
                plan.execute(src, dst, stream)
                moved += int(plan.nbytes)
                done = cp.cuda.Event(disable_timing=True)
                done.record(stream)
                if not buffered:
                    # Ineligible ranks and oversized requests use direct
                    # DMA. It reads live arrays, so the next kernel must wait.
                    compute.wait_event(done)
                    self.output_report["fallback_downloads"] += 1
                self._snapshot_done[rank] = done
                events.append((dev, done))
        self._downloads.append({"events": events, "names": tuple(names)})
        self._fresh.update(names)
        self._downloaded.update(names)
        report = self.output_report
        report["frame_downloads"] += 1
        report["frame_download_bytes"] += moved
        report["frame_snapshot_bytes"] = sum(
            array.nbytes for snapshots in self._frame_snapshots
            for array in snapshots.values())
        report["frame_issue_seconds"].append(round(perf_counter() - start, 6))
        return names

    def zero_scratch(self, key):
        """Zero one store member on every slab, on each slab's compute stream.

        For the running maxima the history route resets after each frame
        (``UP_HELI_MAX``).  Zeroing the HOST copy would need a drain first and
        a full gather after, and would race the frame download that is still
        reading the store; the slabs are where the domain is, so the reset
        goes there, every slab's whole compute window (halo included, so no
        exchange is needed for the halo to agree).  Queued on the compute
        stream, after the frame's device snapshot, so the frame keeps the
        pre-reset values while its download continues. The host copy
        is then stale and is marked so.
        """
        self.fill_scratch(key, 0)

    def fill_scratch(self, key, value, *, plane=None):
        """Fill a carrier (or one leading plane) after its frame snapshot."""
        import cupy as cp
        self._require_open()
        if self._exposed:
            # The store is the authority until the next sweep gathers it.
            self._await_download()
            self._guard_store()
            target = self._home[key]
            (target if plane is None else target[plane])[...] = value
            return
        for rank, dev in enumerate(self.devices):
            with cp.cuda.Device(dev):
                arrays = self.inventory_fn(self.tiles[rank], [key])
            if key not in arrays:
                raise RankedRunError(f"slab {rank} carries no {key!r} to fill")
            with cp.cuda.Device(dev), self.compute_streams[rank]:
                target = arrays[key]
                (target if plane is None else target[plane]).fill(value)
        self._fresh.discard(key)
        self._ahead = True
        self.output_report["scratch_zeroes"] += 1

    def canonical_store(self):
        """Joined carriers plus the resident digest's small forcing weights."""
        import cupy as cp
        arrays = dict(self.store)
        weights = []
        for dev, tile in zip(self.devices, self.tiles):
            value = getattr(tile, "_scratch", {}).get("lbc_weights_0")
            if value is not None:
                with cp.cuda.Device(dev):
                    weights.append(cp.asnumpy(value))
        if weights:
            if len(weights) != len(self.tiles) or any(
                    w.tobytes() != weights[0].tobytes() for w in weights[1:]):
                raise RankedRunError("rank Davies weights disagree; ranks consumed different forcing")
            arrays["scratch/lbc_weights_0"] = weights[0]
        return arrays

    def _order_after_caller(self):
        """Every slab's next kernel waits for what the caller queued since.

        The slabs step on non-blocking streams, which do not order against
        the caller's own stream (the legacy default stream on every card).
        Between two sweeps the caller writes what the slabs read: a nest
        parent's FORCE fills this domain's rolling boundary tables, which
        each slab's launch-time reload copies from.  Without this edge that
        copy could run before FORCE's kernels finish.  One event per card on
        the caller's current stream, waited on by every slab's compute
        stream; nothing blocks the host.
        """
        import cupy as cp
        events = []
        for dev in dict.fromkeys([int(cp.cuda.Device().id), *self.devices]):
            with cp.cuda.Device(dev):
                # One event per card for the run's life, re-recorded each
                # sweep; a per-sweep event is four creations and four
                # cudaEventDestroy calls on the stepping thread every step.
                event = self._caller_events.get(dev)
                if event is None:
                    event = self._caller_events[dev] = cp.cuda.Event(disable_timing=True)
                event.record(cp.cuda.get_current_stream())
                events.append(event)
        for dev, stream in zip(self.devices, self.compute_streams):
            with cp.cuda.Device(dev):
                for event in events:
                    stream.wait_event(event)

    def _gather_exposed(self):
        import cupy as cp
        if not self._exposed:
            return
        for rank, dev in enumerate(self.devices):
            with cp.cuda.Device(dev), self.compute_streams[rank]:
                from tilestream.sfire import configure_tile
                configure_tile(self.tiles[rank], self.specs[rank])
                gather.gather_tile(self._home, self.tiles[rank], self.specs[rank],
                    self.compute_streams[rank], inventory_fn=self.inventory_fn, nz=self.nz)
        self._exposed = False
        self.output_report["full_gathers"] += 1

    def _exchange_fire(self):
        """Join owned native fire cells, then refill physical refined windows.

        Fire refinement and its continuation cells have their own transfer
        geometry. Atmospheric seam bands cannot represent them or the
        terminal native wind faces. The pinned store joins these exact
        windows after all ranks finish, before another fire step reads them.
        """
        if not self.fire_names:
            return
        import cupy as cp
        self._await_download()
        self._guard_store()
        self.sync_compute()
        self.refresh_arrays()
        for rank, dev in enumerate(self.devices):
            with cp.cuda.Device(dev):
                gather.scatter_tile(self.tiles[rank], self._home, self.specs[rank],
                    self.compute_streams[rank], names=self.fire_names,
                    inventory_fn=self.inventory_fn, nz=self.nz)
        self.sync_compute()
        for rank, dev in enumerate(self.devices):
            with cp.cuda.Device(dev):
                gather.gather_tile(self._home, self.tiles[rank], self.specs[rank],
                    self.compute_streams[rank], names=self.fire_names,
                    inventory_fn=self.inventory_fn, nz=self.nz)
        self.sync_compute()

    def read_store(self):
        """Refresh a read-only snapshot without invalidating resident slabs.

        Preserve earlier writable exposure: a reader cannot cancel a
        pending host mutation that the next sweep must upload.
        """
        self._require_open()
        self.drain(read_only=True)
        return self._home

    def read_windows(self, requests):
        """Read one FORCE transaction using a cached staged rank batch."""
        from tilestream.window_batch import read_windows
        return read_windows(self, requests)

    def read_window(self, key, window):
        """Read owned slab cells without exposing or refreshing the host store.

        Return None when the canonical store must supply the operand. Use
        scatter's ownership, including terminal staggered faces and periodic
        aliases, so every word comes from the same rank as a full drain.
        One pinned buffer per overlapping rank holds only requested cells.
        """
        import math
        import cupy as cp

        self._require_open()
        if self._exposed or not self._ahead or key not in self._home:
            return None
        meta = self._home[key]  # Shapes/dtypes only; these values may be stale.
        if meta.ndim < 2 or key in self.global_keys or key.startswith('fire/'):
            return None
        try:
            kind = gather.classify(meta.shape, self.nz, self.ny, self.nx,
                                   layers_ok=True)
        except gather.SpecError:
            return None
        if not isinstance(window, tuple) or len(window) != 2:
            raise ValueError('rank window must be two bounded slices')
        for sl, extent in zip(window, meta.shape[-2:]):
            if (not isinstance(sl, slice) or sl.step not in (None, 1)
                    or sl.start is None or sl.stop is None
                    or not 0 <= sl.start < sl.stop <= extent):
                raise ValueError('rank window is outside the domain')
        y, x = window
        shape = (*meta.shape[:-2], y.stop-y.start, x.stop-x.start)
        depth = math.prod(meta.shape[:-2])
        start = perf_counter()
        self.sync_compute()
        # Re-read the selected inventory: physics may rebind carrier arrays.
        # Plan every rank before issuing any copy, so a missing slab member
        # can fall back without leaving partially issued transfers behind.
        ranks = []
        rects = []
        for rank, dev in enumerate(self.devices):
            blocks = []
            for src_sl, dst_sl in gather.segments_for(self.specs[rank], 'scatter', kind):
                sy, sx = src_sl[-2:]
                dy, dx = dst_sl[-2:]
                j0, j1 = max(y.start, dy.start), min(y.stop, dy.stop)
                i0, i1 = max(x.start, dx.start), min(x.stop, dx.stop)
                if j0 >= j1 or i0 >= i1:
                    continue
                blocks.append((sy.start+j0-dy.start, sx.start+i0-dx.start,
                               j0-y.start, i0-x.start, j1-j0, i1-i0))
                rects.append((j0-y.start, j1-y.start, i0-x.start, i1-x.start))
            if not blocks:
                continue
            with cp.cuda.Device(dev), self.compute_streams[rank]:
                source = self.inventory_fn(self.tiles[rank], (key,)).get(key)
            if source is None:
                return None
            expected = gather.tile_shape_for(kind, meta.shape,
                                             self.specs[rank].cny, self.specs[rank].cnx)
            if (source.shape != expected or source.dtype != meta.dtype
                    or not source.flags.c_contiguous):
                return None
            ranks.append((rank, dev, source, blocks))
        gather._check_destination_rects(rects, shape, require_full=True,
                                        what=f'rank window {key}')
        # Allocation and assembly stay on the caller's coordinator device
        # and stream. Only exact owned overlaps ever leave a slab.
        result = cp.empty(shape, dtype=meta.dtype, order='C')
        coordinator = cp.cuda.get_current_stream()
        for rank, dev, source, blocks in ranks:
            count = depth * sum(h*w for _, _, _, _, h, w in blocks)
            staging = gather.pinned_empty((count,), meta.dtype)
            pieces = []
            offset = 0
            with cp.cuda.Device(dev):
                stream = self.compute_streams[rank]
                for sy, sx, dy, dx, h, w in blocks:
                    size = depth*h*w
                    host = staging[offset:offset+size].reshape((*meta.shape[:-2], h, w))
                    offset += size
                    copy = gather._FieldCopy(
                        key, kind, int(meta.dtype.itemsize),
                        cp.cuda.runtime.memcpyDeviceToHost, source.shape, host.shape,
                        sy, sx, 0, 0, h, w, depth)
                    plan = gather.TilePlan('scatter', self.specs[rank], [copy], None, (key,))
                    plan.execute({key: source}, {key: host}, stream)
                    pieces.append((host, dy, dx, h, w))
                stream.synchronize()
            for host, dy, dx, h, w in pieces:
                copy = gather._FieldCopy(
                    key, kind, int(meta.dtype.itemsize),
                    cp.cuda.runtime.memcpyHostToDevice, host.shape, result.shape,
                    0, 0, dy, dx, h, w, depth)
                plan = gather.TilePlan('gather', self.specs[rank], [copy], None, (key,))
                plan.execute({key: host}, {key: result}, coordinator)
            # Complete uploads before releasing this rank's pinned buffer.
            # No intermediate device array or conversion kernel is needed.
            coordinator.synchronize()
        self.output_report['window_reads'] += 1
        self.output_report['window_read_bytes'] += int(result.nbytes)
        self.output_report['window_read_seconds'] += perf_counter() - start
        return result

    def drain(self, *, read_only=False):
        import cupy as cp
        if self._closed:
            return
        self._await_download()
        # Before the store is written here or handed out for writing: a
        # frame the writer still holds views of keeps its bytes.
        self._guard_store()
        self.sync_all()
        if self._ahead:
            start = perf_counter()
            self.refresh_arrays()
            for rank, dev in enumerate(self.devices):
                with cp.cuda.Device(dev):
                    gather.scatter_tile(self.tiles[rank], self._home, self.specs[rank],
                        self.compute_streams[rank], inventory_fn=self.inventory_fn, nz=self.nz)
            self.sync_compute()
            self._ahead = False
            self.output_report["full_drains"] += 1
            self.output_report["full_drain_seconds"] += perf_counter() - start
        self._exposed = self._exposed or not read_only

    def sync_compute(self):
        import cupy as cp
        for dev, stream in zip(self.devices, self.compute_streams):
            with cp.cuda.Device(dev):
                stream.synchronize()

    def reseed_clock(self, scalars):
        self._require_open()
        self.sync_all()
        self._clock = dict(scalars)
        self.scalars.clear()
        self.scalars.update(self._clock)

    def impose_domain_clock(self, seconds):
        """Integer ticks are the domain authority between model operations."""
        self._clock["elapsed_seconds"] = float(seconds)

    def _drain_after_error(self):
        import cupy as cp
        for dev in dict.fromkeys(self.devices):
            try:
                with cp.cuda.Device(dev):
                    cp.cuda.runtime.deviceSynchronize()
            except BaseException:
                # Preserve the worker exception, but still drain other cards.
                pass

    def _set_live_config(self, incoming):
        from woof.core.adaptive_clock import ADAPTIVE_DERIVED_RUN_FIELDS
        changed = {f.name for f in fields(self.cfg)
                   if getattr(self.cfg, f.name) != getattr(incoming, f.name)}
        structural = changed - ADAPTIVE_DERIVED_RUN_FIELDS
        if structural:
            raise RankedRunError(f"a ranked domain cannot change {sorted(structural)} "
                                 "between steps; rebuild its ranks to prevent stale setup")
        # The same seam fiction the constructor paid (streaming.ranked_halo):
        # a specified or nested slab's seam still writes the specified zone.
        forced = bool(incoming.specified) or bool(getattr(incoming, "nested", False))
        required = multigpu.forced_halo(incoming) if forced else harness.halo_radius(incoming)
        if required > self.halo:
            raise RankedRunError(f"live time_step_sound={incoming.time_step_sound} needs "
                f"halo {required}, but ranks allocated halo {self.halo}; construct with "
                "the adaptive acoustic ceiling to keep seam fiction outside owned cells")
        self.cfg = incoming
        self.sub_cfgs = [replace(cfg, **{name: getattr(incoming, name) for name in changed})
                         for cfg in self.sub_cfgs]
        self.tile_cfg = self.sub_cfgs[0]

    def sweep(self, nsteps=1, *, step_kwargs=None, report=None, progress=None,
              live_config=None, physics_control=None, step_function=None):
        from woof.core import dycore
        self._require_open()
        kwargs = dict(step_kwargs or {})
        for name in ("mass_flux_observer", "mass_flux_accumulator"):
            if kwargs.get(name) is not None:
                raise RankedRunError(f"{name} is DOMAIN-scope; rank compute-window "
                                     "faces are not domain faces and would double-count seam flux")
        if live_config is not None and live_config != self.cfg:
            self.sync_all()
            self._set_live_config(live_config)
        self._gather_exposed()
        self._order_after_caller()
        timing = report is not None and bool(report.get("timing", False))
        step_seconds = exchange_seconds = 0.0
        steps = []
        stochastic = getattr(self, "_ensemble_stochastic_lease", None)
        for index in range(int(nsteps)):
            self._require_boundary_interval(self._clock.get("elapsed_seconds", 0.0))
            if stochastic is not None:
                stochastic.begin(self.cfg, windows=len(self.tiles))
            dycore.begin_wrf_cfl_domain_step(self.cfg, devices=self.devices)
            try:
                if timing:
                    self.sync_all()
                    start = perf_counter()
                schedule = None
                if self.step_mode == "threads":
                    for jobs in self._jobs:
                        jobs.put((kwargs, physics_control, step_function))
                    # A timed sweep keeps the whole exchange after the step
                    # so the two durations stay separable.  So does a run
                    # whose exchange_events was replaced on the instance: the
                    # rank gate's no-exchange and stale-exchange controls and
                    # the delayed-download test replace it, and a sweep that
                    # exchanged behind their back would pass a broken control.
                    replaced = "exchange_events" in vars(self)
                    schedule = None if timing or replaced else self.exchange_schedule()
                    errors = []
                    for _ in self.tiles:
                        rank, exc = self._done.get()
                        if exc is None and schedule is not None and not errors:
                            try:
                                # This slab's seams move while others still step.
                                schedule.rank_ready(rank)
                            except BaseException as failure:
                                exc = failure
                        if exc is not None:
                            errors.append((rank, exc))
                    errors.sort(key=lambda row: row[0])
                    if errors:
                        rank, exc = errors[0]
                        # DRAIN BEFORE ABORTING: queued work may retain arrays.
                        self._drain_after_error()
                        raise RankedRunError(f"rank {rank} card {self.devices[rank]} failed: {exc}") from exc
                else:
                    for rank in range(len(self.tiles)):
                        try:
                            self._step_rank(rank, kwargs, physics_control, step_function)
                        except BaseException as exc:
                            self._drain_after_error()
                            raise RankedRunError(f"rank {rank} card {self.devices[rank]} failed: {exc}") from exc
                dycore.finish_wrf_cfl_domain_step(self.cfg.grid_id)
                if stochastic is not None:
                    self.sync_all()
                    stochastic.finish()
                if timing:
                    self.sync_all()
                    middle = perf_counter()
                    step_seconds += middle - start
                if schedule is None:
                    self.exchange_events()
                else:
                    schedule.finish()
                self._exchange_fire()
                if self.smoke_ledger is not None:
                    self.smoke_ledger.end()
                if timing:
                    self.sync_all()
                    exchange_seconds += perf_counter() - middle
                from tilestream.sfire import GRID_DIAGNOSTICS
                fire_records = [{key: getattr(tile.physics.fire.grid, key) for key in GRID_DIAGNOSTICS}
                    for tile in self.tiles if getattr(getattr(tile, "physics", None), "fire", None) is not None]
                self._clock = driver._advance_clock(self._clock, self.tiles, len(self.tiles)-1,
                    physics, fire_records=fire_records)
                self.scalars.clear()
                self.scalars.update(self._clock)
                self._ahead = True
                self._fresh.clear()
                self._downloaded.clear()
                self._fresh.update(self.fire_names)
                self._fresh.update(self.global_keys)
                self._generation += 1
                steps.append(dict(dt=float(self.cfg.dt), time_step_sound=int(self.cfg.time_step_sound)))
                if progress is not None:
                    for rank, window in enumerate(self.specs):
                        progress(index, rank, window)
            finally:
                dycore.finish_wrf_cfl_domain_step(self.cfg.grid_id, commit=False)
        if report is not None:
            report.update(self.devices_report(), steps=steps, dt=float(self.cfg.dt),
                          time_step_sound=int(self.cfg.time_step_sound))
            if stochastic is not None:
                report["ensemble_stochastic"] = stochastic.receipt()
            if timing:
                report.update(step_seconds=step_seconds, exchange_seconds=exchange_seconds,
                              wall_seconds=step_seconds + exchange_seconds)

    def _release_owners(self):
        self.tiles = self.states = []
        self.arrays = []
        self.channels = []
        self.channel_phases = []
        self._events = None
        self.compute_streams = self.copy_streams = self.unpack_streams = []
        self._streams = []
        self._caller_events = {}
        self._frame_snapshots = []
        self._snapshot_budgets = []
        self._snapshot_done = []
        self._frame_streams = None
        for batch in getattr(self, '_window_batches', {}).values():
            if batch.done is not None:
                import cupy as cp
                with cp.cuda.Device(batch.coordinator_device):
                    batch.done.synchronize()
        self._window_batches = {}
        self._home = None
        self.observer = self.post_step_hook = None
        import gc
        gc.collect()

    def close(self):
        if self._closed:
            return
        # A failed drain leaves the owner intact, as TiledRun.close does.
        # Never free a caller's pinned source while DMA can still reach it.
        self.drain()
        for jobs in self._jobs:
            jobs.put(None)
        for worker in self._workers:
            worker.join()
        self._closed = True
        self._release_owners()
