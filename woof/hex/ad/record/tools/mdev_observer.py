# SPDX-License-Identifier: Apache-2.0
"""(copied 2026-11-13 from the two-card 3 km recorder's mdev_observer.py; only the tools path changed)
multirank (L20; multirank-2 deterministic baton): record_observer.py for a forward whose ranks are threads of one process (--samehost-ranks threads).

The single-rank observer installs its boundary trace with sys.settrace, so only the main thread runs it; with rank
threads the step loop (runtime_v842._run_native_dynamics) runs on the threads hex-rank-<r>_<k> and no boundary is ever
marked. This wrapper:
  1. installs the same trace function in every new thread too (threading.settrace);
  2. makes each step boundary a barrier of all rank threads: every rank parks at the boundary line, then rank 0 runs
     the observer's boundary hooks (binder_mark_step: device snapshot / boundary digest, outer_boundary, the TEMPO
     scheduling capture), then rank 1, ..., then all continue. The device is quiescent (all ranks parked) while the
     snapshot syncs the context, so the boundary is one global boundary of the decomposed forward;
  3. binder_mark_step runs only on rank 0 (one Boundary record per step, as in a one-rank recording).
Also registers faulthandler on SIGUSR1 (all threads) for stall diagnosis.
Env: MR_RANKS (default 2).
"""
import ctypes, faulthandler, linecache, os, re, runpy, signal, sys, threading, time

LOGS = os.environ.get("MR_LOGS") or os.getcwd()  # the recorder sets MR_LOGS (chain.py); a direct run logs where it stands
LOG = f"{LOGS}/fh-{os.getpid()}.txt"
faulthandler.register(signal.SIGUSR1, file=open(LOG, "w"), all_threads=True)
NRANKS = int(os.environ.get("MR_RANKS", "2"))
barrier = threading.Barrier(NRANKS)
RANK = re.compile(r"hex-rank-(\d+)_")

# multirank-2: cyclic garbage is collected only at step boundaries (all rank threads parked), never at an
# allocation-count threshold. THE BREAKAGE THIS PREVENTS: recordings A and B of one argv, with an identical baton grant
# sequence (562 grants, same order), still differed (step 1: 6,118 vs 6,097 commands; a halo descriptor table
# re-uploaded in A only, a cupy module re-loaded at step 2 in A only), because the collector ran at points set by the
# allocation counts of every Python thread, the non-rank ones included (hex-step-records, mpas-output, the tracer's
# host-hash threads, the receipt serialised every 30 s by wall clock): cupy arrays, events and modules held in cycles
# were freed at different commands, so the pool handed later arrays different blocks. MR_GC=auto keeps the default.
import gc  # noqa: E402
GC_BOUNDARY = os.environ.get("MR_GC", "boundary") == "boundary"
if GC_BOUNDARY:
    gc.disable()


def rank_of():
    m = RANK.match(threading.current_thread().name)
    return int(m.group(1)) if m else None


class _MarkerShim:
    """ctypes.CDLL(None) stand-in for observe_cupti_outer: binder_mark_step runs on rank 0 (or a non-rank thread)."""
    def __init__(self):
        self._lib = ctypes.CDLL(None)
        self._real = self._lib.binder_mark_step
        self._real.argtypes = [ctypes.c_uint]
        self._real.restype = ctypes.c_int

    @property
    def binder_mark_step(self):
        real = self._real

        class M:
            argtypes = None
            restype = None

            def __call__(self, step):
                r = rank_of()
                return 0 if (r is not None and r != 0) else real(step)
        return M()

    def __getattr__(self, name):
        return getattr(self._lib, name)


class _CtypesShim:
    def __getattr__(self, name):
        return getattr(ctypes, name)

    def CDLL(self, name, *a, **k):
        return _MarkerShim() if name is None else ctypes.CDLL(name, *a, **k)


def boundary_lines(code):
    lines = linecache.getlines(code.co_filename)
    loop = next(i for i, line in enumerate(lines) if "for step in range(steps+1):" in line)
    before = next(i + 1 for i in range(loop + 1, len(lines)) if "if step and nest_look is not None:" in lines[i])
    finish = next(i + 1 for i in range(loop + 1, len(lines)) if lines[i].strip() == "deferred_checks.finish()")
    return before, finish


# The baton (MR_BATON=1, default): at most one rank thread runs at a time. A rank thread takes it at its first traced
# event and gives it up only inside a blocking wait (every threading.Condition wait: the transport's rounds, ThreadComm,
# Barrier, Event, queue, Future), taking it back when it wakes. So between two waits one rank's host work and CUDA
# commands are contiguous, the way a one-rank recording is. THE BREAKAGE THIS PREVENTS: with free rank threads the
# sealer's host-call audit found other-thread commands inside 13 library host calls ("another thread issued a command
# during the host call") and the module-global audit failed (a two-rank recording).
BATON = os.environ.get("MR_BATON", "1") == "1"
held = threading.local()
_orig_wait = threading.Condition.wait
_orig_notify = threading.Condition.notify


import weakref  # noqa: E402
# Only the waits by which rank threads wait for EACH OTHER give the baton up: the same-host transport's round condition,
# samehost_v842 _Group's barrier (ThreadComm) and this module's boundary barrier. Any other wait (Thread.start, a helper
# pool's futures) keeps it: THE BREAKAGE THIS PREVENTS: rank 0 gave the baton up inside Thread.start while holding
# concurrent.futures' _global_shutdown_lock and rank 1 then blocked on that lock holding the baton (07:50Z).
RANK_CONDS = weakref.WeakSet()

# multirank-2: a DETERMINISTIC baton. The 2026-11-07 baton was a plain Lock, so when the holder gave it up the next
# rank was whichever thread the OS woke first. THE BREAKAGE THIS PREVENTS: recordings A and B of the same argv were not
# command-identical after the TEMPO fix (step 1: 13,002 vs 12,985 commands; stage 2 left 7,374 single-recording frame
# words), because rank work could interleave differently and the shared CuPy pool then handed out different blocks.
# Rule: the baton goes to the LOWEST rank that wants it, and only once no rank is in transit (a rank whose task was
# submitted but has not started, or a rank notified out of a rank wait that has not asked yet). Who is in transit is
# set by the holder's own actions (submit, notify), not by timing, so the grant order is a function of the program.
SCHED = threading.Condition(threading.Lock())
ST = {"holder": None, "wanting": set(), "waking": set(), "pending": {}, "stall": None, "grants": 0, "forced": 0}
BLOCKED = {}   # id(rank condition) -> ranks blocked in it, in entry order
GRACE = float(os.environ.get("MR_BATON_GRACE", "20"))


def _expected():
    return ({r for r, n in ST["pending"].items() if n > 0} | ST["waking"]) - ST["wanting"] - {ST["holder"]}


def take(r, from_pending=False, where=""):
    with SCHED:
        if from_pending:
            ST["pending"][r] = ST["pending"].get(r, 0) - 1
        ST["waking"].discard(r)
        ST["wanting"].add(r)
        SCHED.notify_all()
        while True:
            if ST["holder"] is None and r == min(ST["wanting"]):
                exp = _expected()
                if not exp:
                    break
                if ST["stall"] is None:
                    ST["stall"] = time.monotonic()
                elif time.monotonic() - ST["stall"] > GRACE:
                    # Never silent: a forced grant is a possible ordering difference between two recordings.
                    ST["forced"] += 1
                    with open(f"{LOGS}/baton-forced-{os.getpid()}.txt", "a") as f:
                        f.write(f"{time.time():.3f} grant {ST['grants']} to rank {r} at {where} with ranks {sorted(exp)} "
                                f"still in transit (waking {sorted(ST['waking'])}, pending {ST['pending']})\n")
                    break
            _orig_wait(SCHED, 0.05)
        ST["wanting"].discard(r)
        ST["holder"] = r
        ST["stall"] = None
        ST["grants"] += 1
        if os.environ.get("MR_BATON_TRACE"):
            with open(f"{LOGS}/baton-{os.getpid()}.txt", "a") as f:
                f.write(f"{ST['grants']} {r} {where}\n")
    held.v = True


def give(r):
    held.v = False
    with SCHED:
        if ST["holder"] == r:
            ST["holder"] = None
        ST["stall"] = None
        SCHED.notify_all()


def _baton_wait(self, timeout=None):
    if not getattr(held, "v", False) or self not in RANK_CONDS:
        return _orig_wait(self, timeout)
    r = rank_of()
    with SCHED:   # the caller holds self's lock, so this entry order is the order of self._waiters
        BLOCKED.setdefault(id(self), []).append(r)
    give(r)
    try:
        return _orig_wait(self, timeout)
    finally:
        saved = self._release_save()     # never hold the condition's lock while waiting for the baton
        with SCHED:
            lst = BLOCKED.get(id(self), [])
            if r in lst:                 # woken by its timeout, not by a notify
                lst.remove(r)
        take(r, where="wait")
        self._acquire_restore(saved)


def _baton_notify(self, n=1):
    if self in RANK_CONDS:
        with SCHED:
            lst = BLOCKED.get(id(self))
            if lst:
                k = len(lst) if n >= len(self._waiters) else min(n, len(lst))
                ST["waking"].update(lst[:k])
                del lst[:k]
    return _orig_notify(self, n)


if BATON:
    threading.Condition.wait = _baton_wait
    threading.Condition.notify = _baton_notify     # notify_all is notify(len(self._waiters)) in CPython 3.14
    import concurrent.futures.thread as _cft
    _orig_submit = _cft.ThreadPoolExecutor.submit
    _orig_run = _cft._WorkItem.run

    def _submit(self, fn, /, *args, **kwargs):
        m = re.match(r"hex-rank-(\d+)$", getattr(self, "_thread_name_prefix", "") or "")
        if m:
            with SCHED:
                r = int(m.group(1))
                ST["pending"][r] = ST["pending"].get(r, 0) + 1
        return _orig_submit(self, fn, *args, **kwargs)

    def _run(self, ctx):
        r = rank_of()
        if r is None:
            return _orig_run(self, ctx)
        take(r, from_pending=True, where="task")
        try:
            return _orig_run(self, ctx)
        finally:
            self.task = None   # drop the task's references (and any CuPy blocks) while this rank still holds the baton
            give(r)
    _cft.ThreadPoolExecutor.submit = _submit
    _cft._WorkItem.run = _run


def ranked(trace):
    """Wrap the observer's global trace: at the two boundary lines of the step loop, rank threads take turns."""
    def outer(frame, event, arg):
        r = rank_of()
        local = trace(frame, event, arg)
        if (local is None or r is None or frame.f_code.co_name != "_run_native_dynamics"
                or not frame.f_code.co_filename.endswith("/hexcore/mpasphys/runtime_v842.py")):
            return local
        before, finish = boundary_lines(frame.f_code)
        state = {"local": local}

        def mine(current, kind, value):
            at = kind == "line" and current.f_lineno in (before, finish) and current.f_locals.get("step", 0)
            if not at:
                try:
                    res = state["local"](current, kind, value)
                except BaseException:
                    import traceback
                    with open(f"{LOGS}/hook-{os.getpid()}.txt", "a") as f:
                        f.write(f"rank {r} line {current.f_lineno} kind {kind} (not a boundary line)\n")
                        traceback.print_exc(file=f)
                    raise
                if res is not None:
                    state["local"] = res
                return mine
            with open(f"{LOGS}/barrier-{os.getpid()}.txt", "a") as f:
                f.write(f"{time.time():.3f} rank {r} line {current.f_lineno} step {current.f_locals.get('step')}\n")
            barrier.wait()
            for turn in range(NRANKS):
                if turn == r:
                    if GC_BOUNDARY and turn == 0:
                        gc.collect()
                    try:
                        res = state["local"](current, kind, value)
                    except BaseException:
                        # An exception in a trace function removes the thread's trace, so the rank would run on
                        # without its barriers. Log it, then re-raise.
                        import traceback
                        with open(f"{LOGS}/hook-{os.getpid()}.txt", "a") as f:
                            f.write(f"rank {r} line {current.f_lineno} step {current.f_locals.get('step')}\n")
                            traceback.print_exc(file=f)
                        raise
                    if res is not None:
                        state["local"] = res
                barrier.wait()
            return mine
        return mine
    return outer


_real_settrace = sys.settrace


def settrace(fn):
    if fn is None:
        threading.settrace(None)
        return _real_settrace(None)
    wrapped = ranked(fn)
    threading.settrace(wrapped)
    return _real_settrace(wrapped)


class _SysShim:
    """sys for observe_cupti_outer only: its one sys.settrace(trace) call also arms threading.settrace, wrapped once.
    (Overriding sys.settrace itself re-wrapped the hook at every thread start, because threading bootstraps each
    thread through sys.settrace: rank 1 got one more barrier level than rank 0 and the ranks deadlocked, 07:33Z.)"""
    settrace = staticmethod(settrace)

    def __getattr__(self, name):
        return getattr(sys, name)


# Two cards at 3 km (2026-11-13): the observer this wraps is the record_observer.py of the tree this file sits in
# (p1/tools, two folders up), or MR_TOOLS. THE BREAKAGE THIS PREVENTS: the path was one development checkout's, which no install has,
# and the earlier observer lacks the TEMPO scheduling words today's limited-area seal reads.
tools = os.environ.get("MR_TOOLS") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
obs_dir = sys.argv[sys.argv.index("--observer-dir") + 1]
sys.path.insert(0, obs_dir)
import observe_cupti_outer  # noqa: E402  (the module whose boundary_trace fetches binder_mark_step)
observe_cupti_outer.ctypes = _CtypesShim()
observe_cupti_outer.sys = _SysShim()
RANK_CONDS.add(barrier._cond)
from woof.hex import partition_samehost_transport_v841 as _transport  # noqa: E402
from woof.hex.mpasphys import samehost_v842 as _samehost  # noqa: E402
_t_init, _g_init = _transport.SameHostTransport.__init__, _samehost._Group.__init__


# MR_NULL_STREAMS=1 (default): the rank streams and the transport's copy streams are CuPy's legacy null stream, as every
# command of a one-rank recording is. Each rank's commands still go in the forward's own order; only the overlap between
# ranks' streams goes. THE BREAKAGE THIS PREVENTS: the sealer's library audit admits cub/thrust host calls only on stream
# 0 ("unsupported stream or dtype" x64 on the rank streams), and its replacement proposals and cuBLAS audit assume it too.
NULL_STREAMS = os.environ.get("MR_NULL_STREAMS", "1") == "1"
if NULL_STREAMS:
    os.environ["WOOF_HEX_SAMEHOST_SETUP_STREAM"] = "default"
    from woof.hex import partition_samehost_v841 as _ranks  # noqa: E402
    _r_init = _ranks.SameHostRanks.__init__

    def _r_null(self, *a, **k):
        _r_init(self, *a, **k)
        if self.cp is not None:
            self.streams = [self.cp.cuda.Stream.null for _ in self.streams]
    _ranks.SameHostRanks.__init__ = _r_null
    # TEMPO keeps one cost/order/hist and one warp scratch per (device, stream) so a rank evicts only its own entries
    # (tempo_v303._order_work / _warp_work). On the shared null stream that key is the same for every rank, and each
    # rank's call would evict the other's (the 2026-10-28 breakage those functions name). Key them by rank instead.
    # THE BREAKAGE THIS PREVENTS: a two-rank replay missed allocation 531 at step 1 (rank 0's TEMPO cost
    # words re-used in a pool block that the boundary capture, holding only rank 1's entry, could not mask).
    from woof.hex.mpasphys.rrfs import tempo_v303 as _tempo  # noqa: E402

    def _order_work(device, cells):
        import cupy as cp
        key = (device, ("rank", rank_of()), cells)
        with _tempo._WARP_WORK_LOCK:
            if key not in _tempo._ORDER_WORK:
                for old in [k for k in _tempo._ORDER_WORK if k[:2] == key[:2]]:
                    del _tempo._ORDER_WORK[old]
                _tempo._ORDER_WORK[key] = (cp.zeros(cells, dtype=cp.uint32), cp.empty(cells, dtype=cp.int32),
                                           cp.empty(128, dtype=cp.uint32))
            return _tempo._ORDER_WORK[key]

    def _warp_work(device, nz, warps, sizes):
        import cupy as cp
        key = (device, ("rank", rank_of()), nz, warps, sizes)
        with _tempo._WARP_WORK_LOCK:
            if key not in _tempo._WARP_WORK:
                for old in [k for k in _tempo._WARP_WORK if k[:2] == key[:2]]:
                    del _tempo._WARP_WORK[old]
                work_bytes, state_bytes, _ = sizes
                _tempo._WARP_WORK[key] = (cp.empty(warps * state_bytes, dtype=cp.uint8),
                                          cp.empty(warps * work_bytes, dtype=cp.uint8),
                                          cp.empty(warps * nz * 2, dtype=cp.float32))
            return _tempo._WARP_WORK[key]
    _tempo._order_work, _tempo._warp_work = _order_work, _warp_work


if BATON:
    from woof.hex import partition_samehost_v841 as _ranks_b  # noqa: E402
    _w_init = _ranks_b.SameHostRanks._initialize_worker

    def _init_worker(self, rank):
        # The rank thread's initializer (device bind) runs before its first task: under the baton too.
        take(rank, where="init")
        try:
            return _w_init(self, rank)
        finally:
            give(rank)
    _ranks_b.SameHostRanks._initialize_worker = _init_worker

# multirank-2: retire each halo round on the rank thread that completes it, not on the transport's completion worker.
# THE BREAKAGE THIS PREVENTS: the worker (hex-halo-completion) dropped each round's field references whenever its
# event wait returned, so blocks went back to the shared CuPy pool at a time set by the GPU, a rank's next allocation
# could get a different block, and _prepare_packet re-uploaded (or not) a descriptor table: commands that differ
# between two recordings of the same argv. Inline, after the round's receive events (all recorded by then), the drop
# happens at the same point of the program every time. MR_INLINE_RETIRE=0 keeps the worker.
INLINE_RETIRE = os.environ.get("MR_INLINE_RETIRE", "1") == "1"
if INLINE_RETIRE:
    _t_queue = _transport.SameHostTransport._queue_completion

    def _queue_inline(self, round_):
        if self._cpu:
            return _t_queue(self, round_)
        if self.measure_gpu:
            raise RuntimeError("mr2 observer: inline halo retirement does not support exchange timing (measure_gpu)")
        round_.completion = tuple((rank, self.devices[rank], round_.submissions[rank].received) for rank in range(self.n_ranks))
        for _rank, device, event in round_.completion:
            self.xp.cuda.Device(device).use()
            event.synchronize()
        self.stats["completed_rounds"] += 1
        if self.release_fields_on_retire:
            self._drop_round_fields(round_)
        self._condition.notify_all()
    _transport.SameHostTransport._queue_completion = _queue_inline
    _transport.SameHostTransport._start_completion_workers = lambda self: None


def _t_registered(self, *a, **k):
    _t_init(self, *a, **k)
    RANK_CONDS.add(self._condition)
    if NULL_STREAMS and not getattr(self, "_cpu", True):
        null = self.xp.cuda.Stream.null
        self._copy_streams = [null for _ in self._copy_streams]
        self._staging_streams = {d: null for d in self._staging_streams}


def _g_registered(self, *a, **k):
    _g_init(self, *a, **k)
    RANK_CONDS.add(self.barrier._cond)


_transport.SameHostTransport.__init__ = _t_registered

# multirank-2: the decomposed forward has two upload caches keyed by device ADDRESSES: the halo transport re-uploads a
# packet's descriptor table only when a field's pointer differs from its previous exchange (_prepare_packet), and the
# solve-region validator re-uploads its table only on a new (address, stride, ...) key (SolveRegionKernels.table). Which
# block CuPy's pool hands a temporary is not the same in two processes, so the NUMBER of uploads followed it.
# THE BREAKAGE THIS PREVENTS: recordings g2a/g2b (identical baton grants, cyclic GC only at boundaries) still differed:
# step 1, a 72-byte halo descriptor upload before halo_copy_bytes in g2b only; steps 1-5 all part the same way.
# In the recording every exchange uploads its table and every validation uploads its table: same bytes the forward
# would read either way, and a command stream that no longer depends on where the pool put an array.
# MR_ADDRESS_CACHES=keep restores the forward's caches.
if os.environ.get("MR_ADDRESS_CACHES", "bypass") == "bypass":
    _t_prepare = _transport.SameHostTransport._prepare_packet

    def _prepare_always(self, fields, packet, *, upload=True):
        if packet.pointers:             # a pointer that matches nothing: the table is rebuilt (dims kept while strides/shapes match)
            packet.pointers = tuple((-1,) + tuple(e[1:]) for e in packet.pointers)
        packet.descriptor = None
        return _t_prepare(self, fields, packet, upload=upload)
    _transport.SameHostTransport._prepare_packet = _prepare_always
    # The fused pack/unpack keeps a third address-keyed cache: _Batch.bindings, keyed by the field pointers, holding
    # the concatenated descriptor table (an LRU of 4,096). THE BREAKAGE THIS PREVENTS: g3a/g3b (the two caches above
    # bypassed) still parted at step 1: a 72-byte fused-table upload before halo_copy_bytes in g3b only.
    _t_copy = _transport.SameHostTransport._copy_packets

    def _copy_always(self, fields, packets, batch, *, scatter, stream=None):
        batch.bindings.clear()
        return _t_copy(self, fields, packets, batch, scatter=scatter, stream=stream)
    _transport.SameHostTransport._copy_packets = _copy_always
    from woof.hex import cuda_solve_region_v841 as _sr  # noqa: E402
    _sr_table = _sr.SolveRegionKernels.table

    def _table_always(self, rows):
        self._tables.clear()
        return _sr_table(self, rows)
    _sr.SolveRegionKernels.table = _table_always
_samehost._Group.__init__ = _g_registered
# multirank-2: every forward reading of free device memory returns one fixed size (MR_FIXED_FREE_MIB, default 16384;
# 0 = the card's own), as S01's P1_FIXED_FREE_MIB (ad/s01-convective-3km b7dec10b) does for one-rank recordings.
# THE BREAKAGE THIS PREVENTS: on a shared card the two recordings of one plan read different free memory and take
# different memory-sized batching (the LW wrapper chunk, the MYNN-EDMF arena budget free/4): their launch streams then
# no longer align and the sealer cannot pair them. Only batching of independent columns changes; total stays the card's.
_FIXED_FREE = int(os.environ.get("MR_FIXED_FREE_MIB", "16384"))
if _FIXED_FREE > 0:
    import cupy as _cp  # noqa: E402
    _real_mem = _cp.cuda.runtime.memGetInfo
    _cp.cuda.runtime.memGetInfo = lambda: (_FIXED_FREE << 20, _real_mem()[1])
# The multi-device runner: MR_RANK_POOLS=1 (default) gives every rank thread its own CuPy MemoryPool (a
# thread-local allocator set in the rank worker's initializer, which runs on that thread before its first task). THE
# BREAKAGE THIS PREVENTS: with one shared pool a pool block (a sealed plan allocation) holds arrays of both ranks at
# different offsets (a two-rank TEMPO recording: kernels tie 290 of 483 allocations into one component), so a plan cannot be split by
# allocation into one process per card. With a pool per rank every block a rank thread allocates is that rank's; the
# multi-device splitter (tools/multidevice/split_plan.py) proves it from the sealed plan before it splits.
if os.environ.get("MR_RANK_POOLS", "1") == "1":
    import cupy as _cpp  # noqa: E402
    from woof.hex import partition_samehost_v841 as _ranks_p  # noqa: E402
    _RANK_POOLS = {}
    _p_init = _ranks_p.SameHostRanks._initialize_worker

    def _init_rank_pool(self, rank):
        out = _p_init(self, rank)
        pool = _RANK_POOLS.setdefault(int(rank), _cpp.cuda.MemoryPool())
        _cpp.cuda.memory._set_thread_local_allocator(pool.malloc)
        with open(f"{LOGS}/rank-pools-{os.getpid()}.txt", "a") as f:
            f.write(f"{time.time():.3f} rank {rank} thread {threading.current_thread().name} pool {id(pool):#x}\n")
        return out
    _ranks_p.SameHostRanks._initialize_worker = _init_rank_pool
sys.argv = [os.path.join(tools, "record_observer.py")] + sys.argv[1:]
sys.path.insert(0, tools)
runpy.run_path(sys.argv[0], run_name="__main__")
