"""CPU-only NVRTC replay in an isolated process, never a CUDA context.

The explicit target selector is the same one the AOT builder uses. All
compiler options, including the engine's final numerical options, come from
the parent request. Only compilation runs here; the parent loads its image.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from functools import lru_cache

#: prctl option: the signal this process receives when the thread that created it ends.
_PR_SET_PDEATHSIG = 1
#: Exit status of a worker whose pool's process is gone (nobody reads it).
PARENT_GONE_EXIT = 1
#: How often the fallback watch (no pidfd, not Windows) asks whether the parent is still the parent.
PARENT_POLL_SECONDS = 1.0


def bind_to_parent(parent: int) -> None:
    """Pool initializer: this compile worker ends when the process that opened its pool ends.

    THE BREAKAGE (2026-10-29 08:17-08:21Z): 89 spawn compile workers of hex and woof
    kernel plans outlived the forecasts that opened them and held 62.5 GB (one worker 21.5 GB);
    the box's memory cgroup filled and the OOM killer took a release pass's containers and
    lanes.  A forecast killed by the OOM killer, ended by SIGTERM's default action or by
    ``os._exit`` runs no exit hook, and a worker then waits for its next call forever (it holds
    a write end of its own call queue, so that read never ends), keeping whatever its last
    compile grew to.

    Two layers, either one alone ends the orphan:

    * PR_SET_PDEATHSIG(SIGKILL), Linux: the kernel kills this worker the moment the thread that
      spawned it ends.  :func:`woof.kernel_plan.compiler_pool` spawns every worker on one
      thread that lives as long as the process, so that is the moment the process ends.
    * a daemon thread that waits for the parent's exit (a pidfd on Linux, the parent's handle
      on Windows, a getppid poll elsewhere) and ends this process.

    SIGTERM is also put back to its default and unblocked: 21 of the 89 had inherited it ignored
    or blocked and outlived SIGTERM, which is what the exit drop's ``terminate()`` sends.
    Ending at once loses nothing: a worker returns its image to the parent and writes nothing
    itself.  Never raises, because a failing initializer marks the whole pool broken.
    """
    parent = int(parent)
    try:
        import signal
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        if hasattr(signal, "pthread_sigmask"):
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
    except (ImportError, AttributeError, OSError, ValueError):
        pass
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            import signal
            ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0)
        except (ImportError, AttributeError, OSError):
            pass
    if os.name == "posix" and os.getppid() != parent:
        os._exit(PARENT_GONE_EXIT)   # the parent ended before the death signal was set
    try:
        threading.Thread(target=_watch_parent, args=(parent,), name="kernel-compile-parent-watch",
                         daemon=True).start()
    except RuntimeError:
        pass


def _watch_parent(parent: int) -> None:
    """End this worker when ``parent`` ends (the second layer of :func:`bind_to_parent`)."""
    try:
        if os.name == "nt":
            import multiprocessing
            process = multiprocessing.parent_process()
            if process is not None:
                process.join()          # waits on the parent's process handle
                os._exit(PARENT_GONE_EXIT)
            return
        if hasattr(os, "pidfd_open"):
            try:
                handle = os.pidfd_open(parent)
            except ProcessLookupError:
                os._exit(PARENT_GONE_EXIT)
            except OSError:
                handle = None           # a kernel or sandbox without pidfd: poll below
            if handle is not None:
                import select
                if os.getppid() == parent:   # the pid still names this worker's parent, not a reused one
                    poller = select.poll()
                    poller.register(handle, select.POLLIN)
                    while not poller.poll():
                        pass
                os._exit(PARENT_GONE_EXIT)
        while os.getppid() == parent:
            time.sleep(PARENT_POLL_SECONDS)
        os._exit(PARENT_GONE_EXIT)
    except Exception:
        return


@lru_cache(maxsize=1)
def _identity():
    from .kernel_aot import compiler_identity
    return compiler_identity()


def compile_request(arguments, arch, identity, *, trace=False):
    from cupy.cuda import compiler
    from cupy_backends.cuda.libs import nvrtc
    from . import numerics

    if _identity() != identity:
        raise ValueError("kernel-plan worker compiler identity changed")
    target = str(arch).removeprefix("sm_")
    if compiler._use_ptx or int(target) not in nvrtc.getSupportedArchs():
        raise ValueError("kernel-plan worker requires the explicit supported cubin target")
    selector = compiler._get_arch_for_options_for_nvrtc

    def explicit(selected=None):
        if str(selected) != target:
            raise ValueError("kernel-plan worker target changed")
        return "-arch=sm_" + target, "cubin"

    begin_ns = time.monotonic_ns() if trace else None
    wall, cpu = time.perf_counter(), time.process_time()
    compiler._get_arch_for_options_for_nvrtc = explicit
    previous_mode = numerics._mode
    # The parent has already applied its numerical boundary. In particular,
    # a builder may compile strict AND fast recipes in one process pool.
    # Never reinterpret those final options using this worker's environment.
    numerics._mode = "strict"
    try:
        result = compiler._compile_using_nvrtc_no_warning(
            arguments["source"], tuple(arguments["options"]), target,
            arguments["filename"], arguments["name_expressions"], None,
            arguments["cache_in_memory"], None, arguments["method"])
    finally:
        numerics._mode = previous_mode
        compiler._get_arch_for_options_for_nvrtc = selector
    timing = {"pid": os.getpid(), "seconds": time.perf_counter() - wall,
              "cpu_seconds": time.process_time() - cpu}
    if trace:
        # Same-host monotonic boundaries around the successful compiler call.
        # These include wrapper overhead and are not low-level NVRTC events.
        timing.update(compile_begin_mono_ns=begin_ns,
                      compile_end_mono_ns=time.monotonic_ns())
    return result, timing
