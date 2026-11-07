"""Give a queued card back the moment a composing door's GPU work ends.

THE BREAKAGE THIS PREVENTS (reference system overflow box, 2026-10-27 21:05-21:28Z): zoom-globe arms
(``woof hex mesh-plan --global-mesh --generate --run``) held exclusive cards 6 and 7 at 0 % GPU through their
products stage (the Rust renderers and the ``rw_soundings --columns`` writer: 188 s of a 652 s lease on the
3 h 96 -> 3 km arm) while other jobs queued for those cards. The card queue (``run.sh``) could give a card back
early only once the leased command had EXITED and every process left was CPU-only; a composing door draws its
products as its own last stage, so it exited only after the maps were drawn. The plain ``forecast`` door
already detaches its products (``forecast_door.start_detached_products``); the composing doors had no way to
say "the card is free now, I am still running".

The contract with the queue (run.sh gpu-done, 2026-10-27): the queue gives the command it leases
``GPU_MUTEX_GPU_DONE`` (a file path) and ``GPU_MUTEX_LEASE_PID`` (its own pid). The leased command process
itself (its parent is the queue; a door composed under another command is not the lease's command and says
nothing) appends its pid to that file once it will not touch the card again, and starts everything after it
with no visible CUDA device. The queue then releases the cards while the command keeps running (once no
process in the tree that could still see the card is undeclared or on the card), still waits for the command,
and returns its exit code. The host-RAM lease stays until the command ends. Outside such a queue this does
nothing.
"""

from __future__ import annotations

import os
from typing import MutableMapping

GPU_DONE = "GPU_MUTEX_GPU_DONE"
LEASE_PID = "GPU_MUTEX_LEASE_PID"
CPU_ONLY = {"CUDA_VISIBLE_DEVICES": "", "GPUWM_NO_LOCAL_GPU": "1"}


def gpu_work_done(env: MutableMapping[str, str] | None = None) -> bool:
    """Declare this process's GPU work over; True when a card queue was told.

    From here on this process and ``env`` (the environment a door passes to its later stages) see no CUDA
    device, so the products (CPU work in Rust binaries) cannot open a context on a card the queue has given to
    another job. Does nothing, and changes no environment, outside a lease or when this process is not the
    lease's own command."""

    path, lease = os.environ.get(GPU_DONE), os.environ.get(LEASE_PID)
    if not path or not lease or str(os.getppid()) != lease:
        return False
    # The environment changes before the queue is told, so nothing this door starts after the declaration
    # can see the card the queue is about to give away.
    os.environ.update(CPU_ONLY)
    if env is not None:
        env.update(CPU_ONLY)
    try:
        with open(path, "a", encoding="ascii") as marker:
            marker.write(f"{os.getpid()}\n")
    except OSError:
        return False
    return True


__all__ = ["CPU_ONLY", "GPU_DONE", "LEASE_PID", "gpu_work_done"]
