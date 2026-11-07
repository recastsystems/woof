"""Settle, on native Linux, the questions WSL2 cannot answer.

Everything the out-of-core tiling work has measured so far came from a WSL2 box,
where the GPU is reached through the Windows driver.  Three of those numbers are
suspected WDDM artefacts rather than properties of the hardware, and all three
change how big a domain a user can run:

  1. PINNED CEILING.  WSL2 walled page-locked memory at exactly MemTotal/2
     (46.9375 GiB of 93.9), and managed memory hit the SAME number -- the
     signature of WDDM's shared-GPU-memory budget, which native Linux does not
     have.  If Linux allows more, every domain figure grows with it.
  2. MANAGED OVERSUBSCRIPTION.  concurrentManagedAccess reads 0 under WSL2 and
     native Windows alike, so cudaMemPrefetchAsync/cudaMemAdvise fail outright
     and managed memory never migrates to the device.  On Linux the flag should
     read 1, which would make "allocate past VRAM and let the driver page"
     a real fallback rather than a dead end.
  3. FULL DUPLEX.  asyncEngineCount == 1 on a GeForce 5090, so simultaneous
     upload and download share ONE budget.  Worth confirming per card, since
     professional parts report more and would get true duplex for free.

Run it on any NVIDIA GPU.  Host RAM matters more than the card here -- questions
1 and 2 are about host memory, not VRAM.

    pip install cupy-cuda12x        # or cupy-cuda11x to match the driver
    python linux_probe.py

READ THIS BEFORE TRUSTING THE PINNED NUMBER.  Containers cap locked memory:
Docker's default RLIMIT_MEMLOCK is 64 KiB, and a rented "GPU instance" is very
often a container.  A low ceiling measured inside one says nothing about Linux.
The report prints the limit and the container evidence first, and refuses to
draw a conclusion when the environment cannot support one.  Getting this wrong
would be the fifth false result in this project; run with
`--ulimit memlock=-1` (Docker) or on bare metal.

Safety: allocations bisect upward and stop at a fraction of MemAvailable, and
every block is released before the next probe.  Page-locked memory cannot be
swapped, so an over-large allocation makes a machine unresponsive rather than
merely slow -- the reason for the conservative default.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time

GiB = 1 << 30
MiB = 1 << 20


# ----------------------------------------------------------------- environment
def read_meminfo():
    info = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            key, _, rest = line.partition(":")
            info[key] = int(rest.split()[0]) * 1024  # kB -> bytes
    return info


def container_evidence():
    """Several independent signals, because any one of them can be absent."""
    if sys.platform != "linux":
        raise RuntimeError("the Linux memory probe requires Linux /proc and resource limits")
    import resource

    signals = {}
    signals["dockerenv"] = os.path.exists("/.dockerenv")
    try:
        with open("/proc/1/cgroup") as fh:
            cg = fh.read()
        signals["cgroup_mentions_docker_or_k8s"] = any(
            t in cg for t in ("docker", "kubepods", "containerd", "lxc")
        )
    except OSError:
        signals["cgroup_mentions_docker_or_k8s"] = None
    try:
        with open("/proc/1/sched") as fh:
            first = fh.readline()
        # PID 1 being something other than init/systemd suggests a container.
        signals["pid1"] = first.split()[0]
    except OSError:
        signals["pid1"] = None
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    signals["memlock_soft"] = "unlimited" if soft == resource.RLIM_INFINITY else soft
    signals["memlock_hard"] = "unlimited" if hard == resource.RLIM_INFINITY else hard
    signals["memlock_soft_bytes"] = None if soft == resource.RLIM_INFINITY else soft
    return signals


def memlock_verdict(sig, mem_total):
    """Can a pinned-ceiling measurement here mean anything?"""
    soft = sig["memlock_soft_bytes"]
    if soft is None:
        return "OK", "RLIMIT_MEMLOCK is unlimited; a ceiling measured here is real."
    if soft < 1 * GiB:
        return (
            "INVALID",
            f"RLIMIT_MEMLOCK is only {soft / MiB:.1f} MiB. This is a container "
            "default, not a Linux limit. Re-run with `--ulimit memlock=-1` "
            "(Docker) or on bare metal; any ceiling measured here describes the "
            "container and must NOT be compared against the WSL2 number.",
        )
    if soft < 0.9 * mem_total:
        return (
            "CAPPED",
            f"RLIMIT_MEMLOCK is {soft / GiB:.1f} GiB against {mem_total / GiB:.1f} "
            "GiB of RAM, so the probe may hit the rlimit rather than the OS "
            "policy. Treat the result as a lower bound.",
        )
    return "OK", "RLIMIT_MEMLOCK is at or above physical RAM."


def nvidia_smi(query):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        return out.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"(nvidia-smi unavailable: {exc})"


# ----------------------------------------------------------------- cuda helpers
def load_cupy():
    try:
        import cupy  # noqa: PLC0415
        return cupy
    except Exception as exc:  # noqa: BLE001
        print("FATAL: could not import cupy.", file=sys.stderr)
        print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
        print("  Install the wheel matching your driver, e.g.:", file=sys.stderr)
        print("    pip install cupy-cuda12x", file=sys.stderr)
        raise SystemExit(2)


def device_report(cp):
    p = cp.cuda.runtime.getDeviceProperties(0)
    free, total = cp.cuda.runtime.memGetInfo()
    name = p["name"]
    return {
        "name": name.decode() if isinstance(name, bytes) else str(name),
        "vram_total_gib": total / GiB,
        "vram_free_gib": free / GiB,
        "l2_bytes": p.get("l2CacheSize"),
        # THE THREE FLAGS THAT DIFFER BETWEEN WDDM AND LINUX:
        "asyncEngineCount": p.get("asyncEngineCount"),
        "concurrentManagedAccess": p.get("concurrentManagedAccess"),
        "pageableMemoryAccess": p.get("pageableMemoryAccess"),
        "managedMemory": p.get("managedMemory"),
        "unifiedAddressing": p.get("unifiedAddressing"),
        "pcie_gen": nvidia_smi("pcie.link.gen.current"),
        "pcie_width": nvidia_smi("pcie.link.width.current"),
    }


# ------------------------------------------------------------- 1. pinned ceiling
def pinned_ceiling(cp, mem_total, mem_available, fraction, chunk_gib):
    """Grow a pinned pool in fixed chunks until it refuses; report the total.

    Fixed chunks are used deliberately: on WSL2 the apparent ceiling moved with
    block size, but that turned out to be CuPy's pinned POOL rounding requests
    up to the next power of two -- the measuring instrument, not the driver.
    This goes through cudaHostAlloc directly for exactly that reason.
    """
    if cp.cuda.runtime.is_hip:
        from woof.hip_runtime import library
        runtime = library()
        allocate, release = runtime.hipHostMalloc, runtime.hipHostFree
    else:
        runtime = ctypes.CDLL("libcudart.so")
        allocate, release = runtime.cudaHostAlloc, runtime.cudaFreeHost
    allocate.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    allocate.restype = ctypes.c_int
    release.argtypes = [ctypes.c_void_p]
    release.restype = ctypes.c_int

    budget = int(mem_available * fraction)
    chunk = int(chunk_gib * GiB)
    blocks, total, err = [], 0, None
    try:
        while total + chunk <= budget:
            ptr = ctypes.c_void_p()
            rc = allocate(ctypes.byref(ptr), chunk, 0)
            if rc != 0:
                err = f"host allocation returned {rc} at {total / GiB:.2f} GiB"
                break
            blocks.append(ptr)
            total += chunk
            # Re-read availability: pinned pages are unreclaimable, so if the
            # machine is being squeezed we stop before it becomes unusable.
            avail = read_meminfo()["MemAvailable"]
            if avail < 8 * GiB:
                err = f"stopping early: MemAvailable fell to {avail / GiB:.1f} GiB"
                break
        else:
            err = (f"reached the self-imposed budget of {budget / GiB:.2f} GiB "
                   f"({fraction:.0%} of MemAvailable) without a refusal")
    finally:
        for ptr in blocks:
            release(ptr)

    return {
        "pinned_total_gib": total / GiB,
        "as_fraction_of_memtotal": total / mem_total if mem_total else None,
        "budget_gib": budget / GiB,
        "chunk_gib": chunk_gib,
        "stopped_because": err,
        "hit_self_imposed_budget": "budget" in (err or ""),
    }


# --------------------------------------------------------------- 2. bandwidth
def bandwidth(cp, size_mb=512, reps=7):
    n = size_mb * MiB // 4
    dev = cp.empty(n, dtype=cp.float32)
    pinned_mem = cp.cuda.alloc_pinned_memory(n * 4)
    import numpy as np  # noqa: PLC0415
    host = np.frombuffer(pinned_mem, dtype=np.float32, count=n)
    host[:] = 1.0

    def timed(fn):
        out = []
        for i in range(reps + 2):
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            fn()
            cp.cuda.Stream.null.synchronize()
            if i >= 2:
                out.append(time.perf_counter() - t0)
        return statistics.median(out)

    h2d = timed(lambda: dev.set(host))
    d2h = timed(lambda: dev.get(out=host))
    nb = n * 4

    # Duplex: the answer decides whether a streaming pipeline gets one shared
    # budget or two independent ones.
    up, down = cp.empty(n, dtype=cp.float32), cp.empty(n, dtype=cp.float32)
    s1 = cp.cuda.Stream(non_blocking=True)
    s2 = cp.cuda.Stream(non_blocking=True)

    def both():
        with s1:
            up.set(host, stream=s1)
        with s2:
            down.get(out=host, stream=s2)
        s1.synchronize()
        s2.synchronize()

    dup = timed(both)
    dev = up = down = None  # rebound, not deleted: a closure above reads it
    cp.get_default_memory_pool().free_all_blocks()

    best_simplex = max(nb / h2d, nb / d2h)
    aggregate = 2 * nb / dup
    return {
        "h2d_gbps": nb / h2d / 1e9,
        "d2h_gbps": nb / d2h / 1e9,
        "duplex_aggregate_gbps": aggregate / 1e9,
        "duplex_speedup_over_simplex": aggregate / best_simplex,
        "full_duplex_works": aggregate / best_simplex > 1.5,
    }


# --------------------------------------------- 3. managed memory (the Linux path)
def managed_probe(cp, vram_total, factor=1.5):
    """Can we allocate past VRAM and have the driver page it sensibly?

    Under WSL2 this was a dead end: managed memory never became device-resident
    (16.5x slower than an ordinary array that fit trivially), and both
    cudaMemPrefetchAsync and cudaMemAdvise failed with cudaErrorInvalidDevice
    because concurrentManagedAccess == 0.  On Linux the flag should be 1 and the
    prefetch path should exist, which would make this a genuine zero-effort
    fallback for users who never want to think about tiling.
    """
    out = {"target_bytes": int(vram_total * factor)}
    try:
        n = out["target_bytes"] // 4
        mem = cp.cuda.malloc_managed(n * 4)
        arr = cp.ndarray((n,), dtype=cp.float32, memptr=mem)
        out["allocated"] = True
        out["oversubscribed"] = n * 4 > vram_total

        try:
            cp.cuda.runtime.memPrefetchAsync(mem.ptr, min(n * 4, 1 << 30), 0, 0)
            cp.cuda.Stream.null.synchronize()
            out["prefetch_works"] = True
        except Exception as exc:  # noqa: BLE001
            out["prefetch_works"] = False
            out["prefetch_error"] = f"{type(exc).__name__}: {exc}"

        # Sweep it once so page migration actually happens, then time a second
        # pass -- the first pass pays the fault cost and is not the number we
        # want to characterise steady state with.
        arr.fill(1.0)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        total = float(arr.sum())
        cp.cuda.Stream.null.synchronize()
        dt = time.perf_counter() - t0
        out["sum_check"] = total
        out["managed_read_gbps"] = (n * 4) / dt / 1e9

        ref = cp.ones(min(n, (1 << 30) // 4), dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        float(ref.sum())
        cp.cuda.Stream.null.synchronize()
        out["ordinary_read_gbps"] = ref.nbytes / (time.perf_counter() - t0) / 1e9
        out["managed_penalty_x"] = out["ordinary_read_gbps"] / out["managed_read_gbps"]
        del arr, mem, ref
    except Exception as exc:  # noqa: BLE001
        out["allocated"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
    cp.get_default_memory_pool().free_all_blocks()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.80,
                    help="ceiling probe stops at this fraction of MemAvailable")
    ap.add_argument("--chunk-gib", type=float, default=1.0)
    ap.add_argument("--skip-managed", action="store_true")
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()
    if sys.platform != "linux":
        ap.error("this memory probe requires Linux; use woof doctor for host readiness")

    mem = read_meminfo()
    sig = container_evidence()
    verdict, note = memlock_verdict(sig, mem["MemTotal"])

    print("=" * 68)
    print("ENVIRONMENT")
    print("=" * 68)
    print(f"  MemTotal      {mem['MemTotal'] / GiB:8.2f} GiB")
    print(f"  MemAvailable  {mem['MemAvailable'] / GiB:8.2f} GiB")
    print(f"  kernel        {os.uname().release}")
    print(f"  memlock soft  {sig['memlock_soft']}   hard {sig['memlock_hard']}")
    print(f"  container?    dockerenv={sig['dockerenv']} "
          f"cgroup={sig['cgroup_mentions_docker_or_k8s']} pid1={sig['pid1']}")
    print(f"  MEMLOCK VERDICT: {verdict} -- {note}")
    if "microsoft" in os.uname().release.lower():
        print("  !! This kernel looks like WSL2. The whole point of this probe "
              "is to run somewhere else.")

    cp = load_cupy()
    dev = device_report(cp)
    print()
    print("=" * 68)
    print("DEVICE")
    print("=" * 68)
    for k, v in dev.items():
        print(f"  {k:26s} {v}")
    print()
    print("  Reference, measured on the WSL2 RTX 5090 this work came from:")
    print("    asyncEngineCount 1 | concurrentManagedAccess 0 | pinned wall MemTotal/2")

    print()
    print("=" * 68)
    print(f"1. PINNED CEILING (stops at {args.fraction:.0%} of MemAvailable)")
    print("=" * 68)
    ceil = pinned_ceiling(cp, mem["MemTotal"], mem["MemAvailable"],
                          args.fraction, args.chunk_gib)
    print(f"  pinned         {ceil['pinned_total_gib']:8.2f} GiB")
    print(f"  of MemTotal    {ceil['as_fraction_of_memtotal']:8.1%}")
    print(f"  stopped        {ceil['stopped_because']}")
    if verdict == "INVALID":
        print("  >> MEANINGLESS: see the MEMLOCK VERDICT above.")
    elif ceil["hit_self_imposed_budget"]:
        print(f"  >> No OS refusal at {args.fraction:.0%} of RAM -- so the real "
              "ceiling is HIGHER than this. WSL2 refused at 50%.")
    elif ceil["as_fraction_of_memtotal"] > 0.60:
        print("  >> ABOVE the WSL2 MemTotal/2 wall. Domains scale with this.")
    else:
        print("  >> At or below the WSL2 wall; not a Windows artefact after all.")

    print()
    print("=" * 68)
    print("2. PCIe BANDWIDTH AND DUPLEX")
    print("=" * 68)
    bw = bandwidth(cp)
    for k, v in bw.items():
        print(f"  {k:32s} {v}")
    print("  WSL2 5090 reference: 57.1 H2D / 56.3 D2H, duplex speedup 1.00 (none)")

    managed = None
    if not args.skip_managed:
        print()
        print("=" * 68)
        print("3. MANAGED MEMORY OVERSUBSCRIPTION")
        print("=" * 68)
        managed = managed_probe(cp, int(dev["vram_total_gib"] * GiB))
        for k, v in managed.items():
            print(f"  {k:32s} {v}")
        print("  WSL2 5090 reference: allocates but never migrates, 16.5x slower,")
        print("  prefetch/advise both fail with cudaErrorInvalidDevice.")

    report = {"env": {**{k: mem[k] for k in ("MemTotal", "MemAvailable")},
                      "kernel": os.uname().release, "memlock": sig,
                      "memlock_verdict": verdict},
              "device": dev, "pinned_ceiling": ceil, "bandwidth": bw,
              "managed": managed}
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\nwrote {args.json}")
    else:
        print("\n--- JSON ---")
        print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
