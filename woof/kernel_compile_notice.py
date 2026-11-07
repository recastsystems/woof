"""One accurate sentence before the first run's invisible kernel compile.

The first GPU forecast on a machine pays on the order of two minutes of
NVRTC compilation -- every physics module's ``cupy.RawModule`` and the
direct-NVRTC radiation translation units compile for the local card
before a single model step runs.  CuPy writes the compiled binaries to
its on-disk kernel cache, so every later run loads them in seconds; but
on the run that pays, nothing said so.  The first field run of the
published wheel watched the forecast phase sit at a stale status for a
hundred seconds and reasonably wondered whether it had hung.

This module is the one place that knows how to ask "is that about to
happen?".  It asks it two ways, because the first way provably missed:

**Cold cache.**  The CuPy kernel cache directory (``CUPY_CACHE_DIR``
when set, else ``~/.cupy/kernel_cache`` -- CuPy's own documented
resolution) with no compiled entries in it means every kernel this run
needs is about to be built from source.

**A cache for a different card.**  CuPy's cache key includes the target
architecture (``cupy/cuda/compiler.py`` mixes ``arch`` into the hash it
names the file after), so entries compiled for one card cannot be
loaded by another -- a card swap recompiles everything against a cache
that is not empty at all.  MEASURED, 2026-08-16: this project's
reference box had 7,164 sm_120 entries when its 5090 was moved out and
a 3080 moved in; the cold-cache test said "warm", no notice was
printed, and 51 seconds of sm_86 compilation ran silently inside the
wall clock of model step 1.  That is the case
:func:`kernel_cache_state` exists to catch.

The architecture is read from the entry itself.  A CuPy cache file is
the 40-character SHA1 of the blob followed by the blob, and for a cubin
that blob is a CUDA ELF whose ``e_flags`` carries the SM version.  WHERE
in ``e_flags`` depends on the ELF layout the compiler wrote, which the
header names in ``EI_OSABI`` and ``EI_ABIVERSION``, and real caches hold
both (:data:`_SM_SHIFT_BY_LAYOUT`).  Reading one layout's byte out of
the other's header was a measured false notice: NVRTC 12.9.86 wrote
every sm_86 cubin with ``e_flags`` 0x00560556, the second byte of that
is 0x05, and every warm run on an RTX 3080 announced a recompile "for
sm_86 (the cache carries sm_5)" while it was already stepping.

An entry that does not decode -- a PTX fallback, a truncated write, a
layout this module has not measured -- counts as **possibly this
card's**, never as evidence against it: announcing a two-minute compile
on the strength of a file we could not read is a false positive that
teaches a reader to ignore the line.

No CUDA import happens in any predicate here; every question is
answered from the filesystem alone, so callers may ask before touching
the device.  The one function that does need the device is
:func:`current_compute_capability`, which is separate, optional, and
returns ``None`` rather than raising when there is no card to ask.
"""

from __future__ import annotations

from woof.backend import compilation_target as _compilation_target

import contextlib
import dataclasses
import os
import struct
import threading
import time
from pathlib import Path

#: Environment variable CuPy consults for its kernel cache location.
CUPY_CACHE_ENV = "CUPY_CACHE_DIR"

#: progress.json status published while the first-run compile is likely
#: under way; ``woof go`` relays it verbatim in its heartbeat lines.
COMPILING_STATUS = "COMPILING_GPU_KERNELS"

#: :attr:`KernelCacheState.reason` -- nothing is cached at all.
COLD_CACHE = "cold_cache"

#: :attr:`KernelCacheState.reason` -- entries are cached, none of them
#: for this card's compute capability.
ARCHITECTURE_MISSING = "architecture_missing"

#: Bytes CuPy writes before the compiled blob: the blob's SHA1, hex.
_HASH_PREFIX_BYTES = 40

#: Offset of ``e_flags`` inside a 64-bit ELF header.
_ELF_FLAGS_OFFSET = 48

#: ``e_ident`` bytes and header fields the decode checks before it
#: trusts ``e_flags``: a 64-bit little-endian ELF for ``EM_CUDA``.
_EI_CLASS = 4
_EI_DATA = 5
_EI_OSABI = 7
_EI_ABIVERSION = 8
_ELFCLASS64 = 2
_ELFDATA2LSB = 1
_ELF_MACHINE_OFFSET = 18
_EM_CUDA = 190

#: Where each CUDA ELF layout keeps the SM in ``e_flags``, keyed by the
#: header's ``(EI_OSABI, EI_ABIVERSION)``: the shift to the SM byte.
#: Every row is measured from real CuPy cache entries, not inferred:
#:
#: * ``(0x33, 7)``: the SM is the LOW byte.  NVRTC 12.9.86 on an RTX
#:   3080 wrote all 159 entries of the cache behind the false notice
#:   this way, ``e_flags`` 0x00560556 (0x56 = 86; the third byte repeats
#:   it, and the second byte, 0x05, is not an SM at all).
#: * ``(0x41, 8)``: the SM is the SECOND byte.  The RTX 5070 Ti node's
#:   1,468 sm_120 entries carry ``e_flags`` 0x06007802, and sm_86 entries
#:   in this layout carry 0x06005604.
#:
#: A header naming any other layout decodes to unknown: guessing a byte
#: is how "sm_5" was announced.
_SM_SHIFT_BY_LAYOUT = {(0x33, 7): 0, (0x41, 8): 8}

#: The smallest prefix that can answer "which architecture is this?".
_PROBE_BYTES = _HASH_PREFIX_BYTES + _ELF_FLAGS_OFFSET + 4


def cupy_kernel_cache_dir() -> Path:
    """Where CuPy keeps compiled kernels, by CuPy's own resolution order."""

    override = os.environ.get(CUPY_CACHE_ENV, "").strip()
    if override:
        return Path(override)
    return Path.home() / ".cupy" / "kernel_cache"


def current_compute_capability() -> str | None:
    """This process's card as CuPy spells it (``"86"``, ``"120"``), or None.

    THE ONLY function in this module that touches the device.  It is
    called by the runners, which have already initialised CUDA by the
    time they ask; every predicate below takes the answer as an
    argument, so a caller that must not initialise CUDA simply does not
    call this.

    ``None`` for every failure -- no CuPy, no driver, no card, a driver
    that refuses the query -- because "which card" being unanswerable
    is a reason to fall back to the cold-cache test, never a reason to
    fail a forecast in its telemetry.
    """

    try:
        import cupy

        return str(_compilation_target(xp=cupy))
    except Exception:  # noqa: BLE001 - telemetry never fails a run
        return None


def _entry_architecture(path: str) -> str | None:
    """The SM version one cache entry was compiled for, or ``None``.

    ``None`` means "could not tell", and every caller treats that as
    "possibly mine".  See the module docstring for the layout.
    """

    try:
        with open(path, "rb") as handle:
            head = handle.read(_PROBE_BYTES)
    except OSError:
        return None
    return cubin_architecture(head[_HASH_PREFIX_BYTES:])


def cubin_architecture(blob: bytes) -> str | None:
    """The SM a CUDA ELF header names, read where its layout keeps it.

    ``None`` for anything that is not a 64-bit little-endian CUDA ELF in
    a layout listed in :data:`_SM_SHIFT_BY_LAYOUT`.  The one decoder for
    a cubin's SM in this tree: the kernel cache census reads entries
    through it, and the receipt tools' SASS fallback names its
    ``nvdisasm -b`` target with it.
    """

    if (len(blob) < _ELF_FLAGS_OFFSET + 4 or blob[:4] != b"\x7fELF"
            or blob[_EI_CLASS] != _ELFCLASS64
            or blob[_EI_DATA] != _ELFDATA2LSB):
        return None
    if struct.unpack_from("<H", blob, _ELF_MACHINE_OFFSET)[0] != _EM_CUDA:
        return None
    shift = _SM_SHIFT_BY_LAYOUT.get((blob[_EI_OSABI], blob[_EI_ABIVERSION]))
    if shift is None:
        return None
    flags = struct.unpack_from("<I", blob, _ELF_FLAGS_OFFSET)[0]
    architecture = (flags >> shift) & 0xFF
    return str(architecture) if architecture else None


@dataclasses.dataclass(frozen=True)
class KernelCacheState:
    """What the cache says about the compile this run is about to pay.

    ``reason`` is ``None`` when nothing needs saying.  Everything else
    is the evidence behind it, kept so a receipt can carry WHY a notice
    fired rather than only that one did.
    """

    reason: str | None
    compute_capability: str | None
    entries: int
    entries_for_capability: int
    undecodable: int
    architectures: dict[str, int]

    @property
    def notice(self) -> str | None:
        """The one line to print, or ``None`` when silence is right."""

        if self.reason == COLD_CACHE:
            return (
                "forecast: first run on this machine -- compiling GPU "
                "kernels for the local card (typically 1-3 minutes with "
                "no visible progress; the compiled kernels are cached, "
                "so later runs skip this)")
        if self.reason == ARCHITECTURE_MISSING:
            others = ", ".join(
                f"sm_{name}" for name in sorted(self.architectures))
            return (
                f"forecast: the kernel cache holds {self.entries} entry(s), "
                f"none of them for this card -- compiling GPU kernels for "
                f"sm_{self.compute_capability} (the cache carries {others}; "
                "a card swap recompiles everything, typically 1-3 minutes "
                "with no visible progress, and the result is cached for "
                "later runs)")
        return None


def scan_kernel_cache(cache_dir: Path | None = None
                      ) -> tuple[int, int, dict[str, int]]:
    """Census one cache directory: entries, undecodable, per-architecture.

    A missing or unreadable directory censuses as empty, which is what
    every caller wants: in that state the coming run compiles from
    source either way.
    """

    directory = cupy_kernel_cache_dir() if cache_dir is None else cache_dir
    try:
        with os.scandir(directory) as scan:
            paths = [entry.path for entry in scan if entry.is_file()]
    except OSError:
        return 0, 0, {}
    # Every entry's header is one open and one small read.  On one thread that was 3.1 s for 31,266 entries before
    # every forecast's restore, with the card waiting (measured 2026-10-16 on a DA forecast leg).  The probes run
    # on a thread pool in fixed chunks and are tallied in scandir order, so the counts and the dictionary's key
    # order are the serial loop's.
    found = _probe_entries(paths)
    undecodable = 0
    architectures: dict[str, int] = {}
    for architecture in found:
        if architecture is None:
            undecodable += 1
        else:
            architectures[architecture] = architectures.get(architecture, 0) + 1
    return len(paths), undecodable, architectures


def _probe_entries(paths: list[str]) -> list[str | None]:
    """``_entry_architecture`` of every path, in order, probed on a thread pool when there are many."""

    from woof.ingest.preparation_workers import cpu_budget

    # Affinity capped by the cgroup CPU quota (U-E5: nproc 256, quota 61.44).
    workers = min(32, cpu_budget()["available_cpus"])
    if len(paths) < 512 or workers < 2:
        return [_entry_architecture(path) for path in paths]
    from concurrent.futures import ThreadPoolExecutor

    size = -(-len(paths) // (workers * 4))
    chunks = [paths[start:start + size] for start in range(0, len(paths), size)]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kernel-cache-census") as pool:
        parts = list(pool.map(lambda chunk: [_entry_architecture(path) for path in chunk], chunks))
    return [architecture for part in parts for architecture in part]


def kernel_cache_state(cache_dir: Path | None = None, *,
                       compute_capability: str | None = None,
                       census: tuple[int, int, dict[str, int]] | None = None
                       ) -> KernelCacheState:
    """Whether the coming run compiles, and on what evidence.

    ``compute_capability`` is this run's card, as
    :func:`current_compute_capability` spells it.  ``None`` means "not
    asked / no card to ask", and then only the cold-cache half applies
    -- the architecture half cannot be answered without knowing which
    architecture, and a guess there would announce a compile that never
    happens.

    ``census`` is a :func:`scan_kernel_cache` result taken EARLIER.
    Pass it, and pass it from before the run touched the GPU.  MEASURED
    on the reference box while building this: staging 200 sm_120 entries
    against an sm_86 card and running the chain, the notice stayed
    silent even with the architecture test in place -- because by the
    time the runner asks, its own first kernels have already been
    compiled INTO the cache it is asking about, and a cache with 160
    fresh sm_86 entries is warm.  The question is "was this cache usable
    when the run started", and it can only be answered with a reading
    from when the run started.
    """

    if census is None:
        entries, undecodable, architectures = scan_kernel_cache(cache_dir)
    else:
        entries, undecodable, architectures = census
    mine = (0 if compute_capability is None
            else architectures.get(str(compute_capability), 0))
    if entries == 0:
        reason = COLD_CACHE
    elif (compute_capability is not None and mine == 0
          and undecodable == 0 and architectures):
        # Positive evidence only: entries exist, every one of them
        # decoded, and not one decoded to this card.
        reason = ARCHITECTURE_MISSING
    else:
        reason = None
    return KernelCacheState(
        reason=reason,
        compute_capability=(None if compute_capability is None
                            else str(compute_capability)),
        entries=entries, entries_for_capability=mine,
        undecodable=undecodable, architectures=dict(architectures))


def kernel_cache_is_cold(cache_dir: Path | None = None) -> bool:
    """True when the CuPy kernel cache holds no compiled entries at all.

    A missing directory, an unreadable one, and an existing-but-empty
    one all count as cold: in every one of those states the coming run
    compiles from source.  Any regular file in the directory counts as
    warmth -- this predicate has no business knowing the entries'
    naming scheme.

    UNCHANGED, deliberately, now that :func:`kernel_cache_state` answers
    the larger question.  "Is anything cached at all" is still a real
    question with a cheap answer, and widening it in place would have
    made every existing caller's meaning depend on which card was
    plugged in.
    """

    directory = cupy_kernel_cache_dir() if cache_dir is None else cache_dir
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file():
                    return False
    except OSError:
        return True
    return True


def kernel_compile_notice(cache_dir: Path | None = None, *,
                          compute_capability: str | None = None
                          ) -> str | None:
    """The one line to print before compilation, or ``None``.

    ``None`` means nothing needs saying and saying anything would be
    noise.  Pass ``compute_capability`` (from
    :func:`current_compute_capability`, which the runners call) to get
    the changed-card case as well as the first-run one.
    """

    return kernel_cache_state(
        cache_dir, compute_capability=compute_capability).notice


#: The run-plan ``warning`` code a kernel compile travels under, and the
#: words a page shows for it.  One event per module the loader actually
#: compiled (it wrote to the kernel cache); a module loaded from the cache
#: says nothing, so a warm run emits none.
COMPILE_PROGRESS_CODE = "kernel_compile_progress"
COMPILE_PROGRESS_LABEL = "compiling GPU kernels"

_compile_tally_lock = threading.Lock()
_compile_tally = {"modules": 0, "seconds": 0.0}


#: The subdirectory woof.nvrtc_ptx_cache keeps its images in.
DIRECT_NVRTC_SUBDIR = "gpuwm-direct-nvrtc"


def _cache_file_count(directory: Path) -> int:
    """Files in the kernel cache, without opening any of them."""

    try:
        with os.scandir(directory) as scan:
            return sum(1 for entry in scan if entry.is_file())
    except OSError:
        return 0


@contextlib.contextmanager
def observe_module_compile(module_key: str):
    """Around one module's compile: publish it if it really compiled.

    Nothing is measured unless a run installed an event sink
    (:func:`woof.progress.event_sink`), so a forecast nobody watches pays
    nothing.  Watched, the cache is counted before and after: entries
    written means NVRTC ran, and the event carries the module, its seconds
    and the running totals, under :data:`COMPILE_PROGRESS_CODE`.
    """

    from woof import progress

    if not progress.event_sinks_installed():
        yield
        return
    directory = cupy_kernel_cache_dir()
    # The engine's direct NVRTC route (woof.nvrtc_ptx_cache: the no-FTZ
    # units such as sfclay, noah and the diffusion modules) stores its images
    # in a subdirectory, not beside CuPy's .cubin files.  Counting only the
    # top level published none of those compiles: on a cold cache the MM5
    # surface layer compiled for 110 s inside step 1 with no progress event.
    direct = directory / DIRECT_NVRTC_SUBDIR
    before = _cache_file_count(directory) + _cache_file_count(direct)
    started = time.perf_counter()
    yield
    seconds = time.perf_counter() - started
    written = (_cache_file_count(directory) + _cache_file_count(direct)
               - before)
    if written <= 0:
        return
    with _compile_tally_lock:
        _compile_tally["modules"] += 1
        _compile_tally["seconds"] += seconds
        modules = _compile_tally["modules"]
        total = _compile_tally["seconds"]
    progress.emit_event(
        "warning", code=COMPILE_PROGRESS_CODE,
        message=COMPILE_PROGRESS_LABEL, phase=COMPILE_PROGRESS_LABEL,
        module=module_key, seconds=round(seconds, 3),
        cache_entries_written=int(written), modules_compiled=modules,
        compile_seconds=round(total, 3))


__all__ = [
    "ARCHITECTURE_MISSING", "COLD_CACHE", "COMPILE_PROGRESS_CODE",
    "COMPILE_PROGRESS_LABEL", "COMPILING_STATUS",
    "CUPY_CACHE_ENV", "KernelCacheState", "cupy_kernel_cache_dir",
    "current_compute_capability", "kernel_cache_is_cold",
    "kernel_cache_state", "kernel_compile_notice",
    "observe_module_compile", "scan_kernel_cache",
]
