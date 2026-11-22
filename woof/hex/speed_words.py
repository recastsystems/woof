"""The forward-speed words: changes found by an audit of a forecast cycle on the 8-card CONUS globe (19 Nov 2026),
made engine changes here.

Every word is ON unless set to 0/off/false/no (since 19 Nov 2026, after three runs with every word on against off:
the 8-card CONUS cycle, 0 of 2,256,119,492 words different; a resume, 0 of 564,030,001 words, the restart exact; and
a cycle hour on two ranks, 0 of 1,965,521,201 words); with every word set off a run is the bytes it was.  None of them
changes arithmetic:
each one changes how many launches carry the same per-element work, which stream a copy is queued on, how often a
descriptor table is uploaded, which host copies are made, or when the parent process lets go of memory it no longer
reads.  Each word's proof (a two-rank forecast and cycle hour with the word on against off, every output word
compared) was made before the word was turned on; a word whose proof is not identical stays off.

  GPUWM_HEX_LIMITER_BATCH=N        multi-rank monotonic limiter: N species a pass, so ceil(species/N) round-E halo
                                   exchanges a step instead of one per species (cuda_transport_v841, mpi_v842)
  GPUWM_HEX_HALO_PARALLEL_RECV=1   process halo transport: each peer's receive copy on its own stream, joined before
                                   the unpack (no head-of-line wait behind a late peer); round digests cached
  WOOF_HEX_HALO_TABLE_CACHE=1     descriptor tables: least-recently-used eviction (8192) in place of clearing the
                                   whole 256-entry table cache, idle halo bindings kept 8 steps (was 2), and the
                                   tables uploaded from pinned memory on the caller's stream
  WOOF_HEX_PUBLISH_LEAN=1         the last step of a moving route downloads the host pools only when something
                                   reads them (an export, a checkpoint, a tracker frame)
  GPUWM_HEX_PUBLISH_FAST=1         a host-pool publication transposes on the card and streams through one reusable
                                   pinned buffer, in place of one pageable Fortran-order download per field
  GPUWM_HEX_RESTART_RELEASE=1      a same-host resume lets go of the global carrier and the restored words once every
                                   rank's share of them is projected, before the rank processes start

One more word (19 Nov 2026), ON since its two-rank proof (19 Nov 18:35Z: the 120 km globe on two ranks resumed from
its 3 h seal, from pickled and from raw shards, word on against off: every frame word and every owned word of the next seal
identical, 0 of 493,644,322 words; only halo columns of the next seal's physics pools hold the rank's own words in
place of the owner's):

  GPUWM_HEX_DIRECT_SHARD_RESTORE=1 a same-host resume from a checkpoint sealed as rank shards on the same ranks and the
                                   same partitions: each rank reads and restores its own shard; rank 0 no longer
                                   assembles the globe from every shard and projects it back (checkpoint_shards_v842)
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

LIMITER_BATCH = "GPUWM_HEX_LIMITER_BATCH"
HALO_PARALLEL_RECV = "GPUWM_HEX_HALO_PARALLEL_RECV"
HALO_TABLE_CACHE = "WOOF_HEX_HALO_TABLE_CACHE"
PUBLISH_LEAN = "WOOF_HEX_PUBLISH_LEAN"
PUBLISH_FAST = "GPUWM_HEX_PUBLISH_FAST"
RESTART_RELEASE = "GPUWM_HEX_RESTART_RELEASE"
DIRECT_SHARD_RESTORE = "GPUWM_HEX_DIRECT_SHARD_RESTORE"

WORDS = (LIMITER_BATCH, HALO_PARALLEL_RECV, HALO_TABLE_CACHE, PUBLISH_LEAN, PUBLISH_FAST, RESTART_RELEASE,
         DIRECT_SHARD_RESTORE)
#: Words that are OFF unless set (not yet proven identical on two ranks).
OPT_IN = ()

_OFF = ("0", "off", "false", "no")
#: GPUWM_HEX_LIMITER_BATCH when unset: 7 species a pass (13 tracers in 2 round-E exchanges), the audit's
#: measured best on 8 x 96 GB (14 crossed the 15 % release headroom and churned the pool: +21.8 s of physics).
LIMITER_BATCH_DEFAULT = 7

#: WOOF_HEX_HALO_TABLE_CACHE: entries kept, and steps an idle halo binding is kept.
TABLE_CACHE_ENTRIES = 8192
IDLE_BINDING_STEPS = 8


def on(name: str, environ: Any = None) -> bool:
    raw = (os.environ if environ is None else environ).get(name)
    if raw is None or not str(raw).strip():
        return name not in OPT_IN
    return str(raw).strip().lower() not in _OFF


def limiter_batch(environ: Any = None) -> int:
    """Species a limiter pass carries on a multi-rank schedule: 1 (one round E a species) unless the word asks more."""
    raw = str((os.environ if environ is None else environ).get(LIMITER_BATCH, "")).strip().lower()
    if not raw:
        return LIMITER_BATCH_DEFAULT
    if raw in _OFF:
        return 1
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{LIMITER_BATCH} must be a whole number of species a pass, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{LIMITER_BATCH} must be at least 1, got {value}")
    return value


def schedule_species_per_pass(species: int) -> int:
    """The species a limiter pass carries on a multi-rank schedule (mpi_v842.round_sequence, samehost_forecast_v841).

    The transport's own law first (cuda_transport_v841.limiter_species_per_pass with no exchanger and no shapes: one
    species, or WOOF_HEX_LIMITER_SPECIES_PER_PASS); then, only when that override is not set, GPUWM_HEX_LIMITER_BATCH
    widens it.  The ranks' exchangers declare ceil(species/width) round-E parts (fct_scale_parts) and the transport's
    passes follow them.  Every FCT kernel indexes tracer-major and never reads across tracers, so a pass of N species
    launches the same per-element work as N passes of one (the census-item-4 note in cuda_transport_v841).  The word
    lives here, not in cuda_transport_v841: that file is a regional translation unit whose bytes the minted regional
    classes pin (regional_admission.REGIONAL_KERNEL_SOURCES), and an edit there lapses every limited-area class.
    The memory cost is the wider pass's scratch, about one species' transport workspace for each species added; the
    compact memory rows were measured at one species a pass and do not include it.
    """
    from .cuda_transport_v841 import limiter_species_per_pass
    species = int(species)
    width = limiter_species_per_pass(species) if species else 1
    if species and os.environ.get("WOOF_HEX_LIMITER_SPECIES_PER_PASS") is None:
        width = max(width, min(limiter_batch(), species))
    return width


def active(environ: Any = None) -> dict[str, str]:
    """The words that are on, for a receipt."""
    env = os.environ if environ is None else environ
    return {name: str(env.get(name)) for name in (*WORDS, *OPT_IN) if on(name, env)}


def pinned_upload_int64(xp: Any, host: np.ndarray, stream: Any = None) -> tuple[Any, Any]:
    """An int64 table copied to the card from pinned host memory on ``stream`` (the current one when None).

    Returns (device array, pinned block).  The caller keeps the pinned block referenced until the copy has run
    (as a binding owner retained with its round, or beside the cached table): the copy is queued, not finished.
    """
    flat = np.ascontiguousarray(host, dtype=np.int64)
    if not hasattr(getattr(xp, "cuda", None), "alloc_pinned_memory"):
        return xp.asarray(flat, dtype=xp.int64), None   # a host stand-in for the card (the transport tests)
    device = xp.empty(flat.shape, dtype=xp.int64)
    if not flat.nbytes:
        return device, None
    stream = xp.cuda.get_current_stream() if stream is None else stream
    pinned = xp.cuda.alloc_pinned_memory(int(flat.nbytes))
    staged = np.frombuffer(pinned, dtype=np.int64, count=flat.size).reshape(flat.shape)
    np.copyto(staged, flat)
    xp.cuda.runtime.memcpyAsync(device.data.ptr, pinned.ptr, int(flat.nbytes),
                                xp.cuda.runtime.memcpyHostToDevice, stream.ptr)
    return device, pinned
