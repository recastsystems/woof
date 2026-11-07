"""Named step-local device workspace for the MYNN PBL solver.

Every device array the MYNN column solver needs between its own kernels is
declared here, once, as a named slot with an exact shape formula in
``(ncol, nz)``.  Nothing in :mod:`woof.core.mynn_pbl_gpu` allocates: it
draws views from a holder built here, and on the runtime path that holder
is backed by :meth:`woof.core.state.DomainState.scratch`, which is the
mechanism ``preflight.scratch_slot_registry`` prices and the shared arena
shares between domains.

Why this module exists at all
-----------------------------
The solver used to allocate its whole working set on every call.  Measured
on the RTX 5090 at ``nz = 49`` by counting every request that reached the
CuPy device allocator during one carried ``mynn_pbl_step``:

  ==========  =========================  ==========  ================
  columns     bytes allocated per step   per column  pool allocations
  ==========  =========================  ==========  ================
  25,600      1,332.9 MiB                54,595      484
  102,400     5,331.6 MiB                54,595      484
  360,000     18,743.8 MiB               54,595      484
  ==========  =========================  ==========  ================

360,000 is the d04 nest of the reference case's 4-domain config.  None of those arrays
appeared in ``physics_array_shapes`` or in the scratch registry, so the
preflight's estimate for a MYNN nest was not merely low, it was blind: what
it could not see outweighed what it could.  On a card with no ECC that is
worse than having no preflight, because the headroom check returns a
reassuring number.

The same measurement through this module reports **54.1 bytes per column**,
18.6 MiB per step at the d04 width -- a factor of 1,008 -- and every byte of
what remains resident is a named slot the registry prices.

Two properties of the declaration matter and are enforced by tests:

* **Bounded.** Slot shapes are written against a *column chunk*, not
  against ``ny * nx``, and ``mynn_pbl_runtime`` walks the domain in chunks
  of that width.  :data:`MYNN_PBL_COLUMN_CHUNK` is a ceiling, so a 600x600
  nest and a 501x501 nest declare byte-for-byte the same workspace and a
  domain narrower than the chunk declares less, never more.  A workspace
  that grew with the domain would price accurately and still not fit.
* **Write-before-read, or constant.** Every slot is either overwritten by
  the kernel that owns it before anything reads it, or it is one of the
  explicitly listed constant-zero feeds WRF passes to a system this lane
  has switched off.  :meth:`MynnPblScratch.poison` fills the first group
  with NaNs, and ``tests/test_mynn_pbl_scratch.py`` uses it to show that a
  misclassified slot changes the forecast.  The constant-zero group is
  never poisoned and never arena-aliased, following the
  ``physics_dry_qv``/``physics_dry_qc`` precedent in the lifetime audit.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from math import prod
from typing import TYPE_CHECKING

import numpy as np

# cupy is imported INSIDE the methods that allocate device memory, so
# the slot-SHAPE helpers this module also owns stay readable on
# installs with no GPU runtime -- the preflight estimator prices the
# MYNN workspace there (`woof domain` on a CPU-only box).  The
# ``cp.`` annotations below are strings under `from __future__ import
# annotations`; the TYPE_CHECKING import binds the name they spell for
# a type checker and the undefined-name gate without importing cupy at
# run time (tests/test_no_undefined_names.py).
if TYPE_CHECKING:
    import cupy as cp

from woof.core.state import DTYPE

#: Default columns per MYNN call.  Measured on the RTX 5090 at nz = 49 over
#: the 360,000-column d04 batch: a fresh input batch per row, one
#: ``initflag=1`` call, two warm-up steps, five timed steps, median.  Every
#: width produced the same carried-state hash, including the unchunked call:
#:
#:   ==========  ===========  ===========  ==============
#:   chunk       us / column  s / step     workspace MiB
#:   ==========  ===========  ===========  ==============
#:   2,048            4.4642       1.6071          505.2
#:   4,096            2.5064       0.9023          606.7
#:   8,192            1.5609       0.5619          809.7
#:   16,384           1.4929       0.5374         1215.6
#:   24,576           1.4563       0.5243         1621.6
#:   49,152           1.6716       0.6018         2839.4
#:   360,000          2.1621       0.7784        18242.8
#:   ==========  ===========  ===========  ==============
#:
#: "workspace MiB" is every live slot including the six full-width returned
#: tendency fields, which are 423.4 MiB of it at this nest and do not shrink
#: with the chunk.
#:
#: The plateau is 8k-25k wide.  Below it the walk is kernel-launch bound --
#: at 2,048 columns it is 176 launches of a kernel doing 2,048 columns of
#: work -- and above it each ``(chunk, nz)`` array leaves L2, which is the
#: same effect that made the unchunked call cost 45% more per column than
#: the plateau.  16,384 sits in the plateau, is a power of two, and holds
#: the chunk-bounded part of the workspace under 800 MiB at nz = 49.
#: 16,384 is the FLOOR of :func:`derive_mynn_column_chunk`, which is a
#: receipt term rather than the shipped width, so this constant is the lower
#: bound of the receipt's ``would_have_derived`` and nothing else.  It is NOT
#: what a run walks: :data:`MYNN_PBL_COLUMN_CHUNK_DEFAULT` is 8,192 columns,
#: measured on a card in both directions.  Floor and default were one number
#: until that second sweep; they are two now, and this one does not move.
MYNN_PBL_COLUMN_CHUNK_FLOOR = 16384

#: The widest column chunk a run walks; the width itself is the widest the
#: card's memory admits up to this cap (:func:`choose_mynn_column_chunk`).
#:
#: THE 2026-09-30 SWEEP, on the level-major layout (every MYNN column kernel
#: addresses level k of column c at k * ncol + c, so a warp's loads
#: coalesce).  That layout reversed the finding below: the old kernels
#: addressed c * nz + k, lived on L1 reuse along k, and got slower as more
#: columns shared an SM, which is why narrow chunks won.  Measured on a
#: captured real MYNN call (288x288x59 at 2.25 km, 82,944 columns, midday),
#: test host B RTX 5090 sole tenant, persistent workspace, median of five calls,
#: all arms one digest:
#:
#:   ==========  ======================  ==============
#:   columns     ms per MYNN call        workspace MiB
#:   ==========  ======================  ==============
#:    8,192                      36.59              492
#:   16,384                      23.12              984
#:   24,576                      19.26            1,475
#:   32,768                      16.38            1,967
#:   41,472                      14.45            2,490
#:   82,944                      13.65            4,980
#:   ==========  ======================  ==============
#:
#: Wider is faster all the way to one chunk per domain.  49,152 and 65,536
#: columns measured 15.52 and 15.82 ms because they leave a short last
#: chunk; walked as equal chunks (the runtime does) they are the 41,472
#: arm.  Threads per block 64 against 128 moved no arm by more than 4 per
#: cent.  In a real 1 h forecast of the same grid (Thompson, MYNN, RUC,
#: legacy RRTMG) the median step went from 132.1 ms on the old kernels to
#: 93.3 ms at 8,192 columns and 71.4 ms at 82,944, every history file's
#: data bitwise identical.
#:
#: The cap is 98,304 columns (5,901 MiB at nz = 59): one chunk for every
#: domain up to 313 x 313, taken only where 1/16 of the card holds it
#: (:data:`MYNN_PBL_RUN_WIDTH_TOTAL_VRAM_FRACTION`), so a 96 GB card runs
#: it, a 32 GB card is bounded at 32,768 columns and a 24 GB card at
#: 24,576.
#:
#: HISTORY: the 2026-09-15 sweeps on the old layout, which shipped 8,192.
#:
#: Swept through the real forecast door on an RTX 5090 at nz = 59 on
#: 2026-09-15 -- 200 root steps of the 299x299x59 + 282x129 pair's prepared
#: control leg per arm, two passes, the second in reverse order, sole
#: tenant.  Record: ``tools/mynn_chunk_sweep/summary.txt`` and
#: ``sweep.tsv`` beside it, written by the harness in
#: ``tools/mynn_chunk_sweep``.  Nine widths in two sweeps five hours apart,
#: one upward from the width 2.7.4 shipped and one downward from it; they
#: overlap at 16,384 and agree there to 0.0005 s, which is what lets the
#: nine arms be read as one curve.
#:
#: Upward, the sweep that refuted deriving the width from the card:
#:
#:   ==========  =========================  ==============
#:   columns     quiet root cycle, seconds  workspace MiB
#:   ==========  =========================  ==============
#:   16,384                         0.5263            984
#:   32,768                         0.5819          1,967
#:   45,056                         0.6170          2,705
#:   65,536                         0.6252          3,935
#:   90,112                         0.6395          5,410
#:   ==========  =========================  ==============
#:
#: Monotonic, with no knee anywhere in the range and a pass-to-pass spread
#: never worse than 0.0017 s.  All ten arms wrote the same 18 frames
#: (``faa0944f25f12f39d0002db6129c98ace47917bba153e01c849b9c5bffcbfb7c``),
#: so the width remains workspace shape and nothing else.
#:
#: What that refutes: the derivation's card ceiling on the same card is
#: 65,280 columns -- the width its register file admits in one wave -- and
#: it is 18.8 per cent SLOWER per quiet root cycle than 16,384 while
#: costing 2.9 GiB more scratch.  Wider is slower and bigger everywhere in
#: that range, so neither term of ``max(floor, min(card, VRAM))`` predicts
#: throughput here.  The derivation is kept whole, as the receipt's
#: ``would_have_derived``, so the next card's sweep has something to argue
#: with rather than a number nobody can reconstruct.
#:
#: But that sweep bounds the optimum from ONE side.  It started at the
#: width 2.7.4 shipped and only ever widened, so "narrower is faster" was
#: the whole of what it could find and it ranked its own narrowest arm
#: first.  Downward, same card, same prepared leg, same protocol:
#:
#:   ==========  =========================  ==============
#:   columns     quiet root cycle, seconds  workspace MiB
#:   ==========  =========================  ==============
#:    4,096                         0.5039            246
#:    8,192                         0.4450            492
#:   12,288                         0.4819            738
#:   16,384                         0.5268            984
#:   24,576                         0.5353          1,475
#:   ==========  =========================  ==============
#:
#: 8,192 columns is the optimum and it is INTERIOR: 15.5 per cent cheaper
#: per quiet root cycle than 16,384 on half the workspace, leading the
#: runner-up by 0.0369 s against a worst pass-to-pass spread of 0.0037 s
#: anywhere in the sweep, and with the cost turning back up by 13.2 per
#: cent at 4,096, where the launch count starts to dominate.  A minimum
#: with a slower arm on each side is a measurement; a winner at the end of
#: a ladder is only a direction.  All twenty arms across the two sweeps
#: wrote one digest.
#:
#: Confirmed on the full pair and not on the sweep alone: both 1,500-step
#: legs at 8,192 columns, with the rest of this release's installed-engine
#: work applied, took 2,390.5 s against the 16,384-column baseline's
#: 2,877.5 s -- 487.0 s and 16.9 per cent, with all 244 frames
#: byte-identical.  Per leg, 1,339.0 s to 1,064.4 s and 1,368.2 s to
#: 1,152.3 s.  37.8 s of that total is the baseline leg's cold kernel
#: compile rather than a lever, so the steady-state stepping payoff is
#: 451.7 s, of which the chunk width alone is roughly 242 s priced off the
#: sweep's own arms (164.5 s against 148.3 s per 200 cycles); the rest is
#: this release's land-surface and shortwave work.
#:
#: :data:`MYNN_PBL_COLUMN_CHUNK_FLOOR` stays at 16,384: it bounds the
#: derivation that rides the receipt.  The narrowest width a run walks is
#: :data:`MYNN_PBL_COLUMN_CHUNK_MINIMUM`, the old shipped width.
MYNN_PBL_COLUMN_CHUNK_DEFAULT = 98304

#: The narrowest width the memory bound may give: the width that shipped
#: before the level-major layout, so a card with little room runs what it
#: ran before and never less.
#:
#: It is also the width a STREAMED TILE BUFFER walks
#: (:func:`resolve_mynn_tile_column_chunk`).  Every buffer holds its own
#: workspace, so ``nbuffers`` buffers at the run's width hold ``nbuffers``
#: times what the resident domain holds.  MEASURED at 51693ec76 on the
#: 572x524x49 icon-eu MYNN forecast of tests/test_streamed_pricing_window.py
#: (two 286x286 buffers, no card, so the 98,304-column cap): 4.96 GiB of
#: scratch per buffer, a streamed envelope of 22.80 GiB against 20.14 GiB
#: resident; at this width 14.54 against 15.09.  Streaming exists to shrink
#: the device footprint, so its buffers keep the width they walked in 2.8.0.
MYNN_PBL_COLUMN_CHUNK_MINIMUM = 8192

#: The width this process will actually use, published once by
#: :func:`resolve_mynn_column_chunk` and read by anything that reports the
#: knob (``tilestream.shared_workspace.mynn_column_chunk``).  It starts at
#: the cap, which is what a process that never reaches a card uses as well.
MYNN_PBL_COLUMN_CHUNK = MYNN_PBL_COLUMN_CHUNK_DEFAULT

#: Threads per block every MYNN column kernel launches with
#: (``woof.core.mynn_pbl_gpu._TPB``).  The derived ceiling is expressed in
#: whole blocks of this width, so the last block of a full chunk is full.
MYNN_PBL_COLUMN_TPB = 128

#: Registers per thread nvcc gives ``mynn_dmp_mf_columns``, the kernel that
#: sets the occupancy of the whole family (the other three column kernels
#: are far cheaper and ride the same chunk).  READ BACK from the compiled
#: kernel into the receipt by :func:`observe_dmp_registers_per_thread`, so a
#: compiler or an architecture that allocates differently shows up as a
#: number beside this one rather than as a silent mis-derivation.  It only
#: sets a ceiling: too high leaves throughput unclaimed, too low costs a
#: second wave.  Neither is a correctness question -- every width writes the
#: same bits.
#:
#: It is architecture-dependent and the two numbers seen so far are not the
#: same: 168 from the launch geometry in the 170-SM card's profile, and 128
#: from a compile of the same source for sm_86.  That is exactly why the
#: receipt carries the observed count beside this one -- a card whose kernel
#: is cheaper in registers is being given a narrower chunk than its register
#: file would hold, and the receipt is where that shows.
MYNN_DMP_REGISTERS_PER_THREAD = 168

#: Operator override, honoured verbatim and named in the receipt.  This is
#: the handle the chunk sweep drives; it is not a tuning default.
MYNN_PBL_COLUMN_CHUNK_ENV = "WOOF_MYNN_COLUMN_CHUNK"

#: The workspace may occupy this share of the card and this share of what
#: is free on it, whichever is smaller.
#:
#: TOTAL is the term that normally binds, and that is deliberate: it is a
#: property of the card, so two runs of the same case on the same card
#: derive the same width whether or not somebody else's job is resident.
#: FREE is the clamp that holds the derivation to what is actually there
#: when the card really is occupied -- it is read at resolution time,
#: before the domain states allocate, which is why it is halved rather
#: than spent.
MYNN_PBL_CHUNK_TOTAL_VRAM_FRACTION = 0.125
MYNN_PBL_CHUNK_FREE_VRAM_FRACTION = 0.5

#: The share of the card the width a run WALKS may take (the receipt's
#: derivation above keeps its own 1/8).  Half the derivation's share,
#: because this width is priced into every forecast's memory check: on a
#: 24 GB card 1/16 is 1.5 GiB of workspace, 1.0 GiB more than the 8,192
#: columns that shipped before, which a run that fitted with a few GiB to
#: spare still fits; a larger share would move near-capacity runs onto
#: tiles.  The free-memory clamp is the derivation's.
MYNN_PBL_RUN_WIDTH_TOTAL_VRAM_FRACTION = 0.0625

#: The VRAM-derived ceiling is rounded down to a multiple of this many
#: columns (246 MiB of workspace at nz = 59).  Without it a few hundred MiB
#: of drift in what another process holds would move the derived width, and
#: with it the arena size and the memory ledger, between two runs that are
#: otherwise the same run.
MYNN_PBL_CHUNK_VRAM_QUANTUM = 4096

#: Group slots holding ``count`` arrays of one ``(chunk, nz)`` layer each.
_LAYER_GROUPS: dict[str, tuple[int, tuple[str, ...]]] = {}
#: Group slots holding ``count`` arrays of one ``(chunk, nz + 1)`` face each.
_FACE_GROUPS: dict[str, tuple[int, tuple[str, ...]]] = {}
#: Group slots holding ``count`` arrays of one ``(chunk,)`` column each.
_COLUMN_GROUPS: dict[str, tuple[int, tuple[str, ...]]] = {}


def _layers(slot: str, names: Sequence[str]) -> str:
    _LAYER_GROUPS[slot] = (len(names), tuple(names))
    return slot


def _faces(slot: str, names: Sequence[str]) -> str:
    _FACE_GROUPS[slot] = (len(names), tuple(names))
    return slot


def _columns(slot: str, names: Sequence[str]) -> str:
    _COLUMN_GROUPS[slot] = (len(names), tuple(names))
    return slot


# --- driver assembly (mynn_pbl_gpu.mynn_bl_driver_cuda) --------------------
SLOT_PREP = _layers("mynn_pbl_prep",
                    ("qv1", "sqw", "thl", "thetav", "qke_seed"))
SLOT_ZW = _faces("mynn_pbl_zw", ("zw",))
SLOT_SURFACE = _columns("mynn_pbl_surface",
                        ("flt", "fltv", "flq", "flqv", "flqc", "th_sfc",
                         "rmol", "zet", "pmz", "phh"))
SLOT_DELT = _columns("mynn_pbl_delt", ("delt",))
SLOT_DISS_HEAT = _layers("mynn_pbl_diss_heat", ("diss_heat",))
SLOT_EXCHANGE = _layers("mynn_pbl_exchange", ("exch_m", "exch_h"))

# --- GET_PBLH / SCALE_AWARE -----------------------------------------------
SLOT_PBLH = _columns("mynn_pbl_pblh", ("zi", "psig_bl", "psig_shcu"))

# --- mym_level2 (adjacent-level pairs) ------------------------------------
SLOT_LEVEL2_PAIRS = _layers(
    "mynn_pbl_level2_pairs",
    ("dz", "u", "v", "thl", "thetav", "qw", "ql", "vt", "vq",
     "dz_prev", "u_prev", "v_prev", "thl_prev", "thetav_prev",
     "qw_prev", "ql_prev", "vt_prev", "vq_prev"))
SLOT_LEVEL2_OUT = _layers("mynn_pbl_level2_out",
                          ("dtl", "dqw", "dtv", "gm", "gh", "sm", "sh"))

# --- mym_length -----------------------------------------------------------
SLOT_MIXLENGTH = _layers("mynn_pbl_mixlength", ("el", "qkw"))
SLOT_MIXLENGTH_WORK = _layers("mynn_pbl_mixlength_work",
                              tuple(f"w{k}" for k in range(5)))

# --- mym_turbulence -------------------------------------------------------
SLOT_LEVEL2_FULL = _layers("mynn_pbl_level2_full",
                           ("dtl", "dqw", "dtv", "gm", "gh", "sm", "sh"))
SLOT_TURBULENCE = _layers("mynn_pbl_turbulence",
                          ("dfm", "dfh", "dfq", "tcd", "qcd", "pdk", "pdt",
                           "pdq", "pdc"))

# --- mym_predict ----------------------------------------------------------
SLOT_PREDICT = _layers("mynn_pbl_predict", ("qke", "tsq", "qsq", "cov"))
SLOT_PREDICT_WORK = _layers("mynn_pbl_predict_work",
                            tuple(f"w{k}" for k in range(10)))

# --- mym_condensation -----------------------------------------------------
SLOT_CONDENSATION = _layers("mynn_pbl_condensation",
                            ("qc_bl", "qi_bl", "cldfra", "vt", "vq", "sgm"))

# --- mym_initialize -------------------------------------------------------
SLOT_INITIALIZE = _layers("mynn_pbl_initialize",
                          ("el", "qke", "tsq", "qsq", "cov", "sm", "sh"))
#: The kernel packs qkw, qtke, thetaw, elBLavg, dlu, dld, dtl, dqw, dtv,
#: gm, gh, pdk, pdt, pdq, pdc into one buffer; see
#: ``_INITIALIZE_SCRATCH_VECTORS`` in mynn_pbl_gpu.
SLOT_INITIALIZE_WORK = _layers("mynn_pbl_initialize_work",
                               tuple(f"v{k}" for k in range(15)))

# --- DMP_mf ---------------------------------------------------------------
SLOT_PLUME_LAYER = _layers("mynn_pbl_plume_layer",
                           ("edmf_a", "edmf_w", "edmf_qt", "edmf_thl",
                            "edmf_ent", "edmf_qc", "qc_bl", "cldfra_bl",
                            "vt", "vq"))
SLOT_PLUME_FACE = _faces("mynn_pbl_plume_face",
                         ("s_aw", "s_awthl", "s_awqt", "s_awqv", "s_awqc",
                          "s_awu", "s_awv"))
SLOT_PLUME_COLUMN = _columns("mynn_pbl_plume_column",
                             ("maxwidth", "ztop", "maxmf"))
#: 8 plume vectors x 8 plumes x (nz + 1) faces per column.  The single
#: largest slot in the manifest: 12,800 bytes per column at nz = 49, which
#: is 23.4% of the 54,595 bytes per column the solver used to allocate.
SLOT_PLUME_WORK = _faces("mynn_pbl_plume_work",
                         tuple(f"p{k}" for k in range(64)))
#: 3 work vectors + 8 per-plume vectors, layer sized.
SLOT_PLUME_SCRATCH = _layers("mynn_pbl_plume_scratch",
                             tuple(f"w{k}" for k in range(11)))

# --- GSD MYNN v4.1 (bl_mynn_version = "gsd_41") only ----------------------
#: Layer groups that exist only under the GSD generation.  They are NOT in
#: ``_LAYER_GROUPS``, so the default (wrf_461) workspace, its price and its
#: arena shapes are unchanged; :func:`mynn_pbl_scratch_shapes` adds them when
#: the generation asks.  The breakage they close: 2ea33cea9 and 32b58a2af
#: drew these two working sets with raw ``cp.empty`` calls inside the
#: converted solver, which the run's preflight never priced (only the
#: ensemble plan reserved them, as transients), so a gsd_41 nest could pass
#: its memory check and still allocate six float32 values (24 bytes) per
#: column level of every chunk on top.
_GSD41_LAYER_GROUPS: dict[str, tuple[int, tuple[str, ...]]] = {}


def _gsd41_layers(slot: str, names: Sequence[str]) -> str:
    _GSD41_LAYER_GROUPS[slot] = (len(names), tuple(names))
    return slot


#: mym_condensation CASE(2)'s five work columns (q1, rh, a, b, cld).
SLOT_GSD41_CONDENSATION_WORK = _gsd41_layers(
    "mynn_pbl_gsd41_condensation_work", ("q1", "rh", "a", "b", "cld"))
#: GET_PBLH's theta-v of the liquid-water theta (thvl).
SLOT_GSD41_THVL = _gsd41_layers("mynn_pbl_gsd41_thvl", ("thvl",))
# Five native driver diagnostics. Two words hold int32 plume indices through
# zero-copy views; this slot exists only for the GSD generation.
SLOT_GSD41_DIAGNOSTICS = "mynn_pbl_gsd41_diagnostics"
GSD41_DIAGNOSTICS = ("wstar", "delta", "mf_at_base", "kbot_shallow", "nupdraft")

# --- mynn_tendencies ------------------------------------------------------
SLOT_TENDENCY = _layers("mynn_pbl_tendency",
                        ("du", "dv", "dth", "dqv", "dqc", "dqi", "dqs",
                         "dozone", "thl"))
SLOT_TENDENCY_WORK = _layers("mynn_pbl_tendency_work",
                             ("dtz", "rhoinv", "delp", "a", "b", "c", "d",
                              "cpw", "dpw", "sqv2", "sqc2", "sqi2", "sqs2"))
SLOT_TENDENCY_FACE = _faces("mynn_pbl_tendency_face", ("khdz", "kmdz"))

# --- constant-zero feeds --------------------------------------------------
# WRF passes these to systems this lane's pinned identity switches off.  No
# kernel writes them; they must read as zero on every call, so they are
# excluded from poisoning and from arena aliasing.
SLOT_ZERO_LAYER = _layers("mynn_pbl_zero_layer", ("zero", "snow"))
SLOT_ZERO_FACE = _faces("mynn_pbl_zero_face", ("zero",))
SLOT_PLUME_ZERO_LAYER = _layers(
    "mynn_pbl_plume_zero_layer",
    ("sub_thl", "sub_sqv", "sub_u", "sub_v", "det_thl", "det_sqv",
     "det_sqc", "det_u", "det_v"))
SLOT_PLUME_ZERO_FACE = _faces(
    "mynn_pbl_plume_zero_face",
    ("s_awqke", "s_awqnc", "s_awqni", "s_awqnwfa", "s_awqnifa",
     "s_awqnbca"))
SLOT_TENDENCY_ZERO = _layers("mynn_pbl_tendency_zero",
                             ("dqnc", "dqni", "dqnwfa", "dqnifa", "dqnbca"))

# --- the wrapper's column staging (mynn_pbl_runtime) ----------------------
#: ``mynnedmf_wrapper_run`` transposes ``(nz, ny, nx)`` fields into column
#: batches.  Doing that at full width allocated 27 arrays of ``ny * nx``
#: columns per step -- 1,682.3 MiB at the d04 nest, on top of the solver's
#: own working set and equally invisible to the preflight.  Staged one chunk
#: at a time these are bounded and priced like everything else.
MYNN_PBL_STAGE_LAYERS = (
    "dz", "u", "v", "w", "th", "p", "exner", "rho", "tk",
    "qv", "qc", "qi", "qs", "sqv", "sqc", "sqi", "sqs",
    "qke", "tsq", "qsq", "cov", "el", "sh", "sm",
    "qc_bl", "qi_bl", "cldfra_bl",
)
SLOT_STAGE_LAYER = _layers("mynn_pbl_stage_layer", MYNN_PBL_STAGE_LAYERS)
SLOT_STAGE_DX = _columns("mynn_pbl_stage_dx", ("dx",))
#: woof has no ocean-current coupling, so WRF's UOCE/VOCE stay at their
#: Registry default of zero; one constant-zero column serves both.
SLOT_ZERO_COLUMN = _columns("mynn_pbl_zero_column", ("zero",))

#: Slots that hold zeros for the whole run.  Read before written by
#: construction, so they are correctness-critical to leave alone.
MYNN_PBL_CONSTANT_ZERO_SLOTS = frozenset((
    SLOT_ZERO_LAYER, SLOT_ZERO_FACE, SLOT_PLUME_ZERO_LAYER,
    SLOT_PLUME_ZERO_FACE, SLOT_TENDENCY_ZERO, SLOT_ZERO_COLUMN,
))

#: int32 slots.  ``ScratchArena`` is float32-only, so these are priced by
#: the registry and allocated per state rather than shared.
MYNN_PBL_INDEX_SLOTS: dict[str, tuple[str, ...]] = {
    "mynn_pbl_kpbl": ("kpbl",),
    "mynn_pbl_pblh_index": ("kzi",),
    "mynn_pbl_plume_index": ("ktop",),
}

#: One int32 word per validated array.  ``_tendency_device_arrays`` used to
#: evaluate ``bool(cp.isfinite(a).all())`` per array, which allocated a full
#: ``(ncol, nz)`` bool temporary and forced a device synchronisation for each
#: one.  The reductions now write one of these persistent words per array
#: and the whole group is read back in a single copy, so a predicate group
#: costs one synchronisation instead of one per array.
#:
#: 64 words because the widest group is ``mynn_tendencies``' 50 validated
#: inputs; ``_flag_mask`` reads a longer group a block at a time rather than
#: silently truncating it.
MYNN_PBL_FLAG_SLOTS = ("mynn_pbl_validity_flags",)
_FLAG_WORDS = 64


def mynn_pbl_scratch_shapes(chunk: int, nz: int, *,
                            bl_mynn_version: str = "wrf_461") -> dict[str, tuple[int, ...]]:
    """Flat float32 slot shapes for one MYNN call of ``chunk`` columns.

    Flat because the holder hands out reshaped contiguous prefixes, exactly
    as :meth:`woof.core.state.ScratchArena.view` does, and because the
    oracle callers of the individual leaves pass shapes the driver never
    uses (``mym_level2`` is launched on adjacent-level pairs, one element
    shorter than a layer).
    """
    chunk = int(chunk)
    nz = int(nz)
    if chunk < 1 or nz < 1:
        raise ValueError(f"MYNN scratch needs chunk >= 1 and nz >= 1, got "
                         f"{chunk} and {nz}")
    shapes: dict[str, tuple[int, ...]] = {}
    for slot, (count, _names) in _LAYER_GROUPS.items():
        shapes[slot] = (count * chunk * nz,)
    for slot, (count, _names) in _FACE_GROUPS.items():
        shapes[slot] = (count * chunk * (nz + 1),)
    for slot, (count, _names) in _COLUMN_GROUPS.items():
        shapes[slot] = (count * chunk,)
    if bl_mynn_version == "wrf_471":
        from woof.core.mynn471_gpu import scratch_shapes
        # The 4.7.1 monolithic column owns its internal workspace. Keep
        # only the wrapper's staging slots from the older split solver.
        shapes = {slot: shapes[slot] for slot in
                  (SLOT_STAGE_LAYER, SLOT_STAGE_DX, SLOT_ZERO_COLUMN)}
        shapes.update(scratch_shapes(chunk, nz))
    if bl_mynn_version == "gsd_41":
        shapes[SLOT_GSD41_DIAGNOSTICS] = (len(GSD41_DIAGNOSTICS) * chunk,)
        # Ten plume classes, eight state vectors and three work vectors.
        shapes[SLOT_PLUME_WORK] = (80 * chunk * (nz + 1),)
        shapes[SLOT_PLUME_SCRATCH] = (13 * chunk * nz,)
        for slot, (count, _names) in _GSD41_LAYER_GROUPS.items():
            shapes[slot] = (count * chunk * nz,)
    return shapes


def mynn_pbl_index_shapes(chunk: int, nz: int) -> dict[str, tuple[int, ...]]:
    """Flat int32 slot shapes for one MYNN call of ``chunk`` columns."""
    chunk = int(chunk)
    return {slot: (len(names) * chunk,)
            for slot, names in MYNN_PBL_INDEX_SLOTS.items()}


def mynn_pbl_flag_shapes() -> dict[str, tuple[int, ...]]:
    """Flat int32 validity-flag slot shapes (independent of size)."""
    return {slot: (_FLAG_WORDS,) for slot in MYNN_PBL_FLAG_SLOTS}


def mynn_pbl_scratch_bytes(chunk: int, nz: int, *, bl_mynn_version: str = "wrf_461") -> int:
    """Total device bytes one MYNN workspace occupies."""
    total = sum(shape[0] for shape in
                mynn_pbl_scratch_shapes(chunk, nz, bl_mynn_version=bl_mynn_version).values()) * 4
    total += sum(shape[0] for shape in
                 mynn_pbl_index_shapes(chunk, nz).values()) * 4
    total += sum(shape[0] for shape in mynn_pbl_flag_shapes().values()) * 4
    return int(total)


def mynn_pbl_column_bytes(nz: int, *, bl_mynn_version: str = "wrf_461") -> int:
    """Device bytes one more column of MYNN workspace costs at ``nz``.

    The difference of two widths rather than a division, so the flag words
    (which do not scale with the chunk) are not smeared across the columns.
    62,952 bytes at nz = 59; 52,352 at nz = 49.
    """
    return (mynn_pbl_scratch_bytes(2, nz, bl_mynn_version=bl_mynn_version)
            - mynn_pbl_scratch_bytes(1, nz, bl_mynn_version=bl_mynn_version))


def mynn_pbl_card_chunk_ceiling(
        *, sm_count: int, max_threads_per_sm: int,
        registers_per_sm: int, warp_size: int = 32,
        registers_per_thread: int | None = None,
        threads_per_block: int | None = None) -> int:
    """The widest chunk whose blocks still fit the card in one wave.

    ``mynn_dmp_mf_columns`` gives one thread one whole column and integrates
    it vertically, so a chunk of ``c`` columns is ``ceil(c / TPB)`` blocks of
    ``TPB`` threads and nothing else.  The card holds

      ``SMs x min(max threads per SM, registers per SM / registers per
      thread, rounded down to a warp) / TPB``

    blocks of it at once.  Above that the chunk buys a second wave, which is
    what the fixed width already did the other way round: 128 blocks on a
    170-SM card left 42 SMs with nothing and gave every other SM one block
    where the register file admits three.

    Every term comes from the device at run time.  ``registers_per_thread``
    is the one property of the compiled kernel rather than of the card; it
    defaults to :data:`MYNN_DMP_REGISTERS_PER_THREAD`.
    """
    sm_count = int(sm_count)
    max_threads_per_sm = int(max_threads_per_sm)
    registers_per_sm = int(registers_per_sm)
    warp_size = max(1, int(warp_size))
    tpb = int(MYNN_PBL_COLUMN_TPB if threads_per_block is None
              else threads_per_block)
    regs = int(MYNN_DMP_REGISTERS_PER_THREAD if registers_per_thread is None
               else registers_per_thread)
    if min(sm_count, max_threads_per_sm, registers_per_sm, tpb, regs) < 1:
        raise ValueError(
            "the MYNN chunk ceiling needs a positive SM count, threads per "
            "SM, registers per SM, block width and register count; got "
            f"{sm_count}, {max_threads_per_sm}, {registers_per_sm}, {tpb}, "
            f"{regs}")
    by_registers = (registers_per_sm // regs) // warp_size * warp_size
    threads_per_sm = min(max_threads_per_sm, by_registers)
    blocks_per_sm = threads_per_sm // tpb
    return max(1, sm_count * blocks_per_sm) * tpb


def mynn_pbl_vram_chunk_ceiling(
        nz: int, *, free_bytes: int, total_bytes: int,
        total_fraction: float = MYNN_PBL_CHUNK_TOTAL_VRAM_FRACTION) -> int:
    """The widest chunk the card's memory admits, quantised for stability.

    The budget is the smaller of a share of the card and a share of what is
    free on it (:data:`MYNN_PBL_CHUNK_TOTAL_VRAM_FRACTION`,
    :data:`MYNN_PBL_CHUNK_FREE_VRAM_FRACTION`), and the result is rounded
    down to :data:`MYNN_PBL_CHUNK_VRAM_QUANTUM` columns so that ordinary
    drift in what else is resident does not move the derived width.
    """
    per_column = mynn_pbl_column_bytes(nz)
    budget = min(int(total_bytes) * float(total_fraction),
                 max(0, int(free_bytes)) * MYNN_PBL_CHUNK_FREE_VRAM_FRACTION)
    columns = int(budget // per_column)
    quantum = int(MYNN_PBL_CHUNK_VRAM_QUANTUM)
    return max(0, columns // quantum * quantum)


@dataclasses.dataclass(frozen=True)
class MynnColumnChunk:
    """One derivation of the MYNN column width, with every term it used."""

    chunk: int
    source: str
    nz: int
    floor: int
    column_bytes: int
    workspace_bytes: int
    card_ceiling: int | None = None
    vram_ceiling: int | None = None
    sm_count: int | None = None
    max_threads_per_sm: int | None = None
    registers_per_sm: int | None = None
    registers_per_thread: int | None = None
    threads_per_block: int = MYNN_PBL_COLUMN_TPB
    free_bytes: int | None = None
    total_bytes: int | None = None
    device_name: str | None = None
    #: What :func:`derive_mynn_column_chunk` would have chosen on this card,
    #: carried for the receipt and never used as the width.  It is the whole
    #: derivation -- both ceilings and every device term behind them -- so a
    #: sweep on the NEXT card can be compared against what that card's
    #: register file and memory would have asked for.
    would_have_derived: "MynnColumnChunk | None" = None

    @property
    def blocks(self) -> int:
        return -(-int(self.chunk) // int(self.threads_per_block))

    def receipt(self) -> dict:
        """The width as receipt fields, the override named either way."""
        entry = {key: value for key, value
                 in dataclasses.asdict(self).items() if value is not None}
        entry["blocks"] = self.blocks
        entry["override_env"] = MYNN_PBL_COLUMN_CHUNK_ENV
        if self.would_have_derived is not None:
            nested = self.would_have_derived.receipt()
            nested.pop("override_env", None)
            nested.pop("would_have_derived", None)
            entry["would_have_derived"] = nested
        return entry

    def line(self) -> str:
        derived = self.would_have_derived
        tail = ("" if derived is None
                else (f"; this card would have derived {derived.chunk} "
                      f"({derived.source}, card ceiling "
                      f"{derived.card_ceiling}, VRAM ceiling "
                      f"{derived.vram_ceiling}), which the 2026-09-15 sweep "
                      f"measured slower"))
        ceilings = ("" if self.card_ceiling is None
                    else (f", card ceiling {self.card_ceiling}, VRAM "
                          f"ceiling {self.vram_ceiling}"))
        return (f"MYNN column chunk {self.chunk} columns ({self.source}), "
                f"{self.workspace_bytes / 2 ** 20:.1f} MiB of workspace at "
                f"nz={self.nz}, {self.blocks} blocks x "
                f"{self.threads_per_block} threads; derivation floor "
                f"{self.floor}"
                f"{ceilings}; override with "
                f"{MYNN_PBL_COLUMN_CHUNK_ENV}" + tail)


def _chunk_choice(chunk: int, source: str, nz: int, **terms
                  ) -> "MynnColumnChunk":
    chunk = int(chunk)
    return MynnColumnChunk(
        chunk=chunk, source=source, nz=int(nz),
        floor=int(MYNN_PBL_COLUMN_CHUNK_FLOOR),
        column_bytes=mynn_pbl_column_bytes(nz),
        workspace_bytes=mynn_pbl_scratch_bytes(chunk, nz), **terms)


def mynn_column_chunk_override(environ: Mapping[str, str] | None = None
                               ) -> int | None:
    """The operator's chunk, or ``None``.  Refuses a value that is not one.

    An unparseable override is refused rather than ignored: the sweep sets
    this variable to decide how much workspace the run allocates, so a typo
    that silently fell back to the shipped default would report a width the
    run did not use and a timing that answers a different question.
    """
    import os

    raw = (os.environ if environ is None else environ).get(
        MYNN_PBL_COLUMN_CHUNK_ENV, "")
    raw = str(raw).strip()
    if not raw:
        return None
    try:
        chunk = int(raw, 10)
    except ValueError:
        chunk = 0
    if chunk < 1:
        raise ValueError(
            f"{MYNN_PBL_COLUMN_CHUNK_ENV}={raw!r} is not a positive column "
            "count; it sizes the MYNN workspace and the shared arena, so a "
            "value that cannot be read is refused rather than replaced by "
            f"the shipped {MYNN_PBL_COLUMN_CHUNK_DEFAULT}-column default.  "
            "Unset it, or give a positive integer number of columns")
    return chunk


def probe_mynn_card(device: int | None = None) -> dict | None:
    """The card terms the derivation needs, or ``None`` off a GPU.

    Attributes and free/total memory only: no allocation, no kernel, no
    compile.  A CPU-only box (where ``woof domain`` prices this scheme)
    takes the ``None`` road and runs the shipped default; only the
    derivation's own ``None`` road answers with the floor.
    """
    try:
        import cupy as cp

        index = cp.cuda.runtime.getDevice() if device is None else int(device)
        attributes = cp.cuda.Device(index).attributes
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        name = cp.cuda.runtime.getDeviceProperties(index)["name"]
    except Exception:  # noqa: BLE001 - any unusable runtime means "no card"
        return None
    try:
        return {
            "sm_count": int(attributes["MultiProcessorCount"]),
            "max_threads_per_sm": int(
                attributes["MaxThreadsPerMultiProcessor"]),
            "registers_per_sm": int(
                attributes["MaxRegistersPerMultiprocessor"]),
            "warp_size": int(attributes.get("WarpSize", 32)),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "device_name": (name.decode() if isinstance(name, bytes)
                            else str(name)),
        }
    except KeyError:
        return None


def derive_mynn_column_chunk(nz: int, *, card: Mapping | None = None,
                             environ: Mapping[str, str] | None = None
                             ) -> "MynnColumnChunk":
    """What this card's own numbers ask for.  Pure in its arguments.

    ``max(floor, min(card ceiling, VRAM ceiling))`` -- the width that fills
    the card in one wave, clipped by what the card's memory admits, never
    below :data:`MYNN_PBL_COLUMN_CHUNK_FLOOR`.  That floor bounds THIS
    function and nothing else; the width a run walks is settled by
    :func:`choose_mynn_column_chunk`.

    **This is a receipt term, not the shipped width.**  The 2026-09-15
    sweeps (see :data:`MYNN_PBL_COLUMN_CHUNK_DEFAULT`) measured the card
    ceiling this returns on a 5090 at 18.8 per cent slower per quiet root
    cycle than 16,384 columns, monotonically, with no plateau anywhere in
    that range -- and then measured 8,192 columns 15.5 per cent cheaper
    again, which is what ships.  So :func:`choose_mynn_column_chunk` --
    which is what the run actually calls -- ships a measured policy and
    hangs this whole derivation off it as ``would_have_derived``.  (Those
    sweeps predate the level-major layout; see the 2026-09-30 sweep.)  Keeping
    it costs one device-attribute read and gives the next card's sweep a
    claim to test; using it cost 18.8 per cent and 2.9 GiB against the
    width it was compared with, and more against the one that ships.

    The width is workspace shape only.  Every MYNN column kernel gives one
    thread one whole column, reads no neighbour, and holds no shared memory,
    no atomic and no per-chunk seed, so a run at any width writes the same
    bits; ``tests/test_mynn_pbl_scratch.py`` asserts the split against the
    single wide call rather than assuming it, and all ten arms of the sweep
    wrote one digest.
    """
    override = mynn_column_chunk_override(environ)
    if override is not None:
        return _chunk_choice(override, "override", nz)
    if card is None:
        card = probe_mynn_card()
    if card is None:
        return _chunk_choice(MYNN_PBL_COLUMN_CHUNK_FLOOR, "floor-no-card", nz)
    registers_per_thread = int(card.get("registers_per_thread")
                               or MYNN_DMP_REGISTERS_PER_THREAD)
    card_ceiling = mynn_pbl_card_chunk_ceiling(
        sm_count=card["sm_count"],
        max_threads_per_sm=card["max_threads_per_sm"],
        registers_per_sm=card["registers_per_sm"],
        warp_size=int(card.get("warp_size", 32)),
        registers_per_thread=registers_per_thread)
    vram_ceiling = mynn_pbl_vram_chunk_ceiling(
        nz, free_bytes=card["free_bytes"], total_bytes=card["total_bytes"])
    admitted = min(card_ceiling, vram_ceiling)
    if admitted <= MYNN_PBL_COLUMN_CHUNK_FLOOR:
        chunk, source = MYNN_PBL_COLUMN_CHUNK_FLOOR, "floor"
    elif vram_ceiling < card_ceiling:
        chunk, source = admitted, "vram-ceiling"
    else:
        chunk, source = admitted, "card-ceiling"
    return _chunk_choice(
        chunk, source, nz,
        card_ceiling=card_ceiling, vram_ceiling=vram_ceiling,
        sm_count=int(card["sm_count"]),
        max_threads_per_sm=int(card["max_threads_per_sm"]),
        registers_per_sm=int(card["registers_per_sm"]),
        registers_per_thread=registers_per_thread,
        free_bytes=int(card["free_bytes"]),
        total_bytes=int(card["total_bytes"]),
        device_name=card.get("device_name"))


@lru_cache(maxsize=1)
def _pricing_card_memory_rows() -> dict[float, int]:
    import json
    from importlib.resources import files

    table = json.loads((files("woof") / "authorities" /
                        "mynn-card-memory.v1.json").read_text(encoding="utf-8"))
    return {float(row["capacity_gib"]): int(row["cuda_total_bytes"])
            for row in table["rows"]}


def mynn_pricing_total_bytes(vram_gib: float, *, measured: bool = False) -> int:
    """CUDA-usable total for a capacity tier, or the exact measured total.

    Nameplate capacity is not CUDA's total. At 49 levels, the measured
    16 and 32 GiB reference cards fall one width quantum below nameplate
    pricing. No host probe substitutes its card for the forecast target.
    Tiers with no sample retain their stated total; hardware snapshots
    always retain their own measurement.
    """
    total = int(float(vram_gib) * 1024 ** 3)
    if measured:
        return total
    return _pricing_card_memory_rows().get(float(vram_gib), total)


def mynn_column_chunk_for_memory(nz: int, *, total_bytes: int,
                                 free_bytes: int,
                                 environ: Mapping[str, str] | None = None
                                 ) -> "MynnColumnChunk":
    """The run's memory policy, shared by live cards and target-card fits."""
    override = mynn_column_chunk_override(environ)
    if override is not None:
        return _chunk_choice(override, "override", nz)
    vram_ceiling = mynn_pbl_vram_chunk_ceiling(
        nz, free_bytes=free_bytes, total_bytes=total_bytes,
        total_fraction=MYNN_PBL_RUN_WIDTH_TOTAL_VRAM_FRACTION)
    if vram_ceiling >= MYNN_PBL_COLUMN_CHUNK_DEFAULT:
        chunk, source = MYNN_PBL_COLUMN_CHUNK_DEFAULT, "measured"
    elif vram_ceiling > MYNN_PBL_COLUMN_CHUNK_MINIMUM:
        chunk, source = vram_ceiling, "measured-vram-bounded"
    else:
        chunk, source = MYNN_PBL_COLUMN_CHUNK_MINIMUM, "measured-minimum"
    return _chunk_choice(chunk, source, nz, vram_ceiling=vram_ceiling,
                         free_bytes=int(free_bytes),
                         total_bytes=int(total_bytes))


_PRICING_MEMORY: ContextVar[tuple[int, int] | None] = ContextVar(
    "mynn_pricing_memory", default=None)
_PRICING_RANK_CHUNK: ContextVar[int | None] = ContextVar(
    "mynn_pricing_rank_chunk", default=None)
_PRICING_RESIDENT_CHUNK: ContextVar[int | None] = ContextVar(
    "mynn_pricing_resident_chunk", default=None)


@contextmanager
def mynn_pricing_rank_chunk(chunk: int | None):
    """Price one resident rank at its selected width without changing a run.

    Rank fits can run concurrently, and the runtime's global pin also
    controls other domains. A context-local value keeps a candidate fit
    out of those owners. Only the tile-buffer resolver reads this value.
    """
    if chunk is not None and (type(chunk) is not int or chunk < 1):
        raise ValueError("a ranked MYNN price needs a positive integer width")
    token = _PRICING_RANK_CHUNK.set(chunk)
    try:
        yield
    finally:
        _PRICING_RANK_CHUNK.reset(token)


@contextmanager
def mynn_pricing_resident_chunk(chunk: int | None):
    """Price a RESIDENT domain at ``chunk`` columns without changing a run.

    The single-card road's counterpart of :func:`mynn_pricing_rank_chunk`
    (which only the tile-buffer resolver reads).  The resident road prices
    what the card NEEDS at the narrowest width a run may walk and fits the
    width it walks into what is left (A3c on the ranked road;
    :func:`woof.core.streaming.admit_resident_road` on one card), so its
    pricing must be able to ask for one width without pinning the process
    or touching the memo the runtime resolves from.  ``None`` leaves the
    width to :func:`resolve_mynn_column_chunk`'s usual answer.
    """
    if chunk is not None and (type(chunk) is not int or chunk < 1):
        raise ValueError("a resident MYNN price needs a positive integer width")
    token = _PRICING_RESIDENT_CHUNK.set(chunk)
    try:
        yield
    finally:
        _PRICING_RESIDENT_CHUNK.reset(token)


def mynn_target_fit_active() -> bool:
    """Is a target-card fit (:func:`mynn_pricing_memory`) pricing now?"""
    return _PRICING_MEMORY.get() is not None


@contextmanager
def mynn_pricing_memory(*, total_bytes: int, free_bytes: int):
    """Price a forecast target without changing this process's run width.

    Context-local so concurrent fits for different cards cannot share a
    width. Every nested resident or streamed estimate uses this same
    target, and leaving the fit restores the runtime resolution.
    """
    token = _PRICING_MEMORY.set((int(total_bytes), int(free_bytes)))
    try:
        yield
    finally:
        _PRICING_MEMORY.reset(token)


def choose_mynn_column_chunk(nz: int, *, card: Mapping | None = None,
                             environ: Mapping[str, str] | None = None
                             ) -> "MynnColumnChunk":
    """The width this process runs at, and the derivation it did not use.

    The 2026-09-30 sweep (see :data:`MYNN_PBL_COLUMN_CHUNK_DEFAULT`) found
    the level-major kernels faster at every wider chunk up to one chunk per
    domain, so the width is the widest the card's memory admits
    (:func:`mynn_pbl_vram_chunk_ceiling` at
    :data:`MYNN_PBL_RUN_WIDTH_TOTAL_VRAM_FRACTION` of the card and 1/2 of
    what is free, quantised so a neighbour's drift does not move it), capped at
    :data:`MYNN_PBL_COLUMN_CHUNK_DEFAULT` and never below
    :data:`MYNN_PBL_COLUMN_CHUNK_MINIMUM`. Off a card without a target
    memory sample the cap is priced, the widest any card runs. A fit with
    a target sample uses :func:`mynn_column_chunk_for_memory` instead.
    An operator override replaces it
    verbatim and says so.  The card-ceiling derivation still rides the
    receipt as ``would_have_derived``.

    The width never changes a bit: every MYNN column kernel gives one thread
    one whole column and reads no neighbour, which
    ``tests/test_mynn_pbl_scratch.py::test_the_column_chunk_is_not_a_seam``
    asserts rather than assumes.

    Source is ``measured`` when the cap binds, ``measured-vram-bounded``
    when the card's memory does, ``measured-minimum`` when the minimum
    does, ``measured-no-card`` off a card and ``override`` for the
    environment variable.
    """
    if card is None:
        card = probe_mynn_card()
    derived = derive_mynn_column_chunk(nz, card=card, environ={})
    override = mynn_column_chunk_override(environ)
    if override is not None:
        return _chunk_choice(override, "override", nz,
                             would_have_derived=derived)
    if card is None:
        return _chunk_choice(MYNN_PBL_COLUMN_CHUNK_DEFAULT,
                             "measured-no-card", nz,
                             would_have_derived=derived)
    choice = mynn_column_chunk_for_memory(
        nz, free_bytes=card["free_bytes"], total_bytes=card["total_bytes"],
        environ=environ)
    return dataclasses.replace(choice, device_name=card.get("device_name"),
                               would_have_derived=derived)


def mynn_column_pieces(ncol: int, width: int) -> int:
    """Columns per call when ``ncol`` columns are walked at most ``width``
    at a time, in equal pieces.

    ``ceil(ncol / ceil(ncol / width))``: the fewest calls the width allows,
    as equal as the column count allows, so no call is a short remainder.
    82,944 columns at a 65,536 width walk as two calls of 41,472 (14.45 ms
    on the 2026-09-30 sweep) instead of 65,536 and 17,408 (15.82 ms).
    """
    ncol, width = int(ncol), int(width)
    if ncol < 1 or width < 1:
        raise ValueError("MYNN column pieces need positive column counts")
    calls = -(-ncol // width)
    return -(-ncol // calls)


#: The chosen width, once per process per ``nz``.  Memoised because the
#: preflight registry, the shared arena and the solver must all size the
#: same workspace: a second look at free VRAM between them is a second
#: answer, and ``DomainState.scratch`` refuses a slot that changes shape.
_RESOLVED: dict[int, "MynnColumnChunk"] = {}
_PINNED: int | None = None
#: The width streamed tile buffers walked, per ``nz``, for the receipt.
_TILE_WALKED: dict[int, int] = {}

#: Every module that binds ``MYNN_PBL_COLUMN_CHUNK``.  The resolved width is
#: published into the ones already imported so the knob reports what the run
#: uses; nothing is imported for the sake of publishing it.
_CHUNK_MODULES = ("woof.core.mynn_pbl_scratch",
                  "woof.core.mynn_pbl_runtime",
                  "woof.core.mynn_pbl_gpu")


def _publish_column_chunk(chunk: int) -> None:
    import sys

    for name in _CHUNK_MODULES:
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "MYNN_PBL_COLUMN_CHUNK"):
            module.MYNN_PBL_COLUMN_CHUNK = int(chunk)


def resolve_mynn_column_chunk(nz: int) -> int:
    """This process's MYNN column width at ``nz``, chosen once.

    Called by the preflight registry (which sizes the shared arena) and by
    ``mynn_pbl_runtime`` (which walks the domain).  Both must get the same
    number, which is what the memo is for.
    """
    nz = int(nz)
    priced = _PRICING_RESIDENT_CHUNK.get()
    if priced is not None:
        return int(priced)
    if _PINNED is not None:
        return int(_PINNED)
    memory = _PRICING_MEMORY.get()
    if memory is not None:
        total_bytes, free_bytes = memory
        return mynn_column_chunk_for_memory(
            nz, total_bytes=total_bytes, free_bytes=free_bytes).chunk
    choice = _RESOLVED.get(nz)
    if choice is None:
        choice = choose_mynn_column_chunk(nz)
        _RESOLVED[nz] = choice
        _publish_column_chunk(choice.chunk)
    return int(choice.chunk)


def resolve_mynn_tile_column_chunk(nz: int, *, walking: bool = False) -> int:
    """The MYNN column width a streamed tile buffer walks and is priced at.

    :data:`MYNN_PBL_COLUMN_CHUNK_MINIMUM`, the width buffers walked in
    2.8.0, whatever this process's own width is: every buffer allocates
    its own workspace, and at the run's width ``nbuffers`` of them priced a
    streamed forecast above the resident one (the measurement is at
    :data:`MYNN_PBL_COLUMN_CHUNK_MINIMUM`).  A pin or an operator override
    is honoured verbatim, as it is for the run's width, so a capacity
    measurement that narrows the width still walks what it asked for.

    The width never changes a bit (see :func:`choose_mynn_column_chunk`).
    ``walking=True`` is the solver's call, recorded for the receipt;
    pricing leaves it off, so a plan that is never run writes nothing.
    """
    nz = int(nz)
    priced = _PRICING_RANK_CHUNK.get()
    if priced is not None:
        return priced
    chunk = resolve_mynn_column_chunk(nz)
    choice = _RESOLVED.get(nz)
    if _PRICING_MEMORY.get() is not None:
        # A target fit does not read the host's memo, including its
        # override provenance. The same operator override still applies.
        choice = None
        override = mynn_column_chunk_override()
        if override is not None:
            return int(chunk)
    if _PINNED is None and not (choice is not None
                                and choice.source == "override"):
        chunk = min(chunk, MYNN_PBL_COLUMN_CHUNK_MINIMUM)
    if walking:
        _TILE_WALKED[nz] = int(chunk)
    return int(chunk)


def mynn_rank_chunk_candidates(nz: int) -> tuple[int, ...]:
    """Resident-rank widths to fit before optional output snapshots.

    The resident solver's existing width cap is 98,304 columns. Fit every
    existing 4,096-column memory quantum between that cap and the minimum:
    jumping straight from 8,192 to 32,768 left fitting intermediate widths
    unused on smaller cards. Reused streamed buffers keep their separate
    minimum-width policy. Explicit pins and overrides remain exact requests
    and must pass the ordinary memory gate.
    """
    if _PINNED is not None:
        return (int(_PINNED),)
    choice = _RESOLVED.get(int(nz))
    if choice is not None and choice.source == "override":
        return (int(choice.chunk),)
    override = mynn_column_chunk_override()
    if override is not None:
        return (int(override),)
    return tuple(range(MYNN_PBL_COLUMN_CHUNK_MINIMUM,
                       MYNN_PBL_COLUMN_CHUNK_DEFAULT + 1,
                       MYNN_PBL_CHUNK_VRAM_QUANTUM))


def bind_mynn_rank_chunk(state, cfg, chunk: int) -> int:
    """Bind a rank's immutable, admitted workspace width before first use."""
    if type(chunk) is not int or chunk < 1:
        raise ValueError("a ranked MYNN workspace needs a positive integer width")
    width = min(chunk, int(cfg.nx) * int(cfg.ny))
    previous = getattr(state, "_mynn_rank_column_chunk", None)
    if previous is not None and previous != width:
        raise ValueError(
            f"ranked MYNN width is already {previous}, requested {width}; "
            "changing it after admission would invalidate scratch storage")
    # A constructor may already own scratch, so check it before publishing
    # the binding rather than wait for a differently shaped first request.
    shapes = {**mynn_pbl_scratch_shapes(width, int(cfg.nz),
                                      bl_mynn_version=cfg.bl_mynn_version),
              **mynn_pbl_index_shapes(width, int(cfg.nz))}
    for slot, shape in shapes.items():
        held = getattr(state, "_scratch", {}).get(slot)
        if held is not None and tuple(held.shape) != tuple(shape):
            raise ValueError(
                f"ranked MYNN slot {slot} already has shape {held.shape}, "
                f"admitted width {width} needs {shape}")
    state._mynn_rank_column_chunk = width
    return width


def pin_mynn_column_chunk(chunk: int | None) -> int:
    """Pin the width for this process; returns the previously published one.

    The explicit in-process handle (``tilestream.shared_workspace`` measures
    tile capacity with it).  ``None`` releases the pin and lets the next
    caller derive again.
    """
    global _PINNED

    previous = int(MYNN_PBL_COLUMN_CHUNK)
    if chunk is None:
        _PINNED = None
        _RESOLVED.clear()
        _TILE_WALKED.clear()
        return previous
    chunk = int(chunk)
    if chunk < 1:
        raise ValueError("MYNN column chunk must be a positive column count")
    _PINNED = chunk
    _RESOLVED.clear()
    _TILE_WALKED.clear()
    _publish_column_chunk(chunk)
    return previous


def observe_dmp_registers_per_thread() -> int | None:
    """Registers per thread the compiled DMP kernel actually got.

    Read for the receipt, never for the derivation: it is available only
    once the module is compiled, which is after the arena the derivation
    sized.  ``None`` when the kernel is not loadable.

    The kernel is fetched from ``mynn_pbl_gpu``, which is where every
    ``get_kernel`` call in this scheme lives.  Reaching the registry from
    THIS module would enter it into the physics allocation inventory --
    ``tests/test_physics_allocation_inventory.py`` discovers a kernel-
    launching module by its ``get_kernel`` calls -- and a workspace
    declaration is not a kernel-launching module.
    """
    try:
        from woof.core.mynn_pbl_gpu import dmp_registers_per_thread

        return dmp_registers_per_thread()
    except Exception:  # noqa: BLE001 - a receipt field, never a refusal
        return None


def mynn_column_chunk_receipt() -> dict | None:
    """The derivation this process used, or ``None`` if MYNN never ran.

    Absent rather than null when the scheme is off, so a receipt written by
    a run with no MYNN keeps the bytes 2.7.4 wrote.
    """
    if _PINNED is not None:
        entry = _chunk_choice(
            _PINNED, "pinned", next(iter(_RESOLVED), 1)).receipt()
    elif _RESOLVED:
        entry = max(_RESOLVED.values(),
                    key=lambda choice: choice.chunk).receipt()
    else:
        return None
    observed = observe_dmp_registers_per_thread()
    if observed is not None:
        entry["registers_per_thread_observed"] = observed
    if _TILE_WALKED:
        # A streamed run's buffers walk their own width
        # (:func:`resolve_mynn_tile_column_chunk`), so the receipt names it
        # rather than leave ``chunk`` standing for a width no buffer walked.
        entry["tile_buffer_chunk"] = max(_TILE_WALKED.values())
    return entry


#: The six A-grid tendency fields ``mynn_pbl_step`` returns.  Full ``(nz, ny,
#: nx)`` because ``couple_ysu_tendencies`` consumes whole fields; they were
#: six unpriced per-step allocations before, 423.4 MiB at the d04 nest.
MYNN_PBL_TENDENCY_FIELDS = ("du", "dv", "dtheta", "dqv", "dqc", "dqi")


def mynn_pbl_tendency_field_shapes(nz: int, ny: int, nx: int
                                   ) -> dict[str, tuple[int, ...]]:
    """Slot shapes for the returned A-grid tendency fields."""
    return {f"mynn_pbl_out_{name}": (int(nz), int(ny), int(nx))
            for name in MYNN_PBL_TENDENCY_FIELDS}


def mynn_pbl_slot_names() -> tuple[str, ...]:
    """Every slot name this module declares, float32 then int32 then flags."""
    return (*sorted(_LAYER_GROUPS), *sorted(_GSD41_LAYER_GROUPS),
            SLOT_GSD41_DIAGNOSTICS,
            *sorted(_FACE_GROUPS),
            *sorted(_COLUMN_GROUPS), *sorted(MYNN_PBL_INDEX_SLOTS),
            *MYNN_PBL_FLAG_SLOTS)


class MynnPblScratch:
    """Views onto the declared MYNN slots for one batch of columns.

    ``from_state`` is the runtime path: every slot comes from
    ``DomainState.scratch``, so the registry prices it and the shared arena
    can back it.  ``standalone`` is for the oracle leaves, which are called
    directly on four-column fixtures and have no domain; it allocates each
    slot once, on first use, and holds it for the life of the holder.
    """

    def __init__(self, buffers: Mapping[str, cp.ndarray],
                 *, chunk: int, nz: int, lazy: bool = False):
        self._buffers = dict(buffers)
        self._chunk = int(chunk)
        self._nz = int(nz)
        self._lazy = bool(lazy)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_state(cls, state, chunk: int, nz: int, *,
                   bl_mynn_version: str = "wrf_461") -> "MynnPblScratch":
        """Draw every declared slot from ``DomainState.scratch``."""
        import cupy as cp

        buffers = {}
        for slot, shape in mynn_pbl_scratch_shapes(
                chunk, nz, bl_mynn_version=bl_mynn_version).items():
            buffers[slot] = state.scratch(shape, slot)
        for slot, shape in mynn_pbl_index_shapes(chunk, nz).items():
            buffers[slot] = state.scratch(shape, slot, dtype=cp.int32)
        for slot, shape in mynn_pbl_flag_shapes().items():
            buffers[slot] = state.scratch(shape, slot, dtype=cp.int32)
        return cls(buffers, chunk=chunk, nz=nz)

    @classmethod
    def standalone(cls, chunk: int, nz: int) -> "MynnPblScratch":
        """A self-owned workspace for callers with no ``DomainState``.

        Slots are allocated on first request rather than up front, because
        the oracle leaves exercise one routine at a time and a four-column
        fixture has no reason to pay for the plume block.
        """
        return cls({}, chunk=chunk, nz=nz, lazy=True)

    # -- access ------------------------------------------------------------

    @property
    def chunk(self) -> int:
        return self._chunk

    @property
    def nz(self) -> int:
        return self._nz

    def _backing(self, slot: str, values: int, dtype) -> cp.ndarray:
        import cupy as cp

        buf = self._buffers.get(slot)
        if buf is None:
            if not self._lazy:
                raise KeyError(
                    f"MYNN scratch slot {slot!r} was not provided; a "
                    f"state-backed workspace declares every slot up front")
            buf = cp.zeros((values,), dtype=dtype)
            self._buffers[slot] = buf
        elif buf.size < values:
            if self._lazy:
                buf = cp.zeros((values,), dtype=dtype)
                self._buffers[slot] = buf
            else:
                raise ValueError(
                    f"MYNN scratch slot {slot!r} holds {buf.size} values, "
                    f"this call needs {values}; the declared workspace is "
                    f"{self._chunk} columns at nz={self._nz}")
        if buf.dtype != np.dtype(dtype):
            raise TypeError(f"MYNN scratch slot {slot!r} is {buf.dtype}, "
                            f"requested {np.dtype(dtype)}")
        return buf

    @staticmethod
    def _view(buf, shape):
        # Keep the public column shape while storing each level contiguously.
        # Product stacks retain their leading vector axis.
        if len(shape) >= 2:
            physical = (*shape[:-2], shape[-1], shape[-2])
            return buf.reshape(physical).swapaxes(-1, -2)
        return buf.reshape(shape)

    def group(self, slot: str, names: Iterable[str], shape) -> dict:
        """One contiguous sub-array of ``shape`` per name, in order."""
        names = tuple(names)
        shape = tuple(int(extent) for extent in shape)
        unit = prod(shape)
        buf = self._backing(slot, len(names) * unit, DTYPE)
        return {name: self._view(buf[index * unit:(index + 1) * unit], shape)
                for index, name in enumerate(names)}

    def one(self, slot: str, shape) -> cp.ndarray:
        """The first (and usually only) array in a slot."""
        shape = tuple(int(extent) for extent in shape)
        unit = prod(shape)
        return self._view(self._backing(slot, unit, DTYPE)[:unit], shape)

    def index(self, slot: str, shape) -> cp.ndarray:
        """An int32 array from an index slot."""
        import cupy as cp

        shape = tuple(int(extent) for extent in shape)
        unit = prod(shape)
        return self._backing(slot, unit, cp.int32)[:unit].reshape(shape)

    def flags(self, slot: str = MYNN_PBL_FLAG_SLOTS[0]) -> cp.ndarray:
        """The persistent int32 validity words, zeroed for this use."""
        import cupy as cp

        buf = self._backing(slot, _FLAG_WORDS, cp.int32)
        buf[...] = 0
        return buf

    # -- the write-before-read lever ---------------------------------------

    def poison(self) -> None:
        """NaN-fill every slot that a kernel is required to overwrite.

        The constant-zero feeds are deliberately skipped: nothing writes
        them and the pinned identity requires them to read as zero.  A slot
        that belongs in that set but is not listed there will surface here
        as a NaN in the forecast, which is the point.
        """
        for slot, buf in self._buffers.items():
            if slot in MYNN_PBL_CONSTANT_ZERO_SLOTS:
                continue
            if buf.dtype == np.dtype(DTYPE):
                buf.fill(DTYPE(np.nan))
            else:
                buf.fill(-2147483647)


__all__ = [
    "MYNN_PBL_COLUMN_CHUNK",
    "MYNN_PBL_COLUMN_CHUNK_ENV",
    "MYNN_PBL_COLUMN_CHUNK_FLOOR",
    "MYNN_PBL_COLUMN_CHUNK_MINIMUM",
    "MYNN_PBL_RUN_WIDTH_TOTAL_VRAM_FRACTION",
    "MYNN_PBL_COLUMN_TPB",
    "MYNN_DMP_REGISTERS_PER_THREAD",
    "MynnColumnChunk",
    "MYNN_PBL_CONSTANT_ZERO_SLOTS",
    "MYNN_PBL_FLAG_SLOTS",
    "MYNN_PBL_INDEX_SLOTS",
    "MYNN_PBL_STAGE_LAYERS",
    "MYNN_PBL_TENDENCY_FIELDS",
    "MynnPblScratch",
    "SLOT_STAGE_DX",
    "SLOT_STAGE_LAYER",
    "SLOT_ZERO_COLUMN",
    "mynn_pbl_tendency_field_shapes",
    "SLOT_CONDENSATION",
    "SLOT_DELT",
    "SLOT_DISS_HEAT",
    "SLOT_EXCHANGE",
    "SLOT_GSD41_CONDENSATION_WORK",
    "SLOT_GSD41_THVL",
    "SLOT_GSD41_DIAGNOSTICS",
    "GSD41_DIAGNOSTICS",
    "SLOT_INITIALIZE",
    "SLOT_INITIALIZE_WORK",
    "SLOT_LEVEL2_FULL",
    "SLOT_LEVEL2_OUT",
    "SLOT_LEVEL2_PAIRS",
    "SLOT_MIXLENGTH",
    "SLOT_MIXLENGTH_WORK",
    "SLOT_PBLH",
    "SLOT_PLUME_COLUMN",
    "SLOT_PLUME_FACE",
    "SLOT_PLUME_LAYER",
    "SLOT_PLUME_SCRATCH",
    "SLOT_PLUME_WORK",
    "SLOT_PLUME_ZERO_FACE",
    "SLOT_PLUME_ZERO_LAYER",
    "SLOT_PREDICT",
    "SLOT_PREDICT_WORK",
    "SLOT_PREP",
    "SLOT_SURFACE",
    "SLOT_TENDENCY",
    "SLOT_TENDENCY_FACE",
    "SLOT_TENDENCY_WORK",
    "SLOT_TENDENCY_ZERO",
    "SLOT_TURBULENCE",
    "SLOT_ZERO_FACE",
    "SLOT_ZERO_LAYER",
    "derive_mynn_column_chunk",
    "mynn_column_chunk_for_memory",
    "mynn_pricing_memory",
    "mynn_pricing_rank_chunk",
    "mynn_rank_chunk_candidates",
    "bind_mynn_rank_chunk",
    "mynn_pricing_total_bytes",
    "mynn_column_chunk_override",
    "mynn_column_chunk_receipt",
    "mynn_column_pieces",
    "mynn_pbl_card_chunk_ceiling",
    "mynn_pbl_column_bytes",
    "mynn_pbl_vram_chunk_ceiling",
    "observe_dmp_registers_per_thread",
    "pin_mynn_column_chunk",
    "mynn_pricing_resident_chunk", "mynn_target_fit_active",
    "probe_mynn_card",
    "resolve_mynn_column_chunk",
    "resolve_mynn_tile_column_chunk",
    "mynn_pbl_flag_shapes",
    "mynn_pbl_index_shapes",
    "mynn_pbl_scratch_bytes",
    "mynn_pbl_scratch_shapes",
    "mynn_pbl_slot_names",
]
