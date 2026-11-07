"""Active full-step local time stepping across the cards of one host.

The scheme is the one-device scheme of ``cuda_fullstep_lts_active`` and
``cuda_fullstep_lts_scalars`` -- fine first, every class on its own whole
step, conservative reflux, scalar packets with a per-column reserve -- with
each rate class cut into BLOCKS (``fullstep_lts_blocks``), one per card it
uses.  Every block runs the unchanged native step on its own card.

What crosses a card, and when
-----------------------------
* **Between blocks of the same class** (they step at the same time): the
  same-host byte transport, at every native round site (A, B, C, D, E, P), as
  in the same-host executor, plus three rounds of its own:

  - ``H`` twice inside the flux-corrected scalar stage: the donor-guard factor
    of every coarse interface column, then the reserve-cap factors of every
    finer one, so each copy of an edge flux is scaled by the same number;
  - ``R`` at the class's sync: the refluxed cell state, so a velocity recovered
    on an edge next to another block's refluxed cell reads its corrected
    density.

  A block's mailbox fills run first and the peer exchange second at every
  round, so a carried zone cell reset from the mailbox reaches the peers
  reset.

* **Between classes** (they step at different times): the mailbox.  Every
  card holds a full copy.  A block publishes into its own card's copy; at the
  end of the event the published planes are copied, by pinned host staging,
  to exactly the cards whose blocks read those cells.  The interface
  registers: every card holds a full copy; before a class steps, the slots its
  blocks read are copied from the cards whose finer blocks wrote them, and the
  writer's copy is cleared (it is spent there).

The classes step in the same fine-first order as on one device; within an
event every block of the class steps concurrently on its own card.  The
arithmetic of every kernel is the one-device arithmetic; the per-column
factors are split into a "factor" kernel and an "apply" kernel with the same
expressions, so a multi-card run reproduces the one-card run's bytes (the
identity check in ``tools/run_fullstep_lts_dry.py``).
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .cuda_fp32 import CUDA_FTZ_HELPERS
from .cuda_fullstep_lts_active import (
    CELL_FIELDS, EDGE_FIELDS, INTERFACE_FIELDS, INTERFACE_FLUX_MODES, MAX_CLASSES, _KERNEL_NAMES as ACTIVE_KERNELS,
    _RESET, _SOURCE as ACTIVE_SOURCE, _SPLIT_START_MASK, _WRITE, _ZERO, _Holder, _field)
from .cuda_fullstep_lts_scalars import (
    AMOUNT_NAMES, END, MID, SCALAR_PLANES, START, STAT_NAMES, _FINISH, _KERNEL_NAMES as SCALAR_KERNELS,
    _LOW_ORDER, _RESERVE_MARGIN, _SOURCE as SCALAR_SOURCE, _UntappedAudit, tap_kernel)
from .fullstep_lts_classes import (
    child_flux_intervals, lts_acoustic_substeps, macro_schedule, plan_macros, plane_weights, planes_per_step, trajectory_plane)

_MULTI_SOURCE = CUDA_FTZ_HELPERS + r"""
#define LTS_LOOP(total) \
    for (long long element = (long long)blockDim.x * blockIdx.x + threadIdx.x; \
         element < (long long)(total); element += (long long)gridDim.x * blockDim.x)

/* rows of a (row, column) array at a list of columns <-> a packed buffer */
extern "C" __global__ void lts_pack_f32(const long long rows, const int n, const long long ncol,
    const long long row0, const int *idx, const float *src, float *buf)
{
    LTS_LOOP(rows * n) {
        const long long r = element / n;
        const int i = (int)(element % n);
        buf[element] = src[(row0 + r) * ncol + idx[i]];
    }
}
extern "C" __global__ void lts_unpack_f32(const long long rows, const int n, const long long ncol,
    const long long row0, const int *idx, const float *buf, float *dst)
{
    LTS_LOOP(rows * n) {
        const long long r = element / n;
        const int i = (int)(element % n);
        dst[(row0 + r) * ncol + idx[i]] = buf[element];
    }
}
extern "C" __global__ void lts_pack_f64(const long long rows, const int n, const long long ncol,
    const long long row0, const int *idx, const double *src, double *buf)
{
    LTS_LOOP(rows * n) {
        const long long r = element / n;
        const int i = (int)(element % n);
        buf[element] = src[(row0 + r) * ncol + idx[i]];
    }
}
extern "C" __global__ void lts_unpack_f64(const long long rows, const int n, const long long ncol,
    const long long row0, const int *idx, const double *buf, double *dst)
{
    LTS_LOOP(rows * n) {
        const long long r = element / n;
        const int i = (int)(element % n);
        dst[(row0 + r) * ncol + idx[i]] = buf[element];
    }
}
extern "C" __global__ void lts_zero_f64(const long long rows, const int n, const long long ncol,
    const int *idx, double *dst)
{
    LTS_LOOP(rows * n) {
        const long long r = element / n;
        const int i = (int)(element % n);
        dst[r * ncol + idx[i]] = 0.0;
    }
}

/* lts_donor_guard's arithmetic, factor half: the column's own levels are
   adjusted here (they are the column's alone); the factor that scales its own
   outflow edges is written for every (tracer, level) of every owned row, 1
   where nothing is scaled, and applied by lts_guard_apply on every card that
   holds the column. */
extern "C" __global__ void lts_guard_keep(
    const int ntr, const int nlev, const int n, const int width, const int ncells,
    const int nedges, const int nslots, const int max_edges, const float dt,
    const int *cell, const int *slot, const double *coef, const double *reg_scalar,
    const double *reg_mass,
    const int *n_edges_on_cell, const int *edges_on_cell, const float *acoustic_sign,
    const float *inv_area_cell, const unsigned char *packet_edge,
    const float *source_old, const float *rho_old, const float *ww, const float *rdzw,
    const float *upwind, float *mass, float *vertical_residual, float *keep_out,
    unsigned long long *stats, double *amounts)
{
    LTS_LOOP((long long)ntr * n) {
        const int row = (int)(element % n);
        const long long tracer = element / n;
        const int c = cell[row];
        const double inverse_area = (double)inv_area_cell[c];
        const int count = n_edges_on_cell[c];
        for (int k = 0; k < nlev; ++k) {
            const long long tk = tracer * nlev + k;
            const long long q = tk * ncells + c;
            keep_out[q] = 1.0f;
            const double value = (double)source_old[q];
            const double content = value * (double)rho_old[(long long)k * ncells + c];
            double exported = 0.0, air_out = 0.0;
            for (int w = 0; w < width; ++w) {
                const double outward = -coef[row * width + w];
                const int s = slot[row * width + w];
                exported += outward * reg_scalar[tk * nslots + s];
                air_out += fmax(outward * reg_mass[(long long)k * nslots + s], 0.0);
            }
            const double up = (k + 1 < nlev) ? fmax((double)ww[(long long)(k + 1) * ncells + c], 0.0) : 0.0;
            const double down = (k > 0) ? fmin((double)ww[(long long)k * ncells + c], 0.0) : 0.0;
            double outflow = fmax((double)rdzw[k] * (double)dt * (up - down) * value, 0.0);
            for (int s = 0; s < count; ++s) {
                const int edge = edges_on_cell[c * max_edges + s];
                if (packet_edge[edge]) continue;
                const double inward = -(double)acoustic_sign[c * max_edges + s]
                    * (double)upwind[tk * nedges + edge] * inverse_area;
                if (inward < 0.0) outflow -= inward;
            }
            const double need = exported + outflow - content;
            if (need <= 0.0) continue;
            const double native = fmax(fmax(value, 0.0) * air_out + outflow - content, 0.0);
            const double reduce = need - native;
            if (reduce <= (native > 0.0 ? 4.0e-6 * (fabs(exported) + outflow + fabs(content)) : 0.0))
                continue;
            if (reduce > outflow) atomicAdd(&stats[1], 1ULL);
            if (outflow <= 0.0) continue;
            atomicAdd(&stats[0], 1ULL);
            const double keep = reduce < outflow ? (outflow - reduce) / outflow : 0.0;
            atomicAdd(&amounts[3 * tracer + 2], (1.0 - keep) * outflow / inverse_area);
            const float scale = (float)keep;
            keep_out[q] = scale;
            if (up > 0.0 && value > 0.0) {
                const double removed = (1.0 - keep) * (double)dt * up * value;
                mass[q] = (float)((double)mass[q] + removed * (double)rdzw[k]);
                mass[q + ncells] = (float)((double)mass[q + ncells] - removed * (double)rdzw[k + 1]);
                vertical_residual[(tracer * (nlev + 1) + k + 1) * ncells + c] *= scale;
            }
            if (down < 0.0 && value > 0.0) {
                const double removed = (1.0 - keep) * (double)dt * down * value;
                mass[q - ncells] = (float)((double)mass[q - ncells] + removed * (double)rdzw[k - 1]);
                mass[q] = (float)((double)mass[q] - removed * (double)rdzw[k]);
                vertical_residual[(tracer * (nlev + 1) + k) * ncells + c] *= scale;
            }
        }
    }
}

/* lts_donor_guard's edge half, for every row a card holds (owned or halo):
   the row's own outflow through its non-packet edges is scaled by its factor.
   Each flux has one donor, so one thread writes it. */
extern "C" __global__ void lts_guard_apply(
    const int ntr, const int nlev, const int n, const int ncells, const int nedges,
    const int max_edges, const int *cell, const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const unsigned char *packet_edge, const float *keep,
    float *upwind, float *residual)
{
    LTS_LOOP((long long)ntr * nlev * n) {
        const int row = (int)(element % n);
        const long long tk = element / n;
        const int c = cell[row];
        const float scale = keep[tk * ncells + c];
        if (scale == 1.0f) continue;
        const int count = n_edges_on_cell[c];
        for (int s = 0; s < count; ++s) {
            const int edge = edges_on_cell[c * max_edges + s];
            if (packet_edge[edge]) continue;
            const long long qe = tk * nedges + edge;
            if (-(double)acoustic_sign[c * max_edges + s] * (double)upwind[qe] < 0.0) {
                upwind[qe] *= scale;
                residual[qe] *= scale;
            }
        }
    }
}

/* lts_reserve_cap's arithmetic, factor half, for the owned rows. */
extern "C" __global__ void lts_cap_keep(
    const int ntr, const int nlev, const int n, const int width, const int nedges,
    const int nslots, const int ncells, const int *cell, const int *slot, const double *coef,
    const int *edge_of_slot, const int enforce, const double margin, const double *reserve,
    const double *content_start, const double *air_start, const double *reg_mass,
    const float *upwind, const float *residual, double *keep_low_out, double *keep_high_out,
    unsigned long long *stats, double *amounts)
{
    LTS_LOOP((long long)ntr * nlev * n) {
        const int row = (int)(element % n);
        const long long tk = element / n;
        const long long k = tk % nlev;
        double low = 0.0, high = 0.0, air_out = 0.0;
        for (int w = 0; w < width; ++w) {
            const double c = coef[row * width + w];
            if (c == 0.0) continue;
            const double outward = c < 0.0 ? 1.0 : -1.0;
            const int s = slot[row * width + w];
            const long long qe = tk * nedges + edge_of_slot[s];
            low += fmax(outward * (double)upwind[qe], 0.0);
            high += fmax(outward * (double)residual[qe], 0.0);
            air_out += fmax(outward * reg_mass[k * nslots + s], 0.0);
        }
        double available = 1.0e300;
        if (enforce) {
            const double left = fmax(reserve[tk * n + row], 0.0);
            const double air = air_start[k * n + row];
            if (air_out > air && air > 0.0)
                available = (left + fmax(content_start[tk * n + row], 0.0) * (air_out / air - 1.0))
                    * (1.0 + 4.0e-6);
            else
                available = left * margin;
        }
        double keep_low = 1.0, keep_high = 1.0;
        if (low > available) {
            keep_low = available / low;
            keep_high = 0.0;
            atomicAdd(&stats[2], 1ULL);
        } else if (low + high > available) {
            keep_high = (available - low) / high;
            atomicAdd(&stats[3], 1ULL);
        }
        if (keep_high < 1.0)
            atomicAdd(&amounts[3 * (tk / nlev) + 1],
                      (1.0 - keep_low) * low + (1.0 - keep_high) * high);
        keep_low_out[tk * ncells + cell[row]] = keep_low;
        keep_high_out[tk * ncells + cell[row]] = keep_high;
    }
}

/* lts_reserve_cap's edge half, for every row a card holds. */
extern "C" __global__ void lts_cap_apply(
    const int ntr, const int nlev, const int n, const int width, const int nedges,
    const int ncells, const int *cell, const int *slot, const double *coef, const int *edge_of_slot,
    const double *keep_low_in, const double *keep_high_in, float *upwind, float *residual)
{
    LTS_LOOP((long long)ntr * nlev * n) {
        const int row = (int)(element % n);
        const long long tk = element / n;
        const double keep_low = keep_low_in[tk * ncells + cell[row]];
        const double keep_high = keep_high_in[tk * ncells + cell[row]];
        if (keep_low >= 1.0 && keep_high >= 1.0) continue;
        for (int w = 0; w < width; ++w) {
            const double c = coef[row * width + w];
            if (c == 0.0) continue;
            const double outward = c < 0.0 ? 1.0 : -1.0;
            const int s = slot[row * width + w];
            const int e = edge_of_slot[s];
            if (e < 0) continue;
            const long long qe = tk * nedges + e;
            if (keep_low < 1.0 && outward * (double)upwind[qe] > 0.0)
                upwind[qe] = (float)((double)upwind[qe] * keep_low);
            if (keep_high < 1.0 && outward * (double)residual[qe] > 0.0)
                residual[qe] = (float)((double)residual[qe] * keep_high);
        }
    }
}

/* The capture's low-order part of every slot of the owned rows, after both factors. */
extern "C" __global__ void lts_hold(
    const int ntr, const int nlev, const int n, const int width, const int nedges,
    const int nslots, const int *slot, const double *coef, const int *edge_of_slot,
    const float *upwind, float *hold)
{
    LTS_LOOP((long long)ntr * nlev * n) {
        const int row = (int)(element % n);
        const long long tk = element / n;
        for (int w = 0; w < width; ++w) {
            if (coef[row * width + w] == 0.0) continue;
            const int s = slot[row * width + w];
            hold[tk * nslots + s] = upwind[tk * nedges + edge_of_slot[s]];
        }
    }
}

extern "C" __global__ void lts_fill_f64(const long long total, const double value, double *dst)
{
    LTS_LOOP(total) dst[element] = value;
}
"""

_MULTI_KERNELS = ("lts_pack_f32", "lts_unpack_f32", "lts_pack_f64", "lts_unpack_f64", "lts_zero_f64",
                  "lts_guard_keep", "lts_guard_apply", "lts_cap_keep", "lts_cap_apply", "lts_hold",
                  "lts_fill_f64")


def _launch(kernel: Any, total: int, args: tuple[Any, ...]) -> None:
    if total > 0:
        kernel(((int(total) + 255) // 256,), (256,), args)


def _upload_spec(cp: Any, spec: np.ndarray, weight: np.ndarray) -> tuple[Any, Any]:
    """A fill spec and its weights on the device, complete before any stream reads them.

    BREAKAGE WITHOUT THE WAIT (2026-10-26, 3-zone multi-zoom globe, WOOF route, two LTS blocks on one
    card, ``--samehost-devices=0,0``): the spec cache is per CARD and keyed by class, so the two blocks of
    one class on one card share it.  The block that builds an entry queues its host-to-device copy on its
    own rank stream; the other block, on another stream, found the entry and launched its fill before the
    copy landed, and the forecast stopped with cudaErrorIllegalAddress in the first steps (round B).  With
    ``CUDA_LAUNCH_BLOCKING=1`` the same run went 96 steps clean.  One block per card never shares an entry
    across streams, so it never crashed.  Built once per key, so the wait costs nothing per step.
    """
    arrays = (cp.asarray(spec), cp.asarray(weight))
    cp.cuda.get_current_stream().synchronize()
    return arrays


class _TransportGroup:
    """Abort/close every class transport together (the rank pool's failure contract)."""

    def __init__(self, transports: Sequence[Any]) -> None:
        self.transports = [t for t in transports if t is not None]

    def abort(self, reason: Any) -> None:
        for transport in self.transports:
            try:
                transport.abort(reason)
            except Exception:
                pass

    def close(self) -> None:
        for transport in self.transports:
            transport.close()

    def receipt(self) -> list[Any]:
        return [transport.receipt() for transport in self.transports]


class _Lane:
    """One card: its mailbox copy, register copy, kernels and coordinator stream."""

    def __init__(self, owner: "FullStepLtsMulti", device: int, kernel_cache: Any) -> None:
        cp = owner.cp
        self.device = int(device)
        self.owner = owner
        plan = owner.plan
        nlev = owner.nlev
        with cp.cuda.Device(self.device):
            self.k = {}
            for name in ACTIVE_KERNELS:
                self.k[name] = kernel_cache.raw_kernel(name, ACTIVE_SOURCE, module_key="hexcore.cuda_fullstep_lts_active")
            for name in SCALAR_KERNELS:
                self.k[name] = kernel_cache.raw_kernel(name, SCALAR_SOURCE, module_key="hexcore.cuda_fullstep_lts_scalars")
            for name in _MULTI_KERNELS:
                self.k[name] = kernel_cache.raw_kernel(name, _MULTI_SOURCE, module_key="hexcore.cuda_fullstep_lts_multi")
            planes = owner.truth_plane + 1
            f32 = cp.float32
            self.G_cell = cp.zeros((planes, len(CELL_FIELDS), nlev, max(plan.n_mail_cells, 1)), dtype=f32)
            self.G_face = cp.zeros((planes, len(INTERFACE_FIELDS), nlev + 1, max(plan.n_mail_cells, 1)), dtype=f32)
            self.G_edge = cp.zeros((planes, len(EDGE_FIELDS), nlev, max(plan.n_mail_edges, 1)), dtype=f32)
            slots = owner.nslots
            self.reg_mass_fine = cp.zeros((nlev, slots), dtype=cp.float64)
            self.reg_mass_coarse = cp.zeros_like(self.reg_mass_fine)
            self.reg_theta_fine = cp.zeros_like(self.reg_mass_fine)
            self.reg_theta_coarse = cp.zeros_like(self.reg_mass_fine)
            # Matched interface flux (cuda_fullstep_lts_active): the finer class's accepted mean momentum
            # and heat per register edge and level over each plane of the coarser class's step.  Same slot
            # space as the registers, so the register pull carries it to the coarse block's card.
            self.flux_record = self.theta_record = None
            if owner.interface_flux == "matched":
                self.flux_record = cp.zeros((owner.planes_per_step, nlev, slots), dtype=cp.float64)
                self.theta_record = cp.zeros_like(self.flux_record)
            self.G_s = self.reg_scalar = self.hold = None
            if owner.ntr:
                self.G_s = cp.zeros((owner.scalar_truth_plane + 1, owner.ntr, nlev, max(plan.n_mail_cells, 1)), dtype=f32)
                self.reg_scalar = cp.zeros((owner.ntr, nlev, slots), dtype=cp.float64)
                self.hold = cp.zeros((owner.ntr, nlev, slots), dtype=cp.float32)
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.class_pub_cells = [cp.asarray(v) for v in owner.bplan.class_pub_cells]
            self.class_pub_edges = [cp.asarray(v) for v in owner.bplan.class_pub_edges]
        self.spec_cache: dict[Any, tuple[Any, Any]] = {}
        self.dirty: list[tuple[int, str, int]] = []
        self.index_cache: dict[Any, Any] = {}

    def i32(self, key: Any, values: np.ndarray) -> Any:
        """A cached device index array, complete before any stream reads it.

        BREAKAGE WITHOUT THE WAIT (2026-10-24, 240/60 km fixed region, 2 cards): the coordinator
        builds these on first use, right before ``_move_many`` packs with them on this card's
        non-blocking stream.  ``cp.asarray`` queues its host-to-device copy on the coordinator's
        legacy default stream, which a non-blocking stream does not wait for, so the first pack
        could read the indices before they landed: 3 of 4 two-card LTS runs stopped with an
        illegal address in the first class steps, and 1 refused a step on the validation flag.
        One card never builds them (no routes, no pulls), so it never crashed.
        """
        cached = self.index_cache.get(key)
        if cached is None:
            cp = self.owner.cp
            with cp.cuda.Device(self.device), self.stream:
                cached = cp.asarray(np.ascontiguousarray(values, dtype=np.int32))
                self.stream.synchronize()
            self.index_cache[key] = cached
        return cached

    def run(self, name: str, total: int, args: tuple[Any, ...]) -> None:
        _launch(self.k[name], total, args)


class _BlockExchanger:
    """A block's halo hook: the one-device mailbox fills, then the same-class peer exchange."""

    def __init__(self, owner: "FullStepLtsMulti", block: "_Block") -> None:
        self.owner = owner
        self.block = block
        self.rounds = {name: 0 for name in "ABCDEPHRT"}
        self._schedule = None
        self._split = self._stage = self._sub = self._transport = 0
        self._direct: tuple[bool, ...] = ()
        self._reset: tuple[bool, ...] = ()
        self._event = 0
        self._position = 0
        self.peer_seconds = 0.0

    def begin(self, schedule: Any, direct: Sequence[bool], reset: Sequence[bool], event: int) -> None:
        self._schedule = schedule
        self._split = self._stage = self._sub = self._transport = 0
        self._direct = tuple(direct)
        self._reset = tuple(reset)
        self.begin_event(event)

    def begin_event(self, event: int) -> None:
        self._event = int(event)
        self._position = 0

    @property
    def split(self) -> int:
        return self._split

    # -- peer exchange -----------------------------------------------------------
    def peer(self, letter: str, fields: Sequence[tuple[str, Any, str]]) -> None:
        block = self.block
        self.rounds[letter] += 1
        position = self._position
        self._position += 1
        transport = block.transport
        if transport is None:
            return
        t = block.tables
        trimmed = []
        for name, array, kind in fields:
            extent = t.n_xlocal_cells if kind == "cell" else t.n_xlocal_edges
            trimmed.append((name, array[..., :extent], kind))
        began = time.perf_counter()
        transport.exchange(t.block, f"{self._event}:{position}:{letter}", trimmed)
        self.peer_seconds += time.perf_counter() - began

    @staticmethod
    def _state_fields(state: Any, saved: Any, *, scalars: bool) -> list[tuple[str, Any, str]]:
        fields = [("rho", state.rho, "cell"), ("rho_theta", state.rho_theta, "cell"),
                  ("rho_u", state.rho_u, "edge"), ("rho_w", state.rho_w, "cell")]
        if scalars and int(state.scalars.shape[0]) > 0:
            fields.append(("scalars", state.scalars, "cell"))
        fields += [("theta_m", saved.theta_m, "cell"), ("exner", saved.exner, "cell"),
                   ("density_perturbation", saved.density_perturbation, "cell"),
                   ("rho_theta_perturbation", saved.rho_theta_perturbation, "cell"),
                   ("pressure_perturbation", saved.pressure_perturbation, "cell"),
                   ("normal_velocity", saved.normal_velocity, "edge"),
                   ("vertical_velocity", saved.vertical_velocity, "cell")]
        return fields

    # -- specs (the one-device law, cached per card) -------------------------------
    def _spec(self, kind: str, plane: int) -> tuple[Any, Any]:
        owner, lane = self.owner, self.block.lane
        me = self.block.index
        substeps = owner.acoustic_substeps
        length = self.block.step_length
        effective = tuple(owner.effective)
        key = (me, kind, plane, substeps * self._split, self._direct, self._reset, length, effective)
        cached = lane.spec_cache.get(key)
        if cached is not None:
            return cached
        spec = np.zeros(4 * MAX_CLASSES, dtype=np.int32)
        weight = np.zeros(2 * MAX_CLASSES, dtype=np.float32)
        weight[0::2] = 1.0
        for other in range(owner.n_classes):
            if other == me:
                continue
            row = spec[4 * other:4 * other + 4]
            if other < me:
                row[1:] = owner.truth_plane
                if kind == "A":
                    row[0] = _RESET if self._reset[other] else _WRITE
                else:
                    row[0] = _WRITE if kind in ("B", "P") else _ZERO
                continue
            distance = other - me
            if distance > owner.plan.level_depth:
                continue
            base = (distance - 1) * owner.planes_per_level
            row[0] = _WRITE
            row[3] = base + substeps * self._split
            a, b, wa, wb = plane_weights(plane, length, effective[other], substeps)
            row[1], row[2] = base + a, base + b
            weight[2 * other], weight[2 * other + 1] = wa, wb
        cached = lane.spec_cache[key] = _upload_spec(owner.cp, spec, weight)
        return cached

    def _scalar_spec(self, kind: str, phase: int) -> tuple[Any, Any]:
        owner, lane = self.owner, self.block.lane
        me = self.block.index
        key = (me, "scalar", kind, phase, self._direct, self._reset)
        cached = lane.spec_cache.get(key)
        if cached is not None:
            return cached
        spec = np.zeros(4 * MAX_CLASSES, dtype=np.int32)
        weight = np.zeros(2 * MAX_CLASSES, dtype=np.float32)
        weight[0::2] = 1.0
        for other in range(owner.n_classes):
            if other == me:
                continue
            row = spec[4 * other:4 * other + 4]
            if other < me:
                row[1:] = owner.scalar_truth_plane
                row[0] = _RESET if (kind == "A" and self._reset[other]) else _WRITE
                continue
            distance = other - me
            if distance > owner.plan.level_depth:
                continue
            base = (distance - 1) * SCALAR_PLANES
            row[0] = _WRITE
            if phase != MID or self._direct[other]:
                row[1] = row[2] = base + phase
            else:
                row[1], row[2] = base + START, base + END
                weight[2 * other] = weight[2 * other + 1] = 0.5
        cached = lane.spec_cache[key] = _upload_spec(owner.cp, spec, weight)
        return cached

    # -- the native round sites ---------------------------------------------------
    def round_step_boundary(self, state: Any, saved: Any, dynamics_tendencies: Any = None) -> None:
        """A: the mailbox fills, then the same-class peers' owner truth.

        Physics runs on every local column of a block, so the GF/new-Tiedtke
        advective forcing it reads (rthdynten, rqvdynten, formed from the
        block's previous step over all its local cells) also crosses at A:
        the outer halo rows' tendencies have incomplete stencils, and the
        same defect in the same-host ranks moved a 2-card 30 km globe off
        the 1-card run at step 2, at the partition seam (e03fa99a).
        """
        owner, block = self.owner, self.block
        owner._fill_state(block, state, saved, *self._spec("A", 0), 0)
        if owner.ntr and int(state.scalars.shape[0]) > 0:
            owner._fill_scalars(block, state.scalars, *self._scalar_spec("A", START))
        fields = self._state_fields(state, saved, scalars=True)
        if dynamics_tendencies is not None:
            # rqvdynten is formed only for GF and New Tiedtke on the MPAS-physics route (runtime_v842.prepare_transport)
            fields += [(name, getattr(dynamics_tendencies, name), "cell") for name in ("rthdynten", "rqvdynten")
                       if getattr(dynamics_tendencies, name, None) is not None]
        self.peer("A", fields)

    def round_stage_entry(self, state: Any, saved: Any) -> None:
        stage = self._schedule.stages[self._stage]
        final = stage.stage == 3
        if self._sub != stage.acoustic_steps:
            raise RuntimeError(
                f"stage {stage.stage} recovered after {self._sub} acoustic rounds, not its "
                f"{stage.acoustic_steps}; the halo would be filled at a time the stage did not reach")
        plane = self.owner.acoustic_substeps * self._split + stage.acoustic_steps
        self.owner._fill_state(self.block, state, saved, *self._spec("B", plane),
                               0 if final else _SPLIT_START_MASK)
        self.peer("B", self._state_fields(state, saved, scalars=False))
        self._sub = 0
        self._stage += 1
        if final:
            self._stage = 0
            self._split += 1

    def round_acoustic(self, acoustic: Any) -> None:
        self._sub += 1
        limit = self._schedule.stages[self._stage].acoustic_steps
        if self._sub > limit:
            raise RuntimeError(
                f"acoustic round {self._sub} in a stage of {limit} sub-steps; the interface "
                "schedule would read a plane past the stage's end")
        self.owner._fill_perturbation(
            self.block, acoustic, *self._spec("C", self.owner.acoustic_substeps * self._split + self._sub))
        self.peer("C", [("rho_pp", acoustic.rho_pp, "cell"), ("rtheta_pp", acoustic.rtheta_pp, "cell"),
                        ("ru_p", acoustic.ru_p, "edge")])

    def round_transport(self, scalar_stage: Any) -> None:
        if not self.owner.ntr:
            raise RuntimeError(
                "a block transported scalars but the interface was built without them; the "
                "scalar would cross a rate interface with two different fluxes")
        entry = self._transport
        self._transport += 1
        if entry > 3:
            raise RuntimeError("the scalar interface is specified for three transport stages per step")
        if entry > 0:
            self.owner._fill_scalars(self.block, scalar_stage, *self._scalar_spec("D", END if entry == 3 else MID))
        self.peer("D", [("scalars", scalar_stage, "cell")])

    def round_fct_scale(self, scale_in: Any, scale_out: Any) -> None:
        self.peer("E", [("scale_in", scale_in, "cell"), ("scale_out", scale_out, "cell")])

    def round_physics_candidate(self, atmosphere: Any) -> None:
        if not self.finished():
            raise RuntimeError("the physics halo round was asked for inside the dynamics schedule; "
                               "the halo would be filled at a time no class accepted")
        # The step end is plane 3N (planes_per_step), as in the one-device executor.  A literal 6
        # stood here from the N = 2 clock: with MPAS-A's default N = 4 it read a finer class's
        # mid-step state into the halo the microphysics validates, so a one-card run (one-device
        # executor) and a multi-card run parted at the first coarse step (2026-10-24, 240/60 km
        # fixed region: 0.78 Pa RMS surface pressure at 1 h).
        self.owner._fill_state(self.block, atmosphere.state, atmosphere.saved,
                               *self._spec("P", self.owner.planes_per_step), 0)
        if self.owner.ntr:
            self.owner._fill_scalars(self.block, atmosphere.state.scalars, *self._scalar_spec("D", END))
        self.peer("P", self._state_fields(atmosphere.state, atmosphere.saved, scalars=True))

    def round_phase_one(self, raw: Any) -> None:
        """T: the owners' phase-one column rates on the peer halo, before they are coupled to edges."""

        fields = [("du", raw.du, "cell"), ("dv", raw.dv, "cell"), ("dtheta", raw.dtheta, "cell")]
        fields += [("d_" + name, array, "cell") for name, array in sorted(raw.dscalars.items())]
        self.peer("T", fields)

    def finished(self) -> bool:
        return self._split == self._schedule.dynamics_splits and self._stage == 0


class _Block:
    """One block: its driver, its card, device index tables, hooks and reserve."""

    def __init__(self, owner: "FullStepLtsMulti", tables: Any, driver: Any, lane: _Lane, transport: Any,
                 area_local: np.ndarray) -> None:
        cp = owner.cp
        self.tables = tables
        self.driver = driver
        self.lane = lane
        self.transport = transport
        self.index = int(tables.index)
        self.rate = int(tables.rate)
        self.gid = int(tables.gid)
        self.device = int(tables.device)
        self.rank = int(tables.rank)
        layout = tables.layout
        self.n_owned_cells = int(layout.n_owned_cells)
        self.n_owned_edges = int(layout.n_owned_edges)
        self.n_local_cells = int(layout.n_local_cells)
        self.n_local_edges = int(layout.n_local_edges)
        with cp.cuda.Device(self.device):
            def i32(values: np.ndarray) -> Any:
                return cp.asarray(np.ascontiguousarray(values, dtype=np.int32))

            self.halo_cell_mail = i32(tables.halo_cell_mail)
            self.halo_cell_class = cp.asarray(np.ascontiguousarray(tables.halo_cell_class, dtype=np.uint8))
            self.halo_edge_mail = i32(tables.halo_edge_mail)
            self.halo_edge_class = cp.asarray(np.ascontiguousarray(tables.halo_edge_class, dtype=np.uint8))
            self.pub_cell_local = i32(tables.pub_cell_local)
            self.pub_cell_mail = i32(tables.pub_cell_mail)
            self.pub_edge_local = i32(tables.pub_edge_local)
            self.pub_edge_mail = i32(tables.pub_edge_mail)
            self.fine_edge = i32(tables.fine_edge_local)
            self.fine_slot = i32(tables.fine_slot)
            self.fine_cells = i32(tables.fine_cells.reshape(-1))
            self.up_edge = i32(tables.up_edge_local)
            self.up_cells = i32(tables.up_edge_cells.reshape(-1))
            self.coarse_edge = i32(tables.coarse_edge_local)
            self.coarse_slot = i32(tables.coarse_slot)
            self.coarse_cells = i32(tables.coarse_cells.reshape(-1))
            coarse_local = np.asarray(tables.coarse_edge_local, dtype=np.int64)
            if np.any(coarse_local < self.n_owned_edges):
                raise AssertionError("a coarse-side register edge is owned here; the finer class owns every interface edge")
            self.coarse_edge_mail = i32(np.asarray(tables.halo_edge_mail)[coarse_local - self.n_owned_edges])
            self.reflux_cell = i32(tables.reflux_cell_local)
            self.reflux_slot = i32(tables.reflux_slot.reshape(-1))
            self.reflux_coef = cp.asarray(np.ascontiguousarray(tables.reflux_coef.reshape(-1), dtype=np.float64))
            self.reflux_width = int(tables.reflux_slot.shape[1])
            self.reflux_edge = i32(tables.reflux_edge_local)
            self.reflux_edge_cells = i32(tables.reflux_edge_cells.reshape(-1))
            self.packet_edge_list = i32(tables.packet_edge_local)
            self.packet_slot = i32(tables.packet_slot)
            packet_mask = np.zeros(max(self.n_local_edges, 1), dtype=np.uint8)
            packet_mask[np.asarray(tables.packet_edge_local, dtype=np.int64)] = 1
            self.packet_mask = cp.asarray(packet_mask)
            self.guard_cell = i32(tables.guard_cell_local)
            self.n_guard_owned = int(tables.n_guard_owned)
            self.cap_cell = i32(tables.cap_cell_local)
            self.n_cap_owned = int(tables.n_cap_owned)
            self.cap_slot = i32(tables.cap_slot.reshape(-1))
            self.cap_coef = cp.asarray(np.ascontiguousarray(tables.cap_coef.reshape(-1), dtype=np.float64))
            self.cap_width = int(tables.cap_slot.shape[1])
            self.cap_area = cp.asarray(np.ascontiguousarray(tables.cap_area, dtype=np.float64))
            self.edge_of_slot = i32(tables.edge_of_slot)
            owned_area = cp.asarray(np.asarray(area_local, dtype=np.float64)[:self.n_owned_cells])
            self.mass_weight = owner._dzw[self.device][:, None] * owned_area[None, :]
            ntr, nlev = owner.ntr, owner.nlev
            self.keep_guard = self.keep_low = self.keep_high = None
            self.reserve = self.content_start = self.air_start = None
            if ntr:
                self.keep_guard = cp.ones((ntr, nlev, self.n_local_cells), dtype=cp.float32)
                self.keep_low = cp.ones((ntr, nlev, self.n_local_cells), dtype=cp.float64)
                self.keep_high = cp.ones((ntr, nlev, self.n_local_cells), dtype=cp.float64)
                rows = max(self.n_cap_owned, 1)
                self.reserve = cp.zeros((ntr, nlev, rows), dtype=cp.float64)
                self.content_start = cp.zeros_like(self.reserve)
                self.air_start = cp.zeros((nlev, rows), dtype=cp.float64)
                self.stats = cp.zeros(len(STAT_NAMES), dtype=cp.uint64)
                self.amounts = cp.zeros(3 * ntr, dtype=cp.float64)
        # the row stride of the driver's device arrays (native's one garbage element may follow the local entities)
        self.cell_stride = int(driver.atmosphere.state.rho.shape[-1])
        self.edge_stride = int(driver.atmosphere.state.rho_u.shape[-1])
        if (self.cell_stride - self.n_local_cells, self.edge_stride - self.n_local_edges) not in ((0, 0), (1, 1)):
            raise ValueError(
                f"block {self.gid}: device arrays are {self.cell_stride} cells x {self.edge_stride} edges wide "
                f"for a {self.n_local_cells} x {self.n_local_edges} local mesh")
        self.flux_kept: Any | None = None
        self.flux_imposed = 0
        self.theta_imposed = 0
        near = np.zeros(self.n_owned_cells, dtype=bool)
        near[tables.pub_cell_local] = True
        self.near_interface = near
        self.exchanger: _BlockExchanger | None = None
        self.theta_flux3: Any | None = None
        self.split_seconds = 0.0
        self.step_start_tick = 0
        self.step_length = 0
        self.configs: dict[float, tuple[Any, Any, str, Any]] = {}
        self.steps = 0
        self.expect_mass_flux = False
        self.step_seconds = 0.0
        self.vertical: tuple[Any, ...] | None = None
        self.audit: Any | None = None


class FullStepLtsMulti:
    """Fine-first conservative subcycling over rate classes cut into blocks on several cards."""

    def __init__(
        self,
        bplan: Any,
        drivers: Sequence[Any],
        *,
        base_dt: float,
        kernel_caches: Mapping[int, Any],
        area_cell: Sequence[np.ndarray],
        dzw: np.ndarray,
        clamp_audit: bool = False,
        copy_policy: str = "auto",
        interface_flux: str = "matched",
    ) -> None:
        import cupy as cp

        from .partition_samehost_transport_v841 import SameHostTransport
        from .partition_samehost_v841 import SameHostRanks

        self.cp = cp
        self.bplan = bplan
        plan = bplan.plan
        if interface_flux not in INTERFACE_FLUX_MODES:
            raise ValueError(f"interface_flux must be one of {INTERFACE_FLUX_MODES}")
        # Matched interface flux, as the one-card executor (f3e284e, d5b1024).  BREAKAGE WITHOUT IT
        # (2026-10-13): the 3 km Kalmaegi start member, 2 cards, 1 h, reached max|w| 47 m/s by 450 s
        # and the validation flag refused the step, on every block layout, while one card with the matched
        # flux ran clean; one card with interface_flux="own" failed the same way (47.10 m/s at 450 s).
        self.interface_flux = interface_flux
        self.match_theta = True
        self.plan = plan
        self.n_classes = len(plan.classes)
        if len(drivers) != len(bplan.blocks):
            raise ValueError("one driver per block")
        if self.n_classes > MAX_CLASSES:
            raise ValueError("at most eight classes")
        self.devices = tuple(bplan.devices)
        self.base_dt = float(base_dt)
        self.reflux_scale = (1.0, 1.0)
        self.start_seconds = float(drivers[0].atmosphere.state.time_seconds)
        self.tick = 0
        self.nlev = int(drivers[0].nlev)
        config = drivers[0].config
        transports_scalars = (bool(config.config_scalar_advection)
                              and int(drivers[0].atmosphere.state.scalars.shape[0]) > 0)
        if transports_scalars and not (bool(config.config_monotonic) or bool(config.config_positive_definite)):
            raise ValueError(
                "active full-step LTS moves scalars across a rate interface inside the "
                "flux-corrected third transport stage, where one accepted packet per edge "
                "enters both cells; with the limiter off that stage does not run, the two "
                "classes would each apply their own flux to the edge and tracer mass would "
                "not be conserved; run with config_monotonic=True")
        if (int(config.config_time_integration_order), int(config.config_dynamics_split_steps)) != (2, 3):
            raise ValueError(
                "this full-step interface is specified for RK2 with three dynamics splits: its planes "
                "are the stage and acoustic times of that schedule, so another schedule would read "
                "halo states at times no class computed")
        # ticks of base_dt / (3 N); a class of rate r steps 3 N r ticks
        self.acoustic_substeps = lts_acoustic_substeps(config.config_number_of_sub_steps)
        self.planes_per_step = planes_per_step(self.acoustic_substeps)
        self.planes_per_level = self.planes_per_step + 1
        self.ntr = int(drivers[0].atmosphere.state.scalars.shape[0]) if transports_scalars else 0
        self.nslots = max(int(plan.n_register_edges), 1)
        self.truth_plane = self.planes_per_level * plan.level_depth
        self.scalar_truth_plane = SCALAR_PLANES * plan.level_depth
        # host bookkeeping: the consumer step start tick each interval slot was written for
        self._record_written: list[list[int | None]] = [[None] * self.planes_per_step
                                                         for _ in range(len(plan.classes))]
        self.packets = True
        self.guard = True
        self.cap = True
        self.cards = tuple(sorted(set(self.devices)))
        self._dzw = {}
        for device in self.cards:
            with cp.cuda.Device(device):
                self._dzw[device] = cp.asarray(np.asarray(dzw, dtype=np.float64))
        self.lanes = {device: _Lane(self, device, kernel_caches[device]) for device in self.cards}
        # the first rank of each card rotates that card's mailbox planes
        self.lane_rank = {}
        for rank, device in enumerate(self.devices):
            self.lane_rank.setdefault(device, rank)
        # one transport per class that has more than one block
        self.transports: list[Any] = []
        for c in range(self.n_classes):
            tables = bplan.transport[c]
            if tables is None:
                self.transports.append(None)
                continue
            devices = [bplan.blocks[g].device for g in bplan.by_class[c]]
            self.transports.append(SameHostTransport(tables, devices, xp=cp, copy_policy=copy_policy, timeout=300))
        self.transport_group = _TransportGroup(self.transports)
        self.blocks: list[_Block] = []
        for tables, driver, area in zip(bplan.blocks, drivers, area_cell):
            lane = self.lanes[int(tables.device)]
            block = _Block(self, tables, driver, lane, self.transports[int(tables.index)], area)
            block.exchanger = _BlockExchanger(self, block)
            driver.halo_exchanger_v841 = block.exchanger
            # LTS captures the third-stage correction in the edge flux, as the
            # one-card executor does. The ordinary dycore path stays unchanged.
            driver.lts_theta_correction_in_edge_flux = True
            self._attach_capture(block)
            if self.interface_flux == "matched" and int(block.coarse_edge.size):
                self._attach_matched_flux(block)
            self.blocks.append(block)
        self.class_blocks = [[self.blocks[g] for g in members] for members in bplan.by_class]
        self.ranks = SameHostRanks(self.devices, self.transport_group, cp=cp)
        self.class_tick = [0] * self.n_classes
        self.effective = [int(r) for r in plan.rates]
        self.events = {"step": 0, "sync": 0}
        self.event_counter = 0
        self.wall = {"step": [0.0] * self.n_classes, "sync": [0.0] * self.n_classes,
                     "route": 0.0, "pull": 0.0}
        # the same, counted only after the first ``warm_after_macros`` macros (compilation excluded)
        self.warm_after_macros = 5
        self.macros_done = 0
        self.wall_warm = {"step": [0.0] * self.n_classes, "sync": [0.0] * self.n_classes,
                          "step_events": [0] * self.n_classes, "base_steps": 0}
        self.route_bytes = 0
        self.pull_bytes = 0
        self.time_classes = False
        self.stepper: Callable[[_Block], None] | None = None
        #: The block's held cumulus forcing for its next step (set with ``stepper``), carried at round A.
        self.forcing_of: Callable[[_Block], Any] | None = None
        # a context manager factory around every class step event: (class index, seconds)
        self.around_class_step: Callable[[int, float], Any] = lambda index, seconds: nullcontext()
        self._staging: dict[tuple[int, int], dict[str, Any]] = {}
        if self.ntr:
            self._attach_scalars(clamp_audit)
        # Everything above was built on this thread's legacy default stream (tables, mailboxes,
        # registers, spec arrays); the rank and lane streams are non-blocking and never wait for
        # it, so every card finishes that work before the first rank kernel reads it (the same
        # race as _Lane.i32).
        for device in self.cards:
            with cp.cuda.Device(device):
                cp.cuda.Device(device).synchronize()
        # initial publication: truth plane, sixth 0 of every level group, scalars likewise
        def publish_initial(rank: int) -> None:
            for block in self.blocks:
                if block.rank != rank:
                    continue
                atmosphere = block.driver.atmosphere
                self._publish(block, atmosphere, self.truth_plane)
                for level in range(plan.level_depth):
                    self._publish(block, atmosphere, level * self.planes_per_level)
                if self.ntr:
                    self._publish_scalars(block, atmosphere.state.scalars, self.scalar_truth_plane)
                    for level in range(plan.level_depth):
                        self._publish_scalars(block, atmosphere.state.scalars, level * SCALAR_PLANES + START)
        self.ranks.call(publish_initial)
        self._route()

    @property
    def time_seconds(self) -> float:
        return self.start_seconds + self.tick * self.base_dt / self.planes_per_step

    def close(self) -> None:
        self.ranks.close()

    # -- fills and publication (rank threads) ------------------------------------------
    def _fill_state(self, block: _Block, state: Any, saved: Any, spec: Any, weight: Any, start_mask: int) -> None:
        lane = block.lane
        i32 = np.int32
        nlev = self.nlev
        cells = block.n_local_cells - block.n_owned_cells
        edges = block.n_local_edges - block.n_owned_edges
        holder = _Holder(state, saved)
        lane.run("lts_fill7", nlev * cells, (
            i32(nlev), i32(cells), i32(block.n_local_cells), i32(lane.G_cell.shape[3]),
            i32(block.n_owned_cells), block.halo_cell_mail, block.halo_cell_class, spec, weight,
            i32(start_mask), lane.G_cell, *(_field(holder, name) for name in CELL_FIELDS)))
        lane.run("lts_fill2", (nlev + 1) * cells, (
            i32(nlev + 1), i32(cells), i32(block.n_local_cells), i32(lane.G_face.shape[3]),
            i32(block.n_owned_cells), block.halo_cell_mail, block.halo_cell_class, spec, weight,
            i32(0), lane.G_face, *(_field(holder, name) for name in INTERFACE_FIELDS)))
        lane.run("lts_fill2", nlev * edges, (
            i32(nlev), i32(edges), i32(block.n_local_edges), i32(lane.G_edge.shape[3]),
            i32(block.n_owned_edges), block.halo_edge_mail, block.halo_edge_class, spec, weight,
            i32(0), lane.G_edge, *(_field(holder, name) for name in EDGE_FIELDS)))

    def _fill_perturbation(self, block: _Block, acoustic: Any, spec: Any, weight: Any) -> None:
        lane = block.lane
        i32 = np.int32
        nlev = self.nlev
        cells = block.n_local_cells - block.n_owned_cells
        edges = block.n_local_edges - block.n_owned_edges
        lane.run("lts_perturbation", nlev * cells, (
            i32(nlev), i32(cells), i32(block.n_local_cells), i32(lane.G_cell.shape[3]),
            i32(block.n_owned_cells), block.halo_cell_mail, block.halo_cell_class, spec, weight,
            i32(len(CELL_FIELDS)), lane.G_cell, i32(2), acoustic.rho_pp, acoustic.rtheta_pp))
        lane.run("lts_perturbation", nlev * edges, (
            i32(nlev), i32(edges), i32(block.n_local_edges), i32(lane.G_edge.shape[3]),
            i32(block.n_owned_edges), block.halo_edge_mail, block.halo_edge_class, spec, weight,
            i32(len(EDGE_FIELDS)), lane.G_edge, i32(1), acoustic.ru_p, acoustic.ru_p))

    def _fill_scalars(self, block: _Block, scalars: Any, spec: Any, weight: Any) -> None:
        lane = block.lane
        i32 = np.int32
        cells = block.n_local_cells - block.n_owned_cells
        lane.run("lts_fill_scalar", self.ntr * self.nlev * cells, (
            i32(self.ntr), i32(self.nlev), i32(cells), i32(block.n_local_cells), i32(lane.G_s.shape[3]),
            i32(block.n_owned_cells), block.halo_cell_mail, block.halo_cell_class, spec, weight,
            lane.G_s, scalars))

    def _publish(self, block: _Block, atmosphere: Any, plane: int) -> None:
        lane = block.lane
        i32 = np.int32
        nlev = self.nlev
        cells, edges = int(block.pub_cell_local.size), int(block.pub_edge_local.size)
        lane.run("lts_publish7", nlev * cells, (
            i32(nlev), i32(cells), i32(block.n_local_cells), i32(lane.G_cell.shape[3]),
            block.pub_cell_local, block.pub_cell_mail, i32(plane), lane.G_cell,
            *(_field(atmosphere, name) for name in CELL_FIELDS)))
        lane.run("lts_publish2", (nlev + 1) * cells, (
            i32(nlev + 1), i32(cells), i32(block.n_local_cells), i32(lane.G_face.shape[3]),
            block.pub_cell_local, block.pub_cell_mail, i32(plane), lane.G_face,
            *(_field(atmosphere, name) for name in INTERFACE_FIELDS)))
        lane.run("lts_publish2", nlev * edges, (
            i32(nlev), i32(edges), i32(block.n_local_edges), i32(lane.G_edge.shape[3]),
            block.pub_edge_local, block.pub_edge_mail, i32(plane), lane.G_edge,
            *(_field(atmosphere, name) for name in EDGE_FIELDS)))
        lane.dirty.append((block.gid, "state", int(plane)))

    def _publish_scalars(self, block: _Block, scalars: Any, plane: int) -> None:
        lane = block.lane
        i32 = np.int32
        cells = int(block.pub_cell_local.size)
        lane.run("lts_publish_scalar", self.ntr * self.nlev * cells, (
            i32(self.ntr), i32(self.nlev), i32(cells), i32(block.n_local_cells), i32(lane.G_s.shape[3]),
            block.pub_cell_local, block.pub_cell_mail, i32(plane), lane.G_s, scalars))
        lane.dirty.append((block.gid, "scalar", int(plane)))

    def _publish_trajectory(self, block: _Block, atmosphere: Any, tick: int) -> None:
        for consumer in range(max(0, block.index - self.plan.level_depth), block.index):
            plane = trajectory_plane(self.class_tick[consumer], self.effective[consumer], tick,
                                     self.planes_per_step)
            if plane is not None:
                self._publish(block, atmosphere, (block.index - consumer - 1) * self.planes_per_level + plane)

    def _publish_scalar_step_end(self, block: _Block, tick: int) -> None:
        scalars = block.driver.atmosphere.state.scalars
        for consumer in range(max(0, block.index - self.plan.level_depth), block.index):
            planes = self.planes_per_step
            plane = trajectory_plane(self.class_tick[consumer], self.effective[consumer], tick, planes)
            if plane in (planes // 2, planes):
                self._publish_scalars(block, scalars, (block.index - consumer - 1) * SCALAR_PLANES
                                      + (MID if plane == planes // 2 else END))

    def _rotate(self, lane: _Lane, consumer: int) -> None:
        """On one card: the consumer finished its step, its finer owners' end planes become start planes."""

        i32 = np.int32
        for distance in range(1, self.plan.level_depth + 1):
            provider = consumer + distance
            if provider >= self.n_classes:
                break
            base = (distance - 1) * self.planes_per_level
            for G, mail in ((lane.G_cell, lane.class_pub_cells[provider]), (lane.G_face, lane.class_pub_cells[provider]),
                            (lane.G_edge, lane.class_pub_edges[provider])):
                rows = int(G.shape[1] * G.shape[2])
                count = int(mail.size)
                lane.run("lts_plane_copy", rows * count, (
                    i32(rows), i32(count), i32(G.shape[3]), mail, i32(base + self.planes_per_step),
                    i32(base), G))
            if self.ntr:
                sbase = (distance - 1) * SCALAR_PLANES
                mail = lane.class_pub_cells[provider]
                rows = self.ntr * self.nlev
                lane.run("lts_scalar_plane_copy", rows * int(mail.size), (
                    np.int64(rows), np.int32(mail.size), np.int32(lane.G_s.shape[3]), mail,
                    np.int32(sbase + END), np.int32(sbase + START), lane.G_s))

    # -- card-to-card movement (coordinator thread, between events) ------------------------
    def _buffers(self, source: int, target: int, nbytes: int) -> dict[str, Any]:
        cp = self.cp
        entry = self._staging.get((source, target))
        if entry is None or entry["nbytes"] < nbytes:
            size = max(int(nbytes), 1 << 20)
            if entry is not None:
                size = max(size, 2 * entry["nbytes"])
            with cp.cuda.Device(source):
                send = cp.empty(size, dtype=cp.uint8)
            with cp.cuda.Device(target):
                recv = cp.empty(size, dtype=cp.uint8)
            host = np.frombuffer(cp.cuda.alloc_pinned_memory(size), dtype=np.uint8, count=size)
            entry = self._staging[(source, target)] = {"nbytes": size, "send": send, "recv": recv, "host": host}
        return entry

    def _move_many(self, jobs: list[tuple[int, int, list[tuple[Any, Any, int, int, Any, Any]]]]) -> int:
        """Card-to-card copies, all pairs at once: every pack and download first, one wait per source card,
        then every upload and unpack, one wait per target card.

        A job is (source card, target card, segments); a segment is (source array, target array, row0,
        rows, source index, target index) over (row, column) views.  Each pair appears once per call.
        """

        cp = self.cp
        prepared = []
        for source, target, segments in jobs:
            sizes = []
            for src, dst, row0, rows, idx_s, idx_d in segments:
                if src.dtype != dst.dtype or int(idx_s.size) != int(idx_d.size):
                    raise RuntimeError("a card-to-card segment changes dtype or length; its bytes would land misplaced")
                # 16-byte aligned segments, so a float64 view of the staging buffer is aligned
                sizes.append(-(-int(rows) * int(idx_s.size) * src.dtype.itemsize // 16) * 16)
            total = sum(sizes)
            if total:
                prepared.append((source, target, segments, sizes, total, self._buffers(source, target, total)))
        if not prepared:
            return 0
        for source, target, segments, sizes, total, entry in prepared:
            lane = self.lanes[source]
            with cp.cuda.Device(source), lane.stream:
                offset = 0
                for (src, dst, row0, rows, idx_s, idx_d), size in zip(segments, sizes):
                    if size:
                        view = entry["send"][offset:offset + size].view(src.dtype)
                        name = "lts_pack_f64" if src.dtype == cp.float64 else "lts_pack_f32"
                        lane.run(name, int(rows) * int(idx_s.size), (
                            np.int64(rows), np.int32(idx_s.size), np.int64(src.shape[-1]), np.int64(row0),
                            idx_s, src, view))
                    offset += size
                entry["send"][:total].get(out=entry["host"][:total], stream=lane.stream, blocking=False)
        for source in sorted({job[0] for job in prepared}):
            with cp.cuda.Device(source):
                self.lanes[source].stream.synchronize()
        for source, target, segments, sizes, total, entry in prepared:
            lane = self.lanes[target]
            with cp.cuda.Device(target), lane.stream:
                entry["recv"][:total].set(entry["host"][:total], stream=lane.stream)
                offset = 0
                for (src, dst, row0, rows, idx_s, idx_d), size in zip(segments, sizes):
                    if size:
                        view = entry["recv"][offset:offset + size].view(dst.dtype)
                        name = "lts_unpack_f64" if dst.dtype == cp.float64 else "lts_unpack_f32"
                        lane.run(name, int(rows) * int(idx_d.size), (
                            np.int64(rows), np.int32(idx_d.size), np.int64(dst.shape[-1]), np.int64(row0),
                            idx_d, view, dst))
                    offset += size
        for target in sorted({job[1] for job in prepared}):
            with cp.cuda.Device(target):
                self.lanes[target].stream.synchronize()
        return sum(job[4] for job in prepared)

    def _route(self) -> None:
        """Copy every plane published since the last flush to the cards that read its cells.

        Called before every class step (the only events that read the mailbox), at the end of
        every macro and at construction; the planes of a step and the sync after it travel together.
        """

        began = time.perf_counter()
        bplan = self.bplan
        nlev = self.nlev
        jobs = []
        for source, lane in self.lanes.items():
            if not lane.dirty:
                continue
            dirty = list(dict.fromkeys(lane.dirty))
            lane.dirty.clear()
            for target in self.cards:
                if target == source:
                    continue
                dst = self.lanes[target]
                segments = []
                for gid, kind, plane in dirty:
                    cells = bplan.route_cells.get((gid, target))
                    edges = bplan.route_edges.get((gid, target))
                    if kind == "state":
                        if cells is not None:
                            ic, jc = lane.i32(("rc", gid, target), cells), dst.i32(("rc", gid, target), cells)
                            segments.append((lane.G_cell, dst.G_cell, plane * len(CELL_FIELDS) * nlev,
                                             len(CELL_FIELDS) * nlev, ic, jc))
                            segments.append((lane.G_face, dst.G_face, plane * len(INTERFACE_FIELDS) * (nlev + 1),
                                             len(INTERFACE_FIELDS) * (nlev + 1), ic, jc))
                        if edges is not None:
                            ie, je = lane.i32(("re", gid, target), edges), dst.i32(("re", gid, target), edges)
                            segments.append((lane.G_edge, dst.G_edge, plane * len(EDGE_FIELDS) * nlev,
                                             len(EDGE_FIELDS) * nlev, ie, je))
                    elif cells is not None:
                        ic, jc = lane.i32(("rc", gid, target), cells), dst.i32(("rc", gid, target), cells)
                        rows = self.ntr * nlev
                        segments.append((lane.G_s, dst.G_s, plane * rows, rows, ic, jc))
                if segments:
                    jobs.append((source, target, [
                        (a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]), r0, r, i, j)
                        for a, b, r0, r, i, j in segments]))
        self.route_bytes += self._move_many(jobs)
        self.wall["route"] += time.perf_counter() - began

    def _pull_registers(self, index: int) -> None:
        """Before class ``index`` steps: the slots its blocks read, from the cards whose finer blocks wrote them."""

        began = time.perf_counter()
        jobs = []
        for writer, reader, slots in self.bplan.register_pulls[index]:
            src, dst = self.lanes[writer], self.lanes[reader]
            si, di = src.i32(("pull", index, writer, reader), slots), dst.i32(("pull", index, writer, reader), slots)
            segments = [(src.reg_mass_fine, dst.reg_mass_fine, 0, self.nlev, si, di),
                        (src.reg_theta_fine, dst.reg_theta_fine, 0, self.nlev, si, di)]
            if src.flux_record is not None:
                rows = self.planes_per_step * self.nlev
                segments.append((src.flux_record.reshape(-1, self.nslots), dst.flux_record.reshape(-1, self.nslots),
                                 0, rows, si, di))
                segments.append((src.theta_record.reshape(-1, self.nslots), dst.theta_record.reshape(-1, self.nslots),
                                 0, rows, si, di))
            if self.ntr:
                segments.append((src.reg_scalar.reshape(-1, self.nslots), dst.reg_scalar.reshape(-1, self.nslots),
                                 0, self.ntr * self.nlev, si, di))
            jobs.append((writer, reader, segments))
        moved = self._move_many(jobs)
        cp = self.cp
        for writer, slots in self.bplan.register_zero_after_pull[index]:
            lane = self.lanes[writer]
            idx = lane.i32(("zero", index, writer), slots)
            with cp.cuda.Device(writer), lane.stream:
                for array in (lane.reg_mass_fine, lane.reg_theta_fine):
                    lane.run("lts_zero_f64", self.nlev * int(idx.size), (
                        np.int64(self.nlev), np.int32(idx.size), np.int64(self.nslots), idx, array))
                if self.ntr:
                    rows = self.ntr * self.nlev
                    lane.run("lts_zero_f64", rows * int(idx.size), (
                        np.int64(rows), np.int32(idx.size), np.int64(self.nslots), idx, lane.reg_scalar))
        for writer, _ in self.bplan.register_zero_after_pull[index]:
            with cp.cuda.Device(writer):
                self.lanes[writer].stream.synchronize()
        self.pull_bytes += moved
        self.wall["pull"] += time.perf_counter() - began

    # -- accepted flux capture -------------------------------------------------------------
    def _attach_capture(self, block: _Block) -> None:
        driver = block.driver
        launch = driver._launch
        recover = driver._recover_candidate

        def launch_and_remember(name: str, count: int, args: tuple[Any, ...]) -> None:
            launch(name, count, args)
            if name in ("theta_edge_flux_f32", "theta_edge_flux_hoist_f32") and int(args[4]) == 3:
                block.theta_flux3 = args[16]
                if self.match_theta and block.flux_kept is not None:
                    self._impose_theta(block, args)

        def recover_and_capture(saved_state, saved_diag, acoustic, stage, acoustic_steps, scalars,
                                **lagged_heating):
            # lagged_heating: the SG2 carry (diabatic_theta, large_timestep) the
            # stock recovery removes at the final stage; passed through unchanged.
            result = recover(saved_state, saved_diag, acoustic, stage, acoustic_steps, scalars,
                             **lagged_heating)
            if int(stage) == 3:
                if block.theta_flux3 is None:
                    raise RuntimeError(
                        "the third-stage theta edge flux was not observed before recovery; the "
                        "theta register would close on a stale interval")
                lane = block.lane
                for edge, cells, slot, mass, theta in (
                        (block.fine_edge, block.fine_cells, block.fine_slot, lane.reg_mass_fine, lane.reg_theta_fine),
                        (block.coarse_edge, block.coarse_cells, block.coarse_slot,
                         lane.reg_mass_coarse, lane.reg_theta_coarse)):
                    count = int(edge.size)
                    lane.run("lts_capture", self.nlev * count, (
                        np.int32(self.nlev), np.int32(count), np.int32(block.n_local_cells),
                        np.int32(block.n_local_edges), np.int32(mass.shape[1]), edge, cells, slot,
                        np.float64(block.split_seconds), driver.atmosphere.mesh.dv_edge, result[2],
                        saved_state.rho_u, block.theta_flux3, saved_diag.theta_m, mass, theta))
                split_end = (block.step_start_tick
                             + self.acoustic_substeps * block.step_length * (block.exchanger.split + 1))
                if lane.flux_record is not None and int(block.fine_edge.size):
                    self._record_split_flux(block, result[2], split_end, saved_state.rho_u,
                                            block.theta_flux3, saved_diag.theta_m)
                block.theta_flux3 = None
                self._publish_trajectory(block, _Holder(result[0], result[1]), split_end)
            return result

        driver._launch = launch_and_remember
        driver._recover_candidate = recover_and_capture

    # -- matched interface flux (the one-card executor's, per block and card) ----------------------
    def _record_split_flux(self, block: _Block, flux_u: Any, split_end: int, ru_saved: Any,
                           theta_flux: Any, theta_saved: Any) -> None:
        """Fine side: this split's accepted mean momentum and heat on the block's edges shared with the parent.

        Written on the block's own card in register slot space; the register pull carries the slots to
        the card of the coarse block that reads them (_pull_registers).
        """

        consumer = block.index - 1
        try:
            intervals = child_flux_intervals(self.class_tick[consumer], self.effective[consumer],
                                             split_end, self.acoustic_substeps * block.step_length,
                                             self.planes_per_step)
        except ValueError as error:
            raise RuntimeError(f"class {block.index}: {error}; the parent would read a flux recorded "
                               "for another interval") from error
        lane = block.lane
        n = int(block.fine_edge.size)
        lane.run("lts_flux_record", self.nlev * n, (
            np.int32(self.nlev), np.int32(n), np.int32(block.edge_stride), np.int32(block.cell_stride),
            np.int32(self.nslots), np.int32(intervals.start), np.int32(len(intervals)),
            np.float64(block.split_seconds), block.fine_edge, block.fine_cells, block.fine_slot,
            block.driver.atmosphere.mesh.dv_edge, flux_u, ru_saved, theta_flux, theta_saved,
            lane.flux_record, lane.theta_record))
        for q in intervals:
            self._record_written[consumer][q] = self.class_tick[consumer]

    def _impose_theta(self, block: _Block, args: tuple[Any, ...]) -> None:
        """Coarse side, third stage: the child's heat integral on the interface edges (lts_theta_impose)."""

        split = block.exchanger.split
        substeps = self.acoustic_substeps
        q = substeps * split
        if any(self._record_written[block.index][q + c] != block.step_start_tick for c in range(substeps)):
            raise RuntimeError(
                f"class {block.index} would impose the child's heat flux for split {split} that was not "
                "recorded for this step; theta would cross the interface with a stale flux")
        lane = block.lane
        n = int(block.coarse_edge.size)
        lane.run("lts_theta_impose", self.nlev * n, (
            np.int32(self.nlev), np.int32(n), np.int32(block.edge_stride), np.int32(block.cell_stride),
            np.int32(self.nslots), np.int32(lane.G_edge.shape[3]), np.int32(q),
            np.int32(substeps), np.int32(q),
            np.float64(block.split_seconds), block.coarse_edge, block.coarse_cells, block.coarse_slot,
            block.coarse_edge_mail, lane.flux_record, lane.theta_record, lane.G_edge,
            args[10], args[7], args[16]))
        block.theta_imposed += 1

    def _attach_matched_flux(self, block: _Block) -> None:
        """Coarse side: wrap this block driver's acoustic momentum kernel (cache-local tap)."""

        from .cuda_fullstep_lts_scalars import tap_acoustic_momentum

        cp = self.cp
        lane = block.lane
        n = int(block.coarse_edge.size)
        with cp.cuda.Device(block.device):
            block.flux_kept = cp.zeros((self.nlev, n), dtype=cp.float32)

        def keep(arguments: tuple[Any, ...]) -> None:
            if int(arguments[1]) != block.edge_stride:
                raise RuntimeError("acoustic_ru_v841 ran on another mesh through this block's cache")
            lane.run("lts_flux_keep", self.nlev * n, (
                np.int32(self.nlev), np.int32(n), np.int32(block.edge_stride),
                block.coarse_edge, arguments[-1], block.flux_kept))

        def impose(arguments: tuple[Any, ...]) -> None:
            exchanger = block.exchanger
            small_step = int(arguments[3])
            split = exchanger.split
            stage = exchanger._stage
            substeps = self.acoustic_substeps
            if small_step < 1 or small_step > substeps or (stage < 2 and small_step > substeps // 2):
                raise RuntimeError(
                    f"the matched interface flux is specified for {substeps // 2} acoustic sub-steps in the "
                    f"first two RK stages and {substeps} in the third; another count has no child interval "
                    "to read")
            # sub-step j of any stage spans plane interval N * split + j - 1 of this class's step
            q = substeps * split + small_step - 1
            if self._record_written[block.index][q] != block.step_start_tick:
                raise RuntimeError(
                    f"class {block.index} would impose the child's flux for plane interval {q} that was not "
                    "recorded for this step; mass would cross the interface with a stale flux")
            lane.run("lts_flux_impose", self.nlev * n, (
                np.int32(self.nlev), np.int32(n), np.int32(block.edge_stride),
                np.int32(self.nslots), np.int32(lane.G_edge.shape[3]), np.int32(q),
                np.int32(substeps * split), np.int32(1 if small_step == 1 else 0),
                block.coarse_edge, block.coarse_slot, block.coarse_edge_mail, lane.flux_record,
                lane.G_edge, block.flux_kept, arguments[-2], arguments[-1]))
            block.flux_imposed += 1

        tap_acoustic_momentum(block.driver, before=keep, after=impose,
                              dimensions=(("nlev", self.nlev), ("nedges", block.edge_stride),
                                          ("ncells", block.cell_stride)))

    # -- scalar interface hooks ----------------------------------------------------------------
    def _attach_scalars(self, audit: bool) -> None:
        from . import cuda_dynamics_v841 as dynamics
        from . import cuda_transport_v841 as transport

        for block in self.blocks:
            cache = block.driver.cache
            tap_kernel(dynamics, cache, "split_flux_finish_v841_f32",
                       after=lambda arguments, b=block: self._on_split_flux(b, arguments))
            tap_kernel(transport, cache, "fct_vertical_low_order_v841",   # renamed by identity-extend 4688aaa
                       before=lambda arguments, b=block: self._on_vertical(b, arguments))
            for name in _LOW_ORDER:
                tap_kernel(transport, cache, name, before=lambda arguments, b=block: self._on_low_order(b, arguments))
            for name in _FINISH:
                tap_kernel(transport, cache, name, before=lambda arguments, b=block: self._on_finish(b, arguments))
            if audit:
                with self.cp.cuda.Device(block.device):
                    block.audit = _UntappedAudit(block.driver, block.mass_weight, block.n_owned_cells, block.driver.cache)

    def _on_split_flux(self, block: _Block, arguments: tuple[Any, ...]) -> None:
        result = arguments[4]
        if (not block.expect_mass_flux or int(arguments[1]) != block.n_local_edges
                or int(result.shape[0]) != self.nlev):
            return
        block.expect_mass_flux = False
        count = int(block.packet_edge_list.size)
        if count:
            i32 = np.int32
            block.lane.run("lts_accept_mass_flux", self.nlev * count, (
                i32(self.nlev), i32(count), i32(block.n_local_edges), i32(self.nslots),
                block.packet_edge_list, block.packet_slot, np.float64(1.0 / block.step_seconds),
                block.driver.atmosphere.mesh.dv_edge, block.lane.reg_mass_fine, result))

    def _on_vertical(self, block: _Block, arguments: tuple[Any, ...]) -> None:
        block.vertical = (arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[10])

    def _on_low_order(self, block: _Block, arguments: tuple[Any, ...]) -> None:
        i32 = np.int32
        lane = block.lane
        ntr, nlev, ncells, nedges, max_edges = (int(v) for v in arguments[:5])
        if ntr != self.ntr or nlev != self.nlev or ncells != block.n_local_cells:
            raise RuntimeError("the flux-corrected stage ran on a shape the interface tables do not describe")
        counts, edges, signs, inverse_area, upwind, residual, mass = arguments[5:12]
        if block.vertical is None:
            raise RuntimeError("the vertical low-order launch was not seen before the horizontal one; "
                               "the donor guard would read another step's fluxes")
        dt, source_old, rho_old, ww, rdzw, vertical_residual = block.vertical
        block.vertical = None
        c = block.index
        if c + 1 < self.n_classes and self.packets:
            count = int(block.packet_edge_list.size)
            lane.run("lts_packet_apply", ntr * nlev * count, (
                i32(ntr), i32(nlev), i32(count), i32(nedges), i32(self.nslots),
                block.packet_edge_list, block.packet_slot, lane.reg_scalar, upwind, residual))
            if self.guard:
                rows = block.n_guard_owned
                lane.run("lts_guard_keep", ntr * rows, (
                    i32(ntr), i32(nlev), i32(rows), i32(block.reflux_width), i32(ncells),
                    i32(nedges), i32(self.nslots), i32(max_edges), dt,
                    block.reflux_cell, block.reflux_slot, block.reflux_coef, lane.reg_scalar,
                    lane.reg_mass_fine, counts, edges, signs, inverse_area, block.packet_mask,
                    source_old, rho_old, ww, rdzw, upwind, mass, vertical_residual, block.keep_guard,
                    block.stats, block.amounts))
                block.exchanger.peer("H", [("keep_guard", block.keep_guard, "cell")])
                rows = int(block.guard_cell.size)
                lane.run("lts_guard_apply", ntr * nlev * rows, (
                    i32(ntr), i32(nlev), i32(rows), i32(ncells), i32(nedges), i32(max_edges),
                    block.guard_cell, counts, edges, signs, block.packet_mask, block.keep_guard,
                    upwind, residual))
        if c > 0:
            rows = block.n_cap_owned
            lane.run("lts_cap_keep", ntr * nlev * rows, (
                i32(ntr), i32(nlev), i32(rows), i32(block.cap_width), i32(nedges), i32(self.nslots),
                i32(ncells), block.cap_cell, block.cap_slot, block.cap_coef, block.edge_of_slot,
                i32(1 if self.cap else 0), np.float64(_RESERVE_MARGIN), block.reserve, block.content_start,
                block.air_start, lane.reg_mass_fine, upwind, residual, block.keep_low, block.keep_high,
                block.stats, block.amounts))
            block.exchanger.peer("H", [("keep_low", block.keep_low, "cell"), ("keep_high", block.keep_high, "cell")])
            rows = int(block.cap_cell.size)
            lane.run("lts_cap_apply", ntr * nlev * rows, (
                i32(ntr), i32(nlev), i32(rows), i32(block.cap_width), i32(nedges), i32(ncells),
                block.cap_cell, block.cap_slot, block.cap_coef, block.edge_of_slot,
                block.keep_low, block.keep_high, upwind, residual))
            rows = block.n_cap_owned
            lane.run("lts_hold", ntr * nlev * rows, (
                i32(ntr), i32(nlev), i32(rows), i32(block.cap_width), i32(nedges), i32(self.nslots),
                block.cap_slot, block.cap_coef, block.edge_of_slot, upwind, lane.hold))
            # Every owned row's factors are rewritten at every step and every
            # halo row is a peer entity the exchange rewrites, so no stale
            # factor survives into the next step.

    def _on_finish(self, block: _Block, arguments: tuple[Any, ...]) -> None:
        i32 = np.int32
        ntr, nlev, _ncells, nedges = (int(v) for v in arguments[:4])
        if block.index > 0:
            rows = block.n_cap_owned
            block.lane.run("lts_packet_capture", ntr * nlev * rows, (
                i32(ntr), i32(nlev), i32(rows), i32(block.cap_width), i32(nedges), i32(self.nslots),
                block.cap_slot, block.cap_coef, block.edge_of_slot, block.lane.hold, arguments[10],
                block.lane.reg_scalar, block.reserve, block.amounts))
        if block.audit is not None:
            block.audit.observe(arguments)

    def _reserve_set(self, block: _Block) -> None:
        rows = block.n_cap_owned
        if not rows:
            return
        state = block.driver.atmosphere.state
        i32 = np.int32
        block.lane.run("lts_reserve_set", self.ntr * self.nlev * rows, (
            i32(self.ntr), i32(self.nlev), i32(rows), i32(block.n_local_cells),
            block.cap_cell, block.cap_area, state.rho, state.scalars, block.reserve,
            block.content_start, block.air_start))

    # -- one block step (rank thread) ----------------------------------------------------------
    def _bind_dt(self, block: _Block, seconds: float) -> Any:
        from .cuda_driver import canonical_sha256, cuda_configuration_payload
        from .integration import RKSchedule

        entry = block.configs.get(seconds)
        if entry is None:
            config = replace(block.driver.config, config_dt=float(seconds))
            config.validate()
            payload = cuda_configuration_payload(config)
            schedule = RKSchedule.from_mpas(
                float(seconds), order=config.config_time_integration_order,
                acoustic_substeps=config.config_number_of_sub_steps,
                dynamics_splits=config.config_dynamics_split_steps)
            entry = block.configs[seconds] = (config, payload, canonical_sha256(payload), schedule)
        block.driver.config, block.driver.configuration, block.driver.configuration_sha256 = entry[:3]
        return entry[3]

    def _edge_velocity(self, block: _Block, edges: Any, cells: Any) -> None:
        atmosphere = block.driver.atmosphere
        count = int(edges.size)
        block.lane.run("lts_edge_velocity", self.nlev * count, (
            np.int32(self.nlev), np.int32(count), np.int32(block.n_local_cells),
            np.int32(block.n_local_edges), edges, cells, atmosphere.state.rho,
            atmosphere.state.rho_u, atmosphere.saved.normal_velocity))

    def _step_block(self, block: _Block, start_tick: int, length: int, event: int) -> None:
        driver = block.driver
        seconds = length * self.base_dt
        schedule = self._bind_dt(block, seconds)
        block.split_seconds = seconds / schedule.dynamics_splits
        block.step_start_tick = int(start_tick)
        block.step_length = int(length)
        if self.class_tick[block.index] != start_tick:
            raise RuntimeError(f"class {block.index} was asked to step from tick {start_tick} "
                               f"but stands at {self.class_tick[block.index]}")
        direct = [self.effective[other] < length for other in range(self.n_classes)]
        reset = [other < block.index and self.class_tick[other] == start_tick for other in range(self.n_classes)]
        block.exchanger.begin(schedule, direct, reset, event)
        atmosphere = driver.atmosphere
        atmosphere.state.time_seconds = self.start_seconds + start_tick * self.base_dt / self.planes_per_step
        block.exchanger.round_step_boundary(atmosphere.state, atmosphere.saved,
                                            None if self.forcing_of is None else self.forcing_of(block))
        if any(reset):
            # every local interface edge with the coarser class sits next to a
            # reset density: recover its velocity from the committed momentum
            self._edge_velocity(block, block.up_edge, block.up_cells)
        if self.ntr:
            if block.index > 0 and reset[block.index - 1]:
                # the coarser owner has just accepted: its columns' reserves are renewed from it
                self._reserve_set(block)
            block.expect_mass_flux = True
            block.step_seconds = float(seconds)
            block.vertical = None
        imposed, theta_imposed = block.flux_imposed, block.theta_imposed
        if self.stepper is None:
            result = driver.step_device()
            driver.atmosphere = result.atmosphere
        else:
            self.stepper(block)
        if not block.exchanger.finished():
            raise RuntimeError("a class step ended inside its round schedule; the halo clock is unusable")
        if (block.flux_kept is not None
                and block.flux_imposed - imposed != 2 * self.acoustic_substeps * schedule.dynamics_splits):
            raise RuntimeError(
                f"block {block.gid} (class {block.index}) imposed the child's interface flux "
                f"{block.flux_imposed - imposed} times in one step, not "
                f"{2 * self.acoustic_substeps * schedule.dynamics_splits}; the acoustic momentum kernel was not "
                "observed and mass crossed the interface with the class's own estimate")
        if (block.flux_kept is not None and self.match_theta
                and block.theta_imposed - theta_imposed != schedule.dynamics_splits):
            raise RuntimeError(
                f"block {block.gid} (class {block.index}) imposed the child's heat flux "
                f"{block.theta_imposed - theta_imposed} times in one step, not {schedule.dynamics_splits}; the "
                "third-stage theta edge flux was not observed")
        end_tick = start_tick + self.planes_per_step * length
        expected_end = self.start_seconds + end_tick * self.base_dt / self.planes_per_step
        if abs(float(driver.atmosphere.state.time_seconds) - expected_end) > 1.0e-6:
            raise RuntimeError(
                f"class {block.index} committed time {driver.atmosphere.state.time_seconds} s, not "
                f"{expected_end} s; its clock no longer matches the class schedule")
        self._publish_trajectory(block, driver.atmosphere, end_tick)
        if self.ntr:
            if block.expect_mass_flux:
                raise RuntimeError("the step-mean mass flux was never formed; scalars were moved by a "
                                   "flux the interface register does not hold")
            self._publish_scalar_step_end(block, end_tick)
        block.steps += 1
        # A class step is its transport's step: stale address bindings go (SameHostTransport.retire_stale_bindings).
        retire = getattr(block.transport, "retire_stale_bindings", None)
        if retire is not None:
            retire(block.tables.block)

    def _sync_block(self, block: _Block, event: int) -> None:
        cp = self.cp
        i32 = np.int32
        lane = block.lane
        driver = block.driver
        atmosphere = driver.atmosphere
        state, saved, reference = atmosphere.state, atmosphere.saved, atmosphere.reference
        block.exchanger.begin_event(event)
        count = int(block.reflux_cell.size)
        has_finer = block.index + 1 < self.n_classes
        if count:
            applied = cp.empty((self.nlev, count), dtype=cp.float64)
            lane.run("lts_reflux", self.nlev * count, (
                i32(self.nlev), i32(count), i32(block.reflux_width), i32(block.n_local_cells),
                i32(self.nslots), block.reflux_cell, block.reflux_slot, block.reflux_coef,
                np.float64(self.reflux_scale[0]), np.float64(self.reflux_scale[1]),
                lane.reg_mass_fine, lane.reg_mass_coarse, lane.reg_theta_fine, lane.reg_theta_coarse,
                np.float32(287.0), np.float32(1004.5), np.float32(100_000.0),
                reference.rho_base, reference.rho_theta_base, reference.exner_base, atmosphere.vertical.zz,
                state.rho, state.rho_theta, saved.density_perturbation, saved.rho_theta_perturbation,
                saved.theta_m, saved.exner, saved.pressure_perturbation, applied))
            del applied
        if has_finer:
            # the corrected cells reach the peers before any edge next to them recovers its velocity
            block.exchanger.peer("R", [
                ("rho", state.rho, "cell"), ("rho_theta", state.rho_theta, "cell"),
                ("density_perturbation", saved.density_perturbation, "cell"),
                ("rho_theta_perturbation", saved.rho_theta_perturbation, "cell"),
                ("theta_m", saved.theta_m, "cell"), ("exner", saved.exner, "cell"),
                ("pressure_perturbation", saved.pressure_perturbation, "cell")])
        packets = int(block.packet_slot.size)
        if packets:
            lane.run("lts_zero4", self.nlev * packets, (
                i32(self.nlev), i32(packets), i32(self.nslots), block.packet_slot,
                lane.reg_mass_fine, lane.reg_mass_coarse, lane.reg_theta_fine, lane.reg_theta_coarse))
        if count:
            rebuilt = driver._rebuild_saved_diagnostics(atmosphere)
            lane.run("lts_copy_columns", (self.nlev + 1) * count, (
                i32(self.nlev + 1), i32(count), i32(block.n_local_cells), block.reflux_cell,
                rebuilt.vertical_velocity, saved.vertical_velocity))
            del rebuilt
        if int(block.reflux_edge.size):
            self._edge_velocity(block, block.reflux_edge, block.reflux_edge_cells)
        if has_finer:
            # The velocities just recovered next to refluxed cells (owned edges) and the rebuilt w (owned reflux
            # columns) reach the peers' halo too.  THE BREAKAGE THIS PREVENTS: the peers kept the pre-reflux words
            # there until the next step's round A, so the output diagnostics a frame computes from the halo
            # (divergence, the reconstructed winds at a seam cell) differed in the last place from the one-card
            # class whenever a class with a finer interface was cut (240/60 km globe, 2 ranks, 2026-10-25).
            block.exchanger.peer("R", [("normal_velocity", saved.normal_velocity, "edge"),
                                       ("vertical_velocity", saved.vertical_velocity, "cell")])
        self._publish(block, atmosphere, self.truth_plane)
        if self.ntr:
            if packets:
                lane.run("lts_zero_scalar", self.ntr * self.nlev * packets, (
                    i32(self.ntr), i32(self.nlev), i32(packets), i32(self.nslots), block.packet_slot,
                    lane.reg_scalar))
            self._publish_scalars(block, state.scalars, self.scalar_truth_plane)

    # -- the macro loop (coordinator) ------------------------------------------------------------
    def _class_event(self, kind: str, index: int, begin: int, length: int, start: int) -> None:
        event = self.event_counter
        self.event_counter += 1
        blocks_by_rank = {block.rank: block for block in self.class_blocks[index]}
        devices = self.devices

        if kind == "step":
            # the only events that read the mailbox and the registers: bring both up to date first
            self._route()
            self._pull_registers(index)

            def op(rank: int) -> None:
                block = blocks_by_rank.get(rank)
                if block is not None:
                    self._step_block(block, start + self.planes_per_step * begin, length, event)

            with self.around_class_step(index, length * self.base_dt):
                self.ranks.call(op)
            self.class_tick[index] = start + self.planes_per_step * (begin + length)
        else:
            def op(rank: int) -> None:
                block = blocks_by_rank.get(rank)
                if block is not None:
                    self._sync_block(block, event)
                if self.lane_rank[devices[rank]] == rank:
                    self._rotate(self.lanes[devices[rank]], index)

            self.ranks.call(op)

    def advance_macro(self, macro_rate: int, *, collect: bool = False) -> None:
        start = self.tick
        rates = [int(r) for r in self.plan.rates]
        self.effective = [min(rate, int(macro_rate)) for rate in rates]
        for kind, index, begin, length in macro_schedule(rates, macro_rate):
            if kind == "sync" and self.n_classes == 1:
                continue        # one class: no register, no mailbox, nothing to settle
            self.events[kind] += 1
            began = time.perf_counter()
            self._class_event(kind, index, begin, length, start)
            spent = time.perf_counter() - began
            self.wall[kind][index] += spent
            if self.macros_done >= self.warm_after_macros:
                self.wall_warm[kind][index] += spent
                if kind == "step":
                    self.wall_warm["step_events"][index] += 1
        self.tick = start + self.planes_per_step * int(macro_rate)
        self._route()
        if self.macros_done >= self.warm_after_macros:
            self.wall_warm["base_steps"] += int(macro_rate)
        self.macros_done += 1
        for index, value in enumerate(self.class_tick):
            if value != self.tick:
                raise RuntimeError(f"class {index} ended the macro at tick {value}, not {self.tick}")

    def run(self, n_base_steps: int, *, max_rate: int | None = None, warmup_steps: int = 0) -> tuple[int, ...]:
        top = max(self.plan.rates) if max_rate is None else int(max_rate)
        macros = plan_macros(int(n_base_steps), top, warmup_steps=warmup_steps)
        for macro in macros:
            self.advance_macro(macro)
        return macros

    # -- diagnostics (coordinator thread, after an event) -------------------------------------------
    def registers_empty(self) -> bool:
        cp = self.cp
        for device, lane in self.lanes.items():
            with cp.cuda.Device(device):
                arrays = [lane.reg_mass_fine, lane.reg_mass_coarse, lane.reg_theta_fine, lane.reg_theta_coarse]
                if self.ntr:
                    arrays.append(lane.reg_scalar)
                if any(bool(cp.any(a)) for a in arrays):
                    return False
        return True

    def totals(self) -> dict[str, Any]:
        cp = self.cp
        mass = theta = 0.0
        tracer = np.zeros(self.ntr)
        low = np.full(self.ntr, np.inf)
        high = np.full(self.ntr, -np.inf)
        for block in self.blocks:
            with cp.cuda.Device(block.device):
                state = block.driver.atmosphere.state
                n = block.n_owned_cells
                air = state.rho[:, :n].astype(cp.float64) * block.mass_weight
                mass += float(cp.sum(air))
                theta += float(cp.sum(state.rho_theta[:, :n].astype(cp.float64) * block.mass_weight))
                for t in range(self.ntr):
                    q = state.scalars[t, :, :n]
                    tracer[t] += float(cp.sum(q.astype(cp.float64) * air))
                    low[t] = min(low[t], float(cp.min(q)))
                    high[t] = max(high[t], float(cp.max(q)))
                del air
        result: dict[str, Any] = {"dry_mass": mass, "rho_theta": theta}
        if self.ntr:
            result.update(tracer_mass=tracer.tolist(), tracer_min=low.tolist(), tracer_max=high.tolist())
        return result

    def health(self) -> dict[str, Any]:
        cp = self.cp
        finite = True
        w_max = u_max = 0.0
        rho_min = float("inf")
        courant: list[float] = [0.0] * self.n_classes
        for block in self.blocks:
            with cp.cuda.Device(block.device):
                atmosphere = block.driver.atmosphere
                nc, ne = block.n_owned_cells, block.n_owned_edges
                fields = (atmosphere.state.rho[:, :nc], atmosphere.state.rho_theta[:, :nc],
                          atmosphere.state.rho_w[:, :nc], atmosphere.state.rho_u[:, :ne])
                finite = finite and all(bool(cp.isfinite(value).all()) for value in fields)
                w_max = max(w_max, float(cp.max(cp.abs(atmosphere.saved.vertical_velocity[:, :nc]))))
                speed = cp.abs(atmosphere.saved.normal_velocity[:, :ne])
                if ne:
                    u_max = max(u_max, float(cp.max(speed)))
                    courant[block.index] = max(courant[block.index], float(
                        cp.max(speed / atmosphere.mesh.dc_edge[None, :ne])) * block.rate * self.base_dt)
                rho_min = min(rho_min, float(cp.min(fields[0])))
        return {"finite": finite, "max_abs_w": w_max, "max_abs_u": u_max, "min_rho": rho_min,
                "advective_courant_per_class": courant, "max_w_per_class": []}

    def gather(self, names: Sequence[str] = ("rho", "rho_theta", "rho_u", "rho_w")) -> dict[str, np.ndarray]:
        cp = self.cp
        result: dict[str, np.ndarray] = {}
        first = self.blocks[0].tables.layout
        for name in names:
            edge = name in EDGE_FIELDS
            extent = first._global_edges if edge else first._global_cells
            levels = self.nlev + 1 if name in INTERFACE_FIELDS else self.nlev
            target = np.empty((levels, extent), dtype=np.float32)
            for block in self.blocks:
                layout = block.tables.layout
                owned = block.n_owned_edges if edge else block.n_owned_cells
                l2g = (layout.edge_l2g if edge else layout.cell_l2g)[:owned].astype(np.int64)
                with cp.cuda.Device(block.device):
                    target[:, l2g] = cp.asnumpy(_field(block.driver.atmosphere, name)[:, :owned])
            result[name] = target
        return result

    def gather_scalars(self) -> np.ndarray:
        cp = self.cp
        first = self.blocks[0].tables.layout
        target = np.empty((self.ntr, self.nlev, first._global_cells), dtype=np.float32)
        for block in self.blocks:
            n = block.n_owned_cells
            l2g = block.tables.layout.cell_l2g[:n].astype(np.int64)
            with cp.cuda.Device(block.device):
                target[:, :, l2g] = cp.asnumpy(block.driver.atmosphere.state.scalars[:, :, :n])
        return target

    def scalar_stats(self) -> dict[str, Any]:
        cp = self.cp
        total = np.zeros(len(STAT_NAMES), dtype=np.int64)
        amounts = np.zeros((self.ntr, 3))
        audits = []
        for block in self.blocks:
            with cp.cuda.Device(block.device):
                total += cp.asnumpy(block.stats).astype(np.int64)
                amounts += cp.asnumpy(block.amounts).reshape(self.ntr, 3)
                if block.audit is not None:
                    audits.append(block.audit.read())
        result: dict[str, Any] = {name: int(v) for name, v in zip(STAT_NAMES, total)}
        for column, name in enumerate(AMOUNT_NAMES):
            result[name] = amounts[:, column].tolist()
        result["withheld_fraction_of_throughput"] = [
            float((row[1] + row[2]) / row[0]) if row[0] else 0.0 for row in amounts]
        if audits:
            result["clamped_mass"] = np.sum([a["clamped_mass"] for a in audits], axis=0).tolist()
            result["clamped_values"] = np.sum([a["clamped_values"] for a in audits], axis=0).astype(int).tolist()
        return result

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "gpuwm-hex.fullstep-lts-multi/v1",
            "scope": ("dynamics" + (", scalar transport" if self.ntr else "")
                      + (", per-class column physics" if self.stepper is not None else "")
                      + f", {len(self.devices)} cards, static classes"),
            "order": "fine first; classes step one after another, the blocks of a class concurrently",
            "base_dt": self.base_dt,
            "plan": self.plan.receipt(),
            "blocks": self.bplan.receipt(),
            "events": dict(self.events),
            "class_steps": [[b.steps for b in blocks] for blocks in self.class_blocks],
            "rounds": [dict(b.exchanger.rounds) for b in self.blocks],
            "peer_seconds_per_block": [b.exchanger.peer_seconds for b in self.blocks],
            "registers_empty": self.registers_empty(),
            "event_wall_seconds": {key: (list(v) if isinstance(v, list) else v) for key, v in self.wall.items()},
            "event_wall_seconds_warm": dict(self.wall_warm, after_macros=self.warm_after_macros),
            "mailbox_route_bytes": int(self.route_bytes),
            "register_pull_bytes": int(self.pull_bytes),
            "scalar_interface": self.scalar_stats() if self.ntr else None,
            "interface_flux": self.interface_flux,
            "interface_flux_impositions": [b.flux_imposed for b in self.blocks],
            "interface_theta_matched": bool(self.match_theta and self.interface_flux == "matched"),
            "interface_theta_impositions": [b.theta_imposed for b in self.blocks],
            "transports": self.transport_group.receipt(),
        }


def build_block_drivers(
    host: Mapping[str, Any],
    config: Any,
    bplan: Any,
    *,
    base_dt: float,
    cache_root: Any,
) -> tuple[list[Any], list[np.ndarray], dict[int, Any]]:
    """Slice the prepared host per block and construct one resident driver on the block's card."""

    from pathlib import Path

    import cupy as cp

    from .cuda_backend import require_cuda
    from .cuda_backend.runtime import KernelCache
    from .cuda_driver import CudaDryDycoreDriver
    from .partition_local_mesh_v841 import slice_prepared_host
    from .transport import build_advection_coefficients

    mesh = host["mesh"]
    declarations = dict(getattr(mesh, "variable_dimensions", {}) or {})
    for name in ("defc_a", "defc_b"):
        if name in mesh.arrays:
            declarations.setdefault(name, ("nCells", "maxEdges"))
    mesh.variable_dimensions = declarations
    coefficients = build_advection_coefficients(
        mesh, config_scalar_adv_order=config.config_scalar_adv_order,
        n_vert_levels=int(host["state"].rho.shape[0]), source_order_v841=True)
    bundle = {name: host[name] for name in ("mesh", "state", "vertical", "reference", "saved", "terrain_metrics")}
    bundle["coefficients"] = coefficients
    drivers, areas = [], []
    lane_caches: dict[int, Any] = {}
    for device in sorted(set(bplan.devices)):
        cache_dir = Path(cache_root) / f"lane-card{device}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        with cp.cuda.Device(device):
            capability = require_cuda(device_id=device, min_compute=(12, 0), cache_dir=cache_dir)
            lane_caches[device] = KernelCache(capability=capability, cache_dir=str(cache_dir))
    for tables in bplan.blocks:
        local = slice_prepared_host(bundle, tables.layout)
        cache_dir = Path(cache_root) / f"block{tables.gid}-class{tables.index}-card{tables.device}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        with cp.cuda.Device(tables.device):
            capability = require_cuda(device_id=tables.device, min_compute=(12, 0), cache_dir=cache_dir)
            cache = KernelCache(capability=capability, cache_dir=str(cache_dir))
            driver = CudaDryDycoreDriver.from_host(
                local["mesh"], local["state"], local["vertical"], local["reference"],
                replace(config, config_dt=float(base_dt) * tables.rate),
                saved_diagnostics=local["saved"], terrain_metrics=local["terrain_metrics"],
                advection_coefficients=local["coefficients"], kernel_cache=cache,
                reference_wind_profiles=host["profiles"])
            cp.cuda.get_current_stream().synchronize()
        drivers.append(driver)
        areas.append(np.asarray(mesh.area_cell, dtype=np.float64)[tables.layout.cell_l2g.astype(np.int64)])
    return drivers, areas, lane_caches
