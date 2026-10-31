"""Active full-step local time stepping of the dry dynamics on one device.

Scheme (conservative subcycling, fine first with recorded trajectories)
----------------------------------------------------------------------
Every rate class is one resident :class:`CudaDryDycoreDriver` on its own
sub-mesh, built by the same local-mesh slicer the same-host multi-card executor
uses, and it runs the UNCHANGED native step -- RK2, three dynamics splits, N
acoustic sub-steps (any even N; MPAS-A v8.4.2's default is 4) -- at its own
whole time step ``rate * base_dt``.  No native
kernel or pinned numerical source is edited; the seams used are ones the driver
already exposes:

* ``driver.halo_exchanger_v841`` -- the B (stage recovery) and C (acoustic)
  round sites, where a partition receives a peer's bytes.  A class receives:

  - for a FINER owner, that owner's ACCEPTED state at the round's own time.
    The finer classes cover an interval before the coarser class steps over it
    (``fullstep_lts_classes.macro_schedule``) and record their split ends on
    the planes (multiples of H/(3N)) of the coarser step they land on.  With
    N = 2 every stage and acoustic time is a split end of every finer dyadic
    class, so the value is one the model actually computed; with N = 4 the
    odd planes fall mid-split of the next finer class and take the mean of
    the two recorded split ends around them (``plane_weights``).  Never an
    extrapolation.  An acoustic perturbation is the difference of two such
    states.
  - for a COARSER owner, nothing while the entity is in the class's scratch
    zone (the class advances those cells itself and resets them at the owner's
    step boundaries), and the owner's latest accepted state, frozen, on the
    three rings beyond the zone.  The zone is deeper than a signal travels
    between two resets, so the frozen ring cannot reach an owned cell.

* ``driver._recover_candidate`` and ``driver._launch`` (instance attributes
  that wrap the bound methods) -- at every split end the class's accepted state
  is recorded for the coarser classes, and the accepted mass integral
  ``dt * dvEdge * ruAvg`` and theta integral through every rate-interface edge
  are added to FP64 registers.  ``ruAvg`` is the flux the split's density
  update actually used, so the register and the density change agree to
  rounding.

After a class has stepped, it is refluxed: at every interface edge with the
next finer class the mass and theta its own step moved through the edge are
replaced by what the finer class (the edge's owner) moved through it, one
thread per cell in ascending register order.  Momentum is never refluxed: an
edge has exactly one owner, the finer class, whose momentum is the accepted
one; after a reflux the velocities that divide by a corrected density are
recovered again from the committed momentum.

Scalar transport steps per class too; its interface law (accepted mass flux,
scalar packets, per-column reserve) is ``cuda_fullstep_lts_scalars``.

Column physics steps per class as well: ``stepper`` (set by
``woof.hex.drivers.fullstep_lts_forecast_v841``) replaces the plain dry step of
a class by the two-phase physics transaction at the class's own step, and the
halo round ``P`` hands the unpublished endpoint the owners' accepted end
states before the microphysics reads every local column.

What this does NOT do, stated so nobody reads more into it: more than one
device, re-classing during a run, or reflux of the horizontal-mixing theta
flux (the advective and acoustic theta flux is refluxed exactly; the
Smagorinsky filter's interface contribution is not, and is small).
"""
from __future__ import annotations

from dataclasses import replace
import os
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .cuda_fp32 import CUDA_FTZ_HELPERS
from .fullstep_lts_classes import (
    SCRATCH_FLAG, ClassPlan, child_flux_intervals, lts_acoustic_substeps, macro_schedule,
    plan_macros, plane_weights, planes_per_step, trajectory_plane)

CELL_FIELDS = ("rho", "rho_theta", "theta_m", "exner", "density_perturbation",
               "rho_theta_perturbation", "pressure_perturbation")
INTERFACE_FIELDS = ("rho_w", "vertical_velocity")
EDGE_FIELDS = ("rho_u", "normal_velocity")
_STATE_FIELDS = frozenset(("rho", "rho_theta", "rho_w", "rho_u"))
# exner and pressure_perturbation are only recomputed by the third RK stage
# (recover_cells_f32); after the first two stages the owner's value is still
# the split-start one, so the halo takes the split-start state for those two.
_SPLIT_START_MASK = (1 << 3) | (1 << 6)
MAX_CLASSES = 8
_SKIP, _WRITE, _RESET, _ZERO = 0, 1, 2, 3

_SOURCE = CUDA_FTZ_HELPERS + r"""
#define LTS_LOOP(total) \
    for (long long element = (long long)blockDim.x * blockIdx.x + threadIdx.x; \
         element < (long long)(total); element += (long long)gridDim.x * blockDim.x)

/* spec[4*owner] = action (0 skip, 1 write unless carried, 2 write always),
   then plane a, plane b, split-start plane; weight[2*owner] = wa, wb.  With
   wb == 0 the value is plane a's bytes. */
#define LTS_FILL(NAME, NF, DECL, PTRS) \
extern "C" __global__ void NAME( \
    const int nlev, const int n, const int nloc, const int nmail, const int first, \
    const int *mail, const unsigned char *cls, const int *spec, const float *weight, \
    const int start_mask, const float *G, DECL) \
{ \
    float *local[NF] = PTRS; \
    const long long plane = (long long)nlev * nmail; \
    LTS_LOOP((long long)nlev * n) { \
        const int i = (int)(element % n); \
        const int k = (int)(element / n); \
        const int owner = cls[i] & 31; \
        const int action = spec[4 * owner]; \
        if (action == 0 || ((cls[i] & 64) && action != 2)) continue; \
        const long long at = (long long)k * nmail + mail[i]; \
        const long long target = (long long)k * nloc + first + i; \
        const long long a = (long long)spec[4 * owner + 1] * NF; \
        const long long b = (long long)spec[4 * owner + 2] * NF; \
        const long long s = (long long)spec[4 * owner + 3] * NF; \
        const float wa = weight[2 * owner], wb = weight[2 * owner + 1]; \
        for (int f = 0; f < NF; ++f) { \
            float value; \
            if ((start_mask >> f) & 1) value = G[(s + f) * plane + at]; \
            else if (wb == 0.0f) value = G[(a + f) * plane + at]; \
            else value = wa * G[(a + f) * plane + at] + wb * G[(b + f) * plane + at]; \
            local[f][target] = value; \
        } \
    } \
}

#define LTS_PUBLISH(NAME, NF, DECL, PTRS) \
extern "C" __global__ void NAME( \
    const int nlev, const int n, const int nloc, const int nmail, \
    const int *loc, const int *mail, const int into, float *G, DECL) \
{ \
    const float *local[NF] = PTRS; \
    const long long plane = (long long)nlev * nmail; \
    LTS_LOOP((long long)nlev * n) { \
        const int i = (int)(element % n); \
        const int k = (int)(element / n); \
        const long long source = (long long)k * nloc + loc[i]; \
        const long long target = (long long)k * nmail + mail[i]; \
        for (int f = 0; f < NF; ++f) \
            G[((long long)into * NF + f) * plane + target] = local[f][source]; \
    } \
}

#define P7 float *l0, float *l1, float *l2, float *l3, float *l4, float *l5, float *l6
#define P2 float *l0, float *l1
#define C7 const float *l0, const float *l1, const float *l2, const float *l3, const float *l4, const float *l5, const float *l6
#define C2 const float *l0, const float *l1
#define A7 {l0, l1, l2, l3, l4, l5, l6}
#define A2 {l0, l1}
LTS_FILL(lts_fill7, 7, P7, A7)
LTS_FILL(lts_fill2, 2, P2, A2)
LTS_PUBLISH(lts_publish7, 7, C7, A7)
LTS_PUBLISH(lts_publish2, 2, C2, A2)

/* Acoustic perturbations are differences from the split-start state.  For a
   finer owner: its accepted state now minus its accepted state at the split
   start (action 1).  For a frozen coarser owner: zero (action 3).  Carried
   entities keep what this class's own sub-step computed. */
extern "C" __global__ void lts_perturbation(
    const int nlev, const int n, const int nloc, const int nmail, const int first,
    const int *mail, const unsigned char *cls, const int *spec, const float *weight,
    const int stride, const float *G, const int nfields, float *p0, float *p1)
{
    float *local[2] = {p0, p1};
    const long long plane = (long long)nlev * nmail;
    LTS_LOOP((long long)nlev * n) {
        const int i = (int)(element % n);
        const int k = (int)(element / n);
        const int owner = cls[i] & 31;
        const int action = spec[4 * owner];
        if (action == 0 || (cls[i] & 64)) continue;
        const long long target = (long long)k * nloc + first + i;
        if (action == 3) {
            for (int f = 0; f < nfields; ++f) local[f][target] = 0.0f;
            continue;
        }
        const long long at = (long long)k * nmail + mail[i];
        const long long a = (long long)spec[4 * owner + 1] * stride;
        const long long b = (long long)spec[4 * owner + 2] * stride;
        const long long s = (long long)spec[4 * owner + 3] * stride;
        const float wa = weight[2 * owner], wb = weight[2 * owner + 1];
        for (int f = 0; f < nfields; ++f) {
            const float now = (wb == 0.0f) ? G[(a + f) * plane + at]
                : wa * G[(a + f) * plane + at] + wb * G[(b + f) * plane + at];
            local[f][target] = now - G[(s + f) * plane + at];
        }
    }
}

/* Copy one mailbox plane onto another for a list of mailbox columns; rows is
   fields * levels of that mailbox. */
extern "C" __global__ void lts_plane_copy(
    const int rows, const int n, const int nmail, const int *mail,
    const int from, const int into, float *G)
{
    LTS_LOOP((long long)rows * n) {
        const long long row = element / n;
        const int i = (int)(element % n);
        G[((long long)into * rows + row) * nmail + mail[i]] =
            G[((long long)from * rows + row) * nmail + mail[i]];
    }
}

/* One accepted dynamics split: the dry-mass and theta that crossed each
   rate-interface edge.  flux_u is saved momentum plus the mean acoustic
   perturbation -- the flux the split's density update used.  The theta
   integral is the third-stage edge flux (theta_edge_flux_f32, which already
   carries dvEdge) plus the acoustic heat flux, perturbation momentum times
   the saved endpoint mean, exactly as acoustic_rs_ts_v841 forms it. */
extern "C" __global__ void lts_capture(
    const int nlev, const int n, const int ncells, const int nedges, const int nslots,
    const int *edge, const int *cells, const int *slot, const double dt,
    const float *dv_edge, const float *flux_u, const float *ru_saved,
    const float *theta_flux, const float *theta_saved,
    double *reg_mass, double *reg_theta)
{
    LTS_LOOP((long long)nlev * n) {
        const int i = (int)(element % n);
        const int k = (int)(element / n);
        const long long e = (long long)k * nedges + edge[i];
        const long long r = (long long)k * nslots + slot[i];
        const double dv = (double)dv_edge[edge[i]];
        const double flux = (double)flux_u[e];
        const double mean = 0.5 * ((double)theta_saved[(long long)k * ncells + cells[2 * i]]
            + (double)theta_saved[(long long)k * ncells + cells[2 * i + 1]]);
        reg_mass[r] += dt * dv * flux;
        reg_theta[r] += dt * ((double)theta_flux[e]
            + dv * (flux - (double)ru_saved[e]) * mean);
    }
}

/* Replace the coarse cell's provisional interface integrals by the fine
   side's accepted ones.  Slots of one cell are summed in ascending register
   order by one thread: no atomics, no traversal-order dependence.  The cell
   diagnostics repeat recover_cells_f32's third-stage arithmetic. */
extern "C" __global__ void lts_reflux(
    const int nlev, const int n, const int width, const int ncells, const int nslots,
    const int *cell, const int *slot, const double *coef,
    const double mass_scale, const double theta_scale,
    const double *mass_fine, const double *mass_coarse,
    const double *theta_fine, const double *theta_coarse,
    const float rgas, const float cp, const float p0,
    const float *rho_base, const float *rtheta_base, const float *exner_base,
    const float *zz, float *rho, float *rtheta, float *rho_p, float *rtheta_p,
    float *theta, float *exner, float *pressure_p, double *applied)
{
    LTS_LOOP((long long)nlev * n) {
        const int row = (int)(element % n);
        const int k = (int)(element / n);
        double dm = 0.0, dth = 0.0;
        for (int w = 0; w < width; ++w) {
            const long long r = (long long)k * nslots + slot[row * width + w];
            const double c = coef[row * width + w];
            dm += c * (mass_fine[r] - mass_coarse[r]);
            dth += c * (theta_fine[r] - theta_coarse[r]);
        }
        dm *= mass_scale;
        dth *= theta_scale;
        const long long index = (long long)k * ncells + cell[row];
        const float density_p = (float)((double)rho_p[index] + dm);
        const float theta_p = (float)((double)rtheta_p[index] + dth);
        const float full_rho = mpas_add(density_p, rho_base[index]);
        const float full_rtheta = mpas_add(theta_p, rtheta_base[index]);
        applied[(long long)k * n + row] = dm;
        rho_p[index] = density_p;
        rtheta_p[index] = theta_p;
        rho[index] = full_rho;
        rtheta[index] = full_rtheta;
        theta[index] = mpas_div(full_rtheta, full_rho);
        const float argument = mpas_mul(mpas_mul(zz[index], mpas_div(rgas, p0)), full_rtheta);
        const float value = powf(argument, rgas / (cp - rgas));
        exner[index] = value;
        pressure_p[index] = mpas_mul(mpas_mul(zz[index], rgas),
            mpas_add(mpas_mul(value, theta_p),
                mpas_mul(rtheta_base[index], mpas_sub(value, exner_base[index]))));
    }
}

extern "C" __global__ void lts_zero4(
    const int nlev, const int n, const int nslots, const int *slot,
    double *a, double *b, double *c, double *d)
{
    LTS_LOOP((long long)nlev * n) {
        const long long r = (element / n) * (long long)nslots + slot[element % n];
        a[r] = 0.0; b[r] = 0.0; c[r] = 0.0; d[r] = 0.0;
    }
}

/* recover_edge_velocity_f32's arithmetic on a list of edges. */
extern "C" __global__ void lts_edge_velocity(
    const int nlev, const int n, const int ncells, const int nedges,
    const int *edge, const int *cells, const float *rho, const float *ru, float *u)
{
    LTS_LOOP((long long)nlev * n) {
        const int i = (int)(element % n);
        const long long k = element / n;
        const long long e = k * nedges + edge[i];
        const float denominator = mpas_add(
            rho[k * ncells + cells[2 * i]], rho[k * ncells + cells[2 * i + 1]]);
        u[e] = mpas_div(mpas_mul(2.0f, ru[e]), denominator);
    }
}

/* Matched interface flux, fine side: one accepted split's mean momentum
   (saved momentum plus mean acoustic perturbation, the flux its density update
   used) on every register edge it owns, written into the consumer's interval
   slots q0 .. q0+count-1 (planes of the coarser class's step). */
extern "C" __global__ void lts_flux_record(
    const int nlev, const int n, const int nedges, const int ncells, const int nslots,
    const int q0, const int count, const double dt, const int *edge, const int *cells, const int *slot,
    const float *dv_edge, const float *flux_u, const float *ru_saved,
    const float *theta_flux, const float *theta_saved, double *record, double *record_theta)
{
    LTS_LOOP((long long)nlev * n) {
        const int i = (int)(element % n);
        const int k = (int)(element / n);
        const long long e = (long long)k * nedges + edge[i];
        const double value = (double)flux_u[e];
        /* the theta this split moved through the edge, exactly as lts_capture books it */
        const double mean = 0.5 * ((double)theta_saved[(long long)k * ncells + cells[2 * i]]
            + (double)theta_saved[(long long)k * ncells + cells[2 * i + 1]]);
        const double theta = dt * ((double)theta_flux[e]
            + (double)dv_edge[edge[i]] * (value - (double)ru_saved[e]) * mean) / count;
        for (int c = 0; c < count; ++c) {
            record[((long long)(q0 + c) * nlev + k) * nslots + slot[i]] = value;
            record_theta[((long long)(q0 + c) * nlev + k) * nslots + slot[i]] = theta;
        }
    }
}

/* Coarse side, third RK stage, right after theta_edge_flux: the large-step
   heat flux on an interface edge becomes the child's theta integral over the
   split's N planes less the acoustic heat flux this split will add there
   (dv * mean acoustic perturbation * edge-mean saved theta, the perturbation
   being the imposed child momentum minus the split-start momentum).  The
   split then moves exactly the child's heat through the edge, to rounding. */
extern "C" __global__ void lts_theta_impose(
    const int nlev, const int n, const int nedges, const int ncells, const int nslots, const int nmail,
    const int q, const int count, const int plane, const double dt,
    const int *edge, const int *cells, const int *slot, const int *mail,
    const double *record, const double *record_theta, const float *G,
    const float *dv_edge, const float *theta_saved, float *flux)
{
    LTS_LOOP((long long)nlev * n) {
        const int i = (int)(element % n);
        const int k = (int)(element / n);
        /* the split's N planes q .. q+N-1; with N = 2 this is the original
           0.5 * (r0 + r1) and (t0 + t1) / dt, bit for bit */
        double momentum = 0.0, theta = 0.0;
        for (int c = 0; c < count; ++c) {
            const long long r = ((long long)(q + c) * nlev + k) * nslots + slot[i];
            momentum += record[r];
            theta += record_theta[r];
        }
        const double save = (double)G[(((long long)plane * 2) * nlev + k) * nmail + mail[i]];
        const double perturbation = momentum / count - save;
        const double mean = 0.5 * ((double)theta_saved[(long long)k * ncells + cells[2 * i]]
            + (double)theta_saved[(long long)k * ncells + cells[2 * i + 1]]);
        flux[(long long)k * nedges + edge[i]] = (float)(theta / dt
            - (double)dv_edge[edge[i]] * perturbation * mean);
    }
}

/* Coarse side, before acoustic_ru_v841: keep ru_avg on the interface edges. */
extern "C" __global__ void lts_flux_keep(
    const int nlev, const int n, const int nedges, const int *edge,
    const float *ru_avg, float *kept)
{
    LTS_LOOP((long long)nlev * n) {
        const int i = (int)(element % n);
        const int k = (int)(element / n);
        kept[(long long)k * n + i] = ru_avg[(long long)k * nedges + edge[i]];
    }
}

/* Coarse side, after acoustic_ru_v841: the momentum the continuity and heat
   equations of this sub-step read on an interface edge is the finer class's
   accepted mean over exactly this sub-step's interval, as a perturbation from
   the split-start momentum (the finer class's accepted state at the split
   start, mailbox plane `plane`).  ru_avg is re-accumulated with that value, so
   the split's recovered flux and the coarse register equal the fine register. */
extern "C" __global__ void lts_flux_impose(
    const int nlev, const int n, const int nedges, const int nslots, const int nmail,
    const int q, const int plane, const int first_step,
    const int *edge, const int *slot, const int *mail, const double *record,
    const float *G, const float *kept, float *ru_p, float *ru_avg)
{
    LTS_LOOP((long long)nlev * n) {
        const int i = (int)(element % n);
        const int k = (int)(element / n);
        const long long e = (long long)k * nedges + edge[i];
        /* G_edge is (planes, 2 fields, nlev, nmail); field 0 is rho_u */
        const float save = G[(((long long)plane * 2) * nlev + k) * nmail + mail[i]];
        const float value = (float)(record[((long long)q * nlev + k) * nslots + slot[i]] - (double)save);
        ru_p[e] = value;
        ru_avg[e] = first_step ? value : mpas_add(kept[(long long)k * n + i], value);
    }
}

extern "C" __global__ void lts_copy_columns(
    const int nlev, const int n, const int nloc, const int *index,
    const float *source, float *target)
{
    LTS_LOOP((long long)nlev * n) {
        const long long at = (element / n) * (long long)nloc + index[element % n];
        target[at] = source[at];
    }
}
"""

_KERNEL_NAMES = ("lts_fill7", "lts_fill2", "lts_publish7", "lts_publish2", "lts_perturbation",
                 "lts_plane_copy", "lts_capture", "lts_reflux", "lts_zero4", "lts_edge_velocity",
                 "lts_copy_columns", "lts_flux_record", "lts_flux_keep", "lts_flux_impose",
                 "lts_theta_impose")
INTERFACE_FLUX_MODES = ("matched", "own")


def _field(atmosphere: Any, name: str) -> Any:
    return getattr(atmosphere.state if name in _STATE_FIELDS else atmosphere.saved, name)


class _Holder:
    __slots__ = ("state", "saved")

    def __init__(self, state: Any, saved: Any) -> None:
        self.state = state
        self.saved = saved


class _Exchanger:
    """The driver's halo hook for one class: accepted-state fills at rounds B and C."""

    #: Round E receives nothing (``round_fct_scale``), so the scalar limiter runs its
    #: species in the route's passes (cuda_transport_v841.limiter_species_per_pass, one
    #: species per pass by default) instead of all at once.  THE BREAKAGE THIS PREVENTS
    #: (hex 48/3 km Milton storm-following, 353,352 cells, four rate classes, one RTX 5090,
    #: V8X/V5090 2026-10-27): with no declaration every class step ran all six species in
    #: one pass, so the 264,024-column class held a six-species limiter block (three
    #: 998 MiB edge arrays and ten 332 MiB cell arrays, about 5.7 GiB) on top of the four
    #: classes' resident state, and the run stopped at base step 16, in that class's limiter,
    #: with "Out of memory allocating 1,046,062,080 bytes (allocated so far: 31,434,163,200
    #: bytes)".  The passes compute the same words (each FCT kernel indexes tracer-major and never
    #: reads across tracers; the class's scalar interface works on the pass's tracer rows,
    #: see begin_species_pass).
    fct_scale_parts = None

    def __init__(self, owner: "FullStepLtsActive", klass: "_Class") -> None:
        self.owner = owner
        self.klass = klass
        self.rounds = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "P": 0}
        self._schedule = None
        self._split = self._stage = self._sub = self._transport = 0
        self._direct: tuple[bool, ...] = ()
        self._reset: tuple[bool, ...] = ()

    def begin(self, schedule: Any, direct: Sequence[bool], reset: Sequence[bool]) -> None:
        self._schedule = schedule
        self._split = self._stage = self._sub = self._transport = 0
        self._direct = tuple(direct)
        self._reset = tuple(reset)

    @property
    def split(self) -> int:
        return self._split

    def _spec(self, kind: str, plane: int) -> tuple[Any, Any]:
        """Per owner class: action and planes for this round (cached on the device).

        ``plane`` counts multiples of H/(3N) from this class's step start.
        """

        owner = self.owner
        substeps = owner.acoustic_substeps
        length = self.klass.step_length
        effective = tuple(owner.effective)
        key = (self.klass.index, kind, plane, substeps * self._split, self._direct, self._reset,
               length, effective)
        cached = owner._spec_cache.get(key)
        if cached is not None:
            return cached
        spec = np.zeros(4 * MAX_CLASSES, dtype=np.int32)
        weight = np.zeros(2 * MAX_CLASSES, dtype=np.float32)
        weight[0::2] = 1.0
        me = self.klass.index
        for other in range(len(owner.classes)):
            if other == me:
                continue
            row = spec[4 * other:4 * other + 4]
            if other < me:
                # coarser: frozen at its accepted state; a carried entity is
                # written only by a step-start reset at the owner's boundary
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
            # a recorded plane is read as it is; any other plane interpolates
            # the recorded split ends around it (with N = 2: only the odd
            # sixths of two classes that share a step, an event tail)
            a, b, wa, wb = plane_weights(plane, length, effective[other], substeps)
            row[1], row[2] = base + a, base + b
            weight[2 * other], weight[2 * other + 1] = wa, wb
        cp = owner.cp
        cached = owner._spec_cache[key] = (cp.asarray(spec), cp.asarray(weight))
        return cached

    def _scalar_spec(self, kind: str, phase: int) -> tuple[Any, Any]:
        """Per owner class: action and scalar planes for a step boundary (A) or transport round (D).

        A class's scalars are accepted at its step ends.  A finer dyadic class
        ends a step at this class's step start, middle and end; two classes
        that share the step (an event tail) have no middle state, and the
        mean of start and end stands in for it.
        """

        from .cuda_fullstep_lts_scalars import END, MID, SCALAR_PLANES, START

        owner = self.owner
        key = (self.klass.index, "scalar", kind, phase, self._direct, self._reset)
        cached = owner._spec_cache.get(key)
        if cached is not None:
            return cached
        spec = np.zeros(4 * MAX_CLASSES, dtype=np.int32)
        weight = np.zeros(2 * MAX_CLASSES, dtype=np.float32)
        weight[0::2] = 1.0
        me = self.klass.index
        for other in range(len(owner.classes)):
            if other == me:
                continue
            row = spec[4 * other:4 * other + 4]
            if other < me:
                row[1:] = owner.scalars.truth_plane
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
        cp = owner.cp
        cached = owner._spec_cache[key] = (cp.asarray(spec), cp.asarray(weight))
        return cached

    def round_step_boundary(self, state: Any, saved: Any) -> None:
        self.rounds["A"] += 1
        self.owner._fill_state(self.klass, state, saved, *self._spec("A", 0), 0)
        if self.owner.scalars is not None:
            from .cuda_fullstep_lts_scalars import START

            self.owner.scalars.fill(self.klass, state.scalars, *self._scalar_spec("A", START))

    def round_stage_entry(self, state: Any, saved: Any) -> None:
        stage = self._schedule.stages[self._stage]
        final = stage.stage == 3
        substeps = self.owner.acoustic_substeps
        if self._sub != stage.acoustic_steps:
            raise RuntimeError(
                f"stage {stage.stage} recovered after {self._sub} acoustic rounds, not its "
                f"{stage.acoustic_steps}; the halo would be filled at a time the stage did not reach")
        # the stage ended after its own sub-steps: the half split for the
        # first two RK2 stages, the split end for the third
        plane = substeps * self._split + stage.acoustic_steps
        self.rounds["B"] += 1
        self.owner._fill_state(self.klass, state, saved, *self._spec("B", plane),
                               0 if final else _SPLIT_START_MASK)
        self._sub = 0
        self._stage += 1
        if final:
            self._stage = 0
            self._split += 1

    def round_acoustic(self, acoustic: Any) -> None:
        self._sub += 1
        self.rounds["C"] += 1
        limit = self._schedule.stages[self._stage].acoustic_steps
        if self._sub > limit:
            raise RuntimeError(
                f"acoustic round {self._sub} in a stage of {limit} sub-steps; the interface "
                "schedule would read a plane past the stage's end")
        self.owner._fill_perturbation(
            self.klass, acoustic, *self._spec("C", self.owner.acoustic_substeps * self._split + self._sub))

    def round_transport(self, scalar_stage: Any) -> None:
        """D: halo scalars at a transport stage entry, and once more after the last stage.

        The scalar RK reads the step-start scalars in stage one and the
        half-step predictor in stages two and three; the committed block is
        at the step end.  Finer owners supply their accepted scalars at those
        times, coarser owners their frozen accepted state; carried scratch
        cells keep what this class's own stage computed.
        """

        from .cuda_fullstep_lts_scalars import END, MID

        scalars = self.owner.scalars
        if scalars is None:
            raise RuntimeError(
                "a class transported scalars but the interface was built without them; the "
                "scalar would cross a rate interface with two different fluxes")
        self.rounds["D"] += 1
        entry = self._transport
        self._transport += 1
        if entry > 3:
            raise RuntimeError("the scalar interface is specified for three transport stages per step")
        if entry == 0:
            return      # a copy of the step-start block that round A has just filled
        scalars.fill(self.klass, scalar_stage, *self._scalar_spec("D", END if entry == 3 else MID))

    def begin_species_pass(self, first: int, last: int) -> None:
        """The limiter's next pass carries species [first, last) (cuda_transport_v841).

        The class's scalar interface applies its packets, donor guard and reserve cap
        inside the limiter, per tracer; it reads and writes the rows of that slice.
        """

        if self.owner.scalars is not None:
            self.owner.scalars.classes[self.klass.index].begin_species_pass(first, last)

    def round_fct_scale(self, scale_in: Any, scale_out: Any, *, part: int = 0, parts: int = 1) -> None:
        """E: nothing to receive, in one pass or in ``parts`` species passes (counted once).

        An owned edge reads the limiter factors of its two cells.  A carried
        scratch cell's factors are this class's own; an edge to a finer cell
        is a packet edge whose residual is zero, so the finer cell's factors
        multiply nothing.
        """

        if part == 0:
            self.rounds["E"] += 1

    def round_physics_candidate(self, atmosphere: Any) -> None:
        """P: the unpublished endpoint's halo, before the microphysics reads every local column.

        The dynamics has finished its last split, so the time is the step
        end.  A finer owner has already accepted that instant (plane six, its
        committed endpoint, microphysics included); a coarser owner is frozen
        at its latest accepted state; carried scratch cells keep what this
        class computed.  Without this round the outer halo holds whatever the
        last stage left there, and the column backend validates every local
        column.
        """

        from .cuda_fullstep_lts_scalars import END

        if not self.finished():
            raise RuntimeError("the physics halo round was asked for inside the dynamics schedule; "
                               "the halo would be filled at a time no class accepted")
        self.rounds["P"] += 1
        self.owner._fill_state(self.klass, atmosphere.state, atmosphere.saved,
                               *self._spec("P", self.owner.planes_per_step), 0)
        if self.owner.scalars is not None:
            self.owner.scalars.fill(self.klass, atmosphere.state.scalars, *self._scalar_spec("D", END))

    def finished(self) -> bool:
        return self._split == self._schedule.dynamics_splits and self._stage == 0


class _Class:
    """One rate class: its driver, device index tables and capture state."""

    def __init__(self, tables: Any, driver: Any, cp: Any) -> None:
        self.tables = tables
        self.driver = driver
        self.index = int(tables.index)
        self.rate = int(tables.rate)
        layout = tables.layout
        self.n_owned_cells = int(layout.n_owned_cells)
        self.n_owned_edges = int(layout.n_owned_edges)
        self.n_local_cells = int(layout.n_local_cells)
        self.n_local_edges = int(layout.n_local_edges)
        # The row stride of the driver's device arrays.  A limited-area driver
        # carries native's garbage column/edge past the local entities
        # (``cuda_regional_forecast_v841``); indexing it with the entity count
        # would read every level after the first one column off.
        self.cell_stride = int(driver.atmosphere.state.rho.shape[-1])
        self.edge_stride = int(driver.atmosphere.state.rho_u.shape[-1])
        if (self.cell_stride - self.n_local_cells, self.edge_stride - self.n_local_edges) not in ((0, 0), (1, 1)):
            raise ValueError(
                f"class {self.index}: device arrays are {self.cell_stride} cells x {self.edge_stride} edges "
                f"wide for a {self.n_local_cells} x {self.n_local_edges} local mesh; only native's one "
                "garbage element may follow the local entities")

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
        self.coarse_edge = i32(tables.coarse_edge_local)
        self.coarse_slot = i32(tables.coarse_slot)
        self.coarse_cells = i32(tables.coarse_cells.reshape(-1))
        coarse_local = np.asarray(tables.coarse_edge_local, dtype=np.int64)
        if np.any(coarse_local < self.n_owned_edges):
            raise AssertionError("a coarse-side register edge is owned here; the finer class owns every interface edge")
        self.coarse_edge_mail = i32(np.asarray(tables.halo_edge_mail)[coarse_local - self.n_owned_edges])
        self.flux_kept: Any | None = None
        self.flux_imposed = 0
        self.theta_imposed = 0
        self.reflux_cell = i32(tables.reflux_cell_local)
        self.reflux_slot = i32(tables.reflux_slot.reshape(-1))
        self.reflux_coef = cp.asarray(np.ascontiguousarray(tables.reflux_coef.reshape(-1), dtype=np.float64))
        self.reflux_width = int(tables.reflux_slot.shape[1])
        self.reflux_edge = i32(tables.reflux_edge_local)
        self.reflux_edge_cells = i32(tables.reflux_edge_cells.reshape(-1))
        near = np.zeros(self.n_owned_cells, dtype=bool)
        near[tables.pub_cell_local] = True
        self.near_interface = near
        self.exchanger: _Exchanger | None = None
        self.theta_flux3: Any | None = None
        self.split_seconds = 0.0
        self.step_start_tick = 0
        self.step_length = 0
        self.configs: dict[float, tuple[Any, Any, str, Any]] = {}
        self.steps = 0
        self.mass_weight: Any | None = None
        self.budget_edge: Any | None = None
        self.budget_cells: Any | None = None
        self.budget_slot: Any | None = None
        self.interior_weight: Any | None = None


class FullStepLtsActive:
    """Fine-first conservative subcycling of the dry dynamics over rate classes."""

    def __init__(
        self,
        plan: ClassPlan,
        drivers: Sequence[Any],
        *,
        base_dt: float,
        kernel_cache: Any,
        area_cell: Sequence[np.ndarray],
        dzw: np.ndarray,
        clamp_audit: bool = False,
        interface_flux: str = "matched",
    ) -> None:
        import cupy as cp

        if interface_flux not in INTERFACE_FLUX_MODES:
            raise ValueError(f"interface_flux must be one of {INTERFACE_FLUX_MODES}")
        if len(drivers) != len(plan.classes) or len(drivers) > MAX_CLASSES:
            raise ValueError("one driver per populated class, at most eight classes")
        self.cp = cp
        self.plan = plan
        self.base_dt = float(base_dt)
        # (mass, theta) multipliers of the reflux; anything but (1, 1) is a
        # diagnostic arm that does not conserve.
        self.reflux_scale = (1.0, 1.0)
        self.start_seconds = float(drivers[0].atmosphere.state.time_seconds)
        self.nlev = int(drivers[0].nlev)
        config = drivers[0].config
        transports = (bool(config.config_scalar_advection)
                      and int(drivers[0].atmosphere.state.scalars.shape[0]) > 0)
        if transports and not (bool(config.config_monotonic) or bool(config.config_positive_definite)):
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
        # ticks of base_dt / (3 N) since the start; a class of rate r steps 3 N r ticks
        self.acoustic_substeps = lts_acoustic_substeps(config.config_number_of_sub_steps)
        self.planes_per_step = planes_per_step(self.acoustic_substeps)
        self.planes_per_level = self.planes_per_step + 1   # accepted states at planes 0 .. 3N
        self.tick = 0
        self._kernels = {name: kernel_cache.raw_kernel(name, _SOURCE, module_key="hexcore.cuda_fullstep_lts_active")
                         for name in _KERNEL_NAMES}
        nlev = self.nlev
        f32 = cp.float32
        self.truth_plane = self.planes_per_level * plan.level_depth
        planes = self.truth_plane + 1
        self.G_cell = cp.zeros((planes, len(CELL_FIELDS), nlev, max(plan.n_mail_cells, 1)), dtype=f32)
        self.G_face = cp.zeros((planes, len(INTERFACE_FIELDS), nlev + 1, max(plan.n_mail_cells, 1)), dtype=f32)
        self.G_edge = cp.zeros((planes, len(EDGE_FIELDS), nlev, max(plan.n_mail_edges, 1)), dtype=f32)
        slots = max(plan.n_register_edges, 1)
        self.reg_mass_fine = cp.zeros((nlev, slots), dtype=cp.float64)
        self.reg_mass_coarse = cp.zeros_like(self.reg_mass_fine)
        self.reg_theta_fine = cp.zeros_like(self.reg_mass_fine)
        self.reg_theta_coarse = cp.zeros_like(self.reg_mass_fine)
        # Matched interface flux: per register edge and level, the finer class's
        # accepted mean momentum over each plane (base_dt / 3N of rate one) of
        # the coarser class's step.
        self.interface_flux = interface_flux
        self.flux_record = (cp.zeros((self.planes_per_step, nlev, slots), dtype=cp.float64)
                            if interface_flux == "matched" else None)
        self.theta_record = cp.zeros_like(self.flux_record) if self.flux_record is not None else None
        # the third-stage heat flux on interface edges also takes the child's integral
        self.match_theta = True
        # host bookkeeping: the consumer step start tick each interval slot was written for
        self._record_written: list[list[int | None]] = [[None] * self.planes_per_step for _ in drivers]
        self.class_tick = [0] * len(drivers)
        self.effective = [klass.rate for klass in plan.classes]
        self._spec_cache: dict[Any, tuple[Any, Any]] = {}
        self.classes = [_Class(tables, driver, cp) for tables, driver in zip(plan.classes, drivers)]
        # Each class step's four-byte validation read waits until the macro boundary
        # (cuda_driver.flush_deferred_validation_v841): every class of this executor issues on one device
        # stream, so the read only held the host until the card drained the step.  Breakage it prevents
        # (HEX-LTS-REGRESSION-2026-10-27): one device synchronize per class step left the card idle through
        # each step's host tail and the next class's first launches.  WOOF_HEX_LTS_DEFERRED_VALIDATION=0
        # reads the flag in every class step again.
        deferral = os.environ.get("WOOF_HEX_LTS_DEFERRED_VALIDATION", "1")
        if deferral not in ("0", "1"):
            raise ValueError("WOOF_HEX_LTS_DEFERRED_VALIDATION must be 0 or 1; prevents an ambiguous "
                             "refusal point in the LTS receipt")
        self.deferred_validation = deferral == "1"
        for klass in self.classes:
            klass.driver.deferred_validation_v841 = [] if self.deferred_validation else None
        self.scalars: Any | None = None
        weights_dzw = cp.asarray(np.asarray(dzw, dtype=np.float64))
        for klass, area in zip(self.classes, area_cell):
            owned = cp.asarray(np.asarray(area, dtype=np.float64)[:klass.n_owned_cells])
            klass.mass_weight = weights_dzw[:, None] * owned[None, :]
            klass.exchanger = _Exchanger(self, klass)
            klass.driver.halo_exchanger_v841 = klass.exchanger
            # The capture and the matched heat flux read the third-stage edge
            # flux with the RK correction in it (cuda_driver._theta_tendency).
            klass.driver.lts_theta_correction_in_edge_flux = True
            self._attach_capture(klass)
            if self.flux_record is not None and int(klass.coarse_edge.size):
                self._attach_matched_flux(klass)
            atmosphere = klass.driver.atmosphere
            self._publish(klass, atmosphere, self.truth_plane)
            for level in range(plan.level_depth):
                self._publish(klass, atmosphere, level * self.planes_per_level)
        if transports:
            from .cuda_fullstep_lts_scalars import ScalarInterfaces

            self.scalars = ScalarInterfaces(self, kernel_cache, area_cell, audit=clamp_audit)
        self.events = {"step": 0, "sync": 0}
        self.reflux_stats: list[dict[str, float]] = []
        self.wall = {"step": [0.0] * len(drivers), "sync": [0.0] * len(drivers)}
        self.time_classes = False
        # None: a class step is the driver's plain dry step.  Otherwise a
        # callable ``stepper(klass)`` that advances the class's driver by
        # exactly one whole class step and commits it (the physics
        # transaction); it must leave ``driver.atmosphere`` at the endpoint.
        self.stepper: Any | None = None

    @property
    def time_seconds(self) -> float:
        return self.start_seconds + self.tick * self.base_dt / self.planes_per_step

    # -- launch plumbing ---------------------------------------------------
    def _run(self, name: str, total: int, args: tuple[Any, ...]) -> None:
        if total > 0:
            self._kernels[name](((int(total) + 255) // 256,), (256,), args)

    # -- halo fills ----------------------------------------------------------
    def _fill_state(self, klass: _Class, state: Any, saved: Any, spec: Any, weight: Any,
                    start_mask: int) -> None:
        i32 = np.int32
        cells = klass.n_local_cells - klass.n_owned_cells
        edges = klass.n_local_edges - klass.n_owned_edges
        holder = _Holder(state, saved)
        self._run("lts_fill7", self.nlev * cells, (
            i32(self.nlev), i32(cells), i32(klass.cell_stride), i32(self.G_cell.shape[3]),
            i32(klass.n_owned_cells), klass.halo_cell_mail, klass.halo_cell_class, spec, weight,
            i32(start_mask), self.G_cell, *(_field(holder, name) for name in CELL_FIELDS)))
        self._run("lts_fill2", (self.nlev + 1) * cells, (
            i32(self.nlev + 1), i32(cells), i32(klass.cell_stride), i32(self.G_face.shape[3]),
            i32(klass.n_owned_cells), klass.halo_cell_mail, klass.halo_cell_class, spec, weight,
            i32(0), self.G_face, *(_field(holder, name) for name in INTERFACE_FIELDS)))
        self._run("lts_fill2", self.nlev * edges, (
            i32(self.nlev), i32(edges), i32(klass.edge_stride), i32(self.G_edge.shape[3]),
            i32(klass.n_owned_edges), klass.halo_edge_mail, klass.halo_edge_class, spec, weight,
            i32(0), self.G_edge, *(_field(holder, name) for name in EDGE_FIELDS)))

    def _fill_perturbation(self, klass: _Class, acoustic: Any, spec: Any, weight: Any) -> None:
        i32 = np.int32
        cells = klass.n_local_cells - klass.n_owned_cells
        edges = klass.n_local_edges - klass.n_owned_edges
        # mailbox cell fields 0 and 1 are rho and rho_theta; edge field 0 is rho_u
        self._run("lts_perturbation", self.nlev * cells, (
            i32(self.nlev), i32(cells), i32(klass.cell_stride), i32(self.G_cell.shape[3]),
            i32(klass.n_owned_cells), klass.halo_cell_mail, klass.halo_cell_class, spec, weight,
            i32(len(CELL_FIELDS)), self.G_cell, i32(2), acoustic.rho_pp, acoustic.rtheta_pp))
        self._run("lts_perturbation", self.nlev * edges, (
            i32(self.nlev), i32(edges), i32(klass.edge_stride), i32(self.G_edge.shape[3]),
            i32(klass.n_owned_edges), klass.halo_edge_mail, klass.halo_edge_class, spec, weight,
            i32(len(EDGE_FIELDS)), self.G_edge, i32(1), acoustic.ru_p, acoustic.ru_p))

    def _publish(self, klass: _Class, atmosphere: Any, plane: int) -> None:
        i32 = np.int32
        cells, edges = int(klass.pub_cell_local.size), int(klass.pub_edge_local.size)
        self._run("lts_publish7", self.nlev * cells, (
            i32(self.nlev), i32(cells), i32(klass.cell_stride), i32(self.G_cell.shape[3]),
            klass.pub_cell_local, klass.pub_cell_mail, i32(plane), self.G_cell,
            *(_field(atmosphere, name) for name in CELL_FIELDS)))
        self._run("lts_publish2", (self.nlev + 1) * cells, (
            i32(self.nlev + 1), i32(cells), i32(klass.cell_stride), i32(self.G_face.shape[3]),
            klass.pub_cell_local, klass.pub_cell_mail, i32(plane), self.G_face,
            *(_field(atmosphere, name) for name in INTERFACE_FIELDS)))
        self._run("lts_publish2", self.nlev * edges, (
            i32(self.nlev), i32(edges), i32(klass.edge_stride), i32(self.G_edge.shape[3]),
            klass.pub_edge_local, klass.pub_edge_mail, i32(plane), self.G_edge,
            *(_field(atmosphere, name) for name in EDGE_FIELDS)))

    def _publish_trajectory(self, klass: _Class, atmosphere: Any, tick: int) -> None:
        """Record this class's accepted state at ``tick`` for every coarser class that asks for it."""

        for consumer in range(max(0, klass.index - self.plan.level_depth), klass.index):
            plane = trajectory_plane(self.class_tick[consumer], self.effective[consumer], tick,
                                     self.planes_per_step)
            if plane is not None:
                self._publish(klass, atmosphere,
                              (klass.index - consumer - 1) * self.planes_per_level + plane)

    def _rotate(self, consumer: _Class) -> None:
        """The consumer finished its step: its finer owners' end states become its next start states."""

        i32 = np.int32
        for distance in range(1, self.plan.level_depth + 1):
            provider = consumer.index + distance
            if provider >= len(self.classes):
                break
            base = (distance - 1) * self.planes_per_level
            source = self.classes[provider]
            for G, mail in ((self.G_cell, source.pub_cell_mail), (self.G_face, source.pub_cell_mail),
                            (self.G_edge, source.pub_edge_mail)):
                rows = int(G.shape[1] * G.shape[2])
                count = int(mail.size)
                self._run("lts_plane_copy", rows * count, (
                    i32(rows), i32(count), i32(G.shape[3]), mail, i32(base + self.planes_per_step),
                    i32(base), G))

    # -- accepted flux capture and trajectory record --------------------------
    def _attach_capture(self, klass: _Class) -> None:
        driver = klass.driver
        launch = driver._launch
        recover = driver._recover_candidate

        def launch_and_remember(name: str, count: int, args: tuple[Any, ...]) -> None:
            launch(name, count, args)
            if name in ("theta_edge_flux_f32", "theta_edge_flux_hoist_f32") and int(args[4]) == 3:
                klass.theta_flux3 = args[16]
                if self.match_theta and klass.flux_kept is not None:
                    self._impose_theta(klass, args)

        def recover_and_capture(saved_state, saved_diag, acoustic, stage, acoustic_steps, scalars,
                                **lagged_heating):
            # lagged_heating: the SG2 carry (diabatic_theta, large_timestep) the
            # stock recovery removes at the final stage; passed through unchanged.
            result = recover(saved_state, saved_diag, acoustic, stage, acoustic_steps, scalars,
                             **lagged_heating)
            if int(stage) == 3:
                if klass.theta_flux3 is None:
                    raise RuntimeError(
                        "the third-stage theta edge flux was not observed before recovery; the "
                        "theta register would close on a stale interval")
                for edge, cells, slot, mass, theta in (
                        (klass.fine_edge, klass.fine_cells, klass.fine_slot,
                         self.reg_mass_fine, self.reg_theta_fine),
                        (klass.coarse_edge, klass.coarse_cells, klass.coarse_slot,
                         self.reg_mass_coarse, self.reg_theta_coarse)):
                    count = int(edge.size)
                    self._run("lts_capture", self.nlev * count, (
                        np.int32(self.nlev), np.int32(count), np.int32(klass.cell_stride),
                        np.int32(klass.edge_stride), np.int32(mass.shape[1]), edge, cells, slot,
                        np.float64(klass.split_seconds), driver.atmosphere.mesh.dv_edge, result[2],
                        saved_state.rho_u, klass.theta_flux3, saved_diag.theta_m, mass, theta))
                if klass.budget_edge is not None:
                    # the limited-area interior's boundary register: the same
                    # accepted split flux, through the interior/zone edges
                    count = int(klass.budget_edge.size)
                    self._run("lts_capture", self.nlev * count, (
                        np.int32(self.nlev), np.int32(count), np.int32(klass.cell_stride),
                        np.int32(klass.edge_stride), np.int32(self.budget_mass.shape[1]),
                        klass.budget_edge, klass.budget_cells, klass.budget_slot,
                        np.float64(klass.split_seconds), driver.atmosphere.mesh.dv_edge, result[2],
                        saved_state.rho_u, klass.theta_flux3, saved_diag.theta_m,
                        self.budget_mass, self.budget_theta))
                # the split's recovered state is an accepted state of this class
                split_end = (klass.step_start_tick
                             + self.acoustic_substeps * klass.step_length * (klass.exchanger.split + 1))
                if self.flux_record is not None and int(klass.fine_edge.size):
                    self._record_split_flux(klass, result[2], split_end, saved_state.rho_u,
                                            klass.theta_flux3, saved_diag.theta_m)
                klass.theta_flux3 = None
                self._publish_trajectory(klass, _Holder(result[0], result[1]), split_end)
            return result

        driver._launch = launch_and_remember
        driver._recover_candidate = recover_and_capture

    # -- matched interface flux --------------------------------------------------
    def _record_split_flux(self, klass: _Class, flux_u: Any, split_end: int, ru_saved: Any,
                           theta_flux: Any, theta_saved: Any) -> None:
        """Fine side: this split's accepted mean momentum on the edges it shares with its parent.

        On the even-N clock a split of a dyadic child spans N / 2 planes of its
        parent's step (N planes when the two share a step in an event tail),
        and each of the parent's acoustic sub-steps spans exactly one plane
        (``child_flux_intervals``).  Breakage a mismatch would cause: the parent
        would move mass through the edge over one interval with the child's
        flux of another, and the reflux would have to repair the difference
        after the fact.
        """

        consumer = klass.index - 1
        try:
            intervals = child_flux_intervals(self.class_tick[consumer], self.effective[consumer],
                                             split_end, self.acoustic_substeps * klass.step_length,
                                             self.planes_per_step)
        except ValueError as error:
            raise RuntimeError(f"class {klass.index}: {error}; the parent would read a flux recorded "
                               "for another interval") from error
        n = int(klass.fine_edge.size)
        self._run("lts_flux_record", self.nlev * n, (
            np.int32(self.nlev), np.int32(n), np.int32(klass.edge_stride), np.int32(klass.cell_stride),
            np.int32(self.flux_record.shape[2]), np.int32(intervals.start), np.int32(len(intervals)),
            np.float64(klass.split_seconds), klass.fine_edge, klass.fine_cells, klass.fine_slot,
            klass.driver.atmosphere.mesh.dv_edge, flux_u, ru_saved, theta_flux, theta_saved,
            self.flux_record, self.theta_record))
        for q in intervals:
            self._record_written[consumer][q] = self.class_tick[consumer]

    def _impose_theta(self, klass: _Class, args: tuple[Any, ...]) -> None:
        """Coarse side, third stage: the child's heat integral on the interface edges (lts_theta_impose)."""

        split = klass.exchanger.split
        substeps = self.acoustic_substeps
        q = substeps * split
        if any(self._record_written[klass.index][q + c] != klass.step_start_tick for c in range(substeps)):
            raise RuntimeError(
                f"class {klass.index} would impose the child's heat flux for split {split} that was not "
                "recorded for this step; theta would cross the interface with a stale flux")
        n = int(klass.coarse_edge.size)
        self._run("lts_theta_impose", self.nlev * n, (
            np.int32(self.nlev), np.int32(n), np.int32(klass.edge_stride), np.int32(klass.cell_stride),
            np.int32(self.flux_record.shape[2]), np.int32(self.G_edge.shape[3]), np.int32(q),
            np.int32(substeps), np.int32(q),
            np.float64(klass.split_seconds), klass.coarse_edge, klass.coarse_cells, klass.coarse_slot,
            klass.coarse_edge_mail, self.flux_record, self.theta_record, self.G_edge,
            args[10], args[7], args[16]))
        klass.theta_imposed += 1

    def _attach_matched_flux(self, klass: _Class) -> None:
        """Coarse side: wrap this class driver's acoustic momentum kernel (cache-local tap)."""

        from .cuda_fullstep_lts_scalars import tap_acoustic_momentum

        cp = self.cp
        n = int(klass.coarse_edge.size)
        klass.flux_kept = cp.zeros((self.nlev, n), dtype=cp.float32)

        def keep(arguments: tuple[Any, ...]) -> None:
            if int(arguments[1]) != klass.edge_stride:
                raise RuntimeError("acoustic_ru_v841 ran on another mesh through this class's cache")
            self._run("lts_flux_keep", self.nlev * n, (
                np.int32(self.nlev), np.int32(n), np.int32(klass.edge_stride),
                klass.coarse_edge, arguments[-1], klass.flux_kept))

        def impose(arguments: tuple[Any, ...]) -> None:
            exchanger = klass.exchanger
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
            if self._record_written[klass.index][q] != klass.step_start_tick:
                raise RuntimeError(
                    f"class {klass.index} would impose the child's flux for plane interval {q} that was not recorded "
                    "for this step; mass would cross the interface with a stale flux")
            self._run("lts_flux_impose", self.nlev * n, (
                np.int32(self.nlev), np.int32(n), np.int32(klass.edge_stride),
                np.int32(self.flux_record.shape[2]), np.int32(self.G_edge.shape[3]), np.int32(q),
                np.int32(substeps * split), np.int32(1 if small_step == 1 else 0),
                klass.coarse_edge, klass.coarse_slot, klass.coarse_edge_mail, self.flux_record,
                self.G_edge, klass.flux_kept, arguments[-2], arguments[-1]))
            klass.flux_imposed += 1

        tap_acoustic_momentum(klass.driver, before=keep, after=impose,
                              dimensions=(("nlev", self.nlev), ("nedges", klass.edge_stride),
                                          ("ncells", klass.cell_stride)))

    # -- one class step ------------------------------------------------------
    def _bind_dt(self, klass: _Class, seconds: float) -> Any:
        from .cuda_driver import canonical_sha256, cuda_configuration_payload
        from .integration import RKSchedule

        entry = klass.configs.get(seconds)
        if entry is None:
            config = replace(klass.driver.config, config_dt=float(seconds))
            config.validate()
            payload = cuda_configuration_payload(config)
            schedule = RKSchedule.from_mpas(
                float(seconds), order=config.config_time_integration_order,
                acoustic_substeps=config.config_number_of_sub_steps,
                dynamics_splits=config.config_dynamics_split_steps)
            entry = klass.configs[seconds] = (config, payload, canonical_sha256(payload), schedule)
        klass.driver.config, klass.driver.configuration, klass.driver.configuration_sha256 = entry[:3]
        return entry[3]

    def _step(self, klass: _Class, start_tick: int, length: int) -> None:
        driver = klass.driver
        seconds = length * self.base_dt
        schedule = self._bind_dt(klass, seconds)
        klass.split_seconds = seconds / schedule.dynamics_splits
        klass.step_start_tick = int(start_tick)
        klass.step_length = int(length)
        if self.class_tick[klass.index] != start_tick:
            raise RuntimeError(f"class {klass.index} was asked to step from tick {start_tick} "
                               f"but stands at {self.class_tick[klass.index]}")
        # a finer owner is read directly when its split ends land on every sixth of this step
        direct = [self.effective[other] < length for other in range(len(self.classes))]
        # a coarser owner has an accepted state at this instant exactly when it stands here
        reset = [other < klass.index and self.class_tick[other] == start_tick
                 for other in range(len(self.classes))]
        klass.exchanger.begin(schedule, direct, reset)
        atmosphere = driver.atmosphere
        atmosphere.state.time_seconds = self.start_seconds + start_tick * self.base_dt / self.planes_per_step
        klass.exchanger.round_step_boundary(atmosphere.state, atmosphere.saved)
        if any(reset):
            # owned interface edges now sit next to a reset (possibly refluxed)
            # density: recover their velocity from the committed momentum
            self._edge_velocity(klass, klass.fine_edge, klass.fine_cells)
        if self.scalars is not None:
            self.scalars.begin_step(klass, seconds)
        imposed, theta_imposed = klass.flux_imposed, klass.theta_imposed
        if self.stepper is None:
            result = driver.step_device()
            driver.atmosphere = result.atmosphere
        else:
            self.stepper(klass)
        if not klass.exchanger.finished():
            raise RuntimeError("a class step ended inside its round schedule; the halo clock is unusable")
        if (klass.flux_kept is not None
                and klass.flux_imposed - imposed != 2 * self.acoustic_substeps * schedule.dynamics_splits):
            raise RuntimeError(
                f"class {klass.index} imposed the child's interface flux {klass.flux_imposed - imposed} times in "
                f"one step, not {2 * self.acoustic_substeps * schedule.dynamics_splits}; the acoustic momentum "
                "kernel was not observed and mass crossed the interface with the class's own estimate")
        if (klass.flux_kept is not None and self.match_theta
                and klass.theta_imposed - theta_imposed != schedule.dynamics_splits):
            raise RuntimeError(
                f"class {klass.index} imposed the child's heat flux {klass.theta_imposed - theta_imposed} times "
                f"in one step, not {schedule.dynamics_splits}; the third-stage theta edge flux was not observed")
        end_tick = start_tick + self.planes_per_step * length
        expected_end = self.start_seconds + end_tick * self.base_dt / self.planes_per_step
        if abs(float(driver.atmosphere.state.time_seconds) - expected_end) > 1.0e-6:
            raise RuntimeError(
                f"class {klass.index} committed time {driver.atmosphere.state.time_seconds} s, not "
                f"{expected_end} s; its clock no longer matches the class schedule")
        self.class_tick[klass.index] = end_tick
        # the committed endpoint replaces the last split's recovered record
        self._publish_trajectory(klass, driver.atmosphere, end_tick)
        if self.scalars is not None:
            self.scalars.end_step(klass)
            self.scalars.publish_step_end(klass, end_tick)
        klass.steps += 1

    # -- synchronisation -----------------------------------------------------
    def _sync(self, klass: _Class, *, collect: bool = False) -> None:
        """Reflux at the finer interface, publish the accepted state, rotate the finer records."""

        cp = self.cp
        i32 = np.int32
        driver = klass.driver
        atmosphere = driver.atmosphere
        state, saved, reference = atmosphere.state, atmosphere.saved, atmosphere.reference
        count = int(klass.reflux_cell.size)
        if count:
            applied = cp.empty((self.nlev, count), dtype=cp.float64)
            self._run("lts_reflux", self.nlev * count, (
                i32(self.nlev), i32(count), i32(klass.reflux_width), i32(klass.cell_stride),
                i32(self.reg_mass_fine.shape[1]), klass.reflux_cell, klass.reflux_slot, klass.reflux_coef,
                np.float64(self.reflux_scale[0]), np.float64(self.reflux_scale[1]),
                self.reg_mass_fine, self.reg_mass_coarse, self.reg_theta_fine, self.reg_theta_coarse,
                np.float32(287.0), np.float32(1004.5), np.float32(100_000.0),
                reference.rho_base, reference.rho_theta_base, reference.exner_base, atmosphere.vertical.zz,
                state.rho, state.rho_theta, saved.density_perturbation, saved.rho_theta_perturbation,
                saved.theta_m, saved.exner, saved.pressure_perturbation, applied))
            if collect:
                rho = state.rho[:, klass.reflux_cell]
                self.reflux_stats.append({
                    "class": klass.index, "tick": self.class_tick[klass.index],
                    "max_relative_density_correction": float(cp.max(cp.abs(applied) / rho.astype(cp.float64))),
                })
            self._run("lts_zero4", self.nlev * int(klass.coarse_slot.size), (
                i32(self.nlev), i32(klass.coarse_slot.size), i32(self.reg_mass_fine.shape[1]), klass.coarse_slot,
                self.reg_mass_fine, self.reg_mass_coarse, self.reg_theta_fine, self.reg_theta_coarse))
            # Velocities that divide by a corrected density: w in the refluxed
            # columns (the driver's own recovery, copied there only) and u on
            # this class's edges that touch one.
            rebuilt = driver._rebuild_saved_diagnostics(atmosphere)
            self._run("lts_copy_columns", (self.nlev + 1) * count, (
                i32(self.nlev + 1), i32(count), i32(klass.cell_stride), klass.reflux_cell,
                rebuilt.vertical_velocity, saved.vertical_velocity))
            del rebuilt
            self._edge_velocity(klass, klass.reflux_edge, klass.reflux_edge_cells)
        self._publish(klass, atmosphere, self.truth_plane)
        self._rotate(klass)
        if self.scalars is not None:
            self.scalars.sync(klass)

    def _edge_velocity(self, klass: _Class, edges: Any, cells: Any) -> None:
        atmosphere = klass.driver.atmosphere
        count = int(edges.size)
        self._run("lts_edge_velocity", self.nlev * count, (
            np.int32(self.nlev), np.int32(count), np.int32(klass.cell_stride),
            np.int32(klass.edge_stride), edges, cells, atmosphere.state.rho,
            atmosphere.state.rho_u, atmosphere.saved.normal_velocity))

    # -- the macro loop ------------------------------------------------------
    def advance_macro(self, macro_rate: int, *, collect: bool = False) -> None:
        """Advance every class over ``macro_rate`` base steps."""

        start = self.tick
        rates = [klass.rate for klass in self.classes]
        self.effective = [min(rate, int(macro_rate)) for rate in rates]
        stream = self.cp.cuda.get_current_stream()
        for kind, index, begin, length in macro_schedule(rates, macro_rate):
            klass = self.classes[index]
            self.events[kind] += 1
            began = time.perf_counter() if self.time_classes else 0.0
            if kind == "step":
                self._step(klass, start + self.planes_per_step * begin, length)
            else:
                self._sync(klass, collect=collect)
            if self.time_classes:
                stream.synchronize()
                self.wall[kind][index] += time.perf_counter() - began
        self.flush_deferred_validation()
        self.tick = start + self.planes_per_step * int(macro_rate)
        for index, value in enumerate(self.class_tick):
            if value != self.tick:
                raise RuntimeError(f"class {index} ended the macro at tick {value}, not {self.tick}")

    def flush_deferred_validation(self) -> int:
        """Read every class step's deferred validation flag (one transfer per class); refuse on any set flag."""

        read = 0
        for klass in self.classes:
            flush = getattr(klass.driver, "flush_deferred_validation_v841", None)
            if flush is not None:
                read += int(flush() or 0)
        self.validation_flags_read = getattr(self, "validation_flags_read", 0) + read
        return read

    def run(self, n_base_steps: int, *, max_rate: int | None = None, warmup_steps: int = 0) -> tuple[int, ...]:
        top = max(klass.rate for klass in self.classes) if max_rate is None else int(max_rate)
        macros = plan_macros(int(n_base_steps), top, warmup_steps=warmup_steps)
        for macro in macros:
            self.advance_macro(macro)
        return macros

    # -- diagnostics -----------------------------------------------------------
    def registers_empty(self) -> bool:
        cp = self.cp
        if self.scalars is not None and not self.scalars.register_empty():
            return False
        return not bool(cp.any(self.reg_mass_fine) or cp.any(self.reg_mass_coarse)
                        or cp.any(self.reg_theta_fine) or cp.any(self.reg_theta_coarse))

    def attach_interior_budget(self, interior: np.ndarray, dzw: np.ndarray,
                               cells_on_edge: np.ndarray) -> dict[str, Any]:
        """Account the dry mass of a limited-area mesh's INTERIOR against its boundary flux.

        ``interior`` is the global mask of cells outside the lateral-boundary
        zone (``bdyMaskCell == 0``).  Nothing but horizontal flux through the
        edges between an interior and a zone cell changes the interior's dry
        mass: the specified-zone and relaxation tendencies act on zone cells
        only, physics moves no dry air, and the rate-interface reflux moves
        mass between interior cells.  Every such edge is owned by exactly one
        class, which adds its accepted split flux (the flux its density update
        used) to an FP64 register at every split; ``totals`` then reports the
        interior mass and the integrated outflow, whose sum is constant to
        rounding.  Diagnostic only: no state is changed.
        """

        cp = self.cp
        mask = np.asarray(interior, dtype=bool)
        coe_global = np.asarray(cells_on_edge, dtype=np.int64)
        slots_used = 0
        for klass in self.classes:
            layout = klass.tables.layout
            c_l2g = np.asarray(layout.cell_l2g, dtype=np.int64)
            e_l2g = np.asarray(layout.edge_l2g, dtype=np.int64)
            klass.interior_weight = klass.mass_weight * cp.asarray(
                mask[c_l2g[:klass.n_owned_cells]].astype(np.float64))[None, :]
            pair = coe_global[e_l2g[:layout.n_owned_edges]]
            both = (pair >= 0).all(axis=1)
            inside = mask[np.where(pair >= 0, pair, 0)] & (pair >= 0)
            local_edges = np.flatnonzero(both & (inside[:, 0] != inside[:, 1]))
            if local_edges.size == 0:
                continue
            cells = layout.cell_g2l().astype(np.int64)[pair[local_edges]]
            if np.any(cells < 0):
                raise AssertionError(f"class {klass.index}: an interior boundary edge has a non-local cell")
            klass.budget_edge = cp.asarray(local_edges.astype(np.int32))
            klass.budget_cells = cp.asarray(cells.astype(np.int32).reshape(-1))
            klass.budget_slot = cp.asarray(np.arange(slots_used, slots_used + local_edges.size, dtype=np.int32))
            # positive normal velocity runs from cellsOnEdge[0] to [1]: outward when [0] is interior
            klass.budget_sign = np.where(inside[local_edges, 0], 1.0, -1.0)
            slots_used += int(local_edges.size)
        self.budget_mass = cp.zeros((self.nlev, max(slots_used, 1)), dtype=cp.float64)
        self.budget_theta = cp.zeros_like(self.budget_mass)
        sign_all = np.zeros(max(slots_used, 1))
        for klass in self.classes:
            if klass.budget_edge is not None:
                sign_all[cp.asnumpy(klass.budget_slot)] = klass.budget_sign
        self.budget_sign = cp.asarray(sign_all)
        self.budget_dzw = cp.asarray(np.asarray(dzw, dtype=np.float64))
        return {"interior_cells": int(mask.sum()), "boundary_edges": slots_used,
                "edges_per_class": [0 if k.budget_edge is None else int(k.budget_edge.size) for k in self.classes]}

    def totals(self) -> dict[str, float]:
        """FP64 owned-union integrals of dry mass and coupled theta (area * dzw weights)."""

        cp = self.cp
        mass = theta = 0.0
        interior = None
        for klass in self.classes:
            state = klass.driver.atmosphere.state
            n = klass.n_owned_cells
            mass += float(cp.sum(state.rho[:, :n].astype(cp.float64) * klass.mass_weight))
            theta += float(cp.sum(state.rho_theta[:, :n].astype(cp.float64) * klass.mass_weight))
            if klass.interior_weight is not None:
                interior = (interior or 0.0) + float(cp.sum(state.rho[:, :n].astype(cp.float64)
                                                            * klass.interior_weight))
        result: dict[str, Any] = {"dry_mass": mass, "rho_theta": theta}
        if interior is not None:
            outflow = float(cp.sum(self.budget_dzw[:, None] * self.budget_sign[None, :] * self.budget_mass))
            result.update({"interior_dry_mass": interior, "interior_boundary_outflow": outflow,
                           "interior_mass_plus_outflow": interior + outflow})
        if self.scalars is not None:
            result.update(self.scalars.totals())
        return result

    def health(self) -> dict[str, Any]:
        cp = self.cp
        finite = True
        w_max = u_max = 0.0
        rho_min = float("inf")
        courant: list[float] = []
        where: list[dict[str, Any]] = []
        for klass in self.classes:
            atmosphere = klass.driver.atmosphere
            nc, ne = klass.n_owned_cells, klass.n_owned_edges
            fields = (atmosphere.state.rho[:, :nc], atmosphere.state.rho_theta[:, :nc],
                      atmosphere.state.rho_w[:, :nc], atmosphere.state.rho_u[:, :ne])
            finite = finite and all(bool(cp.isfinite(value).all()) for value in fields)
            w_max = max(w_max, float(cp.max(cp.abs(atmosphere.saved.vertical_velocity[:, :nc]))))
            speed = cp.abs(atmosphere.saved.normal_velocity[:, :ne])
            u_max = max(u_max, float(cp.max(speed)))
            # advective Courant number of this class at its own whole step
            courant.append(float(cp.max(speed / atmosphere.mesh.dc_edge[None, :ne]))
                           * klass.rate * self.base_dt)
            rho_min = min(rho_min, float(cp.min(fields[0])))
            # where the largest vertical velocity of this class sits: level,
            # global cell, and whether that column is one another class reads
            w = cp.abs(atmosphere.saved.vertical_velocity[:, :nc])
            flat = int(cp.argmax(w))
            level, cell = divmod(flat, nc)
            where.append({"max_abs_w": float(w[level, cell]), "level": level,
                          "global_cell": int(klass.tables.layout.cell_l2g[cell]),
                          "near_interface": bool(klass.near_interface[cell])})
        return {"finite": finite, "max_abs_w": w_max, "max_abs_u": u_max, "min_rho": rho_min,
                "advective_courant_per_class": courant, "max_w_per_class": where,
                "max_u_per_class": self._where_fastest()}

    def _where_fastest(self) -> list[dict[str, Any]]:
        """Per class, where its largest normal wind sits: level, global edge, its two global cells."""

        cp = self.cp
        rows: list[dict[str, Any]] = []
        for klass in self.classes:
            atmosphere = klass.driver.atmosphere
            ne = klass.n_owned_edges
            speed = cp.abs(atmosphere.saved.normal_velocity[:, :ne])
            level, edge = divmod(int(cp.argmax(speed)), ne)
            layout = klass.tables.layout
            cells = cp.asnumpy(cp.asarray(atmosphere.mesh.cells_on_edge).reshape(-1, 2)[edge]).astype(np.int64)
            near = any(int(c) < klass.n_owned_cells and bool(klass.near_interface[int(c)]) for c in cells)
            rows.append({"max_abs_u": float(speed[level, edge]), "level": level,
                         "global_edge": int(layout.edge_l2g[edge]),
                         "global_cells": [int(layout.cell_l2g[int(c)]) for c in cells],
                         "near_interface": bool(near)})
        return rows

    def gather(self, names: Sequence[str] = ("rho", "rho_theta", "rho_u", "rho_w")) -> dict[str, np.ndarray]:
        """Owned-union host arrays in global order."""

        cp = self.cp
        result: dict[str, np.ndarray] = {}
        first = self.classes[0].tables.layout
        for name in names:
            edge = name in EDGE_FIELDS
            extent = first._global_edges if edge else first._global_cells
            levels = self.nlev + 1 if name in INTERFACE_FIELDS else self.nlev
            target = np.empty((levels, extent), dtype=np.float32)
            for klass in self.classes:
                layout = klass.tables.layout
                owned = klass.n_owned_edges if edge else klass.n_owned_cells
                l2g = (layout.edge_l2g if edge else layout.cell_l2g)[:owned].astype(np.int64)
                target[:, l2g] = cp.asnumpy(_field(klass.driver.atmosphere, name)[:, :owned])
            result[name] = target
        return result

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "gpuwm-hex.fullstep-lts-active/v2",
            "scope": (("dynamics, scalar transport and per-class column physics, one device, static classes"
                       if self.stepper is not None else
                       "dry dynamics and scalar transport, one device, static classes, no physics")
                      if self.scalars is not None else
                      "dry dynamics, one device, static classes, scalar transport off"),
            "scalar_interface": None if self.scalars is None else self.scalars.stats(),
            "order": "fine first; coarser classes read recorded accepted states of finer classes",
            "interface_flux": self.interface_flux,
            "interface_flux_impositions": [klass.flux_imposed for klass in self.classes],
            "interface_theta_matched": bool(self.match_theta and self.flux_record is not None),
            "interface_theta_impositions": [klass.theta_imposed for klass in self.classes],
            "base_dt": self.base_dt,
            "acoustic_substeps": self.acoustic_substeps,
            "planes_per_step": self.planes_per_step,
            "plan": self.plan.receipt(),
            "events": dict(self.events),
            "class_steps": [klass.steps for klass in self.classes],
            "rounds": [dict(klass.exchanger.rounds) for klass in self.classes],
            "registers_empty": self.registers_empty(),
            "class_wall_seconds": {key: list(values) for key, values in self.wall.items()}
            if self.time_classes else None,
            "step_validation": {"read_at": "macro boundary" if self.deferred_validation else "every class step",
                                "flags_read": int(getattr(self, "validation_flags_read", 0))},
        }


def build_class_drivers(
    host: Mapping[str, Any],
    config: Any,
    plan: ClassPlan,
    *,
    base_dt: float,
    cache_root: Any,
    capability: Any,
) -> tuple[list[Any], list[np.ndarray]]:
    """Slice the prepared host per class and construct one resident driver each."""

    from pathlib import Path

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
    for tables in plan.classes:
        local = slice_prepared_host(bundle, tables.layout)
        cache_dir = Path(cache_root) / f"class{tables.index}-of-{len(plan.classes)}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = KernelCache(capability=capability, cache_dir=str(cache_dir))
        driver = CudaDryDycoreDriver.from_host(
            local["mesh"], local["state"], local["vertical"], local["reference"],
            replace(config, config_dt=float(base_dt) * tables.rate),
            saved_diagnostics=local["saved"], terrain_metrics=local["terrain_metrics"],
            advection_coefficients=local["coefficients"], kernel_cache=cache,
            reference_wind_profiles=host["profiles"])
        drivers.append(driver)
        areas.append(np.asarray(mesh.area_cell, dtype=np.float64)[tables.layout.cell_l2g.astype(np.int64)])
    return drivers, areas
