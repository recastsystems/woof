"""A stand-in engine with the bridge's interface: four fields on a ring of columns, forward and exact reverse by hand.

The tendency is advection by the zonal wind along the ring (nonlinear: u carries theta, qv, u and v), horizontal and
vertical diffusion, Coriolis turning and a linear drag; one step is forward Euler,
    x_{k+1} = x_k + dt * (T(x_k) + e_k),
with e_k the external tendency slot.  The reverse is the hand-written transpose of that step.  Pressure and a
surface temperature are fixed named fields, so a network reads more than the four prognostic fields.

As the real engine: arrays are level-major (nz, ncol); the state is updated IN PLACE (a caller that keeps a view
without cloning it is caught); with pad=1 every state array is one column wider than the slot (a limited-area pool's
unused last column, here filled with a huge value so a network that read it would show it).

ToyCases makes "truth" trajectories with planted terms the model lacks (a bias per level and field, a relaxation of
theta to a profile) to learn back.  Arrays are numpy or cupy (xp), float64 or float32.
"""
from __future__ import annotations

import numpy as np

from .engine import WindowSpec

PROG = ("theta", "qv", "u", "v")


class ToyColumns:
    slot_fields = PROG

    def __init__(self, ncol=16, nz=6, dt=1.0, kh=0.08, kv=0.1, f=0.05, drag=0.01, xp=np, dtype=np.float64,
                 cases=None, pad=0, planted_bias=None, planted_relax=None):
        self.xp, self.dtype = xp, dtype
        self.ncol, self.nz, self.dt, self.pad = ncol, nz, dt, pad
        self.kh, self.kv, self.f, self.drag = kh, kv, f, drag
        self.cases = cases
        # planted terms exist only in a truth model; a model carrying them has no reverse
        self.planted_bias = planted_bias          # {field: (nz,) per second}
        self.planted_relax = planted_relax        # (tau seconds, theta profile (nz,))
        w = ncol + pad
        z = xp.arange(nz, dtype=dtype)[:, None]
        self.p = xp.ascontiguousarray(xp.broadcast_to(1000.0 * xp.exp(-z / 8.0), (nz, w)).astype(dtype))
        i = xp.arange(w, dtype=dtype)
        self.ts = (290.0 + 3.0 * xp.cos(2 * np.pi * i / ncol)).astype(dtype)
        if pad:
            self.p[:, ncol:] = 1e30
            self.ts[ncol:] = 1e30
        self.times = {}
        self._full = None

    # ---- the interface ----------------------------------------------------------------------------------------
    def neighbours(self):
        i = np.arange(self.ncol)
        return np.stack([(i - 1) % self.ncol, (i + 1) % self.ncol], axis=1)

    def run_forward(self, window, callback=None, record=True):
        self.begin(window, record=record)
        for k in range(window.length):
            self.step(callback(k, self.state()) if callback is not None else None)

    def state(self):
        return {**self._full, "p": self.p, "ts": self.ts}

    def state_at(self, k):
        return {**self._store[k], "p": self.p, "ts": self.ts}

    def reverse_window(self, g_by_boundary):
        out = [None] * self._k
        for k in range(self._k - 1, -1, -1):
            out[k] = self.reverse_step(g_by_boundary.get(k + 1))
        self.reverse_end(g_by_boundary.get(0))
        return out

    def reverse_step(self, g_add=None):
        if self.planted_bias or self.planted_relax:
            raise RuntimeError("a truth model with planted terms has no reverse")
        if not self._record:
            raise RuntimeError("reverse of a forward run without record=True")
        if self._rev is None:
            self._rev = {f_: self.xp.zeros((self.nz, self.ncol), self.dtype) for f_ in PROG}
            self._kr = self._k
        self._add(g_add)
        k = self._kr - 1
        g = self._rev
        g_slot = {f_: self.dt * g[f_] for f_ in self.slot_fields}
        x = {f_: a[:, :self.ncol] for f_, a in self._store[k].items()}
        v = self._vjp(x, g)
        self._rev = {f_: g[f_] + self.dt * v[f_] for f_ in PROG}
        self._kr = k
        return g_slot

    def reverse_end(self, g_add=None):
        self._add(g_add)
        out, self._rev = self._rev, None
        return out

    # ---- stepping (also usable directly) ------------------------------------------------------------------------
    def begin(self, window=None, record=True, x0=None):
        if x0 is None:
            x0 = self.cases.start(window)
        w = self.ncol + self.pad
        self._full = {}
        for f_ in PROG:
            a = self.xp.full((self.nz, w), 1e30 if self.pad else 0.0, dtype=self.dtype)
            a[:, :self.ncol] = self.xp.asarray(x0[f_], dtype=self.dtype)[:, :self.ncol]
            self._full[f_] = a
        self._store, self._k, self._record, self._rev = [], 0, record, None

    def step(self, tendency=None):
        x = {f_: a[:, :self.ncol] for f_, a in self._full.items()}
        if self._record:
            self._store.append({f_: a.copy() for f_, a in self._full.items()})
        t = self._tendency(x)
        if self.planted_bias:
            for f_, b in self.planted_bias.items():
                t[f_] = t[f_] + self.xp.asarray(b, dtype=self.dtype)[:, None]
        if self.planted_relax:
            tau, prof = self.planted_relax
            t["theta"] = t["theta"] - (x["theta"] - self.xp.asarray(prof, dtype=self.dtype)[:, None]) / tau
        if tendency:
            for f_, e in tendency.items():
                if e is not None:
                    t[f_] = t[f_] + e
        for f_ in PROG:
            x[f_] += self.dt * t[f_]          # in place, into the pool, as the real engine's buffers
        self._k += 1

    # ---- the model ----------------------------------------------------------------------------------------------
    def _add(self, g_add):
        if g_add:
            for f_, g in g_add.items():
                if f_ in PROG:                     # p and ts are constants here: their cotangent ends
                    self._rev[f_] = self._rev[f_] + g[:, :self.ncol]

    def _dx(self, a):
        return 0.5 * (self.xp.roll(a, -1, axis=1) - self.xp.roll(a, 1, axis=1))

    def _lx(self, a):
        return self.xp.roll(a, -1, axis=1) - 2.0 * a + self.xp.roll(a, 1, axis=1)

    def _lz(self, a):
        d = a[1:, :] - a[:-1, :]
        out = self.xp.zeros_like(a)
        out[:-1, :] += d
        out[1:, :] -= d
        return out

    def _diff(self, a):
        return self.kh * self._lx(a) + self.kv * self._lz(a)

    def _tendency(self, x):
        th, q, u, v = (x[f_] for f_ in PROG)
        return {
            "theta": -u * self._dx(th) + self._diff(th),
            "qv": -u * self._dx(q) + self._diff(q),
            "u": self.f * v - self.drag * u - u * self._dx(u) + self._diff(u),
            "v": -self.f * u - self.drag * v - u * self._dx(v) + self._diff(v),
        }

    def _vjp(self, x, g):
        """Transpose of the tendency's Jacobian at x applied to g (Dx is antisymmetric, Lx and Lz symmetric)."""
        th, q, u, v = (x[f_] for f_ in PROG)
        gt, gq, gu, gv = (g[f_] for f_ in PROG)
        dx = self._dx
        return {
            "theta": dx(u * gt) + self._diff(gt),
            "qv": dx(u * gq) + self._diff(gq),
            "u": (-gt * dx(th) - gq * dx(q) - gu * dx(u) + dx(u * gu) - gv * dx(v)
                  - self.drag * gu - self.f * gv + self._diff(gu)),
            "v": self.f * gu - self.drag * gv + dx(u * gv) + self._diff(gv),
        }


class ToyCases:
    """Truth trajectories from a ToyColumns carrying planted terms; the model to train is a plain ToyColumns."""

    def __init__(self, n_cases=4, n_steps=40, seed=0, planted_bias=None, planted_relax=None, **model_kw):
        self.model_kw = model_kw
        kw = {k: v for k, v in model_kw.items() if k != "pad"}
        truth = ToyColumns(planted_bias=planted_bias, planted_relax=planted_relax, **kw)
        ncol, nz = truth.ncol, truth.nz
        rng = np.random.default_rng(seed)
        i = np.arange(ncol)[None, :]
        z = np.arange(nz)[:, None]
        self.traj = {}
        for c in range(n_cases):
            ph = rng.uniform(0, 2 * np.pi, 4)
            k = rng.integers(1, 3, 4)
            x0 = {   # theta as an anomaly: the toy sees only its differences, and small numbers keep differences exact
                "theta": 1.5 * z + 2.0 * np.sin(2 * np.pi * k[0] * i / ncol + ph[0]) * np.exp(-z / 6),
                "qv": 0.012 * np.exp(-z / 4) * (1.0 + 0.3 * np.sin(2 * np.pi * k[1] * i / ncol + ph[1])),
                "u": 0.3 + 0.1 * np.sin(2 * np.pi * k[2] * i / ncol + ph[2]) + 0.02 * z,
                "v": 0.1 * np.cos(2 * np.pi * k[3] * i / ncol + ph[3]) + 0.0 * z,
            }
            truth.begin(x0=x0, record=False)
            states = [{f_: truth.state()[f_].copy() for f_ in PROG}]
            for _ in range(n_steps):
                truth.step(None)
                states.append({f_: truth.state()[f_].copy() for f_ in PROG})
            self.traj[c] = states

    def model(self, **kw):
        return ToyColumns(cases=self, **{**self.model_kw, **kw})

    def start(self, w: WindowSpec):
        return self.traj[w.case][w.start]

    def targets(self, w: WindowSpec):
        return {b: self.traj[w.case][w.start + b] for b in range(1, w.length + 1)}

    def windows(self, cases, length, stride):
        n = len(self.traj[next(iter(self.traj))]) - 1
        return [WindowSpec(c, s, length) for c in cases for s in range(0, n - length + 1, stride)]
