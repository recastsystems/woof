"""The engine as a PyTorch autograd function, in four forms.

  v0  "window":  one function for the whole window.  The network's inputs are taken as given: the error's sensitivity
                 reaches the weights only through the tendency the network wrote, not through the state it read.
                 Exact while the network's output does not depend on its inputs (iteration 0, and a table always);
                 an approximation after.  Needs one whole reverse that hands out every step's slot cotangent.
  v0i "window, iterated": v0, then `passes` more whole reverses, each seeded with the network's share carried back
                 into the state it read, computed from the previous reverse's slot cotangents.  Pass p is exact for
                 the last p+1 steps; n-1 passes are exact; the miss shrinks by about the size of the network's input
                 Jacobian each pass.  Needs only a whole reverse that takes cotangents at every boundary.
  v0x "window, exact":  as each step is reversed the network's share is added at its boundary before the step before
                 it is reversed.  Exact in one reverse.  Needs reverse_step (a reverse that pauses at boundaries).
  v1  "step":    one function per model step; PyTorch chains them, the network in between, so the flow back through
                 the network's inputs is PyTorch's own.  Exact.  Needs reverse_step.

The forward is the engine's run with a callback: before step k the engine hands the state at boundary k, the
network writes the slot, the engine reads it in place.  What the network's backward needs (keep=, window forms):
  "graph"   the autograd graph of every step's network evaluation (v1 always keeps this)
  "inputs"  a copy of every step's network inputs; the network is evaluated again in the backward
  "engine"  nothing: the inputs are read back from the engine's step store (engine.state_at)
memory_per_window() gives the bytes of each.

Arrays cross between the engine and PyTorch by DLPack, on the same memory.  The bridge clones only what must outlive
the engine's next step (the engine updates its pools in place): the states it returns, keep="inputs", and v1's
per-step states.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .engine import merge, reverse_window, to_engine, to_torch

FORMS = ("v0", "v0i", "v0x", "v1")
KEEPS = ("graph", "inputs", "engine")


@dataclass
class Timings:
    forward: float = 0.0       # the engine's forward (without the network inside it)
    record: float = 0.0        # engine seconds the bridge cannot see (engine.times: recording, sealing, replay)
    reverse: float = 0.0       # the engine's reverse (without the network inside it)
    network: float = 0.0       # network forward and backward
    passes: int = 0            # whole reverses run (v0i runs 1 + passes)
    extra: dict = field(default_factory=dict)

    def line(self):
        return (f"fwd {self.forward:.3f} s | rec {self.record:.3f} s | rev {self.reverse:.3f} s | "
                f"net {self.network:.3f} s")


def _sync(t):
    if t is not None and t.is_cuda:
        torch.cuda.synchronize(t.device)


class _Clock:
    """Wall seconds into timings.<name>, less the network seconds spent inside (the network runs in the engine)."""

    def __init__(self, timings, name, probe=None, less_network=False):
        self.t, self.name, self.probe, self.less = timings, name, probe, less_network

    def __enter__(self):
        _sync(self.probe)
        self.t0, self.n0 = time.perf_counter(), self.t.network

    def __exit__(self, *a):
        _sync(self.probe)
        dt = time.perf_counter() - self.t0
        if self.less:
            dt -= self.t.network - self.n0
        setattr(self.t, self.name, getattr(self.t, self.name) + dt)


class WindowResult:
    """states: {boundary: {name: tensor}}.  attach(loss) ties v1's step chain to the loss (adds an exact zero)."""

    def __init__(self, states, token=None, timings=None):
        self.states, self.token, self.timings = states, token, timings

    def attach(self, loss):
        return loss if self.token is None else loss + self.token


class _Run:
    """One window's forward and backward, shared by the forms."""

    def __init__(self, engine, model, window, boundaries, out_fields, form, keep, passes, timings):
        if form not in FORMS:
            raise ValueError(f"form must be one of {FORMS}")
        if keep not in KEEPS:
            raise ValueError(f"keep must be one of {KEEPS}")
        if keep == "engine" and not hasattr(engine, "state_at"):
            raise ValueError("keep='engine' needs engine.state_at (the step store's named state)")
        if form in ("v0x", "v1") and not hasattr(engine, "reverse_step"):
            raise ValueError(f"form {form} needs engine.reverse_step (a reverse that pauses at each boundary); "
                             "v0i is exact from whole reverses")
        self.engine, self.model, self.window = engine, model, window
        self.n = window.length
        self.boundaries = sorted(set(boundaries))
        self.out_fields = tuple(out_fields)
        self.form, self.keep, self.passes, self.t = form, keep, passes, timings
        self.inputs = tuple(model.input_fields)
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.ncol = engine.ncol
        self.shapes = {}
        self.kept = [None] * self.n
        self.probe = None
        self.live = None

    # ---- arrays across -------------------------------------------------------------------------------------------
    def read(self, state, names, clone):
        out = {}
        for n in names:
            a = state[n]
            self.shapes.setdefault(n, tuple(a.shape))
            t = to_torch(a)
            if t.shape[-1] != self.ncol:
                t = t[..., :self.ncol]
            out[n] = t.clone() if clone else t
        if self.probe is None and out:
            self.probe = next(iter(out.values()))
        return out

    def g_to_engine(self, g_named):
        xp, out = self.engine.xp, {}
        for n, g in g_named.items():
            if g is None:
                continue
            full = self.shapes.get(n)
            if full is not None and full[-1] != g.shape[-1]:
                g = torch.nn.functional.pad(g, (0, full[-1] - g.shape[-1]))
            out[n] = to_engine(g, xp)
        return out

    def slot_to_engine(self, tend):
        self.live = tend                          # alive while the engine reads it during the step
        return {f: to_engine(t, self.engine.xp) for f, t in tend.items()}

    def net(self, inputs):
        with _Clock(self.t, "network", self.probe):
            return self.model(inputs)

    def engine_times(self):
        times = getattr(self.engine, "times", None) or {}
        self.t.record += sum(float(v) for v in times.values())

    # ---- the window forms (v0, v0i, v0x) ------------------------------------------------------------------------
    def window_forward(self):
        outs = {}

        def callback(k, state):
            if k in self.boundaries:
                outs[k] = self.read(state, self.out_fields, clone=True)
            if self.keep == "graph":
                inp = self.read(state, self.inputs, clone=True)
                if self.form != "v0":
                    for v in inp.values():
                        v.requires_grad_(True)
                with torch.enable_grad():
                    tend = self.net(inp)
                self.kept[k] = (inp, tend)
            else:
                inp = self.read(state, self.inputs, clone=(self.keep == "inputs"))
                tend = self.net(inp)
                if self.keep == "inputs":
                    self.kept[k] = inp
            return self.slot_to_engine({f: t.detach() for f, t in tend.items()})

        with _Clock(self.t, "forward", self.probe, less_network=True):
            self.engine.run_forward(self.window, callback, record=True)
        self.live = None
        if self.n in self.boundaries:
            outs[self.n] = self.read(self.engine.state(), self.out_fields, clone=True)
        self.engine_times()
        return [outs[b][f] for b in self.boundaries for f in self.out_fields]

    def _inputs_at(self, k):
        if self.keep == "inputs":
            return self.kept[k]
        if self.keep == "engine":
            return self.read(self.engine.state_at(k), self.inputs, clone=False)
        return self.kept[k][0]

    def network_vjp(self, k, g_slot, params=True, inputs=False, last=True):
        """The network's transpose at step k.  params: add into the weights' gradients.  inputs: return the
        cotangents of the state it read, as engine arrays.  last: the graph (keep="graph") may be dropped."""
        g_t = {f: to_torch(g) for f, g in g_slot.items()}
        with _Clock(self.t, "network", self.probe), torch.enable_grad():
            if self.keep == "graph":
                inp, tend = self.kept[k]
                if last:
                    self.kept[k] = None
            else:
                inp = self._inputs_at(k)
                if inputs:
                    inp = {n: v.detach().requires_grad_(True) for n, v in inp.items()}
                tend = self.model(inp)
            names = [f for f in tend if f in g_t]
            leaves = (self.params if params else []) + ([inp[n] for n in self.inputs] if inputs else [])
            grads = torch.autograd.grad([tend[f] for f in names], leaves, [g_t[f] for f in names],
                                        allow_unused=True, retain_graph=not last)
        if params:
            for acc, g in zip(self.grads, grads[:len(self.params)]):
                if g is not None:
                    acc += g
        if not inputs:
            return None
        g_in = {n: g for n, g in zip(self.inputs, grads[len(self.params) if params else 0:]) if g is not None}
        return self.g_to_engine(g_in) or None

    def _whole_reverse(self, g_by_b):
        with _Clock(self.t, "reverse", self.probe, less_network=True):
            g = reverse_window(self.engine, g_by_b, self.n)
        self.t.passes += 1
        return g

    def window_backward(self, g_outs):
        g_by_b, i = {}, 0
        for b in self.boundaries:
            named = {f: g_outs[i + j] for j, f in enumerate(self.out_fields) if g_outs[i + j] is not None}
            i += len(self.out_fields)
            if named:
                g_by_b[b] = self.g_to_engine(named)
        self.grads = [torch.zeros_like(p) for p in self.params]
        if self.form == "v0x":
            with _Clock(self.t, "reverse", self.probe, less_network=True):
                reverse_window(self.engine, g_by_b, self.n,
                               inject=lambda k, g: self.network_vjp(k, g, params=True, inputs=k > 0))
            self.t.passes += 1
        else:
            g_slots = self._whole_reverse(g_by_b)
            for _ in range(self.passes if self.form == "v0i" else 0):
                inj = {k: self.network_vjp(k, g_slots[k], params=False, inputs=True, last=False)
                       for k in range(1, self.n)}
                g_slots = self._whole_reverse({k: merge(g_by_b.get(k), inj.get(k))
                                               for k in set(g_by_b) | set(inj)})
            for k in range(self.n - 1, -1, -1):
                self.network_vjp(k, g_slots[k], params=True, inputs=False, last=True)
                g_slots[k] = None
        self.kept = [None] * self.n
        return self.grads


class _WindowFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run, *params):
        ctx.run = run
        ctx.set_materialize_grads(False)
        return tuple(run.window_forward())

    @staticmethod
    def backward(ctx, *g_outs):
        return (None, *ctx.run.window_backward(g_outs))


class _StepFn(torch.autograd.Function):
    """Model step k for v1: (token, slot tendencies of step k) -> (token, named state at boundary k+1).

    The engine has already taken the step (inside its forward, through the callback); this node reads the state it
    reached and, backward, reverses that step."""

    @staticmethod
    def forward(ctx, run, token, *tend):
        ctx.run = run
        ctx.set_materialize_grads(False)
        s = run.read(run.pending, run.state_names, clone=True)
        return (token.clone(), *[s[n] for n in run.state_names])

    @staticmethod
    def backward(ctx, g_token, *g_state):
        run = ctx.run
        g_add = run.g_to_engine(dict(zip(run.state_names, g_state)))
        with _Clock(run.t, "reverse", run.probe):
            g_slot = run.engine.reverse_step(g_add or None)
            run.steps_left -= 1
            if run.steps_left == 0:
                run.engine.reverse_end(None)
                run.t.passes += 1
        out = [to_torch(g_slot[f]).clone() for f in run.model.slot_fields]
        return (None, g_token, *out)


def run_window(engine, model, window, boundaries=None, out_fields=None, form="v1", keep="inputs", passes=2,
               timings=None):
    """Run one window with the model's tendency in the slot; returns a WindowResult whose states carry gradients.

    boundaries: the step boundaries (1..length) whose state the loss reads (default: every one).
    out_fields: the named fields returned at those boundaries (default: the slot fields).
    passes: v0i's extra whole reverses.
    """
    t = timings if timings is not None else Timings()
    boundaries = list(range(1, window.length + 1)) if boundaries is None else list(boundaries)
    out_fields = tuple(out_fields or model.slot_fields)
    run = _Run(engine, model, window, boundaries, out_fields, form, keep, passes, t)
    if form != "v1":
        flat = _WindowFn.apply(run, *run.params)
        states, i = {}, 0
        for b in run.boundaries:
            states[b] = {f: flat[i + j] for j, f in enumerate(out_fields)}
            i += len(out_fields)
        return WindowResult(states, None, t)

    # v1: a step node per model step, made in the callback as the engine reaches each boundary
    run.state_names = tuple(dict.fromkeys(run.inputs + out_fields))
    run.steps_left = window.length
    chain = {"token": None, "tend": None}
    states = {}

    def node(k, state):
        if k == 0:
            cur = run.read(state, run.state_names, clone=True)
            p = next(iter(cur.values()))
            chain["token"] = torch.zeros((), dtype=p.dtype, device=p.device)
            return cur
        run.pending = state
        outs = _StepFn.apply(run, chain["token"], *[chain["tend"][f] for f in model.slot_fields])
        run.pending = None
        chain["token"] = outs[0]
        cur = dict(zip(run.state_names, outs[1:]))
        if k in run.boundaries:
            states[k] = {f: cur[f] for f in out_fields}
        return cur

    def callback(k, state):
        cur = node(k, state)
        chain["tend"] = run.net({n: cur[n] for n in run.inputs})
        return run.slot_to_engine(chain["tend"])

    with _Clock(t, "forward", None, less_network=True):
        engine.run_forward(window, callback, record=True)
    node(window.length, engine.state())
    run.live = None
    run.engine_times()
    return WindowResult(states, chain["token"], t)


@torch.no_grad()
def forward_only(engine, model, window, out_fields=None, boundaries=None):
    """The window's forecast with the model in the slot and no recording: for held-out scores and plain runs."""
    out_fields = tuple(out_fields or model.slot_fields)
    boundaries = set(range(1, window.length + 1) if boundaries is None else boundaries)
    run = _Run(engine, model, window, [], out_fields, "v0", "inputs", 0, Timings())
    states = {}

    def callback(k, state):
        if k in boundaries:
            states[k] = run.read(state, out_fields, clone=True)
        return run.slot_to_engine(model(run.read(state, run.inputs, clone=False)))

    engine.run_forward(window, callback, record=False)
    if window.length in boundaries:
        states[window.length] = run.read(engine.state(), out_fields, clone=True)
    return states


def saved_bytes_per_column(model, sample_inputs):
    """Bytes autograd keeps for one network evaluation, per column (counted on the given inputs)."""
    total = [0]

    def pack(t):
        total[0] += t.numel() * t.element_size()
        return t

    inp = {n: v.detach().clone() for n, v in sample_inputs.items()}
    with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        model(inp)
    ncol = next(iter(inp.values())).shape[-1]
    return total[0] / ncol


def memory_per_window(model, sample_inputs, ncol, n_steps):
    """Bytes each keep= choice holds for one window of n_steps over ncol columns (the network's side only)."""
    per_col_graph = saved_bytes_per_column(model, sample_inputs)
    s = next(iter(sample_inputs.values()))
    per_col_in = sum(v.numel() // v.shape[-1] for v in sample_inputs.values()) * s.element_size()
    return {"graph": per_col_graph * ncol * n_steps, "inputs": per_col_in * ncol * n_steps, "engine": 0.0}
