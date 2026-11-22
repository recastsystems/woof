"""The toy proofs, as one command: python -m woof.hex.torch_bridge.proofs [--device cuda] [--json f]

Each proof prints one line and goes in the returned dict:
  reverse    the toy's hand-written reverse against central differences of the forward (slot sensitivity)
  zero       a zero-initialised model in the slot gives the engine's own run, byte for byte (float64 and float32)
  grad       each form's window gradient against central differences, network non-zero; v0's miss against how far
             the network has moved from zero; v0i's miss against the number of passes
  v0_zero    v0 at iteration 0 (network output zero) against v1: the same gradient
  keep       the three keep= choices give the same gradient
  table      a table learns back a planted bias per level and field
  net        a column network learns a planted state-dependent term (held-out loss before and after)
  memory     bytes each keep= choice holds for a window, for a given shape
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .autograd import memory_per_window, run_window
from .engine import WindowSpec, to_torch
from .losses import MultiScaleLoss, NeighbourSmoother, ScaledMSE
from .models import TableCorrection, column_net_for
from .toy import PROG, ToyCases
from .trainer import Trainer

SCALES = {"theta": 1.0, "qv": 1e-3, "u": 0.1, "v": 0.1}
CAPS = {"theta": 0.2, "qv": 5e-5, "u": 0.01, "v": 0.01}
UNITS = {"theta": 1e-2, "qv": 1e-5, "u": 1e-3, "v": 1e-3}
INPUTS = ("theta", "qv", "u", "v", "p", "ts")
NZ = 5


def _xp(device):
    if device == "cuda":
        import cupy
        return cupy
    return np


def _cases(device="cpu", dtype=np.float64, **kw):
    base = dict(n_cases=3, n_steps=12, seed=1, ncol=12, nz=NZ, pad=1, xp=_xp(device), dtype=dtype)
    base.update(kw)
    return ToyCases(**base)


def _to(model, engine):
    s = to_torch(engine.p)
    return model.to(device=s.device, dtype=s.dtype)


def _net(engine, hidden=(16,), seed=0, nonzero=0.0):
    torch.manual_seed(seed)
    engine.begin(WindowSpec(0, 0, 1), record=False)
    s = {n: to_torch(a)[..., :engine.ncol].clone() for n, a in engine.state().items()}
    m = _to(column_net_for(s, engine.nz, CAPS, INPUTS, hidden=hidden), engine)
    m.fit_normaliser([s])
    if nonzero:
        with torch.no_grad():
            m.head.weight.normal_(0.0, nonzero)
            m.head.bias.normal_(0.0, nonzero)
    return m


def proof_reverse(device="cpu"):
    """Toy reverse against central differences: d<w, x_n>/de along a random direction in every step's slot."""
    eng = _cases(device).model()
    xp = eng.xp
    rng = np.random.default_rng(3)
    n = 6
    w = {f: xp.asarray(rng.standard_normal((eng.nz, eng.ncol))) for f in PROG}
    d = [{f: xp.asarray(rng.standard_normal((eng.nz, eng.ncol))) * CAPS[f] for f in PROG} for _ in range(n)]
    win = WindowSpec(0, 0, n)

    def run(eps):
        eng.run_forward(win, lambda k, s: {f: eps * d[k][f] for f in PROG}, record=True)
        return {f: eng.state()[f][:, :eng.ncol].copy() for f in PROG}

    run(0.0)
    g_slots = eng.reverse_window({n: w})
    adj = sum(float((g_slots[k][f] * d[k][f]).sum()) for k in range(n) for f in PROG)
    eps = 1e-3
    plus, minus = run(eps), run(-eps)      # difference the states first: the sum of w * x is large, its change small
    fd = sum(float((w[f] * (plus[f] - minus[f])).sum()) for f in PROG) / (2 * eps)
    return {"adjoint": adj, "fd": fd, "rel": abs(adj - fd) / abs(fd)}


def proof_zero(device="cpu", dtype=np.float64):
    """The zero-initialised network in the slot against no callback at all: every step's state, byte for byte."""
    cases = _cases(device, dtype)
    a, b = cases.model(), cases.model()
    m = _net(a)
    win = WindowSpec(1, 2, 8)
    seen = {"a": [], "b": []}

    def cb(tag, model=None):
        def f(k, s):
            seen[tag].append({x: s[x].copy() for x in PROG})
            if model is None:
                return None
            out = model({n: to_torch(s[n])[..., :a.ncol] for n in model.input_fields})
            return {x: a.xp.from_dlpack(t.detach()) for x, t in out.items()}
        return f

    with torch.no_grad():
        a.run_forward(win, cb("a", m), record=False)
    b.run_forward(win, cb("b"), record=False)
    same = all(a.xp.array_equal(p[x], q[x]) for p, q in zip(seen["a"], seen["b"]) for x in PROG)
    same &= all(a.xp.array_equal(a.state()[x], b.state()[x]) for x in PROG)
    return {"bitwise_equal_all_steps": bool(same), "steps": win.length, "dtype": np.dtype(dtype).name}


def _loss(cases, model, eng, win, form, keep="inputs", passes=2):
    tg = {b: {f: to_torch(v[f]) for f in PROG} for b, v in cases.targets(win).items()}
    res = run_window(eng, model, win, form=form, keep=keep, passes=passes)
    return res.attach(ScaledMSE(SCALES)(res.states, tg))


def _grad(cases, model, eng, win, form, keep="inputs", passes=2):
    model.zero_grad(set_to_none=True)
    _loss(cases, model, eng, win, form, keep, passes).backward()
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()])


def _fd_dir(cases, model, eng, win, d, eps=1e-6):
    params = list(model.parameters())
    p0 = [p.detach().clone() for p in params]

    def at(s):
        with torch.no_grad():
            i = 0
            for p, q in zip(params, p0):
                p.copy_(q + s * d[i:i + p.numel()].reshape(p.shape))
                i += p.numel()
            return float(_loss(cases, model, eng, win, "v0"))

    fd = (at(eps) - at(-eps)) / (2 * eps)
    at(0.0)
    return fd


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def proof_grad(device="cpu", nonzero=0.05):
    """Window gradients against central differences (3 random directions) and against v1."""
    cases = _cases(device)
    eng = cases.model()
    model = _net(eng, nonzero=nonzero)
    win = WindowSpec(0, 1, 8)
    g = {f: _grad(cases, model, eng, win, f) for f in ("v1", "v0x", "v0i", "v0")}
    gen = torch.Generator().manual_seed(5)
    dirs = [torch.randn(g["v1"].numel(), generator=gen, dtype=g["v1"].dtype).to(g["v1"].device) for _ in range(3)]
    fds = [_fd_dir(cases, model, eng, win, d) for d in dirs]
    out = {}
    for form in g:
        out[form] = {"fd_rel_max": max(abs(float(g[form] @ d) - fd) / abs(fd) for d, fd in zip(dirs, fds))}
        if form != "v1":
            out[form]["rel_to_v1"] = _rel(g[form], g["v1"])
    out["v0i_by_passes"] = [{"passes": p, "rel_to_v1": _rel(_grad(cases, model, eng, win, "v0i", passes=p), g["v1"])}
                            for p in (0, 1, 2, 3, win.length - 1)]
    out["v0_by_size"] = []
    for size in (0.005, 0.05, 0.5):     # v0's miss against how far the network has moved from zero
        m = _net(eng, nonzero=size)
        g1, g0 = _grad(cases, m, eng, win, "v1"), _grad(cases, m, eng, win, "v0")
        g2 = _grad(cases, m, eng, win, "v0i", passes=2)
        eng.begin(win, record=False)
        with torch.no_grad():
            tend = m({n: to_torch(a)[..., :eng.ncol] for n, a in eng.state().items() if n in m.input_fields})
        frac = max(float((tend[f].abs() / CAPS[f]).max()) for f in tend)
        out["v0_by_size"].append({"head_std": size, "max_output_over_cap": frac, "v0_rel_to_v1": _rel(g0, g1),
                                  "v0i2_rel_to_v1": _rel(g2, g1)})
    return out


def proof_v0_zero(device="cpu"):
    cases = _cases(device, planted_relax=(30.0, 1.5 * np.arange(NZ) + 0.8))
    eng = cases.model()
    model = _net(eng, nonzero=0.0)
    win = WindowSpec(2, 0, 8)
    g1, g0 = _grad(cases, model, eng, win, "v1"), _grad(cases, model, eng, win, "v0")
    return {"rel_v0_to_v1": _rel(g0, g1), "grad_norm": float(g1.norm())}


def proof_keep(device="cpu"):
    cases = _cases(device)
    eng = cases.model()
    model = _net(eng, nonzero=0.05)
    win = WindowSpec(1, 0, 6)
    out = {}
    for form in ("v0", "v0i", "v0x"):
        gs = {k: _grad(cases, model, eng, win, form, keep=k) for k in ("graph", "inputs", "engine")}
        out[form] = max(float((gs[k] - gs["graph"]).abs().max()) for k in gs)
    return {"max_abs_diff": out}


def proof_table(device="cpu", form="v0", steps=150, log=None, dtype=np.float64):
    """A table recovers a planted bias per level and field (the model lacks it; the truth has it)."""
    z = np.arange(NZ)
    bias = {"theta": 2e-2 * np.cos(z), "qv": 2e-5 * np.sin(z + 1), "u": 4e-3 * (1 - z / 4), "v": -3e-3 * np.ones(NZ)}
    cases = _cases(device, dtype, n_cases=4, n_steps=20, planted_bias=bias)
    eng = cases.model()
    model = _to(TableCorrection(NZ, units=UNITS), eng)
    held = cases.windows([3], length=6, stride=7)
    tr = Trainer(eng, model, cases, ScaledMSE(SCALES), form=form, lr=0.05, clip=10.0, held_out=held,
                 eval_every=50, log=log or (lambda s: None))
    tr.fit(cases.windows([0, 1, 2], length=6, stride=2), steps)
    truth = torch.tensor(np.stack([bias[f] for f in model.slot_fields]), dtype=torch.float64)
    learned = (model.table * model.units[:, None]).detach().cpu().double()
    rel = float((learned - truth).norm(dim=1).div(truth.norm(dim=1)).max())
    return {"rel_err_worst_field": rel, "held_out": tr.eval_history, "steps": steps,
            "first_loss": tr.history[0]["loss"], "last_loss": tr.history[-1]["loss"]}


def proof_net(device="cpu", form="v1", steps=120, log=None, multiscale=False, dtype=np.float64):
    """A column network learns a planted relaxation of theta to a profile (state-dependent: a table cannot)."""
    prof = 1.5 * np.arange(NZ) + 0.8
    cases = _cases(device, dtype, n_cases=4, n_steps=20, planted_relax=(30.0, prof))
    eng = cases.model()
    model = _net(eng, hidden=(32, 32), seed=0)
    held = cases.windows([3], length=6, stride=7)
    loss = (MultiScaleLoss(SCALES, NeighbourSmoother(eng.neighbours(), device=to_torch(eng.p).device),
                           passes=(0, 2, 6)) if multiscale else ScaledMSE(SCALES))
    tr = Trainer(eng, model, cases, loss, form=form, lr=2e-3, clip=1.0, held_out=held, eval_every=40,
                 log=log or (lambda s: None))
    tr.fit(cases.windows([0, 1, 2], length=6, stride=2), steps)
    return {"held_out": tr.eval_history, "first_loss": tr.history[0]["loss"], "last_loss": tr.history[-1]["loss"],
            "last_line": {k: tr.history[-1][k] for k in ("forward", "record", "reverse", "network")}}


def proof_memory(ncol=16933, nz=59, steps=24, hidden=(128, 128), n_surface=6):
    """Bytes held per window by each keep= choice for a ColumnNet reading theta, qv, both winds and pressure per
    level and n_surface surface fields, float32, as GB."""
    s = {n: torch.zeros(nz, 64) for n in ("theta", "qv", "u_zonal", "u_meridional", "pressure")}
    s.update({f"sfc{i}": torch.zeros(64) for i in range(n_surface)})
    m = column_net_for(s, nz, CAPS, tuple(s), hidden=hidden)
    mem = memory_per_window(m, s, ncol, steps)
    return {"shape": {"ncol": ncol, "nz": nz, "steps": steps, "hidden": list(hidden),
                      "params": sum(p.numel() for p in m.parameters())},
            "GB": {k: v / 1e9 for k, v in mem.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--json")
    ap.add_argument("--ncol", type=int, default=16933)
    ap.add_argument("--nz", type=int, default=59)
    ap.add_argument("--steps", type=int, default=24)
    a = ap.parse_args(argv)
    out = {}
    for name, fn in (("reverse", proof_reverse), ("zero", proof_zero), ("grad", proof_grad),
                     ("v0_zero", proof_v0_zero), ("keep", proof_keep), ("table", proof_table),
                     ("net", proof_net)):
        out[name] = fn(a.device)
        print(name, json.dumps(out[name], default=str), flush=True)
    out["zero_f32"] = proof_zero(a.device, np.float32)
    print("zero_f32", json.dumps(out["zero_f32"]))
    out["memory"] = proof_memory(a.ncol, a.nz, a.steps)
    print("memory", json.dumps(out["memory"]))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1, default=str)
    return out


if __name__ == "__main__":
    main()
