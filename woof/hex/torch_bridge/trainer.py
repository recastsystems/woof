"""Training a correction through the engine: windows as a dataset, AdamW, clipping, checkpoints, held-out scores.

cases: an object with targets(window) -> {boundary: {field: engine array}}; the engine's begin(window) loads the
window's start.  One log line per step: loss, gradient norm, and the seconds in forward, record, reverse, network.
"""
from __future__ import annotations

import os
import random
import time

import torch

from .autograd import Timings, forward_only, run_window
from .engine import to_torch


class Trainer:
    def __init__(self, engine, model, cases, loss, *, form="v1", keep="inputs", passes=2, out_fields=None, lr=1e-3,
                 weight_decay=0.0, clip=1.0, held_out=(), eval_every=0, ckpt_path=None, ckpt_every=0, log=print):
        self.engine, self.model, self.cases, self.loss = engine, model, cases, loss
        self.form, self.keep, self.passes = form, keep, passes
        self.out_fields = tuple(out_fields or model.slot_fields)
        self.opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                                     weight_decay=weight_decay)
        self.clip = clip
        self.held_out = list(held_out)
        self.eval_every, self.ckpt_path, self.ckpt_every = eval_every, ckpt_path, ckpt_every
        self.log = log
        self.step_no = 0
        self.history = []
        self.eval_history = []

    def targets(self, window):
        return {b: {f: to_torch(a[f]) for f in self.out_fields} for b, a in self.cases.targets(window).items()}

    def train_step(self, window):
        t = Timings()
        t0 = time.perf_counter()
        self.opt.zero_grad(set_to_none=True)
        res = run_window(self.engine, self.model, window, out_fields=self.out_fields, form=self.form,
                         keep=self.keep, passes=self.passes, timings=t)
        loss = self.loss(res.states, self.targets(window))
        res.attach(loss).backward()
        params = [p for p in self.model.parameters() if p.requires_grad]
        gnorm = torch.nn.utils.clip_grad_norm_(params, self.clip if self.clip else float("inf"))
        self.opt.step()
        self.step_no += 1
        rec = {"step": self.step_no, "loss": float(loss.detach()), "gnorm": float(gnorm), "forward": t.forward,
               "record": t.record, "reverse": t.reverse, "network": t.network,
               "wall": time.perf_counter() - t0, "window": window}
        self.history.append(rec)
        self.log(f"step {self.step_no} | loss {rec['loss']:.4e} | gnorm {rec['gnorm']:.3e} | {t.line()} | "
                 f"window {window.case}@{window.start}+{window.length}")
        return rec

    def evaluate(self):
        if not self.held_out:
            return None
        total = 0.0
        for w in self.held_out:
            states = forward_only(self.engine, self.model, w, out_fields=self.out_fields)
            total += float(self.loss(states, self.targets(w)))
        score = total / len(self.held_out)
        self.eval_history.append((self.step_no, score))
        self.log(f"eval step {self.step_no} | held-out loss {score:.4e} over {len(self.held_out)} windows")
        return score

    def fit(self, windows, n_steps, seed=0):
        rng = random.Random(seed)
        order = []
        if self.eval_every:
            self.evaluate()
        for _ in range(n_steps):
            if not order:
                order = list(windows)
                rng.shuffle(order)
            self.train_step(order.pop())
            if self.eval_every and self.step_no % self.eval_every == 0:
                self.evaluate()
            if self.ckpt_path and self.ckpt_every and self.step_no % self.ckpt_every == 0:
                self.save(self.ckpt_path)
        if self.ckpt_path:
            self.save(self.ckpt_path)
        return self.history

    def save(self, path):
        tmp = f"{path}.tmp"
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(), "step": self.step_no,
                    "form": self.form, "eval": self.eval_history}, tmp)
        os.replace(tmp, path)

    def load(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.step_no = ck["step"]
        self.eval_history = list(ck.get("eval", []))
        return ck
