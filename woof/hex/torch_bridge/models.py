"""The correction models.  Each maps the named state at a boundary to the slot's tendencies, and starts at zero.

Both take {name: tensor}, level-major as the engine gives it (a per-level field (nz, ncol), a surface field (ncol,)),
and return {slot field: tensor (nz, ncol), C-contiguous} in units per second.  Both start with output exactly zero
for every input, so iteration 0 is the engine's own run, and the Jacobian of the output in the inputs is zero there
too (form v0 is exact at iteration 0).
"""
from __future__ import annotations

import torch
from torch import nn

from .engine import SLOT_FIELDS


class TableCorrection(nn.Module):
    """One number per level per field: the smallest thing that can learn (len(fields) x nz parameters).

    units: {field: tendency per second that one table unit means}, so one learning rate fits every field.
    Its output does not depend on the state, so form v0 is exact for it at every iteration.
    """

    def __init__(self, nz, slot_fields=SLOT_FIELDS, units=None, ref_field="theta"):
        super().__init__()
        self.slot_fields = tuple(slot_fields)
        self.input_fields = (ref_field,)          # read only for the column count, device and type
        self.table = nn.Parameter(torch.zeros(len(self.slot_fields), nz))
        self.register_buffer("units", torch.tensor([float((units or {}).get(f, 1.0)) for f in self.slot_fields]))

    def forward(self, state):
        ncol = state[self.input_fields[0]].shape[-1]
        rows = self.table * self.units[:, None]
        return {f: rows[i][:, None].expand(-1, ncol).contiguous() for i, f in enumerate(self.slot_fields)}


class ColumnNet(nn.Module):
    """A small network per column: normalised column in, bounded tendencies of the slot fields per level out.

    in_fields: {name: levels} (levels 1 for a surface field).  caps: {slot field: largest |tendency| per second};
    the output is cap * tanh(raw): the network works in units of each field's cap and cannot push a field faster
    than its cap whatever its weights.  The last layer starts at zero (weights and bias), so the output and its
    Jacobian in the inputs start at zero.  The head is applied transposed, so each field's output is a contiguous
    (nz, ncol) block the engine reads in place.
    """

    def __init__(self, in_fields, nz, caps, hidden=(128, 128), slot_fields=SLOT_FIELDS):
        super().__init__()
        self.in_spec = dict(in_fields)
        self.input_fields = tuple(self.in_spec)
        self.slot_fields = tuple(slot_fields)
        self.nz = nz
        nin = sum(self.in_spec.values())
        self.register_buffer("mean", torch.zeros(nin))
        self.register_buffer("scale", torch.ones(nin))
        self.register_buffer("cap", torch.tensor([float(caps[f]) for f in self.slot_fields]).repeat_interleave(nz))
        layers, w = [], nin
        for h in hidden:
            layers += [nn.Linear(w, h), nn.SiLU()]
            w = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(w, len(self.slot_fields) * nz)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def columns(self, state):
        """(ncol, nin): each column's inputs side by side."""
        parts = []
        for name, levels in self.in_spec.items():
            a = state[name]
            parts.append(a.reshape(levels, a.shape[-1]))
        return torch.cat(parts, dim=0).t()

    @torch.no_grad()
    def fit_normaliser(self, states):
        """Mean and spread per input entry from one or more named states (a list of dicts)."""
        cols = torch.cat([self.columns(s) for s in states], dim=0).to(self.mean)
        mean, spread = cols.mean(0), cols.std(0)
        fallback = mean.abs().clamp_min(1.0)               # an entry that does not vary: scale by its size
        self.mean.copy_(mean)
        self.scale.copy_(torch.where(spread > 1e-6 * fallback, spread, fallback))

    def forward(self, state):
        h = self.body((self.columns(state) - self.mean) / self.scale)          # (ncol, hidden)
        raw = torch.addmm(self.head.bias[:, None], self.head.weight, h.t())    # (fields * nz, ncol), contiguous
        out = self.cap[:, None] * torch.tanh(raw)
        return {f: out[i * self.nz:(i + 1) * self.nz] for i, f in enumerate(self.slot_fields)}


def column_net_for(state, nz, caps, in_fields, hidden=(128, 128), slot_fields=SLOT_FIELDS):
    """A ColumnNet reading the named fields of a sample state (per-level or surface, told by their rank)."""
    spec = {n: (state[n].shape[0] if state[n].ndim == 2 else 1) for n in in_fields}
    return ColumnNet(spec, nz, caps, hidden=hidden, slot_fields=slot_fields)
