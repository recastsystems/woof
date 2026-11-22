"""Losses over a window's states against targets: plain scaled squares, and the multi-scale form.

A loss takes states and targets as {boundary: {field: tensor (..., ncol)}} (level-major, columns last) and returns
a scalar.  Each field's error
is divided by its scale (so kelvin, kg/kg and m/s count alike), squared and averaged over columns, levels, fields and
boundaries.
"""
from __future__ import annotations

import torch


class ScaledMSE:
    def __init__(self, scales):
        self.scales = dict(scales)

    def __call__(self, states, targets):
        terms = []
        for b, fields in states.items():
            for f, x in fields.items():
                e = (x - targets[b][f]) / self.scales[f]
                terms.append((e * e).mean())
        return torch.stack(terms).mean()


class NeighbourSmoother:
    """Repeated averaging of each column with its neighbours; `passes` sets the length (about sqrt(passes) cells).

    neighbours: (ncol, m) int array, -1 where a column has fewer than m (the mesh's cellsOnCell, zero-based).
    """

    def __init__(self, neighbours, device=None):
        nb = torch.as_tensor(neighbours, dtype=torch.long, device=device)
        self_idx = torch.arange(nb.shape[0], device=nb.device)[:, None].expand_as(nb)
        valid = nb >= 0
        self.idx = torch.where(valid, nb, self_idx)
        self.w = valid.to(torch.float64)
        self.norm = 1.0 + self.w.sum(1)

    def __call__(self, x, passes):
        """x: (..., ncol), columns last."""
        if passes == 0:
            return x
        w = self.w.to(x)
        norm = self.norm.to(x)
        for _ in range(passes):
            s = x
            for j in range(self.idx.shape[1]):
                s = s + w[:, j] * x[..., self.idx[:, j]]
            x = s / norm
        return x


class MultiScaleLoss:
    """The error smoothed to several lengths before squaring: large scales count even where small ones are noise.

    passes: smoothing passes per scale (0 = the error itself); weights: one per scale (default equal).
    """

    def __init__(self, scales, smoother, passes=(0, 2, 8), weights=None):
        self.scales = dict(scales)
        self.smoother = smoother
        self.passes = tuple(passes)
        self.weights = tuple(weights) if weights is not None else (1.0 / len(self.passes),) * len(self.passes)

    def __call__(self, states, targets):
        terms = []
        for b, fields in states.items():
            for f, x in fields.items():
                e = (x - targets[b][f]) / self.scales[f]
                terms.append(sum(w * (self.smoother(e, p) ** 2).mean() for w, p in zip(self.weights, self.passes)))
        return torch.stack(terms).mean()
