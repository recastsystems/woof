"""The hex engine behind the bridge's interface: the hook slot for the forward, the hexad C ABI for the reverse.

  forward   the window's forecast in THIS process with the slot's callback installed
            (woof.hex.ad.external_tendency.install): the bridge's callback writes each step's tendency, and the
            adapter copies it on the card into slice k of a table (nsteps, nz, ncol) per field (no host round trip).
  reverse   a hexad session (woof.hex.ad.torch_reverse.Reverse) over a plan recorded with a table in the slot, whose
            table words are the ones this forward wrote: the session replays the forward with them, and after the
            reverse passes step k, slice k of the table allocation's shadow is step k's slot cotangent (contract (b)),
            read in place.  Cotangents into the state at boundary k go in by Reverse.put (the session's named fields:
            theta, rho, u, w, species) or, for a field the session does not name (pressure, the cell winds, surface
            fields), by an add into its plan allocation's shadow (state_allocs).

Two pieces are the caller's, because they belong to the door and the runner, not the bridge:
  forecast(window) -> {name: array}   runs the window's forecast in this process and returns the named state at its
                                      end (the arrays the loss reads)
  open_session(window, tables) -> (reverse, table_allocs)
                                      a Reverse whose trajectory is bound with the given tables as the slot's
                                      boundary-0 words, and the plan allocation id of each field's table
"""
from __future__ import annotations

import numpy as np

from .engine import SLOT_FIELDS

#: The increment names a hexad session takes in put/init (hexad_phys.h, hexad_field).
SESSION_FIELDS = ("theta", "rho", "u", "w", "qv", "qc", "qr", "qi", "qs", "qg", "nifa", "nwfa")


class HexEngine:
    def __init__(self, forecast, open_session, *, ncol, nz, xp=None, dtype=np.float32, slot_fields=SLOT_FIELDS,
                 neighbours=None, state_allocs=None, install=None, uninstall=None):
        if xp is None:
            import cupy as xp
        if install is None or uninstall is None:
            from woof.hex.ad import external_tendency as et
            install, uninstall = et.install, et.uninstall
        self.forecast, self.open_session = forecast, open_session
        self.install, self.uninstall = install, uninstall
        self.xp, self.dtype = xp, dtype
        self.ncol, self.nz = int(ncol), int(nz)
        self.slot_fields = tuple(slot_fields)
        self._neighbours = neighbours
        self.state_allocs = dict(state_allocs or {})
        self.itemsize = np.dtype(dtype).itemsize
        self.times = {}
        self.rev = None

    def neighbours(self):
        if self._neighbours is None:
            raise ValueError("this HexEngine was given no neighbour table (the mesh's cellsOnCell, zero-based)")
        return self._neighbours

    # ---- forward ----------------------------------------------------------------------------------------------
    def run_forward(self, window, callback, record=True):
        n, xp = int(window.length), self.xp
        tables = {f: xp.zeros((n, self.nz, self.ncol), self.dtype) for f in self.slot_fields}

        def slot(k, state):
            out = callback(k, state) if callback is not None else None
            if not out:
                return None
            for f, a in out.items():
                if a is not None:
                    tables[f][k] = a          # on the card: the table the recording's slot replays
            return out

        self.install(slot, fields=self.slot_fields)
        try:
            self._end = self.forecast(window)
        finally:
            self.uninstall()
        self.n, self.tables, self.rev = n, tables, None
        if record:
            self.rev, self.table_allocs = self.open_session(window, tables)
            self._k = None

    def state(self):
        return self._end

    # ---- reverse ----------------------------------------------------------------------------------------------
    def _split(self, g_add):
        put, shadow = {}, {}
        for name, g in (g_add or {}).items():
            if name in SESSION_FIELDS:
                put[name] = g
            elif name in self.state_allocs:
                shadow[name] = g
            else:
                raise ValueError(f"no way to add a cotangent into state field {name!r}: the session names "
                                 f"{SESSION_FIELDS} and no plan allocation was given for it (state_allocs)")
        return put, shadow

    def _shadow_add(self, shadow):
        for name, g in shadow.items():
            view = self.rev.shadow(self.state_allocs[name], 0, g.size * self.itemsize)
            view += g.reshape(-1).astype(view.dtype, copy=False)

    def reverse_step(self, g_add=None):
        put, shadow = self._split(g_add)
        if self._k is None:
            self.rev.init(put or None)
            self._k = self.n
        elif put:
            self.rev.put(put)
        self._shadow_add(shadow)
        self.rev.step()
        self._k -= 1
        return self._slot(self._k)

    def reverse_end(self, g_add=None):
        put, shadow = self._split(g_add)
        if put:
            self.rev.put(put)
        self._shadow_add(shadow)
        self._k = None
        return None

    def reverse_window(self, g_by_boundary):
        out = [None] * self.n
        for k in range(self.n - 1, -1, -1):
            out[k] = self.reverse_step(g_by_boundary.get(k + 1))
        self.reverse_end(g_by_boundary.get(0))
        return out

    def _slot(self, k):
        nbytes = self.nz * self.ncol * self.itemsize
        return {f: self.rev.shadow(self.table_allocs[f], k * nbytes, nbytes).reshape(self.nz, self.ncol)
                for f in self.slot_fields}
