"""One define session for many definitions in a classic (CDF-1/2/5) netCDF file.

THE COST THIS REMOVES (WOOF 3 km region shape on the reference system box, 2026-10-27): netCDF4-python ends
define mode after every classic variable and every attribute it writes.  Once a file holds data, each
of those ends grows the header past its padding and libnetcdf moves the whole data section, so a file
that gains N definitions after its first data is rewritten N times on one core.  The region door paid
26 s for this in the parent static's assembly and 11-15 s more carrying the parent's attributes onto
each cut.

The final file is the same bytes: the header lists the same dimensions, attributes and variables in the
same order, and libnetcdf places the data section and every variable from the final header the same way
whether it got there in one step or many (``NC_begins`` only ever moves a begin forward, to the rounded
header size and the running sum of the variable sizes).  Only the number of rewrites changes.
``woof.hex.vertical_spec`` uses the same deferral for the vertical artifact.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any


def deferred_definition(dataset_type: Any) -> Any:
    """A ``dataset_type`` subclass whose ``definition_batch()`` holds define mode open until it exits.

    Inside the batch, create variables and set attributes only; write data after it.  Outside a batch the
    subclass behaves exactly as ``dataset_type``.  A netCDF-4 (HDF5) file has no define mode to hold, so
    the batch changes nothing there.  Open it with ``keepweakref=True`` (as ``woof.hex.vertical_spec`` does): a
    subclass instance in a variable/group reference cycle prints an ignored ``__dealloc__`` error when the cycle
    is collected.
    """

    state = {"deferred": False}

    class DeferredDefinition(dataset_type):
        def _enddef(self) -> None:
            if not state["deferred"]:
                super()._enddef()

        @contextmanager
        def definition_batch(self):
            if state["deferred"]:
                raise RuntimeError("a definition batch is already open")
            state["deferred"] = True
            try:
                yield self
            finally:
                state["deferred"] = False
                if self.data_model != "NETCDF4":
                    super()._enddef()

    return DeferredDefinition
