"""``woof cells``: storm cells as objects, over WOOF-ArWen history.

WOOF-ArWen writes fields; a meteorologist deciding on a storm needs OBJECTS
-- this cell, its track, how old it is, whether it is growing, where it
will be in twenty minutes.  That object layer is titan, a Rust
implementation of the TITAN storm-cell engine (identification,
tracking, lineage, trend, forecast footprints) that WOOF-ArWen does not ship:
the analyze door is optional and off when no titan binary is installed.
This package is the seam between the two:

* :mod:`woof.cells.export` turns a wrfout series into the checksummed
  Cartesian volumes titan reads, on a fixed height ladder;
* :mod:`woof.cells.titan` resolves and runs the ``titan`` binary and
  reads its analysis bundle;
* :mod:`woof.cells.catalog` joins titan's cells to what only the model
  knows -- updraft speed, cloud top and base, freezing and supercooled
  levels, supercooled liquid water -- sampled inside each cell's own
  footprint;
* :mod:`woof.cells.cli` is the door: ``woof cells export``,
  ``woof cells analyze``, ``woof cells catalog``.

Nothing here decides anything about a storm.  It supplies the objects
and the numbers, with their units and their provenance, and the
decision stays with the person reading them.
"""

from __future__ import annotations

__all__: list[str] = []
