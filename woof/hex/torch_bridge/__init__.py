"""woof.hex.torch_bridge: a learned correction trained through the engine and its reverse (optional extra `torch`).

A model (a table of numbers, or a small network per column) writes an external tendency into the engine at every
step; the engine's reverse hands back the error's sensitivity to that tendency at every step; PyTorch turns it into
weight updates.  The model starts at zero, so iteration 0 is the engine's own run.  Nothing here is imported by the
engine; PyTorch (BSD) is needed only when this package is.

  engine.py    the interface an engine offers (the CONTRACT in code) and the DLPack helpers
  autograd.py  run_window: forms v0 (window, inputs given), v0i (window, iterated whole reverses, exact in the
               limit), v0x (window, pausing reverse, exact), v1 (per step, exact)
  proofs.py    the toy proofs as one command (python -m woof.hex.torch_bridge.proofs)
  models.py    TableCorrection, ColumnNet
  losses.py    ScaledMSE, MultiScaleLoss, NeighbourSmoother
  trainer.py   Trainer: AdamW, clipping, checkpoints, held-out windows
  toy.py       ToyColumns / ToyCases: a stand-in engine with an exact hand-written reverse, for tests
"""
from .engine import SLOT_FIELDS, WindowSpec

__all__ = ["SLOT_FIELDS", "WindowSpec"]
