"""The seam between the cycling spine and the MPAS GPU port.

The seam's *implementation* moved to :mod:`mpas_cycle_bridge.portbind`
and this module re-exports it.  That move is the whole architecture of
the closed loop, so it is worth stating plainly rather than leaving as a
one-line indirection.

The port pins woof by commit and loads it from a frozen Arwen checkout.
Before doing so it checks ``sys.modules['woof']`` and refuses if woof
is already live from a different tree::

    woof was already imported from a different tree; exact frozen
    Arwen v2 Arwen cannot replace a live package

``woof.cycle`` is woof.  So the previous shape of this module -- a
spine module that loads the port into the spine's own process -- could
never have worked past ``_construct_device_stack``, and it did not: the
closed-loop proof never got past its seed phase.  The fix is a process
boundary, not a pin rotation: rotating the port's woof pin would
re-base every MPAS receipt that cites it and would couple cycle
development to a frozen commit permanently, whereas a process boundary
is precisely the thing the port's own restart proof already validates
(15+15 steps across a fresh process, bitwise identical to 30
continuous).

Therefore everything the forecast process calls lives in
``mpas_cycle_bridge``, which has no woof imports anywhere in it, and
this module is the spine's view of the same code.  The spine may import
freely -- it is the forecast process that is constrained.

See :mod:`woof.cycle.mpas_bridge` for the launcher and
:mod:`mpas_cycle_bridge.worker` for the far side.

What the port actually offers, verified against
``development branch@12fe6a3`` on test host A rather than from notes:

``tools/run_cuda_v841_full_physics_x4.py``
    ``_construct_device_stack(*, host, cache, arwen_checkout, state=None,
    saved_diagnostics=None, backend_restart=None)`` -- the rehydration
    seam.  Its three optional arguments are the entire reason a closed
    loop is possible.

    ``_run_steps(*, stack, start_step, end_step, capture_steps,
    boundary_observer=None)``.

``src/hexcore`` (``src/mpas_port`` before the port's 0.2.0 rename)
    ``state.PrognosticState`` -- level-major ``(nVertLevels, nCells)``.
    ``driver.DrySavedDiagnostics`` -- the seven-field sidecar.
    The package is a row of ``PORT_PACKAGES``; a bound tree's is the first
    row it holds, and every port import goes through it.

There is no ``_download_state``.  :func:`download_stack` is that
function, written on this side rather than in the port: the port is
frozen and SHA-verified at runtime, so the spine adapts to it and never
the other way round.  See that function's own note on why it did not go
upstream.

Disposition, 2026-08-31: the port now also ships standalone as the
``woof hex`` package, which carries no woof pin and no import guard.
This seam stays written against the frozen-checkout shape it was proven
on (``development branch@12fe6a3`` via ``--port-root``); adapting
``mpas_cycle_bridge.portbind`` to load an installed ``woof hex`` is a
named follow-up.  The process boundary, anchor schema and receipt
grading are port-shape-independent and carry over unchanged.
"""

from __future__ import annotations

from mpas_cycle_bridge.portbind import (BACKEND_RESTART_KEYS,
                                        PORT_FORECAST_RELPATH,
                                        PORT_MESH_BINDING_RELPATH,
                                        PORT_PACKAGES, PORT_PROOF_RELPATH,
                                        PORT_SRC_RELDIR, PROGNOSTIC_FIELDS,
                                        REQUIRED_FORECAST_SYMBOLS,
                                        REQUIRED_PROOF_SYMBOLS,
                                        SAVED_DIAGNOSTIC_FIELDS, PortBinding,
                                        PortBindingError, bind_port,
                                        download_stack, join_backend_restart,
                                        port_package, prognostic_state,
                                        saved_diagnostics,
                                        split_backend_restart)

__all__ = [
    "BACKEND_RESTART_KEYS", "PORT_FORECAST_RELPATH",
    "PORT_MESH_BINDING_RELPATH", "PORT_PACKAGES", "PORT_PROOF_RELPATH",
    "PORT_SRC_RELDIR", "PROGNOSTIC_FIELDS", "REQUIRED_FORECAST_SYMBOLS",
    "REQUIRED_PROOF_SYMBOLS", "SAVED_DIAGNOSTIC_FIELDS", "PortBinding",
    "PortBindingError", "bind_port", "download_stack", "join_backend_restart",
    "port_package", "prognostic_state", "saved_diagnostics",
    "split_backend_restart",
]
