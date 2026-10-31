"""``[hex_nests]`` trigger looks on the MPAS-physics route (Hex with Regional, shape 3).

The look itself (where, when, what fires) is :mod:`woof.hex.nest_triggers`, armed
by :func:`woof.hex.nest_triggers.arm_loop` exactly as on the ArWen-physics route.
What this module adds is the route's half: the reflectivity a look reads, plus
the dycore sources every route forms (``w``, ``cape``, ``updraft_helicity_max``:
:mod:`woof.hex.nest_look_dycore`, shared with the WOOF route).

On a look step the Thompson slot (``thompson_v842``, and ``thompson_aero_v842``
through it) forms the 10 cm reflectivity from its post-microphysics columns, the
same arrays and the same ``mpas_thompson_radar`` kernel MPAS's own diagnostics
step uses (mpas_atmphys_driver_microphysics.F:738-792).  On a step that is not a
diagnostics step the kernel writes into this module's scratch, never into the
``diag_physics`` pools, so the running ``refl10cm_1km_max`` and every output
word stay what native MPAS writes; on a diagnostics step the look reads the
pool's own ``refl10cm`` the diagnostics just formed.

THE BREAKAGE THIS PREVENTS (parity run group 3, capability 11, 2026-10-24): a
``--nest-triggers`` run on this route finished rc 0 with no look, no firing and
nothing refused, because only the WOOF driver read the table.
"""

from __future__ import annotations

from typing import Any, Mapping

#: The microphysics schemes whose CUDA slot forms the look's reflectivity on this route.
LOOK_REFLECTIVITY_SCHEMES = ("mp_thompson", "mp_thompson_aerosols")


def route_sources(config: Any, *, ported_framework: bool) -> list[str]:
    """The trigger sources this run forms on the card.

    ``refl10cm`` when its microphysics slot forms it, and always the dycore's own
    (:data:`woof.hex.nest_look_dycore.SOURCES`): the dycore is hex's CUDA dycore on
    every MPAS-physics run.  THE BREAKAGE THE DYCORE ROWS FIX: until 2026-10-29 this
    returned ``refl10cm`` or nothing, so a ``wmax``, ``mucape`` or ``uh25`` test (the
    WOOF Hex form's Updraft and CAPE chips) was refused after its box was rented
    (owner canary run_9948508bd7cfdee3).
    """

    from ..nest_look_dycore import SOURCES as DYCORE_SOURCES

    sources = list(DYCORE_SOURCES)
    if not ported_framework:
        # A native referee library runs MPAS's own Fortran microphysics: no hook for a reflectivity look.
        return sources
    selectors = getattr(config, "selectors", None) or {}
    backends = getattr(config, "backends", None) or {}
    scheme = selectors.get("microphysics")
    active = getattr(config, "active", None)
    if (scheme in LOOK_REFLECTIVITY_SCHEMES and str(backends.get("microphysics")) == "hex-cuda"
            and (active is None or active("microphysics"))):
        return ["refl10cm", *sources]
    return sources


class NestLook:
    """The armed trigger, the reflectivity the step's microphysics formed for it, and the dycore sources."""

    def __init__(self, trigger: Any, cp: Any, *, owned_cells: int, dycore_sources: Any | None = None) -> None:
        self.trigger = trigger
        self.cp = cp
        self.owned_cells = int(owned_cells)
        # ``due``: the step's microphysics forms the look's reflectivity (read by the Thompson slots).
        # ``look_due``: the step ends in a look (reflectivity, dycore sources or both).
        self.due = False
        self.look_due = False
        self.refl: Any | None = None
        self._scratch: dict[str, Any] | None = None
        needed = getattr(trigger, "needed_sources", None)
        self.needed = tuple(needed()) if needed is not None else ("refl10cm",)
        # The microphysics forms reflectivity for a look only when a test reads it.
        self.reads_refl = "refl10cm" in self.needed
        self.dycore_sources = dycore_sources

    def begin(self, step: int) -> None:
        """Before ``step``: whether it ends in a look, and whether its microphysics forms reflectivity for it."""

        self.look_due = bool(self.trigger.due(step))
        self.due = self.look_due and self.reads_refl
        self.refl = None

    def scratch(self, nz: int, columns: int) -> Mapping[str, Any]:
        """The radar kernel's four outputs for a look step that is not a diagnostics step."""

        cp = self.cp
        if self._scratch is None or self._scratch["refl10cm"].shape != (nz, columns):
            self._scratch = {"refl10cm": cp.empty((nz, columns), dtype=cp.float32),
                             "refl10cm_max": cp.empty((columns,), dtype=cp.float32),
                             "refl10cm_1km": cp.empty((columns,), dtype=cp.float32),
                             # A running maximum the kernel reads: scratch only, never output.
                             "refl10cm_1km_max": cp.full((columns,), -35.0, dtype=cp.float32)}
        return self._scratch

    def form(self, fields: Mapping[str, Any], diagnostics: Mapping[str, Any], *, diagnostics_formed: bool,
             launch_radar: Any, device_math_source: str) -> None:
        """Called by the Thompson slot after its microphysics substeps, before the conversion back."""

        if not self.due:
            return
        if diagnostics_formed:
            # MPAS's own diagnostics step formed refl10cm into the pool this step: read it.
            self.refl = diagnostics["refl10cm"]
            return
        nz, columns = fields["qv"].shape
        out = self.scratch(nz, columns)
        launch_radar(fields, out, device_math_source=device_math_source)
        self.refl = out["refl10cm"]

    def after_step(self, step: int, dycore: Any | None = None) -> None:
        """After the committed step: launch the look (on the card) and poll the earlier one.

        ``dycore`` is the run's CUDA dycore driver (its committed ``atmosphere``), read for
        the dycore sources (:mod:`woof.hex.nest_look_dycore`) a test names.
        """

        if self.look_due:
            arrays: dict[str, Any] = {}
            if self.reads_refl:
                if self.refl is None:
                    raise RuntimeError(
                        f"[hex_nests] look due at step {step} but this route's microphysics formed no reflectivity "
                        "(no Thompson slot ran on the card this step); the trigger could never fire")
                if self.refl.shape[-1] != self.owned_cells:
                    raise RuntimeError(f"[hex_nests] look reflectivity covers {self.refl.shape[-1]} cells, "
                                       f"the trigger area was built on {self.owned_cells}")
                arrays["refl10cm"] = self.refl
            if self.dycore_sources:
                if dycore is None:
                    raise RuntimeError(f"[hex_nests] look due at step {step} reads "
                                       f"{list(self.dycore_sources.needed)} but the loop passed no dycore")
                arrays.update(self.dycore_sources.form(dycore))
            self.trigger.look(step, arrays)
        self.refl = None
        self.due = False
        self.look_due = False
        self.trigger.poll()

    def finish(self, out: Any) -> dict[str, Any]:
        import json
        from pathlib import Path

        self.trigger.poll(final=True)
        receipt = self.trigger.receipt()
        (Path(out) / "nest-trigger-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8", newline="\n")
        return receipt


def arm(arguments: Any, config: Any, *, mesh: Any, owned_cells: int, start: str, steps: int, dt: float,
        out: Any, cp: Any, ported_framework: bool, multi_rank: bool,
        qv_index: int | None = None) -> NestLook | None:
    """The armed look for this run, or None when it names no ``[hex_nests]`` table."""

    import os
    from pathlib import Path

    from .. import nest_triggers

    table_path = getattr(arguments, "nest_triggers", None) or os.environ.get(nest_triggers.NEST_TRIGGERS_ENV)
    if not table_path:
        return None
    if multi_rank:
        # Each rank holds a partition; a look reads one card's whole mesh.  Refused by name, never ignored.
        raise nest_triggers.NestTriggerRefusal(
            f"[{nest_triggers.TABLE}] trigger looks run on one card on the MPAS-physics route: this run splits "
            "the mesh across cards, so no rank holds the whole trigger area. Run the triggered parent on one card")
    trigger = nest_triggers.arm_loop(
        table_path, run_dir=Path(out), start_text=str(start), start_step=0, dt_s=float(dt),
        parent_seconds=float(steps) * float(dt), mesh=mesh, owned_cells=int(owned_cells),
        available_sources=route_sources(config, ported_framework=ported_framework), cp=cp)
    from ..nest_look_dycore import for_trigger

    return NestLook(trigger, cp, owned_cells=owned_cells,
                    dycore_sources=for_trigger(trigger, cp, owned_cells=int(owned_cells), qv_index=qv_index,
                                               mesh=mesh))
